"""
LLM-based causal link strategy for dynamic causal reasoning.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..memory_engine import fq_table
from .causal_implementations import MemoryLinksCausalStrategy
from .causal_strategies import CausalContext, CausalLinkStrategy, CausalNeighbor, causal_registry

if TYPE_CHECKING:
    from ..llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


@causal_registry.register("llm_dynamic")
class LLMDynamicCausalStrategy(CausalLinkStrategy):
    """
    LLM-based dynamic causal reasoning strategy.

    This strategy uses an LLM to dynamically determine causal relationships
    at query time, rather than relying on pre-computed links.

    This is more flexible but computationally expensive.
    """

    def __init__(
        self,
        llm: Optional["LLMWrapper"] = None,
        causal_boost_factor: float = 2.0,
        decay_factor: float = 0.7,
        max_neighbors_per_seed: int = 10,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize LLM-based causal strategy.

        Args:
            llm: LLM wrapper instance for causal reasoning
            causal_boost_factor: Boost factor for causal links
            decay_factor: Decay factor per hop
            max_neighbors_per_seed: Maximum causal neighbors per seed
            similarity_threshold: Minimum similarity to consider for causal analysis
        """
        self._llm = llm
        self._causal_boost_factor = causal_boost_factor
        self._decay_factor = decay_factor
        self._max_neighbors_per_seed = max_neighbors_per_seed
        self._similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "llm_dynamic"

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        """Return configured causal boost factor."""
        return self._causal_boost_factor

    def _get_decay_factor(self, context: Optional[CausalContext]) -> float:
        """Return configured decay factor."""
        if context is None:
            return self._decay_factor
        return self._decay_factor if context.distance > 0 else 1.0

    async def find_causal_neighbors(
        self,
        conn,
        seed_ids: List[str],
        bank_id: str,
        fact_type: str,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
        tag_groups: Optional[List[Any]] = None,
        created_after: Optional[Any] = None,
        created_before: Optional[Any] = None,
        query_context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find causal neighbors using LLM-based reasoning.

        This method:
        1. Fetches candidate neighbors (semantically similar facts)
        2. Uses LLM to determine which are causally related
        3. Returns only those deemed causally related

        Args:
            conn: Database connection
            seed_ids: List of seed node IDs
            bank_id: Memory bank ID
            fact_type: Fact type to filter
            query_context: Optional query context for LLM reasoning

        Returns:
            Dict mapping seed_id -> list of causally related neighbors
        """
        if not seed_ids:
            return {}

        if self._llm is None:
            logger.warning("LLM not configured, falling back to memory_links strategy")
            from .causal_implementations import MemoryLinksCausalStrategy
            fallback = MemoryLinksCausalStrategy()
            return await fallback.find_causal_neighbors(
                conn, seed_ids, bank_id, fact_type,
                tags=tags, tags_match=tags_match, tag_groups=tag_groups,
                created_after=created_after, created_before=created_before, **kwargs
            )

        candidates = await self._find_candidate_neighbors(conn, seed_ids, bank_id, fact_type)

        causal_neighbors = {}
        for seed_id, seed_candidates in candidates.items():
            causal = await self._llm_determine_causal(
                seed_id=seed_id,
                candidates=seed_candidates,
                query_context=query_context,
            )
            causal_neighbors[seed_id] = causal

        return causal_neighbors

    async def _find_candidate_neighbors(
        self,
        conn,
        seed_ids: List[str],
        bank_id: str,
        fact_type: str,
    ) -> Dict[str, List[CausalNeighbor]]:
        """Find candidate neighbors using semantic similarity."""
        mu = fq_table("memory_units")

        params = [seed_ids, bank_id, fact_type, self._similarity_threshold]

        rows = await conn.fetch(
            f"""
            SELECT
                ml.from_unit_id,
                mu.id AS neighbor_id,
                mu.text AS neighbor_text,
                mu.event_date,
                ml.weight AS similarity_score,
                'semantic' AS link_type
            FROM {fq_table("memory_links")} ml
            JOIN {mu} mu ON mu.id = ml.to_unit_id
            WHERE ml.from_unit_id = ANY($1::uuid[])
              AND ml.link_type = 'semantic'
              AND ml.weight >= $4
              AND mu.bank_id = $2
              AND mu.fact_type = $3
            ORDER BY ml.from_unit_id, ml.weight DESC
            LIMIT $5
            """,
            *params,
            self._max_neighbors_per_seed,
        )

        candidates: Dict[str, List[CausalNeighbor]] = {str(sid): [] for sid in seed_ids}

        for row in rows:
            seed_id = str(row["from_unit_id"])
            neighbor = CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["similarity_score"]),
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                },
            )
            candidates[seed_id].append(neighbor)

        return candidates

    async def _llm_determine_causal(
        self,
        seed_id: str,
        candidates: List[CausalNeighbor],
        query_context: Optional[str] = None,
    ) -> List[CausalNeighbor]:
        """
        Use LLM to determine which candidates are causally related.

        This is a placeholder implementation. In production, this would:
        1. Format the seed fact and candidate facts into a prompt
        2. Call the LLM to identify causal relationships
        3. Parse the response and return causal neighbors with scores

        Args:
            seed_id: ID of the seed fact
            candidates: List of candidate neighbor facts
            query_context: Optional query context

        Returns:
            List of neighbors deemed causally related
        """
        if not candidates:
            return []

        if self._llm is None:
            return candidates[:self._max_neighbors_per_seed]

        seed_text = candidates[0].provenance.get("text", "") if candidates else ""

        prompt = f"""Given the following seed fact:
{seed_text}

And candidate facts:
{chr(10).join([f"- {c.neighbor_id}: {c.provenance.get('text', '')}" for c in candidates])}

Which of these candidate facts are CAUSED BY the seed fact? Respond with a JSON array of the causally related fact IDs with a confidence score between 0 and 1."""

        try:
            response = await self._llm.generate(prompt)
            causal_ids = self._parse_llm_response(response)

            causal_neighbors = [
                c for c in candidates
                if c.neighbor_id in causal_ids
            ]
            return causal_neighbors[:self._max_neighbors_per_seed]

        except Exception as e:
            logger.error(f"LLM causal reasoning failed: {e}")
            return candidates[:self._max_neighbors_per_seed]

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract causal IDs."""
        import json
        import re

        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [item.get("id") or item if isinstance(item, (dict, str)) else None for item in data]
        except (json.JSONDecodeError, AttributeError):
            pass

        return []


@causal_registry.register("hybrid_causal")
class HybridCausalStrategy(CausalLinkStrategy):
    """
    Hybrid causal strategy combining pre-computed and dynamic approaches.

    This strategy:
    1. First uses pre-computed causal links from memory_links
    2. Falls back to semantic neighbors if no causal links found
    3. Optionally validates with LLM for critical queries
    """

    def __init__(
        self,
        causal_boost_factor: float = 2.0,
        decay_factor: float = 0.7,
        max_neighbors_per_seed: int = 20,
        semantic_fallback: bool = True,
        semantic_weight: float = 0.5,
    ):
        """
        Initialize hybrid causal strategy.

        Args:
            causal_boost_factor: Boost factor for causal links
            decay_factor: Decay factor per hop
            max_neighbors_per_seed: Maximum neighbors per seed
            semantic_fallback: Whether to include semantic neighbors as fallback
            semantic_weight: Weight to give semantic neighbors when no causal links
        """
        self._memory_links = MemoryLinksCausalStrategy(
            causal_boost_factor=causal_boost_factor,
            decay_factor=decay_factor,
            max_neighbors_per_seed=max_neighbors_per_seed,
        )
        self._semantic_fallback = semantic_fallback
        self._semantic_weight = semantic_weight

    @property
    def name(self) -> str:
        return "hybrid_causal"

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        """Return configured causal boost factor."""
        return self._memory_links._causal_boost_factor

    def _get_decay_factor(self, context: Optional[CausalContext]) -> float:
        """Return configured decay factor."""
        return self._memory_links._decay_factor

    async def find_causal_neighbors(
        self,
        conn,
        seed_ids: List[str],
        bank_id: str,
        fact_type: str,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
        tag_groups: Optional[List[Any]] = None,
        created_after: Optional[Any] = None,
        created_before: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find causal neighbors with fallback to semantic neighbors.

        1. Query pre-computed causal links
        2. If no causal links found and semantic_fallback=True, add semantic neighbors
        """
        causal_neighbors = await self._memory_links.find_causal_neighbors(
            conn, seed_ids, bank_id, fact_type,
            tags=tags, tags_match=tags_match, tag_groups=tag_groups,
            created_after=created_after, created_before=created_before, **kwargs
        )

        if not self._semantic_fallback:
            return causal_neighbors

        for seed_id in seed_ids:
            seed_str = str(seed_id)
            if not causal_neighbors.get(seed_str):
                semantic_neighbors = await self._find_semantic_neighbors(
                    conn, seed_id, bank_id, fact_type
                )
                if semantic_neighbors:
                    for neighbor in semantic_neighbors:
                        neighbor.link_weight *= self._semantic_weight
                    causal_neighbors[seed_str] = semantic_neighbors

        return causal_neighbors

    async def _find_semantic_neighbors(
        self,
        conn,
        seed_id: str,
        bank_id: str,
        fact_type: str,
    ) -> List[CausalNeighbor]:
        """Find semantic neighbors as fallback."""
        mu = fq_table("memory_units")

        rows = await conn.fetch(
            f"""
            SELECT
                ml.from_unit_id,
                mu.id AS neighbor_id,
                mu.text AS neighbor_text,
                mu.event_date,
                ml.weight AS similarity_score,
                'semantic' AS link_type
            FROM {fq_table("memory_links")} ml
            JOIN {mu} mu ON mu.id = ml.to_unit_id
            WHERE ml.from_unit_id = $1::uuid
              AND ml.link_type = 'semantic'
              AND mu.bank_id = $2
              AND mu.fact_type = $3
            ORDER BY ml.weight DESC
            LIMIT $4
            """,
            seed_id, bank_id, fact_type, self._memory_links._max_neighbors_per_seed
        )

        return [
            CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["similarity_score"]),
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                },
            )
            for row in rows
        ]
