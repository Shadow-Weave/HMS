"""
Session-based causal link strategy.

Causal relationships are determined based on:
1. Nodes belonging to the same document/session
2. Semantic similarity above a threshold (default: 0.3)

This strategy assumes that facts from the same document are likely causally related.
"""

import logging
from typing import Any, Dict, List, Optional

from ..memory_engine import fq_table
from .causal_strategies import CausalContext, CausalLinkStrategy, CausalNeighbor, CausalScore, causal_registry
from .tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause_simple

logger = logging.getLogger(__name__)


@causal_registry.register("session_causal")
class SessionCausalStrategy(CausalLinkStrategy):
    """
    Causal link strategy based on document/session co-occurrence.

    This strategy determines causal relationships using two criteria:
    1. Two nodes belong to the same document/session
    2. Their semantic similarity exceeds a threshold

    The semantic similarity is retrieved from the memory_links table where
    link_type = 'semantic'.

    This is useful when:
    - Documents have implicit causal structure (e.g., articles, stories)
    - Session context provides causal grounding
    - No explicit causal links are available
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        causal_boost_factor: float = 2.0,
        decay_factor: float = 0.7,
        max_neighbors_per_seed: int = 20,
        bidirectional: bool = True,
        fixed_score: Optional[float] = 1.0,
    ):
        """
        Initialize the session causal strategy.

        Args:
            similarity_threshold: Minimum semantic similarity to consider
                                two nodes causally related (default: 0.3)
            causal_boost_factor: Boost factor for causal links (default: 2.0)
            decay_factor: Decay factor per hop (default: 0.7)
            max_neighbors_per_seed: Maximum causal neighbors per seed node
            bidirectional: If True, find neighbors in both directions (A->B and B->A)
            fixed_score: If set, use this fixed score for all causal links instead of
                        computing from similarity. Set to None to use similarity-based scoring.
                        (default: 1.0 for fixed scores)
        """
        self._similarity_threshold = similarity_threshold
        self._causal_boost_factor = causal_boost_factor
        self._decay_factor = decay_factor
        self._max_neighbors_per_seed = max_neighbors_per_seed
        self._bidirectional = bidirectional
        self._fixed_score = fixed_score

    @property
    def name(self) -> str:
        return "session_causal"

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        """Return configured causal boost factor."""
        return self._causal_boost_factor

    def _get_decay_factor(self, context: Optional[CausalContext]) -> float:
        """Return configured decay factor."""
        if context is None:
            return self._decay_factor
        return self._decay_factor if context.distance > 0 else 1.0

    def compute_causal_score(
        self,
        parent_score: float,
        link_weight: float,
        context: Optional[CausalContext] = None,
        **kwargs,
    ):
        """
        Compute causal score with optional fixed scoring.

        If _fixed_score is set, use that value instead of computing from
        similarity and parent score. This ensures deterministic scores.

        Args:
            parent_score: Score from the parent node
            link_weight: Semantic similarity (used if fixed_score is None)
            context: Optional context for score computation

        Returns:
            CausalScore with either fixed or computed score
        """
        if self._fixed_score is not None:
            boosted_score = parent_score * self._fixed_score * self._get_causal_boost_factor(link_weight) * self._get_decay_factor(context)
            return CausalScore(
                base_score=self._fixed_score,
                components={
                    "parent_score": parent_score,
                    "fixed_score": self._fixed_score,
                    "causal_boost_factor": self._get_causal_boost_factor(link_weight),
                    "decay_factor": self._get_decay_factor(context),
                },
                boosted_score=boosted_score,
            )
        else:
            return super().compute_causal_score(parent_score, link_weight, context, **kwargs)

    async def find_causal_neighbors(
        self,
        conn,
        seed_ids: List[str],
        bank_id: str,
        fact_type: str,
        tags: Optional[List[str]] = None,
        tags_match: TagsMatch = "any",
        tag_groups: Optional[List[TagGroup]] = None,
        created_after: Optional[Any] = None,
        created_before: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find causally related neighbors based on session co-occurrence.

        This method:
        1. Gets the document_ids of the seed nodes
        2. Finds all other nodes in the same documents
        3. Filters by semantic similarity threshold
        4. Returns neighbors with session-based causal links

        Args:
            conn: Database connection
            seed_ids: List of seed node IDs
            bank_id: Memory bank ID
            fact_type: Fact type to filter

        Returns:
            Dict mapping seed_id -> list of CausalNeighbor objects
        """
        if not seed_ids:
            return {}

        mu = fq_table("memory_units")
        ml = fq_table("memory_links")

        tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)
        tag_groups_param_start = 6 + (1 if tags else 0)
        groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

        _next_idx = tag_groups_param_start + len(groups_params)
        created_range_clause = ""
        created_range_params: List[Any] = []
        if created_after is not None:
            created_range_params.append(created_after)
            created_range_clause += f" AND mu.updated_at > ${_next_idx}"
            _next_idx += 1
        if created_before is not None:
            created_range_params.append(created_before)
            created_range_clause += f" AND mu.updated_at < ${_next_idx}"
            _next_idx += 1

        params: List[Any] = [seed_ids, bank_id, fact_type, self._similarity_threshold]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        if self._bidirectional:
            query = f"""
            WITH seed_docs AS (
                SELECT DISTINCT document_id
                FROM {mu}
                WHERE id = ANY($1::uuid[])
                  AND document_id IS NOT NULL
            ),
            seed_doc_neighbors AS (
                SELECT DISTINCT
                    ml.from_unit_id AS seed_id,
                    ml.to_unit_id AS neighbor_id,
                    mu_neighbor.text AS neighbor_text,
                    mu_neighbor.event_date,
                    ml.weight AS similarity,
                    'session_bidirectional' AS link_type
                FROM {ml} ml
                JOIN {mu} mu_seed ON mu_seed.id = ml.from_unit_id
                JOIN seed_docs sd ON mu_seed.document_id = sd.document_id
                JOIN {mu} mu_neighbor ON mu_neighbor.id = ml.to_unit_id
                WHERE ml.link_type = 'semantic'
                  AND ml.weight >= $4
                  AND ml.from_unit_id = ANY($1::uuid[])
                  AND mu_neighbor.bank_id = $2
                  AND mu_neighbor.fact_type = $3
                  {tags_clause}
                  {groups_clause}
                  {created_range_clause}

                UNION

                SELECT DISTINCT
                    ml.to_unit_id AS seed_id,
                    ml.from_unit_id AS neighbor_id,
                    mu_seed.text AS neighbor_text,
                    mu_seed.event_date,
                    ml.weight AS similarity,
                    'session_bidirectional' AS link_type
                FROM {ml} ml
                JOIN {mu} mu_neighbor ON mu_neighbor.id = ml.to_unit_id
                JOIN seed_docs sd ON mu_neighbor.document_id = sd.document_id
                JOIN {mu} mu_seed ON mu_seed.id = ml.from_unit_id
                WHERE ml.link_type = 'semantic'
                  AND ml.weight >= $4
                  AND ml.to_unit_id = ANY($1::uuid[])
                  AND mu_seed.bank_id = $2
                  AND mu_seed.fact_type = $3
                  {tags_clause}
                  {groups_clause}
                  {created_range_clause}
            )
            SELECT
                seed_id,
                neighbor_id,
                neighbor_text,
                event_date,
                MAX(similarity) AS similarity,
                link_type
            FROM seed_doc_neighbors
            GROUP BY seed_id, neighbor_id, neighbor_text, event_date, link_type
            ORDER BY seed_id, similarity DESC
            LIMIT $5
            """
        else:
            query = f"""
            WITH seed_docs AS (
                SELECT id AS seed_id, document_id
                FROM {mu}
                WHERE id = ANY($1::uuid[])
                  AND document_id IS NOT NULL
            )
            SELECT
                sd.seed_id,
                mu_neighbor.id AS neighbor_id,
                mu_neighbor.text AS neighbor_text,
                mu_neighbor.event_date,
                ml.weight AS similarity,
                'session_unidirectional' AS link_type
            FROM seed_docs sd
            JOIN {mu} mu_same_doc ON mu_same_doc.document_id = sd.document_id
              AND mu_same_doc.id != sd.seed_id
            JOIN {ml} ml ON ml.from_unit_id = sd.seed_id
              AND ml.to_unit_id = mu_same_doc.id
              AND ml.link_type = 'semantic'
              AND ml.weight >= $4
            JOIN {mu} mu_neighbor ON mu_neighbor.id = ml.to_unit_id
            WHERE mu_neighbor.bank_id = $2
              AND mu_neighbor.fact_type = $3
              {tags_clause}
              {groups_clause}
              {created_range_clause}
            ORDER BY sd.seed_id, ml.weight DESC
            LIMIT $5
            """

        rows = await conn.fetch(query, *params, self._max_neighbors_per_seed)

        neighbors_dict: Dict[str, List[CausalNeighbor]] = {str(sid): [] for sid in seed_ids}

        for row in rows:
            seed_id = str(row["seed_id"])
            neighbor = CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["similarity"]),
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                    "similarity": float(row["similarity"]),
                    "source": "session_causal",
                },
            )
            neighbors_dict[seed_id].append(neighbor)

        total_neighbors = sum(len(v) for v in neighbors_dict.values())
        logger.debug(
            f"Session causal: Found {total_neighbors} neighbors for {len(seed_ids)} seeds "
            f"(similarity >= {self._similarity_threshold})"
        )

        return neighbors_dict


@causal_registry.register("document_graph")
class DocumentGraphCausalStrategy(CausalLinkStrategy):
    """
    Causal link strategy using document structure as causal graph.

    This strategy treats the document as a causal graph where:
    - Earlier chunks in a document cause later chunks
    - Semantic similarity provides edge weights

    Useful for:
    - Sequential documents (articles, reports, stories)
    - Structured documents with clear progression
    - Chain-of-thought or reasoning chains
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        causal_boost_factor: float = 2.0,
        decay_factor: float = 0.7,
        max_neighbors_per_seed: int = 10,
        chunk_order_weight: float = 0.5,
    ):
        """
        Initialize the document graph causal strategy.

        Args:
            similarity_threshold: Minimum semantic similarity
            causal_boost_factor: Boost factor for causal links
            decay_factor: Decay factor per hop
            max_neighbors_per_seed: Maximum neighbors per seed
            chunk_order_weight: Weight for chunk ordering in score (0-1)
        """
        self._similarity_threshold = similarity_threshold
        self._causal_boost_factor = causal_boost_factor
        self._decay_factor = decay_factor
        self._max_neighbors_per_seed = max_neighbors_per_seed
        self._chunk_order_weight = chunk_order_weight

    @property
    def name(self) -> str:
        return "document_graph"

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        return self._causal_boost_factor

    def _get_decay_factor(self, context: Optional[CausalContext]) -> float:
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
        tags_match: TagsMatch = "any",
        tag_groups: Optional[List[TagGroup]] = None,
        created_after: Optional[Any] = None,
        created_before: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find causal neighbors based on document chunk ordering.

        Only considers chunks that appear AFTER the seed chunk in the document,
        assuming forward causation (earlier causes later).
        """
        if not seed_ids:
            return {}

        mu = fq_table("memory_units")
        ml = fq_table("memory_links")

        tags_clause = build_tags_where_clause_simple(tags, 5, match=tags_match)
        tag_groups_param_start = 5 + (1 if tags else 0)
        groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

        _next_idx = tag_groups_param_start + len(groups_params)
        created_range_clause = ""
        created_range_params: List[Any] = []
        if created_after is not None:
            created_range_params.append(created_after)
            created_range_clause += f" AND mu.updated_at > ${_next_idx}"
            _next_idx += 1
        if created_before is not None:
            created_range_params.append(created_before)
            created_range_clause += f" AND mu.updated_at < ${_next_idx}"
            _next_idx += 1

        params: List[Any] = [seed_ids, bank_id, fact_type, self._similarity_threshold]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        rows = await conn.fetch(
            f"""
            SELECT
                seed.id AS seed_id,
                neighbor.id AS neighbor_id,
                neighbor.text AS neighbor_text,
                neighbor.event_date,
                ml.weight AS similarity,
                CASE
                    WHEN neighbor.chunk_id IS NOT NULL AND seed.chunk_id IS NOT NULL THEN
                        CASE
                            WHEN neighbor.created_at >= seed.created_at THEN 1
                            ELSE -1
                        END
                    ELSE 1
                END AS chunk_distance,
                'document_forward' AS link_type
            FROM {mu} seed
            JOIN {ml} ml ON ml.from_unit_id = seed.id
              AND ml.link_type = 'semantic'
              AND ml.weight >= $4
            JOIN {mu} neighbor ON neighbor.id = ml.to_unit_id
            WHERE seed.id = ANY($1::uuid[])
              AND seed.bank_id = $2
              AND seed.fact_type = $3
              AND neighbor.bank_id = $2
              AND neighbor.fact_type = $3
              AND seed.document_id = neighbor.document_id
              AND neighbor.created_at > seed.created_at
              {tags_clause}
              {groups_clause}
              {created_range_clause}
            ORDER BY seed.id, chunk_distance ASC, ml.weight DESC
            LIMIT $6
            """,
            *params,
            self._max_neighbors_per_seed,
        )

        neighbors_dict: Dict[str, List[CausalNeighbor]] = {str(sid): [] for sid in seed_ids}

        for row in rows:
            seed_id = str(row["seed_id"])
            chunk_distance = row["chunk_distance"] or 1
            position_boost = max(0.1, 1.0 - (chunk_distance * self._chunk_order_weight * 0.1))

            neighbor = CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["similarity"]) * position_boost,
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                    "similarity": float(row["similarity"]),
                    "chunk_distance": int(chunk_distance),
                    "position_boost": position_boost,
                    "source": "document_graph",
                },
            )
            neighbors_dict[seed_id].append(neighbor)

        total_neighbors = sum(len(v) for v in neighbors_dict.values())
        logger.debug(
            f"Document graph: Found {total_neighbors} forward neighbors for {len(seed_ids)} seeds"
        )

        return neighbors_dict
