"""
Default causal link strategies based on pre-computed memory_links.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..memory_engine import fq_table
from .tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause_simple
from .causal_strategies import CausalContext, CausalLinkStrategy, CausalNeighbor, causal_registry
from .types import RetrievalResult

logger = logging.getLogger(__name__)


@causal_registry.register("memory_links")
class MemoryLinksCausalStrategy(CausalLinkStrategy):
    """
    Causal link strategy based on pre-computed memory_links table.

    This is the default strategy that uses pre-extracted causal relationships
    stored in the memory_links table during the retain phase.

    Link types supported:
    - causes/caused_by: Explicit causal chains (weight=1.0, boost=2.0)
    - enables/prevents: Causal enablement (weight=1.0, boost=1.5)

    Note: Currently only 'caused_by' is created during retain, but the SQL
    queries support all four types for future extensibility.
    """

    def __init__(
        self,
        causal_boost_factor: float = 2.0,
        decay_factor: float = 0.7,
        max_neighbors_per_seed: int = 20,
    ):
        """
        Initialize the memory links causal strategy.

        Args:
            causal_boost_factor: Boost factor for causal links (default: 2.0)
            decay_factor: Decay factor per hop (default: 0.7)
            max_neighbors_per_seed: Maximum causal neighbors per seed node
        """
        self._causal_boost_factor = causal_boost_factor
        self._decay_factor = decay_factor
        self._max_neighbors_per_seed = max_neighbors_per_seed

    @property
    def name(self) -> str:
        return "memory_links"

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
        tags_match: TagsMatch = "any",
        tag_groups: Optional[List[TagGroup]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find causal neighbors from pre-computed memory_links table.

        Queries the memory_links table for causal links (causes, caused_by,
        enables, prevents) originating from the seed nodes.
        """
        if not seed_ids:
            return {}

        ml = fq_table("memory_links")
        mu = fq_table("memory_units")

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

        params: List[Any] = [seed_ids, bank_id, fact_type]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        rows = await conn.fetch(
            f"""
            SELECT
                ml.from_unit_id,
                mu.id AS neighbor_id,
                mu.text AS neighbor_text,
                ml.weight AS link_weight,
                ml.link_type,
                mu.event_date
            FROM {ml} ml
            JOIN {mu} mu ON mu.id = ml.to_unit_id
            WHERE ml.from_unit_id = ANY($1::uuid[])
              AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
              AND mu.bank_id = $2
              AND mu.fact_type = $3
              {tags_clause}
              {groups_clause}
              {created_range_clause}
            ORDER BY ml.from_unit_id, ml.weight DESC
            LIMIT $4
            """,
            *params,
            self._max_neighbors_per_seed,
        )

        neighbors_dict: Dict[str, List[CausalNeighbor]] = {str(sid): [] for sid in seed_ids}

        for row in rows:
            seed_id = str(row["from_unit_id"])
            neighbor = CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["link_weight"]),
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                },
            )
            neighbors_dict[seed_id].append(neighbor)

        logger.debug(f"Found {sum(len(v) for v in neighbors_dict.values())} causal neighbors for {len(seed_ids)} seeds")
        return neighbors_dict


@causal_registry.register("temporal_causal")
class TemporalCausalStrategy(CausalLinkStrategy):
    """
    Causal link strategy based on temporal proximity.

    This strategy treats temporally proximate facts as having causal relationships,
    using a simplified model where time adjacency implies causation.

    Score formula:
        causal_score = parent_score * temporal_proximity * boost_factor * decay

    This is useful when explicit causal links are not available.
    """

    def __init__(
        self,
        time_window_hours: int = 24,
        temporal_boost_factor: float = 1.5,
        decay_factor: float = 0.7,
        min_temporal_proximity: float = 0.3,
        max_neighbors_per_seed: int = 20,
    ):
        """
        Initialize temporal causal strategy.

        Args:
            time_window_hours: Time window for considering causal proximity
            temporal_boost_factor: Boost factor for temporally proximate facts
            decay_factor: Decay factor per hop
            min_temporal_proximity: Minimum temporal proximity threshold
            max_neighbors_per_seed: Maximum neighbors per seed node
        """
        self._time_window_hours = time_window_hours
        self._temporal_boost_factor = temporal_boost_factor
        self._decay_factor = decay_factor
        self._min_temporal_proximity = min_temporal_proximity
        self._max_neighbors_per_seed = max_neighbors_per_seed

    @property
    def name(self) -> str:
        return "temporal_causal"

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        """Return temporal boost factor."""
        return self._temporal_boost_factor

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
        Compute causal score based on temporal proximity.

        If context is available with parent_date and current_date,
        uses temporal proximity as the link_weight.
        """
        effective_weight = link_weight

        if context and context.parent_date and context.current_date:
            temporal_proximity = self._calculate_temporal_proximity(
                context.parent_date,
                context.current_date,
            )
            effective_weight = max(temporal_proximity, link_weight)

        return super().compute_causal_score(parent_score, effective_weight, context, **kwargs)

    def _calculate_temporal_proximity(
        self,
        parent_date: datetime,
        current_date: datetime,
    ) -> float:
        """Calculate temporal proximity based on time difference."""
        time_diff_hours = abs((parent_date - current_date).total_seconds() / 3600)

        if time_diff_hours > self._time_window_hours:
            return 0.0

        proximity = max(
            self._min_temporal_proximity,
            1.0 - (time_diff_hours / self._time_window_hours)
        )
        return proximity

    async def find_causal_neighbors(
        self,
        conn,
        seed_ids: List[str],
        bank_id: str,
        fact_type: str,
        tags: Optional[List[str]] = None,
        tags_match: TagsMatch = "any",
        tag_groups: Optional[List[TagGroup]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, List[CausalNeighbor]]:
        """
        Find temporally proximate neighbors as proxy for causal neighbors.

        This uses the same temporal link query as the temporal retrieval strategy.
        """
        if not seed_ids:
            return {}

        mu = fq_table("memory_units")

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

        params: List[Any] = [seed_ids, bank_id, fact_type, timedelta(hours=self._time_window_hours)]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        rows = await conn.fetch(
            f"""
            SELECT
                seed.from_unit_id,
                mu.id AS neighbor_id,
                mu.text AS neighbor_text,
                mu.event_date,
                seed.link_weight AS temporal_proximity,
                'temporal' AS link_type
            FROM (
                SELECT
                    ml.from_unit_id,
                    ml.to_unit_id,
                    ml.weight AS link_weight
                FROM {fq_table("memory_links")} ml
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type = 'temporal'
                UNION ALL
                SELECT
                    ml.to_unit_id AS from_unit_id,
                    ml.from_unit_id AS to_unit_id,
                    ml.weight AS link_weight
                FROM {fq_table("memory_links")} ml
                WHERE ml.to_unit_id = ANY($1::uuid[])
                  AND ml.link_type = 'temporal'
            ) seed
            JOIN {mu} mu ON mu.id = seed.to_unit_id
            WHERE mu.bank_id = $2
              AND mu.fact_type = $3
              {tags_clause}
              {groups_clause}
              {created_range_clause}
            ORDER BY seed.from_unit_id, seed.link_weight DESC
            LIMIT $6
            """,
            *params,
            self._max_neighbors_per_seed,
        )

        neighbors_dict: Dict[str, List[CausalNeighbor]] = {str(sid): [] for sid in seed_ids}

        for row in rows:
            seed_id = str(row["from_unit_id"])
            neighbor = CausalNeighbor(
                neighbor_id=str(row["neighbor_id"]),
                link_weight=float(row["temporal_proximity"]),
                link_type=row["link_type"],
                provenance={
                    "text": row.get("neighbor_text"),
                    "event_date": row.get("event_date"),
                },
            )
            neighbors_dict[seed_id].append(neighbor)

        logger.debug(f"Found {sum(len(v) for v in neighbors_dict.values())} temporal neighbors for {len(seed_ids)} seeds")
        return neighbors_dict
