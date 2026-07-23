"""
Default implementations of search strategies.

These wrap existing retrieval logic into the modular strategy interface,
maintaining backward compatibility while enabling easy swapping for ablation experiments.
"""

import asyncio
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...config import (
    DEFAULT_GRAPH_TEMPORAL_MODE,
    DEFAULT_GRAPH_TEMPORAL_SIGMA_HOURS,
    _get_raw_config,
    get_config,
)
from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table, get_current_schema
from ..sql import create_sql_dialect
from .fusion import reciprocal_rank_fusion
from .graph_retrieval import GraphRetriever
from .link_expansion_retrieval import LinkExpansionRetriever
from .strategies import (
    FusionStrategy,
    GraphRetrievalStrategy,
    RerankingStrategy,
    RetrievalStrategy,
    fusion_registry,
    graph_retrieval_registry,
    reranking_registry,
    retrieval_registry,
)
from .tags import (
    TagGroup,
    TagsMatch,
    build_tag_groups_where_clause,
    build_tags_where_clause_simple,
    filter_results_by_tag_groups,
    filter_results_by_tags,
)
from .types import (
    GraphRetrievalTimings,
    MergedCandidate,
    RetrievalResult,
    ScoredResult,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


def _resolve_temporal_graph_config(config: Any | None) -> tuple[str, float]:
    source = config if config is not None else _get_raw_config()
    return (
        getattr(source, "graph_temporal_mode", DEFAULT_GRAPH_TEMPORAL_MODE),
        getattr(source, "graph_temporal_sigma_hours", DEFAULT_GRAPH_TEMPORAL_SIGMA_HOURS),
    )


def tokenize_query(query_text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", query_text.lower()).split()


@retrieval_registry.register("semantic_bm25")
class SemanticBM25Retrieval(RetrievalStrategy):
    """Semantic + BM25 combined retrieval strategy with LLM-driven query rewriting."""

    def __init__(
        self,
        query_rewriting_strategy_name: str = "noop",
        query_rewriting_kwargs: Dict[str, Any] | None = None,
        alias_expansion_enabled: bool = True,
    ):
        """
        Initialize semantic + BM25 retrieval with optional alias expansion.

        Args:
            query_rewriting_strategy_name: Name of query rewriting strategy to use
            query_rewriting_kwargs: Additional kwargs for query rewriting strategy
            alias_expansion_enabled: Whether to enable LLM-based alias expansion
        """
        self._query_rewriting_strategy_name = query_rewriting_strategy_name
        self._query_rewriting_kwargs = query_rewriting_kwargs or {}
        self._alias_expansion_enabled = alias_expansion_enabled
        self._query_rewriting_strategy = None
        self._llm = None

    def set_llm(self, llm: Any) -> None:
        """Set LLM provider for query rewriting."""
        self._llm = llm

    def _get_query_rewriting_strategy(self) -> Any:
        """Get or create the query rewriting strategy instance."""
        if self._query_rewriting_strategy is None:
            from .query_rewriting import query_rewriting_registry

            self._query_rewriting_strategy = query_rewriting_registry.create(
                self._query_rewriting_strategy_name, **self._query_rewriting_kwargs
            )
        return self._query_rewriting_strategy

    @property
    def name(self) -> str:
        return "semantic_bm25"

    async def _hydrate_vector_hits(
        self,
        conn,
        vector_index,
        hits_by_fact_type,
        *,
        bank_id: str,
        fact_types: List[str],
        limit: int,
        tags: List[str] | None,
        tags_match: TagsMatch,
        tag_groups: List[TagGroup] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> Dict[str, List[RetrievalResult]]:
        """Hydrate external vector hits from canonical SQL rows in hit order."""

        hit_ids = list(
            dict.fromkeys(hit.id for fact_type in fact_types for hit in hits_by_fact_type.get(fact_type, []))
        )
        hydrated: Dict[str, List[RetrievalResult]] = {fact_type: [] for fact_type in fact_types}
        if not hit_ids:
            return hydrated

        placeholders = ", ".join(f"${index}" for index in range(1, len(hit_ids) + 1))
        rows = await conn.fetch(
            f"""
            SELECT id, bank_id, text, context, event_date, occurred_start, occurred_end, mentioned_at,
                   fact_type, document_id, chunk_id, tags, metadata, proof_count, updated_at
            FROM {fq_table("memory_units")}
            WHERE id IN ({placeholders})
            """,
            *hit_ids,
        )

        rows_by_id = {
            str(row["id"]): row
            for row in rows
            if str(row["bank_id"]) == bank_id and str(row["fact_type"]) in fact_types
        }
        stale_ids = [hit_id for hit_id in hit_ids if hit_id not in rows_by_id]
        if stale_ids:
            try:
                await vector_index.delete_units(get_current_schema(), stale_ids)
            except Exception:
                logger.warning("Failed to prune %d stale external vector hits", len(stale_ids), exc_info=True)

        for fact_type in fact_types:
            for hit in hits_by_fact_type.get(fact_type, []):
                if hit.similarity < 0.3:
                    continue
                row = rows_by_id.get(hit.id)
                if row is None or str(row["fact_type"]) != fact_type:
                    continue

                updated_at = row.get("updated_at")
                if created_after is not None and updated_at is not None and updated_at <= created_after:
                    continue
                if created_before is not None and updated_at is not None and updated_at >= created_before:
                    continue

                row_dict = dict(row)
                row_dict["similarity"] = hit.similarity
                row_dict["bm25_score"] = None
                result = RetrievalResult.from_db_row(row_dict)
                if not filter_results_by_tags([result], tags, match=tags_match):
                    continue
                if not filter_results_by_tag_groups([result], tag_groups):
                    continue
                hydrated[fact_type].append(result)
                if len(hydrated[fact_type]) >= limit:
                    break

        return hydrated

    async def _retrieve_database_semantic_only(
        self,
        conn,
        *,
        query_embedding_str: str,
        bank_id: str,
        fact_types: List[str],
        limit: int,
        tags: List[str] | None,
        tags_match: TagsMatch,
        tag_groups: List[TagGroup] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> Dict[str, List[RetrievalResult]]:
        """Run the canonical SQL semantic path for selected fact types."""

        result_dict: Dict[str, List[RetrievalResult]] = {fact_type: [] for fact_type in fact_types}
        if not fact_types:
            return result_dict

        dialect = create_sql_dialect(getattr(conn, "backend_type", "postgresql"))
        cols = (
            "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
            "fact_type, document_id, chunk_id, tags, metadata, proof_count"
        )
        tags_clause = build_tags_where_clause_simple(tags, 3, match=tags_match)
        group_start = 3 + (1 if tags else 0)
        groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, group_start)
        next_param = group_start + len(groups_params)
        created_clause = ""
        created_params: List[Any] = []
        if created_after is not None:
            created_params.append(created_after)
            created_clause += f" AND updated_at > ${next_param}"
            next_param += 1
        if created_before is not None:
            created_params.append(created_before)
            created_clause += f" AND updated_at < ${next_param}"

        fetch_limit = max(limit * 5, 100)
        arms = [
            dialect.build_semantic_arm(
                table=fq_table("memory_units"),
                cols=cols,
                fact_type=fact_type,
                embedding_param="$1",
                bank_id_param="$2",
                fetch_limit=fetch_limit,
                tags_clause=tags_clause,
                groups_clause=groups_clause,
                extra_where=created_clause,
            )
            for fact_type in fact_types
        ]
        params: List[Any] = [query_embedding_str, bank_id]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_params)
        rows = await conn.fetch("\nUNION ALL\n".join(arms), *params)
        for raw_row in rows:
            row = dict(raw_row)
            row.pop("source", None)
            fact_type = row.get("fact_type")
            if fact_type in result_dict and len(result_dict[fact_type]) < limit:
                result_dict[fact_type].append(RetrievalResult.from_db_row(row))
        return result_dict

    async def _retrieve_with_external_vector_index(
        self,
        conn,
        vector_index,
        query_vector: list[float],
        *,
        query_embedding_str: str,
        rewritten_query: str,
        tokens: list[str],
        bank_id: str,
        fact_types: List[str],
        limit: int,
        tags: List[str] | None,
        tags_match: TagsMatch,
        tag_groups: List[TagGroup] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> Dict[str, List[RetrievalResult]]:
        """Run Milvus dense retrieval while retaining database BM25/FTS."""

        result_dict: Dict[str, List[RetrievalResult]] = {fact_type: [] for fact_type in fact_types}
        fetch_limit = max(limit * 5, 100)
        vector_task = asyncio.create_task(
            vector_index.search(
                query_vector,
                namespace=get_current_schema(),
                bank_id=bank_id,
                fact_types=fact_types,
                limit=fetch_limit,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
            )
        )
        try:
            if tokens:
                dialect = create_sql_dialect(getattr(conn, "backend_type", "postgresql"))
                config = get_config()
                text_extension = config.text_search_extension
                bm25_text = dialect.prepare_bm25_text(tokens, rewritten_query, text_search_extension=text_extension)
                tags_clause = build_tags_where_clause_simple(tags, 4, match=tags_match)
                group_start = 4 + (1 if tags else 0)
                groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, group_start)
                next_param = group_start + len(groups_params)
                created_clause = ""
                created_params: List[Any] = []
                if created_after is not None:
                    created_params.append(created_after)
                    created_clause += f" AND updated_at > ${next_param}"
                    next_param += 1
                if created_before is not None:
                    created_params.append(created_before)
                    created_clause += f" AND updated_at < ${next_param}"

                cols = (
                    "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
                    "fact_type, document_id, chunk_id, tags, metadata, proof_count"
                )
                bm25_arms = [
                    dialect.build_bm25_arm(
                        table=fq_table("memory_units"),
                        cols=cols,
                        fact_type=fact_type,
                        bank_id_param="$1",
                        limit_param="$2",
                        text_param="$3",
                        tags_clause=tags_clause,
                        groups_clause=groups_clause,
                        arm_index=index,
                        text_search_extension=text_extension,
                        extra_where=created_clause,
                    )
                    for index, fact_type in enumerate(fact_types)
                ]
                bm25_params: List[Any] = [bank_id, limit, bm25_text]
                if tags:
                    bm25_params.append(tags)
                bm25_params.extend(groups_params)
                bm25_params.extend(created_params)
                bm25_rows = await conn.fetch("\nUNION ALL\n".join(bm25_arms), *bm25_params)
                for raw_row in bm25_rows:
                    row = dict(raw_row)
                    row.pop("source", None)
                    fact_type = row.get("fact_type")
                    if fact_type in result_dict:
                        result_dict[fact_type].append(RetrievalResult.from_db_row(row))

            hits_by_fact_type = await vector_task
        except BaseException:
            if not vector_task.done():
                vector_task.cancel()
            try:
                await vector_task
            except (asyncio.CancelledError, Exception):
                pass
            raise

        if hits_by_fact_type is None:
            raise RuntimeError("External vector index did not return a result set")
        semantic = await self._hydrate_vector_hits(
            conn,
            vector_index,
            hits_by_fact_type,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
        )

        missing_fact_types = [fact_type for fact_type in fact_types if not semantic[fact_type]]
        if missing_fact_types:
            sql_semantic = await self._retrieve_database_semantic_only(
                conn,
                query_embedding_str=query_embedding_str,
                bank_id=bank_id,
                fact_types=missing_fact_types,
                limit=limit,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
            )
            semantic.update(sql_semantic)

        for fact_type in fact_types:
            result_dict[fact_type] = semantic[fact_type] + result_dict[fact_type]
        return result_dict

    async def _expand_query(self, query_text: str, question_date: Optional[datetime] = None) -> dict:
        """
        Expand query using alias expansion strategy.

        For LLM-driven strategies, this returns a comprehensive analysis result.

        Args:
            query_text: Original query text
            question_date: Optional date when the question was asked

        Returns:
            Dict with analysis results containing:
                - rewritten_query: The rewritten query (or original if no expansion)
                - expanded_aliases: List of expanded entity terms
                - needs_expansion: Whether expansion was performed
                - needs_time_window: Whether time window filtering is needed
                - time_window_start: Start of time window (datetime or None)
                - time_window_end: End of time window (datetime or None)
        """
        result = {
            "rewritten_query": query_text,
            "expanded_aliases": [],
            "needs_expansion": False,
            "needs_time_window": False,
            "time_window_start": None,
            "time_window_end": None,
        }

        if not self._alias_expansion_enabled:
            return result

        strategy = self._get_query_rewriting_strategy()

        # Check if strategy supports LLM-driven analysis
        if hasattr(strategy, "analyze"):
            try:
                analysis_result = await strategy.analyze(query_text, llm=self._llm, question_date=question_date)

                result["rewritten_query"] = analysis_result.rewritten_query or query_text
                result["expanded_aliases"] = analysis_result.expanded_entities
                result["needs_expansion"] = analysis_result.needs_expansion
                result["needs_time_window"] = analysis_result.needs_time_window
                result["time_window_start"] = analysis_result.time_window_start
                result["time_window_end"] = analysis_result.time_window_end

                logger.debug(
                    f"LLM-driven query analysis: needs_expansion={result['needs_expansion']}, needs_time_window={result['needs_time_window']}"
                )
                if result["time_window_start"]:
                    logger.debug(f"Time window: {result['time_window_start']} to {result['time_window_end']}")

            except Exception as e:
                logger.warning(f"LLM-driven query analysis failed: {e}")
                return result
        else:
            # Legacy alias expansion
            if not strategy.should_expand(query_text):
                return result

            try:
                rewrite_result = await strategy.rewrite(query_text, llm=self._llm)
                aliases = rewrite_result.get(query_text, [])
                if aliases:
                    enriched_text = f"{query_text} {' '.join(aliases)}"
                    result["rewritten_query"] = enriched_text
                    result["expanded_aliases"] = aliases
                    result["needs_expansion"] = True
            except Exception as e:
                logger.warning(f"Legacy query rewriting failed: {e}")

        return result

    async def retrieve(
        self,
        conn,
        query_embedding_str: str,
        query_text: str,
        bank_id: str,
        fact_types: List[str],
        limit: int,
        tags: List[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: List[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        **kwargs,
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve memories using semantic + BM25 combined strategy.

        Args:
            conn: Database connection
            query_embedding_str: Query embedding as string (for semantic search)
            query_text: Original query text
            bank_id: Memory bank ID
            fact_types: List of fact types to retrieve
            limit: Maximum results per fact type
            tags: Optional tags for filtering
            tags_match: How to match tags
            tag_groups: Compound boolean tag filter groups
            created_after: Only return results created after this time
            created_before: Only return results created before this time
            **kwargs: Additional parameters, may include 'question_date' for query analysis

        Returns:
            Dict mapping fact_type -> list of RetrievalResult
        """
        result_dict: Dict[str, List[RetrievalResult]] = {ft: [] for ft in fact_types}

        # Get question_date from kwargs for LLM-driven analysis
        question_date = kwargs.get("question_date")
        # Perform query expansion/analysis
        expansion_result = await self._expand_query(query_text, question_date)
        rewritten_query = expansion_result["rewritten_query"]
        expanded_aliases = expansion_result["expanded_aliases"]

        tokens = tokenize_query(rewritten_query)
        hnsw_fetch = max(limit * 5, 100)

        vector_index = kwargs.get("vector_index")
        query_vector = kwargs.get("query_vector")
        if vector_index is not None and vector_index.is_external and query_vector is not None:
            try:
                return await self._retrieve_with_external_vector_index(
                    conn,
                    vector_index,
                    query_vector,
                    query_embedding_str=query_embedding_str,
                    rewritten_query=rewritten_query,
                    tokens=tokens,
                    bank_id=bank_id,
                    fact_types=fact_types,
                    limit=limit,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    created_after=created_after,
                    created_before=created_before,
                )
            except Exception:
                logger.warning(
                    "External semantic vector retrieval failed; falling back to the canonical database index",
                    exc_info=True,
                )

        cols = (
            "id, text, context, event_date, occurred_start, occurred_end, mentioned_at, "
            "fact_type, document_id, chunk_id, tags, metadata, projection, proof_count"
        )
        table = fq_table("memory_units")
        config = get_config()

        dialect = create_sql_dialect(getattr(conn, "backend_type", "postgresql"))

        _include_bm25 = bool(tokens)
        tags_param_idx = 5 if _include_bm25 else 3
        tags_clause = build_tags_where_clause_simple(tags, tags_param_idx, match=tags_match)

        tag_groups_param_start = tags_param_idx + (1 if tags else 0)
        groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

        _next_idx = tag_groups_param_start + len(groups_params)
        created_range_clause = ""
        created_range_params: List[Any] = []
        if created_after is not None:
            created_range_params.append(created_after)
            created_range_clause += f" AND updated_at > ${_next_idx}"
            _next_idx += 1
        if created_before is not None:
            created_range_params.append(created_before)
            created_range_clause += f" AND updated_at < ${_next_idx}"
            _next_idx += 1

        arms = [
            dialect.build_semantic_arm(
                table=table,
                cols=cols,
                fact_type=ft,
                embedding_param="$1",
                bank_id_param="$2",
                fetch_limit=hnsw_fetch,
                tags_clause=tags_clause,
                groups_clause=groups_clause,
                extra_where=created_range_clause,
            )
            for ft in fact_types
        ]

        if _include_bm25:
            text_ext = config.text_search_extension
            bm25_text_param: str = dialect.prepare_bm25_text(tokens, rewritten_query, text_search_extension=text_ext)
            for i, ft in enumerate(fact_types):
                arms.append(
                    dialect.build_bm25_arm(
                        table=table,
                        cols=cols,
                        fact_type=ft,
                        bank_id_param="$2",
                        limit_param="$3",
                        text_param="$4",
                        tags_clause=tags_clause,
                        groups_clause=groups_clause,
                        arm_index=i,
                        text_search_extension=text_ext,
                        extra_where=created_range_clause,
                    )
                )

        query = "\nUNION ALL\n".join(arms)

        params: List = [query_embedding_str, bank_id]
        if _include_bm25:
            params.append(limit)
            params.append(bm25_text_param)
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        try:
            rows = await conn.fetch(query, *params)
        except Exception as e:
            err_str = str(e)
            if _include_bm25 and ("DRG-10599" in err_str or "ORA-30600" in err_str or "ORA-29902" in err_str):
                logger.warning("Oracle Text CONTAINS failed (%s), falling back to semantic-only search", err_str[:120])
                fb_tags_idx = 3
                fb_tags_clause = build_tags_where_clause_simple(tags, fb_tags_idx, match=tags_match)
                fb_groups_start = fb_tags_idx + (1 if tags else 0)
                fb_groups_clause, _, _ = build_tag_groups_where_clause(tag_groups, fb_groups_start)
                fb_next_idx = fb_groups_start + len(groups_params)
                fb_created_clause = ""
                if created_after is not None:
                    fb_created_clause += f" AND updated_at > ${fb_next_idx}"
                    fb_next_idx += 1
                if created_before is not None:
                    fb_created_clause += f" AND updated_at < ${fb_next_idx}"
                    fb_next_idx += 1
                fb_arms = [
                    dialect.build_semantic_arm(
                        table=table,
                        cols=cols,
                        fact_type=ft,
                        embedding_param="$1",
                        bank_id_param="$2",
                        fetch_limit=hnsw_fetch,
                        tags_clause=fb_tags_clause,
                        groups_clause=fb_groups_clause,
                        extra_where=fb_created_clause,
                    )
                    for ft in fact_types
                ]
                fb_query = "\nUNION ALL\n".join(fb_arms)
                fb_params: List = [query_embedding_str, bank_id]
                if tags:
                    fb_params.append(tags)
                fb_params.extend(groups_params)
                fb_params.extend(created_range_params)
                rows = await conn.fetch(fb_query, *fb_params)
            else:
                raise

        sem_counts: Dict[str, int] = {ft: 0 for ft in fact_types}
        for r in rows:
            row = dict(r)
            source = row.pop("source")
            ft = row.get("fact_type")
            if ft not in result_dict:
                continue
            if source == "semantic":
                if sem_counts[ft] < limit:
                    result_dict[ft].append(RetrievalResult.from_db_row(row))
                    sem_counts[ft] += 1
            else:
                result_dict[ft].append(RetrievalResult.from_db_row(row))

        return result_dict


@retrieval_registry.register("temporal")
class TemporalRetrieval(RetrievalStrategy):
    """Temporal retrieval strategy with spreading activation.
    Supports session-based node expansion.
    """

    def __init__(
        self,
        session_expansion_weight: float = 0.3,
    ):
        """
        Initialize temporal retrieval with session-based node expansion.

        Args:
            session_expansion_weight: Weight for session-based node expansion (default 0.3)
        """
        self._session_expansion_weight = session_expansion_weight

    @property
    def name(self) -> str:
        return "temporal"

    async def retrieve(
        self,
        conn,
        query_embedding_str: str,
        query_text: str,
        bank_id: str,
        fact_types: List[str],
        limit: int,
        tags: List[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: List[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs,
    ) -> Dict[str, List[RetrievalResult]]:
        if start_date is None or end_date is None:
            return {ft: [] for ft in fact_types}

        from ..db.ops_postgresql import PostgreSQLOps

        graph_temporal_mode, graph_temporal_sigma_hours = _resolve_temporal_graph_config(kwargs.get("graph_config"))
        ops = kwargs.get("ops") or PostgreSQLOps()
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        tags_clause = build_tags_where_clause_simple(tags, 7, match=tags_match)
        tag_groups_param_start = 7 + (1 if tags else 0)
        groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

        _next_idx = tag_groups_param_start + len(groups_params)
        created_range_clause = ""
        created_range_params: List[Any] = []
        if created_after is not None:
            created_range_params.append(created_after)
            created_range_clause += f" AND updated_at > ${_next_idx}"
            _next_idx += 1
        if created_before is not None:
            created_range_params.append(created_before)
            created_range_clause += f" AND updated_at < ${_next_idx}"

        params: List = [query_embedding_str, bank_id, fact_types, start_date, end_date, 0.1]
        if tags:
            params.append(tags)
        params.extend(groups_params)
        params.extend(created_range_params)

        entry_points = await conn.fetch(
            f"""
            WITH date_ranked AS MATERIALIZED (
                SELECT id, fact_type,
                       ROW_NUMBER() OVER (
                           PARTITION BY fact_type
                           ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC NULLS LAST
                       ) AS rn
                FROM {fq_table("memory_units")}
                WHERE bank_id = $2
                  AND fact_type = ANY($3)
                  AND embedding IS NOT NULL
                  AND (
                      (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
                       AND occurred_start <= $5 AND occurred_end >= $4)
                      OR
                      (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5)
                      OR
                      (occurred_start IS NOT NULL AND occurred_start BETWEEN $4 AND $5)
                      OR
                      (occurred_end IS NOT NULL AND occurred_end BETWEEN $4 AND $5)
                  )
                  {tags_clause}
                  {groups_clause}
                  {created_range_clause}
            ),
            sim_ranked AS (
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.proof_count, mu.document_id, mu.chunk_id, mu.tags, mu.metadata, mu.projection,
                       1 - (mu.embedding <=> $1::vector) AS similarity,
                       ROW_NUMBER() OVER (PARTITION BY mu.fact_type ORDER BY mu.embedding <=> $1::vector) AS sim_rn
                FROM date_ranked dr
                JOIN {fq_table("memory_units")} mu ON mu.id = dr.id
                WHERE dr.rn <= 50
                  AND (1 - (mu.embedding <=> $1::vector)) >= $6
            )
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, fact_type, proof_count, document_id, chunk_id, tags, metadata, projection, similarity
            FROM sim_ranked
            WHERE sim_rn <= 10
            """,
            *params,
        )

        if not entry_points:
            return {ft: [] for ft in fact_types}

        entries_by_ft: Dict[str, List] = {ft: [] for ft in fact_types}
        for ep in entry_points:
            ft = ep["fact_type"]
            if ft in entries_by_ft:
                entries_by_ft[ft].append(ep)

        total_days = (end_date - start_date).total_seconds() / 86400
        mid_date = start_date + (end_date - start_date) / 2

        results_by_ft: Dict[str, List[RetrievalResult]] = {}

        for ft in fact_types:
            ft_entry_points = entries_by_ft.get(ft, [])
            if not ft_entry_points:
                results_by_ft[ft] = []
                continue

            results = []
            visited = set()
            node_scores = {}

            for ep in ft_entry_points:
                unit_id = str(ep["id"])
                visited.add(unit_id)

                best_date = None
                if ep["occurred_start"] is not None and ep["occurred_end"] is not None:
                    best_date = ep["occurred_start"] + (ep["occurred_end"] - ep["occurred_start"]) / 2
                elif ep["occurred_start"] is not None:
                    best_date = ep["occurred_start"]
                elif ep["occurred_end"] is not None:
                    best_date = ep["occurred_end"]
                elif ep["mentioned_at"] is not None:
                    best_date = ep["mentioned_at"]

                if best_date:
                    days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
                    temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
                else:
                    temporal_proximity = 0.5

                ep_result = RetrievalResult.from_db_row(dict(ep))
                ep_result.temporal_score = temporal_proximity
                ep_result.temporal_proximity = temporal_proximity
                results.append(ep_result)
                node_scores[unit_id] = (ep["similarity"], 1.0)

            frontier = list(node_scores.keys())
            budget_remaining = limit - len(ft_entry_points)
            batch_size = 20
            per_source_limit = 10
            temporal_window_seconds = 24 * 3600.0
            temporal_sigma_seconds = graph_temporal_sigma_hours * 3600.0
            max_iterations = 5
            iteration = 0

            spreading_filter_start = 9 if graph_temporal_mode == "btree" else 7
            spreading_tags_clause = build_tags_where_clause_simple(
                tags, spreading_filter_start, table_alias="mu.", match=tags_match
            )
            spreading_groups_param_start = spreading_filter_start + (1 if tags else 0)
            spreading_groups_clause, spreading_groups_params, _ = build_tag_groups_where_clause(
                tag_groups, spreading_groups_param_start, table_alias="mu."
            )
            spreading_next_idx = spreading_groups_param_start + len(spreading_groups_params)
            spreading_created_range_clause = ""
            spreading_created_range_params: List[Any] = []
            if created_after is not None:
                spreading_created_range_params.append(created_after)
                spreading_created_range_clause += f" AND mu.updated_at > ${spreading_next_idx}"
                spreading_next_idx += 1
            if created_before is not None:
                spreading_created_range_params.append(created_before)
                spreading_created_range_clause += f" AND mu.updated_at < ${spreading_next_idx}"
            while frontier and budget_remaining > 0 and iteration < max_iterations:
                iteration += 1
                batch_ids = frontier[:batch_size]
                frontier = frontier[batch_size:]

                spreading_params = [query_embedding_str, batch_ids, ft, 0.1, per_source_limit, bank_id]
                if graph_temporal_mode == "btree":
                    spreading_lateral = ops.build_temporal_btree_spreading_lateral(
                        fq_table("memory_units"),
                        fq_table("memory_links"),
                        5,
                        6,
                        3,
                        7,
                        8,
                    )
                    spreading_params.extend([temporal_window_seconds, temporal_sigma_seconds])
                else:
                    spreading_lateral = ops.build_temporal_links_spreading_lateral(fq_table("memory_links"), 5)
                if tags:
                    spreading_params.append(tags)
                spreading_params.extend(spreading_groups_params)
                spreading_params.extend(spreading_created_range_params)

                neighbors = await conn.fetch(
                    f"""
                    SELECT src.from_unit_id, mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id, mu.chunk_id, mu.tags, mu.metadata, mu.projection,
                           l.weight, l.link_type,
                           1 - (mu.embedding <=> $1::vector) AS similarity
                    FROM unnest($2::uuid[]) AS src(from_unit_id)
                    JOIN {fq_table("memory_units")} src_mu ON src_mu.id = src.from_unit_id
                    CROSS JOIN LATERAL (
                        {spreading_lateral}
                    ) l
                    JOIN {fq_table("memory_units")} mu ON mu.id = l.to_unit_id
                    WHERE mu.bank_id = $6
                      AND mu.fact_type = $3
                      AND mu.embedding IS NOT NULL
                      AND (1 - (mu.embedding <=> $1::vector)) >= $4
                      {spreading_tags_clause}
                      {spreading_groups_clause}
                      {spreading_created_range_clause}
                    """,
                    *spreading_params,
                )

                for n in neighbors:
                    neighbor_id = str(n["id"])
                    if neighbor_id in visited:
                        continue

                    visited.add(neighbor_id)
                    budget_remaining -= 1

                    parent_id = str(n["from_unit_id"])
                    _, parent_temporal_score = node_scores.get(parent_id, (0.5, 0.5))

                    neighbor_best_date = None
                    if n["occurred_start"] is not None and n["occurred_end"] is not None:
                        neighbor_best_date = n["occurred_start"] + (n["occurred_end"] - n["occurred_start"]) / 2
                    elif n["occurred_start"] is not None:
                        neighbor_best_date = n["occurred_start"]
                    elif n["occurred_end"] is not None:
                        neighbor_best_date = n["occurred_end"]
                    elif n["mentioned_at"] is not None:
                        neighbor_best_date = n["mentioned_at"]

                    if neighbor_best_date:
                        days_from_mid = abs((neighbor_best_date - mid_date).total_seconds() / 86400)
                        neighbor_temporal_proximity = (
                            1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
                        )
                    else:
                        neighbor_temporal_proximity = 0.3

                    # Use session-based node expansion
                    link_type = n["link_type"]
                    if link_type in ("causes", "caused_by"):
                        causal_boost = 2.0
                    elif link_type in ("enables", "prevents"):
                        causal_boost = 1.5
                    else:
                        causal_boost = 1.0

                    propagated_temporal = parent_temporal_score * n["weight"] * causal_boost * 0.7

                    combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)

                    neighbor_result = RetrievalResult.from_db_row(dict(n))
                    neighbor_result.temporal_score = combined_temporal
                    neighbor_result.temporal_proximity = neighbor_temporal_proximity
                    results.append(neighbor_result)

                    if budget_remaining > 0 and combined_temporal > 0.2:
                        node_scores[neighbor_id] = (n["similarity"], combined_temporal)
                        frontier.append(neighbor_id)

                    if budget_remaining <= 0:
                        break

            if self._session_expansion_weight > 0 and ft_entry_points:
                doc_ids = [ep["document_id"] for ep in ft_entry_points if ep.get("document_id")]
                if doc_ids:
                    session_filter_start = 6
                    session_tags_clause = build_tags_where_clause_simple(
                        tags, session_filter_start, table_alias="mu.", match=tags_match
                    )
                    session_groups_param_start = session_filter_start + (1 if tags else 0)
                    session_groups_clause, session_groups_params, session_next_idx = build_tag_groups_where_clause(
                        tag_groups, session_groups_param_start, table_alias="mu."
                    )
                    session_created_range_clause = ""
                    session_created_range_params: List[Any] = []
                    if created_after is not None:
                        session_created_range_params.append(created_after)
                        session_created_range_clause += f" AND mu.updated_at > ${session_next_idx}"
                        session_next_idx += 1
                    if created_before is not None:
                        session_created_range_params.append(created_before)
                        session_created_range_clause += f" AND mu.updated_at < ${session_next_idx}"

                    session_params: List[Any] = [query_embedding_str, bank_id, ft, doc_ids, limit]
                    if tags:
                        session_params.append(tags)
                    session_params.extend(session_groups_params)
                    session_params.extend(session_created_range_params)

                    session_neighbors = await conn.fetch(
                        f"""
                        SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.proof_count, mu.document_id, mu.chunk_id, mu.tags, mu.metadata, mu.projection,
                               1 - (mu.embedding <=> $1::vector) AS similarity
                        FROM {fq_table("memory_units")} mu
                        WHERE mu.bank_id = $2
                          AND mu.fact_type = $3
                          AND mu.document_id = ANY($4::text[])
                          AND mu.embedding IS NOT NULL
                          {session_tags_clause}
                          {session_groups_clause}
                          {session_created_range_clause}
                        LIMIT $5
                        """,
                        *session_params,
                    )

                    result_ids = {r.id for r in results}
                    for sn in session_neighbors:
                        sn_id = str(sn["id"])
                        if sn_id in result_ids or sn_id in visited:
                            continue

                        result_ids.add(sn_id)
                        session_similarity = float(sn["similarity"])
                        session_score = self._session_expansion_weight * session_similarity

                        sn_best_date = None
                        if sn["occurred_start"] is not None and sn["occurred_end"] is not None:
                            sn_best_date = sn["occurred_start"] + (sn["occurred_end"] - sn["occurred_start"]) / 2
                        elif sn["occurred_start"] is not None:
                            sn_best_date = sn["occurred_start"]
                        elif sn["occurred_end"] is not None:
                            sn_best_date = sn["occurred_end"]
                        elif sn["mentioned_at"] is not None:
                            sn_best_date = sn["mentioned_at"]

                        sn_temporal_proximity = 0.5
                        if sn_best_date and total_days > 0:
                            sn_days_from_mid = abs((sn_best_date - mid_date).total_seconds() / 86400)
                            sn_temporal_proximity = 1.0 - min(sn_days_from_mid / (total_days / 2), 1.0)

                        sn_result = RetrievalResult.from_db_row(dict(sn))
                        sn_result.temporal_score = session_score
                        sn_result.temporal_proximity = sn_temporal_proximity
                        results.append(sn_result)

            results_by_ft[ft] = results

        return results_by_ft


@graph_retrieval_registry.register("link_expansion")
class LinkExpansionGraphRetrieval(GraphRetrievalStrategy):
    """Link expansion graph retrieval strategy."""

    def __init__(self):
        self._delegate = LinkExpansionRetriever()

    @property
    def name(self) -> str:
        return "link_expansion"

    async def retrieve(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        query_text: str | None = None,
        semantic_seeds: List[RetrievalResult] | None = None,
        temporal_seeds: List[RetrievalResult] | None = None,
        tags: List[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: List[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        graph_config: Any | None = None,
    ) -> tuple[List[RetrievalResult], GraphRetrievalTimings | None]:
        return await self._delegate.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=fact_type,
            budget=budget,
            query_text=query_text,
            semantic_seeds=semantic_seeds,
            temporal_seeds=temporal_seeds,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
            graph_config=graph_config,
        )


@fusion_registry.register("rrf")
class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion strategy."""

    def __init__(self, k: int = 60):
        self._k = k

    @property
    def name(self) -> str:
        return "rrf"

    def fuse(
        self,
        result_lists: List[List[RetrievalResult]],
        source_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MergedCandidate]:
        return reciprocal_rank_fusion(result_lists, k=self._k)


@fusion_registry.register("weighted")
class WeightedFusion(FusionStrategy):
    """Weighted fusion strategy with configurable weights per source."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or {"semantic": 0.4, "bm25": 0.2, "graph": 0.2, "temporal": 0.2}

    @property
    def name(self) -> str:
        return "weighted"

    def fuse(
        self,
        result_lists: List[List[RetrievalResult]],
        source_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MergedCandidate]:
        default_names = ["semantic", "bm25", "graph", "temporal"]
        names = source_names if source_names else default_names[: len(result_lists)]

        rrf_scores = {}
        source_ranks = {}
        all_retrievals = {}

        for source_idx, (source_name, results) in enumerate(zip(names, result_lists)):
            weight = self._weights.get(source_name, 1.0)
            for rank, retrieval in enumerate(results, start=1):
                doc_id = retrieval.id
                if doc_id not in all_retrievals:
                    all_retrievals[doc_id] = retrieval

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    source_ranks[doc_id] = {}

                rrf_scores[doc_id] += weight / (60 + rank)
                source_ranks[doc_id][f"{source_name}_rank"] = rank

        merged_results = []
        for rrf_rank, (doc_id, rrf_score) in enumerate(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            merged_candidate = MergedCandidate(
                retrieval=all_retrievals[doc_id],
                rrf_score=rrf_score,
                rrf_rank=rrf_rank,
                source_ranks=source_ranks[doc_id],
            )
            merged_results.append(merged_candidate)

        return merged_results


@fusion_registry.register("simple")
class SimpleFusion(FusionStrategy):
    """Simple fusion that combines results without complex scoring."""

    @property
    def name(self) -> str:
        return "simple"

    def fuse(
        self,
        result_lists: List[List[RetrievalResult]],
        source_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MergedCandidate]:
        seen = set()
        merged = []

        for results in result_lists:
            for retrieval in results:
                if retrieval.id not in seen:
                    seen.add(retrieval.id)
                    merged.append(
                        MergedCandidate(
                            retrieval=retrieval,
                            rrf_score=len(seen),
                            rrf_rank=len(merged),
                            source_ranks={},
                        )
                    )

        return merged


_RECENCY_ALPHA: float = 0.2
_TEMPORAL_ALPHA: float = 0.2
_PROOF_COUNT_ALPHA: float = 0.1


def apply_combined_scoring(
    scored_results: List[ScoredResult],
    now: datetime,
    recency_alpha: float = _RECENCY_ALPHA,
    temporal_alpha: float = _TEMPORAL_ALPHA,
    proof_count_alpha: float = _PROOF_COUNT_ALPHA,
    is_passthrough_reranker: bool = False,
) -> None:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    if is_passthrough_reranker and scored_results:
        n = len(scored_results)
        sorted_by_rrf = sorted(
            scored_results,
            key=lambda s: getattr(getattr(s, "candidate", None), "rrf_score", 0.0),
            reverse=True,
        )
        denom = max(1, n - 1)
        for new_rank, sr in enumerate(sorted_by_rrf):
            sr.cross_encoder_score_normalized = 1.0 - (0.9 * new_rank / denom)

    for sr in scored_results:
        sr.recency = 0.5
        if sr.retrieval.occurred_start:
            occurred = sr.retrieval.occurred_start
            if occurred.tzinfo is None:
                occurred = occurred.replace(tzinfo=UTC)
            days_ago = (now - occurred).total_seconds() / 86400
            sr.recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))

        sr.temporal = sr.retrieval.temporal_proximity if sr.retrieval.temporal_proximity is not None else 0.5

        proof_count = sr.retrieval.proof_count
        if proof_count is not None and proof_count >= 1:
            proof_norm = min(1.0, max(0.0, 0.5 + (math.log(proof_count) / 10.0)))
        else:
            proof_norm = 0.5

        sr.rrf_normalized = 0.0

        recency_boost = 1.0 + recency_alpha * (sr.recency - 0.5)
        temporal_boost = 1.0 + temporal_alpha * (sr.temporal - 0.5)
        proof_count_boost = 1.0 + proof_count_alpha * (proof_norm - 0.5)
        sr.combined_score = sr.cross_encoder_score_normalized * recency_boost * temporal_boost * proof_count_boost
        sr.weight = sr.combined_score


@reranking_registry.register("cross_encoder")
class CrossEncoderReranking(RerankingStrategy):
    """Cross-encoder neural reranking strategy."""

    def __init__(self, cross_encoder=None):
        if cross_encoder is None:
            from hms_api.engine.cross_encoder import create_cross_encoder_from_env

            cross_encoder = create_cross_encoder_from_env()
        self.cross_encoder = cross_encoder
        self._initialized = False

    @property
    def name(self) -> str:
        return "cross_encoder"

    @property
    def is_passthrough(self) -> bool:
        return getattr(self.cross_encoder, "is_passthrough", False)

    async def ensure_initialized(self):
        if self._initialized:
            return

        cross_encoder = self.cross_encoder
        if cross_encoder.provider_name == "local":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: asyncio.run(cross_encoder.initialize()))
        else:
            await cross_encoder.initialize()
        self._initialized = True

    async def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate],
        **kwargs,
    ) -> List[ScoredResult]:
        if not candidates:
            return []

        pairs = []
        for candidate in candidates:
            retrieval = candidate.retrieval
            doc_text = retrieval.text
            if retrieval.context:
                doc_text = f"{retrieval.context}: {doc_text}"

            if retrieval.occurred_start:
                occurred_start = retrieval.occurred_start
                date_iso = occurred_start.strftime("%Y-%m-%d")
                date_readable = occurred_start.strftime("%B %d, %Y")
                doc_text = f"[Date: {date_readable} ({date_iso})] {doc_text}"

            pairs.append([query, doc_text])

        scores = await self.cross_encoder.predict(pairs)

        if self.cross_encoder.scores_are_normalized:
            normalized_scores = [float(s) for s in scores]
        else:
            import numpy as np

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            normalized_scores = [sigmoid(score) for score in scores]

        scored_results = []
        for candidate, raw_score, norm_score in zip(candidates, scores, normalized_scores):
            raw = float(raw_score)
            norm = float(norm_score)
            if math.isnan(raw):
                raw = 0.0
            if math.isnan(norm):
                norm = 0.0
            scored_result = ScoredResult(
                candidate=candidate,
                cross_encoder_score=raw,
                cross_encoder_score_normalized=norm,
                weight=norm,
            )
            scored_results.append(scored_result)

        scored_results.sort(key=lambda x: x.weight, reverse=True)

        return scored_results


@reranking_registry.register("passthrough")
class PassthroughReranking(RerankingStrategy):
    """Passthrough reranking strategy (no actual reranking)."""

    @property
    def name(self) -> str:
        return "passthrough"

    @property
    def is_passthrough(self) -> bool:
        return True

    async def ensure_initialized(self):
        pass

    async def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate],
        **kwargs,
    ) -> List[ScoredResult]:
        scored_results = []
        for candidate in candidates:
            scored_result = ScoredResult(
                candidate=candidate,
                cross_encoder_score=0.5,
                cross_encoder_score_normalized=0.5,
                weight=0.5,
            )
            scored_results.append(scored_result)
        return scored_results


@reranking_registry.register("rrf_only")
class RRFOnlyReranking(RerankingStrategy):
    """RRF-only reranking strategy."""

    @property
    def name(self) -> str:
        return "rrf_only"

    @property
    def is_passthrough(self) -> bool:
        return False

    async def ensure_initialized(self):
        pass

    async def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate],
        **kwargs,
    ) -> List[ScoredResult]:
        scored_results = []
        n = len(candidates)
        sorted_by_rrf = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)

        for i, candidate in enumerate(sorted_by_rrf):
            norm_score = 1.0 - (0.9 * i / max(1, n - 1)) if n > 1 else 1.0
            scored_result = ScoredResult(
                candidate=candidate,
                cross_encoder_score=norm_score,
                cross_encoder_score_normalized=norm_score,
                weight=norm_score,
            )
            scored_results.append(scored_result)

        return scored_results
