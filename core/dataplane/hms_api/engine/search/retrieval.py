"""
Retrieval module for 4-way parallel search.

Implements:
1. Semantic retrieval (vector similarity)
2. BM25 retrieval (keyword/full-text search)
3. Graph retrieval (via pluggable GraphRetriever interface)
4. Temporal retrieval (time-aware search with spreading)

Backward-compatible wrapper that delegates to modular implementations.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from ...config import get_config
from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from ..sql import create_sql_dialect
from .graph_retrieval import GraphRetriever
from .link_expansion_retrieval import LinkExpansionRetriever
from .tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause_simple
from .types import GraphRetrievalTimings, RetrievalResult

logger = logging.getLogger(__name__)


def tokenize_query(query_text: str) -> list[str]:
    """Normalize query text and split into BM25 tokens."""
    return re.sub(r"[^\w\s]", " ", query_text.lower()).split()


@dataclass
class ParallelRetrievalResult:
    """Result from parallel retrieval across all methods."""

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph: list[RetrievalResult]
    temporal: list[RetrievalResult] | None
    timings: dict[str, float] = field(default_factory=dict)
    temporal_constraint: tuple | None = None
    graph_timings: list[GraphRetrievalTimings] = field(default_factory=list)
    max_conn_wait: float = 0.0


@dataclass
class MultiFactTypeRetrievalResult:
    """Result from retrieval across all fact types."""

    results_by_fact_type: dict[str, ParallelRetrievalResult]
    timings: dict[str, float] = field(default_factory=dict)
    max_conn_wait: float = 0.0


_default_graph_retriever: GraphRetriever | None = None


def get_default_graph_retriever() -> GraphRetriever:
    """Get or create the default graph retriever based on config."""
    global _default_graph_retriever
    if _default_graph_retriever is None:
        config = get_config()
        retriever_type = config.graph_retriever.lower()
        if retriever_type == "link_expansion":
            _default_graph_retriever = LinkExpansionRetriever()
            logger.info("Using LinkExpansion graph retriever")
        else:
            logger.warning(f"Unknown graph retriever '{retriever_type}', falling back to link_expansion")
            _default_graph_retriever = LinkExpansionRetriever()
    return _default_graph_retriever


def set_default_graph_retriever(retriever: GraphRetriever) -> None:
    """Set the default graph retriever (for configuration/testing)."""
    global _default_graph_retriever
    _default_graph_retriever = retriever


async def retrieve_semantic_bm25_combined(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    llm: Any | None = None,
    query_rewriting_strategy_name: str = "llm_based",
    alias_expansion_enabled: bool = True,
    question_date: datetime | None = None,
    vector_index: Any | None = None,
    query_vector: list[float] | None = None,
) -> tuple[dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]], dict]:
    """
    Combined semantic + BM25 retrieval for multiple fact types in a single query.

    Uses UNION ALL of per-fact_type subqueries so that each arm has its own
    ORDER BY ... LIMIT, enabling the partial HNSW indexes per fact_type instead
    of forcing a full sequential scan.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        query_text: Query text
        bank_id: Memory bank ID
        fact_types: List of fact types to retrieve
        limit: Maximum results per fact type
        tags: Optional tags for filtering
        tags_match: How to match tags
        tag_groups: Compound boolean tag filter groups
        created_after: Only return results created after this time
        created_before: Only return results created before this time
        llm: Optional LLM provider for query rewriting
        query_rewriting_strategy_name: Name of query rewriting strategy
        alias_expansion_enabled: Whether to enable LLM-based alias expansion
        question_date: Optional date when the question was asked (for LLM-driven analysis)

    Returns:
        Tuple of:
            - Dict mapping fact_type -> (semantic_results, bm25_results)
            - Dict containing query analysis metadata (time_window_start, time_window_end, needs_time_window)
    """
    from .implementations import SemanticBM25Retrieval

    strategy = SemanticBM25Retrieval(
        query_rewriting_strategy_name=query_rewriting_strategy_name,
        alias_expansion_enabled=alias_expansion_enabled,
    )
    if llm is not None:
        strategy.set_llm(llm)

    # Get query analysis result from strategy
    analysis_result = {}
    if alias_expansion_enabled and llm is not None:
        analysis_result = await strategy._expand_query(query_text, question_date)

    results = await strategy.retrieve(
        conn=conn,
        query_embedding_str=query_emb_str,
        query_text=query_text,
        bank_id=bank_id,
        fact_types=fact_types,
        limit=limit,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        question_date=question_date,
        vector_index=vector_index,
        query_vector=query_vector,
    )

    result_dict: dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]] = {ft: ([], []) for ft in fact_types}
    for ft, ft_results in results.items():
        for result in ft_results:
            if result.similarity is not None:
                result_dict[ft][0].append(result)
            elif result.bm25_score is not None:
                result_dict[ft][1].append(result)
            else:
                result_dict[ft][0].append(result)

    return result_dict, analysis_result


async def retrieve_temporal_combined(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_types: list[str],
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    session_expansion_weight: float = 0.3,
) -> dict[str, list[RetrievalResult]]:
    """
    Temporal retrieval for multiple fact types in a single query.

    Batches the entry point query using window functions to get top-N per fact type,
    then runs spreading for each fact type.

    Returns:
        Dict mapping fact_type -> list of RetrievalResult
    """
    from .implementations import TemporalRetrieval

    strategy = TemporalRetrieval(
        session_expansion_weight=session_expansion_weight,
    )
    return await strategy.retrieve(
        conn=conn,
        query_embedding_str=query_emb_str,
        query_text="",
        bank_id=bank_id,
        fact_types=fact_types,
        limit=budget,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        start_date=start_date,
        end_date=end_date,
    )


async def retrieve_all_fact_types_parallel(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_types: list[str],
    thinking_budget: int,
    question_date: datetime | None = None,
    query_analyzer: Optional["QueryAnalyzer"] = None,
    graph_retriever: GraphRetriever | None = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    llm: Any | None = None,
    query_rewriting_strategy_name: str = "llm_based",
    alias_expansion_enabled: bool = True,
    session_expansion_weight: float = 0.3,
    vector_index: Any | None = None,
    query_vector: list[float] | None = None,
) -> MultiFactTypeRetrievalResult:
    """
    Optimized retrieval for multiple fact types using batched queries.

    This reduces database round-trips by:
    1. Combining semantic + BM25 into one CTE query for ALL fact types (1 query instead of 2N)
    2. Running graph retrieval per fact type in parallel (N parallel tasks)
    3. Running temporal retrieval per fact type in parallel (N parallel tasks)

    Returns:
        MultiFactTypeRetrievalResult with results organized by fact type
    """
    import time

    retriever = graph_retriever or get_default_graph_retriever()
    start_time = time.time()
    timings: dict[str, float] = {}

    temporal_extraction_start = time.time()
    from .temporal_extraction import extract_temporal_constraint

    # Extract temporal constraint using native logic
    native_temporal_constraint = extract_temporal_constraint(
        query_text, reference_date=question_date, analyzer=query_analyzer
    )
    temporal_extraction_time = time.time() - temporal_extraction_start
    timings["temporal_extraction"] = temporal_extraction_time

    semantic_bm25_start = time.time()
    temporal_results_by_ft: dict[str, list[RetrievalResult]] = {}
    temporal_time = 0.0
    query_analysis_result: dict = {}

    async with acquire_with_retry(pool) as conn:
        conn_wait = time.time() - semantic_bm25_start

        # Retrieve semantic + BM25 results and query analysis metadata
        semantic_bm25_results, query_analysis_result = await retrieve_semantic_bm25_combined(
            conn,
            query_embedding_str,
            query_text,
            bank_id,
            fact_types,
            thinking_budget,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
            llm=llm,
            query_rewriting_strategy_name=query_rewriting_strategy_name,
            alias_expansion_enabled=alias_expansion_enabled,
            question_date=question_date,
            vector_index=vector_index,
            query_vector=query_vector,
        )
        semantic_bm25_time = time.time() - semantic_bm25_start

        # Fusion of native temporal constraint and LLM-driven time window
        # Priority: LLM-driven time window takes precedence when available and requested
        temporal_constraint = native_temporal_constraint

        llm_time_window_start = query_analysis_result.get("time_window_start")
        llm_time_window_end = query_analysis_result.get("time_window_end")
        llm_needs_time_window = query_analysis_result.get("needs_time_window", False)

        if llm_needs_time_window and llm_time_window_start and llm_time_window_end:
            # LLM-driven time window takes precedence
            temporal_constraint = (llm_time_window_start, llm_time_window_end)
            logger.debug(f"Using LLM-driven time window: {llm_time_window_start} to {llm_time_window_end}")
        elif native_temporal_constraint:
            logger.debug(f"Using native temporal constraint: {native_temporal_constraint}")

        if temporal_constraint:
            tc_start, tc_end = temporal_constraint
            temporal_start = time.time()
            temporal_results_by_ft = await retrieve_temporal_combined(
                conn,
                query_embedding_str,
                bank_id,
                fact_types,
                tc_start,
                tc_end,
                budget=thinking_budget,
                semantic_threshold=0.1,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
                session_expansion_weight=session_expansion_weight,
            )
            temporal_time = time.time() - temporal_start

    timings["semantic_bm25_combined"] = semantic_bm25_time
    timings["temporal_combined"] = temporal_time

    async def run_graph_for_fact_type(
        ft: str,
    ) -> tuple[str, list[RetrievalResult], float, GraphRetrievalTimings | None]:
        graph_start = time.time()
        results, graph_timing = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=ft,
            budget=thinking_budget,
            query_text=query_text,
            semantic_seeds=None,
            temporal_seeds=None,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
        )
        return ft, results, time.time() - graph_start, graph_timing

    graph_tasks = [run_graph_for_fact_type(ft) for ft in fact_types]
    graph_results_list = await asyncio.gather(*graph_tasks)

    results_by_fact_type: dict[str, ParallelRetrievalResult] = {}
    max_conn_wait = conn_wait
    all_graph_timings: list[GraphRetrievalTimings] = []

    for ft in fact_types:
        semantic_results, bm25_results = semantic_bm25_results.get(ft, ([], []))

        graph_results = []
        graph_time = 0.0
        graph_timing = None
        for gr in graph_results_list:
            if gr[0] == ft:
                graph_results = gr[1]
                graph_time = gr[2]
                graph_timing = gr[3]
                if graph_timing:
                    all_graph_timings.append(graph_timing)
                break

        temporal_results = temporal_results_by_ft.get(ft) if temporal_constraint else None
        if temporal_results is not None and len(temporal_results) == 0:
            temporal_results = None

        results_by_fact_type[ft] = ParallelRetrievalResult(
            semantic=semantic_results,
            bm25=bm25_results,
            graph=graph_results,
            temporal=temporal_results,
            timings={
                "semantic": semantic_bm25_time / 2,
                "bm25": semantic_bm25_time / 2,
                "graph": graph_time,
                "temporal": temporal_time,
                "temporal_extraction": temporal_extraction_time,
            },
            temporal_constraint=temporal_constraint,
            graph_timings=[graph_timing] if graph_timing else [],
            max_conn_wait=max_conn_wait,
        )

    total_time = time.time() - start_time
    timings["total"] = total_time

    return MultiFactTypeRetrievalResult(
        results_by_fact_type=results_by_fact_type,
        timings=timings,
        max_conn_wait=max_conn_wait,
    )
