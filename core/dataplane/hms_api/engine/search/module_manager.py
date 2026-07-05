"""
Module manager for search retrieval strategies.

Provides a unified interface for configuring and selecting different retrieval,
fusion, and reranking strategies for ablation experiments.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .strategies import (
    FusionStrategy,
    GraphRetrievalStrategy,
    RetrievalStrategy,
    RerankingStrategy,
    fusion_registry,
    graph_retrieval_registry,
    retrieval_registry,
    reranking_registry,
)
from .tags import TagGroup, TagsMatch
from .types import GraphRetrievalTimings, MergedCandidate, RetrievalResult, ScoredResult
from .causal_strategies import CausalLinkStrategy, causal_registry

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    Configuration for search strategy selection.

    Allows easy swapping of strategies for ablation experiments.
    """

    retrieval_strategy: str = "semantic_bm25"
    graph_retrieval_strategy: str = "link_expansion"
    fusion_strategy: str = "rrf"
    reranking_strategy: str = "cross_encoder"
    causal_strategy: str = "memory_links"
    session_expansion_weight: float = 0.3

    retrieval_params: Dict[str, Any] = field(default_factory=dict)
    graph_retrieval_params: Dict[str, Any] = field(default_factory=dict)
    fusion_params: Dict[str, Any] = field(default_factory=dict)
    reranking_params: Dict[str, Any] = field(default_factory=dict)
    causal_params: Dict[str, Any] = field(default_factory=dict)


class SearchModuleManager:
    """
    Unified manager for search modules.

    Provides:
    - Strategy selection by name
    - Strategy instantiation with custom parameters
    - Easy swapping for ablation experiments
    - Centralized configuration
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self._config = config or SearchConfig()
        self._retrieval_strategy: Optional[RetrievalStrategy] = None
        self._graph_retrieval_strategy: Optional[GraphRetrievalStrategy] = None
        self._fusion_strategy: Optional[FusionStrategy] = None
        self._reranking_strategy: Optional[RerankingStrategy] = None
        self._causal_strategy: Optional[CausalLinkStrategy] = None

    @property
    def config(self) -> SearchConfig:
        return self._config

    def set_config(self, config: SearchConfig) -> None:
        """Update configuration and reset cached strategies."""
        self._config = config
        self._retrieval_strategy = None
        self._graph_retrieval_strategy = None
        self._fusion_strategy = None
        self._reranking_strategy = None
        self._causal_strategy = None

    def get_retrieval_strategy(self, name: Optional[str] = None) -> RetrievalStrategy:
        """Get or create a retrieval strategy instance."""
        strategy_name = name or self._config.retrieval_strategy
        if self._retrieval_strategy is None or self._retrieval_strategy.name != strategy_name:
            params = self._config.retrieval_params.get(strategy_name, {})

            if strategy_name == "temporal":
                params["session_expansion_weight"] = self._config.session_expansion_weight
                logger.info(f"Temporal retrieval will use session expansion weight: {self._config.session_expansion_weight}")

            self._retrieval_strategy = retrieval_registry.create(strategy_name, **params)
            logger.info(f"Using retrieval strategy: {strategy_name}")
        return self._retrieval_strategy

    def get_graph_retrieval_strategy(self, name: Optional[str] = None) -> GraphRetrievalStrategy:
        """Get or create a graph retrieval strategy instance."""
        strategy_name = name or self._config.graph_retrieval_strategy
        if self._graph_retrieval_strategy is None or self._graph_retrieval_strategy.name != strategy_name:
            params = self._config.graph_retrieval_params.get(strategy_name, {})
            self._graph_retrieval_strategy = graph_retrieval_registry.create(strategy_name, **params)
            logger.info(f"Using graph retrieval strategy: {strategy_name}")
        return self._graph_retrieval_strategy

    def get_fusion_strategy(self, name: Optional[str] = None) -> FusionStrategy:
        """Get or create a fusion strategy instance."""
        strategy_name = name or self._config.fusion_strategy
        if self._fusion_strategy is None or self._fusion_strategy.name != strategy_name:
            params = self._config.fusion_params.get(strategy_name, {})
            self._fusion_strategy = fusion_registry.create(strategy_name, **params)
            logger.info(f"Using fusion strategy: {strategy_name}")
        return self._fusion_strategy

    def get_reranking_strategy(self, name: Optional[str] = None) -> RerankingStrategy:
        """Get or create a reranking strategy instance."""
        strategy_name = name or self._config.reranking_strategy
        if self._reranking_strategy is None or self._reranking_strategy.name != strategy_name:
            params = self._config.reranking_params.get(strategy_name, {})
            self._reranking_strategy = reranking_registry.create(strategy_name, **params)
            logger.info(f"Using reranking strategy: {strategy_name}")
        return self._reranking_strategy

    def get_causal_strategy(self, name: Optional[str] = None) -> CausalLinkStrategy:
        """Get or create a causal link strategy instance."""
        strategy_name = name or self._config.causal_strategy
        if self._causal_strategy is None or self._causal_strategy.name != strategy_name:
            params = self._config.causal_params.get(strategy_name, {})
            self._causal_strategy = causal_registry.create(strategy_name, **params)
            logger.info(f"Using causal strategy: {strategy_name}")
        return self._causal_strategy

    async def ensure_reranking_initialized(self) -> None:
        """Ensure the reranking strategy is initialized."""
        if self._reranking_strategy is None:
            self.get_reranking_strategy()
        await self._reranking_strategy.ensure_initialized()

    def list_available_strategies(self) -> Dict[str, List[str]]:
        """List all available strategies by category."""
        return {
            "retrieval": retrieval_registry.list(),
            "graph_retrieval": graph_retrieval_registry.list(),
            "fusion": fusion_registry.list(),
            "reranking": reranking_registry.list(),
            "causal": causal_registry.list(),
        }

    def override_strategy(self, category: str, name: str, **kwargs) -> None:
        """
        Override a strategy temporarily.

        Args:
            category: One of 'retrieval', 'graph_retrieval', 'fusion', 'reranking'
            name: Strategy name
            **kwargs: Additional parameters for the strategy
        """
        if category == "retrieval":
            self._retrieval_strategy = retrieval_registry.create(name, **kwargs)
            self._config.retrieval_strategy = name
        elif category == "graph_retrieval":
            self._graph_retrieval_strategy = graph_retrieval_registry.create(name, **kwargs)
            self._config.graph_retrieval_strategy = name
        elif category == "fusion":
            self._fusion_strategy = fusion_registry.create(name, **kwargs)
            self._config.fusion_strategy = name
        elif category == "reranking":
            self._reranking_strategy = reranking_registry.create(name, **kwargs)
            self._config.reranking_strategy = name
        elif category == "causal":
            self._causal_strategy = causal_registry.create(name, **kwargs)
            self._config.causal_strategy = name
        else:
            raise ValueError(f"Unknown strategy category: {category}")
        logger.info(f"Overridden {category} strategy: {name}")


class ModularRetrievalPipeline:
    """
    Modular retrieval pipeline that uses configurable strategies.

    This class provides a high-level interface for running the full retrieval
    pipeline with different strategies selected via configuration.
    """

    def __init__(self, module_manager: Optional[SearchModuleManager] = None):
        self._module_manager = module_manager or SearchModuleManager()

    @property
    def module_manager(self) -> SearchModuleManager:
        return self._module_manager

    async def retrieve_semantic_bm25(
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
    ) -> Dict[str, List[RetrievalResult]]:
        """Retrieve using the configured semantic/BM25 strategy."""
        strategy = self._module_manager.get_retrieval_strategy("semantic_bm25")
        return await strategy.retrieve(
            conn=conn,
            query_embedding_str=query_embedding_str,
            query_text=query_text,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
        )

    async def retrieve_temporal(
        self,
        conn,
        query_embedding_str: str,
        query_text: str,
        bank_id: str,
        fact_types: List[str],
        limit: int,
        start_date: datetime,
        end_date: datetime,
        tags: List[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: List[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> Dict[str, List[RetrievalResult]]:
        """Retrieve using the configured temporal strategy."""
        strategy = self._module_manager.get_retrieval_strategy("temporal")
        return await strategy.retrieve(
            conn=conn,
            query_embedding_str=query_embedding_str,
            query_text=query_text,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
            start_date=start_date,
            end_date=end_date,
        )

    async def retrieve_graph(
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
    ) -> tuple[List[RetrievalResult], GraphRetrievalTimings | None]:
        """Retrieve using the configured graph strategy."""
        strategy = self._module_manager.get_graph_retrieval_strategy()
        return await strategy.retrieve(
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
        )

    def fuse_results(
        self,
        result_lists: List[List[RetrievalResult]],
        source_names: Optional[List[str]] = None,
        fusion_strategy: Optional[str] = None,
    ) -> List[MergedCandidate]:
        """Fuse results using the configured fusion strategy."""
        strategy = self._module_manager.get_fusion_strategy(fusion_strategy)
        return strategy.fuse(result_lists, source_names=source_names)

    async def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate],
        reranking_strategy: Optional[str] = None,
    ) -> List[ScoredResult]:
        """Rerank candidates using the configured reranking strategy."""
        strategy = self._module_manager.get_reranking_strategy(reranking_strategy)
        await strategy.ensure_initialized()
        return await strategy.rerank(query, candidates)


_default_module_manager = SearchModuleManager()
_default_pipeline = ModularRetrievalPipeline(_default_module_manager)


def get_module_manager() -> SearchModuleManager:
    """Get the default module manager."""
    return _default_module_manager


def get_pipeline() -> ModularRetrievalPipeline:
    """Get the default modular retrieval pipeline."""
    return _default_pipeline


def configure_search(config: SearchConfig) -> None:
    """Configure the default search pipeline with new settings."""
    _default_module_manager.set_config(config)


def override_strategy(category: str, name: str, **kwargs) -> None:
    """Override a strategy in the default pipeline."""
    _default_module_manager.override_strategy(category, name, **kwargs)
