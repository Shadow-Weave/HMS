"""
Modular strategy interfaces for search retrieval.

Provides abstract base classes and registration mechanisms for:
- Retrieval strategies (semantic, BM25, graph, temporal)
- Fusion strategies (RRF, etc.)
- Reranking strategies (cross-encoder, etc.)

This allows for easy swapping of algorithms for ablation experiments.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from .tags import TagGroup, TagsMatch
from .types import GraphRetrievalTimings, MergedCandidate, RetrievalResult, ScoredResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StrategyRegistry:
    """Generic registry for strategy classes."""

    def __init__(self):
        self._strategies: Dict[str, Type] = {}

    def register(self, name: str) -> callable:
        """Decorator to register a strategy class."""

        def decorator(cls: Type) -> Type:
            if name in self._strategies:
                logger.warning(f"Strategy '{name}' already registered, overwriting")
            self._strategies[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Optional[Type]:
        """Get a registered strategy class by name."""
        return self._strategies.get(name)

    def list(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a registered strategy."""
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Unknown strategy: {name}")
        return cls(**kwargs)


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.

    Implementations provide specific retrieval methods like semantic, BM25,
    graph, or temporal retrieval.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this retrieval strategy."""
        pass

    @abstractmethod
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
        Retrieve results for multiple fact types.

        Args:
            conn: Database connection
            query_embedding_str: Query embedding as string
            query_text: Query text
            bank_id: Memory bank ID
            fact_types: List of fact types to retrieve
            limit: Maximum results per fact type
            tags: Optional tags for filtering
            tags_match: How to match tags
            tag_groups: Compound boolean tag filter groups

        Returns:
            Dict mapping fact_type -> list of RetrievalResult
        """
        pass


class GraphRetrievalStrategy(ABC):
    """
    Abstract base class for graph retrieval strategies.

    This extends the base GraphRetriever interface with a registry-compatible pattern.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this graph retrieval strategy."""
        pass

    @abstractmethod
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
    ) -> tuple[List[RetrievalResult], GraphRetrievalTimings | None]:
        """
        Retrieve relevant facts via graph traversal.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding as string
            bank_id: Memory bank identifier
            fact_type: Fact type to filter
            budget: Maximum number of nodes to explore/return
            query_text: Original query text
            semantic_seeds: Pre-computed semantic entry points
            temporal_seeds: Pre-computed temporal entry points
            tags: Optional list of tags for visibility filtering

        Returns:
            Tuple of (List of RetrievalResult, optional timing info)
        """
        pass


class FusionStrategy(ABC):
    """
    Abstract base class for result fusion strategies.

    Implementations merge multiple ranked result lists into a single ranked list.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this fusion strategy."""
        pass

    @abstractmethod
    def fuse(
        self,
        result_lists: List[List[RetrievalResult]],
        source_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MergedCandidate]:
        """
        Merge multiple ranked result lists.

        Args:
            result_lists: List of result lists from different retrieval methods
            source_names: Names for each source list (defaults to semantic, bm25, graph, temporal)

        Returns:
            Merged list of MergedCandidate objects sorted by fused score
        """
        pass


class RerankingStrategy(ABC):
    """
    Abstract base class for reranking strategies.

    Implementations reorder candidates based on additional scoring signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this reranking strategy."""
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[MergedCandidate],
        **kwargs,
    ) -> List[ScoredResult]:
        """
        Rerank candidates using this strategy.

        Args:
            query: Search query
            candidates: Merged candidates from fusion

        Returns:
            List of ScoredResult objects sorted by score
        """
        pass

    @abstractmethod
    async def ensure_initialized(self) -> None:
        """Ensure the strategy is initialized (for lazy initialization)."""
        pass

    @property
    @abstractmethod
    def is_passthrough(self) -> bool:
        """Return True if this is a passthrough reranker (no actual reranking)."""
        pass


retrieval_registry = StrategyRegistry()
graph_retrieval_registry = StrategyRegistry()
fusion_registry = StrategyRegistry()
reranking_registry = StrategyRegistry()
