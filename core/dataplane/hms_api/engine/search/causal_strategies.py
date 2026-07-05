"""
Modular causal relationship strategies for search retrieval.

Provides abstract interfaces and implementations for causal link discovery and scoring,
enabling easy swapping of causal reasoning logic for ablation experiments.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .tags import TagGroup, TagsMatch
from .types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CausalNeighbor:
    """
    Represents a causal neighbor node with associated metadata.

    Attributes:
        neighbor_id: ID of the causally related node
        link_weight: Weight of the causal link (0.0 to 1.0)
        link_type: Type of causal relationship ('caused_by', 'causes', 'enables', 'prevents')
        provenance: Optional metadata about how this causal link was determined
    """

    neighbor_id: str
    link_weight: float = 1.0
    link_type: str = "caused_by"
    provenance: Optional[Dict[str, Any]] = None


@dataclass
class CausalContext:
    """
    Context information for causal score computation.

    Attributes:
        parent_score: The temporal/propagation score of the parent node
        parent_date: Event date of the parent node (if available)
        current_date: Event date of the current node being scored (if available)
        distance: Graph distance from the original seed (hops)
    """

    parent_score: float
    parent_date: Optional[datetime] = None
    current_date: Optional[datetime] = None
    distance: int = 1


@dataclass
class CausalScore:
    """
    Result of causal score computation.

    Attributes:
        base_score: The computed causal score
        components: Breakdown of score components for transparency
        boosted_score: Final score after applying causal boost
    """

    base_score: float
    components: Dict[str, float] = field(default_factory=dict)
    boosted_score: Optional[float] = None


class CausalLinkStrategy(ABC):
    """
    Abstract base class for causal link discovery strategies.

    Implementations provide methods to:
    1. Discover which nodes are causally related to given seed nodes
    2. Compute causal propagation scores

    This abstraction allows for different causal reasoning approaches:
    - Pre-computed (from memory_links table)
    - Dynamic LLM-based extraction
    - Hybrid approaches
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this causal strategy."""
        pass

    @abstractmethod
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
        Find causally related neighbors for given seed nodes.

        Args:
            conn: Database connection
            seed_ids: List of seed node IDs to find causal neighbors for
            bank_id: Memory bank identifier
            fact_type: Fact type to filter results
            tags: Optional visibility tags
            tags_match: Tag matching mode
            tag_groups: Compound tag filter groups

        Returns:
            Dict mapping seed_id -> list of CausalNeighbor objects
        """
        pass

    def compute_causal_score(
        self,
        parent_score: float,
        link_weight: float,
        context: Optional[CausalContext] = None,
        **kwargs,
    ) -> CausalScore:
        """
        Compute causal propagation score.

        This method applies the causal boost formula using the link weight
        and parent score. Override for custom scoring logic.

        Default formula:
            causal_score = parent_score * link_weight * causal_boost_factor * decay

        Args:
            parent_score: Score from the parent node
            link_weight: Weight of the causal link (0.0 to 1.0)
            context: Optional context for score computation
            **kwargs: Additional parameters

        Returns:
            CausalScore object with computed scores
        """
        causal_boost_factor = self._get_causal_boost_factor(link_weight)
        decay_factor = self._get_decay_factor(context)

        base_score = parent_score * link_weight
        boosted_score = base_score * causal_boost_factor * decay_factor

        return CausalScore(
            base_score=base_score,
            components={
                "parent_score": parent_score,
                "link_weight": link_weight,
                "causal_boost_factor": causal_boost_factor,
                "decay_factor": decay_factor,
            },
            boosted_score=boosted_score,
        )

    def _get_causal_boost_factor(self, link_weight: float) -> float:
        """
        Get the causal boost factor based on link type/weight.

        Default implementation uses fixed factors:
        - causes/caused_by: 2.0
        - enables/prevents: 1.5
        - others: 1.0

        Override for custom boost logic.
        """
        return 2.0

    def _get_decay_factor(self, context: Optional[CausalContext]) -> float:
        """
        Get decay factor based on propagation distance or other context.

        Default: 0.7 for each hop
        Override for custom decay logic.
        """
        if context is None:
            return 0.7
        return 0.7 if context.distance > 0 else 1.0


class CausalLinkRegistry:
    """Registry for causal link strategies."""

    def __init__(self):
        self._strategies: Dict[str, type] = {}

    def register(self, name: str) -> callable:
        """Decorator to register a causal strategy class."""

        def decorator(cls: type) -> type:
            if name in self._strategies:
                logger.warning(f"Causal strategy '{name}' already registered, overwriting")
            self._strategies[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Optional[type]:
        """Get a registered strategy class by name."""
        return self._strategies.get(name)

    def list(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def create(self, name: str, **kwargs) -> CausalLinkStrategy:
        """Create an instance of a registered strategy."""
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Unknown causal strategy: {name}")
        return cls(**kwargs)


causal_registry = CausalLinkRegistry()
