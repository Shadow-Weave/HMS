"""
Modular query rewriting strategies for alias expansion.

Provides abstract base classes and registration mechanisms for query rewriting
strategies that expand abstract concepts into specific subcategories.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class QueryRewritingStrategy(ABC):
    """
    Abstract base class for query rewriting strategies.

    Implementations take a user query and expand abstract concepts into
    more specific subcategories for improved retrieval coverage.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this rewriting strategy."""
        pass

    @abstractmethod
    async def rewrite(
        self,
        query_text: str,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Rewrite query with alias expansion.

        Args:
            query_text: Original user query text
            llm: Optional LLM provider for LLM-based rewriting
            **kwargs: Additional context (bank_id, fact_types, etc.)

        Returns:
            Dict mapping original query -> list of expanded aliases/subcategories
            Example: {"我上周买了多少件家用电器": ["厨房电器", "卧室电器", "小型家用电器"]}
        """
        pass

    def should_expand(self, query_text: str) -> bool:
        """
        Determine if the query requires alias expansion.

        Default implementation returns True. Override to add heuristics
        for skipping simple queries.

        Args:
            query_text: Query to evaluate

        Returns:
            True if expansion is recommended, False otherwise
        """
        abstract_patterns = [
            r"\b(how many|how much)\b",
            r"\b(items?|things?|products?|activities|expenses|costs)\b",
            r"\b(total|in total|combined)\b",
            r"\b(different types?|kinds?)\b",
            r"\b(household|office|kitchen|bathroom)\s+(items|things|furniture|appliances)\b",
            r"\b(electronic devices|household items|furniture pieces|clothing items)\b",
            r"\b(tools|equipment|gadgets|devices|appliances)\b",
            r"\b(workshops?|classes?|events?|festivals?)\b",
            r"\b(plants?|fish|birds|pets|animals)\b",
            r"\b(pieces?|kinds?|types?|varieties)\b",
            r"\b(charity|concert|musical)\b",
            r"\b(art-related|art related)\b",
            r"\b(furniture)\b",
            r"\b(conferences?|lectures?)\b",
            r"\b(doctors?|medications?|prescriptions)\b",
            r"\b(albums?|eps?|songs?|tracks)\b",
            r"\b(weddings?|dinner parties?|parties?)\b",
            r"\b(breaks?|vacations?|trips?)\b",
            r"\b(baking|cooking)\b",
            r"\b(games?|gaming)\b",
        ]

        query_lower = query_text.lower()
        for pattern in abstract_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        abstract_keywords = [
            "家用电器", "电子产品", "服装", "食品", "饮料", "家具", "电器",
            "设备", "工具", "材料", "产品", "物品",
            "items", "things", "products", "activities", "expenses", "costs",
            "devices", "equipment", "gadgets", "appliances", "tools",
            "electronics", "foods", "drinks", "plants", "fish",
            "events", "festivals", "workshops", "classes", "conferences",
            "charity", "concert", "art-related", "weddings", "dinner parties",
            "doctors", "medications", "albums", "eps", "songs", "baking",
            "games", "gaming", "breaks", "vacations", "trips",
        ]
        return any(kw in query_lower for kw in abstract_keywords)


class StrategyRegistry:
    """Generic registry for strategy classes."""

    def __init__(self):
        self._strategies: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
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


query_rewriting_registry = StrategyRegistry()