"""
Default implementations of query rewriting strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .prompts import (
    build_alias_expansion_messages,
    parse_alias_expansion_response,
)
from .strategies import QueryRewritingStrategy, query_rewriting_registry

logger = logging.getLogger(__name__)


DEFAULT_CONFIDENCE_THRESHOLD = 0.6
MIN_ALIAS_LENGTH = 2
MAX_ALIAS_LENGTH_EN = 30
MAX_ALIAS_LENGTH_CN = 10


def _detect_language(query: str) -> str:
    """Detect query language (en or cn)."""
    chinese_chars = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    return "cn" if chinese_chars > len(query) * 0.3 else "en"


def _compute_alias_confidence(
    alias: str,
    original_query: str,
    query_language: str,
) -> float:
    """
    Compute confidence score for a single alias.

    Args:
        alias: The alias candidate to score
        original_query: Original user query
        query_language: Detected language ('en' or 'cn')

    Returns:
        Confidence score between 0 and 1
    """
    confidence = 1.0

    max_len = MAX_ALIAS_LENGTH_EN if query_language == "en" else MAX_ALIAS_LENGTH_CN
    if len(alias) > max_len:
        confidence -= 0.2 * (len(alias) - max_len) / max_len

    if len(alias) < MIN_ALIAS_LENGTH:
        confidence -= 0.3

    if alias.lower() in original_query.lower():
        confidence += 0.1

    common_generic_terms = ["things", "items", "stuff", "miscellaneous", "others", "其他", "东西", "物品"]
    if any(term in alias.lower() for term in common_generic_terms):
        confidence -= 0.3

    return max(0.0, min(1.0, confidence))


def _filter_aliases_by_confidence(
    aliases: List[str],
    original_query: str,
    query_language: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> List[str]:
    """
    Filter aliases by confidence threshold.

    Args:
        aliases: List of alias candidates
        original_query: Original user query
        query_language: Detected language
        threshold: Minimum confidence score to keep an alias

    Returns:
        Filtered list of aliases
    """
    scored_aliases: List[Tuple[str, float]] = [
        (alias, _compute_alias_confidence(alias, original_query, query_language))
        for alias in aliases
    ]

    scored_aliases.sort(key=lambda x: x[1], reverse=True)

    filtered = [
        alias for alias, score in scored_aliases
        if score >= threshold
    ]

    if len(filtered) < 3 and len(scored_aliases) >= 3:
        filtered = [alias for alias, _ in scored_aliases[:3]]

    return filtered[:10]


@query_rewriting_registry.register("noop")
class NoOpQueryRewriting(QueryRewritingStrategy):
    """
    No-op query rewriting strategy.

    Returns the original query without any modifications.
    Used when alias expansion is disabled or not needed.
    """

    @property
    def name(self) -> str:
        return "noop"

    async def rewrite(
        self,
        query_text: str,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """Return original query as-is without expansion."""
        return {query_text: []}

    def should_expand(self, query_text: str) -> bool:
        """Never expand - this is the no-op strategy."""
        return False


@query_rewriting_registry.register("llm_based")
class LLMBasedQueryRewriting(QueryRewritingStrategy):
    """
    LLM-based query rewriting strategy with confidence-based filtering.

    Uses an LLM to expand abstract concepts into specific subcategories
    for improved retrieval coverage, with quality scoring to filter low-quality aliases.
    """

    def __init__(
        self,
        min_query_length: int = 5,
        max_retries: int = 2,
        cache_enabled: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize LLM-based query rewriting.

        Args:
            min_query_length: Minimum query length to consider for expansion
            max_retries: Maximum retry attempts on LLM failure
            cache_enabled: Whether to cache results per query
            confidence_threshold: Minimum confidence score to keep an alias (0-1)
        """
        self._min_query_length = min_query_length
        self._max_retries = max_retries
        self._cache: Dict[str, List[str]] = {}
        self._cache_enabled = cache_enabled
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "llm_based"

    async def rewrite(
        self,
        query_text: str,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Rewrite query using LLM for alias expansion with confidence filtering.

        Args:
            query_text: Original user query
            llm: LLM provider instance
            **kwargs: Additional context (not used currently)

        Returns:
            Dict mapping original query -> expanded aliases
        """
        if len(query_text) < self._min_query_length:
            return {query_text: []}

        if not self.should_expand(query_text):
            return {query_text: []}

        if self._cache_enabled and query_text in self._cache:
            return {query_text: self._cache[query_text]}

        if llm is None:
            logger.warning("LLM not available, returning original query")
            return {query_text: []}

        messages = build_alias_expansion_messages(query_text)
        query_language = _detect_language(query_text)

        for attempt in range(self._max_retries):
            try:
                response = await llm.call(
                    messages=messages,
                    temperature=0.3,
                    max_completion_tokens=256,
                )

                aliases = parse_alias_expansion_response(response)

                if aliases:
                    filtered_aliases = _filter_aliases_by_confidence(
                        aliases,
                        query_text,
                        query_language,
                        threshold=self._confidence_threshold,
                    )

                    if filtered_aliases:
                        if self._cache_enabled:
                            self._cache[query_text] = filtered_aliases
                        return {query_text: filtered_aliases}

                logger.debug(f"No aliases generated for query: {query_text}")

            except Exception as e:
                logger.warning(f"LLM alias expansion attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries - 1:
                    return {query_text: []}

        return {query_text: []}

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
