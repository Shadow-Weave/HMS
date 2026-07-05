"""
时间约束的提取
"""

import logging
from datetime import datetime

from hms_api.engine.query_analyzer import DateparserQueryAnalyzer, QueryAnalyzer

logger = logging.getLogger(__name__)

# Global default analyzer instance
# Can be overridden by passing a custom analyzer to extract_temporal_constraint
_default_analyzer: QueryAnalyzer | None = None


def get_default_analyzer() -> QueryAnalyzer:
    """
    Get or create the default query analyzer.

    Uses lazy initialization to avoid loading at import time.

    Returns:
        Default DateparserQueryAnalyzer instance
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = DateparserQueryAnalyzer()
    return _default_analyzer


def extract_temporal_constraint(
    query: str,
    reference_date: datetime | None = None,
    analyzer: QueryAnalyzer | None = None,
) -> tuple[datetime, datetime] | None:
    """
    Extract temporal constraint from query.

    Returns (start_date, end_date) tuple if temporal constraint found, else None.

    Args:
        query: Search query
        reference_date: Reference date for relative terms (defaults to now)
        analyzer: Custom query analyzer (defaults to DateparserQueryAnalyzer)

    Returns:
        (start_date, end_date) tuple or None
    """
    if analyzer is None:
        analyzer = get_default_analyzer()

    analysis = analyzer.analyze(query, reference_date)

    if analysis.temporal_constraint:
        result = (analysis.temporal_constraint.start_date, analysis.temporal_constraint.end_date)
        return result

    return None
