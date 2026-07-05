"""
Query rewriting module for enhancing retrieval with alias expansion.

This module provides modular query rewriting strategies that can be used
to expand abstract concepts into specific subcategories for improved BM25 retrieval.
"""

from .strategies import QueryRewritingStrategy, query_rewriting_registry
from .implementations import NoOpQueryRewriting, LLMBasedQueryRewriting
from .llm_driven import LLMDrivenQueryRewriting, QueryAnalysisResult

__all__ = [
    "QueryRewritingStrategy",
    "query_rewriting_registry",
    "NoOpQueryRewriting",
    "LLMBasedQueryRewriting",
    "LLMDrivenQueryRewriting",
    "QueryAnalysisResult",
]
