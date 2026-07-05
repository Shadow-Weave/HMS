"""
Search module for memory retrieval.

Provides modular search architecture:
- Retrieval: 4-way parallel (semantic + BM25 + graph + temporal)
- Graph retrieval: Link expansion strategy
- Reranking: Pluggable strategies (heuristic, cross-encoder)

New modular interface for ablation experiments:
- Strategy registry system for easy swapping
- Unified module manager for configuration
- Default implementations maintain backward compatibility
"""

from .graph_retrieval import GraphRetriever
from .reranking import CrossEncoderReranker
from .retrieval import (
    ParallelRetrievalResult,
    get_default_graph_retriever,
    set_default_graph_retriever,
)

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

from .implementations import (
    RRFFusion,
    WeightedFusion,
    SimpleFusion,
    CrossEncoderReranking,
    PassthroughReranking,
    RRFOnlyReranking,
    SemanticBM25Retrieval,
    TemporalRetrieval,
    LinkExpansionGraphRetrieval,
)

from .module_manager import (
    SearchConfig,
    SearchModuleManager,
    ModularRetrievalPipeline,
    get_module_manager,
    get_pipeline,
    configure_search,
    override_strategy,
)

from .causal_strategies import (
    CausalLinkStrategy,
    CausalNeighbor,
    CausalContext,
    CausalScore,
    causal_registry,
)

from .causal_implementations import (
    MemoryLinksCausalStrategy,
    TemporalCausalStrategy,
)

from .causal_llm_strategy import (
    LLMDynamicCausalStrategy,
    HybridCausalStrategy,
)

from .causal_session_strategy import (
    SessionCausalStrategy,
    DocumentGraphCausalStrategy,
)

from .query_rewriting import (
    QueryRewritingStrategy,
    query_rewriting_registry,
    NoOpQueryRewriting,
    LLMBasedQueryRewriting,
)

__all__ = [
    # Legacy exports (backward compatibility)
    "get_default_graph_retriever",
    "set_default_graph_retriever",
    "ParallelRetrievalResult",
    "GraphRetriever",
    "CrossEncoderReranker",
    # Strategy interfaces
    "RetrievalStrategy",
    "GraphRetrievalStrategy",
    "FusionStrategy",
    "RerankingStrategy",
    # Registries
    "retrieval_registry",
    "graph_retrieval_registry",
    "fusion_registry",
    "reranking_registry",
    # Default implementations
    "SemanticBM25Retrieval",
    "TemporalRetrieval",
    "LinkExpansionGraphRetrieval",
    "RRFFusion",
    "WeightedFusion",
    "SimpleFusion",
    "CrossEncoderReranking",
    "PassthroughReranking",
    "RRFOnlyReranking",
    # Module manager
    "SearchConfig",
    "SearchModuleManager",
    "ModularRetrievalPipeline",
    "get_module_manager",
    "get_pipeline",
    "configure_search",
    "override_strategy",
    # Causal link strategies
    "CausalLinkStrategy",
    "CausalNeighbor",
    "CausalContext",
    "CausalScore",
    "causal_registry",
    "MemoryLinksCausalStrategy",
    "TemporalCausalStrategy",
    "LLMDynamicCausalStrategy",
    "HybridCausalStrategy",
    "SessionCausalStrategy",
    "DocumentGraphCausalStrategy",
    # Query rewriting strategies
    "QueryRewritingStrategy",
    "query_rewriting_registry",
    "NoOpQueryRewriting",
    "LLMBasedQueryRewriting",
]
