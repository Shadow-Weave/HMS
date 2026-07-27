"""Semantic conversation segmentation for Retain chunk planning.

This package is intentionally independent from the Retain application service.
It exposes an asynchronous planner that can be integrated between normalized
document planning and ``ChunkPlan`` construction without changing extraction or
persistence contracts.
"""

from .adapters import build_chunk_plans_from_segmentation
from .models import (
    SEMANTIC_POLICY_VERSION,
    SEMANTIC_PROMPT_VERSION,
    BoundaryResponse,
    ConversationExchange,
    EffectiveSegmentationStrategy,
    MaterializedSegment,
    ParsedConversation,
    SegmentationFailurePolicy,
    SegmentationManifest,
    SegmentationMode,
    SegmentationResult,
    SegmentManifestEntry,
    SemanticSegmentationPolicy,
)
from .planner import (
    SegmentationReuseError,
    SemanticBoundaryValidationError,
    SemanticSegmentationError,
    SemanticSegmenter,
    UnsplittableExchangeError,
    materialize_semantic_boundaries,
    parse_conversation,
    validate_boundary_response,
)

__all__ = [
    "BoundaryResponse",
    "ConversationExchange",
    "EffectiveSegmentationStrategy",
    "MaterializedSegment",
    "ParsedConversation",
    "SEMANTIC_POLICY_VERSION",
    "SEMANTIC_PROMPT_VERSION",
    "SegmentManifestEntry",
    "SegmentationFailurePolicy",
    "SegmentationManifest",
    "SegmentationMode",
    "SegmentationResult",
    "SegmentationReuseError",
    "SemanticBoundaryValidationError",
    "SemanticSegmentationError",
    "SemanticSegmentationPolicy",
    "SemanticSegmenter",
    "UnsplittableExchangeError",
    "build_chunk_plans_from_segmentation",
    "materialize_semantic_boundaries",
    "parse_conversation",
    "validate_boundary_response",
]
