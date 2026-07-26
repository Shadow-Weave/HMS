"""Provider-neutral extraction contracts and strategies."""

from .extractor import FactExtractorAdapter
from .layout import (
    PrechunkedExtractionLayout,
    PrechunkedLayoutError,
    build_prechunked_extraction_layout,
)
from .models import FACT_KEY_VERSION, CausalFactRelation, FactCandidate, compute_fact_key
from .passthrough import (
    SECONDS_PER_FACT,
    build_content_position_map,
    extract_passthrough,
    to_chunk_metadata,
    to_chunk_metadata_batch,
)
from .ports import (
    BatchExtractionUnsupportedError,
    ChunkFactCount,
    ExtractionAdapterError,
    ExtractionContractError,
    ExtractionMode,
    ExtractionModeMismatchError,
    ExtractionPolicy,
    ExtractionRequest,
    ExtractionResult,
    FactExtractor,
)

__all__ = [
    "FACT_KEY_VERSION",
    "SECONDS_PER_FACT",
    "BatchExtractionUnsupportedError",
    "CausalFactRelation",
    "ChunkFactCount",
    "ExtractionAdapterError",
    "ExtractionContractError",
    "ExtractionMode",
    "ExtractionModeMismatchError",
    "ExtractionPolicy",
    "ExtractionRequest",
    "ExtractionResult",
    "FactCandidate",
    "FactExtractor",
    "FactExtractorAdapter",
    "PrechunkedExtractionLayout",
    "PrechunkedLayoutError",
    "build_content_position_map",
    "build_prechunked_extraction_layout",
    "compute_fact_key",
    "extract_passthrough",
    "to_chunk_metadata",
    "to_chunk_metadata_batch",
]
