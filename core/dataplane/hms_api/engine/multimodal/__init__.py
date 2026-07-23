"""Lazy public surface for multimodal media normalization.

The package is referenced by a small admission-time security guard even when
the feature is disabled.  Eagerly importing all exports here would therefore
load Pillow, HTTP transport code, and the optional PyAV decoder on the legacy
text path.  PEP 562 lazy attributes preserve the convenient public imports
without paying that dependency or startup cost until multimodal code is used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    # durable video-map checkpoint contracts
    "SEGMENT_CHECKPOINT_VERSION": "checkpoints",
    "VideoSegmentCheckpoint": "checkpoints",
    "VideoSegmentIdentity": "checkpoints",
    "derive_video_segment_identity": "checkpoints",
    # errors
    "AnimatedImageNotSupportedError": "errors",
    "GroundingError": "errors",
    "MediaBudgetExceededError": "errors",
    "MediaValidationError": "errors",
    "MultimodalError": "errors",
    "ProviderAuthenticationError": "errors",
    "ProviderError": "errors",
    "ProviderIncompleteError": "errors",
    "ProviderRateLimitError": "errors",
    "ProviderRefusalError": "errors",
    "ProviderSchemaError": "errors",
    "ProviderUnavailableError": "errors",
    "VideoDecodeError": "errors",
    "VideoDecoderUnavailableError": "errors",
    # images
    "ImageNormalizationConfig": "images",
    "NormalizedImage": "images",
    "detect_image_mime": "images",
    "image_normalizer_identity": "images",
    "normalize_image": "images",
    # grounded models
    "GroundedEntity": "models",
    "GroundedStatement": "models",
    "MediaAsset": "models",
    "ModelMultimodalDescription": "models",
    "ModelTemporalSegment": "models",
    "MultimodalCapabilities": "models",
    "NormalizedMultimodalDescription": "models",
    "NormalizedTemporalSegment": "models",
    "ProviderResult": "models",
    "SegmentWindow": "models",
    "SystemProvenance": "models",
    "VisibleText": "models",
    "VisualEvidence": "models",
    "VisualObservation": "models",
    "normalize_description": "models",
    # provider
    "FakeMultimodalProvider": "provider",
    "MultimodalDescriptionProvider": "provider",
    "OpenAIProviderConfig": "provider",
    "OpenAIResponsesMultimodalProvider": "provider",
    "ProviderBudget": "provider",
    "estimate_provider_budget": "provider",
    # canonical serialization
    "flat_provenance_metadata": "serialization",
    "format_timecode": "serialization",
    "render_canonical_markdown": "serialization",
    # optional local video stack
    "NormalizedVideo": "video",
    "SamplingCandidateDiagnostic": "video",
    "SelectedFrameDiagnostic": "video",
    "VideoFrameCandidate": "video",
    "VideoMagic": "video",
    "VideoProbe": "video",
    "VideoProcessingConfig": "video",
    "VideoSamplingDiagnostics": "video",
    "VideoSamplingResult": "video",
    "build_video_frame_candidate": "video",
    "decode_and_sample_video": "video",
    "detect_video_magic": "video",
    "normalize_rotation_degrees": "video",
    "require_video_decoder": "video",
    "sample_video_candidates": "video",
    "video_decoder_identity": "video",
    "video_decoder_available": "video",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
