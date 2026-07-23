"""Typed contracts for media evidence, grounded model output, and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import GroundingError

MediaKind = Literal["image", "video"]
Uncertainty = Literal["low", "medium", "high"]


class StrictModel(BaseModel):
    """Base model used for OpenAI Structured Outputs schemas."""

    model_config = ConfigDict(extra="forbid")


class MediaAsset(StrictModel):
    """System-owned identity and decoded metadata for one uploaded asset."""

    asset_id: str = Field(min_length=1, max_length=160)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_kind: MediaKind
    detected_mime: str = Field(min_length=1, max_length=128)
    original_filename: str = Field(min_length=1, max_length=1024)
    byte_size: int = Field(ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_ms: int | None = Field(default=None, ge=0)
    audio_presence: Literal["absent", "present", "unknown"]
    audio_processing: Literal["not_requested", "not_configured", "completed", "failed"]


class VisualEvidence(StrictModel):
    """One normalized image sent to the provider.

    ``encoded_bytes`` is excluded from repr and every Pydantic serialization
    surface.  It only exists long enough to construct the provider data URL.
    """

    evidence_id: str = Field(min_length=1, max_length=160)
    timestamp_ms: int | None = Field(default=None, ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    encoded_bytes: bytes = Field(exclude=True, repr=False)


def _validate_evidence_ids(value: list[str]) -> list[str]:
    if not value:
        raise ValueError("at least one evidence ID is required")
    if any(not item or len(item) > 160 for item in value):
        raise ValueError("evidence IDs must be non-empty and at most 160 characters")
    if len(value) != len(set(value)):
        raise ValueError("evidence IDs must be unique")
    return value


class GroundedStatement(StrictModel):
    text: str = Field(min_length=1, max_length=16_384)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)
    uncertainty: Uncertainty

    _evidence_ids = field_validator("evidence_ids")(_validate_evidence_ids)


class GroundedEntity(StrictModel):
    name: str = Field(min_length=1, max_length=1024)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)
    uncertainty: Uncertainty

    _evidence_ids = field_validator("evidence_ids")(_validate_evidence_ids)


class VisibleText(GroundedStatement):
    """OCR, code, terminal output, or other visible text."""


class VisualObservation(GroundedStatement):
    kind: Literal["object", "action", "event", "ui", "code", "diagram", "spatial"]


class ModelTemporalSegment(StrictModel):
    """Model output for a system-declared time window.

    The model returns a segment ID, never millisecond values.  The latter are
    injected from the system's frame timeline after grounding validation.
    """

    segment_id: str = Field(min_length=1, max_length=160)
    summary: list[GroundedStatement] = Field(max_length=64)
    observations: list[VisualObservation] = Field(max_length=256)
    visible_text: list[VisibleText] = Field(max_length=256)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)

    _evidence_ids = field_validator("evidence_ids")(_validate_evidence_ids)

    @model_validator(mode="after")
    def require_semantics(self) -> "ModelTemporalSegment":
        if not (self.summary or self.observations or self.visible_text):
            raise ValueError("a temporal segment must contain at least one semantic statement")
        return self


class ModelMultimodalDescription(StrictModel):
    """All semantic fields returned by the vision model are evidence-bound."""

    summary: list[GroundedStatement] = Field(min_length=1, max_length=64)
    entities: list[GroundedEntity] = Field(max_length=256)
    observations: list[VisualObservation] = Field(max_length=512)
    visible_text: list[VisibleText] = Field(max_length=512)
    temporal_segments: list[ModelTemporalSegment] = Field(max_length=256)
    limitations: list[GroundedStatement] = Field(max_length=64)


class SegmentWindow(StrictModel):
    """System-owned mapping between one segment and its admissible evidence."""

    segment_id: str = Field(min_length=1, max_length=160)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)

    _evidence_ids = field_validator("evidence_ids")(_validate_evidence_ids)

    @model_validator(mode="after")
    def validate_range(self) -> "SegmentWindow":
        if self.end_ms < self.start_ms:
            raise ValueError("segment end_ms must be greater than or equal to start_ms")
        return self


class SystemProvenance(StrictModel):
    """System-generated descriptor and pipeline identity."""

    provider: str = Field(min_length=1, max_length=128)
    configured_model: str = Field(min_length=1, max_length=256)
    resolved_model: str | None = Field(default=None, max_length=256)
    pipeline_version: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    schema_version: str = Field(min_length=1, max_length=128)
    sampling_version: str = Field(min_length=1, max_length=128)
    pipeline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_request_id: str | None = Field(default=None, max_length=256)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    logical_calls: int = Field(default=0, ge=0)
    physical_attempts: int = Field(default=0, ge=0)


class NormalizedTemporalSegment(StrictModel):
    segment_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    description: ModelTemporalSegment


class NormalizedMultimodalDescription(StrictModel):
    """Validated model output combined with system-controlled provenance."""

    schema_version: str
    media_kind: MediaKind
    asset: MediaAsset
    model_output: ModelMultimodalDescription
    temporal_segments: list[NormalizedTemporalSegment]
    provenance: SystemProvenance


class MultimodalCapabilities(StrictModel):
    image_input: bool
    video_input: bool
    structured_outputs: bool
    accepted_image_mimes: list[str]
    image_detail_levels: list[Literal["low", "high", "auto"]]


T = TypeVar("T")


@dataclass(frozen=True)
class ProviderResult(Generic[T]):
    """Sanitized provider result; request payloads are deliberately absent."""

    value: T
    provider: str
    configured_model: str
    resolved_model: str | None
    request_id: str | None
    input_tokens: int
    output_tokens: int
    logical_calls: int
    physical_attempts: int
    latency_seconds: float


def _iter_grounded(description: ModelMultimodalDescription):
    yield from description.summary
    yield from description.entities
    yield from description.observations
    yield from description.visible_text
    yield from description.limitations
    for segment in description.temporal_segments:
        yield from segment.summary
        yield from segment.observations
        yield from segment.visible_text


def _iter_segment_grounded(segment: ModelTemporalSegment):
    yield from segment.summary
    yield from segment.observations
    yield from segment.visible_text


def normalize_description(
    *,
    media_kind: MediaKind,
    asset: MediaAsset,
    description: ModelMultimodalDescription,
    evidence: list[VisualEvidence],
    segment_windows: list[SegmentWindow],
    provenance: SystemProvenance,
) -> NormalizedMultimodalDescription:
    """Validate grounding and inject system-owned time ranges.

    Raises only sanitized ``GroundingError`` instances for semantic contract
    failures.  The exception never includes model text or evidence bytes.
    """

    if asset.media_kind != media_kind:
        raise GroundingError("grounding.media_kind", "Asset kind does not match the requested normalization path")

    known_evidence = {item.evidence_id for item in evidence}
    if not known_evidence:
        raise GroundingError("grounding.no_evidence", "No visual evidence was supplied")
    if len(known_evidence) != len(evidence):
        raise GroundingError("grounding.duplicate_evidence", "Visual evidence IDs must be unique")

    if media_kind == "image":
        if any(item.timestamp_ms is not None for item in evidence):
            raise GroundingError("grounding.image_timestamp", "Image evidence cannot carry video timestamps")
    else:
        duration_ms = asset.duration_ms
        if duration_ms is None:
            raise GroundingError("grounding.video_duration", "Video assets require a system duration")
        if any(item.timestamp_ms is None or item.timestamp_ms > duration_ms for item in evidence):
            raise GroundingError(
                "grounding.video_timestamp", "Video evidence timestamps must be present and within the asset duration"
            )

    for statement in _iter_grounded(description):
        if not set(statement.evidence_ids).issubset(known_evidence):
            raise GroundingError("grounding.unknown_evidence", "Model output referenced unknown visual evidence")

    windows = {window.segment_id: window for window in segment_windows}
    if len(windows) != len(segment_windows):
        raise GroundingError("grounding.duplicate_segment", "System segment IDs must be unique")

    normalized_segments: list[NormalizedTemporalSegment] = []
    if media_kind == "image":
        if description.temporal_segments or segment_windows:
            raise GroundingError("grounding.image_has_timeline", "Image descriptions cannot contain temporal segments")
    else:
        if not description.temporal_segments:
            raise GroundingError("grounding.video_missing_timeline", "Video descriptions require temporal segments")
        if set(windows) != {segment.segment_id for segment in description.temporal_segments}:
            raise GroundingError(
                "grounding.segment_coverage", "Every system-declared video segment must have exactly one model output"
            )
        evidence_by_id = {item.evidence_id: item for item in evidence}
        seen_segments: set[str] = set()
        last_start = -1
        for segment in description.temporal_segments:
            window = windows.get(segment.segment_id)
            if window is None:
                raise GroundingError("grounding.unknown_segment", "Model output referenced an unknown segment")
            if segment.segment_id in seen_segments:
                raise GroundingError("grounding.duplicate_segment", "Model output repeated a segment")
            if not set(segment.evidence_ids).issubset(set(window.evidence_ids)):
                raise GroundingError(
                    "grounding.segment_evidence_mismatch",
                    "Segment output referenced evidence outside its system window",
                )
            window_evidence = set(window.evidence_ids)
            if not window_evidence.issubset(known_evidence):
                raise GroundingError(
                    "grounding.window_unknown_evidence", "System segment window referenced unknown visual evidence"
                )
            for statement in _iter_segment_grounded(segment):
                if not set(statement.evidence_ids).issubset(window_evidence):
                    raise GroundingError(
                        "grounding.segment_statement_mismatch",
                        "Segment statement referenced evidence outside its system window",
                    )
            for evidence_id in window.evidence_ids:
                timestamp_ms = evidence_by_id[evidence_id].timestamp_ms
                if timestamp_ms is None or not window.start_ms <= timestamp_ms <= window.end_ms:
                    raise GroundingError(
                        "grounding.window_timestamp_mismatch",
                        "System segment evidence timestamp falls outside its declared window",
                    )
            if window.end_ms > (asset.duration_ms or 0):
                raise GroundingError("grounding.segment_out_of_range", "System segment exceeds the asset duration")
            if window.start_ms < last_start:
                raise GroundingError("grounding.segment_order", "Temporal segments must be ordered")
            seen_segments.add(segment.segment_id)
            last_start = window.start_ms
            normalized_segments.append(
                NormalizedTemporalSegment(
                    segment_id=segment.segment_id,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    description=segment,
                )
            )

    return NormalizedMultimodalDescription(
        schema_version=provenance.schema_version,
        media_kind=media_kind,
        asset=asset,
        model_output=description,
        temporal_segments=normalized_segments,
        provenance=provenance,
    )
