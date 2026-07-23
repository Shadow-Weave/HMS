"""Typed, payload-free identities for durable video map checkpoints.

Only schema-validated provider output and bounded usage metadata cross this
boundary.  Normalized frame bytes and their data URLs are represented solely
by system-owned hashes, dimensions, and timestamps.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from pydantic import Field, model_validator

from .models import ModelTemporalSegment, ProviderResult, StrictModel, VisualEvidence

SEGMENT_CHECKPOINT_VERSION = "video-map-checkpoint-v1"
MAX_PROVIDER_TOKEN_COUNT = 2**63 - 1


class VideoSegmentIdentity(StrictModel):
    """Stable identity for one deterministic map input window."""

    segment_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_id: str = Field(min_length=1, max_length=160)
    evidence_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class VideoSegmentCheckpoint(VideoSegmentIdentity):
    """Durable representation of one locally validated provider result."""

    value: ModelTemporalSegment = Field(repr=False)
    provider: str = Field(min_length=1, max_length=128)
    configured_model: str = Field(min_length=1, max_length=256)
    resolved_model: str | None = Field(default=None, max_length=256)
    request_id: str | None = Field(default=None, max_length=256)
    input_tokens: int = Field(ge=0, le=MAX_PROVIDER_TOKEN_COUNT)
    output_tokens: int = Field(ge=0, le=MAX_PROVIDER_TOKEN_COUNT)
    logical_calls: int = Field(ge=1, le=MAX_PROVIDER_TOKEN_COUNT)
    physical_attempts: int = Field(ge=1, le=MAX_PROVIDER_TOKEN_COUNT)

    @model_validator(mode="after")
    def validate_segment_identity(self) -> "VideoSegmentCheckpoint":
        if self.value.segment_id != self.segment_id:
            raise ValueError("checkpoint segment ID does not match its validated value")
        if self.physical_attempts < self.logical_calls:
            raise ValueError("checkpoint physical attempts cannot be lower than logical calls")
        return self

    @classmethod
    def from_provider_result(
        cls,
        identity: VideoSegmentIdentity,
        result: ProviderResult[ModelTemporalSegment],
    ) -> "VideoSegmentCheckpoint":
        return cls(
            **identity.model_dump(),
            value=result.value,
            provider=result.provider,
            configured_model=result.configured_model,
            resolved_model=result.resolved_model,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            logical_calls=result.logical_calls,
            physical_attempts=result.physical_attempts,
        )

    def to_provider_result(self) -> ProviderResult[ModelTemporalSegment]:
        """Rehydrate provenance without pretending a cached call ran now."""

        return ProviderResult(
            value=self.value,
            provider=self.provider,
            configured_model=self.configured_model,
            resolved_model=self.resolved_model,
            request_id=self.request_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            logical_calls=self.logical_calls,
            physical_attempts=self.physical_attempts,
            latency_seconds=0.0,
        )


def derive_video_segment_identity(
    segment_id: str,
    evidence: Sequence[VisualEvidence],
) -> VideoSegmentIdentity:
    """Hash the complete system-owned map window without hashing media bytes."""

    if not segment_id or len(segment_id) > 160:
        raise ValueError("segment_id must be non-empty and at most 160 characters")
    if not evidence or len(evidence) > 64:
        raise ValueError("segment checkpoint evidence count must be between 1 and 64")
    evidence_envelope = []
    seen_ids: set[str] = set()
    for item in evidence:
        if item.timestamp_ms is None:
            raise ValueError("video segment evidence requires a system timestamp")
        if item.evidence_id in seen_ids:
            raise ValueError("video segment evidence IDs must be unique")
        seen_ids.add(item.evidence_id)
        evidence_envelope.append(
            {
                "evidence_id": item.evidence_id,
                "timestamp_ms": item.timestamp_ms,
                "sha256": item.sha256,
                "mime_type": item.mime_type,
                "width": item.width,
                "height": item.height,
            }
        )
    canonical_evidence = json.dumps(
        evidence_envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    evidence_fingerprint = hashlib.sha256(canonical_evidence.encode("utf-8")).hexdigest()
    key_envelope = json.dumps(
        {
            "version": SEGMENT_CHECKPOINT_VERSION,
            "segment_id": segment_id,
            "evidence_fingerprint": evidence_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return VideoSegmentIdentity(
        segment_key=hashlib.sha256(key_envelope.encode("utf-8")).hexdigest(),
        segment_id=segment_id,
        evidence_fingerprint=evidence_fingerprint,
    )
