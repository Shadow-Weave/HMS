"""Immutable contracts for semantic conversation segmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...response_models import TokenUsage

POLICY_SCHEMA_VERSION = "retain-semantic-policy-v1"
MANIFEST_SCHEMA_VERSION = "retain-semantic-manifest-v1"
EXCHANGE_POLICY_VERSION = "user-exchange-v1"
MATERIALIZER_VERSION = "canonical-json-hard-split-v1"
SEMANTIC_POLICY_VERSION = "semantic-boundary-v1"
SEMANTIC_PROMPT_VERSION = "semantic-boundary-prompt-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


class SegmentationFailurePolicy(StrEnum):
    """Behavior when semantic planning cannot produce a valid plan."""

    FIXED_FALLBACK = "fixed_fallback"
    RAISE = "raise"


class SegmentationMode(StrEnum):
    """Caller-selected planning mode for one document item."""

    SEMANTIC = "semantic"
    FIXED_BYPASS = "fixed_bypass"


class EffectiveSegmentationStrategy(StrEnum):
    """The strategy that actually produced the materialized chunks."""

    PASSTHROUGH = "passthrough"
    SEMANTIC = "semantic"
    FIXED_FALLBACK = "fixed_fallback"
    FIXED_BYPASS = "fixed_bypass"


class BoundaryResponse(BaseModel):
    """Provider output containing boundaries and no generated content.

    Each value is the zero-based index of the final exchange in one semantic
    segment. Segment starts are derived deterministically from the preceding
    boundary, so the provider cannot rewrite, omit, or reorder source text.
    """

    model_config = ConfigDict(extra="forbid")

    end_exchange_indices: list[int] = Field(min_length=1)

    @field_validator("end_exchange_indices", mode="before")
    @classmethod
    def _strict_integer_indices(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise TypeError("end_exchange_indices must be a list")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in value):
            raise TypeError("end_exchange_indices must contain only integers")
        return value


@dataclass(frozen=True, slots=True)
class SemanticSegmentationPolicy:
    """Versioned behavioral policy for semantic boundary planning."""

    max_chars: int
    provider: str
    model: str
    version: str = SEMANTIC_POLICY_VERSION
    prompt_version: str = SEMANTIC_PROMPT_VERSION
    failure_policy: SegmentationFailurePolicy = SegmentationFailurePolicy.FIXED_FALLBACK
    max_completion_tokens: int = 1024
    max_retries: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int) or self.max_chars <= 2:
            raise ValueError("max_chars must be an integer greater than two")
        for field_name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("version", self.version),
            ("prompt_version", self.prompt_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.failure_policy, SegmentationFailurePolicy):
            raise TypeError("failure_policy must be a SegmentationFailurePolicy")
        if (
            isinstance(self.max_completion_tokens, bool)
            or not isinstance(self.max_completion_tokens, int)
            or self.max_completion_tokens <= 0
        ):
            raise ValueError("max_completion_tokens must be a positive integer")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return every policy field that can affect a durable planning result."""

        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "version": self.version,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "max_chars": self.max_chars,
            "failure_policy": self.failure_policy.value,
            "max_completion_tokens": self.max_completion_tokens,
            "max_retries": self.max_retries,
            "exchange_policy_version": EXCHANGE_POLICY_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable digest suitable for Delta compatibility checks."""

        return _sha256_canonical(self.fingerprint_payload())


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """A non-empty, contiguous range of original conversation turns."""

    index: int
    start_turn: int
    end_turn: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("index", self.index),
            ("start_turn", self.start_turn),
            ("end_turn", self.end_turn),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.end_turn < self.start_turn:
            raise ValueError("an exchange must contain at least one turn")


@dataclass(frozen=True, slots=True)
class ParsedConversation:
    """Canonical original turns plus their deterministic exchange ledger."""

    canonical_turns: tuple[str, ...]
    exchanges: tuple[ConversationExchange, ...]
    input_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_turns, tuple) or any(
            not isinstance(turn, str) or not turn for turn in self.canonical_turns
        ):
            raise TypeError("canonical_turns must be a tuple of non-empty JSON strings")
        if not isinstance(self.exchanges, tuple) or any(
            not isinstance(exchange, ConversationExchange) for exchange in self.exchanges
        ):
            raise TypeError("exchanges must be a tuple of ConversationExchange values")
        _validate_digest(self.input_hash, field_name="input_hash")
        if not self.canonical_turns:
            if self.exchanges:
                raise ValueError("an empty conversation cannot contain exchanges")
            return
        if not self.exchanges:
            raise ValueError("a non-empty conversation must contain exchanges")
        expected_start = 0
        for expected_index, exchange in enumerate(self.exchanges):
            if exchange.index != expected_index:
                raise ValueError("exchange indices must be contiguous and zero-based")
            if exchange.start_turn != expected_start:
                raise ValueError("exchanges must cover turns without gaps or overlap")
            expected_start = exchange.end_turn + 1
        if expected_start != len(self.canonical_turns):
            raise ValueError("exchanges must cover every conversation turn")

    @property
    def canonical_text(self) -> str:
        return f"[{','.join(self.canonical_turns)}]"

    def render_exchange_range(self, start_exchange: int, end_exchange: int) -> str:
        if (
            isinstance(start_exchange, bool)
            or isinstance(end_exchange, bool)
            or not isinstance(start_exchange, int)
            or not isinstance(end_exchange, int)
            or start_exchange < 0
            or end_exchange < start_exchange
            or end_exchange >= len(self.exchanges)
        ):
            raise ValueError("exchange range is outside the parsed conversation")
        first_turn = self.exchanges[start_exchange].start_turn
        last_turn = self.exchanges[end_exchange].end_turn
        return f"[{','.join(self.canonical_turns[first_turn : last_turn + 1])}]"


@dataclass(frozen=True, slots=True)
class MaterializedSegment:
    """One exact source slice that can later be adapted to a ``ChunkPlan``."""

    ordinal: int
    text: str
    content_hash: str
    semantic_segment_index: int | None = None
    start_exchange: int | None = None
    end_exchange: int | None = None
    oversized_atomic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_digest(self.content_hash, field_name="content_hash")
        if not isinstance(self.oversized_atomic, bool):
            raise TypeError("oversized_atomic must be a boolean")
        positional = (self.semantic_segment_index, self.start_exchange, self.end_exchange)
        if all(value is None for value in positional):
            if self.oversized_atomic:
                raise ValueError("oversized_atomic requires semantic exchange positions")
            return
        if any(value is None for value in positional):
            raise ValueError("semantic segment positions must either all be present or all be absent")
        assert self.semantic_segment_index is not None
        assert self.start_exchange is not None
        assert self.end_exchange is not None
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in positional):
            raise ValueError("semantic segment positions must be non-negative integers")
        if self.end_exchange < self.start_exchange:
            raise ValueError("a materialized semantic segment cannot be empty")
        if self.oversized_atomic and self.start_exchange != self.end_exchange:
            raise ValueError("oversized_atomic must identify exactly one exchange")


@dataclass(frozen=True, slots=True)
class SegmentManifestEntry:
    """Durable, text-free identity for one materialized segment."""

    ordinal: int
    content_hash: str
    semantic_segment_index: int | None
    start_exchange: int | None
    end_exchange: int | None
    oversized_atomic: bool = False

    @classmethod
    def from_segment(cls, segment: MaterializedSegment) -> "SegmentManifestEntry":
        return cls(
            ordinal=segment.ordinal,
            content_hash=segment.content_hash,
            semantic_segment_index=segment.semantic_segment_index,
            start_exchange=segment.start_exchange,
            end_exchange=segment.end_exchange,
            oversized_atomic=segment.oversized_atomic,
        )

    def __post_init__(self) -> None:
        # Reuse the materialized-value validation without retaining source text.
        MaterializedSegment(
            ordinal=self.ordinal,
            text="",
            content_hash=self.content_hash,
            semantic_segment_index=self.semantic_segment_index,
            start_exchange=self.start_exchange,
            end_exchange=self.end_exchange,
            oversized_atomic=self.oversized_atomic,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "content_hash": self.content_hash,
            "semantic_segment_index": self.semantic_segment_index,
            "start_exchange": self.start_exchange,
            "end_exchange": self.end_exchange,
            "oversized_atomic": self.oversized_atomic,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SegmentManifestEntry":
        """Load one entry from the exact text-free checkpoint schema."""

        if not isinstance(value, dict):
            raise TypeError("each manifest chunk must be an object")
        expected_keys = {
            "ordinal",
            "content_hash",
            "semantic_segment_index",
            "start_exchange",
            "end_exchange",
            "oversized_atomic",
        }
        if set(value) != expected_keys:
            raise ValueError("manifest chunk fields do not match the supported schema")
        return cls(
            ordinal=value["ordinal"],
            content_hash=value["content_hash"],
            semantic_segment_index=value["semantic_segment_index"],
            start_exchange=value["start_exchange"],
            end_exchange=value["end_exchange"],
            oversized_atomic=value["oversized_atomic"],
        )


def compute_plan_digest(
    *,
    input_hash: str,
    policy_fingerprint: str,
    effective_strategy: EffectiveSegmentationStrategy,
    end_exchange_indices: tuple[int, ...],
    chunks: tuple[SegmentManifestEntry, ...],
) -> str:
    """Hash every behavioral input and ordered output of materialization."""

    _validate_digest(input_hash, field_name="input_hash")
    _validate_digest(policy_fingerprint, field_name="policy_fingerprint")
    if not isinstance(effective_strategy, EffectiveSegmentationStrategy):
        raise TypeError("effective_strategy must be an EffectiveSegmentationStrategy")
    return _sha256_canonical(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "input_hash": input_hash,
            "policy_fingerprint": policy_fingerprint,
            "effective_strategy": effective_strategy.value,
            "end_exchange_indices": list(end_exchange_indices),
            "chunks": [chunk.as_dict() for chunk in chunks],
        }
    )


@dataclass(frozen=True, slots=True)
class SegmentationManifest:
    """Versioned state for durable policy, Delta, and retry compatibility."""

    input_hash: str
    policy_fingerprint: str
    effective_strategy: EffectiveSegmentationStrategy
    end_exchange_indices: tuple[int, ...]
    chunks: tuple[SegmentManifestEntry, ...]
    plan_digest: str
    fallback_reason: str | None = None
    schema_version: str = MANIFEST_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        input_hash: str,
        policy_fingerprint: str,
        effective_strategy: EffectiveSegmentationStrategy,
        end_exchange_indices: tuple[int, ...],
        segments: tuple[MaterializedSegment, ...],
        fallback_reason: str | None = None,
    ) -> "SegmentationManifest":
        entries = tuple(SegmentManifestEntry.from_segment(segment) for segment in segments)
        return cls(
            input_hash=input_hash,
            policy_fingerprint=policy_fingerprint,
            effective_strategy=effective_strategy,
            end_exchange_indices=end_exchange_indices,
            chunks=entries,
            plan_digest=compute_plan_digest(
                input_hash=input_hash,
                policy_fingerprint=policy_fingerprint,
                effective_strategy=effective_strategy,
                end_exchange_indices=end_exchange_indices,
                chunks=entries,
            ),
            fallback_reason=fallback_reason,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SegmentationManifest":
        """Strictly deserialize the text-free durable checkpoint form.

        Unknown or missing fields are rejected so a caller cannot silently
        reuse a manifest written under a different schema contract.
        """

        if not isinstance(value, dict):
            raise TypeError("segmentation manifest must be an object")
        expected_keys = {
            "schema_version",
            "input_hash",
            "policy_fingerprint",
            "effective_strategy",
            "end_exchange_indices",
            "chunks",
            "plan_digest",
            "fallback_reason",
        }
        if set(value) != expected_keys:
            raise ValueError("segmentation manifest fields do not match the supported schema")

        raw_strategy = value["effective_strategy"]
        if not isinstance(raw_strategy, str):
            raise TypeError("effective_strategy must be a string")
        try:
            strategy = EffectiveSegmentationStrategy(raw_strategy)
        except ValueError as exc:
            raise ValueError("effective_strategy is not supported") from exc

        raw_boundaries = value["end_exchange_indices"]
        if not isinstance(raw_boundaries, list):
            raise TypeError("end_exchange_indices must be a list")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_boundaries):
            raise TypeError("end_exchange_indices must contain only integers")

        raw_chunks = value["chunks"]
        if not isinstance(raw_chunks, list):
            raise TypeError("chunks must be a list")

        return cls(
            schema_version=value["schema_version"],
            input_hash=value["input_hash"],
            policy_fingerprint=value["policy_fingerprint"],
            effective_strategy=strategy,
            end_exchange_indices=tuple(raw_boundaries),
            chunks=tuple(SegmentManifestEntry.from_dict(chunk) for chunk in raw_chunks),
            plan_digest=value["plan_digest"],
            fallback_reason=value["fallback_reason"],
        )

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
        _validate_digest(self.input_hash, field_name="input_hash")
        _validate_digest(self.policy_fingerprint, field_name="policy_fingerprint")
        _validate_digest(self.plan_digest, field_name="plan_digest")
        if not isinstance(self.effective_strategy, EffectiveSegmentationStrategy):
            raise TypeError("effective_strategy must be an EffectiveSegmentationStrategy")
        if not isinstance(self.end_exchange_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.end_exchange_indices
        ):
            raise TypeError("end_exchange_indices must be a tuple of non-negative integers")
        if not isinstance(self.chunks, tuple) or any(
            not isinstance(chunk, SegmentManifestEntry) for chunk in self.chunks
        ):
            raise TypeError("chunks must be a tuple of SegmentManifestEntry values")
        if tuple(chunk.ordinal for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("manifest chunk ordinals must be contiguous and zero-based")
        if not self.chunks:
            raise ValueError("a segmentation manifest must contain at least one chunk")
        if self.fallback_reason is not None and (not isinstance(self.fallback_reason, str) or not self.fallback_reason):
            raise ValueError("fallback_reason must be a non-empty string or None")
        if self.effective_strategy is EffectiveSegmentationStrategy.SEMANTIC:
            if not self.end_exchange_indices:
                raise ValueError("semantic manifests require at least one exchange boundary")
            if any(
                chunk.semantic_segment_index is None or chunk.start_exchange is None or chunk.end_exchange is None
                for chunk in self.chunks
            ):
                raise ValueError("semantic manifest chunks require exchange positions")
            if self.fallback_reason is not None:
                raise ValueError("semantic manifests cannot contain a fallback reason")
        else:
            if self.end_exchange_indices:
                raise ValueError("fixed manifests cannot contain semantic boundaries")
            if any(
                chunk.semantic_segment_index is not None
                or chunk.start_exchange is not None
                or chunk.end_exchange is not None
                for chunk in self.chunks
            ):
                raise ValueError("fixed manifest chunks cannot contain semantic positions")
            if self.effective_strategy is EffectiveSegmentationStrategy.FIXED_FALLBACK and self.fallback_reason is None:
                raise ValueError("fixed fallback manifests require a fallback reason")
            if self.effective_strategy is not EffectiveSegmentationStrategy.FIXED_FALLBACK:
                if self.fallback_reason is not None:
                    raise ValueError("non-fallback manifests cannot contain a fallback reason")
                if any(chunk.oversized_atomic for chunk in self.chunks):
                    raise ValueError("non-semantic manifest chunks cannot be marked oversized_atomic")
        if any(
            boundary <= previous
            for previous, boundary in zip((-1, *self.end_exchange_indices), self.end_exchange_indices)
        ):
            raise ValueError("manifest boundaries must be strictly increasing")
        expected_digest = compute_plan_digest(
            input_hash=self.input_hash,
            policy_fingerprint=self.policy_fingerprint,
            effective_strategy=self.effective_strategy,
            end_exchange_indices=self.end_exchange_indices,
            chunks=self.chunks,
        )
        if self.plan_digest != expected_digest:
            raise ValueError("plan_digest does not match the manifest payload")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_hash": self.input_hash,
            "policy_fingerprint": self.policy_fingerprint,
            "effective_strategy": self.effective_strategy.value,
            "end_exchange_indices": list(self.end_exchange_indices),
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "plan_digest": self.plan_digest,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Materialized segments, durable manifest, and segmentation token usage."""

    segments: tuple[MaterializedSegment, ...]
    manifest: SegmentationManifest
    usage: TokenUsage = field(default_factory=TokenUsage)

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, MaterializedSegment) for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of MaterializedSegment values")
        if tuple(segment.ordinal for segment in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("segment ordinals must be contiguous and zero-based")
        if not isinstance(self.manifest, SegmentationManifest):
            raise TypeError("manifest must be a SegmentationManifest")
        if tuple(SegmentManifestEntry.from_segment(segment) for segment in self.segments) != self.manifest.chunks:
            raise ValueError("segments do not match the durable manifest")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage")


__all__ = [
    "BoundaryResponse",
    "ConversationExchange",
    "EffectiveSegmentationStrategy",
    "EXCHANGE_POLICY_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MATERIALIZER_VERSION",
    "MaterializedSegment",
    "POLICY_SCHEMA_VERSION",
    "ParsedConversation",
    "SegmentManifestEntry",
    "SegmentationFailurePolicy",
    "SegmentationManifest",
    "SegmentationMode",
    "SegmentationResult",
    "SEMANTIC_POLICY_VERSION",
    "SEMANTIC_PROMPT_VERSION",
    "SemanticSegmentationPolicy",
    "compute_plan_digest",
]
