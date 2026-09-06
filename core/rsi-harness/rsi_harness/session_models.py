"""Backend-neutral contracts for chronological multi-session memory runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


class SessionEventType(StrEnum):
    """Semantic events emitted while replaying a session stream."""

    SESSION_START = "session_start"
    MESSAGE = "message"
    FACT_OBSERVED = "fact_observed"
    PREFERENCE_OBSERVED = "preference_observed"
    ENTITY_LINKED = "entity_linked"
    CONTRADICTION_DETECTED = "contradiction_detected"
    KNOWLEDGE_UPDATE_DETECTED = "knowledge_update_detected"
    SESSION_END = "session_end"
    QUERY_RECEIVED = "query_received"
    ANSWER_EMITTED = "answer_emitted"


class MemoryDecisionType(StrEnum):
    """Actions a Memory Orchestrator may propose before candidate staging."""

    NOOP = "noop"
    PROPOSE_CANDIDATE = "propose_candidate"
    SUPERSEDE = "supersede"
    KEEP_CONFLICT = "keep_conflict"
    ABSTAIN = "abstain"
    INVALIDATE = "invalidate"


@dataclass(frozen=True, slots=True)
class SessionInput:
    """One source session in chronological order.

    ``messages`` contains source turns only.  Evaluation labels and gold
    answers deliberately do not belong to this contract.
    """

    stream_id: str
    session_id: str
    source_position: int
    messages: tuple[Mapping[str, Any], ...]
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.stream_id, "stream_id")
        _require_text(self.session_id, "session_id")
        if self.source_position < 0:
            raise ValueError("source_position must be non-negative")
        if not self.messages:
            raise ValueError("session must contain at least one message")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("session ended_at must not precede started_at")
        for message in self.messages:
            if not isinstance(message, Mapping):
                raise ValueError("session messages must be mappings")
            _require_text(message.get("role"), "message.role")
            _require_text(message.get("content"), "message.content")


@dataclass(frozen=True, slots=True)
class QueryInput:
    """A query evaluated against a stream watermark."""

    evaluation_id: str
    stream_id: str
    query_id: str
    query: str
    query_position: int
    asked_at: datetime | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.evaluation_id, "evaluation_id"),
            (self.stream_id, "stream_id"),
            (self.query_id, "query_id"),
            (self.query, "query"),
        ):
            _require_text(value, field_name)
        if self.query_position < 0:
            raise ValueError("query_position must be non-negative")


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """Bounded, ordered event for a chronological session stream."""

    run_id: str
    stream_id: str
    session_id: str
    sequence: int
    source_position: int
    event_type: SessionEventType
    observation: str = ""
    event_time: datetime = field(default_factory=_utc_now)
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.run_id, "run_id"),
            (self.stream_id, "stream_id"),
            (self.session_id, "session_id"),
        ):
            _require_text(value, field_name)
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.source_position < 0:
            raise ValueError("source_position must be non-negative")
        if self.event_type in {
            SessionEventType.MESSAGE,
            SessionEventType.FACT_OBSERVED,
            SessionEventType.PREFERENCE_OBSERVED,
            SessionEventType.CONTRADICTION_DETECTED,
            SessionEventType.KNOWLEDGE_UPDATE_DETECTED,
        } and not self.observation.strip():
            raise ValueError(f"{self.event_type} event must include an observation")


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """Memory Orchestrator output before candidate construction."""

    decision: MemoryDecisionType
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision in {
            MemoryDecisionType.PROPOSE_CANDIDATE,
            MemoryDecisionType.SUPERSEDE,
            MemoryDecisionType.KEEP_CONFLICT,
            MemoryDecisionType.INVALIDATE,
        } and not self.reason.strip():
            raise ValueError(f"{self.decision} decision must include a reason")
