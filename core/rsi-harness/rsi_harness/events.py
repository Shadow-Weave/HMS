"""Worker event and supervisor action protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventType(StrEnum):
    PROGRESS = "progress"
    QUESTION = "question"
    RISK = "risk"
    ERROR = "error"
    RESULT = "result"
    INACTIVITY = "inactivity"


class SupervisorActionType(StrEnum):
    NOTIFY_ACK = "notify_ack"
    STEER = "steer"
    ABORT = "abort"
    SPAWN = "spawn"


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    """Bounded event digest sent from a worker to the supervisor."""

    run_id: str
    worker_id: str
    sequence: int
    event_type: EventType
    goal_delta: str = ""
    observation: str = ""
    attempt: str = ""
    outcome: str = ""
    risk: str = ""
    next_step: str = ""
    trace_refs: tuple[str, ...] = ()
    emitted_at: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("event run_id must not be empty")
        if not self.worker_id.strip():
            raise ValueError("event worker_id must not be empty")
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if self.event_type in (EventType.ERROR, EventType.RISK, EventType.QUESTION) and not (
            self.observation or self.risk or self.outcome
        ):
            raise ValueError(f"{self.event_type} event must include an observation, risk, or outcome")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkerEvent":
        """Parse an adapter payload without accepting arbitrary event types."""

        try:
            event_type = EventType(str(value["type"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("event payload has an invalid type") from exc

        refs = value.get("trace_refs", ())
        if isinstance(refs, str):
            refs = (refs,)
        return cls(
            run_id=str(value["run_id"]),
            worker_id=str(value["worker_id"]),
            sequence=int(value["seq"]),
            event_type=event_type,
            goal_delta=str(value.get("goal_delta", "")),
            observation=str(value.get("observation", "")),
            attempt=str(value.get("attempt", "")),
            outcome=str(value.get("outcome", "")),
            risk=str(value.get("risk", "")),
            next_step=str(value.get("next_step", "")),
            trace_refs=tuple(str(ref) for ref in refs),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SupervisorAction:
    """Command sent from the supervisor to a worker runtime."""

    run_id: str
    worker_id: str
    action_type: SupervisorActionType
    instruction: str = ""
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("action run_id must not be empty")
        if not self.worker_id.strip():
            raise ValueError("action worker_id must not be empty")
        if self.action_type == SupervisorActionType.STEER and not self.instruction.strip():
            raise ValueError("steer action must include an instruction")
        if self.action_type == SupervisorActionType.ABORT and not self.reason.strip():
            raise ValueError("abort action must include a reason")
