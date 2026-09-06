"""Codex hook payload -> backend-neutral WorkerEvent adapter.

The adapter consumes only structured, already-redacted hook fields.  It never
uploads a transcript or tool output by itself.  A runtime can send the returned
event to ``SupervisorController`` or serialize it to an event sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .events import EventType, WorkerEvent


@dataclass(frozen=True, slots=True)
class CodexEventAdapter:
    """Translate stable Codex lifecycle fields into WorkerEvent digests."""

    inactivity_threshold_seconds: int = 300

    def from_payload(self, payload: Mapping[str, Any]) -> WorkerEvent:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("Codex payload must contain session_id")
        run_id = str(payload.get("run_id") or session_id)
        worker_id = str(payload.get("worker_id") or "codex-worker")
        if "seq" not in payload:
            raise ValueError("Codex payload must contain a monotonic seq")

        hook_name = str(payload.get("hook_event_name") or payload.get("hookEventName") or "").lower()
        summary = str(payload.get("summary") or payload.get("observation") or "").strip()
        risk = str(payload.get("risk") or "").strip()
        error = str(payload.get("error") or "").strip()
        next_step = str(payload.get("next_step") or "").strip()

        if error:
            event_type = EventType.ERROR
        elif self._inactivity_seconds(payload) >= self.inactivity_threshold_seconds:
            event_type = EventType.INACTIVITY
            risk = risk or "worker inactivity threshold exceeded"
        elif hook_name in {"stop", "sessionend", "result"}:
            event_type = EventType.RESULT
        elif hook_name in {"userpromptsubmit", "question"}:
            event_type = EventType.QUESTION
        elif risk:
            event_type = EventType.RISK
        else:
            event_type = EventType.PROGRESS

        refs = payload.get("trace_refs", ())
        if isinstance(refs, str):
            refs = (refs,)
        return WorkerEvent(
            run_id=run_id,
            worker_id=worker_id,
            sequence=int(payload["seq"]),
            event_type=event_type,
            goal_delta=str(payload.get("goal_delta") or ""),
            observation=summary or error or hook_name or "codex event",
            attempt=str(payload.get("attempt") or ""),
            outcome=str(payload.get("outcome") or ""),
            risk=risk,
            next_step=next_step,
            trace_refs=tuple(str(ref) for ref in refs if str(ref).strip()),
            metadata={
                "source": "codex_hook",
                "session_id": session_id,
                "hook_event_name": hook_name,
            },
        )

    @staticmethod
    def _inactivity_seconds(payload: Mapping[str, Any]) -> int:
        value = payload.get("inactivity_seconds", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
