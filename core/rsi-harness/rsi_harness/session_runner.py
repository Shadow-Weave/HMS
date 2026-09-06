"""Deterministic chronological replay for multi-session memory streams."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from .models import RunContext
from .session_models import SessionEvent, SessionEventType, SessionInput


class SessionOrderError(ValueError):
    """Raised when a stream contains duplicate, reversed, or future sessions."""


EventSink = Callable[[SessionEvent], Any | Awaitable[Any]]


def _coerce_datetime(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return fallback
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid event_time: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class ChronologicalSessionRunner:
    """Replay source sessions without sorting or silently repairing them.

    The runner is deliberately deterministic and does not call an LLM.  A
    later semantic extractor can classify emitted ``MESSAGE`` events as facts,
    updates, or contradictions, but it cannot change their source order.
    """

    run: RunContext
    event_sink: EventSink | None = None
    max_observation_chars: int = 2_000

    def __post_init__(self) -> None:
        if self.max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")

    async def replay(
        self,
        sessions: Sequence[SessionInput],
        *,
        query_position: int | None = None,
    ) -> tuple[SessionEvent, ...]:
        """Replay sessions strictly before ``query_position`` when provided."""

        if query_position is not None and query_position < 0:
            raise ValueError("query_position must be non-negative")
        events: list[SessionEvent] = []
        previous_position: int | None = None
        seen_session_ids: set[str] = set()
        for session in sessions:
            self._validate_session(session, previous_position, seen_session_ids)
            previous_position = session.source_position
            seen_session_ids.add(session.session_id)
            if query_position is not None and session.source_position >= query_position:
                continue
            events.append(
                await self._emit(
                    SessionEvent(
                        run_id=self.run.run_id,
                        stream_id=session.stream_id,
                        session_id=session.session_id,
                        sequence=len(events),
                        source_position=session.source_position,
                        event_type=SessionEventType.SESSION_START,
                        observation="session started",
                        event_time=session.started_at or datetime.now(timezone.utc),
                        metadata=dict(session.metadata),
                    )
                )
            )
            for turn_index, message in enumerate(session.messages):
                event_type = self._message_event_type(message)
                content = str(message["content"]).strip()
                if len(content) > self.max_observation_chars:
                    content = content[: self.max_observation_chars].rstrip() + "…"
                evidence_ref = f"session://{session.stream_id}/{session.session_id}/{turn_index}"
                message_metadata = self._message_metadata(message)
                events.append(
                    await self._emit(
                        SessionEvent(
                            run_id=self.run.run_id,
                            stream_id=session.stream_id,
                            session_id=session.session_id,
                            sequence=len(events),
                            source_position=session.source_position,
                            event_type=event_type,
                            observation=content,
                            event_time=_coerce_datetime(message.get("event_time"), session.started_at or datetime.now(timezone.utc)),
                            evidence_refs=(evidence_ref,),
                            metadata={
                                "role": str(message["role"]),
                                "turn_index": turn_index,
                                "source_session_metadata": dict(session.metadata),
                                **message_metadata,
                            },
                        )
                    )
                )
            events.append(
                await self._emit(
                    SessionEvent(
                        run_id=self.run.run_id,
                        stream_id=session.stream_id,
                        session_id=session.session_id,
                        sequence=len(events),
                        source_position=session.source_position,
                        event_type=SessionEventType.SESSION_END,
                        observation="session ended",
                        event_time=session.ended_at or session.started_at or datetime.now(timezone.utc),
                        metadata=dict(session.metadata),
                    )
                )
            )
        return tuple(events)

    def _validate_session(
        self,
        session: SessionInput,
        previous_position: int | None,
        seen_session_ids: set[str],
    ) -> None:
        expected_stream = self.run.stream_id
        if expected_stream is not None and session.stream_id != expected_stream:
            raise SessionOrderError("session belongs to a different stream")
        if session.session_id in seen_session_ids:
            raise SessionOrderError(f"duplicate session_id: {session.session_id}")
        if previous_position is not None and session.source_position <= previous_position:
            raise SessionOrderError(
                f"session source_position must increase: {session.source_position} <= {previous_position}"
            )

    @staticmethod
    def _message_event_type(message: dict[str, Any] | Any) -> SessionEventType:
        raw_type = message.get("event_type") if hasattr(message, "get") else None
        if raw_type:
            try:
                event_type = SessionEventType(str(raw_type))
            except ValueError as exc:
                raise ValueError(f"invalid message event_type: {raw_type!r}") from exc
            if event_type in {
                SessionEventType.SESSION_START,
                SessionEventType.SESSION_END,
                SessionEventType.QUERY_RECEIVED,
                SessionEventType.ANSWER_EMITTED,
            }:
                raise ValueError("message event_type cannot be a boundary or answer event")
            return event_type
        return SessionEventType.MESSAGE

    @staticmethod
    def _message_metadata(message: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Copy only structured, pre-verifier metadata into an event.

        Session adapters may provide a semantic event type and a structured
        candidate produced by an upstream extractor.  The runner must not
        copy arbitrary source fields because LongMemEval-style turns can carry
        evaluator annotations.  ``has_answer`` and all answer-like fields are
        therefore ignored even when a caller bypasses the dataset adapter.
        """

        if not hasattr(message, "get"):
            return {}
        allowed = {"candidate", "entity_id", "observed_at", "valid_from", "valid_to", "supersedes_memory_id"}
        metadata: dict[str, Any] = {}
        raw_metadata = message.get("metadata")
        if isinstance(raw_metadata, Mapping):
            for key in allowed:
                if key in raw_metadata:
                    metadata[key] = raw_metadata[key]
        for key in allowed:
            if key in message:
                metadata[key] = message[key]
        return metadata

    async def _emit(self, event: SessionEvent) -> SessionEvent:
        if self.event_sink is not None:
            result = self.event_sink(event)
            if inspect.isawaitable(result):
                await result
        return event


def replay_sessions(
    run: RunContext,
    sessions: Iterable[SessionInput],
    *,
    query_position: int | None = None,
    max_observation_chars: int = 2_000,
) -> tuple[SessionEvent, ...]:
    """Synchronous convenience wrapper for small offline fixtures."""

    import asyncio

    return asyncio.run(
        ChronologicalSessionRunner(run, max_observation_chars=max_observation_chars).replay(
            tuple(sessions), query_position=query_position
        )
    )
