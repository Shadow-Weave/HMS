import asyncio
from datetime import datetime, timezone

import pytest
from rsi_harness import (
    ChronologicalSessionRunner,
    RunContext,
    SessionEventType,
    SessionInput,
    SessionOrderError,
)


def _session(stream: str, session_id: str, position: int, content: str) -> SessionInput:
    timestamp = datetime(2025, 1, position + 1, tzinfo=timezone.utc)
    return SessionInput(
        stream_id=stream,
        session_id=session_id,
        source_position=position,
        started_at=timestamp,
        ended_at=timestamp,
        messages=({"role": "user", "content": content},),
    )


def test_replay_is_chronological_and_emits_bounded_events():
    run = RunContext("run-1", "session-task", stream_id="user-1")
    runner = ChronologicalSessionRunner(run)
    events = asyncio.run(
        runner.replay(
            (
                _session("user-1", "s-1", 0, "My preferred color is blue."),
                _session("user-1", "s-2", 1, "I moved to Paris."),
            )
        )
    )

    assert [event.event_type for event in events] == [
        SessionEventType.SESSION_START,
        SessionEventType.MESSAGE,
        SessionEventType.SESSION_END,
        SessionEventType.SESSION_START,
        SessionEventType.MESSAGE,
        SessionEventType.SESSION_END,
    ]
    assert [event.sequence for event in events] == list(range(6))
    assert [event.source_position for event in events] == [0, 0, 0, 1, 1, 1]
    assert len(events[1].observation) <= 2_000
    assert events[1].evidence_refs == ("session://user-1/s-1/0",)


def test_replay_stops_before_query_position():
    run = RunContext("run-1", "session-task", stream_id="user-1")
    runner = ChronologicalSessionRunner(run)
    events = asyncio.run(
        runner.replay(
            (_session("user-1", "s-1", 0, "before"), _session("user-1", "s-2", 1, "after")),
            query_position=1,
        )
    )
    assert {event.session_id for event in events} == {"s-1"}


def test_replay_rejects_reversed_or_duplicate_sessions():
    run = RunContext("run-1", "session-task", stream_id="user-1")
    runner = ChronologicalSessionRunner(run)
    with pytest.raises(SessionOrderError, match="source_position"):
        asyncio.run(
            runner.replay((_session("user-1", "s-2", 1, "later"), _session("user-1", "s-1", 0, "earlier")))
        )

    runner = ChronologicalSessionRunner(run)
    with pytest.raises(SessionOrderError, match="duplicate"):
        asyncio.run(
            runner.replay((_session("user-1", "s-1", 0, "one"), _session("user-1", "s-1", 1, "duplicate")))
        )
