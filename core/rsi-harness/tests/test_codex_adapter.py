from __future__ import annotations

import pytest
from rsi_harness import CodexEventAdapter, EventType


def test_codex_stop_payload_becomes_result_without_transcript_upload():
    event = CodexEventAdapter().from_payload(
        {
            "session_id": "session-1",
            "seq": 4,
            "hook_event_name": "Stop",
            "summary": "tests passed",
            "trace_refs": ["trace://session-1/4"],
        }
    )
    assert event.event_type is EventType.RESULT
    assert event.trace_refs == ("trace://session-1/4",)
    assert "transcript" not in event.metadata


def test_codex_error_and_inactivity_are_classified():
    adapter = CodexEventAdapter(inactivity_threshold_seconds=60)
    error = adapter.from_payload(
        {"session_id": "s", "seq": 0, "hook_event_name": "Stop", "error": "tests failed"}
    )
    idle = adapter.from_payload(
        {"session_id": "s", "seq": 1, "hook_event_name": "Progress", "inactivity_seconds": 60}
    )
    assert error.event_type is EventType.ERROR
    assert idle.event_type is EventType.INACTIVITY


def test_codex_event_requires_monotonic_sequence_input():
    with pytest.raises(ValueError, match="monotonic seq"):
        CodexEventAdapter().from_payload({"session_id": "s", "hook_event_name": "Stop"})
