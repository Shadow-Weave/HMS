"""Tests for the transcript-free RSI WorkerEvent hook."""

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rsi_event.py"
    spec = importlib.util.spec_from_file_location("rsi_event_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rsi_event_writes_redacted_structured_event(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    module = _load_script()
    payload = {
        "session_id": "session-1",
        "hook_event_name": "Stop",
        "summary": "completed task",
        "transcript_path": "/private/transcript.jsonl",
        "tool_output": "secret output must not be persisted",
    }
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        module.main()

    events = json.loads((tmp_path / ".hms" / "codex" / "state" / "rsi-events.json").read_text())
    assert events[0]["event_type"] == "result"
    serialized = json.dumps(events)
    assert "transcript" not in serialized
    assert "secret output" not in serialized


def test_rsi_event_sequence_is_monotonic(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    module = _load_script()
    payload = {"session_id": "session-1", "hook_event_name": "UserPromptSubmit", "summary": "question"}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        module.main()
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        module.main()

    events = json.loads((tmp_path / ".hms" / "codex" / "state" / "rsi-events.json").read_text())
    assert [event["sequence"] for event in events] == [1, 2]
