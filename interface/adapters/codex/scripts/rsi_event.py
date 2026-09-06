#!/usr/bin/env python3
"""Emit a redacted Codex lifecycle event for the RSI harness.

This hook deliberately does not read ``transcript_path`` or tool output.  It
serializes only a bounded WorkerEvent projection into the local outbox.  A
separate runner can drain this outbox into ``SupervisorController`` or an HMS
event endpoint without changing the Codex hook contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
HARNESS_ROOT = REPO_ROOT / "core" / "rsi-harness"
if HARNESS_ROOT.is_dir():
    sys.path.insert(0, str(HARNESS_ROOT))

from lib.config import debug_log, load_config  # noqa: E402
from lib.state import read_state, write_state  # noqa: E402
from rsi_harness import CodexEventAdapter  # noqa: E402

_MAX_OUTBOX_EVENTS = 1000
_ALLOWED_FIELDS = (
    "session_id",
    "run_id",
    "worker_id",
    "hook_event_name",
    "hookEventName",
    "summary",
    "observation",
    "risk",
    "error",
    "next_step",
    "goal_delta",
    "attempt",
    "outcome",
    "trace_refs",
    "inactivity_seconds",
)


def _next_sequence(session_id: str) -> int:
    sequences = read_state("rsi-sequences.json", {})
    sequence = int(sequences.get(session_id, 0)) + 1
    sequences[session_id] = sequence
    write_state("rsi-sequences.json", sequences)
    return sequence


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project hook input to fields accepted by CodexEventAdapter."""
    redacted = {key: payload[key] for key in _ALLOWED_FIELDS if key in payload}
    redacted["seq"] = _next_sequence(str(payload.get("session_id", "unknown")))
    return redacted


def main() -> None:
    config = load_config()
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        debug_log(config, "RSI event hook received invalid JSON")
        return

    if not isinstance(hook_input, dict) or not hook_input.get("session_id"):
        debug_log(config, "RSI event hook skipped payload without session_id")
        return

    try:
        event = CodexEventAdapter().from_payload(_redact_payload(hook_input))
    except (TypeError, ValueError) as exc:
        debug_log(config, f"RSI event hook skipped payload: {exc}")
        return

    outbox = read_state("rsi-events.json", [])
    if not isinstance(outbox, list):
        outbox = []
    outbox.append(
        {
            "run_id": event.run_id,
            "worker_id": event.worker_id,
            "sequence": event.sequence,
            "event_type": event.event_type.value,
            "goal_delta": event.goal_delta,
            "observation": event.observation[:2000],
            "attempt": event.attempt,
            "outcome": event.outcome,
            "risk": event.risk[:500],
            "next_step": event.next_step[:1000],
            "trace_refs": list(event.trace_refs[:20]),
            "metadata": dict(event.metadata),
        }
    )
    write_state("rsi-events.json", outbox[-_MAX_OUTBOX_EVENTS:])
    debug_log(config, f"RSI WorkerEvent queued: {event.event_type.value} seq={event.sequence}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Hooks must never interrupt the coding agent.
        print(f"[HMS] RSI event hook error: {exc}", file=sys.stderr)
        sys.exit(0)
