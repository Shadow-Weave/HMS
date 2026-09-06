from __future__ import annotations

import pytest
from rsi_harness import (
    EventType,
    MemoryCandidate,
    MemoryNamespace,
    SupervisorAction,
    SupervisorActionType,
    WorkerEvent,
)


def test_worker_event_parses_and_preserves_trace_refs() -> None:
    event = WorkerEvent.from_mapping(
        {
            "run_id": "run-1",
            "worker_id": "worker-1",
            "seq": 3,
            "type": "risk",
            "observation": "tests remain red",
            "trace_refs": "trace://run-1/3",
        }
    )

    assert event.event_type is EventType.RISK
    assert event.trace_refs == ("trace://run-1/3",)


def test_invalid_event_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid type"):
        WorkerEvent.from_mapping(
            {"run_id": "run-1", "worker_id": "worker-1", "seq": 0, "type": "verifier_output"}
        )


def test_error_event_requires_evidence() -> None:
    with pytest.raises(ValueError, match="must include"):
        WorkerEvent(
            run_id="run-1",
            worker_id="worker-1",
            sequence=0,
            event_type=EventType.ERROR,
        )


def test_steer_and_abort_require_action_payloads() -> None:
    with pytest.raises(ValueError, match="instruction"):
        SupervisorAction("run-1", "worker-1", SupervisorActionType.STEER)
    with pytest.raises(ValueError, match="reason"):
        SupervisorAction("run-1", "worker-1", SupervisorActionType.ABORT)


def test_candidate_cannot_be_created_after_verifier() -> None:
    with pytest.raises(ValueError, match="before verifier"):
        MemoryCandidate(
            candidate_id="candidate-1",
            kind="procedure",
            namespace=MemoryNamespace.CANDIDATE,
            trigger="a failing test",
            content="use the generalized recovery procedure",
            source_run_id="run-1",
            evidence_refs=("trace://run-1/1",),
            created_before_verifier=False,
        )


def test_candidate_cannot_use_episode_namespace() -> None:
    with pytest.raises(ValueError, match="candidate namespace"):
        MemoryCandidate(
            candidate_id="candidate-1",
            kind="procedure",
            namespace=MemoryNamespace.EPISODE,
            trigger="a failing test",
            content="use the generalized recovery procedure",
            source_run_id="run-1",
            evidence_refs=("trace://run-1/1",),
        )
