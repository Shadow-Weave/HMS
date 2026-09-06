import asyncio

import pytest
from rsi_harness import (
    MemoryDecisionType,
    MemoryNamespace,
    MemoryOrchestrator,
    RunContext,
    SessionEvent,
    SessionEventType,
)


def _event(
    event_type=SessionEventType.FACT_OBSERVED,
    *,
    content="The preferred color is blue.",
    source_position=0,
    candidate_kind="preference",
):
    return SessionEvent(
        run_id="run-1",
        stream_id="user-1",
        session_id="session-1",
        sequence=0,
        source_position=source_position,
        event_type=event_type,
        observation=content,
        evidence_refs=("session://user-1/session-1/0",),
        metadata={
            "candidate": {
                "kind": candidate_kind,
                "namespace": "general",
                "trigger": "preference_observed",
                "content": content,
                "metadata": {"entity_id": "user-1"},
            }
        },
    )


def test_memory_orchestrator_builds_candidate_without_promoting_it():
    orchestrator = MemoryOrchestrator(RunContext("run-1", "task-1", stream_id="user-1"))
    candidates = asyncio.run(orchestrator.handle(_event()))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.namespace is MemoryNamespace.GENERAL
    assert candidate.kind == "preference"
    assert candidate.created_before_verifier is True
    assert candidate.metadata["session_id"] == "session-1"


def test_update_and_conflict_map_to_explicit_candidate_kinds():
    orchestrator = MemoryOrchestrator(RunContext("run-1", "task-1", stream_id="user-1"))
    update = asyncio.run(
        orchestrator.handle(
            _event(
                SessionEventType.KNOWLEDGE_UPDATE_DETECTED,
                content="The city is now Paris.",
                candidate_kind="temporal_update",
            )
        )
    )
    conflict = asyncio.run(
        orchestrator.handle(_event(SessionEventType.CONTRADICTION_DETECTED, content="The city is both Paris and London."))
    )

    assert update[0].kind == "temporal_update"
    assert conflict[0].metadata["orchestrator_decision"] == MemoryDecisionType.KEEP_CONFLICT.value


def test_memory_orchestrator_rejects_cross_run_and_backwards_positions():
    orchestrator = MemoryOrchestrator(RunContext("run-1", "task-1", stream_id="user-1"))
    with pytest.raises(ValueError, match="different run"):
        asyncio.run(
            orchestrator.handle(
                SessionEvent(
                    run_id="run-2",
                    stream_id="user-1",
                    session_id="session-1",
                    sequence=0,
                    source_position=0,
                    event_type=SessionEventType.MESSAGE,
                    observation="not this run",
                )
            )
        )

    asyncio.run(orchestrator.handle(_event(source_position=1)))
    with pytest.raises(ValueError, match="backwards"):
        asyncio.run(
            orchestrator.handle(
                SessionEvent(
                    run_id="run-1",
                    stream_id="user-1",
                    session_id="session-0",
                    sequence=1,
                    source_position=0,
                    event_type=SessionEventType.MESSAGE,
                    observation="invalid position",
                )
            )
        )
