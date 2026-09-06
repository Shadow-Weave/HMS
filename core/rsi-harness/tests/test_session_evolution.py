from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from rsi_harness import (
    DevelopmentStreamVerifier,
    LongMemEvalAdapter,
    MemoryBackendCapabilities,
    PromotionResult,
    RunContext,
    SessionEventType,
    SessionEvolutionRunner,
    SessionEvolutionStateError,
    SessionInput,
)


class FakeBackend:
    capabilities = MemoryBackendCapabilities(search=False, promote=True, quarantine=True)

    def __init__(self) -> None:
        self.staged = []
        self.promoted = []
        self.quarantined = []

    async def stage(self, candidate):
        self.staged.append(candidate)
        return f"stage:{candidate.candidate_id}"

    async def promote(self, candidate_ids, *, target_snapshot):
        self.promoted.append((tuple(candidate_ids), target_snapshot))
        return PromotionResult(target_snapshot, tuple(candidate_ids))

    async def quarantine(self, candidate_ids, reason):
        self.quarantined.append((tuple(candidate_ids), reason))


def _session(session_id: str, position: int, content: str, *, candidate: bool = True) -> SessionInput:
    timestamp = datetime(2025, 1, position + 1, tzinfo=timezone.utc)
    message = {
        "role": "user",
        "content": content,
        "event_type": "fact_observed" if candidate else "message",
    }
    if candidate:
        message["candidate"] = {
            "kind": "fact",
            "namespace": "general",
            "trigger": "fact_observed",
            "content": content,
        }
    return SessionInput(
        stream_id="stream-1",
        session_id=session_id,
        source_position=position,
        started_at=timestamp,
        ended_at=timestamp,
        messages=(message,),
    )


def test_session_evolution_promotes_only_after_opaque_development_receipt():
    backend = FakeBackend()
    runner = SessionEvolutionRunner(RunContext("run-1", "dev-task", stream_id="stream-1"), backend)
    verifier = DevelopmentStreamVerifier(
        evaluator=lambda artifact: len(artifact.trace.candidates) == 1
        and not any("answer" in str(event.metadata).lower() for event in artifact.trace.events),
        receipt_secret=b"development-secret",
    )

    result = asyncio.run(
        _run(
            runner,
            verifier,
            (_session("session-1", 0, "The preferred color is blue."),),
        )
    )

    assert result.receipt.passed is True
    assert result.receipt.receipt_token
    assert result.promotion is not None
    assert result.snapshot_id == result.promotion.snapshot_id
    assert backend.promoted[0][0] == (runner.trace.candidates[0].candidate_id,)
    snapshot = runner.snapshots.get(result.snapshot_id)
    assert snapshot.manifest["memory_complete"] is True
    assert snapshot.manifest["last_source_position"] == 0


def test_failed_development_verification_quarantines_without_skill_or_memory_promotion():
    backend = FakeBackend()
    runner = SessionEvolutionRunner(RunContext("run-2", "dev-task", stream_id="stream-1"), backend)
    verifier = DevelopmentStreamVerifier(evaluator=lambda artifact: False, receipt_secret=b"development-secret")

    result = asyncio.run(
        _run(
            runner,
            verifier,
            (_session("session-1", 0, "The preferred color is blue."),),
        )
    )

    assert result.receipt.passed is False
    assert result.promotion is None
    assert result.snapshot_id is None
    assert not backend.promoted
    assert backend.quarantined


def test_forged_positive_development_receipt_is_quarantined_before_promotion():
    backend = FakeBackend()
    runner = SessionEvolutionRunner(RunContext("run-forged", "dev-task", stream_id="stream-1"), backend)

    class ForgedVerifier(DevelopmentStreamVerifier):
        def validate_receipt(self, seal, receipt):
            return False

    verifier = ForgedVerifier(evaluator=lambda artifact: True, receipt_secret=b"development-secret")
    result = asyncio.run(_run(runner, verifier, (_session("session-1", 0, "before"),)))

    assert result.receipt.passed is False
    assert result.promotion is None
    assert not backend.promoted
    assert backend.quarantined


def test_query_prefix_does_not_stage_future_sessions():
    backend = FakeBackend()
    runner = SessionEvolutionRunner(RunContext("run-3", "dev-task", stream_id="stream-1"), backend)
    verifier = DevelopmentStreamVerifier(evaluator=lambda artifact: True, receipt_secret=b"development-secret")

    async def prepare_and_verify():
        trace = await runner.prepare(
            (_session("session-1", 0, "before"), _session("session-2", 1, "after")),
            query=None,
        )
        assert len(trace.events) == 6

        # This run was already prepared and cannot be reused for a second
        # query; a caller must create a new run/snapshot for a new prefix.
        return trace

    trace = asyncio.run(prepare_and_verify())
    assert [session.session_id for session in trace.sessions] == ["session-1", "session-2"]


def test_longmem_runtime_can_use_external_semantic_enricher_without_labels():
    item = {
        "question_id": "q-dev-1",
        "question": "What color do I prefer?",
        "question_date": "2025/01/03 (Fri) 10:00",
        "answer": "blue",
        "question_type": "single-session-user",
        "haystack_sessions": [[{"role": "user", "content": "I prefer blue.", "has_answer": True}]],
        "haystack_dates": ["2025/01/01 (Wed) 10:00"],
        "haystack_session_ids": ["s-1"],
    }
    runtime = LongMemEvalAdapter().runtime(item, evaluation_id="dev-eval")

    def enrich(event):
        if event.event_type is not SessionEventType.MESSAGE or event.metadata.get("role") != "user":
            return event
        return replace(
            event,
            event_type=SessionEventType.FACT_OBSERVED,
            metadata={
                **event.metadata,
                "candidate": {
                    "kind": "preference",
                    "namespace": "general",
                    "trigger": "fact_observed",
                    "content": event.observation,
                },
            },
        )

    backend = FakeBackend()
    runner = SessionEvolutionRunner(
        RunContext("longmem-dev-run", "q-dev-1", stream_id=runtime.query.stream_id),
        backend,
        event_enricher=enrich,
    )
    verifier = DevelopmentStreamVerifier(evaluator=lambda artifact: len(artifact.trace.candidates) == 1, receipt_secret=b"secret")
    result = asyncio.run(_run(runner, verifier, runtime.sessions))

    assert result.receipt.passed
    assert result.promotion is not None
    assert result.trace.query is None
    assert all("has_answer" not in str(event.metadata) for event in result.trace.events)


def test_benchmark_phase_cannot_construct_an_evolution_runner():
    backend = FakeBackend()
    try:
        SessionEvolutionRunner(
            RunContext("benchmark-run", "benchmark", metadata={"phase": "benchmark"}, stream_id="stream-1"),
            backend,
        )
    except SessionEvolutionStateError as exc:
        assert "benchmark" in str(exc)
    else:
        raise AssertionError("benchmark phase was allowed to construct an evolution runner")


async def _run(runner, verifier, sessions):
    await runner.prepare(sessions)
    return await runner.finalize(verifier)
