from __future__ import annotations

import asyncio

import pytest
from rsi_harness import (
    EvolutionSession,
    EvolutionStateError,
    LeakageGuard,
    LeakagePolicy,
    MemoryBackendCapabilities,
    MemoryCandidate,
    MemoryNamespace,
    PromotionResult,
    RunContext,
    VerifierReceipt,
)


class FakeBackend:
    capabilities = MemoryBackendCapabilities(promote=True, quarantine=True, immutable_snapshots=True)

    def __init__(self):
        self.staged = []
        self.promoted = []
        self.quarantined = []

    async def stage(self, candidate):
        self.staged.append(candidate)
        return f"backend:{candidate.candidate_id}"

    async def promote(self, candidate_ids, *, target_snapshot):
        self.promoted.append((tuple(candidate_ids), target_snapshot))
        return PromotionResult(target_snapshot, tuple(candidate_ids))

    async def quarantine(self, candidate_ids, reason):
        self.quarantined.append((tuple(candidate_ids), reason))


class ReceiptBackend(FakeBackend):
    async def promote_with_receipt(self, candidate_ids, *, target_snapshot, receipt_token, run_id, candidate_digest):
        self.promoted.append((tuple(candidate_ids), target_snapshot, receipt_token, run_id, candidate_digest))
        return PromotionResult(target_snapshot, tuple(candidate_ids))


def candidate(run_id="run-1", content="general recovery procedure"):
    return MemoryCandidate(
        candidate_id="candidate-1",
        kind="procedure",
        namespace=MemoryNamespace.GENERAL,
        trigger="a recurring error",
        content=content,
        source_run_id=run_id,
        evidence_refs=(f"trace://{run_id}/1",),
    )


def test_candidate_is_sealed_before_verifier_and_promoted_after_pass():
    backend = FakeBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    seal = session.seal()
    assert seal.candidate_ids == ("candidate-1",)
    with pytest.raises(EvolutionStateError, match="after run seal"):
        asyncio.run(session.stage(candidate()))

    result = asyncio.run(session.finalize(VerifierReceipt("run-1", True, "opaque-token")))
    assert result is not None
    assert backend.promoted


def test_receipt_promotion_uses_candidate_ids_not_backend_operation_ids():
    class ReceiptBackend(FakeBackend):
        async def promote_with_receipt(self, candidate_ids, **kwargs):
            self.promoted.append((tuple(candidate_ids), kwargs))
            return PromotionResult(kwargs["target_snapshot"], tuple(candidate_ids))

    backend = ReceiptBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    session.seal()
    result = asyncio.run(session.finalize(VerifierReceipt("run-1", True, "opaque-token")))
    assert result is not None
    assert backend.promoted[0][0] == ("candidate-1",)


def test_duplicate_candidate_id_cannot_be_staged_twice():
    backend = FakeBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    with pytest.raises(EvolutionStateError, match="already staged"):
        asyncio.run(session.stage(candidate()))


def test_failed_verifier_quarantines_without_promoting():
    backend = FakeBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    session.seal()
    result = asyncio.run(session.finalize(VerifierReceipt("run-1", False, "opaque-token")))
    assert result is None
    assert not backend.promoted
    assert backend.quarantined


def test_receipt_promotion_path_uses_persisted_seal_digest():
    backend = ReceiptBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    seal = session.seal()
    result = asyncio.run(session.finalize(VerifierReceipt("run-1", True, "opaque-token")))
    assert result is not None
    assert backend.promoted[0][3:] == ("run-1", seal.candidate_digest)


def test_benchmark_secret_is_rejected_before_backend_write():
    backend = FakeBackend()
    guard = LeakageGuard(LeakagePolicy(forbidden_literals=("gold patch: exact-answer",)))
    session = EvolutionSession(RunContext("run-1", "task-1"), backend, leakage_guard=guard)
    with pytest.raises(ValueError, match="benchmark_secret"):
        asyncio.run(session.stage(candidate(content="gold patch: exact-answer")))
    assert not backend.staged


def test_duplicate_candidate_id_is_rejected_before_backend_write():
    backend = FakeBackend()
    session = EvolutionSession(RunContext("run-1", "task-1"), backend)
    asyncio.run(session.stage(candidate()))
    with pytest.raises(EvolutionStateError, match="already staged"):
        asyncio.run(session.stage(candidate()))
