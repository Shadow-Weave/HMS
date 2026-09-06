from __future__ import annotations

import pytest
from rsi_harness import (
    HarnessAssetKind,
    HarnessCandidate,
    HarnessEvolutionSession,
    RunContext,
    VerifierReceipt,
)


def make_candidate(run_id="run-1", base="harness-base", patch=None):
    return HarnessCandidate(
        candidate_id="harness-candidate-1",
        base_snapshot_id=base,
        kind=HarnessAssetKind.RETRIEVAL_POLICY,
        patch=patch or {"risk_event_budget": 900},
        source_run_id=run_id,
        evidence_refs=("trace://run-1/1",),
    )


def test_harness_candidate_promotes_to_new_snapshot_after_pass():
    session = HarnessEvolutionSession(
        RunContext("run-1", "task-1"),
        base_snapshot_id="harness-base",
        receipt_validator=lambda seal, receipt: receipt.receipt_token == "opaque" and seal.run_id == receipt.run_id,
    )
    session.propose(make_candidate())
    seal = session.seal()
    assert seal.candidate_ids == ("harness-candidate-1",)
    promotion = session.finalize(VerifierReceipt("run-1", True, "opaque"))
    assert promotion is not None
    assert promotion.candidate_id == "harness-candidate-1"
    assert promotion.snapshot_id.startswith("harness-")


def test_harness_candidate_cannot_contain_benchmark_fields():
    session = HarnessEvolutionSession(RunContext("run-1", "task-1"), base_snapshot_id="harness-base")
    with pytest.raises(ValueError, match="forbidden"):
        session.propose(make_candidate(patch={"hidden_test": "expected"}))


def test_promoted_snapshot_manifest_is_immutable():
    session = HarnessEvolutionSession(
        RunContext("run-1", "task-1"),
        base_snapshot_id="harness-base",
        receipt_validator=lambda seal, receipt: True,
    )
    session.propose(make_candidate())
    session.seal()
    promotion = session.finalize(VerifierReceipt("run-1", True, "opaque"))
    assert promotion is not None
    snapshot = session.snapshots.get(promotion.snapshot_id)
    with pytest.raises(TypeError):
        snapshot.manifest["new_value"] = "must fail"


def test_promoted_snapshot_nested_manifest_is_immutable():
    session = HarnessEvolutionSession(
        RunContext("run-1", "task-1"),
        base_snapshot_id="harness-base",
        receipt_validator=lambda seal, receipt: True,
    )
    session.propose(make_candidate(patch={"retrieval": {"budgets": [100, 200]}}))
    session.seal()
    promotion = session.finalize(VerifierReceipt("run-1", True, "opaque"))
    assert promotion is not None
    snapshot = session.snapshots.get(promotion.snapshot_id)
    with pytest.raises(TypeError):
        snapshot.manifest["patch"]["retrieval"]["budgets"] = (1,)
    with pytest.raises(TypeError):
        snapshot.manifest["patch"]["retrieval"]["budgets"][0] = 1


def test_harness_promotion_rejects_unattested_passed_receipt():
    session = HarnessEvolutionSession(RunContext("run-1", "task-1"), base_snapshot_id="harness-base")
    session.propose(make_candidate())
    session.seal()
    with pytest.raises(RuntimeError, match="attestation"):
        session.finalize(VerifierReceipt("run-1", True, "forged"))
