from __future__ import annotations

import asyncio
from pathlib import Path

from rsi_harness import (
    FileSkillLibrary,
    HarnessAssetKind,
    HarnessCandidate,
    InMemorySkillLibrary,
    RunContext,
    RunSeal,
    SkillEvolutionController,
    SkillLibraryError,
    VerifierReceipt,
)


class FixedVerifier:
    def __init__(self, passed: bool) -> None:
        self.passed = passed

    async def verify(self, run, seal: RunSeal):
        return VerifierReceipt(run.run_id, self.passed, f"opaque-{int(self.passed)}")


def _candidate(base_snapshot_id: str, run_id: str = "skill-run-1") -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id=f"skill-{run_id}",
        base_snapshot_id=base_snapshot_id,
        kind=HarnessAssetKind.RETRIEVAL_POLICY,
        patch={"session_expansion_weight": 0.45},
        source_run_id=run_id,
        evidence_refs=(f"dev-trace://{run_id}/1",),
    )


def test_skill_library_advances_only_after_attested_development_pass():
    library = InMemorySkillLibrary(initial_assets=({"asset_id": "baseline", "kind": "routing_policy"},))
    base = library.current_snapshot_id
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    run = RunContext("skill-run-1", "skill-cohort", harness_snapshot=base)

    result = asyncio.run(controller.evolve(run, _candidate(base), FixedVerifier(True)))

    assert result.receipt.passed
    assert result.promotion is not None
    assert result.current_snapshot_id == result.promotion.snapshot_id
    assert result.current_snapshot_id != base
    assert {asset["asset_id"] for asset in library.assets()} == {"baseline", "skill-skill-run-1"}
    assert library.current().manifest["skill_complete"] is True


def test_failed_skill_cohort_keeps_current_snapshot_and_audit_record():
    library = InMemorySkillLibrary()
    base = library.current_snapshot_id
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    run = RunContext("skill-run-2", "skill-cohort", harness_snapshot=base)

    result = asyncio.run(controller.evolve(run, _candidate(base, "skill-run-2"), FixedVerifier(False)))

    assert result.promotion is None
    assert result.current_snapshot_id == base
    assert len(controller.rejected) == 1


def test_skill_candidate_cannot_use_stale_base_snapshot():
    library = InMemorySkillLibrary()
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    run = RunContext("skill-run-3", "skill-cohort", harness_snapshot="stale")

    try:
        asyncio.run(controller.evolve(run, _candidate("stale", "skill-run-3"), FixedVerifier(True)))
    except Exception as exc:
        assert "stale snapshot" in str(exc) or "current skill snapshot" in str(exc)
    else:
        raise AssertionError("stale skill snapshot was accepted")


def test_benchmark_phase_cannot_update_skill_library():
    library = InMemorySkillLibrary()
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    base = library.current_snapshot_id
    run = RunContext("skill-benchmark", "benchmark", harness_snapshot=base, metadata={"phase": "benchmark"})
    try:
        asyncio.run(controller.evolve(run, _candidate(base, "skill-benchmark"), FixedVerifier(True)))
    except SkillLibraryError as exc:
        assert "benchmark" in str(exc)
    else:
        raise AssertionError("benchmark phase was allowed to update skills")


def test_file_skill_library_recovers_current_snapshot_and_lineage(tmp_path: Path):
    root = tmp_path / "skills"
    library = FileSkillLibrary(root, initial_assets=({"asset_id": "baseline", "kind": "routing_policy"},))
    base = library.current_snapshot_id
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    run = RunContext("skill-persist-1", "skill-cohort", harness_snapshot=base)

    result = asyncio.run(controller.evolve(run, _candidate(base, "skill-persist-1"), FixedVerifier(True)))
    assert result.promotion is not None
    promoted = result.current_snapshot_id

    restarted = FileSkillLibrary(root)
    assert restarted.current_snapshot_id == promoted
    assert {asset["asset_id"] for asset in restarted.assets()} == {
        "baseline",
        "skill-skill-persist-1",
    }
    assert restarted.current().manifest["skill_complete"] is True


def test_file_skill_library_failed_update_does_not_move_pointer(tmp_path: Path):
    root = tmp_path / "skills"
    library = FileSkillLibrary(root)
    base = library.current_snapshot_id
    controller = SkillEvolutionController(library, receipt_validator=lambda seal, receipt: True)
    run = RunContext("skill-persist-2", "skill-cohort", harness_snapshot=base)

    result = asyncio.run(controller.evolve(run, _candidate(base, "skill-persist-2"), FixedVerifier(False)))
    assert result.promotion is None
    assert FileSkillLibrary(root).current_snapshot_id == base
