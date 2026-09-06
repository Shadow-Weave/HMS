"""P3 recursive evolution of safe Harness policy/configuration assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol

from .evolution import EvolutionStateError, RunSeal
from .leakage import LeakageGuard
from .models import RunContext, VerifierReceipt
from .snapshots import InMemorySnapshotStore, SnapshotStore


class HarnessAssetKind(StrEnum):
    PROMPT_TEMPLATE = "prompt_template"
    RETRIEVAL_POLICY = "retrieval_policy"
    EVENT_SCHEMA = "event_schema"
    PROMOTION_POLICY = "promotion_policy"
    ROUTING_POLICY = "routing_policy"


@dataclass(frozen=True, slots=True)
class HarnessCandidate:
    """A declarative Harness change; executable code and permissions are excluded."""

    candidate_id: str
    base_snapshot_id: str
    kind: HarnessAssetKind
    patch: Mapping[str, Any]
    source_run_id: str
    evidence_refs: tuple[str, ...]
    created_before_verifier: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.base_snapshot_id.strip():
            raise ValueError("Harness candidate IDs must not be empty")
        if not self.source_run_id.strip():
            raise ValueError("Harness candidate source_run_id must not be empty")
        if not self.patch:
            raise ValueError("Harness candidate patch must not be empty")
        if not self.evidence_refs:
            raise ValueError("Harness candidate must contain evidence references")
        if not self.created_before_verifier:
            raise ValueError("Harness candidate must be created before verifier output")
        forbidden = {"benchmark", "verifier_output", "hidden_test", "gold_patch", "permissions", "executable"}
        if forbidden.intersection(str(key).lower() for key in self.patch):
            raise ValueError("Harness candidate contains a forbidden mutable field")


class HarnessMutationPolicy(Protocol):
    def validate(self, candidate: HarnessCandidate) -> None: ...


class DefaultHarnessMutationPolicy:
    """Conservative policy for declarative, bounded Harness changes."""

    max_patch_chars = 16_000

    def validate(self, candidate: HarnessCandidate) -> None:
        serialized = json.dumps(dict(candidate.patch), sort_keys=True, default=str)
        if len(serialized) > self.max_patch_chars:
            raise ValueError("Harness candidate patch is too large")
        if any(str(key).startswith("__") for key in candidate.patch):
            raise ValueError("private Harness fields cannot be mutated")


@dataclass(frozen=True, slots=True)
class HarnessPromotion:
    snapshot_id: str
    candidate_id: str


class HarnessEvolutionSession:
    """Version and promote Harness candidates after development validation."""

    def __init__(
        self,
        run: RunContext,
        *,
        base_snapshot_id: str,
        snapshots: SnapshotStore | None = None,
        mutation_policy: HarnessMutationPolicy | None = None,
        leakage_guard: LeakageGuard | None = None,
        receipt_validator: Callable[[RunSeal, VerifierReceipt], bool] | None = None,
        snapshot_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.run = run
        self.base_snapshot_id = base_snapshot_id
        self.snapshots = snapshots or InMemorySnapshotStore()
        self.mutation_policy = mutation_policy or DefaultHarnessMutationPolicy()
        self.leakage_guard = leakage_guard or LeakageGuard()
        # Harness changes are executable policy in practice.  A passed receipt
        # must therefore be attested by the deployment's verifier boundary;
        # accepting only ``passed=True`` would let a caller self-promote.
        self.receipt_validator = receipt_validator
        self.snapshot_manifest = dict(snapshot_manifest or {})
        self._candidate: HarnessCandidate | None = None
        self._sealed = False
        self._finalized = False
        self._seal: RunSeal | None = None

    def propose(self, candidate: HarnessCandidate) -> None:
        if self._sealed or self._finalized:
            raise EvolutionStateError("cannot propose Harness changes after sealing")
        if candidate.source_run_id != self.run.run_id:
            raise EvolutionStateError("Harness candidate source_run_id does not match active run")
        if candidate.base_snapshot_id != self.base_snapshot_id:
            raise EvolutionStateError("Harness candidate is based on a different snapshot")
        self.mutation_policy.validate(candidate)
        report = self.leakage_guard.inspect_texts(
            [
                json.dumps(dict(candidate.patch), sort_keys=True, default=str),
                json.dumps(dict(candidate.metadata), sort_keys=True, default=str),
                *candidate.evidence_refs,
            ]
        )
        if not report.accepted:
            raise ValueError(f"Harness candidate rejected as {report.classification.value}: {report.findings}")
        if self._candidate is not None:
            raise EvolutionStateError("one Harness candidate is allowed per evolution run")
        self._candidate = candidate

    def seal(self) -> RunSeal:
        if self._sealed or self._finalized:
            raise EvolutionStateError("Harness evolution run is already sealed or finalized")
        self._sealed = True
        candidate_id = self._candidate.candidate_id if self._candidate else "none"
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        self._seal = RunSeal(self.run.run_id, (candidate_id,), digest)
        return self._seal

    def finalize(self, receipt: VerifierReceipt) -> HarnessPromotion | None:
        if not self._sealed:
            raise EvolutionStateError("seal Harness candidate before verification")
        if self._finalized:
            raise EvolutionStateError("Harness evolution run is already finalized")
        if receipt.run_id != self.run.run_id:
            raise EvolutionStateError("verifier receipt belongs to a different run")
        if self._seal is None:
            raise EvolutionStateError("Harness run seal is missing")
        self._finalized = True
        if not receipt.passed or self._candidate is None:
            return None
        if self.receipt_validator is None or not self.receipt_validator(self._seal, receipt):
            raise EvolutionStateError("a valid verifier receipt attestation is required for Harness promotion")

        candidate = self._candidate
        snapshot = self.snapshots.create(
            parent_snapshot_id=self.base_snapshot_id,
            manifest={
                "asset_kind": candidate.kind.value,
                "candidate_id": candidate.candidate_id,
                "patch": dict(candidate.patch),
                "source_run_id": candidate.source_run_id,
                "evidence_refs": candidate.evidence_refs,
                "verifier_receipt": "opaque",
                **self.snapshot_manifest,
            },
        )
        return HarnessPromotion(snapshot.snapshot_id, candidate.candidate_id)
