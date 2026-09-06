"""P1 candidate lifecycle: stage, seal, verify, and promote."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

from .backend import MemoryBackend, require_capability
from .leakage import LeakageGuard
from .models import MemoryCandidate, PromotionResult, RunContext, VerifierReceipt
from .snapshots import InMemorySnapshotStore, SnapshotStore


class EvolutionStateError(RuntimeError):
    """Raised when the candidate lifecycle is used out of order."""


@dataclass(frozen=True, slots=True)
class RunSeal:
    run_id: str
    candidate_ids: tuple[str, ...]
    candidate_digest: str


class DevelopmentVerifier:
    """Verifier contract that exposes only an opaque binary receipt."""

    async def verify(self, run: RunContext, seal: RunSeal) -> VerifierReceipt:
        raise NotImplementedError


class EvolutionSession:
    """Fail-closed state machine for one development run."""

    def __init__(
        self,
        run: RunContext,
        backend: MemoryBackend,
        *,
        leakage_guard: LeakageGuard | None = None,
        snapshots: SnapshotStore | None = None,
        snapshot_manifest: dict[str, object] | None = None,
    ) -> None:
        self.run = run
        self.backend = backend
        self.leakage_guard = leakage_guard or LeakageGuard()
        self.snapshots = snapshots or InMemorySnapshotStore()
        self.snapshot_manifest = dict(snapshot_manifest or {})
        self._staged_backend_ids: list[str] = []
        self._candidate_ids: list[str] = []
        self._sealed = False
        self._finalized = False
        self._seal: RunSeal | None = None
        self._last_snapshot_id: str | None = None

    @property
    def staged_candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._candidate_ids)

    @property
    def last_snapshot_id(self) -> str | None:
        """Return the promoted snapshot created by this run, if any."""

        return self._last_snapshot_id

    async def stage(self, candidate: MemoryCandidate) -> str:
        if self._sealed:
            raise EvolutionStateError("cannot stage a candidate after run seal")
        if self._finalized:
            raise EvolutionStateError("cannot stage a candidate after finalization")
        if candidate.source_run_id != self.run.run_id:
            raise EvolutionStateError("candidate source_run_id does not match the active run")
        if candidate.candidate_id in self._candidate_ids:
            raise EvolutionStateError("candidate_id is already staged in this run")
        self.leakage_guard.require_accepted(candidate)
        backend_id = await self.backend.stage(candidate)
        self._candidate_ids.append(candidate.candidate_id)
        self._staged_backend_ids.append(backend_id)
        return backend_id

    def seal(self) -> RunSeal:
        if self._finalized:
            raise EvolutionStateError("cannot seal a finalized run")
        if self._sealed:
            raise EvolutionStateError("run is already sealed")
        digest_input = "\n".join(self._candidate_ids).encode("utf-8")
        digest = hashlib.sha256(digest_input).hexdigest()
        self._sealed = True
        self._seal = RunSeal(self.run.run_id, tuple(self._candidate_ids), digest)
        return self._seal

    async def finalize(self, receipt: VerifierReceipt) -> PromotionResult | None:
        if not self._sealed:
            raise EvolutionStateError("seal candidates before verifier finalization")
        if self._finalized:
            raise EvolutionStateError("run is already finalized")
        if receipt.run_id != self.run.run_id:
            raise EvolutionStateError("verifier receipt belongs to a different run")
        if self._seal is None:
            raise EvolutionStateError("run seal is missing")
        self._finalized = True

        if not receipt.passed or not self._staged_backend_ids:
            if self._staged_backend_ids and self.backend.capabilities.quarantine:
                await self.backend.quarantine(self._staged_backend_ids, "development verifier did not pass")
            return None

        require_capability(self.backend, "promote")
        snapshot = self.snapshots.create(
            parent_snapshot_id=self.run.harness_snapshot,
            manifest={
                "run_id": self.run.run_id,
                "candidate_ids": tuple(self._candidate_ids),
                "backend_ids": tuple(self._staged_backend_ids),
                "verifier_receipt": "opaque",
                **self.snapshot_manifest,
            },
        )
        self._last_snapshot_id = snapshot.snapshot_id
        promote_with_receipt = getattr(self.backend, "promote_with_receipt", None)
        if promote_with_receipt is not None:
            promote_kwargs = {
                "target_snapshot": snapshot.snapshot_id,
                "receipt_token": receipt.receipt_token,
                "run_id": self.run.run_id,
                "candidate_digest": self._seal.candidate_digest,
            }
            if "parent_snapshot_id" in inspect.signature(promote_with_receipt).parameters:
                promote_kwargs["parent_snapshot_id"] = self.run.harness_snapshot
            return await promote_with_receipt(
                # HMS promotes by candidate identity.  stage() may return an
                # operation ID, which is not a valid candidate ID.
                self._candidate_ids,
                **promote_kwargs,
            )
        return await self.backend.promote(self._candidate_ids, target_snapshot=snapshot.snapshot_id)
