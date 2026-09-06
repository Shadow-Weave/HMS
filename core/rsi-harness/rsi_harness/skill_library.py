"""Memory-system-neutral skill snapshot registry.

Skills are declarative Harness assets.  A memory system may materialize them
as prompts, retrieval policies, directives, or mental models, but the RSI
boundary only publishes immutable, receipt-attested snapshots.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .evolution import DevelopmentVerifier, RunSeal
from .harness_evolution import HarnessCandidate, HarnessEvolutionSession, HarnessPromotion
from .models import HarnessSnapshot, RunContext, VerifierReceipt
from .snapshots import InMemorySnapshotStore, JsonSnapshotStore, SnapshotStore


class SkillLibraryError(RuntimeError):
    """Raised when a skill library update violates snapshot lineage."""


@dataclass(frozen=True, slots=True)
class SkillEvolutionResult:
    """Opaque result of one development cohort skill update."""

    base_snapshot_id: str
    receipt: VerifierReceipt
    promotion: HarnessPromotion | None
    current_snapshot_id: str


class SkillLibrary:
    """Protocol-like base class for a versioned skill registry."""

    @property
    def current_snapshot_id(self) -> str:
        raise NotImplementedError

    @property
    def snapshot_store(self) -> SnapshotStore:
        raise NotImplementedError

    def current(self) -> HarnessSnapshot:
        raise NotImplementedError

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        raise NotImplementedError

    def advance(self, snapshot_id: str) -> HarnessSnapshot:
        raise NotImplementedError

    def assets(self, snapshot_id: str | None = None) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError


@dataclass(slots=True)
class InMemorySkillLibrary(SkillLibrary):
    """Reference registry used by local development and adapter tests."""

    snapshots: SnapshotStore = field(default_factory=InMemorySnapshotStore)
    _current_snapshot_id: str = field(init=False)

    def __init__(
        self,
        snapshots: SnapshotStore | None = None,
        *,
        initial_assets: Sequence[Mapping[str, Any]] = (),
        initial_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.snapshots = snapshots or InMemorySnapshotStore()
        manifest = {
            "skill_complete": True,
            "generation": 0,
            "assets": tuple(dict(asset) for asset in initial_assets),
            **dict(initial_manifest or {}),
        }
        initial = self.snapshots.create(parent_snapshot_id=None, manifest=manifest)
        self._current_snapshot_id = initial.snapshot_id

    @property
    def current_snapshot_id(self) -> str:
        return self._current_snapshot_id

    @property
    def snapshot_store(self) -> SnapshotStore:
        return self.snapshots

    def current(self) -> HarnessSnapshot:
        return self.get(self._current_snapshot_id)

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        return self.snapshots.get(snapshot_id)

    def advance(self, snapshot_id: str) -> HarnessSnapshot:
        snapshot = self.get(snapshot_id)
        if snapshot.parent_snapshot_id != self._current_snapshot_id:
            raise SkillLibraryError(
                "skill snapshot must be a direct child of the current snapshot: "
                f"{snapshot.parent_snapshot_id!r} != {self._current_snapshot_id!r}"
            )
        if not snapshot.manifest.get("skill_complete", False):
            raise SkillLibraryError("cannot activate an incomplete skill snapshot")
        self._current_snapshot_id = snapshot.snapshot_id
        return snapshot

    def assets(self, snapshot_id: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Resolve declarative assets across the immutable snapshot lineage."""

        return _resolve_skill_assets(self.snapshots, snapshot_id or self._current_snapshot_id)


def _resolve_skill_assets(store: SnapshotStore, snapshot_id: str) -> tuple[Mapping[str, Any], ...]:
    target = store.get(snapshot_id)
    lineage: list[HarnessSnapshot] = []
    while target is not None:
        lineage.append(target)
        if target.parent_snapshot_id is None:
            break
        target = store.get(target.parent_snapshot_id)
    resolved: dict[str, Mapping[str, Any]] = {}
    for snapshot in reversed(lineage):
        base_assets = snapshot.manifest.get("assets", ())
        if isinstance(base_assets, Sequence) and not isinstance(base_assets, (str, bytes)):
            for index, asset in enumerate(base_assets):
                if isinstance(asset, Mapping):
                    key = str(asset.get("asset_id") or asset.get("kind") or index)
                    resolved[key] = dict(asset)
        asset_kind = snapshot.manifest.get("asset_kind")
        patch = snapshot.manifest.get("patch")
        if asset_kind and isinstance(patch, Mapping):
            candidate_id = str(snapshot.manifest.get("candidate_id") or asset_kind)
            resolved[candidate_id] = {
                "asset_id": candidate_id,
                "kind": str(asset_kind),
                "patch": dict(patch),
                "snapshot_id": snapshot.snapshot_id,
            }
    return tuple(resolved.values())


@dataclass(slots=True)
class FileSkillLibrary(SkillLibrary):
    """Persistent skill registry with an atomic current-snapshot pointer."""

    root: Path
    snapshots: JsonSnapshotStore = field(init=False)
    _current_snapshot_id: str = field(init=False)

    def __init__(
        self,
        root: str | Path,
        *,
        initial_assets: Sequence[Mapping[str, Any]] = (),
        initial_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = JsonSnapshotStore(self.root / "snapshots")
        pointer = self.root / "current.json"
        if pointer.exists():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            self._current_snapshot_id = str(payload["snapshot_id"])
            self.snapshots.get(self._current_snapshot_id)
            return
        manifest = {
            "skill_complete": True,
            "generation": 0,
            "assets": tuple(dict(asset) for asset in initial_assets),
            **dict(initial_manifest or {}),
        }
        initial = self.snapshots.create(parent_snapshot_id=None, manifest=manifest)
        self._current_snapshot_id = initial.snapshot_id
        self._write_pointer(pointer)

    @property
    def current_snapshot_id(self) -> str:
        return self._current_snapshot_id

    @property
    def snapshot_store(self) -> SnapshotStore:
        return self.snapshots

    def current(self) -> HarnessSnapshot:
        return self.get(self._current_snapshot_id)

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        return self.snapshots.get(snapshot_id)

    def advance(self, snapshot_id: str) -> HarnessSnapshot:
        snapshot = self.get(snapshot_id)
        if snapshot.parent_snapshot_id != self._current_snapshot_id:
            raise SkillLibraryError("skill snapshot is not a child of the current snapshot")
        if not snapshot.manifest.get("skill_complete", False):
            raise SkillLibraryError("cannot activate an incomplete skill snapshot")
        self._current_snapshot_id = snapshot.snapshot_id
        self._write_pointer(self.root / "current.json")
        return snapshot

    def assets(self, snapshot_id: str | None = None) -> tuple[Mapping[str, Any], ...]:
        return _resolve_skill_assets(self.snapshots, snapshot_id or self._current_snapshot_id)

    def _write_pointer(self, pointer: Path) -> None:
        temporary = pointer.with_name(f".{pointer.name}.tmp")
        temporary.write_text(
            json.dumps({"snapshot_id": self._current_snapshot_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, pointer)


ReceiptValidator = Callable[[RunSeal, VerifierReceipt], bool]


@dataclass(slots=True)
class SkillEvolutionController:
    """Promote one validated Harness skill candidate per development cohort."""

    library: SkillLibrary
    receipt_validator: ReceiptValidator
    rejected: list[Mapping[str, Any]] = field(default_factory=list)

    async def evolve(
        self,
        run: RunContext,
        candidate: HarnessCandidate,
        verifier: DevelopmentVerifier,
    ) -> SkillEvolutionResult:
        phase = str(run.metadata.get("phase", "")).strip().lower()
        if phase in {"benchmark", "evaluation", "test"}:
            raise SkillLibraryError("skill evolution is disabled for benchmark/evaluation runs")
        base_snapshot_id = self.library.current_snapshot_id
        if candidate.base_snapshot_id != base_snapshot_id:
            raise SkillLibraryError("skill candidate is based on a stale snapshot")
        if run.harness_snapshot not in (None, base_snapshot_id):
            raise SkillLibraryError("run harness_snapshot does not match the current skill snapshot")

        session = HarnessEvolutionSession(
            run,
            base_snapshot_id=base_snapshot_id,
            snapshots=self.library.snapshot_store,
            receipt_validator=self.receipt_validator,
            snapshot_manifest={
                "skill_complete": True,
                "generation": int(self.library.current().manifest.get("generation", 0)) + 1,
            },
        )
        session.propose(candidate)
        seal = session.seal()
        receipt = await verifier.verify(run, seal)
        promotion = session.finalize(receipt)
        if promotion is None:
            self.rejected.append(
                {
                    "run_id": run.run_id,
                    "candidate_id": candidate.candidate_id,
                    "base_snapshot_id": base_snapshot_id,
                    "receipt": "opaque",
                }
            )
        else:
            self.library.advance(promotion.snapshot_id)
        return SkillEvolutionResult(
            base_snapshot_id=base_snapshot_id,
            receipt=receipt,
            promotion=promotion,
            current_snapshot_id=self.library.current_snapshot_id,
        )
