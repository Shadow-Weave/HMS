"""P4 benchmark embargo and read-only frozen Harness view."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Mapping, Sequence

from .backend import BackendCapabilityError, MemoryBackend
from .models import HarnessSnapshot, MemoryHit, MemoryItem, MemoryNamespace


class BenchmarkAccessViolation(PermissionError):
    """Raised when a benchmark path attempts to mutate or escape its lease."""


@dataclass(frozen=True, slots=True)
class EvaluationLease:
    """Capability token for one read-only benchmark evaluation."""

    evaluation_id: str
    snapshot_id: str
    benchmark_name: str
    lease_token: str
    memory_snapshot_id: str | None = None
    skill_snapshot_id: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Report object intentionally kept outside the evolution APIs."""

    evaluation_id: str
    passed: bool
    metrics: Mapping[str, float]


class FrozenMemoryView:
    """Read-only, namespace-restricted view exposed to benchmark workers."""

    def __init__(self, backend: MemoryBackend, lease: EvaluationLease):
        if not backend.capabilities.immutable_snapshots:
            raise BackendCapabilityError("benchmark view requires immutable snapshot support")
        self._backend = backend
        self.lease = lease

    async def search(
        self,
        namespace: MemoryNamespace,
        query: str,
        *,
        token_budget: int,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[MemoryHit]:
        if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BenchmarkAccessViolation("benchmark view cannot read episode or candidate memory")
        memory_snapshot_id = self.lease.memory_snapshot_id or self.lease.snapshot_id
        hits = await self._backend.search(
            namespace,
            query,
            token_budget=token_budget,
            filters={**dict(filters or {}), "snapshot_id": memory_snapshot_id, "read_only": True},
        )
        # Do not trust an adapter to honor the namespace boundary implicitly.
        # A buggy backend returning an episode/candidate hit would otherwise
        # turn this read-only facade into a benchmark leakage channel.
        allowed = (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL)
        for hit in hits:
            if hit.namespace not in allowed:
                raise BenchmarkAccessViolation("backend returned episode or candidate memory")
        return hits

    async def fetch(self, memory_id: str, *, detail: str = "summary") -> MemoryItem:
        # ``MemoryBackend.fetch`` has no snapshot argument and therefore can
        # only read the live store.  Never fall back to it for benchmark use:
        # every fetch must carry the immutable snapshot capability explicitly.
        snapshot_fetch = getattr(self._backend, "fetch_snapshot", None)
        if snapshot_fetch is None:
            raise BackendCapabilityError("frozen backend does not provide snapshot fetch")
        memory_snapshot_id = self.lease.memory_snapshot_id or self.lease.snapshot_id
        item = await snapshot_fetch(memory_snapshot_id, memory_id, detail=detail)
        if item.namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BenchmarkAccessViolation("backend returned episode or candidate memory")
        return item

    async def stage(self, *args, **kwargs):
        raise BenchmarkAccessViolation("benchmark view is read-only")

    async def promote(self, *args, **kwargs):
        raise BenchmarkAccessViolation("benchmark view is read-only")

    async def quarantine(self, *args, **kwargs):
        raise BenchmarkAccessViolation("benchmark view is read-only")

    async def invalidate(self, *args, **kwargs):
        raise BenchmarkAccessViolation("benchmark view is read-only")


class BenchmarkFirewall:
    """Create isolated read-only leases after evolution has been sealed."""

    def __init__(self, *, benchmark_name: str):
        if not benchmark_name.strip():
            raise ValueError("benchmark_name must not be empty")
        self.benchmark_name = benchmark_name
        self._open_leases: dict[str, EvaluationLease] = {}

    def open(self, snapshot: HarnessSnapshot, *, evolution_sealed: bool) -> EvaluationLease:
        if not evolution_sealed:
            raise BenchmarkAccessViolation("freeze evolution before opening benchmark evaluation")
        if not snapshot.manifest.get("memory_complete", False):
            raise BenchmarkAccessViolation("snapshot completeness check is required before benchmark evaluation")
        snapshot_status = snapshot.manifest.get("status")
        if snapshot_status is not None and str(snapshot_status).lower() != "frozen":
            raise BenchmarkAccessViolation("benchmark evaluation requires a frozen snapshot")
        evaluation_id = secrets.token_hex(12)
        lease = EvaluationLease(
            evaluation_id=evaluation_id,
            snapshot_id=snapshot.snapshot_id,
            benchmark_name=self.benchmark_name,
            lease_token=secrets.token_urlsafe(32),
            memory_snapshot_id=str(snapshot.manifest.get("memory_snapshot_id") or snapshot.snapshot_id),
            skill_snapshot_id=(
                str(snapshot.manifest["skill_snapshot_id"])
                if snapshot.manifest.get("skill_snapshot_id")
                else None
            ),
        )
        self._open_leases[evaluation_id] = lease
        return lease

    def view(self, backend: MemoryBackend, lease: EvaluationLease) -> FrozenMemoryView:
        if self._open_leases.get(lease.evaluation_id) != lease:
            raise BenchmarkAccessViolation("unknown or expired benchmark lease")
        return FrozenMemoryView(backend, lease)

    def close(self, lease: EvaluationLease) -> None:
        self._open_leases.pop(lease.evaluation_id, None)
