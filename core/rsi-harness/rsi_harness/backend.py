"""Memory backend protocol and capability declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .models import (
    HarnessSnapshot,
    MemoryCandidate,
    MemoryHit,
    MemoryItem,
    MemoryNamespace,
    PromotionResult,
)


@dataclass(frozen=True, slots=True)
class MemoryBackendCapabilities:
    """Explicit capabilities prevent silent weakening of safety guarantees."""

    search: bool = True
    fetch: bool = True
    stage: bool = True
    promote: bool = False
    quarantine: bool = False
    invalidate: bool = False
    immutable_snapshots: bool = False


class MemoryBackend(Protocol):
    """Async contract implemented by any memory system adapter."""

    @property
    def capabilities(self) -> MemoryBackendCapabilities: ...

    async def snapshot(self, namespace: MemoryNamespace) -> HarnessSnapshot: ...

    async def search(
        self,
        namespace: MemoryNamespace,
        query: str,
        *,
        token_budget: int,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[MemoryHit]: ...

    async def fetch(self, memory_id: str, *, detail: str = "summary") -> MemoryItem: ...

    async def stage(self, candidate: MemoryCandidate) -> str: ...

    async def promote(self, candidate_ids: Sequence[str], *, target_snapshot: str) -> PromotionResult: ...

    async def quarantine(self, candidate_ids: Sequence[str], reason: str) -> None: ...

    async def invalidate(self, memory_ids: Sequence[str], reason: str) -> None: ...

    async def restore(self, snapshot_id: str) -> None: ...


class BackendCapabilityError(RuntimeError):
    """Raised instead of silently performing an unsafe best-effort operation."""


def require_capability(backend: MemoryBackend, capability: str) -> None:
    """Fail closed when a requested safety-critical capability is unavailable."""

    if not getattr(backend.capabilities, capability, False):
        raise BackendCapabilityError(f"memory backend does not provide capability: {capability}")
