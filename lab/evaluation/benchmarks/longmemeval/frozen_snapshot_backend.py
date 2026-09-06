"""Read-only PostgreSQL backend for frozen RSI benchmark snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import urlparse

from rsi_harness.backend import BackendCapabilityError, MemoryBackendCapabilities
from rsi_harness.models import HarnessSnapshot, MemoryHit, MemoryItem, MemoryNamespace, PromotionResult


class FrozenSnapshotDatabaseError(RuntimeError):
    """Raised when the benchmark database is not a frozen snapshot database."""


@dataclass(slots=True)
class PostgresFrozenSnapshotBackend:
    """Minimal snapshot-only backend with no live HMS table access.

    The role used by ``database_url`` needs SELECT only on
    ``rsi_harness_snapshots`` and ``rsi_snapshot_memory_units``.  Every search
    requires an explicit frozen snapshot ID; no fallback to ``memory_units``
    or ``documents`` exists in this adapter.
    """

    database_url: str
    bank_id: str
    schema: str = "public"
    connect_timeout: float = 15.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.database_url)
        if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
            raise FrozenSnapshotDatabaseError("database_url must be a PostgreSQL URL")
        username = (parsed.username or "").lower()
        if not any(marker in username for marker in ("readonly", "read_only", "benchmark")):
            raise FrozenSnapshotDatabaseError("database role must be dedicated to a readonly benchmark")
        if not self.bank_id.strip():
            raise FrozenSnapshotDatabaseError("bank_id must not be empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.schema):
            raise FrozenSnapshotDatabaseError("schema must be a simple PostgreSQL identifier")
        if self.connect_timeout <= 0:
            raise FrozenSnapshotDatabaseError("connect_timeout must be positive")

    @property
    def capabilities(self) -> MemoryBackendCapabilities:
        return MemoryBackendCapabilities(
            search=True,
            fetch=True,
            stage=False,
            promote=False,
            quarantine=False,
            invalidate=False,
            immutable_snapshots=True,
        )

    def _table(self, name: str) -> str:
        return f'"{self.schema}"."{name}"'

    async def _connect(self):
        try:
            import asyncpg

            return await asyncpg.connect(self.database_url, timeout=self.connect_timeout)
        except Exception as exc:  # pragma: no cover - driver/network dependent
            raise FrozenSnapshotDatabaseError("could not connect to readonly benchmark database") from exc

    async def _require_frozen_snapshot(self, conn, snapshot_id: str) -> Mapping[str, object]:
        row = await conn.fetchrow(
            f"SELECT status, manifest FROM {self._table('rsi_harness_snapshots')} "
            "WHERE bank_id = $1 AND snapshot_id = $2",
            self.bank_id,
            snapshot_id,
        )
        if row is None:
            raise FrozenSnapshotDatabaseError(f"unknown snapshot: {snapshot_id}")
        if row["status"] != "frozen":
            raise FrozenSnapshotDatabaseError("benchmark may read frozen snapshots only")
        manifest = dict(row["manifest"] or {})
        if not manifest.get("memory_complete", False):
            raise FrozenSnapshotDatabaseError("snapshot completeness check failed")
        return manifest

    async def snapshot(self, namespace: MemoryNamespace) -> HarnessSnapshot:
        del namespace
        raise BackendCapabilityError("benchmark backend cannot create snapshots")

    async def search(
        self,
        namespace: MemoryNamespace,
        query: str,
        *,
        token_budget: int,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[MemoryHit]:
        if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BackendCapabilityError("benchmark backend cannot read episode or candidate memory")
        if not query.strip():
            return ()
        filters = filters or {}
        if not filters.get("read_only"):
            raise BackendCapabilityError("frozen search requires read_only=true")
        snapshot_id = str(filters.get("snapshot_id") or "").strip()
        if not snapshot_id:
            raise BackendCapabilityError("frozen search requires an explicit snapshot_id")
        limit = max(1, min(int(token_budget), 100))
        terms = [term for term in re.findall(r"[A-Za-z0-9_'-]{3,}", query.lower())[:12]]
        if not terms:
            terms = [query.strip()]
        clauses = []
        params: list[object] = [self.bank_id, snapshot_id, namespace.value]
        for term in terms:
            params.append(term)
            clauses.append(f"LOWER(text) LIKE '%' || ${len(params)} || '%'")
        params.append(limit)
        conn = await self._connect()
        try:
            await self._require_frozen_snapshot(conn, snapshot_id)
            rows = await conn.fetch(
                f"""
                SELECT memory_id, namespace, fact_type, text, metadata
                FROM {self._table('rsi_snapshot_memory_units')}
                WHERE bank_id = $1 AND snapshot_id = $2 AND namespace = $3
                  AND ({' OR '.join(clauses)})
                ORDER BY memory_id
                LIMIT ${len(params)}
                """,
                *params,
            )
        finally:
            await conn.close()
        return tuple(
            MemoryHit(
                memory_id=str(row["memory_id"]),
                text=str(row["text"]),
                score=1.0,
                namespace=MemoryNamespace(str(row["namespace"])),
                metadata={**dict(row["metadata"] or {}), "fact_type": row["fact_type"]},
            )
            for row in rows
        )

    async def fetch(self, memory_id: str, *, detail: str = "summary") -> MemoryItem:
        del memory_id, detail
        raise BackendCapabilityError("benchmark backend requires fetch_snapshot")

    async def fetch_snapshot(self, snapshot_id: str, memory_id: str, *, detail: str = "summary") -> MemoryItem:
        del detail
        conn = await self._connect()
        try:
            await self._require_frozen_snapshot(conn, snapshot_id)
            row = await conn.fetchrow(
                f"""
                SELECT memory_id, namespace, text, metadata
                FROM {self._table('rsi_snapshot_memory_units')}
                WHERE bank_id = $1 AND snapshot_id = $2 AND memory_id = $3
                """,
                self.bank_id,
                snapshot_id,
                memory_id,
            )
        finally:
            await conn.close()
        if row is None:
            raise FrozenSnapshotDatabaseError(f"memory is not in frozen snapshot: {memory_id}")
        namespace = MemoryNamespace(str(row["namespace"]))
        if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BackendCapabilityError("snapshot returned an unsafe namespace")
        return MemoryItem(
            memory_id=str(row["memory_id"]),
            text=str(row["text"]),
            namespace=namespace,
            metadata=dict(row["metadata"] or {}),
            version=snapshot_id,
        )

    async def stage(self, candidate):
        del candidate
        raise BackendCapabilityError("benchmark backend is read-only")

    async def promote(self, candidate_ids, *, target_snapshot: str) -> PromotionResult:
        del candidate_ids, target_snapshot
        raise BackendCapabilityError("benchmark backend is read-only")

    async def quarantine(self, candidate_ids, reason: str) -> None:
        del candidate_ids, reason
        raise BackendCapabilityError("benchmark backend is read-only")

    async def invalidate(self, memory_ids, reason: str) -> None:
        del memory_ids, reason
        raise BackendCapabilityError("benchmark backend is read-only")

    async def restore(self, snapshot_id: str) -> None:
        del snapshot_id
        raise BackendCapabilityError("benchmark backend is read-only")

    async def get_snapshot(self, snapshot_id: str) -> HarnessSnapshot:
        """Load a frozen snapshot manifest through the readonly connection."""

        conn = await self._connect()
        try:
            manifest = await self._require_frozen_snapshot(conn, snapshot_id)
            row = await conn.fetchrow(
                f"SELECT parent_snapshot_id FROM {self._table('rsi_harness_snapshots')} "
                "WHERE bank_id = $1 AND snapshot_id = $2",
                self.bank_id,
                snapshot_id,
            )
        finally:
            await conn.close()
        return HarnessSnapshot(snapshot_id, row["parent_snapshot_id"] if row else None, manifest=manifest)
