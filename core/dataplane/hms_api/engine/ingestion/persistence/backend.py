"""Database adapter selection for the Retain ingestion service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..adapters.postgres_fresh_ownership import FreshPostgresDocumentOwnership
from .operation_fence import OperationActivityFence
from .oracle import (
    FreshOracleDocumentOwnership,
    OracleCheckpointStore,
    OracleDocumentOwnership,
    OraclePlanningRepository,
)
from .postgres import (
    PostgresCheckpointStore,
    PostgresDocumentOwnership,
    PostgresPlanningRepository,
)


@dataclass(frozen=True, slots=True)
class RetainBackendAdapters:
    """Factories and transaction semantics for one supported database."""

    backend_type: str

    def planning_repository(self, connection: Any, *, schema: str | None = None) -> Any:
        if self.backend_type == "oracle":
            return OraclePlanningRepository(connection, schema=schema)
        return PostgresPlanningRepository(connection, schema=schema)

    def checkpoint_store(self, connection: Any, *, schema: str | None = None) -> Any:
        if self.backend_type == "oracle":
            return OracleCheckpointStore(connection, schema=schema)
        return PostgresCheckpointStore(connection, schema=schema)

    def document_ownership(self, *, schema: str | None = None, fresh: bool = False) -> Any:
        if self.backend_type == "oracle":
            if fresh:
                return FreshOracleDocumentOwnership(schema=schema)
            return OracleDocumentOwnership(schema=schema)
        if fresh:
            return FreshPostgresDocumentOwnership(schema=schema)
        return PostgresDocumentOwnership(schema=schema)

    def operation_activity_fence(
        self,
        operation_id: str | None,
        *,
        schema: str | None = None,
    ) -> OperationActivityFence | None:
        """Build a database-neutral fence for a tracked core write."""

        if operation_id is None:
            return None
        return OperationActivityFence(operation_id, schema=schema)

    @asynccontextmanager
    async def planning_snapshot(self, connection: Any):
        """Open a backend-native read-only snapshot for Retain planning."""

        if self.backend_type == "oracle":
            # Oracle requires SET TRANSACTION to be the first transaction
            # statement. OracleConnection.transaction() starts with SAVEPOINT,
            # so the outer acquired connection owns this read-only transaction.
            await connection.execute("SET TRANSACTION READ ONLY")
            yield
            return

        async with connection.transaction():
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield


def retain_backend_adapters(backend_type: str) -> RetainBackendAdapters:
    """Return adapters for an explicitly supported database backend."""

    normalized = backend_type.strip().lower() if isinstance(backend_type, str) else ""
    if normalized not in {"postgresql", "oracle"}:
        raise ValueError(f"Unsupported Retain database backend: {backend_type!r}")
    return RetainBackendAdapters(normalized)


__all__ = ["RetainBackendAdapters", "retain_backend_adapters"]
