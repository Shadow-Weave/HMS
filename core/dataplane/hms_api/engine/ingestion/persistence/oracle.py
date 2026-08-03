"""Oracle persistence adapters for the Retain ingestion pipeline."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from ...schema import fq_table_explicit
from ..adapters.postgres_fresh_ownership import FreshDocumentOwnershipConflict
from ..domain import ExistingChunkFingerprint
from .models import CommittedUnitBinding, ExistingDocument, OperationCheckpoint
from .postgres import (
    _COMMITTED_UNIT_IDS_FIELD,
    _COMMITTED_UNIT_IDS_VERSION,
    _committed_unit_ids_by_document,
    _json_object,
    _require_identifier,
    _string_array,
    _tags,
    _unit_id_tuple,
)


def _affected_rows(status: str) -> int:
    """Return the row count from an asyncpg-compatible command status."""

    if not isinstance(status, str):
        return 0
    try:
        return int(status.rsplit(maxsplit=1)[-1])
    except (IndexError, ValueError):
        return 0


class OracleDocumentOwnership:
    """Oracle row-lock and document-hash ownership adapter."""

    def __init__(self, *, schema: str | None = None) -> None:
        self._schema = schema

    async def prepare_first_window(self, connection: Any, *, bank_id: str, document_id: str) -> None:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        documents = fq_table_explicit("documents", self._schema)
        await connection.execute(
            f"""
            INSERT INTO {documents} (id, bank_id, original_text, content_hash)
            VALUES ($1, $2, '', '__pending__')
            ON CONFLICT (id, bank_id) DO NOTHING
            """,
            document_id,
            bank_id,
        )
        await connection.fetchval(
            f"""
            SELECT content_hash
            FROM {documents}
            WHERE id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            document_id,
            bank_id,
        )

    async def validate_later_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
        expected_content_hash: str,
    ) -> bool:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        _require_identifier(expected_content_hash, field_name="expected_content_hash")
        current_hash = await connection.fetchval(
            f"""
            SELECT content_hash
            FROM {fq_table_explicit("documents", self._schema)}
            WHERE id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            document_id,
            bank_id,
        )
        return current_hash is not None and current_hash == expected_content_hash

    async def validate_unhashed_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
    ) -> bool:
        """Lock an upgraded row and confirm that Oracle still sees no hash."""

        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        row = await connection.fetchrow(
            f"""
            SELECT content_hash
            FROM {fq_table_explicit("documents", self._schema)}
            WHERE id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            document_id,
            bank_id,
        )
        return row is not None and not row["content_hash"]

    async def transition_content_hash(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
        expected_content_hash: str,
        new_content_hash: str,
    ) -> bool:
        for field_name, value in (
            ("bank_id", bank_id),
            ("document_id", document_id),
            ("expected_content_hash", expected_content_hash),
            ("new_content_hash", new_content_hash),
        ):
            _require_identifier(value, field_name=field_name)
        status = await connection.execute(
            f"""
            UPDATE {fq_table_explicit("documents", self._schema)}
            SET content_hash = $1, updated_at = now()
            WHERE id = $2 AND bank_id = $3 AND content_hash = $4
            """,
            new_content_hash,
            document_id,
            bank_id,
            expected_content_hash,
        )
        return _affected_rows(status) == 1


class FreshOracleDocumentOwnership(OracleDocumentOwnership):
    """Atomically claim a document that was absent during Oracle preflight."""

    async def prepare_first_window(self, connection: Any, *, bank_id: str, document_id: str) -> None:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        documents = fq_table_explicit("documents", self._schema)
        status = await connection.execute(
            f"""
            INSERT INTO {documents} (id, bank_id, original_text, content_hash)
            VALUES ($1, $2, '', '__pending__')
            ON CONFLICT (id, bank_id) DO NOTHING
            """,
            document_id,
            bank_id,
        )
        if _affected_rows(status) != 1:
            raise FreshDocumentOwnershipConflict("The document is no longer fresh")

        locked = await connection.fetchval(
            f"""
            SELECT id
            FROM {documents}
            WHERE id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            document_id,
            bank_id,
        )
        if locked is None:  # pragma: no cover - same transaction inserted it
            raise RuntimeError("Fresh document ownership row disappeared inside its transaction")

    async def validate_later_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
        expected_content_hash: str,
    ) -> bool:
        del connection, bank_id, document_id, expected_content_hash
        raise RuntimeError("Fresh-document ownership does not support later full-write windows")

    async def validate_unhashed_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
    ) -> bool:
        del connection, bank_id, document_id
        raise RuntimeError("Fresh-document ownership does not support existing unhashed rows")


class OraclePlanningRepository:
    """Load the durable Oracle state required for Retain planning."""

    def __init__(self, connection: Any, *, schema: str | None = None) -> None:
        self._connection = connection
        self._schema = schema

    async def load_document(self, bank_id: str, document_id: str) -> ExistingDocument | None:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        row = await self._connection.fetchrow(
            f"""
            SELECT id, bank_id, original_text, content_hash, retain_params,
                   tags, created_at, updated_at
            FROM {fq_table_explicit("documents", self._schema)}
            WHERE id = $1 AND bank_id = $2
            """,
            document_id,
            bank_id,
        )
        if row is None:
            return None
        return ExistingDocument(
            document_id=str(row["id"]),
            bank_id=str(row["bank_id"]),
            original_text=row["original_text"] or "",
            content_hash=row["content_hash"],
            retain_params=_json_object(row["retain_params"]),
            tags=_tags(row["tags"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def load_chunks(
        self,
        bank_id: str,
        document_id: str,
    ) -> tuple[ExistingChunkFingerprint, ...]:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        rows = await self._connection.fetch(
            f"""
            SELECT chunk_id, chunk_index, content_hash
            FROM {fq_table_explicit("chunks", self._schema)}
            WHERE document_id = $1 AND bank_id = $2
            ORDER BY chunk_index
            """,
            document_id,
            bank_id,
        )
        return tuple(
            ExistingChunkFingerprint(
                chunk_id=str(row["chunk_id"]),
                chunk_index=int(row["chunk_index"]),
                content_hash=row["content_hash"],
            )
            for row in rows
        )

    async def load_document_unit_ids(self, bank_id: str, document_id: str) -> tuple[str, ...]:
        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        rows = await self._connection.fetch(
            f"""
            SELECT id
            FROM {fq_table_explicit("memory_units", self._schema)}
            WHERE bank_id = $1 AND document_id = $2
            ORDER BY created_at, id
            """,
            bank_id,
            document_id,
        )
        unit_ids = tuple(str(row["id"]) for row in rows)
        if any(not unit_id for unit_id in unit_ids):  # pragma: no cover - database PK invariant
            raise ValueError("memory_units recovery query returned an empty ID")
        return unit_ids

    async def load_document_unit_bindings(
        self,
        bank_id: str,
        document_id: str,
        *,
        expected_unit_ids: tuple[str, ...] | None = None,
    ) -> tuple[CommittedUnitBinding, ...]:
        """Load bindings without PostgreSQL array-position ordering."""

        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        if expected_unit_ids is not None:
            expected_unit_ids = _unit_id_tuple(expected_unit_ids, field_name="expected_unit_ids")
            if not expected_unit_ids:
                return ()

        rows = await self._connection.fetch(
            f"""
            SELECT mu.id AS unit_id, c.chunk_index
            FROM {fq_table_explicit("memory_units", self._schema)} mu
            LEFT JOIN {fq_table_explicit("chunks", self._schema)} c
              ON c.chunk_id = mu.chunk_id
             AND c.bank_id = mu.bank_id
             AND c.document_id = mu.document_id
            WHERE mu.bank_id = $1 AND mu.document_id = $2
            ORDER BY c.chunk_index NULLS LAST, mu.created_at, mu.id
            """,
            bank_id,
            document_id,
        )

        bindings_by_unit_id: dict[str, CommittedUnitBinding] = {}
        for row in rows:
            chunk_index = row["chunk_index"]
            binding = CommittedUnitBinding(
                unit_id=str(row["unit_id"]),
                chunk_index=(None if chunk_index is None else int(chunk_index)),
            )
            if binding.unit_id in bindings_by_unit_id:
                raise ValueError(f"memory_units recovery query returned duplicate unit ID: {binding.unit_id!r}")
            bindings_by_unit_id[binding.unit_id] = binding

        if expected_unit_ids is None:
            return tuple(bindings_by_unit_id.values())
        missing = tuple(unit_id for unit_id in expected_unit_ids if unit_id not in bindings_by_unit_id)
        if missing:
            raise ValueError(
                f"checkpoint unit IDs are missing or do not belong to the requested bank/document: {missing!r}"
            )
        return tuple(bindings_by_unit_id[unit_id] for unit_id in expected_unit_ids)


class OracleCheckpointStore:
    """Maintain Retain checkpoints in an Oracle JSON CLOB under a row lock."""

    def __init__(self, connection: Any, *, schema: str | None = None) -> None:
        self._connection = connection
        self._schema = schema

    @property
    def _table(self) -> str:
        return fq_table_explicit("async_operations", self._schema)

    async def recover_document_ids(self, operation_id: str) -> tuple[str, ...]:
        return (await self.recover(operation_id)).document_ids

    async def recover(self, operation_id: str) -> OperationCheckpoint:
        parsed_operation_id = uuid.UUID(operation_id)
        row = await self._connection.fetchrow(
            f"SELECT result_metadata FROM {self._table} WHERE operation_id = $1",
            parsed_operation_id,
        )
        if not row or not row["result_metadata"]:
            return OperationCheckpoint()
        metadata = _json_object(row["result_metadata"])
        unscoped_facts_committed = metadata.get("facts_committed", False)
        if not isinstance(unscoped_facts_committed, bool):
            raise ValueError("async operation facts_committed must be a bool")
        return OperationCheckpoint(
            document_ids=_string_array(metadata.get("document_ids"), field_name="document_ids"),
            core_committed_document_ids=_string_array(
                metadata.get("facts_committed_document_ids"),
                field_name="facts_committed_document_ids",
            ),
            final_ann_pending_document_ids=_string_array(
                metadata.get("final_ann_pending_document_ids"),
                field_name="final_ann_pending_document_ids",
            ),
            committed_unit_ids_by_document=_committed_unit_ids_by_document(metadata),
            unscoped_facts_committed=unscoped_facts_committed,
        )

    async def _locked_metadata(self, operation_id: uuid.UUID) -> dict[str, Any]:
        row = await self._connection.fetchrow(
            f"""
            SELECT result_metadata
            FROM {self._table}
            WHERE operation_id = $1
            FOR UPDATE
            """,
            operation_id,
        )
        if row is None:
            raise RuntimeError("Async operation disappeared before checkpoint update")
        return _json_object(row["result_metadata"])

    async def _write_metadata(self, operation_id: uuid.UUID, metadata: Mapping[str, Any]) -> None:
        status = await self._connection.execute(
            f"""
            UPDATE {self._table}
            SET result_metadata = $1, updated_at = now()
            WHERE operation_id = $2
            """,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            operation_id,
        )
        if _affected_rows(status) != 1:
            raise RuntimeError("Async operation disappeared before checkpoint update")

    @staticmethod
    def _append_unique(metadata: dict[str, Any], field_name: str, value: str) -> None:
        values = list(_string_array(metadata.get(field_name), field_name=field_name))
        if value not in values:
            values.append(value)
        metadata[field_name] = values

    async def record_document_id(self, operation_id: str, document_id: str) -> None:
        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        async with self._connection.transaction():
            metadata = await self._locked_metadata(parsed_operation_id)
            self._append_unique(metadata, "document_ids", document_id)
            await self._write_metadata(parsed_operation_id, metadata)

    async def record_core_committed(
        self,
        operation_id: str,
        document_id: str,
        *,
        unit_ids: tuple[str, ...],
        requires_final_ann: bool,
    ) -> None:
        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        unit_ids = _unit_id_tuple(unit_ids, field_name="unit_ids")
        if not isinstance(requires_final_ann, bool):
            raise TypeError("requires_final_ann must be a bool")

        async with self._connection.transaction():
            metadata = await self._locked_metadata(parsed_operation_id)
            # Validate an existing versioned payload before changing any field.
            _committed_unit_ids_by_document(metadata)
            self._append_unique(metadata, "document_ids", document_id)
            self._append_unique(metadata, "facts_committed_document_ids", document_id)
            metadata["facts_committed"] = True
            metadata["unit_ids_count"] = len(unit_ids)

            existing_payload = metadata.get(_COMMITTED_UNIT_IDS_FIELD)
            if existing_payload is None:
                payload: dict[str, Any] = {
                    "version": _COMMITTED_UNIT_IDS_VERSION,
                    "documents": {},
                }
            else:
                payload = dict(existing_payload)
                payload["documents"] = dict(payload["documents"])
            payload["documents"][document_id] = list(unit_ids)
            metadata[_COMMITTED_UNIT_IDS_FIELD] = payload

            pending = list(
                _string_array(
                    metadata.get("final_ann_pending_document_ids"),
                    field_name="final_ann_pending_document_ids",
                )
            )
            if requires_final_ann and unit_ids and document_id not in pending:
                pending.append(document_id)
            metadata["final_ann_pending_document_ids"] = pending
            await self._write_metadata(parsed_operation_id, metadata)

    async def record_final_ann_completed(self, operation_id: str, document_id: str) -> None:
        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        async with self._connection.transaction():
            metadata = await self._locked_metadata(parsed_operation_id)
            pending = _string_array(
                metadata.get("final_ann_pending_document_ids"),
                field_name="final_ann_pending_document_ids",
            )
            metadata["final_ann_pending_document_ids"] = [value for value in pending if value != document_id]
            await self._write_metadata(parsed_operation_id, metadata)

    async def clear_provider_batch(self, operation_id: str) -> None:
        parsed_operation_id = uuid.UUID(operation_id)
        async with self._connection.transaction():
            metadata = await self._locked_metadata(parsed_operation_id)
            for field_name in ("batch_id", "batch_provider", "chunk_count"):
                metadata.pop(field_name, None)
            await self._write_metadata(parsed_operation_id, metadata)


__all__ = [
    "FreshOracleDocumentOwnership",
    "OracleCheckpointStore",
    "OracleDocumentOwnership",
    "OraclePlanningRepository",
]
