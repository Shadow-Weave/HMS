"""PostgreSQL planning and checkpoint adapters for Retain."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from ...schema import fq_table_explicit
from ..domain import ExistingChunkFingerprint
from .models import CommittedUnitBinding, ExistingDocument, OperationCheckpoint

_COMMITTED_UNIT_IDS_FIELD = "committed_unit_ids_v1"
_COMMITTED_UNIT_IDS_VERSION = 1


def _require_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected a JSON object from PostgreSQL, got {type(value).__name__}")


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected document tags to be an array of strings")
    return tuple(value)


def _string_array(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"async operation {field_name} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"async operation {field_name} must not contain duplicates")
    return tuple(value)


def _unit_id_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if any(not isinstance(unit_id, str) or not unit_id for unit_id in value):
        raise ValueError(f"{field_name} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return value


def _committed_unit_ids_by_document(metadata: Mapping[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if _COMMITTED_UNIT_IDS_FIELD not in metadata:
        return ()

    payload = metadata[_COMMITTED_UNIT_IDS_FIELD]
    if not isinstance(payload, Mapping):
        raise ValueError(f"async operation {_COMMITTED_UNIT_IDS_FIELD} must be an object")
    if set(payload) != {"version", "documents"}:
        raise ValueError(f"async operation {_COMMITTED_UNIT_IDS_FIELD} must contain exactly version and documents")
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != _COMMITTED_UNIT_IDS_VERSION:
        raise ValueError(f"async operation {_COMMITTED_UNIT_IDS_FIELD}.version must be {_COMMITTED_UNIT_IDS_VERSION}")
    documents = payload["documents"]
    if not isinstance(documents, Mapping):
        raise ValueError(f"async operation {_COMMITTED_UNIT_IDS_FIELD}.documents must be an object")

    parsed: list[tuple[str, tuple[str, ...]]] = []
    for document_id, unit_ids in documents.items():
        _require_identifier(document_id, field_name=f"{_COMMITTED_UNIT_IDS_FIELD} document_id")
        parsed.append(
            (
                document_id,
                _string_array(
                    unit_ids,
                    field_name=f"{_COMMITTED_UNIT_IDS_FIELD}.documents[{document_id!r}]",
                ),
            )
        )
    return tuple(sorted(parsed, key=lambda item: item[0]))


class PostgresDocumentOwnership:
    """PostgreSQL row-lock adapter for full-retain write windows.

    The adapter is intentionally connection-agnostic: the caller supplies the
    connection already enlisted in the core write transaction.  Every table
    reference is built from the explicit request schema rather than ambient
    schema context.
    """

    def __init__(self, *, schema: str | None = None) -> None:
        self._schema = schema

    async def prepare_first_window(self, connection: Any, *, bank_id: str, document_id: str) -> None:
        """Ensure a lockable row exists, then lock it for first-window tracking."""

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
        """Lock an existing row and report whether this request still owns it.

        A missing row, a NULL hash, or a different hash means ownership was
        lost.  Later windows never recreate the row: doing so could let an old
        producer resume after another request deleted or replaced the document.
        """

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
        """Lock an upgraded row only while its content hash is absent."""

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
        """Atomically move a locked document between ownership hash states."""

        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        _require_identifier(expected_content_hash, field_name="expected_content_hash")
        _require_identifier(new_content_hash, field_name="new_content_hash")
        updated = await connection.fetchval(
            f"""
            UPDATE {fq_table_explicit("documents", self._schema)}
            SET content_hash = $1, updated_at = now()
            WHERE id = $2 AND bank_id = $3 AND content_hash = $4
            RETURNING id
            """,
            new_content_hash,
            document_id,
            bank_id,
            expected_content_hash,
        )
        return updated is not None


class PostgresPlanningRepository:
    """Read document fingerprints through an already-scoped connection."""

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
        """Load committed unit IDs deterministically for crash recovery."""

        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        rows = await self._connection.fetch(
            f"""
            SELECT id::text AS id
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
        """Load committed unit IDs together with their durable chunk positions.

        The left join is deliberate: memory units may have no
        ``chunk_id`` (or may reference a chunk that no longer exists), and such
        units still belong to the recovered document result.
        """

        _require_identifier(bank_id, field_name="bank_id")
        _require_identifier(document_id, field_name="document_id")
        if expected_unit_ids is not None:
            expected_unit_ids = _unit_id_tuple(expected_unit_ids, field_name="expected_unit_ids")
            if not expected_unit_ids:
                return ()

        expected_filter = ""
        order_by = "c.chunk_index NULLS LAST, mu.created_at, mu.id"
        query_args: tuple[Any, ...] = (bank_id, document_id)
        if expected_unit_ids is not None:
            expected_filter = "AND mu.id::text = ANY($3::text[])"
            order_by = "array_position($3::text[], mu.id::text)"
            query_args = (bank_id, document_id, expected_unit_ids)
        rows = await self._connection.fetch(
            f"""
            SELECT mu.id::text AS unit_id, c.chunk_index
            FROM {fq_table_explicit("memory_units", self._schema)} AS mu
            LEFT JOIN {fq_table_explicit("chunks", self._schema)} AS c
              ON c.chunk_id = mu.chunk_id
             AND c.bank_id = mu.bank_id
             AND c.document_id = mu.document_id
            WHERE mu.bank_id = $1 AND mu.document_id = $2
              {expected_filter}
            ORDER BY {order_by}
            """,
            *query_args,
        )

        bindings_by_unit_id: dict[str, CommittedUnitBinding] = {}
        for row in rows:
            binding = CommittedUnitBinding(
                unit_id=row["unit_id"],
                chunk_index=row["chunk_index"],
            )
            if binding.unit_id in bindings_by_unit_id:
                raise ValueError(f"memory_units recovery query returned duplicate unit ID: {binding.unit_id!r}")
            bindings_by_unit_id[binding.unit_id] = binding

        if expected_unit_ids is None:
            return tuple(bindings_by_unit_id.values())

        unexpected_unit_ids = set(bindings_by_unit_id).difference(expected_unit_ids)
        if unexpected_unit_ids:  # pragma: no cover - SQL predicate invariant
            raise ValueError(
                f"memory_units recovery query returned unexpected unit IDs: {sorted(unexpected_unit_ids)!r}"
            )
        missing_unit_ids = tuple(unit_id for unit_id in expected_unit_ids if unit_id not in bindings_by_unit_id)
        if missing_unit_ids:
            raise ValueError(
                f"checkpoint unit IDs are missing or do not belong to the requested bank/document: {missing_unit_ids!r}"
            )
        return tuple(bindings_by_unit_id[unit_id] for unit_id in expected_unit_ids)


class PostgresCheckpointStore:
    """Read and idempotently update async operation document checkpoints."""

    def __init__(self, connection: Any, *, schema: str | None = None) -> None:
        self._connection = connection
        self._schema = schema

    async def recover_document_ids(self, operation_id: str) -> tuple[str, ...]:
        return (await self.recover(operation_id)).document_ids

    async def recover(self, operation_id: str) -> OperationCheckpoint:
        parsed_operation_id = uuid.UUID(operation_id)
        row = await self._connection.fetchrow(
            f"""
            SELECT result_metadata
            FROM {fq_table_explicit("async_operations", self._schema)}
            WHERE operation_id = $1
            """,
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

    async def record_document_id(self, operation_id: str, document_id: str) -> None:
        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        await self._connection.execute(
            f"""
            UPDATE {fq_table_explicit("async_operations", self._schema)}
            SET result_metadata = jsonb_set(
                COALESCE(result_metadata, '{{}}'::jsonb),
                '{{document_ids}}',
                CASE
                    WHEN COALESCE(result_metadata->'document_ids', '[]'::jsonb) @> $1::jsonb
                        THEN result_metadata->'document_ids'
                    ELSE COALESCE(result_metadata->'document_ids', '[]'::jsonb) || $1::jsonb
                END,
                true
            ),
            updated_at = now()
            WHERE operation_id = $2
            """,
            json.dumps([document_id]),
            parsed_operation_id,
        )

    async def record_core_committed(
        self,
        operation_id: str,
        document_id: str,
        *,
        unit_ids: tuple[str, ...],
        requires_final_ann: bool,
    ) -> None:
        """Atomically checkpoint a document from inside its core write transaction."""

        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        unit_ids = _unit_id_tuple(unit_ids, field_name="unit_ids")
        if not isinstance(requires_final_ann, bool):
            raise TypeError("requires_final_ann must be a bool")
        updated_operation_id = await self._connection.fetchval(
            f"""
            UPDATE {fq_table_explicit("async_operations", self._schema)}
            SET result_metadata = jsonb_set(
                jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            COALESCE(result_metadata, '{{}}'::jsonb) || $1::jsonb,
                            '{{document_ids}}',
                            CASE
                                WHEN COALESCE(result_metadata->'document_ids', '[]'::jsonb) @> $2::jsonb
                                    THEN result_metadata->'document_ids'
                                ELSE COALESCE(result_metadata->'document_ids', '[]'::jsonb) || $2::jsonb
                            END,
                            true
                        ),
                        '{{facts_committed_document_ids}}',
                        CASE
                            WHEN COALESCE(result_metadata->'facts_committed_document_ids', '[]'::jsonb) @> $2::jsonb
                                THEN result_metadata->'facts_committed_document_ids'
                            ELSE COALESCE(result_metadata->'facts_committed_document_ids', '[]'::jsonb) || $2::jsonb
                        END,
                        true
                    ),
                    '{{{_COMMITTED_UNIT_IDS_FIELD}}}',
                    jsonb_set(
                        COALESCE(
                            result_metadata->'{_COMMITTED_UNIT_IDS_FIELD}',
                            '{{"version": {_COMMITTED_UNIT_IDS_VERSION}, "documents": {{}}}}'::jsonb
                        ),
                        ARRAY['documents', $3::text],
                        $4::jsonb,
                        true
                    ),
                    true
                ),
                '{{final_ann_pending_document_ids}}',
                CASE
                    WHEN NOT $5::boolean
                        THEN COALESCE(result_metadata->'final_ann_pending_document_ids', '[]'::jsonb)
                    WHEN COALESCE(result_metadata->'final_ann_pending_document_ids', '[]'::jsonb) @> $2::jsonb
                        THEN result_metadata->'final_ann_pending_document_ids'
                    ELSE COALESCE(result_metadata->'final_ann_pending_document_ids', '[]'::jsonb) || $2::jsonb
                END,
                true
            ),
            updated_at = now()
            WHERE operation_id = $6
              AND (
                  result_metadata->'{_COMMITTED_UNIT_IDS_FIELD}' IS NULL
                  OR (
                      jsonb_typeof(result_metadata->'{_COMMITTED_UNIT_IDS_FIELD}') = 'object'
                      AND result_metadata->'{_COMMITTED_UNIT_IDS_FIELD}'->'version' =
                          '{_COMMITTED_UNIT_IDS_VERSION}'::jsonb
                      AND jsonb_typeof(result_metadata->'{_COMMITTED_UNIT_IDS_FIELD}'->'documents') = 'object'
                  )
              )
            RETURNING operation_id
            """,
            json.dumps({"facts_committed": True, "unit_ids_count": len(unit_ids)}),
            json.dumps([document_id]),
            document_id,
            json.dumps(unit_ids),
            requires_final_ann and bool(unit_ids),
            parsed_operation_id,
        )
        if updated_operation_id is None:
            raise RuntimeError(
                f"Async operation {operation_id} disappeared or had an invalid unit-ID checkpoint before core commit"
            )

    async def record_final_ann_completed(self, operation_id: str, document_id: str) -> None:
        """Clear an idempotent post-commit ANN recovery marker after an attempt."""

        parsed_operation_id = uuid.UUID(operation_id)
        _require_identifier(document_id, field_name="document_id")
        updated_operation_id = await self._connection.fetchval(
            f"""
            UPDATE {fq_table_explicit("async_operations", self._schema)}
            SET result_metadata = jsonb_set(
                COALESCE(result_metadata, '{{}}'::jsonb),
                '{{final_ann_pending_document_ids}}',
                COALESCE(
                    (
                        SELECT jsonb_agg(value)
                        FROM jsonb_array_elements(
                            COALESCE(result_metadata->'final_ann_pending_document_ids', '[]'::jsonb)
                        ) AS pending(value)
                        WHERE value <> to_jsonb($1::text)
                    ),
                    '[]'::jsonb
                ),
                true
            ),
            updated_at = now()
            WHERE operation_id = $2
            RETURNING operation_id
            """,
            document_id,
            parsed_operation_id,
        )
        if updated_operation_id is None:
            raise RuntimeError(f"Async operation {operation_id} disappeared before final ANN checkpoint")

    async def clear_provider_batch(self, operation_id: str) -> None:
        """Retire one completed provider Batch job before another extraction window."""

        parsed_operation_id = uuid.UUID(operation_id)
        updated_operation_id = await self._connection.fetchval(
            f"""
            UPDATE {fq_table_explicit("async_operations", self._schema)}
            SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb)
                                  - 'batch_id'
                                  - 'batch_provider'
                                  - 'chunk_count',
                updated_at = now()
            WHERE operation_id = $1
            RETURNING operation_id
            """,
            parsed_operation_id,
        )
        if updated_operation_id is None:
            raise RuntimeError(f"Async operation {operation_id} disappeared before provider Batch cleanup")
