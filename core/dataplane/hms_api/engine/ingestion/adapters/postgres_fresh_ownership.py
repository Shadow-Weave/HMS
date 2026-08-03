"""Strict PostgreSQL ownership gate for fresh-document writes."""

from __future__ import annotations

from typing import Any

from ...schema import fq_table_explicit


class FreshDocumentOwnershipConflict(RuntimeError):
    """A document appeared after Retain's read-side fresh check."""


class FreshPostgresDocumentOwnership:
    """Claim a previously absent document without replacing concurrent data.

    The general full-write ownership adapter intentionally permits replacement.
    This stricter adapter uses ``INSERT ... RETURNING`` to turn the
    preflight/write race into a typed failure instead of allowing another
    writer's just-committed document to be replaced.
    """

    def __init__(self, *, schema: str | None = None) -> None:
        self._schema = schema

    async def prepare_first_window(self, connection: Any, *, bank_id: str, document_id: str) -> None:
        if not isinstance(bank_id, str) or not bank_id:
            raise ValueError("bank_id must be a non-empty string")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")

        documents = fq_table_explicit("documents", self._schema)
        claimed = await connection.fetchval(
            f"""
            INSERT INTO {documents} (id, bank_id, original_text, content_hash)
            VALUES ($1, $2, '', '__pending__')
            ON CONFLICT (id, bank_id) DO NOTHING
            RETURNING id
            """,
            document_id,
            bank_id,
        )
        if claimed is None:
            raise FreshDocumentOwnershipConflict(f"Document {document_id!r} in bank {bank_id!r} is no longer fresh")

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
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
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
