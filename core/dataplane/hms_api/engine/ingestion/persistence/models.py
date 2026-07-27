"""Database-neutral records returned by Retain persistence ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class CommittedUnitBinding:
    """A committed memory unit and the source chunk position it belongs to.

    ``chunk_index`` is optional because durable memory units can have no chunk
    association. Recovery must preserve those units instead of silently
    dropping them from the public result.
    """

    unit_id: str
    chunk_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise ValueError("unit_id must be a non-empty string")
        if self.chunk_index is None:
            return
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer or None")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")


@dataclass(frozen=True, slots=True)
class ExistingDocument:
    document_id: str
    bank_id: str
    original_text: str
    content_hash: str | None
    retain_params: dict[str, Any]
    tags: tuple[str, ...]
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class OperationCheckpoint:
    """Durable async-operation state needed to resume Retain after a crash."""

    document_ids: tuple[str, ...] = ()
    core_committed_document_ids: tuple[str, ...] = ()
    final_ann_pending_document_ids: tuple[str, ...] = ()
    committed_unit_ids_by_document: tuple[tuple[str, tuple[str, ...]], ...] = ()
    unscoped_facts_committed: bool = False

    def __post_init__(self) -> None:
        for field_name, values in (
            ("document_ids", self.document_ids),
            ("core_committed_document_ids", self.core_committed_document_ids),
            ("final_ann_pending_document_ids", self.final_ann_pending_document_ids),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value for value in values):
                raise TypeError(f"{field_name} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")

        if not isinstance(self.committed_unit_ids_by_document, tuple):
            raise TypeError("committed_unit_ids_by_document must be a tuple")
        seen_document_ids: set[str] = set()
        for binding in self.committed_unit_ids_by_document:
            if not isinstance(binding, tuple) or len(binding) != 2:
                raise TypeError("committed_unit_ids_by_document entries must be (document_id, unit_ids) tuples")
            document_id, unit_ids = binding
            if not isinstance(document_id, str) or not document_id:
                raise TypeError("committed unit document IDs must be non-empty strings")
            if document_id in seen_document_ids:
                raise ValueError("committed_unit_ids_by_document must not contain duplicate document IDs")
            seen_document_ids.add(document_id)
            if not isinstance(unit_ids, tuple) or any(
                not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids
            ):
                raise TypeError("committed unit IDs must be tuples of non-empty strings")
            if len(unit_ids) != len(set(unit_ids)):
                raise ValueError(f"committed unit IDs for document {document_id!r} must not contain duplicates")
        if not isinstance(self.unscoped_facts_committed, bool):
            raise TypeError("unscoped_facts_committed must be a bool")

    def is_core_committed(self, document_id: str) -> bool:
        """Apply the single-document fallback checkpoint rule."""

        if document_id in self.core_committed_document_ids:
            return True
        return (
            self.unscoped_facts_committed
            and not self.core_committed_document_ids
            and (not self.document_ids or self.document_ids == (document_id,))
        )

    def unit_ids_for_document(self, document_id: str) -> tuple[str, ...] | None:
        """Return exact operation-local IDs, or ``None`` for an unscoped checkpoint.

        An empty tuple is a known value: that document committed successfully
        but this operation produced no units.  ``None`` means the versioned
        mapping was absent or did not contain the document, so callers must
        fail closed unless they have another operation-local proof.
        """

        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")
        for committed_document_id, unit_ids in self.committed_unit_ids_by_document:
            if committed_document_id == document_id:
                return unit_ids
        return None
