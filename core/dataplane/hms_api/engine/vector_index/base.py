"""Semantic vector index provider contract.

The relational database remains the canonical data store. External providers
only hold a rebuildable projection used for dense candidate retrieval.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..search.tags import TagGroup, TagsMatch


@dataclass(frozen=True)
class VectorIndexRecord:
    """A canonical memory-unit projection stored in an external vector index."""

    id: str
    namespace: str
    bank_id: str
    fact_type: str
    embedding: list[float]
    document_id: str | None = None
    tags: list[str] | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class VectorSearchHit:
    """A dense candidate returned by a vector index before SQL hydration."""

    id: str
    fact_type: str
    similarity: float


def parse_embedding(value: Any) -> list[float]:
    """Convert database vector representations into a plain float list."""

    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Embedding string must contain a JSON array")
        return [float(item) for item in parsed]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def record_from_row(row: Any, namespace: str) -> VectorIndexRecord:
    """Build a vector-index record from a database result row."""

    updated_at = row.get("updated_at")
    if updated_at is not None and updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    tags = row.get("tags")
    if isinstance(tags, str):
        tags = json.loads(tags)

    return VectorIndexRecord(
        id=str(row["id"]),
        namespace=namespace,
        bank_id=str(row["bank_id"]),
        fact_type=str(row["fact_type"]),
        embedding=parse_embedding(row["embedding"]),
        document_id=str(row["document_id"]) if row.get("document_id") is not None else None,
        tags=list(tags) if tags else [],
        updated_at=updated_at,
    )


class VectorIndex(ABC):
    """Provider interface for dense semantic candidate indexes."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the configured provider name."""

    @property
    @abstractmethod
    def is_external(self) -> bool:
        """Return whether dense search is handled outside the canonical database."""

    @abstractmethod
    async def initialize(self, dimension: int) -> None:
        """Initialize provider resources and validate the embedding dimension."""

    @abstractmethod
    async def upsert(self, records: list[VectorIndexRecord]) -> None:
        """Insert or replace memory-unit projections."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        *,
        namespace: str,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list[VectorSearchHit]] | None:
        """Return dense candidates grouped by fact type.

        The native database provider returns ``None`` so HMS can preserve its
        optimized combined semantic/full-text SQL query.
        """

    @abstractmethod
    async def delete_units(self, namespace: str, unit_ids: list[str]) -> None:
        """Delete specific memory units from the projection."""

    @abstractmethod
    async def delete_document(self, namespace: str, bank_id: str, document_id: str) -> None:
        """Delete every projected memory unit for a document."""

    @abstractmethod
    async def delete_bank(self, namespace: str, bank_id: str, fact_type: str | None = None) -> None:
        """Delete projected memory units for a bank, optionally by fact type."""

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> None:
        """Delete every projected record for a database namespace."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
