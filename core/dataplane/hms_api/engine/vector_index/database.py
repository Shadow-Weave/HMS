"""Native database vector-index provider."""

from datetime import datetime

from ..search.tags import TagGroup, TagsMatch
from .base import VectorIndex, VectorIndexRecord, VectorSearchHit


class DatabaseVectorIndex(VectorIndex):
    """Marker provider that preserves HMS's existing database search path."""

    @property
    def provider_name(self) -> str:
        return "database"

    @property
    def is_external(self) -> bool:
        return False

    async def initialize(self, dimension: int) -> None:
        return None

    async def upsert(self, records: list[VectorIndexRecord]) -> None:
        return None

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
        return None

    async def delete_units(self, namespace: str, unit_ids: list[str]) -> None:
        return None

    async def delete_document(self, namespace: str, bank_id: str, document_id: str) -> None:
        return None

    async def delete_bank(self, namespace: str, bank_id: str, fact_type: str | None = None) -> None:
        return None

    async def delete_namespace(self, namespace: str) -> None:
        return None

    async def close(self) -> None:
        return None
