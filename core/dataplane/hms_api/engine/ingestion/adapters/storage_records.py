"""Conversions from ingestion domain objects to durable storage records."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ...retain.types import ChunkMetadata, ExtractedFact, RetainContent
from ..domain import ChunkPlan, ContentItem, EventDateState, thaw_json
from ..extraction import build_content_position_map
from ..projection import MemoryRecord, to_processed_fact


def content_to_storage(item: ContentItem) -> RetainContent:
    """Snapshot one immutable content item as the current mutable DTO."""

    metadata = thaw_json(item.metadata)
    if not isinstance(metadata, dict):
        raise TypeError("ContentItem.metadata must thaw to an object")

    entities: list[dict[str, Any]] = []
    for index, frozen_entity in enumerate(item.entities):
        entity = thaw_json(frozen_entity)
        if not isinstance(entity, dict):
            raise TypeError(f"ContentItem.entities[{index}] must thaw to an object")
        entities.append(entity)

    observation_scopes: str | list[list[str]] | None
    if isinstance(item.observation_scopes, tuple):
        observation_scopes = [list(scope) for scope in item.observation_scopes]
    else:
        observation_scopes = item.observation_scopes

    return RetainContent(
        content=item.content,
        context=item.context,
        event_date=item.event_date.value,
        metadata=metadata,
        entities=entities,
        tags=list(item.tags),
        observation_scopes=observation_scopes,
    )


def compute_document_hash(combined_content: str) -> str:
    """Hash normalized document text for durable change tracking."""

    if not isinstance(combined_content, str):
        raise TypeError("combined_content must be a string")
    from ...retain.fact_extraction import _sanitize_text

    sanitized = _sanitize_text(combined_content) or ""
    return hashlib.sha256(sanitized.encode()).hexdigest()


def retain_document_metadata(items: Sequence[ContentItem]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the document metadata snapshot from normalized content.

    Defaulted timestamps are deliberately omitted: ``event_date`` is recorded
    in ``retain_params`` only when the caller supplied a truthy value, even
    though extraction itself receives a default timestamp.
    """

    if not items:
        return {}, ()

    first = items[0]
    retain_params: dict[str, Any] = {}
    if first.context:
        retain_params["context"] = first.context
    if first.event_date.state is EventDateState.EXPLICIT and first.event_date.value is not None:
        retain_params["event_date"] = first.event_date.value.isoformat()
    metadata = thaw_json(first.metadata)
    if not isinstance(metadata, dict):
        raise TypeError("ContentItem.metadata must thaw to an object")
    if metadata:
        retain_params["metadata"] = metadata

    seen: set[str] = set()
    tags: list[str] = []
    for item in items:
        for tag in item.tags:
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return retain_params, tuple(tags)


def chunks_to_storage(
    chunks: Sequence[ChunkPlan],
    items: Sequence[ContentItem],
    records: Sequence[MemoryRecord],
) -> tuple[ChunkMetadata, ...]:
    """Create chunk DTOs with fact counts derived by stable chunk key."""

    positions = build_content_position_map(items)
    fact_counts = Counter(record.chunk_key for record in records)
    known_keys = {chunk.chunk_key for chunk in chunks}
    unknown_keys = sorted(set(fact_counts) - known_keys)
    if unknown_keys:
        raise ValueError(f"Projected records reference unknown chunk keys: {unknown_keys!r}")

    result: list[ChunkMetadata] = []
    for chunk in chunks:
        try:
            content_index = positions[chunk.source_index]
        except KeyError as exc:  # pragma: no cover - extraction validates this first
            raise ValueError(f"Missing content position for source_index={chunk.source_index!r}") from exc
        result.append(
            ChunkMetadata(
                chunk_text=chunk.text,
                fact_count=fact_counts[chunk.chunk_key],
                content_index=content_index,
                chunk_index=chunk.global_index,
            )
        )
    return tuple(result)


def record_to_extracted_fact(record: MemoryRecord, *, content_index: int) -> ExtractedFact:
    """Convert a projected record into the raw DTO required by Phase 2."""

    if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
        raise ValueError("content_index must be a non-negative integer")
    metadata = thaw_json(record.metadata)
    if not isinstance(metadata, dict):
        raise TypeError("MemoryRecord.metadata must thaw to an object")
    if isinstance(record.observation_scopes, tuple):
        observation_scopes = [list(scope) for scope in record.observation_scopes]
    else:
        observation_scopes = record.observation_scopes

    return ExtractedFact(
        fact_text=record.text,
        fact_type=record.fact_type,
        entities=list(record.entity_mentions),
        occurred_start=record.occurred_start,
        occurred_end=record.occurred_end,
        where=getattr(record, "where", None),
        causal_relations=[],
        content_index=content_index,
        chunk_index=record.global_index,
        context=record.context,
        mentioned_at=record.mentioned_at,
        metadata=metadata,
        tags=list(record.tags),
        observation_scopes=observation_scopes,
    )


def record_to_processed_fact(
    record: MemoryRecord,
    *,
    document_id: str,
    content_index: int,
    fact_positions: Mapping[str, int] | None = None,
):
    """Create a processed DTO before its durable chunk ID is known.

    The stable chunk key is used only as a non-empty boundary placeholder.
    ``PersistenceWriter`` replaces it with the exact chunk ID returned by the
    chunk upsert inside the same transaction.
    """

    return to_processed_fact(
        record,
        document_id=document_id,
        chunk_id=record.chunk_key,
        content_index=content_index,
        fact_positions=fact_positions,
    )


def content_positions(items: Sequence[ContentItem]) -> Mapping[int | None, int]:
    """Expose the validated stable-source-to-current-position mapping."""

    return build_content_position_map(items)
