"""LLM-free extraction in which every planned chunk is one fact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta

from ..domain import ChunkPlan, ContentItem, ContentOrigin
from .models import FactCandidate, compute_fact_key

# Chunk passthrough offsets facts by absolute extraction order so equal
# timestamps still have a deterministic temporal ordering.
SECONDS_PER_FACT = 0.01


def build_content_position_map(items: Sequence[ContentItem]) -> dict[int | None, int]:
    """Map stable source indices to positional content indices.

    At most one ``None`` key is permitted, and it must represent the synthetic
    existing-document item used by append planning.
    """

    positions: dict[int | None, int] = {}
    for position, item in enumerate(items):
        source_index = item.source_index
        if source_index is None and item.origin is not ContentOrigin.EXISTING_DOCUMENT:
            raise ValueError("source_index=None is reserved for one synthetic existing-document item")
        if source_index in positions:
            label = "synthetic source_index=None" if source_index is None else f"source_index={source_index}"
            raise ValueError(f"Duplicate {label} in content items")
        positions[source_index] = position
    return positions


def _fact_type(override: str | None) -> str:
    if override is None or override == "":
        return "world"
    if not isinstance(override, str):
        raise TypeError("fact_type_override must be a string or None")
    return override


def extract_passthrough(
    chunks: Sequence[ChunkPlan],
    items: Sequence[ContentItem],
    *,
    fact_type_override: str | None = None,
    fact_position_offset: int = 0,
) -> tuple[FactCandidate, ...]:
    """Produce exactly one immutable candidate for every chunk plan."""

    if isinstance(fact_position_offset, bool) or not isinstance(fact_position_offset, int):
        raise TypeError("fact_position_offset must be an integer")
    if fact_position_offset < 0:
        raise ValueError("fact_position_offset must be non-negative")

    content_positions = build_content_position_map(items)
    fact_type = _fact_type(fact_type_override)
    seen_chunk_keys: set[str] = set()
    seen_global_indices: set[int] = set()
    candidates: list[FactCandidate] = []

    for fact_position, chunk in enumerate(chunks):
        if chunk.chunk_key in seen_chunk_keys:
            raise ValueError(f"Duplicate chunk_key: {chunk.chunk_key!r}")
        if chunk.global_index in seen_global_indices:
            raise ValueError(f"Duplicate chunk global_index: {chunk.global_index}")
        seen_chunk_keys.add(chunk.chunk_key)
        seen_global_indices.add(chunk.global_index)

        try:
            item = items[content_positions[chunk.source_index]]
        except KeyError as exc:
            raise ValueError(
                f"Chunk {chunk.chunk_key!r} references unknown source_index={chunk.source_index!r}"
            ) from exc

        mentioned_at = item.event_date.value
        absolute_fact_position = fact_position_offset + fact_position
        if mentioned_at is not None and absolute_fact_position:
            mentioned_at += timedelta(seconds=absolute_fact_position * SECONDS_PER_FACT)

        candidates.append(
            FactCandidate(
                fact_key=compute_fact_key(
                    chunk_key=chunk.chunk_key,
                    source_index=chunk.source_index,
                    global_index=chunk.global_index,
                    extractor_local_index=0,
                    text=chunk.text,
                    fact_type=fact_type,
                ),
                chunk_key=chunk.chunk_key,
                source_index=chunk.source_index,
                global_index=chunk.global_index,
                extractor_local_index=0,
                text=chunk.text,
                fact_type=fact_type,
                context=item.context,
                where=None,
                occurred_start=None,
                occurred_end=None,
                mentioned_at=mentioned_at,
                metadata=item.metadata,
                declared_entities=item.entities,
                tags=item.tags,
                observation_scopes=item.observation_scopes,
                entity_mentions=(),
                causal_relations=(),
            )
        )

    return tuple(candidates)


def to_chunk_metadata(chunk: ChunkPlan, *, content_index: int):
    """Adapt a planned one-fact chunk to persistence metadata.

    ``content_index`` is explicit because source indices are stable request
    identities, whereas the storage field is a position in the current
    extraction window.
    """

    if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
        raise ValueError("content_index must be a non-negative integer")

    from ...retain.types import ChunkMetadata

    return ChunkMetadata(
        chunk_text=chunk.text,
        fact_count=1,
        content_index=content_index,
        chunk_index=chunk.global_index,
    )


def to_chunk_metadata_batch(
    chunks: Sequence[ChunkPlan],
    *,
    content_positions: Mapping[int | None, int],
):
    """Adapt chunks using an explicit stable-source-to-position mapping."""

    metadata = []
    for chunk in chunks:
        try:
            content_index = content_positions[chunk.source_index]
        except KeyError as exc:
            raise ValueError(f"Missing content position for source_index={chunk.source_index!r}") from exc
        metadata.append(to_chunk_metadata(chunk, content_index=content_index))
    return metadata
