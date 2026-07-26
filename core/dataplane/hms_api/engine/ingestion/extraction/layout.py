"""Lossless pre-chunked layout for windowed Retain extraction.

The extraction primitive identifies input content by its zero-based position,
while a document can have several independently active chunks that all belong
to the same stable ``ContentItem.source_index``. This module gives every active
chunk a temporary one-to-one content position and then restores document-level
identities on the extraction result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from ..domain import ChunkPlan, ContentItem
from .models import CausalFactRelation, FactCandidate, compute_fact_key
from .passthrough import build_content_position_map
from .ports import (
    ChunkFactCount,
    ExtractionContractError,
    ExtractionPolicy,
    ExtractionRequest,
    ExtractionResult,
)


class PrechunkedLayoutError(ExtractionContractError):
    """A pre-chunked request or its extraction result is inconsistent."""


@dataclass(frozen=True, slots=True)
class PrechunkedExtractionLayout:
    """Temporary one-content-per-chunk extraction layout.

    ``temporary_items`` and ``temporary_chunks`` are suitable for an
    :class:`ExtractionRequest`.  Temporary source indices are exactly their
    local content positions.  ``temporary_to_original_source`` is an immutable
    map back to the stable source indices in ``document_items``.
    """

    document_items: tuple[ContentItem, ...]
    active_chunks: tuple[ChunkPlan, ...]
    temporary_items: tuple[ContentItem, ...]
    temporary_chunks: tuple[ChunkPlan, ...]
    temporary_to_original_source: Mapping[int, int | None]

    def __post_init__(self) -> None:
        for field_name, values, value_type in (
            ("document_items", self.document_items, ContentItem),
            ("active_chunks", self.active_chunks, ChunkPlan),
            ("temporary_items", self.temporary_items, ContentItem),
            ("temporary_chunks", self.temporary_chunks, ChunkPlan),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, value_type) for value in values):
                raise TypeError(f"{field_name} must be a tuple of {value_type.__name__} values")

        try:
            original_positions = build_content_position_map(self.document_items)
        except (TypeError, ValueError) as exc:
            raise PrechunkedLayoutError(str(exc)) from exc

        cardinality = len(self.active_chunks)
        if len(self.temporary_items) != cardinality or len(self.temporary_chunks) != cardinality:
            raise PrechunkedLayoutError(
                "active chunks, temporary items, and temporary chunks must have equal cardinality"
            )

        if not isinstance(self.temporary_to_original_source, Mapping):
            raise TypeError("temporary_to_original_source must be a mapping")
        source_map = dict(self.temporary_to_original_source)
        expected_temporary_sources = set(range(cardinality))
        if set(source_map) != expected_temporary_sources:
            raise PrechunkedLayoutError("temporary source mapping keys must exactly equal local content positions")
        for temporary_source, original_source in source_map.items():
            if isinstance(temporary_source, bool) or not isinstance(temporary_source, int):
                raise PrechunkedLayoutError("temporary source mapping keys must be integers")
            if original_source is not None and (
                isinstance(original_source, bool) or not isinstance(original_source, int) or original_source < 0
            ):
                raise PrechunkedLayoutError("original source mapping values must be non-negative integers or None")
        object.__setattr__(self, "temporary_to_original_source", MappingProxyType(source_map))

        seen_chunk_keys: set[str] = set()
        seen_global_indices: set[int] = set()
        for position, (original_chunk, temporary_item, temporary_chunk) in enumerate(
            zip(self.active_chunks, self.temporary_items, self.temporary_chunks, strict=True)
        ):
            if original_chunk.chunk_key in seen_chunk_keys:
                raise PrechunkedLayoutError(f"duplicate active chunk_key={original_chunk.chunk_key!r}")
            if original_chunk.global_index in seen_global_indices:
                raise PrechunkedLayoutError(f"duplicate active chunk global_index={original_chunk.global_index}")
            seen_chunk_keys.add(original_chunk.chunk_key)
            seen_global_indices.add(original_chunk.global_index)

            if original_chunk.source_index not in original_positions:
                raise PrechunkedLayoutError(
                    f"active chunk[{position}] references unknown source_index={original_chunk.source_index!r}"
                )
            if source_map[position] != original_chunk.source_index:
                raise PrechunkedLayoutError(f"temporary source mapping disagrees at content position {position}")

            original_item = self.document_items[original_positions[original_chunk.source_index]]
            expected_item = replace(original_item, content=original_chunk.text, source_index=position)
            expected_chunk = replace(original_chunk, source_index=position)
            if temporary_item != expected_item:
                raise PrechunkedLayoutError(
                    f"temporary item[{position}] does not losslessly represent its active chunk"
                )
            if temporary_chunk != expected_chunk:
                raise PrechunkedLayoutError(f"temporary chunk[{position}] changed fields other than source_index")

    def extraction_request(self, policy: ExtractionPolicy) -> ExtractionRequest:
        """Build the extractor request for this temporary layout."""

        return ExtractionRequest(items=self.temporary_items, chunks=self.temporary_chunks, policy=policy)

    def remap_result(self, result: ExtractionResult) -> ExtractionResult:
        """Restore original source indices and every derived fact identity."""

        if not isinstance(result, ExtractionResult):
            raise TypeError("result must be an ExtractionResult")

        self._validate_chunk_counts(result.chunk_fact_counts, result.candidates)

        old_to_new_fact_key: dict[str, str] = {}
        remapped_without_relations: list[FactCandidate] = []
        seen_old_keys: set[str] = set()
        seen_new_keys: set[str] = set()
        seen_local_indices: set[tuple[int, int]] = set()

        for candidate_position, candidate in enumerate(result.candidates):
            temporary_source = candidate.source_index
            if isinstance(temporary_source, bool) or not isinstance(temporary_source, int):
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] has invalid temporary source_index={temporary_source!r}"
                )
            if temporary_source not in self.temporary_to_original_source:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] references unknown temporary source_index={temporary_source}"
                )

            expected_chunk = self.temporary_chunks[temporary_source]
            if candidate.chunk_key != expected_chunk.chunk_key:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] chunk_key does not match its temporary source"
                )
            if candidate.global_index != expected_chunk.global_index:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] global_index does not match its temporary source"
                )

            local_identity = (temporary_source, candidate.extractor_local_index)
            if local_identity in seen_local_indices:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] duplicates extractor-local index "
                    f"{candidate.extractor_local_index} for temporary source {temporary_source}"
                )
            seen_local_indices.add(local_identity)

            expected_old_key = compute_fact_key(
                chunk_key=candidate.chunk_key,
                source_index=temporary_source,
                global_index=candidate.global_index,
                extractor_local_index=candidate.extractor_local_index,
                text=candidate.text,
                fact_type=candidate.fact_type,
            )
            if candidate.fact_key != expected_old_key:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] fact_key is inconsistent with its temporary identity"
                )
            if candidate.fact_key in seen_old_keys:
                raise PrechunkedLayoutError(f"candidate[{candidate_position}] duplicates fact_key")
            seen_old_keys.add(candidate.fact_key)

            original_source = self.temporary_to_original_source[temporary_source]
            new_fact_key = compute_fact_key(
                chunk_key=candidate.chunk_key,
                source_index=original_source,
                global_index=candidate.global_index,
                extractor_local_index=candidate.extractor_local_index,
                text=candidate.text,
                fact_type=candidate.fact_type,
            )
            if new_fact_key in seen_new_keys:
                raise PrechunkedLayoutError(
                    f"candidate[{candidate_position}] collides after restoring its original source"
                )
            seen_new_keys.add(new_fact_key)
            old_to_new_fact_key[candidate.fact_key] = new_fact_key
            remapped_without_relations.append(
                replace(
                    candidate,
                    fact_key=new_fact_key,
                    source_index=original_source,
                    causal_relations=(),
                )
            )

        remapped_relations = tuple(
            self._remap_relation(relation, old_to_new_fact_key, relation_position)
            for relation_position, relation in enumerate(result.causal_relations)
        )
        relations_by_source: dict[str, list[CausalFactRelation]] = {}
        for relation in remapped_relations:
            relations_by_source.setdefault(relation.source_fact_key, []).append(relation)

        candidates = tuple(
            replace(
                candidate,
                causal_relations=tuple(relations_by_source.get(candidate.fact_key, ())),
            )
            for candidate in remapped_without_relations
        )
        if tuple(relation for candidate in candidates for relation in candidate.causal_relations) != remapped_relations:
            raise PrechunkedLayoutError("causal relation ordering cannot be represented by the remapped candidates")

        # Counts and usage contain no source-derived identity, so retain their
        # exact values (and object identity) across the layout boundary.
        return ExtractionResult(
            candidates=candidates,
            chunk_fact_counts=result.chunk_fact_counts,
            usage=result.usage,
            causal_relations=remapped_relations,
        )

    def _validate_chunk_counts(
        self,
        counts: tuple[ChunkFactCount, ...],
        candidates: tuple[FactCandidate, ...],
    ) -> None:
        if len(counts) != len(self.temporary_chunks):
            raise PrechunkedLayoutError("extraction result must contain exactly one fact count for every active chunk")

        actual_counts = Counter(candidate.chunk_key for candidate in candidates)
        for position, (count, chunk) in enumerate(zip(counts, self.temporary_chunks, strict=True)):
            if count.chunk_key != chunk.chunk_key:
                raise PrechunkedLayoutError(f"chunk_fact_counts[{position}] does not match the active chunk order")
            if actual_counts[count.chunk_key] != count.fact_count:
                raise PrechunkedLayoutError(
                    f"chunk_fact_counts[{position}]={count.fact_count} does not match "
                    f"{actual_counts[count.chunk_key]} candidates"
                )

        known_chunk_keys = {chunk.chunk_key for chunk in self.temporary_chunks}
        unknown_chunk_keys = set(actual_counts) - known_chunk_keys
        if unknown_chunk_keys:
            raise PrechunkedLayoutError(f"candidates reference unknown chunk keys: {sorted(unknown_chunk_keys)!r}")

    @staticmethod
    def _remap_relation(
        relation: CausalFactRelation,
        fact_key_map: Mapping[str, str],
        relation_position: int,
    ) -> CausalFactRelation:
        try:
            source_fact_key = fact_key_map[relation.source_fact_key]
        except KeyError as exc:
            raise PrechunkedLayoutError(
                f"causal relation[{relation_position}] source has no candidate mapping"
            ) from exc
        try:
            target_fact_key = fact_key_map[relation.target_fact_key]
        except KeyError as exc:
            raise PrechunkedLayoutError(
                f"causal relation[{relation_position}] target has no candidate mapping"
            ) from exc
        try:
            return CausalFactRelation(
                source_fact_key=source_fact_key,
                target_fact_key=target_fact_key,
                relation_type=relation.relation_type,
            )
        except (TypeError, ValueError) as exc:
            raise PrechunkedLayoutError(
                f"causal relation[{relation_position}] violates remapped relation invariants: {exc}"
            ) from exc


def build_prechunked_extraction_layout(
    document_items: Sequence[ContentItem],
    active_chunks: Sequence[ChunkPlan],
) -> PrechunkedExtractionLayout:
    """Build an immutable one-to-one extraction layout for active chunks."""

    if isinstance(document_items, (str, bytes)) or not isinstance(document_items, Sequence):
        raise TypeError("document_items must be a sequence of ContentItem values")
    if isinstance(active_chunks, (str, bytes)) or not isinstance(active_chunks, Sequence):
        raise TypeError("active_chunks must be a sequence of ChunkPlan values")
    items = tuple(document_items)
    chunks = tuple(active_chunks)
    if any(not isinstance(item, ContentItem) for item in items):
        raise TypeError("document_items must contain only ContentItem values")
    if any(not isinstance(chunk, ChunkPlan) for chunk in chunks):
        raise TypeError("active_chunks must contain only ChunkPlan values")

    try:
        original_positions = build_content_position_map(items)
    except (TypeError, ValueError) as exc:
        raise PrechunkedLayoutError(str(exc)) from exc

    temporary_items: list[ContentItem] = []
    temporary_chunks: list[ChunkPlan] = []
    source_map: dict[int, int | None] = {}
    for temporary_source, chunk in enumerate(chunks):
        try:
            original_item = items[original_positions[chunk.source_index]]
        except KeyError as exc:
            raise PrechunkedLayoutError(
                f"active chunk[{temporary_source}] references unknown source_index={chunk.source_index!r}"
            ) from exc
        temporary_items.append(replace(original_item, content=chunk.text, source_index=temporary_source))
        temporary_chunks.append(replace(chunk, source_index=temporary_source))
        source_map[temporary_source] = chunk.source_index

    return PrechunkedExtractionLayout(
        document_items=items,
        active_chunks=chunks,
        temporary_items=tuple(temporary_items),
        temporary_chunks=tuple(temporary_chunks),
        temporary_to_original_source=source_map,
    )
