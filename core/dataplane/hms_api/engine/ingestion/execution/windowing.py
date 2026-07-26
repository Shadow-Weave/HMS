"""Pure planning and result-mapping helpers for FULL Retain write windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ..domain import ChunkPlan


class FactRecordIdentity(Protocol):
    """Minimum stable identity required to map one committed fact result."""

    fact_key: str
    source_index: int | None


WindowUnitResult: TypeAlias = tuple[
    Sequence[FactRecordIdentity],
    Sequence[tuple[str, str]],
]


@dataclass(frozen=True, slots=True)
class FullWriteWindowPlan:
    """One immutable, globally ordered FULL-document write window."""

    window_index: int
    chunks: tuple[ChunkPlan, ...]
    global_indices: tuple[int, ...]
    is_first: bool
    is_last: bool

    def __post_init__(self) -> None:
        if isinstance(self.window_index, bool) or not isinstance(self.window_index, int):
            raise TypeError("window_index must be an integer")
        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")
        if not isinstance(self.chunks, tuple) or any(not isinstance(chunk, ChunkPlan) for chunk in self.chunks):
            raise TypeError("chunks must be a tuple of ChunkPlan values")
        if not isinstance(self.global_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in self.global_indices
        ):
            raise TypeError("global_indices must be a tuple of integers")
        expected_indices = tuple(chunk.global_index for chunk in self.chunks)
        if self.global_indices != expected_indices:
            raise ValueError("global_indices must exactly match chunks in their stored order")
        if any(left >= right for left, right in zip(self.global_indices, self.global_indices[1:])):
            raise ValueError("global_indices must be strictly increasing")
        if not isinstance(self.is_first, bool) or not isinstance(self.is_last, bool):
            raise TypeError("is_first and is_last must be booleans")
        if self.is_first != (self.window_index == 0):
            raise ValueError("is_first must be true exactly for window_index=0")
        if not self.chunks and not (self.window_index == 0 and self.is_first and self.is_last):
            raise ValueError("an empty FULL window must be the sole first/final window")


def plan_full_write_windows(
    chunks: Sequence[ChunkPlan],
    batch_size: int,
) -> tuple[FullWriteWindowPlan, ...]:
    """Partition a complete, ordered chunk plan into deterministic windows.

    The configured ``retain_chunk_batch_size=0`` convention disables batching,
    so all chunks are placed in one window. Empty documents still receive one
    first/final zero-fact window so document tracking and a final transactional
    callback have a durable execution point.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative; 0 disables batching")

    chunk_batch = tuple(chunks)
    if any(not isinstance(chunk, ChunkPlan) for chunk in chunk_batch):
        raise TypeError("chunks must contain only ChunkPlan values")
    observed_indices = tuple(chunk.global_index for chunk in chunk_batch)
    expected_indices = tuple(range(len(chunk_batch)))
    if observed_indices != expected_indices:
        raise ValueError("chunks must be ordered by unique, continuous global_index values 0..N-1")

    if not chunk_batch:
        return (
            FullWriteWindowPlan(
                window_index=0,
                chunks=(),
                global_indices=(),
                is_first=True,
                is_last=True,
            ),
        )

    effective_batch_size = len(chunk_batch) if batch_size == 0 else batch_size
    windows: list[FullWriteWindowPlan] = []
    for window_index, start in enumerate(range(0, len(chunk_batch), effective_batch_size)):
        window_chunks = chunk_batch[start : start + effective_batch_size]
        windows.append(
            FullWriteWindowPlan(
                window_index=window_index,
                chunks=window_chunks,
                global_indices=tuple(chunk.global_index for chunk in window_chunks),
                is_first=window_index == 0,
                is_last=start + effective_batch_size >= len(chunk_batch),
            )
        )
    return tuple(windows)


def merge_window_unit_ids(
    document_source_indices: Sequence[int | None],
    window_results: Sequence[WindowUnitResult],
) -> tuple[tuple[str, ...], ...]:
    """Map committed window results back to ordered submitted document items.

    ``document_source_indices`` is the document-item order, not a dense numeric
    range. A single ``None`` denotes append's synthetic existing-content item;
    its committed unit IDs are validated but deliberately omitted from the
    public buckets. Mapping order never determines output order: records do.
    """

    sources = tuple(document_source_indices)
    seen_sources: set[int | None] = set()
    for source in sources:
        _validate_source_index(source, field_name="document_source_indices")
        if source in seen_sources:
            raise ValueError(f"document_source_indices contains duplicate source_index={source!r}")
        seen_sources.add(source)

    public_sources = tuple(source for source in sources if source is not None)
    buckets: dict[int, list[str]] = {source: [] for source in public_sources}
    seen_record_keys: set[str] = set()
    seen_mapping_keys: set[str] = set()
    seen_unit_ids: set[str] = set()

    for window_index, window_result in enumerate(window_results):
        if not isinstance(window_result, tuple) or len(window_result) != 2:
            raise TypeError(f"window_results[{window_index}] must be a (records, unit_ids_by_fact_key) tuple")
        records = tuple(window_result[0])
        bindings = tuple(window_result[1])

        local_record_keys: list[str] = []
        for record_index, record in enumerate(records):
            fact_key = getattr(record, "fact_key", None)
            source_index = getattr(record, "source_index", object())
            if not isinstance(fact_key, str) or not fact_key:
                raise ValueError(f"window_results[{window_index}].records[{record_index}] has an invalid fact_key")
            _validate_source_index(
                source_index,
                field_name=f"window_results[{window_index}].records[{record_index}].source_index",
            )
            if source_index not in seen_sources:
                raise ValueError(f"fact_key={fact_key!r} references unknown source_index={source_index!r}")
            if fact_key in seen_record_keys:
                raise ValueError(f"duplicate record fact_key across windows: {fact_key!r}")
            seen_record_keys.add(fact_key)
            local_record_keys.append(fact_key)

        units_by_key: dict[str, str] = {}
        for binding_index, binding in enumerate(bindings):
            if not isinstance(binding, tuple) or len(binding) != 2:
                raise TypeError(
                    f"window_results[{window_index}].unit_ids_by_fact_key[{binding_index}] "
                    "must be a (fact_key, unit_id) tuple"
                )
            fact_key, unit_id = binding
            if not isinstance(fact_key, str) or not fact_key:
                raise ValueError("unit_ids_by_fact_key contains an invalid fact_key")
            if not isinstance(unit_id, str) or not unit_id:
                raise ValueError(f"unit_ids_by_fact_key[{fact_key!r}] has an invalid unit_id")
            if fact_key in seen_mapping_keys:
                raise ValueError(f"duplicate mapped fact_key across windows: {fact_key!r}")
            if unit_id in seen_unit_ids:
                raise ValueError(f"duplicate unit_id across windows: {unit_id!r}")
            seen_mapping_keys.add(fact_key)
            seen_unit_ids.add(unit_id)
            units_by_key[fact_key] = unit_id

        record_key_set = set(local_record_keys)
        mapping_key_set = set(units_by_key)
        if record_key_set != mapping_key_set:
            missing = sorted(record_key_set - mapping_key_set)
            unexpected = sorted(mapping_key_set - record_key_set)
            raise ValueError(f"window fact-key mapping is incomplete (missing={missing}, unexpected={unexpected})")

        for record in records:
            if record.source_index is not None:
                buckets[record.source_index].append(units_by_key[record.fact_key])

    return tuple(tuple(buckets[source]) for source in public_sources)


def _validate_source_index(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must contain only integers or None")
    if value < 0:
        raise ValueError(f"{field_name} must contain only non-negative integers or None")
