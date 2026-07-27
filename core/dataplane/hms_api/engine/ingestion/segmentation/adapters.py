"""Adapters from per-item segmentation results to Retain chunk plans."""

from __future__ import annotations

from collections.abc import Sequence

from ..chunking import compute_content_hash
from ..domain import ChunkPlan, ContentItem
from .models import SegmentationResult


def build_chunk_plans_from_segmentation(
    document_id: str,
    items: Sequence[ContentItem],
    results: Sequence[SegmentationResult],
) -> tuple[ChunkPlan, ...]:
    """Build stable document-wide plans in original item order.

    Callers may plan items concurrently, but ``results`` must be restored to
    the same order as ``items`` before calling this function. ``local_index``
    restarts for each item; ``global_index`` and ``chunk_key`` are assigned
    synchronously across the complete document, matching the existing Retain
    chunk identity contract.
    """

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("items must be a sequence of ContentItem values")
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a sequence of SegmentationResult values")
    if len(items) != len(results):
        raise ValueError("items and results must have the same length")

    plans: list[ChunkPlan] = []
    global_index = 0
    for item, result in zip(items, results, strict=True):
        if not isinstance(item, ContentItem):
            raise TypeError("items must contain only ContentItem values")
        if not isinstance(result, SegmentationResult):
            raise TypeError("results must contain only SegmentationResult values")
        for local_index, segment in enumerate(result.segments):
            content_hash = compute_content_hash(segment.text)
            if content_hash != segment.content_hash:
                raise ValueError("segment content_hash does not match its text")
            chunk_key = f"chunk:{len(document_id)}:{document_id}:{global_index}:{content_hash}"
            plans.append(
                ChunkPlan(
                    chunk_key=chunk_key,
                    source_index=item.source_index,
                    global_index=global_index,
                    local_index=local_index,
                    text=segment.text,
                    content_hash=content_hash,
                )
            )
            global_index += 1
    return tuple(plans)


__all__ = ["build_chunk_plans_from_segmentation"]
