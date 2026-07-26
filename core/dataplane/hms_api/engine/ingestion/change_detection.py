"""Pure document/chunk change classification for Retain."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from .domain import (
    ChunkPlan,
    DocumentChangeKind,
    DocumentChangePlan,
    ExistingChunkFingerprint,
)


def detect_document_change(
    new_chunks: Sequence[ChunkPlan],
    existing_chunks: Sequence[ExistingChunkFingerprint],
    *,
    document_exists: bool,
    existing_document_content_hash: str | None,
    new_document_content_hash: str | None,
    updated_at: datetime | None,
    request_started_at: datetime,
    policy_compatible: bool,
) -> DocumentChangePlan:
    """Classify a document update without performing I/O.

    Chunk identity is the pair ``(global/chunk index, UTF-8 SHA-256)``. Unsafe
    or ambiguous stored state falls back to ``FULL`` with a stable reason. A
    partial delta is used only when at least one existing chunk remains
    unchanged, matching the conservative fallback behavior.
    """

    if not document_exists:
        return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="document_not_found")

    if updated_at is not None and _as_utc(updated_at) > _as_utc(request_started_at):
        return DocumentChangePlan(
            kind=DocumentChangeKind.STALE_SKIP,
            reason="document_updated_after_request_started",
        )

    if not policy_compatible:
        return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="chunk_policy_incompatible")

    if not existing_document_content_hash or not new_document_content_hash:
        return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="missing_document_content_hash")

    if not existing_chunks:
        return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="no_existing_chunks")

    existing_by_index: dict[int, ExistingChunkFingerprint] = {}
    for chunk in existing_chunks:
        if chunk.chunk_index in existing_by_index:
            return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="duplicate_existing_chunk_index")
        if not chunk.content_hash:
            return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="missing_existing_chunk_hash")
        existing_by_index[chunk.chunk_index] = chunk

    new_by_index: dict[int, ChunkPlan] = {}
    for chunk in new_chunks:
        if chunk.global_index in new_by_index:
            return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="duplicate_new_chunk_index")
        if not chunk.content_hash:
            return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="missing_new_chunk_hash")
        new_by_index[chunk.global_index] = chunk

    unchanged: list[int] = []
    changed: list[int] = []
    added: list[int] = []
    removed: list[int] = []

    for index, chunk in sorted(new_by_index.items()):
        existing = existing_by_index.get(index)
        if existing is None:
            added.append(index)
        elif existing.content_hash == chunk.content_hash:
            unchanged.append(index)
        else:
            changed.append(index)

    for index in sorted(existing_by_index):
        if index not in new_by_index:
            removed.append(index)

    if not changed and not added and not removed:
        if existing_document_content_hash != new_document_content_hash:
            return DocumentChangePlan(
                kind=DocumentChangeKind.FULL,
                reason="document_hash_mismatch_without_chunk_changes",
            )
        return DocumentChangePlan(
            kind=DocumentChangeKind.METADATA_ONLY,
            unchanged=tuple(unchanged),
        )

    # Retain deliberately abandons delta when every stored chunk changed.
    # Keeping that rule avoids a delta transaction that is effectively a full
    # replacement but has different observation/outbox/recovery semantics.
    if not unchanged:
        return DocumentChangePlan(kind=DocumentChangeKind.FULL, reason="no_unchanged_chunks")

    return DocumentChangePlan(
        kind=DocumentChangeKind.DELTA,
        unchanged=tuple(unchanged),
        changed=tuple(changed),
        added=tuple(added),
        removed=tuple(removed),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
