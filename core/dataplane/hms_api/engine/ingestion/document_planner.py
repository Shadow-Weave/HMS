"""Pure document grouping and append planning for Retain."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TypeGuard

from .domain import ContentItem, ContentOrigin, DocumentIntent, UpdateMode, freeze_json
from .normalization import Clock, parse_event_date

DocumentIdFactory = Callable[[], str]


def _uuid_document_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _is_valid_document_id(document_id: object) -> TypeGuard[str]:
    return isinstance(document_id, str) and bool(document_id.strip())


def _generated_document_id(id_factory: DocumentIdFactory, *, reserved: set[str]) -> str:
    document_id = id_factory()
    if not _is_valid_document_id(document_id):
        raise ValueError("id_factory must return a non-empty document ID")
    if document_id in reserved:
        raise ValueError(f"id_factory returned a duplicate document ID: {document_id!r}")
    return document_id


def _resolve_shared_document_id(
    items: Sequence[ContentItem],
    *,
    batch_document_id: str | None,
    recovered_document_id: str | None,
    id_factory: DocumentIdFactory,
) -> str:
    if _is_valid_document_id(batch_document_id):
        return batch_document_id
    if _is_valid_document_id(recovered_document_id):
        return recovered_document_id
    if any(item.update_mode is UpdateMode.APPEND for item in items):
        raise ValueError("update_mode='append' requires a valid document ID")
    return _generated_document_id(id_factory, reserved=set())


def _validate_update_mode(document_id: str, items: Sequence[ContentItem]) -> UpdateMode:
    if not items:
        raise ValueError("A document intent requires at least one content item")

    update_mode = items[0].update_mode
    if any(item.update_mode is not update_mode for item in items[1:]):
        modes = ", ".join(sorted({item.update_mode.value for item in items}))
        raise ValueError(f"Conflicting update_mode values for document {document_id!r}: {modes}")
    return update_mode


def _build_intent(document_id: str, items: Sequence[ContentItem]) -> DocumentIntent:
    update_mode = _validate_update_mode(document_id, items)
    if update_mode is UpdateMode.APPEND and not _is_valid_document_id(document_id):
        raise ValueError("update_mode='append' requires a valid document ID")

    source_indices = tuple(item.source_index for item in items if item.source_index is not None)
    return DocumentIntent(
        document_id=document_id,
        items=tuple(items),
        source_indices=source_indices,
        expected_input_slots=source_indices,
        update_mode=update_mode,
    )


def plan_documents(
    items: Iterable[ContentItem],
    *,
    batch_document_id: str | None = None,
    recovered_document_id: str | None = None,
    id_factory: DocumentIdFactory = _uuid_document_id,
) -> tuple[DocumentIntent, ...]:
    """Group normalized Retain items into deterministic document intents.

    Explicit IDs determine grouping.  One explicit ID absorbs items without an
    ID, while multiple explicit IDs leave every missing-ID item as its own
    generated document.  If every item is missing an ID, the batch ID wins over
    a recovered operation ID, which in turn wins over the injected factory.
    """

    normalized_items = tuple(items)
    if not normalized_items:
        return ()

    # A non-empty batch-level document ID is authoritative before per-item IDs
    # are inspected. ``MemoryEngine`` only fills missing item IDs, so a caller
    # can still reach this boundary with both values present.
    if _is_valid_document_id(batch_document_id):
        _validate_update_mode(batch_document_id, normalized_items)
        return (_build_intent(batch_document_id, normalized_items),)

    explicit_groups: dict[str, list[ContentItem]] = {}
    for item in normalized_items:
        if _is_valid_document_id(item.document_id):
            explicit_groups.setdefault(item.document_id, []).append(item)

    explicit_ids = tuple(explicit_groups)

    if not explicit_ids:
        effective_id_hint = (
            batch_document_id
            if _is_valid_document_id(batch_document_id)
            else recovered_document_id
            if _is_valid_document_id(recovered_document_id)
            else "<generated>"
        )
        _validate_update_mode(effective_id_hint, normalized_items)
        document_id = _resolve_shared_document_id(
            normalized_items,
            batch_document_id=batch_document_id,
            recovered_document_id=recovered_document_id,
            id_factory=id_factory,
        )
        return (_build_intent(document_id, normalized_items),)

    if len(explicit_ids) == 1:
        return (_build_intent(explicit_ids[0], normalized_items),)

    # Validate every explicit group before invoking the ID factory.  Factory
    # calls are observable injected side effects, so an invalid batch must fail
    # before generating IDs for otherwise unrelated missing-ID items.
    for document_id, group_items in explicit_groups.items():
        _validate_update_mode(document_id, group_items)

    groups: dict[str, list[ContentItem]] = {}
    reserved = set(explicit_ids)
    for item in normalized_items:
        if _is_valid_document_id(item.document_id):
            document_id = item.document_id
        else:
            if item.update_mode is UpdateMode.APPEND:
                raise ValueError("update_mode='append' requires a valid document ID")
            document_id = _generated_document_id(id_factory, reserved=reserved)
            reserved.add(document_id)
        groups.setdefault(document_id, []).append(item)

    return tuple(_build_intent(document_id, group_items) for document_id, group_items in groups.items())


def make_append_synthetic_item(
    existing_content: str,
    *,
    document_id: str,
    template: ContentItem,
    clock: Clock = _utcnow,
) -> ContentItem:
    """Create the non-result-bearing item that represents existing text.

    Append copies only context and tags from the first submitted item.
    Its missing event date is defaulted at append execution time, while
    metadata, declared entities, and observation scopes remain empty.  Keeping
    that behavior here avoids a structural refactor silently changing what is
    projected from unchanged source text.
    """

    if template.update_mode is not UpdateMode.APPEND:
        raise ValueError("An append synthetic item requires an append-mode template")
    if not _is_valid_document_id(document_id):
        raise ValueError("update_mode='append' requires a valid document ID")
    if not isinstance(existing_content, str):
        raise TypeError("existing_content must be a string")

    return replace(
        template,
        content=existing_content,
        event_date=parse_event_date(clock=clock),
        metadata=freeze_json({}),
        entities=(),
        observation_scopes=None,
        document_id=document_id,
        update_mode=UpdateMode.REPLACE,
        source_index=None,
        origin=ContentOrigin.EXISTING_DOCUMENT,
    )


def prepend_existing_document(
    intent: DocumentIntent,
    existing_content: str,
    *,
    clock: Clock = _utcnow,
) -> DocumentIntent:
    """Prepend existing text to an append intent without adding a result slot."""

    if intent.update_mode is not UpdateMode.APPEND:
        raise ValueError("Existing document text can only be prepended to an append intent")
    if not intent.items:
        raise ValueError("An append intent requires at least one content item")

    synthetic = make_append_synthetic_item(
        existing_content,
        document_id=intent.document_id,
        template=intent.items[0],
        clock=clock,
    )
    return replace(intent, items=(synthetic, *intent.items))
