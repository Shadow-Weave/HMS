"""Pure normalization of caller-owned Retain inputs.

This module is deliberately independent from the database, provider clients,
and process configuration.  It converts the raw compatibility envelope into
immutable domain values without retaining references to caller-owned
containers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Final

from .domain import (
    ContentItem,
    EventDateState,
    EventDateValue,
    FrozenJson,
    ObservationScopes,
    UpdateMode,
    freeze_json,
)

Clock = Callable[[], datetime]

_MISSING: Final = object()
_OBSERVATION_SCOPE_NAMES: Final = frozenset({"per_tag", "combined", "all_combinations"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_aware(value: datetime, *, field_name: str) -> datetime:
    """Return an aware datetime while preserving an explicit timezone.

    Retain interprets naive values as UTC but passes aware values through
    unchanged. Keeping that rule matters before extraction because
    converting to UTC can change the calendar date and weekday near a timezone
    boundary.
    """

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must produce a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value


def parse_event_date(
    value: Any = _MISSING,
    *,
    clock: Clock = _utcnow,
) -> EventDateValue:
    """Normalize Retain event-date semantics into an explicit state.

    A missing value, or any falsey value other than ``None``, retains the
    established behavior of defaulting to the injected clock. An explicit ``None``
    is timeless.  Truthy values must be a :class:`datetime` or an ISO-8601
    string. Naive values are interpreted as UTC; explicit timezones are kept.
    """

    if value is None:
        return EventDateValue(EventDateState.TIMELESS, None)

    if value is _MISSING or not value:
        return EventDateValue(EventDateState.DEFAULTED, _as_aware(clock(), field_name="clock"))

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"event_date must be a valid ISO-8601 datetime, got {value!r}") from exc
    else:
        raise TypeError(f"event_date must be a datetime or ISO-8601 string, got {type(value).__name__}")

    return EventDateValue(EventDateState.EXPLICIT, _as_aware(parsed, field_name="event_date"))


def _normalize_tags(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings or None")

    normalized: list[str] = []
    for index, tag in enumerate(value):
        if not isinstance(tag, str):
            raise TypeError(f"{field_name}[{index}] must be a string, got {type(tag).__name__}")
        normalized.append(tag)
    return tuple(normalized)


def merge_tags(item_tags: Any = None, document_tags: Any = None) -> tuple[str, ...]:
    """Merge item and batch tags with stable, first-occurrence deduplication."""

    candidates = (
        *_normalize_tags(item_tags, field_name="tags"),
        *_normalize_tags(document_tags, field_name="document_tags"),
    )
    seen: set[str] = set()
    merged: list[str] = []
    for tag in candidates:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)
    return tuple(merged)


def _freeze_mapping(value: Any, *, field_name: str) -> FrozenJson:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None, got {type(value).__name__}")

    # ``dict`` snapshots arbitrary Mapping implementations; ``deepcopy``
    # severs nested references before the JSON-like tree is frozen.
    snapshot = deepcopy(dict(value))
    try:
        return freeze_json(snapshot)
    except TypeError as exc:
        raise TypeError(f"{field_name} must contain only JSON-compatible values: {exc}") from exc


def _normalize_entities(value: Any) -> tuple[FrozenJson, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("entities must be a sequence of mappings or None")

    normalized: list[FrozenJson] = []
    for index, entity in enumerate(value):
        if not isinstance(entity, Mapping):
            raise TypeError(f"entities[{index}] must be a mapping, got {type(entity).__name__}")

        snapshot = deepcopy(dict(entity))
        text = snapshot.get("text", _MISSING)
        if text is _MISSING:
            raise ValueError(f"entities[{index}] must contain a 'text' field")
        if not isinstance(text, str):
            raise TypeError(f"entities[{index}].text must be a string, got {type(text).__name__}")
        entity_type = snapshot.get("type")
        if entity_type is not None and not isinstance(entity_type, str):
            raise TypeError(f"entities[{index}].type must be a string or None, got {type(entity_type).__name__}")

        try:
            normalized.append(freeze_json(snapshot))
        except TypeError as exc:
            raise TypeError(f"entities[{index}] must contain only JSON-compatible values: {exc}") from exc
    return tuple(normalized)


def _normalize_observation_scopes(value: Any) -> ObservationScopes:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in _OBSERVATION_SCOPE_NAMES:
            choices = ", ".join(sorted(_OBSERVATION_SCOPE_NAMES))
            raise ValueError(f"observation_scopes must be one of {choices}, or a sequence of tag sequences")
        return value
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise TypeError("observation_scopes must be a supported string, a sequence of tag sequences, or None")

    scopes: list[tuple[str, ...]] = []
    for scope_index, scope in enumerate(value):
        if isinstance(scope, (str, bytes)) or not isinstance(scope, Sequence):
            raise TypeError(f"observation_scopes[{scope_index}] must be a sequence of strings")
        tags: list[str] = []
        for tag_index, tag in enumerate(scope):
            if not isinstance(tag, str):
                raise TypeError(
                    f"observation_scopes[{scope_index}][{tag_index}] must be a string, got {type(tag).__name__}"
                )
            tags.append(tag)
        scopes.append(tuple(tags))
    return tuple(scopes)


def _normalize_document_id(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"document_id must be a string or None, got {type(value).__name__}")
    return value


def _normalize_update_mode(value: Any) -> UpdateMode:
    if value is None:
        return UpdateMode.REPLACE
    if not isinstance(value, str):
        raise TypeError(f"update_mode must be a string or None, got {type(value).__name__}")
    try:
        return UpdateMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in UpdateMode)
        raise ValueError(f"update_mode must be one of: {choices}; got {value!r}") from exc


def _validate_source_index(source_index: Any) -> int:
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise TypeError(f"source_index must be an integer, got {type(source_index).__name__}")
    if source_index < 0:
        raise ValueError("source_index must be non-negative for submitted content")
    return source_index


def normalize_content_item(
    raw_item: Mapping[str, Any],
    *,
    source_index: int = 0,
    document_tags: Sequence[str] | None = None,
    clock: Clock = _utcnow,
) -> ContentItem:
    """Return one immutable submitted item without retaining caller state."""

    if not isinstance(raw_item, Mapping):
        raise TypeError(f"content item must be a mapping, got {type(raw_item).__name__}")
    normalized_source_index = _validate_source_index(source_index)
    item = deepcopy(dict(raw_item))

    if "content" not in item:
        raise ValueError("content is required")
    content = item["content"]
    if not isinstance(content, str):
        raise TypeError(f"content must be a string, got {type(content).__name__}")

    context = item.get("context", "")
    if context is None:
        context = ""
    if not isinstance(context, str):
        raise TypeError(f"context must be a string or None, got {type(context).__name__}")

    document_id = _normalize_document_id(item.get("document_id"))
    update_mode = _normalize_update_mode(item.get("update_mode"))
    if update_mode is UpdateMode.APPEND and document_id is None:
        raise ValueError("update_mode='append' requires a document_id")

    raw_event_date = item["event_date"] if "event_date" in item else _MISSING
    return ContentItem(
        content=content,
        context=context,
        event_date=parse_event_date(raw_event_date, clock=clock),
        metadata=_freeze_mapping(item.get("metadata"), field_name="metadata"),
        entities=_normalize_entities(item.get("entities")),
        tags=merge_tags(item.get("tags"), document_tags),
        observation_scopes=_normalize_observation_scopes(item.get("observation_scopes")),
        document_id=document_id,
        update_mode=update_mode,
        source_index=normalized_source_index,
    )


def normalize_contents(
    raw_contents: Iterable[Mapping[str, Any]],
    *,
    document_tags: Sequence[str] | None = None,
    clock: Clock = _utcnow,
) -> tuple[ContentItem, ...]:
    """Normalize a Retain batch and assign stable zero-based source indices."""

    # Validate and snapshot batch tags once; the tuple is already immutable and
    # can safely be shared while each item applies stable deduplication.
    normalized_document_tags = _normalize_tags(document_tags, field_name="document_tags")
    return tuple(
        normalize_content_item(
            raw_item,
            source_index=source_index,
            document_tags=normalized_document_tags,
            clock=clock,
        )
        for source_index, raw_item in enumerate(raw_contents)
    )
