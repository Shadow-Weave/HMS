"""Immutable domain values shared by Retain planning stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeAlias

FrozenJsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class FrozenObject:
    items: tuple[tuple[str, "FrozenJson"], ...]


@dataclass(frozen=True, slots=True)
class FrozenArray:
    items: tuple["FrozenJson", ...]


FrozenJson: TypeAlias = FrozenJsonScalar | FrozenObject | FrozenArray


def freeze_json(value: Any) -> FrozenJson:
    """Create a recursively immutable, order-preserving JSON-like value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return FrozenObject(tuple((str(key), freeze_json(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return FrozenArray(tuple(freeze_json(item) for item in value))
    raise TypeError(f"Expected a JSON-compatible value, got {type(value).__name__}")


def thaw_json(value: FrozenJson) -> Any:
    """Convert a value produced by :func:`freeze_json` back to containers."""

    if isinstance(value, FrozenObject):
        return {key: thaw_json(item) for key, item in value.items}
    if isinstance(value, FrozenArray):
        return [thaw_json(item) for item in value.items]
    return value


class EventDateState(StrEnum):
    """Meaning of an item's event-date input after compatibility parsing."""

    DEFAULTED = "defaulted"
    TIMELESS = "timeless"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class EventDateValue:
    state: EventDateState
    value: datetime | None

    def __post_init__(self) -> None:
        if self.state is EventDateState.TIMELESS and self.value is not None:
            raise ValueError("A timeless event date cannot carry a datetime")
        if self.state is not EventDateState.TIMELESS and self.value is None:
            raise ValueError(f"{self.state.value} event date requires a datetime")


class ContentOrigin(StrEnum):
    SUBMITTED = "submitted"
    EXISTING_DOCUMENT = "existing_document"


class UpdateMode(StrEnum):
    REPLACE = "replace"
    APPEND = "append"


ObservationScopes: TypeAlias = str | tuple[tuple[str, ...], ...] | None


@dataclass(frozen=True, slots=True)
class ContentItem:
    """Normalized, immutable Retain input item."""

    content: str
    context: str
    event_date: EventDateValue
    metadata: FrozenJson
    entities: tuple[FrozenJson, ...]
    tags: tuple[str, ...]
    observation_scopes: ObservationScopes
    document_id: str | None
    update_mode: UpdateMode
    source_index: int | None
    origin: ContentOrigin = ContentOrigin.SUBMITTED


@dataclass(frozen=True, slots=True)
class DocumentIntent:
    """All normalized content that will form one tracked document."""

    document_id: str
    items: tuple[ContentItem, ...]
    source_indices: tuple[int, ...]
    expected_input_slots: tuple[int, ...]
    update_mode: UpdateMode


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    version: str
    max_chars: int
    conversation_mode: bool = True
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if self.overlap < 0 or self.overlap >= self.max_chars:
            raise ValueError("overlap must be non-negative and smaller than max_chars")


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunk_key: str
    source_index: int | None
    global_index: int
    local_index: int
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class ExistingChunkFingerprint:
    chunk_id: str
    chunk_index: int
    content_hash: str | None


class DocumentChangeKind(StrEnum):
    FULL = "full"
    DELTA = "delta"
    METADATA_ONLY = "metadata_only"
    STALE_SKIP = "stale_skip"


@dataclass(frozen=True, slots=True)
class DocumentChangePlan:
    kind: DocumentChangeKind
    unchanged: tuple[int, ...] = ()
    changed: tuple[int, ...] = ()
    added: tuple[int, ...] = ()
    removed: tuple[int, ...] = ()
    reason: str | None = None

    @property
    def chunks_to_process(self) -> tuple[int, ...]:
        return tuple(sorted((*self.changed, *self.added)))
