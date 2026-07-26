"""Immutable contracts for read-only entity planning and transactional finalization.

The candidate lookup and scoring phase can be expensive, so Retain performs
it before opening the core write transaction.  These values carry only the
decision produced by that read phase; unresolved canonical rows are created
later, on the Retain unit-of-work connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class EntityOccurrenceBinding:
    """Stable source identity for one entity mention in one projected fact."""

    occurrence_key: str
    unit_key: str
    local_index: int
    event_date: datetime | None

    def __post_init__(self) -> None:
        if not self.occurrence_key:
            raise ValueError("occurrence_key must be non-empty")
        if not self.unit_key:
            raise ValueError("unit_key must be non-empty")
        if isinstance(self.local_index, bool) or not isinstance(self.local_index, int):
            raise TypeError("local_index must be an integer")
        if self.local_index < 0:
            raise ValueError("local_index must be non-negative")


@dataclass(frozen=True, slots=True)
class ExistingEntityBinding:
    """Read-phase decision binding an occurrence to an existing canonical row."""

    occurrence_key: str
    entity_id: Any

    def __post_init__(self) -> None:
        if not self.occurrence_key:
            raise ValueError("occurrence_key must be non-empty")
        if self.entity_id is None or str(self.entity_id) == "":
            raise ValueError("entity_id must be non-empty")


@dataclass(frozen=True, slots=True)
class UnresolvedEntityDescriptor:
    """Canonical entity data that may need to be inserted during finalization."""

    occurrence_key: str
    canonical_name: str
    entity_type: str
    event_date: datetime | None
    nearby_occurrence_keys: tuple[str, ...] = ()
    validated_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.occurrence_key:
            raise ValueError("occurrence_key must be non-empty")
        if not self.canonical_name:
            raise ValueError("canonical_name must be non-empty")
        if not self.entity_type:
            raise ValueError("entity_type must be non-empty")
        if len(self.nearby_occurrence_keys) != len(set(self.nearby_occurrence_keys)):
            raise ValueError("nearby_occurrence_keys must be unique")


@dataclass(frozen=True, slots=True)
class EntityResolutionReadPlan:
    """Complete, provider-neutral result of read-only entity resolution."""

    bank_id: str
    occurrences: tuple[EntityOccurrenceBinding, ...]
    existing_bindings: tuple[ExistingEntityBinding, ...] = ()
    unresolved_descriptors: tuple[UnresolvedEntityDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if not self.bank_id:
            raise ValueError("bank_id must be non-empty")
        occurrence_keys = tuple(item.occurrence_key for item in self.occurrences)
        if len(occurrence_keys) != len(set(occurrence_keys)):
            raise ValueError("occurrences contain duplicate occurrence keys")
        existing_keys = tuple(item.occurrence_key for item in self.existing_bindings)
        unresolved_keys = tuple(item.occurrence_key for item in self.unresolved_descriptors)
        if len(existing_keys) != len(set(existing_keys)):
            raise ValueError("existing_bindings contain duplicate occurrence keys")
        if len(unresolved_keys) != len(set(unresolved_keys)):
            raise ValueError("unresolved_descriptors contain duplicate occurrence keys")
        if set(existing_keys) & set(unresolved_keys):
            raise ValueError("an occurrence cannot be both existing and unresolved")
        if set(occurrence_keys) != set(existing_keys) | set(unresolved_keys):
            raise ValueError("entity read plan must resolve every occurrence exactly once")


@dataclass(frozen=True, slots=True)
class FinalizedEntityResolution:
    """Resolved graph inputs after missing canonical rows are finalized."""

    resolved_entity_ids: tuple[Any, ...]
    entity_to_unit: tuple[tuple[str, int, datetime | None], ...]
    unit_to_entity_ids: tuple[tuple[str, tuple[Any, ...]], ...]
