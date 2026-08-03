"""Provider-neutral fact candidates produced by Retain extraction strategies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from ..domain import FrozenJson, ObservationScopes

FACT_KEY_VERSION = "retain-fact-v1"


def compute_fact_key(
    *,
    chunk_key: str,
    source_index: int | None,
    global_index: int,
    extractor_local_index: int,
    text: str,
    fact_type: str,
) -> str:
    """Build an unambiguous deterministic SHA-256 fact identity.

    Canonical JSON avoids delimiter collisions, while the explicit version
    leaves room for a future identity policy without silently changing keys.
    """

    if isinstance(extractor_local_index, bool) or not isinstance(extractor_local_index, int):
        raise TypeError("extractor_local_index must be an integer")
    if extractor_local_index < 0:
        raise ValueError("extractor_local_index must be non-negative")

    payload = json.dumps(
        [FACT_KEY_VERSION, chunk_key, source_index, global_index, extractor_local_index, text, fact_type],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_datetime(value: datetime | None, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime or None")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class CausalFactRelation:
    """Stable provider-neutral causal edge between two extracted facts."""

    source_fact_key: str
    target_fact_key: str
    relation_type: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("source_fact_key", self.source_fact_key),
            ("target_fact_key", self.target_fact_key),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if self.source_fact_key == self.target_fact_key:
            raise ValueError("causal relations cannot be self-referential")
        if self.relation_type != "caused_by":
            raise ValueError("relation_type must be 'caused_by'")


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """Immutable extraction result identified independently of database IDs."""

    fact_key: str
    chunk_key: str
    source_index: int | None
    global_index: int
    extractor_local_index: int
    text: str
    fact_type: str
    context: str
    where: str | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    metadata: FrozenJson
    declared_entities: tuple[FrozenJson, ...]
    tags: tuple[str, ...]
    observation_scopes: ObservationScopes
    # Provider-extracted names are distinct from caller-declared entity
    # objects.  Passthrough extraction deliberately leaves these empty, which
    # preserves the existing projection manifest semantics.
    entity_mentions: tuple[str, ...]
    causal_relations: tuple[CausalFactRelation, ...]

    def __post_init__(self) -> None:
        if len(self.fact_key) != 64 or any(character not in "0123456789abcdef" for character in self.fact_key):
            raise ValueError("fact_key must be a lowercase SHA-256 hex digest")
        if not isinstance(self.chunk_key, str) or not self.chunk_key:
            raise ValueError("chunk_key must be a non-empty string")
        if self.source_index is not None:
            if isinstance(self.source_index, bool) or not isinstance(self.source_index, int):
                raise TypeError("source_index must be an integer or None")
            if self.source_index < 0:
                raise ValueError("source_index must be non-negative")
        if isinstance(self.global_index, bool) or not isinstance(self.global_index, int):
            raise TypeError("global_index must be an integer")
        if self.global_index < 0:
            raise ValueError("global_index must be non-negative")
        if isinstance(self.extractor_local_index, bool) or not isinstance(self.extractor_local_index, int):
            raise TypeError("extractor_local_index must be an integer")
        if self.extractor_local_index < 0:
            raise ValueError("extractor_local_index must be non-negative")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.fact_type, str) or not self.fact_type:
            raise ValueError("fact_type must be a non-empty string")
        if not isinstance(self.context, str):
            raise TypeError("context must be a string")
        if self.where is not None and not isinstance(self.where, str):
            raise TypeError("where must be a string or None")
        _validate_datetime(self.occurred_start, field_name="occurred_start")
        _validate_datetime(self.occurred_end, field_name="occurred_end")
        _validate_datetime(self.mentioned_at, field_name="mentioned_at")
        if not isinstance(self.declared_entities, tuple):
            raise TypeError("declared_entities must be a tuple of frozen JSON values")
        if not isinstance(self.tags, tuple) or any(not isinstance(tag, str) for tag in self.tags):
            raise TypeError("tags must be a tuple of strings")
        if not isinstance(self.entity_mentions, tuple) or any(
            not isinstance(entity, str) for entity in self.entity_mentions
        ):
            raise TypeError("entity_mentions must be a tuple of strings")
        if not isinstance(self.causal_relations, tuple) or any(
            not isinstance(relation, CausalFactRelation) for relation in self.causal_relations
        ):
            raise TypeError("causal_relations must be a tuple of CausalFactRelation values")
        if any(relation.source_fact_key != self.fact_key for relation in self.causal_relations):
            raise ValueError("every causal relation on a candidate must use that candidate as its source")
