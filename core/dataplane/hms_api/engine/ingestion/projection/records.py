"""Immutable projected records and their persistence conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain import FrozenJson, FrozenObject, freeze_json, thaw_json
from ..extraction.models import FactCandidate


@dataclass(frozen=True, slots=True)
class MemoryRecord(FactCandidate):
    """One fact candidate enriched for persistence and current Recall."""

    embedding: tuple[float, ...] | None
    projection: FrozenJson

    def __post_init__(self) -> None:
        FactCandidate.__post_init__(self)
        if self.embedding is not None:
            if not isinstance(self.embedding, tuple) or any(not isinstance(value, float) for value in self.embedding):
                raise TypeError("embedding must be a tuple of floats or None")
        if not isinstance(self.projection, FrozenObject):
            raise TypeError("projection must be a frozen JSON object")

    @classmethod
    def from_candidate(
        cls,
        candidate: FactCandidate,
        *,
        embedding: tuple[float, ...] | None,
        projection: FrozenJson,
    ) -> "MemoryRecord":
        return cls(
            fact_key=candidate.fact_key,
            chunk_key=candidate.chunk_key,
            source_index=candidate.source_index,
            global_index=candidate.global_index,
            extractor_local_index=candidate.extractor_local_index,
            text=candidate.text,
            fact_type=candidate.fact_type,
            context=candidate.context,
            where=candidate.where,
            occurred_start=candidate.occurred_start,
            occurred_end=candidate.occurred_end,
            mentioned_at=candidate.mentioned_at,
            metadata=candidate.metadata,
            declared_entities=candidate.declared_entities,
            tags=candidate.tags,
            observation_scopes=candidate.observation_scopes,
            entity_mentions=candidate.entity_mentions,
            causal_relations=candidate.causal_relations,
            embedding=embedding,
            projection=projection,
        )


def build_projection_manifest(
    candidate: FactCandidate,
    *,
    embedding: tuple[float, ...] | None,
    embedding_model_version: str,
    extraction_version: str,
) -> FrozenObject:
    """Build the read-side manifest currently understood by Recall."""

    if not isinstance(embedding_model_version, str) or not embedding_model_version:
        raise ValueError("embedding_model_version must be a non-empty string")
    if not isinstance(extraction_version, str) or not extraction_version:
        raise ValueError("extraction_version must be a non-empty string")

    temporal_grade = (
        "resolved" if candidate.occurred_start is not None or candidate.mentioned_at is not None else "unresolved"
    )
    manifest = freeze_json(
        {
            "embedding": {"v": embedding_model_version, "ok": embedding is not None},
            "tsvector": {"v": 1, "ok": True},
            "temporal": {"v": 1, "grade": temporal_grade},
            "entities": {"v": 1, "ok": bool(candidate.entity_mentions)},
            "extraction": {"v": extraction_version},
        }
    )
    if not isinstance(manifest, FrozenObject):  # pragma: no cover - guaranteed by the literal above
        raise AssertionError("projection manifest must be an object")
    return manifest


def thaw_declared_entities(record: FactCandidate) -> list[dict[str, Any]]:
    """Return caller-declared entities for entity resolution."""

    entities: list[dict[str, Any]] = []
    for index, frozen_entity in enumerate(record.declared_entities):
        entity = thaw_json(frozen_entity)
        if not isinstance(entity, dict):
            raise TypeError(f"declared_entities[{index}] must thaw to an object")
        entities.append(entity)
    return entities


def to_processed_fact(
    record: MemoryRecord,
    *,
    document_id: str,
    chunk_id: str,
    content_index: int,
    fact_positions: Mapping[str, int] | None = None,
):
    """Adapt a record to ``ProcessedFact`` with explicit identity bindings.

    Stable ``fact_key``/``chunk_key`` remain on the projected record. The adapter is
    intentionally passed the effective document ID, persisted chunk ID, and
    current content position so none of those mappings are inferred from
    mutable list order.
    """

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")
    if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
        raise ValueError("content_index must be a non-negative integer")

    metadata = thaw_json(record.metadata)
    projection = thaw_json(record.projection)
    if not isinstance(metadata, dict):
        raise TypeError("record metadata must thaw to an object")
    if not isinstance(projection, dict):  # pragma: no cover - MemoryRecord enforces this
        raise TypeError("record projection must thaw to an object")

    from ...retain.types import CausalRelation, EntityRef, ProcessedFact

    if isinstance(record.observation_scopes, tuple):
        observation_scopes = [list(scope) for scope in record.observation_scopes]
    else:
        observation_scopes = record.observation_scopes

    causal_relations = []
    if record.causal_relations:
        if fact_positions is None:
            raise ValueError("fact_positions is required when a record has causal relations")
        source_position = fact_positions.get(record.fact_key)
        if isinstance(source_position, bool) or not isinstance(source_position, int) or source_position < 0:
            raise ValueError("fact_positions must contain a non-negative position for the source fact")
        for relation in record.causal_relations:
            target_position = fact_positions.get(relation.target_fact_key)
            if isinstance(target_position, bool) or not isinstance(target_position, int) or target_position < 0:
                raise ValueError("fact_positions must contain a non-negative position for every causal target")
            if target_position >= source_position:
                raise ValueError("causal targets must precede their source fact")
            causal_relations.append(
                CausalRelation(
                    relation_type=relation.relation_type,
                    target_fact_index=target_position,
                )
            )

    return ProcessedFact(
        fact_text=record.text,
        fact_type=record.fact_type,
        embedding=list(record.embedding) if record.embedding is not None else None,
        occurred_start=record.occurred_start,
        occurred_end=record.occurred_end,
        mentioned_at=record.mentioned_at,
        context=record.context,
        metadata=metadata,
        where=record.where,
        entities=[EntityRef(name=name) for name in record.entity_mentions],
        causal_relations=causal_relations,
        chunk_id=chunk_id,
        document_id=document_id,
        content_index=content_index,
        tags=list(record.tags),
        observation_scopes=observation_scopes,
        projection=projection,
    )
