"""Provider-neutral contracts for Retain fact extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ...response_models import TokenUsage
from ..domain import ChunkPlan, ContentItem
from .models import CausalFactRelation, FactCandidate


class ExtractionMode(StrEnum):
    CONCISE = "concise"
    VERBOSE = "verbose"
    CUSTOM = "custom"
    VERBATIM = "verbatim"
    CHUNKS = "chunks"


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    mode: ExtractionMode
    fact_type_override: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExtractionMode):
            raise TypeError("mode must be an ExtractionMode")
        if self.fact_type_override is not None and not isinstance(self.fact_type_override, str):
            raise TypeError("fact_type_override must be a string or None")


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    items: tuple[ContentItem, ...]
    chunks: tuple[ChunkPlan, ...]
    policy: ExtractionPolicy
    preserve_chunk_boundaries: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(not isinstance(item, ContentItem) for item in self.items):
            raise TypeError("items must be a tuple of ContentItem values")
        if not isinstance(self.chunks, tuple) or any(not isinstance(chunk, ChunkPlan) for chunk in self.chunks):
            raise TypeError("chunks must be a tuple of ChunkPlan values")
        if not isinstance(self.policy, ExtractionPolicy):
            raise TypeError("policy must be an ExtractionPolicy")
        if not isinstance(self.preserve_chunk_boundaries, bool):
            raise TypeError("preserve_chunk_boundaries must be a bool")


@dataclass(frozen=True, slots=True)
class ChunkFactCount:
    chunk_key: str
    fact_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_key, str) or not self.chunk_key:
            raise ValueError("chunk_key must be a non-empty string")
        if isinstance(self.fact_count, bool) or not isinstance(self.fact_count, int) or self.fact_count < 0:
            raise ValueError("fact_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    candidates: tuple[FactCandidate, ...]
    chunk_fact_counts: tuple[ChunkFactCount, ...]
    usage: TokenUsage
    causal_relations: tuple[CausalFactRelation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, FactCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of FactCandidate values")
        if not isinstance(self.chunk_fact_counts, tuple) or any(
            not isinstance(count, ChunkFactCount) for count in self.chunk_fact_counts
        ):
            raise TypeError("chunk_fact_counts must be a tuple of ChunkFactCount values")
        if sum(count.fact_count for count in self.chunk_fact_counts) != len(self.candidates):
            raise ValueError("chunk fact counts must equal the number of candidates")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage")
        if not isinstance(self.causal_relations, tuple) or any(
            not isinstance(relation, CausalFactRelation) for relation in self.causal_relations
        ):
            raise TypeError("causal_relations must be a tuple of CausalFactRelation values")
        candidate_keys = {candidate.fact_key for candidate in self.candidates}
        if any(
            relation.source_fact_key not in candidate_keys or relation.target_fact_key not in candidate_keys
            for relation in self.causal_relations
        ):
            raise ValueError("causal relation endpoints must reference candidates in this extraction result")
        candidate_relations = tuple(
            relation for candidate in self.candidates for relation in candidate.causal_relations
        )
        if candidate_relations != self.causal_relations:
            raise ValueError("result causal relations must exactly match the relations attached to candidates")


@runtime_checkable
class FactExtractor(Protocol):
    async def extract(self, request: ExtractionRequest) -> ExtractionResult: ...


class ExtractionAdapterError(RuntimeError):
    """Base error for an extractor boundary failure."""


class ExtractionContractError(ExtractionAdapterError):
    """The extraction backend returned an internally inconsistent result."""


class ExtractionModeMismatchError(ExtractionAdapterError):
    """The provider-neutral policy and resolved configuration disagree."""


class BatchExtractionUnsupportedError(ExtractionAdapterError):
    """Batch extraction was requested but cannot be safely adapted."""
