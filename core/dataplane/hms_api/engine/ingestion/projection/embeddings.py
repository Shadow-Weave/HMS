"""Async embedding projection with explicit whole-batch failure policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..extraction.models import FactCandidate
from .records import MemoryRecord, build_projection_manifest

EmbeddingVector = Sequence[float] | None
FormatDate = Callable[[datetime], str]


class AsyncEmbeddingPort(Protocol):
    """Provider-independent asynchronous batch embedding boundary."""

    async def embed_batch(self, texts: tuple[str, ...]) -> Sequence[EmbeddingVector]: ...


class EmbeddingFailurePolicy(StrEnum):
    STORE_WITHOUT_EMBEDDING = "store_without_embedding"
    RAISE = "raise"


class EmbeddingCardinalityError(RuntimeError):
    """Raised when a backend violates one-output-per-fact alignment."""


def build_embedding_text(candidate: FactCandidate, format_date: FormatDate) -> str:
    """Build temporal and entity context for embedding."""

    fact_date = candidate.occurred_start or candidate.mentioned_at
    if fact_date is not None:
        readable_date = format_date(fact_date)
        if not isinstance(readable_date, str):
            raise TypeError("format_date must return a string")
        if candidate.occurred_end is not None and candidate.occurred_end != candidate.occurred_start:
            readable_end = format_date(candidate.occurred_end)
            if not isinstance(readable_end, str):
                raise TypeError("format_date must return a string")
            text = f"{candidate.text} (happened from {readable_date} to {readable_end})"
        else:
            text = f"{candidate.text} (happened in {readable_date})"
    else:
        text = candidate.text

    # Declared entities stay on their dedicated resolution path.  Only names
    # produced by an extraction strategy augment embeddings, matching the
    # ProcessedFact manifest/entity semantics.
    if candidate.entity_mentions:
        text = f"{text} [{', '.join(candidate.entity_mentions)}]"
    return text


def _freeze_vector(vector: EmbeddingVector, *, index: int) -> tuple[float, ...] | None:
    if vector is None:
        return None
    if isinstance(vector, (str, bytes)):
        raise TypeError(f"embedding[{index}] must be a numeric sequence or None")
    try:
        return tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"embedding[{index}] must be a numeric sequence or None") from exc


async def project_embeddings(
    candidates: Sequence[FactCandidate],
    *,
    embedder: AsyncEmbeddingPort,
    format_date: FormatDate,
    embedding_model_version: str = "unknown",
    extraction_version: str = "5w-v1",
    failure_policy: EmbeddingFailurePolicy | str = EmbeddingFailurePolicy.STORE_WITHOUT_EMBEDDING,
) -> tuple[MemoryRecord, ...]:
    """Project candidates 1:1, degrading only genuine backend exceptions.

    A backend exception defaults to the existing Retain behavior: every fact
    remains persistable with a NULL embedding and ``embedding.ok=false``.
    Returning the wrong number of vectors is instead a port contract violation
    and always raises; no policy may silently truncate or positionally shift
    facts.
    """

    candidate_batch = tuple(candidates)
    if not candidate_batch:
        return ()
    try:
        policy = EmbeddingFailurePolicy(failure_policy)
    except ValueError as exc:
        choices = ", ".join(value.value for value in EmbeddingFailurePolicy)
        raise ValueError(f"failure_policy must be one of: {choices}") from exc

    embedding_texts = tuple(build_embedding_text(candidate, format_date) for candidate in candidate_batch)
    try:
        raw_embeddings = await embedder.embed_batch(embedding_texts)
    except Exception:
        if policy is EmbeddingFailurePolicy.RAISE:
            raise
        embeddings: tuple[tuple[float, ...] | None, ...] = (None,) * len(candidate_batch)
    else:
        if raw_embeddings is None:
            raise TypeError("embedding backend must return a sequence, got None")
        raw_batch = tuple(raw_embeddings)
        if len(raw_batch) != len(candidate_batch):
            raise EmbeddingCardinalityError(
                f"Embedding backend returned {len(raw_batch)} vectors for {len(candidate_batch)} facts; "
                "expected exact 1:1 alignment"
            )
        embeddings = tuple(_freeze_vector(vector, index=index) for index, vector in enumerate(raw_batch))

    return tuple(
        MemoryRecord.from_candidate(
            candidate,
            embedding=embedding,
            projection=build_projection_manifest(
                candidate,
                embedding=embedding,
                embedding_model_version=embedding_model_version,
                extraction_version=extraction_version,
            ),
        )
        for candidate, embedding in zip(candidate_batch, embeddings, strict=True)
    )
