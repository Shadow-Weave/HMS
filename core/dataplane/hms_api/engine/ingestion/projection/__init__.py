"""Projection of extracted facts into persistence-ready memory records."""

from .embeddings import (
    AsyncEmbeddingPort,
    EmbeddingCardinalityError,
    EmbeddingFailurePolicy,
    build_embedding_text,
    project_embeddings,
)
from .records import (
    MemoryRecord,
    build_projection_manifest,
    thaw_declared_entities,
    to_processed_fact,
)

__all__ = [
    "AsyncEmbeddingPort",
    "EmbeddingCardinalityError",
    "EmbeddingFailurePolicy",
    "MemoryRecord",
    "build_embedding_text",
    "build_projection_manifest",
    "project_embeddings",
    "thaw_declared_entities",
    "to_processed_fact",
]
