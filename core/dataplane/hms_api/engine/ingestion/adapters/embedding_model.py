"""Adapter from the async embedding port to the configured model API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...retain import embedding_processing
from ..projection.embeddings import EmbeddingVector


class EmbeddingModelAdapter:
    """Expose the existing synchronous ``encode`` model as an async port.

    The delegated helper already moves CPU-bound model work off the event loop
    and validates one-output-per-input cardinality. The ingestion pipeline
    performs its own
    cardinality check as a second boundary guard before positional projection.
    """

    def __init__(self, embeddings_model: Any) -> None:
        if embeddings_model is None:
            raise ValueError("Retain requires an initialized embeddings model")
        self._embeddings_model = embeddings_model

    async def embed_batch(self, texts: tuple[str, ...]) -> Sequence[EmbeddingVector]:
        return await embedding_processing.generate_embeddings_batch(
            self._embeddings_model,
            list(texts),
        )
