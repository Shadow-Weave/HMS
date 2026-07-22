"""Semantic vector-index provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import VectorIndex, VectorIndexRecord, VectorSearchHit, parse_embedding, record_from_row
from .database import DatabaseVectorIndex

if TYPE_CHECKING:
    from ...config import HMSConfig, StaticConfigProxy


def create_vector_index(config: "HMSConfig | StaticConfigProxy") -> VectorIndex:
    """Create the configured semantic vector-index provider."""

    provider = config.vector_index_provider.lower()
    if provider == "database":
        return DatabaseVectorIndex()
    if provider == "milvus":
        from .milvus import MilvusVectorIndex

        return MilvusVectorIndex(
            uri=config.milvus_uri,
            token=config.milvus_token,
            db_name=config.milvus_db_name,
            collection_name=config.milvus_collection,
            consistency_level=config.milvus_consistency_level,
        )
    raise ValueError(f"Unsupported vector index provider: {provider}")


__all__ = [
    "DatabaseVectorIndex",
    "VectorIndex",
    "VectorIndexRecord",
    "VectorSearchHit",
    "create_vector_index",
    "parse_embedding",
    "record_from_row",
]
