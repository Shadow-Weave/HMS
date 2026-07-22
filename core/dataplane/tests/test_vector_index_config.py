"""Configuration and optional-dependency tests for semantic vector indexes."""

from __future__ import annotations

import pytest

from hms_api.config import HMSConfig
from hms_api.engine.vector_index import DatabaseVectorIndex, create_vector_index
from hms_api.engine.vector_index.milvus import MilvusVectorIndex, _has_lite_3_cosine_distance_bug


def test_database_vector_index_is_the_default(monkeypatch):
    monkeypatch.delenv("HMS_API_VECTOR_INDEX_PROVIDER", raising=False)

    config = HMSConfig.from_env()
    index = create_vector_index(config)

    assert config.vector_index_provider == "database"
    assert isinstance(index, DatabaseVectorIndex)
    assert index.is_external is False


def test_milvus_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("HMS_API_VECTOR_INDEX_PROVIDER", "milvus")
    monkeypatch.setenv("HMS_API_MILVUS_URI", "https://example.invalid")
    monkeypatch.setenv("HMS_API_MILVUS_TOKEN", "secret")
    monkeypatch.setenv("HMS_API_MILVUS_DB_NAME", "tenant_db")
    monkeypatch.setenv("HMS_API_MILVUS_COLLECTION", "custom_collection")
    monkeypatch.setenv("HMS_API_MILVUS_CONSISTENCY_LEVEL", "eventually")

    config = HMSConfig.from_env()

    assert config.vector_index_provider == "milvus"
    assert config.milvus_uri == "https://example.invalid"
    assert config.milvus_token == "secret"
    assert config.milvus_db_name == "tenant_db"
    assert config.milvus_collection == "custom_collection"
    assert config.milvus_consistency_level == "Eventually"
    assert "milvus_token" in config.get_credential_fields()
    assert "milvus_uri" in config.get_credential_fields()


def test_milvus_configuration_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("HMS_API_VECTOR_INDEX_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="Invalid vector_index_provider"):
        HMSConfig.from_env()

    monkeypatch.setenv("HMS_API_VECTOR_INDEX_PROVIDER", "milvus")
    monkeypatch.setenv("HMS_API_MILVUS_CONSISTENCY_LEVEL", "sometimes")
    with pytest.raises(ValueError, match="Invalid milvus_consistency_level"):
        HMSConfig.from_env()


@pytest.mark.asyncio
async def test_milvus_missing_optional_dependency_has_actionable_error(monkeypatch, tmp_path):
    index = MilvusVectorIndex(uri=str(tmp_path / "missing.db"), collection_name="missing_dependency")

    def missing_import():
        raise ImportError("Milvus vector indexing requires the optional dependency")

    monkeypatch.setattr(index, "_import_pymilvus", missing_import)
    with pytest.raises(ImportError, match="optional dependency"):
        await index.initialize(3)


def test_milvus_lite_3_cosine_workaround_is_narrowly_version_gated(monkeypatch):
    monkeypatch.setattr("hms_api.engine.vector_index.milvus.importlib.metadata.version", lambda name: "3.0.0")

    assert _has_lite_3_cosine_distance_bug("./local.db") is True
    assert _has_lite_3_cosine_distance_bug("http://localhost:19530") is False

    monkeypatch.setattr("hms_api.engine.vector_index.milvus.importlib.metadata.version", lambda name: "3.1.0")
    assert _has_lite_3_cosine_distance_bug("./local.db") is False
