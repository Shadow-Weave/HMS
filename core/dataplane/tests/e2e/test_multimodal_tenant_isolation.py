"""Two-schema multimodal upload, ledger, tag, and recall isolation E2E.

This test never calls OpenAI.  It creates two disposable PostgreSQL schemas in
the explicitly supplied test database, then removes them in ``finally``.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import replace

import asyncpg
import httpx
import pytest

from hms_api.api.http import create_app
from hms_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hms_api.engine.memory_engine import MemoryEngine
from hms_api.engine.multimodal import (
    FakeMultimodalProvider,
    GroundedStatement,
    ImageNormalizationConfig,
    ModelMultimodalDescription,
    normalize_image,
)
from hms_api.engine.parsers import MultimodalParserConfig, OpenAIMultimodalParser
from hms_api.engine.query_analyzer import DateparserQueryAnalyzer
from hms_api.engine.task_backend import SyncTaskBackend
from hms_api.extensions.builtin.tenant import DefaultTenantExtension
from hms_api.migrations import run_migrations
from hms_api.models import RequestContext
from tests.e2e.test_multimodal_offline_e2e import _DeterministicEmbeddings, _synthetic_png


async def _schema_ddl(database_url: str, schema: str, *, create: bool) -> None:
    # ``schema`` is generated locally from a UUID and never contains user data.
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        if create:
            await connection.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await connection.close()


async def _create_schema_memory(monkeypatch, database_url: str, schema: str) -> MemoryEngine:
    import hms_api.config

    raw_config = hms_api.config._get_raw_config()
    test_config = replace(
        raw_config,
        enable_file_upload_api=True,
        file_parser_allowlist=None,
        file_delete_after_retain=False,
        multimodal_enabled=False,
    )
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: test_config)
    run_migrations(database_url, schema=schema)

    memory = MemoryEngine(
        db_url=database_url,
        memory_llm_provider="none",
        memory_llm_model="none",
        embeddings=_DeterministicEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        tenant_extension=DefaultTenantExtension({"schema": schema}),
        skip_llm_verification=True,
    )
    await memory.initialize()
    return memory


def _description(text: str, evidence_id: str) -> ModelMultimodalDescription:
    return ModelMultimodalDescription(
        summary=[
            GroundedStatement(
                text=text,
                evidence_ids=[evidence_id],
                uncertainty="low",
            )
        ],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[],
        limitations=[],
    )


@pytest.mark.asyncio
async def test_same_asset_and_ids_are_isolated_across_tenant_schemas(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    suffix = uuid.uuid4().hex
    schema_a = f"hms_mm_tenant_a_{suffix}"
    schema_b = f"hms_mm_tenant_b_{suffix}"
    memory_a: MemoryEngine | None = None
    memory_b: MemoryEngine | None = None
    await _schema_ddl(database_url, schema_a, create=True)
    await _schema_ddl(database_url, schema_b, create=True)
    try:
        memory_a = await _create_schema_memory(monkeypatch, database_url, schema_a)
        memory_b = await _create_schema_memory(monkeypatch, database_url, schema_b)

        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-tenant-isolation.png",
            declared_mime="image/png",
            config=image_config,
        )
        evidence_id = normalized.evidence.evidence_id
        sentinel_a = f"TENANT_A_ONLY_{uuid.uuid4().hex}"
        sentinel_b = f"TENANT_B_ONLY_{uuid.uuid4().hex}"
        provider_a = FakeMultimodalProvider(_description(sentinel_a, evidence_id))
        provider_b = FakeMultimodalProvider(_description(sentinel_b, evidence_id))
        memory_a._parser_registry.register(
            OpenAIMultimodalParser(provider_a, MultimodalParserConfig(image=image_config))
        )
        memory_b._parser_registry.register(
            OpenAIMultimodalParser(provider_b, MultimodalParserConfig(image=image_config))
        )

        bank_id = f"shared-bank-{uuid.uuid4().hex}"
        document_id = f"shared-document-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id, "tags": ["shared-tag"]}],
        }

        async def upload(memory: MemoryEngine) -> None:
            app = create_app(memory, initialize_memory=False)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    f"/v1/default/banks/{bank_id}/files/retain",
                    files={
                        "files": (
                            "synthetic-tenant-isolation.png",
                            image_bytes,
                            "image/png",
                        )
                    },
                    data={"request": json.dumps(request_body)},
                )
                assert response.status_code == 200, response.text
                [operation_id] = response.json()["operation_ids"]
                status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
                assert status.json()["result_metadata"]["multimodal"]["recall_ready"] is True

        await upload(memory_a)
        await upload(memory_b)
        assert provider_a.calls == 1
        assert provider_b.calls == 1

        async def recall(memory: MemoryEngine, query: str) -> str:
            result = await memory.recall_async(
                bank_id,
                query,
                max_tokens=1024,
                include_chunks=True,
                max_chunk_tokens=4096,
                tags=["shared-tag"],
                tags_match="all_strict",
                request_context=RequestContext(),
            )
            return json.dumps(result.model_dump(mode="json"), sort_keys=True)

        tenant_a_result = await recall(memory_a, sentinel_a)
        tenant_b_result = await recall(memory_b, sentinel_b)
        assert sentinel_a in tenant_a_result
        assert sentinel_b not in tenant_a_result
        assert sentinel_b in tenant_b_result
        assert sentinel_a not in tenant_b_result

        connection = await asyncpg.connect(database_url)
        try:
            descriptor_a = await connection.fetchval(
                f'SELECT descriptor_key FROM "{schema_a}".multimodal_descriptor_cache WHERE bank_id = $1',
                bank_id,
            )
            descriptor_b = await connection.fetchval(
                f'SELECT descriptor_key FROM "{schema_b}".multimodal_descriptor_cache WHERE bank_id = $1',
                bank_id,
            )
            source_key_a = await connection.fetchval(
                f'SELECT source_storage_key FROM "{schema_a}".multimodal_document_commands '
                "WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            source_key_b = await connection.fetchval(
                f'SELECT source_storage_key FROM "{schema_b}".multimodal_document_commands '
                "WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        finally:
            await connection.close()
        assert descriptor_a and descriptor_b and descriptor_a != descriptor_b
        expected_a_scope = hashlib.sha256(f"{schema_a}\0{bank_id}".encode()).hexdigest()[:24]
        expected_b_scope = hashlib.sha256(f"{schema_b}\0{bank_id}".encode()).hexdigest()[:24]
        assert source_key_a.startswith(f"media/{expected_a_scope}/")
        assert source_key_b.startswith(f"media/{expected_b_scope}/")
        assert source_key_a != source_key_b
    finally:
        if memory_a is not None:
            await memory_a.close()
        if memory_b is not None:
            await memory_b.close()
        await _schema_ddl(database_url, schema_a, create=False)
        await _schema_ddl(database_url, schema_b, create=False)
