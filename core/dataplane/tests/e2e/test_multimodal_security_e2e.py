"""Opt-in PostgreSQL scan for multimodal transport-payload leakage.

This test uses a local MockTransport for the OpenAI boundary.  It never sends a
network request or incurs provider cost.  A disposable PostgreSQL URL is
required because the test inspects the actual operation/audit/document/chunk
rows produced by the full upload -> retain pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

import asyncpg
import httpx
import pytest

from hms_api.api.http import create_app
from hms_api.engine.multimodal import (
    GroundedStatement,
    ImageNormalizationConfig,
    ModelMultimodalDescription,
    OpenAIProviderConfig,
    OpenAIResponsesMultimodalProvider,
    normalize_image,
)
from hms_api.engine.parsers import OpenAIMultimodalParser
from tests.e2e.test_multimodal_offline_e2e import _create_memory, _synthetic_png


@pytest.mark.asyncio
async def test_postgresql_operation_audit_document_chunk_surfaces_exclude_transport_payload(
    monkeypatch, caplog
) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    provider_client: httpx.AsyncClient | None = None
    try:
        image_bytes = _synthetic_png()
        normalized = normalize_image(
            file_data=image_bytes,
            filename="security-surface.png",
            declared_mime="image/png",
            config=ImageNormalizationConfig(),
        )
        evidence_id = normalized.evidence.evidence_id
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A synthetic editor layout is visible.",
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
        expected_base64 = base64.b64encode(normalized.evidence.encoded_bytes).decode("ascii")
        expected_data_url = f"data:{normalized.evidence.mime_type};base64,{expected_base64}"
        api_key_sentinel = "HMS_POSTGRES_SECURITY_API_KEY_SENTINEL"
        transport_seen = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal transport_seen
            payload = json.loads(request.content)
            assert payload["input"][0]["content"][1]["image_url"] == expected_data_url
            transport_seen = True
            return httpx.Response(
                200,
                json={
                    "id": "resp-postgres-security",
                    "model": "gpt-5-mini-test-revision",
                    "status": "completed",
                    "output_text": description.model_dump_json(),
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )

        provider_config = OpenAIProviderConfig(
            api_key=api_key_sentinel,
            base_url="https://api.openai.test/v1",
            max_retries=0,
            max_schema_repairs=0,
        )
        provider_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=provider_config.base_url + "/",
        )
        memory._parser_registry.register(
            OpenAIMultimodalParser(OpenAIResponsesMultimodalProvider(provider_config, client=provider_client))
        )
        memory._audit_logger._enabled = True
        memory._audit_logger._allowed_actions = None

        bank_id = f"multimodal-security-{uuid.uuid4().hex}"
        document_id = f"security-{uuid.uuid4().hex}"
        app = create_app(memory, initialize_memory=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("security-surface.png", image_bytes, "image/png")},
                data={
                    "request": json.dumps(
                        {
                            "parser": "openai_multimodal",
                            "files_metadata": [{"document_id": document_id, "tags": ["security-scan"]}],
                        }
                    )
                },
            )
            assert upload.status_code == 200, upload.text
            [operation_id] = upload.json()["operation_ids"]
            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert status.status_code == 200, status.text
            assert status.json()["result_metadata"]["multimodal"]["recall_ready"] is True

        assert transport_seen is True

        connection = await asyncpg.connect(database_url)
        try:
            # Audit writes are deliberately fire-and-forget.  Poll only for
            # their committed evidence, with a short deterministic upper bound.
            audit_rows = []
            for _ in range(50):
                audit_rows = await connection.fetch(
                    "SELECT request::text, response::text, metadata::text FROM audit_log WHERE bank_id = $1",
                    bank_id,
                )
                if audit_rows:
                    break
                await asyncio.sleep(0.02)

            operation_rows = await connection.fetch(
                """
                SELECT task_payload::text, result_metadata::text, COALESCE(error_message, '') AS error_message
                FROM async_operations WHERE bank_id = $1
                """,
                bank_id,
            )
            document_rows = await connection.fetch(
                """
                SELECT original_text, content_hash, retain_params::text, tags::text,
                       file_storage_key, file_original_name, file_content_type
                FROM documents WHERE bank_id = $1
                """,
                bank_id,
            )
            chunk_rows = await connection.fetch(
                "SELECT chunk_text FROM chunks WHERE bank_id = $1",
                bank_id,
            )
            memory_rows = await connection.fetch(
                "SELECT text, context, metadata::text FROM memory_units WHERE bank_id = $1",
                bank_id,
            )
            descriptor_rows = await connection.fetch(
                """
                SELECT canonical_markdown, provenance_metadata::text, entities::text
                FROM multimodal_descriptor_cache WHERE bank_id = $1
                """,
                bank_id,
            )
        finally:
            await connection.close()

        assert operation_rows
        assert document_rows
        assert chunk_rows
        assert memory_rows
        assert audit_rows
        storage_keys = [row["file_storage_key"] for row in document_rows if row["file_storage_key"]]
        assert storage_keys
        for storage_key in storage_keys:
            assert storage_key.startswith("media/")
            assert normalized.asset.sha256 not in storage_key
            assert "security-surface.png" not in storage_key

        multimodal_worker_logs = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.name == "hms_api.engine.memory_engine"
            and ("[FILE_CONVERT_RETAIN]" in record.getMessage() or "[BATCH_RETAIN_TASK]" in record.getMessage())
            and "multimodal" in record.getMessage().lower()
        )
        assert multimodal_worker_logs
        orchestrator_logs = "\n".join(
            record.getMessage() for record in caplog.records if record.name == "hms_api.engine.retain.orchestrator"
        )
        assert orchestrator_logs
        assert "<redacted>" in orchestrator_logs
        multimodal_pipeline_logs = f"{multimodal_worker_logs}\n{orchestrator_logs}"
        for forbidden in (bank_id, document_id, normalized.asset.sha256, "security-surface.png"):
            assert forbidden not in multimodal_pipeline_logs

        persisted = json.dumps(
            {
                "operations": [dict(row) for row in operation_rows],
                "audit": [dict(row) for row in audit_rows],
                "documents": [dict(row) for row in document_rows],
                "chunks": [dict(row) for row in chunk_rows],
                "memory_units": [dict(row) for row in memory_rows],
                "descriptor_cache": [dict(row) for row in descriptor_rows],
            },
            default=str,
            sort_keys=True,
        )
        for forbidden in (
            expected_base64,
            expected_data_url,
            normalized.evidence.encoded_bytes.hex(),
            api_key_sentinel,
        ):
            assert forbidden not in persisted
            assert forbidden not in "\n".join(record.getMessage() for record in caplog.records)
    finally:
        if provider_client is not None:
            await provider_client.aclose()
        await memory.close()
