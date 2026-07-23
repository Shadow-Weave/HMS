"""Operator-invoked offline image upload -> retain -> recall verification.

This test never calls OpenAI.  It requires an explicitly supplied disposable
PostgreSQL URL so normal unit runs do not mutate an operator database.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from dataclasses import replace

import httpx
import pytest
from PIL import Image, ImageDraw

from hms_api.api.http import create_app
from hms_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hms_api.engine.embeddings import Embeddings
from hms_api.engine.memory_engine import MemoryEngine
from hms_api.engine.multimodal import (
    FakeMultimodalProvider,
    GroundedEntity,
    GroundedStatement,
    ImageNormalizationConfig,
    ModelMultimodalDescription,
    ModelTemporalSegment,
    ProviderUnavailableError,
    VideoProcessingConfig,
    VisibleText,
    decode_and_sample_video,
    normalize_image,
)
from hms_api.engine.parsers import MultimodalParserConfig, OpenAIMultimodalParser
from hms_api.engine.query_analyzer import DateparserQueryAnalyzer
from hms_api.engine.search.tags import TagGroupLeaf
from hms_api.engine.task_backend import SyncTaskBackend
from hms_api.migrations import run_migrations
from hms_api.models import RequestContext


class _DeterministicEmbeddings(Embeddings):
    model_name = "hms-offline-e2e-hash-v1"

    @property
    def provider_name(self) -> str:
        return "offline-test"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(self.dimension)]
            vectors.append(vector)
        return vectors


class _FailOnceImageProvider(FakeMultimodalProvider):
    """Cross the provider boundary once, then succeed on operation retry."""

    async def describe_image(self, evidence):
        if self.calls == 0:
            self.calls += 1
            raise ProviderUnavailableError(
                "provider.unavailable",
                "Synthetic provider failure",
                retryable=True,
                logical_calls=1,
                physical_attempts=1,
            )
        return await super().describe_image(evidence)


_IMAGE_MANIFEST_REQUIRED_FACTS = (
    "memory_pipeline.py",
    "pytest tests/test_memory.py",
    "AssertionError: recall mismatch",
    "upload -> retain -> recall",
    "OFFLINE_MULTIMODAL_RECALL_SENTINEL",
)
_IMAGE_MANIFEST_QUERY = "synthetic interface IDE terminal architecture panels"
_IMAGE_MANIFEST_UNSUPPORTED_ASSERTIONS = (
    "deployment succeeded",
    "a person is visible",
)


def _synthetic_ide_image(image_format: str = "PNG") -> bytes:
    """Render the exact facts used by the offline grounded manifest."""

    image = Image.new("RGB", (720, 420), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 702, 212), outline="#60a5fa", width=3)
    draw.rectangle((18, 230, 348, 402), outline="#f59e0b", width=3)
    draw.rectangle((366, 230, 702, 402), outline="#34d399", width=3)
    lines = (
        ((34, 34), "IDE"),
        ((34, 66), "FILE: memory_pipeline.py"),
        ((34, 98), "def retain_visual_memory():"),
        ((34, 130), "    return canonical_document"),
        ((34, 162), "OFFLINE_MULTIMODAL_RECALL_SENTINEL"),
        ((34, 246), "TERMINAL"),
        ((34, 278), "$ pytest tests/test_memory.py"),
        ((34, 310), "AssertionError: recall mismatch"),
        ((382, 246), "ARCHITECTURE"),
        ((382, 294), "upload -> retain -> recall"),
    )
    for position, text in lines:
        draw.text(position, text, fill="#f9fafb")
    output = io.BytesIO()
    save_kwargs = {"quality": 95, "subsampling": 0} if image_format == "JPEG" else {}
    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def _synthetic_png() -> bytes:
    return _synthetic_ide_image("PNG")


async def _create_memory(
    monkeypatch,
    database_url: str,
    *,
    file_delete_after_retain: bool = False,
) -> MemoryEngine:
    import hms_api.config

    raw_config = hms_api.config._get_raw_config()
    test_config = replace(
        raw_config,
        enable_file_upload_api=True,
        file_parser_allowlist=None,
        file_delete_after_retain=file_delete_after_retain,
        multimodal_enabled=False,
    )
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: test_config)
    run_migrations(database_url)

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
        skip_llm_verification=True,
    )
    await memory.initialize()
    return memory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_format", "extension", "mime_type"),
    [("PNG", "png", "image/png"), ("JPEG", "jpg", "image/jpeg")],
)
async def test_fake_provider_image_upload_reaches_existing_chunks_recall(
    monkeypatch,
    image_format: str,
    extension: str,
    mime_type: str,
) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_ide_image(image_format)
        filename = f"synthetic-ide.{extension}"
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename=filename,
            declared_mime=mime_type,
            config=image_config,
        )
        evidence_id = normalized.evidence.evidence_id
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A synthetic interface contains IDE, terminal, and architecture panels.",
                    evidence_ids=[evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[GroundedEntity(name="memory_pipeline.py", evidence_ids=[evidence_id], uncertainty="low")],
            observations=[],
            visible_text=[
                VisibleText(
                    text=fact,
                    evidence_ids=[evidence_id],
                    uncertainty="low",
                )
                for fact in _IMAGE_MANIFEST_REQUIRED_FACTS
            ],
            temporal_segments=[],
            limitations=[],
        )
        memory._parser_registry.register(
            OpenAIMultimodalParser(
                FakeMultimodalProvider(description),
                MultimodalParserConfig(image=image_config),
            )
        )

        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-e2e-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [
                {
                    "document_id": document_id,
                    "tags": ["project-a"],
                    "metadata": {"fixture": "offline"},
                }
            ],
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": (filename, image_bytes, mime_type)},
                data={"request": json.dumps(request_body)},
            )
            assert upload.status_code == 200, upload.text
            [operation_id] = upload.json()["operation_ids"]

            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert status.status_code == 200, status.text
            multimodal_status = status.json()["result_metadata"]["multimodal"]
            assert multimodal_status["stage"] == "recall_ready"
            assert multimodal_status["recall_ready"] is True
            assert multimodal_status["child_retain_status"] == "completed"

        recall = await memory.recall_async(
            bank_id,
            _IMAGE_MANIFEST_QUERY,
            max_tokens=2048,
            include_chunks=True,
            max_chunk_tokens=4096,
            tags=["project-a"],
            request_context=RequestContext(internal=True),
        )
        recall_json = json.dumps(recall.model_dump(mode="json"), sort_keys=True)
        for required_fact in _IMAGE_MANIFEST_REQUIRED_FACTS:
            assert required_fact in recall_json
        assert normalized.asset.sha256 in recall_json
        assert document_id in recall_json
        assert "project-a" in recall_json
        assert "media_pipeline_version" in recall_json
        assert "hms-multimodal-v1" in recall_json
        assert "media_descriptor_model" in recall_json
        assert "gpt-5-mini" in recall_json
        assert "media_audio_presence" in recall_json
        assert "media_audio_processing" in recall_json
        assert "base64," not in recall_json

        unsupported_assertion_count = sum(
            assertion in recall_json for assertion in _IMAGE_MANIFEST_UNSUPPORTED_ASSERTIONS
        )
        assert unsupported_assertion_count == 0
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_same_asset_isolated_across_banks_and_strict_tags(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-isolation.png",
            declared_mime="image/png",
            config=image_config,
        )
        evidence_id = normalized.evidence.evidence_id
        bank_a = f"multimodal-isolation-a-{uuid.uuid4().hex}"
        bank_b = f"multimodal-isolation-b-{uuid.uuid4().hex}"
        document_id = f"shared-document-{uuid.uuid4().hex}"
        sentinel_a = f"BANK_A_ONLY_{uuid.uuid4().hex}"
        sentinel_b = f"BANK_B_ONLY_{uuid.uuid4().hex}"

        def description(text: str) -> ModelMultimodalDescription:
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

        async def upload(
            client: httpx.AsyncClient,
            *,
            bank_id: str,
            tag: str,
            provider: FakeMultimodalProvider,
        ) -> str:
            memory._parser_registry.register(
                OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config))
            )
            request_body = {
                "parser": "openai_multimodal",
                "files_metadata": [{"document_id": document_id, "tags": [tag]}],
            }
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-isolation.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert response.status_code == 200, response.text
            [operation_id] = response.json()["operation_ids"]
            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert status.json()["result_metadata"]["multimodal"]["recall_ready"] is True
            return operation_id

        provider_a = FakeMultimodalProvider(description(sentinel_a))
        provider_b = FakeMultimodalProvider(description(sentinel_b))
        app = create_app(memory, initialize_memory=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await upload(client, bank_id=bank_a, tag="scope-a", provider=provider_a)
            await upload(client, bank_id=bank_b, tag="scope-b", provider=provider_b)

        # The same bytes and logical document ID still require one descriptor
        # computation per bank.  Cross-bank cache reuse would leave bank B with
        # bank A's sentinel and provider_b would not be called.
        assert provider_a.calls == 1
        assert provider_b.calls == 1

        async def recall(bank_id: str, query: str, tag: str) -> str:
            result = await memory.recall_async(
                bank_id,
                query,
                max_tokens=1024,
                include_chunks=True,
                max_chunk_tokens=4096,
                tags=[tag],
                tags_match="all_strict",
                request_context=RequestContext(internal=True),
            )
            return json.dumps(result.model_dump(mode="json"), sort_keys=True)

        bank_a_visible = await recall(bank_a, sentinel_a, "scope-a")
        bank_a_wrong_tag = await recall(bank_a, sentinel_a, "scope-b")
        bank_b_visible = await recall(bank_b, sentinel_b, "scope-b")
        bank_b_wrong_tag = await recall(bank_b, sentinel_b, "scope-a")

        assert sentinel_a in bank_a_visible
        assert sentinel_b not in bank_a_visible
        assert sentinel_a not in bank_a_wrong_tag
        assert sentinel_b in bank_b_visible
        assert sentinel_a not in bank_b_visible
        assert sentinel_b not in bank_b_wrong_tag

        async def recall_with_group(bank_id: str, query: str, group_tag: str) -> str:
            result = await memory.recall_async(
                bank_id,
                query,
                max_tokens=1024,
                include_chunks=True,
                max_chunk_tokens=4096,
                tag_groups=[TagGroupLeaf(tags=[group_tag], match="all_strict")],
                request_context=RequestContext(internal=True),
            )
            return json.dumps(result.model_dump(mode="json"), sort_keys=True)

        bank_a_group_visible = await recall_with_group(bank_a, sentinel_a, "scope-a")
        bank_a_group_hidden = await recall_with_group(bank_a, sentinel_a, "scope-b")
        assert sentinel_a in bank_a_group_visible
        assert sentinel_a not in bank_a_group_hidden

        async with memory._backend.acquire() as conn:
            descriptors = await conn.fetch(
                "SELECT bank_id, descriptor_key FROM multimodal_descriptor_cache "
                "WHERE bank_id = ANY($1::text[]) ORDER BY bank_id",
                [bank_a, bank_b],
            )
        assert [row["bank_id"] for row in descriptors] == [bank_a, bank_b]
        assert len({row["descriptor_key"] for row in descriptors}) == 2
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_image_command_dedupes_retry_and_reuses_descriptor_for_new_context(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-dedupe.png",
            declared_mime="image/png",
            config=image_config,
        )
        evidence_id = normalized.evidence.evidence_id
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A deterministic descriptor is reused across document commands.",
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
        provider = FakeMultimodalProvider(description)
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))

        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-dedupe-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [
                {
                    "document_id": document_id,
                    "context": "revision-one",
                    "tags": ["dedupe"],
                    "metadata": {"fixture": "first"},
                }
            ],
        }

        async def upload(client: httpx.AsyncClient, request: dict) -> str:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-dedupe.png", image_bytes, "image/png")},
                data={"request": json.dumps(request)},
            )
            assert response.status_code == 200, response.text
            return response.json()["operation_ids"][0]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first_operation = await upload(client, request_body)
            retry_operation = await upload(client, request_body)
            assert retry_operation == first_operation

            revised_request = json.loads(json.dumps(request_body))
            revised_request["files_metadata"][0]["context"] = "revision-two"
            revised_operation = await upload(client, revised_request)
            assert revised_operation != first_operation

            metadata_request = json.loads(json.dumps(revised_request))
            metadata_request["files_metadata"][0]["metadata"] = {"fixture": "second"}
            metadata_operation = await upload(client, metadata_request)
            assert metadata_operation not in {first_operation, revised_operation}

            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{metadata_operation}")
            assert status.json()["result_metadata"]["multimodal"]["recall_ready"] is True

        # One provider description serves the original command and the new
        # context command.  The exact retry allocates neither a command nor an
        # extra retained document/source object.
        assert provider.calls == 1
        tenant_scope = "public"
        bank_scope = hashlib.sha256(f"{tenant_scope}\0{bank_id}".encode()).hexdigest()[:24]
        async with memory._backend.acquire() as conn:
            descriptor_count = await conn.fetchval(
                "SELECT COUNT(*) FROM multimodal_descriptor_cache WHERE bank_id = $1",
                bank_id,
            )
            command_count = await conn.fetchval(
                "SELECT COUNT(*) FROM multimodal_document_commands WHERE bank_id = $1",
                bank_id,
            )
            deleted_source_count = await conn.fetchval(
                "SELECT COUNT(*) FROM multimodal_document_commands "
                "WHERE bank_id = $1 AND source_deleted_at IS NOT NULL",
                bank_id,
            )
            document_count = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE bank_id = $1 AND id = $2",
                bank_id,
                document_id,
            )
            source_count = await conn.fetchval(
                "SELECT COUNT(*) FROM file_storage WHERE storage_key LIKE $1",
                f"media/{bank_scope}/%",
            )
            head = await conn.fetchrow(
                "SELECT next_sequence, published_sequence, active_sequence "
                "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            document_retain_params = conn.parse_json(
                await conn.fetchval(
                    "SELECT retain_params FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
            )
        assert descriptor_count == 1
        assert command_count == 3
        assert deleted_source_count == 0
        assert document_count == 1
        assert source_count == 3
        assert dict(head) == {"next_sequence": 4, "published_sequence": 3, "active_sequence": None}
        assert document_retain_params["metadata"]["fixture"] == "second"
        assert document_retain_params["metadata"]["media_source_available"] == "true"
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_anonymous_image_retry_uses_stable_bank_scoped_document_in_postgres(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="anonymous.png",
            declared_mime="image/png",
            config=image_config,
        )
        provider = FakeMultimodalProvider(
            ModelMultimodalDescription(
                summary=[
                    GroundedStatement(
                        text="An anonymous visual asset has a deterministic scoped document identity.",
                        evidence_ids=[normalized.evidence.evidence_id],
                        uncertainty="low",
                    )
                ],
                entities=[],
                observations=[],
                visible_text=[],
                temporal_segments=[],
                limitations=[],
            )
        )
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))
        app = create_app(memory, initialize_memory=False)
        bank_a = f"anonymous-a-{uuid.uuid4().hex}"
        bank_b = f"anonymous-b-{uuid.uuid4().hex}"

        async def upload(client: httpx.AsyncClient, bank_id: str) -> str:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("anonymous.png", image_bytes, "image/png")},
                data={"request": json.dumps({"parser": "openai_multimodal"})},
            )
            assert response.status_code == 200, response.text
            return response.json()["operation_ids"][0]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await upload(client, bank_a)
            repeat = await upload(client, bank_a)
            other_bank = await upload(client, bank_b)

        assert first == repeat
        assert other_bank != first
        async with memory._backend.acquire() as conn:
            rows = await conn.fetch(
                "SELECT bank_id, document_id, command_key, operation_id "
                "FROM multimodal_document_commands WHERE bank_id = ANY($1::text[]) ORDER BY bank_id",
                [bank_a, bank_b],
            )
            document_count = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE bank_id = ANY($1::text[])",
                [bank_a, bank_b],
            )

        assert len(rows) == 2
        assert {row["bank_id"] for row in rows} == {bank_a, bank_b}
        assert all(row["document_id"].startswith("file_mm_") for row in rows)
        assert len({row["document_id"] for row in rows}) == 2
        assert len({row["command_key"] for row in rows}) == 2
        assert document_count == 2
        assert provider.calls == 2
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_corrected_declared_mime_revalidates_as_new_postgres_command(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="corrected.png",
            declared_mime="image/png",
            config=image_config,
        )
        provider = FakeMultimodalProvider(
            ModelMultimodalDescription(
                summary=[
                    GroundedStatement(
                        text="Corrected type hints are revalidated before one descriptor is published.",
                        evidence_ids=[normalized.evidence.evidence_id],
                        uncertainty="low",
                    )
                ],
                entities=[],
                observations=[],
                visible_text=[],
                temporal_segments=[],
                limitations=[],
            )
        )
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))
        app = create_app(memory, initialize_memory=False)
        bank_id = f"corrected-mime-{uuid.uuid4().hex}"
        document_id = f"corrected-document-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id}],
        }

        async def upload(client: httpx.AsyncClient, mime: str, filename: str = "corrected.png") -> str:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": (filename, image_bytes, mime)},
                data={"request": json.dumps(request_body)},
            )
            assert response.status_code == 200, response.text
            return response.json()["operation_ids"][0]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            wrong_operation = await upload(client, "image/jpeg")
            corrected_operation = await upload(client, "image/png")

        assert corrected_operation != wrong_operation
        async with memory._backend.acquire() as conn:
            commands = await conn.fetch(
                "SELECT sequence, command_key, descriptor_key, retain_input_fingerprint, status "
                "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2 ORDER BY sequence",
                bank_id,
                document_id,
            )

        assert [row["status"] for row in commands] == ["failed", "completed"]
        # The corrected validator identity must not reuse a checkpoint created
        # under a conflicting MIME hint; the successful command recomputes its
        # descriptor after local validation.
        assert commands[0]["descriptor_key"] != commands[1]["descriptor_key"]
        assert commands[0]["retain_input_fingerprint"] != commands[1]["retain_input_fingerprint"]
        assert commands[0]["command_key"] != commands[1]["command_key"]
        assert provider.calls == 1

        # Reverse the order for a fresh logical document: a completed valid
        # descriptor must not satisfy a later command whose extension hint is
        # invalid.  This specifically guards the descriptor-cache fast path.
        second_document_id = f"corrected-document-two-{uuid.uuid4().hex}"
        request_body["files_metadata"][0]["document_id"] = second_document_id
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            valid_first = await upload(client, "image/png")
            invalid_after = await upload(client, "image/png", "corrected.jpg")
        assert valid_first != invalid_after
        async with memory._backend.acquire() as conn:
            reverse_commands = await conn.fetch(
                "SELECT sequence, command_key, descriptor_key, status "
                "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2 ORDER BY sequence",
                bank_id,
                second_document_id,
            )
        assert [row["status"] for row in reverse_commands] == ["completed", "failed"]
        assert reverse_commands[0]["descriptor_key"] != reverse_commands[1]["descriptor_key"]
        # The valid second document reuses the same hint/policy-scoped
        # descriptor; only the invalid extension command must be rejected
        # before another provider call.
        assert provider.calls == 1
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_delete_after_retain_removes_source_but_keeps_descriptor_and_provenance(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url, file_delete_after_retain=True)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-delete.png",
            declared_mime="image/png",
            config=image_config,
        )
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="The source lifecycle test has grounded visual evidence.",
                    evidence_ids=[normalized.evidence.evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        )
        memory._parser_registry.register(
            OpenAIMultimodalParser(
                FakeMultimodalProvider(description),
                MultimodalParserConfig(image=image_config),
            )
        )

        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-delete-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id, "tags": ["source-lifecycle"]}],
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-delete.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert response.status_code == 200, response.text
            operation_id = response.json()["operation_ids"][0]
            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert status.json()["result_metadata"]["multimodal"]["recall_ready"] is True

        async with memory._backend.acquire() as conn:
            command = await conn.fetchrow(
                "SELECT source_storage_key, source_deleted_at, status "
                "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            descriptor = await conn.fetchrow(
                "SELECT status, canonical_markdown, expires_at FROM multimodal_descriptor_cache WHERE bank_id = $1",
                bank_id,
            )
            retain_params = conn.parse_json(
                await conn.fetchval(
                    "SELECT retain_params FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
            )
        assert command["status"] == "completed"
        assert command["source_deleted_at"] is not None
        assert await memory._file_storage.exists(command["source_storage_key"]) is False
        assert descriptor["status"] == "completed"
        assert normalized.asset.sha256 in descriptor["canonical_markdown"]
        assert descriptor["expires_at"] is not None
        assert retain_params["metadata"]["media_asset_sha256"] == normalized.asset.sha256
        assert retain_params["metadata"]["media_pipeline_version"] == "hms-multimodal-v1"
        assert retain_params["metadata"]["media_audio_presence"] == "absent"
        assert retain_params["metadata"]["media_audio_processing"] == "not_requested"
        assert retain_params["metadata"]["media_source_available"] == "false"
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_failed_provider_command_restarts_from_durable_ledger_without_duplicate_document(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-retry.png",
            declared_mime="image/png",
            config=image_config,
        )
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="The retried command publishes exactly one document.",
                    evidence_ids=[normalized.evidence.evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        )
        provider = _FailOnceImageProvider(description)
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))

        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-retry-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id, "tags": ["retry"]}],
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-retry.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert response.status_code == 200, response.text
            operation_id = response.json()["operation_ids"][0]
            failed = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert failed.json()["status"] == "failed"
            assert failed.json()["result_metadata"]["multimodal"]["sanitized_error_code"] == "provider.unavailable"

        async with memory._backend.acquire() as conn:
            raw_task = await conn.fetchval(
                "SELECT task_payload FROM async_operations WHERE operation_id = $1",
                uuid.UUID(operation_id),
            )
            task_payload = conn.parse_json(raw_task)
            command_before = await conn.fetchrow(
                "SELECT status FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            descriptor_before = await conn.fetchrow(
                "SELECT status, provider_started_at FROM multimodal_descriptor_cache WHERE bank_id = $1",
                bank_id,
            )
            await conn.execute(
                "UPDATE async_operations SET status = 'pending', error_message = NULL WHERE operation_id = $1",
                uuid.UUID(operation_id),
            )
        assert command_before["status"] == "failed"
        assert descriptor_before["status"] == "failed"
        assert descriptor_before["provider_started_at"] is not None

        await memory.execute_task(task_payload)

        status = await memory.get_operation_status(
            bank_id,
            operation_id,
            request_context=RequestContext(internal=True),
        )
        assert status["status"] == "completed"
        assert status["result_metadata"]["multimodal"]["recall_ready"] is True
        assert status["result_metadata"]["multimodal"]["possible_duplicate_provider_attempt"] is True
        assert provider.calls == 2
        async with memory._backend.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
                == 1
            )
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_failed_child_retain_closes_command_and_retries_with_new_child(monkeypatch) -> None:
    """A terminal child failure must release the command for a new child retry."""

    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url, file_delete_after_retain=True)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-child-failure.png",
            declared_mime="image/png",
            config=image_config,
        )
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A failed child retain can be retried from the durable descriptor.",
                    evidence_ids=[normalized.evidence.evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        )
        provider = FakeMultimodalProvider(description)
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))

        submitted_tasks: list[dict] = []

        async def queue_only(payload: dict) -> None:
            submitted_tasks.append(dict(payload))

        memory._task_backend.submit_task = queue_only  # type: ignore[method-assign]
        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-child-failure-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id, "tags": ["child-failure"]}],
        }

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-child-failure.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert upload.status_code == 200, upload.text
            [parent_operation_id] = upload.json()["operation_ids"]

            # Conversion succeeds and creates the first child, but leave the
            # child unexecuted so the terminal failure entry can be exercised.
            await memory.execute_task(dict(submitted_tasks[0]))
            child_payloads = [
                payload
                for payload in submitted_tasks
                if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
            ]
            assert len(child_payloads) == 1
            old_child_payload = child_payloads[0]
            old_child_id = old_child_payload["operation_id"]

            async with memory._backend.acquire() as conn:
                await conn.execute(
                    "UPDATE async_operations SET status = 'processing' WHERE operation_id = $1",
                    uuid.UUID(old_child_id),
                )

            await memory._mark_operation_failed(
                old_child_id,
                "synthetic child retain failure",
                "synthetic traceback must not enter parent metadata",
            )

            failed_status = await client.get(f"/v1/default/banks/{bank_id}/operations/{parent_operation_id}")
            assert failed_status.status_code == 200, failed_status.text
            failed_json = failed_status.json()
            assert failed_json["status"] == "completed"
            failed_multimodal = failed_json["result_metadata"]["multimodal"]
            assert failed_multimodal["stage"] == "retain_failed"
            assert failed_multimodal["child_retain_status"] == "failed"
            assert failed_multimodal["recall_ready"] is False
            assert failed_multimodal["retryable"] is True
            assert failed_multimodal["sanitized_error_code"] == "retain_failed"
            assert "synthetic child retain failure" not in json.dumps(failed_json)
            assert "synthetic traceback" not in json.dumps(failed_json)

            async with memory._backend.acquire() as conn:
                command = await conn.fetchrow(
                    "SELECT status, child_retain_operation_id, descriptor_key, "
                    "source_storage_key, source_deleted_at "
                    "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
                head = await conn.fetchrow(
                    "SELECT active_sequence, published_sequence "
                    "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
                old_child_status = await conn.fetchval(
                    "SELECT status FROM async_operations WHERE operation_id = $1",
                    uuid.UUID(old_child_id),
                )
            assert command["status"] == "failed"
            assert str(command["child_retain_operation_id"]) == old_child_id
            assert command["source_deleted_at"] is not None
            assert await memory._file_storage.exists(command["source_storage_key"]) is False
            assert dict(head) == {"active_sequence": None, "published_sequence": 0}
            assert old_child_status == "failed"

            # Expire the command's only safe descriptor checkpoint and add
            # more rows than one lazy cleanup pass can remove.  The failed
            # child command must pin its own checkpoint because its source is
            # already deleted; unrelated expired rows remain ordinary misses.
            decoys = [
                (
                    bank_id,
                    hashlib.sha256(f"decoy-descriptor-{index}".encode()).hexdigest(),
                    hashlib.sha256(f"decoy-asset-{index}".encode()).hexdigest(),
                    hashlib.sha256(f"decoy-pipeline-{index}".encode()).hexdigest(),
                )
                for index in range(201)
            ]
            async with memory._backend.acquire() as conn:
                await conn.execute(
                    "UPDATE multimodal_descriptor_cache "
                    "SET expires_at = NOW() - INTERVAL '1 second' "
                    "WHERE bank_id = $1 AND descriptor_key = $2",
                    bank_id,
                    command["descriptor_key"],
                )
                await conn.executemany(
                    "INSERT INTO multimodal_descriptor_cache "
                    "(bank_id, descriptor_key, asset_sha256, pipeline_fingerprint, status, "
                    "canonical_markdown, provenance_metadata, entities, checkpointed_at, expires_at) "
                    "VALUES ($1, $2, $3, $4, 'completed', '# expired decoy', '{}'::jsonb, "
                    "'[]'::jsonb, NOW() - INTERVAL '2 days', NOW() - INTERVAL '2 days')",
                    decoys,
                )

            retried = await memory.retry_operation(
                bank_id,
                parent_operation_id,
                request_context=RequestContext(internal=True),
            )
            assert retried["success"] is True

            # The parent task is re-run from its immutable payload.  Although
            # its source no longer exists and TTL elapsed, the command-scoped
            # checkpoint avoids another provider call and creates a fresh child.
            await memory.execute_task(dict(submitted_tasks[0]))
            child_payloads = [
                payload
                for payload in submitted_tasks
                if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
            ]
            assert len(child_payloads) == 2
            new_child_payload = child_payloads[-1]
            assert new_child_payload["operation_id"] != old_child_id
            await memory.execute_task(dict(new_child_payload))

            final_status = await client.get(f"/v1/default/banks/{bank_id}/operations/{parent_operation_id}")
            assert final_status.status_code == 200, final_status.text
            final_json = final_status.json()
            assert final_json["status"] == "completed"
            assert final_json["result_metadata"]["multimodal"]["stage"] == "recall_ready"
            assert final_json["result_metadata"]["multimodal"]["child_retain_status"] == "completed"
            assert final_json["result_metadata"]["multimodal"]["recall_ready"] is True
            assert provider.calls == 1

            async with memory._backend.acquire() as conn:
                command = await conn.fetchrow(
                    "SELECT status, child_retain_operation_id "
                    "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
                old_child_status = await conn.fetchval(
                    "SELECT status FROM async_operations WHERE operation_id = $1",
                    uuid.UUID(old_child_id),
                )
                new_child_status = await conn.fetchval(
                    "SELECT status FROM async_operations WHERE operation_id = $1",
                    uuid.UUID(new_child_payload["operation_id"]),
                )
                document_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
                command_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
            assert command["status"] == "completed"
            assert str(command["child_retain_operation_id"]) == new_child_payload["operation_id"]
            assert old_child_status == "failed"
            assert new_child_status == "completed"
            assert document_count == 1
            assert command_count == 1

            async with memory._backend.acquire() as conn:
                expired_after_first_cleanup = await conn.fetchval(
                    "SELECT COUNT(*) FROM multimodal_descriptor_cache "
                    "WHERE bank_id = $1 AND status = 'completed' AND expires_at <= NOW()",
                    bank_id,
                )
            assert expired_after_first_cleanup == 102

            # A new command for the same bytes is not allowed to borrow the
            # expired recovery exception.  Its cleanup pass removes another
            # 100 decoys but deliberately leaves the expired real descriptor
            # physically present; the read-side TTL predicate must still force
            # one new provider description.
            revised_request = {
                "parser": "openai_multimodal",
                "files_metadata": [
                    {
                        "document_id": document_id,
                        "context": "new command after descriptor expiry",
                        "tags": ["child-failure"],
                    }
                ],
            }
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                revised_upload = await client.post(
                    f"/v1/default/banks/{bank_id}/files/retain",
                    files={"files": ("synthetic-child-failure.png", image_bytes, "image/png")},
                    data={"request": json.dumps(revised_request)},
                )
                assert revised_upload.status_code == 200, revised_upload.text
                [revised_parent_id] = revised_upload.json()["operation_ids"]

            revised_parent_payload = next(
                payload
                for payload in submitted_tasks
                if payload.get("type") == "file_convert_retain" and payload.get("operation_id") == revised_parent_id
            )
            await memory.execute_task(dict(revised_parent_payload))
            assert provider.calls == 2
            async with memory._backend.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT COUNT(*) FROM multimodal_descriptor_cache "
                        "WHERE bank_id = $1 AND status = 'completed' AND expires_at <= NOW()",
                        bank_id,
                    )
                    == 1
                )
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_cancelled_pending_command_restarts_and_finishes_consistently(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-cancel.png",
            declared_mime="image/png",
            config=image_config,
        )
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A cancelled queued command can resume from its immutable source.",
                    evidence_ids=[normalized.evidence.evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        )
        provider = FakeMultimodalProvider(description)
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))

        submitted_tasks: list[dict] = []

        async def queue_only(payload: dict) -> None:
            submitted_tasks.append(dict(payload))

        memory._task_backend.submit_task = queue_only  # type: ignore[method-assign]
        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-cancel-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [{"document_id": document_id, "tags": ["cancel-restart"]}],
        }

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-cancel.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert upload.status_code == 200, upload.text
            operation_id = upload.json()["operation_ids"][0]
            assert len(submitted_tasks) == 1

            cancelled = await client.delete(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert cancelled.status_code == 200, cancelled.text

            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            assert status.json()["status"] == "cancelled"

        async with memory._backend.acquire() as conn:
            command = await conn.fetchrow(
                "SELECT status, source_storage_key FROM multimodal_document_commands "
                "WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            head = await conn.fetchrow(
                "SELECT next_sequence, published_sequence, active_sequence "
                "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            descriptor_count = await conn.fetchval(
                "SELECT COUNT(*) FROM multimodal_descriptor_cache WHERE bank_id = $1",
                bank_id,
            )
        assert command["status"] == "cancelled"
        assert await memory._file_storage.exists(command["source_storage_key"]) is True
        assert dict(head) == {"next_sequence": 2, "published_sequence": 0, "active_sequence": None}
        assert descriptor_count == 0

        # A stale queue delivery after cancellation must observe the operation
        # tombstone before decode/provider work and leave the durable command
        # restartable.
        await memory.execute_task(dict(submitted_tasks[0]))
        assert provider.calls == 0
        async with memory._backend.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT status FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                    bank_id,
                    document_id,
                )
                == "cancelled"
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM multimodal_descriptor_cache WHERE bank_id = $1",
                    bank_id,
                )
                == 0
            )

        retried = await memory.retry_operation(
            bank_id,
            operation_id,
            request_context=RequestContext(internal=True),
        )
        assert retried["success"] is True
        await memory.execute_task(dict(submitted_tasks[0]))
        child_payload = next(
            payload
            for payload in submitted_tasks[1:]
            if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
        )
        await memory.execute_task(dict(child_payload))

        final_status = await memory.get_operation_status(
            bank_id,
            operation_id,
            request_context=RequestContext(internal=True),
        )
        assert final_status["status"] == "completed"
        assert final_status["result_metadata"]["multimodal"]["recall_ready"] is True
        assert final_status["result_metadata"]["multimodal"]["child_retain_status"] == "completed"
        assert provider.calls == 1

        async with memory._backend.acquire() as conn:
            command = await conn.fetchrow(
                "SELECT status, child_retain_operation_id FROM multimodal_document_commands "
                "WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            head = await conn.fetchrow(
                "SELECT next_sequence, published_sequence, active_sequence "
                "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            child_status = await conn.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                command["child_retain_operation_id"],
            )
            document_count = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE bank_id = $1 AND id = $2",
                bank_id,
                document_id,
            )
            command_count = await conn.fetchval(
                "SELECT COUNT(*) FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert command["status"] == "completed"
        assert child_status == "completed"
        assert document_count == 1
        assert command_count == 1
        assert dict(head) == {"next_sequence": 2, "published_sequence": 1, "active_sequence": None}
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_older_child_is_cancelled_when_newer_command_owns_publication(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    memory = await _create_memory(monkeypatch, database_url)
    try:
        image_bytes = _synthetic_png()
        image_config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=1_000_000)
        normalized = normalize_image(
            file_data=image_bytes,
            filename="synthetic-supersede.png",
            declared_mime="image/png",
            config=image_config,
        )
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="Only the newest accepted command may publish this media document.",
                    evidence_ids=[normalized.evidence.evidence_id],
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        )
        provider = FakeMultimodalProvider(description)
        memory._parser_registry.register(OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config)))

        submitted_tasks: list[dict] = []

        async def queue_only(payload: dict) -> None:
            submitted_tasks.append(dict(payload))

        memory._task_backend.submit_task = queue_only  # type: ignore[method-assign]
        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-supersede-{uuid.uuid4().hex}"
        document_id = f"asset-{uuid.uuid4().hex}"

        async def upload(client: httpx.AsyncClient, *, context: str) -> str:
            request_body = {
                "parser": "openai_multimodal",
                "files_metadata": [
                    {
                        "document_id": document_id,
                        "context": context,
                        "tags": ["ordered-publication"],
                        "metadata": {"revision": context},
                    }
                ],
            }
            response = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-supersede.png", image_bytes, "image/png")},
                data={"request": json.dumps(request_body)},
            )
            assert response.status_code == 200, response.text
            return response.json()["operation_ids"][0]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            older_operation_id = await upload(client, context="revision-one")
            older_parent_payload = submitted_tasks[-1]
            await memory.execute_task(dict(older_parent_payload))
            older_child_payload = next(
                payload
                for payload in submitted_tasks
                if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
            )

            newer_operation_id = await upload(client, context="revision-two")
            newer_parent_payload = submitted_tasks[-1]
            await memory.execute_task(dict(newer_parent_payload))
            child_payloads = [
                payload
                for payload in submitted_tasks
                if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
            ]
            assert len(child_payloads) == 2
            newer_child_payload = child_payloads[-1]

            await memory.execute_task(dict(older_child_payload))
            older_status = await client.get(f"/v1/default/banks/{bank_id}/operations/{older_operation_id}")
            assert older_status.status_code == 200, older_status.text
            older_multimodal = older_status.json()["result_metadata"]["multimodal"]
            assert older_status.json()["status"] == "completed"
            assert older_multimodal["stage"] == "failed"
            assert older_multimodal["sanitized_error_code"] == "multimodal.command_superseded"
            assert older_multimodal["child_retain_status"] == "cancelled"
            assert older_multimodal["recall_ready"] is False

        async with memory._backend.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
                == 0
            )
            interim_commands = await conn.fetch(
                "SELECT sequence, status, child_retain_operation_id "
                "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2 "
                "ORDER BY sequence",
                bank_id,
                document_id,
            )
            older_child_status = await conn.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                interim_commands[0]["child_retain_operation_id"],
            )
        assert [row["status"] for row in interim_commands] == ["superseded", "retaining"]
        assert older_child_status == "cancelled"

        await memory.execute_task(dict(newer_child_payload))
        newer_status = await memory.get_operation_status(
            bank_id,
            newer_operation_id,
            request_context=RequestContext(internal=True),
        )
        assert newer_status["status"] == "completed"
        assert newer_status["result_metadata"]["multimodal"]["stage"] == "recall_ready"
        assert newer_status["result_metadata"]["multimodal"]["child_retain_status"] == "completed"
        assert provider.calls == 1

        # Admit a third revision so the already-published second child becomes
        # stale, then replay that completed child.  The replay must be a
        # no-op: it must not downgrade child 2, command 2, or its parent view.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            third_operation_id = await upload(client, context="revision-three")
            third_parent_payload = submitted_tasks[-1]
            await memory.execute_task(dict(third_parent_payload))
            child_payloads = [
                payload
                for payload in submitted_tasks
                if payload.get("type") == "batch_retain" and payload.get("_multimodal_command")
            ]
            assert len(child_payloads) == 3
            third_child_payload = child_payloads[-1]

            await memory.execute_task(dict(newer_child_payload))
            replayed_status = await client.get(f"/v1/default/banks/{bank_id}/operations/{newer_operation_id}")
            assert replayed_status.status_code == 200, replayed_status.text
            replayed_multimodal = replayed_status.json()["result_metadata"]["multimodal"]
            assert replayed_status.json()["status"] == "completed"
            assert replayed_multimodal["stage"] == "recall_ready"
            assert replayed_multimodal["child_retain_status"] == "completed"
            assert replayed_multimodal["recall_ready"] is True

        async with memory._backend.acquire() as conn:
            command_two = await conn.fetchrow(
                "SELECT status FROM multimodal_document_commands "
                "WHERE bank_id = $1 AND document_id = $2 AND sequence = 2",
                bank_id,
                document_id,
            )
            head_before_third_publish = await conn.fetchrow(
                "SELECT next_sequence, published_sequence, active_sequence "
                "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            child_two_status = await conn.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                uuid.UUID(newer_child_payload["operation_id"]),
            )
        assert command_two["status"] == "completed"
        assert child_two_status == "completed"
        assert dict(head_before_third_publish) == {
            "next_sequence": 4,
            "published_sequence": 2,
            "active_sequence": 3,
        }

        await memory.execute_task(dict(third_child_payload))
        third_status = await memory.get_operation_status(
            bank_id,
            third_operation_id,
            request_context=RequestContext(internal=True),
        )
        assert third_status["status"] == "completed"
        assert third_status["result_metadata"]["multimodal"]["stage"] == "recall_ready"
        assert third_status["result_metadata"]["multimodal"]["child_retain_status"] == "completed"
        assert provider.calls == 1

        async with memory._backend.acquire() as conn:
            final_commands = await conn.fetch(
                "SELECT sequence, status, child_retain_operation_id "
                "FROM multimodal_document_commands WHERE bank_id = $1 AND document_id = $2 "
                "ORDER BY sequence",
                bank_id,
                document_id,
            )
            head = await conn.fetchrow(
                "SELECT next_sequence, published_sequence, active_sequence "
                "FROM multimodal_document_heads WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
            retain_params = conn.parse_json(
                await conn.fetchval(
                    "SELECT retain_params FROM documents WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    document_id,
                )
            )
            child_statuses = [
                await conn.fetchval(
                    "SELECT status FROM async_operations WHERE operation_id = $1",
                    row["child_retain_operation_id"],
                )
                for row in final_commands
            ]
        assert [row["status"] for row in final_commands] == ["superseded", "completed", "completed"]
        assert child_statuses == ["cancelled", "completed", "completed"]
        assert dict(head) == {"next_sequence": 4, "published_sequence": 3, "active_sequence": None}
        assert retain_params["metadata"]["revision"] == "revision-three"
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_fake_provider_video_upload_recall_preserves_two_timed_states(monkeypatch) -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    from tests.test_multimodal_video import _config as video_config
    from tests.test_multimodal_video import _make_mp4

    memory = await _create_memory(monkeypatch, database_url)
    try:
        video_bytes = _make_mp4(frame_count=30)
        processing_config: VideoProcessingConfig = video_config(max_frames=5)
        decoded = decode_and_sample_video(
            file_data=video_bytes,
            filename="synthetic-coding.mp4",
            declared_mime="video/mp4",
            config=processing_config,
        )
        evidence = list(decoded.evidence)
        assert [item.timestamp_ms for item in evidence].count(1_400) == 1
        mapped_segments: list[ModelTemporalSegment] = []
        for index, item in enumerate(evidence):
            terminal_state = "TEST FAILED" if item.timestamp_ms == 1_400 else "terminal idle"
            mapped_segments.append(
                ModelTemporalSegment(
                    segment_id=f"segment-{index:03d}",
                    summary=[
                        GroundedStatement(
                            text=f"The synthetic terminal shows {terminal_state}.",
                            evidence_ids=[item.evidence_id],
                            uncertainty="low",
                        )
                    ],
                    observations=[],
                    visible_text=[
                        VisibleText(
                            text=terminal_state,
                            evidence_ids=[item.evidence_id],
                            uncertainty="low",
                        )
                    ],
                    evidence_ids=[item.evidence_id],
                )
            )
        all_evidence_ids = [item.evidence_id for item in evidence]
        description = ModelMultimodalDescription(
            summary=[
                GroundedStatement(
                    text="A synthetic coding session changes between multiple editor states.",
                    evidence_ids=all_evidence_ids,
                    uncertainty="low",
                )
            ],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=mapped_segments,
            limitations=[],
        )
        provider = FakeMultimodalProvider(
            description,
            segments={segment.segment_id: segment for segment in mapped_segments},
        )
        memory._parser_registry.register(
            OpenAIMultimodalParser(
                provider,
                MultimodalParserConfig(
                    video=processing_config,
                    max_frames_per_call=1,
                    sampling_version=processing_config.sampling_version,
                ),
            )
        )

        app = create_app(memory, initialize_memory=False)
        bank_id = f"multimodal-video-e2e-{uuid.uuid4().hex}"
        document_id = f"video-{uuid.uuid4().hex}"
        request_body = {
            "parser": "openai_multimodal",
            "files_metadata": [
                {
                    "document_id": document_id,
                    "tags": ["project-video"],
                }
            ],
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = await client.post(
                f"/v1/default/banks/{bank_id}/files/retain",
                files={"files": ("synthetic-coding.mp4", video_bytes, "video/mp4")},
                data={"request": json.dumps(request_body)},
            )
            assert upload.status_code == 200, upload.text
            [operation_id] = upload.json()["operation_ids"]
            status = await client.get(f"/v1/default/banks/{bank_id}/operations/{operation_id}")
            multimodal_status = status.json()["result_metadata"]["multimodal"]
            assert multimodal_status["media_kind"] == "video"
            assert multimodal_status["recall_ready"] is True
            assert multimodal_status["asset_sha256"] == decoded.asset.sha256

        assert provider.calls == len(mapped_segments) + 1
        recall_payloads: dict[str, str] = {}
        for expected_state in ("terminal idle", "TEST FAILED"):
            recall = await memory.recall_async(
                bank_id,
                expected_state,
                max_tokens=2048,
                include_chunks=True,
                max_chunk_tokens=8192,
                tags=["project-video"],
                request_context=RequestContext(internal=True),
            )
            recall_json = json.dumps(recall.model_dump(mode="json"), sort_keys=True)
            recall_payloads[expected_state] = recall_json
            assert expected_state in recall_json
            assert decoded.asset.sha256 in recall_json
            assert document_id in recall_json
            assert "media_pipeline_version" in recall_json
            assert "hms-multimodal-v1" in recall_json
            assert "media_audio_presence" in recall_json
            assert "media_audio_processing" in recall_json
            assert "segment=segment-" in recall_json
            assert "00:00:" in recall_json
            assert "base64," not in recall_json
        assert "00:00:00.000" in recall_payloads["terminal idle"]
        assert "00:00:01.400" in recall_payloads["TEST FAILED"]
        assert "TEST PASSED" not in recall_payloads["TEST FAILED"]
        assert "production deployment" not in recall_payloads["TEST FAILED"]
    finally:
        await memory.close()
