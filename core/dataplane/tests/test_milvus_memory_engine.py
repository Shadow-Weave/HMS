"""End-to-end HMS recall and lifecycle tests with PostgreSQL and Milvus Lite."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

pytest.importorskip("pymilvus")

from hms_api import MemoryEngine, RequestContext
from hms_api.engine.cross_encoder import CrossEncoderModel
from hms_api.engine.embeddings import Embeddings
from hms_api.engine.query_analyzer import DateparserQueryAnalyzer
from hms_api.engine.retain.bank_utils import DEFAULT_DISPOSITION
from hms_api.engine.task_backend import SyncTaskBackend
from hms_api.engine.vector_index import VectorIndexRecord
from hms_api.engine.vector_index.milvus import MilvusVectorIndex


class DeterministicEmbeddings(Embeddings):
    """Small deterministic semantic model with the production schema dimension."""

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [0.0] * self.dimension
            if "apple" in lowered:
                vector[0] = 1.0
            elif "banana" in lowered:
                vector[1] = 1.0
            elif "cherry" in lowered:
                vector[2] = 1.0
            else:
                vector[3] = 1.0
            vectors.append(vector)
        return vectors


class DeterministicCrossEncoder(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def scores_are_normalized(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0 if set(query.lower().split()) & set(document.lower().split()) else 0.1 for query, document in pairs]


@pytest.mark.asyncio
async def test_memory_engine_milvus_end_to_end_lifecycle_and_fallback(pg0_db_url, tmp_path, monkeypatch):
    index = MilvusVectorIndex(
        uri=str(tmp_path / "engine.db"),
        collection_name=f"hms_engine_{uuid.uuid4().hex}",
    )
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        embeddings=DeterministicEmbeddings(),
        cross_encoder=DeterministicCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        vector_index=index,
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
        lazy_reranker=True,
    )
    await memory.initialize()
    request_context = RequestContext()
    bank_id = f"milvus-e2e-{uuid.uuid4().hex}"

    try:
        first_ids = await memory.retain_batch_async(
            bank_id,
            [
                {
                    "content": "Apple orchards need careful spring pruning.",
                    "document_id": "doc-apple",
                    "tags": ["user:alice", "topic:fruit"],
                },
                {
                    "content": "Banana plants thrive in warm humid climates.",
                    "document_id": "doc-banana",
                    "tags": ["user:bob", "topic:fruit"],
                },
            ],
            fact_type_override="world",
            request_context=request_context,
        )
        apple_id = first_ids[0][0]
        banana_id = first_ids[1][0]

        direct_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
            tags=["user:alice"],
            tags_match="all_strict",
        )
        assert [hit.id for hit in direct_hits["world"]] == [apple_id]

        recall = await memory.recall_async(
            bank_id,
            "apple",
            fact_type=["world"],
            tags=["user:alice"],
            tags_match="all_strict",
            request_context=request_context,
        )
        assert recall.results
        assert recall.results[0].id == apple_id
        assert "Apple orchards" in recall.results[0].text

        # SQL hydration drops and prunes an external record that has no
        # canonical memory_units row.
        stale_id = "00000000-0000-0000-0000-999999999999"
        await index.upsert(
            [
                VectorIndexRecord(
                    id=stale_id,
                    namespace="public",
                    bank_id=bank_id,
                    fact_type="world",
                    embedding=DeterministicEmbeddings().encode(["apple"])[0],
                    tags=["user:alice"],
                    updated_at=datetime.now(UTC),
                )
            ]
        )
        await memory.recall_async(
            bank_id,
            "apple",
            fact_type=["world"],
            tags=["user:alice"],
            tags_match="all_strict",
            request_context=request_context,
        )
        direct_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
        )
        assert stale_id not in {hit.id for hit in direct_hits["world"]}

        # External search failures must preserve recall through the canonical
        # PostgreSQL vector index.
        original_search = index.search

        async def fail_search(*args, **kwargs):
            raise RuntimeError("simulated Milvus outage")

        monkeypatch.setattr(index, "search", fail_search)
        fallback_recall = await memory.recall_async(
            bank_id,
            "banana",
            fact_type=["world"],
            request_context=request_context,
        )
        assert fallback_recall.results
        assert fallback_recall.results[0].id == banana_id
        monkeypatch.setattr(index, "search", original_search)

        # Re-ingesting a document replaces deleted vectors and refreshes tags.
        replacement_ids = await memory.retain_batch_async(
            bank_id,
            [
                {
                    "content": "Cherry trees prefer well-drained soil.",
                    "document_id": "doc-apple",
                    "tags": ["user:carol", "topic:fruit"],
                }
            ],
            fact_type_override="world",
            request_context=request_context,
        )
        replacement_id = replacement_ids[0][0]
        assert replacement_id != apple_id
        cherry_hits = await index.search(
            DeterministicEmbeddings().encode(["cherry"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
            tags=["user:carol"],
            tags_match="all_strict",
        )
        assert [hit.id for hit in cherry_hits["world"]] == [replacement_id]
        all_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
        )
        assert apple_id not in {hit.id for hit in all_hits["world"]}

        assert await memory.update_document(
            "doc-banana",
            bank_id,
            tags=["user:alice", "updated"],
            request_context=request_context,
        )
        updated_hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
            tags=["updated"],
            tags_match="all_strict",
        )
        assert [hit.id for hit in updated_hits["world"]] == [banana_id]

        # A full rebuild restores a deliberately cleared projection.
        await index.delete_namespace("public")
        assert not (
            await index.search(
                DeterministicEmbeddings().encode(["banana"])[0],
                namespace="public",
                bank_id=bank_id,
                fact_types=["world"],
                limit=10,
            )
        )["world"]
        rebuild = await memory.rebuild_vector_index(
            bank_id=bank_id,
            batch_size=1,
            request_context=request_context,
        )
        assert rebuild["indexed"] == 2
        rebuilt_hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
        )
        assert banana_id in {hit.id for hit in rebuilt_hits["world"]}

        deletion = await memory.delete_memory_unit(banana_id, request_context=request_context)
        assert deletion["success"] is True
        remaining = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
        )
        assert banana_id not in {hit.id for hit in remaining["world"]}

        document_deletion = await memory.delete_document(
            "doc-apple",
            bank_id,
            request_context=request_context,
        )
        assert document_deletion["document_deleted"] == 1
        assert not (
            await index.search(
                DeterministicEmbeddings().encode(["cherry"])[0],
                namespace="public",
                bank_id=bank_id,
                fact_types=["world"],
                limit=10,
            )
        )["world"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_background_consolidation_replaces_observation_projection(pg0_db_url, tmp_path, monkeypatch):
    index = MilvusVectorIndex(
        uri=str(tmp_path / "observations.db"),
        collection_name=f"hms_observations_{uuid.uuid4().hex}",
    )
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_model="mock",
        embeddings=DeterministicEmbeddings(),
        cross_encoder=DeterministicCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        vector_index=index,
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
        lazy_reranker=True,
    )
    await memory.initialize()
    request_context = RequestContext()
    bank_id = f"milvus-observation-{uuid.uuid4().hex}"
    observation_id = uuid.uuid4()
    calls = 0

    async def fake_consolidation_job(*, memory_engine, bank_id, request_context, operation_id=None):
        nonlocal calls
        calls += 1
        backend = await memory_engine._get_backend()
        async with backend.transaction() as conn:
            if calls == 1:
                await conn.execute(
                    """
                    INSERT INTO public.banks (bank_id, disposition, mission, internal_id)
                    VALUES ($1, $2::jsonb, '', $3)
                    ON CONFLICT (bank_id) DO NOTHING
                    """,
                    bank_id,
                    json.dumps(DEFAULT_DISPOSITION),
                    uuid.uuid4(),
                )
                await conn.execute(
                    """
                    INSERT INTO public.memory_units (
                        id, bank_id, text, fact_type, embedding, proof_count,
                        source_memory_ids, history, tags, event_date, mentioned_at
                    )
                    VALUES ($1, $2, $3, 'observation', $4::vector, 1, '{}', '[]'::jsonb, $5, now(), now())
                    """,
                    observation_id,
                    bank_id,
                    "Apple harvest patterns are stable.",
                    str(DeterministicEmbeddings().encode(["apple"])[0]),
                    ["initial"],
                )
            elif calls == 2:
                await conn.execute(
                    """
                    INSERT INTO public.memory_units (
                        id, bank_id, text, fact_type, embedding, proof_count,
                        source_memory_ids, history, tags, event_date, mentioned_at
                    )
                    VALUES ($4, $5, $1, 'observation', $2::vector, 1, '{}', '[]'::jsonb, $3, now(), now())
                    ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        tags = EXCLUDED.tags,
                        updated_at = now()
                    """,
                    "Banana harvest patterns changed.",
                    str(DeterministicEmbeddings().encode(["banana"])[0]),
                    ["updated"],
                    observation_id,
                    bank_id,
                )
            else:
                await conn.execute("DELETE FROM public.memory_units WHERE id = $1", observation_id)
        return {"memories_processed": 1, "created": int(calls == 1), "updated": int(calls == 2)}

    from hms_api.engine import consolidation

    monkeypatch.setattr(consolidation, "run_consolidation_job", fake_consolidation_job)

    try:
        await memory._handle_consolidation({"bank_id": bank_id})
        created_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["observation"],
            limit=10,
            tags=["initial"],
            tags_match="all_strict",
        )
        assert [hit.id for hit in created_hits["observation"]] == [str(observation_id)]

        cleared = await memory.clear_observations(bank_id, request_context=request_context)
        assert cleared["deleted_count"] == 1
        cleared_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["observation"],
            limit=10,
        )
        assert not cleared_hits["observation"]

        await memory._handle_consolidation({"bank_id": bank_id})
        updated_hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["observation"],
            limit=10,
            tags=["updated"],
            tags_match="all_strict",
        )
        assert [hit.id for hit in updated_hits["observation"]] == [str(observation_id)]
        old_tag_hits = await index.search(
            DeterministicEmbeddings().encode(["apple"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["observation"],
            limit=10,
            tags=["initial"],
            tags_match="all_strict",
        )
        assert not old_tag_hits["observation"]

        await memory._handle_consolidation({"bank_id": bank_id})
        deleted_hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["observation"],
            limit=10,
        )
        assert not deleted_hits["observation"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_vector_index_sync_failure_degrades_only_affected_bank_until_rebuild(pg0_db_url, tmp_path, monkeypatch):
    index = MilvusVectorIndex(
        uri=str(tmp_path / "degraded.db"),
        collection_name=f"hms_degraded_{uuid.uuid4().hex}",
    )
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        embeddings=DeterministicEmbeddings(),
        cross_encoder=DeterministicCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        vector_index=index,
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
        lazy_reranker=True,
    )
    await memory.initialize()
    request_context = RequestContext()
    bank_id = f"milvus-degraded-{uuid.uuid4().hex}"
    unaffected_bank_id = f"milvus-healthy-{uuid.uuid4().hex}"
    original_upsert = index.upsert

    try:
        healthy_ids = await memory.retain_batch_async(
            unaffected_bank_id,
            [{"content": "Apple trees bloom in spring.", "document_id": "healthy-doc"}],
            fact_type_override="world",
            request_context=request_context,
        )
        healthy_id = healthy_ids[0][0]

        async def fail_upsert(records):
            raise RuntimeError("simulated post-commit Milvus failure")

        monkeypatch.setattr(index, "upsert", fail_upsert)
        retained_ids = await memory.retain_batch_async(
            bank_id,
            [{"content": "Banana plants prefer humid weather.", "document_id": "degraded-doc"}],
            fact_type_override="world",
            request_context=request_context,
        )
        retained_id = retained_ids[0][0]

        assert memory.vector_index_degraded is True
        assert memory._is_vector_index_degraded(bank_id) is True
        assert memory._is_vector_index_degraded(unaffected_bank_id) is False
        health = await memory.health_check()
        assert health["status"] == "healthy"
        assert health["vector_index"] == {"provider": "milvus", "degraded": True}

        fallback = await memory.recall_async(
            bank_id,
            "banana",
            fact_type=["world"],
            request_context=request_context,
        )
        assert retained_id in {result.id for result in fallback.results}

        healthy_recall = await memory.recall_async(
            unaffected_bank_id,
            "apple",
            fact_type=["world"],
            request_context=request_context,
        )
        assert healthy_id in {result.id for result in healthy_recall.results}

        monkeypatch.setattr(index, "upsert", original_upsert)
        rebuild = await memory.rebuild_vector_index(
            bank_id=bank_id,
            batch_size=1,
            request_context=request_context,
        )
        assert rebuild["indexed"] == 1
        assert memory.vector_index_degraded is False
        health = await memory.health_check()
        assert health["vector_index"] == {"provider": "milvus", "degraded": False}

        restored_hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world"],
            limit=10,
        )
        assert [hit.id for hit in restored_hits["world"]] == [retained_id]
    finally:
        monkeypatch.setattr(index, "upsert", original_upsert)
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.delete_bank(unaffected_bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_partial_bank_delete_preserves_other_fact_types_in_milvus(pg0_db_url, tmp_path):
    index = MilvusVectorIndex(
        uri=str(tmp_path / "partial-delete.db"),
        collection_name=f"hms_partial_delete_{uuid.uuid4().hex}",
    )
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        embeddings=DeterministicEmbeddings(),
        cross_encoder=DeterministicCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        vector_index=index,
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
        lazy_reranker=True,
    )
    await memory.initialize()
    request_context = RequestContext()
    bank_id = f"milvus-partial-{uuid.uuid4().hex}"

    try:
        world_ids = await memory.retain_batch_async(
            bank_id,
            [{"content": "Apple orchards are productive.", "document_id": "world-doc"}],
            fact_type_override="world",
            request_context=request_context,
        )
        experience_ids = await memory.retain_batch_async(
            bank_id,
            [{"content": "I harvested bananas today.", "document_id": "experience-doc"}],
            fact_type_override="experience",
            request_context=request_context,
        )
        world_id = world_ids[0][0]
        experience_id = experience_ids[0][0]

        result = await memory.delete_bank(
            bank_id,
            fact_type="world",
            delete_bank_profile=False,
            request_context=request_context,
        )
        assert result["memory_units_deleted"] == 1

        hits = await index.search(
            DeterministicEmbeddings().encode(["banana"])[0],
            namespace="public",
            bank_id=bank_id,
            fact_types=["world", "experience"],
            limit=10,
        )
        assert world_id not in {hit.id for hit in hits["world"]}
        assert [hit.id for hit in hits["experience"]] == [experience_id]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()
