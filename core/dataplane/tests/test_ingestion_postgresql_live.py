"""Live PostgreSQL smoke coverage for the Retain ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from hms_api import MemoryEngine, RequestContext
from hms_api.config import clear_config_cache
from hms_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hms_api.engine.embeddings import Embeddings
from hms_api.engine.ingestion import RetainOperationInactiveError
from hms_api.engine.ingestion.persistence.postgres import PostgresDocumentOwnership
from hms_api.engine.memory_engine import Budget, _RetainOperationCancelled
from hms_api.engine.query_analyzer import DateparserQueryAnalyzer
from hms_api.engine.task_backend import SyncTaskBackend
from hms_api.worker.poller import WorkerPoller


class _DeterministicEmbeddings(Embeddings):
    """Small deterministic vectors with the production schema dimension."""

    model_name = "hms-postgresql-live-hash-v1"

    @property
    def provider_name(self) -> str:
        return "postgresql-live-test"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(self.dimension)])
        return vectors


@pytest.mark.asyncio
async def test_postgresql_retain_fresh_delta_and_recall(pg0_db_url: str) -> None:
    """Exercise durable fresh, no-op, replacement, and retrieval paths."""

    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-live-{uuid.uuid4().hex[:12]}"
    document_id = "profile"
    request_context = RequestContext()
    original = "Alice maintains the Atlas search service."
    replacement = "Alice maintains the Atlas search service. Bob owns incident response."

    try:
        first_ids = await memory.retain_async(
            bank_id=bank_id,
            content=original,
            document_id=document_id,
            request_context=request_context,
        )
        assert first_ids

        unchanged_ids = await memory.retain_async(
            bank_id=bank_id,
            content=original,
            document_id=document_id,
            request_context=request_context,
        )
        assert unchanged_ids == []

        replacement_ids = await memory.retain_async(
            bank_id=bank_id,
            content=replacement,
            document_id=document_id,
            request_context=request_context,
        )
        assert replacement_ids

        document = await memory.get_document(
            document_id,
            bank_id,
            request_context=request_context,
        )
        assert document is not None
        assert document["original_text"] == replacement

        recalled = await memory.recall_async(
            bank_id=bank_id,
            query="Who owns incident response?",
            budget=Budget.LOW,
            max_tokens=512,
            request_context=request_context,
        )
        assert recalled.results
        assert any("incident response" in result.text for result in recalled.results)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_postgresql_retain_replaces_an_existing_unhashed_document(pg0_db_url: str) -> None:
    """Upgrade an unhashed row through the same locked full-replacement path."""

    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-unhashed-{uuid.uuid4().hex[:12]}"
    document_id = "upgrade-document"
    request_context = RequestContext()
    original = "Original-only-token belongs to the old document."
    replacement = "Replacement-only-token belongs to the upgraded document."

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=original,
            document_id=document_id,
            request_context=request_context,
        )
        async with memory._pool.acquire() as connection:
            await connection.execute(
                "UPDATE documents SET content_hash = NULL WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )

        replacement_ids = await memory.retain_async(
            bank_id=bank_id,
            content=replacement,
            document_id=document_id,
            request_context=request_context,
        )

        assert replacement_ids
        async with memory._pool.acquire() as connection:
            document = await connection.fetchrow(
                """
                SELECT original_text, content_hash
                FROM documents
                WHERE id = $1 AND bank_id = $2
                """,
                document_id,
                bank_id,
            )
            facts = await connection.fetch(
                """
                SELECT text
                FROM memory_units
                WHERE document_id = $1 AND bank_id = $2
                """,
                document_id,
                bank_id,
            )
        assert document is not None
        assert document["original_text"] == replacement
        assert document["content_hash"] == hashlib.sha256(replacement.encode()).hexdigest()
        assert facts
        assert all("Original-only-token" not in row["text"] for row in facts)

        # Once a competing writer publishes a hash, an older unhashed snapshot
        # can no longer acquire replacement ownership.
        ownership = PostgresDocumentOwnership()
        async with memory._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE documents SET content_hash = NULL WHERE id = $1 AND bank_id = $2",
                    document_id,
                    bank_id,
                )
                assert await ownership.validate_unhashed_window(
                    connection,
                    bank_id=bank_id,
                    document_id=document_id,
                )
                await connection.execute(
                    "UPDATE documents SET content_hash = $3 WHERE id = $1 AND bank_id = $2",
                    document_id,
                    bank_id,
                    "new-owner-hash",
                )
        async with memory._pool.acquire() as connection:
            async with connection.transaction():
                assert not await ownership.validate_unhashed_window(
                    connection,
                    bank_id=bank_id,
                    document_id=document_id,
                )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_postgresql_token_batching_preserves_document_groups(
    pg0_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep repeated IDs together while restoring the caller's result order."""

    monkeypatch.setenv("HMS_API_RETAIN_BATCH_TOKENS", "1")
    clear_config_cache()
    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-groups-{uuid.uuid4().hex[:12]}"
    request_context = RequestContext()

    try:
        results = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alpha first.", "document_id": "doc-alpha"},
                {"content": "Beta only.", "document_id": "doc-beta"},
                {"content": "Alpha second.", "document_id": "doc-alpha"},
            ],
            request_context=request_context,
        )

        assert len(results) == 3
        assert all(results)
        alpha = await memory.get_document("doc-alpha", bank_id, request_context=request_context)
        beta = await memory.get_document("doc-beta", bank_id, request_context=request_context)
        assert alpha is not None
        assert beta is not None
        assert "Alpha first." in alpha["original_text"]
        assert "Alpha second." in alpha["original_text"]
        assert "Beta only." not in alpha["original_text"]
        assert "Beta only." in beta["original_text"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()
        clear_config_cache()


@pytest.mark.asyncio
async def test_postgresql_cancellation_is_not_reported_as_completion(
    pg0_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate a between-document cancellation and preserve terminal state."""

    monkeypatch.setenv("HMS_API_RETAIN_BATCH_TOKENS", "1")
    clear_config_cache()
    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-cancel-{uuid.uuid4().hex[:12]}"
    operation_id = uuid.uuid4()
    request_context = RequestContext()
    outbox_callback = AsyncMock()

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        async with memory._pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'retain', 'processing')
                """,
                operation_id,
                bank_id,
            )

        monkeypatch.setattr(memory, "_check_op_alive", AsyncMock(side_effect=[True, False]))
        with pytest.raises(_RetainOperationCancelled, match="cancelled between logical document batches"):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {"content": "Committed before cancellation.", "document_id": "doc-before"},
                    {"content": "Must not be committed.", "document_id": "doc-after"},
                ],
                request_context=request_context,
                operation_id=str(operation_id),
                outbox_callback=outbox_callback,
            )
        outbox_callback.assert_not_awaited()

        async with memory._pool.acquire() as connection:
            await connection.execute(
                "UPDATE async_operations SET status = 'cancelled' WHERE operation_id = $1",
                operation_id,
            )
        await memory._mark_operation_completed(str(operation_id))
        async with memory._pool.acquire() as connection:
            status = await connection.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                operation_id,
            )
        assert status == "cancelled"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()
        clear_config_cache()


@pytest.mark.asyncio
async def test_postgresql_parent_cancellation_is_terminal_for_children(pg0_db_url: str) -> None:
    """Cancel active child rows atomically and reject late worker completion."""

    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-parent-cancel-{uuid.uuid4().hex[:12]}"
    parent_id = uuid.uuid4()
    pending_child_id = uuid.uuid4()
    processing_child_id = uuid.uuid4()
    request_context = RequestContext()

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        async with memory._pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO async_operations
                    (operation_id, bank_id, operation_type, status, result_metadata)
                VALUES ($1, $2, 'batch_retain', 'pending', $3::jsonb)
                """,
                parent_id,
                bank_id,
                json.dumps({"is_parent": True, "num_sub_batches": 2, "items_count": 2}),
            )
            for child_id, status, index in (
                (pending_child_id, "pending", 1),
                (processing_child_id, "processing", 2),
            ):
                await connection.execute(
                    """
                    INSERT INTO async_operations
                        (operation_id, bank_id, operation_type, status, result_metadata)
                    VALUES ($1, $2, 'retain', $3, $4::jsonb)
                    """,
                    child_id,
                    bank_id,
                    status,
                    json.dumps(
                        {
                            "parent_operation_id": str(parent_id),
                            "sub_batch_index": index,
                            "total_sub_batches": 2,
                        }
                    ),
                )

        await memory.cancel_operation(
            bank_id=bank_id,
            operation_id=str(parent_id).upper(),
            request_context=request_context,
        )

        poller = WorkerPoller(
            backend=memory._backend,
            worker_id="postgresql-retain-live",
            executor=memory.execute_task,
        )
        await poller._mark_completed(str(processing_child_id), None)
        await memory._mark_operation_completed(str(pending_child_id))

        async with memory._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT operation_id, status
                FROM async_operations
                WHERE operation_id = ANY($1::uuid[])
                """,
                [parent_id, pending_child_id, processing_child_id],
            )
        assert {row["operation_id"]: row["status"] for row in rows} == {
            parent_id: "cancelled",
            pending_child_id: "cancelled",
            processing_child_id: "cancelled",
        }
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_postgresql_accepted_cancellation_blocks_core_retain_and_outbox(pg0_db_url: str) -> None:
    """Reject a stale processing child inside its next core write transaction."""

    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-write-fence-{uuid.uuid4().hex[:12]}"
    document_id = "must-not-exist"
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    request_context = RequestContext()
    outbox_callback = AsyncMock()

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        async with memory._pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO async_operations
                    (operation_id, bank_id, operation_type, status, result_metadata)
                VALUES ($1, $2, 'batch_retain', 'pending', $3::jsonb)
                """,
                parent_id,
                bank_id,
                json.dumps({"is_parent": True, "num_sub_batches": 1, "items_count": 1}),
            )
            await connection.execute(
                """
                INSERT INTO async_operations
                    (operation_id, bank_id, operation_type, status, result_metadata)
                VALUES ($1, $2, 'retain', 'processing', $3::jsonb)
                """,
                child_id,
                bank_id,
                json.dumps(
                    {
                        "parent_operation_id": str(parent_id),
                        "sub_batch_index": 1,
                        "total_sub_batches": 1,
                    }
                ),
            )

        await memory.cancel_operation(
            bank_id=bank_id,
            operation_id=str(parent_id),
            request_context=request_context,
        )

        with pytest.raises(RetainOperationInactiveError, match="no longer active"):
            await memory._retain_batch_async_internal(
                bank_id=bank_id,
                contents=[{"content": "This content must never commit.", "document_id": document_id}],
                request_context=request_context,
                operation_id=str(child_id),
                outbox_callback=outbox_callback,
            )

        outbox_callback.assert_not_awaited()
        async with memory._pool.acquire() as connection:
            child_status = await connection.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                child_id,
            )
            counts = await connection.fetchrow(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents WHERE id = $1 AND bank_id = $2) AS documents,
                    (SELECT COUNT(*) FROM chunks WHERE document_id = $1 AND bank_id = $2) AS chunks,
                    (SELECT COUNT(*) FROM memory_units WHERE document_id = $1 AND bank_id = $2) AS memories
                """,
                document_id,
                bank_id,
            )
        assert child_status == "cancelled"
        assert dict(counts) == {"documents": 0, "chunks": 0, "memories": 0}
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()


@pytest.mark.asyncio
async def test_postgresql_batch_retry_reopens_only_retryable_work(pg0_db_url: str) -> None:
    """Retry a terminal batch or child without reviving completed work."""

    memory = MemoryEngine(
        db_url=pg0_db_url,
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

    bank_id = f"postgresql-retain-batch-retry-{uuid.uuid4().hex[:12]}"
    parent_id = uuid.uuid4()
    completed_child_id = uuid.uuid4()
    failed_child_id = uuid.uuid4()
    cancelled_child_id = uuid.uuid4()
    request_context = RequestContext()

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        async with memory._pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO async_operations
                    (operation_id, bank_id, operation_type, status, result_metadata)
                VALUES ($1, $2, 'batch_retain', 'failed', $3::jsonb)
                """,
                parent_id,
                bank_id,
                json.dumps({"is_parent": True, "num_sub_batches": 3, "items_count": 3}),
            )
            for child_id, status, index in (
                (completed_child_id, "completed", 1),
                (failed_child_id, "failed", 2),
                (cancelled_child_id, "cancelled", 3),
            ):
                await connection.execute(
                    """
                    INSERT INTO async_operations
                        (operation_id, bank_id, operation_type, status, result_metadata, task_payload)
                    VALUES ($1, $2, 'retain', $3, $4::jsonb, $5::jsonb)
                    """,
                    child_id,
                    bank_id,
                    status,
                    json.dumps(
                        {
                            "parent_operation_id": str(parent_id),
                            "sub_batch_index": index,
                            "total_sub_batches": 3,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "batch_retain",
                            "operation_id": str(child_id),
                            "bank_id": bank_id,
                            "contents": [],
                        }
                    ),
                )

        await memory.retry_operation(
            bank_id=bank_id,
            operation_id=str(parent_id).upper(),
            request_context=request_context,
        )

        async with memory._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT operation_id, status
                FROM async_operations
                WHERE operation_id = ANY($1::uuid[])
                """,
                [parent_id, completed_child_id, failed_child_id, cancelled_child_id],
            )
        assert {row["operation_id"]: row["status"] for row in rows} == {
            parent_id: "pending",
            completed_child_id: "completed",
            failed_child_id: "pending",
            cancelled_child_id: "pending",
        }

        # A child-level retry safely reopens a failed/cancelled parent, but
        # still preserves completed siblings.
        async with memory._pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE async_operations
                SET status = CASE
                    WHEN operation_id = $1 THEN 'cancelled'
                    WHEN operation_id = $2 THEN 'cancelled'
                    WHEN operation_id = $3 THEN 'cancelled'
                    WHEN operation_id = $4 THEN 'completed'
                    ELSE status
                END
                WHERE operation_id = ANY($5::uuid[])
                """,
                parent_id,
                failed_child_id,
                cancelled_child_id,
                completed_child_id,
                [parent_id, failed_child_id, cancelled_child_id, completed_child_id],
            )

        await memory.retry_operation(
            bank_id=bank_id,
            operation_id=str(failed_child_id),
            request_context=request_context,
        )

        async with memory._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT operation_id, status
                FROM async_operations
                WHERE operation_id = ANY($1::uuid[])
                """,
                [parent_id, completed_child_id, failed_child_id, cancelled_child_id],
            )
        assert {row["operation_id"]: row["status"] for row in rows} == {
            parent_id: "pending",
            completed_child_id: "completed",
            failed_child_id: "pending",
            cancelled_child_id: "cancelled",
        }
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.close()
