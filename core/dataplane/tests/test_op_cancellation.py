"""Tests for operation cancellation when a bank is deleted.

Covers:
- CASCADE DELETE: deleting a bank removes async_operations and webhooks rows
- _check_op_alive: returns True when op exists, False when deleted
- _mark_operation_completed / _mark_operation_failed: graceful no-op when row is gone
- Consolidation checkpoint: stops early after a batch commit if op was deleted
- Retain checkpoint: stops between sub-batches if op was deleted
"""

import hashlib
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from hms_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hms_api.engine.embeddings import Embeddings
from hms_api.engine.memory_engine import MemoryEngine, _RetainOperationCancelled

pytestmark = pytest.mark.xdist_group("op_cancellation_tests")

_BANK_PREFIX = "test-op-cancel"


class _DeterministicEmbeddings(Embeddings):
    """Provide stable vectors without loading a local model."""

    model_name = "hms-operation-cancellation-test-hash-v1"

    @property
    def provider_name(self) -> str:
        return "operation-cancellation-test"

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


@pytest.fixture(scope="session")
def embeddings() -> Embeddings:
    """Use deterministic embeddings that need no optional ML dependencies."""
    return _DeterministicEmbeddings()


@pytest.fixture(scope="session")
def cross_encoder() -> RRFPassthroughCrossEncoder:
    """Use the dependency-free reciprocal-rank fusion reranker."""
    return RRFPassthroughCrossEncoder()


@pytest_asyncio.fixture
async def pool(pg0_db_url):
    import asyncpg
    from hms_api.pg0 import resolve_database_url

    resolved_url = await resolve_database_url(pg0_db_url)
    p = await asyncpg.create_pool(resolved_url, min_size=1, max_size=5, command_timeout=30)
    yield p
    await p.close()


@pytest_asyncio.fixture(autouse=True)
async def cleanup(pool):
    """Remove test rows before and after each test."""
    await pool.execute(f"DELETE FROM banks WHERE bank_id LIKE '{_BANK_PREFIX}%'")
    yield
    await pool.execute(f"DELETE FROM banks WHERE bank_id LIKE '{_BANK_PREFIX}%'")


async def _insert_bank(pool, bank_id: str):
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _insert_op(
    pool,
    bank_id: str,
    op_id: uuid.UUID | None = None,
    *,
    operation_type: str = "consolidation",
) -> uuid.UUID:
    op_id = op_id or uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
        VALUES ($1, $2, $3, 'processing')
        """,
        op_id,
        bank_id,
        operation_type,
    )
    return op_id


# ---------------------------------------------------------------------------
# CASCADE DELETE tests
# ---------------------------------------------------------------------------


class TestCascadeDeleteOnBankDeletion:
    @pytest.mark.asyncio
    async def test_bank_deletion_cascades_to_async_operations(self, pool):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        op_id = await _insert_op(pool, bank_id)

        # Verify op exists
        row = await pool.fetchrow("SELECT operation_id FROM async_operations WHERE operation_id = $1", op_id)
        assert row is not None

        # Delete the bank — should cascade to async_operations
        await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)

        row = await pool.fetchrow("SELECT operation_id FROM async_operations WHERE operation_id = $1", op_id)
        assert row is None, "async_operations row should be deleted by CASCADE"

    @pytest.mark.asyncio
    async def test_bank_deletion_cascades_to_webhooks(self, pool):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        webhook_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO webhooks (id, bank_id, url, event_types)
            VALUES ($1, $2, 'https://example.com/hook', '{}')
            """,
            webhook_id,
            bank_id,
        )

        row = await pool.fetchrow("SELECT id FROM webhooks WHERE id = $1", webhook_id)
        assert row is not None

        await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)

        row = await pool.fetchrow("SELECT id FROM webhooks WHERE id = $1", webhook_id)
        assert row is None, "webhooks row should be deleted by CASCADE"


# ---------------------------------------------------------------------------
# _check_op_alive tests
# ---------------------------------------------------------------------------


class TestCheckOpAlive:
    @pytest.mark.asyncio
    async def test_returns_true_when_op_exists(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )

        assert await memory._check_op_alive(str(op_id)) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_op_deleted(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )
            await conn.execute("DELETE FROM async_operations WHERE operation_id = $1", op_id)

        assert await memory._check_op_alive(str(op_id)) is False

    @pytest.mark.asyncio
    async def test_returns_false_after_bank_cascade_delete(self, memory: MemoryEngine, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        op_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                VALUES ($1, $2, 'consolidation', 'processing')
                """,
                op_id,
                bank_id,
            )

        # Delete the bank — cascades to the op row
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)

        assert await memory._check_op_alive(str(op_id)) is False


# ---------------------------------------------------------------------------
# Batch parent/child cancellation
# ---------------------------------------------------------------------------


class TestBatchCancellation:
    @pytest.mark.asyncio
    async def test_parent_cancellation_atomically_cancels_active_children(
        self,
        memory: MemoryEngine,
        pool,
        request_context,
    ):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        parent_id = uuid.uuid4()
        pending_child_id = uuid.uuid4()
        processing_child_id = uuid.uuid4()

        await pool.execute(
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
            await pool.execute(
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
            operation_id=str(parent_id),
            request_context=request_context,
        )

        rows = await pool.fetch(
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

        # A late worker completion must lose the terminal-state CAS.
        await memory._mark_operation_completed(str(processing_child_id))
        assert (
            await pool.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                processing_child_id,
            )
            == "cancelled"
        )
        assert (
            await pool.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                parent_id,
            )
            == "cancelled"
        )

    @pytest.mark.asyncio
    async def test_direct_child_cancellation_resolves_parent(self, memory: MemoryEngine, pool, request_context):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        parent_id = uuid.uuid4()
        completed_child_id = uuid.uuid4()
        pending_child_id = uuid.uuid4()

        await pool.execute(
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
            (completed_child_id, "completed", 1),
            (pending_child_id, "pending", 2),
        ):
            await pool.execute(
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
            operation_id=str(pending_child_id),
            request_context=request_context,
        )

        assert (
            await pool.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                pending_child_id,
            )
            == "cancelled"
        )
        assert (
            await pool.fetchval(
                "SELECT status FROM async_operations WHERE operation_id = $1",
                parent_id,
            )
            == "cancelled"
        )


# ---------------------------------------------------------------------------
# _mark_operation_completed / _mark_operation_failed graceful no-op
# ---------------------------------------------------------------------------


class TestMarkOperationGracefulOnMissingRow:
    @pytest.mark.asyncio
    async def test_mark_completed_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        # Row never existed — should log and return cleanly
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_completed(missing_id)  # no exception

    @pytest.mark.asyncio
    async def test_mark_completed_does_not_overwrite_cancelled(self, memory: MemoryEngine, pool):
        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await _insert_bank(pool, bank_id)
        operation_id = await _insert_op(pool, bank_id, operation_type="retain")
        await pool.execute(
            "UPDATE async_operations SET status = 'cancelled' WHERE operation_id = $1",
            operation_id,
        )

        await memory._mark_operation_completed(str(operation_id))

        status = await pool.fetchval(
            "SELECT status FROM async_operations WHERE operation_id = $1",
            operation_id,
        )
        assert status == "cancelled"

    @pytest.mark.asyncio
    async def test_mark_failed_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_failed(missing_id, "some error", "traceback here")  # no exception

    @pytest.mark.asyncio
    async def test_mark_completed_and_fire_webhook_does_not_raise_when_row_missing(self, memory: MemoryEngine):
        missing_id = str(uuid.uuid4())
        await memory._mark_operation_completed_and_fire_webhook(
            operation_id=missing_id,
            bank_id="nonexistent-bank",
            status="completed",
            result=None,
        )  # no exception


# ---------------------------------------------------------------------------
# Consolidation checkpoint
# ---------------------------------------------------------------------------


class TestConsolidationCheckpoint:
    @pytest.mark.asyncio
    async def test_consolidation_stops_early_when_op_cancelled(self, memory: MemoryEngine, request_context):
        """Consolidation returns 'cancelled' status after the first batch if _check_op_alive is False."""
        from hms_api.config import _get_raw_config
        from hms_api.engine.consolidation.consolidator import run_consolidation_job

        config = _get_raw_config()
        original = config.enable_observations
        config.enable_observations = True

        try:
            bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
            await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

            # Insert a few unconsolidated memories directly so we control the batch without LLM
            async with memory._pool.acquire() as conn:
                for i in range(3):
                    await conn.execute(
                        """
                        INSERT INTO memory_units
                            (id, bank_id, text, fact_type, created_at, updated_at)
                        VALUES (gen_random_uuid(), $1, $2, 'experience', NOW(), NOW())
                        """,
                        bank_id,
                        f"Test memory {i} for cancellation test",
                    )

            op_id = str(uuid.uuid4())
            call_count = 0

            async def _fake_check(operation_id: str) -> bool:
                nonlocal call_count
                call_count += 1
                # Return False on the very first checkpoint call
                return False

            with patch.object(memory, "_check_op_alive", side_effect=_fake_check):
                result = await run_consolidation_job(
                    memory_engine=memory,
                    bank_id=bank_id,
                    request_context=request_context,
                    operation_id=op_id,
                )

            assert result["status"] == "cancelled"
            assert call_count >= 1
        finally:
            config.enable_observations = original


# ---------------------------------------------------------------------------
# Retain checkpoint
# ---------------------------------------------------------------------------


class TestRetainCheckpoint:
    @pytest.mark.asyncio
    async def test_retain_stops_between_sub_batches_when_cancelled(self, memory: MemoryEngine, request_context, pool):
        """retain_batch_async raises instead of reporting partial success after cancellation."""
        from hms_api.config import _get_raw_config

        bank_id = f"{_BANK_PREFIX}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Force sub-batch splitting by temporarily lowering the token threshold
        config = _get_raw_config()
        original_tokens = config.retain_batch_tokens
        # Set threshold very low so each item becomes its own sub-batch
        config.retain_batch_tokens = 1

        try:
            # Retain persists its exact recovery checkpoint in the tracked
            # operation's core transaction.  Keep this cancellation fixture
            # faithful to the worker path by creating that operation first.
            op_id = str(await _insert_op(pool, bank_id, operation_type="retain"))
            check_calls = 0

            async def _fake_check(operation_id: str) -> bool:
                nonlocal check_calls
                check_calls += 1
                # Cancel after the first sub-batch completes
                return check_calls <= 1

            contents = [
                {
                    "content": f"Memory item {i} about something interesting.",
                    "document_id": f"document-{i}",
                }
                for i in range(4)
            ]

            with patch.object(memory, "_check_op_alive", side_effect=_fake_check):
                with pytest.raises(_RetainOperationCancelled):
                    await memory.retain_batch_async(
                        bank_id=bank_id,
                        contents=contents,
                        request_context=request_context,
                        operation_id=op_id,
                    )

            assert check_calls >= 1
        finally:
            config.retain_batch_tokens = original_tokens
