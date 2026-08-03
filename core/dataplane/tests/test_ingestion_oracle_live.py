"""Live Oracle 23ai smoke coverage for the Retain ingestion pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import uuid

import pytest
from hms_api import MemoryEngine, RequestContext
from hms_api.config import clear_config_cache
from hms_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hms_api.engine.embeddings import Embeddings
from hms_api.engine.memory_engine import Budget
from hms_api.engine.query_analyzer import DateparserQueryAnalyzer
from hms_api.engine.task_backend import SyncTaskBackend, WorkerTaskBackend

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(
        importlib.util.find_spec("oracledb") is None,
        reason="the oracle optional dependency is required",
    ),
    pytest.mark.skipif(
        not os.getenv("ORACLE_TEST_DSN"),
        reason="ORACLE_TEST_DSN is required for live Oracle tests",
    ),
]


class _DeterministicEmbeddings(Embeddings):
    """Small deterministic vectors with the production schema dimension."""

    model_name = "hms-oracle-live-hash-v1"

    @property
    def provider_name(self) -> str:
        return "oracle-live-test"

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
async def test_oracle_retain_fresh_delta_and_recall(
    oracle_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise durable fresh, no-op, replacement, and retrieval paths."""

    monkeypatch.setenv("HMS_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()
    memory = MemoryEngine(
        db_url=oracle_db_url,
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

    bank_id = f"oracle-retain-live-{uuid.uuid4().hex[:12]}"
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
        clear_config_cache()


@pytest.mark.asyncio
async def test_oracle_async_batch_parent_lifecycle_uses_typed_uuid_predicates(
    oracle_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise parent lookup, cancellation, and retry against Oracle JSON."""

    monkeypatch.setenv("HMS_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()
    memory = MemoryEngine(
        db_url=oracle_db_url,
        memory_llm_provider="none",
        memory_llm_model="none",
        embeddings=_DeterministicEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
        task_backend=WorkerTaskBackend(),
        skip_llm_verification=True,
    )
    await memory.initialize()

    bank_id = f"oracle-parent-live-{uuid.uuid4().hex[:12]}"
    request_context = RequestContext()

    try:
        submitted = await memory.submit_async_retain(
            bank_id=bank_id,
            contents=[{"content": "Alice owns the Atlas service.", "document_id": "profile"}],
            request_context=request_context,
        )
        parent_operation_id = submitted["operation_id"]

        pending = await memory.get_operation_status(
            bank_id=bank_id,
            operation_id=parent_operation_id,
            request_context=request_context,
        )
        assert pending["status"] == "pending"
        assert len(pending["child_operations"]) == 1
        assert pending["child_operations"][0]["status"] == "pending"

        await memory.cancel_operation(
            bank_id=bank_id,
            operation_id=parent_operation_id.upper(),
            request_context=request_context,
        )
        cancelled = await memory.get_operation_status(
            bank_id=bank_id,
            operation_id=parent_operation_id,
            request_context=request_context,
        )
        assert cancelled["status"] == "cancelled"
        assert cancelled["child_operations"][0]["status"] == "cancelled"

        await memory.retry_operation(
            bank_id=bank_id,
            operation_id=parent_operation_id.upper(),
            request_context=request_context,
        )
        retried = await memory.get_operation_status(
            bank_id=bank_id,
            operation_id=parent_operation_id,
            request_context=request_context,
        )
        assert retried["status"] == "pending"
        assert retried["child_operations"][0]["status"] == "pending"
    finally:
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        finally:
            await memory.close()
            clear_config_cache()
