"""Tests for projection-manifest based reprocessing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from hms_api.api import create_app
from hms_api.engine.memory_engine import MemoryEngine, _build_projection_selector_where
from hms_api.models import RequestContext


class _FakeConn:
    backend_type = "postgresql"

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.rows


class _FakePool:
    _wraps_backend = False
    backend_type = "postgresql"

    def __init__(self, conn):
        self.conn = conn

    async def acquire(self):
        return self.conn

    async def release(self, conn):
        assert conn is self.conn

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 1


def _engine_with_rows(rows):
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    conn = _FakeConn(rows)
    engine._get_backend = AsyncMock(return_value=_FakePool(conn))
    return engine, conn


@pytest.mark.asyncio
async def test_reprocess_by_projection_dry_run_counts_documents_chunks_and_skips_documentless_units():
    rows = [
        {"id": "unit-1", "document_id": "doc-a", "chunk_id": "chunk-a"},
        {"id": "unit-2", "document_id": "doc-a", "chunk_id": "chunk-b"},
        {"id": "unit-3", "document_id": "doc-b", "chunk_id": "chunk-c"},
        {"id": "unit-4", "document_id": None, "chunk_id": None},
    ]
    engine, conn = _engine_with_rows(rows)

    result = await MemoryEngine.reprocess_by_projection(
        engine,
        bank_id="bank-1",
        selector={"temporal": {"grade": "unresolved"}},
        dry_run=True,
        request_context=RequestContext(),
    )

    assert result["unit_count"] == 4
    assert result["document_count"] == 2
    assert result["chunk_count"] == 3
    assert result["skipped_unit_count"] == 1
    assert result["document_ids"] == ["doc-a", "doc-b"]
    assert result["chunk_ids"] == ["chunk-a", "chunk-b", "chunk-c"]
    assert result["operation_ids"] == []

    query, args = conn.queries[0]
    assert "projection #>> '{temporal,grade}' = $2" in query
    assert args == ("bank-1", "unresolved")


@pytest.mark.asyncio
async def test_reprocess_by_projection_execute_submits_once_per_distinct_document():
    rows = [
        {"id": "unit-1", "document_id": "doc-a", "chunk_id": "chunk-a"},
        {"id": "unit-2", "document_id": "doc-a", "chunk_id": "chunk-b"},
        {"id": "unit-3", "document_id": "doc-b", "chunk_id": "chunk-c"},
    ]
    engine, _ = _engine_with_rows(rows)
    engine.reprocess_document = AsyncMock(
        side_effect=[
            {"operation_id": "op-doc-a", "items_count": 1},
            {"operation_id": "op-doc-b", "items_count": 1},
        ]
    )

    result = await MemoryEngine.reprocess_by_projection(
        engine,
        bank_id="bank-1",
        selector={"extraction.v": "legacy"},
        dry_run=False,
        request_context=RequestContext(),
    )

    assert result["submitted_count"] == 2
    assert result["operation_ids"] == ["op-doc-a", "op-doc-b"]
    assert [op["document_id"] for op in result["operations"]] == ["doc-a", "doc-b"]
    assert engine.reprocess_document.await_count == 2


def test_projection_selector_rejects_unsafe_keys():
    with pytest.raises(ValueError, match="Invalid projection selector key"):
        _build_projection_selector_where(
            {"extraction.v;DROP": "legacy"},
            backend_type="postgresql",
            first_param=2,
        )


def test_projection_selector_builds_oracle_json_value_predicate():
    where, args = _build_projection_selector_where(
        {"embedding.ok": False},
        backend_type="oracle",
        first_param=3,
    )

    assert "JSON_VALUE(projection, '$.embedding.ok' RETURNING VARCHAR2(4000)) = $3" == where
    assert args == ["false"]


@pytest.mark.asyncio
async def test_http_reprocess_by_projection_endpoint():
    class FakeMemory:
        audit_logger = None

        def __init__(self):
            self.reprocess_by_projection = AsyncMock(
                return_value={
                    "bank_id": "bank-1",
                    "selector": {"extraction.v": "legacy"},
                    "dry_run": True,
                    "unit_count": 2,
                    "document_count": 1,
                    "chunk_count": 1,
                    "skipped_unit_count": 0,
                    "unit_ids": ["unit-1", "unit-2"],
                    "document_ids": ["doc-a"],
                    "chunk_ids": ["chunk-a"],
                    "operation_ids": [],
                    "operations": [],
                    "submitted_count": 0,
                }
            )

    memory = FakeMemory()
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/bank-1/projections/reprocess",
            json={"selector": {"extraction.v": "legacy"}, "dry_run": True},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dry_run"] is True
    assert data["unit_count"] == 2
    memory.reprocess_by_projection.assert_awaited_once()
