"""Failure-path tests for external semantic retrieval orchestration."""

from __future__ import annotations

import asyncio

import pytest

from hms_api.engine.search.implementations import SemanticBM25Retrieval


@pytest.mark.asyncio
async def test_database_search_failure_cancels_in_flight_external_search():
    search_started = asyncio.Event()
    search_cancelled = asyncio.Event()

    class FailingConnection:
        backend_type = "postgresql"

        async def fetch(self, query, *args):
            await search_started.wait()
            raise RuntimeError("simulated database FTS failure")

    class WaitingVectorIndex:
        async def search(self, *args, **kwargs):
            search_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                search_cancelled.set()
                raise

    strategy = SemanticBM25Retrieval(alias_expansion_enabled=False)
    with pytest.raises(RuntimeError, match="database FTS failure"):
        await strategy._retrieve_with_external_vector_index(
            FailingConnection(),
            WaitingVectorIndex(),
            [1.0, 0.0],
            query_embedding_str="[1.0, 0.0]",
            rewritten_query="apple",
            tokens=["apple"],
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
            tags=None,
            tags_match="any",
            tag_groups=None,
            created_after=None,
            created_before=None,
        )

    assert search_cancelled.is_set()
