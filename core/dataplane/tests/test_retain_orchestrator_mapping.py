"""Unit tests for retain orchestrator mapping and embeddings length guarantee.

Regression coverage for issue #1037: a silent length mismatch between the
extracted facts and the generated embeddings caused
`_map_results_to_contents` to raise IndexError during batch_retain.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hms_api.engine.retain import embedding_utils, orchestrator
from hms_api.engine.retain.orchestrator import (
    RetainPublicationAborted,
    _consume_streaming_batches,
    _map_results_to_contents,
)
from hms_api.engine.retain.types import ProcessedFact, RetainContent


def _make_processed_fact(content_index: int, text: str = "fact") -> ProcessedFact:
    return ProcessedFact(
        fact_text=text,
        fact_type="world",
        embedding=[0.0, 0.0, 0.0],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=datetime(2026, 1, 1),
        context="",
        metadata={},
        content_index=content_index,
    )


def _make_content(text: str = "x") -> RetainContent:
    return RetainContent(content=text)


class _FakeTransaction:
    def __init__(self, connection: "_FakeConnection") -> None:
        self.connection = connection

    async def __aenter__(self):
        assert not self.connection.in_transaction
        self.connection.in_transaction = True
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        self.connection.in_transaction = False
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.in_transaction = False
        self.fetchrow = AsyncMock(return_value=None)
        self.fetchval = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock()

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)


def _install_streaming_mocks(monkeypatch: pytest.MonkeyPatch, connection: _FakeConnection):
    @asynccontextmanager
    async def acquire(_pool):
        yield connection

    extract = AsyncMock(return_value=([], [], [], orchestrator.TokenUsage()))
    track_document = AsyncMock()
    update_document = AsyncMock()
    monkeypatch.setattr(orchestrator, "acquire_with_retry", acquire)
    monkeypatch.setattr(orchestrator, "ensure_bank_embedding_fingerprint", AsyncMock())
    monkeypatch.setattr(orchestrator, "_extract_and_embed", extract)
    monkeypatch.setattr(orchestrator.fact_storage, "handle_document_tracking", track_document)
    monkeypatch.setattr(orchestrator.fact_storage, "upsert_document_metadata", update_document)
    return extract, track_document, update_document


async def _run_streaming_publication(
    *,
    callback: AsyncMock,
    contents_dicts: list[dict],
    chunks: list[str],
    operation_id: str | None = None,
):
    return await orchestrator._streaming_retain_batch(
        pool=SimpleNamespace(ops=object()),
        embeddings_model=object(),
        llm_config=None,
        entity_resolver=MagicMock(),
        format_date_fn=None,
        bank_id="bank",
        contents_dicts=contents_dicts,
        contents=[_make_content("")],
        config=SimpleNamespace(
            embedding_fingerprint_policy="strict",
            embedding_fingerprint_legacy_attestation=None,
        ),
        document_id="document",
        is_first_batch=True,
        fact_type_override=None,
        document_tags=None,
        agent_name="test-agent",
        log_buffer=[],
        start_time=datetime.now(UTC).timestamp(),
        all_pre_chunks=list(chunks),
        chunk_to_content=[0 for _ in chunks],
        chunk_batch_size=1,
        operation_id=operation_id,
        outbox_callback=callback,
    )


class TestMapResultsToContents:
    def test_groups_unit_ids_by_content_index(self):
        contents = [_make_content("a"), _make_content("b"), _make_content("c")]
        processed = [
            _make_processed_fact(0, "a1"),
            _make_processed_fact(0, "a2"),
            _make_processed_fact(2, "c1"),
        ]
        unit_ids = ["u-a1", "u-a2", "u-c1"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u-a1", "u-a2"], [], ["u-c1"]]

    def test_handles_out_of_range_content_index(self):
        contents = [_make_content("a"), _make_content("b")]
        processed = [
            _make_processed_fact(-1, "f1"),
            _make_processed_fact(99, "f2"),
        ]
        unit_ids = ["u1", "u2"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u1"], ["u2"]]

    def test_empty_inputs(self):
        assert _map_results_to_contents([], [], []) == []

    def test_length_mismatch_raises(self):
        # Regression for #1037: previously the function silently overran unit_ids.
        contents = [_make_content("a")]
        processed = [_make_processed_fact(0), _make_processed_fact(0)]
        unit_ids = ["u1"]  # one fewer than processed_facts

        with pytest.raises(ValueError, match="length mismatch"):
            _map_results_to_contents(contents, processed, unit_ids)

    def test_unit_ids_assigned_by_processed_fact_position(self):
        # Even if processed_facts are interleaved across contents, each unit_id
        # must follow its corresponding processed_fact (positional alignment).
        contents = [_make_content("a"), _make_content("b")]
        processed = [
            _make_processed_fact(1, "b1"),
            _make_processed_fact(0, "a1"),
            _make_processed_fact(1, "b2"),
        ]
        unit_ids = ["u-b1", "u-a1", "u-b2"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u-a1"], ["u-b1", "u-b2"]]


class TestEmbeddingsBatchLengthGuarantee:
    def test_raises_when_backend_returns_fewer_embeddings(self):
        # Regression for #1037: backends that silently truncate must not pass
        # through — `zip(extracted_facts, embeddings)` would otherwise drop
        # facts and break unit_id alignment downstream.
        backend = MagicMock()
        backend.encode.return_value = [[0.1, 0.2]]  # only 1 vector for 3 inputs

        with pytest.raises(RuntimeError, match="returned 1 vectors for 3 input texts"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b", "c"]))

    def test_raises_when_backend_returns_more_embeddings(self):
        backend = MagicMock()
        backend.encode.return_value = [[0.1], [0.2], [0.3]]

        with pytest.raises(RuntimeError, match="returned 3 vectors for 2 input texts"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))

    def test_passes_through_aligned_embeddings(self):
        backend = MagicMock()
        backend.encode.return_value = [[0.1], [0.2]]

        result = asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))

        assert result == [[0.1], [0.2]]


class TestStreamingPublicationBoundary:
    @pytest.mark.asyncio
    async def test_exact_multiple_marks_only_real_final_batch_as_last(self):
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(("first",))
        queue.put_nowait(("second",))
        queue.put_nowait(None)
        calls: list[tuple[list[tuple], int, bool]] = []

        async def process_batch(batch: list[tuple], batch_index: int, is_last: bool) -> None:
            calls.append((list(batch), batch_index, is_last))

        await _consume_streaming_batches(
            queue,
            chunk_batch_size=1,
            process_batch=process_batch,
            producer_error=[],
            pipeline_aborted=[False],
        )

        assert calls == [
            ([("first",)], 0, False),
            ([("second",)], 1, True),
        ]
        assert sum(is_last for _, _, is_last in calls) == 1

    @pytest.mark.asyncio
    async def test_producer_error_suppresses_pending_final_batch(self):
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(("pending",))
        queue.put_nowait(None)
        calls: list[tuple[list[tuple], int, bool]] = []

        async def process_batch(batch: list[tuple], batch_index: int, is_last: bool) -> None:
            calls.append((list(batch), batch_index, is_last))

        await _consume_streaming_batches(
            queue,
            chunk_batch_size=1,
            process_batch=process_batch,
            producer_error=[RuntimeError("provider failed")],
            pipeline_aborted=[False],
        )

        assert calls == []

    @pytest.mark.asyncio
    async def test_takeover_drains_queue_and_raises_without_final_batch(self):
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(("first",))
        queue.put_nowait(("discarded",))
        queue.put_nowait(None)
        pipeline_aborted = [False]
        calls: list[tuple[list[tuple], int, bool]] = []

        async def process_batch(batch: list[tuple], batch_index: int, is_last: bool) -> None:
            calls.append((list(batch), batch_index, is_last))
            pipeline_aborted[0] = True

        with pytest.raises(RetainPublicationAborted, match="ownership was lost"):
            await _consume_streaming_batches(
                queue,
                chunk_batch_size=1,
                process_batch=process_batch,
                producer_error=[],
                pipeline_aborted=pipeline_aborted,
            )

        assert calls == [([("first",)], 0, False)]
        assert not any(is_last for _, _, is_last in calls)
        assert queue.empty()


class TestStreamingPublicationRecovery:
    @pytest.mark.asyncio
    async def test_zero_fact_final_batch_publishes_once_in_tracking_transaction(self, monkeypatch):
        connection = _FakeConnection()
        _extract, track_document, update_document = _install_streaming_mocks(monkeypatch, connection)

        async def assert_transactional_callback(callback_connection) -> None:
            assert callback_connection is connection
            assert connection.in_transaction

        callback = AsyncMock(side_effect=assert_transactional_callback)
        unit_ids, _usage, processed_tokens = await _run_streaming_publication(
            callback=callback,
            contents_dicts=[{"content": "chunk"}],
            chunks=["chunk"],
        )

        assert unit_ids == [[]]
        assert processed_tokens is None
        callback.assert_awaited_once_with(connection)
        track_document.assert_awaited_once()
        update_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_recovery_chunks_skipped_still_publishes_once(self, monkeypatch):
        connection = _FakeConnection()
        extract, track_document, update_document = _install_streaming_mocks(monkeypatch, connection)
        content = "already committed chunk"
        sanitized = orchestrator.fact_extraction._sanitize_text(content) or ""
        document_hash = orchestrator.hashlib.sha256(sanitized.encode()).hexdigest()
        connection.fetchrow.return_value = {"content_hash": document_hash}
        connection.fetchval.return_value = document_hash
        monkeypatch.setattr(
            orchestrator.chunk_storage,
            "load_existing_chunks",
            AsyncMock(
                return_value=[
                    SimpleNamespace(content_hash=orchestrator.chunk_storage.compute_chunk_hash(content)),
                ]
            ),
        )

        async def assert_transactional_callback(callback_connection) -> None:
            assert callback_connection is connection
            assert connection.in_transaction

        callback = AsyncMock(side_effect=assert_transactional_callback)
        unit_ids, _usage, processed_tokens = await _run_streaming_publication(
            callback=callback,
            contents_dicts=[{"content": content}],
            chunks=[content],
        )

        assert unit_ids == [[]]
        assert processed_tokens is None
        extract.assert_not_awaited()
        callback.assert_awaited_once_with(connection)
        track_document.assert_not_awaited()
        update_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_committed_operation_recovery_revalidates_owner_before_publish(self, monkeypatch):
        connection = _FakeConnection()
        extract, track_document, update_document = _install_streaming_mocks(monkeypatch, connection)
        content = "committed document"
        sanitized = orchestrator.fact_extraction._sanitize_text(content) or ""
        document_hash = orchestrator.hashlib.sha256(sanitized.encode()).hexdigest()
        connection.fetchrow.side_effect = [
            {"content_hash": document_hash},
            {
                "result_metadata": {
                    "document_ids": ["document"],
                    "facts_committed_document_ids": ["document"],
                    "unit_ids_count": 1,
                }
            },
        ]
        connection.fetchval.return_value = document_hash
        connection.fetch.return_value = [{"id": "unit-1"}]
        monkeypatch.setattr(orchestrator.chunk_storage, "load_existing_chunks", AsyncMock(return_value=[]))
        final_ann = AsyncMock()
        monkeypatch.setattr(orchestrator, "_run_final_semantic_ann", final_ann)

        async def assert_transactional_callback(callback_connection) -> None:
            assert callback_connection is connection
            assert connection.in_transaction

        callback = AsyncMock(side_effect=assert_transactional_callback)
        unit_ids, _usage, processed_tokens = await _run_streaming_publication(
            callback=callback,
            contents_dicts=[{"content": content}],
            chunks=[content],
            operation_id="00000000-0000-0000-0000-000000000001",
        )

        assert unit_ids == [["unit-1"]]
        assert processed_tokens is None
        extract.assert_not_awaited()
        callback.assert_awaited_once_with(connection)
        track_document.assert_not_awaited()
        update_document.assert_not_awaited()
        final_ann.assert_awaited_once()


def _install_stale_retain_mocks(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    connection = AsyncMock()
    connection.fetchrow.return_value = {"updated_at": datetime.now(UTC) + timedelta(minutes=1)}

    @asynccontextmanager
    async def acquire(_pool):
        yield connection

    monkeypatch.setattr(orchestrator, "acquire_with_retry", acquire)
    monkeypatch.setattr(
        orchestrator.bank_utils,
        "get_bank_profile",
        AsyncMock(return_value={"name": "test-agent"}),
    )
    monkeypatch.setattr(orchestrator, "ensure_bank_embedding_fingerprint", AsyncMock())
    return SimpleNamespace(
        embedding_fingerprint_policy="strict",
        embedding_fingerprint_legacy_attestation=None,
    )


class TestStaleRetainPublication:
    @pytest.mark.asyncio
    async def test_stale_retain_with_callback_is_not_reported_as_success(self, monkeypatch):
        config = _install_stale_retain_mocks(monkeypatch)
        callback = AsyncMock()

        with pytest.raises(RetainPublicationAborted, match="superseded before publication"):
            await orchestrator.retain_batch(
                pool=object(),
                embeddings_model=object(),
                llm_config=None,
                entity_resolver=None,
                format_date_fn=None,
                bank_id="bank",
                contents_dicts=[{"content": "older content"}],
                config=config,
                document_id="document",
                outbox_callback=callback,
            )

        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_retain_without_callback_preserves_legacy_noop(self, monkeypatch):
        config = _install_stale_retain_mocks(monkeypatch)

        unit_ids, _usage, processed_tokens = await orchestrator.retain_batch(
            pool=object(),
            embeddings_model=object(),
            llm_config=None,
            entity_resolver=None,
            format_date_fn=None,
            bank_id="bank",
            contents_dicts=[{"content": "older content"}],
            config=config,
            document_id="document",
        )

        assert unit_ids == [[]]
        assert processed_tokens == 0


class TestDeltaRetainPublicationFallback:
    @pytest.mark.parametrize("with_callback", [False, True])
    @pytest.mark.asyncio
    async def test_concurrent_hash_change_never_returns_false_success(self, monkeypatch, with_callback):
        old_hash = "old-document-hash"
        load_connection = AsyncMock()
        load_connection.fetchval.return_value = old_hash

        write_connection = AsyncMock()
        write_connection.fetchval.return_value = "newer-document-hash"

        @asynccontextmanager
        async def transaction():
            yield

        write_connection.transaction = transaction
        connections = iter((load_connection, write_connection))

        @asynccontextmanager
        async def acquire(_pool):
            yield next(connections)

        same_hash = orchestrator.chunk_storage.compute_chunk_hash("same")
        old_chunk_hash = orchestrator.chunk_storage.compute_chunk_hash("old")
        existing_chunks = [
            SimpleNamespace(chunk_index=0, content_hash=same_hash, chunk_id="chunk-0"),
            SimpleNamespace(chunk_index=1, content_hash=old_chunk_hash, chunk_id="chunk-1"),
        ]
        monkeypatch.setattr(orchestrator, "acquire_with_retry", acquire)
        monkeypatch.setattr(
            orchestrator.chunk_storage,
            "load_existing_chunks",
            AsyncMock(return_value=existing_chunks),
        )
        monkeypatch.setattr(
            orchestrator,
            "_chunk_contents_for_delta",
            lambda _contents, _config: {0: "same", 1: "changed"},
        )
        monkeypatch.setattr(
            orchestrator,
            "_extract_and_embed",
            AsyncMock(return_value=([], [], [], orchestrator.TokenUsage())),
        )
        monkeypatch.setattr(
            orchestrator,
            "_pre_resolve_phase1",
            AsyncMock(return_value=SimpleNamespace()),
        )

        entity_resolver = MagicMock()
        callback = AsyncMock() if with_callback else None
        config = SimpleNamespace(write_semantic_links=True)
        contents = [_make_content("whole document")]

        async def run_delta():
            return await orchestrator._try_delta_retain(
                pool=SimpleNamespace(ops=object()),
                embeddings_model=object(),
                llm_config=None,
                entity_resolver=entity_resolver,
                format_date_fn=None,
                bank_id="bank",
                contents_dicts=[{"content": "whole document"}],
                contents=contents,
                config=config,
                document_id="document",
                fact_type_override=None,
                document_tags=None,
                agent_name="test-agent",
                log_buffer=[],
                start_time=0.0,
                operation_id=None,
                schema=None,
                outbox_callback=callback,
            )

        if callback is None:
            assert await run_delta() is None
        else:
            with pytest.raises(RetainPublicationAborted, match="changed during retain publication"):
                await run_delta()
            callback.assert_not_awaited()
        assert entity_resolver.discard_pending_stats.call_count == 2
