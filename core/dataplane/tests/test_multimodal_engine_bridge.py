"""Offline tests for the parser-to-existing-retain bridge."""

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

import hms_api.engine.memory_engine as memory_engine_module
from hms_api.engine.memory_engine import MemoryEngine
from hms_api.engine.parsers import ConvertResult
from hms_api.engine.retain.orchestrator import _build_contents
from hms_api.worker.exceptions import DeferOperation


class _AsyncContext:
    def __init__(self, value=None) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Connection:
    def __init__(self, rows: list[dict | None] | None = None) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self.rows = list(rows or [])

    def transaction(self):
        return _AsyncContext()

    async def execute(self, sql, *args):
        self.executions.append((sql, args))

    async def fetchrow(self, sql, *args):
        self.executions.append((sql, args))
        if not self.rows:
            raise AssertionError(f"no scripted row remains for query: {sql}")
        return self.rows.pop(0)

    @staticmethod
    def parse_json(value):
        return json.loads(value) if isinstance(value, str) else value


class _Storage:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.deleted: list[str] = []
        self.available = True

    async def retrieve(self, key: str) -> bytes:
        assert key == "immutable/source/key"
        return self.data

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.available = False

    async def exists(self, key: str) -> bool:
        assert key == "immutable/source/key"
        return self.available


class _Registry:
    def __init__(self, result: ConvertResult) -> None:
        self.result = result
        self.call: dict | None = None

    async def convert_with_fallback(self, **kwargs) -> ConvertResult:
        self.call = kwargs
        return self.result


class _TaskBackend:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def submit_task(self, payload: dict) -> None:
        self.payloads.append(payload)


def _engine(result: ConvertResult, connection: _Connection) -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._file_storage = _Storage(b"raw-image-bytes")
    engine._parser_registry = _Registry(result)
    engine._operation_validator = None
    engine._task_backend = _TaskBackend()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    return engine


def _task(**overrides) -> dict:
    task = {
        "bank_id": "bank-a",
        "storage_key": "immutable/source/key",
        "document_id": "doc-a",
        "operation_id": "00000000-0000-0000-0000-000000000001",
        "original_filename": "screen.png",
        "content_type": "image/png",
        "asset_id": "asset_scoped",
        "asset_sha256": "a" * 64,
        "parser": ["openai_multimodal"],
        "context": "coding session",
        "metadata": {"customer_key": "kept", "media_kind": "spoofed"},
        "tags": ["project-a"],
        "document_tags": ["engineering"],
        "timestamp": "unset",
        "_tenant_id": "tenant-a",
    }
    task.update(overrides)
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_multimodal_child_retain_stage_metric_is_bounded(monkeypatch, failure: bool) -> None:
    metrics = MagicMock()
    monkeypatch.setattr(memory_engine_module, "get_metrics_collector", lambda: metrics)
    engine = object.__new__(MemoryEngine)
    engine._build_retain_outbox_callback = MagicMock(return_value=None)
    engine._build_multimodal_retain_callback = MagicMock(return_value=None)
    engine.retain_batch_async = AsyncMock(side_effect=RuntimeError("private failure") if failure else None)
    child_id = "00000000-0000-0000-0000-000000000099"

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(SimpleNamespace()))

    class _PublishedLedger:
        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                status="completed",
                sequence=1,
                child_retain_operation_id=UUID(child_id),
            )

        async def get_document_head(self, _conn, **_kwargs):
            return SimpleNamespace(published_sequence=1)

    from hms_api.engine.multimodal import ledger as ledger_module

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: _PublishedLedger())
    task = {
        "bank_id": "private-bank-must-not-be-a-label",
        "contents": [{"content": "canonical", "metadata": {"media_kind": "video"}}],
        "operation_id": child_id,
        "_multimodal_command": {
            "bank_id": "private-bank-must-not-be-a-label",
            "document_id": "private-document",
            "command_key": "f" * 64,
            "sequence": 1,
        },
    }

    if failure:
        with pytest.raises(RuntimeError, match="private failure"):
            await MemoryEngine._handle_batch_retain(engine, task)
    else:
        assert await MemoryEngine._handle_batch_retain(engine, task) is True

    metrics.record_multimodal_pipeline.assert_called_once()
    event = metrics.record_multimodal_pipeline.call_args.kwargs
    assert event["media_kind"] == "video"
    assert event["stage"] == "retain"
    assert event["success"] is (not failure)
    assert event.get("reason") == ("retain.failed" if failure else None)
    assert "bank" not in event and "document" not in event


@pytest.mark.asyncio
async def test_multimodal_child_fails_closed_when_publication_ledger_is_not_completed(monkeypatch) -> None:
    from hms_api.engine.multimodal import ledger as ledger_module
    from hms_api.engine.multimodal.ledger import LedgerInvariantError

    metrics = MagicMock()
    monkeypatch.setattr(memory_engine_module, "get_metrics_collector", lambda: metrics)
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(SimpleNamespace()))
    child_id = "00000000-0000-0000-0000-000000000098"
    engine = object.__new__(MemoryEngine)
    engine._build_retain_outbox_callback = MagicMock(return_value=None)
    engine._build_multimodal_retain_callback = MagicMock(return_value=None)
    engine.retain_batch_async = AsyncMock(return_value=None)

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend

    class _RetainingLedger:
        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                status="retaining",
                sequence=1,
                child_retain_operation_id=UUID(child_id),
            )

        async def get_document_head(self, _conn, **_kwargs):
            return SimpleNamespace(published_sequence=0)

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: _RetainingLedger())
    task = {
        "bank_id": "bank-a",
        "contents": [{"content": "canonical", "metadata": {"media_kind": "image"}}],
        "operation_id": child_id,
        "_multimodal_command": {
            "bank_id": "bank-a",
            "document_id": "doc-a",
            "command_key": "f" * 64,
            "sequence": 1,
        },
    }

    with pytest.raises(LedgerInvariantError, match="before durable document publication"):
        await MemoryEngine._handle_batch_retain(engine, task)

    event = metrics.record_multimodal_pipeline.call_args.kwargs
    assert event["success"] is False
    assert event["reason"] == "retain.failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_status", "expected_recall_ready", "expected_stage"),
    [
        ("completed", True, "recall_ready"),
        ("retaining", False, "retain_queued"),
    ],
)
async def test_operation_recall_ready_requires_child_and_command_publication(
    monkeypatch,
    command_status: str,
    expected_recall_ready: bool,
    expected_stage: str,
) -> None:
    from datetime import UTC, datetime

    from hms_api.engine.multimodal import ledger as ledger_module

    parent_id = "00000000-0000-0000-0000-000000000081"
    child_id = UUID("00000000-0000-0000-0000-000000000082")
    now = datetime.now(UTC)
    connection = _Connection(
        [
            {
                "operation_id": UUID(parent_id),
                "operation_type": "file_convert_retain",
                "created_at": now,
                "updated_at": now,
                "completed_at": now,
                "status": "completed",
                "error_message": None,
                "result_metadata": {
                    "multimodal": {
                        "stage": "retain_queued",
                        "child_retain_operation_id": str(child_id),
                        "retryable": False,
                    }
                },
                "retry_count": 0,
                "next_retry_at": None,
                "task_payload": {
                    "document_id": "doc-a",
                    "document_command_key": "f" * 64,
                },
            },
            {"status": "completed"},
        ]
    )
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    class _Ledger:
        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                status=command_status,
                sequence=1,
                child_retain_operation_id=child_id,
            )

        async def get_document_head(self, _conn, **_kwargs):
            return SimpleNamespace(published_sequence=1 if command_status == "completed" else 0)

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: _Ledger())
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock(return_value=None)

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend

    result = await MemoryEngine.get_operation_status(
        engine,
        "bank-a",
        parent_id,
        request_context=SimpleNamespace(),
    )

    multimodal = result["result_metadata"]["multimodal"]
    assert multimodal["child_retain_status"] == "completed"
    assert multimodal["recall_ready"] is expected_recall_ready
    assert multimodal["stage"] == expected_stage
    assert result["task_payload"] is None
    assert "task_payload" in connection.executions[0][0]


@pytest.mark.asyncio
async def test_stale_descriptor_worker_defers_without_terminalizing_shared_command(monkeypatch) -> None:
    from hms_api.engine.multimodal.ledger import LedgerConflictError

    class _ConflictRegistry:
        async def convert_with_fallback(self, **_kwargs):
            raise LedgerConflictError("descriptor lease moved")

    class _Ledger:
        def __init__(self):
            self.fail_called = False
            self.terminal_called = False

        async def purge_expired_video_segment_checkpoints(self, *_args, **_kwargs):
            return 0

        async def purge_expired_descriptors(self, *_args, **_kwargs):
            return 0

        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                operation_id=UUID("00000000-0000-0000-0000-000000000001"),
                status="processing",
            )

        async def get_reusable_descriptor(self, *_args, **_kwargs):
            return None

        async def claim_descriptor(self, *_args, **_kwargs):
            return SimpleNamespace(status="processing")

        async def fail_descriptor(self, *_args, **_kwargs):
            self.fail_called = True
            raise LedgerConflictError("stale descriptor owner")

        async def mark_document_terminal(self, *_args, **_kwargs):
            self.terminal_called = True

    ledger = _Ledger()
    from hms_api.engine.multimodal import ledger as ledger_module

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(SimpleNamespace()))

    engine = object.__new__(MemoryEngine)
    engine._file_storage = _Storage(b"raw-image-bytes")
    engine._parser_registry = _ConflictRegistry()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    task = _task(
        descriptor_key="a" * 64,
        document_command_key="b" * 64,
        pipeline_fingerprint="c" * 64,
    )

    with pytest.raises(DeferOperation, match="descriptor claim moved"):
        await MemoryEngine._handle_file_convert_retain(engine, task)

    assert ledger.fail_called is False
    assert ledger.terminal_called is False


@pytest.mark.asyncio
async def test_descriptor_release_and_terminalization_hold_one_claim_transaction(monkeypatch) -> None:
    """A late failure cannot close a command after a replacement claim wins.

    The failure handler must keep the descriptor row lock from ``fail_descriptor``
    until ``mark_document_terminal`` commits.  This fake models a second worker
    attempting takeover exactly after the release statement: with no outer
    transaction it would acquire the claim before terminalization; with the
    transaction it is deferred until the stale worker has finished both writes.
    """

    from hms_api.engine.parsers import ParserProcessingError

    class _TrackingConnection(_Connection):
        def __init__(self) -> None:
            super().__init__()
            self.transaction_depth = 0
            self.takeover_deferred = False
            self.takeover_won = False

        def transaction(self):
            connection = self

            class _Transaction:
                async def __aenter__(self):
                    connection.transaction_depth += 1
                    return connection

                async def __aexit__(self, exc_type, exc, traceback):
                    connection.transaction_depth -= 1
                    if connection.transaction_depth == 0 and connection.takeover_deferred and exc_type is None:
                        connection.takeover_won = True
                    return False

            return _Transaction()

    class _FailingRegistry:
        async def convert_with_fallback(self, **_kwargs):
            raise ParserProcessingError("media.invalid", "synthetic terminal conversion failure")

    class _Ledger:
        def __init__(self, connection: _TrackingConnection) -> None:
            self.connection = connection
            self.terminal_saw_takeover = None

        async def purge_expired_video_segment_checkpoints(self, *_args, **_kwargs):
            return 0

        async def purge_expired_descriptors(self, *_args, **_kwargs):
            return 0

        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                operation_id=UUID("00000000-0000-0000-0000-000000000001"),
                status="processing",
            )

        async def get_reusable_descriptor(self, *_args, **_kwargs):
            return None

        async def claim_descriptor(self, *_args, **_kwargs):
            return SimpleNamespace(status="processing")

        async def fail_descriptor(self, _conn, **_kwargs):
            # A real takeover blocks on the row lock held by the outer
            # transaction.  Record the attempt and release it only at commit.
            assert self.connection.transaction_depth >= 1
            self.connection.takeover_deferred = True

        async def mark_document_terminal(self, _conn, **_kwargs):
            self.terminal_saw_takeover = self.connection.takeover_won

        async def fail_descriptor_and_mark_document_terminal(self, conn, **kwargs):
            # Mirror the production ledger composite method: both operations
            # must execute under one transaction scope.
            async with conn.transaction():
                await self.fail_descriptor(conn, **kwargs)
                await self.mark_document_terminal(conn, **kwargs)

    connection = _TrackingConnection()
    ledger = _Ledger(connection)
    from hms_api.engine.multimodal import ledger as ledger_module

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    engine = object.__new__(MemoryEngine)
    engine._file_storage = _Storage(b"raw-image-bytes")
    engine._parser_registry = _FailingRegistry()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    task = _task(
        descriptor_key="a" * 64,
        document_command_key="b" * 64,
        pipeline_fingerprint="c" * 64,
    )

    with pytest.raises(RuntimeError, match="Multimodal file conversion failed"):
        await MemoryEngine._handle_file_convert_retain(engine, task)

    assert ledger.terminal_saw_takeover is False
    assert connection.takeover_won is True


@pytest.mark.asyncio
async def test_failure_cleanup_losing_descriptor_claim_defers_without_writing_failure(monkeypatch) -> None:
    from hms_api.engine.multimodal.ledger import LedgerConflictError
    from hms_api.engine.parsers import ParserProcessingError

    class _FailingRegistry:
        async def convert_with_fallback(self, **_kwargs):
            raise ParserProcessingError("media.invalid", "synthetic terminal conversion failure")

    class _Ledger:
        async def purge_expired_video_segment_checkpoints(self, *_args, **_kwargs):
            return 0

        async def purge_expired_descriptors(self, *_args, **_kwargs):
            return 0

        async def get_document_command(self, _conn, **_kwargs):
            return SimpleNamespace(
                operation_id=UUID("00000000-0000-0000-0000-000000000001"),
                status="processing",
            )

        async def get_reusable_descriptor(self, *_args, **_kwargs):
            return None

        async def claim_descriptor(self, *_args, **_kwargs):
            return SimpleNamespace(status="processing")

        async def fail_descriptor_and_mark_document_terminal(self, *_args, **_kwargs):
            raise LedgerConflictError("descriptor claim moved")

    connection = _Connection()
    ledger = _Ledger()
    from hms_api.engine.multimodal import ledger as ledger_module

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    engine = object.__new__(MemoryEngine)
    engine._file_storage = _Storage(b"raw-image-bytes")
    engine._parser_registry = _FailingRegistry()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    task = _task(
        descriptor_key="a" * 64,
        document_command_key="b" * 64,
        pipeline_fingerprint="c" * 64,
    )

    with pytest.raises(DeferOperation, match="descriptor claim moved"):
        await MemoryEngine._handle_file_convert_retain(engine, task)

    assert not any("SET result_metadata" in sql for sql, _args in connection.executions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delete_after_retain", "expected_source_state"),
    [(False, "retained"), (True, "deleted")],
)
async def test_rich_parser_output_is_whitelisted_into_existing_child_retain(
    monkeypatch,
    delete_after_retain: bool,
    expected_source_state: str,
) -> None:
    import hms_api.config

    raw_config = hms_api.config._get_raw_config()
    config = replace(raw_config, file_delete_after_retain=delete_after_retain)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    metrics = MagicMock()
    monkeypatch.setattr(memory_engine_module, "get_metrics_collector", lambda: metrics)

    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    result = ConvertResult(
        content="# Media memory\n\nGrounded description.\n",
        parser_name="openai_multimodal",
        metadata={
            "media_asset_sha256": "a" * 64,
            "media_kind": "image",
            "media_descriptor_model": "gpt-5-mini",
            "media_source_available": "true",
        },
        entities=[{"text": "Python", "type": "CONCEPT"}],
        retain_extraction_mode="chunks",
        pipeline_metadata={
            "asset_id": "asset_scoped",
            "asset_sha256": "a" * 64,
            "media_kind": "image",
            "pipeline_version": "hms-multimodal-v1",
            "descriptor_model": "gpt-5-mini",
            "stage": "normalized",
            "raw_provider_response": "data:image/png;base64,SENSITIVE_SENTINEL",
        },
    )
    engine = _engine(result, connection)

    await MemoryEngine._handle_file_convert_retain(engine, _task())

    assert engine._parser_registry.call["asset_id"] == "asset_scoped"
    assert engine._parser_registry.call["asset_sha256"] == "a" * 64
    [child_payload] = engine._task_backend.payloads
    assert child_payload["type"] == "batch_retain"
    assert child_payload["_retain_extraction_mode"] == "chunks"
    [content] = child_payload["contents"]
    assert child_payload["document_tags"] == ["engineering"]
    assert content["metadata"]["customer_key"] == "kept"
    assert content["metadata"]["media_kind"] == "image"
    merged_content = _build_contents(child_payload["contents"], child_payload["document_tags"])[0]
    assert set(merged_content.tags) == {"project-a", "engineering"}
    assert content["entities"] == [{"text": "Python", "type": "CONCEPT"}]
    assert content["event_date"] is None
    assert content["metadata"]["media_source_available"] == str(not delete_after_retain).lower()
    assert engine._file_storage.deleted == (["immutable/source/key"] if delete_after_retain else [])
    metric_calls = [call.kwargs for call in metrics.record_multimodal_pipeline.call_args_list]
    assert {call["stage"] for call in metric_calls} == {"source_lifecycle", "retain_queue"}
    source_metric = next(call for call in metric_calls if call["stage"] == "source_lifecycle")
    assert source_metric["media_kind"] == "image"
    assert source_metric["success"] is True
    assert source_metric["source_state"] == expected_source_state
    queue_metric = next(call for call in metric_calls if call["stage"] == "retain_queue")
    assert queue_metric["media_kind"] == "image"
    assert queue_metric["success"] is True
    assert queue_metric["duration"] >= 0

    update_execution = next(item for item in connection.executions if "SET status = 'completed'" in item[0])
    public_metadata = json.loads(update_execution[1][1])
    multimodal = public_metadata["multimodal"]
    assert multimodal["stage"] == "retain_queued"
    assert multimodal["child_retain_operation_id"] == child_payload["operation_id"]
    assert "raw_provider_response" not in multimodal

    observable = json.dumps({"task": child_payload, "operation": public_metadata})
    assert "base64," not in observable
    assert "SENSITIVE_SENTINEL" not in observable


@pytest.mark.asyncio
async def test_explicit_public_strategy_cannot_override_canonical_chunks(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    result = ConvertResult(
        content="canonical",
        parser_name="openai_multimodal",
        retain_extraction_mode="chunks",
    )
    engine = _engine(result, connection)

    with pytest.raises(ValueError, match="explicit retain strategy"):
        await MemoryEngine._handle_file_convert_retain(engine, _task(strategy="concise"))

    assert engine._task_backend.payloads == []
    assert connection.executions == []


@pytest.mark.asyncio
async def test_trusted_chunks_override_is_applied_after_public_strategy(monkeypatch) -> None:
    import hms_api.config_resolver
    from hms_api.engine.retain import orchestrator

    engine = object.__new__(MemoryEngine)
    resolved = SimpleNamespace(
        retain_extraction_mode="facts",
        retain_default_strategy=None,
        retain_chunk_size=800,
        enable_observations=True,
    )

    class _Resolver:
        async def resolve_full_config(self, bank_id, request_context):
            return resolved

    engine._config_resolver = _Resolver()
    engine._llm_config = SimpleNamespace(provider="mock")
    engine._backend = SimpleNamespace()
    engine.embeddings = SimpleNamespace()
    engine._retain_llm_config = SimpleNamespace(with_config=lambda config: config)
    engine.entity_resolver = SimpleNamespace()
    engine._format_readable_date = lambda value: str(value)
    engine._put_semaphore = SimpleNamespace()

    async def get_backend():
        return engine._backend

    engine._get_backend = get_backend

    def apply_strategy(config, strategy):
        assert strategy == "public-facts"
        config.retain_extraction_mode = "facts"
        config.enable_observations = True
        return config

    captured = {}

    async def retain_batch(**kwargs):
        captured.update(kwargs)
        return [[]], SimpleNamespace(), 0

    monkeypatch.setattr(hms_api.config_resolver, "apply_strategy", apply_strategy)
    monkeypatch.setattr(orchestrator, "retain_batch", retain_batch)
    monkeypatch.setattr(memory_engine_module, "create_operation_span", lambda *args, **kwargs: nullcontext())

    await MemoryEngine._retain_batch_async_internal(
        engine,
        bank_id="bank-a",
        contents=[{"content": "canonical"}],
        request_context=SimpleNamespace(),
        strategy="public-facts",
        _retain_extraction_mode="chunks",
    )

    assert captured["config"].retain_extraction_mode == "chunks"
    assert captured["config"].enable_observations is False
    assert captured["config"].retain_chunk_size == 2_400


@pytest.mark.asyncio
async def test_failed_multimodal_child_closes_command_and_sanitizes_parent(monkeypatch) -> None:
    child_id = "00000000-0000-0000-0000-000000000002"
    parent_id = UUID("00000000-0000-0000-0000-000000000001")
    command_payload = {
        "bank_id": "bank-a",
        "document_id": "doc-a",
        "command_key": "e" * 64,
        "sequence": 1,
    }
    connection = _Connection(
        [
            {
                "bank_id": "bank-a",
                "operation_type": "retain",
                "status": "processing",
                "task_payload": {
                    "type": "batch_retain",
                    "operation_id": child_id,
                    "bank_id": "bank-a",
                    "_multimodal_command": command_payload,
                    "contents": [{"content": "data:image/png;base64,PRIVATE_PAYLOAD"}],
                },
            },
            {"operation_id": UUID(child_id)},
            {
                "result_metadata": {
                    "original_filename": "screen.png",
                    "multimodal": {
                        "asset_id": "asset-scoped",
                        "child_retain_operation_id": child_id,
                        "stage": "retain_queued",
                    },
                }
            },
        ]
    )
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    captured: dict = {}

    class _Ledger:
        async def mark_document_terminal(self, _conn, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                operation_id=parent_id,
                child_retain_operation_id=UUID(child_id),
            )

    from hms_api.engine.multimodal import ledger as ledger_module

    monkeypatch.setattr(ledger_module.MultimodalLedger, "for_connection", lambda *_args, **_kwargs: _Ledger())

    engine = object.__new__(MemoryEngine)
    engine._maybe_update_parent_operation = AsyncMock()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    await MemoryEngine._mark_operation_failed(
        engine,
        child_id,
        "PRIVATE_PROVIDER_ERROR",
        "PRIVATE_TRACEBACK",
    )

    assert captured["bank_id"] == "bank-a"
    assert captured["document_id"] == "doc-a"
    assert captured["command_key"] == "e" * 64
    assert captured["status"] == "failed"
    assert captured["expected_sequence"] == 1
    assert captured["expected_child_retain_operation_id"] == UUID(child_id)
    engine._maybe_update_parent_operation.assert_not_awaited()

    child_update = next(item for item in connection.executions if "SET status = 'failed'" in item[0])
    assert "status IN ('pending', 'processing')" in child_update[0]
    parent_update = next(item for item in connection.executions if "SET result_metadata = $2" in item[0])
    assert "SET status" not in parent_update[0]
    parent_metadata = json.loads(parent_update[1][1])
    assert parent_metadata["multimodal"] == {
        "asset_id": "asset-scoped",
        "child_retain_operation_id": child_id,
        "stage": "retain_failed",
        "child_retain_status": "failed",
        "recall_ready": False,
        "retryable": True,
        "sanitized_error_code": "retain_failed",
    }
    observable_parent = json.dumps(parent_metadata)
    assert "PRIVATE_PAYLOAD" not in observable_parent
    assert "PRIVATE_PROVIDER_ERROR" not in observable_parent
    assert "PRIVATE_TRACEBACK" not in observable_parent


@pytest.mark.asyncio
async def test_late_completed_multimodal_child_failure_does_not_downgrade(monkeypatch) -> None:
    child_id = "00000000-0000-0000-0000-000000000012"
    connection = _Connection(
        [
            {
                "bank_id": "bank-a",
                "operation_type": "retain",
                "status": "completed",
                "task_payload": {
                    "type": "batch_retain",
                    "operation_id": child_id,
                    "bank_id": "bank-a",
                    "_multimodal_command": {
                        "bank_id": "bank-a",
                        "document_id": "doc-a",
                        "command_key": "e" * 64,
                        "sequence": 1,
                    },
                },
            }
        ]
    )
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    engine = object.__new__(MemoryEngine)
    engine._maybe_update_parent_operation = AsyncMock()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    await MemoryEngine._mark_operation_failed(engine, child_id, "late failure", "late traceback")

    assert len(connection.executions) == 1
    engine._maybe_update_parent_operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_child_failure_keeps_existing_parent_propagation(monkeypatch) -> None:
    operation_id = "00000000-0000-0000-0000-000000000022"
    connection = _Connection(
        [
            {
                "bank_id": "bank-a",
                "operation_type": "retain",
                "status": "processing",
                "task_payload": {
                    "type": "batch_retain",
                    "operation_id": operation_id,
                    "bank_id": "bank-a",
                },
            },
            {"operation_id": UUID(operation_id)},
        ]
    )
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    engine = object.__new__(MemoryEngine)
    engine._maybe_update_parent_operation = AsyncMock()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    await MemoryEngine._mark_operation_failed(engine, operation_id, "legacy failure", "legacy traceback")

    legacy_update = next(item for item in connection.executions if "status <> 'cancelled'" in item[0])
    assert legacy_update[1][0] == UUID(operation_id)
    engine._maybe_update_parent_operation.assert_awaited_once_with(operation_id, connection)
