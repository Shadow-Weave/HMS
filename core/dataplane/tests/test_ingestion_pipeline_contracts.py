"""Focused offline contracts for the Retain ingestion application boundary."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hms_api.engine import embedding_fingerprint as fingerprint_module
from hms_api.engine import memory_engine as memory_engine_module
from hms_api.engine.embedding_fingerprint import (
    EmbeddingFingerprintMismatchError,
)
from hms_api.engine.embedding_fingerprint import (
    embedding_model_version as shared_embedding_model_version,
)
from hms_api.engine.ingestion import (
    RetainExecutionContext,
    RetainInvocation,
    RetainOperationInactiveError,
    RetainPublicationAborted,
)
from hms_api.engine.ingestion import service as service_module
from hms_api.engine.ingestion.domain import DocumentChangeKind
from hms_api.engine.ingestion.persistence import writer as writer_module
from hms_api.engine.ingestion.persistence.operation_fence import OperationActivityFence
from hms_api.engine.ingestion.persistence.unit_of_work import (
    CoreGraphWrite,
    FirstFullWriteWindow,
    MetadataOnlyWriteRequest,
    RetainUnitOfWork,
    WriteWindowRequest,
)
from hms_api.engine.ingestion.persistence.writer import PersistenceWriter
from hms_api.engine.ingestion.redaction import IdentifierSanitizer
from hms_api.engine.ingestion.runtime import embedding_model_version
from hms_api.engine.memory_engine import MemoryEngine
from hms_api.engine.response_models import TokenUsage
from hms_api.engine.retain import fact_storage
from hms_api.engine.retain.types import RetainContent


def _embedding_model() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="local",
        model="BAAI/bge-small-en-v1.5",
        dimension=384,
        normalization=True,
    )


def _config(**overrides) -> SimpleNamespace:
    values = {
        "database_backend": "postgresql",
        "retain_extraction_mode": "chunks",
        "retain_chunk_size": 3000,
        "embedding_fingerprint_policy": "strict",
        "embedding_fingerprint_legacy_attestation": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _execution(*, config=None) -> RetainExecutionContext:
    return RetainExecutionContext(
        pool=SimpleNamespace(backend_type="postgresql", ops=object()),
        embeddings_model=_embedding_model(),
        llm_config=object(),
        entity_resolver=SimpleNamespace(discard_pending_stats=lambda: None),
        format_date_fn=lambda *_args, **_kwargs: "",
        resolved_config=config or _config(),
    )


class _Ownership:
    def __init__(self, events: list[str], *, owns: bool = True) -> None:
        self._events = events
        self._owns = owns

    async def prepare_first_window(self, _connection, *, bank_id, document_id) -> None:
        del bank_id, document_id
        self._events.append("ownership")

    async def validate_later_window(
        self,
        _connection,
        *,
        bank_id,
        document_id,
        expected_content_hash,
    ) -> bool:
        del bank_id, document_id, expected_content_hash
        self._events.append("ownership")
        return self._owns

    async def validate_unhashed_window(
        self,
        _connection,
        *,
        bank_id,
        document_id,
    ) -> bool:
        del bank_id, document_id
        self._events.append("unhashed-ownership")
        return self._owns

    async def transition_content_hash(
        self,
        _connection,
        *,
        bank_id,
        document_id,
        expected_content_hash,
        new_content_hash,
    ) -> bool:
        del bank_id, document_id, expected_content_hash, new_content_hash
        self._events.append("transition")
        return self._owns


class _Connection:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.in_transaction = False

    def transaction(self):
        connection = self

        class _Transaction:
            async def __aenter__(self):
                connection.in_transaction = True
                connection._events.append("begin")
                return connection

            async def __aexit__(self, exc_type, _exc, _traceback):
                connection._events.append("rollback" if exc_type is not None else "commit")
                connection.in_transaction = False
                return False

        return _Transaction()


def _writer(
    events: list[str],
    *,
    sanitize: bool = False,
    owns: bool = True,
    operation_activity=None,
) -> PersistenceWriter:
    return PersistenceWriter(
        pool=SimpleNamespace(ops=object()),
        embeddings_model=_embedding_model(),
        entity_resolver=object(),
        config=_config(),
        ownership=_Ownership(events, owns=owns),
        operation_activity=operation_activity,
        sanitize_log_identifiers=sanitize,
    )


def _unit_of_work(writer: PersistenceWriter, connection: _Connection) -> RetainUnitOfWork:
    @asynccontextmanager
    async def connection_scope():
        yield connection

    return RetainUnitOfWork(connection_scope=connection_scope, adapter=writer)


def test_runtime_uses_shared_embedding_compatibility_version() -> None:
    model = _embedding_model()

    version = embedding_model_version(model)

    assert version == shared_embedding_model_version(model)
    assert version.startswith("fp:")
    assert len(version) == len("fp:") + 64


@pytest.mark.asyncio
async def test_tracked_anonymous_batch_is_one_retry_stable_document(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._check_op_alive = AsyncMock(return_value=True)
    engine._replace_vector_index_document = AsyncMock()
    engine._replace_vector_index_fact_type = AsyncMock()
    engine._sync_vector_index_units = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_observations=False))
    )
    monkeypatch.setattr(
        memory_engine_module,
        "get_config",
        lambda: SimpleNamespace(retain_batch_tokens=1),
    )

    committed_units_by_document: dict[str, list[str]] = {}
    internal_calls: list[tuple[tuple[str, ...], bool, bool]] = []
    outbox_calls: list[object] = []

    async def outbox_callback(connection) -> None:
        outbox_calls.append(connection)

    async def retain_internal(**kwargs):
        document_ids = tuple(item["document_id"] for item in kwargs["contents"])
        callback = kwargs["outbox_callback"]
        internal_calls.append(
            (
                document_ids,
                kwargs["is_first_batch"],
                callback is not None,
            )
        )
        assert len(set(document_ids)) == 1
        document_id = document_ids[0]
        is_recovery = document_id in committed_units_by_document
        if not is_recovery:
            committed_units_by_document[document_id] = [f"unit-{index + 1}" for index in range(len(kwargs["contents"]))]
        if callback is not None and not is_recovery:
            await callback(object())
        return [[unit_id] for unit_id in committed_units_by_document[document_id]], TokenUsage(), 1

    engine._retain_batch_async_internal = retain_internal
    operation_id = str(uuid.uuid4())

    request = {
        "bank_id": "bank",
        "contents": [
            {"content": "first anonymous item"},
            {"content": "second anonymous item"},
            {"content": "third anonymous item"},
        ],
        "request_context": object(),
        "operation_id": operation_id,
        "outbox_callback": outbox_callback,
    }
    result = await engine.retain_batch_async(**request)
    retry_result = await engine.retain_batch_async(**request)

    assert result == [["unit-1"], ["unit-2"], ["unit-3"]]
    assert retry_result == result
    assert len(committed_units_by_document) == 1
    assert len(internal_calls) == 2
    assert internal_calls[0][0] == internal_calls[1][0]
    assert internal_calls[0][1:] == (True, True)
    assert internal_calls[1][1:] == (True, True)
    assert len(outbox_calls) == 1


@pytest.mark.asyncio
async def test_tracked_token_retry_survives_threshold_change_without_checkpoint_alias(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._check_op_alive = AsyncMock(return_value=True)
    engine._replace_vector_index_document = AsyncMock()
    engine._replace_vector_index_fact_type = AsyncMock()
    engine._sync_vector_index_units = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_observations=False))
    )
    config = SimpleNamespace(retain_batch_tokens=1)
    monkeypatch.setattr(memory_engine_module, "get_config", lambda: config)

    operation_id = str(uuid.uuid4())
    committed_units_by_document: dict[str, str] = {}
    anonymous_ids: list[str] = []
    outbox_calls: list[object] = []
    fail_second_call = True
    internal_call_count = 0

    async def outbox_callback(connection) -> None:
        outbox_calls.append(connection)

    async def retain_internal(**kwargs):
        nonlocal fail_second_call, internal_call_count
        internal_call_count += 1
        for item in kwargs["contents"]:
            if item["content"] == "anonymous payload":
                anonymous_ids.append(item["document_id"])
        if fail_second_call and internal_call_count == 2:
            fail_second_call = False
            raise RuntimeError("synthetic crash after first document commit")

        results: list[list[str]] = []
        committed_new_document = False
        for item in kwargs["contents"]:
            document_id = item["document_id"]
            if document_id not in committed_units_by_document:
                committed_units_by_document[document_id] = f"unit-{len(committed_units_by_document) + 1}"
                committed_new_document = True
            results.append([committed_units_by_document[document_id]])
        if kwargs["outbox_callback"] is not None and committed_new_document:
            await kwargs["outbox_callback"](object())
        return results, TokenUsage(), 1

    engine._retain_batch_async_internal = retain_internal
    request = {
        "bank_id": "bank",
        "contents": [
            {"content": "first explicit payload", "document_id": "document-a"},
            {"content": "anonymous payload"},
            {"content": "second explicit payload", "document_id": "document-b"},
        ],
        "request_context": object(),
        "operation_id": operation_id,
        "outbox_callback": outbox_callback,
    }

    with pytest.raises(RuntimeError, match="synthetic crash"):
        await engine.retain_batch_async(**request)

    config.retain_batch_tokens = 10_000
    result = await engine.retain_batch_async(**request)

    assert len(result) == 3
    assert len(committed_units_by_document) == 3
    assert len(anonymous_ids) == 2
    assert len(set(anonymous_ids)) == 1
    assert len(outbox_calls) == 1


@pytest.mark.asyncio
async def test_token_batching_keeps_repeated_document_group_and_restores_input_order(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._check_op_alive = AsyncMock(return_value=True)
    engine._replace_vector_index_document = AsyncMock()
    engine._replace_vector_index_fact_type = AsyncMock()
    engine._sync_vector_index_units = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_observations=False))
    )
    monkeypatch.setattr(
        memory_engine_module,
        "get_config",
        lambda: SimpleNamespace(retain_batch_tokens=1),
    )
    submitted_groups: list[tuple[str, ...]] = []

    async def retain_internal(**kwargs):
        submitted_groups.append(tuple(item["slot"] for item in kwargs["contents"]))
        return [[item["slot"]] for item in kwargs["contents"]], TokenUsage(), 1

    engine._retain_batch_async_internal = retain_internal
    result = await engine.retain_batch_async(
        bank_id="bank",
        contents=[
            {"content": "alpha one", "document_id": "document-a", "slot": "slot-0"},
            {"content": "beta", "document_id": "document-b", "slot": "slot-1"},
            {"content": "alpha two", "document_id": "document-a", "slot": "slot-2"},
        ],
        request_context=object(),
        operation_id=str(uuid.uuid4()),
    )

    assert submitted_groups == [("slot-0", "slot-2"), ("slot-1",)]
    assert result == [["slot-0"], ["slot-1"], ["slot-2"]]


@pytest.mark.asyncio
async def test_tracked_token_cancellation_raises_without_partial_success_or_outbox(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._check_op_alive = AsyncMock(side_effect=[True, False])
    engine._replace_vector_index_document = AsyncMock()
    engine._replace_vector_index_fact_type = AsyncMock()
    engine._sync_vector_index_units = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_observations=False))
    )
    monkeypatch.setattr(
        memory_engine_module,
        "get_config",
        lambda: SimpleNamespace(retain_batch_tokens=1),
    )
    internal = AsyncMock(return_value=([["unit-a"]], TokenUsage(), 1))
    engine._retain_batch_async_internal = internal
    outbox = AsyncMock()

    with pytest.raises(
        memory_engine_module._RetainOperationCancelled,
        match="cancelled between logical document batches",
    ):
        await engine.retain_batch_async(
            bank_id="bank",
            contents=[
                {"content": "first payload", "document_id": "document-a"},
                {"content": "second payload", "document_id": "document-b"},
            ],
            request_context=object(),
            operation_id=str(uuid.uuid4()),
            outbox_callback=outbox,
        )

    assert internal.await_count == 1
    outbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_completed_uses_terminal_safe_compare_and_set(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._get_backend = AsyncMock(return_value=object())
    queries: list[str] = []

    class CompletionConnection:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query, *_args):
            queries.append(query)
            return None

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield CompletionConnection()

    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", connection_scope)

    await engine._mark_operation_completed(str(uuid.uuid4()))

    assert len(queries) == 1
    assert "status IN ('pending', 'processing')" in queries[0]


@pytest.mark.asyncio
async def test_execute_task_treats_retain_cancellation_as_terminal() -> None:
    engine = object.__new__(MemoryEngine)
    engine._audit_logger = None
    engine._handle_batch_retain = AsyncMock(
        side_effect=memory_engine_module._RetainOperationCancelled("cancelled"),
    )
    engine._mark_operation_completed = AsyncMock()

    await engine.execute_task(
        {
            "type": "batch_retain",
            "bank_id": "bank",
            "contents": [],
        }
    )

    engine._mark_operation_completed.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_liveness_fails_closed_when_batch_parent_is_cancelled(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._get_backend = AsyncMock(return_value=object())
    child_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    queries: list[str] = []

    class LivenessConnection:
        def parse_json(self, value):
            return value

        async def fetchrow(self, query, *_args):
            queries.append(query)
            if len(queries) == 1:
                return {
                    "status": "processing",
                    "bank_id": "bank",
                    "result_metadata": {"parent_operation_id": str(parent_id)},
                }
            return {"status": "cancelled"}

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield LivenessConnection()

    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", connection_scope)

    assert await engine._check_op_alive(str(child_id)) is False
    assert len(queries) == 2


@pytest.mark.asyncio
async def test_parent_aggregation_never_overwrites_terminal_cancellation() -> None:
    engine = object.__new__(MemoryEngine)
    parent_id = uuid.uuid4()
    fetch_siblings = AsyncMock(side_effect=AssertionError("terminal parent must stop aggregation"))
    execute = AsyncMock(side_effect=AssertionError("terminal parent must not be updated"))

    class AggregationConnection:
        def __init__(self) -> None:
            self.fetchrow_calls = 0
            self.fetch = fetch_siblings
            self.execute = execute

        def parse_json(self, value):
            return value

        async def fetchrow(self, _query, *_args):
            self.fetchrow_calls += 1
            if self.fetchrow_calls == 1:
                return {
                    "bank_id": "bank",
                    "result_metadata": {"parent_operation_id": str(parent_id)},
                }
            return {"operation_id": parent_id, "status": "cancelled"}

    await engine._maybe_update_parent_operation(str(uuid.uuid4()), AggregationConnection())

    fetch_siblings.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_child_is_terminal_and_cancels_active_parent() -> None:
    engine = object.__new__(MemoryEngine)
    parent_id = uuid.uuid4()
    updates: list[tuple[str, tuple[object, ...]]] = []

    class AggregationConnection:
        def __init__(self) -> None:
            self.fetchrow_calls = 0

        def parse_json(self, value):
            return value

        async def fetchrow(self, _query, *_args):
            self.fetchrow_calls += 1
            if self.fetchrow_calls == 1:
                return {
                    "bank_id": "bank",
                    "result_metadata": {"parent_operation_id": str(parent_id)},
                }
            return {"operation_id": parent_id, "status": "pending"}

        async def fetch(self, _query, *_args):
            return [
                {"status": "completed", "error_message": None},
                {"status": "cancelled", "error_message": None},
            ]

        async def execute(self, query, *args):
            updates.append((query, args))

    await engine._maybe_update_parent_operation(str(uuid.uuid4()), AggregationConnection())

    assert len(updates) == 1
    query, args = updates[0]
    assert args[1] == "cancelled"
    assert "status IN ('pending', 'processing')" in query


@pytest.mark.asyncio
async def test_poller_parent_aggregation_treats_cancelled_as_terminal() -> None:
    from hms_api.worker.poller import WorkerPoller

    poller = object.__new__(WorkerPoller)
    parent_id = uuid.uuid4()
    updates: list[tuple[str, tuple[object, ...]]] = []

    class PollerAggregationConnection:
        def __init__(self) -> None:
            self.fetchrow_calls = 0

        async def fetchrow(self, _query, *_args):
            self.fetchrow_calls += 1
            if self.fetchrow_calls == 1:
                return {
                    "bank_id": "bank",
                    "result_metadata": {"parent_operation_id": str(parent_id)},
                }
            return {"operation_id": parent_id, "status": "pending"}

        async def fetch(self, _query, *_args):
            return [
                {"status": "completed", "error_message": None},
                {"status": "cancelled", "error_message": None},
            ]

        async def execute(self, query, *args):
            updates.append((query, args))

    await poller._maybe_update_parent_operation(
        str(uuid.uuid4()),
        None,
        PollerAggregationConnection(),
    )

    assert len(updates) == 1
    query, _args = updates[0]
    assert "SET status = 'cancelled'" in query
    assert "status IN ('pending', 'processing')" in query


@pytest.mark.asyncio
async def test_retry_parent_reopens_retryable_children_before_parent(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._get_backend = AsyncMock(return_value=object())
    parent_id = uuid.uuid4()
    events: list[tuple[str, str, tuple[object, ...]]] = []

    class RetryConnection:
        def __init__(self) -> None:
            self.fetchrow_calls = 0

        def parse_json(self, value):
            return value

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query, *args):
            self.fetchrow_calls += 1
            events.append(("fetchrow", query, args))
            return {
                "bank_id": "bank",
                "status": "failed",
                "operation_type": "batch_retain",
                "result_metadata": {"is_parent": True},
            }

        async def fetch(self, query, *args):
            events.append(("fetch", query, args))
            return [
                {"operation_id": uuid.uuid4(), "status": "completed"},
                {"operation_id": uuid.uuid4(), "status": "failed"},
                {"operation_id": uuid.uuid4(), "status": "cancelled"},
            ]

        async def execute(self, query, *args):
            events.append(("execute", query, args))

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield RetryConnection()

    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", connection_scope)

    await engine.retry_operation(
        bank_id="bank",
        operation_id=str(parent_id).upper(),
        request_context=object(),
    )

    child_lock_index = next(index for index, event in enumerate(events) if event[0] == "fetch")
    parent_lock_index = next(
        index for index, event in enumerate(events) if event[0] == "fetchrow" and "FOR UPDATE" in event[1]
    )
    assert child_lock_index < parent_lock_index
    updates = [event for event in events if event[0] == "execute"]
    assert len(updates) == 2
    assert "status IN ('failed', 'cancelled')" in updates[0][1]
    assert "result_metadata->>'parent_operation_id'" in updates[0][1]
    assert updates[0][2][1] == parent_id
    assert "status IN ('failed', 'cancelled')" in updates[1][1]


@pytest.mark.asyncio
async def test_retry_child_reopens_only_failed_or_cancelled_parent(monkeypatch) -> None:
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock()
    engine._get_backend = AsyncMock(return_value=object())
    child_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    events: list[tuple[str, str, tuple[object, ...]]] = []

    class RetryConnection:
        def __init__(self) -> None:
            self.fetchrow_calls = 0

        def parse_json(self, value):
            return value

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def fetchrow(self, query, *args):
            self.fetchrow_calls += 1
            events.append(("fetchrow", query, args))
            if self.fetchrow_calls <= 2:
                return {
                    "bank_id": "bank",
                    "status": "cancelled",
                    "operation_type": "retain",
                    "result_metadata": {"parent_operation_id": str(parent_id)},
                }
            return {"status": "failed"}

        async def execute(self, query, *args):
            events.append(("execute", query, args))

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield RetryConnection()

    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", connection_scope)

    await engine.retry_operation(
        bank_id="bank",
        operation_id=str(child_id),
        request_context=object(),
    )

    updates = [event for event in events if event[0] == "execute"]
    assert len(updates) == 2
    assert str(updates[0][2][0]) == str(parent_id)
    assert "status IN ('failed', 'cancelled')" in updates[0][1]
    assert str(updates[1][2][0]) == str(child_id)
    assert "status IN ('failed', 'cancelled')" in updates[1][1]


@pytest.mark.asyncio
async def test_service_preflights_fingerprint_before_extraction(monkeypatch) -> None:
    events: list[str] = []
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=({"content": "payload", "document_id": "doc"},),
        request_context=object(),
    )
    execution = _execution()
    pipeline = service_module.RetainPipelineService()
    plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.FULL),
        recovered_unit_ids=None,
        intent=SimpleNamespace(items=()),
    )

    monkeypatch.setattr(
        service_module,
        "normalize_contents",
        lambda *_args, **_kwargs: [SimpleNamespace(document_id="doc")],
    )
    monkeypatch.setattr(service_module, "plan_documents", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(
        pipeline,
        "_recover_checkpoint",
        AsyncMock(return_value=SimpleNamespace(document_ids=(), core_committed_document_ids=())),
    )
    monkeypatch.setattr(pipeline, "_preflight_documents", AsyncMock(return_value=(plan,)))

    async def get_bank_profile(*_args, **_kwargs):
        events.append("bank")
        return {"name": "agent"}

    async def ensure_fingerprint(*_args, **kwargs):
        assert kwargs.get("for_write", False) is False
        events.append("fingerprint")

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield object()

    async def record_document_ids(*_args, **_kwargs):
        events.append("checkpoint")

    async def execute_document(*_args, **_kwargs):
        events.append("extract-and-write")
        return service_module._DocumentOutcome((), TokenUsage(), 0)

    monkeypatch.setattr(service_module.bank_utils, "get_bank_profile", get_bank_profile)
    monkeypatch.setattr(service_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(service_module, "acquire_with_retry", connection_scope)
    monkeypatch.setattr(pipeline, "_record_document_ids", record_document_ids)
    monkeypatch.setattr(pipeline, "_execute_document", execute_document)
    monkeypatch.setattr(pipeline, "_merge_document_result", lambda *_args, **_kwargs: None)

    await pipeline._retain_in_schema(invocation, execution)

    assert events == ["bank", "fingerprint", "checkpoint", "extract-and-write"]


@pytest.mark.asyncio
async def test_preflight_mismatch_stops_before_write_and_redacts_bank(monkeypatch) -> None:
    bank_id = "sensitive-bank"
    callback = AsyncMock()
    invocation = RetainInvocation(
        bank_id=bank_id,
        raw_contents=({"content": "payload", "document_id": "doc"},),
        request_context=object(),
        outbox_callback=callback,
        sanitize_log_identifiers=True,
    )
    pipeline = service_module.RetainPipelineService()
    plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.FULL),
        recovered_unit_ids=None,
    )
    record_document_ids = AsyncMock()
    execute_document = AsyncMock()

    monkeypatch.setattr(
        service_module,
        "normalize_contents",
        lambda *_args, **_kwargs: [SimpleNamespace(document_id="doc")],
    )
    monkeypatch.setattr(service_module, "plan_documents", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(
        pipeline,
        "_recover_checkpoint",
        AsyncMock(return_value=SimpleNamespace(document_ids=(), core_committed_document_ids=())),
    )
    monkeypatch.setattr(pipeline, "_preflight_documents", AsyncMock(return_value=(plan,)))
    monkeypatch.setattr(
        service_module.bank_utils,
        "get_bank_profile",
        AsyncMock(return_value={"name": "agent"}),
    )

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield object()

    async def reject_fingerprint(*_args, **kwargs):
        assert kwargs["for_write"] is False
        raise EmbeddingFingerprintMismatchError(f"Fingerprint mismatch for {bank_id}")

    monkeypatch.setattr(service_module, "acquire_with_retry", connection_scope)
    monkeypatch.setattr(service_module, "ensure_bank_embedding_fingerprint", reject_fingerprint)
    monkeypatch.setattr(pipeline, "_record_document_ids", record_document_ids)
    monkeypatch.setattr(pipeline, "_execute_document", execute_document)

    with pytest.raises(EmbeddingFingerprintMismatchError) as error:
        await pipeline._retain_in_schema(invocation, _execution())

    assert bank_id not in str(error.value)
    assert "<redacted>" in str(error.value)
    record_document_ids.assert_not_awaited()
    execute_document.assert_not_awaited()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_stale_publication_fence_aborts_without_callback(monkeypatch) -> None:
    bank_id = "sensitive-bank"
    document_id = "sensitive-document"
    operation_id = "sensitive-operation"
    callback = AsyncMock()
    invocation = RetainInvocation(
        bank_id=bank_id,
        raw_contents=({"content": "payload", "document_id": document_id},),
        request_context=object(),
        operation_id=operation_id,
        outbox_callback=callback,
        sanitize_log_identifiers=True,
    )
    pipeline = service_module.RetainPipelineService()
    stale_plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.STALE_SKIP),
        recovered_unit_ids=None,
    )

    monkeypatch.setattr(
        service_module,
        "normalize_contents",
        lambda *_args, **_kwargs: [SimpleNamespace(document_id=document_id)],
    )
    monkeypatch.setattr(service_module, "plan_documents", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(
        pipeline,
        "_recover_checkpoint",
        AsyncMock(return_value=SimpleNamespace(document_ids=(), core_committed_document_ids=())),
    )
    monkeypatch.setattr(pipeline, "_preflight_documents", AsyncMock(return_value=(stale_plan,)))

    with pytest.raises(RetainPublicationAborted, match="superseded before publication") as error:
        await pipeline._retain_in_schema(invocation, _execution())

    for identifier in (bank_id, document_id, operation_id):
        assert identifier not in str(error.value)
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_stale_publication_fence_aborts_before_partial_publish(monkeypatch) -> None:
    callback = AsyncMock()
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=(
            {"content": "current payload", "document_id": "current-doc"},
            {"content": "superseded payload", "document_id": "stale-doc"},
        ),
        request_context=object(),
        outbox_callback=callback,
    )
    pipeline = service_module.RetainPipelineService()
    current_plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.FULL),
        recovered_unit_ids=None,
    )
    stale_plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.STALE_SKIP),
        recovered_unit_ids=None,
    )
    execute_document = AsyncMock()

    monkeypatch.setattr(
        service_module,
        "normalize_contents",
        lambda *_args, **_kwargs: [
            SimpleNamespace(document_id="current-doc"),
            SimpleNamespace(document_id="stale-doc"),
        ],
    )
    monkeypatch.setattr(service_module, "plan_documents", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(
        pipeline,
        "_recover_checkpoint",
        AsyncMock(return_value=SimpleNamespace(document_ids=(), core_committed_document_ids=())),
    )
    monkeypatch.setattr(
        pipeline,
        "_preflight_documents",
        AsyncMock(return_value=(current_plan, stale_plan)),
    )
    monkeypatch.setattr(pipeline, "_execute_document", execute_document)

    with pytest.raises(RetainPublicationAborted, match="superseded before publication"):
        await pipeline._retain_in_schema(invocation, _execution())

    execute_document.assert_not_awaited()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_stale_without_publication_fence_remains_a_noop(monkeypatch) -> None:
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=({"content": "payload", "document_id": "doc"},),
        request_context=object(),
    )
    pipeline = service_module.RetainPipelineService()
    stale_plan = SimpleNamespace(
        change=SimpleNamespace(kind=DocumentChangeKind.STALE_SKIP),
        recovered_unit_ids=None,
        intent=SimpleNamespace(items=(object(),)),
    )
    execute_document = AsyncMock()

    monkeypatch.setattr(
        service_module,
        "normalize_contents",
        lambda *_args, **_kwargs: [SimpleNamespace(document_id="doc")],
    )
    monkeypatch.setattr(service_module, "plan_documents", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(
        pipeline,
        "_recover_checkpoint",
        AsyncMock(return_value=SimpleNamespace(document_ids=(), core_committed_document_ids=())),
    )
    monkeypatch.setattr(pipeline, "_preflight_documents", AsyncMock(return_value=(stale_plan,)))
    monkeypatch.setattr(
        service_module.bank_utils,
        "get_bank_profile",
        AsyncMock(return_value={"name": "agent"}),
    )

    @asynccontextmanager
    async def connection_scope(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(service_module, "acquire_with_retry", connection_scope)
    monkeypatch.setattr(service_module, "ensure_bank_embedding_fingerprint", AsyncMock())
    monkeypatch.setattr(pipeline, "_record_document_ids", AsyncMock())
    monkeypatch.setattr(pipeline, "_execute_document", execute_document)
    monkeypatch.setattr(pipeline, "_merge_document_result", lambda *_args, **_kwargs: None)

    outcome = await pipeline._retain_in_schema(invocation, _execution())

    assert outcome.unit_ids_by_input == [[]]
    execute_document.assert_not_awaited()


class _OperationActivity:
    def __init__(self, events: list[str], *, active: bool = True) -> None:
        self._events = events
        self._active = active

    async def assert_active(self, connection, *, bank_id: str) -> None:
        assert connection.in_transaction
        assert bank_id == "bank"
        self._events.append("operation")
        if not self._active:
            raise RetainOperationInactiveError("inactive")


class _FenceConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = list(rows)
        self.queries: list[str] = []
        self.args: list[tuple[object, ...]] = []

    async def fetchrow(self, query: str, *args):
        self.queries.append(query)
        self.args.append(args)
        return self.rows.pop(0)

    @staticmethod
    def parse_json(value):
        return value


@pytest.mark.asyncio
async def test_operation_activity_fence_locks_active_child_then_parent() -> None:
    child_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    connection = _FenceConnection(
        [
            {
                "status": "processing",
                "result_metadata": {"parent_operation_id": str(parent_id)},
            },
            {"status": "pending"},
        ]
    )

    await OperationActivityFence(str(child_id), schema="tenant").assert_active(
        connection,
        bank_id="bank",
    )

    assert len(connection.queries) == 2
    assert all("FOR UPDATE" in query for query in connection.queries)
    assert '"tenant".async_operations' in connection.queries[0]
    assert connection.args == [
        (child_id, "bank"),
        (parent_id, "bank"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"status": "cancelled", "result_metadata": {}}], "operation is no longer active"),
        (
            [
                {
                    "status": "processing",
                    "result_metadata": {"parent_operation_id": str(uuid.uuid4())},
                },
                {"status": "cancelled"},
            ],
            "parent operation is no longer active",
        ),
    ],
)
async def test_operation_activity_fence_rejects_terminal_child_or_parent(rows, match) -> None:
    connection = _FenceConnection(rows)

    with pytest.raises(RetainOperationInactiveError, match=match):
        await OperationActivityFence(str(uuid.uuid4())).assert_active(
            connection,
            bank_id="bank",
        )

    assert all("FOR UPDATE" in query for query in connection.queries)


@pytest.mark.asyncio
async def test_inactive_operation_rolls_back_before_fingerprint_or_any_core_write(monkeypatch) -> None:
    events: list[str] = []
    writer = _writer(
        events,
        operation_activity=_OperationActivity(events, active=False),
    )
    connection = _Connection(events)
    fingerprint = AsyncMock(side_effect=AssertionError("fingerprint must not run"))
    callback = AsyncMock()
    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", fingerprint)

    with pytest.raises(RetainOperationInactiveError, match="inactive"):
        await _unit_of_work(writer, connection).execute(
            MetadataOnlyWriteRequest(
                bank_id="bank",
                document_id="doc",
                expected_content_hash="hash",
                combined_content="payload",
                input_slot_count=1,
                outbox_callback=callback,
            ),
        )

    assert events == ["begin", "operation", "rollback"]
    fingerprint.assert_not_awaited()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_unhashed_existing_document_uses_locked_full_replacement(monkeypatch) -> None:
    events: list[str] = []
    writer = _writer(events)
    connection = _Connection(events)

    async def ensure_fingerprint(*_args, **_kwargs):
        events.append("fingerprint")

    async def track_document(*_args, **_kwargs):
        events.append("document")

    async def insert_facts(*_args, **_kwargs):
        events.append("facts")
        return [[]], object()

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(writer_module.fact_storage, "handle_document_tracking", track_document)
    monkeypatch.setattr(writer_module.runtime, "insert_facts_and_links", insert_facts)
    monkeypatch.setattr(writer, "_finalize_entity_graph", AsyncMock(return_value=CoreGraphWrite()))

    result = await _unit_of_work(writer, connection).execute(
        WriteWindowRequest(
            bank_id="bank",
            document_id="doc",
            document_window=FirstFullWriteWindow(
                combined_content="payload",
                expects_unhashed_existing_document=True,
            ),
            contents=(RetainContent(content="payload"),),
        ),
    )

    assert result.core.ownership.value == "owned"
    assert events == [
        "begin",
        "fingerprint",
        "unhashed-ownership",
        "document",
        "facts",
        "commit",
    ]


@pytest.mark.asyncio
async def test_unhashed_existing_document_loses_ownership_before_replacement(monkeypatch) -> None:
    events: list[str] = []
    writer = _writer(events, owns=False)
    connection = _Connection(events)
    track_document = AsyncMock(side_effect=AssertionError("replacement must not run"))
    callback = AsyncMock()

    async def ensure_fingerprint(*_args, **_kwargs):
        events.append("fingerprint")

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(writer_module.fact_storage, "handle_document_tracking", track_document)

    result = await _unit_of_work(writer, connection).execute(
        WriteWindowRequest(
            bank_id="bank",
            document_id="doc",
            document_window=FirstFullWriteWindow(
                combined_content="payload",
                expects_unhashed_existing_document=True,
            ),
            contents=(RetainContent(content="payload"),),
            outbox_callback=callback,
        ),
    )

    assert result.core.ownership.value == "lost"
    assert events == ["begin", "fingerprint", "unhashed-ownership", "commit"]
    track_document.assert_not_awaited()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_transaction_guard_covers_zero_fact_full_write(monkeypatch) -> None:
    events: list[str] = []
    writer = _writer(events)
    connection = _Connection(events)

    async def ensure_fingerprint(*args, **kwargs):
        assert args[0] is connection
        assert args[1] == "bank"
        assert connection.in_transaction
        assert kwargs["for_write"] is True
        events.append("fingerprint")

    async def track_document(*_args, **_kwargs):
        events.append("document")

    async def insert_facts(*_args, **kwargs):
        events.append("facts")
        await kwargs["outbox_callback"](connection)
        return [[]], object()

    async def callback(callback_connection):
        assert callback_connection is connection
        events.append("outbox")

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(writer_module.fact_storage, "handle_document_tracking", track_document)
    monkeypatch.setattr(writer_module.runtime, "insert_facts_and_links", insert_facts)
    monkeypatch.setattr(writer, "_finalize_entity_graph", AsyncMock(return_value=CoreGraphWrite()))

    result = await _unit_of_work(writer, connection).execute(
        WriteWindowRequest(
            bank_id="bank",
            document_id="doc",
            document_window=FirstFullWriteWindow(combined_content="payload"),
            contents=(RetainContent(content="payload"),),
            outbox_callback=callback,
        ),
    )

    assert result.core.unit_ids_by_content == ((),)
    assert result.core.post_commit_required is False
    assert events == ["begin", "fingerprint", "ownership", "document", "facts", "outbox", "commit"]


@pytest.mark.asyncio
async def test_transaction_guard_covers_metadata_only_write(monkeypatch) -> None:
    events: list[str] = []
    writer = _writer(events)
    connection = _Connection(events)

    async def ensure_fingerprint(*_args, **kwargs):
        assert connection.in_transaction
        assert kwargs["for_write"] is True
        events.append("fingerprint")

    async def update_metadata(*_args, **_kwargs):
        events.append("metadata")

    async def update_tags(*_args, **_kwargs):
        events.append("tags")

    async def callback(_connection):
        events.append("outbox")

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(writer_module.fact_storage, "upsert_document_metadata", update_metadata)
    monkeypatch.setattr(writer_module.fact_storage, "update_memory_units_tags", update_tags)

    result = await _unit_of_work(writer, connection).execute(
        MetadataOnlyWriteRequest(
            bank_id="bank",
            document_id="doc",
            expected_content_hash="hash",
            combined_content="payload",
            input_slot_count=1,
            outbox_callback=callback,
        ),
    )

    assert result.core.unit_ids_by_content == ((),)
    assert events == ["begin", "fingerprint", "ownership", "metadata", "tags", "outbox", "commit"]


@pytest.mark.asyncio
async def test_fingerprint_failure_prevents_mutation_and_redacts_identifier(monkeypatch) -> None:
    bank_id = "sensitive-bank"
    document_id = "sensitive-document"
    events: list[str] = []
    writer = _writer(events, sanitize=True)
    connection = _Connection(events)
    callback = AsyncMock()

    async def reject_fingerprint(*_args, **_kwargs):
        assert connection.in_transaction
        events.append("fingerprint")
        raise EmbeddingFingerprintMismatchError(f"Embedding fingerprint mismatch for bank {bank_id!r}")

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", reject_fingerprint)
    monkeypatch.setattr(
        writer_module.fact_storage,
        "upsert_document_metadata",
        AsyncMock(side_effect=AssertionError("mutation must not run")),
    )

    with pytest.raises(EmbeddingFingerprintMismatchError) as error:
        await _unit_of_work(writer, connection).execute(
            MetadataOnlyWriteRequest(
                bank_id=bank_id,
                document_id=document_id,
                expected_content_hash="hash",
                combined_content="payload",
                input_slot_count=1,
                outbox_callback=callback,
            ),
        )

    assert bank_id not in str(error.value)
    assert "<redacted>" in str(error.value)
    assert events == ["begin", "fingerprint", "rollback"]
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_sanitized_fingerprint_warning_hides_bank_identifier(monkeypatch, caplog) -> None:
    bank_id = "sensitive-multimodal-bank"
    stored = fingerprint_module.build_embedding_fingerprint(_embedding_model())
    current_model = SimpleNamespace(
        provider_name="openai",
        model="text-embedding-3-small",
        dimension=384,
        normalization=True,
    )
    monkeypatch.setattr(
        fingerprint_module,
        "_bank_state",
        AsyncMock(return_value=(stored, True)),
    )
    sanitizer = IdentifierSanitizer.from_values(enabled=True, values=(bank_id,))
    caplog.set_level(logging.WARNING, logger=fingerprint_module.__name__)

    await fingerprint_module.ensure_bank_embedding_fingerprint(
        object(),
        bank_id,
        current_model,
        policy="warn",
        log_sanitizer=sanitizer,
    )

    assert "<redacted>" in caplog.text
    assert bank_id not in caplog.text


@pytest.mark.asyncio
async def test_sanitized_full_replacement_hides_bank_and_document_logs(monkeypatch, caplog) -> None:
    bank_id = "sensitive-multimodal-bank"
    document_id = "sensitive-multimodal-document"
    source_id = uuid.uuid4()
    observation_id = uuid.uuid4()
    events: list[str] = []

    class ReplacementConnection(_Connection):
        async def fetch(self, query, *_args):
            if "unit_entities" in query:
                return []
            if "fact_type IN ('experience', 'world')" in query:
                return [{"id": source_id}]
            if "fact_type = 'observation'" in query:
                return [{"id": observation_id, "source_memory_ids": []}]
            raise AssertionError(f"Unexpected fetch query: {query}")

        async def execute(self, _query, *_args):
            return "DELETE 1"

        async def fetchval(self, _query, *_args):
            return None

    connection = ReplacementConnection(events)
    ops = SimpleNamespace(
        uses_observation_sources_table=False,
        refresh_entity_fact_counts=AsyncMock(),
    )
    writer = PersistenceWriter(
        pool=SimpleNamespace(ops=ops),
        embeddings_model=_embedding_model(),
        entity_resolver=object(),
        config=_config(),
        ownership=_Ownership(events),
        sanitize_log_identifiers=True,
    )

    async def ensure_fingerprint(*_args, **kwargs):
        assert kwargs["log_sanitizer"].identifier(bank_id) == "<redacted>"

    async def insert_facts(*_args, **_kwargs):
        return [[]], object()

    monkeypatch.setattr(writer_module, "ensure_bank_embedding_fingerprint", ensure_fingerprint)
    monkeypatch.setattr(fact_storage, "_upsert_document_row", AsyncMock())
    monkeypatch.setattr(writer_module.runtime, "insert_facts_and_links", insert_facts)
    monkeypatch.setattr(writer, "_finalize_entity_graph", AsyncMock(return_value=CoreGraphWrite()))
    caplog.set_level(logging.INFO, logger=fact_storage.__name__)

    await _unit_of_work(writer, connection).execute(
        WriteWindowRequest(
            bank_id=bank_id,
            document_id=document_id,
            document_window=FirstFullWriteWindow(combined_content="canonical multimodal payload"),
            contents=(RetainContent(content="canonical multimodal payload"),),
        ),
    )

    assert "<redacted>" in caplog.text
    assert bank_id not in caplog.text
    assert document_id not in caplog.text


@pytest.mark.asyncio
async def test_sanitized_warning_hides_identifiers_and_exception_text(
    monkeypatch,
    caplog,
) -> None:
    bank_id = "sensitive-bank"
    operation_id = "sensitive-operation"
    document_id = "sensitive-document"
    invocation = RetainInvocation(
        bank_id=bank_id,
        raw_contents=({"content": "payload"},),
        request_context=object(),
        operation_id=operation_id,
        sanitize_log_identifiers=True,
    )

    @asynccontextmanager
    async def failing_scope(*_args, **_kwargs):
        raise RuntimeError(f"{bank_id}/{operation_id}/{document_id}")
        yield  # pragma: no cover

    monkeypatch.setattr(service_module, "acquire_with_retry", failing_scope)
    caplog.set_level(logging.WARNING, logger=service_module.__name__)

    await service_module.RetainPipelineService()._record_document_ids(
        invocation,
        _execution(),
        (SimpleNamespace(document_id=document_id),),
    )

    assert "<redacted>" in caplog.text
    assert bank_id not in caplog.text
    assert operation_id not in caplog.text
    assert document_id not in caplog.text
    assert caplog.records[-1].exc_info is None


@pytest.mark.asyncio
async def test_final_ann_failure_is_reported_as_incomplete(monkeypatch) -> None:
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=({"content": "payload"},),
        request_context=object(),
    )
    plan = SimpleNamespace(intent=SimpleNamespace(document_id="doc"))
    monkeypatch.setattr(
        service_module,
        "run_final_semantic_ann",
        AsyncMock(side_effect=RuntimeError("temporary ANN failure")),
    )

    completed = await service_module.RetainPipelineService()._run_full_semantic_ann_best_effort(
        invocation,
        _execution(),
        plan,
        ["unit-a"],
    )

    assert completed is False


@pytest.mark.asyncio
async def test_recovery_preserves_final_ann_retry_marker_after_failure(monkeypatch) -> None:
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=({"content": "payload"},),
        request_context=object(),
        operation_id="operation",
    )
    plan = SimpleNamespace(
        recovered_unit_bindings=(SimpleNamespace(unit_id="unit-a"),),
        final_ann_pending=True,
        intent=SimpleNamespace(document_id="doc"),
    )
    pipeline = service_module.RetainPipelineService()
    run_final_ann = AsyncMock(return_value=False)
    record_completed = AsyncMock()
    monkeypatch.setattr(pipeline, "_recovery_result_buckets", lambda _plan: (("unit-a",),))
    monkeypatch.setattr(pipeline, "_run_full_semantic_ann_best_effort", run_final_ann)
    monkeypatch.setattr(pipeline, "_record_final_ann_completed_best_effort", record_completed)

    outcome = await pipeline._resume_committed_document(invocation, _execution(), plan)

    assert outcome.unit_ids_by_content == (("unit-a",),)
    run_final_ann.assert_awaited_once()
    record_completed.assert_not_awaited()
