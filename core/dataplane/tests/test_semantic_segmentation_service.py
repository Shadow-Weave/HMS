"""Focused service contracts for Retain semantic segmentation integration."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from hms_api.engine.ingestion import service as service_module
from hms_api.engine.ingestion.chunking import build_chunk_plans
from hms_api.engine.ingestion.contracts import RetainExecutionContext, RetainInvocation
from hms_api.engine.ingestion.document_planner import plan_documents, prepend_existing_document
from hms_api.engine.ingestion.domain import (
    ChunkPolicy,
    DocumentChangeKind,
    ExistingChunkFingerprint,
)
from hms_api.engine.ingestion.normalization import normalize_contents
from hms_api.engine.ingestion.persistence.models import (
    CommittedUnitBinding,
    ExistingDocument,
    OperationCheckpoint,
)
from hms_api.engine.ingestion.segmentation import BoundaryResponse
from hms_api.engine.response_models import TokenUsage


def _conversation(*, changed: bool = False) -> str:
    turns: list[dict[str, str]] = []
    for index, topic in enumerate(("basketball", "travel", "music")):
        answer_suffix = " changed" if changed and topic == "music" else ""
        turns.extend(
            (
                {
                    "role": "user",
                    "content": f"{topic} question {index} " + ("q" * 40),
                },
                {
                    "role": "assistant",
                    "content": f"{topic} answer {index}{answer_suffix} " + ("a" * 40),
                },
            )
        )
    return json.dumps(turns, ensure_ascii=False, separators=(",", ":"))


class _BoundaryLLM:
    provider = "mock"
    model = "boundary-model"

    def __init__(
        self,
        boundaries: list[int] | None = None,
        *,
        usage: TokenUsage | None = None,
        snapshot_state: dict[str, bool] | None = None,
        events: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.boundaries = boundaries
        self.usage = usage or TokenUsage()
        self.snapshot_state = snapshot_state
        self.events = events
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def call(self, **kwargs: Any) -> tuple[BoundaryResponse, TokenUsage]:
        self.calls.append(kwargs)
        if self.snapshot_state is not None:
            assert self.snapshot_state["open"] is False
        if self.events is not None:
            self.events.append("llm")
        if self.error is not None:
            raise self.error
        assert self.boundaries is not None
        return BoundaryResponse(end_exchange_indices=self.boundaries), self.usage


class _PlanningRepository:
    def __init__(
        self,
        *,
        snapshot_state: dict[str, bool],
        events: list[str],
        existing: ExistingDocument | None = None,
        chunks: tuple[ExistingChunkFingerprint, ...] = (),
        bindings: tuple[CommittedUnitBinding, ...] | None = None,
    ) -> None:
        self.snapshot_state = snapshot_state
        self.events = events
        self.existing = existing
        self.chunks = chunks
        self.bindings = bindings

    async def load_document(self, bank_id: str, document_id: str) -> ExistingDocument | None:
        assert self.snapshot_state["open"] is True
        assert bank_id == "bank"
        assert document_id == "document"
        self.events.append("load-document")
        return self.existing

    async def load_chunks(
        self,
        bank_id: str,
        document_id: str,
    ) -> tuple[ExistingChunkFingerprint, ...]:
        assert self.snapshot_state["open"] is True
        assert bank_id == "bank"
        assert document_id == "document"
        self.events.append("load-chunks")
        return self.chunks

    async def load_document_unit_bindings(
        self,
        bank_id: str,
        document_id: str,
        *,
        expected_unit_ids: tuple[str, ...] | None,
    ) -> tuple[CommittedUnitBinding, ...]:
        if self.bindings is None:
            raise AssertionError("uncommitted test documents must not load unit bindings")
        assert bank_id == "bank"
        assert document_id == "document"
        if expected_unit_ids is not None:
            assert tuple(binding.unit_id for binding in self.bindings) == expected_unit_ids
        self.events.append("load-bindings")
        return self.bindings


class _BackendAdapters:
    def __init__(
        self,
        *,
        repository: _PlanningRepository,
        snapshot_state: dict[str, bool],
        events: list[str],
    ) -> None:
        self.repository = repository
        self.snapshot_state = snapshot_state
        self.events = events

    def planning_repository(self, _connection: Any, *, schema: str | None = None) -> _PlanningRepository:
        assert schema is None
        assert self.snapshot_state["open"] is True
        return self.repository

    @asynccontextmanager
    async def planning_snapshot(self, _connection: Any):
        assert self.snapshot_state["open"] is False
        self.snapshot_state["open"] = True
        self.events.append("snapshot-enter")
        try:
            yield
        finally:
            self.snapshot_state["open"] = False
            self.events.append("snapshot-exit")


def _execution(
    llm: _BoundaryLLM,
    *,
    extraction_mode: str = "concise",
    semantic_enabled: bool | None = True,
) -> RetainExecutionContext:
    config_values = dict(
        database_backend="postgresql",
        retain_extraction_mode=extraction_mode,
        retain_chunk_size=380,
        retain_semantic_chunking_failure_policy="raise",
        retain_semantic_chunking_max_completion_tokens=128,
        retain_semantic_chunking_max_retries=0,
        retain_llm_max_concurrent=2,
    )
    if semantic_enabled is not None:
        config_values["retain_semantic_chunking_enabled"] = semantic_enabled
    config = SimpleNamespace(**config_values)
    return RetainExecutionContext(
        pool=SimpleNamespace(backend_type="postgresql"),
        embeddings_model=object(),
        llm_config=llm,
        entity_resolver=object(),
        format_date_fn=lambda *_args, **_kwargs: "",
        resolved_config=config,
    )


def _invocation(
    text: str,
    *,
    trusted_prechunked_input: bool = False,
    update_mode: str | None = None,
) -> RetainInvocation:
    content: dict[str, Any] = {"content": text, "document_id": "document"}
    if update_mode is not None:
        content["update_mode"] = update_mode
    return RetainInvocation(
        bank_id="bank",
        raw_contents=(content,),
        request_context=object(),
        trusted_prechunked_input=trusted_prechunked_input,
    )


def _intent(text: str, *, update_mode: str | None = None):
    content: dict[str, Any] = {"content": text, "document_id": "document"}
    if update_mode is not None:
        content["update_mode"] = update_mode
    items = normalize_contents((content,))
    return plan_documents(items)[0]


def _policy() -> ChunkPolicy:
    return ChunkPolicy(
        version="retain-chunker-v1",
        max_chars=380,
        conversation_mode=True,
        overlap=0,
    )


def _install_planning_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: ExistingDocument | None = None,
    chunks: tuple[ExistingChunkFingerprint, ...] = (),
    bindings: tuple[CommittedUnitBinding, ...] | None = None,
) -> tuple[dict[str, bool], list[str]]:
    snapshot_state = {"open": False}
    events: list[str] = []
    repository = _PlanningRepository(
        snapshot_state=snapshot_state,
        events=events,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    adapters = _BackendAdapters(
        repository=repository,
        snapshot_state=snapshot_state,
        events=events,
    )

    @asynccontextmanager
    async def connection_scope(*_args: Any, **_kwargs: Any):
        yield object()

    monkeypatch.setattr(service_module, "_backend_adapters", lambda _execution: adapters)
    monkeypatch.setattr(service_module, "acquire_with_retry", connection_scope)
    return snapshot_state, events


async def _preflight(
    *,
    invocation: RetainInvocation,
    execution: RetainExecutionContext,
    text: str,
    request_started_at: datetime,
    update_mode: str | None = None,
    checkpoint: OperationCheckpoint | None = None,
):
    plans = await service_module.RetainPipelineService()._preflight_documents(
        invocation,
        execution,
        (_intent(text, update_mode=update_mode),),
        _policy(),
        checkpoint=checkpoint or OperationCheckpoint(),
        request_started_at=request_started_at,
    )
    assert len(plans) == 1
    return plans[0]


def _persisted_document(
    plan: Any,
    *,
    updated_at: datetime,
) -> tuple[ExistingDocument, tuple[ExistingChunkFingerprint, ...]]:
    metadata_key = service_module._SEMANTIC_PLAN_METADATA_KEY
    document = ExistingDocument(
        document_id="document",
        bank_id="bank",
        original_text=plan.combined_content,
        content_hash=service_module.compute_document_hash(plan.combined_content),
        retain_params={metadata_key: plan.segmentation_metadata},
        tags=(),
        created_at=updated_at,
        updated_at=updated_at,
    )
    chunks = tuple(
        ExistingChunkFingerprint(
            chunk_id=f"stored-chunk-{chunk.global_index}",
            chunk_index=chunk.global_index,
            content_hash=chunk.content_hash,
        )
        for chunk in plan.chunks
    )
    return document, chunks


def _fixed_persisted_document(
    texts: tuple[str, ...],
    *,
    updated_at: datetime,
) -> tuple[
    RetainInvocation,
    Any,
    ExistingDocument,
    tuple[ExistingChunkFingerprint, ...],
]:
    raw_contents = tuple(
        {
            "content": text,
            "document_id": "document",
        }
        for text in texts
    )
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=raw_contents,
        request_context=object(),
    )
    intent = plan_documents(normalize_contents(raw_contents))[0]
    chunks = build_chunk_plans(intent.document_id, intent.items, _policy())
    combined_content = "\n".join(item.content for item in intent.items)
    document = ExistingDocument(
        document_id="document",
        bank_id="bank",
        original_text=combined_content,
        content_hash=service_module.compute_document_hash(combined_content),
        retain_params={},
        tags=(),
        created_at=updated_at,
        updated_at=updated_at,
    )
    durable_chunks = tuple(
        ExistingChunkFingerprint(
            chunk_id=f"stored-chunk-{chunk.global_index}",
            chunk_index=chunk.global_index,
            content_hash=chunk.content_hash,
        )
        for chunk in chunks
    )
    return invocation, intent, document, durable_chunks


@pytest.mark.asyncio
async def test_planning_failure_cancels_and_awaits_sibling_work() -> None:
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def blocked_sibling() -> int:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_finished.set()
        return 1

    async def failing_task() -> int:
        await sibling_started.wait()
        raise RuntimeError("boundary planning failed")

    with pytest.raises(RuntimeError, match="boundary planning failed"):
        await service_module._gather_planning_tasks(
            (
                blocked_sibling(),
                failing_task(),
            )
        )

    assert sibling_finished.is_set()
    assert all(task.done() for task in asyncio.all_tasks() if task is not asyncio.current_task())


@pytest.mark.asyncio
async def test_semantic_preflight_plans_after_snapshot_and_materializes_complete_exchanges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    snapshot_state, events = _install_planning_backend(monkeypatch)
    usage = TokenUsage(input_tokens=31, output_tokens=3, total_tokens=34)
    llm = _BoundaryLLM(
        [0, 1, 2],
        usage=usage,
        snapshot_state=snapshot_state,
        events=events,
    )

    plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(llm),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert events == ["snapshot-enter", "load-document", "snapshot-exit", "llm"]
    assert snapshot_state["open"] is False
    assert plan.change.kind is DocumentChangeKind.FULL
    assert plan.change.reason == "document_not_found"
    assert plan.segmentation_usage == usage
    assert len(plan.chunks) == 3
    assert [turn for chunk in plan.chunks for turn in json.loads(chunk.text)] == json.loads(text)
    manifest = plan.segmentation_metadata["items"][0]["manifest"]
    assert manifest["effective_strategy"] == "semantic"
    assert manifest["end_exchange_indices"] == [0, 1, 2]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_identical_document_reuses_durable_manifest_without_an_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    first_state, first_events = _install_planning_backend(monkeypatch)
    planning_llm = _BoundaryLLM(
        [0, 1, 2],
        snapshot_state=first_state,
        events=first_events,
    )
    first_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(planning_llm),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        first_plan,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    _install_planning_backend(monkeypatch, existing=existing, chunks=chunks)
    reuse_llm = _BoundaryLLM(error=AssertionError("manifest reuse must not call the provider"))
    reused_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(reuse_llm),
        text=text,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert reuse_llm.calls == []
    assert reused_plan.change.kind is DocumentChangeKind.METADATA_ONLY
    assert reused_plan.change.reason is None
    assert reused_plan.segmentation_usage == TokenUsage()
    assert reused_plan.chunks == first_plan.chunks
    assert reused_plan.segmentation_metadata["plan_digest"] == first_plan.segmentation_metadata["plan_digest"]


@pytest.mark.asyncio
async def test_missing_semantic_flag_uses_enabled_runtime_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    snapshot_state, events = _install_planning_backend(monkeypatch)
    llm = _BoundaryLLM(
        [0, 2],
        snapshot_state=snapshot_state,
        events=events,
    )

    plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(llm, semantic_enabled=None),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert len(llm.calls) == 1
    assert plan.segmentation_metadata is not None
    assert plan.segmentation_metadata["items"][0]["manifest"]["effective_strategy"] == "semantic"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trusted_prechunked_input", "extraction_mode", "semantic_enabled", "provider"),
    (
        (True, "concise", True, "mock"),
        (False, "chunks", True, "mock"),
        (False, "concise", False, "mock"),
        (False, "concise", True, "none"),
    ),
)
async def test_non_semantic_routes_preserve_the_fixed_chunker(
    monkeypatch: pytest.MonkeyPatch,
    trusted_prechunked_input: bool,
    extraction_mode: str,
    semantic_enabled: bool,
    provider: str,
) -> None:
    text = _conversation()
    _install_planning_backend(monkeypatch)
    llm = _BoundaryLLM(error=AssertionError("bypass paths must not call the provider"))
    llm.provider = provider
    intent = _intent(text)

    plan = await _preflight(
        invocation=_invocation(
            text,
            trusted_prechunked_input=trusted_prechunked_input,
        ),
        execution=_execution(
            llm,
            extraction_mode=extraction_mode,
            semantic_enabled=semantic_enabled,
        ),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert llm.calls == []
    assert plan.chunks == build_chunk_plans(intent.document_id, intent.items, _policy())
    assert plan.segmentation_metadata is None
    assert plan.segmentation_usage == TokenUsage()
    assert plan.change.kind is DocumentChangeKind.FULL


@pytest.mark.asyncio
async def test_all_committed_batch_retry_reaches_provider_free_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReachedPreflight(RuntimeError):
        pass

    text = _conversation()
    execution = _execution(_BoundaryLLM(error=AssertionError("committed retry must not call the provider")))
    execution.resolved_config.retain_batch_enabled = True
    checkpoint = OperationCheckpoint(
        document_ids=("document",),
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit",)),),
    )
    service = service_module.RetainPipelineService()

    async def recover_checkpoint(*_args: Any, **_kwargs: Any) -> OperationCheckpoint:
        return checkpoint

    async def reach_preflight(*_args: Any, **_kwargs: Any) -> None:
        raise _ReachedPreflight

    monkeypatch.setattr(service, "_recover_checkpoint", recover_checkpoint)
    monkeypatch.setattr(service, "_preflight_documents", reach_preflight)

    with pytest.raises(_ReachedPreflight):
        await service._retain_in_schema(_invocation(text), execution)


@pytest.mark.asyncio
async def test_uncommitted_semantic_batch_request_remains_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    execution = _execution(_BoundaryLLM([0, 1, 2]))
    execution.resolved_config.retain_batch_enabled = True
    service = service_module.RetainPipelineService()

    async def recover_checkpoint(*_args: Any, **_kwargs: Any) -> OperationCheckpoint:
        return OperationCheckpoint()

    monkeypatch.setattr(service, "_recover_checkpoint", recover_checkpoint)

    with pytest.raises(service_module.RetainUnsupportedError, match="provider Batch extraction"):
        await service._retain_in_schema(_invocation(text), execution)


@pytest.mark.asyncio
async def test_changed_semantic_boundaries_force_a_conservative_full_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_text = _conversation()
    first_state, first_events = _install_planning_backend(monkeypatch)
    first_llm = _BoundaryLLM(
        [0, 2],
        snapshot_state=first_state,
        events=first_events,
    )
    first_plan = await _preflight(
        invocation=_invocation(original_text),
        execution=_execution(first_llm),
        text=original_text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        first_plan,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    changed_text = _conversation(changed=True)
    changed_state, changed_events = _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
    )
    changed_llm = _BoundaryLLM(
        [1, 2],
        snapshot_state=changed_state,
        events=changed_events,
    )
    changed_plan = await _preflight(
        invocation=_invocation(changed_text),
        execution=_execution(changed_llm),
        text=changed_text,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert len(changed_llm.calls) == 1
    assert changed_plan.change.kind is DocumentChangeKind.FULL
    assert changed_plan.change.reason == "chunk_policy_incompatible"
    first_manifest = first_plan.segmentation_metadata["items"][0]["manifest"]
    changed_manifest = changed_plan.segmentation_metadata["items"][0]["manifest"]
    assert first_manifest["end_exchange_indices"] == [0, 2]
    assert changed_manifest["end_exchange_indices"] == [1, 2]
    assert changed_plan.segmentation_metadata["plan_digest"] != first_plan.segmentation_metadata["plan_digest"]


@pytest.mark.asyncio
async def test_semantic_append_replans_combined_document_without_losing_input_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_text = _conversation()
    original_state, original_events = _install_planning_backend(monkeypatch)
    original_plan = await _preflight(
        invocation=_invocation(original_text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=original_state,
                events=original_events,
            )
        ),
        text=original_text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        original_plan,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    appended_text = _conversation(changed=True)
    append_state, append_events = _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
    )
    usage = TokenUsage(input_tokens=13, output_tokens=2, total_tokens=15)
    append_llm = _BoundaryLLM(
        [0, 1, 2],
        usage=usage,
        snapshot_state=append_state,
        events=append_events,
    )
    append_plan = await _preflight(
        invocation=_invocation(appended_text, update_mode="append"),
        execution=_execution(append_llm),
        text=appended_text,
        update_mode="append",
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert len(append_llm.calls) == 2
    assert append_plan.change.kind is DocumentChangeKind.FULL
    assert append_plan.change.reason == "chunk_policy_incompatible"
    assert append_plan.combined_content == f"{original_text}\n{appended_text}"
    assert len(append_plan.intent.items) == 2
    assert append_plan.intent.items[0].source_index is None
    assert append_plan.intent.items[1].source_index == 0
    assert len(append_plan.segmentation_metadata["items"]) == 2
    assert append_plan.segmentation_usage == usage + usage


@pytest.mark.asyncio
async def test_committed_fixed_recovery_uses_durable_layout_after_semantic_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = (_conversation(), _conversation(changed=True))
    invocation, intent, existing, chunks = _fixed_persisted_document(
        texts,
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    second_source_chunk = next(
        chunk for chunk in build_chunk_plans(intent.document_id, intent.items, _policy()) if chunk.source_index == 1
    )
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-from-second-input",
            chunk_index=second_source_chunk.global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("committed fixed-layout recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-from-second-input",)),),
    )

    plans = await service_module.RetainPipelineService()._preflight_documents(
        invocation,
        _execution(retry_llm),
        (intent,),
        _policy(),
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert len(plans) == 1
    plan = plans[0]
    assert plan.change.kind is DocumentChangeKind.METADATA_ONLY
    assert plan.change.reason == "operation core commit recovered"
    assert plan.recovered_chunk_sources is not None
    assert service_module.RetainPipelineService._recovery_result_buckets(plan) == (
        (),
        ("unit-from-second-input",),
    )


@pytest.mark.asyncio
async def test_committed_fixed_append_recovery_maps_only_submitted_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_text = _conversation()
    appended_texts = (_conversation(changed=True), _conversation())
    raw_contents = tuple(
        {
            "content": text,
            "document_id": "document",
            "update_mode": "append",
        }
        for text in appended_texts
    )
    invocation = RetainInvocation(
        bank_id="bank",
        raw_contents=raw_contents,
        request_context=object(),
    )
    submitted_intent = plan_documents(normalize_contents(raw_contents))[0]
    combined_intent = prepend_existing_document(submitted_intent, original_text)
    combined_content = "\n".join(item.content for item in combined_intent.items)
    combined_chunks = build_chunk_plans(
        combined_intent.document_id,
        combined_intent.items,
        _policy(),
    )
    existing = ExistingDocument(
        document_id="document",
        bank_id="bank",
        original_text=combined_content,
        content_hash=service_module.compute_document_hash(combined_content),
        retain_params={},
        tags=(),
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    durable_chunks = tuple(
        ExistingChunkFingerprint(
            chunk_id=f"stored-chunk-{chunk.global_index}",
            chunk_index=chunk.global_index,
            content_hash=chunk.content_hash,
        )
        for chunk in combined_chunks
    )
    second_suffix_chunk = next(chunk for chunk in combined_chunks if chunk.source_index == 1)
    bindings = (
        CommittedUnitBinding(unit_id="unit-prefix", chunk_index=0),
        CommittedUnitBinding(
            unit_id="unit-second-submitted-input",
            chunk_index=second_suffix_chunk.global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=durable_chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("fixed append recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-prefix", "unit-second-submitted-input")),),
    )

    plans = await service_module.RetainPipelineService()._preflight_documents(
        invocation,
        _execution(retry_llm),
        (submitted_intent,),
        _policy(),
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.intent.items) == 2
    assert service_module.RetainPipelineService._recovery_result_buckets(plan) == (
        (),
        ("unit-second-submitted-input",),
    )


@pytest.mark.asyncio
async def test_committed_semantic_recovery_reuses_durable_layout_across_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    initial_state, initial_events = _install_planning_backend(monkeypatch)
    initial_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=initial_state,
                events=initial_events,
            )
        ),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        initial_plan,
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-semantic",
            chunk_index=initial_plan.chunks[-1].global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("committed semantic recovery must not call the provider"))
    retry_llm.model = "different-boundary-model"
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-semantic",)),),
    )

    plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(retry_llm),
        text=text,
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert plan.change.kind is DocumentChangeKind.METADATA_ONLY
    assert plan.recovered_chunk_sources is not None
    assert service_module.RetainPipelineService._recovery_result_buckets(plan) == (("unit-semantic",),)


@pytest.mark.asyncio
async def test_committed_semantic_recovery_rejects_tampered_manifest_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    initial_state, initial_events = _install_planning_backend(monkeypatch)
    initial_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=initial_state,
                events=initial_events,
            )
        ),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        initial_plan,
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    retain_params = json.loads(json.dumps(existing.retain_params))
    retain_params[service_module._SEMANTIC_PLAN_METADATA_KEY]["plan_digest"] = "0" * 64
    existing = replace(existing, retain_params=retain_params)
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-semantic",
            chunk_index=initial_plan.chunks[-1].global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("tampered recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-semantic",)),),
    )

    with pytest.raises(service_module.RetainCheckpointRecoveryError, match="semantic plan"):
        await _preflight(
            invocation=_invocation(text),
            execution=_execution(retry_llm),
            text=text,
            checkpoint=checkpoint,
            request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    assert retry_llm.calls == []


@pytest.mark.asyncio
async def test_committed_semantic_recovery_rejects_changed_retry_input_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    initial_state, initial_events = _install_planning_backend(monkeypatch)
    initial_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=initial_state,
                events=initial_events,
            )
        ),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        initial_plan,
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-semantic",
            chunk_index=initial_plan.chunks[-1].global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("changed recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-semantic",)),),
    )
    changed_text = _conversation(changed=True)

    with pytest.raises(service_module.RetainCheckpointRecoveryError, match="retry input"):
        await _preflight(
            invocation=_invocation(changed_text),
            execution=_execution(retry_llm),
            text=changed_text,
            checkpoint=checkpoint,
            request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    assert retry_llm.calls == []


@pytest.mark.asyncio
async def test_committed_semantic_recovery_rejects_durable_chunk_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    initial_state, initial_events = _install_planning_backend(monkeypatch)
    initial_plan = await _preflight(
        invocation=_invocation(text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=initial_state,
                events=initial_events,
            )
        ),
        text=text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        initial_plan,
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    chunks = (*chunks[:-1], replace(chunks[-1], content_hash="0" * 64))
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-semantic",
            chunk_index=initial_plan.chunks[-1].global_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("mismatched recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-semantic",)),),
    )

    with pytest.raises(service_module.RetainCheckpointRecoveryError, match="recovery layout"):
        await _preflight(
            invocation=_invocation(text),
            execution=_execution(retry_llm),
            text=text,
            checkpoint=checkpoint,
            request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    assert retry_llm.calls == []


@pytest.mark.asyncio
async def test_committed_single_input_fixed_recovery_survives_chunk_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    invocation, intent, existing, chunks = _fixed_persisted_document(
        (text,),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-fixed",
            chunk_index=chunks[-1].chunk_index,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("fixed recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-fixed",)),),
    )
    changed_policy = ChunkPolicy(
        version="retain-chunker-v1",
        max_chars=200,
        conversation_mode=True,
        overlap=0,
    )

    plans = await service_module.RetainPipelineService()._preflight_documents(
        invocation,
        _execution(retry_llm),
        (intent,),
        changed_policy,
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert len(plans) == 1
    assert service_module.RetainPipelineService._recovery_result_buckets(plans[0]) == (("unit-fixed",),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bindings", "unit_ids"),
    (
        ((), ()),
        (
            (CommittedUnitBinding(unit_id="unit-chunkless", chunk_index=None),),
            ("unit-chunkless",),
        ),
    ),
)
async def test_committed_unmapped_recovery_rejects_changed_retry_input(
    monkeypatch: pytest.MonkeyPatch,
    bindings: tuple[CommittedUnitBinding, ...],
    unit_ids: tuple[str, ...],
) -> None:
    original_text = _conversation()
    _invocation_original, _intent_original, existing, chunks = _fixed_persisted_document(
        (original_text,),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    changed_text = _conversation(changed=True)
    changed_invocation = _invocation(changed_text)
    changed_intent = _intent(changed_text)
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("changed committed retry must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", unit_ids),),
    )

    with pytest.raises(service_module.RetainCheckpointRecoveryError, match="durable document"):
        await service_module.RetainPipelineService()._preflight_documents(
            changed_invocation,
            _execution(retry_llm),
            (changed_intent,),
            _policy(),
            checkpoint=checkpoint,
            request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    assert retry_llm.calls == []


@pytest.mark.asyncio
async def test_unscoped_committed_recovery_rejects_non_operation_local_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    invocation, intent, existing, chunks = _fixed_persisted_document(
        (text,),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=(
            CommittedUnitBinding(unit_id="unit-from-older-operation", chunk_index=0),
            CommittedUnitBinding(unit_id="unit-from-this-operation", chunk_index=0),
        ),
    )
    retry_llm = _BoundaryLLM(error=AssertionError("unscoped recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        document_ids=("document",),
        unscoped_facts_committed=True,
    )

    with pytest.raises(service_module.RetainCheckpointRecoveryError, match="operation-local unit IDs"):
        await service_module.RetainPipelineService()._preflight_documents(
            invocation,
            _execution(retry_llm),
            (intent,),
            _policy(),
            checkpoint=checkpoint,
            request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    assert retry_llm.calls == []


@pytest.mark.asyncio
async def test_committed_zero_unit_recovery_needs_no_chunks_or_semantic_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _conversation()
    invocation, intent, existing, chunks = _fixed_persisted_document(
        (text,),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
        bindings=(),
    )
    retry_llm = _BoundaryLLM(error=AssertionError("empty recovery must not call the provider"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ()),),
    )

    plans = await service_module.RetainPipelineService()._preflight_documents(
        invocation,
        _execution(retry_llm),
        (intent,),
        _policy(),
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert len(plans) == 1
    assert plans[0].recovered_chunk_sources == ()
    assert service_module.RetainPipelineService._recovery_result_buckets(plans[0]) == ((),)


@pytest.mark.asyncio
async def test_committed_semantic_append_retry_reuses_trailing_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_text = _conversation()
    original_state, original_events = _install_planning_backend(monkeypatch)
    original_plan = await _preflight(
        invocation=_invocation(original_text),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=original_state,
                events=original_events,
            )
        ),
        text=original_text,
        request_started_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    existing, chunks = _persisted_document(
        original_plan,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    appended_text = _conversation(changed=True)
    append_state, append_events = _install_planning_backend(
        monkeypatch,
        existing=existing,
        chunks=chunks,
    )
    append_plan = await _preflight(
        invocation=_invocation(appended_text, update_mode="append"),
        execution=_execution(
            _BoundaryLLM(
                [0, 1, 2],
                snapshot_state=append_state,
                events=append_events,
            )
        ),
        text=appended_text,
        update_mode="append",
        request_started_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    committed_document, committed_chunks = _persisted_document(
        append_plan,
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    appended_chunk_count = sum(1 for chunk in append_plan.chunks if chunk.source_index == 0)
    appended_suffix_start = len(committed_chunks) - appended_chunk_count
    bindings = (
        CommittedUnitBinding(
            unit_id="unit-prefix",
            chunk_index=0,
        ),
        CommittedUnitBinding(
            unit_id="unit-appended",
            chunk_index=len(committed_chunks) - 1,
        ),
    )
    _install_planning_backend(
        monkeypatch,
        existing=committed_document,
        chunks=committed_chunks,
        bindings=bindings,
    )
    retry_llm = _BoundaryLLM(error=AssertionError("committed append retry must reuse its trailing manifest"))
    checkpoint = OperationCheckpoint(
        core_committed_document_ids=("document",),
        committed_unit_ids_by_document=(("document", ("unit-prefix", "unit-appended")),),
    )

    retry_plan = await _preflight(
        invocation=_invocation(appended_text, update_mode="append"),
        execution=_execution(retry_llm),
        text=appended_text,
        update_mode="append",
        checkpoint=checkpoint,
        request_started_at=datetime(2026, 1, 4, tzinfo=UTC),
    )

    assert retry_llm.calls == []
    assert retry_plan.change.kind is DocumentChangeKind.METADATA_ONLY
    assert retry_plan.change.reason == "operation core commit recovered"
    assert retry_plan.recovered_unit_bindings == bindings
    assert len(retry_plan.intent.items) == 1
    assert (
        retry_plan.segmentation_metadata == committed_document.retain_params[service_module._SEMANTIC_PLAN_METADATA_KEY]
    )
    assert retry_plan.segmentation_usage == TokenUsage()
    assert retry_plan.recovered_chunk_sources is not None
    assert all(source_index is None for _, source_index in retry_plan.recovered_chunk_sources[:appended_suffix_start])
    assert all(source_index == 0 for _, source_index in retry_plan.recovered_chunk_sources[appended_suffix_start:])
    assert service_module.RetainPipelineService._recovery_result_buckets(retry_plan) == (("unit-appended",),)
