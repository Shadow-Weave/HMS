import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from hms_api.engine.ingestion.adapters.storage_records import compute_document_hash

from benchmarks.common.benchmark_runner import (
    BenchmarkRunner,
    IngestionIntegrityError,
    LLMAnswerEvaluator,
    _embedding_runtime_config,
    _endpoint_fingerprint,
    _reranker_runtime_config,
    _write_json_atomic,
    get_artifact_model_config,
    get_model_config,
)

_EVENT_DATE = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class _Dataset:
    def get_item_id(self, item):
        return item["id"]

    def prepare_sessions_for_ingestion(self, item):
        return [
            {
                "content": item["id"],
                "context": f"context-{item['id']}",
                "event_date": _EVENT_DATE,
                "document_id": f"document-{item['id']}",
            }
        ]

    def get_qa_pairs(self, item):
        return [{"question": item["id"], "answer": "answer", "category": "test"}]


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Pool:
    def __init__(self, rows):
        self.connection = SimpleNamespace(fetch=AsyncMock(return_value=rows))

    def acquire(self):
        return _Acquire(self.connection)


def _runner_with_document_rows(rows):
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    pool = _Pool(rows)
    runner.memory = SimpleNamespace(_get_pool=AsyncMock(return_value=pool))
    return runner


def _document_row(
    item_id: str,
    *,
    content_hash: str | None = None,
    retain_params=None,
    chunks: int = 1,
    facts: int = 0,
):
    return {
        "id": f"document-{item_id}",
        "content_hash": content_hash if content_hash is not None else compute_document_hash(item_id),
        "retain_params": (
            retain_params
            if retain_params is not None
            else {
                "context": f"context-{item_id}",
                "event_date": _EVENT_DATE.isoformat(),
            }
        ),
        "chunk_count": chunks,
        "fact_count": facts,
    }


def test_embedding_runtime_config_uses_provider_specific_identity(monkeypatch):
    monkeypatch.setenv("HMS_API_EMBEDDINGS_PROVIDER", "cohere")
    monkeypatch.setenv("HMS_API_EMBEDDINGS_COHERE_MODEL", "embed-v4.0")
    monkeypatch.setenv("HMS_API_EMBEDDINGS_COHERE_BASE_URL", "https://cohere.example/v2")
    monkeypatch.setenv("HMS_API_EMBEDDINGS_OPENAI_MODEL", "must-not-be-used")
    monkeypatch.setenv("HMS_API_EMBEDDINGS_OPENAI_BASE_URL", "https://openai.example/v1")

    config = _embedding_runtime_config()

    assert config == {
        "provider": "cohere",
        "model": "embed-v4.0",
        "fingerprint_policy": "strict",
        "endpoint_fingerprint": _endpoint_fingerprint("https://cohere.example/v2"),
    }


def test_reranker_runtime_config_uses_provider_specific_identity(monkeypatch):
    monkeypatch.setenv("HMS_API_RERANKER_PROVIDER", "litellm-sdk")
    monkeypatch.setenv("HMS_API_RERANKER_LITELLM_SDK_MODEL", "vendor/rerank-v2")
    monkeypatch.setenv("HMS_API_RERANKER_LITELLM_SDK_API_BASE", "https://gateway.example/v1")
    monkeypatch.setenv("HMS_API_RERANKER_LOCAL_MODEL", "must-not-be-used")

    config = _reranker_runtime_config()

    assert config == {
        "provider": "litellm-sdk",
        "model": "vendor/rerank-v2",
        "endpoint_fingerprint": _endpoint_fingerprint("https://gateway.example/v1"),
    }


def test_model_config_uses_retain_provider_default_when_only_provider_is_overridden(monkeypatch):
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "groq")
    monkeypatch.setenv("HMS_API_LLM_MODEL", "custom-memory-model")
    monkeypatch.setenv("HMS_API_RETAIN_LLM_PROVIDER", "openai")
    monkeypatch.delenv("HMS_API_RETAIN_LLM_MODEL", raising=False)

    config = get_model_config()

    assert config["hms"]["model"] == "custom-memory-model"
    assert config["retain"]["provider"] == "openai"
    assert config["retain"]["model"] == "gpt-4o-mini"


def test_reused_bank_model_config_does_not_claim_current_retain_identity(monkeypatch):
    monkeypatch.setenv("HMS_API_RETAIN_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HMS_API_RETAIN_LLM_MODEL", "current-but-not-bank-creator")

    config = get_artifact_model_config(retain_executed=False)

    assert config["retain"] == {
        "execution": "not_executed",
        "bank_creator_identity": "unverifiable",
    }


def test_mixed_ingest_only_model_config_does_not_claim_one_global_retain_identity():
    config = get_artifact_model_config(retain_execution="partial_or_skipped")

    assert config["retain"] == {
        "execution": "partial_or_skipped",
        "bank_creator_identity": "mixed_or_unverifiable",
    }


@pytest.mark.asyncio
async def test_ingestion_audit_accepts_exact_document_with_zero_facts():
    runner = _runner_with_document_rows([_document_row("a", facts=0)])

    report = await runner._audit_durable_ingestion({"id": "a"}, "longmemeval_a")

    assert report["durable_documents"] == 1
    assert report["documents_without_facts"] == ["document-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "failure_field"),
    [
        (_document_row("a", content_hash="stale"), "content_hash_mismatches"),
        (
            _document_row(
                "a",
                retain_params={"context": "stale-context", "event_date": _EVENT_DATE.isoformat()},
            ),
            "context_mismatches",
        ),
        (
            _document_row(
                "a",
                retain_params={"context": "context-a", "event_date": "2020-01-01T00:00:00+00:00"},
            ),
            "event_date_mismatches",
        ),
        (_document_row("a", chunks=0), "documents_without_chunks"),
    ],
)
async def test_ingestion_audit_rejects_stale_or_unqueryable_document(row, failure_field):
    runner = _runner_with_document_rows([row])

    with pytest.raises(IngestionIntegrityError) as exc_info:
        await runner._audit_durable_ingestion({"id": "a"}, "longmemeval_a")

    assert exc_info.value.report[failure_field]


@pytest.mark.asyncio
async def test_ingestion_audit_rejects_unexpected_and_inflight_documents():
    inflight = _document_row("a", content_hash="retain-inflight:operation")
    unexpected = _document_row("other")
    runner = _runner_with_document_rows([inflight, unexpected])

    with pytest.raises(IngestionIntegrityError) as exc_info:
        await runner._audit_durable_ingestion({"id": "a"}, "longmemeval_a")

    assert exc_info.value.report["inflight_documents"] == ["document-a"]
    assert exc_info.value.report["unexpected_documents"] == ["document-other"]


@pytest.mark.asyncio
async def test_fresh_parallel_run_processes_items_before_any_bank_exists():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner.template_path = None
    runner._preflight_reusable_items = AsyncMock()

    async def process_item(item, *args, **kwargs):
        return {
            "item_id": item["id"],
            "metrics": {"correct": 1, "total": 1, "invalid": 0},
            "num_sessions": 1,
        }

    runner.process_single_item = AsyncMock(side_effect=process_item)
    runner._save_incremental_results = Mock()

    results = await runner._process_items_parallel(
        items=[{"id": "a"}, {"id": "b"}],
        agent_id="longmemeval",
        thinking_budget=10,
        max_tokens=100,
        skip_ingestion=False,
        max_questions_per_item=None,
        question_semaphore=asyncio.Semaphore(2),
        eval_semaphore=asyncio.Semaphore(2),
        filln=False,
        max_concurrent_items=2,
    )

    assert {result["item_id"] for result in results} == {"a", "b"}
    assert runner.process_single_item.await_count == 2
    runner._preflight_reusable_items.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_item_reingests_even_when_a_bank_already_exists():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner.template_path = None
    runner.memory = Mock()
    runner.memory.delete_bank = AsyncMock()
    runner.ingest_conversation = AsyncMock(return_value=1)
    runner._audit_durable_ingestion = AsyncMock(
        return_value={
            "durable_documents": 1,
            "expected_documents": 1,
            "missing_documents": [],
            "documents_without_chunks": [],
        }
    )
    runner.evaluate_qa_task = AsyncMock(return_value=[])
    runner.calculate_metrics = AsyncMock(return_value={"accuracy": 0.0, "correct": 0, "total": 0, "invalid": 0})

    await runner.process_single_item(
        {"id": "a"},
        "longmemeval_a",
        1,
        1,
        10,
        100,
        None,
        False,
        asyncio.Semaphore(1),
        asyncio.Semaphore(1),
        clear_this_agent=True,
        skip_if_already_ingested=False,
    )

    runner.memory.delete_bank.assert_awaited_once()
    runner.ingest_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingestion_raises_after_the_last_retry(monkeypatch):
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner.memory = Mock()
    runner.memory.retain_batch_async = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await runner.ingest_conversation({"id": "a"}, "longmemeval_a")

    assert runner.memory.retain_batch_async.await_count == 3


@pytest.mark.asyncio
async def test_judge_provider_failure_is_recorded_as_invalid():
    evaluator = object.__new__(LLMAnswerEvaluator)
    evaluator.llm_config = Mock()
    evaluator.llm_config.call = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    runner = object.__new__(BenchmarkRunner)
    runner.answer_evaluator = evaluator

    metrics = await runner.calculate_metrics(
        [
            {
                "question": "question",
                "correct_answer": "answer",
                "predicted_answer": "prediction",
                "category": "multi-session",
                "is_invalid": False,
                "error": None,
            }
        ],
        asyncio.Semaphore(1),
        "a",
    )

    assert metrics["invalid"] == 1
    assert metrics["correct"] == 0
    assert metrics["detailed_results"][0]["is_invalid"] is True


@pytest.mark.asyncio
async def test_judge_concurrency_limit_is_shared_across_items():
    active = 0
    maximum_active = 0

    async def fake_call(**kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return SimpleNamespace(correct=True, reasoning="ok")

    evaluator = object.__new__(LLMAnswerEvaluator)
    evaluator.llm_config = Mock()
    evaluator.llm_config.call = fake_call
    runner = object.__new__(BenchmarkRunner)
    runner.answer_evaluator = evaluator
    shared_semaphore = asyncio.Semaphore(1)
    result = {
        "question": "question",
        "correct_answer": "answer",
        "predicted_answer": "answer",
        "category": "multi-session",
        "is_invalid": False,
        "error": None,
    }

    await asyncio.gather(
        runner.calculate_metrics([dict(result)], shared_semaphore, "a"),
        runner.calculate_metrics([dict(result)], shared_semaphore, "b"),
    )

    assert maximum_active == 1


@pytest.mark.asyncio
async def test_resume_skips_valid_items_and_reruns_invalid_items(tmp_path: Path):
    output_path = tmp_path / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "item_results": [
                    {
                        "item_id": "valid",
                        "metrics": {
                            "total": 1,
                            "correct": 1,
                            "invalid": 0,
                            "detailed_results": [
                                {
                                    "is_correct": True,
                                    "is_invalid": False,
                                    "error": None,
                                    "predicted_answer": "answer",
                                    "correctness_reasoning": "correct",
                                }
                            ],
                        },
                        "num_sessions": 1,
                    },
                    {
                        "item_id": "invalid",
                        "metrics": {
                            "total": 1,
                            "correct": 0,
                            "invalid": 1,
                            "detailed_results": [
                                {
                                    "is_correct": None,
                                    "is_invalid": True,
                                    "error": "provider unavailable",
                                }
                            ],
                        },
                        "num_sessions": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()

    async def process_item(item, *args, **kwargs):
        return {
            "item_id": item["id"],
            "metrics": {
                "total": 1,
                "correct": 1,
                "invalid": 0,
                "detailed_results": [
                    {
                        "is_correct": True,
                        "is_invalid": False,
                        "error": None,
                        "predicted_answer": "answer",
                        "correctness_reasoning": "correct",
                    }
                ],
            },
            "num_sessions": 1,
        }

    runner.process_single_item = AsyncMock(side_effect=process_item)
    runner._save_incremental_results = Mock()

    results = await runner._process_items_sequential(
        items=[{"id": "valid"}, {"id": "invalid"}],
        agent_id="longmemeval",
        thinking_budget=10,
        max_tokens=100,
        skip_ingestion=False,
        max_questions_per_item=None,
        question_semaphore=asyncio.Semaphore(1),
        eval_semaphore=asyncio.Semaphore(1),
        clear_agent_per_item=True,
        filln=True,
        output_path=output_path,
        merge_with_existing=True,
        rerun_invalid_existing=True,
    )

    assert runner.process_single_item.await_count == 1
    assert runner.process_single_item.await_args.args[0]["id"] == "invalid"
    assert {result["item_id"] for result in results} == {"valid", "invalid"}


@pytest.mark.asyncio
async def test_retrieval_only_fails_when_any_selected_bank_is_missing():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner._audit_durable_ingestion = AsyncMock(
        side_effect=[
            {"durable_documents": 1},
            IngestionIntegrityError(
                {
                    "item_id": "b",
                    "bank_id": "longmemeval_b",
                    "missing_documents": ["document-b"],
                }
            ),
        ]
    )
    runner.process_single_item = AsyncMock()

    with pytest.raises(IngestionIntegrityError, match="reuse-preflight"):
        await runner._process_items_parallel(
            items=[{"id": "a"}, {"id": "b"}],
            agent_id="longmemeval",
            thinking_budget=10,
            max_tokens=100,
            skip_ingestion=True,
            max_questions_per_item=None,
            question_semaphore=asyncio.Semaphore(1),
            eval_semaphore=asyncio.Semaphore(1),
            filln=False,
            max_concurrent_items=2,
        )

    assert runner._audit_durable_ingestion.await_count == 2
    runner.process_single_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_sequential_retrieval_only_preflights_every_bank_before_qa():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner._audit_durable_ingestion = AsyncMock(
        side_effect=[
            {"durable_documents": 1},
            IngestionIntegrityError(
                {
                    "item_id": "b",
                    "bank_id": "longmemeval_b",
                    "content_hash_mismatches": [{"document_id": "document-b"}],
                }
            ),
        ]
    )
    runner.process_single_item = AsyncMock()

    with pytest.raises(IngestionIntegrityError, match="reuse-preflight"):
        await runner._process_items_sequential(
            items=[{"id": "a"}, {"id": "b"}],
            agent_id="longmemeval",
            thinking_budget=10,
            max_tokens=100,
            skip_ingestion=True,
            max_questions_per_item=None,
            question_semaphore=asyncio.Semaphore(1),
            eval_semaphore=asyncio.Semaphore(1),
            clear_agent_per_item=True,
            filln=False,
        )

    assert runner._audit_durable_ingestion.await_count == 2
    runner.process_single_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_only_reingests_nonmatching_bank_and_skips_exact_bank():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner._preflight_reusable_items = AsyncMock(return_value={"a"})
    runner.process_single_item = AsyncMock(
        return_value={
            "item_id": "b",
            "metrics": {"correct": 0, "total": 0, "invalid": 0},
            "num_sessions": 1,
        }
    )

    results = await runner._process_items_sequential(
        items=[{"id": "a"}, {"id": "b"}],
        agent_id="longmemeval",
        thinking_budget=10,
        max_tokens=100,
        skip_ingestion=False,
        max_questions_per_item=None,
        question_semaphore=asyncio.Semaphore(1),
        eval_semaphore=asyncio.Semaphore(1),
        clear_agent_per_item=True,
        filln=False,
        ingest_only=True,
    )

    assert [result["item_id"] for result in results] == ["b"]
    assert runner.process_single_item.await_count == 1
    assert runner.process_single_item.await_args.args[0] == {"id": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ingest_only", [False, True])
async def test_force_reingest_overrides_fill_skip_sequentially(tmp_path: Path, ingest_only: bool):
    output_path = tmp_path / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "item_results": [
                    {
                        "item_id": "a",
                        "metrics": {"correct": 0, "total": 0, "invalid": 0},
                        "num_sessions": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner._preflight_reusable_items = AsyncMock(return_value={"a"})
    runner._save_incremental_results = Mock()
    runner.process_single_item = AsyncMock(
        return_value={
            "item_id": "a",
            "metrics": {"correct": 0, "total": 0, "invalid": 0},
            "num_sessions": 1,
        }
    )

    results = await runner._process_items_sequential(
        items=[{"id": "a"}],
        agent_id="longmemeval",
        thinking_budget=10,
        max_tokens=100,
        skip_ingestion=False,
        max_questions_per_item=None,
        question_semaphore=asyncio.Semaphore(1),
        eval_semaphore=asyncio.Semaphore(1),
        clear_agent_per_item=True,
        filln=True,
        output_path=output_path,
        merge_with_existing=True,
        ingest_only=ingest_only,
        force_reingest=True,
    )

    assert [result["item_id"] for result in results] == ["a"]
    assert runner.process_single_item.await_args.kwargs["force_reingest"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("ingest_only", [False, True])
async def test_force_reingest_overrides_fill_skip_in_parallel(tmp_path: Path, ingest_only: bool):
    output_path = tmp_path / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "item_results": [
                    {
                        "item_id": "a",
                        "metrics": {"correct": 0, "total": 0, "invalid": 0},
                        "num_sessions": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner._preflight_reusable_items = AsyncMock(return_value={"a"})
    runner._save_incremental_results = Mock()
    runner.process_single_item = AsyncMock(
        return_value={
            "item_id": "a",
            "metrics": {"correct": 0, "total": 0, "invalid": 0},
            "num_sessions": 1,
        }
    )

    results = await runner._process_items_parallel(
        items=[{"id": "a"}],
        agent_id="longmemeval",
        thinking_budget=10,
        max_tokens=100,
        skip_ingestion=False,
        max_questions_per_item=None,
        question_semaphore=asyncio.Semaphore(1),
        eval_semaphore=asyncio.Semaphore(1),
        filln=True,
        max_concurrent_items=1,
        output_path=output_path,
        merge_with_existing=True,
        ingest_only=ingest_only,
        force_reingest=True,
    )

    assert [result["item_id"] for result in results] == ["a"]
    assert runner.process_single_item.await_args.kwargs["force_reingest"] is True


@pytest.mark.asyncio
async def test_two_phase_shared_bank_rejects_documents_outside_selected_union_before_qa():
    runner = _runner_with_document_rows(
        [
            _document_row("a"),
            _document_row("b"),
            _document_row("stale"),
        ]
    )
    runner.evaluate_qa_task = AsyncMock()

    with pytest.raises(IngestionIntegrityError) as exc_info:
        await runner._run_two_phase(
            items=[{"id": "a"}, {"id": "b"}],
            agent_id="shared",
            thinking_budget=10,
            max_tokens=100,
            skip_ingestion=True,
            max_questions_per_item=None,
            max_concurrent_questions=1,
            eval_semaphore_size=1,
        )

    assert "document-stale" in str(exc_info.value.report["bank_reports"])
    runner.evaluate_qa_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_trace_populates_diagnostics_but_is_not_sent_to_answer_generator():
    fact = SimpleNamespace(
        id="fact-1",
        document_id="document-a",
        context="context-a",
        occurred_start=_EVENT_DATE.isoformat(),
        fact_type="world",
        metadata={"proof_count": 2},
        model_dump=lambda: {
            "id": "fact-1",
            "document_id": "document-a",
            "text": "remembered fact",
        },
    )
    trace = {
        "rrf_merged": [
            {
                "node_id": "fact-1",
                "text": "remembered fact",
                "rrf_score": 0.5,
                "final_rrf_rank": 1,
            }
        ],
        "reranked": [
            {
                "node_id": "fact-1",
                "text": "remembered fact",
                "rerank_score": 0.9,
                "rerank_rank": 1,
                "rrf_rank": 1,
                "score_components": {"combined_score": 0.8},
            }
        ],
    }

    class _RecallResult:
        results = [fact]
        chunks = None
        entities = None

        def __init__(self):
            self.trace = trace

        def model_dump(self, *, exclude=None):
            payload = {"results": [fact.model_dump()], "trace": trace}
            for field in exclude or set():
                payload.pop(field, None)
            return payload

    generator = Mock()
    generator.needs_external_search.return_value = True
    generator.generate_answer = AsyncMock(return_value=("answer", "reasoning", None))
    runner = object.__new__(BenchmarkRunner)
    runner.answer_generator = generator
    runner.retrieval_planner = None
    runner.query_rewriting_enabled = False
    runner.query_rewriting_strategy_name = "noop"
    runner.session_expansion_weight = 0.3
    runner.memory = SimpleNamespace(
        recall_async=AsyncMock(return_value=_RecallResult()),
        _cross_encoder=SimpleNamespace(model_name="reranker", provider_name="test"),
    )

    _, _, _, _, retrieval_details = await runner.answer_question(
        "longmemeval_a",
        "What happened?",
    )

    assert runner.memory.recall_async.await_args.kwargs["enable_trace"] is True
    generator_payload = generator.generate_answer.await_args.args[1]
    assert "trace" not in generator_payload
    assert retrieval_details["coarse_search_results"].total_candidates == 1
    assert len(retrieval_details["reranked_results"].reranked_candidates) == 1


def test_atomic_json_write_replaces_the_complete_artifact(tmp_path: Path):
    output_path = tmp_path / "result.json"
    output_path.write_text('{"stale": true}\n', encoding="utf-8")

    _write_json_atomic({"item_results": [{"item_id": "a"}]}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"item_results": [{"item_id": "a"}]}
    assert not (tmp_path / ".result.json.tmp").exists()


def test_incremental_checkpoint_preserves_run_manifest(tmp_path: Path):
    runner = object.__new__(BenchmarkRunner)
    runner._run_manifest = {"artifact_schema_version": 2, "dataset": {"sha256": "abc"}}
    runner._model_config = get_artifact_model_config(retain_executed=False)
    output_path = tmp_path / "result.json"

    runner._save_incremental_results(
        [
            {
                "item_id": "a",
                "metrics": {"correct": 1, "total": 1, "invalid": 0},
                "num_sessions": 1,
            }
        ],
        output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run_manifest"] == runner._run_manifest
    assert saved["model_config"]["retain"]["bank_creator_identity"] == "unverifiable"
