import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchmarks.common.benchmark_runner import (
    BenchmarkRunner,
    LLMAnswerEvaluator,
    _embedding_runtime_config,
    _endpoint_fingerprint,
    _reranker_runtime_config,
    _write_json_atomic,
    get_model_config,
)


class _Dataset:
    def get_item_id(self, item):
        return item["id"]

    def prepare_sessions_for_ingestion(self, item):
        return [{"content": item["id"], "document_id": f"document-{item['id']}"}]

    def get_qa_pairs(self, item):
        return [{"question": item["id"], "answer": "answer", "category": "test"}]


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


@pytest.mark.asyncio
async def test_fresh_parallel_run_processes_items_before_any_bank_exists():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner.template_path = None
    runner._agent_has_data = AsyncMock(return_value=False)

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
    runner._agent_has_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_item_reingests_even_when_a_bank_already_exists():
    runner = object.__new__(BenchmarkRunner)
    runner.dataset = _Dataset()
    runner.template_path = None
    runner.memory = Mock()
    runner.memory.delete_bank = AsyncMock()
    runner._agent_has_data = AsyncMock(return_value=True)
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

    runner._agent_has_data.assert_not_awaited()
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
    runner._agent_has_data = AsyncMock(return_value=False)

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
    runner._agent_has_data = AsyncMock(side_effect=[True, False])
    runner.process_single_item = AsyncMock()

    with pytest.raises(RuntimeError, match="missing 1 bank"):
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

    runner.process_single_item.assert_not_awaited()


def test_atomic_json_write_replaces_the_complete_artifact(tmp_path: Path):
    output_path = tmp_path / "result.json"
    output_path.write_text('{"stale": true}\n', encoding="utf-8")

    _write_json_atomic({"item_results": [{"item_id": "a"}]}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"item_results": [{"item_id": "a"}]}
    assert not (tmp_path / ".result.json.tmp").exists()


def test_incremental_checkpoint_preserves_run_manifest(tmp_path: Path):
    runner = object.__new__(BenchmarkRunner)
    runner._run_manifest = {"artifact_schema_version": 1, "dataset": {"sha256": "abc"}}
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
