import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchmarks.longmemeval import longmemeval_benchmark as benchmark


@pytest.mark.asyncio
async def test_answer_provider_failure_propagates_to_the_runner():
    generator = object.__new__(benchmark.LongMemEvalAnswerGenerator)
    generator.context_format = "json"
    generator.evidence_mode = None
    generator.llm_config = type("Config", (), {})()
    generator.llm_config.call = AsyncMock(side_effect=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await generator.generate_answer(
            "What happened?",
            {"results": [{"text": "memory"}]},
        )


def _manifest() -> dict:
    return {
        "artifact_schema_version": 1,
        "dataset": {
            "path": "datasets/dataset.json",
            "sha256": "dataset-sha",
            "revision": "revision",
            "expected_full_items": 500,
        },
        "pipeline": {
            "planner": "ledger",
            "context_format": "structured_source",
            "thinking_budget": 500,
            "max_tokens": 8192,
            "query_expansion_enabled": False,
            "query_rewriting_strategy": "noop",
            "session_expansion_weight": 0.3,
        },
        "database": {
            "backend": "postgresql",
            "schema": "public",
            "vector_extension": "pgvector",
        },
        "concurrency": {"items": 1, "questions": 1, "judge": 1},
        "runtime": {
            "git_commit": "abc123",
            "git_dirty": False,
            "source_tree_fingerprint": None,
        },
    }


def _model_config() -> dict:
    role = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "endpoint_fingerprint": "sha256:endpoint",
    }
    return {
        "hms": dict(role),
        "retain": dict(role),
        "answer_generation": dict(role),
        "judge": dict(role),
        "embeddings": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "fingerprint_policy": "strict",
            "endpoint_fingerprint": "sha256:embedding",
        },
        "reranker": {"provider": "rrf", "model": ""},
    }


def _built_manifest(dataset_path: Path) -> dict:
    return benchmark.build_run_manifest(
        dataset_path=dataset_path,
        context_format="structured_source",
        max_instances=1,
        max_instances_per_category=None,
        max_questions_per_instance=1,
        question_id=None,
        index_range=None,
        category=None,
        max_concurrent_items=1,
        max_concurrent_questions=1,
        eval_semaphore_size=1,
        thinking_budget=500,
        max_tokens=8192,
        oracle_planner_v26=False,
        oracle_planner_v220=False,
        query_expansion_enabled=False,
        query_rewriting_strategy="noop",
        session_expansion_weight=0.3,
        skip_ingestion=False,
        ingest_only=False,
        force_reingest=False,
    )


def test_run_manifest_never_serializes_an_absolute_dataset_path(tmp_path: Path, monkeypatch):
    repository_root = tmp_path / "checkout"
    repository_dataset = repository_root / ".aaaDATA" / "longmemeval" / "dataset.json"
    repository_dataset.parent.mkdir(parents=True)
    repository_dataset.write_text("repository dataset", encoding="utf-8")

    external_dataset = tmp_path / "private" / "custom-dataset.json"
    external_dataset.parent.mkdir()
    external_dataset.write_text("external dataset", encoding="utf-8")

    monkeypatch.setattr(benchmark, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(benchmark, "_git_value", lambda *args: "abc123" if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(benchmark, "_source_tree_fingerprint", lambda: None)

    repository_manifest = _built_manifest(repository_dataset)
    external_manifest = _built_manifest(external_dataset)

    assert repository_manifest["dataset"]["path"] == ".aaaDATA/longmemeval/dataset.json"
    assert external_manifest["dataset"]["path"] == "external:custom-dataset.json"
    assert not Path(repository_manifest["dataset"]["path"]).is_absolute()
    assert not Path(external_manifest["dataset"]["path"]).is_absolute()


def test_resume_compatibility_ignores_concurrency_but_rejects_model_changes(tmp_path: Path):
    output_path = tmp_path / "results.json"
    manifest = _manifest()
    model_config = _model_config()
    output_path.write_text(
        json.dumps(
            {
                "run_manifest": manifest,
                "model_config": model_config,
                "item_results": [],
            }
        ),
        encoding="utf-8",
    )

    current_manifest = copy.deepcopy(manifest)
    current_manifest["dataset"]["path"] = "external:dataset.json"
    current_manifest["concurrency"]["items"] = 8
    benchmark.validate_artifact_compatibility(
        output_path,
        current_manifest=current_manifest,
        current_model_config=model_config,
    )

    incompatible_models = copy.deepcopy(model_config)
    incompatible_models["answer_generation"]["model"] = "different-model"
    with pytest.raises(ValueError, match="model_config"):
        benchmark.validate_artifact_compatibility(
            output_path,
            current_manifest=current_manifest,
            current_model_config=incompatible_models,
        )

    incompatible_source = copy.deepcopy(current_manifest)
    incompatible_source["runtime"]["source_tree_fingerprint"] = "sha256:different"
    with pytest.raises(ValueError, match="source_tree_fingerprint"):
        benchmark.validate_artifact_compatibility(
            output_path,
            current_manifest=incompatible_source,
            current_model_config=model_config,
        )


def test_source_tree_fingerprint_tracks_relevant_dirty_content(monkeypatch):
    def clean_git_bytes(*args: str) -> bytes:
        return b""

    monkeypatch.setattr(benchmark, "_git_bytes", clean_git_bytes)
    assert benchmark._source_tree_fingerprint() is None

    def dirty_git_bytes(*args: str) -> bytes:
        return b"diff --git a/source.py b/source.py\n+changed\n" if args[0] == "diff" else b""

    monkeypatch.setattr(benchmark, "_git_bytes", dirty_git_bytes)
    first = benchmark._source_tree_fingerprint()
    assert first is not None
    assert first.startswith("sha256:")

    def different_git_bytes(*args: str) -> bytes:
        return b"diff --git a/source.py b/source.py\n+different\n" if args[0] == "diff" else b""

    monkeypatch.setattr(benchmark, "_git_bytes", different_git_bytes)
    assert benchmark._source_tree_fingerprint() != first


def test_explicit_canonical_dataset_path_is_checksum_validated(tmp_path: Path, monkeypatch):
    canonical_path = tmp_path / "longmemeval_s_cleaned.json"
    canonical_payload = b"good-data"
    canonical_path.write_bytes(b"evil-data")
    monkeypatch.setattr(benchmark, "DEFAULT_DATASET_PATH", canonical_path)
    monkeypatch.setattr(benchmark, "LONGMEMEVAL_DATASET_SIZE", len(canonical_payload))
    monkeypatch.setattr(
        benchmark,
        "LONGMEMEVAL_DATASET_SHA256",
        hashlib.sha256(canonical_payload).hexdigest(),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.resolve_dataset_path(str(canonical_path))

    canonical_path.write_bytes(canonical_payload)
    assert benchmark.resolve_dataset_path(str(canonical_path)) == canonical_path


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_instances", 0),
        ("max_concurrent_items", 0),
        ("max_concurrent_questions", 0),
        ("eval_semaphore_size", 0),
        ("thinking_budget", 0),
        ("max_tokens", 0),
    ],
)
def test_runtime_limits_must_be_positive(field_name: str, value: int):
    options = {
        "max_instances": 1,
        "max_instances_per_category": None,
        "max_questions_per_instance": 1,
        "max_concurrent_items": 1,
        "max_concurrent_questions": 1,
        "eval_semaphore_size": 1,
        "thinking_budget": 1,
        "max_tokens": 1,
    }
    options[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        benchmark.validate_runtime_options(**options)


def test_fresh_output_refuses_to_overwrite_existing_artifact(tmp_path: Path):
    output_path = tmp_path / "results.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        benchmark.validate_output_target(
            output_path,
            merge_with_existing=False,
            resume=False,
        )


def test_resume_requires_an_existing_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="requires an existing"):
        benchmark.validate_output_target(
            tmp_path / "missing.json",
            merge_with_existing=True,
            resume=True,
        )


def _launcher_environment(tmp_path: Path) -> dict[str, str]:
    environment_file = tmp_path / "empty.env"
    environment_file.write_text("", encoding="utf-8")
    return {
        "PATH": os.environ["PATH"],
        "HMS_ENV_FILE": str(environment_file),
        "HMS_API_DATABASE_URL": "postgresql://hms:test@127.0.0.1:5432/hms",
        "HMS_DATA_DIR": str(tmp_path / "data"),
        "HMS_LOG_DIR": str(tmp_path / "logs"),
        "HMS_RESULT_DIR": str(tmp_path / "results"),
    }


def test_launcher_rejects_an_explicit_missing_environment_file(tmp_path: Path):
    environment = _launcher_environment(tmp_path)
    environment["HMS_ENV_FILE"] = str(tmp_path / "missing.env")

    completed = subprocess.run(
        ["bash", str(benchmark.REPOSITORY_ROOT / ".aaaSCRIPT" / "run_benchmark.sh")],
        cwd=benchmark.REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "HMS_ENV_FILE does not exist" in completed.stderr


def test_launcher_rejects_zero_concurrency_before_starting_python(tmp_path: Path):
    environment = _launcher_environment(tmp_path)
    environment["HMS_PARALLEL"] = "0"

    completed = subprocess.run(
        ["bash", str(benchmark.REPOSITORY_ROOT / ".aaaSCRIPT" / "run_benchmark.sh")],
        cwd=benchmark.REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "HMS_PARALLEL must be a positive integer" in completed.stderr
