from __future__ import annotations

import asyncio
import sys

import pytest
from rsi_harness import (
    BenchmarkVerifierConfigurationError,
    BenchmarkVerifierProcess,
    EvaluationLease,
)


def lease() -> EvaluationLease:
    return EvaluationLease("eval-1", "snapshot-1", "held-out", "opaque-lease")


def test_benchmark_verifier_uses_read_only_environment_and_hides_output(tmp_path):
    verifier = BenchmarkVerifierProcess(
        command=(sys.executable, "-c", "import os; print(os.environ['PGOPTIONS']); print('hidden')"),
        cwd=str(tmp_path),
        read_only_database_url="postgresql://readonly@db.example/hms",
    )
    result = asyncio.run(verifier.evaluate(lease()))
    assert result.passed
    assert result.metrics == {}


def test_benchmark_verifier_rejects_evolution_and_secret_overrides(tmp_path):
    with pytest.raises(BenchmarkVerifierConfigurationError, match="evolution"):
        BenchmarkVerifierProcess(
            command=("run-evolution-check",),
            cwd=str(tmp_path),
            read_only_database_url="postgresql://readonly@db.example/hms",
        )
    with pytest.raises(BenchmarkVerifierConfigurationError, match="custom benchmark environment"):
        BenchmarkVerifierProcess(
            command=(sys.executable, "-c", "pass"),
            cwd=str(tmp_path),
            read_only_database_url="postgresql://readonly@db.example/hms",
            environment={"PGOPTIONS": ""},
        )

    with pytest.raises(BenchmarkVerifierConfigurationError, match="custom benchmark environment"):
        BenchmarkVerifierProcess(
            command=(sys.executable, "-c", "pass"),
            cwd=str(tmp_path),
            read_only_database_url="postgresql://readonly@db.example/hms",
            environment={"RSI_BENCHMARK_INPUT_API_URL": "http://live-hms"},
        )
