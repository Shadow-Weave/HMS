from __future__ import annotations

import asyncio
import sys

import pytest
from rsi_harness import DevelopmentVerifierProcess, RunContext, RunSeal, VerifierConfigurationError


def test_development_verifier_returns_opaque_receipt_and_discards_output(tmp_path):
    verifier = DevelopmentVerifierProcess(
        command=(sys.executable, "-c", "print('hidden test output')"),
        cwd=str(tmp_path),
        receipt_secret=b"fixed-test-secret",
    )
    receipt = asyncio.run(verifier.verify(RunContext("run-1", "task-1"), RunSeal("run-1", (), "digest")))
    assert receipt.passed
    assert receipt.receipt_token
    assert "hidden" not in receipt.receipt_token


def test_development_verifier_rejects_benchmark_command_and_secrets(tmp_path):
    with pytest.raises(VerifierConfigurationError, match="benchmark"):
        DevelopmentVerifierProcess(command=("run-benchmark",), cwd=str(tmp_path))
    with pytest.raises(VerifierConfigurationError, match="environment key"):
        DevelopmentVerifierProcess(
            command=(sys.executable, "-c", "0"),
            cwd=str(tmp_path),
            environment={"HMS_API_TOKEN": "secret"},
        )
