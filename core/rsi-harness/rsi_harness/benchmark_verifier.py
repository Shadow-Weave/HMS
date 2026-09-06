"""Isolated benchmark verifier process.

The benchmark verifier is deliberately a different process contract from the
development verifier. It receives a read-only database URL, enables
PostgreSQL's transaction-level read-only mode, and never returns stdout,
stderr, evaluator text, or benchmark metrics to the evolution controller.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from .benchmark import BenchmarkResult, EvaluationLease


class BenchmarkVerifierConfigurationError(ValueError):
    """Raised when a benchmark runner is not isolated enough to start."""


@dataclass(frozen=True, slots=True)
class BenchmarkVerifierProcess:
    """Run a benchmark command with a physically read-only DB connection."""

    command: tuple[str, ...]
    cwd: str
    read_only_database_url: str
    timeout_seconds: int = 1800
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.command:
            raise BenchmarkVerifierConfigurationError("benchmark verifier command must not be empty")
        lowered = " ".join(self.command).lower()
        if any(term in lowered for term in ("development", "evolution", "promote", "retain")):
            raise BenchmarkVerifierConfigurationError("benchmark verifier cannot invoke evolution or write paths")
        if self.timeout_seconds <= 0:
            raise BenchmarkVerifierConfigurationError("timeout_seconds must be positive")
        if not Path(self.cwd).is_dir():
            raise BenchmarkVerifierConfigurationError(f"benchmark verifier cwd does not exist: {self.cwd}")
        parsed = urlparse(self.read_only_database_url)
        if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
            raise BenchmarkVerifierConfigurationError("read_only_database_url must be a PostgreSQL URL")
        username = (parsed.username or "").lower()
        if not any(marker in username for marker in ("readonly", "read_only", "benchmark")):
            raise BenchmarkVerifierConfigurationError(
                "read_only_database_url must use a dedicated readonly/benchmark database role"
            )
        # Do not accept arbitrary child environment injection.  Even a
        # seemingly harmless custom variable can point the benchmark command
        # at a live HMS API, a writable database, or evaluator output.  Input
        # configuration belongs in the benchmark fixture mounted at ``cwd``.
        if self.environment:
            forbidden = next(iter(self.environment))
            raise BenchmarkVerifierConfigurationError(
                f"custom benchmark environment key is forbidden; use an isolated fixture file: {forbidden}"
            )

    async def evaluate(self, lease: EvaluationLease) -> BenchmarkResult:
        return_code = await asyncio.to_thread(self._run, lease)
        return BenchmarkResult(lease.evaluation_id, return_code == 0, {})

    def _safe_environment(self, lease: EvaluationLease) -> dict[str, str]:
        allowed = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PGOPTIONS": "-c default_transaction_read_only=on",
            "HMS_API_DATABASE_URL": self.read_only_database_url,
            "HMS_RSI_BENCHMARK_MODE": "1",
            "HMS_RSI_EVOLUTION_DISABLED": "1",
            "RSI_BENCHMARK_EVALUATION_ID": lease.evaluation_id,
            "RSI_BENCHMARK_SNAPSHOT_ID": lease.snapshot_id,
            "RSI_BENCHMARK_LEASE_TOKEN": lease.lease_token,
            "RSI_BENCHMARK_MEMORY_SNAPSHOT_ID": lease.memory_snapshot_id or lease.snapshot_id,
            "RSI_BENCHMARK_SKILL_SNAPSHOT_ID": lease.skill_snapshot_id or "",
        }
        return allowed

    def _run(self, lease: EvaluationLease) -> int:
        try:
            completed = subprocess.run(
                self.command,
                cwd=self.cwd,
                env=self._safe_environment(lease),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124
        except OSError:
            return 127
        return int(completed.returncode)
