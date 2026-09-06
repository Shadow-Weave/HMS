"""Development-only verifier process with an opaque result channel."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .evolution import DevelopmentVerifier, RunSeal
from .models import RunContext, VerifierReceipt


class VerifierConfigurationError(ValueError):
    """Raised when a verifier command violates development-only policy."""


def validate_development_receipt(receipt_secret: bytes, seal: RunSeal, receipt: VerifierReceipt) -> bool:
    """Validate a passing receipt without revealing verifier output."""

    if not receipt.passed or not receipt.receipt_token or receipt.run_id != seal.run_id:
        return False
    payload = f"development:{seal.run_id}:{seal.candidate_digest}:0".encode("utf-8")
    expected = hmac.new(receipt_secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, receipt.receipt_token)


@dataclass(frozen=True, slots=True)
class DevelopmentVerifierProcess(DevelopmentVerifier):
    """Run a development verifier without exposing its output to evolution.

    The command receives a deliberately small environment allowlist. stdout,
    stderr, and test names are discarded. Only an opaque receipt token and a
    boolean pass/fail value leave this process.
    """

    command: tuple[str, ...]
    cwd: str
    timeout_seconds: int = 900
    environment: Mapping[str, str] = field(default_factory=dict)
    receipt_secret: bytes = b""

    _PROTECTED_ENVIRONMENT = frozenset(
        {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "RSI_VERIFIER_MODE",
            "RSI_RUN_ID",
            "RSI_CANDIDATE_DIGEST",
        }
    )

    def __post_init__(self) -> None:
        if not self.command:
            raise VerifierConfigurationError("development verifier command must not be empty")
        if any("benchmark" in part.lower() for part in self.command):
            raise VerifierConfigurationError("benchmark verifier commands are forbidden in development verifier")
        if self.timeout_seconds <= 0:
            raise VerifierConfigurationError("timeout_seconds must be positive")
        if not Path(self.cwd).is_dir():
            raise VerifierConfigurationError(f"verifier cwd does not exist: {self.cwd}")
        for key in self.environment:
            if str(key).upper() in self._PROTECTED_ENVIRONMENT | {
                "HMS_API_TOKEN",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "BENCHMARK",
                "EVALUATOR",
            }:
                raise VerifierConfigurationError(f"sensitive or benchmark environment key is forbidden: {key}")
        if not self.receipt_secret:
            raise VerifierConfigurationError("receipt_secret is required for server-verifiable receipts")

    async def verify(self, run: RunContext, seal: RunSeal) -> VerifierReceipt:
        env = self._safe_environment(run, seal)
        return_code = await asyncio.to_thread(self._run, env)
        token = self._receipt_token(run.run_id, seal.candidate_digest, return_code)
        return VerifierReceipt(run.run_id, return_code == 0, token)

    def _safe_environment(self, run: RunContext, seal: RunSeal) -> dict[str, str]:
        allowed = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "RSI_VERIFIER_MODE": "development",
            "RSI_RUN_ID": run.run_id,
            "RSI_CANDIDATE_DIGEST": seal.candidate_digest,
        }
        for key, value in self.environment.items():
            if key.upper() in self._PROTECTED_ENVIRONMENT | {
                "HMS_API_TOKEN",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "BENCHMARK",
                "EVALUATOR",
            }:
                raise VerifierConfigurationError(f"sensitive or benchmark environment key is forbidden: {key}")
            allowed[str(key)] = str(value)
        return allowed

    def _run(self, env: Mapping[str, str]) -> int:
        try:
            completed = subprocess.run(
                self.command,
                cwd=self.cwd,
                env=dict(env),
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

    def _receipt_token(self, run_id: str, digest: str, return_code: int) -> str:
        secret = self.receipt_secret or secrets.token_bytes(32)
        payload = f"development:{run_id}:{digest}:{return_code}".encode("utf-8")
        return hmac.new(secret, payload, hashlib.sha256).hexdigest()

    def validate_receipt(self, seal: RunSeal, receipt: VerifierReceipt) -> bool:
        return validate_development_receipt(self.receipt_secret, seal, receipt)
