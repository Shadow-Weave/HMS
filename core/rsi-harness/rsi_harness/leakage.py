"""Fail-closed candidate classification and benchmark leakage checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from .models import DataClassification, LeakageReport, MemoryCandidate, MemoryNamespace


class LeakageViolation(ValueError):
    """Raised when a candidate cannot safely enter persistent memory."""


@dataclass(frozen=True, slots=True)
class LeakagePolicy:
    """Explicit allow/deny policy for candidate content."""

    forbidden_literals: tuple[str, ...] = ()
    forbidden_regexes: tuple[str, ...] = (
        r"\bgold[_ -]?patch\b",
        r"\bhidden[_ -]?test\b",
        r"\breference[_ -]?answer\b",
        r"\bverifier[_ -]?(?:output|stdout|stderr)\b",
    )
    forbidden_metadata_keys: frozenset[str] = frozenset(
        {
            "benchmark_output",
            "benchmark_score",
            "gold_patch",
            "hidden_test",
            "reference_answer",
            "verifier_output",
            "verifier_stdout",
            "verifier_stderr",
        }
    )
    sensitive_regexes: tuple[str, ...] = (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
    )
    max_content_chars: int = 12_000


@dataclass(frozen=True, slots=True)
class LeakageGuard:
    policy: LeakagePolicy = field(default_factory=LeakagePolicy)

    def inspect(self, candidate: MemoryCandidate) -> LeakageReport:
        findings: list[str] = []
        # Metadata is part of the persisted candidate and can carry the same
        # benchmark answer material as the human-facing content.  Inspect a
        # canonical representation rather than treating it as trusted control
        # data.
        content = candidate.content
        metadata_text = json.dumps(dict(candidate.metadata), sort_keys=True, default=str)
        inspected_text = f"{content}\n{metadata_text}"
        metadata = {str(key).lower() for key in candidate.metadata}

        if len(content) > self.policy.max_content_chars:
            findings.append("candidate_content_too_large")
        if candidate.namespace not in (MemoryNamespace.CANDIDATE, MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            findings.append("candidate_namespace_not_promotable")
        for key in sorted(metadata & self.policy.forbidden_metadata_keys):
            findings.append(f"forbidden_metadata:{key}")
        for literal in self.policy.forbidden_literals:
            if literal and literal in inspected_text:
                findings.append("benchmark_secret_literal")
        for pattern in self.policy.forbidden_regexes:
            if re.search(pattern, inspected_text, flags=re.IGNORECASE | re.MULTILINE):
                findings.append("benchmark_secret_pattern")
        for pattern in self.policy.sensitive_regexes:
            if re.search(pattern, inspected_text, flags=re.IGNORECASE | re.MULTILINE):
                findings.append("sensitive_secret_pattern")

        if any(item.startswith("benchmark_secret") or item.startswith("forbidden_metadata") for item in findings):
            return LeakageReport(DataClassification.BENCHMARK_SECRET, tuple(findings))
        if any(item.startswith("sensitive_") for item in findings):
            return LeakageReport(DataClassification.SENSITIVE, tuple(findings))
        if "candidate_content_too_large" in findings or "candidate_namespace_not_promotable" in findings:
            return LeakageReport(DataClassification.EPISODIC_ONLY, tuple(findings))
        if candidate.namespace == MemoryNamespace.PROJECT:
            return LeakageReport(DataClassification.SAFE_PROJECT)
        return LeakageReport(DataClassification.SAFE_GENERAL)

    def require_accepted(self, candidate: MemoryCandidate) -> LeakageReport:
        report = self.inspect(candidate)
        if not report.accepted:
            raise LeakageViolation(
                f"candidate {candidate.candidate_id} rejected as {report.classification.value}: "
                + ", ".join(report.findings)
            )
        return report

    def inspect_texts(self, texts: Iterable[str]) -> LeakageReport:
        joined = "\n".join(texts)
        if any(literal and literal in joined for literal in self.policy.forbidden_literals):
            return LeakageReport(DataClassification.BENCHMARK_SECRET, ("benchmark_secret_literal",))
        for pattern in self.policy.forbidden_regexes:
            if re.search(pattern, joined, flags=re.IGNORECASE | re.MULTILINE):
                return LeakageReport(DataClassification.BENCHMARK_SECRET, ("benchmark_secret_pattern",))
        for pattern in self.policy.sensitive_regexes:
            if re.search(pattern, joined, flags=re.IGNORECASE | re.MULTILINE):
                return LeakageReport(DataClassification.SENSITIVE, ("sensitive_secret_pattern",))
        return LeakageReport(DataClassification.SAFE_GENERAL)
