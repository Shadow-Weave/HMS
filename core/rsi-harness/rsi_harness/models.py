"""Immutable data contracts shared by the RSI harness and memory adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_value(value: Any) -> Any:
    """Recursively freeze manifest data so snapshots cannot be mutated via nesting."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryNamespace(StrEnum):
    """Visibility boundary for memory records.

    ``EPISODE`` and ``CANDIDATE`` must never be visible in a frozen evaluation
    snapshot.  The backend is responsible for enforcing that boundary too;
    this enum only makes it explicit at the adapter contract.
    """

    EPISODE = "episode"
    CANDIDATE = "candidate"
    PROJECT = "project"
    GENERAL = "general"


class DataClassification(StrEnum):
    SAFE_GENERAL = "safe_general"
    SAFE_PROJECT = "safe_project"
    EPISODIC_ONLY = "episodic_only"
    BENCHMARK_SECRET = "benchmark_secret"
    SENSITIVE = "sensitive"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Stable identity and provenance for one worker/supervisor run."""

    run_id: str
    task_id: str
    repo_id: str | None = None
    commit_sha: str | None = None
    branch: str | None = None
    agent_id: str | None = None
    harness_snapshot: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    stream_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.stream_id is not None and not self.stream_id.strip():
            raise ValueError("stream_id must not be empty when provided")


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A bounded retrieval result with an auditable source handle."""

    memory_id: str
    text: str
    score: float
    namespace: MemoryNamespace
    source_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty")
        if not self.text.strip():
            raise ValueError("memory hit text must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("memory hit score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """Full or expanded memory item returned by an adapter."""

    memory_id: str
    text: str
    namespace: MemoryNamespace
    source_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A candidate generated before verifier output is available.

    ``created_before_verifier`` is deliberately mandatory.  Adapters must
    reject a candidate that does not carry this temporal assertion instead of
    silently accepting post-verifier benchmark feedback.
    """

    candidate_id: str
    kind: str
    namespace: MemoryNamespace
    trigger: str
    content: str
    source_run_id: str
    evidence_refs: tuple[str, ...]
    created_at: datetime = field(default_factory=_utc_now)
    created_before_verifier: bool = True
    preconditions: tuple[str, ...] = ()
    anti_patterns: tuple[str, ...] = ()
    source_commit: str | None = None
    redaction_report: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.kind.strip():
            raise ValueError("candidate kind must not be empty")
        if not self.trigger.strip():
            raise ValueError("candidate trigger must not be empty")
        if not self.content.strip():
            raise ValueError("candidate content must not be empty")
        if not self.source_run_id.strip():
            raise ValueError("candidate source_run_id must not be empty")
        if not self.evidence_refs:
            raise ValueError("candidate must contain at least one evidence reference")
        if not self.created_before_verifier:
            raise ValueError("candidate must be created before verifier output")
        if self.namespace not in (MemoryNamespace.CANDIDATE, MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise ValueError("candidate namespace must be candidate, project, or general")


@dataclass(frozen=True, slots=True)
class HarnessSnapshot:
    """An immutable logical view of a memory/harness state."""

    snapshot_id: str
    parent_snapshot_id: str | None
    created_at: datetime = field(default_factory=_utc_now)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    writable: bool = False

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        if self.writable:
            raise ValueError("HarnessSnapshot must be immutable")
        object.__setattr__(self, "manifest", _freeze_value(self.manifest))


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Result of a backend promotion operation."""

    snapshot_id: str
    promoted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Deterministic result of candidate content inspection."""

    classification: DataClassification
    findings: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.classification in (DataClassification.SAFE_GENERAL, DataClassification.SAFE_PROJECT)


@dataclass(frozen=True, slots=True)
class VerifierReceipt:
    """Opaque binary verifier result without evaluator feedback."""

    run_id: str
    passed: bool
    receipt_token: str
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("verifier receipt run_id must not be empty")
        if not self.receipt_token.strip():
            raise ValueError("verifier receipt token must not be empty")
