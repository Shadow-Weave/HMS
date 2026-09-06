"""Deterministic candidate construction from pre-verifier session evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .leakage import LeakageGuard
from .models import MemoryCandidate, MemoryNamespace, RunContext
from .session_models import MemoryDecision, MemoryDecisionType, SessionEvent


@dataclass(frozen=True, slots=True)
class CandidateFactory:
    """Build at most one auditable candidate from one semantic event.

    The factory does not infer hidden facts.  A semantic extractor must place a
    structured ``candidate`` mapping in the event metadata before this method
    is called.  Missing or incomplete metadata produces no candidate.
    """

    leakage_guard: LeakageGuard = LeakageGuard()

    def from_event(
        self,
        run: RunContext,
        event: SessionEvent,
        decision: MemoryDecision,
    ) -> tuple[MemoryCandidate, ...]:
        if decision.decision not in {
            MemoryDecisionType.PROPOSE_CANDIDATE,
            MemoryDecisionType.SUPERSEDE,
            MemoryDecisionType.KEEP_CONFLICT,
        }:
            return ()
        raw = event.metadata.get("candidate") or decision.metadata.get("candidate")
        if not isinstance(raw, Mapping):
            return ()
        content = str(raw.get("content") or "").strip()
        if not content:
            return ()
        evidence = raw.get("evidence_refs") or event.evidence_refs
        if isinstance(evidence, str):
            evidence = (evidence,)
        evidence_refs = tuple(str(ref).strip() for ref in evidence if str(ref).strip())
        if not evidence_refs:
            return ()
        namespace = self._namespace(raw.get("namespace", "general"))
        kind = str(raw.get("kind") or self._kind_for(decision.decision)).strip()
        trigger = str(raw.get("trigger") or event.event_type.value).strip()
        if not kind or not trigger:
            return ()
        candidate_id = self._candidate_id(run.run_id, event, kind, content)
        metadata: dict[str, Any] = {
            "stream_id": event.stream_id,
            "session_id": event.session_id,
            "source_position": event.source_position,
            "event_sequence": event.sequence,
            "event_type": event.event_type.value,
            "orchestrator_decision": decision.decision.value,
            **dict(raw.get("metadata") or {}),
        }
        for key in ("entity_id", "observed_at", "valid_from", "valid_to", "supersedes_memory_id"):
            if key in raw:
                metadata[key] = raw[key]
        candidate = MemoryCandidate(
            candidate_id=candidate_id,
            kind=kind,
            namespace=namespace,
            trigger=trigger,
            content=content,
            source_run_id=run.run_id,
            evidence_refs=evidence_refs,
            preconditions=self._string_tuple(raw.get("preconditions")),
            anti_patterns=self._string_tuple(raw.get("anti_patterns")),
            source_commit=run.commit_sha,
            metadata=metadata,
        )
        self.leakage_guard.require_accepted(candidate)
        return (candidate,)

    @staticmethod
    def _candidate_id(run_id: str, event: SessionEvent, kind: str, content: str) -> str:
        digest = hashlib.sha256(
            f"{run_id}\0{event.stream_id}\0{event.session_id}\0{event.sequence}\0{kind}\0{content}".encode(
                "utf-8"
            )
        ).hexdigest()[:32]
        return f"session-candidate-{digest}"

    @staticmethod
    def _namespace(value: Any) -> MemoryNamespace:
        try:
            namespace = MemoryNamespace(str(value))
        except ValueError as exc:
            raise ValueError("candidate namespace must be candidate, project, or general") from exc
        if namespace not in (MemoryNamespace.CANDIDATE, MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise ValueError("candidate namespace must be candidate, project, or general")
        return namespace

    @staticmethod
    def _kind_for(decision: MemoryDecisionType) -> str:
        return {
            MemoryDecisionType.SUPERSEDE: "temporal_update",
            MemoryDecisionType.KEEP_CONFLICT: "contradiction",
        }.get(decision, "fact")

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        return tuple(str(item).strip() for item in value if str(item).strip())
