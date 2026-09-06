"""Memory-specific policy separated from runtime Worker steering."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .backend import MemoryBackend
from .candidate_factory import CandidateFactory
from .models import MemoryCandidate, MemoryHit, MemoryNamespace, RunContext
from .session_models import (
    MemoryDecision,
    MemoryDecisionType,
    SessionEvent,
    SessionEventType,
)


@dataclass(frozen=True, slots=True)
class MemoryOrchestratorContext:
    run: RunContext
    current_snapshot_id: str | None = None
    recent_events: tuple[SessionEvent, ...] = ()
    open_conflicts: tuple[str, ...] = ()


class MemoryOrchestratorPolicy(Protocol):
    async def decide(
        self,
        event: SessionEvent,
        *,
        memories: Sequence[MemoryHit],
        context: MemoryOrchestratorContext,
    ) -> MemoryDecision: ...


class RuleBasedMemoryOrchestratorPolicy:
    """Conservative baseline policy for deterministic offline replay."""

    async def decide(
        self,
        event: SessionEvent,
        *,
        memories: Sequence[MemoryHit],
        context: MemoryOrchestratorContext,
    ) -> MemoryDecision:
        del memories, context
        if event.event_type in {SessionEventType.FACT_OBSERVED, SessionEventType.PREFERENCE_OBSERVED}:
            return MemoryDecision(MemoryDecisionType.PROPOSE_CANDIDATE, "explicit fact or preference observation")
        if event.event_type == SessionEventType.KNOWLEDGE_UPDATE_DETECTED:
            return MemoryDecision(MemoryDecisionType.SUPERSEDE, "new observation updates an existing value")
        if event.event_type == SessionEventType.CONTRADICTION_DETECTED:
            return MemoryDecision(MemoryDecisionType.KEEP_CONFLICT, "conflicting observations require provenance")
        return MemoryDecision(MemoryDecisionType.NOOP)


@dataclass(slots=True)
class MemoryOrchestrator:
    """Turn ordered session events into pre-verifier candidates.

    This class never promotes or invalidates memory.  It can only retrieve a
    bounded context, ask a policy for a decision, and return candidates for an
    EvolutionSession to stage.
    """

    run: RunContext
    policy: MemoryOrchestratorPolicy = field(default_factory=RuleBasedMemoryOrchestratorPolicy)
    candidate_factory: CandidateFactory = field(default_factory=CandidateFactory)
    backend: MemoryBackend | None = None
    memory_token_budget: int = 800
    current_snapshot_id: str | None = None
    recent_event_limit: int = 12
    _recent_events: list[SessionEvent] = field(default_factory=list, init=False, repr=False)
    _open_conflicts: list[str] = field(default_factory=list, init=False, repr=False)
    _last_position: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.memory_token_budget <= 0:
            raise ValueError("memory_token_budget must be positive")
        if self.recent_event_limit <= 0:
            raise ValueError("recent_event_limit must be positive")
    async def handle(self, event: SessionEvent) -> tuple[MemoryCandidate, ...]:
        self._validate_event(event)
        self._recent_events.append(event)
        del self._recent_events[:-self.recent_event_limit]
        if event.event_type == SessionEventType.CONTRADICTION_DETECTED:
            self._open_conflicts.append(event.session_id)
            del self._open_conflicts[:-self.recent_event_limit]
        memories = await self._retrieve(event)
        decision = await self.policy.decide(
            event,
            memories=memories,
            context=MemoryOrchestratorContext(
                run=self.run,
                current_snapshot_id=self.current_snapshot_id,
                recent_events=tuple(self._recent_events),
                open_conflicts=tuple(self._open_conflicts),
            ),
        )
        return self.candidate_factory.from_event(self.run, event, decision)

    def _validate_event(self, event: SessionEvent) -> None:
        if event.run_id != self.run.run_id:
            raise ValueError("session event belongs to a different run")
        if self.run.stream_id is not None and event.stream_id != self.run.stream_id:
            raise ValueError("session event belongs to a different stream")
        if self._last_position is not None and event.source_position < self._last_position:
            raise ValueError("session event source_position moved backwards")
        self._last_position = event.source_position

    async def _retrieve(self, event: SessionEvent) -> tuple[MemoryHit, ...]:
        if self.backend is None or not event.observation.strip():
            return ()
        capabilities = self.backend.capabilities
        if not capabilities.search:
            return ()
        result = self.backend.search(
            MemoryNamespace.GENERAL,
            event.observation,
            token_budget=self.memory_token_budget,
            filters={"stream_id": event.stream_id, "snapshot_id": self.current_snapshot_id}
            if self.current_snapshot_id
            else {"stream_id": event.stream_id},
        )
        if inspect.isawaitable(result):
            result = await result
        return tuple(result)
