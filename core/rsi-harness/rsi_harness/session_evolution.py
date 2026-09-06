"""Generic chronological session evolution pipeline.

This module deliberately stops at the development verifier boundary.  It can
prepare and stage candidates from any session-shaped data source, but it
cannot promote a candidate without an opaque verifier receipt.  LongMemEval,
chat sessions, and non-coding agents therefore share the same lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

from .backend import MemoryBackend
from .candidate_factory import CandidateFactory
from .evolution import DevelopmentVerifier, EvolutionSession, RunSeal
from .leakage import LeakageGuard
from .memory_orchestrator import MemoryOrchestrator
from .models import (
    MemoryCandidate,
    PromotionResult,
    RunContext,
    VerifierReceipt,
)
from .session_models import QueryInput, SessionEvent, SessionInput
from .session_runner import ChronologicalSessionRunner
from .snapshots import InMemorySnapshotStore, SnapshotStore


class SessionEvolutionStateError(RuntimeError):
    """Raised when a session evolution run is used out of order."""


@dataclass(frozen=True, slots=True)
class SessionEvolutionTrace:
    """The sealed, pre-verifier artifact for one development run.

    The trace contains only source sessions, bounded events, and candidates.
    It intentionally has no answer, judge output, benchmark score, or
    verifier explanation.
    """

    run: RunContext
    sessions: tuple[SessionInput, ...]
    events: tuple[SessionEvent, ...]
    candidates: tuple[MemoryCandidate, ...]
    seal: RunSeal
    query: QueryInput | None = None


@dataclass(frozen=True, slots=True)
class SessionEvolutionResult:
    """Opaque outcome returned after development verification."""

    trace: SessionEvolutionTrace
    receipt: VerifierReceipt
    promotion: PromotionResult | None
    snapshot_id: str | None


class SessionEvolutionVerifier(DevelopmentVerifier):
    """Verifier protocol with an optional trace binding hook."""

    def bind_trace(self, trace: SessionEvolutionTrace) -> None:
        """Bind a pre-verifier trace before :meth:`verify` is called."""

        raise NotImplementedError


@dataclass(slots=True)
class SessionEvolutionRunner:
    """Run chronological replay, candidate staging, and gated promotion.

    ``prepare`` is the only method that sees source sessions.  ``finalize``
    hands the sealed trace to a development verifier and passes only its
    opaque receipt to :class:`EvolutionSession`.  The runner has no benchmark
    code path and never accepts a gold answer.
    """

    run: RunContext
    backend: MemoryBackend
    snapshots: SnapshotStore = field(default_factory=InMemorySnapshotStore)
    leakage_guard: LeakageGuard = field(default_factory=LeakageGuard)
    session_runner: ChronologicalSessionRunner | None = None
    orchestrator: MemoryOrchestrator | None = None
    evolution: EvolutionSession | None = None
    event_enricher: EventEnricher | None = None
    max_observation_chars: int = 2_000
    _trace: SessionEvolutionTrace | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        phase = str(self.run.metadata.get("phase", "")).strip().lower()
        if phase in {"benchmark", "evaluation", "test"}:
            raise SessionEvolutionStateError("session evolution is disabled for benchmark/evaluation runs")
        self.session_runner = self.session_runner or ChronologicalSessionRunner(
            self.run,
            max_observation_chars=self.max_observation_chars,
        )
        self.orchestrator = self.orchestrator or MemoryOrchestrator(
            self.run,
            backend=self.backend,
            candidate_factory=CandidateFactory(self.leakage_guard),
            current_snapshot_id=self.run.harness_snapshot,
        )
        self.evolution = self.evolution or EvolutionSession(
            self.run,
            self.backend,
            leakage_guard=self.leakage_guard,
            snapshots=self.snapshots,
        )

    @property
    def trace(self) -> SessionEvolutionTrace | None:
        return self._trace

    async def prepare(
        self,
        sessions: Sequence[SessionInput],
        *,
        query: QueryInput | None = None,
    ) -> SessionEvolutionTrace:
        """Replay the allowed session prefix and stage all generated candidates."""

        if self._trace is not None:
            raise SessionEvolutionStateError("session evolution run is already prepared")
        if query is not None:
            if query.stream_id != self.run.stream_id:
                raise SessionEvolutionStateError("query belongs to a different stream")
            if query.query_position < 0:
                raise SessionEvolutionStateError("query_position must be non-negative")

        ordered_sessions = tuple(sessions)
        query_position = query.query_position if query is not None else None
        events = await self.session_runner.replay(ordered_sessions, query_position=query_position)

        candidates: list[MemoryCandidate] = []
        seen_candidate_ids: set[str] = set()
        enriched_events: list[SessionEvent] = []
        for raw_event in events:
            event = raw_event
            if self.event_enricher is not None:
                enriched = self.event_enricher(raw_event)
                if inspect.isawaitable(enriched):
                    enriched = await enriched
                self._validate_enriched_event(raw_event, enriched)
                event = enriched
            enriched_events.append(event)
            generated = await self.orchestrator.handle(event)
            for candidate in generated:
                if candidate.candidate_id in seen_candidate_ids:
                    continue
                await self.evolution.stage(candidate)
                seen_candidate_ids.add(candidate.candidate_id)
                candidates.append(candidate)

        processed_sessions = tuple(
            session
            for session in ordered_sessions
            if query_position is None or session.source_position < query_position
        )
        last_session = processed_sessions[-1] if processed_sessions else None
        self.evolution.snapshot_manifest.update(
            {
                "memory_complete": True,
                "stream_id": self.run.stream_id,
                "last_session_id": last_session.session_id if last_session else None,
                "last_source_position": last_session.source_position if last_session else -1,
                "event_count": len(events),
                "candidate_count": len(candidates),
                "query_position": query_position,
                "memory_scope": "query_prefix" if query is not None else "stream",
            }
        )
        seal = self.evolution.seal()
        self._trace = SessionEvolutionTrace(
            run=self.run,
            sessions=processed_sessions,
            events=tuple(enriched_events),
            candidates=tuple(candidates),
            seal=seal,
            query=query,
        )
        return self._trace

    @staticmethod
    def _validate_enriched_event(raw: SessionEvent, enriched: SessionEvent) -> None:
        if not isinstance(enriched, SessionEvent):
            raise SessionEvolutionStateError("event enricher must return SessionEvent")
        identity_fields = ("run_id", "stream_id", "session_id", "sequence", "source_position")
        if any(getattr(raw, field) != getattr(enriched, field) for field in identity_fields):
            raise SessionEvolutionStateError("event enricher cannot change source identity or order")
        if not set(raw.evidence_refs).issubset(set(enriched.evidence_refs)):
            raise SessionEvolutionStateError("event enricher cannot remove source evidence references")

    async def finalize(self, verifier: DevelopmentVerifier) -> SessionEvolutionResult:
        """Verify the sealed trace and promote only after a passing receipt."""

        if self._trace is None:
            raise SessionEvolutionStateError("prepare the session run before finalization")
        bind_trace = getattr(verifier, "bind_trace", None)
        if bind_trace is not None:
            bind_trace(self._trace)
        receipt = await verifier.verify(self.run, self._trace.seal)
        validate_receipt = getattr(verifier, "validate_receipt", None)
        if receipt.passed and callable(validate_receipt):
            try:
                receipt_is_attested = bool(validate_receipt(self._trace.seal, receipt))
            except Exception:
                receipt_is_attested = False
            if not receipt_is_attested:
                # Preserve the opaque token for audit/quarantine, while
                # preventing a forged positive result from reaching a backend.
                receipt = VerifierReceipt(
                    run_id=receipt.run_id,
                    passed=False,
                    receipt_token=receipt.receipt_token,
                    created_at=receipt.created_at,
                )
        promotion = await self.evolution.finalize(receipt)
        return SessionEvolutionResult(
            trace=self._trace,
            receipt=receipt,
            promotion=promotion,
            snapshot_id=self.evolution.last_snapshot_id,
        )


@dataclass(frozen=True, slots=True)
class DevelopmentStreamArtifact:
    """Verifier input that contains development data but no evaluator output."""

    trace: SessionEvolutionTrace


DevelopmentEvaluator = Callable[[DevelopmentStreamArtifact], bool | Awaitable[bool]]
EventEnricher = Callable[[SessionEvent], SessionEvent | Awaitable[SessionEvent]]


@dataclass(slots=True)
class DevelopmentStreamVerifier(SessionEvolutionVerifier):
    """In-process development verifier with a one-use opaque receipt.

    Production deployments can replace this class with
    :class:`DevelopmentVerifierProcess`.  This implementation exists for
    deterministic fixtures and local integration tests.  The evaluator may
    inspect development-only data, but its return channel is reduced to a
    boolean and an HMAC token.
    """

    evaluator: DevelopmentEvaluator
    receipt_secret: bytes
    _artifacts: dict[str, DevelopmentStreamArtifact] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.receipt_secret:
            raise ValueError("receipt_secret is required")

    def bind_trace(self, trace: SessionEvolutionTrace) -> None:
        if trace.run.run_id in self._artifacts:
            raise SessionEvolutionStateError("a development trace is already bound for this run")
        self._artifacts[trace.run.run_id] = DevelopmentStreamArtifact(trace)

    async def verify(self, run: RunContext, seal: RunSeal) -> VerifierReceipt:
        artifact = self._artifacts.pop(run.run_id, None)
        passed = False
        if artifact is not None and artifact.trace.seal == seal:
            try:
                outcome = self.evaluator(artifact)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                passed = type(outcome) is bool and outcome
            except Exception:
                # The verifier boundary deliberately collapses evaluator
                # failures into a failed opaque receipt.
                passed = False
        return VerifierReceipt(run.run_id, passed, self._receipt_token(run.run_id, seal, passed))

    def validate_receipt(self, seal: RunSeal, receipt: VerifierReceipt) -> bool:
        """Validate a passing receipt at the promotion boundary."""

        if not receipt.passed or receipt.run_id != seal.run_id:
            return False
        expected = self._receipt_token(seal.run_id, seal, True)
        return hmac.compare_digest(expected, receipt.receipt_token)

    def _receipt_token(self, run_id: str, seal: RunSeal, passed: bool) -> str:
        # Match the server-side HMS receipt contract: return code 0 denotes a
        # passing verifier and return code 1 denotes a failed verifier.
        return_code = 0 if passed else 1
        payload = f"development:{run_id}:{seal.candidate_digest}:{return_code}".encode("utf-8")
        return hmac.new(self.receipt_secret, payload, hashlib.sha256).hexdigest()


def run_session_evolution(
    runner: SessionEvolutionRunner,
    sessions: Sequence[SessionInput],
    verifier: DevelopmentVerifier,
    *,
    query: QueryInput | None = None,
) -> SessionEvolutionResult:
    """Synchronous convenience wrapper for small development fixtures."""

    async def _run() -> SessionEvolutionResult:
        await runner.prepare(sessions, query=query)
        return await runner.finalize(verifier)

    return asyncio.run(_run())
