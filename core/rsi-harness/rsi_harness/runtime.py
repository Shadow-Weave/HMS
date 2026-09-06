"""P2 live event bus and supervisor controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .backend import MemoryBackend
from .events import EventType, SupervisorAction, SupervisorActionType, WorkerEvent
from .models import MemoryHit, MemoryNamespace, RunContext


class WorkerRuntime(Protocol):
    """Runtime bridge implemented by Codex, Claude, LangGraph, or another agent."""

    async def apply(self, action: SupervisorAction) -> None: ...


class SupervisorPolicy(Protocol):
    """Decision policy kept independent from the event transport and memory DB."""

    async def decide(
        self,
        event: WorkerEvent,
        *,
        memories: Sequence[MemoryHit],
        context: "SupervisorContext",
    ) -> SupervisorAction | None: ...


@dataclass(frozen=True, slots=True)
class SupervisorContext:
    run: RunContext
    current_goal: str
    recent_events: tuple[WorkerEvent, ...] = ()
    open_risks: tuple[str, ...] = ()


@dataclass
class InMemoryEventBus:
    """Minimal in-process transport for local runs and adapter tests."""

    _events: asyncio.Queue[WorkerEvent] = field(default_factory=asyncio.Queue)
    _actions: dict[str, asyncio.Queue[SupervisorAction]] = field(default_factory=dict)

    async def publish(self, event: WorkerEvent) -> None:
        await self._events.put(event)

    async def receive(self) -> WorkerEvent:
        return await self._events.get()

    async def publish_action(self, action: SupervisorAction) -> None:
        queue = self._actions.setdefault(action.worker_id, asyncio.Queue())
        await queue.put(action)

    async def receive_action(self, worker_id: str) -> SupervisorAction:
        queue = self._actions.setdefault(worker_id, asyncio.Queue())
        return await queue.get()


class EventProtocolError(ValueError):
    """Raised when an adapter violates event ordering or run isolation."""


class SupervisorController:
    """Observe worker events, retrieve bounded memory, and apply actions.

    The controller never sees the raw verifier result.  It only operates on
    worker events and memory snapshots supplied by the runtime.
    """

    _RETRIEVAL_EVENTS = frozenset({EventType.QUESTION, EventType.RISK, EventType.ERROR, EventType.INACTIVITY})

    def __init__(
        self,
        run: RunContext,
        *,
        backend: MemoryBackend,
        policy: SupervisorPolicy,
        worker: WorkerRuntime,
        current_goal: str,
        memory_namespace: MemoryNamespace = MemoryNamespace.GENERAL,
        memory_token_budget: int = 1200,
        memory_filters: Mapping[str, object] | None = None,
        recent_event_limit: int = 8,
    ) -> None:
        if memory_token_budget <= 0:
            raise ValueError("memory_token_budget must be positive")
        if recent_event_limit <= 0:
            raise ValueError("recent_event_limit must be positive")
        self.run = run
        self.backend = backend
        self.policy = policy
        self.worker = worker
        self.current_goal = current_goal
        self.memory_namespace = memory_namespace
        self.memory_token_budget = memory_token_budget
        self.memory_filters = dict(memory_filters or {})
        self.recent_event_limit = recent_event_limit
        self._last_sequence: dict[str, int] = {}
        self._recent_events: list[WorkerEvent] = []
        self._open_risks: list[str] = []
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    async def handle(self, event: WorkerEvent) -> SupervisorAction | None:
        if self._settled:
            raise EventProtocolError("cannot handle events after the run has settled")
        self._validate_event(event)
        self._remember(event)

        memories: Sequence[MemoryHit] = ()
        if event.event_type in self._RETRIEVAL_EVENTS:
            query = self._query_for(event)
            memories = await self.backend.search(
                self.memory_namespace,
                query,
                token_budget=self.memory_token_budget,
                filters=self.memory_filters,
            )

        context = SupervisorContext(
            run=self.run,
            current_goal=self.current_goal,
            recent_events=tuple(self._recent_events),
            open_risks=tuple(self._open_risks),
        )
        action = await self.policy.decide(event, memories=memories, context=context)
        if action is not None:
            if action.run_id != self.run.run_id or action.worker_id != event.worker_id:
                raise EventProtocolError("supervisor action crosses run or worker boundary")
            await self.worker.apply(action)
            if action.action_type == SupervisorActionType.ABORT:
                self._settled = True
        if event.event_type == EventType.RESULT:
            self._settled = True
        return action

    def _validate_event(self, event: WorkerEvent) -> None:
        if event.run_id != self.run.run_id:
            raise EventProtocolError("event belongs to a different run")
        previous = self._last_sequence.get(event.worker_id)
        if previous is not None and event.sequence <= previous:
            raise EventProtocolError(
                f"event sequence must increase for worker {event.worker_id}: {event.sequence} <= {previous}"
            )
        self._last_sequence[event.worker_id] = event.sequence

    def _remember(self, event: WorkerEvent) -> None:
        self._recent_events.append(event)
        del self._recent_events[:-self.recent_event_limit]
        if event.risk:
            self._open_risks.append(event.risk)
            del self._open_risks[:-self.recent_event_limit]

    def _query_for(self, event: WorkerEvent) -> str:
        fields = (
            self.current_goal,
            event.goal_delta,
            event.observation,
            event.attempt,
            event.outcome,
            event.risk,
            event.next_step,
        )
        return "\n".join(part for part in fields if part.strip())


async def serve_events(controller: SupervisorController, bus: InMemoryEventBus) -> None:
    """Consume events until the controller settles or the task is cancelled."""

    while not controller.settled:
        event = await bus.receive()
        await controller.handle(event)
