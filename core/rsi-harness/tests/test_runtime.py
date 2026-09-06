from __future__ import annotations

import asyncio

import pytest
from rsi_harness import (
    EventProtocolError,
    EventType,
    MemoryBackendCapabilities,
    MemoryHit,
    MemoryNamespace,
    RunContext,
    SupervisorAction,
    SupervisorActionType,
    SupervisorController,
    WorkerEvent,
)


class Backend:
    capabilities = MemoryBackendCapabilities()

    def __init__(self):
        self.queries = []

    async def search(self, namespace, query, *, token_budget, filters=None):
        self.queries.append((namespace, query, token_budget, filters))
        return [MemoryHit("m-1", "general recovery", 0.9, namespace)]


class Worker:
    def __init__(self):
        self.actions = []

    async def apply(self, action):
        self.actions.append(action)


class Policy:
    def __init__(self, action=None, action_event_type=EventType.RISK):
        self.action = action
        self.action_event_type = action_event_type
        self.calls = []

    async def decide(self, event, *, memories, context):
        self.calls.append((event, tuple(memories), context))
        return self.action if event.event_type == self.action_event_type else None


def event(seq, event_type=EventType.PROGRESS):
    return WorkerEvent(
        "run-1",
        "worker-1",
        seq,
        event_type,
        observation="tests failed" if event_type != EventType.PROGRESS else "started",
        risk="stalled" if event_type == EventType.RISK else "",
    )


def make_controller(policy=None):
    worker = Worker()
    backend = Backend()
    controller = SupervisorController(
        RunContext("run-1", "task-1"),
        backend=backend,
        policy=policy or Policy(),
        worker=worker,
        current_goal="repair the repository",
        memory_token_budget=300,
    )
    return controller, backend, worker


def test_controller_retrieves_only_on_risk_events_and_applies_steer():
    action = SupervisorAction("run-1", "worker-1", SupervisorActionType.STEER, instruction="try the alternate path")
    policy = Policy(action)
    controller, backend, worker = make_controller(policy)

    asyncio.run(controller.handle(event(0)))
    assert backend.queries == []
    asyncio.run(controller.handle(event(1, EventType.RISK)))
    assert len(backend.queries) == 1
    assert backend.queries[0][2] == 300
    assert worker.actions == [action]


def test_controller_rejects_out_of_order_or_cross_run_events():
    controller, _, _ = make_controller()
    asyncio.run(controller.handle(event(2)))
    with pytest.raises(EventProtocolError, match="sequence"):
        asyncio.run(controller.handle(event(1)))
    with pytest.raises(EventProtocolError, match="different run"):
        asyncio.run(
            controller.handle(
                WorkerEvent("other-run", "worker-1", 3, EventType.PROGRESS, observation="bad")
            )
        )


def test_result_settles_controller_and_abort_settles_immediately():
    controller, _, _ = make_controller()
    asyncio.run(controller.handle(event(0, EventType.RESULT)))
    assert controller.settled

    action = SupervisorAction("run-1", "worker-1", SupervisorActionType.ABORT, reason="no longer useful")
    controller, _, _ = make_controller(Policy(action))
    asyncio.run(controller.handle(event(0, EventType.RISK)))
    assert controller.settled


def test_controller_rejects_events_after_result():
    controller, _, _ = make_controller()
    asyncio.run(controller.handle(event(0, EventType.RESULT)))
    with pytest.raises(EventProtocolError, match="settled"):
        asyncio.run(controller.handle(event(1)))
