from __future__ import annotations

import asyncio

import pytest
from rsi_harness import (
    BenchmarkAccessViolation,
    BenchmarkFirewall,
    EvaluationSnapshotBuilder,
    EvaluationSnapshotError,
    FrozenBenchmarkBoundaryError,
    FrozenBenchmarkRunner,
    HarnessSnapshot,
    MemoryBackendCapabilities,
    MemoryHit,
    MemoryNamespace,
    QueryInput,
)
from rsi_harness.backend import BackendCapabilityError


class Backend:
    capabilities = MemoryBackendCapabilities(immutable_snapshots=True)

    def __init__(self):
        self.queries = []

    async def search(self, namespace, query, *, token_budget, filters=None):
        self.queries.append((namespace, query, token_budget, filters))
        return [MemoryHit("m-1", "safe frozen memory", 1.0, namespace)]

    async def fetch(self, memory_id, *, detail="summary"):
        raise AssertionError("live fetch must never be used by benchmark view")


class LiveFetchOnlyBackend(Backend):
    capabilities = MemoryBackendCapabilities(fetch=True, immutable_snapshots=True)


def test_benchmark_fetch_rejects_backend_without_explicit_snapshot_fetch():
    firewall = BenchmarkFirewall(benchmark_name="held-out")
    lease = firewall.open(snapshot(), evolution_sealed=True)
    with pytest.raises(BackendCapabilityError, match="snapshot fetch"):
        asyncio.run(firewall.view(LiveFetchOnlyBackend(), lease).fetch("m-1"))

    async def fetch(self, memory_id, *, detail="summary"):
        raise AssertionError("benchmark view must never fetch from the live backend")


def snapshot():
    return HarnessSnapshot("harness-frozen", None, manifest={"immutable": True, "memory_complete": True})


def skill_snapshot():
    return HarnessSnapshot("skill-frozen", None, manifest={"skill_complete": True})


def test_evaluation_snapshot_binds_memory_and_skill_versions():
    evaluation = EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
        snapshot(),
        skill_snapshot(),
        evaluation_id="eval-1",
        source_dataset_revision="dataset-rev-1",
        stream_id="stream-1",
        last_session_id="session-2",
        last_source_position=1,
    )
    assert evaluation.manifest["memory_snapshot_id"] == "harness-frozen"
    assert evaluation.manifest["skill_snapshot_id"] == "skill-frozen"
    firewall = BenchmarkFirewall(benchmark_name="longmemeval")
    lease = firewall.open(evaluation, evolution_sealed=True)
    assert lease.memory_snapshot_id == "harness-frozen"
    assert lease.skill_snapshot_id == "skill-frozen"


def test_evaluation_snapshot_rejects_incomplete_skill_or_memory():
    with pytest.raises(EvaluationSnapshotError, match="memory"):
        EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
            HarnessSnapshot("incomplete", None, manifest={}),
            skill_snapshot(),
            evaluation_id="eval-1",
            source_dataset_revision="dataset-rev-1",
        )
    with pytest.raises(EvaluationSnapshotError, match="skill"):
        EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
            snapshot(),
            HarnessSnapshot("incomplete-skill", None, manifest={}),
            evaluation_id="eval-1",
            source_dataset_revision="dataset-rev-1",
        )
    with pytest.raises(EvaluationSnapshotError, match="frozen"):
        EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
            HarnessSnapshot("building", None, manifest={"memory_complete": True, "status": "building"}),
            skill_snapshot(),
            evaluation_id="eval-1",
            source_dataset_revision="dataset-rev-1",
        )


def test_frozen_benchmark_runner_reads_only_the_pinned_prefix():
    evaluation = EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
        HarnessSnapshot(
            "memory-frozen",
            None,
            manifest={"memory_complete": True, "stream_id": "stream-1", "last_source_position": 0},
        ),
        skill_snapshot(),
        evaluation_id="eval-1",
        source_dataset_revision="dataset-rev-1",
    )
    runner = FrozenBenchmarkRunner(
        BenchmarkFirewall(benchmark_name="longmemeval"),
        Backend(),
        evaluation,
        "longmemeval",
    )
    query = QueryInput("eval-1", "stream-1", "q-1", "repair", 1)

    report = asyncio.run(
        runner.run(
            (query,),
            answerer=lambda current_query, hits, view: hits[0].text,
            scorer=lambda current_query, answer, hits: answer == "safe frozen memory",
        )
    )
    assert report.result.passed is True
    assert report.result.metrics["accuracy"] == 1.0
    assert report.records[0].retrieved_memory_ids == ("m-1",)


def test_frozen_benchmark_runner_rejects_future_memory():
    evaluation = EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
        HarnessSnapshot(
            "memory-future",
            None,
            manifest={"memory_complete": True, "stream_id": "stream-1", "last_source_position": 2},
        ),
        skill_snapshot(),
        evaluation_id="eval-1",
        source_dataset_revision="dataset-rev-1",
    )
    runner = FrozenBenchmarkRunner(
        BenchmarkFirewall(benchmark_name="longmemeval"),
        Backend(),
        evaluation,
        "longmemeval",
    )
    query = QueryInput("eval-1", "stream-1", "q-1", "repair", 2)
    with pytest.raises(FrozenBenchmarkBoundaryError, match="at or after"):
        asyncio.run(
            runner.run(
                (query,),
                answerer=lambda current_query, hits, view: "answer",
                scorer=lambda current_query, answer, hits: True,
            )
        )
def test_benchmark_requires_frozen_evolution_and_only_reads_frozen_namespaces():
    firewall = BenchmarkFirewall(benchmark_name="held-out")
    with pytest.raises(BenchmarkAccessViolation, match="freeze"):
        firewall.open(snapshot(), evolution_sealed=False)

    lease = firewall.open(snapshot(), evolution_sealed=True)
    backend = Backend()
    view = firewall.view(backend, lease)
    hits = asyncio.run(view.search(MemoryNamespace.GENERAL, "repair", token_budget=200))
    assert hits[0].text == "safe frozen memory"
    assert backend.queries[0][3]["snapshot_id"] == "harness-frozen"
    with pytest.raises(BenchmarkAccessViolation, match="episode or candidate"):
        asyncio.run(view.search(MemoryNamespace.CANDIDATE, "secret", token_budget=200))


def test_benchmark_view_rejects_all_mutations():
    firewall = BenchmarkFirewall(benchmark_name="held-out")
    lease = firewall.open(snapshot(), evolution_sealed=True)
    view = firewall.view(Backend(), lease)
    with pytest.raises(BenchmarkAccessViolation, match="read-only"):
        asyncio.run(view.stage(None))
    with pytest.raises(BenchmarkAccessViolation, match="read-only"):
        asyncio.run(view.promote([], target_snapshot="other"))
    firewall.close(lease)
    with pytest.raises(BenchmarkAccessViolation, match="unknown"):
        firewall.view(Backend(), lease)


def test_benchmark_view_rejects_live_fetch_without_snapshot_fetch():
    firewall = BenchmarkFirewall(benchmark_name="held-out")
    lease = firewall.open(snapshot(), evolution_sealed=True)
    with pytest.raises(BackendCapabilityError, match="snapshot fetch"):
        asyncio.run(firewall.view(Backend(), lease).fetch("live-memory"))


def test_benchmark_view_rejects_namespace_violation_from_backend():
    class LeakyBackend(Backend):
        async def search(self, namespace, query, *, token_budget, filters=None):
            return [MemoryHit("episode", "transcript", 1.0, MemoryNamespace.EPISODE)]

    firewall = BenchmarkFirewall(benchmark_name="held-out")
    lease = firewall.open(snapshot(), evolution_sealed=True)
    with pytest.raises(BenchmarkAccessViolation, match="episode or candidate"):
        asyncio.run(firewall.view(LeakyBackend(), lease).search(MemoryNamespace.GENERAL, "x", token_budget=10))
