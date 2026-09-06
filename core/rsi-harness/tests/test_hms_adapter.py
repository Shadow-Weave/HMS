from __future__ import annotations

import asyncio

import pytest
from rsi_harness import MemoryCandidate, MemoryNamespace
from rsi_harness.backend import BackendCapabilityError
from rsi_harness.hms_adapter import HMSMemoryAdapter


class FakeHMSClient:
    def __init__(self) -> None:
        self.recall_calls = []
        self.stage_calls = []

    def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return {
            "results": [
                {
                    "id": "m-1",
                    "text": "all_reduce 前不要重复加入 bias",
                    "score": 0.9,
                    "metadata": {"rsi_status": "validated"},
                }
            ]
        }

    def stage_rsi_candidate(self, bank_id, payload):
        self.stage_calls.append((bank_id, payload))
        return {"candidate_id": payload["candidate_id"]}


class SnapshotHMSClient(FakeHMSClient):
    def search_rsi_snapshot(self, bank_id, snapshot_id, namespace, query, *, limit):
        return {"items": [{"id": "m-1", "text": "frozen", "namespace": namespace, "score": 1.0}]}

    def fetch_rsi_snapshot_memory(self, bank_id, snapshot_id, memory_id):
        return {"id": memory_id, "text": "frozen", "namespace": "general", "metadata": {}}


class LeakySnapshotHMSClient(SnapshotHMSClient):
    def search_rsi_snapshot(self, bank_id, snapshot_id, namespace, query, *, limit):
        return {"items": [{"id": "secret", "text": "transcript", "namespace": "episode", "score": 1.0}]}


def test_hms_adapter_maps_recall_and_isolated_candidate_stage() -> None:
    client = FakeHMSClient()
    adapter = HMSMemoryAdapter(client, bank_id="repo-1")

    hits = asyncio.run(adapter.search(MemoryNamespace.GENERAL, "bias all_reduce", token_budget=500))
    assert hits[0].memory_id == "m-1"
    assert hits[0].namespace is MemoryNamespace.GENERAL
    assert client.recall_calls[0]["max_tokens"] == 500

    candidate = MemoryCandidate(
        candidate_id="candidate-1",
        kind="procedure",
        namespace=MemoryNamespace.CANDIDATE,
        trigger="row parallel output",
        content="add bias once after all_reduce",
        source_run_id="run-1",
        evidence_refs=("trace://run-1/3",),
    )
    operation_id = asyncio.run(adapter.stage(candidate))
    assert operation_id == "candidate-1"
    assert client.stage_calls[0][1]["created_before_verifier"] is True
    assert client.stage_calls[0][1]["candidate_id"] == "candidate-1"


def test_hms_adapter_fails_closed_without_snapshot_capability() -> None:
    adapter = HMSMemoryAdapter(FakeHMSClient(), bank_id="repo-1")
    with pytest.raises(BackendCapabilityError, match="snapshot"):
        asyncio.run(adapter.snapshot(MemoryNamespace.GENERAL))


def test_hms_adapter_never_falls_back_to_live_retain_for_candidates() -> None:
    class RetainOnlyClient:
        def retain(self, **kwargs):
            return {"operation_id": "live-write"}

    adapter = HMSMemoryAdapter(RetainOnlyClient(), bank_id="repo-1")
    candidate = MemoryCandidate(
        candidate_id="candidate-unsafe",
        kind="procedure",
        namespace=MemoryNamespace.CANDIDATE,
        trigger="x",
        content="y",
        source_run_id="run-1",
        evidence_refs=("trace://run-1/1",),
    )
    with pytest.raises(BackendCapabilityError, match="isolated RSI"):
        asyncio.run(adapter.stage(candidate))


def test_hms_adapter_snapshot_reads_reject_unsafe_namespaces() -> None:
    adapter = HMSMemoryAdapter(SnapshotHMSClient(), bank_id="repo-1")
    hits = asyncio.run(
        adapter.search(
            MemoryNamespace.GENERAL,
            "x",
            token_budget=10,
            filters={"snapshot_id": "snap-1", "read_only": True},
        )
    )
    assert hits[0].namespace is MemoryNamespace.GENERAL
    with pytest.raises(BackendCapabilityError, match="episode or candidate"):
        asyncio.run(
            adapter.search(
                MemoryNamespace.EPISODE,
                "x",
                token_budget=10,
                filters={"snapshot_id": "snap-1", "read_only": True},
            )
        )
    with pytest.raises(BackendCapabilityError, match="episode or candidate"):
        asyncio.run(
            HMSMemoryAdapter(LeakySnapshotHMSClient(), bank_id="repo-1").search(
                MemoryNamespace.GENERAL,
                "x",
                token_budget=10,
                filters={"snapshot_id": "snap-1", "read_only": True},
            )
        )


def test_hms_adapter_snapshot_fetch_is_explicit() -> None:
    adapter = HMSMemoryAdapter(SnapshotHMSClient(), bank_id="repo-1")
    item = asyncio.run(adapter.fetch_snapshot("snap-1", "m-1"))
    assert item.version == "snap-1"
