"""Small HMS client adapter for the backend-neutral RSI contract.

The adapter accepts an injected client instead of importing HMS internals. It
therefore works with the existing Codex REST client, an async SDK client, or a
test double. Snapshot reads are restricted to explicitly classified project
and general memory.
"""

from __future__ import annotations

import inspect
import secrets
from typing import Any, Mapping, Sequence

from .backend import BackendCapabilityError, MemoryBackendCapabilities
from .models import HarnessSnapshot, MemoryCandidate, MemoryHit, MemoryItem, MemoryNamespace, PromotionResult


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class HMSMemoryAdapter:
    """Adapt an HMS client with ``recall`` and ``retain`` methods."""

    def __init__(self, client: Any, *, bank_id: str):
        if not bank_id.strip():
            raise ValueError("bank_id must not be empty")
        self.client = client
        self.bank_id = bank_id

    @property
    def capabilities(self) -> MemoryBackendCapabilities:
        # Snapshot support is advertised only when both read operations exist.
        return MemoryBackendCapabilities(
            search=hasattr(self.client, "recall"),
            fetch=False,
            # A normal retain writes live memory and cannot satisfy candidate
            # isolation.  Only advertise staging when the durable RSI
            # endpoint exists; silently falling back to retain would make a
            # verifier candidate visible to benchmark/live recall.
            stage=hasattr(self.client, "stage_rsi_candidate"),
            promote=hasattr(self.client, "promote_rsi_candidates"),
            quarantine=False,
            invalidate=hasattr(self.client, "invalidate_rsi_candidate"),
            immutable_snapshots=all(
                hasattr(self.client, name)
                for name in (
                    "create_rsi_snapshot",
                    "freeze_rsi_snapshot",
                    "search_rsi_snapshot",
                    "fetch_rsi_snapshot_memory",
                )
            ),
        )

    async def snapshot(self, namespace: MemoryNamespace) -> HarnessSnapshot:
        if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BackendCapabilityError("HMS snapshots only support project and general namespaces")
        if not hasattr(self.client, "create_rsi_snapshot"):
            raise BackendCapabilityError("HMS client does not provide RSI snapshot API")
        snapshot_id = f"hms-{self.bank_id}-{namespace.value}-{secrets.token_hex(8)}"
        response = await _maybe_await(
            self.client.create_rsi_snapshot(
                self.bank_id,
                snapshot_id,
                manifest={"namespace": namespace.value, "mode": "logical-building"},
            )
        )
        if response.get("status") == "building" and hasattr(self.client, "freeze_rsi_snapshot"):
            response = await _maybe_await(self.client.freeze_rsi_snapshot(self.bank_id, snapshot_id))
        manifest = dict(response.get("manifest") or {})
        if response.get("status") is not None:
            manifest.setdefault("status", response["status"])
        return HarnessSnapshot(
            snapshot_id=str(response.get("snapshot_id", snapshot_id)),
            parent_snapshot_id=response.get("parent_snapshot_id"),
            manifest=manifest,
        )

    async def search(
        self,
        namespace: MemoryNamespace,
        query: str,
        *,
        token_budget: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[MemoryHit]:
        if not self.capabilities.search:
            raise BackendCapabilityError("HMS client does not provide recall")
        if token_budget <= 0:
            return []
        filters = filters or {}
        snapshot_id = filters.get("snapshot_id")
        if snapshot_id:
            if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
                raise BackendCapabilityError("frozen HMS snapshots cannot read episode or candidate memory")
            if not hasattr(self.client, "search_rsi_snapshot"):
                raise BackendCapabilityError("HMS client does not provide frozen snapshot search")
            response = await _maybe_await(
                self.client.search_rsi_snapshot(
                    self.bank_id,
                    str(snapshot_id),
                    namespace.value,
                    query,
                    limit=max(1, min(token_budget, 100)),
                )
            )
            results = response.get("items", []) if isinstance(response, Mapping) else []
            hits: list[MemoryHit] = []
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                memory_id = str(result.get("id") or result.get("memory_id") or "").strip()
                text = str(result.get("text") or result.get("content") or "").strip()
                if not memory_id or not text:
                    continue
                try:
                    hit_namespace = MemoryNamespace(str(result.get("namespace", namespace.value)))
                except ValueError:
                    continue
                if hit_namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
                    raise BackendCapabilityError("snapshot returned episode or candidate memory")
                hits.append(
                    MemoryHit(
                        memory_id=memory_id,
                        text=text,
                        score=max(0.0, min(1.0, float(result.get("score", 0.0)))),
                        namespace=hit_namespace,
                        metadata=dict(result.get("metadata") or {}),
                    )
                )
            return hits
        if filters.get("read_only"):
            raise BackendCapabilityError("read-only HMS search requires an explicit snapshot_id")
        response = await _maybe_await(
            self.client.recall(
                bank_id=self.bank_id,
                query=query,
                max_tokens=token_budget,
                budget=str(filters.get("budget", "low")),
                types=filters.get("fact_types"),
            )
        )
        results = response.get("results", []) if isinstance(response, Mapping) else []
        hits: list[MemoryHit] = []
        for result in results:
            if not isinstance(result, Mapping):
                continue
            raw_namespace = str(result.get("namespace", namespace.value))
            try:
                hit_namespace = MemoryNamespace(raw_namespace)
            except ValueError:
                hit_namespace = namespace
            text = str(result.get("text") or result.get("content") or "").strip()
            memory_id = str(result.get("id") or result.get("memory_id") or "").strip()
            if not text or not memory_id:
                continue
            raw_score = result.get("score", 0.0)
            try:
                score = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                score = 0.0
            refs = result.get("source_refs") or result.get("sources") or ()
            if isinstance(refs, str):
                refs = (refs,)
            hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    text=text,
                    score=score,
                    namespace=hit_namespace,
                    source_refs=tuple(str(ref) for ref in refs),
                    metadata=dict(result.get("metadata") or {}),
                )
            )
        return hits

    async def fetch(self, memory_id: str, *, detail: str = "summary") -> MemoryItem:
        raise BackendCapabilityError("HMS client requires snapshot_id for immutable fetch")

    async def fetch_snapshot(self, snapshot_id: str, memory_id: str, *, detail: str = "summary") -> MemoryItem:
        if not hasattr(self.client, "fetch_rsi_snapshot_memory"):
            raise BackendCapabilityError("HMS client does not provide frozen snapshot fetch")
        result = await _maybe_await(self.client.fetch_rsi_snapshot_memory(self.bank_id, snapshot_id, memory_id))
        try:
            namespace = MemoryNamespace(str(result.get("namespace", "general")))
        except ValueError as exc:
            raise BackendCapabilityError("snapshot returned an invalid memory namespace") from exc
        if namespace not in (MemoryNamespace.PROJECT, MemoryNamespace.GENERAL):
            raise BackendCapabilityError("snapshot returned episode or candidate memory")
        return MemoryItem(
            memory_id=str(result["id"]),
            text=str(result["text"]),
            namespace=namespace,
            source_refs=tuple(str(ref) for ref in (result.get("source_memory_ids") or ())),
            metadata=dict(result.get("metadata") or {}),
            version=snapshot_id,
        )

    async def stage(self, candidate: MemoryCandidate) -> str:
        if not self.capabilities.stage:
            raise BackendCapabilityError("HMS client does not provide isolated RSI candidate staging")
        response = await _maybe_await(
            self.client.stage_rsi_candidate(
                self.bank_id,
                {
                    "candidate_id": candidate.candidate_id,
                    "run_id": candidate.source_run_id,
                    "namespace": candidate.namespace.value,
                    "kind": candidate.kind,
                    "trigger": candidate.trigger,
                    "content": candidate.content,
                    "preconditions": list(candidate.preconditions),
                    "anti_patterns": list(candidate.anti_patterns),
                    "evidence_refs": list(candidate.evidence_refs),
                    "source_commit": candidate.source_commit,
                    "created_before_verifier": True,
                    "metadata": dict(candidate.metadata),
                },
            )
        )
        # Evolution promotes by the durable candidate_id.  A retain endpoint
        # may return an asynchronous operation_id, but that identifier cannot
        # be used by the RSI candidate lifecycle.
        return candidate.candidate_id

    async def promote(self, candidate_ids: Sequence[str], *, target_snapshot: str) -> PromotionResult:
        raise BackendCapabilityError("HMS promotion requires an opaque verifier receipt")

    async def promote_with_receipt(
        self,
        candidate_ids: Sequence[str],
        *,
        target_snapshot: str,
        receipt_token: str,
        run_id: str,
        candidate_digest: str,
        parent_snapshot_id: str | None = None,
    ) -> PromotionResult:
        if not hasattr(self.client, "promote_rsi_candidates"):
            raise BackendCapabilityError("HMS client does not provide candidate promotion API")
        if hasattr(self.client, "create_rsi_snapshot"):
            await _maybe_await(
                self.client.create_rsi_snapshot(
                    self.bank_id,
                    target_snapshot,
                    parent_snapshot_id=parent_snapshot_id,
                    manifest={"candidate_ids": list(candidate_ids), "mode": "logical-building"},
                )
            )
        response = await _maybe_await(
            self.client.promote_rsi_candidates(
                self.bank_id,
                target_snapshot,
                list(candidate_ids),
                run_id,
                candidate_digest,
                receipt_token,
            )
        )
        return PromotionResult(
            snapshot_id=target_snapshot,
            promoted_candidate_ids=tuple(response.get("promoted_candidate_ids", candidate_ids)),
        )

    async def quarantine(self, candidate_ids: Sequence[str], reason: str) -> None:
        raise BackendCapabilityError("current HMS API has no quarantine contract")

    async def invalidate(self, memory_ids: Sequence[str], reason: str) -> None:
        if not hasattr(self.client, "invalidate_rsi_candidate"):
            raise BackendCapabilityError("HMS client does not provide candidate invalidation API")
        for memory_id in memory_ids:
            await _maybe_await(self.client.invalidate_rsi_candidate(self.bank_id, memory_id, reason))

    async def restore(self, snapshot_id: str) -> None:
        raise BackendCapabilityError("current HMS API has no immutable snapshot contract")
