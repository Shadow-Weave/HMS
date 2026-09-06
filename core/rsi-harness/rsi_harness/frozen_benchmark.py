"""Backend-neutral read-only benchmark runner for frozen evaluation snapshots."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .benchmark import BenchmarkFirewall, BenchmarkResult, FrozenMemoryView
from .models import HarnessSnapshot, MemoryHit
from .session_models import QueryInput


class FrozenBenchmarkBoundaryError(PermissionError):
    """Raised when a query would read beyond its frozen session watermark."""


Answerer = Callable[[QueryInput, Sequence[MemoryHit], FrozenMemoryView], str | Awaitable[str]]
Scorer = Callable[[QueryInput, str, Sequence[MemoryHit]], bool | Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkRecord:
    query_id: str
    passed: bool
    retrieved_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkReport:
    result: BenchmarkResult
    records: tuple[FrozenBenchmarkRecord, ...]


@dataclass(slots=True)
class FrozenBenchmarkRunner:
    """Evaluate query inputs through a read-only :class:`FrozenMemoryView`.

    The answerer and scorer run inside the caller's benchmark process.  A
    production caller should keep the gold labels in that process and return
    only this report to the evolution controller.
    """

    firewall: BenchmarkFirewall
    backend: Any
    evaluation_snapshot: HarnessSnapshot
    benchmark_name: str
    token_budget: int = 800

    async def run(
        self,
        queries: Sequence[QueryInput],
        *,
        answerer: Answerer,
        scorer: Scorer,
        evolution_sealed: bool = True,
        filters: Mapping[str, object] | None = None,
    ) -> FrozenBenchmarkReport:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.evaluation_snapshot.manifest.get("benchmark_name") not in (None, self.benchmark_name):
            raise FrozenBenchmarkBoundaryError("evaluation snapshot belongs to a different benchmark")
        lease = self.firewall.open(self.evaluation_snapshot, evolution_sealed=evolution_sealed)
        view = self.firewall.view(self.backend, lease)
        records: list[FrozenBenchmarkRecord] = []
        try:
            for query in queries:
                self._validate_query_watermark(query)
                hits = await view.search(
                    self._namespace_for_query(),
                    query.query,
                    token_budget=self.token_budget,
                    filters=filters,
                )
                answer = answerer(query, hits, view)
                if inspect.isawaitable(answer):
                    answer = await answer
                passed = scorer(query, answer, hits)
                if inspect.isawaitable(passed):
                    passed = await passed
                if type(passed) is not bool:
                    raise FrozenBenchmarkBoundaryError("benchmark scorer must return a boolean")
                records.append(
                    FrozenBenchmarkRecord(
                        query_id=query.query_id,
                        passed=passed,
                        retrieved_memory_ids=tuple(hit.memory_id for hit in hits),
                    )
                )
        finally:
            self.firewall.close(lease)

        correct = sum(record.passed for record in records)
        result = BenchmarkResult(
            evaluation_id=lease.evaluation_id,
            passed=bool(records) and correct == len(records),
            metrics={
                "accuracy": correct / len(records) if records else 0.0,
                "correct": float(correct),
                "total": float(len(records)),
            },
        )
        return FrozenBenchmarkReport(result=result, records=tuple(records))

    def _validate_query_watermark(self, query: QueryInput) -> None:
        stream_id = self.evaluation_snapshot.manifest.get("stream_id")
        if stream_id is not None and query.stream_id != stream_id:
            raise FrozenBenchmarkBoundaryError("query belongs to a different stream")
        last_position = self.evaluation_snapshot.manifest.get("last_source_position")
        if isinstance(last_position, int) and last_position >= query.query_position:
            raise FrozenBenchmarkBoundaryError("frozen memory contains a session at or after query position")

    @staticmethod
    def _namespace_for_query():
        # FrozenMemoryView accepts only project/general.  General is the
        # neutral default; adapters may pass a project namespace through the
        # explicit filters in a future extension.
        from .models import MemoryNamespace

        return MemoryNamespace.GENERAL
