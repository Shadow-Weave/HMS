"""Build an immutable evaluation lease view from frozen memory and skill state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .models import HarnessSnapshot


class EvaluationSnapshotError(ValueError):
    """Raised when an evaluation snapshot is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class EvaluationSnapshotBuilder:
    """Compose a benchmark-only manifest without mutating an evolution store.

    ``memory_snapshot`` must already be frozen by the memory backend.  The
    builder only creates a logical evaluation envelope; the benchmark lease
    continues to read the underlying ``memory_snapshot_id`` through the
    backend's immutable snapshot API.
    """

    benchmark_name: str

    def __post_init__(self) -> None:
        if not self.benchmark_name.strip():
            raise EvaluationSnapshotError("benchmark_name must not be empty")

    def build(
        self,
        memory_snapshot: HarnessSnapshot,
        skill_snapshot: HarnessSnapshot,
        *,
        evaluation_id: str,
        source_dataset_revision: str,
        stream_id: str | None = None,
        last_session_id: str | None = None,
        last_source_position: int | None = None,
        max_event_time: str | None = None,
    ) -> HarnessSnapshot:
        if not evaluation_id.strip() or not source_dataset_revision.strip():
            raise EvaluationSnapshotError("evaluation_id and source_dataset_revision are required")
        if not memory_snapshot.manifest.get("memory_complete", False):
            raise EvaluationSnapshotError("memory snapshot is incomplete")
        memory_status = memory_snapshot.manifest.get("status")
        if memory_status is not None and str(memory_status).lower() != "frozen":
            raise EvaluationSnapshotError("memory snapshot must be frozen")
        if not skill_snapshot.manifest.get("skill_complete", False):
            raise EvaluationSnapshotError("skill snapshot is incomplete")
        if not skill_snapshot.snapshot_id.strip():
            raise EvaluationSnapshotError("skill snapshot ID must not be empty")
        resolved_stream_id = stream_id if stream_id is not None else memory_snapshot.manifest.get("stream_id")
        resolved_last_session_id = (
            last_session_id if last_session_id is not None else memory_snapshot.manifest.get("last_session_id")
        )
        resolved_last_source_position = (
            last_source_position
            if last_source_position is not None
            else memory_snapshot.manifest.get("last_source_position")
        )
        resolved_max_event_time = (
            max_event_time if max_event_time is not None else memory_snapshot.manifest.get("max_event_time")
        )
        manifest: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "benchmark_name": self.benchmark_name,
            "source_dataset_revision": source_dataset_revision,
            "memory_snapshot_id": memory_snapshot.snapshot_id,
            "skill_snapshot_id": skill_snapshot.snapshot_id,
            "memory_complete": True,
            "skill_complete": True,
            "evolution_sealed": True,
            "stream_id": resolved_stream_id,
            "last_session_id": resolved_last_session_id,
            "last_source_position": resolved_last_source_position,
            "max_event_time": resolved_max_event_time,
        }
        canonical = json.dumps(
            {
                "memory_snapshot_id": memory_snapshot.snapshot_id,
                "skill_snapshot_id": skill_snapshot.snapshot_id,
                "evaluation_id": evaluation_id,
                "source_dataset_revision": source_dataset_revision,
                "benchmark_name": self.benchmark_name,
                "stream_id": resolved_stream_id,
                "last_session_id": resolved_last_session_id,
                "last_source_position": resolved_last_source_position,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        snapshot_id = f"evaluation-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"
        return HarnessSnapshot(snapshot_id, memory_snapshot.snapshot_id, manifest=manifest)
