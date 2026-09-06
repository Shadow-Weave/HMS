from __future__ import annotations

import pytest

from benchmarks.longmemeval.frozen_snapshot_backend import (
    FrozenSnapshotDatabaseError,
    PostgresFrozenSnapshotBackend,
)


def test_frozen_backend_requires_a_dedicated_readonly_role():
    with pytest.raises(FrozenSnapshotDatabaseError, match="readonly"):
        PostgresFrozenSnapshotBackend("postgresql://hms_writer@db.example/hms", "bank-1")


def test_frozen_backend_rejects_unsafe_schema_identifier():
    with pytest.raises(FrozenSnapshotDatabaseError, match="schema"):
        PostgresFrozenSnapshotBackend(
            "postgresql://hms_readonly@db.example/hms",
            "bank-1",
            schema="public;DROP TABLE memory_units",
        )


def test_frozen_backend_advertises_no_mutation_capabilities():
    backend = PostgresFrozenSnapshotBackend("postgresql://hms_readonly@db.example/hms", "bank-1")
    assert backend.capabilities.immutable_snapshots
    assert backend.capabilities.stage is False
    assert backend.capabilities.promote is False
    assert backend._table("rsi_snapshot_memory_units") == '"public"."rsi_snapshot_memory_units"'
