"""Immutable logical snapshot store used by the P1 evolution flow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .models import HarnessSnapshot


class SnapshotStore:
    def create(self, *, parent_snapshot_id: str | None, manifest: Mapping[str, object]) -> HarnessSnapshot:
        raise NotImplementedError

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        raise NotImplementedError


def _json_value(value: Any) -> Any:
    """Convert frozen manifest containers into deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=str)
    return value


def _snapshot_id(parent_snapshot_id: str | None, manifest: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {"parent_snapshot_id": parent_snapshot_id, "manifest": _json_value(manifest)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"harness-{digest}"


@dataclass
class InMemorySnapshotStore(SnapshotStore):
    """Reference implementation used by tests and local experiments."""

    _snapshots: dict[str, HarnessSnapshot]

    def __init__(self) -> None:
        self._snapshots = {}

    def create(self, *, parent_snapshot_id: str | None, manifest: Mapping[str, object]) -> HarnessSnapshot:
        snapshot_id = _snapshot_id(parent_snapshot_id, manifest)
        existing = self._snapshots.get(snapshot_id)
        if existing is not None:
            return existing
        snapshot = HarnessSnapshot(snapshot_id, parent_snapshot_id, manifest=dict(manifest))
        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"unknown harness snapshot: {snapshot_id}") from exc


@dataclass(slots=True)
class JsonSnapshotStore(SnapshotStore):
    """Small atomic JSON store for local development skill snapshots."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, snapshot_id: str) -> Path:
        if not snapshot_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in snapshot_id):
            raise ValueError("snapshot_id contains unsupported filesystem characters")
        return self.root / f"{snapshot_id}.json"

    def create(self, *, parent_snapshot_id: str | None, manifest: Mapping[str, object]) -> HarnessSnapshot:
        snapshot_id = _snapshot_id(parent_snapshot_id, manifest)
        canonical = json.dumps(
            {"parent_snapshot_id": parent_snapshot_id, "manifest": _json_value(manifest)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        path = self._path(snapshot_id)
        if path.exists():
            snapshot = self.get(snapshot_id)
            existing_canonical = json.dumps(
                {
                    "parent_snapshot_id": snapshot.parent_snapshot_id,
                    "manifest": _json_value(snapshot.manifest),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if existing_canonical != canonical:
                raise ValueError(f"snapshot '{snapshot_id}' already exists with different content")
            return snapshot
        snapshot = HarnessSnapshot(snapshot_id, parent_snapshot_id, manifest=dict(manifest))
        payload = {
            "snapshot_id": snapshot.snapshot_id,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "created_at": snapshot.created_at.isoformat(),
            "manifest": _json_value(snapshot.manifest),
        }
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return snapshot

    def get(self, snapshot_id: str) -> HarnessSnapshot:
        path = self._path(snapshot_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(f"unknown harness snapshot: {snapshot_id}") from exc
        try:
            if str(payload["snapshot_id"]) != snapshot_id:
                raise ValueError("snapshot ID does not match its filename")
            created_at = datetime.fromisoformat(str(payload["created_at"]))
            snapshot = HarnessSnapshot(
                snapshot_id,
                payload.get("parent_snapshot_id"),
                created_at=created_at,
                manifest=dict(payload.get("manifest") or {}),
            )
            if _snapshot_id(snapshot.parent_snapshot_id, snapshot.manifest) != snapshot.snapshot_id:
                raise ValueError("snapshot content digest does not match snapshot ID")
            return snapshot
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid snapshot file: {path}") from exc
