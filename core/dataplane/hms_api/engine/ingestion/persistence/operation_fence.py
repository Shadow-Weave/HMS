"""Transaction-level activity fence for tracked Retain writes."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from ...schema import fq_table_explicit
from ..contracts import RetainOperationInactiveError

_ACTIVE_STATUSES = frozenset({"pending", "processing"})


def _metadata_object(connection: Any, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    parser = getattr(connection, "parse_json", None)
    if callable(parser):
        value = parser(value)
    elif isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise RetainOperationInactiveError("Tracked Retain operation metadata is invalid")
    return dict(value)


class OperationActivityFence:
    """Serialize a core write against cancellation of its child and parent.

    The child row is locked first and the optional parent row second, matching
    the cancellation and completion aggregation order. Holding both locks until
    the core transaction exits gives cancellation a precise linearization
    point: either cancellation commits first and this write is rejected, or
    this write commits before cancellation can be accepted.
    """

    def __init__(self, operation_id: str, *, schema: str | None = None) -> None:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a non-empty string")
        try:
            self._operation_id = uuid.UUID(operation_id)
        except (AttributeError, ValueError) as exc:
            raise ValueError("operation_id must be a UUID string") from exc
        self._schema = schema

    async def assert_active(self, connection: Any, *, bank_id: str) -> None:
        """Lock the operation chain and fail closed unless every row is active."""

        if not isinstance(bank_id, str) or not bank_id:
            raise ValueError("bank_id must be a non-empty string")
        operations = fq_table_explicit("async_operations", self._schema)
        child = await connection.fetchrow(
            f"""
            SELECT status, result_metadata
            FROM {operations}
            WHERE operation_id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            self._operation_id,
            bank_id,
        )
        if child is None or child["status"] not in _ACTIVE_STATUSES:
            raise RetainOperationInactiveError("Tracked Retain operation is no longer active")

        metadata = _metadata_object(connection, child["result_metadata"])
        parent_value = metadata.get("parent_operation_id")
        if parent_value is None:
            return
        try:
            parent_id = uuid.UUID(str(parent_value))
        except (AttributeError, ValueError) as exc:
            raise RetainOperationInactiveError("Tracked Retain parent operation metadata is invalid") from exc
        if parent_id == self._operation_id:
            raise RetainOperationInactiveError("Tracked Retain operation cannot be its own parent")

        parent = await connection.fetchrow(
            f"""
            SELECT status
            FROM {operations}
            WHERE operation_id = $1 AND bank_id = $2
            FOR UPDATE
            """,
            parent_id,
            bank_id,
        )
        if parent is None or parent["status"] not in _ACTIVE_STATUSES:
            raise RetainOperationInactiveError("Tracked Retain parent operation is no longer active")


__all__ = ["OperationActivityFence"]
