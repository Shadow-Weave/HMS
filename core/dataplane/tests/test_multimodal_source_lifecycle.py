"""Fault-injection tests for verified multimodal source lifecycle state."""

from __future__ import annotations

import pytest

from hms_api.engine.multimodal.source_lifecycle import (
    SourceAvailability,
    resolve_source_lifecycle,
)


class _Storage:
    def __init__(self, *, available: bool = True, delete_mode: str = "normal", exists_error: bool = False) -> None:
        self.available = available
        self.delete_mode = delete_mode
        self.exists_error = exists_error
        self.delete_calls = 0
        self.exists_calls = 0

    async def delete(self, key: str) -> None:
        assert key == "opaque/source/key"
        self.delete_calls += 1
        if self.delete_mode == "response_lost":
            self.available = False
            raise TimeoutError("synthetic response loss")
        if self.delete_mode == "success_but_remains":
            return
        if self.delete_mode == "failed":
            raise RuntimeError("synthetic delete failure")
        self.available = False

    async def exists(self, key: str) -> bool:
        assert key == "opaque/source/key"
        self.exists_calls += 1
        if self.exists_error:
            raise PermissionError("synthetic existence failure")
        return self.available


@pytest.mark.asyncio
async def test_response_lost_delete_is_classified_from_actual_absence() -> None:
    storage = _Storage(delete_mode="response_lost")

    result = await resolve_source_lifecycle(storage, "opaque/source/key", delete_requested=True)

    assert result.availability is SourceAvailability.DELETED
    assert result.delete_raised is True
    assert result.metadata_value == "false"
    assert result.metric_state == "deleted"
    assert result.reason is None
    assert storage.delete_calls == storage.exists_calls == 1


@pytest.mark.asyncio
async def test_successful_delete_response_does_not_hide_remaining_object() -> None:
    storage = _Storage(delete_mode="success_but_remains")

    result = await resolve_source_lifecycle(storage, "opaque/source/key", delete_requested=True)

    assert result.availability is SourceAvailability.AVAILABLE
    assert result.delete_raised is False
    assert result.metadata_value == "true"
    assert result.metric_state == "retained"
    assert result.reason == "source.delete_incomplete"
    assert storage.delete_calls == storage.exists_calls == 1


@pytest.mark.asyncio
async def test_failed_delete_with_remaining_object_reports_actual_availability() -> None:
    storage = _Storage(delete_mode="failed")

    result = await resolve_source_lifecycle(storage, "opaque/source/key", delete_requested=True)

    assert result.availability is SourceAvailability.AVAILABLE
    assert result.metadata_value == "true"
    assert result.reason == "source.delete_failed"


@pytest.mark.asyncio
async def test_unavailable_existence_check_never_defaults_to_available() -> None:
    storage = _Storage(exists_error=True)

    result = await resolve_source_lifecycle(storage, "opaque/source/key", delete_requested=False)

    assert result.availability is SourceAvailability.UNKNOWN
    assert result.metadata_value == "unknown"
    assert result.metric_state == "unknown"
    assert result.reason == "source.availability_unverified"
    assert storage.delete_calls == 0
    assert storage.exists_calls == 1
