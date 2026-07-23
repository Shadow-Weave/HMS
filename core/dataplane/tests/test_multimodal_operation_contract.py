"""Typed additive operation metadata and defensive redaction tests."""

import base64
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from hms_api.api.http import (
    MultimodalOperationMetadata,
    OperationResultMetadata,
    OperationStatusResponse,
    create_app,
)


def _response(result_metadata):
    return OperationStatusResponse.model_validate(
        {
            "operation_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed",
            "operation_type": "file_convert_retain",
            "result_metadata": result_metadata,
        }
    )


def test_legacy_result_metadata_remains_extensible() -> None:
    response = _response({"original_filename": "notes.txt", "legacy_debug_key": 7})

    assert response.result_metadata is not None
    assert isinstance(response.result_metadata, OperationResultMetadata)
    assert response.result_metadata.multimodal is None
    assert response.model_dump(exclude_none=True)["result_metadata"] == {
        "original_filename": "notes.txt",
        "legacy_debug_key": 7,
    }


def test_multimodal_namespace_has_typed_attributes_and_round_trips() -> None:
    sha256 = "a" * 64
    response = _response(
        {
            "original_filename": "screen.png",
            "multimodal": {
                "asset_id": "asset_scoped",
                "asset_sha256": sha256,
                "media_kind": "image",
                "pipeline_version": "hms-multimodal-v1",
                "descriptor_model": "gpt-5-mini",
                "stage": "recall_ready",
                "child_retain_operation_id": "00000000-0000-0000-0000-000000000002",
                "child_retain_status": "completed",
                "recall_ready": True,
                "retryable": False,
                "logical_calls": 2,
                "physical_attempts": 3,
            },
        }
    )

    assert response.result_metadata is not None
    multimodal = response.result_metadata.multimodal
    assert isinstance(multimodal, MultimodalOperationMetadata)
    assert multimodal.asset_sha256 == sha256
    assert multimodal.media_kind == "image"
    assert multimodal.stage == "recall_ready"
    assert multimodal.recall_ready is True
    assert multimodal.logical_calls == 2
    assert response.model_dump(exclude_none=True)["result_metadata"] == {
        "multimodal": {
            "asset_id": "asset_scoped",
            "asset_sha256": sha256,
            "media_kind": "image",
            "pipeline_version": "hms-multimodal-v1",
            "descriptor_model": "gpt-5-mini",
            "stage": "recall_ready",
            "child_retain_operation_id": "00000000-0000-0000-0000-000000000002",
            "child_retain_status": "completed",
            "recall_ready": True,
            "retryable": False,
            "logical_calls": 2,
            "physical_attempts": 3,
        },
        "original_filename": "screen.png",
    }


def test_result_metadata_schema_types_only_the_public_namespace() -> None:
    schema = OperationStatusResponse.model_json_schema()
    assert "result_metadata" not in set(schema.get("required", []))
    metadata_schema = schema["$defs"]["OperationResultMetadata"]
    multimodal_schema = schema["$defs"]["MultimodalOperationMetadata"]
    assert metadata_schema["additionalProperties"] is True
    assert metadata_schema["properties"]["multimodal"]["anyOf"][0]["$ref"] == ("#/$defs/MultimodalOperationMetadata")
    assert multimodal_schema["additionalProperties"] is False
    assert not multimodal_schema.get("required")
    for counter in ("input_tokens", "output_tokens", "logical_calls", "physical_attempts"):
        counter_schema = multimodal_schema["properties"][counter]["anyOf"][0]
        assert counter_schema["format"] == "int64"

    with pytest.raises(ValueError, match="signed int64"):
        MultimodalOperationMetadata(input_tokens=9_223_372_036_854_775_808)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_result", "expected"),
    [
        (
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "processing",
                "operation_type": "retain",
                "created_at": "2026-07-23T00:00:00Z",
            },
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "processing",
                "operation_type": "retain",
                "created_at": "2026-07-23T00:00:00Z",
                "updated_at": None,
                "completed_at": None,
                "error_message": None,
                "retry_count": None,
                "next_retry_at": None,
                "result_metadata": None,
                "child_operations": None,
                "task_payload": None,
            },
        ),
        (
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "not_found",
            },
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "not_found",
                "operation_type": None,
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
                "error_message": None,
                "retry_count": None,
                "next_retry_at": None,
                "result_metadata": None,
                "child_operations": None,
                "task_payload": None,
            },
        ),
        (
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "completed",
                "operation_type": "batch_retain",
                "result_metadata": {"legacy_debug_key": 7},
                "child_operations": [
                    {
                        "operation_id": "00000000-0000-0000-0000-000000000002",
                        "status": "completed",
                    }
                ],
            },
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "status": "completed",
                "operation_type": "batch_retain",
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
                "error_message": None,
                "retry_count": None,
                "next_retry_at": None,
                "result_metadata": {"legacy_debug_key": 7},
                "child_operations": [
                    {
                        "operation_id": "00000000-0000-0000-0000-000000000002",
                        "status": "completed",
                        "sub_batch_index": None,
                        "items_count": None,
                        "error_message": None,
                    }
                ],
                "task_payload": None,
            },
        ),
    ],
)
async def test_operation_status_route_preserves_legacy_non_multimodal_wire(stored_result, expected) -> None:
    """Typing the child namespace must not omit or inject legacy fields."""

    operation_id = stored_result["operation_id"]
    get_operation_status = AsyncMock(return_value=stored_result)
    memory = SimpleNamespace(audit_logger=None, get_operation_status=get_operation_status)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/default/banks/legacy-test/operations/{operation_id}")

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.asyncio
async def test_operation_status_route_redacts_sensitive_invalid_multimodal_metadata(caplog) -> None:
    """Malformed persisted metadata must not be reflected by the public route or logs."""

    operation_id = "00000000-0000-0000-0000-000000000001"
    base64_sentinel = "Uk9VVEVfU0VDUkVUX01BUktFUg=="
    data_url_sentinel = f"data:image/png;base64,{base64_sentinel}"
    get_operation_status = AsyncMock(
        return_value={
            "operation_id": operation_id,
            "status": "failed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "multimodal": {
                    "stage": "failed",
                    # Historical rows can contain malformed internal metadata.
                    # The route must not publish a transport payload even
                    # though result_metadata intentionally remains map-shaped.
                    "data_url": data_url_sentinel,
                }
            },
        }
    )
    memory = SimpleNamespace(audit_logger=None, get_operation_status=get_operation_status)
    app = create_app(memory, initialize_memory=False)

    caplog.set_level(logging.ERROR, logger="hms_api.api.http")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/default/banks/security-test/operations/{operation_id}")

    assert response.status_code == 200
    assert response.json()["result_metadata"]["multimodal"] == {"stage": "failed"}
    assert data_url_sentinel not in response.text
    assert base64_sentinel not in response.text
    assert data_url_sentinel not in caplog.text
    assert base64_sentinel not in caplog.text
    get_operation_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_operation_status_route_rejects_encoded_payload_in_allowlisted_text_field() -> None:
    """An allowed key must not become a payload-smuggling channel."""

    operation_id = "00000000-0000-0000-0000-000000000001"
    encoded_payload = base64.b64encode(b"allowlisted-field-media-sentinel" * 16).decode("ascii")
    get_operation_status = AsyncMock(
        return_value={
            "operation_id": operation_id,
            "status": "failed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "multimodal": {
                    "stage": encoded_payload,
                    "sanitized_error_code": "provider.schema_invalid",
                }
            },
        }
    )
    memory = SimpleNamespace(audit_logger=None, get_operation_status=get_operation_status)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/default/banks/security-test/operations/{operation_id}")

    assert response.status_code == 200
    assert response.json()["result_metadata"]["multimodal"] == {"sanitized_error_code": "provider.schema_invalid"}
    assert encoded_payload not in response.text


@pytest.mark.asyncio
async def test_operation_status_route_drops_invalid_historical_typed_values_and_returns_200() -> None:
    """Bad historical enums, digests, UUIDs, and error text must not break polling."""

    operation_id = "00000000-0000-0000-0000-000000000001"
    get_operation_status = AsyncMock(
        return_value={
            "operation_id": operation_id,
            "status": "failed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "original_filename": "historical.bin",
                "legacy_debug_key": 7,
                "multimodal": {
                    "asset_id": "asset_scoped",
                    "asset_sha256": "NOT-A-LOWERCASE-SHA256",
                    "media_kind": "audio",
                    "stage": "future_unapproved_stage",
                    "child_retain_operation_id": "not-a-uuid",
                    "child_retain_status": "zombie",
                    "sanitized_error_code": "provider rejected: raw body\nsecret",
                    "recall_ready": False,
                    "retryable": True,
                    "physical_attempts": 2,
                },
            },
        }
    )
    memory = SimpleNamespace(audit_logger=None, get_operation_status=get_operation_status)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/default/banks/security-test/operations/{operation_id}")

    assert response.status_code == 200
    result_metadata = response.json()["result_metadata"]
    assert result_metadata["original_filename"] == "historical.bin"
    assert result_metadata["legacy_debug_key"] == 7
    assert result_metadata["multimodal"] == {
        "asset_id": "asset_scoped",
        "recall_ready": False,
        "retryable": True,
        "physical_attempts": 2,
    }
    get_operation_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_operation_status_route_fails_closed_on_contradictory_ready_metadata() -> None:
    """A corrupt row cannot claim ready without the completed-child tuple."""

    operation_id = "00000000-0000-0000-0000-000000000001"
    child_id = "00000000-0000-0000-0000-000000000002"
    get_operation_status = AsyncMock(
        return_value={
            "operation_id": operation_id,
            "status": "failed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "multimodal": {
                    "stage": "failed",
                    "child_retain_operation_id": child_id,
                    "child_retain_status": "failed",
                    "recall_ready": True,
                    "input_tokens": 9_223_372_036_854_775_808,
                }
            },
        }
    )
    memory = SimpleNamespace(audit_logger=None, get_operation_status=get_operation_status)
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/default/banks/security-test/operations/{operation_id}")

    assert response.status_code == 200
    assert response.json()["result_metadata"]["multimodal"] == {
        "stage": "failed",
        "child_retain_operation_id": child_id,
        "child_retain_status": "failed",
        "recall_ready": False,
    }
