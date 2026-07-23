"""Cross-version contract checks for the generated Python client."""

import inspect
from typing import Any

from pydantic import BaseModel, ConfigDict

from hms_client_api.api.monitoring_api import MonitoringApi
from hms_client_api.models.features_info import FeaturesInfo
from hms_client_api.models.operation_status_response import OperationStatusResponse
from hms_client_api.models.version_response import VersionResponse


class _LegacyFeaturesInfo(BaseModel):
    """Frozen shape of the pre-multimodal generated Python model."""

    model_config = ConfigDict(extra="ignore")

    observations: bool
    mcp: bool
    worker: bool
    bank_config_api: bool
    file_upload_api: bool


class _LegacyVersionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_version: str
    features: _LegacyFeaturesInfo


class _LegacyOperationStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operation_id: str
    status: str
    operation_type: str | None = None
    result_metadata: dict[str, Any] | None = None


_OLD_FEATURES = {
    "observations": False,
    "mcp": True,
    "worker": True,
    "bank_config_api": False,
    "file_upload_api": True,
}


def test_new_client_reads_old_version_with_multimodal_defaults_false() -> None:
    version = VersionResponse.from_dict({"api_version": "0.6.1", "features": _OLD_FEATURES})

    assert version is not None
    assert version.features.multimodal_image is False
    assert version.features.multimodal_video is False
    assert version.features.multimodal_live_verified is False

    direct_features = FeaturesInfo.from_dict(_OLD_FEATURES)
    assert direct_features is not None
    assert direct_features.to_dict() == {
        **_OLD_FEATURES,
        "multimodal_image": False,
        "multimodal_video": False,
        "multimodal_live_verified": False,
    }


def test_new_client_reads_old_operation_metadata_and_preserves_legacy_keys() -> None:
    operation = OperationStatusResponse.from_dict(
        {
            "operation_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed",
            "operation_type": "file_convert_retain",
            "result_metadata": {"original_filename": "notes.txt", "legacy_debug_key": 7},
        }
    )

    assert operation is not None
    assert operation.result_metadata is not None
    assert operation.result_metadata.multimodal is None
    assert operation.result_metadata.additional_properties == {
        "original_filename": "notes.txt",
        "legacy_debug_key": 7,
    }
    assert operation.to_dict()["result_metadata"] == {
        "original_filename": "notes.txt",
        "legacy_debug_key": 7,
    }


def test_new_client_reads_typed_multimodal_metadata_and_future_legacy_keys() -> None:
    operation = OperationStatusResponse.from_dict(
        {
            "operation_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "future_operation_key": {"additive": True},
                "multimodal": {
                    "asset_sha256": "a" * 64,
                    "media_kind": "image",
                    "stage": "recall_ready",
                    "child_retain_operation_id": "00000000-0000-0000-0000-000000000002",
                    "child_retain_status": "completed",
                    "recall_ready": True,
                    "retryable": False,
                },
            },
        }
    )

    assert operation is not None and operation.result_metadata is not None
    multimodal = operation.result_metadata.multimodal
    assert multimodal is not None
    assert multimodal.media_kind == "image"
    assert multimodal.stage == "recall_ready"
    assert multimodal.recall_ready is True
    assert operation.result_metadata.additional_properties == {"future_operation_key": {"additive": True}}
    assert operation.to_dict()["result_metadata"]["multimodal"]["asset_sha256"] == "a" * 64


def test_frozen_old_python_client_ignores_opt_in_new_server_fields() -> None:
    version = _LegacyVersionResponse.model_validate(
        {
            "api_version": "0.6.1",
            "features": {
                **_OLD_FEATURES,
                "multimodal_image": True,
                "multimodal_video": False,
                "multimodal_live_verified": False,
            },
        }
    )
    operation = _LegacyOperationStatusResponse.model_validate(
        {
            "operation_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed",
            "operation_type": "file_convert_retain",
            "result_metadata": {
                "multimodal": {
                    "media_kind": "image",
                    "stage": "recall_ready",
                    "recall_ready": True,
                }
            },
        }
    )

    assert version.features.file_upload_api is True
    assert operation.result_metadata == {
        "multimodal": {"media_kind": "image", "stage": "recall_ready", "recall_ready": True}
    }


def test_generated_monitoring_api_exposes_opt_in_capability_query() -> None:
    assert "include_multimodal" in inspect.signature(MonitoringApi.get_version).parameters


def test_generated_monitoring_api_serializes_only_explicit_opt_in_query() -> None:
    class CapturingApiClient:
        def __init__(self) -> None:
            self.parameters = None

        @staticmethod
        def select_header_accept(values):
            return values[0]

        def param_serialize(self, **parameters):
            self.parameters = parameters
            return parameters

    client = CapturingApiClient()
    api = MonitoringApi(client)

    api._get_version_serialize(True, None, None, None, 0)
    assert client.parameters is not None
    assert client.parameters["query_params"] == [("include_multimodal", True)]

    api._get_version_serialize(None, None, None, None, 0)
    assert client.parameters["query_params"] == []
