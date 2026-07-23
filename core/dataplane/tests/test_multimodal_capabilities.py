"""Import isolation and the additive multimodal capability contract."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import hms_api.config as config_module
from hms_api.api.http import create_app


def test_legacy_engine_import_does_not_load_optional_media_stack():
    """A clean text process must not import PyAV or the vision parser."""

    code = """
import sys
import hms_api.engine.memory_engine
assert 'av' not in sys.modules
assert 'hms_api.engine.multimodal.video' not in sys.modules
assert 'hms_api.engine.parsers.openai_multimodal' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _version_only_app():
    # Version handling only needs the audit_logger attribute during app setup;
    # no MemoryEngine method or lifespan initialization is exercised here.
    memory = SimpleNamespace(audit_logger=None)
    return create_app(memory, initialize_memory=False)


def _load_openapi_generator_module():
    script_path = Path(__file__).resolve().parents[3] / "lab" / "evaluation" / "hms_dev" / "generate_openapi.py"
    spec = importlib.util.spec_from_file_location("hms_generate_openapi_contract", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk_json(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def test_openapi_generation_is_deterministic_and_keeps_sdk_compatibility_views(tmp_path) -> None:
    """The 3.0 SDK projection must never weaken or replace canonical 3.1."""

    generator = _load_openapi_generator_module()
    canonical_a = tmp_path / "canonical-a.json"
    canonical_b = tmp_path / "canonical-b.json"
    compatibility = tmp_path / "compatibility-30.json"

    generator.generate_openapi_spec(str(canonical_a))
    generator.generate_openapi_spec(str(canonical_b))
    generator.generate_openapi_spec(str(compatibility), compatibility_openapi_30=True)

    assert canonical_a.read_bytes() == canonical_b.read_bytes()
    canonical_bytes = canonical_a.read_bytes()
    canonical = json.loads(canonical_bytes)
    compat = json.loads(compatibility.read_bytes())
    assert canonical_a.read_bytes() == canonical_bytes
    assert canonical["openapi"] == "3.1.0"
    assert compat["openapi"] == "3.0.3"
    assert len(canonical["paths"]) == len(compat["paths"]) == 50

    file_items = canonical["components"]["schemas"]["Body_file_retain"]["properties"]["files"]["items"]
    assert file_items == {
        "type": "string",
        "contentMediaType": "application/octet-stream",
        "format": "binary",
    }
    compat_file_items = compat["components"]["schemas"]["Body_file_retain"]["properties"]["files"]["items"]
    assert compat_file_items["format"] == "binary"
    assert "contentMediaType" not in compat_file_items
    assert "properties" in canonical["components"]["schemas"]["MultimodalOperationMetadata"]
    assert "multimodal" in canonical["components"]["schemas"]["OperationResultMetadata"]["properties"]
    assert not any(isinstance(node, dict) and node.get("type") == "null" for node in _walk_json(compat))
    assert any(isinstance(node, dict) and node.get("nullable") is True for node in _walk_json(compat))


def test_video_decoder_capability_requires_release_qualified_h264(monkeypatch) -> None:
    """An importable PyAV build without H.264 must not be advertised."""

    import hms_api.engine.multimodal.video as video_module

    class PyAVWithoutH264:
        @staticmethod
        def Codec(name: str, mode: str):
            assert (name, mode) == ("h264", "r")
            raise ValueError("codec unavailable")

    monkeypatch.setattr(video_module, "_av", PyAVWithoutH264())
    assert video_module.video_decoder_available() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "?include_multimodal=false"])
async def test_version_defaults_to_the_exact_legacy_wire_shape(monkeypatch, query: str) -> None:
    """Strict old clients see no additive fields unless they opt in."""

    config = replace(
        config_module._get_raw_config(),
        database_backend="postgresql",
        enable_file_upload_api=True,
        multimodal_enabled=True,
        multimodal_image_enabled=True,
        multimodal_video_enabled=True,
        multimodal_live_verified=True,
        multimodal_capability_responses_api=True,
        multimodal_capability_image_input=True,
        multimodal_capability_structured_outputs=True,
    )
    monkeypatch.setattr(config_module, "_get_raw_config", lambda: config)
    app = _version_only_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/version{query}")

    assert response.status_code == 200
    assert response.json()["features"] == {
        "observations": config.enable_observations,
        "mcp": config.mcp_enabled,
        "worker": config.worker_enabled,
        "bank_config_api": config.enable_bank_config_api,
        "file_upload_api": config.enable_file_upload_api,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("video_enabled", "decoder_available", "expected"),
    [
        (False, False, (True, False, True)),
        (True, False, (True, False, False)),
        (True, True, (True, True, True)),
    ],
)
async def test_version_opt_in_reports_effective_multimodal_capabilities(
    monkeypatch,
    video_enabled: bool,
    decoder_available: bool,
    expected: tuple[bool, bool, bool],
) -> None:
    """Video and live qualification fail closed when the local decoder is absent."""

    import hms_api.engine.multimodal.video as video_module

    config = replace(
        config_module._get_raw_config(),
        database_backend="postgresql",
        enable_file_upload_api=True,
        multimodal_enabled=True,
        multimodal_image_enabled=True,
        multimodal_video_enabled=video_enabled,
        multimodal_live_verified=True,
        multimodal_capability_responses_api=True,
        multimodal_capability_image_input=True,
        multimodal_capability_structured_outputs=True,
    )
    monkeypatch.setattr(config_module, "_get_raw_config", lambda: config)
    monkeypatch.setattr(video_module, "video_decoder_available", lambda: decoder_available)
    app = _version_only_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/version?include_multimodal=true")

    assert response.status_code == 200
    features = response.json()["features"]
    assert (
        features["multimodal_image"],
        features["multimodal_video"],
        features["multimodal_live_verified"],
    ) == expected


@pytest.mark.asyncio
async def test_version_opt_in_never_advertises_unqualified_oracle_runtime(monkeypatch) -> None:
    """The public surface stays false even for a stale, unvalidated Oracle config object."""

    config = replace(
        config_module._get_raw_config(),
        database_backend="oracle",
        enable_file_upload_api=True,
        multimodal_enabled=True,
        multimodal_image_enabled=True,
        multimodal_video_enabled=True,
        multimodal_live_verified=True,
        multimodal_capability_responses_api=True,
        multimodal_capability_image_input=True,
        multimodal_capability_structured_outputs=True,
    )
    monkeypatch.setattr(config_module, "_get_raw_config", lambda: config)
    app = _version_only_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/version?include_multimodal=true")

    assert response.status_code == 200
    features = response.json()["features"]
    assert features["multimodal_image"] is False
    assert features["multimodal_video"] is False
    assert features["multimodal_live_verified"] is False


@pytest.mark.asyncio
async def test_version_opt_in_respects_explicit_parser_allowlist(monkeypatch) -> None:
    """A configured parser that clients cannot select is not a public capability."""

    config = replace(
        config_module._get_raw_config(),
        database_backend="postgresql",
        enable_file_upload_api=True,
        file_parser_allowlist=["markitdown"],
        multimodal_enabled=True,
        multimodal_image_enabled=True,
        multimodal_video_enabled=True,
        multimodal_live_verified=True,
        multimodal_capability_responses_api=True,
        multimodal_capability_image_input=True,
        multimodal_capability_structured_outputs=True,
    )
    monkeypatch.setattr(config_module, "_get_raw_config", lambda: config)
    app = _version_only_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/version?include_multimodal=true")

    assert response.status_code == 200
    features = response.json()["features"]
    assert features["multimodal_image"] is False
    assert features["multimodal_video"] is False
    assert features["multimodal_live_verified"] is False


def test_openapi_publishes_optional_additive_multimodal_contract():
    schema = _version_only_app().openapi()
    parameters = schema["paths"]["/version"]["get"].get("parameters", [])
    include_parameter = next(item for item in parameters if item["name"] == "include_multimodal")
    assert include_parameter["in"] == "query"
    assert include_parameter["required"] is False
    assert include_parameter["schema"]["type"] == "boolean"
    assert include_parameter["schema"]["default"] is False

    schemas = schema["components"]["schemas"]
    features_schema = schemas["FeaturesInfo"]
    feature_properties = features_schema["properties"]
    required_features = set(features_schema["required"])
    for field_name in ("multimodal_image", "multimodal_video", "multimodal_live_verified"):
        assert feature_properties[field_name]["type"] == "boolean"
        assert feature_properties[field_name]["default"] is False
        assert field_name not in required_features

    operation_schema = schemas["OperationStatusResponse"]
    result_metadata_variants = operation_schema["properties"]["result_metadata"]["anyOf"]
    assert {item.get("$ref") for item in result_metadata_variants} >= {"#/components/schemas/OperationResultMetadata"}
    assert "result_metadata" not in set(operation_schema["required"])

    result_schema = schemas["OperationResultMetadata"]
    assert result_schema["additionalProperties"] is True
    assert "multimodal" not in set(result_schema.get("required", []))
    assert result_schema["properties"]["multimodal"]["anyOf"][0]["$ref"] == (
        "#/components/schemas/MultimodalOperationMetadata"
    )

    multimodal_schema = schemas["MultimodalOperationMetadata"]
    assert multimodal_schema["additionalProperties"] is False
    assert not multimodal_schema.get("required")
    assert set(multimodal_schema["properties"]) == {
        "asset_id",
        "asset_sha256",
        "media_kind",
        "pipeline_version",
        "descriptor_model",
        "resolved_model",
        "stage",
        "child_retain_operation_id",
        "child_retain_status",
        "recall_ready",
        "retryable",
        "sanitized_error_code",
        "provider_request_id",
        "input_tokens",
        "output_tokens",
        "logical_calls",
        "physical_attempts",
        "possible_duplicate_provider_attempt",
    }
