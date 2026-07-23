"""Focused tests for the server-static multimodal configuration boundary."""

import logging
from unittest.mock import AsyncMock

import pytest

import hms_api.config as config_module
from hms_api.config import HMSConfig


@pytest.fixture(autouse=True)
def isolated_multimodal_env(monkeypatch):
    """Remove host multimodal settings so defaults are deterministic."""

    for name, value in vars(config_module).items():
        if name.startswith("ENV_MULTIMODAL_") and isinstance(value, str):
            monkeypatch.delenv(value, raising=False)
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "mock")
    config_module.clear_config_cache()
    yield
    config_module.clear_config_cache()


def test_multimodal_defaults_are_opt_in_and_use_exact_requested_model():
    config = HMSConfig.from_env()

    assert config.multimodal_enabled is False
    assert config.multimodal_provider == "openai"
    assert config.multimodal_model == "gpt-5-mini"
    assert config.multimodal_base_url == "https://api.openai.com/v1"
    assert config.multimodal_api_key is None
    assert config.multimodal_image_enabled is True
    assert config.multimodal_video_enabled is False
    assert config.multimodal_live_verified is False
    assert config.multimodal_capability_responses_api is True
    assert config.multimodal_capability_image_input is True
    assert config.multimodal_capability_structured_outputs is True
    assert config.multimodal_descriptor_cache_ttl_seconds == 604800
    assert config.multimodal_prompt_version == "openai-mm-v2"
    # The policy-only prompt change does not alter the structured schema.
    assert config.multimodal_schema_version == "hms-multimodal-v1"


def test_multimodal_flat_environment_loads_full_static_role(monkeypatch):
    values = {
        "HMS_API_MULTIMODAL_ENABLED": "true",
        "HMS_API_MULTIMODAL_PROVIDER": "openai",
        "HMS_API_MULTIMODAL_MODEL": "gpt-5-mini-snapshot",
        "HMS_API_MULTIMODAL_MODEL_BEHAVIOR_VERSION": "alias-contract-v2",
        "HMS_API_MULTIMODAL_API_KEY": "mm-secret-value",
        "HMS_API_MULTIMODAL_BASE_URL": "https://compatible.example/v1",
        "HMS_API_MULTIMODAL_IMAGE_ENABLED": "true",
        "HMS_API_MULTIMODAL_VIDEO_ENABLED": "true",
        "HMS_API_MULTIMODAL_LIVE_VERIFIED": "true",
        "HMS_API_MULTIMODAL_CAPABILITY_RESPONSES_API": "true",
        "HMS_API_MULTIMODAL_CAPABILITY_IMAGE_INPUT": "true",
        "HMS_API_MULTIMODAL_CAPABILITY_STRUCTURED_OUTPUTS": "true",
        "HMS_API_MULTIMODAL_IMAGE_DETAIL": "high",
        "HMS_API_MULTIMODAL_MAX_IMAGE_BYTES": "1000",
        "HMS_API_MULTIMODAL_MAX_IMAGE_PIXELS": "2000",
        "HMS_API_MULTIMODAL_MAX_VIDEO_BYTES": "3000",
        "HMS_API_MULTIMODAL_MAX_VIDEO_DURATION_SECONDS": "120.5",
        "HMS_API_MULTIMODAL_VIDEO_PROBE_INTERVAL_SECONDS": "0.5",
        "HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES": "12",
        "HMS_API_MULTIMODAL_VIDEO_COVERAGE_RATIO": "0.5",
        "HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL": "6",
        "HMS_API_MULTIMODAL_MAX_OUTPUT_TOKENS": "1024",
        "HMS_API_MULTIMODAL_REQUEST_TIMEOUT_SECONDS": "15.5",
        "HMS_API_MULTIMODAL_MAX_RETRIES": "3",
        "HMS_API_MULTIMODAL_MAX_SCHEMA_REPAIRS": "0",
        "HMS_API_MULTIMODAL_MAX_CONCURRENCY": "2",
        "HMS_API_MULTIMODAL_DESCRIPTOR_CACHE_TTL_SECONDS": "3600",
        "HMS_API_MULTIMODAL_PROMPT_VERSION": "prompt-v2",
        "HMS_API_MULTIMODAL_SCHEMA_VERSION": "schema-v3",
        "HMS_API_MULTIMODAL_SAMPLING_VERSION": "sampler-v4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = HMSConfig.from_env()

    assert config.multimodal_enabled is True
    assert config.multimodal_model == "gpt-5-mini-snapshot"
    assert config.multimodal_model_behavior_version == "alias-contract-v2"
    assert config.multimodal_base_url == "https://compatible.example/v1"
    assert config.multimodal_image_detail == "high"
    assert config.multimodal_max_image_bytes == 1000
    assert config.multimodal_max_image_pixels == 2000
    assert config.multimodal_max_video_bytes == 3000
    assert config.multimodal_max_video_duration_seconds == pytest.approx(120.5)
    assert config.multimodal_video_probe_interval_seconds == pytest.approx(0.5)
    assert config.multimodal_video_max_frames == 12
    assert config.multimodal_video_coverage_ratio == pytest.approx(0.5)
    assert config.multimodal_max_frames_per_call == 6
    assert config.multimodal_max_output_tokens == 1024
    assert config.multimodal_request_timeout_seconds == pytest.approx(15.5)
    assert config.multimodal_max_retries == 3
    assert config.multimodal_max_schema_repairs == 0
    assert config.multimodal_max_concurrency == 2
    assert config.multimodal_descriptor_cache_ttl_seconds == 3600
    assert config.multimodal_prompt_version == "prompt-v2"
    assert config.multimodal_schema_version == "schema-v3"
    assert config.multimodal_sampling_version == "sampler-v4"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES", "3", "must be >= 4"),
        ("HMS_API_MULTIMODAL_VIDEO_COVERAGE_RATIO", "0", "must be > 0 and < 1"),
        ("HMS_API_MULTIMODAL_VIDEO_COVERAGE_RATIO", "1", "must be > 0 and < 1"),
        ("HMS_API_MULTIMODAL_MAX_IMAGE_BYTES", "0", "positive integer"),
        ("HMS_API_MULTIMODAL_MAX_IMAGE_PIXELS", "-1", "positive integer"),
        ("HMS_API_MULTIMODAL_MAX_VIDEO_BYTES", "0", "positive integer"),
        ("HMS_API_MULTIMODAL_MAX_VIDEO_DURATION_SECONDS", "0", "positive number"),
        ("HMS_API_MULTIMODAL_VIDEO_PROBE_INTERVAL_SECONDS", "-0.1", "positive number"),
        ("HMS_API_MULTIMODAL_MAX_OUTPUT_TOKENS", "0", "positive integer"),
        ("HMS_API_MULTIMODAL_REQUEST_TIMEOUT_SECONDS", "0", "positive number"),
        ("HMS_API_MULTIMODAL_MAX_CONCURRENCY", "0", "positive integer"),
        ("HMS_API_MULTIMODAL_DESCRIPTOR_CACHE_TTL_SECONDS", "0", "positive integer"),
        ("HMS_API_MULTIMODAL_MAX_RETRIES", "-1", "non-negative integer"),
        ("HMS_API_MULTIMODAL_MAX_SCHEMA_REPAIRS", "2", "must be 0 or 1"),
        ("HMS_API_MULTIMODAL_IMAGE_DETAIL", "original", "auto, low, high"),
    ],
)
def test_multimodal_invalid_scalar_budgets_fail_fast(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        HMSConfig.from_env()


def test_multimodal_frames_per_call_cannot_exceed_total_frame_budget(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES", "8")
    monkeypatch.setenv("HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL", "9")

    with pytest.raises(ValueError, match="MAX_FRAMES_PER_CALL.*must be <=.*VIDEO_MAX_FRAMES"):
        HMSConfig.from_env()


def test_multimodal_segment_plan_must_fit_schema_before_provider_io(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES", "65")
    monkeypatch.setenv("HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL", "65")

    with pytest.raises(ValueError, match="MAX_FRAMES_PER_CALL.*must be <= 64.*schema"):
        HMSConfig.from_env()

    monkeypatch.setenv("HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES", "257")
    monkeypatch.setenv("HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL", "1")

    with pytest.raises(ValueError, match=r"ceil\(.*VIDEO_MAX_FRAMES.*MAX_FRAMES_PER_CALL.*must be <= 256"):
        HMSConfig.from_env()


def test_multimodal_boolean_flags_are_strict(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_ENABLED", "yes")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        HMSConfig.from_env()


def test_enabled_multimodal_role_requires_dedicated_secret(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_ENABLED", "true")

    with pytest.raises(ValueError, match="MULTIMODAL_API_KEY.*required"):
        HMSConfig.from_env()


def test_custom_base_url_requires_all_explicit_provider_capabilities(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_BASE_URL", "https://compatible.example/v1")

    with pytest.raises(ValueError, match="Custom HMS_API_MULTIMODAL_BASE_URL requires explicit true"):
        HMSConfig.from_env()


def test_explicitly_missing_capability_rejects_enabled_official_provider(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_ENABLED", "true")
    monkeypatch.setenv("HMS_API_MULTIMODAL_API_KEY", "mm-secret-value")
    monkeypatch.setenv("HMS_API_MULTIMODAL_CAPABILITY_STRUCTURED_OUTPUTS", "false")

    with pytest.raises(ValueError, match="capability declaration is incomplete"):
        HMSConfig.from_env()


def test_video_intent_requires_image_transport(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_IMAGE_ENABLED", "false")
    monkeypatch.setenv("HMS_API_MULTIMODAL_VIDEO_ENABLED", "true")

    with pytest.raises(ValueError, match="transported as sampled image inputs"):
        HMSConfig.from_env()


def test_live_verified_cannot_be_claimed_while_feature_is_disabled(monkeypatch):
    monkeypatch.setenv("HMS_API_MULTIMODAL_LIVE_VERIFIED", "true")

    with pytest.raises(ValueError, match="LIVE_VERIFIED.*requires.*MULTIMODAL_ENABLED"):
        HMSConfig.from_env()


def test_oracle_backend_rejects_enabled_multimodal_runtime(monkeypatch):
    """The support matrix stays PostgreSQL-only until an Oracle runtime gate exists."""

    monkeypatch.setenv("HMS_API_DATABASE_BACKEND", "oracle")
    monkeypatch.setenv("HMS_API_MULTIMODAL_ENABLED", "true")
    monkeypatch.setenv("HMS_API_MULTIMODAL_API_KEY", "mm-secret-value")

    with pytest.raises(
        ValueError,
        match=r"MULTIMODAL_ENABLED=true.*DATABASE_BACKEND=postgresql.*not runtime-qualified",
    ):
        HMSConfig.from_env()


def test_multimodal_secret_is_not_in_repr_or_startup_log(monkeypatch, caplog):
    secret = "mm-secret-must-not-leak"
    monkeypatch.setenv("HMS_API_MULTIMODAL_ENABLED", "true")
    monkeypatch.setenv("HMS_API_MULTIMODAL_API_KEY", secret)
    caplog.set_level(logging.INFO, logger="hms_api.config")

    config = HMSConfig.from_env()
    config.log_config()
    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert secret not in repr(config)
    assert "multimodal_api_key" not in repr(config)
    assert "multimodal_base_url" not in repr(config)
    assert secret not in messages
    assert config.multimodal_base_url not in messages
    assert "provider=openai, model=gpt-5-mini" in messages


def test_multimodal_role_is_static_and_credentials_are_classified():
    configurable = HMSConfig.get_configurable_fields()
    credentials = HMSConfig.get_credential_fields()
    static = HMSConfig.get_static_fields()

    assert "multimodal_api_key" in credentials
    assert "multimodal_base_url" in credentials
    assert "multimodal_enabled" in static
    assert "multimodal_model" in static
    assert "multimodal_video_max_frames" in static
    assert not any(name.startswith("multimodal_") for name in configurable)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["multimodal_api_key", "multimodal_base_url"])
async def test_resolver_rejects_multimodal_credentials_before_database_access(field):
    from hms_api.config_resolver import ConfigResolver

    backend = AsyncMock()
    resolver = ConfigResolver(backend=backend)

    with pytest.raises(ValueError, match="Cannot set credential fields"):
        await resolver.update_bank_config("bank", {field: "secret-or-private-endpoint"})

    backend.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_rejects_multimodal_budget_as_server_static():
    from hms_api.config_resolver import ConfigResolver

    backend = AsyncMock()
    resolver = ConfigResolver(backend=backend)

    with pytest.raises(ValueError, match="Cannot override static"):
        await resolver.update_bank_config("bank", {"multimodal_video_max_frames": 100})

    backend.acquire.assert_not_called()
