"""Test provider-specific default models in config."""

import os

import pytest


def test_provider_default_models():
    """Test that each provider has a default model and it's used when model is not explicitly set."""
    from hms_api.config import PROVIDER_DEFAULT_MODELS, HMSConfig, clear_config_cache

    # Save original env vars
    original_provider = os.environ.get("HMS_API_LLM_PROVIDER")
    original_model = os.environ.get("HMS_API_LLM_MODEL")

    try:
        # Test each provider has a default
        for provider, expected_model in PROVIDER_DEFAULT_MODELS.items():
            clear_config_cache()
            os.environ["HMS_API_LLM_PROVIDER"] = provider
            # Remove explicit model setting to test default
            if "HMS_API_LLM_MODEL" in os.environ:
                del os.environ["HMS_API_LLM_MODEL"]

            config = HMSConfig.from_env()
            assert config.llm_provider == provider, f"Provider mismatch for {provider}"
            assert config.llm_model == expected_model, f"Expected {expected_model} for {provider}, got {config.llm_model}"

    finally:
        # Restore original env vars
        clear_config_cache()
        if original_provider:
            os.environ["HMS_API_LLM_PROVIDER"] = original_provider
        elif "HMS_API_LLM_PROVIDER" in os.environ:
            del os.environ["HMS_API_LLM_PROVIDER"]

        if original_model:
            os.environ["HMS_API_LLM_MODEL"] = original_model
        elif "HMS_API_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_LLM_MODEL"]


def test_explicit_model_overrides_provider_default():
    """Test that explicit model setting overrides provider default."""
    from hms_api.config import HMSConfig, clear_config_cache

    original_provider = os.environ.get("HMS_API_LLM_PROVIDER")
    original_model = os.environ.get("HMS_API_LLM_MODEL")

    try:
        clear_config_cache()
        os.environ["HMS_API_LLM_PROVIDER"] = "anthropic"
        os.environ["HMS_API_LLM_MODEL"] = "claude-sonnet-4-5-20250929"

        config = HMSConfig.from_env()
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-sonnet-4-5-20250929", "Explicit model should override default"

    finally:
        clear_config_cache()
        if original_provider:
            os.environ["HMS_API_LLM_PROVIDER"] = original_provider
        elif "HMS_API_LLM_PROVIDER" in os.environ:
            del os.environ["HMS_API_LLM_PROVIDER"]

        if original_model:
            os.environ["HMS_API_LLM_MODEL"] = original_model
        elif "HMS_API_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_LLM_MODEL"]


def test_per_operation_provider_default_model():
    """Test that per-operation providers use their own default models."""
    from hms_api.config import HMSConfig, clear_config_cache

    original_provider = os.environ.get("HMS_API_LLM_PROVIDER")
    original_model = os.environ.get("HMS_API_LLM_MODEL")
    original_retain_provider = os.environ.get("HMS_API_RETAIN_LLM_PROVIDER")
    original_retain_model = os.environ.get("HMS_API_RETAIN_LLM_MODEL")

    try:
        clear_config_cache()
        os.environ["HMS_API_LLM_PROVIDER"] = "openai"
        # Remove explicit model to use provider default
        if "HMS_API_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_LLM_MODEL"]

        # Set retain-specific provider but not model
        os.environ["HMS_API_RETAIN_LLM_PROVIDER"] = "anthropic"
        if "HMS_API_RETAIN_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_RETAIN_LLM_MODEL"]

        config = HMSConfig.from_env()

        # Global LLM should use OpenAI default
        assert config.llm_model == "gpt-4o-mini", f"Expected gpt-4o-mini, got {config.llm_model}"

        # Retain should use Anthropic default
        assert (
            config.retain_llm_model == "claude-haiku-4-5"
        ), f"Expected claude-haiku-4-5, got {config.retain_llm_model}"

    finally:
        clear_config_cache()
        if original_provider:
            os.environ["HMS_API_LLM_PROVIDER"] = original_provider
        elif "HMS_API_LLM_PROVIDER" in os.environ:
            del os.environ["HMS_API_LLM_PROVIDER"]

        if original_model:
            os.environ["HMS_API_LLM_MODEL"] = original_model
        elif "HMS_API_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_LLM_MODEL"]

        if original_retain_provider:
            os.environ["HMS_API_RETAIN_LLM_PROVIDER"] = original_retain_provider
        elif "HMS_API_RETAIN_LLM_PROVIDER" in os.environ:
            del os.environ["HMS_API_RETAIN_LLM_PROVIDER"]

        if original_retain_model:
            os.environ["HMS_API_RETAIN_LLM_MODEL"] = original_retain_model
        elif "HMS_API_RETAIN_LLM_MODEL" in os.environ:
            del os.environ["HMS_API_RETAIN_LLM_MODEL"]
