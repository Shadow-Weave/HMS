"""
Tests for configuration validation.

Verifies that config validation catches invalid parameter combinations.
"""

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from hms_api.config import clear_config_cache

    # Save original environment values
    env_vars_to_save = [
        "HMS_API_RETAIN_MAX_COMPLETION_TOKENS",
        "HMS_API_RETAIN_CHUNK_SIZE",
        "HMS_API_LLM_PROVIDER",
        "HMS_API_LLM_MODEL",
        "HMS_API_DATABASE_URL",
        "HMS_API_MIGRATION_DATABASE_URL",
    ]

    # Save original values
    original_values = {}
    for key in env_vars_to_save:
        original_values[key] = os.environ.get(key)

    clear_config_cache()

    yield

    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_retain_max_completion_tokens_must_be_greater_than_chunk_size():
    """Test that RETAIN_MAX_COMPLETION_TOKENS > RETAIN_CHUNK_SIZE validation works."""
    from hms_api.config import HMSConfig

    # Set invalid config: max_completion_tokens <= chunk_size
    os.environ["HMS_API_RETAIN_MAX_COMPLETION_TOKENS"] = "1000"
    os.environ["HMS_API_RETAIN_CHUNK_SIZE"] = "2000"
    os.environ["HMS_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        HMSConfig.from_env()

    error_message = str(exc_info.value)

    # Verify error message contains helpful information
    assert "HMS_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "1000" in error_message
    assert "HMS_API_RETAIN_CHUNK_SIZE" in error_message
    assert "2000" in error_message
    assert "must be greater than" in error_message
    assert "You have two options to fix this:" in error_message
    assert "Increase HMS_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "Use a model that supports" in error_message


def test_retain_max_completion_tokens_equal_to_chunk_size_fails():
    """Test that RETAIN_MAX_COMPLETION_TOKENS == RETAIN_CHUNK_SIZE also fails."""
    from hms_api.config import HMSConfig

    # Set invalid config: max_completion_tokens == chunk_size
    os.environ["HMS_API_RETAIN_MAX_COMPLETION_TOKENS"] = "3000"
    os.environ["HMS_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HMS_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        HMSConfig.from_env()

    error_message = str(exc_info.value)
    assert "must be greater than" in error_message


def test_valid_retain_config_succeeds():
    """Test that valid config with max_completion_tokens > chunk_size works."""
    from hms_api.config import HMSConfig

    # Set valid config: max_completion_tokens > chunk_size
    os.environ["HMS_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HMS_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HMS_API_LLM_PROVIDER"] = "mock"

    # Should not raise
    config = HMSConfig.from_env()
    assert config.retain_max_completion_tokens == 64000
    assert config.retain_chunk_size == 3000


def test_log_config_masks_database_urls(caplog):
    """Config startup logs must not expose database credentials."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_DATABASE_URL"] = "postgresql://hms_user:plain-password@db:5432/hms_db"
    os.environ["HMS_API_MIGRATION_DATABASE_URL"] = (
        "postgresql://migration_user:migration-password@db-admin:5432/hms_db"
    )
    os.environ["HMS_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HMS_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HMS_API_LLM_PROVIDER"] = "mock"

    caplog.set_level(logging.INFO, logger="hms_api.config")

    config = HMSConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "hms_user" not in log_output
    assert "plain-password" not in log_output
    assert "migration_user" not in log_output
    assert "migration-password" not in log_output
    assert "postgresql://***:***@db:5432/hms_db" in log_output
    assert "postgresql://***:***@db-admin:5432/hms_db" in log_output


def test_read_database_url_defaults_to_none_when_unset(monkeypatch):
    """Without HMS_API_READ_DATABASE_URL, the field is None — engine
    will alias the read backend to the primary, preserving today's
    single-pool behaviour byte-for-bit. This is the most important guarantee
    of the change: zero-config means zero behaviour change.
    """
    from hms_api.config import HMSConfig

    monkeypatch.delenv("HMS_API_READ_DATABASE_URL", raising=False)
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "mock")

    config = HMSConfig.from_env()
    assert config.read_database_url is None


def test_read_database_url_is_loaded_when_set(monkeypatch):
    """When HMS_API_READ_DATABASE_URL is set, the value flows into
    config so MemoryEngine.initialize() will open a second backend against
    that URL for recall queries.
    """
    from hms_api.config import HMSConfig

    read_url = "postgresql://reader:secret@replica.example:5432/hms"
    monkeypatch.setenv("HMS_API_READ_DATABASE_URL", read_url)
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "mock")

    config = HMSConfig.from_env()
    assert config.read_database_url == read_url


def test_read_database_url_empty_string_is_treated_as_unset(monkeypatch):
    """Helm sometimes renders an unset env var as the empty string. Treat it
    the same as unset so deployments that conditionally set the var don't
    accidentally try to open a pool against `''`.
    """
    from hms_api.config import HMSConfig

    monkeypatch.setenv("HMS_API_READ_DATABASE_URL", "")
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "mock")

    config = HMSConfig.from_env()
    assert config.read_database_url is None


def test_log_config_masks_read_database_url(monkeypatch, caplog):
    """Read-replica URL credentials must be masked in startup logs, same as
    the primary URL.
    """
    from hms_api.config import HMSConfig

    monkeypatch.setenv("HMS_API_DATABASE_URL", "postgresql://hms:pw@primary:5432/db")
    monkeypatch.setenv("HMS_API_READ_DATABASE_URL", "postgresql://reader:replica-secret@replica:5432/db")
    monkeypatch.setenv("HMS_API_RETAIN_MAX_COMPLETION_TOKENS", "64000")
    monkeypatch.setenv("HMS_API_RETAIN_CHUNK_SIZE", "3000")
    monkeypatch.setenv("HMS_API_LLM_PROVIDER", "mock")

    caplog.set_level(logging.INFO, logger="hms_api.config")

    config = HMSConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "reader" not in log_output
    assert "replica-secret" not in log_output
    assert "Read database" in log_output
    assert "postgresql://***:***@replica:5432/db" in log_output


# Note: The BadRequestError wrapping is implemented in fact_extraction.py
# but requires a complex integration test setup. The functionality is
# straightforward: when a BadRequestError containing keywords like
# "max_tokens", "max_completion_tokens", or "maximum context" is caught,
# it's wrapped in a ValueError with helpful guidance.
#
# The config validation tests above ensure users get early feedback
# about invalid configurations before runtime errors occur.
