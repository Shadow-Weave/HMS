"""
Tests for HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE config wiring.

Regression test for issue #1142: `OpenAIEmbeddings` hardcoded `batch_size=100` is
incompatible with OpenAI-compatible providers that enforce stricter per-request
limits (e.g. DashScope / Aliyun Tongyi cap at 10). Users must be able to override
the batch size via env var so `encode()` splits into smaller chunks.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hms_api.config import clear_config_cache

    env_vars_to_save = [
        "HMS_API_EMBEDDINGS_PROVIDER",
        "HMS_API_EMBEDDINGS_OPENAI_API_KEY",
        "HMS_API_EMBEDDINGS_OPENAI_MODEL",
        "HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE",
        "HMS_API_EMBEDDINGS_OPENROUTER_API_KEY",
        "HMS_API_LLM_API_KEY",
        "HMS_API_LLM_PROVIDER",
    ]

    original_values = {key: os.environ.get(key) for key in env_vars_to_save}

    clear_config_cache()

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_default_openai_batch_size_is_100():
    """Default batch size is 100 when env var unset (preserves legacy behavior)."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE", None)

    config = HMSConfig.from_env()
    assert config.embeddings_openai_batch_size == 100


def test_openai_batch_size_env_var_is_read():
    """HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE overrides the default."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "10"

    config = HMSConfig.from_env()
    assert config.embeddings_openai_batch_size == 10


def test_openai_embeddings_provider_uses_configured_batch_size():
    """create_embeddings_from_env() propagates config to OpenAIEmbeddings for 'openai' provider."""
    from hms_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_PROVIDER"] = "openai"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_API_KEY"] = "sk-test"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "10"

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.batch_size == 10


def test_openrouter_provider_uses_configured_batch_size():
    """'openrouter' provider also honors the shared batch-size config (both paths use OpenAIEmbeddings)."""
    from hms_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_PROVIDER"] = "openrouter"
    os.environ["HMS_API_EMBEDDINGS_OPENROUTER_API_KEY"] = "sk-or-test"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "8"

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.batch_size == 8


def test_zero_batch_size_is_rejected():
    """Zero would cause `range(0, N, 0)` to crash at runtime — fail fast at config load."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "0"

    with pytest.raises(ValueError, match="must be >= 1"):
        HMSConfig.from_env()


def test_negative_batch_size_is_rejected():
    """Negative values would silently skip batching — reject at config load."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "-5"

    with pytest.raises(ValueError, match="must be >= 1"):
        HMSConfig.from_env()


def test_non_numeric_batch_size_is_rejected():
    """Non-integer strings are rejected with a clear error pointing at the env var name."""
    from hms_api.config import HMSConfig

    os.environ["HMS_API_LLM_PROVIDER"] = "mock"
    os.environ["HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "not-a-number"

    with pytest.raises(ValueError, match="HMS_API_EMBEDDINGS_OPENAI_BATCH_SIZE"):
        HMSConfig.from_env()


def test_openai_encode_splits_on_configured_batch_size(monkeypatch):
    """encode() sends multiple upstream requests when len(texts) > batch_size."""
    from types import SimpleNamespace

    from hms_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small", batch_size=10)

    calls: list[int] = []

    def fake_create(*, model, input):
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0] * 1536) for i in range(len(input))])

    emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    emb._dimension = 1536

    vectors = emb.encode(["x"] * 25)

    assert len(vectors) == 25
    assert calls == [10, 10, 5], (
        f"Expected upstream calls of size 10, 10, 5 when batch_size=10 and 25 inputs, got {calls}"
    )
