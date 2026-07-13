"""Integration tests for hms-litellm."""

import pytest

from hms_litellm import (
    configure,
    set_defaults,
    get_defaults,
    enable,
    disable,
    is_enabled,
    cleanup,
    get_config,
    is_configured,
    reset_config,
    MemoryInjectionMode,
)
from hms_litellm.callbacks import HMSCallback


class TestConfiguration:
    """Tests for configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        disable()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_configure_creates_config(self):
        """Test that configure creates a config object."""
        config = configure(
            hms_api_url="http://localhost:8888",
        )
        # Set defaults separately (new API)
        defaults = set_defaults(bank_id="test-agent")

        assert config is not None
        assert config.hms_api_url == "http://localhost:8888"
        assert defaults.bank_id == "test-agent"

    def test_configure_with_all_options(self):
        """Test configure with all options."""
        config = configure(
            hms_api_url="http://custom:9999",
            api_key="secret-key",
            store_conversations=False,
            inject_memories=False,
            injection_mode=MemoryInjectionMode.PREPEND_USER,
            excluded_models=["gpt-3.5*"],
            verbose=True,
            sync_storage=True,
        )

        # Set defaults separately (new API)
        defaults = set_defaults(
            bank_id="custom-agent",
            max_memories=5,
            max_memory_tokens=1000,
            budget="high",
            fact_types=["world", "opinion"],
            document_id="doc-123",
        )

        assert config.hms_api_url == "http://custom:9999"
        assert config.api_key == "secret-key"
        assert config.store_conversations is False
        assert config.inject_memories is False
        assert config.injection_mode == MemoryInjectionMode.PREPEND_USER
        assert config.excluded_models == ["gpt-3.5*"]
        assert config.verbose is True
        assert config.sync_storage is True

        assert defaults.bank_id == "custom-agent"
        assert defaults.max_memories == 5
        assert defaults.max_memory_tokens == 1000
        assert defaults.budget == "high"
        assert defaults.fact_types == ["world", "opinion"]
        assert defaults.document_id == "doc-123"

    def test_is_configured_with_defaults(self):
        """Test is_configured returns True with default bank_id."""
        configure()  # Uses default bank_id="default"
        assert is_configured() is True

    def test_is_configured_with_bank_id_in_defaults(self):
        """Test is_configured returns True with bank_id in defaults."""
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        assert is_configured() is True

    def test_is_configured_with_explicit_bank_id(self):
        """Test is_configured returns True with explicit bank_id."""
        configure(bank_id="test-agent")
        assert is_configured() is True

    def test_reset_config(self):
        """Test reset_config clears the configuration."""
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        assert is_configured() is True

        reset_config()
        assert get_config() is None
        assert get_defaults() is None
        assert is_configured() is False


class TestEnableDisable:
    """Tests for enable/disable functionality."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_enable_without_config_raises(self):
        """Test enable raises error without configuration."""
        with pytest.raises(RuntimeError, match="not configured"):
            enable()

    def test_enable_with_default_bank_id_works(self):
        """Test enable works with default bank_id (no explicit bank_id required)."""
        configure(hms_api_url="http://localhost:8888")
        # Should work - configure() provides default bank_id="default"
        enable()
        assert is_enabled() is True

    def test_enable_sets_enabled_flag(self):
        """Test enable sets the enabled flag."""
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        enable()

        assert is_enabled() is True

    def test_disable_clears_enabled_flag(self):
        """Test disable clears the enabled flag."""
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        enable()
        assert is_enabled() is True

        disable()
        assert is_enabled() is False

    def test_enable_idempotent(self):
        """Test enable is idempotent (can be called multiple times)."""
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        # Enable multiple times
        enable()
        enable()
        enable()

        # Should still be enabled
        assert is_enabled() is True


class TestCallback:
    """Tests for the HMSCallback class."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_extract_user_query_simple(self):
        """Test extracting user query from simple messages."""
        callback = HMSCallback()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        query = callback._extract_user_query(messages)
        assert query == "What is the capital of France?"

    def test_extract_user_query_from_last_user_message(self):
        """Test extracting query from last user message."""
        callback = HMSCallback()
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        query = callback._extract_user_query(messages)
        assert query == "Second question"

    def test_extract_user_query_structured_content(self):
        """Test extracting query from structured content (vision)."""
        callback = HMSCallback()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            },
        ]

        query = callback._extract_user_query(messages)
        assert query == "What's in this image?"

    def test_extract_user_query_multiple_text_parts(self):
        """Test extracting query with multiple text parts."""
        callback = HMSCallback()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            },
        ]

        query = callback._extract_user_query(messages)
        assert query == "First part. Second part."

    def test_format_memories(self):
        """Test formatting memories into context string."""
        callback = HMSCallback()

        # Create config and defaults with new API
        configure(hms_api_url="http://localhost:8888", verbose=False)
        set_defaults(bank_id="test", max_memories=10)

        config = get_config()
        defaults = get_defaults()

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
            {"text": "User works at Google", "fact_type": "world", "weight": 0.8},
        ]

        # Signature is: _format_memories(results, settings, config)
        formatted = callback._format_memories(memories, defaults, config)

        assert "Relevant Memories" in formatted
        assert "User likes Python" in formatted
        assert "User works at Google" in formatted
        assert "[WORLD]" in formatted

    def test_format_memories_with_verbose(self):
        """Test formatting memories with verbose mode shows weights."""
        callback = HMSCallback()

        # Create config and defaults with new API
        configure(hms_api_url="http://localhost:8888", verbose=True)
        set_defaults(bank_id="test", max_memories=10)

        config = get_config()
        defaults = get_defaults()

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
        ]

        # Signature is: _format_memories(results, settings, config)
        formatted = callback._format_memories(memories, defaults, config)

        assert "relevance: 0.95" in formatted

    def test_inject_memories_as_system_message(self):
        """Test injecting memories as system message."""
        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )
        set_defaults(bank_id="test")

        config = get_config()

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(
            messages, memory_context, config
        )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Relevant Memories" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_inject_memories_prepend_to_existing_system(self):
        """Test injecting memories appends to existing system message."""
        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )
        set_defaults(bank_id="test")

        config = get_config()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(
            messages, memory_context, config
        )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Relevant Memories" in result[0]["content"]

    def test_inject_memories_prepend_user_mode(self):
        """Test injecting memories in prepend_user mode."""
        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.PREPEND_USER,
        )
        set_defaults(bank_id="test")

        config = get_config()

        messages = [
            {"role": "user", "content": "What's my name?"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(
            messages, memory_context, config
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Relevant Memories" in result[0]["content"]
        assert "What's my name?" in result[0]["content"]

    def test_inject_memories_uses_last_user_message_when_no_hms_query(self):
        """Regression test: inject_memories=True should not require hms_query.

        The documented Quick Start example does not pass hms_query; the
        injection path must fall back to the last user message automatically.
        See: feat(litellm) #167 regression.
        """
        from unittest.mock import MagicMock, patch

        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            inject_memories=True,
        )
        set_defaults(bank_id="test-agent")

        messages = [{"role": "user", "content": "What did we discuss about AI?"}]
        kwargs = {}  # No hms_query provided — this is the regression scenario

        mock_memory = MagicMock()
        mock_memory.text = "AI is cool"
        mock_memory.type = "world"
        mock_memory.weight = 0.9

        with patch.object(callback, "_recall_memories_sync", return_value=[mock_memory]) as mock_recall:
            callback.log_pre_api_call(
                model="gpt-4o-mini",
                messages=messages,
                kwargs=kwargs,
            )
            # Should have called recall with the last user message as query
            mock_recall.assert_called_once()
            query_used = mock_recall.call_args[0][0]
            assert query_used == "What did we discuss about AI?"

        # Memories should have been injected into messages
        assert any("AI is cool" in str(m.get("content", "")) for m in messages)

    def test_inject_memories_hms_query_takes_precedence(self):
        """When hms_query is provided it should be used over the last user message."""
        from unittest.mock import MagicMock, patch

        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            inject_memories=True,
        )
        set_defaults(bank_id="test-agent")

        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"hms_query": "What do I know about Alice?"}

        mock_memory = MagicMock()
        mock_memory.text = "Alice likes cats"
        mock_memory.type = "world"
        mock_memory.weight = 0.9

        with patch.object(callback, "_recall_memories_sync", return_value=[mock_memory]) as mock_recall:
            callback.log_pre_api_call(
                model="gpt-4o-mini",
                messages=messages,
                kwargs=kwargs,
            )
            query_used = mock_recall.call_args[0][0]
            assert query_used == "What do I know about Alice?"

    def test_should_skip_model_exact_match(self):
        """Test model exclusion with exact match."""
        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5-turbo"],
        )
        set_defaults(bank_id="test")

        config = get_config()

        assert callback._should_skip_model("gpt-3.5-turbo", config) is True
        assert callback._should_skip_model("gpt-4", config) is False

    def test_should_skip_model_wildcard(self):
        """Test model exclusion with wildcard pattern."""
        callback = HMSCallback()

        configure(
            hms_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5*", "claude-instant-*"],
        )
        set_defaults(bank_id="test")

        config = get_config()

        assert callback._should_skip_model("gpt-3.5-turbo", config) is True
        assert callback._should_skip_model("gpt-3.5-turbo-16k", config) is True
        assert callback._should_skip_model("claude-instant-1.2", config) is True
        assert callback._should_skip_model("gpt-4", config) is False
        assert callback._should_skip_model("claude-3-opus", config) is False


class TestDeduplication:
    """Tests for conversation deduplication."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_compute_conversation_hash(self):
        """Test computing conversation hash."""
        callback = HMSCallback()

        hash1 = callback._compute_conversation_hash("Hello", "Hi there!")
        hash2 = callback._compute_conversation_hash("Hello", "Hi there!")
        hash3 = callback._compute_conversation_hash("Hello", "Different response")

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3

    def test_compute_conversation_hash_case_insensitive(self):
        """Test that hash is case insensitive."""
        callback = HMSCallback()

        hash1 = callback._compute_conversation_hash("HELLO", "HI THERE!")
        hash2 = callback._compute_conversation_hash("hello", "hi there!")

        assert hash1 == hash2

    def test_is_duplicate_first_time(self):
        """Test first occurrence is not a duplicate."""
        callback = HMSCallback()

        result = callback._is_duplicate("abc123")

        assert result is False

    def test_is_duplicate_second_time(self):
        """Test second occurrence is a duplicate."""
        callback = HMSCallback()

        callback._is_duplicate("abc123")  # First time
        result = callback._is_duplicate("abc123")  # Second time

        assert result is True

    def test_is_duplicate_different_hashes(self):
        """Test different hashes are not duplicates."""
        callback = HMSCallback()

        callback._is_duplicate("abc123")
        result = callback._is_duplicate("xyz789")

        assert result is False


class TestContextManager:
    """Tests for the hms_memory context manager."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_context_manager_enables_and_disables(self):
        """Test context manager enables and disables correctly."""
        from hms_litellm import hms_memory

        assert is_enabled() is False

        with hms_memory(bank_id="test-agent"):
            assert is_enabled() is True
            defaults = get_defaults()
            assert defaults.bank_id == "test-agent"

        assert is_enabled() is False

    def test_context_manager_restores_previous_config(self):
        """Test context manager restores previous configuration."""
        from hms_litellm import hms_memory

        # Set up initial config
        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="original-agent")
        enable()
        assert get_defaults().bank_id == "original-agent"

        # Use context manager with different config
        with hms_memory(bank_id="temporary-agent"):
            assert get_defaults().bank_id == "temporary-agent"

        # Should restore original config
        assert get_defaults().bank_id == "original-agent"
        assert is_enabled() is True

    def test_context_manager_with_fact_types(self):
        """Test context manager with fact_types parameter."""
        from hms_litellm import hms_memory

        with hms_memory(bank_id="test-agent", fact_types=["world", "opinion"]):
            defaults = get_defaults()
            assert defaults.fact_types == ["world", "opinion"]


class TestFactTypes:
    """Tests for fact_types configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_configure_with_fact_types(self):
        """Test configuring with fact_types."""
        configure(hms_api_url="http://localhost:8888")
        defaults = set_defaults(
            bank_id="test-agent",
            fact_types=["world", "agent", "opinion"],
        )

        assert defaults.fact_types == ["world", "agent", "opinion"]

    def test_configure_without_fact_types(self):
        """Test configuring without fact_types defaults to None."""
        configure(hms_api_url="http://localhost:8888")
        defaults = set_defaults(bank_id="test-agent")

        assert defaults.fact_types is None


class TestSetDefaults:
    """Tests for set_defaults functionality."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_set_defaults_creates_defaults(self):
        """Test set_defaults creates a defaults object."""
        defaults = set_defaults(bank_id="test-agent")

        assert defaults is not None
        assert defaults.bank_id == "test-agent"

    def test_set_defaults_with_all_options(self):
        """Test set_defaults with all options."""
        defaults = set_defaults(
            bank_id="test-agent",
            document_id="doc-123",
            budget="high",
            fact_types=["world", "opinion"],
            max_memories=10,
            max_memory_tokens=2048,
            use_reflect=True,
            reflect_include_facts=True,
            reflect_context="I am a helpful assistant.",
            include_entities=False,
            trace=True,
        )

        assert defaults.bank_id == "test-agent"
        assert defaults.document_id == "doc-123"
        assert defaults.budget == "high"
        assert defaults.fact_types == ["world", "opinion"]
        assert defaults.max_memories == 10
        assert defaults.max_memory_tokens == 2048
        assert defaults.use_reflect is True
        assert defaults.reflect_include_facts is True
        assert defaults.reflect_context == "I am a helpful assistant."
        assert defaults.include_entities is False
        assert defaults.trace is True

    def test_set_defaults_updates_existing(self):
        """Test set_defaults updates existing defaults."""
        set_defaults(bank_id="first-agent", budget="low")
        defaults = set_defaults(budget="high")  # Only update budget

        assert defaults.bank_id == "first-agent"  # Preserved
        assert defaults.budget == "high"  # Updated

    def test_get_defaults_returns_none_initially(self):
        """Test get_defaults returns None when not set."""
        assert get_defaults() is None


class TestStreamingResponseHandling:
    """Tests for streaming response handling in monkeypatch wrappers and callbacks.

    Regression tests for https://github.com/hms-memory/hms/issues/1221
    CustomStreamWrapper objects don't have .choices and crash _format_conversation_for_storage.
    """

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def _make_fake_chunks(self):
        """Create fake streaming chunks mimicking LiteLLM's ModelResponseStream."""
        from unittest.mock import MagicMock

        chunks = []
        for text in ["Hello", " ", "world", "!"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        return chunks

    def test_format_conversation_for_storage_returns_empty_for_stream(self):
        """Test that _format_conversation_for_storage returns empty for stream objects.

        Reproduces the exact bug from issue #1221.
        """
        from hms_litellm import _format_conversation_for_storage

        class FakeStreamWrapper:
            """Mimics litellm.utils.CustomStreamWrapper which lacks .choices."""
            pass

        messages = [{"role": "user", "content": "Hello"}]
        stream_response = FakeStreamWrapper()

        result = _format_conversation_for_storage(messages, stream_response)
        assert result == ""

    def test_store_conversation_skips_stream_response(self):
        """Test that _store_conversation handles streaming responses gracefully."""
        from hms_litellm import _store_conversation

        configure(hms_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        class FakeStreamWrapper:
            pass

        messages = [{"role": "user", "content": "Hello"}]
        stream_response = FakeStreamWrapper()

        # Should not raise
        _store_conversation(messages, stream_response, "gpt-4o-mini")

    def test_wrapped_completion_returns_stream_wrapper(self):
        """Test that completion() wraps streaming responses for deferred storage."""
        from unittest.mock import patch, MagicMock
        from hms_litellm import _LiteLLMStreamWrapper

        configure(hms_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")
        enable()

        # Create a fake stream (no .choices attribute)
        class FakeStreamWrapper:
            pass

        fake_stream = FakeStreamWrapper()

        with patch("litellm.completion", return_value=fake_stream):
            import hms_litellm
            response = hms_litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            # Should return a stream wrapper, not the raw stream
            assert isinstance(response, _LiteLLMStreamWrapper)

    def test_stream_wrapper_yields_all_chunks(self):
        """Test that _LiteLLMStreamWrapper passes through all chunks."""
        from hms_litellm import _LiteLLMStreamWrapper

        configure(hms_api_url="http://localhost:8888", store_conversations=False)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        collected = list(wrapper)
        assert len(collected) == 4

    def test_stream_wrapper_stores_conversation_on_exhaustion(self):
        """Test that _LiteLLMStreamWrapper stores conversation when stream is consumed."""
        from unittest.mock import patch
        from hms_litellm import _LiteLLMStreamWrapper

        configure(hms_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        with patch("hms_litellm._store_conversation_from_text") as mock_store:
            # Consume all chunks
            list(wrapper)

            mock_store.assert_called_once()
            stored_text = mock_store.call_args[0][0]
            assert "USER: Hello" in stored_text
            assert "ASSISTANT: Hello world!" in stored_text

    def test_stream_wrapper_stores_on_context_manager_exit(self):
        """Test that _LiteLLMStreamWrapper stores conversation on context manager exit."""
        from unittest.mock import patch
        from hms_litellm import _LiteLLMStreamWrapper

        configure(hms_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        with patch("hms_litellm._store_conversation_from_text") as mock_store:
            with wrapper:
                for _ in wrapper:
                    pass
            # Should store exactly once (not double-store)
            mock_store.assert_called_once()

    def test_callback_log_success_with_stream_response(self):
        """Test that HMSCallback.log_success_event handles stream responses."""
        from hms_litellm.callbacks import HMSCallback

        configure(hms_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        callback = HMSCallback()

        class FakeStreamWrapper:
            pass

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Should not raise AttributeError
        callback.log_success_event(
            kwargs=kwargs,
            response_obj=FakeStreamWrapper(),
            start_time=0.0,
            end_time=1.0,
        )


class TestStreamingSupport:
    """Tests for streaming support in wrappers."""

    def test_wrap_openai_with_stream_no_error(self):
        """Test that wrap_openai handles streaming without errors."""
        from unittest.mock import Mock, MagicMock
        from hms_litellm.wrappers import wrap_openai

        # Create mock OpenAI client
        mock_client = Mock()
        mock_stream = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        # Wrap the client with store_conversations=False
        wrapped = wrap_openai(
            mock_client,
            hms_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=False,  # Disable storage for this test
        )

        # Call with stream=True
        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Should return the stream without errors
        assert result == mock_stream
        # Verify the underlying client was called with stream=True
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_wrap_anthropic_with_stream_no_error(self):
        """Test that wrap_anthropic handles streaming without errors."""
        from unittest.mock import Mock, MagicMock
        from hms_litellm.wrappers import wrap_anthropic

        # Create mock Anthropic client
        mock_client = Mock()
        mock_stream = MagicMock()
        mock_client.messages.create.return_value = mock_stream

        # Wrap the client with store_conversations=False
        wrapped = wrap_anthropic(
            mock_client,
            hms_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=False,  # Disable storage for this test
        )

        # Call with stream=True
        result = wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Should return the stream without errors
        assert result == mock_stream
        # Verify the underlying client was called with stream=True
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_wrap_openai_stream_stores_conversation(self):
        """Test that streaming stores conversation after all chunks are consumed."""
        from unittest.mock import Mock, MagicMock, patch
        from hms_litellm.wrappers import wrap_openai

        # Create mock OpenAI client
        mock_client = Mock()

        # Create mock stream chunks
        class MockChunk:
            def __init__(self, content):
                self.choices = [MagicMock()]
                self.choices[0].delta.content = content

        chunks = [
            MockChunk("Hello"),
            MockChunk(" "),
            MockChunk("world"),
            MockChunk("!"),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        # Wrap the client
        wrapped = wrap_openai(
            mock_client,
            hms_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=True,  # Enable storage
        )

        # Mock the hms client
        mock_hms_client = MagicMock()
        with patch.object(wrapped, "_get_hms_client", return_value=mock_hms_client):
            # Call with stream=True
            result = wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume all chunks
            collected = []
            for chunk in result:
                collected.append(chunk)

            # Verify all chunks were yielded
            assert len(collected) == 4

            # Verify retain was called with the complete conversation
            mock_hms_client.retain.assert_called_once()
            call_kwargs = mock_hms_client.retain.call_args[1]
            assert "USER: Hello" in call_kwargs["content"]
            assert "ASSISTANT: Hello world!" in call_kwargs["content"]

    def test_wrap_anthropic_stream_stores_conversation(self):
        """Test that streaming stores conversation after all chunks are consumed."""
        from unittest.mock import Mock, MagicMock, patch
        from hms_litellm.wrappers import wrap_anthropic

        # Create mock Anthropic client
        mock_client = Mock()

        # Create mock stream chunks
        class MockChunk:
            def __init__(self, content):
                self.type = "content_block_delta"
                self.delta = MagicMock()
                self.delta.text = content

        chunks = [
            MockChunk("Hello"),
            MockChunk(" "),
            MockChunk("world"),
            MockChunk("!"),
        ]
        mock_client.messages.create.return_value = iter(chunks)

        # Wrap the client
        wrapped = wrap_anthropic(
            mock_client,
            hms_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=True,  # Enable storage
        )

        # Mock the hms client
        mock_hms_client = MagicMock()
        with patch.object(wrapped, "_get_hms_client", return_value=mock_hms_client):
            # Call with stream=True
            result = wrapped.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume all chunks
            collected = []
            for chunk in result:
                collected.append(chunk)

            # Verify all chunks were yielded
            assert len(collected) == 4

            # Verify retain was called with the complete conversation
            mock_hms_client.retain.assert_called_once()
            call_kwargs = mock_hms_client.retain.call_args[1]
            assert "USER: Hello" in call_kwargs["content"]
            assert "ASSISTANT: Hello world!" in call_kwargs["content"]


class TestOpenAIResponsesWrapper:
    """Tests for automatic memory with the OpenAI Responses API."""

    def test_responses_create_injects_recall_and_retains_exchange(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from hms_litellm.wrappers import wrap_openai

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="Alice prefers dark mode.")
        wrapped = wrap_openai(
            mock_client,
            hms_api_url="http://localhost:18080",
            bank_id="alice",
            store_conversations=True,
            inject_memories=True,
        )

        memory = SimpleNamespace(
            text="Alice prefers dark mode.",
            type="world",
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
            context=None,
            metadata=None,
        )
        mock_hms_client = MagicMock()
        mock_hms_client.recall.return_value = [memory]

        with patch.object(wrapped, "_get_hms_client", return_value=mock_hms_client):
            response = wrapped.responses.create(
                model="gpt-4o-mini",
                instructions="Answer concisely.",
                input=[{"role": "user", "content": [{"type": "input_text", "text": "What does Alice prefer?"}]}],
            )

        assert response.output_text == "Alice prefers dark mode."
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"].startswith("Answer concisely.")
        assert "# Relevant Memories" in call_kwargs["instructions"]
        assert "Alice prefers dark mode." in call_kwargs["instructions"]
        mock_hms_client.retain.assert_called_once()
        retained = mock_hms_client.retain.call_args.kwargs["content"]
        assert "USER: What does Alice prefer?" in retained
        assert "ASSISTANT: Alice prefers dark mode." in retained

    def test_responses_create_strips_hms_kwargs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from hms_litellm.wrappers import wrap_openai

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="Hello")
        wrapped = wrap_openai(mock_client, bank_id="default")

        wrapped.responses.create(
            model="gpt-4o-mini",
            input="Hello",
            hms_bank_id="override",
            hms_inject_memories=False,
            hms_store_conversations=False,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs == {"model": "gpt-4o-mini", "input": "Hello"}

    def test_responses_stream_retains_collected_output(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from hms_litellm.wrappers import wrap_openai

        events = iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Hello"),
                SimpleNamespace(type="response.output_text.delta", delta=" Alice"),
                SimpleNamespace(type="response.completed", delta=None),
            ]
        )
        mock_client = MagicMock()
        mock_client.responses.create.return_value = events
        wrapped = wrap_openai(
            mock_client,
            hms_api_url="http://localhost:18080",
            bank_id="alice",
            inject_memories=False,
            store_conversations=True,
        )
        mock_hms_client = MagicMock()

        with patch.object(wrapped, "_get_hms_client", return_value=mock_hms_client):
            result = wrapped.responses.create(model="gpt-4o-mini", input="Say hello", stream=True)
            assert len(list(result)) == 3

        retained = mock_hms_client.retain.call_args.kwargs["content"]
        assert "USER: Say hello" in retained
        assert "ASSISTANT: Hello Alice" in retained
