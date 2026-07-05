"""Unit tests for HMS Strands tools."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from hms_strands import (
    configure,
    create_hms_tools,
    memory_instructions,
    reset_config,
)
from hms_strands.errors import HMSError
from hms_strands.tools import _resolve_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client():
    """Create a mock HMS client."""
    client = MagicMock()
    client.retain = MagicMock()
    client.recall = MagicMock()
    client.reflect = MagicMock()
    client.create_bank = MagicMock()
    return client


def _mock_recall_response(texts: list[str]):
    """Create a mock RecallResponse with results."""
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


def _mock_reflect_response(text: str):
    """Create a mock ReflectResponse."""
    response = MagicMock()
    response.text = text
    return response


def _call_tool(tool_fn, **kwargs):
    """Call a Strands @tool decorated function directly, bypassing the decorator."""
    # Strands @tool stores the original function as __wrapped__ or we can call it directly
    # since the decorator preserves the callable interface
    return tool_fn(**kwargs)


# ---------------------------------------------------------------------------
# _resolve_client
# ---------------------------------------------------------------------------


class TestResolveClient:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_explicit_client(self):
        client = _mock_client()
        assert _resolve_client(client, None, None) is client

    def test_explicit_client_ignores_url_and_key(self):
        client = _mock_client()
        result = _resolve_client(client, "http://ignored", "ignored-key")
        assert result is client

    def test_creates_client_from_url(self):
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", None)
            mock_cls.assert_called_once_with(
                base_url="http://localhost:8888", timeout=30.0
            )

    def test_creates_client_with_api_key(self):
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", "my-key")
            mock_cls.assert_called_once_with(
                base_url="http://localhost:8888", timeout=30.0, api_key="my-key"
            )

    def test_falls_back_to_global_config_url(self):
        configure(hms_api_url="http://config:8888")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once_with(
                base_url="http://config:8888", timeout=30.0
            )

    def test_falls_back_to_global_config_api_key(self):
        configure(hms_api_url="http://config:8888", api_key="config-key")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once_with(
                base_url="http://config:8888", timeout=30.0, api_key="config-key"
            )

    def test_explicit_url_overrides_config(self):
        configure(hms_api_url="http://config:8888")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://explicit:9999", None)
            mock_cls.assert_called_once_with(
                base_url="http://explicit:9999", timeout=30.0
            )

    def test_explicit_api_key_overrides_config(self):
        configure(hms_api_url="http://config:8888", api_key="config-key")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, "explicit-key")
            mock_cls.assert_called_once_with(
                base_url="http://config:8888", timeout=30.0, api_key="explicit-key"
            )

    def test_raises_without_url_or_config(self):
        with pytest.raises(HMSError, match="No HMS API URL"):
            _resolve_client(None, None, None)

    def test_raises_with_empty_config_no_url(self):
        # Config exists but has default URL, so this should NOT raise
        configure()
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# create_hms_tools — factory
# ---------------------------------------------------------------------------


class TestCreateHMSTools:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_creates_three_tools_by_default(self):
        client = _mock_client()
        tools = create_hms_tools(bank_id="test", client=client)
        assert len(tools) == 3

    def test_tool_names(self):
        client = _mock_client()
        tools = create_hms_tools(bank_id="test", client=client)
        names = {t.__name__ for t in tools}
        assert names == {"hms_retain", "hms_recall", "hms_reflect"}

    def test_enable_retain_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            enable_retain=True,
            enable_recall=False,
            enable_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].__name__ == "hms_retain"

    def test_enable_recall_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=True,
            enable_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].__name__ == "hms_recall"

    def test_enable_reflect_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=False,
            enable_reflect=True,
        )
        assert len(tools) == 1
        assert tools[0].__name__ == "hms_reflect"

    def test_no_tools_when_all_disabled(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=False,
            enable_reflect=False,
        )
        assert len(tools) == 0

    def test_raises_without_client_or_config(self):
        with pytest.raises(HMSError, match="No HMS API URL"):
            create_hms_tools(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hms_api_url="http://localhost:8888")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hms_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once_with(
                base_url="http://localhost:8888", timeout=30.0
            )

    def test_api_key_passed_to_client(self):
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hms_tools(
                bank_id="test",
                hms_api_url="http://localhost:8888",
                api_key="secret",
            )
            mock_cls.assert_called_once_with(
                base_url="http://localhost:8888", timeout=30.0, api_key="secret"
            )


# ---------------------------------------------------------------------------
# retain tool
# ---------------------------------------------------------------------------


class TestRetainTool:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def _retain_tool(self, client, **kwargs):
        tools = create_hms_tools(
            bank_id="my-bank",
            client=client,
            enable_recall=False,
            enable_reflect=False,
            **kwargs,
        )
        return tools[0]

    def test_retain_success(self):
        client = _mock_client()
        t = self._retain_tool(client)
        result = _call_tool(t, content="I like dark mode")
        assert result == "Memory stored successfully."
        client.retain.assert_called_once_with(
            bank_id="my-bank", content="I like dark mode"
        )

    def test_retain_with_tags(self):
        client = _mock_client()
        t = self._retain_tool(client, tags=["env:test"])
        _call_tool(t, content="tagged content")
        client.retain.assert_called_once_with(
            bank_id="my-bank", content="tagged content", tags=["env:test"]
        )

    def test_retain_config_tags(self):
        configure(hms_api_url="http://localhost:8888", tags=["config-tag"])
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hms_tools(
                bank_id="test", enable_recall=False, enable_reflect=False
            )
        client = mock_cls.return_value
        _call_tool(tools[0], content="content")
        assert client.retain.call_args[1]["tags"] == ["config-tag"]

    def test_retain_explicit_tags_override_config(self):
        configure(hms_api_url="http://localhost:8888", tags=["config-tag"])
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hms_tools(
                bank_id="test",
                tags=["explicit-tag"],
                enable_recall=False,
                enable_reflect=False,
            )
        client = mock_cls.return_value
        _call_tool(tools[0], content="content")
        assert client.retain.call_args[1]["tags"] == ["explicit-tag"]

    def test_retain_creates_bank(self):
        client = _mock_client()
        t = self._retain_tool(client)
        _call_tool(t, content="content")
        client.create_bank.assert_called_once_with(bank_id="my-bank", name="my-bank")

    def test_retain_creates_bank_only_once(self):
        client = _mock_client()
        t = self._retain_tool(client)
        _call_tool(t, content="first")
        _call_tool(t, content="second")
        client.create_bank.assert_called_once()

    def test_retain_bank_already_exists(self):
        client = _mock_client()
        client.create_bank.side_effect = Exception("already exists")
        t = self._retain_tool(client)
        result = _call_tool(t, content="content")
        assert result == "Memory stored successfully."

    def test_retain_failure_raises_hms_error(self):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("network error")
        t = self._retain_tool(client)
        with pytest.raises(HMSError, match="Retain failed"):
            _call_tool(t, content="content")

    def test_retain_hms_error_not_wrapped(self):
        client = _mock_client()
        client.retain.side_effect = HMSError("original error")
        t = self._retain_tool(client)
        with pytest.raises(HMSError, match="original error"):
            _call_tool(t, content="content")

    def test_retain_failure_logs_error(self, caplog):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("network error")
        t = self._retain_tool(client)
        with caplog.at_level(logging.ERROR), pytest.raises(HMSError):
            _call_tool(t, content="content")
        assert "Retain failed" in caplog.text


# ---------------------------------------------------------------------------
# recall tool
# ---------------------------------------------------------------------------


class TestRecallTool:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def _recall_tool(self, client, **kwargs):
        tools = create_hms_tools(
            bank_id="my-bank",
            client=client,
            enable_retain=False,
            enable_reflect=False,
            **kwargs,
        )
        return tools[0]

    def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2", "fact3"])
        t = self._recall_tool(client)
        result = _call_tool(t, query="preferences")
        assert result == "1. fact1\n2. fact2\n3. fact3"

    def test_recall_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        t = self._recall_tool(client)
        result = _call_tool(t, query="unknown")
        assert result == "No relevant memories found."

    def test_recall_none_results(self):
        client = _mock_client()
        response = MagicMock()
        response.results = None
        client.recall.return_value = response
        t = self._recall_tool(client)
        result = _call_tool(t, query="unknown")
        assert result == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        t = self._recall_tool(client, budget="high", max_tokens=2048)
        _call_tool(t, query="query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_default_budget(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        t = self._recall_tool(client)
        _call_tool(t, query="query")
        assert client.recall.call_args[1]["budget"] == "mid"

    def test_recall_with_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        t = self._recall_tool(
            client, recall_tags=["scope:global"], recall_tags_match="all"
        )
        _call_tool(t, query="query")
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_without_tags_omits_tag_kwargs(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        t = self._recall_tool(client)
        _call_tool(t, query="query")
        call_kwargs = client.recall.call_args[1]
        assert "tags" not in call_kwargs
        assert "tags_match" not in call_kwargs

    def test_recall_failure_raises_hms_error(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("network error")
        t = self._recall_tool(client)
        with pytest.raises(HMSError, match="Recall failed"):
            _call_tool(t, query="query")

    def test_recall_hms_error_not_wrapped(self):
        client = _mock_client()
        client.recall.side_effect = HMSError("original error")
        t = self._recall_tool(client)
        with pytest.raises(HMSError, match="original error"):
            _call_tool(t, query="query")

    def test_recall_failure_logs_error(self, caplog):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("network error")
        t = self._recall_tool(client)
        with caplog.at_level(logging.ERROR), pytest.raises(HMSError):
            _call_tool(t, query="query")
        assert "Recall failed" in caplog.text


# ---------------------------------------------------------------------------
# reflect tool
# ---------------------------------------------------------------------------


class TestReflectTool:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def _reflect_tool(self, client, **kwargs):
        tools = create_hms_tools(
            bank_id="my-bank",
            client=client,
            enable_retain=False,
            enable_recall=False,
            **kwargs,
        )
        return tools[0]

    def test_reflect_returns_text(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("Synthesized answer")
        t = self._reflect_tool(client)
        result = _call_tool(t, query="What are my preferences?")
        assert result == "Synthesized answer"

    def test_reflect_empty_text_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        t = self._reflect_tool(client)
        result = _call_tool(t, query="query")
        assert result == "No relevant memories found."

    def test_reflect_none_text_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response(None)
        t = self._reflect_tool(client)
        result = _call_tool(t, query="query")
        assert result == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        t = self._reflect_tool(client, budget="high")
        _call_tool(t, query="query")
        assert client.reflect.call_args[1]["budget"] == "high"

    def test_reflect_default_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        t = self._reflect_tool(client)
        _call_tool(t, query="query")
        assert client.reflect.call_args[1]["budget"] == "mid"

    def test_reflect_failure_raises_hms_error(self):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("network error")
        t = self._reflect_tool(client)
        with pytest.raises(HMSError, match="Reflect failed"):
            _call_tool(t, query="query")

    def test_reflect_hms_error_not_wrapped(self):
        client = _mock_client()
        client.reflect.side_effect = HMSError("original error")
        t = self._reflect_tool(client)
        with pytest.raises(HMSError, match="original error"):
            _call_tool(t, query="query")

    def test_reflect_failure_logs_error(self, caplog):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("network error")
        t = self._reflect_tool(client)
        with caplog.at_level(logging.ERROR), pytest.raises(HMSError):
            _call_tool(t, query="query")
        assert "Reflect failed" in caplog.text


# ---------------------------------------------------------------------------
# memory_instructions
# ---------------------------------------------------------------------------


class TestMemoryInstructions:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_formatted_memories(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["pref1", "pref2"])
        result = memory_instructions(bank_id="test", client=client)
        assert result == "Relevant memories:\n\n1. pref1\n2. pref2"

    def test_returns_empty_string_when_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        result = memory_instructions(bank_id="test", client=client)
        assert result == ""

    def test_respects_max_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(
            ["a", "b", "c", "d", "e", "f"]
        )
        result = memory_instructions(bank_id="test", client=client, max_results=3)
        lines = result.strip().split("\n")
        assert len(lines) == 5  # prefix + blank line + 3 results
        assert lines[-1] == "3. c"

    def test_custom_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        result = memory_instructions(bank_id="test", client=client, prefix="Context:\n")
        assert result.startswith("Context:\n")

    def test_passes_query_and_budget(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory_instructions(
            bank_id="test", client=client, query="custom query", budget="high"
        )
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["query"] == "custom query"
        assert call_kwargs["budget"] == "high"

    def test_default_budget_is_low(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory_instructions(bank_id="test", client=client)
        assert client.recall.call_args[1]["budget"] == "low"

    def test_passes_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory_instructions(
            bank_id="test", client=client, tags=["scope:global"], tags_match="all"
        )
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_returns_empty_on_exception(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("network error")
        result = memory_instructions(bank_id="test", client=client)
        assert result == ""

    def test_raises_without_client_or_config(self):
        with pytest.raises(HMSError, match="No HMS API URL"):
            memory_instructions(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hms_api_url="http://localhost:8888")
        with patch("hms_strands.tools.HMS") as mock_cls:
            mock_instance = _mock_client()
            mock_instance.recall.return_value = _mock_recall_response(["fact"])
            mock_cls.return_value = mock_instance
            result = memory_instructions(bank_id="test")
            assert "fact" in result
