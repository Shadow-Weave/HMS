"""Unit tests for HMS AutoGen tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import CancellationToken
from hms_autogen import (
    configure,
    create_hms_tools,
    reset_config,
)
from hms_autogen.errors import HMSError


def _mock_client():
    """Create a mock HMS client with async methods."""
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
    return client


def _mock_recall_response(texts: list[str]):
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


def _mock_reflect_response(text: str):
    response = MagicMock()
    response.text = text
    return response


def _mock_retain_response():
    response = MagicMock()
    response.success = True
    return response


async def _run(tool, args: dict) -> str:
    """Invoke an AutoGen FunctionTool and return the string result."""
    result = await tool.run_json(args, CancellationToken())
    return str(result)


class TestCreateHMSTools:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_three_tools_by_default(self):
        client = _mock_client()
        tools = create_hms_tools(bank_id="test", client=client)
        assert len(tools) == 3

    def test_tool_names(self):
        client = _mock_client()
        tools = create_hms_tools(bank_id="test", client=client)
        names = [t.name for t in tools]
        assert names == ["hms_retain", "hms_recall", "hms_reflect"]

    def test_include_retain_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=True,
            include_recall=False,
            include_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].name == "hms_retain"

    def test_include_recall_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=True,
            include_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].name == "hms_recall"

    def test_include_reflect_only(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
            include_reflect=True,
        )
        assert len(tools) == 1
        assert tools[0].name == "hms_reflect"

    def test_no_tools_when_all_excluded(self):
        client = _mock_client()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
            include_reflect=False,
        )
        assert len(tools) == 0

    def test_raises_without_client_or_config(self):
        with pytest.raises(HMSError, match="No HMS API URL"):
            create_hms_tools(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hms_api_url="http://localhost:8888")
        with patch("hms_autogen._client.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            tools = create_hms_tools(bank_id="test")
            assert len(tools) == 3
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_explicit_url_overrides_config(self):
        configure(hms_api_url="http://config:8888")
        with patch("hms_autogen._client.HMS") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_hms_tools(bank_id="test", hms_api_url="http://explicit:9999")
            mock_cls.assert_called_once_with(base_url="http://explicit:9999", timeout=30.0)


class TestRetainTool:
    @pytest.mark.asyncio
    async def test_retain_stores_memory(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hms_tools(
            bank_id="test-bank",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        result = await _run(tools[0], {"content": "The user likes Python"})
        assert result == "Memory stored successfully."
        client.aretain.assert_called_once_with(bank_id="test-bank", content="The user likes Python")

    @pytest.mark.asyncio
    async def test_retain_passes_tags(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hms_tools(
            bank_id="test-bank",
            client=client,
            tags=["source:chat"],
            include_recall=False,
            include_reflect=False,
        )
        await _run(tools[0], {"content": "some content"})
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    @pytest.mark.asyncio
    async def test_retain_passes_metadata(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            retain_metadata={"source": "chat", "session": "abc"},
            include_recall=False,
            include_reflect=False,
        )
        await _run(tools[0], {"content": "content"})
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "chat", "session": "abc"}

    @pytest.mark.asyncio
    async def test_retain_passes_document_id(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            retain_document_id="session-123",
            include_recall=False,
            include_reflect=False,
        )
        await _run(tools[0], {"content": "content"})
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["document_id"] == "session-123"

    @pytest.mark.asyncio
    async def test_retain_raises_hms_error(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        with pytest.raises(HMSError, match="Retain failed"):
            await _run(tools[0], {"content": "content"})


class TestRecallTool:
    @pytest.mark.asyncio
    async def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        tools = create_hms_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        result = await _run(tools[0], {"query": "user preferences"})
        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        result = await _run(tools[0], {"query": "anything"})
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            budget="high",
            max_tokens=2048,
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_recall_passes_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    @pytest.mark.asyncio
    async def test_recall_passes_types(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            recall_types=["world", "experience"],
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["types"] == ["world", "experience"]

    @pytest.mark.asyncio
    async def test_recall_passes_include_entities(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            recall_include_entities=True,
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["include_entities"] is True

    @pytest.mark.asyncio
    async def test_recall_raises_hms_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("timeout")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        with pytest.raises(HMSError, match="Recall failed"):
            await _run(tools[0], {"query": "query"})


class TestReflectTool:
    @pytest.mark.asyncio
    async def test_reflect_returns_text(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        tools = create_hms_tools(
            bank_id="test-bank",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        result = await _run(tools[0], {"query": "What do you know about the user?"})
        assert result == "The user is a Python developer who prefers functional patterns."

    @pytest.mark.asyncio
    async def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        result = await _run(tools[0], {"query": "anything"})
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_reflect_passes_budget(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            budget="high",
            include_retain=False,
            include_recall=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    @pytest.mark.asyncio
    async def test_reflect_passes_context(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            reflect_context="The user is asking about project setup",
            include_retain=False,
            include_recall=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["context"] == "The user is asking about project setup"

    @pytest.mark.asyncio
    async def test_reflect_passes_max_tokens_and_response_schema(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            reflect_max_tokens=2048,
            reflect_response_schema=schema,
            include_retain=False,
            include_recall=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_schema"] == schema

    @pytest.mark.asyncio
    async def test_reflect_passes_tags(self):
        client = _mock_client()
        client.areflect.return_value = _mock_reflect_response("answer")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            reflect_tags=["scope:global"],
            reflect_tags_match="all",
            include_retain=False,
            include_recall=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.areflect.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    @pytest.mark.asyncio
    async def test_reflect_raises_hms_error(self):
        client = _mock_client()
        client.areflect.side_effect = RuntimeError("timeout")
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_recall=False,
        )
        with pytest.raises(HMSError, match="Reflect failed"):
            await _run(tools[0], {"query": "query"})


class TestConfigFallback:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    @pytest.mark.asyncio
    async def test_config_budget_used_when_no_explicit(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        configure(
            hms_api_url="http://localhost:8888",
            budget="low",
        )
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "low"

    @pytest.mark.asyncio
    async def test_explicit_budget_overrides_config(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        configure(
            hms_api_url="http://localhost:8888",
            budget="low",
        )
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            budget="high",
            include_retain=False,
            include_reflect=False,
        )
        await _run(tools[0], {"query": "query"})
        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "high"

    @pytest.mark.asyncio
    async def test_config_tags_used_for_retain(self):
        client = _mock_client()
        client.aretain.return_value = _mock_retain_response()
        configure(
            hms_api_url="http://localhost:8888",
            tags=["env:test"],
        )
        tools = create_hms_tools(
            bank_id="test",
            client=client,
            include_recall=False,
            include_reflect=False,
        )
        await _run(tools[0], {"content": "content"})
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["env:test"]
