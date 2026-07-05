"""Unit tests for HMS LangGraph BaseStore adapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from hms_langgraph.errors import HMSError
from hms_langgraph.store import (
    HMSStore,
    _namespace_to_bank_id,
    _parse_value,
)


def _mock_client():
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.acreate_bank = AsyncMock()
    return client


def _mock_recall_response(texts: list[str], document_ids: list[str] | None = None):
    response = MagicMock()
    results = []
    for i, t in enumerate(texts):
        r = MagicMock()
        r.text = t
        r.document_id = document_ids[i] if document_ids else None
        r.occurred_start = None
        results.append(r)
    response.results = results
    return response


class TestNamespaceMapping:
    def test_simple_namespace(self):
        assert _namespace_to_bank_id(("user", "123")) == "user.123"

    def test_single_element(self):
        assert _namespace_to_bank_id(("memories",)) == "memories"

    def test_empty_namespace(self):
        assert _namespace_to_bank_id(()) == "default"

    def test_deep_namespace(self):
        assert (
            _namespace_to_bank_id(("org", "team", "user", "123")) == "org.team.user.123"
        )


class TestParseValue:
    def test_parses_json_dict(self):
        assert _parse_value('{"name": "Alice"}') == {"name": "Alice"}

    def test_wraps_plain_text(self):
        assert _parse_value("hello world") == {"text": "hello world"}

    def test_wraps_json_non_dict(self):
        assert _parse_value("[1, 2, 3]") == {"text": "[1, 2, 3]"}


class TestHMSStorePut:
    @pytest.mark.asyncio
    async def test_put_calls_retain(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "123"), "pref-1", {"color": "blue"})

        client.aretain.assert_called_once()
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["bank_id"] == "user.123"
        assert call_kwargs["document_id"] == "pref-1"
        assert json.loads(call_kwargs["content"]) == {"color": "blue"}

    @pytest.mark.asyncio
    async def test_put_passes_tags(self):
        client = _mock_client()
        store = HMSStore(client=client, tags=["source:langgraph"])

        await store.aput(("user", "123"), "key", {"value": 1})

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["source:langgraph"]

    @pytest.mark.asyncio
    async def test_put_tracks_namespace(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "123"), "key", {"value": 1})

        namespaces = await store.alist_namespaces()
        assert ("user", "123") in namespaces

    @pytest.mark.asyncio
    async def test_put_none_value_is_delete_noop(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.adelete(("user", "123"), "key")

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_put_raises_on_error(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        store = HMSStore(client=client)

        with pytest.raises(HMSError, match="Store put failed"):
            await store.aput(("user", "123"), "key", {"value": 1})


class TestHMSStoreGet:
    @pytest.mark.asyncio
    async def test_get_returns_item_by_document_id(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ['{"color": "blue"}'], document_ids=["pref-1"]
        )
        store = HMSStore(client=client)

        item = await store.aget(("user", "123"), "pref-1")

        assert item is not None
        assert item.namespace == ("user", "123")
        assert item.key == "pref-1"
        assert item.value == {"color": "blue"}

    @pytest.mark.asyncio
    async def test_get_returns_none_when_empty(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        store = HMSStore(client=client)

        item = await store.aget(("user", "123"), "nonexistent")

        assert item is None

    @pytest.mark.asyncio
    async def test_get_handles_error_gracefully(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("timeout")
        store = HMSStore(client=client)

        item = await store.aget(("user", "123"), "key")

        assert item is None


class TestHMSStoreSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["User likes Python", "User is in NYC"]
        )
        store = HMSStore(client=client)

        results = await store.asearch(("user", "123"), query="preferences")

        assert len(results) == 2
        assert results[0].value == {"text": "User likes Python"}
        assert results[1].value == {"text": "User is in NYC"}

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            ["fact1", "fact2", "fact3", "fact4", "fact5"]
        )
        store = HMSStore(client=client)

        results = await store.asearch(("user", "123"), query="facts", limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        store = HMSStore(client=client)

        results = await store.asearch(("user", "123"), query="anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_filter(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(
            [
                '{"type": "preference", "text": "likes Python"}',
                '{"type": "fact", "text": "lives in NYC"}',
            ]
        )
        store = HMSStore(client=client)

        results = await store.asearch(
            ("user", "123"), query="info", filter={"type": "preference"}
        )

        assert len(results) == 1
        assert results[0].value["type"] == "preference"


class TestHMSStoreListNamespaces:
    @pytest.mark.asyncio
    async def test_lists_known_namespaces(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "123"), "k1", {"v": 1})
        await store.aput(("user", "456"), "k2", {"v": 2})

        namespaces = await store.alist_namespaces()

        assert len(namespaces) == 2
        assert ("user", "123") in namespaces
        assert ("user", "456") in namespaces

    @pytest.mark.asyncio
    async def test_list_respects_max_depth(self):
        """max_depth truncates deep namespaces and deduplicates per BaseStore contract."""
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("a",), "k", {"v": 1})
        await store.aput(("a", "b", "c"), "k", {"v": 2})
        await store.aput(("x", "y"), "k", {"v": 3})

        namespaces = await store.alist_namespaces(max_depth=1)

        # ("a",) stays as-is, ("a", "b", "c") truncated to ("a",) and deduped,
        # ("x", "y") truncated to ("x",)
        assert ("a",) in namespaces
        assert ("x",) in namespaces
        assert ("a", "b", "c") not in namespaces
        assert ("x", "y") not in namespaces
        assert len(namespaces) == 2

    @pytest.mark.asyncio
    async def test_list_filters_by_prefix(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "123"), "k1", {"v": 1})
        await store.aput(("user", "456"), "k2", {"v": 2})
        await store.aput(("org", "abc"), "k3", {"v": 3})

        namespaces = await store.alist_namespaces(prefix=("user",))

        assert ("user", "123") in namespaces
        assert ("user", "456") in namespaces
        assert ("org", "abc") not in namespaces

    @pytest.mark.asyncio
    async def test_list_filters_by_suffix(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "prefs"), "k1", {"v": 1})
        await store.aput(("org", "prefs"), "k2", {"v": 2})
        await store.aput(("user", "history"), "k3", {"v": 3})

        namespaces = await store.alist_namespaces(suffix=("prefs",))

        assert ("user", "prefs") in namespaces
        assert ("org", "prefs") in namespaces
        assert ("user", "history") not in namespaces

    @pytest.mark.asyncio
    async def test_list_filters_by_prefix_and_suffix(self):
        client = _mock_client()
        store = HMSStore(client=client)

        await store.aput(("user", "prefs"), "k1", {"v": 1})
        await store.aput(("org", "prefs"), "k2", {"v": 2})
        await store.aput(("user", "history"), "k3", {"v": 3})

        namespaces = await store.alist_namespaces(prefix=("user",), suffix=("prefs",))

        assert namespaces == [("user", "prefs")]

    @pytest.mark.asyncio
    async def test_list_respects_limit(self):
        client = _mock_client()
        store = HMSStore(client=client)

        for i in range(5):
            await store.aput((f"ns-{i}",), "k", {"v": i})

        namespaces = await store.alist_namespaces(limit=2)

        assert len(namespaces) == 2
