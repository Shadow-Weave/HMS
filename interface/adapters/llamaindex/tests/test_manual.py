"""Manual integration test for HMS LlamaIndex tools.

Requires a running HMS server at http://localhost:8888.
Run with: uv run pytest tests/test_manual.py -v -s --no-header
"""

import uuid

import pytest
from hms_client import HMS
from hms_llamaindex import HMSToolSpec, create_hms_tools

HMS_URL = "http://localhost:8888"


@pytest.fixture
def client():
    return HMS(base_url=HMS_URL, timeout=30.0)


@pytest.fixture
def bank_id(client):
    bid = f"test-llamaindex-{uuid.uuid4().hex[:8]}"
    client.create_bank(bank_id=bid, name=bid)
    return bid


@pytest.mark.skip(reason="Requires running HMS server")
class TestManualToolSpec:
    def test_retain_and_recall_round_trip(self, client, bank_id):
        spec = HMSToolSpec(client=client, bank_id=bank_id)

        # Retain a memory
        result = spec.retain_memory("The user prefers dark mode in all applications.")
        assert result == "Memory stored successfully."

        # Recall it
        result = spec.recall_memory("What are the user's UI preferences?")
        assert "dark mode" in result.lower()

    def test_create_hms_tools_factory(self, client, bank_id):
        tools = create_hms_tools(client=client, bank_id=bank_id)
        assert len(tools) == 3

        # Find retain tool by name
        retain_tool = next(t for t in tools if t.metadata.name == "retain_memory")
        result = retain_tool("The user's favorite language is Python.")
        assert "stored" in result.lower()

    def test_reflect(self, client, bank_id):
        spec = HMSToolSpec(client=client, bank_id=bank_id)

        spec.retain_memory("The user is a backend developer.")
        spec.retain_memory("The user uses Python and Go daily.")
        spec.retain_memory("The user prefers vim keybindings.")

        result = spec.reflect_on_memory("What kind of developer is this user?")
        assert len(result) > 0
        assert result != "No relevant memories found."
