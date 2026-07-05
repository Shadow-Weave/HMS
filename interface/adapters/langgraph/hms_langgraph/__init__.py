"""HMS-LangGraph: Persistent memory for LangGraph and LangChain agents.

Provides HMS-backed tools, nodes, and a BaseStore adapter,
giving agents long-term memory across conversations.

The **tools** pattern works with both LangChain and LangGraph — only
``langchain-core`` is required. The **nodes** and **store** patterns
require ``langgraph`` (install with ``pip install hms-langgraph[langgraph]``).

Basic usage with tools (LangChain or LangGraph)::

    from hms_client import HMS
    from hms_langgraph import create_hms_tools

    client = HMS(base_url="http://localhost:8888")
    tools = create_hms_tools(client=client, bank_id="user-123")

    # Bind tools to your model
    model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

Usage with memory nodes (requires langgraph)::

    from hms_langgraph import create_recall_node, create_retain_node

    recall = create_recall_node(client=client, bank_id="user-123")
    retain = create_retain_node(client=client, bank_id="user-123")

    builder.add_node("recall", recall)
    builder.add_node("agent", agent_node)
    builder.add_node("retain", retain)
    builder.add_edge("recall", "agent")
    builder.add_edge("agent", "retain")

Usage with BaseStore (requires langgraph)::

    from hms_langgraph import HMSStore

    store = HMSStore(client=client)
    graph = builder.compile(checkpointer=checkpointer, store=store)
"""

from .config import (
    HMSLangGraphConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import create_hms_tools


def __getattr__(name: str):
    """Lazy-import LangGraph-specific modules so langgraph is optional."""
    if name == "create_recall_node" or name == "create_retain_node":
        try:
            from .nodes import create_recall_node, create_retain_node
        except ImportError:
            raise ImportError(
                f"'{name}' requires langgraph. Install with: pip install hms-langgraph[langgraph]"
            ) from None
        return create_recall_node if name == "create_recall_node" else create_retain_node

    if name == "HMSStore":
        try:
            from .store import HMSStore
        except ImportError:
            raise ImportError(
                "HMSStore requires langgraph. Install with: pip install hms-langgraph[langgraph]"
            ) from None
        return HMSStore

    raise AttributeError(f"module 'hms_langgraph' has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSLangGraphConfig",
    "HMSError",
    "create_hms_tools",
]

try:
    import langgraph  # noqa: F401

    __all__ += ["create_recall_node", "create_retain_node", "HMSStore"]
except ImportError:
    pass
