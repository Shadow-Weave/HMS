# hms-langgraph

LangGraph and LangChain integration for [HMS](https://github.com/hms-memory/hms) — persistent long-term memory for AI agents.

Provides three integration patterns:
- **Tools** — retain/recall/reflect as LangChain `@tool` functions for agent-driven memory. Works with **both LangChain and LangGraph**.
- **Nodes** *(LangGraph)* — pre-built graph nodes for automatic memory injection and storage
- **BaseStore** *(LangGraph)* — drop-in `BaseStore` adapter for LangGraph's built-in memory system

## Prerequisites

- A running HMS instance ([self-hosted via Docker](https://github.com/hms-memory/hms#quick-start) or [HMS Cloud](https://ui.hms.local/signup))
- Python 3.10+

## Installation

```bash
pip install hms-langgraph
```

## Quick Start: Tools

Bind HMS memory tools to your LangGraph agent so it can store and retrieve memories on demand.

```python
from hms_client import HMS
from hms_langgraph import create_hms_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

client = HMS(base_url="http://localhost:8888")
tools = create_hms_tools(client=client, bank_id="user-123")

agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=tools,
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode"}]}
)
```

## Quick Start: Memory Nodes

Add recall and retain nodes to your graph for automatic memory injection before LLM calls and storage after responses.

```python
from hms_client import HMS
from hms_langgraph import create_recall_node, create_retain_node
from langgraph.graph import StateGraph, MessagesState, START, END

client = HMS(base_url="http://localhost:8888")

recall = create_recall_node(client=client, bank_id="user-123")
retain = create_retain_node(client=client, bank_id="user-123")

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent_node)  # your LLM node
builder.add_node("retain", retain)

builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "retain")
builder.add_edge("retain", END)

graph = builder.compile()
```

### Dynamic Bank IDs

Use `bank_id_from_config` to resolve the bank per-request from the graph's config:

```python
recall = create_recall_node(client=client, bank_id_from_config="user_id")
retain = create_retain_node(client=client, bank_id_from_config="user_id")

# Bank ID resolved at runtime
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    config={"configurable": {"user_id": "user-456"}},
)
```

## Quick Start: BaseStore

Use HMS as a LangGraph `BaseStore` for cross-thread persistent memory with semantic search.

```python
from hms_client import HMS
from hms_langgraph import HMSStore

client = HMS(base_url="http://localhost:8888")
store = HMSStore(client=client)

graph = builder.compile(checkpointer=checkpointer, store=store)

# Store and search memories via the store API
await store.aput(("user", "123", "prefs"), "theme", {"value": "dark mode"})
results = await store.asearch(("user", "123", "prefs"), query="theme preference")
```

## Configuration

### Global config

```python
from hms_langgraph import configure

configure(
    hms_api_url="http://localhost:8888",
    api_key="your-api-key",  # or set HMS_API_KEY env var
    budget="mid",
    tags=["source:langgraph"],
)
```

### Per-call overrides

All factory functions accept `client`, `hms_api_url`, and `api_key` to override the global config.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hms_api_url` | HMS API URL | `https://api.hms.local` |
| `api_key` | API key (or `HMS_API_KEY` env var) | `None` |
| `budget` | Recall budget: `low`, `mid`, `high` | `mid` |
| `max_tokens` | Max tokens for recall results | `4096` |
| `tags` | Tags applied to retain operations | `None` |
| `recall_tags` | Tags to filter recall results | `None` |
| `recall_tags_match` | Tag matching: `any`, `all`, `any_strict`, `all_strict` | `any` |

## Requirements

- Python 3.10+
- `langchain-core >= 0.3.0`
- `hms-client >= 0.4.0`
- `langgraph >= 0.3.0` *(only for nodes and store patterns — install with `pip install hms-langgraph[langgraph]`)*

## Documentation

- [Integration docs](https://docs.hms.local/docs/sdks/integrations/langgraph)
- [Cookbook: ReAct agent with memory](https://docs.hms.local/cookbook/recipes/langgraph-react-agent)
- [HMS API docs](https://docs.hms.local)
