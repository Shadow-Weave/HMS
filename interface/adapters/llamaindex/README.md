# hms-llamaindex

LlamaIndex integration for [HMS](https://github.com/hms-memory/hms) — persistent long-term memory for AI agents.

Provides two complementary patterns:

- **Tools** (`HMSToolSpec`) — Agent-driven memory via LlamaIndex's `BaseToolSpec`. The agent decides when to retain/recall/reflect.
- **Memory** (`HMSMemory`) — Automatic memory via LlamaIndex's `BaseMemory` interface. Messages are stored on every turn and recalled as context.

## Installation

```bash
pip install hms-llamaindex
```

## Quick Start: Agent Tools

```python
import asyncio
from hms_client import HMS
from hms_llamaindex import HMSToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

async def main():
    client = HMS(base_url="http://localhost:8888")

    spec = HMSToolSpec(
        client=client,
        bank_id="user-123",
        mission="Track user preferences",
    )
    tools = spec.to_tool_list()

    agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-4o"))
    response = await agent.run("Remember that I prefer dark mode")
    print(response)

asyncio.run(main())
```

## Quick Start: Automatic Memory

```python
from hms_client import HMS
from hms_llamaindex import HMSMemory

client = HMS(base_url="http://localhost:8888")
memory = HMSMemory.from_client(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
)

agent = ReActAgent(tools=tools, llm=llm, memory=memory)
```

## Configuration

```python
from hms_llamaindex import configure

configure(
    hms_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    tags=["source:llamaindex"],
    context="my-app",
    mission="Track user preferences",
)
```

## Requirements

- Python 3.10+
- `llama-index-core >= 0.11.0`
- `hms-client >= 0.4.0`

## Documentation

- [Integration docs](https://docs.hms.local/docs/sdks/integrations/llamaindex)
- [HMS API docs](https://docs.hms.local)
