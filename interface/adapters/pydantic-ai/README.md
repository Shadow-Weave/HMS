# hms-pydantic-ai

Persistent memory tools for Pydantic AI agents via HMS. Give your agents long-term memory with retain, recall, and reflect — all async-native with no thread-pool hacks.

## Features

- **Async-Native Tools** - Uses Pydantic AI's async tool interface directly (`aretain`, `arecall`, `areflect`)
- **Memory Instructions** - Auto-inject relevant memories into every agent run via `instructions=[...]`
- **Three Memory Tools** - Retain (store), Recall (search), Reflect (synthesize) — include any combination
- **Simple Configuration** - Configure once globally, or pass a client directly
- **Lightweight** - Depends on `pydantic-ai-slim` to avoid pulling in all model providers

## Installation

```bash
pip install hms-pydantic-ai
```

## Quick Start

```python
from hms_client import HMS
from hms_pydantic_ai import create_hms_tools, memory_instructions
from pydantic_ai import Agent

client = HMS(base_url="http://localhost:8888")

agent = Agent(
    "openai:gpt-4o",
    tools=create_hms_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)

result = await agent.run("What do you remember about my preferences?")
print(result.output)
```

The agent now has three tools it can call:

- **`hms_retain`** — Store information to long-term memory
- **`hms_recall`** — Search long-term memory for relevant facts
- **`hms_reflect`** — Synthesize a reasoned answer from memories

The `memory_instructions` callable automatically recalls relevant memories and injects them into the system prompt on every run.

## Tools Only (No Auto-Injection)

If you want the agent to decide when to use memory (rather than always injecting context):

```python
agent = Agent(
    "openai:gpt-4o",
    tools=create_hms_tools(client=client, bank_id="user-123"),
)
```

## Instructions Only (No Tools)

If you just want memories auto-injected without giving the agent explicit memory tools:

```python
agent = Agent(
    "openai:gpt-4o",
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)
```

## Selecting Tools

Include only the tools you need:

```python
tools = create_hms_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=True,
    include_reflect=False,  # Omit reflect
)
```

## Global Configuration

Instead of passing a client to every call, configure once:

```python
from hms_pydantic_ai import configure, create_hms_tools

configure(
    hms_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HMS_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode: any/all/any_strict/all_strict
)

# Now create tools without passing client — uses global config
tools = create_hms_tools(bank_id="user-123")
```

## Per-Tool Overrides

Constructor arguments override global configuration:

```python
tools = create_hms_tools(
    bank_id="user-123",
    budget="high",             # Override global budget
    max_tokens=8192,           # Override global max_tokens
    tags=["session:abc"],      # Override global tags
)
```

## Memory Instructions Options

Customize what memories get injected and how:

```python
instructions_fn = memory_instructions(
    client=client,
    bank_id="user-123",
    query="relevant context about the user",  # What to search for
    budget="low",                              # Keep it fast
    max_results=5,                             # Limit injected memories
    max_tokens=4096,                           # Max recall tokens
    prefix="Relevant memories:\n",             # Text before the memory list
    tags=["scope:global"],                     # Filter by tags
    tags_match="any",                          # Tag match mode
)
```

## Configuration Reference

### `create_hms_tools()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | HMS memory bank ID |
| `client` | `None` | Pre-configured HMS client |
| `hms_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `budget` | `"mid"` | Recall/reflect budget level (low/mid/high) |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `tags` | `None` | Tags applied when storing memories |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode |
| `include_retain` | `True` | Include the retain (store) tool |
| `include_recall` | `True` | Include the recall (search) tool |
| `include_reflect` | `True` | Include the reflect (synthesize) tool |

### `memory_instructions()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | HMS memory bank ID |
| `client` | `None` | Pre-configured HMS client |
| `hms_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `query` | `"relevant context about the user"` | Recall query for memory injection |
| `budget` | `"low"` | Recall budget level |
| `max_results` | `5` | Maximum memories to inject |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `prefix` | `"Relevant memories:\n"` | Text prepended before memory list |
| `tags` | `None` | Tags to filter recall results |
| `tags_match` | `"any"` | Tag matching mode |

### `configure()`

| Parameter | Default | Description |
|---|---|---|
| `hms_api_url` | Production API | HMS API URL |
| `api_key` | `HMS_API_KEY` env | API key for authentication |
| `budget` | `"mid"` | Default recall budget level |
| `max_tokens` | `4096` | Default max tokens for recall |
| `tags` | `None` | Default tags for retain operations |
| `recall_tags` | `None` | Default tags to filter recall |
| `recall_tags_match` | `"any"` | Default tag matching mode |
| `verbose` | `False` | Enable verbose logging |

## Requirements

- Python >= 3.10
- pydantic-ai-slim >= 1.0.0
- hms-client >= 0.4.0
- A running HMS API server

## License

MIT
