# hms-ag2

AG2 integration for [HMS](https://github.com/hms-memory/hms) — persistent long-term memory for AI agents.

Provides HMS-backed tool functions that give [AG2](https://ag2.ai) agents long-term memory across conversations via retain/recall/reflect operations.

## Prerequisites

- Python 3.10+
- Running HMS instance ([quickstart](https://github.com/hms-memory/hms#quick-start))

## Installation

```bash
pip install hms-ag2
```

## Quick Start

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from hms_ag2 import register_hms_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant with long-term memory.",
    )
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
    )

# Register HMS memory tools on both agents
register_hms_tools(
    assistant, user_proxy,
    bank_id="my-bank",
    hms_api_url="http://localhost:8888",
)

# The assistant can now use hms_retain, hms_recall, hms_reflect
result = user_proxy.initiate_chat(
    assistant,
    message="Remember that I prefer Python over JavaScript.",
)
```

## Tools

| Tool | Operation | Description |
|------|-----------|-------------|
| `hms_retain` | Retain | Store facts, preferences, decisions to long-term memory |
| `hms_recall` | Recall | Multi-strategy search across stored memories |
| `hms_reflect` | Reflect | Synthesize reasoned answers from memories |

## Configuration

### Global config

```python
from hms_ag2 import configure

configure(
    hms_api_url="http://localhost:8888",
    api_key="your-key",       # or set HMS_API_KEY env var
    budget="mid",              # low / mid / high
    max_tokens=4096,
    tags=["source:ag2"],       # default tags for retain
)
```

### Per-call overrides

All global settings can be overridden per `create_hms_tools()` call:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `bank_id` | Memory bank ID (required) | — |
| `client` | Pre-configured `HMS` client | — |
| `hms_api_url` | API URL | Global config or production |
| `api_key` | API key | Global config or env var |
| `budget` | Recall/reflect budget | `"mid"` |
| `max_tokens` | Max tokens for recall | `4096` |
| `tags` | Tags for retain operations | `None` |
| `recall_tags` | Tags to filter recall | `None` |
| `recall_tags_match` | Tag match mode | `"any"` |
| `retain_metadata` | Metadata dict for retain | `None` |
| `retain_document_id` | Document ID for retain | `None` |
| `recall_types` | Fact types to filter | `None` |
| `recall_include_entities` | Include entities in recall | `False` |
| `reflect_context` | Additional context for reflect | `None` |
| `reflect_max_tokens` | Max tokens for reflect | `max_tokens` |
| `reflect_response_schema` | JSON schema for reflect output | `None` |
| `reflect_tags` | Tags for reflect (fallback: `recall_tags`) | `None` |
| `reflect_tags_match` | Tag match for reflect | `recall_tags_match` |

## Advanced: Manual Registration

```python
from hms_ag2 import create_hms_tools

tools = create_hms_tools(
    bank_id="my-bank",
    hms_api_url="http://localhost:8888",
)
for tool_fn in tools:
    assistant.register_for_llm(description=tool_fn.__doc__)(tool_fn)
    user_proxy.register_for_execution()(tool_fn)
```

## Advanced: GroupChat with Shared Memory

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig
from hms_ag2 import register_hms_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    researcher = AssistantAgent(name="researcher", system_message="You research topics.")
    writer = AssistantAgent(name="writer", system_message="You write content.")
    executor = UserProxyAgent(name="executor", human_input_mode="NEVER")

# All agents share the same memory bank
for agent in [researcher, writer]:
    register_hms_tools(agent, executor, bank_id="team-memory")

group_chat = GroupChat(agents=[researcher, writer, executor], messages=[])
manager = GroupChatManager(groupchat=group_chat)
```

## Requirements

- `ag2>=0.9.0`
- `hms-client>=0.4.0`

## Documentation

- [HMS Documentation](https://docs.hms.local)
- [AG2 Documentation](https://docs.ag2.ai)
- [API Reference](https://docs.hms.local/api)
