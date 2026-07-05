# HMS Pipecat Integration

Persistent long-term memory for [Pipecat](https://github.com/pipecat-ai/pipecat) voice AI pipelines via [HMS](https://docs.hms.local). A single `FrameProcessor` slots between your user context aggregator and LLM service — recalling relevant memories before each turn and retaining conversation content after.

## Quick Start

```bash
pip install hms-pipecat
```

```python
from pipecat.pipeline.pipeline import Pipeline
from hms_pipecat import HMSMemoryService

memory = HMSMemoryService(
    bank_id="user-123",
    hms_api_url="http://localhost:8888",
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    user_aggregator,
    memory,           # ← add between user_aggregator and LLM
    llm_service,
    assistant_aggregator,
    tts_service,
    transport.output(),
])
```

Or with [HMS Cloud](https://ui.hms.local/signup):

```python
memory = HMSMemoryService(
    bank_id="user-123",
    hms_api_url="https://api.hms.local",
    api_key="hsk_your_token_here",
)
```

## How It Works

```
New turn starts
  └─ OpenAILLMContextFrame arrives
       ├─ Retain previous complete turn (user+assistant) — fire-and-forget
       └─ Recall relevant memories for current user query
            └─ Inject as <hms_memories> system message
                 └─ Forward enriched context to LLM
```

On each `OpenAILLMContextFrame`:

1. **Retain** — any new complete user+assistant turn pairs are sent to HMS asynchronously (non-blocking)
2. **Recall** — the latest user message is used as the search query; results are injected as a system message before the LLM sees the context
3. **Forward** — the enriched context frame is pushed downstream

Memory accumulates across calls. By the third or fourth turn, recall starts surfacing useful context that the pipeline didn't have to re-establish.

## Prerequisites

A running HMS instance:

**Self-hosted:**
```bash
pip install hms-all
export HMS_API_LLM_API_KEY=your-api-key
hms-api  # starts on http://localhost:8888
```

**HMS Cloud:** [Sign up](https://ui.hms.local/signup) — no self-hosting required.

## Configuration

```python
HMSMemoryService(
    bank_id="user-123",               # Required: memory bank to use
    hms_api_url="...",          # HMS API URL
    api_key="hsk_...",                # API key (HMS Cloud)
    recall_budget="mid",              # "low", "mid", or "high"
    recall_max_tokens=4096,           # Max tokens for recall results
    enable_recall=True,               # Inject memories before LLM
    enable_retain=True,               # Store turns after each exchange
    memory_prefix="Relevant memories from past conversations:\n",
)
```

### Global configuration

```python
from hms_pipecat import configure

configure(
    hms_api_url="http://localhost:8888",
    api_key="hsk_...",
    recall_budget="mid",
)

# Now create services without repeating connection details
memory = HMSMemoryService(bank_id="user-123")
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```
