<div align="center">

<img src="docs/assets/branding/hms-hero.png" alt="Holographic Memory System" width="94%">

### Structured Memory Intelligence for Reliable Long-Horizon Reasoning

<table>
  <tr>
    <td valign="middle"><strong>ShadowWeave Team</strong></td>
    <td width="74" align="center" valign="middle">
      <img src="docs/assets/branding/shadowweave-mark.png" alt="ShadowWeave" width="62">
    </td>
  </tr>
</table>

<a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-coming_soon-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: coming soon"></a>
<img src="https://img.shields.io/badge/status-active-145DA0?style=flat-square" alt="Project status: active">

[English](README.md) · [中文](README.zh-CN.md)

</div>

---

## Overview

The **Holographic Memory System (HMS)** is a structured long-term memory layer
for AI applications. It retains conversations and documents, extracts durable
facts, links related entities and events, and recalls relevant context for later
model calls.

HMS is designed for applications that need memory across sessions without
placing an entire conversation history into every prompt.

## One-Command Automatic Memory

HMS can wrap an existing OpenAI client so each model call automatically:

```text
user input -> recall relevant memories -> inject context -> call the LLM
           -> retain the completed user/assistant exchange
```

Configure the model Base URL, API key, and model in `.env`, then run:

```bash
bash scripts/run_memory_demo.sh
```

The script starts PostgreSQL and HMS locally, waits for the memory API, installs
the local SDK adapter in an isolated environment, and runs a two-turn demo. The
first turn stores a user preference and project; the second turn recalls both
without manually calling `retain()` or `recall()`.

The application-side integration is one wrapper call:

```python
from openai import OpenAI
from hms_litellm import wrap_openai

client = wrap_openai(
    OpenAI(),
    hms_api_url="http://127.0.0.1:18080",
    api_key="YOUR_HMS_API_KEY",
    bank_id="user-alice",
)

response = client.responses.create(
    model="gpt-4o-mini",
    input="What do you remember about my current project?",
)
```

`wrap_openai()` supports both `client.responses.create(...)` and
`client.chat.completions.create(...)`, including streaming. Use a stable,
per-user `bank_id`; optionally set `session_id` to accumulate one conversation
as a tracked HMS document.

## Memory Flow

```text
Retain
  -> parse source content
  -> extract structured memories
  -> resolve entities and links
  -> store facts, chunks, and provenance

Recall
  -> analyze the query
  -> retrieve semantic, lexical, graph, and temporal candidates
  -> fuse and rerank evidence
  -> return grounded memory context
```

HMS keeps source provenance and temporal metadata alongside extracted memory,
so applications can inspect where recalled information came from and when it
was observed.

## Visual Demo

The repository includes a database-free illustration of how retrieved sessions
can be organized into grounded evidence before generation.

![Memory evidence organization demo](docs/assets/memory_pipeline_demo.svg)

Open the standalone page directly in a browser:

```text
docs/memory_pipeline_demo.html
```

## Repository Layout

```text
.
├── core/
│   ├── dataplane/
│   ├── daemon/
│   └── local-suite/
├── deploy/
├── docs/
│   ├── assets/
│   └── memory_pipeline_demo.html
├── examples/
├── interface/
├── scripts/
├── vendor_gateway/
├── vendor_sdk/
├── .env.example
├── README.md
└── README.zh-CN.md
```

## Environment Setup

Create a local environment file:

```bash
cp .env.example .env
```

Configure the PostgreSQL connection, core model, retain model, and embedding
provider. Never commit the populated `.env` file.

Start the local stack:

```bash
bash scripts/start.sh
```

Run the smoke test:

```bash
bash scripts/smoke_test.sh
```

## Core Configuration

| Role | Provider | Model | Base URL | API key |
| --- | --- | --- | --- | --- |
| Core memory reasoning | `HMS_API_LLM_PROVIDER` | `HMS_API_LLM_MODEL` | `HMS_API_LLM_BASE_URL` | `HMS_API_LLM_API_KEY` |
| Retain extraction | `HMS_API_RETAIN_LLM_PROVIDER` | `HMS_API_RETAIN_LLM_MODEL` | `HMS_API_RETAIN_LLM_BASE_URL` | `HMS_API_RETAIN_LLM_API_KEY` |
| Embeddings | `HMS_API_EMBEDDINGS_PROVIDER` | `HMS_API_EMBEDDINGS_OPENAI_MODEL` | `HMS_API_EMBEDDINGS_OPENAI_BASE_URL` | `HMS_API_EMBEDDINGS_OPENAI_API_KEY` |

The core and retain roles may use the same OpenAI-compatible endpoint. Embedding
configuration can use a separate provider or a local model.

## Security Notes

- Keep `.env`, private keys, tokens, and populated credentials out of Git.
- Use separate internal and vendor-facing API keys.
- Use a stable tenant or bank boundary for each user or organization.
- Review gateway quotas and rate limits before exposing the service publicly.

## License

See [LICENSE](LICENSE).
