# @hms-memory/hms-paperclip

Persistent long-term memory for Paperclip agents via [HMS](https://github.com/hms-memory/hms).

Install once. Every agent in your Paperclip instance gets memory that persists across runs, companies, and restarts.

## What It Does

- **Before each run** — recalls relevant memories from past runs and caches them for the agent
- **After each run** — retains the agent's output to HMS automatically
- **Agent tools** — `hms_recall` and `hms_retain` tools for agents to query and store memory mid-run

## Installation

```bash
pnpm paperclipai plugin install @hms-memory/hms-paperclip
```

Then configure in **Settings → Plugins → HMS Memory**.

## Prerequisites

Either:

```bash
# Self-hosted (runs locally)
pip install hms-all
export HMS_API_LLM_API_KEY=your-openai-key
hms-api
```

Or [HMS Cloud](https://ui.hms.local/signup) — no self-hosting required.

## Configuration

| Field                | Default                 | Description                                                    |
| -------------------- | ----------------------- | -------------------------------------------------------------- |
| `hmsApiUrl`    | `http://localhost:8888` | HMS server URL                                           |
| `hmsApiKeyRef` | —                       | Paperclip secret name holding HMS Cloud API key          |
| `bankGranularity`    | `["company", "agent"]`  | Memory isolation: per company+agent, per company, or per agent |
| `recallBudget`       | `mid`                   | `low` = fastest, `mid` = balanced, `high` = most thorough      |
| `autoRetain`         | `true`                  | Automatically retain run output after every run                |

## Bank ID Format

```
paperclip::{companyId}::{agentId}    ← default (company + agent granularity)
paperclip::{companyId}               ← company granularity (shared across agents)
paperclip::{agentId}                 ← agent granularity (agent memory across companies)
```

## Agent Tools

Agents can call these tools directly during a run:

**`hms_recall(query)`** — search memory for relevant context. Called automatically at run start; agents can also call it mid-run for targeted queries.

**`hms_retain(content)`** — store a fact or decision immediately, without waiting for run end.

## How It Works

```
agent.run.started
  └─ recall(issueTitle + description)
       └─ store in plugin state for this run (instant lookup by tools)

agent running…
  ├─ hms_recall(query) → returns cached context or live recall
  └─ hms_retain(content) → stores immediately

agent.run.finished
  └─ retain(output) → stored in HMS with runId as document_id
```

Memory is keyed to `companyId` + `agentId`, never to the Paperclip session or run ID — so it survives across any number of runs.

## Development

```bash
npm install
npm run build
npm test
```

Local install into a running Paperclip instance:

```bash
curl -X POST http://127.0.0.1:3100/api/plugins/install \
  -H "Content-Type: application/json" \
  -d '{"packageName":"/absolute/path/to/interface/adapters/paperclip","isLocalPath":true}'
```
