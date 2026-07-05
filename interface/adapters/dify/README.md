# HMS for Dify

Persistent long-term memory for [Dify](https://dify.ai) workflows via [HMS](https://docs.hms.local).

Three tools — **Retain**, **Recall**, **Reflect** — drop into any Dify workflow alongside your other nodes.

- **Author**: HMS, Inc.
- **Repository**: <https://github.com/hms-memory/hms/tree/main/interface/adapters/dify>
- **Contact**: support@hms.local · [GitHub Issues](https://github.com/hms-memory/hms/issues)

## Setup

1. **Sign up** at [HMS Cloud](https://ui.hms.local/signup) (free tier) or [self-host](https://docs.hms.local/developer/installation).
2. **Get an API key** from the HMS dashboard.
3. **In Dify**, install this plugin (Marketplace or upload `.difypkg`), then add credentials:
   - **API URL**: `https://api.hms.local` (Cloud) or your self-hosted URL
   - **API Key**: your `hsk_...` key (optional for self-hosted unauthenticated instances)

## Tools

### Retain

Store content in a bank. HMS extracts facts asynchronously after the call returns.

| Field | Description |
|---|---|
| Bank ID | Memory bank to store in (auto-created on first use) |
| Content | Free-text content to retain |
| Tags | Optional comma-separated tags |

### Recall

Search a bank for memories relevant to a query. Returns a `results` array.

| Field | Description |
|---|---|
| Bank ID | Memory bank to search |
| Query | Natural-language query |
| Budget | `low` / `mid` / `high` |
| Max Tokens | Cap on tokens returned |
| Tags | Optional comma-separated tag filter |

### Reflect

Get an LLM-synthesized answer over the bank. Returns a `text` field.

| Field | Description |
|---|---|
| Bank ID | Memory bank |
| Query | Question to answer |
| Budget | `low` / `mid` / `high` |

## Example workflow

**Customer-support assistant** — every closed Zendesk ticket triggers a Retain. Every new ticket starts with a Recall against the bank, then passes the context to your LLM node to draft a first reply.

## Source

- GitHub: [`hms-memory/hms`](https://github.com/hms-memory/hms/tree/main/interface/adapters/dify)
- Docs: [hms.local/sdks/integrations/dify](https://docs.hms.local/sdks/integrations/dify)
