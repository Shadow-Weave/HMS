# HMS API

**Memory System for AI Agents** — Temporal + Semantic + Entity Memory Architecture using PostgreSQL with pgvector.

HMS gives AI agents persistent memory that works like human memory: it stores facts, tracks entities and relationships, handles temporal reasoning ("what happened last spring?"), and forms opinions based on configurable disposition traits.

## Installation

```bash
pip install hms-api
```

Install the optional Milvus semantic index provider when needed:

```bash
pip install "hms-api[milvus]"
```

## Quick Start

### Run the Server

```bash
# Set your LLM provider
export HMS_API_LLM_PROVIDER=openai
export HMS_API_LLM_API_KEY=sk-xxxxxxxxxxxx

# Start the server (uses embedded PostgreSQL by default)
hms-api
```

The server starts at http://localhost:8888 with:
- REST API for memory operations
- MCP server at `/mcp` for tool-use integration

### Use the Python API

```python
from hms_api import MemoryEngine

# Create and initialize the memory engine
memory = MemoryEngine()
await memory.initialize()

# Create a memory bank for your agent
bank = await memory.create_memory_bank(
    name="my-assistant",
    background="A helpful coding assistant"
)

# Store a memory
await memory.retain(
    memory_bank_id=bank.id,
    content="The user prefers Python for data science projects"
)

# Recall memories
results = await memory.recall(
    memory_bank_id=bank.id,
    query="What programming language does the user prefer?"
)

# Reflect with reasoning
response = await memory.reflect(
    memory_bank_id=bank.id,
    query="Should I recommend Python or R for this ML project?"
)
```

## CLI Options

```bash
hms-api --help

# Common options
hms-api --port 9000          # Custom port (default: 8888)
hms-api --host 127.0.0.1     # Bind to localhost only
hms-api --workers 4          # Multiple worker processes
hms-api --log-level debug    # Verbose logging
```

## Configuration

Configure via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HMS_API_DATABASE_URL` | PostgreSQL connection string | `pg0` (embedded) |
| `HMS_API_LLM_PROVIDER` | `openai`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio` | `openai` |
| `HMS_API_LLM_API_KEY` | API key for LLM provider | - |
| `HMS_API_LLM_MODEL` | Model name | `gpt-4o-mini` |
| `HMS_API_VECTOR_INDEX_PROVIDER` | Dense semantic index: `database` or `milvus` | `database` |
| `HMS_API_HOST` | Server bind address | `0.0.0.0` |
| `HMS_API_PORT` | Server port | `8888` |

### Example with External PostgreSQL

```bash
export HMS_API_DATABASE_URL=postgresql://user:pass@localhost:5432/hms
export HMS_API_LLM_PROVIDER=groq
export HMS_API_LLM_API_KEY=gsk_xxxxxxxxxxxx

hms-api
```

### Optional Milvus Semantic Index

Milvus can replace the database ANN index for dense semantic candidate retrieval. PostgreSQL or Oracle remains the canonical data store: HMS still hydrates every Milvus hit from SQL and continues to run full-text/BM25, graph, temporal, fusion, and reranking logic through the existing database-backed paths.

For a single-process development setup, Milvus Lite needs only a local file:

```bash
export HMS_API_VECTOR_INDEX_PROVIDER=milvus
export HMS_API_MILVUS_URI=./hms_milvus.db

hms-api
```

The same provider connects to Milvus Server or Zilliz Cloud by changing the URI and optional token:

```bash
export HMS_API_VECTOR_INDEX_PROVIDER=milvus
export HMS_API_MILVUS_URI=http://localhost:19530
# export HMS_API_MILVUS_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
# export HMS_API_MILVUS_TOKEN=your-token
export HMS_API_MILVUS_COLLECTION=hms_memory_units
export HMS_API_MILVUS_CONSISTENCY_LEVEL=Session
```

Available Milvus settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `HMS_API_MILVUS_URI` | Lite file path, Server URI, or Zilliz Cloud endpoint | `./hms_milvus.db` |
| `HMS_API_MILVUS_TOKEN` | Server or cloud authentication token | - |
| `HMS_API_MILVUS_DB_NAME` | Optional Milvus database name | - |
| `HMS_API_MILVUS_COLLECTION` | Shared HMS projection collection | `hms_memory_units` |
| `HMS_API_MILVUS_CONSISTENCY_LEVEL` | `Strong`, `Session`, `Bounded`, or `Eventually` | `Session` |

Keep `HMS_API_VECTOR_EXTENSION` configured. The database embedding column remains the source for SQL hydration, fallback search, and rebuilding the external projection. Existing databases should be backfilled after enabling Milvus:

```bash
hms-admin rebuild-vector-index --yes

# Rebuild only one bank or schema when needed
hms-admin rebuild-vector-index --bank-id my-bank --batch-size 1000 --yes
hms-admin rebuild-vector-index --schema tenant_acme --yes
```

External index mutations happen after the canonical SQL transaction commits. If a Milvus sync fails, HMS preserves the SQL write, marks the affected bank as degraded in the running process, and uses database semantic search until a rebuild succeeds. Run the rebuild command after a reported sync failure before relying on the Milvus projection again, including after restarting the process.
The health response keeps the service healthy while SQL fallback is available and reports the active provider plus `vector_index.degraded` for monitoring.

Milvus Lite is intended for a single HMS process. For multiple API workers or separate worker processes, use Milvus Server or Zilliz Cloud. Choose `Strong` consistency when immediate visibility across different Milvus clients is more important than the additional read latency.

## Docker

```bash
docker run --rm -it -p 8888:8888 \
  -e HMS_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hms-docker:/home/hms/.pg0 \
  ghcr.io/hms-memory/hms:latest
```

## MCP Server

For local MCP integration without running the full API server:

```bash
hms-local-mcp
```

This runs a stdio-based MCP server that can be used directly with MCP-compatible clients.

## Key Features

- **Multi-Strategy Retrieval (TEMPR)** — Semantic, keyword, graph, and temporal search combined with RRF fusion
- **Entity Graph** — Automatic entity extraction and relationship tracking
- **Temporal Reasoning** — Native support for time-based queries
- **Disposition Traits** — Configurable skepticism, literalism, and empathy influence opinion formation
- **Three Memory Types** — World facts, bank actions, and formed opinions with confidence scores

## Documentation

Full documentation: [https://docs.hms.local](https://docs.hms.local)

- [Installation Guide](https://docs.hms.local/developer/installation)
- [Configuration Reference](https://docs.hms.local/developer/configuration)
- [API Reference](https://docs.hms.local/api-reference)
- [Python SDK](https://docs.hms.local/sdks/python)

## License

Apache 2.0
