# hms-all

All-in-one package for HMS - Agent Memory That Works Like Human Memory

## Quick Start

```python
from hms import start_server, HMSClient

# Start server with embedded PostgreSQL
server = start_server(
    llm_provider="groq",
    llm_api_key="your-api-key",
    llm_model="openai/gpt-oss-120b"
)

# Create client
client = HMSClient(base_url=server.url)

# Store memories
client.put(agent_id="assistant", content="User prefers Python for data analysis")

# Search memories
results = client.search(agent_id="assistant", query="programming preferences")

# Generate contextual response
response = client.think(agent_id="assistant", query="What languages should I recommend?")

# Stop server when done
server.stop()
```

## Using Context Manager

```python
from hms import HMSServer, HMSClient

with HMSServer(llm_provider="groq", llm_api_key="...") as server:
    client = HMSClient(base_url=server.url)
    # ... use client ...
# Server automatically stops
```

## Installation

```bash
pip install hms-all
```
