# HMS with Timescale Extensions

This Docker Compose setup provides a complete HMS deployment with **Timescale extensions**:
- **pgvectorscale** - DiskANN algorithm for disk-based scalable vector search  
- **pg_textsearch** - High-performance BM25 text search

Both extensions are from [Timescale](https://github.com/timescale) and provide production-grade performance.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (or another LLM provider)

## Quick Start

```bash
# Set environment variables
export HMS_DB_PASSWORD="your-secure-password"
export OPENAI_API_KEY="your-openai-api-key"

# Build and start
docker compose -f deploy/containers/docker-compose/timescale/docker-compose.yaml up -d --build

# Check logs

docker compose -f deploy/containers/docker-compose/timescale/docker-compose.yaml logs -f
```

**Access:**
- API: http://localhost:8888
- Control Plane: http://localhost:9999

## Stop and Clean Up

```bash
# Stop services
docker compose -f deploy/containers/docker-compose/timescale/docker-compose.yaml down

# Remove volumes (deletes all data)
docker compose -f deploy/containers/docker-compose/timescale/docker-compose.yaml down -v
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HMS_DB_PASSWORD` | PostgreSQL password | `hms_password` |
| `HMS_DB_USER` | PostgreSQL username | `hms_user` |
| `HMS_DB_NAME` | Database name | `hms_db` |
| `HMS_VERSION` | HMS Docker image version | `latest` |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `HMS_API_LLM_PROVIDER` | LLM provider | `openai` |

### Why Timescale Extensions?

**pgvectorscale (DiskANN):**
- 28x lower p95 latency vs dedicated vector databases
- 16x higher query throughput at 99% recall
- 60-75% cost reduction (disk is cheaper than RAM)
- Best for large datasets (10M+ vectors)

**pg_textsearch (BM25):**
- High-performance keyword retrieval
- Native BM25 ranking algorithm
- Optimized for full-text search

## Troubleshooting

### Extensions not installed

Check if extensions are available:

```bash
docker exec -it hms-db-timescale psql -U hms_user -d hms_db -c "\dx"
```

You should see:
- `vector` (pgvector)
- `vectorscale` (pgvectorscale/DiskANN)
- `pg_textsearch` (BM25 search)

### Build fails

If the Docker build fails during pgvectorscale compilation:

1. Ensure you have sufficient memory (recommended: 4GB+)
2. Check Docker build logs for Rust compilation errors
3. Try building with more resources: `docker compose build --no-cache --memory 4g`

### Port conflicts

If port 5438 is already in use, modify the `ports` section in docker-compose.yaml.

## Learn More

- [pgvectorscale GitHub](https://github.com/timescale/pgvectorscale)
- [pg_textsearch GitHub](https://github.com/timescale/pg_textsearch)
- [HNSW vs DiskANN](https://www.tigerdata.com/learn/hnsw-vs-diskann)
- [HMS Documentation](https://docs.hms.local)
