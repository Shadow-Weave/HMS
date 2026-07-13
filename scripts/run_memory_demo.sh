#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  bash scripts/generate_keys.sh
  echo "Created .env. Fill the model Base URL, API key, and model fields, then rerun:" >&2
  echo "  bash scripts/run_memory_demo.sh" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

first_valid_value() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && "$candidate" != *_change_me ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

model_key="$(first_valid_value "${OPENAI_API_KEY:-}" "${HMS_API_LLM_API_KEY:-}" || true)"
if [[ -z "$model_key" ]]; then
  echo "Configure OPENAI_API_KEY or HMS_API_LLM_API_KEY in .env first." >&2
  exit 2
fi

model_base_url="${OPENAI_BASE_URL:-${HMS_API_LLM_BASE_URL:-https://api.openai.com/v1}}"
model_name="${OPENAI_MODEL:-${HMS_API_LLM_MODEL:-gpt-4o-mini}}"

# One set of OpenAI-compatible credentials is enough for the demo. Reuse it for
# HMS core reasoning, retain extraction, and embeddings when role-specific
# values are empty or still contain template placeholders.
export HMS_API_LLM_API_KEY="$model_key"
export HMS_API_LLM_BASE_URL="$model_base_url"
export HMS_API_LLM_MODEL="$model_name"

if [[ -z "${HMS_API_RETAIN_LLM_API_KEY:-}" || "${HMS_API_RETAIN_LLM_API_KEY}" == *_change_me ]]; then
  export HMS_API_RETAIN_LLM_API_KEY="$model_key"
fi
if [[ -z "${HMS_API_RETAIN_LLM_BASE_URL:-}" ]]; then
  export HMS_API_RETAIN_LLM_BASE_URL="$model_base_url"
fi
if [[ -z "${HMS_API_RETAIN_LLM_MODEL:-}" ]]; then
  export HMS_API_RETAIN_LLM_MODEL="$model_name"
fi
if [[ -z "${HMS_API_EMBEDDINGS_OPENAI_API_KEY:-}" || "${HMS_API_EMBEDDINGS_OPENAI_API_KEY}" == *_change_me ]]; then
  export HMS_API_EMBEDDINGS_OPENAI_API_KEY="$model_key"
fi
if [[ -z "${HMS_API_EMBEDDINGS_OPENAI_BASE_URL:-}" ]]; then
  export HMS_API_EMBEDDINGS_OPENAI_BASE_URL="$model_base_url"
fi

memory_key="$(first_valid_value "${HMS_MEMORY_API_KEY:-}" "${HMS_API_TENANT_API_KEY:-}" || true)"
if [[ -z "$memory_key" ]]; then
  echo "Configure HMS_MEMORY_API_KEY or run scripts/generate_keys.sh." >&2
  exit 2
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Docker Compose is required for the one-click demo." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv, then rerun this script." >&2
  exit 2
fi

echo "[1/3] Starting PostgreSQL and the HMS API..."
"${COMPOSE[@]}" up -d --build postgres hms-api

api_url="${HMS_MEMORY_API_URL:-http://127.0.0.1:${HMS_LOCAL_API_PORT:-18080}}"
health_url="${api_url%/}/health"
echo "[2/3] Waiting for HMS at ${health_url}..."
for attempt in $(seq 1 90); do
  if curl -fsS "$health_url" >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 90 ]]; then
    echo "HMS did not become healthy. Inspect: ${COMPOSE[*]} logs hms-api" >&2
    exit 1
  fi
  sleep 2
done

echo "[3/3] Running the two-turn automatic retain/recall demo..."
uv run \
  --no-project \
  --with openai \
  --with-editable "$ROOT_DIR/interface/sdk/python" \
  --with-editable "$ROOT_DIR/interface/adapters/litellm" \
  python "$ROOT_DIR/examples/automatic_memory/openai_responses.py"
