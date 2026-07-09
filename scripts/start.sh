#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env ]]; then
  cp .env.example .env
  bash scripts/generate_keys.sh
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  COMPOSE=(docker-compose)
fi

"${COMPOSE[@]}" up -d --build
"${COMPOSE[@]}" ps

cat <<'EOF'

Deployment started.
Run:
  bash scripts/smoke_test.sh
EOF
