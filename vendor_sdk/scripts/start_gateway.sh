#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HMS_GATEWAY_HOST:-0.0.0.0}"
PORT="${HMS_GATEWAY_PORT:-18081}"

if [ -z "${HMS_GATEWAY_API_KEYS:-${HMS_GATEWAY_API_KEY:-}}" ]; then
  echo "HMS_GATEWAY_API_KEYS or HMS_GATEWAY_API_KEY is required" >&2
  exit 2
fi

if [ -z "${HMS_INTERNAL_BASE_URL:-}" ]; then
  echo "HMS_INTERNAL_BASE_URL is required, e.g. http://127.0.0.1:18080" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m hms_vendor_sdk.gateway --host "$HOST" --port "$PORT"
