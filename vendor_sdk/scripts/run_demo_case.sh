#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CASE_PATH="${1:-$ROOT_DIR/examples/cases/shopping_actions.json}"
BANK_ID="${HMS_BANK_ID:-vendor-demo}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m hms_vendor_sdk.cli run-case \
  --case "$CASE_PATH" \
  --bank-id "$BANK_ID" \
  --create-bank \
  --reset-bank
