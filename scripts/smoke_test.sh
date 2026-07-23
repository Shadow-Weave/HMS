#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env ]]; then
  echo ".env not found. Run: cp .env.example .env && bash scripts/generate_keys.sh" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

BASE_URL="${PUBLIC_BASE_URL:-http://localhost:${PUBLIC_HTTP_PORT:-8080}}"
API_KEY="${HMS_SMOKE_API_KEY:-${HMS_GATEWAY_API_KEYS%%,*}}"
BANK_ID="${HMS_SMOKE_BANK_ID:-vendor-smoke}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

status_code() {
  local method="$1"
  local url="$2"
  local body="${3:-}"
  local auth="${4:-}"
  local output="${5:-${tmpdir}/response.json}"

  if [[ -n "${body}" && -n "${auth}" ]]; then
    curl -sS -o "${output}" -w "%{http_code}" -X "${method}" "${url}" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${auth}" \
      --data-binary @"${body}"
  elif [[ -n "${body}" ]]; then
    curl -sS -o "${output}" -w "%{http_code}" -X "${method}" "${url}" \
      -H "Content-Type: application/json" \
      --data-binary @"${body}"
  elif [[ -n "${auth}" ]]; then
    curl -sS -o "${output}" -w "%{http_code}" -X "${method}" "${url}" \
      -H "Authorization: Bearer ${auth}"
  else
    curl -sS -o "${output}" -w "%{http_code}" -X "${method}" "${url}"
  fi
}

echo "1. Checking gateway health at ${BASE_URL}/health"
health_code="$(status_code GET "${BASE_URL}/health")"
if [[ "${health_code}" != "200" ]]; then
  echo "Health check failed with HTTP ${health_code}" >&2
  cat "${tmpdir}/response.json" >&2 || true
  exit 1
fi

cat >"${tmpdir}/organize.json" <<'JSON'
{
  "question": "What drink does the user prefer in the afternoon?",
  "question_date": "2026-01-02T00:00:00Z",
  "recall_response": {
    "bank_id": "smoke",
    "results": [
      {
        "id": "m1",
        "text": "The user prefers tea in the afternoon.",
        "type": "experience",
        "document_id": "demo-session-1",
        "mentioned_at": "2026-01-01T10:00:00Z"
      }
    ],
    "chunks": {}
  }
}
JSON

echo "2. Verifying protected route rejects missing API key"
unauth_code="$(status_code POST "${BASE_URL}/v1/vendor/organize" "${tmpdir}/organize.json")"
if [[ "${unauth_code}" != "401" ]]; then
  echo "Expected 401 without key, got HTTP ${unauth_code}" >&2
  cat "${tmpdir}/response.json" >&2 || true
  exit 1
fi

echo "3. Verifying /v1/vendor/organize returns evidence_packet"
organize_code="$(status_code POST "${BASE_URL}/v1/vendor/organize" "${tmpdir}/organize.json" "${API_KEY}")"
if [[ "${organize_code}" != "200" ]]; then
  echo "Organize failed with HTTP ${organize_code}" >&2
  cat "${tmpdir}/response.json" >&2 || true
  exit 1
fi
if ! grep -q '"evidence_packet"' "${tmpdir}/response.json"; then
  echo "Organize response did not include evidence_packet" >&2
  cat "${tmpdir}/response.json" >&2 || true
  exit 1
fi

if [[ "${HMS_SMOKE_RUN_PIPELINE:-0}" == "1" ]]; then
  echo "4. Running optional retain + recall + result-formatting pipeline"
  python3 - <<PY >"${tmpdir}/pipeline.json"
import json

payload = {
    "bank_id": "${BANK_ID}",
    "sessions": [
        {
            "session_id": "demo-session-1",
            "timestamp": "2026-01-01T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I prefer tea in the afternoon."},
                {"role": "assistant", "content": "Noted."},
            ],
        }
    ],
    "question": "What drink does the user prefer in the afternoon?",
    "question_date": "2026-01-02T00:00:00Z",
    "bank_profile": {
        "retain_mission": "Extract persistent user preferences and updates.",
        "reflect_mission": "Answer from recalled memory evidence.",
    },
    "create_bank": True,
    "reset_bank": True,
    "retain_async": False,
    "wait_for_retain": True,
    "recall_budget": "mid",
    "organize": True,
}
print(json.dumps(payload))
PY
  pipeline_code="$(status_code POST "${BASE_URL}/v1/vendor/pipeline" "${tmpdir}/pipeline.json" "${API_KEY}")"
  if [[ "${pipeline_code}" != "200" ]]; then
    echo "Pipeline failed with HTTP ${pipeline_code}" >&2
    cat "${tmpdir}/response.json" >&2 || true
    exit 1
  fi
  if ! grep -q '"evidence_packet"' "${tmpdir}/response.json"; then
    echo "Pipeline response did not include evidence_packet" >&2
    cat "${tmpdir}/response.json" >&2 || true
    exit 1
  fi
else
  echo "4. Skipping optional pipeline. Set HMS_SMOKE_RUN_PIPELINE=1 after model keys are configured."
fi

echo "Smoke test passed for ${BASE_URL}"
