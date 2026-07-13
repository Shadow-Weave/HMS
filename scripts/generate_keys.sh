#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
ENV_EXAMPLE="${ROOT_DIR}/.env.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
fi

rand_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  else
    tr -dc 'a-f0-9' </dev/urandom | head -c 48
  fi
}

set_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_FILE}"
  else
    printf '\n%s=%s\n' "${key}" "${value}" >>"${ENV_FILE}"
  fi
}

postgres_password="hms_$(rand_key)"
internal_key="hms_internal_$(rand_key)"
vendor_key_a="hms_live_$(rand_key)"
vendor_key_b="hms_live_$(rand_key)"

set_env "POSTGRES_PASSWORD" "${postgres_password}"
set_env "HMS_API_DATABASE_URL" "postgresql://hms:${postgres_password}@postgres:5432/hms"
set_env "HMS_API_TENANT_API_KEY" "${internal_key}"
set_env "HMS_INTERNAL_API_KEY" "${internal_key}"
set_env "HMS_MEMORY_API_KEY" "${internal_key}"
set_env "HMS_GATEWAY_API_KEYS" "${vendor_key_a},${vendor_key_b}"

cat <<EOF
Updated ${ENV_FILE}

Internal HMS API key:
  ${internal_key}

Vendor API keys:
  ${vendor_key_a}
  ${vendor_key_b}

Keep the internal key private. Hand each vendor one hms_live_* key.
EOF
