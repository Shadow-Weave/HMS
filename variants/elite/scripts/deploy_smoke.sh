#!/usr/bin/env bash
set -euo pipefail

: "${HMS_SMOKE_BASE_URL:=http://127.0.0.1:18090}"
: "${HMS_SERVER_TOKEN:?HMS_SERVER_TOKEN is required (source the deploy .env first)}"

# Which compose file(s) define the running stack. `deploy:dev` runs base-only but
# COMPOSE_OVERRIDE is a *global* CI variable holding the prod file, so it leaks
# into deploy:dev — meaning this script's restart/logs would otherwise
# interpolate prod-only vars (the `:?` guards) against a dev .env and fail. Treat
# an explicitly-empty COMPOSE_OVERRIDE as "no override"; only fall back to the
# prod default when it is unset.
compose=(docker compose -f "${COMPOSE_BASE:-docker-compose.yml}")
if [ "${COMPOSE_OVERRIDE+x}" = x ]; then
  [ -n "$COMPOSE_OVERRIDE" ] && compose+=(-f "$COMPOSE_OVERRIDE")
else
  compose+=(-f docker-compose.prod.yml)
fi

bank="deploy-smoke-$(date +%s)-$$"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

auth=(-H "Authorization: Bearer $HMS_SERVER_TOKEN")

wait_for_health() {
  for attempt in $(seq 1 30); do
    if curl --fail --silent --show-error "$HMS_SMOKE_BASE_URL/health" > /dev/null; then
      return 0
    fi
    echo ">>> waiting for hms-elite health ($attempt/30)"
    sleep 5
  done
  return 1
}

status_of() {
  curl --silent --output /dev/null --write-out '%{http_code}' "$@"
}

assert_real_llm_call() {
  python3 - "$1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
calls, estimated = payload.get("llm_calls"), payload.get("estimated")
# `llm_calls > 0` alone proves nothing: an offline `rule` provider increments it
# too, and so does a call that raised (llms/base.py:75). `estimated` is False
# only when the provider itself reported token usage (llms/base.py:88-98), so
# it is the one signal that requires a real, successful API round trip.
if not isinstance(calls, int) or calls <= 0:
    raise SystemExit(f"expected llm_calls > 0, got {calls!r}")
if estimated:
    raise SystemExit("llm_calls > 0 but estimated=true: no provider usage was returned, so nothing reached the LLM API")
PY
}

assert_json_contains() {
  python3 - "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] not in json.dumps(payload, ensure_ascii=False):
    raise SystemExit(f"expected smoke marker missing from {sys.argv[1]}")
PY
}

assert_json_results_empty() {
  python3 - "$1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
results = payload.get("results")
if results:
    raise SystemExit(f"clear left {len(results)} memories behind")
PY
}

search() {
  curl --fail --silent --show-error "${auth[@]}" \
    -H "Content-Type: application/json" \
    --data "{\"query\":\"What drink do I prefer for deployment reviews?\",\"user_id\":\"$bank\",\"top_k\":5}" \
    "$HMS_SMOKE_BASE_URL/search" > "$1"
}

wait_for_health

# The app exposes /docs, /redoc and /openapi.json on the FastAPI app itself,
# bypassing the Bearer dependency mounted on the APIRouter. nginx must 404 them.
for path in /docs /redoc /openapi.json; do
  code="$(status_of "$HMS_SMOKE_BASE_URL$path")"
  if [ "$code" != "404" ]; then
    echo "expected 404 for $path (unauthenticated API docs), got $code" >&2
    exit 1
  fi
done

# A missing token means authentication is entirely disabled (deps.py:114), which
# would leave the service world-writable. Pin that it is actually enforced.
code="$(status_of "$HMS_SMOKE_BASE_URL/stats")"
if [ "$code" != "401" ]; then
  echo "expected 401 for an unauthenticated request, got $code — is HMS_SERVER_TOKEN set?" >&2
  exit 1
fi

# Write path: auth + LLM extraction + embedder + SQLite, end to end.
curl --fail --silent --show-error "${auth[@]}" \
  -H "Content-Type: application/json" \
  --data "{\"messages\":[{\"role\":\"user\",\"content\":\"I prefer jasmine tea for deployment reviews.\"}],\"user_id\":\"$bank\"}" \
  "$HMS_SMOKE_BASE_URL/memories" > "$work_dir/add.json"

# Proves the DeepSeek key and the outbound proxy genuinely work — /health does
# not exercise egress, so this is the only signal for a misconfigured proxy.
curl --fail --silent --show-error "${auth[@]}" "$HMS_SMOKE_BASE_URL/usage" > "$work_dir/usage.json"
assert_real_llm_call "$work_dir/usage.json"

search "$work_dir/search.json"
assert_json_contains "$work_dir/search.json" "jasmine tea"

# Restart persistence: /data must hold the database AND its WAL across restarts.
"${compose[@]}" restart hms-elite > /dev/null
wait_for_health
search "$work_dir/search-after-restart.json"
assert_json_contains "$work_dir/search-after-restart.json" "jasmine tea"

# Clean up, and prove the clear actually took effect.
curl --fail --silent --show-error -X DELETE "${auth[@]}" \
  "$HMS_SMOKE_BASE_URL/memories?user_id=$bank" > "$work_dir/clear.json"
curl --fail --silent --show-error "${auth[@]}" \
  "$HMS_SMOKE_BASE_URL/memories?user_id=$bank" > "$work_dir/list-after-clear.json"
assert_json_results_empty "$work_dir/list-after-clear.json"

if "${compose[@]}" logs --no-color --since 10m | grep -E -q 'Traceback|Unhandled exception'; then
  echo "Unexpected application error found in recent container logs." >&2
  exit 1
fi
echo ">>> authenticated add/usage/search, restart persistence, and clear smoke passed"
