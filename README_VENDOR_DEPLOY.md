# HMS Vendor Deployment

This package exposes HMS to external vendors through a narrow gateway:

```text
vendor -> nginx -> hms-vendor-gateway:18081 -> hms-api:18080 -> postgres:5432
```

Only Nginx is published. Do not publish `hms-api:18080` or `postgres:5432`.

## Files

```text
docker-compose.yml
.env.example
README_VENDOR_DEPLOY.md
vendor_sdk/
scripts/generate_keys.sh
scripts/start.sh
scripts/smoke_test.sh
```

## Quick Start

```bash
cp .env.example .env
bash scripts/generate_keys.sh
```

Edit `.env` and set real model keys:

```bash
HMS_API_LLM_API_KEY=...
HMS_API_LLM_BASE_URL=...              # optional, for OpenAI-compatible proxy
HMS_API_RETAIN_LLM_API_KEY=...
HMS_API_RETAIN_LLM_BASE_URL=...       # optional, defaults to HMS_API_LLM_BASE_URL in HMS
HMS_API_EMBEDDINGS_OPENAI_API_KEY=...
HMS_API_EMBEDDINGS_OPENAI_BASE_URL=... # optional, for OpenAI-compatible embeddings
PUBLIC_BASE_URL=https://hms.your-domain.com
```

Start the stack:

```bash
bash scripts/start.sh
```

Verify the public gateway:

```bash
bash scripts/smoke_test.sh
```

By default the smoke test checks:

1. `GET /health`
2. missing-key requests return `401`
3. `POST /v1/vendor/organize` returns `evidence_packet`

To run the full retain + recall + result-formatting path, set:

```bash
HMS_SMOKE_RUN_PIPELINE=1
bash scripts/smoke_test.sh
```

## Vendor Credentials

Vendors receive only:

```text
base_url = PUBLIC_BASE_URL
api_key = one hms_live_* key from HMS_GATEWAY_API_KEYS
```

They must not receive:

```text
HMS_INTERNAL_API_KEY
HMS_API_TENANT_API_KEY
model provider keys
database credentials
```

## Runtime Services

`postgres`

- Uses `pgvector/pgvector:pg16`.
- Stores data under `./volumes/postgres`.
- Not exposed to the host.

`hms-api`

- Installs `core/dataplane`.
- Listens on `18080` inside the Docker network.
- Uses `HMS_API_TENANT_EXTENSION=hms_api.extensions.builtin.tenant:ApiKeyTenantExtension`.
- Authenticates only the gateway through `HMS_API_TENANT_API_KEY`.

`hms-vendor-gateway`

- Installs `vendor_sdk`.
- Listens on `18081` inside the Docker network.
- Authenticates vendors through `HMS_GATEWAY_API_KEYS`.
- Writes audit logs to `./logs/vendor_gateway_audit.jsonl`.
- Scopes every external `bank_id` by gateway key by default, so two vendors using `vendor-demo` do not share the same internal HMS bank.

`nginx`

- Publishes only the gateway.
- Adds request body limit, proxy timeouts, access logs, and a basic IP rate limit.
- Default local config listens on HTTP for smoke tests.
- Use `docker/nginx.tls.example.conf` as the production TLS template.

## Public API

Recommended vendor path:

```text
POST /v1/vendor/pipeline
```

This demonstrates:

```text
retain sessions -> recall question -> format recalled rows
```

Other endpoints:

```text
GET  /health
POST /v1/vendor/recall
POST /v1/vendor/organize
```

`/v1/vendor/organize` is intentionally useful for smoke tests because it does not call the internal HMS API; it organizes a supplied recall response and returns:

```json
{
  "evidence_packet": {
    "ledger_rows": [],
    "controls": [],
    "answer_ready_context": "..."
  }
}
```

## SDK Stability Review

Stable pieces:

- The SDK package is small and has a clear surface: dataclasses, `HMSVendorClient`, CLI, gateway, and local evidence organizer.
- The default vendor workflow is centered on `/v1/vendor/pipeline`, which covers retain, recall, and ordered result formatting.
- Gateway auth separates public `hms_live_*` keys from the internal HMS API key.
- Gateway audit logs redact authorization headers, API keys, bearer tokens, common model keys, and email addresses.
- The gateway now scopes bank IDs by vendor key hash. This prevents different vendors from sharing an internal bank when they use the same external `bank_id`.
- SDK client path parameters are URL-encoded before hitting HMS internal routes.

Current limitations:

- Gateway rate limiting and daily quota are in memory. Restarting the gateway resets counters. This is acceptable for demos and small tests, but production should move quota and rate state to Redis or Postgres.
- `HMS_GATEWAY_API_KEYS` is a static comma-separated list. For many vendors, move key metadata to a database with owner, quota, status, and rotation timestamps.
- Gateway audit logging is file-based JSONL. For production, ship it to object storage or centralized logging and set retention policy.
- Nginx provides coarse IP rate limiting. Vendor-level enforcement still lives in the gateway.
- The local formatter preserves recall order and adds no filtering, scoring, deduplication, or answer instructions.

## Bank Isolation

Vendors can send simple bank IDs:

```json
{"bank_id": "vendor-demo"}
```

The gateway maps that to an internal HMS bank:

```text
vendor_<key_hash>_vendor-demo
```

Responses are rewritten back to the public bank ID. Keep `HMS_GATEWAY_SCOPE_BANK_IDS=true` unless this is a private single-vendor deployment.

Allowed public `bank_id` format:

```text
letters, digits, underscore, dash, dot, colon; max length 96; no slashes or spaces
```

## Production TLS

Default compose publishes HTTP for local verification:

```text
PUBLIC_BASE_URL=http://localhost:8080
PUBLIC_HTTP_PORT=8080
```

For production, terminate TLS at your external load balancer or replace the Nginx config:

```bash
cp docker/nginx.tls.example.conf docker/nginx.conf
mkdir -p certs
# mount fullchain.pem and privkey.pem into /etc/nginx/certs if you extend compose
```

Then set:

```bash
PUBLIC_BASE_URL=https://hms.your-domain.com
```

## Operational Notes

- Never put model keys in SDK files, frontend files, or example HTML.
- Never expose `hms-api:18080` directly to the public internet.
- Never expose Postgres.
- Do not let multiple vendors intentionally share one public `bank_id`.
- Prefer `/v1/vendor/pipeline` for vendor demos; `/recall` alone does not show retain and result-formatting behavior.
- Rotate `hms_live_*` keys per vendor. Do not reuse a key across companies.
- Back up `./volumes/postgres` and `./logs/vendor_gateway_audit.jsonl`.
