# HMS Vendor Gateway

The vendor gateway is the production-facing layer for external testing.

External vendors call the gateway with a gateway API key. The gateway calls the
internal HMS API with the internal HMS key. Model keys and database credentials
never leave the server.

## Environment

```bash
export HMS_INTERNAL_BASE_URL="http://127.0.0.1:18080"
export HMS_INTERNAL_API_KEY="hms_internal_service_key"

export HMS_GATEWAY_API_KEYS="hms_live_vendor_1,hms_live_vendor_2"
export HMS_GATEWAY_HOST="0.0.0.0"
export HMS_GATEWAY_PORT="18081"

export HMS_GATEWAY_RATE_LIMIT_PER_MINUTE="60"
export HMS_GATEWAY_DAILY_QUOTA="1000"
export HMS_GATEWAY_AUDIT_LOG="/root/autodl-tmp/hanyh/hmsshadow/.aaaLOG/vendor_gateway_audit.jsonl"
```

## Run

```bash
cd /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk
python3 -m pip install -e .
hms-vendor-gateway --host 0.0.0.0 --port 18081
```

The external base URL is then:

```text
http://SERVER_IP:18081
```

Use HTTPS and a reverse proxy before sharing it outside a private network.

## Endpoints

- `GET /health`
- `POST /v1/vendor/pipeline`
- `POST /v1/vendor/recall`
- `POST /v1/vendor/organize`

## Generate Keys

Create an internal HMS key:

```bash
python3 - <<'PY'
import secrets
print("hms_internal_" + secrets.token_urlsafe(32))
PY
```

Create a vendor-facing key:

```bash
python3 - <<'PY'
import secrets
print("hms_live_" + secrets.token_urlsafe(32))
PY
```

Use the internal key in the HMS API service as `HMS_API_TENANT_API_KEY`.
Use the vendor-facing key in the gateway as `HMS_GATEWAY_API_KEYS`.

## Vendor Call Example

```bash
curl -sS \
  -H "Authorization: Bearer hms_live_vendor_key" \
  -H "Content-Type: application/json" \
  http://SERVER_IP:18081/v1/vendor/pipeline \
  -d @examples/cases/shopping_actions.json
```

## Security Layer

The gateway implements:

- Bearer API-key authentication
- in-memory per-key rate limiting
- in-memory per-key daily quota
- JSONL audit logging
- log redaction for API keys, bearer tokens, emails, and long text fields

For production multi-vendor use, put Nginx or an API gateway in front for TLS,
persistent quotas, IP allowlists, and central observability.
