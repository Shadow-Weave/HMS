# HMS Vendor SDK

This directory contains a vendor-facing SDK package for using the HMS memory
stack as an external service. It provides three product-facing stages:

1. `retain`: ingest raw multi-turn sessions into the memory bank
2. `recall`: retrieve memory evidence for a downstream question
3. `organize`: serialize recalled rows in their original order

The SDK is intentionally small and stable:

- it uses the HMS HTTP API directly;
- it accepts raw session objects instead of provider-specific schemas;
- it returns normalized Python dataclasses that are easy to log or serialize;
- it provides a single `pipeline()` helper for end-to-end vendor demos.

## Directory Layout

```text
vendor_sdk/
  DESIGN.md
  README.md
  pyproject.toml
  examples/
    smoke_test.py
  src/
    hms_vendor_sdk/
      __init__.py
      client.py
      models.py
  tests/
    test_client.py
```

## Environment

The SDK reads the following environment variables:

```bash
export HMS_BASE_URL="http://127.0.0.1:8888"
export HMS_API_KEY="replace-me"      # optional if the service has no auth
export HMS_BANK_ID="vendor-demo"     # optional default for CLI demos
```

## Install

Install in editable mode:

```bash
cd vendor_sdk
python3 -m pip install -e .
```

You can also run without installing:

```bash
export PYTHONPATH="$PWD/vendor_sdk/src"
```

## One-Command Demo

Run a caller-supplied case file containing `question` and `sessions`:

```bash
hms-vendor run-case \
  --case /path/to/case.json \
  --bank-id vendor-demo \
  --create-bank \
  --reset-bank
```

Call the deployed gateway directly from Python:

```bash
python3 vendor_sdk/examples/call_gateway_pipeline.py \
  --case /path/to/case.json \
  --bank-id vendor-demo
```

If the package is not installed:

```bash
PYTHONPATH=vendor_sdk/src \
python3 -m hms_vendor_sdk.cli run-case \
  --case /path/to/case.json \
  --bank-id vendor-demo \
  --create-bank \
  --reset-bank
```

## Gateway Deployment

For external vendors, expose the gateway rather than the raw HMS API:

```bash
export HMS_INTERNAL_BASE_URL="http://127.0.0.1:18080"
export HMS_INTERNAL_API_KEY="hms_internal_service_key"
export HMS_GATEWAY_API_KEYS="hms_live_vendor_key"
export HMS_GATEWAY_PORT=18081
export HMS_GATEWAY_SCOPE_BANK_IDS=true

bash vendor_sdk/scripts/start_gateway.sh
```

The vendor-facing `base_url` is:

```text
http://SERVER_IP:18081
```

The vendor-facing `api_key` is one of the keys in `HMS_GATEWAY_API_KEYS`.
The gateway scopes each public `bank_id` by API key before calling HMS, so
different vendors do not share an internal bank when they use the same bank name.

## Python Usage

```python
from hms_vendor_sdk import HMSVendorClient, SessionRecord

client = HMSVendorClient.from_env()

sessions = [
    SessionRecord(
        session_id="s1",
        timestamp="2026-01-01T10:00:00Z",
        messages=[
            {"role": "user", "content": "I prefer tea in the afternoon."},
            {"role": "assistant", "content": "Noted."},
        ],
    )
]

result = client.pipeline(
    bank_id="vendor-demo",
    sessions=sessions,
    question="What drink does the user prefer in the afternoon?",
    create_bank=True,
    reset_bank=True,
)

print(result.to_dict())
```

`pipeline()` runs:

```text
create/reset bank -> retain raw sessions -> wait for async retain if needed -> recall question -> format recalled rows
```

## CLI Commands

```bash
hms-vendor health
hms-vendor run-case --case /path/to/case.json --bank-id vendor-demo --create-bank --reset-bank
hms-vendor recall --bank-id vendor-demo --question "What drink does the user prefer?"
```

## What This SDK Covers

- `create_bank()`
- `delete_bank()`
- `health()`
- `version()`
- `retain_memory()`
- `retain_sessions()`
- `recall()`
- `organize()`
- `wait_for_operation()`
- `pipeline()`

## What It Intentionally Does Not Cover

- private deployment controls
- database-administration APIs
- scoring or answer grading
- downstream answer generation

The answer-generation layer is intentionally left to the vendor. Use
`organize()` or `/v1/vendor/pipeline` to supply an ordered evidence packet to
that downstream answer step.
