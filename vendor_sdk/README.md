# HMS Vendor SDK

This directory contains a vendor-facing SDK package for testing the HMS memory
stack as an external service.

The goal is not to expose the full internal benchmark pipeline. The goal is to
give model providers a clean way to evaluate the two required stages:

1. `retain`: ingest raw multi-turn sessions into the memory bank
2. `recall`: retrieve memory evidence for a downstream question
3. `organize`: convert recalled evidence into a structured answer-time ledger

The SDK is intentionally small and stable:

- it uses the HMS HTTP API directly;
- it accepts raw session objects instead of benchmark-only schemas;
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
cd /root/autodl-tmp/hanyh/hms_shadow/vendor_sdk
python3 -m pip install -e .
```

You can also run without installing:

```bash
export PYTHONPATH=/root/autodl-tmp/hanyh/hms_shadow/vendor_sdk/src
```

## One-Command Demo

Run the built-in shopping-action case:

```bash
hms-vendor run-case \
  --case /root/autodl-tmp/hanyh/hms_shadow/vendor_sdk/examples/cases/shopping_actions.json \
  --bank-id vendor-demo-shopping \
  --create-bank \
  --reset-bank
```

Run the richer multi-session case:

```bash
hms-vendor run-case \
  --case /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk/examples/cases/store_errands_multi_session.json \
  --bank-id vendor-demo-store-errands \
  --create-bank \
  --reset-bank
```

Call the deployed gateway directly from Python:

```bash
python3 /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk/examples/call_gateway_pipeline.py \
  --case /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk/examples/cases/store_errands_multi_session.json \
  --bank-id vendor-demo-store-errands \
  --output /tmp/hms_store_errands_result.json
```

Open the static visual replay:

```text
/root/autodl-tmp/hanyh/hmsshadow/vendor_sdk/examples/outputs/store_errands_visual.html
```

If the package is not installed:

```bash
PYTHONPATH=/root/autodl-tmp/hanyh/hms_shadow/vendor_sdk/src \
python3 -m hms_vendor_sdk.cli run-case \
  --case /root/autodl-tmp/hanyh/hms_shadow/vendor_sdk/examples/cases/shopping_actions.json \
  --bank-id vendor-demo-shopping \
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

bash /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk/scripts/start_gateway.sh
```

The vendor-facing `base_url` is:

```text
http://SERVER_IP:18081
```

The vendor-facing `api_key` is one of the keys in `HMS_GATEWAY_API_KEYS`.

## Python Usage

```python
from hms_vendor_sdk import HMSVendorClient, SessionRecord

client = HMSVendorClient.from_env()

sessions = [
    SessionRecord(
        session_id="s1",
        timestamp="2024-03-01T10:00:00Z",
        messages=[
            {"role": "user", "content": "I need to pick up my blazer from dry cleaning."},
            {"role": "assistant", "content": "Noted."},
        ],
    )
]

result = client.pipeline(
    bank_id="vendor-demo",
    sessions=sessions,
    question="How many items do I need to pick up or return from a store?",
    question_date="2024-03-10T00:00:00Z",
    create_bank=True,
    reset_bank=True,
)

print(result.to_dict())
```

`pipeline()` runs:

```text
create/reset bank -> retain raw sessions -> wait for async retain if needed -> recall question -> organize evidence ledger
```

## CLI Commands

```bash
hms-vendor health
hms-vendor run-case --case examples/cases/shopping_actions.json --bank-id vendor-demo --create-bank --reset-bank
hms-vendor recall --bank-id vendor-demo --question "What do I need to pick up?"
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

- benchmark-only planner flags
- internal oracle variants
- result fabrication or score reporting
- downstream answer generation

If you need the next layer after recall, extend this package with an
`organize()` stage and keep the input/output contracts unchanged.
