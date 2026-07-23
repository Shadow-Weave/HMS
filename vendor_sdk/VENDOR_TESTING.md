# Vendor Testing Guide

This guide describes how a model provider should test HMS with retain, recall,
and evidence organization enabled.

## Setup

Start the HMS service, then configure:

```bash
export HMS_BASE_URL="http://127.0.0.1:8888"
export HMS_API_KEY="replace-me"
export HMS_BANK_ID="vendor-demo"
```

Install the SDK:

```bash
cd vendor_sdk
python3 -m pip install -e .
```

Check service connectivity:

```bash
hms-vendor health
```

If the provider is using the production gateway instead of the raw HMS API,
configure:

```bash
export HMS_BASE_URL="https://your-gateway-domain"
export HMS_API_KEY="hms_live_vendor_key"
```

## First Demo

```bash
hms-vendor run-case \
  --case /path/to/case.json \
  --bank-id "$HMS_BANK_ID" \
  --create-bank \
  --reset-bank
```

The output contains:

- `retain_summary`: how many raw sessions were submitted and whether retain was async
- `recall_bundle.results`: retrieved memory facts
- `recall_bundle.chunks`: source snippets when available
- `recall_bundle.trace`: recall trace when enabled
- `evidence_packet`: recalled rows in their original order and plain formatted context

## Testing Modes

### End-to-End Mode

Use this when the provider has no memory database:

```text
raw sessions -> retain -> wait -> recall -> organize
```

Run with:

```bash
hms-vendor run-case --case /path/to/case.json --bank-id demo --create-bank --reset-bank
```

### Retain-Only Mode

Use the Python client:

```python
summary = client.retain_sessions(bank_id="demo", sessions=sessions)
print(summary.to_dict())
```

This evaluates extraction and storage behavior.

### Recall-Only Mode

Use this when the memory bank is already populated:

```bash
hms-vendor recall \
  --bank-id demo \
  --question "What preference did the user share?"
```

This evaluates retrieval quality without rewriting memories.

## Case File Format

Prepare cases with:

- `case_id`
- `question`
- `question_date`
- `sessions`
- optional `bank_profile`

## Operational Notes

Use `--reset-bank` for isolated demos. Do not use it on shared banks.

Use `--retain-async` only when the server requires background retain. The SDK
waits for async retain by default before recall.
