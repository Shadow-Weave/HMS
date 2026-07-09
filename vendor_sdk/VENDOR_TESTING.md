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
cd /root/autodl-tmp/hanyh/hmsshadow/vendor_sdk
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
  --case examples/cases/shopping_actions.json \
  --bank-id "$HMS_BANK_ID" \
  --create-bank \
  --reset-bank
```

For a richer demonstration with duplicate reminders, canceled state, and a
recommendation distractor:

```bash
hms-vendor run-case \
  --case examples/cases/store_errands_multi_session.json \
  --bank-id vendor-demo-store-errands \
  --create-bank \
  --reset-bank
```

The output contains:

- `retain_summary`: how many raw sessions were submitted and whether retain was async
- `recall_bundle.results`: retrieved memory facts
- `recall_bundle.chunks`: source snippets when available
- `recall_bundle.trace`: recall trace when enabled
- `evidence_packet`: structured ledger rows, active controls, and answer-ready context

## Testing Modes

### End-to-End Mode

Use this when the provider has no memory database:

```text
raw sessions -> retain -> wait -> recall -> organize
```

Run with:

```bash
hms-vendor run-case --case examples/cases/shopping_actions.json --bank-id demo --create-bank --reset-bank
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
  --question "How many items do I need to pick up or return from a store?"
```

This evaluates retrieval quality without rewriting memories.

## Recommended Evaluation Data

Prepare cases with:

- `case_id`
- `question`
- `question_date`
- `sessions`
- `expected_answer`
- optional `expected_evidence`

Focus on memory-specific cases:

- multi-session aggregation
- event deduplication
- knowledge update and current-state arbitration
- relative-date grounding
- amount or difference questions
- cross-session entity consolidation

## Operational Notes

Use `--reset-bank` for isolated demos. Do not use it on shared banks.

Use `--retain-async` only when the server requires background retain. The SDK
waits for async retain by default before recall.

For side-by-side model testing, keep the case file and bank reset policy fixed.
Only change model/provider configuration on the HMS service side.
