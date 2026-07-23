# HMS Vendor SDK Design

## Goal

This SDK packages HMS for external model-provider integration.

The external party should be able to answer three questions quickly:

1. How does HMS retain raw sessions into reusable memories?
2. How does HMS recall evidence for a downstream question?
3. Can we swap the retain model or base model while keeping the memory system fixed?

This package focuses on the first two and keeps a clean extension point for the
third stage: evidence organization.

## Product Shape

The package is a thin client over the HMS HTTP API.

It standardizes vendor inputs into two core objects:

- `SessionRecord`: a raw conversation session
- `RecallBundle`: normalized retrieval output
- `PipelineResult`: retain and recall results in one object

This is more suitable for external integration than exposing the complete
OpenAPI-generated model surface directly.

## Supported Workflows

### 1. End-to-End Vendor Demo

```text
raw sessions
  -> retain_sessions()
  -> recall()
  -> normalized pipeline result
```

Use when the model provider does not already have a memory store.

### 2. Retain-Only Validation

```text
raw sessions
  -> retain_sessions()
  -> inspect written documents / memory count
```

Use when the vendor wants to compare extraction quality across different retain
models while keeping the storage layer fixed.

### 3. Recall-Only Validation

```text
existing bank
  -> recall(question)
  -> inspect retrieved evidence
```

Use when the vendor already has populated memory banks and wants to test recall
quality only.

## API Contract

### `create_bank(bank_id, ...)`

Creates or updates a bank profile.

Important fields:

- `reflect_mission`
- `retain_mission`
- `retain_extraction_mode`
- `enable_observations`

### `retain_sessions(bank_id, sessions, ...)`

Transforms raw sessions into HMS retain items.

Per session:

- `session_id` becomes `document_id`
- `timestamp` becomes `timestamp`
- `context` is forwarded
- `metadata` is forwarded and augmented
- `messages` are flattened into a single retain content string

The flattening rule is intentionally simple and deterministic:

```text
[USER]
message text

[ASSISTANT]
message text
```

This keeps the SDK transparent for vendors. They can replace the flattening
strategy later if they want a different retain surface.

### `recall(bank_id, question, ...)`

Calls the standard HMS recall endpoint and normalizes the output into:

- ranked memory rows
- optional entities
- optional chunks
- optional trace

### `pipeline(bank_id, sessions, question, ...)`

Runs:

1. optional bank creation
2. retain sessions
3. recall question
4. waits for async retain operations when needed
5. returns a single bundle for logging, comparison, or visualization

The wait step matters. Without it, async retain can return before extraction,
embedding, and storage are complete, causing recall to look worse than it is.

## Why Use A Dedicated Wrapper

The wrapper accepts product-level session and recall objects, keeps deployment
controls out of client payloads, and gives vendors the memory interface
directly.

## Why Not Expose OpenAPI Models Directly

The generated OpenAPI client is useful internally, but it is noisy for external
consumers:

- too many types;
- too much surface area;
- generated naming conventions are less stable for demos.

This wrapper keeps only the shapes needed for retain and recall testing.

## Output Shape

The normalized `PipelineResult` returns:

- bank id
- retain summary
- recall bundle

This allows vendors to diff outputs across models without parsing raw HTTP
responses.

## External Test Package

A vendor-ready package should include:

- SDK package
- CLI
- one-command demo case
- clear environment variables
- deterministic bank reset option
- async operation polling
- raw response fields for debugging

This directory includes those pieces so providers can exercise the memory
lifecycle through a stable client surface.

## Integration Guidance

For external testing, keep these layers separate:

- retain model
- memory database
- recall configuration
- downstream answer model

That way a provider can test:

1. their own retain model with HMS storage and recall
2. HMS retain with their own answer model
3. their own model with a pre-populated fixed memory bank

## Future Extension

The next natural addition is:

```text
organize(bank_id, question, recall_bundle) -> evidence packet
```

That should remain a separate stage instead of being hidden inside `recall()`.
