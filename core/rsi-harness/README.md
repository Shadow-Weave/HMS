# HMS RSI Harness

This package contains the backend-neutral runtime contracts for the memory-agnostic RSI design in [`docs/memory_agnostic_rsi_harness_design_zh-CN.md`](../../docs/memory_agnostic_rsi_harness_design_zh-CN.md).

For the end-to-end operating model, Worker/Supervisor responsibilities,
prompts, namespace lifecycle, failure retention, and deployment checklist, see
[`docs/rsi_harness_pipeline_guide_zh-CN.md`](../../docs/rsi_harness_pipeline_guide_zh-CN.md).

It is intentionally independent of HMS, databases, embeddings, and LLM SDKs. A memory system supplies an adapter implementing `MemoryBackend`; the package provides:

- immutable worker events and supervisor actions;
- bounded live steering through `SupervisorController`;
- candidate staging, sealing, opaque verifier receipts, and promotion;
- deterministic leakage checks and namespace rules;
- declarative Harness candidates and versioned snapshots;
- a read-only benchmark view that requires immutable snapshot support;
- an isolated benchmark verifier with a read-only PostgreSQL transaction.
- chronological multi-session contracts, replay, LongMemEval-shaped input
  conversion, and a conservative Memory Orchestrator/Candidate Factory.
- a generic SessionEvolutionRunner, opaque development verifier, immutable
  skill library snapshots, and an evaluation snapshot envelope that pins both
  memory and skill versions.
- `FileSkillLibrary` and `JsonSnapshotStore` provide an atomic local pointer
  and immutable JSON snapshots, so an accepted skill cohort remains active
  after the worker process restarts.

## Current HMS boundary

`HMSMemoryAdapter` maps HMS Recall/Retain and the RSI snapshot/candidate endpoints to the P0/P1 contract. A snapshot is materialized as a self-contained `building` copy of explicitly classified `project`/`general` memory units; a baseline can be explicitly frozen, while an evolution child becomes `frozen` only during one successful opaque-receipt promotion. Snapshot search and fetch are read-only and inherit the parent contents. Unknown or episode-scoped memory is excluded by default, so benchmark access remains fail-closed unless memory has an explicit safe namespace.

For server-verifiable promotion, configure the same high-entropy secret in the
HMS API as `HMS_API_RSI_VERIFIER_SECRET` and pass it as
`DevelopmentVerifierProcess(receipt_secret=...)`. The promotion request also
includes `run_id` and the SHA-256 candidate digest. Receipts are hashed,
recorded, and consumed once; a missing secret, mismatched digest, failed
receipt, or replay is rejected.

Codex retention writes `rsi_namespace=episode` by default. Operators can use
the namespace endpoint with an explicit memory ID allowlist to tag durable
project/general memories. Snapshot manifests record source counts, coverage,
and a source checksum; freezing rejects a snapshot if the classified source
changes after capture. The benchmark process uses a dedicated read-only
PostgreSQL URL and `PGOPTIONS=-c default_transaction_read_only=on`. The URL
must point to a database/schema and login dedicated to that evaluation; a
shared read-only role would still enumerate other frozen banks in the same
database.

`HarnessEvolutionSession` also requires an injected receipt attestation
callback before a passed Harness policy candidate can be promoted. This keeps
the declarative Harness path subject to the same verifier boundary as the HMS
candidate path.

## Local validation

```bash
cd core/rsi-harness
pytest -q
ruff check rsi_harness tests
```

Run the dependency-free LongMemEval-shaped development smoke from the
repository root:

```bash
PYTHONPATH=core/rsi-harness python core/rsi-harness/examples/longmem_rsi_smoke.py
```

This validates candidate promotion and skill promotion without using a
benchmark answer as runtime input. It is a lifecycle smoke test, not a
LongMemEval accuracy run.

The deterministic H0/H1 fixture additionally demonstrates improvement after a
receipt-attested memory promotion:

```bash
PYTHONPATH=core/rsi-harness \
  python core/rsi-harness/examples/longmem_h0_h1_smoke.py
```

For a persistent skill registry, construct `FileSkillLibrary(".rsi/skills")`
and pass it to `SkillEvolutionController`. A development pass advances the
`current.json` pointer only after receipt attestation; a failed pass leaves the
previous snapshot active. The benchmark command should receive the resulting
skill snapshot ID through its frozen evaluation manifest rather than opening
the skill registry itself.
