# LongMemEval reproduction

This adapter runs one LongMemEval question per isolated HMS memory bank. Each
question follows four stages:

1. Retain every haystack session.
2. Recall evidence for the question.
3. Generate an answer from the recalled evidence.
4. Judge the generated answer against the reference.

The repository ships only the adapter. It downloads the canonical dataset on
first use and writes all runtime state to ignored local paths.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker, or another PostgreSQL 16 server with pgvector
- Model and embedding endpoints with enough quota for the selected run

Install the evaluation environment from the repository root:

```bash
uv sync --project lab/evaluation --extra test
```

## Start PostgreSQL

The benchmark process runs on the host, so the database must listen on a
host-reachable address. The main Compose database uses the internal hostname
`postgres` and does not publish its port. Start an isolated benchmark database:

```bash
docker run --name hms-longmemeval-db \
  --detach \
  --publish 127.0.0.1:5432:5432 \
  --env POSTGRES_USER=hms \
  --env POSTGRES_PASSWORD=hms_longmemeval_change_me \
  --env POSTGRES_DB=hms \
  --volume hms-longmemeval-postgres:/var/lib/postgresql/data \
  pgvector/pgvector@sha256:1d533553fefe4f12e5d80c7b80622ba0c382abb5758856f52983d8789179f0fb
```

Use `docker start hms-longmemeval-db` after the first run.

## Configure model roles

Create an ignored benchmark environment file:

```bash
cp lab/evaluation/benchmarks/longmemeval/longmemeval.env.example .env.longmemeval
chmod 600 .env.longmemeval
```

Replace every `*_change_me` value. Keep separate settings for Retain, the core
memory model, Answer, Judge, and embeddings. A score is comparable only when
these roles use the same providers, model identifiers, endpoint behavior, and
prompt configuration.

The sample uses `gpt-5-mini` for all language-model roles and
`text-embedding-3-small` for embeddings. These are examples, not a bundled
score claim.

### Retain chunking

Retain uses semantic boundary planning by default for long JSON conversations.
The planner asks the Retain model for topic boundaries and then materializes
each chunk exclusively from the original complete exchanges. It does not
ask the model to rewrite source values; materialization canonicalizes the JSON
representation while preserving every original turn value. Short content and
trusted pre-chunked input bypass semantic planning. Non-conversation content
uses the deterministic structural chunker, while boundary-planning failures
follow the configured failure policy (`fixed_fallback` by default).

Provider Batch extraction does not yet bind its checkpoint to a semantic plan
digest. Set `HMS_API_RETAIN_SEMANTIC_CHUNKING_ENABLED=false` when Batch
extraction is enabled.

For banks created entirely by the current run, the result `run_manifest`
records the enabled flag, failure policy, boundary-call limits, fixed chunk
size, and versioned semantic policy and prompt identifiers. Resume therefore
rejects results produced with different Retain chunking semantics. Reused banks
instead mark their creator policy as unverifiable; mixed ingest-only runs keep
the current-run policy separately without attributing it to reused banks.

## Dataset pin

The default run downloads this immutable artifact:

- Dataset: `xiaowu0162/longmemeval-cleaned`
- File: `longmemeval_s_cleaned.json`
- Revision: `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- Size: `277383467` bytes
- SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Source: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>

The verified file is stored at
`.aaaDATA/longmemeval/longmemeval_s_cleaned.json`. The downloader writes a
temporary file, verifies its byte size and checksum, then installs it with an
atomic rename. The repository does not redistribute the dataset. Review the
upstream dataset card and terms before use.

## Run a smoke test

Start with one question and one request at a time:

```bash
HMS_ENV_FILE=.env.longmemeval \
HMS_MAX_INSTANCES=1 \
HMS_RESULTS_FILENAME=longmemeval-smoke.json \
bash .aaaSCRIPT/run_benchmark.sh
```

The command downloads and verifies the dataset, migrates the database, runs all
four stages, audits durable source chunks, and writes:

- Results: `.aaaRESULT/longmemeval-smoke.json`
- Summary: `.aaaRESULT/longmemeval-smoke.md`
- Console log: `.aaaLOG/longmemeval_<timestamp>.log`
- Retain state: PostgreSQL banks named `longmemeval_<question_id>`

Fresh runs clear and rebuild every selected bank before evaluation. They also
refuse to overwrite an existing result filename. Delete an obsolete local
artifact intentionally or choose a new filename; use `--resume` only for a
compatible interrupted run.

Retain state is not a standalone output file. Source documents live in
`documents`, source chunks live in `chunks`, and extracted facts live in
`memory_units`. The run fails when an expected document is missing or has no
durable chunk. A document with zero extracted facts remains valid because its
source chunk is still available to recall.

## Run all 500 questions

Choose positive concurrency values that stay within the provider and database
limits. Four parallel items provide a conservative starting point. The question
and Judge limits are shared across the entire run, not multiplied per item:

```bash
HMS_ENV_FILE=.env.longmemeval \
HMS_RESULTS_FILENAME=longmemeval-full.json \
HMS_PARALLEL=4 \
HMS_MAX_CONCURRENT_QUESTIONS=4 \
HMS_EVAL_SEMAPHORE_SIZE=4 \
bash .aaaSCRIPT/run_benchmark.sh
```

A complete artifact must report:

- `num_items: 500`
- `total_questions: 500`
- `total_invalid: 0`
- the pinned dataset revision and checksum in `run_manifest`

Provider failures count as incorrect in the primary accuracy. The artifact also
records valid-only accuracy as a diagnostic. A full run exits with an error
when it has missing questions or invalid judgments.

## Resume an interrupted run

Results are saved after each completed question with an atomic file replace.
Every checkpoint includes the dataset, pipeline, model-role, endpoint, and code
identity needed for compatibility checks. Resume with the same filename:

```bash
HMS_ENV_FILE=.env.longmemeval \
HMS_RESULTS_FILENAME=longmemeval-full.json \
HMS_PARALLEL=4 \
bash .aaaSCRIPT/run_benchmark.sh --resume
```

The runner skips valid completed question IDs. Invalid questions and items that
did not finish run again from Retain, including clearing a partially retained
bank. Resume requires an existing result file and never selects a new
timestamped artifact.

Resume refuses to mix incompatible dataset hashes, pipeline settings, model
roles, endpoint fingerprints, database modes, Git revisions, or relevant dirty
source trees. Keep those settings unchanged for an interrupted run. Use a new
result filename when changing the experiment.

## Reuse retained banks

Use retrieval-only mode only after a successful Retain pass with the same
dataset, embedding fingerprint, database, and bank IDs:

```bash
HMS_ENV_FILE=.env.longmemeval \
HMS_RESULTS_FILENAME=longmemeval-recall-only.json \
HMS_RETRIEVAL_ONLY=1 \
bash .aaaSCRIPT/run_benchmark.sh
```

Before any question is evaluated, the runner matches every selected bank to the
current dataset item. It verifies the complete document-ID set, normalized
content hashes, retained `context` and `event_date`, durable chunks, and that no
Retain write is still in flight. Missing, stale, empty, or unexpected documents
fail retrieval-only mode instead of silently reusing the wrong state. Documents
with zero extracted facts remain valid when their source chunks are durable.
The strict embedding fingerprint policy separately rejects memories built in
another vector space.

Durable rows do not record enough information to reconstruct the Retain
pipeline, model, or source revision that originally created an older bank.
Retrieval-only artifacts therefore omit `retain` from
`run_manifest.pipeline.stages`, mark the reused-bank Retain provenance as
`unverifiable`, and do not present the current Retain model configuration or
chunking policy as the bank creator. The recorded Git/source identity applies
only to stages executed by the current benchmark process. Fresh runs record
`current_run` Retain provenance. Because a non-forced ingest-only run can skip
exact existing banks and ingest only the missing or stale subset, its global
Retain provenance is marked `mixed_or_reused_bank` and unverifiable; the
current-run chunking policy applies only to newly ingested banks. Use
`--force-reingest` when the artifact must attest that every selected bank was
created by the current run.

## Reproduction profiles

`HMS_PIPELINE=ledger` enables a category-aware retrieval plan and the structured
evidence ledger. The plan reads LongMemEval `question_type` metadata. This is a
benchmark-conditioned profile, not label-free production recall.

Use `HMS_PIPELINE=standard` to disable category-conditioned planning. The
`structured_source` context format remains independent and renders bounded
source bundles with retained source windows.

`HMS_PIPELINE=self_evolution` is an optional diagnosis-derived experiment. Do
not compare it as an untuned main result unless the evaluation protocol
explicitly allows tuning from prior failed cases.

## Local validation

The focused test suite does not call external model providers:

```bash
uv run --project lab/evaluation --extra test python -m pytest \
  lab/evaluation/benchmarks/common/test_benchmark_runner.py \
  lab/evaluation/benchmarks/longmemeval/test_evidence_bundles.py \
  lab/evaluation/benchmarks/longmemeval/test_source_backfill.py \
  lab/evaluation/benchmarks/longmemeval/test_source_context_integration.py \
  -q
```

Check the command line and launcher without starting a live run:

```bash
bash -n .aaaSCRIPT/run_benchmark.sh
uv run --project lab/evaluation \
  python -m benchmarks.longmemeval.longmemeval_benchmark --help
```

## Privacy and cost

Remote Retain and embedding endpoints receive the conversation haystacks.
Answer and Judge endpoints receive benchmark questions and generated content.
Use endpoints that satisfy your privacy requirements. A 500-question run can
consume substantial tokens and may take hours. Run the smoke test before
committing that cost.
