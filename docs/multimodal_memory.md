# Multimodal image and video memory

This guide describes the engineering-preview behavior and operator requirements
for converting an uploaded image or video into an ordinary HMS memory document.
The upload endpoint remains the existing public contract. The additive
multimodal capability negotiation and typed operation-status namespace have
been published in OpenAPI without changing the default legacy wire shape.

Source and upstream-documentation review date: 2026-07-23 (UTC).

> Status: opt-in engineering preview. The feature is disabled by default, and
> `HMS_API_MULTIMODAL_LIVE_VERIFIED` is false by default. Clients that opt in to
> the additive `/version` capability fields can read the corresponding
> `multimodal_live_verified` marker, but it is not an online provider probe.
> Deterministic fake-provider tests can qualify the transport and retain/recall
> plumbing, but they do not measure real `gpt-5-mini` visual accuracy. This
> repository does not include a live-provider qualification result.

## What the integration does

Clients keep using the existing multipart file endpoint:

```text
POST /v1/default/banks/{bank_id}/files/retain
```

The client uploads the original file bytes and explicitly selects the
`openai_multimodal` parser. The public API does not accept a base64 media field.

```text
multipart upload
  -> bounded read, SHA-256, and existing FileStorage
  -> local media validation
       image: EXIF/color/size normalization
       video: PyAV decode and deterministic scene + timeline sampling
  -> OpenAI Responses API using gpt-5-mini image inputs
  -> strict grounded schema and system-owned provenance
  -> deterministic canonical Markdown
  -> existing HMS chunks retain pipeline
  -> existing documents, embeddings, tags, links, and recall
```

This is an input-normalization layer, not a second memory system. The visual
description enters the same HMS storage and recall path as other retained text.
Consequently, existing bank, tenant, tag, document replacement, and recall
semantics still apply.

### Image path

HMS detects the image from magic bytes, cross-checks a specific declared MIME
type and known filename extension, applies EXIF orientation, bounds decoded
pixels, removes source metadata, and deterministically encodes a normalized
JPEG or PNG. The current normalizer caps the longest dimension at 2048 pixels;
transparent input remains PNG and opaque input becomes JPEG. Only that
normalized image is converted to a data URL, and only while constructing the
provider request.

Supported still-image inputs are PNG, JPEG, WebP, and single-frame GIF.
Animated GIF is rejected with `media.animated_image_unsupported`; HMS does not
silently describe only its first frame.

### Video path

`gpt-5-mini` receives images, not the raw video. HMS performs all video
probing and decoding locally with the optional PyAV dependency. It samples at
most the configured frame budget using a deterministic combination of visual
scene change and time coverage. Selected frames retain system timestamps.

The selected frames are split into chronological batches. Each batch is
described as a grounded segment, then a text-only reducer may merge and order
the already validated segments. The reducer must preserve the mapped segments
and evidence IDs; it cannot add an unsupported fact or invent a time range.
HMS renders one time-coded Markdown document for the whole asset.

For a configured selected-frame budget `B`, the sampler reserves

```text
K = min(B - 1, max(3, floor(B * coverage_ratio)))
```

coverage slots across equal timeline strata, leaving at least one slot for
scene novelty. Remaining slots use a fixed visual-change score plus temporal
diversity; ties are resolved by timestamp and frame hash. The same bytes and
configuration therefore produce the same ordered timestamp/hash selection,
and a short video is never padded by duplicating a frame. The motivation is to
retain start/middle/end coverage without losing a brief terminal error or small
IDE state change that a pure fixed-interval or pure keyframe policy can miss.

The current engineering baseline exercises MP4/H.264. ISO-BMFF, Matroska/WebM,
and AVI container signatures are recognized, but actual codec support depends
on the PyAV/FFmpeg libraries in the deployed environment. The public
`multimodal_video` flag reports the configured path and local decoder gate; it
does not certify every container/codec combination.

Audio tracks are detected only as provenance. They are not transcribed or sent
to the descriptor provider. A video with an audio track is still described as
visual-only. Recall metadata keeps `media_audio_presence` separate from
`media_audio_processing`, so "an audio stream exists" is never confused with
"audio was transcribed"; the current pipeline records processing as
`not_requested`.

### Provider wire boundary

The equivalent redacted Responses request uses text plus one or more image
content parts and strict Structured Outputs:

```json
{
  "model": "gpt-5-mini",
  "input": [{
    "role": "user",
    "content": [
      {"type": "input_text", "text": "<versioned grounded-description prompt>"},
      {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,<ephemeral-redacted-payload>",
        "detail": "auto"
      }
    ]
  }],
  "text": {
    "format": {
      "type": "json_schema",
      "name": "hms_multimodal_description",
      "schema": {"<strict schema>": "..."},
      "strict": true
    }
  },
  "store": false
}
```

The MIME prefix is derived from the normalized bytes; clients do not provide
this data URL. The complete request body must not be logged or persisted. Video
map requests repeat `input_image` for the selected frames, while the reducer is
text-only and receives validated segment data rather than the images again.

### Lazy dependency boundary

The public `hms_api.engine.multimodal` surface and the package-level
`openai_multimodal` parser exports use lazy imports. On the ordinary text and
legacy-file startup path, this integration therefore does not import Pillow,
its `httpx` provider transport, or PyAV and the linked FFmpeg libraries. Those
modules are loaded only when HMS resolves and constructs the multimodal parser.
Other existing HMS components may independently use `httpx`; the compatibility
guarantee here is that enabling this code in the source tree does not add the
multimodal transport or media stack to a text-only process.

This boundary has both compatibility and performance motivations. A deployment
without the optional PyAV extra can continue to start and serve text memory,
and text-only API/worker processes avoid media codec initialization, import
latency, and the corresponding resident-memory cost. Once multimodal video is
configured, the API and every conversion worker still need the same verified
PyAV/FFmpeg runtime described below.

## Deployment prerequisites

The API, background worker, database, and existing file-upload API must be
available.

### Database qualification matrix

This engineering preview freezes its multimodal/file-retain database matrix as
follows. The matrix is scoped to this media-ingestion path; it does not remove
or redefine Oracle support for ordinary HMS text-memory features.

| Database backend | Qualification for this multimodal/file-retain preview |
| --- | --- |
| PostgreSQL (qualification run: 14.22 with pgvector) | Offline runtime-qualified for migration round trips, durable descriptor/segment checkpoints, command ordering and concurrency, upload, child retain, and recall. Live-provider visual quality remains a separate, unqualified gate. |
| Oracle 23ai | Static compatibility only: the dual-dialect migration, downgrade shape, CLOB/NUMBER mappings, and Oracle ledger SQL branches have static/unit coverage. No Oracle multimodal migration or file-retain runtime qualification has been run. |

HMS therefore must not advertise or describe Oracle multimodal runtime support
for this release. A server configured with both
`HMS_API_DATABASE_BACKEND=oracle` and `HMS_API_MULTIMODAL_ENABLED=true` fails
fast instead of attempting an unqualified media path. This restriction is
specific to multimodal file retain and is not a statement that HMS as a whole
lacks an Oracle backend.

Expanding this matrix to Oracle requires a real Oracle 23ai gate covering an
`upgrade -> downgrade -> upgrade` migration round trip; descriptor and video
segment claim/lease/CAS behavior; concurrent document admission, sequencing,
and publication; a minimum multipart upload -> parser -> child retain -> recall
flow; source retention/deletion; tenant/schema/bank isolation; and compatible
backup and delete cleanup. Static SQL-shape tests cannot substitute for those
runtime checks.

### Media runtime dependencies

Image support uses Pillow, which is a core dataplane dependency. Video support
additionally requires the project extra:

```bash
cd core/dataplane
uv sync --extra multimodal-video
```

For an editable pip installation, the equivalent is:

```bash
python -m pip install -e '.[multimodal-video]'
```

The standalone Dockerfile keeps this native dependency out of its default
image. Build an API image with the explicit opt-in argument:

```bash
docker build \
  --target api-only \
  --build-arg INCLUDE_LOCAL_MODELS=false \
  --build-arg INCLUDE_MULTIMODAL_VIDEO=true \
  -f deploy/containers/standalone/Dockerfile \
  -t hms-api:multimodal-video \
  .
```

The build verifies that PyAV imports and can construct an H.264 decoder in both
the builder and final slim runtime. An additional artifact check can be run
without starting HMS:

```bash
docker run --rm \
  --entrypoint /app/api/.venv/bin/python \
  hms-api:multimodal-video \
  -c 'import av; print(av.__version__, av.codec.CodecContext.create("h264", "r").codec.name)'
```

Until an image is built with this argument, keep
`HMS_API_MULTIMODAL_VIDEO_ENABLED=false`. Install the same dependency in the
API and every conversion worker. In Kubernetes, point both components to the
same verified, non-moving image tag. Plain `GET /version` intentionally keeps
the legacy wire shape; `GET /version?include_multimodal=true` publishes the
approved additive capability flags. The decoder check is local to the API
process and cannot prove that a separately deployed worker uses an identical
image.

### Minimal official OpenAI configuration

The descriptor role is deliberately separate from the text retain LLM role.
Do not reuse a credential merely because one is already configured for text
unless the same data-egress authorization applies to media.

This is also separate from any model configured in a developer's coding tool
or IDE. Only the `HMS_API_MULTIMODAL_*` runtime settings below affect media
description.

```dotenv
HMS_API_ENABLE_FILE_UPLOAD_API=true
HMS_API_FILE_PARSER=markitdown
# Optional allowlist. If set, include openai_multimodal for explicit requests.
HMS_API_FILE_PARSER_ALLOWLIST=markitdown,openai_multimodal
HMS_API_FILE_DELETE_AFTER_RETAIN=true

HMS_API_MULTIMODAL_ENABLED=true
HMS_API_MULTIMODAL_PROVIDER=openai
HMS_API_MULTIMODAL_MODEL=gpt-5-mini
HMS_API_MULTIMODAL_API_KEY=replace_with_secret
HMS_API_MULTIMODAL_BASE_URL=https://api.openai.com/v1
HMS_API_MULTIMODAL_IMAGE_ENABLED=true
HMS_API_MULTIMODAL_VIDEO_ENABLED=false
HMS_API_MULTIMODAL_LIVE_VERIFIED=false
```

Restart the API and every worker after changing these server-static settings,
then verify the configured provider declarations and decoder in both runtime
images. Do not run API and workers with different media budgets or version
identifiers. Plain `GET /version` intentionally retains its legacy shape. Use
`GET /version?include_multimodal=true` to negotiate the approved additive
capability fields.

Enable video only after installing and testing the decoder:

```dotenv
HMS_API_MULTIMODAL_VIDEO_ENABLED=true
```

Do not add `openai_multimodal` to `HMS_API_FILE_PARSER` merely to make it the
implicit default. Explicit parser selection is the safe rollout path: an
ordinary legacy image upload must not start egressing media or incurring model
cost after a server upgrade.

### Custom OpenAI-compatible endpoint

A custom `HMS_API_MULTIMODAL_BASE_URL` is accepted only when the operator
explicitly declares all of these capabilities:

```dotenv
HMS_API_MULTIMODAL_CAPABILITY_RESPONSES_API=true
HMS_API_MULTIMODAL_CAPABILITY_IMAGE_INPUT=true
HMS_API_MULTIMODAL_CAPABILITY_STRUCTURED_OUTPUTS=true
```

These are static operator attestations, not remote probes. Verify the endpoint's
wire compatibility, model mapping, billing, logging, retention, and structured
output behavior before setting them. HMS does not make a potentially billable
startup probe and does not silently fall back to another model.

## Configuration reference

All multimodal settings are server-static. They cannot be weakened through a
bank configuration or upload request.

### Capability and provider

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `HMS_API_MULTIMODAL_ENABLED` | `false` | Deployment-wide egress and registration gate. A key is required when true. |
| `HMS_API_MULTIMODAL_PROVIDER` | `openai` | Only `openai` is currently implemented. |
| `HMS_API_MULTIMODAL_MODEL` | `gpt-5-mini` | Exact configured request string. The default is a rolling alias. |
| `HMS_API_MULTIMODAL_MODEL_BEHAVIOR_VERSION` | `gpt-5-mini-alias-v1` | Operator-controlled semantic version used in the pipeline fingerprint. Bump when an alias behavior change should trigger reprocessing. |
| `HMS_API_MULTIMODAL_API_KEY` | unset | Dedicated provider secret; required when the feature is enabled. |
| `HMS_API_MULTIMODAL_BASE_URL` | `https://api.openai.com/v1` | Responses-compatible provider base URL. |
| `HMS_API_MULTIMODAL_IMAGE_ENABLED` | `true` | Image sub-capability; must be true when the global feature is enabled. |
| `HMS_API_MULTIMODAL_VIDEO_ENABLED` | `false` | Video intent. Effective capability also requires image transport and local PyAV. |
| `HMS_API_MULTIMODAL_LIVE_VERIFIED` | `false` | Operator attestation that the configured live gate passed. It is not set automatically. |
| `HMS_API_MULTIMODAL_CAPABILITY_RESPONSES_API` | true only for the official URL | Static provider-contract declaration. |
| `HMS_API_MULTIMODAL_CAPABILITY_IMAGE_INPUT` | true only for the official URL | Static image-input declaration. |
| `HMS_API_MULTIMODAL_CAPABILITY_STRUCTURED_OUTPUTS` | true only for the official URL | Static strict-structured-output declaration. |
| `HMS_API_MULTIMODAL_IMAGE_DETAIL` | `auto` | One of `auto`, `low`, or `high` for this integration. |

### Media and provider budgets

| Environment variable | Default | Validation / effect |
| --- | ---: | --- |
| `HMS_API_MULTIMODAL_MAX_IMAGE_BYTES` | `20971520` | Maximum raw bytes for one image. |
| `HMS_API_MULTIMODAL_MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded image pixels. |
| `HMS_API_MULTIMODAL_MAX_VIDEO_BYTES` | `104857600` | Maximum raw bytes for one video. |
| `HMS_API_MULTIMODAL_MAX_VIDEO_DURATION_SECONDS` | `600` | Maximum decoded video duration. |
| `HMS_API_MULTIMODAL_VIDEO_PROBE_INTERVAL_SECONDS` | `1` | Candidate-frame interval used by the sampler. Must be positive. |
| `HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES` | `24` | Selected-frame budget `B`; must be at least 4. |
| `HMS_API_MULTIMODAL_VIDEO_COVERAGE_RATIO` | `0.6` | Coverage share; must be strictly between 0 and 1. |
| `HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL` | `8` | Image inputs per video map call; cannot exceed `VIDEO_MAX_FRAMES`. |
| `HMS_API_MULTIMODAL_MAX_OUTPUT_TOKENS` | `4096` | Output-token cap for each physical provider attempt. |
| `HMS_API_MULTIMODAL_REQUEST_TIMEOUT_SECONDS` | `60` | Timeout per HTTP attempt. |
| `HMS_API_MULTIMODAL_MAX_RETRIES` | `2` | Additional transport attempts for network errors, 429, and selected 5xx. |
| `HMS_API_MULTIMODAL_MAX_SCHEMA_REPAIRS` | `1` | Additional structured request after schema-invalid output; only 0 or 1. |
| `HMS_API_MULTIMODAL_MAX_CONCURRENCY` | `4` | Provider semaphore across concurrent calls. It is not a per-asset map parallelism promise. |
| `HMS_API_MULTIMODAL_DESCRIPTOR_CACHE_TTL_SECONDS` | `604800` | Positive TTL for sanitized descriptor checkpoints (7 days by default). |

For retry index `i`, transport retries use full jitter sampled from
`[0, initial_backoff_seconds * 2^i]` (`initial_backoff_seconds` is `0.5` in the
built-in provider configuration). The former deterministic exponential delay
remains the ceiling, so jitter disperses synchronized retry waves without
increasing the previous worst-case wait envelope. Tests inject a deterministic
sampler; runtime requests use independent random samples.

The file endpoint also enforces `HMS_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB`
(default 100 MB across the request) and
`HMS_API_FILE_CONVERSION_MAX_BATCH_SIZE` (default 10 files). The strictest
applicable batch, image, or video byte limit wins.

### Version fields

| Environment variable | Default |
| --- | --- |
| `HMS_API_MULTIMODAL_PROMPT_VERSION` | `openai-mm-v2` |
| `HMS_API_MULTIMODAL_SCHEMA_VERSION` | `hms-multimodal-v1` |
| `HMS_API_MULTIMODAL_SAMPLING_VERSION` | `scene-coverage-v1` |

These non-empty identifiers participate in provenance and cache identity. Bump
the relevant identifier when changing behavior; do not change a prompt or
sampling algorithm while retaining the old version label.

### Pipeline fingerprint contents

`media_pipeline_fingerprint` is a SHA-256 digest over a canonical, non-secret
identity for the descriptor pipeline. In addition to the explicit prompt,
schema, sampling, model-behavior, normalization, and frame-budget settings, the
current identity includes:

- the provider, configured model, image detail, and a SHA-256 fingerprint of
  the trailing-slash-normalized provider endpoint;
- the Pillow version and the image codec versions Pillow reports for JPEG,
  JPEG 2000, WebP, and zlib when available;
- for an enabled video path, the PyAV version and the linked FFmpeg library
  versions reported by PyAV; and
- the maximum transport retries, schema-repair attempts, output tokens per
  physical provider attempt, and retry-backoff algorithm identity.

These inputs can change normalized bytes, selected frames, provider behavior,
or the retry/output envelope, so changing one invalidates descriptor-cache
reuse instead of silently treating a different runtime as the same pipeline.
The identity contains neither the API key nor the raw endpoint URL: only the
normalized endpoint's hash enters the provider identity and final digest. It
also never contains media bytes, a data URL, prompt output, or customer text.

## Capability negotiation contract

The approved public contract preserves strict older clients by default:

- `GET /version` returns the exact legacy feature shape;
- `GET /version?include_multimodal=true` additionally returns
  `multimodal_image`, `multimodal_video`, and
  `multimodal_live_verified`; and
- all three fields are optional with `default=false` in OpenAPI and generated
  SDKs, so a new client can read an older server that omits them.

The server derives these flags conservatively without contacting the provider,
opening customer media, or exposing a credential. `multimodal_image=true`
requires the PostgreSQL-qualified database backend, the file-upload API, global
multimodal and image settings, the Responses/image-input/structured-output
provider declarations, and an unset parser allowlist or one that includes
`openai_multimodal`. `multimodal_video=true` additionally requires the video
setting and a locally available PyAV/FFmpeg decoder. Operators must still ensure
that every conversion worker uses the same decoder-capable image; the API-local
check is not a fleet health probe and does not promise support for every codec.

`multimodal_live_verified=true` is returned only when the effective configured
runtime above is available and the operator has set the live-verification
marker. If video is configured but its decoder gate fails, the live marker also
fails closed. The marker records qualification of that exact configuration; it
does not perform a live OpenAI request.

## Upload an image or video

Set reusable shell variables first:

```bash
export HMS_BASE_URL=http://127.0.0.1:8888
export HMS_API_KEY=replace_with_hms_key
export HMS_BANK_ID=media-demo
```

Upload one synthetic IDE screenshot:

```bash
curl -sS -X POST \
  "$HMS_BASE_URL/v1/default/banks/$HMS_BANK_ID/files/retain" \
  -H "Authorization: Bearer $HMS_API_KEY" \
  -F 'files=@./ide-session.png;type=image/png' \
  -F 'request={"parser":"openai_multimodal","files_metadata":[{"document_id":"ide-image-001","context":"IDE implementation session","tags":["ide","image"]}]}'
```

Upload one short coding screencast through the same endpoint:

```bash
curl -sS -X POST \
  "$HMS_BASE_URL/v1/default/banks/$HMS_BANK_ID/files/retain" \
  -H "Authorization: Bearer $HMS_API_KEY" \
  -F 'files=@./ide-session.mp4;type=video/mp4' \
  -F 'request={"parser":"openai_multimodal","files_metadata":[{"document_id":"ide-video-001","context":"IDE test-and-fix session","tags":["ide","video"]}]}'
```

The response contains one file-conversion operation ID per uploaded file:

```json
{"operation_ids":["550e8400-e29b-41d4-a716-446655440000"]}
```

`files_metadata` is positional and, when present, must have exactly one entry
per uploaded file. A per-file `parser` overrides the request-level parser.

Do not set an explicit retain `strategy` for `openai_multimodal`. The canonical
evidence document requires the trusted operation-scoped `chunks` extraction
mode; the server rejects an explicit strategy rather than risking lost
timecodes or evidence locators.

### Anonymous retries and validation identity

Supplying an explicit `document_id` remains the clearest client contract. If it
is omitted for `openai_multimodal`, HMS derives an opaque, domain-separated
`file_mm_...` document ID from the tenant/schema scope, bank, and raw asset
SHA-256. The raw digest and filename are not exposed in that ID. A byte-for-byte
retry inside the same tenant and bank therefore converges on one logical
document command, while the same bytes in another tenant or bank do not merge.
Use distinct explicit IDs when the same media must intentionally appear as
multiple logical documents. Legacy anonymous non-multimodal file retain keeps
its historical random-ID behavior.

The document-command identity also includes only upload hints that can change
media validation: normalized declared MIME and the recognized final-extension
family. Missing MIME and `application/octet-stream` converge; equivalent
aliases accepted by one validator family converge; arbitrary filename text is
excluded. Correcting a wrong MIME or relevant extension creates a new command
and re-runs validation instead of being mistaken for a retry of the failed
command. This separates transport retry identity from a real correction to the
input contract.

### Fallback behavior

An ordered parser list can include `openai_multimodal`, but fallback is narrow.
The next parser may run only when the multimodal parser determines, before
provider processing, that the input is not applicable. Once media processing
has started, timeout, exhausted 429/5xx, authentication, refusal, incomplete
output, schema/grounding, security, and resource errors are terminal for that
operation. HMS does not hide a paid or semantically different failure by
silently switching to OCR.

## Poll both asynchronous stages

File conversion and memory retention are separate operations. A parent
`file_convert_retain` operation becoming `completed` means conversion succeeded
and the child retain was queued. It does not by itself mean recall can see the
document.

Poll the parent ID returned by the upload:

```bash
curl -sS \
  "$HMS_BASE_URL/v1/default/banks/$HMS_BANK_ID/operations/$OPERATION_ID" \
  -H "Authorization: Bearer $HMS_API_KEY"
```

For multimodal operations, the otherwise open `result_metadata` map contains a
stable, typed public `multimodal` namespace:

The values below are illustrative; they are not a recorded live-provider run.

```json
{
  "operation_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "operation_type": "file_convert_retain",
  "result_metadata": {
    "multimodal": {
      "asset_id": "asset_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "asset_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "media_kind": "video",
      "pipeline_version": "hms-multimodal-v1",
      "descriptor_model": "gpt-5-mini",
      "resolved_model": "provider-returned model when available",
      "stage": "retain_queued",
      "child_retain_operation_id": "660e8400-e29b-41d4-a716-446655440000",
      "child_retain_status": "processing",
      "recall_ready": false,
      "retryable": false,
      "input_tokens": 0,
      "output_tokens": 0,
      "logical_calls": 4,
      "physical_attempts": 4
    }
  }
}
```

This namespace is schema-validated, allowlisted, and redacted at the HTTP
boundary, and is represented by generated SDK types. It is optional and appears
only for multimodal operations. Other legacy keys in `result_metadata` remain
operation-specific diagnostics and are not a stable public contract. On every
poll, the GET handler derives
`child_retain_status`, `recall_ready`, and the final stage from both the current
child operation and the durable document-command/head publication state. A
`completed` child alone is insufficient: the exact command must be completed,
point to that child, and have its sequence published by the head. This fails
closed if a final retain callback was skipped or lost ownership. Clients may use
the following typed fields for polling:

- stop successfully only at `recall_ready=true` / `stage=recall_ready`;
- stop as failed when the parent `status=failed`, using only the sanitized
  `sanitized_error_code` and `retryable` fields for automation;
- treat `stage=retain_failed` as downstream retain failure even though media
  conversion previously completed;
- do not request `include_payload=true` in routine polling. It is unnecessary
  and exposes more task metadata to the caller.

## Recall the canonical memory

After `recall_ready=true`, use the ordinary recall endpoint. Requesting chunks
is useful when the caller needs the canonical video time ranges and evidence
locators:

```bash
curl -sS -X POST \
  "$HMS_BASE_URL/v1/default/banks/$HMS_BANK_ID/memories/recall" \
  -H "Authorization: Bearer $HMS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"Which test failed in the IDE session, and what changed afterward?",
    "budget":"mid",
    "max_tokens":4096,
    "include":{"entities":null,"chunks":{}},
    "tags":["ide","video"],
    "tags_match":"all_strict"
  }'
```

The normal result metadata can contain these bounded, flat system fields:

```text
media_asset_sha256
media_kind
media_detected_mime
media_descriptor_provider
media_descriptor_model
media_pipeline_version
media_prompt_version
media_schema_version
media_sampling_version
media_pipeline_fingerprint
media_audio_presence
media_audio_processing
media_source_available
```

Video time ranges and evidence IDs are part of the canonical chunk text rather
than a large frame manifest copied into every fact. User metadata must not rely
on `media_*` keys; server-generated provenance wins on a conflict.

## Cost and latency bounds

HMS does not publish a fixed dollar cost because OpenAI pricing, image token
calculation, the rolling model alias, media dimensions, retries, and generated
output all vary. Calculate dollar estimates from the current provider pricing
after applying the request-count bounds below.

Let:

- `N` be the actual selected frame count;
- `B` be `HMS_API_MULTIMODAL_VIDEO_MAX_FRAMES`;
- `F` be `HMS_API_MULTIMODAL_MAX_FRAMES_PER_CALL`;
- `r` be `HMS_API_MULTIMODAL_MAX_RETRIES`;
- `s` be `HMS_API_MULTIMODAL_MAX_SCHEMA_REPAIRS`;
- `T` be `HMS_API_MULTIMODAL_MAX_OUTPUT_TOKENS`.

For an image, `N=1`, the logical-call count is 1, and no reducer is used. For a
video, `N <= B`, map calls are `M = ceil(N / F)`, and the current implementation
uses one text-only reduce call, so logical calls are `M + 1`.

The conservative physical-attempt and output-token bounds are:

```text
physical attempts <= logical calls * (1 + r) * (1 + s)
output tokens      <= physical attempts * T
```

After image normalization or video sampling fixes the evidence set, and before
the first provider I/O, the parser computes a payload-free budget object with:

```text
physical_attempts_upper_bound
output_tokens_upper_bound
estimated_image_transport_bytes
```

The first two fields are conservative ceilings under the configured retry,
repair, and per-attempt output-token limits. The transport field estimates the
base64 data-URL bytes for the normalized image evidence; it is not total HTTP
traffic and does not include text, JSON framing, headers, TLS, or repeated
transport caused by retries. On a successful conversion these values are
carried in internal parser pipeline metadata and are covered by offline tests.
The current public operation whitelist does not expose them: parent-operation
responses continue to report the actual `logical_calls`, `physical_attempts`,
`input_tokens`, and `output_tokens` observed when available.

None of these pre-I/O values is an invoice, a prediction of normal usage, or a
substitute for provider usage records. In particular, the upper bounds assume
every allowed retry and repair is consumed, while a successful request usually
uses fewer attempts and output tokens.

With the shipped defaults:

- one image has at most 6 physical attempts and 24,576 output tokens in the
  worst transport-retry-plus-schema-repair envelope;
- a fully selected 24-frame video uses at most 3 map calls plus 1 reducer,
  24 physical attempts, and 98,304 output tokens in that same worst case.

These are safety ceilings, not expected usage or measured cost. Most successful
requests should be below them. Input image tokens, text input tokens, and the
roughly 4/3 base64 transport expansion must still be included in provider-side
cost and payload estimates.

A completed descriptor checkpoint for the same tenant/bank, raw asset hash, and
pipeline fingerprint can avoid a new provider invocation until it expires. The
descriptor identity is separate from the logical document command, so a new
context, tag set, timestamp, or document update still follows the normal HMS
publication path. Cache reuse is never cross-bank and should not be subtracted
from the conservative pre-request budget as if it were guaranteed.

Per-asset video map calls currently execute in chronological order and the
reducer follows them, so network latency is not simply one provider round trip.
`MAX_CONCURRENCY` bounds calls across concurrent work; it does not promise that
all segments of one video run in parallel. Retry backoff, decode work, child
retain, embeddings, and recall indexing also contribute to end-to-end time.

No real-provider P50/P95, dollar cost, or visual recall-accuracy number is
available for this checkout. Do not derive one from fake-provider tests.

### Observability

The parent operation reports sanitized actual token and call counts when the
provider returns usage. The metrics collector exposes these low-cardinality
instruments:

```text
hms.multimodal.stage.duration
hms.multimodal.frames.total
hms.multimodal.provider.calls.total
hms.multimodal.tokens.total
hms.multimodal.dedupe.total
hms.multimodal.assets.total
hms.multimodal.schema.failures.total
hms.multimodal.source.lifecycle.total
hms.multimodal.cancellations.total
hms.multimodal.in_flight
```

Provider calls distinguish logical, physical, retry, and terminal failure
attempts. Frame counts distinguish candidate from selected; source lifecycle,
schema failures, cancellation, admission outcome, retain queue, and in-flight
provider work use finite state/reason sets. Tokens are split by input/output
direction and bucketed. Labels intentionally exclude tenant/schema, filename,
document ID, asset hash, bank ID, prompt, OCR text, description, and data URL.
A zero token count may mean the endpoint omitted usage; it does not prove a
free request.

## Source deletion and retry semantics

`HMS_API_FILE_DELETE_AFTER_RETAIN` defaults to true.

- When true, HMS stores the uploaded source long enough to validate and create
  a durable canonical descriptor/child retain payload, then deletes the source.
  This can happen before the child retain reaches `recall_ready`. The resulting
  memory records `media_source_available=false`; a later child failure cannot
  assume the original bytes are still available.
- When false, the source remains in the configured internal FileStorage and
  `media_source_available=true`. This release does not promise a public download
  URL for that object.
- HMS checks `FileStorage.exists()` after the configured deletion attempt. A
  lost delete response followed by `exists=false` is recorded as deleted; a
  nominally successful delete followed by `exists=true` is recorded as still
  available and emits a bounded failure metric. If the existence check itself
  is unavailable, provenance records `media_source_available=unknown` instead
  of optimistically claiming that the source remains. Object-store permission
  and network errors are not collapsed into `exists=false`.
- In either mode, the asset SHA-256, detected MIME, descriptor version, and
  canonical memory remain. Intermediate decoded frames are not ordinary source
  objects and must not be treated as downloadable artifacts.

A successful descriptor can also be stored as a sanitized durable checkpoint
for retry and deduplication. That checkpoint contains canonical Markdown,
bounded provenance, entities, and usage metadata, but no data URL or media
bytes. Deleting the source object does not immediately delete this derived
record. It expires after `HMS_API_MULTIMODAL_DESCRIPTOR_CACHE_TTL_SECONDS`
(7 days by default); a later multimodal conversion for the same bank removes up
to 100 expired completed checkpoints per cleanup invocation. This is lazy,
bank-scoped cleanup, so the TTL is an eligibility deadline, not a promise that
a background job deletes the row at that exact second. Cache reads enforce the
deadline even when more than 100 expired rows remain. The narrow exception is
an unfinished/retryable document command whose immutable source was already
verified deleted: its own descriptor is pinned as the command's recovery
checkpoint until that command completes or is superseded, but is not reusable
by unrelated commands. Bank deletion still cascades to the multimodal ledger.
There is no public per-asset checkpoint download API. Include the cache in
retention, backup, deletion, and access-control reviews.

For video map work, HMS also keeps short-lived per-segment checkpoints guarded
by the current active descriptor claim. Each row is keyed by the segment ID
plus the complete system-owned evidence fingerprint and contains only
schema-validated derived segment text, bounded provider metadata, and usage
counts. If one map segment fails, a retry may reuse matching, unexpired
successful segments and call only the missing/failed windows. A successor that
holds the active descriptor claim may reuse a row with the same descriptor and
evidence identity; changing the frame selection or pipeline identity, or
letting the checkpoint TTL expire, invalidates it. These checkpoints never
authorize partial child retain publication: the reducer and canonical document
are formed only after every required segment is available, and no media bytes
or data URL is stored in the checkpoint table.

The external provider call is an at-least-once side effect. A timeout or worker
crash after the provider accepted a request but before the local checkpoint can
lead to another billed attempt. Do not promise exactly-once provider billing.
Before enabling high-concurrency updates to the same logical `document_id`,
validate durable command ordering and crash recovery against the exact build
being deployed.

## Data controls and privacy

The data path has three distinct retention surfaces:

1. **HMS source storage.** Original uploaded bytes are stored in the configured
   HMS FileStorage until the source-deletion policy removes them. Apply the same
   tenant ACL, encryption-at-rest, backup, and retention policy as other files.
2. **HMS derived memory.** OCR, code, error text, descriptions, descriptor
   checkpoints, canonical Markdown, embeddings, and provenance are derived
   customer data and remain in HMS even when the source asset is deleted.
3. **Descriptor provider.** OpenAI receives one normalized image, or selected
   normalized video frames plus prompts. It does not receive the raw video or
   audio track in this implementation.

Every Responses request sets `store:false`. According to OpenAI's current data
controls documentation, that disables Responses application-state storage for
this request. It does **not** mean the organization has Zero Data Retention:

- abuse-monitoring logs may contain customer content and are retained for up to
  30 days by default;
- OpenAI states that API data is not used to train or improve its models unless
  the customer explicitly opts in, but this training choice is separate from
  abuse-monitoring and application-state retention;
- Modified Abuse Monitoring and Zero Data Retention require OpenAI approval and
  organization/project configuration, and have documented limitations;
- image and file inputs are scanned for CSAM, and flagged content may be
  retained for manual review even under eligible retention controls;
- a custom OpenAI-compatible endpoint has its own retention and training terms.

Before enabling the feature, the operator must approve media egress, verify the
OpenAI organization/project data-control setting (or the custom provider's
equivalent), confirm region and legal requirements, and use a dedicated secret.
Do not send customer media during engineering smoke tests.

HMS intentionally keeps data URLs out of task payloads, operation metadata,
canonical documents, normal logs, and generic LLM tracing. Infrastructure can
still defeat this boundary if an HTTP proxy, SDK debug logger, packet capture,
or exception middleware records request bodies. Review those components in the
actual deployment.

## Errors and product limitations

Failures discovered after upload are normally asynchronous. During operation
polling, inspect the allowlisted
`result_metadata.multimodal.sanitized_error_code`; do not parse the human error
message. This bounded field is part of the typed public SDK contract. Common
categories include:

| Code/category | Meaning | Normal action |
| --- | --- | --- |
| `media.mime_mismatch`, `media.extension_mismatch` | Declared type or known extension conflicts with decoded bytes. | Correct the upload metadata/file; do not retry unchanged input. |
| `media.image_*_exceeded`, `media.video_*_exceeded` | Local bytes, pixels, duration, frame, work, or memory budget was exceeded. | Reduce/transcode the asset within policy. |
| `media.video_decoder_unavailable` | PyAV is not importable or the linked FFmpeg lacks a usable H.264 decoder. | Install the video extra/decoder or keep video disabled. |
| `media.video_disabled` | The image feature is available but video is not enabled. | Enable only after decoder and egress review. |
| `provider.authentication`, `provider.request_rejected` | Credential, model access, or deterministic provider request failure. | Fix configuration; unchanged retries are not useful. |
| `provider.rate_limited`, `provider.unavailable`, `provider.network_unavailable` | Retryable transport class after bounded internal retries. | Respect `retryable`; account for possible billed attempts. |
| `provider.refusal` | Safety refusal. | Do not treat it as malformed JSON or silently OCR. |
| `provider.incomplete*` | Provider reported incomplete output. | Review output budget/content policy; no partial memory is retained. |
| `provider.schema_invalid`, `grounding.*` | Strict structure or evidence binding failed. | Fix the general schema/prompt/provider issue; do not patch a single fixture. |
| `multimodal.command_superseded` | A newer accepted update owns the same logical document. | Poll the newer operation; do not retry the obsolete command as a new append. |
| `retain_failed` stage | Descriptor exists, but the child HMS retain failed or disappeared. | Inspect/reprocess through the documented operation path; recall is not ready. |

Additional limitations:

- no raw-video provider input, URL fetching, standalone audio ingestion, audio
  transcription, image generation, or video generation;
- no face identification or protected-attribute inference;
- small/rotated text, exact counting, fine spatial relationships, and complex
  diagrams remain error-prone visual tasks;
- model-reported uncertainty is an audit hint, not a calibrated probability;
- descriptions and OCR are untrusted derived data, even when schema-valid;
- one failed required video segment fails the asset; HMS does not retain a
  partial video description.

## Verification and live qualification

### Required offline checks

Use the repository interpreter explicitly because the `pytest` entry point in
some snapshots may have a stale checkout in its shebang:

```bash
cd core/dataplane
../../.venv/bin/python -m pytest -p no:cacheprovider -o addopts='' -q \
  tests/test_multimodal_models.py \
  tests/test_multimodal_image.py \
  tests/test_multimodal_provider.py \
  tests/test_multimodal_video.py \
  tests/test_multimodal_parser.py \
  tests/test_multimodal_capabilities.py \
  tests/test_multimodal_operation_contract.py
```

The PostgreSQL upload-to-recall test requires a disposable database with
pgvector and does not call OpenAI:

```bash
HMS_TEST_MULTIMODAL_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DB' \
../../.venv/bin/python -m pytest -p no:cacheprovider -o addopts='' -q \
  tests/e2e/test_multimodal_offline_e2e.py
```

Passing these tests verifies the offline code path only.

### Live-provider gate

There is currently no checked-in automated billable live-test module. Do not
run a guessed pytest command or set `HMS_API_MULTIMODAL_LIVE_VERIFIED=true`.
Before a live-provider check, obtain explicit approval for cost and data
transmission, use non-sensitive operator-owned media, keep
`HMS_API_MULTIMODAL_LIVE_VERIFIED=false`, and verify the exact server/worker
configuration. Set the marker only after that configuration passes the
operator's independently managed deployment criteria. Do not commit test media,
provider responses, measurements, or qualification results to this repository.

## API and SDK publication status

The existing multipart upload endpoint remains public and unchanged. Only the
additive `/version` negotiation fields and typed
`result_metadata.multimodal` namespace described above are new. The default
`/version` response and all non-multimodal `result_metadata` behavior remain
compatible with the legacy contract.

`knowledge/site/static/openapi.json` is the checked-in canonical generator
input. Python, TypeScript, and Go keep generated sources in the repository; Rust
does not keep a generated source file and instead consumes the same canonical
schema from `interface/sdk/rust/build.rs` during Cargo builds. A contract change
is synchronized only when all four consumers have been regenerated or compiled
and their version-skew tests pass. Package-registry publication is a separate
release action.

From the repository root, regenerate the canonical schema with:

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=core/dataplane .venv/bin/python \
  lab/evaluation/hms_dev/generate_openapi.py \
  knowledge/site/static/openapi.json
```

OpenAPI Generator 7.10 cannot consume the canonical OpenAPI 3.1 null-union
shape for its Python and Go targets. Generate a disposable OpenAPI 3.0
compatibility view first; the checked-in canonical document remains 3.1:

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=core/dataplane .venv/bin/python \
  lab/evaluation/hms_dev/generate_openapi.py \
  /tmp/hms-openapi-compat30.json \
  --compatibility-openapi-30
```

Then use the repository's OpenAPI Generator 7.10 jar. `JAVA_HOME` is a
caller-provided JRE/JDK location:

```bash
(
  cd interface/sdk/python
  "$JAVA_HOME/bin/java" -jar ../go/openapi-generator-cli.jar generate \
    -g python \
    -i /tmp/hms-openapi-compat30.json \
    -c openapi-generator-config.yaml \
    -o .
)

(
  cd interface/sdk/go
  "$JAVA_HOME/bin/java" -jar openapi-generator-cli.jar generate \
    -i /tmp/hms-openapi-compat30.json \
    -c openapi-generator-config.yaml
)
```

Regenerate TypeScript and compile the Rust build-time client from the same
canonical input with:

```bash
npm --prefix interface/sdk/typescript run generate
cargo check --manifest-path interface/sdk/rust/Cargo.toml
```

Review generated diffs rather than editing generated models by hand. In
particular, keep the three capability fields optional/default-false and keep the
`multimodal` child optional while preserving arbitrary legacy sibling keys in
`result_metadata`.

The wire change is additive, but generated static types intentionally narrow
`result_metadata` from an unstructured map to `OperationResultMetadata`.
Legacy sibling keys remain available through Python's
`additional_properties`, Go's `AdditionalProperties`, and TypeScript's index
signature. Go also retains aliases for the pre-7.10 monitoring request type
names. Applications that directly indexed the old Python/Go map should migrate
to those compatibility containers when adopting the newly generated SDK; an
older SDK remains wire-compatible with the new server response.

## References

- [OpenAI GPT-5 mini model](https://developers.openai.com/api/docs/models/gpt-5-mini)
- [OpenAI images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)
