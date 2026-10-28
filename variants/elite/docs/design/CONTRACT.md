# HMS Elite — IMPLEMENTATION CONTRACT (实现契约)
中文版：[CONTRACT.zh.md](CONTRACT.zh.md)

> Normative. Independent agents implement DISJOINT owner groups (A..H, T, X) from this file alone. Where `DESIGN.md` and this file disagree, this file wins. Every signature, enum value, column, JSON key, error code and expected output below is exact; "may" is never used for anything a test asserts.
>
> Repo: `/Users/lazy/Documents/Claude Code/hms-elite` — flat layout (`hms_elite/` at repo root, `tests/` beside it), `pyproject.toml` (hatchling), Python `>=3.10,<4.0`, core dependency ONLY `pydantic>=2.7`; extras `numpy`, `openai`, `anthropic`, `milvus`, `spacy`, `server`, `test` (pytest, pytest-mock, httpx), `dev`. ruff line length 120, isort profile black.

---

## §0 Conventions (all groups)

### §0.1 Identifiers, clock, hashes

| Item | Rule |
|---|---|
| Object ids | `store.new_id(prefix) -> f"{prefix}_{secrets.token_hex(6)}"` (12 hex). Prefixes: `evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst cst`. Models accept any non-empty string (tests inject `f"{prefix}_{n:06d}"`); ids handed to a writer non-empty are preserved (import path). |
| Mem0 `id` | `MemoryAtom.memory_id` (prefix `mem`) = **lineage id**, stable across revisions. `atom_id` (prefix `atm`) = version id. `graph_nodes.node_id == atom_id` (one node per atom version). |
| Clock | `store.now()` is the ONLY clock in the library. Format: fixed-width UTC ISO `YYYY-MM-DDTHH:MM:SS.ffffff+00:00` (32 chars) produced by `timeutil.now_iso()`; every stored instant passes through `timeutil.to_utc_iso()` so TEXT order == time order. |
| Three instants per atom | `time_anchor` (event time; resolved date for `temporal_fact`), `created_at` (ingest time = `store.now()`), validity window `valid_from` (= event time) / `valid_to` (NULL = valid). `as_of` defaults to `store.now()`. |
| Event hash | `payload_hash = sha256(canonical_json({"object_type", "role", "actor_id", "payload"})).hexdigest()` over the **normalized** payload (§4.3), `canonical_json = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`. `Anchor.hash == SourceEvent.payload_hash`. |
| Atom hash | `MemoryAtom.hash = md5(text.encode("utf-8")).hexdigest()` (Mem0 parity for `row['hash']` and dedup). |
| Scope | `Scope(user_id, agent_id, run_id)`; `Scope.key()` = Mem0 `_build_session_scope`: sorted `k=v` joined by `&`, values escaped `%→%25`, `&→%26`, `=→%3D`. Every plane table denormalizes the three columns; DDL `CHECK` requires ≥1 non-NULL. |
| Visibility lattice | `private(0) < team(1) < event(2) < public(3)`; `rank(v)`. An item may be shown to audience `a` iff `rank(item.visibility) >= rank(a)`. |
| Token estimate | `estimate_tokens(text) = (len(text.encode("utf-8")) + 3) // 4 + 8`. |
| Determinism | No `random`, no `hash()` (use `hashlib`), no wall clock outside `timeutil.now_iso()`, every `ORDER BY` ends with an id tie-breaker, every **genuinely unordered** set is iterated sorted. **Ordered unions** keep first-occurrence order and are never sorted: wherever this contract says "ordered union" / `ordered_union(...)` (§9.2) the result is the concatenation of the inputs with later duplicates dropped (used for `w_t`, `kappa.constraints`, `b`, topic lists, `anchor_ids` of derived objects). |

### §0.2 Error codes (single table; `hms_elite/exceptions.py`, §3)

`VALIDATION_001` no scope id · `_002` bad memory_type · `_003` bad messages type · `_004` bad timestamp/date · `_005` malformed event payload · `_006` unknown feedback type · `_007` correction required / internal object_type in add_events · `_008` top-level ids in a read (also is-a `ValueError`) · `_009` bad threshold/top_k · `_010` empty query · `_011` update without text/metadata/expiration_date · `_012` bad filter key/operator · `_013` bad phases/mode/fmt/aggregation literal (also `aggregation='topk'` without `text`) · `NOT_FOUND_001` unknown id (is-a `ValueError`) · `CONFIG_001` pydantic config error · `CONFIG_002` unknown provider · `CONFIG_003` removed config key · `LLM_001` transport · `LLM_002` unknown task marker · `LLM_003` scripted queue exhausted · `EMBED_001` · `STORE_001` sqlite failure · `STORE_002` db newer than library · `STORE_003` invariant trigger fired · `WITNESS_001` · `PCIR_001` · `DEPS_001` · `INTERNAL_000` unmapped exception behind the HTTP server (§17.7; a plain `HMSError`, message redacted).

### §0.3 Bilingual docstring format (HMS §5.2 verbatim) — mandatory on EVERY code file (`hms_elite/**/*.py`, `tests/**/*.py`, `examples/*.py`) and every public function/method

```python
"""<Module purpose in English.>
<模块目的中文说明。>
"""

# Purpose (EN): What this file is responsible for.
# 目的（中）：说明本文件负责什么。

# Input (EN): What enters this module.
# 输入（中）：说明模块接收什么。

# Output (EN): What this module produces.
# 输出（中）：说明模块产出什么。

# State effect (EN): Whether this module mutates state, graph, index, or store.
# 状态影响（中）：说明模块是否修改状态、图、索引或存储。


def fn(store: StateStore, query: str) -> T:
    """<One EN sentence.>
    <一句中文。>

    Input / 输入：
        store: <EN description>.
        store：<中文说明>。

        query: <EN description>.
        query：<中文说明>。

    Output / 输出：
        <EN description>.
        <中文说明>。

    State effect / 状态影响：
        <EN description>.
        <中文说明>。
    """
```

Exact layout rules (asserted by `test_bilingual_docstrings`, §18.3 #64): the module docstring is two lines (EN, then ZH); the four `# Purpose/# 目的/# Input/# 输入/# Output/# 输出/# State effect/# 状态影响` comment pairs follow the module docstring, each pair separated by a blank line; every public function docstring starts with one EN line followed by one ZH line, then a blank line, then the three markers `Input / 输入：`, `Output / 输出：`, `State effect / 状态影响：` in this order; under `Input / 输入：` every argument is an EN line `<arg>: ...` immediately followed by a ZH line `<arg>：...` (full-width colon), arguments separated by one blank line (a function without arguments writes `        none.` / `        无。`); under `Output / 输出：` and `State effect / 状态影响：` exactly one EN line followed by one ZH line. Private helpers (`_name`) need only the two-line EN/ZH summary.

In this contract every function is listed as `signature` followed by `— EN | 中文` (purpose), then `In:`, `Out:`, `Effect:` lines (each `EN | 中文`), then `Notes:`. Implementers expand that into the block above verbatim (one `Input` pair per argument). Code comments are English only.

### §0.4 Import topology (tested by `tests/test_foundation.py::test_import_topology`)

Allowed edges: `B→A`, `C→A,B`, `D→A,B`, `E→A,B,C`, `F→A,B,D,E`, `G→A,B,C,D,E,F`, `H→all`, `T→all`. `hms_elite/mvmi/*` and `hms_elite/utils/scoring.py` must not import `hms_elite.graph`, `hms_elite.compiler`, `hms_elite.ops`, `hms_elite.ingest`. No module imports an optional SDK at module import except `llms/openai.py`, `llms/anthropic.py`, `embeddings/openai.py` (Mem0-style guard raising `DependencyError`); `index/milvus.py`, `entities/spacy.py` and `server/app.py` import their SDK lazily inside the constructor / `create_app()`, so `hms_elite.index`, `hms_elite.entities` and `hms_elite.server` import without the extra. `hms_elite/server/*`, `hms_elite/hms4eye/*` and `memory/async_main.py` wrap the facade and are group H like `cli.py`; `hms_elite/reranker/*` and `hms_elite/entities/*` are provider packages of group B (B → A only). Group C never imports `hms_elite.mvmi` or `hms_elite.ops`: everything C needs from those planes (`MemoryAtom.to_public_dict`, `validate_payload`, `Transition.from_ep`, `store.valid_ids`) lives in group A (§4, §8). Group G imports the pure rule functions of group B (`hms_elite.llms.rule.propose_edges / decide_revision / EXCLUSIVE / PREDICATE_FAMILIES`, §11) instead of re-implementing the tables.

### §0.5 HMS features intentionally out of scope (normative scope table)

| # | Full HMS feature (HMS_arch §) | Lite decision | Reason |
|---|---|---|---|
| 1 | `daemon.py` background loop (§3.9, §5.1) | SHIPPED as `hms_elite/daemon.py::StateDaemon(hms, *, interval_s, phases, filters)` — a named thread running `consolidate()` on an interval, `run_once()` returning the phase report, an exception logged and the loop continued (`tests/test_daemon.py`); `consolidate()` itself stays synchronous and `compile_state` stays consolidate step (7) | the loop only schedules the synchronous consolidate |
| 2 | `runtime.py` agent adapter / action registry (§5.1) | SHIPPED as `hms_elite/runtime.py` (`AgentRuntime`, `ActionRegistry`, `ActionSpec`, `PendingAction`): tool calls and their results become `file_diff / tool_call / test_log` events closed through `add_events()`; `why / handoff / checkpoint / rollback / feedback` delegate to the facade (`tests/test_runtime.py`) | `add_events()` stays the underlying contract |
| 3 | Per-source adapters (slack/jira/feishu/sensor) (§3.1) | SIMPLIFIED → one `SourceEvent` with the 5 extractable `object_type`s `message, tool_call, file_diff, test_log, sensor_trace` | transport glue, not invariants |
| 4 | `segmentation.py` (§5.1) | SHIPPED as `hms_elite/segmentation.py` (`segment`, `segment_messages`, `protected_spans`, `detect_lang`): sentences, paragraphs, code blocks and list items with exact spans; code blocks, URLs and identifiers are never cut; deterministic (`tests/test_segmentation.py`) | rule-based, no model |
| 5 | `coverage.py`/`scc.py` hierarchical coverage compiler (§5.1) | SHIPPED as an optional layer: `compiler/coverage.py` (`extract_requirements`, `covers`, `select_for_coverage` — greedy set cover) and `compiler/scc.py` (`compile_hierarchical` → `HierarchicalIR`: L3/L2 interfaces first, L1 atoms on demand; `lower()` → `HierarchicalPCIR` with coverage obligations; `tests/test_coverage_scc.py`); `compile()` keeps the per-slot share budget with a constraint floor by default | accuracy feature, not an invariant |
| 6 | Vision / perception | SIMPLIFIED → no offline perception: list content reduced to text parts, image parts anchored as `parts` (sha256 + mime, never bytes) and counted in `dropped_parts`; with `llm.enable_vision=True` a vision-capable provider's `describe_image` hook (unlocked, before Txn A) merges one `[Image: …]` line per part into `text`, records `described_parts` and subtracts them from `dropped_parts` (`openai` / `anthropic` implement it; `rule` returns None) | perception is a provider capability, not a library one |
| 7 | FTS5/BM25 lexical axis | SPLIT → recall order comes from the stdlib SQLite FTS5 table `atom_fts` (bm25, `keyword_candidates`, LIKE fallback without the module); the in-pool keyword *score* stays the pure-Python token overlap with identifier tokens ×2 | zero optional deps; bounded recall (RETRIEVAL_DIAGNOSIS §3.2) |
| 8 | Reranker providers | SIMPLIFIED → `overlap` (default, offline weighted token coverage) + `llm` (the `rerank` prompt contract; falls back to overlap on `LLM_002`); `search(rerank=True)` applies `self.reranker` after fusion (unconfigured → a default `OverlapReranker`), rows gain `rerank_score`, a reranker failure keeps the retrieval order | Mem0 kwarg parity with one offline default |
| 9 | Per-topic half-life learner | SIMPLIFIED → one `scoring.half_life_days` | determinism |
| 10 | Embedding-tier entity resolution | SIMPLIFIED → normalized-text exact match + alias list; `run_id` excluded from entity identity | hash embeddings are not semantic |
| 11 | Personal/Edge/Group renderers (§11.2) | SHIPPED as `hms_elite/state/` (`personal / edge / group / enterprise` state kinds, deterministic renderers, `StateAPI`; `tests/test_state.py`); `compile()` / `handoff()` still emit the Work State Card + `StatePackage` | every state is witnessed and its rendering is deterministic |
| 12 | LLM coarsen | SIMPLIFIED → `rule` coarsen default; LLM coarsen behind the same prompt contract | reproducible round-trip test |
| 13 | Skill promotion (§9.7) | SIMPLIFIED → an InterfaceEP invariant of kind `constraint` with ≥2 support nodes becomes an `L3_macro` skill atom + traceable `distill` edge | reachable in one run |
| 14 | Governance/ACL evaluator for κ (§3.8) | SIMPLIFIED → visibility lattice gate at selection time + `governance`/`not_revoked` obligations | hook for a policy engine |
| 15 | Benchmark harness (§12 Phase 5) | SHIPPED as `hms_elite/eval/` (`run_ablation`, `Task`, `TASKS` — the nine §10 tasks —, `AblationReport`, `render`, `to_json`; a LongMemEval adapter enabled when the dataset is present; `tests/test_ablation.py`); the four ablation configurations still map to `CompileMode`: `fast` = Axis only, `axis_graph` = Axis + Graph (expansion, no `refine`), `deep` = Axis + Graph + Support (refine → invariants/residuals); "Graph only" has no counterpart (the graph is reachable only from axis seeds) and is recorded as not applicable; `retrieval['ablation'] ∈ {'axis','axis_graph','axis_graph_support'}` | in the package; datasets are not shipped |
| 16 | HTTP server | SHIPPED as `hms_elite/server` (`create_app()` / `hms-server`, optional extra `server`, group H like `cli.py`); `cli.py` ships | library first, the service face only wraps the facade |
| 17 | External index providers | SIMPLIFIED; `IndexBase` pluggable, ships `sqlite` + `memory` in the core and `milvus` (milvus-lite `.db` file or a server uri) behind the optional `milvus` extra (registry entry + `EXTRA_OF_PROVIDER`); hits are re-scored by `rank_hits`, so scores are bit-identical across providers | fixed decision 3 (core stays pydantic-only) |
| 18 | Async twin, telemetry, notices, `setup_config()` | Async twin and telemetry SHIPPED: `AsyncHMS` (`memory/async_main.py`, the `asyncio.to_thread` twin with pairwise-identical signatures, §17.8) and `hms_elite/telemetry.py` (`Usage`, `UsageMeter`, `trace()`, `usage_of(hms)`: calls, tokens and durations of the LLM / embedding providers; `GET /usage` / `GET /metrics`, §17.7); notices and `setup_config()` stay CUT | fixed decision 9 constrains only the synchronous consolidate |
| 19 | Mem0 `chat()` / `project` | `chat()` raises `NotImplementedError`; `project.update()` raises `ValueError` | Mem0 API-shape parity |
| 20 | Mem0 `history.db` import | CUT (replaced by `hms_elite.migrate.from_mem0`) | no column-set-diff migration |

Everything in HMS_arch §3 not listed above is IN scope: seven query axes with executable `AxisQuery` semantics (§13.3), entity/relation atoms (§16.5 step 0), L2 **and** L3 InterfaceEPs (§16.5 step 4/4b), anchor revoke with cascade (§16.1), state-package import (§16.3), InterfaceEP transitions (§16.4), compiled-state snapshots (§16.5 step 7).

---

## §1 File tree with owner groups (每个文件唯一归属)

```
hms-elite/
├── pyproject.toml                                  X
├── README.md                                       X
├── examples/quickstart.py                          X   (§20.3; runs §19 S1–S6 offline)
├── hms_elite/
│   ├── __init__.py                                 H   exports HMS, HMSConfig, __version__ (importlib.metadata("hms-elite")); no side effects
│   ├── enums.py                                    A   §2
│   ├── exceptions.py                               A   §3
│   ├── models.py                                   A   §4 (all pydantic models, RESERVED_ATOM_KEYS)
│   ├── timeutil.py                                 A   §9.1
│   ├── configs/
│   │   ├── __init__.py                             A
│   │   ├── registry.py                             A   §5.4 REGISTRY (names → dotted paths; no imports)
│   │   ├── base.py                                 A   §5 HMSConfig + sub-configs + envelopes + RENAMED/REMOVED
│   │   └── prompts.py                              A   §6 prompts, task markers, builders, section parser
│   ├── utils/
│   │   ├── __init__.py                             A
│   │   ├── text.py                                 A   §9.2 tokenize/sentences/entities/topics/sentiment/lexicons
│   │   ├── json.py                                 A   §9.3 canonical_json, remove_code_blocks, extract_json, parse_llm_json ladder
│   │   ├── factory.py                              B   §10.4 LlmFactory/EmbedderFactory/IndexFactory, load_class, register_provider
│   │   └── scoring.py                              D   §13.4 signals + fuse + score_details
│   ├── memory/
│   │   ├── __init__.py                             A
│   │   ├── base.py                                 A   §9.4 MemoryBase(ABC) (Mem0 contract)
│   │   ├── utils.py                                A   §9.5 parse_messages, build_scope_and_metadata, strip_reserved_keys, estimate_tokens, _UNSET
│   │   ├── schema.py                               A   §7 DDL + MIGRATIONS + apply_migrations
│   │   ├── storage.py                              A   §8 StateStore
│   │   └── main.py                                 H   §17 class HMS
│   ├── llms/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   §10.1 LLMBase, BaseLlmConfig
│   │   ├── rule.py                                 B   §11 RuleLLM (deterministic per-task heuristics)
│   │   ├── scripted.py                             B   §11.7 ScriptedLLM
│   │   ├── openai.py                               B   §10.1 OpenAILLM (guarded)
│   │   ├── anthropic.py                            B   §10.1 AnthropicLLM (guarded)
│   │   ├── deepseek.py                             B   §10.1 DeepSeekLLM (OpenAI-compatible endpoint; openai SDK, guarded)
│   │   └── ollama.py                               B   §10.1 OllamaLLM (stdlib urllib; no SDK)
│   ├── embeddings/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   §10.2 EmbedderBase
│   │   ├── hash.py                                 B   §10.2 HashEmbedder
│   │   ├── openai.py                               B   §10.2 OpenAIEmbedder (guarded)
│   │   ├── ollama.py                               B   §10.2 OllamaEmbedder (stdlib urllib; dims inferred and cached)
│   │   └── fastembed.py                            B   §10.2 FastEmbedEmbedder (local inference; guarded)
│   ├── index/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   §10.3 IndexBase
│   │   ├── sqlite.py                               B   §10.3 SQLiteIndex (atom_vectors in the same DB)
│   │   ├── memory.py                               B   §10.3 MemoryIndex
│   │   └── milvus.py                               B   §10.3 MilvusIndex (optional `milvus` extra; lazy `pymilvus` import → DependencyError)
│   ├── entities/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   §10.5 EntityExtractorBase + merge_candidates
│   │   ├── rule.py                                 B   §10.5 RuleEntityExtractor (default; == utils.text.extract_entities)
│   │   └── spacy.py                                B   §10.5 SpacyEntityExtractor (lazy `spacy` import → DependencyError)
│   ├── reranker/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   RerankerBase, BaseRerankerConfig, rank_rows (Mem0 rerank shape)
│   │   ├── overlap.py                              B   OverlapReranker (default; offline weighted token coverage)
│   │   ├── llm.py                                  B   LLMReranker (`rerank` prompt contract; overlap fallback on LLM_002)
│   │   └── prompts.py                              B   RERANK_PROMPT + build_rerank_prompt (§6.1 shape)
│   ├── ingest/
│   │   ├── __init__.py                             C
│   │   ├── events.py                               C   §12.1 messages/dicts → SourceEvent, payload normalization, raw_ref, anchoring (Txn A)
│   │   ├── router.py                               C   §12.2 route_event, has_action_closure
│   │   ├── extraction.py                           C   §12.3 LLM-backed observation/EP extraction + revision decision (unlocked phase)
│   │   ├── closure.py                              C   §12.4 structured closure builder (no LLM)
│   │   ├── atoms.py                                C   §12.5 create_atom primitive, link_atom_axes, atom builders
│   │   └── pipeline.py                             C   §12.6 ingest_messages / ingest_events orchestration (Txn A → unlocked → Txn B)
│   ├── mvmi/
│   │   ├── __init__.py                             D
│   │   ├── filters.py                              D   §13.1 FilterExpr AST, parse_filters, lower_to_sql, evaluate, plan_filters
│   │   ├── planner.py                              D   §13.2 plan_query (routes, time/entity/identifier hints)
│   │   └── query.py                                D   §13.3 execute_axis_query, search, get_all, atom_to_row
│   ├── graph/
│   │   ├── __init__.py                             E
│   │   ├── support_graph.py                        E   §14.1 SupportGraph (nodes/edges/neighbors/boundary/cohesion)
│   │   ├── operators.py                            E   §14.2 coarsen / project / refine / promote_skill
│   │   └── explain.py                              E   §14.3 explain (why) + chain rendering
│   ├── compiler/
│   │   ├── __init__.py                             F
│   │   ├── slots.py                                F   §15.1 assign_slot, ContextItem builders, next-action rule
│   │   ├── compile.py                              F   §15.2 StateCompiler.compile, budget
│   │   ├── render.py                               F   §15.3 Work State Card renderer
│   │   ├── pc_ir.py                                F   §15.4 lower(ContextIR) → PCIR, obligations
│   │   ├── coverage.py                             F   optional layer: extract_requirements, covers, select_for_coverage (greedy set cover)
│   │   └── scc.py                                  F   optional layer: compile_hierarchical → HierarchicalIR, lower → HierarchicalPCIR
│   ├── ops/
│   │   ├── __init__.py                             G
│   │   ├── feedback.py                             G   §16.1 FeedbackProcessor + revoke()
│   │   ├── checkpoint.py                           G   §16.2 checkpoint / rollback
│   │   ├── handoff.py                              G   §16.3 StatePackage builder + import_state_package()
│   │   ├── transitions.py                          G   §16.4 export_transitions
│   │   ├── consolidation.py                        G   §16.5 StateScheduler (orient/collect/consolidate[entity atoms, edges, conflicts, coarsen L2+L3, skills, transitions, compile_state]/prune[decay+archive(+§16.6 purge when configured)])
│   │   └── retention.py                            G   §16.6 purge (explicit, default-off, audited physical purge of long-archived versions)
│   ├── memory/async_main.py                        H   AsyncHMS (asyncio.to_thread twin of HMS)
│   ├── daemon.py                                   H   StateDaemon (background consolidate loop over the facade; duck-typed, imports no plane)
│   ├── runtime.py                                  H   AgentRuntime / ActionRegistry (agent adapter over the facade; duck-typed)
│   ├── segmentation.py                             A   segment / segment_messages / protected_spans / detect_lang
│   ├── telemetry.py                                A   Usage / UsageMeter / MeteredProvider / trace / usage_of (§17.8)
│   ├── eval/                                       H   run_ablation, Task / TASKS, AblationReport, render / to_json (§0.5 row 15)
│   ├── state/                                      H   state kinds, deterministic renderers, StateAPI (§0.5 row 11)
│   ├── server/                                     H   FastAPI service face (`create_app`, `hms-server`; optional `server` extra)
│   ├── hms4eye/                                    H   hms4eye capability adapter over the facade
│   ├── migrate.py                                  H   §17.5 from_mem0
│   └── cli.py                                      H   §17.6
└── tests/
    ├── __init__.py                                 T
    ├── conftest.py                                 T   §18.1 (the ONLY conftest)
    ├── fixtures/events/*.json                      T   §18.2
    ├── fixtures/golden/*                           T   §18.2
    ├── test_foundation.py                          T   group A tests
    ├── test_providers.py                           T   group B tests
    ├── test_ingest.py                              T   group C tests
    ├── test_mvmi.py                                T   group D tests
    ├── test_graph.py                               T   group E tests
    ├── test_compiler.py                            T   group F tests
    ├── test_ops.py                                 T   group G tests
    ├── test_facade.py                              T   group H tests (Mem0 parity)
    └── test_scenarios.py                           T   §19 S1–S6 on both index providers
```

Group dependency order (acyclic): **A → B → C → D → E → F → G → H**, with the exact allowed edges of §0.4 (D does not depend on C; E depends on C for `ingest.atoms.create_atom`).

Every code file — modules, every `__init__.py`, every `tests/*.py`, `tests/conftest.py`, `examples/quickstart.py` — starts with the §0.3 module docstring + header. Every `__init__.py` re-exports the public names listed in its section and nothing else.

---

## §2 Enums (`hms_elite/enums.py`, group A)

All enums are `class X(str, Enum)`; member name == value unless stated. `values(cls) -> list[str]` classmethod on a shared `StrEnum` base; DDL `CHECK` lists are generated from these (`schema.py` imports enums).

```python
class EdgeType(StrEnum):        temporal, causal, revision, conflict, use, distill
class FeedbackType(StrEnum):    state_wrong, state_outdated, state_missing, privacy_violation, next_action_wrong,
                                relationship_not_useful, connection_reason_wrong, evidence_wrong,
                                render_not_like_me, render_like_me, object_important
class Scale(StrEnum):           L1_micro, L2_meso, L3_macro, state, event, entity, relation
class AtomType(StrEnum):        message, state, event, entity, relation, episode
class ObservationType(StrEnum): fact, preference_event, habit, constraint, temporal_fact
class Reward(StrEnum):          positive, negative, neutral, unknown
class Visibility(StrEnum):      private, team, event, public          # rank(): private=0, team=1, event=2, public=3
class Role(StrEnum):            user, assistant, system, tool, extracted
class Sentiment(StrEnum):       positive, neutral, negative, mixed, unknown
class Slot(StrEnum):            facts, episodes, preferences, constraints, skills, audit
class ObjectType(StrEnum):      message, tool_call, file_diff, test_log, sensor_trace, feedback, checkpoint, rollback, system, scheduler
class EventType(StrEnum):       created, updated, deleted, feedback, checkpoint, rollback, tool_result, scheduler
class NodeKind(StrEnum):        message, observation, event, entity, ep, interface_ep, skill
class SourceKind(StrEnum):      message, observation, event, entity, ep, interface_ep, skill, procedural, feedback
class Decision(StrEnum):        ADD, REVISE, CONFLICT, NONE
class HistoryEvent(StrEnum):    ADD, UPDATE, DELETE, ARCHIVE, RESTORE, FEEDBACK, CONFLICT, ROLLBACK, PURGE   # PURGE only from §16.6
class Route(StrEnum):           recall, preference, temporal, entity, work_state, detective, handoff
class CompileMode(StrEnum):     auto, fast, axis_graph, deep          # fast = Axis only; axis_graph = Axis + Graph (no refine); deep = Axis + Graph + Support
class Relation(StrEnum):        single_fact, comparison, habit, supersession, constraint
class EntityType(StrEnum):      PROPER, IDENTIFIER, FILE, QUOTED, TOPIC
class MemoryType(StrEnum):      SEMANTIC = "semantic_memory"; EPISODIC = "episodic_memory"; PROCEDURAL = "procedural_memory"
class TransitionKind(StrEnum):  ep, interface_ep, causal, revision, conflict, rollback
class InvariantKind(StrEnum):   constraint, failed_path, stable_fact
class ResidualKind(StrEnum):    unresolved_conflict, missing_outcome, low_cohesion, open_question
class Phase(StrEnum):           orient, collect, consolidate, prune
class ObligationName(StrEnum):  evidence_coverage, no_unresolved_conflict, governance, not_revoked
class ObligationStatus(StrEnum): satisfied, violated, skipped
```

`EXTRACTABLE_OBJECT_TYPES = {message, tool_call, file_diff, test_log, sensor_trace}` (five members); `INTERNAL_OBJECT_TYPES = {feedback, checkpoint, rollback, system, scheduler}` (rejected by `add_events`, `VALIDATION_007`). `WHY_EDGE_TYPES = {causal, use, distill, revision, conflict}` (default for `why()`/`compile` expansion; `temporal` excluded). `VISIBILITY_RANK = {private: 0, team: 1, event: 2, public: 3}`. `KIND_ORDER: dict[str, int] = {ep: 0, interface_ep: 1, causal: 2, revision: 3, conflict: 4, rollback: 5}` (transition ordering; owned by group A, used by §8 `list_transitions` and §16.4). `ABLATION_OF_MODE = {fast: 'axis', axis_graph: 'axis_graph', deep: 'axis_graph_support'}` (`auto` resolves to `fast` or `deep` at run time, §15.2).

---

## §3 Exceptions (`hms_elite/exceptions.py`, group A)

```python
class HMSError(Exception):
    def __init__(self, message: str, error_code: str = "HMS_000", details: dict | None = None,
                 suggestion: str | None = None, debug_info: dict | None = None) -> None
    # attributes message/error_code/details/suggestion/debug_info; __repr__ = f"{cls}(message=..., error_code=...)"; to_dict()
class ValidationError(HMSError, ValueError):   default error_code = "VALIDATION_000"
class ConfigError(ValidationError):            default "CONFIG_001"
class LLMError(HMSError):                      default "LLM_001", suggestion "Please check your LLM configuration and API key"
class EmbeddingError(HMSError):                default "EMBED_001"
class StoreError(HMSError):                    default "STORE_001"
class WitnessMissingError(StoreError):         default "WITNESS_001"; details {"object": "atom"|"edge"|"observation"|"ep"|"interface_ep"|"transition", "id": str|None}; suggestion "Every HMS object must reference at least one anchor_id; ingest the evidence first"
class ProofObligationError(HMSError):          default "PCIR_001"; details {"violated": [names], "missing_witness": [item_ids], "unresolved_conflicts": [edge_ids], "revoked": [anchor_ids], "governance": [item_ids]}
class DependencyError(HMSError):               default "DEPS_001"; details {"package": str, "extra": str}; suggestion f"pip install 'hms-elite[{extra}]'"
```

Rules: `hms_elite.exceptions.ValidationError` carries Mem0's attribute names so Mem0-ported `except ValueError` keeps working; bare `ValueError` is raised ONLY where Mem0 raises bare `ValueError` and is always a `ValidationError` subclass instance in HMS (so both `except ValueError` and `except HMSError` catch it). `sqlite3.IntegrityError` whose message starts with `HMS_INVARIANT` → `StoreError(STORE_003, details={"trigger": <text>})`. Witness checks are **explicit pre-checks, never message parsing**: every `StateStore` writer that takes anchor ids (`insert_atom`, `insert_edge`, `insert_observation`, `insert_ep`, `insert_interface_ep`, `upsert_transition`, `add_history`, `insert_feedback`, `insert_checkpoint`, `add_atom_anchor`, `add_edge_witness`) first rejects an empty list (`WitnessMissingError(details={'object': <kind>, 'id': <id or None>, 'missing': []})`) and then calls `get_anchors(ids)`; any id not returned → `WitnessMissingError(details={'object', 'id', 'missing': [unknown ids]})` before any SQL is built. A `NOT NULL constraint failed: <table>.<col>` error whose column name ends with `anchor_id` (raw-SQL paths only) is mapped to `WitnessMissingError` as a last resort; SQLite's bare `FOREIGN KEY constraint failed` is never interpreted.

---

## §4 Data models (`hms_elite/models.py`, group A)

Module docstring: `"""Pydantic models for every HMS Elite object; witnesses are types, not conventions.\nHMS Elite 全部对象的 pydantic 模型；witness 是类型而非约定。"""` Purpose: own every schema-bearing object | 拥有全部结构对象. Input: dicts/rows | 字典/数据库行. Output: validated models | 已校验模型. Effect: none (pure) | 无（纯函数）.

```python
from __future__ import annotations
import hashlib, json
from typing import Annotated, Any, Literal
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from hms_elite.enums import *  # noqa
from hms_elite.timeutil import now_iso, to_utc_iso

UtcIso = Annotated[str, AfterValidator(to_utc_iso)]                  # normalized to 32-char UTC ISO
IsoDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
Prob = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyIds = Annotated[list[str], Field(min_length=1)]               # the "No Witness, No Meaning" type


class HMSModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True, validate_default=True)


class Scope(HMSModel):
    """Mem0 session scope. Mem0 会话范围。"""
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    # field_validator (each id): coerce to str, strip; empty or internal whitespace → ValidationError(VALIDATION_001)
    def is_empty(self) -> bool: ...
    def require_any(self, *, context: str) -> "Scope": ...      # raises ValidationError(VALIDATION_001, details={"provided_ids": {...}, "context": context})
    def as_filters(self) -> dict[str, str]: ...                 # only set keys
    def key(self) -> str: ...                                    # §0.1 Mem0 scope key
    def matches(self, other: "Scope") -> bool: ...              # every set key of self equals other's


class Witnessed(HMSModel):
    """Any state-bearing object references >= 1 anchor. 任何承载状态的对象至少引用一个锚点。"""
    anchor_ids: NonEmptyIds                                       # anchor_ids[0] is the primary witness — the storage non-emptiness guarantee (NOT NULL column) and default history witness, NOT a necessary condition: the witness set is an existence condition, an object stays valid while any undisputed, unrevoked anchor remains (§16.1 Revoke; docs/design/WITNESS_SEMANTICS.md)


class SourceEvent(HMSModel):
    event_id: str = ""                                            # "" → assigned by the store on insert
    source_system: str = "hms"                                    # 'hms' | 'mem0_api' | 'claude_code' | free string
    object_type: ObjectType = ObjectType.message
    event_type: EventType = EventType.created
    actor_id: str | None = None
    role: Role | None = None                                      # user|assistant|system|tool (never 'extracted')
    timestamp: UtcIso = Field(default_factory=now_iso)           # event time; caller-settable (backdating allowed)
    raw_ref: str | None = None                                    # filled by ingest.events.default_raw_ref when None
    visibility: Visibility = Visibility.private
    scope: Scope
    payload: dict[str, Any]                                       # normalized by validate_payload (§4.3)
    payload_hash: str = ""                                        # computed when "" (§0.1); verified when given (mismatch → VALIDATION_005)
    user_meta: dict[str, Any] = Field(default_factory=dict)
    # model_validator(after): payload = validate_payload(object_type, payload) (§4.3; defined in THIS module); payload_hash computed; role 'extracted' rejected


class Anchor(HMSModel):
    anchor_id: str
    source_event_id: str
    hash: str
    source_ref: str | None = None
    timestamp: UtcIso
    ingested_at: UtcIso
    permission_scope: Visibility = Visibility.private
    scope: Scope
    revoked_at: UtcIso | None = None
    def evidence_line(self) -> str: ...                          # f"{source_ref or 'anchor'} sha256:{hash[:12]} @{timestamp}"


class Observation(Witnessed):
    obs_id: str = ""
    type: ObservationType
    subject: str = "user"                                         # 'user' | 'project' | 'assistant' | entity display text
    predicate: str                                                # lowercase snake verb (§11.1 predicate table)
    objects: list[str] = Field(default_factory=list)
    relation: Relation = Relation.single_fact
    text: str = Field(min_length=1)
    date_observed: IsoDate
    valid_from: UtcIso | None = None
    valid_to: UtcIso | None = None
    confidence: Prob = 0.7
    visibility: Visibility = Visibility.private
    attributed_to: Literal["user", "assistant"] = "user"
    speaker: str | None = None                                    # named person of a multi-speaker transcript (None in plain chat); atom actor_id when the event has none
    sentiment: Sentiment = Sentiment.unknown
    topic_tags: list[str] = Field(default_factory=list)
    entity_mentions: list[str] = Field(default_factory=list)
    supersedes_hint_ids: list[str] = Field(default_factory=list)  # real atom_ids after ref resolution
    extracted_by: str = "rule@obs-v1"
    scope: Scope
    atom_id: str | None = None
    created_at: UtcIso | None = None


class Kappa(HMSModel):
    constraints: list[str] = Field(default_factory=list)
    visibility: Visibility = Visibility.private
    evidence: list[str] = Field(default_factory=list)             # anchor ids


class Slots(HMSModel):
    """Isomorphic six-tuple shared by EP (L1) and InterfaceEP (L2/L3). 跨尺度同构六元组。"""
    u: str = Field(min_length=1)
    b: str = ""
    a: str = Field(min_length=1)
    y: str = ""
    r: Reward = Reward.unknown
    kappa: Kappa = Field(default_factory=Kappa)
    # model_validator(after): closure — y.strip() or kappa.constraints, else ValueError("EP needs an outcome or a constraint")
    def gloss(self) -> str: ...                                   # f"[{r}] {u} → {a} → {y or '?'}"


class EP(Witnessed):
    ep_id: str = ""
    scale: Literal["L1_micro"] = "L1_micro"
    slots: Slots
    time_anchor: UtcIso
    valid_from: UtcIso | None = None                              # defaults to time_anchor on insert
    valid_to: UtcIso | None = None
    entity_mentions: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    uses_ep_ids: list[str] = Field(default_factory=list)          # validated against existing EPs before a 'use' edge
    confidence: Prob = 0.7
    visibility: Visibility = Visibility.private
    extracted_by: str = "rule@closure-v1"
    scope: Scope
    atom_id: str | None = None
    created_at: UtcIso | None = None


class MemoryAtom(Witnessed):
    """Coordinates + provenance only; never edges. 只有坐标与出处，不存边。"""
    atom_id: str = ""
    memory_id: str = ""                                           # lineage id (Mem0 'id'); "" → new lineage on insert
    version: int = Field(default=1, ge=1)
    atom_type: AtomType
    content: dict[str, Any]                                       # §4.2 shapes; must contain 'text' == text
    text: str = Field(min_length=1)
    hash: str = ""                                                # md5(text); computed when ""
    time_anchor: UtcIso | None = None
    valid_from: UtcIso | None = None                              # defaults to time_anchor or created_at on insert
    valid_to: UtcIso | None = None
    entity_ids: list[str] = Field(default_factory=list)           # normalized entity keys
    topic_tags: list[str] = Field(default_factory=list)
    scale_level: Scale
    role: Role = Role.extracted
    sentiment: Sentiment | None = None
    subject: str | None = None
    predicate: str | None = None
    obs_type: ObservationType | None = None                       # promoted for SQL slot rules
    source_kind: SourceKind
    source_id: str                                                # event_id | obs_id | ep_id | interface_ep_id | feedback_id | atom_id(skill)
    source_ep_ids: list[str] = Field(default_factory=list)
    confidence: Prob = 0.7
    extracted_by: str
    visibility: Visibility = Visibility.private
    scope: Scope
    actor_id: str | None = None
    attributed_to: Literal["user", "assistant"] | None = None
    memory_type: str | None = None                                # 'procedural_memory' | None
    user_meta: dict[str, Any] = Field(default_factory=dict)       # caller namespace; reserved keys stripped by memory.utils
    archived_reason: str | None = None                            # revised|deleted|feedback|rollback|decayed|conflict
    created_at: UtcIso | None = None
    updated_at: UtcIso | None = None
    # model_validator(after): content['text'] == text; content has no key in {'edges','edge_ids','relations','src_id','dst_id'} (ValueError);
    #                         valid_to is None or valid_to >= valid_from; hash computed if ""
    def is_valid_at(self, as_of: str) -> bool: ...
    def expiration_date(self) -> str | None: ...                  # valid_to date-1 when valid_to is a next-day-midnight boundary created by expiration_date=, else None
    def to_public_dict(self, *, score: float | None = None, score_details: dict | None = None, anchor_ids: list[str] | None = None) -> dict: ...   # §4.1 — the CANONICAL implementation lives here (group A); mvmi/query.py::atom_to_row is a one-line wrapper


RESERVED_ATOM_KEYS: frozenset[str] = (
    frozenset(MemoryAtom.model_fields)
    | frozenset(f + "_json" for f in ("content", "user_meta", "source_ep_ids"))      # every *_json column of memory_atoms
    | frozenset({"id", "memory", "data", "score", "score_details", "hms", "user_id", "agent_id", "run_id",
                 "expiration_date", "text_lemmatized", "primary_anchor_id"})
)   # test 39 asserts RESERVED_ATOM_KEYS ⊇ every memory_atoms column (PRAGMA table_info)


class AxisQuery(HMSModel):
    """A query plan over the multi-view index (HMS §3.5). 多视角索引上的查询计划。"""
    filters: dict[str, Any] = Field(default_factory=dict)         # Mem0 grammar over the seven axes + scope + meta (§13.1)
    project_axis: Literal["time", "entity", "value", "topic", "atom"] = "atom"
    scale_constraint: list[Scale] | None = None
    aggregation: Literal["list", "latest", "count", "summary", "topk"] | None = "list"
    limit: int = Field(default=20, ge=0)
    as_of: UtcIso | None = None
    include_expired: bool = False
    scope: Scope | None = None                                    # None → derived from the scope keys inside `filters` by execute_axis_query (neither → VALIDATION_001)
    text: str | None = None                                       # semantic/keyword query; None → axis-only, time order
    meta_filters: dict[str, Any] | None = None                    # escape hatch: same grammar, compiled against user_meta only
    plan: dict[str, Any] = Field(default_factory=dict)            # planner output (§13.2); {} when built directly
    # model_validator(after): aggregation == 'topk' and text is None → ValidationError(VALIDATION_013)


class AtomHit(HMSModel):
    atom: MemoryAtom
    score: float = 0.0
    score_details: dict[str, Any] = Field(default_factory=dict)
    matched_axes: list[str] = Field(default_factory=list)


class AxisResult(HMSModel):
    """Executed AxisQuery: hits plus the requested projection/aggregation (HMS §3.5). 已执行的轴查询：命中与投影/聚合结果。"""
    hits: list[AtomHit]                                           # always the (aggregated, limited) atoms in result order
    project_axis: Literal["time", "entity", "value", "topic", "atom"]
    aggregation: Literal["list", "latest", "count", "summary", "topk"]
    projection: list[Any] | dict[str, list[str]] | None = None    # §13.3 step 5 per project_axis; None for 'atom'
    count: int | None = None                                      # aggregation == 'count'
    summary: str | None = None                                    # aggregation == 'summary'
    primary_axis: str = "time"
    retrieval: dict[str, Any] = {}                                # Pool.stats() of the winning ladder level for ranked queries ({} for axis-only / count): pool_size, channel_sizes, channels_used, gate_count (-1 under native pushdown), whole


class GraphNode(Witnessed):
    node_id: str                                                  # == atom_id
    node_kind: NodeKind
    ref_id: str                                                   # event_id | obs_id | ep_id | interface_ep_id | atom_id(skill)
    scale: Scale
    scope: Scope
    valid: bool = True
    invalidated_at: UtcIso | None = None
    mentions: int = 1
    created_at: UtcIso | None = None


class GraphEdge(HMSModel):
    edge_id: str = ""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    label: str = ""
    scale: Literal["L1_micro", "L2_meso", "L3_macro"] = "L1_micro"
    is_boundary: bool = False                                     # True only for edges written by operators.project
    confidence: Prob = 0.7
    mentions: int = 1
    witness_anchor_ids: NonEmptyIds
    witness_edge_id: str | None = None                            # REQUIRED iff is_boundary (validator + DDL CHECK)
    created_by: Literal["ingest", "consolidation", "feedback", "ops", "project"]
    proposer: str = "rule"                                        # 'rule' | 'closure' | 'timestamp' | 'llm:<identity>' | 'project-v1'
    valid: bool = True
    invalidated_at: UtcIso | None = None
    invalidated_reason: str | None = None
    resolved_at: UtcIso | None = None                             # conflict edges only
    resolved_by_anchor_id: str | None = None
    resolution_edge_id: str | None = None
    scope: Scope
    created_at: UtcIso | None = None
    # validators: src_id != dst_id; is_boundary ⇒ witness_edge_id; edge_type != conflict ⇒ resolved_* None; resolved_at ⇒ resolved_by_anchor_id


class BoundaryEdge(HMSModel):
    edge_id: str
    inner_node_id: str
    outer_node_id: str
    direction: Literal["in", "out"]                               # 'out' when src ∈ support set
    edge_type: EdgeType
    witness_anchor_ids: NonEmptyIds


class Invariant(HMSModel):
    text: str = Field(min_length=1)
    kind: InvariantKind = InvariantKind.constraint
    support_node_ids: NonEmptyIds                                 # resolved support refs (§14.2 fills it)
    anchor_ids: NonEmptyIds                                       # == sorted(∪ store.atom_anchor_ids(n) for n in support_node_ids) (§14.2)
    confidence: Prob = 0.7                                        # == round(mean(atom.confidence for n in support_node_ids), 4) (§14.2)


class Residual(HMSModel):
    text: str = Field(min_length=1)
    kind: ResidualKind
    support_node_ids: list[str] = Field(default_factory=list)
    anchor_ids: list[str] = Field(default_factory=list)           # sorted union of support anchors (empty allowed)
    edge_ids: list[str] = Field(default_factory=list)


class InterfaceEP(HMSModel):
    """High-level EP = interface to a low-level subgraph, never a summary. 高层 EP 是低层子图的接口，绝非摘要。"""
    interface_ep_id: str = ""
    scale: Literal["L2_meso", "L3_macro"] = "L2_meso"
    slots: Slots
    slot_support: dict[str, list[str]] = Field(default_factory=dict)   # slot name → support node ids (every slot cites ≥1)
    support_node_ids: Annotated[list[str], Field(min_length=2)]
    support_edge_ids: list[str] = Field(default_factory=list)
    boundary_edges: list[BoundaryEdge] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    residuals: list[Residual] = Field(default_factory=list)
    evidence_anchor_ids: NonEmptyIds                              # == ∪ anchors(support nodes) ∪ witnesses(support edges)
    coarsen_op_version: str = "rule-coarsen-v1"
    cohesion_score: Prob = 0.0
    atom_id: str | None = None                                    # its episode atom (index plane) == node_id
    scope: Scope
    valid: bool = True
    valid_to: UtcIso | None = None
    created_at: UtcIso | None = None
    # model_validator(after): every key in {'u','a','y','r','b','kappa'} present in slot_support with ≥1 id ⊂ support_node_ids when slot value non-empty


class ContextItem(Witnessed):
    item_id: str                                                  # §15.1 id rule (atom_id | '<atom_id>#kappa:<i>' | '<atom_id>#inv:<i>' | 'anc:<anchor_id>' | 'sig:<name>' | 'note:<n>')
    slot: Slot
    text: str = Field(min_length=1)
    atom_id: str | None = None
    memory_id: str | None = None
    ep_id: str | None = None
    score: float = 0.0
    time_anchor: UtcIso | None = None
    valid_from: UtcIso | None = None
    valid_to: UtcIso | None = None
    scale_level: Scale | None = None
    atom_type: AtomType | None = None
    visibility: Visibility = Visibility.private
    source: Literal["axis", "graph", "interface_ep", "system"] = "axis"
    hops: int = 0
    tokens: int = 0
    explain: dict[str, Any] = Field(default_factory=dict)         # {'reason', 'r', 'a', 'y', 'u', 'superseded_by_text', 'edge_ids', 'score_details', 'signal', ...}


class ContextIR(HMSModel):
    query: str
    goal: str | None = None
    as_of: UtcIso
    scope: Scope
    mode: CompileMode = CompileMode.auto
    route: Route = Route.recall
    facts: list[ContextItem] = Field(default_factory=list)
    episodes: list[ContextItem] = Field(default_factory=list)
    preferences: list[ContextItem] = Field(default_factory=list)
    constraints: list[ContextItem] = Field(default_factory=list)
    skills: list[ContextItem] = Field(default_factory=list)
    audit: list[ContextItem] = Field(default_factory=list)
    next_action: str | None = None
    budget_tokens: int = Field(default=1200, gt=0)
    used_tokens: int = 0
    retrieval: dict[str, Any] = Field(default_factory=dict)       # {'signals': [...], 'candidates': n, 'graph_hops': h, 'dropped': n, 'primary_axis': ladder level, 'rerank': None, 'route', 'plan'}
    unresolved_conflicts: list[str] = Field(default_factory=list) # edge ids among selected items
    # model_validator(after): item_ids pairwise disjoint across the six slots; item.slot == its slot name; every item ≥1 anchor (by type)
    def items(self) -> list[ContextItem]: ...                     # facts+episodes+preferences+constraints+skills+audit
    def anchor_ids(self) -> list[str]: ...                        # sorted union
    def to_text(self) -> str: ...                                 # compiler.render.render_card(self) — deterministic
    def to_pcir(self, *, require_witness: bool = True) -> "PCIR": ...   # compiler.pc_ir.lower (store-free; witness rows are embedded in audit items)


class ProofObligation(HMSModel):
    name: ObligationName
    status: ObligationStatus
    details: dict[str, Any] = Field(default_factory=dict)


class PCIR(HMSModel):
    payload: dict[str, Any]                                       # §15.4
    witness: dict[str, Any]                                       # {'items': {item_id: [anchor_ids]}, 'anchors': [Anchor-like dicts], '_all': sorted union}
    human_gloss: str                                              # == ContextIR.to_text()
    proof_obligations: list[ProofObligation]
    def ok(self) -> bool: ...                                     # no obligation violated


class StatePackage(HMSModel):
    package_id: str
    version: Literal["hms-elite/state-package/1"] = "hms-elite/state-package/1"
    created_at: UtcIso
    goal: str | None
    scope: Scope
    context: ContextIR
    pcir: PCIR
    atoms: list[MemoryAtom]                                       # every atom referenced by context items, eps, interface_eps (support nodes) and edges (endpoints)
    eps: list[EP]
    interface_eps: list[InterfaceEP]
    edges: list[GraphEdge]
    anchors: list[Anchor]                                         # witness closure: every anchor id referenced by context/atoms/eps/edges/ieps is present
    source_events: list[SourceEvent]                              # evidence closure: exactly one event per anchor (anchor.source_event_id), so a receiver can recreate anchors
    checkpoint_id: str | None = None
    # model_validator(after): witness closure holds; {a.source_event_id for a in anchors} == {e.event_id for e in source_events}
    def to_json(self, **kw) -> str: return self.model_dump_json(**kw)
    @classmethod
    def from_json(cls, s: str) -> "StatePackage": return cls.model_validate_json(s)


class Checkpoint(HMSModel):
    checkpoint_id: str = ""
    label: str | None = None
    anchor_id: str
    scope: Scope
    atom_state: dict[str, dict[str, str | None]]                  # atom_id → {'valid_to': ..., 'archived_reason': ...} for EVERY atom in scope
    edge_state: dict[str, bool]                                   # edge_id → valid, every edge in scope
    iep_state: dict[str, bool]                                    # interface_ep_id → valid
    ep_state: dict[str, str | None] = Field(default_factory=dict) # ep_id → eps.valid_to, every EP in scope
    history_seq: int
    snapshot_hash: str                                            # sha256(canonical_json({atom_state, edge_state, iep_state, ep_state}))
    created_at: UtcIso


class FeedbackEvent(HMSModel):
    feedback_id: str = ""
    target_id: str | None = None                                  # None only for state_missing without an existing target (§16.1)
    target_kind: Literal["atom", "edge", "interface_ep", "context_item", "anchor", "scope"]
    feedback: FeedbackType
    reason: str | None = None
    correction: str | dict[str, Any] | None = None
    actor_id: str | None = None
    anchor_id: str
    actions: list[dict[str, Any]] = Field(default_factory=list)   # [{'action': str, ...ids}]
    scope: Scope
    created_at: UtcIso


class Transition(HMSModel):
    transition_id: str = ""
    source_kind: TransitionKind
    source_id: str                                                # ep_id | interface_ep_id | edge_id
    scale: Literal["L1_micro", "L2_meso", "L3_macro"]
    s_t: dict[str, Any]
    a_t: str = Field(min_length=1)
    s_t1: dict[str, Any]
    r_t: Reward
    kappa_t: dict[str, Any]
    w_t: NonEmptyIds
    time_t: UtcIso
    time_t1: UtcIso
    scope: Scope
    created_at: UtcIso | None = None
    def to_tuple(self) -> tuple[dict, str, dict, str, dict, list[str]]: ...   # (s_t, a_t, s_t1, r_t, kappa_t, w_t)
    @classmethod
    def from_ep(cls, ep: EP, *, scope: Scope, permission_scope: str = "private") -> "Transition": ...              # §16.4 'ep' row (group A owns it; ingest C and ops G both call it)
    @classmethod
    def from_interface_ep(cls, iep: InterfaceEP, *, scope: Scope, time_t: str, time_t1: str, permission_scope: str = "private") -> "Transition": ...   # §16.4 'interface_ep' row


def validate_payload(object_type: str, payload: dict) -> dict: ...   # §4.3 normalizer — the SINGLE implementation (group A); ingest/events.py re-exports it
def kappa_t_of(*, constraints: list[str], visibility: str, permission_scope: str, provenance: dict) -> dict: ...   # §16.4 uniform κ_t shape: {'constraints', 'visibility', 'permission_scope', 'provenance'}
```

### §4.1 `MemoryAtom.to_public_dict` — the Mem0 row (canonical implementation in `models.py`, group A; `mvmi/query.py::atom_to_row(atom, *, score, score_details, anchor_ids)` is `return atom.to_public_dict(...)`; no lazy import anywhere)

```python
{'id': memory_id, 'memory': text, 'hash': hash, 'metadata': user_meta or None, 'score'?: float,
 'created_at': created_at, 'updated_at': updated_at,
 # promoted when present (Mem0 order):
 'user_id'?, 'agent_id'?, 'run_id'?, 'actor_id'?, 'role'? (atom_type == 'message' only), 'attributed_to'?, 'expiration_date'?, 'memory_type'?,
 # HMS promoted (always):
 'atom_id', 'version', 'atom_type', 'scale_level', 'anchor_ids', 'valid_from', 'valid_to', 'confidence',
 'hms': {'time_anchor', 'entity_ids', 'topic_tags', 'sentiment', 'subject', 'predicate', 'obs_type', 'source_kind', 'source_id',
         'source_ep_ids', 'extracted_by', 'visibility', 'archived_reason'},
 'score_details'?: {...}}                                          # search + explain only
```

`metadata` contains ONLY caller `user_meta` keys. Row key order is exactly the order above. `anchor_ids` defaults to `self.anchor_ids` (readers hydrate it, §8) unless the caller passes the undisputed list.

### §4.2 `MemoryAtom.content` shapes (by `source_kind`)

| source_kind | atom_type / scale_level | content keys |
|---|---|---|
| message | message / L1_micro | `text, role, actor_id, dropped_parts` |
| observation (fact/preference_event/habit/constraint) | state / state | `text, obs_id, obs_type, subject, predicate, objects, relation, date_observed` |
| observation (temporal_fact) | event / event | same + `event_time` (== time_anchor) |
| observation (relation='comparison') additionally | relation / relation | `text, subject, predicate, objects, relation` (a second atom, `source_id` = same obs_id, `text` = `"Comparison: " + text`, `source_kind='observation'`) — only when `ingest.emit_relation_atoms` is True (default **True**) |
| entity | entity / entity | `text (== canonical_text), entity_id, entity_type, aliases: [str], mentions: int` — materialized by consolidate step (0) (§16.5), `source_id = entity_id`, `subject = canonical_text`, `predicate = 'is_entity'`, `anchor_ids = sorted(∪ anchors of the valid non-entity atoms linked to the entity)`, `scope = Scope(user_id=entities.user_id, agent_id=entities.agent_id)` (the entity identity scope; `run_id` None) |
| event | event / event | `text, event_id, object_type, payload` (normalized payload) |
| ep | episode / L1_micro | `text, ep_id, slots` (Slots.model_dump()) |
| interface_ep | episode / L2_meso or L3_macro | `text, interface_ep_id, slots, cohesion_score, invariants: [text], residuals: [text]` |
| skill | state / L3_macro | `text, kind: 'skill', invariant, source_interface_ep_ids, support_node_ids` |
| procedural | episode / L1_micro | `text (markdown), memory_type: 'procedural_memory'` |
| feedback (state_missing / next_action_wrong / render_* created atoms) | state / state | same as observation shape; `obs_type ∈ {constraint, fact, preference_event}`; `source_kind='feedback'`, `source_id=feedback_id`, `obs_id` key carries the feedback_id |

### §4.3 `SourceEvent.payload` normalized shapes (`hms_elite/models.py::validate_payload`, group A — the only normalizer; `ingest/events.py` re-exports it; `payload_hash` and every anchor identity depend on exactly these dicts)

| object_type | normalized payload (keys always present, in this order) |
|---|---|
| message | `{"text": str, "dropped_parts": int}` + optional `"parts": [{"kind": "image", "source": "data"|"url", "sha256": str, "mime": str|None, "url": str|None, "detail": str|None}]` and optional `"described_parts": [{"index": int, "sha256": str, "text": str, "described_by": str|None}]` — both keys exist only when non-empty (text-only payloads keep their historical shape and hash); a part `url` starting with `data:` → `VALIDATION_005` (the bytes are identified by `sha256` only) |
| tool_call | `{"tool": str, "args": dict, "result": str|None, "status": "ok"|"error"|None, "target": str|None, "intent": str|None, "constraints": [str]}` (`arguments` accepted as alias of `args`; non-str result → `json.dumps(result, ensure_ascii=False)`) |
| file_diff | `{"path": str, "patch": str, "summary": str|None, "intent": str|None, "commit": str|None, "constraints": [str]}` |
| test_log | `{"suite": str, "status": "passed"|"failed", "passed": int|None, "failed": int|None, "failures": [{"name": str, "message": str|None}], "log": str|None, "related_paths": [str]}` — `status` derived when absent: `failed > 0` or non-empty failures → `failed`, else `passed`; string failures → `{"name": s, "message": None}`; `failed_tests` accepted as alias of `failures` |
| sensor_trace | `{"sensor": str, "series_ref": str|None, "summary": str|None, "samples": [{"t": str (UTC ISO via to_utc_iso), "v": Any}]}` — routed to `{'structured'}`, produces an `event` atom only (never an EP, never an action); `default_raw_ref → f"sensor://{sensor}"`; role defaults to `tool` |
| feedback | `{"target_id": str, "feedback": str, "reason": str|None, "correction": Any}` |
| checkpoint | `{"label": str|None}` |
| rollback | `{"checkpoint_id": str, "label": str|None, "reason": str|None}` |
| system | `{"op": "update"|"delete"|"delete_all"|"import"|"revoke"|"purge", "memory_id": str|None, "atom_id": str|None, "text": str|None, "metadata": dict|None, "expiration_date": str|None, "package_id": str|None, "anchor_id": str|None}` (`package_id` for `import`, `anchor_id` for `revoke`) |
| scheduler | `{"phases": [str]}` |

Missing required key or wrong type → `ValidationError(VALIDATION_005, details={"object_type", "field", "index"?})`.

---

## §5 Configuration (`hms_elite/configs/base.py`, `configs/registry.py`, group A)

### §5.1 Config dict shape (what `HMS.from_config(dict)` accepts)

```python
{
  # Mem0-compatible keys (same names, same nesting)
  "vector_store": {"provider": "sqlite" | "memory" | "milvus", "config": {...}},          # index envelope (Mem0 name kept)
  "llm":          {"provider": "rule" | "scripted" | "openai" | "anthropic" | "deepseek" | "ollama", "config": {...}},
  "embedder":     {"provider": "hash" | "openai" | "ollama" | "fastembed", "config": {...}},
  "history_db_path": "/path/hms.db" | None,                                     # → store.path (warn if store.path also set and differs)
  "reranker": None | {"provider": "overlap" | "llm", "config": {...}},          # ProviderEnvelope kind 'reranker'; None = unconfigured (search(rerank=True) then uses a default OverlapReranker); unknown provider → CONFIG_002
  "version": "v1.1" | "v2" | "lite-1",                                          # drives nothing
  "custom_instructions": str | None,
  # HMS keys
  "store":     {"path": str, "busy_timeout_ms": 5000, "journal_mode": "wal" | "delete", "checkpoint_on_close": True},
  "ingest":    {"revision_mode": "inline" | "deferred" | "off", "neighbourhood_k": 10, "last_events_k": 10,
                "emit_relation_atoms": True, "closure_max_gap_hours": 24,
                "entity_extractor": "rule" | "spacy", "entity_extractor_config": {},   # §10.5; unknown name → CONFIG_002 (details kind='entity_extractor')
                "extraction_mode": "single" | "staged", "extraction_output": "lines" | "full" | "compact",
                "segment_max_chars": 400, "verdict_min_similarity": 0.5, "zh_rules": False},   # zh_rules: Chinese statement rules, §9.6; extraction_output default 'lines' since 2026-09-18 (§6.6, changelog row 108); 'staged' needs an explicit 'full' | 'compact' (the after-validator rejects lines + staged → CONFIG_001)
  "scoring":   {"w_semantic": 1.0, "w_keyword": 0.5, "w_entity": 0.5, "w_recency": 0.25, "w_support": 0.25,
                "half_life_days": 30.0, "max_candidates": None, "identifier_weight": 2.0},   # max_candidates: deprecated, accepted for compatibility, truncates nothing; the default is the None sentinel (a model_dump round-trip stays silent), a non-None value emits DeprecationWarning
  "retrieval": {"k_semantic": None, "k_keyword": None, "k_entity": None, "k_recency": None, "pool_max_factor": 4,
                "hard_gate_pool": True, "safety_limit": 50000},   # bounded multi-channel recall (RETRIEVAL_DIAGNOSIS §3); None → max(4·top_k, 60) / max(2·top_k, 30)
  "text":      {"cjk_tokens": "bigram"},   # "bigram" | "none": CJK tokenizer mode on both the index and the query side (§9.6; default = the historical bigrams)
  "graph":     {"min_cohesion": 0.3, "coarsen_min_nodes": 2, "coarsen_max_nodes": 12, "coarsen_gap_hours": 48,
                "temporal_window_hours": 24, "coarsen_policy": "rule" | "llm", "causal_proposer": "rule" | "llm", "coarsen_l3": True},
  "compiler":  {"budget_tokens": 1200, "candidate_k": 50, "graph_hops": 1, "require_witness": True,
                "slot_shares": {"facts": 0.30, "episodes": 0.25, "constraints": 0.15, "preferences": 0.10, "skills": 0.05, "audit": 0.15},
                "min_constraints": 3, "max_why_nodes": 60, "default_mode": "auto", "selection": "shares",   # default_mode is READ by compile()/StateCompiler.compile when mode=None (§15.2 step 1)
                "context_session_budget_tokens": 1200, "context_rows_top_k": 20, "context_rows_header": "Recent relevant memories:", "context_timestamps": True},   # context_* → HMS.context() (§6.7)
  "scheduler": {"phases": ["orient", "collect", "consolidate", "prune"], "staleness_tau": 0.8, "confidence_min": 0.4,
                "use_recency_half_life_days": 30.0, "orient_window_days": 30, "decay_rate": 0.1, "compile_state": True},
  "retention":  {"enabled": False, "older_than_days": 90, "run_in_prune": False},   # §16.6 escape hatch; all off by default
  "visibility_default": "private",
}
```

RENAMED (remap + `DeprecationWarning(stacklevel=2)`): `custom_fact_extraction_prompt → custom_instructions`, `custom_update_memory_prompt → custom_instructions`, `index → vector_store`, `db_path → store.path`. REMOVED (`ConfigError(CONFIG_003)` with hint): `graph_store` ("the support graph lives in the same SQLite file; remove the key"), `enable_graph` ("the graph is always on"). Unknown keys anywhere → `ConfigError(CONFIG_001)` whose message names the key and a `difflib.get_close_matches` hint. A stock Mem0 dict fails ONLY on `vector_store.provider` not in `{sqlite, memory, milvus}` (`CONFIG_002`, suggestion `"HMS Elite ships 'sqlite' and 'memory' index providers ('milvus' via the optional [milvus] extra); use {'provider': 'sqlite'}"`). **Where `CONFIG_002`/`CONFIG_003` are raised**: pydantic v2 converts any `ValueError` raised inside a validator into `pydantic.ValidationError`, so these two codes are raised by plain-Python pre-checks in `HMSConfig.from_dict` (§5.3) BEFORE `HMSConfig(**data)` is called — never from inside a validator. The `mode='before'` validator only remaps RENAMED keys (idempotently) so that direct `HMSConfig(**d)` construction still works; direct construction with a REMOVED key or an unknown provider surfaces as `pydantic.ValidationError` (documented; the public entry points `HMS(dict)`, `HMS.from_config`, `HMSConfig.from_dict` always go through the pre-checks).

### §5.2 Classes

```python
class ProviderEnvelope(HMSModel):
    kind: ClassVar[str]
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    @field_validator("config", mode="before")
    def _none_to_dict(cls, v): return {} if v is None else v
    @model_validator(mode="after")
    def _known_provider(self):   # provider ∈ REGISTRY[kind] else ValueError(f"unknown {kind} provider {provider!r}") — surfaces as pydantic.ValidationError on direct construction; from_dict pre-checks it as CONFIG_002

class LlmConfig(ProviderEnvelope):      kind = "llm";      provider: str = "rule"; enable_vision: bool = False; vision_details: str | None = None   # Mem0 places both inside `config`: a before-validator lifts them to the envelope (explicit envelope keys win)
class EmbedderConfig(ProviderEnvelope): kind = "embedder"; provider: str = "hash"
class IndexConfig(ProviderEnvelope):    kind = "index";    provider: str = "sqlite"     # dict key is 'vector_store'
class RerankerEnvelope(ProviderEnvelope): kind = "reranker"; provider: str = "overlap"   # REGISTRY['reranker'] = {overlap, llm}; unknown provider → CONFIG_002 (details['kind'] == 'reranker')

class StoreConfig(HMSModel):     path: str = Field(default_factory=default_db_path); busy_timeout_ms: int = 5000; journal_mode: Literal["wal", "delete"] = "wal"; checkpoint_on_close: bool = True   # False: a file DB keeps its WAL on disk at close() (durable, recovered on the next open) when Python >= 3.12 offers SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE; ignored otherwise and for ':memory:'
class IngestConfig(HMSModel):    revision_mode: Literal["inline", "deferred", "off"] = "inline"; neighbourhood_k: int = 10; last_events_k: int = 10; emit_relation_atoms: bool = True; closure_max_gap_hours: float = 24.0; entity_extractor: str = "rule"; entity_extractor_config: dict[str, Any] = {}; extraction_mode: Literal["single", "staged"] = "single"; extraction_output: Literal["full", "compact", "lines"] = "lines"; segment_max_chars: int = Field(default=400, ge=1); verdict_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)   # verdict_min_similarity → the §12.6 step-6 verdict guard (containment bar for NONE / REVISE, 0.0 disables; §6.3 revision rules); extraction_output 'lines' (the default since 2026-09-18, changelog row 108, §6.6 default paragraph; before that 'full', which 'compact' displaced for part of 2026-09-11 and was reverted that evening under the §6.5 accuracy rule, row 105) → §6.6 (tab-separated OBS / EP / DEC text, no json mode, single mode only — the after-validator rejects 'lines' + 'staged' with a bilingual ValueError, so extraction_mode='staged' now needs an explicit 'full' | 'compact'), 'full' → the byte-identical full contract, 'compact' → §6.5 (the LLM emits COMPACT_OBSERVATION_KEYS only); entity_extractor ∈ REGISTRY['entity_extractor'] (§10.5); from_dict pre-checks CONFIG_002, direct construction → pydantic.ValidationError; None config → {}; zh_rules: bool = False → §9.6 (Chinese statement rules of the compact / lines derivation; forwarded to the rule entity extractor)
class ScoringConfig(HMSModel):   w_semantic: float = 1.0; w_keyword: float = 0.5; w_entity: float = 0.5; w_recency: float = 0.25; w_support: float = 0.25; half_life_days: float = 30.0; max_candidates: int | None = None; identifier_weight: float = 2.0   # max_candidates is a deprecated compatibility field (the newest-first truncation is gone); None sentinel, only a non-None value warns
class RetrievalConfig(HMSModel): k_semantic: int | None = None; k_keyword: int | None = None; k_entity: int | None = None; k_recency: int | None = None; pool_max_factor: int = 4; hard_gate_pool: bool = True; safety_limit: int = 50000
    def k_for(self, top_k: int) -> dict[str, int]   # {'semantic': k_semantic or max(4·top_k, 60), 'keyword' | 'entity' | 'recency': k_x or max(2·top_k, 30), 'pool': pool_max_factor · semantic}; HMSConfig.retrieval
class TextConfig(HMSModel):      cjk_tokens: Literal["bigram", "none"] = "bigram"   # §9.6: CJK tokenizer mode of utils.text.tokenize / content_tokens on both sides; HMSConfig.text (after retrieval)
class GraphConfig(HMSModel):     min_cohesion: float = 0.3; coarsen_min_nodes: int = 2; coarsen_max_nodes: int = 12; coarsen_gap_hours: float = 48.0; temporal_window_hours: float = 24.0; coarsen_policy: Literal["rule", "llm"] = "rule"; causal_proposer: Literal["rule", "llm"] = "rule"; coarsen_l3: bool = True
class CompilerConfig(HMSModel):  budget_tokens: int = 1200; candidate_k: int = 50; graph_hops: int = 1; require_witness: bool = True; slot_shares: dict[str, float] = {...as §5.1}; min_constraints: int = 3; max_why_nodes: int = 60; default_mode: CompileMode = "auto"; selection: Literal["shares", "coverage"] = "shares"; context_session_budget_tokens: int = Field(default=1200, ge=1); context_rows_top_k: int = Field(default=20, ge=1); context_rows_header: str = "Recent relevant memories:"; context_timestamps: bool = True   # context_* → HMS.context() (§6.7): session-card budget, search rows before dedup, the header line between the two parts, '[YYYY-MM-DD HH:MM]' stamps on / off
class SchedulerConfig(HMSModel): phases: list[Phase] = [orient, collect, consolidate, prune]; staleness_tau: float = 0.8; confidence_min: float = 0.4; use_recency_half_life_days: float = 30.0; orient_window_days: int = 30; decay_rate: float = 0.1; compile_state: bool = True
class RetentionConfig(HMSModel): enabled: bool = False; older_than_days: int = Field(default=90, ge=0); run_in_prune: bool = False   # §16.6: enabled=False → HMS.purge(dry_run=False) is ConfigError(CONFIG_001); run_in_prune needs enabled too

class HMSConfig(HMSModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True, use_enum_values=True)
    vector_store: IndexConfig = Field(default_factory=IndexConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    history_db_path: str | None = None
    reranker: RerankerEnvelope | None = None
    version: Literal["v1.1", "v2", "lite-1"] = "lite-1"
    custom_instructions: str | None = None
    store: StoreConfig = Field(default_factory=StoreConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    compiler: CompilerConfig = Field(default_factory=CompilerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    visibility_default: Visibility = Visibility.private
    @model_validator(mode="before")
    @classmethod
    def _mem0_keys(cls, data): return normalize_mem0_keys(data, strict=False)   # RENAMED remap only (idempotent); never mutates caller dict (deepcopy)
    @model_validator(mode="after")
    def _history_path(self): ...       # history_db_path → store.path via object.__setattr__ (frozen); warn if both set and differ
    def db_path(self) -> str: return self.store.path

def default_db_path() -> str: ...      # os.path.join(os.environ.get("HMS_DIR") or os.path.expanduser("~/.hms_elite"), "hms.db"); no makedirs at import
def normalize_mem0_keys(data: dict, *, strict: bool) -> dict   # deepcopy; RENAMED keys remapped (DeprecationWarning once per key; dotted targets set nested keys); strict=True additionally raises ConfigError(CONFIG_003, details={'key', 'hint'}) for REMOVED keys (strict=False leaves them in place for pydantic to reject as unknown)
def check_providers(data: dict) -> None                        # for kind, key in (('index','vector_store'), ('llm','llm'), ('embedder','embedder')): envelope = data.get(key); provider = envelope.get('provider') if dict else None; provider not in REGISTRY[kind] (and not None) → ConfigError(CONFIG_002, details={'kind','provider','supported'}, suggestion=...)
```

Per-provider config classes (pydantic, `extra="forbid"`, plus a `mode="before"` validator that raises `ConfigError(CONFIG_001, "Extra fields not allowed: X. Did you mean 'Y'?")` via `difflib.get_close_matches`), living in their provider modules (group B) but with fields fixed here:

```python
class BaseLlmConfig(HMSModel):      model: str | None = None; temperature: float = 0.1; api_key: str | None = None; max_tokens: int = 2000; top_p: float = 0.1; top_k: int = 1; http_client_proxies: dict | str | None = None
class RuleLlmConfig(BaseLlmConfig): language: Literal["auto", "en", "zh"] = "auto"; strict: bool = False        # strict=True: sentences matching no rule are never emitted as low-confidence facts
class ScriptedLlmConfig(BaseLlmConfig): responses: list[Any] = []; loop: bool = False; raise_when_empty: bool = True
class OpenAILlmConfig(BaseLlmConfig):   model: str | None = "gpt-5-mini"; openai_base_url: str | None = None
class AnthropicLlmConfig(BaseLlmConfig): model: str | None = "claude-sonnet-4-6"; anthropic_base_url: str | None = None
class BaseEmbedderConfig(HMSModel):  model: str | None = None; embedding_dims: int | None = None; api_key: str | None = None
class HashEmbedderConfig(BaseEmbedderConfig): embedding_dims: int = 64; ngram_min: int = 2; ngram_max: int = 3
class OpenAIEmbedderConfig(BaseEmbedderConfig): model: str | None = "text-embedding-3-small"; embedding_dims: int = 1536; openai_base_url: str | None = None
class DeepSeekLlmConfig(BaseLlmConfig):  model: str | None = "deepseek-chat"; deepseek_base_url: str | None = None; thinking: Literal["enabled", "disabled"] | None = None   # OpenAI-compatible endpoint through the openai SDK (extra `openai`)
class OllamaLlmConfig(BaseLlmConfig):    model: str | None = "llama3.1"; ollama_base_url: str | None = None; timeout: float = 120.0   # stdlib urllib; the `ollama` extra is empty
class OllamaEmbedderConfig(BaseEmbedderConfig): model: str | None = "nomic-embed-text"; ollama_base_url: str | None = None; timeout: float = 120.0   # dims inferred on the first call and cached
class FastEmbedEmbedderConfig(BaseEmbedderConfig): model: str | None = "BAAI/bge-small-en-v1.5"; embedding_dims: int = 384; cache_dir: str | None = None; threads: int | None = None   # local inference (extra `fastembed`)
class BaseIndexConfig(HMSModel):     pass
class SQLiteIndexConfig(BaseIndexConfig): use_numpy: Literal["auto", "never"] = "auto"
class MemoryIndexConfig(BaseIndexConfig): pass
class MilvusIndexConfig(BaseIndexConfig): ...   # §10.3 (extra `milvus`)
```

### §5.3 Public functions

- `HMSConfig.from_dict(d: dict) -> HMSConfig` — validate a dict | 校验字典. In: dict | 字典. Out: HMSConfig | 配置. Effect: none | 无. Notes (normative order): (1) `d` not a dict → `ConfigError(CONFIG_001)`; (2) `data = normalize_mem0_keys(d, strict=True)` (→ `CONFIG_003` for REMOVED keys, `DeprecationWarning` for RENAMED); (3) `check_providers(data)` (→ `CONFIG_002`); (4) `HMSConfig(**data)`, wrapping `pydantic.ValidationError` as `ConfigError(CONFIG_001, details={'errors': e.errors()})` — except that if any entry of `e.errors()` has `ctx['error']` that is an `HMSError` instance, that instance is re-raised unchanged; the message of the `CONFIG_001` names the offending key (`loc`) and adds a `difflib.get_close_matches` hint. `HMS.from_config` and `HMS.__init__(dict)` call this; the full pydantic error is logged before re-raising.

### §5.4 Registry (`configs/registry.py`)

```python
REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "llm": {"rule": ("hms_elite.llms.rule.RuleLLM", "hms_elite.llms.rule.RuleLlmConfig"),
            "scripted": ("hms_elite.llms.scripted.ScriptedLLM", "hms_elite.llms.scripted.ScriptedLlmConfig"),
            "openai": ("hms_elite.llms.openai.OpenAILLM", "hms_elite.llms.openai.OpenAILlmConfig"),
            "anthropic": ("hms_elite.llms.anthropic.AnthropicLLM", "hms_elite.llms.anthropic.AnthropicLlmConfig"),
            "deepseek": ("hms_elite.llms.deepseek.DeepSeekLLM", "hms_elite.llms.deepseek.DeepSeekLlmConfig"),
            "ollama": ("hms_elite.llms.ollama.OllamaLLM", "hms_elite.llms.ollama.OllamaLlmConfig")},
    "embedder": {"hash": ("hms_elite.embeddings.hash.HashEmbedder", "hms_elite.embeddings.hash.HashEmbedderConfig"),
                 "openai": ("hms_elite.embeddings.openai.OpenAIEmbedder", "hms_elite.embeddings.openai.OpenAIEmbedderConfig"),
                 "ollama": ("hms_elite.embeddings.ollama.OllamaEmbedder", "hms_elite.embeddings.ollama.OllamaEmbedderConfig"),
                 "fastembed": ("hms_elite.embeddings.fastembed.FastEmbedEmbedder", "hms_elite.embeddings.fastembed.FastEmbedEmbedderConfig")},
    "index": {"sqlite": ("hms_elite.index.sqlite.SQLiteIndex", "hms_elite.index.sqlite.SQLiteIndexConfig"),
              "memory": ("hms_elite.index.memory.MemoryIndex", "hms_elite.index.memory.MemoryIndexConfig"),
              "milvus": ("hms_elite.index.milvus.MilvusIndex", "hms_elite.index.milvus.MilvusIndexConfig")},   # optional `milvus` extra
    "reranker": {"overlap": ("hms_elite.reranker.overlap.OverlapReranker", "hms_elite.reranker.overlap.OverlapRerankerConfig"),
                 "llm": ("hms_elite.reranker.llm.LLMReranker", "hms_elite.reranker.llm.LLMRerankerConfig")},
    "entity_extractor": {"rule": ("hms_elite.entities.rule.RuleEntityExtractor", "hms_elite.entities.rule.RuleEntityExtractorConfig"),
                         "spacy": ("hms_elite.entities.spacy.SpacyEntityExtractor", "hms_elite.entities.spacy.SpacyEntityExtractorConfig")},   # §10.5; selected by IngestConfig.entity_extractor (a name, not an envelope)
}
EXTRA_OF_PROVIDER = {"openai": "openai", "anthropic": "anthropic", "milvus": "milvus", "deepseek": "openai", "ollama": "ollama", "fastembed": "fastembed", "spacy": "spacy"}   # pip extra hint per provider name (deepseek rides on the openai SDK; ollama is stdlib-only, its extra is empty)
def supported(kind: str) -> list[str]: return sorted(REGISTRY[kind])
```

Strings only; no imports of provider modules. `utils/factory.register_provider` mutates `REGISTRY[kind]` in place so envelope validation and factories share one truth.

---

## §6 Prompts and LLM task contracts (`hms_elite/configs/prompts.py`, group A)

Module docstring: `"""Prompt constants with HMS_TASK markers, user-prompt builders and their parsers.\n带 HMS_TASK 标记的提示词常量、用户提示词构造器及其解析器。"""`

### §6.1 Shape rules

- Every system prompt is a module constant `<TASK>_PROMPT` whose **first line is exactly** `HMS_TASK: <task>`; sections in order `# ROLE`, `# INPUTS` (one `## ` heading per input in builder order), `# GUIDELINES` (Bad:/Good: pairs, two-clock rule), `# EXAMPLES`, `# OUTPUT FORMAT` (the verbatim JSON block below + `Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers.`) — **except `procedural_summary`**, whose `# OUTPUT FORMAT` block contains the exact markdown skeleton of §6.3 (the seven headings/labels in order) followed by the line `Return ONLY the markdown above. No JSON, no wrappers.`; it is the one task without a `ROOT_KEY` entry and is produced through `generate_text`, never `generate_json` (test 9 asserts the skeleton headings appear inside `PROCEDURAL_SUMMARY_PROMPT`'s OUTPUT FORMAT section).
- Builders return the user prompt: `## <Section>` headings in the documented order; structured inputs are fenced ` ```json ... ``` ` blocks (`json.dumps(obj, ensure_ascii=False, indent=None, sort_keys=True)`); free text sections are plain; the prompt ends with `# Output:`.
- Real ids never reach a model: existing memories are `'0'..'n'`, events `idx`, EP refs `e0..`, node refs `n0..`, witness refs `w0..`, pair ids `'0'..`; parsers map refs back; an item citing an unknown ref is dropped with `logger.warning` ("witness first").
- `TASKS = ("observation_extraction", "observation_extraction_compact", "ep_extraction", "revision_decision", "edge_proposal", "coarsen", "procedural_summary")`; `PROMPTS: dict[str, str]`; `PAST_MESSAGE_TRUNCATION_LIMIT = 300`; `COMPOSITE_TASKS = ("memory_extraction", "memory_extraction_compact", "memory_extraction_lines")` with `COMPOSITE_PROMPTS` (multi-root replies, outside `ROOT_KEY`); `COMPACT_OBSERVATION_KEYS` (§6.5); `LINE_TYPES` / `TYPE_LETTERS` (§6.6).

### §6.2 Functions

- `task_of(system_prompt: str) -> str | None` — read the `HMS_TASK:` marker from the first line | 读取首行任务标记. In: system prompt | 系统提示. Out: task name or None | 任务名或 None. Effect: none | 无.
- `parse_sections(user_prompt: str) -> dict[str, Any]` — split a built user prompt into `{section_title: parsed_json | text}` | 解析用户提示的分节. In: builder output | 构造器输出. Out: dict keyed by `## ` titles; fenced JSON parsed, else stripped text | 字典. Effect: none | 无. Notes: `# Output:` trailer dropped; every builder below has `parse_sections(build_x(...))` as its inverse (tested).
- `build_observation_prompt(*, existing_memories: list[dict], new_messages: list[dict], last_messages: list[dict], observation_date: str, current_date: str, custom_instructions: str | None = None) -> str` — sections `## Last k Messages` (plain `role: text` lines, each truncated to 300 chars), `## Existing Memories` (json `[{"id": "0", "text": ...}]`), `## New Messages` (json `[{"idx": 0, "role": "user", "text": ..., "timestamp": ...}]`), `## Observation Date` (`YYYY-MM-DD`), `## Current Date`, `## Custom Instructions` (only when given).
- `build_ep_prompt(*, events: list[dict], known_constraints: list[dict], recent_eps: list[dict], observation_date: str) -> str` — `## Events` (json `[{"idx", "role", "object_type", "text", "timestamp"}]`), `## Known Constraints` (json `[{"id": "c0", "text"}]`), `## Recent EPs` (json `[{"id": "e0", "u", "a", "y", "r"}]`), `## Observation Date`.
- `build_revision_prompt(*, new_items: list[dict], candidates: list[dict], current_date: str) -> str` — `## New Memories` (json `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed"}]`), `## Candidate Existing Memories` (json `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed", "valid_from"}]`), `## Current Date`.
- `build_edge_prompt(*, nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> str` — `## Nodes` (json `[{"ref": "n0", "kind": "ep"|"interface_ep"|"observation"|"event"|"skill", "u", "b", "a", "y", "r", "subject", "predicate", "objects": [...], "time", "entities": [...], "topics": [...], "anchor_refs": ["w0"]}]` — EP/InterfaceEP nodes carry `u,b,a,y,r` (slots) with `subject/predicate/objects` null/[]; observation/event/skill nodes carry `subject/predicate/objects` (from the atom) with `u,b,y` = "" , `a` = atom text, `r` = "unknown"; every key is always present so the rule provider reads one shape), `## Candidate Pairs` (json `[["n0","n1"], ...]`), `## Witnesses` (json `[{"ref": "w0", "node_ref": "n0", "source_ref", "time"}]`), `## Allowed Edge Types` (plain: six lines `type: definition`). The `# INPUTS` section of `EDGE_PROPOSAL_PROMPT` lists exactly these node keys.
- `build_coarsen_prompt(*, support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str) -> str` — `## Support Nodes` (json `[{"ref": "n0", "kind", "u", "b", "a", "y", "r", "constraints": [...], "time", "anchor_refs"}]`), `## Internal Edges` (json `[{"ref": "x0", "src": "n0", "dst": "n1", "type"}]`), `## Boundary Edges` (json `[{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type"}]`), `## Target Scale`.
- `build_procedural_messages(messages: list[dict], prompt: str | None) -> list[dict]` — `[system: prompt or PROCEDURAL_SUMMARY_PROMPT, *messages(text only), user: "Create procedural memory of the above conversation."]` (Mem0 `_create_procedural_memory` shape).

### §6.3 Exact JSON contracts (OUTPUT FORMAT blocks, verbatim)

**`HMS_TASK: observation_extraction`** → key `observations`
```
{
  "observations": [
    {"id": "0", "event_idx": 0,
     "type": "preference_event",
     "subject": "user", "predicate": "likes", "objects": ["oat-milk lattes in the morning"],
     "relation": "single_fact",
     "text": "User likes oat-milk lattes in the morning.",
     "date_observed": "2026-04-02", "valid_from": null, "valid_to": null,
     "confidence": 0.8, "attributed_to": "user", "speaker": null,
     "sentiment": "positive",
     "topic_tags": ["coffee", "preference"], "entity_mentions": [],
     "supersedes_hint_ids": []}
  ]
}
```
Field rules: `id` sequential strings from `"0"`; `event_idx` MUST index `## New Messages` (else the item is discarded); `type ∈ fact|preference_event|habit|constraint|temporal_fact`; `subject ∈ user|assistant|project|<entity text>`; `predicate` lowercase snake verb; `relation ∈ single_fact|comparison|habit|supersession|constraint`; `text` 5–80 words, self-contained, third person; `date_observed` `YYYY-MM-DD` resolved ONLY against `## Observation Date`; `confidence ∈ [0,1]`; `attributed_to ∈ user|assistant` (the conversation side, never a person); `speaker` the named person of a multi-speaker transcript, else `null` (never a side word); `sentiment ∈ positive|neutral|negative|mixed|unknown`; `supersedes_hint_ids` ⊂ ids of `## Existing Memories`. Nothing → `{"observations": []}`.

**`HMS_TASK: ep_extraction`** → key `eps`
```
{
  "eps": [
    {"id": "0", "event_idx": 1, "scale": "L1_micro",
     "u": "billing_e2e failed after the retry patch",
     "b": "Do not modify the subscription state machine.",
     "a": "patch retry.ts to add exponential backoff",
     "y": "billing_e2e failed: subscription_state_regression",
     "r": "negative",
     "kappa": {"constraints": ["Do not modify the subscription state machine."], "visibility": "team"},
     "time_anchor": "2026-06-15T10:04:00+00:00",
     "entity_mentions": ["retry.ts", "billing_e2e"], "topic_tags": ["engineering", "billing"],
     "uses_ep_ids": ["e1"], "confidence": 0.7}
  ]
}
```
Rules: emit an EP only when a trigger AND an action AND (an outcome OR a constraint) are present; `r ∈ positive|negative|neutral|unknown`; `uses_ep_ids` ⊂ `## Recent EPs` ids; `event_idx` MUST index `## Events`. Nothing → `{"eps": []}`.

**`HMS_TASK: revision_decision`** → key `decisions`
```
{"decisions": [{"new_id": "0", "decision": "REVISE", "target_id": "2", "reason": "supersession cue: 'switched to'"}]}
```
Rules: exactly one decision per `## New Memories` id; `decision ∈ ADD|REVISE|CONFLICT|NONE`; `target_id` REQUIRED for REVISE/CONFLICT/NONE and MUST be a `## Candidate Existing Memories` id, `null` for ADD. ADD = compatible/additive; REVISE = new supersedes target; CONFLICT = incompatible, neither clearly newer/truer; NONE = duplicate of target. Library-side verdict guard (§12.6 step 6, both contracts and both `extraction_mode`s, since 2026-09-11; head-removal form since changelog row 104): every NONE / REVISE / CONFLICT asserts a relation between two statements about ONE subject, so a verdict whose two atoms carry different subjects (case-insensitive, stripped; both known) is stored as ADD — except a REVISE corroborated by the observation naming the target in `supersedes_hint_ids` or by an equal subject + predicate, which always passes (checked first); a CONFLICT with agreeing subjects passes (it asserts a different value, so its texts are expected to differ); a NONE or an uncorroborated REVISE passes only when `ingest.extraction.verdict_similarity(new.text, target.text, new_subject=…, old_subject=…, new_predicate=…, old_predicate=…)` — containment |A ∩ B| / min(|A|, |B|) of the stemmed content tokens after the head is removed (each side's own subject; the predicate when both carry the same one) — is at or above `ingest.verdict_min_similarity` (0.5) AND the two share at least two tokens or the shorter statement is contained entirely (one shared token of two is a coincidence, not a restatement); anything else is stored as ADD (DEBUG log; an `IngestResult.downgraded` entry with `similarity`, `shared`, `subjects_differ`); 0.0 disables the guard.

**`HMS_TASK: edge_proposal`** → key `edges`
```
{"edges": [{"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal",
            "label": "retry patch caused the billing regression",
            "witness_refs": ["w1"], "confidence": 0.8}]}
```
Rules: `edge_type` MUST be one of `temporal|causal|revision|conflict|use|distill` (anything else discarded); `witness_refs` REQUIRED, a non-empty list of strings ⊂ `## Witnesses` refs belonging to src or dst (else discarded — a scalar / object / nested list is dropped with a warning, never raised); `confidence` unparsable or non-finite (`NaN` / `Infinity`) → 0.7; no self edges; only pairs listed in `## Candidate Pairs`; `{"edges": []}` when nothing is justified.

**`HMS_TASK: coarsen`** → key `interface_ep`
```
{"interface_ep": {
   "slots": {"u": "...", "b": "...", "a": "...", "y": "...", "r": "positive",
             "kappa": {"constraints": ["..."], "visibility": "private"}},
   "slot_support": {"u": ["n0"], "b": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "kappa": ["n0", "n1"]},
   "invariants": [{"text": "Do not modify the subscription state machine.", "kind": "constraint", "support_refs": ["n0", "n1"]}],
   "residuals": [{"text": "outcome of n2 unknown", "kind": "missing_outcome", "support_refs": ["n2"]}]}}
```
Rules: every non-empty slot MUST cite ≥1 `## Support Nodes` ref in `slot_support`; `kind` (invariants) ∈ `constraint|failed_path|stable_fact`; `kind` (residuals) ∈ `unresolved_conflict|missing_outcome|low_cohesion|open_question`; invariants/residuals citing unknown refs are discarded; do not invent nodes.

**`HMS_TASK: procedural_summary`** → markdown (no JSON): Mem0 `PROCEDURAL_MEMORY_SYSTEM_PROMPT` structure — `## Summary of the agent's execution history`, `**Task Objective**:`, `**Progress Status**:`, numbered steps each with `**Agent Action**:` / `**Action Result**:`, `**Key Findings**:`, `**Errors & Challenges**:`, `**Current Context**:`. Its `# OUTPUT FORMAT` block is this skeleton verbatim (headings/labels in this order, placeholders `...`) plus the line `Return ONLY the markdown above. No JSON, no wrappers.` Empty output → `ValueError("Procedural memory generation failed")` (Mem0 parity). Passing `prompt=` on the procedural branch REPLACES this system prompt.

Context rendering (Work State Card, handoff gloss) has NO prompt — deterministic (§15.3).

### §6.4 Parse ladder (`hms_elite/utils/json.py::parse_llm_json`, §9.3) applies to every JSON task; `LLMBase.generate_json(task, system, user)` wraps transport errors as `LLMError(LLM_001, details={'task', 'provider'}) from e`. The lines task (§6.6) is not a JSON task: `LLMBase.generate_lines(task, system, user)` sends no `response_format`, returns the raw text and skips the ladder; `ingest.lines.parse_lines` does the tolerant parse.

### §6.5 Compact output contract (`ingest.extraction_output = 'compact'`; tasks `observation_extraction_compact`, `memory_extraction_compact`)

Why: on the real EgoLife run each 25-window batch made deepseek-flash emit a median 5,562 output tokens (cap 8,192; 22% of batches hit the cap and were bisected), 100% JSON (3.4 chars/token, no reasoning tokens), 17 keys per observation of which nine (`subject`, `predicate`, `objects`, `topic_tags`, `entity_mentions`, `sentiment`, `relation`, `confidence`, `attributed_to`) restate `text`. Output tokens are the write-latency and write-cost driver (output price ≈ 4× cache-miss input, ≈ 200× cache-hit input). The compact contract asks the LLM for the nine `COMPACT_OBSERVATION_KEYS` (eight fields plus the optional `entities` hint) and derives everything else locally; EPs and decisions keep their §6.3 shapes and are emitted only when their triggers are present (an action → outcome closure; a non-empty `## Candidate Existing Memories`, so an absent `decisions` root without candidates is silent, not a warning).

Observation item the LLM emits (OUTPUT FORMAT of both compact prompts — the §6.1 constants `OBSERVATION_EXTRACTION_COMPACT_PROMPT` and `MEMORY_EXTRACTION_COMPACT_PROMPT` in `hms_elite/configs/prompts.py` — verbatim):
```
{"observations": [{"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": null,
                   "entities": ["black coffee"], "supersedes_hint_ids": ["1"], "when": "2026-04-18", "valid_to": null}]}
```
Field rules: `text` third person, ONE statement, at most 25 words, no fact stated twice across items; `type ∈ fact|preference_event|habit|constraint|temporal_fact` (the vocabulary is spelled out in the prompt); `speaker` a named person or null; `entities` the named things the statement mentions (people, places, objects, tools, organisations) as 0–6 short strings, omitted when none — an optional hint that widens the derived `entity_mentions` (table below) and never narrows it; `supersedes_hint_ids` only when a supersession cue is present (the key may be omitted); `when` only when the statement itself carries an ISO date or datetime different from the message date; `valid_to` only for an explicit expiry. The user prompts are built by the same `build_observation_prompt` / `build_memory_extraction_prompt` with the same section titles, so `parse_sections` and the rule handlers are shared. Registration mirrors the full tasks: `observation_extraction_compact` ∈ `TASKS`, `PROMPTS`, `ROOT_KEY` (→ `observations`); `memory_extraction_compact` ∈ `COMPOSITE_TASKS`, `COMPOSITE_PROMPTS`. Each compact prompt is under 0.75 of its full twin's characters (`tests/test_extraction_compact.py`).

Derivation table (`hms_elite/ingest/derive.py::expand_compact_item`, called by `ingest/extraction.py::_compact_observation_from_item`; every derived value then passes the full-contract coercion `_enum_value` / `_prob` / `_date` / `_datetime` / `_str_list` / `_map_refs`, so vocabulary never rejects). Each heuristic has one implementation: the sentiment / topic tables in `utils.text` (group A); the third-person statement parser `llms.rule.statement_triple` (built on the R1–R9 cue tables `_PREF_VERB_MAP`, `_SWITCH_VERBS`, `_LIVE_WORK_PRED`, `CONSTRAINT_RE` / `MUST_RE`, `VERB_LEXICON` and the entity extractor), `observation_topics` and `mention_texts` (group B); `derive.py` (group C) compiles no cue of its own and only composes them (§0.4 forbids the rule LLM importing `ingest`; `tests/test_extraction_compact.py` asserts the only regex in `derive.py` is the full-datetime shape).

| field | derived as |
|---|---|
| `event_idx` → anchor | unchanged: the §12.3 witness rule (out of range / unreadable → item skipped with one warning) |
| `attributed_to` | the witnessed message's role side (`assistant` for an assistant message, else `user`) |
| `subject` | what the statement opens with (`statement_triple`: `User` / `The user` / `用户` → `user`, `Assistant` / `助手` → `assistant`, `The team` / `Team` / `Project` / `团队` / `项目` → `project`, a PROPER entity at position 0 → that name, e.g. `Sarah joined …` → `Sarah`), else `speaker` when named, else `project` for a constraint (R2 / R3's subject for an imperative), else `assistant` / `user` by role (`derive.subject_for`) |
| `predicate` | `statement_triple` on the text after the subject, by type with the R1–R9 cue tables: preference_event → `likes` / `dislikes` / `prefers` (`_PREF_VERB_MAP` in first- and third-person form, `_PREF_VERB_MAP_ZH`; `does not` / `no longer` flip polarity; a switch verb → relation `supersession`; an `over` / `rather than` alternative → `prefers` + relation `comparison`; no cue → `likes`, or `dislikes` when `sentiment` is negative); constraint → `must_not` via `CONSTRAINT_RE` / `CONSTRAINT_ZH_RE` (`does not` read as `do not`), else `must` via `MUST_RE`; habit → `does`; temporal_fact → `plans` when `FUTURE_RE` hits else `did` (R7); fact → `is_named` / `is` / `lives_in` / `works_at` / `studies_at` (`moved to` / `搬到` → `lives_in` + `supersession`) / `compares`, else the first inflected verb of the remainder (third-person `-s`, past `-ed`; `explains`, `keeps`) else `llms.rule.predicate_of` (the R9 lexicon heuristic, default `states`) |
| `objects` | from the same parse: the preference / identity / residence object(s) (`clean_object`), `constraint_objects(rest)` for a constraint (R2 / R3), the remainder minus its habit cue for a habit, the statement's entities minus the subject for a temporal_fact or a fallback fact (R7 / R9) |
| `relation` | `supersession` when `supersedes_hint_ids` is non-empty (the cue was present; unknown refs are still dropped by `_map_refs`), else the statement's own cue (`supersession` for a switch / relocation verb or `no longer`, `comparison` for an alternative), else `habit` for type habit, `constraint` for type constraint, else `single_fact` |
| `confidence` | `0.7` (`derive.COMPACT_CONFIDENCE`, the full contract's default for a missing value) |
| `date_observed` | `when` when it parses as a date (a datetime yields its UTC date), else the witnessed message's own date (`date_of(event.timestamp)`; the batch `observation_date` only when that timestamp is unreadable) |
| `valid_from` | `when` only when it is a full datetime (`derive.is_full_datetime`), else None — `atom_from_observation` then fills the time anchor exactly as in full mode |
| `valid_to` | the item's `valid_to` through `_datetime`, floored exactly as in full mode |
| `sentiment` | `utils.text.sentiment(text)` — the rule LLM's lexicon (positive / negative / mixed / neutral) |
| `topic_tags` | `llms.rule.observation_topics(text, type)` — the rule LLM's `TOPIC_TABLE` hits plus `preference` / `constraint` for those types, at most four |
| `entity_mentions` | `derive.union_mentions(entities, llms.rule.mention_texts(text, entity_extractor), [speaker])` — the LLM's `entities` hint first (coerced by `extraction._entities_hint` through `_str_list`: None → `[]`, a string → one item, dict items → their text; a number or a dict where the list belongs is ignored with one warning and never skips the item), then the configured entity provider's hits in `text` (`ingest.entity_extractor`, the same instance `link_atom_axes` re-classifies with), then `speaker`; case-insensitive de-duplication keeping the first spelling (`_observation_from_item` then finds the speaker already present, as in full mode). The entity axis of a compact atom is therefore a superset of extractor(`text`) ∪ `speaker`, and equal to the full-mode axis whenever the LLM lists under `entities` what it would have listed under `entity_mentions`: `entities: ["python"]` on `User prefers python for scripting.` stores `entity_ids == ["python"]` exactly as full mode does, while the same item without the hint stores `[]` (`tests/test_extraction_compact.py`). The prompt's ≤ 6 bound is not enforced by the coercion (full mode caps nothing either) |
| `speaker` → `actor_id` | unchanged (§12.5): the message `name` wins, else `speaker` |

Invariants in compact mode (`tests/test_extraction_compact.py`): every observation carries exactly its witness anchor; all seven axes (time / validity / entity / topic / role / sentiment / scale) are filled from the derived values (the entity axis as the union above — never narrower than full mode, and equal to it per row (`entity_ids`) and per table (`atom_entity`) on every chat fixture); supersession hints resolve through `id_map`; on every chat fixture in both `extraction_mode`s the derived `subject` / `predicate` / `objects` / `relation` equal the rule LLM's own full observations, so the `atom_anchor` / `atom_entity` / `atom_topic` / `entities` / `history` / `graph_nodes` / `graph_edges` / `edge_witnesses` tables are row-identical between a full and a compact facade (`chat_conflict` keeps its CONFLICT history row, `chat_multisession` keeps subject `Sarah`); a `"decisions": []` root with no candidates shown is silent (ADD, no per-observation warning), exactly like an absent root; the rule LLM answers both compact tasks (`RuleLLM._task_observation_extraction_compact` / `_task_memory_extraction_compact`: the R1–R9 observations projected onto the compact keys — `when` only when `date_observed` differs from the message date, `entities` = the full observation's `entity_mentions` and the hint list only when non-empty — with the decisions judged on the full observations before the projection) and stores the same texts, events, anchors, types and sides as full output on every chat fixture; `extraction_output='full'` is byte-identical (prompt sha256 pinned, goldens unchanged). Applies to both `extraction_mode`s. An unknown value → `ConfigError(CONFIG_001)` from config, `ValidationError(VALIDATION_013)` from `Ingestor(extraction_output=...)` / `extract_all(output=...)` / `extract_observations(output=...)`. `lines` is the default since 2026-09-18 (§6.6, changelog row 108); `full` held it before (`compact` for part of 2026-09-11, reverted the same evening — next paragraph); `full` and `compact` stay available. The compact DECISIONS guideline (2026-09-11, changelog row 103) is stated per NEW observation `id` — never one per candidate, never by position — with NONE reserved for a restatement (same fact, same subject, same value; a new detail, person, time or value is ADD), a Bad / Good pair where a similar candidate still yields ADD, and an example with two candidates whose target id (`1`) differs from the observation id (`0`) — the OUTPUT FORMAT block of both compact prompts cites hint id `1` and (memory task) target `1` for observation `0` as well (row 104), so no observation-id / candidate-id coincidence is exemplified anywhere; the decisions shape and the observation keys are unchanged. Under the earlier wording deepseek-flash paired observation i with neighbourhood candidate i and answered NONE for 1,685 of 7,315 observations on the 204-question EgoLife run (23 of 8,470 under `full`; 1,623 with `target_id == id`), each NONE discarding the statement (`benchmarks/egolife/results/replay_diag/summary.md`); the library guard of §6.3 / §12.6 step 6 catches what the wording does not.

Default and recommendation (2026-09-11, revised the same evening; superseded on 2026-09-18 by the `lines` default of §6.6, changelog row 108 — this paragraph is the history of the `full` vs `compact` decision). Until 2026-09-18 the default was `full` (`IngestConfig.extraction_output = 'full'`). `compact` became the default on the morning of 2026-09-11 (changelog row 102) under a two-condition rule and was reverted on the evening of 2026-09-11 (changelog row 105) under a third condition added the same day; `compact` remains available, fully supported and byte-identical in what it leaves untouched (the full prompts' sha256 pinned, goldens unchanged). The three conditions for `compact` to hold the default: (a) in the zero-API replay A/B on the 204 EgoLife questions the compact arm's hit@20 must not be below the full arm's — met: 161/204 in both arms, MRR 0.5000 vs 0.4973, and on the entity axis compact ⊇ full on 8,439 of 8,439 paired atoms (24 strict supersets, 0 narrowed; `benchmarks/egolife/results/replay_ab_entities`); (b) the final compact prompt (with the `entities` hint, changelog row 100) must complete a real-API smoke on the same 18 batches and confirm the output-token reduction — met: `benchmarks/egolife/results/real/elite_compact_entities_smoke5_r2` (run_id `20260911T090326Z`) completed all 18 batches / 395 windows with a median 2,793 / max 4,312 output tokens per request against 5,631 / 8,192 for `full` on the same batches (−50% median; `full` = the first 18 batches of `results/real/elite_full204_r3`, 34 requests), 0 of 18 vs 8 of 34 attempts at the 8,192 cap, a median model time per request of 8,675 ms vs 17,147 ms (−49%), 672 add() results (606 atoms stored) vs 660 (655 stored), 0 windows lost, 3/5 (rows) and 4/5 (card) correct answers against 3/5 and 3/5, and a bill of $0.089961 (≈ $0.0435 at off-peak prices); the interrupted first attempt `results/real/elite_compact_entities_smoke5` keeps its `INCOMPLETE.md`; (c) on the real 204-question EgoLife run the compact arm's accuracy on the memory-rows context must be within 2 points of the full arm's — not met. All real deepseek-flash runs on the same 204 questions (`docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`): `full` r3 rows 61.3% (125/204) / card 61.8% (126/204), hit@20 78.9%, ¥11.56 re-priced at off-peak, 14.9 s model time per batch; compact v1 (`results/real/elite_compact_full204`) rows 53.9% (110/204) — root cause the index-paired NONE verdicts of the earlier DECISIONS wording, 1,685 NONE of 7,315 observations (row 103); compact v2 (`results/real/elite_compact_v2_full204`: the rewritten DECISIONS wording of rows 103–104 plus the `ingest.verdict_min_similarity` guard at 0.5) rows 56.4% (115/204) / card 62.7% (128/204), hit@20 77.9% / 79.4%, verdicts ADD 7,604 / NONE 15 / UPDATE 9, 0 windows lost, ¥4.35 at off-peak, 7.3 s per batch; mem0 56.9% (116/204). 56.4 vs 61.3 is 4.9 points below, so the rule fails and `full` kept the default over `compact` (until the 2026-09-18 switch to `lines`). Recommendation: `compact` for cost-sensitive bulk ingestion (0.38× write cost, 0.49× model time per batch, 3 vs 94 attempts at the cap) and wherever the state-card context is the reader input (card 62.7% vs 61.8%); `full` where the memory-rows context is read or where the LLM's own `confidence` / `topic_tags` / `sentiment` are needed (compact derives `confidence` 0.7 for every observation and the other two from the lexicons, table above). For reference, the row-99 prompt without the hint (`results/real/elite_compact_smoke5`) measured a median 2,002.5 / max 5,106 output tokens, 0 cap hits, 7,140.5 ms per request, 568 add() results (556 stored) and 3/5 in both contexts. Measurements, definitions and reproduction commands: `docs/design/COMPACT_EXTRACTION_2026-09-11.md`.

### §6.6 Lines output contract (`ingest.extraction_output = 'lines'`; task `memory_extraction_lines`; `extraction_mode = 'single'` only)

Why (measured 2026-09-11/12, off-peak CNY prices, 4,851 windows): write cost mem0 ¥1.85 = input-miss 0.93M tokens + output 0.22M; Elite `full` ¥11.5 (output 2.40M tokens = 83% of the cost); Elite compact v2 ¥4.2 (output 0.76M). The input-miss side is already at parity; the gap is output tokens, and a JSON envelope costs 2–4× the text it carries. The zero-API replay showed retrieval uses only the statement texts and the decisions (entities / type / when and the other fields move hit@20 by 0). The compact prompt's own texts scored below the full prompt's (rows 56.4% vs 61.3%; paired 32 vs 22), so the lines prompt keeps the full prompt's statement rules verbatim and replaces only the OUTPUT FORMAT.

Measured (2026-09-12, `results/real/elite_lines_full204`, all 204 questions, deepseek-flash, off-peak): memory-rows context 54.9% (112/204), state-card context 63.2% (129/204, the highest of every arm); hit@20 84.3% / 82.4% (the highest); 51 output tokens per window (full 495, compact 157, mem0 45), median model time per batch 3.76 s (full 14.9, compact 7.3, mem0 3.5), write + answer re-priced at off-peak ¥2.24 (mem0 ¥1.98, compact ¥4.35, full ¥11.56); 12,994 atoms stored (2.68 per window, 72.5 MB); verdicts ADD 12,963 / NONE 20 / UPDATE 11; no truncation, no window lost. Conclusion: cost and write speed reach mem0's level with higher recall; rows-context accuracy is 4 points below full (72 evidence-present wrong answers), state-card accuracy 1.4 points above full.

Default (2026-09-18, changelog row 108): `lines` is the library default (`IngestConfig.extraction_output = 'lines'`; `extraction_mode` stays `'single'`, so a `staged` config now needs an explicit `full` | `compact`). The decision follows the reader A/B of `docs/design/READER_AB_2026-09-18.md` §六 on the same 204 questions: with the lines store held identical, the hybrid read of §6.7 (the memoized session card as a prefix, then timestamped relevant rows, answered memories-first) answers 70.1% (143/204) — above every full-contract context (rows 61.3%, card 61.8%) at the lines write cost (¥2.24 vs ¥11.56 per 204 questions, off-peak), sign test against the plain rows context p < 0.0001. The library's read path for it is `HMS.context()` (§6.7); the harness's elite_hybrid label is built through the same method but is not a byte reproduction of READER_AB's hybrid_prefix (§6.7, deviations). `full` stays the recommendation where the LLM's own `confidence` / `topic_tags` / `sentiment` are needed; `compact` where a JSON envelope is wanted at lower cost.

Wire format (the OUTPUT FORMAT of `MEMORY_EXTRACTION_LINES_PROMPT`, §6.1 constant in `hms_elite/configs/prompts.py`; plain text, one tab between fields, shown as `⇥`):
```
OBS
<event_idx>⇥<T>⇥<statement>[⇥@<YYYY-MM-DD or ISO datetime>]
EP                  (present ONLY when an action → outcome closure exists)
<event_idx>⇥u=<trigger>⇥a=<action>⇥y=<outcome>⇥r=positive|negative|unknown[⇥k=<constraint>;<constraint>]
DEC                 (present ONLY for non-ADD verdicts; an absent section = every observation is ADD)
<OBS line number, 1-based>⇥REVISE|CONFLICT|NONE⇥<candidate id from Candidate Existing Memories>⇥<short reason>
```
`T ∈ F fact | P preference_event | H habit | C constraint | T temporal_fact` (`configs.prompts.LINE_TYPES`; inverse `TYPE_LETTERS`); a statement never contains a tab; `@date` has exactly the meaning of compact's `when` (only a date the statement itself carries, different from the message date, resolved against Observation Date). Registration: `memory_extraction_lines ∈ COMPOSITE_TASKS` / `COMPOSITE_PROMPTS`, outside `TASKS`, `PROMPTS` and `ROOT_KEY`; same user-prompt builder (`build_memory_extraction_prompt`) and sections as the full task. The prompt carries the full prompt's statement rules verbatim (third person, 5–80 words, one self-contained statement, quantities / dates / times / names kept, the supersession-cue list, questions / greetings / boilerplate yield nothing, assistant statements start with "Assistant", the named person of a multi-speaker transcript is the subject) and the rewritten compact DECISIONS rules (at most one DEC line per NEW observation, only for a non-ADD verdict, never per candidate or by position; NONE only for a same-subject same-value restatement; temporal facts and habits always ADD; the exclusive-attribute rule); its OUTPUT FORMAT contains no JSON braces and no JSON trailer, and its one example has two OBS lines, no EP and one DEC line (`tests/test_extraction_lines.py`). No staged twin: `observation_extraction` has no lines form, so `lines` is single-mode only.

Call path: `LLMBase.generate_lines(task, *, system, user) -> str` builds the two messages and runs `_call` with NO `response_format` (a JSON mode would make the model wrap the lines in an object), returning the raw text (a tool-call dict yields its `content`; None → ''); transport errors are `LLMError(LLM_001)` as for `generate_json`. Every provider honours it without change: deepseek / openai forward `response_format` only when given, anthropic appends its "Respond with JSON only." line only for a JSON format, ollama sets `format='json'` only for one; `RuleLLM._task_memory_extraction_lines` renders the §6.6 lines (§11.8) and `ScriptedLLM` replays a string as is.

Parser (`hms_elite/ingest/lines.py::parse_lines(text, *, what='memory_extraction_lines') -> {'observations', 'eps', 'decisions'}`, the dict shape `extract_all` consumes for compact): observations are compact-shaped items `{'id': the 1-based OBS line number as str, 'event_idx', 'text', 'type', 'when'?}`; EPs `{'id', 'event_idx', 'u', 'b', 'a', 'y', 'r', 'kappa': {'constraints': [k= split on ';']}}`; decisions `{'id': the OBS line number it names, 'decision', 'target_id', 'reason'}`. Rules: `\r\n` is normalised; blank lines, code fences and anything before the `OBS` header are ignored (a `# OBS:` spelling is tolerated; a tabbed `<idx>⇥<T>⇥…` line before any header opens OBS implicitly); inside OBS a line with three or more fields is `<idx> <T> <statement> [@date]`, a two-field line is `<idx> <statement>` (type fact) when its first field is a number or `<T> <statement>` on the previous line's event_idx when it is a type letter, and a line without a tab is a continuation that consumes no line number — the statement of an immediately preceding prefix-only line, else appended after a space to the previous observation's statement (skipped with one warning when there is no previous observation); `@date` is read as its own tab field or glued after a space at the end of the statement; an unknown type letter reads as fact (DEBUG; a full type name maps by its initial); a tabbed line whose statement is shorter than two characters (an empty statement, or a lone letter left by a cut) stores nothing but keeps its line number; a non-numeric event_idx is kept as text so the §12.3 witness rule rejects the item with its own warning; a trailing line without a newline and with fewer than two tabs is a truncated reply and is dropped with one warning (a cut-off statement that still has its two tabs is kept; a header or fence as the last line is not a truncation); an EP field without a known `u= b= a= y= r= k=` prefix is ignored (DEBUG), `r` defaults to `unknown`, an EP line with no slot at all is skipped with a warning; a DEC line naming an OBS line number outside 1..n_lines, a line that stored nothing, a non-number, or with fewer than three fields is skipped with a warning, its verdict is upper-cased, the reason is the rest of the line; an absent section is an empty list; a reply that opens no section at all (prose, a JSON object, '') logs one warning containing `NO_SECTION_MARK` (`'reply has no OBS / EP / DEC section'`), the signal the benchmark harness treats as an unparsable reply. Numbering: every tabbed line of the OBS section consumes one 1-based line number whether or not it stores an observation, so `id` is the line the model wrote and a DEC line's number survives a dropped or wrapped line (2026-09-12 fix, changelog row 107: ids were the ordinal of the kept observations, so a dropped or split line shifted every later DEC target by one).

Derivation = compact. `extract_all(output='lines')` runs `_compact_observation_from_item(..., speaker_from_subject=True)` on every parsed observation — witness first (§12.3 event_idx rule), then `ingest.derive.expand_compact_item` with `speaker=None` and `entities=None`, then the full-contract coercion — so subject / predicate / objects / relation / date_observed / valid_from / confidence 0.7 / attributed_to / sentiment / topic_tags / entity_mentions follow the §6.5 table with `entities = []` and `supersedes_hint_ids = []` (relation therefore never comes from a hint; the statement's own cue or the type decides, and a REVISE is corroborated in the §12.6 guard only by an equal subject + predicate). The one lines-specific rule: the format has no speaker slot, so when the derived subject is not a conversation side (`derive.SIDE_SUBJECTS = {'user', 'assistant', 'project'}`) — the statement's leading proper name, which the full prompt asks to be both subject and speaker of a multi-speaker transcript — it becomes `speaker`, hence an entity mention and (when the message carries no `name`) `actor_id`, exactly where a compact `speaker` would land. EPs: the wire format carries u / a / y / r / k (and an optional b=), so `ingest.derive.expand_lines_ep` fills the rest locally before `_ep_from_item`, exactly as the rule LLM fills its own EPs — `entity_mentions` = `mention_texts('<a> <y> <u>')`, `topic_tags` = `topic_tags('<a> <y> <u>')`, `b` = the k= constraints joined by `' ; '` when no b= was given; `time_anchor` = the event time; `confidence` 0.7; `visibility` = the event's; `uses_ep_ids` empty (2026-09-12, row 107: before this the three columns were stored empty under lines). Decisions: a DEC line joins its observation by line number through `_parse_decision` (candidate id → atom_id; an unknown target → ADD with a warning); an observation without a DEC line is ADD (`'no DEC line'`) with no warning, candidates shown or not (the contract says absence means ADD); the §6.3 / §12.6 verdict guard applies unchanged.

Invariants (`tests/test_extraction_lines.py`): on every chat fixture the lines facade stores the same texts, events, witness anchors, sides, types, `entity_ids`, `revises`, atom (text, subject, predicate, objects) and the same `atom_anchor` / `atom_entity` / `atom_topic` / `entities` / `history` / `graph_nodes` / `graph_edges` / `edge_witnesses` / `eps` rows as the full facade (chat_conflict keeps its CONFLICT row, chat_preference its REVISE, chat_action_closure its EP with the same entity / topic / b columns); the `observations` and `memory_atoms` rows are identical except for exactly the columns the compact derivation owns, which the test pins column by column: `confidence` (the 0.7 constant vs the rule LLM's 0.6–0.9), `relation` and `supersedes_hint_ids` (`single_fact` / `[]` vs `supersession` / the hint id on chat_preference and chat_zh — the lines format has no hint slot), `memory_atoms.actor_id` where the speaker rule names the statement's leading proper name (chat_multisession: `Sarah` on every row, none under `full`), and `observations.entity_mentions` on chat_zh (`用户`, read as a mention from the derived statement text — a compact-path difference that does not reach `atom_entity`); a scripted lines reply through `HMS.add` derives every axis and creates a revision edge from a DEC REVISE while the guard still downgrades a non-restatement NONE; the call carries no `response_format` (ScriptedLLM kwargs `{}`, the DeepSeek provider's `chat.completions.create` kwargs); the full and compact prompts stay byte-identical (four sha256 pinned). `extraction_mode='staged'` with `lines` → `IngestConfig` ValueError with a bilingual message (`ConfigError(CONFIG_001)` through `from_dict`), `Ingestor(...)` / `extract_observations(output='lines')` → `ValidationError(VALIDATION_013, details.supported=['single'])`. `lines` is the default since 2026-09-18 (§5, row 108; `full` before); the benchmark CLI (`benchmarks/egolife/run_real.py --extraction-output`, choices = `EXTRACTION_OUTPUTS`, default = the library default) and `EliteRealArm(extraction_output=...)` pass an explicit value through unchanged; the real-API measurements are the two paragraphs above and `READER_AB_2026-09-18.md`.

### §6.7 Hybrid read context (`HMS.context()`; `hms_elite/compiler/context.py`; 2026-09-18)

The read path the reader A/B found best (`docs/design/READER_AB_2026-09-18.md` §六: hybrid_prefix 70.1% (143/204) on EgoLife, sign test vs the plain rows context p < 0.0001, cold-prefix cost ¥0.00185 per question): a query-independent session card memoized per scope, then the query's relevant memory rows with timestamps, placed BEFORE the user's question.

Signature (§17.3): `context(self, query: str, *, filters: dict, top_k: int | None = None, as_of: Any = None, budget_tokens: int | None = None) -> dict`. Unrelated to `AgentRuntime.context(...) -> ContextIR` (`hms_elite/runtime.py`), which compiles one card; the same name on two objects with two contracts.

Text shape (`compiler.context.hybrid_text`, byte-identical to the harness's `hybrid_context`): `"<session card>\n\n<context_rows_header>\n<rows>"` — `<session card>` is `compile(...).to_text()` or `(none)` while the scope holds no extractable event; `<rows>` is the row block or `(none)` when nothing is kept; the header is `CompilerConfig.context_rows_header` (default `Recent relevant memories:` = `compiler.context.ROWS_HEADER`). The product puts the text before the question — `"Memories:\n<text>\n\n<prompt>"` — because memories-first has no accuracy cost (READER_AB §四) and is the only ordering whose prefix a provider's prompt cache (DeepSeek) can hit while the session card is reused.

Rows: `search(query, filters, top_k=<top_k if given else context_rows_top_k>, reference_date=as_of)` → `dedup_rows` (key = casefolded, whitespace-collapsed text; exact duplicates dropped, the first in rank order survives; then every row whose key is a proper substring of another unique row's key is dropped; rank order kept) → `sort_rows_by_time` (stable ascending by `hms.time_anchor`, fallback `created_at`, rows with neither last in input order) → `render_rows`: one line `- [YYYY-MM-DD HH:MM] <text>` per row with the stamp from `time_anchor` ONLY in UTC (`format_stamp`; the bracket is omitted when the anchor is absent or unparsable — `created_at` never stamps), or `- <text>` when `context_timestamps` is False. `rows_text(raw)` = the three steps over raw rows (= the harness's rows_ts / rows_td contexts).

Session card: `compile(query=<text of the scope's latest extractable event>, filters=<scope filters (+ visibility)>, as_of=<that event's timestamp>, budget_tokens=<budget_tokens if given else context_session_budget_tokens>)`, where the event is `StateStore.latest_event(scope, object_types=[message, tool_call, file_diff, test_log, sensor_trace])` (§8: `ORDER BY timestamp DESC, rowid DESC` — the last WRITTEN of the N messages of one `add()`, which share event_time; `last_events` tie-breaks on the random event_id, hence the separate method); the query is the event payload's `text` (message events) else its canonical JSON. system / feedback / checkpoint / rollback / scheduler events never become the query. The card is memoized in process per `(scope.key(), audience)` together with `StateStore.scope_version(scope)` and the budget, and reused verbatim (`session_reused = True`) while both are unchanged; a different `budget_tokens` recomputes.

Invalidation is a per-scope write version, never a timer (§8): a transaction attributes itself to a scope with `StateStore.touch_scope(scope)` — called by `insert_source_event` (only when a row was stored — a replayed identical event is a no-op), by `insert_atom`, and by every `StateScheduler` phase transaction and its transition export (they write scope-less rows — edges, history, decay, transitions — on behalf of a target scope, so `consolidate(filters=B)` and the daemon leave scope A's memo reused; `feedback` and `purge(dry_run=False)` write inside the one transaction that stores their scoped system event and are per-scope for that reason); at the COMMIT of the outermost transaction the attributed scopes' counters advance (`scope_write_versions[key] += 1`) ONLY IF the transaction wrote versioned rows, so `add()`'s Txn A bumps once (events) and Txn B again (atoms) and a context built during the unlocked phase is invalidated by Txn B, while a transaction that changed no row (an ignored INSERT OR IGNORE, an UPDATE of nothing, a scheduler phase with nothing to do) moves no version at all; rows nothing a scope reads depends on are written inside `StateStore.unversioned()` and count for neither — an internal event and its anchor (`system_event`: scheduler / checkpoint / rollback / feedback / system; the event still attributes the operation it witnesses to its scope, and that operation's own archives / revisions / decays are what invalidate), the `checkpoints` row, the scheduler's `last_consolidation:<scope>` / `work_state:<scope>` meta rows and its compiled-state archives — so a checkpoint, an idle `consolidate(A)` and the daemon's `consolidate()` over scopes with nothing to do leave every memo reused (a consolidate that creates or revises atoms, edges, InterfaceEPs or skills, or decays a stale atom, invalidates as before); a committed transaction that changed versioned rows under no scope and without a scoped caller, every autocommit versioned write, rollback and `reset()` advance the global `write_generation`; `scope_version(scope)` = `(write_generation, data_version(), sorted (key, n) of every recorded scope that covers or is covered by scope)` — so `delete_all(user_id=u)` (a partial-scope system event) invalidates the `(u, agent)` memo, and a write to an unrelated scope leaves it reused. `HMS.reset()` also clears the memo.

Return dict: `{'text', 'session_card', 'session_key' (the event_id the card was compiled from; None when the scope has no extractable event), 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids' (memory ids of the card's items), 'rows' (the kept rows in time order), 'rows_text', 'rows_header', 'search_rows' (the raw search rows in rank order), 'scope'}`. Validation: `parse_filters(require_scope=True)` (VALIDATION_001), `search`'s own checks (VALIDATION_009 / 010); `top_k` / `budget_tokens` are passed through when given (an explicit 0 is not silently the default). `as_of` bounds the ROWS only (search `reference_date`); the session card is always the latest event at its own timestamp (streaming use; the memo is keyed on the scope, not on `as_of`). Read-only in the store (one search, at most one compile); `AsyncHMS.context` mirrors it; no HTTP route and no CLI verb (a `POST /context` route is a follow-up).

Deviations from the READER_AB harness, recorded not fixed: the harness recomputed its session card only when a step's retain produced ≥ 1 ADD / UPDATE and keyed it on the last window actually stored, with `as_of` = that window's own start; the library invalidates on ANY committed write to the scope (an `add()` whose extraction yields only NONE still stores its events) and uses the latest event's timestamp (for the harness: the slice's first-window start). Both instants are ≥ every atom's `valid_from` (the validity predicate is inclusive), so the card content is equivalent on the harness, but the `elite_hybrid` numbers of `run_real.py` are not byte reproductions of READER_AB's hybrid_prefix. Tests: `tests/test_context.py` (shape and '(none)' parts; memo reuse and invalidation by add / update / delete / delete_all incl. the parent scope / a consolidate that changed rows / rollback / reset, precision across scopes; reuse across a checkpoint, an idle consolidate, the daemon-shaped `consolidate()` and zero-row transactions; `scope_version` commit-time and `unversioned()` semantics; stamps; the harness's `reader_ab.dedup_rows` / `sort_rows_by_time` / `format_stamp` / `row_times` / `hybrid_context` ARE these functions and `rows_ts_context == rows_text` on the seven hand rows; config fields and per-call overrides).

---

## §7 SQLite DDL (`hms_elite/memory/schema.py`, group A)

Module docstring: `"""SQLite schema for all HMS planes in one file; invariants as CHECKs and triggers.\n单文件多平面 SQLite 模式；不变量以 CHECK 与触发器表达。"""`

Connection (owned by `StateStore`): `sqlite3.connect(path, check_same_thread=False, isolation_level=None)`; `PRAGMA foreign_keys = ON`; `PRAGMA busy_timeout = <ms>`; migrations (`apply_migrations`: a version-0 database is created directly from `build_ddl(SCHEMA_VERSION)` in ONE `BEGIN IMMEDIATE` transaction — one commit, no freed pages — while an existing file walks the append-only `MIGRATIONS` ladder from its `user_version`, one commit per step; both yield the same `sqlite_master` row set); then `PRAGMA journal_mode = WAL` for file DBs only (the order matters: a fresh file writes its 97 schema pages once, in rollback-journal mode, instead of through the WAL and again at the close-time checkpoint). A private `':memory:'` database does not re-execute the DDL: `clone_schema` copies a process-wide, schema-only template (built once by `apply_migrations` on `':memory:'`, never written, never handed out) page-for-page with the backup API — same `sqlite_master` rows, root pages and `user_version` as the DDL path; `file:` URIs and files always take the statement-by-statement path (it is the only race-free one across processes). `SCOPE` below expands to `user_id TEXT, agent_id TEXT, run_id TEXT, CHECK (user_id IS NOT NULL OR agent_id IS NOT NULL OR run_id IS NOT NULL)`; `VIS` to `CHECK (<col> IN ('private','team','event','public'))`. All CHECK lists are generated from `hms_elite.enums` with exactly two narrowings: `source_events.role` uses `[r for r in Role.values() if r != 'extracted']` (events never carry the synthetic role), and `eps.scale` is the literal `'L1_micro'`. The printed DDL below is the generated output and is authoritative.

```sql
-- ===================== evidence plane =====================
CREATE TABLE source_events (
    event_id        TEXT PRIMARY KEY,
    source_system   TEXT NOT NULL,
    object_type     TEXT NOT NULL CHECK (object_type IN ('message','tool_call','file_diff','test_log','sensor_trace','feedback','checkpoint','rollback','system','scheduler')),
    event_type      TEXT NOT NULL CHECK (event_type IN ('created','updated','deleted','feedback','checkpoint','rollback','tool_result','scheduler')),
    actor_id        TEXT,
    role            TEXT CHECK (role IS NULL OR role IN ('user','assistant','system','tool')),
    timestamp       TEXT NOT NULL,                          -- event time, 32-char UTC ISO
    ingested_at     TEXT NOT NULL,                          -- store.now()
    raw_ref         TEXT,
    visibility      TEXT NOT NULL VIS,
    SCOPE,
    scope_key       TEXT NOT NULL,                          -- Scope.key()
    payload_json    TEXT NOT NULL,                          -- normalized payload, canonical_json
    payload_hash    TEXT NOT NULL,
    user_meta_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX ix_events_scope_time ON source_events(scope_key, timestamp);
CREATE INDEX ix_events_type_time  ON source_events(object_type, timestamp);
CREATE UNIQUE INDEX ux_events_identity ON source_events(payload_hash, timestamp, scope_key);   -- One Event, One Anchor (same content at the same event time in the same scope)
CREATE TRIGGER trg_events_no_delete BEFORE DELETE ON source_events WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: source_events are append-only'); END;   -- §16.6 gate (schema 2)
CREATE TRIGGER trg_events_immutable BEFORE UPDATE ON source_events BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: source_events are immutable'); END;

CREATE TABLE anchors (
    anchor_id        TEXT PRIMARY KEY,
    source_event_id  TEXT NOT NULL UNIQUE REFERENCES source_events(event_id),
    hash             TEXT NOT NULL,
    source_ref       TEXT,
    timestamp        TEXT NOT NULL,
    ingested_at      TEXT NOT NULL,
    permission_scope TEXT NOT NULL VIS,
    SCOPE,
    revoked_at       TEXT
);
CREATE INDEX ix_anchors_hash ON anchors(hash);
CREATE INDEX ix_anchors_scope_time ON anchors(user_id, agent_id, run_id, timestamp);
CREATE TRIGGER trg_anchors_no_delete BEFORE DELETE ON anchors WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: anchors are never deleted (use revoked_at)'); END;   -- §16.6 gate (schema 2)
CREATE TRIGGER trg_anchors_immutable BEFORE UPDATE OF anchor_id, source_event_id, hash, source_ref, timestamp, ingested_at, user_id, agent_id, run_id ON anchors
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: anchors are immutable (only permission_scope/revoked_at may change)'); END;

-- ===================== observation / state layers =====================
CREATE TABLE observations (
    obs_id          TEXT PRIMARY KEY,
    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),          -- primary witness
    anchor_ids_json TEXT NOT NULL DEFAULT '[]',
    type            TEXT NOT NULL CHECK (type IN ('fact','preference_event','habit','constraint','temporal_fact')),
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    objects_json    TEXT NOT NULL DEFAULT '[]',
    relation        TEXT NOT NULL CHECK (relation IN ('single_fact','comparison','habit','supersession','constraint')),
    text            TEXT NOT NULL CHECK (length(text) > 0),
    date_observed   TEXT NOT NULL,
    valid_from      TEXT, valid_to TEXT,
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    visibility      TEXT NOT NULL VIS,
    attributed_to   TEXT NOT NULL CHECK (attributed_to IN ('user','assistant')),
    sentiment       TEXT NOT NULL CHECK (sentiment IN ('positive','neutral','negative','mixed','unknown')),
    topic_tags_json TEXT NOT NULL DEFAULT '[]',
    entity_mentions_json TEXT NOT NULL DEFAULT '[]',
    supersedes_hint_ids_json TEXT NOT NULL DEFAULT '[]',
    extracted_by    TEXT NOT NULL,
    atom_id         TEXT,
    SCOPE,
    created_at      TEXT NOT NULL
);
CREATE INDEX ix_obs_anchor ON observations(anchor_id);
CREATE INDEX ix_obs_subject_pred ON observations(user_id, agent_id, run_id, subject, predicate);
CREATE TRIGGER trg_obs_no_delete BEFORE DELETE ON observations WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: observations'); END;   -- §16.6 gate (schema 2)
CREATE TRIGGER trg_obs_immutable BEFORE UPDATE OF obs_id, anchor_id, type, subject, predicate, objects_json, text, date_observed, created_at ON observations
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: observation content is immutable'); END;

CREATE TABLE eps (
    ep_id           TEXT PRIMARY KEY,
    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),          -- primary witness (outcome anchor for closures)
    anchor_ids_json TEXT NOT NULL,
    scale           TEXT NOT NULL CHECK (scale = 'L1_micro'),
    u TEXT NOT NULL CHECK (length(u) > 0), b TEXT NOT NULL DEFAULT '', a TEXT NOT NULL CHECK (length(a) > 0), y TEXT NOT NULL DEFAULT '',
    r               TEXT NOT NULL CHECK (r IN ('positive','negative','neutral','unknown')),
    kappa_json      TEXT NOT NULL,
    time_anchor     TEXT NOT NULL,
    valid_from      TEXT NOT NULL, valid_to TEXT,
    entity_mentions_json TEXT NOT NULL DEFAULT '[]',
    topic_tags_json TEXT NOT NULL DEFAULT '[]',
    uses_ep_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    visibility      TEXT NOT NULL VIS,
    extracted_by    TEXT NOT NULL,
    atom_id         TEXT,
    SCOPE,
    created_at      TEXT NOT NULL,
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE INDEX ix_eps_scope_time ON eps(user_id, agent_id, run_id, time_anchor);
CREATE TRIGGER trg_eps_no_delete BEFORE DELETE ON eps WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: eps'); END;   -- §16.6 gate (schema 2)
CREATE TRIGGER trg_eps_immutable BEFORE UPDATE OF ep_id, anchor_id, anchor_ids_json, scale, u, b, a, y, r, kappa_json, time_anchor, valid_from, created_at ON eps
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: ep slots are immutable'); END;

-- ===================== State Index plane (MVMI) =====================
CREATE TABLE memory_atoms (
    atom_id           TEXT PRIMARY KEY,                       -- version id
    memory_id         TEXT NOT NULL,                          -- lineage id (Mem0 'id')
    version           INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    atom_type         TEXT NOT NULL CHECK (atom_type IN ('message','state','event','entity','relation','episode')),
    content_json      TEXT NOT NULL,
    text              TEXT NOT NULL CHECK (length(text) > 0),
    hash              TEXT NOT NULL,                          -- md5(text)
    time_anchor       TEXT,
    valid_from        TEXT NOT NULL,
    valid_to          TEXT,
    scale_level       TEXT NOT NULL CHECK (scale_level IN ('L1_micro','L2_meso','L3_macro','state','event','entity','relation')),
    role              TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool','extracted')),
    sentiment         TEXT CHECK (sentiment IS NULL OR sentiment IN ('positive','neutral','negative','mixed','unknown')),
    subject           TEXT,
    predicate         TEXT,
    obs_type          TEXT CHECK (obs_type IS NULL OR obs_type IN ('fact','preference_event','habit','constraint','temporal_fact')),
    source_kind       TEXT NOT NULL CHECK (source_kind IN ('message','observation','event','entity','ep','interface_ep','skill','procedural','feedback')),
    source_id         TEXT NOT NULL,
    source_ep_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    extracted_by      TEXT NOT NULL,
    primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),   -- No Witness, No Meaning (structural)
    visibility        TEXT NOT NULL VIS,
    SCOPE,
    actor_id          TEXT,
    attributed_to     TEXT CHECK (attributed_to IS NULL OR attributed_to IN ('user','assistant')),
    memory_type       TEXT,
    user_meta_json    TEXT NOT NULL DEFAULT '{}',
    archived_reason   TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE UNIQUE INDEX ux_atoms_lineage_version ON memory_atoms(memory_id, version);
CREATE UNIQUE INDEX ux_atoms_head ON memory_atoms(memory_id) WHERE archived_reason IS NULL;   -- lineage head: at most ONE non-archived version per lineage (enforced); an expiring head (valid_to set, archived_reason NULL) is still the head
CREATE INDEX ix_atoms_time      ON memory_atoms(time_anchor);
CREATE INDEX ix_atoms_validity  ON memory_atoms(valid_from, valid_to);
CREATE INDEX ix_atoms_live      ON memory_atoms(valid_from) WHERE valid_to IS NULL;
CREATE INDEX ix_atoms_scale     ON memory_atoms(scale_level);
CREATE INDEX ix_atoms_role      ON memory_atoms(role);
CREATE INDEX ix_atoms_scope     ON memory_atoms(user_id, agent_id, run_id);
CREATE INDEX ix_atoms_subject   ON memory_atoms(subject, predicate);
CREATE INDEX ix_atoms_hash      ON memory_atoms(hash);
CREATE INDEX ix_atoms_source    ON memory_atoms(source_kind, source_id);
CREATE INDEX ix_atoms_obs_type  ON memory_atoms(obs_type);
CREATE TRIGGER trg_atoms_no_delete BEFORE DELETE ON memory_atoms WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: atoms are archived (valid_to), never deleted'); END;   -- §16.6 gate (schema 2)
CREATE TRIGGER trg_atoms_immutable BEFORE UPDATE OF atom_id, memory_id, version, atom_type, content_json, text, hash, time_anchor, valid_from, scale_level, role, subject, predicate, obs_type, source_kind, source_id, primary_anchor_id, created_at, user_id, agent_id, run_id ON memory_atoms
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: atom content is immutable; create a successor version + revision edge'); END;

CREATE TABLE atom_anchor (
    atom_id     TEXT NOT NULL REFERENCES memory_atoms(atom_id),
    anchor_id   TEXT NOT NULL REFERENCES anchors(anchor_id),
    role        TEXT NOT NULL DEFAULT 'primary' CHECK (role IN ('primary','support','feedback')),
    disputed_at TEXT,
    PRIMARY KEY (atom_id, anchor_id)
);
CREATE INDEX ix_atom_anchor_anchor ON atom_anchor(anchor_id);
CREATE TRIGGER trg_atoms_primary_anchor AFTER INSERT ON memory_atoms
BEGIN INSERT OR IGNORE INTO atom_anchor(atom_id, anchor_id, role) VALUES (NEW.atom_id, NEW.primary_anchor_id, 'primary'); END;
CREATE TRIGGER trg_atom_anchor_no_delete BEFORE DELETE ON atom_anchor WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: atom_anchor rows are disputed, never deleted'); END;   -- §16.6 gate (schema 2)

CREATE TABLE entities (
    entity_id       TEXT PRIMARY KEY,
    normalized_text TEXT NOT NULL,                             -- entity key: lowercase, ws-collapsed; FILE → basename
    canonical_text  TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK (entity_type IN ('PROPER','IDENTIFIER','FILE','QUOTED','TOPIC')),
    aliases_json    TEXT NOT NULL DEFAULT '[]',
    user_id TEXT, agent_id TEXT, run_id TEXT,                  -- run_id stored for provenance but NOT part of identity
    created_at      TEXT NOT NULL
);
CREATE UNIQUE INDEX ux_entities_norm_scope ON entities(normalized_text, COALESCE(user_id,''), COALESCE(agent_id,''));   -- run_id excluded (HMS §3.5 entity axis spans sessions)

CREATE TABLE atom_entity (atom_id TEXT NOT NULL REFERENCES memory_atoms(atom_id), entity_id TEXT NOT NULL REFERENCES entities(entity_id), PRIMARY KEY (atom_id, entity_id));
CREATE INDEX ix_entity_atom ON atom_entity(entity_id);
CREATE TABLE atom_topic  (atom_id TEXT NOT NULL REFERENCES memory_atoms(atom_id), topic TEXT NOT NULL, PRIMARY KEY (atom_id, topic));
CREATE INDEX ix_topic_atom ON atom_topic(topic);

CREATE TABLE atom_vectors (
    atom_id   TEXT PRIMARY KEY REFERENCES memory_atoms(atom_id),
    embedder  TEXT NOT NULL,                                   -- EmbedderBase.identity, e.g. 'hash@64'
    dims      INTEGER NOT NULL CHECK (dims > 0),
    embedding BLOB NOT NULL                                    -- little-endian float32 × dims, L2-normalized
);

-- ===================== Support Graph plane =====================
CREATE TABLE graph_nodes (
    node_id           TEXT PRIMARY KEY REFERENCES memory_atoms(atom_id),   -- node_id == atom_id
    node_kind         TEXT NOT NULL CHECK (node_kind IN ('message','observation','event','entity','ep','interface_ep','skill')),
    ref_id            TEXT NOT NULL,
    scale             TEXT NOT NULL CHECK (scale IN ('L1_micro','L2_meso','L3_macro','state','event','entity','relation')),
    primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),
    SCOPE,
    valid             INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
    invalidated_at    TEXT,
    mentions          INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL
);
CREATE INDEX ix_nodes_scope_kind ON graph_nodes(user_id, agent_id, run_id, node_kind, valid);
CREATE INDEX ix_nodes_ref ON graph_nodes(ref_id);
CREATE TRIGGER trg_nodes_no_delete BEFORE DELETE ON graph_nodes WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: graph_nodes'); END;   -- §16.6 gate (schema 2)

CREATE TABLE graph_edges (
    edge_id                   TEXT PRIMARY KEY,
    src_id                    TEXT NOT NULL REFERENCES graph_nodes(node_id),
    dst_id                    TEXT NOT NULL REFERENCES graph_nodes(node_id),
    edge_type                 TEXT NOT NULL CHECK (edge_type IN ('temporal','causal','revision','conflict','use','distill')),
    label                     TEXT NOT NULL DEFAULT '',
    scale                     TEXT NOT NULL CHECK (scale IN ('L1_micro','L2_meso','L3_macro')),
    is_boundary               INTEGER NOT NULL DEFAULT 0 CHECK (is_boundary IN (0,1)),
    confidence                REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    mentions                  INTEGER NOT NULL DEFAULT 1,
    primary_witness_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),   -- No Witness, No Meaning (structural)
    witness_edge_id           TEXT REFERENCES graph_edges(edge_id),          -- low-level witness edge (projected edges only)
    created_by                TEXT NOT NULL CHECK (created_by IN ('ingest','consolidation','feedback','ops','project')),
    proposer                  TEXT NOT NULL,
    valid                     INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
    invalidated_at            TEXT,
    invalidated_reason        TEXT,
    resolved_at               TEXT,
    resolved_by_anchor_id     TEXT REFERENCES anchors(anchor_id),
    resolution_edge_id        TEXT REFERENCES graph_edges(edge_id),
    SCOPE,
    created_at                TEXT NOT NULL,
    CHECK (src_id <> dst_id),
    CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL),                    -- boundary preservation: projected edge ⇒ low witness edge
    CHECK (edge_type = 'conflict' OR (resolved_at IS NULL AND resolution_edge_id IS NULL AND resolved_by_anchor_id IS NULL)),
    CHECK ((resolved_at IS NULL) = (resolved_by_anchor_id IS NULL)),
    UNIQUE (src_id, dst_id, edge_type, scale)                                   -- MERGE semantics
);
CREATE INDEX ix_edges_src ON graph_edges(src_id, valid);
CREATE INDEX ix_edges_dst ON graph_edges(dst_id, valid);
CREATE INDEX ix_edges_type_scope ON graph_edges(edge_type, user_id, agent_id, run_id, valid);
CREATE INDEX ix_edges_label ON graph_edges(label);
CREATE TRIGGER trg_edges_no_delete BEFORE DELETE ON graph_edges BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: graph_edges'); END;
CREATE TRIGGER trg_edges_immutable BEFORE UPDATE OF edge_id, src_id, dst_id, edge_type, scale, is_boundary, primary_witness_anchor_id, witness_edge_id, created_by, created_at ON graph_edges
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: edge identity/witness columns are immutable'); END;

CREATE TABLE edge_witnesses (edge_id TEXT NOT NULL REFERENCES graph_edges(edge_id), anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id), PRIMARY KEY (edge_id, anchor_id));
CREATE INDEX ix_witness_anchor ON edge_witnesses(anchor_id);
CREATE TRIGGER trg_edge_witnesses_no_delete BEFORE DELETE ON edge_witnesses BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: edge_witnesses'); END;

CREATE TABLE edge_projections (
    high_edge_id TEXT NOT NULL REFERENCES graph_edges(edge_id),
    low_edge_id  TEXT NOT NULL REFERENCES graph_edges(edge_id),
    PRIMARY KEY (high_edge_id, low_edge_id),
    CHECK (high_edge_id <> low_edge_id)
);
CREATE INDEX ix_projections_low ON edge_projections(low_edge_id);
CREATE TRIGGER trg_edges_after_insert AFTER INSERT ON graph_edges
BEGIN
    INSERT OR IGNORE INTO edge_witnesses(edge_id, anchor_id) VALUES (NEW.edge_id, NEW.primary_witness_anchor_id);
    INSERT OR IGNORE INTO edge_projections(high_edge_id, low_edge_id) SELECT NEW.edge_id, NEW.witness_edge_id WHERE NEW.witness_edge_id IS NOT NULL;
END;

CREATE TABLE interface_eps (
    interface_ep_id         TEXT PRIMARY KEY,
    scale                   TEXT NOT NULL CHECK (scale IN ('L2_meso','L3_macro')),
    atom_id                 TEXT NOT NULL UNIQUE REFERENCES memory_atoms(atom_id),   -- its episode atom; node_id == atom_id
    slots_json              TEXT NOT NULL,
    slot_support_json       TEXT NOT NULL,
    support_node_ids_json   TEXT NOT NULL,
    support_edge_ids_json   TEXT NOT NULL DEFAULT '[]',
    boundary_edges_json     TEXT NOT NULL DEFAULT '[]',
    invariants_json         TEXT NOT NULL DEFAULT '[]',
    residuals_json          TEXT NOT NULL DEFAULT '[]',
    evidence_anchor_ids_json TEXT NOT NULL,
    primary_support_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),        -- interface ⇒ support (structural)
    primary_anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),
    coarsen_op_version      TEXT NOT NULL,
    cohesion_score          REAL NOT NULL CHECK (cohesion_score BETWEEN 0 AND 1),
    support_hash            TEXT NOT NULL,                     -- sha256 of sorted support_node_ids (dedup of re-coarsen)
    SCOPE,
    valid                   INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),
    valid_to                TEXT,
    created_at              TEXT NOT NULL
);
CREATE INDEX ix_iep_scope ON interface_eps(user_id, agent_id, run_id, valid);
CREATE INDEX ix_iep_support_hash ON interface_eps(support_hash);
CREATE TABLE interface_ep_support (
    interface_ep_id TEXT NOT NULL REFERENCES interface_eps(interface_ep_id),
    member_kind     TEXT NOT NULL CHECK (member_kind IN ('node','edge','boundary_edge')),
    member_id       TEXT NOT NULL,
    PRIMARY KEY (interface_ep_id, member_kind, member_id)
);
CREATE INDEX ix_iep_support_member ON interface_ep_support(member_kind, member_id);
CREATE TRIGGER trg_iep_after_insert AFTER INSERT ON interface_eps
BEGIN INSERT OR IGNORE INTO interface_ep_support VALUES (NEW.interface_ep_id, 'node', NEW.primary_support_node_id); END;
CREATE TRIGGER trg_iep_no_delete BEFORE DELETE ON interface_eps BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: interface_eps'); END;
CREATE TRIGGER trg_iep_immutable BEFORE UPDATE OF interface_ep_id, scale, atom_id, slots_json, slot_support_json, support_node_ids_json, support_edge_ids_json, boundary_edges_json, evidence_anchor_ids_json, primary_support_node_id, primary_anchor_id, coarsen_op_version, support_hash, created_at ON interface_eps
BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: interface_ep support/boundary/evidence are immutable; re-coarsen creates a successor'); END;

-- ===================== ops plane =====================
CREATE TABLE feedback_events (
    feedback_id     TEXT PRIMARY KEY,
    target_id       TEXT NOT NULL,
    target_kind     TEXT NOT NULL CHECK (target_kind IN ('atom','edge','interface_ep','context_item','anchor','scope')),
    feedback        TEXT NOT NULL CHECK (feedback IN ('state_wrong','state_outdated','state_missing','privacy_violation','next_action_wrong','relationship_not_useful','connection_reason_wrong','evidence_wrong','render_not_like_me','render_like_me','object_important')),
    reason          TEXT,
    correction_json TEXT,
    actor_id        TEXT,
    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),
    actions_json    TEXT NOT NULL DEFAULT '[]',
    SCOPE,
    created_at      TEXT NOT NULL
);
CREATE INDEX ix_feedback_target ON feedback_events(target_id);

CREATE TABLE history (                                        -- Mem0's 10 columns first, in Mem0 order; HMS columns after
    id           TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL,
    old_memory   TEXT,
    new_memory   TEXT,
    event        TEXT NOT NULL CHECK (event IN ('ADD','UPDATE','DELETE','ARCHIVE','RESTORE','FEEDBACK','CONFLICT','ROLLBACK','PURGE')),   -- 'PURGE' since schema 2 (§16.6)
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    is_deleted   INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0,1)),
    actor_id     TEXT,
    role         TEXT,
    anchor_id    TEXT NOT NULL REFERENCES anchors(anchor_id),   -- every mutation is witnessed
    atom_id      TEXT,                                          -- version written/affected by this row
    edge_id      TEXT,
    successor_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    seq          INTEGER NOT NULL UNIQUE                        -- store counter; total order independent of timestamp ties
);
CREATE INDEX ix_history_memory ON history(memory_id, seq);
CREATE TRIGGER trg_history_append_only BEFORE UPDATE ON history BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: history is append-only'); END;
CREATE TRIGGER trg_history_no_delete BEFORE DELETE ON history BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: history is append-only'); END;

CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    label         TEXT,
    anchor_id     TEXT NOT NULL REFERENCES anchors(anchor_id),
    SCOPE,
    snapshot_json TEXT NOT NULL,                                -- {'atom_state': {...}, 'edge_state': {...}, 'iep_state': {...}, 'ep_state': {...}}
    history_seq   INTEGER NOT NULL,
    snapshot_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE transitions (
    transition_id         TEXT PRIMARY KEY,
    source_kind           TEXT NOT NULL CHECK (source_kind IN ('ep','interface_ep','causal','revision','conflict','rollback')),
    source_id             TEXT NOT NULL,
    scale                 TEXT NOT NULL CHECK (scale IN ('L1_micro','L2_meso','L3_macro')),
    s_t_json              TEXT NOT NULL,
    a_t                   TEXT NOT NULL CHECK (length(a_t) > 0),
    s_t1_json             TEXT NOT NULL,
    r_t                   TEXT NOT NULL CHECK (r_t IN ('positive','negative','neutral','unknown')),
    kappa_t_json          TEXT NOT NULL,
    w_t_json              TEXT NOT NULL,
    w_t_primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),   -- w_t non-empty, structurally
    time_t                TEXT NOT NULL,
    time_t1               TEXT NOT NULL,
    SCOPE,
    created_at            TEXT NOT NULL,
    UNIQUE (source_kind, source_id)                                     -- idempotent export
);
CREATE INDEX ix_transitions_scope_t ON transitions(user_id, agent_id, run_id, time_t1);

CREATE TABLE compiled_states (                                         -- consolidate step (7) compile_state snapshots (HMS §3.9 / §5.2 "optional persistence")
    compiled_id   TEXT PRIMARY KEY,                                    -- prefix 'cst'
    scope_key     TEXT NOT NULL,
    SCOPE,
    as_of         TEXT NOT NULL,
    mode          TEXT NOT NULL,
    card          TEXT NOT NULL,                                       -- ContextIR.to_text()
    context_json  TEXT NOT NULL,                                       -- ContextIR.model_dump_json()
    hash          TEXT NOT NULL,                                       -- sha256(card)
    used_tokens   INTEGER NOT NULL,
    anchor_id     TEXT NOT NULL REFERENCES anchors(anchor_id),         -- the scheduler anchor
    created_at    TEXT NOT NULL
);
CREATE INDEX ix_compiled_scope ON compiled_states(scope_key, created_at);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);         -- 'history_seq', 'last_consolidation:<scope_key>', 'work_state:<scope_key>', 'schema_created_at', 'purge_in_progress' (only inside StateStore.purge_rows()); generated FIRST since schema 2 because the gated triggers read it
```

### §7.1 Migrations

```python
SCHEMA_VERSION = 3
HISTORY_EVENTS_V1 = tuple(e for e in HistoryEvent.values() if e != "PURGE")
PURGE_FLAG_KEY = "purge_in_progress"; PURGE_GATE = "WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') "
PURGE_GATED_TRIGGERS = ((name, table, message) × 7)          # trg_events/anchors/obs/eps/atoms/atom_anchor/nodes_no_delete
def build_ddl(version: int = SCHEMA_VERSION) -> list[str]  # version 1 = the 0.1 schema (ungated triggers, history without 'PURGE'); 2 = the schema printed above (83 statements); 3 = 2 + FTS_DDL (84 statements)
def upgrade_2() -> list[str]                               # 21 statements: DROP TRIGGER IF EXISTS + gated CREATE TRIGGER ×7; ALTER TABLE history RENAME TO history_v1; CREATE TABLE history (new CHECK); INSERT INTO history(<16 cols>) SELECT … FROM history_v1 ORDER BY seq; DROP TABLE history_v1; re-create ix_history_memory and the two history triggers
FTS_DDL = "CREATE VIRTUAL TABLE atom_fts USING fts5(atom_id UNINDEXED, tokens, tokenize = 'unicode61')"   # the keyword channel: one row per NON-archived atom version, tokens = fts_tokens(text, topic_tags, entity_keys) (content tokens stemmed like utils.scoring.atom_keyword_tokens, path basenames, topics, entity keys + their tokens; CJK bigrams are space-separated so unicode61 keeps them)
def upgrade_3() -> list[str]                               # [FTS_DDL]; skipped (with a warning) when the linked SQLite has no FTS5 module
def fts_tokens(text: str, topic_tags: Iterable[str] = (), entity_keys: Iterable[str] = ()) -> str
def backfill_fts(conn) -> int                              # INSERT INTO atom_fts for every memory_atoms row with archived_reason IS NULL (topics + entity keys joined in Python)
MIGRATIONS: dict[int, list[str]] = {1: build_ddl(1), 2: upgrade_2(), 3: upgrade_3()}   # the ladder upgrades EXISTING files (a version-1 file runs 2 then 3); a version-0 database is created directly from build_ddl(SCHEMA_VERSION) in one transaction; both give the same sqlite_master rows (type, name, tbl_name, sql) (test_fresh_database_is_created_directly_at_schema_version, test_purge_gate_and_schema_migration_2, test_schema_v3_backfills_fts_from_a_v2_file)
POST_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], object]] = {3: backfill_fts}   # Python steps run after a version's statements inside the SAME migration transaction (never on a fresh database, which has no rows)

def apply_migrations(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    """Apply pending migrations statement by statement under BEGIN IMMEDIATE, gated by PRAGMA user_version.
    在 BEGIN IMMEDIATE 下逐条应用待执行迁移，由 PRAGMA user_version 控制。"""
    # current > SCHEMA_VERSION → StoreError(STORE_002, suggestion="pip install -U hms-elite")
    # for version in range(current+1, SCHEMA_VERSION+1): BEGIN IMMEDIATE; execute each stmt (never executescript); [version == 1 and created_at: INSERT OR REPLACE meta('schema_created_at', created_at) — the stamp rides in the migration transaction]; PRAGMA user_version = version; COMMIT; on error ROLLBACK + raise StoreError(STORE_001) from e
def clone_schema(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    # fresh private ':memory:' connections only (user_version 0): backup the process-wide schema template (built once per pid by apply_migrations on ':memory:', guarded by a lock, never written afterwards) into conn, then write meta.schema_created_at; a non-fresh conn or a backup failure falls back to apply_migrations; result identical to the DDL path (sqlite_master rows, root pages, user_version)
def fts5_available() -> bool             # CREATE VIRTUAL TABLE … USING fts5 on a scratch ':memory:' connection; probed once per process
def detect_capabilities(conn) -> dict:   # {'sqlite_version': str, 'json1': bool (SELECT json_extract('{"a":1}','$.a') works; probed once per process, it is a property of the linked library), 'fts5': bool (fts5_available() AND atom_fts exists in sqlite_master), 'wal': bool, 'schema_version': int}
def drop_all(conn) -> None               # DROP TABLE for the virtual table (shadow tables follow), then every base table (reverse order) — the only DELETE-like path; used by StateStore.reset()
TABLES: tuple[str, ...] = ('source_events','anchors','observations','eps','memory_atoms','atom_anchor','entities','atom_entity','atom_topic','atom_vectors','graph_nodes','graph_edges','edge_witnesses','edge_projections','interface_eps','interface_ep_support','feedback_events','history','checkpoints','transitions','compiled_states','meta')   # 22 base tables
VIRTUAL_TABLES: tuple[str, ...] = ('atom_fts',)   # 1 virtual table (FTS5); its shadow tables FTS_SHADOW_TABLES = atom_fts_data / _idx / _content / _docsize / _config belong to SQLite and are listed by name only — TABLES, stats() and TableTrace cover the 22 base tables; sqlite_master holds 22 + 1 + 5 rows of type 'table' when FTS5 is available
```

Rules: migrations are append-only (`{2: [...]}` later); never a column-set-diff copy; SQLite floor 3.31 (expression UNIQUE index with COALESCE, partial indexes, RAISE triggers); `json1` optional (§13.1 fallback).

---

## §8 `StateStore` (`hms_elite/memory/storage.py`, group A) — the shared seam

Module docstring: `"""One SQLite connection, one lock, typed writers for every table; the only place SQL is executed.\n单一 SQLite 连接、单一锁、每张表的类型化写入器；唯一执行 SQL 的地方。"""` Purpose: guarded persistence for all planes | 全平面受控持久化. Input: pydantic models, scopes, ids | 模型、范围、id. Output: models/rows/ids | 模型/行/id. Effect: writes the DB inside `transaction()`; raw cursor is private | 在事务内写库；游标私有.

Behavioural notes: every public write must run inside `with store.transaction():` (nested calls re-enter; the outermost does `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` under `threading.Lock`); calling a writer outside a transaction opens and commits one implicitly. `sqlite3.Error` → `StoreError(STORE_001) from e` after rollback; trigger `RAISE` messages starting with `HMS_INVARIANT` → `StoreError(STORE_003)`; witness pre-checks per §3 (`WitnessMissingError` before any SQL). `_execute` runs the statement **and fetches every row inside the lock**, returning a materialized row list that offers `fetchone` / `fetchall` / iteration / `rowcount` / `lastrowid`; no live `sqlite3.Cursor` is ever read outside the lock, so concurrent readers of one instance (the `check_same_thread=False` design) cannot receive another statement's rows through CPython's per-connection statement cache. `trace: TableTrace` records table names touched by every statement (`trace.reset()`, `trace.tables() -> set[str]`, `trace.enabled`). All `list_*` readers apply `scope` with `Scope.matches` semantics in SQL (`col = ?` for every set key, no predicate for an unset key, `1=1` when none is set — never the `(col = ? OR ? IS NULL)` form, which hides the bound value from the planner and keeps `ix_atoms_scope` unused), i.e. a filter with only `user_id` matches every run. Every scope-filtered SELECT carries a deterministic `ORDER BY`, so the plan never changes results. **Hydration**: every `MemoryAtom` returned by any reader (`get_atom`, `get_atoms`, `axis_select`, `resolve_head`, `list_versions`, `find_atoms_by_hash`, `revision_candidates`, ...) has `entity_ids` = the entity keys (`entities.normalized_text`) joined through `atom_entity`, sorted; `topic_tags` = `atom_topic` rows in insertion order; `anchor_ids` = `atom_anchor_ids(atom_id)` (undisputed, primary first); the three links are read by ONE statement per 500-id chunk (`WITH ids(atom_id) AS (VALUES …)` joined to `atom_entity`/`entities`, `atom_topic`, `atom_anchor`). `insert_atom` returns that same hydrated form rebuilt from the write itself (JSON columns round-tripped through `canonical_json`, links from the inserted lists) instead of re-reading the row; `test_insert_atom_mirrors_read_path` pins `insert_atom(a) == get_atom(a.atom_id)`. Witness pre-checks are memoized per outermost transaction (`_verified_anchors`: an id proven by `get_anchors` or written by `get_or_create_anchor` is not re-selected until COMMIT/ROLLBACK clears the memo; anchors are never deleted, so the memo cannot go stale inside its transaction). **Liveness predicate** (`LIVE_SQL`, one constant used by every `valid_only`/validity clause): `archived_reason IS NULL AND valid_from <= :as_of AND (valid_to IS NULL OR valid_to > :as_of)` with `as_of = now()` unless the caller passes one; `include_expired=True` drops the whole predicate. Ids passed in (`event_id`, `atom_id`, `memory_id`, `ep_id`, ...) are preserved when non-empty (used by `import_state_package`); `""` → `new_id(prefix)`.

```python
class TableTrace:
    enabled: bool = True
    def record(self, sql: str) -> None; def reset(self) -> None; def tables(self) -> set[str]

class StateStore:
    def __init__(self, path: str = ":memory:", *, now: Callable[[], str] | None = None, id_factory: Callable[[str], str] | None = None,
                 busy_timeout_ms: int = 5000, journal_mode: str = "wal", checkpoint_on_close: bool = True) -> None
        # ':memory:'/'' → clone_schema; anything else → apply_migrations, then PRAGMA journal_mode = WAL (file DBs, journal_mode == 'wal'); created_at = now() rides in the migration; checkpoint_on_close=False + file DB → SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE set right before close() when the runtime has it
    path: str; capabilities: dict; trace: TableTrace
    def now(self) -> str                                     # injected clock or timeutil.now_iso
    def new_id(self, prefix: str) -> str                     # injected id_factory or f"{prefix}_{token_hex(6)}"
    def next_seq(self, key: str = "history_seq") -> int      # meta counter, +1, inside the current txn
    def transaction(self) -> ContextManager[None]
    @property in_transaction(self) -> bool
    def reset(self) -> None                                  # drop_all + apply_migrations; meta cleared
    def close(self) -> None                                  # idempotent
    def stats(self, scope: Scope | None = None) -> dict[str, int]   # row counts per table (scope-filtered where the table has scope cols)

    # ---- events / anchors ----
    def insert_source_event(self, ev: SourceEvent) -> tuple[SourceEvent, bool]      # assigns event_id/ingested_at; INSERT OR IGNORE on ux_events_identity → returns existing row, dedup=True
    def get_source_event(self, event_id: str) -> SourceEvent | None
    def list_events(self, scope: Scope, *, object_types: list[str] | None = None, since: str | None = None, until: str | None = None, limit: int | None = None, order: str = "ASC") -> list[SourceEvent]
    def last_events(self, scope: Scope, *, object_types: list[str] | None = None, limit: int = 10) -> list[SourceEvent]   # newest `limit`, returned chronological
    def latest_event(self, scope: Scope, *, object_types: list[str] | None = None) -> SourceEvent | None   # §6.7: ORDER BY timestamp DESC, rowid DESC LIMIT 1 — the last WRITTEN among same-timestamp events (last_events tie-breaks on the random event_id)
    def touch_scope(self, scope: Scope) -> None                                       # §6.7: attribute the open transaction to a scope (bumped at COMMIT only if the transaction wrote versioned rows; at once outside a transaction); called by insert_source_event (stored rows only), insert_atom and the scheduler phases
    def unversioned(self) -> Iterator[None]                                           # §6.7: thread-local block whose write statements move no write version — internal events + anchors (system_event), the checkpoints row, the scheduler's meta rows and compiled-state archives
    def scope_version(self, scope: Scope) -> tuple                                    # §6.7: (write_generation, data_version(), sorted (key, n) of the recorded scopes covering / covered by scope) — the HMS.context() memo key
    def get_or_create_anchor(self, ev: SourceEvent) -> Anchor                        # UNIQUE(source_event_id); permission_scope = ev.visibility; source_ref = ev.raw_ref
    def get_anchor(self, anchor_id: str) -> Anchor | None
    def get_anchors(self, anchor_ids: Iterable[str]) -> list[Anchor]                  # order preserved, unknown ids skipped
    def anchor_for_event(self, event_id: str) -> Anchor | None
    def revoke_anchor(self, anchor_id: str, *, at: str) -> None
    def set_anchor_permission(self, anchor_id: str, visibility: str) -> None
    def events_without_atoms(self, scope: Scope, *, object_types: list[str]) -> list[SourceEvent]   # anchor has no atom_anchor row (crash recovery / collect)
    def last_open_action(self, scope: Scope, *, run_id: str | None, before: str, max_gap_hours: float) -> tuple[SourceEvent, Anchor] | None
        # newest tool_call/file_diff event in the scope (run_id = ? when given, else the scope's run semantics) with timestamp <= before AND age_days(timestamp, before) * 24 <= max_gap_hours
        # whose anchor_id appears in NO eps.anchor_ids_json (json_each when json1, else Python filter); ORDER BY timestamp DESC, event_id DESC LIMIT 1

    # ---- observations / eps ----
    def insert_observation(self, obs: Observation) -> Observation
    def get_observation(self, obs_id: str) -> Observation | None
    def set_observation_atom(self, obs_id: str, atom_id: str) -> None
    def set_observation_valid_to(self, obs_id: str, valid_to: str | None) -> None      # mirror of the observation atom's valid_to (called by archive_atom/restore_atom)
    def insert_ep(self, ep: EP) -> EP                                                 # valid_from defaults to time_anchor
    def get_ep(self, ep_id: str) -> EP | None
    def get_eps(self, ep_ids: Iterable[str]) -> list[EP]
    def set_ep_atom(self, ep_id: str, atom_id: str) -> None
    def list_eps(self, scope: Scope, *, run_id: str | None = None, valid_only: bool = True) -> list[EP]   # ORDER BY time_anchor, ep_id; valid_only ⇔ eps.valid_to IS NULL AND the EP's graph_nodes row (via eps.atom_id) has valid = 1
    def last_ep(self, scope: Scope, *, run_id: str | None) -> EP | None                # newest valid EP (same predicate) by (time_anchor DESC, created_at DESC, ep_id DESC)
    def archive_ep(self, ep_id: str, *, at: str) -> None                              # valid_to = at (called by archive_atom on an ep atom; also public)
    def restore_ep(self, ep_id: str) -> None                                          # valid_to = NULL
    def scope_ep_state(self, scope: Scope) -> dict[str, str | None]                   # every EP in scope → eps.valid_to (checkpoint ep_state)

    # ---- atoms (State Index) ----
    def insert_atom(self, atom: MemoryAtom, *, entity_keys: list[EntityKey] = (), topics: list[str] = ()) -> MemoryAtom
        # EntityKey = tuple[str, str, str, tuple[str, ...]] = (key, canonical, entity_type, aliases) (§12.5)
        # assigns atom_id/memory_id (new lineage when "")/created_at/updated_at/valid_from; writes memory_atoms, atom_anchor for anchor_ids[1:] (role 'support', INSERT OR IGNORE),
        # entities via get_or_create_entity(scope, normalized=key, canonical=canonical, entity_type=entity_type, aliases=aliases) + atom_entity, atom_topic,
        # and the graph_nodes row (node_id == atom_id, node_kind from source_kind); raises WitnessMissingError if anchor_ids is empty or an id is unknown (§3 pre-check)
    def get_atom(self, atom_id: str) -> MemoryAtom | None
    def get_atoms(self, atom_ids: Iterable[str]) -> list[MemoryAtom]
    def resolve_head(self, memory_id: str) -> MemoryAtom | None                       # the version with archived_reason IS NULL (at most one — ux_atoms_head); an expiring head (future valid_to) IS returned
    def latest_version(self, memory_id: str) -> MemoryAtom | None                     # highest version regardless of validity
    def list_versions(self, memory_id: str) -> list[MemoryAtom]                        # ORDER BY version
    def archive_atom(self, atom_id: str, *, reason: str, at: str, valid_to: str | None = None) -> None   # the source cascade (eps / observations / interface_eps) is skipped while another non-archived atom cites the same (source_kind, source_id)
    def reopen_source(self, atom_id: str) -> None                    # re-open the source row cited by a live atom (revise_atom calls it after creating a successor that shares its predecessor's source)
    def shared_source_atom_ids(self, atom_ids: Iterable[str]) -> set[str]   # ids whose (source_kind, source_id) is also cited by an atom outside the list (purge protection `source_shared`)
    @property
    def lock(self) -> threading.RLock                                # the store lock, FIRST in the lock order (index providers take it before their own lock)
        # valid_to = valid_to or at; archived_reason = reason; updated_at = at; graph_nodes.valid = 0, invalidated_at = at;
        # CASCADE by source_kind: 'ep' → archive_ep(source_id, at=valid_to); 'observation' → set_observation_valid_to(source_id, valid_to); 'interface_ep' → interface_eps.valid = 0, valid_to (without re-archiving the atom)
    def restore_atom(self, atom_id: str, *, at: str) -> None
        # valid_to = NULL, archived_reason = NULL, updated_at = at, node valid = 1, invalidated_at = NULL; CASCADE: 'ep' → restore_ep; 'observation' → set_observation_valid_to(None); 'interface_ep' → interface_eps.valid = 1, valid_to = NULL
    def set_valid_to(self, atom_id: str, valid_to: str | None, *, at: str) -> None    # explicit NULL allowed (expiration_date=None)
    def set_atom_confidence(self, atom_id: str, value: float, *, at: str) -> None
    def set_atom_visibility(self, atom_id: str, visibility: str, *, at: str) -> None
    def set_user_meta(self, atom_id: str, user_meta: dict, *, at: str) -> None
    def add_atom_anchor(self, atom_id: str, anchor_id: str, *, role: str = "support") -> bool   # INSERT OR IGNORE on the (atom_id, anchor_id) PK; returns True iff a row was inserted (an anchor that is already the primary witness adds nothing)
    def dispute_atom_anchor(self, atom_id: str, anchor_id: str, *, at: str) -> None
    def atom_anchor_ids(self, atom_id: str, *, include_disputed: bool = False) -> list[str]   # primary first, then insertion order
    def atoms_for_anchor(self, anchor_id: str) -> list[str]
    def axis_select(self, scope: Scope, *, where_sql: str = "", params: Sequence = (), joins: Sequence[str] = (), as_of: str | None = None,
                    include_expired: bool = False, scale_constraint: Sequence[str] | None = None, atom_types: Sequence[str] | None = None,
                    order: str = "time_anchor DESC, created_at DESC, atom_id ASC", limit: int | None = None,
                    atom_ids: Sequence[str] | None = None, index_hint: str | None = None) -> list[MemoryAtom]
        # WHERE scope AND validity(as_of) [unless include_expired] AND (where_sql) — validity is ALWAYS a WHERE clause, never a post-filter
        # atom_ids (the pool hydration): `WITH ids(atom_id) AS (VALUES …) SELECT … FROM ids CROSS JOIN memory_atoms ON …` — one primary-key lookup per id, never a scope scan; index_hint: `FROM memory_atoms INDEXED BY <name>` (schema.TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX, chosen by mvmi from the level's gate; a name that is not an identifier → VALIDATION_000), ignored with atom_ids
    def axis_ids(self, scope, *, where_sql="", params=(), joins=(), as_of=None, include_expired=False, scale_constraint=None, atom_types=None,
                 order="time_anchor DESC, created_at DESC, atom_id ASC", limit=None, index_hint=None) -> list[str]     # the same gate, ids only (no hydration): the gated set G of the retrieval pool; order=None emits NO ORDER BY (index order, a LIMIT stops the scan early — the pool probe and the semantic channel's G)
    def count_axis(self, scope, *, where_sql="", params=(), joins=(), as_of=None, include_expired=False, scale_constraint=None, atom_types=None, index_hint=None) -> int   # |G| (COUNT(DISTINCT atom_id) with joins)
    def keyword_candidates(self, scope: Scope, tokens: Sequence[str], *, as_of: str | None = None, include_expired: bool = False, limit: int,
                           where_sql: str = "", params: Sequence = (), joins: Sequence[str] = (), scale_constraint=None, atom_types=None, index_hint=None) -> list[tuple[str, float]]
        # keyword recall channel INSIDE the level gate (the axis_select gate arguments): two FTS5 phases, strict then loose — `MATCH '"t1" AND "t2" …'` (only with >= 2 tokens) then `MATCH '"t1" OR "t2" …'` fills the remainder; each phase ranks inside atom_fts first (`ORDER BY rank, atom_id LIMIT window OFFSET …`, window = max(4·need, 64), doubling) and re-applies the gate to the ranked window only through a VALUES-CTE CROSS JOIN (primary-key lookups), growing the window until `limit` gated rows are found or the phase is exhausted; score = -bm25 (higher is better), rows in phase order; capabilities['fts5'] False → bounded LIKE fallback with the same gate (score = matched-token count, then DEFAULT_ATOM_ORDER); archived versions carry no atom_fts row; index_hint accepted and ignored
    def entity_candidates(self, scope: Scope, entity_keys: Sequence[str], *, as_of: str | None = None, include_expired: bool = False, limit: int | None,
                          where_sql: str = "", params: Sequence = (), joins: Sequence[str] = (), scale_constraint=None, atom_types=None, index_hint=None) -> list[str]
        # entity recall channel INSIDE the level gate: entity ids resolved first (resolve_entities: normalized_text IN keys, COALESCE(user_id/agent_id) identity) → `FROM atom_entity ae CROSS JOIN memory_atoms ON …` (ae drives through ix_entity_atom, memory_atoms by primary key — the planner would otherwise scan the scope) WHERE ae.entity_id IN (…) AND gate ORDER BY time_anchor DESC, atom_id ASC; no DISTINCT: a limit over-fetches ×len(entity_ids) and Python keeps the first occurrence; limit=None → every gated linked atom in that order (the gated set G of a hard entity gate); index_hint accepted and ignored
    def recent_candidates(self, scope: Scope, *, as_of: str | None = None, include_expired: bool = False, limit: int, scale_in: Sequence[str] | None = None, obs_type_in: Sequence[str] | None = None) -> list[str]
        # recency recall channel: scope + validity [+ scale_level IN] [+ obs_type IN] ORDER BY DEFAULT_ATOM_ORDER LIMIT K_rec (ix_atoms_scope); every channel is an index SEARCH, never a `SCAN memory_atoms` (test_fts_lifecycle_channels_and_query_plans asserts the plans)
    def atom_attrs(self, atom_ids: Iterable[str]) -> dict[str, dict]                 # {atom_id: INDEX_ATTR_KEYS scalars (§10.3, atom_type included), NULL strings as "", archived: bool}; the Milvus provider reads it at upsert when no attrs are given
    attr_listeners: list[Callable[[str, dict], None]]                              # `callback(atom_id, attrs)` for every atom whose validity / archived state changed (archive_atom / restore_atom / set_valid_to); `attrs` is the FULL `atom_attrs(...)` snapshot read inside the transaction just before COMMIT (never a partial diff, so a provider merge is idempotent against SQLite; one callback per changed atom per commit — intermediate states inside one transaction are not replayed); dispatched after COMMIT outside the store lock in commit order (a ticket per committing transaction + a Condition: a later commit's snapshot can never be overtaken by an earlier one; a ROLLBACK discards them); a raising listener is logged, never raised; HMS.__init__ registers index.update_attrs once (skipped when the provider already subscribed itself, as SQLiteIndex does at construction) and HMS.close() removes it
    data_generation: int; rebuild_generation: int                                  # write generations: data_generation advances on every non-SELECT statement of this connection touching memory_atoms / atom_vectors; rebuild_generation on the subset that can remove rows (DELETE / DROP, any ROLLBACK, reset()); they key the SQLite index cache (§10.3)
    def data_version(self) -> int                                                   # `PRAGMA data_version`: changes only when ANOTHER connection commits to the file (own writes do not move it)
    def load_vector_rows(self, after_rowid: int = 0) -> list[tuple[int, str, int, bytes, dict]]   # atom_vectors ⋈ memory_atoms rows with rowid > after_rowid in rowid order: (rowid, atom_id, dims, '<f4' blob, INDEX_ATTR_KEYS attrs) — the incremental feed of the SQLite index cache (INSERT OR REPLACE gives a replaced vector a new rowid, so it reappears)
    # Every `UPDATE memory_atoms` changes only valid_to / archived_reason (through _queue_attrs) or non-gate columns (confidence, visibility, user_meta_json, updated_at): the scalar gates of an atom are otherwise immutable, which is what lets the index cache be extended by rowid instead of rebuilt (test_memory_atoms_updates_stay_listener_covered)
    def count_atoms(self, scope: Scope, *, valid_only: bool = True) -> int
    def find_atoms_by_hash(self, scope: Scope, md5: str, *, valid_only: bool = True) -> list[MemoryAtom]
    def revision_candidates(self, scope: Scope, *, subject: str, predicate: str, obs_type: str, topics: Sequence[str], objects: Sequence[str]) -> list[MemoryAtom]
        # valid state/event atoms in scope with same subject AND obs_type AND (same predicate OR topic overlap OR object overlap); ORDER BY time_anchor DESC
    def scope_atom_state(self, scope: Scope) -> dict[str, dict[str, str | None]]     # every atom in scope → {'valid_to', 'archived_reason'}
    def valid_ids(self, scope: Scope, *, as_of: str | None = None) -> dict[str, list[str]]   # {'atoms', 'eps', 'edges', 'interface_eps'} live ids (LIVE_SQL), sorted; the ONLY candidate-id helper (ingest neighbourhood uses ['atoms'])

    # ---- entities / topics / vectors ----
    def get_or_create_entity(self, scope: Scope, *, normalized: str, canonical: str, entity_type: str, aliases: Sequence[str] = ()) -> str   # identity (normalized, user_id, agent_id); appends new aliases
    def resolve_entity(self, scope: Scope, normalized: str) -> str | None
    def resolve_entities(self, scope: Scope, normalized: Sequence[str]) -> dict[str, str]
    def entity_degree(self, entity_id: str) -> int                                     # count(atom_entity) over ALL linked atoms (entity atoms included)
    def entities_in_scope(self, scope: Scope) -> list[dict]                            # entities linked (atom_entity) to ≥1 live atom matching scope; [{'entity_id','normalized_text','canonical_text','entity_type','aliases','user_id','agent_id'}] ORDER BY normalized_text, entity_id
    def entity_linked_atoms(self, entity_id: str, *, valid_only: bool = True, exclude_entity_atoms: bool = True) -> list[str]   # atom ids, sorted
    def entities_for_atom(self, atom_id: str) -> list[dict]                            # [{'entity_id','normalized_text','canonical_text','entity_type'}]
    def link_atom_entity(self, atom_id: str, entity_id: str) -> None
    def add_atom_topic(self, atom_id: str, topic: str) -> None
    def upsert_vector(self, atom_id: str, vector: Sequence[float], *, embedder: str) -> None
    def load_vectors(self, atom_ids: Iterable[str] | None) -> dict[str, list[float]]  # None → all
    def load_vector_blobs(self, atom_ids: Iterable[str] | None) -> list[tuple[str, int, bytes]]   # raw '<f4' BLOBs, first-seen order (None → all by atom_id); SQLiteIndex.search builds one matrix with numpy.frombuffer
    def delete_vector(self, atom_id: str) -> None
    def count_vectors(self) -> int
    def atoms_without_vectors(self, scope: Scope | None, *, embedder: str) -> list[str]   # missing row OR different embedder identity
    def clear_vectors(self) -> None

    # ---- graph ----
    def insert_node(self, node: GraphNode) -> str                                      # MERGE on node_id (mentions += 1)
    def get_node(self, node_id: str) -> GraphNode | None
    def get_nodes(self, node_ids: Iterable[str]) -> list[GraphNode]
    def list_nodes(self, scope: Scope, *, kinds: Sequence[str] | None = None, valid_only: bool = True, scale: str | None = None) -> list[GraphNode]
    def set_node_valid(self, node_id: str, valid: bool, *, at: str) -> None
    def bump_mentions(self, node_id: str, n: int = 1) -> None
    def insert_edge(self, edge: GraphEdge | dict) -> str
        # a dict is checked BEFORE model construction: `if not edge.get('witness_anchor_ids'): raise WitnessMissingError(details={'object': 'edge', 'id': None, 'missing': []})`,
        # then get_anchors pre-check (§3), then GraphEdge(**edge) (scope defaults to the src node's scope, created_at to now()); a GraphEdge instance is re-checked the same way (model_construct'ed edges included);
        # MERGE on (src,dst,type,scale): mentions += 1, confidence = max, witnesses unioned (add_edge_witness), label kept; returns the (existing or new) edge_id; callers detect "new" by comparing with find_edge() before the call
    def set_edge_label(self, edge_id: str, label: str, *, at: str) -> None           # label is a mutable column (not in trg_edges_immutable)
    def get_edge(self, edge_id: str) -> GraphEdge | None
    def get_edges(self, edge_ids: Iterable[str]) -> list[GraphEdge]
    def find_edge(self, src_id: str, dst_id: str, edge_type: str, scale: str = "L1_micro") -> GraphEdge | None
    def edges_touching(self, node_ids: Iterable[str], *, valid_only: bool = True, edge_types: Sequence[str] | None = None, direction: str = "both") -> list[GraphEdge]   # ORDER BY created_at, edge_id
    def edges_between(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[GraphEdge]   # both endpoints in set
    def list_edges(self, scope: Scope, *, edge_types: Sequence[str] | None = None, valid_only: bool = True, since: str | None = None, scale: str | None = None, label_prefix: str | None = None, include_boundary: bool = True) -> list[GraphEdge]
    def invalidate_edge(self, edge_id: str, *, reason: str, at: str) -> None
    def revalidate_edge(self, edge_id: str, *, at: str) -> None
    def resolve_conflict(self, edge_id: str, *, at: str, by_anchor_id: str, resolution_edge_id: str | None = None) -> None
    def edge_witnesses(self, edge_id: str) -> list[str]                                # primary first
    def add_edge_witness(self, edge_id: str, anchor_id: str) -> None
    def projections_of(self, high_edge_id: str) -> list[str]
    def projections_from(self, low_edge_id: str) -> list[str]
    def add_projection(self, high_edge_id: str, low_edge_id: str) -> None
    def insert_interface_ep(self, iep: InterfaceEP) -> InterfaceEP                     # writes interface_eps + interface_ep_support (node/edge/boundary_edge members); atom must already exist
    def get_interface_ep(self, iep_id: str) -> InterfaceEP | None
    def find_interface_ep_by_support_hash(self, scope: Scope, support_hash: str) -> InterfaceEP | None
    def list_interface_eps(self, scope: Scope, *, valid_only: bool = True, scale: str | None = None) -> list[InterfaceEP]
    def invalidate_interface_ep(self, iep_id: str, *, at: str) -> None                 # valid=0, valid_to=at, atom archived (reason 'revised'), node valid=0
    def revalidate_interface_ep(self, iep_id: str, *, at: str) -> None
    def support_members(self, iep_id: str) -> dict[str, list[str]]                    # {'node': [...], 'edge': [...], 'boundary_edge': [...]}
    def interface_eps_containing(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[InterfaceEP]

    # ---- ops ----
    def insert_feedback(self, fb: FeedbackEvent) -> str
    def list_feedback(self, target_id: str) -> list[FeedbackEvent]
    def add_history(self, *, memory_id: str, old_memory: str | None, new_memory: str | None, event: str, created_at: str, updated_at: str,
                    is_deleted: int = 0, actor_id: str | None = None, role: str | None = None, anchor_id: str, atom_id: str | None = None,
                    edge_id: str | None = None, successor_id: str | None = None, details: dict | None = None) -> str   # id = new_id('hst'); seq = next_seq()
    def get_history(self, memory_id: str) -> list[dict]                                # ORDER BY seq; Mem0's 10 keys + 'anchor_id'; is_deleted → bool
    def history_seq(self) -> int
    def insert_checkpoint(self, cp: Checkpoint) -> str
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None
    def list_checkpoints(self, scope: Scope) -> list[Checkpoint]
    def upsert_transition(self, tr: Transition) -> str                                 # INSERT OR IGNORE on (source_kind, source_id); returns existing id when ignored
    def list_transitions(self, scope: Scope, *, since: str | None = None, scale: str | None = None, kinds: Sequence[str] | None = None) -> list[Transition]   # ORDER BY time_t1, CASE source_kind (hms_elite.enums.KIND_ORDER), source_id
    def insert_compiled_state(self, *, scope: Scope, as_of: str, mode: str, card: str, context_json: str, used_tokens: int, anchor_id: str) -> str   # id prefix 'cst'; hash = sha256(card)
    def last_compiled_state(self, scope: Scope) -> dict | None                        # newest row for scope.key() → {'compiled_id','as_of','hash','used_tokens','created_at','card'}
    def set_meta(self, key: str, value: str) -> None
    def get_meta(self, key: str, default: str | None = None) -> str | None
    # ---- retention (§16.6) — readers are scope/id filters; purge_rows() is the ONLY physical DELETE path of state rows
    def purge_candidates(self, scope: Scope, *, valid_to_before: str) -> list[MemoryAtom]   # archived_reason IS NOT NULL AND valid_to < bound, Scope.matches, atom_id ASC
    def edge_endpoint_ids(self, atom_ids, *, valid_only: bool = False) -> set[str]         # ids that are src/dst of any edge (valid or not)
    def interface_ep_member_ids(self, atom_ids) -> set[str]                                 # interface_eps.atom_id ∪ primary_support_node_id ∪ interface_ep_support(node)
    def checkpoint_live_atom_ids(self, scope: Scope | None = None) -> set[str]              # atom ids any snapshot records with archived_reason None
    def ep_reference_ids(self, ep_ids, *, excluding_atoms=()) -> set[str]                   # EP ids cited by other eps.uses_ep_ids_json / surviving memory_atoms.source_ep_ids_json / transitions(source_kind='ep')
    def anchor_reference_count(self, anchor_id: str) -> int                                 # sum over 14 witness columns + 4 JSON id lists
    def purge_rows(self, atom_ids, *, evidence: bool = False) -> dict                       # one transaction: meta.purge_in_progress='1' → DELETE atom_vectors/atom_topic/atom_entity/atom_anchor/graph_nodes/memory_atoms (+ the atom's own observations/eps source row) [+ orphaned anchors + source_events when evidence] → DELETE the flag; {'atoms','observations','eps','anchors','events','nodes','vectors','links'}
    def vacuum(self) -> dict                                                                # VACUUM outside any transaction (STORE_001 inside); file DBs then wal_checkpoint(TRUNCATE); {'path','before_bytes','after_bytes','freed_bytes'} = page_count*page_size
```

Node kind mapping used by `insert_atom`: `source_kind → node_kind`: message→message, observation→observation, event→event, entity→entity, ep→ep, interface_ep→interface_ep, skill→skill, procedural→ep, feedback→observation. `graph_nodes.scale = atom.scale_level`. Id prefixes (§0.1) additionally include `cst` (compiled_states).

---

## §9 Foundation utilities (group A)

### §9.1 `hms_elite/timeutil.py`

Module docstring: `"""Canonical UTC instants: one clock format for every table.\n规范 UTC 时间：所有表使用同一格式。"""`

- `now_iso() -> str` — current UTC instant, 32-char fixed width | 当前 UTC 时间. Effect: none.
- `to_utc_iso(value: str | datetime | date | int | float) -> str` — normalize to fixed-width UTC ISO | 归一化. In: ISO string (`Z`, `±HH:MM`, naive → UTC, date-only → `T00:00:00`), datetime/date, epoch seconds | 各种时间输入. Out: 32-char string | 定宽字符串. Effect: none. Notes: garbage → `ValidationError(VALIDATION_004)`.
- `parse_iso(value: str) -> datetime` — aware UTC datetime | 解析.
- `date_of(value: str) -> str` — `YYYY-MM-DD` of an instant | 取日期.
- `age_days(t: str, as_of: str) -> float` — `max(0, (as_of - t).total_seconds()/86400)` | 天数差.
- `day_window(date: str) -> tuple[str, str]` — `(T00:00:00.000000, T23:59:59.999999)` | 当日窗口.
- `month_window(year: int, month: int) -> tuple[str, str]`; `week_window(date: str) -> tuple[str, str]` (Monday..Sunday).
- `next_day_midnight(date: str) -> str` — `expiration_date` → `valid_to` (`<date+1>T00:00:00.000000+00:00`) | 次日零点.
- `shift_days(date: str, n: int) -> str`.

### §9.2 `hms_elite/utils/text.py`

Module docstring: `"""Deterministic text utilities and EN/ZH cue lexicons shared by the rule provider, planner and scorer.\n规则提供者、查询规划器与打分器共享的确定性文本工具与中英线索词表。"""`

Constants (exact):
```python
IDENT_RE = re.compile(r"(?<![\w/])(?:[\w./-]*/)?[\w-]+\.(?:ts|tsx|js|jsx|py|go|rs|md|json|yaml|yml|sql|toml|css|html)\b|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[A-Z]{2,}-\d+\b|#\d+\b")
FILE_EXT_RE = re.compile(r"\.(?:ts|tsx|js|jsx|py|go|rs|md|json|yaml|yml|sql|toml|css|html)$")
QUOTED_RE = re.compile(r"[\"“「『]([^\"”」』]{1,60})[\"”」』]|'([^']{2,60})'")
PROPER_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b")
CJK_RE = re.compile(r"[一-鿿]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
STOP_CAPS = {"I","A","An","The","This","That","These","Those","It","He","She","They","We","You","My","Our","Your","His","Her","Its","Their","Me","Us",
             "And","But","Or","So","If","When","While","Because","After","Before","Since","Until","Then","Now","Today","Yesterday","Tomorrow",
             "Do","Does","Did","Don't","Never","Not","No","Yes","Please","Actually","Also","However","Still","Just","Only","Maybe",
             "What","Why","How","Where","Who","Which","Is","Are","Was","Were","Have","Has","Had","Can","Could","Should","Would","Will","Must","May",
             "Let","Ok","OK","Hi","Hello","Thanks","Thank","User","Assistant","Agent","System","Monday","Tuesday","Wednesday","Thursday","Friday",
             "Saturday","Sunday","January","February","March","April","June","July","August","September","October","November","December",
             "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Sept","Oct","Nov","Dec"}
STOPWORDS_EN = {"a","an","the","and","or","but","if","then","so","of","to","in","on","at","by","for","with","from","as","is","are","was","were","be",
                "been","being","do","does","did","have","has","had","i","me","my","we","our","you","your","he","she","it","its","they","them","their",
                "this","that","these","those","what","which","who","whom","when","where","why","how","not","no","yes","about","into","over","under",
                "again","also","just","than","too","very","can","could","should","would","will","shall","may","might","must","did","done","user","s"}
STOPWORDS_ZH = {"的","了","是","在","我","我们","你","他","她","它","和","与","或","吗","呢","吧","啊","这","那","什么","怎么","为什么"}
TOPIC_TABLE: list[tuple[re.Pattern, str]] = [   # first-appearance order in the text; each topic at most once
    (r"\b(coffee|latte|lattes|espresso|cappuccino|tea|mocha)\b|咖啡|拿铁|茶", "coffee"),
    (r"\b(guitar|fender|gibson|stratocaster|piano|drums|band|music)\b|吉他|音乐", "music"),
    (r"\b(retry|patch|test|tests|deploy|webhook|bug|refactor|checkout|regression|backoff|idempotency|commit|diff)\b|重试|测试|部署|回滚", "engineering"),
    (r"\b(billing|subscription|payment|invoice|checkout)\b|计费|订阅|支付", "billing"),
    (r"\b(trip|flight|hotel|visit|visited|travel|vacation)\b|旅行|航班|酒店", "travel"),
    (r"\b(meeting|standup|review|design|team|joined|shipped)\b|会议|团队|设计", "work"),
]
PREF_VERBS_POS = {"like","love","enjoy","adore","prefer"}; PREF_VERBS_NEG = {"hate","dislike","can't stand","cannot stand"}
SUPERSESSION_CUES_EN = r"\b(now|these days|nowadays|switched to|switch to|moved to|no longer|not anymore|anymore|instead|actually|from now on)\b"
SUPERSESSION_CUES_ZH = r"现在|改成|换成|改喝|不再|以后|如今"
OBJ_TRAILING_CUES = {"now","these","days","nowadays","anymore","instead","currently","actually","though","了","的"}
ARTICLES = {"a","an","the","some","my","our"}
EVENT_VERBS_PAST = {"joined","shipped","visited","went","met","moved","started","finished","bought","released","launched","fixed","deployed","left","came","attended","flew","arrived","signed","closed","opened","merged","migrated"}
FUTURE_CUES = r"\b(will|next|tomorrow|plan to|plans to|planning|going to)\b|明天|下周|下个月|将"
VERB_LEXICON = {"is","are","was","were","has","have","had","does","did","works","worked","lives","lived","uses","used","joined","joins","met","moved","moves","started","starts","finished","shipped","ships","reviewing","reviews","reviewed","plans","planned","wants","needs","likes","prefers","decided","chose","bought","visited","went","goes","came","left","built","wrote","said","told","asked","runs","ran","fixed","patched","leads","owns","manages","reports"}
ACTION_VERBS = r"patched|edited|changed|modified|ran|re-ran|reran|tried|reverted|rolled back|deployed|added|removed|fixed|refactored|tested|applied|updated|installed|migrated|rewrote|merged"
CLAUSE_CUT = r"(?=\s+(?:because|since|after|when|so that|and|but)\b|[,;!?]|\.(?!\w)|$)"          # stops before a causal/coordinating conjunction, a clause separator, a sentence-final dot (dots inside identifiers survive) or end of text
ACTION_RE = re.compile(r"\b(?:I|we|agent|it|he|she|they)?\s*(?P<verb>" + ACTION_VERBS + r")\b\s*(?P<target>.+?)" + CLAUSE_CUT, re.I)
CAUSE_RE = re.compile(r"\b(?:because|since|after|when|so that)\s+(?P<u>.+?)(?=[,;!?]|\.(?!\w)|$)|(?:因为|之后)(?P<uzh>[^，。；！？]+)", re.I)   # the trigger clause after the conjunction, cut at the next clause separator
CLAUSE_SPLIT_RE = re.compile(r",\s*(?:and|but|so)?\s*|;\s*|\s+(?:and|but)\s+|，|；")                                                    # clause splitter used for `y`
OUTCOME_RE = re.compile(r"\b(failed|fails|failing|passing|passed|error|errors|errored|succeeded|works now|worked|broke|broken|regression|regressed|green|red|timeout|timed out|crash|crashed)\b|失败|报错|通过|成功|崩溃", re.I)
NEGATIVE_OUTCOMES = {"failed","fails","failing","error","errors","errored","broke","broken","regression","regressed","red","timeout","timed out","crash","crashed","失败","报错","崩溃"}
POSITIVE_OUTCOMES = {"passed","passing","succeeded","works now","worked","green","通过","成功"}
CONSTRAINT_RE = re.compile(r"^(?:(?P<subj>I|We|You|The team|Team)\s+)?(?:(?:must|should|can|could)\s*(?:not|n't)|mustn't|shouldn't|cannot|can't|never|do not|don't|(?:is|are|am)\s+not\s+allowed\s+to|no changes to)\s+(?P<rest>.+)$", re.I)
CONSTRAINT_ZH_RE = re.compile(r"(禁止|不能|不要|不得|不允许|绝不)(?P<rest>.+)")
MUST_RE = re.compile(r"\b(must|should|always|has to|have to|need to|required to)\b", re.I)
WHY_RE = re.compile(r"\b(why|because|root cause|what caused|reason|explain)\b|为什么|原因|为何", re.I)
HANDOFF_RE = re.compile(r"\b(handoff|hand off|status|state of|where (?:are|were) we|next step|what's next|whats next|continue|resume|current state|pick up|work state)\b|接手|进度|下一步|继续|当前状态", re.I)
TEMPORAL_QUERY_RE = re.compile(r"\b(when|what happened|timeline|history|during|on what date)\b|什么时候|发生了什么|时间线", re.I)
ENTITY_QUERY_RE = re.compile(r"\b(everything about|all about|tell me about|who is|what do you know about|anything about)\b|关于|是谁", re.I)
PREF_QUERY_RE = re.compile(r"\b(prefer|prefers|like|likes|favorite|favourite|choice|chose|compare|dislike|enjoy)\b|喜欢|偏好|选择|讨厌", re.I)
MONTHS = {"january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,"april":4,"apr":4,"may":5,"june":6,"jun":6,"july":7,"jul":7,"august":8,"aug":8,"september":9,"sep":9,"sept":9,"october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12}
DATE_RES = [  # applied in order; first match wins (§13.2 / §11.1 share them)
    ("iso_date",  r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    ("iso_month", r"\b(?P<y>\d{4})-(?P<m>\d{2})\b(?!-)"),
    ("month_en",  r"\b(?:on|in|by|since|until|during|of|from)?\s*(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?:\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(?P<y>\d{4}))?\b"),
    ("month_zh",  r"(?:(?P<y>\d{4})年)?(?P<m>\d{1,2})月(?:(?P<d>\d{1,2})[日号])?"),
    ("relative",  r"\b(?P<rel>yesterday|today|tomorrow|last week|last month|last year|next week|next month)\b|(?P<relzh>昨天|今天|明天|上周|上个月|下周|下个月)"),
    ("relative_this", r"\b(?P<rel>this week|this month)\b|(?P<relzh>本周|这周|这个月)"),   # planner only; §11.1 R7 ignores this kind
    ("ago",       r"\b(?P<n>\d+)\s+(?P<unit>days?|weeks?|months?|years?)\s+ago\b|(?P<nzh>\d+)(?P<unitzh>天|周|个月|年)前"),
    ("in_n",      r"\bin\s+(?P<n>\d+)\s+(?P<unit>days?|weeks?|months?)\b"),
]
```

Functions:
- `sentences(text: str) -> list[str]` — split on `SENTENCE_SPLIT_RE`, strip, drop empties | 分句. Notes: a trailing sentence without terminal punctuation is kept.
- `tokenize(text: str, *, cjk: str | None = None) -> list[str]` — lowercase; identifiers (`IDENT_RE`) kept whole; Latin tokens `[a-z0-9]+(?:['-][a-z0-9]+)*`; CJK runs → character bigrams (single char if run length 1) under `cjk='bigram'`, one token per run under `cjk='none'`; `cjk=None` reads the `cjk_tokens_scope` in force (default `bigram`, §9.6); order preserved, duplicates kept | 分词.
- `content_tokens(text: str, *, cjk: str | None = None) -> list[str]` — `tokenize(text, cjk=cjk)` minus `STOPWORDS_EN ∪ STOPWORDS_ZH`, each stemmed by `stem` | 去停用词.
- `stem(token: str) -> str` — identifiers and CJK tokens unchanged; `ies→y`; strip trailing `s` when `len > 3` and not `ss`; `ing`/`ed` stripped when the remainder ≥ 4 chars | 轻量词干.
- `is_identifier(token: str) -> bool` — `IDENT_RE.fullmatch` or contains one of `./_-` with ≥1 letter | 是否标识符.
- `identifier_tokens(text: str) -> list[str]` — `IDENT_RE.findall`, dedup, order kept | 标识符 token.
- `EntityCandidate = NamedTuple(entity_type: str, text: str, normalized: str, key: str)` — **`key` is the identity** stored in `entities.normalized_text` and returned in `MemoryAtom.entity_ids`; `normalized` is only an alias source (for FILE it is the full lower-cased path, stored in `entities.aliases_json`).
- `extract_entities(text: str) -> list[EntityCandidate]` — priority order: FILE (identifier containing `/` or matching `FILE_EXT_RE`; `key = basename.lower()`, `normalized = full path lower`), IDENTIFIER (other `IDENT_RE` hits; key = lower), QUOTED (`QUOTED_RE`, key = normalized text), PROPER (`PROPER_TOKEN_RE` tokens not in `STOP_CAPS`, consecutive tokens joined with one space, sentence-initial allowed; key = lower with ws collapsed), CJK proper nouns: 2–4 char runs immediately before `是|在|和|喜欢|加入|负责` (key = the run); dedup by key keeping the first; overlap suppression by span: candidates claim character spans in the priority order above (FILE, IDENTIFIER, QUOTED, PROPER, CJK), and a later candidate whose span overlaps an already-claimed span is dropped (so the quoted phrase `'patch src/billing/retry.ts: add exponential backoff'` yields only the FILE entity `retry.ts`) | 实体候选. Notes: month/weekday names never become entities; `entity_key(text)` = the key rule above.
- `normalize_entity(text: str) -> str` — strip, lowercase, collapse whitespace | 归一化.
- `topic_tags(text: str) -> list[str]` — `TOPIC_TABLE` hits ordered by the position of each topic's first match in the text (ties broken by table order), each topic once | 主题标签.
- `sentiment(text: str) -> str` — positive lexicon (`like|love|enjoy|great|good|happy|passed|works|green|success|喜欢|通过|成功`) vs negative (`hate|dislike|bad|broke|failed|regression|error|crash|失败|报错|讨厌`); both → `mixed`; none → `neutral` | 情感.
- `rewrite_first_person(sentence: str) -> str` — ordered replacements at word boundaries: `I've`→`User has`, `I'm`→`User is`, `I'd`→`User would`, `I'll`→`User will`, `I am`→`User is`, `I have`→`User has`, `I do not`/`I don't`→`User does not`, `I was`→`User was`, `I`→`User`, `my`→`user's` (`My`→`User's`), `me`→`the user`, `we`/`We`→`the team`/`The team`, `our`→`the team's`; capitalize first char; ensure trailing `.` unless ending in `.!?。！？` | 第一人称改写.
- `clean_object(phrase: str) -> str` — strip; drop trailing tokens in `OBJ_TRAILING_CUES`; drop leading `ARTICLES`; strip trailing `了|的|。`; collapse ws | 宾语清理.
- `extract_dates(text: str, *, base_date: str) -> list[tuple[str, str, str]]` — every `DATE_RES` hit → `(kind, gte, lte)` windows in text order (`base_date` `YYYY-MM-DD`; month without year → `base_date` year; `month_en` requires the month token to be a standalone word, `May` only when followed by a digit OR preceded by `in|on|during|of|since|until` — so "may change" is never a date) | 日期抽取.
- `resolve_date(text: str, *, base_date: str) -> str | None` — first window's `gte` date (`YYYY-MM-DD`) | 解析日期.
- `has_action_closure(text: str) -> bool` — `ACTION_RE` and (`OUTCOME_RE` or `CONSTRAINT_RE` on any sentence) | 是否动作闭包.
- `ordered_union(*seqs: Iterable[str]) -> list[str]` — concatenation with later duplicates dropped; never sorted (§0.1) | 有序并集.
- `is_first_person(sentence: str) -> bool` — `re.search(r"\b(I|I'm|I've|I'd|I'll|my|we|我|我们)\b", sentence)` (search, not match; "Yesterday I visited …" and "Actually I switched …" are first person) | 是否第一人称.
- `sha256_hex(s: str) -> str`; `md5_hex(s: str) -> str`.

### §9.3 `hms_elite/utils/json.py`

- `canonical_json(obj: Any) -> str` — §0.1 | 规范 JSON.
- `remove_code_blocks(content: str | list[str]) -> str` — Mem0 verbatim: join list; strip a whole-string ` ```lang ... ``` ` fence; strip `<think>...</think>` | 去代码围栏.
- `extract_json(text: str) -> str` — first fenced ` ```json ``` ` block, else first `{`..last `}`, else first `[`..last `]` (top-level-array salvage), else `text` | 提取 JSON.
- `parse_llm_json(raw: str, *, key: str, task: str = "") -> list | dict` — ladder: `remove_code_blocks` → blank ⇒ `[]` → `json.loads(strict=False)` → on `JSONDecodeError`: `extract_json` then `json.loads(strict=False)` → any exception ⇒ `logger.error("Error parsing %s response", task)` and `[]`; result: `obj.get(key, [])` when dict, `obj` when list and `key` is None, else `[]`; non-list for a list key ⇒ `[]` | 解析阶梯. Notes: never raises.

### §9.4 `hms_elite/memory/base.py`

```python
class MemoryBase(ABC):
    @abstractmethod def get(self, memory_id): ...
    @abstractmethod def get_all(self, *, filters=None, top_k=20, show_expired=False, **kwargs): ...
    @abstractmethod def update(self, memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None): ...
    @abstractmethod def delete(self, memory_id): ...
    @abstractmethod def history(self, memory_id): ...
```

### §9.5 `hms_elite/memory/utils.py`

- `_UNSET = object()` (module sentinel, exported).
- `parse_messages(messages: list[dict]) -> str` — `"role: content\n"` lines, skipping `content=None`, list content reduced to text parts | 消息拼接.
- `coerce_messages(messages: str | dict | list) -> list[dict]` — `str→[{'role':'user','content':str}]`, `dict→[dict]`, list passthrough, else `ValidationError(VALIDATION_003)`; each item must be a dict with `role` and `content` (else `VALIDATION_003`); list content → `{'content': joined text parts, 'dropped_parts': n}` (+ `'image_parts': [{'url', 'source', 'sha256', 'mime', 'detail'}]` for recognized OpenAI `image_url` / `input_image` and Anthropic `image` parts — in-memory only, still counted in `dropped_parts`); `content None` items dropped | 消息规整.
- `validate_and_trim_entity_id(value: Any, name: str) -> str | None` — Mem0 port; empty/inner whitespace → `ValidationError(VALIDATION_001, ValueError)` | 校验 id.
- `build_scope_and_metadata(*, user_id=None, agent_id=None, run_id=None, actor_id=None, input_metadata=None, context: str = "add") -> tuple[Scope, dict]` — trims ids, requires ≥1 (`VALIDATION_001` with `details={'provided_ids': {...}}`), deep-copies metadata, strips identity keys + reserved keys with ONE `logger.warning` listing them | 范围与元数据.
- `strip_reserved_keys(meta: dict, *, context: str) -> dict` — drops `RESERVED_ATOM_KEYS` ∪ identity keys; warns once | 剥离保留键.
- `reject_top_level_entity_params(kwargs: dict, method: str) -> None` — `ValidationError(VALIDATION_008, ValueError)` when `user_id/agent_id/run_id` in kwargs | 拒绝顶层 id.
- `validate_search_params(threshold: float | None, top_k: int | None) -> None` — `threshold ∈ [0,1]`, `top_k` non-negative int and not bool → else `ValidationError(VALIDATION_009, ValueError)` | 校验搜索参数.
- `validate_and_trim_query(query: str) -> str` — non-empty after strip else `VALIDATION_010` | 校验查询.
- `normalize_expiration_date(value: Any) -> str | None` — `None→None`; date/datetime/`YYYY-MM-DD` string → `YYYY-MM-DD`; else `ValidationError(VALIDATION_004)` | 过期日期归一.
- `estimate_tokens(text: str) -> int` — §0.1 | 估算 token.
- `mask_ids(text: str) -> str` — regex `\b(evt|anc|obs|ep|atm|mem|ent|edg|iep|fb|ckp|trn|pkg|hst)_[0-9a-f]{6,12}\b` → `<prefix>` (used by tests' golden helper; lives here so both T and X can import) | 掩码 id.

### §9.6 CJK text switches (`HMSConfig.text.cjk_tokens`, `IngestConfig.zh_rules`)

Two switches whose defaults are the historical behaviour, so English stores and every recorded EgoLife run stay byte-identical. They exist for the EgoLife-ZH experiment (`docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`: run A = the Chinese data through the English-centric text pipeline, run B = the same data with both switches on; A − B measures what Chinese support recovers).

- `text.cjk_tokens: Literal["bigram", "none"] = "bigram"` — the CJK mode of `tokenize` / `content_tokens` on BOTH sides of the keyword channel: write side `schema.fts_tokens` (atom_fts) and `scoring.atom_keyword_tokens`; read side `plan_query`, `mvmi.query._plan_keyword_tokens`, `scoring.prepare_query`, plus every other consumer (`reranker/overlap.py`, `compiler/coverage.py`, `ingest/extraction.py statement_tokens / verdict_similarity`, `ops/consolidation.py`, `llms/rule.py`). `bigram` = overlapping character bigrams (today); `none` = one token per CJK run, what a unicode61-style tokenizer does — a Chinese keyword query then hits only an identical whole run (the run A baseline). Mechanism: `utils.text.cjk_tokens_scope(mode)` (ContextVar `hms_cjk_tokens`, default `bigram`; `cjk_tokens_mode()` reads it; `CJK_TOKEN_MODES`), entered by the `utils.text.text_scoped` decorator on every method named in `memory.main.TEXT_SCOPED_METHODS` (35 public `HMS` methods — every data method and the §11 state family; excluded: `from_config`, `project`, `state`, `close`, `usage`, `trace`, `start_daemon` / `stop_daemon` (the daemon thread calls the wrapped `consolidate`), `history`, `vacuum`, `get`), on `Ingestor.ingest_messages` / `ingest_events`, `StateCompiler.compile` and `StateScheduler.run`, so standalone collaborators and daemon threads set the mode themselves (`AsyncHMS` runs the wrapped sync method inside `asyncio.to_thread`); `tokenize(text, *, cjk=None)` / `content_tokens(text, *, cjk=None)` read the scope when `cjk` is None; `scoring._content_vocabulary(text, cjk)` / `_key_tokens(key, cjk)` key their lru_caches on the mode; `stem` returns CJK tokens unchanged. Persistence: meta key `text_cjk_tokens` (absent == `bigram`; written only when the mode is not `bigram` or the key already exists, inside `unversioned()`, so a default store gains no meta row); `HMS.__init__` (`_sync_fts_mode`) calls `StateStore.rebuild_fts()` — `DELETE FROM atom_fts` + `schema.backfill_fts` in one transaction inside `unversioned()` (no write version moves, no history row) — when the stored mode differs from the configured one. No `SCHEMA_VERSION` bump (DDL unchanged). Tests: `tests/test_text_cjk.py`.
- `ingest.zh_rules: bool = False` — Chinese statement rules of the compact / lines derivation, threaded as `extract_all(..., zh_rules)` / `extract_observations(..., zh_rules)` → `_compact_observation_from_item` → `ingest.derive.expand_compact_item(..., zh_rules)` / `expand_lines_ep(..., zh_rules)` → `llms.rule.statement_triple(text, obs_type, extractor, *, zh_rules)`: the leading subject comes from the extractor's Chinese proper-name rules (杰克说… → 杰克), a leading 我 (the glossary form of "Me") is `user`, 我们 / 咱们 is `project` (the English `we` mapping) and any other pronoun run (他 / 她 / 他们 / 大家 / 你 …) names nobody but is stepped over so the verb after it is read (她喜欢吃披萨 → None / likes / 吃披萨; 我们打算晚上吃火锅 → project / plans / 火锅), the fallback predicate maps the head verb of the remainder through `VERB_LEXICON_ZH` (`predicate_of(text, *, zh_rules)`, regex at the head after optional adverbs — never through tokens, because under bigrams a one-char verb is no token), a `temporal_fact` whose head verb maps to `plans` is `plans`, and a fact / temporal_fact takes the clause after the verb as its object (`_zh_clause` / `_zh_objects`: descends light verbs 和 / 跟 / 给 + a known — or unlisted 2–4 char — name and 计划 / 打算 / 要 …, skipping an aspect particle 了 / 过 / 着 before the name (给了杰克一杯咖啡 → 一杯咖啡), cuts a 在-locative at the next verb, drops a leading 了 / 过 / 着; else the other entity mentions). After a coverb 和 / 跟 / 给 the fact predicate is that of the verb the descent reached (爱丽丝和塔莎去了超市 → went_to, 塔莎跟杰克说… → says, 和舒尔在阳台… → is_at, 给了杰克一杯咖啡 → gives); `VERB_LEXICON_ZH` maps a bare 和 / 跟 to `states` ('with' exists in no predicate vocabulary), and 计划 / 打算 … keep `plans`. A leading `ZH_NAME_STOP` / time word is skipped before the subject is read (`utils.text.zh_name_stop_prefix`: 今天杰克和爱丽丝一起去了公园 → 杰克 / did / 公园; 下午我去了超市 → user), and the adverb / time-word prefix between subject and verb is one alternation shared with the entity rules (`utils.text.ZH_LEAD_ADVERB_ALT`, so 杰克晚上去了超市 → 杰克 / 超市). `mention_texts(text, extractor, *, zh_rules)` and `observation_topics(text, obs_type, *, zh_rules)` forward the flag; `utils.text.sentiment(text, *, zh_rules)` adds `_POSITIVE_SENT_ZH_RE` / `_NEGATIVE_SENT_ZH_RE` (negated phrases such as 不喜欢 count as negative only; the positive list carries 爱 only in bounded forms — 喜爱 / 热爱 / 爱吃 / 爱上 / 爱玩 — so the glossary name 爱丽丝 never makes a statement positive: 爱丽丝讨厌洗碗 → negative, 爱丽丝去了超市 → neutral); `topic_tags(text, *, zh_rules)` appends `TOPIC_TABLE_ZH` (food, shopping, games, music, travel, housework, work; a topic in both tables keeps its earliest hit). `utils.text.extract_entities(text, *, zh_rules=False, names=())`: `zh_rules` adds `_zh_name_candidates` — a 2–4 char run head right before a `ZH_STATEMENT_VERBS` verb (lead adverbs / time words between them skipped: `ZH_LEAD_ADVERBS` = 也 / 都 / 一起 / … ∪ `ZH_TIME_WORDS` = 今天 / 晚上 / 下午 / 周末 / 然后 / …, one alternation `ZH_LEAD_ADVERB_ALT` shared with `llms.rule`; a `ZH_NAME_STOP` prefix such as 今天 / 下午 stripped first — `ZH_NAME_STOP` ⊇ `ZH_TIME_WORDS`; `zh_name_ok` rejects pronoun-led heads, stop words and a head that OPENS or ENDS with a statement verb or a lead adverb / time word — the two positions where such a word is a boundary — so 打算晚上, 杰克晚上, 在客厅 are never names while 王和平 is; a name whose last character is such a word (李刚: 刚 is a lead adverb) is still rejected because the suffix is indistinguishable from 李 + 刚 — a documented limitation, caught only by the `names` list; until then the 4-char ZH head 李刚说他 (before 明天要) is the output — a `_zh_name_candidates` head, since the legacy `_CJK_PROPER_RE` lookahead does not run under `zh_rules`), or a 2–4 char run right after 和 / 跟 / 给 (an aspect particle 了 / 过 / 着 skipped: 跟了塔莎 → 塔莎, 给了杰克 → 杰克) followed by a verb, a quantifier / particle (`_ZH_NAME_TAIL_RE`: 一个 / 一杯 / 一本 / … / 的 / 们 — not 了, which never follows a name) or the end of the run; `names` are matched exactly wherever they occur (longest first). Candidate priorities: FILE 0 < IDENTIFIER 1 < QUOTED 2 < PROPER 3 < NAME 4 < ZH 5 < CJK 6 (`_CJK_PROPER_RE` is unchanged and lowest, and it runs only when `zh_rules` is False: its 2–4 char heads before 是 / 在 / 和 / 喜欢 / 加入 / 负责 — 要求团队, 都写, 硬盘假装, 他的家乡, 做的布线, 并让大家 in the EgoLife-ZH smoke store — are fragments, not names, so under `zh_rules` the entities of a text are exactly the `names` glossary hits, the `_zh_name_candidates` heads, FILE / IDENTIFIER / QUOTED and Latin PROPER runs; `zh_rules=False` output is byte-identical to before — `tests/test_zh_rules.py::test_zh_rules_skip_the_legacy_cjk_lookahead`). `RuleEntityExtractorConfig(zh_rules: bool = False, names: list[str] = [])`; identity `rule@entities-v1` under the defaults, `rule@entities-zh-v1` when either is set (§10.5); `HMS.__init__` injects `zh_rules=True` into a copy of `entity_extractor_config` when `ingest.zh_rules` and the provider is `rule`. The READ side sees the same extractor: `mvmi.planner.plan_query(query, *, as_of, mode, extractor=None)` takes `extractor.extract(text)` for the entity hints when one is given (else `utils.text.extract_entities`, byte-identical to the default rule extractor), and so do `mvmi.query._scoring_plan` (the ad-hoc plan of a text AxisQuery without `plan`), `mvmi.query.search(..., extractor)`, `StateCompiler(..., entity_extractor)` (`context()` / `compile()`) and `compiler.coverage.extract_requirements(..., extractor)`; `HMS.search` / `context` / `compile` / `coverage` pass `self.entity_extractor`, so a Chinese question such as 杰克买了什么？ yields `plan.entity_hints` [杰克], an `entity_boost` > 0 and `'entity'` in `recalled_by` under `zh_rules` + `names`, and none of them under the defaults (`tests/test_zh_rules.py::test_review_fixes_sentiment_names_pronouns_and_read_side_entity_axis`). Run B of the EgoLife-ZH plan therefore checks `channels.entity_hits > 0` in `questions_elite_rows.jsonl` (the harness reads the signal from `score_details['entity_boost']`, the only entity key `scoring.score_details` writes). The derived Chinese subject reaches `Observation.speaker` → `actor_id` through the unchanged lines speaker rule (§6.6). The rule LLM's own observation rules (§11.1) and `RuleLlmConfig.language` are untouched. Tests: `tests/test_zh_rules.py`.
- Lines prompt: `MEMORY_EXTRACTION_LINES_PROMPT` carries the sentence "Write every statement in the language of the source messages (Chinese messages → Chinese statements, names as written); never translate." after the multi-speaker sentence (no sha pin on this prompt; `KEY_SENTENCES` and the four hashed prompts untouched; the recorded `results/real/elite_lines_full204` replies predate the sentence — replay comparability is unaffected because replay serves recorded replies verbatim).
- `estimate_tokens` unchanged: 3 UTF-8 bytes per CJK character → 0.75 estimated tokens per character, never below the ≈ 0.67 (one token per 1.5 characters) a Chinese tokenizer produces, so compiler budgets over Chinese rows are conservative, not under-counted (`tests/test_foundation.py`).
- Measurement caveat: `cjk_tokens='none'` is global on purpose, so it also changes `statement_tokens` / `verdict_similarity` (a whole-run token rarely overlaps: most NONE / REVISE verdicts fall to ADD), overlap reranking, coverage, consolidation dedup and the rule sentence gate; run A − run B therefore mixes keyword-channel loss with revision-guard loss, and the plan doc reports guard-rejected verdicts per run. Data source: `HMS.last_downgraded` (the `IngestResult.downgraded` entries of the latest `add()` / `add_events()`; the Mem0 envelope stays `{'results': rows}`), which `benchmarks/egolife/real_arms.EliteRealArm._add` attaches to its envelope as `downgraded: {verdict: count}`, `retain_with_bisect` records per sub-add and per batch (`retain_elite.jsonl`), and `run_real.summarize` sums per run (`engines.elite.retain.downgraded`).

---

## §10 Providers (group B)

### §10.1 LLMs (`hms_elite/llms/`)

```python
class LLMBase(ABC):
    def __init__(self, config: BaseLlmConfig | dict | None = None) -> None      # dict → the provider's config class; None → defaults
    @abstractmethod
    def generate_response(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs) -> str | dict
        # kwargs: response_format={'type': 'json_object'}; returns str, or {'content': str|None, 'tool_calls': [...]} when tools were used (Mem0 contract)
    def generate_json(self, task: str, *, system: str, user: str) -> list | dict
        # messages=[{'role':'system','content':system},{'role':'user','content':user}], response_format json_object;
        # transport exception → raise LLMError(f"{task} failed: {e}", details={'task': task, 'provider': self.identity}) from e; parse via parse_llm_json(key=ROOT_KEY[task])
    def generate_lines(self, task: str, *, system: str, user: str) -> str             # memory_extraction_lines path (§6.6): the same two messages, NO response_format, raw text back (dict → content, None → ''); ingest.lines.parse_lines parses it
    def generate_text(self, task: str, *, messages: list[dict]) -> str                 # procedural_summary path; remove_code_blocks applied
    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None   # vision hook, NOT abstract: None = unsupported (base, rule, scripted without image_description); openai → chat.completions image_url part, anthropic → messages image block (base64 source for data URIs, url source for http); called by the ingest pipeline unlocked, before Txn A
    @property
    def identity(self) -> str                                                          # 'rule@v1' | 'scripted@v1' | 'openai:gpt-5-mini' | 'anthropic:claude-sonnet-4-6'
ROOT_KEY = {"observation_extraction": "observations", "ep_extraction": "eps", "revision_decision": "decisions", "edge_proposal": "edges", "coarsen": "interface_ep"}
```

- `RuleLLM(LLMBase)` (`rule.py`, §11): `generate_response` reads `task_of(system)`; unknown → `LLMError(LLM_002)`; a user prompt containing a line `RAISE: <msg>` → `LLMError(LLM_001, msg)` (test hook); dispatches `_task_<name>(sections)`; `calls: list[dict]` records `{'task', 'sections'}`.
- `ScriptedLLM(LLMBase)` (`scripted.py`, §11.7).
- `OpenAILLM(LLMBase)` (`openai.py`): module-level guard `try: from openai import OpenAI except ImportError as e: raise DependencyError("The 'openai' package is required for the openai provider", details={'package': 'openai', 'extra': 'openai'}) from e`; client `OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"), base_url=config.openai_base_url or os.getenv("OPENAI_BASE_URL"))`; `chat.completions.create(model, messages, temperature, max_tokens, top_p, response_format=kwargs.get('response_format'), tools=..., tool_choice=...)`; `_parse_response(response, tools)` → `str | {'content','tool_calls': [{'name','arguments': json.loads}]}`.
- `AnthropicLLM(LLMBase)` (`anthropic.py`): guard on `anthropic`; system message split into `system=`; `messages.create(model, system, messages, max_tokens[, temperature])` — `temperature` is forwarded only when the installed SDK's `Messages.create` still declares it (`inspect.signature`, checked once at construction: `anthropic>=1.0` removed the sampling parameters and raises `TypeError` for them, so the `anthropic>=0.40` extra of §20.1 must work on both 0.x and 1.x; nothing else is sampled); when `response_format` requests JSON, appends `"\nRespond with JSON only."` to the last user message; returns text.

Docstrings: `generate_response` — "Produce a completion for chat messages | 生成对话补全"; In: messages, tools, tool_choice, kwargs | 消息等; Out: str or dict | 文本或字典; Effect: none (network for real providers) | 无（真实提供者有网络调用）.

### §10.2 Embeddings (`hms_elite/embeddings/`)

```python
class EmbedderBase(ABC):
    def __init__(self, config: BaseEmbedderConfig | dict | None = None) -> None
    @abstractmethod def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]
    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]    # sequential default; per-item failure → EmbeddingError(EMBED_001) from e
    @property def dims(self) -> int
    @property def identity(self) -> str                                                          # 'hash@64' | 'openai:text-embedding-3-small@1536'
```

- `HashEmbedder` (`hash.py`): `text.lower()` wrapped as `f" {text} "`; character n-grams for `n ∈ [ngram_min, ngram_max]`; bucket `int(hashlib.sha256(ngram.encode()).hexdigest()[:8], 16) % dims`; sign `+1` if the 9th hex digit is even else `-1`; accumulate; L2-normalize (zero vector for empty text → all zeros); pure Python; deterministic across processes. `dims` from config (default 64).
- `OpenAIEmbedder` (`openai.py`): guard; `embeddings.create(model, input=text, dimensions=embedding_dims)`; `identity = f"openai:{model}@{dims}"`.

### §10.3 Index (`hms_elite/index/`) — the pluggable vector index

```python
INDEX_ATTR_KEYS = ('user_id', 'agent_id', 'run_id', 'time_anchor', 'valid_from', 'valid_to', 'archived', 'scale_level', 'obs_type', 'atom_type')   # scalar gates a provider may keep beside each vector; NULL strings as "", instants fixed-width UTC ISO (lexical order == time order), archived: bool
def atom_index_attrs(atom: MemoryAtom) -> dict      # the INDEX_ATTR_KEYS of a stored atom
def normalize_attrs(attrs: Mapping | None) -> dict  # keep INDEX_ATTR_KEYS only; None → "", instants → to_utc_iso, archived → bool
def select_top_k(query, ids, matrix, *, top_k, rows=None, scores=None) -> list[tuple[str, float]]   # bounded top-k over an (n, dims) float32- or float64-valued matrix: one matmul SELECTS the rows (margin 2e-12, the zero-clamp regime rescored whole), `dot` rescores them → bit-identical to rank_hits(...)[:top_k]; rows = candidate row positions (None → all), scores = precomputed float64 products aligned with rows (a cached full-matrix product's masked slice)
@dataclass(frozen=True) class IndexFilter: scope: Scope | None = None; as_of: str | None = None; include_expired: bool = False; scale_in: tuple[str, ...] | None = None; obs_type_in: tuple[str, ...] | None = None; time_from: str | None = None; time_to: str | None = None; atom_type_in: tuple[str, ...] | None = None
    # the SQL gate of axis_select as data: unset scope keys add nothing (Scope.matches); include_expired=False ⇔ archived == false AND valid_from <= as_of AND (valid_to == "" OR valid_to > as_of); whitelists; inclusive time_anchor bounds (rows without a time_anchor never match a bound)

class IndexBase(ABC):
    def __init__(self, config: BaseIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64) -> None
    @abstractmethod def upsert(self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping | None = None) -> None   # attrs: INDEX_ATTR_KEYS scalars, ignored by providers without scalars
    def upsert_batch(self, items: list[tuple[str, list[float]]], *, embedder: str, attrs: Mapping[str, Mapping] | None = None) -> None   # sequential default; attrs = {atom_id: attrs}
    @abstractmethod def get(self, atom_id: str) -> list[float] | None
    @abstractmethod def delete(self, atom_id: str) -> None
    @abstractmethod def search(self, vector: list[float], *, candidate_ids: list[str] | None = None, filter: IndexFilter | None = None, top_k: int) -> list[tuple[str, float]]
        # cosine similarity clamped max(0.0, dot) (unit vectors), sorted (score DESC, atom_id ASC), AT MOST top_k rows; candidate_ids=None → no id restriction; ids without a vector are skipped; filter only on providers with supports_filter (others raise ValidationError(VALIDATION_000) telling the caller to gate with StateStore.axis_ids); both given → the intersection
    @abstractmethod def count(self, filter: IndexFilter | None = None) -> int   # filter only on supports_filter providers (others raise VALIDATION_000); atoms without a vector are never counted
    @abstractmethod def reset(self) -> None
    def update_attrs(self, atom_id: str, attrs: Mapping) -> None   # non-abstract hook, no-op by default; StateStore.attr_listeners calls it after COMMIT (archive / restore / set_valid_to) with the full atom_attrs snapshot, in commit order
    def gate_ids(self, filter: IndexFilter) -> list[str] | None     # the IndexFilter evaluated as store.axis_ids(order=None) on the bound store (None without a store): the SQL fallback of a pushdown provider
    def close(self) -> None                                         # non-abstract hook, no-op by default; HMS.close() calls it before store.close()
    @property supports_filter -> bool                               # False by default
    def info(self) -> dict                                                                        # {'provider', 'dims', 'count', 'numpy': bool}
```

- `SQLiteIndex` (`sqlite.py`): writes `atom_vectors` through `store.upsert_vector / delete_vector / clear_vectors` (same connection, same transaction) and ranks through a process-local `_VectorCache` per dimensionality — a float64 matrix of the stored float32 vectors plus the INDEX_ATTR_KEYS scalars (integer codes for user_id / agent_id / run_id / scale_level / obs_type / atom_type, fixed-width `S40` bytes for the three instants, a bool for archived), grown by doubling. The cache is keyed on `(store.data_generation, store.rebuild_generation, store.data_version())`: an unchanged key means no SQL at all; a changed data_generation extends it with `store.load_vector_rows(last_rowid)` (appends, and in-place overwrites for INSERT OR REPLACE) and applies the queued `update_attrs` changes (the provider registers itself in `store.attr_listeners`; a direct `update_attrs` call overwrites the cached scalars like Milvus does); a changed rebuild_generation or data_version (DELETE / DROP / ROLLBACK / reset / a foreign commit) rebuilds it from scratch; a row of another dimensionality marks the cache unusable and the provider falls back to `rank_hits` over `store.load_vectors`. `supports_filter=True`: the IndexFilter is a numpy mask over the cached scalars with exactly the semantics of the SQL gate (unset scope keys add nothing; `include_expired=False` ⇔ not archived and valid_from <= as_of and (valid_to == "" or valid_to > as_of), `as_of` defaulting to store.now(); whitelists through the codes; a NULL time_anchor never satisfies a time bound), memoized for the last (cache key, filter) so `search` and `count(filter)` of one query share it; `candidate_ids` is a row gather; `search` computes one product over the whole matrix when at least a quarter of the rows qualify (else over the gathered rows) and hands the masked scores to `select_top_k(rows=, scores=)` → bit-identical to `rank_hits` over the same vectors, nothing hydrated; without numpy (`use_numpy='never'` or not installed) a filter is evaluated as `gate_ids` (SQL). `close()` unsubscribes the listener and drops the cache; `info()` keeps the four keys; the tri-state numpy sentinel logs once. **Lock order**: `search()` / `count(filter)` take `store.lock` (re-entrant) BEFORE the index lock, because synchronizing the cache reads the store; a search issued inside `store.transaction()` therefore never deadlocks against a concurrent search (`tests/test_index_sync.py::test_search_inside_transaction_never_deadlocks`).
- `MemoryIndex` (`memory.py`): `dict[str, list[float]]`; same contract; `reset()` clears; `search` ranks through a cached float32 matrix snapshot (invalidated by every write) or, for n <= top_k, `rank_hits` over a snapshot of the dict so a concurrent `upsert()` (AsyncHMS worker threads) cannot resize it mid-iteration; `supports_filter=False`.
- `MilvusIndex` (`milvus.py`, provider `milvus`, extra `milvus`): `MilvusIndexConfig(uri=$HMS_DIR/milvus.db, token, db_name, collection='hms_atoms', metric='COSINE', index_type='AUTOINDEX'|'HNSW'|'FLAT', M, efConstruction, ef, consistency_level, batch_size, candidate_chunk, ef_recall_factor=4, scalar_index='INVERTED'|'none')`; a local uri must end with `.db` (milvus-lite), `http/https/grpc/grpcs/tcp/unix` uris address a server. `pymilvus` is imported lazily in the constructor (`ImportError → DependencyError(DEPS_001, extra='milvus')`); the collection `(atom_id VARCHAR pk, vector FLOAT_VECTOR(dims), embedder VARCHAR, user_id / agent_id / run_id VARCHAR(256), time_anchor / valid_from / valid_to VARCHAR(40), scale_level VARCHAR(16), obs_type VARCHAR(32), atom_type VARCHAR(16), archived BOOL)` is created (scalar indexes of kind `scalar_index` on user_id, agent_id, run_id, scale_level, obs_type, atom_type, valid_to, archived; ~0.5 s per field once on milvus-lite) or validated against the factory's `dims` and the field set (mismatch / foreign collection / a collection without the scalars, `atom_type` included → `CONFIG_001`, suggestion: new collection name or `consolidate()`) and then `load_collection`ed (a collection reopened by a new process is in state released and search would raise code 101 otherwise). `upsert` layers the scalars: defaults, then `store.atom_attrs` (one batched SELECT, skipped when the caller supplies every key — ingest passes `attrs=atom_index_attrs(stored)`), then the caller's `attrs`; `update_attrs` reads the row, merges, upserts (Milvus has no in-place scalar update). `supports_filter=True`: `filter_expr(IndexFilter)` renders `user_id == "alice" and archived == false and valid_from <= "T" and (valid_to == "" or valid_to > "T") and scale_level in [...] and obs_type in [...] and atom_type in [...] and time_anchor >= "A" and time_anchor <= "B"` (json.dumps escaping; `as_of` defaults to `now_iso()`); `search(filter=)` alone is ONE native filtered ANN with `limit=top_k` and, on HNSW, `ef = max(ef, limit, ef_recall_factor·limit)` (RETRIEVAL_DIAGNOSIS T6 recall gap: callers keep small hard-gated sets out of ANN); with `candidate_ids` the expression is ANDed into every `candidate_chunk` id slice (the exact intersection) and, on HNSW, `ef` is raised to the chunk size (a 1024-id filter over 30k rows returned a handful of hits otherwise). `count(filter)` is a `count(*)` query with the same expression — a full scan on milvus-lite (556 ms at N=30k), so the retrieval pool never calls it (`gate_count = -1` under pushdown). Hits are re-scored with `rank_hits` over the stored float32 vectors, the zero-score tail rebuilt from every gated candidate when a truncated result reaches score 0, so results equal the exhaustive providers (FLAT exactly; HNSW / AUTOINDEX up to ANN recall); client failures wrap as `StoreError(STORE_001)`; `close()` (idempotent, reached by `HMS.close()`) releases the client and later calls raise `StoreError`; `info()` adds `uri`, `collection`, `index_type`. SQLite stays the only source of truth: a stale Milvus scalar can hide a row (the caller re-validates after hydration), never surface a wrong one.

### §10.4 Factories (`hms_elite/utils/factory.py`)

```python
def load_class(path: str) -> type                            # rsplit('.', 1) + importlib; ImportError propagates to create()
class _Factory:
    kind: ClassVar[str]
    @classmethod
    def create(cls, provider: str, config: dict | BaseModel | None = None, **kwargs) -> Any
        # unknown provider → ConfigError(CONFIG_002, details={'kind','provider','supported'}); coercion: None → ConfigClass(**kwargs); dict → ConfigClass(**{**config, **kwargs}) (caller dict never mutated);
        # BaseModel instance → passthrough; DependencyError raised by the provider module propagates; other ImportError → DependencyError(DEPS_001, details={'package': e.name, 'extra': EXTRA_OF_PROVIDER.get(provider, provider)}) from e
    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class_path: str) -> None     # mutates configs.registry.REGISTRY[kind]
    @classmethod
    def get_supported_providers(cls) -> list[str]
class LlmFactory(_Factory): kind = "llm"            # create(provider, config, *, entity_extractor=None): forwarded to constructors that declare the parameter (RuleLLM), dropped otherwise
class EmbedderFactory(_Factory): kind = "embedder"
class IndexFactory(_Factory): kind = "index"        # create(provider, config, *, store, dims)
class RerankerFactory(_Factory): kind = "reranker"  # create(provider, config, *, llm=None): `llm` forwarded only to constructors that declare it (LLMReranker), dropped for OverlapReranker
class EntityExtractorFactory(_Factory): kind = "entity_extractor"   # §10.5
```

### §10.5 Entity extractors (`hms_elite/entities/`) — the pluggable entity extractor

```python
class BaseEntityExtractorConfig(HMSModel)                                 # extra='forbid' + did-you-mean hint; no shared fields
class EntityExtractorBase(ABC):
    def __init__(self, config: BaseEntityExtractorConfig | dict | None = None) -> None
    @property def identity(self) -> str                                    # 'rule@entities-v1' | 'spacy@<model>'
    @abstractmethod def extract(self, text: str) -> list[EntityCandidate]  # utils.text ordering + dedup-by-key semantics; pure
Span = tuple[int, int, EntityCandidate]
def merge_candidates(text: str, base: Sequence[EntityCandidate], spans: Sequence[Span]) -> list[EntityCandidate]
    # locate `base` left to right in `text` (single-space PROPER joins fall back to a `\s+` regex); NER spans overlapping a located rule span
    # are dropped; result ordered by start offset (rule first on ties), dedup by key (first kept); the rule list is always a subsequence
```

- `RuleEntityExtractor` (`rule.py`, provider `rule`, the default): `extract(text) is extract_entities(text)` — byte-identical under the default config; `RuleEntityExtractorConfig(zh_rules: bool = False, names: list[str] = [])` (§9.6) — when either is set, `extract(text) = extract_entities(text, zh_rules=, names=)` and the identity is `rule@entities-zh-v1`.
- `SpacyEntityExtractor` (`spacy.py`, provider `spacy`, extra `spacy`): `SpacyEntityExtractorConfig(model="en_core_web_sm", labels=DEFAULT_LABELS, disable=[], strip_articles=True)`; `labels` maps spaCy NER labels to `EntityType` values (default: PERSON/ORG/GPE/LOC/FAC/NORP/PRODUCT/EVENT/WORK_OF_ART/LAW/LANGUAGE → `PROPER`; DATE/TIME/CARDINAL/… unmapped → dropped; a non-EntityType value → CONFIG_001). Constructor `(config=None, *, nlp=None)`: `nlp` injects a ready pipeline (tests), else `load_pipeline(model, disable=...)` imports `spacy` lazily — `ImportError → DependencyError(DEPS_001, details={'package': 'spacy', 'extra': 'spacy'})`, missing model `OSError → DependencyError(details + {'model'}, suggestion 'python -m spacy download <model>')`. `extract`: `merge_candidates(text, extract_entities(text), spans)` where each `doc.ents` span is trimmed of a leading article (`strip_articles`) and a trailing `'s`, then `EntityCandidate(type, surface, normalize_entity(surface), entity_key(surface))`; blank text returns `[]` without running the pipeline.
- Wiring (one instance per facade): `HMS.__init__` builds `self.entity_extractor = EntityExtractorFactory.create(cfg.ingest.entity_extractor, cfg.ingest.entity_extractor_config)` and hands it to `LlmFactory.create(..., entity_extractor=)` (→ `RuleLLM.entity_extractor`, used for every `entity_mentions`), `Ingestor(..., entity_extractor=)` (→ `create_atom/revise_atom/conflict_atom(extractor=)` → `link_atom_axes(extractor=)` and `close_structured_events(extractor=)`), `FeedbackProcessor(..., entity_extractor=)` and `import_state_package(..., entity_extractor=)`. Every `extractor` parameter defaults to `None` = the rule provider, so callers outside the facade (graph operators, consolidation entity atoms) keep today's behaviour. Query-time hints (`mvmi/planner.py`) stay on `extract_entities`.

---

## §11 The `rule` LLM — deterministic heuristics per task (`hms_elite/llms/rule.py`, group B)

Module docstring: `"""Offline rule-based LLM: recognizes the HMS_TASK marker and answers each prompt contract with regex heuristics.\n离线规则 LLM：识别 HMS_TASK 标记，用正则启发式回答每个提示词契约。"""` Purpose: deterministic default provider | 确定性默认提供者. Input: system+user prompts built by §6 | §6 构造的提示. Output: JSON strings matching §6.3 | 符合 §6.3 的 JSON. Effect: none | 无.

Shared pipeline: `sections = parse_sections(user_prompt)`; every task reads only its documented sections; output is `json.dumps(obj, ensure_ascii=False)`; `RuleLlmConfig.language` selects lexicons (`auto` = EN and ZH both active). All regexes are from `utils/text.py` (§9.2). `strict=True` disables rule R9 (fallback facts). **Every regex in §11 is applied with `re.search` unless the pattern itself begins with `^`** (so `PREF_RE`, which is `$`-anchored only, matches "Actually I switched to black coffee now").

**Public pure functions** (module level in `hms_elite/llms/rule.py`, imported by group G — never duplicated): `decide_revision(new: dict, old: dict) -> tuple[Decision, str]` (§11.3 table for one pair; dicts carry the `## New Memories`/`## Candidate Existing Memories` keys), `propose_edges(nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> list[dict]` (§11.4; returns the `edges` list of the contract), `coarsen_rule(support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str) -> dict` (§11.5; returns the `interface_ep` dict), and the constants `EXCLUSIVE`, `PREDICATE_FAMILIES = {'PREF': {...}, 'IDENTITY': {...}, 'CONSTRAINT': {...}}`, `family(predicate) -> str`. The `RuleLLM._task_*` handlers are thin wrappers that call these functions on the parsed sections. Also public, imported by group C (`ingest/derive.py`, §6.5) and by the R1–R9 rules themselves: `statement_triple(text, obs_type, extractor=None) -> {'subject', 'predicate', 'objects', 'relation'}` (the third-person statement parser of §6.5: subject = opening side / PROPER entity or None; predicate / objects / relation cue by type from the R1–R9 cue tables), `constraint_objects(rest) -> list[str]` (R2 / R3's object rule), `predicate_of(text) -> str` (the R9 lexicon predicate), `observation_topics(text, obs_type) -> list[str]` (`TOPIC_TABLE` hits plus the type's tag, at most four), `mention_texts(text, extractor=None) -> list[str]` (entity mention texts minus pronoun-led CJK fragments).

### §11.1 `observation_extraction`

Per message in `## New Messages` with `role ∈ {user, assistant}` (system never present), per sentence of `sentences(text)`: skip sentences ending with `?`/`？`, greetings (`^(hi|hello|hey|thanks|thank you|ok|okay)\b`), and sentences with `< 3` Latin tokens and no CJK. Apply rules in order; **first match wins**; one observation per sentence. Every rule runs on the sentence with its terminal punctuation stripped; `text` re-adds the period. `IngestConfig.zh_rules` (§9.6) leaves these rules and `RuleLlmConfig.language` untouched; it changes only the compact / lines derivation (`statement_triple`, `predicate_of`, the entity / sentiment / topic lexica).

| # | Rule (EN; ZH twin) | type / relation / predicate / objects / sentiment / confidence |
|---|---|---|
| R1 PREFERENCE | `PREF_RE = r"\b(?P<subj>I|we)\s+(?:really\s+|absolutely\s+|still\s+|now\s+|also\s+)?(?:(?P<neg>don't|do not|no longer)\s+)?(?P<verb>like|love|prefer|enjoy|adore|hate|dislike|can't stand|cannot stand|switched to|switch to)\s+(?P<obj>[^,.;!?]+?)(?:\s+(?:over|rather than|instead of|more than)\s+(?P<alt>[^,.;!?]+))?\s*$"` (re.I, on the sentence stripped of terminal punctuation); ZH `(?P<subj>我|我们)(?:现在|已经|还是|也)?(?:很|更)?(?P<verb>喜欢|偏好|讨厌|不喜欢|改喝|改成|换成|爱喝)(?P<obj>[^，。；！？]+)` | `preference_event`; predicate `likes` for like/love/enjoy/adore/喜欢/爱喝, `prefers` for prefer/偏好, `dislikes` for hate/dislike/can't stand/讨厌/不喜欢; `switched to|switch to|改喝|改成|换成` → `likes` with `relation=supersession` (`moved to` belongs to R5, not here); `neg` flips likes↔dislikes and sets `relation=supersession` for `no longer`; `alt` → `relation=comparison`, objects `[clean_object(obj), clean_object(alt)]`, predicate `prefers`; else objects `[clean_object(obj)]`; if `SUPERSESSION_CUES_EN|ZH` matches the sentence → `relation=supersession`, confidence `0.9`, else `0.8`; sentiment `positive` for likes/prefers, `negative` for dislikes |
| R2 CONSTRAINT | `CONSTRAINT_RE` (whole sentence, re.I) or `CONSTRAINT_ZH_RE` | `constraint` / `constraint` / `must_not`; subject `user` if `subj ∈ {I, We}` else `project`; objects `[clean_object(rest minus its first token)]` when `rest` has ≥2 tokens else `[clean_object(rest)]`; sentiment `neutral`; confidence `0.9` |
| R3 MUST | `MUST_RE` and the sentence is imperative or has subject `we|you|the team` and no negation | `constraint` / `constraint` / `must`; objects as R2; confidence `0.85` |
| R4 IDENTITY | `r"\b(?:my name is|I am called|call me)\s+(?P<obj>[A-Z][\w-]+)"` → predicate `is_named`; `r"\bI(?:'m| am)\s+(?:a|an)\s+(?P<obj>[^,.;]+)"` → `is`; ZH `我叫(?P<obj>\S+)` → `is_named` | `fact` / `single_fact`; objects `[obj]`; confidence `0.85` |
| R5 LIVE/WORK | `r"\bI\s+(?P<verb>live|work|study)\s+(?:in|at|for)\s+(?P<obj>[^,.;]+)"`; ZH `我(住在|在.+?工作|在.+?上学)`; relocation twin (tried only when the live/work pattern misses) `MOVED_RE = r"\bI\s+(?:just\s+|recently\s+)?(?P<verb>moved|relocated)\s+(?:back\s+)?to\s+(?P<obj>[^,.;]+)"` (re.I), ZH `MOVED_ZH_RE = r"我(?:现在|已经|刚)?(?P<verb>搬到|搬去|搬回)(?P<obj>[^，。；！？]+)"` | `fact` / `single_fact` / `lives_in|works_at|studies_at`; objects `[clean_object(obj)]`; confidence `0.85`; the relocation twin → predicate `lives_in` with `relation=supersession` (its verb phrase `moved to|relocated to|搬到|搬去|搬回` is in the `_SWITCH_VERBS` lexicon shared with R1), so §11.3 rule 6 REVISEs the previous residence |
| R6 HABIT | `r"\b(always|usually|every (?:morning|day|week|weekend|month)|each (?:day|week|month)|twice a|tend to)\b"` with first person; ZH `每天|每周|通常|总是|习惯` | `habit` / `habit` / `does`; objects `[clean_object(sentence minus the subject pronoun and cue)]`; confidence `0.65` |
| R7 TEMPORAL | `extract_dates(sentence, base_date=Observation Date)` has a window whose kind is not `relative_this` | `temporal_fact` / `single_fact`; predicate `plans` if `FUTURE_CUES` matches else `did`; `date_observed` = `resolve_date(...)`; subject `user` if first person else first PROPER entity else `user`; objects = entity texts excluding the subject; sentiment via `sentiment()`; confidence `0.75` |
| R8 COMPARISON | `COMPARE_RE = r"(?P<a>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\s+(?:vs\.?|versus|or|and)\s+(?:the\s+)?(?P<b>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"` AND sentence has `compar|decid|choos|between|torn|还是|比较`; PROPER filtering applied to `a`, `b` | `fact` / `comparison` / `compares`; objects `[a, b]`; text `f"User compared {a} and {b}; no exclusive final choice was stated."`; sentiment `neutral`; confidence `0.8` |
| R9 FACT (fallback) | sentence has ≥1 PROPER/FILE/IDENTIFIER entity OR a first-person subject, and ≥4 tokens | `fact` / `single_fact`; predicate = first `VERB_LEXICON` token after the subject span, else `states`; subject `user` if first person else first PROPER entity else `user`; objects = remaining entity texts; confidence `0.6`; skipped when `strict=True` |

Common post-processing per observation:
- `event_idx` = the message idx; `base_date` = `date_of(msg['timestamp'])` when the `## New Messages` item carries a non-null `timestamp`, else `## Observation Date`; `date_observed = base_date` unless R7 resolved one (R7 resolves against the same `base_date`).
- `text`: R1 → `f"User {pred_phrase} {objects[0]}."` with `pred_phrase ∈ {likes: 'likes', prefers: 'prefers', dislikes: 'dislikes'}`; R1 with `alt` → `f"User prefers {a} over {b}."`; ZH R1 → `f"用户{喜欢|偏好|讨厌}{obj}。"`; R8 → the template above; R2/R3/R7/R9 → `rewrite_first_person(sentence)` (verbatim for third person; trailing `.` ensured; first char capitalized); R4/R5/R6 → `rewrite_first_person(sentence)`.
- `subject`: `user` when `is_first_person(sentence)` (§9.2, `re.search`, so "Yesterday I visited the Berlin office." has subject `user`), else as the rule says; for role `assistant` a first-person sentence has `subject = 'assistant'`; `attributed_to = 'assistant'` iff role assistant.
- `speaker = None` on every item: the rule provider never names a person (`_OBSERVATION_KEYS` carries the key so the item shape matches the §6.3 block).
- `topic_tags` = `topic_tags(sentence)` ∪ (`['preference']` for R1, `['constraint']` for R2/R3), table hits first, then the type tag; ≤ 4.
- `entity_mentions` = `[c.text for c in extract_entities(sentence)]` (subject entity included).
- `supersedes_hint_ids`: for R1 observations → ids of `## Existing Memories` whose text starts with `User likes|prefers|dislikes` (or `用户喜欢|偏好|讨厌`) AND (shares ≥ 1 `content_tokens` token with the observation text OR shares ≥ 1 `TOPIC_TABLE` topic with it); for every other rule → ids whose `content_tokens` share ≥ 2 tokens with the observation text; else `[]`. (S1: "User likes black coffee." vs existing "User likes oat-milk lattes in the morning." share the token `like` and the topic `coffee` → `["0"]`.)
- Assistant role: only sentences matching `ASSISTANT_CUE_RE = r"\b(I recommend|I suggest|I've (?:scheduled|booked|created)|I have (?:scheduled|booked|created)|I will|I'll|I created|I scheduled)\b"` OR `ACTION_RE` (the assistant reporting its own action) → R9 fact with `subject='assistant'`, `attributed_to='assistant'`, confidence `0.6`, text = `rewrite_first_person(sentence)` with `Assistant` instead of `User` (e.g. `chat_action_closure` → `"Assistant patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."`, predicate `patched`, objects `['retry.ts', 'billing_e2e']`).
- Dedup within the batch by `(type, predicate, tuple(objects), normalize(text))` (`normalize` as §11.3 rule 1), keeping the first — only literally repeated statements collapse (S2's three Sarah observations survive); ids sequential from `"0"`.

### §11.2 `ep_extraction` (chat messages only; structured events never reach this task)

Per `## Events` item: `acts = ACTION_RE.finditer(text)`; for each action match (one EP per action clause, max 3 per message): `a = f"{m['verb'].lower()} {m['target'].strip()}".strip()` (the target keeps identifier dots and stops before a conjunction/clause separator, §9.2); `y` = the shortest clause (`CLAUSE_SPLIT_RE` split, stripped) containing the first `OUTCOME_RE` hit in the first sentence at or after the action sentence, else `""`; `r` = `negative` if the outcome word ∈ `NEGATIVE_OUTCOMES`, `positive` if ∈ `POSITIVE_OUTCOMES`, else `unknown` when `y == ""` else `neutral`; `u` = `CAUSE_RE` group `u`/`uzh` of the same sentence (the text between the conjunction and the next `,`/`;`/sentence end), else the previous sentence, else `f"task: {first 60 chars of the message}"`; `b` = `' ; '.join(text of ## Known Constraints whose content_tokens ∩ message tokens ≠ ∅)`; `kappa.constraints` = those constraint texts; `kappa.visibility = 'private'`; `time_anchor` = the event's `timestamp`; `entity_mentions` = entities in `a`+`y`+`u`; `topic_tags = topic_tags(a + ' ' + y + ' ' + u)`; `uses_ep_ids` = `## Recent EPs` ids whose `a` shares an identifier token with this `a`; `confidence 0.7`; `event_idx` = item idx. Drop the EP if `not (u and a and (y or kappa.constraints))`.

Worked example (fixture `chat_action_closure`, text `I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed.`): `verb='patched'`, `target='retry.ts to add backoff'` → `a='patched retry.ts to add backoff'`; `CAUSE_RE` → `u='the webhook kept timing out'`; clauses `['I patched retry.ts to add backoff because the webhook kept timing out', 'billing_e2e failed']` → `y='billing_e2e failed'`, `r='negative'`; `entity_mentions=['retry.ts', 'billing_e2e']`; `topic_tags=['engineering', 'billing']`. This is golden `eps_chat_action_closure.json`.

### §11.3 `revision_decision` (rule table; first match wins; candidates ordered as presented) — implemented by the public `decide_revision(new, old)`

For each item in `## New Memories` against each `## Candidate Existing Memories` (stop at the first non-ADD decision; otherwise ADD):
1. `normalize(new.text) == normalize(old.text)` (lowercase, ws-collapsed, terminal punctuation stripped) → `NONE`.
2. `new.type != old.type` → `ADD`.
3. `new.type ∈ {temporal_fact, habit}` → `ADD`.
4. `new.subject != old.subject` → `ADD`.
5. same predicate AND `objects_overlap(new, old)` (any pair with equal normalized strings OR token-Jaccard ≥ 0.5) → `NONE` (duplicate information).
6. `new.relation == 'supersession'` AND `family(new.predicate) == family(old.predicate)` AND `new.date_observed >= old.date_observed` → `REVISE` (reason `"supersession cue"`). Families: `PREF = {likes, dislikes, prefers, avoids}`, `IDENTITY = {is_named, is, lives_in, works_at, studies_at}`, `CONSTRAINT = {must_not, must}`, else the predicate itself.
7. same predicate AND predicate ∈ `EXCLUSIVE = {lives_in, works_at, studies_at, is_named, prefers}` AND objects disjoint → `REVISE` if `new.date_observed >= old.date_observed` (reason `"exclusive attribute updated"`) else `CONFLICT` (reason `"older statement contradicts newer"`).
8. polarity flip (`{likes, dislikes}` on overlapping objects) → `REVISE` if `new.relation == 'supersession'` else `CONFLICT`.
9. constraint vs constraint with overlapping objects and predicates `{must, must_not}` differing → `CONFLICT`.
10. otherwise → `ADD`.
Output: one decision per new id with `target_id` = the matched candidate id (null for ADD) and the reason string above.

### §11.4 `edge_proposal` — implemented by the public `propose_edges(nodes, pairs, witnesses)`

Nodes carry the §6.2 shape (`u, b, a, y, r, subject, predicate, objects, ...` are always present; observation/event/skill nodes have `a` = atom text, `y`/`b`/`u` = ""). For each `[x, y]` in `## Candidate Pairs` with `x.time <= y.time` (swap otherwise): `causal` if `|content_tokens(x.y) ∩ content_tokens(y.u)| ≥ 2` or an identifier of `x.a` appears in `y.u` or (`x.r == 'negative'` and `topics(x) ∩ topics(y) ≠ ∅`); else `use` if an identifier of `x.a`/`x.y` appears in `y.a` or `y.b`; else `revision` if `x.a` and `y.a` share an identifier target, `x.r == 'negative'` and `y.r == 'positive'`; else `conflict` if both are observation nodes with the same `(subject, predicate)` and disjoint objects; else nothing. `temporal` and `distill` are never proposed by the rule (computed elsewhere). `witness_refs = [first anchor ref of y] (+ first of x for revision/conflict)`; `label = f"{type}: {x.a[:40]} → {y.u[:40]}"` (`u` empty → `y.a`); `confidence = min(0.9, 0.6 + 0.1 * evidence_count)` where evidence_count = number of the above conditions satisfied.

### §11.5 `coarsen`

Order `## Support Nodes` by `time`; `P` = nodes with kind ∈ `{ep, interface_ep}` (L2 input = EP nodes; L3 input = InterfaceEP nodes, whose `u,b,a,y,r,constraints` come from their `Slots`); `P` empty → `{"interface_ep": null}` (the caller returns None). `slots.u = P[0].u`; `slots.a = ' → '.join(p.a for p in P)` (max 8, then `…`); `slots.y` = last non-empty `p.y`; `slots.r = 'negative' if any(p.r == 'negative') and P[-1].r != 'positive' else P[-1].r`; `slots.b = ' ; '.join(ordered_union(*[p.b.split(' ; ') for p in P if p.b], kappa.constraints))` (split first, then first-appearance unique — constraint texts never repeat); `kappa.constraints` = `ordered_union(*[p.constraints for p in P])`, `kappa.visibility` = most restrictive (lowest rank) among nodes; `slot_support`: `u → [P[0]]`, `y → [node of the chosen y]`, `r → [P[-1]] (+ negative nodes when r is negative)`, `a → all P`, `b`/`kappa → nodes contributing constraints` (or `[P[0]]` when empty); invariants: every constraint text present in ≥ 2 nodes' constraints → `{"kind": "constraint", "support_refs": those nodes}`; every negative node → `{"text": f"do not: {a}", "kind": "failed_path", "support_refs": [node]}`; residuals: nodes with `r == 'unknown'` and empty `y` → `missing_outcome`; `## Internal Edges` of type `conflict` → `unresolved_conflict` (support = both endpoints); sentences ending in `?` in any `y` → `open_question`.

### §11.6 `procedural_summary`

Markdown from the messages: objective = first user message (≤120 chars); one numbered step per assistant message with an `ACTION_RE` match (`**Agent Action**:` the action sentence, `**Action Result**:` the next tool/user message text ≤200 chars or `n/a`); `**Key Findings**:` sentences with `OUTCOME_RE` positive hits; `**Errors & Challenges**:` negative hits; `**Current Context**:` last message ≤200 chars; `**Progress Status**:` `"completed"` if the last outcome is positive, `"blocked"` if negative, else `"in progress"`.

### §11.8 `observation_extraction_compact` / `memory_extraction_compact` (§6.5)

`_task_observation_extraction_compact(sections)` = `_task_observation_extraction` projected onto `COMPACT_OBSERVATION_KEYS` by `_compact_observations`: the five required keys always (`speaker` stays `None` — plain chat names nobody; the subject is read back from `text` by `statement_triple`), `entities` = the full observation's `entity_mentions` and `supersedes_hint_ids` only when non-empty, `when` = `date_observed` only when it differs from the witnessed message's date (`_message_dates`: the `## New Messages` timestamp, else `## Observation Date`), `valid_to` only when set. `_task_memory_extraction_compact(sections)` = `_task_memory_extraction` with the same projection of `observations`; `eps` and `decisions` are returned unchanged — the decisions were judged by `decide_revision` on the full R1–R9 observations (objects, predicate) before the projection. Both handlers are registered in `_TASKS` under their task names; `tests/test_extraction_compact.py` asserts that on every chat fixture the compact facade stores the same atoms, history rows and graph edges as the full one.

`_task_memory_extraction_lines(sections) -> str` (§6.6) = `_task_memory_extraction` rendered as the tab-separated text: one `<event_idx>⇥<T>⇥<text>` line per observation (`T` from `configs.prompts.TYPE_LETTERS`, `⇥@date_observed` appended only when it differs from the witnessed message's date — `_message_dates`, as in the compact projection), an `EP` section only when `eps` is non-empty (`<event_idx>⇥u=⇥a=⇥y=⇥r=` plus `⇥k=c1;c2` for matched constraints; `;` inside a constraint becomes `,`), a `DEC` section only for non-ADD decisions (`<id + 1>⇥<verdict>⇥<target_id>⇥<reason>` — the R1–R9 ids are positional, so the 1-based OBS line is id + 1); tabs and newlines inside a field become spaces (`_line_field`); the reply ends with a newline. The decisions are judged on the full R1–R9 observations before the rendering. `generate_response` returns the string as is (no JSON dump). `tests/test_extraction_lines.py` asserts row- and table-identity with the full facade on every chat fixture (except the §6.6 speaker-rule `actor_id`) and the exact rendering for an EP, a dated statement and a REVISE.

### §11.7 `ScriptedLLM` (`hms_elite/llms/scripted.py`)

`ScriptedLLM(config)` with `responses: deque` of `str | dict | Exception | Callable[[list[dict]], str | dict]`; `generate_response` pops left (dict → `json.dumps`, Exception → raised, so `LLMError ... from e` is testable; a callable is invoked with the `messages` list and its return value is treated like a str/dict response — used by test 31 to assert `store.in_transaction is False` at call time); `loop=True` re-queues; empty queue → `LLMError(LLM_003, suggestion="queue more responses")` when `raise_when_empty` else `""`; `calls: list[dict]` records `{'task', 'messages', 'kwargs'}`; `push(*responses)`. Tests assert `len(llm.calls) == 0` on structured paths.

---

## §12 Ingest (group C: `hms_elite/ingest/`)

Package docstring: `"""Ingest plane: SourceEvent → Anchor → routing → extraction → atoms, with lock discipline Txn A → unlocked → Txn B.\n摄取平面：事件→锚点→分流→抽取→原子；事务纪律 A→无锁→B。"""`

### §12.1 `ingest/events.py`

- `events_from_messages(messages: list[dict], *, scope: Scope, event_time: str, user_meta: dict, visibility: str, source_system: str = "mem0_api") -> list[SourceEvent]` — one `message` event per coerced message | 消息→事件. In: coerced messages (§9.5), scope, event time | 消息、范围、时间. Out: events in order (`system` role events included, flagged for no extraction) | 事件列表. Effect: none | 无. Notes: payload `{'text': content, 'dropped_parts': n}` (+ `parts` / `described_parts` built from `image_parts`, §4.3: described parts are merged into `text` as `[Image: <description>]` lines by `merge_image_descriptions(message) -> str` and subtracted from `dropped_parts`); `role`, `actor_id = msg.get('name')`, `raw_ref = default_raw_ref(ev, index)`; per-message `timestamp` key (if present in the dict) overrides `event_time`.
- `events_from_dicts(items: list[SourceEvent | dict], *, scope_fallback: Scope | None, visibility_default: str) -> list[SourceEvent]` — validate `add_events` input | 校验结构化输入. Notes: dict without scope keys uses `scope_fallback` (none → `VALIDATION_001`); `object_type ∈ INTERNAL_OBJECT_TYPES` → `ValidationError(VALIDATION_007, details={'index', 'object_type'})`; payload errors → `VALIDATION_005` with `details['index']`; `event_type` defaults: `tool_result` for `test_log`, for `sensor_trace`, and for `tool_call` whose payload carries a non-null `status` or `result`; else `created` (an explicit `event_type` is kept); `role` defaults to `tool` for structured types; result sorted stable by `(timestamp, index)`.
- `validate_payload` — re-exported from `hms_elite.models` (§4.3); NOT re-implemented here | 载荷归一化（来自 models）.
- `default_raw_ref(ev: SourceEvent, index: int) -> str` — `message → f"chat://{scope.key()}/{index}"`, `tool_call → f"tool://{tool}"`, `file_diff → f"file://{path}"`, `test_log → f"test://{suite}"`, `sensor_trace → f"sensor://{sensor}"`, others → `f"hms://{object_type}/{timestamp}"` | 默认引用.
- `anchor_events(store: StateStore, events: list[SourceEvent]) -> list[tuple[SourceEvent, Anchor, bool]]` — **Txn A**: `insert_source_event` + `get_or_create_anchor` for each; `dedup=True` when the event already existed | 事务 A 落锚. Effect: writes source_events, anchors | 写事件与锚点.
- `system_event(store, *, scope: Scope, object_type: str, event_type: str, payload: dict, actor_id: str | None = None, visibility: str = "private") -> tuple[SourceEvent, Anchor]` — the shared prologue for update/delete/feedback/checkpoint/rollback/scheduler (object types from `INTERNAL_OBJECT_TYPES`) | 系统事件前奏. Effect: writes one event + anchor inside the current transaction.

### §12.2 `ingest/router.py`

- `route_event(ev: SourceEvent) -> set[str]` — `{'observation'}` for user/assistant messages; `{'observation', 'state'}` when `has_action_closure(text)` or `ev.user_meta.get('route') == 'state'`; `{'structured'}` for tool_call/file_diff/test_log/sensor_trace; `set()` for system-role messages and internal types | 事件分流. Effect: none.

### §12.3 `ingest/extraction.py` (runs with NO lock held)

- `neighbourhood(store, index, embedder, *, scope: Scope, texts: list[str], k: int) -> tuple[list[dict], list[SourceEvent]]` — `existing = [{'id': '0'.., 'text'}]` from `index.search(embed(parse_messages), candidate_ids=store.valid_ids(scope)['atoms'], top_k=k)` (group A helper — C never imports `hms_elite.mvmi`) plus `last_events(scope, object_types=['message'], limit=k)` | 邻域上下文. Effect: read-only.
- `extract_observations(llm, *, events: list[tuple[SourceEvent, Anchor]], existing: list[dict], last_events: list[SourceEvent], observation_date: str, current_date: str, custom_instructions: str | None, scope: Scope, id_map: dict[str, str], output: str = 'full', entity_extractor: EntityExtractorBase | None = None) -> list[Observation]` — one `observation_extraction` call (`observation_date = date_of(events[-1][0].timestamp)`, `current_date = date_of(store.now())`, both computed by the pipeline; each `## New Messages` item carries its own `timestamp`, which the extractor uses as that message's `base_date`, §11.1); each item → `Observation(anchor_ids=[anchor of event_idx], ...)`. **Coerce, never reject on vocabulary** (per-item guard): an item is skipped with `logger.warning` only when (a) it cannot be witnessed — `event_idx` out of range or unreadable (digit strings `'0'` and integral floats `0.0` are read; a missing `event_idx` with exactly one new message means that message), (b) it has no `text`, or (c) a field carries a malformed type: list fields (`objects`, `topic_tags`, `entity_mentions`, `supersedes_hint_ids`) accept a list or a single string (→ one-element list, never its characters; dict items are flattened to their text; a comma-separated tag string is split) and text fields (`subject`, `predicate`, `text`) must be strings — any other shape skips the item the same way, so a malformed item never escapes `HMS.add()` as a raw `TypeError` nor persists as a corrupt atom. Every other mismatch is coerced to a documented value: `type` / `relation` / `sentiment` → lower-cased, spaces and hyphens → `_`, then exact member, alias table (`preference` → `preference_event`, `rule` → `constraint`, `update` → `supersession`, `happy` → `positive`, …), a member present as whole words (`very positive` → `positive`; never a substring — `artifact` is not `fact`) or a prefix of ≥ 3 chars, else the default `fact` / `single_fact` / `unknown` with a DEBUG line; a **negated** value (`not positive`, `non-habit`, `not a constraint`) never folds to the member it names but to that default with a WARNING; `confidence` → a number in [0, 1] as is, `'85%'` and a bare number in (10, 100] read as a percentage, `high/medium/low` → 0.9/0.7/0.4, None/'' → 0.7, and any other number (negative, in (1, 10], > 100, NaN) → 0.7 with a WARNING naming the field — never a value near 0 that the prune phase would archive as `decayed`; `valid_from` / `valid_to` → ISO or date-only instants kept (date-only → midnight UTC), anything else (`next week`, a bare year) → `None`; a `valid_to` earlier than the floor `atom_from_observation` will assign (`valid_from`, else date_observed midnight for a `temporal_fact`, else the witnessing message's timestamp) is dropped to `None` with a warning, so `MemoryAtom.valid_to >= valid_from` can never abort the batch; `date_observed` → `YYYY-MM-DD` (`/` and `.` separators normalised), else `observation_date`; `attributed_to` → `user|assistant` case-insensitively (`system/tool/bot/ai/model` = assistant side, `human/me` = user side), a collective word (`group`, `others`, `everyone`, …) → the witnessed message's role, any other string → the message's role **and** that string kept as `speaker`; `speaker` → a non-empty string that is neither a side word nor a collective word, else `None`; the speaker (from either key) is appended to `entity_mentions` when not already there (case-insensitive); in compact mode (§6.5) the optional `entities` hint is the one list field whose malformed shape does not skip the item — `_entities_hint` runs `_str_list` and ignores a number / dict with a warning, the derivation falling back to extractor(`text`) ∪ `speaker`; `supersedes_hint_ids` mapped through `id_map` ('0'.. → atom_id). Second line of defence: the pipeline drops with a warning — and lists under `IngestResult.dropped` as `{'kind': 'observation', 'text', 'reason': 'invalid_atom'}` — any observation whose atom still fails a `MemoryAtom` invariant, instead of aborting the batch | 观察抽取.
- `extract_eps(llm, *, events, known_constraints: list[MemoryAtom], recent_eps: list[EP], observation_date, scope) -> list[EP]` — `ep_extraction` call; `uses_ep_ids` mapped `e0.. → ep_id`; `kappa.evidence = [anchor]`; anchors `[anchor of event_idx]`; same coercion as `extract_observations` for `kappa.constraints`, `entity_mentions`, `topic_tags`, `uses_ep_ids` (a string is one element) and the `u/b/a/y` slots (non-string → item skipped with `logger.warning`, so a character list can never satisfy `Slots._closure`); vocabulary follows the observation discipline: `r` through the Reward alias table (`success/pass/ok/fixed` → positive, `fail/error/broken` → negative, `partial/mixed` → neutral, `none/n/a` → unknown), a negation → `unknown` with a WARNING, else `unknown` with a DEBUG line; `confidence` exactly as for observations; an unreadable `time_anchor` → the event's timestamp; `kappa` given as a list is a constraint list, as a string one constraint, any other non-object → `{}`; an out-of-vocabulary `kappa.visibility` → the event's visibility; `uses_ep_ids` bare indices `0` → `e0`; an `eps` root that is a single object is read as a one-item list (one warning) | EP 抽取.
- `decide_revisions(store, llm, *, new: list[tuple[Observation, MemoryAtom]], scope: Scope, current_date: str, revision_mode: str) -> list[tuple[Decision, str | None, str]]` — per new observation-atom: candidates = `store.revision_candidates(...)` (top 5 by time); exact `hash` match among candidates → `NONE` without an LLM call; no candidates → `ADD`; `revision_mode == 'off'` → `ADD`; `'deferred'` → `ADD` (consolidate's `detect_conflicts` decides later); else ONE batched `revision_decision` call; verdict coerced — stripped, upper-cased, spaces/hyphens → `_`, a one-item list yields its item, None/'' → `ADD`, synonyms folded (`UPDATE/UPDATED/REPLACE/SUPERSEDE(S)/MERGE/REVISED/REVISION` → REVISE, `CONTRADICT(S)/CONFLICTING/CONFLICTS` → CONFLICT, `KEEP/NEW/INSERT/ADDED/CREATE/ADDITIVE` → ADD, `DUPLICATE/DUP/SAME` → NONE — the only fold that discards the new observation, so it is traced with a WARNING); `SKIP/NOOP/IGNORE` and any other word are **unknown** verdicts → `ADD` with warning (the observation is stored); `target_id` mapped back (a one-item list yields its item; a candidate atom_id passes through); unknown target → `ADD` with warning — in the single-call `extract_all` path the observation's own resolved `supersedes_hint_ids` stand in first when they name exactly one candidate; there a `decisions` root that is one object, or a dict keyed by observation id, is read as the list, and decision ids join observation ids exactly, then by digits (`obs_0` ↔ `0`), then positionally when counts agree (one warning); a missing decision → `ADD` with warning | 修订决策. Effect: read-only.
- `procedural_summary(llm, *, messages: list[dict], prompt: str | None) -> str` — `generate_text('procedural_summary', build_procedural_messages(...))`; empty → `ValueError("Procedural memory generation failed")` | 过程摘要.
- `hms_elite/ingest/derive.py` (§6.5) — `expand_compact_item(item, *, obs_type, speaker, role, message_date, extractor) -> dict` builds the full item shape of a compact observation (`llms.rule.statement_triple` for subject / predicate / objects / relation cue, `subject_for` for the speaker / role fallback, `relation_of`, `is_full_datetime`, `COMPACT_CONFIDENCE`; no cue table of its own); `expand_lines_ep(item, *, extractor) -> dict` (§6.6) fills a parsed lines EP item's `entity_mentions` / `topic_tags` from `'<a> <y> <u>'` and an empty `b` from its k= constraints, as the rule LLM fills its own EPs, before `extract_all(output='lines')` hands it to `_ep_from_item`; `extract_observations(..., output='compact', entity_extractor=...)` and `extract_all(..., output='compact', entity_extractor=...)` run `_compact_observation_from_item` (witness first, then expansion, then the full-contract coercion) under the `observation_extraction_compact` / `memory_extraction_compact` tasks; `output` ∉ `EXTRACTION_OUTPUTS = ('full', 'compact', 'lines')` → `ValidationError(VALIDATION_013)`; `output = 'lines'` on `extract_observations` (the staged path) → the same code with `details.supported = ['single']` | 紧凑契约的本地推导.
- `hms_elite/ingest/lines.py` (§6.6) — `parse_lines(text, *, what) -> {'observations', 'eps', 'decisions'}`: the tolerant parser of the tab-separated lines reply into compact-shaped items (id = the 1-based OBS line number the model wrote: every tabbed OBS line consumes a number whether or not it stores, a tab-less line is a continuation and consumes none; a reply that opens no section logs one warning containing `NO_SECTION_MARK`); `extract_all(..., output='lines')` runs `llm.generate_lines` + `parse_lines`, then `_compact_observation_from_item(..., speaker_from_subject=True)` (the leading proper name becomes `speaker`), `derive.expand_lines_ep` then `_ep_from_item` for EPs, and joins DEC lines by line number (`_parse_decision`; no line → ADD, silent) | 行式契约的解析.

### §12.4 `ingest/closure.py` — structured closure builder (NO LLM)

- `event_atom_text(ev: SourceEvent) -> str` — `file_diff → f"file_diff {path}: {summary or first non-empty patch line}"`; `test_log → f"test_log {suite}: {status} ({passed} passed, {failed} failed)"` + `f": {', '.join(names[:3])}"` when failures (counts `?` when None); `tool_call → f"tool_call {tool}({args_summary}) → {status or 'unknown'}"` with `args_summary = ', '.join(f"{k}={str(v)[:40]}" for k, v in sorted(args.items()))`; `sensor_trace → f"sensor_trace {sensor}: {summary or f'{len(samples)} samples'}"` | 事件原子文本.
- `action_text(ev) -> str` — `file_diff → f"patch {path}: {summary or first patch line}"`; `tool_call → f"{tool}({args_summary})"` | 动作文本.
- `outcome_text(ev) -> tuple[str, str]` — `test_log → (f"{suite} {status}" + (f": {', '.join(names)}" if names else ""), 'negative' if status == 'failed' else 'positive')`; `tool_call(status='error') → (result or 'error', 'negative')` | 结果文本与奖励.
- `build_ep_from_closure(*, action: tuple[SourceEvent, Anchor], outcome: tuple[SourceEvent, Anchor], background: list[MemoryAtom], previous: EP | None, scope: Scope, extractor: EntityExtractorBase | None = None) -> EP` — `u = action.payload.intent or (previous.slots.y if previous else None) or f"modify {path}" / f"call {tool}"`; `constraints = ordered_union([c.text for c in background], action.payload['constraints'])` (background atoms first, then the action payload's own `constraints`); `b = ' ; '.join(constraints)`; `a = action_text(action)`; `y, r = outcome_text(outcome)` (`('', 'unknown')` for a self-closing open `tool_call`); `kappa = Kappa(constraints=constraints, visibility=min-rank(action.visibility, outcome.visibility), evidence=[outcome_anchor, action_anchor])` (so a self-closing tool_call whose payload carries constraints always satisfies the `Slots` closure validator); `anchor_ids = [outcome_anchor, action_anchor]`; `time_anchor = outcome.timestamp`; `entity_mentions = entities(a + ' ' + y + ' ' + u)` texts; `topic_tags = topic_tags(u + ' ' + a + ' ' + y)`; `extracted_by = 'rule@closure-v1'`; `confidence 0.8` | 闭包构建 EP.
- `derived_constraint(ep: EP, *, scope) -> Observation` — `type=constraint, subject='project', predicate='must_not', objects=[ep.slots.a], relation=constraint, text=f"Do not re-apply '{ep.slots.a}': {ep.slots.y}", date_observed=date_of(ep.time_anchor), anchor_ids=ep.anchor_ids, entity_mentions=ep.entity_mentions, topic_tags=['constraint'] + ep.topic_tags, confidence=0.85, extracted_by='rule@closure-v1', sentiment='negative'` | 派生约束.
- `ClosureBatch` (dataclass): `open_actions: dict[str | None, list[tuple[SourceEvent, Anchor]]]`, `eps: list[EP]`, `derived: list[Observation]`, `edges: list[GraphEdge-spec]`.
- `close_structured_events(store, *, anchored: list[tuple[SourceEvent, Anchor, bool]], scope: Scope, background: list[MemoryAtom], max_gap_hours: float, temporal_window_hours: float, extractor: EntityExtractorBase | None = None) -> ClosureBatch` — algorithm (events already sorted; `sensor_trace` events are skipped here — they only get event atoms): for each `(ev, anc)` with `object_type == tool_call|file_diff`: push onto `open_actions[ev.scope.run_id]`; a `tool_call` with `status == 'error'` instead self-closes immediately (`build_ep_from_closure(action=(ev,anc), outcome=(ev,anc), ...)`, `anchor_ids=[anc]`); for `test_log`: `action = open_actions[run].pop()` if any, else `store.last_open_action(scope, run_id=run, before=ev.timestamp, max_gap_hours=max_gap_hours)` (§8: the newest tool_call/file_diff event in the run not cited by any EP within the gap — cross-batch closure), else `None` → only the event atom is produced (an EP needs an action); `previous = the last EP built in this batch for the run or store.last_ep(scope, run_id=run)`; build the EP; record edge specs: `('use', ep → c.atom_id, witness=[action_anchor], confidence 0.8)` per background constraint; `('temporal', previous → ep, witness=[outcome_anchor], confidence 0.5)` when previous exists and `age(previous.time_anchor, ep.time_anchor) ≤ temporal_window_hours` (the caller passes `config.graph.temporal_window_hours`; `max_gap_hours` is `config.ingest.closure_max_gap_hours` — two different knobs); `('causal', previous → ep, witness=[action_anchor, previous.anchor_ids[0]], confidence 0.7)` when previous exists and (`previous.slots.r == 'negative'` or entity overlap or `u` was derived from `previous.slots.y`); negative EP → `derived_constraint` + `('distill', ep → derived_atom, witness=[outcome_anchor], confidence 0.9)`. At batch end remaining open `tool_call`s self-close as `r='unknown'` EPs with `y=''` ONLY when `constraints` present in their payload (else they stay open, no EP); open `file_diff`s stay open | 结构化闭包.

### §12.5 `ingest/atoms.py` — the single materialization primitive

- `atom_from_observation(obs: Observation, *, scope, time_anchor: str | None = None, actor_id: str | None = None) -> MemoryAtom` — type map: `fact|preference_event|habit|constraint → atom_type=state, scale_level=state`; `temporal_fact → event/event` with `time_anchor = date_observed + 'T00:00:00.000000+00:00'`; others `time_anchor = anchor timestamp` (passed in) or `obs.valid_from` or date_observed midnight; `valid_from = obs.valid_from or time_anchor` (so a `temporal_fact` without `valid_from` starts at date_observed midnight); `content` per §4.2; `subject/predicate/obs_type` promoted; `sentiment`, `topic_tags`, `role=extracted`, `source_kind=observation`, `source_id=obs_id`, `attributed_to`, `extracted_by`; `actor_id = actor_id` (the witnessing event's, i.e. the message `name` — part of the witnessed payload hash, hence authoritative) `else obs.speaker else None` (`_observation_actor`) | 观察→原子.
- `relation_atom_from_observation(obs, *, scope, time_anchor=None, actor_id=None) -> MemoryAtom` — the §4.2 `relation` atom of a `relation='comparison'` observation; same `actor_id` rule (the pipeline passes the observation atom's `actor_id`) | 关系原子.
- `atom_from_ep(ep: EP, *, scope) -> MemoryAtom` — `atom_type=episode, scale_level=L1_micro, text=ep.slots.gloss(), content={'text', 'ep_id', 'slots'}, time_anchor=ep.time_anchor, sentiment by r (negative→negative, positive→positive, else neutral), source_kind=ep, source_id=ep_id, source_ep_ids=[ep_id], topic_tags, entity_ids from entity_mentions, confidence=ep.confidence, visibility=ep.visibility, extracted_by=ep.extracted_by` | EP→原子.
- `atom_from_event(ev: SourceEvent, anchor: Anchor, *, scope) -> MemoryAtom` — `atom_type=event, scale_level=event, role=tool, text=event_atom_text(ev), content={'text','event_id','object_type','payload'}, time_anchor=ev.timestamp, source_kind=event, source_id=event_id, actor_id, extracted_by='raw'` | 事件→原子.
- `atom_from_message(ev, anchor, *, scope) -> MemoryAtom` — `atom_type=message, scale_level=L1_micro, role=ev.role, actor_id, text=payload.text, content={'text','role','actor_id','dropped_parts'}, source_kind=message, extracted_by='raw'` | 消息→原子.
- `EntityKey = tuple[str, str, str, tuple[str, ...]]` = `(key, canonical, entity_type, aliases)` — `key = EntityCandidate.key` (the identity stored in `entities.normalized_text`, e.g. `retry.ts`), `canonical = EntityCandidate.text`, `aliases = (candidate.normalized,)` when it differs from `key` (FILE: the full lower-cased path), else `()`.
- `link_atom_axes(atom: MemoryAtom, *, extra_mentions: Sequence[str] = (), extractor: EntityExtractorBase | None = None) -> tuple[list[EntityKey], list[str]]` — entity keys from `extract(atom.text)` ∪ `extra_mentions` (each mention re-classified by `extract`; dedup by `key`, first wins) where `extract = extractor.extract` or `extract_entities` when `extractor is None` (§10.5) and topics `ordered_union(atom.topic_tags, topic_tags(atom.text))` | 轴链接. Notes: `insert_atom` passes `normalized=key, canonical=canonical, entity_type=entity_type, aliases=aliases` to `get_or_create_entity` (§8).
- `atom_from_entity(entity: dict, *, scope: Scope, anchor_ids: list[str], mentions: int) -> MemoryAtom` — §4.2 `entity` row: `atom_type=entity, scale_level=entity, text=entity['canonical_text'], content={'text', 'entity_id', 'entity_type', 'aliases', 'mentions'}, subject=canonical_text, predicate='is_entity', role=extracted, source_kind=entity, source_id=entity_id, extracted_by='rule@entity-v1', confidence=0.7, time_anchor=None, topic_tags=[]`; the caller links it to its own entity via `extra_mentions=[canonical_text]` | 实体→原子.
- `create_atom(store, index, embedder, atom: MemoryAtom, *, extra_mentions=(), vector: list[float] | None = None, history_event: str | None = "ADD", history_anchor_id: str | None = None, actor_id=None, role=None, history_details: dict | None = None) -> MemoryAtom` — **inside Txn B** (`history_event=None` → no history row; `revise_atom` uses that): `entity_keys, topics = link_atom_axes(...)`; `store.insert_atom(atom, entity_keys=..., topics=...)`; `index.upsert(atom_id, vector or embedder.embed(text, 'add'), embedder=embedder.identity)`; `store.add_history(memory_id, None, text, history_event, created_at=atom.created_at, updated_at=atom.created_at, actor_id, role, anchor_id=history_anchor_id or atom.anchor_ids[0], atom_id)` | 原子创建原语. Effect: writes memory_atoms, atom_anchor, entities, atom_entity, atom_topic, atom_vectors, graph_nodes, history. Notes: embedding is computed BEFORE the transaction by the caller whenever possible (`vector=` passed); `EmbeddingError` propagates and the transaction rolls back.
- `revise_atom(store, index, embedder, *, old: MemoryAtom, new: MemoryAtom, witness_anchor_ids: list[str], at: str, reason: str = "revised", created_by: str = "ingest", history_event: str = "UPDATE", label: str = "") -> MemoryAtom` — `at` semantics: **ingest-time** REVISE/CONFLICT (§12.6) pass `at = max(new.valid_from, old.valid_from)` (the new evidence's event time — S1's archived row gets `valid_to == '2026-04-20T08:00:00.000000+00:00'`), whereas `update()` and feedback handlers pass `at = store.now()`; `new.memory_id = old.memory_id; new.version = old.version + 1`; `store.archive_atom(old.atom_id, reason, at, valid_to=at)` (rows' `updated_at` and the history row's `updated_at` = `at`; the new atom's `created_at` = `store.now()` as always); `create_atom(new, history_event=None)`; `store.insert_edge(GraphEdge(revision, old.atom_id → new.atom_id, witness_anchor_ids, created_by, proposer='rule', confidence 0.9, label))`; `store.add_history(memory_id, old.text, new.text, history_event, created_at=old.created_at, updated_at=at, anchor_id=witness_anchor_ids[0], atom_id=new.atom_id, edge_id, successor_id=new.atom_id)` | 修订原语. Effect: archive + insert + edge + history.
- `conflict_atom(store, index, embedder, *, old: MemoryAtom, new: MemoryAtom, witness_anchor_ids, at, label) -> MemoryAtom` — `create_atom(new)` (new lineage); `insert_edge(conflict, old → new, witnesses, confidence 0.6, created_by ingest)`; history `CONFLICT` on old lineage (`old_memory=old.text, new_memory=new.text, details={'conflicts_with': new.memory_id}`) | 冲突原语.

### §12.6 `ingest/pipeline.py`

```python
@dataclass
class IngestResult:
    rows: list[dict]                 # Mem0 rows + 'event' ('ADD'|'UPDATE'|'NONE') + 'revises'? + 'conflicts_with'?
    events: list[dict]               # [{'event_id','anchor_id','object_type','dedup'}]
    ep_ids: list[str]; edge_ids: list[str]; constraint_ids: list[str]; dropped: list[dict]
    downgraded: list[dict]           # verdicts the §12.6 step-6 guard turned into ADD: [{'kind': 'verdict', 'verdict': 'NONE'|'REVISE'|'CONFLICT', 'target_id', 'similarity', 'text'}]

class Ingestor:
    def __init__(self, store, index, embedder, llm, config: HMSConfig, entity_extractor: EntityExtractorBase | None = None, extraction_mode: str = 'single', extraction_output: str | None = None) -> None   # None → config.ingest.extraction_output (no hard-coded default; the facade passes the configured value)
        # entity_extractor None → EntityExtractorFactory.create(config.ingest.entity_extractor, config.ingest.entity_extractor_config) (§10.5); the facade injects its shared instance
    def ingest_messages(self, messages: list[dict], *, scope: Scope, user_meta: dict, infer: bool, memory_type: str | None, prompt: str | None,
                        event_time: str | None, valid_to: str | None) -> IngestResult
    def ingest_events(self, events: list[SourceEvent], *, infer: bool) -> IngestResult
```

`ingest_messages` steps (normative order):
1. `events = events_from_messages(...)`; **Txn A** `anchored = anchor_events(store, events)`.
2. `infer=False` → for each non-system message: `atom_from_message` (+ `valid_to`), embeddings computed unlocked, **Txn B** `create_atom` each (history `ADD` with `actor_id`, `role`); rows carry `actor_id`, `role`. Return.
3. Procedural branch (`scope.agent_id and memory_type == 'procedural_memory'`): `md = procedural_summary(...)` unlocked; **Txn B** one atom `MemoryAtom(atom_type=episode, scale_level=L1_micro, source_kind=procedural, memory_type='procedural_memory', anchor_ids=[all message anchors], text=md, role=assistant)`; return one row.
4. Routing: `obs_events = [(ev, anc) for role in user/assistant]`; `ep_events = those with 'state' in route_event(ev)`.
5. Unlocked: `observation_date = date_of(events[-1].timestamp)`, `current_date = date_of(store.now())` (per-message `timestamp` overrides inside the prompt, §11.1); `existing, last = neighbourhood(...)`; `observations = extract_observations(...)`; `eps = extract_eps(...)` when `ep_events`; build atoms (`atom_from_observation`, `atom_from_ep`); Mem0 exact dedup: drop atoms whose `hash` ∈ hashes of `existing` valid heads ∪ seen-in-batch (dropped ones get `event='NONE'` rows pointing at the existing memory_id and add a `support` `atom_anchor` row in Txn B); `vectors = embed_batch(texts)` (per-item fallback; all fail → `EmbeddingError`); `decisions = decide_revisions(...)`.
6. **Txn B** in order: `insert_observation` / `insert_ep` (with `atom_id` back-pointers after creation); apply each decision after the verdict guard (`ingest.verdict_min_similarity`, §6.3: a REVISE corroborated by `supersedes_hint_ids` naming the target or an equal subject + predicate always passes; otherwise a verdict whose atoms carry different subjects becomes ADD; a same-subject CONFLICT passes; a NONE / uncorroborated REVISE whose head-removed `verdict_similarity` against its valid target is below the bar, or that shares a single token without containing the shorter statement, becomes ADD — DEBUG-logged and listed in `IngestResult.downgraded`; 0.0 disables) — `ADD` → `create_atom`; `REVISE` → re-check `store.get_atom(target).archived_reason is None` (else downgrade to ADD), `revise_atom(..., at=max(new.valid_from, old.valid_from))` witnessed by the new atom's anchors (event-time `valid_to`, §12.5); `CONFLICT` → `conflict_atom(..., at=new.valid_from)`; `NONE` → `add_atom_anchor(target, anchor, 'support')` (a no-op when the anchor is already the primary witness — same-timestamp re-add), `bump_mentions(target)`; a `relation='comparison'` observation additionally creates its `relation` atom (§4.2) when `ingest.emit_relation_atoms`; inline EP edges — `temporal` between consecutive EP nodes of the same run within `graph.temporal_window_hours` (witness = later EP's primary anchor, proposer `timestamp`), `use` for validated `uses_ep_ids` (witness = using EP's primary anchor); `store.upsert_transition(Transition.from_ep(ep, scope=scope, permission_scope=<primary anchor's permission_scope>))` (§4, group A; §16.4 documents the row) for every new EP; `valid_to` hint applied to every created atom.
7. Rows in extraction order: `{**atom.to_public_dict(), 'event': 'ADD'|'UPDATE'|'NONE', 'revises'?: old_atom_id, 'conflicts_with'?: [memory_id]}` (§4.1, group A). `actor_id` on an `infer=True` observation row = the message's `name` when given, else the LLM-named `speaker` (§12.5), else absent — a deliberate divergence from Mem0, which attaches `actor_id` only on its `infer=False` raw path (MEM0_MAP step 8a); `role` stays limited to `message` atoms.

`ingest_events` steps: **Txn A** anchor all (sorted); split by object_type: `message` events → steps 4–7 above batched per scope (LLM path, unless `infer=False`); structured events (`tool_call`, `file_diff`, `test_log`, `sensor_trace`) → unlocked: `background = valid constraint atoms in scope (obs_type='constraint', ordered by time_anchor)`, `batch = close_structured_events(store, anchored=..., scope=..., background=..., max_gap_hours=config.ingest.closure_max_gap_hours, temporal_window_hours=config.graph.temporal_window_hours)`, atoms built (`atom_from_event` for EVERY structured event, `sensor_trace` included; `atom_from_ep` per EP; `atom_from_observation` per derived constraint — hash-dedup against valid atoms in scope: existing → reuse id, `event='NONE'`), embeddings; **Txn B**: event atoms → `create_atom`; `insert_ep` + EP atoms; derived constraint observations + atoms; edges from `batch.edges` via `store.insert_edge` (`created_by='ingest'`, `proposer='closure'`); `upsert_transition(Transition.from_ep(ep, scope=..., permission_scope=...))` per EP. Zero LLM calls on the structured path (asserted by `hms_scripted.llm.calls == []`).

---

## §13 MVMI — State Index (group D: `hms_elite/mvmi/`, `hms_elite/utils/scoring.py`)

Package docstring: `"""State Index plane: filter AST, query planner, axis execution and score fusion — the hot path; never touches graph tables.\n状态索引平面：过滤 AST、查询规划、轴执行与分数融合——热路径；从不触碰图表。"""`

### §13.1 `mvmi/filters.py` — Filter DSL → `FilterExpr`

```python
class FilterExpr(HMSModel): ...                       # base; subclasses are frozen pydantic models
class Eq(FilterExpr): field: str; value: Any
class Ne(FilterExpr): ...; class Gt; class Gte; class Lt; class Lte   # same shape
class In(FilterExpr): field: str; values: list[Any]
class NotIn(FilterExpr): field: str; values: list[Any]
class Contains(FilterExpr): field: str; value: str
class IContains(FilterExpr): field: str; value: str
class Exists(FilterExpr): field: str
class And(FilterExpr): items: list[FilterExpr]
class Or(FilterExpr): items: list[FilterExpr]
class Not(FilterExpr): item: FilterExpr

AXIS_COLUMNS = {"time": "time_anchor", "time_anchor": "time_anchor", "created_at": "created_at", "updated_at": "updated_at",
                "valid_from": "valid_from", "valid_to": "valid_to", "atom_type": "atom_type", "scale_level": "scale_level", "scale": "scale_level",
                "role": "role", "sentiment": "sentiment", "confidence": "confidence", "memory_type": "memory_type", "attributed_to": "attributed_to",
                "actor_id": "actor_id", "visibility": "visibility", "subject": "subject", "predicate": "predicate", "obs_type": "obs_type",
                "memory_id": "memory_id", "atom_id": "atom_id", "hash": "hash", "source_kind": "source_kind", "extracted_by": "extracted_by"}
TIME_COLUMNS = {"time_anchor", "created_at", "updated_at", "valid_from", "valid_to"}
SET_AXES = {"entity": "entity", "entities": "entity", "topic": "topic", "topics": "topic", "anchor_id": "anchor"}
SCOPE_KEYS = ("user_id", "agent_id", "run_id")
OPERATORS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains")
```

- `parse_filters(filters: dict | None, *, require_scope: bool = True, context: str = "search", force_meta: bool = False) -> tuple[Scope, FilterExpr | None]` — Mem0 grammar: scope keys → `Scope` (trimmed; `require_scope` and none → `VALIDATION_001`); **top-level `AND` hoisting**: Mem0's `_process_metadata_filters` flattens `AND` before the scope is read (MEM0_MAP §2.2 step 5), so a **scalar** scope key written inside the top-level `AND` chain — nested `AND`s included, hoisted recursively — is lifted to the top level and pins the `Scope`, exactly as the flat form does (first occurrence wins; a key already present at top level blocks the hoist and the `AND` occurrence stays a column leaf; a clause emptied by the hoist is dropped, and an emptied `AND` list removes the key); `OR`/`NOT` branches are never hoisted (a scope key there is not a necessary constraint → still `VALIDATION_001`), and list / operator-dict / `'*'` scope forms are not pins anywhere and stay column leaves; the caller's filter dict is never mutated (copy-on-write); other keys: scalar → `Eq`, list → `In`, `'*'` → `Exists`, dict → one node per operator (`AND`-ed); **visibility lattice gate**: a scalar `visibility: a` (audience) lowers to `In('visibility', [v for v in VISIBILITY_RANK if rank(v) >= rank(a)])` (so `team` admits `team|event|public`; dict/list forms are literal); `force_meta=True` forces every leaf into the `meta.<key>` namespace (used for `AxisQuery.meta_filters`); `AND`/`OR`/`NOT` keys → `And/Or/Not` over their list items (each item parsed recursively; `NOT` with a list of one item); field name resolution: `AXIS_COLUMNS` → column; `SET_AXES` → set axis (`entity` values normalized via `normalize_entity`; FILE paths keyed by basename); anything else → `meta.<key>`; explicit `meta.<key>` always means user_meta; unknown operator → `ValidationError(VALIDATION_012, ValueError)` `"Unsupported operator 'X'. Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"` | 解析过滤器. Effect: none. Notes: time-typed literals pass `to_utc_iso` (`lte`/`lt` with a date-only literal → end of that day); list values for `In` on time columns normalized the same way.
- `lower_to_sql(expr: FilterExpr | None, *, capabilities: dict, scope: Scope) -> tuple[str, list[Any], set[str]]` — `(where_sql, params, joins)`; set axes lower to `EXISTS (SELECT 1 FROM atom_entity ae JOIN entities e ON e.entity_id = ae.entity_id WHERE ae.atom_id = memory_atoms.atom_id AND e.normalized_text IN (?, ...) AND COALESCE(e.user_id,'') = COALESCE(?, '') AND COALESCE(e.agent_id,'') = COALESCE(?, ''))` (entity), `EXISTS (SELECT 1 FROM atom_topic ...)` (topic), `EXISTS (SELECT 1 FROM atom_anchor ...)` (anchor); `meta.<key>` → `json_extract(user_meta_json, '$.<key>')` (key escaped; requires `capabilities['json1']`); `Contains` → `LIKE '%v%' ESCAPE '\'`, `IContains` → `LOWER(col) LIKE LOWER(...)`; `Exists` → `col IS NOT NULL` (meta: `json_type(...) IS NOT NULL`) | 降为 SQL.
- `evaluate(expr: FilterExpr | None, atom: MemoryAtom, *, entity_keys: set[str], anchor_ids: set[str]) -> bool` — in-Python twin with identical semantics (used for the `json1` fallback and by tests for conformance) | Python 求值.
- `plan_filters(expr, *, capabilities) -> tuple[FilterExpr | None, FilterExpr | None]` — `(sql_part, python_part)`: when `json1` is available `python_part` is None; otherwise every `meta.*` leaf (and any `Or`/`Not` containing one) moves to `python_part`, evaluated after the SQL gate | 拆分计划.
- `has_advanced_operators(filters: dict) -> bool` — Mem0 port (unused by search; kept for parity tests) | 是否高级运算符.

### §13.2 `mvmi/planner.py` — `plan_query`

```python
@dataclass
class QueryPlan:
    route: str                          # Route value
    time_windows: list[tuple[str, str, str]]   # (kind, gte, lte) from extract_dates(query, base_date=date_of(as_of))
    entity_hints: list[EntityCandidate]
    identifier_tokens: list[str]
    content_tokens: list[str]           # content_tokens(query) with identifiers kept whole
    hard_time_gate: tuple[str, str] | None     # first window
    hard_entity_gate: list[str] | None         # normalized keys, only when route == 'entity'
    hard_type_gate: tuple[str, list[str]] | None   # ('obs_type', ['preference_event', 'habit']) only when route == 'preference'
    scale_default: list[str] | None            # recall → ['state','event','L1_micro','L2_meso','L3_macro','entity','relation'] (None = no constraint); preference → ['state']; temporal → ['event','L1_micro','state','L3_macro']; entity → ['state','event','L1_micro','L2_meso','L3_macro','entity','relation']; work_state/detective → None
    def as_dict(self) -> dict
```

- `plan_query(query: str, *, as_of: str, mode: str | None = None) -> QueryPlan` — route precedence: `WHY_RE → detective`; `HANDOFF_RE → work_state`; `time_windows or TEMPORAL_QUERY_RE → temporal`; `ENTITY_QUERY_RE and entity_hints → entity`; `PREF_QUERY_RE → preference` (HMS §4.1: preference recall is an axis selection — `hard_type_gate = ('obs_type', ['preference_event', 'habit'])`, `scale_default = ['state']`); else `recall`; `mode == 'deep'` forces `work_state` when the query has no stronger cue | 查询规划. Effect: none. Notes: month tokens in the query are dates only under the §9.2 `month_en` rule ("may change" is not May); `"what coffee does the user like"` → route `preference`.

### §13.3 `mvmi/query.py`

- `execute_axis_query(store, index, embedder, aq: AxisQuery, *, scoring: ScoringConfig, capabilities: dict, threshold: float = 0.0, explain: bool = False, retrieval: RetrievalConfig | None = None, retrieval_stats: dict | None = None) -> AxisResult` — normative algorithm (HMS §3.5 semantics executed, not just declared); `retrieval` bounds the recall channels (None → `RetrievalConfig()`), `retrieval_stats` (when given) receives `Pool.stats()` of the winning ladder level:
  1. `fscope, expr = parse_filters(aq.filters, require_scope=False)`; `scope = aq.scope or fscope` (explicit `aq.scope` wins; both empty → `VALIDATION_001`); `_, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context='meta', force_meta=True)` (every leaf in the `meta.<key>` namespace); `sql_expr, py_expr = plan_filters(And([e for e in (expr, meta_expr) if e]))`.
  2. Plan gates from `aq.plan` (set by `search`/`compile`, `{}` for `get_all`): `hard_time_gate` → `time_anchor BETWEEN ? AND ?`; `hard_entity_gate` → entity `In`; `hard_type_gate` → `obs_type IN (?...)`; all appended to the SQL WHERE.
  3. Candidates of a ranked query (`aq.text`) come from **bounded multi-channel recall into a pool, then full scoring inside the pool** (RETRIEVAL_DIAGNOSIS §3–§5); there is no newest-first truncation any more (`ScoringConfig.max_candidates` is a deprecated alias that truncates nothing). Let G be the gated set of the level (scope ∧ validity ∧ `where_sql` ∧ plan gates ∧ scale ∧ `atom_types`) and `ks = retrieval.k_for(aq.limit)` = `{'semantic': max(4·top_k, 60), 'keyword' | 'entity' | 'recency': max(2·top_k, 30), 'pool': pool_max_factor · semantic}`:
     - `recall_pool(store, index, embedder, plan, scope, *, as_of, include_expired, cfg, top_k, text, query_vector, where_sql, params, joins, scale_constraint, atom_types, gate_count=None, index_filter=None, gate_entity_keys=None, index_hint=None) -> Pool` — **G enumeration**: a hard entity gate without joins enumerates G ONCE through `store.entity_candidates(scope, gate keys, limit=None, <gate>)` (atom_entity-driven, `time_anchor DESC, atom_id ASC`), which is the probe, the semantic channel's candidate list and — when the plan's entity hints equal the gate keys — the entity channel itself; every other level probes with one UNORDERED `store.axis_ids(..., order=None, limit=pool + 1)` (index order, stops early, no sort). **Whole gate**: when `cfg.hard_gate_pool` and `|G| <= ks['pool']` the whole G enters the pool (`channel_sizes == {'gate': |G|}`, `recalled_by == ['gate']`, `Pool.whole`; hydration restores axis order). Otherwise the round-robin union (order `semantic, keyword, entity, recency`, duplicates dropped, cut at `ks['pool']`) of the index-backed channels, each deterministically ordered and each searching INSIDE G (the level gate — `where_sql, params, joins, scale_constraint, atom_types, index_hint` — travels into every store channel): **semantic** = `index.search(vector, filter=IndexFilter(scope, as_of, include_expired, scale_in, obs_type_in, time_from, time_to, atom_type_in), top_k=K_sem)` under native pushdown (`index.supports_filter` — the sqlite and milvus providers; only when the level has no filter WHERE / join / entity gate), else `index.search(vector, candidate_ids=G ids, top_k=K_sem)` (ids only, no hydration); the channel's provider scores are kept in `Pool.semantic`; **keyword** = `store.keyword_candidates(scope, content_tokens + identifier_tokens of the plan, limit=K_kw, <gate>)` (two FTS5 phases — all tokens, then any token — or the bounded `LIKE` fallback; the ×2 identifier weight stays in the in-pool `keyword_score`); **entity** = the first `K_ent` of the enumerated G when the hint keys equal the gate keys, else `store.entity_candidates(scope, plan entity keys, limit=K_ent, <gate>)`; **recency** = `store.axis_ids(..., order=AXIS_ORDER, limit=K_rec, <gate>)` (the first `K_rec` ids of G in axis order), run only when the route is `temporal` or the plan carries a hard time gate, when `cfg.hard_gate_pool` is False (so a gated set of `<= K_rec` rows still reaches the pool whole), or as a fallback when the other channels together recalled fewer than `K_rec` ids. `index_hint` is `schema.TIME_INDEX` under a hard time gate, `OBS_TYPE_INDEX` under an `obs_type` type gate, `SCALE_INDEX` when only the scale whitelist gates the level (none with joins or an entity gate): without statistics SQLite drives every one of these through the scope index and walks the whole scope, which made the three empty preference-ladder levels cost a scope scan each. `Pool(ids, recalled_by, channel_sizes, gate_count, whole, semantic)`; `Pool.stats()` = `{'pool_size', 'channel_sizes', 'channels_used', 'gate_count', 'whole'}` (`gate_count` = |G| when G was enumerated, -1 under native pushdown — a filtered count is a full scan on milvus-lite, and the probe only proved `|G| > pool_max`).
     - Hydration: the pool is fetched in batches of 500 with `store.axis_select(scope, where_sql=<level gate>, atom_ids=<batch>, ...)` (a VALUES-CTE CROSS JOIN: one primary-key lookup per id; the gate re-applied here is the pool ∩ G invariant) — then `py_expr` with `evaluate` when present. **Python-side filter**: when `py_expr` is not None (a metadata filter on a SQLite without JSON1) the channels are NOT run — bounding them on the SQL-only gate could recall K ids of which none pass the predicate — the level is enumerated with `_select(limit=None)` (safety_limit, warned), filtered in Python and scored whole (`channel_sizes == {'gate': n}`, `recalled_by == ['gate']`, `whole=True`), with one `logger.warning` ("a Python-side filter (no JSON1) disables bounded recall"); `tests/test_retrieval.py::test_python_filter_scores_the_whole_gate` asserts bit-equality with the exhaustive path under `json1=False`. The in-pool semantic score reuses `Pool.semantic` for the channel's hits and calls `semantic_scores(index, qv, <pool ids the channel did not return>)` for the rest (bit-identical: both are the provider's `dot` scores; on Milvus this removes one RPC per query).
     - Safety: `retrieval.safety_limit` (default 50000) never truncates a pool; crossing it logs one `logger.warning` per condition — `"gated set of N atoms exceeds …"` when |G| is known (`gate_count >= 0`) and `"pool of N atoms exceeds …"` for the pool (the only one reachable under native pushdown, where `gate_count == -1`). Axis-only reads that cannot be bounded in SQL (`latest` over values, `summary`'s count, any `py_expr` post-filter) read at most `safety_limit` rows and warn when they hit it.
     - **Equivalence theorem** (`tests/test_retrieval.py::test_equivalence_theorem`): when `|G| <= min(K_kw, K_ent, K_rec)` or (`hard_gate_pool` and `|G| <= pool`), the pool equals G, so `(atom_id, score)` of every hit and the first 14 `score_details` keys are bit-identical to the exhaustive path (every gated candidate scored): the in-pool semantic score is the provider's `dot` score of the atom (from `Pool.semantic` or `semantic_scores(index, qv, rest)` = `index.search(candidate_ids=rest, top_k=len(rest))`), the other four signals depend only on the hydrated atom and the plan, `fuse` / threshold / sort keys are unchanged. Every existing suite corpus has `|G| < 60`, so the golden scenarios are unchanged. For `|G| > K` the pool is a subset of G (an intended, bounded pruning loss; N=500 overlap is printed by `test_pool_recall_overlap_at_n500`). **Miss condition**: an atom whose fused score is the global maximum is absent from the pool exactly when it ranks below `K_sem` in the semantic channel AND below `K_kw` / `K_ent` / `K_rec` in every other channel that ran (bm25 ties are broken by `atom_id`, not by the fused score); callers detect channel mode from `score_details` / `retrieval`: `recalled_by == ['gate']` (`'gate' in channel_sizes`, `retrieval['whole']`) means the whole gated set was scored, anything else means bounded recall.
     - Axis-only queries (`aq.text is None`: `list` / `latest` / `summary` / `count`) never build a pool: `store.axis_select(..., limit=aq.limit)` in axis order (`limit=1` for a plain `latest`; unbounded-with-safety-limit for `latest` over values and under `py_expr`), `summary` counts G with one `COUNT(*)`.
  4. Fallback ladder, strictest level first, the first non-empty level wins and is recorded as `primary_axis` (every level runs the same bounded recall of step 3; a level whose pool hydrates to nothing after `py_expr` is skipped like an empty one): (plan gates, route scale default) → `'time' | 'entity' | 'type' | 'semantic'` (`'type'` for the preference route); (plan gates, no scale default) → `'scale_fallback'`; (no plan gates, scale default) → `'fallback'`; (no plan gates, no scale default) → `'fallback'`. Only the route's scale *default* is relaxable: an explicit `aq.scale_constraint` is never dropped; `count` walks the same ladder with one `COUNT(*)` per level.
  5. Aggregation and projection (`aggregation` defaults to `'list'` when None):
     - `count` → `count = SELECT COUNT(*)` over the gated WHERE (one SQL statement; no rows loaded), `hits = []`, `projection = None`.
     - axis-only (`aq.text is None`): `ordered` = candidates in axis order (`time_anchor DESC, created_at DESC, atom_id ASC`), `score = 0.0`; `list` → `ordered[:limit]`; `latest` → when `project_axis == 'value'` the newest per `(subject, predicate)` group (group order = first appearance), else the single newest atom; `summary` → `ordered[:limit]` and `summary = f"{len(candidates)} atoms; latest: {ordered[0].text}"` (`"0 atoms"` when empty); `topk` → `VALIDATION_013` (raised by the model validator: needs `text`).
     - ranked (`aq.text`): `signals = compute_signals(...)` (§13.4) over the **pool** of step 3, `fuse` → `final`; drop `final < threshold`; sort `(final DESC, time_anchor DESC, atom_id ASC)`; `topk`/`list` → cut to `limit`; `latest` → the newest (per `(subject, predicate)` for `value`) among the ranked hits; `summary` → as above over the ranked list; with `explain` every hit's `score_details` carries `recalled_by` (its channels), `pool_size` and `channel_sizes`.
     - `projection` from the resulting hits: `'atom'` → None; `'value'` → `[atom.content.get('objects') or [atom.text] for each hit]` (a list of value lists in hit order; HMS "latest preference value" = `result.projection[0]`); `'time'` → `[(time_anchor or created_at, atom_id)]` sorted ascending (timeline); `'entity'` → `{entity_key: [atom_id, ...]}` (keys sorted, atom ids in hit order); `'topic'` → `{topic: [atom_id, ...]}` likewise.
  6. Return `AxisResult(hits=[AtomHit(atom, score, score_details (when explain), matched_axes)], project_axis, aggregation, projection, count, summary, primary_axis, retrieval)` — `retrieval` = `Pool.stats()` of the winning level for ranked queries (`{}` for axis-only / count; `compile()` copies it into `ContextIR.retrieval`); the `retrieval_stats` out-parameter receives the same dict.
  Effect: read-only; touches only `memory_atoms`, `atom_entity`, `entities`, `atom_topic`, `atom_anchor`, `atom_vectors`, `atom_fts` (asserted by `store.trace`).
- `recall_pool(...) -> Pool` and `Pool` — the step-3 recall, public so `compile()` and tests reuse it; `CHANNELS = ('semantic', 'keyword', 'entity', 'recency')`, `GATE_CHANNEL = 'gate'` | 召回池.
- `count_axis_query(store, aq) -> int` — `execute_axis_query(..., aq with aggregation='count').count` | 计数.
- `search(store, index, embedder, *, query: str, filters: dict, top_k: int, threshold: float, explain: bool, as_of: str, include_expired: bool, scoring: ScoringConfig, capabilities: dict, retrieval: RetrievalConfig | None = None) -> list[dict]` — `plan = plan_query(query, as_of=as_of)`; `aq = AxisQuery(filters, text=query, limit=top_k, as_of, include_expired, aggregation='topk', plan=plan.as_dict())`; `execute_axis_query(...).hits` → `atom_to_row(atom, score, score_details)` | 搜索.
- `get_all(store, *, filters, top_k, as_of, include_expired) -> list[dict]` — `AxisQuery(text=None, limit=top_k)` (no plan → no scale constraint); rows without `score`; order `time_anchor DESC, created_at DESC, atom_id ASC` | 全量读取.
- `atom_to_row(atom: MemoryAtom, *, score: float | None = None, score_details: dict | None = None, anchor_ids: list[str] | None = None) -> dict` — `return atom.to_public_dict(score=score, score_details=score_details, anchor_ids=anchor_ids)` (§4.1; the implementation lives in group A) | 行投影.

### §13.4 `utils/scoring.py` — signals and fusion

```python
@dataclass(frozen=True)
class Signals:
    semantic: float; keyword: float; entity: float; recency: float; support: float
    active: tuple[str, ...]                # subset of ('semantic','keyword','entity','recency','support')

def semantic_scores(index, query_vector, candidate_ids) -> dict[str, float]            # max(0, cos); missing → 0.0
def keyword_score(query_tokens: list[str], atom_tokens: set[str], *, identifier_weight: float) -> float
    # Σ_{t ∈ Q ∩ A} w(t) / Σ_{t ∈ Q} w(t), w(t) = identifier_weight if is_identifier(t) else 1.0; Q empty → 0.0; atom_tokens = content_tokens(text) ∪ topic_tags ∪ entity keys ∪ entity canonical tokens
def entity_score(query_keys: set[str], atom_keys: set[str], *, degrees: dict[str, int]) -> float
    # |Q ∩ A| / |Q| × 1/(1 + 0.001·max(0, deg−3)²), deg = max degree among matched entities (hub damping starts above 3 links, so S4b's `retry.ts` (deg 3) scores exactly 1.0); Q empty → 0.0
def recency_score(t: str | None, *, as_of: str, half_life_days: float) -> float       # exp(−ln2 · age_days / half_life); t None → 0.0
def support_score(n_anchors: int) -> float                                            # tanh(n / 3)
def compute_signals(*, atom, semantic, plan, as_of, scoring, degrees, n_anchors) -> Signals
    # active = {semantic, recency, support} ∪ {keyword iff plan.content_tokens} ∪ {entity iff plan.entity_hints}
def fuse(signals: Signals, scoring: ScoringConfig) -> tuple[float, float, float]      # (final, raw, max_possible): raw = Σ_{k∈active} w_k·s_k; max_possible = Σ_{k∈active} w_k; final = raw / max_possible
def score_details(signals, final, raw, max_possible, *, threshold, plan, primary_axis, scope=None, as_of=None, recalled_by=None, pool_size=None, channel_sizes=None) -> dict
```

`score_details` keys (exact order): `semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold, recency_score, support_score, keyword_backend ('overlap'), axis_gates ({'time': [gte, lte] | None, 'entity': [...] | None, 'obs_type': [...] | None, 'scope': {...}, 'validity': as_of}), primary_axis, route, signals (list of active), recalled_by (list of recall channels of §13.3 step 3: `['gate']` or a subset of `CHANNELS` in channel order), pool_size (int), channel_sizes ({channel: recalled ids})`. The first 14 keys and their values are exactly those of the exhaustive path; the three recall keys are appended. `bm25_score` == keyword score (name kept for Mem0 consumers). Threshold applies to `final_score` (documented divergence from Mem0).

---

## §14 Support Graph (group E: `hms_elite/graph/`)

Package docstring: `"""Support Graph plane: witnessed typed edges, boundary detection, coarsen/project/refine, explanation.\n支持图平面：带 witness 的类型化边、边界检测、Coarsen/Project/Refine、解释。"""` Purpose: inheritance path | 继承路径. Input: node id sets, edge specs | 节点集合、边规格. Output: edges, InterfaceEPs, subgraphs | 边、InterfaceEP、子图. Effect: writes graph tables only through `StateStore` | 仅经 StateStore 写图表.

### §14.1 `graph/support_graph.py`

```python
class SupportGraph:
    def __init__(self, store: StateStore, config: GraphConfig) -> None
    def insert_edge(self, *, src_id: str, dst_id: str, edge_type: str, witness_anchor_ids: list[str], created_by: str, proposer: str = "rule",
                    confidence: float = 0.7, label: str = "", scale: str = "L1_micro", is_boundary: bool = False, witness_edge_id: str | None = None) -> str
        # `if not witness_anchor_ids: raise WitnessMissingError(details={'object': 'edge', 'id': None, 'missing': []})` BEFORE building the model (pydantic would otherwise raise first);
        # then builds GraphEdge with scope = src node scope, created_at = store.now(); delegates to store.insert_edge (which re-checks and pre-checks anchors, §3/§8)
    def neighbors(self, node_ids: Iterable[str], *, hops: int = 1, edge_types: Iterable[str] | None = None, valid_only: bool = True,
                  direction: Literal["both", "in", "out"] = "both", max_nodes: int = 60) -> tuple[list[GraphNode], list[GraphEdge], dict[str, list[str]]]
        # BFS; returns (nodes incl. seeds, edges, paths: node_id → [edge_id, ...] from the nearest seed); deterministic: frontier sorted by node_id, edges by (created_at, edge_id)
    def internal_edges(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[GraphEdge]          # both endpoints in S
    def boundary(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[BoundaryEdge]             # exactly one endpoint in S; direction 'out' when src ∈ S
    def cohesion(self, node_ids: Iterable[str]) -> float
        # E_in, E_b as above; 0.0 when |E_in| == 0; else 0.5·min(1, |E_in|/max(1, |S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|)
    def components(self, scope: Scope, *, kinds: Iterable[str] = ("ep",), edge_types: Iterable[str] = ("causal", "use", "temporal"),
                   gap_hours: float = 48.0, min_size: int = 2, max_size: int = 12, scale: str = "L1_micro") -> list[list[str]]
        # connected components over valid edges of `edge_types` at `scale` among valid nodes of the given kinds whose graph_nodes.scale == scale
        # (L2 coarsening: kinds=('ep',), scale='L1_micro'; L3 coarsening: kinds=('interface_ep',), scale='L2_meso'); a component is split where consecutive (by time_anchor) members are > gap_hours apart;
        # members must share ≥1 topic (atom_topic) with at least one other member; sizes outside [min_size, max_size] are dropped (max: keep the newest max_size); each list sorted by time_anchor
    def owner_interface_ep(self, node_id: str, *, scale: str) -> InterfaceEP | None                             # valid IEP of that scale whose support contains node_id
    def stats(self, scope: Scope | None) -> dict                                                                  # None → whole DB
        # {'nodes': {kind: n for EVERY NodeKind (entity included)}, 'edges': {type: n for EVERY EdgeType}, 'valid_edges', 'invalid_edges', 'boundary_edges', 'interface_eps', 'unresolved_conflicts',
        #  'witness_coverage': edges with ≥1 witness / edges (1.0), 'projection_coverage': boundary edges with ≥1 projection row / boundary edges (1.0 when none),
        #  'traceable_high_edges': edges with scale ∈ {L2_meso, L3_macro} having ≥1 edge_projections row / such edges (1.0 when none), 'transitions': n}
        # node and edge counts are over ALL rows matching scope regardless of `valid`; `valid_edges`/`invalid_edges` are the only validity split; `interface_eps` counts valid IEPs
```

Docstring lines: `insert_edge` — "Insert one typed edge with at least one witness anchor | 插入一条至少带一个 witness 的类型化边"; In: endpoints, type, witnesses | 端点、类型、witness; Out: edge_id | 边 id; Effect: writes graph_edges/edge_witnesses (+edge_projections by trigger) | 写边表.

### §14.2 `graph/operators.py`

- `coarsen(store, index, embedder, graph: SupportGraph, node_ids: list[str], *, target_scale: str = "L2_meso", policy: str = "rule", llm: LLMBase | None, at: str, scope: Scope, min_cohesion: float) -> InterfaceEP | None` — Coarsen: low subgraph → InterfaceEP | 低层子图→高层接口. In: ≥2 valid node ids (all `ep` kind for `L2_meso`, all `interface_ep` kind for `L3_macro` — their slots are read from `interface_eps.slots_json`), same scope | 节点集合. Out: persisted InterfaceEP (or None when `cohesion < min_cohesion`) | InterfaceEP. Effect: writes one episode atom (`source_kind=interface_ep`, `scale_level=target_scale`, `anchor_ids=evidence_anchor_ids`, `text=slots.gloss()`, `source_ep_ids = [ref_id of every ep-kind support node]` (for L3: the union of the member IEPs' `source_ep_ids`), `topic_tags = ordered_union(*[atom.topic_tags for n in S])`, `entity_ids = ordered_union(*[atom.entity_ids for n in S])` passed as `extra_mentions`), `interface_eps`, `interface_ep_support` | 写原子与接口表. Notes: `S` sorted by time_anchor; `E_in = internal_edges(S)`, `E_b = boundary(S)`; `cohesion = graph.cohesion(S)` (stored unrounded); slots via `hms_elite.llms.rule.coarsen_rule(...)` (§11.5; policy `rule`) or the `coarsen` prompt (policy `llm`; any slot without valid `slot_support` refs, or refs ∉ S, or with an unusable value — `r` ∉ `Reward.values()`, a non-string text slot — is replaced by the rule value with a warning; `kappa.visibility` absent or ∉ `Visibility.values()` → the rule visibility with the LLM constraints kept); `evidence_anchor_ids = sorted(∪ atom_anchor_ids(n) for n in S ∪ witnesses(E_in))` (asserted equal to the union — evidence preservation); `slots.kappa.evidence = evidence_anchor_ids`; **invariant/residual completion** (the rule/LLM output carries only `support_refs`): for every invariant and residual `support_node_ids = [S[ref] for ref in support_refs]`, `anchor_ids = sorted(∪ store.atom_anchor_ids(n) for n in support_node_ids)`, `Invariant.confidence = round(mean(store.get_atom(n).confidence for n in support_node_ids), 4)`; residuals additionally: `low_cohesion` when `cohesion < min_cohesion + 0.1`; `support_hash = sha256(','.join(sorted(S)))`; an existing valid IEP with the same `support_hash` in scope → returned unchanged (idempotent); an existing valid IEP whose support is a strict subset of `S` → invalidated and a `revision` edge old_iep_node → new_iep_node (`scale=target_scale`, `is_boundary=False`, witnesses = anchors of the new members, `created_by='consolidation'`) is inserted.
- `project(graph, iep: InterfaceEP, *, at: str) -> list[str]` — Project: boundary edges → high edges | 边界边→高层边. Out: high edge ids | 高层边 id. Effect: writes `graph_edges(is_boundary=1, witness_edge_id=low)`; the trigger writes `edge_projections`; extra low edges merged into an existing high edge get `store.add_projection(high, low)` | 写投影. Notes: for each boundary edge `e`: `outer' = graph.owner_interface_ep(outer, scale=iep.scale).atom_id or outer`; oriented like `e` (`iep.atom_id → outer'` for `out`, reverse for `in`); `scale=iep.scale`, `witness_anchor_ids = witnesses(e)`, `created_by='project'`, `proposer='project-v1'`, `confidence = e.confidence`; internal edges are absorbed (never projected). Post-condition asserted: every high edge has ≥1 `edge_projections` row (else `StoreError(STORE_003)`).
- `refine(store, graph, interface_ep_id: str, *, hops: int = 0, edge_types=None) -> dict` — Refine on demand: high object → low support subgraph | 按需细化. Out: `{'interface_ep': InterfaceEP dict, 'nodes': [GraphNode dicts], 'edges': [GraphEdge dicts incl. witness_anchor_ids], 'boundary_edges': [BoundaryEdge dicts], 'anchors': [Anchor dicts], 'projections': {high_edge_id: [low_edge_id]}}` | 子图. Effect: read-only. Notes: `hops > 0` extends beyond the boundary via `graph.neighbors`; callers: `explain()` and `StateCompiler` (resolved mode `deep` ONLY — never `fast`, never `axis_graph`) (tested by `store.trace` in `fast`/`axis_graph` mode and by import/caller inspection).
- `promote_skills(store, index, embedder, graph, *, scope: Scope, at: str, anchor_id: str) -> list[str]` — invariants of kind `constraint` with `len(support_node_ids) >= 2` in valid IEPs → skill atoms | 技能提升. Out: created skill atom ids | 技能原子 id. Effect: `create_atom(MemoryAtom(atom_type=state, scale_level=L3_macro, source_kind=skill, source_id=iep.interface_ep_id, text=f"Skill: {inv.text}", subject='project', predicate='skill', anchor_ids=inv.anchor_ids, topic_tags=['skill'] + topic_tags(inv.text), confidence=inv.confidence, extracted_by='rule@skill-v1'))` (hash-dedup within scope: existing valid skill → reuse) + `distill` edge `iep.atom_id → skill_atom_id` (`scale='L3_macro'`, `is_boundary=False`, witnesses `inv.anchor_ids`, `created_by='consolidation'`, confidence 0.9) — **traceable to the low level** (HMS §3.7): `low = [valid use edges support_node → constraint_atom whose dst atom text == inv.text, ordered by edge_id]`; `witness_edge_id = low[0].edge_id` when `low` is non-empty (the trigger writes the first `edge_projections` row) and `store.add_projection(distill_edge, e.edge_id)` for every further `e ∈ low[1:]`; with no such low edge the distill edge is witnessed by anchors only | 写技能原子与 distill 边.
- `round_trip_check(graph, iep) -> bool` — `set(b.edge_id for b in refine(...)['boundary_edges']) == set(boundary(support_node_ids))` (test helper) | 往返校验.

### §14.3 `graph/explain.py` — `why()` core

- `resolve_target(store, target_id: str) -> dict | None` — order: `memory_id` (→ head atom, else latest version) → `atom_id` → `ep_id` (via `eps.atom_id`) → `obs_id` (via `observations.atom_id`) → `interface_ep_id` → `edge_id` (→ `{'kind': 'edge', 'node_ids': [src, dst]}`); returns `{'kind': 'atom'|'interface_ep'|'edge', 'id', 'node_id'|'node_ids', 'text', 'resolved_from': 'id'}` or None | 解析目标.
- `explain(store, graph, target: dict, *, hops: int = 2, edge_types: Iterable[str] | None = None, include_invalid: bool = False, max_nodes: int = 60) -> dict` — In: resolved target | 已解析目标. Out: §17.3 `why` shape | why 返回. Effect: read-only (calls `refine(hops=0)` for interface_ep nodes encountered). Algorithm: `edge_types = edge_types or WHY_EDGE_TYPES`; level-by-level up to `hops`: for each frontier node take **incoming** valid edges of `edge_types` (sources join the next frontier) and **outgoing** `use` edges (targets join); every `interface_ep` node visited is refined and its support nodes/edges/boundary added (`refined_from` records it); `paths[node] = [edge_ids from target]`; `unresolved_conflicts` = visited `conflict` edges with `resolved_at IS NULL`; anchors = union of edge witnesses ∪ atom anchors of visited nodes (undisputed); `chain = render_chain(...)`; `explanation = "\n".join(chain)`.
- `render_chain(store, target_node_id: str, edges: list[GraphEdge], *, hops: int) -> list[str]` — line 0 = target atom text; then at most `hops` steps: at node `n`, pick the highest-`confidence` incoming edge in `edges` (ties: `created_at`, `edge_id`) → line `f"<-{type}[{refs}]- {src_text}"`, continue from `src`; if none, pick the highest-confidence outgoing `use` edge → `f"-use[{refs}]-> {dst_text}"`, continue from `dst`; stop when neither exists; `refs = ', '.join(anchor.source_ref or 'sha256:' + anchor.hash[:12] for the edge's witness anchors in witness order)` | 链渲染.

---

## §15 Compiler (group F: `hms_elite/compiler/`)

Package docstring: `"""State compiler: axis candidates first, graph on demand, deterministic slots, budget, Work State Card and PC-IR.\n状态编译器：先轴候选，按需展开图，确定性槽位、预算、Work State Card 与 PC-IR。"""`

### §15.1 `compiler/slots.py`

- `SLOT_RULES` (ordered, total): (1) `atom.archived_reason is not None or (atom.valid_to is not None and atom.valid_to <= as_of) → audit` (explain.reason `'archived:' + (archived_reason or 'expired')`; `'revised'` when reached through a `revision` edge, with `superseded_by_text`) — a live atom with a FUTURE `valid_to` (expiration_date) is NOT audit; (2) `source_kind == 'skill' → skills`; (3) `obs_type == 'constraint' → constraints`; (4) `obs_type in {preference_event, habit} → preferences`; (5) `atom_type == 'episode' → episodes` (explain `{'r','a','y','u','ep_id'|'interface_ep_id'}`); (6) else → `facts` (`entity` atoms never reach the compiler — §15.2 step 2 excludes them).
- `assign_slot(atom: MemoryAtom, *, as_of: str, reason: str | None = None) -> Slot` | 槽位分配.
- `item_from_atom(atom, *, slot, score, anchor_ids, source, hops, explain) -> ContextItem` — `item_id = atom_id`, `tokens = estimate_tokens(text)` | 构造项.
- `kappa_items(episodes: list[ContextItem], selected_constraint_texts: set[str]) -> list[ContextItem]` — `f"{atom_id}#kappa:{i}"` for `slots.kappa.constraints[i]` whose normalized text (§11.3 rule 1 normalization) ∉ `selected_constraint_texts` (= normalized texts of every selected `constraints` item ∪ every `audit` item, so an archived/revised constraint is never re-synthesized) AND not already emitted for an earlier episode in this call (episodes are processed in slot order; the first episode wins — the same constraint text never yields two κ items); `anchor_ids` = episode anchors; `source='system'` | κ 合成项.
- `invariant_items(iep: InterfaceEP, iep_atom_id, selected_constraint_texts) -> tuple[list[ContextItem], list[ContextItem]]` — `(constraints from kind=constraint invariants as f"{iep_atom_id}#inv:{i}", audit items for failed_path/stable_fact/residuals with reason 'invariant:<kind>'|'residual:<kind>')` | 不变量项.
- `evidence_items(anchors: list[Anchor]) -> list[ContextItem]` — `item_id=f"anc:{anchor_id}"`, slot audit, `text=anchor.evidence_line()`, `anchor_ids=[anchor_id]`, `explain={'kind': 'evidence', 'source_ref', 'hash', 'timestamp', 'revoked_at', 'permission_scope'}`; ordered `(timestamp, source_ref, anchor_id)` | 证据项.
- `derive_next_action(*, episodes, constraints, residuals: list[str], conflicts: list[tuple[str, str]], goal: str) -> str` — `neg` = newest negative episode, `pos` = newest positive; `kept = [c.text for c in constraints if not c.text.startswith("Do not re-apply")][:3]`; `suffix = f" while keeping constraints: {'; '.join(kept)}" if kept else ""`; (1) `neg and (pos is None or neg.time_anchor > pos.time_anchor)` → `f"Do not re-apply '{neg.explain['a']}'; investigate '{neg.explain['y']}'{suffix}"`; (2) `pos` → `f"Continue from '{pos.explain['a']}' (last outcome: {pos.explain['y']}){suffix}"`; (3) residuals → `f"Resolve: {residuals[0]}"`; (4) conflicts → `f"Resolve conflict between '{x}' and '{y}'"`; (5) `f"Continue: {goal}"` | 下一步规则.

### §15.2 `compiler/compile.py`

```python
class StateCompiler:
    def __init__(self, store, index, embedder, graph: SupportGraph, config: HMSConfig) -> None
    def compile(self, query: str, *, scope: Scope, filters: dict, budget_tokens: int | None = None, mode: str | None = None,
                as_of: str | None = None, require_witness: bool | None = None, audience: str | None = None) -> ContextIR
```

Normative algorithm (read-only; row counts identical before/after):
1. `as_of = as_of or store.now()`; `mode = mode or config.compiler.default_mode` (`CompileMode(mode)` else `VALIDATION_013`); `plan = plan_query(query, as_of=as_of, mode=mode)`; `route = plan.route`; `budget = budget_tokens or config.compiler.budget_tokens`; `audience` gates everything below (`rank(x) >= rank(audience)`; no gate when None).
2. Axis candidates: `hits = execute_axis_query(AxisQuery(filters, scope, text=query, limit=config.compiler.candidate_k, as_of, aggregation='topk', plan={**plan.as_dict(), 'atom_types': [t for t in AtomType.values() if t != 'entity']}), threshold=0.0).hits` (entity atoms are index-only navigation objects and never enter a card; the `visibility` lattice gate of §13.1 already applies when `audience` came from `filters`).
3. Work-state augmentation when `route ∈ {work_state, detective}` or `mode ∈ {deep, axis_graph}` or (`mode == 'auto'` and `len(store.list_eps(scope)) >= 1`): union all valid `episode` atoms, all valid `obs_type='constraint'` atoms and all valid `source_kind='skill'` atoms in scope (score 0.0 unless already a hit; audience-gated).
4. Graph expansion (never when `mode == 'fast'`; in `auto` iff step 3 augmented; always in `deep` and `axis_graph`): seeds = candidate node ids; `nodes, edges, paths = graph.neighbors(seeds, hops=config.compiler.graph_hops, edge_types=WHY_EDGE_TYPES, valid_only=False)`; valid neighbour atoms whose visibility passes the audience gate join candidates with `score = 0.5 × max seed score`, `source='graph'`, `hops` (gated-out neighbours → `retrieval['dropped_governance'] += 1`); a `revision` edge `old → selected` puts `old` (archived) into `audit` (`reason='revised'`, `superseded_by_text = new.text`); `conflict` edges with `resolved_at IS NULL` and both endpoints selected → `unresolved_conflicts`; **only in `deep`** (and `auto` resolved to deep): each visited `interface_ep` node → `refine(hops=0)` → `invariant_items` (invariants whose `anchor_ids` all fail the anchor permission gate are dropped, `dropped_governance += 1`); in `axis_graph` visited IEP nodes join as ordinary episode candidates and `refine` is never called; `retrieval['graph_hops'] = hops`; `retrieval['ablation'] = ABLATION_OF_MODE[resolved mode]` (`auto` → `'axis'` when no expansion happened, else `'axis_graph_support'`).
5. Dedup candidates by `atom_id` (max score); `event` atoms whose primary anchor ∈ anchors of any selected episode → dropped (`retrieval['dropped_covered'] += 1`).
6. Slot assignment (§15.1, `assign_slot(atom, as_of=as_of)`) → six lists; `selected_constraint_texts` = normalized texts of the constraints list ∪ the audit list; `kappa_items(episodes, selected_constraint_texts)` appended to constraints (deduplicated across episodes, §15.1); `anchor_ids` per item = `store.atom_anchor_ids(atom_id, include_disputed=True)` (so an atom whose every anchor is disputed still yields a well-typed item) with `explain['disputed_anchor_ids'] = [ids disputed for this atom]`; evidence anchors whose `permission_scope` fails the audience gate are dropped from the item's `anchor_ids` (`dropped_governance += 1`; an item left with no anchor is dropped entirely).
7. `goal = query`; `next_action = derive_next_action(...)` computed on the pre-budget candidate set (so a dropped episode still informs it).
8. Budget (`fit_budget`): `cap[slot] = floor(share × budget)`; pack each of facts/episodes/preferences/constraints/skills by `(score DESC, time_anchor DESC, item_id ASC)` until `cap`; constraints keep ≥ `min_constraints` items even beyond their cap; unused capacity (`budget − Σ used`) spills to facts, then episodes (continuing their sorted lists); audit: evidence items for every selected item's anchors (ordered as §15.1) are ALWAYS kept (witness completeness — never capped), then other audit items pack within the audit cap; `used_tokens = Σ tokens` of everything kept (evidence included; it may exceed `budget_tokens` only through evidence, recorded in `retrieval['over_budget_evidence']`); `retrieval['dropped']` = number of candidates not selected. After packing each non-audit slot is re-sorted `(time_anchor ASC nulls last, item_id ASC)` for rendering stability.
9. `retrieval = {'signals': plan-active signals, 'route', 'plan': plan.as_dict(), 'mode': resolved mode, 'ablation', 'candidates': n, 'graph_hops', 'dropped', 'dropped_covered', 'dropped_governance', 'over_budget_evidence', 'primary_axis': result.primary_axis (the ladder level of §13.3 step 4), 'pool_size', 'channel_sizes', 'channels_used' (the `Pool.stats()` of the candidate stage, §13.3 step 3; `0 / {} / []` for a text-less compile), 'rerank': None, 'anchors': [Anchor dicts (incl. 'revoked_at', 'permission_scope') for every anchor referenced], 'audience': audience}`; return `ContextIR(mode=resolved mode, ...)` (validator enforces disjoint item ids and witnessed items).

### §15.3 `compiler/render.py` — Work State Card (deterministic)

`render_card(ir: ContextIR) -> str` — exact format (`\n`-joined, no trailing newline; `→` is U+2192; `- (none)` when a section is empty):
```
# Work State Card
Scope: user_id=<u> agent_id=<a> run_id=<r> | As of: <as_of> | Mode: <mode>      ← only set scope keys, in this order, space-separated
## Current Goal
<goal or query>
## Valid Facts
- <fact.text>                                     ← facts in slot order, then episodes with r ∈ {positive, neutral, unknown} as "- {a} → {y or '?'} [{r}]"
## Invalidated Assumptions
- <audit item with reason 'revised'>.text -> superseded by: <superseded_by_text>     ← other archived reasons: "- <text> (archived: <reason>)"
## Failed Paths
- <a> → <y>                                        ← episodes with r == negative, in slot order
## Constraints
- <constraint.text>
## Preferences                                     ← section emitted only when non-empty
- <preference.text>
## Skills                                          ← section emitted only when non-empty
- <skill.text>
## Next Action
<next_action or 'none pending'>
## Evidence
- <source_ref> sha256:<hash12> @<timestamp>        ← audit evidence items in slot order
```
`ContextIR.to_text()` calls this through a lazy import (`from hms_elite.compiler.render import render_card` inside the method) so group A has no module-level dependency on F.

### §15.4 `compiler/pc_ir.py`

- `lower(ir: ContextIR, *, require_witness: bool = True) -> PCIR` — In: ContextIR (its `retrieval['anchors']` carries the anchor rows) | ContextIR. Out: PCIR | PCIR. Effect: none. `payload = {'goal', 'facts': [entry], 'episodes': [entry + r,a,y,u], 'preferences', 'constraints', 'skills', 'invalidated': [entry + superseded_by_text], 'next_action', 'as_of', 'scope'}` with `entry = {'item_id', 'text', 'anchor_ids', 'atom_id', 'memory_id'}`; `witness = {'items': {item_id: anchor_ids}, 'anchors': [anchor dicts], '_all': sorted union}`; `human_gloss = render_card(ir)`; obligations: `evidence_coverage` violated when any non-audit item's **undisputed** anchors (`item.anchor_ids` minus `item.explain.get('disputed_anchor_ids', [])`) are empty or none of them is present in `retrieval['anchors']` (`details={'missing_witness': [item_ids]}`); `no_unresolved_conflict` violated when `ir.unresolved_conflicts`; `governance` skipped when `retrieval['audience']` is None, violated when any item has `rank(item.visibility) < rank(audience)` (cannot happen for compiler output — selection already gates — but holds for hand-built ContextIRs), when a κ constraint of a selected episode is absent from `constraints ∪ audit` (by normalized text), or when an evidence anchor cited by a non-audit item has `rank(permission_scope) < rank(audience)`; `not_revoked` violated when any anchor cited by a non-audit item's undisputed `anchor_ids` has `revoked_at` (`details={'revoked': [anchor_ids]}`). `require_witness` and any violation → `ProofObligationError(PCIR_001, details={'violated': [...], 'missing_witness': [...], 'unresolved_conflicts': [...], 'revoked': [...], 'governance': [...]}, suggestion="call why(<edge_id>) and feedback(..., 'state_wrong'|'state_outdated') to resolve, or lower with require_witness=False")` | PC-IR 降低.

---

## §16 Ops (group G: `hms_elite/ops/`)

Package docstring: `"""Operations plane: feedback, checkpoint/rollback, handoff, transition export and the synchronous StateScheduler.\n运维平面：反馈、检查点/回滚、交接、状态转移导出与同步调度器。"""` Every public write here starts with `ingest.events.system_event(...)` (its anchor witnesses every row created) and runs in ONE `store.transaction()`.

### §16.1 `ops/feedback.py`

```python
class FeedbackProcessor:
    def __init__(self, store, index, embedder, graph, config) -> None
    def process(self, *, target_id: str | None, feedback: str, reason: str | None, correction: str | dict | None, actor_id: str | None, scope_hint: Scope | None) -> dict
```

Prologue: `feedback = FeedbackType(feedback)` (unknown → `ValidationError(VALIDATION_006, details={'allowed': [...]})`); resolve target (`memory_id` → head atom; `atom_id`; `edge_id`; `interface_ep_id`; context item id `'<atom_id>#...'` → atom; `anchor_id` → `{'kind': 'anchor'}` (accepted ONLY by `evidence_wrong` with `correction={'revoke': True}`, which delegates to `revoke()` below); otherwise `NOT_FOUND_001`) — **except `state_missing`**, for which an unresolvable or `None` `target_id` is legal: the scope is then `Scope` parsed from `correction['scope']` (dict) or from `target_id` in `Scope.key()` form (`user_id=alice&run_id=run-a`), else `scope_hint`, else `VALIDATION_001`; `target_kind='scope'`, `feedback_events.target_id` = the scope key; `state_missing` requires `correction` (`VALIDATION_007`; a dict must carry `text`); edge-only types on a non-edge target → `VALIDATION_007`; `scope = target scope` (or the scope above); `ev, fb_anchor = system_event(object_type='feedback', event_type='feedback', payload={target_id, feedback, reason, correction})`; `store.insert_feedback(...)`; atom targets get history `FEEDBACK` (`old_memory=text, new_memory=None, details={'feedback', 'reason'}`). Then the handler:

| FeedbackType | Target | Actions (all witnessed by `fb_anchor`; action dicts in this order) |
|---|---|---|
| `state_wrong` | atom | `archive` head (reason `feedback`); if `correction` (str) → successor via `revise_atom` (text=correction, coordinates copied, `time_anchor=now`, `anchor_ids=[fb_anchor]+old.anchor_ids`, revision edge `witness_anchor_ids` = that same list, confidence 0.95, `created_by='feedback'`, history `UPDATE`) → `create`, `revision_edge`; any unresolved `conflict` edge touching the target → `store.resolve_conflict(edge, at, by_anchor_id=fb_anchor, resolution_edge_id=<revision edge or None>)` → `resolve_conflict`; outgoing `use`/`distill` edges from the target → `invalidate_edge(reason='source_archived')` |
| `state_outdated` | atom | `archive` head (`valid_to = correction['valid_to']` when a dict with that key, else now; reason `feedback`); if a text correction → successor (`valid_from=now`, `time_anchor=now`, `anchor_ids=[fb_anchor]+old.anchor_ids`, entities/topics re-linked from the correction, revision edge witnesses = the same list) + `revision_edge`, history `UPDATE`; confidence unchanged. NOTE: the successor is visible only for `as_of >= the feedback instant` (validity is a WHERE clause, §8); a `compile(as_of=<earlier>)` still shows the archived predecessor as valid |
| `state_missing` | atom / episode / scope (no target) | `create` new lineage from `correction` (str, or dict `{'text', 'obs_type'?, 'scope'?}`: `atom_type=state`, `scale_level=state`, `obs_type=fact` (or `correction['obs_type']`), `subject='user'`, `predicate='states'`, `source_kind=feedback`, `source_id=feedback_id`, `anchor_ids=[fb_anchor]` (+ target anchors when a target exists), history `ADD`); if target is an episode → `insert_edge(use, new → target, witness fb_anchor)` |
| `privacy_violation` | atom | **the only handler that changes visibility**: `set_visibility` `private` on the head and on every atom sharing any of its anchors (`atoms_for_anchor`); `set_anchor_permission(private)` on those anchors; edges untouched |
| `next_action_wrong` | episode atom / context item | `create` constraint atom `(obs_type=constraint, subject='project', predicate='must_not', objects=[target.a], text=correction or f"Do not: {target.a}", topic 'next_action', anchors=[fb_anchor]+target anchors, source_kind=feedback)`; `set_confidence(target, −0.2)`; `insert_edge(distill, target → constraint, witness fb_anchor)` |
| `relationship_not_useful` | edge | `invalidate_edge(reason='feedback:not_useful')`; then the projection cascade of §16.1 Revoke step (3b) with the same reason: a high edge projected from it (`projections_from`) is invalidated only when EVERY one of its `projections_of` sources is now invalid (transitively L2 → L3) → one `invalidate_edge` action per fallen edge, low edge first |
| `connection_reason_wrong` | edge | `correction` may be `{'edge_type': ..., 'label': ...}` or a str label (`corrected_type = correction.get('edge_type') or old.edge_type`); **if `corrected_type == old.edge_type`** (label-only correction): no invalidation — `revalidate_edge(old)` (no-op when valid) + `add_edge_witness(old, fb_anchor)` + `set_edge_label(old, label)` and report `{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}`; **else** `invalidate_edge(old, reason='feedback:wrong_reason')` → `invalidate_edge`, then `insert_edge(same endpoints, corrected_type, label, witnesses = old witnesses + fb_anchor, created_by='feedback')` → `insert_edge` (a fresh row, since the type differs); if the old edge had projections, `project`-style high edges are re-derived for the new edge (`add_projection`) |
| `evidence_wrong` | atom | `correction` = list/CSV of anchor ids (default: all non-primary): `dispute_atom_anchor` each → `dispute_anchor`; if no undisputed anchor remains → `archive` (reason `feedback:evidence_wrong`) (an atom cannot stay valid without an undisputed witness); `set_confidence(−0.3)` |
| `render_not_like_me` | atom / item | `set_confidence(−0.1)`; `create` preference atom `(obs_type=preference_event, predicate='dislikes', objects=['render:'+target_id], text=reason or f"User dislikes how '{target.text[:40]}' is rendered.", topic 'render_style', anchors=[fb_anchor])` |
| `render_like_me` | atom / item | `set_confidence(+0.1)`; same with `likes` |
| `object_important` | atom | `set_confidence(+0.2)`; `add_atom_topic('important')` → `pin`; `bump_mentions(3)` |

Return `{'message': 'Feedback applied', 'feedback_id', 'actions': [{'action': 'archive'|'create'|'revision_edge'|'resolve_conflict'|'invalidate_edge'|'insert_edge'|'set_visibility'|'set_confidence'|'dispute_anchor'|'pin'|'revoke'|'noop', ...ids}]}`. Confidence math clamps to `[0, 1]`. `feedback_events.actions_json` stores the same list.

**Revoke (HMS §3.2 "撤销时能找到派生对象")** — `revoke(store, graph, *, anchor_id: str, reason: str | None, actor_id: str | None = None) -> dict` (same module; facade `HMS.revoke`): unknown anchor → `NOT_FOUND_001`; already revoked → `{'revoked': anchor_id, 'archived': [], 'edges_invalidated': [], 'interface_eps_invalidated': [], 'anchor_id': None, 'noop': True}`. Otherwise ONE transaction witnessed by `sys_anchor` from `system_event(object_type='system', event_type='updated', payload={'op': 'revoke', 'anchor_id': anchor_id, ...})`: (1) `store.revoke_anchor(anchor_id, at=now)`; (2) for every atom in `atoms_for_anchor(anchor_id)` (ascending): `dispute_atom_anchor(atom, anchor_id, at)`; if `atom_anchor_ids(atom)` (undisputed) is now empty → `archive_atom(reason='revoked', at)` + history `ARCHIVE(old=text, new=None, details={'revoked_anchor': anchor_id})` witnessed by `sys_anchor` → `archived`; (3) for every valid edge citing the anchor (ascending `edge_id`) whose witnesses are ALL revoked → `invalidate_edge(reason='witness_revoked', at)` → `edges_invalidated`; (3b) **projection cascade (existence rule, docs/design/WITNESS_SEMANTICS.md)**: worklist seeded with the edges of (3); for every high edge in `projections_from(fallen)` that is still valid and whose `projections_of` sources are now ALL invalid → `invalidate_edge(reason='projection_revoked', at)` → appended to `edges_invalidated` and pushed on the worklist (L2 → L3 chains); a high edge with any valid source is left valid even when its own witness set has lost members — its own witnesses are judged only by (3); (4) every valid IEP whose `evidence_anchor_ids` ⊂ revoked anchors → `invalidate_interface_ep` → `interface_eps_invalidated`; (5) every transition whose `w_t` ⊂ revoked anchors is left in place (evidence is never deleted; consumers filter by `anchors.revoked_at`). Return `{'revoked': anchor_id, 'archived': [...], 'edges_invalidated': [...], 'interface_eps_invalidated': [...], 'anchor_id': sys_anchor}`. `feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` calls this and reports one `{'action': 'revoke', ...}` entry.

### §16.2 `ops/checkpoint.py`

- `checkpoint(store, *, scope: Scope, label: str | None) -> str` — one transaction: `system_event(object_type='checkpoint', event_type='checkpoint', payload={'label'})`; `atom_state = store.scope_atom_state(scope)`; `edge_state = {e.edge_id: e.valid for e in store.list_edges(scope, valid_only=False)}`; `iep_state = {i.interface_ep_id: i.valid for i in store.list_interface_eps(scope, valid_only=False)}`; `ep_state = store.scope_ep_state(scope)`; `history_seq`; `snapshot_hash`; `insert_checkpoint` → id | 检查点. Effect: writes source_events, anchors, checkpoints.
- `rollback(store, graph, *, checkpoint_id: str, reason: str | None) -> dict` — unknown id → `ValidationError(NOT_FOUND_001, ValueError)`; one transaction witnessed by `rb_anchor` (`object_type='rollback'`); processing order is fixed: (1) all atom archives, (2) all atom restores, (2b) EP validity (`ep_state`: every EP in scope whose `eps.valid_to` differs from the recorded value → `archive_ep(at=recorded)` / `restore_ep`; EPs absent from `ep_state` → `archive_ep(at=now)`; this runs AFTER the atom cascade so the snapshot wins), (3) edges, (4) interface_eps, (5) rollback revision edges + transition; within each step ids ascending. For every atom now in scope: not in `atom_state` → `archive_atom(reason=f"rollback:{cp}")`, history `ROLLBACK(old=text, new=None, details={'checkpoint_id'})` → `archived`; in `atom_state` with recorded `valid_to is None` but now archived → `restore_atom`, history `RESTORE(old=None, new=text)` → `restored`; recorded archived but now valid (restored after the checkpoint) → `archive_atom(reason=f"rollback:{cp}", valid_to=recorded valid_to)` → `archived`; edges: not in `edge_state` → `invalidate_edge(reason=f"rollback:{cp}")` → `edges_invalidated`; recorded valid but now invalid → `revalidate_edge` → `edges_revalidated`; recorded invalid but now valid → invalidate; interface_eps likewise (`iep_state`); for each `(archived successor Y, restored X)` pair in the same lineage insert `revision` edge `Y → X` (`label=f"rollback:{cp}"`, witnesses `[rb_anchor]`, `created_by='ops'`, confidence 0.9) → `edges_created`; `upsert_transition(rollback transition: source_kind='rollback', source_id=<that edge id>, s_t={'text': Y.text, 'atom_id': Y, 'memory_id', 'valid_from', 'valid_to'}, a_t='rollback', s_t1={'text': X.text, 'atom_id': X, 'memory_id', 'valid_from', 'valid_to'}, r_t='neutral', kappa_t=kappa_t_of(constraints=<texts of valid constraint atoms in scope at time_t1, time_anchor ASC>, visibility=X.visibility, permission_scope=rb_anchor.permission_scope, provenance={'checkpoint_id', 'label', 'reason'}), w_t=[rb_anchor], time_t=Y.created_at, time_t1=now)` (§16.4 uniform κ_t); return `{'message': f"Rolled back to {label or checkpoint_id}", 'checkpoint_id', 'restored': [...], 'archived': [...], 'edges_invalidated': [...], 'edges_revalidated': [...], 'edges_created': [...], 'anchor_id': rb_anchor}` (EP validity changes of step (2b) mirror the atom lists and are not reported separately) | 回滚. Notes: atom content is never touched; vectors are never deleted; scope-local (edges to nodes outside the scope are invalidated with the snapshot and reported).

### §16.3 `ops/handoff.py`

- `build_state_package(store, graph, compiler, *, config: HMSConfig, scope: Scope, filters: dict, goal: str | None, budget_tokens: int, checkpoint_id: str | None, include_edges: bool = True) -> StatePackage` — `goal = goal or (store.last_ep(scope, run_id=scope.run_id).slots.u if that EP exists) or None`; `ir = compiler.compile(goal or 'handoff', scope=scope, filters=filters, budget_tokens=budget_tokens, mode='deep')`; `pcir = lower(ir, require_witness=config.compiler.require_witness)`; `eps` = EPs of selected episode items; `interface_eps` = valid IEPs whose support intersects selected nodes; `edges` = valid edges among selected nodes (when `include_edges`); `atoms` = every atom referenced by context items, `eps`, `interface_eps` support nodes and `edges` endpoints (`store.get_atoms`, sorted by atom_id); `anchors` = closure of every anchor id referenced (items, atoms, eps, ieps, edge witnesses); `source_events` = `store.get_source_event(a.source_event_id)` for every anchor | 交接包. Effect: read-only except the optional checkpoint the facade takes first.
- `import_state_package(store, index, embedder, graph, *, pkg: StatePackage, scope_override: Scope | None) -> dict` — HMS §9.4: the receiver imports the state package instead of reading history | 导入状态包. Effect: ONE transaction witnessed by `sys_anchor` from `system_event(object_type='system', event_type='created', payload={'op': 'import', 'package_id': pkg.package_id, ...})`; every imported object keeps its id unless it already exists (then it is skipped and reported) or `scope_override` is given (then scope columns are replaced on every object and ids are kept). Order: (1) `source_events` via `insert_source_event` (identity dedup → `id_map[old_event_id] = existing_id`), (2) `anchors` via `get_or_create_anchor` (`id_map` for remapped anchors; every later anchor reference goes through `id_map`), (3) `eps` via `insert_ep` (existing id → skip), (4) `atoms` via `create_atom(atom, history_event='ADD', history_anchor_id=sys_anchor, history_details={'imported_from': pkg.package_id})` (existing id → skip; `entity_ids`/`topic_tags` re-linked through `extra_mentions`), then `set_ep_atom` back-pointers, (5) `interface_eps` via `insert_interface_ep` (their episode atoms come from step 4), (6) `edges` via `store.insert_edge` (MERGE), (7) `Transition.from_ep`/`from_interface_ep` rows `upsert_transition`. Return `{'imported': {'source_events': n, 'anchors': n, 'eps': n, 'atoms': n, 'interface_eps': n, 'edges': n, 'transitions': n}, 'skipped': {...same keys}, 'id_map': {...}, 'anchor_id': sys_anchor}`.

### §16.4 `ops/transitions.py`

- `ep_transition = Transition.from_ep` (re-exported from `hms_elite.models`, group A — G never defines it): `source_kind='ep', source_id=ep_id, scale='L1_micro', s_t={'u', 'b', 'kappa': kappa.model_dump(), 'ep_id', 'valid_from', 'valid_to'}, a_t=a, s_t1={'y', 'r', 'ep_id'}, r_t=r, kappa_t=kappa_t_of(constraints=kappa.constraints, visibility=kappa.visibility, permission_scope=<primary anchor's permission_scope>, provenance={'evidence': kappa.evidence, 'ep_id': ep_id}), w_t=ep.anchor_ids, time_t=time_anchor, time_t1=time_anchor` | EP 转移. (Written at ingest, §12.6.)
- `interface_ep_transition = Transition.from_interface_ep` (group A): `source_kind='interface_ep', source_id=interface_ep_id, scale=iep.scale, s_t={'u', 'b', 'kappa': kappa.model_dump(), 'interface_ep_id', 'support_node_ids', 'valid_from': iep.created_at, 'valid_to': iep.valid_to}, a_t=slots.a, s_t1={'y', 'r', 'interface_ep_id', 'invariants': [inv.text], 'residuals': [res.text]}, r_t=slots.r, kappa_t=kappa_t_of(constraints=kappa.constraints, visibility=kappa.visibility, permission_scope=<primary anchor's permission_scope>, provenance={'evidence': evidence_anchor_ids, 'interface_ep_id', 'cohesion_score'}), w_t=evidence_anchor_ids, time_t=min(support atoms' time_anchor), time_t1=max(support atoms' time_anchor)` (HMS §4.4: EP **and** InterfaceEP are transition sources) | 接口 EP 转移.
- **Uniform `kappa_t`** for EVERY kind (`kappa_t_of`, §4): `{'constraints': [str], 'visibility': str, 'permission_scope': str, 'provenance': {kind-specific keys}}` — a training consumer reads κ_t with one shape.
- `export_transitions(store, *, scope: Scope, since: str | None, scale: str | None, kinds: Iterable[str] | None) -> list[Transition]` — one transaction: (a) every valid InterfaceEP in scope → `Transition.from_interface_ep(...)`; (b) `causal` edges (valid) between two ep/interface_ep nodes → `source_kind='causal', source_id=edge_id, s_t={'u': i.u, 'b': i.b, 'y': i.y, 'r': i.r, 'kappa': i.kappa, 'ep_id'|'interface_ep_id': i, 'valid_from', 'valid_to'}, a_t=j.a, s_t1={'u': j.u, 'y': j.y, 'r': j.r, 'ep_id': j, ...}, r_t=j.r, kappa_t=kappa_t_of(constraints=j.kappa.constraints, visibility=j.kappa.visibility, permission_scope=<edge primary witness anchor's permission_scope>, provenance={'edge_id', 'label', 'proposer', 'evidence': j.kappa.evidence}), w_t=ordered_union(edge_witnesses(edge), j.anchor_ids)` (**first-occurrence order, never sorted** — S6 row 3 is `[<anc_e3>, <anc_e2>, <anc_e4>]`), `time_t=i.time_anchor, time_t1=j.time_anchor, scale=edge.scale`; (c) `revision`/`conflict` edges of ANY validity between atom nodes whose label does not start with `rollback:` → `s_t={'text': old.text, 'atom_id', 'memory_id', 'valid_from', 'valid_to'}, a_t='revise'|'conflict', s_t1={'text': new.text, ...}, r_t='negative' (revision) | 'unknown' (conflict), kappa_t=kappa_t_of(constraints=<texts of valid constraint atoms in scope at time_t1 (axis_select as_of=time_t1, obs_type='constraint', time_anchor ASC)>, visibility=new.visibility, permission_scope=<edge primary witness anchor's permission_scope>, provenance={'label', 'created_by', 'proposer'}), w_t=edge witnesses (witness order), time_t=old.created_at, time_t1=edge.created_at`; rollback transitions are written by `rollback()` (§16.2, same κ_t shape); every row `upsert_transition` (idempotent); result `store.list_transitions(scope, since, scale, kinds)` ordered `(time_t1, KIND_ORDER[source_kind], source_id)` with `KIND_ORDER` from `hms_elite.enums` (§2) | 导出转移. Effect: writes `transitions` only.

### §16.5 `ops/consolidation.py` — `StateScheduler`

```python
class StateScheduler:
    def __init__(self, store, index, embedder, llm, graph, ingestor, compiler, config) -> None
    def run(self, *, scope: Scope | None, phases: list[str] | None) -> dict
```

Prologue: `phases = phases or config.scheduler.phases` (unknown → `VALIDATION_013`); scope None → every distinct `(user_id, agent_id, run_id)` scope in `memory_atoms` (each run separately); a PARTIAL scope (e.g. `filters={'user_id': U}`) → every distinct scope in `memory_atoms` that it `matches` (so `consolidate(filters={'user_id': U})` after S1–S3 runs four scopes: `run-coffee`, `run-a`, `run-b`, `run-c`); entity atoms (step 0) are keyed by the entity identity scope and are therefore created once even when several run scopes share an entity; per scope one `system_event(object_type='scheduler', event_type='scheduler', payload={'phases'})` → `sched_anchor`. Return `{'anchor_id', 'phases': {phase: report}}` (multi-scope: counts summed, list-valued keys concatenated, `anchor_id` = the last scope's anchor).

- **orient** (read-only): `{'active_topics': top-5 topics by count over valid atoms with time_anchor within orient_window_days, 'active_entities': top-5, 'open_failures': [ep_id of valid negative EPs (list_eps valid_only) with no later positive EP in the same run], 'pending_ep_nodes': [ep node ids with zero outgoing valid edges], 'coarsen_candidates': graph.components(...), 'last_compiled': store.last_compiled_state(scope) minus 'card' (or None)}`.
- **collect**: `{'re_embedded': n}` — `atoms_without_vectors(scope, embedder.identity)` re-embedded (`index.upsert`); `{'re_routed': n}` — `events_without_atoms(scope, object_types=EXTRACTABLE)` re-ingested through `ingestor.ingest_events(events, infer=True)` (internal object types are never re-routed).
- **consolidate** (each step one transaction; counts reported; report keys exactly `entity_atoms, temporal_edges, causal_edges, conflicts, interface_eps, projected_edges, interface_eps_l3, skills, transitions, compiled_states`): (0) `entity_atoms` (HMS §3.5 `entity` scale): for every row of `store.entities_in_scope(scope)`: `linked = store.entity_linked_atoms(entity_id)` (valid, non-entity atoms), `mentions = len(linked)`, `anchor_ids = sorted(∪ atom_anchor_ids(a) for a in linked)`; `existing` = the valid atom with `source_kind='entity', source_id=entity_id`; none → `create_atom(atom_from_entity(..., scope=Scope(user_id=e.user_id, agent_id=e.agent_id) if (e.user_id or e.agent_id) else Scope(run_id=e.run_id)), extra_mentions=[canonical_text], history_anchor_id=sched_anchor)`; existing with a different `content['mentions']` or `content['aliases']` → `revise_atom(old=existing, new=..., witness_anchor_ids=[sched_anchor], at=now, created_by='consolidation')`; otherwise skip; count = created + revised; (1) `temporal_edges`: for EVERY pair of consecutive valid EP nodes per run by `time_anchor` within `graph.temporal_window_hours` call `insert_edge(temporal, witness=[later primary anchor], proposer='timestamp', created_by='consolidation')` (MERGE: an existing edge gets `mentions += 1`); count = number of edge ids that did NOT exist before the call (`find_edge` is None before); (2) `causal_edges`: pairs = temporal neighbours ∪ pairs sharing an identifier entity ∪ `uses_ep_ids` pairs; `causal_proposer == 'rule'` → `hms_elite.llms.rule.propose_edges(nodes, pairs, witnesses)` in-process (§11.4, same node shape as the prompt); `'llm'` → `edge_proposal` task in batches of ≤20 pairs (witness refs resolved; a `witness_refs` that is not a list of strings, unknown refs or an empty list → the proposal is dropped with a warning, never raised; `confidence` unparsable or non-finite → 0.7); every accepted proposal → `insert_edge(created_by='consolidation', proposer=llm.identity or 'rule')` (MERGE bumps `mentions` on existing edges); count = newly created edge ids only; (3) `conflicts`: valid `state` atoms sharing `(subject, predicate)` with `predicate ∈ hms_elite.llms.rule.EXCLUSIVE` and disjoint objects and no `revision` edge between them → `insert_edge(conflict, older → newer, witness newer primary anchor)` + history `CONFLICT` on the older lineage; with `ingest.revision_mode == 'deferred'` this step also runs `decide_revision(new, old)` (§11.3) over same-subject pairs and applies REVISE via `revise_atom` (witness = newer anchor, `at = max(new.valid_from, old.valid_from)`); (4) `interface_eps`: for each `graph.components(scope)` set (EP nodes, L1) → `coarsen(..., target_scale='L2_meso')` (skipped/idempotent as §14.2) then `project` → `projected_edges`; (4b) `interface_eps_l3` (when `graph.coarsen_l3`): for each `graph.components(scope, kinds=('interface_ep',), edge_types=('causal','use','temporal'), scale='L2_meso', gap_hours=graph.coarsen_gap_hours * 4)` set → `coarsen(..., target_scale='L3_macro')` then `project` (its boundary edges are the L2 IEPs' projected edges; high edges at `L3_macro`, projections counted in `projected_edges`); (5) `skills`: `promote_skills(...)`; (6) `transitions`: `export_transitions(scope)` → count of new rows (InterfaceEP rows included); (7) `compiled_states` (when `scheduler.compile_state`; HMS §3.9 `compile_state`): `ir = compiler.compile(goal, scope=scope, filters=scope.as_filters(), mode='deep')` with `goal = last_ep.slots.u or 'work state'`; `store.insert_compiled_state(scope, as_of=ir.as_of, mode='deep', card=ir.to_text(), context_json=ir.model_dump_json(), used_tokens=ir.used_tokens, anchor_id=sched_anchor)`; `set_meta(f'work_state:{scope.key()}', canonical_json({'hash': sha256(card), 'as_of', 'used_tokens', 'compiled_id'}))`; count 1 per scope (`compile()` itself stays read-only; only the scheduler persists).
- **prune** (`decay_stale_state`; HMS §3.9 "衰减旧状态、归档低置信对象" — two operations): examined = valid atoms with `atom_type ∈ {state, event, message}` and `obs_type ∉ {constraint}` and `source_kind ∉ {skill, entity}` and topic `important` absent; per atom `recency = recency_score(time_anchor or created_at, as_of=now, half_life=scoring.half_life_days)`; `feedback_boost = clamp(confidence − 0.6, 0, 0.4)`; `use_recency = recency of the newest valid use/causal edge touching the node (0 if none)`; `staleness = (1 − recency)·(1 − feedback_boost)·(1 − use_recency)`; (a) **decay**: when `staleness > staleness_tau`: `new_conf = max(confidence_min / 2, round(confidence − scheduler.decay_rate · staleness, 4))`; if `new_conf != confidence` → `set_atom_confidence(new_conf)` + history `UPDATE(old=new=text, details={'decay': round(staleness, 4), 'confidence': [old, new]})` witnessed by `sched_anchor`, `decayed += 1`; (b) **archive**: after decay, when `confidence < confidence_min` AND no unresolved `conflict` edge touches the node → `archive_atom(reason='decayed')` + history `ARCHIVE` witnessed by `sched_anchor` (`skipped_conflict += 1` when only the conflict check fails); report `{'examined', 'decayed', 'archived': [atom_ids], 'live', 'expired', 'skipped_conflict', 'purged'}`; `purged` is always 0 unless `retention.enabled` **and** `retention.run_in_prune` are both true, in which case, after (b) and in its own transaction, §16.6 `purge(store, index, scope=scope, older_than_days=retention.older_than_days, dry_run=False, keep_witnessed=True)` runs and its `purged` is reported.
- Epilogue: `set_meta(f"last_consolidation:{scope.key()}", now)`.

### §16.6 `ops/retention.py` — the retention escape hatch (explicit, default-off, audited)

Module docstring: `"""Retention escape hatch: explicit, audited, default-off physical purge of long-archived atom versions (§16.6).\n保留策略逃生舱：显式、有审计、默认关闭的长期归档原子版本物理清理（§16.6）。"""` This is the ONLY path that ever physically removes rows and it is never taken by default: revision-instead-of-deletion stays the invariant of every other write, and the `BEFORE DELETE` triggers keep raising `STORE_003` for any DELETE outside `StateStore.purge_rows()`.

- `purge(store, index, *, scope: Scope, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict` — report or remove archived versions nothing live still needs | 清理逃生舱. In: scope with ≥ 1 id (`Scope.matches`, so `user_id` alone sweeps every run; empty → `VALIDATION_001`), a non-negative int (`VALIDATION_009` otherwise), the two switches | 范围、天数、两个开关. Out: `{'dry_run', 'scope': scope.key(), 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable': [{'atom_id','memory_id','version','text_sha256','archived_reason','valid_to','source_kind','source_id','anchor_ids','bytes'}], 'protected': {reason: [ids]}, 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}` | 报告. Effect: `dry_run=True` writes nothing; `dry_run=False` is ONE transaction: `system_event(object_type='system', event_type='deleted', payload={'op': 'purge', 'metadata': {'older_than_days', 'cutoff', 'keep_witnessed', 'atoms'}})` → one history row per version `PURGE(old=None, new=None, is_deleted=1, atom_id, details={'version','text_sha256','archived_reason','valid_to','source_kind','source_id','anchor_ids','cutoff','older_than_days'})` witnessed by the purge anchor → `store.purge_rows(ids, evidence=not keep_witnessed)`; after COMMIT `index.delete(atom_id)` per purged atom | 事件 + PURGE 历史 + 物理删除. Notes: `cutoff = now − older_than_days days`; `examined` = `store.purge_candidates(scope, valid_to_before=cutoff)` (archived_reason non-NULL, valid_to < cutoff). **Protections** (every candidate is kept when ANY applies; `protected` lists each id under every reason that fired, in the order `head_version, edge_endpoint, checkpoint_live, interface_ep_support, ep_referenced, source_shared`): (1) `head_version` — the highest version of a lineage with ≥ 2 versions (the terminal record of its revision chain; a single-version tombstone has no chain and may go); (2) `edge_endpoint` — src or dst of ANY `graph_edges` row, valid or invalid (edges carry witnesses and are never purged; this is a superset of "endpoint of a valid edge" and pins every revised version through its `revision` edge); (3) `checkpoint_live` — recorded with `archived_reason None` in any checkpoint snapshot (a rollback may restore it); (4) `interface_ep_support` — cited by `interface_eps.atom_id`, `primary_support_node_id` or an `interface_ep_support` node member; (5) `ep_referenced` — an `ep` atom whose EP row is still cited by another EP's `uses_ep_ids`, a surviving atom's `source_ep_ids` or a transition; (6) `source_shared` — an atom whose `(source_kind, source_id)` is also cited by an atom outside the purge set (`store.shared_source_atom_ids`): `purge_rows` deletes the atom's own `observations` / `eps` row, which an update / feedback successor (it copies `source_kind` + `source_id`) still needs — independent of the `revision` edge. The same sharing rule governs `archive_atom`: its source cascade is skipped while another non-archived atom cites the source, and `revise_atom` calls `store.reopen_source(new)` after creating a successor that shares its predecessor's source (`tests/test_ops.py::test_feedback_successor_keeps_shared_observation_live`). `bytes` = utf-8 text + canonical content JSON + 4·dims of its stored vector + 256. **Evidence**: `keep_witnessed=True` never touches anchors or `source_events`; `False` additionally deletes, among the anchors of the purged rows, those with `anchor_reference_count == 0` after the state rows are gone (history rows cite anchors, so anchors of atoms that went through `create_atom`/`delete` stay) together with their `source_events` row. `graph_edges`, `edge_witnesses`, `history`, `feedback_events`, `checkpoints`, `transitions`, `compiled_states` are never touched (`feedback_events.target_id` / `history.atom_id` may dangle by design — they are the audit trail).
- Trigger gate: `PURGE_GATE` (§7) on the seven `BEFORE DELETE` triggers; `purge_rows` writes `meta.purge_in_progress='1'` in its own transaction and deletes it before COMMIT (ROLLBACK drops it too); `StateStore.__init__` deletes a stale flag with a warning so a hand-edited file cannot leave the gate open. UPDATE (immutability) triggers are never gated.
- Facade (§17.3): `HMS.purge(*, filters, older_than_days, dry_run=True, keep_witnessed=True)` parses `filters` with `require_scope=True` and raises `ConfigError(CONFIG_001, details={'retention.enabled': False, 'dry_run': False})` for `dry_run=False` while `config.retention.enabled` is False; `HMS.vacuum()` = `store.vacuum()`. Scheduler hook: §16.5 prune.

---

## §17 Facade (group H: `hms_elite/memory/main.py`, `__init__.py`, `migrate.py`, `cli.py`)

Module docstring (`main.py`): `"""HMS: the Mem0-compatible public API over the dual-plane store.\nHMS：双平面存储之上、与 Mem0 兼容的公共 API。"""`

### §17.1 Construction

```python
class HMS(MemoryBase):
    def __init__(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None
        # config = HMSConfig.from_dict(config) if dict else (config or HMSConfig()); store = store or StateStore(config.store.path, busy_timeout_ms=config.store.busy_timeout_ms, journal_mode=config.store.journal_mode, checkpoint_on_close=config.store.checkpoint_on_close);
        # embedder = EmbedderFactory.create(config.embedder.provider, config.embedder.config); index = IndexFactory.create(config.vector_store.provider, config.vector_store.config, store=store, dims=embedder.dims);
        # llm = LlmFactory.create(config.llm.provider, config.llm.config); graph = SupportGraph(store, config.graph); ingestor = Ingestor(...); compiler = StateCompiler(...);
        # feedback_processor = FeedbackProcessor(...); scheduler = StateScheduler(..., compiler=compiler, config=config); reranker = RerankerFactory.create(config.reranker.provider, config.reranker.config, llm=self.llm unless config.reranker.config binds its own 'llm' envelope) when config.reranker is not None, else None (search(rerank=True) then builds one default OverlapReranker lazily, announced once at info level)
    # PUBLIC attributes (tests reach the injected collaborators through them): config: HMSConfig, store: StateStore, index: IndexBase, embedder: EmbedderBase, llm: LLMBase,
    # graph: SupportGraph, ingestor: Ingestor, compiler: StateCompiler, feedback_processor: FeedbackProcessor, scheduler: StateScheduler
    # (e.g. `hms_scripted.llm.calls == []`, `hms.store.trace.tables()`, `hms.store.stats()`)
    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "HMS"          # HMSConfig.from_dict; logs the full error; ConfigError propagates
    def __enter__(self) -> "HMS"; def __exit__(self, *exc) -> None      # close()
    @property
    def project(self) -> "_OSSProject"                                   # .update(custom_instructions=None, custom_categories=None, multilingual=None, decay=None) → ValueError (Mem0 parity)
    def chat(self, query: str) -> None                                   # raises NotImplementedError (Mem0 parity)
```

### §17.2 Mem0-compatible surface (signatures verbatim; `inspect.signature` is tested)

```python
def add(self, messages, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, metadata: dict | None = None,
        timestamp: Any = None, expiration_date: Any = None, infer: bool = True, memory_type: str | None = None, prompt: str | None = None) -> dict
def get(self, memory_id: str) -> dict | None
def get_all(self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs) -> dict
def search(self, query: str, *, top_k: int = 20, filters: dict | None = None, threshold: float = 0.1, rerank: bool = False, explain: bool = False,
           reference_date: Any = None, show_expired: bool = False, **kwargs) -> dict   # mvmi.search(..., scoring=config.scoring, retrieval=config.retrieval, ...): the configured RetrievalConfig reaches every search, not only compile()
def update(self, memory_id: str, text: str | None = None, metadata: dict | None = None, expiration_date: Any = _UNSET, data: str | None = None) -> dict
def delete(self, memory_id: str) -> dict
def delete_all(self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> dict
def history(self, memory_id: str) -> list[dict]
def reset(self) -> None
def close(self) -> None
```

| Method | Validation (error) | Semantics | Returns |
|---|---|---|---|
| `add` | `build_scope_and_metadata` (`VALIDATION_001`, ids trimmed); `memory_type ∉ {None, 'procedural_memory'}` → `VALIDATION_002`; `coerce_messages` (`VALIDATION_003`); `timestamp` → `to_utc_iso` (`VALIDATION_004`), accepted; `expiration_date` → `normalize_expiration_date` → `valid_to = next_day_midnight(date)` | `ingestor.ingest_messages(...)`; procedural branch iff `agent_id and memory_type == 'procedural_memory'`; `system` role messages are anchored, never extracted | `{'results': [row + 'event': 'ADD'|'UPDATE'|'NONE' (+ 'actor_id','role' when infer=False) (+ 'revises': <archived atom_id>) (+ 'conflicts_with': [memory_id])]}`; nothing extracted → `{'results': []}` (events still persisted) |
| `get` | — | `store.resolve_head(memory_id)`; None when unknown, deleted, or every version archived | row (§4.1) or `None` |
| `get_all` | `reject_top_level_entity_params(kwargs, 'get_all')` (`VALIDATION_008`, ValueError); `parse_filters` requires ≥1 scope key (`VALIDATION_001`); `top_k ≥ 0` | `mvmi.get_all(...)`; `show_expired` → `include_expired=True` (archived/deleted versions shown with `valid_to`) | `{'results': [row...]}` (no `score`), order `time_anchor DESC, created_at DESC, atom_id ASC` |
| `search` | as `get_all` + `validate_search_params` (`VALIDATION_009`) + `validate_and_trim_query` (`VALIDATION_010`); `reference_date` → `as_of` (`to_utc_iso`) | `mvmi.search(...)`; `rerank=True` → `self.reranker.rerank(query, rows, top_k=top_k)` after fusion (unconfigured → default `OverlapReranker`, one info log); rows come back in rerank order with `rerank_score` as the last key; `explain=True` appends `rerank_score` and `reranker` (identity) to `score_details`; any reranker exception keeps the retrieval order with a warning; `rerank=False` is untouched | `{'results': [row + 'score' (+ 'score_details' when explain) (+ 'rerank_score' when rerank)]}` |
| `update` | `data=` → `logger.warning` deprecated alias for `text`; none of text/metadata/expiration_date provided → `VALIDATION_011` (ValueError); unknown/deleted id → `NOT_FOUND_001` (`ValueError("Memory with id X not found")`) | one transaction: `system_event(object_type='system', event_type='updated', payload={'op': 'update', ...})`; text given → `revise_atom(old=head, new=copy with text, hash, re-linked entities/topics, user_meta merged (identity keys stripped, warn only if changed), anchor_ids=[sys_anchor]+old.anchor_ids, valid_to per expiration_date, created_by='ops')` + history `UPDATE`; metadata-only → `set_user_meta` on the head (no new version; Mem0 parity: metadata is not state) + history `UPDATE(old=new=text, details={'metadata_only': True})`; `expiration_date` explicit `None` → `set_valid_to(NULL)`; a date → `set_valid_to(next_day_midnight)` | `{'message': 'Memory updated successfully!'}` |
| `delete` | unknown/deleted → `NOT_FOUND_001` | `system_event(event_type='deleted')`; `archive_atom(head, reason='deleted', at=now)`; history `DELETE(old=text, new=None, is_deleted=1)`; edges kept | `{'message': 'Memory deleted successfully!'}` |
| `delete_all` | none of the three ids → `ValidationError(VALIDATION_001, ValueError)` with message `'At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method.'` | one transaction, one system event; every valid head in scope archived (`deleted`) with a `DELETE` history row | `{'message': 'Memories deleted successfully!'}` |
| `history` | — | `store.get_history(memory_id)` | bare list of `{'id','memory_id','old_memory','new_memory','event','created_at','updated_at','is_deleted': bool,'actor_id','role','anchor_id'}` in `seq` order; unknown id → `[]` (documented parity delta: `+anchor_id`, extra `event` values `ARCHIVE|RESTORE|FEEDBACK|CONFLICT|ROLLBACK`) |
| `reset` | — | `store.reset()`; `index.reset()` | `None` |
| `close` | — | removes `self.index.update_attrs` from `store.attr_listeners` (a shared store must not keep calling a closed provider), then `index.close()` (idempotent; milvus releases its client), then `store.close()` in a `finally` (both idempotent) | `None` |

### §17.3 HMS-native surface

```python
def add_events(self, events: list[SourceEvent | dict], *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, infer: bool = True) -> dict
    # → {'results': [rows as add()], 'events': [{'event_id','anchor_id','object_type','dedup'}], 'eps': [ep_id], 'edges': [edge_id], 'constraints': [atom_id]}
def compile(self, query: str, *, filters: dict, budget_tokens: int | None = None, mode: str | None = None, as_of: Any = None, require_witness: bool | None = None) -> ContextIR
    # filters validated like search (scope required); mode None → config.compiler.default_mode, else ∈ CompileMode (VALIDATION_013); audience = filters.get('visibility') when it is a scalar
    # (KEPT in the axis filters — §13.1 lowers it to the lattice gate — and passed as audience= so graph neighbours, invariants and evidence anchors are gated too); read-only
def axis_query(self, aq: AxisQuery | dict) -> AxisResult
    # HMS-native seven-axis query (HMS §3.5): dict → AxisQuery(**dict); scope from aq.scope or the scope keys in aq.filters (neither → VALIDATION_001); as_of defaults to store.now();
    # delegates to mvmi.query.execute_axis_query(...); e.g. axis_query({'filters': {'user_id': U, 'topic': 'coffee'}, 'project_axis': 'value', 'scale_constraint': ['state'], 'aggregation': 'latest'}).projection == [['black coffee']]
def why(self, target_id: str, *, hops: int = 2, filters: dict | None = None, edge_types: list[str] | None = None, include_invalid: bool = False) -> dict
    # target = resolve_target(store, target_id); if None and filters: ir = compile(target_id, filters=filters, mode='deep', budget_tokens=600); candidates = ir.constraints + ir.episodes + ir.facts;
    #   prefer items whose atom entity keys intersect identifier_tokens(target_id) (entity_key form), else the first; target = {'kind': 'atom', ..., 'resolved_from': 'query'}; nothing → NOT_FOUND_001;
    #   hops ∉ [0, 6] → VALIDATION_013; returns explain(...) → {'target': {'kind','id','node_id','text','resolved_from'}, 'nodes': [...], 'edges': [... incl. 'witness_anchor_ids'], 'anchors': [...],
    #   'interface_eps': [...], 'paths': {node_id: [edge_id, ...]}, 'unresolved_conflicts': [edge_id], 'refined_from': [iep_id], 'chain': [str], 'explanation': str}
def feedback(self, target_id: str | None, feedback: FeedbackType | str, *, reason: str | None = None, correction: str | dict | None = None, actor_id: str | None = None) -> dict
    # → {'message': 'Feedback applied', 'feedback_id', 'actions': [...]}; target_id may be None (or a Scope.key() string) ONLY for 'state_missing' (§16.1); an anchor_id target is accepted ONLY for evidence_wrong + correction={'revoke': True}
def revoke(self, anchor_id: str, *, reason: str | None = None, actor_id: str | None = None) -> dict
    # → ops.feedback.revoke(...) (§16.1): {'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'}
def handoff(self, *, filters: dict, goal: str | None = None, budget_tokens: int | None = None, checkpoint: bool = True, include_edges: bool = True) -> StatePackage
    # checkpoint=True → self.checkpoint(filters, label=f"handoff:{goal or ''}") first, id embedded; build_state_package(..., config=self.config, ...)
def import_package(self, pkg: StatePackage | str | dict, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> dict
    # str → StatePackage.from_json, dict → model_validate; scope_override = Scope(user_id, agent_id, run_id) when any is given; → ops.handoff.import_state_package(...) (§16.3)
def checkpoint(self, *, filters: dict, label: str | None = None) -> str
def rollback(self, checkpoint_id: str, *, reason: str | None = None) -> dict
def transitions(self, *, filters: dict, since: Any = None, scale: str | None = None, fmt: str = "tuples", kinds: list[str] | None = None) -> list | str
    # fmt ∈ {'tuples','records','jsonl'} (VALIDATION_013); 'tuples' → [t.to_tuple()], 'records' → [t.model_dump()], 'jsonl' → '\n'.join(json.dumps(record, ensure_ascii=False))
def consolidate(self, *, filters: dict | None = None, phases: list[str] | None = None) -> dict
def graph_stats(self, *, filters: dict | None = None) -> dict          # graph.stats(scope) (filters None → scope None → whole DB; §14.1 counts all nodes/edges regardless of validity)
def context(self, query: str, *, filters: dict, top_k: int | None = None, as_of: Any = None, budget_tokens: int | None = None) -> dict   # §6.7 hybrid read context: memoized session card + timestamped rows → {'text', 'session_card', 'session_key', 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids', 'rows', 'rows_text', 'rows_header', 'search_rows', 'scope'}; read-only
def purge(self, *, filters: dict, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict   # §16.6 escape hatch: parse_filters(require_scope=True); dry_run=False needs config.retention.enabled (else CONFIG_001); → ops.retention.purge report
def vacuum(self) -> dict                                               # store.vacuum(): VACUUM outside any transaction → {'path','before_bytes','after_bytes','freed_bytes'}
```

Every public write is ONE `store.transaction()` (add/add_events use Txn A + Txn B around an unlocked phase); every public read calls `store.now()` once for `as_of`.

### §17.4 `hms_elite/__init__.py`

`from hms_elite.memory.main import HMS`; `from hms_elite.configs.base import HMSConfig`; `__version__ = importlib.metadata.version("hms-elite")` (fallback `"0.0.0"`); `__all__ = ["HMS", "HMSConfig", "__version__"]`. No other side effects.

### §17.5 `hms_elite/migrate.py`

- `from_mem0(memory: Any, hms: HMS, *, filters: dict, page_size: int = 100) -> dict` — walks `memory.get_all(filters=filters, top_k=page_size)` (and `show_expired=True`), re-adds each row with `hms.add(row['memory'], infer=False, timestamp=row.get('created_at'), metadata=row.get('metadata') or None, user_id=..., agent_id=..., run_id=..., expiration_date=row.get('expiration_date'))`, returns `{'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}` | 从 Mem0 迁移. Effect: writes through `hms.add`.

### §17.6 `hms_elite/cli.py` (≤ 90 lines, `argparse`)

`hms add <text> --user U [--agent A] [--run R]`, `hms search <query> --user U [--top-k N]`, `hms compile <query> --user U [--run R] [--mode auto|fast|deep]`, `hms why <id-or-question> --user U [--run R]`, `hms transitions --user U [--run R] [--fmt jsonl]`, `hms consolidate --user U`, `hms stats --user U`; env `HMS_DIR`; prints JSON (`ensure_ascii=False`) or the card text; entry point `hms = "hms_elite.cli:main"`. An `HMSError` raised by the command is printed as `error.to_dict()` JSON (`ensure_ascii=False`) on stderr and the process exits 1 — a traceback is never the CLI's error surface. README §12 seeds `run-billing` with `hms add "Do not modify the subscription state machine." … --run run-billing` before `hms why … --run run-billing`, because question-mode `why` (§17.3) needs at least one constraint / episode / fact in scope.

### §17.7 `hms_elite/server/` — the HTTP service face (group H; optional extra `server`; `tests/test_server.py`)

- `create_app(config: HMSConfig | dict | None = None, *, hms: HMS | None = None) -> FastAPI` (lazy `fastapi` import → `DependencyError(DEPS_001, extra='server')`); `main(argv)` runs uvicorn (`hms-server`). An injected `hms` stays open for its owner; a facade built by `create_app` is closed by the lifespan.
- Route table (`_ROUTES`, operation label in parentheses): `POST /memories` (add), `GET /memories` (get_all), `GET /memories/{memory_id}` (get), `POST /search` (search), `PUT /memories/{memory_id}` (update), `GET /memories/{memory_id}/history` (history), `DELETE /memories/{memory_id}` (delete), `DELETE /memories` (delete_all), `POST /reset` (reset), `POST /events` (add_events), `POST /compile` (compile), `GET /why/{target}` (why), `POST /feedback` (feedback), `POST /checkpoint` (checkpoint), `POST /rollback` (rollback), `POST /handoff` (handoff), `GET /transitions` (transitions), `POST /consolidate` (consolidate), `GET /stats` (graph_stats), `GET /usage` (usage), `GET /metrics` (metrics); plus `GET /health` (liveness, version, provider names).
- **Auth**: `$HMS_SERVER_TOKEN` set and non-empty → every route of the table requires `Authorization: Bearer <token>` (`deps.require_bearer`; missing / wrong → `AuthError(AUTH_001)` → 401); unset or empty → every request is accepted. `GET /health` is registered outside the authenticated router and never checks the token.
- **Error envelope** (`error_response`): every mapped error is `HMSError.to_dict()` = `{error_type, error_code, message, details, suggestion, debug_info}`; `error_status`: `AuthError` → 401, `error_code` starting with `NOT_FOUND` → 404, `ValidationError` → 400, any other `HMSError` → 500; a bare `ValueError` / pydantic error raised inside the facade → `ValidationError(VALIDATION_000, details={'exception': <type>})` → 400; any other exception → `HMSError('internal error', INTERNAL_000)` → 500. **A 500 body is redacted**: `error_type` and `error_code` are kept, `message` is `'internal error'`, `details` / `debug_info` are `{}` and `suggestion` is `None` (a `StoreError` carries the SQLite path and SQL fragments in its message / details; they are logged server-side at ERROR with `exc_info` and the request id, never sent). FastAPI's own 422 for a malformed body is untouched. The catch-all lives in `RequestTelemetry` (a pure ASGI middleware), so the INTERNAL_000 response carries `X-Request-ID` like every other and the trace records the error; a response that already started streaming is not replaced.
- **Telemetry**: `X-Request-ID` is echoed (or generated) on every response; each request is one `telemetry.trace(operation, request_id)` observed into `RequestMetrics` (`GET /usage`, `GET /metrics`).

### §17.8 `hms_elite/memory/async_main.py` — `AsyncHMS`; `hms_elite/telemetry.py` — metering (groups H / A; `tests/test_async.py`, `tests/test_telemetry.py`)

```python
class AsyncHMS:
    def __init__(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None
    @classmethod def from_config(cls, config_dict: dict) -> AsyncHMS
    @classmethod async def create(cls, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> AsyncHMS
    async def aclose(self) -> None                       # idempotent; `async with AsyncHMS(...) as hms:` calls it
    sync: HMS; config: HMSConfig; store: StateStore; project: _OSSProject     # synchronous properties
    async def add / get / get_all / search / update / delete / delete_all / history / reset / close /
              add_events / compile / axis_query / why / feedback / revoke / handoff / import_package /
              checkpoint / rollback / transitions / consolidate / graph_stats / context / purge / vacuum  # same names and signatures as HMS (inspect.signature is compared pairwise); each is `await asyncio.to_thread(<sync method>, ...)`
    async def chat(self, query: str) -> None             # raises NotImplementedError like HMS.chat
    def usage(self) -> telemetry.Usage                   # synchronous extra: telemetry.usage_of(self.sync)
    def trace(self, operation: str, request_id: str | None = None) -> AbstractContextManager[telemetry.RequestTrace]   # synchronous extra
```

- Rules: the event loop is never blocked (`test_async_does_not_block_the_event_loop`); concurrent `add` + `search` serialize on the store lock and return what the synchronous facade returns (`test_async_concurrent_add_and_search`, `test_async_results_equal_sync_results`); every argument is forwarded verbatim (`test_async_forwards_every_argument_verbatim`); offline defaults, error codes and envelopes pass through unchanged; `close()` / `aclose()` are idempotent.
- `telemetry`: `Usage(llm_calls, input_tokens, output_tokens, total_tokens, estimated, embedding_calls, embedded_texts, duration_ms)` with `merge` / `to_dict`; `UsageMeter` (`add_llm`, `add_embedding`, `snapshot`, `reset`; lock-protected — `test_concurrent_calls_lose_no_counts`); `MeteredProvider` (the `usage` meter every LLM / embedder base class carries: `openai` / `anthropic` record the SDK's exact usage, `rule` / `scripted` estimate through `memory.utils` and set `estimated`); `trace(operation, request_id=None)` (a contextvar-local `RequestTrace` collecting `StageRecord`s from `stage(...)` / `record_stage(...)`; `current_trace()`); `usage_of(hms) -> Usage` (llm ∪ embedder meters). The server's `GET /usage` returns `usage_of(hms).to_dict()` and `GET /metrics` renders `RequestMetrics` (§17.7).

---

## §18 Tests (group T: `tests/`) — pytest, fully offline, deterministic

### §18.1 `tests/conftest.py` (the ONLY conftest)

```python
class FrozenClock:                      # start="2026-09-02T10:00:00+00:00"; .now() -> fixed-width ISO (does NOT auto-advance); .tick(seconds=0, days=0)
class SequentialIds:                    # __call__(prefix) -> f"{prefix}_{n:06d}" with one counter per prefix
@pytest.fixture(autouse=True) def no_home_writes(monkeypatch, tmp_path): monkeypatch.setenv("HMS_DIR", str(tmp_path)); monkeypatch.delenv("OPENAI_API_KEY", raising=False); monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
@pytest.fixture(autouse=True) def no_network(monkeypatch): ...            # socket.socket → raises unless HMS_TEST_ALLOW_NETWORK=1
@pytest.fixture def clock(): return FrozenClock()
@pytest.fixture def ids(): return SequentialIds()
@pytest.fixture def store(clock, ids): return StateStore(":memory:", now=clock.now, id_factory=ids)
@pytest.fixture(params=["sqlite", "memory"]) def index_provider(request): return request.param
@pytest.fixture def config(index_provider): return HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"})
@pytest.fixture def hms(config, store): return HMS(config, store=store)                                   # rule + hash + parametrized index
@pytest.fixture def hms_scripted(store): return HMS(HMSConfig(llm={"provider": "scripted", "config": {"responses": []}}, store={"path": ":memory:"}), store=store)
@pytest.fixture def rule_llm(): return RuleLLM(); @pytest.fixture def scripted_llm(): factory(responses) -> ScriptedLLM
@pytest.fixture def hash_embedder(): return HashEmbedder({"embedding_dims": 64})
@pytest.fixture def events(): loader(name) -> dict (tests/fixtures/events/<name>.json)
@pytest.fixture def golden(request): return Golden(request)             # two helpers, chosen by the caller:
#   golden.text(name, text)   — assert_golden_text: BYTE-FOR-BYTE, no masking (Work State Cards, why_chain.txt — they contain fixture timestamps and no ids)
#   golden.masked(name, obj)  — assert_golden_masked: JSON/JSONL after memory.utils.mask_ids AND masking ONLY of instants >= FrozenClock.start ("2026-09-02…" → "<ts>"); fixture instants (2026-03…2026-06) stay literal
#   (transitions_s6.jsonl, observation/EP JSON, revision_cases.json); --update-golden rewrites the file for either helper
@pytest.fixture def scope4(): return {"user_id": "alice", "agent_id": "coder", "run_id": "run-billing"}
```

Rules (from Mem0 `tests/AGENTS.md`): mock the provider SDK, never the code under test; every bug fix ships a regression test that fails without it; `monkeypatch.setenv` only; no timing assertions; no network.

### §18.2 Fixtures

`tests/fixtures/events/`:
- `chat_preference.json` — S1 messages (§19.1)
- `chat_conflict.json` — `[{"role":"user","content":"I live in Paris.","timestamp":"2026-05-01T09:00:00Z"}, {"role":"user","content":"I live in Berlin.","timestamp":"2026-04-01T09:00:00Z"}]` (second added after the first) → CONFLICT
- `chat_multisession.json` — S2/S3 messages (§19.2)
- `chat_action_closure.json` — `[{"role":"assistant","content":"I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."}]`
- `chat_habit_temporal.json` — `[{"role":"user","content":"Yesterday I visited the Berlin office.","timestamp":"2026-05-10T09:00:00Z"}, {"role":"user","content":"I usually run every morning."}]`
- `chat_zh.json` — `[{"role":"user","content":"我喜欢燕麦拿铁。","timestamp":"2026-04-02T08:00:00Z"}, {"role":"user","content":"我现在改喝黑咖啡了。","timestamp":"2026-04-20T08:00:00Z"}]`
- `structured_billing.json` — S4 events (§19.3)
- `mem0_config.json` — `{"llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini", "temperature": 0.1, "top_p": 0.1, "top_k": 1}}, "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}, "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}, "history_db_path": "/tmp/x/history.db", "version": "v1.1", "custom_instructions": null}`

`tests/fixtures/golden/`: `observations_chat_preference.json`, `observations_chat_multisession.json`, `observations_chat_zh.json`, `eps_chat_action_closure.json`, `revision_cases.json`, `work_state_card_failed.txt`, `work_state_card_fixed.txt`, `why_chain.txt`, `transitions_s6.jsonl`, `search_explain_keys.json`.

### §18.3 Named tests per group (each names the public functions it exercises)

**A — `test_foundation.py`**
1. `test_import_topology` — module import graph respects §0.4 (`mvmi/*` never imports graph/compiler/ops/ingest; `models` imports no compiler at module level).
2. `test_enums_are_str_enums_and_values` — every enum in §2 is `(str, Enum)` with the exact member sets; `FeedbackType.state_wrong`, `EdgeType.causal` exist.
3. `test_witnessed_models_reject_empty_anchor_ids` — `Observation/EP/MemoryAtom/GraphNode/GraphEdge/InterfaceEP/ContextItem/Transition` with `[]` raise pydantic errors; `Slots` closure validator.
4. `test_memory_atom_rejects_edge_keys_and_text_mismatch` — `content={'edges': []}` / `content['text'] != text` rejected.
5. `test_contextir_rejects_duplicate_item_ids_across_slots`.
6. `test_timeutil_fixed_width_and_parsers` — `to_utc_iso` on `Z`, `+05:30`, naive, date-only, datetime, epoch → 32 chars; `next_day_midnight`, `month_window`, `age_days`.
7. `test_text_tokenize_entities_topics_dates` — `tokenize('patch retry.ts and billing_e2e')` keeps identifiers; `extract_entities('Sarah joined the billing team in March.')` → PROPER Sarah only; `extract_dates('what happened in May', base_date='2026-09-02')` → May 2026 window; `'retry.ts may change'` → no dates; ZH `'5月发生了什么'`.
8. `test_parse_llm_json_ladder` — fenced, prose-wrapped, top-level array salvage, garbage → `[]` + `caplog` error.
9. `test_prompt_builders_round_trip` — for each builder `parse_sections(build_x(...))` returns the inputs; every `*_PROMPT` first line is `HMS_TASK: <task>`; the six JSON prompts' (the `ROOT_KEY` tasks: the five staged JSON tasks plus `observation_extraction_compact`) OUTPUT FORMAT sections parse as JSON with the §6.3 root key; `PROCEDURAL_SUMMARY_PROMPT`'s OUTPUT FORMAT section contains the seven §6.3 markdown headings/labels in order and the line `Return ONLY the markdown above. No JSON, no wrappers.`; `build_edge_prompt` nodes carry all of `u,b,a,y,r,subject,predicate,objects`.
10. `test_config_mem0_dict_and_negative_cases` — `HMSConfig.from_dict(mem0_config.json)` (the UNMODIFIED fixture dict) fails ONLY with `CONFIG_002` on `qdrant` (raised by the plain-Python pre-check, §5.3); same dict with `vector_store.provider='sqlite'` constructs and `store.path == history_db_path`; renamed key warns (`DeprecationWarning`) + remaps; removed key → `CONFIG_003`; unknown key → `CONFIG_001` naming it; bogus provider with and without `config` → same `CONFIG_002`; `config: None → {}`; `HMSConfig()` constructs with no optional deps; every Literal rejects a bogus value.
11. `test_schema_migrations_and_capabilities` — fresh DB `user_version == 3` (`SCHEMA_VERSION`; `MIGRATIONS == {1: build_ddl(1), 2: upgrade_2(), 3: upgrade_3()}`, `build_ddl() == build_ddl(3)`, `'PURGE'` only from version 2, `FTS_DDL` only from version 3), the 22 base `TABLES` + 1 `VIRTUAL_TABLES` (+ 5 FTS shadow tables) exist (`compiled_states` included), re-open idempotent, newer `user_version` → `STORE_002`; `capabilities()` has `json1`, `fts5`, `sqlite_version`; `PRAGMA table_info(history)` starts with Mem0's 10 columns in order; `source_events.role` CHECK excludes `'extracted'`; `ux_atoms_head` rejects a second non-archived version in one lineage.
11b. `test_purge_gate_and_schema_migration_2` — the seven `PURGE_GATED_TRIGGERS` carry `PURGE_GATE`, the other 15 triggers have no `WHEN`; a raw DELETE is `STORE_003` without the flag and succeeds with it; `purge_rows` removes the flag with the rows; a stale flag in a file is cleared at open; a version-1 file (`build_ddl(1)`, rows inserted raw, `'PURGE'` rejected by its CHECK, DELETE blocked) opens at version 2 with `sqlite_master` (type, name, tbl_name, sql) equal to a fresh store, its history row intact and a `PURGE` row now accepted.
12. `test_store_invariants_at_ddl` — raw INSERT without `primary_anchor_id` → `IntegrityError`; DELETE/UPDATE of content on anchors/atoms/edges/eps/history → `StoreError(STORE_003)`; `set_valid_to`/`set_atom_confidence`/`set_edge_label` succeed; `graph.insert_edge(src_id=..., dst_id=..., edge_type='use', witness_anchor_ids=[], created_by='ingest')` AND `store.insert_edge({'src_id': ..., 'dst_id': ..., 'edge_type': 'use', 'witness_anchor_ids': [], 'created_by': 'ingest'})` AND `store.insert_edge(GraphEdge.model_construct(witness_anchor_ids=[], ...))` each raise `WitnessMissingError` before any SQL (`store.trace.tables() == set()` after `trace.reset()`); an unknown anchor id → `WitnessMissingError(details['missing'])`; boundary edge without `witness_edge_id` fails CHECK; with it `edge_projections` row exists (trigger); `insert_edge` MERGE bumps `mentions` and unions witnesses; `add_atom_anchor` on the primary anchor returns False and adds no row.
13. `test_store_transaction_rollback_and_trace` — exception inside `transaction()` leaves no rows; `trace.tables()` lists touched tables; nested transactions re-enter.
14. `test_store_lineage_helpers` — `insert_atom` assigns memory_id/version; `resolve_head`, `list_versions`, `archive_atom`, `restore_atom`; `ux_atoms_lineage_version` rejects a duplicate version.
15. `test_store_entities_keyed_without_run_id` — `get_or_create_entity` for `sarah` under two run_ids returns one id; aliases appended.

**B — `test_providers.py`**
16. `test_hash_embedder_deterministic_normalized` — same text → identical vector across instances, `‖v‖₂ ≈ 1`, dims 64, `sim('oat latte','oat-milk latte') > sim('oat latte','billing test')`, CJK input non-zero.
17. `test_index_parity_sqlite_memory` — NOT parametrized: constructs both `SQLiteIndex(store=store, dims=64)` and `MemoryIndex(dims=64)` in one test; same vectors → identical `search` ordering and scores; `candidate_ids` restriction; `reset`.
18. `test_rule_llm_observation_golden` — `RuleLLM` on `chat_preference`/`chat_multisession`/`chat_zh` prompts == golden JSON (types/predicates/objects/text/relation exactly as §19).
18b. `test_moved_to_yields_lives_in_supersession` — `'I moved to Berlin.'` / `'I relocated to Berlin.'` / `'我搬到柏林。'` extract as R5 `fact`/`lives_in`/`supersession`, objects `['Berlin']`/`['柏林']`, confidence `0.85` (never R1 `likes`/`preference_event`); `decide_revision(<berlin>, <paris>)` → `(REVISE, 'supersession cue')`.
19. `test_rule_llm_ep_and_revision_and_edge_and_coarsen_contracts` — outputs validate against the §6.3 contracts; `chat_action_closure` yields one EP (`a='patched retry.ts to add backoff'`, `u='the webhook kept timing out'`, `y='billing_e2e failed'`, `r=negative`) == golden `eps_chat_action_closure.json` (masked helper); `revision_cases.json` table (each row: new, old → decision) matches §11.3 via the public `decide_revision`; `propose_edges` on the S4 EP pair proposes `causal`; `coarsen_rule` on two `interface_ep`-kind nodes returns an L3 `interface_ep` dict (P non-empty).
20. `test_rule_llm_unknown_task_and_raise_hook` — unknown marker → `LLMError(LLM_002)`; `RAISE:` line → `LLMError(LLM_001)`.
21. `test_scripted_llm_queue_and_cause` — dict/str/Exception responses; exhausted → `LLM_003`; `LLMBase.generate_json` wraps as `LLMError` with `__cause__`.
22. `test_optional_providers_guarded` — `patch.dict(sys.modules, {'openai': None})` → `LlmFactory.create('openai')` raises `DependencyError(DEPS_001)` with pip hint; with a fake `openai` module the provider passes `response_format={'type': 'json_object'}`; same for `anthropic` (system split) and `OpenAIEmbedder`.
23. `test_factory_registry_single_source_of_truth` — `register_provider('llm', 'x', ...)` makes both `LlmConfig(provider='x')` validate and `LlmFactory.create('x')` work; caller dict not mutated.

**C — `test_ingest.py`**
24. `test_one_event_one_anchor_and_dedup_at_same_event_time` — N messages → N events/anchors; identical re-add with the same `timestamp=` → `dedup=True`, no new anchor, rows `event='NONE'`, `store.atom_anchor_ids(atom) == [<anc_1>]` unchanged (the anchor is already the primary witness; `atom_anchor` row count unchanged) and `graph_nodes.mentions == 2` (`bump_mentions`); same text at a later `timestamp=` → new event + anchor, rows `event='NONE'`, and NOW the existing atom gains a `support` `atom_anchor` row for the new anchor (`atom_anchor_ids == [<anc_1>, <anc_2>]`).
25. `test_payload_normalization_and_internal_types_rejected` — §4.3 shapes; `failed_tests` alias; `status` derived; `add_events([{'object_type': 'feedback', ...}])` → `VALIDATION_007`; malformed → `VALIDATION_005` with `index`.
26. `test_router_observation_vs_state` — `chat_preference` → observations only; `chat_action_closure` → both an observation and an EP; `system` role → anchored, not extracted.
27. `test_structured_closure_builds_eps_without_llm` — setup WITHOUT any LLM call: `(ev, anc, _), = ingest.events.anchor_events(store, events_from_messages([{'role': 'user', 'content': 'Do not modify the subscription state machine.'}], scope=scope4, event_time='2026-06-15T09:00:00Z', user_meta={}, visibility='private'))`, then `ingest.atoms.create_atom(store, index, embedder, atom_from_observation(Observation(type='constraint', subject='project', predicate='must_not', objects=['subscription state machine'], relation='constraint', text='Do not modify the subscription state machine.', date_observed='2026-06-15', anchor_ids=[anc.anchor_id], scope=scope4), scope=scope4))` (the background constraint); then `hms_scripted.llm.calls == []` is asserted ONLY around `hms_scripted.add_events(structured_billing.events_fail)` → EP1 (`u`, `a`, `y`, `r`, `anchor_ids=[outcome, action]`, `kappa.evidence == anchor_ids`), derived constraint atom with entity `retry.ts`, `use` edge to the background constraint (witness = action anchor), `distill` edge (witness = outcome anchor); second batch → EP2, `temporal` + `causal` EP1→EP2 across batches (via `store.last_ep`/`last_open_action`); a `test_log` in a LATER batch closes a `file_diff` from an EARLIER batch (`last_open_action`) but not one older than `closure_max_gap_hours`; `tool_call status=error` self-closes; an open `tool_call` whose payload carries `constraints` self-closes with `y=''`, `r='unknown'`, `kappa.constraints == payload constraints`; `test_log` without an open action → event atom only; a `sensor_trace` event → event atom only, `event_type == 'tool_result'`, `raw_ref == 'sensor://<sensor>'`.
28. `test_infer_false_raw_atoms_link_entities` — rows carry `actor_id`/`role`; `atom_entity` rows exist; `history` `ADD`.
29. `test_procedural_branch_is_l1_episode` — `add(..., agent_id, memory_type='procedural_memory')` → one atom `scale_level='L1_micro'`, `memory_type` promoted; `prompt=` replaces the system prompt (scripted asserts).
30. `test_revision_inline_deferred_off` — `chat_preference` with each `ingest.revision_mode`: inline → UPDATE; deferred → two valid atoms until `consolidate()` resolves; off → two valid atoms.
30b. `test_relocation_revises_previous_residence` — `'I live in Paris.'` then `'I moved to Berlin.'` in one scope → the second row is `UPDATE` with `predicate='lives_in'`, `version == 2`, the Paris atom `archived_reason='revised'`, one `revision` edge Paris → Berlin, one valid atom left.
31. `test_lock_discipline_llm_runs_unlocked` — a `ScriptedLLM` queued with a callable response (`lambda messages: {'observations': []}`) that asserts `store.in_transaction is False` when invoked.
32. `test_create_atom_rolls_back_on_embedding_error` — embedder raising on the second text → no partial rows in any plane table.

**D — `test_mvmi.py`**
33. `test_filter_dsl_conformance[index_provider]` — 14 cases (scalar eq, list in, `'*'`, `{gte}` on `created_at`, `{lte}` date-only literal, `{icontains}` on a metadata key, `meta.key`, `OR`, `NOT`, nested `AND(OR)`, entity axis, topic axis, `anchor_id`, unknown operator → `ValueError`) identical between `lower_to_sql` and `evaluate`; json1 fallback path via monkeypatched capabilities.
34. `test_plan_query_routes_and_gates` — `"what happened in May"` → temporal + May window; `"5月发生了什么"` same; `"everything about Sarah"` → entity + hard gate; `"why must retry.ts not be modified"` → detective with identifier `retry.ts`; `"retry.ts may change"` → no time window; `"what coffee does the user like"` → route `preference` with `hard_type_gate == ('obs_type', ['preference_event', 'habit'])` and `scale_default == ['state']`.
35. `test_axis_query_seven_axes_and_aggregations` — seeded store: time/validity(as_of)/entity/topic/role/sentiment/scale gates; then per HMS §3.5, through `hms.axis_query(...)` → `AxisResult`: `project_axis='value', aggregation='latest'` → `projection == [[<latest objects>]]` per `(subject, predicate)`; `project_axis='time'` → ascending `(time_anchor, atom_id)` timeline; `project_axis='entity'` → `{key: [atom_ids]}`; `project_axis='topic'` → `{topic: [atom_ids]}`; `aggregation='count'` → `count == n`, `hits == []`, and the trace shows one `memory_atoms` COUNT only; `aggregation='summary'` → `summary == f"{n} atoms; latest: {text}"`; `aggregation='topk'` with `text=None` → `VALIDATION_013`; `AxisQuery(filters={'user_id': U, ...})` without `scope` validates and runs; `scale_constraint=['event','entity','L1_micro']` after `consolidate()` returns the `entity` atom for Sarah; `filters={'visibility': 'team'}` returns team+event+public items only (lattice gate).
36. `test_search_fusion_explain_threshold_after_fusion` — `explain=True` keys exactly §13.4 (`axis_gates` has `obs_type`); an identifier hit with weak semantic survives `threshold=0.3`; `entity_boost == 1.0` for an entity with degree ≤ 3 and `< 1.0` for degree 5; `max_possible_score` reflects active signals; two calls byte-identical; the preference route applies the `obs_type` gate (`primary_axis == 'type'`) and falls back (`'fallback'`) when nothing matches; `search` touches no `graph_*` table (`store.trace`).
37. `test_search_hard_time_gate_and_fallback` — May query returns only May atoms; a query with a window that matches nothing falls back with `primary_axis='fallback'`.
38. `test_get_all_show_expired_and_order`.
39. `test_atom_to_row_shape_and_no_metadata_leak` — key order §4.1; `metadata` only caller keys; `RESERVED_ATOM_KEYS ⊇ set(PRAGMA table_info(memory_atoms) names)` (including `source_ep_ids_json`); `mvmi.query.atom_to_row(atom) == atom.to_public_dict()`.

**E — `test_graph.py`**
40. `test_cohesion_formula_values` — chain of 3 with no boundary → 1.0; isolated → 0.0; S = {EP1, EP2} of §19 → `pytest.approx(2/3, abs=1e-6)` (`cohesion_score` is stored unrounded).
41. `test_coarsen_project_refine_roundtrip` — on the S4 graph: IEP `cohesion_score == pytest.approx(2/3, abs=1e-4)`, slots per §11.5 (`r == 'positive'`, `b` has each constraint text exactly once), `slots.kappa.evidence == evidence_anchor_ids`, invariant `"Do not modify the subscription state machine."` with `support_node_ids == [<ep1_atom>, <ep2_atom>]`, `anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`, `confidence == 0.8`; the `failed_path` invariant has `anchor_ids == sorted({<anc_e1>, <anc_e2>})`; the IEP atom has `source_ep_ids == [<ep1>, <ep2>]`, `topic_tags == ordered union of the support atoms' topics`, entity keys ⊇ `{'retry.ts', 'idempotency.ts', 'billing_e2e'}`; `evidence_anchor_ids == ∪ support anchors ∪ witnesses(E_in)`; `project` writes 3 boundary edges each with `edge_projections` (4 rows total); `refine(project(coarsen(S)))['boundary_edges']` ids == `boundary(S)` ids; re-coarsen is idempotent (`support_hash`).
42. `test_interface_ep_is_not_a_summary` — `InterfaceEP` with 1 support node or empty evidence rejected; every slot cites support ids; `Slots` is the same class for EP and IEP; `Invariant` without `anchor_ids` rejected, `Residual.anchor_ids` defaults to `[]`.
43. `test_promote_skills_from_one_interface_ep` — invariant with ≥2 support nodes → skill atom `L3_macro` (`anchor_ids == inv.anchor_ids`, `confidence == inv.confidence`) + `distill` edge at `L3_macro` with `is_boundary=0`, `witness_edge_id == <lowest-id valid use edge support_node → constraint atom>` and `store.projections_of(distill_edge) == [both use edges]` (trigger + `add_projection`); `graph.stats(scope)['traceable_high_edges'] == 1.0`; second run is idempotent.
44. `test_explain_backward_bfs_and_chain` — `why()` core on the derived constraint: nodes/edges/paths/anchors as §19.5; `render_chain` == golden `why_chain.txt`; `hops=0` returns only the target.
45. `test_refine_called_only_by_explain_and_compile` — source inspection: the token `refine(` appears only in `graph/operators.py` (definition), `graph/explain.py` and `compiler/compile.py` (handoff goes through compile); `compile(mode='fast')` trace has no `graph_*`/`interface_ep*` tables; `compile(mode='axis_graph')` trace touches `graph_*` but not `interface_ep*` tables.
45b. `test_coarsen_l3_from_two_interface_eps` — two L2 IEPs (two S4-like runs in one scope, joined by a projected `causal`/`use` edge) → `consolidate()` reports `interface_eps_l3 == 1`; the L3 IEP's `support_node_ids` are the two L2 IEP atoms, `scale == 'L3_macro'`, `refine(hops=0)['nodes']` returns exactly those two nodes; `coarsen_l3=False` → 0.

45c. `test_coarsen_llm_policy_validates_enums` — policy `llm` with a proposal whose refs are valid but `slots.r == 'great'` and `kappa.visibility == 'everyone'` → no exception: `coarsen_op_version == 'llm-coarsen-v1'`, `slots.r` and `slot_support['r']` are the rule values, `kappa.visibility == 'private'` (rule) while the LLM constraints and `slot_support['kappa']` are kept; one warning each; `round_trip_check` holds.

**F — `test_compiler.py`**
46. `test_slot_rules_total_and_disjoint` — every atom kind lands in exactly one slot; a live atom with a FUTURE `valid_to` (`expiration_date`) lands in its natural slot, an expired one (`valid_to <= as_of`) in `audit` with reason `archived:expired`; kappa items deduped against selected constraints AND audit items, and across episodes (one κ item per text); event atoms covered by episodes dropped; `compile()` with `mode=None` uses `config.compiler.default_mode` (`default_mode='fast'` → `retrieval['ablation'] == 'axis'`).
47. `test_work_state_card_failed_golden` — after constraint + `events_fail`: `compile("fix checkout webhook retry", filters=scope4, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()` == `work_state_card_failed.txt` (§19.3, `golden.text`); slots disjoint; every item witnessed; `used_tokens ≤ 800`; `retrieval['dropped_covered'] == 2`; row counts unchanged.
48. `test_work_state_card_fixed_golden` — after `events_fix`: == `work_state_card_fixed.txt`; `retrieval['dropped_covered'] == 4`; `mode='deep'` and `mode='axis_graph'` produce the same card after normalizing the `Mode:` header.
49. `test_budget_shares_min_constraints_and_spill` — tiny budget keeps ≥3 constraints; unused facts share spills to episodes.
50. `test_compile_modes` — `fast` never touches graph tables; `deep` on the S1 scope yields the invalidated latte item in `audit` (reason `revised`); `auto` without EPs does not expand.
51. `test_pcir_obligations` — parametrized: an atom whose only anchor is disputed → `compile()` still builds the item (`anchor_ids` includes the disputed anchor, `explain['disputed_anchor_ids']` non-empty) and `to_pcir()` raises `ProofObligationError` with `evidence_coverage` violated (`details['missing_witness'] == [item_id]`); unresolved conflict → `no_unresolved_conflict`; `filters={'visibility': 'team'}` with a private atom → the private atom is NOT selected (`retrieval['dropped_governance'] >= 1` or simply absent) and `governance` is `satisfied`; a hand-built `ContextIR` (`retrieval={'audience': 'team', 'anchors': [...]}`) containing a `visibility='private'` item → `lower()` reports `governance` violated; revoked anchor (via `hms.revoke`) → `not_revoked`; `require_witness=False` returns `ok() is False` instead of raising.

**G — `test_ops.py`**
52. `test_feedback_matrix[feedback_type]` — 11 cases + 3 sub-cases: expected `actions` names in order, history `FEEDBACK` row, only `privacy_violation` changes `visibility` (asserted for the other 10), `evidence_wrong` never leaves an atom valid without an undisputed witness, `state_missing` without correction → `VALIDATION_007`, unknown type → `VALIDATION_006`; sub-cases: `state_missing_no_target` (`feedback(None, 'state_missing', correction={'scope': scope4, 'text': ...})` and `feedback('user_id=alice&run_id=run-a', 'state_missing', correction='...')` both create a `state` atom in that scope with `target_kind='scope'`; without any scope → `VALIDATION_001`), `connection_reason_wrong_same_type` (label-only correction → `[{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}]`, edge still valid, label updated, `fb_anchor` added as witness), `evidence_wrong_revoke` (`feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` → one `revoke` action; `anchors.revoked_at` set).
53. `test_checkpoint_rollback_symmetric` — S6 flow (the `mid` compile uses the default `as_of`): `restored`/`archived`/`edges_invalidated`/`edges_created` exactly; `get(mem)` back to the original text; history `[ADD, FEEDBACK, UPDATE, ROLLBACK, RESTORE]`; rollback of an atom archived-before-checkpoint-and-restored-after re-archives it; `ep_state`: `delete(<episode memory>)` after a checkpoint → `store.list_eps(scope4)` shrinks, `handoff(goal=None).goal` changes, and `rollback(cp)` restores `eps.valid_to IS NULL` and the previous goal; unknown id → `ValueError`.
54. `test_handoff_state_package_roundtrip` — `handoff(filters=scope4, goal=...)` → JSON → `StatePackage.from_json` equal; witness closure incl. `source_events` (one per anchor) and `atoms`; `checkpoint_id` set; `rollback(pkg.checkpoint_id)` works.
54b. `test_handoff_import_roundtrip` — after S4 (`card2` stage): `pkg = A.handoff(filters=scope4, checkpoint=False)`; a fresh `B = HMS(config, store=StateStore(':memory:', now=clock.now, id_factory=SequentialIds()))`; `rep = B.import_package(pkg.to_json())` → `rep['imported']['atoms'] == len(pkg.atoms)`, `rep['imported']['anchors'] == len(pkg.anchors)`; `B.compile('fix checkout webhook retry', filters=scope4, budget_tokens=800, as_of='2026-06-15T12:00:00Z').to_text() == card2` (cards carry no ids); importing the same package again → `rep['imported']` all zero, `rep['skipped']` non-zero; `import_package(pkg, run_id='run-import')` re-scopes every object (`get_all(filters={'user_id': U, 'run_id': 'run-import'})` non-empty) and ids stay stable (`B.get(<mem_uc>)` works).
55. `test_transitions_export_golden_and_idempotent` — S6 tuples == §19.6; `fmt='jsonl'` == golden (`golden.masked`); every row's `kappa_t` has exactly the keys `constraints, visibility, permission_scope, provenance`; second export adds no rows; `since`/`scale`/`kinds` filters; after `consolidate()` one `interface_ep` row exists (`kinds=['interface_ep']`, `scale='L2_meso'`).
56. `test_consolidate_phases_reports_and_decay` — after S4: `consolidate()` → report == §19.7 (`entity_atoms == 6`, `interface_eps == 1`, `projected_edges == 3`, `interface_eps_l3 == 0`, `skills == 1`, `transitions == 1`, `compiled_states == 1`); temporal/causal edges not duplicated (`temporal_edges == causal_edges == 0`, `mentions == 2` on both — MERGE); `store.last_compiled_state(scope4)['card']` == `compile(..., mode='deep').to_text()`; `orient` reports `last_compiled`; then `store.set_atom_confidence(<atm_e1>, 0.2, at=clock.now())`, `clock.tick(days=400)`, `consolidate(phases=['prune'])` → `report['decayed'] == 3` (`<atm_e2>`, `<atm_e3>`, `<atm_e4>`: 0.7 → 0.61 with `UPDATE` history rows), `report['archived'] == [<atm_e1>]` with an `ARCHIVE` history row witnessed by the scheduler anchor; pinned/constraint/skill/entity/conflicted atoms untouched; `collect` re-routes an event whose atoms were rolled back from an aborted Txn B.
56b. `test_revoke_cascade` — after S4: `hms.revoke(<anc_e2>)` → `archived == [<atm_e2>]` (its only witness), `<ep1_atom>` and `<atm_dc>` stay valid (other witnesses remain; the revoked anchor is disputed on them), `edges_invalidated == [<edg_distill_ep1_dc>]` (its only witness was `<anc_e2>`), `anchors.revoked_at` set, history `ARCHIVE` witnessed by the system anchor; `compile(...).to_pcir()` afterwards passes `not_revoked` (no live item cites the revoked anchor undisputed); a second `revoke` → `noop`.
56e. `test_revoke_existence_semantics_and_projection_sources` — after S4 + `consolidate()`: the two low `use` edges `<ep1> → <atm_uc>` / `<ep2> → <atm_uc>` are the `projections_of` both the L2 `use` edge `<iep> → <atm_uc>` and the L3 skill `distill` edge; `revoke(<anc_e4>)` (EP2's PRIMARY witness, `anchor_ids[0]`) archives only `<atm_e4>`, `<ep2_atom>` stays valid with `atom_anchor_ids == [<anc_e3>]` (`include_disputed=True` → `[<anc_e4>, <anc_e3>]`), the IEP and all four edges stay valid; `revoke(<anc_e1>)` invalidates `<use ep1→uc>` (`witness_revoked`) and NOT the L2/L3 edges (a valid source remains; `to_pcir().ok()`); after `add_edge_witness(<L2 use>, <anc_uc>)`, `revoke(<anc_e3>)` archives `<atm_e3>` and `<ep2_atom>` (`archived_reason='revoked'`), invalidates `<use ep2→uc>` (`witness_revoked`) and then the L2 `use` and L3 `distill` edges with `invalidated_reason='projection_revoked'` while each still has an unrevoked witness; projected edges are reported after the witness-rule edges; the IEP stays valid (`<anc_e2>` in its evidence); second revoke → `noop`. A helper asserts the global rule over every edge of the scope after each step: valid ⇒ (≥1 unrevoked witness ∧ (no projections ∨ ≥1 valid source)); `witness_revoked`/`projection_revoked` ⇒ the negation.
56f. `test_relationship_not_useful_keeps_high_edge_with_other_sources` — same setup: `feedback(<use ep1→uc>, 'relationship_not_useful')` → exactly one `invalidate_edge` action, the L2 `use` and L3 `distill` edges stay valid; `feedback(<use ep2→uc>, 'relationship_not_useful')` → three `invalidate_edge` actions in the order `[<use ep2→uc>, <L2 use>, <L3 distill>]`, all with `reason='feedback:not_useful'`.
56c. `test_entity_atoms_after_consolidate` — after S2/S3 + `consolidate(filters={'user_id': U})`: `get_all(filters={'user_id': U, 'scale': 'entity'})['results']` contains one row with `memory == 'Sarah'`, `atom_type == 'entity'`, `hms.source_kind == 'entity'`, `content['mentions'] == 3`, `anchor_ids` == the three Sarah anchors sorted; `axis_query({'filters': {'user_id': U, 'entity': 'Sarah'}, 'scale_constraint': ['event', 'entity', 'L1_micro']})` returns the entity atom plus the two `event` atoms; a second `consolidate` creates nothing new; adding a fourth Sarah message and consolidating revises the entity atom (`version == 2`, revision edge witnessed by the scheduler anchor).

56g. `test_purge_dry_run_reports_and_changes_nothing` — a deleted tombstone 100 days old: `purge(dry_run=True)` reports it (`text_sha256`, `bytes`, `protected` keyed by the five reasons, `anchor_id None`) and `store.stats()` / its history are unchanged; a 200-day threshold examines nothing; `older_than_days=-1` → `VALIDATION_009`; empty scope → `VALIDATION_001`.
56h. `test_purge_protections_keep_needed_rows` — after S4 + consolidate: the head of a two-version lineage (`head_version`), a tombstone that is the src of a `use` edge (`edge_endpoint`), a tombstone recorded live by a checkpoint (`checkpoint_live`, restored by a later `rollback`), the EP atom of the InterfaceEP (`interface_ep_support`) all survive `purge(dry_run=False)` while a tombstone sharing its anchor with a live atom is purged with the anchor kept (`anchors == before + 1`, the purge's own); `keep_witnessed=False` then removes only the anchor + source event of a raw atom nothing else cites.
56i. `test_purge_materializes_with_audit_and_keeps_survivors_intact` — two tombstones (`deleted`, `decayed`): `purged == 2`, rows/nodes/links/vectors gone, `history` ends `[..., 'DELETE'|'ARCHIVE', 'PURGE']` with `is_deleted True`, `old_memory None`, `details_json` carrying `text_sha256` and `cutoff`, the purge `system` event has `payload['op'] == 'purge'`; `stats()` deltas exact (+1 anchor/event, +2 history, edges unchanged); the deep card is byte-identical before/after; `rollback` to a later checkpoint restores nothing; a second purge is a no-op; a raw `DELETE FROM memory_atoms` is `STORE_003` again and `meta.purge_in_progress` is absent.
56j. `test_prune_runs_retention_only_when_both_switches_are_on` — `prune` reports `purged == 0` by default and with `retention.enabled` alone; with `enabled + run_in_prune` (`older_than_days=30`, clock +60 d) `purged == 1`, the row is gone and its history ends with `PURGE`.
56k. `test_vacuum_shrinks_file_after_purge` — 40 tombstones of 4 KB on a file store: after `purge(dry_run=False)`, `vacuum()` inside `transaction()` → `STORE_001`; outside it `after_bytes < before_bytes`, `freed_bytes >= 160 000`, `os.path.getsize(path) == after_bytes`.
56d. `test_consolidate_llm_proposer_drops_malformed_proposals` — S4 with `graph.causal_proposer='llm'` and a scripted `edge_proposal` reply carrying `witness_refs` of `7`, `[['w1']]` and `[{'ref': 'w1'}]` plus two valid edges whose `confidence` is `NaN` (literal) and `'nan'` (string) → `consolidate()` completes for the scope (`interface_eps == 1`, `compiled_states == 1`, `last_consolidation` meta set), the three malformed proposals are dropped with one warning each, the new `use` edge is persisted with `confidence == 0.7` and `proposer == 'scripted@v1'`, and the existing `causal` edge keeps `confidence == 0.7` with `mentions == 2`.

**H — `test_facade.py`**
57. `test_mem0_signatures_verbatim` — `inspect.signature` of the 10 Mem0 methods + `chat`/`project` equals §17.2 and of the HMS-native methods equals §17.3 (`compile(mode: str | None = None)`, `feedback(target_id: str | None, ...)`, `context(query, *, filters, top_k, as_of, budget_tokens)`); `MemoryBase` abstract set; the public attributes of §17.1 exist.
58. `test_mem0_envelopes_and_validation_codes` — `add` envelope; `update/delete/delete_all` messages verbatim; `get` missing → `None`; `history` bare list; `VALIDATION_001/002/003/008/009/010/011`; whitespace id → `ValueError`; top-level `user_id` in `search` → `ValueError` subclass; `delete_all()` message verbatim.
59b. `test_purge_facade_is_explicit_default_off_and_audited` — `hms.delete(mem)` then +100 d: `purge(filters=S1, older_than_days=90)` is a dry run listing `mem`; `dry_run=False` with the default config → `ConfigError(CONFIG_001, details={'retention.enabled': False, 'dry_run': False})`; `filters={}` → `VALIDATION_001`; an `HMS` with `retention.enabled=True` on the same store purges it (`get(mem) None`, `latest_version None`, `history(mem)` events `['ADD', 'DELETE', 'PURGE']` with Mem0's key set, `why(kept)` still resolves, `why(mem)` → `NOT_FOUND_001`), `vacuum()` returns the four keys, a second purge purges 0.
59. `test_update_delete_expiration_lineage` — `update(id, text=)` → same `id`, `version == 2`, old archived, revision edge, `UPDATE` row; `update(metadata=)` → no new version; `update(expiration_date=None)` clears `valid_to`; `update(expiration_date='2026-12-31')` → `valid_to == '2027-01-01T00:00:00.000000+00:00'`; `delete` → `get is None`, row kept with `archived_reason='deleted'`, `DELETE is_deleted True`.
60. `test_from_config_mem0_dict_and_rerank_noop` — OFFLINE: `d = deepcopy(mem0_config.json)`; `d['llm'] = {'provider': 'rule'}`, `d['embedder'] = {'provider': 'hash'}`, `d['vector_store'] = {'provider': 'sqlite', 'config': {}}`, `d['reranker'] = {'provider': 'overlap', 'config': {}}` (keep `history_db_path`, `version`, `custom_instructions`); `HMS.from_config(d)` constructs (`store.path == history_db_path`) with an `OverlapReranker` and no warning, `search(rerank=True)` rows carry `rerank_score`; `{'provider': 'cohere'}` → `CONFIG_002` with `details['kind'] == 'reranker'`; the UNMODIFIED fixture is only used by test 10.
61. `test_migrate_from_mem0_walks_get_all` — a fake `memory` object → rows re-added with `infer=False` and `timestamp=created_at`.
62. `test_cli_smoke` — `hms add/search/compile` via `main([...])` on `HMS_DIR=tmp_path`.

**T — `test_scenarios.py`**
63. `test_scenarios_end_to_end[index_provider]` — S1–S6 (§19) in sequence on ONE `hms`, every `expect` block asserted exactly (cards byte-for-byte via `golden.text`, chain byte-for-byte, transitions via `golden.masked`), then `consolidate()` + `graph_stats()` == §19.7 including the ablation block (`fast`/`axis_graph`/`deep` on the S4 query).

**Retrieval pool (group D, `tests/test_retrieval.py`)**
64a. `test_equivalence_theorem[index_provider × n5/n30/n60 × whole/channels]` — a seeded random corpus (names, files, CJK, 18 months, ~10% without vectors), 20 queries × `top_k ∈ {3, 5, 10}` × `threshold ∈ {0.0, 0.3}`: `(atom_id, score)`, the 7 Mem0 keys and the 7 HMS keys of `score_details` are `==` (bit-identical) between `_exhaustive_search()` (the pre-pool algorithm kept in the test as the reference) and `execute_axis_query`, with the whole-gate rule on and with `hard_gate_pool=False` (channels only); `test_equivalence_theorem_milvus` repeats it on the Milvus provider (`importorskip`).
64b. `test_pool_recall_overlap_at_n500` — N=500 > pool_max: prints the per-query top-20 overlap with the exhaustive path (no hard threshold), asserts every channel size `<= K` and the pool `<= pool_max`.
64c. `test_no_newest_first_truncation` — an old unique identifier atom is found with `ScoringConfig(max_candidates=10)` (which warns `DeprecationWarning`) in both recall modes, while the reference with the same cap loses it.
64d. `test_channel_sizes_and_recalled_by`, `test_hard_gate_whole_set_enters_pool_or_channels`, `test_safety_limit_warns_never_truncates`, `test_axis_only_paths_keep_sql_semantics`, `test_max_candidates_is_a_deprecated_alias`, `test_compile_candidates_reuse_the_pool`, `test_python_filter_scores_the_whole_gate` — provenance keys, whole-gate vs channels on a hard time gate (pool ∩ G), the two safety-limit warnings told apart by prefix (the pool warning alone under native pushdown), text-less paths never calling `recall_pool`, the deprecated alias (None sentinel: `ScoringConfig(**ScoringConfig().model_dump())`, `HMSConfig.model_validate(HMSConfig().model_dump())` and `coerce_config` are warning-free), `ContextIR.retrieval['channels_used'] == ['gate']` on a small compile, and the Python-side filter (json1 off, N > pool_max) scoring G whole bit-equal to the exhaustive path.
64e. `tests/test_index_sync.py` — `test_listener_dispatch_follows_commit_order` (sqlite / memory / milvus: archive with a slowed listener then restore from two threads → listener calls in commit order, each the full snapshot, the provider live again), `test_search_inside_transaction_never_deadlocks` (search inside `store.transaction()` vs a concurrent search, 5 s join), `test_listener_registered_once_and_removed_on_close` (two facades on one store: one listener each, `close()` removes exactly its own), `test_late_topic_and_entity_reach_the_keyword_channel` (a topic / entity linked after insert is in `atom_fts`; an archived version never regains a row).
64f. `tests/test_server.py::test_internal_errors_are_redacted` — a `StoreError` carrying `path` / `sql` details returns 500 with `message == 'internal error'`, empty `details` / `debug_info`, the path only in the server log line with the request id; `RuntimeError` / `KeyError` → 500 `INTERNAL_000` JSON with `X-Request-ID`, never Starlette's plain-text 500.
64g. `tests/test_retrieval_v3.py` (RETRIEVAL_BENCH_2026-09-10) — `test_axis_ids_unordered_and_index_hints` (order=None emits no ORDER BY; `index_hint=TIME_INDEX` plans through ix_atoms_time; a non-identifier hint → VALIDATION_000), `test_channels_search_inside_the_level_gate` (keyword / entity candidates under a March gate with |G| > K stay inside G and find gated atoms the scope-wide top-K misses; a channel-only pool under the gate is ⊆ G), `test_entity_gate_is_a_non_correlated_subquery` (`_plan_gates` lowers the entity gate to `atom_id IN (SELECT …)` selecting the DSL's EXISTS set; G enumerated once through atom_entity, no probe), `test_keyword_channel_two_phases` (AND phase first, OR phase only to fill; LIKE fallback parity), `test_write_generations_and_data_version`, `test_memory_atoms_updates_stay_listener_covered` (source scan), `test_sqlite_cache_follows_every_write` (upsert / replace / delete / archive / restore / set_valid_to / rollback / clear / reset / a second connection, parity with `rank_hits` and with `axis_ids` after each), `test_select_top_k_rows_and_scores_are_bit_equal`, `test_recency_channel_rule`, `test_semantic_channel_scores_are_reused_in_pool` (one channel search, an in-pool search only over the remainder, bit-equal to a full recompute), `test_axis_result_carries_retrieval_stats`, `test_atom_type_pushdown_matches_sql`, `test_facade_search_passes_retrieval_config`, `test_ingest_upserts_carry_index_attrs`, `test_milvus_reopen_loads_collection` (`importorskip`). `tests/test_perf_paths.py::test_memory_store_clones_template_identically` / `test_file_store_migrates_before_wal_switch` additionally pin the file-database clone (sqlite_master rows, user_version and page_count equal to the migration ladder; an existing file still runs the ladder).

**Cross-cutting (group T, in `test_foundation.py`)**
64. `test_bilingual_docstrings` — walks every `hms_elite/**/*.py`, `tests/**/*.py`, `examples/*.py` module: the module docstring has an EN line followed by a ZH line (contains CJK), the eight `# Purpose/# 目的/# Input/# 输入/# Output/# 输出/# State effect/# 状态影响` header lines are present in order; every public function/method (`inspect`, name not starting with `_`, defined in the module) has a docstring whose first two lines are EN then ZH and which contains the markers `Input / 输入：`, `Output / 输出：`, `State effect / 状态影响：` in that order, with every `<arg>:` line under Input immediately followed by an `<arg>：` line (full-width colon).
65. `tests/test_context.py` (§6.7) — `test_context_shape_and_none_parts`, `test_context_memo_reuse_and_invalidation`, `test_context_idle_writes_keep_the_memo`, `test_store_scope_version_moves_at_commit`, `test_context_timestamps_from_time_anchor`, `test_context_builders_are_the_harness_builders`, `test_context_config_fields_and_overrides` — `HMS.context` / `StateStore.latest_event` / `touch_scope` / `unversioned` / `scope_version` / `compiler.context.*` / `CompilerConfig.context_*`; `tests/test_async.py` and `tests/test_facade.py` pin `context` in the 41-method / 18-native parity and signature lists.


---

## §19 Golden scenarios S1–S6 (exact inputs and expected outputs)

All six run in sequence on ONE `HMS` (`rule` LLM, `hash` embedder, index provider parametrized, `ingest.extraction_output = 'full'` pinned by the module-level `config` fixture of `tests/test_scenarios.py` because the expectations below are stated on the rule LLM's full observations — R1 preference confidence 0.8, R2 constraint 0.9, the §19.8 hashes — so a change of the library default — `lines` since 2026-09-18 (row 108), which like `compact` derives confidence 0.7, §6.5 / §6.6 — cannot rewrite them) with the §18.1 fixtures (`FrozenClock` start `2026-09-02T10:00:00.000000+00:00`, `SequentialIds`). The scenario runner calls `clock.tick(seconds=60)` before every public call. Ids in expectations are symbolic (`<mem_coffee>` etc.) and resolved from earlier return values; golden text files never contain ids or clock instants (the golden cards use explicit `as_of`; evidence lines use fixture timestamps; the S6 `mid` card is asserted by slot content only, never against a golden, because it must use the clock). `sha256:` prefixes below are computed with the §0.1 rule over the §4.3 normalized payloads (full hashes listed in §19.8) — the rule LLM, the ingest and the test suite are aligned on exactly these.

### §19.1 S1 — preference recall with revision (`tests/fixtures/events/chat_preference.json`)

```python
U = "alice"; s1 = {"user_id": U, "run_id": "run-coffee"}
r1 = hms.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", **s1)
r2 = hms.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", **s1)
```
Rule extraction (§11.1 R1): msg1 → `{"type": "preference_event", "subject": "user", "predicate": "likes", "objects": ["oat-milk lattes in the morning"], "relation": "single_fact", "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02", "sentiment": "positive", "topic_tags": ["coffee", "preference"], "entity_mentions": [], "confidence": 0.8}`; msg2 → `{"type": "preference_event", "subject": "user", "predicate": "likes", "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.", "date_observed": "2026-04-20", "sentiment": "positive", "topic_tags": ["coffee", "preference"], "confidence": 0.9, "supersedes_hint_ids": ["0"]}`. Revision decision (§11.3 rule 6) → `REVISE`.

Expect:
- `r1 == {'results': [{'id': <mem_coffee>, 'memory': 'User likes oat-milk lattes in the morning.', 'event': 'ADD', 'atom_id': <atm_v1>, 'version': 1, 'atom_type': 'state', 'scale_level': 'state', 'anchor_ids': [<anc_1>], ...}]}`.
- `r2['results'] == [{'id': <mem_coffee>, 'memory': 'User likes black coffee.', 'event': 'UPDATE', 'revises': <atm_v1>, 'version': 2, ...}]` (same lineage id).
- `hms.get(<mem_coffee>)['memory'] == 'User likes black coffee.'` and `['version'] == 2`; `hms.get(<mem_coffee>)['valid_to'] is None`.
- `[h['event'] for h in hms.history(<mem_coffee>)] == ['ADD', 'UPDATE']`; the UPDATE row has `old_memory == 'User likes oat-milk lattes in the morning.'`, `new_memory == 'User likes black coffee.'`, `anchor_id == <anc_2>`.
- `len(hms.get_all(filters=s1)['results']) == 1`; `len(hms.get_all(filters=s1, show_expired=True)['results']) == 2` and the archived row has `valid_to == '2026-04-20T08:00:00.000000+00:00'` (ingest-time REVISE uses the new event time, §12.5) and `hms.archived_reason == 'revised'`; the UPDATE history row has `updated_at == '2026-04-20T08:00:00.000000+00:00'`.
- `hms.search("what coffee does the user like", filters={"user_id": U})['results'][0]['memory'] == 'User likes black coffee.'` and no result equals the latte text; with `explain=True`: `score_details['route'] == 'preference'`, `['primary_axis'] == 'type'`, `['axis_gates']['obs_type'] == ['preference_event', 'habit']`.
- `hms.axis_query({'filters': {'user_id': U, 'topic': 'coffee'}, 'project_axis': 'value', 'scale_constraint': ['state'], 'aggregation': 'latest'})` (HMS §3.5 preference query) → `projection == [['black coffee']]`, `len(hits) == 1`; the same with `aggregation='count'` → `count == 1`.
- exactly one valid `revision` edge `<atm_v1> → <atm_v2>` with `witness_anchor_ids == [<anc_2>]`, `created_by == 'ingest'`.
- `ir = hms.compile("coffee", filters=s1)`: `[i.text for i in ir.preferences] == ['User likes black coffee.']`, `ir.facts == []`, `ir.audit` has no `revised` item (auto mode, no EPs); `hms.compile("coffee", filters=s1, mode="deep").audit` contains exactly one item with `explain['reason'] == 'revised'`, `text == 'User likes oat-milk lattes in the morning.'`, `explain['superseded_by_text'] == 'User likes black coffee.'`.

### §19.2 S2 temporal + S3 multi-session entity (`chat_multisession.json`)

```python
hms.add("Sarah joined the billing team in March.", timestamp="2026-03-02T09:00:00Z", user_id=U, run_id="run-a")
hms.add("Sarah shipped the webhook retry design on May 6.", timestamp="2026-05-06T09:00:00Z", user_id=U, run_id="run-b")
hms.add("Sarah is reviewing the checkout refactor this week.", timestamp="2026-06-11T09:00:00Z", user_id=U, run_id="run-c")
```
Rule extraction: #1 (R7) `temporal_fact`, subject `Sarah`, predicate `did`, objects `[]`, `date_observed 2026-03-01`, text verbatim, topics `["work", "billing"]`, entity_mentions `["Sarah"]` → atom_type `event`, `time_anchor == '2026-03-01T00:00:00.000000+00:00'`; #2 (R7) `temporal_fact`, `date_observed 2026-05-06`, topics `["work", "engineering"]`, `time_anchor == '2026-05-06T00:00:00.000000+00:00'`; #3 (R9) `fact` (`this week` is `relative_this`, ignored by R7), predicate `is`, topics `["engineering", "billing"]`, `time_anchor == '2026-06-11T09:00:00.000000+00:00'`. All three decisions ADD (different run scopes → no candidates).

Expect:
- `[r['memory'] for r in hms.search("what happened in May", filters={"user_id": U})['results']] == ['Sarah shipped the webhook retry design on May 6.']`; with `explain=True`: `score_details['axis_gates']['time'] == ['2026-05-01T00:00:00.000000+00:00', '2026-05-31T23:59:59.999999+00:00']`, `['primary_axis'] == 'time'`, `['route'] == 'temporal'`.
- `[r['memory'] for r in hms.get_all(filters={"user_id": U, "entity": "Sarah"})['results']] == ['Sarah is reviewing the checkout refactor this week.', 'Sarah shipped the webhook retry design on May 6.', 'Sarah joined the billing team in March.']` (time_anchor DESC).
- `hms.get_all(filters={"user_id": U, "run_id": "run-b", "entity": "sarah"})['results']` has exactly one row; one `entities` row with `normalized_text == 'sarah'` for `(alice, NULL)` and three `atom_entity` links (`store.entity_degree == 3`).

### §19.3 S4 — Work State Card (`structured_billing.json`)

```python
scope4 = {"user_id": U, "agent_id": "coder", "run_id": "run-billing"}
hms.add("Do not modify the subscription state machine.", timestamp="2026-06-15T09:00:00Z", **scope4)
events_fail = [
  {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:00:00Z",
   "payload": {"path": "src/billing/retry.ts", "patch": "+ backoff = 2 ** attempt", "summary": "add exponential backoff", "intent": "fix checkout webhook retry"}},
  {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:04:00Z",
   "payload": {"suite": "billing_e2e", "passed": 11, "failed": 3, "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}]}}]
events_fix = [
  {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:10:00Z",
   "payload": {"path": "src/billing/idempotency.ts", "patch": "+ idempotencyKey", "summary": "add idempotency key", "intent": "fix checkout webhook retry"}},
  {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:14:00Z",
   "payload": {"suite": "billing_e2e", "passed": 14, "failed": 0, "failures": []}}]
ra = hms.add_events(events_fail, **scope4)
card1 = hms.compile("fix checkout webhook retry", filters=scope4, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()
rb = hms.add_events(events_fix, **scope4)
card2 = hms.compile("fix checkout webhook retry", filters=scope4, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()
```
Constraint message (R2): `{"type": "constraint", "subject": "project", "predicate": "must_not", "objects": ["subscription state machine"], "relation": "constraint", "text": "Do not modify the subscription state machine.", "topic_tags": ["billing", "constraint"], "confidence": 0.9}` → atom `<atm_uc>` (memory `<mem_uc>`), anchor `<anc_uc>` (`source_ref == 'chat://agent_id=coder&run_id=run-billing&user_id=alice/0'`, `hash == '3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc'`).

`ra` expectations (§12.4): event atoms `file_diff src/billing/retry.ts: add exponential backoff` (`<atm_e1>`, anchor `<anc_e1>` `file://src/billing/retry.ts`) and `test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression` (`<atm_e2>`, `<anc_e2>` `test://billing_e2e`); EP1 `<ep1>`: `u='fix checkout webhook retry'`, `b='Do not modify the subscription state machine.'`, `a='patch src/billing/retry.ts: add exponential backoff'`, `y='billing_e2e failed: subscription_state_regression'`, `r='negative'`, `kappa={'constraints': ['Do not modify the subscription state machine.'], 'visibility': 'private', 'evidence': [<anc_e2>, <anc_e1>]}`, `anchor_ids == [<anc_e2>, <anc_e1>]`, `time_anchor == '2026-06-15T10:04:00.000000+00:00'`, episode atom text `[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression`, entity keys `{'retry.ts', 'billing_e2e', 'subscription_state_regression'}`; derived constraint `<atm_dc>`: text `Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression`, `obs_type='constraint'`, `subject='project'`, `predicate='must_not'`, `anchor_ids == [<anc_e2>, <anc_e1>]`, entity keys include `retry.ts`; edges: `use <ep1_atom> → <atm_uc>` (witness `[<anc_e1>]`, conf 0.8), `distill <ep1_atom> → <atm_dc>` (witness `[<anc_e2>]`, conf 0.9); `ra['eps'] == [<ep1>]`, `len(ra['edges']) == 2`, `ra['constraints'] == [<atm_dc>]`, `ra['events']` has 2 entries with `dedup False`; one `transitions` row `(ep, <ep1>)`.

`rb` expectations: event atoms `<atm_e3>` (`file://src/billing/idempotency.ts`), `<atm_e4>` (`test://billing_e2e`); EP2 `<ep2>`: `u='fix checkout webhook retry'`, `b='Do not modify the subscription state machine. ; Do not re-apply \'patch src/billing/retry.ts: add exponential backoff\': billing_e2e failed: subscription_state_regression'`, `a='patch src/billing/idempotency.ts: add idempotency key'`, `y='billing_e2e passed'`, `r='positive'`, `kappa.constraints` = both constraint texts (that order), `anchor_ids == [<anc_e4>, <anc_e3>]`; edges: `use <ep2_atom> → <atm_uc>` and `use <ep2_atom> → <atm_dc>` (witness `[<anc_e3>]`), `temporal <ep1_atom> → <ep2_atom>` (witness `[<anc_e4>]`, conf 0.5), `causal <ep1_atom> → <ep2_atom>` (witnesses `[<anc_e3>, <anc_e2>]`, conf 0.7); no derived constraint; `len(rb['edges']) == 4`; second `transitions` row `(ep, <ep2>)`.

`card1` (golden `work_state_card_failed.txt`, exact):
```
# Work State Card
Scope: user_id=alice agent_id=coder run_id=run-billing | As of: 2026-06-15T12:00:00.000000+00:00 | Mode: auto
## Current Goal
fix checkout webhook retry
## Valid Facts
- (none)
## Invalidated Assumptions
- (none)
## Failed Paths
- patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
## Constraints
- Do not modify the subscription state machine.
- Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
## Next Action
Do not re-apply 'patch src/billing/retry.ts: add exponential backoff'; investigate 'billing_e2e failed: subscription_state_regression' while keeping constraints: Do not modify the subscription state machine.
## Evidence
- chat://agent_id=coder&run_id=run-billing&user_id=alice/0 sha256:3ed3645c1244 @2026-06-15T09:00:00.000000+00:00
- file://src/billing/retry.ts sha256:a27b81718fae @2026-06-15T10:00:00.000000+00:00
- test://billing_e2e sha256:e51db27bfdf9 @2026-06-15T10:04:00.000000+00:00
```
Slots behind `card1`: `constraints == [<atm_uc>, <atm_dc>]`, `episodes == [<ep1_atom>]`, `facts == []` (the 2 event atoms `<atm_e1>`, `<atm_e2>` dropped as `covered_by_episode`; `retrieval['dropped_covered'] == 2`), `preferences == skills == []`, `audit` = 3 evidence items, `unresolved_conflicts == []`, `route == 'recall'`, `retrieval['graph_hops'] == 1`, `retrieval['ablation'] == 'axis_graph_support'` (auto resolved to deep: EPs exist).

`card2` (golden `work_state_card_fixed.txt`, exact):
```
# Work State Card
Scope: user_id=alice agent_id=coder run_id=run-billing | As of: 2026-06-15T12:00:00.000000+00:00 | Mode: auto
## Current Goal
fix checkout webhook retry
## Valid Facts
- patch src/billing/idempotency.ts: add idempotency key → billing_e2e passed [positive]
## Invalidated Assumptions
- (none)
## Failed Paths
- patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
## Constraints
- Do not modify the subscription state machine.
- Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
## Next Action
Continue from 'patch src/billing/idempotency.ts: add idempotency key' (last outcome: billing_e2e passed) while keeping constraints: Do not modify the subscription state machine.
## Evidence
- chat://agent_id=coder&run_id=run-billing&user_id=alice/0 sha256:3ed3645c1244 @2026-06-15T09:00:00.000000+00:00
- file://src/billing/retry.ts sha256:a27b81718fae @2026-06-15T10:00:00.000000+00:00
- test://billing_e2e sha256:e51db27bfdf9 @2026-06-15T10:04:00.000000+00:00
- file://src/billing/idempotency.ts sha256:cae8b909c617 @2026-06-15T10:10:00.000000+00:00
- test://billing_e2e sha256:ea5edcbd2c3f @2026-06-15T10:14:00.000000+00:00
```
Slots behind `card2`: `retrieval['dropped_covered'] == 4` (all four event atoms covered by `<ep1_atom>`/`<ep2_atom>`). `card2.to_pcir().ok() is True` and every obligation is `satisfied` except `governance == 'skipped'` (no audience). Ablation at this stage: `hms.compile(..., mode='deep').to_text().replace('Mode: deep', 'Mode: auto') == card2` and `hms.compile(..., mode='axis_graph').to_text().replace('Mode: axis_graph', 'Mode: auto') == card2` (no InterfaceEP exists yet, so Support adds nothing); `hms.compile(..., mode='fast')` has `retrieval['ablation'] == 'axis'`, `retrieval['graph_hops'] == 0` and touches no `graph_*` table.

### §19.4 S4b — search by identifier

`hits = hms.search("retry.ts", filters=scope4, top_k=3)['results']`: `{h['atom_id'] for h in hits} == {<atm_e1>, <ep1_atom>, <atm_dc>}`; every hit has `'retry.ts' in h['hms']['entity_ids']`; with `explain=True` each has `score_details['entity_boost'] == 1.0` (the entity `retry.ts` has degree 3 here — `<atm_e1>`, `<ep1_atom>`, `<atm_dc>` — and hub damping starts above 3, §13.4) and `'entity' in score_details['signals']`.

### §19.5 S5 — detective `why()` (question mode)

```python
trace = hms.why("why must retry.ts not be modified", filters=scope4)
```
Expect: `trace['target'] == {'kind': 'atom', 'id': <atm_dc>, 'node_id': <atm_dc>, 'text': "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'resolved_from': 'query'}`; `{n['node_id'] for n in trace['nodes']} == {<atm_dc>, <ep1_atom>, <ep2_atom>, <atm_uc>}`; `{(e['src_id'], e['dst_id'], e['edge_type']) for e in trace['edges']} == {(<ep1_atom>, <atm_dc>, 'distill'), (<ep2_atom>, <atm_dc>, 'use'), (<ep1_atom>, <atm_uc>, 'use'), (<ep1_atom>, <ep2_atom>, 'causal'), (<ep2_atom>, <atm_uc>, 'use')}`; every `witness_anchor_ids` ⊂ `{a['anchor_id'] for a in trace['anchors']}`; `trace['unresolved_conflicts'] == []`; `trace['refined_from'] == []`; `trace['chain']` == golden `why_chain.txt`:
```
Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
-use[file://src/billing/retry.ts]-> Do not modify the subscription state machine.
```
`trace['explanation'] == "\n".join(trace['chain'])`. `hms.why(<ep1>)` (by ep_id) resolves `resolved_from == 'id'`, `target['node_id'] == <ep1_atom>`; `hms.why("nonsense-id")` without filters → `ValueError` (`NOT_FOUND_001`).

### §19.6 S6 — checkpoint → feedback → rollback → transitions

```python
cp = hms.checkpoint(filters=scope4, label="before-feedback")
fb = hms.feedback(<mem_uc>, "state_outdated", correction="retry.ts may change once billing_e2e is isolated.")
mid = hms.compile("fix checkout webhook retry", filters=scope4)          # NO as_of: defaults to store.now(), i.e. one tick AFTER the feedback instant (the successor is valid only from that instant, §16.1)
rb = hms.rollback(cp)
tr = hms.transitions(filters=scope4)
```
Expect:
- `fb['actions'] == [{'action': 'archive', 'atom_id': <atm_uc>}, {'action': 'create', 'atom_id': <atm_uc2>}, {'action': 'revision_edge', 'edge_id': <edg_rev>}]`; `hms.get(<mem_uc>)['memory'] == 'retry.ts may change once billing_e2e is isolated.'`, `['version'] == 2`; the new atom's entity keys ⊇ `{'retry.ts', 'billing_e2e'}`, `obs_type == 'constraint'`, `anchor_ids[0] == <anc_fb>` (the feedback event's anchor).
- `mid`: `[c.text for c in mid.constraints] == ["Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'retry.ts may change once billing_e2e is isolated.']` — EXACTLY these two items (time_anchor ASC: the successor's `time_anchor` is the feedback instant; no `#kappa:` item is synthesized because both EPs' κ text `Do not modify the subscription state machine.` is present as an audit item, §15.1); `mid.audit` contains one `reason == 'revised'` item with `text == 'Do not modify the subscription state machine.'`; `mid.as_of == store.now()` at call time (a clock instant, so `mid.to_text()` is never compared to a golden).
- `rb == {'message': 'Rolled back to before-feedback', 'checkpoint_id': cp, 'restored': [<atm_uc>], 'archived': [<atm_uc2>], 'edges_invalidated': [<edg_rev>], 'edges_revalidated': [], 'edges_created': [<edg_rb>], 'anchor_id': <anc_rb>}`; `hms.get(<mem_uc>)['memory'] == 'Do not modify the subscription state machine.'`, `['version'] == 1`; `<edg_rb>` is `revision <atm_uc2> → <atm_uc>` with `label == f'rollback:{cp}'`, `witness_anchor_ids == [<anc_rb>]`, `created_by == 'ops'`.
- `[h['event'] for h in hms.history(<mem_uc>)] == ['ADD', 'FEEDBACK', 'UPDATE', 'ROLLBACK', 'RESTORE']`.
- `tr` (default `fmt='tuples'`) has exactly 5 six-tuples in this order (each `w_t` non-empty):
  1. `(s_t, 'patch src/billing/retry.ts: add exponential backoff', s_t1, 'negative', kappa, w_t)` — `s_t['u'] == 'fix checkout webhook retry'`, `s_t['ep_id'] == <ep1>`, `s_t1['y'] == 'billing_e2e failed: subscription_state_regression'`, `kappa['constraints'] == ['Do not modify the subscription state machine.']`, `w_t == [<anc_e2>, <anc_e1>]`
  2. `(…, 'patch src/billing/idempotency.ts: add idempotency key', …, 'positive', …, [<anc_e4>, <anc_e3>])` — `s_t['ep_id'] == <ep2>`
  3. causal: `(s_t with ep_id <ep1> and y == 'billing_e2e failed: subscription_state_regression', 'patch src/billing/idempotency.ts: add idempotency key', s_t1 with ep_id <ep2>, 'positive', kappa, w_t == [<anc_e3>, <anc_e2>, <anc_e4>])` — `w_t = ordered_union(edge witnesses [<anc_e3>, <anc_e2>], EP2 anchors [<anc_e4>, <anc_e3>])`, first-occurrence order (§16.4); `kappa['constraints'] == [both constraint texts]` (EP2's κ), `kappa['provenance']['edge_id'] == <edg_causal>`
  4. revision: `({'text': 'Do not modify the subscription state machine.', 'atom_id': <atm_uc>, ...}, 'revise', {'text': 'retry.ts may change once billing_e2e is isolated.', 'atom_id': <atm_uc2>, ...}, 'negative', kappa, [<anc_fb>, <anc_uc>])` — `kappa == {'constraints': [<atm_dc text>, 'retry.ts may change once billing_e2e is isolated.'], 'visibility': 'private', 'permission_scope': 'private', 'provenance': {'label': '', 'created_by': 'feedback', 'proposer': 'rule'}}` (constraints valid at `time_t1` = the feedback instant: `<atm_uc>` is archived at that instant, `<atm_uc2>` valid from it)
  5. rollback: `({'text': 'retry.ts may change once billing_e2e is isolated.', 'atom_id': <atm_uc2>, ...}, 'rollback', {'text': 'Do not modify the subscription state machine.', 'atom_id': <atm_uc>, ...}, 'neutral', kappa, [<anc_rb>])` — `kappa == {'constraints': ['Do not modify the subscription state machine.', <atm_dc text>], 'visibility': 'private', 'permission_scope': 'private', 'provenance': {'checkpoint_id': cp, 'label': 'before-feedback', 'reason': None}}` (constraints valid at the rollback instant: `<atm_uc>` restored, `<atm_uc2>` archived)
  Every row's `kappa_t` has exactly the keys `constraints, visibility, permission_scope, provenance`; rows 1–2 carry `provenance == {'evidence': [...], 'ep_id': ...}`.
- `hms.transitions(filters=scope4, fmt='jsonl')` equals golden `transitions_s6.jsonl` after `mask_ids` + timestamp masking; calling it again adds no `transitions` rows (`store.stats()['transitions'] == 5`).

### §19.7 Epilogue — consolidate + graph_stats

```python
rep = hms.consolidate(filters=scope4)
st = hms.graph_stats(filters=scope4)
```
Expect `rep['phases']['consolidate'] == {'entity_atoms': 6, 'temporal_edges': 0, 'causal_edges': 0, 'conflicts': 0, 'interface_eps': 1, 'projected_edges': 3, 'interface_eps_l3': 0, 'skills': 1, 'transitions': 1, 'compiled_states': 1}` — `entity_atoms`: the six entities linked to live scope4 atoms (`retry.ts`, `file_diff`, `billing_e2e`, `test_log`, `subscription_state_regression`, `idempotency.ts`; keys of §9.2 over the event/EP/constraint atom texts) each get an `entity` atom at scope `(alice, coder, run_id=None)`; `temporal_edges`/`causal_edges` are 0 because `insert_edge` MERGEd into the existing edges (both now have `mentions == 2`); `transitions == 1` is the new `interface_ep` row; `compiled_states == 1` (`store.last_compiled_state(scope4)['card'] == hms.compile('fix checkout webhook retry', filters=scope4, mode='deep').to_text()`). The InterfaceEP: `support_node_ids == [<ep1_atom>, <ep2_atom>]`, `cohesion_score == pytest.approx(2/3, abs=1e-4)`, `slots.u == 'fix checkout webhook retry'`, `slots.a == 'patch src/billing/retry.ts: add exponential backoff → patch src/billing/idempotency.ts: add idempotency key'`, `slots.b == 'Do not modify the subscription state machine. ; ' + <atm_dc text>` (each constraint once), `slots.y == 'billing_e2e passed'`, `slots.r == 'positive'`, `slots.kappa.evidence == evidence_anchor_ids`, `invariants` = `[constraint 'Do not modify the subscription state machine.' (support [<ep1_atom>, <ep2_atom>], anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>}), confidence 0.8), failed_path "do not: patch src/billing/retry.ts: add exponential backoff" (support [<ep1_atom>], anchor_ids == sorted({<anc_e1>, <anc_e2>}))]`, `residuals == []`, `evidence_anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`; the IEP atom has `source_ep_ids == [<ep1>, <ep2>]`; high edges: `use <iep_atom> → <atm_uc>` (2 projection rows), `use <iep_atom> → <atm_dc>` (1), `distill <iep_atom> → <atm_dc>` (1); skill atom text `'Skill: Do not modify the subscription state machine.'` (`scale_level == 'L3_macro'`, `anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`, `confidence == 0.8`) with `distill <iep_atom> → <skill>` at `L3_macro`, `is_boundary == False`, `witness_edge_id == <edg_use_ep1_uc>` and `store.projections_of(<edg_skill>) == [<edg_use_ep1_uc>, <edg_use_ep2_uc>]`; the `interface_ep` transition row: `a_t == slots.a`, `r_t == 'positive'`, `s_t1['invariants'] == [both invariant texts]`, `w_t == evidence_anchor_ids`, `time_t == '2026-06-15T10:04:00.000000+00:00'`, `time_t1 == '2026-06-15T10:14:00.000000+00:00'`.
`st == {'nodes': {'message': 0, 'observation': 3, 'event': 4, 'entity': 0, 'ep': 2, 'interface_ep': 1, 'skill': 1}, 'edges': {'temporal': 1, 'causal': 1, 'use': 5, 'distill': 3, 'revision': 2, 'conflict': 0}, 'valid_edges': 11, 'invalid_edges': 1, 'boundary_edges': 3, 'interface_eps': 1, 'unresolved_conflicts': 0, 'witness_coverage': 1.0, 'projection_coverage': 1.0, 'traceable_high_edges': 1.0, 'transitions': 6}` (`nodes.entity == 0` because entity atoms live at `(alice, coder)` without `run_id` and `scope4` filters on `run_id`; `hms.graph_stats(filters={'user_id': U, 'agent_id': 'coder'})['nodes']['entity'] == 6`; `traceable_high_edges`: the 3 projected L2 edges and the L3 skill distill edge all have projection rows).
After consolidate: `hms.compile('fix checkout webhook retry', filters=scope4, mode='deep')` has `[s.text for s in .skills] == ['Skill: Do not modify the subscription state machine.']`, `retrieval['ablation'] == 'axis_graph_support'`, and ≥1 audit item whose `explain['reason'] == 'invariant:failed_path'`; `mode='axis_graph'` yields the same `skills` but NO item whose `item_id` contains `#inv:` and no audit item with reason starting `invariant:`/`residual:` (`retrieval['ablation'] == 'axis_graph'`); `mode='fast'` also yields the same `skills` (the skill atom is `L3_macro`, which is in the recall `scale_default`, so the axis step alone finds it) with `retrieval['ablation'] == 'axis'` and no `#inv:`/`invariant:` items — only the Support layer (`refine`) distinguishes `deep` from the other two modes.

### §19.8 Full payload hashes (for the fixtures above; §0.1 rule)

| anchor | hash object (canonical) | sha256 |
|---|---|---|
| `<anc_uc>` | `{"actor_id":null,"object_type":"message","payload":{"dropped_parts":0,"text":"Do not modify the subscription state machine."},"role":"user"}` | `3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc` |
| `<anc_e1>` | `{"actor_id":"coder","object_type":"file_diff","payload":{"commit":null,"constraints":[],"intent":"fix checkout webhook retry","patch":"+ backoff = 2 ** attempt","path":"src/billing/retry.ts","summary":"add exponential backoff"},"role":"tool"}` | `a27b81718fae86837895813746ec188afca9087544df8950c3b5832742ebfa8d` |
| `<anc_e2>` | `{"actor_id":"coder","object_type":"test_log","payload":{"failed":3,"failures":[{"message":"state machine changed","name":"subscription_state_regression"}],"log":null,"passed":11,"related_paths":[],"status":"failed","suite":"billing_e2e"},"role":"tool"}` | `e51db27bfdf98d59d0abccc1fb59f7413e1f9bbaf1260406fdc2154e4b7ce1e2` |
| `<anc_e3>` | `{"actor_id":"coder","object_type":"file_diff","payload":{"commit":null,"constraints":[],"intent":"fix checkout webhook retry","patch":"+ idempotencyKey","path":"src/billing/idempotency.ts","summary":"add idempotency key"},"role":"tool"}` | `cae8b909c617c029c16e12f5ef04e0906be8a797b07eca94132db1654344fb2a` |
| `<anc_e4>` | `{"actor_id":"coder","object_type":"test_log","payload":{"failed":0,"failures":[],"log":null,"passed":14,"related_paths":[],"status":"passed","suite":"billing_e2e"},"role":"tool"}` | `ea5edcbd2c3f0b79ef01e3768590081cf1a024f8081b4c2b2d17155dd68ff272` |

---

## §20 Docs and examples (group X)

### §20.1 `pyproject.toml` (essentials, exact)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hms-elite"
version = "0.1.0"
description = "HMS Elite: Mem0-compatible memory API over an experience-to-state dual-plane store (State Index + Support Graph)."
readme = "README.md"
requires-python = ">=3.10,<4.0"
license = {text = "Apache-2.0"}
dependencies = ["pydantic>=2.7"]

[project.optional-dependencies]
numpy = ["numpy>=1.24"]
openai = ["openai>=1.90"]
anthropic = ["anthropic>=0.40"]
server = ["fastapi>=0.110", "uvicorn>=0.30"]
test = ["pytest>=8", "pytest-mock>=3"]
dev = ["ruff==0.16.0", "isort>=5"]

[project.scripts]
hms = "hms_elite.cli:main"

[tool.hatch.build.targets.wheel]
only-include = ["hms_elite"]

[tool.pytest.ini_options]
pythonpath = ["."]
addopts = "-q"

[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]

[tool.isort]
profile = "black"
line_length = 120
known_first_party = ["hms_elite"]
```

`tests/conftest.py` registers `--update-golden` via `pytest_addoption`.

### §20.2 `README.md` outline

1. Install (`pip install hms-elite`; extras). 2. 10-line quickstart (Mem0-shaped `add/search`, then `compile(...).to_text()`, `why`, `feedback`, `transitions`). 3. "Axes find, graph inherits, ContextIR acts" (three paragraphs, one diagram). 4. Migrating from Mem0: table — filters identical; `update` ⇒ revision (same id, new version); `history` `+anchor_id` and extra events; config dict keys accepted/renamed/removed; `vector_store` providers `sqlite|memory`; feedback enum. 5. Offline determinism (`rule` LLM, `hash` embedder) and switching to `openai`/`anthropic` (one config dict). 6. Invariants (four bullets) and how they are enforced (type / DDL / store). 7. API reference table (Mem0 + HMS-native, incl. `context()` with its memories-first snippet, README §5.13). 8. Bilingual note (EN/ZH docstrings). All examples runnable offline.

### §20.3 `examples/quickstart.py` (≈45 lines, offline; output byte-identical to §19 under the frozen clock is NOT required — it runs on the real clock and prints the same card texts because cards use explicit `as_of`)

```python
"""HMS Elite quickstart: six offline scenarios — recall, revision, temporal, multi-session, work state, detective, rollback.
HMS Elite 快速上手：六个离线场景——召回、修订、时间、多会话、工作状态、侦探、回滚。"""
import json
from hms_elite import HMS

hms = HMS({"store": {"path": ":memory:"}})          # rule LLM + hash embedder + sqlite index; no API key
U = "alice"

# S1 preference → revision (same Mem0 id, new version)
s1 = {"user_id": U, "run_id": "run-coffee"}
first = hms.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", **s1)["results"][0]
hms.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", **s1)
print("S1:", hms.get(first["id"])["memory"], "| history:", [h["event"] for h in hms.history(first["id"])])

# S2 temporal + S3 entity across sessions
hms.add("Sarah joined the billing team in March.", timestamp="2026-03-02T09:00:00Z", user_id=U, run_id="run-a")
hms.add("Sarah shipped the webhook retry design on May 6.", timestamp="2026-05-06T09:00:00Z", user_id=U, run_id="run-b")
hms.add("Sarah is reviewing the checkout refactor this week.", timestamp="2026-06-11T09:00:00Z", user_id=U, run_id="run-c")
print("S2:", [r["memory"] for r in hms.search("what happened in May", filters={"user_id": U})["results"]])
print("S3:", [r["memory"] for r in hms.get_all(filters={"user_id": U, "entity": "Sarah"})["results"]])

# S4 constraint + structured agent events (no LLM) → Work State Card
scope = {"user_id": U, "agent_id": "coder", "run_id": "run-billing"}
hms.add("Do not modify the subscription state machine.", timestamp="2026-06-15T09:00:00Z", **scope)
hms.add_events([
    {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:00:00Z",
     "payload": {"path": "src/billing/retry.ts", "patch": "+ backoff = 2 ** attempt", "summary": "add exponential backoff", "intent": "fix checkout webhook retry"}},
    {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:04:00Z",
     "payload": {"suite": "billing_e2e", "passed": 11, "failed": 3, "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}]}},
], **scope)
print(hms.compile("fix checkout webhook retry", filters=scope, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text())

# S5 detective
print("S5:", hms.why("why must retry.ts not be modified", filters=scope)["explanation"])

# S6 checkpoint → feedback → rollback → transitions
constraint_id = hms.get_all(filters={**scope, "obs_type": "constraint"})["results"][-1]["id"]
cp = hms.checkpoint(filters=scope, label="before-feedback")
print("S6 feedback:", [a["action"] for a in hms.feedback(constraint_id, "state_outdated", correction="retry.ts may change once billing_e2e is isolated.")["actions"]])
print("S6 rollback:", hms.rollback(cp)["restored"])
for s_t, a_t, s_t1, r_t, kappa_t, w_t in hms.transitions(filters=scope):
    print("S6 transition:", a_t, "->", r_t, "| witnesses:", len(w_t))

print(json.dumps(hms.consolidate(filters=scope)["phases"]["consolidate"]), hms.graph_stats(filters=scope)["edges"])
hms.close()
```

Expected stdout (abridged): `S1: User likes black coffee. | history: ['ADD', 'UPDATE']`; `S2: ['Sarah shipped the webhook retry design on May 6.']`; `S3:` the three Sarah facts newest first; the `work_state_card_failed.txt` card verbatim; `S5:` the three-line chain of §19.5; `S6 feedback: ['archive', 'create', 'revision_edge']`; `S6 rollback: [<atom id>]`; transition lines `patch src/billing/retry.ts: add exponential backoff -> negative | witnesses: 2`, `revise -> negative | witnesses: 2`, `rollback -> neutral | witnesses: 1`; consolidate report `{"entity_atoms": 5, "temporal_edges": 0, "causal_edges": 0, "conflicts": 0, "interface_eps": 0, "projected_edges": 0, "interface_eps_l3": 0, "skills": 0, "transitions": 0, "compiled_states": 1}` (one EP only → no coarsen; five entities `retry.ts, file_diff, billing_e2e, test_log, subscription_state_regression`) and the edge counts.

---

## Change log (post-review)

Every defect raised by the two review passes (the list is de-duplicated below; duplicates are noted) and how this revision resolves it. Section numbers refer to this document after the edit. No defect was resolved by shortening the contract.

| # | Severity | Where | Resolution |
|---|---|---|---|
| 1 | blocker (raised twice) | §19.6 `mid` compile with `as_of="2026-06-15T12:00:00Z"` vs `state_outdated` successor `valid_from=now` vs §8 validity WHERE | `mid` is compiled WITHOUT `as_of` (defaults to `store.now()`, one tick after the feedback instant). §8 validity semantics unchanged. §16.1 `state_outdated` row now notes that the successor is visible only for `as_of >= feedback instant`; §19 preamble and test 53 say the `mid` card is asserted by slot content, never against a golden. DESIGN §12 updated. |
| 2 | blocker | §16.5 consolidate `kappa_items` giving 3–4 `mid.constraints` (undeduplicated `#kappa:` items for an archived κ text) | §15.1 `kappa_items`: `selected_constraint_texts` = normalized texts of selected constraints ∪ audit items; κ items deduplicated by normalized text across episodes (first episode wins). §15.2 step 6 builds that set. §19.6 states `mid.constraints` has EXACTLY the two items. |
| 3 | blocker | §11.1 batch dedup key `(type, predicate, tuple(objects))` collapsing S2's two temporal facts | Dedup key is `(type, predicate, tuple(objects), normalize(text))`. |
| 4 | blocker (raised twice) | §9.2 `ACTION_RE` target `[^,.;!?]*` cutting `retry.ts` → `a='patched retry'` | `ACTION_RE` redefined with a named `verb` group and a lazy target terminated by `CLAUSE_CUT` (conjunction / clause separator / sentence-final dot / end); new `CAUSE_RE` (trigger clause after the conjunction) and `CLAUSE_SPLIT_RE` (outcome clause). §11.2 rewritten with the worked example `a='patched retry.ts to add backoff'`, `u='the webhook kept timing out'`, `y='billing_e2e failed'`; test 19 asserts it. |
| 5 | blocker (raised twice) | §11.1 assistant-role cue regex admitting only recommendation/scheduling sentences, so `chat_action_closure` yields no observation | Assistant gate = `ASSISTANT_CUE_RE` OR `ACTION_RE`; such sentences become R9 facts with `subject='assistant'`, `attributed_to='assistant'`, `Assistant …` text. Test 26 ("both an observation and an EP") stands; fixture role stays `assistant`. |
| 6 | blocker | §12.6/§16.4 `ep_transition` defined in group G but called by group C | `Transition.from_ep(...)` (and `from_interface_ep`) are classmethods in `hms_elite/models.py` (group A); `ops/transitions.py` re-exports `ep_transition = Transition.from_ep`. §0.4, §12.6, §16.4 updated. |
| 7 | blocker | §4/§4.3 `validate_payload` owned by C but required by A's `SourceEvent` validator | `validate_payload` is declared in `hms_elite/models.py` (group A) as the single implementation; `ingest/events.py` re-exports it; the `(ingest/events.py::validate_payload …)` attribution in §4.3 removed. |
| 8 | blocker + major (raised twice) | §0.4 vs §12.6 step 7 `atom_to_row` and §13.3 `axis_ids` (C importing D) | `MemoryAtom.to_public_dict(...)` is the canonical implementation in models.py (A); `mvmi/query.py::atom_to_row` is a one-line wrapper (no lazy import); `axis_ids` deleted from §13.3; §12.3 `neighbourhood` uses `store.valid_ids(scope)['atoms']` (§8). §0.4 states C never imports mvmi/ops and G imports B's rule functions. |
| 9 | blocker + major (raised twice) | §12.4 `store.last_open_action(scope, run)` undeclared in §8; DESIGN called it `last_open_action_event` | §8 declares `last_open_action(self, scope, *, run_id, before, max_gap_hours) -> tuple[SourceEvent, Anchor] | None` with the exact SQL semantics; §12.4 calls it with keywords; DESIGN §6.2 uses the same name. |
| 10 | blocker | §18.1 `golden` masking every 32-char timestamp vs byte-for-byte cards | Two helpers: `golden.text(name, text)` (byte-for-byte, no masking — cards, chain) and `golden.masked(name, obj)` (`mask_ids` + masking only of instants ≥ FrozenClock start). Tests 47/48/55/63 name the helper they use. DESIGN §11 updated. |
| 11 | blocker | §18.3 test 60 constructing `HMS` from the stock Mem0 dict (openai providers, qdrant config) offline | Test 60 substitutes `llm=rule`, `embedder=hash`, `vector_store=sqlite/{}` (keeps `history_db_path`, `version`, `reranker`); the unmodified fixture is used only by test 10 (`HMSConfig.from_dict` → `CONFIG_002`). |
| 12 | major (raised twice) | §4 `GraphEdge.witness_anchor_ids: NonEmptyIds` making the store-level `WitnessMissingError` unreachable (test 12 vs test 3) | `SupportGraph.insert_edge` raises `WitnessMissingError` on `[]` BEFORE building the model; `StateStore.insert_edge(edge: GraphEdge | dict)` checks a dict before `GraphEdge(**edge)` and re-checks model instances (incl. `model_construct`); test 12 exercises all three paths with `store.trace.tables() == set()`. Test 3 (pydantic rejection of `[]`) still holds. DESIGN §1 commitment 1 reworded. |
| 13 | major | §5 `CONFIG_002`/`CONFIG_003` raised inside pydantic validators (converted to `CONFIG_001`) | `HMSConfig.from_dict` runs plain-Python pre-checks `normalize_mem0_keys(strict=True)` (→ `CONFIG_003`) and `check_providers` (→ `CONFIG_002`) BEFORE `HMSConfig(**data)`, and re-raises any `HMSError` found in `e.errors()[i]['ctx']['error']` unchanged; the before-validator only remaps RENAMED keys; `_known_provider` raises a plain `ValueError` for direct construction. §5.1/§5.2/§5.3 updated; test 10 notes the pre-check. |
| 14 | major + minor (raised three times) | `Invariant.anchor_ids`/`confidence` and `Residual` anchors never derived by any producer; `kappa.evidence` empty; IEP atom `source_ep_ids`/`topic_tags` unspecified | §14.2 coarsen: `support_node_ids` from refs, `anchor_ids = sorted(∪ atom_anchor_ids)`, `confidence = round(mean, 4)`; `Residual.anchor_ids` field added (§4); `slots.kappa.evidence = evidence_anchor_ids`; IEP atom `source_ep_ids`, `topic_tags` (ordered union), `entity_ids` (ordered union) defined. §19.7 asserts the skill atom's `anchor_ids`/`confidence`; tests 41/42/43 extended. `atom_from_ep` now carries `confidence=ep.confidence` so the numbers are determinate. |
| 15 | major (raised twice) | §19.4 `entity_boost == 1.0` vs hub damping at degree 3 | Damping formula changed to `1/(1 + 0.001·max(0, deg−3)²)` (§13.4, DESIGN §8): degree ≤ 3 scores exactly 1.0, so the §19.4 assertion is exact; test 36 asserts `< 1.0` at degree 5. |
| 16 | major (raised twice) | §11.1 `supersedes_hint_ids` (≥2 shared tokens) unreachable for S1 (`["0"]`) | R1 hint rule: existing text starts with `User likes|prefers|dislikes` AND (≥1 shared content token OR ≥1 shared `TOPIC_TABLE` topic); other rules keep ≥2 tokens. S1 shares `like` and topic `coffee`. |
| 17 | major | §6.2 edge prompt `## Nodes` lacking `b`, `subject`, `predicate`, `objects` used by §11.4 | `## Nodes` payload is `{ref, kind, u, b, a, y, r, subject, predicate, objects, time, entities, topics, anchor_refs}` (every key always present; EP vs observation shapes defined); `EDGE_PROPOSAL_PROMPT` `# INPUTS` mirrors it; §11.4 states the node shape. |
| 18 | major + minor (raised twice) | §12.5 `link_atom_axes` 3-tuple `(normalized, canonical, entity_type)` — FILE identity ambiguity, no alias slot; readers not hydrating `entity_ids`/`topic_tags` | `EntityKey = (key, canonical, entity_type, aliases)`; `insert_atom(entity_keys: list[EntityKey])` passes `normalized=key, aliases=aliases`; §9.2 states `EntityCandidate.key` is the identity; §8 "Hydration" note: every reader fills `entity_ids` (keys), `topic_tags`, `anchor_ids`. |
| 19 | major | §15.1 rule (1) routing live atoms with a future `expiration_date` to `audit` | Rule (1) is `archived_reason is not None or (valid_to is not None and valid_to <= as_of)`; `assign_slot(atom, *, as_of, reason=None)`; §15.2 passes `as_of`; test 46 covers both cases. Related: `resolve_head`/`ux_atoms_head` keyed on `archived_reason IS NULL` so `get()` works for expiring memories. |
| 20 | major | §18.3 test 27 needing a background constraint without an LLM call | Test 27 seeds the constraint via `ingest.events.anchor_events` + `ingest.atoms.create_atom(atom_from_observation(Observation(type='constraint', …)))` and asserts `hms_scripted.llm.calls == []` only around `add_events`. |
| 21 | major (raised twice) | §18.3 test 24 / DESIGN §3.2 "identical re-add gains a support atom_anchor row" (PK collision) | `add_atom_anchor` is `INSERT OR IGNORE` returning bool (§8); test 24 asserts unchanged `atom_anchor_ids` and `mentions == 2` for the same-timestamp re-add, and the support row only for the later-timestamp re-add; DESIGN §3.2 row rewritten; §12.6 step 6 notes the no-op. |
| 22 | minor (raised three times) | §19.3 card1 "4 event atoms dropped" | card1: 2 (`dropped_covered == 2`); card2: `dropped_covered == 4`; tests 47/48 assert them. |
| 23 | major + minor (raised twice) | §16.5 steps (1)/(2) "lacking an edge" vs §19.7 `mentions == 2` | Steps (1)/(2) always call `insert_edge` (MERGE bumps `mentions`); counts = edge ids that did not exist before the call. §19.7 and test 56 spell out `temporal_edges == causal_edges == 0`, `mentions == 2`. |
| 24 | minor | §4 `RESERVED_ATOM_KEYS` missing `source_ep_ids_json` | Set derived as model fields ∪ `{f + '_json' for f in ('content','user_meta','source_ep_ids')}` ∪ the explicit extras; test 39 asserts ⊇ `PRAGMA table_info` names. |
| 25 | minor | §13.3 step 1 `parse_filters(aq.meta_filters)` with default `require_scope=True` and tuple misuse | `_, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context='meta', force_meta=True)`; `force_meta` kwarg added to §13.1. |
| 26 | minor + major (raised three times) | `observation_date`/`current_date` undefined; ingest REVISE `at` undefined (S1 expects event-time `valid_to`) | §12.3/§12.6: `observation_date = date_of(events[-1].timestamp)`, `current_date = date_of(store.now())`, per-message `timestamp` is the message's `base_date` (§11.1); §12.5 `revise_atom`: ingest passes `at = max(new.valid_from, old.valid_from)` (event time) for `valid_to` and history `updated_at`; `update()`/feedback pass `store.now()`. §19.1 asserts the UPDATE row's `updated_at`. |
| 27 | minor (raised twice) | Keyword-only signatures invoked positionally; `build_state_package` without `config`; `history_event=None`; `StateStore(...)` positional | §12.4 `store.last_ep(scope, run_id=run)`; §16.3 `build_state_package(..., config: HMSConfig, ...)` with a keyword `compiler.compile(...)` call; `create_atom(history_event: str | None)`; §17.1 `StateStore(path, busy_timeout_ms=…, journal_mode=…)`. |
| 28 | minor | §16.1 `connection_reason_wrong` with the same edge type MERGEing into the invalidated row | Same type → no invalidation: `revalidate_edge` + `add_edge_witness(fb_anchor)` + `set_edge_label` (new §8 method), reported as `{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}`; a different type creates a fresh row. Test 52 sub-case added. |
| 29 | minor + major (raised twice) | `state_missing` without a target unreachable; §4.2 feedback row forbidding `preference_event` atoms | `process(*, target_id: str | None, …)`, `HMS.feedback(target_id: str | None, …)`; prologue: for `state_missing` an unresolvable/None target is legal — scope from `correction['scope']`, a `Scope.key()` string, or `scope_hint`; `FeedbackEvent.target_kind` gains `anchor`/`scope` (DDL CHECK updated); §4.2 feedback row allows `obs_type ∈ {constraint, fact, preference_event}`. Test 52 sub-case `state_missing_no_target`. |
| 30 | minor | `compiler.default_mode` never read | `HMS.compile(mode: str | None = None)` and `StateCompiler.compile(mode=None)`; §15.2 step 1 resolves `mode = mode or config.compiler.default_mode`; test 46/57 cover it. |
| 31 | minor (raised twice) | Skill atoms (`L3_macro`) excluded by recall/temporal `scale_default`; augmentation ignoring skills | `L3_macro` added to the recall/temporal/entity `scale_default`; §15.2 step 3 unions valid `source_kind='skill'` atoms; §19.7 asserts `compile(...).skills` after consolidate. |
| 32 | minor (raised twice) | Test 56 prune with no atom below `confidence_min` | Test 56 sets `store.set_atom_confidence(<atm_e1>, 0.2)` before `clock.tick(days=400)`; expected `archived == [<atm_e1>]`, `decayed == 3` (with the new decay rule, #46). |
| 33 | minor + major (raised twice) | `close_structured_events` lacking `temporal_window_hours`; payload `constraints` never reaching `b`/`kappa.constraints` | Signature gains `temporal_window_hours` (pipeline passes `config.graph.temporal_window_hours`); `build_ep_from_closure` uses `constraints = ordered_union(background texts, payload constraints)` for `b` and `kappa.constraints`, so self-closing tool_calls satisfy the `Slots` validator. Test 27 asserts it. |
| 34 | major | `AxisQuery` semantics (project_axis / aggregation) not executed; `scope` required | New `AxisResult(hits, projection, count, summary, …)` model (§4); §13.3 step 5 defines every `project_axis` and `aggregation` (`count` = one SQL COUNT, `latest`, `summary`, `topk` needs `text` → `VALIDATION_013`); `AxisQuery.scope` optional (derived from filters); `HMS.axis_query()` facade; test 35 extended; §19.1 HMS §3.5 preference example. |
| 35 | major | `entity`/`relation` scales dead (no entity atoms; relation atoms off by default and out of scale defaults) | `SourceKind.entity`/`NodeKind.entity`; §4.2 `entity` content row; `atom_from_entity` builder (§12.5); consolidate step (0) `entity_atoms` materializes one atom per entity at the entity identity scope, revised when mentions/aliases change; `emit_relation_atoms` default True; `entity`/`relation` added to recall/entity `scale_default`; `graph_stats.nodes.entity`; test 56c; §19.7 expects `entity_atoms == 6` and explains the scope. Entity atoms never enter cards (§15.2 step 2). |
| 36 | major | `PREF_QUERY_RE → recall` no-op (preference recall not an axis selection) | `Route.preference` with `hard_type_gate = ('obs_type', ['preference_event','habit'])`, `scale_default=['state']`; `execute_axis_query` step 2 applies it with the usual fallback; `primary_axis='type'`; `score_details.axis_gates.obs_type`; tests 34/36; §19.1 asserts the route. |
| 37 | major | EP validity never maintained (`archive_ep` uncalled; no `ep_state` in checkpoints; `list_eps` resurrecting archived EPs) | §8 `archive_atom`/`restore_atom` cascade to `eps.valid_to` (and `observations.valid_to`, `interface_eps.valid`); `restore_ep`, `scope_ep_state`, `set_observation_valid_to` added; `list_eps(valid_only)` = `eps.valid_to IS NULL AND node valid`; `Checkpoint.ep_state` + snapshot hash + rollback step (2b); test 53 covers delete → handoff goal → rollback. |
| 38 | major | No revoke path (`anchors.revoked_at` never set; no cascade) | `revoke(store, graph, *, anchor_id, reason)` in §16.1 with a witnessed cascade (dispute → archive when unwitnessed → invalidate edges/IEPs), `HMS.revoke()`, `feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` delegating to it, `system` payload `op='revoke'`; test 56b and test 51 (`not_revoked`) use it. |
| 39 | major | InterfaceEPs never yielding transitions | `TransitionKind.interface_ep`, DDL CHECK, `KIND_ORDER` (now in enums, group A — also resolves the "KIND_ORDER declared in G" minor); `Transition.from_interface_ep`; `export_transitions` step (a); §19.7 `transitions == 1`, `st['transitions'] == 6`; test 55 checks the row. |
| 40 | major | L3 InterfaceEPs unreachable (`P = nodes with kind ep`; scheduler coarsens EP components only) | §11.5 `P` = kinds `{ep, interface_ep}`; `components(..., scale=)` parameter; consolidate step (4b) over L2 IEP components → `coarsen(target_scale='L3_macro')` → `project`; `GraphConfig.coarsen_l3`; report key `interface_eps_l3`; test 45b. DESIGN §14 risk 3 updated. |
| 41 | major | Governance checked only after selection (team handoff unusable; `visibility` filter lowered to equality) | §13.1 lattice gate for a scalar `visibility` filter; §15.2 keeps the key in the axis filters and gates graph neighbours, invariants and evidence anchors (`dropped_governance`); §15.4 `governance` covers κ omission and anchor `permission_scope`; test 51 governance case changed (private item not selected, obligation satisfied; hand-built IR violates). DESIGN §6.3/§6.4/§14 updated. |
| 42 | major | Handoff package not importable; no `source_events` | `StatePackage.source_events` and `atoms` (§4, validator checks the evidence closure); `import_state_package(...)` (§16.3) with id preservation, identity dedup, optional re-scoping; `HMS.import_package()`; `system` payload `op='import'`/`package_id`; test 54b round-trip (card equality on the receiver). |
| 43 | minor | `sensor_trace` object type dropped; `EventType.tool_result` never assigned | `ObjectType.sensor_trace` with normalized payload (§4.3), structured route producing an `event` atom only, `sensor://` raw_ref, `event_atom_text`; `EXTRACTABLE_OBJECT_TYPES` has five members (DESIGN §2 row 3 now lists them); `events_from_dicts` sets `event_type='tool_result'` for `test_log`, `sensor_trace`, and `tool_call` with `status`/`result`. |
| 44 | minor | §0.3 docstring template collapsing HMS §5.2's two-line-per-argument layout; tests/examples exempt | §0.3 rewritten to HMS §5.2 verbatim (EN line + ZH line per argument with full-width colon, blank line between arguments, EN+ZH under Output/State effect); applies to `tests/**/*.py` and `examples/*.py` (§1); test 64 `test_bilingual_docstrings`. |
| 45 | minor | Scheduler task `compile_state` absent | Consolidate step (7): `compiler.compile(mode='deep')` per scope persisted to the new `compiled_states` table (DDL, `TABLES` now 22, id prefix `cst`) + `meta['work_state:<scope>']`; `orient` reports `last_compiled`; `SchedulerConfig.compile_state`; `StateScheduler.__init__` takes `compiler`; report key `compiled_states`; `compile()` itself stays read-only. |
| 46 | minor | Prune collapsing decay and archival | Prune split into (a) decay (`confidence − decay_rate·staleness`, floor `confidence_min/2`, `UPDATE` history) and (b) archive below `confidence_min`; `SchedulerConfig.decay_rate = 0.1`; report gains `decayed`. The subtractive step (rather than the suggested `×(1−0.5·staleness)`) keeps S4's 0.7-confidence event atoms above the archive line so test 56's `archived == [<atm_e1>]` stays exact. DESIGN §6.9/§8 updated. |
| 47 | minor | `kappa_t` meaning differing per transition kind | `kappa_t_of(...)` (§4) gives every kind the shape `{constraints, visibility, permission_scope, provenance}`; §16.2/§16.4 define the per-kind contents; §19.6 rows 3–5 restated (`provenance` carries the old keys); test 55 asserts the key set. |
| 48 | minor | Skill `distill` edge not traceable to low-level `use` edges | `promote_skills` sets `witness_edge_id` to the lowest-id valid `use` edge support_node → constraint atom and `add_projection` for the rest; `graph_stats.traceable_high_edges`; §19.7 and test 43 assert the projection rows. |
| 49 | minor | `procedural_summary` vs the blanket "OUTPUT FORMAT is JSON" rule | §6.1 exception: its OUTPUT FORMAT is the §6.3 markdown skeleton + `Return ONLY the markdown above. No JSON, no wrappers.`; test 9 asserts the headings. |
| 50 | minor + major (raised twice) | Disputed-anchor items breaking `ContextItem` (`anchor_ids=[]`) before `evidence_coverage` can fire | Items carry `atom_anchor_ids(include_disputed=True)` plus `explain['disputed_anchor_ids']`; `evidence_coverage` (and `not_revoked`) evaluate the undisputed subset; test 51 spells out the observable behaviour. |
| 51 | minor | No normative scope table; ablation configurations unmapped | New §0.5 (rows 1–20 from DESIGN §2 + "everything else is in scope" line); `CompileMode.axis_graph`; `ABLATION_OF_MODE`; `retrieval['ablation']`; §15.2 step 4 distinguishes `axis_graph` (no `refine`) from `deep`; §19.3/§19.7 ablation assertions; test 63 runs the three modes. |
| 52 | major | §16.4 `w_t` sorted vs §19.6 order `[e3, e2, e4]` | `w_t = ordered_union(edge_witnesses, j.anchor_ids)` (first-occurrence order); §0.1 Determinism row distinguishes ordered unions from sorted sets; `ordered_union` declared in §9.2. |
| 53 | major | Test 40 tolerance `0.6667 ± 1e-6` | `pytest.approx(2/3, abs=1e-6)`; `cohesion_score` stored unrounded; test 41 and §19.7 use `approx(2/3, abs=1e-4)`. |
| 54 | major | Group G told to apply B's rule tables in-process with no importable functions | §11 declares public `decide_revision`, `propose_edges`, `coarsen_rule`, `EXCLUSIVE`, `PREDICATE_FAMILIES`, `family` in `hms_elite/llms/rule.py`; §16.5 steps (2)/(3) and §14.2 call them. |
| 55 | major | §11.1 `^`-anchored first-person test and unstated search-vs-match | `is_first_person()` (§9.2, `re.search`); §11 preamble: every regex is applied with `re.search` unless it begins with `^`. |
| 56 | major | §3 message-based `WitnessMissingError` mapping (SQLite FK errors are anonymous) | Explicit pre-checks in every anchor-taking writer (`get_anchors`, `details['missing']`); only the `NOT NULL … *anchor_id` message is mapped as a last resort. Test 12 asserts the unknown-id path. |
| 57 | minor | `stats(scope: Scope)` non-Optional; validity of node counts unspecified | `stats(scope: Scope | None)`; counts over all rows regardless of `valid`; `valid_edges`/`invalid_edges` the only validity split. |
| 58 | minor | `ix_atoms_head` not UNIQUE vs DESIGN's uniqueness claim | `CREATE UNIQUE INDEX ux_atoms_head ON memory_atoms(memory_id) WHERE archived_reason IS NULL` (on `archived_reason`, not `valid_to`, so expiring heads remain unique heads); `resolve_head` and `LIVE_SQL` defined accordingly; test 11 asserts the index; DESIGN §1 commitment 4 updated. |
| 59 | minor | `source_events.role` CHECK vs enum-driven generation | §7 states the two narrowings (`role` without `extracted`; `eps.scale` literal); test 11 checks it. |
| 60 | minor | `ScriptedLLM` without callable responses (test 31) | `Callable[[list[dict]], str | dict]` accepted (§11.7); test 31 rewritten. |
| 61 | minor | No public `HMS` attributes for tests | §17.1 declares `config, store, index, embedder, llm, graph, ingestor, compiler, feedback_processor, scheduler`; tests reference `hms.llm.calls`, `hms.store.trace`. |
| 62 | minor | Test 17 parametrized per provider but asserting parity | Test 17 constructs both indexes in one unparametrized test. |
| 63 | minor | §11.5 `b` join duplicating constraint texts | `b = ' ; '.join(ordered_union(split(p.b, ' ; ')…, kappa.constraints))`; §19.7 asserts each text once; DESIGN §7 updated. |
| 64 | major | §11.1 R1 listing `moved to` among the preference verbs: `'I moved to Berlin.'` matched R1 before R5 and became `likes Berlin` / `preference_event`, so a relocation never revised the previous `lives_in` | `moved to` removed from the R1 `PREF_RE` verb group (`switched to|switch to` stay); R5 gains the relocation twin `MOVED_RE`/`MOVED_ZH_RE` → `lives_in` with `relation=supersession`, tried only when the live/work pattern misses; `_SWITCH_VERBS` is now the lexicon shared by R1 and R5 (`moved to`, `relocated to`, `搬到`, `搬去`, `搬回` added); §11.3 rule 6 then REVISEs the previous residence. `SUPERSESSION_CUES_EN` is unchanged (`moved to` is still a cue). Tests 18b and 30b. |
| 65 | major | §13.1 read scope keys only from the top level, so the flattened-`AND` form Mem0 users write (`{'AND': [{'user_id': 'alice'}, {'OR': [...]}]}`) raised `VALIDATION_001` although Mem0's `_process_metadata_filters` flattens `AND` before reading the scope (MEM0_MAP §2.2 step 5) | §13.1 `parse_filters` hoists **scalar** scope keys out of the top-level `AND` chain (recursively through nested `AND`s) before building the `Scope`: first occurrence wins, a top-level key blocks the hoist, emptied clauses and an emptied `AND` are dropped, `OR`/`NOT` branches and list / operator-dict / `'*'` forms are untouched (still column leaves, still `VALIDATION_001` without a pin), and the caller's dict is never mutated. Tests 33c and 33d. |
| 66 | major | §10.1 `AnthropicLLM` always passed `temperature` to `messages.create`, but the `anthropic>=0.40` pin (§20.1) resolves to the 1.x SDK whose `Messages.create` no longer accepts the sampling parameters, so every call raised `TypeError` (surfaced as `LLM_001`) before any request | `temperature` is forwarded only when the installed SDK's `Messages.create` declares it (`inspect.signature` at construction); the pin is unchanged. Test 22b. |
| 67 | major | §12.3 coerced LLM fields blindly and outside the per-item guard: a string in a list field was iterated character by character (`objects: "black coffee"` → `['b','l',…]`; `kappa.constraints` likewise, and such a character list even satisfied `Slots._closure`), non-string text fields were stored as Python reprs, and a scalar in a list field raised a raw `TypeError` out of `HMS.add()` after Txn A | List fields accept a list or a single string (→ one element); text fields and slots must be strings; the coercion runs inside the guard so a malformed item is skipped with `logger.warning` (§12.3). Tests 32b, 32c. |
| 68 | major | §8 `_execute` released the lock before callers fetched, so concurrent readers of one `StateStore` (the `check_same_thread=False` design) received rows of another statement through CPython's statement cache (IndexError / pydantic ValidationError / short results) | `_execute` fetches every row inside the lock and returns a materialized row list; no live cursor leaves the lock (§8). Test 15b. |
| 69 | minor | README §3 showed `store: {path: ~/.hms_elite/hms.db}` as "the defaults", but `db_path()` returns `store.path` verbatim (§5, Mem0 parity), so loading that YAML created a literal `./~/.hms_elite/` under the cwd | README §3 (EN+ZH) omits `path` (→ `default_db_path()`) and states that `~` / `$VAR` are not expanded; `test_readme_config_yaml_is_the_defaults` loads the block through `HMSConfig.from_dict`. §5 unchanged. |
| 70 | minor | README §12 `hms why … --run run-billing` could not succeed as written (nothing in `run-billing` → `NOT_FOUND_001`, as §17.3 mandates) and `cli.py` let `HMSError` escape as a traceback | README §12 (EN+ZH) seeds `run-billing` with one constraint first; §17.6: `HMSError` → `to_dict()` JSON on stderr, exit 1; line cap raised to ≤ 90. Tests 62b, 62c. |
| 71 | major | §16.5 consolidate step (2) with `causal_proposer='llm'` iterated `witness_refs` unguarded inside the step's transaction, so a scalar (`7`), a nested list or a list of objects raised a raw `TypeError` that aborted `consolidate()` for every target scope (no `last_consolidation` meta, prune never ran) although §6.2 mandates "else discarded" | `_causal_edges` accepts only a list of strings for `witness_refs`; anything else drops the proposal with `logger.warning` before the guard (§6.2 and §16.5 step 2 restated). Test 56d. |
| 72 | major | §14.2 `_merge_proposal` accepted any string for slot `r` and for `kappa.visibility`, so an off-enum LLM value (`'great'`, `'everyone'`) raised `pydantic.ValidationError` from `Slots`/`Kappa` inside the coarsen transaction and aborted `consolidate()` instead of falling back to the rule value | `r` must be in `Reward.values()` (else the rule `r` with a warning); `kappa.visibility` absent or ∉ `Visibility.values()` → the rule visibility with the LLM constraints kept (§14.2 restated). Test 45c. |
| 73 | minor | §16.5 step (2) clamped `confidence` with `max(0.0, min(1.0, x))`, which turns `NaN` (accepted by `json.loads`, or `float('nan')` from the string `'nan'`) into 1.0 — a garbage confidence persisted as the maximum, irreversible through the `MAX(confidence, ?)` MERGE | `_proposal_confidence`: unparsable or non-finite → 0.7 with a warning, then clamp (§6.2 and §16.5 step 2). Test 56d. |
| 74 | major | §13.3 step 3 scored the WHOLE gated set G: `semantic_scores` asked the index for every candidate (`index.search(candidate_ids=G, top_k=len(G))`), hydrated each one and computed the four other signals — O(N) per query (≈45 µs per candidate: 133 ms at N=3000, 1.6 s at N=30000; RETRIEVAL_DIAGNOSIS T2/T3) — while `ScoringConfig.max_candidates=2000` silently dropped every memory older than the newest 2000 regardless of relevance (a correctness defect, not only a cost) | §13.3 step 3 is bounded multi-channel recall into a pool (`recall_pool` / `Pool`: semantic / keyword / entity / recency channels with per-channel K = `RetrievalConfig.k_for(top_k)`, the whole-gate rule, a round-robin union cut at `pool_max`, hydration of the pool only), every fallback level running the same recall; `RetrievalConfig` (§5.1 / §5.2, `HMSConfig.retrieval`, passed by `HMS.search` and `compile`); `max_candidates` is a deprecated None-sentinel alias that warns and truncates nothing; `safety_limit` (50000) warns and never truncates; `score_details` gains `recalled_by / pool_size / channel_sizes` (§13.4) and `AxisResult.retrieval` carries `Pool.stats()` (§4). The equivalence theorem is stated in §13.3 and pinned by 64a (bit-equality with the exhaustive path whenever G fits the channel bounds); 64b–64d cover the N=500 overlap, no truncation, provenance, the safety limit and the Python-side filter. RETRIEVAL_BENCH_2026-09-10: sqlite search@30000 172–210 ms (capped) / 1.6 s (uncapped) → 13.6–19.6 ms. |
| 75 | major | §10.3 `IndexBase.search` took only `candidate_ids`, so the pushdown-capable Milvus provider received id lists (up to 1024 ids per chunk, 13–30 KB expressions) with `limit=len(chunk)`, which turns HNSW into a brute-force scan (~58 ms per chunk, T4–T6) and left ANN no pruning at all | §10.3: `IndexFilter` (scope, `as_of` validity, scale / obs_type / atom_type whitelists, inclusive time bounds), `search(filter=)` / `count(filter=)` on `supports_filter` providers, `INDEX_ATTR_KEYS` scalars with `atom_index_attrs` / `normalize_attrs`, `upsert(attrs=)`, the non-abstract `update_attrs` / `close` hooks and the `gate_ids` SQL fallback; `MilvusIndex` keeps the scalars (INVERTED indexes), renders `filter_expr`, raises `ef` under filters (the T6 recall gap), `load_collection`s on reopen and never counts on the hot path; `SQLiteIndex` gains the generation-keyed in-process matrix + scalar cache and `select_top_k` (bit-identical to `rank_hits`); `MemoryIndex` ranks through a cached matrix snapshot. `tests/test_index_milvus.py` (native filter == SQL gate, scalar schema guard, default scalar indexes, attrs from the store, reopen), 64g. |
| 76 | major | §8 gave a provider that keeps scalars no way to follow validity / archived changes, so an archived atom stayed searchable in Milvus until its next upsert | §8 `StateStore.attr_listeners`: `archive_atom` / `restore_atom` / `set_valid_to` queue the atom and, after COMMIT, dispatch the FULL `atom_attrs` snapshot in ticket (commit) order outside the store lock; `HMS.__init__` registers `index.update_attrs`, `HMS.close` removes it before `index.close()` (§17.2); hydration re-checks validity in SQL, so a stale scalar can only hide a row, never surface one. 64e (`tests/test_index_sync.py`) and `tests/test_index_milvus.py::test_archive_hides_from_every_provider_and_rollback_restores` (sqlite / memory / milvus). |
| 77 | major | §7 had no lexical index, so a keyword channel could not be bounded: the keyword signal existed only after hydration | §7.1 schema version 3: the FTS5 virtual table `atom_fts(atom_id UNINDEXED, tokens)` fed by `fts_tokens` (the project tokenizer, so CJK bigrams and identifiers work under `unicode61`), `upgrade_3` + `backfill_fts` (`POST_MIGRATIONS`), `VIRTUAL_TABLES` / `FTS_SHADOW_TABLES` beside the 22-entry `TABLES`, `capabilities['fts5']`, `drop_all` dropping the virtual table first; the fresh-database path and the v1 → v2 → v3 ladder give identical `sqlite_master` rows. §8: `keyword_candidates` (two bm25 phases gated inside G, bounded `LIKE` fallback), `entity_candidates`, `axis_ids(order=None)`, `axis_select(atom_ids=)` (VALUES-CTE), `index_hint` (`TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX`), the write generations `data_generation / rebuild_generation / data_version()` and `clone_schema` for `':memory:'`. `test_schema_migrations_and_capabilities` (22 base + 1 virtual), `test_fresh_database_is_created_directly_at_schema_version`, `test_schema_v3_backfills_fts_from_a_v2_file`, 64g, `tests/test_perf_paths.py`. |
| 78 | minor | §15.2 step 2 ran its own candidate scan (`candidate_k=50`, threshold 0) instead of the search recall, so `compile()` would have stayed O(N) after `search()` was bounded | §15.2 step 2 calls `execute_axis_query(..., retrieval=config.retrieval)` and copies `Pool.stats()` into `ContextIR.retrieval` (`test_compile_candidates_reuse_the_pool`). |
| 79 | minor | §0.5 rows 1 / 2 / 4 / 5 / 11 / 15 / 18 still read CUT / SIMPLIFIED for modules that ship (`daemon.py`, `runtime.py`, `segmentation.py`, `compiler/coverage.py` + `scc.py`, `state/`, `eval/`, `AsyncHMS` + `telemetry.py`), §1 did not list those files, §5.1 / §5.2 / §5.4 omitted the `milvus` index and the `deepseek` / `ollama` / `fastembed` providers, and §17 had no entry for `AsyncHMS` / telemetry | §0.5 rows restated as SHIPPED with their entry points; §1 lists the files; §5.1 / §5.2 / §5.4 carry the full provider sets and `EXTRA_OF_PROVIDER`; new §17.8. README §4.1 / §10 / §10.1 / §11 / §13 (EN + ZH) and DESIGN §6.3 / §6.4 restated accordingly. |
| 80 | minor | §18.3 numbered two groups `64e` | the `tests/test_retrieval_v3.py` group is `64g`. |
| 81 | minor | `IngestConfig` / `CompilerConfig` / `SchedulerConfig` had no key for the wired modules, so `extraction_mode`, the step-8 packer and the daemon interval were constructor-only | §5 `ingest.extraction_mode` ('single' \| 'staged', forwarded to `Ingestor` by `HMS.__init__`), `ingest.segment_max_chars` (400), `compiler.selection` ('shares' \| 'coverage'), `scheduler.daemon_interval_s` (None; > 0 when set). RENAMED / REMOVED unchanged. `tests/test_wiring.py::test_config_fields_reach_their_collaborators`. |
| 82 | minor | Telemetry reachable only through `telemetry.usage_of(hms)` / `telemetry.trace` and `AsyncHMS` | §17.3 `HMS.usage() -> Usage` and `HMS.trace(operation, request_id=None)`; `AsyncHMS.usage()` delegates; `GET /usage` calls `hms.usage()`. `test_facade_usage_and_trace`, `tests/test_server.py::test_usage_and_metrics`. |
| 83 | minor | The §11 cards existed only as `hms_elite.state.StateAPI(hms)`, off the facade | §17.3 `HMS.state` (lazy `StateAPI`) and the twelve same-named delegates with `StateAPI`'s signatures verbatim; `AsyncHMS` mirrors them (`tests/test_async.py`: 40 signatures). `test_state_property_and_delegates_mirror_state_api`. |
| 84 | minor | `StateDaemon` and `AgentRuntime` were importable only from their modules; nothing on the facade started or stopped the daemon | §17.4 root exports `AgentRuntime, StateDaemon, StateAPI, StateKind, Usage`; §17.3 `HMS.start_daemon(*, interval_s=None, phases=None, filters=None) -> StateDaemon` (interval from `scheduler.daemon_interval_s`, `VALIDATION_009` when both are None; a running daemon is returned as is) and `HMS.stop_daemon(timeout=None) -> bool`; `close()` stops the daemon before the index and store. `test_start_and_stop_daemon_on_the_facade`, `test_root_exports_the_main_line_entry_points`. |
| 85 | major | `segmentation.py` shipped but the ingest plane never called it: a long message was one extraction unit, so an LLM that extracts one fact per unit lost every fact but the first | §12.6 step 5: messages longer than `ingest.segment_max_chars` are cut by `segmentation.segment` into extraction units — payload-only copies keyed to the parent anchor, so `event_idx` resolves to the parent, every observation anchors to the parent message and no segment mints a `SourceEvent` / `Anchor` (one event, one anchor); EP units stay whole; messages at or under the threshold are byte-identical to before. `tests/test_segmentation.py::test_long_messages_become_extraction_units_anchored_to_the_parent_message` (>= 3 observations on one anchor vs one with the threshold at 10**9) and `test_rule_llm_rows_are_unchanged_by_segmentation_and_short_messages_are_untouched`. |
| 86 | major | `compiler/coverage.py` + `scc.py` shipped as an optional layer no `compile()` path used | §15.2 step 8 under `selection='coverage'` (config or `compile(mode='coverage')`): `extract_requirements` → `compile_hierarchical` (L3 skills, L2 invariants, L1 atoms for what stayed uncovered, κ closure, evidence) → `select_for_coverage` floors (one item per empty non-audit slot within the leftover budget); `retrieval['selection' \| 'coverage' \| 'hierarchy']`, a trailing `## Coverage` card section, and `ContextIR.to_pcir()` lowering through `scc.lower_ir` with the fifth obligation `requirements_covered`; the `shares` path is byte-identical. `tests/test_coverage_scc.py::test_compile_selection_coverage_replaces_shares_and_lowers_with_requirements_covered` (uncovered empty on S4 + one constraint; deleting the only covering atom → uncovered non-empty, obligation violated, `PCIR_001` by default) and `test_compile_selection_coverage_refines_interface_first_after_consolidation`. |
| 87 | minor | hms4eye `reflect` had a compile entry but no evidence packet in upstream's `organize` shape | `HMSCapabilities.reflect_evidence(query, *, user_id)`: the Detective Report as `{question, question_date, mode='detective_report', ledger_rows (each with anchor_ids / atom_id / type / occurred / mentioned / entities), controls, source_snippets, answer_ready_context, conclusion, chain}`; the empty packet when nothing resolves; `manifest.SURFACE['reflect']` names it (`reflect` stays native, `compare_with_upstream()` unchanged). `tests/test_hms4eye_capability.py::test_reflect_evidence_is_an_organize_shaped_ledger_whose_rows_cite_anchors`. |
| 88 | minor | `tests/test_foundation.py::_group` left state / eval / runtime / daemon / telemetry / segmentation / coverage / scc / server in the default group, and the prompt / registry assertions predated the composite task and the later providers | §0.4 groups stated: state / eval / runtime / daemon → H (never import the facade), reranker / entities / telemetry → B, segmentation → A (A-only imports), coverage / scc → F; mvmi isolation unchanged; `COMPOSITE_TASKS` / `COMPOSITE_PROMPTS` / `MEMORY_EXTRACTION_ROOTS` asserted; `supported()` lists asserted sorted for every kind with deepseek / ollama / fastembed / spacy / milvus; `hms_elite.index` re-exports `MilvusIndex`. |
| 89 | minor | README API / config tables and this log did not list the wiring | README §3 YAML (`extraction_mode`, `segment_max_chars`, `selection`, `daemon_interval_s`) and §11 rows (coverage compile, `usage / trace`, `state` + delegates, `start_daemon / stop_daemon`, `reflect_evidence`, root exports) in both languages; rows 81–89 here. |
| 90 | major | The pool was scored on fully hydrated atoms (two pydantic validations + three JSON columns per atom, ~19 µs each), so ~150 pool rows cost 3–5 ms of the 16 ms search@30000 | §8 / §13.3 `StateStore.axis_lite(scope, atom_ids, …)` returns `AtomLite` scoring rows (the signal columns, sorted entity keys, ordered topics, `n_anchors` = the hydrated `len(anchor_ids)`, and `age_us`) in two statements per 500 ids; `execute_axis_query` scores the pool on them and hydrates ONLY the final top_k through `axis_select(atom_ids=…, trusted=True)` (`_hydrate_atoms(trusted=True)` = `model_construct` over the validated rows, field-for-field equal to the validated model). `utils.scoring.compute_signals` accepts either shape (`prepared=`, `age_us=`), `prepare_query` / `fusion_weights` derive the query side once. `tests/test_retrieval.py::test_trusted_hydration_equals_validated`, `test_axis_lite_matches_hydrated_atoms`, `test_prepared_signals_match_per_atom_path`, `test_pool_hydrates_only_top_k`. |
| 91 | major | The keyword channel ranked every FTS5 match by bm25 (`ORDER BY rank, atom_id`: linear in the matches, 4–8 ms at N=30000) and the SQL channels were one statement each | §13.3 `StateStore.recall_channels(...)`: keyword / entity / recency (and the ungated links of a hard entity gate) as ONE `UNION ALL` statement, each branch with its own LIMIT; the keyword channel reads the newest-match window (`ORDER BY rowid DESC LIMIT max(4·K_kw, 64)`, a MATERIALIZED CTE) per phase, re-applies the gate to the window, orders the survivors by `(bm25, time_anchor DESC, atom_id ASC)` and pages the window when short (`keyword_candidates` keeps the same semantics standalone); the recency branch defaults to `INDEXED BY ix_atoms_time`; the loose FTS5 phase and the small recency fallback run as their own bounded statement only when needed. `test_recall_channels_equals_channel_functions`, `test_keyword_window_is_newest_first_then_bm25`. |
| 92 | minor | `recency_score` re-parsed two ISO strings per atom (`to_utc_iso` + `fromisoformat` twice) | `axis_lite` computes the exact integer microsecond age in SQL (`strftime('%s')` + the fixed micro field of the 32-char form; NULL for any other form → the string path) and `recency_from_age_us` reproduces `timedelta.total_seconds() / 86400` bit for bit (`julianday` cannot: millisecond resolution and a ~2.4e6-magnitude double subtraction). Asserted bit-equal over four `as_of` instants in `test_axis_lite_matches_hydrated_atoms`. |
| 93 | minor | The hard entity gate was one `atom_id IN (subquery)` clause in every statement — materialized (~0.7 ms for 3000 links) even by the reads already driven through `atom_entity`, and it made the pool probe list-driven | `StateStore.entity_gate_sql(scope, keys, correlated=)` gives the IN-list form (scans: axis-only reads, counts, the recency branch) and a correlated EXISTS form (the id-driven pool reads: lite rows, final top_k, keyword window); `_plan_gates(…, entity_form='in'\|'exists'\|'none')`; `recall_pool` reads G through `atom_entity` exactly once (`entity_candidates(order=None)`: an unordered `pool_max + 1` probe under pushdown, else the whole unordered G) and, under pushdown, hands the provider the ungated links plus the filter (`_index_filter` no longer bails on an entity gate). `test_entity_gate_forms_select_the_same_set`; `tests/test_retrieval_v3.py::test_entity_gate_is_a_non_correlated_subquery` unchanged. |
| 94 | minor | `select_top_k` rescored every chosen row through `dot(matrix[r].tolist(), query)` (two list → array round trips, ~2.5 µs a row) | A float64 matrix row is already the array `dot` builds: `float(np.dot(matrix[r], q))` — same inputs, same BLAS call, same bits (asserted against `rank_hits` on 30000 stored vectors and random queries, `test_select_top_k_row_dot_is_exact`); float32 matrices keep the list path. `test_sqlite_index_cache_is_per_store` pins that the SQLite provider's cache never crosses StateStore instances. |
| 95 | minor | The pool's primary-key reads hit pages scattered over the whole file under SQLite's 2 MB default page cache (~1 ms per search at N=30000, ~100 MB file) | `StateStore(page_cache_kib=32768)` sets `PRAGMA cache_size` (0 keeps SQLite's default); `test_page_cache_pragma`. |
| 96 | minor | The Milvus in-pool rescoring was an id-filtered HNSW search per chunk (6.6 ms for 100 ids at N=30000) and multi-chunk candidate lists ran their RPCs one after another | `MilvusIndex.search(candidate_ids=…)` reads a chunk's vectors by primary key (`_vectors_of`: a `query`, 2.4 ms, exact) whenever `top_k` covers the chunk, and `MilvusIndexConfig.search_workers` (default 1) runs several chunks on a thread pool — kept sequential by default because milvus-lite serializes them (3 × 1024 ids: 41 ms sequential, 63 ms on 4 threads). `test_milvus_pool_vectors_by_primary_key`. RETRIEVAL_BENCH_2026-09-10 tables re-measured. |
| 97 | minor | RETRIEVAL_BENCH §1–§2 / README §4.1 / `bench_results/scale_v3.json` still carried the numbers measured before rows 90–96 (sqlite 15.9 ms at N=30000); `test_milvus_candidate_chunks_merge_like_sqlite` still counted `_ann` calls for whole chunks after row 96 moved them to `_vectors_of`; two numpy-only tests failed instead of skipping under a `[test]`-only Python 3.10 venv | Three arms re-measured after rows 90–96 (search median at N = 300 / 3000 / 30000: sqlite 2.71 / 2.06 / 3.41 ms, milvus 11.6 / 49.2 / 14.4 ms, mem0 0.20 / 0.23 / 0.32 ms; overlap 1.0 / 0.80 / 0.62 and 1.0 / 0.81 / 0.65; the first measurement kept in BENCH §5), README §4.2 prints the nine-task × four-ablation matrix straight from `run_ablation` (`test_readme` diffs it: Axis + Graph + Support 1.00 on every task, Axis only 0.65 mean); the chunk test asserts `_vectors_of` chunk sizes `[1024, 1024, 482]` and no `_ann` call for `top_k = len(candidates)`; `test_bounded_search_is_bit_identical_to_rank_hits` and `test_sqlite_index_cache_is_per_store` skip without numpy. |
| 98 | major | The extraction reply carried 17 keys per observation, nine of them restating `text`; on the EgoLife run the median batch emitted 5,562 output tokens (22% hit the 8,192 cap and were bisected), and output tokens are 4× cache-miss / 200× cache-hit input in price | §5 `ingest.extraction_output` ('full' \| 'compact', default 'full', forwarded to `Ingestor(extraction_output=...)`); §6.5 compact contract (`observation_extraction_compact` / `memory_extraction_compact`, `COMPACT_OBSERVATION_KEYS`, derivation table) and `hms_elite/ingest/derive.py`; §11 the rule LLM answers both compact tasks; §12.3 `extract_observations` / `extract_all` take `output` and `entity_extractor`; `llms.rule` exports `predicate_of`, `observation_topics`, `mention_texts` (R9 / topic / mention heuristics factored, byte-identical). `tests/test_extraction_compact.py` (7 tests, 12 cases). |
| 99 | major | Compact mode derived `predicate` by lexicon only and `objects = []`, so a stored compact candidate no longer matched the rule LLM's revision table (`chat_conflict` lost its CONFLICT row / conflict edge); a third-person subject named only in `text` (`Sarah joined …`) was demoted to `user`; `derive.py` compiled preference / constraint cue regexes of its own next to `PREF_RE` / `CONSTRAINT_RE`; a `"decisions": []` root without candidates still warned per observation; §11 did not name the compact handlers or the shared helpers; the compact prompt constants did not follow the `<TASK>_PROMPT` rule; test 9 said five JSON prompts; README §8 said six staged tasks | `llms.rule.statement_triple` (third-person statement parser on the R1–R9 cue tables) and `constraint_objects` are public; `derive.py` composes them (`subject_for`, `relation_of` with the statement's cue) and defines no cue; §6.5 table rows `subject` / `predicate` / `objects` / `relation` / `entity_mentions` (entity-axis narrowing stated) and invariants (table identity incl. `history` / `graph_edges`, silent empty decisions); `extract_all` treats `decisions: []` without candidates as silent; constants renamed `OBSERVATION_EXTRACTION_COMPACT_PROMPT` / `MEMORY_EXTRACTION_COMPACT_PROMPT` and named in §6.5; §11 public-function list extended and new §11.8; §12.3 bullet; test 9 says six JSON prompts; README §8 lists seven staged and two composite tasks. `tests/test_extraction_compact.py` now 9 tests, 14 cases (statement-parser table; silent empty decisions; fixture identity over `IDENTICAL_TABLES` + atom triples). |
| 100 | major | Compact mode derived `entity_mentions` as extractor(`text`) ∪ `speaker` only, so a mention only the LLM knew (a lower-case common noun such as `python`, an unrecognised name) vanished from the entity axis — the one axis the compact contract narrowed (row 99) | `COMPACT_OBSERVATION_KEYS` gains the optional `entities` hint (0–6 short strings; both compact prompts: the OUTPUT FORMAT item, one guideline line, the supersession example; the memory compact prompt reworded so it stays under 0.75 of its full twin — 5,195 / 6,936 chars, the observation one 2,978 / 4,105); `derive.union_mentions` and `expand_compact_item(entities=)`: `entity_mentions = union(entities, extractor(text), speaker)`, LLM order first, case-insensitive de-duplication; `extraction._entities_hint` (`_str_list`; a malformed shape is ignored with a warning, never skips the item); `RuleLLM._compact_observations` emits `entities` from the full observation's `entity_mentions` when non-empty; §6.5 row `entity_mentions` rewritten (narrowing sentence removed), field rules, invariants, §11.8, §12.3; README §3 / §8, DESIGN §9. `tests/test_extraction_compact.py` now 12 tests, 17 cases (full-mode parity on `User prefers python for scripting.`; union order and case folding; malformed hints ignored; prompt key set and guideline; `entity_ids` identity on every chat fixture). |
| 101 | minor | The default of `ingest.extraction_output` was left open in the docs (README §3 said the measurement phase would decide) while an incomplete real-API smoke of the row-100 prompt (`benchmarks/egolife/results/real/elite_compact_entities_smoke5`, 12 of 18 batches, no summary) sat unlabelled on disk | Decision recorded: the default stays `full`; the two-condition switching rule and the measured recommendation (`compact` for long multi-speaker transcripts and bulk ingestion) added to §6.5 and README §3 / §4.4; the incomplete run carries `INCOMPLETE.md`; new design note `docs/design/COMPACT_EXTRACTION_2026-09-11.md` (contract, same-18-batch smoke, 204-question replay A/B, derived-vs-LLM agreement, what compact drops, reproduction); pointer in EGOLIFE_REAL_LLM_ELITE_2026-09-11 §7. No code change; suite unchanged. |
| 102 | major | `ingest.extraction_output` still defaulted to `full` although both conditions of the §6.5 switching rule were met once the final compact prompt's real-API smoke completed (`results/real/elite_compact_entities_smoke5_r2`: 18/18 batches, median 2,793 vs 5,631 output tokens per request, 0 vs 8 attempts at the cap, model time −49%; replay A/B hit@20 161 = 161, entity axis compact ⊇ full on 8,439/8,439 atoms) | Default switched to `compact` (2026-09-11): `IngestConfig.extraction_output = 'compact'` (§5, bilingual comment states the rule and date); `Ingestor(extraction_output=None)` falls back to `config.ingest.extraction_output` instead of a hard-coded `'full'` (§12.1); `benchmarks/egolife/run_real.py --extraction-output` defaults to `IngestConfig().extraction_output` (manifest unchanged); `full` unchanged and byte-identical. Docs: §6.5 default paragraph and the §6.5 invariants sentence, README §3 / §4.4 / §8 (config YAML; the two output literals now print the derived confidence 0.7 → feedback `[0.7, 0.9]`), DESIGN §9, `COMPACT_EXTRACTION_2026-09-11` §0 / §2 / §6 / §7 / §8 / §9 rewritten with the `_r2` numbers (the interrupted first attempt keeps its `INCOMPLETE.md`), EGOLIFE_REAL_LLM_ELITE §7 pointer. Tests: `tests/test_extraction_compact.py` asserts the compact default in `IngestConfig`, through a default-config `HMS` (call log `memory_extraction_compact`) and in an `Ingestor` built without the argument, and that a config pinned to `full` still reaches such an Ingestor; the `tests/test_ingest.py` and `tests/test_extraction_single.py` harnesses and a module-level `config` fixture in `tests/test_scenarios.py` pin `extraction_output='full'` because they state the full contract (scripted full replies, `memory_extraction` call logs, R1 / R2 confidences 0.8 / 0.9 and the §19.8 hashes; §19 header says so). Suite 680 passed, 2 skipped. |
| 103 | major | Under `MEMORY_EXTRACTION_COMPACT_PROMPT` deepseek-flash paired observation i with neighbourhood candidate i and answered NONE 'duplicate of existing memory': 1,685 NONE of 7,315 observations on the 204-question EgoLife run (23 of 8,470 under `full`; 1,623 with `target_id == id`; 156 of 288 batches exactly observations 0..9), and the library turns a NONE into a support witness without storing the statement, so 21.8% of the extracted statements were never stored (hit@20 141 vs 161; forcing every verdict to ADD in replay gives 154 — `benchmarks/egolife/results/replay_diag/summary.md`) | (1) Prompt: the compact DECISIONS guideline is stated per NEW observation `id` (never per candidate, never by position), NONE is reserved for a restatement (same fact, same subject, same value; a new detail / person / time / value is ADD), a Bad / Good pair shows a similar candidate still yielding ADD, and the example carries two candidates with target `1` for observation `0`; observation keys and decisions shape unchanged, the rule LLM untouched, 5,192 chars against the 0.75 cap of 5,202; `MEMORY_EXTRACTION_PROMPT` / `OBSERVATION_EXTRACTION_PROMPT` byte-identical (sha256 pinned). (2) Library guard (§6.3, §12.6 step 6, both contracts, both modes): `ingest.verdict_min_similarity` (0.5; 0.0 disables) — a NONE / REVISE is honoured only when `ingest.extraction.verdict_similarity` (containment |A ∩ B| / min(|A|, |B|) of the stemmed content tokens plus the atom subject, `statement_tokens`) is at or above the bar; a REVISE also when corroborated by `supersedes_hint_ids` naming the target or an equal subject + predicate (the exclusive-attribute case: `lives_in` Paris → Berlin scores 1/3); a CONFLICT only when the texts share a token; below the bar the verdict becomes ADD with a DEBUG line and an `IngestResult.downgraded` entry. Containment over Jaccard: with the candidate texts recovered by replaying `results/real/elite_full204_r3` through the pipeline, containment ≥ 0.5 keeps 21/23 NONE and 9/9 REVISE (Jaccard ≥ 0.5: 14/23 and 1/9); the guarded replay of the recorded `full` replies keeps hit@20 161/204 and MRR 0.4973 (2 NONE downgraded, live atoms 8,439 → 8,441), the guarded replay of the recorded compact replies (old prompt) lifts hit@20 from 141 to 151/204, MRR 0.3873 → 0.4927 and live atoms 5,717 → 7,196 (1,475 NONE, 4 REVISE, 1 CONFLICT downgraded). Docs: §5, §6.3, §6.5, §12.1, §12.6, README §3 / §5.1, DESIGN §9. Tests: `tests/test_revision_guard.py` (19 cases: index-paired NONE under both contracts keeps only the restatement; threshold 0.0 restores the old behaviour and the field validates; every chat fixture row- and table-identical with the guard on and off; the canonical revisions survive; the compact DECISIONS wording; containment values; subject re-add). Suite 699 passed, 2 skipped; ruff clean; TOTAL_ENGLISH_ONLY_BLOCKS=0. |
| 104 | major | The first cut of the row-103 guard counted the atom subject in the compared token sets and applied an inclusive 0.5 bar to min-size containment, so any short statement that shared only its speaker and a reporting verb with its target scored exactly 0.5 and the index-paired NONE was honoured ("User has dogs." after "User has cats." with subject `user`; "Jake said there are two pots." against "Jake said it was put in the fridge."): on the recorded compact replies 112 NONE survived (80 at exactly 0.5, 94 sharing a reporting verb, ~110 not restatements) and the guarded replay stopped at hit@20 151/204 against the force-ADD ceiling 154 (questions 44, 301, 309); the compact OUTPUT FORMAT block still paired observation `0` with target `0` / hint `["0"]` | (1) `ingest.extraction.verdict_similarity` now removes the head instead of adding the subject: `statement_tokens(text, subject, predicate)` = stemmed content tokens minus the subject tokens minus the predicate tokens (`_head_tokens`: tokenised like the text, underscores → blanks), the predicate removed only when both atoms carry the same one; extra `with_shared=True` returns (similarity, ǀA ∩ Bǀ). (2) `Ingestor._guard_verdict` (§6.3, §12.6 step 6): a corroborated REVISE (hint or equal subject + predicate) passes first; then a verdict whose atoms carry different subjects becomes ADD (NONE / REVISE / CONFLICT alike — each asserts a relation about one subject); a same-subject CONFLICT passes (lexical overlap is no evidence for a different value); a NONE / uncorroborated REVISE needs containment ≥ `verdict_min_similarity` and at least two shared tokens or full containment of the shorter statement; `downgraded` entries carry `shared` and `subjects_differ`, the DEBUG line reads `similarity S (bar T[, subjects differ ǀ, one shared token])`. (3) Both compact prompts' OUTPUT FORMAT cite hint `["1"]`, the memory task's decision `target_id` `"1"` for observation `"0"` (5,192 chars unchanged; full prompts byte-identical). Zero-API replays (`results/replay_diag2/summary.md`, G1v2 / F1v2, same argv as G1 / F1): compact hit@20 154/204 = the W5 ceiling and the same hit set (MRR 0.5038, live atoms 7,299, 1,583 downgraded = NONE 1,577 + REVISE 5 + CONFLICT 1; 10 NONE and 1 CONFLICT honoured; 44 / 301 / 309 recovered over G1); full 161/204, MRR 0.4973, live 8,442, 3 of 28 guarded verdicts downgraded (the two G1 NONE plus one uncorroborated REVISE at 0.333), top-20 rows identical to F1 on 204/204. Docs: §6.3, §6.5, §12.6, README §3 / §5.1, DESIGN §9. Tests: `tests/test_revision_guard.py` 34 cases (+15: short user facts sharing only the head are stored under both contracts with the `today` pair rejected by the single-token rule; cross-subject NONE / REVISE / CONFLICT downgraded and a same-subject CONFLICT honoured; twelve head-aware containment values; `statement_tokens` drops the head; the OUTPUT FORMAT pairing), `tests/test_extraction_compact.py` asserts the hint id `1`. Suite 714 passed, 2 skipped; ruff clean; TOTAL_ENGLISH_ONLY_BLOCKS=0. |
| 105 | major | `ingest.extraction_output` defaulted to `compact` (row 102) on the strength of a zero-API replay and an 18-batch smoke, but the real 204-question EgoLife run scored 53.9% on memory rows under compact v1 (`results/real/elite_compact_full204`; root cause the index-paired NONE verdicts of row 103) and, after the rows 103–104 wording and `verdict_min_similarity` guard, 56.4% (115/204) under compact v2 (`results/real/elite_compact_v2_full204`; card 62.7%, hit@20 77.9% / 79.4%, NONE 15, 0 windows lost, ¥4.35 off-peak, 7.3 s per batch) against 61.3% (125/204) for `full` r3 (card 61.8%, hit@20 78.9%, ¥11.56, 14.9 s) — 4.9 points below, outside the 2-point tolerance | Default reverted to `full` (2026-09-11 evening): `IngestConfig.extraction_output = 'full'` (§5; the bilingual comment states the revert, the date, the rule and the v2 numbers); `Ingestor(extraction_output=None)` still follows `config.ingest.extraction_output` (§12.1 unchanged); `benchmarks/egolife/run_real.py --extraction-output` still defaults to `IngestConfig().extraction_output`; both prompts byte-identical. §6.5 default paragraph rewritten with the third condition (rows accuracy within 2 points of `full` on the real 204-question run), the v1 root cause, the v2 numbers and the recommendation (`compact` for cost-sensitive bulk ingestion and state-card readers, 0.38× write cost / 0.49× batch time; `full` for memory-rows readers); §5 field list and the §6.5 invariants sentence; README §3 / §4.4 / §8 (config YAML `extraction_output: full`; the two output literals print the LLM's confidence 0.8 → feedback `[0.9, 1.0]` again), DESIGN §9, `COMPACT_EXTRACTION_2026-09-11` §0 / §6 / §8, `HIGHLIGHTS_2026-09-11` §1.4, `SCENARIO_COMPANION_TOY_2026-09-11`, `EGOLIFE_REAL_LLM_ELITE_2026-09-11` §7, the two `docs/design/artifacts` pages (compact is worded as optional; measured numbers unchanged); `EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md` holds the decision. Tests: `tests/test_extraction_compact.py` asserts the full default in `IngestConfig`, through a default-config `HMS` (call log `memory_extraction`) and in an `Ingestor` built without the argument, and that a config pinned to `compact` reaches such an Ingestor (call log `memory_extraction_compact`); the `full` pins in `tests/test_ingest.py`, `tests/test_extraction_single.py` and `tests/test_scenarios.py` stay (harmless, they state the contract they test). |
| 106 | major | Output tokens are 83% of the full contract's write cost on the 4,851-window EgoLife run (2.40M output tokens of ¥11.5; compact v2 0.76M of ¥4.2; mem0 0.22M of ¥1.85 with the input-miss side already at parity): the JSON envelope costs 2–4× the text it carries, retrieval uses only the texts and the decisions (zero-API replay: entities / type / when move hit@20 by 0), and the compact prompt's own texts score below the full prompt's (rows 56.4% vs 61.3%), so neither existing contract can close the gap | New `ingest.extraction_output='lines'` (§6.6, 2026-09-12): the LLM answers `memory_extraction_lines` with tab-separated `OBS` / `EP` / `DEC` text (no JSON, no json mode) — `<event_idx>⇥<T>⇥<statement>[⇥@date]`, `<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`, `<OBS line>⇥REVISE|CONFLICT|NONE⇥<candidate>⇥<reason>`; the prompt keeps the full prompt's statement rules verbatim and the compact DECISIONS rules, replacing only the OUTPUT FORMAT (`MEMORY_EXTRACTION_LINES_PROMPT`, `COMPOSITE_TASKS`, `LINE_TYPES` / `TYPE_LETTERS`); `LLMBase.generate_lines` calls `_call` with no `response_format` (§10.1); `hms_elite/ingest/lines.py::parse_lines` is the tolerant parser (fences, prose, missing tab, unknown letter, truncated last line, DEC out of range); `extract_all` branches on `output == 'lines'` and then runs the compact item path with `speaker_from_subject=True` (the leading proper name becomes speaker → entity mention / actor_id; `derive.SIDE_SUBJECTS`), a missing DEC line is ADD silently; `RuleLLM._task_memory_extraction_lines` renders its R1–R9 result as lines (§11.8); single mode only — `IngestConfig` validator (bilingual ValueError), `Ingestor` and `extract_observations` (`VALIDATION_013`); `EXTRACTION_OUTPUTS = ('full', 'compact', 'lines')`, the benchmark CLI choices follow it. Default unchanged (`full`); full and compact prompts byte-identical (four sha256 pinned). Docs: §5, §6.1, §6.4, §6.6, §10.1, §11.8, §12.3, README §3, DESIGN §9. Tests: `tests/test_extraction_lines.py` (16 cases: parser well-formed / tolerance / truncation / DEC range + EP shapes; a scripted lines reply through `HMS.add`; DEC REVISE → revision edge with the guard still applying; the rule LLM's lines facade row- and table-identical to the full facade on every chat fixture except the speaker-rule `actor_id`; EP / @date / DEC rendering; config validation and wiring; prompt registration and statement rules; DeepSeek sends no `response_format`); `tests/test_foundation.py` and `tests/test_extraction_compact.py` pins of the tuples updated. Suite 730 passed, 2 skipped (714 + 16); ruff clean; TOTAL_ENGLISH_ONLY_BLOCKS=0. |
| 107 | major | Review of the lines contract after the 18-batch smoke (`results/real/elite_lines_smoke5`, 2026-09-12): (1) `parse_lines` numbered observations by how many it had kept, not by the OBS line the model wrote, so a dropped line (empty statement) or a wrapped statement (a tab-less continuation stored as its own fragment atom) shifted every later DEC target by one and a REVISE could revise the wrong atom (the guard only checks subject + predicate); (2) the lines wire format carries u / a / y / r / k only and nothing derived `entity_mentions` / `topic_tags` / `b` for EPs, so lines EPs stored `[]` / `[]` / `''` where full stored the rule LLM's values; (3) the §5 signature still read `Literal["full", "compact"]`; (4) README §3 / §6.6 / DESIGN §9 said the lines facade was "row- and table-identical" to full while the test compared 8 tables and the atom quads (confidence, relation, actor_id and the EP columns differed unpinned); (5–7) the benchmark harness: an unusable finish=stop lines reply (prose, a JSON object) was labelled empty_extraction instead of invalid_json because `parse_lines` never emits the JSON path's log strings, so the retry / bisect never ran; a finish=length lines reply was stored in part by the library before the slice was bisected (parent atoms next to the halves' atoms, `added` double-counted); the items_skipped grep counted the lines parser's own "skipped" warnings as schema rejections; `manifest.extract_params.response_format` hard-coded `json_object` for the lines arm. None fired in the smoke (0 dropped / wrapped lines, 0 EP lines, 0 truncated attempts, 0 parser warnings) | `parse_lines`: every tabbed OBS line consumes one 1-based line number whether or not it stores an observation (a prefix-only line keeps its number and waits for a continuation), a tab-less line is a continuation (fills the pending slot, else is appended to the previous statement) and consumes no number, a DEC line naming a line that stored nothing is skipped with a warning, and a reply that opens no section logs one warning containing the new `NO_SECTION_MARK`; new `ingest.derive.expand_lines_ep` (entity_mentions / topic_tags from `'<a> <y> <u>'`, `b` = the k= constraints joined by `' ; '` when empty) runs before `_ep_from_item` under lines — the `eps` table is now identical to full on every chat fixture; §5 signature fixed in both languages; §6.6 invariants, README §3 and DESIGN §9 now name the identical tables (nine incl. `eps`) and pin the derived columns of `observations` / `memory_atoms` (`confidence`, `relation` / `supersedes_hint_ids`, `actor_id`, chat_zh `entity_mentions`) column by column; harness: `parse_error` also counts `NO_SECTION_MARK` (→ invalid_json_retry / bisect / invalid_json), parser warnings are counted apart as `lines_dropped` (sub record, `classify_retain`, `_parse_failures`, `other_warnings`), `EliteRealArm._reject_truncated_lines` hands a finish=length reply to the library as `''` so nothing of it is stored before the bisect, and the manifest's `response_format` reads `none` for the lines arm. Re-parsing the smoke's 18 raw replies with the new parser gives the same 1231 observations / 82 decisions / 63 distinct ids (no line consumed a number without storing, no continuation). Tests: `tests/test_extraction_lines.py` 16 → 21 cases (numbering with a dropped / wrapped / prefix-only line; the no-section warning; the parent repro through `HMS.add` — the REVISE meant for line 3 no longer revises the oat-milk atom; `expand_lines_ep` and a scripted EP through `HMS.add`; the identity test compares `eps` and the two derived-column tables; harness source pins); the tolerance test now expects a tab-less line to continue the previous statement. Suite 735 passed, 2 skipped; ruff clean; TOTAL_ENGLISH_ONLY_BLOCKS=0. |
| 108 | major | Default extraction contract and read path (2026-09-18, `docs/design/READER_AB_2026-09-18.md`): the library default `ingest.extraction_output` was `full` (¥11.56 per 204 EgoLife questions) while the reader A/B on the lines store measured 70.1% (143/204) for a hybrid context — a session card memoized per timeline step as a prefix plus timestamped, deduplicated, time-sorted search rows, answered memories-first — against 61.3% / 61.8% for the full contract's rows / card at the lines write cost ¥2.24 (sign test vs rows p < 0.0001); the hybrid builders lived only in `benchmarks/egolife/reader_ab.py`, so no library caller could reproduce the measured read. **Breaking for `staged` configs**: with `lines` the default, `{"ingest": {"extraction_mode": "staged"}}` without an explicit `extraction_output` is now `CONFIG_001` (the unchanged `_lines_needs_single` validator). | `IngestConfig.extraction_output` default `full` → `lines` (`hms_elite/configs/base.py`, comment history extended; `hms_elite/ingest/extraction.py` header). New `hms_elite/compiler/context.py` (`ROWS_HEADER`, `NONE_TEXT`, `normalize_text`, `row_times`, `dedup_rows`, `sort_rows_by_time`, `format_stamp`, `render_rows`, `rows_text`, `hybrid_text`; exported by `hms_elite.compiler`) and `benchmarks/egolife/reader_ab.py` now imports them (its `dedup_rows` / `sort_rows_by_time` / `format_stamp` / `row_times` / `normalize_text` / `hybrid_context` are the library functions, `rows_ts_context` delegates to `rows_text`, `HYBRID_ROWS_HEADER = ROWS_HEADER`, `build_prompt = real_arms.build_prompt`). New `HMS.context(query, *, filters, top_k=None, as_of=None, budget_tokens=None) -> dict` (§6.7, §17.3) with the in-process session-card memo (`_session_card`, `_context_memo`, `_context_lock`; cleared by `reset()`), mirrored by `AsyncHMS.context` (§17.8; 18 native methods); `StateStore.latest_event` / `touch_scope` / `scope_version` / `unversioned` and the per-scope write versions (`write_generation`, `scope_write_versions`, `_txn_scope_keys`, `_txn_wrote`; bumped at COMMIT only when the transaction wrote versioned rows, on rollback and reset; only row-changing statements outside `unversioned()` count, so an internal event + anchor (`system_event`), a checkpoint, the scheduler's meta rows and compiled-state archives, an idle scheduler phase and a zero-row transaction move no version — §8); `CompilerConfig.context_session_budget_tokens` 1200 / `context_rows_top_k` 20 / `context_rows_header` / `context_timestamps` True (§5). Harness: `benchmarks/egolife/arms.py` no longer pins `full`; `real_arms.build_prompt(question, context, order)` + `answer(..., order=)` + `EliteRealArm.hybrid()` + `stats()` context fields; `run_real.py` Elite labels `elite_rows` + `elite_hybrid` by default (`ENGINE_LABELS`, memories-first prompt, `hybrid` summary block and table, `--with-card` re-adds `elite_card`; `--extraction-output full` stays selectable); `report_real.py` reports the arms a run directory carries (`Run.arms`, `OPTIONAL_ARMS`), `casestudy_report.py` treats the card file as optional. Tests: `tests/test_context.py` (7 cases, §18.3 #65); `tests/test_async.py` 40 → 41 / 17 → 18 with `context`; `tests/test_facade.py` signature pinned; `tests/test_wiring.py` pins the four config defaults; `test_extraction_compact.py` / `test_extraction_lines.py` expect `lines` for the untouched facade; `test_wiring.py` / `test_extraction_single.py` / `test_segmentation.py` pin `full` where staged mode or full-contract JSON replies are used; README literals regenerated (`[0.7, 0.9]`, `'confidence': 0.7`). Docs: CONTRACT §5.1 / §5.2 / §6.5 / §6.6 / new §6.7 / §8 / §17.3 / §17.8 / §18.3 / §19 (both languages), README §3 / §4.4 / §5 (new `context()` snippet) / §8 / §11, DESIGN §6.11 / §9 / §13, OVERVIEW / HIGHLIGHTS / COMPACT_EXTRACTION one-line notes. Follow-up: a `POST /context` route. Suite 763 passed, 3 skipped; ruff clean; TOTAL_ENGLISH_ONLY_BLOCKS=0. |
| 109 | minor | CJK text switches for the EgoLife-ZH experiment (2026-09-22, `docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`): the library emitted CJK character bigrams on both sides of the keyword channel with no way to measure the English-centric baseline, and the compact / lines derivation read no Chinese subject, verb or object, so a Chinese statement stored as `user / states / []` with no `actor_id`. Defaults reproduce the previous behaviour byte for byte (English stores, recorded runs, the `rule@entities-v1` identity, no new meta row). | New §9.6. `TextConfig.cjk_tokens: Literal["bigram", "none"] = "bigram"` (`HMSConfig.text`, after `retrieval`; `hms_elite/configs/base.py`, exported by `hms_elite.configs`); `utils.text.cjk_tokens_scope` / `cjk_tokens_mode` / `text_scoped` / `CJK_TOKEN_MODES`, `tokenize(text, *, cjk=None)`, `content_tokens(text, *, cjk=None)`, `stem` CJK guard; `memory.main.TEXT_SCOPED_METHODS` (35 wrapped `HMS` methods) + `Ingestor.ingest_messages / ingest_events`, `StateCompiler.compile`, `StateScheduler.run` under `text_scoped`; `scoring._content_vocabulary(text, cjk)` / `_key_tokens(key, cjk)`; `StateStore.rebuild_fts()` + `HMS._sync_fts_mode` + meta key `text_cjk_tokens` (`TEXT_CJK_META_KEY`). `IngestConfig.zh_rules: bool = False`; `extract_all / extract_observations / _compact_observation_from_item / expand_compact_item / expand_lines_ep(..., zh_rules)`; `llms.rule.statement_triple(..., *, zh_rules)`, `predicate_of(..., *, zh_rules)`, `mention_texts(..., *, zh_rules)`, `observation_topics(..., *, zh_rules)`, `VERB_LEXICON_ZH`, `_zh_objects`; `utils.text.extract_entities(text, *, zh_rules=False, names=())`, `_zh_name_candidates`, `ZH_STATEMENT_VERBS`, `ZH_NAME_STOP`, `TOPIC_TABLE_ZH`, `sentiment(text, *, zh_rules)`, `topic_tags(text, *, zh_rules)`, priorities NAME 4 / ZH 5 / CJK 6; `RuleEntityExtractorConfig(zh_rules, names)` and the `rule@entities-zh-v1` identity (§10.5); `HMS.__init__` forwards `ingest.zh_rules` to the rule extractor. Review fixes (same day): `utils.text.ZH_TIME_WORDS` / `ZH_LEAD_ADVERBS` / `ZH_LEAD_ADVERB_ALT` / `zh_name_ok` / `zh_name_stop_prefix` (heads containing a verb or a time word are never names; 了 / 过 / 着 skipped after 和 / 跟 / 给; 了 no longer a name boundary), `llms.rule._zh_clause` (coverb predicate = the descended verb; `VERB_LEXICON_ZH` 和 / 跟 → `states`, no `with`), `statement_triple` skips a leading stop / time word before the subject, one adverb alternation for both sides; `HMS.last_downgraded` (guard-downgraded verdicts of the latest add, envelope unchanged). `MEMORY_EXTRACTION_LINES_PROMPT` gains the source-language sentence (no pin). Harness: `benchmarks/egolife/translate_zh.py` (data_zh builder: deepseek-flash window translation through `real_arms.DeepSeekClient`, resumable JSONL cache, `--cost-cap-cny`, `--workers`, deterministic `NAME_GLOSSARY`, validation, bisect on truncation, `translate_report.json`), `windows.build_windows(..., text_lang='en' | 'zh')`, `real_arms.EliteRealArm(embed_model, embed_dims, cjk, fastembed_cache_dir, entity_names)` / `Mem0RealArm(embed_model, embed_dims)` (`stats()['embedder']` = `real_arms.embedder_identity`: `fastembed:<model>` for the default embedder — the recorded `results/real/mem0_full204/summary.json` string — and `fastembed:<model>@<dims>` otherwise), `EliteRealArm.recall` searches with `explain=True` and carries `keyword` / `entity` / `recalled_by` / `channel_sizes` per row (`questions_elite_rows.jsonl` `channels`: `keyword_hits` / `entity_hits` / `channel_sizes` / `recalled_by`), `retain_with_bisect` / `summarize` carry `downgraded`, `run_real.py --text-lang / --embed-model / --embed-dims / --cjk / --fastembed-cache-dir / --entity-names` (manifest `embedding.dims_source / cache_dir / fastembed_cache_path_env`, `data_root`, `text_lang`, `cjk`, `library_switches`, `entity_names`; summary top-level `embedding` / `text_lang` / `cjk`; `default_entity_names`: the glossary names only for `--text-lang zh --cjk on`, so run A keeps `rule@entities-v1`; `FASTEMBED_CACHE_PATH` assigned unconditionally — so the default English invocation's process environment now carries that variable although its outputs are unchanged: same ONNX weights, the en model already present under `~/.cache/hms-elite-bench/fastembed_cache`, `manifest.embedding.fastembed_cache_path_env` records it), `translate_zh.py` `SpendLedger` / `translate_spent.json` (the cap is cumulative over resumes), `--allow-tight-cap` (a projection above 80% of the remaining cap is refused), `--retry-failed` / `resume_skip_ids` (a failed single-window retry is final), bisect halves stop once the cap trips (rows with reason `cap`), `load_cache` skips a truncated last line, `questions_source_note` records the native-questions deviation, `compare_real.py` / `report_real.py` print each run's embedder identity, text language and CJK switch instead of the hard-coded ones. Second review round (same day): the read side of the entity axis takes the facade's extractor (`plan_query(..., extractor)`, `mvmi.query._scoring_plan` / `search(..., extractor)`, `StateCompiler(..., entity_extractor)`, `coverage.extract_requirements(..., extractor)`; `HMS.search` / `context` / `compile` / `coverage` pass `self.entity_extractor`), `_POSITIVE_SENT_ZH_RE` bounded 爱 forms (爱丽丝 is a name), `statement_triple` steps over a non-我 pronoun head (我们 → `project`, 他 / 她 / 他们 / 大家 → None) before reading the verb, `zh_name_ok` tests verbs / adverbs at the head's edges only (王和平 accepted; 李刚 still rejected, documented), `real_arms.EliteRealArm.recall` reads `entity` from `score_details['entity_boost']` (it read a non-existent `entity_score`, so `channels.entity_hits` was always 0), `translate_zh.validate_window(..., question=True)` judges the Latin rate on the question line only (`translate_items` passes it for `q:` items), `write_dataset` marks `native_fallback` with the `build_windows(zh)` predicate (English AND Chinese on a cue) and reports `questions_translated` / `questions_native_fallback`, `translate_all` prints progress every ten completed requests whatever `--workers`, `--glossary-add NAME=中文` (`parse_glossary_add`; report `glossary_added` / `latin_names_note` — the Jack == Jake merge is a user decision, EGOLIFE_ZH_PLAN §3), the `--allow-tight-cap` help escapes its `%`, `compare_real.unique_run_roots` states each run's controls once, `run_real` records `manifest.data_root` absolute (`resolve()`). Tests: `tests/test_text_cjk.py` (5), `tests/test_zh_rules.py` (5), `tests/test_egolife_zh.py` (harness, offline, 6), `test_wiring.py` / `test_foundation.py` / `test_extraction_lines.py` pins. Docs: CONTRACT §5.1 / §5.2 / §9.2 / §9.6 / §10.5 / §11.1 (both languages), README §3 YAML + provider table, `docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`. |
| 110 | minor | §9.6 `extract_entities(zh_rules=True)` still ran the legacy `_CJK_PROPER_RE` lookahead, so the EgoLife-ZH smoke store (`results/real/zh_smoke5_B`, 2026-09-22) linked 350 fragment rows such as 要求团队 / 都写 / 硬盘假装 / 他的家乡 / 做的布线 / 并让大家 in `atom_entity` and the FTS row text, although the name rules and the closed `names` glossary already gave clean subjects on 979 / 980 atoms. | The lookahead is skipped when `zh_rules` is True (`hms_elite/utils/text.py`); `zh_rules=False` is byte-identical. Under `zh_rules` the entities of a text are the `names` hits, the `_zh_name_candidates` heads, FILE / IDENTIFIER / QUOTED and Latin PROPER runs. The ingest path already passes the facade's extractor to every `create_atom`; `ops.consolidation` and `graph.operators` still link with the plain rule extractor (unchanged). `tests/test_zh_rules.py::test_zh_rules_skip_the_legacy_cjk_lookahead`. |

Cross-cutting consequences recorded above: §7 DDL (`sensor_trace`, `entity` kinds, `interface_ep` transitions, `target_kind` values, `compiled_states`, `ux_atoms_head`); §2 enums (`Route.preference`, `CompileMode.axis_graph`, `TransitionKind.interface_ep`, `SourceKind/NodeKind.entity`, `ObjectType.sensor_trace`, `KIND_ORDER`, `ABLATION_OF_MODE`); §5 config keys (`graph.coarsen_l3`, `scheduler.decay_rate`, `scheduler.compile_state`, `ingest.emit_relation_atoms=True`); §18.3 test numbering keeps 1–63 and adds 18b, 30b, 45b, 54b, 56b, 56c, 64; §19.7 report/stats dictionaries restated in full. Unresolved items: none.
