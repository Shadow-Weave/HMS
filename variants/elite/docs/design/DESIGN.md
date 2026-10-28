# HMS Elite — Final Design (设计定稿)
中文版：[DESIGN.zh.md](DESIGN.zh.md)

> Base: `proposal_hms_invariants` (winner, 3/3 judges). Grafted: lineage ids + Mem0 config/signature parity (`proposal_mem0_parity`), structured closure builder + query planner + scenario suite + full-state checkpoints (`proposal_scenario_mvp`). Every blocker/major defect listed by the judges is closed in §3. The normative artefact is `CONTRACT.md` (same directory); this document explains *why*. Where the two disagree, `CONTRACT.md` wins.

---

## 1. 总纲 Thesis

HMS Elite is a Python package (`hms_elite`, PyPI `hms-elite`) that wears Mem0 v2.0.19's skin — same signatures, envelopes, filter grammar, config dict, error shape, parse ladder, factory style — over an HMS dual-plane skeleton:

```
axes find            State Index (memory_atoms + 7 axis tables)   hot path: add/search/get_all
graph inherits       Support Graph (graph_nodes/edges/witnesses/interface_eps/projections)   inheritance path: why/compile(deep)/consolidate
ContextIR acts       StateCompiler → ContextIR → Work State Card / PC-IR / StatePackage
experience trains    transitions table: (S_t, a_t, S_{t+1}, r_t, κ_t, w_t) per EP / InterfaceEP / causal / revision / conflict / rollback (uniform κ_t)
```

Ten commitments, each enforced in code, not prose:

1. **No Witness, No Meaning — three layers.** pydantic (`Witnessed.anchor_ids: list[str]` with `min_length=1`), DDL (`NOT NULL REFERENCES anchors(anchor_id)` primary-witness column on observations, eps, memory_atoms, graph_nodes, graph_edges, interface_eps, feedback_events, history, checkpoints, transitions), and guarded writers (`SupportGraph.insert_edge` / `StateStore.insert_edge` accept a plain dict and raise `WitnessMissingError` on an empty witness list BEFORE the pydantic model is built and before any SQL; every anchor-taking writer pre-checks ids with `get_anchors`; the raw cursor is private). Anchors can be revoked (`HMS.revoke`) with a cascade to derived atoms/edges/InterfaceEPs.
2. **Atoms store no edges — by type and by import topology.** `MemoryAtom` has no edge field (`extra='forbid'`, a validator rejects `edges/edge_ids/src_id/dst_id` keys inside `content`); `hms_elite/mvmi/*` has no import path to `hms_elite/graph/*`; `search()` never touches a `graph_*` table (asserted by `store.trace`). The `instr()` string CHECK of the winning proposal is dropped (false positives on the value `"edges"`).
3. **One node per atom version, `node_id == atom_id`.** Every atom (message, observation, event, EP, InterfaceEP, skill, procedural) gets exactly one `graph_nodes` row written by `insert_atom`. This removes the dual atom/EP node ambiguity and makes `why()`, rollback and feedback one id space.
4. **Stable Mem0 id = lineage id.** `memory_atoms.memory_id` (Mem0 `id`) + `version`; `atom_id` is the version id. `update(id, text=)` → archive head + insert version+1 in the same lineage + witnessed `revision` edge; `get(id)` returns the head's *new* text; `history(id)` spans the lineage; `add()` REVISE joins the old lineage and reports `event='UPDATE'`. Partial unique index `ux_atoms_head` (`WHERE archived_reason IS NULL`) guarantees at most one non-archived version per lineage — an expiring head (future `valid_to`) is still the head, so `get()` keeps working for memories with an `expiration_date`.
5. **Revision instead of deletion, structurally.** `BEFORE DELETE` triggers on every evidence/state table; `BEFORE UPDATE OF <content columns>` triggers make atom/EP/observation/edge/InterfaceEP content immutable; only `valid_to / confidence / visibility / user_meta_json / updated_at / archived_reason / valid / invalidated_* / resolved_*` may change. `delete()` = `valid_to=now, archived_reason='deleted'` + witnessed `DELETE` history row.
6. **Every mutation is itself an event.** `update/delete/delete_all/feedback/checkpoint/rollback/consolidate` first write a `SourceEvent` (`object_type` ∈ `system|feedback|checkpoint|rollback|scheduler`) + `Anchor`; that anchor witnesses every row they create. Nothing in the DB lacks a chain to raw evidence.
7. **High-level objects are interfaces, not summaries — by schema.** `interface_eps.primary_support_node_id NOT NULL`, `support_node_ids` `min_length=2`, `Slots` is one shared pydantic model for EP and InterfaceEP (isomorphic grammar), and **only boundary (projected) edges** carry `witness_edge_id` (`CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL)`); the `AFTER INSERT` trigger writes `edge_projections` for them. High-scale `revision`/`distill` edges are witnessed by anchors only — the winning proposal's over-broad CHECK (every non-L1 edge must be a projection) is fixed.
8. **Structured events never touch an LLM and close into real EPs.** `tool_call`/`file_diff`/`test_log` become one `(u,b,a,y,r,κ)` EP per action→outcome closure (P3 closure builder), witnessed by both anchors; a negative EP mints a derived `constraint` atom joined by a `distill` edge; `use` edges point at background constraints; `causal`/`temporal` edges join consecutive EPs. The Work State Card's *Failed Paths* and *Next Action* are reachable right after `add_events()` with zero LLM calls.
9. **Lock discipline: Txn A → unlocked extraction → Txn B.** Evidence (events + anchors) commits first; LLM/embedding calls run with no lock held; materialization (atoms, M2M, vectors, nodes, edges, history, transitions) commits second. A crash between A and B is recoverable by `consolidate(phases=['collect'])`.
10. **Offline determinism is a correctness feature.** `rule` LLM + `hash` embedder + injectable `now()`/`id_factory` on `StateStore` make every test byte-reproducible; the six golden scenarios (§12) are asserted byte-for-byte on both index providers.

---

## 2. Lite 裁剪 What is cut / simplified, and why

| # | Full HMS feature | Lite decision | Reason |
|---|---|---|---|
| 1 | `daemon.py` background loop | CUT → `consolidate()` runs `orient/collect/consolidate/prune` synchronously | fixed decision 9 (sync only) |
| 2 | `runtime.py` agent adapter / action registry | CUT | `add_events()` is the adapter contract |
| 3 | Per-source adapters (slack/jira/feishu/sensor) | SIMPLIFIED → one `SourceEvent` with 5 extractable `object_type`s (`message, tool_call, file_diff, test_log, sensor_trace`) | transport glue, not invariants |
| 4 | `segmentation.py` | SIMPLIFIED → sentence split inside the `rule` extractor | needs a tuned model |
| 5 | `coverage.py`/`scc.py` hierarchical coverage compiler | SIMPLIFIED → per-slot share budget with a constraint floor | accuracy feature, not an invariant |
| 6 | Vision / perception | SIMPLIFIED → list content reduced to text parts, image parts anchored as sha256/mime (`parts`), `dropped_parts` recorded; `llm.enable_vision` + the provider `describe_image` hook (unlocked, before Txn A) merges descriptions into the text and records `described_parts` | no offline perception; vision is a provider capability |
| 7 | FTS5/BM25 lexical axis | SPLIT → recall order comes from the stdlib SQLite FTS5 table `atom_fts` (bm25, `keyword_candidates`; LIKE fallback when `capabilities['fts5']` is False); the in-pool keyword *score* stays the pure-Python token overlap with identifier tokens ×2 | zero optional deps; bounded recall |
| 8 | Reranker providers | SIMPLIFIED → `overlap` (default, offline weighted token coverage) + `llm` (`rerank` prompt contract, overlap fallback on `LLM_002`); `search(rerank=True)` reranks after fusion, rows gain `rerank_score`, failures keep the retrieval order; `retrieval['rerank']` stays `None` (compile never reranks) | Mem0 kwarg parity with one offline default |
| 9 | Per-topic half-life learner | SIMPLIFIED → one `scoring.half_life_days` | determinism |
| 10 | Embedding-tier entity resolution (cosine ≥ 0.95) | SIMPLIFIED → normalized-text exact match + alias list; `run_id` excluded from entity identity | hash embeddings are not semantic; S3 needs cross-run aggregation |
| 11 | Personal/Edge/Group renderers | CUT; Work State Card + `StatePackage` only | one deterministic renderer proves the contract |
| 12 | LLM coarsen | SIMPLIFIED → `rule` coarsen default; LLM coarsen behind the same prompt contract, refs validated against V | reproducible round-trip test |
| 13 | Skill promotion | SIMPLIFIED → an InterfaceEP invariant of kind `constraint` with ≥2 support nodes becomes an `L3_macro` skill atom + a `distill` edge that is traceable to the low-level `use` edges (`witness_edge_id` + `edge_projections`) | reachable in ONE run (fixes the winner's two-run gap) |
| 14 | Governance/ACL evaluator for κ | SIMPLIFIED → visibility lattice gate applied at SELECTION time (filters, graph neighbours, invariants, evidence anchors) + `governance`/`not_revoked` obligations | hook in place for a policy engine |
| 15 | Benchmark & ablation harness | CUT; the HMS Phase-5 ablation configurations map to `CompileMode`: `fast` = Axis only, `axis_graph` = Axis + Graph, `deep` = Axis + Graph + Support; "Graph only" is not applicable (the graph is entered only from axis seeds); `retrieval['ablation']` records which ran | package scope |
| 16 | HTTP server | SHIPPED as `hms_elite/server` (`create_app()` / `hms-server`, optional `server` extra, group H); `cli.py` ships | library first; the service face only wraps the facade |
| 17 | External index providers (Qdrant/FAISS) | SIMPLIFIED; `IndexBase` pluggable, ships `sqlite` + `memory`; `milvus` (milvus-lite file or server) behind the optional `milvus` extra, re-scored by `rank_hits` for bit-identical scores | fixed decision 3 (core stays pydantic-only) |
| 18 | Async twin, telemetry, notices, `setup_config()` | CUT | fixed decision 9 |
| 19 | Mem0 `chat()` / `project` | `chat()` raises `NotImplementedError`; `project.update()` raises `ValueError` | keep Mem0 API-shape smoke tests green |
| 20 | Mem0 `history.db` import | CUT (replaced by `hms_elite.migrate.from_mem0(memory, hms, *, filters)` walking `get_all` and re-adding with `infer=False, timestamp=created_at`) | 20 lines, no column-set-diff migration |

---

## 3. 评审意见的采纳 Judge verdicts → design changes

### 3.1 Grafts (all three judges)

| Graft | Source | Where it lands |
|---|---|---|
| Lineage id `memory_id` + `version`, `ux_atoms_lineage_version`, partial `ux_atoms_head`, `store.resolve_head()`, `update()` returns `{'message'}` with the same id, `history()` spans the lineage, `add()` rows carry `revises`/`conflicts_with` | P1 §A.2/§C/§D/§E.1 | CONTRACT §4 `MemoryAtom`, §7 DDL, §8 store, §17 envelopes |
| Mem0-byte-compatible config: `vector_store` is the index envelope key, `history_db_path` → `store.path`, `version ∈ {'v1.1','v2','lite-1'}`, `reranker` is a registered envelope (`overlap` | `llm`), RENAMED/REMOVED tables in one `mode='before'` validator; `ingest.revision_mode: inline|deferred|off` | P1 §H.1, §K.1 | CONTRACT §5 |
| `_UNSET` sentinel, `data=` alias, `expiration_date` → `valid_to` (next-day midnight UTC), `reference_date` → `as_of`, `timestamp=` accepted, `show_expired`; `MemoryBase(ABC)`, `chat()`, `project`; `inspect.signature` parity test; `RESERVED_ATOM_KEYS == column set` test | P1 §E.1, §I | CONTRACT §17, §18 |
| Procedural branch (`agent_id and memory_type=='procedural_memory'`) → `procedural_summary` task → ONE `L1_micro` episode atom (never L2) | P1 §F.1.5 / §G.6 + judge fix | CONTRACT §6.3, §12.3, §12.6 |
| `LLMBase.generate_response(messages, tools=None, tool_choice='auto', **kwargs)`; import-guard test with `patch.dict(sys.modules, {'openai': None})` | P1 §H.2, test 25 | CONTRACT §10, §18 |
| `migrate.from_mem0()` helper | P1 §K.12 | CONTRACT §17.5 |
| Filter DSL resolution order axis → set axes → `json_extract(user_meta_json)`; `meta.<key>` escape hatch; `AxisQuery.meta_filters`; parametrized conformance test | P1 §E.1 | CONTRACT §13.1 |
| Closure builder (`open_actions[run]`, test_log closes the latest open action, error tool_calls self-close, `u` = intent or previous EP's `y`, derived constraint + `distill`, `use` to background constraints) | P3 §F.2 | CONTRACT §12.4 |
| `plan_query()` time hints (EN/ZH months, ISO, relative) → hard `time_anchor BETWEEN` gate with one fallback retry, identifier tokens ×2 in keyword, entity route with hard entity gate | P3 §F.3.1 | CONTRACT §13.2 |
| `ux_entities_norm_scope(name_norm, user_id, agent_id)` — `run_id` excluded | P3 §D | CONTRACT §7 |
| `tests/test_scenarios.py` with six golden scenarios parametrized over `['memory','sqlite']`; deterministic `chain`/`explanation` in `why()` | P3 §I/§F.5 | CONTRACT §18, §19 |
| Budget shares with `min_constraints=3` | P3 §F.4 | CONTRACT §15.2 |
| `why()` question mode (`target_id` not an id + `filters` → compile detective → constraint/episode/fact preferring identifier-entity matches) | P3 §F.5 | CONTRACT §14.3, §17.3 |
| Full-state checkpoints (`atom_state`, `edge_state`, `iep_state` over every object in scope) and symmetric rollback | P3 §C/§F.7 | CONTRACT §16.2 |
| `ScoringConfig` as its own sub-config; `collect` phase re-routes only extractable events without atoms | P3 §H, §F.9 + judge fix | CONTRACT §5, §16.5 |
| `score_details` = Mem0's 7 keys + `recency_score, support_score, keyword_backend, axis_gates, primary_axis, route, signals` + `recalled_by, pool_size, channel_sizes` (the recall channels of the row, the pool size, the per-channel sizes); `max_possible` driven by active signals, not hits | P1 §H.2/§F.3 | CONTRACT §13.4 |

### 3.2 Defects of the winner and how each is fixed

| Defect (judge) | Fix |
|---|---|
| `CHECK (scale='L1_micro' OR witness_edge_id IS NOT NULL)` forces every non-L1 edge to be a projection; distill/revision edges fabricated projection rows | CHECK is now `is_boundary = 0 OR witness_edge_id IS NOT NULL`; trigger writes `edge_projections` only when `witness_edge_id` is set; `project()` is the only writer of `is_boundary=1` edges; high `revision`/`distill` edges carry anchors only. `graph_stats.projection_coverage` = boundary edges with ≥1 projection row / boundary edges |
| `update()` returned a new id; `get(old)` stale; history split | Lineage id (graft above) |
| `instr()` CHECK on `content_json` | Dropped; pydantic validator only |
| `ux_events_identity` includes `timestamp` → "idempotent re-ingest" is clock-dependent | Kept **deliberately** (two utterances at different times are two events — One Event, One Anchor), and the test passes explicit `timestamp=` so it is not clock-dependent; an identical re-add at the same event time dedups (`dedup=True`, no new anchor, atom decision `NONE`, `graph_nodes.mentions` bumped; the `atom_anchor` rows are unchanged because the reused anchor already is the primary witness); the same text at a LATER event time creates a new event + anchor and the existing atom gains a `support` `atom_anchor` row |
| `history()` rows carried ~4 undocumented extra keys | Rows are Mem0's 10 keys **plus exactly one additive key `anchor_id`** (documented parity delta); HMS-only events (`ARCHIVE/RESTORE/FEEDBACK/CONFLICT/ROLLBACK`) are documented; `atom_id/edge_id/details_json` stay in the table, not the row |
| Cosine mapped as `(cos+1)/2` | `max(0.0, cos)` (Mem0 contract); `score_details.semantic_score` is comparable |
| Skill promotion needed ≥2 InterfaceEPs (two runs) | Invariant with ≥2 support nodes in one InterfaceEP suffices (the constraint used by both EPs in S4) |
| "switched to" ambiguous predicate → REVISE unreachable | Supersession cues map to predicate `likes` with `relation='supersession'`; the revision rule fires on `relation=='supersession'` + same subject + topic/objects overlap (predicate family, not predicate equality) |
| `IndexConfig` carried scoring knobs; `half_life_days` duplicated in `SchedulerConfig` | `ScoringConfig` owns weights/half-life/candidate cap; the index envelope is `{provider, config}` only |
| Entities keyed with `run_id` | `run_id` excluded from the entity key; FILE identifiers keyed by basename with the full path as alias |
| `meta.<key>` filters need JSON1 with no fallback | `capabilities()['json1']`; `plan_filters()` splits the AST into a SQL part and a Python-evaluated remainder when JSON1 is absent |
| `create_atom` in `ingest.py` called from `graph/support_graph.py` → import cycle | Atom materialization lives in `hms_elite/ingest/atoms.py` (group C); inline edges use `store.insert_edge` (group A); `graph/*` (E) imports C, never the reverse |
| Structured events became one EP each (`file_diff` with `y=''`), so *Failed Paths* was wrong | Closure builder (graft) |
| `HMSConfig.version` Literal `'lite-1'` + `extra='forbid'` rejected stock Mem0 dicts; per-provider configs rejected `top_p/top_k/openai_base_url` | `version ∈ {'v1.1','v2','lite-1'}`; `BaseLlmConfig(model, temperature, api_key, max_tokens, top_p, top_k, http_client_proxies)`, `OpenAILlmConfig(+openai_base_url)`, `AnthropicLlmConfig(+anthropic_base_url)`; unknown keys still fail with a `difflib` hint |
| LLM revision decision ran inside Txn B | `decide_revision()` runs unlocked between Txn A and Txn B; Txn B re-checks the target is still valid (else downgrades to ADD) |
| Prompt fields `supersedes_existing_ids`/`uses_recent_ep_ids` did not match models | Prompt contracts and model fields now share names: `supersedes_hint_ids`, `uses_ep_ids` (refs are resolved by the parser) |
| Governance obligation direction inverted | An item may be shown to audience `a` iff `rank(item.visibility) >= rank(a)` (lattice private<team<event<public); audience absent → obligation `skipped` |
| `derive_goal` disagreed with the example card | `compile()`: goal = the query; `handoff(goal=None)`: goal = `u` of the newest valid EP |
| Slot rule `scale_level == 'L3_macro' → skills` captured L3 InterfaceEP episodes | Rule is `source_kind == 'skill'` only |
| `transitions(fmt='records')` default vs fixed decision 2 | `fmt='tuples'` is the default |
| Feedback table had unfinished prose in `state_wrong`/`privacy_violation` rows | Table rewritten (CONTRACT §16.1); conflict resolution is witnessed (`resolved_by_anchor_id NOT NULL` whenever `resolved_at` is set; `resolution_edge_id` optional) |
| `extract_entities` excluded sentence-initial names ("Sarah joined…") | Sentence-initial capitalized tokens are PROPER unless in `STOP_CAPS` (explicit stop-list incl. months/weekdays/pronouns/common verbs) |
| Update/delete events overloaded `object_type='feedback'` | `object_type='system'` with `event_type ∈ {updated, deleted}`; `add_events()` rejects internal object types (`VALIDATION_007`) |

### 3.3 Defects of the other proposals that we must not inherit

- No `superseded_by` column on atoms (P3): the successor is found through the `revision` edge / lineage version. 
- No `deleted_at` twin flag (P3): `archived_reason='deleted'` on the one validity axis.
- No `event` atom + `episode` atom duplication in the card (P3): event atoms whose primary anchor is cited by a selected episode are suppressed (`explain.reason='covered_by_episode'`).
- No 60-newest candidate cap before ranking (P3), and no newest-first cap at all any more: candidates come from bounded multi-channel recall into a pool (semantic / keyword / entity / recency, each index-backed with its own K) that is scored in full; a gated set that fits the pool enters whole (CONTRACT §13.3, RETRIEVAL_DIAGNOSIS §3–§5). `scoring.max_candidates` is a deprecated alias that truncates nothing.
- No `GraphBase`/`GraphFactory` provider abstraction for a single implementation (P1): `SupportGraph` is a plain class over `StateStore` (the reranker did grow a second implementation, `overlap` + `llm`, so it has `RerankerBase` + `RerankerFactory`).
- No search-time support term from graph edges (P1): `support = tanh(n_undisputed_anchors / 3)` from `atom_anchor` only.
- No edge-only transitions (P1): `ep` rows are written at ingest.
- No `executescript` inside `BEGIN IMMEDIATE` (P1): migrations run statement-by-statement.
- No monkeypatching of `models.new_id` (P1): ids and clock are injected on the store.
- No `_hms_pin` inside caller metadata (P1): pinning is `atom_topic='important'` + `user_meta` untouched.

---

## 4. 包布局 Package layout (owner groups)

Flat layout at `/Users/lazy/Documents/Claude Code/hms-elite`: `hms_elite/` beside `tests/`, `examples/`, `pyproject.toml` (hatchling, `pydantic>=2.7` only core dep, extras `numpy/openai/anthropic/milvus/spacy/server/test/dev`, ruff 120, isort black). Eight owner groups with an acyclic dependency order:

```
A foundation  models, enums, exceptions, timeutil, configs (base, registry, prompts), utils/text, utils/json, memory/utils, memory/base, memory/schema, memory/storage
B providers   llms (base, rule, scripted, openai, anthropic), embeddings (base, hash, openai), index (base, sqlite, memory, milvus), entities (base, rule, spacy), reranker (base, overlap, llm, prompts), utils/factory
C ingest      ingest/events, router, extraction, closure, atoms, pipeline
D mvmi        mvmi/filters, planner, query; utils/scoring
E graph       graph/support_graph, operators, explain
F compiler    compiler/slots, compile, render, pc_ir
G ops         ops/feedback, checkpoint, handoff, transitions, consolidation
H facade      memory/main (HMS), __init__, cli, migrate
T tests       ONE conftest.py + fixtures + per-group test files + test_scenarios.py
X docs        pyproject.toml, README.md, examples/quickstart.py
```

Import rules (tested by `test_import_topology`): `A ← B ← C ← D`? No — D depends only on A, B. Exact edges: `B→A`, `C→A,B`, `D→A,B`, `E→A,B,C`, `F→A,B,D,E`, `G→A,B,C,D,E,F`, `H→all`. `mvmi/*` must not import `graph/*`, `compiler/*`, `ops/*`. The full tree with every file is CONTRACT §1.

---

## 5. 数据模型与不变量 Data model & invariants

- **Planes in one SQLite file** (`~/.hms_elite/hms.db`, `$HMS_DIR`, `':memory:'`): evidence (`source_events`, `anchors`), observation/state (`observations`, `eps`), State Index (`memory_atoms`, `atom_anchor`, `entities`, `atom_entity`, `atom_topic`, `atom_vectors`), Support Graph (`graph_nodes`, `graph_edges`, `edge_witnesses`, `edge_projections`, `interface_eps`, `interface_ep_support`), ops (`feedback_events`, `history`, `checkpoints`, `transitions`, `meta`). `PRAGMA user_version` migrations, `threading.Lock` + explicit `BEGIN IMMEDIATE/COMMIT/ROLLBACK`, `check_same_thread=False`, WAL for file DBs (switched on after the migrations so a fresh file writes its schema pages once; `StoreConfig.checkpoint_on_close=False` keeps the WAL at close on Python >= 3.12). A private `':memory:'` store clones a process-wide schema-only template with the backup API instead of re-running the migration ladder (~1 ms → ~0.02 ms per open; the template is never written, so instances share nothing). Schema version 2 (retention, below) is reached by `MIGRATIONS = {1: build_ddl(1), 2: upgrade_2()}`: existing files upgrade in place on open.
- **Ids**: `<prefix>_<12 hex>` from `store.new_id(prefix)`; prefixes `evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst`. Tests inject a per-prefix counter (`atm_000001`). `memory_id` (Mem0 `id`) uses prefix `mem`.
- **Clock**: `store.now()` → fixed-width UTC ISO (`isoformat(timespec='microseconds')`, `+00:00`) so TEXT order == time order. Three instants per atom: `time_anchor` (event time, or resolved date for `temporal_fact`), `created_at` (ingest time), validity window `valid_from/valid_to` (`valid_from` = event time so backdated imports work; `as_of` defaults to `now()`).
- **Scope**: `user_id/agent_id/run_id` denormalized on every plane table (≥1 required, CHECK).
- **Seven axes**: time (`time_anchor`), validity (`valid_from/valid_to` + `as_of`), entity (`atom_entity` ⋈ `entities`), topic (`atom_topic`), role, sentiment, scale (`scale_level`). `AxisQuery(filters, project_axis, scale_constraint, aggregation, limit, as_of, include_expired, scope?, text, meta_filters, plan)` executes HMS §3.5 semantics for real: `project_axis ∈ {atom, value, time, entity, topic}` produces a projection, `aggregation ∈ {list, latest, count, summary, topk}` is applied server-side, and the result is an `AxisResult(hits, projection, count, summary)`; exposed as `HMS.axis_query()`. `entity` atoms (one per entity, materialized by consolidate) and `relation` atoms (comparisons) make all seven `scale` values queryable.
- **Enums** exactly as fixed decision 6 plus `Role, Sentiment, Slot, ObjectType, EventType, NodeKind, SourceKind, Decision, HistoryEvent, Route, CompileMode, Relation, EntityType, MemoryType` — all `(str, Enum)`.
- **Retention escape hatch (CONTRACT §16.6)** — the store grows monotonically under every default path, by design (revision instead of deletion). The only way to shrink it is `HMS.purge(*, filters, older_than_days, dry_run=True, keep_witnessed=True)`: explicit (a call, never a side effect), default-off (`RetentionConfig.enabled=False` makes `dry_run=False` a `ConfigError`; `run_in_prune=False` keeps the scheduler out of it), audited (one `system(op='purge')` event + anchor per call, one `PURGE` history row per removed version carrying the text's sha256 instead of the text), and narrow (only state rows of archived versions that no edge, checkpoint snapshot, InterfaceEP, EP reference or multi-version lineage still needs; anchors and `source_events` are untouched unless `keep_witnessed=False`, and even then only anchors with zero remaining references). SQLite cannot disable a trigger per statement, so the seven `BEFORE DELETE` triggers carry a `WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1')` gate that only `StateStore.purge_rows()` opens, inside its own transaction; `HMS.vacuum()` reclaims the file afterwards. This keeps the *default* store semantics of §5 untouched while giving operators a documented, testable materialization path.
- **Invariants → tests**: (a) witness at type/DDL/store — the witness set is an *existence* condition (an object stays valid while ≥1 undisputed, unrevoked anchor remains; `anchor_ids[0]` is storage-primary, not necessary) and a projected edge additionally needs ≥1 valid projection source: decision record in `WITNESS_SEMANTICS.md`; (b) atoms carry no edges + import topology + trace; (c) `InterfaceEP` requires ≥2 support nodes, `primary_support_node_id NOT NULL`, boundary edges ⇒ `edge_projections`; `refine(project(coarsen(S)))` round-trip; (d) `refine()` is called only by `why()` and `compile(mode='deep'|'auto' with expansion)`; `mode='fast'` touches no graph table.

---

## 6. 算法 Algorithms (normative detail in CONTRACT §12–§16)

### 6.1 `add()`
1. Validate (Mem0 rules, `VALIDATION_001/002/003`, ids trimmed, reserved metadata keys stripped with one warning). Coerce messages; `system` role events are anchored but never extracted.
2. **Txn A**: one `SourceEvent(object_type='message')` + `Anchor` per message (dedup on `(payload_hash, timestamp, scope)`).
3. `infer=False` → raw `message` atoms (role/actor_id promoted, entities still linked). `infer=True` observation atoms also carry `actor_id` = the message `name`, else the LLM-named `speaker` (CONTRACT §12.5) — a deliberate departure from Mem0, which sets `actor_id` only on its infer=False path. Procedural branch → `procedural_summary` task → one `L1_micro` episode atom.
4. Unlocked: Phase-0 context (last 10 events, top-10 valid neighbours as `'0'..'n'`), `observation_extraction` LLM call, `ep_extraction` for `'state'|'both'` routed messages, embeddings, and `decide_revision()` per observation (candidates = valid state atoms with same subject and same predicate/topic/objects overlap; `revision_decision` LLM task; rule provider implements the table in CONTRACT §11.3).
5. **Txn B**: insert observation/EP rows → build atoms → apply decision (`REVISE`: archive old, new atom joins lineage v+1, `revision` edge old→new witnessed by the new anchor, history `UPDATE`; `CONFLICT`: new lineage, `conflict` edge, history `CONFLICT`; `NONE`: existing atom gains `support` anchor row, `mentions++`) → inline cheap edges for chat EPs (`temporal`, `use` from `uses_ep_ids`) → `ep` transition per EP → history `ADD`.
6. Return `{'results': [...]}`.

### 6.2 `add_events()` — structured closure (no LLM)
Events sorted by `(timestamp, index)`; Txn A anchors all; per event: an `event` atom always; `tool_call/file_diff` push onto `open_actions[run]` (error tool_calls self-close as negative EPs); `test_log` pops the latest open action (or `store.last_open_action(scope, run_id=run, before=outcome.timestamp, max_gap_hours=ingest.closure_max_gap_hours)` across batches — the newest uncited `tool_call`/`file_diff` within the gap) and builds one EP: `u` = action `intent` | previous EP's `y` | `modify {path}`; `b` = background constraint texts ∪ the action payload's own `constraints`; `a` = `patch {path}[: summary]` / `{tool}({args})`; `y` = `{suite} {status}[: failed_tests]`; `r` from status; `κ.constraints` = the same ordered union; anchors `[outcome, action]`. `sensor_trace` events yield an `event` atom only. Edges: `use` EP→each background constraint (witness action anchor); `temporal` prev→this; `causal` prev→this iff prev is negative or shares an entity or `u` was derived from prev's `y`; negative EP → derived constraint atom `Do not re-apply '{a}': {y}` + `distill` edge (witness outcome anchor). A `test_log` with no open action yields only its event atom (no EP without an action). At batch end, remaining open `tool_call`s self-close as `r=unknown` EPs only when their payload carries `constraints` (closure condition); otherwise they stay open, as do open `file_diff`s.

### 6.3 `search()`
`parse_filters` (a scalar `visibility` filter is the audience and lowers to the lattice gate `rank(item) >= rank(audience)`) → `plan_query` (route, time hints, entity hints, identifiers; preference questions get their own `preference` route with a hard `obs_type` gate, so HMS §4.1 preference recall is an axis selection) → SQL axis gates (scope, validity, filter AST, scale, hard time gate, hard entity gate for the entity route, hard type gate for the preference route; a four-level fallback ladder: drop the route's scale default → `primary_axis='scale_fallback'`, then the plan gates → `'fallback'`; an explicit `scale_constraint` is never relaxed) → bounded multi-channel recall into a pool (semantic ANN / FTS5 keyword / entity / recency, each with its own K, every channel searching inside the gated set; a gated set that fits the pool enters whole; CONTRACT §13.3, RETRIEVAL_BENCH_2026-09-10) → signals over the pool: semantic `max(0,cos)`, keyword overlap (identifiers ×2), entity overlap with hub damping, recency `exp(-ln2·age/half_life)`, support `tanh(n_anchors/3)` → fusion `Σw·s / Σw over active signals` → threshold on `final` **after** fusion → sort `(final DESC, time_anchor DESC, atom_id ASC)` → rows + `score` (+ `score_details`). Never reads a graph table.

### 6.4 `compile()`
Plan → axis candidates (`candidate_k=50`, threshold 0, through the same bounded pool recall as `search()`; `ContextIR.retrieval` carries the pool stats) → work-state augmentation (all valid episode + constraint atoms in scope) when route ∈ {work_state, detective} or `mode='deep'` or (`mode='auto'` and the scope has ≥1 valid EP) → work-state augmentation also unions valid `skill` atoms (and `L3_macro` is in the recall scale default, so distilled skills are reusable) → graph expansion 1 hop over `{causal, revision, conflict, use, distill}` (never in `fast`; in `axis_graph` and `deep`) with revision predecessors → `audit(reason='revised')`, unresolved conflicts recorded, InterfaceEP nodes refined (`hops=0`, `deep` only — `axis_graph` is the ablation without Support) for invariants → constraints and residuals → audit; audience gating drops private neighbours/invariants/evidence at selection time so a team handoff never trips `governance` on its own output → ordered slot rules (archived or expired-at-`as_of` → audit — a live atom with a future `expiration_date` stays in its natural slot; skill → skills; constraint → constraints; preference/habit → preferences; episode → episodes; else facts; event atoms covered by a selected episode dropped; entity atoms never enter a card) → synthetic κ items only when no equal-text constraint is selected or archived (one per text across all episodes) → goal = query → next action rule (newest negative vs newest positive episode, residual, conflict, fallback) → budget (shares, `min_constraints=3`, spill facts→episodes) → `ContextIR` (validator: disjoint item ids, every item witnessed). Read-only.

### 6.5 `why()`
Resolve `memory_id | atom_id | ep_id | obs_id | iep_id | edge_id`; otherwise, with `filters`, treat as a question (compile deep/detective; among constraints → episodes → facts prefer the item whose entity keys intersect the question's identifier tokens, e.g. `retry.ts`, else the first). Backward BFS `hops` levels over incoming `{causal, use, distill, revision, conflict}` + outgoing `use`; InterfaceEP nodes refined on demand; returns nodes, edges (with witnesses), anchors, interface_eps, paths, unresolved conflicts, deterministic `chain` (highest-confidence incoming edge first) and `explanation`.

### 6.6 `feedback()`
Prologue writes the feedback event + anchor, `feedback_events`, history `FEEDBACK`; then one handler per `FeedbackType` (CONTRACT §16.1). `state_missing` needs no existing target (`target_id=None` + a scope in `correction`, or a scope key string). Only `privacy_violation` changes visibility; `evidence_wrong` disputes anchors (never deletes); an atom left without an undisputed witness is archived, never un-witnessed. `connection_reason_wrong` with the same edge type only relabels + re-witnesses the edge (MERGE would otherwise resurrect the row it just invalidated). `HMS.revoke(anchor_id)` (or `evidence_wrong` with `correction={'revoke': True}`) sets `anchors.revoked_at` and cascades: derived atoms lose the witness (archived when none remains), edges whose witnesses are all revoked are invalidated, InterfaceEPs whose evidence is all revoked are invalidated — HMS §3.2 Revoke.

### 6.7 `checkpoint()` / `rollback()`
Checkpoint snapshots `{atom_id: {valid_to, archived_reason}}`, `{edge_id: valid}`, `{iep_id: valid}`, `{ep_id: valid_to}` for every object in scope (EP validity is maintained: archiving an episode atom cascades to `eps.valid_to`, so `list_eps`, `last_ep`, handoff goals and orient never resurrect a deleted EP). Rollback (one transaction, witnessed by a `rollback` event): archive atoms created after the checkpoint; restore atoms that were valid then; invalidate/revalidate edges and InterfaceEPs to the snapshot; insert a `revision` edge successor→restored labelled `rollback:<ckp>`; upsert a `rollback` transition.

### 6.8 `transitions()`
`ep` rows exist since ingest (`Transition.from_ep`, group A); export adds `interface_ep` rows (HMS §4.4: InterfaceEPs are transition sources), `causal` (valid), `revision`/`conflict` (any validity, excluding `rollback:` labels) and `rollback` rows; `κ_t` has one shape for every kind (`constraints, visibility, permission_scope, provenance`); `w_t` is an ordered union (first-occurrence order); idempotent on `(source_kind, source_id)`; default `fmt='tuples'`.

### 6.9 `consolidate()`
Scheduler event + anchor; `orient` (read-only plan incl. `last_compiled`), `collect` (re-embed missing vectors, re-route extractable events lacking atoms), `consolidate` (entity atoms materialized per entity, temporal edges, causal proposals (rule functions imported from `llms/rule.py`, or the LLM `edge_proposal` task), conflict detection, coarsen candidates per run → `coarsen` → `project` (L2), then L2 IEP components → L3 `coarsen` → `project`, skill promotion, transition export, and `compile_state` — a persisted Work State snapshot per scope in `compiled_states`), `prune` (decay = confidence step-down for stale atoms, then archive below `confidence_min`; never pinned, never conflicted, never constraints/skills/entities).

### 6.10 `axis_query()` / `revoke()` / `import_package()`
`axis_query(AxisQuery)` executes the seven-axis query with projections and aggregations (HMS §3.5). `revoke(anchor_id)` is the HMS §3.2 Revoke path with cascade. `import_package(StatePackage)` is the HMS §9.4 receiver: the package carries its `source_events`, `anchors`, `atoms`, `eps`, `interface_eps`, `edges`, and the receiver recreates them (ids preserved, identity dedup, optional re-scoping) so `compile()` on the receiver renders the same card.

### 6.11 `context()` — the hybrid read (2026-09-18)
`context(query, *, filters, top_k=None, as_of=None, budget_tokens=None) -> dict` (CONTRACT §6.7) returns `"<session card>\n\nRecent relevant memories:\n<rows>"`: the card is `compile(<text of the scope's latest extractable event>, as_of=<its timestamp>, budget_tokens=compiler.context_session_budget_tokens)` memoized per scope and dropped by the store's per-scope write version (`StateStore.scope_version`: bumped at the COMMIT of every transaction that stored an event or atom of the scope, by scope-less writes, rollback and reset — never a timer); the rows are `search(query, top_k=compiler.context_rows_top_k)` deduplicated (exact + proper-substring, case / whitespace-insensitive), sorted by `time_anchor` ascending and stamped `- [YYYY-MM-DD HH:MM] text` (`hms_elite/compiler/context.py`, the functions `benchmarks/egolife/reader_ab.py` imports). The product places the text before the question (`Memories:\n<text>\n\n<question>`): on EgoLife this read answers 70.1% against 54.9% for plain rows and 63.2% for the per-question card, and the memoized prefix is what a provider's prompt cache can hit (`docs/design/READER_AB_2026-09-18.md`). Unrelated to `AgentRuntime.context()`.

---

## 7. Coarsen / Project / Refine

For a node set `S` (valid L1 nodes, one scope): `E_in` = valid edges inside `S`, `E_b` = valid edges with exactly one endpoint in `S`. `cohesion(S) = 0.5·min(1, |E_in|/max(1,|S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|)`, `0` when `|E_in|=0`. `coarsen(S)` (rule, `rule-coarsen-v1`; input EP nodes for L2, InterfaceEP nodes for L3 — one operator, two scales): slots `u=first.u`, `b=' ; '.join(ordered_union(split(b_i, ' ; ')..., constraints))` (each constraint text once), `a=' → '.join(a_i)`, `y=last non-empty y`, `r=negative if any member is negative and the last member is not positive, else last.r`, `κ=union`, `κ.evidence = evidence_anchor_ids`; invariants = constraint strings in ≥2 members' κ (or in all), each completed with `support_node_ids`, `anchor_ids` (union of support anchors) and `confidence` (mean of support confidences); residuals = unresolved conflicts, missing outcomes, low cohesion (with `anchor_ids` likewise); `evidence_anchor_ids = ∪anchors(S) ∪ witnesses(E_in)` (checked equal). Persisted as an `episode` atom at the target scale (+ node), `interface_eps`, `interface_ep_support`. `project(iep)` MERGEs one high edge per `(outer', type)` with `is_boundary=1, witness_edge_id=<low edge>`; the trigger writes `edge_projections`. `refine(iep_id, hops)` is read-only and is the only way high-level objects are opened. Round-trip law: `set(refine(project(coarsen(S)))['boundary_edges']) == E_b(S)`.

---

## 8. 评分公式 Scoring

```
semantic = max(0, cos(q, v))                 (hash embedder: char 2/3-grams → sha256 buckets → 64-d L2)
keyword  = Σ_{t∈Q∩A} w(t) / Σ_{t∈Q} w(t)     w(t)=2 for identifier tokens, else 1
entity   = |Q_ent ∩ A_ent| / |Q_ent| · 1/(1 + 0.001·max(0, deg−3)²)     (hub damping starts above 3 links)
recency  = exp(−ln2 · age_days(time_anchor or created_at, as_of) / half_life_days)   (30)
support  = tanh(n_undisputed_anchors / 3)
final    = Σ_{k∈active} w_k·s_k / Σ_{k∈active} w_k      w = {semantic 1.0, keyword 0.5, entity 0.5, recency 0.25, support 0.25}
active   = {semantic, recency, support} ∪ {keyword iff query has ≥1 content token} ∪ {entity iff Q_ent ≠ ∅}
threshold applies to final (documented divergence: axis/identifier hits survive weak embeddings)
```
`score_details` maps to Mem0's keys: `semantic_score, bm25_score(=keyword), entity_boost, raw_score(=Σw·s), max_possible_score(=Σw active), final_score, threshold` + `recency_score, support_score, keyword_backend='overlap', axis_gates, primary_axis, route, signals` + `recalled_by` (channels that recalled the row: `['gate']` when the whole gated set was scored, else a subset of `semantic / keyword / entity / recency`), `pool_size`, `channel_sizes`.

Decay (prune): `staleness = (1−recency)·(1−clamp(confidence−0.6,0,0.4))·(1−use_recency)`; when `staleness > 0.8` the confidence is stepped down (`confidence − 0.1·staleness`, floor `confidence_min/2`); afterwards archive when `confidence < 0.4` and not pinned/conflicted/constraint/skill/entity — HMS §3.9's two operations (衰减, 归档) stay separate.

---

## 9. Prompt 契约 Prompt contracts

Six tasks in `hms_elite/configs/prompts.py`; each system prompt's first line is `HMS_TASK: <task>`; sections `# ROLE / # INPUTS / # GUIDELINES / # EXAMPLES / # OUTPUT FORMAT`; builders emit `## Section` blocks whose structured inputs are fenced JSON so the `rule` provider parses them back. Real ids are never shown (`'0'..'n'`, `n0..`, `w0..`); unknown refs are dropped (witness first). Contracts (exact JSON in CONTRACT §6): `observation_extraction → {"observations":[...]}`, `ep_extraction → {"eps":[...]}`, `revision_decision → {"decisions":[...]}` (candidates carry structured fields so nothing is re-parsed), `edge_proposal → {"edges":[...]}` (typed enum only, witness refs required), `coarsen → {"interface_ep":{slots, slot_support, invariants, residuals}}`, `procedural_summary → markdown` (Mem0 structure). Context rendering is deterministic (no prompt). Parse ladder copied from Mem0 with top-level-array salvage; transport errors → `LLMError(...) from e`.

Compact output contract (`ingest.extraction_output = 'compact'`, CONTRACT §6.5). Output tokens dominate write cost and write latency: on the real EgoLife run each 25-window batch made deepseek-flash emit a median 5,562 output tokens (cap 8,192; 22% of batches hit it and were bisected), all JSON, and nine of the seventeen observation keys restate `text`; output price is ≈ 4× cache-miss input and ≈ 200× cache-hit input. The compact tasks (`observation_extraction_compact`, `memory_extraction_compact`; both prompts under 0.75 of their full twins) ask for nine keys — id, event_idx, text (third person, ≤ 25 words), type, speaker, optional supersedes_hint_ids / when / valid_to, and an optional `entities` hint (the named things the statement mentions, 0–6 short strings) — and `hms_elite/ingest/derive.py` fills the rest from the text, the witnessed message and the configured entity extractor, reusing the rule LLM's own lexicons (`utils.text.sentiment`, `llms.rule.statement_triple` — the third-person statement parser built on the R1–R9 cue tables, which reads subject / predicate / objects back out of the text — `observation_topics`, `mention_texts`) so the two paths cannot drift; `entity_mentions` is the union of the `entities` hint, the extractor's hits in the text and the speaker (LLM order first, case-insensitive de-duplication), so the entity axis is never narrower than in full mode — a mention only the LLM knew (`python`) stays on it; every derived value still passes the full-contract coercion. EPs and decisions keep their shapes and are emitted only on their triggers. `full` is the default; `compact` is optional. It held the default for part of 2026-09-11: the same-18-batch real-API smoke of the final prompt (`benchmarks/egolife/results/real/elite_compact_entities_smoke5_r2`) measured a median 2,793 vs 5,631 output tokens per request and 0 vs 8 attempts at the cap, and the 204-question zero-API replay A/B kept hit@20 at 161 = 161 (`results/replay_ab_entities`) — but on the real 204-question run compact v1 scored 53.9% on memory rows (the index-paired NONE verdicts below) and compact v2, with the rewritten wording and the verdict guard, 56.4% (115/204; card 62.7%, hit@20 77.9% / 79.4%, ¥4.35 off-peak, 7.3 s per batch) against 61.3% (125/204; card 61.8%, hit@20 78.9%, ¥11.56, 14.9 s) for `full`, 4.9 points outside the 2-point tolerance, so the default was reverted the same evening (`docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`, CONTRACT §6.5 / changelog row 105). `compact` stays the recommendation for cost-sensitive bulk ingestion (0.38× write cost, 0.49× batch time) and for state-card readers; both prompts byte-identical either way. Decision guard (2026-09-11): the earlier compact DECISIONS wording made deepseek-flash pair observation i with candidate i and answer NONE (1,685 of 7,315 observations; 23 of 8,470 under `full`), each NONE discarding the statement; the prompt now states one decision per NEW observation with NONE reserved for a restatement, and the library honours a verdict only when the two atoms share a subject and, for a NONE or an uncorroborated REVISE, the containment of their content tokens after the head (subject, shared predicate) is removed is at or above `ingest.verdict_min_similarity` (0.5) with at least two shared tokens or the shorter statement contained entirely — the first cut counted the subject instead and let short statements sharing only a speaker and a verb through at exactly 0.5 (replaying the recorded compact replies: 151 of 204 hit@20; the head-removed guard reaches 154 = the force-ADD ceiling, MRR 0.5038, live atoms 5,717 → 7,299, while the recorded `full` replies keep 161 with 3 of 28 guarded verdicts downgraded; CONTRACT §6.3 / §12.6, changelog rows 103–104).

Lines output contract (`ingest.extraction_output = 'lines'`, CONTRACT §6.6, 2026-09-12). The compact contract halved the output but did not close the gap, and the cost accounting says where the gap is: on the 4,851-window real run the write cost of `full` was ¥11.5 with 2.40M output tokens — 83% of the cost — against mem0's ¥1.85 (input-miss 0.93M tokens + output 0.22M); compact v2 ¥4.2 with 0.76M output. The input-miss side is already at parity, so only output tokens can move the bill, and a JSON envelope costs 2–4× the text it carries. Two further measurements fixed the design: the zero-API replay showed retrieval uses only the statement texts and the decisions (entities / type / when and the other fields move hit@20 by 0), and the compact prompt's own texts scored below the full prompt's (rows 56.4% vs 61.3%; paired 32 vs 22) — the loss was in the writing rules, not in the dropped fields. So the lines task keeps the full prompt's statement rules verbatim (third person, 5–80 words, one self-contained statement, quantities / dates / times kept, the supersession-cue list, questions yield nothing) and the compact DECISIONS rules, and replaces only the OUTPUT FORMAT with tab-separated `OBS` / `EP` / `DEC` lines: `<event_idx>⇥<T>⇥<statement>[⇥@date]`, `<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`, `<OBS line>⇥REVISE|CONFLICT|NONE⇥<candidate>⇥<reason>` (a DEC line only for a non-ADD verdict; absence is ADD). No json mode (`LLMBase.generate_lines` sends no `response_format`), a tolerant parser (`ingest/lines.py`: fences, prose, a missing tab, an unknown letter, a truncated last line and an out-of-range DEC each cost one line at most), and after the parse the compact derivation path unchanged — the one lines-specific rule is that the format has no speaker slot, so the statement's leading proper name is the speaker (entity axis, `actor_id`). Single mode only; the rule LLM renders the same lines, so on the six chat fixtures the axis / history / graph / `eps` tables and the atom texts, subjects, predicates and objects are identical to `full`, and the `observations` / `memory_atoms` rows differ only in the pinned derived columns (`confidence` 0.7, `relation` / `supersedes_hint_ids` without a hint slot, that `actor_id`); the full and compact prompts stay byte-identical; `lines` is the library default since 2026-09-18 (CONTRACT changelog row 108; `full` before), see the next paragraph.

Measured (2026-09-12, `results/real/elite_lines_full204`, all 204 questions, deepseek-flash, off-peak): memory-rows context 54.9% (112/204), state-card context 63.2% (129/204, the highest of every arm); hit@20 84.3% / 82.4% (the highest); 51 output tokens per window (full 495, compact 157, mem0 45), median model time per batch 3.76 s (full 14.9, compact 7.3, mem0 3.5), write + answer re-priced at off-peak ¥2.24 (mem0 ¥1.98, compact ¥4.35, full ¥11.56); 12,994 atoms stored (2.68 per window, 72.5 MB); verdicts ADD 12,963 / NONE 20 / UPDATE 11; no truncation, no window lost. Conclusion: cost and write speed reach mem0's level with higher recall; rows-context accuracy is 4 points below full (72 evidence-present wrong answers), state-card accuracy 1.4 points above full. Default since 2026-09-18 (row 108): `lines`, because the reader A/B on the same lines store (`docs/design/READER_AB_2026-09-18.md` §六) measured 70.1% (143/204) for the hybrid read of §6.11 (`HMS.context()`: memoized session-card prefix + timestamped rows, memories first) against full's rows 61.3% / card 61.8%, sign test vs rows p < 0.0001; `full` / `compact` stay optional, and a `staged` config now names one of them explicitly.

---

## 10. Provider 层 Provider layer

`LLMBase(config)` with `generate_response(messages, tools=None, tool_choice='auto', **kwargs) -> str | dict` and `generate_json(task, system, user)`; providers `rule` (default), `scripted` (tests), `openai`, `anthropic` (import-guarded → `DependencyError` with pip hint). `EmbedderBase` with `embed/embed_batch/dims/identity`; `hash` (default), `openai`. `IndexBase` with `upsert/upsert_batch/search(candidate_ids)/get/delete/count/reset/info`; `sqlite` (BLOB in the same DB, numpy if importable) and `memory`. `EntityExtractorBase` with `extract(text) -> list[EntityCandidate]`; `rule` (default, byte-identical to `utils.text.extract_entities`) and `spacy` (optional `[spacy]` extra; NER spans are merged into the rule result, rule spans win overlaps, so regex-only FILE/IDENTIFIER candidates survive); selected by `ingest.entity_extractor`, one instance per facade shared by the Ingestor, the rule LLM (`entity_mentions`), feedback corrections and package import. `REGISTRY` in `configs/registry.py` is the single source of truth for both envelope validation and factories; `register_provider()` extends both. Config classes are pydantic `extra='forbid'` with a `difflib` "did you mean" hint.

---

## 11. 测试计划 Test plan

ONE `tests/conftest.py` (`clock`, `ids`, `store`, `rule_llm`, `scripted_llm`, `hash_embedder`, `hms`, `hms_scripted`, `events`, `golden`, `index_provider`, autouse `no_home_writes`). Fixtures are JSON event lists (`tests/fixtures/events/*.json`) and golden outputs (`tests/fixtures/golden/*`). Test files map 1:1 to owner groups (A–H) plus `test_scenarios.py` (S1–S6 on both index providers) — 63 named tests plus the post-review additions (`45b` L3 coarsen, `54b` handoff import round-trip, `56b` revoke cascade, `56c` entity atoms, `64` bilingual docstrings) in CONTRACT §18, each naming the public functions it exercises. Two golden helpers: `golden.text` (byte-for-byte: cards, chain) and `golden.masked` (ids + clock-era instants masked: JSON/JSONL). Rules carried from Mem0: mock the provider SDK, never the code under test; regression test per bug fix; `monkeypatch.setenv` only; no timing assertions; no network.

---

## 12. Demo `examples/quickstart.py`

Runs S1 (coffee preference → revision), S2/S3 (Sarah across runs, "what happened in May"), S4 (constraint + the failing `file_diff`/`test_log` pair → Work State Card with *Failed Paths* and a `Do not re-apply …` *Next Action*; the scenario test additionally adds the fixing pair and asserts a second card whose *Next Action* is `Continue from …`), S5 (`why("why must retry.ts not be modified")`), S6 (checkpoint → `state_outdated` feedback → a `compile()` WITHOUT `as_of` — the successor is valid only from the feedback instant, so the mid-scenario card must use the clock — → rollback → transitions), then `consolidate()` and `graph_stats()`. Offline, ~45 lines; the cards use explicit `as_of` and evidence lines carry fixture timestamps + payload sha256 prefixes, so the printed card equals `tests/fixtures/golden/work_state_card_failed.txt` on the real clock. Exact inputs and expected outputs are CONTRACT §19.

---

## 13. HMS → Lite → Mem0 映射表

| HMS concept (§) | HMS Elite object / call | Mem0 counterpart | Delta |
|---|---|---|---|
| SourceEvent (§3.1) | `SourceEvent`, `source_events`, `add_events()` | `messages` buffer + `save_messages` | first-class table, 5 extractable object types (incl. `sensor_trace`), `timestamp=` honoured |
| Anchor (§3.2) | `Anchor`, `anchors`, `atom_anchor`, `edge_witnesses` | payload `hash` (md5) + history row | sha256 event hash; every row references ≥1 anchor |
| Observation (§3.3) | `Observation` → `state|event|relation` atom | `ADDITIVE_EXTRACTION_PROMPT` memory text | typed/dated/witnessed; `linked_memory_ids` consumed as `supersedes_hint_ids` |
| EP (§3.4) | `EP`, `eps`, closure builder, `ep_extraction` | procedural memory string | six-tuple per closure; no LLM for structured events |
| MemoryAtom / AxisQuery (§3.5) | `MemoryAtom`, `memory_atoms` + axis tables, `AxisQuery`, `FilterExpr` | flat payload + `_process_metadata_filters` | validity in WHERE, entity/topic as M2M, lineage id keeps Mem0 `id` stable |
| Support Graph (§3.6–3.7) | `graph_*`, `SupportGraph`, `InterfaceEP`, coarsen/project/refine | (removed in v2; entity linking) | six typed edges, witnesses, projections |
| ContextIR / PC-IR (§3.8) | `compile()`, `ContextIR.to_text()/to_pcir()` | result formatting | six disjoint slots, obligations, `ProofObligationError` |
| StateScheduler (§3.9) | `consolidate(phases=)` | Phase 0–8 add pipeline | synchronous four phases, witnessed by a scheduler anchor |
| StateFeedback (§3.10) | `feedback(target, type, reason, correction)` | `update()/delete()` | 11 typed handlers, revision edges, `FEEDBACK` history |
| Transition (§4.4) | `transitions()` | `history()` replay | `(S_t,a_t,S_{t+1},r_t,κ_t,w_t)` from EPs, InterfaceEPs, edges and rollbacks; uniform κ_t |
| Handoff package (§9.4) | `handoff()` → `StatePackage`; `import_package()` on the receiver | — | receiver imports state (events, anchors, atoms, EPs, IEPs, edges), not history |
| Revoke (§3.2) | `revoke(anchor_id)` | `delete()` | cascade to derived atoms/edges/IEPs; evidence rows never deleted |
| Hybrid read (CONTRACT §6.7) | `context()` → memoized session card + timestamped rows, memories-first | `search()` rows formatted by the caller | per-scope memo invalidated by the store's write version; `compiler.context_*` fields; 70.1% vs 54.9% (rows) on EgoLife |
| add/get/get_all/search/update/delete/delete_all/history/reset/close | same names, same signatures, same envelopes | v2.0.19 | `add` may report `UPDATE`/`NONE`; extra promoted keys; `history` + `anchor_id` |
| config dict | `HMSConfig` accepts `vector_store/llm/embedder/history_db_path/version/custom_instructions/reranker` | `MemoryConfig` | + `store/ingest/scoring/graph/compiler/scheduler`; RENAMED/REMOVED tables |

---

## 14. 风险 Risks & open questions

1. **Rule extractor ceiling** — regex classification is tuned for EN/ZH cue lexicons and the fixtures; real prose yields misses (no EP) more than false EPs by design. Real providers drive identical prompts; `RuleLlmConfig.language` only selects lexicons.
2. **Hash embeddings are lexical** — semantic recall depends on character overlap; axis gates, keyword and entity signals carry ranking. Switching to `openai` embeddings requires `consolidate(phases=['collect'])` to re-embed.
3. **Cohesion formula and per-run components** may over-coarsen long runs; `coarsen_max_nodes=12` bounds it; L3 coarsening (IEP of IEPs) runs in consolidate step (4b) over connected L2 InterfaceEPs and is tested with two runs in one scope (`test_coarsen_l3_from_two_interface_eps`).
4. **Closure rule is greedy** — a `test_log` closes the latest open action; interleaved edits attribute the outcome to the last one. Seam: `ingest/closure.py::build_ep_from_closure`.
5. **Rollback is scope-local** — edges whose other endpoint is outside the scope are invalidated with the snapshot and reported.
6. **Governance is a lattice, not a policy engine** — the audience gate is applied at selection time (filters, graph neighbours, invariants, evidence anchors) and the `governance` obligation only re-checks κ omissions and anchor permissions; anchor `permission_scope` is consulted, revocation cascades, but there is no ACL evaluator.
7. **SQLite floor** — expression unique indexes with `COALESCE`, partial indexes, `RAISE` triggers, `json_extract` (JSON1) need SQLite ≥ 3.31 (Python 3.10 bundles ≥ 3.31 on all supported platforms); `capabilities()` records `json1` and the planner degrades to Python evaluation.
8. **History vocabulary** — HMS-only events can surprise `if row['event'] == ...` code; documented in the migration table.
9. **`why()` question mode overloads `target_id`** — kept to honour the fixed signature; `resolved_from` makes it explicit.
10. **Token estimate** `(utf8_len+3)//4 + 8` is crude for CJK (~0.75 tokens/char); acceptable for budgets.
11. **Concurrency** — one process per DB file (WAL + `BEGIN IMMEDIATE` + in-process lock); documented.
12. **`timestamp` in the event identity key** — re-adding the same message at a later wall-clock time creates a second event (by design); Mem0 users expecting silent dedup get a `NONE` decision plus an extra witness instead.
