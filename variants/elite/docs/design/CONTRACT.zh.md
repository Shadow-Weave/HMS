# HMS Elite — IMPLEMENTATION CONTRACT (实现契约)
English version: [CONTRACT.md](CONTRACT.md)

> 规范性文档。各自独立的 agent 仅依据本文件实现互不相交的归属组（A..H、T、X）。若 `DESIGN.md` 与本文件冲突，以本文件为准。下文出现的每一个签名、枚举值、列名、JSON 键、错误码和预期输出都是精确的；凡测试会断言的内容，一律不使用「可以（may）」这类措辞。
>
> 仓库：`/Users/lazy/Documents/Claude Code/hms-elite` — 扁平布局（`hms_elite/` 位于仓库根目录，`tests/` 与之并列），`pyproject.toml`（hatchling），Python `>=3.10,<4.0`，核心依赖**仅** `pydantic>=2.7`；可选依赖组 `numpy`、`openai`、`anthropic`、`milvus`、`spacy`、`server`、`test`（pytest、pytest-mock、httpx）、`dev`。ruff 行宽 120，isort profile 为 black。

---

## §0 约定（适用于所有组）

### §0.1 标识符、时钟、哈希

| 项 | 规则 |
|---|---|
| 对象 id | `store.new_id(prefix) -> f"{prefix}_{secrets.token_hex(6)}"`（12 位十六进制）。前缀：`evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst cst`。模型接受任意非空字符串（测试注入 `f"{prefix}_{n:06d}"`）；传给 writer 的非空 id 予以保留（导入路径）。 |
| Mem0 `id` | `MemoryAtom.memory_id`（前缀 `mem`）= **lineage id**，跨修订保持稳定。`atom_id`（前缀 `atm`）= 版本 id。`graph_nodes.node_id == atom_id`（每个 atom 版本对应一个节点）。 |
| 时钟 | `store.now()` 是本库中**唯一**的时钟。格式：定宽 UTC ISO `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`（32 字符），由 `timeutil.now_iso()` 产生；每一个入库的时刻都要经过 `timeutil.to_utc_iso()`，从而保证 TEXT 顺序 == 时间顺序。 |
| 每个 atom 的三个时刻 | `time_anchor`（事件时间；`temporal_fact` 取解析出的日期）、`created_at`（摄入时间 = `store.now()`）、有效期窗口 `valid_from`（= 事件时间）/ `valid_to`（NULL 表示仍然有效）。`as_of` 默认取 `store.now()`。 |
| 事件哈希 | `payload_hash = sha256(canonical_json({"object_type", "role", "actor_id", "payload"})).hexdigest()`，其中 payload 是**归一化后**的 payload（§4.3），`canonical_json = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`。`Anchor.hash == SourceEvent.payload_hash`。 |
| Atom 哈希 | `MemoryAtom.hash = md5(text.encode("utf-8")).hexdigest()`（与 Mem0 的 `row['hash']` 及去重逻辑对齐）。 |
| Scope | `Scope(user_id, agent_id, run_id)`；`Scope.key()` 等同于 Mem0 的 `_build_session_scope`：排序后的 `k=v` 用 `&` 连接，值做转义 `%→%25`、`&→%26`、`=→%3D`。每张平面表都反范式化保存这三列；DDL 的 `CHECK` 要求至少 1 列非 NULL。 |
| 可见性格 | `private(0) < team(1) < event(2) < public(3)`；`rank(v)`。当且仅当 `rank(item.visibility) >= rank(a)` 时，某条目可以展示给受众 `a`。 |
| Token 估算 | `estimate_tokens(text) = (len(text.encode("utf-8")) + 3) // 4 + 8`。 |
| 确定性 | 不用 `random`，不用 `hash()`（用 `hashlib`），除 `timeutil.now_iso()` 外不读挂钟时间，每个 `ORDER BY` 都以 id 作为最后的决胜键，每个**真正无序**的集合都按排序后迭代。**有序并集**保持首次出现顺序，永不排序：凡本契约写「ordered union」/ `ordered_union(...)`（§9.2）之处，结果都是各输入的拼接并丢弃后出现的重复项（用于 `w_t`、`kappa.constraints`、`b`、topic 列表、派生对象的 `anchor_ids`）。 |

### §0.2 错误码（单一表；`hms_elite/exceptions.py`，§3）

`VALIDATION_001` 缺少 scope id · `_002` memory_type 非法 · `_003` messages 类型非法 · `_004` 时间戳/日期非法 · `_005` 事件 payload 格式错误 · `_006` 未知的 feedback 类型 · `_007` add_events 中缺少必需的 correction / 使用了内部 object_type · `_008` 读取路径出现顶层 ids（同时 is-a `ValueError`） · `_009` threshold/top_k 非法 · `_010` 空查询 · `_011` update 未提供 text/metadata/expiration_date · `_012` filter 的键或运算符非法 · `_013` phases/mode/fmt/aggregation 字面量非法（也包括 `aggregation='topk'` 但未提供 `text`） · `NOT_FOUND_001` 未知 id（is-a `ValueError`） · `CONFIG_001` pydantic 配置错误 · `CONFIG_002` 未知 provider · `CONFIG_003` 已移除的配置键 · `LLM_001` 传输层 · `LLM_002` 未知 task marker · `LLM_003` scripted 队列耗尽 · `EMBED_001` · `STORE_001` sqlite 失败 · `STORE_002` 数据库版本高于库版本 · `STORE_003` 不变式触发器被触发 · `WITNESS_001` · `PCIR_001` · `DEPS_001` · `INTERNAL_000` HTTP 服务器背后未被映射的异常（§17.7；普通 `HMSError`，消息脱敏）。

### §0.3 双语 docstring 格式（HMS §5.2 原文照搬）—— 对**每一个**代码文件（`hms_elite/**/*.py`、`tests/**/*.py`、`examples/*.py`）以及每一个公开函数/方法强制要求

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

精确的排版规则（由 `test_bilingual_docstrings` 断言，§18.3 #64）：模块 docstring 为两行（先 EN，后 ZH）；紧随模块 docstring 之后是 `# Purpose/# 目的`、`# Input/# 输入`、`# Output/# 输出`、`# State effect/# 状态影响` 这四对注释，每对之间用一个空行分隔；每个公开函数的 docstring 以一行 EN 开头、紧跟一行 ZH，然后一个空行，然后按此顺序出现三个标记 `Input / 输入：`、`Output / 输出：`、`State effect / 状态影响：`；在 `Input / 输入：` 之下，每个参数写一行 EN `<arg>: ...`，紧接着一行 ZH `<arg>：...`（全角冒号），参数之间用一个空行分隔（无参数的函数写 `        none.` / `        无。`）；在 `Output / 输出：` 和 `State effect / 状态影响：` 之下，各恰好一行 EN 后跟一行 ZH。私有辅助函数（`_name`）只需要那两行 EN/ZH 摘要。

本契约中，每个函数都以 `signature` 列出，后跟 `— EN | 中文`（用途），然后是 `In:`、`Out:`、`Effect:` 三行（每行都是 `EN | 中文`），最后是 `Notes:`。实现者需把它逐字展开成上面的块（每个参数一对 `Input` 行）。代码注释一律只用英文。

### §0.4 导入拓扑（由 `tests/test_foundation.py::test_import_topology` 测试）

允许的边：`B→A`、`C→A,B`、`D→A,B`、`E→A,B,C`、`F→A,B,D,E`、`G→A,B,C,D,E,F`、`H→all`、`T→all`。`hms_elite/mvmi/*` 和 `hms_elite/utils/scoring.py` 不得导入 `hms_elite.graph`、`hms_elite.compiler`、`hms_elite.ops`、`hms_elite.ingest`。除 `llms/openai.py`、`llms/anthropic.py`、`embeddings/openai.py`（采用 Mem0 风格的守卫，抛出 `DependencyError`）外，任何模块都不得在模块导入阶段导入可选 SDK；`index/milvus.py`、`entities/spacy.py` 与 `server/app.py` 在构造函数 / `create_app()` 内部惰性导入各自的 SDK，因此 `hms_elite.index`、`hms_elite.entities`、`hms_elite.server` 没有 extra 也能导入。`hms_elite/server/*`、`hms_elite/hms4eye/*` 与 `memory/async_main.py` 只包装门面，与 `cli.py` 同属 H 组；`hms_elite/reranker/*` 与 `hms_elite/entities/*` 是 B 组的提供者包（B → 仅 A）。组 C 从不导入 `hms_elite.mvmi` 或 `hms_elite.ops`：C 需要从这两个平面取用的一切（`MemoryAtom.to_public_dict`、`validate_payload`、`Transition.from_ep`、`store.valid_ids`）都放在组 A 中（§4、§8）。组 G 导入组 B 的纯规则函数（`hms_elite.llms.rule.propose_edges / decide_revision / EXCLUSIVE / PREDICATE_FAMILIES`，§11），而不是重新实现那些表。

### §0.5 有意排除在范围之外的 HMS 特性（规范性范围表）

| # | 完整 HMS 特性（HMS_arch §） | Lite 的决定 | 理由 |
|---|---|---|---|
| 1 | `daemon.py` 后台循环（§3.9、§5.1） | 已发布为 `hms_elite/daemon.py::StateDaemon(hms, *, interval_s, phases, filters)`——在命名线程上按间隔运行 `consolidate()`，`run_once()` 返回阶段报告，异常记日志后循环继续（`tests/test_daemon.py`）；`consolidate()` 本身仍是同步调用，`compile_state` 仍是 consolidate 的第 (7) 步 | 后台循环只是对同步 consolidate 的调度 |
| 2 | `runtime.py` agent 适配器 / action registry（§5.1） | 已发布为 `hms_elite/runtime.py`（`AgentRuntime`、`ActionRegistry`、`ActionSpec`、`PendingAction`）：工具调用及其结果映射为 `file_diff / tool_call / test_log` 事件并经 `add_events()` 闭合；`why / handoff / checkpoint / rollback / feedback` 直接委托门面（`tests/test_runtime.py`） | `add_events()` 仍是底层契约 |
| 3 | 各来源专用适配器（slack/jira/feishu/sensor）（§3.1） | 简化 → 统一为一个 `SourceEvent`，配 5 种可抽取的 `object_type`：`message, tool_call, file_diff, test_log, sensor_trace` | 属于传输层胶水，不是不变式 |
| 4 | `segmentation.py`（§5.1） | 已发布为 `hms_elite/segmentation.py`（`segment`、`segment_messages`、`protected_spans`、`detect_lang`）：句子、段落、代码块、列表项，带精确区间；代码块、URL、标识符不被切断；确定性（`tests/test_segmentation.py`） | 规则切分，不依赖模型 |
| 5 | `coverage.py`/`scc.py` 层级式覆盖编译器（§5.1） | 已发布为可选层：`compiler/coverage.py`（`extract_requirements`、`covers`、`select_for_coverage`——贪心集合覆盖）与 `compiler/scc.py`（`compile_hierarchical` → `HierarchicalIR`：先 L3/L2 接口、按需下钻 L1；`lower()` → 带覆盖义务的 `HierarchicalPCIR`；`tests/test_coverage_scc.py`）；`compile()` 默认仍是每个 slot 的份额预算 + 约束下限 | 属于精度特性，不是不变式 |
| 6 | 视觉 / 感知 | 简化 → 无离线感知：list 类内容降级为文本 part，图片 part 以 `parts`（sha256 + mime，绝不含字节）落锚并计入 `dropped_parts`；`llm.enable_vision=True` 时，带视觉的提供者通过 `describe_image` 钩子（无锁、事务 A 之前）把每个 part 描述为一行 `[Image: …]` 并入 `text`，记录 `described_parts` 并从 `dropped_parts` 中扣除（`openai` / `anthropic` 实现；`rule` 返回 None） | 感知是提供者能力，不是库能力 |
| 7 | FTS5/BM25 词法轴 | 拆分 → 召回顺序来自标准库 SQLite 的 FTS5 表 `atom_fts`（bm25，`keyword_candidates`，无模块时 LIKE 退路）；池内关键词**分数**仍是纯 Python 的 token 重叠，标识符 token 权重 ×2 | 零可选依赖；有界召回（RETRIEVAL_DIAGNOSIS §3.2） |
| 8 | Reranker provider | 简化 → `overlap`（默认，离线的加权词元覆盖率）+ `llm`（`rerank` 提示词契约；`LLM_002` 时回退 overlap）；`search(rerank=True)` 在融合后套用 `self.reranker`（未配置 → 默认 `OverlapReranker`），各行追加 `rerank_score`，重排器失败时保持检索顺序 | Mem0 kwarg 对齐，外加一个离线默认 |
| 9 | 按 topic 的半衰期学习器 | 简化 → 单一的 `scoring.half_life_days` | 确定性 |
| 10 | embedding 层级的实体消解 | 简化 → 归一化文本精确匹配 + 别名列表；`run_id` 不参与实体身份 | 哈希 embedding 不具语义 |
| 11 | Personal/Edge/Group 渲染器（§11.2） | 已发布为 `hms_elite/state/`（`personal / edge / group / enterprise` 状态种类、确定性渲染器、`StateAPI`；`tests/test_state.py`）；`compile()` / `handoff()` 的输出仍是 Work State Card + `StatePackage` | 每种状态都带见证，渲染确定性受测 |
| 12 | LLM coarsen | 简化 → 默认使用 `rule` coarsen；LLM coarsen 走同一套 prompt 契约 | 可复现的往返测试 |
| 13 | 技能提升（§9.7） | 简化 → 一个 kind 为 `constraint`、支持节点 ≥2 的 InterfaceEP 不变式，转为一个 `L3_macro` 技能 atom + 可追溯的 `distill` 边 | 一次运行内即可达到 |
| 14 | 面向 κ 的治理/ACL 求值器（§3.8） | 简化 → 选择阶段的可见性格闸门 + `governance`/`not_revoked` 义务 | 为策略引擎预留钩子 |
| 15 | 基准测试框架（§12 Phase 5） | 已发布为 `hms_elite/eval/`（`run_ablation`、`Task`、`TASKS`——§10 的九个任务——、`AblationReport`、`render`、`to_json`；LongMemEval 适配器在数据集存在时启用；`tests/test_ablation.py`）；四种消融配置仍映射到 `CompileMode`：`fast` = 仅 Axis，`axis_graph` = Axis + Graph（做扩展，不做 `refine`），`deep` = Axis + Graph + Support（`refine` → 不变式/残差）；「仅 Graph」没有对应项（图只能从 axis 种子出发到达），记为不适用；`retrieval['ablation'] ∈ {'axis','axis_graph','axis_graph_support'}` | 包内自带；数据集不随包分发 |
| 16 | HTTP server | 已发布为 `hms_elite/server`（`create_app()` / `hms-server`，可选 extra `server`，与 `cli.py` 同属 H 组）；`cli.py` 照常发布 | 库优先，服务面只包装门面 |
| 17 | 外部索引 provider | 简化；`IndexBase` 可插拔，核心随包提供 `sqlite` + `memory`，`milvus`（milvus-lite `.db` 文件或服务器 uri）放在可选 extra `milvus` 之后（注册表条目 + `EXTRA_OF_PROVIDER`）；命中经 `rank_hits` 重算分数，各提供者分数逐位一致 | 既定决策 3（核心只依赖 pydantic） |
| 18 | 异步孪生、遥测、通知、`setup_config()` | 异步孪生与遥测已发布：`AsyncHMS`（`memory/async_main.py`，`asyncio.to_thread` 孪生，签名逐一相同，§17.8）与 `hms_elite/telemetry.py`（`Usage`、`UsageMeter`、`trace()`、`usage_of(hms)`：LLM / 嵌入提供者的调用次数、token 与时长；`GET /usage` / `GET /metrics`，§17.7）；通知与 `setup_config()` 仍砍掉 | 既定决策 9 只约束 consolidate 的同步语义 |
| 19 | Mem0 `chat()` / `project` | `chat()` 抛 `NotImplementedError`；`project.update()` 抛 `ValueError` | Mem0 API 形状对齐 |
| 20 | Mem0 `history.db` 导入 | 砍掉（由 `hms_elite.migrate.from_mem0` 取代） | 不做列集合差异迁移 |

HMS_arch §3 中未列入上表的一切内容都**在**范围内：七条查询轴及其可执行的 `AxisQuery` 语义（§13.3）、entity/relation atom（§16.5 step 0）、L2 **与** L3 InterfaceEP（§16.5 step 4/4b）、带级联的 anchor revoke（§16.1）、state-package 导入（§16.3）、InterfaceEP transition（§16.4）、编译态快照（§16.5 step 7）。

---

## §1 文件树与归属组（每个文件唯一归属）

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
│   │   └── milvus.py                               B   §10.3 MilvusIndex（可选 extra `milvus`；惰性导入 `pymilvus` → DependencyError）
│   ├── entities/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   §10.5 EntityExtractorBase + merge_candidates
│   │   ├── rule.py                                 B   §10.5 RuleEntityExtractor（默认；== utils.text.extract_entities）
│   │   └── spacy.py                                B   §10.5 SpacyEntityExtractor（惰性导入 `spacy` → DependencyError）
│   ├── reranker/
│   │   ├── __init__.py                             B
│   │   ├── base.py                                 B   RerankerBase、BaseRerankerConfig、rank_rows（Mem0 的 rerank 形状）
│   │   ├── overlap.py                              B   OverlapReranker（默认；离线加权词元覆盖率）
│   │   ├── llm.py                                  B   LLMReranker（`rerank` 提示词契约；LLM_002 时回退 overlap）
│   │   └── prompts.py                              B   RERANK_PROMPT + build_rerank_prompt（§6.1 形状）
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
│   ├── memory/async_main.py                        H   AsyncHMS（HMS 的 asyncio.to_thread 孪生）
│   ├── daemon.py                                   H   StateDaemon（门面之上的后台 consolidate 循环；鸭子类型，不导入任何平面）
│   ├── runtime.py                                  H   AgentRuntime / ActionRegistry（门面之上的 agent 适配器；鸭子类型）
│   ├── segmentation.py                             A   segment / segment_messages / protected_spans / detect_lang
│   ├── telemetry.py                                A   Usage / UsageMeter / MeteredProvider / trace / usage_of（§17.8）
│   ├── eval/                                       H   run_ablation、Task / TASKS、AblationReport、render / to_json（§0.5 第 15 行）
│   ├── state/                                      H   状态种类、确定性渲染器、StateAPI（§0.5 第 11 行）
│   ├── server/                                     H   FastAPI 服务面（`create_app`、`hms-server`；可选 extra `server`）
│   ├── hms4eye/                                    H   基于门面的 hms4eye 能力适配层
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

组之间的依赖顺序（无环）：**A → B → C → D → E → F → G → H**，具体允许的边见 §0.4（D 不依赖 C；E 因为 `ingest.atoms.create_atom` 而依赖 C）。

每一个代码文件 —— 各模块、每个 `__init__.py`、每个 `tests/*.py`、`tests/conftest.py`、`examples/quickstart.py` —— 都以 §0.3 的模块 docstring + 头部注释开始。每个 `__init__.py` 只重新导出其对应章节所列的公开名字，不多不少。

---

## §2 枚举（`hms_elite/enums.py`，组 A）

所有枚举都是 `class X(str, Enum)`；除非另有说明，成员名 == 值。共享的 `StrEnum` 基类上有 classmethod `values(cls) -> list[str]`；DDL 的 `CHECK` 列表由这些枚举生成（`schema.py` 导入 enums）。

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

`EXTRACTABLE_OBJECT_TYPES = {message, tool_call, file_diff, test_log, sensor_trace}`（五个成员）；`INTERNAL_OBJECT_TYPES = {feedback, checkpoint, rollback, system, scheduler}`（被 `add_events` 拒绝，`VALIDATION_007`）。`WHY_EDGE_TYPES = {causal, use, distill, revision, conflict}`（`why()`/`compile` 扩展的默认集合；不含 `temporal`）。`VISIBILITY_RANK = {private: 0, team: 1, event: 2, public: 3}`。`KIND_ORDER: dict[str, int] = {ep: 0, interface_ep: 1, causal: 2, revision: 3, conflict: 4, rollback: 5}`（transition 排序用；归组 A 所有，由 §8 的 `list_transitions` 和 §16.4 使用）。`ABLATION_OF_MODE = {fast: 'axis', axis_graph: 'axis_graph', deep: 'axis_graph_support'}`（`auto` 在运行时解析为 `fast` 或 `deep`，§15.2）。

---

## §3 异常（`hms_elite/exceptions.py`，组 A）

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

规则：`hms_elite.exceptions.ValidationError` 沿用 Mem0 的属性名，因此从 Mem0 移植过来的 `except ValueError` 仍然有效；只有在 Mem0 抛裸 `ValueError` 的地方才抛裸 `ValueError`，且在 HMS 中它始终是 `ValidationError` 子类的实例（这样 `except ValueError` 和 `except HMSError` 都能捕获）。消息以 `HMS_INVARIANT` 开头的 `sqlite3.IntegrityError` → 映射为 `StoreError(STORE_003, details={"trigger": <text>})`。witness 检查是**显式的前置检查，绝不靠解析消息**：每一个接收 anchor id 的 `StateStore` writer（`insert_atom`、`insert_edge`、`insert_observation`、`insert_ep`、`insert_interface_ep`、`upsert_transition`、`add_history`、`insert_feedback`、`insert_checkpoint`、`add_atom_anchor`、`add_edge_witness`）都先拒绝空列表（`WitnessMissingError(details={'object': <kind>, 'id': <id or None>, 'missing': []})`），然后调用 `get_anchors(ids)`；任何未被返回的 id → 在构造任何 SQL 之前抛 `WitnessMissingError(details={'object', 'id', 'missing': [unknown ids]})`。作为最后的兜底，列名以 `anchor_id` 结尾的 `NOT NULL constraint failed: <table>.<col>` 错误（仅限裸 SQL 路径）被映射为 `WitnessMissingError`；SQLite 裸的 `FOREIGN KEY constraint failed` 永不作解释。

---

## §4 数据模型（`hms_elite/models.py`，A 组）

模块 docstring：`"""Pydantic models for every HMS Elite object; witnesses are types, not conventions.\nHMS Elite 全部对象的 pydantic 模型；witness 是类型而非约定。"""` 职责：拥有全部承载 schema 的对象。输入：字典 / 数据库行。输出：已校验的模型。副作用：无（纯函数）。

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
    anchor_ids: NonEmptyIds                                       # anchor_ids[0] 是主见证——存储层的非空保证（NOT NULL 列）与默认的 history 见证，不是必要条件：见证集是存在条件，只要还剩任一未争议、未撤销的锚点对象就有效（§16.1 撤销；docs/design/WITNESS_SEMANTICS.md）


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
    speaker: str | None = None                                    # 多说话人转录中的具名人物（普通对话为 None）；事件无 actor_id 时作为原子的 actor_id
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
    retrieval: dict[str, Any] = {}                                # 排序查询胜出阶梯级别的 Pool.stats()（仅轴 / count 为 {}）：pool_size、channel_sizes、channels_used、gate_count（原生下推时为 -1）、whole


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

### §4.1 `MemoryAtom.to_public_dict` —— Mem0 行（规范实现位于 `models.py`，A 组；`mvmi/query.py::atom_to_row(atom, *, score, score_details, anchor_ids)` 的实现就是 `return atom.to_public_dict(...)`；任何位置都不做延迟导入）

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

`metadata` 只包含调用方的 `user_meta` 键。行的键顺序与上面完全一致。`anchor_ids` 默认取 `self.anchor_ids`（由读取方补齐，见 §8），除非调用方传入了无争议的列表。

### §4.2 `MemoryAtom.content` 的形状（按 `source_kind` 区分）

| source_kind | atom_type / scale_level | content 键 |
|---|---|---|
| message | message / L1_micro | `text, role, actor_id, dropped_parts` |
| observation（fact/preference_event/habit/constraint） | state / state | `text, obs_id, obs_type, subject, predicate, objects, relation, date_observed` |
| observation（temporal_fact） | event / event | 同上，另加 `event_time`（== time_anchor） |
| observation（relation='comparison'，额外产出） | relation / relation | `text, subject, predicate, objects, relation`（第二个 atom：`source_id` 为同一个 obs_id，`text` = `"Comparison: " + text`，`source_kind='observation'`）—— 仅当 `ingest.emit_relation_atoms` 为 True 时产出（默认 **True**） |
| entity | entity / entity | `text (== canonical_text), entity_id, entity_type, aliases: [str], mentions: int` —— 由 consolidate 的步骤 (0) 物化（§16.5），`source_id = entity_id`，`subject = canonical_text`，`predicate = 'is_entity'`，`anchor_ids = sorted(∪ anchors of the valid non-entity atoms linked to the entity)`，`scope = Scope(user_id=entities.user_id, agent_id=entities.agent_id)`（实体身份 scope；`run_id` 为 None） |
| event | event / event | `text, event_id, object_type, payload`（归一化后的 payload） |
| ep | episode / L1_micro | `text, ep_id, slots`（Slots.model_dump()） |
| interface_ep | episode / L2_meso 或 L3_macro | `text, interface_ep_id, slots, cohesion_score, invariants: [text], residuals: [text]` |
| skill | state / L3_macro | `text, kind: 'skill', invariant, source_interface_ep_ids, support_node_ids` |
| procedural | episode / L1_micro | `text (markdown), memory_type: 'procedural_memory'` |
| feedback（state_missing / next_action_wrong / render_* 所创建的 atom） | state / state | 与 observation 的形状相同；`obs_type ∈ {constraint, fact, preference_event}`；`source_kind='feedback'`，`source_id=feedback_id`，`obs_id` 这个键承载的是 feedback_id |

### §4.3 `SourceEvent.payload` 的归一化形状（`hms_elite/models.py::validate_payload`，A 组 —— 唯一的归一化器；`ingest/events.py` 只是把它再导出；`payload_hash` 与每一个 anchor 的身份都严格依赖这些字典）

| object_type | 归一化后的 payload（这些键始终存在，且按此顺序排列） |
|---|---|
| message | `{"text": str, "dropped_parts": int}` + 可选 `"parts": [{"kind": "image", "source": "data"|"url", "sha256": str, "mime": str|None, "url": str|None, "detail": str|None}]` 与可选 `"described_parts": [{"index": int, "sha256": str, "text": str, "described_by": str|None}]`——两键仅在非空时存在（纯文本载荷保持历史形状与哈希）；part 的 `url` 以 `data:` 开头 → `VALIDATION_005`（字节只以 `sha256` 标识） |
| tool_call | `{"tool": str, "args": dict, "result": str|None, "status": "ok"|"error"|None, "target": str|None, "intent": str|None, "constraints": [str]}`（`arguments` 可作为 `args` 的别名；非 str 的 result → `json.dumps(result, ensure_ascii=False)`） |
| file_diff | `{"path": str, "patch": str, "summary": str|None, "intent": str|None, "commit": str|None, "constraints": [str]}` |
| test_log | `{"suite": str, "status": "passed"|"failed", "passed": int|None, "failed": int|None, "failures": [{"name": str, "message": str|None}], "log": str|None, "related_paths": [str]}` —— 缺省时推导 `status`：`failed > 0` 或 failures 非空 → `failed`，否则 `passed`；字符串形式的 failure → `{"name": s, "message": None}`；`failed_tests` 可作为 `failures` 的别名 |
| sensor_trace | `{"sensor": str, "series_ref": str|None, "summary": str|None, "samples": [{"t": str (UTC ISO via to_utc_iso), "v": Any}]}` —— 路由到 `{'structured'}`，只产出一个 `event` atom（绝不产出 EP，也绝不产出 action）；`default_raw_ref → f"sensor://{sensor}"`；role 默认为 `tool` |
| feedback | `{"target_id": str, "feedback": str, "reason": str|None, "correction": Any}` |
| checkpoint | `{"label": str|None}` |
| rollback | `{"checkpoint_id": str, "label": str|None, "reason": str|None}` |
| system | `{"op": "update"|"delete"|"delete_all"|"import"|"revoke"|"purge", "memory_id": str|None, "atom_id": str|None, "text": str|None, "metadata": dict|None, "expiration_date": str|None, "package_id": str|None, "anchor_id": str|None}`（`import` 用 `package_id`，`revoke` 用 `anchor_id`） |
| scheduler | `{"phases": [str]}` |

缺少必需的键或类型不对 → `ValidationError(VALIDATION_005, details={"object_type", "field", "index"?})`。

---

## §5 配置（`hms_elite/configs/base.py`、`configs/registry.py`，group A）

### §5.1 配置字典的形状（`HMS.from_config(dict)` 接受什么）

```python
{
  # Mem0-compatible keys (same names, same nesting)
  "vector_store": {"provider": "sqlite" | "memory" | "milvus", "config": {...}},          # index envelope (Mem0 name kept)
  "llm":          {"provider": "rule" | "scripted" | "openai" | "anthropic" | "deepseek" | "ollama", "config": {...}},
  "embedder":     {"provider": "hash" | "openai" | "ollama" | "fastembed", "config": {...}},
  "history_db_path": "/path/hms.db" | None,                                     # → store.path (warn if store.path also set and differs)
  "reranker": None | {"provider": "overlap" | "llm", "config": {...}},          # 种类为 'reranker' 的 ProviderEnvelope；None = 未配置（search(rerank=True) 此时使用默认 OverlapReranker）；未知 provider → CONFIG_002
  "version": "v1.1" | "v2" | "lite-1",                                          # drives nothing
  "custom_instructions": str | None,
  # HMS keys
  "store":     {"path": str, "busy_timeout_ms": 5000, "journal_mode": "wal" | "delete", "checkpoint_on_close": True},
  "ingest":    {"revision_mode": "inline" | "deferred" | "off", "neighbourhood_k": 10, "last_events_k": 10,
                "emit_relation_atoms": True, "closure_max_gap_hours": 24,
                "entity_extractor": "rule" | "spacy", "entity_extractor_config": {},   # §10.5；未知名称 → CONFIG_002（details kind='entity_extractor'）
                "extraction_mode": "single" | "staged", "extraction_output": "lines" | "full" | "compact",
                "segment_max_chars": 400, "verdict_min_similarity": 0.5, "zh_rules": False},   # zh_rules：中文陈述规则，§9.6；extraction_output 自 2026-09-18 起默认 'lines'（§6.6，变更记录第 108 行）；'staged' 必须显式给出 'full' | 'compact'（after 校验器拒绝 lines + staged → CONFIG_001）
  "scoring":   {"w_semantic": 1.0, "w_keyword": 0.5, "w_entity": 0.5, "w_recency": 0.25, "w_support": 0.25,
                "half_life_days": 30.0, "max_candidates": None, "identifier_weight": 2.0},   # max_candidates：已弃用，为兼容而接受，不再截断任何东西；默认是 None 哨兵（model_dump 往返不出声），非 None 的值发出 DeprecationWarning
  "retrieval": {"k_semantic": None, "k_keyword": None, "k_entity": None, "k_recency": None, "pool_max_factor": 4,
                "hard_gate_pool": True, "safety_limit": 50000},   # 有界多通道召回（RETRIEVAL_DIAGNOSIS §3）；None → max(4·top_k, 60) / max(2·top_k, 30)
  "text":      {"cjk_tokens": "bigram"},   # "bigram" | "none"：索引侧与查询侧共用的 CJK 分词模式（§9.6；默认 = 历史的二元组）
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

RENAMED（重映射 + `DeprecationWarning(stacklevel=2)`）：`custom_fact_extraction_prompt → custom_instructions`、`custom_update_memory_prompt → custom_instructions`、`index → vector_store`、`db_path → store.path`。REMOVED（抛 `ConfigError(CONFIG_003)` 并附提示）：`graph_store`（"the support graph lives in the same SQLite file; remove the key"）、`enable_graph`（"the graph is always on"）。任何位置出现未知键 → `ConfigError(CONFIG_001)`，消息里点名该键，并附一条 `difflib.get_close_matches` 提示。一份原样的 Mem0 字典**只**会因为 `vector_store.provider` 不在 `{sqlite, memory, milvus}` 里而失败（`CONFIG_002`，建议文案 `"HMS Elite ships 'sqlite' and 'memory' index providers ('milvus' via the optional [milvus] extra); use {'provider': 'sqlite'}"`）。**`CONFIG_002`/`CONFIG_003` 在哪里抛出**：pydantic v2 会把验证器内部抛出的任何 `ValueError` 转成 `pydantic.ValidationError`，所以这两个错误码由 `HMSConfig.from_dict`（§5.3）里的纯 Python 前置检查抛出，时机在调用 `HMSConfig(**data)` **之前** —— 绝不从验证器内部抛出。`mode='before'` 验证器只做 RENAMED 键的重映射（幂等），这样直接构造 `HMSConfig(**d)` 也仍然可用；直接构造时若带 REMOVED 键或未知 provider，会以 `pydantic.ValidationError` 的形式暴露（这是文档化的行为；公开入口 `HMS(dict)`、`HMS.from_config`、`HMSConfig.from_dict` 一律走前置检查）。

### §5.2 类

```python
class ProviderEnvelope(HMSModel):
    kind: ClassVar[str]
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    @field_validator("config", mode="before")
    def _none_to_dict(cls, v): return {} if v is None else v
    @model_validator(mode="after")
    def _known_provider(self):   # provider ∈ REGISTRY[kind] else ValueError(f"unknown {kind} provider {provider!r}") — surfaces as pydantic.ValidationError on direct construction; from_dict pre-checks it as CONFIG_002

class LlmConfig(ProviderEnvelope):      kind = "llm";      provider: str = "rule"; enable_vision: bool = False; vision_details: str | None = None   # Mem0 把两者放在 `config` 里：前置校验器把它们提升到信封（信封显式键优先）
class EmbedderConfig(ProviderEnvelope): kind = "embedder"; provider: str = "hash"
class IndexConfig(ProviderEnvelope):    kind = "index";    provider: str = "sqlite"     # dict key is 'vector_store'
class RerankerEnvelope(ProviderEnvelope): kind = "reranker"; provider: str = "overlap"   # REGISTRY['reranker'] = {overlap, llm}；未知 provider → CONFIG_002（details['kind'] == 'reranker'）

class StoreConfig(HMSModel):     path: str = Field(default_factory=default_db_path); busy_timeout_ms: int = 5000; journal_mode: Literal["wal", "delete"] = "wal"; checkpoint_on_close: bool = True   # False：当 Python >= 3.12 提供 SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE 时，文件库在 close() 时把 WAL 留在磁盘上（仍耐久，下次打开时恢复）；否则以及 ':memory:' 一律忽略
class IngestConfig(HMSModel):    revision_mode: Literal["inline", "deferred", "off"] = "inline"; neighbourhood_k: int = 10; last_events_k: int = 10; emit_relation_atoms: bool = True; closure_max_gap_hours: float = 24.0; entity_extractor: str = "rule"; entity_extractor_config: dict[str, Any] = {}; extraction_mode: Literal["single", "staged"] = "single"; extraction_output: Literal["full", "compact", "lines"] = "lines"; segment_max_chars: int = Field(default=400, ge=1); verdict_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)   # verdict_min_similarity → §12.6 第 6 步的裁决守卫（NONE / REVISE 的包含度门槛，0.0 关闭；§6.3 修订规则）；extraction_output 'lines'（2026-09-18 起为默认，变更记录第 108 行、§6.6 默认段；此前为 'full'，'compact' 在 2026-09-11 当天曾取代它、同日晚按 §6.5 的准确率规则回退，第 105 行）→ §6.6（tab 分隔的 OBS / EP / DEC 文本，不用 json 模式，仅 single 模式——after 校验器以双语 ValueError 拒绝 'lines' + 'staged'，因此 extraction_mode='staged' 现在必须显式给出 'full' | 'compact'），'full' → 逐字节不变的完整契约，'compact' → §6.5（LLM 只输出 COMPACT_OBSERVATION_KEYS）；entity_extractor ∈ REGISTRY['entity_extractor']（§10.5）；from_dict 预检查 CONFIG_002，直接构造 → pydantic.ValidationError；config 为 None → {}；zh_rules: bool = False → §9.6（紧凑 / 行式推导的中文陈述规则；转给规则实体抽取器）
class ScoringConfig(HMSModel):   w_semantic: float = 1.0; w_keyword: float = 0.5; w_entity: float = 0.5; w_recency: float = 0.25; w_support: float = 0.25; half_life_days: float = 30.0; max_candidates: int | None = None; identifier_weight: float = 2.0   # max_candidates 是已弃用的兼容字段（最新优先截断已移除）；None 哨兵，只有非 None 的值才告警
class RetrievalConfig(HMSModel): k_semantic: int | None = None; k_keyword: int | None = None; k_entity: int | None = None; k_recency: int | None = None; pool_max_factor: int = 4; hard_gate_pool: bool = True; safety_limit: int = 50000
    def k_for(self, top_k: int) -> dict[str, int]   # {'semantic': k_semantic 或 max(4·top_k, 60), 'keyword' | 'entity' | 'recency': k_x 或 max(2·top_k, 30), 'pool': pool_max_factor · semantic}；HMSConfig.retrieval
class TextConfig(HMSModel):      cjk_tokens: Literal["bigram", "none"] = "bigram"   # §9.6：utils.text.tokenize / content_tokens 在两侧共用的 CJK 分词模式；HMSConfig.text（在 retrieval 之后）
class GraphConfig(HMSModel):     min_cohesion: float = 0.3; coarsen_min_nodes: int = 2; coarsen_max_nodes: int = 12; coarsen_gap_hours: float = 48.0; temporal_window_hours: float = 24.0; coarsen_policy: Literal["rule", "llm"] = "rule"; causal_proposer: Literal["rule", "llm"] = "rule"; coarsen_l3: bool = True
class CompilerConfig(HMSModel):  budget_tokens: int = 1200; candidate_k: int = 50; graph_hops: int = 1; require_witness: bool = True; slot_shares: dict[str, float] = {...as §5.1}; min_constraints: int = 3; max_why_nodes: int = 60; default_mode: CompileMode = "auto"; selection: Literal["shares", "coverage"] = "shares"; context_session_budget_tokens: int = Field(default=1200, ge=1); context_rows_top_k: int = Field(default=20, ge=1); context_rows_header: str = "Recent relevant memories:"; context_timestamps: bool = True   # context_* → HMS.context()（§6.7）：会话卡片预算、去重前的检索行数、两部分之间的标题行、'[YYYY-MM-DD HH:MM]' 时间戳开/关
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

各 provider 的配置类（pydantic，`extra="forbid"`，另加一个 `mode="before"` 验证器，经 `difflib.get_close_matches` 抛出 `ConfigError(CONFIG_001, "Extra fields not allowed: X. Did you mean 'Y'?")`）。它们住在各自的 provider 模块里（group B），但字段在此处定死：

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

### §5.3 公开函数

- `HMSConfig.from_dict(d: dict) -> HMSConfig` —— 校验一份字典。入参：dict。出参：HMSConfig。副作用：无。说明（规范顺序）：(1) `d` 不是 dict → `ConfigError(CONFIG_001)`；(2) `data = normalize_mem0_keys(d, strict=True)`（REMOVED 键 → `CONFIG_003`，RENAMED 键 → `DeprecationWarning`）；(3) `check_providers(data)`（→ `CONFIG_002`）；(4) `HMSConfig(**data)`，把 `pydantic.ValidationError` 包成 `ConfigError(CONFIG_001, details={'errors': e.errors()})` —— 但若 `e.errors()` 中任一条目的 `ctx['error']` 是 `HMSError` 实例，则原样重新抛出该实例；`CONFIG_001` 的消息会点名出问题的键（`loc`），并加上一条 `difflib.get_close_matches` 提示。`HMS.from_config` 和 `HMS.__init__(dict)` 都调用它；重新抛出之前会先把完整的 pydantic 错误写进日志。

### §5.4 注册表（`configs/registry.py`）

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
              "milvus": ("hms_elite.index.milvus.MilvusIndex", "hms_elite.index.milvus.MilvusIndexConfig")},   # 可选 extra `milvus`
    "reranker": {"overlap": ("hms_elite.reranker.overlap.OverlapReranker", "hms_elite.reranker.overlap.OverlapRerankerConfig"),
                 "llm": ("hms_elite.reranker.llm.LLMReranker", "hms_elite.reranker.llm.LLMRerankerConfig")},
    "entity_extractor": {"rule": ("hms_elite.entities.rule.RuleEntityExtractor", "hms_elite.entities.rule.RuleEntityExtractorConfig"),
                         "spacy": ("hms_elite.entities.spacy.SpacyEntityExtractor", "hms_elite.entities.spacy.SpacyEntityExtractorConfig")},   # §10.5；由 IngestConfig.entity_extractor 选择（是名称，不是信封）
}
EXTRA_OF_PROVIDER = {"openai": "openai", "anthropic": "anthropic", "milvus": "milvus", "deepseek": "openai", "ollama": "ollama", "fastembed": "fastembed", "spacy": "spacy"}   # pip extra hint per provider name (deepseek rides on the openai SDK; ollama is stdlib-only, its extra is empty)
def supported(kind: str) -> list[str]: return sorted(REGISTRY[kind])
```

只存字符串；不 import 任何 provider 模块。`utils/factory.register_provider` 就地修改 `REGISTRY[kind]`，因此 envelope 校验和工厂函数共用同一份事实来源。

---

## §6 Prompt 与 LLM 任务契约（`hms_elite/configs/prompts.py`，group A）

模块 docstring：`"""Prompt constants with HMS_TASK markers, user-prompt builders and their parsers.\n带 HMS_TASK 标记的提示词常量、用户提示词构造器及其解析器。"""`

### §6.1 形状规则

- 每个系统提示词都是模块常量 `<TASK>_PROMPT`，其**首行必须恰好是** `HMS_TASK: <task>`；分节按顺序为 `# ROLE`、`# INPUTS`（每个输入一个 `## ` 标题，顺序与构造器一致）、`# GUIDELINES`（Bad:/Good: 成对示例，双时钟规则）、`# EXAMPLES`、`# OUTPUT FORMAT`（下文逐字给出的 JSON 块，外加 `Return ONLY valid JSON parsable by json.loads(). No text, reasoning, explanations, or wrappers.`）—— **`procedural_summary` 除外**：它的 `# OUTPUT FORMAT` 块装的是 §6.3 那个精确的 markdown 骨架（七个标题/标签，按序排列），后面跟一行 `Return ONLY the markdown above. No JSON, no wrappers.`；它是唯一没有 `ROOT_KEY` 条目的任务，且通过 `generate_text` 产出，绝不走 `generate_json`（test 9 断言这些骨架标题出现在 `PROCEDURAL_SUMMARY_PROMPT` 的 OUTPUT FORMAT 分节内）。
- 构造器返回用户提示词：`## <Section>` 标题按文档规定的顺序排列；结构化输入是 ` ```json ... ``` ` 围栏块（`json.dumps(obj, ensure_ascii=False, indent=None, sort_keys=True)`）；自由文本分节为纯文本；提示词以 `# Output:` 结尾。
- 真实 id 绝不进入模型：已有记忆用 `'0'..'n'`，事件用 `idx`，EP ref 用 `e0..`，节点 ref 用 `n0..`，witness ref 用 `w0..`，pair id 用 `'0'..`；解析器负责把 ref 映射回来；引用了未知 ref 的条目会被丢弃，并打一条 `logger.warning`（"witness first"）。
- `TASKS = ("observation_extraction", "observation_extraction_compact", "ep_extraction", "revision_decision", "edge_proposal", "coarsen", "procedural_summary")`；`PROMPTS: dict[str, str]`；`PAST_MESSAGE_TRUNCATION_LIMIT = 300`；`COMPOSITE_TASKS = ("memory_extraction", "memory_extraction_compact", "memory_extraction_lines")` 与 `COMPOSITE_PROMPTS`（多根回复，不进 `ROOT_KEY`）；`COMPACT_OBSERVATION_KEYS`（§6.5）；`LINE_TYPES` / `TYPE_LETTERS`（§6.6）。

### §6.2 函数

- `task_of(system_prompt: str) -> str | None` —— 从首行读取 `HMS_TASK:` 标记。入参：系统提示词。出参：任务名或 None。副作用：无。
- `parse_sections(user_prompt: str) -> dict[str, Any]` —— 把构造好的用户提示词拆成 `{section_title: parsed_json | text}`。入参：构造器的输出。出参：以 `## ` 标题为键的字典；围栏 JSON 被解析，否则是去掉首尾空白的文本。副作用：无。说明：结尾的 `# Output:` 被丢弃；下面每个构造器都满足 `parse_sections(build_x(...))` 为其逆运算（有测试覆盖）。
- `build_observation_prompt(*, existing_memories: list[dict], new_messages: list[dict], last_messages: list[dict], observation_date: str, current_date: str, custom_instructions: str | None = None) -> str` —— 分节为 `## Last k Messages`（纯文本 `role: text` 行，每行截断到 300 字符）、`## Existing Memories`（json `[{"id": "0", "text": ...}]`）、`## New Messages`（json `[{"idx": 0, "role": "user", "text": ..., "timestamp": ...}]`）、`## Observation Date`（`YYYY-MM-DD`）、`## Current Date`、`## Custom Instructions`（仅在给出时出现）。
- `build_ep_prompt(*, events: list[dict], known_constraints: list[dict], recent_eps: list[dict], observation_date: str) -> str` —— `## Events`（json `[{"idx", "role", "object_type", "text", "timestamp"}]`）、`## Known Constraints`（json `[{"id": "c0", "text"}]`）、`## Recent EPs`（json `[{"id": "e0", "u", "a", "y", "r"}]`）、`## Observation Date`。
- `build_revision_prompt(*, new_items: list[dict], candidates: list[dict], current_date: str) -> str` —— `## New Memories`（json `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed"}]`）、`## Candidate Existing Memories`（json `[{"id": "0", "type", "subject", "predicate", "objects", "relation", "text", "date_observed", "valid_from"}]`）、`## Current Date`。
- `build_edge_prompt(*, nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> str` —— `## Nodes`（json `[{"ref": "n0", "kind": "ep"|"interface_ep"|"observation"|"event"|"skill", "u", "b", "a", "y", "r", "subject", "predicate", "objects": [...], "time", "entities": [...], "topics": [...], "anchor_refs": ["w0"]}]` —— EP/InterfaceEP 节点带 `u,b,a,y,r`（slot），其 `subject/predicate/objects` 为 null/[]；observation/event/skill 节点带 `subject/predicate/objects`（取自 atom），其 `u,b,y` = "" ，`a` = atom 文本，`r` = "unknown"；所有键始终存在，好让 rule provider 只按一种形状读取）、`## Candidate Pairs`（json `[["n0","n1"], ...]`）、`## Witnesses`（json `[{"ref": "w0", "node_ref": "n0", "source_ref", "time"}]`）、`## Allowed Edge Types`（纯文本：六行 `type: definition`）。`EDGE_PROPOSAL_PROMPT` 的 `# INPUTS` 分节列出的正是这些节点键。
- `build_coarsen_prompt(*, support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str) -> str` —— `## Support Nodes`（json `[{"ref": "n0", "kind", "u", "b", "a", "y", "r", "constraints": [...], "time", "anchor_refs"}]`）、`## Internal Edges`（json `[{"ref": "x0", "src": "n0", "dst": "n1", "type"}]`）、`## Boundary Edges`（json `[{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type"}]`）、`## Target Scale`。
- `build_procedural_messages(messages: list[dict], prompt: str | None) -> list[dict]` —— 返回 `[system: prompt or PROCEDURAL_SUMMARY_PROMPT, *messages(text only), user: "Create procedural memory of the above conversation."]`（与 Mem0 `_create_procedural_memory` 的形状一致）。

### §6.3 精确的 JSON 契约（OUTPUT FORMAT 块，逐字照录）

**`HMS_TASK: observation_extraction`** → 根键 `observations`
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
字段规则：`id` 是从 `"0"` 起递增的字符串；`event_idx` **必须**是 `## New Messages` 的合法下标（否则该条目被丢弃）；`type ∈ fact|preference_event|habit|constraint|temporal_fact`；`subject ∈ user|assistant|project|<entity text>`；`predicate` 为小写 snake 形式的动词；`relation ∈ single_fact|comparison|habit|supersession|constraint`；`text` 5–80 词，自足，第三人称；`date_observed` 为 `YYYY-MM-DD`，**只**依据 `## Observation Date` 解析；`confidence ∈ [0,1]`；`attributed_to ∈ user|assistant`（对话的侧别，永远不是人名）；`speaker` 为多说话人转录中具名的人物，否则 `null`（绝不是侧别词）；`sentiment ∈ positive|neutral|negative|mixed|unknown`；`supersedes_hint_ids` ⊂ `## Existing Memories` 的 id 集合。没有任何内容 → `{"observations": []}`。

**`HMS_TASK: ep_extraction`** → 根键 `eps`
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
规则：只有当触发（trigger）**且**动作（action）**且**（结果 outcome **或**约束 constraint）同时存在时，才产出一个 EP；`r ∈ positive|negative|neutral|unknown`；`uses_ep_ids` ⊂ `## Recent EPs` 的 id 集合；`event_idx` **必须**是 `## Events` 的合法下标。没有任何内容 → `{"eps": []}`。

**`HMS_TASK: revision_decision`** → 根键 `decisions`
```
{"decisions": [{"new_id": "0", "decision": "REVISE", "target_id": "2", "reason": "supersession cue: 'switched to'"}]}
```
规则：`## New Memories` 中每个 id 恰好对应一条 decision；`decision ∈ ADD|REVISE|CONFLICT|NONE`；REVISE/CONFLICT/NONE **必须**给出 `target_id`，且它必须是 `## Candidate Existing Memories` 中的 id，ADD 时为 `null`。ADD = 相容/增量；REVISE = 新条目取代目标；CONFLICT = 不相容，且无法明确判断谁更新、谁更真；NONE = 与目标重复。库侧裁决守卫（§12.6 第 6 步，两种契约、两种 `extraction_mode` 都适用，2026-09-11 起；去头部形式自变更记录第 104 行起）：每个 NONE / REVISE / CONFLICT 都断言两条关于同一主语的陈述之间的关系，因此两原子主语不同（不分大小写、去空白；双方都已知）的裁决按 ADD 存入——被观察在 `supersedes_hint_ids` 里点名目标、或相同主语 + 谓词佐证的 REVISE 例外，无条件通过（最先检查）；主语一致的 CONFLICT 通过（它断言的正是不同取值，文本本该不同）；NONE 或未被佐证的 REVISE 只在 `ingest.extraction.verdict_similarity(new.text, target.text, new_subject=…, old_subject=…, new_predicate=…, old_predicate=…)`——去掉头部（各自的主语；双方相同时的谓词）后词干化内容词元的包含度 |A ∩ B| / min(|A|, |B|)——不低于 `ingest.verdict_min_similarity`（0.5）、且双方至少共享两个词元或较短陈述被整体包含时通过（两个词元里共享一个是巧合而非复述）；其余按 ADD 存入（DEBUG 日志；`IngestResult.downgraded` 条目带 `similarity`、`shared`、`subjects_differ`）；0.0 关闭守卫。

**`HMS_TASK: edge_proposal`** → 根键 `edges`
```
{"edges": [{"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal",
            "label": "retry patch caused the billing regression",
            "witness_refs": ["w1"], "confidence": 0.8}]}
```
规则：`edge_type` **必须**是 `temporal|causal|revision|conflict|use|distill` 之一（其余一律丢弃）；`witness_refs` 必填、非空，且必须是字符串列表、是 `## Witnesses` 中隶属于 src 或 dst 的那些 ref 的子集（否则丢弃——标量 / 对象 / 嵌套列表一律告警丢弃，绝不抛错）；`confidence` 无法解析或非有限（`NaN` / `Infinity`）→ 0.7；不允许自环边；只允许 `## Candidate Pairs` 中列出的配对；没有任何边站得住脚时输出 `{"edges": []}`。

**`HMS_TASK: coarsen`** → 根键 `interface_ep`
```
{"interface_ep": {
   "slots": {"u": "...", "b": "...", "a": "...", "y": "...", "r": "positive",
             "kappa": {"constraints": ["..."], "visibility": "private"}},
   "slot_support": {"u": ["n0"], "b": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "kappa": ["n0", "n1"]},
   "invariants": [{"text": "Do not modify the subscription state machine.", "kind": "constraint", "support_refs": ["n0", "n1"]}],
   "residuals": [{"text": "outcome of n2 unknown", "kind": "missing_outcome", "support_refs": ["n2"]}]}}
```
规则：每个非空 slot **必须**在 `slot_support` 中引用至少 1 个 `## Support Nodes` 的 ref；invariants 的 `kind` ∈ `constraint|failed_path|stable_fact`；residuals 的 `kind` ∈ `unresolved_conflict|missing_outcome|low_cohesion|open_question`；引用了未知 ref 的 invariant/residual 一律丢弃；不得凭空捏造节点。

**`HMS_TASK: procedural_summary`** → markdown（不是 JSON）：沿用 Mem0 `PROCEDURAL_MEMORY_SYSTEM_PROMPT` 的结构 —— `## Summary of the agent's execution history`、`**Task Objective**:`、`**Progress Status**:`、编号步骤（每步含 `**Agent Action**:` / `**Action Result**:`）、`**Key Findings**:`、`**Errors & Challenges**:`、`**Current Context**:`。它的 `# OUTPUT FORMAT` 块就是这个骨架的逐字照录（标题/标签按此顺序，占位符写作 `...`），再加上一行 `Return ONLY the markdown above. No JSON, no wrappers.` 输出为空 → `ValueError("Procedural memory generation failed")`（与 Mem0 对齐）。在 procedural 分支上传入 `prompt=` 会**替换**掉这个系统提示词。

上下文渲染（Work State Card、交接说明文字）没有提示词 —— 完全确定性生成（§15.3）。

### §6.4 解析阶梯（`hms_elite/utils/json.py::parse_llm_json`，§9.3）适用于每一个 JSON 任务；`LLMBase.generate_json(task, system, user)` 会把传输层错误包装成 `LLMError(LLM_001, details={'task', 'provider'}) from e`。行式任务（§6.6）不是 JSON 任务：`LLMBase.generate_lines(task, system, user)` 不发送 `response_format`、返回原始文本、不走阶梯；由 `ingest.lines.parse_lines` 做容错解析。

### §6.5 紧凑输出契约（`ingest.extraction_output = 'compact'`；任务 `observation_extraction_compact`、`memory_extraction_compact`）

动机：真实 EgoLife 运行中，每批 25 个窗口让 deepseek-flash 的输出 token 中位数达到 5,562（上限 8,192；22% 的批次触顶并被对半重提），100% 是 JSON（3.4 字符/token，没有推理 token），每条观察 17 个键，其中九个（`subject`、`predicate`、`objects`、`topic_tags`、`entity_mentions`、`sentiment`、`relation`、`confidence`、`attributed_to`）只是复述 `text`。输出 token 是写入延迟与写入成本的主项（输出单价约为缓存未命中输入的 4 倍、缓存命中输入的 200 倍）。紧凑契约只要求 LLM 输出 `COMPACT_OBSERVATION_KEYS` 九个键（八个字段加可选的 `entities` 提示），其余全部在本地推导；EP 与决策保持 §6.3 的形状，且只在触发条件出现时输出（行动 → 结果闭包；非空的 `## Candidate Existing Memories`，因此没有候选时缺失 `decisions` 根不出声、不告警）。

LLM 输出的观察条目（两个紧凑提示词——`hms_elite/configs/prompts.py` 中符合 §6.1 命名的常量 `OBSERVATION_EXTRACTION_COMPACT_PROMPT` 与 `MEMORY_EXTRACTION_COMPACT_PROMPT`——的 OUTPUT FORMAT，逐字照录）：
```
{"observations": [{"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": null,
                   "entities": ["black coffee"], "supersedes_hint_ids": ["1"], "when": "2026-04-18", "valid_to": null}]}
```
字段规则：`text` 第三人称、单一陈述、至多 25 个词、同一事实不在多个条目里重复；`type ∈ fact|preference_event|habit|constraint|temporal_fact`（词表在提示词中逐字写明）；`speaker` 为具名人物或 null；`entities` 为陈述点名的事物（人、地点、物件、工具、组织），0–6 个短字符串，没有时省略——这是可选提示，只会扩大推导出的 `entity_mentions`（见下表），绝不收窄它；`supersedes_hint_ids` 仅在出现取代线索时给出（键可省略）；`when` 仅当陈述本身带有与消息日期不同的 ISO 日期或日期时间；`valid_to` 仅用于显式到期。用户提示词由同一个 `build_observation_prompt` / `build_memory_extraction_prompt` 构造、分节标题相同，因此 `parse_sections` 与规则处理器共用。注册方式与完整任务对称：`observation_extraction_compact` ∈ `TASKS`、`PROMPTS`、`ROOT_KEY`（→ `observations`）；`memory_extraction_compact` ∈ `COMPOSITE_TASKS`、`COMPOSITE_PROMPTS`。每个紧凑提示词的字符数低于其完整孪生的 0.75（`tests/test_extraction_compact.py`）。

推导表（`hms_elite/ingest/derive.py::expand_compact_item`，由 `ingest/extraction.py::_compact_observation_from_item` 调用；每个推导值随后都经过完整契约的规整 `_enum_value` / `_prob` / `_date` / `_datetime` / `_str_list` / `_map_refs`，因此词表绝不拒绝）。每个启发式只有一份实现：情感 / 主题表在 `utils.text`（A 组）；第三人称陈述解析器 `llms.rule.statement_triple`（建立在 R1–R9 的线索表 `_PREF_VERB_MAP`、`_SWITCH_VERBS`、`_LIVE_WORK_PRED`、`CONSTRAINT_RE` / `MUST_RE`、`VERB_LEXICON` 与实体抽取器之上）、`observation_topics` 与 `mention_texts` 在 `llms.rule`（B 组）；`derive.py`（C 组）不编译任何自己的线索，只做组合（§0.4 禁止规则 LLM import `ingest`；`tests/test_extraction_compact.py` 断言 `derive.py` 里唯一的正则是完整日期时间的形状）。

| 字段 | 推导方式 |
|---|---|
| `event_idx` → 锚点 | 不变：§12.3 的见证规则（越界 / 不可读 → 跳过该条目并记一条警告） |
| `attributed_to` | 被见证消息的角色侧别（assistant 消息为 `assistant`，否则 `user`） |
| `subject` | 陈述开头所指（`statement_triple`：`User` / `The user` / `用户` → `user`，`Assistant` / `助手` → `assistant`，`The team` / `Team` / `Project` / `团队` / `项目` → `project`，位于句首的 PROPER 实体 → 该名字，如 `Sarah joined …` → `Sarah`），否则有具名 `speaker` 时取它，否则约束取 `project`（R2 / R3 对祈使句的主语），否则按角色取 `assistant` / `user`（`derive.subject_for`） |
| `predicate` | `statement_triple` 对主语之后的文本按类型用 R1–R9 线索表读取：preference_event → `likes` / `dislikes` / `prefers`（`_PREF_VERB_MAP` 的第一与第三人称形式、`_PREF_VERB_MAP_ZH`；`does not` / `no longer` 翻转极性；切换动词 → 关系 `supersession`；`over` / `rather than` 备选项 → `prefers` + 关系 `comparison`；无线索 → `likes`，`sentiment` 为负面时 `dislikes`）；constraint → 经 `CONSTRAINT_RE` / `CONSTRAINT_ZH_RE` 为 `must_not`（`does not` 视为 `do not`），否则经 `MUST_RE` 为 `must`；habit → `does`；temporal_fact → `FUTURE_RE` 命中为 `plans` 否则 `did`（R7）；fact → `is_named` / `is` / `lives_in` / `works_at` / `studies_at`（`moved to` / `搬到` → `lives_in` + `supersession`）/ `compares`，否则取剩余部分的首个变形动词（第三人称 `-s`、过去式 `-ed`；`explains`、`keeps`），再否则 `llms.rule.predicate_of`（R9 词表启发式，默认 `states`） |
| `objects` | 来自同一次解析：偏好 / 身份 / 居住的宾语（`clean_object`），约束为 `constraint_objects(rest)`（R2 / R3），习惯为去掉习惯线索的剩余部分，temporal_fact 与兜底 fact 为陈述的实体减去主语（R7 / R9） |
| `relation` | `supersedes_hint_ids` 非空时为 `supersession`（线索出现过；未知引用仍由 `_map_refs` 丢弃），否则取陈述自身的线索（切换 / 搬迁动词或 `no longer` 为 `supersession`，有备选项为 `comparison`），否则 habit 类型为 `habit`、constraint 类型为 `constraint`，其余 `single_fact` |
| `confidence` | `0.7`（`derive.COMPACT_CONFIDENCE`，即完整契约对缺失值的默认） |
| `date_observed` | `when` 能解析为日期时取它（日期时间取其 UTC 日期），否则取被见证消息自身的日期（`date_of(event.timestamp)`；仅当该时间戳不可读时才用批次的 `observation_date`） |
| `valid_from` | 仅当 `when` 是完整日期时间（`derive.is_full_datetime`）时取它，否则 None——`atom_from_observation` 随后按完整模式同样的方式补时间锚 |
| `valid_to` | 条目的 `valid_to` 经 `_datetime`，下限处理与完整模式完全相同 |
| `sentiment` | `utils.text.sentiment(text)`——规则 LLM 的词表（positive / negative / mixed / neutral） |
| `topic_tags` | `llms.rule.observation_topics(text, type)`——规则 LLM 的 `TOPIC_TABLE` 命中加上该类型的 `preference` / `constraint`，最多四个 |
| `entity_mentions` | `derive.union_mentions(entities, llms.rule.mention_texts(text, entity_extractor), [speaker])`——先取 LLM 的 `entities` 提示（由 `extraction._entities_hint` 经 `_str_list` 规整：None → `[]`，字符串 → 单元素，dict 项取其文本；列表位置出现数字或 dict 时记一条警告并忽略，绝不跳过该条目），再取配置的实体提供者在 `text` 中的命中（`ingest.entity_extractor`，与 `link_atom_axes` 重新分类所用的同一实例），最后取 `speaker`；不分大小写去重并保留最先出现的写法（`_observation_from_item` 随后发现说话人已在其中，与完整模式相同）。因此紧凑原子的实体轴是 extractor(`text`) ∪ `speaker` 的超集，且只要 LLM 在 `entities` 里列出它在完整模式下会写进 `entity_mentions` 的内容，就与完整模式的轴相同：`User prefers python for scripting.` 上的 `entities: ["python"]` 存下 `entity_ids == ["python"]`，与完整模式完全一致，而不带提示的同一条目存下 `[]`（`tests/test_extraction_compact.py`）。提示词的 ≤ 6 上限不由规整强制（完整模式同样不设上限） |
| `speaker` → `actor_id` | 不变（§12.5）：消息 `name` 优先，否则 `speaker` |

紧凑模式下成立的不变量（`tests/test_extraction_compact.py`）：每条观察恰好带自己的见证锚点；七个轴（时间 / 有效期 / 实体 / 主题 / 角色 / 情感 / 尺度）全部由推导值填充（实体轴按上文取并集——绝不比完整模式更窄，且在每个聊天夹具上逐行（`entity_ids`）、逐表（`atom_entity`）与之相同）；取代提示经 `id_map` 解析；在每个聊天夹具、两种 `extraction_mode` 下，推导出的 `subject` / `predicate` / `objects` / `relation` 都等于规则 LLM 自己的完整观察，因此完整门面与紧凑门面的 `atom_anchor` / `atom_entity` / `atom_topic` / `entities` / `history` / `graph_nodes` / `graph_edges` / `edge_witnesses` 各表逐行相同（`chat_conflict` 保留 CONFLICT 历史行，`chat_multisession` 保留主语 `Sarah`）；未呈现候选时 `"decisions": []` 根不出声（ADD，不逐条警告），与缺失该根相同；规则 LLM 实现两个紧凑任务（`RuleLLM._task_observation_extraction_compact` / `_task_memory_extraction_compact`：把 R1–R9 的观察投影到紧凑键——`when` 仅在 `date_observed` 与消息日期不同时给出，`entities` = 完整观察的 `entity_mentions`、提示列表均仅在非空时给出——决策在投影之前基于完整观察判定），且在每个聊天夹具上存下与完整输出相同的文本、事件、锚点、类型与侧别；`extraction_output='full'` 逐字节不变（提示词 sha256 钉住，黄金文件不变）。两种 `extraction_mode` 都适用。未知值 → 配置层 `ConfigError(CONFIG_001)`，`Ingestor(extraction_output=...)` / `extract_all(output=...)` / `extract_observations(output=...)` 抛 `ValidationError(VALIDATION_013)`。自 2026-09-18 起默认为 `lines`（§6.6，变更记录第 108 行）；此前默认为 `full`（`compact` 在 2026-09-11 当天曾为默认，同日晚回退——见下一段）；`full` 与 `compact` 仍可用。紧凑提示词的 DECISIONS 准则（2026-09-11，变更记录第 103 行）按每条新观察 `id` 陈述——绝不按候选、绝不按位置配对——NONE 只留给复述（同一事实、同一主语、同一取值；新的细节、人物、时间或取值都是 ADD），附一对相似候选仍为 ADD 的 Bad / Good 示例，以及一个两候选、目标 id（`1`）与观察 id（`0`）不同的示例——两个紧凑提示词的 OUTPUT FORMAT 块同样为观察 `0` 引用提示 id `1`、（记忆任务）目标 `1`（第 104 行），因此任何地方都不再示范观察 id 与候选 id 的重合；决策形状与观察键不变。旧措辞下 deepseek-flash 把观察 i 与邻域候选 i 配对，在 204 题 EgoLife 运行中对 7,315 条观察里的 1,685 条回答 NONE（`full` 下 8,470 条中 23 条；1,623 条 `target_id == id`），每个 NONE 都丢掉该陈述（`benchmarks/egolife/results/replay_diag/summary.md`）；§6.3 / §12.6 第 6 步的库侧守卫兜住措辞兜不住的部分。

默认值与建议（2026-09-11，同日晚修订；2026-09-18 被 §6.6 的 `lines` 默认值取代，变更记录第 108 行——本段是 `full` 对 `compact` 决定的历史）。2026-09-18 之前默认为 `full`（`IngestConfig.extraction_output = 'full'`）。`compact` 于 2026-09-11 上午按两条件规则成为默认（变更记录第 102 行），并于 2026-09-11 晚按同日追加的第三个条件回退（变更记录第 105 行）；`compact` 仍可用、完整支持，且它不触碰的部分逐字节不变（完整提示词 sha256 钉住，黄金文件不变）。`compact` 保持默认须同时满足的三个条件：(a) 在 204 道 EgoLife 题的零 API 回放 A/B 中，紧凑臂的 hit@20 不得低于完整臂——已满足：两臂均 161/204，MRR 0.5000 vs 0.4973，实体轴在 8,439 对配对原子中全部 compact ⊇ full（24 个严格超集、0 个收窄；`benchmarks/egolife/results/replay_ab_entities`）；(b) 最终的紧凑提示词（带 `entities` 提示，变更记录第 100 行）须在真实 API 上完成同 18 批的冒烟并确认输出 token 降幅——已满足：`benchmarks/egolife/results/real/elite_compact_entities_smoke5_r2`（run_id `20260911T090326Z`）完成全部 18 批 / 395 窗口，每次请求输出 token 中位 2,793 / 最大 4,312，同批次 `full` 为 5,631 / 8,192（中位 −50%；`full` = `results/real/elite_full204_r3` 的前 18 批，34 次请求）；触及 8,192 上限 0/18 次 vs 8/34 次；每次请求模型耗时中位 8,675 ms vs 17,147 ms（−49%）；672 次 add() 结果（存入 606 条原子），`full` 为 660（存入 655）；窗口丢失 0；5 题答对 rows 3/5、card 4/5，`full` 为 3/5、3/5；账单 $0.089961（按非高峰价约 $0.0435）；中断的第一次尝试 `results/real/elite_compact_entities_smoke5` 保留其 `INCOMPLETE.md`；(c) 在真实 204 题 EgoLife 运行中，紧凑臂在记忆行上下文上的准确率与完整臂相差不得超过 2 个点——不满足。同 204 题的全部真实 deepseek-flash 运行（`docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`）：`full` r3 记忆行 61.3%（125/204）/ 状态卡片 61.8%（126/204），hit@20 78.9%，按空闲价重算 ¥11.56，每批模型耗时 14.9 s；compact 初版（`results/real/elite_compact_full204`）记忆行 53.9%（110/204）——根因是旧 DECISIONS 措辞下按下标配对的 NONE 裁决，7,315 条观察中 1,685 个 NONE（第 103 行）；compact 修订版（`results/real/elite_compact_v2_full204`：第 103–104 行重写的 DECISIONS 措辞加 0.5 的 `ingest.verdict_min_similarity` 守卫）记忆行 56.4%（115/204）/ 卡片 62.7%（128/204），hit@20 77.9% / 79.4%，裁决 ADD 7,604 / NONE 15 / UPDATE 9，窗口丢失 0，空闲价 ¥4.35，每批 7.3 s；mem0 56.9%（116/204）。56.4 比 61.3 低 4.9 个点，规则不满足，`full` 相对 `compact` 保持默认（直到 2026-09-18 切换为 `lines`）。建议：成本敏感的批量灌入（写入费用 0.38 倍、每批模型耗时 0.49 倍、触顶 3 次对 94 次）与以状态卡片上下文为阅读器输入的场景（卡片 62.7% 对 61.8%）用 `compact`；读记忆行上下文、或需要 LLM 自己给出 `confidence` / `topic_tags` / `sentiment` 的场景用 `full`（compact 把每条观察的 `confidence` 推导为 0.7，其余两项按词表推导，见上表）。作为参照，不带提示的第 99 行提示词（`results/real/elite_compact_smoke5`）测得输出 token 中位 2,002.5 / 最大 5,106、触顶 0 次、每次请求 7,140.5 ms、568 次 add() 结果（存入 556）、两种上下文均 3/5。测量、口径与复现命令：`docs/design/COMPACT_EXTRACTION_2026-09-11.md`。

### §6.6 行式输出契约（`ingest.extraction_output = 'lines'`；任务 `memory_extraction_lines`；仅 `extraction_mode = 'single'`）

动机（2026-09-11/12 实测，CNY 空闲价，4,851 个窗口）：写入费用 mem0 ¥1.85 = 输入未命中 0.93M token + 输出 0.22M；Elite `full` ¥11.5（输出 2.40M token = 费用的 83%）；Elite compact v2 ¥4.2（输出 0.76M）。输入未命中侧已经持平；差距在输出 token，而 JSON 信封的开销是其承载文本的 2–4 倍。零 API 回放表明检索只用到陈述文本与决策（entities / type / when 及其它字段对 hit@20 的影响为 0）。紧凑提示词自己写出的文本低于完整提示词（记忆行 56.4% 对 61.3%；配对 32 对 22），因此行式提示词逐字保留完整提示词的陈述规则，只替换 OUTPUT FORMAT。

测量结果（2026-09-12，`results/real/elite_lines_full204`，204 题全量，deepseek-flash，空闲档）：记忆行上下文 54.9%（112/204），状态卡片上下文 63.2%（129/204，所有臂中最高）；hit@20 84.3% / 82.4%（最高）；每窗口输出 51 token（full 495、compact 157、mem0 45），每批模型耗时中位 3.76 s（full 14.9、compact 7.3、mem0 3.5），写入 + 回答全按空闲价 ¥2.24（mem0 ¥1.98、compact ¥4.35、full ¥11.56）；存入 12,994 条原子（2.68 条/窗口，库 72.5 MB）；判定 ADD 12,963 / NONE 20 / UPDATE 11；零截断、零丢失。结论：成本与写入速度达到 mem0 水平且召回更高；记忆行上下文的准确率低于 full 4 个点（有据答错 72 题），状态卡片上下文高于 full 1.4 个点。

默认值（2026-09-18，变更记录第 108 行）：`lines` 是库默认值（`IngestConfig.extraction_output = 'lines'`；`extraction_mode` 仍为 `'single'`，因此 `staged` 配置现在必须显式给出 `full` | `compact`）。该决定依据 `docs/design/READER_AB_2026-09-18.md` §六 在同 204 题上的阅读端 A/B：行式存储保持完全相同时，§6.7 的混合读（记忆化会话卡片作前缀，再接带时间戳的相关记忆行，记忆在前作答）答对 70.1%（143/204）——高于完整契约的每种上下文（记忆行 61.3%、卡片 61.8%），而写入费用是行式的（每 204 题 ¥2.24 对 ¥11.56，空闲价），与纯记忆行上下文的符号检验 p < 0.0001。库中对应的读路径是 `HMS.context()`（§6.7）；基准脚本的 elite_hybrid 标签经同一方法构建，但不是 READER_AB 的 hybrid_prefix 的逐字节复现（见 §6.7 的偏差说明）。需要 LLM 自己给出 `confidence` / `topic_tags` / `sentiment` 的场景仍推荐 `full`；想要 JSON 信封又要更低费用的场景用 `compact`。

线上格式（`MEMORY_EXTRACTION_LINES_PROMPT` 的 OUTPUT FORMAT，`hms_elite/configs/prompts.py` 中符合 §6.1 命名的常量；纯文本，字段间一个 tab，下文以 `⇥` 表示）：
```
OBS
<event_idx>⇥<T>⇥<statement>[⇥@<YYYY-MM-DD 或 ISO 日期时间>]
EP                  （仅当存在行动 → 结果闭包时出现）
<event_idx>⇥u=<触发>⇥a=<行动>⇥y=<结果>⇥r=positive|negative|unknown[⇥k=<约束>;<约束>]
DEC                 （仅对非 ADD 裁决出现；分节缺失 = 每条观察都是 ADD）
<OBS 行号，1 起>⇥REVISE|CONFLICT|NONE⇥<Candidate Existing Memories 的候选 id>⇥<简短理由>
```
`T ∈ F fact | P preference_event | H habit | C constraint | T temporal_fact`（`configs.prompts.LINE_TYPES`；逆映射 `TYPE_LETTERS`）；陈述里绝不含 tab；`@date` 的含义与紧凑契约的 `when` 完全相同（只有陈述本身带有、且与消息日期不同、按 Observation Date 解析的日期）。注册：`memory_extraction_lines ∈ COMPOSITE_TASKS` / `COMPOSITE_PROMPTS`，不进 `TASKS`、`PROMPTS`、`ROOT_KEY`；与完整任务共用用户提示词构造器（`build_memory_extraction_prompt`）与分节。提示词逐字带有完整提示词的陈述规则（第三人称、5–80 词、一条自足陈述、保留数量 / 日期 / 时刻 / 名字、取代线索列表、问句 / 问候 / 套话不产出、助手陈述以 "Assistant" 开头、多说话人转录中具名者为主语）与重写后的紧凑 DECISIONS 规则（每条新观察至多一行 DEC，只写非 ADD 裁决，绝不按候选、绝不按位置；NONE 只留给同主语同取值的复述；时态事实与习惯恒为 ADD；互斥属性规则）；其 OUTPUT FORMAT 不含 JSON 花括号与 JSON 尾句，唯一示例为两行 OBS、无 EP、一行 DEC（`tests/test_extraction_lines.py`）。没有分段孪生：`observation_extraction` 没有行式形态，因此 `lines` 仅支持 single 模式。

调用路径：`LLMBase.generate_lines(task, *, system, user) -> str` 构造两条消息并以**不带** `response_format` 的方式执行 `_call`（JSON 模式会让模型把各行包进对象），返回原始文本（工具调用字典取其 `content`；None → ''）；传输错误与 `generate_json` 一样是 `LLMError(LLM_001)`。每个提供者无需改动即遵守：deepseek / openai 只在给定时转发 `response_format`，anthropic 只对 JSON 格式追加 "Respond with JSON only."，ollama 只对 JSON 格式设置 `format='json'`；`RuleLLM._task_memory_extraction_lines` 渲染 §6.6 的行（§11.8），`ScriptedLLM` 原样回放字符串。

解析器（`hms_elite/ingest/lines.py::parse_lines(text, *, what='memory_extraction_lines') -> {'observations', 'eps', 'decisions'}`，即 `extract_all` 为 compact 所消费的字典形状）：观察是紧凑形状条目 `{'id': 1 起的 OBS 行号（字符串）, 'event_idx', 'text', 'type', 'when'?}`；EP 为 `{'id', 'event_idx', 'u', 'b', 'a', 'y', 'r', 'kappa': {'constraints': [k= 按 ';' 拆分]}}`；决策为 `{'id': 其点名的 OBS 行号, 'decision', 'target_id', 'reason'}`。规则：`\r\n` 归一化；空行、代码围栏与 `OBS` 标题之前的内容忽略（容忍 `# OBS:` 写法；任何标题之前的 tab 分隔 `<idx>⇥<T>⇥…` 行隐式开启 OBS）；OBS 内三个及以上字段的行是 `<idx> <T> <statement> [@date]`，两字段的行在首字段为数字时是 `<idx> <statement>`（类型 fact）、首字段为类型字母时是沿用上一行 event_idx 的 `<T> <statement>`，无 tab 的行是不占行号的续行——紧接在只有前缀的行之后时就是该行的陈述，否则加一个空格接到上一条观察的陈述之后（没有上一条观察时记一条警告并跳过）；`@date` 既可以是独立的 tab 字段，也可以以空格粘在陈述末尾；未知类型字母按 fact 读取（DEBUG；完整类型名按首字母映射）；带 tab 而陈述短于两个字符的行（空陈述，或截断后只剩的单个字母）不存内容但保留行号；非数字的 event_idx 保留为文本，由 §12.3 的见证规则以自己的警告拒绝该条目；末尾没有换行且少于两个 tab 的行视为被截断的回复，记一条警告并丢弃（仍有两个 tab 的被截断陈述保留；末行是标题或围栏不算截断）；没有已知 `u= b= a= y= r= k=` 前缀的 EP 字段忽略（DEBUG），`r` 默认 `unknown`，完全没有槽位的 EP 行记警告并跳过；DEC 行点名的 OBS 行号不在 1..n_lines 内、该行没有存下观察、不是数字、或不足三个字段时记警告并跳过，裁决转大写，理由为该行余下部分；缺失的分节是空列表；完全没有开启任何分节的回复（散文、JSON 对象、''）记一条含 `NO_SECTION_MARK`（`'reply has no OBS / EP / DEC section'`）的警告，基准脚本据此把它当作不可解析的回复。编号：OBS 分节里每个带 tab 的行都占用一个 1 起的行号，无论是否存下观察，因此 `id` 就是模型所写的那一行，DEC 行的行号在丢行或折行之后仍然有效（2026-09-12 修正，变更记录第 107 行：此前 id 是已保留观察的序号，丢掉或拆开一行会让之后每个 DEC 目标偏移一位）。

推导 = compact。`extract_all(output='lines')` 对每条解析出的观察运行 `_compact_observation_from_item(..., speaker_from_subject=True)`——先见证（§12.3 的 event_idx 规则），再以 `speaker=None`、`entities=None` 调用 `ingest.derive.expand_compact_item`，最后走完整契约规整——因此 subject / predicate / objects / relation / date_observed / valid_from / confidence 0.7 / attributed_to / sentiment / topic_tags / entity_mentions 遵循 §6.5 的推导表，且 `entities = []`、`supersedes_hint_ids = []`（relation 因此绝不来自提示；由陈述自身的线索或类型决定，REVISE 在 §12.6 守卫中只能由相同主语 + 谓词佐证）。唯一的行式专属规则：该格式没有说话人槽位，因此当推导出的主语不是对话侧别（`derive.SIDE_SUBJECTS = {'user', 'assistant', 'project'}`）——即陈述开头的专有名，完整提示词要求多说话人转录中的具名者同时为主语与说话人——它就成为 `speaker`，从而成为实体提及并（消息不带 `name` 时）成为 `actor_id`，与紧凑契约的 `speaker` 落点完全相同。EP：线上格式只带 u / a / y / r / k（及可选的 b=），因此 `ingest.derive.expand_lines_ep` 在 `_ep_from_item` 之前在本地补齐其余字段，与规则 LLM 补齐自己的 EP 完全一样——`entity_mentions` = `mention_texts('<a> <y> <u>')`，`topic_tags` = `topic_tags('<a> <y> <u>')`，未给 b= 时 `b` = k= 约束以 `' ; '` 连接；`time_anchor` = 事件时间；`confidence` 0.7；`visibility` = 事件的；`uses_ep_ids` 为空（2026-09-12，第 107 行：此前这三列在行式下存为空）。决策：DEC 行按行号经 `_parse_decision` 连接到其观察（候选 id → atom_id；未知目标 → ADD 并记警告）；没有 DEC 行的观察是 ADD（`'no DEC line'`）且不记警告，无论是否呈现候选（契约规定缺失即 ADD）；§6.3 / §12.6 的裁决守卫原样适用。

不变量（`tests/test_extraction_lines.py`）：在每个聊天夹具上，行式门面存下与完整门面相同的文本、事件、见证锚点、侧别、类型、`entity_ids`、`revises`、原子 (text, subject, predicate, objects) 以及相同的 `atom_anchor` / `atom_entity` / `atom_topic` / `entities` / `history` / `graph_nodes` / `graph_edges` / `edge_witnesses` / `eps` 行（chat_conflict 保留 CONFLICT 行，chat_preference 保留 REVISE，chat_action_closure 保留带相同实体 / 主题 / b 列的 EP）；`observations` 与 `memory_atoms` 行除紧凑推导所拥有的那几列外完全相同，测试逐列钉住：`confidence`（常量 0.7，规则 LLM 为 0.6–0.9）、`relation` 与 `supersedes_hint_ids`（chat_preference 与 chat_zh 上为 `single_fact` / `[]`，`full` 为 `supersession` / 提示 id——行式格式没有提示槽位）、说话人规则点名陈述开头专有名处的 `memory_atoms.actor_id`（chat_multisession：每行都是 `Sarah`，`full` 下没有），以及 chat_zh 上的 `observations.entity_mentions`（从推导出的陈述文本读出的 `用户`——紧凑路径同样如此，不进入 `atom_entity`）；脚本化行式回复经 `HMS.add` 推导出每个轴、由 DEC REVISE 生成修订边，而守卫仍把非复述的 NONE 降级；调用不带 `response_format`（ScriptedLLM 的 kwargs 为 `{}`、DeepSeek 提供者的 `chat.completions.create` 参数）；完整与紧凑提示词逐字节不变（钉住四个 sha256）。`extraction_mode='staged'` 配 `lines` → `IngestConfig` 抛带双语消息的 ValueError（经 `from_dict` 为 `ConfigError(CONFIG_001)`），`Ingestor(...)` / `extract_observations(output='lines')` → `ValidationError(VALIDATION_013, details.supported=['single'])`。自 2026-09-18 起默认为 `lines`（§5，第 108 行；此前为 `full`）；基准 CLI（`benchmarks/egolife/run_real.py --extraction-output`，选项 = `EXTRACTION_OUTPUTS`，默认 = 库默认值）与 `EliteRealArm(extraction_output=...)` 原样透传显式给出的值；真实 API 测量见上面两段与 `READER_AB_2026-09-18.md`。

### §6.7 混合读上下文（`HMS.context()`；`hms_elite/compiler/context.py`；2026-09-18）

阅读端 A/B 测得最好的读路径（`docs/design/READER_AB_2026-09-18.md` §六：hybrid_prefix 在 EgoLife 上 70.1%（143/204），与纯记忆行上下文的符号检验 p < 0.0001，冷前缀每题 ¥0.00185）：按范围记忆化的、与查询无关的会话卡片，再接本次查询带时间戳的相关记忆行，放在用户问题**之前**。

签名（§17.3）：`context(self, query: str, *, filters: dict, top_k: int | None = None, as_of: Any = None, budget_tokens: int | None = None) -> dict`。与 `AgentRuntime.context(...) -> ContextIR`（`hms_elite/runtime.py`，编译一张卡片）无关：两个对象上的同名方法，两份契约。

文本形状（`compiler.context.hybrid_text`，与基准脚本的 `hybrid_context` 逐字节相同）：`"<会话卡片>\n\n<context_rows_header>\n<记忆行>"`——`<会话卡片>` 是 `compile(...).to_text()`，范围内没有可抽取事件时为 `(none)`；`<记忆行>` 是行段，没有保留行时为 `(none)`；标题是 `CompilerConfig.context_rows_header`（默认 `Recent relevant memories:` = `compiler.context.ROWS_HEADER`）。产品把该文本放在问题之前——`"Memories:\n<text>\n\n<prompt>"`——因为记忆在前不损失准确率（READER_AB §四），且是会话卡片复用期间唯一能命中提供者提示词缓存（DeepSeek）前缀的顺序。

记忆行：`search(query, filters, top_k=<给定则 top_k，否则 context_rows_top_k>, reference_date=as_of)` → `dedup_rows`（键 = casefold 并折叠空白的文本；精确重复剔除，排名靠前者保留；再剔除键是另一唯一行的键的真子串的每一行；保持排名顺序）→ `sort_rows_by_time`（按 `hms.time_anchor` 稳定升序，回退 `created_at`，两者都缺的行按输入顺序排最后）→ `render_rows`：每行一条 `- [YYYY-MM-DD HH:MM] <文本>`，时间戳**只**取自 `time_anchor`、UTC（`format_stamp`；锚定缺失或不可解析时省略方括号——`created_at` 绝不作时间戳），`context_timestamps` 为 False 时为 `- <文本>`。`rows_text(raw)` = 对原始行做这三步（= 基准脚本的 rows_ts / rows_td 上下文）。

会话卡片：`compile(query=<范围内最新可抽取事件的文本>, filters=<范围过滤（+ visibility）>, as_of=<该事件的时间戳>, budget_tokens=<给定则 budget_tokens，否则 context_session_budget_tokens>)`，其中事件由 `StateStore.latest_event(scope, object_types=[message, tool_call, file_diff, test_log, sensor_trace])` 给出（§8：`ORDER BY timestamp DESC, rowid DESC`——同一次 `add()` 的 N 条消息共享事件时间，取最后**写入**者；`last_events` 以随机的 event_id 打破平局，故另设方法）；查询是事件载荷的 `text`（消息事件），否则是其规范 JSON。system / feedback / checkpoint / rollback / scheduler 事件绝不会成为查询。卡片按 `(scope.key(), audience)` 连同 `StateStore.scope_version(scope)` 与预算在进程内记忆化，两者都不变时原样复用（`session_reused = True`）；`budget_tokens` 不同则重算。

失效依据是按范围的写入版本，绝不是计时器（§8）：事务通过 `StateStore.touch_scope(scope)` 把自己归于某个范围——`insert_source_event`（仅当真正存入了一行——重放的相同事件是空操作）、`insert_atom`，以及 `StateScheduler` 的每个阶段事务及其转移导出都调用它（它们代表目标范围写入无范围的行——边、历史、衰减、转移——因此 `consolidate(filters=B)` 与守护进程让范围 A 的记忆保持复用；`feedback` 与 `purge(dry_run=False)` 在存入其带范围系统事件的同一事务内写入，因此同样按范围）；最外层事务 COMMIT 时，**仅当**该事务写入了带版本的行，被归属范围的计数才推进（`scope_write_versions[key] += 1`），因此 `add()` 的事务 A 推进一次（事件）、事务 B 再推进一次（原子），无锁阶段构建的上下文会被事务 B 失效，而没有改变任何行的事务（被忽略的 INSERT OR IGNORE、未命中的 UPDATE、无事可做的调度器阶段）不推进任何版本；任何范围的读取都不依赖的行在 `StateStore.unversioned()` 内写入、两者都不计——内部事件及其锚点（`system_event`：scheduler / checkpoint / rollback / feedback / system；事件仍把它见证的操作归于其范围，真正使记忆失效的是该操作自身的归档 / 修订 / 衰减）、`checkpoints` 行、调度器的 `last_consolidation:<scope>` / `work_state:<scope>` meta 行与其编译态归档——因此检查点、空转的 `consolidate(A)` 与守护进程对无事可做范围的 `consolidate()` 让每个记忆保持复用（创建或修订了原子、边、InterfaceEP 或技能、或衰减了陈旧原子的 consolidate 一如既往地失效）；在没有任何范围、也没有范围内调用方的情况下改变了带版本行的已提交事务，以及每条自动提交的带版本写入、回滚与 `reset()` 推进全局 `write_generation`；`scope_version(scope)` = `(write_generation, data_version(), 覆盖 scope 或被 scope 覆盖的每个已记录范围的有序 (key, n))`——因此 `delete_all(user_id=u)`（部分范围的系统事件）使 `(u, agent)` 的记忆失效，而对无关范围的写入保持复用。`HMS.reset()` 也清空记忆。

返回字典：`{'text', 'session_card', 'session_key'（卡片所据事件的 event_id；范围内无可抽取事件时为 None）, 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids'（卡片各项目的 memory id）, 'rows'（按时间排序的保留行）, 'rows_text', 'rows_header', 'search_rows'（按排名的原始检索行）, 'scope'}`。校验：`parse_filters(require_scope=True)`（VALIDATION_001）、`search` 自身的检查（VALIDATION_009 / 010）；给出的 `top_k` / `budget_tokens` 原样透传（显式的 0 不会悄悄变成默认值）。`as_of` 只约束记忆行（search 的 `reference_date`）；会话卡片始终是最新事件、在其自身时间戳处编译（面向流式使用；记忆按范围作键，不按 `as_of`）。存储只读（一次 search，至多一次 compile）；`AsyncHMS.context` 镜像之；没有 HTTP 路由与 CLI 动词（`POST /context` 路由留作后续）。

与 READER_AB 基准脚本的偏差，记录而不修正：基准脚本只在某步的 retain 产生 ≥ 1 个 ADD / UPDATE 时重算会话卡片，以最后真正入库的窗口作键、`as_of` = 该窗口自身的起点；库在对该范围的任何已提交写入时失效（抽取只产生 NONE 的 `add()` 仍存入其事件），并用最新事件的时间戳（对基准脚本而言是切片首窗口的起点）。两个时刻都 ≥ 每条原子的 `valid_from`（有效性谓词含端点），因此在基准脚本上卡片内容等价，但 `run_real.py` 的 `elite_hybrid` 数字不是 READER_AB hybrid_prefix 的逐字节复现。测试：`tests/test_context.py`（形状与 '(none)' 部分；记忆复用与 add / update / delete / delete_all 含父范围 / 改变了行的 consolidate / rollback / reset 的失效、跨范围精度；跨检查点、空转的 consolidate、守护进程形态的 `consolidate()` 与零行事务的复用；`scope_version` 的提交时与 `unversioned()` 语义；时间戳；基准脚本的 `reader_ab.dedup_rows` / `sort_rows_by_time` / `format_stamp` / `row_times` / `hybrid_context` 就是这些函数且七条手写行上 `rows_ts_context == rows_text`；配置字段与按次覆盖参数）。

---

## §7 SQLite DDL（`hms_elite/memory/schema.py`，group A）

模块 docstring：`"""SQLite schema for all HMS planes in one file; invariants as CHECKs and triggers.\n单文件多平面 SQLite 模式；不变量以 CHECK 与触发器表达。"""`

连接（由 `StateStore` 持有）：`sqlite3.connect(path, check_same_thread=False, isolation_level=None)`；`PRAGMA foreign_keys = ON`；`PRAGMA busy_timeout = <ms>`；先跑迁移（`apply_migrations`：版本 0 的库直接按 `build_ddl(SCHEMA_VERSION)` 在**一个** `BEGIN IMMEDIATE` 事务内建成——一次提交、无释放页；已有文件则从其 `user_version` 沿只追加的 `MIGRATIONS` 阶梯逐级升级，每级一次提交；两者得到相同的 `sqlite_master` 行集合），之后仅文件型数据库才设 `PRAGMA journal_mode = WAL`（顺序有讲究：新文件库的 97 个模式页因此在回滚日志模式下只写一次，而不是先经 WAL 写一次、再在关闭时的 checkpoint 中写第二次）。私有 `':memory:'` 库不重新执行 DDL：`clone_schema` 用 backup API 逐页复制进程级、仅含模式的模板（由 `apply_migrations` 在 `':memory:'` 上构建一次，此后从不写入、从不外泄）——与 DDL 路径有相同的 `sqlite_master` 行、根页与 `user_version`；`file:` URI 与文件库始终走逐条语句路径（它是跨进程唯一无竞争的路径）。下文的 `SCOPE` 展开为 `user_id TEXT, agent_id TEXT, run_id TEXT, CHECK (user_id IS NOT NULL OR agent_id IS NOT NULL OR run_id IS NOT NULL)`；`VIS` 展开为 `CHECK (<col> IN ('private','team','event','public'))`。所有 CHECK 列表都由 `hms_elite.enums` 生成，只有两处收窄：`source_events.role` 取 `[r for r in Role.values() if r != 'extracted']`（事件永远不携带这个合成角色），`eps.scale` 是字面量 `'L1_micro'`。下方打印出的 DDL 就是生成结果，并以它为准。

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

### §7.1 迁移

```python
SCHEMA_VERSION = 3
HISTORY_EVENTS_V1 = tuple(e for e in HistoryEvent.values() if e != "PURGE")
PURGE_FLAG_KEY = "purge_in_progress"; PURGE_GATE = "WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1') "
PURGE_GATED_TRIGGERS = ((name, table, message) × 7)          # trg_events/anchors/obs/eps/atoms/atom_anchor/nodes_no_delete
def build_ddl(version: int = SCHEMA_VERSION) -> list[str]  # version 1 = 0.1 版模式（无门触发器、history 无 'PURGE'）；2 = 上面打印的模式（83 条语句）；3 = 2 + FTS_DDL（84 条语句）
def upgrade_2() -> list[str]                               # 21 条：DROP TRIGGER IF EXISTS + 带门 CREATE TRIGGER ×7；ALTER TABLE history RENAME TO history_v1；CREATE TABLE history（新 CHECK）；INSERT INTO history(<16 列>) SELECT … FROM history_v1 ORDER BY seq；DROP TABLE history_v1；重建 ix_history_memory 与两个 history 触发器
FTS_DDL = "CREATE VIRTUAL TABLE atom_fts USING fts5(atom_id UNINDEXED, tokens, tokenize = 'unicode61')"   # 关键词通道：每个**未归档**原子版本一行，tokens = fts_tokens(text, topic_tags, entity_keys)（内容词元按 utils.scoring.atom_keyword_tokens 同样词干化、路径基名、主题、实体键及其词元；CJK 二元组以空格分隔，unicode61 原样保留）
def upgrade_3() -> list[str]                               # [FTS_DDL]；所链接的 SQLite 没有 FTS5 模块时跳过（并告警）
def fts_tokens(text: str, topic_tags: Iterable[str] = (), entity_keys: Iterable[str] = ()) -> str
def backfill_fts(conn) -> int                              # 对每个 archived_reason IS NULL 的 memory_atoms 行 INSERT INTO atom_fts（主题 + 实体键在 Python 侧拼接）
MIGRATIONS: dict[int, list[str]] = {1: build_ddl(1), 2: upgrade_2(), 3: upgrade_3()}   # 阶梯只升级**已有**文件（版本 1 的文件先跑 2 再跑 3）；版本 0 的库直接按 build_ddl(SCHEMA_VERSION) 在一个事务内建成；两者的 sqlite_master 行 (type, name, tbl_name, sql) 相同（test_fresh_database_is_created_directly_at_schema_version、test_purge_gate_and_schema_migration_2、test_schema_v3_backfills_fts_from_a_v2_file）
POST_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], object]] = {3: backfill_fts}   # 某版本语句之后、**同一**迁移事务内执行的 Python 步骤（全新库没有行，不执行）

def apply_migrations(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    """Apply pending migrations statement by statement under BEGIN IMMEDIATE, gated by PRAGMA user_version.
    在 BEGIN IMMEDIATE 下逐条应用待执行迁移，由 PRAGMA user_version 控制。"""
    # current > SCHEMA_VERSION → StoreError(STORE_002, suggestion="pip install -U hms-elite")
    # for version in range(current+1, SCHEMA_VERSION+1): BEGIN IMMEDIATE; execute each stmt (never executescript); [version == 1 且给了 created_at：INSERT OR REPLACE meta('schema_created_at', created_at)——时间戳搭乘迁移事务]; PRAGMA user_version = version; COMMIT; on error ROLLBACK + raise StoreError(STORE_001) from e
def clone_schema(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    # 仅限全新的私有 ':memory:' 连接（user_version 0）：把进程级模式模板（每个 pid 由 apply_migrations 在 ':memory:' 上构建一次，加锁，此后从不写入）backup 进 conn，再写 meta.schema_created_at；非全新连接或 backup 失败时回退到 apply_migrations；结果与 DDL 路径相同（sqlite_master 行、根页、user_version）
def fts5_available() -> bool             # 在一个临时 ':memory:' 连接上 CREATE VIRTUAL TABLE … USING fts5；每进程只探测一次
def detect_capabilities(conn) -> dict:   # {'sqlite_version': str, 'json1': bool (SELECT json_extract('{"a":1}','$.a') works；每进程只探测一次，它是所链接库的属性), 'fts5': bool（fts5_available() 且 sqlite_master 中存在 atom_fts）, 'wal': bool, 'schema_version': int}
def drop_all(conn) -> None               # 先 DROP TABLE 虚拟表（影子表随之删除），再逆序删除全部基表——唯一类似 DELETE 的路径；供 StateStore.reset() 使用
TABLES: tuple[str, ...] = ('source_events','anchors','observations','eps','memory_atoms','atom_anchor','entities','atom_entity','atom_topic','atom_vectors','graph_nodes','graph_edges','edge_witnesses','edge_projections','interface_eps','interface_ep_support','feedback_events','history','checkpoints','transitions','compiled_states','meta')   # 22 张基表
VIRTUAL_TABLES: tuple[str, ...] = ('atom_fts',)   # 1 张虚拟表（FTS5）；其影子表 FTS_SHADOW_TABLES = atom_fts_data / _idx / _content / _docsize / _config 属于 SQLite，只按名字列出——TABLES、stats() 与 TableTrace 只覆盖 22 张基表；FTS5 可用时 sqlite_master 中 type='table' 的行为 22 + 1 + 5
```

规则：迁移只允许追加（后续版本写成 `{2: [...]}`）；绝不使用「列集合 diff 后整表复制」的做法；SQLite 最低版本 3.31（需要带 COALESCE 的表达式 UNIQUE 索引、部分索引、RAISE 触发器）；`json1` 为可选依赖（§13.1 有回退路径）。

---

## §8 `StateStore`（`hms_elite/memory/storage.py`，group A）—— 公共接缝

模块 docstring：`"""One SQLite connection, one lock, typed writers for every table; the only place SQL is executed.\n单一 SQLite 连接、单一锁、每张表的类型化写入器；唯一执行 SQL 的地方。"""` 用途：全平面受控持久化。输入：pydantic 模型、scope、id。输出：模型/行/id。副作用：在 `transaction()` 内写库；原始游标私有。

行为约定：每个对外写操作都必须运行在 `with store.transaction():` 之内（嵌套调用可重入；最外层在 `threading.Lock` 保护下执行 `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK`）；在事务之外调用写入器，会隐式开启并提交一个事务。`sqlite3.Error` 先回滚，再抛 `StoreError(STORE_001) from e`；以 `HMS_INVARIANT` 开头的触发器 `RAISE` 消息映射为 `StoreError(STORE_003)`；witness 前置检查按 §3 执行（在任何 SQL 之前抛 `WitnessMissingError`）。`_execute` 在锁内执行语句**并取完所有结果行**，返回一个已物化的行列表（提供 `fetchone` / `fetchall` / 迭代 / `rowcount` / `lastrowid`）；任何活动的 `sqlite3.Cursor` 都不会在锁外被读取，因此同一实例的并发读取方（`check_same_thread=False` 的设计）不可能经 CPython 的每连接语句缓存拿到别的语句的行。`trace: TableTrace` 记录每条语句触及的表名（`trace.reset()`、`trace.tables() -> set[str]`、`trace.enabled`）。所有 `list_*` 读取器都在 SQL 中按 `Scope.matches` 语义施加 `scope`（每个已设置键 `col = ?`，未设置键不产生谓词，全未设置时为 `1=1`——绝不用 `(col = ? OR ? IS NULL)` 形式，它会把绑定值藏起来，使 `ix_atoms_scope` 永远用不上），也就是说只带 `user_id` 的过滤条件会匹配所有 run。每条按范围过滤的 SELECT 都带确定性 `ORDER BY`，因此计划变化从不改变结果。三种轴链接由每 500 个 id 一条语句读取（`WITH ids(atom_id) AS (VALUES …)` 连接 `atom_entity`/`entities`、`atom_topic`、`atom_anchor`）；`insert_atom` 返回由写入本身重建的同一水合形式（JSON 列经 `canonical_json` 往返，链接来自插入的列表），不再回读该行，`test_insert_atom_mirrors_read_path` 钉住 `insert_atom(a) == get_atom(a.atom_id)`。见证预检按最外层事务记忆化（`_verified_anchors`：经 `get_anchors` 证明或由 `get_or_create_anchor` 写入的 id 在 COMMIT/ROLLBACK 清空备忘之前不再重复 SELECT；锚点从不删除，备忘在其事务内不会过期）。**填充（Hydration）**：任何读取器（`get_atom`、`get_atoms`、`axis_select`、`resolve_head`、`list_versions`、`find_atoms_by_hash`、`revision_candidates` 等）返回的每个 `MemoryAtom`，其 `entity_ids` 为经 `atom_entity` 连接得到的实体键（`entities.normalized_text`），已排序；`topic_tags` 为 `atom_topic` 的行，按插入顺序；`anchor_ids` 为 `atom_anchor_ids(atom_id)`（不含被争议的行，primary 在前）。**存活判定式（liveness predicate）**（`LIVE_SQL`，一个常量，被每个 `valid_only`/有效性子句复用）：`archived_reason IS NULL AND valid_from <= :as_of AND (valid_to IS NULL OR valid_to > :as_of)`；除非调用方显式传入，否则 `as_of = now()`；`include_expired=True` 会整体丢弃该判定式。传入的 id（`event_id`、`atom_id`、`memory_id`、`ep_id` 等）非空时原样保留（`import_state_package` 依赖此行为）；传入 `""` 则生成 `new_id(prefix)`。

```python
class TableTrace:
    enabled: bool = True
    def record(self, sql: str) -> None; def reset(self) -> None; def tables(self) -> set[str]

class StateStore:
    def __init__(self, path: str = ":memory:", *, now: Callable[[], str] | None = None, id_factory: Callable[[str], str] | None = None,
                 busy_timeout_ms: int = 5000, journal_mode: str = "wal", checkpoint_on_close: bool = True) -> None
        # ':memory:'/'' → clone_schema；其它 → apply_migrations，随后（文件库且 journal_mode == 'wal'）PRAGMA journal_mode = WAL；created_at = now() 搭乘迁移事务；checkpoint_on_close=False 且为文件库 → 运行时具备时在 close() 前设置 SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
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
    def archive_atom(self, atom_id: str, *, reason: str, at: str, valid_to: str | None = None) -> None   # 另一未归档原子引用相同 (source_kind, source_id) 时跳过来源级联（eps / observations / interface_eps）
    def reopen_source(self, atom_id: str) -> None                    # 重新打开存活原子引用的来源行（revise_atom 在创建与前驱共享来源的后继后调用）
    def shared_source_atom_ids(self, atom_ids: Iterable[str]) -> set[str]   # 其 (source_kind, source_id) 也被列表之外的原子引用的 id（清理保护 `source_shared`）
    @property
    def lock(self) -> threading.RLock                                # store 锁，锁序中的第一把（索引提供者先取它再取自身的锁）
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
        # atom_ids（池的水合）：`WITH ids(atom_id) AS (VALUES …) SELECT … FROM ids CROSS JOIN memory_atoms ON …`——每个 id 一次主键查找，绝不扫描 scope；index_hint：`FROM memory_atoms INDEXED BY <名称>`（schema.TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX，由 mvmi 按该级别的门控选择；不是标识符的名称 → VALIDATION_000），给了 atom_ids 时忽略
    def axis_ids(self, scope, *, where_sql="", params=(), joins=(), as_of=None, include_expired=False, scale_constraint=None, atom_types=None,
                 order="time_anchor DESC, created_at DESC, atom_id ASC", limit=None, index_hint=None) -> list[str]     # 同一门控，只取 id（不水合）：检索池的门控集合 G；order=None 表示**不加 ORDER BY**（索引序，LIMIT 让扫描提前停止——池探测与语义通道的 G）
    def count_axis(self, scope, *, where_sql="", params=(), joins=(), as_of=None, include_expired=False, scale_constraint=None, atom_types=None, index_hint=None) -> int   # |G|（有 joins 时 COUNT(DISTINCT atom_id)）
    def keyword_candidates(self, scope: Scope, tokens: Sequence[str], *, as_of: str | None = None, include_expired: bool = False, limit: int,
                           where_sql: str = "", params: Sequence = (), joins: Sequence[str] = (), scale_constraint=None, atom_types=None, index_hint=None) -> list[tuple[str, float]]
        # 在级别门控**之内**的关键词召回通道（门控参数同 axis_select）：两个 FTS5 阶段、先严后宽——`MATCH '"t1" AND "t2" …'`（仅当词元 >= 2）再由 `MATCH '"t1" OR "t2" …'` 补足；每个阶段先在 atom_fts 内排序（`ORDER BY rank, atom_id LIMIT window OFFSET …`，window = max(4·need, 64)，倍增），只对排好序的窗口经 VALUES-CTE CROSS JOIN（主键查找）重新施加门控，窗口扩大直到找到 `limit` 个门控内的行或该阶段耗尽；score = -bm25（越大越好），行按阶段顺序；capabilities['fts5'] 为 False → 带同样门控的有界 LIKE 退路（score = 匹配词元数，再按 DEFAULT_ATOM_ORDER）；已归档版本没有 atom_fts 行；index_hint 接受但忽略
    def entity_candidates(self, scope: Scope, entity_keys: Sequence[str], *, as_of: str | None = None, include_expired: bool = False, limit: int | None,
                          where_sql: str = "", params: Sequence = (), joins: Sequence[str] = (), scale_constraint=None, atom_types=None, index_hint=None) -> list[str]
        # 在级别门控**之内**的实体召回通道：先解析实体 id（resolve_entities：normalized_text IN keys，COALESCE(user_id/agent_id) 身份）→ `FROM atom_entity ae CROSS JOIN memory_atoms ON …`（ae 经 ix_entity_atom 驱动，memory_atoms 按主键——否则规划器会扫描 scope）WHERE ae.entity_id IN (…) AND 门控 ORDER BY time_anchor DESC, atom_id ASC；不用 DISTINCT：limit 按 len(entity_ids) 过取、Python 保留首次出现；limit=None → 该顺序下全部门控内链接原子（硬实体门的门控集合 G）；index_hint 接受但忽略
    def recent_candidates(self, scope: Scope, *, as_of: str | None = None, include_expired: bool = False, limit: int, scale_in: Sequence[str] | None = None, obs_type_in: Sequence[str] | None = None) -> list[str]
        # 新近度召回通道：scope + 有效期 [+ scale_level IN] [+ obs_type IN] ORDER BY DEFAULT_ATOM_ORDER LIMIT K_rec（ix_atoms_scope）；每条通道都是索引 SEARCH，绝不是 `SCAN memory_atoms`（test_fts_lifecycle_channels_and_query_plans 断言计划）
    def atom_attrs(self, atom_ids: Iterable[str]) -> dict[str, dict]                 # {atom_id: INDEX_ATTR_KEYS 标量（§10.3，含 atom_type），字符串 NULL 为 ""，archived: bool}；未给 attrs 时 Milvus 提供者在 upsert 时读取
    attr_listeners: list[Callable[[str, dict], None]]                              # 对有效期 / 归档状态发生变化的每个原子（archive_atom / restore_atom / set_valid_to）回调 `callback(atom_id, attrs)`；`attrs` 是 COMMIT 前在事务内读取的**完整** `atom_attrs(...)` 快照（绝不是部分差异，因此提供者的合并对 SQLite 幂等；每次提交每个变化原子一次回调——同一事务内的中间状态不重放）；COMMIT 后在 store 锁之外按提交顺序派发（每个提交事务一个票号 + 一个 Condition：较晚提交的快照绝不会被较早的超越；ROLLBACK 丢弃）；抛错的监听器只记日志、从不抛出；HMS.__init__ 只注册一次 index.update_attrs（提供者已自行订阅时跳过，SQLiteIndex 在构造时就是如此），HMS.close() 移除它
    data_generation: int; rebuild_generation: int                                  # 写入代数：本连接每条触及 memory_atoms / atom_vectors 的非 SELECT 语句推进 data_generation；其中可能删除行的子集（DELETE / DROP、任何 ROLLBACK、reset()）推进 rebuild_generation；它们是 SQLite 索引缓存（§10.3）的键
    def data_version(self) -> int                                                   # `PRAGMA data_version`：只在**其它**连接向文件提交时改变（本连接的写入不改变它）
    def load_vector_rows(self, after_rowid: int = 0) -> list[tuple[int, str, int, bytes, dict]]   # rowid > after_rowid 的 atom_vectors ⋈ memory_atoms 行，按 rowid 序：(rowid, atom_id, dims, '<f4' blob, INDEX_ATTR_KEYS 标量)——SQLite 索引缓存的增量数据源（INSERT OR REPLACE 给被替换的向量一个新 rowid，因此它会再次出现）
    # 每条 `UPDATE memory_atoms` 只改 valid_to / archived_reason（经 _queue_attrs）或非门控列（confidence、visibility、user_meta_json、updated_at）：原子的标量门控在其它情况下不可变，这正是索引缓存能按 rowid 扩展而非重建的前提（test_memory_atoms_updates_stay_listener_covered）
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
    def load_vector_blobs(self, atom_ids: Iterable[str] | None) -> list[tuple[str, int, bytes]]   # 原始 '<f4' BLOB，按首次出现顺序（None → 全部按 atom_id）；SQLiteIndex.search 用 numpy.frombuffer 构造一个矩阵
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

`insert_atom` 使用的节点类型映射 `source_kind → node_kind`：message→message、observation→observation、event→event、entity→entity、ep→ep、interface_ep→interface_ep、skill→skill、procedural→ep、feedback→observation。`graph_nodes.scale = atom.scale_level`。id 前缀（§0.1）中另外包含 `cst`（compiled_states）。

---

## §9 基础工具（A 组）

### §9.1 `hms_elite/timeutil.py`

模块 docstring：`"""Canonical UTC instants: one clock format for every table.\n规范 UTC 时间：所有表使用同一格式。"""`

- `now_iso() -> str` — 当前 UTC 时刻，32 字符定宽。副作用：无。
- `to_utc_iso(value: str | datetime | date | int | float) -> str` — 归一化为定宽 UTC ISO。入参：ISO 字符串（`Z`、`±HH:MM`、无时区者视为 UTC、纯日期补 `T00:00:00`）、datetime/date、epoch 秒。返回：32 字符字符串。副作用：无。说明：无法解析的输入 → `ValidationError(VALIDATION_004)`。
- `parse_iso(value: str) -> datetime` — 返回带时区的 UTC datetime。
- `date_of(value: str) -> str` — 取某一时刻的 `YYYY-MM-DD`。
- `age_days(t: str, as_of: str) -> float` — `max(0, (as_of - t).total_seconds()/86400)`，即天数差。
- `day_window(date: str) -> tuple[str, str]` — 当日窗口 `(T00:00:00.000000, T23:59:59.999999)`。
- `month_window(year: int, month: int) -> tuple[str, str]`；`week_window(date: str) -> tuple[str, str]`（周一至周日）。
- `next_day_midnight(date: str) -> str` — 由 `expiration_date` 得到 `valid_to`（`<date+1>T00:00:00.000000+00:00`），即次日零点。
- `shift_days(date: str, n: int) -> str`。

### §9.2 `hms_elite/utils/text.py`

模块 docstring：`"""Deterministic text utilities and EN/ZH cue lexicons shared by the rule provider, planner and scorer.\n规则提供者、查询规划器与打分器共享的确定性文本工具与中英线索词表。"""`

常量（严格照此）：
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

函数：
- `sentences(text: str) -> list[str]` — 按 `SENTENCE_SPLIT_RE` 切分，去首尾空白，丢弃空串，即分句。说明：末尾没有终止标点的句子也保留。
- `tokenize(text: str, *, cjk: str | None = None) -> list[str]` — 转小写；标识符（`IDENT_RE`）整体保留；拉丁 token 为 `[a-z0-9]+(?:['-][a-z0-9]+)*`；`cjk='bigram'` 下连续 CJK 串切成字符二元组（串长为 1 时取单字），`cjk='none'` 下每个连续串一个 token；`cjk=None` 读取当前生效的 `cjk_tokens_scope`（默认 `bigram`，§9.6）；保持原顺序，重复项保留。即分词。
- `content_tokens(text: str, *, cjk: str | None = None) -> list[str]` — `tokenize(text, cjk=cjk)` 的结果减去 `STOPWORDS_EN ∪ STOPWORDS_ZH`，每个 token 再经 `stem` 取词干。即去停用词。
- `stem(token: str) -> str` — 标识符与 CJK token 不变；`ies→y`；当 `len > 3` 且结尾不是 `ss` 时去掉末尾 `s`；当剩余部分 ≥ 4 字符时去掉 `ing`/`ed`。即轻量词干。
- `is_identifier(token: str) -> bool` — `IDENT_RE.fullmatch` 命中，或含 `./_-` 之一且至少有 1 个字母。即是否标识符。
- `identifier_tokens(text: str) -> list[str]` — `IDENT_RE.findall`，去重，保持顺序。即标识符 token。
- `EntityCandidate = NamedTuple(entity_type: str, text: str, normalized: str, key: str)` — **`key` 才是身份**：它存入 `entities.normalized_text`，也是 `MemoryAtom.entity_ids` 返回的值；`normalized` 仅作别名来源（对 FILE 而言是全小写的完整路径，存入 `entities.aliases_json`）。
- `extract_entities(text: str) -> list[EntityCandidate]` — 优先级顺序：FILE（含 `/` 或匹配 `FILE_EXT_RE` 的标识符；`key = basename.lower()`，`normalized` 为全小写完整路径）、IDENTIFIER（其余 `IDENT_RE` 命中；key 取小写）、QUOTED（`QUOTED_RE`，key 为归一化后的文本）、PROPER（不在 `STOP_CAPS` 中的 `PROPER_TOKEN_RE` token，连续 token 用一个空格连接，允许出现在句首；key 为小写并压缩空白）、CJK 专名：紧跟在 `是|在|和|喜欢|加入|负责` 之前的 2–4 字连续串（key 即该串）；按 key 去重并保留首次出现；按跨度做重叠抑制：候选按上述优先级（FILE、IDENTIFIER、QUOTED、PROPER、CJK）占用字符跨度，后来的候选若跨度与已占用的跨度重叠则丢弃（因此引号短语 `'patch src/billing/retry.ts: add exponential backoff'` 只产出 FILE 实体 `retry.ts`）。即实体候选。说明：月份名与星期名永远不会成为实体；`entity_key(text)` 就是上面这条 key 规则。
- `normalize_entity(text: str) -> str` — 去首尾空白、转小写、压缩空白。即归一化。
- `topic_tags(text: str) -> list[str]` — 取 `TOPIC_TABLE` 的命中项，按每个主题在文本中首次匹配的位置排序（位置相同则按表中顺序），每个主题只出现一次。即主题标签。
- `sentiment(text: str) -> str` — 正向词表（`like|love|enjoy|great|good|happy|passed|works|green|success|喜欢|通过|成功`）对负向词表（`hate|dislike|bad|broke|failed|regression|error|crash|失败|报错|讨厌`）；两者都命中 → `mixed`；都不命中 → `neutral`。即情感。
- `rewrite_first_person(sentence: str) -> str` — 在词边界上按顺序替换：`I've`→`User has`、`I'm`→`User is`、`I'd`→`User would`、`I'll`→`User will`、`I am`→`User is`、`I have`→`User has`、`I do not`/`I don't`→`User does not`、`I was`→`User was`、`I`→`User`、`my`→`user's`（`My`→`User's`）、`me`→`the user`、`we`/`We`→`the team`/`The team`、`our`→`the team's`；首字符大写；除非已以 `.!?。！？` 结尾，否则补一个 `.`。即第一人称改写。
- `clean_object(phrase: str) -> str` — 去首尾空白；丢弃末尾属于 `OBJ_TRAILING_CUES` 的 token；丢弃开头的 `ARTICLES`；去掉末尾的 `了|的|。`；压缩空白。即宾语清理。
- `extract_dates(text: str, *, base_date: str) -> list[tuple[str, str, str]]` — 每一个 `DATE_RES` 命中产出一个 `(kind, gte, lte)` 窗口，按文本顺序排列（`base_date` 为 `YYYY-MM-DD`；只有月份没有年份时取 `base_date` 的年份；`month_en` 要求月份 token 是独立单词，`May` 只有在后接数字、或前面有 `in|on|during|of|since|until` 时才算——所以 "may change" 永远不会被当成日期）。即日期抽取。
- `resolve_date(text: str, *, base_date: str) -> str | None` — 第一个窗口的 `gte` 日期（`YYYY-MM-DD`）。即解析日期。
- `has_action_closure(text: str) -> bool` — `ACTION_RE` 命中，且（任一句上 `OUTCOME_RE` 或 `CONSTRAINT_RE` 命中）。即是否动作闭包。
- `ordered_union(*seqs: Iterable[str]) -> list[str]` — 直接拼接，丢弃后出现的重复项；绝不排序（§0.1）。即有序并集。
- `is_first_person(sentence: str) -> bool` — `re.search(r"\b(I|I'm|I've|I'd|I'll|my|we|我|我们)\b", sentence)`（用 search，不是 match；因此 "Yesterday I visited …" 与 "Actually I switched …" 都算第一人称）。即是否第一人称。
- `sha256_hex(s: str) -> str`；`md5_hex(s: str) -> str`。

### §9.3 `hms_elite/utils/json.py`

- `canonical_json(obj: Any) -> str` — 见 §0.1。即规范 JSON。
- `remove_code_blocks(content: str | list[str]) -> str` — 与 Mem0 逐字一致：列表先 join；剥掉整串的 ` ```lang ... ``` ` 围栏；剥掉 `<think>...</think>`。即去代码围栏。
- `extract_json(text: str) -> str` — 先取第一个 ` ```json ``` ` 围栏块；否则取第一个 `{` 到最后一个 `}`；否则取第一个 `[` 到最后一个 `]`（顶层数组补救）；再否则返回 `text`。即提取 JSON。
- `parse_llm_json(raw: str, *, key: str, task: str = "") -> list | dict` — 阶梯式处理：`remove_code_blocks` → 结果为空 ⇒ `[]` → `json.loads(strict=False)` → 抛 `JSONDecodeError` 时先 `extract_json` 再 `json.loads(strict=False)` → 任何异常 ⇒ 记 `logger.error("Error parsing %s response", task)` 并返回 `[]`；取值：dict 时取 `obj.get(key, [])`，list 且 `key` 为 None 时取 `obj`，否则 `[]`；列表型 key 拿到非列表 ⇒ `[]`。即解析阶梯。说明：永不抛异常。

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

- `_UNSET = object()`（模块级哨兵，对外导出）。
- `parse_messages(messages: list[dict]) -> str` — 拼成 `"role: content\n"` 形式的行，跳过 `content=None` 的项，列表型 content 归约为其中的文本片段。即消息拼接。
- `coerce_messages(messages: str | dict | list) -> list[dict]` — `str→[{'role':'user','content':str}]`，`dict→[dict]`，list 原样透传，其余 → `ValidationError(VALIDATION_003)`；每个元素必须是带 `role` 与 `content` 的 dict（否则 `VALIDATION_003`）；列表型 content → `{'content': 拼接后的文本片段, 'dropped_parts': n}`；`content` 为 None 的元素丢弃。即消息规整。
- `validate_and_trim_entity_id(value: Any, name: str) -> str | None` — 移植自 Mem0；空值或内部含空白 → `ValidationError(VALIDATION_001, ValueError)`。即校验 id。
- `build_scope_and_metadata(*, user_id=None, agent_id=None, run_id=None, actor_id=None, input_metadata=None, context: str = "add") -> tuple[Scope, dict]` — 裁剪各 id，要求至少提供 1 个（否则 `VALIDATION_001`，`details={'provided_ids': {...}}`），深拷贝 metadata，剥掉身份键与保留键并用一条 `logger.warning` 把它们列出来。即范围与元数据。
- `strip_reserved_keys(meta: dict, *, context: str) -> dict` — 丢弃 `RESERVED_ATOM_KEYS` ∪ 身份键；只告警一次。即剥离保留键。
- `reject_top_level_entity_params(kwargs: dict, method: str) -> None` — kwargs 中出现 `user_id/agent_id/run_id` 时抛 `ValidationError(VALIDATION_008, ValueError)`。即拒绝顶层 id。
- `validate_search_params(threshold: float | None, top_k: int | None) -> None` — 要求 `threshold ∈ [0,1]`，`top_k` 为非负整数且不能是 bool；否则 `ValidationError(VALIDATION_009, ValueError)`。即校验搜索参数。
- `validate_and_trim_query(query: str) -> str` — 去空白后必须非空，否则 `VALIDATION_010`。即校验查询。
- `normalize_expiration_date(value: Any) -> str | None` — `None→None`；date/datetime/`YYYY-MM-DD` 字符串 → `YYYY-MM-DD`；其余 → `ValidationError(VALIDATION_004)`。即过期日期归一。
- `estimate_tokens(text: str) -> int` — 见 §0.1。即估算 token。
- `mask_ids(text: str) -> str` — 正则 `\b(evt|anc|obs|ep|atm|mem|ent|edg|iep|fb|ckp|trn|pkg|hst)_[0-9a-f]{6,12}\b` → `<prefix>`（供测试的 golden 辅助函数使用；放在这里是为了 T 组与 X 组都能 import）。即掩码 id。

### §9.6 CJK 文本开关（`HMSConfig.text.cjk_tokens`、`IngestConfig.zh_rules`）

两个开关的默认值都是历史行为，因此英文存储与每一次已记录的 EgoLife 运行逐字节不变。它们为 EgoLife-ZH 实验而设（`docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`：运行 A = 中文数据走英文中心的文本流水线，运行 B = 同一数据打开两个开关；A − B 度量中文支持挽回了多少）。

- `text.cjk_tokens: Literal["bigram", "none"] = "bigram"` — `tokenize` / `content_tokens` 在关键词通道**两侧**的 CJK 模式：写侧 `schema.fts_tokens`（atom_fts）与 `scoring.atom_keyword_tokens`；读侧 `plan_query`、`mvmi.query._plan_keyword_tokens`、`scoring.prepare_query`，以及其它每个消费者（`reranker/overlap.py`、`compiler/coverage.py`、`ingest/extraction.py statement_tokens / verdict_similarity`、`ops/consolidation.py`、`llms/rule.py`）。`bigram` = 重叠字符二元组（今日行为）；`none` = 每个 CJK 连续串一个 token，即 unicode61 一类分词器的做法——中文关键词查询只在整串完全相同时命中（运行 A 的基线）。机制：`utils.text.cjk_tokens_scope(mode)`（ContextVar `hms_cjk_tokens`，默认 `bigram`；`cjk_tokens_mode()` 读取；`CJK_TOKEN_MODES`），由 `utils.text.text_scoped` 装饰器在 `memory.main.TEXT_SCOPED_METHODS` 列出的每个方法上进入（35 个 `HMS` 公共方法——全部数据方法与 §11 状态族；排除 `from_config`、`project`、`state`、`close`、`usage`、`trace`、`start_daemon` / `stop_daemon`（守护线程调用的是已包装的 `consolidate`）、`history`、`vacuum`、`get`），也套在 `Ingestor.ingest_messages` / `ingest_events`、`StateCompiler.compile` 与 `StateScheduler.run` 上，使独立使用的协作对象与守护线程自行设置模式（`AsyncHMS` 在 `asyncio.to_thread` 内运行已包装的同步方法）；`tokenize(text, *, cjk=None)` / `content_tokens(text, *, cjk=None)` 在 `cjk` 为 None 时读取作用域；`scoring._content_vocabulary(text, cjk)` / `_key_tokens(key, cjk)` 的 lru_cache 以模式为键；`stem` 对 CJK token 原样返回。持久化：meta 键 `text_cjk_tokens`（缺失即 `bigram`；只在模式不是 `bigram` 或该键已存在时写入，且在 `unversioned()` 内，因此默认存储不多出 meta 行）；配置模式与存储模式不同时，`HMS.__init__`（`_sync_fts_mode`）调用 `StateStore.rebuild_fts()`——在 `unversioned()` 内的一个事务里 `DELETE FROM atom_fts` + `schema.backfill_fts`（不推进写入版本、不写 history 行）。不提升 `SCHEMA_VERSION`（DDL 不变）。测试：`tests/test_text_cjk.py`。
- `ingest.zh_rules: bool = False` — 紧凑 / 行式推导的中文陈述规则，沿 `extract_all(..., zh_rules)` / `extract_observations(..., zh_rules)` → `_compact_observation_from_item` → `ingest.derive.expand_compact_item(..., zh_rules)` / `expand_lines_ep(..., zh_rules)` → `llms.rule.statement_triple(text, obs_type, extractor, *, zh_rules)` 传递：开头主语来自抽取器的中文人名规则（杰克说… → 杰克），开头的 我（"Me" 的术语表形式）即 `user`，我们 / 咱们 即 `project`（与英文 `we` 的映射一致），其它代词串（他 / 她 / 他们 / 大家 / 你 …）不指名任何人但被跨过，使其后的动词被读取（她喜欢吃披萨 → None / likes / 吃披萨；我们打算晚上吃火锅 → project / plans / 火锅），回退谓词把剩余部分开头的动词经 `VERB_LEXICON_ZH` 映射（`predicate_of(text, *, zh_rules)`，在可跳过副词的开头用正则——绝不经 token，因为二元组下单字动词不是 token），开头动词映射为 `plans` 的 `temporal_fact` 即 `plans`，fact / temporal_fact 取动词之后的子句为宾语（`_zh_clause` / `_zh_objects`：穿过轻动词 和 / 跟 / 给 + 已知（或未列出的 2–4 字）人名与 计划 / 打算 / 要 …，人名之前的体貌助词 了 / 过 / 着 被跳过（给了杰克一杯咖啡 → 一杯咖啡），在-处所短语在下一个动词处截断，去掉开头的 了 / 过 / 着；否则取其余实体提及）。介动词 和 / 跟 / 给 之后，fact 的谓词取下钻到达的动词（爱丽丝和塔莎去了超市 → went_to，塔莎跟杰克说… → says，和舒尔在阳台… → is_at，给了杰克一杯咖啡 → gives）；`VERB_LEXICON_ZH` 把单独的 和 / 跟 映射为 `states`（'with' 不在任何谓词词表里），计划 / 打算 … 保持 `plans`。读主语之前跳过开头的 `ZH_NAME_STOP` / 时间词（`utils.text.zh_name_stop_prefix`：今天杰克和爱丽丝一起去了公园 → 杰克 / did / 公园；下午我去了超市 → user），主语与动词之间的副词 / 时间词前缀与实体规则共用同一个备选式（`utils.text.ZH_LEAD_ADVERB_ALT`，因此 杰克晚上去了超市 → 杰克 / 超市）。`mention_texts(text, extractor, *, zh_rules)` 与 `observation_topics(text, obs_type, *, zh_rules)` 透传该标志；`utils.text.sentiment(text, *, zh_rules)` 加入 `_POSITIVE_SENT_ZH_RE` / `_NEGATIVE_SENT_ZH_RE`（不喜欢 之类的否定短语只计负面；正面词表里的 爱 只以受限形式出现——喜爱 / 热爱 / 爱吃 / 爱上 / 爱玩——因此术语表人名 爱丽丝 绝不使陈述变为正面：爱丽丝讨厌洗碗 → negative，爱丽丝去了超市 → neutral）；`topic_tags(text, *, zh_rules)` 追加 `TOPIC_TABLE_ZH`（food、shopping、games、music、travel、housework、work；两表都有的主题保留最早命中）。`utils.text.extract_entities(text, *, zh_rules=False, names=())`：`zh_rules` 加入 `_zh_name_candidates`——紧接 `ZH_STATEMENT_VERBS` 动词之前的 2–4 字连续串开头（二者之间的前导副词 / 时间词被跳过：`ZH_LEAD_ADVERBS` = 也 / 都 / 一起 / … ∪ `ZH_TIME_WORDS` = 今天 / 晚上 / 下午 / 周末 / 然后 / …，与 `llms.rule` 共用同一个备选式 `ZH_LEAD_ADVERB_ALT`；先剥离 今天 / 下午 之类的 `ZH_NAME_STOP` 前缀——`ZH_NAME_STOP` ⊇ `ZH_TIME_WORDS`；`zh_name_ok` 拒绝代词开头、停用词，以及以陈述动词或前导副词 / 时间词**开头或结尾**的开头——只有这两个位置上这类词才构成边界——因此 打算晚上、杰克晚上、在客厅 永不成为人名而 王和平 可以；末字是这类词的人名（李刚：刚 是前导副词）仍被拒绝，因为该后缀与 李 + 刚 无法区分——已记录的限制，只有 `names` 表能接住；在此之前输出的是 4 字 ZH 开头 李刚说他（明天要 之前）——它来自 `_zh_name_candidates`，因为既有的 `_CJK_PROPER_RE` 前瞻在 `zh_rules` 下不运行），或 和 / 跟 / 给 之后（跳过体貌助词 了 / 过 / 着：跟了塔莎 → 塔莎，给了杰克 → 杰克）、后接动词、量词 / 助词（`_ZH_NAME_TAIL_RE`：一个 / 一杯 / 一本 / … / 的 / 们——不含 了，人名之后从不接它）或串尾的 2–4 字串；`names` 在任何出现位置精确匹配（长者优先）。候选优先级：FILE 0 < IDENTIFIER 1 < QUOTED 2 < PROPER 3 < NAME 4 < ZH 5 < CJK 6（`_CJK_PROPER_RE` 不变且最低，且只在 `zh_rules` 为 False 时运行：它在 是 / 在 / 和 / 喜欢 / 加入 / 负责 之前取的 2–4 字开头——EgoLife-ZH 冒烟存储里的 要求团队、都写、硬盘假装、他的家乡、做的布线、并让大家——是片段而非人名，因此 `zh_rules` 下一段文本的实体恰为 `names` 术语表命中、`_zh_name_candidates` 开头、FILE / IDENTIFIER / QUOTED 与拉丁 PROPER 串；`zh_rules=False` 的输出与此前逐字节一致——`tests/test_zh_rules.py::test_zh_rules_skip_the_legacy_cjk_lookahead`）。`RuleEntityExtractorConfig(zh_rules: bool = False, names: list[str] = [])`；默认下身份为 `rule@entities-v1`，任一设置时为 `rule@entities-zh-v1`（§10.5）；`ingest.zh_rules` 为真且提供者为 `rule` 时，`HMS.__init__` 把 `zh_rules=True` 注入 `entity_extractor_config` 的副本。读侧看到同一个抽取器：`mvmi.planner.plan_query(query, *, as_of, mode, extractor=None)` 给了抽取器时用 `extractor.extract(text)` 取实体线索（否则 `utils.text.extract_entities`，与默认规则抽取器逐字节一致），`mvmi.query._scoring_plan`（没有 `plan` 的文本 AxisQuery 的临时规划）、`mvmi.query.search(..., extractor)`、`StateCompiler(..., entity_extractor)`（`context()` / `compile()`）与 `compiler.coverage.extract_requirements(..., extractor)` 亦然；`HMS.search` / `context` / `compile` / `coverage` 传入 `self.entity_extractor`，因此 杰克买了什么？ 这样的中文提问在 `zh_rules` + `names` 下得到 `plan.entity_hints` [杰克]、`entity_boost` > 0 且 `recalled_by` 含 `'entity'`，默认下三者皆无（`tests/test_zh_rules.py::test_review_fixes_sentiment_names_pronouns_and_read_side_entity_axis`）。EgoLife-ZH 计划的运行 B 因此检查 `questions_elite_rows.jsonl` 的 `channels.entity_hits > 0`（基准脚本从 `score_details['entity_boost']` 读该信号，它是 `scoring.score_details` 写出的唯一实体键）。推导出的中文主语经未改动的行式说话人规则（§6.6）到达 `Observation.speaker` → `actor_id`。规则 LLM 自己的观察规则（§11.1）与 `RuleLlmConfig.language` 不动。测试：`tests/test_zh_rules.py`。
- 行式提示词：`MEMORY_EXTRACTION_LINES_PROMPT` 在多说话人句之后带有一句 "Write every statement in the language of the source messages (Chinese messages → Chinese statements, names as written); never translate."（该提示词没有 sha 钉住；`KEY_SENTENCES` 与四个带哈希的提示词不动；已记录的 `results/real/elite_lines_full204` 回复早于这句话——回放按原样提供记录的回复，可比性不受影响）。
- `estimate_tokens` 不变：每个 CJK 字 3 个 UTF-8 字节 → 每字 0.75 个估算 token，绝不低于中文分词器约 0.67（每 1.5 字 1 个 token）的实际值，因此中文行上的编译器预算偏保守而不会少算（`tests/test_foundation.py`）。
- 度量口径说明：`cjk_tokens='none'` 有意全局生效，因此它也改变 `statement_tokens` / `verdict_similarity`（整串 token 几乎不重叠：多数 NONE / REVISE 裁决落为 ADD）、重叠重排、覆盖选择、整合去重与规则句子门；故运行 A − 运行 B 混合了关键词通道损失与修订守卫损失，计划文档按运行报告被守卫拒绝的裁决数。数据来源：`HMS.last_downgraded`（最近一次 `add()` / `add_events()` 的 `IngestResult.downgraded` 条目；Mem0 信封仍为 `{'results': rows}`），`benchmarks/egolife/real_arms.EliteRealArm._add` 把它以 `downgraded: {裁决: 数量}` 附到信封上，`retain_with_bisect` 按子 add 与按批次记录（`retain_elite.jsonl`），`run_real.summarize` 按运行求和（`engines.elite.retain.downgraded`）。

---

## §10 Provider 层（B 组）

### §10.1 LLM（`hms_elite/llms/`）

```python
class LLMBase(ABC):
    def __init__(self, config: BaseLlmConfig | dict | None = None) -> None      # dict → the provider's config class; None → defaults
    @abstractmethod
    def generate_response(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs) -> str | dict
        # kwargs: response_format={'type': 'json_object'}; returns str, or {'content': str|None, 'tool_calls': [...]} when tools were used (Mem0 contract)
    def generate_json(self, task: str, *, system: str, user: str) -> list | dict
        # messages=[{'role':'system','content':system},{'role':'user','content':user}], response_format json_object;
        # transport exception → raise LLMError(f"{task} failed: {e}", details={'task': task, 'provider': self.identity}) from e; parse via parse_llm_json(key=ROOT_KEY[task])
    def generate_lines(self, task: str, *, system: str, user: str) -> str             # memory_extraction_lines 路径（§6.6）：同样的两条消息，不带 response_format，原样返回文本（dict → content，None → ''）；由 ingest.lines.parse_lines 解析
    def generate_text(self, task: str, *, messages: list[dict]) -> str                 # procedural_summary path; remove_code_blocks applied
    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None   # 视觉钩子，非抽象：None = 不支持（base、rule、未配置 image_description 的 scripted）；openai → chat.completions 的 image_url 部件，anthropic → messages 的 image 块（data URI → base64 来源，http → url 来源）；由摄取流水线在无锁、事务 A 之前调用
    @property
    def identity(self) -> str                                                          # 'rule@v1' | 'scripted@v1' | 'openai:gpt-5-mini' | 'anthropic:claude-sonnet-4-6'
ROOT_KEY = {"observation_extraction": "observations", "ep_extraction": "eps", "revision_decision": "decisions", "edge_proposal": "edges", "coarsen": "interface_ep"}
```

- `RuleLLM(LLMBase)`（`rule.py`，见 §11）：`generate_response` 读取 `task_of(system)`；任务未知 → `LLMError(LLM_002)`；user 提示中出现 `RAISE: <msg>` 这样一行 → `LLMError(LLM_001, msg)`（测试钩子）；随后派发到 `_task_<name>(sections)`；`calls: list[dict]` 记录 `{'task', 'sections'}`。
- `ScriptedLLM(LLMBase)`（`scripted.py`，见 §11.7）。
- `OpenAILLM(LLMBase)`（`openai.py`）：模块级守卫 `try: from openai import OpenAI except ImportError as e: raise DependencyError("The 'openai' package is required for the openai provider", details={'package': 'openai', 'extra': 'openai'}) from e`；客户端 `OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"), base_url=config.openai_base_url or os.getenv("OPENAI_BASE_URL"))`；调用 `chat.completions.create(model, messages, temperature, max_tokens, top_p, response_format=kwargs.get('response_format'), tools=..., tool_choice=...)`；`_parse_response(response, tools)` → `str | {'content','tool_calls': [{'name','arguments': json.loads}]}`。
- `AnthropicLLM(LLMBase)`（`anthropic.py`）：对 `anthropic` 做守卫；system 消息拆出来放进 `system=` 参数；调用 `messages.create(model, system, messages, max_tokens[, temperature])`——仅当已安装 SDK 的 `Messages.create` 仍声明 `temperature` 时才转发它（构造时用 `inspect.signature` 检查一次：`anthropic>=1.0` 移除了采样参数、传入即抛 `TypeError`，因此 §20.1 的 `anthropic>=0.40` extra 必须在 0.x 与 1.x 上都能用；不传其它采样参数）；当 `response_format` 要求 JSON 时，在最后一条 user 消息末尾追加 `"\nRespond with JSON only."`；返回文本。

Docstring 约定：`generate_response` — "Produce a completion for chat messages | 生成对话补全"；入参：messages、tools、tool_choice、kwargs；返回：str 或 dict；副作用：无（真实提供者会有网络调用）。

### §10.2 Embedding（`hms_elite/embeddings/`）

```python
class EmbedderBase(ABC):
    def __init__(self, config: BaseEmbedderConfig | dict | None = None) -> None
    @abstractmethod def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]
    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]    # sequential default; per-item failure → EmbeddingError(EMBED_001) from e
    @property def dims(self) -> int
    @property def identity(self) -> str                                                          # 'hash@64' | 'openai:text-embedding-3-small@1536'
```

- `HashEmbedder`（`hash.py`）：先 `text.lower()`，再包成 `f" {text} "`；对 `n ∈ [ngram_min, ngram_max]` 取字符 n-gram；桶位 `int(hashlib.sha256(ngram.encode()).hexdigest()[:8], 16) % dims`；第 9 位十六进制数字为偶数时符号取 `+1`，否则取 `-1`；逐项累加；做 L2 归一化（空文本得到零向量，即全 0）；纯 Python 实现；跨进程结果确定一致。`dims` 取自配置（默认 64）。
- `OpenAIEmbedder`（`openai.py`）：带守卫；`embeddings.create(model, input=text, dimensions=embedding_dims)`；`identity = f"openai:{model}@{dims}"`。

### §10.3 索引（`hms_elite/index/`）——可插拔的向量索引

```python
INDEX_ATTR_KEYS = ('user_id', 'agent_id', 'run_id', 'time_anchor', 'valid_from', 'valid_to', 'archived', 'scale_level', 'obs_type', 'atom_type')   # 提供者可随向量保存的标量门控；字符串 NULL 为 ""，时刻为定宽 UTC ISO（字典序 == 时间序），archived: bool
def atom_index_attrs(atom: MemoryAtom) -> dict      # 已存原子的 INDEX_ATTR_KEYS
def normalize_attrs(attrs: Mapping | None) -> dict  # 只保留 INDEX_ATTR_KEYS；None → ""，时刻 → to_utc_iso，archived → bool
def select_top_k(query, ids, matrix, *, top_k, rows=None, scores=None) -> list[tuple[str, float]]   # 对 (n, dims) float32 或 float64 取值矩阵的有界 top-k：一次矩阵乘**挑选**行（余量 2e-12，零钳位区间整体重算），`dot` 重算分数 → 与 rank_hits(...)[:top_k] 逐位相同；rows = 候选行位置（None → 全部），scores = 与 rows 对齐的预先算好的 float64 乘积（缓存的整矩阵乘积的掩码切片）
@dataclass(frozen=True) class IndexFilter: scope: Scope | None = None; as_of: str | None = None; include_expired: bool = False; scale_in: tuple[str, ...] | None = None; obs_type_in: tuple[str, ...] | None = None; time_from: str | None = None; time_to: str | None = None; atom_type_in: tuple[str, ...] | None = None
    # axis_select 的 SQL 门控的数据形式：未设置的 scope 键不加谓词（Scope.matches）；include_expired=False ⇔ archived == false AND valid_from <= as_of AND (valid_to == "" OR valid_to > as_of)；白名单；time_anchor 闭区间（没有 time_anchor 的行不匹配任何时间界）

class IndexBase(ABC):
    def __init__(self, config: BaseIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64) -> None
    @abstractmethod def upsert(self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping | None = None) -> None   # attrs：INDEX_ATTR_KEYS 标量，无标量的提供者忽略
    def upsert_batch(self, items: list[tuple[str, list[float]]], *, embedder: str, attrs: Mapping[str, Mapping] | None = None) -> None   # 顺序默认实现；attrs = {atom_id: attrs}
    @abstractmethod def get(self, atom_id: str) -> list[float] | None
    @abstractmethod def delete(self, atom_id: str) -> None
    @abstractmethod def search(self, vector: list[float], *, candidate_ids: list[str] | None = None, filter: IndexFilter | None = None, top_k: int) -> list[tuple[str, float]]
        # 余弦相似度钳制 max(0.0, dot)（单位向量），按（score 降序，atom_id 升序）排序，**至多** top_k 行；candidate_ids=None → 不限定 id；没有向量的 id 被跳过；filter 只在 supports_filter 的提供者上可用（其它提供者抛 ValidationError(VALIDATION_000)，提示调用方用 StateStore.axis_ids 门控）；两者同时给出 → 交集
    @abstractmethod def count(self, filter: IndexFilter | None = None) -> int   # filter 仅限 supports_filter 的提供者（其它抛 VALIDATION_000）；没有向量的原子从不计入
    @abstractmethod def reset(self) -> None
    def update_attrs(self, atom_id: str, attrs: Mapping) -> None   # 非抽象钩子，默认空操作；StateStore.attr_listeners 在 COMMIT 后按提交顺序、携带完整 atom_attrs 快照调用它（archive / restore / set_valid_to）
    def gate_ids(self, filter: IndexFilter) -> list[str] | None     # 在绑定的 store 上把 IndexFilter 评估为 store.axis_ids(order=None)（无 store 时为 None）：下推提供者的 SQL 退路
    def close(self) -> None                                         # 非抽象钩子，默认空操作；HMS.close() 在 store.close() 之前调用它
    @property supports_filter -> bool                               # 默认 False
    def info(self) -> dict                                                                        # {'provider', 'dims', 'count', 'numpy': bool}
```

- `SQLiteIndex`（`sqlite.py`）：通过 `store.upsert_vector / delete_vector / clear_vectors` 写 `atom_vectors`（同一连接、同一事务），并经每个维数一个的进程内 `_VectorCache` 排序——已存 float32 向量的 float64 矩阵，外加 INDEX_ATTR_KEYS 标量（user_id / agent_id / run_id / scale_level / obs_type / atom_type 为整数码，三个时刻为定宽 `S40` 字节，archived 为布尔），倍增扩容。缓存以 `(store.data_generation, store.rebuild_generation, store.data_version())` 为键：键未变则完全不发 SQL；data_generation 变化时用 `store.load_vector_rows(last_rowid)` 增量扩展（追加，INSERT OR REPLACE 则原地覆盖）并应用排队的 `update_attrs` 变化（提供者在构造时把自己注册进 `store.attr_listeners`；直接调用 `update_attrs` 与 Milvus 一样覆盖缓存标量）；rebuild_generation 或 data_version 变化（DELETE / DROP / ROLLBACK / reset / 其它连接提交）则从头重建；出现其它维数的行时缓存标记为不可用，提供者退回对 `store.load_vectors` 做 `rank_hits`。`supports_filter=True`：IndexFilter 是缓存标量上的 numpy 掩码，语义与 SQL 门控完全相同（未设置的 scope 键不加谓词；`include_expired=False` ⇔ 未归档且 valid_from <= as_of 且 (valid_to == "" 或 valid_to > as_of)，`as_of` 默认 store.now()；白名单经整数码；NULL 的 time_anchor 从不满足时间边界），对最近一次 (缓存键, 过滤器) 记忆化，使同一查询的 `search` 与 `count(filter)` 共享；`candidate_ids` 是行聚集；至少四分之一的行合格时 `search` 对整个矩阵做一次乘积（否则只对聚集的行），再把掩码后的分数交给 `select_top_k(rows=, scores=)` → 与同一批向量上的 `rank_hits` 逐位相同，从不水合；无 numpy（`use_numpy='never'` 或未安装）时过滤器按 `gate_ids`（SQL）求值。`close()` 取消监听并丢弃缓存；`info()` 保持四个键；三态 numpy 哨兵只记一次日志。**锁序**：`search()` / `count(filter)` 先取 `store.lock`（可重入）再取索引锁，因为同步缓存要读 store；因此在 `store.transaction()` 内发起的搜索绝不会与并发搜索死锁（`tests/test_index_sync.py::test_search_inside_transaction_never_deadlocks`）。
- `MemoryIndex`（`memory.py`）：底层是 `dict[str, list[float]]`；契约相同；`reset()` 清空；`search` 通过缓存的 float32 矩阵快照（每次写入使其失效）排序，n <= top_k 时对字典快照做 `rank_hits`，因此并发的 `upsert()`（AsyncHMS 工作线程）不会在迭代中途改变其大小；`supports_filter=False`。
- `MilvusIndex`（`milvus.py`，提供者 `milvus`，extra `milvus`）：`MilvusIndexConfig(uri=$HMS_DIR/milvus.db, token, db_name, collection='hms_atoms', metric='COSINE', index_type='AUTOINDEX'|'HNSW'|'FLAT', M, efConstruction, ef, consistency_level, batch_size, candidate_chunk, ef_recall_factor=4, scalar_index='INVERTED'|'none')`；本地 uri 必须以 `.db` 结尾（milvus-lite），`http/https/grpc/grpcs/tcp/unix` uri 指向服务器。`pymilvus` 在构造函数里惰性导入（`ImportError → DependencyError(DEPS_001, extra='milvus')`）；集合 `(atom_id VARCHAR pk, vector FLOAT_VECTOR(dims), embedder VARCHAR, user_id / agent_id / run_id VARCHAR(256), time_anchor / valid_from / valid_to VARCHAR(40), scale_level VARCHAR(16), obs_type VARCHAR(32), atom_type VARCHAR(16), archived BOOL)` 被创建（user_id、agent_id、run_id、scale_level、obs_type、atom_type、valid_to、archived 上建 `scalar_index` 种类的标量索引；milvus-lite 上每字段约 0.5 s、每集合一次）或按工厂的 `dims` 与字段集合校验（维数不一致 / 外来集合 / 缺少标量（含 `atom_type`）的集合 → `CONFIG_001`，建议：换集合名或 `consolidate()`），然后 `load_collection`（新进程重新打开的集合处于 released 状态，否则 search 会抛 code 101）。`upsert` 分层叠加标量：默认值，再 `store.atom_attrs`（一次批量 SELECT；调用方给全所有键时跳过——摄取传入 `attrs=atom_index_attrs(stored)`），最后调用方的 `attrs`；`update_attrs` 读回该行、合并、upsert（Milvus 没有原地标量更新）。`supports_filter=True`：`filter_expr(IndexFilter)` 渲染为 `user_id == "alice" and archived == false and valid_from <= "T" and (valid_to == "" or valid_to > "T") and scale_level in [...] and obs_type in [...] and atom_type in [...] and time_anchor >= "A" and time_anchor <= "B"`（json.dumps 转义；`as_of` 默认 `now_iso()`）；单独 `search(filter=)` 是**一次** `limit=top_k` 的原生过滤 ANN，HNSW 下 `ef = max(ef, limit, ef_recall_factor·limit)`（RETRIEVAL_DIAGNOSIS T6 召回缺口：调用方让小的硬门集合不走 ANN）；与 `candidate_ids` 同时给出时表达式 AND 进每个 `candidate_chunk` id 块（精确交集），且 HNSW 下 `ef` 提到分块大小（否则 30k 行上 1024 个 id 的过滤只返回寥寥几个命中）。`count(filter)` 是带同一表达式的 `count(*)` 查询——milvus-lite 上是全量扫描（N=30k 时 556 ms），因此检索池从不调用它（下推时 `gate_count = -1`）。命中经 `rank_hits` 基于已存 float32 向量重算，截断结果触及 0 分时零分尾部从全部门控候选重建，因此结果与穷举式提供者相等（FLAT 精确；HNSW / AUTOINDEX 受 ANN 召回限制）；客户端失败包装为 `StoreError(STORE_001)`；`close()`（幂等，`HMS.close()` 会调到）释放客户端，之后的调用抛 `StoreError`；`info()` 增加 `uri`、`collection`、`index_type`。SQLite 仍是唯一真源：Milvus 标量过期只会**漏检**一行（调用方水合后再校验），绝不会**误检**。

### §10.4 工厂（`hms_elite/utils/factory.py`）

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
class LlmFactory(_Factory): kind = "llm"            # create(provider, config, *, entity_extractor=None)：构造函数声明了该参数（RuleLLM）时转交，否则丢弃
class EmbedderFactory(_Factory): kind = "embedder"
class IndexFactory(_Factory): kind = "index"        # create(provider, config, *, store, dims)
class EntityExtractorFactory(_Factory): kind = "entity_extractor"   # §10.5
```

### §10.5 实体抽取器（`hms_elite/entities/`）——可插拔的实体抽取器

```python
class BaseEntityExtractorConfig(HMSModel)                                 # extra='forbid' + did-you-mean 提示；无共享字段
class EntityExtractorBase(ABC):
    def __init__(self, config: BaseEntityExtractorConfig | dict | None = None) -> None
    @property def identity(self) -> str                                    # 'rule@entities-v1' | 'spacy@<model>'
    @abstractmethod def extract(self, text: str) -> list[EntityCandidate]  # 沿用 utils.text 的顺序与按 key 去重语义；纯函数
Span = tuple[int, int, EntityCandidate]
def merge_candidates(text: str, base: Sequence[EntityCandidate], spans: Sequence[Span]) -> list[EntityCandidate]
    # 从左到右在 `text` 中定位 `base`（单空格拼接的 PROPER 连串回退到 `\s+` 正则）；与已定位规则区间重叠的 NER 区间被丢弃；
    # 结果按起始偏移排序（并列时规则在前），按 key 去重（保留首个）；规则列表总是结果的子序列
```

- `RuleEntityExtractor`（`rule.py`，提供者 `rule`，默认）：默认配置下 `extract(text)` 即 `extract_entities(text)`——逐字节一致；`RuleEntityExtractorConfig(zh_rules: bool = False, names: list[str] = [])`（§9.6）——任一设置时 `extract(text) = extract_entities(text, zh_rules=, names=)`，身份为 `rule@entities-zh-v1`。
- `SpacyEntityExtractor`（`spacy.py`，提供者 `spacy`，extra `spacy`）：`SpacyEntityExtractorConfig(model="en_core_web_sm", labels=DEFAULT_LABELS, disable=[], strip_articles=True)`；`labels` 把 spaCy NER 标签映射到 `EntityType` 值（默认 PERSON/ORG/GPE/LOC/FAC/NORP/PRODUCT/EVENT/WORK_OF_ART/LAW/LANGUAGE → `PROPER`；DATE/TIME/CARDINAL/… 未映射 → 丢弃；非 EntityType 的值 → CONFIG_001）。构造函数 `(config=None, *, nlp=None)`：`nlp` 注入现成流水线（测试用），否则 `load_pipeline(model, disable=...)` 惰性导入 `spacy`——`ImportError → DependencyError(DEPS_001, details={'package': 'spacy', 'extra': 'spacy'})`，模型缺失 `OSError → DependencyError(details 另含 'model'，suggestion 'python -m spacy download <model>')`。`extract`：`merge_candidates(text, extract_entities(text), spans)`，每个 `doc.ents` 区间先去掉开头冠词（`strip_articles`）与结尾 `'s`，再构成 `EntityCandidate(type, surface, normalize_entity(surface), entity_key(surface))`；空白文本直接返回 `[]`，不运行流水线。
- 接线（每个门面一个实例）：`HMS.__init__` 构造 `self.entity_extractor = EntityExtractorFactory.create(cfg.ingest.entity_extractor, cfg.ingest.entity_extractor_config)`，并交给 `LlmFactory.create(..., entity_extractor=)`（→ `RuleLLM.entity_extractor`，所有 `entity_mentions` 都用它）、`Ingestor(..., entity_extractor=)`（→ `create_atom/revise_atom/conflict_atom(extractor=)` → `link_atom_axes(extractor=)`，以及 `close_structured_events(extractor=)`）、`FeedbackProcessor(..., entity_extractor=)` 与 `import_state_package(..., entity_extractor=)`。每个 `extractor` 参数默认 `None` = 规则提供者，因此门面之外的调用方（图算子、consolidation 的实体原子）保持现有行为。查询侧线索（`mvmi/planner.py`）仍用 `extract_entities`。

---

## §11 `rule` LLM——逐任务的确定性启发式（`hms_elite/llms/rule.py`，B 组）

模块 docstring：`"""Offline rule-based LLM: recognizes the HMS_TASK marker and answers each prompt contract with regex heuristics.\n离线规则 LLM：识别 HMS_TASK 标记，用正则启发式回答每个提示词契约。"""` 用途：确定性的默认提供者。输入：§6 构造的 system+user 提示。输出：符合 §6.3 的 JSON 字符串。副作用：无。

公共流水线：`sections = parse_sections(user_prompt)`；每个任务只读自己文档里写明的 section；输出为 `json.dumps(obj, ensure_ascii=False)`；`RuleLlmConfig.language` 选择词表（`auto` 表示中英词表同时生效）。所有正则都来自 `utils/text.py`（§9.2）。`strict=True` 会禁用规则 R9（兜底 fact）。**§11 中的每一条正则，除非模式本身以 `^` 开头，否则一律用 `re.search` 施加**（因此只锚定了 `$` 的 `PREF_RE` 能匹配 "Actually I switched to black coffee now"）。

**公开的纯函数**（位于 `hms_elite/llms/rule.py` 模块级，由 G 组 import——绝不重复实现）：`decide_revision(new: dict, old: dict) -> tuple[Decision, str]`（对单个配对套用 §11.3 的表；两个 dict 分别带着 `## New Memories` / `## Candidate Existing Memories` 的键），`propose_edges(nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> list[dict]`（§11.4；返回契约中的 `edges` 列表），`coarsen_rule(support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str) -> dict`（§11.5；返回 `interface_ep` 字典），以及常量 `EXCLUSIVE`、`PREDICATE_FAMILIES = {'PREF': {...}, 'IDENTITY': {...}, 'CONSTRAINT': {...}}` 和 `family(predicate) -> str`。`RuleLLM._task_*` 各处理器只是薄封装，把解析好的 sections 交给这些函数。同样公开、由 C 组（`ingest/derive.py`，§6.5）以及 R1–R9 规则自身 import 的还有：`statement_triple(text, obs_type, extractor=None) -> {'subject', 'predicate', 'objects', 'relation'}`（§6.5 的第三人称陈述解析器：主语 = 开头的侧别 / PROPER 实体或 None；谓词 / 宾语 / 关系线索按类型由 R1–R9 线索表读出）、`constraint_objects(rest) -> list[str]`（R2 / R3 的宾语规则）、`predicate_of(text) -> str`（R9 词表谓词）、`observation_topics(text, obs_type) -> list[str]`（`TOPIC_TABLE` 命中加类型标签，最多四个）、`mention_texts(text, extractor=None) -> list[str]`（实体提及文本，去掉以代词开头的 CJK 片段）。

### §11.1 `observation_extraction`

对 `## New Messages` 中 `role ∈ {user, assistant}` 的每条消息（system 永远不会出现），再对 `sentences(text)` 的每一句：跳过以 `?`/`？` 结尾的句子、问候语（`^(hi|hello|hey|thanks|thank you|ok|okay)\b`）、以及拉丁 token 数 `< 3` 且不含 CJK 的句子。规则按顺序施加；**首个命中者胜出**；每句最多产出一条 observation。每条规则都作用在剥掉终止标点的句子上；`text` 字段再把句号补回去。 `IngestConfig.zh_rules`（§9.6）不改变这些规则与 `RuleLlmConfig.language`；它只改变紧凑 / 行式推导（`statement_triple`、`predicate_of`、实体 / 情感 / 主题词表）。

| # | 规则（英文；中文孪生） | type / relation / predicate / objects / sentiment / confidence |
|---|---|---|
| R1 PREFERENCE | `PREF_RE = r"\b(?P<subj>I|we)\s+(?:really\s+|absolutely\s+|still\s+|now\s+|also\s+)?(?:(?P<neg>don't|do not|no longer)\s+)?(?P<verb>like|love|prefer|enjoy|adore|hate|dislike|can't stand|cannot stand|switched to|switch to)\s+(?P<obj>[^,.;!?]+?)(?:\s+(?:over|rather than|instead of|more than)\s+(?P<alt>[^,.;!?]+))?\s*$"`（re.I，作用于剥掉终止标点的句子）；中文孪生 `(?P<subj>我|我们)(?:现在|已经|还是|也)?(?:很|更)?(?P<verb>喜欢|偏好|讨厌|不喜欢|改喝|改成|换成|爱喝)(?P<obj>[^，。；！？]+)` | `preference_event`；predicate：like/love/enjoy/adore/喜欢/爱喝 → `likes`，prefer/偏好 → `prefers`，hate/dislike/can't stand/讨厌/不喜欢 → `dislikes`；`switched to|switch to|改喝|改成|换成` → `likes` 且 `relation=supersession`（`moved to` 属于 R5，不在此处）；`neg` 使 likes↔dislikes 翻转，且 `no longer` 时置 `relation=supersession`；命中 `alt` → `relation=comparison`，objects 为 `[clean_object(obj), clean_object(alt)]`，predicate 为 `prefers`；否则 objects 为 `[clean_object(obj)]`；若 `SUPERSESSION_CUES_EN|ZH` 匹配该句 → `relation=supersession`、confidence `0.9`，否则 `0.8`；likes/prefers 的 sentiment 为 `positive`，dislikes 为 `negative` |
| R2 CONSTRAINT | `CONSTRAINT_RE`（整句匹配，re.I）或 `CONSTRAINT_ZH_RE` | `constraint` / `constraint` / `must_not`；`subj ∈ {I, We}` 时 subject 为 `user`，否则为 `project`；`rest` 有 ≥2 个 token 时 objects 为 `[clean_object(rest 去掉首个 token)]`，否则为 `[clean_object(rest)]`；sentiment `neutral`；confidence `0.9` |
| R3 MUST | `MUST_RE` 命中，且该句为祈使句、或主语为 `we|you|the team` 且不含否定 | `constraint` / `constraint` / `must`；objects 同 R2；confidence `0.85` |
| R4 IDENTITY | `r"\b(?:my name is|I am called|call me)\s+(?P<obj>[A-Z][\w-]+)"` → predicate `is_named`；`r"\bI(?:'m| am)\s+(?:a|an)\s+(?P<obj>[^,.;]+)"` → `is`；中文 `我叫(?P<obj>\S+)` → `is_named` | `fact` / `single_fact`；objects `[obj]`；confidence `0.85` |
| R5 LIVE/WORK | `r"\bI\s+(?P<verb>live|work|study)\s+(?:in|at|for)\s+(?P<obj>[^,.;]+)"`；中文 `我(住在|在.+?工作|在.+?上学)`；搬迁孪生（仅在居住/工作模式未命中时才尝试）`MOVED_RE = r"\bI\s+(?:just\s+|recently\s+)?(?P<verb>moved|relocated)\s+(?:back\s+)?to\s+(?P<obj>[^,.;]+)"`（re.I），中文 `MOVED_ZH_RE = r"我(?:现在|已经|刚)?(?P<verb>搬到|搬去|搬回)(?P<obj>[^，。；！？]+)"` | `fact` / `single_fact` / `lives_in|works_at|studies_at`；objects `[clean_object(obj)]`；confidence `0.85`；搬迁孪生 → predicate 为 `lives_in` 且 `relation=supersession`（其动词短语 `moved to|relocated to|搬到|搬去|搬回` 属于与 R1 共享的 `_SWITCH_VERBS` 词表），因此 §11.3 规则 6 会修订此前的居住地 |
| R6 HABIT | `r"\b(always|usually|every (?:morning|day|week|weekend|month)|each (?:day|week|month)|twice a|tend to)\b"` 命中且为第一人称；中文 `每天|每周|通常|总是|习惯` | `habit` / `habit` / `does`；objects 为 `[clean_object(句子去掉主语代词与线索词)]`；confidence `0.65` |
| R7 TEMPORAL | `extract_dates(sentence, base_date=Observation Date)` 得到的窗口中，存在 kind 不是 `relative_this` 的项 | `temporal_fact` / `single_fact`；`FUTURE_CUES` 命中时 predicate 为 `plans`，否则为 `did`；`date_observed` = `resolve_date(...)`；第一人称时 subject 为 `user`，否则取第一个 PROPER 实体，再否则 `user`；objects 为除主语外的实体文本；sentiment 由 `sentiment()` 给出；confidence `0.75` |
| R8 COMPARISON | `COMPARE_RE = r"(?P<a>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\s+(?:vs\.?|versus|or|and)\s+(?:the\s+)?(?P<b>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"` 命中，且句中含 `compar|decid|choos|between|torn|还是|比较`；对 `a`、`b` 施加 PROPER 过滤 | `fact` / `comparison` / `compares`；objects `[a, b]`；text 为 `f"User compared {a} and {b}; no exclusive final choice was stated."`；sentiment `neutral`；confidence `0.8` |
| R9 FACT（兜底） | 句中至少有 1 个 PROPER/FILE/IDENTIFIER 实体，或有第一人称主语，且 token 数 ≥4 | `fact` / `single_fact`；predicate 取主语跨度之后第一个属于 `VERB_LEXICON` 的 token，否则为 `states`；第一人称时 subject 为 `user`，否则取第一个 PROPER 实体，再否则 `user`；objects 为其余实体文本；confidence `0.6`；`strict=True` 时跳过本规则 |

每条 observation 的公共后处理：
- `event_idx` = 该消息的 idx；`base_date`：当 `## New Messages` 条目带有非 null 的 `timestamp` 时取 `date_of(msg['timestamp'])`，否则取 `## Observation Date`；`date_observed = base_date`，除非 R7 解析出了日期（R7 也是基于同一个 `base_date` 解析的）。
- `text`：R1 → `f"User {pred_phrase} {objects[0]}."`，其中 `pred_phrase ∈ {likes: 'likes', prefers: 'prefers', dislikes: 'dislikes'}`；R1 带 `alt` → `f"User prefers {a} over {b}."`；中文 R1 → `f"用户{喜欢|偏好|讨厌}{obj}。"`；R8 → 上表中的模板；R2/R3/R7/R9 → `rewrite_first_person(sentence)`（第三人称句逐字保留；确保末尾有 `.`；首字符大写）；R4/R5/R6 → `rewrite_first_person(sentence)`。
- `subject`：`is_first_person(sentence)` 为真时取 `user`（§9.2，用 `re.search`，所以 "Yesterday I visited the Berlin office." 的 subject 是 `user`），否则按各规则的说明取值；当 role 为 assistant 时，第一人称句的 `subject = 'assistant'`；`attributed_to = 'assistant'` 当且仅当 role 为 assistant。
- 每个条目的 `speaker = None`：rule 提供者从不点名人物（`_OBSERVATION_KEYS` 携带该键，使条目形状与 §6.3 块一致）。
- `topic_tags` = `topic_tags(sentence)` ∪（R1 加 `['preference']`，R2/R3 加 `['constraint']`），表命中项在前，然后是类型标签；总数 ≤ 4。
- `entity_mentions` = `[c.text for c in extract_entities(sentence)]`（包含 subject 实体）。
- `supersedes_hint_ids`：对 R1 产出的 observation，取 `## Existing Memories` 中文本以 `User likes|prefers|dislikes`（或 `用户喜欢|偏好|讨厌`）开头，且（与该 observation 文本至少共享 1 个 `content_tokens` token，或至少共享 1 个 `TOPIC_TABLE` 主题）的那些条目的 id；对其他所有规则，取 `content_tokens` 与该 observation 文本至少共享 2 个 token 的条目 id；都没有则为 `[]`。（S1："User likes black coffee." 与已有的 "User likes oat-milk lattes in the morning." 共享 token `like` 和主题 `coffee` → `["0"]`。）
- assistant 角色：只有匹配 `ASSISTANT_CUE_RE = r"\b(I recommend|I suggest|I've (?:scheduled|booked|created)|I have (?:scheduled|booked|created)|I will|I'll|I created|I scheduled)\b"` 或 `ACTION_RE`（助手在报告自己的动作）的句子，才产出 R9 fact，其 `subject='assistant'`、`attributed_to='assistant'`、confidence `0.6`，text 为 `rewrite_first_person(sentence)` 但把 `User` 换成 `Assistant`（例如 `chat_action_closure` → `"Assistant patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."`，predicate 为 `patched`，objects 为 `['retry.ts', 'billing_e2e']`）。
- 批内去重：按 `(type, predicate, tuple(objects), normalize(text))` 去重（`normalize` 同 §11.3 规则 1），保留首个——只有字面重复的陈述才会被合并（S2 中三条 Sarah 的 observation 都会保留）；id 从 `"0"` 开始顺序编号。

### §11.2 `ep_extraction`（仅限对话消息；结构化事件永远不会进入这个任务）

对 `## Events` 的每个条目：`acts = ACTION_RE.finditer(text)`；对每个动作命中（每个动作子句产出一条 EP，每条消息最多 3 条）：`a = f"{m['verb'].lower()} {m['target'].strip()}".strip()`（target 保留标识符中的点号，并在遇到连词或子句分隔符时停下，见 §9.2）；`y` 取从动作所在句起（含该句）第一个含 `OUTCOME_RE` 命中的句子中，包含该命中词的最短子句（用 `CLAUSE_SPLIT_RE` 切分后去首尾空白），没有则为 `""`；`r`：结果词属于 `NEGATIVE_OUTCOMES` 时为 `negative`，属于 `POSITIVE_OUTCOMES` 时为 `positive`，否则 `y == ""` 时为 `unknown`、其余为 `neutral`；`u` 取同一句 `CAUSE_RE` 的 `u`/`uzh` 分组（连词之后到下一个 `,`/`;`/句末之间的文本），没有则取上一句，再没有则取 `f"task: {消息前 60 个字符}"`；`b` = `' ; '.join(## Known Constraints 中 content_tokens 与消息 token 交集非空的那些约束文本)`；`kappa.constraints` 即上述约束文本；`kappa.visibility = 'private'`；`time_anchor` 取该事件的 `timestamp`；`entity_mentions` 为 `a`+`y`+`u` 中的实体；`topic_tags = topic_tags(a + ' ' + y + ' ' + u)`；`uses_ep_ids` 取 `## Recent EPs` 中 `a` 与本条 `a` 共享标识符 token 的那些 id；`confidence 0.7`；`event_idx` = 条目 idx。若 `not (u and a and (y or kappa.constraints))` 则丢弃这条 EP。

完整示例（fixture `chat_action_closure`，文本 `I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed.`）：`verb='patched'`，`target='retry.ts to add backoff'` → `a='patched retry.ts to add backoff'`；`CAUSE_RE` → `u='the webhook kept timing out'`；子句切分得到 `['I patched retry.ts to add backoff because the webhook kept timing out', 'billing_e2e failed']` → `y='billing_e2e failed'`，`r='negative'`；`entity_mentions=['retry.ts', 'billing_e2e']`；`topic_tags=['engineering', 'billing']`。这就是 golden 文件 `eps_chat_action_closure.json`。

### §11.3 `revision_decision`（规则表；首个命中者胜出；候选按给出的顺序处理）——由公开函数 `decide_revision(new, old)` 实现

用 `## New Memories` 中的每一项，逐个比对 `## Candidate Existing Memories`（一旦得到非 ADD 的判定就停止；否则结果为 ADD）：
1. `normalize(new.text) == normalize(old.text)`（转小写、压缩空白、剥掉终止标点）→ `NONE`。
2. `new.type != old.type` → `ADD`。
3. `new.type ∈ {temporal_fact, habit}` → `ADD`。
4. `new.subject != old.subject` → `ADD`。
5. predicate 相同且 `objects_overlap(new, old)`（存在一对归一化字符串相等、或 token Jaccard ≥ 0.5 的 object）→ `NONE`（信息重复）。
6. `new.relation == 'supersession'` 且 `family(new.predicate) == family(old.predicate)` 且 `new.date_observed >= old.date_observed` → `REVISE`（reason 为 `"supersession cue"`）。家族划分：`PREF = {likes, dislikes, prefers, avoids}`、`IDENTITY = {is_named, is, lives_in, works_at, studies_at}`、`CONSTRAINT = {must_not, must}`，其余情况家族就是 predicate 本身。
7. predicate 相同、predicate ∈ `EXCLUSIVE = {lives_in, works_at, studies_at, is_named, prefers}` 且 objects 不相交：`new.date_observed >= old.date_observed` 时 → `REVISE`（reason 为 `"exclusive attribute updated"`），否则 → `CONFLICT`（reason 为 `"older statement contradicts newer"`）。
8. 极性翻转（在重叠 objects 上出现 `{likes, dislikes}` 的对立）：`new.relation == 'supersession'` 时 → `REVISE`，否则 → `CONFLICT`。
9. constraint 对 constraint，objects 重叠且 predicate 在 `{must, must_not}` 之间不一致 → `CONFLICT`。
10. 其余情况 → `ADD`。
输出：每个 new id 一条决策，`target_id` 为命中的候选 id（ADD 时为 null），reason 为上面给出的字符串。

### §11.4 `edge_proposal`——由公开函数 `propose_edges(nodes, pairs, witnesses)` 实现

节点采用 §6.2 的形状（`u, b, a, y, r, subject, predicate, objects, ...` 始终存在；observation/event/skill 类节点的 `a` 为 atom 文本，`y`/`b`/`u` 为 `""`）。对 `## Candidate Pairs` 中每个 `[x, y]`，先保证 `x.time <= y.time`（否则交换）：满足 `|content_tokens(x.y) ∩ content_tokens(y.u)| ≥ 2`、或 `x.a` 的某个标识符出现在 `y.u` 中、或（`x.r == 'negative'` 且 `topics(x) ∩ topics(y) ≠ ∅`）时判为 `causal`；否则，若 `x.a`/`x.y` 的某个标识符出现在 `y.a` 或 `y.b` 中，判为 `use`；否则，若 `x.a` 与 `y.a` 共享同一个标识符目标、且 `x.r == 'negative'`、`y.r == 'positive'`，判为 `revision`；否则，若两者都是 observation 节点、`(subject, predicate)` 相同且 objects 不相交，判为 `conflict`；否则不产出边。`temporal` 与 `distill` 永远不由该规则提出（在别处计算）。`witness_refs = [y 的第一个 anchor ref]`（revision/conflict 时再加上 x 的第一个）；`label = f"{type}: {x.a[:40]} → {y.u[:40]}"`（`u` 为空时用 `y.a`）；`confidence = min(0.9, 0.6 + 0.1 * evidence_count)`，其中 evidence_count 为上述条件被满足的个数。

### §11.5 `coarsen`

把 `## Support Nodes` 按 `time` 排序；`P` 为 kind ∈ `{ep, interface_ep}` 的节点（L2 的输入是 EP 节点；L3 的输入是 InterfaceEP 节点，其 `u,b,a,y,r,constraints` 来自各自的 `Slots`）；`P` 为空 → `{"interface_ep": null}`（调用方返回 None）。`slots.u = P[0].u`；`slots.a = ' → '.join(p.a for p in P)`（最多 8 段，其后接 `…`）；`slots.y` 取最后一个非空的 `p.y`；`slots.r = 'negative' if any(p.r == 'negative') and P[-1].r != 'positive' else P[-1].r`；`slots.b = ' ; '.join(ordered_union(*[p.b.split(' ; ') for p in P if p.b], kappa.constraints))`（先切分再按首次出现去重——约束文本绝不重复）；`kappa.constraints` = `ordered_union(*[p.constraints for p in P])`，`kappa.visibility` 取各节点中最严格的一个（rank 最小者）；`slot_support`：`u → [P[0]]`，`y → [被选中的 y 所属节点]`，`r → [P[-1]]`（当 r 为 negative 时再加上各 negative 节点），`a → 全部 P`，`b`/`kappa → 贡献了约束的节点`（为空时取 `[P[0]]`）；不变式（invariants）：凡是在 ≥ 2 个节点的 constraints 中都出现的约束文本 → `{"kind": "constraint", "support_refs": 这些节点}`；每个 negative 节点 → `{"text": f"do not: {a}", "kind": "failed_path", "support_refs": [node]}`；残留项（residuals）：`r == 'unknown'` 且 `y` 为空的节点 → `missing_outcome`；`## Internal Edges` 中类型为 `conflict` 的边 → `unresolved_conflict`（support 为两个端点）；任何 `y` 中以 `?` 结尾的句子 → `open_question`。

### §11.6 `procedural_summary`

由消息生成 Markdown：objective 取第一条 user 消息（≤120 字符）；每条含 `ACTION_RE` 命中的 assistant 消息生成一个编号步骤（`**Agent Action**:` 为该动作句，`**Action Result**:` 为下一条 tool/user 消息文本，截至 ≤200 字符，没有则为 `n/a`）；`**Key Findings**:` 为 `OUTCOME_RE` 正向命中的句子；`**Errors & Challenges**:` 为负向命中；`**Current Context**:` 为最后一条消息，≤200 字符；`**Progress Status**:` 在最后一个结果为正向时取 `"completed"`，为负向时取 `"blocked"`，其余取 `"in progress"`。

### §11.8 `observation_extraction_compact` / `memory_extraction_compact`（§6.5）

`_task_observation_extraction_compact(sections)` = `_task_observation_extraction` 经 `_compact_observations` 投影到 `COMPACT_OBSERVATION_KEYS`：五个必需键总是输出（`speaker` 保持 `None`——普通聊天没有具名说话人；主语由 `statement_triple` 从 `text` 读回），`entities` = 完整观察的 `entity_mentions` 与 `supersedes_hint_ids` 均仅在非空时输出，`when` = `date_observed` 且仅当它与被见证消息的日期不同（`_message_dates`：`## New Messages` 的时间戳，否则 `## Observation Date`），`valid_to` 仅在设置时输出。`_task_memory_extraction_compact(sections)` = `_task_memory_extraction` 并对 `observations` 做同样投影；`eps` 与 `decisions` 原样返回——决策在投影之前由 `decide_revision` 基于完整的 R1–R9 观察（objects、predicate）判定。两个处理器都以任务名注册在 `_TASKS` 中；`tests/test_extraction_compact.py` 断言在每个聊天夹具上紧凑门面存下与完整门面相同的原子、历史行与图边。

`_task_memory_extraction_lines(sections) -> str`（§6.6）= `_task_memory_extraction` 渲染为 tab 分隔文本：每条观察一行 `<event_idx>⇥<T>⇥<text>`（`T` 取自 `configs.prompts.TYPE_LETTERS`，仅当 date_observed 与被见证消息的日期不同时追加 `⇥@date_observed`——`_message_dates`，与紧凑投影相同），`eps` 非空时才有 `EP` 分节（`<event_idx>⇥u=⇥a=⇥y=⇥r=`，匹配的约束追加 `⇥k=c1;c2`；约束内的 `;` 变为 `,`），只有非 ADD 决策才有 `DEC` 分节（`<id + 1>⇥<裁决>⇥<target_id>⇥<理由>`——R1–R9 的 id 是位置序号，故 1 起的 OBS 行号 = id + 1）；字段内的 tab 与换行变为空格（`_line_field`）；回复以换行结尾。决策在渲染之前基于完整的 R1–R9 观察判定。`generate_response` 原样返回该字符串（不做 JSON 序列化）。`tests/test_extraction_lines.py` 断言在每个聊天夹具上与完整门面逐行、逐表相同（§6.6 说话人规则的 `actor_id` 除外），并断言 EP、带日期陈述与 REVISE 的精确渲染。

### §11.7 `ScriptedLLM`（`hms_elite/llms/scripted.py`）

`ScriptedLLM(config)` 持有 `responses: deque`，元素类型为 `str | dict | Exception | Callable[[list[dict]], str | dict]`；`generate_response` 从左侧弹出（dict 则 `json.dumps`，Exception 则抛出，因而 `LLMError ... from e` 可被测试；callable 会被传入 `messages` 列表调用，其返回值按 str/dict 响应处理——测试 31 用它在调用时刻断言 `store.in_transaction is False`）；`loop=True` 时弹出的元素会重新入队；队列为空时，`raise_when_empty` 为真则抛 `LLMError(LLM_003, suggestion="queue more responses")`，否则返回 `""`；`calls: list[dict]` 记录 `{'task', 'messages', 'kwargs'}`；`push(*responses)` 用于追加。测试在结构化路径上断言 `len(llm.calls) == 0`。

---

## §12 摄取 Ingest（group C：`hms_elite/ingest/`）

包 docstring：`"""Ingest plane: SourceEvent → Anchor → routing → extraction → atoms, with lock discipline Txn A → unlocked → Txn B.\n摄取平面：事件→锚点→分流→抽取→原子；事务纪律 A→无锁→B。"""`

### §12.1 `ingest/events.py`

- `events_from_messages(messages: list[dict], *, scope: Scope, event_time: str, user_meta: dict, visibility: str, source_system: str = "mem0_api") -> list[SourceEvent]` — 每条归一化后的消息产出一个 `message` 事件 | 消息→事件。输入：归一化后的消息（§9.5）、scope、事件时间 | 消息、范围、时间。输出：按序排列的事件（`system` 角色的事件也在内，但被标记为不参与抽取）| 事件列表。副作用：无 | 无。说明：payload 为 `{'text': content, 'dropped_parts': n}`；`role`、`actor_id = msg.get('name')`、`raw_ref = default_raw_ref(ev, index)`；单条消息若自带 `timestamp` 键（dict 中存在时），覆盖 `event_time`。
- `events_from_dicts(items: list[SourceEvent | dict], *, scope_fallback: Scope | None, visibility_default: str) -> list[SourceEvent]` — 校验 `add_events` 的输入 | 校验结构化输入。说明：不含 scope 键的 dict 使用 `scope_fallback`（为空则 `VALIDATION_001`）；`object_type ∈ INTERNAL_OBJECT_TYPES` → `ValidationError(VALIDATION_007, details={'index', 'object_type'})`；payload 错误 → `VALIDATION_005`，并带 `details['index']`；`event_type` 的默认值规则：`test_log`、`sensor_trace`，以及 payload 中 `status` 或 `result` 非空的 `tool_call`，默认为 `tool_result`；其余默认为 `created`（显式给出的 `event_type` 保持不变）；结构化类型的 `role` 默认为 `tool`；结果按 `(timestamp, index)` 做稳定排序。
- `validate_payload` — 从 `hms_elite.models` 重导出（§4.3）；这里**不**重新实现 | 载荷归一化（来自 models）。
- `default_raw_ref(ev: SourceEvent, index: int) -> str` — `message → f"chat://{scope.key()}/{index}"`、`tool_call → f"tool://{tool}"`、`file_diff → f"file://{path}"`、`test_log → f"test://{suite}"`、`sensor_trace → f"sensor://{sensor}"`，其余 → `f"hms://{object_type}/{timestamp}"` | 默认引用。
- `anchor_events(store: StateStore, events: list[SourceEvent]) -> list[tuple[SourceEvent, Anchor, bool]]` — **Txn A**：对每个事件执行 `insert_source_event` + `get_or_create_anchor`；事件此前已存在时 `dedup=True` | 事务 A 落锚。副作用：写 source_events、anchors | 写事件与锚点。
- `system_event(store, *, scope: Scope, object_type: str, event_type: str, payload: dict, actor_id: str | None = None, visibility: str = "private") -> tuple[SourceEvent, Anchor]` — update/delete/feedback/checkpoint/rollback/scheduler 共用的前奏（object type 取自 `INTERNAL_OBJECT_TYPES`）| 系统事件前奏。副作用：在当前事务内写入一条事件 + 一个 anchor。

### §12.2 `ingest/router.py`

- `route_event(ev: SourceEvent) -> set[str]` — user/assistant 消息 → `{'observation'}`；`has_action_closure(text)` 为真或 `ev.user_meta.get('route') == 'state'` 时 → `{'observation', 'state'}`；tool_call/file_diff/test_log/sensor_trace → `{'structured'}`；system 角色消息与内部类型 → `set()` | 事件分流。副作用：无。

### §12.3 `ingest/extraction.py`（执行期间**不持有任何锁**）

- `neighbourhood(store, index, embedder, *, scope: Scope, texts: list[str], k: int) -> tuple[list[dict], list[SourceEvent]]` — `existing = [{'id': '0'.., 'text'}]`，来自 `index.search(embed(parse_messages), candidate_ids=store.valid_ids(scope)['atoms'], top_k=k)`（group A 的辅助函数——C 从不 import `hms_elite.mvmi`），外加 `last_events(scope, object_types=['message'], limit=k)` | 邻域上下文。副作用：只读。
- `extract_observations(llm, *, events: list[tuple[SourceEvent, Anchor]], existing: list[dict], last_events: list[SourceEvent], observation_date: str, current_date: str, custom_instructions: str | None, scope: Scope, id_map: dict[str, str], output: str = 'full', entity_extractor: EntityExtractorBase | None = None) -> list[Observation]` — 一次 `observation_extraction` 调用（`observation_date = date_of(events[-1][0].timestamp)`、`current_date = date_of(store.now())`，两者均由 pipeline 计算；`## New Messages` 中的每一项自带 `timestamp`，抽取器把它当作该条消息的 `base_date`，见 §11.1）；每一项 → `Observation(anchor_ids=[anchor of event_idx], ...)`。**词表不匹配只规整、不拒绝**（逐条守卫）：只有在 (a) 无法见证——`event_idx` 越界或不可读（数字串 `'0'` 与整值浮点 `0.0` 可读；只有一条新消息时缺失的 `event_idx` 指该消息）、(b) 没有 `text`、或 (c) 字段类型畸形时才跳过并 `logger.warning`：列表字段（`objects`、`topic_tags`、`entity_mentions`、`supersedes_hint_ids`）接受列表或单个字符串（→ 单元素列表，绝不是逐个字符；dict 项摊平为其文本；逗号分隔的标签串被拆开），文本字段（`subject`、`predicate`、`text`）必须是字符串——其它任何形状都同样跳过该条目，因此畸形条目既不会以原生 `TypeError` 逃出 `HMS.add()`，也不会持久化为损坏的原子。其它所有不匹配都规整为有据可查的值：`type` / `relation` / `sentiment` → 小写、空格与连字符 → `_`，依次尝试精确成员、别名表（`preference` → `preference_event`、`rule` → `constraint`、`update` → `supersession`、`happy` → `positive`……）、以整词出现在文本中的成员（`very positive` → `positive`；绝不按子串——`artifact` 不是 `fact`）、≥ 3 字符的前缀，否则取默认值 `fact` / `single_fact` / `unknown` 并记一条 DEBUG；**被否定**的值（`not positive`、`non-habit`、`not a constraint`）绝不折叠到它点名的成员，而是取该默认值并记一条 WARNING；`confidence` → [0, 1] 内的数原样，`'85%'` 与 (10, 100] 内的裸数字按百分比读，`high/medium/low` → 0.9/0.7/0.4，None/'' → 0.7，其它任何数（负数、(1, 10] 内、大于 100、NaN）→ 0.7 并记一条点名字段的 WARNING——绝不变成接近 0、被 prune 阶段当作 `decayed` 归档的值；`valid_from` / `valid_to` → 保留 ISO 或仅日期的时刻（仅日期 → UTC 零点），其它（`next week`、裸年份）→ `None`；早于 `atom_from_observation` 将赋予的下限（`valid_from`，否则 `temporal_fact` 取 date_observed 零点，否则取见证消息的时间戳）的 `valid_to` 被置为 `None` 并记警告，因此 `MemoryAtom.valid_to >= valid_from` 永远无法中止批次；`date_observed` → `YYYY-MM-DD`（`/` 与 `.` 分隔符归一），否则取 `observation_date`；`attributed_to` → 不分大小写的 `user|assistant`（`system/tool/bot/ai/model` 视为 assistant 侧，`human/me` 视为 user 侧），集体词（`group`、`others`、`everyone`……）→ 被见证消息的角色，其它任何字符串 → 消息的角色**并**把该字符串保留为 `speaker`；`speaker` → 既非侧别词也非集体词的非空字符串，否则 `None`；说话人（来自任一键）不在 `entity_mentions` 中时追加进去（不分大小写）；紧凑模式（§6.5）下可选的 `entities` 提示是唯一形状畸形也不跳过条目的列表字段——`_entities_hint` 跑 `_str_list`，遇到数字 / dict 时记警告并忽略，推导回退到 extractor(`text`) ∪ `speaker`；`supersedes_hint_ids` 经 `id_map` 映射（'0'.. → atom_id）。第二道防线：原子仍违反 `MemoryAtom` 不变量的观察由 pipeline 记警告丢弃——并以 `{'kind': 'observation', 'text', 'reason': 'invalid_atom'}` 列入 `IngestResult.dropped`——而不是中止批次 | 观察抽取。
- `extract_eps(llm, *, events, known_constraints: list[MemoryAtom], recent_eps: list[EP], observation_date, scope) -> list[EP]` — 一次 `ep_extraction` 调用；`uses_ep_ids` 按 `e0.. → ep_id` 映射；`kappa.evidence = [anchor]`；anchors 为 `[anchor of event_idx]` ；对 `kappa.constraints`、`entity_mentions`、`topic_tags`、`uses_ep_ids`（字符串算一个元素）以及 `u/b/a/y` 槽位（不是字符串 → 跳过该条目并 `logger.warning`，因此字符列表永远无法满足 `Slots._closure`）采用与 `extract_observations` 相同的规整；词表遵循观察的纪律：`r` 经 Reward 别名表（`success/pass/ok/fixed` → positive，`fail/error/broken` → negative，`partial/mixed` → neutral，`none/n/a` → unknown），否定 → `unknown` 并记 WARNING，其它 → `unknown` 并记 DEBUG；`confidence` 与观察完全相同；不可读的 `time_anchor` → 事件的时间戳；`kappa` 给列表视为约束列表、给字符串视为一条约束、其它非对象 → `{}`；词表外的 `kappa.visibility` → 事件的可见性；`uses_ep_ids` 的裸索引 `0` → `e0`；`eps` 根是单个对象时按单元素列表读取（一条警告）| EP 抽取。
- `decide_revisions(store, llm, *, new: list[tuple[Observation, MemoryAtom]], scope: Scope, current_date: str, revision_mode: str) -> list[tuple[Decision, str | None, str]]` — 对每个新的「观察-原子」对：候选集 = `store.revision_candidates(...)`（按时间取前 5）；候选中存在 `hash` 完全相同者 → 直接 `NONE`，不调用 LLM；无候选 → `ADD`；`revision_mode == 'off'` → `ADD`；`'deferred'` → `ADD`（由 consolidate 的 `detect_conflicts` 后续判定）；否则合并成**一次** `revision_decision` 调用；裁决经规整——去空白、大写、空格/连字符 → `_`，单元素列表取其元素，None/'' → `ADD`，同义词折叠（`UPDATE/UPDATED/REPLACE/SUPERSEDE(S)/MERGE/REVISED/REVISION` → REVISE，`CONTRADICT(S)/CONFLICTING/CONFLICTS` → CONFLICT，`KEEP/NEW/INSERT/ADDED/CREATE/ADDITIVE` → ADD，`DUPLICATE/DUP/SAME` → NONE——这是唯一会丢弃新观察的折叠，因此记一条 WARNING）；`SKIP/NOOP/IGNORE` 与其它任何词都是**未知**裁决 → `ADD` 并告警（观察被存储）；`target_id` 反向映射（单元素列表取其元素；候选 atom_id 直接通过）；目标未知 → `ADD` 并告警——在单次调用的 `extract_all` 路径上，观察自身解析出的 `supersedes_hint_ids` 恰好指向一个候选时先用它替代；该路径上 `decisions` 根是单个对象、或以观察 id 为键的 dict 时按列表读取，决策 id 与观察 id 先精确连接，再按数字连接（`obs_0` ↔ `0`），数量一致时再按位置配对（一条警告）；缺失的决策 → `ADD` 并告警 | 修订决策。副作用：只读。
- `procedural_summary(llm, *, messages: list[dict], prompt: str | None) -> str` — `generate_text('procedural_summary', build_procedural_messages(...))`；结果为空 → `ValueError("Procedural memory generation failed")` | 过程摘要。
- `hms_elite/ingest/derive.py`（§6.5）— `expand_compact_item(item, *, obs_type, speaker, role, message_date, extractor) -> dict` 构造紧凑观察的完整条目形状（`llms.rule.statement_triple` 给出 subject / predicate / objects / 关系线索，`subject_for` 负责说话人 / 角色回退，`relation_of`、`is_full_datetime`、`COMPACT_CONFIDENCE`；没有自己的线索表）；`expand_lines_ep(item, *, extractor) -> dict`（§6.6）在 `extract_all(output='lines')` 把解析出的行式 EP 条目交给 `_ep_from_item` 之前，从 `'<a> <y> <u>'` 补齐其 `entity_mentions` / `topic_tags`、从 k= 约束补齐空的 `b`，与规则 LLM 补齐自己的 EP 相同；`extract_observations(..., output='compact', entity_extractor=...)` 与 `extract_all(..., output='compact', entity_extractor=...)` 在 `observation_extraction_compact` / `memory_extraction_compact` 任务下运行 `_compact_observation_from_item`（先见证，再扩展，最后走完整契约的规整）；`output` ∉ `EXTRACTION_OUTPUTS = ('full', 'compact', 'lines')` → `ValidationError(VALIDATION_013)`；`extract_observations`（分段路径）上 `output = 'lines'` → 同一错误码且 `details.supported = ['single']` | 紧凑契约的本地推导。
- `hms_elite/ingest/lines.py`（§6.6）— `parse_lines(text, *, what) -> {'observations', 'eps', 'decisions'}`：把 tab 分隔的行式回复容错解析为紧凑形状条目（id = 模型所写的 1 起 OBS 行号：每个带 tab 的 OBS 行都占用一个行号，无论是否存下内容，无 tab 的行是不占行号的续行；没有开启任何分节的回复记一条含 `NO_SECTION_MARK` 的警告）；`extract_all(..., output='lines')` 执行 `llm.generate_lines` + `parse_lines`，随后 `_compact_observation_from_item(..., speaker_from_subject=True)`（开头的专有名成为 `speaker`）、EP 先经 `derive.expand_lines_ep` 再经 `_ep_from_item`，并按行号连接 DEC 行（`_parse_decision`；没有行 → ADD，不出声）| 行式契约的解析。

### §12.4 `ingest/closure.py` — 结构化闭包构建器（**不调用 LLM**）

- `event_atom_text(ev: SourceEvent) -> str` — `file_diff → f"file_diff {path}: {summary or first non-empty patch line}"`；`test_log → f"test_log {suite}: {status} ({passed} passed, {failed} failed)"`，有失败时再追加 `f": {', '.join(names[:3])}"`（计数为 None 时记作 `?`）；`tool_call → f"tool_call {tool}({args_summary}) → {status or 'unknown'}"`，其中 `args_summary = ', '.join(f"{k}={str(v)[:40]}" for k, v in sorted(args.items()))`；`sensor_trace → f"sensor_trace {sensor}: {summary or f'{len(samples)} samples'}"` | 事件原子文本。
- `action_text(ev) -> str` — `file_diff → f"patch {path}: {summary or first patch line}"`；`tool_call → f"{tool}({args_summary})"` | 动作文本。
- `outcome_text(ev) -> tuple[str, str]` — `test_log → (f"{suite} {status}" + (f": {', '.join(names)}" if names else ""), 'negative' if status == 'failed' else 'positive')`；`tool_call(status='error') → (result or 'error', 'negative')` | 结果文本与奖励。
- `build_ep_from_closure(*, action: tuple[SourceEvent, Anchor], outcome: tuple[SourceEvent, Anchor], background: list[MemoryAtom], previous: EP | None, scope: Scope, extractor: EntityExtractorBase | None = None) -> EP` — `u = action.payload.intent or (previous.slots.y if previous else None) or f"modify {path}" / f"call {tool}"`；`constraints = ordered_union([c.text for c in background], action.payload['constraints'])`（背景原子在前，随后是动作 payload 自带的 `constraints`）；`b = ' ; '.join(constraints)`；`a = action_text(action)`；`y, r = outcome_text(outcome)`（自闭合的未决 `tool_call` 取 `('', 'unknown')`）；`kappa = Kappa(constraints=constraints, visibility=min-rank(action.visibility, outcome.visibility), evidence=[outcome_anchor, action_anchor])`（因此 payload 里带 constraints 的自闭合 tool_call 总能通过 `Slots` 的闭包校验器）；`anchor_ids = [outcome_anchor, action_anchor]`；`time_anchor = outcome.timestamp`；`entity_mentions` 取 `entities(a + ' ' + y + ' ' + u)` 的文本；`topic_tags = topic_tags(u + ' ' + a + ' ' + y)`；`extracted_by = 'rule@closure-v1'`；`confidence 0.8` | 闭包构建 EP。
- `derived_constraint(ep: EP, *, scope) -> Observation` — `type=constraint, subject='project', predicate='must_not', objects=[ep.slots.a], relation=constraint, text=f"Do not re-apply '{ep.slots.a}': {ep.slots.y}", date_observed=date_of(ep.time_anchor), anchor_ids=ep.anchor_ids, entity_mentions=ep.entity_mentions, topic_tags=['constraint'] + ep.topic_tags, confidence=0.85, extracted_by='rule@closure-v1', sentiment='negative'` | 派生约束。
- `ClosureBatch`（dataclass）：`open_actions: dict[str | None, list[tuple[SourceEvent, Anchor]]]`、`eps: list[EP]`、`derived: list[Observation]`、`edges: list[GraphEdge-spec]`。
- `close_structured_events(store, *, anchored: list[tuple[SourceEvent, Anchor, bool]], scope: Scope, background: list[MemoryAtom], max_gap_hours: float, temporal_window_hours: float, extractor: EntityExtractorBase | None = None) -> ClosureBatch` — 算法（事件已排好序；`sensor_trace` 事件在此跳过——它们只产出事件原子）：对每个 `object_type == tool_call|file_diff` 的 `(ev, anc)`，压入 `open_actions[ev.scope.run_id]`；但 `status == 'error'` 的 `tool_call` 改为立即自闭合（`build_ep_from_closure(action=(ev,anc), outcome=(ev,anc), ...)`，`anchor_ids=[anc]`）；对 `test_log`：`action = open_actions[run].pop()`（若有），否则 `store.last_open_action(scope, run_id=run, before=ev.timestamp, max_gap_hours=max_gap_hours)`（§8：该 run 内、在间隔窗口内、尚未被任何 EP 引用的最新 tool_call/file_diff 事件——跨批次闭合），再否则为 `None` → 只产出事件原子（EP 必须有 action）；`previous` = 本批次中为该 run 构建的最后一个 EP，或 `store.last_ep(scope, run_id=run)`；构建 EP；记录边规格：对每条背景约束记 `('use', ep → c.atom_id, witness=[action_anchor], confidence 0.8)`；当 previous 存在且 `age(previous.time_anchor, ep.time_anchor) ≤ temporal_window_hours` 时记 `('temporal', previous → ep, witness=[outcome_anchor], confidence 0.5)`（调用方传入 `config.graph.temporal_window_hours`；`max_gap_hours` 则是 `config.ingest.closure_max_gap_hours`——两个不同的旋钮）；当 previous 存在且（`previous.slots.r == 'negative'` 或实体重叠 或 `u` 由 `previous.slots.y` 派生）时记 `('causal', previous → ep, witness=[action_anchor, previous.anchor_ids[0]], confidence 0.7)`；负向 EP → `derived_constraint` + `('distill', ep → derived_atom, witness=[outcome_anchor], confidence 0.9)`。批次结束时，仍处于未决状态的 `tool_call` **仅当** payload 中带 `constraints` 才自闭合为 `r='unknown'`、`y=''` 的 EP（否则保持未决，不产 EP）；未决的 `file_diff` 保持未决 | 结构化闭包。

### §12.5 `ingest/atoms.py` — 唯一的物化原语

- `atom_from_observation(obs: Observation, *, scope, time_anchor: str | None = None, actor_id: str | None = None) -> MemoryAtom` — 类型映射：`fact|preference_event|habit|constraint → atom_type=state, scale_level=state`；`temporal_fact → event/event`，且 `time_anchor = date_observed + 'T00:00:00.000000+00:00'`；其余 `time_anchor = anchor timestamp`（由外部传入），否则 `obs.valid_from`，否则 date_observed 零点；`valid_from = obs.valid_from or time_anchor`（因此没有 `valid_from` 的 `temporal_fact` 从 date_observed 零点开始）；`content` 见 §4.2；`subject/predicate/obs_type` 上提；`sentiment`、`topic_tags`、`role=extracted`、`source_kind=observation`、`source_id=obs_id`、`attributed_to`、`extracted_by`；`actor_id = actor_id`（见证事件的，即消息 `name`——属于被见证的载荷哈希，因此是权威身份）`else obs.speaker else None`（`_observation_actor`）| 观察→原子。
- `relation_atom_from_observation(obs, *, scope, time_anchor=None, actor_id=None) -> MemoryAtom` — `relation='comparison'` 观察的 §4.2 `relation` 原子；`actor_id` 规则相同（pipeline 传入观察原子的 `actor_id`）| 关系原子。
- `atom_from_ep(ep: EP, *, scope) -> MemoryAtom` — `atom_type=episode, scale_level=L1_micro, text=ep.slots.gloss(), content={'text', 'ep_id', 'slots'}, time_anchor=ep.time_anchor`，sentiment 由 r 决定（negative→negative，positive→positive，其余 neutral），`source_kind=ep, source_id=ep_id, source_ep_ids=[ep_id], topic_tags`，`entity_ids` 来自 `entity_mentions`，`confidence=ep.confidence, visibility=ep.visibility, extracted_by=ep.extracted_by` | EP→原子。
- `atom_from_event(ev: SourceEvent, anchor: Anchor, *, scope) -> MemoryAtom` — `atom_type=event, scale_level=event, role=tool, text=event_atom_text(ev), content={'text','event_id','object_type','payload'}, time_anchor=ev.timestamp, source_kind=event, source_id=event_id, actor_id, extracted_by='raw'` | 事件→原子。
- `atom_from_message(ev, anchor, *, scope) -> MemoryAtom` — `atom_type=message, scale_level=L1_micro, role=ev.role, actor_id, text=payload.text, content={'text','role','actor_id','dropped_parts'}, source_kind=message, extracted_by='raw'` | 消息→原子。
- `EntityKey = tuple[str, str, str, tuple[str, ...]]` = `(key, canonical, entity_type, aliases)` — `key = EntityCandidate.key`（存放在 `entities.normalized_text` 中的身份标识，例如 `retry.ts`），`canonical = EntityCandidate.text`，`aliases` 在 `candidate.normalized` 与 `key` 不同时取 `(candidate.normalized,)`（FILE：完整的小写路径），否则为 `()`。
- `link_atom_axes(atom: MemoryAtom, *, extra_mentions: Sequence[str] = (), extractor: EntityExtractorBase | None = None) -> tuple[list[EntityKey], list[str]]` — 实体键取 `extract(atom.text)` ∪ `extra_mentions`（每个 mention 都用 `extract` 重新分类；按 `key` 去重，先到者胜），其中 `extract = extractor.extract`，`extractor is None` 时为 `extract_entities`（§10.5），话题取 `ordered_union(atom.topic_tags, topic_tags(atom.text))` | 轴链接。说明：`insert_atom` 会把 `normalized=key, canonical=canonical, entity_type=entity_type, aliases=aliases` 传给 `get_or_create_entity`（§8）。
- `atom_from_entity(entity: dict, *, scope: Scope, anchor_ids: list[str], mentions: int) -> MemoryAtom` — §4.2 的 `entity` 行：`atom_type=entity, scale_level=entity, text=entity['canonical_text'], content={'text', 'entity_id', 'entity_type', 'aliases', 'mentions'}, subject=canonical_text, predicate='is_entity', role=extracted, source_kind=entity, source_id=entity_id, extracted_by='rule@entity-v1', confidence=0.7, time_anchor=None, topic_tags=[]`；调用方通过 `extra_mentions=[canonical_text]` 把它链接到自身对应的实体 | 实体→原子。
- `create_atom(store, index, embedder, atom: MemoryAtom, *, extra_mentions=(), vector: list[float] | None = None, history_event: str | None = "ADD", history_anchor_id: str | None = None, actor_id=None, role=None, history_details: dict | None = None) -> MemoryAtom` — **在 Txn B 内执行**（`history_event=None` → 不写 history 行；`revise_atom` 用的就是这种方式）：`entity_keys, topics = link_atom_axes(...)`；`store.insert_atom(atom, entity_keys=..., topics=...)`；`index.upsert(atom_id, vector or embedder.embed(text, 'add'), embedder=embedder.identity)`；`store.add_history(memory_id, None, text, history_event, created_at=atom.created_at, updated_at=atom.created_at, actor_id, role, anchor_id=history_anchor_id or atom.anchor_ids[0], atom_id)` | 原子创建原语。副作用：写 memory_atoms、atom_anchor、entities、atom_entity、atom_topic、atom_vectors、graph_nodes、history。说明：只要可能，embedding 由调用方在事务**之前**算好并以 `vector=` 传入；`EmbeddingError` 向上传播，事务回滚。
- `revise_atom(store, index, embedder, *, old: MemoryAtom, new: MemoryAtom, witness_anchor_ids: list[str], at: str, reason: str = "revised", created_by: str = "ingest", history_event: str = "UPDATE", label: str = "") -> MemoryAtom` — `at` 的语义：**摄取期**的 REVISE/CONFLICT（§12.6）传入 `at = max(new.valid_from, old.valid_from)`（即新证据的事件时间——S1 中被归档的行会得到 `valid_to == '2026-04-20T08:00:00.000000+00:00'`），而 `update()` 与 feedback 处理器传入 `at = store.now()`；`new.memory_id = old.memory_id; new.version = old.version + 1`；`store.archive_atom(old.atom_id, reason, at, valid_to=at)`（相关行的 `updated_at` 以及 history 行的 `updated_at` 均为 `at`；新原子的 `created_at` 一如既往为 `store.now()`）；`create_atom(new, history_event=None)`；`store.insert_edge(GraphEdge(revision, old.atom_id → new.atom_id, witness_anchor_ids, created_by, proposer='rule', confidence 0.9, label))`；`store.add_history(memory_id, old.text, new.text, history_event, created_at=old.created_at, updated_at=at, anchor_id=witness_anchor_ids[0], atom_id=new.atom_id, edge_id, successor_id=new.atom_id)` | 修订原语。副作用：归档 + 插入 + 边 + history。
- `conflict_atom(store, index, embedder, *, old: MemoryAtom, new: MemoryAtom, witness_anchor_ids, at, label) -> MemoryAtom` — `create_atom(new)`（新的 lineage）；`insert_edge(conflict, old → new, witnesses, confidence 0.6, created_by ingest)`；在旧 lineage 上写 `CONFLICT` history（`old_memory=old.text, new_memory=new.text, details={'conflicts_with': new.memory_id}`）| 冲突原语。

### §12.6 `ingest/pipeline.py`

```python
@dataclass
class IngestResult:
    rows: list[dict]                 # Mem0 rows + 'event' ('ADD'|'UPDATE'|'NONE') + 'revises'? + 'conflicts_with'?
    events: list[dict]               # [{'event_id','anchor_id','object_type','dedup'}]
    ep_ids: list[str]; edge_ids: list[str]; constraint_ids: list[str]; dropped: list[dict]
    downgraded: list[dict]           # 被 §12.6 第 6 步守卫改为 ADD 的裁决：[{'kind': 'verdict', 'verdict': 'NONE'|'REVISE'|'CONFLICT', 'target_id', 'similarity', 'text'}]

class Ingestor:
    def __init__(self, store, index, embedder, llm, config: HMSConfig, entity_extractor: EntityExtractorBase | None = None, extraction_mode: str = 'single', extraction_output: str | None = None) -> None   # None → config.ingest.extraction_output（无硬编码默认；门面传入配置值）
        # entity_extractor 为 None → EntityExtractorFactory.create(config.ingest.entity_extractor, config.ingest.entity_extractor_config)（§10.5）；门面注入其共享实例
    def ingest_messages(self, messages: list[dict], *, scope: Scope, user_meta: dict, infer: bool, memory_type: str | None, prompt: str | None,
                        event_time: str | None, valid_to: str | None) -> IngestResult
    def ingest_events(self, events: list[SourceEvent], *, infer: bool) -> IngestResult
```

`ingest_messages` 的步骤（规范顺序）：
1. `events = events_from_messages(...)`；**Txn A** `anchored = anchor_events(store, events)`。
2. `infer=False` → 对每条非 system 消息：`atom_from_message`（外加 `valid_to`），embedding 在无锁状态下计算，**Txn B** 内逐个 `create_atom`（history 记 `ADD`，带 `actor_id`、`role`）；返回行携带 `actor_id`、`role`。随后返回。
3. 过程记忆分支（`scope.agent_id and memory_type == 'procedural_memory'`）：无锁执行 `md = procedural_summary(...)`；**Txn B** 写入一个原子 `MemoryAtom(atom_type=episode, scale_level=L1_micro, source_kind=procedural, memory_type='procedural_memory', anchor_ids=[all message anchors], text=md, role=assistant)`；返回单行。
4. 分流：`obs_events = [(ev, anc) for role in user/assistant]`；`ep_events` = `route_event(ev)` 结果中含 `'state'` 的那些。
5. 无锁阶段：`observation_date = date_of(events[-1].timestamp)`、`current_date = date_of(store.now())`（prompt 内部由单条消息的 `timestamp` 覆盖，§11.1）；`existing, last = neighbourhood(...)`；`observations = extract_observations(...)`；存在 `ep_events` 时 `eps = extract_eps(...)`；构建原子（`atom_from_observation`、`atom_from_ep`）；Mem0 精确去重：丢弃 `hash` 属于 `existing` 有效头部 ∪ 本批已见集合的原子（被丢弃者产出 `event='NONE'` 的行，指向已存在的 memory_id，并在 Txn B 中补一条 `support` 的 `atom_anchor` 行）；`vectors = embed_batch(texts)`（逐条回退；全部失败 → `EmbeddingError`）；`decisions = decide_revisions(...)`。
6. **Txn B**，按序执行：`insert_observation` / `insert_ep`（原子创建后回填 `atom_id` 反向指针）；先过裁决守卫再逐条应用决策（`ingest.verdict_min_similarity`，§6.3：被 `supersedes_hint_ids` 点名目标或主语 + 谓词相同佐证的 REVISE 无条件通过；否则两原子主语不同的裁决改为 ADD；同主语的 CONFLICT 通过；对有效目标的去头部 `verdict_similarity` 低于门槛、或只共享一个词元且未整体包含较短陈述的 NONE / 未被佐证的 REVISE 改为 ADD——记 DEBUG 日志并列入 `IngestResult.downgraded`；0.0 关闭）——`ADD` → `create_atom`；`REVISE` → 重新确认 `store.get_atom(target).archived_reason is None`（否则降级为 ADD），`revise_atom(..., at=max(new.valid_from, old.valid_from))`，以新原子的 anchors 作为 witness（事件时间的 `valid_to`，§12.5）；`CONFLICT` → `conflict_atom(..., at=new.valid_from)`；`NONE` → `add_atom_anchor(target, anchor, 'support')`（该 anchor 已是主 witness 时为空操作——同时间戳的重复 add），并 `bump_mentions(target)`；当 `ingest.emit_relation_atoms` 开启时，`relation='comparison'` 的观察额外创建其 `relation` 原子（§4.2）；内联 EP 边——同一 run 内相邻 EP 节点之间、且落在 `graph.temporal_window_hours` 内的 `temporal` 边（witness = 较晚 EP 的主 anchor，proposer 为 `timestamp`），以及针对已校验 `uses_ep_ids` 的 `use` 边（witness = 引用方 EP 的主 anchor）；对每个新 EP 执行 `store.upsert_transition(Transition.from_ep(ep, scope=scope, permission_scope=<primary anchor's permission_scope>))`（§4，group A；行结构见 §16.4）；`valid_to` 提示应用到每个新建原子上。
7. 行按抽取顺序返回：`{**atom.to_public_dict(), 'event': 'ADD'|'UPDATE'|'NONE', 'revises'?: old_atom_id, 'conflicts_with'?: [memory_id]}`（§4.1，group A）。`infer=True` 观察行的 `actor_id` = 消息给出的 `name`，否则 LLM 点名的 `speaker`（§12.5），否则缺省——这是有意偏离 Mem0：Mem0 只在 `infer=False` 原始路径上附加 `actor_id`（MEM0_MAP 第 8a 步）；`role` 仍仅限 `message` 原子。

`ingest_events` 的步骤：**Txn A** 对全部事件落锚（已排序）；按 object_type 拆分：`message` 事件 → 按 scope 分批走上面的步骤 4–7（LLM 路径，除非 `infer=False`）；结构化事件（`tool_call`、`file_diff`、`test_log`、`sensor_trace`）→ 无锁阶段：`background` = scope 内有效的约束原子（`obs_type='constraint'`，按 `time_anchor` 排序），`batch = close_structured_events(store, anchored=..., scope=..., background=..., max_gap_hours=config.ingest.closure_max_gap_hours, temporal_window_hours=config.graph.temporal_window_hours)`，构建原子（对**每一个**结构化事件执行 `atom_from_event`，`sensor_trace` 也不例外；每个 EP 执行 `atom_from_ep`；每条派生约束执行 `atom_from_observation`——与 scope 内的有效原子做 hash 去重：已存在则复用 id，`event='NONE'`），并计算 embedding；**Txn B**：事件原子 → `create_atom`；`insert_ep` + EP 原子；派生约束的观察及其原子；`batch.edges` 中的边经 `store.insert_edge` 写入（`created_by='ingest'`、`proposer='closure'`）；每个 EP 执行 `upsert_transition(Transition.from_ep(ep, scope=..., permission_scope=...))`。结构化路径上零 LLM 调用（由 `hms_scripted.llm.calls == []` 断言）。

---

## §13 MVMI — State Index 状态索引（group D：`hms_elite/mvmi/`、`hms_elite/utils/scoring.py`）

包 docstring：`"""State Index plane: filter AST, query planner, axis execution and score fusion — the hot path; never touches graph tables.\n状态索引平面：过滤 AST、查询规划、轴执行与分数融合——热路径；从不触碰图表。"""`

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

（块内英文注释对照：`FilterExpr` 为基类，其子类都是 frozen 的 pydantic 模型；`Ne`/`Gt`/`Gte`/`Lt`/`Lte` 与 `Eq` 结构相同。）

- `parse_filters(filters: dict | None, *, require_scope: bool = True, context: str = "search", force_meta: bool = False) -> tuple[Scope, FilterExpr | None]` — Mem0 语法：scope 键 → `Scope`（做 trim；`require_scope` 为真且为空 → `VALIDATION_001`）；**顶层 `AND` 上提**：Mem0 的 `_process_metadata_filters` 会在读取 scope 之前展平 `AND`（MEM0_MAP §2.2 第 5 步），因此写在顶层 `AND` 链中的**标量** scope 键——含嵌套的 `AND`，递归上提——会被提升到顶层并固定 `Scope`，与扁平写法完全等价（以首次出现者为准；顶层已存在该键则不上提，`AND` 中的那次仍作为列过滤叶节点；因上提而清空的子句会被丢弃，`AND` 列表被清空则删除该键）；`OR`/`NOT` 分支永不上提（那里的 scope 键不是必然约束 → 仍抛 `VALIDATION_001`）；列表 / 运算符字典 / `'*'` 形式的 scope 键在任何位置都不是固定值，仍作为列过滤叶节点；调用方的过滤字典绝不被修改（写时复制）；其他键：标量 → `Eq`，列表 → `In`，`'*'` → `Exists`，dict → 每个运算符一个节点（彼此 `AND`）；**visibility 格（lattice）门控**：标量形式的 `visibility: a`（受众）降为 `In('visibility', [v for v in VISIBILITY_RANK if rank(v) >= rank(a)])`（因此 `team` 允许 `team|event|public`；dict/list 形式按字面处理）；`force_meta=True` 强制把每个叶子放进 `meta.<key>` 命名空间（供 `AxisQuery.meta_filters` 使用）；`AND`/`OR`/`NOT` 键 → 对其列表项构造 `And/Or/Not`（每项递归解析；`NOT` 的列表只含一项）；字段名解析顺序：`AXIS_COLUMNS` → 列；`SET_AXES` → 集合轴（`entity` 的值经 `normalize_entity` 归一化；FILE 路径以 basename 作键）；其余 → `meta.<key>`；显式写出的 `meta.<key>` 永远指 user_meta；未知运算符 → `ValidationError(VALIDATION_012, ValueError)`，消息为 `"Unsupported operator 'X'. Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"` | 解析过滤器。副作用：无。说明：时间类型的字面量走 `to_utc_iso`（`lte`/`lt` 搭配纯日期字面量 → 当天末尾）；时间列上 `In` 的列表值按同样规则归一化。
- `lower_to_sql(expr: FilterExpr | None, *, capabilities: dict, scope: Scope) -> tuple[str, list[Any], set[str]]` — 返回 `(where_sql, params, joins)`；集合轴分别降为 `EXISTS (SELECT 1 FROM atom_entity ae JOIN entities e ON e.entity_id = ae.entity_id WHERE ae.atom_id = memory_atoms.atom_id AND e.normalized_text IN (?, ...) AND COALESCE(e.user_id,'') = COALESCE(?, '') AND COALESCE(e.agent_id,'') = COALESCE(?, ''))`（entity）、`EXISTS (SELECT 1 FROM atom_topic ...)`（topic）、`EXISTS (SELECT 1 FROM atom_anchor ...)`（anchor）；`meta.<key>` → `json_extract(user_meta_json, '$.<key>')`（键做转义；需要 `capabilities['json1']`）；`Contains` → `LIKE '%v%' ESCAPE '\'`，`IContains` → `LOWER(col) LIKE LOWER(...)`；`Exists` → `col IS NOT NULL`（meta 情形为 `json_type(...) IS NOT NULL`）| 降为 SQL。
- `evaluate(expr: FilterExpr | None, atom: MemoryAtom, *, entity_keys: set[str], anchor_ids: set[str]) -> bool` — 语义完全一致的 Python 版孪生实现（用于 `json1` 缺失时的回退，以及测试中的一致性校验）| Python 求值。
- `plan_filters(expr, *, capabilities) -> tuple[FilterExpr | None, FilterExpr | None]` — 返回 `(sql_part, python_part)`：`json1` 可用时 `python_part` 为 None；否则每个 `meta.*` 叶子（以及任何包含它的 `Or`/`Not`）移入 `python_part`，在 SQL 门控之后再求值 | 拆分计划。
- `has_advanced_operators(filters: dict) -> bool` — Mem0 移植（search 不使用；保留用于对齐测试）| 是否高级运算符。

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

块内英文注释对照：
- `route`：Route 枚举值。
- `time_windows`：`(kind, gte, lte)`，来自 `extract_dates(query, base_date=date_of(as_of))`。
- `content_tokens`：`content_tokens(query)`，其中标识符保持完整不拆。
- `hard_time_gate`：取第一个时间窗。
- `hard_entity_gate`：归一化后的键，仅当 `route == 'entity'` 时存在。
- `hard_type_gate`：`('obs_type', ['preference_event', 'habit'])`，仅当 `route == 'preference'` 时存在。
- `scale_default`：recall → `['state','event','L1_micro','L2_meso','L3_macro','entity','relation']`（None 表示不加约束）；preference → `['state']`；temporal → `['event','L1_micro','state','L3_macro']`；entity → `['state','event','L1_micro','L2_meso','L3_macro','entity','relation']`；work_state/detective → None。

- `plan_query(query: str, *, as_of: str, mode: str | None = None) -> QueryPlan` — 路由优先级：`WHY_RE → detective`；`HANDOFF_RE → work_state`；`time_windows or TEMPORAL_QUERY_RE → temporal`；`ENTITY_QUERY_RE and entity_hints → entity`；`PREF_QUERY_RE → preference`（HMS §4.1：偏好召回是一次轴选择——`hard_type_gate = ('obs_type', ['preference_event', 'habit'])`，`scale_default = ['state']`）；其余 → `recall`；当查询没有更强线索时，`mode == 'deep'` 强制走 `work_state` | 查询规划。副作用：无。说明：查询中的月份词只有在满足 §9.2 的 `month_en` 规则时才算日期（"may change" 里的 may 不是五月）；`"what coffee does the user like"` → 路由为 `preference`。

### §13.3 `mvmi/query.py`

- `execute_axis_query(store, index, embedder, aq: AxisQuery, *, scoring: ScoringConfig, capabilities: dict, threshold: float = 0.0, explain: bool = False, retrieval: RetrievalConfig | None = None, retrieval_stats: dict | None = None) -> AxisResult` — 规范算法（HMS §3.5 的语义是被真正执行的，而不只是声明）；`retrieval` 约束召回通道（None → `RetrievalConfig()`），`retrieval_stats`（给定时）接收胜出阶梯级别的 `Pool.stats()`：
  1. `fscope, expr = parse_filters(aq.filters, require_scope=False)`；`scope = aq.scope or fscope`（显式的 `aq.scope` 优先；两者都为空 → `VALIDATION_001`）；`_, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context='meta', force_meta=True)`（每个叶子都在 `meta.<key>` 命名空间内）；`sql_expr, py_expr = plan_filters(And([e for e in (expr, meta_expr) if e]))`。
  2. 由 `aq.plan` 生成计划门控（由 `search`/`compile` 设置，`get_all` 传 `{}`）：`hard_time_gate` → `time_anchor BETWEEN ? AND ?`；`hard_entity_gate` → entity 的 `In`；`hard_type_gate` → `obs_type IN (?...)`；三者一并追加到 SQL 的 WHERE。
  3. 排序查询（有 `aq.text`）的候选来自**有界多通道召回入池，再在池内全量打分**（RETRIEVAL_DIAGNOSIS §3–§5）；不再有最新优先截断（`ScoringConfig.max_candidates` 是已弃用的别名，不截断任何东西）。记 G 为该级别的门控集合（scope ∧ 有效性 ∧ `where_sql` ∧ 规划门控 ∧ 尺度 ∧ `atom_types`），`ks = retrieval.k_for(aq.limit)` = `{'semantic': max(4·top_k, 60), 'keyword' | 'entity' | 'recency': max(2·top_k, 30), 'pool': pool_max_factor · semantic}`：
     - `recall_pool(store, index, embedder, plan, scope, *, as_of, include_expired, cfg, top_k, text, query_vector, where_sql, params, joins, scale_constraint, atom_types, gate_count=None, index_filter=None, gate_entity_keys=None, index_hint=None) -> Pool` — **G 的枚举**：无 joins 的硬实体门经 `store.entity_candidates(scope, 门控键, limit=None, <门控>)`（由 atom_entity 驱动，`time_anchor DESC, atom_id ASC`）**一次**枚举 G，它既是探测、也是语义通道的候选列表，且当规划的实体线索等于门控键时就是实体通道本身；其它级别都用一次**不排序**的 `store.axis_ids(..., order=None, limit=pool + 1)` 探测（索引序、提前停止、不排序）。**整体入池**：当 `cfg.hard_gate_pool` 且 `|G| <= ks['pool']` 时整个 G 入池（`channel_sizes == {'gate': |G|}`、`recalled_by == ['gate']`、`Pool.whole`；水合恢复轴序）。否则取由索引支撑、各自确定性排序、且都在 G **之内**搜索的通道的轮转并集（顺序 `semantic, keyword, entity, recency`，去重，截到 `ks['pool']`；级别门控——`where_sql, params, joins, scale_constraint, atom_types, index_hint`——传入每个 store 通道）：**semantic** = 原生下推时 `index.search(vector, filter=IndexFilter(scope, as_of, include_expired, scale_in, obs_type_in, time_from, time_to, atom_type_in), top_k=K_sem)`（`index.supports_filter`——sqlite 与 milvus 提供者；仅当该级别没有过滤 WHERE / 连接 / 实体门），否则 `index.search(vector, candidate_ids=G 的 id, top_k=K_sem)`（只取 id，不水合）；通道的提供者分数保存在 `Pool.semantic`；**keyword** = `store.keyword_candidates(scope, 规划的 content_tokens + identifier_tokens, limit=K_kw, <门控>)`（两个 FTS5 阶段——全部词元、再任一词元——或有界的 `LIKE` 退路；标识符 ×2 权重仍在池内的 `keyword_score` 里）；**entity** = 线索键等于门控键时取已枚举 G 的前 `K_ent` 个，否则 `store.entity_candidates(scope, 规划的实体键, limit=K_ent, <门控>)`；**recency** = `store.axis_ids(..., order=AXIS_ORDER, limit=K_rec, <门控>)`（G 按轴序的前 `K_rec` 个 id），只在路由为 `temporal` 或规划带硬时间门、`cfg.hard_gate_pool` 为 False（使 `<= K_rec` 行的门控集合仍整体到达池）、或其它通道合计召回不足 `K_rec` 个 id 时作为兜底运行。`index_hint` 在硬时间门下为 `schema.TIME_INDEX`，在 `obs_type` 类型门下为 `OBS_TYPE_INDEX`，只有尺度白名单门控该级别时为 `SCALE_INDEX`（有 joins 或实体门时为空）：没有统计时 SQLite 会让这些查询全走 scope 索引并走遍整个 scope，preference 阶梯的三个空级别就各付了一次 scope 扫描。`Pool(ids, recalled_by, channel_sizes, gate_count, whole, semantic)`；`Pool.stats()` = `{'pool_size', 'channel_sizes', 'channels_used', 'gate_count', 'whole'}`（`gate_count` 在 G 被枚举时为 |G|，原生下推时为 -1——带过滤的计数在 milvus-lite 上是全量扫描，而探测只证明了 `|G| > pool_max`）。
     - 水合：池按每批 500 个 id 用 `store.axis_select(scope, where_sql=<级别门控>, atom_ids=<批>, ...)` 读取（VALUES-CTE CROSS JOIN：每个 id 一次主键查找；这里重新施加的门控就是池 ∩ G 不变量）——若 `py_expr` 存在再用 `evaluate` 过一遍。**Python 侧过滤**：当 `py_expr` 不为 None（无 JSON1 的 SQLite 上的元数据过滤）时**不**运行通道——只按 SQL 门控约束通道可能召回 K 个 id 而没有一个通过谓词——该级别用 `_select(limit=None)` 枚举（safety_limit，越过时告警）、在 Python 中过滤、整体打分（`channel_sizes == {'gate': n}`、`recalled_by == ['gate']`、`whole=True`），并记一条 `logger.warning`（"a Python-side filter (no JSON1) disables bounded recall"）；`tests/test_retrieval.py::test_python_filter_scores_the_whole_gate` 断言 `json1=False` 下与穷举路径逐位相同。池内语义分对通道命中复用 `Pool.semantic`，对其余池内 id 调用 `semantic_scores(index, qv, <通道未返回的池内 id>)`（逐位相同：两者都是提供者的 `dot` 分数；在 Milvus 上每次查询少一次 RPC）。
     - 安全：`retrieval.safety_limit`（默认 50000）绝不截断池；越过它时每个条件各记一条 `logger.warning`——|G| 已知（`gate_count >= 0`）时为 `"gated set of N atoms exceeds …"`，池为 `"pool of N atoms exceeds …"`（原生下推下 `gate_count == -1`，只有后者可达）。无法在 SQL 里有界化的仅轴读取（按值的 `latest`、`summary` 的计数、任何 `py_expr` 后过滤）至多读 `safety_limit` 行并在触顶时告警。
     - **等价性定理**（`tests/test_retrieval.py::test_equivalence_theorem`）：当 `|G| <= min(K_kw, K_ent, K_rec)` 或（`hard_gate_pool` 且 `|G| <= pool`）时池等于 G，因此每个命中的 `(atom_id, score)` 与 `score_details` 的前 14 个键都与穷举路径（每个门控候选都打分）逐位相同：池内语义分是该原子的提供者 `dot` 分数（来自 `Pool.semantic` 或 `semantic_scores(index, qv, rest)` = `index.search(candidate_ids=rest, top_k=len(rest))`），其余四路信号只依赖水合后的原子与规划，`fuse` / 阈值 / 排序键不变。现有测试套件的每份语料都有 `|G| < 60`，因此黄金场景不变。`|G| > K` 时池是 G 的子集（有意的、有界的剪枝损失；N=500 的重叠率由 `test_pool_recall_overlap_at_n500` 打印）。**漏检条件**：融合分是全局最大的原子恰好在以下情况下不在池中——它在语义通道里排在 `K_sem` 之后，且在其它每个运行了的通道里都排在 `K_kw` / `K_ent` / `K_rec` 之后（bm25 平局按 `atom_id` 而非融合分打破）；调用方从 `score_details` / `retrieval` 识别通道模式：`recalled_by == ['gate']`（`'gate' in channel_sizes`、`retrieval['whole']`）表示整个门控集合都打了分，其它都表示有界召回。
     - 仅轴查询（`aq.text is None`：`list` / `latest` / `summary` / `count`）从不构造池：`store.axis_select(..., limit=aq.limit)` 按轴序（普通 `latest` 用 `limit=1`；按值的 `latest` 与 `py_expr` 下不限量但受安全上限约束），`summary` 用一条 `COUNT(*)` 数 G。
  4. 回退阶梯，从最严格一级开始，第一个非空的级别胜出并记录为 `primary_axis`（每一级都走第 3 步的同一套有界召回；池经 `py_expr` 水合后为空的级别视同空级别跳过）：（计划门控，路由默认尺度）→ `'time' | 'entity' | 'type' | 'semantic'`（preference 路由取 `'type'`）；（计划门控，无默认尺度）→ `'scale_fallback'`；（无计划门控，默认尺度）→ `'fallback'`；（无计划门控，无默认尺度）→ `'fallback'`。只有路由的尺度*默认值*可放宽：显式的 `aq.scale_constraint` 从不放弃；`count` 走同一阶梯，每级一条 `COUNT(*)`。
  5. 聚合与投影（`aggregation` 为 None 时默认 `'list'`）：
     - `count` → `count = SELECT COUNT(*)`，在门控后的 WHERE 上执行（一条 SQL；不加载任何行），`hits = []`，`projection = None`。
     - 仅轴（`aq.text is None`）：`ordered` = 按轴序排列的候选（`time_anchor DESC, created_at DESC, atom_id ASC`），`score = 0.0`；`list` → `ordered[:limit]`；`latest` → 当 `project_axis == 'value'` 时取每个 `(subject, predicate)` 分组内最新的一条（分组顺序按首次出现），否则取全局最新的单个原子；`summary` → `ordered[:limit]`，且 `summary = f"{len(candidates)} atoms; latest: {ordered[0].text}"`（为空时是 `"0 atoms"`）；`topk` → `VALIDATION_013`（由模型校验器抛出：必须有 `text`）。
     - 排序打分（有 `aq.text`）：在第 3 步的**池**上 `signals = compute_signals(...)`（§13.4），`fuse` → `final`；丢弃 `final < threshold`；按 `(final DESC, time_anchor DESC, atom_id ASC)` 排序；`topk`/`list` → 截到 `limit`；`latest` → 在已排序命中中取最新的一条（`value` 情形按 `(subject, predicate)` 分组）；`summary` → 在排序后的列表上按上面的方式生成；`explain` 时每个命中的 `score_details` 带 `recalled_by`（其通道）、`pool_size` 与 `channel_sizes`。
     - `projection` 由最终命中集合得到：`'atom'` → None；`'value'` → `[atom.content.get('objects') or [atom.text] for each hit]`（按命中顺序排列的「值列表」的列表；HMS 所说的「最新偏好值」即 `result.projection[0]`）；`'time'` → `[(time_anchor or created_at, atom_id)]` 按升序排列（时间线）；`'entity'` → `{entity_key: [atom_id, ...]}`（键有序，atom id 按命中顺序）；`'topic'` → `{topic: [atom_id, ...]}`，规则同上。
  6. 返回 `AxisResult(hits=[AtomHit(atom, score, score_details (when explain), matched_axes)], project_axis, aggregation, projection, count, summary, primary_axis, retrieval)`——排序查询的 `retrieval` 是胜出级别的 `Pool.stats()`（仅轴 / count 为 `{}`；`compile()` 把它复制进 `ContextIR.retrieval`）；`retrieval_stats` 出参收到同一个字典。
  副作用：只读；只触及 `memory_atoms`、`atom_entity`、`entities`、`atom_topic`、`atom_anchor`、`atom_vectors`、`atom_fts`（由 `store.trace` 断言）。
- `recall_pool(...) -> Pool` 与 `Pool` — 第 3 步的召回，公开以便 `compile()` 与测试复用；`CHANNELS = ('semantic', 'keyword', 'entity', 'recency')`，`GATE_CHANNEL = 'gate'` | 召回池。
- `count_axis_query(store, aq) -> int` — `execute_axis_query(..., aq with aggregation='count').count` | 计数。
- `search(store, index, embedder, *, query: str, filters: dict, top_k: int, threshold: float, explain: bool, as_of: str, include_expired: bool, scoring: ScoringConfig, capabilities: dict, retrieval: RetrievalConfig | None = None) -> list[dict]` — `plan = plan_query(query, as_of=as_of)`；`aq = AxisQuery(filters, text=query, limit=top_k, as_of, include_expired, aggregation='topk', plan=plan.as_dict())`；`execute_axis_query(...).hits` → `atom_to_row(atom, score, score_details)` | 搜索。
- `get_all(store, *, filters, top_k, as_of, include_expired) -> list[dict]` — `AxisQuery(text=None, limit=top_k)`（无 plan → 不加 scale 约束）；返回行不带 `score`；顺序为 `time_anchor DESC, created_at DESC, atom_id ASC` | 全量读取。
- `atom_to_row(atom: MemoryAtom, *, score: float | None = None, score_details: dict | None = None, anchor_ids: list[str] | None = None) -> dict` — `return atom.to_public_dict(score=score, score_details=score_details, anchor_ids=anchor_ids)`（§4.1；实现位于 group A）| 行投影。

### §13.4 `utils/scoring.py` — 信号与融合

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

块内英文注释对照：
- `Signals.active`：`('semantic','keyword','entity','recency','support')` 的子集。
- `semantic_scores`：取 `max(0, cos)`；缺失 → `0.0`。
- `keyword_score`：`Σ_{t ∈ Q ∩ A} w(t) / Σ_{t ∈ Q} w(t)`，其中 `w(t) = identifier_weight if is_identifier(t) else 1.0`；Q 为空 → `0.0`；`atom_tokens = content_tokens(text) ∪ topic_tags ∪ 实体键 ∪ 实体 canonical 分词`。
- `entity_score`：`|Q ∩ A| / |Q| × 1/(1 + 0.001·max(0, deg−3)²)`，`deg` 取命中实体中的最大度数（枢纽衰减在超过 3 条链接后才开始，因此 S4b 中 `retry.ts`（度数 3）恰好得 1.0）；Q 为空 → `0.0`。
- `recency_score`：`exp(−ln2 · age_days / half_life)`；`t` 为 None → `0.0`。
- `support_score`：`tanh(n / 3)`。
- `compute_signals`：`active = {semantic, recency, support} ∪ {keyword iff plan.content_tokens} ∪ {entity iff plan.entity_hints}`。
- `fuse`：返回 `(final, raw, max_possible)`，其中 `raw = Σ_{k∈active} w_k·s_k`，`max_possible = Σ_{k∈active} w_k`，`final = raw / max_possible`。

`score_details` 的键（顺序严格如下）：`semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold, recency_score, support_score, keyword_backend ('overlap'), axis_gates ({'time': [gte, lte] | None, 'entity': [...] | None, 'obs_type': [...] | None, 'scope': {...}, 'validity': as_of}), primary_axis, route, signals (list of active), recalled_by（§13.3 第 3 步的召回通道列表：`['gate']` 或按通道顺序排列的 `CHANNELS` 子集）, pool_size（整数）, channel_sizes（{通道: 召回的 id 数}）`。前 14 个键及其值与穷举路径完全相同；三个召回键追加在后。`bm25_score` 就是关键词分（沿用该名字是为了兼容 Mem0 的消费方）。阈值作用于 `final_score`（与 Mem0 的差异，此处明确记录）。

---

## §14 Support Graph 支持图（group E：`hms_elite/graph/`）

包 docstring：`"""Support Graph plane: witnessed typed edges, boundary detection, coarsen/project/refine, explanation.\n支持图平面：带 witness 的类型化边、边界检测、Coarsen/Project/Refine、解释。"""` 目的：继承路径 | 继承路径。输入：节点 id 集合、边规格 | 节点集合、边规格。输出：边、InterfaceEP、子图 | 边、InterfaceEP、子图。副作用：只通过 `StateStore` 写图相关表 | 仅经 StateStore 写图表。

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

块内英文注释对照：
- `insert_edge`：必须在构造模型**之前**执行 `if not witness_anchor_ids: raise WitnessMissingError(details={'object': 'edge', 'id': None, 'missing': []})`（否则 pydantic 会先抛错）；随后构造 `GraphEdge`，其 scope 取源节点的 scope、`created_at = store.now()`；再委托给 `store.insert_edge`（后者会重新校验并预检 anchors，见 §3/§8）。
- `neighbors`：BFS；返回 `(nodes incl. seeds, edges, paths: node_id → [edge_id, ...] from the nearest seed)`；确定性保证：前沿按 `node_id` 排序，边按 `(created_at, edge_id)` 排序。
- `internal_edges`：两个端点都在 S 内的边。
- `boundary`：恰有一个端点在 S 内的边；`src ∈ S` 时方向为 `'out'`。
- `cohesion`：`E_in`、`E_b` 含义同上；`|E_in| == 0` 时为 `0.0`；否则为 `0.5·min(1, |E_in|/max(1, |S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|)`。
- `components`：在给定 kinds 的有效节点（且 `graph_nodes.scale == scale`）之间，基于 `scale` 层上 `edge_types` 类型的有效边求连通分量（L2 粗化：`kinds=('ep',), scale='L1_micro'`；L3 粗化：`kinds=('interface_ep',), scale='L2_meso'`）；当相邻成员（按 `time_anchor` 排序）间隔超过 `gap_hours` 时，分量在此处被切开；每个成员必须与至少另一个成员共享 ≥1 个话题（`atom_topic`）；尺寸落在 `[min_size, max_size]` 之外的分量被丢弃（超出上限时保留最新的 `max_size` 个）；每个列表按 `time_anchor` 排序。
- `owner_interface_ep`：该 scale 层上、support 中包含 `node_id` 的有效 IEP。
- `stats`：`scope` 为 None 表示统计整库。返回 `{'nodes': {kind: n for EVERY NodeKind (entity included)}, 'edges': {type: n for EVERY EdgeType}, 'valid_edges', 'invalid_edges', 'boundary_edges', 'interface_eps', 'unresolved_conflicts', 'witness_coverage', 'projection_coverage', 'traceable_high_edges', 'transitions': n}`；其中 `witness_coverage` = 至少带 1 个 witness 的边 / 全部边（应为 1.0），`projection_coverage` = 至少有 1 行 projection 的边界边 / 全部边界边（无边界边时为 1.0），`traceable_high_edges` = `scale ∈ {L2_meso, L3_macro}` 且至少有 1 行 `edge_projections` 的边 / 这类边总数（无此类边时为 1.0）。节点与边的计数覆盖匹配 scope 的**所有**行，不区分 `valid`；`valid_edges`/`invalid_edges` 是唯一按有效性拆分的两项；`interface_eps` 只统计有效 IEP。

Docstring 行：`insert_edge` — "Insert one typed edge with at least one witness anchor | 插入一条至少带一个 witness 的类型化边"；In: endpoints, type, witnesses | 端点、类型、witness；Out: edge_id | 边 id；Effect: writes graph_edges/edge_witnesses (+edge_projections by trigger) | 写边表。

### §14.2 `graph/operators.py`

- `coarsen(store, index, embedder, graph: SupportGraph, node_ids: list[str], *, target_scale: str = "L2_meso", policy: str = "rule", llm: LLMBase | None, at: str, scope: Scope, min_cohesion: float) -> InterfaceEP | None` — Coarsen：低层子图 → InterfaceEP | 低层子图→高层接口。输入：≥2 个有效节点 id（`L2_meso` 时全部为 `ep` 类型，`L3_macro` 时全部为 `interface_ep` 类型——后者的 slots 从 `interface_eps.slots_json` 读取），且 scope 相同 | 节点集合。输出：已持久化的 InterfaceEP（`cohesion < min_cohesion` 时为 None）| InterfaceEP。副作用：写入一个 episode 原子（`source_kind=interface_ep`、`scale_level=target_scale`、`anchor_ids=evidence_anchor_ids`、`text=slots.gloss()`、`source_ep_ids = [ref_id of every ep-kind support node]`（L3 情形：取各成员 IEP 的 `source_ep_ids` 并集）、`topic_tags = ordered_union(*[atom.topic_tags for n in S])`、`entity_ids = ordered_union(*[atom.entity_ids for n in S])` 作为 `extra_mentions` 传入），以及 `interface_eps`、`interface_ep_support` | 写原子与接口表。说明：`S` 按 `time_anchor` 排序；`E_in = internal_edges(S)`，`E_b = boundary(S)`；`cohesion = graph.cohesion(S)`（存储时不做四舍五入）；slots 由 `hms_elite.llms.rule.coarsen_rule(...)` 生成（§11.5；policy 为 `rule`）或由 `coarsen` prompt 生成（policy 为 `llm`；任何缺少有效 `slot_support` 引用、或引用 ∉ S、或取值不可用——`r` ∉ `Reward.values()`、文本槽位非字符串——的 slot 都会被替换为规则值并告警；`kappa.visibility` 缺失或 ∉ `Visibility.values()` → 采用规则可见性、保留 LLM 的约束）；`evidence_anchor_ids = sorted(∪ atom_anchor_ids(n) for n in S ∪ witnesses(E_in))`（断言其等于该并集——证据保全）；`slots.kappa.evidence = evidence_anchor_ids`；**invariant/residual 的补全**（规则或 LLM 的输出只带 `support_refs`）：对每个 invariant 与 residual，`support_node_ids = [S[ref] for ref in support_refs]`，`anchor_ids = sorted(∪ store.atom_anchor_ids(n) for n in support_node_ids)`，`Invariant.confidence = round(mean(store.get_atom(n).confidence for n in support_node_ids), 4)`；residual 另有：`cohesion < min_cohesion + 0.1` 时标 `low_cohesion`；`support_hash = sha256(','.join(sorted(S)))`；scope 内已存在相同 `support_hash` 的有效 IEP → 原样返回（幂等）；已存在的有效 IEP 若其 support 是 `S` 的真子集 → 将其置为无效，并插入一条 `revision` 边 old_iep_node → new_iep_node（`scale=target_scale`、`is_boundary=False`、witness 为新成员的 anchors、`created_by='consolidation'`）。
- `project(graph, iep: InterfaceEP, *, at: str) -> list[str]` — Project：边界边 → 高层边 | 边界边→高层边。输出：高层边 id | 高层边 id。副作用：写 `graph_edges(is_boundary=1, witness_edge_id=low)`；触发器写 `edge_projections`；被合并进同一条既有高层边的额外低层边通过 `store.add_projection(high, low)` 记录 | 写投影。说明：对每条边界边 `e`：`outer' = graph.owner_interface_ep(outer, scale=iep.scale).atom_id or outer`；方向与 `e` 一致（`out` 时为 `iep.atom_id → outer'`，`in` 时反向）；`scale=iep.scale`、`witness_anchor_ids = witnesses(e)`、`created_by='project'`、`proposer='project-v1'`、`confidence = e.confidence`；内部边被吸收（永不投影）。后置条件断言：每条高层边至少有 1 行 `edge_projections`（否则 `StoreError(STORE_003)`）。
- `refine(store, graph, interface_ep_id: str, *, hops: int = 0, edge_types=None) -> dict` — Refine：按需把高层对象展开为低层支撑子图 | 按需细化。输出：`{'interface_ep': InterfaceEP dict, 'nodes': [GraphNode dicts], 'edges': [GraphEdge dicts incl. witness_anchor_ids], 'boundary_edges': [BoundaryEdge dicts], 'anchors': [Anchor dicts], 'projections': {high_edge_id: [low_edge_id]}}` | 子图。副作用：只读。说明：`hops > 0` 时经 `graph.neighbors` 越过边界继续扩展；调用方：`explain()` 与 `StateCompiler`（**仅**在解析后的 mode 为 `deep` 时——绝不在 `fast`、也绝不在 `axis_graph` 时调用）（由 `fast`/`axis_graph` 模式下的 `store.trace` 以及 import／调用方检查来验证）。
- `promote_skills(store, index, embedder, graph, *, scope: Scope, at: str, anchor_id: str) -> list[str]` — 把有效 IEP 中 kind 为 `constraint` 且 `len(support_node_ids) >= 2` 的 invariant 提升为 skill 原子 | 技能提升。输出：新建的 skill 原子 id | 技能原子 id。副作用：`create_atom(MemoryAtom(atom_type=state, scale_level=L3_macro, source_kind=skill, source_id=iep.interface_ep_id, text=f"Skill: {inv.text}", subject='project', predicate='skill', anchor_ids=inv.anchor_ids, topic_tags=['skill'] + topic_tags(inv.text), confidence=inv.confidence, extracted_by='rule@skill-v1'))`（在 scope 内按 hash 去重：已存在有效 skill 则复用）+ 一条 `distill` 边 `iep.atom_id → skill_atom_id`（`scale='L3_macro'`、`is_boundary=False`、witness 为 `inv.anchor_ids`、`created_by='consolidation'`、confidence 0.9）——并且**可追溯到低层**（HMS §3.7）：`low = [valid use edges support_node → constraint_atom whose dst atom text == inv.text, ordered by edge_id]`；`low` 非空时 `witness_edge_id = low[0].edge_id`（触发器写入第一行 `edge_projections`），其余每条 `e ∈ low[1:]` 执行 `store.add_projection(distill_edge, e.edge_id)`；若不存在这样的低层边，则该 distill 边只由 anchors 作 witness | 写技能原子与 distill 边。
- `round_trip_check(graph, iep) -> bool` — `set(b.edge_id for b in refine(...)['boundary_edges']) == set(boundary(support_node_ids))`（测试辅助）| 往返校验。

### §14.3 `graph/explain.py` — `why()` 内核

- `resolve_target(store, target_id: str) -> dict | None` — 解析顺序：`memory_id`（→ 头部原子，否则取最新版本）→ `atom_id` → `ep_id`（经 `eps.atom_id`）→ `obs_id`（经 `observations.atom_id`）→ `interface_ep_id` → `edge_id`（→ `{'kind': 'edge', 'node_ids': [src, dst]}`）；返回 `{'kind': 'atom'|'interface_ep'|'edge', 'id', 'node_id'|'node_ids', 'text', 'resolved_from': 'id'}` 或 None | 解析目标。
- `explain(store, graph, target: dict, *, hops: int = 2, edge_types: Iterable[str] | None = None, include_invalid: bool = False, max_nodes: int = 60) -> dict` — 输入：已解析的目标 | 已解析目标。输出：§17.3 的 `why` 结构 | why 返回。副作用：只读（遇到 interface_ep 节点时调用 `refine(hops=0)`）。算法：`edge_types = edge_types or WHY_EDGE_TYPES`；逐层展开，最多 `hops` 层：对前沿中的每个节点，取 `edge_types` 类型的**入边**（有效边，其源节点加入下一层前沿）以及**出边**中的 `use` 边（其目标节点加入）；每访问到一个 `interface_ep` 节点就对其执行 refine，并把它的支撑节点／边／边界并入结果（由 `refined_from` 记录来源）；`paths[node] = [edge_ids from target]`；`unresolved_conflicts` = 访问到的、`resolved_at IS NULL` 的 `conflict` 边；anchors = 边的 witness 并集 ∪ 已访问节点的原子 anchors（未被质疑的）；`chain = render_chain(...)`；`explanation = "\n".join(chain)`。
- `render_chain(store, target_node_id: str, edges: list[GraphEdge], *, hops: int) -> list[str]` — 第 0 行是目标原子的文本；随后最多走 `hops` 步：在节点 `n` 处，从 `edges` 中挑 `confidence` 最高的入边（并列时依次比 `created_at`、`edge_id`）→ 输出 `f"<-{type}[{refs}]- {src_text}"`，然后从 `src` 继续；若无入边，则挑 confidence 最高的出向 `use` 边 → 输出 `f"-use[{refs}]-> {dst_text}"`，然后从 `dst` 继续；两者都不存在时停止；`refs = ', '.join(anchor.source_ref or 'sha256:' + anchor.hash[:12] for the edge's witness anchors in witness order)` | 链渲染。

---

## §15 编译器 Compiler（group F：`hms_elite/compiler/`）

包 docstring：`"""State compiler: axis candidates first, graph on demand, deterministic slots, budget, Work State Card and PC-IR.\n状态编译器：先轴候选，按需展开图，确定性槽位、预算、Work State Card 与 PC-IR。"""`

### §15.1 `compiler/slots.py`

- `SLOT_RULES`（有序、完备）：(1) `atom.archived_reason is not None or (atom.valid_to is not None and atom.valid_to <= as_of) → audit`（explain.reason 为 `'archived:' + (archived_reason or 'expired')`；若是经由 `revision` 边到达则为 `'revised'`，并带 `superseded_by_text`）——`valid_to`（expiration_date）位于**未来**的存活原子**不**归入 audit；(2) `source_kind == 'skill' → skills`；(3) `obs_type == 'constraint' → constraints`；(4) `obs_type in {preference_event, habit} → preferences`；(5) `atom_type == 'episode' → episodes`（explain 为 `{'r','a','y','u','ep_id'|'interface_ep_id'}`）；(6) 其余 → `facts`（`entity` 原子永远到不了编译器——§15.2 第 2 步已将其排除）。
- `assign_slot(atom: MemoryAtom, *, as_of: str, reason: str | None = None) -> Slot` | 槽位分配。
- `item_from_atom(atom, *, slot, score, anchor_ids, source, hops, explain) -> ContextItem` — `item_id = atom_id`，`tokens = estimate_tokens(text)` | 构造项。
- `kappa_items(episodes: list[ContextItem], selected_constraint_texts: set[str]) -> list[ContextItem]` — 为 `slots.kappa.constraints[i]` 生成 `f"{atom_id}#kappa:{i}"`，条件是其归一化文本（按 §11.3 规则 1 归一化）∉ `selected_constraint_texts`（即所有被选中的 `constraints` 项 ∪ 所有 `audit` 项的归一化文本，因此已归档／已修订的约束绝不会被重新合成），**且**本次调用中尚未为更早的 episode 输出过（episodes 按 slot 顺序处理；先到者胜——同一条约束文本绝不产生两个 κ 项）；`anchor_ids` 取该 episode 的 anchors；`source='system'` | κ 合成项。
- `invariant_items(iep: InterfaceEP, iep_atom_id, selected_constraint_texts) -> tuple[list[ContextItem], list[ContextItem]]` — 返回 `(由 kind=constraint 的 invariant 生成的约束项，item_id 为 f"{iep_atom_id}#inv:{i}"；以及为 failed_path/stable_fact/residuals 生成的 audit 项，reason 为 'invariant:<kind>'|'residual:<kind>')` | 不变量项。
- `evidence_items(anchors: list[Anchor]) -> list[ContextItem]` — `item_id=f"anc:{anchor_id}"`，slot 为 audit，`text=anchor.evidence_line()`，`anchor_ids=[anchor_id]`，`explain={'kind': 'evidence', 'source_ref', 'hash', 'timestamp', 'revoked_at', 'permission_scope'}`；按 `(timestamp, source_ref, anchor_id)` 排序 | 证据项。
- `derive_next_action(*, episodes, constraints, residuals: list[str], conflicts: list[tuple[str, str]], goal: str) -> str` — `neg` = 最新的负向 episode，`pos` = 最新的正向 episode；`kept = [c.text for c in constraints if not c.text.startswith("Do not re-apply")][:3]`；`suffix = f" while keeping constraints: {'; '.join(kept)}" if kept else ""`；(1) `neg and (pos is None or neg.time_anchor > pos.time_anchor)` → `f"Do not re-apply '{neg.explain['a']}'; investigate '{neg.explain['y']}'{suffix}"`；(2) `pos` → `f"Continue from '{pos.explain['a']}' (last outcome: {pos.explain['y']}){suffix}"`；(3) 有 residuals → `f"Resolve: {residuals[0]}"`；(4) 有 conflicts → `f"Resolve conflict between '{x}' and '{y}'"`；(5) `f"Continue: {goal}"` | 下一步规则。

### §15.2 `compiler/compile.py`

```python
class StateCompiler:
    def __init__(self, store, index, embedder, graph: SupportGraph, config: HMSConfig) -> None
    def compile(self, query: str, *, scope: Scope, filters: dict, budget_tokens: int | None = None, mode: str | None = None,
                as_of: str | None = None, require_witness: bool | None = None, audience: str | None = None) -> ContextIR
```

规范算法（只读；执行前后行数完全一致）：
1. `as_of = as_of or store.now()`；`mode = mode or config.compiler.default_mode`（`CompileMode(mode)`，否则 `VALIDATION_013`）；`plan = plan_query(query, as_of=as_of, mode=mode)`；`route = plan.route`；`budget = budget_tokens or config.compiler.budget_tokens`；`audience` 对下面所有环节生效（`rank(x) >= rank(audience)`；为 None 时不做门控）。
2. 轴候选：`hits = execute_axis_query(AxisQuery(filters, scope, text=query, limit=config.compiler.candidate_k, as_of, aggregation='topk', plan={**plan.as_dict(), 'atom_types': [t for t in AtomType.values() if t != 'entity']}), threshold=0.0).hits`（entity 原子只是索引内部的导航对象，绝不进入 card；若 `audience` 来自 `filters`，§13.1 的 `visibility` 格门控此时已经生效）。
3. 工作态增强，触发条件为 `route ∈ {work_state, detective}`、或 `mode ∈ {deep, axis_graph}`、或（`mode == 'auto'` 且 `len(store.list_eps(scope)) >= 1`）：并入 scope 内全部有效的 `episode` 原子、全部有效的 `obs_type='constraint'` 原子、以及全部有效的 `source_kind='skill'` 原子（若尚未命中则 score 记 0.0；同样受 audience 门控）。
4. 图展开（`mode == 'fast'` 时绝不进行；`auto` 时当且仅当第 3 步做了增强；`deep` 与 `axis_graph` 时总是进行）：种子 = 候选节点 id；`nodes, edges, paths = graph.neighbors(seeds, hops=config.compiler.graph_hops, edge_types=WHY_EDGE_TYPES, valid_only=False)`；有效的邻居原子若其 visibility 通过 audience 门控，则以 `score = 0.5 × max seed score`、`source='graph'`、`hops` 加入候选（被门控挡下的邻居 → `retrieval['dropped_governance'] += 1`）；`revision` 边 `old → selected` 会把 `old`（已归档）放入 `audit`（`reason='revised'`，`superseded_by_text = new.text`）；`resolved_at IS NULL` 且两端都被选中的 `conflict` 边 → 计入 `unresolved_conflicts`；**仅在 `deep` 模式下**（以及 `auto` 解析为 deep 时）：每个访问到的 `interface_ep` 节点 → `refine(hops=0)` → `invariant_items`（若某 invariant 的 `anchor_ids` 全部未通过 anchor 权限门控，则丢弃，`dropped_governance += 1`）；在 `axis_graph` 模式下，访问到的 IEP 节点作为普通 episode 候选加入，且从不调用 `refine`；`retrieval['graph_hops'] = hops`；`retrieval['ablation'] = ABLATION_OF_MODE[resolved mode]`（`auto` 情形：未发生展开时为 `'axis'`，否则为 `'axis_graph_support'`）。
5. 按 `atom_id` 对候选去重（取最大 score）；主 anchor 属于任一被选 episode 的 anchors 的 `event` 原子被丢弃（`retrieval['dropped_covered'] += 1`）。
6. 槽位分配（§15.1，`assign_slot(atom, as_of=as_of)`）→ 六个列表；`selected_constraint_texts` = constraints 列表 ∪ audit 列表的归一化文本；`kappa_items(episodes, selected_constraint_texts)` 追加到 constraints（跨 episode 去重，§15.1）；每项的 `anchor_ids` = `store.atom_anchor_ids(atom_id, include_disputed=True)`（因此即使某原子的全部 anchor 都被质疑，它仍能产出一个类型完好的项），并设 `explain['disputed_anchor_ids'] = [ids disputed for this atom]`；`permission_scope` 未通过 audience 门控的证据 anchor 会从该项的 `anchor_ids` 中剔除（`dropped_governance += 1`；若某项因此一个 anchor 都不剩，则整项丢弃）。
7. `goal = query`；`next_action = derive_next_action(...)` 在**预算裁剪之前**的候选集合上计算（因此被裁掉的 episode 仍会影响它）。
8. 预算（`fit_budget`）：`cap[slot] = floor(share × budget)`；facts/episodes/preferences/constraints/skills 各自按 `(score DESC, time_anchor DESC, item_id ASC)` 装填直到 `cap`；constraints 即使超出其 cap 也至少保留 `min_constraints` 项；未用完的容量（`budget − Σ used`）先溢出给 facts，再给 episodes（各自沿已排序列表继续取）；audit 部分：为每个被选中项的 anchors 生成的证据项（顺序见 §15.1）**永远保留**（witness 完备性——不受 cap 限制），其他 audit 项则在 audit cap 内装填；`used_tokens` = 所有保留项的 `tokens` 之和（含证据；只有证据可能使其超过 `budget_tokens`，此情况记入 `retrieval['over_budget_evidence']`）；`retrieval['dropped']` = 未被选中的候选数量。装填完成后，每个非 audit 槽位按 `(time_anchor ASC nulls last, item_id ASC)` 重新排序，以保证渲染稳定。
9. `retrieval = {'signals': plan-active signals, 'route', 'plan': plan.as_dict(), 'mode': resolved mode, 'ablation', 'candidates': n, 'graph_hops', 'dropped', 'dropped_covered', 'dropped_governance', 'over_budget_evidence', 'primary_axis': result.primary_axis（§13.3 第 4 步的阶梯级别）, 'pool_size', 'channel_sizes', 'channels_used'（候选阶段的 `Pool.stats()`，§13.3 第 3 步；无文本编译时为 `0 / {} / []`）, 'rerank': None, 'anchors': [Anchor dicts (incl. 'revoked_at', 'permission_scope') for every anchor referenced], 'audience': audience}`；返回 `ContextIR(mode=resolved mode, ...)`（校验器强制要求 item id 互不重复且每项都有 witness）。

### §15.3 `compiler/render.py` — Work State Card（确定性输出）

`render_card(ir: ContextIR) -> str` — 精确格式（用 `\n` 连接，结尾不带换行；`→` 为 U+2192；某个小节为空时输出 `- (none)`）：
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

块内 `←` 注释对照：
- `Scope:` 行——只输出已设置的 scope 键，按上述顺序、以空格分隔。
- `## Valid Facts`——先按 slot 顺序输出 facts，再输出 `r ∈ {positive, neutral, unknown}` 的 episodes，形如 `"- {a} → {y or '?'} [{r}]"`。
- `## Invalidated Assumptions`——其他归档原因的写法为 `"- <text> (archived: <reason>)"`。
- `## Failed Paths`——`r == negative` 的 episodes，按 slot 顺序。
- `## Preferences`、`## Skills`——仅在非空时才输出该小节。
- `## Evidence`——audit 中的证据项，按 slot 顺序。

`ContextIR.to_text()` 通过延迟导入调用它（方法内部执行 `from hms_elite.compiler.render import render_card`），从而使 group A 在模块级别不依赖 F。

### §15.4 `compiler/pc_ir.py`

- `lower(ir: ContextIR, *, require_witness: bool = True) -> PCIR` — 输入：ContextIR（其 `retrieval['anchors']` 携带 anchor 行）| ContextIR。输出：PCIR | PCIR。副作用：无。`payload = {'goal', 'facts': [entry], 'episodes': [entry + r,a,y,u], 'preferences', 'constraints', 'skills', 'invalidated': [entry + superseded_by_text], 'next_action', 'as_of', 'scope'}`，其中 `entry = {'item_id', 'text', 'anchor_ids', 'atom_id', 'memory_id'}`；`witness = {'items': {item_id: anchor_ids}, 'anchors': [anchor dicts], '_all': sorted union}`；`human_gloss = render_card(ir)`；证明义务：当任一非 audit 项的**未被质疑的** anchors（`item.anchor_ids` 减去 `item.explain.get('disputed_anchor_ids', [])`）为空、或其中没有任何一个出现在 `retrieval['anchors']` 中时，`evidence_coverage` 被违反（`details={'missing_witness': [item_ids]}`）；当 `ir.unresolved_conflicts` 非空时，`no_unresolved_conflict` 被违反；`governance` 在 `retrieval['audience']` 为 None 时跳过，在以下情形被违反：任一项满足 `rank(item.visibility) < rank(audience)`（编译器输出不会出现这种情况——选择阶段已经门控——但手工构造的 ContextIR 仍可能触发）；或某个被选 episode 的 κ 约束（按归一化文本比对）不在 `constraints ∪ audit` 中；或某个被非 audit 项引用的证据 anchor 满足 `rank(permission_scope) < rank(audience)`；当任一被非 audit 项的未质疑 `anchor_ids` 引用的 anchor 带有 `revoked_at` 时，`not_revoked` 被违反（`details={'revoked': [anchor_ids]}`）。`require_witness` 为真且存在任何违反 → `ProofObligationError(PCIR_001, details={'violated': [...], 'missing_witness': [...], 'unresolved_conflicts': [...], 'revoked': [...], 'governance': [...]}, suggestion="call why(<edge_id>) and feedback(..., 'state_wrong'|'state_outdated') to resolve, or lower with require_witness=False")` | PC-IR 降低。

---

## §16 运维操作（分组 G：`hms_elite/ops/`）

包 docstring：`"""Operations plane: feedback, checkpoint/rollback, handoff, transition export and the synchronous StateScheduler.\n运维平面：反馈、检查点/回滚、交接、状态转移导出与同步调度器。"""` 本组的每一个公开写操作都以 `ingest.events.system_event(...)` 起头（它产生的 anchor 见证本次创建的每一行），并且整体跑在同一个 `store.transaction()` 里。

### §16.1 `ops/feedback.py`

```python
class FeedbackProcessor:
    def __init__(self, store, index, embedder, graph, config) -> None
    def process(self, *, target_id: str | None, feedback: str, reason: str | None, correction: str | dict | None, actor_id: str | None, scope_hint: Scope | None) -> dict
```

前置流程：`feedback = FeedbackType(feedback)`（未知值 → `ValidationError(VALIDATION_006, details={'allowed': [...]})`）；解析目标（`memory_id` → head atom；`atom_id`；`edge_id`；`interface_ep_id`；上下文条目 id `'<atom_id>#...'` → atom；`anchor_id` → `{'kind': 'anchor'}`（**仅**在 `evidence_wrong` 且 `correction={'revoke': True}` 时接受，转交下文的 `revoke()`）；其余 → `NOT_FOUND_001`）——**`state_missing` 除外**，这一类型允许 `target_id` 无法解析或为 `None`：此时 scope 依次取 `correction['scope']`（dict）解析出的 `Scope`、或 `Scope.key()` 形式的 `target_id`（`user_id=alice&run_id=run-a`），再退到 `scope_hint`，都没有则 `VALIDATION_001`；`target_kind='scope'`，`feedback_events.target_id` = 该 scope key；`state_missing` 必须带 `correction`（否则 `VALIDATION_007`；若为 dict 则必须含 `text`）；edge 专用的反馈类型作用在非 edge 目标上 → `VALIDATION_007`；`scope = 目标所在 scope`（或上面得到的 scope）；`ev, fb_anchor = system_event(object_type='feedback', event_type='feedback', payload={target_id, feedback, reason, correction})`；`store.insert_feedback(...)`；atom 类目标写一条 `FEEDBACK` history（`old_memory=text, new_memory=None, details={'feedback', 'reason'}`）。之后进入对应处理器：

| FeedbackType | 目标 | 动作（全部由 `fb_anchor` 见证；action 字典按此顺序排列） |
|---|---|---|
| `state_wrong` | atom | `archive` head（reason 为 `feedback`）；若 `correction` 是 str → 用 `revise_atom` 生成后继版本（text=correction，坐标照抄，`time_anchor=now`，`anchor_ids=[fb_anchor]+old.anchor_ids`，revision 边的 `witness_anchor_ids` 用同一个列表，confidence 0.95，`created_by='feedback'`，history `UPDATE`）→ `create`、`revision_edge`；触及该目标的任何未解决 `conflict` 边 → `store.resolve_conflict(edge, at, by_anchor_id=fb_anchor, resolution_edge_id=<revision 边或 None>)` → `resolve_conflict`；从该目标出发的 `use`/`distill` 边 → `invalidate_edge(reason='source_archived')` |
| `state_outdated` | atom | `archive` head（当 `correction` 是带 `valid_to` 键的 dict 时 `valid_to = correction['valid_to']`，否则取 now；reason 为 `feedback`）；若 correction 是文本 → 生成后继版本（`valid_from=now`，`time_anchor=now`，`anchor_ids=[fb_anchor]+old.anchor_ids`，实体/主题按 correction 重新挂接，revision 边的见证者用同一个列表）+ `revision_edge`，history `UPDATE`；confidence 不变。注意：后继版本只在 `as_of >= 反馈时刻` 时可见（有效性是一个 WHERE 子句，§8）；`compile(as_of=<更早时刻>)` 仍会看到已归档的前驱版本处于有效状态 |
| `state_missing` | atom / episode / scope（无目标） | 由 `correction` 新建一条 lineage（`correction` 为 str，或 dict `{'text', 'obs_type'?, 'scope'?}`：`atom_type=state`、`scale_level=state`、`obs_type=fact`（或 `correction['obs_type']`）、`subject='user'`、`predicate='states'`、`source_kind=feedback`、`source_id=feedback_id`、`anchor_ids=[fb_anchor]`（存在目标时再加上目标的 anchors），history `ADD`）→ `create`；若目标是 episode → `insert_edge(use, new → target, witness fb_anchor)` |
| `privacy_violation` | atom | **唯一会改动可见性的处理器**：对 head 以及与它共享任一 anchor 的每个 atom（`atoms_for_anchor`）执行 `set_visibility` 为 `private`；对这些 anchors 执行 `set_anchor_permission(private)`；边不动 |
| `next_action_wrong` | episode atom / 上下文条目 | `create` 一个 constraint atom `(obs_type=constraint, subject='project', predicate='must_not', objects=[target.a], text=correction or f"Do not: {target.a}", topic 'next_action', anchors=[fb_anchor]+目标 anchors, source_kind=feedback)`；`set_confidence(target, −0.2)`；`insert_edge(distill, target → constraint, witness fb_anchor)` |
| `relationship_not_useful` | edge | `invalidate_edge(reason='feedback:not_useful')`；然后以同一 reason 执行 §16.1 撤销步骤 (3b) 的投影级联：由它投影出的高层边（`projections_from`）只在其 `projections_of` 的来源**全部**已失效时才失效（沿 L2 → L3 传递）→ 每条倒下的边一个 `invalidate_edge` 动作，低层边在前 |
| `connection_reason_wrong` | edge | `correction` 可以是 `{'edge_type': ..., 'label': ...}` 或一个 str 标签（`corrected_type = correction.get('edge_type') or old.edge_type`）；**若 `corrected_type == old.edge_type`**（仅改标签）：不做失效——`revalidate_edge(old)`（本就有效则为空操作）+ `add_edge_witness(old, fb_anchor)` + `set_edge_label(old, label)`，并报告 `{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}`；**否则** `invalidate_edge(old, reason='feedback:wrong_reason')` → `invalidate_edge`，随后 `insert_edge(same endpoints, corrected_type, label, witnesses = old witnesses + fb_anchor, created_by='feedback')` → `insert_edge`（类型不同，所以是全新一行）；若旧边有投影，则为新边重新派生 `project` 风格的高层边（`add_projection`） |
| `evidence_wrong` | atom | `correction` = anchor id 的列表或 CSV（默认取全部非主 anchor）：逐个 `dispute_atom_anchor` → `dispute_anchor`；若不再剩下任何未被质疑的 anchor → `archive`（reason 为 `feedback:evidence_wrong`）（atom 没有未被质疑的见证者就不能保持有效）；`set_confidence(−0.3)` |
| `render_not_like_me` | atom / item | `set_confidence(−0.1)`；`create` 一个 preference atom `(obs_type=preference_event, predicate='dislikes', objects=['render:'+target_id], text=reason or f"User dislikes how '{target.text[:40]}' is rendered.", topic 'render_style', anchors=[fb_anchor])` |
| `render_like_me` | atom / item | `set_confidence(+0.1)`；同上，改用 `likes` |
| `object_important` | atom | `set_confidence(+0.2)`；`add_atom_topic('important')` → `pin`；`bump_mentions(3)` |

返回 `{'message': 'Feedback applied', 'feedback_id', 'actions': [{'action': 'archive'|'create'|'revision_edge'|'resolve_conflict'|'invalidate_edge'|'insert_edge'|'set_visibility'|'set_confidence'|'dispute_anchor'|'pin'|'revoke'|'noop', ...ids}]}`。置信度运算钳制在 `[0, 1]`。`feedback_events.actions_json` 存放同一份列表。

**撤销（HMS §3.2「撤销时能找到派生对象」）** —— `revoke(store, graph, *, anchor_id: str, reason: str | None, actor_id: str | None = None) -> dict`（同一模块；门面为 `HMS.revoke`）：未知 anchor → `NOT_FOUND_001`；已撤销 → `{'revoked': anchor_id, 'archived': [], 'edges_invalidated': [], 'interface_eps_invalidated': [], 'anchor_id': None, 'noop': True}`。否则在一个事务内完成，由 `system_event(object_type='system', event_type='updated', payload={'op': 'revoke', 'anchor_id': anchor_id, ...})` 产出的 `sys_anchor` 见证：(1) `store.revoke_anchor(anchor_id, at=now)`；(2) 对 `atoms_for_anchor(anchor_id)` 中的每个 atom（按升序）：`dispute_atom_anchor(atom, anchor_id, at)`；若此时 `atom_anchor_ids(atom)`（未被质疑的）为空 → `archive_atom(reason='revoked', at)` + 由 `sys_anchor` 见证的 history `ARCHIVE(old=text, new=None, details={'revoked_anchor': anchor_id})` → 计入 `archived`；(3) 对引用该 anchor 的每条有效边（按 `edge_id` 升序）、见证者**全部**已撤销的 → `invalidate_edge(reason='witness_revoked', at)` → 计入 `edges_invalidated`；(3b) **投影级联（存在规则，docs/design/WITNESS_SEMANTICS.md）**：以 (3) 的边为种子建工作队列；对 `projections_from(倒下的边)` 中仍有效、且其 `projections_of` 来源**全部**已失效的每条高层边 → `invalidate_edge(reason='projection_revoked', at)` → 追加到 `edges_invalidated` 并入队（L2 → L3 链）；只要还有任一有效来源，高层边即使自身见证少了成员也保持有效——它自己的见证只由 (3) 判定；(4) 每个 `evidence_anchor_ids` ⊂ 已撤销 anchors 的有效 IEP → `invalidate_interface_ep` → 计入 `interface_eps_invalidated`；(5) 每条 `w_t` ⊂ 已撤销 anchors 的 transition 原样保留（证据永不删除；消费方自行按 `anchors.revoked_at` 过滤）。返回 `{'revoked': anchor_id, 'archived': [...], 'edges_invalidated': [...], 'interface_eps_invalidated': [...], 'anchor_id': sys_anchor}`。`feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` 会调用它，并报告一条 `{'action': 'revoke', ...}` 记录。

### §16.2 `ops/checkpoint.py`

- `checkpoint(store, *, scope: Scope, label: str | None) -> str` —— 单事务：`system_event(object_type='checkpoint', event_type='checkpoint', payload={'label'})`；`atom_state = store.scope_atom_state(scope)`；`edge_state = {e.edge_id: e.valid for e in store.list_edges(scope, valid_only=False)}`；`iep_state = {i.interface_ep_id: i.valid for i in store.list_interface_eps(scope, valid_only=False)}`；`ep_state = store.scope_ep_state(scope)`；`history_seq`；`snapshot_hash`；`insert_checkpoint` → id | 检查点。状态影响：写入 source_events、anchors、checkpoints。
- `rollback(store, graph, *, checkpoint_id: str, reason: str | None) -> dict` —— 未知 id → `ValidationError(NOT_FOUND_001, ValueError)`；单事务，由 `rb_anchor`（`object_type='rollback'`）见证；处理顺序固定：(1) 所有 atom 归档，(2) 所有 atom 恢复，(2b) EP 有效性（`ep_state`：scope 内 `eps.valid_to` 与记录值不同的每个 EP → `archive_ep(at=recorded)` / `restore_ep`；不在 `ep_state` 中的 EP → `archive_ep(at=now)`；这一步跑在 atom 级联之后，以保证快照优先），(3) 边，(4) interface_eps，(5) 回滚 revision 边 + transition；每一步内部按 id 升序。对当前 scope 内的每个 atom：不在 `atom_state` 中 → `archive_atom(reason=f"rollback:{cp}")`，history `ROLLBACK(old=text, new=None, details={'checkpoint_id'})` → 计入 `archived`；在 `atom_state` 中且记录的 `valid_to is None` 但当前已归档 → `restore_atom`，history `RESTORE(old=None, new=text)` → 计入 `restored`；记录为已归档但当前有效（在检查点之后被恢复过）→ `archive_atom(reason=f"rollback:{cp}", valid_to=记录的 valid_to)` → 计入 `archived`；边：不在 `edge_state` 中 → `invalidate_edge(reason=f"rollback:{cp}")` → 计入 `edges_invalidated`；记录为有效但当前无效 → `revalidate_edge` → 计入 `edges_revalidated`；记录为无效但当前有效 → 置为无效；interface_eps 同理（用 `iep_state`）；对同一 lineage 内的每一对 `(被归档的后继 Y, 被恢复的 X)` 插入 `revision` 边 `Y → X`（`label=f"rollback:{cp}"`，见证者 `[rb_anchor]`，`created_by='ops'`，confidence 0.9）→ 计入 `edges_created`；`upsert_transition(rollback transition: source_kind='rollback', source_id=<that edge id>, s_t={'text': Y.text, 'atom_id': Y, 'memory_id', 'valid_from', 'valid_to'}, a_t='rollback', s_t1={'text': X.text, 'atom_id': X, 'memory_id', 'valid_from', 'valid_to'}, r_t='neutral', kappa_t=kappa_t_of(constraints=<texts of valid constraint atoms in scope at time_t1, time_anchor ASC>, visibility=X.visibility, permission_scope=rb_anchor.permission_scope, provenance={'checkpoint_id', 'label', 'reason'}), w_t=[rb_anchor], time_t=Y.created_at, time_t1=now)`（§16.4 的统一 κ_t）；返回 `{'message': f"Rolled back to {label or checkpoint_id}", 'checkpoint_id', 'restored': [...], 'archived': [...], 'edges_invalidated': [...], 'edges_revalidated': [...], 'edges_created': [...], 'anchor_id': rb_anchor}`（第 (2b) 步的 EP 有效性变更与 atom 列表一一对应，不单独报告）| 回滚。注意：atom 内容永不改动；向量永不删除；作用范围是 scope 局部的（指向 scope 外节点的边会随快照一并失效并被报告）。

### §16.3 `ops/handoff.py`

- `build_state_package(store, graph, compiler, *, config: HMSConfig, scope: Scope, filters: dict, goal: str | None, budget_tokens: int, checkpoint_id: str | None, include_edges: bool = True) -> StatePackage` —— `goal = goal or (store.last_ep(scope, run_id=scope.run_id).slots.u if that EP exists) or None`；`ir = compiler.compile(goal or 'handoff', scope=scope, filters=filters, budget_tokens=budget_tokens, mode='deep')`；`pcir = lower(ir, require_witness=config.compiler.require_witness)`；`eps` = 被选中 episode 条目对应的 EP；`interface_eps` = 支撑集与被选中节点有交集的有效 IEP；`edges` = 被选中节点之间的有效边（`include_edges` 为真时）；`atoms` = 上下文条目、`eps`、`interface_eps` 支撑节点以及 `edges` 端点所引用的全部 atom（`store.get_atoms`，按 atom_id 排序）；`anchors` = 所有被引用 anchor id 的闭包（条目、atoms、eps、ieps、边见证者）；`source_events` = 对每个 anchor 取 `store.get_source_event(a.source_event_id)` | 交接包。状态影响：只读，除非门面先行创建了可选的检查点。
- `import_state_package(store, index, embedder, graph, *, pkg: StatePackage, scope_override: Scope | None) -> dict` —— HMS §9.4：接收方导入状态包，而不是重读历史 | 导入状态包。状态影响：单事务，由 `system_event(object_type='system', event_type='created', payload={'op': 'import', 'package_id': pkg.package_id, ...})` 产出的 `sys_anchor` 见证；每个被导入对象保留原 id，除非该 id 已存在（此时跳过并计入报告），或给出了 `scope_override`（此时替换每个对象上的 scope 列，id 仍保留）。顺序：(1) `source_events` 走 `insert_source_event`（同一性去重 → `id_map[old_event_id] = existing_id`），(2) `anchors` 走 `get_or_create_anchor`（被重映射的 anchor 记入 `id_map`；此后所有 anchor 引用都经过 `id_map`），(3) `eps` 走 `insert_ep`（id 已存在则跳过），(4) `atoms` 走 `create_atom(atom, history_event='ADD', history_anchor_id=sys_anchor, history_details={'imported_from': pkg.package_id})`（id 已存在则跳过；`entity_ids`/`topic_tags` 通过 `extra_mentions` 重新挂接），随后回填 `set_ep_atom` 反向指针，(5) `interface_eps` 走 `insert_interface_ep`（它们的 episode atom 来自第 4 步），(6) `edges` 走 `store.insert_edge`（MERGE），(7) `Transition.from_ep`/`from_interface_ep` 行走 `upsert_transition`。返回 `{'imported': {'source_events': n, 'anchors': n, 'eps': n, 'atoms': n, 'interface_eps': n, 'edges': n, 'transitions': n}, 'skipped': {...same keys}, 'id_map': {...}, 'anchor_id': sys_anchor}`。

### §16.4 `ops/transitions.py`

- `ep_transition = Transition.from_ep`（从 `hms_elite.models`（分组 A）再导出——G 永不自行定义）：`source_kind='ep', source_id=ep_id, scale='L1_micro', s_t={'u', 'b', 'kappa': kappa.model_dump(), 'ep_id', 'valid_from', 'valid_to'}, a_t=a, s_t1={'y', 'r', 'ep_id'}, r_t=r, kappa_t=kappa_t_of(constraints=kappa.constraints, visibility=kappa.visibility, permission_scope=<primary anchor's permission_scope>, provenance={'evidence': kappa.evidence, 'ep_id': ep_id}), w_t=ep.anchor_ids, time_t=time_anchor, time_t1=time_anchor` | EP 转移。（在 ingest 阶段写入，§12.6。）
- `interface_ep_transition = Transition.from_interface_ep`（分组 A）：`source_kind='interface_ep', source_id=interface_ep_id, scale=iep.scale, s_t={'u', 'b', 'kappa': kappa.model_dump(), 'interface_ep_id', 'support_node_ids', 'valid_from': iep.created_at, 'valid_to': iep.valid_to}, a_t=slots.a, s_t1={'y', 'r', 'interface_ep_id', 'invariants': [inv.text], 'residuals': [res.text]}, r_t=slots.r, kappa_t=kappa_t_of(constraints=kappa.constraints, visibility=kappa.visibility, permission_scope=<primary anchor's permission_scope>, provenance={'evidence': evidence_anchor_ids, 'interface_ep_id', 'cohesion_score'}), w_t=evidence_anchor_ids, time_t=min(support atoms' time_anchor), time_t1=max(support atoms' time_anchor)`（HMS §4.4：EP **和** InterfaceEP 都是转移来源）| 接口 EP 转移。
- **统一的 `kappa_t`**，对每一种来源都一样（`kappa_t_of`，§4）：`{'constraints': [str], 'visibility': str, 'permission_scope': str, 'provenance': {kind-specific keys}}` —— 训练侧消费方只需按一种形状读取 κ_t。
- `export_transitions(store, *, scope: Scope, since: str | None, scale: str | None, kinds: Iterable[str] | None) -> list[Transition]` —— 单事务：(a) scope 内每个有效 InterfaceEP → `Transition.from_interface_ep(...)`；(b) 两个 ep/interface_ep 节点之间的（有效）`causal` 边 → `source_kind='causal', source_id=edge_id, s_t={'u': i.u, 'b': i.b, 'y': i.y, 'r': i.r, 'kappa': i.kappa, 'ep_id'|'interface_ep_id': i, 'valid_from', 'valid_to'}, a_t=j.a, s_t1={'u': j.u, 'y': j.y, 'r': j.r, 'ep_id': j, ...}, r_t=j.r, kappa_t=kappa_t_of(constraints=j.kappa.constraints, visibility=j.kappa.visibility, permission_scope=<edge primary witness anchor's permission_scope>, provenance={'edge_id', 'label', 'proposer', 'evidence': j.kappa.evidence}), w_t=ordered_union(edge_witnesses(edge), j.anchor_ids)`（**按首次出现顺序，绝不排序**——S6 第 3 行是 `[<anc_e3>, <anc_e2>, <anc_e4>]`），`time_t=i.time_anchor, time_t1=j.time_anchor, scale=edge.scale`；(c) atom 节点之间、标签不以 `rollback:` 开头的 `revision`/`conflict` 边（不论有效与否）→ `s_t={'text': old.text, 'atom_id', 'memory_id', 'valid_from', 'valid_to'}, a_t='revise'|'conflict', s_t1={'text': new.text, ...}, r_t='negative' (revision) | 'unknown' (conflict), kappa_t=kappa_t_of(constraints=<texts of valid constraint atoms in scope at time_t1 (axis_select as_of=time_t1, obs_type='constraint', time_anchor ASC)>, visibility=new.visibility, permission_scope=<edge primary witness anchor's permission_scope>, provenance={'label', 'created_by', 'proposer'}), w_t=edge witnesses (witness order), time_t=old.created_at, time_t1=edge.created_at`；rollback 转移由 `rollback()` 写入（§16.2，κ_t 形状相同）；每一行都走 `upsert_transition`（幂等）；结果为 `store.list_transitions(scope, since, scale, kinds)`，按 `(time_t1, KIND_ORDER[source_kind], source_id)` 排序，`KIND_ORDER` 来自 `hms_elite.enums`（§2）| 导出转移。状态影响：只写 `transitions`。

### §16.5 `ops/consolidation.py` —— `StateScheduler`

```python
class StateScheduler:
    def __init__(self, store, index, embedder, llm, graph, ingestor, compiler, config) -> None
    def run(self, *, scope: Scope | None, phases: list[str] | None) -> dict
```

前置流程：`phases = phases or config.scheduler.phases`（未知阶段 → `VALIDATION_013`）；scope 为 None → 取 `memory_atoms` 中每一个不同的 `(user_id, agent_id, run_id)` scope（每个 run 单独处理）；**部分** scope（例如 `filters={'user_id': U}`）→ 取 `memory_atoms` 中所有与之 `matches` 的不同 scope（因此 S1–S3 之后执行 `consolidate(filters={'user_id': U})` 会跑四个 scope：`run-coffee`、`run-a`、`run-b`、`run-c`）；实体 atom（第 0 步）以实体身份 scope 为键，因此即使多个 run scope 共享同一实体也只创建一次；每个 scope 一次 `system_event(object_type='scheduler', event_type='scheduler', payload={'phases'})` → `sched_anchor`。返回 `{'anchor_id', 'phases': {phase: report}}`（多 scope 时：计数相加，列表型键拼接，`anchor_id` 取最后一个 scope 的 anchor）。

- **orient**（只读）：`{'active_topics': top-5 topics by count over valid atoms with time_anchor within orient_window_days, 'active_entities': top-5, 'open_failures': [ep_id of valid negative EPs (list_eps valid_only) with no later positive EP in the same run], 'pending_ep_nodes': [ep node ids with zero outgoing valid edges], 'coarsen_candidates': graph.components(...), 'last_compiled': store.last_compiled_state(scope) minus 'card' (or None)}`。
- **collect**：`{'re_embedded': n}` —— 对 `atoms_without_vectors(scope, embedder.identity)` 重新做 embedding（`index.upsert`）；`{'re_routed': n}` —— 对 `events_without_atoms(scope, object_types=EXTRACTABLE)` 通过 `ingestor.ingest_events(events, infer=True)` 重新入库（内部对象类型永不重路由）。
- **consolidate**（每一步一个事务；报告计数；报告键恰好为 `entity_atoms, temporal_edges, causal_edges, conflicts, interface_eps, projected_edges, interface_eps_l3, skills, transitions, compiled_states`）：(0) `entity_atoms`（HMS §3.5 的 `entity` 尺度）：对 `store.entities_in_scope(scope)` 的每一行：`linked = store.entity_linked_atoms(entity_id)`（有效且非 entity 的 atom），`mentions = len(linked)`，`anchor_ids = sorted(∪ atom_anchor_ids(a) for a in linked)`；`existing` = 满足 `source_kind='entity', source_id=entity_id` 的有效 atom；不存在 → `create_atom(atom_from_entity(..., scope=Scope(user_id=e.user_id, agent_id=e.agent_id) if (e.user_id or e.agent_id) else Scope(run_id=e.run_id)), extra_mentions=[canonical_text], history_anchor_id=sched_anchor)`；已存在但 `content['mentions']` 或 `content['aliases']` 不同 → `revise_atom(old=existing, new=..., witness_anchor_ids=[sched_anchor], at=now, created_by='consolidation')`；其余跳过；计数 = 新建数 + 修订数；(1) `temporal_edges`：对每个 run 内按 `time_anchor` 相邻、且间隔在 `graph.temporal_window_hours` 之内的**每一对**有效 EP 节点调用 `insert_edge(temporal, witness=[later primary anchor], proposer='timestamp', created_by='consolidation')`（MERGE：已存在的边 `mentions += 1`）；计数 = 调用前尚不存在的边 id 数量（调用前 `find_edge` 为 None）；(2) `causal_edges`：候选对 = 时间相邻对 ∪ 共享标识符实体的对 ∪ `uses_ep_ids` 对；`causal_proposer == 'rule'` → 进程内调用 `hms_elite.llms.rule.propose_edges(nodes, pairs, witnesses)`（§11.4，节点形状与提示词一致）；`'llm'` → 以每批 ≤20 对的方式发起 `edge_proposal` 任务（见证者引用需解析；`witness_refs` 不是字符串列表、引用未知或为空 → 告警丢弃该提议、绝不抛错；`confidence` 无法解析或非有限 → 0.7）；每个被接受的提议 → `insert_edge(created_by='consolidation', proposer=llm.identity or 'rule')`（MERGE 会给已存在的边加 `mentions`）；计数只算新建的边 id；(3) `conflicts`：共享 `(subject, predicate)`、`predicate ∈ hms_elite.llms.rule.EXCLUSIVE`、objects 不相交、且彼此之间没有 `revision` 边的有效 `state` atom → `insert_edge(conflict, older → newer, witness newer primary anchor)` + 在较旧的 lineage 上写 `CONFLICT` history；当 `ingest.revision_mode == 'deferred'` 时，本步还会对同 subject 的配对运行 `decide_revision(new, old)`（§11.3），并通过 `revise_atom` 执行 REVISE（见证者 = 较新的 anchor，`at = max(new.valid_from, old.valid_from)`）；(4) `interface_eps`：对 `graph.components(scope)` 得到的每个集合（EP 节点，L1）→ `coarsen(..., target_scale='L2_meso')`（按 §14.2 可跳过、幂等）再 `project` → 计入 `projected_edges`；(4b) `interface_eps_l3`（当 `graph.coarsen_l3` 为真）：对 `graph.components(scope, kinds=('interface_ep',), edge_types=('causal','use','temporal'), scale='L2_meso', gap_hours=graph.coarsen_gap_hours * 4)` 得到的每个集合 → `coarsen(..., target_scale='L3_macro')` 再 `project`（其边界边就是 L2 IEP 投影出的边；高层边位于 `L3_macro`，投影计入 `projected_edges`）；(5) `skills`：`promote_skills(...)`；(6) `transitions`：`export_transitions(scope)` → 新增行数（含 InterfaceEP 行）；(7) `compiled_states`（当 `scheduler.compile_state` 为真；HMS §3.9 的 `compile_state`）：`ir = compiler.compile(goal, scope=scope, filters=scope.as_filters(), mode='deep')`，其中 `goal = last_ep.slots.u or 'work state'`；`store.insert_compiled_state(scope, as_of=ir.as_of, mode='deep', card=ir.to_text(), context_json=ir.model_dump_json(), used_tokens=ir.used_tokens, anchor_id=sched_anchor)`；`set_meta(f'work_state:{scope.key()}', canonical_json({'hash': sha256(card), 'as_of', 'used_tokens', 'compiled_id'}))`；每个 scope 计 1（`compile()` 本身保持只读；只有调度器负责持久化）。
- **prune**（`decay_stale_state`；HMS §3.9「衰减旧状态、归档低置信对象」——两个操作）：受检对象 = 满足 `atom_type ∈ {state, event, message}`、`obs_type ∉ {constraint}`、`source_kind ∉ {skill, entity}` 且不带 `important` 主题的有效 atom；对每个 atom：`recency = recency_score(time_anchor or created_at, as_of=now, half_life=scoring.half_life_days)`；`feedback_boost = clamp(confidence − 0.6, 0, 0.4)`；`use_recency = recency of the newest valid use/causal edge touching the node (0 if none)`；`staleness = (1 − recency)·(1 − feedback_boost)·(1 − use_recency)`；(a) **衰减**：当 `staleness > staleness_tau` 时：`new_conf = max(confidence_min / 2, round(confidence − scheduler.decay_rate · staleness, 4))`；若 `new_conf != confidence` → `set_atom_confidence(new_conf)` + 由 `sched_anchor` 见证的 history `UPDATE(old=new=text, details={'decay': round(staleness, 4), 'confidence': [old, new]})`，`decayed += 1`；(b) **归档**：衰减之后，当 `confidence < confidence_min` 且没有未解决的 `conflict` 边触及该节点时 → `archive_atom(reason='decayed')` + 由 `sched_anchor` 见证的 history `ARCHIVE`（若仅仅是冲突检查未通过，则 `skipped_conflict += 1`）；报告 `{'examined', 'decayed', 'archived': [atom_ids], 'live', 'expired', 'skipped_conflict', 'purged'}`；`purged` 恒为 0，除非 `retention.enabled` **且** `retention.run_in_prune` 同时为真——此时在 (b) 之后、于独立事务内调用 §16.6 的 `purge(store, index, scope=scope, older_than_days=retention.older_than_days, dry_run=False, keep_witnessed=True)` 并报告其 `purged`。
- 收尾：`set_meta(f"last_consolidation:{scope.key()}", now)`。

### §16.6 `ops/retention.py` —— 保留策略逃生舱（显式、默认关闭、有审计）

模块文档字符串：`"""Retention escape hatch: explicit, audited, default-off physical purge of long-archived atom versions (§16.6).\n保留策略逃生舱：显式、有审计、默认关闭的长期归档原子版本物理清理（§16.6）。"""` 这是**唯一**会物理删除行的路径，且默认从不走：修订代替删除仍是其它所有写操作的不变量，`BEFORE DELETE` 触发器对 `StateStore.purge_rows()` 之外的任何 DELETE 照常抛 `STORE_003`。

- `purge(store, index, *, scope: Scope, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict` —— 报告或删除再无存活对象需要的已归档版本 | 清理逃生舱。输入：至少一个 id 的范围（`Scope.matches`，只给 `user_id` 会扫过该用户的所有 run；空 → `VALIDATION_001`）、非负整数（否则 `VALIDATION_009`）、两个开关。输出：`{'dry_run', 'scope': scope.key(), 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable': [{'atom_id','memory_id','version','text_sha256','archived_reason','valid_to','source_kind','source_id','anchor_ids','bytes'}], 'protected': {原因: [ids]}, 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}`。状态影响：`dry_run=True` 不写任何东西；`dry_run=False` 是**一个**事务：`system_event(object_type='system', event_type='deleted', payload={'op': 'purge', 'metadata': {'older_than_days', 'cutoff', 'keep_witnessed', 'atoms'}})` → 每个版本一条由清理锚点见证的历史行 `PURGE(old=None, new=None, is_deleted=1, atom_id, details={'version','text_sha256','archived_reason','valid_to','source_kind','source_id','anchor_ids','cutoff','older_than_days'})` → `store.purge_rows(ids, evidence=not keep_witnessed)`；COMMIT 之后对每个被清原子 `index.delete(atom_id)`。说明：`cutoff = now − older_than_days 天`；`examined` = `store.purge_candidates(scope, valid_to_before=cutoff)`（archived_reason 非 NULL 且 valid_to < cutoff）。**保护**（任一命中即保留；`protected` 把每个 id 列在所有命中的原因下，顺序为 `head_version, edge_endpoint, checkpoint_live, interface_ep_support, ep_referenced, source_shared`）：(1) `head_version` —— 版本数 ≥ 2 的谱系的最高版本（它是修订链的终端记录；单版本墓碑没有链，可以清理）；(2) `edge_endpoint` —— 任何 `graph_edges` 行（有效或失效）的 src 或 dst（边携带见证、从不清理；这是"有效边端点"的超集，并通过 `revision` 边钉住每个被修订过的版本）；(3) `checkpoint_live` —— 任一检查点快照记录为 `archived_reason None`（回滚可能恢复它）；(4) `interface_ep_support` —— 被 `interface_eps.atom_id`、`primary_support_node_id` 或 `interface_ep_support` 的 node 成员引用；(5) `ep_referenced` —— 其 EP 行仍被其它 EP 的 `uses_ep_ids`、幸存原子的 `source_ep_ids` 或某条转移引用的 `ep` 原子；(6) `source_shared` —— 其 `(source_kind, source_id)` 也被清理集合之外的原子引用的原子（`store.shared_source_atom_ids`）：`purge_rows` 会删除原子自身的 `observations` / `eps` 行，而更新 / 反馈的后继（它复制 `source_kind` + `source_id`）仍需要它——不依赖 `revision` 边。同一共享规则约束 `archive_atom`：另一未归档原子仍引用来源时跳过来源级联，且 `revise_atom` 在创建与前驱共享来源的后继后调用 `store.reopen_source(new)`（`tests/test_ops.py::test_feedback_successor_keeps_shared_observation_live`）。`bytes` = utf-8 文本 + canonical 内容 JSON + 4·dims 的已存向量 + 256。**证据**：`keep_witnessed=True` 从不触碰锚点与 `source_events`；`False` 只额外删除被清行的锚点中、状态行删除后 `anchor_reference_count == 0` 的那些（history 行引用锚点，因此经过 `create_atom`/`delete` 的原子的锚点总会留下）及其 `source_events` 行。`graph_edges`、`edge_witnesses`、`history`、`feedback_events`、`checkpoints`、`transitions`、`compiled_states` 永不触碰（`feedback_events.target_id` / `history.atom_id` 可以悬空，这是设计——它们就是审计线索）。
- 触发器门：七个 `BEFORE DELETE` 触发器上的 `PURGE_GATE`（§7）；`purge_rows` 在自己的事务内写入 `meta.purge_in_progress='1'` 并在 COMMIT 前删除（ROLLBACK 同样丢弃）；`StateStore.__init__` 发现残留标志时告警并删除，手工改过的文件也不能让门开着。UPDATE（不可变）触发器从不设门。
- 门面（§17.3）：`HMS.purge(*, filters, older_than_days, dry_run=True, keep_witnessed=True)` 以 `require_scope=True` 解析 `filters`；`config.retention.enabled` 为 False 时 `dry_run=False` 抛 `ConfigError(CONFIG_001, details={'retention.enabled': False, 'dry_run': False})`；`HMS.vacuum()` = `store.vacuum()`。调度器钩子见 §16.5 prune。

---

## §17 门面（分组 H：`hms_elite/memory/main.py`、`__init__.py`、`migrate.py`、`cli.py`）

模块 docstring（`main.py`）：`"""HMS: the Mem0-compatible public API over the dual-plane store.\nHMS：双平面存储之上、与 Mem0 兼容的公共 API。"""`

### §17.1 构造

```python
class HMS(MemoryBase):
    def __init__(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None
        # config = HMSConfig.from_dict(config) if dict else (config or HMSConfig()); store = store or StateStore(config.store.path, busy_timeout_ms=config.store.busy_timeout_ms, journal_mode=config.store.journal_mode, checkpoint_on_close=config.store.checkpoint_on_close);
        # embedder = EmbedderFactory.create(config.embedder.provider, config.embedder.config); index = IndexFactory.create(config.vector_store.provider, config.vector_store.config, store=store, dims=embedder.dims);
        # llm = LlmFactory.create(config.llm.provider, config.llm.config); graph = SupportGraph(store, config.graph); ingestor = Ingestor(...); compiler = StateCompiler(...);
        # feedback_processor = FeedbackProcessor(...); scheduler = StateScheduler(..., compiler=compiler, config=config); reranker = RerankerFactory.create(config.reranker.provider, config.reranker.config, llm=self.llm，除非 config.reranker.config 自带 'llm' 信封) 当 config.reranker 不为 None，否则 None（search(rerank=True) 此时惰性构造一个默认 OverlapReranker，以 info 级别宣告一次）
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

上面注释的中文说明：构造顺序为 config → store → embedder/index → llm → graph → ingestor → compiler → feedback_processor → scheduler；`reranker` 恒为 None（配置了 `config.reranker` 时只告警一次）。列出的公开属性是测试访问被注入协作者的唯一途径。`from_config` 会记录完整错误，`ConfigError` 向上抛出；`project.update(...)` 抛 `ValueError`、`chat` 抛 `NotImplementedError`，与 Mem0 对齐。

### §17.2 Mem0 兼容接口（签名逐字照抄；`inspect.signature` 会被测试）

```python
def add(self, messages, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, metadata: dict | None = None,
        timestamp: Any = None, expiration_date: Any = None, infer: bool = True, memory_type: str | None = None, prompt: str | None = None) -> dict
def get(self, memory_id: str) -> dict | None
def get_all(self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs) -> dict
def search(self, query: str, *, top_k: int = 20, filters: dict | None = None, threshold: float = 0.1, rerank: bool = False, explain: bool = False,
           reference_date: Any = None, show_expired: bool = False, **kwargs) -> dict   # mvmi.search(..., scoring=config.scoring, retrieval=config.retrieval, ...)：所配置的 RetrievalConfig 到达每次 search，而不只是 compile()
def update(self, memory_id: str, text: str | None = None, metadata: dict | None = None, expiration_date: Any = _UNSET, data: str | None = None) -> dict
def delete(self, memory_id: str) -> dict
def delete_all(self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> dict
def history(self, memory_id: str) -> list[dict]
def reset(self) -> None
def close(self) -> None
```

| 方法 | 校验（错误码） | 语义 | 返回值 |
|---|---|---|---|
| `add` | `build_scope_and_metadata`（`VALIDATION_001`，id 去空白）；`memory_type ∉ {None, 'procedural_memory'}` → `VALIDATION_002`；`coerce_messages`（`VALIDATION_003`）；`timestamp` → `to_utc_iso`（`VALIDATION_004`），接受；`expiration_date` → `normalize_expiration_date` → `valid_to = next_day_midnight(date)` | `ingestor.ingest_messages(...)`；当且仅当 `agent_id and memory_type == 'procedural_memory'` 走 procedural 分支；`system` 角色的消息只锚定、不抽取 | `{'results': [row + 'event': 'ADD'|'UPDATE'|'NONE' (+ 'actor_id','role' when infer=False) (+ 'revises': <archived atom_id>) (+ 'conflicts_with': [memory_id])]}`；什么都没抽出来 → `{'results': []}`（事件仍然落库） |
| `get` | — | `store.resolve_head(memory_id)`；id 未知、已删除或全部版本均已归档时返回 None | 一行记录（§4.1）或 `None` |
| `get_all` | `reject_top_level_entity_params(kwargs, 'get_all')`（`VALIDATION_008`，ValueError）；`parse_filters` 至少需要一个 scope 键（`VALIDATION_001`）；`top_k ≥ 0` | `mvmi.get_all(...)`；`show_expired` → `include_expired=True`（连同 `valid_to` 展示已归档/已删除的版本） | `{'results': [row...]}`（无 `score`），排序 `time_anchor DESC, created_at DESC, atom_id ASC` |
| `search` | 同 `get_all`，外加 `validate_search_params`（`VALIDATION_009`）和 `validate_and_trim_query`（`VALIDATION_010`）；`reference_date` → `as_of`（`to_utc_iso`） | `mvmi.search(...)`；`rerank=True` → 融合后调用 `self.reranker.rerank(query, rows, top_k=top_k)`（未配置 → 默认 `OverlapReranker`，一条 info 日志）；各行按重排顺序返回，`rerank_score` 为行尾键；`explain=True` 时 `score_details` 追加 `rerank_score` 与 `reranker`（身份）；重排器任何异常都保持检索顺序并告警；`rerank=False` 路径不变 | `{'results': [row + 'score' (+ 'score_details' when explain) (+ 'rerank_score' when rerank)]}` |
| `update` | 传 `data=` → `logger.warning`，它是 `text` 的废弃别名；text/metadata/expiration_date 一个都没给 → `VALIDATION_011`（ValueError）；id 未知或已删除 → `NOT_FOUND_001`（`ValueError("Memory with id X not found")`） | 单事务：`system_event(object_type='system', event_type='updated', payload={'op': 'update', ...})`；给了 text → `revise_atom(old=head, new=copy with text, hash, re-linked entities/topics, user_meta merged (identity keys stripped, warn only if changed), anchor_ids=[sys_anchor]+old.anchor_ids, valid_to per expiration_date, created_by='ops')` + history `UPDATE`；只给 metadata → 在 head 上 `set_user_meta`（不产生新版本；与 Mem0 对齐：metadata 不是状态）+ history `UPDATE(old=new=text, details={'metadata_only': True})`；`expiration_date` 显式传 `None` → `set_valid_to(NULL)`；传日期 → `set_valid_to(next_day_midnight)` | `{'message': 'Memory updated successfully!'}` |
| `delete` | id 未知或已删除 → `NOT_FOUND_001` | `system_event(event_type='deleted')`；`archive_atom(head, reason='deleted', at=now)`；history `DELETE(old=text, new=None, is_deleted=1)`；边保留 | `{'message': 'Memory deleted successfully!'}` |
| `delete_all` | 三个 id 一个都没给 → `ValidationError(VALIDATION_001, ValueError)`，消息为 `'At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method.'` | 单事务、单个系统事件；scope 内每个有效 head 都被归档（`deleted`）并写一条 `DELETE` history | `{'message': 'Memories deleted successfully!'}` |
| `history` | — | `store.get_history(memory_id)` | 按 `seq` 顺序返回裸列表 `{'id','memory_id','old_memory','new_memory','event','created_at','updated_at','is_deleted': bool,'actor_id','role','anchor_id'}`；id 未知 → `[]`（已记录的对齐差异：多出 `anchor_id`，`event` 多出 `ARCHIVE|RESTORE|FEEDBACK|CONFLICT|ROLLBACK` 几种取值） |
| `reset` | — | `store.reset()`；`index.reset()` | `None` |
| `close` | — | 先从 `store.attr_listeners` 移除 `self.index.update_attrs`（共享 store 不得继续调用已关闭的提供者），再 `index.close()`（幂等；milvus 释放其客户端），再在 `finally` 里 `store.close()`（两者幂等） | `None` |

### §17.3 HMS 原生接口

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
def purge(self, *, filters: dict, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict   # §16.6 逃生舱：parse_filters(require_scope=True)；dry_run=False 需要 config.retention.enabled（否则 CONFIG_001）；→ ops.retention.purge 的报告
def vacuum(self) -> dict                                               # store.vacuum()：在任何事务之外执行 VACUUM → {'path','before_bytes','after_bytes','freed_bytes'}
```

每一个公开写操作都是单个 `store.transaction()`（`add`/`add_events` 使用 Txn A + Txn B，中间夹一个不持锁阶段）；每一个公开读操作只调用一次 `store.now()` 取 `as_of`。

### §17.4 `hms_elite/__init__.py`

`from hms_elite.memory.main import HMS`；`from hms_elite.configs.base import HMSConfig`；`__version__ = importlib.metadata.version("hms-elite")`（回退为 `"0.0.0"`）；`__all__ = ["HMS", "HMSConfig", "__version__"]`。不产生任何其他副作用。

### §17.5 `hms_elite/migrate.py`

- `from_mem0(memory: Any, hms: HMS, *, filters: dict, page_size: int = 100) -> dict` —— 遍历 `memory.get_all(filters=filters, top_k=page_size)`（并带 `show_expired=True`），对每一行调用 `hms.add(row['memory'], infer=False, timestamp=row.get('created_at'), metadata=row.get('metadata') or None, user_id=..., agent_id=..., run_id=..., expiration_date=row.get('expiration_date'))` 重新写入，返回 `{'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}` | 从 Mem0 迁移。状态影响：通过 `hms.add` 写入。

### §17.6 `hms_elite/cli.py`（≤ 90 行，`argparse`）

`hms add <text> --user U [--agent A] [--run R]`、`hms search <query> --user U [--top-k N]`、`hms compile <query> --user U [--run R] [--mode auto|fast|deep]`、`hms why <id-or-question> --user U [--run R]`、`hms transitions --user U [--run R] [--fmt jsonl]`、`hms consolidate --user U`、`hms stats --user U`；环境变量 `HMS_DIR`；输出 JSON（`ensure_ascii=False`）或卡片文本；入口点 `hms = "hms_elite.cli:main"`。命令抛出的 `HMSError` 以 `error.to_dict()` JSON（`ensure_ascii=False`）打印到 stderr 并以退出码 1 退出——回溯永远不是 CLI 的错误界面。README §12 在 `hms why … --run run-billing` 之前先用 `hms add "Do not modify the subscription state machine." … --run run-billing` 给 `run-billing` 灌入一条约束，因为问题模式的 `why`（§17.3）要求范围内至少有一条约束 / 片段 / 事实。

### §17.7 `hms_elite/server/` —— HTTP 服务面（H 组；可选 extra `server`；`tests/test_server.py`）

- `create_app(config: HMSConfig | dict | None = None, *, hms: HMS | None = None) -> FastAPI`（惰性导入 `fastapi` → `DependencyError(DEPS_001, extra='server')`）；`main(argv)` 运行 uvicorn（`hms-server`）。注入的 `hms` 留给其所有者；由 `create_app` 自建的门面由 lifespan 关闭。
- 路由表（`_ROUTES`，括号内为操作标签）：`POST /memories`（add）、`GET /memories`（get_all）、`GET /memories/{memory_id}`（get）、`POST /search`（search）、`PUT /memories/{memory_id}`（update）、`GET /memories/{memory_id}/history`（history）、`DELETE /memories/{memory_id}`（delete）、`DELETE /memories`（delete_all）、`POST /reset`（reset）、`POST /events`（add_events）、`POST /compile`（compile）、`GET /why/{target}`（why）、`POST /feedback`（feedback）、`POST /checkpoint`（checkpoint）、`POST /rollback`（rollback）、`POST /handoff`（handoff）、`GET /transitions`（transitions）、`POST /consolidate`（consolidate）、`GET /stats`（graph_stats）、`GET /usage`（usage）、`GET /metrics`（metrics）；另有 `GET /health`（存活、版本、提供者名）。
- **鉴权**：`$HMS_SERVER_TOKEN` 已设置且非空 → 表中每条路由都要求 `Authorization: Bearer <token>`（`deps.require_bearer`；缺失 / 错误 → `AuthError(AUTH_001)` → 401）；未设置或为空 → 接受每个请求。`GET /health` 注册在鉴权路由器之外，从不检查令牌。
- **错误信封**（`error_response`）：每个被映射的错误都是 `HMSError.to_dict()` = `{error_type, error_code, message, details, suggestion, debug_info}`；`error_status`：`AuthError` → 401，`error_code` 以 `NOT_FOUND` 开头 → 404，`ValidationError` → 400，其余 `HMSError` → 500；门面内部抛出的裸 `ValueError` / pydantic 错误 → `ValidationError(VALIDATION_000, details={'exception': <类型>})` → 400；其余任何异常 → `HMSError('internal error', INTERNAL_000)` → 500。**500 的响应体被脱敏**：保留 `error_type` 与 `error_code`，`message` 为 `'internal error'`，`details` / `debug_info` 为 `{}`，`suggestion` 为 `None`（`StoreError` 的消息 / 细节里带 SQLite 路径与 SQL 片段；它们只以 ERROR 级别带 `exc_info` 与请求 id 记在服务端日志里，绝不发出）。FastAPI 自身对格式错误请求体的 422 保持不变。兜底捕获位于 `RequestTelemetry`（纯 ASGI 中间件），因此 INTERNAL_000 响应与其它响应一样带 `X-Request-ID`，追踪也记录该错误；已经开始流式输出的响应不会被替换。
- **遥测**：每个响应回写（或生成）`X-Request-ID`；每个请求是一条 `telemetry.trace(operation, request_id)`，观测进 `RequestMetrics`（`GET /usage`、`GET /metrics`）。

### §17.8 `hms_elite/memory/async_main.py` —— `AsyncHMS`；`hms_elite/telemetry.py` —— 计量（H / A 组；`tests/test_async.py`、`tests/test_telemetry.py`）

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

块内英文注释对照：`aclose` 幂等，`async with AsyncHMS(...) as hms:` 会调用它；`sync / config / store / project` 是同步属性；协程方法与 `HMS` 同名同签名（`inspect.signature` 逐一比对），每个都是 `await asyncio.to_thread(<同步方法>, ...)`；`chat` 与 `HMS.chat` 一样抛 `NotImplementedError`；`usage()` / `trace()` 是同步附加项。

- 规则：绝不阻塞事件循环（`test_async_does_not_block_the_event_loop`）；并发的 `add` + `search` 在 store 锁上串行，返回与同步门面相同的结果（`test_async_concurrent_add_and_search`、`test_async_results_equal_sync_results`）；每个参数原样转发（`test_async_forwards_every_argument_verbatim`）；离线默认、错误码与信封原样透传；`close()` / `aclose()` 幂等。
- `telemetry`：`Usage(llm_calls, input_tokens, output_tokens, total_tokens, estimated, embedding_calls, embedded_texts, duration_ms)`，带 `merge` / `to_dict`；`UsageMeter`（`add_llm`、`add_embedding`、`snapshot`、`reset`；加锁——`test_concurrent_calls_lose_no_counts`）；`MeteredProvider`（每个 LLM / 嵌入器基类携带的 `usage` 计量器：`openai` / `anthropic` 记录 SDK 的精确 usage，`rule` / `scripted` 经 `memory.utils` 估算并置 `estimated`）；`trace(operation, request_id=None)`（contextvar 局部的 `RequestTrace`，收集 `stage(...)` / `record_stage(...)` 产生的 `StageRecord`；`current_trace()`）；`usage_of(hms) -> Usage`（llm ∪ embedder 计量器）。服务面的 `GET /usage` 返回 `usage_of(hms).to_dict()`，`GET /metrics` 渲染 `RequestMetrics`（§17.7）。

---

## §18 测试（分组 T：`tests/`）—— pytest，完全离线，结果确定

### §18.1 `tests/conftest.py`（唯一的 conftest）

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

上面注释的中文说明：`FrozenClock` 不会自动推进，只能 `.tick()`；`golden` 提供两个由调用方选择的断言助手——`golden.text` 逐字节比对、不做掩码（Work State Card、`why_chain.txt` 含 fixture 时间戳且不含 id）；`golden.masked` 先经 `memory.utils.mask_ids`，并**只**掩码 `>= FrozenClock.start` 的时刻（`"2026-09-02…"` → `"<ts>"`），fixture 里 2026-03 至 2026-06 的时刻保持原样；两个助手都支持 `--update-golden` 重写文件。

规则（来自 Mem0 的 `tests/AGENTS.md`）：mock 提供方 SDK，绝不 mock 被测代码；每个缺陷修复都要带一个没有修复就会失败的回归测试；只用 `monkeypatch.setenv`；不做时间相关断言；不联网。

### §18.2 测试夹具（Fixtures）

`tests/fixtures/events/`：
- `chat_preference.json` —— S1 消息（§19.1）
- `chat_conflict.json` —— `[{"role":"user","content":"I live in Paris.","timestamp":"2026-05-01T09:00:00Z"}, {"role":"user","content":"I live in Berlin.","timestamp":"2026-04-01T09:00:00Z"}]`（第二条在第一条之后加入）→ CONFLICT
- `chat_multisession.json` —— S2/S3 消息（§19.2）
- `chat_action_closure.json` —— `[{"role":"assistant","content":"I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."}]`
- `chat_habit_temporal.json` —— `[{"role":"user","content":"Yesterday I visited the Berlin office.","timestamp":"2026-05-10T09:00:00Z"}, {"role":"user","content":"I usually run every morning."}]`
- `chat_zh.json` —— `[{"role":"user","content":"我喜欢燕麦拿铁。","timestamp":"2026-04-02T08:00:00Z"}, {"role":"user","content":"我现在改喝黑咖啡了。","timestamp":"2026-04-20T08:00:00Z"}]`
- `structured_billing.json` —— S4 事件（§19.3）
- `mem0_config.json` —— `{"llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini", "temperature": 0.1, "top_p": 0.1, "top_k": 1}}, "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}, "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}, "history_db_path": "/tmp/x/history.db", "version": "v1.1", "custom_instructions": null}`

`tests/fixtures/golden/`：`observations_chat_preference.json`、`observations_chat_multisession.json`、`observations_chat_zh.json`、`eps_chat_action_closure.json`、`revision_cases.json`、`work_state_card_failed.txt`、`work_state_card_fixed.txt`、`why_chain.txt`、`transitions_s6.jsonl`、`search_explain_keys.json`。

### §18.3 分组具名测试（每个测试都点名它所覆盖的公开函数）

**A —— `test_foundation.py`**
1. `test_import_topology` —— 模块导入图符合 §0.4（`mvmi/*` 永不导入 graph/compiler/ops/ingest；`models` 在模块层不导入 compiler）。
2. `test_enums_are_str_enums_and_values` —— §2 中的每个枚举都是 `(str, Enum)`，成员集合完全一致；`FeedbackType.state_wrong`、`EdgeType.causal` 存在。
3. `test_witnessed_models_reject_empty_anchor_ids` —— `Observation/EP/MemoryAtom/GraphNode/GraphEdge/InterfaceEP/ContextItem/Transition` 传 `[]` 时抛 pydantic 错误；`Slots` 的闭包校验器生效。
4. `test_memory_atom_rejects_edge_keys_and_text_mismatch` —— `content={'edges': []}` 与 `content['text'] != text` 均被拒绝。
5. `test_contextir_rejects_duplicate_item_ids_across_slots`。
6. `test_timeutil_fixed_width_and_parsers` —— `to_utc_iso` 处理 `Z`、`+05:30`、无时区、纯日期、datetime、epoch → 均为 32 字符；`next_day_midnight`、`month_window`、`age_days`。
7. `test_text_tokenize_entities_topics_dates` —— `tokenize('patch retry.ts and billing_e2e')` 保留标识符；`extract_entities('Sarah joined the billing team in March.')` → 只得到专名 Sarah；`extract_dates('what happened in May', base_date='2026-09-02')` → 2026 年 5 月的窗口；`'retry.ts may change'` → 不产生日期；中文 `'5月发生了什么'`。
8. `test_parse_llm_json_ladder` —— 围栏代码块、被散文包裹、顶层数组抢救、垃圾输入 → `[]` 并在 `caplog` 里记 error。
9. `test_prompt_builders_round_trip` —— 对每个 builder，`parse_sections(build_x(...))` 能还原输入；每个 `*_PROMPT` 的首行都是 `HMS_TASK: <task>`；六个 JSON 提示词（`ROOT_KEY` 中的任务：五个分段 JSON 任务加 `observation_extraction_compact`）的 OUTPUT FORMAT 段可解析为 JSON 且带 §6.3 的根键；`PROCEDURAL_SUMMARY_PROMPT` 的 OUTPUT FORMAT 段按顺序包含 §6.3 的七个 markdown 标题/标签，以及 `Return ONLY the markdown above. No JSON, no wrappers.` 这一行；`build_edge_prompt` 的节点携带 `u,b,a,y,r,subject,predicate,objects` 全部字段。
10. `test_config_mem0_dict_and_negative_cases` —— `HMSConfig.from_dict(mem0_config.json)`（**未经修改**的 fixture dict）**只**因 `qdrant` 而失败于 `CONFIG_002`（由纯 Python 预检抛出，§5.3）；同一个 dict 把 `vector_store.provider` 改成 `'sqlite'` 后可构造，且 `store.path == history_db_path`；被重命名的键触发 `DeprecationWarning` 并重映射；被移除的键 → `CONFIG_003`；未知键 → `CONFIG_001` 并指出键名；伪造的 provider 无论带不带 `config` → 同样是 `CONFIG_002`；`config: None → {}`；`HMSConfig()` 在没有任何可选依赖时也能构造；每个 Literal 都拒绝非法取值。
11. `test_schema_migrations_and_capabilities` —— 全新 DB 的 `user_version == 3`（`SCHEMA_VERSION`；`MIGRATIONS == {1: build_ddl(1), 2: upgrade_2(), 3: upgrade_3()}`，`build_ddl() == build_ddl(3)`，`'PURGE'` 从版本 2 起出现、`FTS_DDL` 从版本 3 起出现），22 张基表 `TABLES` + 1 张 `VIRTUAL_TABLES`（+ 5 张 FTS 影子表）全部存在（含 `compiled_states`），重新打开幂等，`user_version` 更高 → `STORE_002`；`capabilities()` 含 `json1`、`fts5`、`sqlite_version`；`PRAGMA table_info(history)` 以 Mem0 的 10 个列开头且顺序一致；`source_events.role` 的 CHECK 排除 `'extracted'`；`ux_atoms_head` 拒绝同一 lineage 内出现第二个未归档版本。
11b. `test_purge_gate_and_schema_migration_2` —— 七个 `PURGE_GATED_TRIGGERS` 带 `PURGE_GATE`，其余 15 个触发器没有 `WHEN`；无标志时裸 DELETE 为 `STORE_003`、有标志时成功；`purge_rows` 随行一起移除标志；文件中的残留标志在打开时被清除；版本 1 的文件（`build_ddl(1)`，裸插入行，其 CHECK 拒绝 `'PURGE'`，DELETE 被拦）以版本 2 打开，`sqlite_master`（type, name, tbl_name, sql）与全新 store 相同，历史行完好，`PURGE` 行现在可写。
12. `test_store_invariants_at_ddl` —— 不带 `primary_anchor_id` 的裸 INSERT → `IntegrityError`；对 anchors/atoms/edges/eps/history 做内容 DELETE/UPDATE → `StoreError(STORE_003)`；`set_valid_to`/`set_atom_confidence`/`set_edge_label` 可成功；`graph.insert_edge(src_id=..., dst_id=..., edge_type='use', witness_anchor_ids=[], created_by='ingest')`、`store.insert_edge({'src_id': ..., 'dst_id': ..., 'edge_type': 'use', 'witness_anchor_ids': [], 'created_by': 'ingest'})` 以及 `store.insert_edge(GraphEdge.model_construct(witness_anchor_ids=[], ...))` 都在任何 SQL 执行前抛出 `WitnessMissingError`（`trace.reset()` 之后 `store.trace.tables() == set()`）；未知 anchor id → `WitnessMissingError(details['missing'])`；没有 `witness_edge_id` 的边界边违反 CHECK；带上它之后 `edge_projections` 会出现对应行（触发器）；`insert_edge` 的 MERGE 会累加 `mentions` 并合并见证者；对主 anchor 调用 `add_atom_anchor` 返回 False 且不新增行。
13. `test_store_transaction_rollback_and_trace` —— `transaction()` 内抛异常后不留下任何行；`trace.tables()` 列出被触及的表；事务可重入嵌套。
14. `test_store_lineage_helpers` —— `insert_atom` 分配 memory_id/version；`resolve_head`、`list_versions`、`archive_atom`、`restore_atom`；`ux_atoms_lineage_version` 拒绝重复版本号。
15. `test_store_entities_keyed_without_run_id` —— 在两个 run_id 下对 `sarah` 调用 `get_or_create_entity` 返回同一个 id；别名会被追加。

**B —— `test_providers.py`**
16. `test_hash_embedder_deterministic_normalized` —— 同一文本跨实例得到完全相同的向量，`‖v‖₂ ≈ 1`，维度 64，`sim('oat latte','oat-milk latte') > sim('oat latte','billing test')`，中日韩文输入结果非零。
17. `test_index_parity_sqlite_memory` —— **不**参数化：在同一个测试里同时构造 `SQLiteIndex(store=store, dims=64)` 和 `MemoryIndex(dims=64)`；相同向量下 `search` 的排序与分数完全一致；`candidate_ids` 限定；`reset`。
18. `test_rule_llm_observation_golden` —— `RuleLLM` 在 `chat_preference`/`chat_multisession`/`chat_zh` 提示词上的输出 == golden JSON（types/predicates/objects/text/relation 与 §19 完全一致）。
18b. `test_moved_to_yields_lives_in_supersession` —— `'I moved to Berlin.'`／`'I relocated to Berlin.'`／`'我搬到柏林。'` 抽取为 R5 的 `fact`/`lives_in`/`supersession`，objects 为 `['Berlin']`／`['柏林']`，confidence `0.85`（绝不落到 R1 的 `likes`/`preference_event`）；`decide_revision(<berlin>, <paris>)` → `(REVISE, 'supersession cue')`。
19. `test_rule_llm_ep_and_revision_and_edge_and_coarsen_contracts` —— 输出满足 §6.3 的各项契约；`chat_action_closure` 产出一个 EP（`a='patched retry.ts to add backoff'`、`u='the webhook kept timing out'`、`y='billing_e2e failed'`、`r=negative`）== golden `eps_chat_action_closure.json`（用 masked 助手）；`revision_cases.json` 表（每行：new、old → decision）经公开的 `decide_revision` 验证与 §11.3 一致；`propose_edges` 对 S4 的 EP 对提出 `causal`；`coarsen_rule` 作用于两个 `interface_ep` 类型的节点时返回一个 L3 `interface_ep` dict（P 非空）。
20. `test_rule_llm_unknown_task_and_raise_hook` —— 未知标记 → `LLMError(LLM_002)`；`RAISE:` 行 → `LLMError(LLM_001)`。
21. `test_scripted_llm_queue_and_cause` —— 支持 dict/str/Exception 三种响应；队列耗尽 → `LLM_003`；`LLMBase.generate_json` 包装为 `LLMError` 并带 `__cause__`。
22. `test_optional_providers_guarded` —— `patch.dict(sys.modules, {'openai': None})` → `LlmFactory.create('openai')` 抛 `DependencyError(DEPS_001)` 并附 pip 提示；换成伪造的 `openai` 模块时，provider 会传 `response_format={'type': 'json_object'}`；`anthropic`（system 分离）与 `OpenAIEmbedder` 同理。
23. `test_factory_registry_single_source_of_truth` —— `register_provider('llm', 'x', ...)` 之后 `LlmConfig(provider='x')` 校验通过且 `LlmFactory.create('x')` 可用；调用方传入的 dict 不被修改。

**C —— `test_ingest.py`**
24. `test_one_event_one_anchor_and_dedup_at_same_event_time` —— N 条消息 → N 个事件/anchor；用相同的 `timestamp=` 原样重新加入 → `dedup=True`，不产生新 anchor，行的 `event='NONE'`，`store.atom_anchor_ids(atom) == [<anc_1>]` 保持不变（该 anchor 本就是主见证者；`atom_anchor` 行数不变），且 `graph_nodes.mentions == 2`（`bump_mentions`）；相同文本换一个更晚的 `timestamp=` → 产生新事件 + 新 anchor，行的 `event='NONE'`，且此时已存在的 atom 为新 anchor 增加一条 `support` 型 `atom_anchor` 行（`atom_anchor_ids == [<anc_1>, <anc_2>]`）。
25. `test_payload_normalization_and_internal_types_rejected` —— §4.3 的各种形状；`failed_tests` 别名；`status` 推导；`add_events([{'object_type': 'feedback', ...}])` → `VALIDATION_007`；格式错误 → `VALIDATION_005` 并带 `index`。
26. `test_router_observation_vs_state` —— `chat_preference` → 只产生 observation；`chat_action_closure` → 同时产生 observation 和 EP；`system` 角色 → 只锚定、不抽取。
27. `test_structured_closure_builds_eps_without_llm` —— 前置准备**不**调用任何 LLM：`(ev, anc, _), = ingest.events.anchor_events(store, events_from_messages([{'role': 'user', 'content': 'Do not modify the subscription state machine.'}], scope=scope4, event_time='2026-06-15T09:00:00Z', user_meta={}, visibility='private'))`，然后 `ingest.atoms.create_atom(store, index, embedder, atom_from_observation(Observation(type='constraint', subject='project', predicate='must_not', objects=['subscription state machine'], relation='constraint', text='Do not modify the subscription state machine.', date_observed='2026-06-15', anchor_ids=[anc.anchor_id], scope=scope4), scope=scope4))`（背景约束）；`hms_scripted.llm.calls == []` **只**在 `hms_scripted.add_events(structured_billing.events_fail)` 前后断言 → 得到 EP1（`u`、`a`、`y`、`r`、`anchor_ids=[outcome, action]`、`kappa.evidence == anchor_ids`）、带实体 `retry.ts` 的派生 constraint atom、指向背景约束的 `use` 边（见证者 = action anchor）、`distill` 边（见证者 = outcome anchor）；第二批 → EP2，并跨批次产生 EP1→EP2 的 `temporal` + `causal` 边（借助 `store.last_ep`/`last_open_action`）；**较晚**批次里的 `test_log` 能闭合**较早**批次里的 `file_diff`（`last_open_action`），但超过 `closure_max_gap_hours` 的则不能；`tool_call status=error` 自闭合；payload 带 `constraints` 的未闭合 `tool_call` 自闭合，`y=''`、`r='unknown'`、`kappa.constraints == payload constraints`；没有未闭合动作的 `test_log` → 只产生 event atom；`sensor_trace` 事件 → 只产生 event atom，`event_type == 'tool_result'`，`raw_ref == 'sensor://<sensor>'`。
28. `test_infer_false_raw_atoms_link_entities` —— 行携带 `actor_id`/`role`；`atom_entity` 行存在；history 为 `ADD`。
29. `test_procedural_branch_is_l1_episode` —— `add(..., agent_id, memory_type='procedural_memory')` → 一个 `scale_level='L1_micro'` 的 atom，`memory_type` 被提升；`prompt=` 会替换系统提示词（用 scripted 断言）。
30. `test_revision_inline_deferred_off` —— 对 `chat_preference` 分别取三种 `ingest.revision_mode`：inline → UPDATE；deferred → 在 `consolidate()` 解决之前保持两个有效 atom；off → 保持两个有效 atom。
30b. `test_relocation_revises_previous_residence` —— 同一 scope 内先 `'I live in Paris.'` 再 `'I moved to Berlin.'` → 第二行为 `UPDATE`，`predicate='lives_in'`、`version == 2`，Paris 原子 `archived_reason='revised'`，产生一条 Paris → Berlin 的 `revision` 边，最终只剩一个有效原子。
31. `test_lock_discipline_llm_runs_unlocked` —— 给 `ScriptedLLM` 排入一个可调用响应（`lambda messages: {'observations': []}`），它在被调用时断言 `store.in_transaction is False`。
32. `test_create_atom_rolls_back_on_embedding_error` —— embedder 在第二段文本上抛错 → 两个平面的任何表都不留下半成品行。

**D —— `test_mvmi.py`**
33. `test_filter_dsl_conformance[index_provider]` —— 14 个用例（标量 eq、列表 in、`'*'`、`created_at` 上的 `{gte}`、纯日期字面量的 `{lte}`、metadata 键上的 `{icontains}`、`meta.key`、`OR`、`NOT`、嵌套 `AND(OR)`、实体轴、主题轴、`anchor_id`、未知操作符 → `ValueError`），在 `lower_to_sql` 和 `evaluate` 两条路径上结果一致；通过 monkeypatch capabilities 走 json1 回退路径。
34. `test_plan_query_routes_and_gates` —— `"what happened in May"` → temporal + 五月窗口；`"5月发生了什么"` 同上；`"everything about Sarah"` → entity + 硬闸门；`"why must retry.ts not be modified"` → detective，标识符为 `retry.ts`；`"retry.ts may change"` → 不产生时间窗口；`"what coffee does the user like"` → 路由 `preference`，`hard_type_gate == ('obs_type', ['preference_event', 'habit'])`，`scale_default == ['state']`。
35. `test_axis_query_seven_axes_and_aggregations` —— 预置数据的 store：时间/有效性(as_of)/实体/主题/角色/情感/尺度各闸门；随后按 HMS §3.5，通过 `hms.axis_query(...)` → `AxisResult`：`project_axis='value', aggregation='latest'` → 每个 `(subject, predicate)` 得到 `projection == [[<latest objects>]]`；`project_axis='time'` → 按 `(time_anchor, atom_id)` 升序的时间线；`project_axis='entity'` → `{key: [atom_ids]}`；`project_axis='topic'` → `{topic: [atom_ids]}`；`aggregation='count'` → `count == n`、`hits == []`，且 trace 只显示一次 `memory_atoms` 的 COUNT；`aggregation='summary'` → `summary == f"{n} atoms; latest: {text}"`；`aggregation='topk'` 且 `text=None` → `VALIDATION_013`；`AxisQuery(filters={'user_id': U, ...})` 不带 `scope` 也能校验通过并执行；`consolidate()` 之后 `scale_constraint=['event','entity','L1_micro']` 会返回 Sarah 的 `entity` atom；`filters={'visibility': 'team'}` 只返回 team+event+public 的条目（格闸门）。
36. `test_search_fusion_explain_threshold_after_fusion` —— `explain=True` 的键与 §13.4 完全一致（`axis_gates` 含 `obs_type`）；语义分很弱但命中标识符的结果能通过 `threshold=0.3`；度数 ≤ 3 的实体 `entity_boost == 1.0`，度数为 5 时 `< 1.0`；`max_possible_score` 反映当前激活的信号；两次调用字节级一致；preference 路由施加 `obs_type` 闸门（`primary_axis == 'type'`），无结果时回退（`'fallback'`）；`search` 不触及任何 `graph_*` 表（`store.trace`）。
37. `test_search_hard_time_gate_and_fallback` —— 五月查询只返回五月的 atom；窗口内什么都匹配不到的查询会回退，`primary_axis='fallback'`。
38. `test_get_all_show_expired_and_order`。
39. `test_atom_to_row_shape_and_no_metadata_leak` —— 键顺序符合 §4.1；`metadata` 只含调用方自己的键；`RESERVED_ATOM_KEYS ⊇ set(PRAGMA table_info(memory_atoms) names)`（含 `source_ep_ids_json`）；`mvmi.query.atom_to_row(atom) == atom.to_public_dict()`。

**E —— `test_graph.py`**
40. `test_cohesion_formula_values` —— 三节点链且无边界 → 1.0；孤立节点 → 0.0；§19 的 S = {EP1, EP2} → `pytest.approx(2/3, abs=1e-6)`（`cohesion_score` 存储时不做舍入）。
41. `test_coarsen_project_refine_roundtrip` —— 在 S4 图上：IEP 的 `cohesion_score == pytest.approx(2/3, abs=1e-4)`，slots 符合 §11.5（`r == 'positive'`，`b` 里每条约束文本恰好出现一次），`slots.kappa.evidence == evidence_anchor_ids`，不变量 `"Do not modify the subscription state machine."` 的 `support_node_ids == [<ep1_atom>, <ep2_atom>]`，`anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`，`confidence == 0.8`；`failed_path` 不变量的 `anchor_ids == sorted({<anc_e1>, <anc_e2>})`；IEP atom 的 `source_ep_ids == [<ep1>, <ep2>]`，`topic_tags == ordered union of the support atoms' topics`，实体键 ⊇ `{'retry.ts', 'idempotency.ts', 'billing_e2e'}`；`evidence_anchor_ids == ∪ support anchors ∪ witnesses(E_in)`；`project` 写出 3 条边界边，每条都有 `edge_projections`（共 4 行）；`refine(project(coarsen(S)))['boundary_edges']` 的 id == `boundary(S)` 的 id；重复 coarsen 幂等（`support_hash`）。
42. `test_interface_ep_is_not_a_summary` —— 只有 1 个支撑节点或 evidence 为空的 `InterfaceEP` 被拒绝；每个 slot 都引用支撑 id；EP 与 IEP 使用同一个 `Slots` 类；不带 `anchor_ids` 的 `Invariant` 被拒绝，`Residual.anchor_ids` 默认为 `[]`。
43. `test_promote_skills_from_one_interface_ep` —— 支撑节点 ≥2 的不变量 → 生成 `L3_macro` 的 skill atom（`anchor_ids == inv.anchor_ids`、`confidence == inv.confidence`）+ 一条 `L3_macro` 的 `distill` 边，`is_boundary=0`，`witness_edge_id == <lowest-id valid use edge support_node → constraint atom>`，且 `store.projections_of(distill_edge) == [both use edges]`（触发器 + `add_projection`）；`graph.stats(scope)['traceable_high_edges'] == 1.0`；第二次运行幂等。
44. `test_explain_backward_bfs_and_chain` —— 对派生约束调用 `why()` 的核心逻辑：nodes/edges/paths/anchors 符合 §19.5；`render_chain` == golden `why_chain.txt`；`hops=0` 只返回目标本身。
45. `test_refine_called_only_by_explain_and_compile` —— 源码检查：`refine(` 这个 token 只出现在 `graph/operators.py`（定义处）、`graph/explain.py` 和 `compiler/compile.py`（handoff 经由 compile 调用）；`compile(mode='fast')` 的 trace 不含 `graph_*`/`interface_ep*` 表；`compile(mode='axis_graph')` 的 trace 触及 `graph_*` 但不触及 `interface_ep*` 表。
45b. `test_coarsen_l3_from_two_interface_eps` —— 两个 L2 IEP（同一 scope 内两个类 S4 的 run，由一条投影出的 `causal`/`use` 边相连）→ `consolidate()` 报告 `interface_eps_l3 == 1`；该 L3 IEP 的 `support_node_ids` 就是那两个 L2 IEP atom，`scale == 'L3_macro'`，`refine(hops=0)['nodes']` 恰好返回这两个节点；`coarsen_l3=False` → 0。

45c. `test_coarsen_llm_policy_validates_enums` —— policy 为 `llm`，提案引用有效但 `slots.r == 'great'`、`kappa.visibility == 'everyone'` → 不抛异常：`coarsen_op_version == 'llm-coarsen-v1'`，`slots.r` 与 `slot_support['r']` 取规则值，`kappa.visibility == 'private'`（规则值）而 LLM 的约束与 `slot_support['kappa']` 保留；各一条告警；`round_trip_check` 成立。

**F —— `test_compiler.py`**
46. `test_slot_rules_total_and_disjoint` —— 每种 atom 都恰好落入一个 slot；带**未来** `valid_to`（`expiration_date`）的存活 atom 落在它本该在的 slot，已过期的（`valid_to <= as_of`）落在 `audit` 且 reason 为 `archived:expired`；kappa 条目要针对已选约束**以及** audit 条目去重，并跨 episode 去重（每段文本只保留一个 κ 条目）；被 episode 覆盖的 event atom 被丢弃；`compile()` 传 `mode=None` 时使用 `config.compiler.default_mode`（`default_mode='fast'` → `retrieval['ablation'] == 'axis'`）。
47. `test_work_state_card_failed_golden` —— 在约束 + `events_fail` 之后：`compile("fix checkout webhook retry", filters=scope4, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()` == `work_state_card_failed.txt`（§19.3，用 `golden.text`）；slot 之间互不相交；每个条目都有见证；`used_tokens ≤ 800`；`retrieval['dropped_covered'] == 2`；行数不变。
48. `test_work_state_card_fixed_golden` —— 在 `events_fix` 之后：== `work_state_card_fixed.txt`；`retrieval['dropped_covered'] == 4`；归一化 `Mode:` 头之后，`mode='deep'` 与 `mode='axis_graph'` 产出同一张卡片。
49. `test_budget_shares_min_constraints_and_spill` —— 极小预算下仍保留 ≥3 条约束；facts 未用完的份额溢出给 episodes。
50. `test_compile_modes` —— `fast` 绝不触及图表；`deep` 在 S1 的 scope 上会把已失效的拿铁条目放进 `audit`（reason 为 `revised`）；`auto` 在没有 EP 时不做扩展。
51. `test_pcir_obligations` —— 参数化：唯一 anchor 被质疑的 atom → `compile()` 仍会构造该条目（`anchor_ids` 包含被质疑的 anchor，`explain['disputed_anchor_ids']` 非空），而 `to_pcir()` 抛出 `ProofObligationError`，违反 `evidence_coverage`（`details['missing_witness'] == [item_id]`）；未解决的冲突 → `no_unresolved_conflict`；`filters={'visibility': 'team'}` 遇到 private atom → 该 private atom 不被选中（`retrieval['dropped_governance'] >= 1`，或干脆不出现）且 `governance` 为 `satisfied`；手工构造的 `ContextIR`（`retrieval={'audience': 'team', 'anchors': [...]}`）若含 `visibility='private'` 的条目 → `lower()` 报告 `governance` 被违反；被撤销的 anchor（经 `hms.revoke`）→ `not_revoked`；`require_witness=False` 时返回 `ok() is False` 而不抛异常。

**G —— `test_ops.py`**
52. `test_feedback_matrix[feedback_type]` —— 11 个用例 + 3 个子用例：期望的 `actions` 名称及顺序、`FEEDBACK` history 行、只有 `privacy_violation` 会改动 `visibility`（对其余 10 个逐一断言）、`evidence_wrong` 绝不让一个没有未被质疑见证者的 atom 保持有效、`state_missing` 不带 correction → `VALIDATION_007`、未知类型 → `VALIDATION_006`；子用例：`state_missing_no_target`（`feedback(None, 'state_missing', correction={'scope': scope4, 'text': ...})` 与 `feedback('user_id=alice&run_id=run-a', 'state_missing', correction='...')` 都会在对应 scope 内创建一个 `state` atom，`target_kind='scope'`；完全没有 scope → `VALIDATION_001`）、`connection_reason_wrong_same_type`（仅改标签 → `[{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}]`，边仍有效，标签已更新，`fb_anchor` 被加为见证者）、`evidence_wrong_revoke`（`feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` → 一个 `revoke` 动作；`anchors.revoked_at` 被设置）。
53. `test_checkpoint_rollback_symmetric` —— S6 流程（`mid` 阶段的 compile 使用默认 `as_of`）：`restored`/`archived`/`edges_invalidated`/`edges_created` 完全一致；`get(mem)` 回到原文本；history 为 `[ADD, FEEDBACK, UPDATE, ROLLBACK, RESTORE]`；对「检查点之前归档、之后恢复」的 atom 做回滚会重新归档它；`ep_state`：检查点之后执行 `delete(<episode memory>)` → `store.list_eps(scope4)` 变短、`handoff(goal=None).goal` 改变，而 `rollback(cp)` 会恢复 `eps.valid_to IS NULL` 和之前的 goal；未知 id → `ValueError`。
54. `test_handoff_state_package_roundtrip` —— `handoff(filters=scope4, goal=...)` → JSON → `StatePackage.from_json` 相等；见证闭包包含 `source_events`（每个 anchor 一条）和 `atoms`；`checkpoint_id` 已设置；`rollback(pkg.checkpoint_id)` 可用。
54b. `test_handoff_import_roundtrip` —— 在 S4（`card2` 阶段）之后：`pkg = A.handoff(filters=scope4, checkpoint=False)`；新建 `B = HMS(config, store=StateStore(':memory:', now=clock.now, id_factory=SequentialIds()))`；`rep = B.import_package(pkg.to_json())` → `rep['imported']['atoms'] == len(pkg.atoms)`、`rep['imported']['anchors'] == len(pkg.anchors)`；`B.compile('fix checkout webhook retry', filters=scope4, budget_tokens=800, as_of='2026-06-15T12:00:00Z').to_text() == card2`（卡片不含 id）；再次导入同一个包 → `rep['imported']` 全为 0，`rep['skipped']` 非零；`import_package(pkg, run_id='run-import')` 会给每个对象重新划定 scope（`get_all(filters={'user_id': U, 'run_id': 'run-import'})` 非空）且 id 保持稳定（`B.get(<mem_uc>)` 可用）。
55. `test_transitions_export_golden_and_idempotent` —— S6 的元组 == §19.6；`fmt='jsonl'` == golden（`golden.masked`）；每一行的 `kappa_t` 键恰好是 `constraints, visibility, permission_scope, provenance`；第二次导出不新增行；`since`/`scale`/`kinds` 过滤生效；`consolidate()` 之后存在一条 `interface_ep` 行（`kinds=['interface_ep']`、`scale='L2_meso'`）。
56. `test_consolidate_phases_reports_and_decay` —— S4 之后：`consolidate()` → 报告 == §19.7（`entity_atoms == 6`、`interface_eps == 1`、`projected_edges == 3`、`interface_eps_l3 == 0`、`skills == 1`、`transitions == 1`、`compiled_states == 1`）；temporal/causal 边不重复创建（`temporal_edges == causal_edges == 0`，两者 `mentions == 2` —— MERGE）；`store.last_compiled_state(scope4)['card']` == `compile(..., mode='deep').to_text()`；`orient` 报告 `last_compiled`；随后 `store.set_atom_confidence(<atm_e1>, 0.2, at=clock.now())`、`clock.tick(days=400)`、`consolidate(phases=['prune'])` → `report['decayed'] == 3`（`<atm_e2>`、`<atm_e3>`、`<atm_e4>`：0.7 → 0.61，各带一条 `UPDATE` history 行），`report['archived'] == [<atm_e1>]` 并带一条由调度器 anchor 见证的 `ARCHIVE` history 行；被 pin 的/约束/skill/entity/处于冲突中的 atom 不受影响；`collect` 会重新路由那些 atom 因 Txn B 中止而被回滚掉的事件。
56b. `test_revoke_cascade` —— S4 之后：`hms.revoke(<anc_e2>)` → `archived == [<atm_e2>]`（它唯一的见证者），`<ep1_atom>` 与 `<atm_dc>` 保持有效（还有别的见证者；被撤销的 anchor 在它们上面被标记为质疑），`edges_invalidated == [<edg_distill_ep1_dc>]`（其唯一见证者是 `<anc_e2>`），`anchors.revoked_at` 被设置，history `ARCHIVE` 由系统 anchor 见证；此后 `compile(...).to_pcir()` 通过 `not_revoked`（没有存活条目在未被质疑的情况下引用被撤销的 anchor）；第二次 `revoke` → `noop`。
56e. `test_revoke_existence_semantics_and_projection_sources` —— S4 + `consolidate()` 之后：两条低层 `use` 边 `<ep1> → <atm_uc>` / `<ep2> → <atm_uc>` 同时是 L2 `use` 边 `<iep> → <atm_uc>` 与 L3 技能 `distill` 边的 `projections_of`；`revoke(<anc_e4>)`（EP2 的**主见证** `anchor_ids[0]`）只归档 `<atm_e4>`，`<ep2_atom>` 保持有效且 `atom_anchor_ids == [<anc_e3>]`（`include_disputed=True` → `[<anc_e4>, <anc_e3>]`），IEP 与四条边都有效；`revoke(<anc_e1>)` 使 `<use ep1→uc>` 失效（`witness_revoked`）但**不**使 L2/L3 边失效（还有一个有效来源；`to_pcir().ok()`）；`add_edge_witness(<L2 use>, <anc_uc>)` 之后 `revoke(<anc_e3>)` 归档 `<atm_e3>` 与 `<ep2_atom>`（`archived_reason='revoked'`），使 `<use ep2→uc>` 失效（`witness_revoked`），随后 L2 `use` 与 L3 `distill` 边以 `invalidated_reason='projection_revoked'` 失效、而各自仍有未撤销见证；投影边排在见证规则失效的边之后报告；IEP 保持有效（`<anc_e2>` 仍在其证据中）；第二次 revoke → `noop`。一个辅助函数在每一步之后对 scope 内所有边断言全局规则：有效 ⇒（≥1 个未撤销见证 ∧（无投影 ∨ ≥1 个有效来源））；`witness_revoked`/`projection_revoked` ⇒ 其否定。
56f. `test_relationship_not_useful_keeps_high_edge_with_other_sources` —— 同一设置：`feedback(<use ep1→uc>, 'relationship_not_useful')` → 恰好一个 `invalidate_edge` 动作，L2 `use` 与 L3 `distill` 边保持有效；`feedback(<use ep2→uc>, 'relationship_not_useful')` → 三个 `invalidate_edge` 动作，顺序为 `[<use ep2→uc>, <L2 use>, <L3 distill>]`，reason 都是 `'feedback:not_useful'`。
56c. `test_entity_atoms_after_consolidate` —— S2/S3 + `consolidate(filters={'user_id': U})` 之后：`get_all(filters={'user_id': U, 'scale': 'entity'})['results']` 含一行 `memory == 'Sarah'`、`atom_type == 'entity'`、`hms.source_kind == 'entity'`、`content['mentions'] == 3`、`anchor_ids` == 三个 Sarah anchor 的排序结果；`axis_query({'filters': {'user_id': U, 'entity': 'Sarah'}, 'scale_constraint': ['event', 'entity', 'L1_micro']})` 返回该 entity atom 以及两个 `event` atom；第二次 `consolidate` 不产生任何新东西；再加入第四条 Sarah 消息并 consolidate 会修订该 entity atom（`version == 2`，revision 边由调度器 anchor 见证）。

56g. `test_purge_dry_run_reports_and_changes_nothing` —— 一个 100 天前删除的墓碑：`purge(dry_run=True)` 报告它（`text_sha256`、`bytes`、按五个原因分组的 `protected`、`anchor_id None`），`store.stats()` 与其历史不变；200 天阈值不检查任何行；`older_than_days=-1` → `VALIDATION_009`；空范围 → `VALIDATION_001`。
56h. `test_purge_protections_keep_needed_rows` —— S4 + consolidate 之后：双版本谱系的头（`head_version`）、作为 `use` 边源点的墓碑（`edge_endpoint`）、检查点记录为存活的墓碑（`checkpoint_live`，随后被 `rollback` 恢复）、InterfaceEP 的 EP 原子（`interface_ep_support`）都在 `purge(dry_run=False)` 中幸存；与存活原子共享锚点的墓碑被清而锚点保留（`anchors == before + 1`，即清理自身的锚点）；随后 `keep_witnessed=False` 只删除一个无人引用的裸原子的锚点 + 源事件。
56i. `test_purge_materializes_with_audit_and_keeps_survivors_intact` —— 两个墓碑（`deleted`、`decayed`）：`purged == 2`，行/节点/链接/向量消失，`history` 以 `[..., 'DELETE'|'ARCHIVE', 'PURGE']` 结尾且 `is_deleted True`、`old_memory None`、`details_json` 带 `text_sha256` 与 `cutoff`，清理 `system` 事件的 `payload['op'] == 'purge'`；`stats()` 增量精确（+1 锚点/事件、+2 历史、边不变）；deep 卡片前后逐字节相同；`rollback` 到后来的检查点什么也不恢复；第二次清理是空操作；裸 `DELETE FROM memory_atoms` 重新变回 `STORE_003`，`meta.purge_in_progress` 不存在。
56j. `test_prune_runs_retention_only_when_both_switches_are_on` —— `prune` 默认与只开 `retention.enabled` 时都报告 `purged == 0`；`enabled + run_in_prune`（`older_than_days=30`，时钟 +60 天）时 `purged == 1`，行消失且其历史以 `PURGE` 结尾。
56k. `test_vacuum_shrinks_file_after_purge` —— 文件库上 40 个 4 KB 的墓碑：`purge(dry_run=False)` 之后，在 `transaction()` 内 `vacuum()` → `STORE_001`；在外则 `after_bytes < before_bytes`、`freed_bytes >= 160 000`、`os.path.getsize(path) == after_bytes`。
56d. `test_consolidate_llm_proposer_drops_malformed_proposals` —— S4 配置 `graph.causal_proposer='llm'`，脚本化的 `edge_proposal` 回复带有 `witness_refs` 为 `7`、`[['w1']]`、`[{'ref': 'w1'}]` 的三条提议，以及 `confidence` 为 `NaN`（字面量）与 `'nan'`（字符串）的两条有效边 → `consolidate()` 对该 scope 正常完成（`interface_eps == 1`、`compiled_states == 1`、`last_consolidation` meta 已写），三条畸形提议各以一条告警丢弃，新建的 `use` 边以 `confidence == 0.7`、`proposer == 'scripted@v1'` 落库，已有的 `causal` 边保持 `confidence == 0.7` 且 `mentions == 2`。

**H —— `test_facade.py`**
57. `test_mem0_signatures_verbatim` —— 10 个 Mem0 方法加上 `chat`/`project` 的 `inspect.signature` 与 §17.2 完全一致，HMS 原生方法与 §17.3 完全一致（`compile(mode: str | None = None)`、`feedback(target_id: str | None, ...)`、`context(query, *, filters, top_k, as_of, budget_tokens)`）；`MemoryBase` 的抽象方法集合正确；§17.1 列出的公开属性均存在。
58. `test_mem0_envelopes_and_validation_codes` —— `add` 的返回信封；`update/delete/delete_all` 的消息逐字一致；`get` 未命中 → `None`；`history` 返回裸列表；`VALIDATION_001/002/003/008/009/010/011`；全空白 id → `ValueError`；`search` 里出现顶层 `user_id` → `ValueError` 的子类；`delete_all()` 的消息逐字一致。
59b. `test_purge_facade_is_explicit_default_off_and_audited` —— `hms.delete(mem)` 后 +100 天：`purge(filters=S1, older_than_days=90)` 是试运行且列出 `mem`；默认配置下 `dry_run=False` → `ConfigError(CONFIG_001, details={'retention.enabled': False, 'dry_run': False})`；`filters={}` → `VALIDATION_001`；同一 store 上 `retention.enabled=True` 的 `HMS` 将其清理（`get(mem) None`、`latest_version None`、`history(mem)` 事件为 `['ADD', 'DELETE', 'PURGE']` 且键集与 Mem0 一致、`why(kept)` 仍可解析、`why(mem)` → `NOT_FOUND_001`），`vacuum()` 返回四个键，再清理一次 `purged == 0`。
59. `test_update_delete_expiration_lineage` —— `update(id, text=)` → id 不变、`version == 2`、旧版本归档、产生 revision 边、写 `UPDATE` 行；`update(metadata=)` → 不产生新版本；`update(expiration_date=None)` 清空 `valid_to`；`update(expiration_date='2026-12-31')` → `valid_to == '2027-01-01T00:00:00.000000+00:00'`；`delete` → `get is None`，行保留且 `archived_reason='deleted'`，`DELETE` 的 `is_deleted` 为 True。
60. `test_from_config_mem0_dict_and_rerank_noop` —— **离线**：`d = deepcopy(mem0_config.json)`；`d['llm'] = {'provider': 'rule'}`、`d['embedder'] = {'provider': 'hash'}`、`d['vector_store'] = {'provider': 'sqlite', 'config': {}}`、`d['reranker'] = {'provider': 'overlap', 'config': {}}`（保留 `history_db_path`、`version`、`custom_instructions`）；`HMS.from_config(d)` 可构造（`store.path == history_db_path`）且带一个 `OverlapReranker`、无告警，`search(rerank=True)` 各行带 `rerank_score`；`{'provider': 'cohere'}` → `CONFIG_002` 且 `details['kind'] == 'reranker'`；**未经修改**的 fixture 只被测试 10 使用。
61. `test_migrate_from_mem0_walks_get_all` —— 用一个伪造的 `memory` 对象 → 各行以 `infer=False` 和 `timestamp=created_at` 被重新写入。
62. `test_cli_smoke` —— 在 `HMS_DIR=tmp_path` 下通过 `main([...])` 跑 `hms add/search/compile`。

**T —— `test_scenarios.py`**
63. `test_scenarios_end_to_end[index_provider]` —— 在**同一个** `hms` 上依次执行 S1–S6（§19），逐条断言每个 `expect` 块（卡片用 `golden.text` 逐字节比对、chain 逐字节比对、transitions 用 `golden.masked`），随后 `consolidate()` + `graph_stats()` == §19.7，含消融对比块（在 S4 查询上跑 `fast`/`axis_graph`/`deep`）。

**召回池（分组 D，`tests/test_retrieval.py`）**
64a. `test_equivalence_theorem[index_provider × n5/n30/n60 × whole/channels]` —— 定种随机语料（人名、文件、CJK、18 个月、约 10% 无向量），20 个查询 × `top_k ∈ {3, 5, 10}` × `threshold ∈ {0.0, 0.3}`：`_exhaustive_search()`（作为参考保留在测试里的池化前算法）与 `execute_axis_query` 的 `(atom_id, score)`、`score_details` 的 7 个 Mem0 键与 7 个 HMS 键 `==`（逐位相同），整体入池规则开启与 `hard_gate_pool=False`（仅通道）两种模式都成立；`test_equivalence_theorem_milvus` 在 Milvus 提供者上重复（`importorskip`）。
64b. `test_pool_recall_overlap_at_n500` —— N=500 > pool_max：逐查询打印与穷举路径的 top-20 重叠率（不设硬阈值），断言每个通道大小 `<= K` 且池 `<= pool_max`。
64c. `test_no_newest_first_truncation` —— 在两种召回模式下，`ScoringConfig(max_candidates=10)`（发出 `DeprecationWarning`）仍能找到一条久远的唯一标识符原子，而带同样上限的参考实现会丢掉它。
64d. `test_channel_sizes_and_recalled_by`、`test_hard_gate_whole_set_enters_pool_or_channels`、`test_safety_limit_warns_never_truncates`、`test_axis_only_paths_keep_sql_semantics`、`test_max_candidates_is_a_deprecated_alias`、`test_compile_candidates_reuse_the_pool`、`test_python_filter_scores_the_whole_gate` —— 溯源键、硬时间门上的整体入池与通道切换（池 ∩ G）、按前缀区分的两条安全上限警告（原生下推下只有池警告）、无文本路径从不调用 `recall_pool`、已弃用别名（None 哨兵：`ScoringConfig(**ScoringConfig().model_dump())`、`HMSConfig.model_validate(HMSConfig().model_dump())` 与 `coerce_config` 都不告警）、小规模 compile 的 `ContextIR.retrieval['channels_used'] == ['gate']`，以及 Python 侧过滤（json1 关闭、N > pool_max）对 G 整体打分且与穷举路径逐位相同。
64e. `tests/test_index_sync.py` —— `test_listener_dispatch_follows_commit_order`（sqlite / memory / milvus：两个线程先以变慢的监听器归档再恢复 → 监听器按提交顺序调用、每次都是完整快照、提供者重新存活）、`test_search_inside_transaction_never_deadlocks`（`store.transaction()` 内的搜索对并发搜索，5 s join）、`test_listener_registered_once_and_removed_on_close`（同一 store 上两个门面：各一个监听器，`close()` 恰好移除自己的）、`test_late_topic_and_entity_reach_the_keyword_channel`（插入后链接的主题 / 实体在 `atom_fts` 中；已归档版本绝不重新获得行）。
64f. `tests/test_server.py::test_internal_errors_are_redacted` —— 携带 `path` / `sql` 细节的 `StoreError` 返回 500 且 `message == 'internal error'`、`details` / `debug_info` 为空，路径只出现在带请求 id 的服务端日志行里；`RuntimeError` / `KeyError` → 带 `X-Request-ID` 的 500 `INTERNAL_000` JSON，绝不是 Starlette 的纯文本 500。
64g. `tests/test_retrieval_v3.py`（RETRIEVAL_BENCH_2026-09-10）—— `test_axis_ids_unordered_and_index_hints`（order=None 不产生 ORDER BY；`index_hint=TIME_INDEX` 的计划经 ix_atoms_time；非标识符提示 → VALIDATION_000）、`test_channels_search_inside_the_level_gate`（三月门控且 |G| > K 时关键词 / 实体候选留在 G 内，并找到 scope 范围前 K 漏掉的门控内原子；门控下的仅通道池 ⊆ G）、`test_entity_gate_is_a_non_correlated_subquery`（`_plan_gates` 把实体门降级为 `atom_id IN (SELECT …)` 且选出 DSL 的 EXISTS 集合；G 经 atom_entity 一次枚举、无探测）、`test_keyword_channel_two_phases`（先 AND 阶段，OR 阶段只用于补足；LIKE 退路一致）、`test_write_generations_and_data_version`、`test_memory_atoms_updates_stay_listener_covered`（源码扫描）、`test_sqlite_cache_follows_every_write`（upsert / 替换 / 删除 / 归档 / 恢复 / set_valid_to / 回滚 / 清空 / reset / 第二个连接，每步之后都与 `rank_hits` 及 `axis_ids` 一致）、`test_select_top_k_rows_and_scores_are_bit_equal`、`test_recency_channel_rule`、`test_semantic_channel_scores_are_reused_in_pool`（一次通道搜索、一次只针对剩余部分的池内搜索，与全量重算逐位相同）、`test_axis_result_carries_retrieval_stats`、`test_atom_type_pushdown_matches_sql`、`test_facade_search_passes_retrieval_config`、`test_ingest_upserts_carry_index_attrs`、`test_milvus_reopen_loads_collection`（`importorskip`）。`tests/test_perf_paths.py::test_memory_store_clones_template_identically` / `test_file_store_migrates_before_wal_switch` 另外钉住文件库克隆（sqlite_master 行、user_version 与 page_count 与迁移阶梯相同；已有文件仍走阶梯）。

**横切项（分组 T，位于 `test_foundation.py`）**
64. `test_bilingual_docstrings` —— 遍历每个 `hms_elite/**/*.py`、`tests/**/*.py`、`examples/*.py` 模块：模块 docstring 为一行英文后跟一行中文（含 CJK 字符），并按顺序出现 `# Purpose/# 目的/# Input/# 输入/# Output/# 输出/# State effect/# 状态影响` 这八行表头；每个公开函数/方法（用 `inspect` 检查，名称不以 `_` 开头，且定义在本模块内）都有 docstring，其前两行为先英后中，并按顺序包含 `Input / 输入：`、`Output / 输出：`、`State effect / 状态影响：` 三个标记，且 Input 之下的每个 `<arg>:` 行紧跟一个 `<arg>：` 行（全角冒号）。
65. `tests/test_context.py`（§6.7）—— `test_context_shape_and_none_parts`、`test_context_memo_reuse_and_invalidation`、`test_context_idle_writes_keep_the_memo`、`test_store_scope_version_moves_at_commit`、`test_context_timestamps_from_time_anchor`、`test_context_builders_are_the_harness_builders`、`test_context_config_fields_and_overrides` —— `HMS.context` / `StateStore.latest_event` / `touch_scope` / `unversioned` / `scope_version` / `compiler.context.*` / `CompilerConfig.context_*`；`tests/test_async.py` 与 `tests/test_facade.py` 把 `context` 钉入 41 个方法 / 18 个原生方法的一致性与签名列表。


---

## §19 黄金场景 S1–S6（精确输入与预期输出）

六个场景在同一个 `HMS` 实例上按顺序运行（`rule` LLM、`hash` embedder、index provider 参数化；`tests/test_scenarios.py` 的模块级 `config` 夹具把 `ingest.extraction_output` 钉在 `'full'`，因为下文的预期是在规则 LLM 的完整观察上陈述的——R1 偏好置信度 0.8、R2 约束 0.9、§19.8 哈希——这样库默认值的变化——2026-09-18 起为 `lines`（第 108 行），它与 `compact` 一样把置信度推导为 0.7，§6.5 / §6.6——不会改写它们），配合 §18.1 的 fixture（`FrozenClock` 起点 `2026-09-02T10:00:00.000000+00:00`，`SequentialIds`）。场景运行器在每次公开调用之前调用 `clock.tick(seconds=60)`。预期值中的 id 是符号化的（例如 `<mem_coffee>`），由更早的返回值解析得到；黄金文本文件中绝不包含 id 或时钟瞬时值（黄金卡片使用显式 `as_of`；证据行使用 fixture 时间戳；S6 的 `mid` 卡片只按 slot 内容断言，永远不与黄金文件比对，因为它必须使用时钟）。下文的 `sha256:` 前缀按 §0.1 的规则、对 §4.3 归一化后的 payload 计算得到（完整哈希见 §19.8）——rule LLM、ingest 和测试套件三者与这些值完全对齐。

### §19.1 S1 — 带修订的偏好召回（`tests/fixtures/events/chat_preference.json`）

```python
U = "alice"; s1 = {"user_id": U, "run_id": "run-coffee"}
r1 = hms.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", **s1)
r2 = hms.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", **s1)
```
规则抽取（§11.1 R1）：msg1 → `{"type": "preference_event", "subject": "user", "predicate": "likes", "objects": ["oat-milk lattes in the morning"], "relation": "single_fact", "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02", "sentiment": "positive", "topic_tags": ["coffee", "preference"], "entity_mentions": [], "confidence": 0.8}`；msg2 → `{"type": "preference_event", "subject": "user", "predicate": "likes", "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.", "date_observed": "2026-04-20", "sentiment": "positive", "topic_tags": ["coffee", "preference"], "confidence": 0.9, "supersedes_hint_ids": ["0"]}`。修订判定（§11.3 规则 6）→ `REVISE`。

预期：
- `r1 == {'results': [{'id': <mem_coffee>, 'memory': 'User likes oat-milk lattes in the morning.', 'event': 'ADD', 'atom_id': <atm_v1>, 'version': 1, 'atom_type': 'state', 'scale_level': 'state', 'anchor_ids': [<anc_1>], ...}]}`。
- `r2['results'] == [{'id': <mem_coffee>, 'memory': 'User likes black coffee.', 'event': 'UPDATE', 'revises': <atm_v1>, 'version': 2, ...}]`（同一条 lineage id）。
- `hms.get(<mem_coffee>)['memory'] == 'User likes black coffee.'` 且 `['version'] == 2`；`hms.get(<mem_coffee>)['valid_to'] is None`。
- `[h['event'] for h in hms.history(<mem_coffee>)] == ['ADD', 'UPDATE']`；UPDATE 行的 `old_memory == 'User likes oat-milk lattes in the morning.'`、`new_memory == 'User likes black coffee.'`、`anchor_id == <anc_2>`。
- `len(hms.get_all(filters=s1)['results']) == 1`；`len(hms.get_all(filters=s1, show_expired=True)['results']) == 2`，且归档行的 `valid_to == '2026-04-20T08:00:00.000000+00:00'`（ingest 阶段的 REVISE 使用新事件的时间，§12.5），`hms.archived_reason == 'revised'`；UPDATE 历史行的 `updated_at == '2026-04-20T08:00:00.000000+00:00'`。
- `hms.search("what coffee does the user like", filters={"user_id": U})['results'][0]['memory'] == 'User likes black coffee.'`，且没有任何结果等于 latte 那条文本；带 `explain=True` 时：`score_details['route'] == 'preference'`、`['primary_axis'] == 'type'`、`['axis_gates']['obs_type'] == ['preference_event', 'habit']`。
- `hms.axis_query({'filters': {'user_id': U, 'topic': 'coffee'}, 'project_axis': 'value', 'scale_constraint': ['state'], 'aggregation': 'latest'})`（HMS §3.5 偏好查询）→ `projection == [['black coffee']]`、`len(hits) == 1`；同一查询改用 `aggregation='count'` → `count == 1`。
- 恰好存在一条有效的 `revision` 边 `<atm_v1> → <atm_v2>`，其 `witness_anchor_ids == [<anc_2>]`、`created_by == 'ingest'`。
- `ir = hms.compile("coffee", filters=s1)`：`[i.text for i in ir.preferences] == ['User likes black coffee.']`、`ir.facts == []`、`ir.audit` 中没有 `revised` 条目（auto 模式，无 EP）；`hms.compile("coffee", filters=s1, mode="deep").audit` 恰好包含一条条目，其 `explain['reason'] == 'revised'`、`text == 'User likes oat-milk lattes in the morning.'`、`explain['superseded_by_text'] == 'User likes black coffee.'`。

### §19.2 S2 时间 + S3 跨会话实体（`chat_multisession.json`）

```python
hms.add("Sarah joined the billing team in March.", timestamp="2026-03-02T09:00:00Z", user_id=U, run_id="run-a")
hms.add("Sarah shipped the webhook retry design on May 6.", timestamp="2026-05-06T09:00:00Z", user_id=U, run_id="run-b")
hms.add("Sarah is reviewing the checkout refactor this week.", timestamp="2026-06-11T09:00:00Z", user_id=U, run_id="run-c")
```
规则抽取：#1（R7）`temporal_fact`，subject `Sarah`，predicate `did`，objects `[]`，`date_observed 2026-03-01`，text 逐字保留，topics `["work", "billing"]`，entity_mentions `["Sarah"]` → atom_type `event`，`time_anchor == '2026-03-01T00:00:00.000000+00:00'`；#2（R7）`temporal_fact`，`date_observed 2026-05-06`，topics `["work", "engineering"]`，`time_anchor == '2026-05-06T00:00:00.000000+00:00'`；#3（R9）`fact`（`this week` 属于 `relative_this`，被 R7 忽略），predicate `is`，topics `["engineering", "billing"]`，`time_anchor == '2026-06-11T09:00:00.000000+00:00'`。三条判定均为 ADD（run scope 不同 → 无候选）。

预期：
- `[r['memory'] for r in hms.search("what happened in May", filters={"user_id": U})['results']] == ['Sarah shipped the webhook retry design on May 6.']`；带 `explain=True` 时：`score_details['axis_gates']['time'] == ['2026-05-01T00:00:00.000000+00:00', '2026-05-31T23:59:59.999999+00:00']`、`['primary_axis'] == 'time'`、`['route'] == 'temporal'`。
- `[r['memory'] for r in hms.get_all(filters={"user_id": U, "entity": "Sarah"})['results']] == ['Sarah is reviewing the checkout refactor this week.', 'Sarah shipped the webhook retry design on May 6.', 'Sarah joined the billing team in March.']`（按 time_anchor DESC）。
- `hms.get_all(filters={"user_id": U, "run_id": "run-b", "entity": "sarah"})['results']` 恰好一行；对 `(alice, NULL)` 存在一条 `normalized_text == 'sarah'` 的 `entities` 行，以及三条 `atom_entity` 链接（`store.entity_degree == 3`）。

### §19.3 S4 — Work State Card（`structured_billing.json`）

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
约束消息（R2）：`{"type": "constraint", "subject": "project", "predicate": "must_not", "objects": ["subscription state machine"], "relation": "constraint", "text": "Do not modify the subscription state machine.", "topic_tags": ["billing", "constraint"], "confidence": 0.9}` → atom `<atm_uc>`（memory `<mem_uc>`），anchor `<anc_uc>`（`source_ref == 'chat://agent_id=coder&run_id=run-billing&user_id=alice/0'`、`hash == '3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc'`）。

`ra` 的预期（§12.4）：事件 atom `file_diff src/billing/retry.ts: add exponential backoff`（`<atm_e1>`，anchor `<anc_e1>` `file://src/billing/retry.ts`）与 `test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression`（`<atm_e2>`，`<anc_e2>` `test://billing_e2e`）；EP1 `<ep1>`：`u='fix checkout webhook retry'`、`b='Do not modify the subscription state machine.'`、`a='patch src/billing/retry.ts: add exponential backoff'`、`y='billing_e2e failed: subscription_state_regression'`、`r='negative'`、`kappa={'constraints': ['Do not modify the subscription state machine.'], 'visibility': 'private', 'evidence': [<anc_e2>, <anc_e1>]}`、`anchor_ids == [<anc_e2>, <anc_e1>]`、`time_anchor == '2026-06-15T10:04:00.000000+00:00'`，episode atom 文本为 `[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression`，实体键为 `{'retry.ts', 'billing_e2e', 'subscription_state_regression'}`；派生约束 `<atm_dc>`：文本 `Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression`、`obs_type='constraint'`、`subject='project'`、`predicate='must_not'`、`anchor_ids == [<anc_e2>, <anc_e1>]`，实体键包含 `retry.ts`；边：`use <ep1_atom> → <atm_uc>`（witness `[<anc_e1>]`，置信度 0.8）、`distill <ep1_atom> → <atm_dc>`（witness `[<anc_e2>]`，置信度 0.9）；`ra['eps'] == [<ep1>]`、`len(ra['edges']) == 2`、`ra['constraints'] == [<atm_dc>]`、`ra['events']` 有 2 条且 `dedup False`；`transitions` 表新增一行 `(ep, <ep1>)`。

`rb` 的预期：事件 atom `<atm_e3>`（`file://src/billing/idempotency.ts`）、`<atm_e4>`（`test://billing_e2e`）；EP2 `<ep2>`：`u='fix checkout webhook retry'`、`b='Do not modify the subscription state machine. ; Do not re-apply \'patch src/billing/retry.ts: add exponential backoff\': billing_e2e failed: subscription_state_regression'`、`a='patch src/billing/idempotency.ts: add idempotency key'`、`y='billing_e2e passed'`、`r='positive'`，`kappa.constraints` 为两条约束文本（按该顺序），`anchor_ids == [<anc_e4>, <anc_e3>]`；边：`use <ep2_atom> → <atm_uc>` 与 `use <ep2_atom> → <atm_dc>`（witness `[<anc_e3>]`）、`temporal <ep1_atom> → <ep2_atom>`（witness `[<anc_e4>]`，置信度 0.5）、`causal <ep1_atom> → <ep2_atom>`（witness `[<anc_e3>, <anc_e2>]`，置信度 0.7）；无派生约束；`len(rb['edges']) == 4`；`transitions` 表新增第二行 `(ep, <ep2>)`。

`card1`（黄金文件 `work_state_card_failed.txt`，精确匹配）：
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
`card1` 背后的 slot：`constraints == [<atm_uc>, <atm_dc>]`、`episodes == [<ep1_atom>]`、`facts == []`（两个事件 atom `<atm_e1>`、`<atm_e2>` 因 `covered_by_episode` 被丢弃；`retrieval['dropped_covered'] == 2`）、`preferences == skills == []`、`audit` 为 3 条证据项、`unresolved_conflicts == []`、`route == 'recall'`、`retrieval['graph_hops'] == 1`、`retrieval['ablation'] == 'axis_graph_support'`（auto 解析为 deep：存在 EP）。

`card2`（黄金文件 `work_state_card_fixed.txt`，精确匹配）：
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
`card2` 背后的 slot：`retrieval['dropped_covered'] == 4`（四个事件 atom 全部被 `<ep1_atom>`/`<ep2_atom>` 覆盖）。`card2.to_pcir().ok() is True`，且除 `governance == 'skipped'`（无受众）外每条义务均为 `satisfied`。此阶段的消融：`hms.compile(..., mode='deep').to_text().replace('Mode: deep', 'Mode: auto') == card2`，且 `hms.compile(..., mode='axis_graph').to_text().replace('Mode: axis_graph', 'Mode: auto') == card2`（此时尚不存在 InterfaceEP，因此 Support 不增加任何内容）；`hms.compile(..., mode='fast')` 的 `retrieval['ablation'] == 'axis'`、`retrieval['graph_hops'] == 0`，且不触碰任何 `graph_*` 表。

### §19.4 S4b — 按标识符检索

`hits = hms.search("retry.ts", filters=scope4, top_k=3)['results']`：`{h['atom_id'] for h in hits} == {<atm_e1>, <ep1_atom>, <atm_dc>}`；每条命中都满足 `'retry.ts' in h['hms']['entity_ids']`；带 `explain=True` 时每条都有 `score_details['entity_boost'] == 1.0`（此处实体 `retry.ts` 的度为 3——`<atm_e1>`、`<ep1_atom>`、`<atm_dc>`——而 hub 阻尼从大于 3 才开始生效，§13.4），并且 `'entity' in score_details['signals']`。

### §19.5 S5 — 侦探式 `why()`（问题模式）

```python
trace = hms.why("why must retry.ts not be modified", filters=scope4)
```
预期：`trace['target'] == {'kind': 'atom', 'id': <atm_dc>, 'node_id': <atm_dc>, 'text': "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'resolved_from': 'query'}`；`{n['node_id'] for n in trace['nodes']} == {<atm_dc>, <ep1_atom>, <ep2_atom>, <atm_uc>}`；`{(e['src_id'], e['dst_id'], e['edge_type']) for e in trace['edges']} == {(<ep1_atom>, <atm_dc>, 'distill'), (<ep2_atom>, <atm_dc>, 'use'), (<ep1_atom>, <atm_uc>, 'use'), (<ep1_atom>, <ep2_atom>, 'causal'), (<ep2_atom>, <atm_uc>, 'use')}`；每一条 `witness_anchor_ids` ⊂ `{a['anchor_id'] for a in trace['anchors']}`；`trace['unresolved_conflicts'] == []`；`trace['refined_from'] == []`；`trace['chain']` 等于黄金文件 `why_chain.txt`：
```
Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
-use[file://src/billing/retry.ts]-> Do not modify the subscription state machine.
```
`trace['explanation'] == "\n".join(trace['chain'])`。`hms.why(<ep1>)`（按 ep_id 调用）解析出 `resolved_from == 'id'`、`target['node_id'] == <ep1_atom>`；`hms.why("nonsense-id")` 不带 filters 时 → `ValueError`（`NOT_FOUND_001`）。

### §19.6 S6 — checkpoint → feedback → rollback → transitions

```python
cp = hms.checkpoint(filters=scope4, label="before-feedback")
fb = hms.feedback(<mem_uc>, "state_outdated", correction="retry.ts may change once billing_e2e is isolated.")
mid = hms.compile("fix checkout webhook retry", filters=scope4)          # NO as_of: defaults to store.now(), i.e. one tick AFTER the feedback instant (the successor is valid only from that instant, §16.1)
rb = hms.rollback(cp)
tr = hms.transitions(filters=scope4)
```
预期：
- `fb['actions'] == [{'action': 'archive', 'atom_id': <atm_uc>}, {'action': 'create', 'atom_id': <atm_uc2>}, {'action': 'revision_edge', 'edge_id': <edg_rev>}]`；`hms.get(<mem_uc>)['memory'] == 'retry.ts may change once billing_e2e is isolated.'`、`['version'] == 2`；新 atom 的实体键 ⊇ `{'retry.ts', 'billing_e2e'}`，`obs_type == 'constraint'`，`anchor_ids[0] == <anc_fb>`（该 feedback 事件的 anchor）。
- `mid`：`[c.text for c in mid.constraints] == ["Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'retry.ts may change once billing_e2e is isolated.']`——恰好是这两项（按 time_anchor 升序：后继者的 `time_anchor` 就是 feedback 瞬时；不会合成 `#kappa:` 条目，因为两个 EP 的 κ 文本 `Do not modify the subscription state machine.` 已作为 audit 条目出现，§15.1）；`mid.audit` 包含一条 `reason == 'revised'` 的条目，其 `text == 'Do not modify the subscription state machine.'`；`mid.as_of == store.now()`（调用时刻，是一个时钟瞬时值，因此 `mid.to_text()` 永不与黄金文件比对）。
- `rb == {'message': 'Rolled back to before-feedback', 'checkpoint_id': cp, 'restored': [<atm_uc>], 'archived': [<atm_uc2>], 'edges_invalidated': [<edg_rev>], 'edges_revalidated': [], 'edges_created': [<edg_rb>], 'anchor_id': <anc_rb>}`；`hms.get(<mem_uc>)['memory'] == 'Do not modify the subscription state machine.'`、`['version'] == 1`；`<edg_rb>` 是 `revision <atm_uc2> → <atm_uc>`，其 `label == f'rollback:{cp}'`、`witness_anchor_ids == [<anc_rb>]`、`created_by == 'ops'`。
- `[h['event'] for h in hms.history(<mem_uc>)] == ['ADD', 'FEEDBACK', 'UPDATE', 'ROLLBACK', 'RESTORE']`。
- `tr`（默认 `fmt='tuples'`）恰好包含 5 个六元组，顺序如下（每个 `w_t` 均非空）：
  1. `(s_t, 'patch src/billing/retry.ts: add exponential backoff', s_t1, 'negative', kappa, w_t)` — `s_t['u'] == 'fix checkout webhook retry'`、`s_t['ep_id'] == <ep1>`、`s_t1['y'] == 'billing_e2e failed: subscription_state_regression'`、`kappa['constraints'] == ['Do not modify the subscription state machine.']`、`w_t == [<anc_e2>, <anc_e1>]`
  2. `(…, 'patch src/billing/idempotency.ts: add idempotency key', …, 'positive', …, [<anc_e4>, <anc_e3>])` — `s_t['ep_id'] == <ep2>`
  3. causal：`(s_t with ep_id <ep1> and y == 'billing_e2e failed: subscription_state_regression', 'patch src/billing/idempotency.ts: add idempotency key', s_t1 with ep_id <ep2>, 'positive', kappa, w_t == [<anc_e3>, <anc_e2>, <anc_e4>])` — `w_t = ordered_union(edge witnesses [<anc_e3>, <anc_e2>], EP2 anchors [<anc_e4>, <anc_e3>])`，按首次出现顺序（§16.4）；`kappa['constraints'] == [both constraint texts]`（EP2 的 κ），`kappa['provenance']['edge_id'] == <edg_causal>`
  4. revision：`({'text': 'Do not modify the subscription state machine.', 'atom_id': <atm_uc>, ...}, 'revise', {'text': 'retry.ts may change once billing_e2e is isolated.', 'atom_id': <atm_uc2>, ...}, 'negative', kappa, [<anc_fb>, <anc_uc>])` — `kappa == {'constraints': [<atm_dc text>, 'retry.ts may change once billing_e2e is isolated.'], 'visibility': 'private', 'permission_scope': 'private', 'provenance': {'label': '', 'created_by': 'feedback', 'proposer': 'rule'}}`（在 `time_t1` = feedback 瞬时处有效的约束：`<atm_uc>` 在该瞬时被归档，`<atm_uc2>` 自该瞬时起有效）
  5. rollback：`({'text': 'retry.ts may change once billing_e2e is isolated.', 'atom_id': <atm_uc2>, ...}, 'rollback', {'text': 'Do not modify the subscription state machine.', 'atom_id': <atm_uc>, ...}, 'neutral', kappa, [<anc_rb>])` — `kappa == {'constraints': ['Do not modify the subscription state machine.', <atm_dc text>], 'visibility': 'private', 'permission_scope': 'private', 'provenance': {'checkpoint_id': cp, 'label': 'before-feedback', 'reason': None}}`（在 rollback 瞬时有效的约束：`<atm_uc>` 已恢复，`<atm_uc2>` 已归档）
  每一行的 `kappa_t` 都恰好含有键 `constraints, visibility, permission_scope, provenance`；第 1–2 行携带 `provenance == {'evidence': [...], 'ep_id': ...}`。
- `hms.transitions(filters=scope4, fmt='jsonl')` 在经过 `mask_ids` + 时间戳掩码后等于黄金文件 `transitions_s6.jsonl`；再次调用不会新增 `transitions` 行（`store.stats()['transitions'] == 5`）。

### §19.7 收尾 — consolidate + graph_stats

```python
rep = hms.consolidate(filters=scope4)
st = hms.graph_stats(filters=scope4)
```
预期 `rep['phases']['consolidate'] == {'entity_atoms': 6, 'temporal_edges': 0, 'causal_edges': 0, 'conflicts': 0, 'interface_eps': 1, 'projected_edges': 3, 'interface_eps_l3': 0, 'skills': 1, 'transitions': 1, 'compiled_states': 1}` —— `entity_atoms`：与 scope4 存活 atom 关联的六个实体（`retry.ts`、`file_diff`、`billing_e2e`、`test_log`、`subscription_state_regression`、`idempotency.ts`；由 §9.2 在 event/EP/constraint atom 文本上取键）各自在 scope `(alice, coder, run_id=None)` 下获得一个 `entity` atom；`temporal_edges`/`causal_edges` 为 0，因为 `insert_edge` MERGE 进了已有的边（两条边此时 `mentions == 2`）；`transitions == 1` 是新增的 `interface_ep` 行；`compiled_states == 1`（`store.last_compiled_state(scope4)['card'] == hms.compile('fix checkout webhook retry', filters=scope4, mode='deep').to_text()`）。该 InterfaceEP：`support_node_ids == [<ep1_atom>, <ep2_atom>]`、`cohesion_score == pytest.approx(2/3, abs=1e-4)`、`slots.u == 'fix checkout webhook retry'`、`slots.a == 'patch src/billing/retry.ts: add exponential backoff → patch src/billing/idempotency.ts: add idempotency key'`、`slots.b == 'Do not modify the subscription state machine. ; ' + <atm_dc text>`（每条约束只出现一次）、`slots.y == 'billing_e2e passed'`、`slots.r == 'positive'`、`slots.kappa.evidence == evidence_anchor_ids`、`invariants` = `[constraint 'Do not modify the subscription state machine.' (support [<ep1_atom>, <ep2_atom>], anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>}), confidence 0.8), failed_path "do not: patch src/billing/retry.ts: add exponential backoff" (support [<ep1_atom>], anchor_ids == sorted({<anc_e1>, <anc_e2>}))]`、`residuals == []`、`evidence_anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`；IEP atom 的 `source_ep_ids == [<ep1>, <ep2>]`；高层边：`use <iep_atom> → <atm_uc>`（2 条 projection 行）、`use <iep_atom> → <atm_dc>`（1 条）、`distill <iep_atom> → <atm_dc>`（1 条）；skill atom 文本为 `'Skill: Do not modify the subscription state machine.'`（`scale_level == 'L3_macro'`、`anchor_ids == sorted({<anc_e1>, <anc_e2>, <anc_e3>, <anc_e4>})`、`confidence == 0.8`），并在 `L3_macro` 处有 `distill <iep_atom> → <skill>`，`is_boundary == False`、`witness_edge_id == <edg_use_ep1_uc>`，且 `store.projections_of(<edg_skill>) == [<edg_use_ep1_uc>, <edg_use_ep2_uc>]`；`interface_ep` transition 行：`a_t == slots.a`、`r_t == 'positive'`、`s_t1['invariants'] == [both invariant texts]`、`w_t == evidence_anchor_ids`、`time_t == '2026-06-15T10:04:00.000000+00:00'`、`time_t1 == '2026-06-15T10:14:00.000000+00:00'`。
`st == {'nodes': {'message': 0, 'observation': 3, 'event': 4, 'entity': 0, 'ep': 2, 'interface_ep': 1, 'skill': 1}, 'edges': {'temporal': 1, 'causal': 1, 'use': 5, 'distill': 3, 'revision': 2, 'conflict': 0}, 'valid_edges': 11, 'invalid_edges': 1, 'boundary_edges': 3, 'interface_eps': 1, 'unresolved_conflicts': 0, 'witness_coverage': 1.0, 'projection_coverage': 1.0, 'traceable_high_edges': 1.0, 'transitions': 6}`（`nodes.entity == 0`，因为实体 atom 位于 `(alice, coder)` 且无 `run_id`，而 `scope4` 按 `run_id` 过滤；`hms.graph_stats(filters={'user_id': U, 'agent_id': 'coder'})['nodes']['entity'] == 6`；`traceable_high_edges`：3 条投影出的 L2 边和 L3 skill distill 边都有 projection 行）。
consolidate 之后：`hms.compile('fix checkout webhook retry', filters=scope4, mode='deep')` 的 `[s.text for s in .skills] == ['Skill: Do not modify the subscription state machine.']`、`retrieval['ablation'] == 'axis_graph_support'`，并且至少有 1 条 audit 条目满足 `explain['reason'] == 'invariant:failed_path'`；`mode='axis_graph'` 给出相同的 `skills`，但不含任何 `item_id` 里带 `#inv:` 的条目，也没有 reason 以 `invariant:`/`residual:` 开头的 audit 条目（`retrieval['ablation'] == 'axis_graph'`）；`mode='fast'` 同样给出相同的 `skills`（skill atom 是 `L3_macro`，它在 recall 的 `scale_default` 中，因此仅靠 axis 一步就能找到），且 `retrieval['ablation'] == 'axis'`，没有 `#inv:`/`invariant:` 条目——只有 Support 层（`refine`）把 `deep` 与其他两种模式区分开。

### §19.8 完整 payload 哈希（对应上述 fixture；§0.1 规则）

| anchor | 哈希对象（规范化） | sha256 |
|---|---|---|
| `<anc_uc>` | `{"actor_id":null,"object_type":"message","payload":{"dropped_parts":0,"text":"Do not modify the subscription state machine."},"role":"user"}` | `3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc` |
| `<anc_e1>` | `{"actor_id":"coder","object_type":"file_diff","payload":{"commit":null,"constraints":[],"intent":"fix checkout webhook retry","patch":"+ backoff = 2 ** attempt","path":"src/billing/retry.ts","summary":"add exponential backoff"},"role":"tool"}` | `a27b81718fae86837895813746ec188afca9087544df8950c3b5832742ebfa8d` |
| `<anc_e2>` | `{"actor_id":"coder","object_type":"test_log","payload":{"failed":3,"failures":[{"message":"state machine changed","name":"subscription_state_regression"}],"log":null,"passed":11,"related_paths":[],"status":"failed","suite":"billing_e2e"},"role":"tool"}` | `e51db27bfdf98d59d0abccc1fb59f7413e1f9bbaf1260406fdc2154e4b7ce1e2` |
| `<anc_e3>` | `{"actor_id":"coder","object_type":"file_diff","payload":{"commit":null,"constraints":[],"intent":"fix checkout webhook retry","patch":"+ idempotencyKey","path":"src/billing/idempotency.ts","summary":"add idempotency key"},"role":"tool"}` | `cae8b909c617c029c16e12f5ef04e0906be8a797b07eca94132db1654344fb2a` |
| `<anc_e4>` | `{"actor_id":"coder","object_type":"test_log","payload":{"failed":0,"failures":[],"log":null,"passed":14,"related_paths":[],"status":"passed","suite":"billing_e2e"},"role":"tool"}` | `ea5edcbd2c3f0b79ef01e3768590081cf1a024f8081b4c2b2d17155dd68ff272` |

---

## §20 文档与示例（分组 X）

### §20.1 `pyproject.toml`（必要部分，精确内容）

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

`tests/conftest.py` 通过 `pytest_addoption` 注册 `--update-golden`。

### §20.2 `README.md` 大纲

1. 安装（`pip install hms-elite`；extras）。2. 10 行快速上手（Mem0 形态的 `add/search`，然后是 `compile(...).to_text()`、`why`、`feedback`、`transitions`）。3. 「轴负责找、图负责继承、ContextIR 负责行动」（三段文字，一张图）。4. 从 Mem0 迁移：一张表——filters 完全相同；`update` ⇒ 修订（同一 id，新版本）；`history` 增加 `+anchor_id` 与额外事件；config 字典键的接受/重命名/移除情况；`vector_store` provider 为 `sqlite|memory`；feedback 枚举。5. 离线确定性（`rule` LLM、`hash` embedder）以及切换到 `openai`/`anthropic` 的方式（一个 config 字典）。6. 不变式（四条要点）及其执行方式（type / DDL / store）。7. API 参考表（Mem0 + HMS 原生，含 `context()` 与其记忆在前的片段，README §5.13）。8. 双语说明（EN/ZH docstring）。所有示例都可离线运行。

### §20.3 `examples/quickstart.py`（约 45 行，离线；不要求在冻结时钟下与 §19 逐字节一致——它跑在真实时钟上，但因为卡片使用显式 `as_of`，打印出的卡片文本相同）

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

预期 stdout（节选）：`S1: User likes black coffee. | history: ['ADD', 'UPDATE']`；`S2: ['Sarah shipped the webhook retry design on May 6.']`；`S3:` 三条 Sarah 事实，最新的在前；`work_state_card_failed.txt` 卡片逐字输出；`S5:` §19.5 的三行链条；`S6 feedback: ['archive', 'create', 'revision_edge']`；`S6 rollback: [<atom id>]`；transition 行 `patch src/billing/retry.ts: add exponential backoff -> negative | witnesses: 2`、`revise -> negative | witnesses: 2`、`rollback -> neutral | witnesses: 1`；consolidate 报告 `{"entity_atoms": 5, "temporal_edges": 0, "causal_edges": 0, "conflicts": 0, "interface_eps": 0, "projected_edges": 0, "interface_eps_l3": 0, "skills": 0, "transitions": 0, "compiled_states": 1}`（只有一个 EP → 不做 coarsen；五个实体 `retry.ts, file_diff, billing_e2e, test_log, subscription_state_regression`）以及各类边的计数。

---

## 变更日志（评审后）

下面列出两轮评审提出的每一个缺陷（已去重；重复提出的会标注）以及本次修订如何解决它。章节号指的是编辑之后的本文档。没有任何一个缺陷是靠删减契约内容来解决的。

| # | 严重级别 | 位置 | 解决方式 |
|---|---|---|---|
| 1 | 阻断（blocker，两次提出） | §19.6 中 `mid` 用 `as_of="2026-06-15T12:00:00Z"` 编译，与 `state_outdated` 后继者的 `valid_from=now` 以及 §8 的有效性 WHERE 冲突 | `mid` 改为**不带** `as_of` 编译（默认取 `store.now()`，即 feedback 瞬时之后的一个 tick）。§8 的有效性语义不变。§16.1 的 `state_outdated` 行现在说明：后继者仅在 `as_of >= feedback 瞬时` 时可见；§19 前言与测试 53 说明 `mid` 卡片按 slot 内容断言，绝不与黄金文件比对。DESIGN §12 已更新。 |
| 2 | 阻断 | §16.5 consolidate 的 `kappa_items` 导致 `mid.constraints` 出现 3–4 项（对已归档的 κ 文本产生了未去重的 `#kappa:` 条目） | §15.1 `kappa_items`：`selected_constraint_texts` = 已选约束 ∪ audit 条目的归一化文本；κ 条目按归一化文本跨 episode 去重（第一个 episode 胜出）。§15.2 第 6 步构建该集合。§19.6 声明 `mid.constraints` **恰好**为两项。 |
| 3 | 阻断 | §11.1 批量去重键 `(type, predicate, tuple(objects))` 把 S2 的两条 temporal fact 合并了 | 去重键改为 `(type, predicate, tuple(objects), normalize(text))`。 |
| 4 | 阻断（两次提出） | §9.2 `ACTION_RE` 的 target `[^,.;!?]*` 把 `retry.ts` 截断成 `a='patched retry'` | 重新定义 `ACTION_RE`：使用具名 `verb` 组，target 改为惰性匹配并以 `CLAUSE_CUT`（连词／分句分隔符／句末句点／串尾）终止；新增 `CAUSE_RE`（连词之后的触发子句）与 `CLAUSE_SPLIT_RE`（结果子句）。§11.2 重写，附带完整示例 `a='patched retry.ts to add backoff'`、`u='the webhook kept timing out'`、`y='billing_e2e failed'`；测试 19 对此断言。 |
| 5 | 阻断（两次提出） | §11.1 assistant 角色的线索正则只接受建议类／排期类句子，导致 `chat_action_closure` 产出不了 observation | assistant 门控改为 `ASSISTANT_CUE_RE` **或** `ACTION_RE`；此类句子成为 R9 fact，`subject='assistant'`、`attributed_to='assistant'`、文本形如 `Assistant …`。测试 26（「既有 observation 又有 EP」）保持不变；fixture 的 role 仍为 `assistant`。 |
| 6 | 阻断 | §12.6/§16.4 的 `ep_transition` 定义在分组 G，却被分组 C 调用 | `Transition.from_ep(...)`（以及 `from_interface_ep`）作为 classmethod 放在 `hms_elite/models.py`（分组 A）；`ops/transitions.py` 重新导出 `ep_transition = Transition.from_ep`。§0.4、§12.6、§16.4 已更新。 |
| 7 | 阻断 | §4/§4.3 的 `validate_payload` 归属分组 C，却被分组 A 的 `SourceEvent` 校验器需要 | `validate_payload` 在 `hms_elite/models.py`（分组 A）中声明，作为唯一实现；`ingest/events.py` 重新导出；§4.3 中的 `(ingest/events.py::validate_payload …)` 归属标注已删除。 |
| 8 | 阻断 + 重要（两次提出） | §0.4 与 §12.6 第 7 步的 `atom_to_row` 及 §13.3 的 `axis_ids`（C 引入 D） | `MemoryAtom.to_public_dict(...)` 是 models.py（A）中的规范实现；`mvmi/query.py::atom_to_row` 只是一行包装（无延迟导入）；`axis_ids` 从 §13.3 删除；§12.3 的 `neighbourhood` 改用 `store.valid_ids(scope)['atoms']`（§8）。§0.4 声明 C 永不导入 mvmi/ops，G 导入 B 的 rule 函数。 |
| 9 | 阻断 + 重要（两次提出） | §12.4 的 `store.last_open_action(scope, run)` 未在 §8 声明；DESIGN 里叫它 `last_open_action_event` | §8 声明 `last_open_action(self, scope, *, run_id, before, max_gap_hours) -> tuple[SourceEvent, Anchor] | None` 及精确的 SQL 语义；§12.4 用关键字参数调用；DESIGN §6.2 使用同一名称。 |
| 10 | 阻断 | §18.1 的 `golden` 会掩码所有 32 字符时间戳，与逐字节比对卡片冲突 | 拆成两个辅助函数：`golden.text(name, text)`（逐字节，不做掩码——用于卡片、链条）与 `golden.masked(name, obj)`（`mask_ids` + 只对 ≥ FrozenClock 起点的瞬时做掩码）。测试 47/48/55/63 各自写明所用的辅助函数。DESIGN §11 已更新。 |
| 11 | 阻断 | §18.3 测试 60 在离线环境下用原版 Mem0 字典（openai provider、qdrant 配置）构造 `HMS` | 测试 60 替换为 `llm=rule`、`embedder=hash`、`vector_store=sqlite/{}`（保留 `history_db_path`、`version`、`reranker`）；未经修改的 fixture 只被测试 10 使用（`HMSConfig.from_dict` → `CONFIG_002`）。 |
| 12 | 重要（两次提出） | §4 的 `GraphEdge.witness_anchor_ids: NonEmptyIds` 使 store 层的 `WitnessMissingError` 不可达（测试 12 与测试 3 冲突） | `SupportGraph.insert_edge` 在构造模型**之前**对 `[]` 抛出 `WitnessMissingError`；`StateStore.insert_edge(edge: GraphEdge | dict)` 在 `GraphEdge(**edge)` 之前检查 dict，并对模型实例（包括 `model_construct` 构造的）重新检查；测试 12 用 `store.trace.tables() == set()` 覆盖全部三条路径。测试 3（pydantic 拒绝 `[]`）依然成立。DESIGN §1 承诺 1 已重新措辞。 |
| 13 | 重要 | §5 的 `CONFIG_002`/`CONFIG_003` 在 pydantic 校验器内部抛出（被转成了 `CONFIG_001`） | `HMSConfig.from_dict` 在 `HMSConfig(**data)` **之前**先跑纯 Python 前置检查 `normalize_mem0_keys(strict=True)`（→ `CONFIG_003`）与 `check_providers`（→ `CONFIG_002`），并把在 `e.errors()[i]['ctx']['error']` 中找到的任何 `HMSError` 原样重抛；before-validator 只重映射 RENAMED 键；`_known_provider` 对直接构造抛出普通 `ValueError`。§5.1/§5.2/§5.3 已更新；测试 10 注明了前置检查。 |
| 14 | 重要 + 次要（三次提出） | `Invariant.anchor_ids`/`confidence` 以及 `Residual` 的 anchor 从未被任何生产者推导；`kappa.evidence` 为空；IEP atom 的 `source_ep_ids`/`topic_tags` 未指定 | §14.2 coarsen：`support_node_ids` 取自 refs，`anchor_ids = sorted(∪ atom_anchor_ids)`，`confidence = round(mean, 4)`；新增 `Residual.anchor_ids` 字段（§4）；`slots.kappa.evidence = evidence_anchor_ids`；定义 IEP atom 的 `source_ep_ids`、`topic_tags`（有序并集）、`entity_ids`（有序并集）。§19.7 断言 skill atom 的 `anchor_ids`/`confidence`；测试 41/42/43 相应扩展。`atom_from_ep` 现在携带 `confidence=ep.confidence`，使数值可确定。 |
| 15 | 重要（两次提出） | §19.4 的 `entity_boost == 1.0` 与度为 3 时的 hub 阻尼冲突 | 阻尼公式改为 `1/(1 + 0.001·max(0, deg−3)²)`（§13.4，DESIGN §8）：度 ≤ 3 时精确得 1.0，因此 §19.4 的断言是精确的；测试 36 断言度为 5 时 `< 1.0`。 |
| 16 | 重要（两次提出） | §11.1 的 `supersedes_hint_ids`（要求 ≥2 个共享 token）对 S1 不可达（期望 `["0"]`） | R1 的提示规则：已有文本以 `User likes|prefers|dislikes` 开头，且（共享 ≥1 个实义 token **或** 共享 ≥1 个 `TOPIC_TABLE` 主题）；其他规则仍要求 ≥2 个 token。S1 共享 `like` 和主题 `coffee`。 |
| 17 | 重要 | §6.2 的 edge prompt 中 `## Nodes` 缺少 §11.4 用到的 `b`、`subject`、`predicate`、`objects` | `## Nodes` 的 payload 为 `{ref, kind, u, b, a, y, r, subject, predicate, objects, time, entities, topics, anchor_refs}`（每个键始终存在；定义了 EP 与 observation 两种形态）；`EDGE_PROPOSAL_PROMPT` 的 `# INPUTS` 与之对应；§11.4 说明节点形态。 |
| 18 | 重要 + 次要（两次提出） | §12.5 的 `link_atom_axes` 三元组 `(normalized, canonical, entity_type)`——FILE 身份有歧义、没有别名槽位；读取方未 hydrate `entity_ids`/`topic_tags` | `EntityKey = (key, canonical, entity_type, aliases)`；`insert_atom(entity_keys: list[EntityKey])` 传入 `normalized=key, aliases=aliases`；§9.2 声明 `EntityCandidate.key` 即身份；§8 新增「Hydration」说明：每个读取方都要填充 `entity_ids`（键）、`topic_tags`、`anchor_ids`。 |
| 19 | 重要 | §15.1 规则 (1) 把 `expiration_date` 在未来的存活 atom 也路由到了 `audit` | 规则 (1) 改为 `archived_reason is not None or (valid_to is not None and valid_to <= as_of)`；签名改为 `assign_slot(atom, *, as_of, reason=None)`；§15.2 传入 `as_of`；测试 46 覆盖两种情形。相关：`resolve_head`/`ux_atoms_head` 以 `archived_reason IS NULL` 为键，使 `get()` 对将过期的记忆仍然可用。 |
| 20 | 重要 | §18.3 测试 27 需要一条背景约束，但不能调用 LLM | 测试 27 通过 `ingest.events.anchor_events` + `ingest.atoms.create_atom(atom_from_observation(Observation(type='constraint', …)))` 播种该约束，并且只在 `add_events` 前后断言 `hms_scripted.llm.calls == []`。 |
| 21 | 重要（两次提出） | §18.3 测试 24 / DESIGN §3.2 声称「完全相同的重复 add 会新增一条 support atom_anchor 行」（主键冲突） | `add_atom_anchor` 改为 `INSERT OR IGNORE` 并返回 bool（§8）；测试 24 断言：同一时间戳的重复 add 下 `atom_anchor_ids` 不变且 `mentions == 2`，只有更晚时间戳的重复 add 才产生 support 行；DESIGN §3.2 相应行重写；§12.6 第 6 步注明该操作为 no-op。 |
| 22 | 次要（三次提出） | §19.3 中 card1 写的是「丢弃 4 个事件 atom」 | card1 为 2（`dropped_covered == 2`）；card2 为 `dropped_covered == 4`；测试 47/48 对此断言。 |
| 23 | 重要 + 次要（两次提出） | §16.5 步骤 (1)/(2) 的「缺少边」表述与 §19.7 的 `mentions == 2` 冲突 | 步骤 (1)/(2) 总是调用 `insert_edge`（MERGE 会递增 `mentions`）；计数只统计调用前尚不存在的边 id。§19.7 与测试 56 明确写出 `temporal_edges == causal_edges == 0`、`mentions == 2`。 |
| 24 | 次要 | §4 的 `RESERVED_ATOM_KEYS` 缺少 `source_ep_ids_json` | 该集合改为由模型字段 ∪ `{f + '_json' for f in ('content','user_meta','source_ep_ids')}` ∪ 显式补充项推导得出；测试 39 断言其 ⊇ `PRAGMA table_info` 的字段名。 |
| 25 | 次要 | §13.3 第 1 步调用 `parse_filters(aq.meta_filters)` 时用了默认的 `require_scope=True`，且元组使用有误 | 改为 `_, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context='meta', force_meta=True)`；§13.1 增加 `force_meta` 关键字参数。 |
| 26 | 次要 + 重要（三次提出） | `observation_date`/`current_date` 未定义；ingest 阶段 REVISE 的 `at` 未定义（S1 期望 `valid_to` 为事件时间） | §12.3/§12.6：`observation_date = date_of(events[-1].timestamp)`、`current_date = date_of(store.now())`，每条消息的 `timestamp` 即该消息的 `base_date`（§11.1）；§12.5 `revise_atom`：ingest 传入 `at = max(new.valid_from, old.valid_from)`（事件时间）用于 `valid_to` 与历史 `updated_at`；`update()`/feedback 传入 `store.now()`。§19.1 断言 UPDATE 行的 `updated_at`。 |
| 27 | 次要（两次提出） | 仅限关键字的签名被按位置调用；`build_state_package` 缺 `config`；`history_event=None`；`StateStore(...)` 按位置传参 | §12.4 改为 `store.last_ep(scope, run_id=run)`；§16.3 改为 `build_state_package(..., config: HMSConfig, ...)` 并以关键字方式调用 `compiler.compile(...)`；`create_atom(history_event: str | None)`；§17.1 `StateStore(path, busy_timeout_ms=…, journal_mode=…)`。 |
| 28 | 次要 | §16.1 的 `connection_reason_wrong` 在边类型相同时会 MERGE 进已失效的行 | 类型相同 → 不作废：改为 `revalidate_edge` + `add_edge_witness(fb_anchor)` + `set_edge_label`（§8 新增方法），报告为 `{'action': 'insert_edge', 'edge_id': old_id, 'merged': True}`；类型不同则新建一行。测试 52 增加对应子用例。 |
| 29 | 次要 + 重要（两次提出） | 没有 target 的 `state_missing` 不可达；§4.2 的 feedback 行禁止 `preference_event` atom | 改为 `process(*, target_id: str | None, …)`、`HMS.feedback(target_id: str | None, …)`；前言说明：对 `state_missing` 而言，无法解析或为 None 的 target 是合法的——scope 取自 `correction['scope']`、一个 `Scope.key()` 字符串或 `scope_hint`；`FeedbackEvent.target_kind` 新增 `anchor`/`scope`（DDL CHECK 同步更新）；§4.2 的 feedback 行允许 `obs_type ∈ {constraint, fact, preference_event}`。测试 52 增加子用例 `state_missing_no_target`。 |
| 30 | 次要 | `compiler.default_mode` 从未被读取 | 改为 `HMS.compile(mode: str | None = None)` 与 `StateCompiler.compile(mode=None)`；§15.2 第 1 步解析 `mode = mode or config.compiler.default_mode`；测试 46/57 覆盖。 |
| 31 | 次要（两次提出） | Skill atom（`L3_macro`）被 recall/temporal 的 `scale_default` 排除；增补阶段忽略 skill | `L3_macro` 加入 recall/temporal/entity 的 `scale_default`；§15.2 第 3 步并入有效的 `source_kind='skill'` atom；§19.7 断言 consolidate 之后的 `compile(...).skills`。 |
| 32 | 次要（两次提出） | 测试 56 的 prune 场景中没有任何 atom 低于 `confidence_min` | 测试 56 在 `clock.tick(days=400)` 之前先执行 `store.set_atom_confidence(<atm_e1>, 0.2)`；预期 `archived == [<atm_e1>]`、`decayed == 3`（配合新的衰减规则，见 #46）。 |
| 33 | 次要 + 重要（两次提出） | `close_structured_events` 缺少 `temporal_window_hours`；payload 中的 `constraints` 从未进入 `b`/`kappa.constraints` | 签名新增 `temporal_window_hours`（流水线传入 `config.graph.temporal_window_hours`）；`build_ep_from_closure` 用 `constraints = ordered_union(background texts, payload constraints)` 生成 `b` 与 `kappa.constraints`，使自闭合的 tool_call 也能通过 `Slots` 校验器。测试 27 对此断言。 |
| 34 | 重要 | `AxisQuery` 的语义（project_axis / aggregation）未被执行；`scope` 为必填 | 新增 `AxisResult(hits, projection, count, summary, …)` 模型（§4）；§13.3 第 5 步定义每一种 `project_axis` 与 `aggregation`（`count` = 一条 SQL COUNT，另有 `latest`、`summary`，`topk` 需要 `text` → 否则 `VALIDATION_013`）；`AxisQuery.scope` 改为可选（由 filters 推导）；新增 `HMS.axis_query()` 门面；测试 35 扩展；§19.1 给出 HMS §3.5 的偏好示例。 |
| 35 | 重要 | `entity`/`relation` 两个尺度是死的（没有实体 atom；relation atom 默认关闭且不在 scale 默认集中） | 新增 `SourceKind.entity`/`NodeKind.entity`；§4.2 增加 `entity` content 行；新增 `atom_from_entity` 构建器（§12.5）；consolidate 步骤 (0) `entity_atoms` 在实体身份 scope 下为每个实体物化一个 atom，当 mentions/aliases 变化时进行修订；`emit_relation_atoms` 默认改为 True；`entity`/`relation` 加入 recall/entity 的 `scale_default`；新增 `graph_stats.nodes.entity`；新增测试 56c；§19.7 期望 `entity_atoms == 6` 并解释其 scope。实体 atom 永不进入卡片（§15.2 第 2 步）。 |
| 36 | 重要 | `PREF_QUERY_RE → recall` 是空操作（偏好召回不是一次轴选择） | 新增 `Route.preference`，`hard_type_gate = ('obs_type', ['preference_event','habit'])`、`scale_default=['state']`；`execute_axis_query` 第 2 步按常规回退策略应用它；`primary_axis='type'`；新增 `score_details.axis_gates.obs_type`；测试 34/36；§19.1 断言该 route。 |
| 37 | 重要 | EP 的有效性从未被维护（`archive_ep` 从不被调用；checkpoint 中没有 `ep_state`；`list_eps` 会复活已归档的 EP） | §8 的 `archive_atom`/`restore_atom` 级联到 `eps.valid_to`（以及 `observations.valid_to`、`interface_eps.valid`）；新增 `restore_ep`、`scope_ep_state`、`set_observation_valid_to`；`list_eps(valid_only)` = `eps.valid_to IS NULL AND node valid`；新增 `Checkpoint.ep_state` + 快照哈希 + rollback 步骤 (2b)；测试 53 覆盖「删除 → 交接目标 → 回滚」。 |
| 38 | 重要 | 没有撤销路径（`anchors.revoked_at` 从未被设置；无级联） | §16.1 新增 `revoke(store, graph, *, anchor_id, reason)`，带见证级联（dispute → 无 witness 时归档 → 作废边/IEP），新增 `HMS.revoke()`，`feedback(<anchor_id>, 'evidence_wrong', correction={'revoke': True})` 委派给它，`system` payload 增加 `op='revoke'`；测试 56b 与测试 51（`not_revoked`）使用它。 |
| 39 | 重要 | InterfaceEP 从不产出 transition | 新增 `TransitionKind.interface_ep`、DDL CHECK、`KIND_ORDER`（现位于 enums，分组 A——同时解决「KIND_ORDER 声明在 G」这一次要问题）；新增 `Transition.from_interface_ep`；`export_transitions` 增加步骤 (a)；§19.7 `transitions == 1`、`st['transitions'] == 6`；测试 55 检查该行。 |
| 40 | 重要 | L3 InterfaceEP 不可达（`P = nodes with kind ep`；调度器只对 EP 组件做 coarsen） | §11.5 的 `P` 改为 kind ∈ `{ep, interface_ep}`；`components(..., scale=)` 增加参数；consolidate 步骤 (4b) 对 L2 IEP 组件执行 `coarsen(target_scale='L3_macro')` → `project`；新增 `GraphConfig.coarsen_l3`；报告新增键 `interface_eps_l3`；新增测试 45b。DESIGN §14 风险 3 已更新。 |
| 41 | 重要 | 治理只在选择之后才检查（团队交接不可用；`visibility` 过滤被降级为相等匹配） | §13.1 对标量 `visibility` 过滤引入格（lattice）门控；§15.2 在轴过滤中保留该键，并对图邻居、invariant 与证据 anchor 做门控（`dropped_governance`）；§15.4 的 `governance` 覆盖 κ 省略与 anchor 的 `permission_scope`；测试 51 的治理用例改为（private 条目不被选中，义务判为 satisfied；手工构造的 IR 则违反）。DESIGN §6.3/§6.4/§14 已更新。 |
| 42 | 重要 | 交接包无法导入；缺少 `source_events` | 新增 `StatePackage.source_events` 与 `atoms`（§4，校验器检查证据闭包）；新增 `import_state_package(...)`（§16.3），支持保留 id、按身份去重、可选重新划定 scope；新增 `HMS.import_package()`；`system` payload 增加 `op='import'`/`package_id`；测试 54b 做往返验证（在接收端比对卡片是否相等）。 |
| 43 | 次要 | `sensor_trace` 对象类型被漏掉；`EventType.tool_result` 从未被赋值 | 新增 `ObjectType.sensor_trace` 及其归一化 payload（§4.3），结构化路由仅产出 `event` atom，raw_ref 为 `sensor://`，并有 `event_atom_text`；`EXTRACTABLE_OBJECT_TYPES` 有五个成员（DESIGN §2 第 3 行现已列出）；`events_from_dicts` 为 `test_log`、`sensor_trace` 以及带 `status`/`result` 的 `tool_call` 设置 `event_type='tool_result'`。 |
| 44 | 次要 | §0.3 的 docstring 模板把 HMS §5.2 每个参数两行的排版压扁了；tests/examples 被豁免 | §0.3 按 HMS §5.2 逐字重写（每个参数一行英文 + 一行中文，用全角冒号，参数之间空一行，Output/State effect 下同时给出 EN+ZH）；适用范围包含 `tests/**/*.py` 与 `examples/*.py`（§1）；新增测试 64 `test_bilingual_docstrings`。 |
| 45 | 次要 | 调度器任务 `compile_state` 缺失 | consolidate 步骤 (7)：对每个 scope 执行 `compiler.compile(mode='deep')` 并持久化到新表 `compiled_states`（DDL，`TABLES` 现为 22，id 前缀 `cst`）+ `meta['work_state:<scope>']`；`orient` 报告 `last_compiled`；新增 `SchedulerConfig.compile_state`；`StateScheduler.__init__` 接收 `compiler`；报告新增键 `compiled_states`；`compile()` 本身保持只读。 |
| 46 | 次要 | prune 把衰减与归档混为一谈 | prune 拆成 (a) 衰减（`confidence − decay_rate·staleness`，下限 `confidence_min/2`，写 `UPDATE` 历史）与 (b) 低于 `confidence_min` 时归档；新增 `SchedulerConfig.decay_rate = 0.1`；报告新增 `decayed`。采用减法步骤（而非建议的 `×(1−0.5·staleness)`）是为了让 S4 中置信度 0.7 的事件 atom 保持在归档线之上，从而使测试 56 的 `archived == [<atm_e1>]` 精确成立。DESIGN §6.9/§8 已更新。 |
| 47 | 次要 | `kappa_t` 的含义随 transition 类型而异 | 新增 `kappa_t_of(...)`（§4），使每种类型都具有 `{constraints, visibility, permission_scope, provenance}` 这一形态；§16.2/§16.4 定义各类型的具体内容；§19.6 的第 3–5 行重新表述（`provenance` 承载原有的键）；测试 55 断言键集合。 |
| 48 | 次要 | Skill 的 `distill` 边无法追溯到底层 `use` 边 | `promote_skills` 把 `witness_edge_id` 设为 support_node → constraint atom 之间 id 最小的有效 `use` 边，其余通过 `add_projection` 记录；新增 `graph_stats.traceable_high_edges`；§19.7 与测试 43 断言这些 projection 行。 |
| 49 | 次要 | `procedural_summary` 与「OUTPUT FORMAT 一律为 JSON」的通则冲突 | §6.1 增加例外：它的 OUTPUT FORMAT 是 §6.3 的 markdown 骨架 + `Return ONLY the markdown above. No JSON, no wrappers.`；测试 9 断言各级标题。 |
| 50 | 次要 + 重要（两次提出） | 带 disputed anchor 的条目会在 `evidence_coverage` 生效之前先破坏 `ContextItem`（`anchor_ids=[]`） | 条目携带 `atom_anchor_ids(include_disputed=True)` 以及 `explain['disputed_anchor_ids']`；`evidence_coverage`（与 `not_revoked`）只在未被 dispute 的子集上求值；测试 51 写明可观察行为。 |
| 51 | 次要 | 没有规范性的范围表；消融配置未做映射 | 新增 §0.5（第 1–20 行取自 DESIGN §2，外加「其余一切均在范围内」一行）；新增 `CompileMode.axis_graph`、`ABLATION_OF_MODE`、`retrieval['ablation']`；§15.2 第 4 步区分 `axis_graph`（不做 `refine`）与 `deep`；§19.3/§19.7 增加消融断言；测试 63 运行这三种模式。 |
| 52 | 重要 | §16.4 的 `w_t` 用了排序，与 §19.6 的顺序 `[e3, e2, e4]` 冲突 | 改为 `w_t = ordered_union(edge_witnesses, j.anchor_ids)`（按首次出现顺序）；§0.1 的确定性一行区分有序并集与已排序集合；`ordered_union` 在 §9.2 中声明。 |
| 53 | 重要 | 测试 40 的容差 `0.6667 ± 1e-6` | 改为 `pytest.approx(2/3, abs=1e-6)`；`cohesion_score` 不做四舍五入存储；测试 41 与 §19.7 使用 `approx(2/3, abs=1e-4)`。 |
| 54 | 重要 | 分组 G 被要求在进程内应用 B 的规则表，却没有可导入的函数 | §11 声明 `hms_elite/llms/rule.py` 中的公开对象 `decide_revision`、`propose_edges`、`coarsen_rule`、`EXCLUSIVE`、`PREDICATE_FAMILIES`、`family`；§16.5 步骤 (2)/(3) 与 §14.2 调用它们。 |
| 55 | 重要 | §11.1 用 `^` 锚定的第一人称判断，且未说明 search 与 match 的区别 | 新增 `is_first_person()`（§9.2，使用 `re.search`）；§11 前言：除非正则以 `^` 开头，否则一律用 `re.search` 应用。 |
| 56 | 重要 | §3 基于消息文本的 `WitnessMissingError` 映射（SQLite 外键错误没有名字） | 每个接收 anchor 的写入方都做显式前置检查（`get_anchors`、`details['missing']`）；只有 `NOT NULL … *anchor_id` 这条消息作为最后手段被映射。测试 12 断言未知 id 的路径。 |
| 57 | 次要 | `stats(scope: Scope)` 不可为 None；节点计数的有效性语义未指定 | 改为 `stats(scope: Scope | None)`；计数覆盖所有行，不论 `valid`；`valid_edges`/`invalid_edges` 是唯一按有效性拆分的项。 |
| 58 | 次要 | `ix_atoms_head` 不是 UNIQUE，与 DESIGN 的唯一性声明冲突 | 改为 `CREATE UNIQUE INDEX ux_atoms_head ON memory_atoms(memory_id) WHERE archived_reason IS NULL`（以 `archived_reason` 而非 `valid_to` 为条件，使将过期的 head 仍是唯一 head）；`resolve_head` 与 `LIVE_SQL` 据此定义；测试 11 断言该索引；DESIGN §1 承诺 4 已更新。 |
| 59 | 次要 | `source_events.role` 的 CHECK 与由枚举驱动的生成方式冲突 | §7 说明两处收窄（`role` 不含 `extracted`；`eps.scale` 为字面量）；测试 11 检查这一点。 |
| 60 | 次要 | `ScriptedLLM` 不支持可调用的 response（测试 31） | 接受 `Callable[[list[dict]], str | dict]`（§11.7）；测试 31 重写。 |
| 61 | 次要 | 没有供测试使用的公开 `HMS` 属性 | §17.1 声明 `config, store, index, embedder, llm, graph, ingestor, compiler, feedback_processor, scheduler`；测试引用 `hms.llm.calls`、`hms.store.trace`。 |
| 62 | 次要 | 测试 17 按 provider 参数化，却断言两者一致 | 测试 17 改为在一个不参数化的测试中同时构造两种 index。 |
| 63 | 次要 | §11.5 的 `b` 拼接会重复约束文本 | 改为 `b = ' ; '.join(ordered_union(split(p.b, ' ; ')…, kappa.constraints))`；§19.7 断言每条文本只出现一次；DESIGN §7 已更新。 |
| 64 | 主要 | §11.1 的 R1 把 `moved to` 列入偏好动词：`'I moved to Berlin.'` 在 R5 之前命中 R1，被判成 `likes Berlin` / `preference_event`，导致搬家永远不会修订此前的 `lives_in` | 从 R1 的 `PREF_RE` 动词组中移除 `moved to`（保留 `switched to|switch to`）；R5 新增搬迁孪生 `MOVED_RE`/`MOVED_ZH_RE` → `lives_in` 且 `relation=supersession`，仅在居住/工作模式未命中时才尝试；`_SWITCH_VERBS` 现为 R1 与 R5 共享的词表（新增 `moved to`、`relocated to`、`搬到`、`搬去`、`搬回`）；随后 §11.3 规则 6 会修订此前的居住地。`SUPERSESSION_CUES_EN` 保持不变（`moved to` 仍是一个线索词）。测试 18b 与 30b。 |
| 65 | 主要 | §13.1 只从顶层读取 scope 键，导致 Mem0 用户常写的展平 `AND` 形式（`{'AND': [{'user_id': 'alice'}, {'OR': [...]}]}`）抛出 `VALIDATION_001`，而 Mem0 的 `_process_metadata_filters` 会在读取 scope 前先展平 `AND`（MEM0_MAP §2.2 第 5 步） | §13.1 的 `parse_filters` 在构造 `Scope` 之前，把顶层 `AND` 链中的**标量** scope 键上提（递归穿过嵌套 `AND`）：以首次出现者为准，顶层已定义该键则不上提，被清空的子句与被清空的 `AND` 会被丢弃，`OR`/`NOT` 分支以及列表 / 运算符字典 / `'*'` 形式保持原样（仍是列过滤叶节点，没有固定值时仍抛 `VALIDATION_001`），且调用方的字典绝不被修改。测试 33c 与 33d。 |
| 66 | 主要 | §10.1 的 `AnthropicLLM` 总是把 `temperature` 传给 `messages.create`，但 `anthropic>=0.40` 的 pin（§20.1）会解析到 1.x SDK，其 `Messages.create` 已不再接受采样参数，于是每次调用都在任何请求之前抛出 `TypeError`（表现为 `LLM_001`） | 仅当已安装 SDK 的 `Messages.create` 声明 `temperature` 时才转发它（构造时 `inspect.signature`）；pin 保持不变。测试 22b。 |
| 67 | 主要 | §12.3 对 LLM 字段做盲目规整且放在逐条守卫之外：列表字段里的字符串被逐个字符迭代（`objects: "black coffee"` → `['b','l',…]`；`kappa.constraints` 同理，而且这种字符列表还能满足 `Slots._closure`），非字符串的文本字段被存成 Python repr，列表字段出现标量则在事务 A 之后以原生 `TypeError` 逃出 `HMS.add()` | 列表字段接受列表或单个字符串（→ 单元素）；文本字段与槽位必须是字符串；规整在守卫之内进行，畸形条目跳过并 `logger.warning`（§12.3）。测试 32b、32c。 |
| 68 | 主要 | §8 的 `_execute` 在调用方取行之前就释放了锁，导致同一 `StateStore` 的并发读取方（`check_same_thread=False` 的设计）经 CPython 语句缓存拿到别的语句的行（IndexError / pydantic ValidationError / 结果缺行） | `_execute` 在锁内取完所有行并返回已物化的行列表；活动游标不再离开锁（§8）。测试 15b。 |
| 69 | 次要 | README §3 把 `store: {path: ~/.hms_elite/hms.db}` 写成"默认值"，但 `db_path()` 原样返回 `store.path`（§5，与 Mem0 一致），加载该 YAML 会在 cwd 下创建字面的 `./~/.hms_elite/` | README §3（中英）省略 `path`（→ `default_db_path()`）并注明 `~` / `$VAR` 不会展开；`test_readme_config_yaml_is_the_defaults` 经 `HMSConfig.from_dict` 加载该代码块。§5 不变。 |
| 70 | 次要 | README §12 的 `hms why … --run run-billing` 按原文无法成功（`run-billing` 为空 → `NOT_FOUND_001`，正是 §17.3 的规定），且 `cli.py` 让 `HMSError` 以回溯逃出 | README §12（中英）先给 `run-billing` 灌入一条约束；§17.6：`HMSError` → `to_dict()` JSON 到 stderr、退出码 1；行数上限放宽到 ≤ 90。测试 62b、62c。 |
| 71 | 主要 | §16.5 consolidate 第 (2) 步在 `causal_proposer='llm'` 时于本步事务内无守卫地迭代 `witness_refs`，标量（`7`）、嵌套列表或对象列表会抛出原生 `TypeError`，让所有目标 scope 的 `consolidate()` 一并中止（不写 `last_consolidation` meta、prune 不再执行），而 §6.2 明文要求"否则丢弃" | `_causal_edges` 只接受字符串列表形式的 `witness_refs`；其余一律在守卫之前以 `logger.warning` 丢弃该提议（§6.2 与 §16.5 第 2 步重述）。测试 56d。 |
| 72 | 主要 | §14.2 的 `_merge_proposal` 对槽位 `r` 与 `kappa.visibility` 接受任意字符串，LLM 输出的枚举外取值（`'great'`、`'everyone'`）会在 coarsen 事务内由 `Slots`/`Kappa` 抛出 `pydantic.ValidationError`，让 `consolidate()` 中止而非回退到规则值 | `r` 必须 ∈ `Reward.values()`（否则取规则 `r` 并告警）；`kappa.visibility` 缺失或 ∉ `Visibility.values()` → 采用规则可见性、保留 LLM 的约束（§14.2 重述）。测试 45c。 |
| 73 | 次要 | §16.5 第 (2) 步用 `max(0.0, min(1.0, x))` 钳位 `confidence`，会把 `NaN`（`json.loads` 接受该字面量，字符串 `'nan'` 经 `float()` 亦然）变成 1.0——垃圾置信度以最大值落库，且经 `MAX(confidence, ?)` 的 MERGE 不可逆 | `_proposal_confidence`：无法解析或非有限 → 0.7 并告警，然后再钳位（§6.2 与 §16.5 第 2 步）。测试 56d。 |
| 74 | 主要 | §13.3 第 3 步对**整个**门控集合 G 打分：`semantic_scores` 向索引索要每个候选（`index.search(candidate_ids=G, top_k=len(G))`），逐个水合并计算其余四路信号——每次查询 O(N)（每候选约 45 µs：N=3000 时 133 ms，N=30000 时 1.6 s；RETRIEVAL_DIAGNOSIS T2/T3）——而 `ScoringConfig.max_candidates=2000` 又把最新 2000 条之外的记忆不分相关性地静默丢弃（正确性缺陷，不只是成本） | §13.3 第 3 步改为有界多通道召回入池（`recall_pool` / `Pool`：semantic / keyword / entity / recency 通道各带 K = `RetrievalConfig.k_for(top_k)`、整体入池规则、截到 `pool_max` 的轮转并集、只水合池），回退阶梯每级走同一套召回；`RetrievalConfig`（§5.1 / §5.2，`HMSConfig.retrieval`，由 `HMS.search` 与 `compile` 传入）；`max_candidates` 是已弃用的 None 哨兵别名，只告警、不截断；`safety_limit`（50000）只告警、绝不截断；`score_details` 新增 `recalled_by / pool_size / channel_sizes`（§13.4），`AxisResult.retrieval` 携带 `Pool.stats()`（§4）。等价性定理写在 §13.3 并由 64a 钉住（G 不超过通道上限时与穷举路径逐位相同）；64b–64d 覆盖 N=500 重叠率、无截断、溯源键、安全上限与 Python 侧过滤。RETRIEVAL_BENCH_2026-09-10：sqlite search@30000 从 172–210 ms（截断）/ 1.6 s（不截断）降到 13.6–19.6 ms。 |
| 75 | 主要 | §10.3 的 `IndexBase.search` 只接受 `candidate_ids`，于是具备下推能力的 Milvus 提供者收到的是 id 列表（每块最多 1024 个 id、13–30 KB 表达式）且 `limit=len(chunk)`，HNSW 退化为暴力扫描（每块约 58 ms，T4–T6），ANN 完全没有剪枝 | §10.3：`IndexFilter`（scope、`as_of` 有效性、scale / obs_type / atom_type 白名单、闭区间时间界）、`supports_filter` 提供者上的 `search(filter=)` / `count(filter=)`、`INDEX_ATTR_KEYS` 标量及 `atom_index_attrs` / `normalize_attrs`、`upsert(attrs=)`、非抽象钩子 `update_attrs` / `close` 与 SQL 退路 `gate_ids`；`MilvusIndex` 保存标量（INVERTED 索引）、渲染 `filter_expr`、过滤时提高 `ef`（T6 召回缺口）、重开时 `load_collection`、热路径从不计数；`SQLiteIndex` 新增以写入代数为键的进程内矩阵 + 标量缓存与 `select_top_k`（与 `rank_hits` 逐位相同）；`MemoryIndex` 经缓存的矩阵快照排序。`tests/test_index_milvus.py`（原生过滤 == SQL 门控、标量模式守卫、默认标量索引、从 store 读标量、重开），64g。 |
| 76 | 主要 | §8 没有给保存标量的提供者任何跟随有效期 / 归档变化的途径，因此已归档原子在下一次 upsert 之前仍可被 Milvus 搜到 | §8 `StateStore.attr_listeners`：`archive_atom` / `restore_atom` / `set_valid_to` 把原子排队，COMMIT 后在 store 锁之外按票号（提交）顺序派发**完整**的 `atom_attrs` 快照；`HMS.__init__` 注册 `index.update_attrs`，`HMS.close` 在 `index.close()` 之前移除它（§17.2）；水合后仍在 SQL 里复核有效期，因此过期标量只会藏起一行、绝不会多出一行。64e（`tests/test_index_sync.py`）与 `tests/test_index_milvus.py::test_archive_hides_from_every_provider_and_rollback_restores`（sqlite / memory / milvus）。 |
| 77 | 主要 | §7 没有词法索引，关键词通道无法有界：关键词信号只在水合之后才存在 | §7.1 模式版本 3：FTS5 虚拟表 `atom_fts(atom_id UNINDEXED, tokens)`，由 `fts_tokens` 喂入（项目自己的分词器，因此 CJK 二元组与标识符在 `unicode61` 下可检索），`upgrade_3` + `backfill_fts`（`POST_MIGRATIONS`），22 项 `TABLES` 旁的 `VIRTUAL_TABLES` / `FTS_SHADOW_TABLES`，`capabilities['fts5']`，`drop_all` 先删虚拟表；全新建库路径与 v1 → v2 → v3 阶梯的 `sqlite_master` 行相同。§8：`keyword_candidates`（G 内的两阶段 bm25，有界的 `LIKE` 退路）、`entity_candidates`、`axis_ids(order=None)`、`axis_select(atom_ids=)`（VALUES-CTE）、`index_hint`（`TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX`）、写入代数 `data_generation / rebuild_generation / data_version()` 与 `':memory:'` 的 `clone_schema`。`test_schema_migrations_and_capabilities`（22 基表 + 1 虚拟表）、`test_fresh_database_is_created_directly_at_schema_version`、`test_schema_v3_backfills_fts_from_a_v2_file`、64g、`tests/test_perf_paths.py`。 |
| 78 | 次要 | §15.2 第 2 步自己扫候选（`candidate_k=50`，阈值 0）而不走检索召回，`search()` 有界之后 `compile()` 本会仍是 O(N) | §15.2 第 2 步调用 `execute_axis_query(..., retrieval=config.retrieval)` 并把 `Pool.stats()` 复制进 `ContextIR.retrieval`（`test_compile_candidates_reuse_the_pool`）。 |
| 79 | 次要 | §0.5 第 1 / 2 / 4 / 5 / 11 / 15 / 18 行对已发布的模块（`daemon.py`、`runtime.py`、`segmentation.py`、`compiler/coverage.py` + `scc.py`、`state/`、`eval/`、`AsyncHMS` + `telemetry.py`）仍写着砍掉 / 简化，§1 未列出这些文件，§5.1 / §5.2 / §5.4 缺少 `milvus` 索引与 `deepseek` / `ollama` / `fastembed` 提供者，§17 没有 `AsyncHMS` / 遥测的条目 | §0.5 各行改写为"已发布"并给出入口；§1 列出文件；§5.1 / §5.2 / §5.4 补全提供者集合与 `EXTRA_OF_PROVIDER`；新增 §17.8。README §4.1 / §10 / §10.1 / §11 / §13（中英）与 DESIGN §6.3 / §6.4 相应改写。 |
| 80 | 次要 | §18.3 有两组编号 `64e` | `tests/test_retrieval_v3.py` 那组改为 `64g`。 |
| 81 | 次要 | `IngestConfig` / `CompilerConfig` / `SchedulerConfig` 没有已接线模块的键，`extraction_mode`、第 8 步打包器与守护进程间隔只能在构造时给 | §5 `ingest.extraction_mode`（'single' \| 'staged'，由 `HMS.__init__` 透传给 `Ingestor`）、`ingest.segment_max_chars`（400）、`compiler.selection`（'shares' \| 'coverage'）、`scheduler.daemon_interval_s`（None；设置时 > 0）。RENAMED / REMOVED 不变。`tests/test_wiring.py::test_config_fields_reach_their_collaborators`。 |
| 82 | 次要 | 遥测只能经 `telemetry.usage_of(hms)` / `telemetry.trace` 与 `AsyncHMS` 到达 | §17.3 `HMS.usage() -> Usage` 与 `HMS.trace(operation, request_id=None)`；`AsyncHMS.usage()` 委托之；`GET /usage` 调用 `hms.usage()`。`test_facade_usage_and_trace`、`tests/test_server.py::test_usage_and_metrics`。 |
| 83 | 次要 | §11 状态卡只存在于门面之外的 `hms_elite.state.StateAPI(hms)` | §17.3 `HMS.state`（惰性 `StateAPI`）与十二个同名委托，签名逐字同 `StateAPI`；`AsyncHMS` 镜像（`tests/test_async.py`：40 个签名）。`test_state_property_and_delegates_mirror_state_api`。 |
| 84 | 次要 | `StateDaemon` 与 `AgentRuntime` 只能从各自模块导入；门面上没有任何东西启动或停止守护进程 | §17.4 根包导出 `AgentRuntime, StateDaemon, StateAPI, StateKind, Usage`；§17.3 `HMS.start_daemon(*, interval_s=None, phases=None, filters=None) -> StateDaemon`（间隔缺省取 `scheduler.daemon_interval_s`，两者皆 None → `VALIDATION_009`；正在运行时原样返回）与 `HMS.stop_daemon(timeout=None) -> bool`；`close()` 先于索引与存储停掉守护进程。`test_start_and_stop_daemon_on_the_facade`、`test_root_exports_the_main_line_entry_points`。 |
| 85 | 主要 | `segmentation.py` 已发布但摄取平面从未调用它：一条长消息就是一个抽取单元，每单元只抽一个事实的 LLM 会丢掉第一个之外的全部事实 | §12.6 第 5 步：长于 `ingest.segment_max_chars` 的消息由 `segmentation.segment` 切成抽取单元——只改载荷的父消息副本，键到父锚点，因此 `event_idx` 解析为父消息、每条观察都锚到父消息、没有分段铸造 `SourceEvent` / `Anchor`（一事一锚）；EP 单元保持整条；不超过阈值的消息与之前逐字节相同。`tests/test_segmentation.py::test_long_messages_become_extraction_units_anchored_to_the_parent_message`（同一锚点上 >= 3 条观察，阈值 10**9 时只有一条）与 `test_rule_llm_rows_are_unchanged_by_segmentation_and_short_messages_are_untouched`。 |
| 86 | 主要 | `compiler/coverage.py` + `scc.py` 作为可选层发布，没有任何 `compile()` 路径使用 | §15.2 第 8 步在 `selection='coverage'`（配置或 `compile(mode='coverage')`）下：`extract_requirements` → `compile_hierarchical`（L3 技能、L2 不变量、为仍未覆盖者下钻 L1 原子、κ 闭包、证据）→ `select_for_coverage` 保底（剩余预算内每个空的非 audit 槽位一项）；`retrieval['selection' \| 'coverage' \| 'hierarchy']`、卡片末尾的 `## Coverage` 段，`ContextIR.to_pcir()` 经 `scc.lower_ir` 降低并携带第五条义务 `requirements_covered`；`shares` 路径逐字节不变。`tests/test_coverage_scc.py::test_compile_selection_coverage_replaces_shares_and_lowers_with_requirements_covered`（S4 + 一条约束下 uncovered 为空；删掉唯一覆盖原子后 uncovered 非空、义务 violated、默认抛 `PCIR_001`）与 `test_compile_selection_coverage_refines_interface_first_after_consolidation`。 |
| 87 | 次要 | hms4eye `reflect` 只有编译入口，没有上游 `organize` 形状的证据包 | `HMSCapabilities.reflect_evidence(query, *, user_id)`：把侦探报告表示为 `{question, question_date, mode='detective_report', ledger_rows（每行带 anchor_ids / atom_id / type / occurred / mentioned / entities）, controls, source_snippets, answer_ready_context, conclusion, chain}`；无法解析时为空包；`manifest.SURFACE['reflect']` 点名它（`reflect` 仍为 native，`compare_with_upstream()` 不变）。`tests/test_hms4eye_capability.py::test_reflect_evidence_is_an_organize_shaped_ledger_whose_rows_cite_anchors`。 |
| 88 | 次要 | `tests/test_foundation.py::_group` 把 state / eval / runtime / daemon / telemetry / segmentation / coverage / scc / server 留在默认组，提示词 / 注册表断言早于复合任务与后加入的提供者 | §0.4 分组明确：state / eval / runtime / daemon → H（从不导入门面），reranker / entities / telemetry → B，segmentation → A（只导入 A），coverage / scc → F；mvmi 隔离不变；断言 `COMPOSITE_TASKS` / `COMPOSITE_PROMPTS` / `MEMORY_EXTRACTION_ROOTS`；每种 `supported()` 列表按排序断言并含 deepseek / ollama / fastembed / spacy / milvus；`hms_elite.index` 再导出 `MilvusIndex`。 |
| 89 | 次要 | README 的 API / 配置表与本日志未列出这些接线 | README §3 YAML（`extraction_mode`、`segment_max_chars`、`selection`、`daemon_interval_s`）与 §11 各行（覆盖式编译、`usage / trace`、`state` + 委托、`start_daemon / stop_daemon`、`reflect_evidence`、根包导出），中英两版；本表第 81–89 行。 |
| 90 | 主要 | 池内打分用的是完整水合的原子（每原子两次 pydantic 校验 + 三个 JSON 列，约 19 µs），约 150 个池行占 search@30000 16 ms 中的 3–5 ms | §8 / §13.3 `StateStore.axis_lite(scope, atom_ids, …)` 返回 `AtomLite` 打分行（信号列、排序后的实体键、有序主题、`n_anchors` = 水合后的 `len(anchor_ids)`、`age_us`），每 500 个 id 两条语句；`execute_axis_query` 在其上给池打分，只把最终 top_k 经 `axis_select(atom_ids=…, trusted=True)` 水合（`_hydrate_atoms(trusted=True)` = 对已校验行做 `model_construct`，与校验模型逐字段相等）。`utils.scoring.compute_signals` 接受两种形状（`prepared=`、`age_us=`），`prepare_query` / `fusion_weights` 只推导一次查询侧。`tests/test_retrieval.py::test_trusted_hydration_equals_validated`、`test_axis_lite_matches_hydrated_atoms`、`test_prepared_signals_match_per_atom_path`、`test_pool_hydrates_only_top_k`。 |
| 91 | 主要 | 关键词通道对每个 FTS5 匹配按 bm25 排序（`ORDER BY rank, atom_id`：随匹配数线性，N=30000 时 4–8 ms），且各 SQL 通道各自一条语句 | §13.3 `StateStore.recall_channels(...)`：关键词 / 实体 / 新近度（以及硬实体门的未门控链接）合成**一条** `UNION ALL` 语句，每个分支自带 LIMIT；关键词通道每阶段只读最新匹配窗口（`ORDER BY rowid DESC LIMIT max(4·K_kw, 64)`，MATERIALIZED CTE），在窗口内重新施加门控，幸存者按 `(bm25, time_anchor DESC, atom_id ASC)` 排列，不足时分页（`keyword_candidates` 单独调用时语义相同）；新近度分支默认 `INDEXED BY ix_atoms_time`；宽松的 FTS5 阶段与小的新近度兜底只在需要时作为独立的有界语句运行。`test_recall_channels_equals_channel_functions`、`test_keyword_window_is_newest_first_then_bm25`。 |
| 92 | 次要 | `recency_score` 对每个原子重新解析两个 ISO 字符串（`to_utc_iso` + 两次 `fromisoformat`） | `axis_lite` 在 SQL 中算出精确的整数微秒年龄（`strftime('%s')` + 32 字符形式的固定微秒字段；其它形式为 NULL → 走字符串路径），`recency_from_age_us` 逐位复现 `timedelta.total_seconds() / 86400`（`julianday` 做不到：毫秒分辨率与约 2.4e6 量级的双精度相减）。`test_axis_lite_matches_hydrated_atoms` 在四个 `as_of` 时刻断言逐位相等。 |
| 93 | 次要 | 硬实体门在每条语句里都是一个 `atom_id IN (子查询)` 子句——即使已经由 `atom_entity` 驱动的读取也会把它物化（3000 条链接约 0.7 ms），并且让池探测变成列表驱动 | `StateStore.entity_gate_sql(scope, keys, correlated=)` 给出 IN 列表形式（扫描：仅轴读取、计数、新近度分支）与相关 EXISTS 形式（id 驱动的池读取：轻量行、最终 top_k、关键词窗口）；`_plan_gates(…, entity_form='in'\|'exists'\|'none')`；`recall_pool` 只经 `atom_entity` 读一次 G（`entity_candidates(order=None)`：下推时不排序的 `pool_max + 1` 探测，否则整个不排序的 G），下推时把未门控链接连同过滤器交给提供者（`_index_filter` 不再因实体门放弃）。`test_entity_gate_forms_select_the_same_set`；`tests/test_retrieval_v3.py::test_entity_gate_is_a_non_correlated_subquery` 不变。 |
| 94 | 次要 | `select_top_k` 对每个选中行经 `dot(matrix[r].tolist(), query)` 重算（两次 list → array 往返，每行约 2.5 µs） | float64 矩阵行本身就是 `dot` 要构造的数组：`float(np.dot(matrix[r], q))`——同样的输入、同一个 BLAS 调用、同样的位（`test_select_top_k_row_dot_is_exact` 在 30000 个已存向量与随机查询上对照 `rank_hits` 断言）；float32 矩阵保留列表路径。`test_sqlite_index_cache_is_per_store` 钉住 SQLite 提供者的缓存绝不跨 StateStore 实例。 |
| 95 | 次要 | 池的主键读取在 SQLite 默认 2 MB 页缓存下命中散布于整个文件的页（N=30000、约 100 MB 文件时每次搜索约 1 ms） | `StateStore(page_cache_kib=32768)` 设置 `PRAGMA cache_size`（0 保持 SQLite 默认）；`test_page_cache_pragma`。 |
| 96 | 次要 | Milvus 的池内重算是每块一次 id 过滤的 HNSW 搜索（N=30000 时 100 个 id 6.6 ms），多块候选列表的 RPC 逐个串行 | `MilvusIndex.search(candidate_ids=…)` 在 `top_k` 覆盖整块时按主键读取该块向量（`_vectors_of`：一次 `query`，2.4 ms，精确），`MilvusIndexConfig.search_workers`（默认 1）用线程池并发多块——默认保持串行，因为 milvus-lite 会把它们串行化（3 × 1024 个 id：串行 41 ms，4 线程 63 ms）。`test_milvus_pool_vectors_by_primary_key`。RETRIEVAL_BENCH_2026-09-10 的表已复测。 |
| 97 | 次要 | RETRIEVAL_BENCH §1–§2 / README §4.1 / `bench_results/scale_v3.json` 仍是第 90–96 行之前测得的数字（sqlite N=30000 时 15.9 ms）；第 96 行把整块读取改到 `_vectors_of` 之后，`test_milvus_candidate_chunks_merge_like_sqlite` 仍在数 `_ann` 调用；只装 `[test]` 的 Python 3.10 venv 下两条仅 numpy 的测试失败而不是跳过 | 第 90–96 行之后三臂复测（N = 300 / 3000 / 30000 的 search 中位：sqlite 2.71 / 2.06 / 3.41 ms，milvus 11.6 / 49.2 / 14.4 ms，mem0 0.20 / 0.23 / 0.32 ms；重叠率 1.0 / 0.80 / 0.62 与 1.0 / 0.81 / 0.65；首次测量保留在 BENCH §5），README §4.2 直接用 `run_ablation` 打印九任务 × 四消融矩阵（`test_readme` 逐字比对：Axis + Graph + Support 九项全 1.00，Axis only 均值 0.65）；分块测试断言 `top_k = len(candidates)` 时 `_vectors_of` 的块大小为 `[1024, 1024, 482]` 且没有 `_ann` 调用；`test_bounded_search_is_bit_identical_to_rank_hits` 与 `test_sqlite_index_cache_is_per_store` 在没有 numpy 时跳过。 |
| 98 | 主要 | 抽取回复每条观察带 17 个键，其中九个只是复述 `text`；EgoLife 运行中批次输出 token 中位数 5,562（22% 触及 8,192 上限并被对半重提），而输出 token 单价是缓存未命中输入的 4 倍、缓存命中输入的 200 倍 | §5 `ingest.extraction_output`（'full' \| 'compact'，默认 'full'，透传给 `Ingestor(extraction_output=...)`）；§6.5 紧凑契约（`observation_extraction_compact` / `memory_extraction_compact`、`COMPACT_OBSERVATION_KEYS`、推导表）与 `hms_elite/ingest/derive.py`；§11 规则 LLM 实现两个紧凑任务；§12.3 `extract_observations` / `extract_all` 接受 `output` 与 `entity_extractor`；`llms.rule` 导出 `predicate_of`、`observation_topics`、`mention_texts`（R9 / 主题 / 提及启发式抽出为公共函数，逐字节一致）。`tests/test_extraction_compact.py`（7 条测试、12 个用例）。 |
| 99 | 主要 | 紧凑模式只按词表推导 `predicate` 且 `objects = []`，存下的紧凑候选不再命中规则 LLM 的修订表（`chat_conflict` 丢失 CONFLICT 行 / 冲突边）；只在 `text` 中点名的第三人称主语（`Sarah joined …`）被降为 `user`；`derive.py` 在 `PREF_RE` / `CONSTRAINT_RE` 之外自行编译偏好 / 约束线索正则；无候选时的 `"decisions": []` 根仍逐条警告；§11 未命名紧凑处理器与共享辅助函数；紧凑提示词常量不符合 `<TASK>_PROMPT` 规则；测试 9 写着五个 JSON 提示词；README §8 写着六个分段任务 | `llms.rule.statement_triple`（建立在 R1–R9 线索表上的第三人称陈述解析器）与 `constraint_objects` 公开；`derive.py` 只组合它们（`subject_for`、带陈述线索的 `relation_of`）且不定义线索；§6.5 表的 `subject` / `predicate` / `objects` / `relation` / `entity_mentions` 行（写明实体轴的收窄）与不变量（含 `history` / `graph_edges` 的各表相同、空决策不出声）；`extract_all` 把无候选时的 `decisions: []` 视为不出声；常量改名为 `OBSERVATION_EXTRACTION_COMPACT_PROMPT` / `MEMORY_EXTRACTION_COMPACT_PROMPT` 并在 §6.5 点名；§11 公开函数列表扩充并新增 §11.8；§12.3 条目；测试 9 改为六个 JSON 提示词；README §8 列出七个分段任务与两个复合任务。`tests/test_extraction_compact.py` 现为 9 个测试、14 个用例（陈述解析器表；空决策不出声；夹具在 `IDENTICAL_TABLES` + 原子三元组上相同）。 |
| 100 | 主要 | 紧凑模式只把 `entity_mentions` 推导为 extractor(`text`) ∪ `speaker`，于是只有 LLM 才知道的提及（小写普通名词如 `python`、未被识别的名字）从实体轴上消失——这是紧凑契约唯一收窄的轴（第 99 行） | `COMPACT_OBSERVATION_KEYS` 增加可选的 `entities` 提示（0–6 个短字符串；两个紧凑提示词：OUTPUT FORMAT 条目、一行准则、取代示例；memory 紧凑提示词改写措辞以保持低于完整孪生的 0.75——5,195 / 6,936 字符，observation 的为 2,978 / 4,105）；`derive.union_mentions` 与 `expand_compact_item(entities=)`：`entity_mentions = union(entities, extractor(text), speaker)`，LLM 顺序在前、不分大小写去重；`extraction._entities_hint`（`_str_list`；形状畸形时记警告并忽略，绝不跳过条目）；`RuleLLM._compact_observations` 在完整观察的 `entity_mentions` 非空时输出 `entities`；§6.5 的 `entity_mentions` 行重写（删去收窄句）、字段规则、不变量，§11.8，§12.3；README §3 / §8，DESIGN §9。`tests/test_extraction_compact.py` 现为 12 个测试、17 个用例（`User prefers python for scripting.` 上与完整模式等价；并集顺序与大小写折叠；畸形提示被忽略；提示词键集合与准则；每个聊天夹具上 `entity_ids` 相同）。 |
| 101 | 次要 | 文档里 `ingest.extraction_output` 的默认值悬而未决（README §3 写着由测量阶段决定），同时第 100 行提示词的一次未完成真实 API 冒烟（`benchmarks/egolife/results/real/elite_compact_entities_smoke5`，18 批只记录 12 批，无汇总）无标记地留在磁盘上 | 记录决定：默认仍为 `full`；两条件切换规则与实测建议（长的多说话人转录与批量灌入用 `compact`）写入 §6.5 与 README §3 / §4.4；未完成的运行附 `INCOMPLETE.md`；新增设计说明 `docs/design/COMPACT_EXTRACTION_2026-09-11.md`（契约、同 18 批冒烟、204 题回放 A/B、推导与 LLM 一致率、compact 丢失项、复现）；EGOLIFE_REAL_LLM_ELITE_2026-09-11 §7 加指针。无代码改动；测试套件不变。 |
| 102 | 主要 | 最终紧凑提示词的真实 API 冒烟完成后（`results/real/elite_compact_entities_smoke5_r2`：18/18 批，每次请求输出 token 中位 2,793 vs 5,631，触顶 0 vs 8 次，模型耗时 −49%；回放 A/B hit@20 161 = 161，实体轴 8,439/8,439 原子上 compact ⊇ full），§6.5 切换规则的两个条件均已满足，但 `ingest.extraction_output` 仍默认 `full` | 默认切换为 `compact`（2026-09-11）：`IngestConfig.extraction_output = 'compact'`（§5，双语注释写明规则与日期）；`Ingestor(extraction_output=None)` 回退到 `config.ingest.extraction_output` 而非硬编码的 `'full'`（§12.1）；`benchmarks/egolife/run_real.py --extraction-output` 默认取 `IngestConfig().extraction_output`（manifest 不变）；`full` 不变且逐字节一致。文档：§6.5 默认段与不变量句、README §3 / §4.4 / §8（配置 YAML；两处输出字面改为推导置信度 0.7 → 反馈 `[0.7, 0.9]`）、DESIGN §9、`COMPACT_EXTRACTION_2026-09-11` §0 / §2 / §6 / §7 / §8 / §9 按 `_r2` 数字改写（中断的首次尝试保留其 `INCOMPLETE.md`）、EGOLIFE_REAL_LLM_ELITE §7 指针。测试：`tests/test_extraction_compact.py` 断言 `IngestConfig`、默认配置的 `HMS`（调用日志 `memory_extraction_compact`）与不传参数构造的 `Ingestor` 三处默认为 compact，且钉为 `full` 的配置仍能到达这样的 Ingestor；`tests/test_ingest.py`、`tests/test_extraction_single.py` 的夹具与 `tests/test_scenarios.py` 的模块级 `config` 夹具钉住 `extraction_output='full'`，因为它们陈述的是完整契约（脚本化完整回复、`memory_extraction` 调用日志、R1 / R2 置信度 0.8 / 0.9 与 §19.8 哈希；§19 标题段已说明）。测试套件 680 passed, 2 skipped。 |
| 103 | 主要 | `MEMORY_EXTRACTION_COMPACT_PROMPT` 下 deepseek-flash 把观察 i 与邻域候选 i 配对并回答 NONE 'duplicate of existing memory'：204 题 EgoLife 运行中 7,315 条观察里 1,685 条 NONE（`full` 下 8,470 条中 23 条；1,623 条 `target_id == id`；288 批中 156 批恰好是观察 0..9），而库把 NONE 变成目标上的支持见证、不存该陈述，因此 21.8% 的抽取陈述从未入库（hit@20 141 vs 161；回放中把所有裁决强制为 ADD 得 154——`benchmarks/egolife/results/replay_diag/summary.md`） | (1) 提示词：紧凑 DECISIONS 准则按每条新观察 `id` 陈述（绝不按候选、绝不按位置），NONE 只留给复述（同一事实、同一主语、同一取值；新的细节 / 人物 / 时间 / 取值都是 ADD），一对 Bad / Good 示例展示相似候选仍为 ADD，示例带两个候选、观察 `0` 的目标为 `1`；观察键与决策形状不变、规则 LLM 不动，5,192 字符对 0.75 上限 5,202；`MEMORY_EXTRACTION_PROMPT` / `OBSERVATION_EXTRACTION_PROMPT` 逐字节不变（sha256 钉住）。(2) 库侧守卫（§6.3、§12.6 第 6 步，两种契约、两种模式）：`ingest.verdict_min_similarity`（0.5；0.0 关闭）——NONE / REVISE 只在 `ingest.extraction.verdict_similarity`（词干化内容词元并上原子主语的包含度 |A ∩ B| / min(|A|, |B|)，`statement_tokens`）不低于门槛时采纳；REVISE 在 `supersedes_hint_ids` 点名目标或主语 + 谓词相同（互斥属性情形：`lives_in` 巴黎 → 柏林仅 1/3）时同样采纳；CONFLICT 只要共享一个词元即采纳；低于门槛时改为 ADD，记 DEBUG 日志与一条 `IngestResult.downgraded`。选包含度而非 Jaccard：把 `results/real/elite_full204_r3` 经流水线回放恢复候选文本后，包含度 ≥ 0.5 保留 21/23 NONE、9/9 REVISE（Jaccard ≥ 0.5：14/23、1/9）；记录的 `full` 回复经守卫回放 hit@20 保持 161/204、MRR 0.4973（2 个 NONE 降级，存活原子 8,439 → 8,441），记录的紧凑回复（旧提示词）经守卫回放 hit@20 从 141 升到 151/204、MRR 0.3873 → 0.4927、存活原子 5,717 → 7,196（降级 NONE 1,475、REVISE 4、CONFLICT 1）。文档：§5、§6.3、§6.5、§12.1、§12.6、README §3 / §5.1、DESIGN §9。测试：`tests/test_revision_guard.py`（19 个用例：两种契约下按下标配对的 NONE 只保留复述；门槛 0.0 恢复旧行为且字段有校验；每个聊天夹具在守卫开 / 关下逐行、逐表相同；经典修订仍成立；紧凑 DECISIONS 措辞；包含度取值；主语补回）。测试套件 699 passed, 2 skipped；ruff 干净；TOTAL_ENGLISH_ONLY_BLOCKS=0。 |
| 104 | 主要 | 第 103 行守卫的第一版把原子主语计入比较的词元集合，并对最小侧包含度用含等号的 0.5 门槛，于是任何只与目标共享说话人和一个转述动词的短陈述都恰得 0.5，按下标配对的 NONE 被采纳（主语 `user` 的 "User has cats." 之后的 "User has dogs."；"Jake said there are two pots." 对 "Jake said it was put in the fridge."）：记录的紧凑回复上 112 条 NONE 越线（80 条恰为 0.5、94 条共享转述动词、约 110 条并非复述），守卫回放停在 hit@20 151/204，而强制 ADD 的上限是 154（第 44、301、309 题）；紧凑 OUTPUT FORMAT 块仍把观察 `0` 与目标 `0` / 提示 `["0"]` 配对 | （1）`ingest.extraction.verdict_similarity` 改为去掉头部而非补上主语：`statement_tokens(text, subject, predicate)` = 词干化内容词元减去主语词元减去谓词词元（`_head_tokens`：与文本同样切分，下划线 → 空白），谓词只在两原子谓词相同时去掉；新增 `with_shared=True` 返回 (相似度, ǀA ∩ Bǀ)。（2）`Ingestor._guard_verdict`（§6.3、§12.6 第 6 步）：被佐证的 REVISE（提示或相同主语 + 谓词）最先通过；然后两原子主语不同的裁决改为 ADD（NONE / REVISE / CONFLICT 一视同仁——每个都断言关于同一主语的关系）；同主语的 CONFLICT 通过（词法重合不是不同取值的证据）；NONE / 未被佐证的 REVISE 须包含度 ≥ `verdict_min_similarity` 且至少共享两个词元或较短陈述被整体包含；`downgraded` 条目带 `shared` 与 `subjects_differ`，DEBUG 行为 `similarity S (bar T[, subjects differ ǀ, one shared token])`。（3）两个紧凑提示词的 OUTPUT FORMAT 引用提示 `["1"]`，记忆任务的决策为观察 `"0"` 给出 `target_id` `"1"`（5,192 字符不变；完整提示词逐字节不变）。零 API 回放（`results/replay_diag2/summary.md`，G1v2 / F1v2，argv 同 G1 / F1）：紧凑 hit@20 154/204 = W5 上限且命中集相同（MRR 0.5038、存活原子 7,299、降级 1,583 = NONE 1,577 + REVISE 5 + CONFLICT 1；采纳 10 条 NONE 与 1 条 CONFLICT；相对 G1 找回第 44 / 301 / 309 题）；完整 161/204、MRR 0.4973、存活 8,442，28 个过守卫的裁决中 3 个降级（G1 的两条 NONE 加一条 0.333 的未佐证 REVISE），前 20 行与 F1 在 204/204 题上相同。文档：§6.3、§6.5、§12.6、README §3 / §5.1、DESIGN §9。测试：`tests/test_revision_guard.py` 34 例（+15：只共享头部的短用户事实在两种契约下都被存入、`today` 对被单词元规则拒绝；跨主语的 NONE / REVISE / CONFLICT 降级而同主语 CONFLICT 采纳；十二个去头部包含度取值；`statement_tokens` 去头部；OUTPUT FORMAT 配对），`tests/test_extraction_compact.py` 断言提示 id `1`。套件 714 通过、2 跳过；ruff 干净；TOTAL_ENGLISH_ONLY_BLOCKS=0。 |
| 105 | 主要 | `ingest.extraction_output` 凭零 API 回放与 18 批冒烟默认为 `compact`（第 102 行），但真实 204 题 EgoLife 运行中 compact 初版记忆行 53.9%（`results/real/elite_compact_full204`；根因是第 103 行的按下标配对 NONE 裁决），经第 103–104 行的措辞与 `verdict_min_similarity` 守卫后的 compact 修订版 56.4%（115/204；`results/real/elite_compact_v2_full204`；卡片 62.7%，hit@20 77.9% / 79.4%，NONE 15，窗口丢失 0，空闲价 ¥4.35，每批 7.3 s），而 `full` r3 为 61.3%（125/204；卡片 61.8%，hit@20 78.9%，¥11.56，14.9 s）——低 4.9 个点，超出 2 个点容忍 | 默认回退为 `full`（2026-09-11 晚）：`IngestConfig.extraction_output = 'full'`（§5；双语注释写明回退、日期、规则与修订版数字）；`Ingestor(extraction_output=None)` 仍跟随 `config.ingest.extraction_output`（§12.1 不变）；`benchmarks/egolife/run_real.py --extraction-output` 仍默认 `IngestConfig().extraction_output`；两个提示词逐字节不变。§6.5 默认段按第三个条件（真实 204 题运行的记忆行准确率与 `full` 相差不超过 2 个点）、初版根因、修订版数字与建议重写（成本敏感的批量灌入与状态卡片阅读器用 `compact`，写入费用 0.38 倍 / 每批耗时 0.49 倍；记忆行阅读器用 `full`）；§5 字段列表与 §6.5 不变量句；README §3 / §4.4 / §8（配置 YAML `extraction_output: full`；两处输出字面重新打印 LLM 的置信度 0.8 → feedback `[0.9, 1.0]`）、DESIGN §9、`COMPACT_EXTRACTION_2026-09-11` §0 / §6 / §8、`HIGHLIGHTS_2026-09-11` §1.4、`SCENARIO_COMPANION_TOY_2026-09-11`、`EGOLIFE_REAL_LLM_ELITE_2026-09-11` §7、`docs/design/artifacts` 的两个页面（compact 措辞改为可选；实测数字不变）；决定记录在 `EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`。测试：`tests/test_extraction_compact.py` 断言 `IngestConfig`、默认配置的 `HMS`（调用日志 `memory_extraction`）与不传该参数构造的 `Ingestor` 三处默认 `full`，且钉在 `compact` 的配置到达这样的 Ingestor（调用日志 `memory_extraction_compact`）；`tests/test_ingest.py`、`tests/test_extraction_single.py`、`tests/test_scenarios.py` 的 `full` 钉住保留（无害，它们陈述的就是所测的契约）。 |
| 106 | 主要 | 4,851 窗口的 EgoLife 运行中，输出 token 占完整契约写入费用的 83%（¥11.5 中 2.40M 输出 token；compact v2 ¥4.2 中 0.76M；mem0 ¥1.85 中 0.22M，输入未命中侧已持平）：JSON 信封的开销是其承载文本的 2–4 倍，检索只用到文本与决策（零 API 回放：entities / type / when 对 hit@20 的影响为 0），而紧凑提示词自己写出的文本低于完整提示词（记忆行 56.4% 对 61.3%），两种既有契约都无法弥合差距 | 新增 `ingest.extraction_output='lines'`（§6.6，2026-09-12）：LLM 以 tab 分隔的 `OBS` / `EP` / `DEC` 文本回答 `memory_extraction_lines`（无 JSON、不用 json 模式）——`<event_idx>⇥<T>⇥<statement>[⇥@date]`、`<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`、`<OBS 行号>⇥REVISE|CONFLICT|NONE⇥<候选>⇥<理由>`；提示词逐字保留完整提示词的陈述规则与紧凑 DECISIONS 规则，只替换 OUTPUT FORMAT（`MEMORY_EXTRACTION_LINES_PROMPT`、`COMPOSITE_TASKS`、`LINE_TYPES` / `TYPE_LETTERS`）；`LLMBase.generate_lines` 以不带 `response_format` 的方式调用 `_call`（§10.1）；`hms_elite/ingest/lines.py::parse_lines` 为容错解析器（围栏、散文、缺 tab、未知字母、被截断的末行、越界的 DEC）；`extract_all` 按 `output == 'lines'` 分支，随后以 `speaker_from_subject=True` 走紧凑条目路径（开头的专有名成为说话人 → 实体提及 / actor_id；`derive.SIDE_SUBJECTS`），缺少 DEC 行静默为 ADD；`RuleLLM._task_memory_extraction_lines` 把 R1–R9 结果渲染为行（§11.8）；仅 single 模式——`IngestConfig` 校验器（双语 ValueError）、`Ingestor` 与 `extract_observations`（`VALIDATION_013`）；`EXTRACTION_OUTPUTS = ('full', 'compact', 'lines')`，基准 CLI 选项随之。默认不变（`full`）；完整与紧凑提示词逐字节不变（钉住四个 sha256）。文档：§5、§6.1、§6.4、§6.6、§10.1、§11.8、§12.3、README §3、DESIGN §9。测试：`tests/test_extraction_lines.py`（16 例：解析器规范输入 / 容错 / 截断 / DEC 范围 + EP 形状；脚本化行式回复经 `HMS.add`；DEC REVISE → 修订边且守卫仍生效；规则 LLM 的行式门面在每个聊天夹具上与完整门面逐行、逐表相同，只有说话人规则的 `actor_id` 除外；EP / @date / DEC 渲染；配置校验与接线；提示词注册与陈述规则；DeepSeek 不发送 `response_format`）；`tests/test_foundation.py` 与 `tests/test_extraction_compact.py` 对元组的钉住已更新。套件 730 通过、2 跳过（714 + 16）；ruff 干净；TOTAL_ENGLISH_ONLY_BLOCKS=0。 |
| 107 | 主要 | 18 批冒烟（`results/real/elite_lines_smoke5`，2026-09-12）之后对行式契约的复核：（1）`parse_lines` 按已保留的观察数编号，而不是按模型所写的 OBS 行号，因此丢掉一行（空陈述）或折行的陈述（无 tab 的续行被存为单独的片段原子）会让之后每个 DEC 目标偏移一位，REVISE 可能修订错误的原子（守卫只查主语 + 谓词）；（2）行式线上格式只带 u / a / y / r / k，没有任何环节为 EP 推导 `entity_mentions` / `topic_tags` / `b`，行式 EP 存下 `[]` / `[]` / `''`，而 full 存的是规则 LLM 的值；（3）§5 签名仍写 `Literal["full", "compact"]`；（4）README §3 / §6.6 / DESIGN §9 说行式门面与 full "逐行、逐表相同"，而测试只比较 8 张表与原子四元组（confidence、relation、actor_id 与 EP 各列的差异未被钉住）；（5–7）基准脚本：finish=stop 但不可用的行式回复（散文、JSON 对象）被标为 empty_extraction 而非 invalid_json，因为 `parse_lines` 从不输出 JSON 路径的日志文本，重试 / 二分从未运行；finish=length 的行式回复在切片被二分之前已被库部分存入（父切片原子与两半的原子并存，`added` 重复计数）；items_skipped 的 grep 把行式解析器自己的 "skipped" 警告计为模式拒绝；`manifest.extract_params.response_format` 对行式臂硬编码为 `json_object`。冒烟中均未触发（0 条丢弃 / 折行、0 条 EP 行、0 次截断、0 条解析器警告） | `parse_lines`：OBS 分节里每个带 tab 的行都占用一个 1 起的行号，无论是否存下观察（只有前缀的行保留行号并等待续行），无 tab 的行是续行（填入等待中的槽位，否则接到上一条陈述之后）且不占行号，DEC 行点名没有存下观察的行时记警告并跳过，没有开启任何分节的回复记一条含新增 `NO_SECTION_MARK` 的警告；新增 `ingest.derive.expand_lines_ep`（entity_mentions / topic_tags 取自 `'<a> <y> <u>'`，b 为空时 = k= 约束以 `' ; '` 连接），行式下在 `_ep_from_item` 之前运行——`eps` 表现在在每个聊天夹具上与 full 相同；两种语言的 §5 签名已修正；§6.6 不变量、README §3 与 DESIGN §9 现在点名相同的各表（含 `eps` 共九张）并逐列钉住 `observations` / `memory_atoms` 的推导列（`confidence`、`relation` / `supersedes_hint_ids`、`actor_id`、chat_zh 的 `entity_mentions`）；基准脚本：`parse_error` 同时计入 `NO_SECTION_MARK`（→ invalid_json_retry / bisect / invalid_json），解析器警告另计为 `lines_dropped`（子记录、`classify_retain`、`_parse_failures`、`other_warnings`），`EliteRealArm._reject_truncated_lines` 把 finish=length 的回复以 `''` 交给库，使二分之前不存任何内容，manifest 的 `response_format` 对行式臂写为 `none`。用新解析器重新解析冒烟的 18 条原始回复得到相同的 1231 条观察 / 82 条决策 / 63 个不同 id（没有占号不存的行、没有续行）。测试：`tests/test_extraction_lines.py` 16 → 21 例（丢行 / 折行 / 只有前缀的行的编号；无分节警告；父任务的复现经 `HMS.add`——本意指向第 3 行的 REVISE 不再修订燕麦拿铁原子；`expand_lines_ep` 与脚本化 EP 经 `HMS.add`；一致性测试比较 `eps` 与两张推导列表；基准脚本源码钉住）；容错测试现在期望无 tab 的行接续上一条陈述。套件 735 通过、2 跳过；ruff 干净；TOTAL_ENGLISH_ONLY_BLOCKS=0。 |
| 108 | 主要 | 默认抽取契约与读路径（2026-09-18，`docs/design/READER_AB_2026-09-18.md`）：库默认 `ingest.extraction_output` 原为 `full`（每 204 道 EgoLife 题 ¥11.56），而行式存储上的阅读端 A/B 测得混合上下文——按时间线步骤记忆化的会话卡片作前缀，加去重、按时间排序、带时间戳的检索行，记忆在前作答——70.1%（143/204），完整契约的记忆行 / 卡片为 61.3% / 61.8%，行式写入费用 ¥2.24（与记忆行的符号检验 p < 0.0001）；混合构造器只存在于 `benchmarks/egolife/reader_ab.py`，库的调用者无法复现测得的读法。**对 `staged` 配置是破坏性变更**：`lines` 成为默认后，不显式给出 `extraction_output` 的 `{"ingest": {"extraction_mode": "staged"}}` 现在是 `CONFIG_001`（`_lines_needs_single` 校验器未变）。 | `IngestConfig.extraction_output` 默认 `full` → `lines`（`hms_elite/configs/base.py`，注释历史续写；`hms_elite/ingest/extraction.py` 头注释）。新增 `hms_elite/compiler/context.py`（`ROWS_HEADER`、`NONE_TEXT`、`normalize_text`、`row_times`、`dedup_rows`、`sort_rows_by_time`、`format_stamp`、`render_rows`、`rows_text`、`hybrid_text`；由 `hms_elite.compiler` 导出），`benchmarks/egolife/reader_ab.py` 改为导入它们（其 `dedup_rows` / `sort_rows_by_time` / `format_stamp` / `row_times` / `normalize_text` / `hybrid_context` 就是库函数，`rows_ts_context` 委托给 `rows_text`，`HYBRID_ROWS_HEADER = ROWS_HEADER`，`build_prompt = real_arms.build_prompt`）。新增 `HMS.context(query, *, filters, top_k=None, as_of=None, budget_tokens=None) -> dict`（§6.7、§17.3）及进程内会话卡片记忆（`_session_card`、`_context_memo`、`_context_lock`；`reset()` 清空），`AsyncHMS.context` 镜像（§17.8；18 个原生方法）；`StateStore.latest_event` / `touch_scope` / `scope_version` / `unversioned` 与按范围写入版本（`write_generation`、`scope_write_versions`、`_txn_scope_keys`、`_txn_wrote`；仅当事务写入了带版本的行才在 COMMIT 时推进，回滚与 reset 时推进；只有 `unversioned()` 之外改变了行的语句计数，因此内部事件 + 锚点（`system_event`）、检查点、调度器的 meta 行与编译态归档、空转的调度器阶段与零行事务都不推进任何版本——§8）；`CompilerConfig.context_session_budget_tokens` 1200 / `context_rows_top_k` 20 / `context_rows_header` / `context_timestamps` True（§5）。基准脚本：`benchmarks/egolife/arms.py` 不再钉住 `full`；`real_arms.build_prompt(question, context, order)` + `answer(..., order=)` + `EliteRealArm.hybrid()` + `stats()` 的 context 字段；`run_real.py` 的 Elite 标签默认为 `elite_rows` + `elite_hybrid`（`ENGINE_LABELS`、记忆在前的提示词、`hybrid` 汇总块与表格，`--with-card` 加回 `elite_card`；`--extraction-output full` 仍可选）；`report_real.py` 按运行目录实际携带的臂报告（`Run.arms`、`OPTIONAL_ARMS`），`casestudy_report.py` 把卡片文件视为可选。测试：`tests/test_context.py`（7 例，§18.3 第 65 条）；`tests/test_async.py` 40 → 41 / 17 → 18 并加入 `context`；`tests/test_facade.py` 钉住签名；`tests/test_wiring.py` 钉住四个配置默认值；`test_extraction_compact.py` / `test_extraction_lines.py` 对未触碰的门面期望 `lines`；`test_wiring.py` / `test_extraction_single.py` / `test_segmentation.py` 在使用分段模式或完整契约 JSON 回复处钉住 `full`；README 字面输出重新生成（`[0.7, 0.9]`、`'confidence': 0.7`）。文档：CONTRACT §5.1 / §5.2 / §6.5 / §6.6 / 新 §6.7 / §8 / §17.3 / §17.8 / §18.3 / §19（两种语言），README §3 / §4.4 / §5（新 `context()` 片段）/ §8 / §11，DESIGN §6.11 / §9 / §13，OVERVIEW / HIGHLIGHTS / COMPACT_EXTRACTION 各一行说明。后续：`POST /context` 路由。套件 763 通过、3 跳过；ruff 干净；TOTAL_ENGLISH_ONLY_BLOCKS=0。 |
| 109 | 次要 | 为 EgoLife-ZH 实验加入 CJK 文本开关（2026-09-22，`docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`）：此前库在关键词通道两侧都发出 CJK 字符二元组、无法度量英文中心基线，且紧凑 / 行式推导读不出中文主语、动词与宾语，一条中文陈述存为 `user / states / []` 且没有 `actor_id`。默认值逐字节复现此前行为（英文存储、已记录运行、`rule@entities-v1` 身份、不新增 meta 行）。 | 新增 §9.6。`TextConfig.cjk_tokens: Literal["bigram", "none"] = "bigram"`（`HMSConfig.text`，在 `retrieval` 之后；`hms_elite/configs/base.py`，由 `hms_elite.configs` 导出）；`utils.text.cjk_tokens_scope` / `cjk_tokens_mode` / `text_scoped` / `CJK_TOKEN_MODES`、`tokenize(text, *, cjk=None)`、`content_tokens(text, *, cjk=None)`、`stem` 的 CJK 守卫；`memory.main.TEXT_SCOPED_METHODS`（35 个被包装的 `HMS` 方法）+ `Ingestor.ingest_messages / ingest_events`、`StateCompiler.compile`、`StateScheduler.run` 套 `text_scoped`；`scoring._content_vocabulary(text, cjk)` / `_key_tokens(key, cjk)`；`StateStore.rebuild_fts()` + `HMS._sync_fts_mode` + meta 键 `text_cjk_tokens`（`TEXT_CJK_META_KEY`）。`IngestConfig.zh_rules: bool = False`；`extract_all / extract_observations / _compact_observation_from_item / expand_compact_item / expand_lines_ep(..., zh_rules)`；`llms.rule.statement_triple(..., *, zh_rules)`、`predicate_of(..., *, zh_rules)`、`mention_texts(..., *, zh_rules)`、`observation_topics(..., *, zh_rules)`、`VERB_LEXICON_ZH`、`_zh_objects`；`utils.text.extract_entities(text, *, zh_rules=False, names=())`、`_zh_name_candidates`、`ZH_STATEMENT_VERBS`、`ZH_NAME_STOP`、`TOPIC_TABLE_ZH`、`sentiment(text, *, zh_rules)`、`topic_tags(text, *, zh_rules)`，优先级 NAME 4 / ZH 5 / CJK 6；`RuleEntityExtractorConfig(zh_rules, names)` 与 `rule@entities-zh-v1` 身份（§10.5）；`HMS.__init__` 把 `ingest.zh_rules` 转给规则抽取器。评审修正（同日）：`utils.text.ZH_TIME_WORDS` / `ZH_LEAD_ADVERBS` / `ZH_LEAD_ADVERB_ALT` / `zh_name_ok` / `zh_name_stop_prefix`（含动词或时间词的开头永不成为人名；和 / 跟 / 给 之后跳过 了 / 过 / 着；了 不再是人名边界），`llms.rule._zh_clause`（介动词的谓词 = 下钻到达的动词；`VERB_LEXICON_ZH` 的 和 / 跟 → `states`，不再有 `with`），`statement_triple` 读主语之前跳过开头的停用词 / 时间词，两侧共用一个副词备选式；`HMS.last_downgraded`（最近一次 add 被守卫降级的裁决，信封不变）。`MEMORY_EXTRACTION_LINES_PROMPT` 加入源语言句（无钉住）。基准脚本：`benchmarks/egolife/translate_zh.py`（data_zh 构建器：经 `real_arms.DeepSeekClient` 用 deepseek-flash 翻译窗口、可续跑的 JSONL 缓存、`--cost-cap-cny`、`--workers`、确定性 `NAME_GLOSSARY`、校验、截断时二分、`translate_report.json`），`windows.build_windows(..., text_lang='en' | 'zh')`，`real_arms.EliteRealArm(embed_model, embed_dims, cjk, fastembed_cache_dir, entity_names)` / `Mem0RealArm(embed_model, embed_dims)`（`stats()['embedder']` = `real_arms.embedder_identity`：默认嵌入器为 `fastembed:<model>`——即已记录的 `results/real/mem0_full204/summary.json` 字符串——其它为 `fastembed:<model>@<dims>`），`EliteRealArm.recall` 以 `explain=True` 检索并逐行带 `keyword` / `entity` / `recalled_by` / `channel_sizes`（`questions_elite_rows.jsonl` 的 `channels`：`keyword_hits` / `entity_hits` / `channel_sizes` / `recalled_by`），`retain_with_bisect` / `summarize` 带 `downgraded`，`run_real.py --text-lang / --embed-model / --embed-dims / --cjk / --fastembed-cache-dir / --entity-names`（manifest 的 `embedding.dims_source / cache_dir / fastembed_cache_path_env`、`data_root`、`text_lang`、`cjk`、`library_switches`、`entity_names`；summary 顶层 `embedding` / `text_lang` / `cjk`；`default_entity_names`：只在 `--text-lang zh --cjk on` 时取术语表人名，因此运行 A 保持 `rule@entities-v1`；`FASTEMBED_CACHE_PATH` 无条件赋值——因此默认的英文调用其进程环境现在多了这个变量，尽管输出不变：同一份 ONNX 权重，en 模型已在 `~/.cache/hms-elite-bench/fastembed_cache` 下，`manifest.embedding.fastembed_cache_path_env` 记录它），`translate_zh.py` 的 `SpendLedger` / `translate_spent.json`（上限跨续跑累计）、`--allow-tight-cap`（预估超过剩余上限 80% 即拒绝）、`--retry-failed` / `resume_skip_ids`（单窗口重试失败即为最终状态）、上限触发后二分的两半也停止（reason 为 `cap` 的行）、`load_cache` 跳过被截断的末行、`questions_source_note` 记录题目取原生字段的偏离，`compare_real.py` / `report_real.py` 打印各运行的嵌入器身份、文本语言与 CJK 开关而不是硬编码值。第二轮评审（同日）：实体轴读侧使用门面的抽取器（`plan_query(..., extractor)`、`mvmi.query._scoring_plan` / `search(..., extractor)`、`StateCompiler(..., entity_extractor)`、`coverage.extract_requirements(..., extractor)`；`HMS.search` / `context` / `compile` / `coverage` 传入 `self.entity_extractor`），`_POSITIVE_SENT_ZH_RE` 的 爱 改为受限形式（爱丽丝 是人名），`statement_triple` 在读动词之前跨过非 我 的代词开头（我们 → `project`，他 / 她 / 他们 / 大家 → None），`zh_name_ok` 只在开头两端检查动词 / 副词（王和平 接受；李刚 仍拒绝，已记录），`real_arms.EliteRealArm.recall` 从 `score_details['entity_boost']` 读 `entity`（此前读不存在的 `entity_score`，`channels.entity_hits` 恒为 0），`translate_zh.validate_window(..., question=True)` 只按题干行判断拉丁字母率（`translate_items` 对 `q:` 单元传入），`write_dataset` 以 `build_windows(zh)` 的谓词（cue 同时带英文与中文）记 `native_fallback` 并报告 `questions_translated` / `questions_native_fallback`，`translate_all` 不论 `--workers` 每完成十个请求打印进度，`--glossary-add NAME=中文`（`parse_glossary_add`；报告 `glossary_added` / `latin_names_note`——Jack == Jake 的合并是用户决定，EGOLIFE_ZH_PLAN §3），`--allow-tight-cap` 的帮助文本转义其 `%`，`compare_real.unique_run_roots` 对每次运行只陈述一次控制变量，`run_real` 把 `manifest.data_root` 记为绝对路径（`resolve()`）。测试：`tests/test_text_cjk.py`（5）、`tests/test_zh_rules.py`（5）、`tests/test_egolife_zh.py`（基准脚本，离线，6）、`test_wiring.py` / `test_foundation.py` / `test_extraction_lines.py` 的钉住。文档：CONTRACT §5.1 / §5.2 / §9.2 / §9.6 / §10.5 / §11.1（两种语言）、README §3 YAML 与提供者表、`docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md`。 |
| 110 | 次要 | §9.6 `extract_entities(zh_rules=True)` 仍运行既有的 `_CJK_PROPER_RE` 前瞻，因此 EgoLife-ZH 冒烟存储（`results/real/zh_smoke5_B`，2026-09-22）在 `atom_entity` 与 FTS 行文本里链接了 350 条 要求团队 / 都写 / 硬盘假装 / 他的家乡 / 做的布线 / 并让大家 之类的片段行，尽管人名规则与封闭的 `names` 术语表已在 979 / 980 个原子上给出干净主语。 | `zh_rules` 为 True 时跳过该前瞻（`hms_elite/utils/text.py`）；`zh_rules=False` 逐字节不变。`zh_rules` 下一段文本的实体为 `names` 命中、`_zh_name_candidates` 开头、FILE / IDENTIFIER / QUOTED 与拉丁 PROPER 串。摄取路径已把门面的抽取器传给每个 `create_atom`；`ops.consolidation` 与 `graph.operators` 仍用普通规则抽取器链接（未改）。`tests/test_zh_rules.py::test_zh_rules_skip_the_legacy_cjk_lookahead`。 |

上表带来的跨章节影响记录如下：§7 DDL（`sensor_trace`、`entity` kind、`interface_ep` transition、`target_kind` 取值、`compiled_states`、`ux_atoms_head`）；§2 枚举（`Route.preference`、`CompileMode.axis_graph`、`TransitionKind.interface_ep`、`SourceKind/NodeKind.entity`、`ObjectType.sensor_trace`、`KIND_ORDER`、`ABLATION_OF_MODE`）；§5 配置键（`graph.coarsen_l3`、`scheduler.decay_rate`、`scheduler.compile_state`、`ingest.emit_relation_atoms=True`）；§18.3 测试编号保持 1–63 并新增 18b、30b、45b、54b、56b、56c、64；§19.7 的报告/统计字典已完整重述。未解决项：无。
