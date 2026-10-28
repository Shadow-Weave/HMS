# HMS Elite
中文版：[README.zh.md](README.zh.md)

**Mem0 兼容的记忆 API，跑在一套 Experience-to-State 双平面存储之上。**
HMS Elite: a Mem0-compatible memory API over an experience-to-state dual-plane store (State Index + Support Graph).

- 与 Mem0 v2 相同的 `add / get / get_all / search / update / delete / delete_all / history / reset / close` 签名、返回信封、filters 语法与 config dict。
- 之外多出一层 **HMS 原生接口**：`add_events / compile / context / why / feedback / checkpoint / rollback / handoff / import_package / transitions / consolidate / axis_query / revoke / graph_stats / purge / vacuum`。
- 核心依赖只有 `pydantic`；默认 `rule` LLM + `hash` embedder + SQLite，**完全离线、确定性**，零 API key 即可跑通本文全部示例。
- 检索是**有界多通道召回 → 池内打分**（语义 ANN / FTS5 关键词 / 实体 / 新近度各有上限 K；4.1 节）：门控集合不超过各通道上限时与穷举打分逐位相同，`search` 在 N=30000 的 SQLite 库上中位 15.9 ms（改造前 1.6 s，或截断到最新 2000 条）。
- 一句话：**轴负责找，图负责继承，ContextIR 负责行动。**（Axes find, graph inherits, ContextIR acts.）

> 本文所有 Python 示例都可以离线运行；标有 `# （示意）` 的代码块除外（需要额外 SDK / 文件）。示例输出中的 id（`mem_…`、`atm_…`、`anc_…`）与 `created_at` 每次运行不同，其余文本逐字一致。

---

## 目录 Contents

1. [安装 Install](#1-安装-install)
2. [10 行快速上手 Quickstart](#2-10-行快速上手-quickstart)
3. [配置 Configuration](#3-配置-configuration)
4. [核心概念：轴负责找，图负责继承，ContextIR 负责行动](#4-核心概念轴负责找图负责继承contextir-负责行动)
   - 4.1 [检索架构 v3：有界多通道召回](#41-检索架构-v3有界多通道召回-bounded-multi-channel-recall)
5. [操作页 Operations](#5-操作页-operations)
6. [HMS → Lite → Mem0 映射表](#6-hms--lite--mem0-映射表)
7. [从 Mem0 迁移 Migrating from Mem0](#7-从-mem0-迁移-migrating-from-mem0)
8. [离线确定性与切换 LLM Provider](#8-离线确定性与切换-llm-provider)
9. [不变量 Invariants 及其强制方式](#9-不变量-invariants-及其强制方式)
10. [范围表：HMS 完整版中哪些被裁剪](#10-范围表hms-完整版中哪些被裁剪)
   - 10.1 [补齐的短板（两轮）](#101-补齐的短板两轮-gaps-closed-in-two-rounds)
11. [API 速查表 API reference](#11-api-速查表-api-reference)
12. [命令行 CLI](#12-命令行-cli)
13. [测试与开发 Testing / dev](#13-测试与开发-testing--dev)
14. [双语注释约定 Bilingual docstrings](#14-双语注释约定-bilingual-docstrings)

---

## 1. 安装 Install

```bash
pip install hms-elite                      # 核心：仅依赖 pydantic>=2.7；Python >= 3.10
pip install "hms-elite[numpy]"             # 可选：SQLite 索引用 numpy 做向量点积（不装也能跑，纯 Python 回退）
pip install "hms-elite[openai]"            # 可选：OpenAI LLM / embedder（llm.provider: deepseek 复用同一个 SDK）
pip install "hms-elite[anthropic]"         # 可选：Anthropic LLM
pip install "hms-elite[ollama]"            # 可选：Ollama LLM / embedder（只用标准库 urllib；extra 为空，仅让 DEPS_001 提示统一）
pip install "hms-elite[fastembed]"         # 可选：fastembed 本地嵌入（embedder.provider: fastembed，默认 BAAI/bge-small-en-v1.5，384 维）
pip install "hms-elite[milvus]"            # 可选：Milvus 向量索引（vector_store.provider: milvus；默认走 milvus-lite 本地 .db，也可连服务器）
pip install "hms-elite[spacy]"             # 可选：spaCy 实体抽取（ingest.entity_extractor: spacy；还需 python -m spacy download en_core_web_sm）
pip install "hms-elite[server]"            # 可选：FastAPI 服务面（hms-server / create_app）
pip install "hms-elite[test,dev]"          # 开发：pytest / pytest-mock / httpx（fastapi.testclient 所需）/ ruff / isort
```

从源码开发（本仓库为 flat layout：`hms_elite/` 与 `tests/`、`examples/` 并列）：

```bash
git clone <this repo> && cd hms-elite
pip install -e ".[test,dev]"
python examples/quickstart.py             # 或不安装：PYTHONPATH=. python examples/quickstart.py
```

默认数据库路径：`$HMS_DIR/hms.db`，未设置 `HMS_DIR` 时为 `~/.hms_elite/hms.db`；传 `{"store": {"path": ":memory:"}}` 则完全在内存中运行。

---

## 2. 10 行快速上手 Quickstart

Mem0 风格的 `add / search`，随后是 HMS 原生的 `compile(...).to_text()`、`why`、`feedback`、`transitions`：

```python
from hms_elite import HMS

hms = HMS({"store": {"path": ":memory:"}})  # rule LLM + hash embedder + sqlite index：离线、零 API key
scope = {"user_id": "alice", "agent_id": "coder", "run_id": "run-billing"}
hms.add("Do not modify the subscription state machine.", timestamp="2026-06-15T09:00:00Z", **scope)  # Mem0 形状
hms.add_events([  # 结构化 Agent 事件：不经过任何 LLM，直接闭合成 EP（action → outcome）
    {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:00:00Z",
     "payload": {"path": "src/billing/retry.ts", "patch": "+ backoff = 2 ** attempt",
                 "summary": "add exponential backoff", "intent": "fix checkout webhook retry"}},
    {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:04:00Z",
     "payload": {"suite": "billing_e2e", "passed": 11, "failed": 3,
                 "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}]}},
], **scope)
print(hms.search("retry.ts", filters=scope, top_k=1)["results"][0]["memory"])  # Mem0 形状的 search
print(hms.compile("fix checkout webhook retry", filters=scope, as_of="2026-06-15T12:00:00Z").to_text())
print(hms.why("why must retry.ts not be modified", filters=scope)["explanation"])
constraint_id = hms.get_all(filters={**scope, "obs_type": "constraint"})["results"][-1]["id"]
print(hms.feedback(constraint_id, "object_important")["actions"])
print(hms.transitions(filters=scope)[0][1:4])  # (s_t, a_t, s_t1, r_t, kappa_t, w_t) 的 a_t / s_t1 / r_t
hms.close()
```

输出（逐字）：

```text
[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
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
Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
-use[file://src/billing/retry.ts]-> Do not modify the subscription state machine.
[{'action': 'set_confidence', 'atom_id': 'atm_…', 'confidence': [0.7, 0.9]}, {'action': 'pin', 'atom_id': 'atm_…', 'topic': 'important'}]
('patch src/billing/retry.ts: add exponential backoff', {'ep_id': 'ep_…', 'r': 'negative', 'y': 'billing_e2e failed: subscription_state_regression'}, 'negative')
```

完整的六场景演示见 [`examples/quickstart.py`](examples/quickstart.py)（S1 偏好修订 → S2 时间 → S3 多会话实体 → S4 工作状态卡 → S5 侦探 → S6 检查点/反馈/回滚/转移 → consolidate + graph_stats），其真实输出（`S6 rollback` 里的 atom id 每次不同）：

```text
S1: User likes black coffee. | history: ['ADD', 'UPDATE']
S2: ['Sarah shipped the webhook retry design on May 6.']
S3: ['Sarah is reviewing the checkout refactor this week.', 'Sarah shipped the webhook retry design on May 6.', 'Sarah joined the billing team in March.']
# Work State Card
Scope: user_id=alice agent_id=coder run_id=run-billing | As of: 2026-06-15T12:00:00.000000+00:00 | Mode: auto
## Current Goal
fix checkout webhook retry
... （与上方卡片逐字相同）
S5: Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
-use[file://src/billing/retry.ts]-> Do not modify the subscription state machine.
S6 feedback: ['archive', 'create', 'revision_edge']
S6 rollback: ['atm_ca601545113a']
S6 transition: patch src/billing/retry.ts: add exponential backoff -> negative | witnesses: 2
S6 transition: revise -> negative | witnesses: 2
S6 transition: rollback -> neutral | witnesses: 1
{"entity_atoms": 5, "temporal_edges": 0, "causal_edges": 0, "conflicts": 0, "interface_eps": 0, "projected_edges": 0, "interface_eps_l3": 0, "skills": 0, "transitions": 0, "compiled_states": 1} {'temporal': 0, 'causal': 0, 'revision': 2, 'conflict': 0, 'use': 1, 'distill': 1}
```

---

## 3. 配置 Configuration

`HMS(config)` 接受 `HMSConfig`、dict 或 `None`；dict 走 `HMSConfig.from_dict`（等价于 `HMS.from_config(dict)`）。**Mem0 的 config dict 键名原样保留**（`vector_store / llm / embedder / history_db_path / reranker / version / custom_instructions`），HMS 自己的键放在旁边。下面这份 YAML 与 `from_config` 接受的 dict 一一对应（写出的都是默认值）：

```yaml
# hms.yaml —— 每个键都对应 HMSConfig 的一个字段；省略的键取默认值
vector_store: {provider: sqlite, config: {use_numpy: auto}}   # 索引封套（沿用 Mem0 键名）：sqlite | memory | milvus（需 [milvus] extra）
llm:          {provider: rule,   config: {language: auto, strict: false},   # rule | scripted | openai | anthropic | deepseek | ollama
               enable_vision: false, vision_details: null}   # enable_vision: add() 用 provider 的 describe_image 把图片部件描述成文字并入正文（openai / anthropic；rule 无视觉，图片仍计入 dropped_parts）；vision_details: auto | low | high
embedder:     {provider: hash,   config: {embedding_dims: 64, ngram_min: 2, ngram_max: 3}}   # hash | openai | ollama | fastembed
history_db_path: null          # Mem0 键；非空时映射到 store.path
reranker: null                 # Mem0 键；null = 未配置（search(rerank=True) 用默认离线 overlap 重排器）；也可写 {provider: overlap | llm, config: {...}}
version: lite-1                # v1.1 | v2 | lite-1（不驱动任何行为）
custom_instructions: null      # 追加到 observation_extraction prompt 的 Custom Instructions 段
store:     {busy_timeout_ms: 5000, journal_mode: wal,   # 省略 path → default_db_path()，即 $HMS_DIR/hms.db（未设 HMS_DIR 时为 ~/.hms_elite/hms.db，已展开）
            checkpoint_on_close: true}                  # 显式 path 原样交给 sqlite3——不展开 ~ 与 $VAR，请写绝对路径；":memory:" 为纯内存
                                                        # checkpoint_on_close: false → 文件库 close() 时不做 WAL checkpoint（WAL 留在磁盘、仍耐久，下次打开时恢复；需 Python >= 3.12，否则忽略）
ingest:    {revision_mode: inline, neighbourhood_k: 10, last_events_k: 10,
            emit_relation_atoms: true, closure_max_gap_hours: 24,
            entity_extractor: rule, entity_extractor_config: {},   # entity_extractor: rule | spacy（spacy 需 `[spacy]` extra 与模型；config 见 3.1）
            extraction_mode: single, extraction_output: lines, segment_max_chars: 400,
            verdict_min_similarity: 0.5, zh_rules: false}   # zh_rules：中文陈述规则（CONTRACT §9.6；默认关闭，英文输出逐字节不变）；verdict_min_similarity：LLM 的 NONE / REVISE / CONFLICT 裁决只在两原子主语一致时采纳，NONE 与未被佐证的 REVISE 还须新陈述与目标去掉头部（主语、共同谓词）后的词元包含度 ≥ 此值且至少共享两个词元（或较短陈述被整体包含）；REVISE 被取代提示 / 相同主语+谓词佐证时无条件采纳；否则按 ADD 存入（0.0 关闭；CONTRACT §12.6 第 6 步）；extraction_mode: single（每批一次 LLM 往返）| staged（观察 → EP → 修订，最多三次）；extraction_output: lines（默认，2026-09-18 起，CONTRACT 变更记录第 108 行；LLM 以 tab 分隔的 OBS / EP / DEC 纯文本回答而非 JSON，不用 json 模式，其余与 compact 同样本地推导，见 CONTRACT §6.6；仅 extraction_mode: single——staged 须显式写 full 或 compact）| full（可选；LLM 输出全部观察字段；2026-09-18 之前的默认）| compact（可选；只输出 9 个键——8 个字段加可选的 entities 提示——其余本地推导，见 CONTRACT §6.5；2026-09-11 上午曾切为默认，同日晚按 §6.5 的准确率规则回退）；超过 segment_max_chars 字符的消息先经 segmentation 切成抽取单元（段不产生新锚点，观察锚到父消息）
scoring:   {w_semantic: 1.0, w_keyword: 0.5, w_entity: 0.5, w_recency: 0.25, w_support: 0.25,
            half_life_days: 30.0, identifier_weight: 2.0}   # max_candidates：已弃用哨兵（默认 None，不截断任何东西；给出整数会发 DeprecationWarning）
retrieval: {k_semantic: null, k_keyword: null, k_entity: null, k_recency: null,   # null → 按 top_k 推导：k_semantic = max(4·top_k, 60)，其余 = max(2·top_k, 30)
            pool_max_factor: 4, hard_gate_pool: true, safety_limit: 50000}   # 池上限 = pool_max_factor·k_semantic；装得下池的门控集合整体入池；越过 safety_limit 只告警不截断
text:      {cjk_tokens: bigram}   # bigram（默认，CJK 字符二元组，索引侧与查询侧同用）| none（每个 CJK 连续串一个词元，英文中心基线；CONTRACT §9.6）
graph:     {min_cohesion: 0.3, coarsen_min_nodes: 2, coarsen_max_nodes: 12, coarsen_gap_hours: 48,
            temporal_window_hours: 24, coarsen_policy: rule, causal_proposer: rule, coarsen_l3: true}
compiler:  {budget_tokens: 1200, candidate_k: 50, graph_hops: 1, require_witness: true,
            slot_shares: {facts: 0.30, episodes: 0.25, constraints: 0.15, preferences: 0.10, skills: 0.05, audit: 0.15},
            min_constraints: 3, max_why_nodes: 60, default_mode: auto,   # default_mode: auto | fast | axis_graph | deep
            selection: shares,   # selection: shares（槽位份额预算，默认）| coverage（需求覆盖 + L3 → L2 → L1 分层下钻；也可按次 compile(mode='coverage')）
            context_session_budget_tokens: 1200, context_rows_top_k: 20, context_rows_header: 'Recent relevant memories:', context_timestamps: true}   # context_*：hms.context() 混合读上下文（5.13）——会话卡片预算、去重前的检索行数、两段之间的标题行、记忆行是否带 [YYYY-MM-DD HH:MM] 时间戳
scheduler: {phases: [orient, collect, consolidate, prune], staleness_tau: 0.8, confidence_min: 0.4,
            use_recency_half_life_days: 30.0, orient_window_days: 30, decay_rate: 0.1, compile_state: true,
            daemon_interval_s: null}   # start_daemon() 不带 interval_s 时的默认间隔（秒）；null 时必须显式传入
retention: {enabled: false, older_than_days: 90, run_in_prune: false}   # 5.12 逃生舱：enabled=false 时 purge(dry_run=False) 抛 ConfigError
visibility_default: private    # private < team < event < public
```


`ingest.extraction_output` 选择抽取的**输出契约**（2026-09-18 起默认 `lines`，见下文与 CONTRACT §6.6 / 变更记录第 108 行）：`full`（可选；2026-09-18 之前的默认）让 LLM 为每条观察输出全部 17 个字段；`compact`（可选）只要求 9 个键（`id / event_idx / text / type / speaker`、可选的 `supersedes_hint_ids / when / valid_to`，以及可选的 `entities` 提示——陈述点名的事物，0–6 个短字符串），其余（subject、predicate、objects、relation、date_observed、valid_from、confidence、attributed_to、sentiment、topic_tags）由 `hms_elite/ingest/derive.py` 按 CONTRACT §6.5 的推导表在本地补齐，并经与完整契约相同的规整函数落到原子上；其中 subject / predicate / objects 由 `llms.rule.statement_triple` 用规则 LLM R1–R9 的同一套线索表从第三人称陈述中读回（`User lives in Paris.` → `lives_in` / `["Paris"]`，`Sarah joined …` → 主语 `Sarah`），因此修订决策、历史行与图边与完整输出相同；实体轴取 `entities`、实体抽取器在 `text` 中找到的实体与 `speaker` 三者的并集（LLM 顺序在前，不分大小写去重），因此只有 LLM 才会列出的小写普通名词（`User prefers python for scripting.` 里的 `python`）与完整模式一样在轴上；`entities` 缺失或形状畸形时只回退到抽取器 ∪ `speaker`，不丢弃该条观察。动机来自 EgoLife 实测：每批 25 个窗口的抽取让 deepseek-flash 的输出 token 中位数达到 5,562（上限 8,192；22% 的批次触顶后被对半重提），而输出 token 是写入延迟与成本的主项（输出单价约为缓存未命中输入的 4 倍、缓存命中输入的 200 倍）。两种 `extraction_mode` 都支持 compact；`rule` LLM 同样实现两个紧凑任务（`observation_extraction_compact` / `memory_extraction_compact`）；`tests/test_extraction_compact.py` 断言 compact 下见证锚点、七个轴、取代提示解析与 `speaker → actor_id` 映射不变，六个聊天夹具在两种 `extraction_mode` 下的原子三元组、`entity_ids`、轴表、`history` 与 `graph_edges` 与 `full` 逐行相同，且 `full` 的提示词逐字节不变。2026-09-18 之前默认值是 `full`：2026-09-11 上午曾按两条件规则切为 `compact`，同日晚按 CONTRACT §6.5 追加的准确率规则回退。切换默认值的前两个条件均已满足：(a) 零 API 回放中 compact 的 hit@20 不低于 `full`——204 题两臂均 161 命中，MRR 0.5000 vs 0.4973，实体轴在 8,439 对配对原子上 compact ⊇ full（24 个严格超集、0 个收窄；`benchmarks/egolife/results/replay_ab_entities`）；(b) 带 `entities` 提示的最终紧凑提示词在真实 API 上完成同 18 批（395 窗口、deepseek-flash、max_tokens 8,192）冒烟并确认输出 token 降幅——`results/real/elite_compact_entities_smoke5_r2`（run_id `20260911T090326Z`）：18 次抽取请求，每次请求输出 token 中位 2,793 / 最大 4,312，`full` 为 5,631 / 8,192（−50%）；触顶截断 0/18 次 vs 8/34 次（`full` 有 6 批对半重提）；每次请求模型耗时中位 8.7 s vs 17.1 s（−49%）；prompt token 中位 6,643.5；672 次 add 结果（存入 606 条原子），窗口丢失 0；5 题答对 rows 3/5、card 4/5（`full` 为 3/5、3/5）；账单 $0.089961（28 次计费请求）。第一次尝试 `results/real/elite_compact_entities_smoke5` 在 12/18 批处中断（目录内 `INCOMPLETE.md`），由 `_r2` 取代。第三个条件（2026-09-11 晚加入 CONTRACT §6.5）不满足：compact 只有在同 204 题真实运行的记忆行准确率与 `full` 相差不超过 2 个点时才保持默认。同 204 题的真实 deepseek-flash 运行（`docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`）：`full` r3 记忆行 61.3%（125/204）/ 状态卡片 61.8%，hit@20 78.9%，空闲价 ¥11.56，每批模型耗时 14.9 s；compact 初版记忆行 53.9%（根因：模型把观察 i 与候选 i 配对判 NONE，1,685 个 NONE）；compact 修订版（DECISIONS 措辞重写 + `verdict_min_similarity` 守卫，`results/real/elite_compact_v2_full204`）记忆行 56.4%（115/204）/ 卡片 62.7%（128/204），hit@20 77.9% / 79.4%，¥4.35，每批 7.3 s，NONE 15，窗口丢失 0；mem0 56.9%。56.4 比 61.3 低 4.9 个点，默认回退为 `full`（直到 2026-09-18 切为 `lines`）。compact 仍完整支持，推荐用于成本敏感的批量灌入（写入费用 0.38 倍、每批耗时 0.49 倍）与以状态卡片上下文为阅读器输入的场景（卡片 62.7% 对 61.8%）；compact 把每条观察的置信度推导为 0.7，`topic_tags` / `sentiment` 按词表推导（CONTRACT §6.5 推导表），需要 LLM 直接给出这些值时保持 `full`。数字、口径与复现命令见 `docs/design/COMPACT_EXTRACTION_2026-09-11.md`。

`ingest.extraction_output: lines`（默认，2026-09-12 引入、2026-09-18 起为默认，CONTRACT §6.6）走第三条路：LLM 以 tab 分隔的纯文本回答 `memory_extraction_lines`——`OBS` 段每行 `<event_idx>⇥<T>⇥<陈述>[⇥@日期]`（`T` 为 F/P/H/C/T 五个类型字母），`EP` 段只在有行动 → 结果闭包时出现（`<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`），`DEC` 段只写非 ADD 裁决（`<OBS 行号>⇥REVISE|CONFLICT|NONE⇥<候选 id>⇥<理由>`，缺失即全部 ADD）——没有 JSON、不用 json 模式（`LLMBase.generate_lines` 不发送 `response_format`），由 `hms_elite/ingest/lines.py::parse_lines` 容错解析（围栏、散文、缺 tab、未知字母、被截断的末行、越界的 DEC 各自只丢一行），随后与 `compact` 走同一条本地推导路径；唯一的额外规则是行式格式没有说话人槽位，陈述开头的专有名（`Katrina must not …`）即说话人，进入实体轴与 `actor_id`（消息带 `name` 时 `name` 优先）。动机：4,851 窗口的 EgoLife 真实运行中，输出 token 占 `full` 写入费用的 83%（¥11.5 中 2.40M 输出 token；compact v2 ¥4.2 中 0.76M；mem0 ¥1.85 中 0.22M，输入未命中侧已持平），JSON 信封的开销是其承载文本的 2–4 倍，而零 API 回放表明检索只用到陈述文本与决策；`compact` 提示词自己写出的文本又低于 `full`（记忆行 56.4% 对 61.3%），因此行式提示词逐字保留 `full` 的陈述规则，只替换输出格式。仅支持 `extraction_mode: single`（配 `staged` 时 `IngestConfig` 以双语消息拒绝）；`rule` LLM 实现 `memory_extraction_lines`，`tests/test_extraction_lines.py` 断言六个聊天夹具下行式门面与 `full` 门面逐行、逐表相同（`actor_id` 的说话人规则除外）且 `full` / `compact` 提示词逐字节不变。2026-09-18 起 `lines` 是库默认（变更记录第 108 行）；`full` / `compact` 可选；真实 API 测量见下一段。

测量结果（2026-09-12，`results/real/elite_lines_full204`，204 题全量，deepseek-flash，空闲档）：记忆行上下文 54.9%（112/204），状态卡片上下文 63.2%（129/204，所有臂中最高）；hit@20 84.3% / 82.4%（最高）；每窗口输出 51 token（full 495、compact 157、mem0 45），每批模型耗时中位 3.76 s（full 14.9、compact 7.3、mem0 3.5），写入 + 回答全按空闲价 ¥2.24（mem0 ¥1.98、compact ¥4.35、full ¥11.56）；存入 12,994 条原子（2.68 条/窗口，库 72.5 MB）；判定 ADD 12,963 / NONE 20 / UPDATE 11；零截断、零丢失。结论：成本与写入速度达到 mem0 水平且召回更高；记忆行上下文的准确率低于 full 4 个点（有据答错 72 题），状态卡片上下文高于 full 1.4 个点。2026-09-18 起 lines 是库默认（变更记录第 108 行）：同一行式存储上的阅读端 A/B（`docs/design/READER_AB_2026-09-18.md` §六）测得 `hms.context()` 的混合上下文（记忆化会话卡片前缀 + 带时间戳的相关记忆行，记忆在前作答，见 5.13）70.1%（143/204），高于 full 的记忆行 61.3% / 卡片 61.8%，与记忆行的符号检验 p < 0.0001；full / compact 可选。

`ingest.extraction_output: lines` (the default; introduced 2026-09-12, default since 2026-09-18, CONTRACT §6.6) is the third route: the LLM answers `memory_extraction_lines` with tab-separated plain text — an `OBS` section of `<event_idx>⇥<T>⇥<statement>[⇥@date]` lines (`T` one of the five type letters F/P/H/C/T), an `EP` section only when an action → outcome closure exists (`<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`), a `DEC` section only for non-ADD verdicts (`<OBS line>⇥REVISE|CONFLICT|NONE⇥<candidate id>⇥<reason>`; absent means all ADD) — no JSON and no json mode (`LLMBase.generate_lines` sends no `response_format`), parsed tolerantly by `hms_elite/ingest/lines.py::parse_lines` (fences, prose, a missing tab, an unknown letter, a truncated last line, an out-of-range DEC each cost one line at most) and then derived locally on the same path as `compact`; the one extra rule is that the format has no speaker slot, so the statement's leading proper name (`Katrina must not …`) is the speaker and reaches the entity axis and `actor_id` (a message `name` still wins). Why: on the 4,851-window real EgoLife run output tokens were 83% of the `full` write cost (2.40M output tokens of ¥11.5; compact v2 0.76M of ¥4.2; mem0 0.22M of ¥1.85 with the input-miss side already at parity), a JSON envelope costs 2–4× the text it carries, and the zero-API replay showed retrieval uses only the statement texts and the decisions; the `compact` prompt's own texts also score below `full` (rows 56.4% vs 61.3%), so the lines prompt keeps the `full` statement rules verbatim and changes only the output format. Single mode only (`IngestConfig` rejects `staged` with a bilingual message); the `rule` LLM implements `memory_extraction_lines`, and `tests/test_extraction_lines.py` asserts that on the six chat fixtures the lines facade stores the same texts, events, anchors, entity ids and the same nine axis / history / graph / `eps` tables as the `full` facade, with the `observations` / `memory_atoms` rows identical except for the pinned derived columns (`confidence` 0.7, `relation` / `supersedes_hint_ids` without a hint slot, the speaker-rule `actor_id`) and that the `full` / `compact` prompts stay byte-identical. `lines` is the library default since 2026-09-18 (changelog row 108); `full` / `compact` stay optional; the real-API measurement is the next paragraph.

Measured (2026-09-12, `results/real/elite_lines_full204`, all 204 questions, deepseek-flash, off-peak): memory-rows context 54.9% (112/204), state-card context 63.2% (129/204, the highest of every arm); hit@20 84.3% / 82.4% (the highest); 51 output tokens per window (full 495, compact 157, mem0 45), median model time per batch 3.76 s (full 14.9, compact 7.3, mem0 3.5), write + answer re-priced at off-peak ¥2.24 (mem0 ¥1.98, compact ¥4.35, full ¥11.56); 12,994 atoms stored (2.68 per window, 72.5 MB); verdicts ADD 12,963 / NONE 20 / UPDATE 11; no truncation, no window lost. Conclusion: cost and write speed reach mem0's level with higher recall; rows-context accuracy is 4 points below full (72 evidence-present wrong answers), state-card accuracy 1.4 points above full. `lines` is the library default since 2026-09-18 (changelog row 108): the reader A/B on the same lines store (`docs/design/READER_AB_2026-09-18.md` §六) measured 70.1% (143/204) for `hms.context()`'s hybrid context (memoized session-card prefix + timestamped relevant rows, answered memories-first, 5.13) against full's rows 61.3% / card 61.8%, sign test vs rows p < 0.0001; `full` / `compact` stay optional.

```python
# （示意）PyYAML 不是 hms-elite 的依赖；用 JSON 文件时把 yaml.safe_load 换成 json.load 即可
import yaml
from hms_elite import HMS

with open("hms.yaml", encoding="utf-8") as f:
    hms = HMS.from_config(yaml.safe_load(f))
```

等价的 Python dict（这是可运行的）：

```python
from hms_elite import HMS, HMSConfig

config = {
    "vector_store": {"provider": "sqlite", "config": {}},
    "llm": {"provider": "rule", "config": {"language": "auto"}},
    "embedder": {"provider": "hash", "config": {"embedding_dims": 64}},
    "store": {"path": ":memory:"},
    "compiler": {"budget_tokens": 800, "default_mode": "auto"},
}
cfg = HMSConfig.from_dict(config)        # 校验；错误一律是 hms_elite.exceptions.ConfigError
print(cfg.llm.provider, cfg.embedder.provider, cfg.compiler.budget_tokens, cfg.db_path())
HMS(config).close()                      # 或 HMS.from_config(config) / HMS(cfg)
```

```text
rule hash 800 :memory:
```

### 3.1 Provider 表

| 封套键 | provider | config 类（`extra="forbid"`，拼错会给出 *Did you mean* 提示） | 说明 |
|---|---|---|---|
| `llm` | `rule`（默认） | `RuleLlmConfig(language="auto"\|"en"\|"zh", strict=False, …)` | 确定性规则抽取器（EN/ZH 词典），零依赖 |
| `llm` | `scripted` | `ScriptedLlmConfig(responses=[...], loop=False, raise_when_empty=True)` | 测试用：按队列回放响应 |
| `llm` | `openai` | `OpenAILlmConfig(model="gpt-5-mini", api_key, openai_base_url, temperature, max_tokens, top_p, top_k, http_client_proxies)` | 需要 `[openai]`；`describe_image` 走 chat.completions 的 `image_url` 内容部件（data URI 与 http URL 均可） |
| `llm` | `anthropic` | `AnthropicLlmConfig(model="claude-sonnet-4-6", api_key, anthropic_base_url, …)` | 需要 `[anthropic]`；`temperature` 仅在已安装 SDK 的 `Messages.create` 仍接受时才转发（anthropic 1.x 已移除采样参数）；`describe_image` 走 Messages API 的 `image` 块（data URI → base64 来源，http → url 来源） |
| `llm` | `deepseek` | `DeepSeekLlmConfig(model="deepseek-chat", api_key, deepseek_base_url, thinking="enabled"\|"disabled"\|null, …)` | 需要 `[openai]`（DeepSeek 走 OpenAI 兼容接口）；`thinking` 开关按请求转发；工具调用参数按 JSON 解析 |
| `llm` | `ollama` | `OllamaLlmConfig(model="llama3.1", ollama_base_url="http://localhost:11434", timeout=120)` | 零依赖（标准库 `urllib`）；本地模型 |
| `embedder` | `hash`（默认） | `HashEmbedderConfig(embedding_dims=64, ngram_min=2, ngram_max=3)` | 字符 n-gram → sha256 桶 → L2 归一化；确定性 |
| `embedder` | `openai` | `OpenAIEmbedderConfig(model="text-embedding-3-small", embedding_dims=1536, …)` | 需要 `[openai]` |
| `embedder` | `ollama` | `OllamaEmbedderConfig(model="nomic-embed-text", ollama_base_url="http://localhost:11434", timeout=120)` | 零依赖；维数在首次调用时推断并缓存 |
| `embedder` | `fastembed` | `FastEmbedEmbedderConfig(model="BAAI/bge-small-en-v1.5", embedding_dims=384, cache_dir, threads)` | 需要 `[fastembed]`；本地推理，无 API key |
| `vector_store` | `sqlite`（默认） | `SQLiteIndexConfig(use_numpy="auto"\|"never")` | 向量存在同一个 SQLite 文件的 `atom_vectors` 表 |
| `vector_store` | `memory` | `MemoryIndexConfig()` | 纯内存 dict |
| `vector_store` | `milvus` | `MilvusIndexConfig(uri="$HMS_DIR/milvus.db", token, db_name, collection="hms_atoms", index_type="AUTOINDEX"\|"HNSW"\|"FLAT", M, efConstruction, ef, batch_size, candidate_chunk, search_workers=1)` | 需要 `[milvus]`；本地 uri 必须以 `.db` 结尾（milvus-lite 嵌入模式），`http/grpc` uri 连服务器；命中经 `rank_hits` 重算分数，与 sqlite 逐位一致；`HMS.close()` 会释放客户端 |
| `reranker` | `overlap`（未配置时的默认） | `OverlapRerankerConfig(identifier_weight=2.0, top_k, text_key)` | 离线加权词元覆盖率；`search(rerank=True)` 时生效，行尾追加 `rerank_score` |
| `reranker` | `llm` | `LLMRerankerConfig(llm={provider, config}, max_chars=4000, fallback=True, top_k, text_key)` | 走 `rerank` 提示词契约；`llm` 省略则共用门面的 LLM；`LLM_002`（如 rule LLM）时回退 overlap；其他失败保持检索顺序 |
| `ingest.entity_extractor` | `rule`（默认） | `RuleEntityExtractorConfig(zh_rules=False, names=[])` | 正则实体抽取（FILE / IDENTIFIER / QUOTED / PROPER / CJK），零依赖；与 `utils.text.extract_entities` 逐字节一致 |
| `ingest.entity_extractor` | `spacy` | `SpacyEntityExtractorConfig(model="en_core_web_sm", labels={PERSON: PROPER, …}, disable=[], strip_articles=True)` | 需要 `[spacy]` 与模型（`python -m spacy download en_core_web_sm`，缺任一者 → `DependencyError`）；NER 区间**并入**规则结果而非替换（规则抓到的 FILE/IDENTIFIER 保留，重叠时规则胜出）；配置写在 `ingest.entity_extractor_config` |

### 3.2 Mem0 键：接受 / 改名 / 移除

| 类别 | 键 | 行为 |
|---|---|---|
| 接受 | `vector_store`, `llm`, `embedder`, `history_db_path`, `reranker`, `version`, `custom_instructions` | 同名同嵌套 |
| 改名（`DeprecationWarning`，自动重映射） | `custom_fact_extraction_prompt` → `custom_instructions`；`custom_update_memory_prompt` → `custom_instructions`；`index` → `vector_store`；`db_path` → `store.path` | |
| 移除（`ConfigError(CONFIG_003)`） | `graph_store`（"the support graph lives in the same SQLite file; remove the key"）；`enable_graph`（"the graph is always on"） | |
| 未知键 | 任何层级 → `ConfigError(CONFIG_001)`，消息带 `difflib` 提示 | |
| 未知 provider | `vector_store.provider` 不在 `{sqlite, memory, milvus}`（`milvus` 需 `[milvus]` extra；例如 Mem0 的 `qdrant`）→ `ConfigError(CONFIG_002)` | 这是一份原生 Mem0 dict **唯一**会失败的地方 |

```python
import warnings
from hms_elite import HMSConfig
from hms_elite.exceptions import ConfigError

mem0_dict = {"llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini"}},
             "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
             "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
             "history_db_path": "/tmp/x/history.db", "version": "v1.1"}
for bad in (mem0_dict, {"enable_graph": True}, {"scorng": {}}):
    try:
        HMSConfig.from_dict(bad)
    except ConfigError as e:
        print(e.error_code, "|", e, "|", e.suggestion)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cfg = HMSConfig.from_dict({"db_path": "/tmp/hms-elite-doc/hms.db", "custom_fact_extraction_prompt": "Extract facts."})
print(cfg.store.path, "|", cfg.custom_instructions, "|", [str(x.message) for x in w])
```

```text
CONFIG_002 | unknown index provider 'qdrant' (config key 'vector_store'); supported: memory, milvus, sqlite | HMS Elite ships 'sqlite' and 'memory' index providers ('milvus' via the optional [milvus] extra); use {'provider': 'sqlite'}
CONFIG_003 | config key 'enable_graph' was removed in HMS Elite: the graph is always on | the graph is always on
CONFIG_001 | Unknown config key 'scorng'. Did you mean 'scoring'? | None
/tmp/hms-elite-doc/hms.db | Extract facts. | ["config key 'custom_fact_extraction_prompt' is deprecated; use 'custom_instructions'", "config key 'db_path' is deprecated; use 'store.path'"]
```

（`CONFIG_001` 抛出前还会通过 `logging` 在 stderr 打一行完整的 pydantic 错误 `HMSConfig validation failed: …`。）

---

## 4. 核心概念：轴负责找，图负责继承，ContextIR 负责行动

```text
 SourceEvent ──▶ Anchor(sha256) ──▶ Observation / EP ──▶ MemoryAtom（memory_id = Mem0 id 谱系；atom_id = 版本）
 （message / tool_call / file_diff / test_log / sensor_trace）                │
                                                                              ├──▶ State Index  ── 7 条轴：time · validity · entity · topic · role · sentiment · scale
                                                                              │    热路径：add / get / get_all / search / axis_query          「轴负责找」
                                                                              │
                                                                              └──▶ Support Graph ── 6 种边：temporal · causal · use · distill · revision · conflict
                                                                                   继承路径：why / compile(deep) / consolidate / rollback     「图负责继承」
                                                                                        │  coarsen → InterfaceEP(L2/L3) → project → refine
                                                                                        ▼
                                              StateCompiler ──▶ ContextIR（六个槽位 + 证据）──▶ Work State Card · PC-IR · StatePackage 「ContextIR 负责行动」
                                                                                        │
                                                            transitions 表：(S_t, a_t, S_{t+1}, r_t, κ_t, w_t)   ── EP / InterfaceEP / causal / revision / rollback  「经验负责训练」
```

**轴负责找（State Index）。** 每条记忆都是一个 `MemoryAtom`，同时挂在七条轴上：`time_anchor`（事件时间）、有效期 `valid_from / valid_to`（读取时用 `as_of` 切片）、实体（`atom_entity`，按归一化文本去重，`run_id` 不参与实体身份）、话题（`atom_topic`）、`role`、`sentiment`、`scale_level`（`state / event / relation / entity / L1_micro / L2_meso / L3_macro`）。`search()` 先由 planner 决定路由（`recall / preference / temporal / entity / work_state / detective / handoff`），把硬门控（时间窗、实体、`obs_type`）下推为 SQL，由索引支撑的有界通道（语义 ANN / FTS5 关键词 / 实体 / 新近度）在门控集合内召回一个池，再只对池融合五个信号（semantic、keyword、entity、recency、support；4.1 节）。`search()` 永远不读任何 `graph_*` 表——这是被 `store.trace` 断言的。

**图负责继承（Support Graph）。** 每个 atom 版本就是一个图节点（`node_id == atom_id`），边只有六种且**每条边至少一个证据锚点**（`witness_anchor_ids`）。`add_events()` 把 `file_diff / tool_call` 与随后的 `test_log` 闭合成一个 EP 六元组 `(u, b, a, y, r, κ)`：负向 EP 会蒸馏出一条 `Do not re-apply …` 约束（`distill` 边），用到的背景约束以 `use` 边相连，相邻 EP 以 `temporal / causal` 相连。`consolidate()` 把内聚的 EP 组 `coarsen` 成 InterfaceEP（L2，再到 L3），把内部边 `project` 成带 `witness_edge_id` 的边界边；`refine()` 是唯一能"打开"高层对象的方法，只被 `why()` 与 `compile(mode='deep')` 调用。修订、冲突、回滚都以边的形式留在图里，从不删除。

**ContextIR 负责行动（Compiler）。** `compile(query, filters)` 把轴召回的候选 + 图上一跳邻居 + InterfaceEP 精炼出的不变量，按确定性规则塞进六个互斥槽位 `facts / episodes / preferences / constraints / skills / audit`，在 token 预算内保留至少 `min_constraints` 条约束，产出 `ContextIR`。`to_text()` 渲染成 Work State Card（Current Goal / Valid Facts / Invalidated Assumptions / Failed Paths / Constraints / Next Action / Evidence），`to_pcir()` 降为带四条证明义务（`evidence_coverage / no_unresolved_conflict / governance / not_revoked`）的 PC-IR，`handoff()` 打包成可被另一个实例 `import_package()` 的 `StatePackage`。每一次 `feedback / checkpoint / rollback / consolidate` 自身也是一个 SourceEvent + Anchor，因此库里没有任何一行数据缺少通往原始证据的链条。

### 4.1 检索架构 v3：有界多通道召回 Bounded multi-channel recall

改造前 `search()` 的成本与库大小成正比：门控 SQL 选出候选集 G 之后，`scoring.semantic_scores` 以 `index.search(vector, candidate_ids=G, top_k=len(G))` 向索引索要**全部**候选的语义分，再逐个水合（pydantic 校验 + 链接表读取）并计算 keyword / entity / recency / support 四路信号。每个候选约 45 µs（`docs/design/RETRIEVAL_DIAGNOSIS.md` T2），N=3000 时 133 ms、N=30000 时 1.6 s（T3，不截断）；`scoring.max_candidates=2000` 的"最新优先"截断把 N=30000 压回 108–122 ms，代价是 N > 2000 之后较旧的记忆**静默**退出候选，与相关性无关——这是正确性缺陷，不只是性能。Milvus 提供者同样没有获益：`top_k=len(ids)` 把 HNSW 变成暴力扫描，每 1024 个 id 一块、每块 13–30 KB 表达式、约 58 ms（T4–T6）。对照组 mem0 + FAISS（IndexFlat）同规模 0.20–0.35 ms。

v3 把打分范围从 G 缩到一个有界的池：**每路通道由索引支撑、各自有界 K、输出确定性排序 → 并集去重入池（记录每条来自哪些通道）→ 只对池水合并计算五路信号 → 融合 → 阈值 → 排序 → top_k**。`compile()` 的候选阶段（`candidate_k=50`，阈值 0）走同一套召回，不再自己穷举。

| 通道 | 依托 | 顺序 / 上限 | 何时运行 |
|---|---|---|---|
| `semantic` | 索引提供者。`sqlite`：进程内缓存的 float64 矩阵（以 `data_generation / rebuild_generation / data_version` 为键增量同步），一次矩阵乘挑出前 K、再用 `dot` 逐位重算；`memory`：同一个 `select_top_k`；`milvus`：把 scope / `as_of` 有效期 / 尺度 / `obs_type` / `atom_type` 门下推成标量表达式做**原生过滤 ANN**，不再传 id 列表 | `(score DESC, atom_id ASC)`；`K_sem = max(4·top_k, 60)` | 有查询向量时总是运行 |
| `keyword` | SQLite FTS5 虚拟表 `atom_fts`（模式版本 3；写入时按项目自己的 `tokenize()` 预分词，CJK 二元组与标识符经 `unicode61` 可检索）；无 FTS5 模块时退回有界的 `LIKE` | 两阶段（先全部词元 AND，再任一词元 OR）；每阶段只读**最新匹配的窗口**（`ORDER BY rowid DESC LIMIT max(4·K_kw, 64)`，倒序遍历倒排表、提前停止，不给全部命中排 bm25），窗口内重新施加门控并按 `bm25 DESC, time_anchor DESC, atom_id ASC` 排列，不足 K 时倍增窗口分页；`K_kw = max(2·top_k, 30)` | 规划给出内容词元或标识符词元时（与实体 / 新近度分支合成一条 UNION ALL 语句） |
| `entity` | `atom_entity` 索引 | `time_anchor DESC, atom_id ASC`；`K_ent = max(2·top_k, 30)` | 规划给出实体线索时（硬实体门下 G 只经 atom_entity 以不排序的 `pool_max + 1` 探测读取一次，装不下池时该通道是同一语句里按时间序的有界分支） |
| `recency` | `ix_atoms_time` | 轴序 `time_anchor DESC, created_at DESC, atom_id ASC`；`K_rec = max(2·top_k, 30)` | 路由为 `temporal` 或带硬时间门；`hard_gate_pool=false`；或其它通道合计不足 `K_rec` 时兜底 |
| `gate` | 门控集合本身 | 轴序 | 规划给出硬门（时间 / 实体 / `obs_type`）或任何级别的 `len(G) <= pool_max` 时整体入池，不走通道 |

池上限 `pool_max = pool_max_factor · K_sem`（默认 4 × 80 = 320，`top_k=20`）。每个通道都在该级别的门控集合 G **之内**搜索（scope、有效期、过滤 DSL、尺度、规划门控随每个通道下推），所以池 ⊆ G。四级回退阶梯不变：（规划门控，路由默认尺度）→ 去掉默认尺度（`scale_fallback`）→ 去掉规划门控（`fallback`）；每一级都走同一套召回，第一个非空级别胜出并记入 `primary_axis`。

- **配置**：`retrieval: {k_semantic, k_keyword, k_entity, k_recency, pool_max_factor: 4, hard_gate_pool: true, safety_limit: 50000}`（第 3 节 YAML；`null` 表示按 `top_k` 推导）。`scoring.max_candidates` 保留为已弃用哨兵：默认 `None`，给出整数只发 `DeprecationWarning`，不再截断任何东西。`safety_limit` 是远高于正常的安全上限：门控集合或池越过它时记一条 `logger.warning`，**绝不静默截断**。
- **解释块**：`explain=True` 时 `score_details` 在 Mem0 的 7 个键与 HMS 的 7 个键之后追加 `recalled_by`（该行来自哪些通道；`['gate']` 表示整个门控集合都打了分）、`pool_size`、`channel_sizes`；`axis_gates` 语义不变。`axis_query()` 的 `AxisResult.retrieval` 与 `compile()` 的 `ContextIR.retrieval` 带同一份 `{pool_size, channel_sizes, channels_used, gate_count, whole}`（`gate_count = -1` 表示 Milvus 原生下推、未枚举 G）。
- **等价性定理**（`tests/test_retrieval.py::test_equivalence_theorem`：`sqlite` / `memory` 参数化，N ∈ {5, 30, 60}，20 个查询 × `top_k ∈ {3, 5, 10}`；`test_equivalence_theorem_milvus` 在 milvus-lite 上重复）：当 `|G| <= min(K_sem, K_kw, K_ent, K_rec)`（或 `hard_gate_pool` 且 `|G| <= pool_max`）时池等于 G，新路径每个命中的 `(atom_id, score)` 与 `score_details` 的前 14 个键与旧穷举路径**逐位相同**。测试套件与黄金场景 S1–S6 的语料都满足 `|G| < 60`，因此没有一条现有断言因本次改造而改变。`|G| > K` 时池是 G 的子集：损失有界，且可从 `recalled_by != ['gate']` 识别。
- **Milvus 标量一致性**：SQLite 仍是唯一真源。`archive_atom / restore_atom / set_valid_to`（`delete / feedback / rollback / prune` 都经由它们）在 COMMIT 后按提交顺序回调 `store.attr_listeners` → `index.update_attrs(atom_id, attrs)`（`IndexBase` 的非抽象钩子：`sqlite` 更新缓存标量，`memory` 空操作，`milvus` 重写该行）；水合后仍按 SQL 校验有效期，所以过期的 Milvus 标量只可能漏检、不可能误检。`tests/test_index_milvus.py::test_archive_hides_from_every_provider_and_rollback_restores` 在三种提供者上断言归档后 `search` 不再返回该行、`rollback` 后再次返回。
- **表数**：FTS5 会创建 5 张影子表。`schema.TABLES` 只列 22 张基表，另有 `VIRTUAL_TABLES = ('atom_fts',)`；`stats()` 与 `store.trace` 只覆盖基表；模式版本 v3 经 `MIGRATIONS` 阶梯（v1 → v2 → v3，v3 从 `memory_atoms` 回填 `atom_fts`）升级旧库，全新建库与逐级升级的 `sqlite_master` 行逐条相同（`test_fresh_database_is_created_directly_at_schema_version`、`test_schema_v3_backfills_fts_from_a_v2_file`）。

实测（`docs/design/RETRIEVAL_BENCH_2026-09-10.md`：`hash@64` 嵌入器，`top_k=20`，`threshold=0.1`，5 个查询各取中位数再取中位，预热 1 次、计时 20 次；oracle = 旧穷举路径的 top-20，重叠率 = |新 ∩ oracle| / 20）：

| 臂 | N | search 中位 (ms) | p90 (ms) | 与穷举 top-20 的重叠率（均值 / 最小） |
|---|---|---|---|---|
| mem0 2.0.20 + FAISS (IndexFlat) | 300 / 3000 / 30000 | 0.20 / 0.23 / 0.32 | 0.21 / 0.23 / 0.33 | 0.98 / 0.99 / 0.96（对 FAISS 自身暴力 top-20，不是融合分，不可横向比较） |
| elite `sqlite` | 300 / 3000 / 30000 | 2.71 / 2.06 / 3.41 | 5.92 / 3.40 / 7.86 | 1.0 / 0.80 / 0.62（最小 1.0 / 0.45 / 0.20） |
| elite `milvus`（milvus-lite，HNSW） | 300 / 3000 / 30000 | 11.6 / 49.2 / 14.4 | 11.8 / 50.3 / 53.2 | 1.0 / 0.81 / 0.65（最小 1.0 / 0.50 / 0.20） |

- 与改造前对照：`sqlite` N=30000 从 172–210 ms（`max_candidates=2000`，且静默丢旧记忆）/ 1.6 s（不截断）降到 2.1–7.1 ms（中位 3.41；池化召回落地时为 13.6–19.6 ms，之后 CONTRACT 变更日志 90–96 的精简——`axis_lite` 轻量打分行、只水合最终 top_k、单条 `UNION ALL` 召回语句、FTS5 最新窗口、Milvus 主键读块——再降 ×4.7）；3000 → 30000（N ×10）的增长为 ×1.7，低于线性。写入延迟与 N 无关（0.7–1.2 ms/条）；mem0 的 `_save` 每次 add 重新 pickle 整个 docstore，N=30000 时 48–96 ms/条。
- 未达标处：`milvus` 臂 search@30000 未进入个位数毫秒（中位 14.4 ms，13.2–53.2 ms；`sqlite` 臂五个查询 2.1–7.1 ms 已达标）；N ≥ 3000 时重叠率低于 0.9（`sqlite` 0.80 / 0.62，`milvus` 0.81 / 0.65）。归因（BENCH §3–§4）：`sqlite` 臂剩余成本在实体门的 G 枚举与 FTS5 窗口（entity / identifier 6.7–7.1 ms，其余 2.1–3.4 ms）；漏掉的 oracle 原子在模板语料里与命中原子的 keyword / entity 分完全并列，穷举的 top-20 由 hash 嵌入 0.40–0.46 之间的语义分与新近度决定，而这些原子在语义通道排第 100–6000 名、bm25 通道第 150–3000 名——这是有界召回的固有损失，换用有语义区分度的嵌入器（`fastembed` / `openai`）会直接改变该结论。`milvus` 臂的成本是 milvus-lite 每次 RPC 的固定开销（N=3000 时三个查询的池内 `candidate_ids` 读取各约 49 ms，因此 3000 反而比 30000 慢；N=30000 时实体查询 53 ms、其余 13–17 ms）。

### 4.2 消融矩阵 Ablation matrix

`hms_elite.eval` 把 §10 的九个任务在四种只读消融下各跑一遍：每个任务一个新的离线 HMS（RuleLLM + HashEmbedder + SQLite 内存库），任务先构建语料，再由四种探针只读地查询同一实例。下表就是这段代码的输出（`tests/test_readme.py` 逐字比对）：

```python
from hms_elite import HMS, HMSConfig
from hms_elite.eval import run_ablation, render

report = run_ablation(lambda: HMS(HMSConfig(store={"path": ":memory:"})))  # 九个任务 × 四种消融，全离线
print(render(report))
```

```text
# HMS §12 Phase 5 ablation

| task | Axis only | Graph only | Axis + Graph | Axis + Graph + Support |
| --- | --- | --- | --- | --- |
| preference_recall | 1.00 | 0.75 | 1.00 | 1.00 |
| temporal | 1.00 | 0.25 | 1.00 | 1.00 |
| multi_session | 1.00 | 1.00 | 1.00 | 1.00 |
| handoff | 0.86 | 0.86 | 0.86 | 1.00 |
| rollback | 0.40 | 1.00 | 1.00 | 1.00 |
| conflict_revision | 0.33 | 0.83 | 0.83 | 1.00 |
| evidence_audit | 0.67 | 0.83 | 0.83 | 1.00 |
| skill_distillation | 0.25 | 0.75 | 0.75 | 1.00 |
| world_model_transition | 0.33 | 0.83 | 0.83 | 1.00 |
| mean | 0.65 | 0.79 | 0.90 | 1.00 |
```

- 列：`Axis only` = `fast` 模式（只查轴，不碰 `graph_*` 表）；`Graph only` = 只从种子沿图遍历，不做轴投影与时间门（评测专用，`compile()` 没有这个模式）；`Axis + Graph` = `axis_graph`；`Axis + Graph + Support` = `deep`。分数是该任务全部检查项的均值（0–1）；`render(report, details=True)` 列出每个失分的检查项与原因，`to_json(report)` 带每格的延迟 / LLM 调用 / token 数（离线全为 0）。
- 读数：`Axis + Graph + Support` 九项全为 1.00；在 handoff / rollback / conflict_revision / evidence_audit 上相对 `Axis only` 为 0.86 / 0.40 / 0.33 / 0.67 → 1.00，没有一项低于 `Axis only`。`Axis only` 的失分全部落在需要图遍历（修订 / 冲突 / 蒸馏边、见证）或 PC-IR 证明义务的检查项；`Graph only` 在 preference_recall / temporal 上失分，因为没有最新投影与时间门。`tests/test_ablation.py` 在 `sqlite` 与 `memory` 两种索引上钉着这张矩阵（两次运行逐格相同、`deep >= fast`）。

### 4.3 EgoLife 流式检索对照 EgoLife streaming benchmark

EgoLife A1_JAKE 204 题、4851 个转写窗口按 query_time 流式写入，五臂共用同一窗口切分、同一 `hash@64` 嵌入、同一时间线（每步保留集 sha256 逐位相同）、`top_k=20`、`threshold=0.1` 与同一确定性阅读器；全离线，mem0 侧 LLM 为桩。主指标是证据命中（gold = 包含 target_time 的窗口进入 top-k）；阅读器四选一只是代理（并列计错），oracle = 阅读器只读 gold 窗口，随机 25%。方法、分类型表、成本表与"不能得出的结论"见 `docs/design/EGOLIFE_BENCH_2026-09-10.md`；数据在 `benchmarks/egolife/results/final/`（`summary.json` / `summary.md` / `questions_<arm>.jsonl` / `manifest.json`），代码在 `benchmarks/egolife/`。

| 臂 | hit@1 | hit@5 | hit@20 | MRR | nb-hit@20 (±60 s) | reader acc first-50 (valid-only, n) | reader acc all (valid-only, n) | oracle 上限 all (valid-only, n) | oracle first-50 | 随机 |
|---|---|---|---|---|---|---|---|---|---|---|
| mem0_faiss | 2.0 | 6.4 | 10.3 | 0.037 | 14.2 | 26.0 (34.2, 38) | 23.5 (29.3, 164) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| mem0_milvus | 8.3 | 15.2 | 18.1 | 0.117 | 27.9 | 34.0 (37.8, 45) | 27.9 (32.4, 176) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_matched | 2.0 | 6.4 | 10.8 | 0.037 | 14.7 | 26.0 (34.2, 38) | 23.5 (29.3, 164) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_full | 7.8 | 20.1 | 31.4 | 0.134 | 39.2 | 28.0 (34.2, 41) | 25.5 (29.9, 174) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_infer | 10.3 | 20.6 | 33.8 | 0.154 | 40.7 | 32.0 (45.7, 35) | 23.5 (35.6, 135) | 65.2 (76.9, 173) | 70.0 | 25.0 |

### 4.4 EgoLife 真实 LLM 评测（仅 Elite）EgoLife real-LLM run (Elite only)

deepseek-flash 抽取与回答、本地 fastembed 嵌入、并发 1、204 题流式写入，两个回答上下文（检索行 / 状态卡片）共用一次写入。r3 是库端修完「输出截断对半重提」与「attributed_to 规整」之后的一轮；首轮 r1 有 31.3% 窗口未入库，作对照。全表与口径见 `docs/design/EGOLIFE_REAL_LLM_ELITE_2026-09-11.md`（`benchmarks/egolife/report_real.py` 从 jsonl 独立重算）。
deepseek-flash extracts and answers, local fastembed embeds, concurrency 1, 204 questions streamed; the two answer contexts (rows / state card) share one physical retain. r3 is the run after the library fixes (bisect on truncated output, `attributed_to` coercion); r1 lost 31.3% of windows and is the baseline. Full tables and definitions: `docs/design/EGOLIFE_REAL_LLM_ELITE_2026-09-11.md` (recomputed from the jsonl by `benchmarks/egolife/report_real.py`).

| 轮 run | rows 准确率 acc | card 准确率 acc | hit@20 批次 batch | 窗口丢失 windows lost | 实际账单 billed | 全按非高峰 at off-peak | 写入每步中位/P95 retain step median/P95 | 回答中位 answer median |
|---|---|---|---|---|---|---|---|---|
| r3（修复后 fixed） | 61.3% (125/204) | 61.8% (126/204) | 78.9% | 0 / 4851 | $3.29（773/886 请求在高峰档 peak tier） | $1.78 | 18.7 s / 174 s | 0.78 s |
| r1（首轮 baseline） | 53.4% (109/204) | 60.3% (123/204) | 63.7% | 1519 / 4851 | $1.11 | $1.09 | 16.9 s / 72 s | 0.62 s |

紧凑抽取契约（`extraction_output: compact`，可选；2026-09-11 上午曾为默认，同日晚因 204 题真实运行记忆行 56.4% 对 `full` 61.3% 超出 2 个点容忍而回退；2026-09-18 起库默认为 `lines`，变更记录第 108 行，见 `docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`）与 r3 前 18 批的同批次对照、204 题零 API 回放 A/B 及默认值决定的依据见 `docs/design/COMPACT_EXTRACTION_2026-09-11.md`。
The compact extraction contract (`extraction_output: compact`, optional; made the default on the morning of 2026-09-11 and reverted the same evening because the real 204-question run scored 56.4% on memory rows against 61.3% for `full`, outside the 2-point tolerance; the library default is `lines` since 2026-09-18, changelog row 108 — `docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`), its same-18-batch comparison against r3, the 204-question zero-API replay A/B and the evidence behind the default decision: `docs/design/COMPACT_EXTRACTION_2026-09-11.md`.

---

## 5. 操作页 Operations

以下示例按顺序运行在同一个 `hms` 上（`U = "alice"`），与 [`examples/quickstart.py`](examples/quickstart.py) 和测试套件的黄金场景 S1–S6 使用完全相同的输入。

```python
from hms_elite import HMS

hms = HMS({"store": {"path": ":memory:"}})
U = "alice"
```

### 5.1 `add` — Mem0 形状的写入，多出 `UPDATE / NONE` 与 HMS 字段

```python
s1 = {"user_id": U, "run_id": "run-coffee"}
r1 = hms.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", **s1)
r2 = hms.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", **s1)
row = r1["results"][0]
print({k: row[k] for k in ("id", "memory", "event", "version", "atom_type", "scale_level", "valid_from", "confidence")})
print(row["hms"])
print({k: r2["results"][0].get(k) for k in ("id", "memory", "event", "version", "revises")})
print(r2["results"][0]["id"] == row["id"])
```

```text
{'id': 'mem_…', 'memory': 'User likes oat-milk lattes in the morning.', 'event': 'ADD', 'version': 1, 'atom_type': 'state', 'scale_level': 'state', 'valid_from': '2026-04-02T08:00:00.000000+00:00', 'confidence': 0.7}
{'time_anchor': '2026-04-02T08:00:00.000000+00:00', 'entity_ids': [], 'topic_tags': ['coffee', 'preference'], 'sentiment': 'positive', 'subject': 'user', 'predicate': 'likes', 'obs_type': 'preference_event', 'source_kind': 'observation', 'source_id': 'obs_…', 'source_ep_ids': [], 'extracted_by': 'rule@v1', 'visibility': 'private', 'archived_reason': None}
{'id': 'mem_…', 'memory': 'User likes black coffee.', 'event': 'UPDATE', 'version': 2, 'revises': 'atm_…'}
True
```

- 签名与 Mem0 一致：`add(messages, *, user_id, agent_id, run_id, metadata, timestamp, expiration_date, infer, memory_type, prompt)`；`messages` 可以是 str、`{"role","content"}` dict 或其列表；`system` 消息只做锚定不抽取。
- 返回 `{'results': [row + 'event': 'ADD'|'UPDATE'|'NONE' (+ 'revises') (+ 'conflicts_with')]}`。Mem0 v2 只会返回 `ADD`；HMS Elite 在检测到同主语的取代（supersession）时返回 `UPDATE`——**同一个 `id`（谱系 id `mem_…`），`version` +1**，旧版本被归档并以 `revision` 边相连；互斥谓词冲突时返回 `conflicts_with`。`NONE` / `UPDATE` 只在 LLM 裁决的目标与新陈述主语一致、且去掉头部（主语、共同谓词）后确实相似时生效（`ingest.verdict_min_similarity`，默认 0.5，否则该行按 `ADD` 存入）——紧凑提示词曾诱发按下标配对的 NONE，见 CONTRACT §6.5 / 变更记录第 103、104 行。
- `timestamp=` 被当作**事件时间**（`time_anchor` / `valid_from`），`created_at` 是摄入时间；`expiration_date="2026-05-17"` → `valid_to = 2026-05-18T00:00:00.000000+00:00`（次日零点 UTC）。
- `infer=False` 写入原始 `message` 原子（行里多出 `actor_id / role`）；`infer=True` 的观察行在消息带 `name`、或 LLM 点名了说话人（`speaker`）时也带 `actor_id`（消息 `name` 优先；Mem0 只在 infer=False 路径给 `actor_id`，这是有意的偏离）；`agent_id` + `memory_type="procedural_memory"` 走 Mem0 的过程记忆分支，产出一条 `L1_micro` 的 episode 原子（Mem0 markdown 结构）。
- 行的字段顺序固定：Mem0 的 `id, memory, hash, metadata, score?, created_at, updated_at` + 提升键 `user_id?, agent_id?, run_id?, actor_id?, role?, attributed_to?, expiration_date?, memory_type?` + HMS 的 `atom_id, version, atom_type, scale_level, anchor_ids, valid_from, valid_to, confidence, hms{…}`。`metadata` 只含调用方自己的键。

中文同样走 `rule` LLM（EN/ZH 词典）：

```python
hms.add("我喜欢燕麦拿铁。", user_id="bob", timestamp="2026-04-02T08:00:00Z")
zh = hms.add("我现在改喝黑咖啡了。", user_id="bob", timestamp="2026-04-20T08:00:00Z")["results"][0]
print(zh["memory"], zh["event"], [r["memory"] for r in hms.search("用户喜欢什么咖啡", filters={"user_id": "bob"})["results"]])
```

```text
用户喜欢黑咖啡。 UPDATE ['用户喜欢黑咖啡。']
```

### 5.2 `add_events` — 结构化 Agent 事件，零 LLM，直接闭合成 EP

```python
scope = {"user_id": U, "agent_id": "coder", "run_id": "run-billing"}
hms.add("Do not modify the subscription state machine.", timestamp="2026-06-15T09:00:00Z", **scope)  # 背景约束
events_fail = [
    {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:00:00Z",
     "payload": {"path": "src/billing/retry.ts", "patch": "+ backoff = 2 ** attempt",
                 "summary": "add exponential backoff", "intent": "fix checkout webhook retry"}},
    {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:04:00Z",
     "payload": {"suite": "billing_e2e", "passed": 11, "failed": 3,
                 "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}]}},
]
ra = hms.add_events(events_fail, **scope)
print([(r["memory"], r["atom_type"]) for r in ra["results"]])
print({k: ra[k] for k in ("events", "eps", "edges", "constraints")})
```

```text
[('file_diff src/billing/retry.ts: add exponential backoff', 'event'), ('test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression', 'event'), ('[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression', 'episode'), ("Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'state')]
{'events': [{'event_id': 'evt_…', 'anchor_id': 'anc_…', 'object_type': 'file_diff', 'dedup': False}, {'event_id': 'evt_…', 'anchor_id': 'anc_…', 'object_type': 'test_log', 'dedup': False}], 'eps': ['ep_…'], 'edges': ['edg_…', 'edg_…'], 'constraints': ['atm_…']}
```

- 五种可抽取的 `object_type`：`message`（走 LLM 抽取）、`tool_call / file_diff / test_log / sensor_trace`（结构化，**永不经过 LLM**）；payload 会被归一化成固定形状（例如 `test_log.status` 由 `failed > 0` 推导，`failed_tests` 是 `failures` 的别名），锚点哈希 `sha256` 正是算在这个归一化 payload 上。内部类型（`feedback / checkpoint / rollback / system / scheduler`）传入会得到 `VALIDATION_007`。
- 闭合规则：`file_diff / tool_call` 压入本 run 的 open actions，`test_log` 弹出最近的一个（跨批次时通过 `closure_max_gap_hours` 内的 `last_open_action` 查找）并生成一条 EP：`u` = action 的 `intent`（否则上一条 EP 的 `y`），`b` / `κ.constraints` = 背景约束 ∪ payload 自带 `constraints`，`a = patch <path>: <summary>`，`y = <suite> failed: <tests>` / `passed`，`r ∈ {positive, negative, neutral, unknown}`，锚点 `[outcome, action]`。
- 负向 EP 额外产出一条派生约束 `Do not re-apply '<a>': <y>` 与 `distill` 边；背景约束以 `use` 边（见证 = action 锚点）相连；下一条 EP 与上一条以 `temporal` + `causal` 相连。因此 `events_fail` 得到 4 行、1 个 EP、2 条边、1 条派生约束；修好之后的第二批（见 5.4）得到 1 个 EP、4 条边、0 条派生约束。
- 相同 `(payload_hash, timestamp, scope)` 的事件去重（`dedup: True`，不新建锚点）。

### 5.3 `search` / `get_all` / `axis_query` — filters 语法与 Mem0 相同

```python
for text, at, run in [("Sarah joined the billing team in March.", "2026-03-02T09:00:00Z", "run-a"),
                      ("Sarah shipped the webhook retry design on May 6.", "2026-05-06T09:00:00Z", "run-b"),
                      ("Sarah is reviewing the checkout refactor this week.", "2026-06-11T09:00:00Z", "run-c")]:
    hms.add(text, timestamp=at, user_id=U, run_id=run)
hit = hms.search("what happened in May", filters={"user_id": U}, explain=True)["results"][0]      # 时间轴硬门控
print(hit["memory"], "|", hit["score_details"]["route"], hit["score_details"]["primary_axis"], hit["score_details"]["axis_gates"]["time"])
print([r["memory"] for r in hms.get_all(filters={"user_id": U, "entity": "Sarah"})["results"]])  # 实体轴，跨 run
hit = hms.search("what coffee does the user like", filters={"user_id": U}, explain=True)["results"][0]
print(hit["memory"], "|", hit["score_details"]["route"], hit["score_details"]["axis_gates"]["obs_type"], hit["score_details"]["signals"])
print(sorted(hit["score_details"]))
print([r["memory"] for r in hms.get_all(filters={"user_id": U, "OR": [{"obs_type": "constraint"}, {"scale": "L1_micro"}]})["results"]])
print(len(hms.get_all(filters={"user_id": U, "created_at": {"gte": "2026-01-01"}, "obs_type": {"in": ["fact", "temporal_fact"]}})["results"]))
```

```text
Sarah shipped the webhook retry design on May 6. | temporal time ['2026-05-01T00:00:00.000000+00:00', '2026-05-31T23:59:59.999999+00:00']
['Sarah is reviewing the checkout refactor this week.', 'Sarah shipped the webhook retry design on May 6.', 'Sarah joined the billing team in March.']
User likes black coffee. | preference ['preference_event', 'habit'] ['semantic', 'keyword', 'recency', 'support']
['axis_gates', 'bm25_score', 'channel_sizes', 'entity_boost', 'final_score', 'keyword_backend', 'max_possible_score', 'pool_size', 'primary_axis', 'raw_score', 'recalled_by', 'recency_score', 'route', 'semantic_score', 'signals', 'support_score', 'threshold']
["Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", '[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression', 'Do not modify the subscription state machine.']
3
```

filters 语法（与 Mem0 逐字相同，至少一个 scope 键）：

| 写法 | 含义 |
|---|---|
| `{"user_id": "alice"}` / `agent_id` / `run_id` | scope（必填 ≥ 1 个；顶层传给 `search(user_id=...)` 会抛 `VALIDATION_008`，是 `ValueError` 子类） |
| `"key": v` | 等于 |
| `"key": [v1, v2]` | `in` |
| `"key": "*"` | 存在 |
| `"key": {"eq"\|"ne"\|"gt"\|"gte"\|"lt"\|"lte"\|"in"\|"nin"\|"contains"\|"icontains": v}` | 运算符（未知运算符 → `VALIDATION_012`） |
| `"AND"/"OR"/"NOT": [ {...}, ... ]` | 逻辑组合，可嵌套 |
| `"visibility": "team"` | 可见性**格**门控：返回 `team \| event \| public`（`private < team < event < public`） |

键名解析顺序：轴列（`time / time_anchor / created_at / updated_at / valid_from / valid_to / atom_type / scale / scale_level / role / sentiment / confidence / memory_type / attributed_to / actor_id / visibility / subject / predicate / obs_type / memory_id / atom_id / hash / source_kind / extracted_by`）→ 集合轴（`entity / entities`、`topic / topics`、`anchor_id`）→ 其余任何键都落到调用方 `metadata`（`json_extract(user_meta_json)`；显式写 `meta.<key>` 也可以）。时间字面量会经 `to_utc_iso` 归一化，`lte` 给日期只写到天则取当天结束。

- `search(query, *, top_k=20, filters, threshold=0.1, rerank=False, explain=False, reference_date=None, show_expired=False)`：`reference_date` 就是 `as_of`；`rerank=True` 在融合后用 `self.reranker` 重排（未配置时为默认离线 overlap 重排器，一条 info 日志），各行追加 `rerank_score`，重排器出错时保持检索顺序；`threshold` 作用在**融合后**的分数上（所以标识符命中不会被弱语义分挡掉）；`explain=True` 时 `score_details` 是 Mem0 的 7 个键 + `recency_score, support_score, keyword_backend, axis_gates, primary_axis, route, signals` + `recalled_by, pool_size, channel_sizes`（该行来自哪些召回通道——`['gate']` 表示整个门控集合都打了分，否则是 `semantic / keyword / entity / recency` 的子集——、池规模、每个通道召回的 id 数）。
- `get_all(*, filters, top_k=20, show_expired=False)`：无 `score`，顺序 `time_anchor DESC, created_at DESC, atom_id ASC`；`show_expired=True` 把归档 / 过期版本一起返回（带 `valid_to` 与 `hms.archived_reason`）。
- `axis_query()` 是 HMS §3.5 七轴查询的原生入口——投影 + 聚合在服务端执行：

```python
ax = hms.axis_query({"filters": {"user_id": U, "topic": "coffee"}, "project_axis": "value",
                     "scale_constraint": ["state"], "aggregation": "latest"})
print(ax.projection, len(ax.hits), hms.axis_query({"filters": {"user_id": U, "topic": "coffee"}, "aggregation": "count"}).count)
```

```text
[['black coffee']] 1 1
```

`AxisQuery(filters, project_axis ∈ {atom, value, time, entity, topic}, scale_constraint, aggregation ∈ {list, latest, count, summary, topk}, limit, as_of, include_expired, text, meta_filters)` → `AxisResult(hits, projection, count, summary, primary_axis)`；`topk` 需要 `text`（否则 `VALIDATION_013`）。

### 5.4 `compile` — Work State Card、编译模式与 PC-IR

先把修复批次也写进去（第二条 EP、`temporal + causal` 边）：

```python
events_fix = [
    {"object_type": "file_diff", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:10:00Z",
     "payload": {"path": "src/billing/idempotency.ts", "patch": "+ idempotencyKey",
                 "summary": "add idempotency key", "intent": "fix checkout webhook retry"}},
    {"object_type": "test_log", "role": "tool", "actor_id": "coder", "timestamp": "2026-06-15T10:14:00Z",
     "payload": {"suite": "billing_e2e", "passed": 14, "failed": 0, "failures": []}},
]
rb = hms.add_events(events_fix, **scope)
print(len(rb["eps"]), len(rb["edges"]), rb["constraints"])
ir = hms.compile("fix checkout webhook retry", filters=scope, budget_tokens=800, as_of="2026-06-15T12:00:00Z")
print(ir.to_text())
```

```text
1 4 []
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

（修复前的卡片见第 2 节：Valid Facts 为空，Next Action 是 `Do not re-apply … ; investigate …`。）这两张卡就是测试套件的黄金文件 `tests/fixtures/golden/work_state_card_{failed,fixed}.txt`，逐字节比对。

```python
print([c.text for c in ir.constraints], [e.text[:40] for e in ir.episodes], ir.facts, ir.next_action[:30])
print({k: ir.retrieval[k] for k in ("ablation", "graph_hops", "candidates", "dropped_covered")}, ir.used_tokens, "<=", ir.budget_tokens)
for mode in ("fast", "axis_graph", "deep"):
    r = hms.compile("fix checkout webhook retry", filters=scope, as_of="2026-06-15T12:00:00Z", mode=mode).retrieval
    print(mode, "->", r["ablation"], r["graph_hops"])
pc = ir.to_pcir()
print(pc.ok(), [(o.name, o.status) for o in pc.proof_obligations], sorted(pc.payload))
```

```text
['Do not modify the subscription state machine.', "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression"] ['[negative] fix checkout webhook retry → ', '[positive] fix checkout webhook retry → '] [] Continue from 'patch src/billi
{'ablation': 'axis_graph_support', 'graph_hops': 1, 'candidates': 8, 'dropped_covered': 4} 288 <= 800
fast -> axis 0
axis_graph -> axis_graph 1
deep -> axis_graph_support 1
True [('evidence_coverage', 'satisfied'), ('no_unresolved_conflict', 'satisfied'), ('governance', 'skipped'), ('not_revoked', 'satisfied')] ['as_of', 'constraints', 'episodes', 'facts', 'goal', 'invalidated', 'next_action', 'preferences', 'scope', 'skills']
```

- `compile(query, *, filters, budget_tokens=None, mode=None, as_of=None, require_witness=None) -> ContextIR`，**只读**。`ContextIR` 字段：`query, goal, as_of, scope, mode, route, facts, episodes, preferences, constraints, skills, audit, next_action, budget_tokens, used_tokens, retrieval, unresolved_conflicts`；每个槽位项都是带 `anchor_ids` 的 `ContextItem`，六个槽位互斥。
- 模式（HMS 消融配置的对应物）：`fast` = 仅轴（不碰任何 `graph_*` 表）；`axis_graph` = 轴 + 图一跳扩展（`causal / revision / conflict / use / distill`，修订前驱进 `audit(reason='revised')`）；`deep` = 轴 + 图 + Support 层（`refine` InterfaceEP → 不变量进 constraints、残差进 audit）；`auto`（默认）在 scope 里有 EP 时解析为 `deep`，否则不扩展。`retrieval['ablation'] ∈ {axis, axis_graph, axis_graph_support}`。
- 槽位规则（有序）：归档或在 `as_of` 已过期 → `audit`；`skill` 原子 → `skills`；`constraint` → `constraints`；`preference_event / habit` → `preferences`；episode → `episodes`；其余 → `facts`；被已选 episode 覆盖的 event 原子被丢弃（`retrieval['dropped_covered']`）；`entity` 原子永不进卡。预算按 `slot_shares` 分配，`min_constraints` 兜底，facts 剩余预算溢出给 episodes。
- `filters={"visibility": "team"}` 时该值同时是 audience：轴过滤、图邻居、不变量、证据锚点都会按可见性格门控（`retrieval['dropped_governance']`），`governance` 义务才会真正被检查；无 audience 时为 `skipped`。
- `to_pcir(require_witness=True)` 返回 `PCIR(payload, witness, human_gloss, proof_obligations)`；任一义务 `violated` 时抛 `ProofObligationError`（`require_witness=False` 则只让 `ok()` 返回 False）。

### 5.5 `why` — 侦探模式：反向 BFS + 确定性证据链

```python
trace = hms.why("why must retry.ts not be modified", filters=scope)      # 不是 id → 当作问题：compile(deep) 后取最相关的约束
print(trace["target"])
print("\n".join(trace["chain"]))
print(sorted(trace), "|", len(trace["nodes"]), "nodes", len(trace["edges"]), "edges", len(trace["anchors"]), "anchors")
print(sorted((e["edge_type"], len(e["witness_anchor_ids"])) for e in trace["edges"]), trace["unresolved_conflicts"], trace["refined_from"])
print(hms.why(ra["eps"][0])["target"]["resolved_from"])                 # 也接受 memory_id / atom_id / ep_id / obs_id / iep_id / edge_id
```

```text
{'kind': 'atom', 'id': 'atm_…', 'node_id': 'atm_…', 'text': "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'resolved_from': 'query'}
Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression
<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression
-use[file://src/billing/retry.ts]-> Do not modify the subscription state machine.
['anchors', 'chain', 'edges', 'explanation', 'interface_eps', 'nodes', 'paths', 'refined_from', 'target', 'unresolved_conflicts'] | 4 nodes 5 edges 5 anchors
[('causal', 2), ('distill', 1), ('use', 1), ('use', 1), ('use', 1)] [] []
id
```

`why(target_id, *, hops=2, filters=None, edge_types=None, include_invalid=False)`：沿入边 `{causal, use, distill, revision, conflict}` + 出边 `use` 做 `hops` 层反向 BFS，途中的 InterfaceEP 节点按需 `refine`；`chain` 是确定性的（每步取置信度最高的入边），`explanation == "\n".join(chain)`；`chain` 与黄金文件 `why_chain.txt` 逐字节相同。无 `filters` 且不是已知 id → `NOT_FOUND_001`（`ValueError`）。

### 5.6 `feedback` / `revoke` — 十一种类型化反馈；只修订，不删除

```python
constraint_id = hms.get_all(filters={**scope, "obs_type": "constraint"})["results"][-1]["id"]   # "Do not modify …"
cp = hms.checkpoint(filters=scope, label="before-feedback")                                     # 见 5.7
fb = hms.feedback(constraint_id, "state_outdated", correction="retry.ts may change once billing_e2e is isolated.")
print(fb["message"], [a["action"] for a in fb["actions"]])
print(hms.get(constraint_id)["memory"], "| version", hms.get(constraint_id)["version"])
mid = hms.compile("fix checkout webhook retry", filters=scope)      # 不传 as_of：后继版本从反馈时刻起才有效
print([c.text for c in mid.constraints])
print([(a.text, a.explain["reason"]) for a in mid.audit if a.explain.get("reason") == "revised"])
```

```text
Feedback applied ['archive', 'create', 'revision_edge']
retry.ts may change once billing_e2e is isolated. | version 2
["Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': billing_e2e failed: subscription_state_regression", 'retry.ts may change once billing_e2e is isolated.']
[('Do not modify the subscription state machine.', 'revised')]
```

`feedback(target_id, feedback, *, reason=None, correction=None, actor_id=None) -> {'message': 'Feedback applied', 'feedback_id', 'actions': [...]}`。`target_id` 可以是 `memory_id / atom_id / edge_id / interface_ep_id / 上下文项 id`；每次反馈本身先写成一个 `feedback` SourceEvent + Anchor，它见证随后创建的每一行，并给目标谱系写一条 `FEEDBACK` history。

| `FeedbackType` | 目标 | 动作（按顺序） |
|---|---|---|
| `state_wrong` | atom | `archive` 当前版本；有 `correction` 文本则 `create` 后继版本 + `revision_edge`；解决触及的冲突边；使其 `use/distill` 出边失效 |
| `state_outdated` | atom | `archive`（`valid_to` = `correction['valid_to']` 或现在）；文本 `correction` → `create` + `revision_edge`（后继 `valid_from` = 反馈时刻） |
| `state_missing` | atom / episode / **无目标** | 从 `correction`（str 或 `{'text', 'obs_type'?, 'scope'?}`）`create` 一条新谱系；`target_id=None` 时 scope 取自 `correction['scope']`，或把 `target_id` 写成 `"user_id=alice&run_id=run-a"` |
| `privacy_violation` | atom | **唯一会改可见性的类型**：目标及共享其锚点的原子 → `set_visibility(private)`，锚点权限收紧 |
| `next_action_wrong` | episode / 上下文项 | `create` 一条 `Do not: <a>`（或 `correction`）约束 + `set_confidence(-0.2)` + `distill` 边 |
| `relationship_not_useful` | edge | `invalidate_edge`；由它投影出的高层边只在**不再剩任何有效投影来源**时才一并失效（存在规则，见 §9） |
| `connection_reason_wrong` | edge | `correction={'edge_type', 'label'}`：同类型只重标注 + 加见证（`insert_edge … merged: True`）；不同类型则失效旧边、`insert_edge` 新边 |
| `evidence_wrong` | atom / anchor | 争议指定锚点（`dispute_anchor`），若无未争议锚点剩余则 `archive`；`set_confidence(-0.3)`；`correction={'revoke': True}` + anchor_id 目标 → 委托给 `revoke()` |
| `render_not_like_me` / `render_like_me` | atom / 上下文项 | `set_confidence(∓0.1)` + `create` 一条 `render:<id>` 的偏好原子 |
| `object_important` | atom | `set_confidence(+0.2)` + `pin`（话题 `important`，永不被 prune） |

未知类型 → `VALIDATION_006`；`state_missing` 无 `correction` → `VALIDATION_007`。

`revoke(anchor_id, *, reason=None, actor_id=None)` 是 HMS "撤销时能找到派生对象" 的路径：设置 `anchors.revoked_at`，级联争议每个派生原子（没有未争议见证者的原子被归档）、使所有见证者全部被撤销的边（`invalidated_reason='witness_revoked'`）与 InterfaceEP 失效，再沿 L1 → L2 → L3 使**不再剩任何有效投影来源**的投影边失效（`'projection_revoked'`，即使它自己的见证还在）；还剩任一未撤销见证、或任一有效来源的对象保持有效——主见证 `anchor_ids[0]` 并不特殊（存在语义，§9 与 `docs/design/WITNESS_SEMANTICS.md`）；证据行永不删除。返回 `{'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'}`（重复撤销 → `'noop': True`）。

### 5.7 `checkpoint` / `rollback` — 全量快照，对称回滚

```python
rbk = hms.rollback(cp)
print(rbk["message"], "| restored", len(rbk["restored"]), "archived", len(rbk["archived"]),
      "edges_invalidated", len(rbk["edges_invalidated"]), "edges_created", len(rbk["edges_created"]))
print(hms.get(constraint_id)["memory"], "| version", hms.get(constraint_id)["version"])
print([h["event"] for h in hms.history(constraint_id)])
```

```text
Rolled back to before-feedback | restored 1 archived 1 edges_invalidated 1 edges_created 1
Do not modify the subscription state machine. | version 1
['ADD', 'FEEDBACK', 'UPDATE', 'ROLLBACK', 'RESTORE']
```

- `checkpoint(*, filters, label=None) -> "ckp_…"`：一次事务里记录 scope 内每个原子的 `{valid_to, archived_reason}`、每条边 / InterfaceEP 的 `valid`、每个 EP 的 `valid_to`，外加 history 序号与快照哈希；自身是一个 `checkpoint` 事件。
- `rollback(checkpoint_id, *, reason=None)`：固定顺序（归档 → 恢复 → EP 有效性 → 边 → InterfaceEP → 回滚修订边 + 转移），返回 `{'message': 'Rolled back to <label>', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}`。原子内容永不改写，向量永不删除；回滚本身留下一条 `revision` 边（`label='rollback:<ckp>'`）与一条 `rollback` 转移。

### 5.8 `handoff` / `import_package` — 可交接的 StatePackage

```python
pkg = hms.handoff(filters=scope, goal="fix checkout webhook retry", checkpoint=False)
print(sorted(type(pkg).model_fields))
print({"atoms": len(pkg.atoms), "eps": len(pkg.eps), "edges": len(pkg.edges), "anchors": len(pkg.anchors),
       "source_events": len(pkg.source_events), "goal": pkg.goal, "pcir_ok": pkg.pcir.ok()})
receiver = HMS({"store": {"path": ":memory:"}})
rep = receiver.import_package(pkg.to_json())                      # 也接受 StatePackage 实例或 dict
print(rep["imported"], rep["skipped"]["atoms"])
card_a = hms.compile("fix checkout webhook retry", filters=scope, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()
card_b = receiver.compile("fix checkout webhook retry", filters=scope, budget_tokens=800, as_of="2026-06-15T12:00:00Z").to_text()
print(card_a == card_b, receiver.import_package(pkg)["imported"]["atoms"])
receiver.close()
```

```text
['anchors', 'atoms', 'checkpoint_id', 'context', 'created_at', 'edges', 'eps', 'goal', 'interface_eps', 'package_id', 'pcir', 'scope', 'source_events', 'version']
{'atoms': 5, 'eps': 2, 'edges': 7, 'anchors': 7, 'source_events': 7, 'goal': 'fix checkout webhook retry', 'pcir_ok': True}
{'source_events': 7, 'anchors': 7, 'eps': 2, 'atoms': 5, 'interface_eps': 0, 'edges': 7, 'transitions': 2} 0
True 0
```

- `handoff(*, filters, goal=None, budget_tokens=None, checkpoint=True, include_edges=True) -> StatePackage`：`goal=None` 时取最新有效 EP 的 `u`；`checkpoint=True`（默认）先打一个 `handoff:<goal>` 检查点并把 id 嵌进包里；包内是 `compile(mode='deep')` 的 `ContextIR` + PC-IR + 见证闭包（原子、EP、InterfaceEP、边、锚点、源事件），`to_json()` / `StatePackage.from_json()` 往返相等。
- `import_package(pkg, *, user_id=None, agent_id=None, run_id=None)`：接收端**导入状态而不是回放历史**（HMS §9.4）——id 保持不变，已存在的对象按身份去重进 `skipped`，给出任一 scope 参数则整包重定 scope；导入后接收端编译出同一张卡片。

### 5.9 `transitions` — `(S_t, a_t, S_{t+1}, r_t, κ_t, w_t)` 训练元组

```python
for s_t, a_t, s_t1, r_t, kappa_t, w_t in hms.transitions(filters=scope):          # 默认 fmt="tuples"
    print(f"{a_t[:48]:<48} -> {r_t:<8} | kappa: {sorted(kappa_t)} | w_t: {len(w_t)}")
rec = hms.transitions(filters=scope, fmt="records")[0]
print(sorted(rec), "|", rec["source_kind"], rec["scale"], rec["time_t"])
print(rec["s_t"]["u"], "|", rec["s_t1"], "|", rec["kappa_t"]["constraints"])
print(len(hms.transitions(filters=scope, fmt="jsonl").splitlines()), "jsonl lines;",
      [r["source_kind"] for r in hms.transitions(filters=scope, fmt="records", kinds=["revision", "rollback"])])
```

```text
patch src/billing/retry.ts: add exponential back -> negative | kappa: ['constraints', 'permission_scope', 'provenance', 'visibility'] | w_t: 2
patch src/billing/idempotency.ts: add idempotenc -> positive | kappa: ['constraints', 'permission_scope', 'provenance', 'visibility'] | w_t: 2
patch src/billing/idempotency.ts: add idempotenc -> positive | kappa: ['constraints', 'permission_scope', 'provenance', 'visibility'] | w_t: 3
revise                                           -> negative | kappa: ['constraints', 'permission_scope', 'provenance', 'visibility'] | w_t: 2
rollback                                         -> neutral  | kappa: ['constraints', 'permission_scope', 'provenance', 'visibility'] | w_t: 1
['a_t', 'created_at', 'kappa_t', 'r_t', 's_t', 's_t1', 'scale', 'scope', 'source_id', 'source_kind', 'time_t', 'time_t1', 'transition_id', 'w_t'] | ep L1_micro 2026-06-15T10:04:00.000000+00:00
fix checkout webhook retry | {'ep_id': 'ep_…', 'r': 'negative', 'y': 'billing_e2e failed: subscription_state_regression'} | ['Do not modify the subscription state machine.']
5 jsonl lines; ['revision', 'rollback']
```

- `transitions(*, filters, since=None, scale=None, fmt="tuples"|"records"|"jsonl", kinds=None)`。EP 行在摄入时就写好（`source_kind='ep'`）；导出时补齐 `interface_ep`（consolidate 之后）、`causal`（有效因果边：`s_t` = 前一 EP，`a_t` = 后一 EP 的动作）、`revision` / `conflict`（`a_t='revise'|'conflict'`）与 `rollback` 行；幂等（重复导出不加行）。
- `κ_t` 对**每一种**行都是同一形状 `{constraints, visibility, permission_scope, provenance}`；`w_t` 是**有序并集**（首次出现顺序，不排序），例如因果行的 `w_t` = 边见证者 ∪ 后一 EP 的锚点。这就是 HMS §4.4 的 world-model transition 数据。

### 5.10 `consolidate` / `graph_stats` — 同步的 StateScheduler

```python
rep = hms.consolidate(filters=scope)                     # phases 默认 [orient, collect, consolidate, prune]
print(rep["phases"]["consolidate"])
print(rep["phases"]["prune"])
st = hms.graph_stats(filters=scope)
print(st)
print(hms.graph_stats(filters={"user_id": U, "agent_id": "coder"})["nodes"]["entity"])
ir = hms.compile("fix checkout webhook retry", filters=scope, mode="deep")
print([s.text for s in ir.skills], [a.text for a in ir.audit if a.explain.get("reason") == "invariant:failed_path"])
print([(r["source_kind"], r["scale"], r["r_t"]) for r in hms.transitions(filters=scope, fmt="records", kinds=["interface_ep"])])
```

```text
{'entity_atoms': 6, 'temporal_edges': 0, 'causal_edges': 0, 'conflicts': 0, 'interface_eps': 1, 'projected_edges': 3, 'interface_eps_l3': 0, 'skills': 1, 'transitions': 1, 'compiled_states': 1}
{'examined': 4, 'decayed': 0, 'archived': [], 'live': 10, 'expired': 1, 'skipped_conflict': 0, 'purged': 0}
{'nodes': {'message': 0, 'observation': 3, 'event': 4, 'entity': 0, 'ep': 2, 'interface_ep': 1, 'skill': 1}, 'edges': {'temporal': 1, 'causal': 1, 'revision': 2, 'conflict': 0, 'use': 5, 'distill': 3}, 'valid_edges': 11, 'invalid_edges': 1, 'boundary_edges': 3, 'interface_eps': 1, 'unresolved_conflicts': 0, 'witness_coverage': 1.0, 'projection_coverage': 1.0, 'traceable_high_edges': 1.0, 'transitions': 6}
6
['Skill: Do not modify the subscription state machine.'] ['do not: patch src/billing/retry.ts: add exponential backoff']
[('interface_ep', 'L2_meso', 'positive')]
```

四个阶段（`consolidate(*, filters=None, phases=None)`；`filters=None` 或部分 scope 时对每个匹配的 `(user_id, agent_id, run_id)` 分别运行）：

| phase | 做什么 | 报告键 |
|---|---|---|
| `orient`（只读） | 活跃话题 / 实体、未解决的负向 EP、待连边的 EP 节点、coarsen 候选组、上次编译 | `active_topics, active_entities, open_failures, pending_ep_nodes, coarsen_candidates, last_compiled` |
| `collect` | 补齐缺失向量（换 embedder 之后用）、重新路由没有原子的可抽取事件（Txn B 崩溃恢复） | `re_embedded, re_routed` |
| `consolidate` | (0) 每个实体一条 `entity` 原子（scope 为实体身份 scope，不含 `run_id`）；(1) `temporal` 边；(2) `causal` 提议（`rule` 或 LLM `edge_proposal`）；(3) 互斥谓词冲突边（`deferred` 修订模式在此结算）；(4) `coarsen → project` 成 L2 InterfaceEP；(4b) L2 组 → L3；(5) 有 ≥2 支持节点的 `constraint` 不变量提升为 `L3_macro` 技能原子 + 可追溯的 `distill` 边；(6) 导出转移；(7) 每 scope 持久化一张 `deep` 卡片到 `compiled_states` | `entity_atoms, temporal_edges, causal_edges, conflicts, interface_eps, projected_edges, interface_eps_l3, skills, transitions, compiled_states` |
| `prune` | 陈旧度 `(1−recency)(1−feedback_boost)(1−use_recency) > staleness_tau` 的原子先**衰减**置信度，再把 `confidence < confidence_min` 的**归档**（从不动 pinned / constraint / skill / entity / 有未解决冲突的原子）；仅当 `retention.enabled` 且 `retention.run_in_prune` 时再调用 5.12 的 `purge(dry_run=False)` | `examined, decayed, archived, live, expired, skipped_conflict, purged` |

`graph_stats(filters=None)` 统计 scope 内的节点 / 边（不分有效性）、`valid_edges / invalid_edges / boundary_edges / interface_eps / unresolved_conflicts`，以及三个覆盖率：`witness_coverage`（有见证者的边占比）、`projection_coverage`（边界边有投影行的占比）、`traceable_high_edges`（高层边可追溯到低层边的占比）。上面 `nodes.entity == 0` 是因为实体原子挂在 `(alice, coder)` 而 `scope` 过滤了 `run_id`。

### 5.11 `get` / `update` / `delete` / `delete_all` / `history` — Mem0 语义，谱系实现

```python
mem = r1["results"][0]["id"]
print(hms.update(mem, text="User likes black coffee with no sugar."), hms.get(mem)["version"])
print(hms.update(mem, metadata={"source": "chat"}), hms.get(mem)["metadata"], hms.get(mem)["version"])
print(hms.update(mem, expiration_date="2026-12-31"), hms.get(mem)["valid_to"])
print(hms.delete(mem), hms.get(mem))
print([(h["event"], h["old_memory"], h["new_memory"], h["is_deleted"]) for h in hms.history(mem)])
print(sorted(hms.history(mem)[0]))
print([r["hms"]["archived_reason"] for r in hms.get_all(filters=s1, show_expired=True)["results"]])
print(hms.delete_all(user_id=U, run_id="run-coffee"))
for call in (lambda: hms.update(mem, text="x"), lambda: hms.delete_all(), lambda: hms.search("x", user_id=U)):
    try:
        call()
    except ValueError as e:
        print(type(e).__name__, getattr(e, "error_code", None), "|", str(e)[:70])
hms.close()
```

```text
{'message': 'Memory updated successfully!'} 3
{'message': 'Memory updated successfully!'} {'source': 'chat'} 3
{'message': 'Memory updated successfully!'} 2027-01-01T00:00:00.000000+00:00
{'message': 'Memory deleted successfully!'} None
[('ADD', None, 'User likes oat-milk lattes in the morning.', False), ('UPDATE', 'User likes oat-milk lattes in the morning.', 'User likes black coffee.', False), ('UPDATE', 'User likes black coffee.', 'User likes black coffee with no sugar.', False), ('UPDATE', 'User likes black coffee with no sugar.', 'User likes black coffee with no sugar.', False), ('DELETE', 'User likes black coffee with no sugar.', None, True)]
['actor_id', 'anchor_id', 'created_at', 'event', 'id', 'is_deleted', 'memory_id', 'new_memory', 'old_memory', 'role', 'updated_at']
['deleted', 'revised', 'revised']
{'message': 'Memories deleted successfully!'}
ValidationError NOT_FOUND_001 | Memory with id mem_… not found
ValidationError VALIDATION_001 | At least one filter is required to delete all memories. If you want to
ValidationError VALIDATION_008 | Top-level user_id is not supported in search(); pass it inside filters
```

- `update(memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None)`：`text` → **修订**（同 `id`，`version+1`，`revision` 边，history `UPDATE`）；只给 `metadata` → 原地写 `user_meta`，不出新版本（Mem0：metadata 不是状态），history 记一条 `old == new` 的 `UPDATE`；`expiration_date` 只改 `valid_to`（`None` 清空，日期 → 次日零点），不写 history；`data=` 是 `text` 的废弃别名。三者都不给 → `VALIDATION_011`。
- `delete(memory_id)` = `archived_reason='deleted'` + `DELETE` history（`is_deleted=True`），边保留；`get()` 返回 `None`，行仍可用 `show_expired=True` 看到。`delete_all()` 不带任何 id 时的报错文案与 Mem0 逐字相同。
- `history(memory_id)` 返回裸 list，Mem0 的 10 个键 + **恰好一个**新增键 `anchor_id`；`event` 词汇在 `ADD / UPDATE / DELETE` 之外还有 `ARCHIVE / RESTORE / FEEDBACK / CONFLICT / ROLLBACK / PURGE`（最后一种只由 5.12 的逃生舱写入）。
- 其他 Mem0 形状：`get()` 找不到返回 `None`（不抛）；`reset()` 清空存储与索引；`close()` 幂等；`chat()` 抛 `NotImplementedError`；`project.update()` 抛 `ValueError`；所有 `VALIDATION_*` / `NOT_FOUND_001` 异常都是 `ValueError` 子类，带 `error_code / details / suggestion`。

### 5.12 `purge` / `vacuum` — 保留策略逃生舱：显式、默认关闭、有审计

"修订代替删除"仍是唯一的默认路径：`delete / feedback / rollback / prune` 只写 `archived_reason`，`BEFORE DELETE` 触发器照常拦截一切物理删除，库随写入单调增长。`purge()` 是给运维准备的**逃生舱**，不是行为默认值：它只清理**状态行**（`memory_atoms` 与其 `atom_anchor / atom_entity / atom_topic / atom_vectors / graph_nodes` 从属行，以及该原子自己的 `observations / eps` 来源行），锚点、`source_events`、`graph_edges`、`history` 永远不清；每次清理本身是一条 `system(op='purge')` 事件 + 锚点，每个被清版本在 `history()` 里留下一条 `PURGE` 行（记录 `text` 的 sha256、版本、`archived_reason`、`valid_to`，不记录文本）。

```python
from hms_elite import HMS

hms = HMS({"store": {"path": ":memory:"}, "retention": {"enabled": True}})   # 默认 enabled=False：purge(dry_run=False) 直接抛 ConfigError
mem = hms.add("I like oat-milk lattes in the morning.", user_id=U, run_id="run-coffee")["results"][0]["id"]
keep = hms.add("I moved to Berlin last spring.", user_id=U, run_id="run-coffee")["results"][0]["id"]
print(hms.delete(mem), hms.get_all(filters={"user_id": U}, show_expired=True)["results"][-1]["hms"]["archived_reason"])
plan = hms.purge(filters={"user_id": U}, older_than_days=0)          # 默认 dry_run=True：只报告，不写任何行
print(plan["dry_run"], plan["examined"], plan["purged"], plan["anchor_id"], sorted(plan["protected"]))
print([(p["memory_id"] == mem, p["version"], p["archived_reason"], len(p["text_sha256"]), p["bytes"] > 0) for p in plan["purgeable"]])
done = hms.purge(filters={"user_id": U}, older_than_days=0, dry_run=False)   # 真正清理：需要 retention.enabled=True
print(done["purged"], done["anchor_id"], done["anchors_purged"], hms.get(mem))
print([r["id"] == keep for r in hms.get_all(filters={"user_id": U}, show_expired=True)["results"]])
print([(h["event"], h["old_memory"], h["is_deleted"]) for h in hms.history(mem)])
print(sorted(hms.vacuum()), hms.purge(filters={"user_id": U}, older_than_days=0, dry_run=False)["purged"])
hms.close()
```

```text
{'message': 'Memory deleted successfully!'} deleted
True 1 0 None ['checkpoint_live', 'edge_endpoint', 'ep_referenced', 'head_version', 'interface_ep_support', 'source_shared']
[(True, 1, 'deleted', 64, True)]
1 anc_… [] None
[True]
[('ADD', None, False), ('DELETE', 'User likes oat-milk lattes in the morning.', True), ('PURGE', None, True)]
['after_bytes', 'before_bytes', 'freed_bytes', 'path'] 0
```

- **两把钥匙**：`config.retention.enabled=True`（默认 `False`）**且** `dry_run=False`（默认 `True`），缺一则要么只报告、要么抛 `ConfigError(CONFIG_001)`；`filters` 至少一个范围键（只给 `user_id` 会扫过该用户所有 run），否则 `VALIDATION_001`；`older_than_days` 为负 → `VALIDATION_009`。
- **可清理的行必须同时满足**：`archived_reason` 非空且 `valid_to < now − older_than_days`；不是任何边（有效或失效）的端点——`graph_edges` 携带见证、从不删除，因此被修订过的版本（有 `revision` 边）永远留下；不是任何 checkpoint 快照里"当时有效"的原子（回滚仍可能要恢复它）；不被任何 InterfaceEP 引用（自身原子、主支持节点或支持成员）；其 EP 行不被其它 EP / 幸存原子 / 转移引用；多版本谱系的最高版本不清（它是修订链的终端记录）；其来源行（observation / ep）仍被清理集合之外的原子引用时不清（`source_shared`——更新 / 反馈的后继与前驱共享来源）。被拦下的 id 按原因列在 `protected` 里，一个 id 可以同时出现在多个原因下。
- **证据的去留由 `keep_witnessed` 决定**：`True`（默认）一律保留锚点与 `source_events`；`False` 只额外删除被清行的、库中**再无任何对象引用**的锚点（14 个见证列 + 4 个 JSON id 列表全部计数为 0）及其源事件——被 `history` 引用的锚点因此实际上总会留下。
- **实现**：SQLite 无法按语句禁用触发器，7 个 `BEFORE DELETE` 触发器（`source_events / anchors / observations / eps / memory_atoms / atom_anchor / graph_nodes`）带 `WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = 'purge_in_progress' AND value = '1')`；只有 `StateStore.purge_rows()` 会在自己的事务内写入这行 meta，并在 COMMIT 前删掉（回滚时随事务消失；打开库时若发现残留标志会先清除）。其余任何 `DELETE` 仍然 `STORE_003`。模式版本因此升到 2：旧库在下次打开时自动迁移（替换这 7 个触发器、按新的 `event` 词表重建 `history` 表并原样复制每一行）。
- `vacuum()` 在无事务时执行 `VACUUM`（文件库先 checkpoint 并截断 WAL），返回 `{'path', 'before_bytes', 'after_bytes', 'freed_bytes'}`；`purge` 之后调用它文件才真正缩小。`consolidate()` 的 `prune` 阶段只在 `retention.enabled` **且** `retention.run_in_prune` 都为 `True` 时调用 `purge(dry_run=False, keep_witnessed=True)`，报告键 `purged` 默认为 0。

### 5.13 `context` — 混合读上下文：记忆化会话卡片前缀 + 带时间戳的相关记忆行，放在问题之前

读侧的默认推荐（2026-09-18，`docs/design/READER_AB_2026-09-18.md`，CONTRACT §6.7）：`hms.context(query, filters=...)` 返回 `"<会话卡片>\n\nRecent relevant memories:\n<记忆行>"`——会话卡片是 `compile(<范围内最新一条消息的文本>, as_of=<该消息的时间戳>, budget_tokens=1200)` 的卡片，按范围记忆化，任何触及该范围的写入（`add / update / delete / delete_all / rollback / consolidate / reset`……）之后才重算（`session_reused` 说明本次是否复用）；记忆行是 `search(query, top_k=20)` 去重（大小写 / 空白不敏感的重复与真子串剔除）、按 `time_anchor` 升序（未知时间在最后）的 `- [YYYY-MM-DD HH:MM] 文本` 行；两部分为空时各自渲染为 `(none)`。把它放在用户问题**之前**（`"Memories:\n" + ctx["text"] + "\n\n" + question`）：EgoLife 上记忆在前没有准确率代价，且卡片复用期间这段前缀能命中 DeepSeek 的提示词缓存；同一存储上这一读法答对 70.1%（143/204），纯记忆行 54.9%、逐题卡片 63.2%。
The read-side default since 2026-09-18 (`docs/design/READER_AB_2026-09-18.md`, CONTRACT §6.7): `hms.context(query, filters=...)` returns `"<session card>\n\nRecent relevant memories:\n<rows>"` — the session card is `compile(<text of the scope's latest message>, as_of=<its timestamp>, budget_tokens=1200)`, memoized per scope and recomputed only after a write touching the scope (`add / update / delete / delete_all / rollback / consolidate / reset` …; `session_reused` says whether this call reused it); the rows are `search(query, top_k=20)` deduplicated (case / whitespace-insensitive duplicates and proper substrings dropped), sorted by `time_anchor` ascending (unknown last), rendered as `- [YYYY-MM-DD HH:MM] text` lines; an empty part renders as `(none)`. Put it BEFORE the question (`"Memories:\n" + ctx["text"] + "\n\n" + question`): memories-first cost no accuracy on EgoLife and the prefix hits DeepSeek's prompt cache while the card is reused; on the same store this read answers 70.1% (143/204) against 54.9% for plain rows and 63.2% for the per-question card.

```python
from hms_elite import HMS

hms = HMS({"store": {"path": ":memory:"}})
hms.add("I drink oat-milk lattes every morning.", user_id=U, timestamp="2026-04-02T08:00:00Z")
hms.add("I drink black coffee after lunch.", user_id=U, timestamp="2026-04-09T08:00:00Z")
ctx = hms.context("what do I drink?", filters={"user_id": U})
prompt = "Memories:\n" + ctx["text"] + "\n\n" + "What do I drink? Answer in one line."   # 记忆在前 / memories first
print(ctx["text"].split("\n")[0], ctx["session_reused"], ctx["session_key"] is not None, ctx["rows_header"])
print(ctx["rows_text"])
again = hms.context("coffee", filters={"user_id": U})                      # 无写入 → 会话卡片复用 / no write → the card is reused
print(again["session_reused"], again["session_card"] == ctx["session_card"], prompt.startswith("Memories:\n# Work State Card"))
hms.add("I gave up coffee entirely.", user_id=U, timestamp="2026-04-16T08:00:00Z")
print(hms.context("coffee", filters={"user_id": U})["session_reused"], hms.context("tea", filters={"user_id": "nobody"})["text"])
hms.close()
```

```text
# Work State Card False True Recent relevant memories:
- [2026-04-02 08:00] User drink oat-milk lattes every morning.
- [2026-04-09 08:00] User drink black coffee after lunch.
True True True
False (none)

Recent relevant memories:
(none)
```

---

## 6. HMS → Lite → Mem0 映射表

| HMS 概念（HMS_arch §） | HMS Elite 对象 / 调用 | Mem0 对应物 | 差异 |
|---|---|---|---|
| SourceEvent（§3.1） | `SourceEvent`、`source_events` 表、`add_events()` | `messages` 缓冲 + `save_messages` | 一等公民表，5 种可抽取 `object_type`（含 `sensor_trace`），`timestamp=` 被尊重 |
| Anchor（§3.2） | `Anchor`、`anchors` / `atom_anchor` / `edge_witnesses` | payload `hash`（md5）+ history 行 | sha256 事件哈希；每一行都引用 ≥1 个锚点；`revoke()` 级联 |
| Observation（§3.3） | `Observation` → `state \| event \| relation` 原子 | `ADDITIVE_EXTRACTION_PROMPT` 的 memory 文本 | 有类型、有日期、有见证；`linked_memory_ids` 变成 `supersedes_hint_ids` |
| EP（§3.4） | `EP`、`eps` 表、闭合构建器、`ep_extraction` 任务 | 过程记忆字符串 | 每个 action→outcome 闭合一个六元组；结构化事件不经过 LLM |
| MemoryAtom / AxisQuery（§3.5） | `MemoryAtom`、`memory_atoms` + 轴表、`AxisQuery`、`FilterExpr` | 扁平 payload + `_process_metadata_filters` | 有效期在 WHERE 里，实体 / 话题为多对多，谱系 id 保持 Mem0 `id` 稳定 |
| Support Graph（§3.6–3.7） | `graph_*` 表、`SupportGraph`、`InterfaceEP`、coarsen / project / refine | （v2 已移除；仅实体链接） | 六种类型化边、见证者、投影 |
| ContextIR / PC-IR（§3.8） | `compile()`、`ContextIR.to_text() / to_pcir()` | 结果格式化 | 六个互斥槽位、证明义务、`ProofObligationError` |
| StateScheduler（§3.9） | `consolidate(phases=)` | add 管线的 Phase 0–8 | 同步四阶段，由 scheduler 锚点见证；`compile_state` 落库 |
| StateFeedback（§3.10） | `feedback(target, type, reason, correction)` | `update() / delete()` | 11 个类型化处理器、修订边、`FEEDBACK` history |
| Transition（§4.4） | `transitions()` | `history()` 回放 | 来自 EP / InterfaceEP / 边 / 回滚的 `(S_t, a_t, S_{t+1}, r_t, κ_t, w_t)`，统一 κ_t |
| Handoff package（§9.4） | `handoff()` → `StatePackage`；接收端 `import_package()` | — | 接收端导入状态（事件、锚点、原子、EP、IEP、边），不是历史 |
| Revoke（§3.2） | `revoke(anchor_id)` | `delete()` | 级联到派生原子 / 边 / IEP；证据行永不删除 |
| add / get / get_all / search / update / delete / delete_all / history / reset / close | 同名、同签名、同信封 | Mem0 v2.0.19 | `add` 可能返回 `UPDATE / NONE`；行多出提升键；`history` 多 `anchor_id` |
| config dict | `HMSConfig` 接受 `vector_store / llm / embedder / history_db_path / version / custom_instructions / reranker` | `MemoryConfig` | 另加 `store / ingest / scoring / retrieval / graph / compiler / scheduler / retention`；改名 / 移除表 |

---

## 7. 从 Mem0 迁移 Migrating from Mem0

| 主题 | Mem0 v2 | HMS Elite | 你需要改什么 |
|---|---|---|---|
| filters | `{"user_id": ..., key: v \| [..] \| "*" \| {op: v}, "AND"/"OR"/"NOT": [...]}` | **完全相同**（见 5.3）；`visibility` 标量额外按格门控 | 不用改 |
| `add()` 返回 | 只有 `event: 'ADD'` | `ADD \| UPDATE \| NONE`（+ `revises` / `conflicts_with`）；行多出 `atom_id, version, …, hms{}` | 不要假设 `event == 'ADD'` |
| id | uuid4 | `mem_<12 hex>` 谱系 id；`atom_id` 是版本 id | 不要解析 id 格式 |
| `update(id, text=)` | 原地改写，`updated_at` 变 | **修订**：同 `id`、`version+1`、旧版本归档 + `revision` 边；`get(id)` 返回新文本 | 不用改；想看旧版本用 `get_all(show_expired=True)` |
| `update(id, metadata=)` | 原地 | 原地（不出新版本） | 不用改 |
| `history()` | 10 个键 | 10 个键 **+ `anchor_id`**；额外事件 `ARCHIVE / RESTORE / FEEDBACK / CONFLICT / ROLLBACK` | 对 `event` 做白名单匹配的代码要加分支 |
| `delete()` | 物理删除 | 归档（`archived_reason='deleted'`），`get()` → `None` | 不用改 |
| `timestamp=` | 存进 payload | 事件时间：`time_anchor` / `valid_from`，参与去重身份 | 同一文本在更晚的 `timestamp` 再 add 会得到 `NONE` + 新锚点，而不是静默去重 |
| `threshold` | 作用在向量分上 | 作用在融合分上 | 阈值含义略有变化（标识符命中不再被弱语义分挡掉） |
| config：接受的键 | `vector_store / llm / embedder / history_db_path / reranker / version / custom_instructions` | 同名同嵌套 | 不用改 |
| config：改名的键 | `custom_fact_extraction_prompt`、`custom_update_memory_prompt`、`index`、`db_path` | → `custom_instructions`、`custom_instructions`、`vector_store`、`store.path`（`DeprecationWarning`） | 可以晚点改 |
| config：移除的键 | `graph_store`、`enable_graph` | `ConfigError(CONFIG_003)` | 删掉这两个键（图永远开着，且在同一个 SQLite 文件里） |
| `vector_store.provider` | qdrant / faiss / chroma / … | **只有 `sqlite \| memory \| milvus`**（`milvus` 需 `[milvus]` extra；其余 → `CONFIG_002`） | 改成 `{"provider": "sqlite"}`（或装 `[milvus]` 后用 `{"provider": "milvus"}`） |
| `llm` / `embedder` provider | openai / anthropic / … | `rule \| scripted \| openai \| anthropic \| deepseek \| ollama`；`hash \| openai \| ollama \| fastembed` | 离线用 `rule` + `hash`；线上保留 `openai` / `anthropic`，或本地 `ollama` / `fastembed`（见第 8 节） |
| `reranker` | 生效（cohere / …） | 只有 `overlap \| llm`（其余 → `CONFIG_002`）；`null` 时 `search(rerank=True)` 用默认 overlap | 删掉或改成 `{"provider": "overlap"}` / `{"provider": "llm"}` |
| feedback | 平台 API：`POSITIVE / NEGATIVE / VERY_NEGATIVE` | 11 个 `FeedbackType`（5.6） | 映射：`NEGATIVE` → `state_wrong` / `evidence_wrong`，`POSITIVE` → `object_important` / `render_like_me` |
| `chat()` / `project` | 平台 | `NotImplementedError` / `ValueError` | — |
| `history.db` 导入 | — | 没有列集合差分迁移；用 `hms_elite.migrate.from_mem0` | 见下 |

用 `migrate.from_mem0` 把一个已有的 Mem0 `Memory` 实例的记忆搬过来（它只调用 `memory.get_all(filters=..., top_k=page_size)` 与 `show_expired=True`，再以 `infer=False, timestamp=created_at` 逐行 `hms.add`）：

```python
from hms_elite import HMS
from hms_elite.migrate import from_mem0

class FakeMem0:  # 任何有 get_all(filters=..., top_k=..., show_expired=...) 的对象都行，例如 mem0.Memory()
    def get_all(self, *, filters, top_k=100, show_expired=False, **kw):
        rows = [{"id": "old-1", "memory": "User likes tea.", "created_at": "2026-01-05T10:00:00Z",
                 "metadata": {"src": "mem0"}, "user_id": "carol"}]
        return {"results": [] if show_expired else rows}

hms = HMS({"store": {"path": ":memory:"}})
print(from_mem0(FakeMem0(), hms, filters={"user_id": "carol"}))
print([(r["memory"], r["valid_from"], r["metadata"], r["atom_type"]) for r in hms.get_all(filters={"user_id": "carol"})["results"]])
hms.close()
```

```text
{'migrated': 1, 'skipped': 0, 'ids': {'old-1': 'mem_…'}}
[('User likes tea.', '2026-01-05T10:00:00.000000+00:00', {'src': 'mem0'}, 'message')]
```

---

## 8. 离线确定性与切换 LLM Provider

默认配置（`rule` LLM + `hash` embedder + `sqlite` 索引）**不联网、不随机、不用墙上时钟**：库里唯一的时钟是 `store.now()`，id 由 `store.id_factory` 生成（测试注入冻结时钟与顺序 id，六个黄金场景逐字节复现，两种索引 provider 结果一致）；每条 `ORDER BY` 都以 id 收尾；`hash` embedder 是字符 2/3-gram → sha256 桶 → 64 维 L2 归一化向量；`rule` LLM 用同一套 prompt 契约（`HMS_TASK: <task>` + `## Section` 分段 + 固定 JSON 根键）实现七个分段任务（`observation_extraction / observation_extraction_compact / ep_extraction / revision_decision / edge_proposal / coarsen / procedural_summary`）与两个复合任务 `memory_extraction` / `memory_extraction_compact`（默认的 `Ingestor(extraction_mode='single')`：每批消息一次往返完成观察抽取、EP 抽取与修订决策，`tests/test_extraction_single.py` 断言与分段三件套结果相同；`_compact` 孪生即 `ingest.extraction_output='compact'`（可选）时同一套观察投影到九个紧凑键（含 `entities`），`tests/test_extraction_compact.py`）的确定性启发式；`rerank` 任务在 `rule` 下抛 `LLM_002`，重排器据此回退 `overlap`。这意味着：本 README 与 `examples/quickstart.py` 的每一段输出都可以在任何机器上复现。

切换到真实模型只改一个 config dict——签名、prompt、JSON 解析阶梯（Mem0 同款 `remove_code_blocks → extract_json → 顶层数组打捞`）全都不变：

```python
# （示意）需要 pip install "hms-elite[openai]" 与 OPENAI_API_KEY；anthropic 同理
from hms_elite import HMS

hms = HMS({
    "llm": {"provider": "openai", "config": {"model": "gpt-5-mini", "temperature": 0.1}},          # 或 anthropic / claude-sonnet-4-6
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", "embedding_dims": 1536}},
    "vector_store": {"provider": "sqlite"},
    "store": {"path": "./hms.db"},
    "graph": {"causal_proposer": "llm", "coarsen_policy": "llm"},   # 可选：因果提议与 coarsen 也交给 LLM（同一 prompt 契约）
})
```

- SDK 缺失时构造报 `DependencyError(DEPS_001)`，消息带 `pip install 'hms-elite[openai]'` 提示；config 本身（`HMSConfig.from_dict`）无需 SDK 即可校验。
- 换 embedder 之后跑 `hms.consolidate(phases=["collect"])` 重新嵌入缺失向量（向量按 `embedder.identity` 记账）。
- 混合使用：`llm=openai` + `embedder=hash` 或反之都合法；`scripted` provider 按队列回放响应（`responses=[dict | str | Exception | callable]`），是给测试用的。

---

## 9. 不变量 Invariants 及其强制方式

| 不变量 | 类型层（pydantic） | DDL 层（SQLite） | 存储层（`StateStore`） |
|---|---|---|---|
| **No Witness, No Meaning** —— 观察、EP、原子、节点、边、InterfaceEP、反馈、history、检查点、转移都至少有一个证据锚点。见证集是**存在条件**（OR，不是 AND）：只要还剩任一未争议、未撤销的锚点，对象就有效；`anchor_ids[0]` 只是存储层的主见证（非空保证），撤销它不会单独让对象失效；投影边另需**至少一个有效投影来源**（决定记录：`docs/design/WITNESS_SEMANTICS.md`） | `Witnessed.anchor_ids: list[str]`（`min_length=1`）；`GraphEdge.witness_anchor_ids` 非空 | 每张表都有 `NOT NULL REFERENCES anchors(anchor_id)` 的主见证列 | `insert_edge` 等写入器在任何 SQL 之前对空列表 / 未知锚点抛 `WitnessMissingError`（`details['missing']`）；`revoke()` 按存在规则级联：无未争议锚点 → 归档，见证全部撤销 → `witness_revoked`，投影来源全部失效 → `projection_revoked` |
| **原子不存边** —— 图关系只在 `graph_edges` | `MemoryAtom` 没有边字段，`content` 里出现 `edges / edge_ids / src_id / dst_id` 被拒 | 原子表没有边列 | `hms_elite/mvmi/*` 没有任何导入路径通向 `hms_elite/graph`；`search()` 不读 `graph_*`（`store.trace` 断言） |
| **高层对象是接口，不是摘要** —— InterfaceEP 必须可以被打开回到支持节点 | `InterfaceEP.support_node_ids` `min_length=2`；`Slots` 是 EP 与 InterfaceEP 共用的同一个模型 | `interface_eps.primary_support_node_id NOT NULL`；`CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL)`；`AFTER INSERT` 触发器写 `edge_projections` | `refine(project(coarsen(S)))['boundary_edges'] == boundary(S)` 往返律；`refine` 只被 `why()` 与 `compile(deep)` 调用（源码级断言） |
| **只修订，不删除；每次变更本身是事件** | `Checkpoint / FeedbackEvent / Transition` 都带锚点 | 所有证据 / 状态表有 `BEFORE DELETE` 触发器；`BEFORE UPDATE OF <内容列>` 触发器让内容不可变（只允许 `valid_to / confidence / visibility / user_meta / archived_reason / valid / …` 变化）；`ux_atoms_head` 保证每个谱系至多一个未归档版本 | `update / delete / delete_all / feedback / checkpoint / rollback / consolidate` 先写 `system \| feedback \| checkpoint \| rollback \| scheduler` 事件 + 锚点，再在同一事务里写其余行；触发器触发 → `StoreError(STORE_003)`。唯一的物理删除路径是 5.12 的 `purge()`：默认关闭、只清状态行、每个被清版本留下 `PURGE` history 行，7 个 `BEFORE DELETE` 触发器只在 `meta.purge_in_progress` 存在的那一个事务内放行 |

锁纪律：`add / add_events` 是 **Txn A（事件 + 锚点）→ 无锁阶段（LLM / 嵌入 / 修订决策）→ Txn B（原子、多对多、向量、节点、边、history、转移）**；A 与 B 之间崩溃可由 `consolidate(phases=['collect'])` 恢复。其他公共写操作各是一个事务；每个公共读操作只取一次 `store.now()` 作为 `as_of`。

---

## 10. 范围表：HMS 完整版中哪些被裁剪

| # | HMS 完整版特性（HMS_arch §） | Lite 决定 | 原因 |
|---|---|---|---|
| 1 | `daemon.py` 后台循环（§3.9, §5.1） | 已发布：`hms_elite/daemon.py::StateDaemon(hms, *, interval_s, phases, filters)`——命名线程上按间隔运行 `consolidate()`，`run_once()` 返回阶段报告，异常记日志后继续；`consolidate()` 本身仍是同步调用，`compile_state` 仍是 consolidate 第 (7) 步 | 后台循环只是对同步 consolidate 的调度 |
| 2 | `runtime.py` Agent 适配器 / action 注册表（§5.1） | 已发布：`hms_elite/runtime.py`（`AgentRuntime` / `ActionRegistry` / `ActionSpec` / `PendingAction`）把工具调用与结果映射为 `file_diff / tool_call / test_log` 事件并经 `add_events()` 闭合；`why / handoff / checkpoint / rollback / feedback` 直接委托门面 | `add_events()` 仍是底层契约 |
| 3 | 逐来源适配器（slack / jira / feishu / sensor）（§3.1） | 简化 → 一个 `SourceEvent`，5 种可抽取 `object_type`：`message, tool_call, file_diff, test_log, sensor_trace` | 传输胶水，不是不变量 |
| 4 | `segmentation.py`（§5.1） | 已发布：`hms_elite/segmentation.py`（`segment / segment_messages / protected_spans / detect_lang`）：句子、段落、代码块、列表项，带精确区间；代码块、URL、标识符不被切断；确定性 | 规则切分，不依赖模型 |
| 5 | `coverage.py` / `scc.py` 层级覆盖编译器（§5.1） | 已发布为可选层：`compiler/coverage.py`（`extract_requirements / covers / select_for_coverage`，贪心集合覆盖）与 `compiler/scc.py`（`compile_hierarchical` → `HierarchicalIR`，先 L3 / L2 接口、按需下钻 L1；`lower()` → `HierarchicalPCIR` 带覆盖义务）；`compile()` 默认仍是槽位份额预算 + 约束下限 | 精度特性，不是不变量 |
| 6 | 视觉 / 感知 | 简化 → list content 只保留文本部分，图片部件以 sha256/mime 落锚（payload `parts`，不存 base64）并计入 `dropped_parts`；`llm.enable_vision: true` 时由 provider 的 `describe_image` 钩子（无锁、事务 A 之前）把图片描述成文字并入正文、记录 `described_parts`（`openai` / `anthropic` 实现；`rule` 无视觉） | 无离线感知；视觉是 provider 能力 |
| 7 | FTS5 / BM25 词法轴 | 拆分 → 召回顺序来自标准库 SQLite 的 FTS5 虚拟表 `atom_fts`（bm25，`keyword_candidates`；`capabilities['fts5']` 为 False 时退回 `LIKE`）；池内关键词*分数*仍是纯 Python token 重叠，标识符 token ×2 | 零可选依赖；有界召回 |
| 8 | Reranker provider | 简化 → `overlap`（默认，离线）+ `llm`（`rerank` 提示词契约，`LLM_002` 时回退 overlap）；`rerank=True` 融合后重排、追加 `rerank_score`，失败保持原序 | Mem0 kwarg 兼容 + 一个离线默认 |
| 9 | 逐话题半衰期学习器 | 简化 → 单一 `scoring.half_life_days` | 确定性 |
| 10 | 嵌入级实体消解 | 简化 → 归一化文本精确匹配 + 别名表；`run_id` 不参与实体身份 | hash 嵌入不是语义嵌入 |
| 11 | Personal / Edge / Group 渲染器（§11.2） | 已发布：`hms_elite/state/`（`personal / edge / group / enterprise` 状态种类、确定性渲染器、`StateAPI`）；`compile()` / `handoff()` 的输出仍是 Work State Card + `StatePackage` | 每种状态都带见证，渲染确定性受测 |
| 12 | LLM coarsen | 简化 → 默认 `rule` coarsen；LLM coarsen 走同一 prompt 契约 | 可复现的往返测试 |
| 13 | 技能提升（§9.7） | 简化 → 有 ≥2 支持节点的 `constraint` 不变量 → `L3_macro` 技能原子 + 可追溯 `distill` 边 | 一次 run 内可达 |
| 14 | κ 的治理 / ACL 评估器（§3.8） | 简化 → 选择时的可见性格门控 + `governance` / `not_revoked` 义务 | 给策略引擎留的钩子 |
| 15 | Benchmark 框架（§12 Phase 5） | 已发布：`hms_elite/eval/`（`run_ablation` / `Task` / `TASKS`（§10 的九个任务）/ `AblationReport` / `render` / `to_json`；LongMemEval 适配器在数据集存在时启用）；消融配置仍映射到 `CompileMode`：`fast` = 仅轴，`axis_graph` = 轴 + 图，`deep` = 轴 + 图 + Support；"仅图" 不适用（图只能从轴种子进入）；`retrieval['ablation'] ∈ {axis, axis_graph, axis_graph_support}` | 包内自带，数据集不随包分发 |
| 16 | HTTP 服务 | 已发布：`hms_elite/server`（`create_app()` / `hms-server`，可选 extra `[server]`）；附带 `cli.py` | 库优先，服务面只包装门面 |
| 17 | 外部索引 provider | 简化；`IndexBase` 可插拔，核心自带 `sqlite` + `memory`，`milvus` 放在可选 extra `[milvus]` 后 | 固定决策 3（核心只依赖 pydantic） |
| 18 | 异步孪生、遥测、通知、`setup_config()` | 异步孪生与遥测已发布：`AsyncHMS`（`memory/async_main.py`，`asyncio.to_thread` 孪生，签名逐一镜像）与 `hms_elite/telemetry.py`（`Usage / UsageMeter / trace() / usage_of(hms)`：LLM 与嵌入调用的次数、token、时长；`GET /usage` / `GET /metrics`）；通知与 `setup_config()` 仍裁剪 | 固定决策 9 只约束 consolidate 的同步语义 |
| 19 | Mem0 `chat()` / `project` | `chat()` 抛 `NotImplementedError`；`project.update()` 抛 `ValueError` | Mem0 API 形状兼容 |
| 20 | Mem0 `history.db` 导入 | 裁剪（由 `hms_elite.migrate.from_mem0` 代替） | 不做列集合差分迁移 |

HMS_arch §3 中未列在上表的一切都**在范围内**：七条查询轴及可执行的 `AxisQuery` 语义、实体 / 关系原子、L2 **与** L3 InterfaceEP、锚点撤销级联、状态包导入、InterfaceEP 转移、编译状态快照。

### 10.1 补齐的短板（两轮） Gaps closed in two rounds

上表第 7、8、16、17 行在两轮补短板中改写；下面按"原状 → 改动 → 验证 → 剩余限制"记账（详细总账见 `docs/design/GAP_CLOSURE_2026-09-10.md`；成本 / 速度 / 设计三份审计在 `docs/design/gaps/`）。

| # | 短板 | 原状 | 改动 | 验证 | 剩余限制 |
|---|---|---|---|---|---|
| 1 | Milvus 索引提供者 | 核心只带 `sqlite` + `memory`；加入 Milvus 后仍按 1024 个 id 一块传 `atom_id in [...]`（每块 13–30 KB，`limit=len(chunk)` 让 HNSW 退化为暴力扫描，每块约 58 ms；RETRIEVAL_DIAGNOSIS T4–T6） | 集合新增 10 个标量字段（scope、三个时刻、`archived`、`scale_level`、`obs_type`、`atom_type`）并建 INVERTED 索引；`IndexFilter` 下推为一次原生过滤 ANN；过滤搜索 `ef = max(ef, limit, 4·limit)`；重开集合 `load_collection`；热路径不再 `count(filter)`；`update_attrs` 跟随 SQLite | `tests/test_index_milvus.py` 12 条（三提供者分数逐位相同、原生过滤 = SQL 门控、archive / rollback 跨提供者、标量模式守卫）；BENCH：N=30000 中位 27.0 ms | milvus-lite 每次 RPC 16–18 ms 固定开销；过滤式 HNSW 在选择性门控下有召回缺口（T6），因此硬门小集合不走 ANN；`count(filter)` 是全量扫描（556 ms@30000） |
| 2 | 重排器 | `search(rerank=True)` 只为 Mem0 签名兼容，不生效 | `reranker` 信封：`overlap`（离线加权词元覆盖率，未配置时的默认）与 `llm`（`rerank` 提示词契约，`LLM_002` 时回退 overlap，其它失败保持检索顺序）；行尾追加 `rerank_score` | `tests/test_reranker.py` 12 条 | 无延迟基准；`llm` 重排每次查询一次 LLM 往返 |
| 3 | AsyncHMS | 范围表第 18 行"异步孪生：裁剪" | `hms_elite.memory.async_main.AsyncHMS`：`asyncio.to_thread` 孪生，方法签名逐一镜像（`inspect.signature` 受测），`async with` / `aclose()`，`usage()` / `trace()` 为同步附加项 | `tests/test_async.py` 9 条（不阻塞事件循环、并发 add + search、结果与同步版相同） | 线程池包装而非原生异步 I/O；同一 store 的写入经 store 锁串行 |
| 4 | hms-server | 范围表第 16 行"HTTP 服务：裁剪" | `hms_elite/server`：`create_app()` / `hms-server`，21 条路由 + `/health`，`HMS_SERVER_TOKEN` Bearer 鉴权，`HMSError.to_dict()` 错误信封（500 脱敏），`X-Request-ID`，`GET /usage` / `GET /metrics` | `tests/test_server.py` 13 条 | 单进程；无多租户 / 配额 / 审计表（`docs/design/gaps/DESIGN_PARITY_AUDIT.md` §1、§3、§4） |
| 5 | purge / vacuum | 库随写入单调增长，`delete_all` 之后没有任何回收路径 | `purge()`（两把钥匙：`retention.enabled` 且 `dry_run=False`；只清状态行；每个被清版本留 `PURGE` history 行）与 `vacuum()`；7 个 `BEFORE DELETE` 触发器带 `meta.purge_in_progress` 门（模式 v2） | 5.12 节的可运行示例；`tests/test_ops.py` 5 条 + `tests/test_facade.py` 1 条 | `keep_witnessed=False` 实测回收 0 个锚点 / 0 个事件（history 行引用锚点）；1000 次 add → delete_all → purge → vacuum 后库体为峰值的 55–66%（COST_AUDIT §3.3：3,346,432 → 2,543,616 B） |
| 6 | 多部件消息与视觉钩子 | list 类 content 只保留文本，图片被丢弃且不留痕 | 图片部件以 sha256 + mime 落锚（payload `parts`，不存 base64）并计入 `dropped_parts`；`llm.enable_vision=true` 时 `describe_image` 钩子（无锁、事务 A 之前）把描述并入正文并记录 `described_parts`；`openai` / `anthropic` 实现 | `tests/test_ingest.py` 5 条（无视觉、有视觉、失败先于任何写入、过程分支）+ `tests/test_providers.py` 3 条 | `rule` LLM 无视觉；无视频；描述不缓存 |
| 7 | spaCy 实体抽取 | 只有正则实体抽取器 | `ingest.entity_extractor: spacy`（`SpacyEntityExtractorConfig`）：NER 区间并入规则结果（规则抓到的 FILE / IDENTIFIER 保留，重叠时规则胜出）；缺 SDK 或模型 → `DependencyError` | `tests/test_entities.py` 5 条（真实模型的 1 条在未安装 spaCy 时跳过） | 实体身份仍是归一化文本精确匹配；模型不随包分发 |
| 8 | 检索 O(N) 与最新优先截断（本轮） | 4.1 节 | 4.1 节 | `tests/test_retrieval.py` 30 条、`tests/test_retrieval_v3.py` 23 条、`tests/test_index_sync.py` 8 条、`tests/test_perf_paths.py` 13 条 | 4.1 节 |

上表之外，范围表第 1、2、4、5、11、15、18 行原本标为"裁剪 / 简化"的模块也已落地，行文已改：`daemon.py`（`StateDaemon`）、`runtime.py`（`AgentRuntime` / `ActionRegistry`）、`segmentation.py`、`compiler/coverage.py` + `scc.py`、`state/`（`StateAPI` 与各状态渲染器）、`eval/`（消融矩阵）、`telemetry.py`（`usage_of` / `trace`）、单次调用抽取（`Ingestor(extraction_mode='single')`，每批消息一次 LLM 往返代替最多三次；`tests/test_extraction_single.py`）与 `deepseek` / `ollama` / `fastembed` 提供者（`tests/test_providers_extra.py`）。它们的入口见第 11 节。

---

## 11. API 速查表 API reference

Mem0 兼容面（`inspect.signature` 逐字受测）：

| 方法 | 返回 | 说明 |
|---|---|---|
| `HMS(config=None, *, store=None)` / `HMS.from_config(dict)` | — | 公共属性：`config, store, index, embedder, llm, graph, ingestor, compiler, feedback_processor, scheduler`；支持 `with HMS(...) as hms:` |
| `add(messages, *, user_id, agent_id, run_id, metadata, timestamp, expiration_date, infer=True, memory_type, prompt)` | `{'results': [row + event (+ revises) (+ conflicts_with)]}` | 5.1 |
| `get(memory_id)` | row 或 `None` | 当前版本 |
| `get_all(*, filters, top_k=20, show_expired=False)` | `{'results': [row]}` | 5.3 |
| `search(query, *, top_k=20, filters, threshold=0.1, rerank=False, explain=False, reference_date, show_expired=False)` | `{'results': [row + score (+ score_details)]}` | 5.3 |
| `update(memory_id, text, metadata, expiration_date, data)` | `{'message': 'Memory updated successfully!'}` | 5.11 |
| `delete(memory_id)` / `delete_all(user_id, agent_id, run_id)` | `{'message': ...}` | 5.11 |
| `history(memory_id)` | `list[dict]` | 5.11 |
| `reset()` / `close()` | `None` | |
| `chat(query)` / `project.update(...)` | 抛 `NotImplementedError` / `ValueError` | Mem0 形状兼容 |

HMS 原生面：

| 方法 | 返回 | 说明 |
|---|---|---|
| `add_events(events, *, user_id, agent_id, run_id, infer=True)` | `{'results', 'events', 'eps', 'edges', 'constraints'}` | 5.2 |
| `compile(query, *, filters, budget_tokens, mode, as_of, require_witness)` | `ContextIR`（`.to_text()`, `.to_pcir()`） | 5.4 |
| `axis_query(AxisQuery \| dict)` | `AxisResult` | 5.3 |
| `why(target_id, *, hops=2, filters, edge_types, include_invalid=False)` | `{'target', 'nodes', 'edges', 'anchors', 'interface_eps', 'paths', 'unresolved_conflicts', 'refined_from', 'chain', 'explanation'}` | 5.5 |
| `feedback(target_id, feedback, *, reason, correction, actor_id)` | `{'message', 'feedback_id', 'actions'}` | 5.6 |
| `revoke(anchor_id, *, reason, actor_id)` | `{'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'}` | 5.6 |
| `checkpoint(*, filters, label)` | `"ckp_…"` | 5.7 |
| `rollback(checkpoint_id, *, reason)` | `{'message', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}` | 5.7 |
| `handoff(*, filters, goal, budget_tokens, checkpoint=True, include_edges=True)` | `StatePackage` | 5.8 |
| `import_package(pkg, *, user_id, agent_id, run_id)` | `{'imported', 'skipped', 'id_map', 'anchor_id'}` | 5.8 |
| `transitions(*, filters, since, scale, fmt='tuples', kinds)` | `list[tuple]` / `list[dict]` / JSONL `str` | 5.9 |
| `consolidate(*, filters, phases)` | `{'anchor_id', 'phases': {...}}` | 5.10 |
| `graph_stats(*, filters)` | dict | 5.10 |
| `context(query, *, filters, top_k, as_of, budget_tokens)` | `{'text', 'session_card', 'session_key', 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids', 'rows', 'rows_text', 'rows_header', 'search_rows', 'scope'}`——`text` 放在问题之前（与 `AgentRuntime.context()` 无关） | 5.13 |
| `purge(*, filters, older_than_days, dry_run=True, keep_witnessed=True)` | `{'dry_run', 'scope', 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable', 'protected', 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}` | 5.12（逃生舱，默认试运行） |
| `vacuum()` | `{'path', 'before_bytes', 'after_bytes', 'freed_bytes'}` | 5.12 |
| `compile(query, *, filters, mode='coverage', …)`（或配置 `compiler.selection: coverage`） | `ContextIR`：第 8 步改为需求覆盖（`coverage.extract_requirements` + `select_for_coverage`）并经 `scc.compile_hierarchical` 先 L3 / L2 后 L1 下钻；`retrieval['selection' \| 'coverage' \| 'hierarchy']`，卡片末尾多一段 `## Coverage`；`.to_pcir()` → `HierarchicalPCIR`（第五条义务 `requirements_covered`，未覆盖时默认抛 `PCIR_001`）；默认 `shares` 路径逐字节不变 | 5.4 |
| `usage()` / `trace(operation, request_id=None)` | `telemetry.Usage` 快照 / `telemetry.trace` 上下文管理器（同 `AsyncHMS`；`/usage` 路由即 `hms.usage()`） | 10.1 |
| `state` 属性（惰性 `StateAPI(hms)`）与同名委托 `work_state / handoff_package / detective_report / rollback_point / personal_state / shared_card / edge_state / group_state / customer_issue_state / incident_state / meeting_decision_state / account_handoff_state` | 各返回 `(状态模型, 渲染文本)`；签名逐字同 `StateAPI`，`AsyncHMS` 镜像为 `async` | 10 |
| `start_daemon(*, interval_s=None, phases=None, filters=None)` / `stop_daemon(timeout=None)` | `StateDaemon`（`interval_s` 缺省取 `scheduler.daemon_interval_s`，两者皆空 → `VALIDATION_009`；正在运行时原样返回）/ `bool`；`close()` 先 `stop_daemon()` | 10 |
| `hms_elite.migrate.from_mem0(memory, hms, *, filters, page_size=100)` | `{'migrated', 'skipped', 'ids'}` | 第 7 节 |

库级其它入口（不在 Mem0 兼容面内）：

| 入口 | 说明 | 节 |
|---|---|---|
| `AsyncHMS(config)` / `await AsyncHMS.create(config)` / `AsyncHMS.from_config(dict)` | 与 `HMS` 同名的 `async` 方法（签名逐一镜像，`asyncio.to_thread`）；`async with`、`aclose()`；`sync` 属性取同步门面；`usage()` / `trace()` 同步 | 10.1 |
| `hms_elite.telemetry.usage_of(hms)` → `Usage`；`telemetry.trace(operation, request_id=None)` → `RequestTrace` | LLM / 嵌入调用的次数、token（`openai` / `anthropic` 记 SDK 精确值，`rule` 记估算并标 `estimated`）、时长 | 10.1 |
| `hms_elite.server.create_app(hms=None, *, config=None)` → FastAPI；命令 `hms-server` | 21 条路由 + `/health`；`HMS_SERVER_TOKEN` 非空时全部路由要求 Bearer | 10.1 |
| `hms_elite.daemon.StateDaemon(hms, *, interval_s=60.0, phases=None, filters=None)`：`start() / stop(timeout) / run_once()`，属性 `runs / failures / last_report / last_error / is_running` | 命名线程上按间隔运行 `consolidate()`；异常记日志后循环继续 | 10 |
| `hms_elite.runtime.AgentRuntime(hms, *, user_id, agent_id, run_id, registry=None, goal=None, source_system='agent', visibility=None, infer=True)` + `ActionRegistry` / `ActionSpec` | 工具调用与结果映射为 `file_diff / tool_call / test_log` 事件并经 `add_events()` 闭合成 EP | 10 |
| `hms_elite.state.StateAPI(hms)`：`work_state / handoff_package / detective_report / rollback_point / personal_state / shared_card / edge_state / group_state / customer_issue_state / incident_state / meeting_decision_state / account_handoff_state` | 各返回 `(状态模型, 渲染文本)`；渲染确定性且带见证 | 10 |
| `hms_elite.segmentation.segment(text, *, max_chars=400, lang=None)` / `segment_messages(messages, ...)` | 句子 / 段落 / 代码块 / 列表项，带精确区间；代码块、URL、标识符不被切断 | 10 |
| `hms_elite.compiler.coverage.select_for_coverage(items, requirements, *, budget_tokens, min_per_slot=0)`；`hms_elite.compiler.scc.compile_hierarchical(ir, *, requirements=None, budget_tokens=None, layers=None)` → `HierarchicalIR`（`scc.lower()` → `HierarchicalPCIR`） | 覆盖式选择与分层编译（先 L3 / L2 接口、按需下钻 L1）；可选层，`compile()` 默认仍用槽位份额预算 | 10 |
| `hms_elite.eval.run_ablation(hms_factory, tasks=None, ablations=None)` → `AblationReport`（`render()` / `to_json()`） | 九个 §10 任务 × 消融配置的矩阵；LongMemEval 适配器在 `HMS_LONGMEMEVAL_PATH` 指向数据集时启用 | 10 |
| `hms_elite.hms4eye.capability.HMSCapabilities(hms).reflect_evidence(query, *, user_id)` | hms4eye `reflect` 的 `organize` 形状证据包（`question / question_date / mode / ledger_rows / controls / source_snippets / answer_ready_context` + `conclusion / chain`），由侦探报告构造，每行 `ledger_rows` 带 `anchor_ids`；无法解析时为空包 | 10 |
| `from hms_elite import AgentRuntime, StateDaemon, StateAPI, StateKind, Usage` | 根包在 `HMS / AsyncHMS / HMSConfig` 与重排器类之外再导出这五个 §2 尾部入口 | 10 |

错误码（都在 `hms_elite.exceptions`，`HMSError(message, error_code, details, suggestion)`）：`VALIDATION_001…013`、`NOT_FOUND_001`（以上是 `ValueError` 子类）、`CONFIG_001/002/003`、`LLM_001/002/003`、`EMBED_001`、`STORE_001/002/003`、`WITNESS_001`、`PCIR_001`、`DEPS_001`。

---

## 12. 命令行 CLI

安装后可用 `hms` 命令（`argparse`，存储在 `$HMS_DIR/hms.db`）：

```bash
export HMS_DIR=/tmp/hms-demo
hms add "I like oat-milk lattes in the morning." --user alice --run run-coffee     # JSON（add 信封）
hms search latte --user alice --top-k 3                                             # JSON（search 信封）
hms compile coffee --user alice --run run-coffee --mode fast                        # Work State Card 文本
hms add "Do not modify the subscription state machine." --user alice --run run-billing   # 先给 run-billing 一条约束
hms why "why must retry.ts not be modified" --user alice --run run-billing          # JSON（why 结果）
hms transitions --user alice --run run-coffee --fmt jsonl                           # JSONL
hms consolidate --user alice                                                        # JSON（consolidate 报告）
hms stats --user alice                                                              # JSON（graph_stats）
```

`why` 的问题模式需要范围内至少有一条约束 / 片段 / 事实（所以上面先向 `run-billing` `add` 了一条约束）；找不到可解析目标时，以及任何其它 `HMSError`，CLI 都把 `error.to_dict()` JSON 打印到 stderr 并以退出码 1 退出，而不是回溯。

```text
$ hms compile coffee --user alice --run run-coffee --mode fast
# Work State Card
Scope: user_id=alice run_id=run-coffee | As of: 2026-09-02T22:43:00.701756+00:00 | Mode: fast
## Current Goal
coffee
## Valid Facts
- (none)
## Invalidated Assumptions
- (none)
## Failed Paths
- (none)
## Constraints
- (none)
## Preferences
- User likes oat-milk lattes in the morning.
## Next Action
Continue: coffee
## Evidence
- chat://run_id=run-coffee&user_id=alice/0 sha256:78bbbb8162b1 @2026-09-02T22:43:00.446447+00:00
```

（`Preferences` / `Skills` 段只在非空时输出；这里 `As of` 与证据时间戳是真实时钟，因为 CLI 的 `add` 没有 `--timestamp`。）

---

## 13. 测试与开发 Testing / dev

当前测试套件共 **524 个测试**（522 通过、2 条按条件跳过），分布在 31 个测试文件里，本机（Apple Silicon，CPython 3.14）约 33 s：`test_ops` 63、`test_retrieval` 42、`test_foundation` 35、`test_compiler` 27、`test_ingest` 25、`test_graph` 24、`test_state` 23、`test_retrieval_v3` 23、`test_facade` 22、`test_providers` 20、`test_mvmi` 20、`test_extraction_single` 20、`test_ablation` 20（1 条在未设 `HMS_LONGMEMEVAL_PATH` 时跳过）、`test_hms4eye_adapter` 14、`test_server` 13、`test_perf_paths` 13、`test_telemetry` 12、`test_reranker` 12、`test_index_milvus` 12（需 `[milvus]`，缺失时整文件 `importorskip`）、`test_hms4eye_capability` 12、`test_coverage_scc` 11、`test_async` 9、`test_providers_extra` 8、`test_index_sync` 8、`test_wiring` 7、`test_segmentation` 7、`test_runtime` 7、`test_daemon` 6、`test_entities` 5（真实 spaCy 模型的 1 条在未安装时跳过）、`test_scenarios` 2（S1–S6 在 `sqlite` 与 `memory` 两种索引上各跑一遍）、`test_readme` 2。
(498 tests across 30 files: 496 passed, 2 conditionally skipped.)

```bash
pip install -e ".[test,dev]"
pytest -q                                   # 全离线；conftest 自动把 HMS_DIR 指向 tmp_path 并禁用 socket
pytest -q tests/test_scenarios.py           # S1–S6 黄金场景，在 sqlite 与 memory 两种索引上各跑一遍
pytest -q tests/test_retrieval.py tests/test_retrieval_v3.py   # 4.1 节：等价性定理、无截断、通道 / 池 / 安全上限、索引缓存
pytest --update-golden                      # 重新生成 tests/fixtures/golden/*（卡片、why 链、transitions JSONL、观察 JSON）
ruff check hms_elite tests examples          # line-length 120，规则 E4/E7/E9/F/I
ruff format --check hms_elite tests examples
isort --check-only --profile black hms_elite tests examples
PYTHONPATH=. python examples/quickstart.py  # 与第 2 节的输出逐字一致（S6 rollback 的 atom id 除外）
```

测试约定（沿用 Mem0 的 `tests/AGENTS.md`）：mock provider SDK，永不 mock 被测代码；每个 bug 修复附带一个没有它就会失败的回归测试；只用 `monkeypatch.setenv`；无时间断言；无网络。`tests/conftest.py` 是唯一的 conftest，提供 `FrozenClock`（起点 `2026-09-02T10:00:00+00:00`，只在 `tick()` 时前进）、`SequentialIds`（`atm_000001` …）、`store / hms / hms_scripted / events / golden / scope4` 等夹具。黄金卡片使用显式 `as_of`，证据行只含夹具时间戳，所以在真实时钟下同样逐字节相同。

---

## 14. 双语注释约定 Bilingual docstrings

每个代码文件（`hms_elite/**/*.py`、`tests/**/*.py`、`examples/*.py`）都以两行模块 docstring（英文一行、中文一行）与八行头注释开头，每个公开函数 / 方法的 docstring 都是"英文一句 + 中文一句 + `Input / 输入：` + `Output / 输出：` + `State effect / 状态影响：`"，每个参数一行英文（`arg: …`）紧跟一行中文（`arg：…`，全角冒号）。这个布局由 `tests/test_foundation.py::test_bilingual_docstrings` 遍历全部模块强制执行：

```python
# （示意）
"""HMS: the Mem0-compatible public API over the dual-plane store.
HMS：双平面存储之上、与 Mem0 兼容的公共 API。
"""

# Purpose (EN): What this file is responsible for.
# 目的（中）：说明本文件负责什么。

# Input (EN): What enters this module.
# 输入（中）：说明模块接收什么。

# Output (EN): What this module produces.
# 输出（中）：说明模块产出什么。

# State effect (EN): Whether this module mutates state, graph, index, or store.
# 状态影响（中）：说明模块是否修改状态、图、索引或存储。


def compile(self, query: str, *, filters: dict) -> ContextIR:
    """Compile the current work state for a query into a ContextIR.
    把某个查询对应的当前工作状态编译成 ContextIR。

    Input / 输入：
        query: The goal or question to compile for.
        query：要编译的目标或问题。

        filters: Mem0 filters; at least one scope key.
        filters：Mem0 过滤器；至少一个 scope 键。

    Output / 输出：
        A ContextIR with six disjoint slots and evidence.
        带六个互斥槽位与证据的 ContextIR。

    State effect / 状态影响：
        Read-only.
        只读。
    """
```

代码注释一律英文；设计文档见 `docs/design/`（`CONTRACT.md` 为规范，`DESIGN.md` 为理由，`HMS_arch.md` 为架构意图，`MEM0_MAP.md` 为 Mem0 工程参考）。

## 15. 部署 Deployment

容器化部署到 dev 服务器，nginx 前置，宿主端口 `18090`（容器内 app `8000` / nginx `8080`，均不对外发布）。

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 单阶段镜像；装 `libgomp1`（onnxruntime 需要），build 期把 `bge-small-en-v1.5` 烘进 `/app/model-cache/fastembed` |
| `docker-compose.yml` | app + nginx；命名卷 `hms_data:/data` 存 `hms.db` + WAL；配置以只读 bind mount 挂入 |
| `docker-compose.prod.yml` | 仅为基础/覆盖的分层与 CI 语法闸存在；密钥用 `:?` fail-fast。当前没有 prod 目标 |
| `docker/nginx.conf` | 单入口；**挡掉 `/docs`、`/redoc`、`/openapi.json`** |
| `deploy/config.dev.json` | 提交进仓库的**无密钥**配置模板 |
| `.env.example` | 服务器侧 `.env` 的字段清单（`.env` 本身不入库，0600） |
| `scripts/deploy_smoke.sh` | 部署后冒烟：鉴权 401、docs 404、写入、LLM 出网探针、检索、重启持久化、清理 |
| `.gitlab-ci.yml` | `verify` → `build` → `deploy:dev`；push 到 `dev` 分支即部署 |

**密钥不进任何文件。** 配置里的 `llm.config` 不写 `api_key`，由容器环境变量 `DEEPSEEK_API_KEY` 提供（`llms/deepseek.py:127`）。因此 `deploy/config.dev.json` 可以直接提交。

### 首次部署（服务器侧）

```bash
mkdir -p /home/dingjiangxu/deploy/hms_elite
# 写 .env（见 .env.example，chmod 600），HMS_SERVER_TOKEN 用 openssl rand -hex 32 生成
```

部署目录由 CI 自己 `git init` 维护，**不要手工 clone 进去** —— 只需放 `.env`。GitLab 侧只需要一个 **project runner**（shell executor，tags `dev,deploy`；**不要勾 Protected** —— `dev` 是普通分支，勾了 job 永远不派，表现是 pipeline 卡在 pending）。

`.env` **手工放在部署目录即可，不要建 File 型 CI 变量**：`.gitlab-ci.yml` 里的 `install -m 600` 只在 `HMS_ELITE_DEV_ENV` 存在时执行，而它一旦存在就会在**每次部署时覆盖**手写的那份 —— 两处真相必然漂移。密钥只该有一个落点。

### 验证

```bash
curl -s $BASE/health                                    # providers.llm 必须是 deepseek
curl -s -o /dev/null -w '%{http_code}\n' $BASE/stats    # 无 token 必须 401
curl -s -o /dev/null -w '%{http_code}\n' $BASE/docs     # 必须 404
curl -s -H "Authorization: Bearer $TOKEN" $BASE/usage   # 必须 estimated=false 才算出了网
```

两个最容易看错的判据：

- `/health` 的 `providers.llm` 是判断「配置有没有挂上」的关键 —— 若显示 `rule`/`hash`，说明容器跑的是离线默认配置。
- `/usage` 要看的是 **`estimated=false`**，不是 `llm_calls>0`。`llm_calls` 对离线 `rule` 提供者同样会递增，调用抛异常时也会递增（`llms/base.py:75`）；只有提供者真的回了 token 用量时 `estimated` 才为 false（`llms/base.py:88-98`）。这条已在本机用离线配置实测：`llm_calls=1` 而 `estimated=true`，冒烟脚本按预期失败。

### 三个运维坑

1. **改 `deploy/config.dev.json` 不会自动生效**：`docker compose up -d` 只在 compose 配置变化时重建。改配置内容后要 `docker compose up -d --force-recreate hms-elite`。
2. **`HMS_SERVER_TOKEN` 与 `DEEPSEEK_API_KEY` 只在构造时读一次**：改完必须重跑一次 `deploy:dev` 才生效。
3. **`/docs` 靠 nginx 挡，是架构层面的口子**：`create_app` 没传 `docs_url=None`，而 Bearer 依赖挂在 `APIRouter` 上，所以这三个路径天然绕开鉴权。**给 app 服务加 `ports:` 就等于让它裸奔** —— 冒烟脚本里有一条 404 断言钉住这件事。

**模型不进卷。** 给 `/app/model-cache` 挂空命名卷会**遮住**烘进镜像的模型，逼容器去 HF 下载，而容器里 `huggingface.co` 不可达。换模型只能改 Dockerfile 重建（这是刻意的：把「改模型」变成一次可评审的提交）。

License: Apache-2.0.
