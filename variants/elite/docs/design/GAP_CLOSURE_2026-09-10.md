# GAP_CLOSURE_2026-09-10 — 两轮补短板总账

日期：2026-09-10。对象：`hms_elite`（HMS Elite）相对 Mem0 2.0.20 与 hms_lite-main 的短板，以及本仓库自己的结构性缺陷。
每个条目按 **原状 → 改动 → 验证 → 剩余限制** 记账。数字只引用仓库内的文件：`docs/design/RETRIEVAL_DIAGNOSIS.md`（改造前归因，表 T1–T8）、`docs/design/RETRIEVAL_BENCH_2026-09-10.md`（改造后规模基准）、`docs/design/gaps/{COST_AUDIT,SPEED_AUDIT,DESIGN_PARITY_AUDIT}.md`（三份审计）、`docs/design/bench_results/*.json`（原始数据）与 `tests/`（回归测试）。没有数字的地方明确写"无基准"，不补数。

## 0. 口径

| 项目 | 取值 |
|---|---|
| 机器 / 解释器 | Apple Silicon（darwin 27.0）；CPython 3.14.6（anaconda；numpy 2.4.6，pymilvus 3.0.1 + milvus-lite，sqlite 3.53.2）；mem0 对照臂 CPython 3.12.14（mem0ai 2.0.20 + faiss） |
| 嵌入器 / LLM | `hash@64` / `rule`（除非注明） |
| 检索基准 | `scripts/bench/run_bench_v3.py`（语料 `scripts/bench/profile_search.py::gen_texts(N)`，N ∈ {300, 3000, 30000}，5 个查询，`top_k=20`，`threshold=0.1`，预热 1 次、计时 20 次取中位与 p90） |
| oracle | 旧穷举路径（`tests/test_retrieval.py::_exhaustive_search(max_candidates=10**6)`）的 top-20；重叠率 = \|新 ∩ oracle\| / 20 |
| 测试基线 | 本轮结束：498 collected = 496 passed + 2 skipped（LongMemEval 数据集缺失、spaCy 未安装），约 33 s；ruff 干净；`scripts/scan_en.py` 纯英文注释块 = 0 |

## 1. 第一轮：七个功能短板

| # | 短板 | 原状 | 改动 | 验证 | 剩余限制 |
|---|---|---|---|---|---|
| 1.1 | Milvus 索引提供者 | 核心只带 `sqlite` + `memory`（范围表第 17 行）；提供者接入后第一版按 1024 个 id 一块传 `atom_id in [...]`，表达式 4.5 KB（300 id）/ 15.4 KB（1024）/ 30 KB（2000），`limit=len(chunk)` 让 HNSW 的 ef 抬到块大小、等于暴力扫描并回传全部向量：每块 22–28 ms（T6，`limit=len + vector`），在 search 路径上每块约 58 ms（T4/T5：3 块 173 ms，2 块 120 ms） | `hms_elite/index/milvus.py`：`MilvusIndexConfig(uri, token, db_name, collection, index_type ∈ {AUTOINDEX, HNSW, FLAT}, M, efConstruction, ef, batch_size, candidate_chunk, ef_recall_factor=4, scalar_index ∈ {INVERTED, none})`；集合新增 10 个标量字段（`user_id / agent_id / run_id`、`time_anchor / valid_from / valid_to`、`archived`、`scale_level / obs_type / atom_type`）；`IndexFilter` 渲染为一条标量表达式做原生过滤 ANN；过滤搜索 `ef = max(ef, limit, ef_recall_factor·limit)`，id 块 `ef` 提到块大小；重开集合 `load_collection`；热路径不再 `count(filter)`；命中经 `rank_hits` 重算，分数与 sqlite 逐位一致；`HMS.close()` 释放客户端 | `tests/test_index_milvus.py` 12 条（契约与 sqlite / memory 一致、分数逐位相同、候选分块合并、零分尾部、维数与外来集合守卫、端到端与 sqlite 相同、原生过滤 = SQL 门控、标量来自 store 与 `update_attrs`、标量模式守卫、archive / rollback 三提供者、默认标量索引）；BENCH §1：`elite_milvus` search 中位 20.8 / 87.0 / 27.0 ms（N=300 / 3000 / 30000） | milvus-lite 每次 RPC 16–18 ms 固定开销（无过滤 HNSW 一次，`ef=320`）；N=3000 时四个查询整体入池，池内 `candidate_ids` 调用 50–60 ms，因此 3000 比 30000 慢；过滤式 HNSW 在匹配率 ≤ 5% 的门控下有召回缺口（T6：`limit=80` 只得 51–67 行），所以硬门小集合不交给 ANN；`count(filter)` 是全量扫描（556 ms@30000）；标量 INVERTED 索引建集合时每字段约 0.5 s |
| 1.2 | 重排器 | `search(rerank=True)` 只为 Mem0 签名兼容，参数不生效；范围表第 8 行"仅接受 kwarg" | `hms_elite/reranker/`：`RerankerBase` 契约；`OverlapReranker`（离线加权词元覆盖率，标识符 ×2；`reranker: null` 时 `search(rerank=True)` 的默认，info 日志宣告一次）与 `LLMReranker`（`rerank` 提示词契约，合成 id 防串改，`LLM_002` 时回退 overlap，其它失败保持检索顺序，`max_chars` 截断输入）；行尾追加 `rerank_score`；`REGISTRY['reranker']`，未知 provider → `CONFIG_002` | `tests/test_reranker.py` 12 条（契约形状、加权覆盖率排序、平局稳定、`top_k` 在排序后截断、LLM 成功路径、畸形输出保持原序、未知 id 丢弃、rule LLM 回退 overlap、传输错误传播、提示词往返、配置禁止多余字段、根包导出） | 无延迟基准；`llm` 重排每次查询一次 LLM 往返；重排只作用于已融合的 top-k，不改变召回 |
| 1.3 | AsyncHMS | 范围表第 18 行"异步孪生：裁剪"；hms_lite-main 有异步服务层，Elite 只能同步调用 | `hms_elite/memory/async_main.py::AsyncHMS`：`asyncio.to_thread` 孪生，每个公开方法同名同签名（`inspect.signature` 逐一比对），`create()` / `from_config()` / `async with` / `aclose()`，`sync / config / store / project` 同步属性，`usage()` / `trace()` 同步附加项 | `tests/test_async.py` 9 条（签名镜像、参数原样转发、不阻塞事件循环、并发 add + search、上下文管理器与幂等关闭、离线默认与错误透传、结果与同步版相同、usage / trace 为同步附加项） | 线程池包装而非原生异步 I/O；同一 store 上的写入经 store 锁串行（`MemoryIndex.search` 对字典做快照以容忍并发 `upsert`） |
| 1.4 | hms-server | 范围表第 16 行"HTTP 服务：裁剪"；hms_lite-main 有 FastAPI 面 | `hms_elite/server/`（`create_app(hms=None, *, config=None)`，命令 `hms-server`，可选 extra `[server]`）：21 条路由（`/memories` 的 5 个 Mem0 动作、`/search`、`/reset`、`/events`、`/compile`、`/why/{target}`、`/feedback`、`/checkpoint`、`/rollback`、`/handoff`、`/transitions`、`/consolidate`、`/stats`、`/usage`、`/metrics`）+ `/health`；`HMS_SERVER_TOKEN` 非空时全部路由要求 Bearer；`HMSError.to_dict()` 错误信封（`AuthError` 401 / `NOT_FOUND_*` 404 / `ValidationError` 400 / 其它 500，500 脱敏只留 `error_type` 与 `error_code`）；`X-Request-ID` 回写；每个请求一条 `telemetry.trace` | `tests/test_server.py` 13 条（健康与路由表、Mem0 端点往返、`expiration_date` 缺席与 null 的区分、错误映射、内部错误脱敏、原生端点流程、Bearer 鉴权、usage 与 metrics、request id 往返、默认门面与 lifespan、依赖守卫与 `main`、导入无副作用、scope 合并优先级） | 单进程；无多租户（`DESIGN_PARITY_AUDIT` §1）、无配额 / 密钥表（§4）、无审计表（§3）；`/metrics` 只是请求计数与时长，不是 Prometheus 全量指标 |
| 1.5 | purge / vacuum | "修订代替删除"让库随写入单调增长；`delete_all` 后没有任何回收路径；COST_AUDIT F5：存储 3.0–3.5 × Mem0（2,949–3,285 B/add vs 942–985） | `hms_elite/ops/retention.py` + `HMS.purge(filters, older_than_days, dry_run=True, keep_witnessed=True)` / `HMS.vacuum()`：两把钥匙（`retention.enabled` 且 `dry_run=False`）；只清状态行（`memory_atoms` 及从属行、自身的 `observations / eps` 来源行），锚点、`source_events`、`graph_edges`、`history` 永不清；每次清理是一条 `system(op='purge')` 事件 + 锚点，每个被清版本留一条 `PURGE` history 行（sha256、版本、原因，不留文本）；6 类保护（`checkpoint_live / edge_endpoint / ep_referenced / head_version / interface_ep_support / source_shared`）按原因列出；7 个 `BEFORE DELETE` 触发器带 `meta.purge_in_progress` 门（模式版本 2，旧库自动迁移）；`consolidate` 的 `prune` 只在 `retention.enabled` 且 `run_in_prune` 时调用 | README 5.12 的可运行示例（`tests/test_readme.py` 逐字比对）；`tests/test_ops.py` 5 条（试运行不改任何行、保护规则、真实清理带审计且幸存者完整、prune 双开关、vacuum 缩小文件）+ `tests/test_facade.py::test_purge_facade_is_explicit_default_off_and_audited`；`test_purge_gate_and_schema_migration_2` 断言 v1 → v2 迁移的 `sqlite_master` 与新建库逐行相同 | COST_AUDIT §3.3：1000 次 add（3,346,432 B）→ delete_all（3,846,144，+0.5 MB 的 history 行）→ purge（4,419,584，又写 1000 条 PURGE 行）→ vacuum 后 2,543,616 B，即峰值的 55–66%；`keep_witnessed=False` 实测 `anchors_purged=0, events_purged=0`，因为 history 行引用锚点且 history 只追加（F6 建议的 tombstone 方案未做） |
| 1.6 | 多部件消息与视觉钩子 | list 类 content 只保留文本部分，图片被丢弃且不留痕；范围表第 6 行"无离线感知" | 图片部件以 sha256 + mime 落锚（payload `parts`，不存 base64）并计入 `dropped_parts`（纯文本哈希不变）；`llm.enable_vision=true` 时 `LLMBase.describe_image` 钩子（无锁、事务 A 之前）把描述并入正文并记录 `described_parts`，`vision_details ∈ {auto, low, high}`；`openai` 走 chat.completions 的 `image_url` 部件，`anthropic` 走 Messages 的 `image` 块（data URI → base64，http → url）；`rule` 无视觉，图片仍计入 `dropped_parts`；视觉失败 → `LLM_001` 且不写任何行 | `tests/test_ingest.py` 5 条（无视觉保持旧行为、有视觉在事务 A 之前描述、无感知能力保持丢弃、失败先于任何写入、过程分支看到描述）+ `tests/test_providers.py` 3 条（默认 rule / scripted 的 `describe_image`、openai 的 `image_url` 部件、anthropic 的 `image` 块）；`tests/test_foundation.py` 的 payload 归一化用例断言 `dropped_parts` 键 | 无视频、无音频；描述不缓存（同一图片重复 add 重复调用）；无大小准入；`DESIGN_PARITY_AUDIT` §8 列出的 hms_lite 视频抽帧 + 逐句接地未做 |
| 1.7 | spaCy 实体抽取 | 只有正则实体抽取器（FILE / IDENTIFIER / QUOTED / PROPER / CJK）；`DESIGN_PARITY_AUDIT` §7 provider 广度不足 | `hms_elite/entities/`：`EntityExtractorBase` 契约，`RuleEntityExtractor`（与 `utils.text.extract_entities` 逐字节一致）与 `SpacyEntityExtractor(model="en_core_web_sm", labels={PERSON: PROPER, …}, disable, strip_articles)`；NER 区间**并入**规则结果（规则抓到的 FILE / IDENTIFIER 保留，重叠时规则胜出）；`ingest.entity_extractor` 按名称选择，配置在 `ingest.entity_extractor_config`；缺 SDK 或模型 → `DependencyError(DEPS_001)`；抽取器同时到达 ingest、rule LLM、feedback 与 import 路径 | `tests/test_entities.py` 5 条（规则抽取器逐字节一致并注册、spaCy 合并保留规则候选并加入 NER 区间、依赖与模型守卫、真实模型合并（未安装 spaCy 时跳过）、配置的抽取器到达四条路径） | 实体身份仍是归一化文本精确匹配 + 别名（范围表第 10 行）；模型不随包分发；无中文 NER 模型的默认配置 |

## 2. 第二轮：检索 O(N) 的根治（本轮）

### 2.1 缺陷与归因（改造前）

- 数据流：`HMS.search` → `mvmi.query.search` → `execute_axis_query` → 门控 SQL 选出 G → `scoring.semantic_scores` 以 `index.search(vector, candidate_ids=G, top_k=len(G))` 索要全部候选的语义分 → 逐个 `hydrate`（pydantic `MemoryAtom` + `Scope` 各校验一次、JSON 列、链接查询）→ `atom_keyword_tokens` 分词 → keyword / entity / recency / support → `fuse` → 阈值 → 排序。每候选约 45 µs（T2，explain=True）。
- 规模：T3（`max_candidates=10**6`，即不截断）sqlite N=3000 identifier 查询 132.9 ms（2917 候选），N=30000 1,612.9 ms（29,167 候选）；成本构成：hydrate 35%、分词 24%、`index.search` 10%、axis SQL 8%、其余信号 7%、fuse + explain 3%、排序与循环 13%。
- 截断：`max_candidates=2000`（出厂默认）把 N=30000 压到 108–122 ms（T1），但 `ORDER BY time_anchor DESC LIMIT 2000` 让 N > 2000 之后较旧的记忆不分相关性地退出候选（`who is Sarah` 的硬实体门在 N=30000 命中 3000 行，只打分最新 2000 行）。
- Milvus：T4/T5 每块 id 列表 ANN 约 58 ms，`entity` 查询 300 个 id 也要 43–49 ms；capped 219 ms / uncapped 317 ms（N=3000），比 sqlite 的 89 / 133 ms 更慢——ANN 在这条路径上没有任何价值。
- 过滤式 ANN 的召回缺口：T6 显示 AUTOINDEX / HNSW 在匹配率 ≤ 5% 的门控下 `limit=80` 只得 51–67 行，FLAT 无缺口。

### 2.2 改动

| 项 | 改动 | 位置 |
|---|---|---|
| 池化召回 | `recall_pool(...) -> Pool`：semantic（索引）/ keyword（FTS5）/ entity（`atom_entity`）/ recency（`ix_atoms_time`）四通道各带 K，轮转并集去重截到 `pool_max`，记录 `recalled_by`；硬门或 `len(G) <= pool_max` 时整体入池；只对池水合并算五路信号；回退阶梯每级走同一套召回；`Pool.stats()` 进入 `AxisResult.retrieval` 与 `ContextIR.retrieval` | `hms_elite/mvmi/query.py` |
| 配置 | `RetrievalConfig(k_semantic, k_keyword, k_entity, k_recency, pool_max_factor=4, hard_gate_pool=True, safety_limit=50000)`，`k_for(top_k)`；`ScoringConfig.max_candidates` 改为 None 哨兵的已弃用别名（非 None 发 `DeprecationWarning`，不截断）；`HMS.search` 与 `compile` 都传 `config.retrieval` | `hms_elite/configs/base.py`、`memory/main.py` |
| 解释块 | `score_details` 追加 `recalled_by / pool_size / channel_sizes`，前 14 键不变 | `hms_elite/utils/scoring.py` |
| 索引契约 | `IndexFilter`、`search(filter=)` / `count(filter=)`、`supports_filter`、`INDEX_ATTR_KEYS` / `atom_index_attrs` / `normalize_attrs`、`upsert(attrs=)`、非抽象钩子 `update_attrs` / `close`、`gate_ids` SQL 退路、`select_top_k`（矩阵乘挑选 + `dot` 逐位重算） | `hms_elite/index/base.py` |
| sqlite 提供者 | 进程内 float64 矩阵 + 标量缓存，以 `(data_generation, rebuild_generation, data_version)` 为键增量同步；`IndexFilter` 为 numpy 掩码（语义与 SQL 门控相同）；锁序 store 锁先于索引锁 | `hms_elite/index/sqlite.py` |
| memory 提供者 | 缓存的 float32 矩阵快照 + `select_top_k`，与 `rank_hits` 逐位相同 | `hms_elite/index/memory.py` |
| Milvus 提供者 | 10 个标量字段 + INVERTED 索引；`filter_expr` 原生过滤 ANN；`ef` 规则；`load_collection`；不再 `count(filter)`；`update_attrs` 读-合并-重写 | `hms_elite/index/milvus.py` |
| 标量同步 | `StateStore.attr_listeners`：`archive_atom / restore_atom / set_valid_to` 排队，COMMIT 后按票号顺序在 store 锁之外派发完整 `atom_attrs` 快照；`HMS.__init__` 注册 `index.update_attrs`，`HMS.close` 先移除再 `index.close()` | `hms_elite/memory/storage.py`、`memory/main.py` |
| FTS5 / 模式 v3 | `atom_fts(atom_id UNINDEXED, tokens, tokenize='unicode61')`，`fts_tokens` 预分词（CJK 二元组、标识符、主题、实体键）；`upgrade_3` + `backfill_fts`；`VIRTUAL_TABLES = ('atom_fts',)`、`FTS_SHADOW_TABLES`；`capabilities['fts5']`；无模块时跳过并告警、关键词通道退回 `LIKE` | `hms_elite/memory/schema.py` |
| store 通道 | `keyword_candidates`（两阶段 bm25，先在 `atom_fts` 内排序再对窗口施加门控）、`entity_candidates`、`axis_ids(order=None)`、`axis_select(atom_ids=)`（VALUES-CTE CROSS JOIN）、`index_hint`（`TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX`）、写入代数、`clone_schema` | `hms_elite/memory/storage.py` |
| compile 复用 | 候选阶段调用 `execute_axis_query(..., retrieval=config.retrieval)`，`ContextIR.retrieval` 携带池统计 | `hms_elite/compiler/compile.py` |

### 2.3 验证

| 项 | 结果 |
|---|---|
| 等价性定理 | `tests/test_retrieval.py::test_equivalence_theorem`（sqlite / memory × N ∈ {5, 30, 60} × whole / channels；20 个查询 × `top_k ∈ {3, 5, 10}`）与 `test_equivalence_theorem_milvus`：`|G| <= min(K)`（或整体入池）时 `(atom_id, score)` 与 `score_details` 前 14 键逐位相同；黄金场景 `tests/test_scenarios.py` 与 `test_mvmi` / `test_facade` 原样通过（其语料 `|G| < 60`） |
| 无截断 | `test_no_newest_first_truncation`：`ScoringConfig(max_candidates=10)` 下仍找到久远的唯一标识符原子，而带同样上限的参考实现丢掉它 |
| 溯源 / 安全上限 / 仅轴路径 / 别名 / compile / Python 侧过滤 | `test_channel_sizes_and_recalled_by`、`test_hard_gate_whole_set_enters_pool_or_channels`、`test_safety_limit_warns_never_truncates`（两条告警按前缀区分，原生下推下只有池告警）、`test_axis_only_paths_keep_sql_semantics`、`test_max_candidates_is_a_deprecated_alias`、`test_compile_candidates_reuse_the_pool`、`test_python_filter_scores_the_whole_gate`（json1 关闭时对 G 整体打分且与穷举逐位相同） |
| 通道与缓存 | `tests/test_retrieval_v3.py` 23 条：不排序探测与索引提示、通道在级别门控内搜索、实体门为非相关子查询、关键词两阶段、写入代数与 `data_version`、`memory_atoms` 更新受监听器覆盖、sqlite 缓存跟随每次写入、`select_top_k` 逐位相同、新近度规则、语义分复用、`AxisResult.retrieval`、`atom_type` 下推、门面传递 `RetrievalConfig`、摄取 upsert 携带标量、Milvus 重开 |
| 标量同步 | `tests/test_index_sync.py` 8 条（三提供者提交顺序派发、事务内搜索不死锁、监听器只注册一次并在 close 时移除、迟到的主题 / 实体到达关键词通道）；`tests/test_index_milvus.py::test_archive_hides_from_every_provider_and_rollback_restores` |
| 表数与迁移 | `test_schema_migrations_and_capabilities`（22 基表 + 1 虚拟表 + 5 影子表按名列出）、`test_fresh_database_is_created_directly_at_schema_version`、`test_schema_v3_backfills_fts_from_a_v2_file`（v1 → v2 → v3 的 `sqlite_master` 与新建库逐行相同、回填只含存活原子） |
| 规模（BENCH §1） | `elite_sqlite` search 中位 8.21 / 4.54 / 15.9 ms，p90 10.1 / 9.30 / 21.9 ms（N=300 / 3000 / 30000）；`elite_milvus` 20.8 / 87.0 / 27.0 ms；mem0 + FAISS 0.20 / 0.22 / 0.35 ms。与改造前对照：sqlite@30000 从 172–210 ms（截断）/ 1.6 s（不截断）降到 13.6–19.6 ms（×11–13 / ×100）；3000 → 30000 增长 ×3.5（低于线性）。写入与 N 无关（0.7–1.2 ms/条），mem0 的 `_save` 在 N=30000 时 49–98 ms/条 |
| 重叠率（BENCH §1、§4） | N=300：三臂 1.0；N=3000：sqlite 0.75（最小 0.35）、milvus 0.81；N=30000：sqlite 0.60（最小 0.20）、milvus 0.46。identifier 与 month 查询在 sqlite 上始终 1.0 |

### 2.4 剩余限制

- 判据"search@30000 个位数毫秒"未达标（sqlite 中位 15.9 ms，只有时间门查询 5.3 ms）：固定的 O(pool) 部分 6–8 ms（水合每原子约 19 µs：pydantic 校验 2 次 + JSON 列 + 链接查询；信号每原子约 7 µs）；FTS5 的 bm25 排序随匹配文档数线性（12000 个匹配 9 ms）；实体门的 G 枚举随链接数线性（3000 条约 4–6 ms）。进入个位数毫秒需要 `model_construct` 水合（改变 §8 读路径的校验语义）与不带 `atom_id` 次序键的 FTS 排序（失去确定性）——本轮都未做。
- 判据"重叠率 ≥ 0.9"在 N ≥ 3000 未达标：漏掉的 oracle 原子都在 G 内，在模板语料里与命中原子的 keyword / entity 分完全并列，穷举的 top-20 由 hash 嵌入 0.40–0.46 之间的语义分与新近度决定；这些原子在语义通道排第 100–6000 名、bm25 通道第 150–3000 名、新近度通道第 89–491 名，任何 K=40–80 的有界通道都够不到。这是有界召回的固有损失；换用有语义区分度的嵌入器（`fastembed` / `openai`）会直接改变该结论，本轮按要求固定 `hash@64`。
- SQLite 规划器：不收集 `ANALYZE` 统计（收集后无门控查询反而变慢 4–5 倍，ENABLE_STAT4 未编译），改由 mvmi 按级别门控指名索引。
- Milvus：milvus-lite 的 RPC 固定开销与过滤式 HNSW 的召回缺口如 1.1 所述；远程 Milvus 服务器未测。
- `scoring.max_candidates` 仍存在于配置字典（兼容），只是不再生效。

## 3. 同期落地的其它模块（对三份审计的回应）

这些模块不在两轮"短板"清单里，但已在仓库中并有测试；README 范围表第 1 / 2 / 4 / 5 / 11 / 15 / 18 行与 CONTRACT §0.5 已据此改写。

| 模块 | 回应的审计发现 | 验证 | 无基准 / 限制 |
|---|---|---|---|
| `hms_elite/telemetry.py`（`Usage / UsageMeter / MeteredProvider / trace / usage_of`）+ `GET /usage` / `GET /metrics` | COST_AUDIT F1（无 token / usage 计量）、DESIGN_PARITY §3（无 request_id、无 /metrics） | `tests/test_telemetry.py` 12 条（计量累加与重置、rule 估算、openai / anthropic 精确 usage、trace 收集阶段、嵌入失败计数、并发不丢计数、线程局部）；`tests/test_server.py::test_usage_and_metrics`、`test_request_id_round_trip` | 审计表未做（DESIGN_PARITY §3 的 `audit_log`） |
| `Ingestor(extraction_mode='single')`（默认）+ `memory_extraction` 复合提示词 | COST_AUDIT F3（有修订候选或行动闭包时 2–3 次串行 LLM 往返，Mem0 恒 1 次） | `tests/test_extraction_single.py` 20 条（`test_single_mode_makes_exactly_one_llm_call_per_add`、与分段三件套结果相同、scripted 决策退化为 ADD、非对象回复不崩溃、EP 同一次调用抽取、rule LLM 与三个分段处理器一致） | token 数未重跑 COST_AUDIT §1.1 的表 |
| `deepseek` / `ollama` LLM，`ollama` / `fastembed` 嵌入器 | DESIGN_PARITY §7（provider 广度：LLM 4 / embedder 2；中文用户群缺 DeepSeek / Ollama） | `tests/test_providers_extra.py` 8 条（注册表与 extras、DeepSeek 缺 SDK 提示、请求体与 thinking 开关、工具调用参数解析、Ollama 请求、Ollama 嵌入维数推断与缓存、fastembed 缺 SDK 提示、本地嵌入） | 都是替身 SDK / 替身 HTTP 的离线测试；未对真实服务计时 |
| `daemon.py::StateDaemon` | 范围表第 1 行 | `tests/test_daemon.py` 6 条（含前台 add / search 与后台 consolidate 的竞争） | — |
| `runtime.py::AgentRuntime / ActionRegistry` | 范围表第 2 行 | `tests/test_runtime.py` 7 条 | — |
| `segmentation.py` | 范围表第 4 行 | `tests/test_segmentation.py` 5 条 | 规则切分 |
| `compiler/coverage.py` + `scc.py` | 范围表第 5 行 | `tests/test_coverage_scc.py` 7 条（含 S4 先接口后精化） | 可选层，`compile()` 默认不启用 |
| `state/`（`StateAPI`、personal / edge / group / enterprise） | 范围表第 11 行 | `tests/test_state.py` 23 条 | — |
| `eval/`（`run_ablation`、九个 §10 任务、LongMemEval 适配器） | 范围表第 15 行 | `tests/test_ablation.py` 20 条（1 条无数据集时跳过） | 数据集不随包分发 |
| `clone_schema` / 文件库先迁移后开 WAL / `checkpoint_on_close` | SPEED_AUDIT §3 init、§5 close | `tests/test_perf_paths.py` 13 条；BENCH §2 判据 4：文件库 `apply_migrations` 2.13 ms → `clone_schema` 1.13 ms，内存库 1.41 → 0.28 ms | SPEED_AUDIT §8 第 1–4、7 项（add 往返数、REVISE 重复读、pydantic 重校验、水合 `model_construct`、逐 id 查询批量化）未做 |

## 4. 未关闭的短板（供下一轮）

| 来源 | 项 | 状态 |
|---|---|---|
| COST_AUDIT F2 | add() 的修订候选查询水合 O(N) 个 pydantic 原子（2000 次 add 后每次 add 1,078 次构造） | 未做；不在 mvmi 重写范围内 |
| COST_AUDIT F5 / F6 | 存储 3.0–3.5 × Mem0；`keep_witnessed=False` 回收 0 锚点 | 未做（tombstone 方案未采用） |
| COST_AUDIT F7 | consolidate `llm` 策略每 batch 重发全部节点与见证（32 EP → 224,854 输入 token） | 未做；默认 `rule` 策略 0 次调用 |
| DESIGN_PARITY §1 / §2 / §3 / §5 / §6 | 多租户守卫、jobs 表 + runner、audit 表、`raw_ref` 索引与按来源撤销、进程内 webhook 订阅 | 未做 |
| SPEED_AUDIT §8 | add 往返 26 → 16、REVISE 38 → 27、读路径 `model_construct`、批量 id 查询 | 未做 |
| RETRIEVAL_BENCH §2 | search@30000 个位数毫秒、N ≥ 3000 重叠率 ≥ 0.9 | 未达标，原因见 2.4 |
| hms4eye 能力表 | `compare_with_upstream()`：68 个 operationId 中上游 33 个 no-op，本引擎点亮 13 个（5 native + 8 adapted），仍不支持 20 个，另放弃上游 24 个服务层能力 | 本轮检索改造不改变任何 operationId 的判定，`hms_elite/hms4eye/manifest.py` 未改 |
