# HMS Elite — 设计定稿（Final Design）
English version: [DESIGN.md](DESIGN.md)

> 基线：`proposal_hms_invariants`（获胜方案，3/3 评审通过）。移植进来的部分：lineage id 与 Mem0 config/签名对齐（`proposal_mem0_parity`），结构化闭包构建器、查询规划器、场景测试套件、全量状态检查点（`proposal_scenario_mvp`）。评审列出的每一条阻断级/重大缺陷都在 §3 中关闭。规范性文档是同目录下的 `CONTRACT.md`；本文解释的是*为什么*。两者冲突时，以 `CONTRACT.md` 为准。

---

## 1. 总纲（Thesis）

HMS Elite 是一个 Python 包（`hms_elite`，PyPI 名 `hms-elite`），它对外披着 Mem0 v2.0.19 的皮——相同的签名、相同的信封结构、相同的过滤器文法、相同的 config dict、相同的错误形态、相同的解析阶梯、相同的工厂风格——皮下是一套 HMS 双平面骨架：

```
axes find            State Index (memory_atoms + 7 axis tables)   hot path: add/search/get_all
graph inherits       Support Graph (graph_nodes/edges/witnesses/interface_eps/projections)   inheritance path: why/compile(deep)/consolidate
ContextIR acts       StateCompiler → ContextIR → Work State Card / PC-IR / StatePackage
experience trains    transitions table: (S_t, a_t, S_{t+1}, r_t, κ_t, w_t) per EP / InterfaceEP / causal / revision / conflict / rollback (uniform κ_t)
```

十条工程承诺，每一条都由代码强制执行，而不是写在文档里：

1. **无 witness 即无意义（No Witness, No Meaning）——三层防线。** pydantic 层（`Witnessed.anchor_ids: list[str]`，带 `min_length=1`）；DDL 层（observations、eps、memory_atoms、graph_nodes、graph_edges、interface_eps、feedback_events、history、checkpoints、transitions 上的主 witness 列声明为 `NOT NULL REFERENCES anchors(anchor_id)`）；以及带守卫的写入器（`SupportGraph.insert_edge` / `StateStore.insert_edge` 接收一个普通 dict，在构造 pydantic 模型之前、在任何 SQL 之前，就对空 witness 列表抛出 `WitnessMissingError`；每个接受 anchor 的写入器都先用 `get_anchors` 预检 id；原始 cursor 是私有的）。Anchor 可以被撤销（`HMS.revoke`），并级联到派生的 atom / edge / InterfaceEP。
2. **Atom 不存边——由类型和导入拓扑共同保证。** `MemoryAtom` 没有 edge 字段（`extra='forbid'`，另有一个 validator 拒绝 `content` 内出现 `edges/edge_ids/src_id/dst_id` 键）；`hms_elite/mvmi/*` 到 `hms_elite/graph/*` 不存在导入路径；`search()` 永远不触碰 `graph_*` 表（由 `store.trace` 断言）。获胜提案里那条 `instr()` 字符串 CHECK 被删掉了（对值 `"edges"` 会误报）。
3. **一个 atom 版本对应一个节点，`node_id == atom_id`。** 每个 atom（message、observation、event、EP、InterfaceEP、skill、procedural）都由 `insert_atom` 写入且只写入一行 `graph_nodes`。这消除了 atom 节点与 EP 节点并存带来的歧义，也让 `why()`、回滚和反馈共用一个 id 空间。
4. **稳定的 Mem0 id 就是 lineage id。** `memory_atoms.memory_id`（即 Mem0 的 `id`）加 `version`；`atom_id` 是版本 id。`update(id, text=)` → 归档 head，在同一 lineage 内插入 version+1，并写一条带 witness 的 `revision` 边；`get(id)` 返回 head 的*新*文本；`history(id)` 覆盖整条 lineage；`add()` 的 REVISE 路径会并入旧 lineage 并上报 `event='UPDATE'`。局部唯一索引 `ux_atoms_head`（`WHERE archived_reason IS NULL`）保证每条 lineage 至多只有一个未归档版本——一个即将过期的 head（`valid_to` 在未来）仍然是 head，所以带 `expiration_date` 的记忆用 `get()` 依然能取到。
5. **用修订替代删除，且由结构强制。** 每张证据表/状态表上都有 `BEFORE DELETE` 触发器；`BEFORE UPDATE OF <content columns>` 触发器使 atom / EP / observation / edge / InterfaceEP 的内容不可变；只有 `valid_to / confidence / visibility / user_meta_json / updated_at / archived_reason / valid / invalidated_* / resolved_*` 允许变更。`delete()` = `valid_to=now, archived_reason='deleted'` 加一行带 witness 的 `DELETE` history。
6. **每一次变更本身也是一个事件。** `update/delete/delete_all/feedback/checkpoint/rollback/consolidate` 都先写一个 `SourceEvent`（`object_type` ∈ `system|feedback|checkpoint|rollback|scheduler`）加一个 `Anchor`；它们创建的每一行都由该 anchor 作 witness。数据库里没有任何一行缺少通往原始证据的链条。
7. **高层对象是接口，不是摘要——由 schema 保证。** `interface_eps.primary_support_node_id NOT NULL`，`support_node_ids` 的 `min_length=2`，`Slots` 是 EP 与 InterfaceEP 共用的同一个 pydantic 模型（同构文法），并且**只有边界（投影）边**携带 `witness_edge_id`（`CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL)`）；`AFTER INSERT` 触发器为它们写入 `edge_projections`。高层的 `revision`/`distill` 边只由 anchor 作 witness——获胜提案那条过宽的 CHECK（要求每条非 L1 边都必须是投影）已被修正。
8. **结构化事件永不调用 LLM，并且会闭合成真正的 EP。** `tool_call`/`file_diff`/`test_log` 会按「动作→结果」闭合（P3 闭包构建器）生成一个 `(u,b,a,y,r,κ)` EP，由两侧 anchor 共同作 witness；负向 EP 会铸出一个派生的 `constraint` atom，并用 `distill` 边相连；`use` 边指向背景约束；`causal`/`temporal` 边连接相邻的 EP。调用完 `add_events()` 后，Work State Card 的 *Failed Paths* 与 *Next Action* 立刻可达，零 LLM 调用。
9. **锁纪律：Txn A → 无锁抽取 → Txn B。** 证据（事件 + anchor）先提交；LLM / embedding 调用在不持锁的状态下运行；物化（atom、M2M、向量、节点、边、history、transitions）后提交。A 与 B 之间崩溃可由 `consolidate(phases=['collect'])` 恢复。
10. **离线确定性是一项正确性特性。** `rule` LLM 加 `hash` embedder，再加 `StateStore` 上可注入的 `now()`/`id_factory`，使每个测试都能逐字节复现；六个黄金场景（§12）在两种 index provider 上都做逐字节断言。

---

## 2. Lite 裁剪（What is cut / simplified, and why）

| # | 完整 HMS 特性 | Lite 决策 | 理由 |
|---|---|---|---|
| 1 | `daemon.py` 后台循环 | 裁剪 → `consolidate()` 同步执行 `orient/collect/consolidate/prune` | 既定决策 9（仅同步） |
| 2 | `runtime.py` agent 适配器 / 动作注册表 | 裁剪 | `add_events()` 就是适配器契约 |
| 3 | 各来源专用适配器（slack/jira/feishu/sensor） | 简化 → 一个 `SourceEvent`，带 5 种可抽取的 `object_type`（`message, tool_call, file_diff, test_log, sensor_trace`） | 属于传输胶水层，不是不变量 |
| 4 | `segmentation.py` | 简化 → 在 `rule` 抽取器内部做句子切分 | 需要调好的模型 |
| 5 | `coverage.py`/`scc.py` 层次化覆盖编译器 | 简化 → 按 slot 分配份额预算，并设约束下限 | 属于精度特性，不是不变量 |
| 6 | 视觉 / 感知 | 简化 → 列表型内容降解为文本片段，图片片段以 sha256/mime（`parts`）落锚并记录 `dropped_parts`；`llm.enable_vision` + 提供者的 `describe_image` 钩子（无锁、事务 A 之前）把描述并入正文并记录 `described_parts` | 离线环境没有感知能力；视觉是提供者能力 |
| 7 | FTS5/BM25 词法轴 | 拆分 → 召回顺序来自标准库 SQLite 的 FTS5 虚拟表 `atom_fts`（bm25，`keyword_candidates`；`capabilities['fts5']` 为 False 时退回 `LIKE`）；池内关键词*分数*仍是纯 Python token 重叠，标识符 token 权重 ×2 | 零可选依赖；有界召回 |
| 8 | Reranker provider | 简化 → `overlap`（默认，离线加权词元覆盖率）+ `llm`（`rerank` 提示词契约，`LLM_002` 时回退 overlap）；`search(rerank=True)` 在融合后重排，各行追加 `rerank_score`，失败时保持检索顺序；`retrieval['rerank']` 保持 `None`（compile 从不重排） | Mem0 kwarg 对齐，外加一个离线默认 |
| 9 | 按主题的半衰期学习器 | 简化 → 单一 `scoring.half_life_days` | 确定性 |
| 10 | embedding 层的实体消解（cosine ≥ 0.95） | 简化 → 归一化文本精确匹配加别名表；`run_id` 排除在实体身份之外 | hash embedding 不具语义；S3 需要跨 run 聚合 |
| 11 | Personal/Edge/Group 渲染器 | 裁剪；只保留 Work State Card 加 `StatePackage` | 一个确定性渲染器足以证明契约 |
| 12 | LLM coarsen | 简化 → 默认 `rule` coarsen；LLM coarsen 走同一份 prompt 契约，引用需对 V 校验 | 可复现的往返测试 |
| 13 | 技能提升 | 简化 → 一个 kind 为 `constraint`、支撑节点 ≥2 的 InterfaceEP 不变式，会变成一个 `L3_macro` skill atom，加一条可回溯到底层 `use` 边的 `distill` 边（`witness_edge_id` + `edge_projections`） | 单次 run 内即可达成（修掉获胜方案的两次 run 缺口） |
| 14 | 面向 κ 的治理/ACL 评估器 | 简化 → 在选择阶段施加可见性格（lattice）门限（过滤器、图邻居、不变式、证据 anchor），加 `governance`/`not_revoked` 义务 | 为策略引擎预留挂载点 |
| 15 | 基准与消融实验框架 | 裁剪；HMS 第 5 阶段的消融配置映射到 `CompileMode`：`fast` = 仅 Axis，`axis_graph` = Axis + Graph，`deep` = Axis + Graph + Support；「仅 Graph」不适用（图只能从轴的种子进入）；`retrieval['ablation']` 记录实际运行的配置 | 包的范围所限 |
| 16 | HTTP 服务端 | 已发布为 `hms_elite/server`（`create_app()` / `hms-server`，可选 extra `server`，H 组）；`cli.py` 照常发布 | 库优先；服务面只包装门面 |
| 17 | 外部 index provider（Qdrant/FAISS） | 简化；`IndexBase` 可插拔，随包提供 `sqlite` 与 `memory`；`milvus`（milvus-lite 文件或服务器）放在可选 extra `milvus` 之后，经 `rank_hits` 重算分数以保证逐位一致 | 既定决策 3（核心只依赖 pydantic） |
| 18 | 异步孪生实现、遥测、通知、`setup_config()` | 裁剪 | 既定决策 9 |
| 19 | Mem0 的 `chat()` / `project` | `chat()` 抛 `NotImplementedError`；`project.update()` 抛 `ValueError` | 让 Mem0 API 形态的冒烟测试保持通过 |
| 20 | Mem0 `history.db` 导入 | 裁剪（改为 `hms_elite.migrate.from_mem0(memory, hms, *, filters)`，遍历 `get_all` 并以 `infer=False, timestamp=created_at` 重新写入） | 20 行代码，不需要做列集合差异迁移 |

---

## 3. 评审意见的采纳（Judge verdicts → design changes）

### 3.1 移植项（三位评审一致要求）

| 移植项 | 来源 | 落点 |
|---|---|---|
| lineage id `memory_id` + `version`、`ux_atoms_lineage_version`、局部索引 `ux_atoms_head`、`store.resolve_head()`、`update()` 返回带同一 id 的 `{'message'}`、`history()` 覆盖整条 lineage、`add()` 返回行携带 `revises`/`conflicts_with` | P1 §A.2/§C/§D/§E.1 | CONTRACT §4 `MemoryAtom`、§7 DDL、§8 store、§17 信封 |
| 与 Mem0 逐字节兼容的 config：`vector_store` 是 index 信封的键，`history_db_path` → `store.path`，`version ∈ {'v1.1','v2','lite-1'}`，`reranker` 是注册过的信封（`overlap` | `llm`），RENAMED/REMOVED 表放在同一个 `mode='before'` validator 里；`ingest.revision_mode: inline|deferred|off` | P1 §H.1、§K.1 | CONTRACT §5 |
| `_UNSET` 哨兵值、`data=` 别名、`expiration_date` → `valid_to`（次日 UTC 午夜）、`reference_date` → `as_of`、接受 `timestamp=`、`show_expired`；`MemoryBase(ABC)`、`chat()`、`project`；`inspect.signature` 对齐测试；`RESERVED_ATOM_KEYS == 列集合` 测试 | P1 §E.1、§I | CONTRACT §17、§18 |
| 程序性分支（`agent_id and memory_type=='procedural_memory'`）→ `procedural_summary` 任务 → 生成一个 `L1_micro` episode atom（绝不生成 L2） | P1 §F.1.5 / §G.6 加评审修正 | CONTRACT §6.3、§12.3、§12.6 |
| `LLMBase.generate_response(messages, tools=None, tool_choice='auto', **kwargs)`；用 `patch.dict(sys.modules, {'openai': None})` 做导入守卫测试 | P1 §H.2、测试 25 | CONTRACT §10、§18 |
| `migrate.from_mem0()` 辅助函数 | P1 §K.12 | CONTRACT §17.5 |
| 过滤器 DSL 的解析顺序：轴 → 集合轴 → `json_extract(user_meta_json)`；`meta.<key>` 逃生通道；`AxisQuery.meta_filters`；参数化的一致性测试 | P1 §E.1 | CONTRACT §13.1 |
| 闭包构建器（`open_actions[run]`、test_log 闭合最近一个未闭合动作、错误的 tool_call 自闭合、`u` = intent 或上一个 EP 的 `y`、派生约束加 `distill`、`use` 指向背景约束） | P3 §F.2 | CONTRACT §12.4 |
| `plan_query()` 的时间线索（中英月份、ISO、相对时间）→ 硬性 `time_anchor BETWEEN` 门限，带一次回退重试；关键词中标识符 token ×2；实体路由带硬实体门限 | P3 §F.3.1 | CONTRACT §13.2 |
| `ux_entities_norm_scope(name_norm, user_id, agent_id)` —— `run_id` 被排除 | P3 §D | CONTRACT §7 |
| `tests/test_scenarios.py`，六个黄金场景在 `['memory','sqlite']` 上参数化；`why()` 中确定性的 `chain`/`explanation` | P3 §I/§F.5 | CONTRACT §18、§19 |
| 预算份额，带 `min_constraints=3` | P3 §F.4 | CONTRACT §15.2 |
| `why()` 的提问模式（`target_id` 不是 id 且带 `filters` → 编译侦探模式 → 约束/情节/事实，优先匹配标识符实体） | P3 §F.5 | CONTRACT §14.3、§17.3 |
| 全量状态检查点（`atom_state`、`edge_state`、`iep_state` 覆盖 scope 内每一个对象）与对称的回滚 | P3 §C/§F.7 | CONTRACT §16.2 |
| `ScoringConfig` 独立成一个子配置；`collect` 阶段只重新路由那些没有 atom 的可抽取事件 | P3 §H、§F.9 加评审修正 | CONTRACT §5、§16.5 |
| `score_details` = Mem0 的 7 个键，加 `recency_score, support_score, keyword_backend, axis_gates, primary_axis, route, signals`，再加 `recalled_by, pool_size, channel_sizes`（该行的召回通道、池规模、各通道规模）；`max_possible` 由激活的信号决定，而非由命中决定 | P1 §H.2/§F.3 | CONTRACT §13.4 |

### 3.2 获胜方案的缺陷及其修复

| 缺陷（评审指出） | 修复 |
|---|---|
| `CHECK (scale='L1_micro' OR witness_edge_id IS NOT NULL)` 强迫每条非 L1 边都是投影；distill/revision 边被迫伪造投影行 | CHECK 改为 `is_boundary = 0 OR witness_edge_id IS NOT NULL`；只有在 `witness_edge_id` 被设置时触发器才写 `edge_projections`；`project()` 是 `is_boundary=1` 边的唯一写入者；高层 `revision`/`distill` 边只带 anchor。`graph_stats.projection_coverage` = 至少有 1 行投影的边界边 / 全部边界边 |
| `update()` 返回了新 id；`get(old)` 拿到陈旧数据；history 被割裂 | 改用 lineage id（见上文移植项） |
| 在 `content_json` 上使用 `instr()` CHECK | 删除；只保留 pydantic validator |
| `ux_events_identity` 包含 `timestamp`，导致「幂等重复摄入」依赖时钟 | **有意保留**（不同时间的两句话就是两个事件——One Event, One Anchor），测试显式传入 `timestamp=`，因此不依赖时钟；在相同事件时间上完全相同的重复 add 会去重（`dedup=True`，不新建 anchor，atom 决策为 `NONE`，`graph_nodes.mentions` 自增；`atom_anchor` 行保持不变，因为被复用的 anchor 本来就是主 witness）；相同文本但事件时间*更晚*，则会创建新事件加新 anchor，已有 atom 增加一行 `support` 类型的 `atom_anchor` |
| `history()` 返回行带了约 4 个未记录的额外键 | 返回行是 Mem0 的 10 个键**外加恰好一个新增键 `anchor_id`**（已记录的对齐差异）；HMS 独有的事件（`ARCHIVE/RESTORE/FEEDBACK/CONFLICT/ROLLBACK`）已写入文档；`atom_id/edge_id/details_json` 留在表里，不出现在返回行 |
| cosine 被映射为 `(cos+1)/2` | 改为 `max(0.0, cos)`（Mem0 契约）；`score_details.semantic_score` 由此具备可比性 |
| 技能提升需要 ≥2 个 InterfaceEP（即两次 run） | 单个 InterfaceEP 内有 ≥2 个支撑节点的不变式即可（即 S4 中被两个 EP 共用的那条约束） |
| 「switched to」谓词歧义 → REVISE 不可达 | 取代类线索统一映射到谓词 `likes` 并带 `relation='supersession'`；修订规则在 `relation=='supersession'` 加同一主语加主题/宾语重叠时触发（按谓词族匹配，而非谓词相等） |
| `IndexConfig` 里塞了打分旋钮；`half_life_days` 在 `SchedulerConfig` 中重复 | 由 `ScoringConfig` 持有权重/半衰期/候选上限；index 信封只保留 `{provider, config}` |
| 实体键包含 `run_id` | 实体键排除 `run_id`；FILE 类标识符以文件名（basename）为键，完整路径作为别名 |
| `meta.<key>` 过滤器需要 JSON1 且无回退 | 提供 `capabilities()['json1']`；JSON1 缺失时 `plan_filters()` 把 AST 拆成 SQL 部分与由 Python 求值的剩余部分 |
| `ingest.py` 中的 `create_atom` 被 `graph/support_graph.py` 调用 → 循环导入 | atom 物化移到 `hms_elite/ingest/atoms.py`（C 组）；内联边改用 `store.insert_edge`（A 组）；`graph/*`（E 组）导入 C，反向永不成立 |
| 结构化事件各自变成一个 EP（`file_diff` 的 `y=''`），导致 *Failed Paths* 出错 | 闭包构建器（见移植项） |
| `HMSConfig.version` 的 Literal `'lite-1'` 加 `extra='forbid'` 拒绝了原生 Mem0 dict；各 provider 配置拒绝 `top_p/top_k/openai_base_url` | 改为 `version ∈ {'v1.1','v2','lite-1'}`；`BaseLlmConfig(model, temperature, api_key, max_tokens, top_p, top_k, http_client_proxies)`、`OpenAILlmConfig(+openai_base_url)`、`AnthropicLlmConfig(+anthropic_base_url)`；未知键仍然报错，并给出 `difflib` 提示 |
| LLM 修订决策跑在 Txn B 内部 | `decide_revision()` 在 Txn A 与 Txn B 之间以无锁方式运行；Txn B 会重新校验目标仍然有效（否则降级为 ADD） |
| Prompt 字段 `supersedes_existing_ids`/`uses_recent_ep_ids` 与模型字段不一致 | prompt 契约与模型字段现在共用同一套名字：`supersedes_hint_ids`、`uses_ep_ids`（引用由解析器解析） |
| 治理义务方向反了 | 一个条目可以展示给受众 `a`，当且仅当 `rank(item.visibility) >= rank(a)`（格序 private<team<event<public）；受众缺失时该义务为 `skipped` |
| `derive_goal` 与示例卡片不一致 | `compile()`：goal = 查询本身；`handoff(goal=None)`：goal = 最新有效 EP 的 `u` |
| slot 规则 `scale_level == 'L3_macro' → skills` 误捕了 L3 的 InterfaceEP 情节 | 规则改为仅 `source_kind == 'skill'` |
| `transitions(fmt='records')` 默认值与既定决策 2 冲突 | 默认改为 `fmt='tuples'` |
| 反馈表在 `state_wrong`/`privacy_violation` 行留有未完成的散文 | 该表重写（CONTRACT §16.1）；冲突消解带 witness（只要设置了 `resolved_at`，`resolved_by_anchor_id NOT NULL`；`resolution_edge_id` 可选） |
| `extract_entities` 漏掉句首人名（「Sarah joined…」） | 句首的首字母大写 token 一律视为 PROPER，除非落在 `STOP_CAPS` 中（显式停用表，含月份/星期/代词/常见动词） |
| update/delete 事件滥用了 `object_type='feedback'` | 改为 `object_type='system'` 加 `event_type ∈ {updated, deleted}`；`add_events()` 拒绝内部 object type（`VALIDATION_007`） |

### 3.3 其他提案中不得继承的缺陷

- atom 上不设 `superseded_by` 列（P3）：后继版本通过 `revision` 边 / lineage 版本号查找。
- 不设 `deleted_at` 这个孪生标志位（P3）：在唯一的有效性轴上用 `archived_reason='deleted'`。
- 卡片中不允许 `event` atom 与 `episode` atom 重复（P3）：主 anchor 已被选中的 episode 引用的 event atom 会被抑制（`explain.reason='covered_by_episode'`）。
- 排序前不设「最新 60 条」候选上限（P3），也不再有任何最新优先截断：候选来自有界多通道召回入池（语义 / 关键词 / 实体 / 新近度，各由索引支撑、各有自己的 K），池内全量打分；装得下池的门控集合整体入池（CONTRACT §13.3、RETRIEVAL_DIAGNOSIS §3–§5）。`scoring.max_candidates` 是不截断任何东西的已弃用别名。
- 不为单一实现引入 `GraphBase`/`GraphFactory` 这类 provider 抽象（P1）：`SupportGraph` 就是 `StateStore` 之上的一个普通类（重排器后来有了第二个实现 `overlap` + `llm`，因此有 `RerankerBase` + `RerankerFactory`）。
- 检索期不从图的边计算 support 项（P1）：`support = tanh(n_undisputed_anchors / 3)`，只取自 `atom_anchor`。
- 不做「只有边才产生 transition」（P1）：`ep` 行在摄入时就写入。
- 不在 `BEGIN IMMEDIATE` 内使用 `executescript`（P1）：迁移逐条语句执行。
- 不 monkeypatch `models.new_id`（P1）：id 与时钟从 store 上注入。
- 不在调用方 metadata 里塞 `_hms_pin`（P1）：置顶用 `atom_topic='important'`，`user_meta` 保持不动。

---

## 4. 包布局（Package layout，按属主分组）

扁平布局，根目录为 `/Users/lazy/Documents/Claude Code/hms-elite`：`hms_elite/` 与 `tests/`、`examples/`、`pyproject.toml` 平级（hatchling，核心依赖只有 `pydantic>=2.7`，extras 为 `numpy/openai/anthropic/milvus/spacy/server/test/dev`，ruff 行宽 120，isort 采用 black 风格）。八个属主分组，依赖顺序无环：

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

导入规则（由 `test_import_topology` 测试）：`A ← B ← C ← D`？不是——D 只依赖 A、B。确切的边为：`B→A`、`C→A,B`、`D→A,B`、`E→A,B,C`、`F→A,B,D,E`、`G→A,B,C,D,E,F`、`H→all`。`mvmi/*` 不得导入 `graph/*`、`compiler/*`、`ops/*`。逐文件的完整目录树见 CONTRACT §1。

---

## 5. 数据模型与不变量（Data model & invariants）

- **两个平面装在同一个 SQLite 文件里**（`~/.hms_elite/hms.db`、`$HMS_DIR`、`':memory:'`）：证据面（`source_events`、`anchors`），观测/状态面（`observations`、`eps`），State Index（`memory_atoms`、`atom_anchor`、`entities`、`atom_entity`、`atom_topic`、`atom_vectors`），Support Graph（`graph_nodes`、`graph_edges`、`edge_witnesses`、`edge_projections`、`interface_eps`、`interface_ep_support`），运维面（`feedback_events`、`history`、`checkpoints`、`transitions`、`meta`）。用 `PRAGMA user_version` 做迁移，`threading.Lock` 加显式 `BEGIN IMMEDIATE/COMMIT/ROLLBACK`，`check_same_thread=False`，文件型 DB 启用 WAL。
- **Id**：形如 `<prefix>_<12 hex>`，由 `store.new_id(prefix)` 生成；前缀有 `evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst`。测试注入按前缀计数的生成器（如 `atm_000001`）。`memory_id`（即 Mem0 的 `id`）使用前缀 `mem`。
- **时钟**：`store.now()` → 定宽 UTC ISO 字符串（`isoformat(timespec='microseconds')`，`+00:00`），使 TEXT 排序等于时间排序。每个 atom 有三个时刻：`time_anchor`（事件时间，`temporal_fact` 则为解析出的日期）、`created_at`（摄入时间）、有效期窗口 `valid_from/valid_to`（`valid_from` 取事件时间，以便回填导入正常工作；`as_of` 默认为 `now()`）。
- **Scope**：`user_id/agent_id/run_id` 在每张平面表上都做反范式冗余（至少需要 1 个，由 CHECK 约束）。
- **七条轴**：时间（`time_anchor`）、有效性（`valid_from/valid_to` 加 `as_of`）、实体（`atom_entity` ⋈ `entities`）、主题（`atom_topic`）、角色、情感、尺度（`scale_level`）。`AxisQuery(filters, project_axis, scale_constraint, aggregation, limit, as_of, include_expired, scope?, text, meta_filters, plan)` 真正执行 HMS §3.5 的语义：`project_axis ∈ {atom, value, time, entity, topic}` 产生一个投影，`aggregation ∈ {list, latest, count, summary, topk}` 在服务端施加，结果是一个 `AxisResult(hits, projection, count, summary)`；对外暴露为 `HMS.axis_query()`。`entity` atom（每个实体一个，由 consolidate 物化）与 `relation` atom（比较关系）使全部七个 `scale` 取值都可查询。
- **枚举**：严格按既定决策 6，另加 `Role, Sentiment, Slot, ObjectType, EventType, NodeKind, SourceKind, Decision, HistoryEvent, Route, CompileMode, Relation, EntityType, MemoryType`——全部为 `(str, Enum)`。
- **不变量 → 测试**：(a) witness 在类型层/DDL 层/store 层——见证集是*存在*条件（只要还剩 ≥1 个未争议、未撤销的锚点对象就有效；`anchor_ids[0]` 只是存储层主见证，不是必要条件），投影边另需 ≥1 个有效投影来源：决定记录见 `WITNESS_SEMANTICS.md`；(b) atom 不携带边，加导入拓扑，加 trace；(c) `InterfaceEP` 要求 ≥2 个支撑节点、`primary_support_node_id NOT NULL`、边界边 ⇒ 必有 `edge_projections`；`refine(project(coarsen(S)))` 往返；(d) `refine()` 只由 `why()` 和 `compile(mode='deep'|'auto' 且发生扩展)` 调用；`mode='fast'` 不触碰任何图表。

---

## 6. 算法（Algorithms，规范性细节见 CONTRACT §12–§16）

### 6.1 `add()`
1. 校验（Mem0 规则，`VALIDATION_001/002/003`，id 去空白，保留的 metadata 键被剥离并给出一次告警）。消息做类型强制；`system` 角色的事件会建立 anchor，但绝不参与抽取。
2. **Txn A**：每条消息一个 `SourceEvent(object_type='message')` 加一个 `Anchor`（按 `(payload_hash, timestamp, scope)` 去重）。
3. `infer=False` → 生成原始 `message` atom（role/actor_id 被提升为字段，实体仍然建立关联）。`infer=True` 的观察 atom 也带 `actor_id` = 消息 `name`，否则 LLM 点名的 `speaker`（CONTRACT §12.5）——这是有意偏离 Mem0（Mem0 只在 infer=False 路径设置 `actor_id`）。程序性分支 → `procedural_summary` 任务 → 生成一个 `L1_micro` episode atom。
4. 无锁阶段：Phase-0 上下文（最近 10 个事件，前 10 个有效邻居编号为 `'0'..'n'`）、`observation_extraction` LLM 调用、对路由为 `'state'|'both'` 的消息做 `ep_extraction`、embedding 计算，以及对每个 observation 执行 `decide_revision()`（候选集为同主语、且谓词/主题/宾语有重叠的有效状态 atom；`revision_decision` LLM 任务；rule provider 实现 CONTRACT §11.3 中的表）。
5. **Txn B**：插入 observation/EP 行 → 构建 atom → 应用决策（`REVISE`：归档旧版，新 atom 并入 lineage 的 v+1，写一条由新 anchor 作 witness 的 `revision` 边 旧→新，history 记 `UPDATE`；`CONFLICT`：新建 lineage，写 `conflict` 边，history 记 `CONFLICT`；`NONE`：已有 atom 增加一行 `support` anchor，`mentions++`）→ 为 chat EP 内联写入低成本边（`temporal`，以及来自 `uses_ep_ids` 的 `use`）→ 每个 EP 写一条 `ep` transition → history 记 `ADD`。
6. 返回 `{'results': [...]}`。

### 6.2 `add_events()` —— 结构化闭包（不调用 LLM）
事件按 `(timestamp, index)` 排序；Txn A 为全部事件建立 anchor；逐事件处理：总是生成一个 `event` atom；`tool_call/file_diff` 压入 `open_actions[run]`（错误的 tool_call 自闭合为负向 EP）；`test_log` 弹出最近一个未闭合动作（或跨批次调用 `store.last_open_action(scope, run_id=run, before=outcome.timestamp, max_gap_hours=ingest.closure_max_gap_hours)` —— 即时间间隔内最新且尚未被引用的 `tool_call`/`file_diff`），并构建一个 EP：`u` = 动作的 `intent` | 上一个 EP 的 `y` | `modify {path}`；`b` = 背景约束文本 ∪ 该动作 payload 自带的 `constraints`；`a` = `patch {path}[: summary]` / `{tool}({args})`；`y` = `{suite} {status}[: failed_tests]`；`r` 由 status 决定；`κ.constraints` = 同一个有序并集；anchor 为 `[outcome, action]`。`sensor_trace` 事件只产出一个 `event` atom。边：`use` 从 EP 指向每个背景约束（witness 为动作 anchor）；`temporal` 从前一个指向当前；`causal` 从前一个指向当前，当且仅当前一个为负向、或二者共享某个实体、或 `u` 派生自前一个的 `y`；负向 EP → 派生约束 atom `Do not re-apply '{a}': {y}` 加一条 `distill` 边（witness 为 outcome anchor）。没有未闭合动作的 `test_log` 只产出它自己的 event atom（没有动作就不产生 EP）。批次结束时，剩余未闭合的 `tool_call` 仅在其 payload 携带 `constraints`（闭合条件）时才自闭合为 `r=unknown` 的 EP；否则保持未闭合，未闭合的 `file_diff` 同样保持未闭合。

### 6.3 `search()`
`parse_filters`（标量 `visibility` 过滤器即受众，并降解为格门限 `rank(item) >= rank(audience)`）→ `plan_query`（路由、时间线索、实体线索、标识符；偏好类问题走独立的 `preference` 路由并带硬性 `obs_type` 门限，于是 HMS §4.1 的偏好召回变成一次轴选择）→ SQL 轴门限（scope、有效性、过滤器 AST、尺度、硬时间门限、实体路由的硬实体门限、偏好路由的硬类型门限；四级回退阶梯：先去掉路由的默认尺度 → `primary_axis='scale_fallback'`，再去掉 plan 门限 → `'fallback'`；显式的 `scale_constraint` 永不放宽）→ 有界多通道召回入池（语义 ANN / FTS5 关键词 / 实体 / 新近度，各有自己的 K，每条通道都在门限集合内搜索；装得下池的门限集合整体入池；CONTRACT §13.3、RETRIEVAL_BENCH_2026-09-10）→ 在池上计算信号：语义 `max(0,cos)`、关键词重叠（标识符 ×2）、带 hub 抑制的实体重叠、新近度 `exp(-ln2·age/half_life)`、支撑度 `tanh(n_anchors/3)` → 融合 `Σw·s / Σw over active signals` → 融合**之后**再对 `final` 施加阈值 → 按 `(final DESC, time_anchor DESC, atom_id ASC)` 排序 → 返回行加 `score`（以及 `score_details`）。全过程绝不读取任何图表。

### 6.4 `compile()`
规划 → 轴候选（`candidate_k=50`，阈值 0；走与 `search()` 相同的有界池召回，`ContextIR.retrieval` 带池统计）→ 当路由 ∈ {work_state, detective}、或 `mode='deep'`、或（`mode='auto'` 且该 scope 内至少有 1 个有效 EP）时，进行工作状态增强（纳入 scope 内所有有效的 episode 与 constraint atom）→ 工作状态增强同时并入有效的 `skill` atom（并且 `L3_macro` 在召回尺度的默认集合中，因此蒸馏出的技能可被复用）→ 在 `{causal, revision, conflict, use, distill}` 上做 1 跳图扩展（`fast` 模式下绝不进行；在 `axis_graph` 与 `deep` 中进行），并带上修订前驱 → `audit(reason='revised')`，记录未消解的冲突，为不变式精化（refine）InterfaceEP 节点（`hops=0`，仅 `deep` —— `axis_graph` 是不含 Support 的消融配置）→ 得出约束与残差 → 审计；受众门限在选择阶段就丢弃私有的邻居/不变式/证据，因此团队交接不会因自身输出而触发 `governance` → 有序的 slot 规则（已归档或在 `as_of` 时点已过期 → 进入 audit —— 一个 `expiration_date` 在未来的存活 atom 仍留在它自然所属的 slot；skill → skills；constraint → constraints；preference/habit → preferences；episode → episodes；其余 → facts；被选中 episode 覆盖的 event atom 被丢弃；entity atom 永不进入卡片）→ 只有当没有同文本的约束被选中或归档时，才生成合成 κ 条目（同一文本在所有 episode 中只出现一次）→ goal = 查询本身 → next action 规则（最新负向 vs 最新正向 episode、残差、冲突、回退）→ 预算（份额分配，`min_constraints=3`，facts 溢出到 episodes）→ 产出 `ContextIR`（validator：条目 id 互不相交，每个条目都有 witness）。整个过程只读。

### 6.5 `why()`
解析 `memory_id | atom_id | ep_id | obs_id | iep_id | edge_id`；否则若带 `filters`，则按问题处理（编译 deep/detective 模式；在约束 → 情节 → 事实的顺序中，优先选择实体键与问题中标识符 token 相交的条目，例如 `retry.ts`，否则取第一个）。在入边 `{causal, use, distill, revision, conflict}` 加出边 `use` 上做 `hops` 层反向 BFS；InterfaceEP 节点按需精化；返回节点、边（含 witness）、anchor、interface_eps、路径、未消解冲突，以及确定性的 `chain`（置信度最高的入边优先）与 `explanation`。

### 6.6 `feedback()`
序幕阶段写入反馈事件加 anchor、`feedback_events`、history 记 `FEEDBACK`；随后每种 `FeedbackType` 走各自的处理器（CONTRACT §16.1）。`state_missing` 不需要已有目标（`target_id=None` 加 `correction` 中的一个 scope，或一个 scope 键字符串）。只有 `privacy_violation` 会改动可见性；`evidence_wrong` 只是对 anchor 提出争议（绝不删除）；失去全部无争议 witness 的 atom 会被归档，而不会变成无 witness 状态。同类型边上的 `connection_reason_wrong` 只做重新标注加重新 witness（否则 MERGE 会把刚刚失效的那一行又复活）。`HMS.revoke(anchor_id)`（或带 `correction={'revoke': True}` 的 `evidence_wrong`）设置 `anchors.revoked_at` 并级联：派生 atom 失去该 witness（全部失去则归档），witness 全部被撤销的边被置为失效，证据全部被撤销的 InterfaceEP 被置为失效——即 HMS §3.2 的 Revoke。

### 6.7 `checkpoint()` / `rollback()`
检查点为 scope 内每个对象快照 `{atom_id: {valid_to, archived_reason}}`、`{edge_id: valid}`、`{iep_id: valid}`、`{ep_id: valid_to}`（EP 有效性是被维护的：归档一个 episode atom 会级联到 `eps.valid_to`，因此 `list_eps`、`last_ep`、交接目标和 orient 永远不会复活一个已删除的 EP）。回滚（单个事务，由一个 `rollback` 事件作 witness）：归档检查点之后创建的 atom；恢复当时有效的 atom；把边与 InterfaceEP 的有效性还原到快照；插入一条从后继指向被恢复对象的 `revision` 边，标注为 `rollback:<ckp>`；upsert 一条 `rollback` transition。

### 6.8 `transitions()`
`ep` 行自摄入时就存在（`Transition.from_ep`，A 组）；导出时追加 `interface_ep` 行（HMS §4.4：InterfaceEP 也是 transition 的来源）、`causal`（有效的）、`revision`/`conflict`（不论有效性，但排除带 `rollback:` 标签的）以及 `rollback` 行；`κ_t` 对所有种类只有一种形态（`constraints, visibility, permission_scope, provenance`）；`w_t` 是有序并集（按首次出现顺序）；对 `(source_kind, source_id)` 幂等；默认 `fmt='tuples'`。

### 6.9 `consolidate()`
写调度器事件加 anchor；`orient`（只读计划，含 `last_compiled`），`collect`（补算缺失的向量，重新路由缺少 atom 的可抽取事件），`consolidate`（按实体物化 entity atom、时间边、因果候选（rule 函数从 `llms/rule.py` 导入，或走 LLM 的 `edge_proposal` 任务）、冲突检测、按 run 选出 coarsen 候选 → `coarsen` → `project`（L2），随后对 L2 的 IEP 连通分量做 L3 `coarsen` → `project`、技能提升、transition 导出，以及 `compile_state` —— 为每个 scope 在 `compiled_states` 中持久化一份 Work State 快照），`prune`（衰减 = 对陈旧 atom 逐级下调置信度，随后归档低于 `confidence_min` 的；置顶的、有冲突的、以及 constraint/skill/entity 类一律不动）。

### 6.10 `axis_query()` / `revoke()` / `import_package()`
`axis_query(AxisQuery)` 执行带投影与聚合的七轴查询（HMS §3.5）。`revoke(anchor_id)` 是 HMS §3.2 的 Revoke 路径，含级联。`import_package(StatePackage)` 是 HMS §9.4 的接收端：package 携带自己的 `source_events`、`anchors`、`atoms`、`eps`、`interface_eps`、`edges`，接收端将其重建（保留 id、按身份去重、可选重新 scope 化），从而在接收端调用 `compile()` 能渲染出同一张卡片。

### 6.11 `context()` —— 混合读（2026-09-18）
`context(query, *, filters, top_k=None, as_of=None, budget_tokens=None) -> dict`（CONTRACT §6.7）返回 `"<会话卡片>\n\nRecent relevant memories:\n<记忆行>"`：卡片是 `compile(<范围内最新可抽取事件的文本>, as_of=<其时间戳>, budget_tokens=compiler.context_session_budget_tokens)`，按范围记忆化，由存储的按范围写入版本使其失效（`StateStore.scope_version`：在每个存入了该范围事件或原子的事务 COMMIT 时、无范围写入时、回滚与 reset 时推进——绝不是计时器）；记忆行是 `search(query, top_k=compiler.context_rows_top_k)` 去重（精确重复 + 真子串，大小写 / 空白不敏感）、按 `time_anchor` 升序并渲染为 `- [YYYY-MM-DD HH:MM] 文本`（`hms_elite/compiler/context.py`，即 `benchmarks/egolife/reader_ab.py` 所导入的函数）。产品把文本放在问题之前（`Memories:\n<text>\n\n<question>`）：EgoLife 上这一读法答对 70.1%，纯记忆行 54.9%、逐题卡片 63.2%，且记忆化的前缀正是提供者提示词缓存能命中的部分（`docs/design/READER_AB_2026-09-18.md`）。与 `AgentRuntime.context()` 无关。

---

## 7. Coarsen / Project / Refine

对于一个节点集合 `S`（同一 scope 下的有效 L1 节点）：`E_in` 为 `S` 内部的有效边，`E_b` 为恰好一个端点落在 `S` 内的有效边。`cohesion(S) = 0.5·min(1, |E_in|/max(1,|S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|)`，当 `|E_in|=0` 时为 `0`。`coarsen(S)`（rule 实现，`rule-coarsen-v1`；L2 的输入是 EP 节点，L3 的输入是 InterfaceEP 节点——一个算子，两种尺度）：slot 取 `u=first.u`，`b=' ; '.join(ordered_union(split(b_i, ' ; ')..., constraints))`（每条约束文本只出现一次），`a=' → '.join(a_i)`，`y=last non-empty y`，`r=negative if any member is negative and the last member is not positive, else last.r`，`κ=union`，`κ.evidence = evidence_anchor_ids`；不变式 = 出现在 ≥2 个成员 κ 中（或出现在全部成员中）的约束字符串，每条都补齐 `support_node_ids`、`anchor_ids`（支撑 anchor 的并集）和 `confidence`（支撑置信度的均值）；残差 = 未消解的冲突、缺失的结果、低内聚（同样补齐 `anchor_ids`）；`evidence_anchor_ids = ∪anchors(S) ∪ witnesses(E_in)`（会校验二者相等）。结果持久化为目标尺度上的一个 `episode` atom（并附带节点）、`interface_eps`、`interface_ep_support`。`project(iep)` 对每个 `(outer', type)` MERGE 出一条高层边，带 `is_boundary=1, witness_edge_id=<low edge>`；触发器写入 `edge_projections`。`refine(iep_id, hops)` 是只读的，也是打开高层对象的唯一途径。往返定律：`set(refine(project(coarsen(S)))['boundary_edges']) == E_b(S)`。

---

## 8. 评分公式（Scoring）

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
`score_details` 映射到 Mem0 的键：`semantic_score, bm25_score(=keyword), entity_boost, raw_score(=Σw·s), max_possible_score(=Σw active), final_score, threshold`，再加 `recency_score, support_score, keyword_backend='overlap', axis_gates, primary_axis, route, signals`，以及 `recalled_by`（召回该行的通道：整个门控集合都打分时为 `['gate']`，否则是 `semantic / keyword / entity / recency` 的子集）、`pool_size`、`channel_sizes`。

衰减（prune）：`staleness = (1−recency)·(1−clamp(confidence−0.6,0,0.4))·(1−use_recency)`；当 `staleness > 0.8` 时下调置信度（`confidence − 0.1·staleness`，下限 `confidence_min/2`）；此后若 `confidence < 0.4` 且不属于置顶/有冲突/constraint/skill/entity，则归档——HMS §3.9 的两个操作（衰减、归档）保持分离。

---

## 9. Prompt 契约（Prompt contracts）

`hms_elite/configs/prompts.py` 中有六个任务；每个 system prompt 的第一行是 `HMS_TASK: <task>`；分节为 `# ROLE / # INPUTS / # GUIDELINES / # EXAMPLES / # OUTPUT FORMAT`；构建器输出 `## Section` 块，其中的结构化输入以 JSON 代码围栏包裹，便于 `rule` provider 反向解析。真实 id 从不出现（使用 `'0'..'n'`、`n0..`、`w0..`）；未知引用被丢弃（witness 优先）。契约（精确的 JSON 见 CONTRACT §6）：`observation_extraction → {"observations":[...]}`，`ep_extraction → {"eps":[...]}`，`revision_decision → {"decisions":[...]}`（候选携带结构化字段，因此无需二次解析），`edge_proposal → {"edges":[...]}`（只接受类型化枚举，必须给出 witness 引用），`coarsen → {"interface_ep":{slots, slot_support, invariants, residuals}}`，`procedural_summary → markdown`（Mem0 结构）。上下文渲染是确定性的（不经过 prompt）。解析阶梯照搬 Mem0，并带顶层数组抢救逻辑；传输层错误 → `LLMError(...) from e`。

紧凑输出契约（`ingest.extraction_output = 'compact'`，CONTRACT §6.5）。输出 token 是写入成本与写入延迟的主项：真实 EgoLife 运行中，每批 25 个窗口让 deepseek-flash 的输出 token 中位数达到 5,562（上限 8,192；22% 的批次触顶并被对半重提），全部是 JSON，而 17 个观察键里有九个只是复述 `text`；输出单价约为缓存未命中输入的 4 倍、缓存命中输入的 200 倍。紧凑任务（`observation_extraction_compact`、`memory_extraction_compact`；两个提示词都低于完整孪生的 0.75）只要求九个键——id、event_idx、text（第三人称、≤ 25 词）、type、speaker、可选的 supersedes_hint_ids / when / valid_to，以及可选的 `entities` 提示（陈述点名的事物，0–6 个短字符串）——其余由 `hms_elite/ingest/derive.py` 从文本、被见证消息与配置的实体抽取器补齐，复用规则 LLM 自己的词表（`utils.text.sentiment`，`llms.rule.statement_triple`——建立在 R1–R9 线索表上的第三人称陈述解析器，从文本读回 subject / predicate / objects——以及 `observation_topics`、`mention_texts`），两条路径因此不会分叉；`entity_mentions` 取 `entities` 提示、抽取器在文本中的命中与说话人三者的并集（LLM 顺序在前，不分大小写去重），因此实体轴绝不比完整模式更窄——只有 LLM 才知道的提及（`python`）仍留在轴上；每个推导值仍经过完整契约的规整。EP 与决策保持原形状，只在触发条件出现时输出。默认为 `full`，`compact` 可选。它在 2026-09-11 当天曾为默认：最终提示词在同 18 批真实 API 冒烟（`benchmarks/egolife/results/real/elite_compact_entities_smoke5_r2`）中每次请求输出 token 中位 2,793 vs 5,631、触顶 0 vs 8 次，204 题零 API 回放 A/B 的 hit@20 保持 161 = 161（`results/replay_ab_entities`）——但在真实 204 题运行中 compact 初版记忆行 53.9%（下文的按下标配对 NONE 裁决），措辞重写加裁决守卫后的修订版 56.4%（115/204；卡片 62.7%，hit@20 77.9% / 79.4%，空闲价 ¥4.35，每批 7.3 s），而 `full` 为 61.3%（125/204；卡片 61.8%，hit@20 78.9%，¥11.56，14.9 s），低 4.9 个点、超出 2 个点容忍，默认于同日晚回退（`docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`，CONTRACT §6.5 / 变更记录第 105 行）。`compact` 仍是成本敏感批量灌入（写入费用 0.38 倍、每批耗时 0.49 倍）与状态卡片阅读器的推荐选项；两种情况下提示词都逐字节不变。裁决守卫（2026-09-11）：紧凑提示词旧的 DECISIONS 措辞让 deepseek-flash 把观察 i 与候选 i 配对并回答 NONE（7,315 条观察中 1,685 条；`full` 下 8,470 条中 23 条），每个 NONE 都丢掉该陈述；提示词现按每条新观察一个决策陈述、NONE 只留给复述，库只在两原子主语一致、且（对 NONE 或未被佐证的 REVISE）去掉头部（主语、共同谓词）后的内容词元包含度不低于 `ingest.verdict_min_similarity`（0.5）并至少共享两个词元或较短陈述被整体包含时采纳裁决——第一版把主语计入，只共享说话人和一个动词的短陈述恰以 0.5 越线（回放记录的紧凑回复：hit@20 151/204；去头部守卫达到 154 = 强制 ADD 的上限，MRR 0.5038，存活原子 5,717 → 7,299；记录的 `full` 回复保持 161，28 个过守卫的裁决中 3 个降级；CONTRACT §6.3 / §12.6，变更记录第 103–104 行）。

行式输出契约（`ingest.extraction_output = 'lines'`，CONTRACT §6.6，2026-09-12）。紧凑契约把输出减半却没有弥合差距，而费用核算指明了差距所在：4,851 窗口的真实运行中，`full` 的写入费用 ¥11.5 含 2.40M 输出 token——占费用的 83%——而 mem0 为 ¥1.85（输入未命中 0.93M token + 输出 0.22M）；compact v2 ¥4.2、输出 0.76M。输入未命中侧已经持平，因此只有输出 token 能撼动账单，而 JSON 信封的开销是其承载文本的 2–4 倍。另有两项测量定下了设计：零 API 回放表明检索只用到陈述文本与决策（entities / type / when 及其它字段对 hit@20 的影响为 0）；紧凑提示词自己写出的文本低于完整提示词（记忆行 56.4% 对 61.3%；配对 32 对 22）——损失在写文本的规则，不在被删的字段。因此行式任务逐字保留完整提示词的陈述规则（第三人称、5–80 词、一条自足陈述、保留数量 / 日期 / 时刻、取代线索列表、问句不产出）与紧凑 DECISIONS 规则，只把 OUTPUT FORMAT 换成 tab 分隔的 `OBS` / `EP` / `DEC` 行：`<event_idx>⇥<T>⇥<陈述>[⇥@日期]`、`<event_idx>⇥u=⇥a=⇥y=⇥r=[⇥k=]`、`<OBS 行号>⇥REVISE|CONFLICT|NONE⇥<候选>⇥<理由>`（DEC 行只写非 ADD 裁决；缺失即 ADD）。不用 json 模式（`LLMBase.generate_lines` 不发送 `response_format`），容错解析器（`ingest/lines.py`：围栏、散文、缺 tab、未知字母、被截断的末行、越界的 DEC 各自至多丢一行），解析之后走不变的紧凑推导路径——唯一的行式专属规则是该格式没有说话人槽位，陈述开头的专有名即说话人（实体轴、`actor_id`）。仅 single 模式；规则 LLM 渲染同样的行，因此六个聊天夹具上轴 / 历史 / 图 / `eps` 各表以及原子的文本、主语、谓词、宾语与 `full` 相同，`observations` / `memory_atoms` 行只在钉住的推导列上不同（`confidence` 0.7、没有提示槽位的 `relation` / `supersedes_hint_ids`、该 `actor_id`）；完整与紧凑提示词逐字节不变；2026-09-18 起 `lines` 是库默认（CONTRACT 变更记录第 108 行；此前为 `full`），见下一段。

测量结果（2026-09-12，`results/real/elite_lines_full204`，204 题全量，deepseek-flash，空闲档）：记忆行上下文 54.9%（112/204），状态卡片上下文 63.2%（129/204，所有臂中最高）；hit@20 84.3% / 82.4%（最高）；每窗口输出 51 token（full 495、compact 157、mem0 45），每批模型耗时中位 3.76 s（full 14.9、compact 7.3、mem0 3.5），写入 + 回答全按空闲价 ¥2.24（mem0 ¥1.98、compact ¥4.35、full ¥11.56）；存入 12,994 条原子（2.68 条/窗口，库 72.5 MB）；判定 ADD 12,963 / NONE 20 / UPDATE 11；零截断、零丢失。结论：成本与写入速度达到 mem0 水平且召回更高；记忆行上下文的准确率低于 full 4 个点（有据答错 72 题），状态卡片上下文高于 full 1.4 个点。2026-09-18 起默认为 `lines`（第 108 行）：同一行式存储上的阅读端 A/B（`docs/design/READER_AB_2026-09-18.md` §六）测得 §6.11 的混合读（`HMS.context()`：记忆化会话卡片前缀 + 带时间戳记忆行，记忆在前）70.1%（143/204），full 的记忆行 61.3% / 卡片 61.8%，与记忆行的符号检验 p < 0.0001；`full` / `compact` 仍可选，`staged` 配置现在须显式写出其一。

---

## 10. Provider 层

`LLMBase(config)` 提供 `generate_response(messages, tools=None, tool_choice='auto', **kwargs) -> str | dict` 和 `generate_json(task, system, user)`；provider 有 `rule`（默认）、`scripted`（测试用）、`openai`、`anthropic`（带导入守卫 → 抛 `DependencyError` 并给出 pip 安装提示）。`EmbedderBase` 提供 `embed/embed_batch/dims/identity`；实现有 `hash`（默认）与 `openai`。`IndexBase` 提供 `upsert/upsert_batch/search(candidate_ids)/get/delete/count/reset/info`；实现有 `sqlite`（BLOB 存在同一个 DB 里，可导入 numpy 时使用 numpy）与 `memory`。`EntityExtractorBase` 提供 `extract(text) -> list[EntityCandidate]`；实现有 `rule`（默认，与 `utils.text.extract_entities` 逐字节一致）与 `spacy`（可选 `[spacy]` extra；NER 区间并入规则结果，重叠时规则区间胜出，因此只有正则能抓到的 FILE/IDENTIFIER 候选得以保留）；由 `ingest.entity_extractor` 选择，每个门面一个实例，摄取器、规则 LLM（`entity_mentions`）、反馈纠正与交接包导入共用。`configs/registry.py` 中的 `REGISTRY` 是信封校验与工厂两者的唯一真源；`register_provider()` 同时扩展二者。配置类都是 `extra='forbid'` 的 pydantic 模型，并带 `difflib` 的「你是不是想输入」提示。

---

## 11. 测试计划（Test plan）

只有一个 `tests/conftest.py`（提供 `clock`、`ids`、`store`、`rule_llm`、`scripted_llm`、`hash_embedder`、`hms`、`hms_scripted`、`events`、`golden`、`index_provider`，以及 autouse 的 `no_home_writes`）。夹具是 JSON 事件列表（`tests/fixtures/events/*.json`）与黄金输出（`tests/fixtures/golden/*`）。测试文件与属主分组（A–H）一一对应，另加 `test_scenarios.py`（S1–S6，在两种 index provider 上运行）——共 63 个具名测试，外加评审后新增的用例（`45b` L3 coarsen、`54b` 交接导入往返、`56b` revoke 级联、`56c` entity atom、`64` 双语 docstring），见 CONTRACT §18，每个用例都注明它覆盖的公开函数。两个黄金比对辅助：`golden.text`（逐字节：卡片、chain）与 `golden.masked`（掩掉 id 与时钟纪元时刻：JSON/JSONL）。沿用自 Mem0 的规则：mock provider SDK，绝不 mock 被测代码；每修一个 bug 加一个回归测试；只用 `monkeypatch.setenv`；不做时序断言；不联网。

---

## 12. 演示 `examples/quickstart.py`

依次运行 S1（咖啡偏好 → 修订）、S2/S3（跨 run 的 Sarah、「what happened in May」）、S4（约束加上失败的 `file_diff`/`test_log` 配对 → 生成带 *Failed Paths* 和 `Do not re-apply …` *Next Action* 的 Work State Card；场景测试还会再补上修复配对，并断言第二张卡片的 *Next Action* 为 `Continue from …`）、S5（`why("why must retry.ts not be modified")`）、S6（检查点 → `state_outdated` 反馈 → 一次**不带** `as_of` 的 `compile()` —— 后继版本只从反馈那一刻起有效，所以场景中段的卡片必须依赖时钟 —— → 回滚 → transitions），然后执行 `consolidate()` 与 `graph_stats()`。全离线，约 45 行；卡片使用显式 `as_of`，证据行携带夹具时间戳加 payload sha256 前缀，因此在真实时钟下打印出的卡片等于 `tests/fixtures/golden/work_state_card_failed.txt`。精确的输入与预期输出见 CONTRACT §19。

---

## 13. HMS → Lite → Mem0 映射表

| HMS 概念（§） | HMS Elite 对象 / 调用 | Mem0 对应物 | 差异 |
|---|---|---|---|
| SourceEvent (§3.1) | `SourceEvent`、`source_events`、`add_events()` | `messages` 缓冲区 + `save_messages` | 提升为一等表，5 种可抽取 object type（含 `sensor_trace`），`timestamp=` 被尊重 |
| Anchor (§3.2) | `Anchor`、`anchors`、`atom_anchor`、`edge_witnesses` | payload `hash`（md5）+ history 行 | 改用 sha256 事件哈希；每一行都引用 ≥1 个 anchor |
| Observation (§3.3) | `Observation` → `state|event|relation` atom | `ADDITIVE_EXTRACTION_PROMPT` 的记忆文本 | 带类型、带日期、带 witness；`linked_memory_ids` 被当作 `supersedes_hint_ids` 消费 |
| EP (§3.4) | `EP`、`eps`、闭包构建器、`ep_extraction` | 程序性记忆字符串 | 每个闭包一个六元组；结构化事件不调用 LLM |
| MemoryAtom / AxisQuery (§3.5) | `MemoryAtom`、`memory_atoms` + 轴表、`AxisQuery`、`FilterExpr` | 扁平 payload + `_process_metadata_filters` | 有效性进入 WHERE，实体/主题为 M2M，lineage id 让 Mem0 的 `id` 保持稳定 |
| Support Graph (§3.6–3.7) | `graph_*`、`SupportGraph`、`InterfaceEP`、coarsen/project/refine | （v2 中已移除；实体链接） | 六种类型化边、witness、投影 |
| ContextIR / PC-IR (§3.8) | `compile()`、`ContextIR.to_text()/to_pcir()` | 结果格式化 | 六个互不相交的 slot、义务、`ProofObligationError` |
| StateScheduler (§3.9) | `consolidate(phases=)` | Phase 0–8 的 add 流水线 | 同步四阶段，由一个 scheduler anchor 作 witness |
| StateFeedback (§3.10) | `feedback(target, type, reason, correction)` | `update()/delete()` | 11 个类型化处理器、revision 边、`FEEDBACK` history |
| Transition (§4.4) | `transitions()` | `history()` 重放 | `(S_t,a_t,S_{t+1},r_t,κ_t,w_t)`，来自 EP、InterfaceEP、边和回滚；κ_t 统一形态 |
| Handoff package (§9.4) | `handoff()` → `StatePackage`；接收端调用 `import_package()` | — | 接收端导入的是状态（事件、anchor、atom、EP、IEP、边），而不是 history |
| Revoke (§3.2) | `revoke(anchor_id)` | `delete()` | 级联到派生 atom/edge/IEP；证据行永不删除 |
| 混合读（CONTRACT §6.7） | `context()` → 记忆化会话卡片 + 带时间戳记忆行，记忆在前 | 调用方自行拼装的 `search()` 行 | 按范围记忆、由存储写入版本失效；`compiler.context_*` 字段；EgoLife 上 70.1% 对记忆行 54.9% |
| add/get/get_all/search/update/delete/delete_all/history/reset/close | 同名、同签名、同信封 | v2.0.19 | `add` 可能上报 `UPDATE`/`NONE`；额外提升的键；`history` 多一个 `anchor_id` |
| config dict | `HMSConfig` 接受 `vector_store/llm/embedder/history_db_path/version/custom_instructions/reranker` | `MemoryConfig` | 另加 `store/ingest/scoring/graph/compiler/scheduler`；RENAMED/REMOVED 表 |

---

## 14. 风险与未决问题（Risks & open questions）

1. **规则抽取器的天花板** —— 正则分类是针对中英线索词表和夹具调过的；真实散文更多是漏检（不产生 EP），而不是误产生 EP，这是有意的设计取向。真实 provider 驱动完全相同的 prompt；`RuleLlmConfig.language` 只用于选择词表。
2. **hash embedding 是词法的** —— 语义召回依赖字符重叠；排序主要由轴门限、关键词和实体信号承担。切换到 `openai` embedding 需要执行 `consolidate(phases=['collect'])` 重新计算向量。
3. **内聚度公式与按 run 划分的分量**可能对长 run 过度粗化；`coarsen_max_nodes=12` 给出上界；L3 粗化（IEP 的 IEP）在 consolidate 的步骤 (4b) 中对连通的 L2 InterfaceEP 执行，并用同一 scope 下的两个 run 做测试（`test_coarsen_l3_from_two_interface_eps`）。
4. **闭包规则是贪心的** —— 一个 `test_log` 闭合最近一个未闭合动作；交错编辑会把结果归因到最后一个动作。接缝在 `ingest/closure.py::build_ep_from_closure`。
5. **回滚是 scope 局部的** —— 另一端点落在 scope 之外的边，会随快照一并失效并被报告。
6. **治理是一个格，不是策略引擎** —— 受众门限在选择阶段施加（过滤器、图邻居、不变式、证据 anchor），而 `governance` 义务只重新检查 κ 的遗漏与 anchor 权限；anchor 的 `permission_scope` 会被查阅，撤销会级联，但没有 ACL 评估器。
7. **SQLite 下限** —— 带 `COALESCE` 的表达式唯一索引、局部索引、`RAISE` 触发器、`json_extract`（JSON1）需要 SQLite ≥ 3.31（Python 3.10 在所有受支持平台上都捆绑 ≥ 3.31）；`capabilities()` 记录 `json1`，规划器在缺失时降级为 Python 求值。
8. **History 词汇表** —— HMS 独有的事件可能让 `if row['event'] == ...` 这类代码意外落空；已在迁移表中记录。
9. **`why()` 的提问模式复用了 `target_id`** —— 为遵守既定签名而保留；`resolved_from` 把这一点显式化。
10. **Token 估算** `(utf8_len+3)//4 + 8` 对中日韩文本较粗糙（约 0.75 token/字）；用于预算是可接受的。
11. **并发** —— 每个 DB 文件一个进程（WAL + `BEGIN IMMEDIATE` + 进程内锁）；已写入文档。
12. **事件身份键中包含 `timestamp`** —— 在更晚的挂钟时间重复添加同一条消息会创建第二个事件（有意为之）；期待静默去重的 Mem0 用户，得到的是一个 `NONE` 决策外加一个额外的 witness。
