# 从 HMS 初衷看 HMS Elite（2026-09-22）

来源与前提。初衷取自 `HMS_arch.md`（2026-05-10 版，1326 行）；Elite 设计定稿取英文 `DESIGN.md` 与 `CONTRACT.md`（两者冲突以 CONTRACT 为准，DESIGN.md 首段）；代码现状取自 `hms_elite/`（`find -name '*.py'` 排除 `__pycache__` 得 108 个文件，其中 `__init__.py` 18 个；`wc -l` 合计 56,805 行）与 `tests/`（`pytest --collect-only` 783 项；本文未重跑全量，引用的测试均以源码位置为准）；实测数字取自 `docs/design/` 下的 EgoLife 报告与 `benchmarks/egolife/results/` 下的原始记录，其中回放库计数（EP、transitions、边）来自对 `hms.db` 的直接 SQL 查询。三条前提：HMS_arch.md 是意图不是记录，全文无一个测量数字；DESIGN.md §2 的裁剪表有 7 行已被 CONTRACT.md §0.5 改为 SHIPPED 但未同步；全部准确率来自 EgoLife A1 的 204 道四选一题，单次运行，无置信区间（audit_2026-09-20/README.md 注意事项 1）。本文按英文定稿逐条对照 Elite 的设计思想，不另写英文版。

英文定稿本身的来路要先记下，因为后面多处对照依赖它。DESIGN.md 首段写明：基线是 `proposal_hms_invariants`（三位评审 3/3 判胜），从 `proposal_mem0_parity` 嫁接 lineage id 与 Mem0 配置/签名一致性，从 `proposal_scenario_mvp` 嫁接结构化闭合构造器、查询规划器、场景套件与全状态检查点。DESIGN.md §3 把评审裁决作为一个过程记录：§3.1 嫁接表（16 项）、§3.2 胜出方案的缺陷修复表（23 项）、§3.3 不得从其它方案继承的缺陷（10 项，例如不设 `superseded_by` 列、不设 `deleted_at` 双标志、不在排序前做 60 条最新截断、不从图边取检索时的 support 项）。本文正文引用的 lineage id、`ux_events_identity` 保留 `timestamp`、supersession 谓词、单 IEP 技能提升，都出自 §3.1 或 §3.2，是评审裁决的结果，不是原始提案。

## 一、当初为什么这样设计（本心）

**Experience-to-State。** 总纲一句：HMS 把行动、判断、失败、约束和结果"编译成下一次行动可以继承、调用、纠偏、训练和审计的状态"（HMS_arch.md 第 10 行）。紧接着的否定句是本心的边界："它的核心不是'保存更多历史'"（第 12 行）。目标表示是六元组 (S_t, a_t, S_{t+1}, r_t, κ_t, w_t)（第 15 行）：行动前后状态、动作、结果、约束、证据锚点。没有动作和结果，这个表示就退化成普通记忆。

**为什么从分形案例塔改成双平面。** 早期是 Micro/Meso/Macro 的 3D-EP Stack，作者自评"这个直觉是对的，但需要精确化"（第 55 行）。收紧后的三条：EP 结构可以跨尺度同构（第 68 行）；图结构不要求严格自相似（第 69 行）；跨层关系改为"可追溯"，高层节点须有低层 support，高层边须有低层 witness（第 70 行）。重构的直接动机是两种需求混在一张图上。一类是定位问题，原文一句："这条问题应该查哪个时间、哪个实体、哪个主题、哪个 scale？"（第 80 行）。另一类是继承问题，原文是五个独立问句，其中三句："这个高层状态由哪些低层证据支撑？""发生冲突时如何下钻？""回滚到哪里？"（第 90、92、93 行）。文档没有给出旧设计在哪个场景失败的证据，这是设计判断。

**每个平面解决的痛点。** State Index"不存边，只存每个原子在多个轴上的坐标"（第 83 行），解决定位；目标是把 preference/temporal/multi-session 从"相似度检索"改成"轴选择"（第 770 行）。Support Graph"不追求查询最快，而追求证据、结构、回滚、蒸馏和解释"（第 97 行），核心判断是"高层 EP 是低层子图的接口，不是摘要"（第 416 行）。ContextIR 把候选原子和支持图编译成下一步状态，输出 Work State 卡，含 Invalidated Assumptions 与 Failed Paths（第 518–537 行）。压成一句："轴负责找，图负责继承，ContextIR 负责行动"（第 49 行）。两个平面之下是 Anchor 层："One Event, One Anchor"、"No Witness, No Meaning"（第 214–215 行），服务 Verify/Revoke/Rollback/Audit。分流规则决定什么进 State Layer："只有被行动引用、引发结果、参与修订或冲突时"（第 267 行）；"Preference 不是 EP"（第 265 行）。

**三条容易被略过的原则。** 第一，输入层："不直接把 raw data 当记忆。先把输入规范化为 SourceEvent。"（第 181–184 行）。这是锚层能成立的前提：没有规范化事件，就没有一事一锚。第二，边的语法：同层 EP 图与跨层投影都只用六类边 temporal / causal / revision / conflict / use / distill（第 473–478 行），跨层投影三规则是"内部边被 InterfaceEP 吸收。边界边投影为高层边。每条高层边必须能回溯到低层 witness edge。"（第 484–486 行）。第三，产品映射：§11.3 Enterprise State 把 Slack / 飞书 / Teams / Zendesk / Jira / CRM / 会议纪要作为输入，输出 CustomerIssueState / IncidentState / MeetingDecisionState / AccountHandoffState 四种状态（第 1185–1201 行）。

**八项优势各押在什么机制上**（第 1016–1101 行）。省 token 押 InterfaceEP 替代低层细节 + MVMI 直接命中，并注明"token 不是主目标，只是资源约束下的副产品"（第 1028 行）；速度押 indexed axis lookup；准确率押检索对象从 raw message 变成 atom；可交接押的是导入状态包，原文："接手方不是读历史，而是导入状态包。"（第 1065 行）；可回滚押 Anchor + EP version chain + revision edge；可审计押每个输出带 witness；可蒸馏押 InterfaceEP invariants 上升为 skill；可训练押六元组进 world model。后五项全部依赖 Support Graph 与 Anchor 的存在。

一处文内张力要先记下。§0 第 29–32 行把两类目标并列："快速定位状态"（第 31 行，普通长期记忆、偏好、时间、实体、多会话问答）与"继承行动经验"（第 32 行，handoff、rollback、detective、蒸馏、world-model transition）；§0 说的"核心"是第 12 行"不是保存更多历史，而是转成六元组"，没有说哪一类目标优先。§12 Phase 1 目标是 LongMemEval 的 preference/temporal/multi-session（第 1225 行），属于第一类。两个重心并列，谁优先没写。后面的历史基本沿着 Phase 1 走了。

## 二、逐条对照：初衷 → Elite 设计（英文定稿）→ 现在的代码 → 证据

编号约定。H-xx 是本文为 HMS_arch.md 段落起的顺序编号，只在本文内使用，不是原文编号。编号 → 原文行号：H-01 第 12 行；H-02 第 15 行；H-03 第 181–184 行；H-04 第 49 行；H-08/09/10 第 68–70 行；H-11 第 83 行；H-13 第 416 行；H-14/23/24 第 267 行；H-19/20 第 214–215 行；H-21 第 221–224 行；H-25 第 376 行；H-26/28 第 285–287、314 行；H-30/31 第 349–369 行；H-32 第 254 行；H-33 第 403 行；H-34 第 770 行；H-35–37 第 416–469 行；H-38 第 470–487 行；H-44–47 第 498–537 行；H-48–50 第 544–568 行；H-51/52 第 580–600 行；H-53–55 第 611–669 行；H-56 第 676–682 行；H-60–63 第 920–1009 行；H-64 第 1028 行；H-67 第 1065 行；H-68 第 1072 行；H-70 第 1090 行；H-71 第 676–682 行（与 H-56 同段，训练侧）；H-72 第 1185–1201 行。代码路径均相对 `hms_elite/`；证据文档名不带日期后缀时指 `docs/design/` 下同名文件。

| 初衷（一句话） | Elite 设计定稿怎么承接 | 现在的实现 | 在实测里起的作用 | 评语 |
|---|---|---|---|---|
| H-01 经验编译成状态，不是存更多历史（第 12 行） | 保留为 DESIGN.md §1 四动词 axes find / graph inherits / ContextIR acts / experience trains；新增 Mem0 v2.0.19 外皮 | 有：`memory/main.py` 门面 | 评测台只调 `add/search/compile/context`（`real_arms.py:1352`、`:1450`），走的是"存 + 找 + 编译"半边 | 代码承接了全句，数据只检验了被否定的那半句 |
| H-02 六元组是目标表示（第 15 行） | 保留并新增：`transitions` 表，每 EP/IEP/causal/revision/conflict/rollback 一行，统一 κ_t 形态（CONTRACT.md §16.4）；`ep` 行在摄取时写入，不做"只有边"的 transitions（DESIGN.md §3.3） | 有：`models.py:1216`、`ops/transitions.py:321` | 回放库有行：lines 库 eps 4 / transitions 4，r2 库 eps 2 / transitions 2（`reader_ab_full204/work/lines/hms.db`、`elite_full204_r2/work/elite/hms.db` 实查）；黄金文件 `fixtures/golden/transitions_s6.jsonl` 5 行；无消费者读过 | 有结构、有少量行、无消费者 |
| H-03 不直接把 raw data 当记忆，先规范化为 SourceEvent（第 181–184 行） | 保留：一个 `SourceEvent` 承载 5 种可抽取 `object_type`（message, tool_call, file_diff, test_log, sensor_trace），逐源适配器裁掉（CONTRACT.md §0.5 行 3）；`add()` 内部也先写事件 + 锚再抽取（承诺 6、9） | 有：`ingest/` | 4,851 窗口全部经 `SourceEvent` 入库，规整后 0 窗口丢失（HIGHLIGHTS §1.4） | 成立；它是 H-19/20 在实测里能起作用的前置条件 |
| H-04 轴找、图继承、ContextIR 行动（第 49 行） | 保留为承诺 2：`search()` 不碰 `graph_*`（`store.trace` 断言），拓扑上 `mvmi/*` 不导入 `graph/*` | 有：`test_mvmi.py:958`、`test_foundation.py:328` | 检索中位 9.6–17.7 ms、P95 21.4–28.4 ms（EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md §2）。图平面不是零调用：`compile()` 默认 `auto` 模式在 scope 内有 ≥1 条有效 EP 时做工作态增补并进入图扩展与支持层（`compiler/compile.py:508`、`:525–530`），card 臂每题调 `compile` 不传 mode（`real_arms.py:1450`），lines 回放库有 4 条 EP，第一条在 DAY2，此后大部分卡片编译走了图平面；评测记录未存 `retrieval['ablation']`，图扩展的贡献不可量化。`consolidate/why/refine` 在 `real_arms.py` 无调用 | "找"与"图"的分工在代码里是硬约束；实测里"找"有数，"图"走过但没有计量，"继承"没走 |
| H-08/09/10 同构、不自相似、可追溯（第 68–70 行） | 保留并收窄成 schema：`CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL)`，只有 `project()` 写 `is_boundary=1`（DESIGN.md §3.2） | 有：`memory/schema.py:533`、`graph/operators.py:506` | 未触发（回放库 `interface_eps` 0）。单元级消融 axis_only 0.65 / axis_graph 0.90 / +support 1.00（HIGHLIGHTS_2026-09-11.md §1.1），非 EgoLife | 可追溯被做成数据库不变量，这是收紧的最终形态 |
| H-11/33 atom 不存边（第 83、403 行） | 保留：`extra='forbid'` + validator（承诺 2） | 有：`test_foundation.py:522` | 同上，`search` 不碰图层（HIGHLIGHTS §1.1） | 成立 |
| H-19/20 一事一锚、无 witness 无意义（第 214–215 行） | 保留并三层强制：pydantic `min_length=1`、DDL `NOT NULL`、写入器预检抛 `WitnessMissingError`（承诺 1）；新增 `ux_events_identity` 含 `timestamp`，评审曾质疑其时钟依赖，定稿有意保留（DESIGN.md §3.2） | 有：`memory/schema.py:407/520`、`memory/storage.py:4908` | 零 API 回放：10,611 条观察重建库（OVERVIEW_2026-09-12.md §三.2）；读侧 A/B 290 条回复 0 次 API（READER_AB_2026-09-18.md §一） | 这是实测里回报最大的初衷。反例：锚保原话可回放，不保卡片可复现，57/204 题不同（READER_AB §五） |
| H-21 Anchor 服务 Revoke/Rollback（第 221–224 行） | 保留并加存在语义：R1 存在规则 + R2 投影来源规则（WITNESS_SEMANTICS.md）；承诺 3：每个原子版本恰一个 `graph_nodes` 行，`node_id == atom_id`，由 `insert_atom` 写入，使 `why()`、rollback、feedback 共用一个 id 空间（DESIGN.md §1 第 3 条） | 有：`ops/feedback.py:933`、`memory/storage.py:5370/5391` | 只在 condense 23 事件夹具验证，agreed 3 / only_hms 0（WITNESS_SEMANTICS.md §7）；EgoLife 无 revoke | 未在真实数据上验证 |
| H-14/23/24 Observation 与 State 分流，只有行动闭包进 State（第 267 行） | 保留：Event Router 按 `has_action_closure` 分流，或 `user_meta.route == 'state'` 强制分流（CONTRACT.md §12.2）；`has_action_closure` = 动作动词 且（结果句 或 约束句）（CONTRACT.md §9.2） | 有：`ingest/router.py:59`、`utils/text.py:141–176`、`:1088` | 本文独立复算：对 `elite_lines_full204/windows.jsonl` 逐窗口跑 `has_action_closure`，4,851 窗口 5 个命中（DAY2_10000000#0015、DAY2_13000000#0001、DAY5_16000000#0079、DAY5_17000000#0003、DAY7_18000000#0050），均为闲聊里的 fail/fixed/broke/install，无一是真实动作→结果；`zh_full204_A/windows.jsonl` 0 个 | 分流规则在生活转写上接近关掉 State Layer。DESIGN.md §14 风险 1 预告了这一点：规则抽取器按设计宁漏勿误 |
| H-26/28 EP 准入三条件；EP 可建边、可回滚（第 285–287、314 行） | 保留并新增承诺 8：结构化事件零 LLM 闭合成 EP，负向 EP 派生 constraint + distill 边；`Slots` 闭合 validator；闭合规则贪心，DESIGN.md §14 风险 4 承认交错编辑会归因错 | 有：`ingest/closure.py:456`、`test_ingest.py:481` | lines 运行 290 条回复里 4 条含 EP 段（`extract_raw_elite.jsonl`，本文核对），先前核对计 39 行 EP 候选；回放库物化 4 条 EP、4 行 transitions、`memory_atoms.source_kind='ep'` 4 行（`hms.db` 实查）；`retain_elite.jsonl` 无 `eps` 键，评测日志里看不到这个数 | 物化了 4 条，此后没有任何算子消费它们 |
| H-30/31/34 七轴、AxisQuery、轴选择替代相似度（第 349–369、770 行） | 收窄并新增：有界多通道召回（semantic/keyword/entity/recency 各 K）、等价性定理、漏检条件明写（CONTRACT.md §13.3）；DESIGN.md §14 风险 2 事先写明"哈希嵌入是词面的，轴门、关键词和实体信号承担排序" | 有：`mvmi/query.py:650/1111`；`test_mvmi.py:748` | 哈希嵌入下加轴门 10.8%→31.4%（HIGHLIGHTS §1.3），与风险 2 的预告一致；真实嵌入下语义撑主力，二元组关键词反降 84.8%→80.4%（EGOLIFE_ZH_2026-09-22.md §二）；30,000 条 3.41 ms，改前 1.6 s（RETRIEVAL_BENCH_2026-09-10.md §2） | "轴选择替代相似度"没有成立；实测形态是"相似度为主 + 轴做门限"。速度主张成立 |
| H-32/25 preference 取 latest；supersession 关系（第 254、376 行） | 保留：取代线索→`likes` + `relation='supersession'`（DESIGN.md §3.2，评审指出原谓词"switched to"使 REVISE 不可达后改的） | 有 | 未单独测；修订链极少：lines 库 revision 边 11 条（对应 `summary.json` UPDATE 11），zh_B UPDATE 22，zh_A 19，compact 5 | 数据里修订链极少，机制未被检验 |
| H-13/35–37 InterfaceEP 是接口非摘要，五组成，Coarsen/Project/Refine（第 416–469 行） | 保留为承诺 7（schema 层面：`interface_eps.primary_support_node_id NOT NULL`、`support_node_ids min_length=2`、EP 与 IEP 共用 `Slots`）；往返定律 `refine(project(coarsen(S)))`，`refine` 只被 `why` 与 `deep` 调用，L2/L3 同一算子（CONTRACT.md §14.2） | 有：`graph/operators.py:194/506/574`；`round_trip_check` 库内无调用者，只测试用 | 未触发：`real_arms.py` 无 consolidate/why/refine 调用，回放库 `interface_eps` 0 | 代码最完整、证据最少的一块 |
| H-38 六类边与投影三规则（第 470–487 行） | 保留六类边 temporal/causal/revision/conflict/use/distill 为 `graph_edges.edge_type`；投影三规则收成一条 CHECK：只有边界边带 `witness_edge_id`，`AFTER INSERT` 触发器写 `edge_projections`；评审指出胜出方案的 CHECK 过宽（所有非 L1 边都须是投影），定稿改为 `is_boundary` 判据（DESIGN.md §3.2 第 1 行） | 有：`memory/schema.py:533`、`graph/operators.py:506` | lines 回放库 `graph_edges`：revision 11、temporal 2；r2 库 temporal 1；无 causal/conflict/use/distill，无边界边 | 六类边的语法在库里成立；跨层投影没有一条真实边走过 |
| H-44–47 ContextIR 六槽、PC-IR 四义务、Work State 卡（第 498–537 行） | 保留并新增：槽位规则有序全覆盖、份额预算、证据永不封顶、`next_action` 在预算前、`context()` 混合读（CONTRACT.md §15、§6.7） | 有：`compiler/compile.py:370`、`compiler/pc_ir.py:77`、`compiler/context.py:211` | 卡片 vs 记忆行：full 61.8/61.3，lines 63.2/54.9（EGOLIFE_REAL_LLM_MEM0_VS_ELITE §1）；hybrid 70.1%（READER_AB §六）；预算 1200→400 为 64.7→53.9%（READER_AB §二） | 卡片起作用，作用来自装进更多证据，不是槽位结构（READER_AB §三.2）；Invalidated Assumptions / Failed Paths 未测 |
| H-48–50 Consolidation 异步四阶段（第 544–568 行） | 先裁剪为同步（DESIGN.md §2 行 1），后 SHIPPED `daemon.py`（CONTRACT.md §0.5 行 1） | 有：`ops/consolidation.py:179`、`daemon.py:38` | 未触发 | DESIGN §2 未随之更新 |
| H-51/52 11 种反馈；archive + revision 边，不覆盖（第 580–600 行） | 保留并硬化：`BEFORE DELETE`/`BEFORE UPDATE` 触发器（承诺 5）；每次变更自身是事件（承诺 6）；新增裁决守卫 `verdict_min_similarity`（CONTRACT.md §12.6） | 有：`ops/feedback.py:157`、`memory/schema.py:201` | `feedback()` 未触发；守卫拦截 zh_A 710 个 NONE、12 个 REVISE（`zh_full204_A/summary.json` retain.downgraded）；compact 初版 NONE 导致 22% 陈述未入库（HIGHLIGHTS §1.4 记 1,685 个 NONE；同运行 `elite_compact_full204/summary.json` 记 NONE 1595、windows_stored 4801，即另有 50 窗口整窗丢失，两个口径不同，本文引 HIGHLIGHTS） | 修订机制的实测工作面是"防误判丢数据"，不是"纠偏" |
| H-53–55 三种查询流（第 611–669 行） | 保留为 `CompileMode` fast/deep/axis_graph，消融配置写进 `retrieval['ablation']` | 有：`test_compiler.py:674` | 只有 §4.1/§4.2 路径被走；评测记录未存 `ablation` 字段 | §4.3 handoff/detective 路径无数据 |
| H-56/71 六元组进 world model（第 676–682 行） | 保留：`transitions(fmt='tuples')` 为默认（评审指出默认曾与固定决策 2 冲突，DESIGN.md §3.2） | 有 | 有行（lines 库 4 行），无训练消费者 | 尚未验证 |
| H-64 token 是副产品（第 1028 行） | 替换：成本成为默认切换的决策变量，compact/lines 的三条件规则（CONTRACT.md §6.5、§6.6） | — | lines 写入 ¥2.24 vs full ¥11.56；每窗口 51 vs 495 输出 token（EGOLIFE_REAL_LLM_MEM0_VS_ELITE §2–§5） | 权重与初衷相反。有反例记录：compact 因准确率 56.4 vs 61.3 回退 |
| H-67/68 可交接 = 导入状态包；可回滚（第 1065、1072 行） | 保留：`StatePackage` 见证闭包 validator、对称 checkpoint/rollback（CONTRACT.md §16.2、§16.3）；回滚只作用于 scope 内，跨 scope 边随快照失效并报告（DESIGN.md §14 风险 5） | 有：`ops/handoff.py:39`、`ops/checkpoint.py:61` | 未触发 | 实现了，评测没用到 |
| H-70 可蒸馏（第 1090 行） | 收窄：单 IEP 内 ≥2 支撑节点的 constraint 不变式→L3 skill（DESIGN.md §2 行 13；评审指出原方案需两个 IEP、两次运行才可达，故改） | 有：`graph/operators.py:655`；`test_facade.py:1149` | 未触发 | 尚未验证 |
| H-72 Enterprise State 产品映射（第 1185–1201 行） | 先裁后 SHIPPED：`hms_elite/state/` 含 personal / edge / group / enterprise 四族（CONTRACT.md §0.5 行 11） | 有：`state/kinds.py:43` 起 `StateKind` 枚举 12 个成员，其中企业族 4 个 | 无数据 | 有渲染器无输入源 |
| H-60–63 vs RAG / Observation Memory / Memory OS / 长上下文（第 920–1009 行） | 保留原话；新增 Mem0 逐字节兼容作为对照臂 | — | 对 mem0：hit@20 84.3 vs 79.4，延迟 10–18 vs 115–213 ms，写入 ¥2.24 vs ¥1.98，库 72.5 MB 为其 5.6 倍（EGOLIFE_REAL_LLM_MEM0_VS_ELITE §1–§4） | 只测了"看见过去"的一半；"继承行动"的差异化无数据 |
| （无初衷对应）Elite 自有：Mem0 外皮与 lineage id；锁纪律 Txn A→无锁→Txn B；离线确定性；"规整而不拒绝"；full/compact/lines 抽取契约；retention `purge`；治理格 | 新增（DESIGN.md §1 承诺 4、9、10；CONTRACT.md §12.3、§6.5–6.6、§16.6、§0.1）；治理格明言"是格不是策略引擎"，无 ACL 求值器（DESIGN.md §14 风险 6） | 有 | 规整：首轮 31% 窗口丢失→此后 0/4,851（HIGHLIGHTS §1.4；EGOLIFE_REAL_LLM_MEM0_VS_ELITE §5）；确定性反例：真实运行 id 由 `secrets.token_hex` 生成，卡片回放 57 题不同（READER_AB §五）；锁纪律、purge 无实测 | 规整是 Elite 自己的贡献；确定性在测试里成立、在真实运行里不成立 |

## 三、仍在发挥作用的想法，以及它们为什么起作用

**见证锚点与回放。** 起作用的原因不是审计，而是省钱和隔离变量。三轮 10,611 条观察零 API 回放，把 compact 掉 7 个点的根因定位到判定：判定全改 ADD 命中 141→154，其它改动全为零（OVERVIEW_2026-09-12.md §三.2）。读侧 A/B 全部建立在回放上，290 条回复 0 次 API（READER_AB §一）。这是初衷 H-19/20 在实测里的直接回报，前提是 H-03 的 SourceEvent 规范化。它的边界也清楚：原话可回放，编译输出不可复现（READER_AB §五）。

**轴优先的有界多通道召回。** 初衷说"轴负责找"。实测证明轴起作用的方式是门限而不是替代：哈希嵌入下加关键词、实体、时间门后 hit@20 从 10.8% 到 31.4%（HIGHLIGHTS §1.3），这正是 DESIGN.md §14 风险 2 预告的形态；真实嵌入下语义一路撑起大部分召回，二元组关键词权重 0.5 会挤掉语义命中，hit@20 84.8%→80.4%（EGOLIFE_ZH §二、§三.2）。Elite 新增的有界池是速度主张成立的原因：30,000 条 3.41 ms（RETRIEVAL_BENCH §2），并把漏检条件写成定理而不是隐藏（CONTRACT.md §13.3）。未验证项：真实嵌入下没有逐通道消融；本文核对 `docs/design/` 与 `benchmarks/egolife/results/`，无此实验记录。

**修订代替删除与判定守卫。** 初衷是"archive old + add corrected"。EgoLife 上修订链极少：lines 库 revision 边 11 条，各运行 UPDATE 5–22 条（`summary.json`），对 4,851 窗口不到 0.5%。机制真正工作的地方是反面：模型按位置配对判 NONE 时，1,685 条 NONE（HIGHLIGHTS §1.4 口径；`summary.json` 记 1595 加 50 个整窗丢失）让 22% 陈述未入库，记忆行 53.9%；守卫后 NONE 15、56.4%（HIGHLIGHTS §1.4）。所以修订机制在此数据上保护的是"不丢"，不是"改对"。

**ContextIR / 状态卡与混合读。** 卡片在 lines 契约下 63.2% 对记忆行 54.9%（EGOLIFE_REAL_LLM_MEM0_VS_ELITE §1）；回放上 card_1200 对 rows 净 +22、p=0.001（READER_AB §三.1）；会话卡前缀 + 检索行的 hybrid 70.1%（143/204），对 rows p<0.0001（READER_AB §六）。起作用的机制是预算：1200→800→600→400 为 64.7/60.8/58.3/53.9%（READER_AB §二）；去重 0 行触发，时间排序无增益。hybrid 对 card_1200 只 +11 题、p=0.10（audit README 翻转表 p=0.099），未过显著性；"只有新默认对"的 45 题里 40 题 rows 本来就含 gold（audit README 注意事项 6）。缓存目标未达成：冷前缀命中 14.9%，相邻两版会话卡公共前缀中位 77 字符（READER_AB §六.7）。另一点要记下：这些卡片是在 `auto` 模式下编译的，lines 库有 4 条 EP，所以 DAY2 之后的卡片编译进入了图扩展与支持层（`compiler/compile.py:508–530`）；图扩展对分数的贡献没有单独记录，不能把 63.2% 归到"轴"一路。

**锁纪律 / 两段事务。** 承诺 9 有测试 `test_ingest.py:881` 与 `test_daemon.py:230` 前台/后台竞争测试。EgoLife 写入并发固定 1（HIGHLIGHTS §4），无实测。列在这里是因为它是代码里成立的不变量，不是因为有数据。

**离线确定性。** 六黄金场景在 memory/sqlite 两个 provider 上逐字节断言（`test_scenarios.py:887`）。真实运行不用可注入 `id_factory`，`secrets.token_hex` 的 id 决定并列排序，导致回放批次集合 43 题不同、pred 21 题不同（audit README 注意事项 3）。确定性是测试属性，不是运行属性。

## 四、没有体现或没有被检验的想法，以及原因

**分形塔 / InterfaceEP / L2–L3 / 巩固。** 原因：数据集不匹配，加上评测台没有调用巩固。EP 抽取只对被路由到 state 的事件发起，条件是动作动词且伴随结果句或约束句，或调用方显式 `route='state'`（`ingest/router.py:59`、`utils/text.py:141–176`、`:1088`；CONTRACT.md §9.2、§12.2）；本文复算 4,851 窗口 5 个满足且全为误触，lines 运行的抽取回复里 39 行 EP 候选，回放库物化 4 条 EP（DAY2 与 DAY5 各 2 条）。DESIGN.md §14 风险 1 事先写明规则抽取器"宁漏勿误"，真实文本主要是漏 EP；EgoLife 上的 5 个误触是它的另一面。L2 为空的直接原因不是 EP 数为 0，而是 `consolidate()` 在 `real_arms.py` 里从未被调用，4 条 EP 没有机会被 coarsen；即使调用，同日两条闲聊误触的 EP 能否过 `min_cohesion` 也未知。`why`、`refine` 同样无调用。初衷 H-13"高层 EP 不退化为 summary"在代码里由 `primary_support_node_id NOT NULL`、`support_node_ids min_length=2` 保证（承诺 7），但没有任何真实高层对象被生成过。另一原因是尚未验证：LLM coarsen 只用 ScriptedLLM 测过；`round_trip_check` 库内无调用者，`consolidate()` 不自检。

**transitions / 可训练（H-56、H-71）。** 原因：尚未验证。表和导出都在，黄金文件是 S6 的 5 行（`transitions_s6.jsonl`；CONTRACT.md §19.6 `store.stats()['transitions'] == 5`，§19.7 consolidate 后才是 6）；回放库里有 4 行 `ep` transitions。没有任何训练消费者读过它。初衷 §11.4 把它列为独立产品，这一条到今天只有格式和 4 行样本。

**feedback / 可蒸馏（H-51、H-70）。** 原因：数据集不匹配 + 有意裁剪。11 种反馈类型有矩阵测试，但四选一题没有用户纠偏这一步；技能提升被裁成单 IEP 可达（DESIGN.md §2 行 13），前提仍是有 IEP。

**handoff / rollback / 可交接可回滚（H-67、H-68）。** 原因：尚未验证。`StatePackage` 见证闭包、对称回滚、导入去重都有代码与测试，评测台没有调用（`real_arms.py:1352`、`:1450`）。承诺 3 的 `node_id == atom_id` 让 rollback 与 why 共用 id 空间，这一点只在 S6 场景测试里走过（CONTRACT.md §19.6）。初衷 §9.4 的核心句是接手方导入状态包而不是读历史（第 1065 行），没有一次真实交接。

**Lite 裁剪的 20 项里属于初衷的部分。** 按主题半衰期（DESIGN.md §2 行 9）对应初衷 `decay_stale_state`（第 559 行），被简化成单一 `half_life_days`，原因是确定性；embedding 实体消解（行 10）对应七轴里的 entity 轴，被简化成归一化精确匹配，原因是 hash embedding 无语义。中文实验显示实体轴在英文规则下几乎为空、开中文规则后实体命中 14→107 题但准确率无变化（EGOLIFE_ZH §二），说明实体轴目前只影响结构不影响答案。Personal/Edge/Group 渲染器（行 11）对应初衷 §11.2 RESONANCE 与 §11.3 Enterprise State，先裁后发布为四族（CONTRACT.md §0.5 行 11），代码里 `StateKind` 枚举 12 个成员（`state/kinds.py:43` 起；12 这个数只见于代码，设计文档未列），无数据。分段（行 4）、覆盖编译器（行 5）、daemon（行 1）同样先裁后发布，DESIGN.md §2 未同步。以上六项都是有意裁剪，其中四项（行 1、4、5、11）后来在 CONTRACT.md §0.5 改为 SHIPPED，行 9、10 未变；补回来的部分仍无实测。

**差异化主张的证据缺口。** vs RAG 的"改变检索对象"只在四选一上测了召回与延迟；vs Observation Memory 的"不仅看见过去，还能继承行动"（第 958 行）后半句无数据；vs Memory OS 的"输出可行动状态"只有 Work State 卡在四选一上的分数，工作场景（当前有效事实 / 失效假设 / 下一步）未测（HIGHLIGHTS §1.5）。§10 的 9 任务 × 4 方案全"强"表仍然是预期。

**token 权重反转（H-64）。** 原因：有意决策。初衷说 token 是副产品，Elite 用成本作为默认契约切换的三条件之一（CONTRACT.md §6.5）。这不是遗漏，是重心变化；它带来的收益（lines ¥2.24）和代价（记忆行低 4 点，库 5.6 倍）都有记录。

## 五、回归本心：哪些初衷值得重新拿起来，哪些可以放下

1. **拿起来：给 State Layer 一个真正有动作→结果闭包的数据集。** 这是全部未验证项的共同前提。需要的数据：agent transcript + tool call + test log 的真实工作日志，规模至少让 `has_action_closure` 命中率离开 5/4,851，或直接走结构化事件（承诺 8 的 tool_call/file_diff/test_log 路径不经规则分流）。实验：同一日志上跑 `add_events` → `consolidate` → `compile(deep)`，报告 EP 物化数、L2/L3 数量、`round_trip_check` 通过率；`hms.db` 里已有 `eps` 与 `transitions` 计数，评测日志 `retain_elite.jsonl` 当前没有 `eps` 键，应补上，并把 `retrieval['ablation']` 存进答题记录。没有这一步，§9.4–9.7 永远是断言。

2. **拿起来：把 Invalidated Assumptions / Failed Paths 作为独立指标测。** 初衷 Work State 卡的价值不在四选一。实验：在上一条的数据上，让接手方只读状态包回答"哪条路径已失败、下一步做什么"，对照读全量 history，比较正确率与 token。这直接检验 H-67 的主张：接手方导入状态包而不是读历史。

3. **拿起来：把确定性从测试属性变成运行属性。** 承诺 10 在测试成立、在运行不成立。现状分三层：原始 09-11 运行的原子 id 由 `secrets.token_hex` 生成、已固定，任何新 id 方案都复现不了它；`reader_ab.py` 已默认改用计数器 id（`--random-ids` 关闭），回放之间可复现，与原始运行的差异是结构性的（READER_AB §五）；差异只落在 card 变体 57/204 题，rows 变体 204 题 parity 全过（audit README 注意事项 2、3）。剩下的库层改动：`compiler/compile.py` 的槽内打平键从原子 id 改为写入序号或文本哈希，使同一份数据重建的库编出相同卡片；同时真实运行注入 `id_factory`。目标不是"与 09-11 一致"，而是"此后任意两次重建一致"。成本低，收益是后续 A/B 可信。

4. **放下或降级："轴选择替代相似度"（H-34）。** 真实嵌入下语义撑主力，关键词通道加权反而降召回（EGOLIFE_ZH §三.2）。保留轴作为门限和结构（实体、时间、可见性），不再把"替代相似度"当目标。若要翻案，需要真实嵌入下的逐通道消融，目前没有。

5. **暂放：可训练（H-56/71）与 Transition Data 产品线（§11.4）。** 在第 1 条有结果之前，transitions 表只有 4 行闲聊误触的 `ep` 记录可导出。保留格式与幂等键，不再投入。何时重拾：当 L2 IEP 在真实日志上稳定产出后，用 `transitions(fmt='tuples')` 导出并至少训一个 critic 做 sanity check。

一句总结，不加修饰：Elite 把 HMS 的不变量大部分写进了 schema 和触发器（见证三层、不可覆盖、边界边投影、一版一节点），其余靠类型与测试（atom 不存边靠 validator 与导入拓扑测试，"按需细化"靠调用者源码检查，治理靠选择时的可见性格而非策略引擎），这一步比初衷走得更远；但被数据检验的只有"找"和"锚"这两层，"图"走过但没有计量，"继承"与"行动"两层至今没有一条真实的动作→结果闭包穿过它们。
