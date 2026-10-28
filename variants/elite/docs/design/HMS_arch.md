# HMS 完整架构框架：Experience-to-State 双平面版本

> 版本：2026-05-10  
> 目标：把 HMS 的整体脉络、数据流转、模块职责、输入输出、代码文件职责、双语注释规范、差异化优势与可验证指标统一到一份工程可执行文档中。

---

## 0. 一句话总纲

HMS 是 **Experience-to-State Infrastructure**：把人和 Agent 在真实任务中的行动、判断、失败、约束和结果，编译成下一次行动可以继承、调用、纠偏、训练和审计的状态。

它的核心不是“保存更多历史”，而是把原始经验转成：

\[
(S_t, a_t, S_{t+1}, r_t, \kappa_t, w_t)
\]

其中：

| 符号 | 含义 | 中文解释 |
|---|---|---|
| \(S_t\) | current state | 行动前状态 |
| \(a_t\) | action | 动作 |
| \(S_{t+1}\) | next state | 行动后状态 |
| \(r_t\) | reward / evaluation | 结果评价 |
| \(\kappa_t\) | constraints / governance | 约束与治理边界 |
| \(w_t\) | witness | 证据锚点 |

HMS 当前需要同时服务两类目标：

1. **快速定位状态**：普通长期记忆、偏好、时间、实体、多会话问答，需要快速查到相关状态原子。
2. **继承行动经验**：handoff、rollback、detective、skill distillation、world-model transition，需要知道状态如何由低层证据支撑、如何被修订、如何被回滚。

因此最终架构是：

```text
State Index（多轴状态索引，热路径）
  负责定位、过滤、投影、快速回答

Support Graph（多尺度支持图，继承路径）
  负责证据、因果、修订、冲突、回滚、下钻、蒸馏

ContextIR / StatePackage（输出层）
  负责把候选状态编译成下一次行动可用的当前状态
```

一句话：

> **轴负责找，图负责继承，ContextIR 负责行动。**

---

## 1. 为什么要把“分形案例塔”重构成“双平面架构”

早期设计中，HMS 使用 Micro / Meso / Macro 的 3D-EP Stack，强调同一对象语法和多尺度经验结构。这个直觉是对的，但需要精确化。

原始直觉：

```text
Micro-EP / Meso-EP / Macro-EP 使用相同六元组语法。
同一层 EP 之间有 typed edges。
不同层图之间似乎也应该有某种自相似性。
```

修正后的严格表述：

```text
EP 结构可以跨尺度同构。
图结构不要求严格自相似。
跨层关系要求“可追溯”：高层节点必须有低层 support，高层边必须有低层 witness。
```

因此，不再把架构主概念叫“分形案例塔”，而是拆为两个互补平面：

### 1.1 多轴状态索引 State Index

用于热路径查询。解决：

```text
这条问题应该查哪个时间、哪个实体、哪个主题、哪个 scale？
```

它不存边，只存每个原子在多个轴上的坐标。

### 1.2 多尺度支持图 Support Graph

用于状态继承。解决：

```text
这个高层状态由哪些低层证据支撑？
这个结论为什么成立？
发生冲突时如何下钻？
回滚到哪里？
技能从哪些经验中蒸馏出来？
```

它不追求查询最快，而追求证据、结构、回滚、蒸馏和解释。

---

## 2. 端到端数据流

整体流程如下：

```text
Raw Experience
  对话 / 工具调用 / 文件修改 / 测试结果 / 会议 / 客户工单 / 社交反馈 / 传感器
        ↓
SourceEvent
  标准化事件，统一来源、时间、权限、原始引用
        ↓
Anchor
  一事一锚，证据只存一次，所有上层对象引用 anchor
        ↓
Event Router
  根据事件类型分流
        ↓
Observation Layer                 State Layer
普通事实 / 偏好 / 时间 / 个人资料      动作 / 修订 / 冲突 / 回滚 / 交接
        ↓                          ↓
MemoryAtom / StateAtom             EP / Case Graph / InterfaceEP
        ↓                          ↓
State Index                        Support Graph
多轴索引，快速定位                   多尺度图，支持继承与下钻
        ↓                          ↓
Query Planner / State Compiler
  先走轴查询，必要时展开支持图
        ↓
ContextIR / StatePackage / WorkState / PersonalState / EdgeState
        ↓
Renderer / Tool Output
  Work State Card / Handoff Package / Detective Report / Personal Card / Edge Card
        ↓
Feedback
  准确 / 不准确 / 已过期 / 不公开 / 下一步不对 / 推荐有用 / 推荐无用
        ↓
State Update / Graph Update / Index Update
```

---

## 3. 各模块职责、输入、输出

### 3.1 SourceEvent 标准化层

**职责**：把所有外部输入统一成事件。

输入：

```text
Agent transcript
Tool call
File diff
Test log
Slack / 飞书 / Teams message
Zendesk / Jira / CRM event
Meeting transcript
User feedback
Social action
Sensor trace
```

输出：

```json
{
  "event_id": "evt_001",
  "source_system": "claude_code|slack|jira|resonance|sensor",
  "object_type": "message|tool_call|file_diff|feedback|sensor_trace",
  "actor_id": "user_or_agent_id",
  "timestamp": "2026-05-10T00:00:00Z",
  "event_type": "created|updated|feedback|rollback|tool_result",
  "raw_ref": "source pointer",
  "visibility": "private|team|event|public",
  "payload_hash": "sha256..."
}
```

关键原则：

```text
不直接把 raw data 当记忆。
先把输入规范化为 SourceEvent。
```

---

### 3.2 Anchor 证据层

**职责**：给每个原始事件生成证据锚点。

输入：

```text
SourceEvent
```

输出：

```json
{
  "anchor_id": "anc_001",
  "source_event_id": "evt_001",
  "hash": "sha256...",
  "source_ref": "slack://...|file://...|git://...",
  "timestamp": "2026-05-10T00:00:00Z",
  "permission_scope": "team"
}
```

设计原则：

```text
One Event, One Anchor.
No Witness, No Meaning.
```

作用：

```text
Verify：证明状态从哪里来。
Revoke：撤销时能找到派生对象。
Rollback：回到某个证据支撑的版本。
Audit：解释输出依据。
```

---

### 3.3 Observation Layer

**职责**：处理普通长期记忆。用于“看见过去”。

适合输入：

```text
用户偏好
普通事实
时间锚点
个人资料
日常对话信息
非行动闭包的观察
```

输出：

```json
{
  "obs_id": "obs_001",
  "anchor_id": "anc_001",
  "type": "fact|preference_event|habit|constraint|temporal_fact",
  "subject": "user|project|customer|agent",
  "predicate": "likes|uses|compares|plans|updated|avoids",
  "objects": ["object_a", "object_b"],
  "relation": "single_fact|comparison|habit|supersession",
  "text": "User compared Fender Stratocaster and Gibson Les Paul; no exclusive final choice was stated.",
  "date_observed": "2026-05-10",
  "confidence": 0.83,
  "visibility": "private"
}
```

核心边界：

```text
Preference 不是 EP。
普通事实不是 EP。
只有被行动引用、引发结果、参与修订或冲突时，才进入 State Layer。
```

---

### 3.4 EP / Case Object 层

**职责**：将具有行动闭包的经验转成案例对象。

EP 定义：

\[
c_i=(u_i,b_i,a_i,y_i,r_i,\kappa_i)
\]

输入条件：

```text
存在 trigger / background
存在 action
存在 outcome 或 constraint
```

输出：

```json
{
  "case_id": "ep_001",
  "anchor_id": "anc_001",
  "scale": "L1_micro",
  "u": "test failed after retry patch",
  "b": "subscription state machine must not change",
  "a": "patch retry.ts",
  "y": "billing_e2e failed",
  "r": "negative",
  "kappa": {
    "constraints": ["do not modify subscription state machine"],
    "visibility": "team",
    "evidence": ["anc_test_log"]
  }
}
```

作用：

```text
把日志转成行动经验。
把经验变成可建边、可回滚、可继承对象。
```

---

### 3.5 State Index / MVMI 多轴状态索引

**职责**：热路径检索。快速找到与问题相关的状态原子。

核心对象：`MemoryAtom`

```python
@dataclass
class MemoryAtom:
    """Unified indexable atom for hot-path query.
    统一可索引原子，用于热路径查询。"""
    atom_id: str
    atom_type: Literal["message", "state", "event", "entity", "relation", "episode"]
    content: dict
    time_anchor: str | None
    valid_from: str | None
    valid_to: str | None
    entity_ids: list[str]
    topic_tags: list[str]
    scale_level: Literal["L1_micro", "L2_meso", "L3_macro", "state", "event", "entity", "relation"]
    role: Literal["user", "assistant", "system", "tool", "extracted"]
    sentiment: Literal["positive", "neutral", "negative", "mixed", "unknown"] | None
    source_ep_ids: list[str]
    confidence: float
    extracted_by: str
```

七个查询轴：

```text
time
validity
entity
topic
role
sentiment
scale
```

查询对象：`AxisQuery`

```python
@dataclass
class AxisQuery:
    """A query plan over the multi-view index.
    多视角索引上的查询计划。"""
    filters: dict[str, Any]
    project_axis: Literal["time", "entity", "value", "topic", "atom"]
    scale_constraint: list[str] | None
    aggregation: Literal["list", "latest", "count", "summary", "topk"] | None
    limit: int
```

典型查询：

```python
# preference
AxisQuery(
    filters={"topic": "preference", "subject": "coffee"},
    project_axis="value",
    scale_constraint=["state"],
    aggregation="latest",
)

# temporal
AxisQuery(
    filters={"topic": "trip"},
    project_axis="atom",
    scale_constraint=["event", "L1_micro"],
    aggregation="list",
)

# multi-session
AxisQuery(
    filters={"entity": "Sarah"},
    project_axis="atom",
    scale_constraint=["event", "entity", "L1_micro"],
    aggregation="list",
)
```

关键边界：

```text
MemoryAtom 不存边。
State Index 只负责定位，不负责解释因果或继承。
```

---

### 3.6 Support Graph 多尺度支持图

**职责**：状态继承路径。保存因果、修订、冲突、support、boundary、rollback、distillation。

核心判断：

```text
高层 EP 是低层子图的接口，不是摘要。
```

核心对象：`InterfaceEP`

```python
@dataclass
class InterfaceEP:
    """High-level EP = interface to low-level subgraph, not a summary.
    高层 EP = 低层子图的接口对象，而不是摘要。"""
    ep_id: str
    scale: Literal["L2_meso", "L3_macro"]
    slots: dict
    support_node_ids: list[str]
    support_edge_ids: list[str]
    boundary_edges: list[BoundaryEdge]
    invariants: list[Invariant]
    residuals: list[Residual]
    evidence_anchor_ids: list[str]
    coarsen_op_version: str
    cohesion_score: float
```

五个组成：

```text
slots：高层自己的六元组槽位
support：低层子图，带节点和边
boundary：跨子图边，高层接口的本质
invariants：可向上继承的稳定不变量
residuals：必须下钻才能解决的问题
```

三算子：

```text
Coarsen：低层子图 → 高层 InterfaceEP
Project：低层边界边 → 高层边
Refine：高层 InterfaceEP → 低层支持子图
```

四不变量：

```text
同构对象语法
证据保持
边界保持
按需细化
```

---

### 3.7 Case Graph 边类型

同层 EP 图与跨层投影都使用六类边。

```text
temporal：时间先后
causal：结果成为后续前提
revision：新结论替代旧结论
conflict：结论不兼容且未解决
use：后续行动使用前序对象
distill：低层案例被蒸馏为高层技能或原则
```

跨层投影规则：

```text
内部边被 InterfaceEP 吸收。
边界边投影为高层边。
每条高层边必须能回溯到低层 witness edge。
```

---

### 3.8 ContextIR / PC-IR 编译层

**职责**：把候选原子与支持图编译成下一步行动所需状态。

ContextIR 槽位：

```text
facts
episodes
preferences
constraints
skills
audit
```

PC-IR 扩展：

```text
payload：机器可用状态对象
witness：证据集合
human gloss：确定性可读解释
proof obligations：证据覆盖、无未解冲突、治理约束、未撤销
```

输出示例：

```text
Current Goal:
修复 checkout webhook retry

Valid Facts:
idempotency.ts patch 当前有效

Invalidated Assumptions:
直接修改 retry.ts 能解决问题

Failed Paths:
retry.ts patch 造成 subscription state regression

Constraints:
subscription state machine 不能改

Next Action:
检查 test isolation

Evidence:
git diff / test log / rollback anchor
```

---

### 3.9 StateScheduler / Consolidation 后台固化

**职责**：异步整理状态，不阻塞前台查询。

任务：

```text
observe_event
extract_observation
create_ep
build_edges
update_mvmi
coarsen_support_graph
compile_state
process_feedback
detect_conflict
promote_skill
decay_stale_state
```

四阶段：

```text
Orient：分析最近活跃主题
Collect：导入新事件
Consolidate：建边、合并、coarsen、project、更新索引
Prune：衰减旧状态、归档低置信对象
```

---

### 3.10 StateFeedback 反馈层

**职责**：把用户和 Agent 的纠偏变成状态更新。

反馈类型：

```text
state_wrong
state_outdated
state_missing
privacy_violation
next_action_wrong
relationship_not_useful
connection_reason_wrong
evidence_wrong
render_not_like_me
render_like_me
object_important
```

处理结果：

```text
archive old state
add corrected state
update visibility
recompute confidence
create revision edge
rerender cards
update MVMI indexes
```

---

## 4. 查询 / 编译执行流程

### 4.1 普通 recall 查询

例如：用户之前喜欢什么咖啡？

```text
query
  ↓
route = observation / preference
  ↓
AxisQuery(topic=preference, subject=coffee, scale=state)
  ↓
MVMI returns MemoryAtoms
  ↓
ContextIR.preference slot
  ↓
answer / state card
```

只走索引平面。

---

### 4.2 temporal / multi-session 查询

例如：Sarah 相关的所有事件是什么？

```text
query
  ↓
AxisQuery(entity=Sarah, scale=[event, L1_micro])
  ↓
MVMI returns event atoms sorted by time
  ↓
optional light support graph expansion
  ↓
ContextIR.timeline
```

主要走索引平面，必要时轻量展开支持图。

---

### 4.3 handoff / rollback / detective 查询

例如：为什么不能改 retry.ts？

```text
query
  ↓
AxisQuery(topic=retry.ts, scale=[L2_meso, L1_micro])
  ↓
MVMI finds InterfaceEP / relevant atoms
  ↓
Support Graph refine(boundary_witness / evidence)
  ↓
retrieve low-level support nodes + edges
  ↓
PC-IR renders evidence-backed state
```

索引负责定位，支持图负责继承与解释。

---

### 4.4 world-model transition extraction

```text
EP / InterfaceEP
  ↓
extract S_t, a_t, S_{t+1}, r_t, κ_t, w_t
  ↓
store transition dataset
  ↓
train predictor / critic / policy prior / world model
```

---

## 5. 代码文件职责与双语注释规范

### 5.1 现有核心文件职责

| 文件 | 职责 |
|---|---|
| `models.py` | Anchor、EP、CaseEdge、ContextIR 等核心数据模型 |
| `schema.py` | SQLite 表结构，需新增 MVMI 与 SupportGraph 表 |
| `store.py` | 读写 Anchor、EP、Edge、State、Feedback、Atom、InterfaceEP |
| `ingest.py` | 原始事件摄取，生成 Anchor / Observation / EP / MemoryAtom |
| `segmentation.py` | 长消息切分为语义原子 |
| `coverage.py` | 覆盖选择与需求抽取 |
| `embeddings.py` | embedding、cosine、MMR 等语义检索工具 |
| `compiler.py` | 主 ContextIR 编译器 |
| `scc.py` | 分层覆盖编译器与 PC-IR lowering |
| `pc_ir.py` | 携带 witness 与 proof obligation 的输出结构 |
| `consolidation.py` | 离线建边、合并、提升、蒸馏 |
| `daemon.py` | 后台固化循环 |
| `ops.py` | handoff、rollback、checkpoint |
| `runtime.py` | Agent runtime adapter 与 action registry |
| `cli.py` | 命令行接口 |
| `mvmi.py` | 新增：多轴状态索引 |
| `support_graph.py` | 新增：多尺度支持图 |

---

### 5.2 双语注释要求

所有代码文件应包含：

```python
"""Module purpose in English.
模块目的中文说明。
"""

# Purpose (EN): What this file is responsible for.
# 目的（中）：说明本文件负责什么。

# Input (EN): What enters this module.
# 输入（中）：说明模块接收什么。

# Output (EN): What this module produces.
# 输出（中）：说明模块产出什么。

# State effect (EN): Whether this function mutates state, graph, index, or store.
# 状态影响（中）：说明函数是否修改状态、图、索引或存储。
```

函数级示例：

```python
def compile_state(store: StateStore, query: QueryContext, budget: int) -> ContextIR:
    """Compile task-ready state from indexed atoms and support graph.
    从多轴索引原子和支持图中编译任务可用状态。

    Input / 输入：
        store: state store containing EPs, atoms, edges, and anchors.
        store：包含 EP、原子、边和锚点的状态存储。

        query: task or user request.
        query：任务或用户请求。

    Output / 输出：
        ContextIR with facts, episodes, constraints, skills, and audit witnesses.
        包含事实、经历、约束、技能和审计 witness 的 ContextIR。

    State effect / 状态影响：
        Read-only by default; optional persistence records compiled state snapshot.
        默认只读；可选持久化编译状态快照。
    """
```

---

## 6. 新增模块设计

### 6.1 `src/hms/mvmi.py`

职责：

```text
统一所有可索引原子。
提供 AxisQuery。
把 preference / temporal / multi-session 等任务从“相似度检索”改成“轴选择”。
```

输入：

```text
EP
StateUpdate
CrystalEvent
CrystalRelation
Observation
InterfaceEP reference
```

输出：

```text
MemoryAtom
AxisQuery result
```

关键边界：

```text
MemoryAtom 不存边。
MemoryAtom 只存坐标与出处。
```

---

### 6.2 `src/hms/support_graph.py`

职责：

```text
把高层 EP 建成低层子图的接口。
支持 Coarsen / Project / Refine。
保存 support、boundary、invariants、residuals。
```

输入：

```text
低层 EP 子图
低层 case_edges
anchors
coarsen policy
```

输出：

```text
InterfaceEP
projected high-level edges
refinement node ids
```

核心断言：

```text
高层 EP = 低层子图接口。
高层边 = 低层边界边投影。
高层状态 = 低层 witness 支撑。
```

---

## 7. 需要新增 / 修改的数据库表

### 7.1 `memory_atoms`

```sql
CREATE TABLE memory_atoms (
    atom_id TEXT PRIMARY KEY,
    atom_type TEXT,
    content TEXT,
    time_anchor TEXT,
    valid_from TEXT,
    valid_to TEXT,
    scale_level TEXT,
    role TEXT,
    sentiment TEXT,
    confidence REAL,
    extracted_by TEXT
);
```

### 7.2 多对多轴表

```sql
CREATE TABLE atom_entity (
    atom_id TEXT,
    entity_id TEXT,
    PRIMARY KEY (atom_id, entity_id)
);

CREATE TABLE atom_topic (
    atom_id TEXT,
    topic TEXT,
    PRIMARY KEY (atom_id, topic)
);
```

### 7.3 `interface_eps`

```sql
CREATE TABLE interface_eps (
    interface_ep_id TEXT PRIMARY KEY,
    scale TEXT,
    slots_json TEXT,
    support_node_ids_json TEXT,
    support_edge_ids_json TEXT,
    boundary_edges_json TEXT,
    invariants_json TEXT,
    residuals_json TEXT,
    evidence_anchor_ids_json TEXT,
    coarsen_op_version TEXT,
    cohesion_score REAL
);
```

### 7.4 索引

```sql
CREATE INDEX ix_atoms_time ON memory_atoms(time_anchor);
CREATE INDEX ix_atoms_validity ON memory_atoms(valid_from, valid_to);
CREATE INDEX ix_atoms_scale ON memory_atoms(scale_level);
CREATE INDEX ix_atoms_role ON memory_atoms(role);
CREATE INDEX ix_entity_atom ON atom_entity(entity_id);
CREATE INDEX ix_topic_atom ON atom_topic(topic);
```

---

## 8. 差异化优势到底是什么

### 8.1 与 RAG 的差异

RAG：

```text
query → retrieve chunks → prompt → answer
```

HMS：

```text
experience → anchor / atom / EP / graph → state index + support graph → ContextIR → action
```

HMS 改变的不是“评分函数”，而是检索对象本身：

```text
chunk → state atom / EP / interface EP / state package
```

优势：

```text
更少 query-time 推理负担。
更低噪声。
更强证据追溯。
支持 handoff / rollback / detective。
```

---

### 8.2 与 Observation Memory 的差异

Observation Memory 擅长：

```text
事实、偏好、时间、普通 recall
```

HMS 吸收它作为 Observation Layer，但继续往上做：

```text
行动状态
失败路径
冲突修订
回滚点
状态转移
```

优势：

```text
不仅看见过去，还能继承行动。
```

---

### 8.3 与 Memory OS 的差异

Memory OS 擅长：

```text
存、取、调度、治理 memory resource
```

HMS 重点：

```text
把 memory + action + evidence + feedback 结算成 state
```

优势：

```text
HMS 输出的是可行动状态，不只是可检索记忆。
```

---

### 8.4 与长上下文 / compaction 的差异

长上下文：

```text
让模型看到更多历史
```

Compaction：

```text
把历史压短
```

HMS：

```text
把经验重构成状态
```

优势：

```text
不依赖模型在推理时重新整理全部历史。
状态已经被预计算、证据化、结构化。
```

---

## 9. 优势到底体现在哪些指标上

HMS 的优势不应只说“省 token”。

### 9.1 省 token

来自：

```text
高层 InterfaceEP 替代大量低层细节。
MVMI 直接命中 state/event/entity 原子。
ContextIR 只渲染任务需要的状态。
```

但 token 不是主目标，只是资源约束下的副产品。

### 9.2 速度更快

普通问答走 MVMI：

```text
indexed axis lookup → atoms
```

减少：

```text
全量 embedding rerank
长 prompt 拼接
反复 LLM 总结
```

### 9.3 准确率更高

原因：

```text
检索对象从 raw message 变成 state/event/entity/relation atom。
preference 保留 context。
temporal 直接查 event time。
multi-session 直接按 entity 聚合。
```

### 9.4 可交接

因为：

```text
ContextIR + EP + support graph + witness anchors
```

接手方不是读历史，而是导入状态包。

### 9.5 可回滚

因为：

```text
Anchor + EP version chain + support graph + invalidation / revision edge
```

### 9.6 可审计

因为：

```text
每个状态输出都带 witness。
高层边有低层 witness edge。
高层 EP 有 support graph。
```

### 9.7 可蒸馏

因为：

```text
InterfaceEP invariants 可向上形成 skill / policy / macro state。
```

### 9.8 可训练

因为：

```text
每次经验最终可转成 (S_t, a_t, S_{t+1}, r_t, κ_t, w_t)。
```

这可以进入 world model、RL、continual training、adapter training。

---

## 10. 为什么这套架构比别人强

HMS 的强点不是单点算法，而是组合结构：

```text
1. MVMI 让查询更快、更准。
2. Support Graph 让状态可继承、可下钻。
3. Anchor 让所有状态可验证、可撤销。
4. ContextIR 让输出直接服务下一步行动。
5. StateFeedback 让系统能被用户和 Agent 纠偏。
6. Transition Layer 让经验成为可训练数据。
```

对应任务：

| 任务 | 只用 RAG | 只用 Observation | 只用 Graph | HMS 双平面 |
|---|---|---|---|---|
| preference recall | 中 | 强 | 弱 | 强 |
| temporal | 中 | 强 | 中 | 强 |
| multi-session | 中 | 中 | 中 | 强 |
| handoff | 弱 | 弱 | 强 | 强 |
| rollback | 弱 | 弱 | 强 | 强 |
| conflict / revision | 弱 | 中 | 强 | 强 |
| evidence audit | 中 | 中 | 强 | 强 |
| skill distillation | 弱 | 中 | 强 | 强 |
| world-model transition | 弱 | 弱 | 中 | 强 |

---

## 11. 产品和场景如何映射到架构

### 11.1 Agent WorkState

输入：

```text
agent logs / tool calls / diffs / tests / constraints
```

输出：

```text
Work State Card
Handoff Package
Rollback Point
Detective Report
```

使用：

```text
Claude Code / Codex / Cursor / OpenClaw / internal agents
```

---

### 11.2 Personal API / RESONANCE

输入：

```text
用户自述 / 小人反馈 / 关系互动 / 活动选择 / 可见性设置
```

输出：

```text
PersonalState
Shared Card
EdgeState
GroupState
```

使用：

```text
状态卡 / 小人 / 共鸣 / 关系后续 / 群体图
```

---

### 11.3 Enterprise State

输入：

```text
Slack / 飞书 / Teams / Zendesk / Jira / CRM / meeting notes
```

输出：

```text
CustomerIssueState
IncidentState
MeetingDecisionState
AccountHandoffState
```

---

### 11.4 World Model Transition Data

输入：

```text
EP / InterfaceEP / ContextIR / feedback / reward
```

输出：

```text
(S_t, a_t, S_{t+1}, r_t, κ_t, w_t)
```

---

## 12. 实施路线

### Phase 1：MVMI Hot Path

目标：提升 LongMemEval / S benchmark 的 preference、temporal、multi-session。

工作：

```text
新增 mvmi.py
新增 memory_atoms 表
新增 AxisQuery planner
USTS / StateUpdate 升级 context field
CrystalEvent / CrystalRelation 抽取
```

---

### Phase 2：Support Graph

目标：让高层 EP 不再退化为 summary。

工作：

```text
新增 support_graph.py
InterfaceEP 五元组
Coarsen / Project / Refine
interface_eps 表
高层边 witness 约束
```

---

### Phase 3：Compiler Integration

目标：查询先走索引，复杂任务展开支持图。

工作：

```text
Query Router
AxisQuery execution
SupportGraph refinement
PC-IR witness rendering
```

---

### Phase 4：State Feedback

目标：所有用户和 Agent 纠偏进入状态更新。

工作：

```text
feedback_events 表
StateFeedback processor
revision edge
rerender
index update
```

---

### Phase 5：Benchmark and Ablation

必须测试：

```text
Axis only
Graph only
Axis + Graph
Axis + Graph + Support
```

任务：

```text
LongMemEval
Handoff
Rollback
Conflict
Next-action
Skill reuse
```

---

## 13. 最终一句

HMS 最终不是“更复杂的记忆系统”。

它是：

```text
多轴状态索引：让系统快速找到正确状态。
多尺度支持图：让状态可继承、可解释、可回滚。
ContextIR：让状态进入下一次行动。
Transition Layer：让经验进入模型训练。
```

**最终定义：**

> HMS 把经验编译成状态，把状态组织成可继承结构，把可继承结构变成下一代 AI 的输入。

