# 见证语义决定 Witness semantics decision

日期：2026-09-07。适用范围：`hms_elite/ops/feedback.py::revoke` 与 `FeedbackProcessor._relationship_not_useful`、`hms_elite/models.py::Witnessed`、README §5.6 / §9、`CONTRACT.md` §16.1 与测试 56b / 56e / 56f。

## 1. 起因

condense 仓库的 `docs/hms_bridge.md` §"revoke_consistency" 用一个 23 事件的离线 Memory（rule LLM + hash embedder + `:memory:` store）把 hms-elite 的 `revoke()` 与 condense 的撤销级联放在同一份数据上比对，报告了两处 hms-elite 自身的不一致：

1. 原子与 InterfaceEP 的见证按 OR 处理（`revoke` 里 `if store.atom_anchor_ids(atom_id): continue`），而 README §9 的口号 "No Witness, No Meaning" 与 `models.py` 的注释 "anchor_ids[0] is the primary witness" 被读成 AND。AND 读法下应墓碑化而实际保留的对象有 9 个 MemoryAtom 与 1 个 InterfaceEP。
2. 高层投影边：`projections_from(低层边)` 得到的每条高层边在低层边失效时一律失效，不看该高层边自身的剩余见证（一条 L3 distill 边 12 个见证只倒 1 个也失效）。于是高层边对见证是 OR、对投影来源是 AND。

## 2. 契约原意核对

逐条核对后，契约写的是 OR，即存在条件。证据：

- `CONTRACT.md` §16.1 Revoke 步骤 (2)(3)(4) 的判定条件分别是"`atom_anchor_ids(atom)`（未争议）为空 → 归档"、"见证者**全部**已撤销 → 失效"、"`evidence_anchor_ids` ⊂ 已撤销 → 失效"。
- `CONTRACT.md` 测试 56b：撤销 `<anc_e2>`，它是 EP1 的 outcome 锚点，也就是 `ep1.anchor_ids[0]`，即主见证；契约要求 `<ep1_atom>` 与 `<atm_dc>` 保持有效。这条直接否定了"主见证必要、其余可选"的第三种读法。
- `CONTRACT.md` 测试 52 `evidence_wrong`："an atom cannot stay valid without an undisputed witness"，反过来，只要有一个未争议见证就有效。
- `CONTRACT.md` §16 `lower()`：`evidence_coverage` 与 `not_revoked` 只评估非 audit 项的未争议子集。
- README §5.6 原文："没有未争议见证者的原子被归档、使所有见证者全部被撤销的边与 InterfaceEP 失效"。

README §9 原文只写"都至少有一个证据锚点"，没有写撤销时的判定规则；`models.py` 的注释没有说明"主"的含义。AND 读法来自口号字面意思加上这条注释，不来自契约条款。所以不一致 (1) 不是实现错误，是 README §9 与 `models.py` 的措辞欠明确。主见证列在存储层的全部作用：`primary_anchor_id NOT NULL` 让非空成为结构保证，`trg_atoms_primary_anchor` 把它写进 `atom_anchor`（role `primary`），`create_atom` 用它作 history 行的默认见证，`add_atom_anchor` 对它幂等。这些都不蕴含语义上的必要性。

## 3. 投影边：两种候选规则的实测

`project(iep)` 把支持集 S 的每条边界边映射成一条高层边，按 `(outer', type)` MERGE，因此一条高层边可以有多个低层来源（`edge_projections` 一对多，`witness_edge_id` 只是第一个）；它的 `witness_anchor_ids` 是来源见证的并集。`promote_skills` 的 L3 distill 边同理，来源是不变量支持节点到约束原子的全部有效 `use` 边。高层边是低层边在商映射下的像。

实测 A（condense 夹具 `tests/test_hms_bridge.py::build_memory`，撤销第一条 `file_diff` 的锚点，变更前的实现）：被级联失效的 4 条高层边各有 6 个投影来源，只倒了 1 个；两条 L2 `use` 边自身 6 个见证剩 5 个，两条 L3 `distill` 边 12 个见证剩 11 个。

实测 B（同一 store，随后 `consolidate(filters=SCOPE)`）：`find_edge` 按 `(src, dst, type, scale)` 查找且不分有效性，`insert_edge` 对已存在的边只做 MERGE，`mentions` 从 6 变 11、从 1 变 2，`valid` 保持 0。4 条边无一恢复。所以"任一来源倒即失效"不是"先失效、下次 consolidate 重建"的保守策略，而是永久空洞：IEP 到约束的 `use` 关系与 IEP 到技能的 `distill` 关系在 5 个来源仍有效的情况下永久消失。这排除了把现状文档化为"每个投影来源都是结构性必要条件"的方案。

复现脚本（在 condense 目录下，`PYTHONPATH=../hms-elite`）：

```python
from tests.test_hms_bridge import build_memory, _action_anchor, SCOPE
hms = build_memory(); store = hms.store
rep = hms.revoke(_action_anchor(hms))
for eid in rep["edges_invalidated"]:
    srcs = store.projections_of(eid)
    print(eid, store.get_edge(eid).scale, len(srcs), [store.get_edge(s).valid for s in srcs])
hms.consolidate(filters=SCOPE)
print([store.get_edge(e).valid for e in rep["edges_invalidated"]])
```

## 4. 决定：统一的存在语义

两条规则，都以"至少剩一个"为判定：

- R1 见证规则（所有带见证的对象）：对象有效，当且仅当至少一个未争议、未撤销的锚点仍在其见证集中。`anchor_ids[0]` 不特殊。原子看 `atom_anchor` 未争议行，边看 `edge_witnesses` 的 `revoked_at`，InterfaceEP 看 `evidence_anchor_ids`。
- R2 来源规则（仅投影边，即有 `edge_projections` 行的边）：在 R1 之外，还需至少一个投影来源仍有效，沿 L1 → L2 → L3 传递。理由：投影边是低层边的像，没有有效原像时 `refine` 打不开它，违反"高层对象是接口，不是摘要"与往返律 `refine(project(coarsen(S)))['boundary_edges'] == boundary(S)`；`graph_stats.traceable_high_edges` 也是按"≥1 条投影行"定义的。

不采用的三种方案与理由：AND（契约 56b 否定；实测下会多墓碑 9 个原子、1 个 IEP、4 条边）；主见证必要（56b 撤销的正是主见证；主见证列只有 §2 末尾列出的存储层作用）；每个来源都必要（§3 实测 B 的永久空洞）。

InterfaceEP 的支持节点不另设结构条件：`evidence_anchor_ids` 按 §14.2 等于支持节点锚点与支持边见证的并集，支持节点全部无见证时证据必然全部撤销，R1 已蕴含；被归档的原子行不删除，`refine` 仍能打开。

## 5. 对 revoke 级联的影响

实现变更（`hms_elite/ops/feedback.py`）：

- 新增模块级 `_cascade_projections(store, fallen, *, reason, at)`：以刚失效的边为种子的工作队列；对 `projections_from(倒下的边)` 中仍有效的高层边，只有当 `projections_of(高层边)` 的来源全部无效时才 `invalidate_edge`，并把它入队以传递到 L3。仍有有效来源的边不动，同一队列里另一个来源倒下时再检查。
- `revoke` 步骤 (3) 只按 R1 使见证全部撤销的边失效（`witness_revoked`），然后以这些边为种子调用级联，原因字符串为新的 `projection_revoked`。一条高层边可能自身见证仍在却因来源全失效而倒下，旧原因 `witness_revoked` 对它是错的。报告键不变，投影边追加在见证规则的边之后。
- `_relationship_not_useful` 走同一个助手，原因仍是 `feedback:not_useful`。契约原文把撤销的投影处理写成"同 `relationship_not_useful`"，两处必须同规则。

变更前后的差别只在多来源高层边上：单见证、单来源的 56b 场景逐字不变。R1 对原子与 IEP 的处理没有变，变的是文档。

测试：56e `test_revoke_existence_semantics_and_projection_sources` 用 S4 + consolidate 的两条低层 `use` 边、它们的 L2 投影与 L3 distill 边，先撤销 EP2 的主见证（原子存活），再撤销一个来源的锚点（高层边存活），给高层边补一个无关见证后撤销最后一个来源的锚点（高层边以 `projection_revoked` 倒下、自身见证仍在），每步之后对 scope 内全部边断言 R1 ∧ R2。56f 对 `relationship_not_useful` 断言同样的"最后一个来源才连带"。

## 6. 对 PC-IR `not_revoked` 义务的影响

义务本身不变。`lower()` 对每个非 audit 项取 `anchor_ids` 减去 `explain['disputed_anchor_ids']`，其中任一锚点 `revoked_at` 非空即 `violated`。`revoke` 在每个引用该锚点的原子上把链接标为争议，所以撤销后存活的原子项其未争议子集不含被撤销锚点，`not_revoked` 与 R1 给出同一判决：项有效 ⇔ 未争议子集非空且干净。被撤销锚点仍留在 `anchor_ids` 里供审计。义务只会在绕过级联直接调用 `store.revoke_anchor` 时 `violated`，测试 51 就是这样构造的。

## 7. condense 桥接复核

命令：`cd ../condense && PYTHONPATH=../hms-elite python3 -m pytest tests/test_hms_bridge.py -q`。日期 2026-09-07，变更后的 hms-elite。

| 语义 | agreed | only_condense | only_hms | consistent |
|---|---|---|---|---|
| AND（变更前，`hms_bridge.md` 记录） | 7 | 10（9 MemoryAtom + 1 InterfaceEP） | 0 | False |
| AND（变更后） | 3 | 14（9 MemoryAtom + 4 GraphEdge + 1 InterfaceEP） | 0 | False |
| OR（变更前，`hms_bridge.md` 记录） | 3 | 0 | 4（GraphEdge，投影级联） | False |
| OR（变更后） | 3 | 0 | 0 | True |

变更后 hms-elite 的报告：`archived` 1、`edges_invalidated` 2（两条单见证 `use` 边，原因都是 `witness_revoked`）、`interface_eps_invalidated` 0；OR 下 `stale_vs_disputed.match == True`，9 个 condense `stale` 原子等于 hms-elite "已争议但存活"的 9 个原子。AND 下 only_condense 从 10 变 14 是因为那 4 条高层边现在保留，而 AND 映射把它们墓碑化；AND 从来不是契约语义，这一行只作对照。

桥接测试结果：8 个中 7 通过、1 失败。失败的是 `tests/test_hms_bridge.py::test_revoke_consistency_or_semantics_matches_except_projections` 第 233 行 `assert rep.only_hms`，它把变更前的结构级联钉成"OR 下唯一剩余的差异"。按本次任务的指示没有改 condense。condense 侧要跟进的改动：该断言改为 `assert rep.only_hms == []`（或 `assert rep.consistent`），删除随后基于 `fallen_projection_sources` 的循环；`docs/hms_bridge.md` 的结论 1、2、"接口建议 (c)" 与对象映射表里 `Witnessed.anchor_ids` 一行的 "README §9 把它读成 AND" 需要按本文改写。
