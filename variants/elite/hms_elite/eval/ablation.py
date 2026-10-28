"""HMS §12 Phase 5 ablation framework: the four ablations as read-only run switches over one HMS, and the runner.
HMS §12 第五阶段消融框架：四种消融作为同一个 HMS 上的只读运行开关，以及运行器。
"""

# Purpose (EN): Define Ablation (Axis only / Graph only / Axis + Graph / Axis + Graph + Support), the Probe that
#   applies an ablation to the public HMS API, the Task / TaskScore contracts and run_ablation → AblationReport.
# 目的（中）：定义 Ablation（仅轴 / 仅图 / 轴 + 图 / 轴 + 图 + 支持层）、把消融施加到 HMS 公开 API 的 Probe、
#   Task / TaskScore 契约，以及 run_ablation → AblationReport。

# Input (EN): An HMS factory, Task objects (scenario build + query + scoring) and the ablations to run.
# 输入（中）：HMS 工厂、Task 对象（场景构建 + 查询 + 打分）以及要运行的消融。

# Output (EN): AblationReport cells (score in [0, 1], explained checks, latency, LLM calls, tokens from telemetry meters).
# 输出（中）：AblationReport 单元格（[0, 1] 分数、可解释的检查项、延迟、LLM 调用数、来自 telemetry 计量器的 token 数）。

# State effect (EN): Task builds write into the HMS handed to them; the Probe itself only reads (transitions export is
#   idempotent INSERT OR IGNORE). Nothing outside the HMS store is touched.
# 状态影响（中）：Task 的构建阶段会写入交给它的 HMS；Probe 本身只读（转移导出是幂等的 INSERT OR IGNORE）。
#   不触碰 HMS 存储之外的任何东西。

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from hms_elite.enums import (
    ABLATION_OF_MODE,
    WHY_EDGE_TYPES,
    AtomType,
    CompileMode,
    EdgeType,
    NodeKind,
    Route,
    Slot,
    StrEnum,
)
from hms_elite.enums import TransitionKind as TK
from hms_elite.models import PCIR, AxisQuery, AxisResult, ContextIR, ContextItem, Scope

# Graph-only expansion depth (SupportGraph.neighbors hops) and the number of newest nodes used as seeds.
# 仅图消融的展开深度（SupportGraph.neighbors 跳数）与作为种子的最新节点数量。
GRAPH_ONLY_HOPS = 2
GRAPH_ONLY_SEEDS = 3

# Node kinds preferred as graph-only seeds (state layer first; falls back to any valid node).
# 仅图消融优先作为种子的节点种类（先状态层；退回到任意有效节点）。
_SEED_KINDS: tuple[str, ...] = (NodeKind.ep.value, NodeKind.interface_ep.value, NodeKind.skill.value)

# Transition kinds each plane can account for: EP objects live on the axes, edges on the graph, InterfaceEPs
# in the support layer (HMS §4.4 / §11.4).
# 各平面能够解释的转移种类：EP 对象在轴上，边在图上，InterfaceEP 在支持层（HMS §4.4 / §11.4）。
_AXIS_KINDS: tuple[str, ...] = (TK.ep.value,)
_GRAPH_KINDS: tuple[str, ...] = (TK.causal.value, TK.revision.value, TK.conflict.value, TK.rollback.value)
_SUPPORT_KINDS: tuple[str, ...] = (TK.interface_ep.value,)


@dataclass(frozen=True)
class AblationSwitches:
    """The read-only run switches one ablation applies to HMS.
    一种消融施加到 HMS 上的只读运行开关。
    """

    use_axis: bool
    use_graph: bool
    use_support: bool
    compile_mode: str | None
    transition_kinds: tuple[str, ...]


class Ablation(StrEnum):
    """HMS §12 Phase 5 ablations; values are the `retrieval['ablation']` labels of enums.ABLATION_OF_MODE.
    HMS §12 第五阶段消融；取值即 enums.ABLATION_OF_MODE 中的 `retrieval['ablation']` 标签。
    """

    axis_only = "axis"
    graph_only = "graph"
    axis_graph = "axis_graph"
    axis_graph_support = "axis_graph_support"

    @property
    def switches(self) -> AblationSwitches:
        """The switch set of this ablation.
        该消融的开关集合。

        Input / 输入：
            none.
            无。

        Output / 输出：
            AblationSwitches (compile_mode is None for graph_only, which never runs the axis compiler).
            AblationSwitches（graph_only 的 compile_mode 为 None，它从不运行轴编译器）。

        State effect / 状态影响：
            None.
            无。
        """
        return _SWITCHES[self.value]

    @property
    def compile_mode(self) -> str | None:
        """CompileMode value this ablation compiles with (None for graph_only).
        该消融所用的 CompileMode 值（graph_only 为 None）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            'fast' | 'axis_graph' | 'deep' | None.
            'fast' | 'axis_graph' | 'deep' | None。

        State effect / 状态影响：
            None.
            无。
        """
        return self.switches.compile_mode

    @classmethod
    def of_mode(cls, mode: str) -> "Ablation":
        """The ablation a CompileMode resolves to, through enums.ABLATION_OF_MODE.
        通过 enums.ABLATION_OF_MODE 得到某编译模式对应的消融。

        Input / 输入：
            mode: 'fast' | 'axis_graph' | 'deep' (CompileMode member or value); 'auto' has no fixed ablation → KeyError.
            mode：'fast' | 'axis_graph' | 'deep'（CompileMode 成员或值）；'auto' 无固定消融 → KeyError。

        Output / 输出：
            The Ablation member.
            Ablation 成员。

        State effect / 状态影响：
            None.
            无。
        """
        return cls(ABLATION_OF_MODE[str(getattr(mode, "value", mode))])


# Inverse of ABLATION_OF_MODE: ablation label → compile mode.
# ABLATION_OF_MODE 的逆映射：消融标签 → 编译模式。
MODE_OF_ABLATION: dict[str, str] = {label: mode for mode, label in ABLATION_OF_MODE.items()}

_SWITCHES: dict[str, AblationSwitches] = {
    Ablation.axis_only.value: AblationSwitches(
        use_axis=True,
        use_graph=False,
        use_support=False,
        compile_mode=MODE_OF_ABLATION[Ablation.axis_only.value],
        transition_kinds=_AXIS_KINDS,
    ),
    Ablation.graph_only.value: AblationSwitches(
        use_axis=False,
        use_graph=True,
        use_support=False,
        compile_mode=None,
        transition_kinds=_AXIS_KINDS + _GRAPH_KINDS,
    ),
    Ablation.axis_graph.value: AblationSwitches(
        use_axis=True,
        use_graph=True,
        use_support=False,
        compile_mode=MODE_OF_ABLATION[Ablation.axis_graph.value],
        transition_kinds=_AXIS_KINDS + _GRAPH_KINDS,
    ),
    Ablation.axis_graph_support.value: AblationSwitches(
        use_axis=True,
        use_graph=True,
        use_support=True,
        compile_mode=MODE_OF_ABLATION[Ablation.axis_graph_support.value],
        transition_kinds=_AXIS_KINDS + _GRAPH_KINDS + _SUPPORT_KINDS,
    ),
}


# ---------------------------------------------------------------------------
# Task and score contracts
# 任务与分数契约
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """One explained scoring check: name, partial score in [0, 1] and the reason for hit / miss.
    一条可解释的评分检查：名称、[0, 1] 内的部分分与命中 / 未命中的原因。
    """

    name: str
    score: float
    detail: str

    @property
    def hit(self) -> bool:
        """True when the check scored 1.0.
        检查得满分时为真。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        return self.score >= 1.0


@dataclass(frozen=True)
class TaskScore:
    """A task outcome: the mean of its checks in [0, 1] plus every check for explanation.
    任务结果：各检查项均值（[0, 1]）以及用于解释的全部检查项。
    """

    value: float
    checks: tuple[Check, ...]

    @classmethod
    def from_checks(cls, checks: Iterable[Check]) -> "TaskScore":
        """Average a check list into a TaskScore (empty list → 0.0).
        将检查列表平均为 TaskScore（空列表 → 0.0）。

        Input / 输入：
            checks: Check objects in report order.
            checks：按报告顺序的 Check 对象。

        Output / 输出：
            TaskScore with value rounded to 4 decimals.
            值四舍五入到 4 位小数的 TaskScore。

        State effect / 状态影响：
            None.
            无。
        """
        items = tuple(checks)
        if not items:
            return cls(0.0, items)
        value = sum(max(0.0, min(1.0, c.score)) for c in items) / len(items)
        return cls(round(value, 4), items)

    def misses(self) -> list[Check]:
        """Checks that did not fully hit.
        未完全命中的检查项。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of Check.
            Check 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [c for c in self.checks if not c.hit]


@dataclass(frozen=True)
class Task:
    """An executable benchmark task: build the scenario on an HMS, then query through a Probe and score.
    可执行的基准任务：在 HMS 上构建场景，再通过 Probe 查询并打分。
    """

    name: str
    description: str
    build: Callable[[Any], dict]
    run: Callable[["Probe", dict], TaskScore]


# ---------------------------------------------------------------------------
# Probe: the ablation applied to the public HMS API
# Probe：施加到 HMS 公开 API 上的消融
# ---------------------------------------------------------------------------
class Probe:
    """Read-only view of an HMS under one ablation: every query goes through the switches, never around them.
    某种消融下 HMS 的只读视图：每个查询都经过开关，绝不绕过。
    """

    def __init__(self, hms: Any, ablation: Ablation | str) -> None:
        """Bind an HMS and an ablation.
        绑定 HMS 与消融。

        Input / 输入：
            hms: The HMS facade (public API only is used).
            hms：HMS 门面（只使用公开 API）。

            ablation: Ablation member or its value.
            ablation：Ablation 成员或其值。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.hms = hms
        self.ablation = Ablation(str(getattr(ablation, "value", ablation)))
        self.switches = self.ablation.switches

    # ------------------------------------------------------------------ compile / PC-IR
    # ------------------------------------------------------------------ 编译 / PC-IR
    def compile(self, query: str, *, filters: dict, budget_tokens: int | None = None, as_of: Any = None) -> ContextIR:
        """Compile the Work State under the ablation (axis modes → HMS.compile(mode=…); graph_only → why() expansion).
        在消融下编译工作态（轴类消融 → HMS.compile(mode=…)；graph_only → why() 展开）。

        Input / 输入：
            query: Goal / question text.
            query：目标 / 问题文本。

            filters: Mem0 filter dict with >= 1 scope key.
            filters：含至少一个范围键的 Mem0 过滤字典。

            budget_tokens: Token budget (None → config default).
            budget_tokens：token 预算（None → 配置默认值）。

            as_of: Reference instant (None → store.now()).
            as_of：参考时刻（None → store.now()）。

        Output / 输出：
            ContextIR; retrieval['ablation'] == this ablation's value.
            ContextIR；retrieval['ablation'] == 本消融的取值。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        mode = self.switches.compile_mode
        if mode is None:
            return self._graph_compile(query, filters=filters, budget_tokens=budget_tokens, as_of=as_of)
        return self.hms.compile(query, filters=filters, budget_tokens=budget_tokens, mode=mode, as_of=as_of)

    def pcir(self, ir: ContextIR) -> PCIR | None:
        """Lower a ContextIR to PC-IR with its proof obligations; None unless the support layer is on.
        将 ContextIR 下沉为带证明义务的 PC-IR；支持层关闭时为 None。

        Input / 输入：
            ir: A ContextIR from `compile`.
            ir：来自 `compile` 的 ContextIR。

        Output / 输出：
            PCIR (obligations evaluated, never raised) or None.
            PCIR（义务已评估、从不抛出）或 None。

        State effect / 状态影响：
            None.
            无。
        """
        if not self.switches.use_support:
            return None
        return ir.to_pcir(require_witness=False)

    # ------------------------------------------------------------------ axis plane
    # ------------------------------------------------------------------ 轴平面
    def search(self, query: str, *, filters: dict, top_k: int = 10) -> list[dict]:
        """Ranked recall rows (Mem0 shape); graph_only ranks by graph reach and time instead of axis scores.
        排序后的回忆行（Mem0 形状）；graph_only 按图可达与时间排序而非轴分数。

        Input / 输入：
            query: Query text.
            query：查询文本。

            filters: Mem0 filter dict with >= 1 scope key.
            filters：含至少一个范围键的 Mem0 过滤字典。

            top_k: Maximum rows.
            top_k：最大行数。

        Output / 输出：
            Rows with at least 'id', 'memory', 'atom_id' and 'hms' {'entity_ids', ...}.
            至少含 'id'、'memory'、'atom_id' 与 'hms' {'entity_ids', …} 的行。

        State effect / 状态影响：
            None.
            无。
        """
        if not self.switches.use_axis:
            return self._graph_rows(query, filters=filters)[:top_k]
        return list(self.hms.search(query, filters=filters, top_k=top_k)["results"])

    def recall(self, *, filters: dict, top_k: int = 20) -> list[dict]:
        """Filtered listing without a query (Mem0 get_all); graph_only lists the graph nodes of the scope.
        无查询的过滤列表（Mem0 get_all）；graph_only 列出该范围的图节点。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key (axis filters such as 'entity' are ignored by graph_only).
            filters：含至少一个范围键的 Mem0 过滤字典（graph_only 忽略 'entity' 等轴过滤）。

            top_k: Maximum rows.
            top_k：最大行数。

        Output / 输出：
            Rows newest first.
            按最新在前排序的行。

        State effect / 状态影响：
            None.
            无。
        """
        if not self.switches.use_axis:
            return self._graph_rows("", filters=filters)[:top_k]
        return list(self.hms.get_all(filters=filters, top_k=top_k)["results"])

    def axis_query(self, aq: AxisQuery | dict) -> AxisResult | None:
        """HMS-native seven-axis query; None when the axis plane is off.
        HMS 原生七轴查询；轴平面关闭时为 None。

        Input / 输入：
            aq: AxisQuery or its dict form.
            aq：AxisQuery 或其字典形式。

        Output / 输出：
            AxisResult or None.
            AxisResult 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        if not self.switches.use_axis:
            return None
        return self.hms.axis_query(aq)

    # ------------------------------------------------------------------ graph plane
    # ------------------------------------------------------------------ 图平面
    def why(self, target_id: str, *, filters: dict | None = None, hops: int = 2) -> dict | None:
        """Backward explanation over the Support Graph; None when the graph plane is off.
        在支持图上反向解释；图平面关闭时为 None。

        Input / 输入：
            target_id: Object id (or a question when filters are given).
            target_id：对象 id（给定 filters 时可为问题）。

            filters: Scope filters enabling question mode.
            filters：启用问题模式的范围过滤。

            hops: Expansion levels.
            hops：展开层数。

        Output / 输出：
            HMS.why() dict or None.
            HMS.why() 的字典或 None。

        State effect / 状态影响：
            None.
            无。
        """
        if not self.switches.use_graph:
            return None
        return self.hms.why(target_id, filters=filters, hops=hops)

    # ------------------------------------------------------------------ transition layer
    # ------------------------------------------------------------------ 转移层
    def transitions(self, *, filters: dict) -> list[dict]:
        """Transition records restricted to the kinds this ablation's planes can account for.
        限制为本消融各平面能够解释的种类的转移记录。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：含至少一个范围键的 Mem0 过滤字典。

        Output / 输出：
            Transition dicts (HMS.transitions fmt='records').
            转移字典（HMS.transitions fmt='records'）。

        State effect / 状态影响：
            Idempotent transition materialization inside HMS.transitions (INSERT OR IGNORE).
            HMS.transitions 内部的幂等转移物化（INSERT OR IGNORE）。
        """
        kinds = list(self.switches.transition_kinds)
        return list(self.hms.transitions(filters=filters, fmt="records", kinds=kinds))

    # ------------------------------------------------------------------ graph-only internals
    # ------------------------------------------------------------------ 仅图消融内部实现
    def _scope(self, filters: dict) -> Scope:
        """Scope of a filter dict (VALIDATION_001 when none).
        过滤字典的范围（无范围时 VALIDATION_001）。
        """
        # Imported here (not at module level) so hms_elite.eval stays outside the §0.4 plane edges like a facade user.
        # 在此处（而非模块级）导入，使 hms_elite.eval 像门面使用者一样处于 §0.4 平面边之外。
        from hms_elite.mvmi.filters import parse_filters

        scope, _expr = parse_filters(filters, require_scope=True, context="ablation")
        return scope

    def _graph_seeds(self, scope: Scope) -> list[str]:
        """Newest valid state-layer nodes of the scope (EP / InterfaceEP / skill), else newest nodes of any kind.
        范围内最新的有效状态层节点（EP / InterfaceEP / 技能），否则任意种类的最新节点。
        """
        nodes = self.hms.store.list_nodes(scope, valid_only=True)
        preferred = [n for n in nodes if str(n.node_kind) in _SEED_KINDS]
        chosen = preferred or nodes
        return [n.node_id for n in chosen[-GRAPH_ONLY_SEEDS:]][::-1]

    def _graph_reach(self, scope: Scope) -> tuple[list[str], list[dict], dict[str, int]]:
        """SupportGraph.neighbors expansion from the seeds: (node ids in reach order, edge dicts, hop distance per node).
        从种子出发的 SupportGraph.neighbors 展开：（按到达顺序的节点 id、边字典、每个节点的跳数）。
        """
        seeds = self._graph_seeds(scope)
        if not seeds:
            return [], [], {}
        nodes, edges, paths = self.hms.graph.neighbors(
            seeds, hops=GRAPH_ONLY_HOPS, edge_types=WHY_EDGE_TYPES, valid_only=False
        )
        node_ids = [n.node_id for n in nodes]
        hops = {node_id: len(paths.get(node_id, [])) for node_id in node_ids}
        return node_ids, [e.model_dump() for e in edges], hops

    def _graph_rows(self, query: str, *, filters: dict) -> list[dict]:
        """Mem0-shaped rows of the live atoms in graph reach, newest first (no axis scores).
        图可达范围内存活原子的 Mem0 形状行，最新在前（无轴分数）。
        """
        scope = self._scope(filters)
        node_ids, _edges, _hops = self._graph_reach(scope)
        atoms = [a for a in self.hms.store.get_atoms(node_ids) if a.archived_reason is None and a.valid_to is None]
        atoms.sort(key=lambda a: (a.time_anchor or a.created_at or "", a.atom_id), reverse=True)
        return [
            {
                "id": atom.memory_id,
                "memory": atom.text,
                "atom_id": atom.atom_id,
                "atom_type": str(atom.atom_type),
                "hms": {"entity_ids": list(atom.entity_ids), "source": "graph", "query": query},
            }
            for atom in atoms
            if str(atom.atom_type) != AtomType.entity.value
        ]

    def _graph_compile(self, query: str, *, filters: dict, budget_tokens: int | None, as_of: Any) -> ContextIR:
        """Graph-only Work State: slots filled from graph reach, ordered by time, no axis score, no budget packing.
        仅图工作态：槽位由图可达集合填充，按时间排序，无轴分数，不做预算打包。
        """
        # Slot rules and item builders are the compiler's public helpers; imported lazily for the same §0.4 reason.
        # 槽位规则与条目构造器是编译器的公开辅助函数；出于同样的 §0.4 原因延迟导入。
        from hms_elite.compiler import assign_slot, derive_next_action, evidence_items, item_from_atom

        store = self.hms.store
        scope = self._scope(filters)
        instant = str(as_of) if as_of is not None else store.now()
        node_ids, edges, hop_of = self._graph_reach(scope)
        seeds = [node_id for node_id, distance in hop_of.items() if distance == 0]
        atoms = {a.atom_id: a for a in store.get_atoms(node_ids)}
        # Only VALID revision edges explain a supersession (an invalidated feedback edge must not re-archive the
        # restored version after a rollback); conflict edges are likewise read only while valid.
        # 只有有效的 revision 边才解释替代关系（回滚后失效的反馈边不能再把恢复的版本归档）；conflict 边同样只在有效时读取。
        live_edges = [e for e in edges if e.get("valid", True)]
        superseded: dict[str, str] = {}
        for edge in live_edges:
            if edge["edge_type"] == EdgeType.revision.value and edge["dst_id"] in atoms and edge["src_id"] in atoms:
                superseded[edge["src_id"]] = atoms[edge["dst_id"]].text
        slots: dict[str, list[ContextItem]] = {s.value: [] for s in Slot}
        for node_id in node_ids:
            atom = atoms.get(node_id)
            if atom is None or str(atom.atom_type) == AtomType.entity.value:
                continue
            anchor_ids = store.atom_anchor_ids(atom.atom_id, include_disputed=True)
            if not anchor_ids:
                continue
            revised = atom.atom_id in superseded and atom.archived_reason is not None
            slot = assign_slot(atom, as_of=instant, reason="revised" if revised else None)
            explain: dict[str, Any] = {
                "disputed_anchor_ids": [a for a in anchor_ids if a not in store.atom_anchor_ids(atom.atom_id)]
            }
            if revised:
                explain["reason"] = "revised"
                explain["superseded_by_text"] = superseded[atom.atom_id]
            slots[slot.value].append(
                item_from_atom(
                    atom,
                    slot=slot,
                    score=0.0,
                    anchor_ids=anchor_ids,
                    source="graph",
                    hops=hop_of.get(atom.atom_id, 0),
                    explain=explain,
                )
            )
        for name, items in slots.items():
            items.sort(key=lambda i: (i.time_anchor is None, i.time_anchor or "", i.item_id))
        cited = sorted({a for items in slots.values() for i in items for a in i.anchor_ids})
        anchors = store.get_anchors(cited)
        evidence = evidence_items(anchors)
        slots[Slot.audit.value] = evidence + slots[Slot.audit.value]
        unresolved = [
            e["edge_id"]
            for e in live_edges
            if e["edge_type"] == EdgeType.conflict.value
            and e.get("resolved_at") is None
            and e["src_id"] in atoms
            and e["dst_id"] in atoms
        ]
        conflicts = [
            (atoms[e["src_id"]].text, atoms[e["dst_id"]].text) for e in live_edges if e["edge_id"] in unresolved
        ]
        next_action = derive_next_action(
            episodes=slots[Slot.episodes.value],
            constraints=slots[Slot.constraints.value],
            residuals=[],
            conflicts=conflicts,
            goal=query,
        )
        budget = int(budget_tokens or self.hms.config.compiler.budget_tokens)
        used = sum(i.tokens for items in slots.values() for i in items)
        return ContextIR(
            query=query,
            goal=query,
            as_of=instant,
            scope=scope,
            mode=CompileMode.auto,
            route=Route.detective,
            facts=slots[Slot.facts.value],
            episodes=slots[Slot.episodes.value],
            preferences=slots[Slot.preferences.value],
            constraints=slots[Slot.constraints.value],
            skills=slots[Slot.skills.value],
            audit=slots[Slot.audit.value],
            next_action=next_action,
            budget_tokens=budget,
            used_tokens=used,
            unresolved_conflicts=unresolved,
            retrieval={
                "ablation": self.ablation.value,
                "route": Route.detective.value,
                "mode": None,
                "graph_hops": GRAPH_ONLY_HOPS,
                "seeds": seeds,
                "candidates": len(node_ids),
                "anchors": [a.model_dump() for a in anchors],
                "audience": None,
                "signals": [],
            },
        )


# ---------------------------------------------------------------------------
# Report
# 报告
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    """One task × ablation outcome.
    一个任务 × 消融的结果。
    """

    task: str
    ablation: str
    score: float
    checks: list[dict] = field(default_factory=list)
    latency_ms: float | None = None
    llm_calls: int | None = None
    tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Plain dict of the cell.
        单元格的普通字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with task, ablation, score, checks, latency_ms, llm_calls, tokens, error.
            含 task、ablation、score、checks、latency_ms、llm_calls、tokens、error 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "task": self.task,
            "ablation": self.ablation,
            "score": self.score,
            "checks": [dict(c) for c in self.checks],
            "latency_ms": self.latency_ms,
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "error": self.error,
        }


@dataclass
class AblationReport:
    """The task × ablation matrix produced by run_ablation.
    run_ablation 产出的任务 × 消融矩阵。
    """

    tasks: list[str]
    ablations: list[str]
    cells: list[Cell]

    def cell(self, task: str, ablation: Ablation | str) -> Cell:
        """The cell of a task × ablation (KeyError when absent).
        某任务 × 消融的单元格（不存在时 KeyError）。

        Input / 输入：
            task: Task name.
            task：任务名。

            ablation: Ablation member or value.
            ablation：Ablation 成员或值。

        Output / 输出：
            Cell.
            Cell。

        State effect / 状态影响：
            None.
            无。
        """
        label = str(getattr(ablation, "value", ablation))
        for c in self.cells:
            if c.task == task and c.ablation == label:
                return c
        raise KeyError((task, label))

    def score(self, task: str, ablation: Ablation | str) -> float:
        """Score of a task × ablation.
        某任务 × 消融的分数。

        Input / 输入：
            task: Task name.
            task：任务名。

            ablation: Ablation member or value.
            ablation：Ablation 成员或值。

        Output / 输出：
            Float in [0, 1].
            [0, 1] 内的浮点数。

        State effect / 状态影响：
            None.
            无。
        """
        return self.cell(task, ablation).score

    def mean(self, ablation: Ablation | str) -> float:
        """Mean score of an ablation over every task (0.0 when there are no tasks).
        某消融在全部任务上的平均分（无任务时 0.0）。

        Input / 输入：
            ablation: Ablation member or value.
            ablation：Ablation 成员或值。

        Output / 输出：
            Float rounded to 4 decimals.
            四舍五入到 4 位小数的浮点数。

        State effect / 状态影响：
            None.
            无。
        """
        label = str(getattr(ablation, "value", ablation))
        scores = [c.score for c in self.cells if c.ablation == label]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict:
        """Plain dict form (JSON-serializable).
        普通字典形式（可 JSON 序列化）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'tasks', 'ablations', 'cells': [Cell dicts]}.
            {'tasks', 'ablations', 'cells': [Cell 字典]}。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "tasks": list(self.tasks),
            "ablations": list(self.ablations),
            "cells": [c.to_dict() for c in self.cells],
        }

    def to_json(self, *, timing: bool = True) -> str:
        """JSON export (sorted keys, ensure_ascii=False); timing=False blanks latency for byte-stable output.
        JSON 导出（键排序、ensure_ascii=False）；timing=False 抹去延迟以得到逐字节稳定的输出。

        Input / 输入：
            timing: Keep latency_ms values.
            timing：是否保留 latency_ms。

        Output / 输出：
            JSON string.
            JSON 字符串。

        State effect / 状态影响：
            None.
            无。
        """
        data = self.to_dict()
        if not timing:
            for cell in data["cells"]:
                cell["latency_ms"] = None
        return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)


def _usage_snapshots(hms: Any) -> list[Any]:
    """Usage snapshots of every metered provider on the facade (llm, embedder) — hms_elite.telemetry.UsageMeter.
    门面上每个带计量的提供者（llm、embedder）的 Usage 快照 —— hms_elite.telemetry.UsageMeter。
    """
    out: list[Any] = []
    for name in ("llm", "embedder"):
        meter = getattr(getattr(hms, name, None), "usage", None)
        snapshot = getattr(meter, "snapshot", None)
        if callable(snapshot):
            out.append(snapshot())
    return out


def _llm_call_count(hms: Any) -> int | None:
    """LLM calls so far: the telemetry meter when present, else the provider's `calls` log (RuleLLM), else None.
    迄今的 LLM 调用数：有 telemetry 计量器时取之，否则取提供者的 `calls` 日志（RuleLLM），再否则 None。
    """
    meter = getattr(getattr(hms, "llm", None), "usage", None)
    snapshot = getattr(meter, "snapshot", None)
    if callable(snapshot):
        value = getattr(snapshot(), "llm_calls", None)
        if isinstance(value, int):
            return value
    calls = getattr(getattr(hms, "llm", None), "calls", None)
    return len(calls) if isinstance(calls, list) else None


def _token_count(hms: Any) -> int | None:
    """Total tokens metered so far across the facade's providers, or None when no telemetry meter exists.
    门面各提供者迄今计量的 token 总数；没有 telemetry 计量器时为 None。
    """
    snapshots = _usage_snapshots(hms)
    totals = [getattr(u, "total_tokens", None) for u in snapshots]
    if not totals or not all(isinstance(t, int) for t in totals):
        return None
    return sum(totals)


def _delta(before: int | None, after: int | None) -> int | None:
    """after − before when both are known, else None.
    两者均已知时返回 after − before，否则 None。
    """
    if before is None or after is None:
        return None
    return after - before


def _run_cell(hms: Any, task: Task, ablation: Ablation, ctx: dict) -> Cell:
    """Query + score one task under one ablation, timing it and counting provider calls.
    在一种消融下查询并评分一个任务，计时并统计提供者调用数。
    """
    probe = Probe(hms, ablation)
    calls0, tokens0 = _llm_call_count(hms), _token_count(hms)
    started = time.perf_counter()
    try:
        result = task.run(probe, ctx)
    except Exception as exc:  # noqa: BLE001 - a failing task is a report row, not a crash
        elapsed = (time.perf_counter() - started) * 1000.0
        return Cell(
            task=task.name,
            ablation=ablation.value,
            score=0.0,
            checks=[],
            latency_ms=round(elapsed, 3),
            llm_calls=_delta(calls0, _llm_call_count(hms)),
            tokens=_delta(tokens0, _token_count(hms)),
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    return Cell(
        task=task.name,
        ablation=ablation.value,
        score=result.value,
        checks=[{"name": c.name, "score": c.score, "hit": c.hit, "detail": c.detail} for c in result.checks],
        latency_ms=round(elapsed, 3),
        llm_calls=_delta(calls0, _llm_call_count(hms)),
        tokens=_delta(tokens0, _token_count(hms)),
        error=None,
    )


def run_ablation(
    hms_factory: Callable[[], Any],
    tasks: Sequence[Task] | None = None,
    ablations: Sequence[Ablation | str] | None = None,
) -> AblationReport:
    """Run every task × ablation: one fresh HMS per task (built once), the four read-only probes over it.
    运行每个任务 × 消融：每个任务一个新 HMS（只构建一次），四种只读探针在其上运行。

    Input / 输入：
        hms_factory: Zero-argument callable returning a fresh HMS (closed after the task).
        hms_factory：无参可调用对象，返回一个新 HMS（任务结束后关闭）。

        tasks: Task objects (None → hms_elite.eval.tasks.TASKS, the nine §10 tasks).
        tasks：Task 对象（None → hms_elite.eval.tasks.TASKS，即 §10 的九个任务）。

        ablations: Ablation members/values (None → all four in enum order).
        ablations：Ablation 成员/取值（None → 按枚举顺序的全部四种）。

    Output / 输出：
        AblationReport; a task whose build fails yields error cells for every ablation.
        AblationReport；构建失败的任务在每种消融下都得到错误单元格。

    State effect / 状态影响：
        Writes only into the HMS instances the factory creates; closes each after use.
        只写入工厂创建的 HMS 实例；用后逐个关闭。
    """
    if tasks is None:
        from hms_elite.eval.tasks import TASKS

        tasks = TASKS
    chosen = [Ablation(str(getattr(a, "value", a))) for a in (ablations if ablations is not None else list(Ablation))]
    cells: list[Cell] = []
    for task in tasks:
        hms = hms_factory()
        try:
            try:
                ctx = task.build(hms)
            except Exception as exc:  # noqa: BLE001 - a failing build is reported per ablation
                message = f"build failed: {type(exc).__name__}: {exc}"
                cells.extend(Cell(task=task.name, ablation=a.value, score=0.0, error=message) for a in chosen)
                continue
            for ablation in chosen:
                cells.append(_run_cell(hms, task, ablation, ctx))
        finally:
            close = getattr(hms, "close", None)
            if callable(close):
                close()
    return AblationReport(tasks=[t.name for t in tasks], ablations=[a.value for a in chosen], cells=cells)


__all__ = [
    "GRAPH_ONLY_HOPS",
    "GRAPH_ONLY_SEEDS",
    "MODE_OF_ABLATION",
    "Ablation",
    "AblationReport",
    "AblationSwitches",
    "Cell",
    "Check",
    "Probe",
    "Task",
    "TaskScore",
    "run_ablation",
]
