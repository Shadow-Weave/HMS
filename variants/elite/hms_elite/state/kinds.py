"""State kinds of HMS §2 / §11: one kind = one slot-selection rule + one render template + kind-specific fields.
HMS §2 / §11 的状态种类：一个种类 = 一条槽位选择规则 + 一个渲染模板 + 一组该状态特有的字段。
"""

# Purpose (EN): Own the StateKind enum, the StateSpec registry and every state model (Work / Personal / Edge / Group /
# Enterprise), all built on ContextIR items and MemoryAtoms; every entry carries anchor_ids (No Witness, No Meaning).
# 目的（中）：拥有 StateKind 枚举、StateSpec 注册表与全部状态模型（Work / Personal / Edge / Group / Enterprise），
# 全部建立在 ContextIR 项与 MemoryAtom 之上；每一项都带 anchor_ids（无见证，即无意义）。

# Input (EN): ContextIR, StatePackage, Checkpoint, MemoryAtom, GraphEdge, Anchor and the why() dict.
# 输入（中）：ContextIR、StatePackage、Checkpoint、MemoryAtom、GraphEdge、Anchor 与 why() 字典。

# Output (EN): Validated state models plus the pure item constructors used by the builders.
# 输出（中）：已校验的状态模型，以及构建器使用的纯项构造函数。

# State effect (EN): None (pure models and constructors; no store access).
# 状态影响（中）：无（纯模型与构造函数；不访问存储）。

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from hms_elite.enums import Reward, StrEnum, Visibility
from hms_elite.models import (
    Anchor,
    Checkpoint,
    ContextIR,
    ContextItem,
    HMSModel,
    MemoryAtom,
    Scope,
    StatePackage,
    UtcIso,
    Witnessed,
)

SCOPE_KEYS = ("user_id", "agent_id", "run_id")
_NON_NEGATIVE = frozenset({Reward.positive.value, Reward.neutral.value, Reward.unknown.value})


class StateKind(StrEnum):
    """The twelve state kinds named by HMS §2 and §11.
    HMS §2 与 §11 命名的十二种状态种类。
    """

    work_state = "work_state"
    handoff_package = "handoff_package"
    detective_report = "detective_report"
    rollback_point = "rollback_point"
    personal_state = "personal_state"
    shared_card = "shared_card"
    edge_state = "edge_state"
    group_state = "group_state"
    customer_issue_state = "customer_issue_state"
    incident_state = "incident_state"
    meeting_decision_state = "meeting_decision_state"
    account_handoff_state = "account_handoff_state"


class StateSpec(HMSModel):
    """What defines a kind: the slot rule per field, the render template (section titles) and the field list.
    定义一个种类的三件事：每个字段的槽位规则、渲染模板（章节标题）与字段列表。
    """

    kind: StateKind
    title: str
    # 字段名 → 从原子/边/锚点推导该字段的规则（确定性文本）
    slot_rule: dict[str, str]  # field → deterministic derivation rule over atoms / edges / anchors
    # 渲染模板：按顺序输出的章节标题
    template: list[str]  # section titles in render order
    fields: list[str]


class StateItem(Witnessed):
    """One witnessed entry of a state: text + time + provenance + anchor_ids.
    状态中的一条带见证条目：文本 + 时间 + 出处 + anchor_ids。
    """

    text: str = Field(min_length=1)
    atom_id: str | None = None
    memory_id: str | None = None
    time: UtcIso | None = None
    valid_from: UtcIso | None = None
    valid_to: UtcIso | None = None
    source_kind: str | None = None
    atom_type: str | None = None
    obs_type: str | None = None
    subject: str | None = None
    predicate: str | None = None
    visibility: Visibility = Visibility.private
    # 种类特有的附加键（如 superseded_by、member、side、r）
    extra: dict[str, Any] = Field(default_factory=dict)  # kind-specific keys (superseded_by, member, side, r, ...)


class RelationEdge(Witnessed):
    """A support-graph edge between two endpoints of an EdgeState, cited by its witnesses.
    EdgeState 两端之间的一条支持图边，以其见证引用。
    """

    edge_id: str
    edge_type: str
    src_id: str
    dst_id: str
    src_text: str
    dst_text: str
    label: str = ""
    confidence: float = 0.0


class MemberConflict(Witnessed):
    """Two members' statements that cannot both hold (same subject + exclusive predicate, different objects).
    两个成员无法同时成立的陈述（同主语 + 排他谓词，宾语不同）。
    """

    members: list[str]
    left: StateItem
    right: StateItem
    reason: str


class BaseState(HMSModel):
    """Common head of every state: kind, scope, as_of.
    每种状态的公共头：kind、scope、as_of。
    """

    kind: StateKind
    scope: Scope
    as_of: UtcIso

    def items(self) -> list[StateItem]:
        """Every StateItem the state carries, in field order.
        该状态携带的全部 StateItem，按字段顺序。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Flat list of StateItem.
            StateItem 的扁平列表。

        State effect / 状态影响：
            None.
            无。
        """
        out: list[StateItem] = []
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, StateItem):
                out.append(value)
            elif isinstance(value, ContextIR):
                # A wrapped compile() contributes its six slots as witnessed items.
                # 被包装的 compile() 以其六个槽位贡献带见证的项。
                out.extend(item_from_context_item(i) for i in value.items())
            elif isinstance(value, StatePackage):
                out.extend(item_from_context_item(i) for i in value.context.items())
            elif isinstance(value, list):
                out.extend(v for v in value if isinstance(v, StateItem))
                for v in value:
                    if isinstance(v, MemberConflict):
                        out.extend((v.left, v.right))
        return out

    def witness_ids(self) -> list[str]:
        """Sorted union of every witnessed entry's anchor ids.
        全部带见证条目锚点 id 的排序并集。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Sorted anchor id list.
            排序后的锚点 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        ids: set[str] = set()
        # A state that is itself witnessed (RollbackPoint) contributes its own anchor list.
        # 自身带见证的状态（RollbackPoint）贡献它自己的锚点列表。
        own = getattr(self, "anchor_ids", None)
        if isinstance(own, list):
            ids.update(str(a) for a in own)
        for name in type(self).model_fields:
            value = getattr(self, name)
            values = value if isinstance(value, list) else [value]
            for v in values:
                if isinstance(v, Witnessed):
                    ids.update(v.anchor_ids)
                elif isinstance(v, ContextIR):
                    ids.update(v.anchor_ids())
                elif isinstance(v, StatePackage):
                    ids.update(a.anchor_id for a in v.anchors)
                elif isinstance(v, BaseState):
                    ids.update(v.witness_ids())
        return sorted(ids)


# ---------------------------------------------------------------------------
# §11.1 Agent WorkState
# §11.1 Agent 工作态
# ---------------------------------------------------------------------------
class WorkState(BaseState):
    """The compiled Work State: wraps compile() — the ContextIR is the state, the card is its rendering.
    编译后的工作态：包装 compile() —— ContextIR 即状态，卡片是其渲染。
    """

    kind: Literal["work_state"] = StateKind.work_state.value
    context: ContextIR
    goal: str | None = None
    next_action: str | None = None

    @classmethod
    def from_context(cls, ir: ContextIR) -> "WorkState":
        """Wrap a compiled ContextIR.
        包装一个已编译的 ContextIR。

        Input / 输入：
            ir: The ContextIR returned by HMS.compile.
            ir：HMS.compile 返回的 ContextIR。

        Output / 输出：
            WorkState.
            WorkState。

        State effect / 状态影响：
            None.
            无。
        """
        return cls(scope=ir.scope, as_of=ir.as_of, context=ir, goal=ir.goal or ir.query, next_action=ir.next_action)


class RollbackPoint(BaseState):
    """A checkpoint seen as a state: where rollback() can return to, witnessed by the checkpoint anchor.
    作为状态的检查点：rollback() 可回到的位置，由检查点锚点见证。
    """

    kind: Literal["rollback_point"] = StateKind.rollback_point.value
    checkpoint_id: str
    label: str | None = None
    created_at: UtcIso
    history_seq: int
    snapshot_hash: str
    anchor_ids: list[str] = Field(min_length=1)
    atoms_live: int = 0
    edges_valid: int = 0

    @classmethod
    def from_checkpoint(cls, cp: Checkpoint) -> "RollbackPoint":
        """Build from a stored Checkpoint.
        由已存储的 Checkpoint 构建。

        Input / 输入：
            cp: The checkpoint row.
            cp：检查点行。

        Output / 输出：
            RollbackPoint.
            RollbackPoint。

        State effect / 状态影响：
            None.
            无。
        """
        live = sum(1 for st in cp.atom_state.values() if st.get("archived_reason") is None)
        return cls(
            scope=cp.scope,
            as_of=cp.created_at,
            checkpoint_id=cp.checkpoint_id,
            label=cp.label,
            created_at=cp.created_at,
            history_seq=cp.history_seq,
            snapshot_hash=cp.snapshot_hash,
            anchor_ids=[cp.anchor_id],
            atoms_live=live,
            edges_valid=sum(1 for v in cp.edge_state.values() if v),
        )


class HandoffPackageState(BaseState):
    """The handoff package as a state: the StatePackage plus its rollback point.
    作为状态的交接包：StatePackage 加上其回滚点。
    """

    kind: Literal["handoff_package"] = StateKind.handoff_package.value
    package: StatePackage
    rollback_point: RollbackPoint | None = None

    @classmethod
    def from_package(cls, pkg: StatePackage, rollback_point: RollbackPoint | None = None) -> "HandoffPackageState":
        """Wrap a StatePackage (and the checkpoint it was taken after, when known).
        包装 StatePackage（以及已知时的交接前检查点）。

        Input / 输入：
            pkg: The StatePackage built by HMS.handoff.
            pkg：HMS.handoff 构建的 StatePackage。

            rollback_point: The RollbackPoint of pkg.checkpoint_id, or None.
            rollback_point：pkg.checkpoint_id 对应的 RollbackPoint，或 None。

        Output / 输出：
            HandoffPackageState.
            HandoffPackageState。

        State effect / 状态影响：
            None.
            无。
        """
        return cls(scope=pkg.scope, as_of=pkg.created_at, package=pkg, rollback_point=rollback_point)


class DetectiveReport(BaseState):
    """why() as a state: question → conclusion → evidence chain → anchored evidence → unresolved conflicts.
    作为状态的 why()：问题 → 结论 → 证据链 → 带锚点的证据 → 未解冲突。
    """

    kind: Literal["detective_report"] = StateKind.detective_report.value
    question: str
    conclusion: StateItem
    # why() 的 chain 行（确定性）
    chain: list[str]  # the deterministic why() chain lines
    steps: list[StateItem]
    evidence: list[StateItem]
    open_conflicts: list[StateItem] = Field(default_factory=list)
    refined_from: list[str] = Field(default_factory=list)

    @classmethod
    def from_why(cls, why: dict, *, question: str, scope: Scope, as_of: str) -> "DetectiveReport":
        """Build from the HMS.why() dict (its 'anchors' list is the evidence; no store access needed).
        由 HMS.why() 字典构建（其 'anchors' 列表即证据；无需访问存储）。

        Input / 输入：
            why: The why() result dict.
            why：why() 结果字典。

            question: The question or target id that was asked.
            question：所问的问题或目标 id。

            scope: Scope of the report.
            scope：报告的范围。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            DetectiveReport; raises ValueError when the why() result carries no anchors.
            DetectiveReport；why() 结果不含锚点时抛出 ValueError。

        State effect / 状态影响：
            None.
            无。
        """
        anchors = [Anchor(**a) for a in why.get("anchors") or []]
        anchor_map = {a.anchor_id: a for a in anchors}
        if not anchors:
            raise ValueError("DetectiveReport needs at least one anchor in the why() result")
        all_ids = sorted(anchor_map)
        target = dict(why.get("target") or {})
        node_anchor: dict[str, list[str]] = {}
        for node in why.get("nodes") or []:
            node_anchor[str(node.get("node_id"))] = (
                sorted(a for a in (node.get("anchor_ids") or []) if a in anchor_map) or all_ids
            )
        target_node = str(target.get("node_id") or (target.get("node_ids") or [""])[-1])
        conclusion = StateItem(
            text=str(target.get("text") or target_node or "?"),
            atom_id=target_node or None,
            anchor_ids=node_anchor.get(target_node) or all_ids,
            extra={"kind": target.get("kind"), "resolved_from": target.get("resolved_from")},
        )
        steps: list[StateItem] = []
        for node in why.get("nodes") or []:
            node_id = str(node.get("node_id"))
            steps.append(
                StateItem(
                    text=str(node.get("text") or node_id),
                    atom_id=node_id,
                    time=node.get("created_at"),
                    anchor_ids=node_anchor[node_id],
                    extra={"node_kind": node.get("node_kind"), "scale": node.get("scale"), "valid": node.get("valid")},
                )
            )
        evidence = [
            StateItem(
                text=a.evidence_line(),
                time=a.timestamp,
                anchor_ids=[a.anchor_id],
                visibility=a.permission_scope,
                extra={"source_ref": a.source_ref, "source_event_id": a.source_event_id, "revoked_at": a.revoked_at},
            )
            for a in sorted(anchors, key=lambda a: (a.timestamp, a.anchor_id))
        ]
        edge_map = {str(e.get("edge_id")): e for e in why.get("edges") or []}
        conflicts: list[StateItem] = []
        for edge_id in why.get("unresolved_conflicts") or []:
            e = edge_map.get(str(edge_id))
            if e is None:
                continue
            conflicts.append(
                StateItem(
                    text=str(e.get("label") or f"conflict: {e.get('src_id')} ⟂ {e.get('dst_id')}"),
                    anchor_ids=[a for a in e.get("witness_anchor_ids") or [] if a in anchor_map] or all_ids,
                    extra={"edge_id": edge_id, "src_id": e.get("src_id"), "dst_id": e.get("dst_id")},
                )
            )
        return cls(
            scope=scope,
            as_of=as_of,
            question=question,
            conclusion=conclusion,
            chain=[str(line) for line in why.get("chain") or []],
            steps=steps,
            evidence=evidence,
            open_conflicts=conflicts,
            refined_from=[str(i) for i in why.get("refined_from") or []],
        )


# ---------------------------------------------------------------------------
# §11.2 Personal API / RESONANCE
# §11.2 个人 API / 共鸣
# ---------------------------------------------------------------------------
class PersonalState(BaseState):
    """The state of one subject: valid + invalidated preferences (with times), habits, constraints, facts, timeline.
    一个主体的状态：有效 + 失效偏好（带时间）、习惯、约束、资料事实、时间线。
    """

    kind: Literal["personal_state", "shared_card"] = StateKind.personal_state.value
    subject: str
    # 受众；None 表示不过滤可见性
    audience: Visibility | None = None  # None → no visibility gate
    preferences_valid: list[StateItem] = Field(default_factory=list)
    preferences_invalid: list[StateItem] = Field(default_factory=list)
    habits: list[StateItem] = Field(default_factory=list)
    constraints: list[StateItem] = Field(default_factory=list)
    profile_facts: list[StateItem] = Field(default_factory=list)
    timeline: list[StateItem] = Field(default_factory=list)


class EdgeState(BaseState):
    """The state between two endpoints (two entity keys or two user scopes).
    两个端点（两个实体键或两个用户范围）之间的状态。
    """

    kind: Literal["edge_state"] = StateKind.edge_state.value
    a: str
    b: str
    endpoint_kind: Literal["entity", "scope"]
    shared_facts: list[StateItem] = Field(default_factory=list)
    # 按时间升序
    interactions: list[StateItem] = Field(default_factory=list)  # ascending by time
    last_contact: UtcIso | None = None
    relation_edges: list[RelationEdge] = Field(default_factory=list)
    open_conflicts: list[StateItem] = Field(default_factory=list)


class GroupState(BaseState):
    """The state of a member set: shared constraints, decisions, conflicts between members.
    一组成员的状态：共享约束、决策、成员间冲突。
    """

    kind: Literal["group_state"] = StateKind.group_state.value
    members: list[str]
    shared_constraints: list[StateItem] = Field(default_factory=list)
    decisions: list[StateItem] = Field(default_factory=list)
    conflicts_between_members: list[MemberConflict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §11.3 Enterprise State
# §11.3 企业状态
# ---------------------------------------------------------------------------
class CustomerIssueState(BaseState):
    """issue / status / owner / failed_paths / next_action, all derived from the compiled Work State.
    issue / status / owner / failed_paths / next_action，全部由编译后的工作态推导。
    """

    kind: Literal["customer_issue_state"] = StateKind.customer_issue_state.value
    issue: StateItem
    status: StateItem
    owner: StateItem | None = None
    failed_paths: list[StateItem] = Field(default_factory=list)
    constraints: list[StateItem] = Field(default_factory=list)
    next_action: StateItem | None = None
    context: ContextIR


class IncidentState(BaseState):
    """timeline / root_cause (the why() chain) / constraints / rollback_point.
    timeline / root_cause（why() 链）/ constraints / rollback_point。
    """

    kind: Literal["incident_state"] = StateKind.incident_state.value
    timeline: list[StateItem] = Field(default_factory=list)
    root_cause: DetectiveReport | None = None
    constraints: list[StateItem] = Field(default_factory=list)
    rollback_point: RollbackPoint | None = None


class MeetingDecisionState(BaseState):
    """decisions / owners / deadlines / superseded.
    决策 / 负责人 / 截止 / 被取代项。
    """

    kind: Literal["meeting_decision_state"] = StateKind.meeting_decision_state.value
    decisions: list[StateItem] = Field(default_factory=list)
    owners: list[StateItem] = Field(default_factory=list)
    deadlines: list[StateItem] = Field(default_factory=list)
    superseded: list[StateItem] = Field(default_factory=list)


class AccountHandoffState(BaseState):
    """The handoff package plus the customer constraints of the account.
    交接包加上该账户的客户约束。
    """

    kind: Literal["account_handoff_state"] = StateKind.account_handoff_state.value
    package: StatePackage
    customer_constraints: list[StateItem] = Field(default_factory=list)
    rollback_point: RollbackPoint | None = None


# ---------------------------------------------------------------------------
# Pure constructors shared by the builders
# 构建器共用的纯构造函数
# ---------------------------------------------------------------------------
def scope_from_filters(filters: dict | None) -> Scope:
    """The Scope pinned by the scalar scope keys of a Mem0 filter dict (VALIDATION_001 when none).
    由 Mem0 过滤字典的标量范围键钉住的 Scope（没有时 VALIDATION_001）。

    Input / 输入：
        filters: Mem0 filter dict (user_id / agent_id / run_id plus axis keys).
        filters：Mem0 过滤字典（user_id / agent_id / run_id 与轴键）。

    Output / 输出：
        Scope with at least one id.
        至少含一个 id 的 Scope。

    State effect / 状态影响：
        None.
        无。
    """
    data = dict(filters or {})
    scope = Scope(**{k: data[k] for k in SCOPE_KEYS if isinstance(data.get(k), str)})
    return scope.require_any(context="state")


def item_from_atom(atom: MemoryAtom, **extra: Any) -> StateItem:
    """The StateItem of a MemoryAtom (anchor_ids = the atom's witnesses).
    MemoryAtom 的 StateItem（anchor_ids = 该原子的见证）。

    Input / 输入：
        atom: A hydrated atom.
        atom：已水合的原子。

        extra: Kind-specific keys stored under StateItem.extra.
        extra：存入 StateItem.extra 的种类特有键。

    Output / 输出：
        StateItem.
        StateItem。

    State effect / 状态影响：
        None.
        无。
    """
    return StateItem(
        text=atom.text,
        atom_id=atom.atom_id,
        memory_id=atom.memory_id or None,
        time=atom.time_anchor or atom.created_at,
        valid_from=atom.valid_from,
        valid_to=atom.valid_to,
        source_kind=str(atom.source_kind),
        atom_type=str(atom.atom_type),
        obs_type=None if atom.obs_type is None else str(atom.obs_type),
        subject=atom.subject,
        predicate=atom.predicate,
        visibility=atom.visibility,
        anchor_ids=list(atom.anchor_ids),
        extra=dict(extra),
    )


def item_from_context_item(item: ContextItem, **extra: Any) -> StateItem:
    """The StateItem of a ContextIR item (anchor_ids = the item's witnesses).
    ContextIR 项的 StateItem（anchor_ids = 该项的见证）。

    Input / 输入：
        item: A ContextItem.
        item：ContextItem。

        extra: Kind-specific keys merged over the item's explain dict.
        extra：合并到该项 explain 字典之上的种类特有键。

    Output / 输出：
        StateItem.
        StateItem。

    State effect / 状态影响：
        None.
        无。
    """
    merged = {k: v for k, v in item.explain.items() if k in ("r", "a", "y", "u", "reason", "kind", "ep_id")}
    merged.update(extra)
    return StateItem(
        text=item.text,
        atom_id=item.atom_id,
        memory_id=item.memory_id,
        time=item.time_anchor,
        valid_from=item.valid_from,
        valid_to=item.valid_to,
        source_kind=item.source,
        atom_type=None if item.atom_type is None else str(item.atom_type),
        visibility=item.visibility,
        anchor_ids=list(item.anchor_ids),
        extra=merged,
    )


def context_sections(ir: ContextIR) -> dict[str, list[ContextItem]]:
    """The §3.8 sections of a ContextIR: valid_facts, invalidated, failed_paths, constraints, evidence.
    ContextIR 的 §3.8 分区：valid_facts、invalidated、failed_paths、constraints、evidence。

    Input / 输入：
        ir: The compiled ContextIR.
        ir：已编译的 ContextIR。

    Output / 输出：
        Dict of section name → ContextItems in render order.
        分区名 → 按渲染顺序排列的 ContextItem 的字典。

    State effect / 状态影响：
        None.
        无。
    """
    valid = list(ir.facts) + [
        i for i in ir.episodes if str(i.explain.get("r") or Reward.unknown.value) in _NON_NEGATIVE
    ]
    failed = [i for i in ir.episodes if str(i.explain.get("r") or "") == Reward.negative.value]
    invalidated = [
        i
        for i in ir.audit
        if str(i.explain.get("reason") or "") == "revised" or str(i.explain.get("reason") or "").startswith("archived:")
    ]
    evidence = [i for i in ir.audit if i.explain.get("kind") == "evidence"]
    return {
        "valid_facts": valid,
        "invalidated": invalidated,
        "failed_paths": failed,
        "constraints": list(ir.constraints),
        "evidence": evidence,
    }


STATE_SPECS: dict[str, StateSpec] = {
    StateKind.work_state.value: StateSpec(
        kind=StateKind.work_state,
        title="Work State Card",
        slot_rule={
            "context": "HMS.compile(query, filters, mode) — six ContextIR slots",
            "goal": "ir.goal or ir.query",
            "next_action": "compiler.slots.derive_next_action over episodes + constraints",
        },
        template=[
            "Current Goal",
            "Valid Facts",
            "Invalidated Assumptions",
            "Failed Paths",
            "Constraints",
            "Next Action",
            "Evidence",
        ],  # fmt: skip
        fields=["context", "goal", "next_action"],
    ),
    StateKind.handoff_package.value: StateSpec(
        kind=StateKind.handoff_package,
        title="Handoff Package",
        slot_rule={
            "package": "HMS.handoff: deep compile + witness closure + evidence closure",
            "rollback_point": "store.get_checkpoint(package.checkpoint_id)",
        },
        template=[
            "Current Goal",
            "Valid Facts",
            "Invalidated Assumptions",
            "Failed Paths",
            "Constraints",
            "Next Action",
            "Evidence",
            "Rollback Point",
            "Checkpoint id",
        ],  # fmt: skip
        fields=["package", "rollback_point"],
    ),
    StateKind.detective_report.value: StateSpec(
        kind=StateKind.detective_report,
        title="Detective Report",
        slot_rule={
            "conclusion": "why().target",
            "chain": "why().chain (strongest incoming edge per hop)",
            "steps": "why().nodes with their anchors",
            "evidence": "why().anchors sorted by (timestamp, anchor_id)",
            "open_conflicts": "why().unresolved_conflicts",
        },
        template=["Question", "Conclusion", "Evidence Chain", "Evidence", "Unresolved Conflicts"],
        fields=["question", "conclusion", "chain", "steps", "evidence", "open_conflicts"],
    ),
    StateKind.rollback_point.value: StateSpec(
        kind=StateKind.rollback_point,
        title="Rollback Point",
        slot_rule={"checkpoint": "HMS.checkpoint / store.list_checkpoints(scope)[-1]"},
        template=["Checkpoint", "Snapshot", "Evidence"],
        fields=["checkpoint_id", "label", "created_at", "history_seq", "snapshot_hash", "atoms_live", "edges_valid"],
    ),
    StateKind.personal_state.value: StateSpec(
        kind=StateKind.personal_state,
        title="Personal Card",
        slot_rule={
            "preferences_valid": "obs_type=preference_event AND is_valid_at(as_of)",
            "preferences_invalid": "obs_type=preference_event AND NOT is_valid_at(as_of) (archived or expired)",
            "habits": "obs_type=habit AND live",
            "constraints": "obs_type=constraint AND live",
            "profile_facts": "obs_type IN (fact, temporal_fact) AND atom_type=state AND live",
            "timeline": "atom_type IN (event, episode) AND live, ascending by time",
            "audience": "rank(atom.visibility) >= rank(audience)",
        },
        template=[
            "Subject",
            "Preferences (valid)",
            "Preferences (invalidated)",
            "Habits",
            "Constraints",
            "Profile Facts",
            "Timeline",
            "Evidence",
        ],  # fmt: skip
        fields=[
            "subject",
            "audience",
            "preferences_valid",
            "preferences_invalid",
            "habits",
            "constraints",
            "profile_facts",
            "timeline",
        ],  # fmt: skip
    ),
    StateKind.shared_card.value: StateSpec(
        kind=StateKind.shared_card,
        title="Shared Card",
        slot_rule={"*": "PersonalState with audience >= team (visibility lattice gate)"},
        template=[
            "Subject",
            "Preferences (valid)",
            "Preferences (invalidated)",
            "Habits",
            "Constraints",
            "Profile Facts",
            "Timeline",
            "Evidence",
        ],  # fmt: skip
        fields=[
            "subject",
            "audience",
            "preferences_valid",
            "preferences_invalid",
            "habits",
            "constraints",
            "profile_facts",
            "timeline",
        ],  # fmt: skip
    ),
    StateKind.edge_state.value: StateSpec(
        kind=StateKind.edge_state,
        title="Edge Card",
        slot_rule={
            "shared_facts": "live non-event atoms linked to both endpoints (entity) / to a shared entity (scope)",
            "interactions": "live event/episode atoms linked to both endpoints, ascending by time",
            "last_contact": "max(time) over interactions ∪ shared_facts",
            "relation_edges": "valid use/causal edges touching the shared atoms",
            "open_conflicts": "unresolved conflict edges touching either endpoint / cross-scope exclusive conflicts",
        },
        template=[
            "Endpoints",
            "Shared Facts",
            "Interactions",
            "Last Contact",
            "Relation Edges",
            "Open Conflicts",
            "Evidence",
        ],  # fmt: skip
        fields=[
            "a",
            "b",
            "endpoint_kind",
            "shared_facts",
            "interactions",
            "last_contact",
            "relation_edges",
            "open_conflicts",
        ],  # fmt: skip
    ),
    StateKind.group_state.value: StateSpec(
        kind=StateKind.group_state,
        title="Group Card",
        slot_rule={
            "shared_constraints": "constraint atoms whose normalized text is held by >= 2 members",
            "decisions": "atoms matching DECISION_RE or positive-reward episodes, per member",
            "conflicts_between_members": "same subject + exclusive predicate, different objects, across members",
        },
        template=["Members", "Shared Constraints", "Decisions", "Conflicts Between Members", "Evidence"],
        fields=["members", "shared_constraints", "decisions", "conflicts_between_members"],
    ),
    StateKind.customer_issue_state.value: StateSpec(
        kind=StateKind.customer_issue_state,
        title="Customer Issue State",
        slot_rule={
            "issue": "goal (newest valid EP.u or the given issue) witnessed by that EP / the context",
            "status": "newest EP reward: positive→resolved, negative→blocked, else open",
            "owner": "actor_id of the newest structured event (else scope.agent_id)",
            "failed_paths": "negative-reward episodes of the compiled context",
            "next_action": "ir.next_action witnessed by episodes + constraints",
        },
        template=["Issue", "Status", "Owner", "Failed Paths", "Constraints", "Next Action", "Evidence"],
        fields=["issue", "status", "owner", "failed_paths", "constraints", "next_action", "context"],
    ),
    StateKind.incident_state.value: StateSpec(
        kind=StateKind.incident_state,
        title="Incident State",
        slot_rule={
            "timeline": "atom_type=event live, ascending by time",
            "root_cause": "why(question | newest negative EP) → DetectiveReport (chain == why().chain)",
            "constraints": "obs_type=constraint live",
            "rollback_point": "newest checkpoint of the scope",
        },
        template=["Timeline", "Root Cause", "Constraints", "Rollback Point", "Evidence"],
        fields=["timeline", "root_cause", "constraints", "rollback_point"],
    ),
    StateKind.meeting_decision_state.value: StateSpec(
        kind=StateKind.meeting_decision_state,
        title="Meeting Decision State",
        slot_rule={
            "decisions": "atoms matching DECISION_RE or positive-reward episodes",
            "owners": "atoms whose subject is a named party (not user/project/assistant)",
            "deadlines": "temporal atoms whose text matches DEADLINE_RE",
            "superseded": "atoms with archived_reason=revised plus their successor",
        },
        template=["Decisions", "Owners", "Deadlines", "Superseded", "Evidence"],
        fields=["decisions", "owners", "deadlines", "superseded"],
    ),
    StateKind.account_handoff_state.value: StateSpec(
        kind=StateKind.account_handoff_state,
        title="Account Handoff State",
        slot_rule={
            "package": "HMS.handoff (Handoff Package)",
            "customer_constraints": "obs_type=constraint live in the account scope",
            "rollback_point": "package.checkpoint_id",
        },
        template=["Customer Constraints", "Handoff Package", "Rollback Point", "Checkpoint id"],
        fields=["package", "customer_constraints", "rollback_point"],
    ),
}


def spec_of(kind: StateKind | str) -> StateSpec:
    """The StateSpec of a kind.
    某种类的 StateSpec。

    Input / 输入：
        kind: StateKind member or its value.
        kind：StateKind 成员或其值。

    Output / 输出：
        StateSpec; KeyError for unknown kinds.
        StateSpec；未知种类抛出 KeyError。

    State effect / 状态影响：
        None.
        无。
    """
    return STATE_SPECS[str(getattr(kind, "value", kind))]


__all__ = [
    "SCOPE_KEYS",
    "STATE_SPECS",
    "AccountHandoffState",
    "BaseState",
    "CustomerIssueState",
    "DetectiveReport",
    "EdgeState",
    "GroupState",
    "HandoffPackageState",
    "IncidentState",
    "MeetingDecisionState",
    "MemberConflict",
    "PersonalState",
    "RelationEdge",
    "RollbackPoint",
    "StateItem",
    "StateKind",
    "StateSpec",
    "WorkState",
    "context_sections",
    "item_from_atom",
    "item_from_context_item",
    "scope_from_filters",
    "spec_of",
]
