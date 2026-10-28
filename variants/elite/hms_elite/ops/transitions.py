"""Transition export: EP, InterfaceEP, causal and revision/conflict rows with one κ_t shape, idempotent per source.
状态转移导出：EP、InterfaceEP、causal 与 revision/conflict 行，统一 κ_t 形状，按来源幂等。
"""

# Purpose (EN): Materialize (S_t, a_t, S_t+1, r_t, κ_t, w_t) training tuples from EPs, InterfaceEPs and typed edges (§16.4).
# 目的（中）：由 EP、InterfaceEP 与类型化边物化 (S_t, a_t, S_t+1, r_t, κ_t, w_t) 训练元组（§16.4）。

# Input (EN): A StateStore, a scope and optional since/scale/kinds filters.
# 输入（中）：StateStore、范围以及可选的 since/scale/kinds 过滤。

# Output (EN): Transition models ordered by (time_t1, KIND_ORDER[source_kind], source_id).
# 输出（中）：按 (time_t1, KIND_ORDER[source_kind], source_id) 排序的 Transition 模型。

# State effect (EN): Writes the transitions table only (upsert per (source_kind, source_id)); never touches atoms or edges.
# 状态影响（中）：仅写 transitions 表（按 (source_kind, source_id) upsert）；不触碰原子或边。

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hms_elite.enums import EdgeType, NodeKind, ObservationType, Reward, TransitionKind
from hms_elite.memory.storage import StateStore
from hms_elite.models import GraphEdge, GraphNode, InterfaceEP, Scope, Slots, Transition, kappa_t_of
from hms_elite.utils.text import ordered_union

ep_transition = Transition.from_ep
interface_ep_transition = Transition.from_interface_ep

CONSTRAINT_ORDER = "time_anchor ASC, created_at ASC, atom_id ASC"
_EPISODE_KINDS = frozenset({NodeKind.ep.value, NodeKind.interface_ep.value})
_REVISION_ACTION = {EdgeType.revision.value: "revise", EdgeType.conflict.value: "conflict"}
_REVISION_REWARD = {EdgeType.revision.value: Reward.negative.value, EdgeType.conflict.value: Reward.unknown.value}


@dataclass
class _EpisodeView:
    """Uniform view of an EP or InterfaceEP node for causal rows.
    面向 causal 行的 EP / InterfaceEP 节点统一视图。
    """

    kind: str
    id: str
    slots: Slots
    anchor_ids: list[str]
    time_anchor: str
    valid_from: str | None
    valid_to: str | None

    def state(self) -> dict:
        """The s_t / s_t1 dict of this episode.
        该情节的 s_t / s_t1 字典。
        """
        return {
            "u": self.slots.u,
            "b": self.slots.b,
            "y": self.slots.y,
            "r": str(self.slots.r),
            "kappa": self.slots.kappa.model_dump(),
            f"{self.kind}_id": self.id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


def constraint_texts_at(store: StateStore, *, scope: Scope, as_of: str) -> list[str]:
    """Texts of the constraint atoms valid in a scope at an instant, time_anchor ascending.
    某范围在某时刻有效的约束原子文本，按 time_anchor 升序。

    Input / 输入：
        store: The state store.
        store：状态存储。

        scope: Scope filter (Scope.matches semantics).
        scope：范围过滤（Scope.matches 语义）。

        as_of: Validity instant.
        as_of：有效性参考时刻。

    Output / 输出：
        Constraint texts ordered by (time_anchor, created_at, atom_id).
        按 (time_anchor, created_at, atom_id) 排序的约束文本。

    State effect / 状态影响：
        None.
        无。
    """
    atoms = store.axis_select(
        scope,
        where_sql="obs_type = ?",
        params=[ObservationType.constraint.value],
        as_of=as_of,
        order=CONSTRAINT_ORDER,
    )
    return [a.text for a in atoms]


def permission_scope_of(store: StateStore, anchor_id: str | None) -> str:
    """The permission_scope of an anchor ('private' when unknown).
    某锚点的 permission_scope（未知时为 'private'）。

    Input / 输入：
        store: The state store.
        store：状态存储。

        anchor_id: Anchor id (None allowed).
        anchor_id：锚点 id（允许 None）。

    Output / 输出：
        Visibility value.
        可见性值。

    State effect / 状态影响：
        None.
        无。
    """
    anchor = store.get_anchor(anchor_id) if anchor_id else None
    return str(anchor.permission_scope) if anchor is not None else "private"


def interface_ep_window(store: StateStore, iep: InterfaceEP) -> tuple[str, str]:
    """(min, max) time_anchor over the support atoms of an InterfaceEP.
    InterfaceEP 支持原子 time_anchor 的 (最小值, 最大值)。

    Input / 输入：
        store: The state store.
        store：状态存储。

        iep: The InterfaceEP.
        iep：接口 EP。

    Output / 输出：
        (time_t, time_t1); both equal the InterfaceEP's created_at when no support atom carries a time.
        (time_t, time_t1)；支持原子均无时间时二者均为 InterfaceEP 的 created_at。

    State effect / 状态影响：
        None.
        无。
    """
    atoms = store.get_atoms(iep.support_node_ids)
    times = sorted(t for t in (a.time_anchor or a.created_at for a in atoms) if t)
    if not times:
        fallback = iep.created_at or store.now()
        return fallback, fallback
    return times[0], times[-1]


def _episode_view(store: StateStore, node: GraphNode) -> _EpisodeView | None:
    """Resolve an ep / interface_ep node to its slots, anchors and validity.
    将 ep / interface_ep 节点解析为其槽位、锚点与有效期。
    """
    kind = str(node.node_kind)
    if kind == NodeKind.ep.value:
        ep = store.get_ep(node.ref_id)
        if ep is None:
            return None
        return _EpisodeView(
            kind="ep",
            id=ep.ep_id,
            slots=ep.slots,
            anchor_ids=list(ep.anchor_ids),
            time_anchor=ep.time_anchor,
            valid_from=ep.valid_from or ep.time_anchor,
            valid_to=ep.valid_to,
        )
    if kind == NodeKind.interface_ep.value:
        iep = store.get_interface_ep(node.ref_id)
        if iep is None:
            return None
        _t0, t1 = interface_ep_window(store, iep)
        return _EpisodeView(
            kind="interface_ep",
            id=iep.interface_ep_id,
            slots=iep.slots,
            anchor_ids=list(iep.evidence_anchor_ids),
            time_anchor=t1,
            valid_from=iep.created_at,
            valid_to=iep.valid_to,
        )
    return None


def _iep_rows(store: StateStore, scope: Scope) -> list[Transition]:
    """(a) one 'interface_ep' row per valid InterfaceEP in scope.
    (a) 范围内每个有效 InterfaceEP 一行 'interface_ep'。
    """
    rows: list[Transition] = []
    for iep in store.list_interface_eps(scope, valid_only=True):
        time_t, time_t1 = interface_ep_window(store, iep)
        rows.append(
            Transition.from_interface_ep(
                iep,
                scope=iep.scope,
                time_t=time_t,
                time_t1=time_t1,
                permission_scope=permission_scope_of(store, iep.evidence_anchor_ids[0]),
            )
        )
    return rows


def _causal_rows(store: StateStore, scope: Scope) -> list[Transition]:
    """(b) one 'causal' row per valid causal edge between two episode nodes.
    (b) 每条连接两个情节节点的有效 causal 边一行 'causal'。
    """
    rows: list[Transition] = []
    for edge in store.list_edges(scope, edge_types=[EdgeType.causal.value], valid_only=True):
        nodes = {n.node_id: n for n in store.get_nodes([edge.src_id, edge.dst_id])}
        src, dst = nodes.get(edge.src_id), nodes.get(edge.dst_id)
        if src is None or dst is None or str(src.node_kind) not in _EPISODE_KINDS:
            continue
        if str(dst.node_kind) not in _EPISODE_KINDS:
            continue
        i, j = _episode_view(store, src), _episode_view(store, dst)
        if i is None or j is None:
            continue
        witnesses = store.edge_witnesses(edge.edge_id)
        rows.append(
            Transition(
                source_kind=TransitionKind.causal,
                source_id=edge.edge_id,
                scale=str(edge.scale),
                s_t=i.state(),
                a_t=j.slots.a,
                s_t1=j.state(),
                r_t=str(j.slots.r),
                kappa_t=kappa_t_of(
                    constraints=list(j.slots.kappa.constraints),
                    visibility=str(j.slots.kappa.visibility),
                    permission_scope=permission_scope_of(store, witnesses[0] if witnesses else None),
                    provenance={
                        "edge_id": edge.edge_id,
                        "label": edge.label,
                        "proposer": edge.proposer,
                        "evidence": list(j.slots.kappa.evidence),
                    },
                ),
                w_t=ordered_union(witnesses, j.anchor_ids),
                time_t=i.time_anchor,
                time_t1=j.time_anchor,
                scope=edge.scope,
            )
        )
    return rows


def _atom_state(atom) -> dict:
    """The s_t / s_t1 dict of an atom version.
    原子版本的 s_t / s_t1 字典。
    """
    return {
        "text": atom.text,
        "atom_id": atom.atom_id,
        "memory_id": atom.memory_id,
        "valid_from": atom.valid_from,
        "valid_to": atom.valid_to,
    }


def revision_transition(store: StateStore, edge: GraphEdge) -> Transition | None:
    """The 'revision' / 'conflict' row of one revision or conflict edge between two atom versions (§16.4 c).
    两个原子版本之间一条 revision / conflict 边对应的 'revision' / 'conflict' 行（§16.4 c）。

    Input / 输入：
        store: The state store.
        store：状态存储。

        edge: A revision or conflict edge (rollback-labelled edges and other types → None).
        edge：revision 或 conflict 边（rollback 标签的边与其他类型 → None）。

    Output / 输出：
        The Transition, or None when an endpoint atom is missing or the edge is not a revision/conflict edge.
        Transition；端点原子缺失或边不是 revision/conflict 边时为 None。

    State effect / 状态影响：
        None (the caller upserts it; the writer that creates the edge does so at creation time).
        无（由调用方 upsert；创建边的写入者在创建时写入）。
    """
    kind = str(edge.edge_type)
    if kind not in _REVISION_ACTION or str(edge.label or "").startswith("rollback:"):
        return None
    old, new = store.get_atom(edge.src_id), store.get_atom(edge.dst_id)
    if old is None or new is None:
        return None
    witnesses = store.edge_witnesses(edge.edge_id) or list(edge.witness_anchor_ids)
    time_t1 = edge.created_at or store.now()
    return Transition(
        source_kind=TransitionKind(kind),
        source_id=edge.edge_id,
        scale=str(edge.scale),
        s_t=_atom_state(old),
        a_t=_REVISION_ACTION[kind],
        s_t1=_atom_state(new),
        r_t=_REVISION_REWARD[kind],
        kappa_t=kappa_t_of(
            constraints=constraint_texts_at(store, scope=edge.scope, as_of=time_t1),
            visibility=str(new.visibility),
            permission_scope=permission_scope_of(store, witnesses[0] if witnesses else None),
            provenance={"label": edge.label, "created_by": edge.created_by, "proposer": edge.proposer},
        ),
        w_t=list(witnesses),
        time_t=old.created_at or time_t1,
        time_t1=time_t1,
        scope=edge.scope,
    )


def _revision_rows(store: StateStore, scope: Scope) -> list[Transition]:
    """(c) one 'revision' / 'conflict' row per revision or conflict edge (any validity) not labelled rollback:.
    (c) 每条未以 rollback: 标注的 revision / conflict 边（不论有效性）一行 'revision' / 'conflict'。
    """
    rows: list[Transition] = []
    edge_types = [EdgeType.revision.value, EdgeType.conflict.value]
    for edge in store.list_edges(scope, edge_types=edge_types, valid_only=False):
        row = revision_transition(store, edge)
        if row is not None:
            rows.append(row)
    return rows


def export_transitions(
    store: StateStore,
    *,
    scope: Scope,
    since: str | None,
    scale: str | None,
    kinds: Iterable[str] | None,
) -> list[Transition]:
    """Export every transition of a scope (InterfaceEP, causal, revision/conflict rows added idempotently) and list them.
    导出某范围的全部状态转移（幂等地补充 InterfaceEP、causal、revision/conflict 行）并列出。

    Input / 输入：
        store: The state store.
        store：状态存储。

        scope: Scope filter (Scope.matches semantics; a partial scope covers every run it matches).
        scope：范围过滤（Scope.matches 语义；部分范围覆盖其匹配的每个运行）。

        since: Inclusive lower bound on time_t1 (None → no bound).
        since：time_t1 的下界（含；None 表示不限）。

        scale: Optional scale filter ('L1_micro' | 'L2_meso' | 'L3_macro').
        scale：可选的尺度过滤（'L1_micro' | 'L2_meso' | 'L3_macro'）。

        kinds: Optional source_kind whitelist.
        kinds：可选的 source_kind 白名单。

    Output / 输出：
        Transitions ordered by (time_t1, KIND_ORDER[source_kind], source_id).
        按 (time_t1, KIND_ORDER[source_kind], source_id) 排序的 Transition 列表。

    State effect / 状态影响：
        One transaction writing only the transitions table (INSERT OR IGNORE per (source_kind, source_id)).
        一个事务，仅写 transitions 表（按 (source_kind, source_id) INSERT OR IGNORE）。
    """
    with store.transaction():
        for row in _iep_rows(store, scope) + _causal_rows(store, scope) + _revision_rows(store, scope):
            store.upsert_transition(row)
    kind_list = [str(k) for k in kinds] if kinds else None
    return store.list_transitions(scope, since=since, scale=scale, kinds=kind_list)


__all__ = [
    "CONSTRAINT_ORDER",
    "constraint_texts_at",
    "ep_transition",
    "export_transitions",
    "interface_ep_transition",
    "interface_ep_window",
    "permission_scope_of",
    "revision_transition",
]
