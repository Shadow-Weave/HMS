"""EdgeState builder (HMS §11.2): the state between two entities or two user scopes.
EdgeState 构建器（HMS §11.2）：两个实体或两个用户范围之间的状态。
"""

# Purpose (EN): Derive shared facts, time-ordered interactions, last contact, support-graph relation edges and open
# conflicts for an endpoint pair through public store / graph reads.
# 目的（中）：通过存储 / 图的公开读方法，为一对端点推导共享事实、按时间排序的互动、最近联系、支持图关系边与未解冲突。

# Input (EN): The HMS facade, two endpoints (entity texts with a scope filter, or two scope dicts), as_of.
# 输入（中）：HMS 门面、两个端点（带范围过滤的实体文本，或两个范围字典）、as_of。

# Output (EN): EdgeState whose every entry carries anchor_ids.
# 输出（中）：每一项都带 anchor_ids 的 EdgeState。

# State effect (EN): None (read-only).
# 状态影响（中）：无（只读）。

from __future__ import annotations

from typing import Any

from hms_elite.enums import AtomType, EdgeType
from hms_elite.models import GraphEdge, MemoryAtom, Scope
from hms_elite.state.group import find_member_conflicts, live_atoms, member_key
from hms_elite.state.kinds import EdgeState, RelationEdge, StateItem, item_from_atom, scope_from_filters
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.text import entity_key

_INTERACTION_TYPES = frozenset({AtomType.event.value, AtomType.episode.value})
_RELATION_TYPES = [EdgeType.use.value, EdgeType.causal.value]


def _by_time(items: list[StateItem]) -> list[StateItem]:
    """Ascending by (time, atom_id) — the interaction order.
    按 (time, atom_id) 升序 —— 互动顺序。
    """
    return sorted(items, key=lambda i: (i.time or "", i.atom_id or ""))


def _entity_id(store: Any, scope: Scope, text: str) -> str | None:
    """Entity id of an entity text inside a scope (matched on entities.normalized_text).
    某范围内实体文本对应的实体 id（按 entities.normalized_text 匹配）。
    """
    key = entity_key(text)
    for row in store.entities_in_scope(scope):
        if row["normalized_text"] == key:
            return str(row["entity_id"])
    return None


def _atoms_of_entity(store: Any, entity_id: str, *, as_of: str) -> dict[str, MemoryAtom]:
    """Live atoms linked to an entity, keyed by atom_id.
    链接到某实体的存活原子，按 atom_id 索引。
    """
    ids = store.entity_linked_atoms(entity_id, valid_only=True)
    return {a.atom_id: a for a in store.get_atoms(ids) if a.is_valid_at(as_of)}


def relation_edges(store: Any, atom_ids: list[str]) -> list[RelationEdge]:
    """Valid use/causal edges touching a node set, as witnessed RelationEdges (ordered by created_at, edge_id).
    触及某节点集合的有效 use/causal 边，转成带见证的 RelationEdge（按 created_at、edge_id 排序）。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        atom_ids: Node ids (== atom ids).
        atom_ids：节点 id（== 原子 id）。

    Output / 输出：
        RelationEdge list.
        RelationEdge 列表。

    State effect / 状态影响：
        None.
        无。
    """
    if not atom_ids:
        return []
    edges: list[GraphEdge] = store.edges_touching(atom_ids, valid_only=True, edge_types=_RELATION_TYPES)
    texts = {a.atom_id: a.text for a in store.get_atoms({e.src_id for e in edges} | {e.dst_id for e in edges})}
    return [
        RelationEdge(
            edge_id=e.edge_id,
            edge_type=str(e.edge_type),
            src_id=e.src_id,
            dst_id=e.dst_id,
            src_text=texts.get(e.src_id, e.src_id),
            dst_text=texts.get(e.dst_id, e.dst_id),
            label=e.label,
            confidence=float(e.confidence),
            anchor_ids=list(e.witness_anchor_ids),
        )
        for e in edges
    ]


def open_conflict_items(store: Any, atom_ids: list[str]) -> list[StateItem]:
    """Unresolved conflict edges touching a node set, one StateItem per edge (witnessed by the edge).
    触及某节点集合的未解冲突边，每条边一个 StateItem（由该边见证）。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        atom_ids: Node ids.
        atom_ids：节点 id。

    Output / 输出：
        StateItem list ordered by (created_at, edge_id).
        按 (created_at, edge_id) 排序的 StateItem 列表。

    State effect / 状态影响：
        None.
        无。
    """
    if not atom_ids:
        return []
    edges = [
        e
        for e in store.edges_touching(atom_ids, valid_only=True, edge_types=[EdgeType.conflict.value])
        if e.resolved_at is None
    ]
    texts = {a.atom_id: a.text for a in store.get_atoms({e.src_id for e in edges} | {e.dst_id for e in edges})}
    return [
        StateItem(
            text=f"{texts.get(e.src_id, e.src_id)} ⟂ {texts.get(e.dst_id, e.dst_id)}",
            time=e.created_at,
            anchor_ids=list(e.witness_anchor_ids),
            extra={"edge_id": e.edge_id, "src_id": e.src_id, "dst_id": e.dst_id, "label": e.label},
        )
        for e in edges
    ]


def _entity_edge(store: Any, scope: Scope, a: str, b: str, *, as_of: str) -> EdgeState:
    """EdgeState between two entity keys inside one scope.
    同一范围内两个实体键之间的 EdgeState。
    """
    ent_a, ent_b = _entity_id(store, scope, a), _entity_id(store, scope, b)
    atoms_a = _atoms_of_entity(store, ent_a, as_of=as_of) if ent_a else {}
    atoms_b = _atoms_of_entity(store, ent_b, as_of=as_of) if ent_b else {}
    both = [atoms_a[i] for i in sorted(set(atoms_a) & set(atoms_b)) if scope.matches(atoms_a[i].scope)]
    shared = [item_from_atom(x) for x in both if str(x.atom_type) not in _INTERACTION_TYPES]
    interactions = _by_time([item_from_atom(x) for x in both if str(x.atom_type) in _INTERACTION_TYPES])
    touched = sorted(set(atoms_a) | set(atoms_b))
    times = [i.time for i in interactions + shared if i.time]
    return EdgeState(
        scope=scope,
        as_of=as_of,
        a=a,
        b=b,
        endpoint_kind="entity",
        shared_facts=sorted(shared, key=lambda i: (i.time or "", i.atom_id or "")),
        interactions=interactions,
        last_contact=max(times) if times else None,
        relation_edges=relation_edges(store, [x.atom_id for x in both]),
        open_conflicts=open_conflict_items(store, touched),
    )


def _scope_edge(store: Any, scope_a: Scope, scope_b: Scope, *, as_of: str) -> EdgeState:
    """EdgeState between two user scopes: shared entities bridge the two stores of facts.
    两个用户范围之间的 EdgeState：共享实体连接双方的事实。
    """
    key_a, key_b = member_key(scope_a), member_key(scope_b)
    ents_a = {r["normalized_text"] for r in store.entities_in_scope(scope_a)}
    ents_b = {r["normalized_text"] for r in store.entities_in_scope(scope_b)}
    # The other party's user id is itself a bridge when one side mentions the other.
    # 一方提到另一方时，对方的 user id 本身就是桥。
    bridges = (ents_a & ents_b) | ({entity_key(scope_b.user_id)} & ents_a if scope_b.user_id else set())
    bridges |= {entity_key(scope_a.user_id)} & ents_b if scope_a.user_id else set()
    atoms_a = live_atoms(store, scope_a, as_of=as_of)
    atoms_b = live_atoms(store, scope_b, as_of=as_of)
    shared: list[StateItem] = []
    interactions: list[StateItem] = []
    linked: list[str] = []
    for side, atoms in ((key_a, atoms_a), (key_b, atoms_b)):
        for atom in atoms:
            if not bridges.intersection(atom.entity_ids):
                continue
            linked.append(atom.atom_id)
            item = item_from_atom(atom, side=side, shared_entities=sorted(bridges.intersection(atom.entity_ids)))
            (interactions if str(atom.atom_type) in _INTERACTION_TYPES else shared).append(item)
    interactions = _by_time(interactions)
    times = [i.time for i in interactions + shared if i.time]
    conflicts = open_conflict_items(store, linked)
    for c in find_member_conflicts([(key_a, atoms_a), (key_b, atoms_b)]):
        conflicts.append(
            StateItem(
                text=f"{c.left.text} ⟂ {c.right.text}",
                anchor_ids=list(c.anchor_ids),
                extra={
                    "members": list(c.members),
                    "reason": c.reason,
                    "left": c.left.atom_id,
                    "right": c.right.atom_id,
                },
            )
        )
    return EdgeState(
        scope=scope_a,
        as_of=as_of,
        a=key_a,
        b=key_b,
        endpoint_kind="scope",
        shared_facts=sorted(shared, key=lambda i: (i.time or "", i.atom_id or "")),
        interactions=interactions,
        last_contact=max(times) if times else None,
        relation_edges=relation_edges(store, linked),
        open_conflicts=conflicts,
    )


def build_edge_state(
    hms: Any, a: str | dict, b: str | dict, *, filters: dict | None = None, as_of: Any = None
) -> EdgeState:
    """Build the EdgeState of an endpoint pair: two entity texts inside `filters`, or two scope dicts.
    构建一对端点的 EdgeState：`filters` 范围内的两个实体文本，或两个范围字典。

    Input / 输入：
        hms: The HMS facade (only `hms.store` is used).
        hms：HMS 门面（仅使用 `hms.store`）。

        a: Entity text (str) or a Mem0 scope dict.
        a：实体文本（str）或 Mem0 范围字典。

        b: Entity text (str) or a Mem0 scope dict (same form as `a`; mixing raises ValueError).
        b：实体文本（str）或 Mem0 范围字典（与 `a` 同形；混用抛出 ValueError）。

        filters: Scope filters required in entity mode (VALIDATION_001 when missing).
        filters：实体模式下必需的范围过滤（缺失时 VALIDATION_001）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

    Output / 输出：
        EdgeState with interactions ascending by time.
        互动按时间升序的 EdgeState。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    if isinstance(a, dict) and isinstance(b, dict):
        return _scope_edge(store, scope_from_filters(a), scope_from_filters(b), as_of=instant)
    if isinstance(a, str) and isinstance(b, str):
        return _entity_edge(store, scope_from_filters(filters), a, b, as_of=instant)
    raise ValueError("EdgeState endpoints must both be entity texts or both be scope dicts")


__all__ = ["build_edge_state", "open_conflict_items", "relation_edges"]
