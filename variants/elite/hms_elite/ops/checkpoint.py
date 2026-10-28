"""Checkpoint / rollback: full-state snapshots of a scope and their symmetric, witnessed restoration.
检查点 / 回滚：某范围的全状态快照及其对称、带见证的恢复。
"""

# Purpose (EN): Snapshot atom/edge/InterfaceEP/EP validity per scope and roll a scope back to a snapshot (§16.2).
# 目的（中）：按范围快照原子/边/InterfaceEP/EP 的有效性，并将范围回滚到某快照（§16.2）。

# Input (EN): A StateStore (+ SupportGraph for rollback), a scope or a checkpoint id, an optional label/reason.
# 输入（中）：StateStore（回滚时另需 SupportGraph）、范围或检查点 id、可选的标签/原因。

# Output (EN): A checkpoint id, or the rollback report listing restored/archived atoms and edge changes.
# 输出（中）：检查点 id，或列出已恢复/已归档原子与边变化的回滚报告。

# State effect (EN): checkpoint writes source_events/anchors/checkpoints; rollback archives/restores atoms, EPs, edges,
#                    InterfaceEPs, adds revision edges + history rows and one rollback transition — never deletes.
# 状态影响（中）：checkpoint 写 source_events/anchors/checkpoints；rollback 归档/恢复原子、EP、边、InterfaceEP，
#                添加 revision 边、历史行与一条 rollback 转移——绝不删除。

from __future__ import annotations

from hms_elite.enums import EdgeType, HistoryEvent, Reward, TransitionKind
from hms_elite.exceptions import ValidationError
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.ingest.events import system_event
from hms_elite.memory.storage import StateStore
from hms_elite.models import Checkpoint, MemoryAtom, Scope, Transition, kappa_t_of
from hms_elite.ops.transitions import constraint_texts_at
from hms_elite.utils.json import canonical_json
from hms_elite.utils.text import sha256_hex


def snapshot_hash(atom_state: dict, edge_state: dict, iep_state: dict, ep_state: dict) -> str:
    """sha256 over canonical_json({atom_state, edge_state, iep_state, ep_state}).
    对 canonical_json({atom_state, edge_state, iep_state, ep_state}) 取 sha256。

    Input / 输入：
        atom_state: atom_id → {'valid_to', 'archived_reason'}.
        atom_state：atom_id → {'valid_to', 'archived_reason'}。

        edge_state: edge_id → valid.
        edge_state：edge_id → valid。

        iep_state: interface_ep_id → valid.
        iep_state：interface_ep_id → valid。

        ep_state: ep_id → eps.valid_to.
        ep_state：ep_id → eps.valid_to。

    Output / 输出：
        Hex digest.
        十六进制摘要。

    State effect / 状态影响：
        None.
        无。
    """
    payload = {"atom_state": atom_state, "edge_state": edge_state, "iep_state": iep_state, "ep_state": ep_state}
    return sha256_hex(canonical_json(payload))


def checkpoint(store: StateStore, *, scope: Scope, label: str | None) -> str:
    """Snapshot the validity of every atom, edge, InterfaceEP and EP of a scope.
    快照某范围内每个原子、边、InterfaceEP 与 EP 的有效性。

    Input / 输入：
        store: The state store.
        store：状态存储。

        scope: The scope to snapshot (>= 1 id).
        scope：要快照的范围（至少一个 id）。

        label: Optional human label.
        label：可选的人类可读标签。

    Output / 输出：
        The checkpoint id (prefix 'ckp').
        检查点 id（前缀 'ckp'）。

    State effect / 状态影响：
        One transaction writing source_events, anchors and checkpoints — all unversioned rows (§6.7): a checkpoint changes nothing a scope reads, so HMS.context()'s memo stays reused (rollback does invalidate).
        一个事务，写入 source_events、anchors 与 checkpoints——全部为不带版本的行（§6.7）：检查点不改变范围读到的任何内容，因此 HMS.context() 的记忆保持复用（rollback 则会使其失效）。
    """
    scope.require_any(context="checkpoint")
    with store.transaction(), store.unversioned():
        _ev, anchor = system_event(
            store, scope=scope, object_type="checkpoint", event_type="checkpoint", payload={"label": label}
        )
        atom_state = store.scope_atom_state(scope)
        edge_state = {e.edge_id: bool(e.valid) for e in store.list_edges(scope, valid_only=False)}
        iep_state = {i.interface_ep_id: bool(i.valid) for i in store.list_interface_eps(scope, valid_only=False)}
        ep_state = store.scope_ep_state(scope)
        cp = Checkpoint(
            label=label,
            anchor_id=anchor.anchor_id,
            scope=scope,
            atom_state=atom_state,
            edge_state=edge_state,
            iep_state=iep_state,
            ep_state=ep_state,
            history_seq=store.history_seq(),
            snapshot_hash=snapshot_hash(atom_state, edge_state, iep_state, ep_state),
            created_at=store.now(),
        )
        return store.insert_checkpoint(cp)


def _clamp_valid_to(atom: MemoryAtom, valid_to: str | None, fallback: str) -> str:
    """valid_to that satisfies the DDL CHECK valid_to >= valid_from.
    满足 DDL CHECK valid_to >= valid_from 的 valid_to。
    """
    value = valid_to or fallback
    if atom.valid_from is not None and value < atom.valid_from:
        return atom.valid_from
    return value


def _history(store: StateStore, atom: MemoryAtom, *, event: str, anchor_id: str, at: str, details: dict) -> None:
    """Append one ROLLBACK / RESTORE history row for an atom.
    为某原子追加一条 ROLLBACK / RESTORE 历史行。
    """
    old_memory = atom.text if event == HistoryEvent.ROLLBACK.value else None
    new_memory = atom.text if event == HistoryEvent.RESTORE.value else None
    store.add_history(
        memory_id=atom.memory_id,
        old_memory=old_memory,
        new_memory=new_memory,
        event=event,
        created_at=atom.created_at or at,
        updated_at=at,
        anchor_id=anchor_id,
        atom_id=atom.atom_id,
        details=details,
    )


def _atom_state(atom: MemoryAtom) -> dict:
    """s_t / s_t1 dict of an atom version for the rollback transition.
    回滚转移中原子版本的 s_t / s_t1 字典。
    """
    return {
        "text": atom.text,
        "atom_id": atom.atom_id,
        "memory_id": atom.memory_id,
        "valid_from": atom.valid_from,
        "valid_to": atom.valid_to,
    }


def rollback(store: StateStore, graph: SupportGraph, *, checkpoint_id: str, reason: str | None) -> dict:
    """Restore a scope to a checkpoint: archives, restores, EP validity, edges, InterfaceEPs, rollback edges + transition.
    将范围恢复到检查点：归档、恢复、EP 有效性、边、InterfaceEP、rollback 边与转移。

    Input / 输入：
        store: The state store.
        store：状态存储。

        graph: The SupportGraph (writes the rollback revision edges).
        graph：SupportGraph（写入 rollback 修订边）。

        checkpoint_id: The checkpoint to restore (unknown → NOT_FOUND_001, a ValueError).
        checkpoint_id：要恢复的检查点（未知 → NOT_FOUND_001，属 ValueError）。

        reason: Optional reason recorded on the rollback event and transition.
        reason：记录在回滚事件与转移上的可选原因。

    Output / 输出：
        {'message', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}.
        {'message', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}。

    State effect / 状态影响：
        One transaction witnessed by the rollback anchor; content is never touched, vectors never deleted.
        一个由回滚锚点见证的事务；从不修改内容，从不删除向量。
    """
    cp = store.get_checkpoint(checkpoint_id)
    if cp is None:
        raise ValidationError(
            f"unknown checkpoint {checkpoint_id!r}", "NOT_FOUND_001", details={"checkpoint_id": checkpoint_id}
        )
    scope = cp.scope
    tag = f"rollback:{cp.checkpoint_id}"
    restored: list[str] = []
    archived: list[str] = []
    edges_invalidated: list[str] = []
    edges_revalidated: list[str] = []
    edges_created: list[str] = []
    with store.transaction():
        _ev, rb_anchor = system_event(
            store,
            scope=scope,
            object_type="rollback",
            event_type="rollback",
            payload={"checkpoint_id": cp.checkpoint_id, "label": cp.label, "reason": reason},
        )
        now = store.now()
        rb_id = rb_anchor.anchor_id
        current = store.scope_atom_state(scope)
        atoms = {a.atom_id: a for a in store.get_atoms(sorted(current))}
        details = {"checkpoint_id": cp.checkpoint_id}

        # (1) archives: atoms unknown to the snapshot, and atoms recorded archived but live now.
        # (1) 归档：快照中不存在的原子，以及快照记录为已归档但当前存活的原子。
        for atom_id in sorted(current):
            atom = atoms.get(atom_id)
            if atom is None:
                continue
            live_now = current[atom_id]["archived_reason"] is None
            recorded = cp.atom_state.get(atom_id)
            if recorded is None:
                if live_now:
                    store.archive_atom(atom_id, reason=tag, at=now)
                    _history(store, atom, event=HistoryEvent.ROLLBACK.value, anchor_id=rb_id, at=now, details=details)
                    archived.append(atom_id)
            elif recorded.get("archived_reason") is not None and live_now:
                valid_to = _clamp_valid_to(atom, recorded.get("valid_to"), now)
                store.archive_atom(atom_id, reason=tag, at=now, valid_to=valid_to)
                _history(store, atom, event=HistoryEvent.ROLLBACK.value, anchor_id=rb_id, at=now, details=details)
                archived.append(atom_id)

        # (2) restores: atoms recorded live but archived now (successors are archived already, so heads stay unique).
        # (2) 恢复：快照记录为存活但当前已归档的原子（后继此时已归档，故头版本保持唯一）。
        for atom_id in sorted(current):
            atom = atoms.get(atom_id)
            recorded = cp.atom_state.get(atom_id)
            if atom is None or recorded is None:
                continue
            if recorded.get("archived_reason") is None and current[atom_id]["archived_reason"] is not None:
                store.restore_atom(atom_id, at=now)
                if recorded.get("valid_to") is not None:
                    store.set_valid_to(atom_id, _clamp_valid_to(atom, recorded.get("valid_to"), now), at=now)
                _history(store, atom, event=HistoryEvent.RESTORE.value, anchor_id=rb_id, at=now, details=details)
                restored.append(atom_id)

        # (2b) EP validity follows the snapshot (after the atom cascade, so the snapshot wins).
        # (2b) EP 的有效性跟随快照（在原子级联之后执行，因此以快照为准）。
        ep_now = store.scope_ep_state(scope)
        for ep_id in sorted(ep_now):
            current_valid_to = ep_now[ep_id]
            if ep_id not in cp.ep_state:
                if current_valid_to is None:
                    store.archive_ep(ep_id, at=now)
                continue
            recorded_valid_to = cp.ep_state[ep_id]
            if recorded_valid_to == current_valid_to:
                continue
            if recorded_valid_to is None:
                store.restore_ep(ep_id)
            else:
                store.archive_ep(ep_id, at=recorded_valid_to)

        # (3) edges.
        # (3) 边。
        for edge in sorted(store.list_edges(scope, valid_only=False), key=lambda e: e.edge_id):
            recorded_valid = cp.edge_state.get(edge.edge_id)
            if recorded_valid is None:
                if edge.valid:
                    store.invalidate_edge(edge.edge_id, reason=tag, at=now)
                    edges_invalidated.append(edge.edge_id)
            elif recorded_valid and not edge.valid:
                store.revalidate_edge(edge.edge_id, at=now)
                edges_revalidated.append(edge.edge_id)
            elif not recorded_valid and edge.valid:
                store.invalidate_edge(edge.edge_id, reason=tag, at=now)
                edges_invalidated.append(edge.edge_id)

        # (4) InterfaceEPs.
        # (4) 各 InterfaceEP。
        for iep in sorted(store.list_interface_eps(scope, valid_only=False), key=lambda i: i.interface_ep_id):
            recorded_valid = cp.iep_state.get(iep.interface_ep_id)
            if recorded_valid is None:
                if iep.valid:
                    store.invalidate_interface_ep(iep.interface_ep_id, at=now)
            elif recorded_valid and not iep.valid:
                store.revalidate_interface_ep(iep.interface_ep_id, at=now)
            elif not recorded_valid and iep.valid:
                store.invalidate_interface_ep(iep.interface_ep_id, at=now)

        # (5) rollback revision edges (archived successor → restored predecessor) + one rollback transition each.
        # (5) 回滚 revision 边（已归档的后继 → 已恢复的前驱），并各配一条 rollback 转移。
        archived_by_lineage: dict[str, list[str]] = {}
        for atom_id in archived:
            archived_by_lineage.setdefault(atoms[atom_id].memory_id, []).append(atom_id)
        constraints = constraint_texts_at(store, scope=scope, as_of=now)
        for restored_id in restored:
            successors = sorted(archived_by_lineage.get(atoms[restored_id].memory_id, []))
            for successor_id in successors:
                edge_id = graph.insert_edge(
                    src_id=successor_id,
                    dst_id=restored_id,
                    edge_type=EdgeType.revision.value,
                    witness_anchor_ids=[rb_id],
                    created_by="ops",
                    proposer="rule",
                    confidence=0.9,
                    label=tag,
                )
                edges_created.append(edge_id)
                successor = store.get_atom(successor_id) or atoms[successor_id]
                predecessor = store.get_atom(restored_id) or atoms[restored_id]
                store.upsert_transition(
                    Transition(
                        source_kind=TransitionKind.rollback,
                        source_id=edge_id,
                        scale="L1_micro",
                        s_t=_atom_state(successor),
                        a_t="rollback",
                        s_t1=_atom_state(predecessor),
                        r_t=Reward.neutral,
                        kappa_t=kappa_t_of(
                            constraints=constraints,
                            visibility=str(predecessor.visibility),
                            permission_scope=str(rb_anchor.permission_scope),
                            provenance={"checkpoint_id": cp.checkpoint_id, "label": cp.label, "reason": reason},
                        ),
                        w_t=[rb_id],
                        time_t=successor.created_at or now,
                        time_t1=now,
                        scope=predecessor.scope,
                    )
                )
    return {
        "message": f"Rolled back to {cp.label or cp.checkpoint_id}",
        "checkpoint_id": cp.checkpoint_id,
        "restored": restored,
        "archived": archived,
        "edges_invalidated": edges_invalidated,
        "edges_revalidated": edges_revalidated,
        "edges_created": edges_created,
        "anchor_id": rb_id,
    }


__all__ = ["checkpoint", "rollback", "snapshot_hash"]
