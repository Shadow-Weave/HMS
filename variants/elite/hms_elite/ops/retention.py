"""Retention escape hatch: explicit, audited, default-off physical purge of long-archived atom versions (§16.6).
保留策略逃生舱：显式、有审计、默认关闭的长期归档原子版本物理清理（§16.6）。
"""

# Purpose (EN): Select the archived versions that nothing live still needs, report them (dry run) or remove them in
#               one witnessed transaction — the only path that ever shrinks the store. Not a default behaviour.
# 目的（中）：挑出再无存活对象需要的已归档版本，报告（试运行）或在一个带见证的事务内删除——
#            唯一会让存储变小的路径。不是默认行为。

# Input (EN): A StateStore + index, a scope (>= 1 id, Scope.matches semantics), older_than_days, dry_run, keep_witnessed.
# 输入（中）：StateStore + 索引、范围（至少一个 id，Scope.matches 语义）、older_than_days、dry_run、keep_witnessed。

# Output (EN): A report: cutoff, examined, purgeable summaries (atom_id, memory_id, version, text_sha256,
#              archived_reason, valid_to, bytes), protected ids per reason, estimated_bytes, purged counts, anchor_id.
# 输出（中）：报告：cutoff、examined、可清理摘要（atom_id、memory_id、version、text_sha256、archived_reason、
#            valid_to、bytes）、按原因分组的受保护 id、estimated_bytes、已清理计数、anchor_id。

# State effect (EN): dry_run=True writes nothing. dry_run=False: one transaction writing a 'purge' system event +
#                    anchor, one PURGE history row per version, then StateStore.purge_rows() (state rows only unless
#                    keep_witnessed=False, which also drops anchors nothing cites any more); index vectors removed after.
# 状态影响（中）：dry_run=True 不写任何东西。dry_run=False：一个事务，写入 'purge' 系统事件 + 锚点、每个版本一条
#                PURGE 历史行，然后 StateStore.purge_rows()（只清状态行；keep_witnessed=False 时还删除再无引用的
#                锚点）；事务提交后再删除索引向量。

from __future__ import annotations

from datetime import timedelta
from typing import Any

from hms_elite.enums import HistoryEvent, SourceKind
from hms_elite.exceptions import ValidationError
from hms_elite.ingest.events import system_event
from hms_elite.memory.storage import StateStore
from hms_elite.models import MemoryAtom, Scope
from hms_elite.timeutil import parse_iso, to_utc_iso
from hms_elite.utils.json import canonical_json
from hms_elite.utils.text import sha256_hex

# Protection reasons, in report order; an id can appear under several reasons.
# 保护原因，按报告顺序；一个 id 可以同时出现在多个原因下。
PROTECTED_REASONS: tuple[str, ...] = (
    "head_version",
    "edge_endpoint",
    "checkpoint_live",
    "interface_ep_support",
    "ep_referenced",
    "source_shared",
)
# Fixed per-row overhead added to the byte estimate (ids, timestamps, index entries).
# 字节估算中每行固定加上的开销（id、时间戳、索引项）。
_ROW_OVERHEAD_BYTES = 256


def cutoff_instant(now: str, older_than_days: int) -> str:
    """The purge cutoff: `now` minus `older_than_days` days as a fixed-width UTC instant.
    清理截止时刻：`now` 减去 `older_than_days` 天，定宽 UTC 时刻。

    Input / 输入：
        now: The store clock reading.
        now：存储时钟读数。

        older_than_days: Non-negative integer number of days.
        older_than_days：非负整数天数。

    Output / 输出：
        The cutoff instant; rows with valid_to strictly before it are old enough.
        截止时刻；valid_to 严格早于它的行才算足够旧。

    State effect / 状态影响：
        None.
        无。
    """
    return to_utc_iso(parse_iso(now) - timedelta(days=int(older_than_days)))


def _check_days(older_than_days: Any) -> int:
    """older_than_days must be a non-negative int (bool excluded) → VALIDATION_009 otherwise.
    older_than_days 必须是非负整数（排除 bool），否则 VALIDATION_009。
    """
    if isinstance(older_than_days, bool) or not isinstance(older_than_days, int) or older_than_days < 0:
        raise ValidationError(
            f"older_than_days must be a non-negative integer, got {older_than_days!r}",
            "VALIDATION_009",
            details={"older_than_days": older_than_days},
        )
    return older_than_days


def _row_bytes(store: StateStore, atom: MemoryAtom) -> int:
    """Byte estimate of one version: text + content JSON + its embedding blob + a fixed row overhead.
    单个版本的字节估算：文本 + 内容 JSON + 其嵌入 BLOB + 固定行开销。
    """
    vector = store.load_vectors([atom.atom_id]).get(atom.atom_id)
    embedding = 4 * len(vector) if vector else 0
    return (
        len(atom.text.encode("utf-8"))
        + len(canonical_json(atom.content).encode("utf-8"))
        + embedding
        + _ROW_OVERHEAD_BYTES
    )


def _summary(store: StateStore, atom: MemoryAtom) -> dict:
    """The audit summary of one version (what the PURGE history row and the report carry; never the text itself).
    单个版本的审计摘要（PURGE 历史行与报告携带的内容；绝不含文本本身）。
    """
    return {
        "atom_id": atom.atom_id,
        "memory_id": atom.memory_id,
        "version": int(atom.version),
        "text_sha256": sha256_hex(atom.text),
        "archived_reason": atom.archived_reason,
        "valid_to": atom.valid_to,
        "source_kind": str(atom.source_kind),
        "source_id": atom.source_id,
        "anchor_ids": list(atom.anchor_ids),
        "bytes": _row_bytes(store, atom),
    }


def _protect(store: StateStore, candidates: list[MemoryAtom]) -> dict[str, set[str]]:
    """Apply the six protections to the candidate set; returns reason → protected atom ids.
    对候选集合应用六条保护；返回 原因 → 受保护原子 id。
    """
    ids = [a.atom_id for a in candidates]
    out: dict[str, set[str]] = {reason: set() for reason in PROTECTED_REASONS}
    # (1) head_version: the highest version of a multi-version lineage is the terminal record of its revision chain
    #     (history / why resolve to it); a single-version tombstone has no chain to anchor and may go.
    # (1) head_version：多版本谱系的最高版本是其修订链的终端记录（history / why 解析到它）；
    #     单版本墓碑没有链可锚定，可以清理。
    seen: dict[str, list[MemoryAtom]] = {}
    for atom in candidates:
        seen.setdefault(atom.memory_id, [])
    for memory_id in seen:
        versions = store.list_versions(memory_id)
        if len(versions) >= 2:
            out["head_version"].add(versions[-1].atom_id)
    out["head_version"] &= set(ids)
    # (2) edge_endpoint: graph_edges are never purged (they carry witnesses), so ANY edge — valid or invalid — pins
    #     both endpoints; this is a superset of "endpoint of a valid edge".
    # (2) edge_endpoint：graph_edges 从不清理（它们携带见证），因此任何边——有效或失效——都钉住两个端点；
    #     这是"有效边端点"的超集。
    out["edge_endpoint"] = store.edge_endpoint_ids(ids, valid_only=False)
    # (3) checkpoint_live: a rollback may still need to restore the version.
    # (3) checkpoint_live：回滚仍可能需要恢复该版本。
    out["checkpoint_live"] = store.checkpoint_live_atom_ids() & set(ids)
    # (4) interface_ep_support: InterfaceEP rows cite the atom (FK) or the node as support.
    # (4) interface_ep_support：InterfaceEP 行引用该原子（外键）或该节点作为支持。
    out["interface_ep_support"] = store.interface_ep_member_ids(ids)
    # (5) ep_referenced: the atom's own EP row is still cited by another EP / surviving atom / transition.
    # (5) ep_referenced：该原子自身的 EP 行仍被其它 EP / 幸存原子 / 转移引用。
    ep_atoms = {a.source_id: a.atom_id for a in candidates if str(a.source_kind) == SourceKind.ep.value}
    if ep_atoms:
        cited = store.ep_reference_ids(ep_atoms, excluding_atoms=ids)
        out["ep_referenced"] = {ep_atoms[ep_id] for ep_id in cited}
    # (6) source_shared: purge_rows deletes the atom's own observations / eps row, which a surviving version (an
    #     update / feedback successor copies source_kind + source_id) may still cite — independent of the revision edge.
    # (6) source_shared：purge_rows 会删除原子自身的 observations / eps 行，而幸存版本（更新 / 反馈的后继复制
    #     source_kind + source_id）可能仍引用它——不依赖 revision 边。
    out["source_shared"] = store.shared_source_atom_ids(ids)
    return out


def purge(
    store: StateStore,
    index: Any,
    *,
    scope: Scope,
    older_than_days: int,
    dry_run: bool = True,
    keep_witnessed: bool = True,
) -> dict:
    """Report (dry run) or physically remove archived versions older than the cutoff that nothing live still needs.
    报告（试运行）或物理删除早于截止时刻、且再无存活对象需要的已归档版本。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index (vectors of purged atoms are removed after the transaction commits).
        index：向量索引（事务提交后删除被清原子的向量）。

        scope: Scope with >= 1 id (Scope.matches semantics: a user_id-only scope sweeps every run of that user).
        scope：至少含一个 id 的范围（Scope.matches 语义：只给 user_id 会扫过该用户的所有 run）。

        older_than_days: Rows whose valid_to is before now − older_than_days days qualify (VALIDATION_009 if negative).
        older_than_days：valid_to 早于 now − older_than_days 天的行才符合（负数 → VALIDATION_009）。

        dry_run: True (default) writes nothing and only reports; False materializes the purge.
        dry_run：True（默认）不写任何东西、只报告；False 真正执行清理。

        keep_witnessed: True (default) never touches anchors or source events; False also removes the anchors of the purged rows that nothing in the database cites any more (with their source events).
        keep_witnessed：True（默认）从不触碰锚点与源事件；False 还会删除被清行的、库中再无任何对象引用的锚点（及其源事件）。

    Output / 输出：
        {'dry_run', 'scope', 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable': [summaries], 'protected': {reason: [ids]}, 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'} (anchor_id None on a dry run).
        {'dry_run', 'scope', 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable': [摘要], 'protected': {原因: [id]}, 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}（试运行时 anchor_id 为 None）。

    State effect / 状态影响：
        dry_run=True: none. dry_run=False: one transaction (system event op='purge' + anchor, PURGE history rows, purge_rows), then index.delete per purged atom.
        dry_run=True：无。dry_run=False：一个事务（op='purge' 的系统事件 + 锚点、PURGE 历史行、purge_rows），随后对每个被清原子调用 index.delete。
    """
    scope.require_any(context="purge")
    days = _check_days(older_than_days)
    now = store.now()
    cutoff = cutoff_instant(now, days)
    candidates = store.purge_candidates(scope, valid_to_before=cutoff)
    protected = _protect(store, candidates)
    blocked = set().union(*protected.values()) if protected else set()
    purgeable = [a for a in candidates if a.atom_id not in blocked]
    summaries = [_summary(store, a) for a in purgeable]
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "scope": scope.key(),
        "older_than_days": days,
        "cutoff": cutoff,
        "keep_witnessed": bool(keep_witnessed),
        "examined": len(candidates),
        "purgeable": summaries,
        "protected": {reason: sorted(ids) for reason, ids in protected.items()},
        "estimated_bytes": sum(s["bytes"] for s in summaries),
        "purged": 0,
        "anchors_purged": [],
        "events_purged": [],
        "anchor_id": None,
    }
    if dry_run or not purgeable:
        return report
    with store.transaction():
        _ev, anchor = system_event(
            store,
            scope=scope,
            object_type="system",
            event_type="deleted",
            payload={
                "op": "purge",
                "metadata": {
                    "older_than_days": days,
                    "cutoff": cutoff,
                    "keep_witnessed": bool(keep_witnessed),
                    "atoms": len(purgeable),
                },
            },
        )
        for atom, summary in zip(purgeable, summaries):
            store.add_history(
                memory_id=atom.memory_id,
                old_memory=None,
                new_memory=None,
                event=HistoryEvent.PURGE.value,
                created_at=atom.created_at or now,
                updated_at=now,
                is_deleted=1,
                actor_id=atom.actor_id,
                anchor_id=anchor.anchor_id,
                atom_id=atom.atom_id,
                details={k: v for k, v in summary.items() if k not in ("atom_id", "memory_id", "bytes")}
                | {"cutoff": cutoff, "older_than_days": days},
            )
        removed = store.purge_rows([a.atom_id for a in purgeable], evidence=not keep_witnessed)
    for atom_id in removed["atoms"]:
        index.delete(atom_id)
    report["purged"] = len(removed["atoms"])
    report["anchors_purged"] = list(removed["anchors"])
    report["events_purged"] = list(removed["events"])
    report["anchor_id"] = anchor.anchor_id
    return report


__all__ = ["PROTECTED_REASONS", "cutoff_instant", "purge"]
