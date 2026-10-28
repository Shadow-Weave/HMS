"""Slot rules, ContextItem builders, kappa/invariant/evidence synthesis and the next-action rule of the compiler.
编译器的槽位规则、ContextItem 构造器、κ/不变量/证据合成与下一步规则。
"""

# Purpose (EN): Decide which of the six ContextIR slots an atom belongs to and build every kind of ContextItem.
# 目的（中）：判定原子归属六个 ContextIR 槽位中的哪一个，并构造每一类 ContextItem。

# Input (EN): MemoryAtoms, InterfaceEPs, Anchors, already-built ContextItems and the reference instant as_of.
# 输入（中）：MemoryAtom、InterfaceEP、Anchor、已构造的 ContextItem 与参考时刻 as_of。

# Output (EN): Slot values, ContextItems and the next-action string.
# 输出（中）：槽位值、ContextItem 与下一步字符串。

# State effect (EN): None; pure functions over models.
# 状态影响（中）：无；基于模型的纯函数。

from __future__ import annotations

from typing import Any, Callable, Iterable

from hms_elite.enums import AtomType, InvariantKind, ObservationType, Reward, Slot, SourceKind
from hms_elite.llms.rule import normalize
from hms_elite.memory.utils import estimate_tokens
from hms_elite.models import Anchor, ContextItem, InterfaceEP, MemoryAtom
from hms_elite.timeutil import to_utc_iso

PREFERENCE_OBS_TYPES: frozenset[str] = frozenset({ObservationType.preference_event.value, ObservationType.habit.value})
NON_NEGATIVE_REWARDS: frozenset[str] = frozenset({Reward.positive.value, Reward.neutral.value, Reward.unknown.value})
_REAPPLY_PREFIX = "Do not re-apply"


def _rule_archived_or_expired(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (1): archived, or valid_to already reached at as_of (a future valid_to is NOT audit).
    规则 (1)：已归档，或 valid_to 在 as_of 已到达（未来的 valid_to 不进入 audit）。
    """
    if atom.archived_reason is not None:
        return True
    return atom.valid_to is not None and atom.valid_to <= as_of


def _rule_skill(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (2): source_kind == 'skill'.
    规则 (2)：source_kind == 'skill'。
    """
    return str(atom.source_kind) == SourceKind.skill.value


def _rule_constraint(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (3): obs_type == 'constraint'.
    规则 (3)：obs_type == 'constraint'。
    """
    return atom.obs_type is not None and str(atom.obs_type) == ObservationType.constraint.value


def _rule_preference(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (4): obs_type in {preference_event, habit}.
    规则 (4)：obs_type ∈ {preference_event, habit}。
    """
    return atom.obs_type is not None and str(atom.obs_type) in PREFERENCE_OBS_TYPES


def _rule_episode(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (5): atom_type == 'episode'.
    规则 (5)：atom_type == 'episode'。
    """
    return str(atom.atom_type) == AtomType.episode.value


def _rule_fact(atom: MemoryAtom, as_of: str) -> bool:
    """Rule (6): everything else is a fact.
    规则 (6)：其余皆为事实。
    """
    return True


# Ordered and total: the first matching rule wins (§15.1).
# 有序且完备：第一条匹配的规则胜出（§15.1）。
SLOT_RULES: tuple[tuple[str, Callable[[MemoryAtom, str], bool], Slot], ...] = (
    ("archived_or_expired", _rule_archived_or_expired, Slot.audit),
    ("skill", _rule_skill, Slot.skills),
    ("constraint", _rule_constraint, Slot.constraints),
    ("preference", _rule_preference, Slot.preferences),
    ("episode", _rule_episode, Slot.episodes),
    ("fact", _rule_fact, Slot.facts),
)


def audit_reason(atom: MemoryAtom) -> str:
    """The audit explain reason of an archived or expired atom: 'archived:<archived_reason or expired>'.
    已归档或已过期原子的 audit 解释原因：'archived:<archived_reason 或 expired>'。

    Input / 输入：
        atom: The atom routed to the audit slot by rule (1).
        atom：被规则 (1) 分配到 audit 槽位的原子。

    Output / 输出：
        'archived:' + (archived_reason or 'expired').
        'archived:' + (archived_reason 或 'expired')。

    State effect / 状态影响：
        None.
        无。
    """
    return "archived:" + (atom.archived_reason or "expired")


def assign_slot(atom: MemoryAtom, *, as_of: str, reason: str | None = None) -> Slot:
    """Assign the slot of an atom by the ordered total SLOT_RULES; an explicit reason (e.g. 'revised') forces audit.
    按有序且完备的 SLOT_RULES 为原子分配槽位；显式 reason（如 'revised'）强制进入 audit。

    Input / 输入：
        atom: The candidate atom.
        atom：候选原子。

        as_of: Reference instant deciding expiry (any form accepted by to_utc_iso).
        as_of：判定过期的参考时刻（to_utc_iso 接受的任意形式）。

        reason: Non-None when the atom was reached through a revision edge; forces Slot.audit.
        reason：原子经 revision 边到达时非空；强制 Slot.audit。

    Output / 输出：
        The Slot value.
        Slot 值。

    State effect / 状态影响：
        None.
        无。
    """
    if reason is not None:
        return Slot.audit
    instant = to_utc_iso(as_of)
    for _name, predicate, slot in SLOT_RULES:
        if predicate(atom, instant):
            return slot
    # 不可达：规则 (6) 是全覆盖的
    return Slot.facts  # unreachable: rule (6) is total


def _episode_explain(atom: MemoryAtom) -> dict[str, Any]:
    """The {'r','a','y','u','ep_id'|'interface_ep_id','kappa'} explain of an episode atom (from content['slots']).
    情节原子的 {'r','a','y','u','ep_id'|'interface_ep_id','kappa'} 解释（取自 content['slots']）。
    """
    content = atom.content if isinstance(atom.content, dict) else {}
    slots = content.get("slots") if isinstance(content.get("slots"), dict) else {}
    out: dict[str, Any] = {}
    if slots:
        out["r"] = str(slots.get("r") or Reward.unknown.value)
        out["a"] = str(slots.get("a") or "")
        out["y"] = str(slots.get("y") or "")
        out["u"] = str(slots.get("u") or "")
        kappa = slots.get("kappa") if isinstance(slots.get("kappa"), dict) else {}
        out["kappa"] = [str(c) for c in (kappa.get("constraints") or [])]
    kind = str(atom.source_kind)
    if kind == SourceKind.interface_ep.value:
        out["interface_ep_id"] = str(content.get("interface_ep_id") or atom.source_id)
    elif kind == SourceKind.ep.value:
        out["ep_id"] = str(content.get("ep_id") or atom.source_id)
    return out


def item_from_atom(
    atom: MemoryAtom,
    *,
    slot: Slot | str,
    score: float,
    anchor_ids: list[str],
    source: str,
    hops: int,
    explain: dict[str, Any] | None,
) -> ContextItem:
    """Build the ContextItem of an atom (item_id == atom_id, tokens == estimate_tokens(text)).
    构造原子的 ContextItem（item_id == atom_id，tokens == estimate_tokens(text)）。

    Input / 输入：
        atom: The atom (hydrated).
        atom：原子（已水合）。

        slot: Target slot (assign_slot result).
        slot：目标槽位（assign_slot 的结果）。

        score: Retrieval score.
        score：检索分数。

        anchor_ids: Witness anchors of the item (non-empty; disputed ones included, gated by audience).
        anchor_ids：该项的见证锚点（非空；含有争议锚点，已按受众门控）。

        source: 'axis' | 'graph' | 'interface_ep' | 'system'.
        source：'axis' | 'graph' | 'interface_ep' | 'system'。

        hops: Graph distance from the nearest axis seed (0 for axis items).
        hops：距最近轴种子的图跳数（轴项为 0）。

        explain: Extra explain keys merged over the slot defaults (episode slots, audit reason).
        explain：合并到槽位默认值之上的额外解释键（情节槽位、audit 原因）。

    Output / 输出：
        ContextItem.
        ContextItem。

    State effect / 状态影响：
        None.
        无。
    """
    slot_value = Slot(str(slot))
    base: dict[str, Any] = {}
    if slot_value == Slot.episodes or str(atom.atom_type) == AtomType.episode.value:
        base.update(_episode_explain(atom))
    if slot_value == Slot.audit:
        base["reason"] = audit_reason(atom)
    base.update(explain or {})
    ep_id = atom.source_id if str(atom.source_kind) == SourceKind.ep.value else None
    return ContextItem(
        item_id=atom.atom_id,
        slot=slot_value,
        text=atom.text,
        atom_id=atom.atom_id,
        memory_id=atom.memory_id or None,
        ep_id=ep_id,
        score=float(score),
        time_anchor=atom.time_anchor,
        valid_from=atom.valid_from,
        valid_to=atom.valid_to,
        scale_level=atom.scale_level,
        atom_type=atom.atom_type,
        visibility=atom.visibility,
        source=source,  # type: ignore[arg-type]
        hops=int(hops),
        tokens=estimate_tokens(atom.text),
        explain=base,
        anchor_ids=list(anchor_ids),
    )


def normalized_texts(items: Iterable[ContextItem]) -> set[str]:
    """Normalized (§11.3 rule 1) texts of a list of items, used for constraint de-duplication.
    一组项的归一化（§11.3 规则 1）文本，用于约束去重。

    Input / 输入：
        items: ContextItems.
        items：ContextItem 列表。

    Output / 输出：
        Set of normalized texts.
        归一化文本集合。

    State effect / 状态影响：
        None.
        无。
    """
    return {normalize(item.text) for item in items}


def kappa_items(episodes: list[ContextItem], selected_constraint_texts: set[str]) -> list[ContextItem]:
    """Synthesize '<atom_id>#kappa:<i>' constraint items for κ constraints not already selected or archived.
    为尚未被选中或归档的 κ 约束合成 '<atom_id>#kappa:<i>' 约束项。

    Input / 输入：
        episodes: Selected episode items in slot order (explain['kappa'] carries the κ constraint texts).
        episodes：按槽位顺序的已选情节项（explain['kappa'] 携带 κ 约束文本）。

        selected_constraint_texts: Normalized texts of every selected constraint AND audit item.
        selected_constraint_texts：所有已选约束项与 audit 项的归一化文本。

    Output / 输出：
        Constraint items (source 'system', anchors = the episode's anchors); one per text, first episode wins.
        约束项（source 为 'system'，锚点 = 情节锚点）；每个文本一项，先出现的情节优先。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[ContextItem] = []
    emitted: set[str] = set(selected_constraint_texts)
    for episode in episodes:
        constraints = episode.explain.get("kappa") or []
        for index, text in enumerate(constraints):
            text = str(text)
            key = normalize(text)
            if not key or key in emitted:
                continue
            emitted.add(key)
            out.append(
                ContextItem(
                    item_id=f"{episode.atom_id or episode.item_id}#kappa:{index}",
                    slot=Slot.constraints,
                    text=text,
                    atom_id=episode.atom_id,
                    memory_id=episode.memory_id,
                    ep_id=episode.ep_id,
                    score=episode.score,
                    time_anchor=episode.time_anchor,
                    valid_from=episode.valid_from,
                    valid_to=episode.valid_to,
                    scale_level=episode.scale_level,
                    atom_type=episode.atom_type,
                    visibility=episode.visibility,
                    source="system",
                    hops=episode.hops,
                    tokens=estimate_tokens(text),
                    explain={
                        "reason": "kappa",
                        "from_item_id": episode.item_id,
                        "index": index,
                        "disputed_anchor_ids": list(episode.explain.get("disputed_anchor_ids") or []),
                    },
                    anchor_ids=list(episode.anchor_ids),
                )
            )
    return out


def invariant_items(
    iep: InterfaceEP, iep_atom_id: str, selected_constraint_texts: set[str]
) -> tuple[list[ContextItem], list[ContextItem]]:
    """Turn an InterfaceEP's invariants/residuals into '<iep_atom_id>#inv:<i>' constraint and audit items.
    将 InterfaceEP 的不变量/残差转为 '<iep_atom_id>#inv:<i>' 约束项与 audit 项。

    Input / 输入：
        iep: The refined InterfaceEP.
        iep：已细化的 InterfaceEP。

        iep_atom_id: Its episode atom id (== node id), the item id prefix.
        iep_atom_id：其情节原子 id（== 节点 id），作为项 id 前缀。

        selected_constraint_texts: Normalized texts of every selected constraint AND audit item (duplicates skipped).
        selected_constraint_texts：所有已选约束项与 audit 项的归一化文本（重复者跳过）。

    Output / 输出：
        (constraint items from kind=constraint invariants, audit items for failed_path/stable_fact and residuals).
        （来自 kind=constraint 不变量的约束项，failed_path/stable_fact 与残差的 audit 项）。

    State effect / 状态影响：
        None.
        无。
    """
    constraints: list[ContextItem] = []
    audit: list[ContextItem] = []
    seen: set[str] = set(selected_constraint_texts)
    common: dict[str, Any] = {
        "atom_id": iep_atom_id,
        "time_anchor": None,
        "source": "interface_ep",
        "visibility": iep.slots.kappa.visibility,
    }
    index = 0
    for inv in iep.invariants:
        item_id = f"{iep_atom_id}#inv:{index}"
        index += 1
        kind = str(inv.kind)
        explain = {
            "reason": f"invariant:{kind}",
            "interface_ep_id": iep.interface_ep_id,
            "support_node_ids": list(inv.support_node_ids),
            "confidence": inv.confidence,
            "disputed_anchor_ids": [],
        }
        if kind == InvariantKind.constraint.value:
            key = normalize(inv.text)
            if key in seen:
                continue
            seen.add(key)
            constraints.append(
                ContextItem(
                    item_id=item_id,
                    slot=Slot.constraints,
                    text=inv.text,
                    score=float(inv.confidence),
                    scale_level=iep.scale,
                    atom_type=AtomType.episode,
                    tokens=estimate_tokens(inv.text),
                    explain=explain,
                    anchor_ids=list(inv.anchor_ids),
                    **common,
                )
            )
        else:
            audit.append(
                ContextItem(
                    item_id=item_id,
                    slot=Slot.audit,
                    text=inv.text,
                    score=float(inv.confidence),
                    scale_level=iep.scale,
                    atom_type=AtomType.episode,
                    tokens=estimate_tokens(inv.text),
                    explain=explain,
                    anchor_ids=list(inv.anchor_ids),
                    **common,
                )
            )
    for res in iep.residuals:
        item_id = f"{iep_atom_id}#inv:{index}"
        index += 1
        anchors = list(res.anchor_ids) or list(iep.evidence_anchor_ids)
        audit.append(
            ContextItem(
                item_id=item_id,
                slot=Slot.audit,
                text=res.text,
                score=0.0,
                scale_level=iep.scale,
                atom_type=AtomType.episode,
                tokens=estimate_tokens(res.text),
                explain={
                    "reason": f"residual:{res.kind}",
                    "interface_ep_id": iep.interface_ep_id,
                    "support_node_ids": list(res.support_node_ids),
                    "edge_ids": list(res.edge_ids),
                    "disputed_anchor_ids": [],
                },
                anchor_ids=anchors,
                **common,
            )
        )
    return constraints, audit


def evidence_items(anchors: list[Anchor]) -> list[ContextItem]:
    """Build one 'anc:<anchor_id>' audit item per anchor, ordered by (timestamp, source_ref, anchor_id).
    为每个锚点构造一个 'anc:<anchor_id>' audit 项，按 (timestamp, source_ref, anchor_id) 排序。

    Input / 输入：
        anchors: Anchor rows (duplicates collapsed by anchor_id).
        anchors：锚点行（按 anchor_id 去重）。

    Output / 输出：
        Evidence ContextItems (text = anchor.evidence_line()).
        证据 ContextItem（text = anchor.evidence_line()）。

    State effect / 状态影响：
        None.
        无。
    """
    unique: dict[str, Anchor] = {}
    for anchor in anchors:
        unique.setdefault(anchor.anchor_id, anchor)
    ordered = sorted(unique.values(), key=lambda a: (a.timestamp, a.source_ref or "", a.anchor_id))
    out: list[ContextItem] = []
    for anchor in ordered:
        text = anchor.evidence_line()
        out.append(
            ContextItem(
                item_id=f"anc:{anchor.anchor_id}",
                slot=Slot.audit,
                text=text,
                time_anchor=anchor.timestamp,
                visibility=anchor.permission_scope,
                source="system",
                tokens=estimate_tokens(text),
                explain={
                    "kind": "evidence",
                    "source_ref": anchor.source_ref,
                    "hash": anchor.hash,
                    "timestamp": anchor.timestamp,
                    "revoked_at": anchor.revoked_at,
                    "permission_scope": str(anchor.permission_scope),
                },
                anchor_ids=[anchor.anchor_id],
            )
        )
    return out


def _newest(items: list[ContextItem], reward: str) -> ContextItem | None:
    """The newest episode item (by time_anchor, then item_id) carrying reward `reward`.
    带有奖励 `reward` 的最新情节项（按 time_anchor，其次 item_id）。
    """
    matching = [item for item in items if str(item.explain.get("r") or "") == reward]
    if not matching:
        return None
    return max(matching, key=lambda item: (item.time_anchor or "", item.item_id))


def derive_next_action(
    *,
    episodes: list[ContextItem],
    constraints: list[ContextItem],
    residuals: list[str],
    conflicts: list[tuple[str, str]],
    goal: str,
) -> str:
    """The deterministic next-action rule: newest negative vs newest positive episode, residual, conflict, fallback.
    确定性的下一步规则：最新负向与最新正向情节、残差、冲突、兜底。

    Input / 输入：
        episodes: Episode items (pre-budget candidate set).
        episodes：情节项（预算前的候选集）。

        constraints: Constraint items (κ items included).
        constraints：约束项（含 κ 项）。

        residuals: Residual texts of refined InterfaceEPs.
        residuals：已细化 InterfaceEP 的残差文本。

        conflicts: (text, text) pairs of unresolved conflicts.
        conflicts：未解决冲突的 (文本, 文本) 对。

        goal: The compile goal (query).
        goal：编译目标（查询）。

    Output / 输出：
        The next-action sentence.
        下一步句子。

    State effect / 状态影响：
        None.
        无。
    """
    neg = _newest(episodes, Reward.negative.value)
    pos = _newest(episodes, Reward.positive.value)
    kept = [c.text for c in constraints if not c.text.startswith(_REAPPLY_PREFIX)][:3]
    suffix = f" while keeping constraints: {'; '.join(kept)}" if kept else ""
    if neg is not None and (pos is None or (neg.time_anchor or "") > (pos.time_anchor or "")):
        return f"Do not re-apply '{neg.explain.get('a', '')}'; investigate '{neg.explain.get('y', '')}'{suffix}"
    if pos is not None:
        return f"Continue from '{pos.explain.get('a', '')}' (last outcome: {pos.explain.get('y', '')}){suffix}"
    if residuals:
        return f"Resolve: {residuals[0]}"
    if conflicts:
        x, y = conflicts[0]
        return f"Resolve conflict between '{x}' and '{y}'"
    return f"Continue: {goal}"
