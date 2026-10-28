"""GroupState builder (HMS §11.2): shared constraints, decisions and conflicts between members.
GroupState 构建器（HMS §11.2）：共享约束、决策与成员间冲突。
"""

# Purpose (EN): Derive the state of a member set from each member's live atoms (public store reads, no LLM).
# 目的（中）：由每个成员的存活原子推导一组成员的状态（存储公开读方法，无 LLM）。

# Input (EN): The HMS facade, a list of member scope dicts, as_of.
# 输入（中）：HMS 门面、成员范围字典列表、as_of。

# Output (EN): GroupState whose every entry carries anchor_ids; MemberConflict pairs cite both sides' witnesses.
# 输出（中）：每一项都带 anchor_ids 的 GroupState；MemberConflict 对引用双方的见证。

# State effect (EN): None (read-only).
# 状态影响（中）：无（只读）。

from __future__ import annotations

import re
from typing import Any

from hms_elite.enums import AtomType, ObservationType, Reward
from hms_elite.models import MemoryAtom, Scope
from hms_elite.state.kinds import GroupState, MemberConflict, StateItem, item_from_atom, scope_from_filters
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.text import normalize_entity

_ASC = "time_anchor ASC, created_at ASC, atom_id ASC"
# Decision cues (EN + ZH) recognised in atom text; deterministic, no LLM.
# 原子文本中识别的决策线索（英 + 中）；确定性，无 LLM。
DECISION_RE = re.compile(
    r"\b(decided|decision|agreed|approved|resolved to|signed off|will ship|going with)\b|决定|同意|批准|拍板", re.I
)
# Exclusive predicates: one subject holds at most one object at a time.
# 排他谓词：一个主语同一时间至多持有一个宾语。
EXCLUSIVE_PREDICATES = frozenset({"lives_in", "works_at", "studies_at", "is_named", "prefers"})
_POLAR = {"must": "must_not", "must_not": "must", "likes": "dislikes", "dislikes": "likes"}


def member_key(scope: Scope) -> str:
    """Display key of a member scope ('alice', 'alice/coder', ...).
    成员范围的显示键（'alice'、'alice/coder'……）。

    Input / 输入：
        scope: The member scope.
        scope：成员范围。

    Output / 输出：
        Slash-joined set ids in user/agent/run order.
        按 user/agent/run 顺序、以斜杠连接的已设置 id。

    State effect / 状态影响：
        None.
        无。
    """
    return "/".join(v for v in (scope.user_id, scope.agent_id, scope.run_id) if v)


def live_atoms(store: Any, scope: Scope, *, as_of: str) -> list[MemoryAtom]:
    """Live atoms of a scope at `as_of`, ascending by time (the axis gate of every builder).
    某范围在 `as_of` 的存活原子，按时间升序（每个构建器的轴门控）。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        scope: Scope filter.
        scope：范围过滤。

        as_of: Validity instant.
        as_of：有效性时刻。

    Output / 输出：
        Atoms ordered by (time_anchor, created_at, atom_id).
        按 (time_anchor, created_at, atom_id) 排序的原子。

    State effect / 状态影响：
        None.
        无。
    """
    return list(store.axis_select(scope, as_of=as_of, order=_ASC))


def _objects_of(atom: MemoryAtom) -> tuple[str, ...]:
    """Normalized objects of an observation atom (from content['objects']).
    观察原子的归一化宾语（取自 content['objects']）。
    """
    objs = atom.content.get("objects") if isinstance(atom.content, dict) else None
    if not isinstance(objs, list):
        return ()
    return tuple(sorted(normalize_entity(str(o)) for o in objs if str(o).strip()))


def is_decision(atom: MemoryAtom) -> bool:
    """Decision rule: DECISION_RE hits the text, or the atom is a positive-reward L1 episode.
    决策规则：DECISION_RE 命中文本，或该原子是正回报的 L1 情节。

    Input / 输入：
        atom: A live atom.
        atom：存活原子。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    if DECISION_RE.search(atom.text):
        return True
    if str(atom.atom_type) == AtomType.episode.value and isinstance(atom.content, dict):
        slots = atom.content.get("slots") if isinstance(atom.content.get("slots"), dict) else {}
        return str(slots.get("r") or "") == Reward.positive.value
    return False


def find_member_conflicts(
    members: list[tuple[str, list[MemoryAtom]]],
) -> list[MemberConflict]:
    """Cross-member conflicts: same subject and (exclusive predicate with different objects | polar predicates).
    成员间冲突：同主语且（排他谓词宾语不同 | 极性相反的谓词）。

    Input / 输入：
        members: [(member_key, live atoms of that member), ...].
        members：[(成员键, 该成员的存活原子), ...]。

    Output / 输出：
        MemberConflict list ordered by (left member, right member, left atom_id, right atom_id).
        按 (左成员, 右成员, 左原子 id, 右原子 id) 排序的 MemberConflict 列表。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[MemberConflict] = []
    for i, (ma, atoms_a) in enumerate(members):
        for mb, atoms_b in members[i + 1 :]:
            for x in atoms_a:
                if not x.predicate or str(x.atom_type) != AtomType.state.value:
                    continue
                for y in atoms_b:
                    if not y.predicate or str(y.atom_type) != AtomType.state.value:
                        continue
                    if (x.subject or "") != (y.subject or ""):
                        continue
                    ox, oy = _objects_of(x), _objects_of(y)
                    reason = ""
                    if x.predicate == y.predicate and x.predicate in EXCLUSIVE_PREDICATES and ox and oy and ox != oy:
                        reason = f"exclusive predicate '{x.predicate}' with different objects"
                    elif _POLAR.get(str(x.predicate)) == y.predicate and ox and ox == oy:
                        reason = f"polar predicates '{x.predicate}' vs '{y.predicate}' on the same object"
                    if not reason:
                        continue
                    left = item_from_atom(x, member=ma)
                    right = item_from_atom(y, member=mb)
                    out.append(
                        MemberConflict(
                            members=[ma, mb],
                            left=left,
                            right=right,
                            reason=reason,
                            anchor_ids=sorted(set(left.anchor_ids) | set(right.anchor_ids)),
                        )
                    )
    out.sort(key=lambda c: (c.members[0], c.members[1], c.left.atom_id or "", c.right.atom_id or ""))
    return out


def build_group_state(hms: Any, *, members: list[dict], as_of: Any = None) -> GroupState:
    """Build the GroupState of a member set (each member is a Mem0 scope dict).
    构建一组成员（每个成员是 Mem0 范围字典）的 GroupState。

    Input / 输入：
        hms: The HMS facade (only `hms.store` is used).
        hms：HMS 门面（仅使用 `hms.store`）。

        members: Scope dicts, one per member (>= 1 scope key each; VALIDATION_001 otherwise).
        members：范围字典，每个成员一个（各至少一个范围键；否则 VALIDATION_001）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

    Output / 输出：
        GroupState: shared constraints (held by >= 2 members), decisions per member, conflicts between members.
        GroupState：共享约束（≥2 名成员持有）、各成员的决策、成员间冲突。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    scopes = [scope_from_filters(m) for m in members]
    keys = [member_key(s) for s in scopes]
    per_member: list[tuple[str, list[MemoryAtom]]] = [
        (key, live_atoms(store, scope, as_of=instant)) for key, scope in zip(keys, scopes)
    ]

    # Shared constraints: same normalized text held by at least two members; witnesses are the union.
    # 共享约束：至少两名成员持有相同归一化文本；见证取并集。
    holders: dict[str, dict[str, MemoryAtom]] = {}
    for key, atoms in per_member:
        for atom in atoms:
            if str(atom.obs_type or "") == ObservationType.constraint.value:
                holders.setdefault(normalize_entity(atom.text), {}).setdefault(key, atom)
    shared: list[StateItem] = []
    for _text, by_member in sorted(holders.items()):
        if len(by_member) < 2:
            continue
        first = by_member[min(by_member)]
        anchors = sorted({a for atom in by_member.values() for a in atom.anchor_ids})
        item = item_from_atom(first, members=sorted(by_member))
        shared.append(item.model_copy(update={"anchor_ids": anchors}))

    decisions: list[StateItem] = []
    for key, atoms in per_member:
        decisions.extend(item_from_atom(atom, member=key) for atom in atoms if is_decision(atom))
    decisions.sort(key=lambda i: (i.time or "", i.atom_id or ""))

    return GroupState(
        scope=scopes[0],
        as_of=instant,
        members=keys,
        shared_constraints=shared,
        decisions=decisions,
        conflicts_between_members=find_member_conflicts(per_member),
    )


__all__ = [
    "DECISION_RE",
    "EXCLUSIVE_PREDICATES",
    "build_group_state",
    "find_member_conflicts",
    "is_decision",
    "live_atoms",
    "member_key",
]
