"""PersonalState builder (HMS §11.2): preferences (valid / invalidated), habits, constraints, facts, timeline.
PersonalState 构建器（HMS §11.2）：偏好（有效 / 失效）、习惯、约束、事实、时间线。
"""

# Purpose (EN): Derive one subject's PersonalState / Shared Card from the State Index through public store reads.
# 目的（中）：通过存储的公开读方法，从状态索引推导一个主体的 PersonalState / Shared Card。

# Input (EN): The HMS facade (its store), Mem0 scope filters, a subject, an optional audience and as_of.
# 输入（中）：HMS 门面（其 store）、Mem0 范围过滤、主体、可选受众与 as_of。

# Output (EN): A PersonalState whose every entry carries anchor_ids.
# 输出（中）：每一项都带 anchor_ids 的 PersonalState。

# State effect (EN): None (read-only).
# 状态影响（中）：无（只读）。

from __future__ import annotations

from typing import Any

from hms_elite.enums import AtomType, ObservationType, Visibility, rank
from hms_elite.models import MemoryAtom, Scope
from hms_elite.state.kinds import PersonalState, StateItem, StateKind, item_from_atom, scope_from_filters
from hms_elite.timeutil import to_utc_iso

_ASC = "time_anchor ASC, created_at ASC, atom_id ASC"
_SUBJECT_ALIASES = {"user": {"user"}, "assistant": {"assistant"}, "project": {"project"}}


def _subject_matches(atom: MemoryAtom, subject: str) -> bool:
    """True when the atom is about `subject` ('user' also admits atoms without a subject).
    原子是否关于 `subject`（'user' 也接纳无主语的原子）。
    """
    wanted = subject.strip().lower()
    have = (atom.subject or "").strip().lower()
    if not have:
        return wanted == "user"
    aliases = _SUBJECT_ALIASES.get(wanted, {wanted})
    return have in aliases


def _visible(atom: MemoryAtom, audience: str | None) -> bool:
    """Visibility lattice gate: keep atoms at least as public as the audience.
    可见性格门控：保留可见性不低于受众的原子。
    """
    return audience is None or rank(str(atom.visibility)) >= rank(audience)


def _superseded_by(store: Any, atom: MemoryAtom) -> str | None:
    """Text of the successor version of an archived atom, via the lineage head.
    通过血缘头获得已归档原子后继版本的文本。
    """
    if not atom.memory_id:
        return None
    head = store.resolve_head(atom.memory_id)
    if head is None or head.atom_id == atom.atom_id:
        return None
    return head.text


def build_personal_state(
    hms: Any,
    *,
    filters: dict,
    subject: str = "user",
    audience: Visibility | str | None = None,
    as_of: Any = None,
    timeline_limit: int = 20,
    kind: StateKind | str = StateKind.personal_state,
) -> PersonalState:
    """Build the PersonalState of a subject inside a scope (public store reads only, no LLM).
    在某范围内构建一个主体的 PersonalState（仅用存储公开读方法，无 LLM）。

    Input / 输入：
        hms: The HMS facade (only `hms.store` is used).
        hms：HMS 门面（仅使用 `hms.store`）。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        subject: 'user' | 'assistant' | 'project' | an entity display text.
        subject：'user' | 'assistant' | 'project' | 实体显示文本。

        audience: Visibility floor (None → no gate; the Shared Card uses 'team').
        audience：可见性下限（None → 不门控；Shared Card 使用 'team'）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

        timeline_limit: Keep the newest N timeline entries.
        timeline_limit：时间线保留最新的 N 条。

        kind: 'personal_state' or 'shared_card'.
        kind：'personal_state' 或 'shared_card'。

    Output / 输出：
        PersonalState with valid and invalidated preference groups (each entry dated and witnessed).
        含有效与失效偏好两组（每项带时间与见证）的 PersonalState。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    scope: Scope = scope_from_filters(filters)
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    aud = None if audience is None else str(getattr(audience, "value", audience))
    # Every version (archived included) so invalidated preferences keep their valid_to and successor.
    # 取全部版本（含已归档），使失效偏好保留 valid_to 与后继。
    atoms = store.axis_select(scope, include_expired=True, order=_ASC)
    # Constraints extracted from a user's own messages carry subject 'project' (§11.1 R2/R3); they bind the user.
    # 从用户自己消息中抽取的约束主语为 'project'（§11.1 R2/R3）；它们约束的是该用户。
    atoms = [
        a
        for a in atoms
        if _visible(a, aud)
        and (
            _subject_matches(a, subject)
            or (
                subject == "user"
                and str(a.obs_type or "") == ObservationType.constraint.value
                and (a.subject or "") == "project"
            )
        )
    ]

    valid_prefs: list[StateItem] = []
    invalid_prefs: list[StateItem] = []
    habits: list[StateItem] = []
    constraints: list[StateItem] = []
    facts: list[StateItem] = []
    timeline: list[StateItem] = []
    for atom in atoms:
        live = atom.is_valid_at(instant)
        obs = None if atom.obs_type is None else str(atom.obs_type)
        kind_of = str(atom.atom_type)
        if obs == ObservationType.preference_event.value:
            if live:
                valid_prefs.append(item_from_atom(atom))
            elif atom.archived_reason is not None or (atom.valid_to is not None and atom.valid_to <= instant):
                # Archived or expired at as_of: the invalidated group keeps the time it stopped holding.
                # 在 as_of 已归档或已过期：失效组保留其停止成立的时间。
                invalid_prefs.append(
                    item_from_atom(
                        atom,
                        archived_reason=atom.archived_reason,
                        invalidated_at=atom.valid_to,
                        superseded_by=_superseded_by(store, atom),
                    )
                )
            continue
        if not live:
            continue
        if obs == ObservationType.habit.value:
            habits.append(item_from_atom(atom))
        elif obs == ObservationType.constraint.value:
            constraints.append(item_from_atom(atom))
        elif kind_of in (AtomType.event.value, AtomType.episode.value):
            timeline.append(item_from_atom(atom))
        elif obs in (ObservationType.fact.value, ObservationType.temporal_fact.value):
            facts.append(item_from_atom(atom))
    if timeline_limit >= 0:
        timeline = timeline[-timeline_limit:] if timeline_limit else []
    return PersonalState(
        kind=str(getattr(kind, "value", kind)),
        scope=scope,
        as_of=instant,
        subject=subject,
        audience=aud,
        preferences_valid=valid_prefs,
        preferences_invalid=invalid_prefs,
        habits=habits,
        constraints=constraints,
        profile_facts=facts,
        timeline=timeline,
    )


def build_shared_card(
    hms: Any,
    *,
    filters: dict,
    subject: str = "user",
    audience: Visibility | str = Visibility.team,
    as_of: Any = None,
    timeline_limit: int = 20,
) -> PersonalState:
    """The Shared Card: a PersonalState gated to what the audience may see (default 'team').
    Shared Card：按受众可见性门控的 PersonalState（默认 'team'）。

    Input / 输入：
        hms: The HMS facade.
        hms：HMS 门面。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        subject: Subject selector as in build_personal_state.
        subject：主体选择器，同 build_personal_state。

        audience: Visibility floor; private atoms never appear on a shared card.
        audience：可见性下限；私有原子绝不出现在共享卡上。

        as_of: Reference instant.
        as_of：参考时刻。

        timeline_limit: Keep the newest N timeline entries.
        timeline_limit：时间线保留最新的 N 条。

    Output / 输出：
        PersonalState with kind 'shared_card'.
        kind 为 'shared_card' 的 PersonalState。

    State effect / 状态影响：
        None.
        无。
    """
    aud = str(getattr(audience, "value", audience))
    if rank(aud) < rank(Visibility.team.value):
        aud = Visibility.team.value
    return build_personal_state(
        hms,
        filters=filters,
        subject=subject,
        audience=aud,
        as_of=as_of,
        timeline_limit=timeline_limit,
        kind=StateKind.shared_card,
    )


__all__ = ["build_personal_state", "build_shared_card"]
