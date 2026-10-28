"""Enterprise State builders (HMS §11.3): CustomerIssue / Incident / MeetingDecision / AccountHandoff.
企业状态构建器（HMS §11.3）：CustomerIssue / Incident / MeetingDecision / AccountHandoff。
"""

# Purpose (EN): Map the four enterprise state kinds onto compile() / why() / handoff() / checkpoints plus axis reads.
# 目的（中）：把四种企业状态种类映射到 compile() / why() / handoff() / 检查点与轴读取之上。

# Input (EN): The HMS facade, Mem0 scope filters, optional issue / question / goal, as_of.
# 输入（中）：HMS 门面、Mem0 范围过滤、可选的 issue / question / goal、as_of。

# Output (EN): CustomerIssueState, IncidentState, MeetingDecisionState, AccountHandoffState (every entry witnessed).
# 输出（中）：CustomerIssueState、IncidentState、MeetingDecisionState、AccountHandoffState（每项均有见证）。

# State effect (EN): Read-only, except build_account_handoff_state(checkpoint=True) which takes a checkpoint via
# HMS.handoff.
# 状态影响（中）：只读；例外是 build_account_handoff_state(checkpoint=True) 会经 HMS.handoff 建立检查点。

from __future__ import annotations

import re
from typing import Any

from hms_elite.enums import AtomType, CompileMode, ObservationType, Reward
from hms_elite.models import EP, ContextIR, MemoryAtom, Scope
from hms_elite.state.group import is_decision, live_atoms
from hms_elite.state.kinds import (
    AccountHandoffState,
    CustomerIssueState,
    DetectiveReport,
    IncidentState,
    MeetingDecisionState,
    RollbackPoint,
    StateItem,
    context_sections,
    item_from_atom,
    item_from_context_item,
    scope_from_filters,
)
from hms_elite.timeutil import to_utc_iso

# Deadline cues (EN + ZH); a deadline is a dated atom whose text carries one.
# 截止线索（英 + 中）；截止项 = 文本含线索且带日期的原子。
DEADLINE_RE = re.compile(r"\b(by|before|until|due|deadline|no later than)\b|截止|之前|前完成|期限", re.I)
_GENERIC_SUBJECTS = frozenset({"", "user", "assistant", "project"})
_STATUS_OF_REWARD = {
    Reward.positive.value: "resolved",
    Reward.negative.value: "blocked",
    Reward.neutral.value: "open",
    Reward.unknown.value: "open",
}


def _constraint_items(atoms: list[MemoryAtom]) -> list[StateItem]:
    """Constraint atoms as StateItems.
    约束原子转为 StateItem。
    """
    return [item_from_atom(a) for a in atoms if str(a.obs_type or "") == ObservationType.constraint.value]


def latest_rollback_point(store: Any, scope: Scope) -> RollbackPoint | None:
    """The newest checkpoint of a scope as a RollbackPoint (None when the scope has none).
    某范围最新的检查点，转为 RollbackPoint（没有时为 None）。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        scope: Scope filter.
        scope：范围过滤。

    Output / 输出：
        RollbackPoint or None.
        RollbackPoint 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    checkpoints = store.list_checkpoints(scope)
    return RollbackPoint.from_checkpoint(checkpoints[-1]) if checkpoints else None


def _newest_ep(store: Any, scope: Scope, *, as_of: str) -> EP | None:
    """Newest valid EP of the scope at as_of (by time_anchor, ep_id).
    范围在 as_of 时最新的有效 EP（按 time_anchor、ep_id）。
    """
    eps = [ep for ep in store.list_eps(scope, run_id=scope.run_id) if ep.time_anchor <= as_of]
    return eps[-1] if eps else None


def build_customer_issue_state(
    hms: Any, *, filters: dict, issue: str | None = None, as_of: Any = None, budget_tokens: int | None = None
) -> CustomerIssueState:
    """CustomerIssueState: issue / status / owner / failed_paths / next_action derived from a deep compile.
    CustomerIssueState：由 deep 编译推导 issue / status / owner / failed_paths / next_action。

    Input / 输入：
        hms: The HMS facade (compile + store reads).
        hms：HMS 门面（compile + 存储读取）。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        issue: Issue text; None → the newest valid EP's goal `u` (ValueError when the scope has no EP either).
        issue：问题文本；None → 最新有效 EP 的目标 `u`（范围也没有 EP 时抛出 ValueError）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

        budget_tokens: Compile budget (None → config default).
        budget_tokens：编译预算（None → 配置默认值）。

    Output / 输出：
        CustomerIssueState (status ∈ resolved | blocked | open, witnessed by the newest EP or the context).
        CustomerIssueState（status ∈ resolved | blocked | open，由最新 EP 或上下文见证）。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    scope = scope_from_filters(filters)
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    newest = _newest_ep(store, scope, as_of=instant)
    goal = issue if issue is not None else (newest.slots.u if newest is not None else None)
    if goal is None:
        raise ValueError("CustomerIssueState needs an issue text or at least one EP in the scope")
    ir: ContextIR = hms.compile(
        goal, filters=filters, mode=CompileMode.deep.value, as_of=instant, budget_tokens=budget_tokens
    )
    sections = context_sections(ir)
    ctx_anchors = ir.anchor_ids()
    ep_anchors = list(newest.anchor_ids) if newest is not None else ctx_anchors
    if not ep_anchors:
        raise ValueError("CustomerIssueState has no witness: the scope holds no anchored state for this issue")
    issue_item = StateItem(text=goal, time=newest.time_anchor if newest else None, anchor_ids=ep_anchors)
    reward = str(newest.slots.r) if newest is not None else Reward.unknown.value
    status = StateItem(
        text=_STATUS_OF_REWARD.get(reward, "open"),
        time=newest.time_anchor if newest else None,
        anchor_ids=ep_anchors,
        extra={"reward": reward, "last_action": newest.slots.a if newest else None},
    )
    owner: StateItem | None = None
    for ev in reversed(store.last_events(scope, limit=20)):
        actor = ev.actor_id or scope.agent_id
        anchor = store.anchor_for_event(ev.event_id)
        if actor and anchor is not None:
            owner = StateItem(
                text=actor, time=ev.timestamp, anchor_ids=[anchor.anchor_id], extra={"event_id": ev.event_id}
            )
            break
    failed = [item_from_context_item(i) for i in sections["failed_paths"]]
    constraints = [item_from_context_item(i) for i in sections["constraints"]]
    next_action: StateItem | None = None
    if ir.next_action:
        witnesses = sorted({a for i in ir.episodes + ir.constraints for a in i.anchor_ids}) or ep_anchors
        next_action = StateItem(text=ir.next_action, anchor_ids=witnesses)
    return CustomerIssueState(
        scope=scope,
        as_of=instant,
        issue=issue_item,
        status=status,
        owner=owner,
        failed_paths=failed,
        constraints=constraints,
        next_action=next_action,
        context=ir,
    )


def build_incident_state(
    hms: Any, *, filters: dict, question: str | None = None, as_of: Any = None, hops: int = 2
) -> IncidentState:
    """IncidentState: event timeline, root cause (the why() chain), constraints and the rollback point.
    IncidentState：事件时间线、根因（why() 链）、约束与回滚点。

    Input / 输入：
        hms: The HMS facade (why + store reads).
        hms：HMS 门面（why + 存储读取）。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        question: Detective question; None → why() of the newest negative-reward EP (root_cause None without one).
        question：侦探式问题；None → 对最新负回报 EP 做 why()（没有时 root_cause 为 None）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

        hops: why() expansion levels.
        hops：why() 扩展层数。

    Output / 输出：
        IncidentState whose root_cause.chain == HMS.why(...)['chain'].
        root_cause.chain 等于 HMS.why(...)['chain'] 的 IncidentState。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    scope = scope_from_filters(filters)
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    atoms = live_atoms(store, scope, as_of=instant)
    timeline = [item_from_atom(a) for a in atoms if str(a.atom_type) == AtomType.event.value]
    root: DetectiveReport | None = None
    if question is not None:
        why = hms.why(question, hops=hops, filters=filters)
        root = DetectiveReport.from_why(why, question=question, scope=scope, as_of=instant)
    else:
        failed = [
            ep
            for ep in store.list_eps(scope, run_id=scope.run_id)
            if str(ep.slots.r) == Reward.negative.value and ep.atom_id and ep.time_anchor <= instant
        ]
        if failed:
            target = failed[-1]
            why = hms.why(target.atom_id, hops=hops)
            root = DetectiveReport.from_why(why, question=target.slots.gloss(), scope=scope, as_of=instant)
    return IncidentState(
        scope=scope,
        as_of=instant,
        timeline=timeline,
        root_cause=root,
        constraints=_constraint_items(atoms),
        rollback_point=latest_rollback_point(store, scope),
    )


def build_meeting_decision_state(hms: Any, *, filters: dict, as_of: Any = None) -> MeetingDecisionState:
    """MeetingDecisionState: decisions / owners / deadlines / superseded from the scope's atoms.
    MeetingDecisionState：由范围内原子推导 decisions / owners / deadlines / superseded。

    Input / 输入：
        hms: The HMS facade (only `hms.store` is used).
        hms：HMS 门面（仅使用 `hms.store`）。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

    Output / 输出：
        MeetingDecisionState; `superseded` pairs each revised atom with its successor text.
        MeetingDecisionState；`superseded` 将每个被修订原子与其后继文本配对。

    State effect / 状态影响：
        None.
        无。
    """
    store = hms.store
    scope = scope_from_filters(filters)
    instant = to_utc_iso(as_of) if as_of is not None else store.now()
    atoms = live_atoms(store, scope, as_of=instant)
    decisions = [item_from_atom(a) for a in atoms if is_decision(a)]
    owners = [
        item_from_atom(a, owner=a.subject)
        for a in atoms
        if str(a.atom_type) == AtomType.state.value and (a.subject or "").strip().lower() not in _GENERIC_SUBJECTS
    ]
    deadlines = [
        item_from_atom(a, due=a.time_anchor)
        for a in atoms
        if a.time_anchor is not None and str(a.atom_type) == AtomType.event.value and DEADLINE_RE.search(a.text)
    ]
    superseded: list[StateItem] = []
    for a in store.axis_select(scope, include_expired=True, order="time_anchor ASC, created_at ASC, atom_id ASC"):
        if a.archived_reason != "revised":
            continue
        head = store.resolve_head(a.memory_id) if a.memory_id else None
        successor = head.text if head is not None and head.atom_id != a.atom_id else None
        superseded.append(item_from_atom(a, superseded_by=successor, invalidated_at=a.valid_to))
    return MeetingDecisionState(
        scope=scope, as_of=instant, decisions=decisions, owners=owners, deadlines=deadlines, superseded=superseded
    )


def build_account_handoff_state(
    hms: Any, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
) -> AccountHandoffState:
    """AccountHandoffState = the Handoff Package + the customer constraints of the account scope.
    AccountHandoffState = 交接包 + 账户范围内的客户约束。

    Input / 输入：
        hms: The HMS facade (handoff + store reads).
        hms：HMS 门面（handoff + 存储读取）。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        goal: Handoff goal (None → newest valid EP's u).
        goal：交接目标（None → 最新有效 EP 的 u）。

        checkpoint: Take a checkpoint before packaging (its RollbackPoint is embedded).
        checkpoint：打包前建立检查点（嵌入其 RollbackPoint）。

        budget_tokens: Compile budget (None → config default).
        budget_tokens：编译预算（None → 配置默认值）。

    Output / 输出：
        AccountHandoffState.
        AccountHandoffState。

    State effect / 状态影响：
        Writes the checkpoint when `checkpoint` is True; otherwise read-only.
        `checkpoint` 为 True 时写入检查点；否则只读。
    """
    store = hms.store
    scope = scope_from_filters(filters)
    pkg = hms.handoff(filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens)
    rollback = None
    if pkg.checkpoint_id:
        cp = store.get_checkpoint(pkg.checkpoint_id)
        rollback = RollbackPoint.from_checkpoint(cp) if cp is not None else None
    atoms = live_atoms(store, scope, as_of=pkg.context.as_of)
    return AccountHandoffState(
        scope=scope,
        as_of=pkg.created_at,
        package=pkg,
        customer_constraints=_constraint_items(atoms),
        rollback_point=rollback,
    )


__all__ = [
    "DEADLINE_RE",
    "build_account_handoff_state",
    "build_customer_issue_state",
    "build_incident_state",
    "build_meeting_decision_state",
    "latest_rollback_point",
]
