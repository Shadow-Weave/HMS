"""StateAPI: the HMS §11 names as one object over the facade — every call returns (state model, rendered text).
StateAPI：把 HMS §11 的名字作为一个对象暴露在门面之上 —— 每次调用返回（状态模型，渲染文本）。
"""

# Purpose (EN): work_state / handoff_package / detective_report / rollback_point / personal_state / shared_card /
# edge_state / group_state / customer_issue_state / incident_state / meeting_decision_state / account_handoff_state.
# 目的（中）：work_state / handoff_package / detective_report / rollback_point / personal_state / shared_card /
# edge_state / group_state / customer_issue_state / incident_state / meeting_decision_state / account_handoff_state。

# Input (EN): An HMS facade plus the arguments of the underlying builders.
# 输入（中）：HMS 门面以及底层构建器的参数。

# Output (EN): (state, text) tuples; the text is the deterministic rendering of the state.
# 输出（中）：(state, text) 元组；text 是该状态的确定性渲染。

# State effect (EN): Read-only except handoff_package / account_handoff_state (checkpoint=True) and
# rollback_point(create=True), which take a checkpoint through the facade.
# 状态影响（中）：只读；例外是 handoff_package / account_handoff_state（checkpoint=True）与 rollback_point(create=True)，
# 它们经门面建立检查点。

from __future__ import annotations

from typing import Any

from hms_elite.enums import Visibility
from hms_elite.models import Scope
from hms_elite.state.edge import build_edge_state
from hms_elite.state.enterprise import (
    build_account_handoff_state,
    build_customer_issue_state,
    build_incident_state,
    build_meeting_decision_state,
    latest_rollback_point,
)
from hms_elite.state.group import build_group_state
from hms_elite.state.kinds import (
    AccountHandoffState,
    CustomerIssueState,
    DetectiveReport,
    EdgeState,
    GroupState,
    HandoffPackageState,
    IncidentState,
    MeetingDecisionState,
    PersonalState,
    RollbackPoint,
    WorkState,
    scope_from_filters,
)
from hms_elite.state.personal import build_personal_state, build_shared_card
from hms_elite.state.render import (
    render_account_handoff_state,
    render_customer_issue_state,
    render_detective_report,
    render_edge_card,
    render_group_card,
    render_handoff_package,
    render_incident_state,
    render_meeting_decision_state,
    render_personal_card,
    render_rollback_point,
    render_work_state_card,
)
from hms_elite.timeutil import to_utc_iso


def build_work_state(
    hms: Any, query: str, *, filters: dict, mode: str | None = None, as_of: Any = None, budget_tokens: int | None = None
) -> WorkState:
    """WorkState = HMS.compile wrapped (the Work State Card is its rendering).
    WorkState = 包装后的 HMS.compile（Work State Card 是其渲染）。

    Input / 输入：
        hms: The HMS facade.
        hms：HMS 门面。

        query: The goal / question text.
        query：目标 / 问题文本。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        mode: Compile mode (None → config default).
        mode：编译模式（None → 配置默认值）。

        as_of: Reference instant (None → store.now()).
        as_of：参考时刻（None → store.now()）。

        budget_tokens: Token budget (None → config default).
        budget_tokens：token 预算（None → 配置默认值）。

    Output / 输出：
        WorkState.
        WorkState。

    State effect / 状态影响：
        None.
        无。
    """
    ir = hms.compile(query, filters=filters, mode=mode, as_of=as_of, budget_tokens=budget_tokens)
    return WorkState.from_context(ir)


def build_handoff_package(
    hms: Any, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
) -> HandoffPackageState:
    """HandoffPackageState = HMS.handoff plus the RollbackPoint of its checkpoint.
    HandoffPackageState = HMS.handoff 加上其检查点的 RollbackPoint。

    Input / 输入：
        hms: The HMS facade.
        hms：HMS 门面。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        goal: Handoff goal (None → newest valid EP's u).
        goal：交接目标（None → 最新有效 EP 的 u）。

        checkpoint: Take a checkpoint first.
        checkpoint：先建立检查点。

        budget_tokens: Token budget (None → config default).
        budget_tokens：token 预算（None → 配置默认值）。

    Output / 输出：
        HandoffPackageState.
        HandoffPackageState。

    State effect / 状态影响：
        Writes the checkpoint when `checkpoint` is True.
        `checkpoint` 为 True 时写入检查点。
    """
    pkg = hms.handoff(filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens)
    rollback = None
    if pkg.checkpoint_id:
        cp = hms.store.get_checkpoint(pkg.checkpoint_id)
        rollback = RollbackPoint.from_checkpoint(cp) if cp is not None else None
    return HandoffPackageState.from_package(pkg, rollback)


def build_detective_report(
    hms: Any, target: str, *, filters: dict | None = None, hops: int = 2, as_of: Any = None
) -> DetectiveReport:
    """DetectiveReport = HMS.why (an object id, or a question when `filters` are given).
    DetectiveReport = HMS.why（对象 id，或给定 `filters` 时的一个问题）。

    Input / 输入：
        hms: The HMS facade.
        hms：HMS 门面。

        target: Object id or question text.
        target：对象 id 或问题文本。

        filters: Scope filters enabling question mode.
        filters：启用问题模式的范围过滤。

        hops: why() expansion levels.
        hops：why() 扩展层数。

        as_of: Reference instant recorded on the report (None → store.now()).
        as_of：记录在报告上的参考时刻（None → store.now()）。

    Output / 输出：
        DetectiveReport.
        DetectiveReport。

    State effect / 状态影响：
        None.
        无。
    """
    why = hms.why(target, hops=hops, filters=filters)
    instant = to_utc_iso(as_of) if as_of is not None else hms.store.now()
    if filters:
        scope = scope_from_filters(filters)
    else:
        first = (why.get("anchors") or [{}])[0]
        scope = Scope(**(first.get("scope") or {}))
    return DetectiveReport.from_why(why, question=target, scope=scope, as_of=instant)


def build_rollback_point(
    hms: Any, *, filters: dict, label: str | None = None, create: bool = False
) -> RollbackPoint | None:
    """RollbackPoint = a new checkpoint (create=True) or the newest existing one of the scope.
    RollbackPoint = 新检查点（create=True）或该范围现有的最新检查点。

    Input / 输入：
        hms: The HMS facade.
        hms：HMS 门面。

        filters: Mem0 filter dict with >= 1 scope key.
        filters：至少含一个范围键的 Mem0 过滤字典。

        label: Checkpoint label when creating.
        label：创建时的检查点标签。

        create: Take a checkpoint now instead of reading the newest one.
        create：现在建立检查点，而不是读取最新的那个。

    Output / 输出：
        RollbackPoint, or None when the scope has no checkpoint and create is False.
        RollbackPoint；范围没有检查点且 create 为 False 时为 None。

    State effect / 状态影响：
        Writes a checkpoint when create is True.
        create 为 True 时写入检查点。
    """
    scope = scope_from_filters(filters)
    if create:
        cp_id = hms.checkpoint(filters=filters, label=label)
        cp = hms.store.get_checkpoint(cp_id)
        return RollbackPoint.from_checkpoint(cp) if cp is not None else None
    return latest_rollback_point(hms.store, scope)


class StateAPI:
    """The §11 state surface of one HMS facade: each method returns (state model, rendered text).
    一个 HMS 门面的 §11 状态表面：每个方法返回（状态模型，渲染文本）。
    """

    def __init__(self, hms: Any) -> None:
        """Bind to a facade.
        绑定到一个门面。

        Input / 输入：
            hms: The HMS facade (compile / search / why / handoff / checkpoint / store / graph).
            hms：HMS 门面（compile / search / why / handoff / checkpoint / store / graph）。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.hms = hms

    # ------------------------------------------------------------------ §11.1 Agent WorkState
    # ------------------------------------------------------------------ §11.1 Agent 工作态
    def work_state(
        self, query: str, *, filters: dict, mode: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[WorkState, str]:
        """Work State Card.
        工作状态卡。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            mode: Compile mode (None → config default).
            mode：编译模式（None → 配置默认值）。

            as_of: Reference instant.
            as_of：参考时刻。

            budget_tokens: Token budget.
            budget_tokens：token 预算。

        Output / 输出：
            (WorkState, card text).
            (WorkState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_work_state(self.hms, query, filters=filters, mode=mode, as_of=as_of, budget_tokens=budget_tokens)
        return state, render_work_state_card(state)

    def handoff_package(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[HandoffPackageState, str]:
        """Handoff Package.
        交接包。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Handoff goal (None → newest valid EP's u).
            goal：交接目标（None → 最新有效 EP 的 u）。

            checkpoint: Take a checkpoint first.
            checkpoint：先建立检查点。

            budget_tokens: Token budget.
            budget_tokens：token 预算。

        Output / 输出：
            (HandoffPackageState, package text).
            (HandoffPackageState, 交接包文本)。

        State effect / 状态影响：
            Writes the checkpoint when `checkpoint` is True.
            `checkpoint` 为 True 时写入检查点。
        """
        state = build_handoff_package(
            self.hms, filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens
        )
        return state, render_handoff_package(state)

    def detective_report(
        self, target: str, *, filters: dict | None = None, hops: int = 2, as_of: Any = None
    ) -> tuple[DetectiveReport, str]:
        """Detective Report.
        侦探报告。

        Input / 输入：
            target: Object id or question text.
            target：对象 id 或问题文本。

            filters: Scope filters enabling question mode.
            filters：启用问题模式的范围过滤。

            hops: why() expansion levels.
            hops：why() 扩展层数。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (DetectiveReport, report text).
            (DetectiveReport, 报告文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_detective_report(self.hms, target, filters=filters, hops=hops, as_of=as_of)
        return state, render_detective_report(state)

    def rollback_point(
        self, *, filters: dict, label: str | None = None, create: bool = False
    ) -> tuple[RollbackPoint | None, str]:
        """Rollback Point.
        回滚点。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            label: Checkpoint label when creating.
            label：创建时的检查点标签。

            create: Take a checkpoint now.
            create：现在建立检查点。

        Output / 输出：
            (RollbackPoint or None, card text or '(none)').
            (RollbackPoint 或 None, 卡片文本或 '(none)')。

        State effect / 状态影响：
            Writes a checkpoint when create is True.
            create 为 True 时写入检查点。
        """
        state = build_rollback_point(self.hms, filters=filters, label=label, create=create)
        return state, render_rollback_point(state) if state is not None else "(none)"

    # ------------------------------------------------------------------ §11.2 Personal API / RESONANCE
    # ------------------------------------------------------------------ §11.2 个人 API / 共鸣
    def personal_state(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str | None = None,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """PersonalState + Personal Card.
        PersonalState 与个人卡。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            subject: 'user' | 'assistant' | 'project' | entity text.
            subject：'user' | 'assistant' | 'project' | 实体文本。

            audience: Visibility floor (None → no gate).
            audience：可见性下限（None → 不门控）。

            as_of: Reference instant.
            as_of：参考时刻。

            timeline_limit: Newest N timeline entries.
            timeline_limit：时间线最新的 N 条。

        Output / 输出：
            (PersonalState, card text).
            (PersonalState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_personal_state(
            self.hms, filters=filters, subject=subject, audience=audience, as_of=as_of, timeline_limit=timeline_limit
        )
        return state, render_personal_card(state)

    def shared_card(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str = Visibility.team,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """Shared Card (PersonalState gated to the audience, at least 'team').
        Shared Card（按受众门控的 PersonalState，至少 'team'）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            subject: Subject selector.
            subject：主体选择器。

            audience: Visibility floor.
            audience：可见性下限。

            as_of: Reference instant.
            as_of：参考时刻。

            timeline_limit: Newest N timeline entries.
            timeline_limit：时间线最新的 N 条。

        Output / 输出：
            (PersonalState with kind 'shared_card', card text).
            (kind 为 'shared_card' 的 PersonalState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_shared_card(
            self.hms, filters=filters, subject=subject, audience=audience, as_of=as_of, timeline_limit=timeline_limit
        )
        return state, render_personal_card(state)

    def edge_state(
        self, a: str | dict, b: str | dict, *, filters: dict | None = None, as_of: Any = None
    ) -> tuple[EdgeState, str]:
        """EdgeState + Edge Card.
        EdgeState 与关系卡。

        Input / 输入：
            a: Entity text or scope dict.
            a：实体文本或范围字典。

            b: Entity text or scope dict (same form as `a`).
            b：实体文本或范围字典（与 `a` 同形）。

            filters: Scope filters (entity mode).
            filters：范围过滤（实体模式）。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (EdgeState, card text).
            (EdgeState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_edge_state(self.hms, a, b, filters=filters, as_of=as_of)
        return state, render_edge_card(state)

    def group_state(self, *, members: list[dict], as_of: Any = None) -> tuple[GroupState, str]:
        """GroupState + Group Card.
        GroupState 与群体卡。

        Input / 输入：
            members: Scope dicts, one per member.
            members：范围字典，每个成员一个。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (GroupState, card text).
            (GroupState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_group_state(self.hms, members=members, as_of=as_of)
        return state, render_group_card(state)

    # ------------------------------------------------------------------ §11.3 Enterprise State
    # ------------------------------------------------------------------ §11.3 企业状态
    def customer_issue_state(
        self, *, filters: dict, issue: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[CustomerIssueState, str]:
        """CustomerIssueState + card.
        CustomerIssueState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            issue: Issue text (None → newest EP goal).
            issue：问题文本（None → 最新 EP 的目标）。

            as_of: Reference instant.
            as_of：参考时刻。

            budget_tokens: Compile budget.
            budget_tokens：编译预算。

        Output / 输出：
            (CustomerIssueState, card text).
            (CustomerIssueState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_customer_issue_state(
            self.hms, filters=filters, issue=issue, as_of=as_of, budget_tokens=budget_tokens
        )
        return state, render_customer_issue_state(state)

    def incident_state(
        self, *, filters: dict, question: str | None = None, as_of: Any = None, hops: int = 2
    ) -> tuple[IncidentState, str]:
        """IncidentState + card.
        IncidentState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            question: Detective question (None → newest negative EP).
            question：侦探式问题（None → 最新负回报 EP）。

            as_of: Reference instant.
            as_of：参考时刻。

            hops: why() expansion levels.
            hops：why() 扩展层数。

        Output / 输出：
            (IncidentState, card text).
            (IncidentState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_incident_state(self.hms, filters=filters, question=question, as_of=as_of, hops=hops)
        return state, render_incident_state(state)

    def meeting_decision_state(self, *, filters: dict, as_of: Any = None) -> tuple[MeetingDecisionState, str]:
        """MeetingDecisionState + card.
        MeetingDecisionState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (MeetingDecisionState, card text).
            (MeetingDecisionState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        state = build_meeting_decision_state(self.hms, filters=filters, as_of=as_of)
        return state, render_meeting_decision_state(state)

    def account_handoff_state(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[AccountHandoffState, str]:
        """AccountHandoffState + card.
        AccountHandoffState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Handoff goal.
            goal：交接目标。

            checkpoint: Take a checkpoint first.
            checkpoint：先建立检查点。

            budget_tokens: Compile budget.
            budget_tokens：编译预算。

        Output / 输出：
            (AccountHandoffState, card text).
            (AccountHandoffState, 卡片文本)。

        State effect / 状态影响：
            Writes the checkpoint when `checkpoint` is True.
            `checkpoint` 为 True 时写入检查点。
        """
        state = build_account_handoff_state(
            self.hms, filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens
        )
        return state, render_account_handoff_state(state)


__all__ = [
    "StateAPI",
    "build_detective_report",
    "build_handoff_package",
    "build_rollback_point",
    "build_work_state",
]
