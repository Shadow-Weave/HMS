"""Deterministic renderers of HMS §2: Work State Card, Handoff Package, Detective Report, Personal/Edge/Group/Enterprise.
HMS §2 的确定性渲染器：Work State Card、Handoff Package、Detective Report、Personal/Edge/Group 卡与四张企业卡。
"""

# Purpose (EN): Turn state models into byte-stable text (no LLM, no store, no clock): the same model always renders the
# same string.
# 目的（中）：把状态模型变成字节稳定的文本（无 LLM、无存储、无时钟）：同一模型总是渲染出同一字符串。

# Input (EN): State models from hms_elite.state.kinds, a StatePackage, or the raw why() dict.
# 输入（中）：hms_elite.state.kinds 的状态模型、StatePackage 或原始 why() 字典。

# Output (EN): '\n'-joined card text without a trailing newline.
# 输出（中）：以 '\n' 连接、无尾随换行的卡片文本。

# State effect (EN): None; pure formatting.
# 状态影响（中）：无；纯格式化。

from __future__ import annotations

from typing import Any

from hms_elite.enums import Reward
from hms_elite.models import ContextItem, StatePackage
from hms_elite.state.kinds import (
    AccountHandoffState,
    BaseState,
    CustomerIssueState,
    DetectiveReport,
    EdgeState,
    GroupState,
    HandoffPackageState,
    IncidentState,
    MeetingDecisionState,
    PersonalState,
    RollbackPoint,
    StateItem,
    StateKind,
    WorkState,
    context_sections,
)

_NONE = "- (none)"


def _scope_line(state: BaseState) -> str:
    """'user_id=<u> agent_id=<a> run_id=<r>' restricted to the set keys.
    仅含已设置键的 'user_id=<u> agent_id=<a> run_id=<r>'。
    """
    return " ".join(f"{k}={v}" for k, v in state.scope.as_filters().items())


def _witness(ids: list[str]) -> str:
    """'[anc_1, anc_2]' in the given order.
    按给定顺序的 '[anc_1, anc_2]'。
    """
    return "[" + ", ".join(ids) + "]"


def _line(item: StateItem, *, with_time: bool = False, suffix: str = "") -> str:
    """'- <text>[ @<time>][ <suffix>] [anchors]'.
    '- <文本>[ @<时间>][ <后缀>] [锚点]'。
    """
    parts = [f"- {item.text}"]
    if with_time and item.time:
        parts.append(f"@{item.time}")
    if suffix:
        parts.append(suffix)
    parts.append(_witness(item.anchor_ids))
    return " ".join(parts)


def _section(lines: list[str], heading: str, body: list[str]) -> None:
    """Append '## heading' and its body ('- (none)' when empty).
    追加 '## heading' 与其内容（为空时 '- (none)'）。
    """
    lines.append(heading)
    lines.extend(body or [_NONE])


def _episode_line(item: ContextItem, *, failed: bool) -> str:
    """'- {a} → {y or ?} [{r}]' for valid facts, '- {a} → {y}' for failed paths; text-only without slots.
    有效事实 '- {a} → {y or ?} [{r}]'，失败路径 '- {a} → {y}'；无槽位时仅文本。
    """
    if "a" not in item.explain:
        return f"- {item.text}"
    a = str(item.explain.get("a") or "")
    y = str(item.explain.get("y") or "") or "?"
    return f"- {a} → {y}" if failed else f"- {a} → {y} [{item.explain.get('r') or Reward.unknown.value}]"


def _evidence_lines(state: BaseState) -> list[str]:
    """'- <anchor_id>' for the sorted witness union of a state.
    状态见证并集（排序）的 '- <anchor_id>' 行。
    """
    return [f"- {a}" for a in state.witness_ids()]


def render_work_state_card(state: WorkState) -> str:
    """The Work State Card of a WorkState (== ContextIR.to_text()).
    WorkState 的 Work State Card（== ContextIR.to_text()）。

    Input / 输入：
        state: The WorkState.
        state：WorkState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    return state.context.to_text()


def render_rollback_point(point: RollbackPoint) -> str:
    """The Rollback Point card.
    Rollback Point 卡片。

    Input / 输入：
        point: The RollbackPoint.
        point：RollbackPoint。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Rollback Point", f"Scope: {_scope_line(point)} | As of: {point.as_of}"]
    _section(
        lines,
        "## Checkpoint",
        [
            f"- id: {point.checkpoint_id}",
            f"- label: {point.label or '(none)'}",
            f"- created_at: {point.created_at}",
            f"- history_seq: {point.history_seq}",
        ],
    )
    _section(
        lines,
        "## Snapshot",
        [
            f"- hash: {point.snapshot_hash}",
            f"- atoms_live: {point.atoms_live}",
            f"- edges_valid: {point.edges_valid}",
        ],
    )
    _section(lines, "## Evidence", [f"- {a}" for a in point.anchor_ids])
    return "\n".join(lines)


def render_handoff_package(pkg: StatePackage | HandoffPackageState) -> str:
    """The Handoff Package text in the HMS §3.8 layout plus Rollback Point and Checkpoint id.
    HMS §3.8 版式的交接包文本，外加 Rollback Point 与 Checkpoint id。

    Input / 输入：
        pkg: A StatePackage or a HandoffPackageState (whose rollback point is rendered when present).
        pkg：StatePackage 或 HandoffPackageState（含回滚点时一并渲染）。

    Output / 输出：
        Text: Current Goal / Valid Facts / Invalidated Assumptions / Failed Paths / Constraints / Next Action /
        Evidence / Rollback Point / Checkpoint id.
        文本：Current Goal / Valid Facts / Invalidated Assumptions / Failed Paths / Constraints / Next Action /
        Evidence / Rollback Point / Checkpoint id。

    State effect / 状态影响：
        None.
        无。
    """
    rollback: RollbackPoint | None = None
    package = pkg
    if isinstance(pkg, HandoffPackageState):
        rollback = pkg.rollback_point
        package = pkg.package
    ir = package.context
    sections = context_sections(ir)
    scope = " ".join(f"{k}={v}" for k, v in package.scope.as_filters().items())
    lines = [
        "# Handoff Package",
        f"Package: {package.package_id} | Version: {package.version} | Scope: {scope} | Created: {package.created_at}",
        "Current Goal:",
        package.goal or ir.goal or ir.query,
    ]
    facts = [f"- {i.text}" for i in ir.facts]
    facts.extend(_episode_line(i, failed=False) for i in sections["valid_facts"][len(ir.facts) :])
    _section(lines, "Valid Facts:", facts)
    invalidated: list[str] = []
    for i in sections["invalidated"]:
        reason = str(i.explain.get("reason") or "")
        if reason == "revised":
            invalidated.append(f"- {i.text} -> superseded by: {i.explain.get('superseded_by_text') or ''}")
        else:
            invalidated.append(f"- {i.text} ({reason})")
    _section(lines, "Invalidated Assumptions:", invalidated)
    _section(lines, "Failed Paths:", [_episode_line(i, failed=True) for i in sections["failed_paths"]])
    _section(lines, "Constraints:", [f"- {i.text}" for i in sections["constraints"]])
    lines.append("Next Action:")
    lines.append(ir.next_action or "none pending")
    anchors = sorted(package.anchors, key=lambda a: (a.timestamp, a.anchor_id))
    _section(lines, "Evidence:", [f"- {a.evidence_line()} [{a.anchor_id}]" for a in anchors])
    lines.append("Rollback Point:")
    if rollback is not None:
        lines.append(
            f"{rollback.checkpoint_id} | label: {rollback.label or '(none)'} | created_at: {rollback.created_at}"
        )
    else:
        lines.append(package.checkpoint_id or "(none)")
    lines.append("Checkpoint id:")
    lines.append(package.checkpoint_id or "(none)")
    return "\n".join(lines)


def render_detective_report(why_result: dict | DetectiveReport, *, question: str | None = None) -> str:
    """The Detective Report: question → conclusion → evidence chain → anchored evidence → unresolved conflicts.
    Detective Report：问题 → 结论 → 证据链 → 带锚点的证据 → 未解冲突。

    Input / 输入：
        why_result: The HMS.why() dict, or an already built DetectiveReport.
        why_result：HMS.why() 字典，或已构建的 DetectiveReport。

        question: The question text (dict input only; defaults to the target text).
        question：问题文本（仅字典输入需要；默认取目标文本）。

    Output / 输出：
        Report text.
        报告文本。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(why_result, DetectiveReport):
        report = why_result
    else:
        target = dict(why_result.get("target") or {})
        first_anchor = (why_result.get("anchors") or [{}])[0]
        scope = first_anchor.get("scope") or {}
        report = DetectiveReport.from_why(
            why_result,
            question=question or str(target.get("text") or ""),
            scope=scope,
            as_of=str(first_anchor.get("ingested_at") or first_anchor.get("timestamp") or "1970-01-01T00:00:00Z"),
        )
    lines = ["# Detective Report", "Question:", report.question, "Conclusion:", report.conclusion.text]
    lines.append("Evidence Chain:")
    lines.extend(f"{n}. {line}" for n, line in enumerate(report.chain, start=1))
    if not report.chain:
        lines.append(_NONE)
    _section(lines, "Evidence:", [f"- {e.text} [{', '.join(e.anchor_ids)}]" for e in report.evidence])
    _section(lines, "Unresolved Conflicts:", [_line(c) for c in report.open_conflicts])
    return "\n".join(lines)


def render_personal_card(state: PersonalState) -> str:
    """The Personal Card (or Shared Card when state.kind is 'shared_card').
    Personal Card（state.kind 为 'shared_card' 时是 Shared Card）。

    Input / 输入：
        state: The PersonalState.
        state：PersonalState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    title = "# Shared Card" if str(state.kind) == StateKind.shared_card.value else "# Personal Card"
    lines = [
        title,
        f"Subject: {state.subject} | Scope: {_scope_line(state)} | As of: {state.as_of} | "
        f"Audience: {state.audience or 'all'}",
    ]
    _section(
        lines,
        "## Preferences (valid)",
        [_line(i, suffix=f"(since {i.valid_from or i.time or '?'})") for i in state.preferences_valid],
    )
    invalid: list[str] = []
    for i in state.preferences_invalid:
        successor = i.extra.get("superseded_by")
        tail = f"(from {i.valid_from or i.time or '?'} until {i.extra.get('invalidated_at') or i.valid_to or '?'}"
        tail += f"; superseded by: {successor})" if successor else ")"
        invalid.append(_line(i, suffix=tail))
    _section(lines, "## Preferences (invalidated)", invalid)
    _section(lines, "## Habits", [_line(i) for i in state.habits])
    _section(lines, "## Constraints", [_line(i) for i in state.constraints])
    _section(lines, "## Profile Facts", [_line(i) for i in state.profile_facts])
    _section(lines, "## Timeline", [_line(i, with_time=True) for i in state.timeline])
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_edge_card(state: EdgeState) -> str:
    """The Edge Card between two endpoints.
    两个端点之间的 Edge Card。

    Input / 输入：
        state: The EdgeState.
        state：EdgeState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = [
        "# Edge Card",
        f"Endpoints: {state.a} ⟷ {state.b} ({state.endpoint_kind}) | Scope: {_scope_line(state)} | As of: {state.as_of}",
    ]
    _section(lines, "## Shared Facts", [_line(i) for i in state.shared_facts])
    _section(lines, "## Interactions", [_line(i, with_time=True) for i in state.interactions])
    lines.append("## Last Contact")
    lines.append(state.last_contact or "(none)")
    _section(
        lines,
        "## Relation Edges",
        [f"- {e.src_text} -{e.edge_type}-> {e.dst_text} {_witness(e.anchor_ids)}" for e in state.relation_edges],
    )
    _section(lines, "## Open Conflicts", [_line(i) for i in state.open_conflicts])
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_group_card(state: GroupState) -> str:
    """The Group Card of a member set.
    一组成员的 Group Card。

    Input / 输入：
        state: The GroupState.
        state：GroupState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Group Card", f"Members: {', '.join(state.members)} | As of: {state.as_of}"]
    _section(
        lines,
        "## Shared Constraints",
        [_line(i, suffix=f"(held by: {', '.join(i.extra.get('members') or [])})") for i in state.shared_constraints],
    )
    _section(lines, "## Decisions", [_line(i, suffix=f"({i.extra.get('member')})") for i in state.decisions])
    _section(
        lines,
        "## Conflicts Between Members",
        [
            f"- {c.members[0]}: {c.left.text} ⟂ {c.members[1]}: {c.right.text} ({c.reason}) {_witness(c.anchor_ids)}"
            for c in state.conflicts_between_members
        ],
    )
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_customer_issue_state(state: CustomerIssueState) -> str:
    """The Customer Issue State card.
    Customer Issue State 卡片。

    Input / 输入：
        state: The CustomerIssueState.
        state：CustomerIssueState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Customer Issue State", f"Scope: {_scope_line(state)} | As of: {state.as_of}"]
    _section(lines, "## Issue", [_line(state.issue)])
    _section(lines, "## Status", [_line(state.status)])
    _section(lines, "## Owner", [_line(state.owner)] if state.owner else [])
    _section(lines, "## Failed Paths", [_line(i) for i in state.failed_paths])
    _section(lines, "## Constraints", [_line(i) for i in state.constraints])
    _section(lines, "## Next Action", [_line(state.next_action)] if state.next_action else [])
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_incident_state(state: IncidentState) -> str:
    """The Incident State card (root cause = the why() chain).
    Incident State 卡片（根因 = why() 链）。

    Input / 输入：
        state: The IncidentState.
        state：IncidentState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Incident State", f"Scope: {_scope_line(state)} | As of: {state.as_of}"]
    _section(lines, "## Timeline", [_line(i, with_time=True) for i in state.timeline])
    root: list[str] = []
    if state.root_cause is not None:
        root.append(f"Question: {state.root_cause.question}")
        root.extend(f"{n}. {line}" for n, line in enumerate(state.root_cause.chain, start=1))
    _section(lines, "## Root Cause", root)
    _section(lines, "## Constraints", [_line(i) for i in state.constraints])
    lines.append("## Rollback Point")
    if state.rollback_point is not None:
        rp = state.rollback_point
        lines.append(f"{rp.checkpoint_id} | label: {rp.label or '(none)'} | created_at: {rp.created_at}")
    else:
        lines.append("(none)")
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_meeting_decision_state(state: MeetingDecisionState) -> str:
    """The Meeting Decision State card.
    Meeting Decision State 卡片。

    Input / 输入：
        state: The MeetingDecisionState.
        state：MeetingDecisionState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Meeting Decision State", f"Scope: {_scope_line(state)} | As of: {state.as_of}"]
    _section(lines, "## Decisions", [_line(i, with_time=True) for i in state.decisions])
    _section(lines, "## Owners", [_line(i, suffix=f"(owner: {i.extra.get('owner')})") for i in state.owners])
    _section(lines, "## Deadlines", [_line(i, suffix=f"(due: {i.extra.get('due')})") for i in state.deadlines])
    _section(
        lines,
        "## Superseded",
        [_line(i, suffix=f"-> superseded by: {i.extra.get('superseded_by') or '?'}") for i in state.superseded],
    )
    _section(lines, "## Evidence", _evidence_lines(state))
    return "\n".join(lines)


def render_account_handoff_state(state: AccountHandoffState) -> str:
    """The Account Handoff State card: customer constraints + the embedded Handoff Package.
    Account Handoff State 卡片：客户约束 + 内嵌的交接包。

    Input / 输入：
        state: The AccountHandoffState.
        state：AccountHandoffState。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines = ["# Account Handoff State", f"Scope: {_scope_line(state)} | As of: {state.as_of}"]
    _section(lines, "## Customer Constraints", [_line(i) for i in state.customer_constraints])
    lines.append("## Handoff Package")
    package = HandoffPackageState.from_package(state.package, state.rollback_point)
    lines.extend(render_handoff_package(package).split("\n"))
    lines.append("## Rollback Point")
    lines.append(state.rollback_point.checkpoint_id if state.rollback_point else "(none)")
    lines.append("## Checkpoint id")
    lines.append(state.package.checkpoint_id or "(none)")
    return "\n".join(lines)


_RENDERERS: dict[str, Any] = {
    StateKind.work_state.value: render_work_state_card,
    StateKind.handoff_package.value: render_handoff_package,
    StateKind.detective_report.value: render_detective_report,
    StateKind.rollback_point.value: render_rollback_point,
    StateKind.personal_state.value: render_personal_card,
    StateKind.shared_card.value: render_personal_card,
    StateKind.edge_state.value: render_edge_card,
    StateKind.group_state.value: render_group_card,
    StateKind.customer_issue_state.value: render_customer_issue_state,
    StateKind.incident_state.value: render_incident_state,
    StateKind.meeting_decision_state.value: render_meeting_decision_state,
    StateKind.account_handoff_state.value: render_account_handoff_state,
}


def render_state(state: BaseState) -> str:
    """Dispatch to the renderer of state.kind.
    按 state.kind 分派到对应渲染器。

    Input / 输入：
        state: Any state model.
        state：任意状态模型。

    Output / 输出：
        Card text.
        卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    return _RENDERERS[str(state.kind)](state)


__all__ = [
    "render_account_handoff_state",
    "render_customer_issue_state",
    "render_detective_report",
    "render_edge_card",
    "render_group_card",
    "render_handoff_package",
    "render_incident_state",
    "render_meeting_decision_state",
    "render_personal_card",
    "render_rollback_point",
    "render_state",
    "render_work_state_card",
]
