"""Deterministic Work State Card renderer over a ContextIR.
基于 ContextIR 的确定性 Work State Card 渲染器。
"""

# Purpose (EN): Render the exact §15.3 card text (newline-joined, no trailing newline) from the six slots.
# 目的（中）：由六个槽位渲染精确的 §15.3 卡片文本（换行连接、无尾随换行）。

# Input (EN): A ContextIR.
# 输入（中）：一个 ContextIR。

# Output (EN): The card string.
# 输出（中）：卡片字符串。

# State effect (EN): None; pure formatting.
# 状态影响（中）：无；纯格式化。

from __future__ import annotations

from hms_elite.enums import Reward
from hms_elite.models import ContextIR, ContextItem

_NONE_LINE = "- (none)"
_ARCHIVED_PREFIX = "archived:"
_NON_NEGATIVE = frozenset({Reward.positive.value, Reward.neutral.value, Reward.unknown.value})


def _value(obj: object) -> str:
    """The string value of an enum member or plain string.
    枚举成员或普通字符串的字符串值。
    """
    return str(getattr(obj, "value", obj))


def _scope_line(ir: ContextIR) -> str:
    """'user_id=<u> agent_id=<a> run_id=<r>' restricted to the set keys, in this order.
    仅含已设置键、按此顺序的 'user_id=<u> agent_id=<a> run_id=<r>'。
    """
    parts = [f"{key}={value}" for key, value in ir.scope.as_filters().items()]
    return " ".join(parts)


def _episode_line(item: ContextItem, *, failed: bool) -> str:
    """'- {a} → {y or ?} [{r}]' for valid facts, '- {a} → {y}' for failed paths; text-only when slots are absent.
    有效事实为 '- {a} → {y or ?} [{r}]'，失败路径为 '- {a} → {y}'；无槽位时仅输出文本。
    """
    if "a" not in item.explain:
        return f"- {item.text}"
    a = str(item.explain.get("a") or "")
    y = str(item.explain.get("y") or "") or "?"
    if failed:
        return f"- {a} → {y}"
    return f"- {a} → {y} [{item.explain.get('r') or Reward.unknown.value}]"


def _section(lines: list[str], heading: str, body: list[str], *, optional: bool = False) -> None:
    """Append a '## heading' block; '- (none)' when empty unless the section is optional (then omitted).
    追加 '## heading' 块；为空时输出 '- (none)'，可选块为空则整体省略。
    """
    if optional and not body:
        return
    lines.append(heading)
    lines.extend(body or [_NONE_LINE])


def render_card(ir: ContextIR) -> str:
    """Render the Work State Card (§15.3) — byte-for-byte deterministic for a given ContextIR.
    渲染 Work State Card（§15.3）——对给定 ContextIR 逐字节确定。

    Input / 输入：
        ir: The compiled ContextIR.
        ir：已编译的 ContextIR。

    Output / 输出：
        The card text, '\\n'-joined without a trailing newline.
        以 '\\n' 连接、无尾随换行的卡片文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines: list[str] = ["# Work State Card"]
    lines.append(f"Scope: {_scope_line(ir)} | As of: {ir.as_of} | Mode: {_value(ir.mode)}")
    lines.append("## Current Goal")
    lines.append(ir.goal or ir.query)

    facts = [f"- {item.text}" for item in ir.facts]
    facts.extend(
        _episode_line(item, failed=False)
        for item in ir.episodes
        if str(item.explain.get("r") or Reward.unknown.value) in _NON_NEGATIVE
    )
    _section(lines, "## Valid Facts", facts)

    invalidated: list[str] = []
    for item in ir.audit:
        reason = str(item.explain.get("reason") or "")
        if reason == "revised":
            invalidated.append(f"- {item.text} -> superseded by: {item.explain.get('superseded_by_text') or ''}")
        elif reason.startswith(_ARCHIVED_PREFIX):
            invalidated.append(f"- {item.text} (archived: {reason[len(_ARCHIVED_PREFIX) :]})")
    _section(lines, "## Invalidated Assumptions", invalidated)

    failed = [
        _episode_line(item, failed=True)
        for item in ir.episodes
        if str(item.explain.get("r") or "") == Reward.negative.value
    ]
    _section(lines, "## Failed Paths", failed)
    _section(lines, "## Constraints", [f"- {item.text}" for item in ir.constraints])
    _section(lines, "## Preferences", [f"- {item.text}" for item in ir.preferences], optional=True)
    _section(lines, "## Skills", [f"- {item.text}" for item in ir.skills], optional=True)
    lines.append("## Next Action")
    lines.append(ir.next_action or "none pending")
    evidence = [f"- {item.text}" for item in ir.audit if item.explain.get("kind") == "evidence"]
    _section(lines, "## Evidence", evidence)
    # Only a ContextIR packed by selection='coverage' carries retrieval['coverage']; default cards end at Evidence.
    # 只有以 selection='coverage' 打包的 ContextIR 才带 retrieval['coverage']；默认卡片到 Evidence 为止。
    coverage = ir.retrieval.get("coverage")
    if isinstance(coverage, dict):
        requirements = list(coverage.get("requirements") or [])
        uncovered = [str(req_id) for req_id in coverage.get("uncovered") or []]
        body = [f"- requirements: {len(requirements)} | covered: {len(requirements) - len(uncovered)} | uncovered: {len(uncovered)}"]
        body.extend(f"- uncovered: {req_id}" for req_id in uncovered)
        _section(lines, "## Coverage", body)
    return "\n".join(lines)
