"""Requirement extraction and greedy set-cover selection of ContextItems (HMS §5.1 `coverage.py`).
需求抽取与 ContextItem 的贪心集合覆盖选择（HMS §5.1 `coverage.py`）。
"""

# Purpose (EN): Turn a query (plus its planner output) into the requirements the compiled state MUST cover, then pick
#               the cheapest item set that covers them: greedy set cover, per-slot floors, evidence rows always kept.
# 目的（中）：把查询（及其规划器输出）转成编译态**必须**覆盖的需求，再挑选覆盖它们的最省项集：
#             贪心集合覆盖、槽位保底、证据行永远保留。

# Input (EN): Query text, an optional QueryPlan (or its dict), candidate ContextItems, a token budget.
# 输入（中）：查询文本、可选的 QueryPlan（或其字典）、候选 ContextItem、token 预算。

# Output (EN): Requirement records and a CoverageSelection (selected items, covered map, uncovered list, overlap stats).
# 输出（中）：Requirement 记录与 CoverageSelection（选中项、覆盖映射、未覆盖列表、重叠统计）。

# State effect (EN): None (pure functions over in-memory objects; no store, index or graph access).
# 状态影响（中）：无（对内存对象的纯函数；不访问存储、索引或图）。

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from hms_elite.enums import Slot
from hms_elite.llms.rule import normalize
from hms_elite.models import ContextItem
from hms_elite.mvmi.planner import QueryPlan, plan_query
from hms_elite.timeutil import now_iso, to_utc_iso
from hms_elite.utils.text import CONSTRAINT_RE, CONSTRAINT_ZH_RE, MUST_RE, content_tokens, is_identifier, sentences

__all__ = [
    "COVERAGE_RATIO",
    "KINDS",
    "CoverageSelection",
    "Requirement",
    "coverage_matrix",
    "covers",
    "extract_requirements",
    "select_for_coverage",
]

KINDS: tuple[str, ...] = ("goal", "constraint", "entity", "time", "identifier")
# Share of a goal / constraint requirement's content tokens an item must repeat to cover it.
# 一个项要覆盖 goal / constraint 需求，必须重复该需求内容词元的比例。
COVERAGE_RATIO = 0.5
WEIGHTS: dict[str, float] = {"goal": 1.0, "constraint": 0.9, "identifier": 1.0, "entity": 0.8, "time": 0.6}
_NON_AUDIT_SLOTS = ("facts", "episodes", "preferences", "constraints", "skills")
_EVIDENCE_PREFIX = "anc:"

_GOAL_VERBS = (
    r"fix|implement|add|remove|delete|refactor|debug|investigate|migrate|write|update|deploy|test|check|resolve|build"
    r"|make|create|improve|optimi[sz]e|ship|patch|revert|roll back|rollback|handle|support|enable|disable|set up"
    r"|setup|configure|install|upgrade|clean up|document|review|merge|release|verify|validate|find|figure out"
    r"|diagnose|repair|restore|continue|finish|complete|debug|rewrite|replace|rename|move|split|extract"
)
_GOAL_RE = re.compile(
    r"^\s*(?:please\s+|let'?s\s+|help me\s+|we need to\s+|i need to\s+|i want to\s+|we should\s+|i should\s+"
    r"|how (?:do|can|should) (?:i|we)\s+|how to\s+)?(?P<verb>" + _GOAL_VERBS + r")\b\s*(?P<obj>.+?)"
    r"(?=[,;?!]|\.(?!\w)|\s+\b(?:because|since|so that|but|and then|without|while)\b|$)",
    re.I,
)
_GOAL_ZH_RE = re.compile(
    r"(?P<verb>修复|实现|添加|新增|删除|重构|排查|调查|迁移|部署|测试|检查|解决|优化|升级|安装|配置|继续|完成|修改|处理)"
    r"(?P<obj>[^，。；！？]+)"
)
# Negative / restrictive cue words that mark an explicit constraint clause (EN + ZH).
# 标记显式约束子句的否定 / 限制线索词（中英）。
_NEGATION_RE = re.compile(
    r"\b(?:not|never|without|avoid|forbidden|only|no longer|don't|do not|must not|mustn't|cannot|can't)\b"
    r"|不能|不要|不得|禁止|不允许|绝不|必须|只能|避免",
    re.I,
)
_LEAD_WORDS_RE = re.compile(
    r"^\s*(?:why|how|what|when|where|who|which|explain|tell me|for what reason)\b[\s,]*(?:do|does|did|is|are|was|were)?"
    r"\s*",
    re.I,
)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:，；：]\s*|\s+(?:and|but|because|since|so that|while)\s+", re.I)


@dataclass(frozen=True)
class Requirement:
    """One thing the compiled state must cover: a goal phrase, a constraint, an entity, a time window or an identifier.
    编译态必须覆盖的一件事：目标短语、约束、实体、时间窗或标识符。
    """

    req_id: str
    kind: str
    text: str
    weight: float
    tokens: tuple[str, ...] = ()
    window: tuple[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict of the requirement.
        需求的 JSON 友好字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with req_id, kind, text, weight, tokens (list), window (list | None).
            含 req_id、kind、text、weight、tokens（列表）、window（列表 | None）的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "req_id": self.req_id,
            "kind": self.kind,
            "text": self.text,
            "weight": self.weight,
            "tokens": list(self.tokens),
            "window": list(self.window) if self.window else None,
        }


@dataclass
class CoverageSelection:
    """Result of select_for_coverage: the kept items, what covers what, what stayed uncovered, and overlap statistics.
    select_for_coverage 的结果：保留项、覆盖关系、未覆盖需求与重叠统计。
    """

    selected: list[ContextItem] = field(default_factory=list)
    covered: dict[str, list[str]] = field(default_factory=dict)
    uncovered: list[Requirement] = field(default_factory=list)
    overlap_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_ids(self) -> list[str]:
        """Item ids of the selection in selection order.
        按选择顺序的选中项 id。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of item ids.
            项 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [item.item_id for item in self.selected]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (item ids instead of items).
        JSON 友好字典（用项 id 代替项）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with selected (ids), covered, uncovered (requirement dicts), overlap_stats.
            含 selected（id）、covered、uncovered（需求字典）、overlap_stats 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "selected": self.selected_ids,
            "covered": {k: list(v) for k, v in self.covered.items()},
            "uncovered": [req.as_dict() for req in self.uncovered],
            "overlap_stats": dict(self.overlap_stats),
        }


def _plan_of(query: str, plan: Any, as_of: str | None, extractor: Any | None = None) -> dict[str, Any]:
    """Normalize a QueryPlan / plan dict / None into the planner dict (None → plan_query at `as_of` or now, entity hints from `extractor` when given).
    把 QueryPlan / 规划字典 / None 归一为规划器字典（None → 在 `as_of` 或当前时刻运行 plan_query，给了 `extractor` 时实体线索来自它）。
    """
    if plan is None:
        instant = to_utc_iso(as_of) if as_of else now_iso()
        return plan_query(query, as_of=instant, extractor=extractor).as_dict()
    if isinstance(plan, QueryPlan):
        return plan.as_dict()
    if isinstance(plan, dict):
        return plan
    raise TypeError("plan must be a QueryPlan, a plan dict or None")


def _req_tokens(text: str) -> tuple[str, ...]:
    """Ordered, de-duplicated content tokens of a requirement text.
    需求文本的有序去重内容词元。
    """
    seen: list[str] = []
    for tok in content_tokens(text):
        if tok not in seen:
            seen.append(tok)
    return tuple(seen)


def _goal_phrase(query: str) -> str | None:
    """The action-verb phrase of a query (EN or ZH), or None.
    查询中的动作动词短语（中或英），无则为 None。
    """
    match = _GOAL_RE.match(query)
    if match:
        return f"{match.group('verb')} {match.group('obj').strip()}".strip()
    match = _GOAL_ZH_RE.search(query)
    if match:
        return f"{match.group('verb')}{match.group('obj').strip()}"
    return None


def _constraint_clauses(query: str) -> list[str]:
    """Clauses of the query that carry an explicit constraint cue (must / not / never / 不能 / 必须 ...).
    查询中带显式约束线索（must / not / never / 不能 / 必须……）的子句。
    """
    out: list[str] = []
    for sentence in sentences(query):
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            match = CONSTRAINT_RE.match(clause)
            if match:
                out.append(match.group("rest").strip())
                continue
            match = CONSTRAINT_ZH_RE.search(clause)
            if match:
                out.append(match.group(0).strip())
                continue
            must = MUST_RE.search(clause)
            if must:
                out.append(clause[must.end() :].strip() or clause)
                continue
            if _NEGATION_RE.search(clause):
                out.append(_LEAD_WORDS_RE.sub("", clause).strip() or clause)
    return [c.rstrip(".?!。？！") for c in out if c.strip(".?!。？！ ")]


def extract_requirements(query: str, plan: Any = None, *, as_of: str | None = None, extractor: Any | None = None) -> list[Requirement]:
    """Extract what a compiled state must cover: goal phrase, constraints, identifiers, entities, time windows.
    抽取编译态必须覆盖的需求：目标短语、约束、标识符、实体、时间窗。

    Input / 输入：
        query: The query / goal text (EN/ZH).
        query：查询 / 目标文本（中英）。

        plan: QueryPlan, its as_dict() form (e.g. ContextIR.retrieval['plan']) or None (→ plan_query at `as_of`).
        plan：QueryPlan、其 as_dict() 形式（如 ContextIR.retrieval['plan']）或 None（→ 在 `as_of` 运行 plan_query）。

        as_of: Reference instant for relative dates when `plan` is None (None → now).
        as_of：`plan` 为 None 时相对日期的参考时刻（None → 当前）。

        extractor: The facade's entity extractor for the entity hints when `plan` is None (None → utils.text.extract_entities).
        extractor：`plan` 为 None 时实体线索所用的门面实体抽取器（None → utils.text.extract_entities）。

    Output / 输出：
        Requirements in the order goal, constraint, identifier, entity, time; req_id = '<kind>:<key>', unique.
        按 goal、constraint、identifier、entity、time 顺序的需求；req_id = '<kind>:<key>'，唯一。

    State effect / 状态影响：
        None.
        无。
    """
    text = str(query or "")
    data = _plan_of(text, plan, as_of, extractor)
    out: list[Requirement] = []
    seen: set[str] = set()

    def push(kind: str, key: str, req_text: str, *, tokens: tuple[str, ...], window: tuple[str, str] | None = None):
        """Append one requirement unless its id was already emitted.
        追加一个需求，除非其 id 已经出现过。
        """
        req_id = f"{kind}:{key}"
        if req_id in seen or not key:
            return
        seen.add(req_id)
        out.append(
            Requirement(req_id=req_id, kind=kind, text=req_text, weight=WEIGHTS[kind], tokens=tokens, window=window)
        )

    goal = _goal_phrase(text)
    if goal:
        push("goal", normalize(goal), goal, tokens=_req_tokens(goal))
    for clause in _constraint_clauses(text):
        tokens = _req_tokens(clause)
        if tokens:
            push("constraint", normalize(clause), clause, tokens=tokens)
    for ident in data.get("identifier_tokens") or []:
        key = str(ident).lower()
        push("identifier", key, str(ident), tokens=(key,))
    for hint in data.get("entity_hints") or []:
        if isinstance(hint, dict):
            kind, key, hint_text = str(hint.get("kind")), str(hint.get("key") or ""), str(hint.get("text") or "")
        else:
            kind, key, hint_text = (
                str(getattr(hint, "kind", "")),
                str(getattr(hint, "key", "")),
                str(getattr(hint, "text", "")),
            )
        if kind in ("FILE", "IDENTIFIER") or f"identifier:{key}" in seen or is_identifier(key):
            continue
        push("entity", key, hint_text or key, tokens=(key,))
    for window in data.get("time_windows") or []:
        if not isinstance(window, (list, tuple)) or len(window) != 3:
            continue
        kind, gte, lte = (str(w) for w in window)
        push("time", f"{gte}..{lte}", f"{kind} {gte}..{lte}", tokens=(), window=(gte, lte))
    if not out:
        tokens = _req_tokens(text)
        if tokens:
            push("goal", normalize(text), text.strip(), tokens=tokens)
    return out


def _item_tokens(item: ContextItem) -> set[str]:
    """Content tokens of an item's text.
    项文本的内容词元集合。
    """
    return set(content_tokens(item.text))


def _item_instant(item: ContextItem) -> str | None:
    """The instant an item is anchored at (time_anchor, else valid_from).
    项所锚定的时刻（time_anchor，否则 valid_from）。
    """
    return item.time_anchor or item.valid_from


def covers(item: ContextItem, requirement: Requirement) -> bool:
    """True when an item satisfies a requirement (identifier / entity containment, time window, token-share for text).
    项满足需求时为真（标识符 / 实体包含、时间窗、文本类按词元占比）。

    Input / 输入：
        item: A ContextItem.
        item：ContextItem。

        requirement: A Requirement from extract_requirements.
        requirement：来自 extract_requirements 的 Requirement。

    Output / 输出：
        identifier: key (or its basename) in the lowercased text; entity: key in the normalized text; time: the item's
        instant inside the window; goal / constraint: ≥ COVERAGE_RATIO of the requirement tokens present in the item.
        identifier：键（或其基名）出现在小写文本中；entity：键出现在归一化文本中；time：项时刻落在窗内；
        goal / constraint：需求词元中至少 COVERAGE_RATIO 的比例出现在项中。

    State effect / 状态影响：
        None.
        无。
    """
    kind = requirement.kind
    if kind == "identifier":
        key = requirement.tokens[0] if requirement.tokens else requirement.text.lower()
        low = item.text.lower()
        return key in low or key.rsplit("/", 1)[-1] in low
    if kind == "entity":
        key = requirement.tokens[0] if requirement.tokens else normalize(requirement.text)
        return bool(key) and key in normalize(item.text)
    if kind == "time":
        instant = _item_instant(item)
        if not instant or not requirement.window:
            return False
        gte, lte = requirement.window
        return gte <= to_utc_iso(instant) <= lte
    needed = requirement.tokens
    if not needed:
        return False
    have = _item_tokens(item)
    low = item.text.lower()
    # An identifier token counts when it (or its basename) appears anywhere, so 'retry.ts' hits 'src/billing/retry.ts'.
    # 标识符词元只要（或其基名）在任何位置出现即算命中，故 'retry.ts' 命中 'src/billing/retry.ts'。
    hits = sum(
        1 for tok in needed if tok in have or (is_identifier(tok) and (tok in low or tok.rsplit("/", 1)[-1] in low))
    )
    return hits >= max(1, math.ceil(len(needed) * COVERAGE_RATIO))


def coverage_matrix(items: Iterable[ContextItem], requirements: Iterable[Requirement]) -> dict[str, list[str]]:
    """item_id → req_ids it covers (requirement order kept).
    item_id → 其覆盖的 req_id（保持需求顺序）。

    Input / 输入：
        items: Candidate items.
        items：候选项。

        requirements: Requirements.
        requirements：需求。

    Output / 输出：
        Dict for every item (possibly an empty list).
        每个项一个条目（可能为空列表）。

    State effect / 状态影响：
        None.
        无。
    """
    reqs = list(requirements)
    return {item.item_id: [req.req_id for req in reqs if covers(item, req)] for item in items}


def _is_evidence(item: ContextItem) -> bool:
    """True for the 'anc:<anchor_id>' evidence rows of the audit slot.
    audit 槽位中 'anc:<anchor_id>' 证据行为真。
    """
    return str(item.slot) == Slot.audit.value and item.item_id.startswith(_EVIDENCE_PREFIX)


def select_for_coverage(
    items: list[ContextItem],
    requirements: list[Requirement],
    *,
    budget_tokens: int,
    min_per_slot: int = 0,
) -> CoverageSelection:
    """Greedy set cover: each round take the item covering the most uncovered requirements at the best weight per token.
    贪心集合覆盖：每轮取覆盖最多未覆盖需求、且单位 token 权重收益最高的项。

    Input / 输入：
        items: Candidate ContextItems (any slots; 'anc:' evidence rows are kept unconditionally and never compete).
        items：候选 ContextItem（任意槽位；'anc:' 证据行无条件保留且不参与竞争）。

        requirements: Requirements to cover (extract_requirements output).
        requirements：待覆盖的需求（extract_requirements 的输出）。

        budget_tokens: Token budget for the competing items (evidence rows may push the total over it).
        budget_tokens：竞争项的 token 预算（证据行可能使总量超出）。

        min_per_slot: Floor per non-audit slot, filled after coverage by score desc then item_id, budget permitting.
        min_per_slot：每个非 audit 槽位的保底数量，覆盖之后按分数降序再按 item_id 填充，以预算为限。

    Output / 输出：
        CoverageSelection; ties break on item_id, so equal inputs always give equal output.
        CoverageSelection；并列按 item_id 决定，相同输入恒得相同输出。

    State effect / 状态影响：
        None.
        无。
    """
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    reqs = list(requirements)
    by_req = {req.req_id: req for req in reqs}
    evidence = [item for item in items if _is_evidence(item)]
    pool = sorted((item for item in items if not _is_evidence(item)), key=lambda i: i.item_id)
    matrix = coverage_matrix(pool, reqs)
    selected: list[ContextItem] = list(evidence)
    chosen: set[str] = set()
    # `spent` is what the competing items consume; evidence rows sit on top of the budget, never inside it.
    # `spent` 是竞争项消耗的量；证据行叠加在预算之上，绝不占用预算。
    spent = 0
    uncovered: dict[str, Requirement] = {req.req_id: req for req in reqs}
    skipped_over_budget: set[str] = set()
    rounds = 0
    while uncovered:
        best: tuple[tuple[int, float, str], ContextItem, list[str]] | None = None
        for item in pool:
            if item.item_id in chosen:
                continue
            new = [r for r in matrix[item.item_id] if r in uncovered]
            if not new:
                continue
            if spent + item.tokens > budget_tokens:
                skipped_over_budget.add(item.item_id)
                continue
            gain = sum(by_req[r].weight for r in new)
            key = (-len(new), -gain / max(1, item.tokens), item.item_id)
            if best is None or key < best[0]:
                best = (key, item, new)
        if best is None:
            break
        _key, item, new = best
        selected.append(item)
        chosen.add(item.item_id)
        spent += item.tokens
        rounds += 1
        for req_id in new:
            uncovered.pop(req_id, None)
    floors_added = 0
    if min_per_slot > 0:
        for slot in _NON_AUDIT_SLOTS:
            have = sum(1 for item in selected if str(item.slot) == slot)
            ranked = sorted(
                (item for item in pool if str(item.slot) == slot and item.item_id not in chosen),
                key=lambda i: (-i.score, i.item_id),
            )
            for item in ranked:
                if have >= min_per_slot:
                    break
                if spent + item.tokens > budget_tokens:
                    skipped_over_budget.add(item.item_id)
                    continue
                selected.append(item)
                chosen.add(item.item_id)
                spent += item.tokens
                have += 1
                floors_added += 1
    covered: dict[str, list[str]] = {}
    for item in selected:
        for req_id in matrix.get(item.item_id, []):
            covered.setdefault(req_id, []).append(item.item_id)
    covered = {req.req_id: covered[req.req_id] for req in reqs if req.req_id in covered}
    left = [req for req in reqs if req.req_id not in covered]
    n_req = len(reqs)
    used = spent + sum(item.tokens for item in evidence)
    stats: dict[str, Any] = {
        "requirements": n_req,
        "covered": len(covered),
        "uncovered": len(left),
        "coverage_ratio": 1.0 if n_req == 0 else round(len(covered) / n_req, 4),
        "selected": len(selected),
        "evidence": len(evidence),
        "floors_added": floors_added,
        "rounds": rounds,
        "used_tokens": used,
        "budget_tokens": budget_tokens,
        "spent_tokens": spent,
        "over_budget_evidence": max(0, used - budget_tokens),
        "skipped_over_budget": sorted(skipped_over_budget - chosen),
        "redundancy": 0.0 if not covered else round(sum(len(v) for v in covered.values()) / len(covered), 4),
    }
    return CoverageSelection(selected=selected, covered=covered, uncovered=left, overlap_stats=stats)
