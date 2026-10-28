"""Query planner: route selection plus time / entity / identifier hints that become hard axis gates.
查询规划器：路由选择，以及成为硬轴门控的时间 / 实体 / 标识符线索。
"""

# Purpose (EN): Turn a natural-language query into a QueryPlan (route, windows, entity hints, identifiers, gates).
# 目的（中）：把自然语言查询转换为 QueryPlan（路由、时间窗、实体线索、标识符、门控）。

# Input (EN): Query text, the `as_of` instant (base date for relative dates), an optional compile mode.
# 输入（中）：查询文本、`as_of` 时刻（相对日期的基准）、可选的编译模式。

# Output (EN): QueryPlan dataclasses (and their JSON-friendly as_dict form stored in AxisQuery.plan).
# 输出（中）：QueryPlan 数据类（及其存入 AxisQuery.plan 的 JSON 友好 as_dict 形式）。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hms_elite.enums import Route
from hms_elite.timeutil import date_of
from hms_elite.utils.text import (
    ENTITY_QUERY_RE,
    HANDOFF_RE,
    PREF_QUERY_RE,
    TEMPORAL_QUERY_RE,
    WHY_RE,
    EntityCandidate,
    content_tokens,
    extract_dates,
    extract_entities,
    identifier_tokens,
)

ALL_SCALES: list[str] = ["state", "event", "L1_micro", "L2_meso", "L3_macro", "entity", "relation"]
SCALE_DEFAULTS: dict[str, list[str] | None] = {
    Route.recall.value: list(ALL_SCALES),
    Route.preference.value: ["state"],
    Route.temporal.value: ["event", "L1_micro", "state", "L3_macro"],
    Route.entity.value: list(ALL_SCALES),
    Route.work_state.value: None,
    Route.detective.value: None,
    Route.handoff.value: None,
}
PREFERENCE_TYPE_GATE: tuple[str, list[str]] = ("obs_type", ["preference_event", "habit"])


@dataclass
class QueryPlan:
    """The planner output consumed by execute_axis_query, search and compile.
    供 execute_axis_query、search 与 compile 使用的规划器输出。
    """

    route: str
    time_windows: list[tuple[str, str, str]] = field(default_factory=list)
    entity_hints: list[EntityCandidate] = field(default_factory=list)
    identifier_tokens: list[str] = field(default_factory=list)
    content_tokens: list[str] = field(default_factory=list)
    hard_time_gate: tuple[str, str] | None = None
    hard_entity_gate: list[str] | None = None
    hard_type_gate: tuple[str, list[str]] | None = None
    scale_default: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict of the plan (tuples → lists, EntityCandidate → dict) for AxisQuery.plan.
        规划的 JSON 友好字典（元组 → 列表，EntityCandidate → 字典），用于 AxisQuery.plan。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with keys route, time_windows, entity_hints, identifier_tokens, content_tokens, hard_time_gate,
            hard_entity_gate, hard_type_gate, scale_default.
            含键 route、time_windows、entity_hints、identifier_tokens、content_tokens、hard_time_gate、
            hard_entity_gate、hard_type_gate、scale_default 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "route": self.route,
            "time_windows": [list(window) for window in self.time_windows],
            "entity_hints": [dict(hint._asdict()) for hint in self.entity_hints],
            "identifier_tokens": list(self.identifier_tokens),
            "content_tokens": list(self.content_tokens),
            "hard_time_gate": list(self.hard_time_gate) if self.hard_time_gate else None,
            "hard_entity_gate": list(self.hard_entity_gate) if self.hard_entity_gate is not None else None,
            "hard_type_gate": [self.hard_type_gate[0], list(self.hard_type_gate[1])] if self.hard_type_gate else None,
            "scale_default": list(self.scale_default) if self.scale_default is not None else None,
        }


def plan_query(query: str, *, as_of: str, mode: str | None = None, extractor: Any | None = None) -> QueryPlan:
    """Plan a query: route precedence detective > work_state > temporal > entity > preference > recall, plus gates.
    规划查询：路由优先级 detective > work_state > temporal > entity > preference > recall，并附带门控。

    Input / 输入：
        query: Natural-language query (EN/ZH).
        query：自然语言查询（中英）。

        as_of: Reference instant; its date is the base for relative dates and year-less months.
        as_of：参考时刻；其日期是相对日期与无年份月份的基准。

        mode: Compile mode; 'deep' forces work_state when the query has no stronger cue.
        mode：编译模式；'deep' 在查询无更强线索时强制 work_state。

        extractor: The facade's EntityExtractorBase (HMS.entity_extractor) for the entity hints, so the read side sees the same zh_rules / names as the write side (CONTRACT §9.6); None → utils.text.extract_entities (byte-identical to the default rule extractor).
        extractor：门面的 EntityExtractorBase（HMS.entity_extractor），用于实体线索，使读侧与写侧看到同样的 zh_rules / names（CONTRACT §9.6）；None → utils.text.extract_entities（与默认规则抽取器逐字节一致）。

    Output / 输出：
        QueryPlan with route, time windows, entity hints, identifier/content tokens, hard gates and scale default.
        含路由、时间窗、实体线索、标识符/内容词元、硬门控与默认尺度的 QueryPlan。

    State effect / 状态影响：
        None.
        无。
    """
    text = query or ""
    base_date = date_of(as_of)
    windows = extract_dates(text, base_date=base_date)
    hints = list(extractor.extract(text)) if extractor is not None else extract_entities(text)
    identifiers = identifier_tokens(text)
    tokens = content_tokens(text)
    if WHY_RE.search(text):
        route = Route.detective.value
    elif HANDOFF_RE.search(text):
        route = Route.work_state.value
    elif windows or TEMPORAL_QUERY_RE.search(text):
        route = Route.temporal.value
    elif ENTITY_QUERY_RE.search(text) and hints:
        route = Route.entity.value
    elif PREF_QUERY_RE.search(text):
        route = Route.preference.value
    else:
        route = Route.recall.value
    if mode == "deep" and route == Route.recall.value:
        route = Route.work_state.value
    first = windows[0] if windows else None
    return QueryPlan(
        route=route,
        time_windows=[(kind, gte, lte) for kind, gte, lte in windows],
        entity_hints=list(hints),
        identifier_tokens=list(identifiers),
        content_tokens=list(tokens),
        hard_time_gate=(first[1], first[2]) if first else None,
        hard_entity_gate=[hint.key for hint in hints] if route == Route.entity.value else None,
        hard_type_gate=(PREFERENCE_TYPE_GATE[0], list(PREFERENCE_TYPE_GATE[1]))
        if route == Route.preference.value
        else None,
        scale_default=list(SCALE_DEFAULTS[route]) if SCALE_DEFAULTS.get(route) is not None else None,
    )


__all__ = ["ALL_SCALES", "PREFERENCE_TYPE_GATE", "SCALE_DEFAULTS", "QueryPlan", "plan_query"]
