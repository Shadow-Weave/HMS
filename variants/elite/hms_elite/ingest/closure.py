"""Structured closure builder: tool_call / file_diff / test_log events close into (u, b, a, y, r, κ) EPs with no LLM.
结构化闭包构建器：tool_call / file_diff / test_log 事件无需 LLM 即闭合为 (u, b, a, y, r, κ) EP。
"""

# Purpose (EN): Pair actions with outcomes per run, mint EPs, derived constraints and edge specs (use/temporal/causal/distill).
# 目的（中）：按 run 配对动作与结果，生成 EP、派生约束以及边规格（use/temporal/causal/distill）。

# Input (EN): Anchored structured events (sorted), background constraint atoms, the closure and temporal window knobs.
# 输入（中）：已落锚的结构化事件（有序）、背景约束原子、闭包与时间窗口参数。

# Output (EN): A ClosureBatch (open actions, EPs, derived observations, edge specs) for the pipeline to materialize.
# 输出（中）：供流水线物化的 ClosureBatch（未闭合动作、EP、派生观察、边规格）。

# State effect (EN): Read-only (store.last_open_action / store.last_ep lookups); nothing is written here.
# 状态影响（中）：只读（查询 store.last_open_action / store.last_ep）；此处不写入。

from __future__ import annotations

from dataclasses import dataclass, field

from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import ObjectType, Relation, Reward, Sentiment, Visibility, rank
from hms_elite.memory.storage import StateStore
from hms_elite.models import EP, Anchor, Kappa, MemoryAtom, Observation, Scope, Slots, SourceEvent
from hms_elite.timeutil import age_days, date_of
from hms_elite.utils.text import entity_key, extract_entities, ordered_union, topic_tags

CLOSURE_EXTRACTOR = "rule@closure-v1"
_ACTION_TYPES = frozenset({ObjectType.tool_call.value, ObjectType.file_diff.value})


def _first_patch_line(patch: str) -> str:
    """First non-empty line of a patch (empty string when none).
    补丁的第一个非空行（没有时为空串）。
    """
    for line in str(patch or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _args_summary(args: dict) -> str:
    """`k=v` pairs of a tool_call args dict, sorted by key, values cut at 40 chars.
    tool_call args 字典的 `k=v` 对，按键排序，值截断到 40 字符。
    """
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in sorted((args or {}).items()))


def _failure_names(payload: dict) -> list[str]:
    """Names of the failures listed in a test_log payload.
    test_log 载荷中列出的失败用例名。
    """
    return [str(f.get("name")) for f in (payload.get("failures") or []) if isinstance(f, dict) and f.get("name")]


def event_atom_text(ev: SourceEvent) -> str:
    """Deterministic one-line text of the event atom of a structured event.
    结构化事件的事件原子的确定性单行文本。

    Input / 输入：
        ev: A tool_call / file_diff / test_log / sensor_trace event (normalized payload).
        ev：tool_call / file_diff / test_log / sensor_trace 事件（载荷已归一化）。

    Output / 输出：
        'file_diff <path>: <summary>' | 'test_log <suite>: <status> (<p> passed, <f> failed)[: names]' | 'tool_call <tool>(<args>) → <status>' | 'sensor_trace <sensor>: <summary>'.
        'file_diff <路径>: <摘要>' | 'test_log <套件>: <状态> (<p> passed, <f> failed)[: 用例]' | 'tool_call <工具>(<参数>) → <状态>' | 'sensor_trace <传感器>: <摘要>'。

    State effect / 状态影响：
        None.
        无。
    """
    payload = ev.payload or {}
    object_type = str(ev.object_type)
    if object_type == ObjectType.file_diff.value:
        summary = payload.get("summary") or _first_patch_line(payload.get("patch", ""))
        return f"file_diff {payload.get('path', '')}: {summary}"
    if object_type == ObjectType.test_log.value:
        passed = payload.get("passed")
        failed = payload.get("failed")
        text = (
            f"test_log {payload.get('suite', '')}: {payload.get('status', 'passed')} "
            f"({'?' if passed is None else passed} passed, {'?' if failed is None else failed} failed)"
        )
        names = _failure_names(payload)
        if names:
            text += f": {', '.join(names[:3])}"
        return text
    if object_type == ObjectType.tool_call.value:
        return f"tool_call {payload.get('tool', '')}({_args_summary(payload.get('args') or {})}) → {payload.get('status') or 'unknown'}"
    if object_type == ObjectType.sensor_trace.value:
        summary = payload.get("summary") or f"{len(payload.get('samples') or [])} samples"
        return f"sensor_trace {payload.get('sensor', '')}: {summary}"
    text = str(payload.get("text") or "")
    return f"{object_type}: {text}" if text else object_type


def action_text(ev: SourceEvent) -> str:
    """The `a` slot of an action event.
    动作事件的 `a` 槽位文本。

    Input / 输入：
        ev: A file_diff or tool_call event.
        ev：file_diff 或 tool_call 事件。

    Output / 输出：
        'patch <path>: <summary or first patch line>' or '<tool>(<args>)'.
        'patch <路径>: <摘要或补丁首行>' 或 '<工具>(<参数>)'。

    State effect / 状态影响：
        None.
        无。
    """
    payload = ev.payload or {}
    if str(ev.object_type) == ObjectType.file_diff.value:
        summary = payload.get("summary") or _first_patch_line(payload.get("patch", ""))
        return f"patch {payload.get('path', '')}: {summary}"
    if str(ev.object_type) == ObjectType.tool_call.value:
        return f"{payload.get('tool', '')}({_args_summary(payload.get('args') or {})})"
    return event_atom_text(ev)


def outcome_text(ev: SourceEvent) -> tuple[str, str]:
    """The `y` slot and reward of an outcome event.
    结果事件的 `y` 槽位文本与回报。

    Input / 输入：
        ev: A test_log event, or a tool_call with status 'error'.
        ev：test_log 事件，或 status 为 'error' 的 tool_call。

    Output / 输出：
        (y, r): ('<suite> <status>[: names]', negative|positive) for logs; (result or 'error', negative) for failed tool calls.
        (y, r)：日志为 ('<套件> <状态>[: 用例]', negative|positive)；失败工具调用为 (result 或 'error', negative)。

    State effect / 状态影响：
        None.
        无。
    """
    payload = ev.payload or {}
    object_type = str(ev.object_type)
    if object_type == ObjectType.test_log.value:
        status = str(payload.get("status") or "passed")
        text = f"{payload.get('suite', '')} {status}"
        names = _failure_names(payload)
        if names:
            text += f": {', '.join(names)}"
        return text, Reward.negative.value if status == "failed" else Reward.positive.value
    if object_type == ObjectType.tool_call.value:
        if payload.get("status") == "error":
            return str(payload.get("result") or "error"), Reward.negative.value
        result = payload.get("result")
        if result:
            return str(result), Reward.neutral.value if payload.get("status") != "ok" else Reward.positive.value
        return "", Reward.unknown.value
    return event_atom_text(ev), Reward.unknown.value


def _min_visibility(*values: str) -> str:
    """The most restrictive (lowest-rank) visibility among the values.
    各值中最严格（秩最低）的可见性。
    """
    best = Visibility.public.value
    for value in values:
        if rank(str(value)) < rank(best):
            best = str(value)
    return best


def build_ep_from_closure(
    *,
    action: tuple[SourceEvent, Anchor],
    outcome: tuple[SourceEvent, Anchor],
    background: list[MemoryAtom],
    previous: EP | None,
    scope: Scope,
    extractor: EntityExtractorBase | None = None,
) -> EP:
    """Close one action → outcome pair into an EP witnessed by both anchors (outcome first).
    将一对 动作 → 结果 闭合为由两个锚点见证的 EP（结果锚点在前）。

    Input / 输入：
        action: (event, anchor) of the file_diff / tool_call.
        action：file_diff / tool_call 的 (事件, 锚点)。

        outcome: (event, anchor) of the test_log (or the action itself when it self-closes).
        outcome：test_log 的 (事件, 锚点)（自闭合时即动作本身）。

        background: Valid constraint atoms in scope, time order.
        background：范围内有效的约束原子，按时间排序。

        previous: The previous EP of the run (its `y` seeds `u` when the action has no intent).
        previous：该 run 的上一个 EP（动作无 intent 时以其 `y` 作为 `u`）。

        scope: Scope of the EP.
        scope：EP 的范围。

        extractor: Entity extractor for entity_mentions; None means utils.text.extract_entities (rule).
        extractor：entity_mentions 使用的实体抽取器；None 表示 utils.text.extract_entities（规则）。

    Output / 输出：
        EP with u/b/a/y/r/κ per §12.4, anchor_ids [outcome, action], time_anchor = outcome time, extracted_by 'rule@closure-v1'.
        按 §12.4 的 u/b/a/y/r/κ 的 EP，anchor_ids 为 [结果, 动作]，time_anchor 为结果时间，extracted_by 为 'rule@closure-v1'。

    State effect / 状态影响：
        None.
        无。
    """
    ev_a, anc_a = action
    ev_o, anc_o = outcome
    payload = ev_a.payload or {}
    intent = payload.get("intent")
    if intent:
        u = str(intent)
    elif previous is not None and previous.slots.y.strip():
        u = previous.slots.y
    elif str(ev_a.object_type) == ObjectType.file_diff.value:
        u = f"modify {payload.get('path', '')}"
    else:
        u = f"call {payload.get('tool', '')}"
    constraints = ordered_union([c.text for c in background], [str(c) for c in payload.get("constraints") or []])
    b = " ; ".join(constraints)
    a = action_text(ev_a)
    self_closing_open = (
        ev_o.event_id == ev_a.event_id
        and str(ev_a.object_type) == ObjectType.tool_call.value
        and payload.get("status") != "error"
    )
    if self_closing_open:
        y, r = "", Reward.unknown.value
    else:
        y, r = outcome_text(ev_o)
    anchor_ids = ordered_union([anc_o.anchor_id, anc_a.anchor_id])
    kappa = Kappa(
        constraints=constraints,
        visibility=_min_visibility(str(ev_a.visibility), str(ev_o.visibility)),
        evidence=list(anchor_ids),
    )
    slots = Slots(u=u, b=b, a=a, y=y, r=r, kappa=kappa)
    extract = extractor.extract if extractor is not None else extract_entities
    return EP(
        slots=slots,
        time_anchor=ev_o.timestamp,
        entity_mentions=[c.text for c in extract(f"{a} {y} {u}")],
        topic_tags=topic_tags(f"{u} {a} {y}"),
        confidence=0.8,
        visibility=kappa.visibility,
        extracted_by=CLOSURE_EXTRACTOR,
        scope=scope,
        anchor_ids=anchor_ids,
    )


def derived_constraint(ep: EP, *, scope: Scope) -> Observation:
    """Mint the 'Do not re-apply ...' constraint observation of a negative EP.
    为负向 EP 生成 'Do not re-apply ...' 约束观察。

    Input / 输入：
        ep: A negative EP.
        ep：负向 EP。

        scope: Scope of the observation.
        scope：观察的范围。

    Output / 输出：
        Observation(type constraint, subject 'project', predicate 'must_not', objects [a], text "Do not re-apply '<a>': <y>", anchors = ep anchors).
        Observation（类型 constraint、主体 'project'、谓词 'must_not'、宾语 [a]、文本 "Do not re-apply '<a>': <y>"、锚点同 EP）。

    State effect / 状态影响：
        None.
        无。
    """
    return Observation(
        type="constraint",
        subject="project",
        predicate="must_not",
        objects=[ep.slots.a],
        relation=Relation.constraint,
        text=f"Do not re-apply '{ep.slots.a}': {ep.slots.y}",
        date_observed=date_of(ep.time_anchor),
        valid_from=ep.time_anchor,
        confidence=0.85,
        visibility=ep.visibility,
        sentiment=Sentiment.negative,
        topic_tags=ordered_union(["constraint"], ep.topic_tags),
        entity_mentions=list(ep.entity_mentions),
        extracted_by=CLOSURE_EXTRACTOR,
        scope=scope,
        anchor_ids=list(ep.anchor_ids),
    )


@dataclass
class EdgeSpec:
    """A graph edge to be written once its endpoints exist: refs are ('ep', i) | ('derived', i) | ('atom', atom_id).
    端点存在后才写入的图边：引用形如 ('ep', i) | ('derived', i) | ('atom', atom_id)。
    """

    edge_type: str
    src: tuple[str, object]
    dst: tuple[str, object]
    witness_anchor_ids: list[str]
    confidence: float
    label: str = ""


@dataclass
class ClosureBatch:
    """Result of close_structured_events: what stays open, the EPs, the derived constraints and the edges to write.
    close_structured_events 的结果：仍未闭合的动作、EP、派生约束与待写入的边。
    """

    open_actions: dict[str | None, list[tuple[SourceEvent, Anchor]]] = field(default_factory=dict)
    eps: list[EP] = field(default_factory=list)
    derived: list[Observation] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)


def _entity_keys(mentions: list[str]) -> set[str]:
    """Entity keys of mention texts.
    提及文本的实体键集合。
    """
    return {entity_key(m) for m in mentions if m and m.strip()}


class _Closer:
    """Stateful helper running the §12.4 algorithm over one anchored batch.
    在一个已落锚批次上执行 §12.4 算法的有状态辅助类。
    """

    def __init__(
        self,
        store: StateStore,
        *,
        scope: Scope,
        background: list[MemoryAtom],
        max_gap_hours: float,
        temporal_window_hours: float,
        extractor: EntityExtractorBase | None = None,
    ) -> None:
        """Bind the store, scope, background constraints, window knobs and the entity extractor.
        绑定 store、范围、背景约束、窗口参数与实体抽取器。
        """
        self.store = store
        self.scope = scope
        self.background = list(background)
        self.max_gap_hours = float(max_gap_hours)
        self.temporal_window_hours = float(temporal_window_hours)
        self.extractor = extractor
        self.batch = ClosureBatch()
        self.last_built: dict[str | None, int] = {}
        self._store_previous: dict[str | None, EP | None] = {}

    def previous_for(self, run: str | None) -> tuple[EP | None, tuple[str, object] | None]:
        """The previous EP of a run (batch first, then the store) and the graph ref of its node.
        某 run 的上一个 EP（先看本批次，再看 store）及其节点引用。
        """
        if run in self.last_built:
            index = self.last_built[run]
            return self.batch.eps[index], ("ep", index)
        if run not in self._store_previous:
            self._store_previous[run] = self.store.last_ep(self.scope, run_id=run)
        previous = self._store_previous[run]
        if previous is None or not previous.atom_id:
            return previous, None
        return previous, ("atom", previous.atom_id)

    def close(self, action: tuple[SourceEvent, Anchor], outcome: tuple[SourceEvent, Anchor]) -> EP:
        """Build one EP, record its edges and derived constraint, and remember it as the run's last EP.
        构建一个 EP，记录其边与派生约束，并将其记为该 run 的最后一个 EP。
        """
        ev_a, anc_a = action
        ev_o, anc_o = outcome
        run = ev_a.scope.run_id
        previous, previous_ref = self.previous_for(run)
        ep = build_ep_from_closure(
            action=action,
            outcome=outcome,
            background=self.background,
            previous=previous,
            scope=self.scope,
            extractor=self.extractor,
        )
        index = len(self.batch.eps)
        self.batch.eps.append(ep)
        ref: tuple[str, object] = ("ep", index)
        for constraint in self.background:
            self.batch.edges.append(
                EdgeSpec(
                    "use", ref, ("atom", constraint.atom_id), [anc_a.anchor_id], 0.8, "closure: background constraint"
                )
            )
        if previous is not None and previous_ref is not None:
            if age_days(previous.time_anchor, ep.time_anchor) * 24.0 <= self.temporal_window_hours:
                self.batch.edges.append(
                    EdgeSpec("temporal", previous_ref, ref, [anc_o.anchor_id], 0.5, "closure: next")
                )
            u_from_previous = not (ev_a.payload or {}).get("intent") and bool(previous.slots.y.strip())
            overlap = bool(_entity_keys(previous.entity_mentions) & _entity_keys(ep.entity_mentions))
            if previous.slots.r == Reward.negative.value or overlap or u_from_previous:
                witnesses = ordered_union([anc_a.anchor_id], [previous.anchor_ids[0]])
                self.batch.edges.append(
                    EdgeSpec(
                        "causal",
                        previous_ref,
                        ref,
                        witnesses,
                        0.7,
                        f"closure: {previous.slots.y[:40]} → {ep.slots.u[:40]}",
                    )
                )
        if ep.slots.r == Reward.negative.value:
            derived = derived_constraint(ep, scope=self.scope)
            d_index = len(self.batch.derived)
            self.batch.derived.append(derived)
            self.batch.edges.append(
                EdgeSpec("distill", ref, ("derived", d_index), [anc_o.anchor_id], 0.9, "closure: failed path")
            )
        self.last_built[run] = index
        return ep

    def run(self, anchored: list[tuple[SourceEvent, Anchor, bool]]) -> ClosureBatch:
        """Walk the sorted events: push actions, close on outcomes, self-close error tool calls, finish open tool calls.
        遍历有序事件：压入动作、遇结果闭合、错误工具调用自闭合、收尾带约束的未闭合工具调用。
        """
        open_actions = self.batch.open_actions
        for ev, anc, _dedup in anchored:
            object_type = str(ev.object_type)
            run = ev.scope.run_id
            if object_type in _ACTION_TYPES:
                if object_type == ObjectType.tool_call.value and (ev.payload or {}).get("status") == "error":
                    self.close((ev, anc), (ev, anc))
                    continue
                open_actions.setdefault(run, []).append((ev, anc))
            elif object_type == ObjectType.test_log.value:
                action: tuple[SourceEvent, Anchor] | None = None
                if open_actions.get(run):
                    action = open_actions[run].pop()
                else:
                    action = self.store.last_open_action(
                        self.scope, run_id=run, before=ev.timestamp, max_gap_hours=self.max_gap_hours
                    )
                if action is None:
                    continue
                self.close(action, (ev, anc))
        for run in list(open_actions):
            remaining: list[tuple[SourceEvent, Anchor]] = []
            for ev, anc in open_actions[run]:
                payload = ev.payload or {}
                if str(ev.object_type) == ObjectType.tool_call.value and payload.get("constraints"):
                    self.close((ev, anc), (ev, anc))
                else:
                    remaining.append((ev, anc))
            open_actions[run] = remaining
        return self.batch


def close_structured_events(
    store: StateStore,
    *,
    anchored: list[tuple[SourceEvent, Anchor, bool]],
    scope: Scope,
    background: list[MemoryAtom],
    max_gap_hours: float,
    temporal_window_hours: float,
    extractor: EntityExtractorBase | None = None,
) -> ClosureBatch:
    """Run the §12.4 closure algorithm over a sorted batch of structured events (sensor traces skipped).
    对一批有序结构化事件执行 §12.4 闭包算法（跳过传感器轨迹）。

    Input / 输入：
        store: The state store (cross-batch lookups: last_open_action, last_ep).
        store：状态存储（跨批查询：last_open_action、last_ep）。

        anchored: `[(event, anchor, dedup)]` sorted by time.
        anchored：按时间排序的 `[(事件, 锚点, dedup)]`。

        scope: Scope of the batch.
        scope：批次的范围。

        background: Valid constraint atoms in scope, time order (become `b`, κ.constraints and `use` targets).
        background：范围内有效的约束原子，按时间排序（成为 `b`、κ.constraints 与 `use` 的目标）。

        max_gap_hours: config.ingest.closure_max_gap_hours (cross-batch action lookup).
        max_gap_hours：config.ingest.closure_max_gap_hours（跨批动作查找）。

        temporal_window_hours: config.graph.temporal_window_hours (temporal edge window).
        temporal_window_hours：config.graph.temporal_window_hours（temporal 边窗口）。

        extractor: Entity extractor for EP entity_mentions; None means the rule provider.
        extractor：EP entity_mentions 使用的实体抽取器；None 表示规则提供者。

    Output / 输出：
        ClosureBatch with open actions per run, EPs, derived constraint observations and edge specs.
        含各 run 未闭合动作、EP、派生约束观察与边规格的 ClosureBatch。

    State effect / 状态影响：
        None (read-only lookups).
        无（只读查询）。
    """
    closer = _Closer(
        store,
        scope=scope,
        background=background,
        max_gap_hours=max_gap_hours,
        temporal_window_hours=temporal_window_hours,
        extractor=extractor,
    )
    return closer.run(anchored)


__all__ = [
    "CLOSURE_EXTRACTOR",
    "ClosureBatch",
    "EdgeSpec",
    "action_text",
    "build_ep_from_closure",
    "close_structured_events",
    "derived_constraint",
    "event_atom_text",
    "outcome_text",
]
