"""Closed vocabularies of HMS Elite as string enums; DDL CHECK lists are generated from them.
HMS Elite 的封闭词汇表（字符串枚举）；DDL 的 CHECK 列表由此生成。
"""

# Purpose (EN): Own every enumerated value used by models, schema, planner and compiler.
# 目的（中）：拥有模型、模式、规划器与编译器使用的全部枚举值。

# Input (EN): Nothing; constants only.
# 输入（中）：无；仅常量。

# Output (EN): StrEnum classes plus derived constant sets and orderings.
# 输出（中）：StrEnum 类以及派生的常量集合与排序表。

# State effect (EN): None (pure constants).
# 状态影响（中）：无（纯常量）。

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Base for every HMS enum: members are plain strings and hash like their value.
    所有 HMS 枚举的基类：成员即普通字符串，并按其值哈希。
    """

    def __str__(self) -> str:  # noqa: D105 - dunder
        return str.__str__(self)

    def __format__(self, spec: str) -> str:  # noqa: D105 - dunder
        return str.__format__(str(self), spec)

    def __hash__(self) -> int:  # noqa: D105 - dunder
        return str.__hash__(self)

    @classmethod
    def values(cls) -> list[str]:
        """Return every member value in declaration order.
        按声明顺序返回全部成员值。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of member values (plain strings).
            成员值（普通字符串）列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [member.value for member in cls]


class EdgeType(StrEnum):
    """Support Graph edge types.
    支持图的边类型。
    """

    temporal = "temporal"
    causal = "causal"
    revision = "revision"
    conflict = "conflict"
    use = "use"
    distill = "distill"


class FeedbackType(StrEnum):
    """The eleven StateFeedback types.
    十一种状态反馈类型。
    """

    state_wrong = "state_wrong"
    state_outdated = "state_outdated"
    state_missing = "state_missing"
    privacy_violation = "privacy_violation"
    next_action_wrong = "next_action_wrong"
    relationship_not_useful = "relationship_not_useful"
    connection_reason_wrong = "connection_reason_wrong"
    evidence_wrong = "evidence_wrong"
    render_not_like_me = "render_not_like_me"
    render_like_me = "render_like_me"
    object_important = "object_important"


class Scale(StrEnum):
    """Scale axis values (episode scales plus the state/event/entity/relation atoms).
    尺度轴取值（情节尺度以及 state/event/entity/relation 原子）。
    """

    L1_micro = "L1_micro"
    L2_meso = "L2_meso"
    L3_macro = "L3_macro"
    state = "state"
    event = "event"
    entity = "entity"
    relation = "relation"


class AtomType(StrEnum):
    """MemoryAtom kinds.
    记忆原子种类。
    """

    message = "message"
    state = "state"
    event = "event"
    entity = "entity"
    relation = "relation"
    episode = "episode"


class ObservationType(StrEnum):
    """Observation types produced by extraction.
    抽取产生的观察类型。
    """

    fact = "fact"
    preference_event = "preference_event"
    habit = "habit"
    constraint = "constraint"
    temporal_fact = "temporal_fact"


class Reward(StrEnum):
    """EP reward signal r.
    EP 的回报信号 r。
    """

    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    unknown = "unknown"


class Visibility(StrEnum):
    """Visibility lattice private < team < event < public.
    可见性格：private < team < event < public。
    """

    private = "private"
    team = "team"
    event = "event"
    public = "public"


class Role(StrEnum):
    """Message roles; 'extracted' is the synthetic role of derived atoms.
    消息角色；'extracted' 是派生原子的合成角色。
    """

    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"
    extracted = "extracted"


class Sentiment(StrEnum):
    """Sentiment axis values.
    情感轴取值。
    """

    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    mixed = "mixed"
    unknown = "unknown"


class Slot(StrEnum):
    """ContextIR slots.
    ContextIR 槽位。
    """

    facts = "facts"
    episodes = "episodes"
    preferences = "preferences"
    constraints = "constraints"
    skills = "skills"
    audit = "audit"


class ObjectType(StrEnum):
    """SourceEvent object types (five extractable + five internal).
    源事件对象类型（五种可抽取 + 五种内部）。
    """

    message = "message"
    tool_call = "tool_call"
    file_diff = "file_diff"
    test_log = "test_log"
    sensor_trace = "sensor_trace"
    feedback = "feedback"
    checkpoint = "checkpoint"
    rollback = "rollback"
    system = "system"
    scheduler = "scheduler"


class EventType(StrEnum):
    """SourceEvent event types.
    源事件的事件类型。
    """

    created = "created"
    updated = "updated"
    deleted = "deleted"
    feedback = "feedback"
    checkpoint = "checkpoint"
    rollback = "rollback"
    tool_result = "tool_result"
    scheduler = "scheduler"


class NodeKind(StrEnum):
    """Support Graph node kinds.
    支持图节点种类。
    """

    message = "message"
    observation = "observation"
    event = "event"
    entity = "entity"
    ep = "ep"
    interface_ep = "interface_ep"
    skill = "skill"


class SourceKind(StrEnum):
    """Where a MemoryAtom came from.
    记忆原子的来源种类。
    """

    message = "message"
    observation = "observation"
    event = "event"
    entity = "entity"
    ep = "ep"
    interface_ep = "interface_ep"
    skill = "skill"
    procedural = "procedural"
    feedback = "feedback"


class Decision(StrEnum):
    """Revision decisions.
    修订决策。
    """

    ADD = "ADD"
    REVISE = "REVISE"
    CONFLICT = "CONFLICT"
    NONE = "NONE"


class HistoryEvent(StrEnum):
    """History row events (Mem0's four plus HMS-only ones).
    历史行事件（Mem0 的四种加 HMS 特有事件）。
    """

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    ARCHIVE = "ARCHIVE"
    RESTORE = "RESTORE"
    FEEDBACK = "FEEDBACK"
    CONFLICT = "CONFLICT"
    ROLLBACK = "ROLLBACK"
    # PURGE: the retention escape hatch physically removed this version (ops/retention.py, §16.6).
    # PURGE：保留策略逃生舱物理删除了该版本（ops/retention.py，§16.6）。
    PURGE = "PURGE"


class Route(StrEnum):
    """Query planner routes.
    查询规划路由。
    """

    recall = "recall"
    preference = "preference"
    temporal = "temporal"
    entity = "entity"
    work_state = "work_state"
    detective = "detective"
    handoff = "handoff"


class CompileMode(StrEnum):
    """Compile modes: fast = Axis only; axis_graph = Axis + Graph; deep = Axis + Graph + Support.
    编译模式：fast 仅轴；axis_graph 轴 + 图；deep 轴 + 图 + 支持层。
    """

    auto = "auto"
    fast = "fast"
    axis_graph = "axis_graph"
    deep = "deep"


class Relation(StrEnum):
    """Observation relation kinds.
    观察关系种类。
    """

    single_fact = "single_fact"
    comparison = "comparison"
    habit = "habit"
    supersession = "supersession"
    constraint = "constraint"


class EntityType(StrEnum):
    """Entity candidate types in priority order.
    实体候选类型（按优先级）。
    """

    PROPER = "PROPER"
    IDENTIFIER = "IDENTIFIER"
    FILE = "FILE"
    QUOTED = "QUOTED"
    TOPIC = "TOPIC"


class MemoryType(StrEnum):
    """Mem0 memory types.
    Mem0 记忆类型。
    """

    SEMANTIC = "semantic_memory"
    EPISODIC = "episodic_memory"
    PROCEDURAL = "procedural_memory"


class TransitionKind(StrEnum):
    """Transition source kinds.
    转移来源种类。
    """

    ep = "ep"
    interface_ep = "interface_ep"
    causal = "causal"
    revision = "revision"
    conflict = "conflict"
    rollback = "rollback"


class InvariantKind(StrEnum):
    """InterfaceEP invariant kinds.
    接口 EP 不变量种类。
    """

    constraint = "constraint"
    failed_path = "failed_path"
    stable_fact = "stable_fact"


class ResidualKind(StrEnum):
    """InterfaceEP residual kinds.
    接口 EP 残差种类。
    """

    unresolved_conflict = "unresolved_conflict"
    missing_outcome = "missing_outcome"
    low_cohesion = "low_cohesion"
    open_question = "open_question"


class Phase(StrEnum):
    """Scheduler phases.
    调度器阶段。
    """

    orient = "orient"
    collect = "collect"
    consolidate = "consolidate"
    prune = "prune"


class ObligationName(StrEnum):
    """PC-IR proof obligations.
    PC-IR 证明义务名。
    """

    evidence_coverage = "evidence_coverage"
    no_unresolved_conflict = "no_unresolved_conflict"
    governance = "governance"
    not_revoked = "not_revoked"


class ObligationStatus(StrEnum):
    """Proof obligation status.
    证明义务状态。
    """

    satisfied = "satisfied"
    violated = "violated"
    skipped = "skipped"


# Derived constants (plain-string sets so enum members and raw strings hash alike).
# 派生常量（使用纯字符串集合，使枚举成员与原始字符串哈希一致）。
EXTRACTABLE_OBJECT_TYPES: frozenset[str] = frozenset(
    {
        ObjectType.message.value,
        ObjectType.tool_call.value,
        ObjectType.file_diff.value,
        ObjectType.test_log.value,
        ObjectType.sensor_trace.value,
    }
)
INTERNAL_OBJECT_TYPES: frozenset[str] = frozenset(
    {
        ObjectType.feedback.value,
        ObjectType.checkpoint.value,
        ObjectType.rollback.value,
        ObjectType.system.value,
        ObjectType.scheduler.value,
    }
)
WHY_EDGE_TYPES: frozenset[str] = frozenset(
    {
        EdgeType.causal.value,
        EdgeType.use.value,
        EdgeType.distill.value,
        EdgeType.revision.value,
        EdgeType.conflict.value,
    }
)
VISIBILITY_RANK: dict[str, int] = {
    Visibility.private.value: 0,
    Visibility.team.value: 1,
    Visibility.event.value: 2,
    Visibility.public.value: 3,
}
KIND_ORDER: dict[str, int] = {
    TransitionKind.ep.value: 0,
    TransitionKind.interface_ep.value: 1,
    TransitionKind.causal.value: 2,
    TransitionKind.revision.value: 3,
    TransitionKind.conflict.value: 4,
    TransitionKind.rollback.value: 5,
}
ABLATION_OF_MODE: dict[str, str] = {
    CompileMode.fast.value: "axis",
    CompileMode.axis_graph.value: "axis_graph",
    CompileMode.deep.value: "axis_graph_support",
}


def rank(visibility: str) -> int:
    """Return the lattice rank of a visibility value (private=0 ... public=3).
    返回可见性值在格中的秩（private=0 … public=3）。

    Input / 输入：
        visibility: A Visibility member or its string value.
        visibility：Visibility 成员或其字符串值。

    Output / 输出：
        Integer rank; unknown values raise KeyError.
        整数秩；未知值抛出 KeyError。

    State effect / 状态影响：
        None.
        无。
    """
    return VISIBILITY_RANK[str(visibility)]


__all__ = [
    "ABLATION_OF_MODE",
    "EXTRACTABLE_OBJECT_TYPES",
    "INTERNAL_OBJECT_TYPES",
    "KIND_ORDER",
    "VISIBILITY_RANK",
    "WHY_EDGE_TYPES",
    "AtomType",
    "CompileMode",
    "Decision",
    "EdgeType",
    "EntityType",
    "EventType",
    "FeedbackType",
    "HistoryEvent",
    "InvariantKind",
    "MemoryType",
    "NodeKind",
    "ObjectType",
    "ObligationName",
    "ObligationStatus",
    "ObservationType",
    "Phase",
    "Relation",
    "ResidualKind",
    "Reward",
    "Role",
    "Route",
    "Scale",
    "Sentiment",
    "Slot",
    "SourceKind",
    "StrEnum",
    "TransitionKind",
    "Visibility",
    "rank",
]
