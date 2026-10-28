"""Pydantic models for every HMS Elite object; witnesses are types, not conventions.
HMS Elite 全部对象的 pydantic 模型；witness 是类型而非约定。
"""

# Purpose (EN): Own every schema-bearing object (events, anchors, atoms, EPs, graph, context, ops records).
# 目的（中）：拥有全部结构对象（事件、锚点、原子、EP、图、上下文、运维记录）。

# Input (EN): Dicts and database rows.
# 输入（中）：字典与数据库行。

# Output (EN): Validated models, the Mem0 public row, normalized payloads and hashes.
# 输出（中）：已校验模型、Mem0 公开行、归一化载荷与哈希。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from hms_elite.enums import (
    AtomType,
    CompileMode,
    EdgeType,
    EventType,
    FeedbackType,
    InvariantKind,
    NodeKind,
    ObjectType,
    ObligationName,
    ObligationStatus,
    ObservationType,
    Relation,
    ResidualKind,
    Reward,
    Role,
    Route,
    Scale,
    Sentiment,
    Slot,
    SourceKind,
    TransitionKind,
    Visibility,
)
from hms_elite.exceptions import ValidationError
from hms_elite.timeutil import now_iso, shift_days, to_utc_iso
from hms_elite.utils.json import canonical_json

# 归一化为 32 字符的 UTC ISO 字符串
UtcIso = Annotated[str, BeforeValidator(to_utc_iso)]  # normalized to 32-char UTC ISO
IsoDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
Prob = Annotated[float, Field(ge=0.0, le=1.0)]
# 对应“无见证，即无意义”的类型
NonEmptyIds = Annotated[list[str], Field(min_length=1)]  # the "No Witness, No Meaning" type

_FORBIDDEN_CONTENT_KEYS = frozenset({"edges", "edge_ids", "relations", "src_id", "dst_id"})
_SLOT_NAMES = ("facts", "episodes", "preferences", "constraints", "skills", "audit")
_MIDNIGHT_SUFFIX = "T00:00:00.000000+00:00"


class HMSModel(BaseModel):
    """Base model: unknown keys rejected, assignments validated, enums stored as values.
    基础模型：拒绝未知键、赋值即校验、枚举按值存储。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True, validate_default=True)


class Scope(HMSModel):
    """Mem0 session scope.
    Mem0 会话范围。
    """

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    @field_validator("user_id", "agent_id", "run_id", mode="before")
    @classmethod
    def _trim_id(cls, value: Any, info: Any) -> str | None:
        """Coerce to str and strip; empty or inner whitespace → ValidationError(VALIDATION_001).
        转为字符串并去空白；为空或含内部空白 → ValidationError(VALIDATION_001)。
        """
        if value is None:
            return None
        text = str(value).strip()
        name = getattr(info, "field_name", "id")
        if not text:
            raise ValidationError(f"Invalid {name}: cannot be empty", "VALIDATION_001", details={"field": name})
        if any(ch.isspace() for ch in text):
            raise ValidationError(
                f"Invalid {name}: cannot contain whitespace", "VALIDATION_001", details={"field": name}
            )
        return text

    def is_empty(self) -> bool:
        """True when no id is set.
        未设置任何 id 时为真。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        return self.user_id is None and self.agent_id is None and self.run_id is None

    def require_any(self, *, context: str) -> "Scope":
        """Raise VALIDATION_001 unless at least one id is set.
        至少设置一个 id，否则抛出 VALIDATION_001。

        Input / 输入：
            context: Calling operation name, recorded in details.
            context：调用操作名，记录在 details 中。

        Output / 输出：
            The scope itself.
            范围自身。

        State effect / 状态影响：
            None.
            无。
        """
        if self.is_empty():
            raise ValidationError(
                "At least one of user_id, agent_id, or run_id is required",
                "VALIDATION_001",
                details={
                    "provided_ids": {"user_id": self.user_id, "agent_id": self.agent_id, "run_id": self.run_id},
                    "context": context,
                },
                suggestion="Provide at least one of user_id, agent_id, or run_id",
            )
        return self

    def as_filters(self) -> dict[str, str]:
        """Return only the set ids as a filter dict.
        以过滤字典返回已设置的 id。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'user_id': ..., 'agent_id': ..., 'run_id': ...} restricted to set keys.
            仅含已设置键的 {'user_id': ..., 'agent_id': ..., 'run_id': ...}。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            k: v
            for k, v in (("user_id", self.user_id), ("agent_id", self.agent_id), ("run_id", self.run_id))
            if v is not None
        }

    def key(self) -> str:
        """Mem0 session scope key: sorted 'k=v' joined by '&' with %, &, = escaped.
        Mem0 会话范围键：排序后的 'k=v' 以 '&' 连接，并转义 %、&、=。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Scope key string.
            范围键字符串。

        State effect / 状态影响：
            None.
            无。
        """

        def _escape(value: str) -> str:
            return value.replace("%", "%25").replace("&", "%26").replace("=", "%3D")

        return "&".join(f"{k}={_escape(v)}" for k, v in sorted(self.as_filters().items()))

    def matches(self, other: "Scope") -> bool:
        """True when every set key of this scope equals the other's value.
        当本范围的每个已设置键都等于对方的值时为真。

        Input / 输入：
            other: The scope to compare against.
            other：被比较的范围。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        return all(getattr(other, k) == v for k, v in self.as_filters().items())


class Witnessed(HMSModel):
    """Any state-bearing object references >= 1 anchor.
    任何承载状态的对象至少引用一个锚点。
    """

    # anchor_ids[0] 是主见证锚点：存储层的非空保证（NOT NULL 列）与默认的 history 见证，不是语义上的必要条件。
    # 见证集是存在条件：只要还剩任一未争议、未撤销的锚点，对象就有效（docs/design/WITNESS_SEMANTICS.md）。
    # anchor_ids[0] is the primary witness: the storage non-emptiness guarantee (NOT NULL column) and the default
    # history witness, NOT a necessary condition. The witness set is an existence condition: the object stays valid
    # while any undisputed, unrevoked anchor remains (docs/design/WITNESS_SEMANTICS.md).
    anchor_ids: NonEmptyIds


def event_payload_hash(*, object_type: str, role: str | None, actor_id: str | None, payload: dict) -> str:
    """sha256 of canonical_json({'object_type', 'role', 'actor_id', 'payload'}) over the normalized payload (§0.1).
    对归一化载荷计算 canonical_json({'object_type', 'role', 'actor_id', 'payload'}) 的 sha256（§0.1）。

    Input / 输入：
        object_type: Event object type value.
        object_type：事件对象类型值。

        role: Message role or None.
        role：消息角色或 None。

        actor_id: Actor id or None.
        actor_id：行动者 id 或 None。

        payload: Payload already normalized by validate_payload.
        payload：已由 validate_payload 归一化的载荷。

    Output / 输出：
        64-char hex digest.
        64 字符十六进制摘要。

    State effect / 状态影响：
        None.
        无。
    """
    blob = canonical_json(
        {
            "object_type": str(object_type),
            "role": None if role is None else str(role),
            "actor_id": actor_id,
            "payload": payload,
        }
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SourceEvent(HMSModel):
    """Raw evidence: one event per message/tool call/diff/log; payload normalized and hashed on construction.
    原始证据：每条消息/工具调用/差异/日志一个事件；构造时归一化载荷并计算哈希。
    """

    # "" → 插入时由 store 分配
    event_id: str = ""  # "" → assigned by the store on insert
    # 'hms' | 'mem0_api' | 'claude_code' | 任意字符串
    source_system: str = "hms"  # 'hms' | 'mem0_api' | 'claude_code' | free string
    object_type: ObjectType = ObjectType.message
    event_type: EventType = EventType.created
    actor_id: str | None = None
    # user|assistant|system|tool（绝不为 'extracted'）
    role: Role | None = None  # user|assistant|system|tool (never 'extracted')
    # 事件时间；调用方可设置（允许回填历史时间）
    timestamp: UtcIso = Field(default_factory=now_iso)  # event time; caller-settable (backdating allowed)
    # 为 None 时由 ingest.events.default_raw_ref 填充
    raw_ref: str | None = None  # filled by ingest.events.default_raw_ref when None
    visibility: Visibility = Visibility.private
    scope: Scope
    # 由 validate_payload 归一化（§4.3）
    payload: dict[str, Any]  # normalized by validate_payload (§4.3)
    # 为 "" 时计算（§0.1）；给定时校验（不匹配 → VALIDATION_005）
    payload_hash: str = ""  # computed when "" (§0.1); verified when given (mismatch → VALIDATION_005)
    user_meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "SourceEvent":
        """Normalize the payload, reject role 'extracted', compute/verify payload_hash.
        归一化载荷、拒绝角色 'extracted'、计算/校验 payload_hash。
        """
        if self.role == Role.extracted.value:
            raise ValidationError(
                "SourceEvent.role may not be 'extracted'",
                "VALIDATION_005",
                details={"object_type": self.object_type, "field": "role"},
            )
        payload = validate_payload(self.object_type, self.payload)
        digest = event_payload_hash(
            object_type=self.object_type, role=self.role, actor_id=self.actor_id, payload=payload
        )
        if self.payload_hash and self.payload_hash != digest:
            raise ValidationError(
                "payload_hash does not match the normalized payload",
                "VALIDATION_005",
                details={"object_type": self.object_type, "field": "payload_hash", "expected": digest},
            )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_hash", digest)
        return self


class Anchor(HMSModel):
    """Evidence anchor: the witness every HMS object must cite.
    证据锚点：每个 HMS 对象必须引用的见证。
    """

    anchor_id: str
    source_event_id: str
    hash: str
    source_ref: str | None = None
    timestamp: UtcIso
    ingested_at: UtcIso
    permission_scope: Visibility = Visibility.private
    scope: Scope
    revoked_at: UtcIso | None = None

    def evidence_line(self) -> str:
        """Render the one-line evidence citation used by Work State Cards.
        渲染工作状态卡使用的单行证据引用。

        Input / 输入：
            none.
            无。

        Output / 输出：
            f"{source_ref or 'anchor'} sha256:{hash[:12]} @{timestamp}".
            f"{source_ref or 'anchor'} sha256:{hash[:12]} @{timestamp}"。

        State effect / 状态影响：
            None.
            无。
        """
        return f"{self.source_ref or 'anchor'} sha256:{self.hash[:12]} @{self.timestamp}"


class Observation(Witnessed):
    """A typed, dated statement extracted from messages.
    从消息中抽取的带类型与日期的陈述。
    """

    obs_id: str = ""
    type: ObservationType
    # 'user' | 'project' | 'assistant' | 实体显示文本
    subject: str = "user"  # 'user' | 'project' | 'assistant' | entity display text
    # 小写 snake 形式的动词（§11.1 谓词表）
    predicate: str  # lowercase snake verb (§11.1 predicate table)
    objects: list[str] = Field(default_factory=list)
    relation: Relation = Relation.single_fact
    text: str = Field(min_length=1)
    date_observed: IsoDate
    valid_from: UtcIso | None = None
    valid_to: UtcIso | None = None
    confidence: Prob = 0.7
    visibility: Visibility = Visibility.private
    attributed_to: Literal["user", "assistant"] = "user"
    # The named person who made the statement when the transcript is not a plain user/assistant chat (multi-speaker
    # recordings); None in ordinary chat. `attributed_to` stays the conversation side. Not persisted on the
    # observations row; it reaches the atom as actor_id when the witnessing event carries none (ingest.atoms).
    # 当转录不是普通的 user/assistant 对话（多说话人录音）时，做出该陈述的具名人物；普通聊天为 None。
    # `attributed_to` 仍是对话侧别。不持久化到 observations 行；当见证事件不带 actor_id 时作为原子的 actor_id（ingest.atoms）。
    speaker: str | None = None
    sentiment: Sentiment = Sentiment.unknown
    topic_tags: list[str] = Field(default_factory=list)
    entity_mentions: list[str] = Field(default_factory=list)
    # 引用解析之后的真实 atom_id
    supersedes_hint_ids: list[str] = Field(default_factory=list)  # real atom_ids after ref resolution
    extracted_by: str = "rule@obs-v1"
    scope: Scope
    atom_id: str | None = None
    created_at: UtcIso | None = None


class Kappa(HMSModel):
    """κ: constraints, visibility and evidence anchors of an EP.
    κ：EP 的约束、可见性与证据锚点。
    """

    constraints: list[str] = Field(default_factory=list)
    visibility: Visibility = Visibility.private
    # 锚点 id 列表
    evidence: list[str] = Field(default_factory=list)  # anchor ids


class Slots(HMSModel):
    """Isomorphic six-tuple shared by EP (L1) and InterfaceEP (L2/L3).
    跨尺度同构六元组，EP（L1）与 InterfaceEP（L2/L3）共用。
    """

    u: str = Field(min_length=1)
    b: str = ""
    a: str = Field(min_length=1)
    y: str = ""
    r: Reward = Reward.unknown
    kappa: Kappa = Field(default_factory=Kappa)

    @model_validator(mode="after")
    def _closure(self) -> "Slots":
        """Closure: an outcome or a constraint must be present.
        闭包：必须有结果或约束。
        """
        if not (self.y.strip() or self.kappa.constraints):
            raise ValueError("EP needs an outcome or a constraint")
        return self

    def gloss(self) -> str:
        """One-line human gloss of the tuple.
        六元组的单行可读描述。

        Input / 输入：
            none.
            无。

        Output / 输出：
            f"[{r}] {u} → {a} → {y or '?'}".
            f"[{r}] {u} → {a} → {y or '?'}"。

        State effect / 状态影响：
            None.
            无。
        """
        return f"[{self.r}] {self.u} → {self.a} → {self.y or '?'}"


class EP(Witnessed):
    """L1 experience package (u, b, a, y, r, κ) at a time anchor.
    带时间锚的 L1 经验包（u, b, a, y, r, κ）。
    """

    ep_id: str = ""
    scale: Literal["L1_micro"] = "L1_micro"
    slots: Slots
    time_anchor: UtcIso
    # 插入时默认取 time_anchor
    valid_from: UtcIso | None = None  # defaults to time_anchor on insert
    valid_to: UtcIso | None = None
    entity_mentions: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    # 建立 'use' 边之前先校验其指向已存在的 EP
    uses_ep_ids: list[str] = Field(default_factory=list)  # validated against existing EPs before a 'use' edge
    confidence: Prob = 0.7
    visibility: Visibility = Visibility.private
    extracted_by: str = "rule@closure-v1"
    scope: Scope
    atom_id: str | None = None
    created_at: UtcIso | None = None


class MemoryAtom(Witnessed):
    """Coordinates + provenance only; never edges.
    只有坐标与出处，不存边。
    """

    atom_id: str = ""
    # 血缘 id（Mem0 的 'id'）；为 "" 时插入将新建血缘
    memory_id: str = ""  # lineage id (Mem0 'id'); "" → new lineage on insert
    version: int = Field(default=1, ge=1)
    atom_type: AtomType
    # §4.2 定义的形状；必须包含与 text 相等的 'text'
    content: dict[str, Any]  # §4.2 shapes; must contain 'text' == text
    text: str = Field(min_length=1)
    # md5(text)；为 "" 时计算
    hash: str = ""  # md5(text); computed when ""
    time_anchor: UtcIso | None = None
    # 插入时默认取 time_anchor 或 created_at
    valid_from: UtcIso | None = None  # defaults to time_anchor or created_at on insert
    valid_to: UtcIso | None = None
    # 归一化后的实体键
    entity_ids: list[str] = Field(default_factory=list)  # normalized entity keys
    topic_tags: list[str] = Field(default_factory=list)
    scale_level: Scale
    role: Role = Role.extracted
    sentiment: Sentiment | None = None
    subject: str | None = None
    predicate: str | None = None
    # 为 SQL 槽位规则提升到列级的字段
    obs_type: ObservationType | None = None  # promoted for SQL slot rules
    source_kind: SourceKind
    # 来源 id：event_id | obs_id | ep_id | interface_ep_id | feedback_id | atom_id(skill)
    source_id: str  # event_id | obs_id | ep_id | interface_ep_id | feedback_id | atom_id(skill)
    source_ep_ids: list[str] = Field(default_factory=list)
    confidence: Prob = 0.7
    extracted_by: str
    visibility: Visibility = Visibility.private
    scope: Scope
    actor_id: str | None = None
    attributed_to: Literal["user", "assistant"] | None = None
    # 记忆类型：'procedural_memory' 或 None
    memory_type: str | None = None  # 'procedural_memory' | None
    # 调用方命名空间；保留键由 memory.utils 剥离
    user_meta: dict[str, Any] = Field(default_factory=dict)  # caller namespace; reserved keys stripped by memory.utils
    # 归档原因：revised|deleted|feedback|rollback|decayed|conflict
    archived_reason: str | None = None  # revised|deleted|feedback|rollback|decayed|conflict
    created_at: UtcIso | None = None
    updated_at: UtcIso | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "MemoryAtom":
        """content['text'] == text; no edge keys in content; valid_to >= valid_from; hash computed when empty.
        content['text'] == text；content 无边键；valid_to >= valid_from；hash 为空时计算。
        """
        forbidden = sorted(_FORBIDDEN_CONTENT_KEYS.intersection(self.content))
        if forbidden:
            raise ValueError(f"MemoryAtom.content must not carry edges: {forbidden}")
        if self.content.get("text") != self.text:
            raise ValueError("MemoryAtom.content['text'] must equal MemoryAtom.text")
        if self.valid_from is not None and self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("MemoryAtom.valid_to must be >= valid_from")
        if not self.hash:
            object.__setattr__(self, "hash", hashlib.md5(self.text.encode("utf-8")).hexdigest())
        return self

    def is_valid_at(self, as_of: str) -> bool:
        """Liveness at an instant (mirrors LIVE_SQL: not archived, valid_from <= as_of < valid_to).
        某时刻的存活性（等价于 LIVE_SQL：未归档、valid_from <= as_of < valid_to）。

        Input / 输入：
            as_of: Reference instant (any form accepted by to_utc_iso).
            as_of：参考时刻（to_utc_iso 接受的任意形式）。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        instant = to_utc_iso(as_of)
        if self.archived_reason is not None:
            return False
        if self.valid_from is not None and self.valid_from > instant:
            return False
        return self.valid_to is None or self.valid_to > instant

    def expiration_date(self) -> str | None:
        """Mem0 expiration_date: valid_to's date minus one when valid_to is a next-day-midnight boundary.
        Mem0 的 expiration_date：当 valid_to 为次日零点边界时，取其日期减一天。

        Input / 输入：
            none.
            无。

        Output / 输出：
            'YYYY-MM-DD' or None.
            'YYYY-MM-DD' 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        if self.valid_to is not None and self.valid_to.endswith(_MIDNIGHT_SUFFIX):
            return shift_days(self.valid_to[:10], -1)
        return None

    def to_public_dict(
        self, *, score: float | None = None, score_details: dict | None = None, anchor_ids: list[str] | None = None
    ) -> dict:
        """The Mem0 row (§4.1) with promoted keys in Mem0 order and the 'hms' sub-dict.
        Mem0 行（§4.1）：按 Mem0 顺序提升的键与 'hms' 子字典。

        Input / 输入：
            score: Search score; included only when given.
            score：搜索得分；给定时才包含。

            score_details: Explain block; included only when given.
            score_details：解释块；给定时才包含。

            anchor_ids: Undisputed anchor list from the reader; defaults to self.anchor_ids.
            anchor_ids：读取器提供的无争议锚点列表；默认 self.anchor_ids。

        Output / 输出：
            Dict whose key order is exactly the §4.1 order; 'metadata' holds only caller user_meta (or None).
            键顺序严格为 §4.1 顺序的字典；'metadata' 仅含调用方 user_meta（或 None）。

        State effect / 状态影响：
            None.
            无。
        """
        row: dict[str, Any] = {
            "id": self.memory_id,
            "memory": self.text,
            "hash": self.hash,
            "metadata": dict(self.user_meta) if self.user_meta else None,
        }
        if score is not None:
            row["score"] = score
        row["created_at"] = self.created_at
        row["updated_at"] = self.updated_at
        for key in ("user_id", "agent_id", "run_id"):
            value = getattr(self.scope, key)
            if value is not None:
                row[key] = value
        if self.actor_id is not None:
            row["actor_id"] = self.actor_id
        if self.atom_type == AtomType.message.value:
            row["role"] = self.role
        if self.attributed_to is not None:
            row["attributed_to"] = self.attributed_to
        expiration = self.expiration_date()
        if expiration is not None:
            row["expiration_date"] = expiration
        if self.memory_type is not None:
            row["memory_type"] = self.memory_type
        row["atom_id"] = self.atom_id
        row["version"] = self.version
        row["atom_type"] = self.atom_type
        row["scale_level"] = self.scale_level
        row["anchor_ids"] = list(anchor_ids) if anchor_ids is not None else list(self.anchor_ids)
        row["valid_from"] = self.valid_from
        row["valid_to"] = self.valid_to
        row["confidence"] = self.confidence
        row["hms"] = {
            "time_anchor": self.time_anchor,
            "entity_ids": list(self.entity_ids),
            "topic_tags": list(self.topic_tags),
            "sentiment": self.sentiment,
            "subject": self.subject,
            "predicate": self.predicate,
            "obs_type": self.obs_type,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_ep_ids": list(self.source_ep_ids),
            "extracted_by": self.extracted_by,
            "visibility": self.visibility,
            "archived_reason": self.archived_reason,
        }
        if score_details is not None:
            row["score_details"] = score_details
        return row


# 测试 39 断言 RESERVED_ATOM_KEYS ⊇ memory_atoms 的每一列（PRAGMA table_info）
RESERVED_ATOM_KEYS: frozenset[str] = (
    frozenset(MemoryAtom.model_fields)
    # memory_atoms 的每个 *_json 列
    | frozenset(f + "_json" for f in ("content", "user_meta", "source_ep_ids"))  # every *_json column of memory_atoms
    | frozenset(
        {
            "id",
            "memory",
            "data",
            "score",
            "score_details",
            "hms",
            "user_id",
            "agent_id",
            "run_id",
            "expiration_date",
            "text_lemmatized",
            "primary_anchor_id",
        }
    )
)  # test 39: RESERVED_ATOM_KEYS ⊇ every memory_atoms column / 测试 39：保留键覆盖全部列


class AxisQuery(HMSModel):
    """A query plan over the multi-view index (HMS §3.5).
    多视角索引上的查询计划（HMS §3.5）。
    """

    # 覆盖七个轴 + scope + meta 的 Mem0 过滤语法（§13.1）
    filters: dict[str, Any] = Field(default_factory=dict)  # Mem0 grammar over the seven axes + scope + meta (§13.1)
    project_axis: Literal["time", "entity", "value", "topic", "atom"] = "atom"
    scale_constraint: list[Scale] | None = None
    aggregation: Literal["list", "latest", "count", "summary", "topk"] | None = "list"
    limit: int = Field(default=20, ge=0)
    as_of: UtcIso | None = None
    include_expired: bool = False
    # None → 由 execute_axis_query 从 `filters` 内的 scope 键推导
    scope: Scope | None = None  # None → derived from the scope keys inside `filters` by execute_axis_query
    # 语义/关键词查询；None → 仅按轴过滤，按时间排序
    text: str | None = None  # semantic/keyword query; None → axis-only, time order
    # 逃生舱：同一套语法，但只针对 user_meta 编译
    meta_filters: dict[str, Any] | None = None  # escape hatch: same grammar, compiled against user_meta only
    # planner 输出（§13.2）；直接构造时为 {}
    plan: dict[str, Any] = Field(default_factory=dict)  # planner output (§13.2); {} when built directly

    @model_validator(mode="after")
    def _topk_needs_text(self) -> "AxisQuery":
        """aggregation 'topk' requires a text query (VALIDATION_013).
        聚合 'topk' 需要文本查询（VALIDATION_013）。
        """
        if self.aggregation == "topk" and self.text is None:
            raise ValidationError(
                "aggregation='topk' requires a text query",
                "VALIDATION_013",
                details={"aggregation": "topk", "field": "text"},
            )
        return self


class AtomHit(HMSModel):
    """One scored atom in a query result.
    查询结果中一个带得分的原子。
    """

    atom: MemoryAtom
    score: float = 0.0
    score_details: dict[str, Any] = Field(default_factory=dict)
    matched_axes: list[str] = Field(default_factory=list)


class AxisResult(HMSModel):
    """Executed AxisQuery: hits plus the requested projection/aggregation (HMS §3.5).
    已执行的轴查询：命中与投影/聚合结果（HMS §3.5）。
    """

    # 始终是按结果顺序排列的（已聚合、已截断的）原子
    hits: list[AtomHit]  # always the (aggregated, limited) atoms in result order
    project_axis: Literal["time", "entity", "value", "topic", "atom"]
    aggregation: Literal["list", "latest", "count", "summary", "topk"]
    # §13.3 第 5 步按 project_axis 产出；project_axis 为 'atom' 时是 None
    projection: list[Any] | dict[str, list[str]] | None = None  # §13.3 step 5 per project_axis; None for 'atom'
    # 仅当 aggregation == 'count' 时有值
    count: int | None = None  # aggregation == 'count'
    # 仅当 aggregation == 'summary' 时有值
    summary: str | None = None  # aggregation == 'summary'
    primary_axis: str = "time"
    # 排序查询的池溯源（Pool.stats()：pool_size、channel_sizes、channels_used、gate_count、whole）；仅轴查询为 {}
    retrieval: dict[str, Any] = Field(default_factory=dict)  # Pool.stats() of the winning level; {} for axis-only


class GraphNode(Witnessed):
    """Support Graph node; node_id == atom_id (one node per atom version).
    支持图节点；node_id == atom_id（每个原子版本一个节点）。
    """

    # 节点 id 等于 atom_id
    node_id: str  # == atom_id
    node_kind: NodeKind
    # 来源引用：event_id | obs_id | ep_id | interface_ep_id | atom_id(skill)
    ref_id: str  # event_id | obs_id | ep_id | interface_ep_id | atom_id(skill)
    scale: Scale
    scope: Scope
    valid: bool = True
    invalidated_at: UtcIso | None = None
    mentions: int = 1
    created_at: UtcIso | None = None


class GraphEdge(HMSModel):
    """Witnessed typed edge; boundary edges additionally cite the low-level edge they project.
    带见证的类型化边；边界边还需引用其投影的低层边。
    """

    edge_id: str = ""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    label: str = ""
    scale: Literal["L1_micro", "L2_meso", "L3_macro"] = "L1_micro"
    # 仅 operators.project 写入的边为 True
    is_boundary: bool = False  # True only for edges written by operators.project
    confidence: Prob = 0.7
    mentions: int = 1
    witness_anchor_ids: NonEmptyIds
    # 当且仅当 is_boundary 为真时必填（校验器 + DDL CHECK）
    witness_edge_id: str | None = None  # REQUIRED iff is_boundary (validator + DDL CHECK)
    created_by: Literal["ingest", "consolidation", "feedback", "ops", "project"]
    # 提议者：'rule' | 'closure' | 'timestamp' | 'llm:<identity>' | 'project-v1'
    proposer: str = "rule"  # 'rule' | 'closure' | 'timestamp' | 'llm:<identity>' | 'project-v1'
    valid: bool = True
    invalidated_at: UtcIso | None = None
    invalidated_reason: str | None = None
    # 仅 conflict 边使用
    resolved_at: UtcIso | None = None  # conflict edges only
    resolved_by_anchor_id: str | None = None
    resolution_edge_id: str | None = None
    scope: Scope
    created_at: UtcIso | None = None

    @model_validator(mode="after")
    def _edge_rules(self) -> "GraphEdge":
        """src != dst; boundary ⇒ witness_edge_id; resolution fields only on conflict edges and witnessed.
        src != dst；边界边必须有 witness_edge_id；解决字段仅限冲突边且需见证。
        """
        if self.src_id == self.dst_id:
            raise ValueError("GraphEdge must not be a self edge")
        if self.is_boundary and not self.witness_edge_id:
            raise ValueError("a boundary edge requires witness_edge_id")
        if self.edge_type != EdgeType.conflict.value and (
            self.resolved_at is not None
            or self.resolved_by_anchor_id is not None
            or self.resolution_edge_id is not None
        ):
            raise ValueError("resolved_* fields are only allowed on conflict edges")
        if self.resolved_at is not None and not self.resolved_by_anchor_id:
            raise ValueError("resolved_at requires resolved_by_anchor_id")
        return self


class BoundaryEdge(HMSModel):
    """An edge crossing the support set boundary of an InterfaceEP.
    跨越接口 EP 支持集边界的边。
    """

    edge_id: str
    inner_node_id: str
    outer_node_id: str
    # src ∈ 支持集时为 'out'
    direction: Literal["in", "out"]  # 'out' when src ∈ support set
    edge_type: EdgeType
    witness_anchor_ids: NonEmptyIds


class Invariant(HMSModel):
    """A statement that holds across the support set of an InterfaceEP.
    在接口 EP 支持集上成立的陈述。
    """

    text: str = Field(min_length=1)
    kind: InvariantKind = InvariantKind.constraint
    # 已解析的支持节点引用（由 §14.2 填充）
    support_node_ids: NonEmptyIds  # resolved support refs (§14.2 fills it)
    # 等于 sorted(∪ store.atom_anchor_ids(n) for n in support_node_ids)，见 §14.2
    anchor_ids: NonEmptyIds  # == sorted(∪ store.atom_anchor_ids(n) for n in support_node_ids) (§14.2)
    # 等于 round(mean(atom.confidence for n in support_node_ids), 4)，见 §14.2
    confidence: Prob = 0.7  # == round(mean(atom.confidence for n in support_node_ids), 4) (§14.2)


class Residual(HMSModel):
    """What the coarsening could not resolve.
    粗化未能解决的残差。
    """

    text: str = Field(min_length=1)
    kind: ResidualKind
    support_node_ids: list[str] = Field(default_factory=list)
    # 支持锚点的有序并集（允许为空）
    anchor_ids: list[str] = Field(default_factory=list)  # sorted union of support anchors (empty allowed)
    edge_ids: list[str] = Field(default_factory=list)


class InterfaceEP(HMSModel):
    """High-level EP = interface to a low-level subgraph, never a summary.
    高层 EP 是低层子图的接口，绝非摘要。
    """

    interface_ep_id: str = ""
    scale: Literal["L2_meso", "L3_macro"] = "L2_meso"
    slots: Slots
    # 槽位名 → 支持节点 id（每个槽位至少引用 1 个）
    slot_support: dict[str, list[str]] = Field(
        default_factory=dict
    )  # slot name → support node ids, every slot cites ≥1 / 槽位名 → 支撑节点 id，每槽至少引用 1 个
    support_node_ids: Annotated[list[str], Field(min_length=2)]
    support_edge_ids: list[str] = Field(default_factory=list)
    boundary_edges: list[BoundaryEdge] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    residuals: list[Residual] = Field(default_factory=list)
    # == ∪ anchors(支持节点) ∪ witnesses(支持边)
    evidence_anchor_ids: NonEmptyIds  # == ∪ anchors(support nodes) ∪ witnesses(support edges)
    coarsen_op_version: str = "rule-coarsen-v1"
    cohesion_score: Prob = 0.0
    # 其在索引平面的 episode 原子，等于 node_id
    atom_id: str | None = None  # its episode atom (index plane) == node_id
    scope: Scope
    valid: bool = True
    valid_to: UtcIso | None = None
    created_at: UtcIso | None = None

    @model_validator(mode="after")
    def _slots_cite_support(self) -> "InterfaceEP":
        """Every non-empty slot cites ≥1 support node id; cited ids ⊂ support_node_ids.
        每个非空槽位至少引用一个支持节点 id；被引用 id ⊂ support_node_ids。
        """
        support = set(self.support_node_ids)
        slots = self.slots
        non_empty = {
            "u": bool(slots.u.strip()),
            "a": bool(slots.a.strip()),
            "y": bool(slots.y.strip()),
            "b": bool(slots.b.strip()),
            "r": slots.r != Reward.unknown.value,
            "kappa": bool(slots.kappa.constraints),
        }
        for name, needed in non_empty.items():
            cited = self.slot_support.get(name, [])
            if needed and not cited:
                raise ValueError(f"InterfaceEP slot '{name}' cites no support node")
            unknown = [node for node in cited if node not in support]
            if unknown:
                raise ValueError(f"InterfaceEP slot '{name}' cites nodes outside support_node_ids: {unknown}")
        return self


class ContextItem(Witnessed):
    """One selected item of a ContextIR slot.
    ContextIR 槽位中一个被选中的项。
    """

    # §15.1 id rule: atom_id | '<atom_id>#kappa:<i>' | '<atom_id>#inv:<i>' | 'anc:<anchor_id>'
    #                | 'sig:<name>' | 'note:<n>'
    # §15.1 的 id 规则：atom_id | '<atom_id>#kappa:<i>' | '<atom_id>#inv:<i>' | 'anc:<anchor_id>'
    #                   | 'sig:<name>' | 'note:<n>'，共六种形式
    item_id: str
    slot: Slot
    text: str = Field(min_length=1)
    atom_id: str | None = None
    memory_id: str | None = None
    ep_id: str | None = None
    score: float = 0.0
    time_anchor: UtcIso | None = None
    valid_from: UtcIso | None = None
    valid_to: UtcIso | None = None
    scale_level: Scale | None = None
    atom_type: AtomType | None = None
    visibility: Visibility = Visibility.private
    source: Literal["axis", "graph", "interface_ep", "system"] = "axis"
    hops: int = 0
    tokens: int = 0
    explain: dict[str, Any] = Field(default_factory=dict)


class ContextIR(HMSModel):
    """Compiled context: six disjoint witnessed slots plus retrieval provenance.
    编译后的上下文：六个互斥且有见证的槽位加检索出处。
    """

    query: str
    goal: str | None = None
    as_of: UtcIso
    scope: Scope
    mode: CompileMode = CompileMode.auto
    route: Route = Route.recall
    facts: list[ContextItem] = Field(default_factory=list)
    episodes: list[ContextItem] = Field(default_factory=list)
    preferences: list[ContextItem] = Field(default_factory=list)
    constraints: list[ContextItem] = Field(default_factory=list)
    skills: list[ContextItem] = Field(default_factory=list)
    audit: list[ContextItem] = Field(default_factory=list)
    next_action: str | None = None
    budget_tokens: int = Field(default=1200, gt=0)
    used_tokens: int = 0
    retrieval: dict[str, Any] = Field(default_factory=dict)
    # 被选中项之间的冲突边 id
    unresolved_conflicts: list[str] = Field(default_factory=list)  # edge ids among selected items

    @model_validator(mode="after")
    def _disjoint_slots(self) -> "ContextIR":
        """item_ids pairwise disjoint across the six slots; item.slot == its slot name.
        六个槽位间 item_id 两两互斥；item.slot 等于所在槽位名。
        """
        seen: dict[str, str] = {}
        for slot_name in _SLOT_NAMES:
            for item in getattr(self, slot_name):
                if item.slot != slot_name:
                    raise ValueError(f"item {item.item_id!r} carries slot {item.slot!r} but sits in {slot_name!r}")
                if item.item_id in seen:
                    raise ValueError(
                        f"item id {item.item_id!r} appears in both {seen[item.item_id]!r} and {slot_name!r}"
                    )
                seen[item.item_id] = slot_name
        return self

    def items(self) -> list[ContextItem]:
        """All items in slot order facts, episodes, preferences, constraints, skills, audit.
        按 facts、episodes、preferences、constraints、skills、audit 顺序的全部项。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Flat list of ContextItem.
            ContextItem 的扁平列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [item for slot_name in _SLOT_NAMES for item in getattr(self, slot_name)]

    def anchor_ids(self) -> list[str]:
        """Sorted union of every item's anchor ids.
        全部项锚点 id 的排序并集。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Sorted anchor id list.
            排序后的锚点 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return sorted({anchor for item in self.items() for anchor in item.anchor_ids})

    def to_text(self) -> str:
        """Render the deterministic Work State Card (compiler.render.render_card).
        渲染确定性的工作状态卡（compiler.render.render_card）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Card text.
            卡片文本。

        State effect / 状态影响：
            None (imports the renderer lazily to respect the import topology).
            无（延迟导入渲染器以遵守导入拓扑）。
        """
        from hms_elite.compiler.render import render_card

        return render_card(self)

    def to_pcir(self, *, require_witness: bool = True) -> "PCIR":
        """Lower to PC-IR (compiler.pc_ir.lower); store-free, witness rows come from audit items.
        下沉为 PC-IR（compiler.pc_ir.lower）；不依赖存储，见证行来自 audit 项。

        Input / 输入：
            require_witness: Raise ProofObligationError on violations instead of returning ok() False.
            require_witness：违反义务时抛出 ProofObligationError，而非返回 ok() False。

        Output / 输出：
            PCIR; a ContextIR compiled with selection='coverage' (retrieval['hierarchy'] present) lowers through
            compiler.scc.lower_ir and carries the fifth obligation requirements_covered.
            PCIR；以 selection='coverage' 编译的 ContextIR（retrieval 含 'hierarchy'）经 compiler.scc.lower_ir 降低，
            并携带第五条义务 requirements_covered。

        State effect / 状态影响：
            None.
            无。
        """
        if self.retrieval.get("hierarchy"):
            from hms_elite.compiler.scc import lower_ir

            return lower_ir(self, require_witness=require_witness)
        from hms_elite.compiler.pc_ir import lower

        return lower(self, require_witness=require_witness)


class ProofObligation(HMSModel):
    """One PC-IR obligation and its status.
    一条 PC-IR 义务及其状态。
    """

    name: ObligationName
    status: ObligationStatus
    details: dict[str, Any] = Field(default_factory=dict)


class PCIR(HMSModel):
    """Proof-carrying IR: payload, witness map, human gloss and obligations.
    携带证明的 IR：载荷、见证映射、可读描述与义务。
    """

    # 载荷，形状见 §15.4
    payload: dict[str, Any]  # §15.4
    # {'items': {item_id: [anchor_ids]}, 'anchors': [Anchor 形状的字典], '_all': 有序并集}
    witness: dict[str, Any]  # {'items': {item_id: [anchor_ids]}, 'anchors': [Anchor-like dicts], '_all': sorted union}
    # 等于 ContextIR.to_text() 的结果
    human_gloss: str  # == ContextIR.to_text()
    proof_obligations: list[ProofObligation]

    def ok(self) -> bool:
        """True when no obligation is violated.
        没有义务被违反时为真。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        return all(ob.status != ObligationStatus.violated.value for ob in self.proof_obligations)


class StatePackage(HMSModel):
    """Handoff package: context + PC-IR + the evidence closure a receiver needs to recreate the state.
    交接包：上下文 + PC-IR + 接收方重建状态所需的证据闭包。
    """

    package_id: str
    version: Literal["hms-elite/state-package/1"] = "hms-elite/state-package/1"
    created_at: UtcIso
    goal: str | None
    scope: Scope
    context: ContextIR
    pcir: PCIR
    atoms: list[MemoryAtom]
    eps: list[EP]
    interface_eps: list[InterfaceEP]
    edges: list[GraphEdge]
    # 见证闭包
    anchors: list[Anchor]  # witness closure
    # 证据闭包：每个锚点恰好对应一个事件
    source_events: list[SourceEvent]  # evidence closure: exactly one event per anchor
    checkpoint_id: str | None = None

    @model_validator(mode="after")
    def _closures(self) -> "StatePackage":
        """Witness closure and evidence closure must hold.
        见证闭包与证据闭包必须成立。
        """
        known = {a.anchor_id for a in self.anchors}
        referenced: set[str] = set(self.context.anchor_ids())
        for atom in self.atoms:
            referenced.update(atom.anchor_ids)
        for ep in self.eps:
            referenced.update(ep.anchor_ids)
        for iep in self.interface_eps:
            referenced.update(iep.evidence_anchor_ids)
        for edge in self.edges:
            referenced.update(edge.witness_anchor_ids)
        missing = sorted(referenced - known)
        if missing:
            raise ValueError(f"StatePackage witness closure broken; missing anchors: {missing}")
        anchor_events = {a.source_event_id for a in self.anchors}
        event_ids = [e.event_id for e in self.source_events]
        if anchor_events != set(event_ids) or len(event_ids) != len(set(event_ids)):
            raise ValueError("StatePackage evidence closure broken: source_events must be exactly one event per anchor")
        return self

    def to_json(self, **kw: Any) -> str:
        """Serialize with model_dump_json.
        用 model_dump_json 序列化。

        Input / 输入：
            kw: Passed through to model_dump_json.
            kw：透传给 model_dump_json。

        Output / 输出：
            JSON string.
            JSON 字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return self.model_dump_json(**kw)

    @classmethod
    def from_json(cls, s: str) -> "StatePackage":
        """Parse a package produced by to_json.
        解析 to_json 产生的交接包。

        Input / 输入：
            s: JSON string.
            s：JSON 字符串。

        Output / 输出：
            StatePackage.
            StatePackage。

        State effect / 状态影响：
            None.
            无。
        """
        return cls.model_validate_json(s)


class Checkpoint(HMSModel):
    """Full-state snapshot of a scope.
    某范围的全状态快照。
    """

    checkpoint_id: str = ""
    label: str | None = None
    anchor_id: str
    scope: Scope
    # atom_id → {'valid_to', 'archived_reason'}，覆盖 scope 内的每一个原子
    atom_state: dict[str, dict[str, str | None]]  # atom_id → {'valid_to', 'archived_reason'} for EVERY atom in scope
    # edge_id → valid，覆盖 scope 内的每一条边
    edge_state: dict[str, bool]  # edge_id → valid, every edge in scope
    # interface_ep_id → valid（接口 EP 的有效性）
    iep_state: dict[str, bool]  # interface_ep_id → valid
    # ep_id → eps.valid_to，覆盖 scope 内的每一个 EP
    ep_state: dict[str, str | None] = Field(default_factory=dict)  # ep_id → eps.valid_to, every EP in scope
    history_seq: int
    # 等于 sha256(canonical_json({atom_state, edge_state, iep_state, ep_state}))
    snapshot_hash: str  # sha256(canonical_json({atom_state, edge_state, iep_state, ep_state}))
    created_at: UtcIso


class FeedbackEvent(HMSModel):
    """A StateFeedback record and the actions it triggered.
    一条状态反馈记录及其触发的动作。
    """

    feedback_id: str = ""
    # 仅当 state_missing 且没有既有目标时才为 None（§16.1）
    target_id: str | None = None  # None only for state_missing without an existing target (§16.1)
    target_kind: Literal["atom", "edge", "interface_ep", "context_item", "anchor", "scope"]
    feedback: FeedbackType
    reason: str | None = None
    correction: str | dict[str, Any] | None = None
    actor_id: str | None = None
    anchor_id: str
    # 形如 [{'action': str, ...ids}]
    actions: list[dict[str, Any]] = Field(default_factory=list)  # [{'action': str, ...ids}]
    scope: Scope
    created_at: UtcIso


class Transition(HMSModel):
    """(S_t, a_t, S_{t+1}, r_t, κ_t, w_t) training tuple with uniform κ_t.
    统一 κ_t 形状的 (S_t, a_t, S_{t+1}, r_t, κ_t, w_t) 训练元组。
    """

    transition_id: str = ""
    source_kind: TransitionKind
    # 来源 id：ep_id | interface_ep_id | edge_id
    source_id: str  # ep_id | interface_ep_id | edge_id
    scale: Literal["L1_micro", "L2_meso", "L3_macro"]
    s_t: dict[str, Any]
    a_t: str = Field(min_length=1)
    s_t1: dict[str, Any]
    r_t: Reward
    kappa_t: dict[str, Any]
    w_t: NonEmptyIds
    time_t: UtcIso
    time_t1: UtcIso
    scope: Scope
    created_at: UtcIso | None = None

    def to_tuple(self) -> tuple[dict, str, dict, str, dict, list[str]]:
        """The six-tuple (s_t, a_t, s_t1, r_t, kappa_t, w_t).
        六元组 (s_t, a_t, s_t1, r_t, kappa_t, w_t)。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Tuple of the six components.
            六个分量组成的元组。

        State effect / 状态影响：
            None.
            无。
        """
        return (self.s_t, self.a_t, self.s_t1, self.r_t, self.kappa_t, list(self.w_t))

    @classmethod
    def from_ep(cls, ep: EP, *, scope: Scope, permission_scope: str = "private") -> "Transition":
        """Build the §16.4 'ep' transition row from an EP (group A owns it; ingest and ops both call it).
        由 EP 构建 §16.4 的 'ep' 转移行（A 组拥有；ingest 与 ops 都会调用）。

        Input / 输入：
            ep: The EP (after insert, so ids and valid_from are set).
            ep：EP（插入后，id 与 valid_from 已设置）。

            scope: Scope of the transition row.
            scope：转移行的范围。

            permission_scope: The primary anchor's permission_scope.
            permission_scope：主锚点的 permission_scope。

        Output / 输出：
            Transition with source_kind 'ep', w_t = ep.anchor_ids, time_t == time_t1 == time_anchor.
            source_kind 为 'ep'、w_t = ep.anchor_ids、time_t == time_t1 == time_anchor 的 Transition。

        State effect / 状态影响：
            None.
            无。
        """
        kappa = ep.slots.kappa
        valid_from = ep.valid_from if ep.valid_from is not None else ep.time_anchor
        return cls(
            source_kind=TransitionKind.ep,
            source_id=ep.ep_id,
            scale="L1_micro",
            s_t={
                "u": ep.slots.u,
                "b": ep.slots.b,
                "kappa": kappa.model_dump(),
                "ep_id": ep.ep_id,
                "valid_from": valid_from,
                "valid_to": ep.valid_to,
            },
            a_t=ep.slots.a,
            s_t1={"y": ep.slots.y, "r": ep.slots.r, "ep_id": ep.ep_id},
            r_t=ep.slots.r,
            kappa_t=kappa_t_of(
                constraints=kappa.constraints,
                visibility=kappa.visibility,
                permission_scope=permission_scope,
                provenance={"evidence": list(kappa.evidence), "ep_id": ep.ep_id},
            ),
            w_t=list(ep.anchor_ids),
            time_t=ep.time_anchor,
            time_t1=ep.time_anchor,
            scope=scope,
        )

    @classmethod
    def from_interface_ep(
        cls, iep: InterfaceEP, *, scope: Scope, time_t: str, time_t1: str, permission_scope: str = "private"
    ) -> "Transition":
        """Build the §16.4 'interface_ep' transition row from an InterfaceEP.
        由 InterfaceEP 构建 §16.4 的 'interface_ep' 转移行。

        Input / 输入：
            iep: The InterfaceEP.
            iep：接口 EP。

            scope: Scope of the transition row.
            scope：转移行的范围。

            time_t: min(support atoms' time_anchor).
            time_t：支持原子 time_anchor 的最小值。

            time_t1: max(support atoms' time_anchor).
            time_t1：支持原子 time_anchor 的最大值。

            permission_scope: The primary anchor's permission_scope.
            permission_scope：主锚点的 permission_scope。

        Output / 输出：
            Transition with source_kind 'interface_ep', scale == iep.scale, w_t = evidence_anchor_ids.
            source_kind 为 'interface_ep'、scale == iep.scale、w_t = evidence_anchor_ids 的 Transition。

        State effect / 状态影响：
            None.
            无。
        """
        kappa = iep.slots.kappa
        return cls(
            source_kind=TransitionKind.interface_ep,
            source_id=iep.interface_ep_id,
            scale=iep.scale,
            s_t={
                "u": iep.slots.u,
                "b": iep.slots.b,
                "kappa": kappa.model_dump(),
                "interface_ep_id": iep.interface_ep_id,
                "support_node_ids": list(iep.support_node_ids),
                "valid_from": iep.created_at,
                "valid_to": iep.valid_to,
            },
            a_t=iep.slots.a,
            s_t1={
                "y": iep.slots.y,
                "r": iep.slots.r,
                "interface_ep_id": iep.interface_ep_id,
                "invariants": [inv.text for inv in iep.invariants],
                "residuals": [res.text for res in iep.residuals],
            },
            r_t=iep.slots.r,
            kappa_t=kappa_t_of(
                constraints=kappa.constraints,
                visibility=kappa.visibility,
                permission_scope=permission_scope,
                provenance={
                    "evidence": list(iep.evidence_anchor_ids),
                    "interface_ep_id": iep.interface_ep_id,
                    "cohesion_score": iep.cohesion_score,
                },
            ),
            w_t=list(iep.evidence_anchor_ids),
            time_t=time_t,
            time_t1=time_t1,
            scope=scope,
        )


# ---------------------------------------------------------------------------
# Payload normalization (§4.3) — the single implementation
# 载荷归一化（§4.3）—— 唯一实现
# ---------------------------------------------------------------------------
def _payload_error(
    object_type: str, field: str, index: int | None = None, message: str | None = None
) -> ValidationError:
    """Build the VALIDATION_005 error for a malformed payload field.
    为畸形载荷字段构造 VALIDATION_005 错误。
    """
    details: dict[str, Any] = {"object_type": object_type, "field": field}
    if index is not None:
        details["index"] = index
    return ValidationError(
        message or f"malformed {object_type} payload: field {field!r}", "VALIDATION_005", details=details
    )


def _req_str(payload: dict, object_type: str, field: str) -> str:
    """Required string field.
    必需的字符串字段。
    """
    value = payload.get(field)
    if not isinstance(value, str):
        raise _payload_error(object_type, field, message=f"{object_type} payload requires string field {field!r}")
    return value


def _opt_str(payload: dict, object_type: str, field: str) -> str | None:
    """Optional string field (None when absent).
    可选的字符串字段（缺省为 None）。
    """
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _payload_error(object_type, field)
    return value


def _opt_int(payload: dict, object_type: str, field: str) -> int | None:
    """Optional integer field (bool rejected).
    可选的整数字段（拒绝布尔值）。
    """
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _payload_error(object_type, field)
    return value


def _str_list(payload: dict, object_type: str, field: str) -> list[str]:
    """Optional list of strings (default []).
    可选的字符串列表（默认 []）。
    """
    value = payload.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _payload_error(object_type, field)
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _payload_error(object_type, field, index)
    return list(value)


def _literal(
    payload: dict, object_type: str, field: str, allowed: tuple[str, ...], *, required: bool = False
) -> str | None:
    """Optional/required literal field.
    可选/必需的字面量字段。
    """
    value = payload.get(field)
    if value is None and not required:
        return None
    if value not in allowed:
        raise _payload_error(
            object_type, field, message=f"{object_type} payload field {field!r} must be one of {list(allowed)}"
        )
    return value


MESSAGE_PART_SOURCES: tuple[str, ...] = ("data", "url")


def _message_parts(payload: dict, object_type: str) -> list[dict]:
    """Normalize message `parts`: [{'kind', 'source', 'sha256', 'mime', 'url', 'detail'}]; a data: URL is rejected.
    归一化消息 `parts`：[{'kind', 'source', 'sha256', 'mime', 'url', 'detail'}]；data: URL 会被拒绝。
    """
    value = payload.get("parts")
    if value is None:
        return []
    if not isinstance(value, list):
        raise _payload_error(object_type, "parts")
    out: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _payload_error(object_type, "parts", index)
        kind = item.get("kind")
        sha256 = item.get("sha256")
        source = item.get("source")
        if not isinstance(kind, str) or not kind or not isinstance(sha256, str) or not sha256:
            raise _payload_error(object_type, "parts", index)
        if source not in MESSAGE_PART_SOURCES:
            raise _payload_error(
                object_type, "parts", index, message=f"{object_type} part source must be one of {list(MESSAGE_PART_SOURCES)}"
            )
        url = item.get("url")
        if url is not None and not isinstance(url, str):
            raise _payload_error(object_type, "parts", index)
        # The anchored payload carries the sha256 of the bytes, never the bytes: a data: URI here would put the
        # base64 body into source_events and anchors.
        # 落锚载荷只带字节的 sha256，绝不带字节：此处若出现 data: URI，base64 正文就会进入 source_events 与 anchors。
        if isinstance(url, str) and url[:5].lower() == "data:":
            raise _payload_error(
                object_type, "parts", index, message=f"{object_type} part url may not be a data: URI (store its sha256)"
            )
        mime = item.get("mime")
        detail = item.get("detail")
        if (mime is not None and not isinstance(mime, str)) or (detail is not None and not isinstance(detail, str)):
            raise _payload_error(object_type, "parts", index)
        out.append({"kind": kind, "source": source, "sha256": sha256, "mime": mime, "url": url, "detail": detail})
    return out


def _described_parts(payload: dict, object_type: str) -> list[dict]:
    """Normalize message `described_parts`: [{'index', 'sha256', 'text', 'described_by'}].
    归一化消息 `described_parts`：[{'index', 'sha256', 'text', 'described_by'}]。
    """
    value = payload.get("described_parts")
    if value is None:
        return []
    if not isinstance(value, list):
        raise _payload_error(object_type, "described_parts")
    out: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _payload_error(object_type, "described_parts", index)
        part_index = item.get("index")
        sha256 = item.get("sha256")
        text = item.get("text")
        described_by = item.get("described_by")
        if isinstance(part_index, bool) or not isinstance(part_index, int) or part_index < 0:
            raise _payload_error(object_type, "described_parts", index)
        if not isinstance(sha256, str) or not sha256 or not isinstance(text, str) or not text:
            raise _payload_error(object_type, "described_parts", index)
        if described_by is not None and not isinstance(described_by, str):
            raise _payload_error(object_type, "described_parts", index)
        out.append({"index": part_index, "sha256": sha256, "text": text, "described_by": described_by})
    return out


def validate_payload(object_type: str, payload: dict) -> dict:
    """Normalize a SourceEvent payload to the exact §4.3 shape for its object_type (the SINGLE implementation).
    将 SourceEvent 载荷归一化为其 object_type 对应的 §4.3 精确形状（唯一实现）。

    Input / 输入：
        object_type: An ObjectType value.
        object_type：ObjectType 的值。

        payload: Caller-supplied dict (aliases 'arguments' → 'args', 'failed_tests' → 'failures' accepted).
        payload：调用方提供的字典（接受别名 'arguments' → 'args'、'failed_tests' → 'failures'）。

    Output / 输出：
        New dict with every documented key present in the documented order; extra keys dropped.
        新字典，按文档顺序包含全部键；多余键被丢弃。

    State effect / 状态影响：
        None; malformed input raises ValidationError(VALIDATION_005, details={'object_type', 'field', 'index'?}).
        无；畸形输入抛出 ValidationError(VALIDATION_005, details={'object_type', 'field', 'index'?})。
    """
    object_type = str(object_type)
    if not isinstance(payload, dict):
        raise _payload_error(object_type, "payload", message=f"{object_type} payload must be a dict")
    if object_type == ObjectType.message.value:
        text = _req_str(payload, object_type, "text")
        dropped = payload.get("dropped_parts", 0)
        if isinstance(dropped, bool) or not isinstance(dropped, int):
            raise _payload_error(object_type, "dropped_parts")
        normalized: dict[str, Any] = {"text": text, "dropped_parts": dropped}
        # Multipart messages (§4.3): `parts` records the identity of every non-text part (sha256 + mime, never the
        # bytes) and `described_parts` the vision descriptions merged into `text`; both keys exist only when
        # non-empty so text-only payloads keep their historical shape and hash.
        # 多部件消息（§4.3）：`parts` 记录每个非文本部件的身份（sha256 + mime，绝不含字节），
        # `described_parts` 记录并入 `text` 的视觉描述；两键仅在非空时存在，纯文本载荷保持历史形状与哈希。
        parts = _message_parts(payload, object_type)
        if parts:
            normalized["parts"] = parts
        described = _described_parts(payload, object_type)
        if described:
            normalized["described_parts"] = described
        return normalized
    if object_type == ObjectType.tool_call.value:
        tool = _req_str(payload, object_type, "tool")
        args = payload.get("args", payload.get("arguments"))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise _payload_error(object_type, "args")
        result = payload.get("result")
        if result is not None and not isinstance(result, str):
            # 非字符串结果 → JSON 文本
            result = json.dumps(result, ensure_ascii=False)  # non-str result → JSON text
        return {
            "tool": tool,
            "args": dict(args),
            "result": result,
            "status": _literal(payload, object_type, "status", ("ok", "error")),
            "target": _opt_str(payload, object_type, "target"),
            "intent": _opt_str(payload, object_type, "intent"),
            "constraints": _str_list(payload, object_type, "constraints"),
        }
    if object_type == ObjectType.file_diff.value:
        return {
            "path": _req_str(payload, object_type, "path"),
            "patch": _req_str(payload, object_type, "patch"),
            "summary": _opt_str(payload, object_type, "summary"),
            "intent": _opt_str(payload, object_type, "intent"),
            "commit": _opt_str(payload, object_type, "commit"),
            "constraints": _str_list(payload, object_type, "constraints"),
        }
    if object_type == ObjectType.test_log.value:
        suite = _req_str(payload, object_type, "suite")
        raw_failures = payload.get("failures", payload.get("failed_tests"))
        failures: list[dict[str, Any]] = []
        if raw_failures is not None:
            if not isinstance(raw_failures, list):
                raise _payload_error(object_type, "failures")
            for index, item in enumerate(raw_failures):
                if isinstance(item, str):
                    failures.append({"name": item, "message": None})
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    message = item.get("message")
                    if message is not None and not isinstance(message, str):
                        raise _payload_error(object_type, "failures", index)
                    failures.append({"name": item["name"], "message": message})
                else:
                    raise _payload_error(object_type, "failures", index)
        passed = _opt_int(payload, object_type, "passed")
        failed = _opt_int(payload, object_type, "failed")
        status = _literal(payload, object_type, "status", ("passed", "failed"))
        if status is None:
            status = "failed" if ((failed or 0) > 0 or failures) else "passed"
        return {
            "suite": suite,
            "status": status,
            "passed": passed,
            "failed": failed,
            "failures": failures,
            "log": _opt_str(payload, object_type, "log"),
            "related_paths": _str_list(payload, object_type, "related_paths"),
        }
    if object_type == ObjectType.sensor_trace.value:
        sensor = _req_str(payload, object_type, "sensor")
        raw_samples = payload.get("samples")
        samples: list[dict[str, Any]] = []
        if raw_samples is not None:
            if not isinstance(raw_samples, list):
                raise _payload_error(object_type, "samples")
            for index, item in enumerate(raw_samples):
                if not isinstance(item, dict) or "t" not in item:
                    raise _payload_error(object_type, "samples", index)
                try:
                    instant = to_utc_iso(item["t"])
                except ValidationError as exc:
                    raise _payload_error(object_type, "samples", index) from exc
                samples.append({"t": instant, "v": item.get("v")})
        return {
            "sensor": sensor,
            "series_ref": _opt_str(payload, object_type, "series_ref"),
            "summary": _opt_str(payload, object_type, "summary"),
            "samples": samples,
        }
    if object_type == ObjectType.feedback.value:
        return {
            "target_id": _opt_str(payload, object_type, "target_id"),
            "feedback": _req_str(payload, object_type, "feedback"),
            "reason": _opt_str(payload, object_type, "reason"),
            "correction": payload.get("correction"),
        }
    if object_type == ObjectType.checkpoint.value:
        return {"label": _opt_str(payload, object_type, "label")}
    if object_type == ObjectType.rollback.value:
        return {
            "checkpoint_id": _req_str(payload, object_type, "checkpoint_id"),
            "label": _opt_str(payload, object_type, "label"),
            "reason": _opt_str(payload, object_type, "reason"),
        }
    if object_type == ObjectType.system.value:
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise _payload_error(object_type, "metadata")
        return {
            "op": _literal(
                payload, object_type, "op", ("update", "delete", "delete_all", "import", "revoke", "purge"), required=True
            ),
            "memory_id": _opt_str(payload, object_type, "memory_id"),
            "atom_id": _opt_str(payload, object_type, "atom_id"),
            "text": _opt_str(payload, object_type, "text"),
            "metadata": dict(metadata) if metadata is not None else None,
            "expiration_date": _opt_str(payload, object_type, "expiration_date"),
            "package_id": _opt_str(payload, object_type, "package_id"),
            "anchor_id": _opt_str(payload, object_type, "anchor_id"),
        }
    if object_type == ObjectType.scheduler.value:
        phases = payload.get("phases")
        if not isinstance(phases, list):
            raise _payload_error(object_type, "phases")
        for index, item in enumerate(phases):
            if not isinstance(item, str):
                raise _payload_error(object_type, "phases", index)
        return {"phases": list(phases)}
    raise _payload_error(object_type, "object_type", message=f"unknown object_type {object_type!r}")


def kappa_t_of(*, constraints: list[str], visibility: str, permission_scope: str, provenance: dict) -> dict:
    """Uniform κ_t shape for every transition kind (§16.4).
    每种转移的统一 κ_t 形状（§16.4）。

    Input / 输入：
        constraints: Constraint texts in force.
        constraints：生效的约束文本。

        visibility: Visibility of the state.
        visibility：状态的可见性。

        permission_scope: Permission scope of the witnessing anchor.
        permission_scope：见证锚点的权限范围。

        provenance: Kind-specific provenance keys.
        provenance：与种类相关的出处键。

    Output / 输出：
        {'constraints', 'visibility', 'permission_scope', 'provenance'} in this key order.
        按此键序的 {'constraints', 'visibility', 'permission_scope', 'provenance'}。

    State effect / 状态影响：
        None.
        无。
    """
    return {
        "constraints": list(constraints),
        "visibility": str(visibility),
        "permission_scope": str(permission_scope),
        "provenance": dict(provenance),
    }


__all__ = [
    "MESSAGE_PART_SOURCES",
    "RESERVED_ATOM_KEYS",
    "Anchor",
    "AtomHit",
    "AxisQuery",
    "AxisResult",
    "BoundaryEdge",
    "Checkpoint",
    "ContextIR",
    "ContextItem",
    "EP",
    "FeedbackEvent",
    "GraphEdge",
    "GraphNode",
    "HMSModel",
    "InterfaceEP",
    "Invariant",
    "IsoDate",
    "Kappa",
    "MemoryAtom",
    "NonEmptyIds",
    "Observation",
    "PCIR",
    "Prob",
    "ProofObligation",
    "Residual",
    "Scope",
    "Slots",
    "SourceEvent",
    "StatePackage",
    "Transition",
    "UtcIso",
    "Witnessed",
    "event_payload_hash",
    "kappa_t_of",
    "validate_payload",
]
