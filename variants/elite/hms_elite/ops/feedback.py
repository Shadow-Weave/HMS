"""StateFeedback: eleven witnessed correction handlers plus anchor revocation with derived-object cascade.
状态反馈：十一种带见证的纠偏处理器，以及带派生对象级联的锚点撤销。
"""

# Purpose (EN): Turn user/agent corrections into state updates (§16.1 table) and implement HMS §3.2 Revoke.
# 目的（中）：把用户/Agent 的纠偏变成状态更新（§16.1 表），并实现 HMS §3.2 撤销。

# Input (EN): A target id (memory/atom/edge/InterfaceEP/context item/anchor/scope key), a FeedbackType, reason, correction.
# 输入（中）：目标 id（memory/atom/边/InterfaceEP/上下文项/锚点/范围键）、FeedbackType、原因、纠正内容。

# Output (EN): {'message', 'feedback_id', 'actions': [...]} or the revoke report.
# 输出（中）：{'message', 'feedback_id', 'actions': [...]} 或撤销报告。

# State effect (EN): One transaction per call: feedback event + anchor, feedback_events, history, then archive/create/
#                    revise/dispute/visibility/confidence/edge changes — never deletes.
# 状态影响（中）：每次调用一个事务：反馈事件 + 锚点、feedback_events、历史，然后归档/创建/修订/争议/可见性/置信度/
#                边变更——绝不删除。

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import (
    AtomType,
    EdgeType,
    FeedbackType,
    HistoryEvent,
    ObservationType,
    Role,
    Scale,
    Sentiment,
    SourceKind,
    Visibility,
)
from hms_elite.exceptions import ValidationError
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.ingest.atoms import create_atom, revise_atom
from hms_elite.ingest.events import system_event
from hms_elite.memory.storage import StateStore
from hms_elite.models import Anchor, FeedbackEvent, GraphEdge, MemoryAtom, Scope
from hms_elite.ops.transitions import revision_transition
from hms_elite.timeutil import date_of, to_utc_iso
from hms_elite.utils.text import ordered_union

logger = logging.getLogger(__name__)

_EDGE_ONLY = frozenset({FeedbackType.relationship_not_useful.value, FeedbackType.connection_reason_wrong.value})
_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


@dataclass
class _Target:
    """A resolved feedback target.
    已解析的反馈目标。
    """

    kind: str
    id: str | None
    scope: Scope
    atom: MemoryAtom | None = None
    edge: GraphEdge | None = None
    anchor: Anchor | None = None


def _clamp(value: float) -> float:
    """Confidence clamped to [0, 1] and rounded to 4 decimals.
    置信度限制在 [0, 1] 并保留 4 位小数。
    """
    return max(0.0, min(1.0, round(value, 4)))


def _unescape(value: str) -> str:
    """Reverse the Scope.key() escaping.
    反转 Scope.key() 的转义。
    """
    return value.replace("%3D", "=").replace("%26", "&").replace("%25", "%")


def scope_from_key(key: str | None) -> Scope | None:
    """Parse a Scope.key() string (`user_id=alice&run_id=run-a`) back into a Scope.
    将 Scope.key() 字符串（`user_id=alice&run_id=run-a`）解析回 Scope。

    Input / 输入：
        key: The key string (None or malformed → None).
        key：键字符串（None 或格式错误 → None）。

    Output / 输出：
        A non-empty Scope, or None.
        非空的 Scope，或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if not key or "=" not in key:
        return None
    fields: dict[str, str] = {}
    for part in key.split("&"):
        if "=" not in part:
            return None
        name, value = part.split("=", 1)
        if name not in _SCOPE_KEYS or not value:
            return None
        fields[name] = _unescape(value)
    try:
        scope = Scope(**fields)
    except (ValidationError, ValueError):
        return None
    return None if scope.is_empty() else scope


def _scope_from_correction(correction: Any) -> Scope | None:
    """Scope carried by a dict correction ({'scope': {...}}), if any.
    字典型纠正中携带的范围（{'scope': {...}}），若有。
    """
    if not isinstance(correction, dict):
        return None
    raw = correction.get("scope")
    if isinstance(raw, Scope):
        return None if raw.is_empty() else raw
    if isinstance(raw, dict):
        try:
            scope = Scope(**{k: raw.get(k) for k in _SCOPE_KEYS if raw.get(k) is not None})
        except (ValidationError, ValueError):
            return None
        return None if scope.is_empty() else scope
    if isinstance(raw, str):
        return scope_from_key(raw)
    return None


def _correction_text(correction: Any) -> str | None:
    """The text of a str / {'text': ...} correction.
    str / {'text': ...} 型纠正的文本。
    """
    if isinstance(correction, str):
        return correction.strip() or None
    if isinstance(correction, dict):
        text = correction.get("text")
        return str(text).strip() or None if text is not None else None
    return None


def _episode_action(atom: MemoryAtom) -> str:
    """The `a` slot of an episode atom (its text otherwise).
    情节原子的 `a` 槽位（否则为其文本）。
    """
    slots = atom.content.get("slots") if isinstance(atom.content, dict) else None
    if isinstance(slots, dict) and slots.get("a"):
        return str(slots["a"])
    return atom.text


class FeedbackProcessor:
    """Applies one FeedbackType to a resolved target inside a single witnessed transaction.
    在单个带见证的事务内对已解析目标应用一种 FeedbackType。
    """

    def __init__(
        self,
        store: StateStore,
        index: Any,
        embedder: Any,
        graph: SupportGraph,
        config: Any,
        entity_extractor: EntityExtractorBase | None = None,
    ) -> None:
        """Bind the collaborators.
        绑定协作对象。

        Input / 输入：
            store: The state store.
            store：状态存储。

            index: The vector index (new atoms are embedded).
            index：向量索引（新原子会被嵌入）。

            embedder: The embedder.
            embedder：嵌入器。

            graph: The SupportGraph (edges).
            graph：SupportGraph（边）。

            config: HMSConfig.
            config：HMSConfig。

            entity_extractor: Entity extractor for correction atoms (the facade's configured one); None means rule.
            entity_extractor：纠正原子使用的实体抽取器（门面配置的那一个）；None 表示规则。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.store = store
        self.index = index
        self.embedder = embedder
        self.graph = graph
        self.config = config
        self.entity_extractor = entity_extractor

    # ------------------------------------------------------------------ resolution
    # ------------------------------------------------------------------ 目标解析
    def _resolve(self, target_id: str | None) -> _Target | None:
        """memory_id → atom_id → ep_id → obs_id → interface_ep_id → edge_id → anchor_id; '<atom_id>#…' → atom.
        memory_id → atom_id → ep_id → obs_id → interface_ep_id → edge_id → anchor_id；'<atom_id>#…' → 原子。
        """
        store = self.store
        tid = str(target_id).strip() if target_id is not None else ""
        if not tid:
            return None
        if "#" in tid:
            atom = store.get_atom(tid.split("#", 1)[0])
            if atom is not None:
                return _Target(kind="context_item", id=tid, scope=atom.scope, atom=atom)
            return None
        atom = store.resolve_head(tid) or store.latest_version(tid) or store.get_atom(tid)
        if atom is not None:
            return _Target(kind="atom", id=tid, scope=atom.scope, atom=atom)
        ep = store.get_ep(tid)
        if ep is not None and ep.atom_id:
            atom = store.get_atom(ep.atom_id)
            if atom is not None:
                return _Target(kind="atom", id=tid, scope=atom.scope, atom=atom)
        obs = store.get_observation(tid)
        if obs is not None and obs.atom_id:
            atom = store.get_atom(obs.atom_id)
            if atom is not None:
                return _Target(kind="atom", id=tid, scope=atom.scope, atom=atom)
        iep = store.get_interface_ep(tid)
        if iep is not None and iep.atom_id:
            atom = store.get_atom(iep.atom_id)
            if atom is not None:
                return _Target(kind="interface_ep", id=tid, scope=atom.scope, atom=atom)
        edge = store.get_edge(tid)
        if edge is not None:
            return _Target(kind="edge", id=tid, scope=edge.scope, edge=edge)
        anchor = store.get_anchor(tid)
        if anchor is not None:
            return _Target(kind="anchor", id=tid, scope=anchor.scope, anchor=anchor)
        return None

    # ------------------------------------------------------------------ public entry
    # ------------------------------------------------------------------ 公开入口
    def process(
        self,
        *,
        target_id: str | None,
        feedback: str,
        reason: str | None,
        correction: str | dict | None,
        actor_id: str | None,
        scope_hint: Scope | None,
    ) -> dict:
        """Apply one feedback to a target (§16.1 prologue + handler table).
        对目标应用一条反馈（§16.1 前奏 + 处理器表）。

        Input / 输入：
            target_id: memory_id | atom_id | ep_id | obs_id | interface_ep_id | edge_id | '<atom_id>#…' | anchor_id | scope key | None.
            target_id：memory_id | atom_id | ep_id | obs_id | interface_ep_id | edge_id | '<atom_id>#…' | anchor_id | 范围键 | None。

            feedback: FeedbackType value (unknown → VALIDATION_006).
            feedback：FeedbackType 值（未知 → VALIDATION_006）。

            reason: Free-text reason.
            reason：自由文本原因。

            correction: Correction payload (str, dict or None; semantics per feedback type).
            correction：纠正载荷（str、dict 或 None；语义随反馈类型而定）。

            actor_id: Who gave the feedback.
            actor_id：反馈提供者。

            scope_hint: Scope used by state_missing when neither correction nor target_id carries one.
            scope_hint：state_missing 在 correction 与 target_id 都不含范围时使用的范围。

        Output / 输出：
            {'message': 'Feedback applied', 'feedback_id', 'actions': [{'action': ..., ...ids}]}.
            {'message': 'Feedback applied', 'feedback_id', 'actions': [{'action': ..., ...ids}]}。

        State effect / 状态影响：
            One transaction: feedback event + anchor, history FEEDBACK (atom targets), the handler's writes, feedback_events.
            一个事务：反馈事件 + 锚点、历史 FEEDBACK（原子目标）、处理器写入、feedback_events。
        """
        store = self.store
        try:
            ftype = FeedbackType(str(getattr(feedback, "value", feedback)))
        except ValueError as exc:
            raise ValidationError(
                f"unknown feedback type {feedback!r}", "VALIDATION_006", details={"allowed": FeedbackType.values()}
            ) from exc
        kind = ftype.value
        target = self._resolve(target_id)
        if kind == FeedbackType.state_missing.value:
            text = _correction_text(correction)
            if text is None:
                raise ValidationError(
                    "state_missing requires a correction with text",
                    "VALIDATION_007",
                    details={"feedback": kind, "target_id": target_id},
                )
            if target is not None and target.kind in ("edge", "anchor"):
                raise ValidationError(
                    "state_missing targets an atom, an episode or a scope",
                    "VALIDATION_007",
                    details={"feedback": kind, "target_kind": target.kind},
                )
            if target is None:
                scope = _scope_from_correction(correction) or scope_from_key(target_id)
                if scope is None and scope_hint is not None and not scope_hint.is_empty():
                    scope = scope_hint
                if scope is None:
                    raise ValidationError(
                        "state_missing without a target needs a scope (correction['scope'], a scope key or filters)",
                        "VALIDATION_001",
                        details={"feedback": kind, "target_id": target_id, "context": "feedback"},
                    )
                target = _Target(kind="scope", id=None, scope=scope)
        elif target is None:
            raise ValidationError(
                f"unknown feedback target {target_id!r}", "NOT_FOUND_001", details={"target_id": target_id}
            )
        if target.kind == "anchor":
            wants_revoke = isinstance(correction, dict) and bool(correction.get("revoke"))
            if kind != FeedbackType.evidence_wrong.value or not wants_revoke:
                raise ValidationError(
                    "an anchor target is accepted only by evidence_wrong with correction={'revoke': True}",
                    "VALIDATION_007",
                    details={"feedback": kind, "target_id": target_id},
                )
        if kind in _EDGE_ONLY and target.kind != "edge":
            raise ValidationError(
                f"{kind} targets an edge", "VALIDATION_007", details={"feedback": kind, "target_kind": target.kind}
            )
        if kind not in _EDGE_ONLY and target.kind == "edge":
            raise ValidationError(
                f"{kind} does not accept an edge target",
                "VALIDATION_007",
                details={"feedback": kind, "target_kind": target.kind},
            )
        scope = target.scope
        stored_correction: str | dict | None
        if isinstance(correction, (list, tuple)):
            # FeedbackEvent 只存 str | dict
            stored_correction = {"anchor_ids": [str(a) for a in correction]}  # FeedbackEvent stores str | dict
        else:
            stored_correction = correction
        with store.transaction():
            now = store.now()
            _ev, fb_anchor = system_event(
                store,
                scope=scope,
                object_type="feedback",
                event_type="feedback",
                payload={"target_id": target_id, "feedback": kind, "reason": reason, "correction": stored_correction},
                actor_id=actor_id,
            )
            feedback_id = store.new_id("fb")
            if target.atom is not None:
                store.add_history(
                    memory_id=target.atom.memory_id,
                    old_memory=target.atom.text,
                    new_memory=None,
                    event=HistoryEvent.FEEDBACK.value,
                    created_at=target.atom.created_at or now,
                    updated_at=now,
                    actor_id=actor_id,
                    anchor_id=fb_anchor.anchor_id,
                    atom_id=target.atom.atom_id,
                    details={"feedback": kind, "reason": reason},
                )
            ctx = _Context(
                target=target,
                fb=fb_anchor,
                now=now,
                feedback_id=feedback_id,
                reason=reason,
                correction=correction,
                actor_id=actor_id,
                target_id=target_id,
            )
            actions = self._dispatch(kind, ctx)
            store.insert_feedback(
                FeedbackEvent(
                    feedback_id=feedback_id,
                    target_id=target.id,
                    target_kind=target.kind,
                    feedback=ftype,
                    reason=reason,
                    correction=stored_correction,
                    actor_id=actor_id,
                    anchor_id=fb_anchor.anchor_id,
                    actions=actions,
                    scope=scope,
                    created_at=now,
                )
            )
        return {"message": "Feedback applied", "feedback_id": feedback_id, "actions": actions}

    # ------------------------------------------------------------------ dispatch
    # ------------------------------------------------------------------ 分派
    def _dispatch(self, kind: str, ctx: "_Context") -> list[dict]:
        """Route to the handler of a FeedbackType.
        按 FeedbackType 分派到处理器。
        """
        handlers = {
            FeedbackType.state_wrong.value: self._state_wrong,
            FeedbackType.state_outdated.value: self._state_outdated,
            FeedbackType.state_missing.value: self._state_missing,
            FeedbackType.privacy_violation.value: self._privacy_violation,
            FeedbackType.next_action_wrong.value: self._next_action_wrong,
            FeedbackType.relationship_not_useful.value: self._relationship_not_useful,
            FeedbackType.connection_reason_wrong.value: self._connection_reason_wrong,
            FeedbackType.evidence_wrong.value: self._evidence_wrong,
            FeedbackType.render_not_like_me.value: self._render_not_like_me,
            FeedbackType.render_like_me.value: self._render_like_me,
            FeedbackType.object_important.value: self._object_important,
        }
        return handlers[kind](ctx)

    # ------------------------------------------------------------------ shared primitives
    # ------------------------------------------------------------------ 共享原语
    def _set_confidence(self, atom: MemoryAtom, delta: float, now: str) -> dict:
        """Shift an atom's confidence by delta (clamped) and report it.
        按 delta 调整原子置信度（有界）并报告。
        """
        old = float(atom.confidence)
        new = _clamp(old + delta)
        if new != old:
            self.store.set_atom_confidence(atom.atom_id, new, at=now)
        return {"action": "set_confidence", "atom_id": atom.atom_id, "confidence": [old, new]}

    def _feedback_atom(
        self,
        ctx: "_Context",
        *,
        text: str,
        obs_type: str,
        subject: str,
        predicate: str,
        objects: list[str],
        topics: list[str],
        anchor_ids: list[str],
        sentiment: str | None = None,
    ) -> MemoryAtom:
        """Build a §4.2 feedback-shaped state atom (new lineage).
        构造 §4.2 反馈型 state 原子（新谱系）。
        """
        return MemoryAtom(
            atom_type=AtomType.state,
            content={
                "text": text,
                "obs_id": ctx.feedback_id,
                "obs_type": obs_type,
                "subject": subject,
                "predicate": predicate,
                "objects": list(objects),
                "relation": "single_fact",
                "date_observed": date_of(ctx.now),
            },
            text=text,
            time_anchor=ctx.now,
            valid_from=ctx.now,
            topic_tags=list(topics),
            scale_level=Scale.state,
            role=Role.extracted,
            sentiment=sentiment,
            subject=subject,
            predicate=predicate,
            obs_type=obs_type,
            source_kind=SourceKind.feedback,
            source_id=ctx.feedback_id,
            confidence=0.8,
            extracted_by="feedback@v1",
            visibility=ctx.target.atom.visibility if ctx.target.atom is not None else Visibility.private,
            scope=ctx.target.scope,
            actor_id=ctx.actor_id,
            anchor_ids=list(anchor_ids),
        )

    def _successor(self, old: MemoryAtom, text: str, ctx: "_Context", *, keep_topics: bool) -> MemoryAtom:
        """The successor version of `old` carrying the corrected text (coordinates copied).
        携带纠正文本的 `old` 后继版本（坐标复制）。
        """
        content = dict(old.content) if isinstance(old.content, dict) else {}
        content["text"] = text
        return old.model_copy(
            update={
                "atom_id": "",
                "memory_id": "",
                "content": content,
                "text": text,
                "hash": "",
                "time_anchor": ctx.now,
                "valid_from": ctx.now,
                "valid_to": None,
                "entity_ids": [],
                "topic_tags": list(old.topic_tags) if keep_topics else [],
                "anchor_ids": ordered_union([ctx.fb.anchor_id], list(old.anchor_ids)),
                "archived_reason": None,
                "created_at": None,
                "updated_at": None,
            }
        )

    def _revision_edge_id(self, old_id: str, new_id: str) -> str | None:
        """The revision edge written by revise_atom; its 'revision' transition row is written now (§16.4 c).
        revise_atom 写入的 revision 边；其 'revision' 转移行此刻写入（§16.4 c）。
        """
        edge = self.store.find_edge(old_id, new_id, EdgeType.revision.value, "L1_micro")
        if edge is None:
            return None
        row = revision_transition(self.store, edge)
        if row is not None:
            self.store.upsert_transition(row)
        return edge.edge_id

    # ------------------------------------------------------------------ handlers
    # ------------------------------------------------------------------ 处理器
    def _state_wrong(self, ctx: "_Context") -> list[dict]:
        """archive head; successor (conf 0.95 edge) when corrected; resolve conflicts; invalidate outgoing use/distill.
        归档头版本；有纠正时创建后继（0.95 边）；解决冲突；使外向 use/distill 边失效。
        """
        store, old, now = self.store, ctx.target.atom, ctx.now
        actions: list[dict] = []
        text = _correction_text(ctx.correction)
        revision_edge = None
        if text is not None and text != old.text:
            new = revise_atom(
                store,
                self.index,
                self.embedder,
                old=old,
                new=self._successor(old, text, ctx, keep_topics=True),
                witness_anchor_ids=ordered_union([ctx.fb.anchor_id], list(old.anchor_ids)),
                at=now,
                reason="feedback",
                created_by="feedback",
                edge_confidence=0.95,
                extractor=self.entity_extractor,
            )
            revision_edge = self._revision_edge_id(old.atom_id, new.atom_id)
            actions.append({"action": "archive", "atom_id": old.atom_id})
            actions.append({"action": "create", "atom_id": new.atom_id, "memory_id": new.memory_id})
            actions.append({"action": "revision_edge", "edge_id": revision_edge})
        else:
            if old.archived_reason is None:
                store.archive_atom(old.atom_id, reason="feedback", at=now)
            actions.append({"action": "archive", "atom_id": old.atom_id})
        for edge in store.edges_touching([old.atom_id], valid_only=True, edge_types=[EdgeType.conflict.value]):
            if edge.resolved_at is None:
                store.resolve_conflict(
                    edge.edge_id, at=now, by_anchor_id=ctx.fb.anchor_id, resolution_edge_id=revision_edge
                )
                actions.append({"action": "resolve_conflict", "edge_id": edge.edge_id})
        outgoing = store.edges_touching(
            [old.atom_id], valid_only=True, edge_types=[EdgeType.use.value, EdgeType.distill.value], direction="out"
        )
        for edge in outgoing:
            store.invalidate_edge(edge.edge_id, reason="source_archived", at=now)
            actions.append({"action": "invalidate_edge", "edge_id": edge.edge_id, "reason": "source_archived"})
        return actions

    def _state_outdated(self, ctx: "_Context") -> list[dict]:
        """archive head (valid_to from the correction or now); successor valid from now when a text is given.
        归档头版本（valid_to 来自纠正或当前时刻）；给出文本时创建从现在起有效的后继。
        """
        store, old, now = self.store, ctx.target.atom, ctx.now
        actions: list[dict] = []
        explicit_valid_to = None
        if isinstance(ctx.correction, dict) and ctx.correction.get("valid_to"):
            explicit_valid_to = to_utc_iso(ctx.correction["valid_to"])
            if old.valid_from is not None and explicit_valid_to < old.valid_from:
                explicit_valid_to = old.valid_from
        text = _correction_text(ctx.correction)
        if text is not None and text != old.text:
            new = revise_atom(
                store,
                self.index,
                self.embedder,
                old=old,
                new=self._successor(old, text, ctx, keep_topics=False),
                witness_anchor_ids=ordered_union([ctx.fb.anchor_id], list(old.anchor_ids)),
                at=now,
                reason="feedback",
                created_by="feedback",
                extractor=self.entity_extractor,
            )
            if explicit_valid_to is not None and explicit_valid_to != now:
                store.set_valid_to(old.atom_id, explicit_valid_to, at=now)
            actions.append({"action": "archive", "atom_id": old.atom_id})
            actions.append({"action": "create", "atom_id": new.atom_id, "memory_id": new.memory_id})
            actions.append({"action": "revision_edge", "edge_id": self._revision_edge_id(old.atom_id, new.atom_id)})
        else:
            if old.archived_reason is None:
                store.archive_atom(old.atom_id, reason="feedback", at=now, valid_to=explicit_valid_to or now)
            actions.append({"action": "archive", "atom_id": old.atom_id})
        return actions

    def _state_missing(self, ctx: "_Context") -> list[dict]:
        """create a new state lineage from the correction; a use edge when the target is an episode.
        由纠正创建新的 state 谱系；目标为情节时添加 use 边。
        """
        text = _correction_text(ctx.correction) or ""
        obs_type = ObservationType.fact.value
        if isinstance(ctx.correction, dict) and ctx.correction.get("obs_type"):
            obs_type = str(ctx.correction["obs_type"])
            if obs_type not in ObservationType.values():
                raise ValidationError(
                    f"unknown obs_type {obs_type!r}",
                    "VALIDATION_013",
                    details={"obs_type": obs_type, "supported": ObservationType.values()},
                )
        target_atom = ctx.target.atom
        anchors = ordered_union([ctx.fb.anchor_id], list(target_atom.anchor_ids) if target_atom is not None else [])
        atom = self._feedback_atom(
            ctx,
            text=text,
            obs_type=obs_type,
            subject="user",
            predicate="states",
            objects=[],
            topics=[],
            anchor_ids=anchors,
        )
        stored = create_atom(
            self.store,
            self.index,
            self.embedder,
            atom,
            history_anchor_id=ctx.fb.anchor_id,
            actor_id=ctx.actor_id,
            history_details={"feedback": FeedbackType.state_missing.value, "feedback_id": ctx.feedback_id},
            extractor=self.entity_extractor,
        )
        actions: list[dict] = [{"action": "create", "atom_id": stored.atom_id, "memory_id": stored.memory_id}]
        if target_atom is not None and str(target_atom.atom_type) == AtomType.episode.value:
            edge_id = self.graph.insert_edge(
                src_id=stored.atom_id,
                dst_id=target_atom.atom_id,
                edge_type=EdgeType.use.value,
                witness_anchor_ids=[ctx.fb.anchor_id],
                created_by="feedback",
                proposer="rule",
                confidence=0.7,
                label="feedback: state_missing",
            )
            actions.append({"action": "insert_edge", "edge_id": edge_id})
        return actions

    def _privacy_violation(self, ctx: "_Context") -> list[dict]:
        """private on the head and every atom sharing its anchors; anchors' permission_scope private; edges untouched.
        头版本及共享其锚点的每个原子设为 private；锚点 permission_scope 设为 private；边不变。
        """
        store, head, now = self.store, ctx.target.atom, ctx.now
        anchors = store.atom_anchor_ids(head.atom_id, include_disputed=True) or list(head.anchor_ids)
        related: set[str] = set()
        for anchor_id in anchors:
            related.update(store.atoms_for_anchor(anchor_id))
        related.discard(head.atom_id)
        actions: list[dict] = []
        for atom_id in [head.atom_id] + sorted(related):
            store.set_atom_visibility(atom_id, Visibility.private.value, at=now)
            actions.append({"action": "set_visibility", "atom_id": atom_id, "visibility": Visibility.private.value})
        for anchor_id in sorted(set(anchors)):
            store.set_anchor_permission(anchor_id, Visibility.private.value)
            actions.append({"action": "set_visibility", "anchor_id": anchor_id, "visibility": Visibility.private.value})
        return actions

    def _next_action_wrong(self, ctx: "_Context") -> list[dict]:
        """constraint atom 'Do not: <a>' distilled from the episode; episode confidence −0.2.
        由情节蒸馏出约束原子 'Do not: <a>'；情节置信度 −0.2。
        """
        target = ctx.target.atom
        action_text = _episode_action(target)
        text = _correction_text(ctx.correction) or f"Do not: {action_text}"
        constraint = self._feedback_atom(
            ctx,
            text=text,
            obs_type=ObservationType.constraint.value,
            subject="project",
            predicate="must_not",
            objects=[action_text],
            topics=["next_action"],
            anchor_ids=ordered_union([ctx.fb.anchor_id], list(target.anchor_ids)),
            sentiment=Sentiment.negative.value,
        )
        stored = create_atom(
            self.store,
            self.index,
            self.embedder,
            constraint,
            history_anchor_id=ctx.fb.anchor_id,
            actor_id=ctx.actor_id,
            history_details={"feedback": FeedbackType.next_action_wrong.value, "feedback_id": ctx.feedback_id},
            extractor=self.entity_extractor,
        )
        actions: list[dict] = [{"action": "create", "atom_id": stored.atom_id, "memory_id": stored.memory_id}]
        actions.append(self._set_confidence(target, -0.2, ctx.now))
        edge_id = self.graph.insert_edge(
            src_id=target.atom_id,
            dst_id=stored.atom_id,
            edge_type=EdgeType.distill.value,
            witness_anchor_ids=[ctx.fb.anchor_id],
            created_by="feedback",
            proposer="rule",
            confidence=0.9,
            label="feedback: next_action_wrong",
        )
        actions.append({"action": "insert_edge", "edge_id": edge_id})
        return actions

    def _relationship_not_useful(self, ctx: "_Context") -> list[dict]:
        """invalidate the edge, then every projected edge left without a valid projection source (existence rule).
        使该边失效，再使不再剩任何有效投影来源的投影边失效（存在规则）。
        """
        store, edge, now = self.store, ctx.target.edge, ctx.now
        actions: list[dict] = []
        store.invalidate_edge(edge.edge_id, reason="feedback:not_useful", at=now)
        actions.append({"action": "invalidate_edge", "edge_id": edge.edge_id, "reason": "feedback:not_useful"})
        for high_id in _cascade_projections(store, [edge.edge_id], reason="feedback:not_useful", at=now):
            actions.append({"action": "invalidate_edge", "edge_id": high_id, "reason": "feedback:not_useful"})
        return actions

    def _connection_reason_wrong(self, ctx: "_Context") -> list[dict]:
        """same type → relabel + re-witness (merged); different type → invalidate and insert a fresh edge.
        类型相同 → 重贴标签并补见证（merged）；类型不同 → 失效并插入新边。
        """
        store, old, now = self.store, ctx.target.edge, ctx.now
        correction = ctx.correction
        if isinstance(correction, dict):
            corrected_type = str(correction.get("edge_type") or old.edge_type)
            label = old.label if correction.get("label") is None else str(correction["label"])
        elif isinstance(correction, str):
            corrected_type, label = str(old.edge_type), correction
        else:
            corrected_type, label = str(old.edge_type), old.label
        corrected_type = str(getattr(corrected_type, "value", corrected_type))
        if corrected_type not in EdgeType.values():
            raise ValidationError(
                f"unknown edge_type {corrected_type!r}",
                "VALIDATION_013",
                details={"edge_type": corrected_type, "supported": EdgeType.values()},
            )
        if corrected_type == str(old.edge_type):
            store.revalidate_edge(old.edge_id, at=now)
            store.add_edge_witness(old.edge_id, ctx.fb.anchor_id)
            store.set_edge_label(old.edge_id, label, at=now)
            return [{"action": "insert_edge", "edge_id": old.edge_id, "merged": True}]
        actions: list[dict] = []
        store.invalidate_edge(old.edge_id, reason="feedback:wrong_reason", at=now)
        actions.append({"action": "invalidate_edge", "edge_id": old.edge_id, "reason": "feedback:wrong_reason"})
        witnesses = ordered_union(store.edge_witnesses(old.edge_id), [ctx.fb.anchor_id])
        new_id = store.insert_edge(
            {
                "src_id": old.src_id,
                "dst_id": old.dst_id,
                "edge_type": corrected_type,
                "label": label,
                "scale": str(old.scale),
                "confidence": float(old.confidence),
                "witness_anchor_ids": witnesses,
                "created_by": "feedback",
                "proposer": "rule",
                "scope": old.scope,
            }
        )
        for high_id in store.projections_from(old.edge_id):
            if high_id != new_id:
                store.add_projection(high_id, new_id)
        actions.append({"action": "insert_edge", "edge_id": new_id, "merged": False})
        return actions

    def _evidence_wrong(self, ctx: "_Context") -> list[dict]:
        """dispute the listed anchors (default: all non-primary); archive when unwitnessed; confidence −0.3; or revoke.
        对列出的锚点（默认：全部非主锚点）标记争议；无见证时归档；置信度 −0.3；或撤销。
        """
        store, now = self.store, ctx.now
        if ctx.target.kind == "anchor":
            report = revoke(store, self.graph, anchor_id=ctx.target.id, reason=ctx.reason, actor_id=ctx.actor_id)
            return [{"action": "revoke", **report}]
        atom = ctx.target.atom
        linked = store.atom_anchor_ids(atom.atom_id, include_disputed=True) or list(atom.anchor_ids)
        correction = ctx.correction
        if isinstance(correction, str):
            chosen = [part.strip() for part in correction.split(",") if part.strip()]
        elif isinstance(correction, list):
            chosen = [str(part) for part in correction]
        elif isinstance(correction, dict) and correction.get("anchor_ids"):
            chosen = [str(part) for part in correction["anchor_ids"]]
        else:
            chosen = linked[1:]
        actions: list[dict] = []
        for anchor_id in ordered_union(chosen):
            if anchor_id not in linked:
                logger.warning("evidence_wrong: anchor %s is not a witness of %s; ignored", anchor_id, atom.atom_id)
                continue
            store.dispute_atom_anchor(atom.atom_id, anchor_id, at=now)
            actions.append({"action": "dispute_anchor", "atom_id": atom.atom_id, "anchor_id": anchor_id})
        if not store.atom_anchor_ids(atom.atom_id) and atom.archived_reason is None:
            store.archive_atom(atom.atom_id, reason="feedback:evidence_wrong", at=now)
            actions.append({"action": "archive", "atom_id": atom.atom_id})
        actions.append(self._set_confidence(atom, -0.3, now))
        return actions

    def _render_pref(self, ctx: "_Context", *, predicate: str, delta: float) -> list[dict]:
        """confidence ±0.1 on the target and a render_style preference atom.
        目标置信度 ±0.1，并创建 render_style 偏好原子。
        """
        target = ctx.target.atom
        verb = "likes" if predicate == "likes" else "dislikes"
        text = ctx.reason or f"User {verb} how '{target.text[:40]}' is rendered."
        actions: list[dict] = [self._set_confidence(target, delta, ctx.now)]
        pref = self._feedback_atom(
            ctx,
            text=text,
            obs_type=ObservationType.preference_event.value,
            subject="user",
            predicate=predicate,
            objects=[f"render:{ctx.target_id}"],
            topics=["render_style"],
            anchor_ids=[ctx.fb.anchor_id],
            sentiment=Sentiment.positive.value if predicate == "likes" else Sentiment.negative.value,
        )
        stored = create_atom(
            self.store,
            self.index,
            self.embedder,
            pref,
            history_anchor_id=ctx.fb.anchor_id,
            actor_id=ctx.actor_id,
            history_details={"feedback": f"render_{'like' if predicate == 'likes' else 'not_like'}_me"},
            extractor=self.entity_extractor,
        )
        actions.append({"action": "create", "atom_id": stored.atom_id, "memory_id": stored.memory_id})
        return actions

    def _render_not_like_me(self, ctx: "_Context") -> list[dict]:
        """confidence −0.1 and a 'dislikes' render preference.
        置信度 −0.1 并创建 'dislikes' 渲染偏好。
        """
        return self._render_pref(ctx, predicate="dislikes", delta=-0.1)

    def _render_like_me(self, ctx: "_Context") -> list[dict]:
        """confidence +0.1 and a 'likes' render preference.
        置信度 +0.1 并创建 'likes' 渲染偏好。
        """
        return self._render_pref(ctx, predicate="likes", delta=0.1)

    def _object_important(self, ctx: "_Context") -> list[dict]:
        """confidence +0.2, topic 'important' (pin), mentions +3.
        置信度 +0.2、主题 'important'（置顶）、mentions +3。
        """
        atom = ctx.target.atom
        actions = [self._set_confidence(atom, 0.2, ctx.now)]
        self.store.add_atom_topic(atom.atom_id, "important")
        actions.append({"action": "pin", "atom_id": atom.atom_id, "topic": "important"})
        self.store.bump_mentions(atom.atom_id, 3)
        return actions


@dataclass
class _Context:
    """Per-call handler context.
    单次调用的处理器上下文。
    """

    target: _Target
    fb: Anchor
    now: str
    feedback_id: str
    reason: str | None
    correction: Any
    actor_id: str | None
    target_id: str | None


def _cascade_projections(store: StateStore, fallen: list[str], *, reason: str, at: str) -> list[str]:
    """Invalidate every projected edge whose projection sources are ALL invalid, transitively (existence rule).
    传递地使投影来源**全部**失效的投影边失效（存在规则）。

    A projected edge (one with `edge_projections` rows) is the image of its low edges; it stays valid while at
    least one of them is valid, exactly as a witnessed object stays valid while at least one witness remains
    (docs/design/WITNESS_SEMANTICS.md). Edges that keep a valid source are left untouched and are re-examined
    whenever another of their sources falls later in the same worklist.
    投影边（有 `edge_projections` 行的边）是其低层边的像；只要还有一个低层边有效它就有效，正如带见证的对象只要还剩
    一个见证就有效（docs/design/WITNESS_SEMANTICS.md）。仍有有效来源的边不动，同一工作队列里其他来源倒下时再检查。

    Input / 输入：
        store: The state store.
        store：状态存储。

        fallen: Edge ids invalidated by the caller (the worklist seed).
        fallen：调用方刚使之失效的边 id（工作队列种子）。

        reason: `invalidated_reason` written on every edge this cascade invalidates.
        reason：本次级联写到每条失效边上的 `invalidated_reason`。

        at: Invalidation instant.
        at：失效时刻。

    Output / 输出：
        The projected edge ids invalidated here, in cascade order (never contains a seed id).
        本次失效的投影边 id，按级联顺序（不含种子 id）。

    State effect / 状态影响：
        invalidate_edge on each returned id.
        对返回的每个 id 调用 invalidate_edge。
    """
    invalidated: list[str] = []
    queue = list(fallen)
    while queue:
        low_id = queue.pop(0)
        for high_id in store.projections_from(low_id):
            high = store.get_edge(high_id)
            if high is None or not high.valid:
                continue
            sources = [store.get_edge(s) for s in store.projections_of(high_id)]
            if any(s is not None and s.valid for s in sources):
                continue  # 至少一个原像仍有效 / at least one preimage is still valid
            store.invalidate_edge(high_id, reason=reason, at=at)
            invalidated.append(high_id)
            queue.append(high_id)  # L2 → L3 chains / L2 → L3 链
    return invalidated


def revoke(
    store: StateStore, graph: SupportGraph, *, anchor_id: str, reason: str | None, actor_id: str | None = None
) -> dict:
    """Revoke an anchor and cascade under the existence rule (HMS §3.2): dispute → archive, edges, projections, IEPs.
    撤销锚点并按存在规则级联（HMS §3.2）：争议 → 归档、边、投影边、InterfaceEP。

    Every derived object stays valid while at least one undisputed, unrevoked witness remains (the primary witness
    `anchor_ids[0]` is not special); a projected edge additionally needs at least one valid projection source
    (docs/design/WITNESS_SEMANTICS.md).
    每个派生对象只要还剩任一未争议、未撤销的见证就保持有效（主见证 `anchor_ids[0]` 并不特殊）；投影边另需至少一个
    有效投影来源（docs/design/WITNESS_SEMANTICS.md）。

    Input / 输入：
        store: The state store.
        store：状态存储。

        graph: The SupportGraph (kept for parity with the other ops entry points).
        graph：SupportGraph（与其他运维入口保持一致）。

        anchor_id: The anchor to revoke (unknown → NOT_FOUND_001).
        anchor_id：要撤销的锚点（未知 → NOT_FOUND_001）。

        reason: Optional reason recorded on the system event.
        reason：记录在系统事件上的可选原因。

        actor_id: Optional actor of the system event.
        actor_id：系统事件的可选行动者。

    Output / 输出：
        {'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'} (+ 'noop' when already revoked).
        {'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'}（已撤销时另有 'noop'）。

    State effect / 状态影响：
        One transaction: anchors.revoked_at, atom_anchor disputes, archives + history, edge (`witness_revoked`),
        projected edge (`projection_revoked`) and InterfaceEP invalidations.
        一个事务：anchors.revoked_at、atom_anchor 争议、归档与历史、边（`witness_revoked`）、投影边
        （`projection_revoked`）与 InterfaceEP 失效。
    """
    anchor = store.get_anchor(anchor_id)
    if anchor is None:
        raise ValidationError(f"unknown anchor {anchor_id!r}", "NOT_FOUND_001", details={"anchor_id": anchor_id})
    if anchor.revoked_at is not None:
        return {
            "revoked": anchor_id,
            "archived": [],
            "edges_invalidated": [],
            "interface_eps_invalidated": [],
            "anchor_id": None,
            "noop": True,
        }
    archived: list[str] = []
    edges_invalidated: list[str] = []
    ieps_invalidated: list[str] = []
    with store.transaction():
        _ev, sys_anchor = system_event(
            store,
            scope=anchor.scope,
            object_type="system",
            event_type="updated",
            payload={"op": "revoke", "anchor_id": anchor_id, "text": reason},
            actor_id=actor_id,
        )
        now = store.now()
        store.revoke_anchor(anchor_id, at=now)
        for atom_id in store.atoms_for_anchor(anchor_id):
            store.dispute_atom_anchor(atom_id, anchor_id, at=now)
            atom = store.get_atom(atom_id)
            if atom is None or store.atom_anchor_ids(atom_id) or atom.archived_reason is not None:
                continue
            store.archive_atom(atom_id, reason="revoked", at=now)
            store.add_history(
                memory_id=atom.memory_id,
                old_memory=atom.text,
                new_memory=None,
                event=HistoryEvent.ARCHIVE.value,
                created_at=atom.created_at or now,
                updated_at=now,
                actor_id=actor_id,
                anchor_id=sys_anchor.anchor_id,
                atom_id=atom_id,
                details={"revoked_anchor": anchor_id},
            )
            archived.append(atom_id)
        everywhere = Scope()
        candidates = [e for e in store.list_edges(everywhere, valid_only=True) if anchor_id in e.witness_anchor_ids]
        for edge in sorted(candidates, key=lambda e: e.edge_id):
            witnesses = store.get_anchors(store.edge_witnesses(edge.edge_id))
            if witnesses and all(w.revoked_at is not None for w in witnesses):
                store.invalidate_edge(edge.edge_id, reason="witness_revoked", at=now)
                edges_invalidated.append(edge.edge_id)
        # 投影边：只有当它的每个投影来源都已失效时才倒下（自身见证可能仍然存活）
        # projected edges fall only once EVERY projection source is invalid (their own witnesses may survive)
        edges_invalidated.extend(
            _cascade_projections(store, list(edges_invalidated), reason="projection_revoked", at=now)
        )
        for iep in store.list_interface_eps(everywhere, valid_only=True):
            if anchor_id not in iep.evidence_anchor_ids:
                continue
            evidence = store.get_anchors(iep.evidence_anchor_ids)
            if evidence and all(a.revoked_at is not None for a in evidence):
                store.invalidate_interface_ep(iep.interface_ep_id, at=now)
                ieps_invalidated.append(iep.interface_ep_id)
    return {
        "revoked": anchor_id,
        "archived": archived,
        "edges_invalidated": edges_invalidated,
        "interface_eps_invalidated": ieps_invalidated,
        "anchor_id": sys_anchor.anchor_id,
    }


__all__ = ["FeedbackProcessor", "revoke", "scope_from_key"]
