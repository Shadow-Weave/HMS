"""Messages and structured dicts become SourceEvents; anchoring is Txn A of the ingest lock discipline.
消息与结构化字典转为 SourceEvent；落锚是摄取事务纪律中的事务 A。
"""

# Purpose (EN): Build SourceEvents from Mem0 messages / add_events dicts, derive raw_refs, and anchor them (Txn A).
# 目的（中）：由 Mem0 消息 / add_events 字典构造 SourceEvent，推导 raw_ref，并完成落锚（事务 A）。

# Input (EN): Coerced messages, event dicts or SourceEvents, a scope, an event time and a StateStore.
# 输入（中）：规整后的消息、事件字典或 SourceEvent、范围、事件时间与 StateStore。

# Output (EN): SourceEvents (sorted, normalized) and (event, anchor, dedup) triples.
# 输出（中）：SourceEvent（已排序、已归一化）以及 (event, anchor, dedup) 三元组。

# State effect (EN): anchor_events / system_event write source_events and anchors; everything else is pure.
# 状态影响（中）：anchor_events / system_event 写入 source_events 与 anchors；其余为纯函数。

from __future__ import annotations

from typing import Any

import pydantic

from hms_elite.enums import INTERNAL_OBJECT_TYPES, EventType, ObjectType, Role
from hms_elite.exceptions import ValidationError
from hms_elite.memory.storage import StateStore
from hms_elite.models import Anchor, Scope, SourceEvent, validate_payload
from hms_elite.timeutil import to_utc_iso

_STRUCTURED_TYPES = frozenset(
    {
        ObjectType.tool_call.value,
        ObjectType.file_diff.value,
        ObjectType.test_log.value,
        ObjectType.sensor_trace.value,
    }
)
_SCOPE_KEYS = ("user_id", "agent_id", "run_id")
# One line per described image part, appended after the message text (the description "joins the body").
# 每个已描述的图片部件一行，追加在消息文本之后（描述"并入正文"）。
IMAGE_LINE = "[Image: {description}]"


def _described(part: Any) -> str | None:
    """The stripped description of a coerced image part, or None when it was not described.
    规整后图片部件的去空白描述；未被描述时为 None。
    """
    if not isinstance(part, dict):
        return None
    description = part.get("description")
    if not isinstance(description, str) or not description.strip():
        return None
    return description.strip()


def merge_image_descriptions(message: dict) -> str:
    """The text of a coerced message with one IMAGE_LINE per described image part appended (unchanged when none).
    规整消息的文本，末尾追加每个已描述图片部件的一行 IMAGE_LINE（没有描述时原样返回）。

    Input / 输入：
        message: Coerced message (`content` text, optional `image_parts` with `description` set by the vision pass).
        message：规整后的消息（`content` 文本，可选的 `image_parts`，其 `description` 由视觉前置阶段写入）。

    Output / 输出：
        `content` when no part is described; else `content` + newline + the IMAGE_LINEs in part order (no leading newline when `content` is empty).
        无已描述部件时为 `content`；否则为 `content` + 换行 + 按部件顺序的 IMAGE_LINE（`content` 为空时无前导换行）。

    State effect / 状态影响：
        None.
        无。
    """
    content = message.get("content")
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    lines = [IMAGE_LINE.format(description=d) for d in map(_described, message.get("image_parts") or []) if d]
    if not lines:
        return text
    return "\n".join(([text] if text else []) + lines)


def _message_payload(message: dict, text: str) -> dict:
    """The normalized message payload: text, dropped_parts, and parts / described_parts from `image_parts`.
    归一化的消息载荷：text、dropped_parts，以及来自 `image_parts` 的 parts / described_parts。
    """
    dropped = message.get("dropped_parts", 0)
    if isinstance(dropped, bool) or not isinstance(dropped, int):
        dropped = 0
    parts: list[dict] = []
    described: list[dict] = []
    for index, part in enumerate(message.get("image_parts") or []):
        if not isinstance(part, dict) or not isinstance(part.get("sha256"), str):
            continue
        source = part.get("source") if part.get("source") in ("data", "url") else "url"
        # Only a remote URL is evidence worth keeping verbatim; a data URI is identified by its sha256 alone.
        # 只有远程 URL 值得原样保留；data URI 仅以其 sha256 标识。
        parts.append(
            {
                "kind": "image",
                "source": source,
                "sha256": part["sha256"],
                "mime": part.get("mime") if isinstance(part.get("mime"), str) else None,
                "url": part.get("url") if source == "url" and isinstance(part.get("url"), str) else None,
                "detail": part.get("detail") if isinstance(part.get("detail"), str) else None,
            }
        )
        description = _described(part)
        if description is not None:
            described_by = part.get("described_by")
            described.append(
                {
                    "index": len(parts) - 1,
                    "sha256": part["sha256"],
                    "text": description,
                    "described_by": described_by if isinstance(described_by, str) else None,
                }
            )
    payload: dict[str, Any] = {"text": text, "dropped_parts": max(0, dropped - len(described))}
    if parts:
        payload["parts"] = parts
    if described:
        payload["described_parts"] = described
    return payload


def default_raw_ref(ev: SourceEvent, index: int) -> str:
    """Derive the default source reference of an event from its object type and payload.
    根据对象类型与载荷推导事件的默认来源引用。

    Input / 输入：
        ev: The event (payload already normalized).
        ev：事件（载荷已归一化）。

        index: Position of the event within its add() / add_events() call (chat refs only).
        index：事件在本次 add() / add_events() 调用中的位置（仅聊天引用使用）。

    Output / 输出：
        'chat://<scope key>/<index>', 'tool://<tool>', 'file://<path>', 'test://<suite>', 'sensor://<sensor>' or 'hms://<type>/<timestamp>'.
        'chat://<范围键>/<序号>'、'tool://<工具>'、'file://<路径>'、'test://<套件>'、'sensor://<传感器>' 或 'hms://<类型>/<时间>'。

    State effect / 状态影响：
        None.
        无。
    """
    object_type = str(ev.object_type)
    payload = ev.payload or {}
    if object_type == ObjectType.message.value:
        return f"chat://{ev.scope.key()}/{index}"
    if object_type == ObjectType.tool_call.value:
        return f"tool://{payload.get('tool', '')}"
    if object_type == ObjectType.file_diff.value:
        return f"file://{payload.get('path', '')}"
    if object_type == ObjectType.test_log.value:
        return f"test://{payload.get('suite', '')}"
    if object_type == ObjectType.sensor_trace.value:
        return f"sensor://{payload.get('sensor', '')}"
    return f"hms://{object_type}/{ev.timestamp}"


def _with_raw_ref(ev: SourceEvent, index: int) -> SourceEvent:
    """Fill raw_ref with default_raw_ref when it is missing.
    raw_ref 缺失时用 default_raw_ref 填充。
    """
    if ev.raw_ref:
        return ev
    return ev.model_copy(update={"raw_ref": default_raw_ref(ev, index)})


def events_from_messages(
    messages: list[dict],
    *,
    scope: Scope,
    event_time: str,
    user_meta: dict,
    visibility: str,
    source_system: str = "mem0_api",
) -> list[SourceEvent]:
    """Turn coerced chat messages into one 'message' SourceEvent each (system messages included, never extracted).
    将规整后的聊天消息逐条转为 'message' SourceEvent（含 system 消息，但它们不会被抽取）。

    Input / 输入：
        messages: Coerced messages `{'role', 'content', 'dropped_parts'?, 'image_parts'?, 'timestamp'?, 'name'?}` (memory.utils.coerce_messages, optionally annotated by the vision pass).
        messages：规整后的消息 `{'role', 'content', 'dropped_parts'?, 'image_parts'?, 'timestamp'?, 'name'?}`（memory.utils.coerce_messages，可经视觉前置阶段标注）。

        scope: Session scope of the call.
        scope：本次调用的会话范围。

        event_time: Event time for messages without their own `timestamp` (any to_utc_iso form).
        event_time：没有自带 `timestamp` 的消息的事件时间（to_utc_iso 接受的任意形式）。

        user_meta: Caller metadata stored on every event.
        user_meta：存于每个事件上的调用方元数据。

        visibility: Visibility value applied to every event.
        visibility：应用于每个事件的可见性值。

        source_system: Source system label (default 'mem0_api').
        source_system：来源系统标签（默认 'mem0_api'）。

    Output / 输出：
        Events in message order; payload `{'text', 'dropped_parts'}` plus `parts` (image sha256/mime, never bytes) and `described_parts` when image parts exist / were described — descriptions are merged into `text` (merge_image_descriptions) and subtracted from dropped_parts; role, actor_id = name, raw_ref = chat://<scope>/<index>.
        按消息顺序的事件；载荷 `{'text', 'dropped_parts'}`，有图片部件 / 已描述时另含 `parts`（图片 sha256/mime，绝不含字节）与 `described_parts`——描述并入 `text`（merge_image_descriptions）并从 dropped_parts 中扣除；角色、actor_id = name、raw_ref = chat://<范围>/<序号>。

    State effect / 状态影响：
        None.
        无。
    """
    base_time = to_utc_iso(event_time)
    out: list[SourceEvent] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"messages[{index}] must be a dict", "VALIDATION_003", details={"index": index})
        text = merge_image_descriptions(message)
        role = message.get("role") or Role.user.value
        timestamp = message.get("timestamp")
        ev = SourceEvent(
            source_system=source_system,
            object_type=ObjectType.message,
            event_type=EventType.created,
            actor_id=message.get("name"),
            role=str(role),
            timestamp=to_utc_iso(timestamp) if timestamp not in (None, "") else base_time,
            visibility=str(visibility),
            scope=scope,
            payload=_message_payload(message, text),
            user_meta=dict(user_meta or {}),
        )
        out.append(_with_raw_ref(ev, index))
    return out


def _scope_of_item(item: dict, scope_fallback: Scope | None, index: int) -> Scope:
    """Scope of an add_events dict: its own scope keys, else the fallback, else VALIDATION_001.
    add_events 字典的范围：自带范围键，否则回退范围，否则 VALIDATION_001。
    """
    raw_scope = item.get("scope")
    if isinstance(raw_scope, Scope):
        return raw_scope
    if isinstance(raw_scope, dict):
        return Scope(**{k: raw_scope.get(k) for k in _SCOPE_KEYS if raw_scope.get(k) is not None})
    keys = {k: item.get(k) for k in _SCOPE_KEYS if item.get(k) is not None}
    if keys:
        return Scope(**keys)
    if scope_fallback is not None and not scope_fallback.is_empty():
        return scope_fallback
    raise ValidationError(
        f"events[{index}]: at least one of user_id, agent_id, or run_id is required",
        "VALIDATION_001",
        details={"index": index, "provided_ids": {k: None for k in _SCOPE_KEYS}, "context": "add_events"},
        suggestion="Pass user_id/agent_id/run_id to add_events or on each event dict",
    )


def _default_event_type(object_type: str, payload: Any) -> str:
    """Default event_type: tool_result for logs, traces and answered tool calls; else created.
    默认 event_type：日志、轨迹与已有结果的工具调用为 tool_result；否则 created。
    """
    if object_type in (ObjectType.test_log.value, ObjectType.sensor_trace.value):
        return EventType.tool_result.value
    if object_type == ObjectType.tool_call.value and isinstance(payload, dict):
        if payload.get("status") is not None or payload.get("result") is not None:
            return EventType.tool_result.value
    return EventType.created.value


def events_from_dicts(
    items: list[SourceEvent | dict],
    *,
    scope_fallback: Scope | None,
    visibility_default: str,
    event_time: str | None = None,
) -> list[SourceEvent]:
    """Validate add_events input: scope fallback, internal types rejected, defaults for event_type / role / raw_ref.
    校验 add_events 输入：范围回退、拒绝内部类型、event_type / role / raw_ref 的默认值。

    Input / 输入：
        items: SourceEvents or dicts with object_type, payload, optional role/actor_id/timestamp/visibility/scope keys.
        items：SourceEvent 或含 object_type、payload 及可选 role/actor_id/timestamp/visibility/范围键的字典。

        scope_fallback: Scope used by dicts without scope keys (None or empty → VALIDATION_001).
        scope_fallback：无范围键的字典使用的范围（None 或空 → VALIDATION_001）。

        visibility_default: Visibility applied when an item carries none.
        visibility_default：条目未指定时应用的可见性。

        event_time: Timestamp for items without one (defaults to the library clock of the caller; None → now_iso()).
        event_time：条目无时间戳时使用的时间（调用方应传库时钟；None → now_iso()）。

    Output / 输出：
        Events sorted stably by (timestamp, index); VALIDATION_007 for internal types, VALIDATION_005 (with index) for bad payloads.
        按 (timestamp, 序号) 稳定排序的事件；内部类型 → VALIDATION_007，坏载荷 → 带 index 的 VALIDATION_005。

    State effect / 状态影响：
        None.
        无。
    """
    default_time = to_utc_iso(event_time) if event_time else None
    built: list[tuple[str, int, SourceEvent]] = []
    for index, item in enumerate(items):
        if isinstance(item, SourceEvent):
            if str(item.object_type) in INTERNAL_OBJECT_TYPES:
                raise ValidationError(
                    f"events[{index}]: object_type {item.object_type!r} is internal and cannot be added",
                    "VALIDATION_007",
                    details={"index": index, "object_type": str(item.object_type)},
                )
            ev = _with_raw_ref(item, index)
            built.append((ev.timestamp, index, ev))
            continue
        if not isinstance(item, dict):
            raise ValidationError(
                f"events[{index}] must be a dict or SourceEvent",
                "VALIDATION_005",
                details={"index": index, "field": "event"},
            )
        object_type = str(item.get("object_type") or ObjectType.message.value)
        if object_type in INTERNAL_OBJECT_TYPES:
            raise ValidationError(
                f"events[{index}]: object_type {object_type!r} is internal and cannot be added",
                "VALIDATION_007",
                details={"index": index, "object_type": object_type},
            )
        if object_type not in ObjectType.values():
            raise ValidationError(
                f"events[{index}]: unknown object_type {object_type!r}",
                "VALIDATION_005",
                details={"index": index, "object_type": object_type, "field": "object_type"},
            )
        scope = _scope_of_item(item, scope_fallback, index)
        payload = item.get("payload")
        if payload is None and object_type == ObjectType.message.value and isinstance(item.get("content"), str):
            payload = {"text": item["content"], "dropped_parts": 0}
        try:
            normalized = validate_payload(object_type, payload if payload is not None else {})
        except ValidationError as exc:
            details = dict(exc.details or {})
            details.setdefault("index", index)
            raise ValidationError(exc.message, "VALIDATION_005", details=details, suggestion=exc.suggestion) from exc
        role = item.get("role")
        if role is None:
            role = Role.tool.value if object_type in _STRUCTURED_TYPES else Role.user.value
        event_type = item.get("event_type") or _default_event_type(object_type, normalized)
        timestamp = item.get("timestamp")
        fields: dict[str, Any] = {
            "event_id": item.get("event_id") or "",
            "source_system": item.get("source_system") or "hms",
            "object_type": object_type,
            "event_type": str(event_type),
            "actor_id": item.get("actor_id"),
            "role": str(role),
            "visibility": str(item.get("visibility") or visibility_default),
            "scope": scope,
            "payload": normalized,
            "user_meta": dict(item.get("user_meta") or item.get("metadata") or {}),
        }
        if item.get("raw_ref"):
            fields["raw_ref"] = item["raw_ref"]
        if timestamp not in (None, ""):
            fields["timestamp"] = timestamp
        elif default_time is not None:
            fields["timestamp"] = default_time
        try:
            ev = SourceEvent(**fields)
        except ValidationError as exc:
            details = dict(exc.details or {})
            details.setdefault("index", index)
            raise ValidationError(exc.message, "VALIDATION_005", details=details, suggestion=exc.suggestion) from exc
        except (pydantic.ValidationError, ValueError, TypeError) as exc:
            raise ValidationError(
                f"events[{index}]: malformed event: {exc}",
                "VALIDATION_005",
                details={"index": index, "object_type": object_type, "field": "event"},
            ) from exc
        ev = _with_raw_ref(ev, index)
        built.append((ev.timestamp, index, ev))
    built.sort(key=lambda entry: (entry[0], entry[1]))
    return [ev for _ts, _i, ev in built]


def anchor_events(store: StateStore, events: list[SourceEvent]) -> list[tuple[SourceEvent, Anchor, bool]]:
    """Txn A: persist every event and its anchor (One Event, One Anchor); identical events dedup.
    事务 A：持久化每个事件及其锚点（一事件一锚点）；相同事件去重。

    Input / 输入：
        store: The state store.
        store：状态存储。

        events: Events to anchor, in order.
        events：待落锚的事件（有序）。

    Output / 输出：
        `[(stored_event, anchor, dedup)]` in input order; dedup is True when the event already existed.
        按输入顺序的 `[(已存事件, 锚点, dedup)]`；事件已存在时 dedup 为 True。

    State effect / 状态影响：
        Writes source_events and anchors inside one transaction.
        在一个事务内写入 source_events 与 anchors。
    """
    out: list[tuple[SourceEvent, Anchor, bool]] = []
    with store.transaction():
        for ev in events:
            stored, dedup = store.insert_source_event(ev)
            anchor = store.get_or_create_anchor(stored)
            out.append((stored, anchor, dedup))
    return out


def system_event(
    store: StateStore,
    *,
    scope: Scope,
    object_type: str,
    event_type: str,
    payload: dict,
    actor_id: str | None = None,
    visibility: str = "private",
) -> tuple[SourceEvent, Anchor]:
    """Write the internal event + anchor that witnesses an update/delete/feedback/checkpoint/rollback/scheduler run.
    写入见证 update/delete/feedback/checkpoint/rollback/scheduler 操作的内部事件与锚点。

    Input / 输入：
        store: The state store (its clock stamps the event).
        store：状态存储（其时钟为事件盖时间戳）。

        scope: Scope of the operation.
        scope：操作的范围。

        object_type: One of INTERNAL_OBJECT_TYPES (feedback, checkpoint, rollback, system, scheduler).
        object_type：INTERNAL_OBJECT_TYPES 之一（feedback、checkpoint、rollback、system、scheduler）。

        event_type: EventType value describing the mutation.
        event_type：描述该变更的 EventType 值。

        payload: Payload normalized by validate_payload for the object type.
        payload：按对象类型经 validate_payload 归一化的载荷。

        actor_id: Optional actor.
        actor_id：可选的行动者。

        visibility: Visibility (= anchor permission_scope) of the event.
        visibility：事件的可见性（即锚点的 permission_scope）。

    Output / 输出：
        (stored event, anchor).
        (已存事件, 锚点)。

    State effect / 状态影响：
        Writes one source_events row and one anchors row inside the current transaction, as unversioned rows (StateStore.unversioned(), §6.7): the event attributes the enclosing operation to `scope`, but only that operation's own row changes (an archive, a revision, a decay …) move the scope's write version — a scheduler or checkpoint event alone leaves HMS.context()'s memo reused.
        在当前事务内写入一行 source_events 与一行 anchors，且为不带版本的行（StateStore.unversioned()，§6.7）：事件把外层操作归于 `scope`，但只有该操作自身改变的行（归档、修订、衰减……）才推进范围的写入版本——单独的调度器或检查点事件让 HMS.context() 的记忆保持复用。
    """
    now = store.now()
    ev = SourceEvent(
        source_system="hms",
        object_type=str(object_type),
        event_type=str(event_type),
        actor_id=actor_id,
        role=None,
        timestamp=now,
        visibility=str(visibility),
        scope=scope,
        payload=dict(payload),
    )
    ev = _with_raw_ref(ev, 0)
    with store.transaction(), store.unversioned():
        stored, _dedup = store.insert_source_event(ev)
        anchor = store.get_or_create_anchor(stored)
    return stored, anchor


__all__ = [
    "IMAGE_LINE",
    "anchor_events",
    "default_raw_ref",
    "events_from_dicts",
    "events_from_messages",
    "merge_image_descriptions",
    "system_event",
    "validate_payload",
]
