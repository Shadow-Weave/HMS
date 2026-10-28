"""Event routing: which extraction paths (observation / state / structured) an event enters.
事件分流：决定事件进入哪些抽取路径（observation / state / structured）。
"""

# Purpose (EN): Decide per SourceEvent whether it feeds observation extraction, EP extraction, the closure builder, or nothing.
# 目的（中）：为每个 SourceEvent 决定其进入观察抽取、EP 抽取、闭包构建器，还是不抽取。

# Input (EN): A SourceEvent (role, object_type, payload text, user_meta['route']).
# 输入（中）：一个 SourceEvent（角色、对象类型、载荷文本、user_meta['route']）。

# Output (EN): A set of route names drawn from {'observation', 'state', 'structured'}.
# 输出（中）：取自 {'observation', 'state', 'structured'} 的路由名集合。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

from hms_elite.enums import ObjectType, Role
from hms_elite.models import SourceEvent
from hms_elite.utils.text import has_action_closure

_STRUCTURED_TYPES = frozenset(
    {
        ObjectType.tool_call.value,
        ObjectType.file_diff.value,
        ObjectType.test_log.value,
        ObjectType.sensor_trace.value,
    }
)
_CHAT_ROLES = frozenset({Role.user.value, Role.assistant.value})


def route_event(ev: SourceEvent) -> set[str]:
    """Route an event: chat messages → observation (+ state on action closure), structured types → structured, else nothing.
    为事件分流：聊天消息 → observation（动作闭包时加 state），结构化类型 → structured，其余不抽取。

    Input / 输入：
        ev: The event.
        ev：事件。

    Output / 输出：
        {'observation'} | {'observation', 'state'} | {'structured'} | set() (system-role messages and internal types).
        {'observation'} | {'observation', 'state'} | {'structured'} | set()（system 角色消息与内部类型）。

    State effect / 状态影响：
        None.
        无。
    """
    object_type = str(ev.object_type)
    if object_type in _STRUCTURED_TYPES:
        return {"structured"}
    if object_type != ObjectType.message.value:
        return set()
    if ev.role is None or str(ev.role) not in _CHAT_ROLES:
        return set()
    routes = {"observation"}
    text = str((ev.payload or {}).get("text") or "")
    if (ev.user_meta or {}).get("route") == "state" or has_action_closure(text):
        routes.add("state")
    return routes


__all__ = ["has_action_closure", "route_event"]
