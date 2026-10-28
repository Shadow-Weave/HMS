"""Agent runtime adapter and action registry: the HMS_arch §5.1 `runtime.py` face for coding agents.
Agent 运行时适配器与动作注册表：HMS_arch §5.1 的 `runtime.py`，面向编码 Agent 的接入面。
"""

# Purpose (EN): The §11.1 adapter through which Claude Code / Codex / Cursor / OpenClaw feed HMS: an ActionRegistry
#               maps action calls and their results onto SourceEvent payloads (tool_call / file_diff / test_log /
#               sensor_trace), and an AgentRuntime turns observe / act / outcome / state / why / handoff / checkpoint /
#               rollback / feedback / context into calls on the public HMS facade.
# 目的（中）：§11.1 的适配器，Claude Code / Codex / Cursor / OpenClaw 经由它接入 HMS：ActionRegistry 把动作调用及其结果
#            映射为 SourceEvent 载荷（tool_call / file_diff / test_log / sensor_trace），AgentRuntime 把 observe / act /
#            outcome / state / why / handoff / checkpoint / rollback / feedback / context 变成对公共 HMS 门面的调用。

# Input (EN): An HMS facade, a scope (user_id / agent_id / run_id), action names with keyword arguments, results and
#             statuses, dialogue messages, questions, feedback.
# 输入（中）：HMS 门面、范围（user_id / agent_id / run_id）、带关键字参数的动作名、结果与状态、对话消息、问题、反馈。

# Output (EN): Normalized event dicts ready for add_events, action ids, outcome reports (EP / constraint ids), Work
#              State Card text, ContextIR, StatePackage, checkpoint ids and facade report dicts.
# 输出（中）：可直接交给 add_events 的归一化事件字典、动作 id、结果报告（EP / 约束 id）、Work State Card 文本、
#            ContextIR、StatePackage、检查点 id 与门面报告字典。

# State effect (EN): The registry is pure; every AgentRuntime write goes through the facade (add_events, checkpoint,
#                    rollback, feedback, handoff) and the runtime itself only tracks the open actions in memory.
# 状态影响（中）：注册表是纯函数；AgentRuntime 的每次写入都经过门面（add_events、checkpoint、rollback、feedback、
#                handoff），运行时自身只在内存中跟踪未闭合的动作。

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from hms_elite.enums import ObjectType
from hms_elite.exceptions import ValidationError
from hms_elite.models import ContextIR, Scope, StatePackage, validate_payload
from hms_elite.timeutil import to_utc_iso

# 调用映射：args -> {'object_type', 'payload'}；结果映射：(args, result, status) -> {'object_type', 'payload'}
CallMapper = Callable[[dict[str, Any]], dict[str, Any]]  # args -> {'object_type', 'payload'}
OutcomeMapper = Callable[[dict[str, Any], Any, str], dict[str, Any]]  # (args, result, status) -> event dict

STATUS_VALUES = ("ok", "error")
# 动作事件只能是闭包构建器的两种动作类型；结果事件只能是其结果类型（test_log 闭合，sensor_trace 仅落原子）
ACTION_OBJECT_TYPES = frozenset({ObjectType.tool_call.value, ObjectType.file_diff.value})  # closure actions
OUTCOME_OBJECT_TYPES = frozenset({ObjectType.test_log.value, ObjectType.sensor_trace.value})  # closure outcomes
DEFAULT_ACTIONS = ("patch", "run_tests", "shell", "http")
_DEFAULT_QUERY = "current work state"
_LINE_LIMIT = 120
# 带这些键之一的字典结果被视为测试结果（走 outcome_from_test_result 而不是 JSON 文本）
_TEST_KEYS = frozenset({"suite", "passed", "failed", "failures", "failed_tests", "log"})  # test-shaped results


def _bad_literal(message: str, **details: Any) -> ValidationError:
    """A VALIDATION_013 error (bad literal) with details.
    带 details 的 VALIDATION_013 错误（非法字面量）。
    """
    return ValidationError(message, "VALIDATION_013", details=details)


def _text_of(result: Any) -> str:
    """Text form of an action result: str as is, None → '', containers → JSON, anything else → str().
    动作结果的文本形式：str 原样、None → ''、容器 → JSON、其余 → str()。
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    return str(result)


def _first_line(text: str, limit: int = _LINE_LIMIT) -> str:
    """The first non-empty line of `text`, truncated to `limit` characters.
    `text` 的第一个非空行，截断到 `limit` 个字符。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def _str_or_none(value: Any) -> str | None:
    """`value` as a non-empty string, else None.
    非空字符串形式的 `value`，否则为 None。
    """
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _constraints_of(args: dict[str, Any]) -> list[str]:
    """The `constraints` list of an action's args (a lone string is wrapped).
    动作参数中的 `constraints` 列表（单个字符串会被包装成列表）。
    """
    raw = args.get("constraints")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _require_arg(action: str, args: dict[str, Any], name: str) -> str:
    """The required string argument `name` of `action` (VALIDATION_005 when missing or empty).
    `action` 必需的字符串参数 `name`（缺失或为空时 VALIDATION_005）。
    """
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"action {action!r} needs a non-empty string argument {name!r}",
            "VALIDATION_005",
            details={"object_type": "action", "field": name, "action": action},
        )
    return value


def generic_outcome(suite: str, args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """Map any action result onto a test_log outcome event: 'ok' → passed (1/0), 'error' → failed (0/1) with one failure.
    把任意动作结果映射为 test_log 结果事件：'ok' → passed（1/0），'error' → failed（0/1）并带一条失败项。

    Input / 输入：
        suite: The outcome suite name (becomes the first word of the EP's y).
        suite：结果的 suite 名（成为 EP 中 y 的第一个词）。

        args: The action's keyword arguments (only `related_paths` / `path` are read).
        args：动作的关键字参数（只读取 `related_paths` / `path`）。

        result: The raw result (str, dict, list or None).
        result：原始结果（str、dict、list 或 None）。

        status: 'ok' or 'error'.
        status：'ok' 或 'error'。

    Output / 输出：
        {'object_type': 'test_log', 'payload': {suite, status, passed, failed, failures, log, related_paths}}.
        {'object_type': 'test_log', 'payload': {suite, status, passed, failed, failures, log, related_paths}}。

    State effect / 状态影响：
        None (pure).
        无（纯函数）。
    """
    if isinstance(result, dict) and _TEST_KEYS & set(result):
        # A test-shaped result (suite / passed / failed / failures / log) keeps its structure instead of a JSON dump.
        # 测试形状的结果（suite / passed / failed / failures / log）保留其结构，而不是被转成 JSON 文本。
        return outcome_from_test_result(suite, args, result, status)
    text = _text_of(result)
    failed = status == "error"
    failures: list[dict[str, Any]] = []
    if failed:
        failures.append({"name": _first_line(text) or suite, "message": text or None})
    related = args.get("related_paths")
    paths = [str(p) for p in related] if isinstance(related, (list, tuple)) else []
    if isinstance(args.get("path"), str) and args["path"] not in paths:
        paths.append(args["path"])
    return {
        "object_type": ObjectType.test_log.value,
        "payload": {
            "suite": suite,
            "status": "failed" if failed else "passed",
            "passed": 0 if failed else 1,
            "failed": 1 if failed else 0,
            "failures": failures,
            "log": text or None,
            "related_paths": paths,
        },
    }


def _patch_call(args: dict[str, Any]) -> dict[str, Any]:
    """`patch(path, patch, summary?, intent?, commit?, constraints?)` → file_diff.
    `patch(path, patch, summary?, intent?, commit?, constraints?)` → file_diff。
    """
    return {
        "object_type": ObjectType.file_diff.value,
        "payload": {
            "path": _require_arg("patch", args, "path"),
            "patch": _require_arg("patch", args, "patch"),
            "summary": _str_or_none(args.get("summary")),
            "intent": _str_or_none(args.get("intent")),
            "commit": _str_or_none(args.get("commit")),
            "constraints": _constraints_of(args),
        },
    }


def _patch_outcome(args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """A patch result → test_log with suite 'patch <path>'.
    补丁结果 → suite 为 'patch <path>' 的 test_log。
    """
    return generic_outcome(f"patch {args.get('path', '')}".strip(), args, result, status)


def _tool_args(args: dict[str, Any], *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """The JSON-safe tool args: every argument except the meta keys (intent, constraints, target) and `exclude`.
    JSON 安全的工具参数：除元键（intent、constraints、target）与 `exclude` 之外的全部参数。
    """
    skip = {"intent", "constraints", "target", *exclude}
    out: dict[str, Any] = {}
    for key in sorted(args):
        if key in skip:
            continue
        value = args[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = _text_of(value)
    return out


def _tool_call(tool: str, args: dict[str, Any], *, target: str | None) -> dict[str, Any]:
    """An open tool_call event (no status → stays open until an outcome closes it).
    未闭合的 tool_call 事件（无 status → 保持打开直到结果将其闭合）。
    """
    return {
        "object_type": ObjectType.tool_call.value,
        "payload": {
            "tool": tool,
            "args": _tool_args(args),
            "result": None,
            "status": None,
            "target": _str_or_none(args.get("target")) or target,
            "intent": _str_or_none(args.get("intent")),
            "constraints": _constraints_of(args),
        },
    }


def _run_tests_call(args: dict[str, Any]) -> dict[str, Any]:
    """`run_tests(suite, ...)` → tool_call(tool='run_tests', target=suite).
    `run_tests(suite, ...)` → tool_call(tool='run_tests', target=suite)。
    """
    return _tool_call("run_tests", args, target=_str_or_none(args.get("suite")))


def outcome_from_test_result(suite: str, args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """Map a test run result (dict with suite/passed/failed/failures/log, or a log string) onto a real test_log event.
    把测试运行结果（含 suite/passed/failed/failures/log 的字典，或日志字符串）映射为真正的 test_log 事件。

    Input / 输入：
        suite: Fallback suite name (a 'suite' key in the result wins).
        suite：回退的 suite 名（结果中的 'suite' 键优先）。

        args: The action's keyword arguments (`related_paths` is read).
        args：动作的关键字参数（读取 `related_paths`）。

        result: dict with suite/passed/failed/failures|failed_tests/log/status/related_paths, or a log string.
        result：含 suite/passed/failed/failures|failed_tests/log/status/related_paths 的字典，或日志字符串。

        status: 'ok' or 'error' ('error' forces status 'failed').
        status：'ok' 或 'error'（'error' 强制 status 为 'failed'）。

    Output / 输出：
        {'object_type': 'test_log', 'payload': {suite, status, passed, failed, failures, log, related_paths}}.
        {'object_type': 'test_log', 'payload': {suite, status, passed, failed, failures, log, related_paths}}。

    State effect / 状态影响：
        None (pure).
        无（纯函数）。
    """
    data = result if isinstance(result, dict) else {}
    failures = data.get("failures", data.get("failed_tests")) or []
    failed_count = data.get("failed")
    explicit = data.get("status")
    if status == "error" or (failed_count or 0) > 0 or failures or explicit == "failed":
        state = "failed"
    else:
        state = "passed"
    log = data.get("log") if isinstance(result, dict) else _text_of(result)
    related = data.get("related_paths") or args.get("related_paths") or []
    return {
        "object_type": ObjectType.test_log.value,
        "payload": {
            "suite": str(data.get("suite") or suite),
            "status": state,
            "passed": data.get("passed"),
            "failed": failed_count,
            "failures": list(failures),
            "log": _str_or_none(log),
            "related_paths": [str(p) for p in related],
        },
    }


def _run_tests_outcome(args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """`run_tests` result → test_log with the suite from args (default 'tests').
    `run_tests` 结果 → suite 取自参数（默认 'tests'）的 test_log。
    """
    return outcome_from_test_result(_str_or_none(args.get("suite")) or "tests", args, result, status)


def _shell_call(args: dict[str, Any]) -> dict[str, Any]:
    """`shell(cmd, cwd?, ...)` → tool_call(tool='shell', target=cwd).
    `shell(cmd, cwd?, ...)` → tool_call(tool='shell', target=cwd)。
    """
    _require_arg("shell", args, "cmd")
    return _tool_call("shell", args, target=_str_or_none(args.get("cwd")))


def _shell_outcome(args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """A shell result → test_log with suite 'shell'.
    shell 结果 → suite 为 'shell' 的 test_log。
    """
    return generic_outcome("shell", args, result, status)


def _http_call(args: dict[str, Any]) -> dict[str, Any]:
    """`http(url, method='GET', ...)` → tool_call(tool='http', target=url).
    `http(url, method='GET', ...)` → tool_call(tool='http', target=url)。
    """
    url = _require_arg("http", args, "url")
    args = {"method": "GET", **args}
    return _tool_call("http", args, target=url)


def _http_outcome(args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
    """An HTTP result → test_log with suite 'http <METHOD> <url>'.
    HTTP 结果 → suite 为 'http <METHOD> <url>' 的 test_log。
    """
    method = str(args.get("method") or "GET").upper()
    return generic_outcome(f"http {method} {args.get('url', '')}".strip(), args, result, status)


@dataclass(frozen=True)
class ActionSpec:
    """One registered action: its name and the two mappers onto SourceEvent dicts.
    一个已注册的动作：名称与两个到 SourceEvent 字典的映射器。
    """

    name: str
    call: CallMapper
    outcome: OutcomeMapper
    description: str = ""


class ActionRegistry:
    """Action name → how a call and its result become SourceEvents (HMS_arch §5.1 'action registry').
    动作名 → 调用及其结果如何变成 SourceEvent（HMS_arch §5.1 的 'action registry'）。
    """

    def __init__(self, *, defaults: bool = True) -> None:
        """Create a registry, optionally pre-loaded with patch / run_tests / shell / http.
        创建注册表，可预装 patch / run_tests / shell / http。

        Input / 输入：
            defaults: Register the four default actions.
            defaults：是否注册四个默认动作。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None outside the registry object.
            注册表对象之外无影响。
        """
        self._specs: dict[str, ActionSpec] = {}
        if defaults:
            self.register("patch", call=_patch_call, outcome=_patch_outcome, description="file_diff + test_log")
            self.register(
                "run_tests", call=_run_tests_call, outcome=_run_tests_outcome, description="tool_call + test_log"
            )
            self.register("shell", call=_shell_call, outcome=_shell_outcome, description="tool_call + test_log")
            self.register("http", call=_http_call, outcome=_http_outcome, description="tool_call + test_log")

    def register(
        self, name: str, *, call: CallMapper, outcome: OutcomeMapper | None = None, description: str = ""
    ) -> ActionSpec:
        """Register (or replace) an action.
        注册（或替换）一个动作。

        Input / 输入：
            name: Action name (non-empty).
            name：动作名（非空）。

            call: args → {'object_type': tool_call|file_diff, 'payload': {...}}.
            call：args → {'object_type': tool_call|file_diff, 'payload': {...}}。

            outcome: (args, result, status) → {'object_type': test_log|sensor_trace, 'payload'}; None → generic_outcome(name).
            outcome：(args, result, status) → {'object_type': test_log|sensor_trace, 'payload'}；None → generic_outcome(name)。

            description: Free text shown by names().
            description：names() 中展示的自由文本。

        Output / 输出：
            The ActionSpec stored.
            存入的 ActionSpec。

        State effect / 状态影响：
            Mutates the registry only.
            仅修改注册表。
        """
        if not isinstance(name, str) or not name.strip():
            raise _bad_literal("action name must be a non-empty string", name=name)
        if not callable(call):
            raise _bad_literal(f"action {name!r}: call mapper must be callable", name=name)
        if outcome is None:

            def _default_outcome(args: dict[str, Any], result: Any, status: str, _suite: str = name) -> dict[str, Any]:
                """generic_outcome bound to the action name.
                绑定到动作名的 generic_outcome。
                """
                return generic_outcome(_suite, args, result, status)

            outcome = _default_outcome
        spec = ActionSpec(name=name, call=call, outcome=outcome, description=description)
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> bool:
        """Remove an action.
        移除一个动作。

        Input / 输入：
            name: Action name.
            name：动作名。

        Output / 输出：
            True when it existed.
            存在时为 True。

        State effect / 状态影响：
            Mutates the registry only.
            仅修改注册表。
        """
        return self._specs.pop(name, None) is not None

    def get(self, name: str) -> ActionSpec:
        """Look up an action (unknown → VALIDATION_013).
        查找动作（未知 → VALIDATION_013）。

        Input / 输入：
            name: Action name.
            name：动作名。

        Output / 输出：
            The ActionSpec.
            对应的 ActionSpec。

        State effect / 状态影响：
            None.
            无。
        """
        spec = self._specs.get(name)
        if spec is None:
            raise _bad_literal(f"unknown action {name!r}", action=name, supported=self.names())
        return spec

    def names(self) -> list[str]:
        """Registered action names in registration order.
        按注册顺序排列的动作名。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of names.
            名称列表。

        State effect / 状态影响：
            None.
            无。
        """
        return list(self._specs)

    def __contains__(self, name: object) -> bool:
        """`name in registry`.
        `name in registry`。
        """
        return name in self._specs

    def call_event(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The normalized action event of a call (payload passed through validate_payload).
        一次调用的归一化动作事件（载荷经 validate_payload）。

        Input / 输入：
            name: Action name.
            name：动作名。

            args: Keyword arguments of the call.
            args：调用的关键字参数。

        Output / 输出：
            {'object_type': tool_call|file_diff, 'payload': <normalized>} (other types → VALIDATION_013).
            {'object_type': tool_call|file_diff, 'payload': <归一化>}（其他类型 → VALIDATION_013）。

        State effect / 状态影响：
            None (pure).
            无（纯函数）。
        """
        return self._normalize(name, self.get(name).call(dict(args)), ACTION_OBJECT_TYPES, "call")

    def outcome_event(self, name: str, args: dict[str, Any], result: Any, status: str) -> dict[str, Any]:
        """The normalized outcome event of a result (payload passed through validate_payload).
        一次结果的归一化结果事件（载荷经 validate_payload）。

        Input / 输入：
            name: Action name.
            name：动作名。

            args: Keyword arguments of the original call.
            args：原调用的关键字参数。

            result: Raw result.
            result：原始结果。

            status: 'ok' or 'error' (VALIDATION_013 otherwise).
            status：'ok' 或 'error'（否则 VALIDATION_013）。

        Output / 输出：
            {'object_type': test_log|sensor_trace, 'payload': <normalized>} (other types → VALIDATION_013).
            {'object_type': test_log|sensor_trace, 'payload': <归一化>}（其他类型 → VALIDATION_013）。

        State effect / 状态影响：
            None (pure).
            无（纯函数）。
        """
        if status not in STATUS_VALUES:
            raise _bad_literal(f"status must be one of {STATUS_VALUES}, got {status!r}", status=status)
        return self._normalize(
            name, self.get(name).outcome(dict(args), result, status), OUTCOME_OBJECT_TYPES, "outcome"
        )

    @staticmethod
    def _normalize(name: str, event: Any, allowed: frozenset[str], kind: str) -> dict[str, Any]:
        """Check the mapper output shape and normalize its payload.
        检查映射器输出形状并归一化其载荷。
        """
        if not isinstance(event, dict) or "object_type" not in event or "payload" not in event:
            raise _bad_literal(f"action {name!r}: {kind} mapper must return {{'object_type', 'payload'}}", action=name)
        object_type = str(getattr(event["object_type"], "value", event["object_type"]))
        if object_type not in allowed:
            raise _bad_literal(
                f"action {name!r}: {kind} mapper returned object_type {object_type!r}, allowed {sorted(allowed)}",
                action=name,
                object_type=object_type,
            )
        return {"object_type": object_type, "payload": validate_payload(object_type, event["payload"])}


@dataclass
class PendingAction:
    """An action recorded by act() whose outcome has not arrived yet.
    act() 已记录、结果尚未到达的动作。
    """

    action_id: str
    action: str
    args: dict[str, Any]
    object_type: str
    event_id: str
    anchor_id: str
    timestamp: str | None = None
    dedup: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict form.
        普通字典形式。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'action_id', 'action', 'args', 'object_type', 'event_id', 'anchor_id', 'timestamp', 'dedup'}.
            {'action_id', 'action', 'args', 'object_type', 'event_id', 'anchor_id', 'timestamp', 'dedup'}。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "action_id": self.action_id,
            "action": self.action,
            "args": dict(self.args),
            "object_type": self.object_type,
            "event_id": self.event_id,
            "anchor_id": self.anchor_id,
            "timestamp": self.timestamp,
            "dedup": self.dedup,
        }


class AgentRuntime:
    """The agent-side adapter (HMS_arch §11.1): one runtime per (user, agent, run), every write via the HMS facade.
    Agent 侧适配器（HMS_arch §11.1）：每个 (user, agent, run) 一个运行时，所有写入都经过 HMS 门面。
    """

    def __init__(
        self,
        hms: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        registry: ActionRegistry | None = None,
        goal: str | None = None,
        source_system: str = "agent",
        visibility: str | None = None,
        infer: bool = True,
    ) -> None:
        """Bind a runtime to an HMS facade and a scope.
        把运行时绑定到一个 HMS 门面与一个范围。

        Input / 输入：
            hms: The HMS facade (only its public methods are used).
            hms：HMS 门面（只使用其公共方法）。

            user_id: Mem0 user id (at least one of the three ids, VALIDATION_001).
            user_id：Mem0 用户 id（三个 id 至少一个，否则 VALIDATION_001）。

            agent_id: Agent id (also the actor_id of action / outcome events).
            agent_id：代理 id（也是动作 / 结果事件的 actor_id）。

            run_id: Run id (the closure builder pairs actions and outcomes per run).
            run_id：运行 id（闭包构建器按 run 配对动作与结果）。

            registry: ActionRegistry (None → the default four actions).
            registry：ActionRegistry（None → 默认四个动作）。

            goal: The current goal used by state() / context() / handoff() when no query is given.
            goal：state() / context() / handoff() 未给查询时使用的当前目标。

            source_system: SourceEvent.source_system of every event ('claude_code', 'cursor', ...).
            source_system：每个事件的 SourceEvent.source_system（'claude_code'、'cursor' 等）。

            visibility: Visibility of the events (None → the facade's default).
            visibility：事件可见性（None → 门面默认值）。

            infer: Whether observe() messages go through LLM extraction.
            infer：observe() 的消息是否走 LLM 抽取。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None (nothing is written until the first observe / act).
            无（首次 observe / act 之前不写入任何东西）。
        """
        self.hms = hms
        self.scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id).require_any(context="AgentRuntime")
        self.registry = registry if registry is not None else ActionRegistry()
        self.goal = goal
        self.source_system = source_system
        self.visibility = visibility
        self.infer = bool(infer)
        self._open: dict[str, PendingAction] = {}
        self._counter = 0

    # ------------------------------------------------------------------ scope helpers
    # ------------------------------------------------------------------ 范围辅助
    @property
    def filters(self) -> dict[str, str]:
        """The Mem0 filter dict of this runtime's scope.
        本运行时范围的 Mem0 过滤字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'user_id'?, 'agent_id'?, 'run_id'?} (set keys only).
            {'user_id'?, 'agent_id'?, 'run_id'?}（仅已设置的键）。

        State effect / 状态影响：
            None.
            无。
        """
        return self.scope.as_filters()

    @property
    def actor_id(self) -> str | None:
        """The actor recorded on action / outcome events: agent_id, else user_id.
        动作 / 结果事件上记录的 actor：agent_id，否则 user_id。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The actor id or None.
            actor id 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        return self.scope.agent_id or self.scope.user_id

    def set_goal(self, goal: str | None) -> None:
        """Set the current goal.
        设置当前目标。

        Input / 输入：
            goal: Goal text or None.
            goal：目标文本或 None。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Runtime memory only.
            仅运行时内存。
        """
        self.goal = goal

    def _event(self, event: dict[str, Any], *, role: str, actor_id: str | None, timestamp: Any) -> dict[str, Any]:
        """Complete an event dict with scope, role, actor, source_system, visibility and timestamp.
        用范围、角色、actor、source_system、可见性与时间戳补全事件字典。
        """
        out: dict[str, Any] = {
            "object_type": event["object_type"],
            "payload": event["payload"],
            "role": role,
            "actor_id": actor_id,
            "source_system": self.source_system,
            **self.scope.as_filters(),
        }
        if self.visibility is not None:
            out["visibility"] = self.visibility
        if timestamp is not None:
            out["timestamp"] = to_utc_iso(timestamp)
        return out

    def _query(self, query: str | None) -> str:
        """The compile query: explicit query, else the goal, else a default.
        编译查询：显式查询，否则目标，否则默认值。
        """
        return query or self.goal or _DEFAULT_QUERY

    # ------------------------------------------------------------------ ingestion face
    # ------------------------------------------------------------------ 摄取面
    def observe(
        self,
        message: str,
        *,
        role: str = "user",
        actor_id: str | None = None,
        timestamp: Any = None,
        infer: bool | None = None,
    ) -> dict:
        """Record one dialogue message (HMS_arch §3.3 observation input) through add_events.
        通过 add_events 记录一条对话消息（HMS_arch §3.3 的观察输入）。

        Input / 输入：
            message: Message text.
            message：消息文本。

            role: 'user' | 'assistant' | 'system' | 'tool'.
            role：'user' | 'assistant' | 'system' | 'tool'。

            actor_id: Who said it (None → user_id for 'user', else the runtime actor).
            actor_id：说话者（None → 'user' 角色取 user_id，否则取运行时 actor）。

            timestamp: Event time (None → the store clock).
            timestamp：事件时间（None → 存储时钟）。

            infer: Override the runtime's infer flag.
            infer：覆盖运行时的 infer 标志。

        Output / 输出：
            The add_events report ({'results', 'events', 'eps', 'edges', 'constraints'}).
            add_events 的报告（{'results', 'events', 'eps', 'edges', 'constraints'}）。

        State effect / 状态影响：
            One add_events call (anchor + atoms).
            一次 add_events 调用（锚点 + 原子）。
        """
        if actor_id is None:
            actor_id = self.scope.user_id if role == "user" else self.actor_id
        event = {"object_type": ObjectType.message.value, "payload": {"text": message}}
        return self.hms.add_events(
            [self._event(event, role=role, actor_id=actor_id, timestamp=timestamp)],
            infer=self.infer if infer is None else bool(infer),
        )

    def act(self, action: str, *, timestamp: Any = None, **args: Any) -> str:
        """Record an action call as an open tool_call / file_diff event (opens the closure) and return its action id.
        把动作调用记录为未闭合的 tool_call / file_diff 事件（打开闭包），返回动作 id。

        Input / 输入：
            action: Registered action name (VALIDATION_013 otherwise).
            action：已注册的动作名（否则 VALIDATION_013）。

            timestamp: Event time (None → the store clock).
            timestamp：事件时间（None → 存储时钟）。

            args: The action's keyword arguments (mapped by the registry).
            args：动作的关键字参数（由注册表映射）。

        Output / 输出：
            The action id (the recorded event id).
            动作 id（即记录的事件 id）。

        State effect / 状态影响：
            One add_events call; the pending action is kept in memory until outcome().
            一次 add_events 调用；未闭合动作保存在内存中直到 outcome()。
        """
        event = self.registry.call_event(action, args)
        report = self.hms.add_events(
            [self._event(event, role="tool", actor_id=self.actor_id, timestamp=timestamp)], infer=False
        )
        recorded = report["events"][0]
        action_id = str(recorded["event_id"])
        self._counter += 1
        self._open[action_id] = PendingAction(
            action_id=action_id,
            action=action,
            args=dict(args),
            object_type=event["object_type"],
            event_id=action_id,
            anchor_id=str(recorded["anchor_id"]),
            timestamp=to_utc_iso(timestamp) if timestamp is not None else None,
            dedup=bool(recorded.get("dedup")),
            extra={"seq": self._counter},
        )
        return action_id

    def outcome(self, action_id: str, result: Any = None, *, status: str = "ok", timestamp: Any = None) -> dict:
        """Record the result of an action; the closure builder pairs it with the open action into an EP.
        记录动作结果；闭包构建器把它与未闭合动作配对成 EP。

        Input / 输入：
            action_id: The id returned by act() (unknown → NOT_FOUND_001).
            action_id：act() 返回的 id（未知 → NOT_FOUND_001）。

            result: Raw result (str, dict, list or None).
            result：原始结果（str、dict、list 或 None）。

            status: 'ok' or 'error' (VALIDATION_013 otherwise).
            status：'ok' 或 'error'（否则 VALIDATION_013）。

            timestamp: Event time (None → the store clock; must not precede the action).
            timestamp：事件时间（None → 存储时钟；不得早于动作）。

        Output / 输出：
            {'action_id', 'action', 'status', 'event_id', 'anchor_id', 'dedup', 'closed', 'eps', 'edges', 'constraints', 'results'}.
            {'action_id', 'action', 'status', 'event_id', 'anchor_id', 'dedup', 'closed', 'eps', 'edges', 'constraints', 'results'}。

        State effect / 状态影响：
            One add_events call (outcome anchor, event atom, EP, edges, derived constraint on a negative result).
            一次 add_events 调用（结果锚点、事件原子、EP、边，负结果时还有派生约束）。
        """
        pending = self._open.get(str(action_id))
        if pending is None:
            raise ValidationError(
                f"outcome(): unknown or already closed action {action_id!r}",
                "NOT_FOUND_001",
                details={"action_id": action_id, "open": sorted(self._open)},
            )
        event = self.registry.outcome_event(pending.action, pending.args, result, status)
        report = self.hms.add_events(
            [self._event(event, role="tool", actor_id=self.actor_id, timestamp=timestamp)], infer=False
        )
        recorded = report["events"][0]
        eps = list(report.get("eps", []))
        closed = self._cites(eps, pending.anchor_id)
        self._open.pop(pending.action_id, None)
        return {
            "action_id": pending.action_id,
            "action": pending.action,
            "status": status,
            "event_id": str(recorded["event_id"]),
            "anchor_id": str(recorded["anchor_id"]),
            "dedup": bool(recorded.get("dedup")),
            "closed": closed,
            "eps": eps,
            "edges": list(report.get("edges", [])),
            "constraints": list(report.get("constraints", [])),
            "results": list(report.get("results", [])),
        }

    def _cites(self, ep_ids: list[str], anchor_id: str) -> bool:
        """Whether one of the EPs cites the action anchor (public store lookups only).
        是否有 EP 引用了动作锚点（仅用 store 的公开查询）。
        """
        store = getattr(self.hms, "store", None)
        if store is None:
            return bool(ep_ids)
        for ep_id in ep_ids:
            ep = store.get_ep(ep_id)
            if ep is not None and anchor_id in ep.anchor_ids:
                return True
        return False

    def open_actions(self) -> list[dict[str, Any]]:
        """The actions recorded by act() that have no outcome yet, oldest first.
        act() 已记录但尚无结果的动作，最早在前。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of PendingAction.to_dict().
            PendingAction.to_dict() 的列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [p.to_dict() for p in self._open.values()]

    def discard(self, action_id: str) -> bool:
        """Forget a pending action without recording an outcome (the event itself stays in the store).
        放弃一个未闭合动作而不记录结果（事件本身仍留在存储中）。

        Input / 输入：
            action_id: The id returned by act().
            action_id：act() 返回的 id。

        Output / 输出：
            True when it was pending.
            之前处于未闭合状态时为 True。

        State effect / 状态影响：
            Runtime memory only.
            仅运行时内存。
        """
        return self._open.pop(str(action_id), None) is not None

    # ------------------------------------------------------------------ state face
    # ------------------------------------------------------------------ 状态面
    def context(
        self,
        budget_tokens: int | None = None,
        *,
        query: str | None = None,
        mode: str | None = None,
        as_of: Any = None,
    ) -> ContextIR:
        """Compile the ContextIR to inject into the next prompt.
        编译要注入下一个提示词的 ContextIR。

        Input / 输入：
            budget_tokens: Token budget (None → the facade default).
            budget_tokens：token 预算（None → 门面默认值）。

            query: Compile query (None → the goal, else 'current work state').
            query：编译查询（None → 目标，否则 'current work state'）。

            mode: 'auto' | 'fast' | 'axis_graph' | 'deep' (None → config default).
            mode：'auto' | 'fast' | 'axis_graph' | 'deep'（None → 配置默认值）。

            as_of: Reference instant (None → the store clock).
            as_of：参考时刻（None → 存储时钟）。

        Output / 输出：
            ContextIR.
            ContextIR。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        return self.hms.compile(
            self._query(query), filters=self.filters, budget_tokens=budget_tokens, mode=mode, as_of=as_of
        )

    def state(
        self,
        budget_tokens: int | None = None,
        *,
        query: str | None = None,
        mode: str | None = None,
        as_of: Any = None,
    ) -> str:
        """The Work State Card text (HMS_arch §11.1) for this run.
        本运行的 Work State Card 文本（HMS_arch §11.1）。

        Input / 输入：
            budget_tokens: Token budget (None → the facade default).
            budget_tokens：token 预算（None → 门面默认值）。

            query: Compile query (None → the goal).
            query：编译查询（None → 目标）。

            mode: Compile mode (None → config default).
            mode：编译模式（None → 配置默认值）。

            as_of: Reference instant (None → the store clock).
            as_of：参考时刻（None → 存储时钟）。

        Output / 输出：
            The card text (ContextIR.to_text()).
            卡片文本（ContextIR.to_text()）。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        return self.context(budget_tokens, query=query, mode=mode, as_of=as_of).to_text()

    def why(self, question: str, *, hops: int = 2) -> dict:
        """Detective query: explain an object id or a natural-language question over this scope.
        侦探查询：解释一个对象 id 或本范围内的自然语言问题。

        Input / 输入：
            question: Object id or question text.
            question：对象 id 或问题文本。

            hops: Graph expansion levels.
            hops：图扩展层数。

        Output / 输出：
            The facade why() report ({'target', 'nodes', 'edges', 'chain', 'explanation', ...}).
            门面 why() 的报告（{'target', 'nodes', 'edges', 'chain', 'explanation', ...}）。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        return self.hms.why(question, hops=hops, filters=self.filters)

    def handoff(
        self, *, goal: str | None = None, budget_tokens: int | None = None, checkpoint: bool = True
    ) -> StatePackage:
        """Build the Handoff Package (StatePackage) of this run.
        构建本运行的交接包（StatePackage）。

        Input / 输入：
            goal: Explicit goal (None → the runtime goal).
            goal：显式目标（None → 运行时目标）。

            budget_tokens: Token budget (None → the facade default).
            budget_tokens：token 预算（None → 门面默认值）。

            checkpoint: Take a checkpoint first.
            checkpoint：是否先建立检查点。

        Output / 输出：
            StatePackage.
            StatePackage。

        State effect / 状态影响：
            Only the optional checkpoint writes.
            仅可选的检查点会写入。
        """
        return self.hms.handoff(
            filters=self.filters, goal=goal or self.goal, budget_tokens=budget_tokens, checkpoint=checkpoint
        )

    def checkpoint(self, label: str | None = None) -> str:
        """Take a Rollback Point for this scope.
        为本范围建立回滚点。

        Input / 输入：
            label: Optional label.
            label：可选标签。

        Output / 输出：
            The checkpoint id.
            检查点 id。

        State effect / 状态影响：
            One facade checkpoint() write.
            一次门面 checkpoint() 写入。
        """
        return self.hms.checkpoint(filters=self.filters, label=label)

    def rollback(self, checkpoint_id: str, *, reason: str | None = None) -> dict:
        """Restore this scope to a checkpoint and forget the pending actions.
        把本范围恢复到某检查点并清空未闭合动作。

        Input / 输入：
            checkpoint_id: The checkpoint to restore.
            checkpoint_id：要恢复的检查点。

            reason: Optional reason.
            reason：可选原因。

        Output / 输出：
            The facade rollback() report.
            门面 rollback() 的报告。

        State effect / 状态影响：
            One facade rollback() write; the runtime's open-action table is cleared.
            一次门面 rollback() 写入；运行时的未闭合动作表被清空。
        """
        report = self.hms.rollback(checkpoint_id, reason=reason)
        self._open.clear()
        return report

    def feedback(
        self,
        target_id: str | None,
        feedback: str,
        *,
        reason: str | None = None,
        correction: str | dict | None = None,
    ) -> dict:
        """Apply a state feedback (HMS_arch §3.10) on behalf of this runtime's actor.
        以本运行时的 actor 身份应用一条状态反馈（HMS_arch §3.10）。

        Input / 输入：
            target_id: Target object id (None for state_missing).
            target_id：目标对象 id（state_missing 时为 None）。

            feedback: FeedbackType value.
            feedback：FeedbackType 的值。

            reason: Free-text reason.
            reason：自由文本原因。

            correction: Correction payload.
            correction：纠正载荷。

        Output / 输出：
            The facade feedback() report.
            门面 feedback() 的报告。

        State effect / 状态影响：
            One facade feedback() write.
            一次门面 feedback() 写入。
        """
        return self.hms.feedback(target_id, feedback, reason=reason, correction=correction, actor_id=self.actor_id)
