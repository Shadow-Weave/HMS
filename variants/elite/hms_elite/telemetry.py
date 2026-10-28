"""Token and call metering for LLM / embedding providers plus a context-local request trace (Lite field names).
LLM / 嵌入提供者的 token 与调用计量，以及上下文局部的请求追踪（字段名与 Lite 对齐）。
"""

# Purpose (EN): Give every provider a thread-safe UsageMeter and let callers group provider calls into RequestTrace
#               stages whose names match hms_lite.telemetry (StageMeasurement / RequestTrace / stage names).
# 目的（中）：为每个提供者提供线程安全的 UsageMeter，并让调用方把提供者调用归入 RequestTrace 阶段，
#             阶段命名与 hms_lite.telemetry（StageMeasurement / RequestTrace / 阶段名）一致。

# Input (EN): Token counts and durations from the provider wrappers; provider `usage` objects; chat messages / texts.
# 输入（中）：提供者包装层给出的 token 数与耗时；提供者的 `usage` 对象；对话消息 / 文本。

# Output (EN): Usage snapshots, RequestTrace objects with StageRecord entries, token estimates.
# 输出（中）：Usage 快照、带 StageRecord 条目的 RequestTrace、token 估算值。

# State effect (EN): Mutates only the meters and the active trace; no store, graph or index access.
# 状态影响（中）：只修改计量器与活动追踪；不访问存储、图或索引。

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from hms_elite.memory.utils import estimate_tokens

# Stage names shared with hms_lite.telemetry so an hms4eye adapter can feed either recorder unchanged.
# 与 hms_lite.telemetry 共用的阶段名，使 hms4eye 适配器无需改动即可喂给任一记录器。
STAGE_LLM_GENERATE = "llm.generate"
STAGE_EMBED = "embedding.embed"
STAGE_EMBED_BATCH = "embedding.embed_batch"

# Prefix shared by every embedding stage name, so "what counts as an embedding stage" has a single definition.
# 所有嵌入阶段名共享的前缀，使「什么算嵌入阶段」只有一处定义。
EMBEDDING_STAGE_PREFIX = "embedding."


def is_embedding_stage(operation: str) -> bool:
    """Whether a stage operation name belongs to the embedder (`embedding.embed` / `embedding.embed_batch`).
    某个阶段操作名是否属于嵌入器（`embedding.embed` / `embedding.embed_batch`）。

    Input / 输入：
        operation: A StageRecord.operation value.
        operation：StageRecord.operation 的值。

    Output / 输出：
        True when the name starts with EMBEDDING_STAGE_PREFIX.
        名称以 EMBEDDING_STAGE_PREFIX 开头时为真。

    State effect / 状态影响：
        None.
        无。
    """
    return operation.startswith(EMBEDDING_STAGE_PREFIX)


@dataclass(slots=True)
class Usage:
    """Cumulative provider usage: LLM calls / tokens, embedding calls / texts, wall time.
    累计提供者用量：LLM 调用 / token、嵌入调用 / 文本数、墙钟时间。
    """

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False
    embedding_calls: int = 0
    embedded_texts: int = 0
    duration_ms: float = 0.0

    def merge(self, other: Usage) -> Usage:
        """Sum two usage snapshots into a new one (`estimated` is OR-ed).
        把两个用量快照相加为一个新快照（`estimated` 取或）。

        Input / 输入：
            other: The snapshot to add.
            other：要相加的快照。

        Output / 输出：
            A new Usage with every counter summed.
            每个计数器相加后的新 Usage。

        State effect / 状态影响：
            None (neither operand is modified).
            无（两个操作数都不被修改）。
        """
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated=self.estimated or other.estimated,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            embedded_texts=self.embedded_texts + other.embedded_texts,
            duration_ms=self.duration_ms + other.duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for logging / JSON.
        用于日志 / JSON 的普通字典形式。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `{field: value}` for every field.
            每个字段的 `{字段: 值}`。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "embedding_calls": self.embedding_calls,
            "embedded_texts": self.embedded_texts,
            "duration_ms": self.duration_ms,
        }


class UsageMeter:
    """Thread-safe accumulator behind `provider.usage`.
    `provider.usage` 背后的线程安全累加器。
    """

    def __init__(self) -> None:
        """Start at zero with a private lock.
        从零开始并持有私有锁。
        """
        self._lock = threading.Lock()
        self._usage = Usage()

    def add_llm(self, input_tokens: int, output_tokens: int, *, estimated: bool = False, duration_ms: float = 0.0) -> None:
        """Record one LLM call.
        记录一次 LLM 调用。

        Input / 输入：
            input_tokens: Prompt tokens (exact or estimated).
            input_tokens：提示 token 数（精确或估算）。

            output_tokens: Completion tokens (exact or estimated).
            output_tokens：补全 token 数（精确或估算）。

            estimated: True when the counts came from estimate_tokens rather than the provider.
            estimated：计数来自 estimate_tokens 而非提供者时为真。

            duration_ms: Wall time of the call.
            duration_ms：调用的墙钟耗时。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Increments the meter under its lock.
            在锁内递增计量器。
        """
        with self._lock:
            usage = self._usage
            usage.llm_calls += 1
            usage.input_tokens += int(input_tokens)
            usage.output_tokens += int(output_tokens)
            usage.total_tokens += int(input_tokens) + int(output_tokens)
            usage.estimated = usage.estimated or bool(estimated)
            usage.duration_ms += float(duration_ms)

    def add_embedding(self, n_texts: int, duration_ms: float = 0.0) -> None:
        """Record one embedding call covering `n_texts` texts.
        记录一次覆盖 `n_texts` 段文本的嵌入调用。

        Input / 输入：
            n_texts: Number of texts embedded by the call.
            n_texts：该调用嵌入的文本数量。

            duration_ms: Wall time of the call.
            duration_ms：调用的墙钟耗时。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Increments the meter under its lock.
            在锁内递增计量器。
        """
        with self._lock:
            self._usage.embedding_calls += 1
            self._usage.embedded_texts += int(n_texts)
            self._usage.duration_ms += float(duration_ms)

    def snapshot(self) -> Usage:
        """Copy of the current totals.
        当前总量的副本。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A detached Usage (later adds do not change it).
            分离的 Usage（之后的累加不会改变它）。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return replace(self._usage)

    def reset(self) -> None:
        """Zero every counter.
        将所有计数器归零。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Replaces the totals with a fresh Usage.
            用新的 Usage 替换总量。
        """
        with self._lock:
            self._usage = Usage()


class MeteredProvider:
    """Mixin giving a provider a lazily created `usage` meter and a per-thread `_last_usage` slot.
    为提供者提供懒创建的 `usage` 计量器与线程私有 `_last_usage` 槽位的混入类。
    """

    @property
    def usage(self) -> UsageMeter:
        """The provider's UsageMeter (created lazily, so subclasses that skip `super().__init__` still have one).
        提供者的 UsageMeter（懒创建，跳过 `super().__init__` 的子类同样拥有）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The UsageMeter accumulating every metered call.
            累计每次计量调用的 UsageMeter。

        State effect / 状态影响：
            Creates the meter on first access.
            首次访问时创建计量器。
        """
        meter = self.__dict__.get("_usage_meter")
        if meter is None:
            meter = self.__dict__.setdefault("_usage_meter", UsageMeter())
        return meter

    @usage.setter
    def usage(self, meter: UsageMeter) -> None:
        """Replace the meter (callers may share one meter across providers this way).
        替换计量器（调用方可借此让多个提供者共用一个计量器）。
        """
        self.__dict__["_usage_meter"] = meter

    @property
    def _last_usage(self) -> Any:
        """Per-thread slot where a provider leaves the SDK `usage` object for the wrapper to read.
        线程私有槽位：提供者把 SDK 的 `usage` 对象留在这里供包装层读取。
        """
        return getattr(self._usage_slot, "value", None)

    @_last_usage.setter
    def _last_usage(self, value: Any) -> None:
        """Store the SDK usage object for the current thread only.
        仅为当前线程存放 SDK 用量对象。
        """
        self._usage_slot.value = value

    @property
    def _usage_slot(self) -> threading.local:
        """The lazily created thread-local behind `_last_usage`.
        `_last_usage` 背后按需创建的线程局部对象。
        """
        slot = self.__dict__.get("_usage_local")
        if slot is None:
            slot = self.__dict__.setdefault("_usage_local", threading.local())
        return slot


@dataclass(slots=True)
class StageRecord:
    """One timed provider call inside a RequestTrace (Lite StageMeasurement shape plus provider_calls / estimated).
    RequestTrace 内一次计时的提供者调用（Lite StageMeasurement 形状，外加 provider_calls / estimated）。
    """

    operation: str
    duration_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_calls: int = 1
    estimated: bool = False
    error: str | None = None

    @property
    def name(self) -> str:
        """Lite alias of `operation` (StageMeasurement.name).
        `operation` 的 Lite 别名（StageMeasurement.name）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The stage operation string.
            阶段操作字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return self.operation

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready copy of this stage, one key per field (mirrors Usage.to_dict).
        本阶段的 JSON 就绪副本，每个字段一个键（与 Usage.to_dict 一致）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A new dict; `input_tokens` / `output_tokens` / `total_tokens` keep their None when unknown.
            一个新字典；未知时 `input_tokens` / `output_tokens` / `total_tokens` 保持 None。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "provider_calls": self.provider_calls,
            "estimated": self.estimated,
            "error": self.error,
        }


@dataclass(slots=True)
class RequestTrace:
    """Stages recorded while one operation ran (Lite RequestTrace shape).
    一次操作运行期间记录的阶段（Lite RequestTrace 形状）。
    """

    request_id: str
    operation: str
    stages: list[StageRecord] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        """Sum of stage input tokens.
        各阶段输入 token 之和。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Integer sum (None counts as 0).
            整数和（None 计为 0）。

        State effect / 状态影响：
            None.
            无。
        """
        return sum(stage.input_tokens or 0 for stage in self.stages)

    @property
    def output_tokens(self) -> int:
        """Sum of stage output tokens.
        各阶段输出 token 之和。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Integer sum (None counts as 0).
            整数和（None 计为 0）。

        State effect / 状态影响：
            None.
            无。
        """
        return sum(stage.output_tokens or 0 for stage in self.stages)

    @property
    def total_tokens(self) -> int:
        """Sum of stage total tokens.
        各阶段总 token 之和。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Integer sum (None counts as 0).
            整数和（None 计为 0）。

        State effect / 状态影响：
            None.
            无。
        """
        return sum(stage.total_tokens or 0 for stage in self.stages)

    @property
    def estimated(self) -> bool:
        """True when any stage carries estimated tokens.
        任一阶段的 token 为估算值时为真。

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
        return any(stage.estimated for stage in self.stages)

    def add_stage(self, record: StageRecord) -> StageRecord:
        """Append one stage record.
        追加一条阶段记录。

        Input / 输入：
            record: The stage to append.
            record：要追加的阶段。

        Output / 输出：
            The same record, for chaining.
            同一条记录，便于链式调用。

        State effect / 状态影响：
            Appends to `stages`.
            追加到 `stages`。
        """
        self.stages.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready report of this request: request_id, LLM and embedding tokens split apart, per-stage detail.
        本次请求的 JSON 就绪报告：request_id、分开的 LLM 与嵌入 token、逐阶段明细。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A new dict with keys request_id, llm_input_tokens, llm_output_tokens, embedding_tokens, total_tokens,
            estimated and stages. LLM and embedding tokens are reported separately because they carry different
            prices: an embedder charges for input only, an LLM for input and output alike.
            一个新字典，键为 request_id、llm_input_tokens、llm_output_tokens、embedding_tokens、total_tokens、
            estimated 与 stages。LLM 与嵌入 token 分开报告，因为二者单价不同：嵌入器只按输入计费，
            LLM 则输入输出都计费。

        State effect / 状态影响：
            None.
            无。
        """
        llm_stages = [stage for stage in self.stages if stage.operation == STAGE_LLM_GENERATE]
        return {
            "request_id": self.request_id,
            "llm_input_tokens": sum(stage.input_tokens or 0 for stage in llm_stages),
            "llm_output_tokens": sum(stage.output_tokens or 0 for stage in llm_stages),
            "embedding_tokens": sum(
                stage.input_tokens or 0 for stage in self.stages if is_embedding_stage(stage.operation)
            ),
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "stages": [stage.to_dict() for stage in self.stages],
        }


_ACTIVE_TRACE: ContextVar[RequestTrace | None] = ContextVar("hms_elite_active_trace", default=None)


def _error_text(error: BaseException) -> str:
    """'ExcType: message' as Lite formats it.
    按 Lite 的格式给出 'ExcType: message'。
    """
    return f"{type(error).__name__}: {error}"


def current_trace() -> RequestTrace | None:
    """The trace opened by the innermost active `trace()` in this context, or None.
    当前上下文中最内层活动 `trace()` 打开的追踪；没有时为 None。

    Input / 输入：
        none.
        无。

    Output / 输出：
        RequestTrace | None.
        RequestTrace | None。

    State effect / 状态影响：
        None.
        无。
    """
    return _ACTIVE_TRACE.get()


@contextmanager
def trace(operation: str, request_id: str | None = None) -> Iterator[RequestTrace]:
    """Open a RequestTrace for `operation`; provider calls inside the block append stages to it.
    为 `operation` 打开一个 RequestTrace；块内的提供者调用会向其追加阶段。

    Input / 输入：
        operation: Name of the traced operation (e.g. 'add', 'search').
        operation：被追踪操作的名称（如 'add'、'search'）。

        request_id: Caller-supplied id; a uuid4 hex when None.
        request_id：调用方给定的 id；None 时为 uuid4 十六进制串。

    Output / 输出：
        A context manager yielding the RequestTrace (duration_ms / error filled on exit).
        产出 RequestTrace 的上下文管理器（退出时填好 duration_ms / error）。

    State effect / 状态影响：
        Sets the context-local active trace for the block and restores the previous one on exit.
        在块内设置上下文局部的活动追踪，退出时恢复之前的追踪。
    """
    record = RequestTrace(request_id=request_id or uuid.uuid4().hex, operation=operation)
    token = _ACTIVE_TRACE.set(record)
    started = time.perf_counter()
    try:
        yield record
    except BaseException as error:
        record.error = _error_text(error)
        raise
    finally:
        record.duration_ms = (time.perf_counter() - started) * 1000
        _ACTIVE_TRACE.reset(token)


@contextmanager
def stage(operation: str) -> Iterator[StageRecord | None]:
    """Time one named stage inside the active trace (no-op yielding None without a trace; Lite `recorder.stage`).
    在活动追踪内为一个具名阶段计时（无追踪时为空操作并产出 None；对应 Lite 的 `recorder.stage`）。

    Input / 输入：
        operation: Stage name.
        operation：阶段名。

    Output / 输出：
        A context manager yielding the StageRecord, or None when no trace is active.
        产出 StageRecord 的上下文管理器；没有活动追踪时产出 None。

    State effect / 状态影响：
        Appends the record to the active trace and fills duration_ms / error on exit.
        向活动追踪追加记录，退出时填好 duration_ms / error。
    """
    active = _ACTIVE_TRACE.get()
    if active is None:
        yield None
        return
    record = active.add_stage(StageRecord(operation=operation, provider_calls=0))
    started = time.perf_counter()
    try:
        yield record
    except BaseException as error:
        record.error = _error_text(error)
        raise
    finally:
        record.duration_ms = (time.perf_counter() - started) * 1000


def record_stage(
    operation: str,
    *,
    duration_ms: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated: bool = False,
    error: str | None = None,
) -> StageRecord | None:
    """Append one already-measured provider call to the active trace (used by the provider wrappers).
    把一次已计量的提供者调用追加到活动追踪（供提供者包装层使用）。

    Input / 输入：
        operation: Stage name (llm.generate / embedding.embed / embedding.embed_batch).
        operation：阶段名（llm.generate / embedding.embed / embedding.embed_batch）。

        duration_ms: Measured wall time.
        duration_ms：测得的墙钟耗时。

        input_tokens: Prompt tokens, or None when unknown.
        input_tokens：提示 token 数；未知时为 None。

        output_tokens: Completion tokens, or None when unknown.
        output_tokens：补全 token 数；未知时为 None。

        estimated: True when the tokens were estimated.
        estimated：token 为估算值时为真。

        error: 'ExcType: message' when the call failed.
        error：调用失败时为 'ExcType: message'。

    Output / 输出：
        The appended StageRecord, or None when no trace is active.
        追加的 StageRecord；没有活动追踪时为 None。

    State effect / 状态影响：
        Appends to the active trace only.
        只向活动追踪追加。
    """
    active = _ACTIVE_TRACE.get()
    if active is None:
        return None
    known = (input_tokens or 0) + (output_tokens or 0)
    total = known if (input_tokens is not None or output_tokens is not None) else None
    return active.add_stage(
        StageRecord(
            operation=operation,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            provider_calls=1,
            estimated=estimated,
            error=error,
        )
    )


def _usage_value(usage: Any, *names: str) -> int | None:
    """First present attribute / key among `names` as int (Lite `_usage_value`).
    `names` 中第一个存在的属性 / 键的整数值（Lite 的 `_usage_value`）。
    """
    for name in names:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def usage_tokens(usage: Any) -> tuple[int, int] | None:
    """(input, output) tokens from a provider `usage` object; None when neither count is present.
    从提供者 `usage` 对象取 (输入, 输出) token；两者都缺失时为 None。

    Input / 输入：
        usage: OpenAI (`prompt_tokens` / `completion_tokens`) or Anthropic (`input_tokens` / `output_tokens`) usage.
        usage：OpenAI（`prompt_tokens` / `completion_tokens`）或 Anthropic（`input_tokens` / `output_tokens`）用量。

    Output / 输出：
        Tuple of ints, missing sides as 0; None when the object carries no counts.
        整数二元组，缺失一侧记 0；对象不含任何计数时为 None。

    State effect / 状态影响：
        None.
        无。
    """
    if usage is None:
        return None
    prompt = _usage_value(usage, "prompt_tokens", "input_tokens")
    completion = _usage_value(usage, "completion_tokens", "output_tokens")
    if prompt is None and completion is None:
        return None
    return prompt or 0, completion or 0


def _content_text(content: Any) -> str:
    """Flatten a message content (str or list of parts) into the text that counts toward tokens.
    把消息内容（字符串或部件列表）压平为计入 token 的文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(part, ensure_ascii=False, default=str))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimated prompt tokens: estimate_tokens over every message content.
    估算的提示 token 数：对每条消息内容求 estimate_tokens 之和。

    Input / 输入：
        messages: `[{'role', 'content'}]` chat messages.
        messages：`[{'role', 'content'}]` 对话消息。

    Output / 输出：
        Integer estimate (0 for an empty list).
        整数估算值（空列表为 0）。

    State effect / 状态影响：
        None.
        无。
    """
    return sum(estimate_tokens(_content_text(message.get("content"))) for message in messages)


def estimate_completion_tokens(result: Any) -> int:
    """Estimated completion tokens of a generate_response return value (str or {'content', 'tool_calls'}).
    generate_response 返回值（字符串或 {'content', 'tool_calls'}）的估算补全 token 数。

    Input / 输入：
        result: The provider's return value.
        result：提供者的返回值。

    Output / 输出：
        Integer estimate (text plus serialized tool calls).
        整数估算值（文本加序列化后的工具调用）。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(result, dict):
        text = str(result.get("content") or "")
        calls = result.get("tool_calls")
        if calls:
            text += json.dumps(calls, ensure_ascii=False, default=str)
        return estimate_tokens(text)
    return estimate_tokens(str(result) if result is not None else "")


def usage_of(hms: Any) -> Usage:
    """Merged usage of `hms.llm` and `hms.embedder` (interim entry until `HMS.usage()` exists).
    `hms.llm` 与 `hms.embedder` 的合并用量（`HMS.usage()` 接线前的临时入口）。

    Input / 输入：
        hms: An HMS facade or any object exposing `.llm` / `.embedder` with a `usage` meter.
        hms：HMS 门面，或任何暴露带 `usage` 计量器的 `.llm` / `.embedder` 的对象。

    Output / 输出：
        One Usage summing every meter found (missing providers contribute zero).
        把找到的每个计量器相加得到的 Usage（缺失的提供者贡献为零）。

    State effect / 状态影响：
        None.
        无。
    """
    merged = Usage()
    for attr in ("llm", "embedder"):
        meter = getattr(getattr(hms, attr, None), "usage", None)
        if isinstance(meter, UsageMeter):
            merged = merged.merge(meter.snapshot())
    return merged


__all__ = [
    "STAGE_EMBED",
    "STAGE_EMBED_BATCH",
    "STAGE_LLM_GENERATE",
    "EMBEDDING_STAGE_PREFIX",
    "RequestTrace",
    "StageRecord",
    "MeteredProvider",
    "Usage",
    "UsageMeter",
    "current_trace",
    "estimate_completion_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
    "is_embedding_stage",
    "record_stage",
    "stage",
    "trace",
    "usage_of",
    "usage_tokens",
]
