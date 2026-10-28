"""LLM provider contract: BaseLlmConfig, LLMBase with generate_response / generate_json / generate_lines / generate_text.
LLM 提供者契约：BaseLlmConfig 与含 generate_response / generate_json / generate_lines / generate_text 的 LLMBase。
"""

# Purpose (EN): Define the one interface every LLM provider (rule, scripted, openai, anthropic) implements.
# 目的（中）：定义所有 LLM 提供者（rule、scripted、openai、anthropic）实现的统一接口。

# Input (EN): Chat messages built by configs.prompts, optional tools, response_format kwargs.
# 输入（中）：由 configs.prompts 构造的对话消息、可选工具、response_format 参数。

# Output (EN): Raw completions (str | dict) and parsed JSON / raw lines / text per task contract; LLMError on transport failure.
# 输出（中）：原始补全（str | dict）以及按任务契约解析的 JSON / 原始行 / 文本；传输失败抛出 LLMError。

# State effect (EN): Every generate_response call is metered into `self.usage` (tokens, wall time) and, when a
#                    telemetry trace is active, appended to it as an 'llm.generate' stage; network for real providers.
# 状态影响（中）：每次 generate_response 调用都计入 `self.usage`（token、墙钟耗时），存在活动追踪时作为
#               'llm.generate' 阶段追加；真实提供者有网络调用。

from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, ClassVar

from pydantic import model_validator

from hms_elite.configs.prompts import ROOT_KEY
from hms_elite.exceptions import LLMError
from hms_elite.models import HMSModel
from hms_elite.telemetry import (
    STAGE_LLM_GENERATE,
    MeteredProvider,
    UsageMeter,
    estimate_completion_tokens,
    estimate_messages_tokens,
    record_stage,
    usage_tokens,
)
from hms_elite.utils.factory import check_extra_fields, coerce_config
from hms_elite.utils.json import parse_llm_json, remove_code_blocks

# Marker set on a wrapped generate_response so re-wrapping (subclass of a subclass) never nests two meters.
# 打在已包装 generate_response 上的标记，避免子类的子类再次包装而叠加两层计量。
_INSTRUMENTED = "__hms_metered__"
# True while a metered generate_response runs in this context, so an override calling super().generate_response
# is one call, not two.
# 计量中的 generate_response 在本上下文运行期间为真，使覆盖方法调用 super().generate_response 只算一次。
_IN_CALL: ContextVar[bool] = ContextVar("hms_elite_llm_in_call", default=False)


def _meter_generate_response(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a provider's generate_response: time it, take exact tokens from `_last_usage` or estimate, feed meter + trace.
    包装提供者的 generate_response：计时，从 `_last_usage` 取精确 token 或估算，喂给计量器与追踪。
    """
    if getattr(original, _INSTRUMENTED, False):
        return original

    @functools.wraps(original)
    def metered(self: LLMBase, messages: list[dict], *args: Any, **kwargs: Any) -> str | dict:
        if _IN_CALL.get():
            return original(self, messages, *args, **kwargs)
        guard = _IN_CALL.set(True)
        self._last_usage = None
        started = time.perf_counter()
        try:
            result = original(self, messages, *args, **kwargs)
        except BaseException as exc:
            # A failed call still cost the prompt; the reply is unknown, so both sides stay estimated.
            # 失败的调用依然消耗了提示词；回复未知，因此两侧都按估算记录。
            duration_ms = (time.perf_counter() - started) * 1000
            prompt = estimate_messages_tokens(messages)
            self.usage.add_llm(prompt, 0, estimated=True, duration_ms=duration_ms)
            record_stage(
                STAGE_LLM_GENERATE,
                duration_ms=duration_ms,
                input_tokens=prompt,
                output_tokens=0,
                estimated=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            _IN_CALL.reset(guard)
        duration_ms = (time.perf_counter() - started) * 1000
        exact = usage_tokens(self._last_usage)
        self._last_usage = None
        if exact is None:
            prompt, completion, estimated = (
                estimate_messages_tokens(messages),
                estimate_completion_tokens(result),
                True,
            )
        else:
            prompt, completion, estimated = exact[0], exact[1], False
        self.usage.add_llm(prompt, completion, estimated=estimated, duration_ms=duration_ms)
        record_stage(
            STAGE_LLM_GENERATE,
            duration_ms=duration_ms,
            input_tokens=prompt,
            output_tokens=completion,
            estimated=estimated,
        )
        return result

    setattr(metered, _INSTRUMENTED, True)
    return metered

# Mem0's vision prompt: the description replaces the image in the message text before extraction.
# Mem0 的视觉提示词：描述文本在抽取前替代消息中的图片。
DESCRIBE_IMAGE_PROMPT = (
    "A user is providing an image. Provide a high level description of the image and do not include any "
    "additional text."
)


class BaseLlmConfig(HMSModel):
    """Fields shared by every LLM provider config (Mem0 BaseLlmConfig parity).
    所有 LLM 提供者配置共享的字段（与 Mem0 BaseLlmConfig 对齐）。
    """

    model: str | None = None
    temperature: float = 0.1
    api_key: str | None = None
    max_tokens: int = 2000
    top_p: float = 0.1
    top_k: int = 1
    http_client_proxies: dict | str | None = None

    @model_validator(mode="before")
    @classmethod
    def _extra_fields(cls, data: Any) -> Any:
        """Unknown keys → ConfigError(CONFIG_001) with a did-you-mean hint.
        未知键 → 带 did-you-mean 提示的 ConfigError(CONFIG_001)。
        """
        return check_extra_fields(cls, data)


class LLMBase(MeteredProvider, ABC):
    """Abstract LLM provider: one raw completion primitive plus the JSON / lines / text task helpers.
    抽象 LLM 提供者：一个原始补全原语，加上 JSON / 行式 / 文本任务辅助方法。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = BaseLlmConfig
    provider_name: ClassVar[str] = "base"

    def __init__(self, config: BaseLlmConfig | dict | None = None) -> None:
        """Coerce the config into the provider's config class (dict → class, None → defaults); start a UsageMeter.
        将配置规整为提供者的配置类（字典 → 类，None → 默认值）；启动一个 UsageMeter。
        """
        self.config: BaseLlmConfig = coerce_config(self.config_class, config)  # type: ignore[assignment]
        self.usage = UsageMeter()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Meter the subclass's own generate_response (signature and return value unchanged).
        为子类自己的 generate_response 加计量（签名与返回值不变）。
        """
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("generate_response")
        if callable(own):
            cls.generate_response = _meter_generate_response(own)  # type: ignore[method-assign]

    @abstractmethod
    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: `[{'role', 'content'}]` chat messages (system first for HMS tasks).
            messages：`[{'role', 'content'}]` 对话消息（HMS 任务以 system 开头）。

            tools: Optional tool definitions (Mem0 contract).
            tools：可选的工具定义（Mem0 契约）。

            tool_choice: Tool selection policy when tools are given.
            tool_choice：给定工具时的工具选择策略。

            kwargs: Provider options such as `response_format={'type': 'json_object'}`.
            kwargs：提供者选项，如 `response_format={'type': 'json_object'}`。

        Output / 输出：
            str, or `{'content': str | None, 'tool_calls': [...]}` when tools were used.
            文本；使用工具时为 `{'content': str | None, 'tool_calls': [...]}`。

        State effect / 状态影响：
            None (network for real providers).
            无（真实提供者有网络调用）。
        """

    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
        """Vision hook: describe one image as text; None means the provider has no vision (the default).
        视觉钩子：把一张图片描述为文字；返回 None 表示提供者不支持视觉（默认）。

        Input / 输入：
            url_or_data_uri: A base64 `data:` URI or an http(s) URL of the image.
            url_or_data_uri：图片的 base64 `data:` URI 或 http(s) URL。

            detail: Provider detail hint ('auto' | 'low' | 'high'); providers without the notion ignore it.
            detail：提供者细节提示（'auto' | 'low' | 'high'）；无此概念的提供者忽略。

        Output / 输出：
            The description text, or None when unsupported / empty (the part is then counted in dropped_parts).
            描述文本；不支持或为空时为 None（该部件随后计入 dropped_parts）。

        State effect / 状态影响：
            None here; vision providers perform one network call per image.
            此处无；视觉提供者每张图片一次网络调用。
        """
        return None

    @property
    def identity(self) -> str:
        """Provider identity string such as 'rule@v1' or 'openai:gpt-5-mini'.
        提供者身份字符串，如 'rule@v1' 或 'openai:gpt-5-mini'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `f"{provider_name}:{model}"` by default; subclasses override.
            默认 `f"{provider_name}:{model}"`；子类可覆盖。

        State effect / 状态影响：
            None.
            无。
        """
        return f"{self.provider_name}:{self.config.model}"

    def _call(self, task: str, messages: list[dict], **kwargs: Any) -> str | dict:
        """Run generate_response and wrap every exception as LLMError (code preserved for LLMError causes).
        执行 generate_response 并把所有异常包装为 LLMError（LLMError 原因保留其错误码）。
        """
        try:
            return self.generate_response(messages, **kwargs)
        except Exception as exc:
            code = exc.error_code if isinstance(exc, LLMError) else "LLM_001"
            raise LLMError(f"{task} failed: {exc}", code, details={"task": task, "provider": self.identity}) from exc

    def generate_json(self, task: str, *, system: str, user: str) -> list | dict:
        """Run one JSON task (system + user prompt) and parse the reply through the Mem0 ladder.
        执行一个 JSON 任务（system + user 提示词）并通过 Mem0 阶梯解析回复。

        Input / 输入：
            task: Task name (key of configs.prompts.ROOT_KEY).
            task：任务名（configs.prompts.ROOT_KEY 的键）。

            system: The `<TASK>_PROMPT` system prompt.
            system：`<TASK>_PROMPT` 系统提示词。

            user: The built user prompt.
            user：构造好的用户提示词。

        Output / 输出：
            The list/dict under the task's root key ([] on unparsable output).
            任务根键下的列表/字典（无法解析时为 []）。

        State effect / 状态影响：
            None; transport errors raise LLMError(LLM_001, details={'task', 'provider'}) from the cause.
            无；传输错误抛出 LLMError(LLM_001, details={'task', 'provider'}) 并保留原因。
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = self._call(task, messages, response_format={"type": "json_object"})
        if isinstance(raw, dict):
            raw = raw.get("content") or ""
        return parse_llm_json(raw if raw is not None else "", key=ROOT_KEY.get(task), task=task)

    def generate_lines(self, task: str, *, system: str, user: str) -> str:
        """Run one plain-text task (memory_extraction_lines, CONTRACT §6.6) and return the raw reply; NO JSON mode.
        执行一个纯文本任务（memory_extraction_lines，CONTRACT §6.6）并返回原始回复；不使用 JSON 模式。

        Input / 输入：
            task: Task name (a COMPOSITE_TASKS member; used in error details).
            task：任务名（COMPOSITE_TASKS 成员；用于错误详情）。

            system: The `<TASK>_PROMPT` system prompt.
            system：`<TASK>_PROMPT` 系统提示词。

            user: The built user prompt.
            user：构造好的用户提示词。

        Output / 输出：
            The reply text as the provider returned it (a tool-call dict yields its `content`); '' for None. The
            call carries no `response_format` — a JSON mode would make the model wrap the lines in an object — and
            the caller (ingest.lines.parse_lines) strips fences and prose itself.
            提供者返回的原始回复文本（工具调用字典取其 `content`）；None 时为 ''。调用不带 `response_format`——
            JSON 模式会让模型把各行包进对象——围栏与散文由调用方（ingest.lines.parse_lines）自行剥离。

        State effect / 状态影响：
            None; transport errors raise LLMError(LLM_001, details={'task', 'provider'}) from the cause.
            无；传输错误抛出 LLMError(LLM_001, details={'task', 'provider'}) 并保留原因。
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = self._call(task, messages)
        if isinstance(raw, dict):
            raw = raw.get("content") or ""
        return "" if raw is None else str(raw)

    def generate_text(self, task: str, *, messages: list[dict]) -> str:
        """Run a text task (procedural_summary) and strip code fences from the reply.
        执行文本任务（procedural_summary）并去除回复中的代码围栏。

        Input / 输入：
            task: Task name used in error details.
            task：错误详情中使用的任务名。

            messages: Full message list (system prompt included).
            messages：完整消息列表（含系统提示词）。

        Output / 输出：
            The reply text after remove_code_blocks.
            经 remove_code_blocks 处理后的回复文本。

        State effect / 状态影响：
            None; transport errors raise LLMError from the cause.
            无；传输错误抛出 LLMError 并保留原因。
        """
        raw = self._call(task, messages)
        if isinstance(raw, dict):
            raw = raw.get("content") or ""
        return remove_code_blocks(raw if raw is not None else "")


__all__ = ["DESCRIBE_IMAGE_PROMPT", "ROOT_KEY", "BaseLlmConfig", "LLMBase"]
