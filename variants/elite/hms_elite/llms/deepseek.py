"""DeepSeek chat-completions provider over the OpenAI-compatible endpoint (reuses the `openai` extra).
DeepSeek chat-completions 提供者，走 OpenAI 兼容端点（复用 `openai` extra）。
"""

# Purpose (EN): Drive the HMS prompt contracts through DeepSeek's OpenAI-compatible API, with the `thinking` toggle.
# 目的（中）：通过 DeepSeek 的 OpenAI 兼容 API 执行 HMS 提示词契约，并支持 `thinking` 开关。

# Input (EN): DeepSeekLlmConfig (model, api_key, base_url, sampling, thinking), chat messages, optional tools.
# 输入（中）：DeepSeekLlmConfig（模型、api_key、base_url、采样参数、thinking）、对话消息、可选工具。

# Output (EN): Completion text or {'content', 'tool_calls'}; SDK exceptions propagate to LLMBase.generate_json.
# 输出（中）：补全文本或 {'content', 'tool_calls'}；SDK 异常传播给 LLMBase.generate_json。

# State effect (EN): Network calls; each completion's `usage` (with DeepSeek's prompt_cache_hit_tokens /
#                    prompt_cache_miss_tokens) is handed to LLMBase's meter via `_last_usage` and kept in `last_usage`.
# 状态影响（中）：网络调用；每次补全的 `usage`（含 DeepSeek 的 prompt_cache_hit_tokens / prompt_cache_miss_tokens）
#               经 `_last_usage` 交给 LLMBase 的计量器，并保留在 `last_usage` 中。

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Literal

from hms_elite.exceptions import DependencyError
from hms_elite.llms.base import BaseLlmConfig, LLMBase

# Mem0-style import guard: the check runs at module import; the error is raised when a provider is constructed so
# that the module itself stays importable (test_bilingual_docstrings imports every module).
# Mem0 风格的导入守卫：检查在模块导入时执行，错误则推迟到构造 provider 时抛出，
# 从而让模块本身保持可导入（test_bilingual_docstrings 会导入每一个模块）。
_IMPORT_ERROR: ImportError | None = None
try:
    from openai import OpenAI
except ImportError as _exc:  # pragma: no cover - depends on the environment
    OpenAI = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Usage fields copied off the SDK object: the OpenAI pair plus DeepSeek's context-cache split (hit + miss = prompt).
# 从 SDK 对象复制的用量字段：OpenAI 的两项，加上 DeepSeek 的上下文缓存拆分（命中 + 未命中 = 提示词）。
USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def usage_to_dict(usage: Any) -> dict[str, int] | None:
    """Copy the token counts of an SDK / dict `usage` object into a plain dict (DeepSeek cache fields included).
    把 SDK / 字典形式的 `usage` 对象中的 token 计数复制为普通字典（含 DeepSeek 缓存字段）。

    Input / 输入：
        usage: The SDK CompletionUsage (pydantic, extra fields allowed), a dict, or None.
        usage：SDK 的 CompletionUsage（pydantic，允许额外字段）、字典或 None。

    Output / 输出：
        {field: int} for every USAGE_FIELDS entry that is present and integer-like; None when nothing is present.
        USAGE_FIELDS 中每个存在且可转为整数的字段的 {字段: 整数}；全部缺失时为 None。

    State effect / 状态影响：
        None.
        无。
    """
    if usage is None:
        return None
    extra: dict[str, Any] = {}
    if isinstance(usage, dict):
        extra = usage
    else:
        dump = getattr(usage, "model_dump", None)
        if callable(dump):
            try:
                extra = dict(dump())
            except Exception:  # pragma: no cover - defensive against unusual SDK objects
                extra = {}
        extra = {**extra, **(getattr(usage, "__pydantic_extra__", None) or {})}
    out: dict[str, int] = {}
    for name in USAGE_FIELDS:
        value = extra.get(name) if isinstance(extra, dict) else None
        if value is None and not isinstance(usage, dict):
            value = getattr(usage, name, None)
        if value is None:
            continue
        try:
            out[name] = int(value)
        except (TypeError, ValueError):
            continue
    return out or None


class DeepSeekLlmConfig(BaseLlmConfig):
    """DeepSeek provider config; `thinking` maps to `extra_body={'thinking': {'type': ...}}` when set.
    DeepSeek 提供者配置；设置 `thinking` 时映射为 `extra_body={'thinking': {'type': ...}}`。
    """

    model: str | None = DEFAULT_MODEL
    deepseek_base_url: str | None = None
    thinking: Literal["enabled", "disabled"] | None = None


class DeepSeekLLM(LLMBase):
    """DeepSeek chat-completions provider (OpenAI SDK against api.deepseek.com).
    DeepSeek chat-completions 提供者（OpenAI SDK 指向 api.deepseek.com）。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = DeepSeekLlmConfig
    provider_name: ClassVar[str] = "deepseek"

    def __init__(self, config: DeepSeekLlmConfig | dict | None = None) -> None:
        """Build the client from config.api_key / DEEPSEEK_API_KEY and config.deepseek_base_url / DEEPSEEK_BASE_URL.
        由 config.api_key / DEEPSEEK_API_KEY 与 config.deepseek_base_url / DEEPSEEK_BASE_URL 构建客户端。
        """
        if _IMPORT_ERROR is not None:
            raise DependencyError(
                "The 'openai' package is required for the deepseek provider",
                "DEPS_001",
                details={"package": "openai", "extra": "openai"},
            ) from _IMPORT_ERROR
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = (
            getattr(self.config, "deepseek_base_url", None) or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Last completion's usage dict (cache fields included); unlike `_last_usage` it survives the meter's reset.
        # 最近一次补全的用量字典（含缓存字段）；与 `_last_usage` 不同，它不会被计量器清空。
        self.last_usage: dict[str, int] | None = None

    @property
    def identity(self) -> str:
        """'deepseek:<model>'.
        'deepseek:<模型>'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Identity string.
            身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return f"deepseek:{self.config.model}"

    def _parse_response(self, response: Any, tools: list[dict] | None) -> str | dict:
        """Mem0 `_parse_response`: text, or {'content', 'tool_calls': [{'name', 'arguments'}]} when tools were used.
        Mem0 `_parse_response`：文本；使用工具时为 {'content', 'tool_calls': [{'name', 'arguments'}]}。
        """
        # Exact usage (prompt / completion plus DeepSeek's cache hit / miss split) feeds LLMBase's meter; a missing
        # usage keeps the estimate path.
        # 精确用量（提示 / 补全，加 DeepSeek 的缓存命中 / 未命中拆分）喂给 LLMBase 的计量器；缺失时走估算路径。
        self.last_usage = usage_to_dict(getattr(response, "usage", None))
        self._last_usage = self.last_usage
        message = response.choices[0].message
        if tools:
            processed: dict = {"content": message.content, "tool_calls": []}
            for call in getattr(message, "tool_calls", None) or []:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                processed["tool_calls"].append({"name": call.function.name, "arguments": arguments})
            return processed
        return message.content

    def _extra_body(self, given: dict | None) -> dict | None:
        """Merge the configured `thinking` toggle into the caller's extra_body (hms_lite deepseek_runtime idiom).
        把配置的 `thinking` 开关合并进调用方的 extra_body（hms_lite deepseek_runtime 的做法）。
        """
        extra_body = dict(given or {})
        mode = getattr(self.config, "thinking", None)
        if mode:
            thinking = dict(extra_body.get("thinking") or {})
            thinking["type"] = mode
            extra_body["thinking"] = thinking
        return extra_body or None

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: Chat messages passed verbatim to chat.completions.create.
            messages：原样传给 chat.completions.create 的对话消息。

            tools: Optional tool definitions.
            tools：可选工具定义。

            tool_choice: Tool selection policy.
            tool_choice：工具选择策略。

            kwargs: `response_format` is forwarded when given; `extra_body` is merged with the thinking toggle.
            kwargs：给定 `response_format` 时转发；`extra_body` 与 thinking 开关合并后转发。

        Output / 输出：
            str or dict per _parse_response.
            按 _parse_response 返回文本或字典。

        State effect / 状态影响：
            None (network for real providers).
            无（真实提供者有网络调用）。
        """
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        response_format = kwargs.get("response_format")
        if response_format:
            params["response_format"] = response_format
        extra_body = self._extra_body(kwargs.get("extra_body"))
        if extra_body:
            params["extra_body"] = extra_body
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "USAGE_FIELDS", "DeepSeekLLM", "DeepSeekLlmConfig", "usage_to_dict"]
