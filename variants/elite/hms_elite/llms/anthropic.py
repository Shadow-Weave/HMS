"""Anthropic messages provider (optional `anthropic` extra; DependencyError when the SDK is missing).
Anthropic messages 提供者（可选 `anthropic` extra；缺少 SDK 时抛出 DependencyError）。
"""

# Purpose (EN): Drive the HMS prompt contracts through the Anthropic SDK (system prompt split out of messages).
# 目的（中）：通过 Anthropic SDK 执行 HMS 提示词契约（系统提示词从消息中拆出）。

# Input (EN): AnthropicLlmConfig (model, api_key, base_url, sampling), chat messages, optional tools.
# 输入（中）：AnthropicLlmConfig（模型、api_key、base_url、采样参数）、对话消息、可选工具。

# Output (EN): Completion text or {'content', 'tool_calls'}; SDK exceptions propagate to LLMBase.generate_json.
# 输出（中）：补全文本或 {'content', 'tool_calls'}；SDK 异常传播给 LLMBase.generate_json。

# State effect (EN): Network calls; each reply's `usage` is handed to LLMBase's meter via `_last_usage`.
# 状态影响（中）：网络调用；每次回复的 `usage` 经 `_last_usage` 交给 LLMBase 的计量器。

from __future__ import annotations

import inspect
import os
import re
from typing import Any, ClassVar

from hms_elite.exceptions import DependencyError
from hms_elite.llms.base import DESCRIBE_IMAGE_PROMPT, BaseLlmConfig, LLMBase

# Mem0-style import guard: the check runs at module import; the error is raised when a provider is constructed so
# that the module itself stays importable (test_bilingual_docstrings imports every module).
# Mem0 风格的导入守卫：检查在模块导入时执行，错误则推迟到构造 provider 时抛出，
# 从而让模块本身保持可导入（test_bilingual_docstrings 会导入每一个模块）。
_IMPORT_ERROR: ImportError | None = None
try:
    import anthropic
except ImportError as _exc:  # pragma: no cover - depends on the environment
    anthropic = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

DEFAULT_MODEL = "claude-sonnet-4-6"
JSON_SUFFIX = "\nRespond with JSON only."
# RFC 2397 data URI: the base64 body is forwarded to the Messages API as a base64 image source.
# RFC 2397 data URI：base64 正文作为 base64 图片来源转发给 Messages API。
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<params>(?:;[^;,]*)*),(?P<data>.*)$", re.DOTALL)


def _accepts_keyword(func: Any, name: str) -> bool:
    """True when `func` takes keyword `name` (explicitly or through **kwargs); False when its signature is unreadable.
    当 `func` 接受关键字 `name`（显式声明或经 **kwargs）时为真；签名无法读取时为假。
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class AnthropicLlmConfig(BaseLlmConfig):
    """Anthropic provider config.
    Anthropic 提供者配置。
    """

    model: str | None = DEFAULT_MODEL
    anthropic_base_url: str | None = None


class AnthropicLLM(LLMBase):
    """Anthropic messages provider.
    Anthropic messages 提供者。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = AnthropicLlmConfig
    provider_name: ClassVar[str] = "anthropic"

    def __init__(self, config: AnthropicLlmConfig | dict | None = None) -> None:
        """Build the client from config.api_key / ANTHROPIC_API_KEY and config.anthropic_base_url / ANTHROPIC_BASE_URL.
        由 config.api_key / ANTHROPIC_API_KEY 与 config.anthropic_base_url / ANTHROPIC_BASE_URL 构建客户端。
        """
        if _IMPORT_ERROR is not None:
            raise DependencyError(
                "The 'anthropic' package is required for the anthropic provider",
                "DEPS_001",
                details={"package": "anthropic", "extra": "anthropic"},
            ) from _IMPORT_ERROR
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = getattr(self.config, "anthropic_base_url", None) or os.getenv("ANTHROPIC_BASE_URL")
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        # anthropic>=1.0 removed the sampling parameters from Messages.create (passing one raises TypeError before any
        # HTTP request), so `temperature` is forwarded only when the installed SDK still declares it.
        # anthropic>=1.0 已从 Messages.create 移除采样参数（传入即在任何 HTTP 请求之前抛出 TypeError），
        # 因此仅当已安装的 SDK 仍声明 `temperature` 时才转发它。
        self._pass_temperature = _accepts_keyword(self.client.messages.create, "temperature")

    @property
    def identity(self) -> str:
        """'anthropic:<model>'.
        'anthropic:<模型>'。

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
        return f"anthropic:{self.config.model}"

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Join system messages into one `system=` string; return the remaining messages (copied).
        将 system 消息合并为一个 `system=` 字符串；返回其余消息（拷贝）。
        """
        system_parts: list[str] = []
        rest: list[dict] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(str(message.get("content") or ""))
            else:
                rest.append(dict(message))
        return "\n".join(system_parts), rest

    def _parse_response(self, response: Any, tools: list[dict] | None) -> str | dict:
        """Text of the reply, or {'content', 'tool_calls'} built from tool_use blocks when tools were used.
        回复文本；使用工具时由 tool_use 块构造 {'content', 'tool_calls'}。
        """
        # The SDK usage (input_tokens / output_tokens) feeds LLMBase's meter; None keeps the estimate path.
        # SDK 用量（input_tokens / output_tokens）喂给 LLMBase 的计量器；为 None 时走估算路径。
        self._last_usage = getattr(response, "usage", None)
        blocks = list(getattr(response, "content", None) or [])
        texts = [str(getattr(b, "text", "")) for b in blocks if getattr(b, "type", "text") == "text"]
        text = "".join(texts)
        if tools:
            calls = [
                {"name": getattr(b, "name", ""), "arguments": getattr(b, "input", {}) or {}}
                for b in blocks
                if getattr(b, "type", "") == "tool_use"
            ]
            return {"content": text or None, "tool_calls": calls}
        return text

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: Chat messages; system messages are split into the `system=` argument.
            messages：对话消息；system 消息拆到 `system=` 参数。

            tools: Optional tool definitions (forwarded as `tools=`).
            tools：可选工具定义（作为 `tools=` 转发）。

            tool_choice: Tool selection policy ('auto' | 'any' | tool name).
            tool_choice：工具选择策略（'auto' | 'any' | 工具名）。

            kwargs: A JSON `response_format` appends "Respond with JSON only." to the last user message.
            kwargs：JSON `response_format` 会在最后一条 user 消息后追加 "Respond with JSON only."。

        Output / 输出：
            str or dict per _parse_response (config.temperature is sent only when the SDK's Messages.create accepts it).
            按 _parse_response 返回文本或字典（仅当 SDK 的 Messages.create 接受时才发送 config.temperature）。

        State effect / 状态影响：
            None (network for real providers).
            无（真实提供者有网络调用）。
        """
        system, rest = self._split_system(messages)
        response_format = kwargs.get("response_format") or {}
        if isinstance(response_format, dict) and "json" in str(response_format.get("type", "")).lower():
            for message in reversed(rest):
                if message.get("role") == "user":
                    message["content"] = str(message.get("content") or "") + JSON_SUFFIX
                    break
        params: dict[str, Any] = {
            "model": self.config.model,
            "system": system,
            "messages": rest,
            "max_tokens": self.config.max_tokens,
        }
        if self._pass_temperature:
            params["temperature"] = self.config.temperature
        if tools:
            params["tools"] = tools
            if tool_choice == "auto":
                params["tool_choice"] = {"type": "auto"}
            elif tool_choice == "any":
                params["tool_choice"] = {"type": "any"}
            else:
                params["tool_choice"] = {"type": "tool", "name": tool_choice}
        response = self.client.messages.create(**params)
        return self._parse_response(response, tools)

    @staticmethod
    def _image_source(url_or_data_uri: str) -> dict | None:
        """Messages API image source: base64 data URI → {'type': 'base64', 'media_type', 'data'}; http(s) → {'type': 'url', 'url'}.
        Messages API 的图片来源：base64 data URI → {'type': 'base64', 'media_type', 'data'}；http(s) → {'type': 'url', 'url'}。
        """
        if url_or_data_uri[:5].lower() == "data:":
            match = _DATA_URI_RE.match(url_or_data_uri)
            if match is None or "base64" not in match.group("params").lower():
                return None
            media_type = match.group("mime").strip().lower() or "image/jpeg"
            return {"type": "base64", "media_type": media_type, "data": match.group("data").strip()}
        if url_or_data_uri[:7].lower() == "http://" or url_or_data_uri[:8].lower() == "https://":
            return {"type": "url", "url": url_or_data_uri}
        return None

    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
        """Describe one image through a Messages API `image` block followed by the vision prompt.
        通过 Messages API 的 `image` 块加视觉提示词描述一张图片。

        Input / 输入：
            url_or_data_uri: A base64 `data:` URI (sent as a base64 source) or an http(s) URL (sent as a url source).
            url_or_data_uri：base64 `data:` URI（作为 base64 来源发送）或 http(s) URL（作为 url 来源发送）。

            detail: Ignored (the Messages API has no detail level).
            detail：忽略（Messages API 没有细节等级）。

        Output / 输出：
            The concatenated text blocks, or None when the reference is unusable or the reply is empty.
            拼接后的文本块；引用不可用或回复为空时为 None。

        State effect / 状态影响：
            One network call (none when the reference is unusable).
            一次网络调用（引用不可用时无调用）。
        """
        source = self._image_source(url_or_data_uri)
        if source is None:
            return None
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": source},
                        {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
        }
        if self._pass_temperature:
            params["temperature"] = self.config.temperature
        response = self.client.messages.create(**params)
        text = self._parse_response(response, None)
        text = text.strip() if isinstance(text, str) else ""
        return text or None


__all__ = ["DEFAULT_MODEL", "JSON_SUFFIX", "AnthropicLLM", "AnthropicLlmConfig"]
