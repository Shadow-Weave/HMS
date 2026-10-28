"""OpenAI chat-completions provider (optional `openai` extra; DependencyError when the SDK is missing).
OpenAI chat-completions 提供者（可选 `openai` extra；缺少 SDK 时抛出 DependencyError）。
"""

# Purpose (EN): Drive the HMS prompt contracts through the OpenAI SDK with Mem0's client conventions.
# 目的（中）：以 Mem0 的客户端约定，通过 OpenAI SDK 执行 HMS 提示词契约。

# Input (EN): OpenAILlmConfig (model, api_key, base_url, sampling), chat messages, optional tools.
# 输入（中）：OpenAILlmConfig（模型、api_key、base_url、采样参数）、对话消息、可选工具。

# Output (EN): Completion text or {'content', 'tool_calls'}; SDK exceptions propagate to LLMBase.generate_json.
# 输出（中）：补全文本或 {'content', 'tool_calls'}；SDK 异常传播给 LLMBase.generate_json。

# State effect (EN): Network calls; each completion's `usage` is handed to LLMBase's meter via `_last_usage`.
# 状态影响（中）：网络调用；每次补全的 `usage` 经 `_last_usage` 交给 LLMBase 的计量器。

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from hms_elite.exceptions import DependencyError
from hms_elite.llms.base import DESCRIBE_IMAGE_PROMPT, BaseLlmConfig, LLMBase

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

DEFAULT_MODEL = "gpt-5-mini"


class OpenAILlmConfig(BaseLlmConfig):
    """OpenAI provider config.
    OpenAI 提供者配置。
    """

    model: str | None = DEFAULT_MODEL
    openai_base_url: str | None = None


class OpenAILLM(LLMBase):
    """OpenAI chat-completions provider.
    OpenAI chat-completions 提供者。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = OpenAILlmConfig
    provider_name: ClassVar[str] = "openai"

    def __init__(self, config: OpenAILlmConfig | dict | None = None) -> None:
        """Build the client from config.api_key / OPENAI_API_KEY and config.openai_base_url / OPENAI_BASE_URL.
        由 config.api_key / OPENAI_API_KEY 与 config.openai_base_url / OPENAI_BASE_URL 构建客户端。
        """
        if _IMPORT_ERROR is not None:
            raise DependencyError(
                "The 'openai' package is required for the openai provider",
                "DEPS_001",
                details={"package": "openai", "extra": "openai"},
            ) from _IMPORT_ERROR
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = getattr(self.config, "openai_base_url", None) or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def identity(self) -> str:
        """'openai:<model>'.
        'openai:<模型>'。

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
        return f"openai:{self.config.model}"

    def _parse_response(self, response: Any, tools: list[dict] | None) -> str | dict:
        """Mem0 `_parse_response`: text, or {'content', 'tool_calls': [{'name', 'arguments'}]} when tools were used.
        Mem0 `_parse_response`：文本；使用工具时为 {'content', 'tool_calls': [{'name', 'arguments'}]}。
        """
        # The SDK usage (prompt_tokens / completion_tokens) feeds LLMBase's meter; None keeps the estimate path.
        # SDK 用量（prompt_tokens / completion_tokens）喂给 LLMBase 的计量器；为 None 时走估算路径。
        self._last_usage = getattr(response, "usage", None)
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

            kwargs: `response_format` is forwarded when given.
            kwargs：给定 `response_format` 时转发。

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
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
        """Describe one image through a chat.completions `image_url` content part (data URIs pass through as-is).
        通过 chat.completions 的 `image_url` 内容部件描述一张图片（data URI 原样传入）。

        Input / 输入：
            url_or_data_uri: A base64 `data:` URI or an http(s) URL of the image.
            url_or_data_uri：图片的 base64 `data:` URI 或 http(s) URL。

            detail: Forwarded as `image_url.detail` when given ('auto' | 'low' | 'high').
            detail：给定时作为 `image_url.detail` 转发（'auto' | 'low' | 'high'）。

        Output / 输出：
            The stripped reply text, or None when the reply is empty.
            去空白后的回复文本；回复为空时为 None。

        State effect / 状态影响：
            One network call.
            一次网络调用。
        """
        image: dict[str, Any] = {"url": url_or_data_uri}
        if detail:
            image["detail"] = detail
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                        {"type": "image_url", "image_url": image},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        text = content.strip() if isinstance(content, str) else ""
        return text or None


__all__ = ["DEFAULT_MODEL", "OpenAILLM", "OpenAILlmConfig"]
