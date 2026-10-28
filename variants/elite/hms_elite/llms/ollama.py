"""Ollama chat provider over the local HTTP API (`/api/chat`; stdlib urllib, no SDK, no API key).
Ollama 对话提供者，走本地 HTTP API（`/api/chat`；仅用标准库 urllib，无 SDK，无 API key）。
"""

# Purpose (EN): Drive the HMS prompt contracts through a local Ollama server with Mem0's config conventions.
# 目的（中）：以 Mem0 的配置约定，通过本地 Ollama 服务执行 HMS 提示词契约。

# Input (EN): OllamaLlmConfig (model, ollama_base_url, sampling, timeout), chat messages, optional tools.
# 输入（中）：OllamaLlmConfig（模型、ollama_base_url、采样参数、超时）、对话消息、可选工具。

# Output (EN): Completion text or {'content', 'tool_calls'}; HTTP exceptions propagate to LLMBase.generate_json.
# 输出（中）：补全文本或 {'content', 'tool_calls'}；HTTP 异常传播给 LLMBase.generate_json。

# State effect (EN): Network calls only (loopback by default).
# 状态影响（中）：仅有网络调用（默认走本机回环）。

from __future__ import annotations

import json
import os
from typing import Any, ClassVar
from urllib.request import Request, urlopen

from hms_elite.llms.base import BaseLlmConfig, LLMBase

DEFAULT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"
CHAT_PATH = "/api/chat"


class OllamaLlmConfig(BaseLlmConfig):
    """Ollama provider config (`api_key` is accepted for parity and ignored).
    Ollama 提供者配置（`api_key` 为对齐而保留，实际忽略）。
    """

    model: str | None = DEFAULT_MODEL
    ollama_base_url: str | None = None
    timeout: float = 120.0


def _post_json(url: str, body: dict, timeout: float) -> dict:
    """POST a JSON body with urllib and decode the JSON reply (HTTPError / URLError propagate).
    用 urllib POST 一个 JSON 体并解码 JSON 回复（HTTPError / URLError 直接传播）。
    """
    data = json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaLLM(LLMBase):
    """Ollama chat provider.
    Ollama 对话提供者。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = OllamaLlmConfig
    provider_name: ClassVar[str] = "ollama"

    def __init__(self, config: OllamaLlmConfig | dict | None = None) -> None:
        """Resolve the base url from config.ollama_base_url / OLLAMA_BASE_URL / the localhost default.
        由 config.ollama_base_url / OLLAMA_BASE_URL / 本机默认值解析基础 url。
        """
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        base_url = getattr(self.config, "ollama_base_url", None) or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        """'ollama:<model>'.
        'ollama:<模型>'。

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
        return f"ollama:{self.config.model}"

    def _parse_response(self, response: dict, tools: list[dict] | None) -> str | dict:
        """Mem0 `_parse_response` over the `/api/chat` JSON: text, or {'content', 'tool_calls'} when tools were used.
        针对 `/api/chat` JSON 的 Mem0 `_parse_response`：文本；使用工具时为 {'content', 'tool_calls'}。
        """
        message = response.get("message") or {}
        content = message.get("content")
        if tools:
            processed: dict = {"content": content, "tool_calls": []}
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                processed["tool_calls"].append({"name": function.get("name"), "arguments": arguments})
            return processed
        return content

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: Chat messages sent verbatim as the `/api/chat` `messages` field.
            messages：原样作为 `/api/chat` 的 `messages` 字段发送的对话消息。

            tools: Optional tool definitions (sent as `tools`).
            tools：可选工具定义（作为 `tools` 发送）。

            tool_choice: Accepted for parity; Ollama has no tool_choice field, so it is not sent.
            tool_choice：为对齐而接受；Ollama 没有 tool_choice 字段，故不发送。

            kwargs: `response_format={'type': 'json_object'}` becomes `format='json'`.
            kwargs：`response_format={'type': 'json_object'}` 转为 `format='json'`。

        Output / 输出：
            str or dict per _parse_response.
            按 _parse_response 返回文本或字典。

        State effect / 状态影响：
            One HTTP POST to `<base_url>/api/chat` with `stream=False`.
            向 `<base_url>/api/chat` 发送一次 `stream=False` 的 HTTP POST。
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            },
        }
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            body["format"] = "json"
        if tools:
            body["tools"] = tools
        response = _post_json(f"{self.base_url}{CHAT_PATH}", body, float(self.config.timeout))
        return self._parse_response(response, tools)


__all__ = ["CHAT_PATH", "DEFAULT_BASE_URL", "DEFAULT_MODEL", "OllamaLLM", "OllamaLlmConfig"]
