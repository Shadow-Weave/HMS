"""Scripted LLM for tests: replays a queue of canned responses and records every call.
用于测试的脚本化 LLM：回放预置响应队列并记录每次调用。
"""

# Purpose (EN): Make LLM-dependent code paths testable without heuristics or network.
# 目的（中）：让依赖 LLM 的代码路径无需启发式或网络即可测试。

# Input (EN): A queue of str | dict | Exception | callable responses; chat messages at call time.
# 输入（中）：由 str | dict | Exception | 可调用对象组成的响应队列；调用时的对话消息。

# Output (EN): The next queued response (dict → JSON string); LLMError(LLM_003) when exhausted.
# 输出（中）：下一个队列响应（字典 → JSON 字符串）；队列耗尽时抛出 LLMError(LLM_003)。

# State effect (EN): Mutates its own queue and `calls` list only.
# 状态影响（中）：仅修改自身队列与 `calls` 列表。

from __future__ import annotations

import json
from collections import deque
from typing import Any, ClassVar

from pydantic import Field

from hms_elite.configs.prompts import task_of
from hms_elite.exceptions import LLMError
from hms_elite.llms.base import BaseLlmConfig, LLMBase


class ScriptedLlmConfig(BaseLlmConfig):
    """Scripted provider config: the response queue and its exhaustion policy.
    脚本化提供者配置：响应队列及其耗尽策略。
    """

    responses: list[Any] = Field(default_factory=list)
    loop: bool = False
    raise_when_empty: bool = True
    # Fixed reply of describe_image (None keeps the provider vision-less, like the rule provider).
    # describe_image 的固定回复（None 让提供者保持无视觉，与规则提供者一致）。
    image_description: str | None = None


class ScriptedLLM(LLMBase):
    """Deterministic replay provider: pops one response per call, records `{'task', 'messages', 'kwargs'}`.
    确定性回放提供者：每次调用弹出一个响应，并记录 `{'task', 'messages', 'kwargs'}`。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = ScriptedLlmConfig
    provider_name: ClassVar[str] = "scripted"

    def __init__(self, config: ScriptedLlmConfig | dict | None = None) -> None:
        """Build the queue from config.responses.
        由 config.responses 构建队列。
        """
        super().__init__(config)
        self.responses: deque = deque(self.config.responses)  # type: ignore[attr-defined]
        self.calls: list[dict] = []

    @property
    def identity(self) -> str:
        """The constant identity 'scripted@v1'.
        常量身份 'scripted@v1'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The constant identity string.
            常量身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return "scripted@v1"

    def push(self, *responses: Any) -> None:
        """Append responses to the queue.
        向队列追加响应。

        Input / 输入：
            responses: str | dict | Exception | callable items.
            responses：str | dict | Exception | 可调用对象。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Extends the queue.
            扩展队列。
        """
        self.responses.extend(responses)

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: Chat messages; the system message's HMS_TASK marker is recorded as `task`.
            messages：对话消息；system 消息的 HMS_TASK 标记记录为 `task`。

            tools: Ignored (recorded in kwargs only when given).
            tools：忽略（仅在给定时记录到 kwargs）。

            tool_choice: Ignored.
            tool_choice：忽略。

            kwargs: Recorded verbatim.
            kwargs：原样记录。

        Output / 输出：
            str or dict: next queued item (dict → JSON string; Exception → raised; callable → called with messages).
            文本或字典：下一个队列项（字典 → JSON 字符串；异常 → 抛出；可调用 → 以 messages 调用）。

        State effect / 状态影响：
            Pops the queue (re-queues when loop=True) and appends to `calls`; empty queue → LLMError(LLM_003) or ''.
            弹出队列（loop=True 时重新入队）并追加 `calls`；队列为空 → LLMError(LLM_003) 或 ''。
        """
        system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
        task = task_of(system) if isinstance(system, str) else None
        record = {"task": task, "messages": [dict(m) for m in messages], "kwargs": dict(kwargs)}
        if tools is not None:
            record["kwargs"]["tools"] = tools
        self.calls.append(record)
        if not self.responses:
            if self.config.raise_when_empty:  # type: ignore[attr-defined]
                raise LLMError(
                    "scripted response queue exhausted",
                    "LLM_003",
                    details={"task": task, "provider": self.identity, "calls": len(self.calls)},
                    suggestion="queue more responses",
                )
            return ""
        item = self.responses.popleft()
        if self.config.loop:  # type: ignore[attr-defined]
            self.responses.append(item)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted exception")
        if callable(item):
            item = item(messages)
        if isinstance(item, (dict, list)):
            # 字典与列表都按 JSON 序列化，才能重放"裸列表回复"这类真实模型输出 / dicts and lists both serialise as
            # JSON so a bare-list reply (a real model output shape) can be replayed
            return json.dumps(item, ensure_ascii=False)
        return "" if item is None else str(item)

    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
        """Record the call and reply with the configured `image_description` (None → vision unsupported).
        记录调用并回复配置的 `image_description`（None → 不支持视觉）。

        Input / 输入：
            url_or_data_uri: Recorded under `calls[-1]['kwargs']['url']`.
            url_or_data_uri：记录在 `calls[-1]['kwargs']['url']`。

            detail: Recorded under `calls[-1]['kwargs']['detail']`.
            detail：记录在 `calls[-1]['kwargs']['detail']`。

        Output / 输出：
            config.image_description.
            config.image_description。

        State effect / 状态影响：
            Appends `{'task': 'describe_image', 'messages': [], 'kwargs': {'url', 'detail'}}` to `calls`.
            向 `calls` 追加 `{'task': 'describe_image', 'messages': [], 'kwargs': {'url', 'detail'}}`。
        """
        self.calls.append(
            {"task": "describe_image", "messages": [], "kwargs": {"url": url_or_data_uri, "detail": detail}}
        )
        return self.config.image_description  # type: ignore[attr-defined]


__all__ = ["ScriptedLLM", "ScriptedLlmConfig"]
