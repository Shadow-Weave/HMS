"""Canonical JSON and the Mem0 parse ladder for LLM output.
规范 JSON 与 Mem0 风格的 LLM 输出解析阶梯。
"""

# Purpose (EN): Deterministic serialization plus tolerant JSON extraction from model replies.
# 目的（中）：确定性序列化，以及对模型回复的容错 JSON 提取。

# Input (EN): Python objects; raw LLM strings (possibly fenced or prose-wrapped).
# 输入（中）：Python 对象；原始 LLM 字符串（可能带围栏或散文包裹）。

# Output (EN): Canonical JSON strings; parsed lists/dicts (never raises).
# 输出（中）：规范 JSON 字符串；解析后的列表/字典（从不抛出）。

# State effect (EN): None; parse failures are logged.
# 状态影响（中）：无；解析失败写入日志。

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
DICT_ROOT_KEYS: frozenset[str] = frozenset({"interface_ep"})


def canonical_json(obj: Any) -> str:
    """Serialize deterministically: sorted keys, no spaces, unicode kept.
    确定性序列化：键排序、无空格、保留 unicode。

    Input / 输入：
        obj: Any JSON-serializable object.
        obj：任意可 JSON 序列化的对象。

    Output / 输出：
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))。

    State effect / 状态影响：
        None.
        无。
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def remove_code_blocks(content: str | list[str]) -> str:
    """Strip a whole-string code fence and <think> blocks (Mem0 verbatim behaviour).
    去除整串代码围栏与 <think> 块（与 Mem0 行为一致）。

    Input / 输入：
        content: Raw string, or a list of string blocks joined with newlines.
        content：原始字符串，或用换行拼接的字符串块列表。

    Output / 输出：
        The inner text of the fence when the whole string is fenced, else the stripped content.
        整串被围栏包裹时返回内部文本，否则返回去除首尾空白的内容。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(content, list):
        content = "\n".join(str(c) for c in content)
    content = "" if content is None else str(content)
    content = _THINK_RE.sub("", content).strip()
    match = _FENCE_RE.match(content.strip())
    return match.group(1).strip() if match else content.strip()


def extract_json(text: str) -> str:
    """Pull the most plausible JSON fragment out of prose.
    从散文中提取最可能的 JSON 片段。

    Input / 输入：
        text: Raw model output.
        text：原始模型输出。

    Output / 输出：
        First fenced ```json block, else the outermost {...} / [...] span (whichever opens first), else the text.
        首个 ```json 围栏块；否则最外层 {...} / [...]（以先出现者为准）；否则原文。

    State effect / 状态影响：
        None.
        无。
    """
    text = text.strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    brace, bracket = text.find("{"), text.find("[")
    openers: list[tuple[int, str, str]] = []
    if brace != -1:
        openers.append((brace, "{", "}"))
    if bracket != -1:
        openers.append((bracket, "[", "]"))
    for start, _open, close in sorted(openers):
        end = text.rfind(close)
        if end > start:
            return text[start : end + 1]
    return text


def parse_llm_json(raw: str, *, key: str | None, task: str = "") -> list | dict:
    """Parse an LLM reply through the ladder: fence strip → json.loads → extract_json → [] on failure.
    通过阶梯解析 LLM 回复：去围栏 → json.loads → extract_json → 失败返回 []。

    Input / 输入：
        raw: The model's text reply.
        raw：模型文本回复。

        key: Root key to select (list keys, or 'interface_ep' for a dict root); None returns the whole object.
        key：要选取的根键（列表键，或 'interface_ep' 这样的字典根）；None 返回整个对象。

        task: Task name used in the error log line.
        task：错误日志中使用的任务名。

    Output / 输出：
        The selected list/dict; [] for blanks, garbage, or a non-list value under a list key. Never raises.
        选取的列表/字典；空白、垃圾或列表键下非列表值返回 []。从不抛出。

    State effect / 状态影响：
        None (logs an error on parse failure).
        无（解析失败时记录错误日志）。
    """
    text = remove_code_blocks(raw if raw is not None else "")
    if not text.strip():
        return []
    try:
        obj = json.loads(text, strict=False)
    except json.JSONDecodeError:
        try:
            obj = json.loads(extract_json(text), strict=False)
        except Exception:
            logger.error("Error parsing %s response", task)
            return []
    except Exception:
        logger.error("Error parsing %s response", task)
        return []
    if isinstance(obj, dict):
        if key is None:
            return obj
        value = obj.get(key, [])
        if key in DICT_ROOT_KEYS:
            return value if isinstance(value, dict) else []
        return value if isinstance(value, list) else []
    if isinstance(obj, list):
        return obj if key is None else []
    return []


__all__ = ["DICT_ROOT_KEYS", "canonical_json", "extract_json", "parse_llm_json", "remove_code_blocks"]
