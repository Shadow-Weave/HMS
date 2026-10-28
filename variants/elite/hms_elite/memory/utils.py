"""Mem0-ported input helpers: message coercion, scope/metadata building, parameter validation, id masking.
移植自 Mem0 的输入辅助：消息规整、范围/元数据构建、参数校验、id 掩码。
"""

# Purpose (EN): Validate and normalize everything that crosses the public API boundary before it reaches the planes.
# 目的（中）：在进入各平面前校验并归一化跨越公开 API 边界的一切输入。

# Input (EN): Raw messages, ids, metadata, search parameters, expiration dates, texts.
# 输入（中）：原始消息、id、元数据、搜索参数、过期日期、文本。

# Output (EN): Coerced messages, (Scope, metadata) pairs, normalized values; ValidationError on bad input.
# 输出（中）：规整后的消息、(Scope, metadata) 对、归一化的值；输入非法时抛出 ValidationError。

# State effect (EN): None (warnings are logged when reserved metadata keys are dropped).
# 状态影响（中）：无（丢弃保留元数据键时记录警告日志）。

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import logging
import mimetypes
import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from hms_elite.exceptions import ValidationError
from hms_elite.models import RESERVED_ATOM_KEYS, Scope

logger = logging.getLogger(__name__)

IDENTITY_KEYS: tuple[str, ...] = ("user_id", "agent_id", "run_id", "actor_id")
_ID_RE = re.compile(r"\b(evt|anc|obs|ep|atm|mem|ent|edg|iep|fb|ckp|trn|pkg|hst|cst)_[0-9a-f]{6,12}\b")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# RFC 2397: data:[<mediatype>][;base64],<data> — only the base64 form is an image the providers can forward.
# RFC 2397：data:[<mediatype>][;base64],<data> —— 只有 base64 形式才是提供者能转发的图片。
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<params>(?:;[^;,]*)*),(?P<data>.*)$", re.DOTALL)


def parse_data_uri(uri: str) -> tuple[str | None, bytes] | None:
    """Decode a base64 `data:` URI into (mime, bytes); None for anything else (non-base64, malformed, not a data URI).
    将 base64 的 `data:` URI 解码为 (mime, 字节)；其他情况（非 base64、畸形、非 data URI）返回 None。

    Input / 输入：
        uri: Candidate data URI.
        uri：候选的 data URI。

    Output / 输出：
        (lower-cased media type or None, decoded bytes) or None.
        (小写媒体类型或 None, 解码后的字节) 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(uri, str) or uri[:5].lower() != "data:":
        return None
    match = _DATA_URI_RE.match(uri)
    if match is None:
        return None
    params = [p.lower() for p in match.group("params").split(";") if p]
    if "base64" not in params:
        return None
    try:
        raw = base64.b64decode(match.group("data").strip(), validate=False)
    except (binascii.Error, ValueError):
        return None
    mime = match.group("mime").strip().lower() or None
    return mime, raw


def image_ref(url: str, detail: str | None = None) -> dict | None:
    """Describe an image reference as {'url', 'source', 'sha256', 'mime', 'detail'}; None when it is not usable.
    将图片引用描述为 {'url', 'source', 'sha256', 'mime', 'detail'}；不可用时返回 None。

    Input / 输入：
        url: A base64 data URI or an http(s) URL.
        url：base64 data URI 或 http(s) URL。

        detail: Optional provider detail hint ('auto' | 'low' | 'high').
        detail：可选的提供者细节提示（'auto' | 'low' | 'high'）。

    Output / 输出：
        Data URI → sha256 of the decoded bytes, source 'data'; http(s) → sha256 of the URL text, source 'url', mime guessed from the path; anything else → None.
        data URI → 解码字节的 sha256、source 'data'；http(s) → URL 文本的 sha256、source 'url'、mime 按路径猜测；其他 → None。

    State effect / 状态影响：
        None (no network; the URL is never fetched).
        无（无网络；从不抓取 URL）。
    """
    if not isinstance(url, str) or not url:
        return None
    detail = detail if isinstance(detail, str) and detail else None
    decoded = parse_data_uri(url)
    if decoded is not None:
        mime, raw = decoded
        return {"url": url, "source": "data", "sha256": hashlib.sha256(raw).hexdigest(), "mime": mime, "detail": detail}
    if url[:5].lower() == "data:":
        return None
    split = urlsplit(url)
    if split.scheme.lower() not in ("http", "https") or not split.netloc:
        return None
    guessed, _encoding = mimetypes.guess_type(split.path)
    mime = guessed if isinstance(guessed, str) and guessed.startswith("image/") else None
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return {"url": url, "source": "url", "sha256": digest, "mime": mime, "detail": detail}


def _image_part(part: dict) -> dict | None:
    """Recognize an OpenAI `image_url` / `input_image` part or an Anthropic `image` block; None otherwise.
    识别 OpenAI 的 `image_url` / `input_image` 部件或 Anthropic 的 `image` 块；否则返回 None。
    """
    kind = part.get("type")
    url: Any = None
    detail: Any = None
    if kind == "image_url":
        raw = part.get("image_url")
        if isinstance(raw, dict):
            url, detail = raw.get("url"), raw.get("detail")
        elif isinstance(raw, str):
            url = raw
    elif kind == "input_image":
        raw = part.get("image_url")
        url = raw if isinstance(raw, str) else (raw.get("url") if isinstance(raw, dict) else None)
        detail = part.get("detail")
    elif kind == "image":
        source = part.get("source")
        if isinstance(source, dict):
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                media_type = source.get("media_type") if isinstance(source.get("media_type"), str) else ""
                url = f"data:{media_type};base64,{source['data']}"
            elif source.get("type") == "url":
                url = source.get("url")
    if not isinstance(url, str):
        return None
    return image_ref(url, detail if isinstance(detail, str) else None)


class _Unset:
    """Tri-state sentinel type ('not provided' vs None).
    三态哨兵类型（'未提供' 与 None 区分）。
    """

    __slots__ = ()

    def __repr__(self) -> str:  # noqa: D105 - dunder
        return "_UNSET"

    def __bool__(self) -> bool:  # noqa: D105 - dunder
        return False


# 模块级哨兵对象，对外导出
_UNSET: Any = _Unset()  # module sentinel, exported


def _text_parts(content: Any) -> tuple[str | None, int, list[dict]]:
    """Reduce a content value to (text, dropped_parts, image_parts); None text when nothing textual.
    将内容值归约为 (文本, 丢弃分片数, 图片分片)；无文本时文本为 None。
    """
    if content is None:
        return None, 0, []
    if isinstance(content, str):
        return content, 0, []
    if isinstance(content, list):
        parts: list[str] = []
        images: list[dict] = []
        dropped = 0
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
                continue
            # Image parts stay "dropped" from the text (Mem0 parity without vision); the ingest pipeline subtracts
            # the ones a vision-capable provider describes.
            # 图片部件在文本层面仍算"丢弃"（无视觉时与 Mem0 一致）；摄取流水线会减去被视觉提供者描述的那些。
            image = _image_part(part) if isinstance(part, dict) else None
            if image is not None:
                images.append(image)
            dropped += 1
        return "\n".join(parts), dropped, images
    return str(content), 0, []


def parse_messages(messages: list[dict]) -> str:
    """Render messages as 'role: content' lines (Mem0 parse_messages); None content skipped, list content reduced.
    将消息渲染为 'role: content' 行（Mem0 parse_messages）；跳过 None 内容，列表内容归约为文本。

    Input / 输入：
        messages: Chat messages `{'role', 'content'}`.
        messages：聊天消息 `{'role', 'content'}`。

    Output / 输出：
        One line per message, each terminated by a newline.
        每条消息一行，每行以换行结尾。

    State effect / 状态影响：
        None.
        无。
    """
    out = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text, _dropped, _images = _text_parts(message.get("content"))
        if text is None:
            continue
        out.append(f"{message.get('role', 'user')}: {text}\n")
    return "".join(out)


def coerce_messages(messages: str | dict | list) -> list[dict]:
    """Coerce Mem0 `messages` input to a list of {'role', 'content'} dicts (VALIDATION_003 on bad shapes).
    将 Mem0 的 `messages` 输入规整为 {'role', 'content'} 字典列表（形状非法时 VALIDATION_003）。

    Input / 输入：
        messages: str → user message; dict → [dict]; list passthrough; anything else → VALIDATION_003.
        messages：str → 用户消息；dict → [dict]；list 原样；其他 → VALIDATION_003。

    Output / 输出：
        New list; list content becomes joined text plus 'dropped_parts' (and 'image_parts' [{'url', 'source', 'sha256', 'mime', 'detail'}] when image parts were recognized, the url possibly a data URI held in memory only); items with content None are dropped.
        新列表；列表内容变为拼接文本并附 'dropped_parts'（识别到图片部件时还附 'image_parts' [{'url', 'source', 'sha256', 'mime', 'detail'}]，其 url 可能是仅存于内存的 data URI）；content 为 None 的项被丢弃。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(messages, str):
        items: list[Any] = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        items = [messages]
    elif isinstance(messages, list):
        items = messages
    else:
        raise ValidationError(
            f"messages must be a string, a dict, or a list of dicts, got {type(messages).__name__}",
            "VALIDATION_003",
            details={"type": type(messages).__name__},
        )
    out: list[dict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "role" not in item or "content" not in item:
            raise ValidationError(
                f"messages[{index}] must be a dict with 'role' and 'content'",
                "VALIDATION_003",
                details={"index": index},
            )
        text, dropped, images = _text_parts(item["content"])
        if text is None:
            continue
        message = dict(item)
        message["content"] = text
        if isinstance(item["content"], list):
            message["dropped_parts"] = dropped
            if images:
                message["image_parts"] = images
        out.append(message)
    return out


def validate_and_trim_entity_id(value: Any, name: str) -> str | None:
    """Mem0 port: coerce to str and strip; empty or inner whitespace → ValidationError(VALIDATION_001).
    移植自 Mem0：转为字符串并去空白；为空或含内部空白 → ValidationError(VALIDATION_001)。

    Input / 输入：
        value: Raw id (None passes through).
        value：原始 id（None 直接返回）。

        name: Parameter name for the message.
        name：用于消息的参数名。

    Output / 输出：
        Trimmed id or None.
        去空白后的 id 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValidationError(f"Invalid {name}: cannot be empty", "VALIDATION_001", details={"field": name})
    if any(ch.isspace() for ch in text):
        raise ValidationError(f"Invalid {name}: cannot contain whitespace", "VALIDATION_001", details={"field": name})
    return text


def strip_reserved_keys(meta: dict, *, context: str) -> dict:
    """Drop RESERVED_ATOM_KEYS and identity keys from caller metadata with ONE warning listing them.
    从调用方元数据中丢弃 RESERVED_ATOM_KEYS 与身份键，并用一条警告列出它们。

    Input / 输入：
        meta: Caller metadata dict.
        meta：调用方元数据字典。

        context: Operation name for the warning ('add', 'update', ...).
        context：警告中的操作名（'add'、'update'……）。

    Output / 输出：
        New deep-copied dict without the reserved keys.
        不含保留键的新深拷贝字典。

    State effect / 状态影响：
        None (one logger.warning when keys are dropped).
        无（丢弃键时记录一条 logger.warning）。
    """
    dropped = sorted(k for k in meta if k in RESERVED_ATOM_KEYS or k in IDENTITY_KEYS)
    if dropped:
        logger.warning("Ignoring reserved metadata keys in %s: %s", context, ", ".join(dropped))
    return {k: copy.deepcopy(v) for k, v in meta.items() if k not in dropped}


def build_scope_and_metadata(
    *,
    user_id: Any = None,
    agent_id: Any = None,
    run_id: Any = None,
    actor_id: Any = None,
    input_metadata: dict | None = None,
    context: str = "add",
) -> tuple[Scope, dict]:
    """Trim ids, require at least one scope id, and clean caller metadata (Mem0 _build_filters_and_metadata).
    去空白 id、要求至少一个范围 id，并清理调用方元数据（Mem0 _build_filters_and_metadata）。

    Input / 输入：
        user_id: Raw user id.
        user_id：原始用户 id。

        agent_id: Raw agent id.
        agent_id：原始代理 id。

        run_id: Raw run id.
        run_id：原始运行 id。

        actor_id: Raw actor id (validated, never stored in metadata).
        actor_id：原始行动者 id（校验但不写入元数据）。

        input_metadata: Caller metadata (deep-copied; None → {}).
        input_metadata：调用方元数据（深拷贝；None → {}）。

        context: Operation name for warnings and error details.
        context：用于警告与错误细节的操作名。

    Output / 输出：
        (Scope, metadata) — VALIDATION_001 with details={'provided_ids'} when no scope id is set.
        (Scope, metadata)——未设置任何范围 id 时抛出带 details={'provided_ids'} 的 VALIDATION_001。

    State effect / 状态影响：
        None.
        无。
    """
    ids = {
        "user_id": validate_and_trim_entity_id(user_id, "user_id"),
        "agent_id": validate_and_trim_entity_id(agent_id, "agent_id"),
        "run_id": validate_and_trim_entity_id(run_id, "run_id"),
    }
    validate_and_trim_entity_id(actor_id, "actor_id")
    if not any(ids.values()):
        raise ValidationError(
            "At least one of user_id, agent_id, or run_id is required",
            "VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}, "context": context},
            suggestion="Provide at least one of user_id, agent_id, or run_id",
        )
    if input_metadata is None:
        meta: dict = {}
    elif isinstance(input_metadata, dict):
        meta = copy.deepcopy(input_metadata)
    else:
        raise ValidationError(
            f"metadata must be a dict, got {type(input_metadata).__name__}",
            "VALIDATION_000",
            details={"type": type(input_metadata).__name__, "context": context},
        )
    return Scope(**ids), strip_reserved_keys(meta, context=context)


def reject_top_level_entity_params(kwargs: dict, method: str) -> None:
    """Reject user_id/agent_id/run_id passed at top level of a read (VALIDATION_008, is-a ValueError).
    拒绝在读取方法顶层传入 user_id/agent_id/run_id（VALIDATION_008，同时是 ValueError）。

    Input / 输入：
        kwargs: The method's extra keyword arguments.
        kwargs：方法的额外关键字参数。

        method: Method name for the message.
        method：用于消息的方法名。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        None.
        无。
    """
    for name in ("user_id", "agent_id", "run_id"):
        if name in kwargs:
            raise ValidationError(
                f"Top-level {name} is not supported in {method}(); pass it inside filters={{'{name}': ...}}",
                "VALIDATION_008",
                details={"method": method, "param": name},
                suggestion=f"Use filters={{'{name}': ...}}",
            )


def validate_search_params(threshold: float | None, top_k: int | None) -> None:
    """threshold ∈ [0, 1] and top_k a non-negative int (never bool), else VALIDATION_009 (is-a ValueError).
    threshold ∈ [0, 1] 且 top_k 为非负整数（不可为布尔值），否则 VALIDATION_009（同时是 ValueError）。

    Input / 输入：
        threshold: Score threshold or None.
        threshold：得分阈值或 None。

        top_k: Result cap or None.
        top_k：结果上限或 None。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        None.
        无。
    """
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValidationError(
                "threshold must be a number between 0 and 1", "VALIDATION_009", details={"threshold": threshold}
            )
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValidationError("top_k must be a non-negative integer", "VALIDATION_009", details={"top_k": top_k})


def validate_and_trim_query(query: str) -> str:
    """Require a non-empty query string (VALIDATION_010).
    要求非空查询字符串（VALIDATION_010）。

    Input / 输入：
        query: Raw query.
        query：原始查询。

    Output / 输出：
        Stripped query.
        去空白后的查询。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("query must be a non-empty string", "VALIDATION_010", details={"query": query})
    return query.strip()


def normalize_expiration_date(value: Any) -> str | None:
    """None → None; date/datetime/'YYYY-MM-DD' → 'YYYY-MM-DD'; anything else → VALIDATION_004.
    None → None；date/datetime/'YYYY-MM-DD' → 'YYYY-MM-DD'；其他 → VALIDATION_004。

    Input / 输入：
        value: Expiration date input.
        value：过期日期输入。

    Output / 输出：
        'YYYY-MM-DD' or None.
        'YYYY-MM-DD' 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        instant = value.astimezone(timezone.utc) if value.tzinfo is not None else value
        return instant.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if _DATE_RE.match(text):
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError:
                pass
    raise ValidationError(
        "expiration_date must be a valid date in YYYY-MM-DD format.", "VALIDATION_004", details={"value": value}
    )


def estimate_tokens(text: str) -> int:
    """Token estimate (utf8_len + 3) // 4 + 8 (§0.1).
    token 估算 (utf8_len + 3) // 4 + 8（§0.1）。

    Input / 输入：
        text: Any text.
        text：任意文本。

    Output / 输出：
        Integer token estimate.
        整数 token 估算。

    State effect / 状态影响：
        None.
        无。
    """
    return (len(text.encode("utf-8")) + 3) // 4 + 8


def mask_ids(text: str) -> str:
    """Replace every '<prefix>_<hex>' object id with '<prefix>' for golden comparisons.
    将每个 '<prefix>_<hex>' 对象 id 替换为 '<prefix>'，用于黄金比对。

    Input / 输入：
        text: Text possibly containing ids.
        text：可能含 id 的文本。

    Output / 输出：
        Masked text.
        掩码后的文本。

    State effect / 状态影响：
        None.
        无。
    """
    return _ID_RE.sub(lambda m: f"<{m.group(1)}>", text)


__all__ = [
    "IDENTITY_KEYS",
    "_UNSET",
    "build_scope_and_metadata",
    "coerce_messages",
    "estimate_tokens",
    "image_ref",
    "mask_ids",
    "normalize_expiration_date",
    "parse_data_uri",
    "parse_messages",
    "reject_top_level_entity_params",
    "strip_reserved_keys",
    "validate_and_trim_entity_id",
    "validate_and_trim_query",
    "validate_search_params",
]
