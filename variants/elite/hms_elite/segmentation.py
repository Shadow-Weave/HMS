"""Split long messages into semantic atoms (sentences, paragraphs, code blocks, list items) with exact spans (HMS §5.1).
把长消息切分为带精确区间的语义原子（句子、段落、代码块、列表项）（HMS §5.1）。
"""

# Purpose (EN): The `segmentation.py` of HMS_arch §5.1 — deterministic EN/ZH boundary detection that never cuts an
#               identifier, URL or path, so extraction can run per atom instead of per message.
# 目的（中）：HMS_arch §5.1 的 `segmentation.py`——确定性的中英边界识别，绝不切断标识符、URL 或路径，
#             使抽取可以按原子而非按整条消息进行。

# Input (EN): Free text (EN/ZH/mixed, optionally with ``` fences and bullet lists) or a list of role/content messages.
# 输入（中）：自由文本（中英/混合，可含 ``` 围栏与项目符号列表）或 role/content 消息列表。

# Output (EN): Segment records: index, exact text, (start, end) source span, kind, language, role, message index.
# 输出（中）：Segment 记录：序号、精确文本、指向源文本的 (start, end) 区间、种类、语言、角色、消息序号。

# State effect (EN): None (pure functions; no store, index or graph access).
# 状态影响（中）：无（纯函数；不访问存储、索引或图）。

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Literal

from hms_elite.utils.text import CJK_RE, IDENT_RE, sentences

__all__ = ["KINDS", "Segment", "detect_lang", "protected_spans", "segment", "segment_messages"]

SegmentKind = Literal["sentence", "paragraph", "code", "list_item"]
KINDS: tuple[str, ...] = ("sentence", "paragraph", "code", "list_item")

# A fence line: up to 3 spaces, ``` or ~~~ (3+), optional info string; the closing fence has no info string.
# 围栏行：至多 3 个空格、``` 或 ~~~（3 个以上）、可选的信息串；闭合围栏没有信息串。
_FENCE_LINE_RE = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$", re.M)
# Blank-line separator between paragraphs.
# 段落之间的空行分隔符。
_PARA_SEP_RE = re.compile(r"\n[ \t]*\n+")
# Bullet / numbered / CJK-numbered list markers at the start of a line.
# 行首的项目符号 / 编号 / 中文编号列表标记。
_BULLET_RE = re.compile(
    r"^[ \t]*(?:[-*+•·]\s+|\d{1,3}[.)]\s+|[(（]\d{1,3}[)）]\s*|[一二三四五六七八九十]+[、.]\s*|\[[ xX]\]\s+)", re.M
)
# CJK terminal punctuation (plus closing quotes / brackets) ends a sentence even without following whitespace.
# CJK 终止标点（含其后的闭合引号 / 括号）即便后面没有空白也结束一个句子。
_CJK_TERMINAL_RE = re.compile(r"[。！？][”」』）)]*")
# Clause cut points: punctuation (kept with the left piece) or a conjunction (kept with the right piece).
# 子句切点：标点（归左段）或连词（归右段）。
_PUNCT_CUT_RE = re.compile(r"[,;:，；：、]\s*")
_CONJ_CUT_RE = re.compile(r"\s+(?=(?:and|but|or|so|because|since|while|whereas|then|unless|although)\s)", re.I)
_WS_RE = re.compile(r"\s+")
# Protected spans: URLs, inline code, absolute / relative paths and identifiers (IDENT_RE of utils.text).
# 受保护区间：URL、行内代码、绝对 / 相对路径与标识符（utils.text 的 IDENT_RE）。
_URL_RE = re.compile(r"(?:https?|ftp|file|s3|git|ssh)://[^\s<>\"'）)\]]+|\bwww\.[^\s<>\"'）)\]]+", re.I)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_PATH_RE = re.compile(r"(?<![\w/])(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)+/?")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Segment:
    """One semantic atom of a text: exact substring, span, kind and language (plus role / message index for chats).
    文本中的一个语义原子：精确子串、区间、种类与语言（对话时另含角色 / 消息序号）。
    """

    index: int
    text: str
    span: tuple[int, int]
    kind: str
    lang: str
    role: str | None = None
    message_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict of the segment.
        分段的 JSON 友好字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with index, text, span (list), kind, lang, role, message_index.
            含 index、text、span（列表）、kind、lang、role、message_index 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "index": self.index,
            "text": self.text,
            "span": [self.span[0], self.span[1]],
            "kind": self.kind,
            "lang": self.lang,
            "role": self.role,
            "message_index": self.message_index,
        }


def detect_lang(text: str) -> str:
    """Language tag of a text: 'zh' when CJK characters dominate, 'en' for Latin text, 'und' when neither is present.
    文本的语言标签：CJK 字符占优为 'zh'，拉丁文本为 'en'，两者皆无为 'und'。

    Input / 输入：
        text: Any text.
        text：任意文本。

    Output / 输出：
        'zh' | 'en' | 'und' (a CJK character counts as a word; 'zh' needs at least as many CJK chars as 2× Latin words).
        'zh' | 'en' | 'und'（一个 CJK 字符按一个词计；'zh' 要求 CJK 字符数至少为拉丁词数的 2 倍）。

    State effect / 状态影响：
        None.
        无。
    """
    cjk = sum(len(m) for m in CJK_RE.findall(text))
    latin = len(_LATIN_WORD_RE.findall(text))
    if cjk and cjk >= 2 * latin:
        return "zh"
    if latin:
        return "en"
    return "zh" if cjk else "und"


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Spans that a cut must never fall inside: URLs, inline code, paths and identifiers.
    切点绝不能落入的区间：URL、行内代码、路径与标识符。

    Input / 输入：
        text: Any text.
        text：任意文本。

    Output / 输出：
        Sorted, merged (start, end) spans.
        已排序并合并的 (start, end) 区间。

    State effect / 状态影响：
        None.
        无。
    """
    raw: list[tuple[int, int]] = []
    for pattern in (_URL_RE, _INLINE_CODE_RE, _PATH_RE, IDENT_RE):
        raw.extend((m.start(), m.end()) for m in pattern.finditer(text))
    raw.sort()
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    """True when `pos` lies strictly inside one of the spans.
    `pos` 严格落在某个区间内部时为真。
    """
    return any(start < pos < end for start, end in spans)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Move the span edges inward over whitespace.
    将区间两端向内收缩，跳过空白。
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _code_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, info) of every fenced code block; an unterminated fence runs to the end of the text.
    每个围栏代码块的 (start, end, info)；未闭合的围栏延伸到文本末尾。
    """
    spans: list[tuple[int, int, str]] = []
    open_at: int | None = None
    fence = ""
    info = ""
    for match in _FENCE_LINE_RE.finditer(text):
        marker = match.group("fence")
        if open_at is None:
            open_at, fence, info = match.start(), marker, match.group("info").strip()
        elif marker[0] == fence[0] and len(marker) >= len(fence) and not match.group("info").strip():
            spans.append((open_at, match.end(), info))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(text), info))
    return spans


def _cut_points(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Valid clause cuts inside text[start:end] as (left_end, right_start), never inside a protected span.
    text[start:end] 内的合法子句切点 (left_end, right_start)，绝不落入受保护区间。
    """
    cuts: list[tuple[int, int]] = []
    for match in _PUNCT_CUT_RE.finditer(text, start, end):
        if _inside(match.start(), spans) or _inside(match.end(), spans):
            continue
        cuts.append((match.end(), match.end()))
    for match in _CONJ_CUT_RE.finditer(text, start, end):
        if _inside(match.start(), spans) or _inside(match.end(), spans):
            continue
        cuts.append((match.start(), match.end()))
    cuts = sorted(set(c for c in cuts if start < c[0] and c[1] < end))
    return cuts


def _whitespace_cuts(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Whitespace fallback cuts inside text[start:end], never inside a protected span.
    text[start:end] 内的空白回退切点，绝不落入受保护区间。
    """
    cuts: list[tuple[int, int]] = []
    for match in _WS_RE.finditer(text, start, end):
        if _inside(match.start(), spans) or _inside(match.end(), spans):
            continue
        if start < match.start() and match.end() < end:
            cuts.append((match.start(), match.end()))
    return cuts


def _split_long(text: str, start: int, end: int, max_chars: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Split text[start:end] into pieces of at most `max_chars` at clause cuts (whitespace as fallback).
    在子句切点（回退为空白）把 text[start:end] 切成不超过 `max_chars` 的片段。
    """
    pieces: list[tuple[int, int]] = []
    pos = start
    while end - pos > max_chars:
        limit = pos + max_chars
        clause = [c for c in _cut_points(text, pos, end, spans) if c[0] <= limit]
        if clause:
            left_end, right_start = clause[-1]
        else:
            spaces = [c for c in _whitespace_cuts(text, pos, end, spans) if c[0] <= limit]
            if spaces:
                left_end, right_start = spaces[-1]
            else:
                # No cut fits: take the first cut after the limit (never inside a protected span) or stop.
                # 没有合适切点：取限长之后的第一个切点（绝不落入受保护区间），否则停止。
                later = _cut_points(text, pos, end, spans) or _whitespace_cuts(text, pos, end, spans)
                if not later:
                    break
                left_end, right_start = later[0]
        piece = _strip_span(text, pos, left_end)
        if piece[1] > piece[0]:
            pieces.append(piece)
        pos = right_start
    last = _strip_span(text, pos, end)
    if last[1] > last[0]:
        pieces.append(last)
    return pieces


def _cjk_split(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split text[start:end] after every CJK terminal that is directly followed by more text.
    在每个后面直接跟有更多文本的 CJK 终止标点之后切分 text[start:end]。
    """
    out: list[tuple[int, int]] = []
    pos = start
    for match in _CJK_TERMINAL_RE.finditer(text, start, end):
        if match.end() >= end:
            break
        piece = _strip_span(text, pos, match.end())
        if piece[1] > piece[0]:
            out.append(piece)
        pos = match.end()
    last = _strip_span(text, pos, end)
    if last[1] > last[0]:
        out.append(last)
    return out


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Spans of utils.text.sentences() over text[start:end] (located sequentially), then CJK terminals split further.
    utils.text.sentences() 在 text[start:end] 上的区间（按顺序定位），再按 CJK 终止标点进一步切分。
    """
    out: list[tuple[int, int]] = []
    pos = start
    for sentence in sentences(text[start:end]):
        at = text.find(sentence, pos, end)
        if at < 0:
            continue
        out.extend(_cjk_split(text, at, at + len(sentence)))
        pos = at + len(sentence)
    return out


def _list_item_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Spans of the items of a bullet / numbered list paragraph (marker stripped, continuation lines attached).
    项目符号 / 编号列表段落中各项的区间（去掉标记，续行并入前一项）。
    """
    items: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = end if newline < 0 else newline
        match = _BULLET_RE.match(text, pos, line_end)
        if match:
            items.append((match.end(), line_end))
        elif items:
            items[-1] = (items[-1][0], line_end)
        else:
            items.append((pos, line_end))
        pos = line_end + 1
    return [_strip_span(text, s, e) for s, e in items if _strip_span(text, s, e)[1] > _strip_span(text, s, e)[0]]


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph spans (blank-line separated) of text[start:end], whitespace-stripped.
    text[start:end] 的段落区间（按空行分隔），已去除首尾空白。
    """
    out: list[tuple[int, int]] = []
    pos = start
    for sep in _PARA_SEP_RE.finditer(text, start, end):
        span = _strip_span(text, pos, sep.start())
        if span[1] > span[0]:
            out.append(span)
        pos = sep.end()
    span = _strip_span(text, pos, end)
    if span[1] > span[0]:
        out.append(span)
    return out


def _prose_segments(text: str, start: int, end: int, max_chars: int, spans: list[tuple[int, int]]) -> list[tuple]:
    """(start, end, kind) atoms of one non-code region: list items, or sentences / paragraphs (long ones cut).
    单个非代码区域的 (start, end, kind) 原子：列表项，或句子 / 段落（过长者切分）。
    """
    out: list[tuple[int, int, str]] = []
    for p_start, p_end in _paragraph_spans(text, start, end):
        first_line_end = text.find("\n", p_start, p_end)
        if _BULLET_RE.match(text, p_start, p_end if first_line_end < 0 else first_line_end):
            for i_start, i_end in _list_item_spans(text, p_start, p_end):
                out.extend((s, e, "list_item") for s, e in _split_long(text, i_start, i_end, max_chars, spans))
            continue
        sents = _sentence_spans(text, p_start, p_end)
        para_text = text[p_start:p_end]
        if len(sents) <= 1 and not re.search(r"[.!?。！？]", para_text):
            out.extend((s, e, "paragraph") for s, e in _split_long(text, p_start, p_end, max_chars, spans))
            continue
        for s_start, s_end in sents:
            out.extend((s, e, "sentence") for s, e in _split_long(text, s_start, s_end, max_chars, spans))
    return out


def segment(text: str, *, max_chars: int = 400, lang: str | None = None) -> list[Segment]:
    """Cut a text into ordered semantic atoms whose spans map back exactly onto the source.
    把文本切成有序语义原子，其区间精确映射回源文本。

    Input / 输入：
        text: Free text (EN/ZH/mixed); ``` / ~~~ fences become `code` atoms, bullet lines `list_item` atoms.
        text：自由文本（中英/混合）；``` / ~~~ 围栏成为 `code` 原子，项目符号行成为 `list_item` 原子。

        max_chars: Longest prose atom; longer sentences are cut at clause boundaries, never inside identifiers/URLs.
        max_chars：散文原子的最大长度；更长的句子在子句边界切分（绝不切入标识符/URL/路径）。

        lang: Force the language tag of prose atoms (None → detect_lang per atom; code atoms always carry 'code').
        lang：强制散文原子的语言标签（None → 逐原子 detect_lang；代码原子恒为 'code'）。

    Output / 输出：
        Segments in source order with index 0..n-1 and text[start:end] == segment.text.
        按源顺序的分段，index 为 0..n-1，且 text[start:end] == segment.text。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(text, str):
        raise TypeError("segment() expects a str")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    spans = protected_spans(text)
    atoms: list[tuple[int, int, str]] = []
    pos = 0
    for c_start, c_end, _info in _code_spans(text):
        atoms.extend(_prose_segments(text, pos, c_start, max_chars, spans))
        atoms.append((c_start, c_end, "code"))
        pos = c_end
    atoms.extend(_prose_segments(text, pos, len(text), max_chars, spans))
    out: list[Segment] = []
    for index, (start, end, kind) in enumerate(atoms):
        piece = text[start:end]
        tag = "code" if kind == "code" else (lang or detect_lang(piece))
        out.append(Segment(index=index, text=piece, span=(start, end), kind=kind, lang=tag))
    return out


def _content_text(content: Any) -> str:
    """Plain text of a message content (str, or list of parts carrying 'text' / 'content').
    消息内容的纯文本（str，或携带 'text' / 'content' 的部件列表）。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") if part.get("text") is not None else part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content)


def segment_messages(messages: Any, *, max_chars: int = 400, lang: str | None = None) -> list[Segment]:
    """Cut every message of a conversation into atoms tagged with the message's role and position.
    把对话中的每条消息切成带角色与位置标签的原子。

    Input / 输入：
        messages: A str, one {'role', 'content'} dict, or a list of them (list content → its text parts joined).
        messages：一个 str、一个 {'role', 'content'} 字典或其列表（列表内容 → 其文本部件拼接）。

        max_chars: Longest prose atom (see segment()).
        max_chars：散文原子的最大长度（见 segment()）。

        lang: Force the language tag (see segment()).
        lang：强制语言标签（见 segment()）。

    Output / 输出：
        Segments with a conversation-wide index, `role`, `message_index`, and spans relative to that message's text.
        带全对话序号、`role`、`message_index` 与相对于该消息文本的区间的分段。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        raise TypeError("segment_messages() expects a str, a dict or a list of dicts")
    out: list[Segment] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {message_index} is not a dict")
        role = message.get("role")
        for seg in segment(_content_text(message.get("content")), max_chars=max_chars, lang=lang):
            out.append(
                replace(
                    seg,
                    index=len(out),
                    role=None if role is None else str(getattr(role, "value", role)),
                    message_index=message_index,
                )
            )
    return out
