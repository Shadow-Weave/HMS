"""Parser of the lines extraction contract: tab-separated OBS / EP / DEC text into compact-shaped items.
行式抽取契约的解析器：把 tab 分隔的 OBS / EP / DEC 文本解析为紧凑形状的条目。
"""

# Purpose (EN): Turn the plain-text reply of the memory_extraction_lines task (CONTRACT §6.6) into the same
# {'observations', 'eps', 'decisions'} dict the compact path consumes, so ingest.extraction shares every step after
# the parse (witness, ingest.derive expansion, the full-contract coercion, the decision join). Tolerant by design: a
# model that wraps the reply in a code fence, prefixes prose, forgets a tab, misspells a type letter or is cut off
# mid-line loses at most that line, never the batch.
# 目的（中）：把 memory_extraction_lines 任务（CONTRACT §6.6）的纯文本回复变成紧凑路径所消费的同一个
# {'observations', 'eps', 'decisions'} 字典，使 ingest.extraction 在解析之后的每一步（见证、ingest.derive 扩展、
# 完整契约规整、决策连接）都共用。刻意容错：模型用代码围栏包裹、前置散文、漏掉一个 tab、写错类型字母或在行中被
# 截断时，最多只丢那一行，绝不丢整批。

# Input (EN): The raw reply text.
# 输入（中）：原始回复文本。

# Output (EN): {'observations': [compact items with id = the 1-based OBS line number — every tabbed line of the OBS
#              section consumes one number whether or not it stores anything, so a DEC line's number always names
#              the line the model wrote], 'eps': [EP items],
#              'decisions': [{'id': OBS line number, 'decision', 'target_id', 'reason'}]}.
# 输出（中）：{'observations': [id 为 1 起 OBS 行号的紧凑条目——OBS 分节里每个带 tab 的行都占用一个行号，无论是否存下
#             内容，因此 DEC 行的行号总是指向模型所写的那一行], 'eps': [EP 条目],
#             'decisions': [{'id': OBS 行号, 'decision', 'target_id', 'reason'}]}。

# State effect (EN): None (pure); malformed lines are logged at WARNING / DEBUG; a reply with no section at all logs
#                    one WARNING containing NO_SECTION_MARK (the harness's "unparsable reply" signal).
# 状态影响（中）：无（纯函数）；畸形行记 WARNING / DEBUG 日志；完全没有分节的回复记一条含 NO_SECTION_MARK 的
#                 WARNING（基准脚本据此识别"不可用的回复"）。

from __future__ import annotations

import logging
import re
from typing import Any

from hms_elite.configs.prompts import LINE_TYPES, TYPE_LETTERS

logger = logging.getLogger(__name__)

# The one-letter type vocabulary of an OBS line and its inverse live with the contract (configs.prompts, group A)
# so the rule LLM's line rendering and this parser cannot drift apart.
# OBS 行的单字母类型词表及其逆映射与契约同住（configs.prompts，A 组），使规则 LLM 的行渲染与本解析器不会分叉。
SECTIONS = ("OBS", "EP", "DEC")
# EP field prefixes: u trigger, b background, a action, y outcome, r reward, k constraints (';'-separated).
# EP 字段前缀：u 触发、b 背景、a 行动、y 结果、r 回报、k 约束（';' 分隔）。
EP_KEYS = ("u", "b", "a", "y", "r", "k")
# The warning text of a reply that opened no section (prose, a JSON object, an empty string): the lines analogue of
# an unparsable JSON reply, grepped by the benchmark harness to schedule the same retry / bisect as invalid JSON.
# 没有开启任何分节的回复（散文、JSON 对象、空串）的警告文本：相当于 JSON 路径的不可解析回复，基准脚本据此安排与非法
# JSON 相同的重试 / 二分。
NO_SECTION_MARK = "reply has no OBS / EP / DEC section"
_HEADER_RE = re.compile(r"^[#\s]*(OBS|EP|DEC)\s*:?\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```")
_TRAILING_DATE_RE = re.compile(r"\s+@(\S+)$")
_EP_FIELD_RE = re.compile(r"^\s*([ubayrk])\s*=\s*(.*)$", re.IGNORECASE | re.DOTALL)


def _header(line: str) -> str | None:
    """The section a line opens ('OBS' / 'EP' / 'DEC'; a leading '#' or a trailing ':' is tolerated), else None.
    该行所开启的分节（'OBS' / 'EP' / 'DEC'；容忍前置 '#' 或尾随 ':'），否则 None。
    """
    match = _HEADER_RE.match(line)
    return match.group(1).upper() if match else None


def _index(field: str) -> int | str:
    """A digit field as int, anything else as the stripped string (the witness rule then rejects it with a warning).
    数字字段转为 int，其它保留为去空白字符串（随后由见证规则记警告拒绝）。
    """
    text = field.strip()
    return int(text) if text.isdigit() else text


def _type_of(field: str, *, what: str) -> str:
    """The observation type of a T field: its first letter through LINE_TYPES (a full type name maps by its initial), an unknown letter → 'fact'.
    T 字段的观察类型：首字母经 LINE_TYPES 映射（完整类型名按首字母映射），未知字母 → 'fact'。
    """
    key = field.strip()[:1].upper()
    name = LINE_TYPES.get(key)
    if name is None:
        logger.debug("%s: unknown type letter %r read as fact", what, field.strip())
        return "fact"
    return name


def _split_date(statement: str, extra: list[str]) -> tuple[str, str | None]:
    """Detach the optional '@date' — its own tab field, or glued to the statement after a space — from a statement.
    从陈述中分离可选的 '@date'——独立的 tab 字段，或以空格粘在陈述末尾。
    """
    when: str | None = None
    for field in extra:
        text = field.strip()
        if text.startswith("@") and len(text) > 1:
            when = text[1:].strip()
    text = statement.strip()
    if when is None:
        match = _TRAILING_DATE_RE.search(text)
        if match:
            when, text = match.group(1), text[: match.start()].rstrip()
    return text, when


def _observation(fields: list[str], *, previous_idx: int | str | None, what: str) -> dict | None:
    """One tabbed OBS line → a compact item (without id; `text` may be shorter than two characters when the line
    carried a prefix but no statement) or None when it has no event_idx at all.
    一行带 tab 的 OBS → 一个紧凑条目（不含 id；该行只有前缀没有陈述时 `text` 可能短于两个字符），完全没有 event_idx 时
    为 None。

    Three or more fields are `<event_idx> <T> <statement> [@date]`. Two fields are `<event_idx> <statement>` (type
    fact) when the first is a number, `<T> <statement>` on the previous line's event_idx when it is a type letter,
    else the whole line is a statement on the previous line's event_idx. A line without a tab never reaches here:
    parse_lines treats it as the continuation of the previous line.
    三个及以上字段是 `<event_idx> <T> <statement> [@date]`。两个字段：首字段为数字则是 `<event_idx> <statement>`
    （类型 fact），首字段为类型字母则是 `<T> <statement>` 且沿用上一行的 event_idx，否则整行是沿用上一行 event_idx 的
    陈述。无 tab 的行不会到这里：parse_lines 把它当作上一行的续行。
    """
    if len(fields) >= 3:
        text, when = _split_date(fields[2], fields[3:])
        idx, obs_type = _index(fields[0]), _type_of(fields[1], what=what)
    elif fields[0].strip().isdigit():
        text, when = _split_date(fields[1], [])
        idx, obs_type = _index(fields[0]), "fact"
    elif fields[0].strip()[:1].upper() in LINE_TYPES and len(fields[0].strip()) <= 16:
        text, when = _split_date(fields[1], [])
        idx, obs_type = previous_idx, _type_of(fields[0], what=what)
    else:
        text, when = _split_date(" ".join(fields), [])
        idx, obs_type = previous_idx, "fact"
    if idx is None:
        logger.warning("%s: line %r has no event_idx and no previous line; skipped", what, text[:60])
        return None
    item: dict[str, Any] = {"event_idx": idx, "text": text, "type": obs_type}
    if when:
        item["when"] = when
    return item


def _ep(fields: list[str], *, what: str) -> dict | None:
    """One EP line → an EP item (without id); a field without a known `key=` prefix is ignored with a DEBUG line.
    一行 EP → 一个 EP 条目（不含 id）；没有已知 `key=` 前缀的字段记 DEBUG 日志并忽略。
    """
    if len(fields) < 2:
        logger.warning("%s: EP line %r has no slots; skipped", what, "\t".join(fields)[:60])
        return None
    slots: dict[str, str] = {}
    for field in fields[1:]:
        match = _EP_FIELD_RE.match(field)
        if match is None:
            logger.debug("%s: EP field %r has no slot prefix; ignored", what, field[:40])
            continue
        slots[match.group(1).lower()] = match.group(2).strip()
    constraints = [c.strip() for c in slots.get("k", "").split(";") if c.strip()]
    return {
        "event_idx": _index(fields[0]),
        "u": slots.get("u", ""),
        "b": slots.get("b", ""),
        "a": slots.get("a", ""),
        "y": slots.get("y", ""),
        "r": slots.get("r", "unknown") or "unknown",
        "kappa": {"constraints": constraints},
    }


def _decision(fields: list[str], *, n_lines: int, kept: set[str], what: str) -> dict | None:
    """One DEC line → a decision item keyed by the OBS line number it names; a number outside 1..n_lines, or one whose
    line stored no observation, is skipped with a warning.
    一行 DEC → 以其点名的 OBS 行号为键的决策条目；行号不在 1..n_lines 内、或该行没有存下观察时，记警告并跳过。
    """
    if len(fields) < 3:
        logger.warning("%s: DEC line %r is incomplete; skipped", what, "\t".join(fields)[:60])
        return None
    number = fields[0].strip()
    if not number.isdigit() or not 1 <= int(number) <= n_lines:
        logger.warning("%s: DEC line names OBS line %r outside 1..%d; skipped", what, number, n_lines)
        return None
    if str(int(number)) not in kept:
        logger.warning("%s: DEC line names OBS line %s, which stored no observation; skipped", what, int(number))
        return None
    return {
        "id": str(int(number)),
        "decision": fields[1].strip().upper(),
        "target_id": fields[2].strip(),
        "reason": "\t".join(fields[3:]).strip(),
    }


def parse_lines(text: str | None, *, what: str = "memory_extraction_lines") -> dict[str, list[dict]]:
    """Parse a lines-contract reply into the compact-shaped roots.
    把行式契约的回复解析为紧凑形状的各根。

    Input / 输入：
        text: The raw reply (None reads as '').
        text：原始回复（None 视为 ''）。

        what: The task name used in log lines.
        what：日志中使用的任务名。

    Output / 输出：
        {'observations': [{'id': OBS line number, 'event_idx', 'text', 'type', 'when'?}], 'eps': [{'id', 'event_idx',
        'u', 'b', 'a', 'y', 'r', 'kappa'}], 'decisions': [{'id': OBS line number, 'decision', 'target_id', 'reason'}]}.
        Blank lines, code fences and anything before the OBS header are ignored (a tabbed `<idx> <T> ...` line before
        any header opens OBS implicitly). Numbering: every tabbed line of the OBS section consumes one 1-based line
        number, whether it stores an observation or not — a prefix-only line (`<idx> <T>` with an empty statement, or
        a lone letter left by a cut) stores nothing but keeps its number — so the numbers a DEC line names are the
        lines the model wrote. A line without a tab consumes no number: it is a continuation — the statement of an
        immediately preceding prefix-only line, else appended (after a space) to the previous observation's statement;
        with no previous observation it is skipped with one warning. An unknown type letter reads as fact; a DEC line
        naming a number outside 1..n_lines or a line that stored nothing is skipped with a warning; a trailing line
        without a newline and with fewer than two tabs is a truncated reply and is dropped with a warning. An absent
        section is an empty list; a reply that opens no section at all (prose, a JSON object, '') logs one warning
        containing NO_SECTION_MARK.
        {'observations': [{'id': OBS 行号, 'event_idx', 'text', 'type', 'when'?}], 'eps': [{'id', 'event_idx', 'u', 'b',
        'a', 'y', 'r', 'kappa'}], 'decisions': [{'id': OBS 行号, 'decision', 'target_id', 'reason'}]}。空行、代码围栏与
        OBS 标题之前的内容被忽略（任何标题之前的 tab 分隔 `<idx> <T> ...` 行隐式开启 OBS）。编号：OBS 分节里每个带 tab 的
        行都占用一个 1 起的行号，无论是否存下观察——只有前缀的行（`<idx> <T>` 而陈述为空，或被截断只剩一个字母）不存内容但
        保留行号——因此 DEC 行点名的行号就是模型所写的行。无 tab 的行不占行号：它是续行——紧接在只有前缀的行之后时就是
        该行的陈述，否则（加一个空格）接到上一条观察的陈述之后；没有上一条观察时记一条警告并跳过。未知类型字母按 fact
        读取；DEC 行点名的行号不在 1..n_lines 内、或该行没有存下观察时记警告并跳过；末尾没有换行且少于两个 tab 的行视为
        被截断的回复，记警告并丢弃。缺失的分节是空列表；完全没有开启任何分节的回复（散文、JSON 对象、''）记一条含
        NO_SECTION_MARK 的警告。

    State effect / 状态影响：
        None; logs only.
        无；仅日志。
    """
    raw = "" if text is None else str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    if raw and not raw.endswith("\n"):
        last = lines[-1]
        stripped = last.strip()
        if stripped and _header(last) is None and not _FENCE_RE.match(last) and last.count("\t") < 2:
            logger.warning("%s: reply ends mid-line; trailing line %r dropped as truncated", what, stripped[:60])
            lines = lines[:-1]
    observations: list[dict] = []
    eps: list[dict] = []
    decisions: list[dict] = []
    section: str | None = None
    previous_idx: int | str | None = None
    n_lines = 0  # tabbed OBS lines seen so far = the number the next one takes minus one / 已见带 tab 的 OBS 行数
    pending: dict | None = None  # a prefix-only OBS line waiting for its statement / 等待陈述的只有前缀的 OBS 行
    for line in lines:
        if not line.strip() or _FENCE_RE.match(line):
            continue
        header = _header(line)
        if header is not None:
            section = header
            pending = None
            continue
        fields = line.split("\t")
        if section is None:
            if len(fields) >= 3 and fields[0].strip().isdigit() and fields[1].strip()[:1].upper() in LINE_TYPES:
                section = "OBS"
            else:
                logger.debug("%s: line before the OBS header ignored: %r", what, line.strip()[:60])
                continue
        if section == "OBS":
            if len(fields) == 1:
                # No tab: a continuation. It is the statement of a prefix-only line just above, else more of the
                # previous statement (a wrapped line); either way it consumes no line number.
                # 无 tab：续行。紧接在只有前缀的行之后时就是它的陈述，否则是上一条陈述的延续（被折行）；都不占行号。
                text_, when = _split_date(line, [])
                if pending is not None:
                    pending["text"] = text_
                    if when and not pending.get("when"):
                        pending["when"] = when
                    if len(text_) >= 2:
                        observations.append(pending)
                        pending = None
                elif observations:
                    observations[-1]["text"] = f"{observations[-1]['text']} {text_}".strip()
                else:
                    logger.warning("%s: line %r has no event_idx and no previous line; skipped", what, text_[:60])
                continue
            n_lines += 1
            pending = None
            item = _observation(fields, previous_idx=previous_idx, what=what)
            if item is None:
                continue
            previous_idx = item["event_idx"]
            item = {"id": str(n_lines), **item}
            if len(item["text"]) < 2:
                # A prefix without a statement (or a lone letter left by a cut): keep the number, wait for a continuation.
                # 只有前缀没有陈述（或被截断只剩一个字母）：保留行号，等待续行。
                pending = item
                continue
            observations.append(item)
        elif section == "EP":
            item = _ep(fields, what=what)
            if item is not None:
                eps.append({"id": str(len(eps)), **item})
        else:
            item = _decision(fields, n_lines=n_lines, kept={o["id"] for o in observations}, what=what)
            if item is not None:
                decisions.append(item)
    if section is None:
        logger.warning("%s: %s; nothing parsed (%d chars)", what, NO_SECTION_MARK, len(raw))
    return {"observations": observations, "eps": eps, "decisions": decisions}


__all__ = ["EP_KEYS", "LINE_TYPES", "NO_SECTION_MARK", "SECTIONS", "TYPE_LETTERS", "parse_lines"]
