"""Hybrid read context builders: deduplicated, time-sorted, timestamped memory rows under a memoized session card.
混合读上下文构造器：记忆化会话卡片之下去重、按时间排序、带时间戳的记忆行。
"""

# Purpose (EN): The single source of truth for the text HMS.context() returns (CONTRACT §6.7): the row key, the
#               duplicate / substring drop, the stable time sort, the 'YYYY-MM-DD HH:MM' stamp and the two-part
#               '<session card>\n\nRecent relevant memories:\n<rows>' body that the READER_AB harness measured at
#               70.1% on EgoLife (docs/design/READER_AB_2026-09-18.md §六). benchmarks/egolife/reader_ab.py imports
#               these functions instead of keeping copies, so the harness and the library agree byte for byte.
# 目的（中）：HMS.context() 返回文本的唯一真源（CONTRACT §6.7）：行键、重复 / 子串剔除、稳定时间排序、
#            'YYYY-MM-DD HH:MM' 时间戳，以及 READER_AB 基准在 EgoLife 上测得 70.1% 的两段式正文
#            '<会话卡片>\n\nRecent relevant memories:\n<记忆行>'（docs/design/READER_AB_2026-09-18.md §六）。
#            benchmarks/egolife/reader_ab.py 导入这些函数而不再保留副本，因此基准脚本与库逐字节一致。

# Input (EN): Public search rows (MemoryAtom.to_public_dict: 'memory', 'created_at', 'hms.time_anchor') in rank
#             order, a rendered session card text and a header line.
# 输入（中）：按排名的公开检索行（MemoryAtom.to_public_dict：'memory'、'created_at'、'hms.time_anchor'）、
#            已渲染的会话卡片文本与标题行。

# Output (EN): Pure functions returning the kept rows or the rendered text; nothing else.
# 输出（中）：只返回保留行或渲染文本的纯函数；别无其它。

# State effect (EN): None; no store, index or clock access.
# 状态影响（中）：无；不访问存储、索引或时钟。

from __future__ import annotations

from typing import Any

from hms_elite.timeutil import to_utc_iso

# The header line between the session card and the rows — the string READER_AB measured with; CompilerConfig's
# context_rows_header default equals it (pinned by tests/test_context.py).
# 会话卡片与记忆行之间的标题行——READER_AB 测量所用的字符串；CompilerConfig 的 context_rows_header 默认值与之相等
# （由 tests/test_context.py 钉住）。
ROWS_HEADER = "Recent relevant memories:"
# A blank part of the hybrid body renders as this, so the body is never blank and the prompt shape never changes.
# 混合正文中空白的部分渲染为此值，正文因此永不为空，提示词形状也永不改变。
NONE_TEXT = "(none)"


def normalize_text(text: str) -> str:
    """Case- and whitespace-insensitive key of a row text (casefold, whitespace runs collapsed to one space, trimmed).
    行文本的大小写与空白不敏感键（casefold、连续空白折成一个空格、去首尾）。

    Input / 输入：
        text: Row text.
        text：行文本。

    Output / 输出：
        The key.
        键。

    State effect / 状态影响：
        None.
        无。
    """
    return " ".join(str(text or "").split()).casefold()


def row_times(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """(anchor time, insertion time) of a public row: hms.time_anchor and created_at (None when absent); time_anchor is the instant the atom's add() was ingested with (add(timestamp=...), else the ingest instant), so rows of one add() share it.
    公开行的（锚定时间，插入时间）：hms.time_anchor 与 created_at（缺失为 None）；time_anchor 是该原子 add() 摄入时所带的时刻（add(timestamp=...)，否则为摄入时刻），因此同一次 add() 的行共享它。

    Input / 输入：
        row: A public search / get_all row.
        row：公开的 search / get_all 行。

    Output / 输出：
        Two ISO strings or None.
        两个 ISO 字符串或 None。

    State effect / 状态影响：
        None.
        无。
    """
    anchor = (row.get("hms") or {}).get("time_anchor")
    created = row.get("created_at")
    return (str(anchor) if anchor else None, str(created) if created else None)


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicates (normalized text; the first in rank order survives), then drop every row whose normalized text is a proper substring of another unique row's normalized text; rank order kept, original dicts returned.
    先去掉精确重复（归一化文本；排名靠前者保留），再去掉归一化文本是另一唯一行归一化文本真子串的行；保持排名顺序，返回原字典。

    Input / 输入：
        rows: Public search rows in rank order (`memory` key).
        rows：按排名的公开检索行（`memory` 键）。

    Output / 输出：
        The kept rows in rank order.
        保留的行，按排名。

    State effect / 状态影响：
        None.
        无。
    """
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for r in rows:
        key = normalize_text(r.get("memory", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((key, r))
    keys = [k for k, _ in unique]
    return [r for k, r in unique if not any(k != other and k in other for other in keys)]


def sort_rows_by_time(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable ascending sort by hms.time_anchor (fallback: created_at); rows with neither go last in their input order, and rows sharing the key keep their input (search-rank) order since sorted() is stable.
    按 hms.time_anchor（回退：created_at）稳定升序排序；两者都缺的行按输入顺序排在最后，键相同的行保持输入（检索排名）顺序，因为 sorted() 是稳定的。

    Input / 输入：
        rows: Public search rows.
        rows：公开检索行。

    Output / 输出：
        The sorted rows.
        排序后的行。

    State effect / 状态影响：
        None.
        无。
    """

    def key(r: dict[str, Any]) -> tuple[int, str]:
        anchor, created = row_times(r)
        t = anchor or created
        return (1, "") if t is None else (0, t)

    return sorted(rows, key=key)


def format_stamp(iso: str | None) -> str | None:
    """'YYYY-MM-DD HH:MM' (UTC) of an ISO instant; None when absent or unparsable.
    ISO 时刻的 'YYYY-MM-DD HH:MM'（UTC）；缺失或不可解析时为 None。

    Input / 输入：
        iso: ISO-8601 string or None.
        iso：ISO-8601 字符串或 None。

    Output / 输出：
        The stamp or None.
        时间戳或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if not iso:
        return None
    try:
        utc = to_utc_iso(iso)
    except Exception:  # noqa: BLE001 - an unparsable anchor renders as unknown / 不可解析的锚定渲染为未知
        return None
    return f"{utc[:10]} {utc[11:16]}"


def render_rows(rows: list[dict[str, Any]], *, timestamps: bool = True) -> str:
    """Render already kept rows in the given order: '- [YYYY-MM-DD HH:MM] text' stamped from hms.time_anchor ONLY (created_at is a sort fallback, never a stamp source; the bracket is omitted when the anchor is unknown), or '- text' when timestamps is False; blank texts skipped; '' for no lines.
    按给定顺序渲染已保留的行：'- [YYYY-MM-DD HH:MM] 文本'，时间戳只来自 hms.time_anchor（created_at 只是排序回退，绝不作时间戳来源；锚定未知时省略方括号）；timestamps 为 False 时为 '- 文本'；跳过空白文本；无行时为 ''。

    Input / 输入：
        rows: Rows in their final order (dedup_rows then sort_rows_by_time).
        rows：已按最终顺序排列的行（dedup_rows 再 sort_rows_by_time）。

        timestamps: Prefix each line with the anchor stamp.
        timestamps：每行加锚定时间戳前缀。

    Output / 输出：
        The joined lines.
        连接后的各行。

    State effect / 状态影响：
        None.
        无。
    """
    lines: list[str] = []
    for r in rows:
        text = str(r.get("memory") or "").strip()
        if not text:
            continue
        stamp = format_stamp(row_times(r)[0]) if timestamps else None
        lines.append(f"- [{stamp}] {text}" if stamp else f"- {text}")
    return "\n".join(lines)


def rows_text(rows: list[dict[str, Any]], *, timestamps: bool = True) -> str:
    """dedup_rows → sort_rows_by_time → render_rows over raw search rows (the harness's rows_ts / rows_td contexts).
    对原始检索行做 dedup_rows → sort_rows_by_time → render_rows（基准脚本的 rows_ts / rows_td 上下文）。

    Input / 输入：
        rows: Public search rows in rank order.
        rows：按排名的公开检索行。

        timestamps: True → '- [stamp] text' lines (rows_ts); False → '- text' lines (rows_td).
        timestamps：True → '- [时间戳] 文本' 行（rows_ts）；False → '- 文本' 行（rows_td）。

    Output / 输出：
        The rows block ('' when nothing is kept).
        记忆行段（无保留行时为 ''）。

    State effect / 状态影响：
        None.
        无。
    """
    return render_rows(sort_rows_by_time(dedup_rows(rows)), timestamps=timestamps)


def hybrid_text(session_text: str, rows_block: str, *, header: str = ROWS_HEADER) -> str:
    """The hybrid body: the session card, a blank line, the header on its own line, then the rows block, a blank part rendering as NONE_TEXT; put it BEFORE the question ('Memories:\\n<body>\\n\\n<question>') — memories-first costs no accuracy and is the only order whose prefix a provider's prompt cache can hit while the session card is reused.
    混合正文：会话卡片、空行、单独一行的标题、再记忆行段，空白部分渲染为 NONE_TEXT；把它放在问题**之前**（'Memories:\\n<正文>\\n\\n<问题>'）——记忆在前不损失准确率，且是会话卡片复用期间唯一能命中提供者提示词缓存前缀的顺序。

    Input / 输入：
        session_text: The memoized session card text ('' before the first ingest).
        session_text：记忆化的会话卡片文本（首次摄入前为 ''）。

        rows_block: rows_text of the query's search rows.
        rows_block：本次查询检索行的 rows_text。

        header: The line between the two parts (CompilerConfig.context_rows_header).
        header：两部分之间的那一行（CompilerConfig.context_rows_header）。

    Output / 输出：
        The body text.
        正文文本。

    State effect / 状态影响：
        None.
        无。
    """
    head = session_text if session_text.strip() else NONE_TEXT
    tail = rows_block if rows_block.strip() else NONE_TEXT
    return head + "\n\n" + header + "\n" + tail


__all__ = [
    "NONE_TEXT",
    "ROWS_HEADER",
    "dedup_rows",
    "format_stamp",
    "hybrid_text",
    "normalize_text",
    "render_rows",
    "row_times",
    "rows_text",
    "sort_rows_by_time",
]
