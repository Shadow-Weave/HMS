"""Window segmentation shared by every arm: cues folded into <=400-char / <=60 s windows per SRT file.
所有臂共用的窗口切分：每个 SRT 文件的 cue 按顺序折叠为 ≤400 字符 / ≤60 s 的窗口。
"""

# Purpose (EN): Produce the one window list all arms retain, plus its JSONL serialization and sha256 so the
#               shared-variable evidence can be recorded in the results.
# 目的（中）：产出所有臂共同 retain 的唯一窗口列表，以及其 JSONL 序列化与 sha256，供结果里记录共享变量证据。

# Input (EN): {file_stem: [Cue]} from loader.load_transcripts; the two closing limits.
# 输入（中）：loader.load_transcripts 产出的 {文件主干: [Cue]}；两个封口上限。

# Output (EN): Window dataclasses with window_id = f"{file_stem}#{index:04d}"; write_windows_jsonl returns the sha256.
# 输出（中）：window_id = f"{文件主干}#{序号:04d}" 的 Window 数据类；write_windows_jsonl 返回 sha256。

# State effect (EN): write_windows_jsonl writes one file.
# 状态影响（中）：write_windows_jsonl 写一个文件。

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from loader import Cue

MAX_CHARS = 400
MAX_SPAN_CS = 60 * 100


@dataclass(frozen=True)
class Window:
    """One retained unit: consecutive cues of one file, English text for retrieval, Chinese kept for provenance.
    一个 retain 单元：同一文件的连续 cue，英文文本用于检索，中文保留做溯源。
    """

    window_id: str
    source: str
    day: int
    start_cs: int  # absolute centiseconds, first cue start / 绝对厘秒，首 cue 起点
    end_cs: int  # absolute centiseconds, last cue end / 绝对厘秒，末 cue 终点
    cue_from: int  # first cue index (inclusive) / 首 cue 序号（含）
    cue_to: int  # last cue index (inclusive) / 末 cue 序号（含）
    text: str  # English lines joined by newline / 英文行以换行连接
    text_zh: str

    @property
    def start_iso(self) -> str:
        """Synthetic UTC instant for the window start (see clock_to_iso).
        窗口起点的合成 UTC 时刻（见 clock_to_iso）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            ISO-8601 string.
            ISO-8601 字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return clock_to_iso(self.day, self.start_cs)


def clock_to_iso(day: int, centiseconds: int) -> str:
    """Map (DAY n, centiseconds since midnight) to a synthetic UTC instant: 2024-01-(n) plus the clock.
    把（DAY n，自午夜厘秒）映射为合成 UTC 时刻：2024-01-(n) 加上该时钟。

    Input / 输入：
        day: 1..7.
        day：1..7。

        centiseconds: Clock since midnight (may exceed 24 h only for the END_OF_DAY sentinel; clamped).
        centiseconds：自午夜的时钟（只有 END_OF_DAY 哨兵会超过 24 h；截断）。

    Output / 输出：
        'YYYY-MM-DDTHH:MM:SS.ffffffZ'.
        'YYYY-MM-DDTHH:MM:SS.ffffffZ'。

    State effect / 状态影响：
        None. The dataset has no calendar dates; every arm uses this same mapping.
        无。数据集没有日历日期；所有臂使用同一映射。
    """
    cs = min(max(int(centiseconds), 0), 24 * 3600 * 100 - 1)
    seconds, frac = divmod(cs, 100)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"2024-01-{day:02d}T{hours:02d}:{minutes:02d}:{secs:02d}.{frac * 10000:06d}Z"


def build_windows(
    transcripts: dict[str, list[Cue]],
    *,
    max_chars: int = MAX_CHARS,
    max_span_cs: int = MAX_SPAN_CS,
    text_lang: Literal["en", "zh"] = "en",
) -> list[Window]:
    """Fold cues into windows: a window closes when adding the next cue would exceed max_chars or max_span_cs.
    把 cue 折叠为窗口：再加入下一条 cue 会超过 max_chars 或 max_span_cs 时封口。

    Input / 输入：
        transcripts: {file_stem: cues} in file order.
        transcripts：按文件顺序的 {文件主干: cues}。

        max_chars: Character budget on the English text (400).
        max_chars：英文文本的字符预算（400）。

        max_span_cs: Span budget from first cue start to last cue end (60 s).
        max_span_cs：首 cue 起点到末 cue 终点的跨度预算（60 s）。

        text_lang: 'en' (default) → `Window.text` = English lines, `text_zh` = Chinese lines (today's behaviour); 'zh' (the EgoLife-ZH runs on data_zh) → the two slots are swapped for the SAME cue set: `text` = the Chinese lines of the cues that carry English text, `text_zh` = the English lines. Boundaries, ids, timestamps and cue ranges are always computed from the English characters, so a zh window is the exact twin of its en window; cues with Chinese but no English text (2,092 in A1) contribute to neither, so both runs retrieve over the same content set. A zh window whose text would be empty (a window of English-only cues without translations) raises ValueError naming every such window: the dataset is incomplete, not silently thinner.
        text_lang：'en'（默认）→ `Window.text` = 英文行、`text_zh` = 中文行（今日行为）；'zh'（data_zh 上的 EgoLife-ZH 运行）→ 对同一 cue 集合交换两个槽位：`text` = 带英文文本的 cue 的中文行，`text_zh` = 英文行。边界、id、时间戳与 cue 区间始终按英文字符计算，因此 zh 窗口是 en 窗口的精确孪生；只有中文没有英文的 cue（A1 中 2,092 条）两边都不计入，两次运行检索的内容集合相同。zh 文本为空的窗口（只含无译文的纯英文 cue）抛出列出全部此类窗口的 ValueError：数据集不完整，而不是悄悄变薄。

    Output / 输出：
        Windows in (file, index) order; cues with empty English text count for the span but add no chars, and a
        window with no English text at all is dropped.
        按（文件，序号）顺序的窗口；英文为空的 cue 计入跨度但不增加字符，完全没有英文的窗口被丢弃。

    State effect / 状态影响：
        None.
        无。
    """
    if text_lang not in ("en", "zh"):
        raise ValueError(f"text_lang must be 'en' or 'zh', got {text_lang!r}")
    windows: list[Window] = []
    empty_zh: list[str] = []
    for stem, cues in transcripts.items():
        current: list[Cue] = []
        chars = 0
        index = 0

        def close() -> None:
            nonlocal current, chars, index
            if not current:
                return
            # A window without any English line has nothing to retrieve: it is dropped, not numbered.
            # 没有任何英文行的窗口无可检索内容：丢弃且不占序号。
            if not any(c.text_en for c in current):
                current, chars = [], 0
                return
            en_text = "\n".join(c.text_en for c in current if c.text_en)
            if text_lang == "zh":
                text = "\n".join(c.text_zh for c in current if c.text_en and c.text_zh)
                text_zh = en_text
                if not text:
                    empty_zh.append(f"{stem}#{index:04d}")
            else:
                text = en_text
                text_zh = "\n".join(c.text_zh for c in current if c.text_zh)
            windows.append(
                Window(
                    window_id=f"{stem}#{index:04d}",
                    source=stem,
                    day=current[0].day,
                    start_cs=current[0].start_cs,
                    end_cs=max(c.end_cs for c in current),
                    cue_from=current[0].index,
                    cue_to=current[-1].index,
                    text=text,
                    text_zh=text_zh,
                )
            )
            index += 1
            current, chars = [], 0

        for cue in cues:
            add_chars = len(cue.text_en) + (1 if current and cue.text_en else 0)
            if current and (chars + add_chars > max_chars or cue.end_cs - current[0].start_cs > max_span_cs):
                close()
                add_chars = len(cue.text_en)
            current.append(cue)
            chars += add_chars
        close()
    if empty_zh:
        raise ValueError(f"{len(empty_zh)} window(s) have no Chinese text under text_lang='zh': {empty_zh[:20]}")
    return windows


def write_windows_jsonl(windows: list[Window], path: Path) -> str:
    """Serialize the window list (one JSON object per line, sorted keys) and return the file's sha256.
    序列化窗口列表（每行一个 JSON 对象，键排序）并返回文件的 sha256。

    Input / 输入：
        windows: From build_windows.
        windows：build_windows 的输出。

        path: Destination file.
        path：目标文件。

    Output / 输出：
        Hex sha256 of the written bytes.
        写入字节的十六进制 sha256。

    State effect / 状态影响：
        Writes path.
        写入 path。
    """
    payload = "".join(json.dumps(asdict(w), ensure_ascii=False, sort_keys=True) + "\n" for w in windows).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def window_ids_digest(window_ids: list[str]) -> str:
    """sha256 over the sorted, newline-joined window ids (the per-arm 'same window set' evidence).
    对排序、换行连接的 window id 求 sha256（各臂「同一窗口集合」的证据）。

    Input / 输入：
        window_ids: Any order, duplicates allowed (they change the digest).
        window_ids：任意顺序，允许重复（重复会改变摘要）。

    Output / 输出：
        Hex digest.
        十六进制摘要。

    State effect / 状态影响：
        None.
        无。
    """
    return hashlib.sha256("\n".join(sorted(window_ids)).encode("utf-8")).hexdigest()


__all__ = ["MAX_CHARS", "MAX_SPAN_CS", "Window", "build_windows", "clock_to_iso", "window_ids_digest", "write_windows_jsonl"]
