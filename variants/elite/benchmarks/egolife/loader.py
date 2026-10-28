"""EgoLife loader: QA items and SRT cues with absolute centisecond clocks.
EgoLife 加载器：QA 条目与带绝对厘秒时钟的 SRT 字幕块。
"""

# Purpose (EN): Parse EgoLifeQA_A1_JAKE_TEXT_ONLY.json and the DAY1..7 SRT files into plain records; the cue
#               absolute clock reuses hms_lite-main/benchmarks/egolife.py (hhmmsscc_to_centiseconds + cue offset).
# 目的（中）：把 EgoLifeQA_A1_JAKE_TEXT_ONLY.json 与 DAY1..7 的 SRT 文件解析为普通记录；cue 绝对时钟逐字复用
#            hms_lite-main/benchmarks/egolife.py（hhmmsscc_to_centiseconds + cue 偏移）。

# Input (EN): The dataset root directory (data/ with the QA JSON and Transcript/A1_JAKE/DAY*/*.srt).
# 输入（中）：数据集根目录（data/，含 QA JSON 与 Transcript/A1_JAKE/DAY*/*.srt）。

# Output (EN): Question and Cue dataclasses; nothing is written.
# 输出（中）：Question 与 Cue 数据类；不写任何文件。

# State effect (EN): None (read-only).
# 状态影响（中）：无（只读）。

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Copied verbatim from /Users/lazy/Desktop/hms_lite-main/benchmarks/egolife.py (their protocol implementation).
# 逐字复制自 /Users/lazy/Desktop/hms_lite-main/benchmarks/egolife.py（对方的协议实现）。
_CHINESE = re.compile(r"[\u3400-\u9fff]")
_SRT_START = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->.*")
_SOURCE_NAME = re.compile(r"_DAY(\d+)_(\d{8})$")

# Full cue timing line (start and end); their regex only captures the start, we also need the end for windows.
# 完整的 cue 时间行（起与止）；对方的正则只取起点，我们做窗口还需要终点。
_SRT_RANGE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})")
_SPEAKER = re.compile(r"^\s*([A-Za-z][A-Za-z ]{0,30}?)\s*[:：]\s*(.*)$")

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
QA_FILENAME = "EgoLifeQA_A1_JAKE_TEXT_ONLY.json"
TRANSCRIPT_SUBDIR = Path("Transcript") / "A1_JAKE"

# Whole-day cutoff used by their build_streaming_deltas for earlier days ("23595999").
# 对方 build_streaming_deltas 对前几日使用的全天截止值（"23595999"）。
END_OF_DAY_HHMMSSCC = "23595999"


def hhmmsscc_to_centiseconds(value: str) -> int:
    """Convert the dataset's HHMMSScc clock representation.
    转换数据集的 HHMMSScc 时钟表示。

    Input / 输入：
        value: Eight digits HHMMSScc.
        value：八位数字 HHMMSScc。

    Output / 输出：
        Centiseconds since midnight; ValueError on a malformed string.
        自午夜起的厘秒数；格式错误时抛 ValueError。

    State effect / 状态影响：
        None. (Copied verbatim from hms_lite-main/benchmarks/egolife.py.)
        无。（逐字复制自 hms_lite-main/benchmarks/egolife.py。）
    """
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError("time must use HHMMSScc format")
    return ((int(value[:2]) * 60 + int(value[2:4])) * 60 + int(value[4:6])) * 100 + int(value[6:])


def cue_absolute_centiseconds(file_start: int, hours: int, minutes: int, seconds: int, milliseconds: int) -> int:
    """Absolute cue clock = file start + relative offset (their select_transcript_cues formula, verbatim).
    cue 绝对时钟 = 文件起始 + 相对偏移（对方 select_transcript_cues 的公式，逐字）。

    Input / 输入：
        file_start: Centiseconds of the file name's HHMMSScc.
        file_start：文件名 HHMMSScc 的厘秒数。

        hours, minutes, seconds, milliseconds: The SRT relative timestamp fields.
        hours、minutes、seconds、milliseconds：SRT 相对时间戳字段。

    Output / 输出：
        Absolute centiseconds since midnight of that day.
        该日自午夜起的绝对厘秒数。

    State effect / 状态影响：
        None.
        无。
    """
    return file_start + ((hours * 60 + minutes) * 60 + seconds) * 100 + milliseconds // 10


def day_number(day: str) -> int:
    """'DAY3' -> 3 (their _query_day_number rule).
    'DAY3' -> 3（对方 _query_day_number 规则）。

    Input / 输入：
        day: 'DAY<number>'.
        day：'DAY<数字>'。

    Output / 输出：
        The integer day; ValueError otherwise.
        整数日号；否则抛 ValueError。

    State effect / 状态影响：
        None.
        无。
    """
    match = re.fullmatch(r"DAY(\d+)", day)
    if match is None:
        raise ValueError("query day must use DAY<number> format")
    return int(match.group(1))


@dataclass(frozen=True)
class Cue:
    """One SRT cue with both language lines and its absolute clock.
    一条 SRT 字幕块：中英两行文本与绝对时钟。
    """

    source: str  # file stem, e.g. A1_JAKE_DAY1_11000000 / 文件名主干
    day: int
    index: int  # cue number inside the file / 文件内序号
    start_cs: int  # absolute centiseconds / 绝对厘秒
    end_cs: int
    speaker: str
    text_zh: str
    text_en: str


@dataclass(frozen=True)
class Question:
    """One EgoLife QA item; hidden fields are kept only for scoring, never handed to the reader.
    一条 EgoLife QA 条目；隐藏字段只用于计分，绝不交给阅读器。
    """

    id: str
    order: int  # position in the original JSON array / 原始数组位置
    type: str
    question: str
    options: dict[str, str]  # {'A': ..., 'D': ...}
    query_day: int
    query_cs: int
    query_hhmmsscc: str
    answer: str  # hidden / 隐藏
    target_day: int  # hidden; the target_time.date / 隐藏；target_time.date
    # hidden; (day, centiseconds) per target instant: 'time', or each 'time_list' entry, where an entry may carry
    # its own 'DAYn_' prefix (3 items in this file span several days).
    # 隐藏；每个 target 时刻的（日，厘秒）：'time'，或 'time_list' 的每一项，其中一项可自带 'DAYn_' 前缀
    #（本文件有 3 条跨多日）。
    targets: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def load_questions(path: Path) -> list[Question]:
    """Load all QA items in array order (their load_questions checks: array, IDs present and unique).
    按数组顺序加载全部 QA 条目（对方 load_questions 的检查：数组、ID 存在且唯一）。

    Input / 输入：
        path: The QA JSON file.
        path：QA JSON 文件。

    Output / 输出：
        Question list in file order.
        按文件顺序的 Question 列表。

    State effect / 状态影响：
        None.
        无。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("EgoLife QA must be a JSON array")
    if not all(isinstance(item, dict) and item.get("ID") is not None for item in raw):
        raise ValueError("Every question must have an ID")
    ids = [str(item["ID"]) for item in raw]
    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique")
    out: list[Question] = []
    for order, item in enumerate(raw):
        target = item["target_time"]
        target_day = day_number(target["date"])
        times = target.get("time_list") if "time_list" in target else [target["time"]]
        targets: list[tuple[int, int]] = []
        for value in times:
            match = re.fullmatch(r"(DAY\d+)_(\d{8})", str(value))
            if match is not None:
                targets.append((day_number(match.group(1)), hhmmsscc_to_centiseconds(match.group(2))))
            else:
                targets.append((target_day, hhmmsscc_to_centiseconds(str(value))))
        out.append(
            Question(
                id=str(item["ID"]),
                order=order,
                type=str(item.get("type", "")),
                question=str(item["question"]).strip(),
                options={letter.upper(): str(item[f"choice_{letter}"]).strip() for letter in "abcd"},
                query_day=day_number(item["query_time"]["date"]),
                query_cs=hhmmsscc_to_centiseconds(item["query_time"]["time"]),
                query_hhmmsscc=item["query_time"]["time"],
                answer=str(item["answer"]).strip().upper(),
                target_day=target_day,
                targets=tuple(targets),
                raw=item,
            )
        )
    return out


def _split_speaker(line: str) -> tuple[str, str]:
    match = _SPEAKER.match(line)
    if match is None:
        return "", line.strip()
    return match.group(1).strip(), match.group(2).strip()


def parse_srt(path: Path) -> list[Cue]:
    """Parse one SRT file into cues with absolute clocks (block splitting mirrors their select_transcript_cues).
    把一个 SRT 文件解析为带绝对时钟的 cue（分块方式与对方 select_transcript_cues 一致）。

    Input / 输入：
        path: DAYn/A1_JAKE_DAYn_HHMMSScc.srt.
        path：DAYn/A1_JAKE_DAYn_HHMMSScc.srt。

    Output / 输出：
        Cue list in file order; blocks without a timing line or without text are skipped like theirs.
        按文件顺序的 cue 列表；无时间行或无文本的块与对方一样跳过。

    State effect / 状态影响：
        None.
        无。
    """
    match = _SOURCE_NAME.search(path.stem)
    if match is None:
        raise ValueError(f"unexpected transcript name: {path.name}")
    day, file_start = int(match.group(1)), hhmmsscc_to_centiseconds(match.group(2))
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        if _SRT_START.fullmatch(lines[1]) is None:
            continue
        rng = _SRT_RANGE.match(lines[1])
        if rng is None:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in rng.groups())
        start_cs = cue_absolute_centiseconds(file_start, h1, m1, s1, ms1)
        end_cs = max(start_cs, cue_absolute_centiseconds(file_start, h2, m2, s2, ms2))
        zh_lines = [line for line in lines[2:] if _CHINESE.search(line) is not None]
        en_lines = [line for line in lines[2:] if _CHINESE.search(line) is None]
        speaker = ""
        for line in en_lines + zh_lines:
            speaker, _ = _split_speaker(line)
            if speaker:
                break
        cues.append(
            Cue(
                source=path.stem,
                day=day,
                index=int(lines[0]) if lines[0].isdigit() else len(cues) + 1,
                start_cs=start_cs,
                end_cs=end_cs,
                speaker=speaker,
                text_zh=" ".join(zh_lines),
                text_en=" ".join(en_lines),
            )
        )
    return cues


def transcript_files(root: Path) -> list[Path]:
    """Sorted DAY*/*.srt paths (their sorted(glob) order).
    排序后的 DAY*/*.srt 路径（对方 sorted(glob) 的顺序）。

    Input / 输入：
        root: Dataset root.
        root：数据集根目录。

    Output / 输出：
        Path list.
        路径列表。

    State effect / 状态影响：
        None.
        无。
    """
    return sorted((root / TRANSCRIPT_SUBDIR).glob("DAY*/*.srt"))


def load_transcripts(root: Path) -> dict[str, list[Cue]]:
    """Parse every transcript file.
    解析全部转写文件。

    Input / 输入：
        root: Dataset root.
        root：数据集根目录。

    Output / 输出：
        {file_stem: cues} in sorted file order.
        按文件排序的 {文件主干: cues}。

    State effect / 状态影响：
        None.
        无。
    """
    return {path.stem: parse_srt(path) for path in transcript_files(root)}


__all__ = [
    "DEFAULT_DATA_ROOT",
    "END_OF_DAY_HHMMSSCC",
    "QA_FILENAME",
    "Cue",
    "Question",
    "cue_absolute_centiseconds",
    "day_number",
    "hhmmsscc_to_centiseconds",
    "load_questions",
    "load_transcripts",
    "parse_srt",
    "transcript_files",
]
