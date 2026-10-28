"""Build the parallel EgoLife-ZH dataset (data_zh/): transcript windows translated to Chinese by deepseek-flash, questions
from the official Chinese fields, a deterministic name glossary, and a translate_report.json with coverage and cost.
构建平行的 EgoLife-ZH 数据集（data_zh/）：转写窗口由 deepseek-flash 译成中文，题目取官方中文字段，确定性人名术语表，
以及带覆盖率与费用的 translate_report.json。
"""

# Purpose (EN): One resumable, cost-capped script that produces data_zh with the SAME file layout, ids, window
#               boundaries, timestamps and gold letters as data/, so run_real.py --data-root data_zh --text-lang zh
#               retains and answers over the Chinese twin of every English window (docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md).
# 目的（中）：一个可断点续跑、有费用上限的脚本，产出与 data/ 文件布局、id、窗口边界、时间戳与答案字母完全相同的
#             data_zh，使 run_real.py --data-root data_zh --text-lang zh 对每个英文窗口的中文孪生做写入与答题
#             （docs/design/EGOLIFE_ZH_PLAN_2026-09-22.md）。

# Input (EN): data/ (74 SRT files, the QA file), the DeepSeek key through real_arms.load_api_key (never printed), flags.
# 输入（中）：data/（74 个 SRT 文件、QA 文件）、经 real_arms.load_api_key 读入的 DeepSeek 密钥（绝不打印）、命令行参数。

# Output (EN): data_zh/Transcript/A1_JAKE/DAYn/*.srt, data_zh/EgoLifeQA_A1_JAKE_TEXT_ONLY.json, data_zh/translate_cache.jsonl,
#              data_zh/translate_report.json.
# 输出（中）：data_zh/Transcript/A1_JAKE/DAYn/*.srt、data_zh/EgoLifeQA_A1_JAKE_TEXT_ONLY.json、data_zh/translate_cache.jsonl、
#             data_zh/translate_report.json。

# State effect (EN): Network calls to api.deepseek.com only in translate mode and only within --cost-cap-cny, which is
#                    cumulative over resumes (translate_spent.json next to the cache holds every invocation's actual CNY;
#                    projection + previous spend before the first call, running ledger + previous spend after every
#                    call); writes under --out only.
# 状态影响（中）：仅在翻译模式且在 --cost-cap-cny 之内调用 api.deepseek.com；上限跨续跑累计（缓存旁的 translate_spent.json
#                记录每次调用脚本的实际人民币；首次调用前按预估 + 既往花费、其后按累计台账 + 既往花费）；只写 --out。

from __future__ import annotations

import argparse
import json
import queue
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accounting import PRICES_USD_PER_M, Meter, rate_tier, utc_now  # noqa: E402
from loader import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    QA_FILENAME,
    TRANSCRIPT_SUBDIR,
    Cue,
    load_questions,
    load_transcripts,
    transcript_files,
)
from windows import MAX_CHARS, Window, build_windows  # noqa: E402

DEFAULT_OUT = HERE / "data_zh"
ENV_PATH = HERE / ".env"
SPENT_FILENAME = "translate_spent.json"
# A projection above this share of the remaining cap is refused without --allow-tight-cap: an output-token
# under-estimate would otherwise end in CostCapExceeded mid-run and a partial dataset (peak ¥5.72 vs cap ¥6 = 95%).
# 预估超过剩余上限的这一比例时，没有 --allow-tight-cap 就拒绝运行：否则输出 token 估少了会在半途 CostCapExceeded 并留下
# 不完整的数据集（高峰 ¥5.72 对上限 ¥6 = 95%）。
TIGHT_CAP_FRACTION = 0.8
MODEL = "deepseek-flash"
TRANSLATE_MAX_TOKENS = 8192
TRANSLATE_TEMPERATURE = 0.0
# Accepted zh / en character ratio of a window's body (labels stripped). Measured on the native A1 lines per window
# (n = 4,845): min 0.148, p01 0.203, p05 0.225, p50 0.279, p95 0.368, max 0.843 — the brief's 0.3 floor would reject
# 70% of genuine windows, so the floor is 0.12 and the ceiling 1.5 (every native window passes; 37 sit under 0.2).
# 窗口正文（去掉标签）可接受的中文 / 英文字符比。按 A1 原生行逐窗口实测（n = 4,845）：最小 0.148、p01 0.203、p05 0.225、
# p50 0.279、p95 0.368、最大 0.843——任务书的 0.3 下限会拒绝 70% 的真实窗口，故下限取 0.12、上限取 1.5（所有原生窗口
# 通过；37 个低于 0.2）。
LENGTH_RATIO: tuple[float, float] = (0.12, 1.5)
# DeepSeek list price in CNY per 1M tokens (deepseek-flash, 2026-09), the same table as reader_ab.PRICES_CNY_PER_M
# (kept here so --dry-run and the offline tests need no API SDK; tests/test_egolife_zh.py pins the equality).
# DeepSeek 人民币定价（每 1M token，deepseek-flash，2026-09），与 reader_ab.PRICES_CNY_PER_M 同一张表
# （放在这里使 --dry-run 与离线测试不需要 API SDK；tests/test_egolife_zh.py 钉住二者相等）。
PRICES_CNY_PER_M: dict[str, dict[str, float]] = {
    "offpeak": {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0},
    "peak": {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0},
}
# The known EgoLife A1 people (user decision); every other capitalised name stays Latin and is reported.
# 已知的 EgoLife A1 人物（用户决定）；其它首字母大写的名字保持拉丁形式并写入报告。
NAME_GLOSSARY: dict[str, str] = {
    "Jake": "杰克", "Alice": "爱丽丝", "Tasha": "塔莎", "Katrina": "卡特里娜", "Lucia": "露西亚", "Shure": "舒尔",
    "Nicous": "尼克斯", "Choiszt": "乔伊斯特", "Me": "我",
}  # fmt: skip
# Capitalised option tokens that are not person names (sentence-initial words, brands, foods; harness report §12).
# 选项里首字母大写却不是人名的词（句首词、品牌、食物；基准报告 §12）。
OPTION_STOP: frozenset[str] = frozenset(
    {
        "Because", "About", "Yesterday", "The", "Last", "Make", "Play", "Use", "One", "This", "Too", "Wash", "Watch", "Convert",
        "Cut", "Eat", "Eating", "Go", "Half", "Hot", "Today", "We", "What", "Bake", "Buy", "How", "Pick", "Playing", "Ask", "Ate",
        "Based", "Broke", "Collect", "Cutting", "Dancing", "Find", "Got", "Install", "On", "Order", "Prepare", "Sing", "Throw",
        "Wipe", "Two", "Three", "Day", "Night", "Dance", "Shopping", "Pizza", "Dumplings", "Noodles", "Taobao", "Tequila",
        "Barbecue", "Burger", "Chicken", "Cola", "Ele", "Fishing", "Grocery", "Meituan", "Park", "Pot", "Reward", "She",
        "Sprite", "Steamed", "Arcade", "Aromatherapy", "Baked", "Beef", "Bread", "Bun", "Cake", "Cardboard", "Dessert",
        "Earth", "Egg", "Fruits", "Hamburger", "Ingredients", "Juice", "Meat", "Milk", "Mineral", "Miniature", "Nongfu",
        "Pillow", "Power", "Rice", "Rum", "Seaweed", "Spring", "Strawberries", "Sweet", "Taro", "Tart", "Wahaha", "Water",
        "Yibao", "Baisui", "Binglu", "Dicos", "Dudu", "Fanta", "Fenjiu", "Hema", "Kangshifu", "Lele", "Pinduoduo", "Vipshop",
        "Wallace", "Xiaobao", "Xiaohei", "Yakult", "Laba", "Mandarin", "Wangzai", "Others", "Other", "Everyone", "Someone",
        "Nobody", "Yes", "No", "It", "They", "He", "I", "A", "An", "In", "At", "To", "For", "With", "And", "Or", "Not", "Then",
        "After", "Before", "When", "While", "From", "Into", "Over", "Under", "Up", "Down", "Out", "Off", "Some", "All", "Both",
        "Each", "Every", "Many", "Much", "More", "Most", "Few", "Several", "First", "Second", "Third", "Next", "New", "Old",
        "Big", "Small", "Red", "Blue", "Green", "White", "Black", "Morning", "Afternoon", "Evening", "Tonight", "Tomorrow",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "None", "Nothing", "Something",
    }
)  # fmt: skip
CJK_RE = re.compile(r"[㐀-鿿]")
LATIN_RE = re.compile(r"[A-Za-z]")
# A speaker label is the loader's `_SPEAKER` shape widened to the apostrophe / parenthesis labels it rejects (Shure's GF).
# 说话人标签是把加载器 `_SPEAKER` 的形状放宽到它会拒绝的撇号 / 括号标签（Shure's GF）。
LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z' ()]{0,30}?)\s*[:：]\s*")
CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+\b")
TRANSLATE_SYSTEM = (
    "你是一名字幕翻译。把每一行英文口语逐行翻译成自然、简洁的简体中文。规则：\n"
    "1. 每行开头的说话人标签（例如 `Jake:`、`Shure's GF:`）原样保留，不翻译、不改动；\n"
    "2. 人名、品牌名、你无法翻译的菜名保持原来的拉丁拼写；下面九个名字必须原样保留：Jake、Alice、Tasha、Katrina、Lucia、"
    "Shure、Nicous、Choiszt、Me；其它内容全部译成中文；\n"
    "3. 输出严格为 JSON：{\"items\": [{\"window_id\": \"...\", \"lines\": [\"...\"]}]}，window_id 与输入完全一致，"
    "每个窗口的行数与顺序与输入完全一致，不合并、不拆分、不省略、不加解释。"
)


@dataclass
class Item:
    """One translation unit: a window (or a question) and its English lines in order.
    一个翻译单元：一个窗口（或一道题）及其按顺序的英文行。
    """

    window_id: str
    lines: list[str]


@dataclass
class Request:
    """One API request: up to `windows_per_request` items of one transcript file.
    一次 API 请求：同一转写文件的至多 `windows_per_request` 个单元。
    """

    request_id: str
    items: list[Item]


@dataclass
class Translation:
    """A cache row: the translated lines of one item, or the reason it failed.
    一条缓存行：某单元的译文行，或失败原因。
    """

    window_id: str
    lines: list[str]
    request_id: str
    attempt: int
    validated: bool
    reason: str | None = None
    model: str = MODEL

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready dict.
        可直接写 JSON 的字典。
        """
        return {
            "window_id": self.window_id, "lines": self.lines, "request_id": self.request_id, "attempt": self.attempt,
            "validated": self.validated, "reason": self.reason, "model": self.model,
        }  # fmt: skip


# ---------------------------------------------------------------------------
# Glossary
# 术语表
# ---------------------------------------------------------------------------
def collect_names(transcripts: dict[str, list[Cue]], questions: list[Any]) -> dict[str, int]:
    """Count the speaker labels of every English line plus the capitalised tokens of the question options minus OPTION_STOP.
    统计每条英文行的说话人标签，加上题目选项中首字母大写的词元（去掉 OPTION_STOP）。

    Input / 输入：
        transcripts: loader.load_transcripts output.
        transcripts：loader.load_transcripts 的输出。

        questions: loader.load_questions output.
        questions：loader.load_questions 的输出。

    Output / 输出：
        {name: count} (labels first by count, then option tokens), a superset of the glossary keys actually seen.
        {名字: 次数}（先标签，后选项词元），是实际出现的术语表键的超集。

    State effect / 状态影响：
        None.
        无。
    """
    counts: dict[str, int] = {}
    for cues in transcripts.values():
        for cue in cues:
            if not cue.text_en:
                continue
            match = LABEL_RE.match(cue.text_en)
            if match:
                label = match.group(1).strip()
                counts[label] = counts.get(label, 0) + 1
    for q in questions:
        for option in q.options.values():
            for token in CAP_TOKEN_RE.findall(option):
                if token not in OPTION_STOP:
                    counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _glossary_regex(glossary: dict[str, str]) -> re.Pattern:
    """One alternation of the glossary keys, longest first, whole-word (no Latin letter on either side), optional 's.
    术语表键的一个交替式，长者优先，整词（两侧无拉丁字母），可带 's。
    """
    keys = sorted(glossary, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in keys) + r")(?P<poss>'s)?(?![A-Za-z])")


_GLOSSARY_RE = _glossary_regex(NAME_GLOSSARY)


def apply_glossary(text: str, glossary: dict[str, str] | None = None) -> str:
    """Deterministic whole-word replacement of the glossary names (longest first); `Name's` → `值的`, `Name:` keeps its colon.
    确定性的整词替换术语表人名（长者优先）；`Name's` → `值的`，`Name:` 保留冒号。

    Input / 输入：
        text: Any line (a Chinese subtitle line, a question, an option).
        text：任意一行（中文字幕行、题目、选项）。

        glossary: {Latin: Chinese}; None → NAME_GLOSSARY.
        glossary：{拉丁: 中文}；None → NAME_GLOSSARY。

    Output / 输出：
        The rewritten text (unchanged when no key occurs).
        改写后的文本（没有键出现时不变）。

    State effect / 状态影响：
        None.
        无。
    """
    table = NAME_GLOSSARY if glossary is None else glossary
    pattern = _GLOSSARY_RE if glossary is None else _glossary_regex(table)

    def _sub(match: re.Match) -> str:
        return table[match.group(1)] + ("的" if match.group("poss") else "")

    return pattern.sub(_sub, text)


# ---------------------------------------------------------------------------
# Requests, validation, projection
# 请求、校验、预估
# ---------------------------------------------------------------------------
def window_cues(window: Window, cues: list[Cue]) -> list[Cue]:
    """The cues of a window (by index range) in file order.
    某窗口（按序号区间）的 cue，按文件顺序。
    """
    return [c for c in cues if window.cue_from <= c.index <= window.cue_to]


def cues_to_translate(window: Window, cues: list[Cue], mode: str) -> list[Cue]:
    """The cues whose English line is sent for translation: every English-bearing cue in 'translate' mode, only the
    English-only cues (no native Chinese line) in 'native' mode.
    送去翻译的 cue：'translate' 模式下每条带英文的 cue，'native' 模式下只有纯英文（没有原生中文行）的 cue。
    """
    chosen = window_cues(window, cues)
    if mode == "native":
        return [c for c in chosen if c.text_en and not c.text_zh]
    return [c for c in chosen if c.text_en]


def build_requests(
    windows: list[Window], transcripts: dict[str, list[Cue]], *, size: int = 25, mode: str = "translate", skip: set[str] | None = None
) -> list[Request]:
    """Group the windows' English lines into requests of at most `size` windows that never span two transcript files.
    把各窗口的英文行分组为至多 `size` 个窗口、绝不跨转写文件的请求。

    Input / 输入：
        windows: windows.build_windows output (English mode).
        windows：windows.build_windows 的输出（英文模式）。

        transcripts: The cues the windows were built from.
        transcripts：构建窗口所用的 cue。

        size: Windows per request (25).
        size：每请求窗口数（25）。

        mode: 'translate' | 'native' (which cues are sent; see cues_to_translate).
        mode：'translate' | 'native'（送哪些 cue；见 cues_to_translate）。

        skip: Window ids already translated (resume); None → none skipped.
        skip：已翻译的窗口 id（续跑）；None → 不跳过。

    Output / 输出：
        Requests in window order with ids r0000…; windows with nothing to translate are left out.
        按窗口顺序、id 为 r0000… 的请求；无需翻译的窗口不入请求。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[Request] = []
    current: list[Item] = []
    current_source: str | None = None

    def flush() -> None:
        nonlocal current
        if current:
            out.append(Request(f"r{len(out):04d}", current))
            current = []

    for window in windows:
        if skip and window.window_id in skip:
            continue
        lines = [c.text_en for c in cues_to_translate(window, transcripts[window.source], mode)]
        if not lines:
            continue
        if current_source != window.source or len(current) >= size:
            flush()
            current_source = window.source
        current.append(Item(window.window_id, lines))
    flush()
    return out


def strip_label(line: str) -> str:
    """The line without its speaker label.
    去掉说话人标签后的行。
    """
    return LABEL_RE.sub("", line, count=1)


def latin_letter_rate(lines: list[str]) -> float:
    """Share of Latin letters among the non-space characters of the lines after stripping speaker labels (0.0 when empty).
    去掉说话人标签后，各行非空白字符中拉丁字母的占比（为空时 0.0）。
    """
    body = "".join(strip_label(line) for line in lines)
    chars = [ch for ch in body if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if LATIN_RE.match(ch)) / len(chars)


def validate_window(src_lines: list[str], out_lines: Any, *, question: bool = False) -> str | None:
    """Reject a translation that is empty, has another line count, an empty line where the source has text, a body
    length outside LENGTH_RATIO × the source, or more than 50% Latin letters; None when it passes. For a question item
    (`question=True`: the question line + 4 options) the Latin-letter rule reads the question line only, because the
    options are mostly bare proper names the instruction tells the model to keep Latin.
    拒绝为空、行数不同、源有文本而译行为空、正文长度在源的 LENGTH_RATIO 倍之外、或拉丁字母超过 50% 的译文；通过时返回 None。
    题目单元（`question=True`：题干 + 4 个选项）的拉丁字母规则只看题干行，因为选项多是指令要求保留拉丁形式的裸人名。

    Input / 输入：
        src_lines: The English lines sent.
        src_lines：送出的英文行。

        out_lines: The model's `lines` value (any JSON value).
        out_lines：模型的 `lines` 值（任意 JSON 值）。

        question: True for a 'q:<ID>' item (translate_items passes it from the item id).
        question：'q:<ID>' 单元为 True（translate_items 按单元 id 传入）。

    Output / 输出：
        A short reason string or None.
        简短的原因字符串或 None。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(out_lines, list) or not out_lines:
        return "empty"
    if len(out_lines) != len(src_lines):
        return f"line_count {len(out_lines)} != {len(src_lines)}"
    if any(not isinstance(line, str) for line in out_lines):
        return "non_string_line"
    if any(src.strip() and not out.strip() for src, out in zip(src_lines, out_lines)):
        return "empty_line"
    src_len = sum(len(strip_label(s)) for s in src_lines)
    out_len = sum(len(strip_label(o)) for o in out_lines)
    if src_len and not (LENGTH_RATIO[0] * src_len <= out_len <= LENGTH_RATIO[1] * src_len):
        return f"length_ratio {out_len / src_len:.2f}"
    judged = out_lines[:1] if question else out_lines
    if latin_letter_rate(judged) > 0.5:
        return f"latin_rate {latin_letter_rate(judged):.2f}"
    return None


def cny_cost(hit: int | None, miss: int | None, output: int | None, tier: str) -> float | None:
    """CNY of a token triple at one tier (None when any count is unknown); the reader_ab formula.
    某档位下一组 token 的人民币费用（任一计数未知时为 None）；与 reader_ab 同一公式。
    """
    if hit is None or miss is None or output is None:
        return None
    p = PRICES_CNY_PER_M[tier]
    return (hit * p["cache_hit"] + miss * p["cache_miss"] + output * p["output"]) / 1_000_000


def project_cost(requests: list[Request], *, out_ratio: float = 0.44, chars_per_token_out: float = 1.45) -> dict[str, Any]:
    """Pre-flight token and cost projection: input ≈ Σ en_chars / 4 + 400 per request (instruction + JSON scaffolding),
    output ≈ Σ en_chars × out_ratio / chars_per_token_out + 15 per window; CNY / USD at both tiers.
    调用前的 token 与费用预估：输入 ≈ Σ 英文字符 / 4 + 每请求 400（指令 + JSON 骨架），输出 ≈ Σ 英文字符 × out_ratio /
    chars_per_token_out + 每窗口 15；两个档位的人民币 / 美元费用。

    Input / 输入：
        requests: build_requests output.
        requests：build_requests 的输出。

        out_ratio: zh chars per en char (0.44 measured on the native A1 lines).
        out_ratio：每个英文字符对应的中文字符数（按 A1 原生行实测 0.44）。

        chars_per_token_out: zh chars per output token (1.45).
        chars_per_token_out：每个输出 token 的中文字符数（1.45）。

    Output / 输出：
        {'requests', 'windows', 'en_chars', 'in_tokens', 'out_tokens', 'cny': {tier: …}, 'usd': {tier: …}}.
        {'requests', 'windows', 'en_chars', 'in_tokens', 'out_tokens', 'cny': {档位: …}, 'usd': {档位: …}}。

    State effect / 状态影响：
        None.
        无。
    """
    en_chars = sum(len(line) for r in requests for it in r.items for line in it.lines)
    n_windows = sum(len(r.items) for r in requests)
    in_tokens = int(en_chars / 4 + 400 * len(requests))
    out_tokens = int(en_chars * out_ratio / chars_per_token_out + 15 * n_windows)
    cny = {tier: round((in_tokens * p["cache_miss"] + out_tokens * p["output"]) / 1_000_000, 4) for tier, p in PRICES_CNY_PER_M.items()}
    usd = {tier: round((in_tokens * p["cache_miss"] + out_tokens * p["output"]) / 1_000_000, 4) for tier, p in PRICES_USD_PER_M.items()}
    return {"requests": len(requests), "windows": n_windows, "en_chars": en_chars, "in_tokens": in_tokens, "out_tokens": out_tokens, "cny": cny, "usd": usd}


# ---------------------------------------------------------------------------
# Cache and ledger
# 缓存与台账
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict[str, Translation]:
    """Read the JSONL cache; the last row per window id wins.
    读取 JSONL 缓存；每个窗口 id 以最后一行为准。
    """
    out: dict[str, Translation] = {}
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8").splitlines()
    for n, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as exc:
            # Rows are appended per request; a kill mid-write leaves one partial LAST line, which resume skips with a
            # warning (the window is simply re-requested). A malformed line anywhere else is corruption: fail loudly.
            # 行按请求追加；写到一半被杀会留下一条不完整的末行，续跑时告警跳过（该窗口只是被重新请求）。其它位置的畸形行
            # 是损坏：明确报错。
            if n == len(lines):
                print(f"[translate_zh] warning: skipping truncated last cache line {n} of {path}", file=sys.stderr, flush=True)
                continue
            raise ValueError(f"malformed cache line {n} of {path}: {exc}") from exc
        out[row["window_id"]] = Translation(
            row["window_id"], list(row.get("lines") or []), row.get("request_id", ""), int(row.get("attempt", 0)),
            bool(row.get("validated")), row.get("reason"), row.get("model", MODEL),
        )  # fmt: skip
    return out


def resume_skip_ids(cache: dict[str, Translation], *, retry_failed: bool = False) -> tuple[set[str], set[str]]:
    """(done, failed_final) of a cache: `done` = validated ids; `failed_final` = ids whose single-window retry
    (attempt ≥ 1) failed for a reason other than the cost cap ('cap' rows were never translated). A resume skips
    `done`, and `failed_final` too unless `retry_failed` — otherwise every resume would re-bill the same windows.
    缓存的 (done, failed_final)：`done` = 已通过校验的 id；`failed_final` = 单窗口重试（attempt ≥ 1）因费用上限之外的原因
    失败的 id（'cap' 行从未翻译过）。续跑跳过 `done`，除 `retry_failed` 外也跳过 `failed_final`——否则每次续跑都会为同一批
    窗口再付费。

    Input / 输入：
        cache: load_cache output.
        cache：load_cache 的输出。

        retry_failed: --retry-failed; True re-opens the final failures (they are then not in the second set).
        retry_failed：--retry-failed；为真时重新打开最终失败（此时不在第二个集合里）。

    Output / 输出：
        (done ids, final-failure ids to skip).
        （已完成 id，需跳过的最终失败 id）。

    State effect / 状态影响：
        None.
        无。
    """
    done = {wid for wid, tr in cache.items() if tr.validated}
    if retry_failed:
        return done, set()
    failed_final = {wid for wid, tr in cache.items() if not tr.validated and tr.attempt >= 1 and tr.reason != "cap"}
    return done, failed_final


class TranslateMeter(Meter):
    """accounting.Meter with one lock around every mutation and a CNY total, shared by all worker clients.
    每次变更都加锁并带人民币合计的 accounting.Meter，供所有工作线程的客户端共用。
    """

    def __init__(self) -> None:
        """Empty ledger plus the lock.
        空台账加锁。
        """
        super().__init__()
        self._lock = threading.Lock()

    def next_is_cold(self, arm: str) -> bool:
        """Meter.next_is_cold under the lock.
        加锁的 Meter.next_is_cold。
        """
        with self._lock:
            return super().next_is_cold(arm)

    def add(self, call: Any) -> Any:
        """Meter.add under the lock.
        加锁的 Meter.add。
        """
        with self._lock:
            return super().add(call)

    def add_embedding(self, arm: str, stage: str, texts: int, local_ms: float) -> Any:
        """Meter.add_embedding under the lock.
        加锁的 Meter.add_embedding。
        """
        with self._lock:
            return super().add_embedding(arm, stage, texts, local_ms)

    def spent_cny(self) -> float:
        """CNY of every priced attempt at the tier it ran (unpriced attempts count 0).
        每次已计价尝试按其实际档位的人民币合计（未计价的尝试计 0）。
        """
        with self._lock:
            return sum(cny_cost(c.cache_hit_tokens, c.cache_miss_tokens, c.completion_tokens, c.rate_tier) or 0.0 for c in self.calls)

    def cost_summary(self) -> dict[str, Any]:
        """Calls, retries, tokens by kind, CNY (actual by tier and all-offpeak / all-peak), USD.
        调用数、重试数、各类 token、人民币（按实际档位与全部按空闲 / 全部按高峰）、美元。
        """
        with self._lock:
            calls = list(self.calls)
        hit = sum(c.cache_hit_tokens or 0 for c in calls)
        miss = sum(c.cache_miss_tokens or 0 for c in calls)
        out = sum(c.completion_tokens or 0 for c in calls)
        return {
            "calls": len(calls),
            "retries": sum(1 for c in calls if c.attempt > 0),
            "failed_attempts": sum(1 for c in calls if c.status != "ok"),
            "truncated_attempts": sum(1 for c in calls if c.finish_reason == "length"),
            "tokens": {"prompt": sum(c.prompt_tokens or 0 for c in calls), "cache_hit": hit, "cache_miss": miss, "completion": out},
            "by_tier": {t: sum(1 for c in calls if c.rate_tier == t) for t in ("peak", "offpeak")},
            "cost_cny": {
                "actual": round(sum(cny_cost(c.cache_hit_tokens, c.cache_miss_tokens, c.completion_tokens, c.rate_tier) or 0.0 for c in calls), 4),
                "all_offpeak": round(cny_cost(hit, miss, out, "offpeak") or 0.0, 4),
                "all_peak": round(cny_cost(hit, miss, out, "peak") or 0.0, 4),
            },
            "cost_usd": round(sum(c.cost_usd or 0.0 for c in calls), 6),
        }


class CostCapExceeded(RuntimeError):
    """Raised when the running ledger passes --cost-cap-cny; in-flight requests finish, no new one starts.
    累计台账超过 --cost-cap-cny 时抛出；飞行中的请求完成，不再发起新请求。
    """


class SpendLedger:
    """The cumulative CNY spend of every invocation on one cache (translate_spent.json next to it), so --cost-cap-cny
    bounds the dataset, not one run: `spent_before` is the sum of earlier invocations, `record(current)` rewrites the
    file with this invocation's running total after every priced call (a killed run therefore loses at most one call).
    同一缓存上每次调用脚本的累计人民币花费（缓存旁的 translate_spent.json），使 --cost-cap-cny 约束整个数据集而非单次运行：
    `spent_before` 为既往调用之和，`record(current)` 在每次计费调用后用本次的累计值重写文件（被杀的运行至多丢一次调用）。
    """

    def __init__(self, path: Path, started_utc: str) -> None:
        """Load the previous runs; the current run starts at 0.
        读取既往运行；本次运行从 0 起。
        """
        self.path = path
        self.started_utc = started_utc
        self.runs: list[dict[str, Any]] = []
        if path.exists():
            try:
                self.runs = list(json.loads(path.read_text(encoding="utf-8")).get("runs") or [])
            except ValueError:
                self.runs = []
        self.spent_before = round(sum(float(r.get("cost_cny") or 0.0) for r in self.runs), 6)
        self.current = 0.0
        self._lock = threading.Lock()

    def record(self, current: float, *, calls: int | None = None, stopped: str | None = None, finished: bool = False) -> None:
        """Rewrite the file with the earlier runs plus this one at `current` CNY.
        以既往运行加上本次（`current` 元）重写文件。
        """
        with self._lock:
            self.current = float(current)
            entry: dict[str, Any] = {"started_utc": self.started_utc, "cost_cny": round(self.current, 6)}
            if calls is not None:
                entry["calls"] = calls
            if stopped:
                entry["stopped"] = stopped
            if finished:
                entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
            payload = {"runs": self.runs + [entry], "spent_cny": round(self.spent_before + self.current, 6)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    @property
    def total(self) -> float:
        """Earlier runs plus the current one.
        既往运行加本次。
        """
        return self.spent_before + self.current


# ---------------------------------------------------------------------------
# API round trip
# API 往返
# ---------------------------------------------------------------------------
def _user_payload(items: list[Item]) -> str:
    """The request body: the items as JSON.
    请求正文：以 JSON 表示的单元。
    """
    return json.dumps({"items": [{"window_id": it.window_id, "lines": it.lines} for it in items]}, ensure_ascii=False)


def _parse_reply(content: str) -> dict[str, Any]:
    """{window_id: lines} out of the reply JSON (a bare list is read as the items array; anything else → {}).
    从回复 JSON 得到 {window_id: lines}（裸列表按 items 数组读；其它 → {}）。
    """
    try:
        raw = json.loads(content)
    except (TypeError, ValueError):
        return {}
    items = raw.get("items") if isinstance(raw, dict) else raw
    out: dict[str, Any] = {}
    for it in items or []:
        if isinstance(it, dict) and it.get("window_id") is not None:
            out[str(it["window_id"])] = it.get("lines")
    return out


def translate_items(
    client: Any,
    items: list[Item],
    request_id: str,
    attempt: int,
    *,
    on_call: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Translation]:
    """One request (bisected on finish_reason == 'length' like the harness; a single truncated window fails) → one
    Translation per item, validated by validate_window. Once `should_stop()` is true (the cost cap tripped) no further
    request is issued: the items still pending come back as validated=False rows with reason 'cap', which a resume
    re-requests.
    一次请求（finish_reason == 'length' 时像基准脚本一样二分；单个仍被截断的窗口判失败）→ 每单元一条 Translation，
    经 validate_window 校验。`should_stop()` 为真（费用上限已触发）后不再发起任何请求：尚未处理的单元以 validated=False、
    reason 'cap' 的行返回，续跑时重新请求。

    Input / 输入：
        client: real_arms.DeepSeekClient.
        client：real_arms.DeepSeekClient。

        items: The units of this request (one file).
        items：本请求的单元（同一文件）。

        request_id: For the cache rows and the ledger's batch_id.
        request_id：用于缓存行与台账的 batch_id。

        attempt: 0 = batch round, 1 = the single-window retry.
        attempt：0 = 批次轮，1 = 单窗口重试轮。

        on_call: Called after every API call (the cost-cap check).
        on_call：每次 API 调用后调用（费用上限检查）。

        should_stop: Consulted before every request, including the bisect halves; None → never stop.
        should_stop：每次请求（含二分的两半）之前查询；None → 从不停止。

    Output / 输出：
        Translations in item order.
        按单元顺序的 Translation。

    State effect / 状态影响：
        Network calls through the client (ledger rows).
        经客户端的网络调用（台账行）。
    """
    if should_stop is not None and should_stop():
        return [Translation(it.window_id, [], request_id, attempt, False, "cap") for it in items]
    with client.label(arm="translate", stage="translate", batch_id=request_id):
        try:
            response = client.create(
                messages=[{"role": "system", "content": TRANSLATE_SYSTEM}, {"role": "user", "content": _user_payload(items)}],
                temperature=TRANSLATE_TEMPERATURE,
                max_tokens=TRANSLATE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - every failure becomes cache rows / 每个失败都写成缓存行
            if on_call is not None:
                on_call()
            return [Translation(it.window_id, [], request_id, attempt, False, f"api_error: {type(exc).__name__}") for it in items]
    if on_call is not None:
        on_call()
    last = client.last_calls[-1] if client.last_calls else None
    if last is not None and last.finish_reason == "length":
        if len(items) == 1:
            return [Translation(items[0].window_id, [], request_id, attempt, False, "truncated")]
        half = len(items) // 2
        left = translate_items(client, items[:half], f"{request_id}a", attempt, on_call=on_call, should_stop=should_stop)
        right = translate_items(client, items[half:], f"{request_id}b", attempt, on_call=on_call, should_stop=should_stop)
        return left + right
    choice = response.choices[0] if getattr(response, "choices", None) else None
    content = getattr(getattr(choice, "message", None), "content", None) or ""
    parsed = _parse_reply(content)
    out: list[Translation] = []
    for it in items:
        lines = parsed.get(it.window_id)
        reason = validate_window(it.lines, lines, question=it.window_id.startswith("q:")) if lines is not None else "missing_window"
        clean = [str(line).strip() for line in lines] if reason is None else []
        out.append(Translation(it.window_id, clean, request_id, attempt, reason is None, reason))
    return out


def translate_all(
    client_factory: Callable[[], Any],
    requests: list[Request],
    cache_path: Path,
    meter: TranslateMeter,
    *,
    cap_cny: float,
    workers: int = 3,
    attempt: int = 0,
    progress: Callable[[str], None] | None = None,
    spent_before: float = 0.0,
    on_spend: Callable[[float], None] | None = None,
) -> dict[str, Translation]:
    """Run the requests over `workers` threads (one client each, one shared meter), appending every result to the JSONL
    cache as it lands; stop starting new requests (bisect halves included) once `spent_before` + the ledger passes the cap.
    用 `workers` 个线程（各自一个客户端、共用一个台账）执行请求，结果落地即追加到 JSONL 缓存；`spent_before` + 台账超过
    上限后不再发起新请求（含二分的两半）。

    Input / 输入：
        client_factory: () -> DeepSeekClient bound to `meter`.
        client_factory：() -> 绑定到 `meter` 的 DeepSeekClient。

        requests: build_requests output for the still-missing windows.
        requests：仍缺失窗口的 build_requests 输出。

        cache_path: The JSONL cache (appended).
        cache_path：JSONL 缓存（追加）。

        meter: The shared TranslateMeter.
        meter：共用的 TranslateMeter。

        cap_cny: --cost-cap-cny.
        cap_cny：--cost-cap-cny。

        workers: Thread count.
        workers：线程数。

        attempt: Cache row attempt number (0 batch round, 1 retry round).
        attempt：缓存行的尝试序号（0 批次轮，1 重试轮）。

        progress: Optional line printer.
        progress：可选的进度打印函数。

        spent_before: CNY already spent by earlier invocations (SpendLedger.spent_before); counted against the cap.
        spent_before：既往调用脚本已花的人民币（SpendLedger.spent_before）；计入上限。

        on_spend: Called under the meter's cap check after every API call with this run's running CNY (the ledger file).
        on_spend：每次 API 调用后在上限检查中以本次运行的累计人民币调用（写台账文件）。

    Output / 输出：
        {window_id: Translation} for every item of the requests that ran.
        已运行请求的每个单元的 {window_id: Translation}。

    State effect / 状态影响：
        Network calls; appends cache rows; raises CostCapExceeded after finishing in-flight requests when the cap is hit.
        网络调用；追加缓存行；超过上限时在飞行中请求结束后抛 CostCapExceeded。
    """
    workers = max(1, int(workers))
    clients: queue.Queue = queue.Queue()
    for _ in range(workers):
        clients.put(client_factory())
    write_lock = threading.Lock()
    results: dict[str, Translation] = {}
    stop = threading.Event()

    def check_cap() -> None:
        spent = meter.spent_cny()
        if on_spend is not None:
            on_spend(spent)
        if spent_before + spent > cap_cny:
            stop.set()

    def one(request: Request) -> list[Translation]:
        if stop.is_set():
            return []
        client = clients.get()
        try:
            rows = translate_items(client, request.items, request.request_id, attempt, on_call=check_cap, should_stop=stop.is_set)
        finally:
            clients.put(client)
        with write_lock:
            with cache_path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
            for row in rows:
                results[row.window_id] = row
        return rows

    done = 0
    printed = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(requests), workers):
            if stop.is_set():
                break
            chunk = requests[start : start + workers]
            futures = [pool.submit(one, r) for r in chunk]
            for fut in futures:
                rows = fut.result()
                done += 1 if rows else 0
            # Chunks complete `workers` requests at a time, so test the decade crossed since the last line, not `done % 10`.
            # 每块一次完成 `workers` 个请求，因此判断自上次打印以来是否跨过一个十位，而不是 `done % 10`。
            if progress is not None and done // 10 > printed // 10:
                printed = done
                progress(f"  [translate] requests {done}/{len(requests)} windows {len(results)} cost_cny={meter.spent_cny():.4f}")
    if stop.is_set():
        raise CostCapExceeded(
            f"running cost {meter.spent_cny():.4f} CNY (+ {spent_before:.4f} CNY from earlier runs) exceeds cap {cap_cny} CNY after {done} requests"
        )
    return results


# ---------------------------------------------------------------------------
# Dataset writer and self-check
# 数据集写入与自检
# ---------------------------------------------------------------------------
def _zh_line(cue: Cue, translated: str | None, glossary: dict[str, str]) -> tuple[str | None, str]:
    """The Chinese line written for a cue and its provenance: the translation (glossary applied) when it carries a CJK
    character, else the native Chinese line (glossary applied), else none.
    为某 cue 写出的中文行及其来源：译文（套用术语表）含 CJK 字符时取它，否则原生中文行（套用术语表），否则没有。
    """
    if translated is not None:
        candidate = apply_glossary(translated.strip(), glossary)
        if CJK_RE.search(candidate):
            return candidate, "translated"
    if cue.text_zh:
        return apply_glossary(cue.text_zh, glossary), "native"
    return None, "none"


def write_dataset(
    out_root: Path,
    data_root: Path,
    transcripts: dict[str, list[Cue]],
    windows: list[Window],
    translations: dict[str, Translation],
    questions_raw: list[dict],
    *,
    mode: str = "translate",
    questions_source: str = "native",
    glossary: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write data_zh: every SRT block keeps its index and timing lines and its English line verbatim; the Chinese line
    is the translation mapped back by position, else the native line, each with the glossary applied; the QA file
    keeps every item verbatim except question / choice_x, replaced by the glossary-applied Chinese fields.
    写出 data_zh：每个 SRT 块保留序号行、时间行与英文行原文；中文行是按位置映射回来的译文，否则为原生行，都套用术语表；
    QA 文件除 question / choice_x（换成套用术语表的中文字段）外逐项原样。

    Input / 输入：
        out_root: data_zh/.
        out_root：data_zh/。

        data_root: data/ (the source SRT files are re-read block by block).
        data_root：data/（逐块重读源 SRT 文件）。

        transcripts: loader.load_transcripts(data_root).
        transcripts：loader.load_transcripts(data_root)。

        windows: build_windows(transcripts) (English mode).
        windows：build_windows(transcripts)（英文模式）。

        translations: {window_id: Translation} (validated rows are used; others fall back to native lines).
        translations：{window_id: Translation}（用通过校验的行；其余回退到原生行）。

        questions_raw: The QA JSON array.
        questions_raw：QA JSON 数组。

        mode: 'translate' | 'native' (which cues the translation lines map to; cues_to_translate).
        mode：'translate' | 'native'（译文行映射到哪些 cue；cues_to_translate）。

        questions_source: 'native' → question_chinese / choice_x_chinese; 'translate' → translations['q:<ID>'] lines.
        questions_source：'native' → question_chinese / choice_x_chinese；'translate' → translations['q:<ID>'] 的行。

        glossary: None → NAME_GLOSSARY.
        glossary：None → NAME_GLOSSARY。

    Output / 输出：
        Per-window provenance counts, per-cue counts, glossary hit counts, the windows written without any Chinese line,
        and the question provenance (questions_translated / questions_native_fallback — the latter only counts under
        questions_source='translate', where a native question is a fallback rather than the source).
        逐窗口来源计数、逐 cue 计数、术语表命中计数、没有任何中文行的窗口，以及题目来源（questions_translated /
        questions_native_fallback——后者只在 questions_source='translate' 下计数，那时取原生题目是回退而非来源）。

    State effect / 状态影响：
        Writes the SRT files and the QA file under out_root.
        在 out_root 下写 SRT 文件与 QA 文件。
    """
    table = NAME_GLOSSARY if glossary is None else glossary
    by_source: dict[str, list[Window]] = {}
    for w in windows:
        by_source.setdefault(w.source, []).append(w)
    # (source, position in the cue list) → (window_id, translated line or None)
    # （文件, cue 列表中的位置）→ （窗口 id, 译文行或 None）
    cue_line: dict[tuple[str, int], tuple[str, str | None]] = {}
    window_kind: dict[str, str] = {}
    for source, cues in transcripts.items():
        pos_of = {id(c): i for i, c in enumerate(cues)}
        for w in by_source.get(source, []):
            targets = cues_to_translate(w, cues, mode)
            tr = translations.get(w.window_id)
            usable = tr is not None and tr.validated and len(tr.lines) == len(targets)
            # The same predicate as windows.build_windows(text_lang='zh'): a cue contributes Chinese only when it also
            # carries English text; a validated translation without any CJK character falls back per line (_zh_line).
            # 与 windows.build_windows(text_lang='zh') 同一谓词：cue 只在同时带英文文本时贡献中文；通过校验却不含 CJK
            # 字符的译文按行回退（_zh_line）。
            native_zh = any(c.text_en and c.text_zh for c in window_cues(w, cues))
            translated_zh = usable and tr is not None and any(CJK_RE.search(line) for line in tr.lines)
            window_kind[w.window_id] = "translated" if translated_zh else ("native_fallback" if native_zh else "no_chinese")
            for i, cue in enumerate(targets):
                cue_line[(source, pos_of[id(cue)])] = (w.window_id, tr.lines[i] if usable and tr is not None else None)
            for cue in window_cues(w, cues):
                cue_line.setdefault((source, pos_of[id(cue)]), (w.window_id, None))
    cue_counts = {"translated": 0, "native": 0, "none": 0}
    glossary_used: dict[str, int] = {k: 0 for k in table}
    key_re = {k: re.compile(r"(?<![A-Za-z])" + re.escape(k) + r"(?:'s)?(?![A-Za-z])") for k in table}

    def _count(text: str) -> None:
        for k, pat in key_re.items():
            glossary_used[k] += len(pat.findall(text))

    for path in transcript_files(data_root):
        source = path.stem
        cues = transcripts[source]
        out_blocks: list[str] = []
        position = 0
        for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8").strip()):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 3 or not re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->", lines[1]):
                out_blocks.append("\n".join(lines))
                continue
            cue = cues[position]
            _wid, translated = cue_line.get((source, position), (None, None))
            position += 1
            en_lines = [line for line in lines[2:] if CJK_RE.search(line) is None]
            zh_out, kind = _zh_line(cue, translated, table)
            cue_counts[kind] += 1
            body = [lines[0], lines[1]]
            if zh_out is not None:
                _count(translated if kind == "translated" and translated is not None else cue.text_zh)
                body.append(zh_out)
            body.extend(en_lines)
            out_blocks.append("\n".join(body))
        target = out_root / TRANSCRIPT_SUBDIR / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")
    items_out: list[dict] = []
    question_counts = {"translated": 0, "native": 0}
    for item in questions_raw:
        copy = dict(item)
        if questions_source == "translate" and (tr := translations.get(f"q:{item['ID']}")) is not None and tr.validated and len(tr.lines) == 5:
            texts = tr.lines
            question_counts["translated"] += 1
        else:
            texts = [str(item.get("question_chinese") or item["question"])] + [str(item.get(f"choice_{k}_chinese") or item[f"choice_{k}"]) for k in "abcd"]
            question_counts["native"] += 1
        for text in texts:
            _count(text)
        copy["question"] = apply_glossary(texts[0].strip(), table)
        for k, text in zip("abcd", texts[1:]):
            copy[f"choice_{k}"] = apply_glossary(text.strip(), table)
        items_out.append(copy)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / QA_FILENAME).write_text(json.dumps(items_out, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = {k: sum(1 for v in window_kind.values() if v == k) for k in ("translated", "native_fallback", "no_chinese")}
    return {
        "windows": kinds,
        "windows_native_fallback": sorted(w for w, k in window_kind.items() if k == "native_fallback"),
        "windows_no_chinese": sorted(w for w, k in window_kind.items() if k == "no_chinese"),
        "cues": cue_counts,
        "glossary_used": glossary_used,
        "questions": len(items_out),
        "questions_translated": question_counts["translated"],
        "questions_native_fallback": question_counts["native"] if questions_source == "translate" else 0,
    }


def self_check(out_root: Path, source_windows: list[Window], questions: list[Any], *, max_chars: int = MAX_CHARS) -> dict[str, Any]:
    """Reload data_zh through the unchanged loader: same question ids, and windows (English mode) identical in id,
    clock, cue range and English text to the source; the zh-mode build reports how many windows have Chinese text.
    用未改动的加载器重读 data_zh：题目 id 相同，窗口（英文模式）在 id、时钟、cue 区间与英文文本上与源一致；zh 模式
    构建报告有多少窗口带中文文本。

    Input / 输入：
        out_root: data_zh/.
        out_root：data_zh/。

        source_windows: build_windows over data/.
        source_windows：data/ 上的 build_windows。

        questions: load_questions over data/.
        questions：data/ 上的 load_questions。

        max_chars: The window character budget the source windows were built with (windows.MAX_CHARS).
        max_chars：构建源窗口所用的窗口字符预算（windows.MAX_CHARS）。

    Output / 输出：
        {'questions_ok', 'question_ids', 'windows_ok', 'windows', 'mismatches', 'zh_windows_ok', 'zh_error'}.
        {'questions_ok', 'question_ids', 'windows_ok', 'windows', 'mismatches', 'zh_windows_ok', 'zh_error'}。

    State effect / 状态影响：
        None (reads out_root).
        无（读取 out_root）。
    """
    zh_questions = load_questions(out_root / QA_FILENAME)
    questions_ok = [q.id for q in zh_questions] == [q.id for q in questions] and all(
        (a.answer, a.targets, a.query_cs) == (b.answer, b.targets, b.query_cs) for a, b in zip(zh_questions, questions)
    )
    transcripts = load_transcripts(out_root)
    windows = build_windows(transcripts, max_chars=max_chars)
    key = lambda w: (w.window_id, w.day, w.start_cs, w.end_cs, w.cue_from, w.cue_to, w.text)  # noqa: E731
    mismatches = [a.window_id for a, b in zip(windows, source_windows) if key(a) != key(b)]
    windows_ok = len(windows) == len(source_windows) and not mismatches
    zh_error: str | None = None
    try:
        zh_windows = build_windows(transcripts, max_chars=max_chars, text_lang="zh")
        zh_ok = len(zh_windows) == len(source_windows) and all(w.text for w in zh_windows)
    except ValueError as exc:
        zh_ok, zh_error = False, str(exc)
    return {
        "questions_ok": questions_ok, "question_ids": len(zh_questions), "windows_ok": windows_ok, "windows": len(windows),
        "mismatches": mismatches[:20], "zh_windows_ok": zh_ok, "zh_error": zh_error,
    }  # fmt: skip


def variant_rates(windows: list[Window], transcripts: dict[str, list[Cue]], translations: dict[str, Translation], mode: str, glossary: dict[str, str]) -> dict[str, Any]:
    """Per glossary name: windows whose English mentions it and whose validated translation carries neither the Latin
    form nor the glossary form (a transliteration variant or a dropped name).
    对每个术语表人名：英文提到它、而通过校验的译文既无拉丁形式也无术语表形式的窗口（音译变体或丢名）。

    Input / 输入：
        windows, transcripts, translations, mode, glossary: As in write_dataset.
        windows、transcripts、translations、mode、glossary：同 write_dataset。

    Output / 输出：
        {name: {'mentions', 'missing', 'rate', 'examples'}}.
        {人名: {'mentions', 'missing', 'rate', 'examples'}}。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, Any] = {}
    for latin, zh in glossary.items():
        pat = re.compile(r"(?<![A-Za-z])" + re.escape(latin) + r"(?![A-Za-z])")
        mentions = missing = 0
        examples: list[str] = []
        for w in windows:
            tr = translations.get(w.window_id)
            if tr is None or not tr.validated:
                continue
            src = [c.text_en for c in cues_to_translate(w, transcripts[w.source], mode)]
            if not any(pat.search(s) for s in src):
                continue
            mentions += 1
            joined = "\n".join(tr.lines)
            if not pat.search(joined) and zh not in joined:
                missing += 1
                if len(examples) < 5:
                    examples.append(w.window_id)
        out[latin] = {"mentions": mentions, "missing": missing, "rate": round(missing / mentions, 4) if mentions else None, "examples": examples}
    return out


# ---------------------------------------------------------------------------
# CLI
# 命令行
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The flags (see the module docstring).
    命令行参数（见模块文档字符串）。
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", type=Path, default=None, help="default <out>/translate_cache.jsonl")
    p.add_argument("--report", type=Path, default=None, help="default <out>/translate_report.json")
    p.add_argument("--transcript-source", choices=("translate", "native"), default="translate")
    p.add_argument("--questions-source", choices=("native", "translate"), default="native")
    p.add_argument("--windows-per-request", type=int, default=25)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cost-cap-cny", type=float, default=6.0, help="cumulative over resumes of the same cache (translate_spent.json)")
    # '%%' because argparse %-formats help strings (Python 3.14 validates them at add_argument time).
    # 写成 '%%'，因为 argparse 对帮助文本做 % 格式化（Python 3.14 在 add_argument 时就校验）。
    p.add_argument("--allow-tight-cap", action="store_true", help=f"run even when the projection exceeds {TIGHT_CAP_FRACTION:.0%}% of the remaining cap")
    p.add_argument("--retry-failed", action="store_true", help="re-send windows whose single-window retry already failed (otherwise they are final and only reported)")
    p.add_argument("--limit", type=int, default=None, help="only the first N windows (dry runs); the self-check is then informational")
    p.add_argument("--dry-run", action="store_true", help="projection only; no API call, nothing written except the report")
    p.add_argument("--glossary-add", action="append", default=[], metavar="NAME=中文",
                   help="extra whole-word glossary pairs on top of NAME_GLOSSARY (repeatable; e.g. Jack=杰克 when Jack is the ASR variant of Jake — a user decision, see EGOLIFE_ZH_PLAN §3); recorded as glossary_added")
    return p


def parse_glossary_add(pairs: list[str]) -> dict[str, str]:
    """`NAME=中文` flag values → {Latin: Chinese}; a value without '=' or with an empty side raises ValueError.
    `NAME=中文` 形式的参数值 → {拉丁: 中文}；没有 '=' 或任一侧为空时抛 ValueError。

    Input / 输入：
        pairs: The --glossary-add values.
        pairs：--glossary-add 的值。

    Output / 输出：
        {Latin: Chinese} in flag order.
        按参数顺序的 {拉丁: 中文}。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, str] = {}
    for raw in pairs:
        key, sep, value = raw.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"--glossary-add expects NAME=中文, got {raw!r}")
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    """Translate (resumable), write data_zh, self-check, report.
    翻译（可续跑）、写出 data_zh、自检、报告。

    Input / 输入：
        argv: CLI arguments.
        argv：命令行参数。

    Output / 输出：
        Exit code (0; 2 when the projection or the running ledger exceeds the cap; 3 when the self-check fails on a full run).
        退出码（0；预估或累计台账超上限为 2；完整运行自检失败为 3）。

    State effect / 状态影响：
        See the module docstring.
        见模块文档字符串。
    """
    args = build_parser().parse_args(argv)
    out: Path = args.out
    cache_path: Path = args.cache or (out / "translate_cache.jsonl")
    report_path: Path = args.report or (out / "translate_report.json")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    questions = load_questions(args.data_root / QA_FILENAME)
    questions_raw = json.loads((args.data_root / QA_FILENAME).read_text(encoding="utf-8"))
    transcripts = load_transcripts(args.data_root)
    windows = build_windows(transcripts)
    target_windows = windows[: args.limit] if args.limit else windows
    names = collect_names(transcripts, questions)
    glossary_added = parse_glossary_add(args.glossary_add)
    glossary = {**NAME_GLOSSARY, **glossary_added}
    latin_kept = {n: c for n, c in names.items() if n not in glossary}
    cache = load_cache(cache_path)
    # A window whose single-window retry (attempt 1) failed is final: reported, not re-billed on every resume
    # (--retry-failed re-opens it). Rows the cost cap cut short ('cap') were never translated and are always resent.
    # 单窗口重试（attempt 1）失败的窗口为最终状态：只报告，不在每次续跑时再计费（--retry-failed 重新打开）。被费用上限
    # 截断的行（'cap'）从未翻译过，总是重发。
    done_ids, failed_final = resume_skip_ids(cache, retry_failed=args.retry_failed)
    skip_ids = done_ids | failed_final
    requests = build_requests(target_windows, transcripts, size=args.windows_per_request, mode=args.transcript_source, skip=skip_ids)
    if args.questions_source == "translate":
        q_items = [
            Item(f"q:{item['ID']}", [str(item["question"])] + [str(item[f"choice_{k}"]) for k in "abcd"])
            for item in questions_raw if f"q:{item['ID']}" not in skip_ids
        ]  # fmt: skip
        for i in range(0, len(q_items), args.windows_per_request):
            requests.append(Request(f"q{i // args.windows_per_request:04d}", q_items[i : i + args.windows_per_request]))
    projection = project_cost(requests)
    tier_now = rate_tier(utc_now())
    projection["tier_now"] = tier_now
    projection["cached_windows"] = len(done_ids)
    projection["failed_final_skipped"] = len(failed_final)
    ledger = SpendLedger(cache_path.with_name(SPENT_FILENAME), started)
    remaining = round(args.cost_cap_cny - ledger.spent_before, 6)
    projection["spent_before_cny"] = ledger.spent_before
    projection["remaining_cap_cny"] = remaining
    print(f"[translate_zh] windows {len(target_windows)} (cached {len(done_ids)}, failed-final skipped {projection['failed_final_skipped']}), requests {projection['requests']}, "
          f"projected in/out tokens {projection['in_tokens']:,}/{projection['out_tokens']:,}, CNY {projection['cny']} (tier now: {tier_now}), "
          f"cap {args.cost_cap_cny} (spent before {ledger.spent_before}, remaining {remaining})", flush=True)
    report: dict[str, Any] = {
        "started_utc": started, "model": MODEL, "data_root": str(args.data_root), "out": str(out), "cache": str(cache_path),
        "spent_ledger": str(ledger.path),
        "transcript_source": args.transcript_source, "questions_source": args.questions_source,
        # Brief deviation, decided in EGOLIFE_ZH_PLAN §3: 'native' copies the official question_chinese / choice_*_chinese
        # fields (human translation) instead of translating questions with deepseek-flash; transcript and question
        # vocabularies then come from different translators. '--questions-source translate' removes that variable.
        # 与任务书的偏离，见 EGOLIFE_ZH_PLAN §3：'native' 复制官方 question_chinese / choice_*_chinese 字段（人工翻译）而不用
        # deepseek-flash 翻译题目；此时转写词汇与题目词汇来自不同译者。'--questions-source translate' 消除该变量。
        "questions_source_note": (
            "official question_chinese / choice_*_chinese fields (human translation) + glossary; transcripts are machine-translated — a translator mismatch the brief did not ask for; use --questions-source translate for one translator"
            if args.questions_source == "native" else "questions and options translated by deepseek-flash like the transcripts (the brief's protocol) + glossary"
        ),
        "windows_per_request": args.windows_per_request,
        "workers": args.workers, "cost_cap_cny": args.cost_cap_cny, "allow_tight_cap": bool(args.allow_tight_cap), "retry_failed": bool(args.retry_failed),
        "limit": args.limit, "projection": projection,
        "glossary": glossary, "glossary_added": glossary_added, "names_seen": names, "latin_names_kept": latin_kept, "dry_run": bool(args.dry_run),
        # Every name outside the glossary stays Latin by the brief's rule; whether one of them is a variant of a glossary
        # person (A1: 'Jack' next to 'Jake') is a user decision taken with --glossary-add, never silently.
        # 术语表之外的名字按任务书规则保持拉丁；其中某个是否为术语表人物的变体（A1：'Jack' 与 'Jake'）由用户经
        # --glossary-add 决定，绝不悄悄合并。
        "latin_names_note": "names outside the glossary are kept Latin (brief rule); a variant of a glossary person (e.g. Jack vs Jake in A1) is merged only through --glossary-add NAME=中文",
    }  # fmt: skip
    if args.dry_run:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        out.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[translate_zh] dry run: wrote {report_path}", flush=True)
        return 0
    projected_now = projection["cny"][tier_now]
    stopped: str | None = None
    if projected_now > remaining:
        stopped = f"projected {projected_now} CNY at {tier_now} exceeds the remaining cap {remaining} (cap {args.cost_cap_cny} - spent {ledger.spent_before}); no request sent"
    elif requests and projected_now > TIGHT_CAP_FRACTION * remaining and not args.allow_tight_cap:
        stopped = (
            f"projected {projected_now} CNY at {tier_now} is {projected_now / remaining:.0%} of the remaining cap {remaining} (> {TIGHT_CAP_FRACTION:.0%}): "
            f"an output-token under-estimate would stop the run mid-way; run off-peak or pass --allow-tight-cap; no request sent"
        )
    if stopped is not None:
        report["stopped"] = stopped
        out.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[translate_zh] " + stopped, flush=True)
        return 2
    meter = TranslateMeter()
    exit_code = 0
    if requests:
        from real_arms import DeepSeekClient, load_api_key

        load_api_key(ENV_PATH)
        factory = lambda: DeepSeekClient(meter)  # noqa: E731
        run_kwargs = dict(cap_cny=args.cost_cap_cny, workers=args.workers, progress=lambda s: print(s, flush=True), spent_before=ledger.spent_before, on_spend=ledger.record)
        try:
            first = translate_all(factory, requests, cache_path, meter, attempt=0, **run_kwargs)
            cache.update(first)
            # One retry alone for every window the batch round rejected (validation or truncation).
            # 批次轮拒绝（校验或截断）的每个窗口单独重试一次。
            retry_ids = {wid for wid, tr in cache.items() if not tr.validated and tr.attempt == 0}
            retry_windows = [w for w in target_windows if w.window_id in retry_ids]
            retry_requests = build_requests(retry_windows, transcripts, size=1, mode=args.transcript_source)
            retry_requests = [Request(f"x{i:04d}", r.items) for i, r in enumerate(retry_requests)]
            for item in (it for r in requests for it in r.items if it.window_id.startswith("q:") and it.window_id in retry_ids):
                retry_requests.append(Request(f"xq{len(retry_requests):04d}", [item]))
            if retry_requests:
                print(f"[translate_zh] retrying {len(retry_requests)} window(s) alone", flush=True)
                cache.update(translate_all(factory, retry_requests, cache_path, meter, attempt=1, **run_kwargs))
        except CostCapExceeded as exc:
            report["stopped"] = str(exc)
            print("[translate_zh] " + str(exc), flush=True)
            exit_code = 2
        ledger.record(meter.spent_cny(), calls=len(meter.calls), stopped=report.get("stopped"), finished=True)
        # The file holds every row that landed, including those of an interrupted round.
        # 文件里有落地的每一行，包括被中断那一轮的。
        cache = load_cache(cache_path)
    validated = {wid: tr for wid, tr in cache.items() if tr.validated}
    written = write_dataset(out, args.data_root, transcripts, windows, cache, questions_raw, mode=args.transcript_source, questions_source=args.questions_source, glossary=glossary)
    check = self_check(out, windows, questions)
    rates = [latin_letter_rate(tr.lines) for tr in validated.values() if not tr.window_id.startswith("q:")]
    failed = sorted(wid for wid, tr in cache.items() if not tr.validated and tr.attempt >= 1 and tr.reason != "cap")
    capped = sorted(wid for wid, tr in cache.items() if not tr.validated and tr.reason == "cap")
    report.update(
        {
            "coverage": {
                "windows_total": len(windows), "windows_targeted": len(target_windows), "windows_translated": sum(1 for w in target_windows if w.window_id in validated),
                "windows_written": written["windows"], "windows_native_fallback": written["windows_native_fallback"], "windows_no_chinese": written["windows_no_chinese"],
                "windows_failed": failed, "windows_capped": capped, "cues": written["cues"], "questions": f"{written['questions']}/{len(questions)}",
                "questions_translated": written["questions_translated"], "questions_native_fallback": written["questions_native_fallback"],
            },
            "glossary_used": written["glossary_used"],
            "latin_letter_rate": {
                "min": round(min(rates), 4) if rates else None, "mean": round(statistics.mean(rates), 4) if rates else None,
                "p95": round(sorted(rates)[int(0.95 * (len(rates) - 1))], 4) if rates else None,
                "over_0_5": sorted(wid for wid, tr in validated.items() if not wid.startswith("q:") and latin_letter_rate(tr.lines) > 0.5),
            },
            "variant_rates": variant_rates(target_windows, transcripts, cache, args.transcript_source, glossary),
            "ledger": meter.cost_summary(),
            "spent_cny": {"before": ledger.spent_before, "this_run": round(meter.spent_cny(), 6), "total": round(ledger.spent_before + meter.spent_cny(), 6), "cap": args.cost_cap_cny},
            "self_check": check,
            "wall_s": round(time.perf_counter() - t0, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
    )  # fmt: skip
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[translate_zh] coverage {report['coverage']['windows_translated']}/{len(target_windows)} windows, questions {report['coverage']['questions']}, "
          f"cost_cny={report['ledger']['cost_cny']['actual']} calls={report['ledger']['calls']} self_check={check['windows_ok'] and check['questions_ok'] and check['zh_windows_ok']} -> {report_path}", flush=True)
    if not args.limit and exit_code == 0 and not (check["windows_ok"] and check["questions_ok"] and check["zh_windows_ok"]):
        return 3
    return exit_code


__all__ = [
    "NAME_GLOSSARY", "OPTION_STOP", "PRICES_CNY_PER_M", "SPENT_FILENAME", "TIGHT_CAP_FRACTION", "TRANSLATE_SYSTEM", "CostCapExceeded", "Item",
    "Request", "SpendLedger", "TranslateMeter", "Translation",
    "apply_glossary", "build_requests", "collect_names", "cny_cost", "cues_to_translate", "latin_letter_rate", "load_cache",
    "main", "parse_glossary_add", "project_cost", "resume_skip_ids", "self_check", "translate_all", "translate_items", "validate_window", "variant_rates",
    "write_dataset",
]  # fmt: skip

if __name__ == "__main__":
    raise SystemExit(main())
