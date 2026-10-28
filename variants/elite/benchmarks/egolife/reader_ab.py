"""Reader-side A/B over one recorded lines run: ingest replayed with zero API calls, eleven answer-context variants captured
at every question's exact timeline step, then one deepseek-flash answer call per (variant, question).
在一次记录的行式 run 上做阅读端 A/B：零 API 调用回放摄入，在每道题的精确时间线步骤捕获十一种答题上下文变体，
再对每个（变体，题目）做一次 deepseek-flash 答题调用。
"""

# Purpose (EN): Hold the store IDENTICAL (same recorded extraction replies, same timeline) and vary only what the
#               reader sees — the 20 search rows as today, deduplicated / time-sorted / time-stamped, the compile()
#               card at four budgets, a query-independent session card memoized per timeline step in two prompt
#               orders, and two memories-first probes (card_prefix: the query card answered memories-first;
#               hybrid_prefix: the memoized session card followed by the rows_ts block, then the question) — so
#               accuracy, prompt tokens, prompt-cache hits and CNY cost are attributable to the reader alone.
#               rows contexts are the recorded run's (parity exact under fastembed 0.8.0 + onnxruntime 1.29.0); card_* /
#               session_* contexts are a RE-DERIVED baseline: the recorded run minted random atom ids and compile()
#               tie-breaks within-slot order on item id, so no card text can be byte-identical to that run. Atom ids are
#               deterministic (counter) by default so two replays agree with each other; --random-ids opts out.
# 目的（中）：把存储保持完全相同（同一批记录的抽取回复、同一条时间线），只改变阅读器看到的东西——今日的 20 条检索行、
#            去重 / 按时间排序 / 加时间戳的行、四种预算的 compile() 卡片、按时间线步骤记忆化的与题无关的会话卡片
#            （两种提示词顺序）、以及两个记忆在前的探针（card_prefix：查询卡片以记忆在前作答；hybrid_prefix：记忆化的
#            会话卡片后接 rows_ts 段、再题目）——使准确率、提示词 token、提示词缓存命中与人民币费用只归因于阅读端。
#            rows 上下文即记录 run 的（fastembed 0.8.0 + onnxruntime 1.29.0 下一致性精确）；card_* / session_* 上下文是
#            "重新导出的基线"：记录 run 用随机原子 id，而 compile() 以项目 id 打破槽内平局，故任何卡片文本都不可能与该
#            run 逐字节相同。原子 id 默认确定（计数器），两次回放互相一致；--random-ids 可退出。

# Input (EN): results/real/<run>/{extract_raw_elite.jsonl, batches.json, windows.jsonl, questions_elite_rows.jsonl,
#             questions_elite_card.jsonl, summary.json}; the dataset under data/; benchmarks/egolife/.env for the
#             answer pass (DEEPSEEK_API_KEY; loaded by replay_ab.load_api_key, never printed or written).
# 输入（中）：results/real/<run>/{extract_raw_elite.jsonl, batches.json, windows.jsonl, questions_elite_rows.jsonl,
#            questions_elite_card.jsonl, summary.json}；data/ 下的数据集；答题阶段读取 benchmarks/egolife/.env
#            （DEEPSEEK_API_KEY；由 replay_ab.load_api_key 加载，绝不打印或写出）。

# Output (EN): <out>/{summary.json, summary.md, per_question_<variant>.jsonl (eleven files), per_question_lines.jsonl,
#              calls.jsonl (empty under --no-answer), retain_lines.jsonl, entity_axis_lines.jsonl, parity.json with
#              --parity-check}; when the ingest replay had a gap (exit 4, no answer pass) only ingest_gaps.json, a minimal
#              summary.json (stopped=True) and the retain / entity_axis files. An earlier run's outputs in <out> are
#              removed first.
# 输出（中）：<out>/{summary.json, summary.md, per_question_<variant>.jsonl（十一个文件）, per_question_lines.jsonl,
#            calls.jsonl（--no-answer 时为空）, retain_lines.jsonl, entity_axis_lines.jsonl, --parity-check 时另有
#            parity.json}；摄入回放有缺口时（退出码 4，不答题）只有 ingest_gaps.json、最小的 summary.json（stopped=True）
#            与 retain / entity_axis 文件。<out> 中先前 run 的输出会先被删除。

# State effect (EN): Reads the run directory (never writes into it); builds one SQLite store under <out>/work (removed
#                    unless --keep-db); the answer pass makes at most --max-answer-requests API calls and stops when the
#                    projected or running cost exceeds --cost-cap-usd; --workers > 1 uses one DeepSeekClient per worker
#                    sharing one lock-guarded Meter.
# 状态影响（中）：只读 run 目录（绝不写入）；在 <out>/work 下建一个 SQLite 库（除非 --keep-db 否则删除）；答题阶段至多
#                --max-answer-requests 次 API 调用，预计或累计费用超过 --cost-cap-usd 即停止；--workers > 1 时每个
#                工作线程一个 DeepSeekClient，共用一个带锁的 Meter。

from __future__ import annotations

import argparse
import json
import queue
import shutil
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accounting import PRICES_USD_PER_M, ApiCall, EmbeddingWork, Meter, rate_tier, utc_now  # noqa: E402
from loader import DEFAULT_DATA_ROOT, QA_FILENAME, Question, load_questions, load_transcripts  # noqa: E402
from real_arms import (  # noqa: E402
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    BANK,
    BATCH_SIZE,
    AnswerResult,
    DeepSeekClient,
    load_api_key,
    parse_choice,
    plan_batches,
    rows_to_context,
)
from real_arms import build_prompt as _real_build_prompt  # noqa: E402
from replay_ab import ENV_PATH, load_recorded, resolve_leaves, run_arm  # noqa: E402
from timeline import Step, schedule  # noqa: E402
from windows import Window, build_windows  # noqa: E402

from hms_elite import HMS  # noqa: E402
from hms_elite.compiler import context as _context  # noqa: E402
from hms_elite.compiler.context import (  # noqa: E402
    dedup_rows,
    format_stamp,
    hybrid_text,
    normalize_text,  # noqa: F401 - re-exported for callers of the old harness name / 为旧基准名的调用者再导出
    row_times,
    sort_rows_by_time,
)
from hms_elite.memory.utils import estimate_tokens  # noqa: E402

DEFAULT_RUN = HERE / "results" / "real" / "elite_lines_full204"
DEFAULT_OUT = HERE / "results" / "reader_ab"
ARM = "lines"
VARIANTS = (
    "rows", "rows_td", "rows_ts", "card_1200", "card_800", "card_600", "card_400", "session_1200", "session_prefix",
    "card_prefix", "hybrid_prefix",
)  # fmt: skip
ROWS_VARIANTS = ("rows", "rows_td", "rows_ts")
# compile() budgets; card_1200 passes NO budget_tokens (the library default is 1200) so its CALL is the one of
# real_arms.EliteRealArm.card. The rendered text is NOT reproducible byte for byte against the recorded run: that run
# minted atom ids with secrets.token_hex, compile() orders every non-evidence slot by (time_anchor, item_id), and all
# atoms of one add() slice share time_anchor, so its within-slot line order was decided by ids this replay cannot
# recover. The harness mints counter ids by default (counter_id_factory; --random-ids restores the library's), so two
# replays agree with each other; every card_* / session_* context is a re-derived baseline on an equivalent store, and
# --parity-check verifies chars / batches / measurements, never the text.
# compile() 预算；card_1200 不传 budget_tokens（库默认 1200），使其"调用"与 real_arms.EliteRealArm.card 相同。渲染文本
# 无法与记录 run 逐字节复现：该 run 用 secrets.token_hex 铸造原子 id，compile() 对每个非证据槽按 (time_anchor, item_id)
# 排序，而同一 add() 切片的所有原子共享 time_anchor，其槽内行序由本次回放无法恢复的 id 决定。本脚本默认用计数器铸造 id
# （counter_id_factory；--random-ids 恢复库的随机 id），两次回放互相一致；每个 card_* / session_* 上下文都是等价库上
# 重新导出的基线，--parity-check 校验字符数 / 批次 / 度量，从不校验文本。
CARD_BUDGETS: dict[str, int | None] = {"card_1200": None, "card_800": 800, "card_600": 600, "card_400": 400}
# The baseline statement summary.json / summary.md carry (one place, so the method text and the tests quote one wording).
# summary.json / summary.md 所带的基线声明（集中一处，使方法文本与测试引用同一措辞）。
CARD_BASELINE_NOTE = (
    "card_* / session_* contexts are a re-derived baseline, not the recorded run's text: the recorded run minted random atom "
    "ids (secrets.token_hex) and compile() tie-breaks within-slot line order on item id, so card text cannot be byte-identical "
    "to the recorded run even when parity passes; this replay mints deterministic counter ids by default (--random-ids restores "
    "the library's), so two replays agree with each other. rows parity (batches / counts / chars / scores) is exact under "
    "fastembed 0.8.0 + onnxruntime 1.29.0, the recorded run's wheel set."
)
# Files every completed run rewrites; an earlier run's copies are removed before this run writes anything.
# 每次完整 run 都会重写的文件；本次写入任何内容之前先删除先前 run 的副本。
REWRITTEN_OUTPUTS = ("calls.jsonl", "summary.json", "summary.md")
CARD_DEFAULT_BUDGET = 1200
SESSION_VARIANTS = ("session_1200", "session_prefix")
SESSION_BUDGET = 1200
# card_prefix = card_1200's text (the same compile call and measurements; one compile per question, shared when both
# run) answered memories-first, isolating the ordering penalty on a query-specific card. hybrid_prefix = the memoized
# session card (session_1200's text and memo; one compile per step, shared with it) followed by a second block holding
# rows_ts's text under HYBRID_ROWS_HEADER, then the question: a stable cacheable prefix plus query-specific recall in
# the suffix. Records of SESSION_MEMO_VARIANTS carry the memo fields (`session`), so the summary's memo column and
# cache-hit split apply to all three.
# card_prefix = card_1200 的文本（同一 compile 调用与度量；每题一次 compile，两者同时运行时共享）以记忆在前作答，用于隔离
# 查询卡片上的顺序代价。hybrid_prefix = 记忆化的会话卡片（session_1200 的文本与记忆；每步一次 compile，与之共享）后接第二段、
# 以 HYBRID_ROWS_HEADER 为题的 rows_ts 文本，最后是题目：稳定可缓存的前缀加后缀中的查询特定召回。SESSION_MEMO_VARIANTS 的
# 记录携带记忆字段（`session`），故汇总的记忆列与缓存命中拆分对三者都适用。
MEMORIES_FIRST_VARIANTS = ("session_prefix", "card_prefix", "hybrid_prefix")
SESSION_MEMO_VARIANTS = (*SESSION_VARIANTS, "hybrid_prefix")
# The library's ROWS_HEADER (hms_elite.compiler.context) — HMS.context() renders the same string; the row builders
# below (normalize_text / row_times / dedup_rows / sort_rows_by_time / format_stamp / hybrid_context) ARE the
# library's functions since 2026-09-18, so the harness and HMS.context() agree byte for byte.
# 库的 ROWS_HEADER（hms_elite.compiler.context）——HMS.context() 渲染同一字符串；下方的行构造器（normalize_text /
# row_times / dedup_rows / sort_rows_by_time / format_stamp / hybrid_context）自 2026-09-18 起就是库的函数，
# 因此基准脚本与 HMS.context() 逐字节一致。
HYBRID_ROWS_HEADER = _context.ROWS_HEADER
PROMPT_ORDER = {v: ("memories_first" if v in MEMORIES_FIRST_VARIANTS else "question_first") for v in VARIANTS}
ANSWER_LABEL = {v: f"reader_{v}" for v in VARIANTS}
# DeepSeek list price in CNY per 1M tokens (deepseek-flash, 2026-09): off-peak cache_hit 0.02 / cache_miss 1 / output 4;
# peak 0.04 / 2 / 8. Both figures are computed from the measured token counts regardless of the tier the call ran at.
# DeepSeek 人民币定价（每 1M token，deepseek-flash，2026-09）：非高峰 缓存命中 0.02 / 未命中 1 / 输出 4；高峰 0.04 / 2 / 8。
# 两个数字都按实测 token 数计算，与调用实际所处档位无关。
PRICES_CNY_PER_M: dict[str, dict[str, float]] = {
    "offpeak": {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0},
    "peak": {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0},
}
MAX_ANSWER_REQUESTS = 2000
COST_CAP_USD = 0.8
STORED_COUNT_ORACLE = 12963
N_QUESTIONS = 204
# Pre-flight projection chars-per-token for THIS harness: replay_ab's 3.0 under-estimates card prompts. Measured over
# the recorded card run (questions_elite_card.jsonl, prompt_tokens over the exact question_first prompt): min 2.504,
# p05 2.605, median 2.728 chars/token; the rows run's minimum is 3.763. 2.4 sits below the measured minimum, so no
# recorded card prompt is under-projected; rows_ts (digit-heavy '[YYYY-MM-DD HH:MM] ' prefixes) has no measurement and
# is covered by the same margin only by assumption. The running-ledger check still bounds overshoot to one chunk.
# 本脚本的预检字符/词元比：replay_ab 的 3.0 会低估卡片提示词。按记录的卡片 run（questions_elite_card.jsonl，对精确的
# question_first 提示词的 prompt_tokens）实测：最小 2.504、p05 2.605、中位 2.728 字符/词元；行式 run 最小 3.763。取 2.4
# 低于实测最小值，故没有记录的卡片提示词会被低估；rows_ts（含数字的 '[YYYY-MM-DD HH:MM] ' 前缀）无实测，只是假定同样的
# 余量覆盖它。累计台账检查仍把超支限制在一个分块内。
PROJECTION_CHARS_PER_TOKEN = 2.4
# Packages whose versions decide embedding numerics (and so store identity); recorded in summary.json config.
# 决定嵌入数值（从而决定库是否相同）的包；版本记入 summary.json 的 config。
VERSIONED_PACKAGES = ("fastembed", "onnxruntime", "numpy", "tokenizers", "hms-elite")
# top_scores parity tolerance: the recorded scores are 4-dp roundings of fastembed cosine scores, which differ by
# ~1e-4 between fastembed / onnxruntime builds; batches, counts and chars stay strict.
# top_scores 一致性容差：记录的分数是 fastembed 余弦分数的 4 位小数取整，不同 fastembed / onnxruntime 构建之间相差约
# 1e-4；批次、计数与字符数仍严格比较。
SCORE_TOL = 5e-4


# ---------------------------------------------------------------------------
# Row-context builders
# 行式上下文构造
# ---------------------------------------------------------------------------
def project_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The recall projection of real_arms.EliteRealArm.recall / replay_ab.run_arm: {batch_id, text, score, memory_id}.
    real_arms.EliteRealArm.recall / replay_ab.run_arm 的召回投影：{batch_id, text, score, memory_id}。

    Input / 输入：
        raw: hms.search rows (§4.1 public rows) in rank order.
        raw：hms.search 的行（§4.1 公开行），按排名。

    Output / 输出：
        Projected rows in the same order.
        同序的投影行。

    State effect / 状态影响：
        None.
        无。
    """
    return [
        {
            "batch_id": ((r.get("metadata") or {}).get("batch_id")), "text": r.get("memory", ""),
            "score": float(r.get("score", 0.0)), "memory_id": r.get("id"),
        }
        for r in raw
    ]  # fmt: skip


def rows_context(raw: list[dict[str, Any]]) -> str:
    """Baseline: real_arms.rows_to_context over the projected rows (the text the streaming run answered on).
    基线：投影行上的 real_arms.rows_to_context（流式 run 答题所用的文本）。

    Input / 输入：
        raw: hms.search rows in rank order.
        raw：按排名的 hms.search 行。

    Output / 输出：
        '- text' lines.
        '- text' 行。

    State effect / 状态影响：
        None.
        无。
    """
    return rows_to_context(project_rows(raw))


def rows_td_context(raw: list[dict[str, Any]]) -> str:
    """rows_td: the search rows deduplicated (dedup_rows) and time-sorted (sort_rows_by_time), rendered by
    real_arms.rows_to_context ('- text' lines, no timestamps).
    rows_td：检索行去重（dedup_rows）并按时间排序（sort_rows_by_time），由 real_arms.rows_to_context 渲染
    （'- text' 行，无时间戳）。

    Input / 输入：
        raw: hms.search rows in rank order.
        raw：按排名的 hms.search 行。

    Output / 输出：
        The context text.
        上下文文本。

    State effect / 状态影响：
        None.
        无。
    """
    return rows_to_context(project_rows(sort_rows_by_time(dedup_rows(raw))))


def rows_ts_context(raw: list[dict[str, Any]]) -> str:
    """rows_ts: rows_td plus a '[YYYY-MM-DD HH:MM] ' prefix from hms.time_anchor, which is the slice / batch timestamp
    the atom was ingested with (windows[0].start_iso of its add() slice: batch-start granularity, several minutes
    coarse, not the window's own time); the bracket is omitted when the anchor is unknown.
    rows_ts：rows_td 再加 hms.time_anchor 的 '[YYYY-MM-DD HH:MM] ' 前缀，该锚定是原子摄入时所带的切片 / 批次时间戳
    （其 add() 切片的 windows[0].start_iso：批次起点粒度、粗至数分钟，不是窗口自身时间）；锚定未知时省略方括号。

    Input / 输入：
        raw: hms.search rows in rank order.
        raw：按排名的 hms.search 行。

    Output / 输出：
        The context text.
        上下文文本。

    State effect / 状态影响：
        None.
        无。
    """
    return _context.rows_text(raw, timestamps=True)


# ---------------------------------------------------------------------------
# Card builders and prompts
# 卡片构造与提示词
# ---------------------------------------------------------------------------
def compile_card(hms: HMS, query: str, as_of: str, budget_tokens: int | None) -> dict[str, Any]:
    """real_arms.EliteRealArm.card with an optional budget: hms.compile(query, filters={'user_id': BANK}, as_of=as_of
    [, budget_tokens=...]).to_text() plus the items' batch ids (via hms.get) and measurements. With budget_tokens None
    the call carries no budget argument, exactly as the streaming run's card.
    带可选预算的 real_arms.EliteRealArm.card：hms.compile(query, filters={'user_id': BANK}, as_of=as_of
    [, budget_tokens=...]).to_text()，附各项目的批次 id（经 hms.get）与度量。budget_tokens 为 None 时调用不带预算参数，
    与流式 run 的卡片完全一致。

    Input / 输入：
        hms: The facade at the step's state.
        hms：处于该步状态的门面。

        query: compile query text ('' allowed: axis-only listing).
        query：compile 查询文本（允许 ''：仅轴列表）。

        as_of: Validity instant.
        as_of：有效性时刻。

        budget_tokens: None (library default) or the budget.
        budget_tokens：None（库默认）或预算。

    Output / 输出：
        {'text', 'chars', 'used_tokens', 'items', 'mode', 'compile_ms', 'batch_ids', 'budget_tokens'}.
        {'text', 'chars', 'used_tokens', 'items', 'mode', 'compile_ms', 'batch_ids', 'budget_tokens'}。

    State effect / 状态影响：
        Local only (one query embedding when the query is non-empty).
        仅本地（查询非空时一次查询嵌入）。
    """
    t0 = time.perf_counter()
    if budget_tokens is None:
        ir = hms.compile(query, filters={"user_id": BANK}, as_of=as_of)
    else:
        ir = hms.compile(query, filters={"user_id": BANK}, as_of=as_of, budget_tokens=int(budget_tokens))
    text = ir.to_text()
    ms = (time.perf_counter() - t0) * 1000.0
    items = ir.items()
    batch_ids: list[str] = []
    for item in items:
        mid = getattr(item, "memory_id", None)
        if not mid:
            continue
        row = hms.get(mid)
        bid = ((row or {}).get("metadata") or {}).get("batch_id")
        if bid and bid not in batch_ids:
            batch_ids.append(bid)
    return {
        "text": text,
        "chars": len(text),
        "used_tokens": int(getattr(ir, "used_tokens", 0)),
        "items": len(items),
        "mode": str(getattr(getattr(ir, "mode", ""), "value", getattr(ir, "mode", ""))),
        "compile_ms": round(ms, 3),
        "batch_ids": batch_ids,
        "budget_tokens": int(budget_tokens or getattr(ir, "budget_tokens", CARD_DEFAULT_BUDGET)),
    }


# build_prompt(question, context, order) = real_arms.build_prompt: one implementation of the two prompt orders and
# the '(none)' rule, shared with run_real's answer().
# build_prompt(question, context, order) = real_arms.build_prompt：两种提示词顺序与 '(none)' 规则的唯一实现，
# 与 run_real 的 answer() 共用。
build_prompt = _real_build_prompt


# hybrid_context(session_text, rows_text) = hms_elite.compiler.context.hybrid_text with the library header.
# hybrid_context(session_text, rows_text) = 带库默认标题的 hms_elite.compiler.context.hybrid_text。
hybrid_context = hybrid_text


def ask(client: DeepSeekClient, question: Question, prompt: str, *, arm: str) -> AnswerResult:
    """real_arms.answer with the prompt supplied by the caller (same call: temperature 0, max_tokens 8, one user
    message, labels arm/stage='answer'/question_id, letter parsed with their parse_choice, invalid output marked
    'empty').
    调用方给定提示词的 real_arms.answer（同样的调用：temperature 0、max_tokens 8、一条 user 消息、标签
    arm/stage='answer'/question_id、用对方的 parse_choice 解析字母、无效输出标为 'empty'）。

    Input / 输入：
        client: The accounting client.
        client：记账客户端。

        question: The question (its id labels the ledger row).
        question：题目（其 id 作为台账标签）。

        prompt: The single user message.
        prompt：唯一的 user 消息。

        arm: Ledger label.
        arm：台账标签。

    Output / 输出：
        AnswerResult.
        AnswerResult。

    State effect / 状态影响：
        One to three ApiCalls appended to the meter.
        向台账追加 1 到 3 条 ApiCall。
    """
    t_wall = time.perf_counter()
    with client.label(arm=arm, stage="answer", question_id=question.id, batch_id=None):
        try:
            response = client.create(
                messages=[{"role": "user", "content": prompt}], temperature=ANSWER_TEMPERATURE, max_tokens=ANSWER_MAX_TOKENS
            )
        except Exception:  # noqa: BLE001 - recorded in the ledger by create()
            calls = list(client.last_calls)
            return AnswerResult(
                letter=None, raw=None, status=calls[-1].status if calls else "http_error", prompt_chars=len(prompt), calls=calls,
                model_ms=sum(c.latency_ms for c in calls), local_ms=0.0, wall_ms=(time.perf_counter() - t_wall) * 1000.0,
            )  # fmt: skip
    t_local = time.perf_counter()
    content = response.choices[0].message.content if response.choices else None
    raw = content if isinstance(content, str) else None
    letter = parse_choice(raw) if raw else None
    if letter is None:
        client.mark_last(status="empty", error="invalid_output" if (raw and raw.strip()) else "empty_content")
    calls = list(client.last_calls)
    return AnswerResult(
        letter=letter, raw=raw, status=calls[-1].status if calls else "ok", prompt_chars=len(prompt), calls=calls,
        model_ms=sum(c.latency_ms for c in calls), local_ms=(time.perf_counter() - t_local) * 1000.0,
        wall_ms=(time.perf_counter() - t_wall) * 1000.0,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# The per-question hook: eleven contexts at the step's exact store state
# 逐题钩子：在该步精确的库状态下捕获十一种上下文
# ---------------------------------------------------------------------------
def store_changes(rec: dict[str, Any]) -> int:
    """ADD + UPDATE events of a run_arm record's retain (rec['retain_events']); a record without the histogram (older
    replay_ab records) falls back to retain_added, which also counts NONE events.
    run_arm 记录的 retain 的 ADD + UPDATE 事件数（rec['retain_events']）；没有直方图的记录（旧版 replay_ab 记录）退回
    retain_added，它也把 NONE 事件计入。

    Input / 输入：
        rec: run_arm's record.
        rec：run_arm 的记录。

    Output / 输出：
        The count.
        计数。

    State effect / 状态影响：
        None.
        无。
    """
    events = rec.get("retain_events")
    if isinstance(events, dict):
        return int(events.get("ADD", 0)) + int(events.get("UPDATE", 0))
    return int(rec.get("retain_added") or 0)


@dataclass
class ReaderContexts:
    """run_arm context hook: builds the requested variants per question, memoizes the session card per timeline step
    (recomputed only after a step whose retain changed the store: >= 1 ADD or UPDATE event; the query is the last
    window of the last leaf that did so), and keeps the baseline figures for the parity check.
    run_arm 的上下文钩子：逐题构造所请求的变体，按时间线步骤记忆化会话卡片（只在 retain 改变了库的步骤之后重算：
    >= 1 个 ADD 或 UPDATE 事件；查询取最后一个改变库的叶子的最后一个窗口），并保留基线数字供一致性检查。
    """

    variants: tuple[str, ...] = VARIANTS
    by_question: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    baseline: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_key: str | None = None
    session_card: dict[str, Any] | None = None
    positions: dict[str, int] = field(default_factory=dict)
    session_compiles: int = 0
    session_reuses: int = 0
    session_without_ingest: int = 0
    session_ingest_skipped: int = 0
    session_query_fallback: int = 0
    rows_baseline_mismatch: int = 0

    def __call__(self, hms: HMS, step: Step, query_iso: str, raw_rows: list[dict[str, Any]], rec: dict[str, Any]) -> None:
        """Capture the variants for one question (the ContextHook contract of replay_ab.run_arm).
        为一道题捕获各变体（replay_ab.run_arm 的 ContextHook 契约）。

        Input / 输入：
            hms: The facade at the step's state.
            hms：处于该步状态的门面。

            step: The timeline step (question, new_windows).
            step：时间线步骤（题目、new_windows）。

            query_iso: The question's instant (search reference_date / compile as_of).
            query_iso：题目时刻（检索 reference_date / compile as_of）。

            raw_rows: The baseline hms.search rows (top_k 20, threshold 0.1).
            raw_rows：基线 hms.search 行（top_k 20、threshold 0.1）。

            rec: run_arm's record (read: gold_batches, hit_batch, context, returned_*; not modified).
            rec：run_arm 的记录（读取 gold_batches、hit_batch、context、returned_*；不修改）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Fills by_question[q.id] and baseline[q.id]; updates the session memo and counters.
            填充 by_question[q.id] 与 baseline[q.id]；更新会话记忆与计数。
        """
        q = step.question
        gold = set(rec.get("gold_batches") or [])
        out: dict[str, dict[str, Any]] = {}
        self.baseline[q.id] = {
            "returned_batches": list(rec.get("returned_batches") or []), "top_scores": list(rec.get("top_scores") or []),
            "returned_count": rec.get("returned_count"), "context_chars": rec.get("context_chars"),
        }  # fmt: skip
        self.positions[q.id] = int(rec.get("position") or 0)
        want_hybrid = "hybrid_prefix" in self.variants
        texts: dict[str, str] = {}
        kept_batches: list[str] = []
        top20_batches: list[str] = []
        rows_info: dict[str, Any] = {}
        rows_hit = top20_hit = False
        # The rows facts also feed hybrid_prefix's second block, so they are built when it is requested on its own.
        # 行式事实也供 hybrid_prefix 的第二段使用，故只请求它时同样构建。
        if want_hybrid or any(v in self.variants for v in ROWS_VARIANTS):
            kept = dedup_rows(raw_rows)
            texts = {"rows": rows_context(raw_rows), "rows_td": rows_td_context(raw_rows), "rows_ts": rows_ts_context(raw_rows)}
            if texts["rows"] != rec.get("context"):
                self.rows_baseline_mismatch += 1
            top20_batches = list(rec.get("returned_batches") or [])
            top20_hit = bool(rec.get("hit_batch"))
            # rows_td / rows_ts: the retrieval facts describe the rows that are IN the context (the kept rows), so a
            # dedup that drops every row of a gold batch lowers hit; the shared hit@20 travels under hit_batch_top20.
            # rows_td / rows_ts：检索事实描述真正进入上下文的行（保留行），去重若丢掉某 gold 批次的全部行则命中下降；
            # 共享的 hit@20 放在 hit_batch_top20 下。
            for r in project_rows(kept):
                if r["batch_id"] and r["batch_id"] not in kept_batches:
                    kept_batches.append(r["batch_id"])
            rows_hit = any(b in gold for b in kept_batches)
            # Stamp granularity of the kept rows: distinct '[YYYY-MM-DD HH:MM]' stamps vs distinct batches. time_anchor
            # is the add() slice's timestamp, so the two counts are expected to be close (one stamp per slice).
            # 保留行的时间戳粒度：不同 '[YYYY-MM-DD HH:MM]' 戳数 vs 不同批次数。time_anchor 是 add() 切片的时间戳，
            # 两个计数预期接近（每切片一个戳）。
            stamps = {format_stamp(row_times(r)[0]) for r in kept}
            stamps.discard(None)
            rows_info = {"rows_in": len(raw_rows), "rows_kept": len(kept), "rows_dropped": len(raw_rows) - len(kept),
                         "stamps_distinct": len(stamps), "batches_distinct": len(kept_batches),
                         "stamp_source": "hms.time_anchor = windows[0].start_iso of the atom's add() slice (batch-start granularity)"}  # fmt: skip
            for v in ROWS_VARIANTS:
                if v not in self.variants:
                    continue
                if v == "rows":
                    out[v] = self._ctx(texts[v], returned_count=len(raw_rows), returned_batches=top20_batches, hit_batch=top20_hit,
                                       rows=dict(rows_info))  # fmt: skip
                else:
                    out[v] = self._ctx(texts[v], returned_count=len(kept), returned_batches=kept_batches,
                                       hit_batch=rows_hit, hit_batch_top20=top20_hit,
                                       returned_batches_top20=top20_batches, rows=dict(rows_info))  # fmt: skip
        for v, budget in CARD_BUDGETS.items():
            if v in self.variants:
                card = compile_card(hms, q.question, query_iso, budget)
                out[v] = self._card_ctx(card, gold)
        if "card_prefix" in self.variants:
            # The same compile call as card_1200 (no budget argument); when card_1200 also ran its record is copied, so
            # the two carry one text and one measurement set and compile runs once per question.
            # 与 card_1200 相同的 compile 调用（不带预算参数）；card_1200 也运行时直接复制其记录，两者共用一份文本与度量，
            # 每题只 compile 一次。
            card_ctx = out.get("card_1200") or self._card_ctx(compile_card(hms, q.question, query_iso, CARD_BUDGETS["card_1200"]), gold)
            out["card_prefix"] = dict(card_ctx)
        session: dict[str, Any] | None = None
        if want_hybrid or any(v in self.variants for v in SESSION_VARIANTS):
            session = self._session(hms, step, gold, changed=store_changes(rec), last_window=rec.get("retain_last_stored_window"))
            for v in SESSION_VARIANTS:
                if v in self.variants:
                    out[v] = dict(session)
        if want_hybrid and session is not None:
            out["hybrid_prefix"] = self._hybrid_ctx(session, texts["rows_ts"], kept_batches=kept_batches, rows_hit=rows_hit,
                                                    rows_kept=int(rows_info.get("rows_kept", 0)), top20_hit=top20_hit,
                                                    top20_batches=top20_batches, rows_info=rows_info)  # fmt: skip
        self.by_question[q.id] = out

    def _session(self, hms: HMS, step: Step, gold: set[str], *, changed: int, last_window: str | None) -> dict[str, Any]:
        """The memoized session card: recomputed when the step scheduled windows AND its retain changed the store
        (changed = ADD + UPDATE events >= 1; NONE events and failed / missing / empty leaves do not count). The query
        window is `last_window` — the last window of the last leaf that produced an ADD or UPDATE (run_arm's
        retain_last_stored_window) — so a step whose final leaf failed is keyed on a window that was actually stored;
        when it is not among the step's windows the step's last window is used and session_query_fallback counts it.
        query = that window's text, as_of = its start instant, budget SESSION_BUDGET. A scheduled step with changed == 0
        leaves the store unchanged, reuses the previous card and is counted under session_ingest_skipped; otherwise
        the card is reused verbatim.
        记忆化的会话卡片：该步安排了窗口且其 retain 改变了库（changed = ADD + UPDATE 事件数 >= 1；NONE 事件与失败 /
        缺失 / 空叶子不计）时重算。查询窗口取 `last_window`——最后一个产生 ADD 或 UPDATE 的叶子的最后一个窗口
        （run_arm 的 retain_last_stored_window）——因此末叶失败的步骤仍以真正入库的窗口为键；它不在该步窗口中时退回该步
        最后一个窗口并计入 session_query_fallback。query = 该窗口文本，as_of = 其起点时刻，预算 SESSION_BUDGET。安排了
        窗口但 changed == 0 的步骤库未变化，复用上一张卡片并计入 session_ingest_skipped；其余情况原样复用。

        Input / 输入：
            hms: The facade at the step's state.
            hms：处于该步状态的门面。

            step: The timeline step.
            step：时间线步骤。

            gold: The question's gold batch ids.
            gold：本题的 gold 批次 id。

            changed: ADD + UPDATE events of the step's retain (store_changes(rec)).
            changed：该步 retain 的 ADD + UPDATE 事件数（store_changes(rec)）。

            last_window: window_id of the last window stored at this step, or None.
            last_window：本步最后入库窗口的 window_id，或 None。

        Output / 输出：
            The context record (a copy when reused).
            上下文记录（复用时为副本）。

        State effect / 状态影响：
            Updates the memo and counters.
            更新记忆与计数。
        """
        if step.new_windows and changed <= 0:
            self.session_ingest_skipped += 1
        if step.new_windows and changed > 0:
            by_id = {w.window_id: w for w in step.new_windows}
            last: Window | None = by_id.get(last_window) if last_window else None
            if last is None:
                self.session_query_fallback += 1
                last = step.new_windows[-1]
            card = compile_card(hms, last.text, last.start_iso, SESSION_BUDGET)
            self.session_key = last.window_id
            self.session_card = self._card_ctx(card, gold)
            self.session_card["session"] = {"key": last.window_id, "query_window": last.window_id, "as_of": last.start_iso, "reused": False,
                                            "store_changes": int(changed)}  # fmt: skip
            self.session_compiles += 1
            return self.session_card
        if self.session_card is None:
            self.session_without_ingest += 1
            ctx = self._ctx("", returned_count=0, returned_batches=[], hit_batch=False)
            ctx["session"] = {"key": None, "query_window": None, "as_of": None, "reused": False}
            return ctx
        self.session_reuses += 1
        ctx = dict(self.session_card)
        # hit is re-judged against THIS question's gold batches; the text is the memoized one.
        # 命中按本题的 gold 批次重新判定；文本是记忆化的那份。
        ctx["hit_batch"] = any(b in gold for b in ctx["returned_batches"])
        ctx["session"] = dict(self.session_card["session"], reused=True)
        return ctx

    @staticmethod
    def _ctx(text: str, *, returned_count: int, returned_batches: list[str], hit_batch: bool, **extra: Any) -> dict[str, Any]:
        """One context record: text, chars, estimated tokens (hms_elite.memory.utils.estimate_tokens), retrieval facts.
        一条上下文记录：文本、字符数、估算 token（hms_elite.memory.utils.estimate_tokens）、检索事实。
        """
        return {
            "context": text, "context_chars": len(text), "est_tokens": estimate_tokens(text) if text else 0,
            "returned_count": returned_count, "returned_batches": returned_batches, "hit_batch": hit_batch, **extra,
        }  # fmt: skip

    def _card_ctx(self, card: dict[str, Any], gold: set[str]) -> dict[str, Any]:
        """Context record of a compiled card (hit = any item batch in gold, as run_real judges the card).
        编译卡片的上下文记录（命中 = 任一项目批次在 gold 中，与 run_real 对卡片的判定相同）。
        """
        return self._ctx(
            card["text"], returned_count=card["items"], returned_batches=list(card["batch_ids"]),
            hit_batch=any(b in gold for b in card["batch_ids"]),
            card={k: card[k] for k in ("chars", "used_tokens", "items", "mode", "compile_ms", "budget_tokens")},
        )  # fmt: skip

    def _hybrid_ctx(self, session: dict[str, Any], rows_text: str, *, kept_batches: list[str], rows_hit: bool, rows_kept: int,
                    top20_hit: bool, top20_batches: list[str], rows_info: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
        """Context record of hybrid_prefix: text = hybrid_context(session card, rows_ts); returned_batches = the session
        card's item batches then the kept rows' batches (deduplicated), returned_count = session items + kept rows;
        hit = session-card hit OR kept-rows hit (each re-judged on this question's gold). `session` and `card` are the
        session card's memo and measurements (session_1200's fields; `card` absent before the first compile), `rows`
        the rows facts, and `hybrid` the two parts' chars, batches and component hits.
        hybrid_prefix 的上下文记录：文本 = hybrid_context(会话卡片, rows_ts)；returned_batches = 会话卡片项目批次接保留行
        批次（去重），returned_count = 会话项目数 + 保留行数；命中 = 会话卡片命中 OR 保留行命中（都按本题 gold 判定）。
        `session` 与 `card` 是会话卡片的记忆与度量（session_1200 的字段；首次 compile 前无 `card`），`rows` 是行式事实，
        `hybrid` 是两部分的字符数、批次与各自命中。
        """
        session_batches = list(session["returned_batches"])
        batches = session_batches + [b for b in kept_batches if b not in session_batches]
        session_hit = bool(session["hit_batch"])
        extra: dict[str, Any] = {"session": dict(session["session"]), "rows": dict(rows_info)}
        if "card" in session:
            extra["card"] = dict(session["card"])
        return self._ctx(
            hybrid_context(session["context"], rows_text), returned_count=int(session["returned_count"]) + rows_kept,
            returned_batches=batches, hit_batch=session_hit or rows_hit,
            hybrid={"session_chars": len(session["context"]), "rows_chars": len(rows_text), "session_hit": session_hit,
                    "rows_hit": rows_hit, "rows_hit_top20": top20_hit, "session_returned_count": int(session["returned_count"]),
                    "rows_kept": rows_kept, "session_batches": session_batches, "rows_batches": list(kept_batches),
                    "rows_batches_top20": list(top20_batches), "rows_header": HYBRID_ROWS_HEADER},
            **extra,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# Answer pass (sequential or one client per worker over a locked meter)
# 答题阶段（顺序，或每工作线程一个客户端共用带锁台账）
# ---------------------------------------------------------------------------
class LockedMeter(Meter):
    """accounting.Meter whose mutating methods run under one lock, so several DeepSeekClients (one per worker thread)
    can share a single ledger.
    变更方法都在同一把锁下运行的 accounting.Meter，使多个 DeepSeekClient（每工作线程一个）能共用一份台账。
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

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

        Output / 输出：
            True exactly once per arm.
            每个臂恰好一次为 True。

        State effect / 状态影响：
            Adds arm to the seen set.
            把臂加入已见集合。
        """
        with self._lock:
            return super().next_is_cold(arm)

    def add(self, call: ApiCall) -> ApiCall:
        """Meter.add under the lock.
        加锁的 Meter.add。

        Input / 输入：
            call: The record.
            call：记录。

        Output / 输出：
            The same record.
            同一条记录。

        State effect / 状态影响：
            Appends to calls.
            追加到 calls。
        """
        with self._lock:
            return super().add(call)

    def add_embedding(self, arm: str, stage: str, texts: int, local_ms: float) -> EmbeddingWork:
        """Meter.add_embedding under the lock.
        加锁的 Meter.add_embedding。

        Input / 输入：
            arm, stage, texts, local_ms: As Meter.add_embedding.
            arm、stage、texts、local_ms：同 Meter.add_embedding。

        Output / 输出：
            The record.
            该记录。

        State effect / 状态影响：
            Appends to embeddings.
            追加到 embeddings。
        """
        with self._lock:
            return super().add_embedding(arm, stage, texts, local_ms)

    def spent_usd(self) -> float:
        """Sum of priced attempts under the lock.
        加锁求已计价尝试之和。

        Input / 输入：
            none.
            无。

        Output / 输出：
            USD.
            美元。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return sum(c.cost_usd or 0.0 for c in self.calls)


def project_cost(plan: list[tuple[str, dict[str, Any]]], questions: dict[str, Question]) -> float:
    """Pre-flight USD projection at the current tier: prompt chars / PROJECTION_CHARS_PER_TOKEN tokens all cache-miss
    plus ANSWER_MAX_TOKENS output tokens per request (replay_ab.project_cost's rule on the variant prompts).
    按当前档位预估美元费用：提示词 chars / PROJECTION_CHARS_PER_TOKEN 个 token 全按缓存未命中，加每次请求
    ANSWER_MAX_TOKENS 个输出 token（replay_ab.project_cost 的规则用于变体提示词）。

    Input / 输入：
        plan: (variant, record) pairs.
        plan：（变体，记录）对。

        questions: id → Question.
        questions：id → Question。

    Output / 输出：
        USD.
        美元。

    State effect / 状态影响：
        None.
        无。
    """
    price = PRICES_USD_PER_M[rate_tier(utc_now())]
    total = 0.0
    for variant, rec in plan:
        prompt = build_prompt(questions[rec["id"]], rec["context"], PROMPT_ORDER[variant])
        tokens = len(prompt) / PROJECTION_CHARS_PER_TOKEN
        total += (tokens * price["cache_miss"] + ANSWER_MAX_TOKENS * price["output"]) / 1_000_000.0
    return total


def apply_answer(rec: dict[str, Any], res: AnswerResult) -> None:
    """Write an AnswerResult into a variant record (the replay_ab.answer_pass field set plus latency / tier).
    把 AnswerResult 写入变体记录（replay_ab.answer_pass 的字段集加延迟 / 档位）。

    Input / 输入：
        rec: The variant record.
        rec：变体记录。

        res: The answer.
        res：答题结果。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Mutates rec.
        修改 rec。
    """
    last = res.calls[-1] if res.calls else None
    rec.update(
        {
            "pred": res.letter, "raw_output": res.raw, "correct": res.letter == rec["gold"], "answer_status": res.status,
            "answer_attempts": len(res.calls), "prompt_chars": res.prompt_chars,
            "answer_usage": {
                "prompt_tokens": last.prompt_tokens if last else None, "completion_tokens": last.completion_tokens if last else None,
                "cache_hit_tokens": last.cache_hit_tokens if last else None, "cache_miss_tokens": last.cache_miss_tokens if last else None,
                "rate_tier": last.rate_tier if last else None,
                "cost_usd_all_attempts": round(sum(c.cost_usd or 0.0 for c in res.calls), 8),
                "unpriced_attempts": sum(1 for c in res.calls if c.cost_usd is None),
            },
            "latency_ms": {"model": round(res.model_ms, 3), "wall": round(res.wall_ms, 3)},
            "tier": last.rate_tier if last else None,
        }
    )  # fmt: skip


def build_plan(records: dict[str, list[dict[str, Any]]], max_requests: int) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """The answer plan, variant-major in timeline order, truncated PER VARIANT so every variant answers the same
    leading question subset: each variant keeps its first max_requests // len(records) records (an empty plan when
    max_requests < len(records)); the remainder max_requests % len(records) is left unused rather than handed to the
    first variants, so accuracies and CNY per question stay comparable across variants.
    答题计划，按变体优先、时间线顺序，按"每变体"截断使各变体回答同一前缀题目子集：每个变体保留其前
    max_requests // len(records) 条记录（max_requests < len(records) 时计划为空）；余数 max_requests % len(records)
    弃置而不是分给靠前的变体，使各变体的准确率与每题人民币可比。

    Input / 输入：
        records: variant → records in timeline order.
        records：变体 → 按时间线顺序的记录。

        max_requests: Hard cap on requests.
        max_requests：请求数硬上限。

    Output / 输出：
        (plan, per_variant) — plan = (variant, record) pairs; per_variant = variant → records planned.
        (plan, per_variant)——plan = （变体，记录）对；per_variant = 变体 → 计划的记录数。

    State effect / 状态影响：
        None.
        无。
    """
    if not records or max_requests <= 0:
        return [], {v: 0 for v in records}
    quota = max_requests // len(records)
    per_variant = {v: min(quota, len(recs)) for v, recs in records.items()}
    plan = [(v, rec) for v, recs in records.items() for rec in recs[: per_variant[v]]]
    return plan, per_variant


def answer_pass(
    records: dict[str, list[dict[str, Any]]], questions: dict[str, Question], meter: LockedMeter, *,
    max_requests: int, cost_cap: float, workers: int = 1, flush: Callable[[int], None] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """One deepseek-flash call per (variant, question), variant-major in timeline order (so questions sharing a
    session card are consecutive), the plan truncated per variant by build_plan; stops before the first call when
    the projection exceeds the cap and mid-way when the running ledger does. workers > 1: a ThreadPoolExecutor with
    one DeepSeekClient per worker over the locked meter, processing the plan in chunks of `workers` (call order
    inside a chunk is not guaranteed); an exception or interrupt that surfaces while a chunk is in flight waits for
    the chunk's other futures and writes their answers into the records before re-raising, so the ledger and the
    records describe the same requests. `flush(done)` is called every 100 completed requests so the caller can write
    the ledger incrementally.
    对每个（变体，题目）调用 deepseek-flash 一次，按变体优先、时间线顺序（使共享会话卡片的题目连续），计划由 build_plan
    按变体截断；预估超过上限则在第一次调用前停止，累计台账超过上限则中途停止。workers > 1：ThreadPoolExecutor，每工作
    线程一个 DeepSeekClient，共用带锁台账，按 `workers` 大小分块处理（块内调用顺序不保证）；分块飞行中出现的异常或中断
    会先等该块其余期货完成并把答案写入记录再重新抛出，使台账与记录描述同一批请求。每完成 100 次请求调用一次
    `flush(done)`，使调用方能增量写台账。

    Input / 输入：
        records: variant → records in timeline order.
        records：变体 → 按时间线顺序的记录。

        questions: id → Question.
        questions：id → Question。

        meter: The ledger.
        meter：台账。

        max_requests: Hard cap on requests (split evenly across variants).
        max_requests：请求数硬上限（均分到各变体）。

        cost_cap: USD cap.
        cost_cap：美元上限。

        workers: Parallel clients.
        workers：并行客户端数。

        flush: Optional progress callback (completed request count).
        flush：可选的进度回调（已完成请求数）。

    Output / 输出：
        {'projected_usd', 'stopped', 'reason', 'requests', 'cost_usd', 'tier_at_start', 'planned_requests',
        'planned_per_variant', 'workers'}.
        {'projected_usd', 'stopped', 'reason', 'requests', 'cost_usd', 'tier_at_start', 'planned_requests',
        'planned_per_variant', 'workers'}。

    State effect / 状态影响：
        Network calls; mutates the records in place.
        网络调用；原地修改记录。
    """
    plan, per_variant = build_plan(records, max_requests)
    projected = project_cost(plan, questions)
    status: dict[str, Any] = {"projected_usd": round(projected, 6), "stopped": False, "reason": None, "requests": 0, "cost_usd": 0.0,
                              "tier_at_start": rate_tier(utc_now()), "planned_requests": len(plan), "planned_per_variant": per_variant,
                              "workers": int(workers)}  # fmt: skip
    if projected > cost_cap:
        status.update({"stopped": True, "reason": f"projected {projected:.4f} USD exceeds cap {cost_cap} USD; no request sent"})
        return status
    load_api_key(ENV_PATH)
    workers = max(1, int(workers))
    clients: queue.Queue[DeepSeekClient] = queue.Queue()
    for _ in range(workers):
        clients.put(DeepSeekClient(meter))

    def one(item: tuple[str, dict[str, Any]]) -> tuple[dict[str, Any], AnswerResult]:
        variant, rec = item
        client = clients.get()
        try:
            q = questions[rec["id"]]
            return rec, ask(client, q, build_prompt(q, rec["context"], PROMPT_ORDER[variant]), arm=ANSWER_LABEL[variant])
        finally:
            clients.put(client)

    done = 0

    def settle(rec: dict[str, Any], res: AnswerResult) -> None:
        nonlocal done
        apply_answer(rec, res)
        done += 1
        status["requests"] = done
        if done % 100 == 0:
            print(f"  [answer] {done}/{len(plan)} cost_usd={meter.spent_usd():.5f}", flush=True)
            if flush is not None:
                flush(done)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(plan), workers):
            spent = meter.spent_usd()
            if spent > cost_cap:
                status.update({"stopped": True, "reason": f"running cost {spent:.4f} USD exceeds cap {cost_cap} USD after {done} requests"})
                break
            chunk = plan[start : start + workers]
            if workers == 1:
                settle(*one(chunk[0]))
                continue
            # Futures settle in plan order. When an exception or an interrupt surfaces while the chunk is in flight,
            # the other workers' calls are already in the ledger, so the handler waits for them and writes every
            # finished answer into its record before re-raising: calls.jsonl and per_question_*.jsonl then describe
            # the same requests (pool.map discarded the whole chunk). A record that already carries `pred` is skipped.
            # 期货按计划顺序落账。分块飞行中出现异常或中断时，其他工作线程的调用已在台账里，因此处理器先等它们完成、
            # 把每个已完成的答案写入记录再重新抛出：calls.jsonl 与 per_question_*.jsonl 描述同一批请求（pool.map 会丢掉
            # 整个分块）。已带 `pred` 的记录跳过。
            futures = [pool.submit(one, item) for item in chunk]
            try:
                for fut in futures:
                    settle(*fut.result())
            except BaseException:
                pending = sum(1 for f in futures if not f.done())
                if pending:
                    print(f"  [answer] interrupted; waiting for {pending} in-flight request(s) before writing the ledger", flush=True)
                wait(futures)
                for fut in futures:
                    if fut.cancelled() or fut.exception() is not None:
                        continue
                    rec, res = fut.result()
                    if "pred" not in rec:
                        settle(rec, res)
                raise
    status["requests"] = done
    status["cost_usd"] = round(meter.spent_usd(), 6)
    return status


# ---------------------------------------------------------------------------
# Parity against the recorded run
# 与记录 run 的一致性
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file (blank and unparsable lines skipped).
    解析 JSONL 文件（跳过空行与不可解析行）。
    """
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _scores_close(a: Any, b: Any) -> bool:
    """Whether two top_scores lists agree element-wise within SCORE_TOL.
    两个 top_scores 列表是否逐项在 SCORE_TOL 内一致。
    """
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    try:
        return all(abs(float(x) - float(y)) <= SCORE_TOL for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def parity_check(
    run_dir: Path, hook: ReaderContexts, *, atoms: int | None, complete: bool, retain: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Compare the replay with the run's recorded files: 'rows' on returned_batches / returned_count / context_chars
    (strict) and top_scores (within SCORE_TOL; exact mismatches reported as informational) against
    questions_elite_rows.jsonl; 'card_1200' on returned_batches / context_chars / used_tokens / items / mode against
    questions_elite_card.jsonl; 'retain' on each replayed batch's added count and event histogram against
    retain_elite.jsonl; the final stored_count against summary.json's engines.elite.stats.stored_count when every
    step ran. The run stores no context text, so this oracle is NECESSARY, not sufficient: equal chars + batches +
    measurements cannot distinguish a card whose lines are permuted (compile() breaks within-slot ties by atom id,
    which the recorded run minted at random). Each section also aggregates its mismatches per field (count, histogram of replayed - recorded
    for numeric fields, first / last timeline position, whether the positions form one contiguous run) so a single
    store-level divergence shows as one event, and the card section reports for each mismatched question whether the
    char delta fits inside the replayed card's '## Next Action' section (the only rendered text that is not counted in
    used_tokens and that names the first failed path / first three constraints, whose order is id-dependent).
    把回放与 run 记录文件比较：'rows' 对 questions_elite_rows.jsonl 比 returned_batches / returned_count / context_chars
    （严格）与 top_scores（SCORE_TOL 内；精确不一致作为信息报告）；'card_1200' 对 questions_elite_card.jsonl 比
    returned_batches / context_chars / used_tokens / items / mode；'retain' 对 retain_elite.jsonl 比每个回放批次的 added
    计数与事件直方图；全部步骤都跑时再比最终 stored_count 与 summary.json 的 engines.elite.stats.stored_count。run 没有存
    上下文文本，因此这个 oracle 是"必要"而非"充分"：字符数 + 批次 + 度量相等无法区分行序被置换的卡片（compile() 用原子
    id 打破槽内平局，而记录 run 的 id 是随机铸造的）。每节还按字段汇总不一致（计数、数值字段的 replayed - recorded 直方图、首 / 末时间
    线位置、位置是否连成一段），使单个库级分歧显示为一个事件；卡片节还对每道不一致题目报告字符差是否落在回放卡片的
    '## Next Action' 节内（唯一不计入 used_tokens 的渲染文本，它点名首个失败路径 / 前三条约束，其顺序依赖 id）。

    Input / 输入：
        run_dir: The run directory.
        run_dir：run 目录。

        hook: The filled ReaderContexts.
        hook：填好的 ReaderContexts。

        atoms: Live atom count at close (run_arm 'atoms').
        atoms：关闭时的存活原子数（run_arm 的 'atoms'）。

        complete: Whether every timeline step ran (stored_count is only comparable then).
        complete：是否跑完全部时间线步骤（只有那时 stored_count 才可比）。

        retain: run_arm's retain rows (None skips the retain comparison).
        retain：run_arm 的 retain 行（None 则跳过 retain 比较）。

    Output / 输出：
        {'rows', 'card_1200', 'retain', 'stored_count', 'ok', 'oracle_note'}; each section has compared / mismatches /
        questions_mismatched / first (<= 5 {id, position, field, recorded, replayed}) / by_field ({field: {mismatches,
        delta_histogram, first_position, last_position, contiguous}}); rows adds top_scores_exact_mismatches; card_1200
        adds next_action ({questions, delta_within_next_action, next_action_chars_first}).
        {'rows', 'card_1200', 'retain', 'stored_count', 'ok', 'oracle_note'}；每节含 compared / mismatches /
        questions_mismatched / first（≤ 5 条 {id, position, field, recorded, replayed}）/ by_field（{字段: {mismatches,
        delta_histogram, first_position, last_position, contiguous}}）；rows 另有 top_scores_exact_mismatches；card_1200
        另有 next_action（{questions, delta_within_next_action, next_action_chars_first}）。

    State effect / 状态影响：
        Reads the run directory.
        读取 run 目录。
    """

    def compare(recorded: list[dict[str, Any]], fields: dict[str, Any], key: str = "id", positions: dict[str, int] | None = None) -> dict[str, Any]:
        by_id = {str(r.get(key)): r for r in recorded}
        rec_pos = {str(r.get(key)): r.get("position") for r in recorded}
        compared = 0
        exact_score_mism = 0
        mism: list[dict[str, Any]] = []
        for qid, replayed in fields.items():
            rec = by_id.get(qid)
            pos = (positions or {}).get(qid, rec_pos.get(qid))
            if rec is None or replayed is None:
                mism.append({"id": qid, "position": pos, "field": "(record)", "recorded": rec is not None, "replayed": replayed is not None})
                continue
            compared += 1
            for name, value in replayed.items():
                got = rec.get("card", {}).get(name) if name in ("used_tokens", "items", "mode") else rec.get(name)
                if name == "top_scores":
                    if got != value:
                        exact_score_mism += 1
                    if _scores_close(got, value):
                        continue
                if got != value:
                    mism.append({"id": qid, "position": pos, "field": name, "recorded": got, "replayed": value})
        # Per-field aggregation: a store-level divergence at one batch shows up as one contiguous run of positions
        # with one repeated delta, which the first-5 listing alone cannot convey.
        # 按字段汇总：某一批次处的库级分歧表现为一段连续位置、同一个重复的差值，仅列前 5 条无法传达这一点。
        by_field: dict[str, dict[str, Any]] = {}
        for name in sorted({m["field"] for m in mism}):
            rows_f = [m for m in mism if m["field"] == name]
            pos_f = sorted({int(m["position"]) for m in rows_f if isinstance(m.get("position"), int)})
            deltas: Counter = Counter()
            for m in rows_f:
                if isinstance(m["recorded"], (int, float)) and isinstance(m["replayed"], (int, float)) and not isinstance(m["recorded"], bool):
                    deltas[str(round(m["replayed"] - m["recorded"], 6))] += 1
            by_field[name] = {
                "mismatches": len(rows_f), "delta_histogram": dict(deltas) or None,
                "first_position": pos_f[0] if pos_f else None, "last_position": pos_f[-1] if pos_f else None,
                "contiguous": (pos_f[-1] - pos_f[0] + 1 == len(pos_f)) if pos_f else None,
            }  # fmt: skip
        out = {"compared": compared, "mismatches": len(mism), "questions_mismatched": len({m["id"] for m in mism}), "first": mism[:5],
               "by_field": by_field}  # fmt: skip
        if any("top_scores" in f for f in fields.values() if f):
            out["top_scores_exact_mismatches"] = exact_score_mism
            out["top_scores_tolerance"] = SCORE_TOL
        return out

    skipped_section = {"compared": 0, "mismatches": 0, "questions_mismatched": 0, "first": [], "by_field": {}}
    rows_fields = {qid: b for qid, b in hook.baseline.items()}
    # Only questions that actually carry a card_1200 context enter the card comparison; with the variant not run the
    # section is skipped instead of reporting every question as a missing record.
    # 只有真正带 card_1200 上下文的题目进入卡片比较；该变体未运行时整节跳过，而不是把每道题报成缺记录。
    card_fields: dict[str, dict[str, Any]] = {
        qid: {
            "returned_batches": c["returned_batches"], "context_chars": c["context_chars"],
            "used_tokens": c["card"]["used_tokens"], "items": c["card"]["items"], "mode": c["card"]["mode"],
        }
        for qid, variants in hook.by_question.items()
        if (c := variants.get("card_1200")) is not None
    }  # fmt: skip
    rows = compare(_read_jsonl(run_dir / "questions_elite_rows.jsonl"), rows_fields, positions=hook.positions)
    recorded_cards = _read_jsonl(run_dir / "questions_elite_card.jsonl")
    card = compare(recorded_cards, card_fields, positions=hook.positions) if card_fields else dict(skipped_section, skipped="card_1200 not run")
    if card_fields:
        card["next_action"] = next_action_diagnostic(hook, {str(r.get("id")): r.get("context_chars") for r in recorded_cards})
    retain_fields: dict[str, dict[str, Any]] = {}
    replayed_info: dict[str, dict[str, int]] = {}
    for r in retain or []:
        events: dict[str, int] = {}
        for sub in r.get("subs", []):
            for ev, n in (sub.get("events") or {}).items():
                events[ev] = events.get(ev, 0) + int(n)
        retain_fields[str(r["batch_id"])] = {"added": r.get("added"), "events": events}
        replayed_info[str(r["batch_id"])] = {
            "warnings": sum(int(sub.get("warnings") or 0) for sub in r.get("subs", [])),
            "items_skipped": sum(int(sub.get("items_skipped") or 0) for sub in r.get("subs", [])),
        }
    recorded_retain = _read_jsonl(run_dir / "retain_elite.jsonl")
    retain_cmp = compare(recorded_retain, retain_fields, key="batch_id") if retain_fields else dict(skipped_section, skipped="no retain rows")
    # Informational (never fails parity): per-batch library warning counts and items_skipped. real_arms stores the
    # warning texts and counts items_skipped only from messages naming 'memory_extraction'; replay_ab stores the count
    # and also counts 'invalid event_idx' drops, so items_skipped is expected to differ where warnings agree.
    # 信息项（不影响一致性判定）：逐批次的库警告数与 items_skipped。real_arms 存警告文本、只从含 'memory_extraction'
    # 的消息计 items_skipped；replay_ab 存计数且把 'invalid event_idx' 丢弃也计入，因此警告一致而 items_skipped 不同是预期的。
    info_mism: list[dict[str, Any]] = []
    warn_diff = skip_diff = 0
    recorded_by_batch = {str(r.get("batch_id")): r for r in recorded_retain}
    for bid, got in replayed_info.items():
        rec = recorded_by_batch.get(bid)
        if rec is None:
            continue
        rec_warn = sum(len(sub.get("warnings") or []) for sub in rec.get("subs", []))
        rec_skip = sum(int(sub.get("items_skipped") or 0) for sub in rec.get("subs", []))
        if rec_warn != got["warnings"]:
            warn_diff += 1
            info_mism.append({"id": bid, "field": "warnings", "recorded": rec_warn, "replayed": got["warnings"]})
        if rec_skip != got["items_skipped"]:
            skip_diff += 1
            info_mism.append({"id": bid, "field": "items_skipped", "recorded": rec_skip, "replayed": got["items_skipped"]})
    retain_cmp["informational"] = {
        "warnings_count_diff_batches": warn_diff, "items_skipped_diff_batches": skip_diff, "first": info_mism[:5],
        "note": "not part of ok; items_skipped rules differ between real_arms ('memory_extraction' messages only) and "
                "replay_ab (also 'invalid event_idx'), warnings should agree",
    }  # fmt: skip
    recorded_count: int | None = None
    try:
        s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        recorded_count = int(s["engines"]["elite"]["stats"]["stored_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        recorded_count = None
    stored = {
        "replayed": atoms, "recorded": recorded_count, "expected_constant": STORED_COUNT_ORACLE,
        "match": (atoms == recorded_count) if (complete and atoms is not None and recorded_count is not None) else None,
        "note": None if complete else "skipped: not every timeline step ran (--steps)",
    }  # fmt: skip
    ok = rows["mismatches"] == 0 and card["mismatches"] == 0 and retain_cmp["mismatches"] == 0 and stored["match"] is not False
    return {
        "rows": rows, "card_1200": card, "retain": retain_cmp, "stored_count": stored, "ok": ok,
        "oracle_note": "necessary, not sufficient: the run dir stores no context text; equal chars / batches / measurements do not "
                       "prove byte identity (card lines within one compile slot are tie-broken by atom id, which the recorded run "
                       "minted at random).",
    }  # fmt: skip


def next_action_section_chars(card_text: str) -> int:
    """Length in chars of the '## Next Action' section of a rendered card (heading line through the line before the
    next '## ' heading, newline included); 0 when absent.
    渲染卡片中 '## Next Action' 节的字符数（标题行到下一个 '## ' 标题之前的行，含换行）；缺失为 0。

    Input / 输入：
        card_text: ir.to_text().
        card_text：ir.to_text()。

    Output / 输出：
        The length.
        长度。

    State effect / 状态影响：
        None.
        无。
    """
    start = card_text.find("## Next Action")
    if start < 0:
        return 0
    end = card_text.find("\n## ", start + 1)
    return len(card_text) - start if end < 0 else end + 1 - start


def next_action_diagnostic(hook: ReaderContexts, recorded_chars: dict[str, Any]) -> dict[str, Any]:
    """For every card_1200 question whose context_chars mismatches, whether |replayed - recorded| is at most the
    replayed card's '## Next Action' section length; a run-level hint (not a proof) that the divergence sits in that
    derived sentence, since the run dir keeps no card text to compare line by line.
    对每道 context_chars 不一致的 card_1200 题目，判断 |replayed - recorded| 是否不超过回放卡片 '## Next Action' 节的
    长度；这是 run 级别的提示（非证明），提示分歧位于该派生句，因为 run 目录没有可逐行比较的卡片文本。

    Input / 输入：
        hook: The filled ReaderContexts.
        hook：填好的 ReaderContexts。

        recorded_chars: question id → recorded card context_chars.
        recorded_chars：题目 id → 记录的卡片 context_chars。

    Output / 输出：
        {'questions', 'delta_within_next_action', 'delta_exceeds_next_action', 'first' (<= 5 {id, delta_chars,
        next_action_chars})}.
        {'questions', 'delta_within_next_action', 'delta_exceeds_next_action', 'first'（≤ 5 条 {id, delta_chars,
        next_action_chars}）}。

    State effect / 状态影响：
        None.
        无。
    """
    within = exceeds = 0
    first: list[dict[str, Any]] = []
    for qid, variants in hook.by_question.items():
        ctx = variants.get("card_1200")
        rec_chars = recorded_chars.get(qid)
        if ctx is None or not isinstance(rec_chars, int):
            continue
        delta = abs(int(ctx["context_chars"]) - int(rec_chars))
        if delta == 0:
            continue
        na = next_action_section_chars(ctx["context"])
        if delta <= na:
            within += 1
        else:
            exceeds += 1
        if len(first) < 5:
            first.append({"id": qid, "delta_chars": delta, "next_action_chars": na})
    return {"questions": within + exceeds, "delta_within_next_action": within, "delta_exceeds_next_action": exceeds, "first": first}


def ingest_gaps(res: dict[str, Any]) -> dict[str, Any]:
    """Collect the ingest replay's gaps from run_arm's result: generate_lines misses, api_error subs and unresolved
    ('missing') leaves, with their batch ids.
    从 run_arm 的结果收集摄入回放的缺口：generate_lines 未命中、api_error 子批次与未解析（'missing'）叶子，附批次 id。

    Input / 输入：
        res: run_arm's result ('replay', 'retain').
        res：run_arm 的结果（'replay'、'retain'）。

    Output / 输出：
        {'fatal', 'lines_misses', 'api_error': [{batch_id, split, error}], 'missing': [{batch_id, split}], 'batch_ids'}.
        {'fatal', 'lines_misses', 'api_error': [{batch_id, split, error}], 'missing': [{batch_id, split}], 'batch_ids'}。

    State effect / 状态影响：
        None.
        无。
    """
    api_error: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in res.get("retain", []):
        for sub in row.get("subs", []):
            if sub.get("outcome") == "api_error":
                api_error.append({"batch_id": row["batch_id"], "split": sub.get("split"), "error": sub.get("error")})
            elif sub.get("outcome") == "missing":
                missing.append({"batch_id": row["batch_id"], "split": sub.get("split")})
    lines_misses = int((res.get("replay") or {}).get("lines_misses") or 0)
    batch_ids = sorted({g["batch_id"] for g in api_error + missing})
    return {
        "fatal": bool(lines_misses or api_error or missing), "lines_misses": lines_misses,
        "api_error": api_error, "missing": missing, "batch_ids": batch_ids,
    }  # fmt: skip


def package_versions() -> dict[str, str | None]:
    """Installed versions of VERSIONED_PACKAGES (None when absent), so a replay can be matched to the environment of
    the recorded run (the 2026-09-11 run recorded none; its cp312 wheel set was fastembed 0.8.0 + onnxruntime 1.29.0).
    VERSIONED_PACKAGES 的已安装版本（缺失为 None），使回放能与记录 run 的环境对上（2026-09-11 的 run 没有记录；其
    cp312 wheel 集合是 fastembed 0.8.0 + onnxruntime 1.29.0）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        name → version string or None.
        名称 → 版本字符串或 None。

    State effect / 状态影响：
        None.
        无。
    """
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def parity_caveats(parity: dict[str, Any] | None) -> list[str]:
    """Caveat lines derived from a parity result: nothing when it passed, otherwise one line per failing section
    naming the mismatch counts, so summary.md never calls a non-identical baseline 'identical'.
    由一致性结果派生的注意事项：通过时为空，否则每个失败的节一行、写明不一致计数，使 summary.md 不会把并非相同的基线
    称为"相同"。

    Input / 输入：
        parity: parity_check's result or None.
        parity：parity_check 的结果或 None。

    Output / 输出：
        Lines (possibly empty).
        文本行（可能为空）。

    State effect / 状态影响：
        None.
        无。
    """
    if parity is None:
        return ["No --parity-check was run: rows and card_1200 are re-derived baselines, not verified against the recorded run."]
    if parity["ok"]:
        return []
    out: list[str] = []
    for label, name in (("rows", "rows"), ("card_1200", "card_1200")):
        sec = parity[label]
        if sec["mismatches"]:
            fields = "; ".join(
                f"{f}: {d['mismatches']} (positions {d['first_position']}..{d['last_position']}"
                f"{', contiguous' if d['contiguous'] else ''}; deltas {d['delta_histogram']})"
                for f, d in (sec.get("by_field") or {}).items()
            )
            na = sec.get("next_action")
            na_txt = (f" Char deltas within the replayed '## Next Action' section: {na['delta_within_next_action']}/{na['questions']}."
                      if na and na["questions"] else "")
            out.append(
                f"PARITY: {name} differs from the recorded run on {sec['questions_mismatched']}/{sec['compared']} questions "
                f"({sec['mismatches']} field mismatches; {fields}); treat {name} as a re-derived baseline, not the recorded one.{na_txt}"
            )
    if parity["retain"]["mismatches"]:
        out.append(f"PARITY: retain differs on {parity['retain']['questions_mismatched']}/{parity['retain']['compared']} batches.")
    if parity["stored_count"]["match"] is False:
        out.append(f"PARITY: stored_count replayed={parity['stored_count']['replayed']} recorded={parity['stored_count']['recorded']}.")
    return out


# ---------------------------------------------------------------------------
# Summary
# 汇总
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def cny_cost(hit: int | None, miss: int | None, output: int | None, tier: str) -> float | None:
    """CNY of a token triple at one tier (None when any count is unknown).
    某档位下一组 token 的人民币费用（任一计数未知时为 None）。

    Input / 输入：
        hit, miss, output: Token counts.
        hit、miss、output：token 数。

        tier: 'offpeak' | 'peak'.
        tier：'offpeak' | 'peak'。

    Output / 输出：
        CNY.
        人民币。

    State effect / 状态影响：
        None.
        无。
    """
    if hit is None or miss is None or output is None:
        return None
    p = PRICES_CNY_PER_M[tier]
    return (hit * p["cache_hit"] + miss * p["cache_miss"] + output * p["output"]) / 1_000_000.0


def variant_summary(variant: str, recs: list[dict[str, Any]], meter: Meter) -> dict[str, Any]:
    """Accuracy, context size, prompt tokens, cache-hit share, CNY (off-peak / peak) and USD ledger of one variant.
    一个变体的准确率、上下文大小、提示词 token、缓存命中占比、人民币（非高峰 / 高峰）与美元台账。

    Input / 输入：
        variant: Variant name.
        variant：变体名。

        recs: Its records (answered in place).
        recs：其记录（已原地答题）。

        meter: The ledger.
        meter：台账。

    Output / 输出：
        The summary dict.
        汇总字典。

    State effect / 状态影响：
        None.
        无。
    """
    answered = [r for r in recs if "pred" in r]
    scored = [r for r in recs if not r.get("excluded_future")]
    stage = meter.stage_summary(meter.calls_for(ANSWER_LABEL[variant], "answer"))
    tok = stage["tokens"]
    # priced = records whose last attempt carried usage; the CNY total covers exactly these, so the per-question CNY
    # and the mean prompt tokens share this denominator (a request whose every attempt failed adds 0 CNY and no usage).
    # priced = 最后一次尝试带 usage 的记录；人民币总额恰好覆盖这些，因此每题人民币与平均提示词 token 共用此分母
    # （所有尝试都失败的请求贡献 0 人民币且无 usage）。
    priced = [r for r in answered if r.get("answer_usage", {}).get("prompt_tokens") is not None]
    prompt_tokens = [r["answer_usage"]["prompt_tokens"] for r in priced]
    hit_share = (tok["cache_hit"] / tok["prompt"]) if (tok["cache_hit"] is not None and tok["prompt"]) else None
    cny = {tier: cny_cost(tok["cache_hit"], tok["cache_miss"], tok["completion"], tier) for tier in ("offpeak", "peak")}
    # Cache-neutral CNY: every prompt token at the cache-miss price. Variants are answered variant-major, so a later
    # variant can hit on the question head another variant sent minutes earlier; the neutral figure removes that
    # order dependence and is the one to compare across variants (the measured CNY stays for session_prefix vs
    # session_1200, where the hit is the quantity of interest).
    # 缓存中性人民币：所有提示词 token 按未命中价。变体按变体优先顺序答题，靠后的变体可能命中另一变体几分钟前发送
    # 的题目头部；中性数字去掉这种顺序依赖，是跨变体比较应看的数字（实测人民币保留给 session_prefix vs session_1200，
    # 那里命中本身才是关注量）。
    cny_neutral = {tier: cny_cost(0, tok["prompt"], tok["completion"], tier) for tier in ("offpeak", "peak")}
    n_ans = len(answered)
    n_priced = len(priced)
    cny_q = {tier: (v / n_priced if (v is not None and n_priced) else None) for tier, v in cny.items()}
    cny_q_neutral = {tier: (v / n_priced if (v is not None and n_priced) else None) for tier, v in cny_neutral.items()}
    # A variant answered on fewer questions than it holds (plan truncation, cap stop) is not extrapolated to 204.
    # 回答题数少于其持有题数的变体（计划截断、上限停止）不外推到 204 题。
    partial = 0 < n_ans < len(recs)
    cards = [r["card"] for r in recs if "card" in r]
    rows_info = [r["rows"] for r in recs if isinstance(r.get("rows"), dict)]
    rows_totals: dict[str, Any] | None = None
    if rows_info:
        rows_totals = {
            "rows_in_total": sum(int(x.get("rows_in", 0)) for x in rows_info),
            "rows_kept_total": sum(int(x.get("rows_kept", 0)) for x in rows_info),
            "rows_dropped_total": sum(int(x.get("rows_in", 0)) - int(x.get("rows_kept", 0)) for x in rows_info),
            "questions_with_drops": sum(1 for x in rows_info if int(x.get("rows_in", 0)) > int(x.get("rows_kept", 0))),
            "stamps_distinct_mean": _mean([float(x["stamps_distinct"]) for x in rows_info if "stamps_distinct" in x]),
            "batches_distinct_mean": _mean([float(x["batches_distinct"]) for x in rows_info if "batches_distinct" in x]),
        }
    if variant == "rows":
        hit_note = "hit@20 of the shared top-20 search"
    elif variant in ROWS_VARIANTS:
        hit_note = "any kept (deduplicated) row's batch is gold (k = rows_kept); hit_batch_top20 = the shared hit@20"
    elif variant == "hybrid_prefix":
        hit_note = "any session-card item's batch is gold OR any kept rows_ts row's batch is gold (components under hybrid)"
    else:
        hit_note = "any compile item's batch is gold (k = items)"
    # Component hits of hybrid_prefix over the scored records; part sizes over all records (as context_chars_mean).
    # hybrid_prefix 各部分的命中按计分记录计；部分大小按全部记录计（与 context_chars_mean 同口径）。
    hybrid: dict[str, Any] | None = None
    parts = [r["hybrid"] for r in scored if isinstance(r.get("hybrid"), dict)]
    if parts:
        n_parts = len(parts)
        hybrid = {
            "hit_session": round(sum(1 for p in parts if p["session_hit"]) / n_parts, 4),
            "hit_rows": round(sum(1 for p in parts if p["rows_hit"]) / n_parts, 4),
            "hit_both": round(sum(1 for p in parts if p["session_hit"] and p["rows_hit"]) / n_parts, 4),
            "hit_rows_only": round(sum(1 for p in parts if p["rows_hit"] and not p["session_hit"]) / n_parts, 4),
            "session_chars_mean": _mean([float(r["hybrid"]["session_chars"]) for r in recs if isinstance(r.get("hybrid"), dict)]),
            "rows_chars_mean": _mean([float(r["hybrid"]["rows_chars"]) for r in recs if isinstance(r.get("hybrid"), dict)]),
            "rows_header": HYBRID_ROWS_HEADER,
        }
    session: dict[str, Any] | None = None
    if variant in SESSION_MEMO_VARIANTS:
        # Cache-hit tokens split by memo state: the first question at a step sends a card DeepSeek has not seen in
        # this variant; a reused-card question is where the memories-first prefix can hit. Both shares are over the
        # prompt tokens of the priced records in that group.
        # 按记忆状态拆分缓存命中 token：某步的第一道题发送的是本变体里 DeepSeek 未见过的卡片；复用卡片的题目才是
        # 记忆在前的前缀可能命中的地方。两个占比都以该组已计价记录的提示词 token 为分母。
        def share(group: list[dict[str, Any]]) -> float | None:
            p = sum(r["answer_usage"]["prompt_tokens"] for r in group)
            h = sum(r["answer_usage"]["cache_hit_tokens"] or 0 for r in group)
            return round(h / p, 4) if p else None

        session = {
            "reused": sum(1 for r in recs if (r.get("session") or {}).get("reused")),
            "recomputed": sum(1 for r in recs if r.get("session") and not r["session"]["reused"] and r["session"]["key"]),
            "cache_hit_share_first_at_step": share([r for r in priced if r.get("session") and not r["session"]["reused"]]),
            "cache_hit_share_reused_card": share([r for r in priced if (r.get("session") or {}).get("reused")]),
        }
    return {
        "variant": variant,
        "prompt_order": PROMPT_ORDER[variant],
        "n": len(recs),
        "n_scored": len(scored),
        "answered_n": n_ans,
        "priced_n": n_priced,
        "answer_correct": sum(1 for r in answered if r.get("correct")),
        "answer_accuracy": round(sum(1 for r in answered if r.get("correct")) / n_ans, 4) if n_ans else None,
        "answer_valid_n": sum(1 for r in answered if r.get("pred")),
        "hit_batch": round(sum(1 for r in scored if r["hit_batch"]) / len(scored), 4) if scored else None,
        "hit_batch_top20": (round(sum(1 for r in scored if r.get("hit_batch_top20")) / len(scored), 4) if scored else None) if variant in ROWS_VARIANTS and variant != "rows" else None,
        "hit_note": hit_note,
        "context_chars_mean": _mean([float(r["context_chars"]) for r in recs]),
        "context_chars_median": _median([float(r["context_chars"]) for r in recs]),
        "est_tokens_mean": _mean([float(r["est_tokens"]) for r in recs]),
        "prompt_tokens_mean": _mean([float(t) for t in prompt_tokens]),
        "prompt_tokens_median": _median([float(t) for t in prompt_tokens]),
        "cache_hit_share": round(hit_share, 4) if hit_share is not None else None,
        "answer_tokens": tok,
        "answer_cost_usd": stage["cost_usd"],
        "answer_attempts": stage["attempts"],
        "answer_tier_attempts": stage["tier_attempts"],
        "latency_ms": stage["latency_ms"],
        "cny_total": {t: (round(v, 6) if v is not None else None) for t, v in cny.items()},
        "cny_per_question": {t: (round(v, 6) if v is not None else None) for t, v in cny_q.items()},
        "cny_per_question_denominator": "priced_n (records whose last attempt carried usage)",
        "cny_per_204": {t: (round(v * N_QUESTIONS, 4) if (v is not None and not partial) else None) for t, v in cny_q.items()},
        "cny_total_no_cache": {t: (round(v, 6) if v is not None else None) for t, v in cny_neutral.items()},
        "cny_per_question_no_cache": {t: (round(v, 6) if v is not None else None) for t, v in cny_q_neutral.items()},
        "cny_per_204_no_cache": {t: (round(v * N_QUESTIONS, 4) if (v is not None and not partial) else None) for t, v in cny_q_neutral.items()},
        "cny_no_cache_rule": "every prompt token at the cache_miss price (order-independent; use for cross-variant comparison)",
        "partial": partial,
        "rows": rows_totals,
        "card": {
            "compile_ms_median": _median([float(c["compile_ms"]) for c in cards]),
            "used_tokens_mean": _mean([float(c["used_tokens"]) for c in cards]),
            "items_mean": _mean([float(c["items"]) for c in cards]),
            "over_budget_questions": sum(1 for c in cards if c["used_tokens"] > c["budget_tokens"]),
            "budget_tokens": cards[0]["budget_tokens"] if cards else None,
        } if cards else None,
        "session": session,
        "hybrid": hybrid,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render summary.md: the variant table (accuracy x tokens/question x cache-hit share x CNY per 204 x CNY per
    question) plus run, parity, answer-pass status, method and caveats.
    渲染 summary.md：变体表（准确率 x 每题 token x 缓存命中占比 x 每 204 题人民币 x 每题人民币）加 run、一致性、
    答题状态、方法与注意事项。

    Input / 输入：
        summary: The summary dict.
        summary：汇总字典。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """

    def pct(v: Any) -> str:
        return "-" if v is None else f"{100 * v:.1f}%"

    def num(v: Any, digits: int = 1) -> str:
        return "-" if v is None else f"{v:.{digits}f}"

    def pair(d: dict[str, Any] | None, digits: int) -> str:
        if not d:
            return "-"
        return f"{num(d.get('offpeak'), digits)} / {num(d.get('peak'), digits)}"

    v = summary["variants"]
    lines = [
        "# Reader-side A/B: eleven answer contexts on one replayed lines store",
        "# 阅读端 A/B：同一回放行式库上的十一种答题上下文",
        "",
        f"run: `{summary['run_dir']}` (replayed at {summary['started_utc']}; questions={summary['questions']} steps_run={summary['steps_run']} "
        f"batches={summary['batches']} replies replayed={summary['replies_replayed']} missing={summary['replies_missing']}; "
        f"atoms at close={summary['atoms']}; replay lines_calls={summary['replay']['lines_calls']} lines_misses={summary['replay']['lines_misses']})",
        "",
        f"session card memo: compiles={summary['session_memo']['compiles']} reuses={summary['session_memo']['reuses']} "
        f"without_ingest={summary['session_memo']['without_ingest']} ingest_skipped={summary['session_memo']['ingest_skipped']}; "
        f"rows baseline text mismatches vs run_arm={summary['session_memo']['rows_baseline_mismatch']}",
        "",
        f"packages: {json.dumps(summary['config']['package_versions'], ensure_ascii=False)}; "
        f"atom ids: {summary['config'].get('atom_ids', 'unknown')} (card_* / session_* are re-derived baselines, see Method)",
        "",
        "## Variants / 变体 (answer pass: deepseek-flash, temperature 0, max_tokens 8; CNY = off-peak / peak at 0.02·1·4 / 0.04·2·8 per 1M)",
        "",
        "| variant | prompt order | accuracy (c/n) | hit | rows kept / dropped | context chars (mean) | prompt tokens / question | cache-hit share | "
        "session memo (recomputed / reused; hit share first / reused) | CNY per 204 questions, cache-neutral (off-peak / peak) | "
        "CNY per 204 questions, measured (off-peak / peak) | CNY per question, measured (off-peak / peak) | USD ledger |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in summary["variants_run"]:
        s = v[name]
        hit = pct(s["hit_batch"]) + (f" (top-20 {pct(s['hit_batch_top20'])})" if s.get("hit_batch_top20") is not None else "")
        if s.get("hybrid"):
            hit += f" (session {pct(s['hybrid']['hit_session'])} / rows {pct(s['hybrid']['hit_rows'])})"
        memo = "-"
        if s.get("session"):
            m = s["session"]
            memo = f"{m['recomputed']} / {m['reused']}; {pct(m['cache_hit_share_first_at_step'])} / {pct(m['cache_hit_share_reused_card'])}"
        rows_col = "-"
        if s.get("rows"):
            r = s["rows"]
            rows_col = f"{r['rows_kept_total']} / {r['rows_dropped_total']} (q with drops {r['questions_with_drops']})"
        lines.append(
            f"| {name} | {s['prompt_order']} | {pct(s['answer_accuracy'])} ({s['answer_correct']}/{s['answered_n']}) | {hit} | {rows_col} | "
            f"{num(s['context_chars_mean'], 0)} | {num(s['prompt_tokens_mean'], 1)} | {pct(s['cache_hit_share'])} | {memo} | "
            f"{pair(s.get('cny_per_204_no_cache'), 4)} | {pair(s['cny_per_204'], 4)} | {pair(s['cny_per_question'], 6)} | {num(s['answer_cost_usd'], 6)} |"
        )
    lines += [
        "",
        "hit: rows = hit@20 of the shared search; rows_td / rows_ts = a kept (deduplicated) row's batch is gold, with the shared "
        "hit@20 in parentheses; card / session = any compile item's batch is gold; hybrid_prefix = session-card hit OR kept rows_ts "
        "hit, the two component hits in parentheses (its session memo column is session_1200's memo). rows kept / dropped = totals of the dedup + "
        "substring rule over all questions (dropped 0 means rows_td is purely a time-sort of rows). CNY per question divides the "
        "ledger by priced_n (records whose last attempt carried usage); prompt tokens / question averages the same records. "
        "cache-neutral CNY prices every prompt token at the cache-miss rate and is the column to compare across variants "
        "(measured CNY carries incidental cross-variant prompt-cache hits); CNY per 204 is blank for a variant answered on fewer "
        "questions than it holds.",
        "hit：rows = 共享检索的 hit@20；rows_td / rows_ts = 保留（去重后）行的批次命中 gold，括号内为共享 hit@20；card / session = "
        "任一编译项目的批次命中 gold；hybrid_prefix = 会话卡片命中 OR 保留 rows_ts 行命中，括号内为两部分各自的命中（其 session memo "
        "列即 session_1200 的记忆）。rows kept / dropped = 去重 + 子串规则在全部题目上的合计（dropped 为 0 表示 rows_td 只是 rows "
        "的时间排序）。每题人民币以 priced_n（最后一次尝试带 usage 的记录）为分母；每题提示词 token 取同一批记录的均值。缓存中性"
        "人民币把所有提示词 token 按未命中价计，是跨变体比较应看的列（实测人民币含跨变体的偶然缓存命中）；回答题数少于持有题数的"
        "变体不给出每 204 题人民币。",
        "",
        "## Answer pass / 答题", "", f"status: {json.dumps(summary['answer_pass'], ensure_ascii=False)}", "",
    ]
    if summary.get("parity") is not None:
        p = summary["parity"]
        lines += [
            "## Parity vs the recorded run / 与记录 run 的一致性",
            "",
            f"ok={p['ok']}; rows: compared={p['rows']['compared']} mismatches={p['rows']['mismatches']} "
            f"(questions {p['rows']['questions_mismatched']}; top_scores exact mismatches={p['rows'].get('top_scores_exact_mismatches')} "
            f"within tolerance {p['rows'].get('top_scores_tolerance')}); card_1200: compared={p['card_1200']['compared']} "
            f"mismatches={p['card_1200']['mismatches']} (questions {p['card_1200']['questions_mismatched']}); "
            f"retain batches: compared={p['retain']['compared']} mismatches={p['retain']['mismatches']} "
            f"(informational warnings-count diffs={p['retain'].get('informational', {}).get('warnings_count_diff_batches')} "
            f"items_skipped diffs={p['retain'].get('informational', {}).get('items_skipped_diff_batches')}); "
            f"stored_count replayed={p['stored_count']['replayed']} recorded={p['stored_count']['recorded']} match={p['stored_count']['match']}",
            "",
        ]
        lines.append(f"oracle: {p.get('oracle_note', '')}")
        lines.append("")
        for label in ("rows", "card_1200", "retain"):
            for f_name, d in (p[label].get("by_field") or {}).items():
                lines.append(f"- {label} {f_name}: {d['mismatches']} mismatches at positions {d['first_position']}..{d['last_position']} "
                             f"(contiguous={d['contiguous']}); replayed-recorded histogram {json.dumps(d['delta_histogram'])}")
            na = p[label].get("next_action")
            if na and na["questions"]:
                lines.append(f"- {label} char deltas within the replayed '## Next Action' section: {na['delta_within_next_action']}/{na['questions']} "
                             f"(exceeding: {na['delta_exceeds_next_action']}); first {json.dumps(na['first'])}")
            for m in p[label]["first"]:
                lines.append(f"- {label} q{m['id']} (pos {m.get('position')}) {m['field']}: recorded={json.dumps(m['recorded'], ensure_ascii=False)} "
                             f"replayed={json.dumps(m['replayed'], ensure_ascii=False)}")
        lines.append("")
    lines += ["## Method / 方法", "", summary["method"], "", "## Caveats / 注意", ""]
    lines += [f"- {c}" for c in summary["caveats"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# 主流程
# ---------------------------------------------------------------------------
def variant_records(variants: tuple[str, ...], base: list[dict[str, Any]], hook: ReaderContexts) -> dict[str, list[dict[str, Any]]]:
    """One record per (variant, question) from run_arm's records and the hook's contexts, in timeline order.
    由 run_arm 的记录与钩子的上下文生成每（变体，题目）一条记录，按时间线顺序。

    Input / 输入：
        variants: Variants run.
        variants：运行的变体。

        base: run_arm records.
        base：run_arm 记录。

        hook: The filled ReaderContexts.
        hook：填好的 ReaderContexts。

    Output / 输出：
        variant → records.
        变体 → 记录。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    for rec in base:
        ctxs = hook.by_question.get(rec["id"], {})
        for v in variants:
            ctx = ctxs.get(v)
            if ctx is None:
                continue
            row = {
                "variant": v, "prompt_order": PROMPT_ORDER[v], "id": rec["id"], "order": rec["order"], "position": rec["position"],
                "type": rec["type"], "query_day": rec["query_day"], "query_time": rec["query_time"],
                "excluded_future": rec["excluded_future"], "gold_batches": rec["gold_batches"],
                "gold_batches_stored": rec["gold_batches_stored"], "gold": rec["gold"],
            }  # fmt: skip
            row.update(ctx)
            out[v].append(row)
    return out


def counter_id_factory() -> Callable[[str], str]:
    """A deterministic atom-id maker for StateStore(id_factory=...): `prefix_<12-digit zero-padded counter>` (same
    width as the library's `prefix_<12 hex>`), one shared counter across prefixes.
    供 StateStore(id_factory=...) 使用的确定性原子 id 生成器：`prefix_<12 位零填充计数>`（与库的 `prefix_<12 hex>` 等宽），
    各前缀共用一个计数器。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The factory.
        工厂函数。

    State effect / 状态影响：
        The returned closure advances its counter.
        返回的闭包推进其计数器。
    """
    n = 0

    def make(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"{prefix}_{n:012d}"

    return make


def remove_stale_outputs(out: Path, *, keep_parity: bool, keep_gaps: bool) -> list[str]:
    """Delete an earlier run's per_question_*.jsonl, calls.jsonl, summary.json and summary.md (and parity.json /
    ingest_gaps.json when this run will not rewrite them) from --out, so the directory holds one run's files only;
    retain_*.jsonl / entity_axis_*.jsonl are overwritten by run_arm itself.
    删除 --out 中先前 run 的 per_question_*.jsonl、calls.jsonl、summary.json 与 summary.md（以及本次不会重写的
    parity.json / ingest_gaps.json），使目录只含一次 run 的文件；retain_*.jsonl / entity_axis_*.jsonl 由 run_arm 自行覆盖。

    Input / 输入：
        out: The result directory.
        out：结果目录。

        keep_parity: Whether this run writes parity.json (an old one is then overwritten, not deleted here).
        keep_parity：本次是否写 parity.json（旧文件将被覆盖，此处不删）。

        keep_gaps: Whether this run writes ingest_gaps.json.
        keep_gaps：本次是否写 ingest_gaps.json。

    Output / 输出：
        The names removed.
        删除的文件名。

    State effect / 状态影响：
        Deletes files under out.
        删除 out 下的文件。
    """
    removed: list[str] = []
    victims = list(out.glob("per_question_*.jsonl")) + [out / name for name in REWRITTEN_OUTPUTS]
    if not keep_parity:
        victims.append(out / "parity.json")
    if not keep_gaps:
        victims.append(out / "ingest_gaps.json")
    for path in victims:
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return removed


def baseline_status(name: str, parity: dict[str, Any] | None) -> str:
    """The parenthetical the method text attaches to a baseline variant: what the parity check established.
    方法文本附在基线变体后的括注：一致性检查所确立的内容。

    Input / 输入：
        name: 'rows' | 'card_1200'.
        name：'rows' | 'card_1200'。

        parity: parity_check's result or None.
        parity：parity_check 的结果或 None。

    Output / 输出：
        The phrase.
        短语。

    State effect / 状态影响：
        None.
        无。
    """
    same_call = "the same call as the streaming run"
    if parity is None:
        return f"{same_call}; re-derived, no --parity-check"
    sec = parity.get(name) or {}
    if sec.get("mismatches", 0) == 0 and sec.get("compared", 0) > 0:
        if name == "card_1200":
            return f"{same_call}; parity ok on chars / batches / measurements, line order within a slot not verifiable"
        return f"{same_call}; parity ok on batches / counts / chars / scores"
    return f"{same_call}; PARITY FAILED on {sec.get('questions_mismatched', '?')}/{sec.get('compared', '?')} questions, re-derived baseline"


def truncation_caveats(variants: dict[str, dict[str, Any]], status: dict[str, Any]) -> list[str]:
    """Caveat lines naming the variants answered on fewer questions than they hold (plan truncation or a cap stop),
    whose accuracy / CNY per question cover a different subset and whose CNY per 204 is withheld.
    点名回答题数少于持有题数的变体（计划截断或上限停止）的注意事项：其准确率 / 每题人民币覆盖不同子集，且不给出每 204
    题人民币。

    Input / 输入：
        variants: variant → variant_summary.
        variants：变体 → variant_summary。

        status: answer_pass's status.
        status：answer_pass 的状态。

    Output / 输出：
        Lines (empty when no answer pass ran or every variant is complete).
        文本行（未答题或所有变体完整时为空）。

    State effect / 状态影响：
        None.
        无。
    """
    if status.get("reason") == "--no-answer":
        return []
    partial = {v: (s["answered_n"], s["n"]) for v, s in variants.items() if s.get("partial")}
    unanswered = [v for v, s in variants.items() if s["answered_n"] == 0 and s["n"] > 0]
    out: list[str] = []
    if partial:
        out.append("TRUNCATION: " + ", ".join(f"{v} answered {a}/{n}" for v, (a, n) in partial.items()) +
                   "; their accuracy and CNY per question cover a different question subset and CNY per 204 is withheld.")
    if unanswered:
        out.append(f"TRUNCATION: no answers for {', '.join(unanswered)} (plan truncated or stopped before them).")
    return out


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (kept apart from main so the defaults are testable without a run directory).
    命令行解析器（与 main 分开，使默认值无需 run 目录即可测试）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The parser.
        解析器。

    State effect / 状态影响：
        None.
        无。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=N_QUESTIONS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=0, help="smoke: only the first N timeline steps (0 = all) / 冒烟：只跑前 N 步（0 = 全部）")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of {VARIANTS} / 逗号分隔的变体")
    parser.add_argument("--no-answer", action="store_true", help="skip the API answer pass / 跳过 API 答题阶段")
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--max-answer-requests", type=int, default=MAX_ANSWER_REQUESTS)
    parser.add_argument("--cost-cap-usd", type=float, default=COST_CAP_USD)
    parser.add_argument("--workers", type=int, default=1,
                        help="answer-pass clients (one DeepSeekClient per worker, locked meter; `workers` requests are in flight at once and "
                             "an interrupt waits for the in-flight ones and records them before the ledger is written) / 答题并行客户端数"
                             "（每工作线程一个 DeepSeekClient，共用带锁台账；同时有 `workers` 个请求在飞行，中断时先等飞行中的请求完成"
                             "并记录再写台账）")  # fmt: skip
    parser.add_argument("--parity-check", action="store_true",
                        help="compare rows / card_1200 / stored_count with the run's recorded files / 与 run 记录文件比较 rows / card_1200 / stored_count")  # fmt: skip
    parser.add_argument("--tolerate-ingest-gaps", action="store_true",
                        help="continue past generate_lines misses / api_error subs / unresolved leaves instead of exit 4 / 摄入缺口时继续而非退出码 4")  # fmt: skip
    parser.add_argument("--deterministic-ids", action="store_true",
                        help="no-op kept for compatibility: counter atom ids are the default (see --random-ids) / 兼容用空操作："
                             "计数器原子 id 已是默认（见 --random-ids）")  # fmt: skip
    parser.add_argument("--random-ids", action="store_true",
                        help="mint atom ids with the library's secrets.token_hex instead of the default counter. The counter makes two "
                             "replays render the same within-slot card order; neither choice reproduces the recorded run's random ids, so "
                             "card_* / session_* contexts stay a re-derived baseline either way / 用库的 secrets.token_hex 而非默认计数器铸造"
                             "原子 id。计数器使两次回放的卡片槽内顺序相同；两种选择都不复现记录 run 的随机 id，card_* / session_* 上下文"
                             "无论如何都是重新导出的基线")  # fmt: skip
    return parser


def select_id_factory(args: argparse.Namespace) -> tuple[str, Callable[[str], str] | None]:
    """The atom-id mode of a parsed command line: ('deterministic', counter_id_factory()) unless --random-ids, then
    ('random', None) so build_hms keeps the library's secrets.token_hex ids.
    解析后命令行的原子 id 模式：默认 ('deterministic', counter_id_factory())；--random-ids 时为 ('random', None)，
    使 build_hms 保留库的 secrets.token_hex id。

    Input / 输入：
        args: parse_args result.
        args：parse_args 的结果。

    Output / 输出：
        (mode, id_factory or None).
        （模式，id_factory 或 None）。

    State effect / 状态影响：
        None (a fresh counter per call).
        无（每次调用一个新计数器）。
    """
    if args.random_ids:
        return "random", None
    return "deterministic", counter_id_factory()


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments (None -> sys.argv[1:]).
        argv：参数（None -> sys.argv[1:]）。

    Output / 输出：
        Exit code 0; 2 on a bad --variants / --workers, on --deterministic-ids together with --random-ids, or when the
        batch plan differs from the run's batches.json; 3 when --parity-check finds a mismatch; 4 when the ingest
        replay had a gap (generate_lines miss, api_error sub or unresolved leaf) and --tolerate-ingest-gaps is not set
        (nothing is answered then; only ingest_gaps.json and a minimal summary.json are left beside the retain files).
        退出码 0；--variants / --workers 非法、--deterministic-ids 与 --random-ids 同时给出、或批次计划与 run 的 batches.json
        不一致时为 2；--parity-check 发现不一致时为 3；摄入回放有缺口（generate_lines 未命中、api_error 子批次或未解析叶子）
        且未设 --tolerate-ingest-gaps 时为 4（此时不答题；retain 文件旁只留 ingest_gaps.json 与最小的 summary.json）。

    State effect / 状态影响：
        Removes an earlier run's outputs from --out, then writes the result directory; the answer pass makes network
        calls.
        先删除 --out 中先前 run 的输出，再写结果目录；答题阶段有网络调用。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    variants = tuple(s.strip() for s in args.variants.split(",") if s.strip())
    if not variants or any(v not in VARIANTS for v in variants):
        print(f"unknown variant in --variants {args.variants!r}; choose from {VARIANTS} / --variants 含未知变体")
        return 2
    if args.workers < 1:
        print("--workers must be >= 1 / --workers 须 >= 1")
        return 2
    if args.deterministic_ids and args.random_ids:
        print("--deterministic-ids and --random-ids conflict / --deterministic-ids 与 --random-ids 冲突")
        return 2
    atom_ids, id_factory = select_id_factory(args)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    work_root = out / "work"

    # ---- timeline identical to the real run / 与真实 run 相同的时间线
    questions = load_questions(args.data_root / QA_FILENAME)[: args.limit]
    windows: list[Window] = build_windows(load_transcripts(args.data_root))
    steps = schedule(questions, windows)
    batches = plan_batches(steps, args.batch_size)
    plan_recorded = json.loads((args.run / "batches.json").read_text(encoding="utf-8"))
    plan_here = [{"batch_id": b.batch_id, "position": b.position, "window_ids": b.window_ids} for b in batches]
    # Under --limit N < 204 only the positions the shorter timeline schedules exist; the batches at those positions
    # must still be the recorded ones (same ids, same window ids), so the recorded plan is compared over them.
    # --limit N < 204 时只有较短时间线安排的位置存在；这些位置上的批次仍须与记录相同（同 id、同窗口 id），因此记录的
    # 计划只在这些位置上比较。
    if plan_here != [b for b in plan_recorded if int(b["position"]) < len(steps)]:
        print("batch plan differs from the run's batches.json; aborting / 批次计划与 run 的 batches.json 不一致；中止")
        return 2
    recorded = load_recorded(args.run / "extract_raw_elite.jsonl")
    leaves_all = {b.batch_id: resolve_leaves(b, recorded) for b in batches}
    replayed = sum(1 for ls in leaves_all.values() for lf in ls if lf.row is not None)
    unresolved = [(lf.batch_id, lf.split) for ls in leaves_all.values() for lf in ls if lf.row is None]
    run_steps = steps[: args.steps] if args.steps > 0 else steps
    print(f"timeline: questions={len(questions)} windows={len(windows)} batches={len(batches)} steps_run={len(run_steps)} "
          f"recorded_lines={recorded.lines} leaves_replayable={replayed} unresolved={len(unresolved)} variants={list(variants)} "
          f"atom_ids={atom_ids}", flush=True)
    # Stale outputs of an earlier run into the same --out would otherwise sit next to this run's files; they go
    # before the ingest writes anything, so an interrupted ingest leaves no earlier summary / ledger behind either.
    # 否则先前写入同一 --out 的过期输出会与本次文件混在一起；在摄入写任何内容之前删除，摄入被中断时也不会残留先前的
    # 汇总 / 台账。
    remove_stale_outputs(out, keep_parity=args.parity_check, keep_gaps=False)

    # ---- ingest replay + context capture (local only) / 摄入回放 + 上下文捕获（仅本地）
    hook = ReaderContexts(variants=variants)
    t0 = time.perf_counter()
    res = run_arm(ARM, run_steps, batches, recorded, work_root=work_root, out=out, speaker_from_subject=True,
                  entities_from_mentions=True, contract="lines", context_hook=hook, id_factory=id_factory)  # fmt: skip
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    print(f"[{ARM}] done in {wall_ms / 1000:.1f} s; atoms={res['atoms']} replay={res['replay']} session_compiles={hook.session_compiles} "
          f"session_reuses={hook.session_reuses} session_ingest_skipped={hook.session_ingest_skipped} "
          f"session_query_fallback={hook.session_query_fallback} rows_mismatch={hook.rows_baseline_mismatch}", flush=True)
    # ---- ingest gaps are fatal / 摄入缺口即致命
    # A generate_lines miss (LLMError LLM_003), an api_error sub or an unresolved leaf leaves the store without that
    # batch's atoms for every later question; run_arm records it and continues, so the harness stops HERE, before any
    # paid call, unless --tolerate-ingest-gaps.
    # generate_lines 未命中（LLMError LLM_003）、api_error 子批次或未解析叶子会让此后每道题的库都缺该批次的原子；run_arm
    # 记录后继续，因此除非 --tolerate-ingest-gaps，脚本在此、在任何付费调用之前停止。
    gaps = ingest_gaps(res)
    if gaps["fatal"] and not args.tolerate_ingest_gaps:
        # parity.json was kept for --parity-check to overwrite; that never happens on this path, so it goes too, and a
        # minimal summary.json marks the stop, so no earlier run's summary / ledger / parity sits beside ingest_gaps.json.
        # parity.json 原是留给 --parity-check 覆盖的；本路径不会覆盖，故一并删除，并写一份标记停止的最小 summary.json，
        # 使先前 run 的汇总 / 台账 / 一致性文件不会与 ingest_gaps.json 并存。
        remove_stale_outputs(out, keep_parity=False, keep_gaps=True)
        (out / "ingest_gaps.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=1), encoding="utf-8")
        stopped = {
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(args.run), "out_dir": str(out),
            "questions": len(questions), "steps_run": len(run_steps), "batches": len(batches), "atoms": res["atoms"], "replay": res["replay"],
            "variants_run": list(variants), "stopped": True, "exit_code": 4,
            "reason": "ingest gaps (generate_lines miss / api_error sub / unresolved leaf); no answer pass, no variant files",
            "ingest_gaps": gaps, "variants": {}, "parity": None,
            "answer_pass": {"stopped": True, "reason": "ingest gaps", "requests": 0, "cost_usd": 0.0, "projected_usd": None,
                            "planned_requests": 0, "planned_per_variant": {v: 0 for v in variants}, "workers": args.workers},
            "api_cost_usd": 0.0,
            "config": {"arm": ARM, "atom_ids": atom_ids, "argv": list(argv) if argv is not None else sys.argv[1:],
                       "python": sys.version.split()[0], "package_versions": package_versions()},
        }  # fmt: skip
        (out / "summary.json").write_text(json.dumps(stopped, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ingest gaps: lines_misses={gaps['lines_misses']} api_error subs={len(gaps['api_error'])} missing leaves={len(gaps['missing'])}; "
              f"batches={gaps['batch_ids'][:20]}; aborting before the answer pass (see ingest_gaps.json) / 摄入缺口，答题前中止", flush=True)
        if not args.keep_db:
            shutil.rmtree(work_root, ignore_errors=True)
        return 4
    records = variant_records(variants, res["records"], hook)

    # ---- parity / 一致性
    parity = None
    if args.parity_check:
        parity = parity_check(args.run, hook, atoms=res["atoms"], complete=len(run_steps) == len(steps), retain=res["retain"])
        (out / "parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"parity: ok={parity['ok']} rows_mismatches={parity['rows']['mismatches']}/{parity['rows']['compared']} "
              f"(top_scores exact mismatches={parity['rows'].get('top_scores_exact_mismatches')}) "
              f"card_mismatches={parity['card_1200']['mismatches']}/{parity['card_1200']['compared']} "
              f"retain_mismatches={parity['retain']['mismatches']}/{parity['retain']['compared']} stored_count={parity['stored_count']}", flush=True)
        for label in ("rows", "card_1200", "retain"):
            for m in parity[label]["first"]:
                print(f"  {label} q{m['id']} {m['field']}: recorded={m['recorded']} replayed={m['replayed']}")

    # ---- answer pass / 答题阶段
    # The ledger is the cost record: calls.jsonl is flushed every 100 requests and written again in `finally`, and a
    # non-API exception or an interrupt still produces the per-question files and a summary marked stopped, then
    # re-raises after they are on disk.
    # 台账即费用记录：calls.jsonl 每 100 次请求刷一次并在 finally 再写一次；非 API 异常或中断仍会产出逐题文件与标记
    # stopped 的汇总，落盘后再重新抛出。
    meter = LockedMeter()
    qmap = {q.id: q for q in questions}
    interrupted: BaseException | None = None
    if args.no_answer:
        status: dict[str, Any] = {"stopped": True, "reason": "--no-answer", "requests": 0, "cost_usd": 0.0, "projected_usd": None,
                                  "planned_requests": 0, "planned_per_variant": {v: 0 for v in variants}, "workers": args.workers}  # fmt: skip
        # The empty ledger is written here too, so --out never carries the calls.jsonl of an earlier answered run.
        # 此处也写出空台账，使 --out 不会残留先前答题 run 的 calls.jsonl。
        meter.write_jsonl(out / "calls.jsonl")
    else:
        try:
            status = answer_pass(records, qmap, meter, max_requests=args.max_answer_requests, cost_cap=args.cost_cap_usd,
                                 workers=args.workers, flush=lambda _n: meter.write_jsonl(out / "calls.jsonl"))  # fmt: skip
        except BaseException as exc:  # noqa: BLE001 - re-raised after the ledger and summary are written
            interrupted = exc
            answered = sum(1 for v in variants for r in records[v] if "pred" in r)
            status = {"stopped": True, "reason": f"exception: {type(exc).__name__}: {str(exc)[:200]}", "requests": answered,
                      "cost_usd": round(meter.spent_usd(), 6), "projected_usd": None, "planned_requests": None,
                      "planned_per_variant": None, "workers": args.workers}  # fmt: skip
        finally:
            meter.write_jsonl(out / "calls.jsonl")

    # ---- files / 文件
    with (out / f"per_question_{ARM}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in res["records"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for v in variants:
        with (out / f"per_question_{v}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records[v]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary_variants = {v: variant_summary(v, records[v], meter) for v in variants}
    summary: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run),
        "out_dir": str(out),
        "questions": len(questions),
        "steps_run": len(run_steps),
        "windows": len(windows),
        "batches": len(batches),
        "recorded": {"lines": recorded.lines, "lines_truncated": recorded.lines_truncated, "batches_seen": len(recorded.batches_seen)},
        "replies_replayed": replayed,
        "replies_missing": len(unresolved),
        "atoms": res["atoms"],
        "atoms_added_sum": sum(r["added"] for r in res["retain"]),
        "replay": res["replay"],
        "ingest_wall_ms": wall_ms,
        "variants_run": list(variants),
        "session_memo": {"compiles": hook.session_compiles, "reuses": hook.session_reuses, "without_ingest": hook.session_without_ingest,
                         "ingest_skipped": hook.session_ingest_skipped, "query_fallback": hook.session_query_fallback,
                         "rows_baseline_mismatch": hook.rows_baseline_mismatch,
                         "rule": "recomputed only at a step that scheduled windows AND whose retain changed the store (>= 1 ADD or "
                                 "UPDATE event; NONE events and failed / missing / empty leaves do not count); query = text of the "
                                 "last window of the last leaf that produced an ADD or UPDATE (fallback: the step's last window, "
                                 "counted under query_fallback); as_of = that window's start instant; "
                                 f"budget_tokens = {SESSION_BUDGET}; reused verbatim otherwise (ingest_skipped = scheduled steps "
                                 "that changed nothing)"},  # fmt: skip
        "ingest_gaps": gaps,
        "variants": summary_variants,
        "parity": parity,
        "answer_pass": status,
        "api_cost_usd": round(sum(c.cost_usd or 0.0 for c in meter.calls), 6),
        "config": {
            "arm": ARM, "extraction_output": "lines", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "index": "sqlite",
            "top_k": 20, "threshold": 0.1, "batch_size": args.batch_size, "card_budgets": {k: (v or CARD_DEFAULT_BUDGET) for k, v in CARD_BUDGETS.items()},
            "session_budget": SESSION_BUDGET, "prompt_order": PROMPT_ORDER, "memories_first_variants": list(MEMORIES_FIRST_VARIANTS),
            "hybrid_rows_header": HYBRID_ROWS_HEADER, "prices_cny_per_m": PRICES_CNY_PER_M,
            "est_tokens_rule": "hms_elite.memory.utils.estimate_tokens = (utf8_len + 3) // 4 + 8",
            "projection_chars_per_token": PROJECTION_CHARS_PER_TOKEN, "max_answer_requests": args.max_answer_requests,
            "cost_cap_usd": args.cost_cap_usd, "workers": args.workers, "atom_ids": atom_ids,
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "python": sys.version.split()[0], "package_versions": package_versions(),
        },  # fmt: skip
        "method": (
            "Ingest = replay_ab.run_arm over the recorded lines replies (extract_raw_elite.jsonl) served verbatim through "
            "ReplayLLM.generate_lines under extraction_output=lines (zero API calls); the timeline, batch plan, add() and search() "
            "arguments are those of the streaming run. At each question, after that step's batches are retained and the baseline "
            "search (top_k=20, threshold=0.1, reference_date=query_time) has run, a hook captures: rows = real_arms.rows_to_context of "
            f"the projected rows ({baseline_status('rows', parity)}); rows_td = the same rows deduplicated (casefold + "
            "whitespace-collapsed exact duplicates dropped, then rows whose normalized text is a proper substring of another surviving "
            "row dropped) and stably sorted by hms.time_anchor (= the timestamp of the atom's add() slice, windows[0].start_iso, i.e. "
            "batch-start granularity, so rows of one slice keep search-rank order; fallback created_at; unknown last), same '- text' "
            "shape; rows_ts = rows_td with a '[YYYY-MM-DD HH:MM] ' prefix from the same slice timestamp (omitted when unknown); "
            f"card_1200 = hms.compile(question, filters={{user_id}}, as_of=query_time).to_text() ({baseline_status('card_1200', parity)}); "
            "card_800/600/400 = the same call with budget_tokens; session_1200 = compile(query=<text of the last window stored at this "
            "step>, as_of=<that window's start instant>, budget_tokens=1200).to_text(), memoized until the next step whose retain "
            "produces an ADD or UPDATE; session_prefix = the same memoized text answered with the Memories block FIRST and the question "
            "prompt LAST; card_prefix = card_1200's text (one compile per question, shared with card_1200 when both run) answered "
            "memories-first; hybrid_prefix = 'Memories:\\n<session card>\\n\\nRecent relevant memories:\\n<rows_ts text or (none)>"
            "\\n\\n<question prompt>', the session card being session_1200's memoized text (one compile per step, shared with it) and "
            "the second block rows_ts's text. Answer pass = one deepseek-flash call per (variant, question), variant-major in timeline order (plan "
            "truncated per variant), temperature 0, max_tokens 8, thinking disabled, priced in USD by accounting.Meter at the "
            "per-request tier and in CNY (off-peak and peak) from the measured cache_hit / cache_miss / output token counts, plus a "
            "cache-neutral CNY with every prompt token at the cache-miss price. hit for rows variants = hit@20 of the shared search; "
            "for card variants = any compile item's batch is gold; for hybrid_prefix = session-card hit OR kept rows_ts hit (both "
            "components reported). " + CARD_BASELINE_NOTE
        ),
        "caveats": [
            "The run directory stores no context text; --parity-check compares chars, returned batches, top scores (within "
            f"{SCORE_TOL}; the 4-dp recorded scores differ by ~1e-4 across fastembed builds), card used_tokens / items / mode, per-batch "
            "added / events and the final stored_count. This oracle is necessary, not sufficient: it cannot see a permutation of the "
            "same lines, and within-slot line order is tie-broken by atom id (compiler _render_order sorts by (time_anchor, item_id) "
            "and all atoms of one add() slice share time_anchor); the recorded run's ids were random (hms_elite/memory/storage.py "
            f"secrets.token_hex) and this replay's are {atom_ids} "
            f"({'counter_id_factory' if atom_ids == 'deterministic' else 'secrets.token_hex, --random-ids'}), so every card_* / "
            "session_* context is a re-derived baseline on an equivalent store, not a byte-for-byte reproduction, even when parity "
            "passes. Byte identity would need the recording to store card text and the exact embedding wheel set (package_versions "
            "records it for future runs).",
            "CNY figures are computed at both tiers from the measured token counts; the USD ledger uses the tier each call ran at.",
            "cache_hit_tokens is what the provider reports; nothing here controls DeepSeek's cache (granularity, TTL, warm-up), so the "
            "prefix effect of session_prefix is an observation, not a guaranteed property.",
            "--workers > 1 answers a chunk of questions concurrently, so consecutive same-step questions may not be sent back to back; "
            "run with --workers 1 when the cache-hit share of session_prefix / hybrid_prefix is the quantity of interest.",
            "Under --steps N the session memo, parity and stored_count cover only the first N steps.",
            "Cache-hit attribution: the eight question_first variants send the same first chunk (fixed header + question + choices) "
            "for one question up to eight times within one run, well inside DeepSeek's cache TTL, so a small non-zero cache-hit share "
            "on rows / card / session_1200 is cross-variant prefix reuse, not an ordering effect; session_1200 and session_prefix also "
            "share the whole Memories body. The ordering effect of session_prefix is bounded by the memo structure (the table's "
            "'session memo' column: recomputed / reused counts and the cache-hit share of first-at-step vs reused-card questions). "
            "card_prefix's prompt starts with its query card, which no other variant sends as a prefix, so its cache-hit share is the "
            "ordering effect alone; hybrid_prefix starts with the same 'Memories:\\n<session card>' chunk as session_prefix, so when "
            "both run its first-at-step questions can hit the prefix session_prefix sent earlier (variant-major order).",
            f"Pre-flight projection uses {PROJECTION_CHARS_PER_TOKEN} chars/token, below the recorded card run's measured minimum "
            "(2.504 chars/token over questions_elite_card.jsonl; median 2.728); rows_ts has no measurement. The running-ledger check "
            "still bounds spending to the cap plus one chunk.",
            "cache-neutral CNY (every prompt token at the cache-miss price) is the cross-variant cost column; the measured CNY of "
            "every variant after the first carries incidental hits on question heads sent by earlier variants.",
            *parity_caveats(parity),
            *truncation_caveats(summary_variants, status),
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    if not args.keep_db:
        shutil.rmtree(work_root, ignore_errors=True)
    for v in variants:
        s = summary["variants"][v]
        print(f"{v}: n={s['n']} hit={s['hit_batch']} chars_mean={s['context_chars_mean']} acc={s['answer_accuracy']} "
              f"({s['answer_correct']}/{s['answered_n']}) prompt_tokens_mean={s['prompt_tokens_mean']} cache_hit_share={s['cache_hit_share']} "
              f"cny_per_204={s['cny_per_204']} usd={s['answer_cost_usd']}")
    print(f"api_cost_usd={summary['api_cost_usd']} written to {out}")
    if interrupted is not None:
        raise interrupted
    if parity is not None and not parity["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
