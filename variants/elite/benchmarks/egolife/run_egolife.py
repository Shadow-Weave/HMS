"""EgoLife streaming retrieval benchmark: HMS Elite (three configs) versus official mem0ai (two stores), offline.
EgoLife 流式检索评测：HMS Elite（三种配置）对官方 mem0ai（两种向量库），全离线。
"""

# Purpose (EN): Drive the shared timeline through every arm, score evidence hits (gold = window containing
#               target_time; ±60 s neighbourhood reported as a secondary hit), run the shared deterministic reader
#               (a tie or all-zero score counts as wrong), and write per-question JSONL plus summary tables.
# 目的（中）：让共享时间线依次通过每个臂，计证据命中（gold = 包含 target_time 的窗口；±60 s 邻域作为次要命中报告）、
#            跑共享确定性阅读器（并列或全零计为错），写逐题 JSONL 与汇总表。

# Input (EN): CLI flags --arms --limit --repeats --out --data-root --keep-db --run-id --summarize; one arm per
#             process is supported: every process writes its arm file and re-merges the summary from disk.
# 输入（中）：命令行参数 --arms --limit --repeats --out --data-root --keep-db --run-id --summarize；支持每进程一臂：
#            每个进程写自己的臂文件并从磁盘重新合并汇总。

# Output (EN): <out>/{manifest.json, windows.jsonl, questions_<arm>.jsonl, arm_<arm>.json, summary.json, summary.md}.
# 输出（中）：<out>/{manifest.json, windows.jsonl, questions_<arm>.jsonl, arm_<arm>.json, summary.json, summary.md}。

# State effect (EN): Engine files live under <out>/work/<arm>/ and are deleted after measurement unless --keep-db.
# 状态影响（中）：引擎文件放在 <out>/work/<arm>/，度量完成后删除，除非 --keep-db。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arms import (  # noqa: E402
    ARM_NAMES,
    BANK,
    EMBEDDING_DIMS,
    THRESHOLD,
    TOP_K,
    EliteInferArm,
    embedding_identity_check,
    make_arm,
)
from loader import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    QA_FILENAME,
    Question,
    load_questions,
    load_transcripts,
    transcript_files,
)
from reader import EVIDENCE_CHARS, Reader  # noqa: E402
from timeline import GOLD_HALF_WIDTH_CS, GOLD_NEIGHBOURHOOD_CS, Step, schedule  # noqa: E402
from windows import (  # noqa: E402
    MAX_CHARS,
    MAX_SPAN_CS,
    Window,
    build_windows,
    clock_to_iso,
    window_ids_digest,
    write_windows_jsonl,
)

PROCESS_T0 = time.perf_counter()  # process start marker for per-arm wall clock / 进程起点，用于逐臂总耗时
RANDOM_BASELINE = 0.25
HIT_KS = (1, 5, 20)
FIRST_N = 50


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def unique_windows(rows: list[dict[str, Any]]) -> list[str]:
    """Returned window ids in rank order, first occurrence only (several atoms may share a window).
    按排名顺序的返回窗口 id，只取首次出现（多个原子可能同属一个窗口）。

    Input / 输入：
        rows: Arm.recall output.
        rows：Arm.recall 的输出。

    Output / 输出：
        Ordered unique ids (None ids dropped).
        有序去重 id（None 丢弃）。

    State effect / 状态影响：
        None.
        无。
    """
    seen: list[str] = []
    for row in rows:
        wid = row.get("window_id")
        if wid and wid not in seen:
            seen.append(wid)
    return seen


def _rank_of(uniq: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, wid in enumerate(uniq) if wid in gold), None)


def evaluate_step(
    arm_name: str,
    step: Step,
    rows: list[dict[str, Any]],
    reader: Reader,
    text_by_id: dict[str, str],
    retained_digest: str,
) -> dict[str, Any]:
    """Score one question for one arm: hit@k on containment gold, nb_hit@k on the ±60 s neighbourhood, reader
    prediction on returned texts and on window texts, and the oracle (reader on the gold texts) at this step.
    为一个臂的一道题计分：包含 gold 上的 hit@k、±60 s 邻域上的 nb_hit@k、阅读器在返回文本与窗口文本上的预测，
    以及本步的 oracle（阅读器读 gold 文本）。

    Input / 输入：
        arm_name: Arm label.
        arm_name：臂名。

        step: Timeline step (gold ids and flags).
        step：时间线步骤（gold id 与标记）。

        rows: Recall rows.
        rows：召回行。

        reader: Shared reader, already fed this step's new windows (IDF = retained prefix).
        reader：共享阅读器，已喂入本步新窗口（IDF = 已 retain 前缀）。

        text_by_id: window_id -> window text.
        text_by_id：window_id -> 窗口文本。

        retained_digest: sha256 of the arm's retained window ids after this step's retain.
        retained_digest：本步 retain 之后该臂已保留 window id 的 sha256。

    Output / 输出：
        Per-question record (no timing fields; the caller adds them). A reader/oracle 'correct' requires a valid
        (non-tie, non-zero) prediction: invalid counts as incorrect, as in their spec.
        逐题记录（不含计时字段；由调用方补充）。阅读器 / oracle 的 'correct' 要求有效（非并列、非全零）预测：
        无效计为错，与对方规范一致。

    State effect / 状态影响：
        None.
        无。
    """
    q = step.question
    gold = set(step.gold_ids)
    gold_wide = set(step.gold_wide_ids)
    uniq = unique_windows(rows)
    rank = _rank_of(uniq, gold)
    rank_wide = _rank_of(uniq, gold_wide)
    read_rows = reader.read(q.question, q.options, [r["text"] for r in rows])
    read_windows = reader.read(q.question, q.options, [text_by_id.get(wid, "") for wid in uniq])
    oracle = reader.read(q.question, q.options, reader.oracle_evidence(q))
    oracle_wide = reader.read(q.question, q.options, reader.oracle_evidence(q, GOLD_NEIGHBOURHOOD_CS))

    def correct(res: Any) -> bool:
        return (not res.invalid_tie) and res.pred == q.answer

    return {
        "arm": arm_name,
        "id": q.id,
        "order": q.order,
        "position": step.position,
        "type": q.type,
        "query_day": q.query_day,
        "query_time": q.query_hhmmsscc,
        "retained_count": step.retained_count,
        "retained_digest": retained_digest,
        "new_windows": len(step.new_windows),
        "reader_idf_windows": reader.observed,
        "gold_window_count": len(step.gold_ids),
        "gold_retained_count": len(step.gold_retained_ids),
        "gold_wide_window_count": len(step.gold_wide_ids),
        "gold_wide_retained_count": len(step.gold_wide_retained_ids),
        "excluded_future": step.excluded_future,
        "target_after_query": step.target_after_query,
        "returned_count": len(rows),
        "returned_unique_windows": len(uniq),
        "rank": rank,
        **{f"hit@{k}": bool(rank is not None and rank <= k) for k in HIT_KS},
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "rank_wide": rank_wide,
        **{f"nb_hit@{k}": bool(rank_wide is not None and rank_wide <= k) for k in HIT_KS},
        "reciprocal_rank_wide": (1.0 / rank_wide) if rank_wide else 0.0,
        "top_scores": [round(r["score"], 4) for r in rows[:5]],
        "top_windows": uniq[:TOP_K],
        "reader_pred": read_rows.pred,
        "reader_scores": {k: round(v, 3) for k, v in read_rows.scores.items()},
        "reader_invalid_tie": read_rows.invalid_tie,
        "reader_evidence_chars": read_rows.evidence_chars,
        "reader_window_pred": read_windows.pred,
        "reader_window_invalid_tie": read_windows.invalid_tie,
        "gold": q.answer,
        "reader_correct": correct(read_rows),
        "reader_window_correct": correct(read_windows),
        "oracle_pred": oracle.pred,
        "oracle_correct": correct(oracle),
        "oracle_invalid_tie": oracle.invalid_tie,
        "oracle_evidence_chars": oracle.evidence_chars,
        "oracle_wide_pred": oracle_wide.pred,
        "oracle_wide_correct": correct(oracle_wide),
        "oracle_wide_invalid_tie": oracle_wide.invalid_tie,
    }


def run_arm(
    name: str,
    steps: list[Step],
    reader: Reader,
    text_by_id: dict[str, str],
    *,
    work_root: Path,
    repeats: int,
    out_path: Path,
    keep_db: bool,
) -> dict[str, Any]:
    """Stream every step through one arm and write its questions JSONL.
    让每个步骤流过一个臂并写出其逐题 JSONL。

    Input / 输入：
        name: Arm name.
        name：臂名。

        steps: Shared timeline.
        steps：共享时间线。

        reader: A fresh Reader (empty corpus); fed step.new_windows before each question.
        reader：新的 Reader（空语料）；每题之前喂入 step.new_windows。

        text_by_id: window_id -> text.
        text_by_id：window_id -> 文本。

        work_root: Parent of the arm working directory.
        work_root：臂工作目录的父目录。

        repeats: Search timing repetitions (first call's rows and latency are the primary figures; all repeats
                 must return the same ids; the median over repeats is kept as the warmed figure).
        repeats：检索计时重复次数（第一次调用的行与延迟是主数字；所有重复必须返回相同 id；重复的中位数作为预热
                 后的数字保留）。

        out_path: questions_<arm>.jsonl.
        out_path：questions_<arm>.jsonl。

        keep_db: Keep engine files after measuring their size.
        keep_db：度量尺寸后保留引擎文件。

    Output / 输出：
        {'records': [...], 'stats': arm.stats() after close, 'retained_digest', 'retained_count_by_position',
         'retained_digest_by_position', 'search_repeat_mismatch': n}.
        {'records': [...], 'stats': close 后的 arm.stats(), 'retained_digest', 'retained_count_by_position',
         'retained_digest_by_position', 'search_repeat_mismatch': n}。

    State effect / 状态影响：
        Writes out_path; creates and (unless keep_db) deletes work_root/<name>.
        写 out_path；创建并（除非 keep_db）删除 work_root/<name>。
    """
    arm = make_arm(name, work_root)
    records: list[dict[str, Any]] = []
    mismatches = 0
    retained_by_position: list[int] = []
    digest_by_position: list[str] = []
    t_arm = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for step in steps:
            q = step.question
            query_iso = clock_to_iso(q.query_day, q.query_cs)
            retain_ms = arm.retain(step.new_windows)
            if len(arm.retained_ids) != step.retained_count:
                raise RuntimeError(f"{name}: retained {len(arm.retained_ids)} != timeline {step.retained_count}")
            retained_by_position.append(len(arm.retained_ids))
            step_digest = window_ids_digest(arm.retained_ids)
            digest_by_position.append(step_digest)
            reader.observe(step.new_windows)  # IDF corpus = retained prefix, same for every arm / IDF 语料 = 已 retain 前缀
            timings: list[float] = []
            rows: list[dict[str, Any]] = []
            first_ids: list[str] | None = None
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                got = arm.recall(q.question, query_iso)
                timings.append((time.perf_counter() - t0) * 1000.0)
                ids = [r.get("memory_id") for r in got]
                if first_ids is None:
                    first_ids, rows = ids, got
                elif ids != first_ids:
                    mismatches += 1
            record = evaluate_step(name, step, rows, reader, text_by_id, step_digest)
            record.update(
                {
                    "retain_ms": round(sum(retain_ms), 3),
                    "retain_ms_per_window": round(sum(retain_ms) / len(retain_ms), 3) if retain_ms else None,
                    "search_ms_first": round(timings[0], 3),
                    "search_ms_median": round(statistics.median(timings), 3),
                    "search_ms_all": [round(t, 3) for t in timings],
                }
            )
            if isinstance(arm, EliteInferArm):
                record["compile"] = arm.compile_card(q.question, query_iso)
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    wall_ms = (time.perf_counter() - t_arm) * 1000.0
    arm.close()
    stats = arm.stats()
    stats["arm_wall_ms"] = round(wall_ms, 1)
    if isinstance(arm, EliteInferArm):
        cards = arm.compile_stats
        stats["compile"] = {
            "count": len(cards),
            "card_chars_median": statistics.median(c["card_chars"] for c in cards) if cards else 0,
            "card_chars_mean": round(statistics.fmean(c["card_chars"] for c in cards), 1) if cards else 0,
            "used_tokens_median": statistics.median(c["used_tokens"] for c in cards) if cards else 0,
            "items_median": statistics.median(c["items"] for c in cards) if cards else 0,
            "compile_ms_median": round(statistics.median(c["compile_ms"] for c in cards), 3) if cards else 0,
        }
    if not keep_db:
        shutil.rmtree(work_root / name, ignore_errors=True)
    return {
        "records": records,
        "stats": stats,
        "retained_digest": window_ids_digest(arm.retained_ids),
        "retained_total": len(arm.retained_ids),
        "retained_count_by_position": retained_by_position,
        "retained_digest_by_position": digest_by_position,
        "search_repeat_mismatch": mismatches,
    }


def aggregate(records: list[dict[str, Any]], *, denominator: int | None = None) -> dict[str, Any]:
    """hit@k / MRR (containment gold), nb_hit@k / nb_mrr (±60 s), reader / oracle (invalid = wrong, plus
    valid-only rates) over a record subset with a fixed or exclusion-adjusted denominator.
    在记录子集上求 hit@k / MRR（包含 gold）、nb_hit@k / nb_mrr（±60 s）、阅读器 / oracle（无效计错，另给仅有效
    子集的比率），分母固定或按剔除调整。

    Input / 输入：
        records: Per-question records of one arm.
        records：一个臂的逐题记录。

        denominator: Fixed denominator (first-50 rule) or None -> len(records) - excluded_future.
        denominator：固定分母（first-50 规则）或 None -> len(records) − excluded_future。

    Output / 输出：
        Metrics dict with counts and rates (rates as fractions).
        含计数与比率（比率为小数）的指标字典。

    State effect / 状态影响：
        None.
        无。
    """
    excluded = sum(1 for r in records if r["excluded_future"])
    scored = [r for r in records if not r["excluded_future"]]
    denom = denominator if denominator is not None else len(scored)
    denom = max(denom, 0)

    def rate(count: float) -> float | None:
        return round(count / denom, 4) if denom else None

    out: dict[str, Any] = {
        "questions": len(records),
        "excluded_future": excluded,
        "denominator": denom,
        "random_baseline": RANDOM_BASELINE,
    }
    for k in HIT_KS:
        out[f"hit@{k}"] = rate(sum(1 for r in scored if r[f"hit@{k}"]))
    out["mrr"] = rate(sum(r["reciprocal_rank"] for r in scored))
    for k in HIT_KS:
        out[f"nb_hit@{k}"] = rate(sum(1 for r in scored if r[f"nb_hit@{k}"]))
    out["nb_mrr"] = rate(sum(r["reciprocal_rank_wide"] for r in scored))

    def valid_only(correct_key: str, tie_key: str) -> tuple[int, float | None]:
        valid = [r for r in scored if not r[tie_key]]
        return len(valid), (round(sum(1 for r in valid if r[correct_key]) / len(valid), 4) if valid else None)

    out["reader_acc"] = rate(sum(1 for r in scored if r["reader_correct"]))  # invalid (tie / all-zero) = wrong / 无效（并列 / 全零）记为答错
    out["reader_valid_n"], out["reader_acc_valid_only"] = valid_only("reader_correct", "reader_invalid_tie")
    out["reader_window_acc"] = rate(sum(1 for r in scored if r["reader_window_correct"]))
    out["reader_invalid_ties"] = sum(1 for r in scored if r["reader_invalid_tie"])
    out["oracle_acc"] = rate(sum(1 for r in scored if r["oracle_correct"]))
    out["oracle_valid_n"], out["oracle_acc_valid_only"] = valid_only("oracle_correct", "oracle_invalid_tie")
    out["oracle_invalid_ties"] = sum(1 for r in scored if r["oracle_invalid_tie"])
    out["oracle_wide_acc"] = rate(sum(1 for r in scored if r["oracle_wide_correct"]))
    out["oracle_wide_invalid_ties"] = sum(1 for r in scored if r["oracle_wide_invalid_tie"])
    out["returned_unique_windows_mean"] = round(statistics.fmean(r["returned_unique_windows"] for r in records), 2) if records else 0
    return out


def timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain totals and search latency percentiles for one arm.
    一个臂的 retain 总量与检索延迟分位数。

    Input / 输入：
        records: Per-question records.
        records：逐题记录。

    Output / 输出：
        Dict of milliseconds and counts.
        毫秒与计数字典。

    State effect / 状态影响：
        None.
        无。
    """
    search = [r["search_ms_median"] for r in records]
    first = [r.get("search_ms_first", r["search_ms_all"][0]) for r in records]
    per_window = [r["retain_ms_per_window"] for r in records if r["retain_ms_per_window"] is not None]
    total_ms = sum(r["retain_ms"] for r in records)
    n_windows = sum(r["new_windows"] for r in records)
    # Per-window cost by schedule quartile: mem0's FAISS _save rewrites the whole store on every add (O(n)).
    # 按调度四分位的每窗口成本：mem0 的 FAISS _save 每次 add 都重写整个库（O(n)）。
    ordered = sorted(records, key=lambda r: r["position"])
    quartiles: list[float | None] = []
    for i in range(4):
        part = ordered[i * len(ordered) // 4 : (i + 1) * len(ordered) // 4]
        ms = sum(r["retain_ms"] for r in part)
        n = sum(r["new_windows"] for r in part)
        quartiles.append(round(ms / n, 3) if n else None)
    return {
        "retain_total_ms": round(total_ms, 1),
        "retain_windows": n_windows,
        "retain_ms_per_window_mean": round(total_ms / n_windows, 3) if n_windows else None,
        "retain_ms_per_window_median_over_questions": round(statistics.median(per_window), 3) if per_window else None,
        "retain_ms_per_window_by_quartile": quartiles,
        "search_first_ms_median": round(statistics.median(first), 3) if first else None,
        "search_first_ms_p90": round(_p(first, 0.9), 3) if first else None,
        "search_first_ms_max": round(max(first), 3) if first else None,
        "search_first_ms_sum": round(sum(first), 1),
        "search_ms_median": round(statistics.median(search), 3) if search else None,
        "search_ms_p90": round(_p(search, 0.9), 3) if search else None,
        "search_ms_max": round(max(search), 3) if search else None,
        "engine_only_ms": round(total_ms + sum(first), 1),
    }


def markdown_tables(summary: dict[str, Any]) -> str:
    """Render the summary as Markdown tables only (no prose).
    只把汇总渲染为 Markdown 表格（无散文）。

    Input / 输入：
        summary: The summary.json structure.
        summary：summary.json 的结构。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """
    arms = list(summary["arms"])
    lines: list[str] = []

    def pct(v: Any) -> str:
        return "-" if v is None else f"{100 * v:.1f}%"

    def fmt(v: Any) -> str:
        return "-" if v is None else (f"{v:,}" if isinstance(v, int) else (f"{v:.2f}" if isinstance(v, float) else str(v)))

    for scope in ("all", "first50"):
        denominators = ", ".join(f"{a}={summary['arms'][a][scope]['denominator']}" for a in arms)
        lines.append(f"## {scope} (denominator: {denominators}; gold = window containing target_time; invalid reader/oracle = wrong)")
        lines.append("")
        lines.append("| arm | hit@1 | hit@5 | hit@20 | MRR | nb-hit@20 (±60 s) | reader acc | reader acc valid-only (n) | reader acc (window text) | ties | oracle acc | oracle valid-only (n) | oracle acc (±60 s) | random |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            m = summary["arms"][a][scope]
            lines.append(
                f"| {a} | {pct(m['hit@1'])} | {pct(m['hit@5'])} | {pct(m['hit@20'])} | {fmt(m['mrr'])} | {pct(m['nb_hit@20'])} | "
                f"{pct(m['reader_acc'])} | {pct(m['reader_acc_valid_only'])} ({m['reader_valid_n']}) | {pct(m['reader_window_acc'])} | "
                f"{m['reader_invalid_ties']} | {pct(m['oracle_acc'])} | {pct(m['oracle_acc_valid_only'])} ({m['oracle_valid_n']}) | "
                f"{pct(m['oracle_wide_acc'])} | {pct(m['random_baseline'])} |"
            )
        lines.append("")
    lines.append("## by type (all; hit@5 / hit@20 / MRR / reader acc / oracle acc)")
    lines.append("")
    types = sorted({t for a in arms for t in summary["arms"][a]["by_type"]})
    lines.append("| type | n | " + " | ".join(arms) + " |")
    lines.append("|---|---|" + "---|" * len(arms))
    for t in types:
        cells = []
        n = None
        for a in arms:
            m = summary["arms"][a]["by_type"].get(t)
            if m is None:
                cells.append("-")
                continue
            n = m["denominator"]
            cells.append(f"{pct(m['hit@5'])} / {pct(m['hit@20'])} / {fmt(m['mrr'])} / {pct(m['reader_acc'])} / {pct(m['oracle_acc'])}")
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## cost")
    lines.append("")
    lines.append(
        "retain ms/window: mean = total / windows; median = median over questions of the per-question mean; quartiles = "
        "mean per-window ms in schedule quartiles Q1..Q4 (store size grows along the stream). search first = first call after "
        "the retain (the streaming figure); search med3 = median of 3 back-to-back identical calls (warmed). "
        "engine-only ms = retain total + sum of first-call search. arm loop wall ms includes 3x search, reader, oracle and "
        "compile (elite_infer). db bytes = files on disk after close (see db files)."
    )
    lines.append("")
    sv_proc = summary["shared"].get("arm_process_ms", {})
    lines.append("| arm | windows | stored rows | retain total ms | retain ms/window mean | retain ms/window median | retain ms/window Q1..Q4 | search first ms median | search first ms p90 | search med3 ms median | search med3 ms p90 | engine-only ms | embedding calls | llm calls | llm tokens (est.) | db bytes after close | db files | arm loop wall ms | process ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        s = summary["arms"][a]["stats"]
        t = summary["arms"][a]["timing"]
        tokens = s.get("llm_tokens", {}).get("total", 0) if isinstance(s.get("llm_tokens"), dict) else 0
        quart = " / ".join(fmt(v) for v in t["retain_ms_per_window_by_quartile"])
        files = "; ".join(f"{k} {v:,}" for k, v in (s.get("db_files") or {}).items())
        lines.append(
            f"| {a} | {t['retain_windows']} | {s['stored_count']} | {fmt(t['retain_total_ms'])} | {fmt(t['retain_ms_per_window_mean'])} | "
            f"{fmt(t['retain_ms_per_window_median_over_questions'])} | {quart} | {fmt(t['search_first_ms_median'])} | {fmt(t['search_first_ms_p90'])} | "
            f"{fmt(t['search_ms_median'])} | {fmt(t['search_ms_p90'])} | {fmt(t['engine_only_ms'])} | {s['embedding_calls']} | {s['llm_calls']} | {tokens} | "
            f"{s['db_bytes']:,} | {files} | {fmt(s['arm_wall_ms'])} | {fmt(sv_proc.get(a))} |"
        )
    lines.append("")
    if any("compile" in summary["arms"][a]["stats"] for a in arms):
        lines.append("## elite_infer compile()")
        lines.append("")
        lines.append("| arm | count | card chars median | card chars mean | used_tokens median | items median | compile ms median |")
        lines.append("|---|---|---|---|---|---|---|")
        for a in arms:
            c = summary["arms"][a]["stats"].get("compile")
            if c:
                lines.append(
                    f"| {a} | {c['count']} | {c['card_chars_median']} | {c['card_chars_mean']} | {c['used_tokens_median']} | "
                    f"{c['items_median']} | {c['compile_ms_median']} |"
                )
        lines.append("")
    lines.append("## shared-variable evidence")
    lines.append("")
    lines.append("| check | value |")
    lines.append("|---|---|")
    sv = summary["shared"]
    for key, value in sv.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments (None -> sys.argv[1:]).
        argv：参数（None -> sys.argv[1:]）。

    Output / 输出：
        Exit code 0; non-zero on a failed consistency check.
        退出码 0；一致性检查失败时非零。

    State effect / 状态影响：
        Writes the result directory.
        写结果目录。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default=",".join(ARM_NAMES), help="comma-separated arm names")
    parser.add_argument("--limit", type=int, default=204, help="first N questions in original array order")
    parser.add_argument("--repeats", type=int, default=3, help="search timing repetitions per question")
    parser.add_argument("--out", type=Path, default=None, help="result directory (default results/<run_id>)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--keep-db", action="store_true", help="keep engine files under <out>/work/")
    parser.add_argument(
        "--summarize", action="store_true",
        help="run no arm; rebuild summary.json/md from the arm_<name>.json + questions_<name>.jsonl already in <out>",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (HERE / "results" / run_id)
    out.mkdir(parents=True, exist_ok=True)
    work_root = out / "work"
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.summarize:
        manifest_path = out / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--summarize needs {manifest_path}")
        return write_summary(out, run_id, json.loads(manifest_path.read_text(encoding="utf-8")))

    qa_path = args.data_root / QA_FILENAME
    all_questions = load_questions(qa_path)
    if args.limit <= 0 or args.limit > len(all_questions):
        raise SystemExit(f"--limit must be in 1..{len(all_questions)}")
    questions: list[Question] = all_questions[: args.limit]
    transcripts = load_transcripts(args.data_root)
    windows: list[Window] = build_windows(transcripts)
    windows_sha = write_windows_jsonl(windows, out / "windows.jsonl")
    text_by_id = {w.window_id: w.text for w in windows}
    steps = schedule(questions, windows)
    probe_texts = [w.text for w in windows[:20]]
    embed_check = embedding_identity_check(probe_texts)
    if not embed_check["equal"]:
        raise SystemExit("shared embedder mismatch between the mem0 shell and hms_elite HashEmbedder")

    manifest = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qa_file": str(qa_path),
        "qa_sha256": _sha256_file(qa_path),
        "questions_in_file": len(all_questions),
        "limit": args.limit,
        "question_ids": [q.id for q in questions],
        "transcript_files": len(transcript_files(args.data_root)),
        "cues": sum(len(c) for c in transcripts.values()),
        "windows": len(windows),
        "windows_sha256": windows_sha,
        "window_rule": {"max_chars": MAX_CHARS, "max_span_s": MAX_SPAN_CS / 100, "text": "english lines only"},
        "gold_rule": {
            "primary": f"window contains target_time (start <= t <= end); half width {GOLD_HALF_WIDTH_CS / 100:g} s",
            "secondary_nb": f"window overlaps [t - {GOLD_NEIGHBOURHOOD_CS / 100:g} s, t + {GOLD_NEIGHBOURHOOD_CS / 100:g} s] (nb_hit@k)",
        },
        "cutoff_rule": (
            "same day: window visible iff window.end_cs <= query_time (conservative deviation from their cue.start <= "
            "query_time: a window straddling the query instant is withheld for at most 60 s); earlier days: whole day "
            "(their 23595999 sentinel); later days: invisible"
        ),
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "threshold_note": (
            "threshold gates the semantic score on mem0 (both stores) and elite_matched; on elite_full / elite_infer it "
            "gates the fused score, whose recency + support floor exceeds 0.1, so it removes no row there"
        ),
        "embedding": {"algorithm": "hms_elite.embeddings.hash.HashEmbedder", "dims": EMBEDDING_DIMS, "probe": embed_check},
        "reader": {
            "evidence_chars": EVIDENCE_CHARS,
            "rule": "option-token count x IDF + 0.5 x same-window question-token IDF; tie or all-zero -> invalid (counted wrong)",
            "idf": "document frequencies over the windows retained so far at each step (no future windows)",
            "oracle": "reader on the gold-window texts visible at the step (primary: containment; also ±60 s)",
        },
        "bank": BANK,
        "arms": arm_names,
        "repeats": args.repeats,
        "excluded_future": sum(1 for s in steps if s.excluded_future),
        "target_after_query": sum(1 for s in steps if s.target_after_query),
        "schedule": [
            {"position": s.position, "id": s.question.id, "day": s.question.query_day, "time": s.question.query_hhmmsscc, "new": len(s.new_windows), "retained": s.retained_count}
            for s in steps
        ],
    }
    manifest = merge_manifest(out / "manifest.json", manifest, arm_names)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in arm_names:
        print(f"[{run_id}] arm {name}: {len(steps)} questions, {len(windows)} windows", flush=True)
        t_proc = time.perf_counter()
        result = run_arm(
            name, steps, Reader(), text_by_id,
            work_root=work_root, repeats=args.repeats, out_path=out / f"questions_{name}.jsonl", keep_db=args.keep_db,
        )
        t = timing_summary(result["records"])
        s = result["stats"]
        arm_file = {k: v for k, v in result.items() if k != "records"}
        arm_file.update(
            {
                "arm": name,
                "run_id": run_id,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "arm_process_ms": round((time.perf_counter() - PROCESS_T0) * 1000.0, 1),
                "arm_loop_ms": round((time.perf_counter() - t_proc) * 1000.0, 1),
            }
        )
        (out / f"arm_{name}.json").write_text(json.dumps(arm_file, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            f"  done: hit@20={aggregate(result['records'])['hit@20']} retain={t['retain_total_ms']} ms "
            f"search_first_med={t['search_first_ms_median']} ms emb={s['embedding_calls']} llm={s['llm_calls']} db={s['db_bytes']:,} B "
            f"wall={s['arm_wall_ms']} ms process={arm_file['arm_process_ms']} ms",
            flush=True,
        )
    if not args.keep_db:
        shutil.rmtree(work_root, ignore_errors=True)
    return write_summary(out, run_id, manifest)


def merge_manifest(path: Path, manifest: dict[str, Any], arm_names: list[str]) -> dict[str, Any]:
    """Reconcile a fresh manifest with one already in the result directory (arms run as separate processes).
    把新算出的 manifest 与结果目录里已有的合并（各臂作为独立进程运行）。

    Input / 输入：
        path: manifest.json location.
        path：manifest.json 位置。

        manifest: Manifest computed by this process.
        manifest：本进程算出的 manifest。

        arm_names: Arms this process will run.
        arm_names：本进程将运行的臂。

    Output / 输出：
        Manifest whose 'arms' is the union (ARM_NAMES order) and whose 'processes' lists every launch; the
        shared fields (qa_sha256, question_ids, windows_sha256, top_k, threshold, repeats) must match byte for byte.
        'arms' 取并集（按 ARM_NAMES 顺序）、'processes' 列出每次启动的 manifest；共享字段（qa_sha256、question_ids、
        windows_sha256、top_k、threshold、repeats）必须逐字节一致。

    State effect / 状态影响：
        None (the caller writes the file).
        无（由调用方写文件）。

    Raises / 异常：
        SystemExit when the existing directory was built from different inputs.
        已有目录由不同输入构建时抛 SystemExit。
    """
    launch = {"arms": arm_names, "started_utc": manifest["started_utc"], "pid": os.getpid()}
    if not path.exists():
        manifest["processes"] = [launch]
        return manifest
    prev = json.loads(path.read_text(encoding="utf-8"))
    for key in ("qa_sha256", "question_ids", "windows_sha256", "top_k", "threshold", "repeats", "limit"):
        if prev.get(key) != manifest[key]:
            raise SystemExit(f"manifest mismatch on {key}: {path.parent} was built from different inputs")
    if prev["embedding"]["probe"]["sha256"] != manifest["embedding"]["probe"]["sha256"]:
        raise SystemExit(f"manifest mismatch on embedding probe: {path.parent}")
    manifest["started_utc"] = prev["started_utc"]
    manifest["arms"] = [a for a in ARM_NAMES if a in set(prev.get("arms", [])) | set(arm_names)]
    manifest["processes"] = prev.get("processes", []) + [launch]
    return manifest


def _keyword_channel(extra: dict[str, Any]) -> bool:
    """Whether an arm had a lexical channel: mem0 -> store keyword_search present; elite -> w_keyword > 0.
    该臂是否有词法通道：mem0 -> 向量库有 keyword_search；elite -> w_keyword > 0。

    Input / 输入：
        extra: stats['extra'] of the arm.
        extra：该臂的 stats['extra']。

    Output / 输出：
        Boolean.
        布尔。

    State effect / 状态影响：
        None.
        无。
    """
    if "keyword_search" in extra:
        return bool(extra["keyword_search"])
    return float((extra.get("scoring") or {}).get("w_keyword", 0.0)) > 0.0


def load_arm_results(out: Path) -> dict[str, dict[str, Any]]:
    """Read every arm_<name>.json + questions_<name>.jsonl pair present in the result directory.
    读取结果目录中每对 arm_<name>.json + questions_<name>.jsonl。

    Input / 输入：
        out: Result directory.
        out：结果目录。

    Output / 输出：
        {arm: {'records': [...], 'stats', 'retained_digest', 'retained_count_by_position', 'search_repeat_mismatch',
        ...}} in ARM_NAMES order, arms without both files skipped.
        {arm: {...}} 按 ARM_NAMES 顺序，缺任一文件的臂跳过。

    State effect / 状态影响：
        None.
        无。
    """
    results: dict[str, dict[str, Any]] = {}
    for name in ARM_NAMES:
        arm_path = out / f"arm_{name}.json"
        q_path = out / f"questions_{name}.jsonl"
        if not (arm_path.exists() and q_path.exists()):
            continue
        r = json.loads(arm_path.read_text(encoding="utf-8"))
        with q_path.open("r", encoding="utf-8") as fh:
            r["records"] = [json.loads(line) for line in fh if line.strip()]
        results[name] = r
    return results


def write_summary(out: Path, run_id: str, manifest: dict[str, Any]) -> int:
    """Assemble summary.json / summary.md from the per-arm files on disk and run the shared-variable checks.
    从磁盘上的逐臂文件组装 summary.json / summary.md 并做共享变量检查。

    Input / 输入：
        out: Result directory holding manifest.json and arm files.
        out：含 manifest.json 与臂文件的结果目录。

        run_id: Label for log lines.
        run_id：日志标签。

        manifest: Manifest dict (its 'schedule' supplies the timeline retained counts).
        manifest：manifest 字典（其 'schedule' 提供时间线保留计数）。

    Output / 输出：
        Exit code 0 when every consistency check passes, else 2.
        全部一致性检查通过返回 0，否则 2。

    State effect / 状态影响：
        Writes summary.json and summary.md.
        写 summary.json 与 summary.md。
    """
    results = load_arm_results(out)
    if not results:
        raise SystemExit(f"no arm files in {out}")
    embed_check = manifest["embedding"]["probe"]
    timeline_counts = [s["retained"] for s in manifest["schedule"]]
    summary: dict[str, Any] = {"run_id": run_id, "manifest": {k: v for k, v in manifest.items() if k != "schedule"}, "arms": {}}
    digests = {name: r["retained_digest"] for name, r in results.items()}
    retained_seq = {name: r["retained_count_by_position"] for name, r in results.items()}
    digest_seq = {name: r.get("retained_digest_by_position", []) for name, r in results.items()}
    summary["shared"] = {
        "windows_sha256": manifest["windows_sha256"],
        "window_count": manifest["windows"],
        "retained_window_set_digest_identical": len(set(digests.values())) <= 1,
        "retained_window_set_digests": digests,
        "retained_count_per_question_identical": len({tuple(v) for v in retained_seq.values()}) <= 1,
        "retained_digest_per_question_identical": len({tuple(v) for v in digest_seq.values()}) <= 1
        and all(len(v) == len(timeline_counts) for v in digest_seq.values()),
        "timeline_retained_counts_match": all(list(v) == timeline_counts for v in retained_seq.values()),
        "embedding_probe_equal": embed_check["equal"],
        "embedding_probe_sha256": embed_check["sha256"],
        "search_repeat_mismatches": {name: r["search_repeat_mismatch"] for name, r in results.items()},
        "control_arm_llm_calls_zero": all(
            results[n]["stats"]["llm_calls"] == 0 for n in results if n != "elite_infer"
        ),
        "keyword_channel": {name: _keyword_channel(r["stats"].get("extra", {})) for name, r in results.items()},
        "gold_rule": manifest.get("gold_rule"),
        "cutoff_rule": manifest.get("cutoff_rule"),
        "threshold_note": manifest.get("threshold_note"),
        "arm_pids": {name: r.get("pid") for name, r in results.items()},
        "arm_process_ms": {name: r.get("arm_process_ms") for name, r in results.items()},
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "reader": "shared deterministic reader (reader.Reader)",
    }
    for name, r in results.items():
        recs = r["records"]
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            by_type[rec["type"]].append(rec)
        first = [rec for rec in recs if rec["order"] < FIRST_N]
        summary["arms"][name] = {
            "all": aggregate(recs),
            "first50": aggregate(first, denominator=min(FIRST_N, manifest["limit"])),
            "by_type": {t: aggregate(v) for t, v in sorted(by_type.items())},
            "timing": timing_summary(recs),
            "stats": r["stats"],
        }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(markdown_tables(summary), encoding="utf-8")
    ok = (
        summary["shared"]["retained_window_set_digest_identical"]
        and summary["shared"]["retained_count_per_question_identical"]
        and summary["shared"]["retained_digest_per_question_identical"]
        and summary["shared"]["timeline_retained_counts_match"]
        and summary["shared"]["embedding_probe_equal"]
        and summary["shared"]["control_arm_llm_calls_zero"]
    )
    print(f"[{run_id}] wrote {out} arms={list(results)} consistency={'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
