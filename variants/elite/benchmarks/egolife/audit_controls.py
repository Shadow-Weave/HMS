"""Audit 1 (offline): are the four EgoLife arms actually controlled? Re-derives every shared variable from disk.
审计一（离线）：EgoLife 四臂对照是否成立？从磁盘重新推导每个共享变量。
"""

# Purpose (EN): Independent checks over results/<run>/ plus a small stepwise rebuild of mem0_faiss / elite_matched /
#               elite_full with explain=True: retained sets, future leakage, embedding bits, gold definition,
#               reader isolation, threshold semantics. Writes one JSON and one Markdown table; no prose.
# 目的（中）：对 results/<run>/ 做独立检查，并对 mem0_faiss / elite_matched / elite_full 做一次带 explain=True 的
#            小规模逐步重建：保留集合、未来泄漏、嵌入比特、gold 定义、阅读器隔离、阈值语义。只写一个 JSON 和一个
#            Markdown 表；无散文。

# Input (EN): --run (result directory, default results/full), --rebuild-steps (default 52), --data-root.
# 输入（中）：--run（结果目录，默认 results/full）、--rebuild-steps（默认 52）、--data-root。

# Output (EN): <run>/../audit1/controls.json and controls.md.
# 输出（中）：<run>/../audit1/controls.json 与 controls.md。

# State effect (EN): Builds throw-away engines under <audit>/work/ and deletes them.
# 状态影响（中）：在 <audit>/work/ 下构建一次性引擎并删除。

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arms import (  # noqa: E402
    BANK,
    THRESHOLD,
    TOP_K,
    EliteFullArm,
    EliteMatchedArm,
    Mem0FaissArm,
    SharedHashEmbedding,
    shared_hash_embedder,
)
from loader import DEFAULT_DATA_ROOT, QA_FILENAME, Question, load_questions, load_transcripts  # noqa: E402
from timeline import GOLD_HALF_WIDTH_CS, Step, schedule  # noqa: E402
from windows import Window, build_windows, clock_to_iso, write_windows_jsonl  # noqa: E402

ARMS = ("mem0_faiss", "elite_matched", "elite_full", "elite_infer")
HIT_KS = (1, 5, 20)


def load_records(run: Path) -> dict[str, list[dict[str, Any]]]:
    """questions_<arm>.jsonl for every arm present.
    读取每个存在的臂的 questions_<arm>.jsonl。

    Input / 输入：
        run: Result directory.
        run：结果目录。

    Output / 输出：
        {arm: records in schedule order}.
        {臂: 按调度顺序的记录}。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        p = run / f"questions_{arm}.jsonl"
        if p.exists():
            out[arm] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def check_retained(run: Path, steps: list[Step], records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """(1) Same retained set per question: arm counts vs timeline, digests, and returned ⊆ retained-at-step.
    (1) 每题保留集合相同：臂计数对时间线、摘要，以及返回窗口 ⊆ 该步已保留集合。

    Input / 输入：
        run, steps, records: Result dir, rebuilt timeline, per-arm records.
        run、steps、records：结果目录、重建的时间线、逐臂记录。

    Output / 输出：
        Check dict.
        检查字典。

    State effect / 状态影响：
        None.
        无。
    """
    arm_files = {a: json.loads((run / f"arm_{a}.json").read_text(encoding="utf-8")) for a in ARMS if (run / f"arm_{a}.json").exists()}
    timeline_counts = [s.retained_count for s in steps]
    retained_so_far: set[str] = set()
    retained_at: list[set[str]] = []
    for s in steps:
        retained_so_far |= {w.window_id for w in s.new_windows}
        retained_at.append(set(retained_so_far))
    outside = {a: 0 for a in records}
    for a, recs in records.items():
        for rec in recs:
            pos = rec["position"]
            outside[a] += sum(1 for wid in rec["top_windows"] if wid not in retained_at[pos])
    return {
        "arm_counts_equal_timeline": {a: f["retained_count_by_position"] == timeline_counts for a, f in arm_files.items()},
        "arm_digests": {a: f["retained_digest"] for a, f in arm_files.items()},
        "digests_identical": len({f["retained_digest"] for f in arm_files.values()}) == 1,
        "final_retained_total": {a: f["retained_total"] for a, f in arm_files.items()},
        "returned_windows_outside_retained_set": outside,
        "note": "per-step sets are not logged by the runner; the count sequence + final digest + returned⊆retained(step) are the evidence",
    }


def check_leakage(records: dict[str, list[dict[str, Any]]], by_id: dict[str, Window], questions: dict[str, Question]) -> dict[str, Any]:
    """(2) Every returned window starts (and ends) no later than the question's cutoff.
    (2) 每个返回窗口的起点（与终点）都不晚于该题的截止时刻。

    Input / 输入：
        records, by_id, questions: Per-arm records, window_id → Window, id → Question.
        records、by_id、questions：逐臂记录、window_id → Window、id → Question。

    Output / 输出：
        Violation counts per arm (start-rule and end-rule) and the worst margin.
        每臂违规计数（起点规则与终点规则）及最差余量。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, Any] = {}
    for a, recs in records.items():
        start_viol = end_viol = later_day = 0
        max_end_over = -10**12
        for rec in recs:
            q = questions[rec["id"]]
            for wid in rec["top_windows"]:
                w = by_id[wid]
                if w.day > q.query_day:
                    later_day += 1
                    continue
                if w.day == q.query_day:
                    if w.start_cs > q.query_cs:
                        start_viol += 1
                    if w.end_cs > q.query_cs:
                        end_viol += 1
                    max_end_over = max(max_end_over, w.end_cs - q.query_cs)
        out[a] = {
            "later_day": later_day,
            "start_after_query": start_viol,
            "end_after_query": end_viol,
            "max_same_day_end_minus_query_cs": max_end_over if max_end_over > -10**12 else None,
        }
    return out


def check_cutoff_rule(steps: list[Step], windows: list[Window]) -> dict[str, Any]:
    """Compare the runner's window rule (end <= query) with their cue rule (start <= query): what is withheld.
    比较运行器的窗口规则（end ≤ query）与对方的 cue 规则（start ≤ query）：被扣留了什么。

    Input / 输入：
        steps, windows: Timeline and windows.
        steps、windows：时间线与窗口。

    Output / 输出：
        Counts of straddling windows and of questions whose strict-gold window straddles the query instant.
        跨越截止时刻的窗口数，以及严格 gold 窗口跨越提问时刻的题数。

    State effect / 状态影响：
        None.
        无。
    """
    straddle_q = 0
    strict_gold_straddle: list[str] = []
    for s in steps:
        q = s.question
        strad = [w for w in windows if w.day == q.query_day and w.start_cs <= q.query_cs < w.end_cs]
        straddle_q += bool(strad)
        strict = strict_gold_ids(q, windows)
        if any(w.window_id in strict for w in strad):
            strict_gold_straddle.append(q.id)
    return {
        "runner_rule": "window visible iff window.end_cs <= cutoff (same day) / whole earlier days",
        "their_rule": "cue visible iff cue.start <= query_time (select_transcript_cues) / whole earlier days (23595999)",
        "questions_with_a_straddling_window_withheld": straddle_q,
        "questions_whose_strict_gold_window_straddles_query": strict_gold_straddle,
        "max_delay_s": 60.0,
    }


def strict_gold_ids(q: Question, windows: list[Window]) -> list[str]:
    """Windows that contain a target instant (start <= t <= end, same day) — the brief's definition.
    包含 target 时刻（同日且 start ≤ t ≤ end）的窗口——任务说明中的定义。

    Input / 输入：
        q, windows: Question and windows.
        q、windows：题目与窗口。

    Output / 输出：
        window_id list.
        window_id 列表。

    State effect / 状态影响：
        None.
        无。
    """
    return [w.window_id for w in windows if any(w.day == d and w.start_cs <= t <= w.end_cs for d, t in q.targets)]


def nearest_gold_ids(q: Question, windows: list[Window]) -> list[str]:
    """Fallback when no window contains the target: the window on that day with the smallest clock distance.
    没有窗口包含 target 时的兜底：同日时钟距离最小的窗口。

    Input / 输入：
        q, windows: Question and windows.
        q、windows：题目与窗口。

    Output / 输出：
        window_id list (one per target instant).
        window_id 列表（每个 target 时刻一个）。

    State effect / 状态影响：
        None.
        无。
    """
    out: list[str] = []
    for d, t in q.targets:
        same = [w for w in windows if w.day == d]
        if not same:
            continue
        best = min(same, key=lambda w: (0 if w.start_cs <= t <= w.end_cs else min(abs(w.start_cs - t), abs(w.end_cs - t)), w.window_id))
        out.append(best.window_id)
    return out


def check_gold(steps: list[Step], windows: list[Window], records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """(4) Gold definition: ±60 s overlap (runner) vs containment (brief); recompute hit@k under both.
    (4) gold 定义：±60 s 重叠（运行器）对包含（任务说明）；在两种定义下重算 hit@k。

    Input / 输入：
        steps, windows, records.
        steps、windows、records。

    Output / 输出：
        Distribution of gold counts and hit@k tables per definition.
        gold 计数分布与各定义下的 hit@k 表。

    State effect / 状态影响：
        None.
        无。
    """
    wide_counts = Counter(len(s.gold_ids) for s in steps)
    strict = {s.question.id: strict_gold_ids(s.question, windows) for s in steps}
    nearest = {s.question.id: nearest_gold_ids(s.question, windows) for s in steps}
    strict_empty = [qid for qid, g in strict.items() if not g]
    multi_target = sum(1 for s in steps if len(s.question.targets) > 1)

    def hits(gold_of: dict[str, list[str]], scored_ids: set[str]) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        for a, recs in records.items():
            n = 0
            hit = {k: 0 for k in HIT_KS}
            rr = 0.0
            for rec in recs:
                if rec["id"] not in scored_ids:
                    continue
                n += 1
                gold = set(gold_of[rec["id"]])
                rank = next((i + 1 for i, wid in enumerate(rec["top_windows"]) if wid in gold), None)
                for k in HIT_KS:
                    hit[k] += bool(rank is not None and rank <= k)
                rr += (1.0 / rank) if rank else 0.0
            table[a] = {"n": n, **{f"hit@{k}": round(hit[k] / n, 4) if n else None for k in HIT_KS}, "mrr": round(rr / n, 4) if n else None}
        return table

    all_ids = {s.question.id for s in steps}
    wide = {s.question.id: s.gold_ids for s in steps}
    return {
        "runner_definition": f"window overlaps [t-{GOLD_HALF_WIDTH_CS / 100:.0f}s, t+{GOLD_HALF_WIDTH_CS / 100:.0f}s] on the same day, any target instant",
        "brief_definition": "window contains t (start <= t <= end)",
        "gold_windows_per_question_runner": dict(sorted(wide_counts.items())),
        "gold_windows_per_question_strict": dict(sorted(Counter(len(g) for g in strict.values()).items())),
        "questions_multi_target": multi_target,
        "strict_gold_empty_questions": strict_empty,
        "excluded_future_runner": sum(1 for s in steps if s.excluded_future),
        "target_after_query": sum(1 for s in steps if s.target_after_query),
        "hit_runner_pm60s": hits(wide, all_ids),
        "hit_strict_containment": hits(strict, all_ids - set(strict_empty)),
        "hit_strict_or_nearest": hits({k: (v or nearest[k]) for k, v in strict.items()}, all_ids),
    }


def check_reader(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """(5) Reader isolation: source grep for hidden fields, and the tie→'A' free-correct count.
    (5) 阅读器隔离：源码中对隐藏字段的引用，以及并列→'A' 的白得正确数。

    Input / 输入：
        records.
        records。

    Output / 输出：
        Check dict.
        检查字典。

    State effect / 状态影响：
        None.
        无。
    """
    hidden = re.compile(r"\.answer\b|\.raw\b|keywords|\breason\b|\.targets\b|target_day|target_time")
    refs: dict[str, list[str]] = {}
    for name in ("reader.py", "arms.py", "windows.py", "run_egolife.py", "timeline.py"):
        lines = (HERE / name).read_text(encoding="utf-8").splitlines()
        hits = [f"{i + 1}: {ln.strip()}" for i, ln in enumerate(lines) if hidden.search(ln) and not ln.strip().startswith("#")]
        refs[name] = hits
    ties: dict[str, Any] = {}
    for a, recs in records.items():
        t = [r for r in recs if r["reader_invalid_tie"]]
        n = len(recs)
        correct = sum(1 for r in recs if r["reader_correct"])
        tie_a = sum(1 for r in t if r["gold"] == "A")
        ties[a] = {
            "invalid_ties": len(t),
            "ties_scored_correct_because_gold_is_A": tie_a,
            "reader_acc_as_reported": round(correct / n, 4),
            # Tie → 'A' is a coin with P(A) = 43/204; score a tie as the random expectation instead.
            # 并列 → 'A' 是一枚 P(A) = 43/204 的硬币；把并列按随机期望计分。
            "reader_acc_ties_as_0.25": round((correct - tie_a + 0.25 * len(t)) / n, 4),
            "reader_acc_ties_as_wrong": round((correct - tie_a) / n, 4),
            "reader_acc_non_tie_subset": round((correct - tie_a) / (n - len(t)), 4) if n > len(t) else None,
            "returned_unique_windows_mean": round(sum(r["returned_unique_windows"] for r in recs) / n, 2),
            "gold_windows_not_yet_retained_at_query(sum)": sum(r["gold_window_count"] - r["gold_retained_count"] for r in recs),
            "reader_pred_equals_window_pred_all": all(r["reader_pred"] == r["reader_window_pred"] for r in recs),
        }
    return {"hidden_field_references": refs, "ties": ties, "same_reader_call": "run_egolife.evaluate_step: one Reader instance, same read() and truncate(4000) for every arm"}


def rebuild(steps: list[Step], windows: list[Window], work: Path, n_steps: int) -> dict[str, Any]:
    """(3)(7) Stepwise rebuild of three arms with explain=True: stored bits, scores, threshold quantity, texts.
    (3)(7) 三个臂带 explain=True 的逐步重建：存储比特、分数、阈值对象、文本。

    Input / 输入：
        steps, windows: Timeline and windows.
        steps、windows：时间线与窗口。

        work: Scratch directory (deleted afterwards).
        work：临时目录（结束后删除）。

        n_steps: Number of leading steps to replay.
        n_steps：重放的前几步数。

    Output / 输出：
        Check dict.
        检查字典。

    State effect / 状态影响：
        Creates and removes work/.
        创建并删除 work/。
    """
    shutil.rmtree(work, ignore_errors=True)
    mem0 = Mem0FaissArm(work / "mem0_faiss")
    matched = EliteMatchedArm(work / "elite_matched")
    full = EliteFullArm(work / "elite_full")
    plain = shared_hash_embedder()
    by_id = {w.window_id: w for w in windows}
    list_equal = 0
    list_diff: list[dict[str, Any]] = []
    score_abs_diff_max = 0.0
    m0_final_eq_sem = e_final_eq_sem = e_max_possible_one = 0
    rows_checked = 0
    full_rows_below_sem_threshold = 0
    full_rows_total = 0
    full_max_possible: Counter = Counter()
    text_mismatch = {"mem0_faiss": 0, "elite_matched": 0, "elite_full": 0}
    sem_neg_in_faiss_top = 0
    for step in steps[:n_steps]:
        q = step.question
        iso = clock_to_iso(q.query_day, q.query_cs)
        for arm in (mem0, matched, full):
            arm.retain(step.new_windows)
        r_m = mem0.memory.search(q.question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, explain=True)["results"]
        r_e = matched.hms.search(q.question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, reference_date=iso, explain=True)["results"]
        r_f = full.hms.search(q.question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, reference_date=iso, explain=True)["results"]
        ids_m = [r["metadata"]["window_id"] for r in r_m]
        ids_e = [r["metadata"]["window_id"] for r in r_e]
        if ids_m == ids_e:
            list_equal += 1
        else:
            list_diff.append({"id": q.id, "common": len(set(ids_m) & set(ids_e)), "n_mem0": len(ids_m), "n_elite": len(ids_e)})
        sm = {r["metadata"]["window_id"]: r for r in r_m}
        se = {r["metadata"]["window_id"]: r for r in r_e}
        for wid in set(sm) & set(se):
            score_abs_diff_max = max(score_abs_diff_max, abs(sm[wid]["score"] - se[wid]["score"]))
        for r in r_m:
            d = r["score_details"]
            rows_checked += 1
            m0_final_eq_sem += d["final_score"] == d["semantic_score"] == r["score"] and d["max_possible_score"] == 1.0
            sem_neg_in_faiss_top += d["semantic_score"] < 0
            text_mismatch["mem0_faiss"] += r["memory"] != by_id[r["metadata"]["window_id"]].text
        for r in r_e:
            d = r["score_details"]
            e_final_eq_sem += d["final_score"] == d["semantic_score"] == r["score"]
            e_max_possible_one += d["max_possible_score"] == 1.0
            text_mismatch["elite_matched"] += r["memory"] != by_id[r["metadata"]["window_id"]].text
        for r in r_f:
            d = r["score_details"]
            full_rows_total += 1
            full_rows_below_sem_threshold += d["semantic_score"] < THRESHOLD
            full_max_possible[round(d["max_possible_score"], 3)] += 1
            text_mismatch["elite_full"] += r["memory"] != by_id[r["metadata"]["window_id"]].text
    # Stored vectors: FAISS float32 rows vs Elite float32 blobs vs the plain embedder rounded to float32.
    # 存储向量：FAISS 的 float32 行、Elite 的 float32 blob、普通嵌入器取整到 float32。
    vs = mem0.memory.vector_store
    faiss_bits_equal = elite_bits_equal = 0
    faiss_rows = 0
    for idx, vid in vs.index_to_id.items():
        payload = vs.docstore[vid]
        wid = payload.get("window_id")
        vec = vs.index.reconstruct(int(idx))
        ref = np.asarray(plain.embed(by_id[wid].text), dtype=np.float32)
        faiss_rows += 1
        faiss_bits_equal += bool(np.array_equal(vec.astype(np.float32), ref)) and vec.dtype == np.float32
    elite_vecs = matched.hms.store.load_vectors(None)
    rows = matched.hms.get_all(filters={"user_id": BANK}, top_k=1_000_000)["results"]
    # Mem0 'id' is the lineage id; atom_vectors is keyed by atom_id (the head version's atom).
    # Mem0 的 'id' 是谱系 id；atom_vectors 以 atom_id（头版本原子）为键。
    atom_to_wid = {matched.hms.store.resolve_head(r["id"]).atom_id: r["metadata"]["window_id"] for r in rows}
    elite_rows = 0
    for atom_id, vec in elite_vecs.items():
        wid = atom_to_wid.get(atom_id)
        if wid is None:
            continue
        elite_rows += 1
        ref = np.asarray(plain.embed(by_id[wid].text), dtype=np.float32)
        elite_bits_equal += bool(np.array_equal(np.asarray(vec, dtype=np.float32), ref))
    retained_n = len(mem0.retained_ids)
    for arm in (mem0, matched, full):
        arm.close()
    shutil.rmtree(work, ignore_errors=True)
    return {
        "steps_replayed": n_steps,
        "windows_retained": retained_n,
        "mem0_vs_elite_matched_top20_lists_equal": list_equal,
        "mem0_vs_elite_matched_top20_lists_differ": list_diff,
        "shared_window_score_abs_diff_max": score_abs_diff_max,
        "mem0_rows": rows_checked,
        "mem0_rows_final==semantic&max_possible==1": m0_final_eq_sem,
        "mem0_rows_semantic_negative": sem_neg_in_faiss_top,
        "elite_matched_rows_final==semantic": e_final_eq_sem,
        "elite_matched_rows_max_possible==1": e_max_possible_one,
        "elite_full_rows": full_rows_total,
        "elite_full_rows_with_semantic_below_threshold": full_rows_below_sem_threshold,
        "elite_full_max_possible_distribution": dict(full_max_possible),
        "returned_text_differs_from_window_text": text_mismatch,
        "faiss_stored_rows": faiss_rows,
        "faiss_rows_bitwise_equal_float32(embed)": faiss_bits_equal,
        "elite_stored_rows": elite_rows,
        "elite_rows_bitwise_equal_float32(embed)": elite_bits_equal,
        "precision_note": "both stores hold float32; FAISS IndexFlatIP accumulates in float32, Elite decodes to float64 and dots in float64",
    }


def check_embedding_probe(windows: list[Window], questions: list[Question]) -> dict[str, Any]:
    """(3) 50 window texts + 20 question texts through the mem0 shell and the plain HashEmbedder: list equality.
    (3) 50 段窗口文本 + 20 段题干经 mem0 薄壳与普通 HashEmbedder：列表相等。

    Input / 输入：
        windows, questions.
        windows、questions。

    Output / 输出：
        Check dict with sha256 of both sides.
        含两侧 sha256 的检查字典。

    State effect / 状态影响：
        None.
        无。
    """
    shell = SharedHashEmbedding(type("Cfg", (), {"embedding_dims": 64})())
    plain = shared_hash_embedder()
    texts = [w.text for w in windows[:50]] + [q.question for q in questions[:20]]
    a = [shell.embed(t, "add") for t in texts]
    b = [plain.embed(t) for t in texts]
    return {
        "texts": len(texts),
        "equal": a == b,
        "sha256_shell": hashlib.sha256(json.dumps(a).encode()).hexdigest(),
        "sha256_plain": hashlib.sha256(json.dumps(b).encode()).hexdigest(),
    }


def markdown(report: dict[str, Any]) -> str:
    """Flatten the report into one two-column table per check.
    把报告压成每个检查一张两列表。

    Input / 输入：
        report: The controls.json structure.
        report：controls.json 的结构。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """
    lines: list[str] = []
    for section, body in report.items():
        lines += [f"## {section}", "", "| key | value |", "|---|---|"]
        if isinstance(body, dict):
            for k, v in body.items():
                lines.append(f"| {k} | {json.dumps(v, ensure_ascii=False)} |")
        else:
            lines.append(f"| value | {json.dumps(body, ensure_ascii=False)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: --run --rebuild-steps --data-root.
        argv：--run --rebuild-steps --data-root。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Writes <run>/../audit1/controls.{json,md}.
        写 <run>/../audit1/controls.{json,md}。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "results" / "full")
    ap.add_argument("--rebuild-steps", type=int, default=52)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = ap.parse_args(argv)
    out = args.run.parent / "audit1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    questions = load_questions(args.data_root / QA_FILENAME)[: manifest["limit"]]
    windows = build_windows(load_transcripts(args.data_root))
    sha = write_windows_jsonl(windows, out / "windows_rebuilt.jsonl")
    steps = schedule(questions, windows)
    by_id = {w.window_id: w for w in windows}
    qmap = {q.id: q for q in questions}
    records = load_records(args.run)
    report: dict[str, Any] = {
        "inputs": {
            "run": str(args.run),
            "windows_sha256_rebuilt": sha,
            "windows_sha256_manifest": manifest["windows_sha256"],
            "windows_equal": sha == manifest["windows_sha256"],
            "transcript_files_on_disk": manifest["transcript_files"],
            "questions": len(questions),
        },
        "1_retained": check_retained(args.run, steps, records),
        "2_leakage": check_leakage(records, by_id, qmap),
        "2b_cutoff_rule": check_cutoff_rule(steps, windows),
        "3_embedding_probe": check_embedding_probe(windows, questions),
        "4_gold": check_gold(steps, windows, records),
        "5_reader": check_reader(records),
        "3_7_rebuild": rebuild(steps, windows, out / "work", args.rebuild_steps),
    }
    (out / "controls.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "controls.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
