"""Audit 1 (6): how much of the Elite arms' hit@k comes from the event timestamp mem0 OSS cannot accept?
审计一（6）：Elite 各臂的 hit@k 有多少来自 mem0 OSS 无法接受的事件时间戳？
"""

# Purpose (EN): Re-run Elite variants through the unchanged runner (same timeline, reader, top_k, threshold):
#               timestamp=None (no time axis at all), w_recency=0 (time axis kept: hard time gate, recency
#               channel, tie-break), and both; then tabulate hit@k deltas and per-question flips against
#               results/full, annotated with whether the planner emitted a hard time gate for that question.
# 目的（中）：用不变的运行器（同一时间线、阅读器、top_k、阈值）重跑 Elite 变体：timestamp=None（完全无时间轴）、
#            w_recency=0（保留时间轴：硬时间门、新近度通道、平局打破）、两者同时；然后对照 results/full 列出 hit@k
#            差值与逐题翻转，并标注规划器是否对该题产生了硬时间门。

# Input (EN): --full (baseline result dir), --out (audit dir), --data-root, --repeats.
# 输入（中）：--full（基线结果目录）、--out（审计目录）、--data-root、--repeats。

# Output (EN): <out>/{questions_<variant>.jsonl, arm_<variant>.json, timestamp.json, timestamp.md}.
# 输出（中）：<out>/{questions_<variant>.jsonl, arm_<variant>.json, timestamp.json, timestamp.md}。

# State effect (EN): Engine files under <out>/work/ are deleted after each variant.
# 状态影响（中）：<out>/work/ 下的引擎文件在每个变体结束后删除。

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import arms as arms_mod  # noqa: E402
from arms import BANK, EliteFullArm, EliteMatchedArm  # noqa: E402
from loader import DEFAULT_DATA_ROOT, QA_FILENAME, load_questions, load_transcripts  # noqa: E402
from reader import Reader  # noqa: E402
from run_egolife import aggregate, run_arm  # noqa: E402
from timeline import schedule  # noqa: E402
from windows import Window, build_windows, clock_to_iso  # noqa: E402

from hms_elite.mvmi.planner import plan_query  # noqa: E402

HIT_KS = (1, 5, 20)


class _NoTimestamp:
    """Mixin: time-blind Elite — add() without timestamp AND search() without reference_date (the mem0 OSS
    surface). Dropping only the timestamp is not a control: time_anchor then defaults to the library clock
    (2026), every atom lies after a 2024 reference_date, the validity predicate returns 0 rows (measured:
    hit@20 = 0, reader = 43/204 ties).
    混入：时间盲的 Elite——add() 不带 timestamp 且 search() 不带 reference_date（mem0 OSS 的接口面）。只去掉
    timestamp 不是对照：time_anchor 退化为库时钟（2026），所有原子都晚于 2024 的 reference_date，有效性谓词
    返回 0 行（实测：hit@20 = 0，阅读器 = 43/204 平局）。
    """

    def _add(self, window: Window) -> None:  # type: ignore[override]
        self.hms.add(  # type: ignore[attr-defined]
            [{"role": "user", "content": window.text}],
            user_id=BANK,
            metadata={"window_id": window.window_id, "day": window.day, "start": window.start_cs, "end": window.end_cs, "source": window.source},
            infer=self.infer,  # type: ignore[attr-defined]
        )

    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:  # type: ignore[override]
        result = self.hms.search(question, filters={"user_id": BANK}, top_k=arms_mod.TOP_K, threshold=arms_mod.THRESHOLD)  # type: ignore[attr-defined]
        return [
            {"window_id": (row.get("metadata") or {}).get("window_id"), "text": row.get("memory", ""), "score": float(row.get("score", 0.0)), "memory_id": row.get("id")}
            for row in result.get("results", [])
        ]


class EliteMatchedNoTs(_NoTimestamp, EliteMatchedArm):
    """Semantic-only weights, no timestamp.
    仅语义权重，无时间戳。
    """

    name = "elite_matched_nots"


class EliteFullNoTs(_NoTimestamp, EliteFullArm):
    """Default weights, no timestamp.
    默认权重，无时间戳。
    """

    name = "elite_full_nots"


class EliteFullNoRecency(EliteFullArm):
    """Default weights except w_recency=0; timestamp kept (gate, recency channel and tie-break remain).
    除 w_recency=0 外默认权重；保留时间戳（硬时间门、新近度通道与平局打破仍在）。
    """

    name = "elite_full_norec"
    scoring = {"w_recency": 0.0}


class EliteFullNoRecencyNoTs(_NoTimestamp, EliteFullArm):
    """w_recency=0 and no timestamp.
    w_recency=0 且无时间戳。
    """

    name = "elite_full_norec_nots"
    scoring = {"w_recency": 0.0}


VARIANTS = (EliteMatchedNoTs, EliteFullNoTs, EliteFullNoRecency, EliteFullNoRecencyNoTs)
BASELINE_OF = {
    "elite_matched_nots": "elite_matched",
    "elite_full_nots": "elite_full",
    "elite_full_norec": "elite_full",
    "elite_full_norec_nots": "elite_full",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one questions JSONL.
    读取一个 questions JSONL。

    Input / 输入：
        path: File.
        path：文件。

    Output / 输出：
        Records.
        记录。

    State effect / 状态影响：
        None.
        无。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flips(base: list[dict[str, Any]], var: list[dict[str, Any]], gates: dict[str, bool]) -> dict[str, Any]:
    """Per-question hit@20 flips between a baseline arm and a variant, with the time-gate flag.
    基线臂与变体之间逐题的 hit@20 翻转，附时间门标记。

    Input / 输入：
        base, var: Records of the two arms (same order).
        base、var：两个臂的记录（同序）。

        gates: question id → planner emitted a hard time gate.
        gates：题目 id → 规划器是否产生硬时间门。

    Output / 输出：
        {'lost': [...], 'gained': [...], 'rank_changed': n, 'top_list_changed': n}.
        {'lost': [...], 'gained': [...], 'rank_changed': n, 'top_list_changed': n}。

    State effect / 状态影响：
        None.
        无。
    """
    lost: list[dict[str, Any]] = []
    gained: list[dict[str, Any]] = []
    rank_changed = list_changed = 0
    for b, v in zip(base, var):
        assert b["id"] == v["id"]
        if b["rank"] != v["rank"]:
            rank_changed += 1
        if b["top_windows"] != v["top_windows"]:
            list_changed += 1
        entry = {"id": b["id"], "type": b["type"], "base_rank": b["rank"], "variant_rank": v["rank"], "time_gate": gates[b["id"]]}
        if b["hit@20"] and not v["hit@20"]:
            lost.append(entry)
        elif v["hit@20"] and not b["hit@20"]:
            gained.append(entry)
    return {"lost_by_variant": lost, "gained_by_variant": gained, "rank_changed": rank_changed, "top_list_changed": list_changed}


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: --full --out --data-root --repeats.
        argv：--full --out --data-root --repeats。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Writes the variant files and the two tables under --out.
        在 --out 下写变体文件与两张表。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=HERE / "results" / "full")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "audit1")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.full / "manifest.json").read_text(encoding="utf-8"))
    questions = load_questions(args.data_root / QA_FILENAME)[: manifest["limit"]]
    windows = build_windows(load_transcripts(args.data_root))
    steps = schedule(questions, windows)
    text_by_id = {w.window_id: w.text for w in windows}
    reader = Reader(windows)
    oracle_by_id: dict[str, dict[str, Any]] = {}
    gates: dict[str, bool] = {}
    for step in steps:
        q = step.question
        res = reader.read(q.question, q.options, reader.oracle_evidence(q))
        oracle_by_id[q.id] = {"pred": res.pred, "correct": res.pred == q.answer, "invalid_tie": res.invalid_tie}
        gates[q.id] = plan_query(q.question, as_of=clock_to_iso(q.query_day, q.query_cs)).hard_time_gate is not None
    for cls in VARIANTS:
        arms_mod.ARMS[cls.name] = cls  # registered for make_arm / 注册给 make_arm
    work = args.out / "work"
    table: dict[str, Any] = {}
    for cls in VARIANTS:
        print(f"[audit1] variant {cls.name}", flush=True)
        result = run_arm(cls.name, steps, reader, text_by_id, oracle_by_id, work_root=work, repeats=args.repeats, out_path=args.out / f"questions_{cls.name}.jsonl", keep_db=False)
        arm_file = {k: v for k, v in result.items() if k != "records"}
        arm_file["arm"] = cls.name
        (args.out / f"arm_{cls.name}.json").write_text(json.dumps(arm_file, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        base_name = BASELINE_OF[cls.name]
        base = load_jsonl(args.full / f"questions_{base_name}.jsonl")
        agg_v = aggregate(result["records"])
        agg_b = aggregate(base)
        table[cls.name] = {
            "baseline": base_name,
            "scoring": result["stats"]["extra"]["scoring"],
            "timestamp_passed": not issubclass(cls, _NoTimestamp),
            "retained_digest_equals_baseline": result["retained_digest"] == json.loads((args.full / f"arm_{base_name}.json").read_text())["retained_digest"],
            **{f"{k}_variant": agg_v[k] for k in ("hit@1", "hit@5", "hit@20", "mrr", "reader_acc")},
            **{f"{k}_baseline": agg_b[k] for k in ("hit@1", "hit@5", "hit@20", "mrr", "reader_acc")},
            **{f"{k}_delta": round(agg_v[k] - agg_b[k], 4) for k in ("hit@1", "hit@5", "hit@20", "mrr", "reader_acc")},
            "flips_vs_baseline": flips(base, result["records"], gates),
        }
    shutil.rmtree(work, ignore_errors=True)
    mem0 = load_jsonl(args.full / "questions_mem0_faiss.jsonl")
    matched = load_jsonl(args.full / "questions_elite_matched.jsonl")
    table["elite_matched_vs_mem0_faiss(full run)"] = flips(mem0, matched, gates)
    table["questions_with_hard_time_gate"] = [qid for qid, g in gates.items() if g]
    (args.out / "timestamp.json").write_text(json.dumps(table, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["| variant | baseline | ts | hit@1 | hit@5 | hit@20 | MRR | reader | Δhit@20 | ΔMRR | lost | gained | lists changed |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in table.items():
        if not isinstance(row, dict) or "baseline" not in row:
            continue
        f = row["flips_vs_baseline"]
        lines.append(
            f"| {name} | {row['baseline']} | {row['timestamp_passed']} | {row['hit@1_variant']} | {row['hit@5_variant']} | {row['hit@20_variant']} | "
            f"{row['mrr_variant']} | {row['reader_acc_variant']} | {row['hit@20_delta']:+.4f} | {row['mrr_delta']:+.4f} | {len(f['lost_by_variant'])} | {len(f['gained_by_variant'])} | {f['top_list_changed']} |"
        )
    (args.out / "timestamp.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in table.items() if not isinstance(v, dict) or "baseline" not in v}, ensure_ascii=False, indent=1))
    for name, row in table.items():
        if isinstance(row, dict) and "baseline" in row:
            print(name, json.dumps(row["flips_vs_baseline"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
