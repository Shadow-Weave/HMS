"""Render the three design-document tables (main / by type / cost) from a result directory's summary.json.
从结果目录的 summary.json 渲染设计文档的三张表（主表 / 分类型 / 成本）。
"""

# Purpose (EN): One source for every number quoted in docs/design/EGOLIFE_BENCH_*.md and the README subsection,
#               so the tables are regenerated rather than retyped.
# 目的（中）：docs/design/EGOLIFE_BENCH_*.md 与 README 小节引用的每个数字都只有这一个来源，表格是重新生成而不是手抄。

# Input (EN): --out <result dir> (default results/final); --which main|type|cost|all.
# 输入（中）：--out <结果目录>（默认 results/final）；--which main|type|cost|all。

# Output (EN): Markdown tables on stdout.
# 输出（中）：标准输出上的 Markdown 表格。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def pct(v: Any) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def num(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def main_table(summary: dict[str, Any]) -> str:
    """Arms × {hit@1, hit@5, hit@20, MRR, nb-hit@20, reader acc first-50 / all, oracle, random}.
    臂 × {hit@1、hit@5、hit@20、MRR、nb-hit@20、阅读器 first-50 / all、oracle、随机}。

    Input / 输入：
        summary: summary.json.
        summary：summary.json。

    Output / 输出：
        Markdown.
        Markdown。

    State effect / 状态影响：
        None.
        无。
    """
    rows = [
        "| 臂 | hit@1 | hit@5 | hit@20 | MRR | nb-hit@20 (±60 s) | reader acc first-50 (valid-only, n) | reader acc all (valid-only, n) | oracle 上限 all (valid-only, n) | oracle first-50 | 随机 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, m in summary["arms"].items():
        a, f = m["all"], m["first50"]
        rows.append(
            f"| {arm} | {pct(a['hit@1'])} | {pct(a['hit@5'])} | {pct(a['hit@20'])} | {num(a['mrr'], 3)} | {pct(a['nb_hit@20'])} | "
            f"{pct(f['reader_acc'])} ({pct(f['reader_acc_valid_only'])}, {f['reader_valid_n']}) | "
            f"{pct(a['reader_acc'])} ({pct(a['reader_acc_valid_only'])}, {a['reader_valid_n']}) | "
            f"{pct(a['oracle_acc'])} ({pct(a['oracle_acc_valid_only'])}, {a['oracle_valid_n']}) | {pct(f['oracle_acc'])} | 25.0 |"
        )
    return "\n".join(rows)


def type_table(summary: dict[str, Any]) -> str:
    """Question type × arm: hit@5 (all scope).
    题型 × 臂：hit@5（全量）。

    Input / 输入：
        summary: summary.json.
        summary：summary.json。

    Output / 输出：
        Markdown.
        Markdown。

    State effect / 状态影响：
        None.
        无。
    """
    arms = list(summary["arms"])
    types = sorted({t for a in arms for t in summary["arms"][a]["by_type"]})
    rows = ["| 题型 | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for t in types:
        n = summary["arms"][arms[0]]["by_type"][t]["denominator"]
        rows.append(f"| {t} | {n} | " + " | ".join(pct(summary["arms"][a]["by_type"][t]["hit@5"]) for a in arms) + " |")
    return "\n".join(rows)


def cost_table(summary: dict[str, Any]) -> str:
    """Arm cost row: retain total, per-window mean / median, search first-call median / p90, calls, bytes, windows.
    臂成本行：retain 总时长、每窗口均值 / 中位、首次 search 中位 / p90、调用数、字节数、窗口数。

    Input / 输入：
        summary: summary.json.
        summary：summary.json。

    Output / 输出：
        Markdown.
        Markdown。

    State effect / 状态影响：
        None.
        无。
    """
    rows = [
        "| 臂 | 窗口数 | 存储行 | retain 总 ms | retain ms/窗口 均值 | retain ms/窗口 题中位 | Q1..Q4 ms/窗口 | search 首次 中位 ms | search 首次 p90 ms | search 3 次中位 ms | embedding 调用 | LLM 调用 | LLM token (est.) | 库字节 (close 后) | 文件明细 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, m in summary["arms"].items():
        s, t = m["stats"], m["timing"]
        tokens = s.get("llm_tokens", {}).get("total", 0) if isinstance(s.get("llm_tokens"), dict) else 0
        quart = " / ".join(num(v) for v in t["retain_ms_per_window_by_quartile"])
        files = "; ".join(f"{k} {v:,}" for k, v in (s.get("db_files") or {}).items())
        rows.append(
            f"| {arm} | {t['retain_windows']} | {s['stored_count']:,} | {num(t['retain_total_ms'], 1)} | {num(t['retain_ms_per_window_mean'])} | "
            f"{num(t['retain_ms_per_window_median_over_questions'])} | {quart} | {num(t['search_first_ms_median'])} | {num(t['search_first_ms_p90'])} | "
            f"{num(t['search_ms_median'])} | {s['embedding_calls']:,} | {s['llm_calls']:,} | {tokens:,} | {s['db_bytes']:,} | {files} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments.
        argv：参数。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Prints to stdout.
        打印到标准输出。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "results" / "final")
    parser.add_argument("--which", choices=("main", "type", "cost", "all"), default="all")
    args = parser.parse_args(argv)
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    parts = {"main": main_table, "type": type_table, "cost": cost_table}
    for key, fn in parts.items():
        if args.which in (key, "all"):
            print(fn(summary))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
