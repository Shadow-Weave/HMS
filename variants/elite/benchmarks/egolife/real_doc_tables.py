"""Render the design-document tables of a real-LLM run (five dimensions + hit table) from its summary.json.
从真实 LLM 运行的 summary.json 渲染设计文档的表（五维 + 命中率表）。
"""

# Purpose (EN): One source for every number quoted in docs/design/EGOLIFE_REAL_LLM_*.md and the README subsection;
#               every cell names its summary.json path so the tables are regenerated rather than retyped.
# 目的（中）：docs/design/EGOLIFE_REAL_LLM_*.md 与 README 小节引用的每个数字都只有这一个来源；每张表标出 summary.json
#            路径，表格是重新生成而不是手抄。

# Input (EN): --out <result dir> (default results/real/elite_full204_r2); --which accuracy|latency|tokens|money|stability|hit|all.
# 输入（中）：--out <结果目录>（默认 results/real/elite_full204_r2）；--which accuracy|latency|tokens|money|stability|hit|all。

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
DEFAULT_OUT = HERE / "results" / "real" / "elite_full204_r2"
TYPES = ("EntityLog", "EventRecall", "HabitInsight", "RelationMap", "TaskMaster")


def pct(v: Any) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def num(v: Any, nd: int = 1) -> str:
    if v is None:
        return "未实测"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{nd}f}"


def usd(v: Any, nd: int = 4) -> str:
    return "未实测" if v is None else f"{v:.{nd}f}"


def accuracy_table(summary: dict[str, Any]) -> str:
    """Rows: first-50 / all 204 / by type / error categories; one column per context label.
    行：first-50 / 全 204 / 分题型 / 错误原因；每个上下文标签一列。

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
    L = ["| 准确率（正确数 / 分母 = %） | " + " | ".join(arms) + " | 出处 summary.json |", "|---|" + "---|" * len(arms) + "---|"]

    def row(name: str, cells: list[str], src: str) -> None:
        L.append(f"| {name} | " + " | ".join(cells) + f" | {src} |")

    def acc(a: dict[str, Any], c: str, n: str, p: str) -> str:
        return f"{a[c]}/{a[n]} = {pct(a[p])}"

    A = {a: summary["arms"][a]["accuracy"] for a in arms}
    row("first-50（QA 文件顺序前 50 题）", [acc(A[a], "correct_first50", "n_first50", "accuracy_first50") for a in arms], "arms.<arm>.accuracy.accuracy_first50")
    row("全 204", [acc(A[a], "correct", "n", "accuracy") for a in arms], "arms.<arm>.accuracy.accuracy")
    for t in TYPES:
        row(f"题型 {t}", ["-" if t not in summary["arms"][a]["by_type"] else acc(summary["arms"][a]["by_type"][t], "correct", "n", "accuracy") for a in arms], f"arms.<arm>.by_type.{t}.accuracy")
    row(
        "诊断：gold 批次已入库子集",
        ["-" if not A[a]["n_gold_stored"] else f"{round(A[a]['accuracy_gold_stored_subset'] * A[a]['n_gold_stored'])}/{A[a]['n_gold_stored']} = {pct(A[a]['accuracy_gold_stored_subset'])}" for a in arms],
        "arms.<arm>.accuracy.accuracy_gold_stored_subset / n_gold_stored",
    )
    e = {a: A[a]["error_categories"] for a in arms}
    row("失败：未检回证据（其中 gold 批次未入库）", [f"{e[a]['no_evidence_retrieved']}（{A[a]['no_evidence_gold_unstored']}）" for a in arms], "arms.<arm>.accuracy.error_categories.no_evidence_retrieved / no_evidence_gold_unstored")
    row("失败：有证据答错", [str(e[a]["evidence_present_wrong_answer"]) for a in arms], "…error_categories.evidence_present_wrong_answer")
    row("失败：无效输出 / API 错误 / 超时", [f"{e[a]['invalid_output']} / {e[a]['api_error']} / {e[a]['timeout']}" for a in arms], "…error_categories.invalid_output / api_error / timeout")
    row("随机基线", [pct(A[a]["random_baseline"]) for a in arms], "arms.<arm>.accuracy.random_baseline")
    return "\n".join(L)


def latency_table(summary: dict[str, Any]) -> str:
    """Stage × (median / p95) with model vs local split and end-to-end.
    阶段 ×（中位 / P95），拆 model 与 local，并给端到端。
    """
    arms = list(summary["arms"])
    eng = summary["engines"][summary["arms"][arms[0]]["engine"]]
    L = ["| 延迟 ms（中位 / P95，nearest-rank） | " + " | ".join(arms) + " | 出处 summary.json |", "|---|" + "---|" * len(arms) + "---|"]

    def d(x: dict[str, Any]) -> str:
        return f"{num(x['median'])} / {num(x['p95'])}"

    T = {a: summary["arms"][a]["latency"] for a in arms}
    L.append("| 写入（每步；model = 抽取 API 含二分子 add） | " + " | ".join(d(T[a]["retain"]["model"]) for a in arms) + " | arms.<arm>.latency.retain.model |")
    L.append("| 写入（每步；local = 嵌入 + SQLite） | " + " | ".join(d(T[a]["retain"]["local"]) for a in arms) + " | arms.<arm>.latency.retain.local |")
    r = eng["retain"]
    L.append("| 写入（每次 add；model） | " + " | ".join(d(r["add_model_ms"]) for _ in arms) + " | engines.elite.retain.add_model_ms |")
    L.append("| 检索 / compile（local；无 API） | " + " | ".join(d(T[a]["recall_local"]) for a in arms) + " | arms.<arm>.latency.recall_local |")
    L.append("| 上下文构建（local） | " + " | ".join(d(T[a]["context_local"]) for a in arms) + " | arms.<arm>.latency.context_local |")
    L.append("| 答案生成（model） | " + " | ".join(d(T[a]["answer_model"]) for a in arms) + " | arms.<arm>.latency.answer_model |")
    L.append("| 答案生成（model，去首次调用） | " + " | ".join(d(T[a]["answer_model_excluding_cold_start"]) for a in arms) + " | …answer_model_excluding_cold_start |")
    L.append("| 端到端（写入 + 检索 + 构建 + 答题） | " + " | ".join(d(T[a]["end_to_end"]) for a in arms) + " | arms.<arm>.latency.end_to_end |")
    L.append("| 端到端（不含写入） | " + " | ".join(d(T[a]["end_to_end_excluding_retain"]) for a in arms) + " | …end_to_end_excluding_retain |")
    ex = eng["tokens"]["extract"]["latency_ms"]
    L.append("| 冷启动：建库 + 加载 fastembed / 抽取首次调用 / 答题首次调用 | " + " | ".join(f"{num(eng['stats'].get('init_ms'))} / {num(ex['cold_start_ms'])} / {num(summary['_arm_file']['labels'][a]['answer']['latency_ms']['cold_start_ms'])}" for a in arms) + " | engines.elite.stats.init_ms; engines.elite.tokens.extract.latency_ms.cold_start_ms; arm_elite.json labels.<arm>.answer.latency_ms.cold_start_ms |")
    L.append("| 并发 / 答题重试 | " + " | ".join(f"1 / {T[a]['answer_retries']}" for a in arms) + " | arms.<arm>.latency.concurrency / answer_retries |")
    return "\n".join(L)


def tokens_table(summary: dict[str, Any]) -> str:
    """Stage × prompt / completion / cache hit / cache miss, plus embedding text counts.
    阶段 × prompt / completion / 缓存命中 / 未命中，加嵌入文本数。
    """
    arms = list(summary["arms"])
    eng_name = summary["arms"][arms[0]]["engine"]
    eng = summary["engines"][eng_name]
    ex = eng["tokens"]["extract"]
    L = ["| Token（API 返回 usage） | 尝试数 | 输入 prompt | 输出 completion | 缓存命中 | 缓存未命中 | 出处 summary.json |", "|---|---|---|---|---|---|---|"]
    t = ex["tokens"]
    L.append(f"| 抽取（两种上下文共用一次 retain） | {ex['attempts']} | {num(t['prompt'])} | {num(t['completion'])} | {num(t['cache_hit'])} | {num(t['cache_miss'])} | engines.{eng_name}.tokens.extract |")
    for a in arms:
        an = summary["arms"][a]["stability"]["answer"]
        t = summary["arms"][a]["tokens"]["answer"]
        L.append(f"| 答题 {a} | {an['attempts']} | {num(t['prompt'])} | {num(t['completion'])} | {num(t['cache_hit'])} | {num(t['cache_miss'])} | arms.{a}.tokens.answer |")
    emb = eng["tokens"]["embedding"]
    bs = emb["by_stage"]
    L.append("| 检索 / compile | 0 | 本地，无 API token | | | | arms.<arm>.tokens.organize |")
    L.append(f"| 嵌入 fastembed bge-small（本地） | {emb['calls']} 次调用 | 文本数 {emb['texts']}（抽取 {bs.get('extract', {}).get('texts', 0)}，查询 {bs.get('recall', {}).get('texts', 0)}） | 本地 {num(emb['local_ms_total'])} ms | 无 | 无 | engines.{eng_name}.tokens.embedding |")
    return "\n".join(L)


def money_table(summary: dict[str, Any]) -> str:
    """USD totals and unit costs, tier and cache shares.
    美元总额与单位成本、档位与缓存占比。
    """
    arms = list(summary["arms"])
    lg = summary["ledger"]
    L = ["| 金额 USD | " + " | ".join(arms) + " | 全程（抽取只计一次） | 出处 summary.json |", "|---|" + "---|" * (len(arms) + 1) + "---|"]
    M = {a: summary["arms"][a]["money"] for a in arms}
    L.append("| 抽取 + 答题 = 总额 | " + " | ".join(f"{usd(M[a]['extract_cost_usd'])} + {usd(M[a]['answer_cost_usd'])} = {usd(M[a]['total_cost_usd'])}" for a in arms) + f" | {usd(lg['cost_usd_by_stage']['extract'])} + {usd(lg['cost_usd_by_stage']['answer'])} = {usd(lg['cost_usd_total'])} | arms.<arm>.money.*_cost_usd; ledger.cost_usd_total |")
    L.append("| 每题 / 每答对一题 | " + " | ".join(f"{usd(M[a]['cost_per_question_usd'], 5)} / {usd(M[a]['cost_per_correct_usd'], 5)}" for a in arms) + " | | arms.<arm>.money.cost_per_question_usd / cost_per_correct_usd |")
    L.append("| 每窗口摊销（抽取） / 每批次 | " + " | ".join(f"{usd(M[a]['retain_cost_per_window_usd'], 7)} / {usd(M[a]['retain_cost_per_batch_usd'], 5)}" for a in arms) + " | | arms.<arm>.money.retain_cost_per_window_usd / retain_cost_per_batch_usd |")
    L.append("| 高峰 / 非高峰（高峰占比） | " + " | ".join(f"{usd(M[a]['cost_by_tier_usd']['peak'])} / {usd(M[a]['cost_by_tier_usd']['offpeak'])}（{pct(M[a]['peak_share'])}%）" for a in arms) + f" | {usd(lg['cost_usd_by_tier']['peak'])} / {usd(lg['cost_usd_by_tier']['offpeak'])}；尝试数 {lg['by_tier']} | arms.<arm>.money.cost_by_tier_usd; ledger.cost_usd_by_tier / by_tier |")
    L.append("| 缓存命中占输入 token | " + " | ".join(f"{pct(M[a]['cache_hit_share_of_prompt_tokens'])}%" for a in arms) + " | | arms.<arm>.money.cache_hit_share_of_prompt_tokens |")
    L.append("| 未计价尝试（无 usage） | " + " | ".join(str(M[a]["unpriced_attempts"]) for a in arms) + f" | {lg['unpriced_attempts']} | arms.<arm>.money.unpriced_attempts |")
    return "\n".join(L)


def stability_table(summary: dict[str, Any]) -> str:
    """Timeout / failure / empty rates, retries, extraction truncation and parse failures.
    超时 / 失败 / 空答案率、重试、抽取截断与解析失败。
    """
    arms = list(summary["arms"])
    eng_name = summary["arms"][arms[0]]["engine"]
    r = summary["engines"][eng_name]["retain"]
    pf = r["extraction_parse_failures"]
    lo = pf["leaf_outcomes"]
    st0 = summary["arms"][arms[0]]["stability"]
    ex = st0["extract"]
    L = ["| 稳定性 | 值 | 出处 summary.json |", "|---|---|---|"]
    L.append(f"| 抽取：尝试 / 请求 / 重试；超时率 / 失败率 / 空回复率 | {ex['attempts']} / {ex['requests']} / {ex['retries']}；{pct(ex['timeout_rate'])}% / {pct(ex['failure_rate'])}% / {pct(ex['empty_rate'])}% | arms.<arm>.stability.extract |")
    for a in arms:
        an = summary["arms"][a]["stability"]["answer"]
        L.append(f"| 答题 {a}：尝试 / 重试；超时率 / 失败率 / 空答案率（题级） | {an['attempts']} / {an['retries']}；{pct(an['timeout_rate'])}% / {pct(an['failure_rate'])}% / {pct(summary['arms'][a]['stability']['answer_empty_question_rate'])}% | arms.{a}.stability.answer / answer_empty_question_rate |")
    L.append(f"| 抽取截断（finish=length）：尝试数 / 占 add 次数；被二分批次 / 完全恢复 | {r['truncated_attempts']} / {pct(st0['extraction_truncation_rate_over_adds'])}%（{r['adds']} 次 add，{r['batches']} 批）；{pf['batches_bisected']} / {pf['batches_truncation_recovered']} | engines.{eng_name}.retain.truncated_attempts / adds; …extraction_parse_failures.batches_bisected / batches_truncation_recovered |")
    L.append(f"| 抽取解析失败（未恢复）：批次；非法 JSON（finish=stop）/ 单窗口仍截断 | {pf['batches']}（{pct(pf['rate_over_batches'])}%）；{lo['invalid_json']} / {lo['truncated_unresolved']} | …extraction_parse_failures.batches / leaf_outcomes |")
    L.append(f"| 抽取解析成功但未入库：全部条目被模式拒绝 / 空抽取；被拒条目数（attributed_to） | {lo['schema_rejected_all']} / {lo['empty_extraction']}；{pf['items_skipped']}（{pf['items_skipped_attributed_to']}） | …leaf_outcomes.schema_rejected_all / empty_extraction; items_skipped |")
    L.append(f"| 未入库批次数 / 窗口：入库 / 丢失（按原因） | {pf['batches_nothing_stored']}（{pct(pf['batches_nothing_stored_rate'])}%）/ {r['windows_stored']} / {r['windows_lost']} {json.dumps(pf['windows_lost_by_cause'])} | …batches_nothing_stored; engines.{eng_name}.retain.windows_stored / windows_lost |")
    ev = r["events"]
    L.append(f"| 记忆条目：add() 结果（ADD / UPDATE / NONE）/ 库内行数 | {r['memories_added']}（{ev.get('ADD', 0)} / {ev.get('UPDATE', 0)} / {ev.get('NONE', 0)}）/ {r['stored_count']} | engines.{eng_name}.retain.memories_added / events / stored_count |")
    return "\n".join(L)


def hit_table(summary: dict[str, Any]) -> str:
    """Retrieval hit at batch granularity per label and per type (not accuracy).
    批次粒度的检索命中，按标签与题型（不是准确率）。
    """
    arms = list(summary["arms"])
    L = ["| 命中率（批次粒度；不是准确率） | " + " | ".join(arms) + " | 出处 summary.json |", "|---|" + "---|" * len(arms) + "---|"]
    A = {a: summary["arms"][a]["accuracy"] for a in arms}
    L.append("| k（返回条目数，中位 / 最大） | " + " | ".join(f"{num(A[a]['hit_k']['median'], 0)} / {num(A[a]['hit_k']['max'], 0)}" for a in arms) + " | arms.<arm>.accuracy.hit_k |")
    L.append("| hit@k（gold 窗口所在批次） | " + " | ".join(f"{pct(A[a]['hit_batch'])}" for a in arms) + " | arms.<arm>.accuracy.hit_batch |")
    L.append("| hit@k（±60 s 邻域） | " + " | ".join(f"{pct(A[a]['hit_batch_wide'])}" for a in arms) + " | arms.<arm>.accuracy.hit_batch_wide |")
    for t in TYPES:
        L.append(f"| hit@k 题型 {t} | " + " | ".join("-" if t not in summary["arms"][a]["by_type"] else pct(summary["arms"][a]["by_type"][t]["hit_batch"]) for a in arms) + f" | arms.<arm>.by_type.{t}.hit_batch |")
    return "\n".join(L)


TABLES = {
    "accuracy": accuracy_table,
    "latency": latency_table,
    "tokens": tokens_table,
    "money": money_table,
    "stability": stability_table,
    "hit": hit_table,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: --out, --which.
        argv：--out、--which。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Prints to stdout.
        打印到标准输出。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--which", default="all", choices=["all", *TABLES])
    args = parser.parse_args(argv)
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    engine = next(iter(summary["engines"]))
    summary["_arm_file"] = json.loads((args.out / f"arm_{engine}.json").read_text(encoding="utf-8"))  # per-label answer first-call ms / 各标签答题首次调用耗时
    names = list(TABLES) if args.which == "all" else [args.which]
    print("\n\n".join(TABLES[n](summary) for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
