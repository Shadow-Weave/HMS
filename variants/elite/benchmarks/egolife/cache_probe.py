"""Prompt-cache probe: one card text sent under three prompt patterns to deepseek-flash, recording the provider's
prompt_cache_hit_tokens / prompt_cache_miss_tokens per call and the effective CNY per call against a no-cache baseline.
提示词缓存探针：同一张卡片文本以三种提示词模式发给 deepseek-flash，逐次记录 API 返回的 prompt_cache_hit_tokens /
prompt_cache_miss_tokens，并给出相对无缓存基线的每次调用有效人民币费用。
"""

# Purpose (EN): Measure, with the real client, whether a shared 'Memories:' prefix ahead of N different questions is
#               served from DeepSeek's prompt cache (pass a), whether the question-first order defeats it (pass b), and
#               what an exactly repeated prompt yields (pass c, on an (N+1)-th question that passes a / b never sent,
#               so its first call is a cold miss for the question suffix) — the empirical basis for session_prefix.
# 目的（中）：用真实客户端测量：N 道不同题目前置同一段 'Memories:' 前缀时是否命中 DeepSeek 的提示词缓存（pass a），
#            题目在前的顺序是否使其失效（pass b），以及完全相同的提示词重复发送得到什么（pass c，用 pass a / b 从未发送
#            的第 N+1 道题，使其首次调用对题目后缀是冷未命中）——session_prefix 变体的经验依据。

# Input (EN): --card <text file> (a compiled card), the EgoLife question file under data/ (the N+1 shortest distinct
#             questions are taken from it: N for passes a / b, the last one for pass c), benchmarks/egolife/.env
#             (DEEPSEEK_API_KEY via real_arms.load_api_key; never printed or written).
# 输入（中）：--card <文本文件>（一张编译好的卡片）、data/ 下的 EgoLife 题目文件（取其中最短且互不相同的 N+1 道题：
#            N 道用于 pass a / b，最后一道用于 pass c）、benchmarks/egolife/.env（经 real_arms.load_api_key 读取
#            DEEPSEEK_API_KEY；绝不打印或写出）。

# Output (EN): <out>/{probe.json, probe.md, calls.jsonl}; a table on stdout.
# 输出（中）：<out>/{probe.json, probe.md, calls.jsonl}；标准输出一张表。

# State effect (EN): About N x 2 + repeat API calls (~30 x 1.6k tokens; negligible cost), --sleep seconds apart;
#                    the answers are not scored.
# 状态影响（中）：约 N x 2 + repeat 次 API 调用（约 30 x 1.6k token；费用可忽略），调用间隔 --sleep 秒；不给答案计分。

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accounting import Meter  # noqa: E402
from loader import DEFAULT_DATA_ROOT, QA_FILENAME, Question, load_questions  # noqa: E402
from reader_ab import PRICES_CNY_PER_M, ask, build_prompt, cny_cost, normalize_text  # noqa: E402
from real_arms import DeepSeekClient, load_api_key  # noqa: E402
from replay_ab import ENV_PATH  # noqa: E402

DEFAULT_N = 12
DEFAULT_REPEAT = 5
DEFAULT_SLEEP_S = 2.0
PASSES = ("a_memories_first_distinct", "b_question_first_distinct", "c_memories_first_repeated")


def split_questions(qs: list[Question], n: int) -> tuple[list[Question], Question, bool]:
    """Split the picked questions into the N of passes a / b and the one of pass c: the (N+1)-th when available
    (its memories-first prompt was never sent, so pass c's first call is cold for the question suffix; the shared
    card prefix may still be warm from pass a); when the file holds only N distinct texts, pass c falls back to
    qs[0] and the flag says so (its first call is then already warm from pass a call #0).
    把选出的题目拆成 pass a / b 的 N 道与 pass c 的一道：有第 N+1 道时用它（其记忆在前的提示词从未发送过，故 pass c
    首次调用对题目后缀是冷的；共享的卡片前缀仍可能因 pass a 而热）；文件只有 N 道不同题干时退回 qs[0] 并以标志说明
    （其首次调用已被 pass a 第 0 次调用预热）。

    Input / 输入：
        qs: pick_questions' output (up to N+1).
        qs：pick_questions 的输出（至多 N+1 道）。

        n: N.
        n：N。

    Output / 输出：
        (pass a / b questions, pass c question, pass_c_cold).
        （pass a / b 题目，pass c 题目，pass_c_cold）。

    State effect / 状态影响：
        None.
        无。
    """
    head = qs[:n]
    if len(qs) > n:
        return head, qs[n], True
    return head, qs[0], False


def pick_questions(questions: list[Question], n: int, *, max_chars: int | None = None) -> list[Question]:
    """The n SHORTEST questions with pairwise distinct question text (case / whitespace-insensitive): candidates are
    stably sorted by question length (file order breaks ties), optionally filtered to len <= max_chars first, and the
    first n distinct texts are taken, so the non-shared suffix of every pass stays as small and as uniform as possible.
    n 道题干两两不同（大小写 / 空白不敏感）的最短题目：候选按题干长度稳定排序（同长按文件顺序），可先按 len <= max_chars
    过滤，再取前 n 道不同题干，使每个 pass 中不共享的后缀尽量短且均匀。

    Input / 输入：
        questions: All questions.
        questions：全部题目。

        n: How many.
        n：数量。

        max_chars: Optional upper bound on question length (None = no bound).
        max_chars：题干长度上限（None = 不限）。

    Output / 输出：
        The chosen questions, shortest first (fewer than n only when the file has fewer distinct texts).
        选出的题目，最短在前（只有文件里不同题干不足 n 道时才少于 n）。

    State effect / 状态影响：
        None.
        无。
    """
    candidates = [q for q in questions if max_chars is None or len(q.question) <= max_chars]
    out: list[Question] = []
    seen: set[str] = set()
    for q in sorted(candidates, key=lambda q: len(q.question)):
        key = normalize_text(q.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n:
            break
    return out


def run_pass(client: DeepSeekClient, name: str, items: list[tuple[Question, str]], *, sleep_s: float, first_call: bool) -> list[dict[str, Any]]:
    """Send every (question, prompt) of one pass in order, sleep_s apart, and record the provider usage per call.
    按序发送一个 pass 的每个（题目，提示词），间隔 sleep_s，逐次记录 API 返回的 usage。

    Input / 输入：
        client: The accounting client.
        client：记账客户端。

        name: Pass name (ledger label 'probe_<name>').
        name：pass 名（台账标签 'probe_<name>'）。

        items: (question, prompt) pairs.
        items：（题目，提示词）对。

        sleep_s: Seconds between calls.
        sleep_s：调用间隔秒数。

        first_call: Whether this is the very first call of the probe (no sleep before it).
        first_call：是否为探针的第一次调用（其前不等待）。

    Output / 输出：
        One row per call: pass, i, question_id, prompt_chars, prompt_tokens, cache_hit_tokens, cache_miss_tokens,
        completion_tokens, latency_ms, tier, status, pred.
        每次调用一行：pass、i、question_id、prompt_chars、prompt_tokens、cache_hit_tokens、cache_miss_tokens、
        completion_tokens、latency_ms、tier、status、pred。

    State effect / 状态影响：
        Network calls; sleeps.
        网络调用；等待。
    """
    rows: list[dict[str, Any]] = []
    for i, (q, prompt) in enumerate(items):
        if not (first_call and i == 0):
            time.sleep(sleep_s)
        res = ask(client, q, prompt, arm=f"probe_{name}")
        last = res.calls[-1] if res.calls else None
        rows.append(
            {
                "pass": name, "i": i, "question_id": q.id, "prompt_chars": len(prompt),
                "prompt_tokens": last.prompt_tokens if last else None, "cache_hit_tokens": last.cache_hit_tokens if last else None,
                "cache_miss_tokens": last.cache_miss_tokens if last else None, "completion_tokens": last.completion_tokens if last else None,
                "latency_ms": round(last.latency_ms, 1) if last else None, "tier": last.rate_tier if last else None,
                "status": res.status, "attempts": len(res.calls), "pred": res.letter,
            }
        )  # fmt: skip
        print(f"  {name} #{i} q{q.id}: prompt={rows[-1]['prompt_tokens']} hit={rows[-1]['cache_hit_tokens']} "
              f"miss={rows[-1]['cache_miss_tokens']} latency_ms={rows[-1]['latency_ms']} status={res.status}", flush=True)
    return rows


def pass_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-pass totals: token sums, cache-hit share, effective CNY per call (off-peak / peak) from the measured
    hit / miss / output counts, the no-cache baseline (every prompt token at the cache-miss price) and the saving.
    每 pass 汇总：token 总和、缓存命中占比、按实测命中 / 未命中 / 输出计算的每次调用有效人民币（非高峰 / 高峰）、
    无缓存基线（所有提示词 token 按未命中价）与节省比例。

    Input / 输入：
        rows: run_pass rows.
        rows：run_pass 的行。

    Output / 输出：
        The dict.
        字典。

    State effect / 状态影响：
        None.
        无。
    """
    priced = [r for r in rows if r["cache_hit_tokens"] is not None and r["cache_miss_tokens"] is not None and r["completion_tokens"] is not None]
    n = len(priced)
    hit = sum(r["cache_hit_tokens"] for r in priced)
    miss = sum(r["cache_miss_tokens"] for r in priced)
    out_tok = sum(r["completion_tokens"] for r in priced)
    prompt = sum((r["prompt_tokens"] or 0) for r in priced)
    result: dict[str, Any] = {
        "calls": len(rows), "priced_calls": n, "prompt_tokens": prompt, "cache_hit_tokens": hit, "cache_miss_tokens": miss,
        "completion_tokens": out_tok, "cache_hit_share": round(hit / prompt, 4) if prompt else None,
        "calls_with_any_hit": sum(1 for r in priced if r["cache_hit_tokens"]),
    }  # fmt: skip
    for tier in ("offpeak", "peak"):
        eff = cny_cost(hit, miss, out_tok, tier)
        base = cny_cost(0, prompt, out_tok, tier)
        result[f"cny_per_call_{tier}"] = round(eff / n, 6) if (eff is not None and n) else None
        result[f"cny_per_call_no_cache_{tier}"] = round(base / n, 6) if (base is not None and n) else None
        result[f"saving_{tier}"] = round(1 - eff / base, 4) if (eff is not None and base) else None
    return result


def markdown(report: dict[str, Any]) -> str:
    """Render probe.md: the per-call table and the per-pass cost lines.
    渲染 probe.md：逐次调用表与每 pass 费用行。

    Input / 输入：
        report: The probe report.
        report：探针报告。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """

    def f(v: Any, digits: int = 6) -> str:
        return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))

    lines = [
        "# DeepSeek prompt-cache probe / 提示词缓存探针",
        "",
        f"card: `{report['card_path']}` ({report['card_chars']} chars); questions={report['n_questions']} repeat={report['repeat']} "
        f"sleep_s={report['sleep_s']}; started {report['started_utc']}; CNY per 1M: {json.dumps(PRICES_CNY_PER_M)}",
        "",
        f"pass c question: {report['pass_c_question_id']} (first call cold for the question suffix: {report['pass_c_first_call_cold_for_question']}); "
        f"{report['pass_c_note']}",
        "",
        "| pass | i | question | prompt tokens | cache hit | cache miss | completion | latency ms | tier | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["calls"]:
        lines.append(f"| {r['pass']} | {r['i']} | {r['question_id']} | {f(r['prompt_tokens'])} | {f(r['cache_hit_tokens'])} | "
                     f"{f(r['cache_miss_tokens'])} | {f(r['completion_tokens'])} | {f(r['latency_ms'], 1)} | {r['tier']} | {r['status']} |")
    lines += [
        "",
        "## Effective CNY per call / 每次调用有效人民币",
        "",
        "| pass | calls | cache-hit share | calls with any hit | CNY/call off-peak (no-cache) | saving | CNY/call peak (no-cache) | saving |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in report["passes"].items():
        lines.append(f"| {name} | {c['calls']} | {f(c['cache_hit_share'], 4)} | {c['calls_with_any_hit']} | "
                     f"{f(c['cny_per_call_offpeak'])} ({f(c['cny_per_call_no_cache_offpeak'])}) | {f(c['saving_offpeak'], 4)} | "
                     f"{f(c['cny_per_call_peak'])} ({f(c['cny_per_call_no_cache_peak'])}) | {f(c['saving_peak'], 4)} |")
    lines += ["", f"ledger USD: {report['cost_usd']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments (None -> sys.argv[1:]).
        argv：参数（None -> sys.argv[1:]）。

    Output / 输出：
        Exit code 0.
        退出码 0。

    State effect / 状态影响：
        Network calls; writes --out.
        网络调用；写 --out。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card", type=Path, required=True, help="text file holding one compiled card / 存有一张编译卡片的文本文件")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="distinct questions per pass / 每 pass 的不同题目数")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="repetitions of the identical prompt in pass c / pass c 中相同提示词的重复次数")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S, help="seconds between calls / 调用间隔秒数")
    parser.add_argument("--max-question-chars", type=int, default=None,
                        help="only questions with at most this many chars are candidates (default: the n shortest) / 只取题干不超过此字符数的题目（默认取最短的 n 道）")  # fmt: skip
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    card = args.card.read_text(encoding="utf-8")
    picked = pick_questions(load_questions(args.data_root / QA_FILENAME), args.n + 1, max_chars=args.max_question_chars)
    if not picked:
        print("no questions / 没有题目")
        return 2
    qs, q_repeat, pass_c_cold = split_questions(picked, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    load_api_key(ENV_PATH)
    meter = Meter()
    client = DeepSeekClient(meter)
    plans: dict[str, list[tuple[Question, str]]] = {
        PASSES[0]: [(q, build_prompt(q, card, "memories_first")) for q in qs],
        PASSES[1]: [(q, build_prompt(q, card, "question_first")) for q in qs],
        PASSES[2]: [(q_repeat, build_prompt(q_repeat, card, "memories_first"))] * max(1, args.repeat),
    }
    calls: list[dict[str, Any]] = []
    for k, (name, items) in enumerate(plans.items()):
        print(f"[{name}] {len(items)} calls", flush=True)
        calls.extend(run_pass(client, name, items, sleep_s=args.sleep, first_call=(k == 0)))
    report: dict[str, Any] = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "card_path": str(args.card),
        "card_chars": len(card), "n_questions": len(qs), "question_ids": [q.id for q in qs],
        "question_chars": {q.id: len(q.question) for q in qs}, "question_selection": "shortest distinct texts (stable by file order)",
        "pass_c_question_id": q_repeat.id, "pass_c_first_call_cold_for_question": pass_c_cold,
        "pass_c_note": ("pass c uses the (N+1)-th shortest question, never sent in passes a / b: call #0 is a cold miss for the question "
                        "suffix (the card prefix may be warm from pass a), calls #1.. are exact repeats" if pass_c_cold else
                        "fewer than N+1 distinct questions: pass c repeats qs[0], whose identical prompt pass a call #0 already sent, "
                        "so call #0 is expected warm"),
        "max_question_chars": args.max_question_chars, "repeat": args.repeat, "sleep_s": args.sleep,
        "prices_cny_per_m": PRICES_CNY_PER_M, "calls": calls,
        "passes": {name: pass_costs([r for r in calls if r["pass"] == name]) for name in plans},
        "cost_usd": round(sum(c.cost_usd or 0.0 for c in meter.calls), 6),
        "note": "cache_hit_tokens / cache_miss_tokens are the provider's usage fields; the probe controls nothing about DeepSeek's "
                "cache (granularity, TTL, warm-up), so pass differences are observations at this time of day.",
    }  # fmt: skip
    meter.write_jsonl(args.out / "calls.jsonl")
    (args.out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "probe.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
