"""Side-by-side five-dimension comparison of two real-LLM EgoLife runs (e.g. mem0_real vs elite_rows / elite_hybrid / elite_card),
recomputed from their raw result files.
两次真实 LLM EgoLife 运行（如 mem0_real 对 elite_rows / elite_hybrid / elite_card）的五维并排对照，从原始结果文件重算。

Usage / 用法：
    python compare_real.py --run results/real/mem0_full204 --run results/real/elite_full204_r3 \
        --run results/real/elite_compact_full204 --out ../../docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

TYPES = ("EventRecall", "TaskMaster", "RelationMap", "EntityLog", "HabitInsight")
# 元 / 百万 token，deepseek-flash 官方价（2026-09-11）/ CNY per 1M tokens
CNY = {"offpeak": {"hit": 0.02, "miss": 1.0, "out": 4.0}, "peak": {"hit": 0.04, "miss": 2.0, "out": 8.0}}


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _q(v: list[float], p: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]


def _ms(v: list[float]) -> str:
    return "未实测" if not v else f"{statistics.median(v):.1f} / {_q(v, 0.95):.1f}"


def _pct(a: float, n: float) -> str:
    return "—" if not n else f"{100.0 * a / n:.1f}%"


class Run:
    """One run directory; arms discovered from questions_*.jsonl. / 一个运行目录；臂由 questions_*.jsonl 发现。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.summary = json.loads((root / "summary.json").read_text(encoding="utf-8")) if (root / "summary.json").exists() else {}
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {}
        self.arms: dict[str, list[dict]] = {}
        for p in sorted(root.glob("questions_*.jsonl")):
            self.arms[p.stem[len("questions_"):]] = _rows(p)
        self.calls = _rows(root / "calls.jsonl")
        self.retain: dict[str, list[dict]] = {}
        for p in sorted(root.glob("retain_*.jsonl")):
            self.retain[p.stem[len("retain_"):]] = _rows(p)
        self.windows = len(_rows(root / "windows.jsonl"))

    def engine_of(self, arm: str) -> str:
        return "elite" if arm.startswith("elite") else arm

    def extract_calls(self, engine: str) -> list[dict]:
        return [c for c in self.calls if c.get("stage") == "extract" and (c.get("arm") == engine or (engine == "elite" and str(c.get("arm", "")).startswith("elite")))]

    def answer_calls(self, arm: str) -> list[dict]:
        return [c for c in self.calls if c.get("stage") == "answer" and c.get("arm") == arm]

    def cny(self, calls: list[dict], tier: str) -> float:
        p = CNY[tier]
        return sum(((c.get("cache_hit_tokens") or 0) * p["hit"] + (c.get("cache_miss_tokens") or 0) * p["miss"] + (c.get("completion_tokens") or 0) * p["out"]) / 1e6 for c in calls)

    def usd(self, calls: list[dict]) -> float:
        return sum(float(c.get("cost_usd") or 0) for c in calls)


def arm_block(run: Run, arm: str) -> dict:
    rows = run.arms[arm]
    eng = run.engine_of(arm)
    ret = run.retain.get(eng) or run.retain.get(arm) or []
    ex = run.extract_calls(eng)
    an = run.answer_calls(arm)
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    by_type = {t: (sum(1 for r in rows if r.get("type") == t and r.get("correct")), sum(1 for r in rows if r.get("type") == t)) for t in TYPES}
    hit = sum(1 for r in rows if r.get("hit_batch", r.get("hit@20_batch")))
    cats = Counter(r.get("error_category") or "none" for r in rows if not r.get("correct"))
    t = [r.get("timing_ms") or {} for r in rows]
    atoms = sum(int(r.get("added") or 0) for r in ret)
    windows = sum(int(r.get("windows") or 0) for r in ret) or run.windows
    return {
        "n": n, "correct": correct, "acc": _pct(correct, n), "by_type": by_type, "hit20": _pct(hit, n), "cats": dict(cats),
        "invalid": sum(1 for r in rows if r.get("pred") not in ("A", "B", "C", "D")),
        "empty": sum(1 for r in rows if not (r.get("raw_output") or "").strip()),
        "api_fail": sum(1 for r in rows if r.get("answer_status") not in (None, "ok")),
        "retain_model_ms_batch": [float(r.get("model_ms") or 0) for r in ret],
        "retain_local_ms_batch": [float(r.get("local_ms") or 0) for r in ret],
        "retain_wall_step": [float(d["retain_wall"]) for d in t if d.get("retain_attempts")],
        "recall_ms": [float(d.get("recall_local") or 0) for d in t if "recall_local" in d],
        "answer_ms": [float(c.get("latency_ms") or 0) for c in an],
        "e2e_no_retain": [float(d.get("recall_local", 0)) + float(d.get("context_local", 0)) + float(d.get("answer_wall", 0)) for d in t],
        "ex_attempts": len(ex), "ex_in": sum(int(c.get("prompt_tokens") or 0) for c in ex), "ex_hit": sum(int(c.get("cache_hit_tokens") or 0) for c in ex),
        "ex_miss": sum(int(c.get("cache_miss_tokens") or 0) for c in ex), "ex_out": sum(int(c.get("completion_tokens") or 0) for c in ex),
        "ex_truncated": sum(1 for c in ex if c.get("finish_reason") == "length"), "ex_fail": sum(1 for c in ex if c.get("status") != "ok"),
        "an_in": sum(int(c.get("prompt_tokens") or 0) for c in an), "an_out": sum(int(c.get("completion_tokens") or 0) for c in an), "an_fail": sum(1 for c in an if c.get("status") != "ok"),
        "ex_usd": run.usd(ex), "an_usd": run.usd(an), "ex_cny_off": run.cny(ex, "offpeak"), "ex_cny_peak": run.cny(ex, "peak"), "an_cny_off": run.cny(an, "offpeak"), "an_cny_peak": run.cny(an, "peak"),
        "tiers": dict(Counter(c.get("rate_tier") for c in ex)), "atoms": atoms, "windows": windows, "windows_lost": sum(int(r.get("windows_lost") or 0) for r in ret),
        "batches": len(ret), "adds": sum(int(r.get("adds") or 1) for r in ret),
        "db_bytes": ((run.summary.get("engines") or {}).get(eng) or {}).get("stats", {}).get("db_bytes") if run.summary else None,
    }


def _controls(run: Run) -> dict[str, Any]:
    """The control variables of one run as recorded in its manifest / summary (embedder identity, text language, CJK switch, batch and window counts, question count).
    一次运行在 manifest / summary 中记录的控制变量（嵌入器身份、文本语言、CJK 开关、批次与窗口数、题数）。
    """
    man = run.manifest or (run.summary.get("manifest") or {})
    emb = man.get("embedding") or run.summary.get("embedding") or {}
    model, dims = emb.get("model", "BAAI/bge-small-en-v1.5"), emb.get("dims", 384)
    return {
        "embedder": f"{model}@{dims}",
        "text_lang": man.get("text_lang") or run.summary.get("text_lang") or "en",
        "cjk": man.get("cjk", run.summary.get("cjk")) or "库默认",
        "batches": man.get("batches", 290),
        "windows": man.get("windows", 4851),
        "questions": man.get("limit") or man.get("questions_in_file") or 204,
    }


def unique_run_roots(runs: list[tuple[Run, str]]) -> list[Run]:
    """One Run per result directory in first-seen order: a run contributes one column per arm, but its control
    variables (embedder, text language, CJK switch, batches, questions) are stated once.
    每个结果目录一个 Run，按首次出现顺序：一次运行按臂各出一列，但其控制变量（嵌入器、文本语言、CJK 开关、批次、题数）只陈述一次。
    """
    return list({r.root: r for r, _arm in runs}.values())


def render(runs: list[tuple[Run, str]], note: list[str]) -> str:
    blocks = [(r.root.name, arm, arm_block(r, arm)) for r, arm in runs]
    runs = list(runs)
    L: list[str] = []
    A = L.append
    A("# EgoLife 真实 LLM 同口径对照：mem0 vs HMS Elite（2026-09-11）")
    A("")
    unique_runs = unique_run_roots(runs)
    controls = [_controls(r) for r in unique_runs]
    same = len({json.dumps(c, ensure_ascii=False, sort_keys=True) for c in controls}) <= 1
    c0 = controls[0] if controls else _controls(Run(Path(".")))
    if same:
        A(f"控制变量：同一题库（{c0['questions']} 题）、同一窗口切分与批次（{c0['batches']} 批 / {c0['windows']:,} 窗口）、同一流式时间线、同一抽取模型与参数（deepseek-flash，temperature 与 max_tokens 相同）、"
          f"同一本地嵌入器（fastembed {c0['embedder']}）、文本语言 {c0['text_lang']}、CJK 开关 {c0['cjk']}、同一阅读器提示词与参数（四选一，temperature 0，max_tokens 8）、同一 top-k=20 与阈值、并发 1。"
          "不同点只在引擎本身：mem0 用其官方 Memory（milvus-lite 混合检索，add 无时间戳参数，v2 写入只 ADD）；Elite 用 sqlite 索引与 compact 抽取契约、`reference_date` 有效期判定。")
    else:
        # Runs differ in a control variable (EgoLife-ZH A / B against the English baseline): say so per run instead of
        # claiming the same embedder / text.
        # 各运行的控制变量不同（EgoLife-ZH 的 A / B 对英文基线）：逐运行说明，而不是声称同一嵌入器 / 文本。
        A("控制变量不同：" + "；".join(
            f"{r.root.name}：嵌入器 {c['embedder']}，文本语言 {c['text_lang']}，CJK 开关 {c['cjk']}，{c['batches']} 批 / {c['windows']:,} 窗口，{c['questions']} 题"
            for r, c in zip(unique_runs, controls)
        ) + "。其余相同：同一流式时间线、同一抽取模型与参数（deepseek-flash）、同一阅读器提示词与参数（四选一，temperature 0，max_tokens 8）、同一 top-k=20 与阈值、并发 1。")
    A("所有数字由 questions_*.jsonl / calls.jsonl / retain_*.jsonl 重算；费用同时给 API 实际账单（USD，按请求时刻档位）与人民币同口径重算（全部按空闲 / 全部按高峰）。")
    A("")
    def _label(run: Run, arm: str) -> str:
        eng = (run.summary.get("engines") or {}).get(run.engine_of(arm)) or {}
        contract = (eng.get("stats") or {}).get("extraction_output") or eng.get("extraction_output") or (run.manifest.get(run.engine_of(arm)) or {}).get("extraction_output")
        return f"{arm}<br>`{run.root.name}`" + (f"<br>{contract} 契约" if contract else "")
    hdr = "| 指标 | " + " | ".join(_label(r, arm) for r, arm in runs) + " |"
    sep = "|---|" + "---|" * len(blocks)
    A("## 1. 准确率")
    A("")
    A(hdr)
    A(sep)
    A("| 答对 / 204 | " + " | ".join(f"{b['correct']} / {b['n']}" for _, _, b in blocks) + " |")
    A("| 准确率 | " + " | ".join(f"**{b['acc']}**" for _, _, b in blocks) + " |")
    A("| hit@20（gold 批次进入返回列表） | " + " | ".join(b["hit20"] for _, _, b in blocks) + " |")
    for t in TYPES:
        A(f"| {t} | " + " | ".join(f"{b['by_type'][t][0]} / {b['by_type'][t][1]} = {_pct(*b['by_type'][t])}" for _, _, b in blocks) + " |")
    A("| 失败归类：无据 / 有据答错 | " + " | ".join(f"{b['cats'].get('no_evidence_retrieved', 0)} / {b['cats'].get('evidence_present_wrong_answer', 0)}" for _, _, b in blocks) + " |")
    A("| 无效输出 / 空答 / API 失败 | " + " | ".join(f"{b['invalid']} / {b['empty']} / {b['api_fail']}" for _, _, b in blocks) + " |")
    A("")
    A("## 2. 延迟（ms，中位 / P95）")
    A("")
    A(hdr)
    A(sep)
    A("| 写入·每批模型 | " + " | ".join(_ms(b["retain_model_ms_batch"]) for _, _, b in blocks) + " |")
    A("| 写入·每批本地 | " + " | ".join(_ms(b["retain_local_ms_batch"]) for _, _, b in blocks) + " |")
    A("| 写入·每时间步墙钟 | " + " | ".join(_ms(b["retain_wall_step"]) for _, _, b in blocks) + " |")
    A("| 检索·本地 | " + " | ".join(_ms(b["recall_ms"]) for _, _, b in blocks) + " |")
    A("| 回答·模型 | " + " | ".join(_ms(b["answer_ms"]) for _, _, b in blocks) + " |")
    A("| 端到端（不含写入） | " + " | ".join(_ms(b["e2e_no_retain"]) for _, _, b in blocks) + " |")
    A("")
    A("## 3. Token（API 实际 usage）")
    A("")
    A(hdr)
    A(sep)
    A("| 抽取请求数（含拆分重提） | " + " | ".join(str(b["ex_attempts"]) for _, _, b in blocks) + " |")
    A("| 抽取输入 token（缓存命中 / 未命中） | " + " | ".join(f"{b['ex_in']:,}（{b['ex_hit']:,} / {b['ex_miss']:,}）" for _, _, b in blocks) + " |")
    A("| 抽取输出 token | " + " | ".join(f"{b['ex_out']:,}" for _, _, b in blocks) + " |")
    A("| 每窗口 输入 / 输出 token | " + " | ".join(f"{b['ex_in'] / b['windows']:.0f} / {b['ex_out'] / b['windows']:.0f}" if b["windows"] else "—" for _, _, b in blocks) + " |")
    A("| 回答输入 / 输出 token | " + " | ".join(f"{b['an_in']:,} / {b['an_out']:,}" for _, _, b in blocks) + " |")
    A("| 嵌入 | " + " | ".join("本地，无 API token" for _ in blocks) + " |")
    A("")
    A("## 4. 金额与效率")
    A("")
    A(hdr)
    A(sep)
    A("| 实际账单 USD（写入 + 回答） | " + " | ".join(f"${b['ex_usd'] + b['an_usd']:.4f}（写入 ${b['ex_usd']:.4f}）" for _, _, b in blocks) + " |")
    A("| 抽取请求档位 | " + " | ".join(str(b["tiers"]) for _, _, b in blocks) + " |")
    A("| 人民币，全按空闲价（写入 + 回答） | " + " | ".join(f"¥{b['ex_cny_off'] + b['an_cny_off']:.3f}" for _, _, b in blocks) + " |")
    A("| 人民币，全按高峰价 | " + " | ".join(f"¥{b['ex_cny_peak'] + b['an_cny_peak']:.3f}" for _, _, b in blocks) + " |")
    A("| 每题费用（空闲价） | " + " | ".join(f"¥{(b['ex_cny_off'] + b['an_cny_off']) / b['n']:.5f}" if b["n"] else "—" for _, _, b in blocks) + " |")
    A("| 每答对一题（空闲价） | " + " | ".join(f"¥{(b['ex_cny_off'] + b['an_cny_off']) / b['correct']:.5f}" if b["correct"] else "—" for _, _, b in blocks) + " |")
    A("| 写入摊销：每窗口（空闲价） | " + " | ".join(f"¥{b['ex_cny_off'] / b['windows']:.6f}" if b["windows"] else "—" for _, _, b in blocks) + " |")
    A("| 存入记忆条数 / 每窗口 | " + " | ".join(f"{b['atoms']:,} / {b['atoms'] / b['windows']:.2f}" if b["windows"] else "—" for _, _, b in blocks) + " |")
    A("| 每条记忆写入费用（空闲价） | " + " | ".join(f"¥{b['ex_cny_off'] / b['atoms']:.6f}" if b["atoms"] else "—" for _, _, b in blocks) + " |")
    A("| 库体积 | " + " | ".join(f"{b['db_bytes'] / 1e6:.1f} MB" if b.get("db_bytes") else "未记录" for _, _, b in blocks) + " |")
    A("")
    A("## 5. 稳定性")
    A("")
    A(hdr)
    A(sep)
    A("| 批次 / add 请求（含拆分） | " + " | ".join(f"{b['batches']} / {b['adds']}" for _, _, b in blocks) + " |")
    A("| 撞 max_tokens 的尝试 | " + " | ".join(str(b["ex_truncated"]) for _, _, b in blocks) + " |")
    A("| 抽取请求失败 / 回答请求失败 | " + " | ".join(f"{b['ex_fail']} / {b['an_fail']}" for _, _, b in blocks) + " |")
    A("| 窗口丢失 | " + " | ".join(f"{b['windows_lost']} / {b['windows']}" for _, _, b in blocks) + " |")
    A("")
    if note:
        A("## 6. 口径说明")
        A("")
        for x in note:
            A(f"- {x}")
        A("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True, help="run dirs in column order (repeatable) / 运行目录，按列序可重复")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args()
    runs = []
    for root in args.run:
        r = Run(root)
        runs += [(r, arm) for arm in r.arms]
    text = render(runs, args.note)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
