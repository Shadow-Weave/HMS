"""Five-dimension report for a real-LLM EgoLife run, recomputed from the raw result files.
真实 LLM EgoLife 运行的五维报告，从原始结果文件独立重算。

Reads a `run_real.py` output directory (questions_*.jsonl, calls.jsonl, retain_*.jsonl, windows.jsonl, manifest.json,
summary.json) and writes a Chinese markdown report covering the user's rubric: accuracy, latency, token cost, money
and efficiency, stability. Every number is recomputed here from the jsonl rows; summary.json is only used for
cross-checking and the differences (if any) are listed at the end.
读取 `run_real.py` 的输出目录（questions_*.jsonl、calls.jsonl、retain_*.jsonl、windows.jsonl、manifest.json、
summary.json），写出覆盖用户评测口径的中文 markdown 报告：准确率、延迟、Token 成本、金额与效率、稳定性。
所有数字都在这里由 jsonl 行重算；summary.json 只用于交叉核对，差异（如有）列在文末。

Usage / 用法：
    python report_real.py --run results/real/elite_full204_r3 --out ../../docs/design/EGOLIFE_REAL_LLM_ELITE_2026-09-10.md
    python report_real.py --run results/real/elite_full204_r3 --baseline results/real/elite_full204 --out ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Default Elite labels since 2026-09-18 (run_real.ENGINE_LABELS); elite_card is optional (--with-card) and reported
# only when its questions file exists, so old and new result directories both parse.
# 2026-09-18 起 Elite 的默认标签（run_real.ENGINE_LABELS）；elite_card 可选（--with-card），只在其逐题文件存在时报告，
# 旧结果目录与新结果目录都能解析。
ARMS = ("elite_rows", "elite_hybrid")
OPTIONAL_ARMS = ("elite_card",)
TYPES = ("EventRecall", "TaskMaster", "RelationMap", "EntityLog", "HabitInsight")


def _load_jsonl(path: Path) -> list[dict]:
    """Rows of a jsonl file; [] when the file is missing. / jsonl 文件的行；文件缺失时为 []。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(x: float, n: float) -> str:
    """Percent string with one decimal; '—' for an empty denominator. / 一位小数的百分比；分母为空时为 '—'。"""
    return "—" if not n else f"{100.0 * x / n:.1f}%"


def _q(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in [0, 1]); None for no values. / 最近秩分位数（p ∈ [0, 1]）；无值时为 None。"""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _ms(values: list[float]) -> str:
    """'median / P95 / max (n)' in ms. / 毫秒的「中位 / P95 / 最大 (n)」。"""
    if not values:
        return "未实测"
    return f"{statistics.median(values):.1f} / {_q(values, 0.95):.1f} / {max(values):.1f} (n={len(values)})"


def _usd(x: float | None) -> str:
    """USD with 6 decimals; '未实测' for None. / 6 位小数的美元；None 为 '未实测'。"""
    return "未实测" if x is None else f"${x:.6f}"


def _fmt_int(x: int | None) -> str:
    """Thousands-grouped integer; '未实测' for None. / 千分位整数；None 为 '未实测'。"""
    return "未实测" if x is None else f"{x:,}"


def _dict(v: Any) -> dict:
    """A dict field that may have been serialised as its repr. / 可能被序列化成 repr 的字典字段。"""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            import ast

            try:
                return ast.literal_eval(v)
            except Exception:
                return {}
    return {}


class Run:
    """All raw rows of one run directory. / 一个运行目录的全部原始行。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        self.summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}
        self.arms = tuple(a for a in ARMS + OPTIONAL_ARMS if (root / f"questions_{a}.jsonl").exists()) or ARMS
        self.questions = {arm: _load_jsonl(root / f"questions_{arm}.jsonl") for arm in self.arms}
        self.calls = _load_jsonl(root / "calls.jsonl")
        self.retain = _load_jsonl(root / "retain_elite.jsonl")
        self.windows = _load_jsonl(root / "windows.jsonl")
        self.batches = json.loads((root / "batches.json").read_text()) if (root / "batches.json").exists() else []

    def batch_storage(self) -> dict[str, str]:
        """batch_id → 'stored' | 'partial' | 'lost' | 'unknown' from the retain log. Runs recorded before the harness
        counted windows (no `windows_lost` field) fall back to `added == 0` as the 'lost' proxy and never say 'partial'.
        由 retain 记录得到 batch_id → 'stored' | 'partial' | 'lost' | 'unknown'。评测台记录窗口数之前的运行（没有
        `windows_lost` 字段）用 `added == 0` 作为 'lost' 的代理，且不会给出 'partial'。
        """
        out: dict[str, str] = {}
        for x in self.retain:
            bid = x.get("batch_id")
            if "windows_lost" in x:
                if x.get("windows_stored", 0) == 0:
                    out[bid] = "lost"
                elif x.get("windows_lost", 0) > 0:
                    out[bid] = "partial"
                else:
                    out[bid] = "stored"
            else:
                out[bid] = "lost" if x.get("added", 0) == 0 else "stored"
        return out

    # ---- 准确率 / accuracy -------------------------------------------------------------------------------------
    def accuracy(self, arm: str) -> dict:
        rows = self.questions.get(arm, [])
        n = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        storage = self.batch_storage()

        def _gold_state(r: dict) -> str:
            states = [storage.get(b, "unknown") for b in (r.get("gold_batches") or [])]
            if not states:
                return "no_gold"
            if all(s == "lost" for s in states):
                return "all_lost"
            if all(s == "stored" for s in states):
                return "all_stored"
            return "partial"
        by_type = {}
        for t in TYPES:
            sub = [r for r in rows if r.get("type") == t]
            by_type[t] = (sum(1 for r in sub if r.get("correct")), len(sub))
        cats = Counter(r.get("error_category") or "none" for r in rows if not r.get("correct"))
        gold_kept = [r for r in rows if _gold_state(r) == "all_stored"]
        gold_lost = [r for r in rows if _gold_state(r) == "all_lost"]
        gold_partial = [r for r in rows if _gold_state(r) == "partial"]
        hit = sum(1 for r in rows if r.get("hit@20_batch", r.get("hit_batch")))
        invalid = sum(1 for r in rows if r.get("pred") not in ("A", "B", "C", "D"))
        empty = sum(1 for r in rows if not (r.get("raw_output") or "").strip())
        api_fail = sum(1 for r in rows if r.get("answer_status") not in (None, "ok"))
        return {
            "n": n, "correct": correct, "by_type": by_type, "categories": dict(cats),
            "gold_kept_n": len(gold_kept), "gold_kept_correct": sum(1 for r in gold_kept if r.get("correct")),
            "gold_lost_n": len(gold_lost), "gold_partial_n": len(gold_partial),
            "hit20": hit, "invalid": invalid, "empty": empty, "api_fail": api_fail,
            "failures": [r for r in rows if not r.get("correct")],
        }

    # ---- 延迟 / latency -----------------------------------------------------------------------------------------
    def latency(self, arm: str) -> dict:
        rows = self.questions.get(arm, [])
        t = [_dict(r.get("timing_ms")) for r in rows]
        out: dict[str, list[float]] = defaultdict(list)
        for d in t:
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k].append(float(v))
        retain_steps = [d for d in t if d.get("retain_attempts")]
        out["retain_wall(steps_with_retain)"] = [float(d["retain_wall"]) for d in retain_steps if "retain_wall" in d]
        out["retain_model(steps_with_retain)"] = [float(d["retain_model"]) for d in retain_steps if "retain_model" in d]
        out["retain_local(steps_with_retain)"] = [float(d["retain_local"]) for d in retain_steps if "retain_local" in d]
        out["answer_model(excl_cold_start)"] = [
            float(d["answer_model"]) for d in t if "answer_model" in d and not d.get("answer_cold_start")
        ]
        e2e_no_retain = [
            float(d.get("recall_local", 0)) + float(d.get("context_local", 0)) + float(d.get("answer_wall", 0))
            for d in t
        ]
        out["end_to_end_excluding_retain"] = e2e_no_retain
        cold = sum(1 for r in rows if r.get("answer_cold_start"))
        return {"series": dict(out), "cold_start_questions": cold, "steps_with_retain": len(retain_steps)}

    def call_latency(self) -> dict:
        """Per-stage model latency from the call ledger. / 按阶段的模型请求耗时（来自调用账本）。"""
        out: dict[str, list[float]] = defaultdict(list)
        for c in self.calls:
            key = f"{c.get('stage')}:{c.get('status')}"
            out[key].append(float(c.get("latency_ms") or 0.0))
            if c.get("stage") == "extract" and c.get("finish_reason") == "length":
                out["extract:finish=length"].append(float(c.get("latency_ms") or 0.0))
        return dict(out)

    # ---- token 与费用 / token and cost ----------------------------------------------------------------------------
    def tokens(self) -> dict:
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for c in self.calls:
            key = (c.get("stage") or "?", c.get("arm") or "?")
            a = agg.setdefault(key, {"attempts": 0, "prompt": 0, "completion": 0, "cache_hit": 0, "cache_miss": 0,
                                     "reasoning": 0, "reasoning_known": False, "no_usage": 0, "cost": 0.0,
                                     "unpriced": 0, "by_tier": Counter(), "retries": 0, "status": Counter(),
                                     "finish": Counter()})
            a["attempts"] += 1
            a["status"][c.get("status")] += 1
            a["finish"][c.get("finish_reason")] += 1
            if (c.get("attempt") or 0) > 0:
                a["retries"] += 1
            if c.get("prompt_tokens") is None and c.get("completion_tokens") is None:
                a["no_usage"] += 1
            a["prompt"] += int(c.get("prompt_tokens") or 0)
            a["completion"] += int(c.get("completion_tokens") or 0)
            a["cache_hit"] += int(c.get("cache_hit_tokens") or 0)
            a["cache_miss"] += int(c.get("cache_miss_tokens") or 0)
            if c.get("reasoning_tokens") is not None:
                a["reasoning"] += int(c["reasoning_tokens"])
                a["reasoning_known"] = True
            if c.get("cost_usd") is None:
                a["unpriced"] += 1
            else:
                a["cost"] += float(c["cost_usd"])
                a["by_tier"][c.get("rate_tier") or "?"] += 1
        return agg

    def normalised_cost(self) -> dict:
        """Cost of every priced call re-priced at ONE tier (offpeak and peak), plus the output tokens burnt by attempts that
        hit max_tokens (finish_reason=length) and were discarded before bisection.
        把每个已计价请求按同一档位（非高峰 / 高峰）重算的费用，以及撞 max_tokens（finish_reason=length）后被丢弃的
        尝试所烧掉的输出 token。
        """
        pricing = (((self.summary.get("arms") or {}).get("elite_rows") or {}).get("money") or {}).get("pricing") or {}
        per_1m = pricing.get("usd_per_1m") or {}
        out: dict[str, Any] = {"tiers": {}, "truncated_attempts": 0, "truncated_output_tokens": 0}
        for tier in ("offpeak", "peak"):
            p = per_1m.get(tier)
            if not p:
                continue
            total = extract = 0.0
            for c in self.calls:
                v = ((c.get("cache_hit_tokens") or 0) * p["cache_hit"] + (c.get("cache_miss_tokens") or 0) * p["cache_miss"]
                     + (c.get("completion_tokens") or 0) * p["output"]) / 1e6
                total += v
                if c.get("stage") == "extract":
                    extract += v
            out["tiers"][tier] = {"total": total, "extract": extract}
        trunc = [c for c in self.calls if c.get("stage") == "extract" and c.get("finish_reason") == "length"]
        out["truncated_attempts"] = len(trunc)
        out["truncated_output_tokens"] = sum(int(c.get("completion_tokens") or 0) for c in trunc)
        for tier, p in per_1m.items():
            out[f"truncated_cost_{tier}"] = out["truncated_output_tokens"] * p["output"] / 1e6
        return out

    def embedding(self) -> dict | None:
        """Embedding usage as the harness recorded it (local fastembed → no API tokens). / 嵌入用量（本地 fastembed → 无 API token）。"""
        for arm in self.arms:
            e = (((self.summary.get("arms") or {}).get(arm) or {}).get("tokens") or {}).get("embedding")
            if e:
                return e
        return None

    # ---- 稳定性 / stability ----------------------------------------------------------------------------------------
    def retain_stats(self) -> dict:
        r = self.retain
        lost_batches = [x for x in r if x.get("windows_lost", 0) > 0]
        warn_kinds: Counter = Counter()
        for x in r:
            w = x.get("library_warnings")
            if isinstance(w, str):
                try:
                    w = json.loads(w)
                except Exception:
                    w = [w]
            for m in w or []:
                if "attributed_to" in m:
                    warn_kinds["Observation.attributed_to 校验拒绝（说话人名）"] += 1
                elif "reply is a JSON list" in m:
                    warn_kinds["裸 JSON 列表回复（已按 observations 读取）"] += 1
                elif "reply is not a JSON object" in m:
                    warn_kinds["非对象回复（整批未抽出）"] += 1
                elif "Error parsing" in m:
                    warn_kinds["JSON 解析失败（含截断）"] += 1
                elif "skipped" in m:
                    warn_kinds["条目校验跳过（其它字段）"] += 1
                else:
                    warn_kinds["其它告警"] += 1
        if r and "windows_lost" not in r[0]:
            # 旧格式：用批次表补窗口数，added==0 视为整批丢失 / old format: window counts from batches.json, added==0 = batch lost
            sizes = {b.get("batch_id"): len(b.get("windows") or b.get("window_ids") or []) for b in self.batches} if isinstance(self.batches, list) else {}
            for x in r:
                w = sizes.get(x.get("batch_id"), 0)
                x.setdefault("windows", w)
                x.setdefault("windows_stored", 0 if x.get("added", 0) == 0 else w)
                x.setdefault("windows_lost", w if x.get("added", 0) == 0 else 0)
                x.setdefault("adds", x.get("attempts", 1))
        return {
            "batches": len(r), "adds": sum(x.get("adds", 0) for x in r), "bisected": sum(1 for x in r if x.get("bisected")),
            "truncated_attempts": sum(x.get("truncated_attempts", 0) for x in r),
            "windows": sum(x.get("windows", 0) for x in r), "windows_stored": sum(x.get("windows_stored", 0) for x in r),
            "windows_lost": sum(x.get("windows_lost", 0) for x in r), "lost_batches": len(lost_batches),
            "atoms_added": sum(x.get("added", 0) for x in r), "failed_batches": sum(1 for x in r if not x.get("ok")),
            "warn_kinds": dict(warn_kinds), "wall_ms": [float(x.get("wall_ms") or 0) for x in r],
            "model_ms": [float(x.get("model_ms") or 0) for x in r], "local_ms": [float(x.get("local_ms") or 0) for x in r],
        }


def _pricing_block(run: Run) -> str:
    m = (((run.summary.get("arms") or {}).get("elite_rows") or {}).get("money") or {}).get("pricing") or {}
    if not m:
        return "计价：summary.json 未记录计价表。"
    p = m.get("usd_per_1m", {})
    off, peak = p.get("offpeak", {}), p.get("peak", {})
    return (
        f"计价（{m.get('model')}，USD / 1M token，来源 {m.get('source')}）：非高峰 缓存命中 {off.get('cache_hit')} / 未命中 "
        f"{off.get('cache_miss')} / 输出 {off.get('output')}；高峰 缓存命中 {peak.get('cache_hit')} / 未命中 {peak.get('cache_miss')} / "
        f"输出 {peak.get('output')}。每个请求按其发起时刻所属档位计价；失败与重试请求同样计入。"
    )


def render(run: Run, baseline: Run | None, title: str, notes: list[str] | None = None) -> str:
    man = run.manifest
    lines: list[str] = []
    A = lines.append
    arms = run.arms
    A(f"# {title}")
    A("")
    emb = man.get("embedding") or {}
    embedder_id = f"{emb.get('model', 'BAAI/bge-small-en-v1.5')}@{emb.get('dims', 384)}"
    A(f"运行目录 `{run.root.as_posix()}`；run_id `{man.get('run_id', run.summary.get('run_id'))}`；开始 {man.get('started_utc')}；"
      f"模型 `{man.get('model')}` @ `{man.get('base_url')}`；并发 {man.get('concurrency', 1)}；Python {man.get('python')}；{man.get('platform')}；"
      f"嵌入器 `{embedder_id}`；文本语言 {man.get('text_lang', 'en')}；CJK {man.get('cjk') or '库默认'}。")
    A("")
    A("本报告全部数字由 `questions_*.jsonl` / `calls.jsonl` / `retain_elite.jsonl` 重算（`report_real.py`），"
      "`summary.json` 只做交叉核对；差异列在文末。凡未实测的量明确标「未实测」，不用估算冒充。")
    A("")
    A("## 0. 实验条件")
    A("")
    n_questions = man.get("limit") or max((len(v) for v in run.questions.values()), default=0)
    A(f"- 数据：EgoLife A1_JAKE（`{man.get('data_root', 'data')}`），{n_questions} 题，5 类题型；窗口 ≤400 字符 / ≤60 s，共 "
      f"{len(run.windows):,} 个窗口，{len(run.batches):,} 个批次（每批 ≤25 窗口，一次 `HMS.add(infer=True)`）。")
    A("- 流式协议：按题目 query_time 推进时间线；同一天只放行 query_time 之前的窗口，此前各天全部放行；写入发生在题目之前，"
      "题目只能检索到此刻已入库的记忆。")
    A("- 各回答上下文共用同一份物理写入：`elite_rows`（检索命中的记忆行）、`elite_hybrid`（`HMS.context()`：按范围记忆化的会话卡片前缀 + 带时间戳的相关记忆行，记忆在前作答）"
      "以及可选的 `elite_card`（逐题 HMS 状态卡片，`--with-card`）；写入成本在各臂之间不重复计费。")
    A("- 写入抽取：deepseek-flash 单次复合抽取（`memory_extraction`）；输出 finish_reason=length 的尝试被拒绝，其窗口切片对半重提（递归到单窗口），"
      "每次子请求都是计费请求。")
    A("- 回答：deepseek-flash，temperature 0，max_tokens 8，与 hms_lite-main 相同的选择题提示词；无效字母 / 空输出 / API 失败一律记为答错。")
    A(f"- 嵌入：本地 fastembed `{embedder_id}`（无 API token，只有本地耗时）。检索：`HMS.search(limit=20)`。")
    A(f"- {_pricing_block(run)}")
    if baseline is not None:
        A(f"- 基线对照：`{baseline.root.as_posix()}`（首轮，修复前：截断输出未重提、`attributed_to` 校验拒绝整条观察、裸列表回复整批丢弃）。")
    A("")

    # ---------------- 1. 准确率 ----------------
    A("## 1. 准确率")
    A("")
    A("分母恒为 204 题；检索命中率（hit@20，批次粒度）单列，不当作问答准确率。")
    A("")
    A("「gold 批次全部完整入库的题」由 retain 记录判定（该题全部 gold 批次 windows_lost == 0）；「gold 全丢」= 全部 gold 批次一个窗口都没存进去；"
      "「部分」= 介于两者之间。")
    A("")
    A("| 臂 | 答对 / 204 | 准确率 | gold 批次全部完整入库的题 (n) 上的准确率 | gold 全丢的题 | gold 部分丢的题 | hit@20（批次） | 无效输出 | 空输出 | API 失败 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    accs = {arm: run.accuracy(arm) for arm in arms}
    for arm, a in accs.items():
        A(f"| {arm} | {a['correct']} / {a['n']} | {_pct(a['correct'], a['n'])} | {_pct(a['gold_kept_correct'], a['gold_kept_n'])} (n={a['gold_kept_n']}) "
          f"| {a['gold_lost_n']} | {a['gold_partial_n']} | {_pct(a['hit20'], a['n'])} | {a['invalid']} | {a['empty']} | {a['api_fail']} |")
    if baseline is not None:
        for arm in arms:
            b = baseline.accuracy(arm)
            A(f"| {arm}（首轮基线，丢失以 added==0 代理） | {b['correct']} / {b['n']} | {_pct(b['correct'], b['n'])} | {_pct(b['gold_kept_correct'], b['gold_kept_n'])} (n={b['gold_kept_n']}) "
              f"| {b['gold_lost_n']} | {b['gold_partial_n']} | {_pct(b['hit20'], b['n'])} | {b['invalid']} | {b['empty']} | {b['api_fail']} |")
    A("")
    A("### 1.1 分题型准确率（答对 / n）")
    A("")
    hdr = "| 题型 | " + " | ".join(arms) + (" | " + " | ".join(f"{a}（基线）" for a in arms) if baseline else "") + " |"
    A(hdr)
    A("|---|" + "---|" * (len(arms) * (2 if baseline else 1)))
    for t in TYPES:
        cells = []
        for arm in arms:
            c, n = accs[arm]["by_type"][t]
            cells.append(f"{c} / {n} = {_pct(c, n)}")
        if baseline:
            for arm in arms:
                c, n = baseline.accuracy(arm)["by_type"][t]
                cells.append(f"{c} / {n} = {_pct(c, n)}")
        A(f"| {t} | " + " | ".join(cells) + " |")
    A("")
    A("### 1.2 失败题归类")
    A("")
    A("`no_evidence_retrieved`：检索结果中没有 gold 批次；`evidence_present_wrong_answer`：gold 批次在上下文里但模型选错；"
      "`invalid_output` / `api_error` / `timeout`：输出不是 A–D / 请求失败 / 超时。")
    A("")
    A("| 臂 | " + " | ".join(sorted({k for a in accs.values() for k in a["categories"]})) + " |")
    keys = sorted({k for a in accs.values() for k in a["categories"]})
    A("|---|" + "---|" * len(keys))
    for arm, a in accs.items():
        A(f"| {arm} | " + " | ".join(str(a["categories"].get(k, 0)) for k in keys) + " |")
    A("")
    A("### 1.3 失败题清单（id / 题型 / gold → 预测 / 归类 / gold 批次是否在返回结果中）")
    A("")
    for arm, a in accs.items():
        A(f"<details><summary>{arm}：{len(a['failures'])} 题</summary>")
        A("")
        A("| id | 题型 | gold → pred | 原始输出 | 归类 | hit@20 |")
        A("|---|---|---|---|---|---|")
        for r in a["failures"]:
            A(f"| {r.get('id')} | {r.get('type')} | {r.get('gold')} → {r.get('pred')} | `{(r.get('raw_output') or '')[:12]}` | {r.get('error_category')} | {r.get('hit@20_batch', r.get('hit_batch'))} |")
        A("")
        A("</details>")
        A("")

    # ---------------- 2. 延迟 ----------------
    A("## 2. 延迟（毫秒，中位 / P95 / 最大，n）")
    A("")
    A("模型请求耗时与本地处理耗时分开报；并发固定为 1；冷启动（首个请求）与重试单独注明。写入延迟按「时间线步」统计：一步 = 某题之前放行的全部批次。")
    A("")
    A("| 阶段 | 口径 | " + " | ".join(arms) + " |")
    A("|---|---|" + "---|" * len(arms))
    lat = {arm: run.latency(arm) for arm in arms}
    stage_rows = [
        ("写入（含抽取模型）", "retain_wall(steps_with_retain)", "墙钟，每步"),
        ("写入·模型", "retain_model(steps_with_retain)", "抽取尝试耗时之和（含重试 / 拆分子请求）"),
        ("写入·本地", "retain_local(steps_with_retain)", "嵌入 + 落库 + 编译"),
        ("检索·本地", "recall_local", "HMS.search(limit=20)"),
        ("上下文构建·本地", "context_local", "拼接记忆行 / 状态卡片"),
        ("回答·模型", "answer_model", "deepseek-flash 一次请求"),
        ("回答·模型（去冷启动）", "answer_model(excl_cold_start)", "剔除首个请求"),
        ("回答·本地", "answer_local", "解析字母"),
        ("端到端（不含写入）", "end_to_end_excluding_retain", "检索 + 上下文 + 回答"),
        ("端到端（含本步写入）", "end_to_end", "写入 + 检索 + 上下文 + 回答"),
    ]
    for label, key, note in stage_rows:
        A(f"| {label} | {note} | " + " | ".join(_ms(lat[arm]["series"].get(key, [])) for arm in arms) + " |")
    A("")
    cold_txt = ", ".join(f"{arm} {lat[arm]['cold_start_questions']}" for arm in arms)
    A(f"冷启动题数：{cold_txt}；含写入的步数：{lat[arms[0]]['steps_with_retain']} / {len(run.questions[arms[0]])}。")
    A("")
    A("### 2.1 按请求的模型耗时（来自调用账本，逐次请求）")
    A("")
    A("| 阶段:状态 | 中位 / P95 / 最大 (n) |")
    A("|---|---|")
    for k, v in sorted(run.call_latency().items()):
        A(f"| {k} | {_ms(v)} |")
    rs = run.retain_stats()
    A("")
    A(f"写入批次粒度（每次 `add` 批次，n={rs['batches']}）：墙钟 {_ms(rs['wall_ms'])}；模型 {_ms(rs['model_ms'])}；本地 {_ms(rs['local_ms'])}。")
    A("")

    # ---------------- 3. Token 成本 / token cost ----------------
    A("## 3. Token 成本（API 返回的实际 usage；无 usage 的请求单列，不估算）")
    A("")
    tok = run.tokens()
    A("| 阶段 | 臂 | 请求数 | 重试数 | 输入 token | 其中缓存命中 | 其中缓存未命中 | 输出 token | 推理 token | 无 usage 的请求 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for (stage, arm), a in sorted(tok.items()):
        reasoning = _fmt_int(a["reasoning"]) if a["reasoning_known"] else "未实测（API 未返回）"
        A(f"| {stage} | {arm} | {a['attempts']} | {a['retries']} | {_fmt_int(a['prompt'])} | {_fmt_int(a['cache_hit'])} | {_fmt_int(a['cache_miss'])} | {_fmt_int(a['completion'])} | {reasoning} | {a['no_usage']} |")
    A("| organize（整理 / 编译） | elite | 0 | 0 | 本地（无 API token） | — | — | — | — | — |")
    emb = run.embedding()
    if emb:
        A(f"| embedding | elite | {emb.get('calls')} 次 / {emb.get('texts')} 段文本 | — | 本地（无 API token） | — | — | — | — | 本地总耗时 {float(emb.get('local_ms_total') or 0):.0f} ms |")
    else:
        A("| embedding | elite | 未实测 | — | 本地（无 API token） | — | — | — | — | — |")
    A("")
    A("> 抽取阶段的输入 token 中，缓存命中部分是系统提示词与重复前缀在 DeepSeek 侧命中 prompt cache 的量；缓存命中与未命中按不同单价计费（见 §0）。")
    A("")

    # ---------------- 4. 金额与效率 ----------------
    A("## 4. 金额与效率（USD，同一计价口径，含失败与重试请求）")
    A("")
    extract_cost = sum(a["cost"] for (s, _), a in tok.items() if s == "extract")
    extract_tiers = Counter()
    for (s, _), a in tok.items():
        if s == "extract":
            extract_tiers.update(a["by_tier"])
    n_windows = len(run.windows) or rs["windows"]
    A("| 臂 | 写入（抽取）费用 | 回答费用 | 总费用 | 每题费用 | 每答对一题的费用 | 写入摊销：每窗口 | 写入摊销：每批次 | 未计价请求 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        ans = tok.get(("answer", arm)) or {"cost": 0.0, "unpriced": 0}
        total = extract_cost + ans["cost"]
        n = accs[arm]["n"] or 1
        correct = accs[arm]["correct"]
        unpriced = ans["unpriced"] + sum(a["unpriced"] for (s, _), a in tok.items() if s == "extract")
        A(f"| {arm} | {_usd(extract_cost)} | {_usd(ans['cost'])} | {_usd(total)} | {_usd(total / n)} | {_usd(total / correct) if correct else '—'} "
          f"| {_usd(extract_cost / n_windows) if n_windows else '—'} | {_usd(extract_cost / len(run.batches)) if run.batches else '—'} | {unpriced} |")
    A("")
    A(f"写入费用两臂共享一次（同一物理库），上表在每臂各计一次以便得到该臂独立部署时的总价；两臂合并跑一次的真实总账 = 写入 {_usd(extract_cost)} + 两臂回答之和。")
    A(f"抽取请求按档位：{dict(extract_tiers)}（peak = 工作日 UTC 01–04 与 06–10）。")
    A("")
    A("### 4.1 同一计价口径（把全部请求按同一档位单价重算；实际账单见上表）")
    A("")
    A("| 轮次 | 实际账单（写入 + 两臂回答） | 全按非高峰：总 / 写入 | 全按高峰：总 / 写入 | 撞 max_tokens 被丢弃的尝试 | 其烧掉的输出 token | 该浪费的费用（非高峰 / 高峰） |")
    A("|---|---|---|---|---|---|---|")
    def _nc_row(label: str, r: Run) -> str:
        n = r.normalised_cost()
        actual = sum(float(c.get("cost_usd") or 0) for c in r.calls)
        o, pk = n["tiers"].get("offpeak", {}), n["tiers"].get("peak", {})
        return (f"| {label} | {_usd(actual)} | {_usd(o.get('total'))} / {_usd(o.get('extract'))} | {_usd(pk.get('total'))} / {_usd(pk.get('extract'))} "
                f"| {n['truncated_attempts']} | {_fmt_int(n['truncated_output_tokens'])} | {_usd(n.get('truncated_cost_offpeak'))} / {_usd(n.get('truncated_cost_peak'))} |")
    A(_nc_row("本轮", run))
    if baseline is not None:
        A(_nc_row("首轮基线", baseline))
    A("")
    A("> 被截断的尝试按合同必须丢弃再对半重提（截断的 JSON 不可解析），因此这部分输出 token 是纯浪费；它随批次大小和抽取输出的冗长度上升，是压缩抽取输出契约要解决的对象。")
    if baseline is not None:
        btok = baseline.tokens()
        b_extract = sum(a["cost"] for (s, _), a in btok.items() if s == "extract")
        A("")
        A("| 对照 | 写入费用 | 写入请求数 | 已存窗口 | 每已存窗口的写入费用 |")
        A("|---|---|---|---|---|")
        brs = baseline.retain_stats()
        A(f"| 本轮 | {_usd(extract_cost)} | {sum(a['attempts'] for (s, _), a in tok.items() if s == 'extract')} | {rs['windows_stored']} / {rs['windows']} | {_usd(extract_cost / rs['windows_stored']) if rs['windows_stored'] else '—'} |")
        A(f"| 首轮基线 | {_usd(b_extract)} | {sum(a['attempts'] for (s, _), a in btok.items() if s == 'extract')} | {brs['windows_stored']} / {brs['windows']} | {_usd(b_extract / brs['windows_stored']) if brs['windows_stored'] else '—'} |")
    A("")

    # ---------------- 5. 稳定性 ----------------
    A("## 5. 稳定性（失败请求保留在结果中）")
    A("")
    A("| 阶段 | 臂 | 请求数 | ok | timeout | http_error | empty | 重试数 | finish_reason=length |")
    A("|---|---|---|---|---|---|---|---|---|")
    for (stage, arm), a in sorted(tok.items()):
        st = a["status"]
        A(f"| {stage} | {arm} | {a['attempts']} | {st.get('ok', 0)} | {st.get('timeout', 0)} | {st.get('http_error', 0)} | {st.get('empty', 0)} | {a['retries']} | {a['finish'].get('length', 0)} |")
    A("")
    A("| 写入指标 | 值 |")
    A("|---|---|")
    A(f"| 批次数 / add 请求数（含拆分子请求） | {rs['batches']} / {rs['adds']} |")
    A(f"| 因输出截断而拆分的批次 / 截断尝试次数 | {rs['bisected']} / {rs['truncated_attempts']} |")
    A(f"| 窗口：计划 / 已入库 / 丢失 | {rs['windows']} / {rs['windows_stored']} / {rs['windows_lost']}（{_pct(rs['windows_lost'], rs['windows'])}） |")
    A(f"| 丢窗口的批次数 / add 失败的批次数 | {rs['lost_batches']} / {rs['failed_batches']} |")
    A(f"| 抽出的原子数 | {rs['atoms_added']:,} |")
    for k, v in sorted(rs["warn_kinds"].items(), key=lambda kv: -kv[1]):
        A(f"| 库端告警：{k} | {v} |")
    for arm, a in accs.items():
        A(f"| {arm}：空答率 / 无效输出率 / 回答请求失败率 | {_pct(a['empty'], a['n'])} / {_pct(a['invalid'], a['n'])} / {_pct(a['api_fail'], a['n'])} |")
    if baseline is not None:
        brs = baseline.retain_stats()
        A(f"| 首轮基线：窗口 计划 / 已入库 / 丢失 | {brs['windows']} / {brs['windows_stored']} / {brs['windows_lost']}（{_pct(brs['windows_lost'], brs['windows'])}） |")
        A(f"| 首轮基线：截断尝试 / 丢窗口批次 | {brs['truncated_attempts']} / {brs['lost_batches']} |")
    A("")

    # ---------------- 6. 交叉核对 ----------------
    A("## 6. 与 summary.json 的交叉核对")
    A("")
    diffs: list[str] = []
    for arm in arms:
        s = ((run.summary.get("arms") or {}).get(arm) or {})
        sa = (s.get("accuracy") or {})
        if sa and sa.get("correct") != accs[arm]["correct"]:
            diffs.append(f"{arm} 答对数：重算 {accs[arm]['correct']} vs summary {sa.get('correct')}")
        sm = (s.get("money") or {})
        ans = tok.get(("answer", arm)) or {"cost": 0.0}
        if sm and abs(float(sm.get("total_cost_usd") or 0) - (extract_cost + ans["cost"])) > 1e-5:
            diffs.append(f"{arm} 总费用：重算 {_usd(extract_cost + ans['cost'])} vs summary {_usd(sm.get('total_cost_usd'))}")
    ledger = run.summary.get("ledger") or {}
    if ledger and ledger.get("attempts") != len(run.calls):
        diffs.append(f"请求数：重算 {len(run.calls)} vs summary {ledger.get('attempts')}")
    if diffs:
        for d in diffs:
            A(f"- {d}")
    else:
        A("- 答对数、总费用、请求数与 summary.json 一致。")
    A("")
    A("## 7. 未实测 / 口径说明")
    A("")
    A("- 推理 token：deepseek-flash 的 usage 若未返回 reasoning_tokens，则标「未实测」，不从输出 token 中反推。")
    A(f"- 嵌入：本地 fastembed `{embedder_id}`，无 API token；只有本地耗时。")
    A("- 并发固定 1，因此 P95 反映串行请求的尾部，不含排队。")
    A("- 高峰 / 非高峰按请求发起时刻的 UTC 判定；同一轮内可能跨档位，费用按各请求实际档位累加。")
    A("- mem0 真实 LLM 臂未运行（按用户指示先只测 HMS Elite）。")
    for n in notes or []:
        A(f"- {n}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run_real.py output directory / run_real.py 输出目录")
    ap.add_argument("--baseline", default=None, help="optional earlier run for a before/after row / 可选：修复前的一轮，作对照行")
    ap.add_argument("--out", required=True, help="markdown path to write / 要写出的 markdown 路径")
    ap.add_argument("--title", default="HMS Elite × EgoLife 真实 LLM 评测（deepseek-flash，仅 Elite）")
    ap.add_argument("--note", action="append", default=[], help="extra caveat lines appended to §7 / 追加到 §7 的口径说明")
    args = ap.parse_args()
    run = Run(Path(args.run))
    baseline = Run(Path(args.baseline)) if args.baseline else None
    text = render(run, baseline, args.title, args.note)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
