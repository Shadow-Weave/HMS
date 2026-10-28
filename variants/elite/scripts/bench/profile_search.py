"""Attribute HMS Elite search latency to its stages (axis SQL, index, hydrate, five signals, fuse, rows) at N = 300 / 3000 / 30000.
把 HMS Elite 检索延迟归因到各阶段（轴 SQL、索引、水合、五路信号、融合、行格式化），规模 N = 300 / 3000 / 30000。
"""

# Purpose (EN): Produce the timing evidence for docs/design/RETRIEVAL_DIAGNOSIS.md (manual timers + cProfile).
# 目的（中）：为 docs/design/RETRIEVAL_DIAGNOSIS.md 生产计时证据（手工计时器 + cProfile）。

# Input (EN): --provider sqlite|milvus, --sizes, --repeat, --out; a hash embedder and infer=False raw message atoms.
# 输入（中）：--provider sqlite|milvus、--sizes、--repeat、--out；哈希嵌入器与 infer=False 的原始消息原子。

# Output (EN): results/profile_search.json plus a markdown table on stdout.
# 输出（中）：results/profile_search.json 以及打印到 stdout 的 markdown 表格。

# State effect (EN): Writes temporary SQLite / milvus-lite files under the bench work directory only.
# 状态影响（中）：只在 bench 工作目录下写入临时 SQLite / milvus-lite 文件。

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import statistics
import sys
import time
from typing import Any, Callable

PROJECT = "/Users/lazy/Documents/Claude Code/hms-elite"
sys.path.insert(0, PROJECT)

import hms_elite.mvmi.query as query_mod  # noqa: E402
import hms_elite.utils.scoring as scoring_mod  # noqa: E402
from hms_elite import HMS  # noqa: E402
from hms_elite.configs.base import HMSConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AS_OF = "2026-06-15T09:00:00Z"
USER = "alice"

# Five representative queries: entity gate, identifier, month (time gate), preference (type gate), CJK recall.
# 五个代表性查询：实体门、标识符、月份（时间门）、偏好（类型门）、中文召回。
QUERIES: list[tuple[str, str]] = [
    ("entity", "who is Sarah"),
    ("identifier", "retry.ts backoff patch"),
    ("month", "what happened in March"),
    ("preference", "what coffee does alice like"),
    ("cjk", "部署 回滚 测试"),
]

NAMES = ["Sarah", "Tom", "Priya", "Wei", "Lucas", "Mina", "Diego", "Anna", "Noah", "Ines"]
FILES = ["src/billing/retry.ts", "src/checkout/cart.py", "api/webhook.go", "docs/design.md", "lib/auth_token.py"]
DRINKS = ["latte", "espresso", "oolong tea", "mocha", "cappuccino"]
ZH_ACTS = ["部署了支付服务", "回滚了昨天的发布", "修复了重试逻辑", "重写了结账页面", "测试了 webhook 超时"]


def gen_texts(n: int) -> list[tuple[str, str]]:
    """Deterministic (text, timestamp) pairs over 18 months; every text is distinct through its item token.
    确定性的 (文本, 时间戳) 对，跨 18 个月；每条文本通过 item 词元互不相同。
    """
    out: list[tuple[str, str]] = []
    for i in range(n):
        # 18-month cycle 2025-01 .. 2026-06 so a "March" gate selects ~1/18 of the corpus.
        # 18 个月循环 2025-01 .. 2026-06，使 "March" 门控选中约 1/18 的语料。
        month_index = i % 18
        year = 2025 + month_index // 12
        month = month_index % 12 + 1
        day = 1 + (i * 7) % 28
        hour = (i * 3) % 24
        ts = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00Z"
        kind = i % 5
        # Sub-choices use i // 5 so they are independent of `kind` (i % 5 would pin each kind to one variant).
        # 子选择用 i // 5，使其与 `kind` 无关（用 i % 5 会把每种 kind 钉死在一个变体上）。
        pick = i // 5
        name = NAMES[i % len(NAMES)]
        if kind == 0:
            text = f"{name} shipped the checkout webhook change item{i} and reviewed the backoff design"
        elif kind == 1:
            text = f"{name} patched {FILES[pick % len(FILES)]} in item{i} to fix the retry backoff regression"
        elif kind == 2:
            text = f"{name} said they like {DRINKS[pick % len(DRINKS)]} these days item{i}"
        elif kind == 3:
            text = f"{name} 今天{ZH_ACTS[pick % len(ZH_ACTS)]}，编号 item{i}"
        else:
            text = f"{name} joined the billing standup and planned the deploy item{i}"
        out.append((text, ts))
    return out


class Timers:
    """Accumulate wall time per bucket for one search call.
    为一次 search 调用按桶累计墙钟时间。
    """

    def __init__(self) -> None:
        self.buckets: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.candidates = 0
        self.axis_selects = 0

    def add(self, name: str, dt: float, n: int = 1) -> None:
        self.buckets[name] = self.buckets.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + n

    def reset(self) -> None:
        self.buckets.clear()
        self.counts.clear()
        self.candidates = 0
        self.axis_selects = 0


TIMERS = Timers()


def timed(bucket: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap fn so its wall time lands in `bucket`.
    包装 fn，使其墙钟时间计入 `bucket`。
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMERS.add(bucket, time.perf_counter() - t0)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def instrument(hms: HMS, provider: str) -> None:
    """Patch the hot path so every stage reports into TIMERS.
    给热路径打补丁，使每个阶段都上报到 TIMERS。
    """
    store = hms.store
    original_execute = store._execute
    original_hydrate = store._hydrate_atoms

    def select_atoms(sql: str, params: Any = ()) -> list:
        t0 = time.perf_counter()
        rows = original_execute(sql, params).fetchall()
        t1 = time.perf_counter()
        atoms = original_hydrate(rows)
        t2 = time.perf_counter()
        TIMERS.add("axis_sql", t1 - t0)
        TIMERS.add("hydrate", t2 - t1, len(rows))
        TIMERS.candidates += len(rows)
        TIMERS.axis_selects += 1
        return atoms

    store._select_atoms = select_atoms  # type: ignore[method-assign]
    store.resolve_entities = timed("entity_lookup", store.resolve_entities)  # type: ignore[method-assign]
    store.entity_degree = timed("entity_lookup", store.entity_degree)  # type: ignore[method-assign]
    hms.embedder.embed = timed("embed", hms.embedder.embed)  # type: ignore[method-assign]
    hms.index.search = timed("index_search", hms.index.search)  # type: ignore[method-assign]
    # Module globals looked up at call time: patching the module attribute is enough.
    # 调用时才查找的模块全局名：给模块属性打补丁即可。
    query_mod.plan_query = timed("plan", query_mod.plan_query)
    query_mod.semantic_scores = timed("semantic_scores_total", query_mod.semantic_scores)
    query_mod.compute_signals = timed("signals_total", query_mod.compute_signals)
    query_mod.fuse = timed("fuse", query_mod.fuse)
    query_mod.score_details = timed("score_details", query_mod.score_details)
    query_mod.atom_to_row = timed("row_format", query_mod.atom_to_row)
    query_mod.execute_axis_query = timed("execute_total", query_mod.execute_axis_query)
    scoring_mod.atom_keyword_tokens = timed("sig_keyword_tokens", scoring_mod.atom_keyword_tokens)
    scoring_mod.keyword_score = timed("sig_keyword_overlap", scoring_mod.keyword_score)
    scoring_mod.entity_score = timed("sig_entity", scoring_mod.entity_score)
    scoring_mod.recency_score = timed("sig_recency", scoring_mod.recency_score)
    scoring_mod.support_score = timed("sig_support", scoring_mod.support_score)
    if provider == "milvus":
        import hms_elite.index.milvus as milvus_mod

        hms.index._ann = timed("milvus_ann_rpc", hms.index._ann)  # type: ignore[attr-defined]
        hms.index._existing_ids = timed("milvus_existing_ids", hms.index._existing_ids)  # type: ignore[attr-defined]
        milvus_mod._in_expr = timed("milvus_in_expr", milvus_mod._in_expr)
        milvus_mod.rank_hits = timed("milvus_rescore", milvus_mod.rank_hits)


def build(n: int, provider: str, workdir: str, max_candidates: int) -> HMS:
    """Create a file-backed HMS and load n raw message atoms (infer=False) in batches of 100.
    创建文件库 HMS，并按每批 100 条载入 n 个原始消息原子（infer=False）。
    """
    os.makedirs(workdir, exist_ok=True)
    vector_store: dict[str, Any] = {"provider": provider}
    if provider == "milvus":
        vector_store["config"] = {"uri": os.path.join(workdir, f"milvus_{n}.db"), "index_type": "HNSW"}
    cfg = HMSConfig(
        vector_store=vector_store,
        store={"path": os.path.join(workdir, f"hms_{provider}_{n}.db")},
        scoring={"max_candidates": max_candidates},
    )
    hms = HMS(cfg)
    texts = gen_texts(n)
    t0 = time.perf_counter()
    batch = 100
    for start in range(0, n, batch):
        chunk = texts[start : start + batch]
        # One add() per timestamp keeps the event time exact; group by identical timestamp inside the chunk.
        # 每个时间戳一次 add() 以保持事件时间精确；块内按相同时间戳分组。
        by_ts: dict[str, list[dict]] = {}
        for text, ts in chunk:
            by_ts.setdefault(ts, []).append({"role": "user", "content": text})
        for ts, messages in by_ts.items():
            hms.add(messages, user_id=USER, infer=False, timestamp=ts)
    load_s = time.perf_counter() - t0
    print(f"[build] provider={provider} N={n} loaded in {load_s:.1f}s ({load_s / n * 1000:.2f} ms/atom)", file=sys.stderr)
    return hms


def run_query(hms: HMS, text: str, repeat: int, top_k: int) -> dict[str, Any]:
    """Run one query `repeat` times; return per-bucket medians (ms), counts, candidates and one cProfile top list.
    对一个查询运行 `repeat` 次；返回各桶中位数（ms）、计数、候选数以及一份 cProfile 榜单。
    """
    filters = {"user_id": USER}
    # Warm-up call: the first search pays the SQLite page cache and pydantic schema build.
    # 预热调用：首次 search 要付出 SQLite 页缓存与 pydantic 模式构建的成本。
    hms.search(text, filters=filters, top_k=top_k, threshold=0.0, explain=True, reference_date=AS_OF)
    samples: dict[str, list[float]] = {}
    totals: list[float] = []
    counts: dict[str, int] = {}
    candidates: list[int] = []
    axis_selects: list[int] = []
    rows_out = 0
    for _ in range(repeat):
        TIMERS.reset()
        t0 = time.perf_counter()
        result = hms.search(text, filters=filters, top_k=top_k, threshold=0.0, explain=True, reference_date=AS_OF)
        totals.append(time.perf_counter() - t0)
        rows_out = len(result["results"])
        for name, value in TIMERS.buckets.items():
            samples.setdefault(name, []).append(value)
        counts = dict(TIMERS.counts)
        candidates.append(TIMERS.candidates)
        axis_selects.append(TIMERS.axis_selects)
    profile = cProfile.Profile()
    profile.enable()
    hms.search(text, filters=filters, top_k=top_k, threshold=0.0, explain=True, reference_date=AS_OF)
    profile.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("cumulative")
    top: list[dict[str, Any]] = []
    for (filename, line, func), (_cc, ncalls, tottime, cumtime, _callers) in sorted(
        stats.stats.items(), key=lambda kv: -kv[1][3]
    )[:30]:
        short = filename.replace(PROJECT + "/", "")
        top.append({"func": f"{short}:{line}({func})", "ncalls": ncalls, "tottime_ms": round(tottime * 1000, 3), "cumtime_ms": round(cumtime * 1000, 3)})
    medians = {name: round(statistics.median(values) * 1000, 3) for name, values in samples.items()}
    return {
        "query": text,
        "rows_out": rows_out,
        "candidates_hydrated": int(statistics.median(candidates)),
        "axis_selects": int(statistics.median(axis_selects)),
        "total_ms_median": round(statistics.median(totals) * 1000, 3),
        "total_ms_min": round(min(totals) * 1000, 3),
        "buckets_ms": medians,
        "bucket_calls": counts,
        "cprofile_top": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="sqlite", choices=["sqlite", "milvus"])
    parser.add_argument("--sizes", default="300,3000,30000")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--out", default=os.path.join(HERE, "results", "profile_search.json"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]
    workdir = os.path.join(HERE, "work", args.provider)
    shutil.rmtree(workdir, ignore_errors=True)
    report: dict[str, Any] = {
        "provider": args.provider,
        "python": sys.version.split()[0],
        "repeat": args.repeat,
        "top_k": args.top_k,
        "max_candidates": args.max_candidates,
        "as_of": AS_OF,
        "runs": [],
    }
    for n in sizes:
        hms = build(n, args.provider, workdir, args.max_candidates)
        instrument(hms, args.provider)
        for label, text in QUERIES:
            entry = run_query(hms, text, args.repeat, args.top_k)
            entry["label"] = label
            entry["n"] = n
            report["runs"].append(entry)
            print(f"[run] N={n} {label:10s} total={entry['total_ms_median']:8.2f} ms cands={entry['candidates_hydrated']}", file=sys.stderr)
        hms.close()
        # Restore module globals so the next size is instrumented once, not twice.
        # 还原模块全局名，使下一规模只被包装一次而不是两次。
        for mod in (query_mod, scoring_mod):
            for name in list(vars(mod)):
                value = getattr(mod, name)
                inner = getattr(value, "__wrapped__", None)
                if inner is not None:
                    setattr(mod, name, inner)
        if args.provider == "milvus":
            import hms_elite.index.milvus as milvus_mod

            for name in list(vars(milvus_mod)):
                value = getattr(milvus_mod, name)
                inner = getattr(value, "__wrapped__", None)
                if inner is not None:
                    setattr(milvus_mod, name, inner)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(render_table(report))
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)


BUCKET_ORDER = [
    "plan",
    "embed",
    "axis_sql",
    "hydrate",
    "index_search",
    "entity_lookup",
    "sig_keyword_tokens",
    "sig_keyword_overlap",
    "sig_entity",
    "sig_recency",
    "sig_support",
    "fuse",
    "score_details",
    "row_format",
]


def render_table(report: dict[str, Any]) -> str:
    """Markdown table: one row per (N, query), one column per bucket, plus the unattributed residual.
    Markdown 表：每 (N, 查询) 一行，每桶一列，外加未归因的残差。
    """
    extra = [k for k in ("milvus_in_expr", "milvus_ann_rpc", "milvus_rescore", "milvus_existing_ids") if any(k in r["buckets_ms"] for r in report["runs"])]
    # The milvus sub-buckets partition index_search, so index_search is dropped from the sum when they are present.
    # milvus 子桶是 index_search 的划分，因此有子桶时把 index_search 从求和中去掉。
    cols = [c for c in BUCKET_ORDER if not (extra and c == "index_search")] + extra
    head = "| N | query | cands | total ms | " + " | ".join(cols) + " | residual |"
    sep = "|" + "---|" * (len(cols) + 5)
    lines = [head, sep]
    for run in report["runs"]:
        b = run["buckets_ms"]
        attributed = sum(b.get(c, 0.0) for c in cols)
        residual = run["total_ms_median"] - attributed
        cells = [f"{b.get(c, 0.0):.2f}" for c in cols]
        lines.append(
            f"| {run['n']} | {run['label']} | {run['candidates_hydrated']} | {run['total_ms_median']:.2f} | "
            + " | ".join(cells)
            + f" | {residual:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
