"""Render every bench result JSON into the markdown tables embedded in docs/design/RETRIEVAL_DIAGNOSIS.md.
把每个基准结果 JSON 渲染成 docs/design/RETRIEVAL_DIAGNOSIS.md 中嵌入的 markdown 表格。
"""

# Purpose (EN): One place that turns results/*.json into tables so the diagnosis text never hand-copies numbers.
# 目的（中）：统一把 results/*.json 转成表格，使诊断文本永远不用手抄数字。

# Input (EN): results/profile_search*.json, milvus_scalar_probe.json, fts5_probe.json, sqlite_semantic_probe.json.
# 输入（中）：results/profile_search*.json、milvus_scalar_probe.json、fts5_probe.json、sqlite_semantic_probe.json。

# Output (EN): results/tables.md (stdout as well).
# 输出（中）：results/tables.md（同时打印到 stdout）。

# State effect (EN): Writes results/tables.md only.
# 状态影响（中）：只写 results/tables.md。

from __future__ import annotations

import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ATTR_COLS = [
    ("axis_sql", "axis SQL"),
    ("hydrate", "hydrate"),
    ("index_search", "index.search"),
    ("sig_keyword_tokens", "kw tokens"),
    ("sig_keyword_overlap", "kw overlap"),
    ("sig_entity", "entity"),
    ("sig_recency", "recency"),
    ("sig_support", "support"),
    ("fuse", "fuse"),
    ("score_details", "explain"),
    ("row_format", "rows"),
]
SMALL = ["plan", "embed", "entity_lookup"]


def load(name: str) -> dict[str, Any] | None:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def attribution_table(report: dict[str, Any], *, milvus: bool = False) -> str:
    cols = list(ATTR_COLS)
    if milvus:
        cols = [c for c in cols if c[0] != "index_search"] + [
            ("milvus_in_expr", "expr build"),
            ("milvus_ann_rpc", "ANN rpc+parse"),
            ("milvus_rescore", "rescore"),
        ]
    head = "| N | query | cands | selects | total ms | " + " | ".join(label for _, label in cols) + " | plan+embed+ent | sort/loop residual |"
    sep = "|" + "---|" * (len(cols) + 7)
    lines = [head, sep]
    for run in report["runs"]:
        b = run["buckets_ms"]
        small = sum(b.get(k, 0.0) for k in SMALL)
        attributed = sum(b.get(k, 0.0) for k, _ in cols) + small
        residual = run["total_ms_median"] - attributed
        cells = " | ".join(f"{b.get(k, 0.0):.2f}" for k, _ in cols)
        lines.append(
            f"| {run['n']} | {run['label']} | {run['candidates_hydrated']} | {run['axis_selects']} | {run['total_ms_median']:.2f} | {cells} | {small:.2f} | {residual:.2f} |"
        )
    return "\n".join(lines)


def per_candidate_table(report: dict[str, Any]) -> str:
    keys = [("axis_sql", "axis SQL"), ("hydrate", "hydrate"), ("index_search", "index.search"), ("sig_keyword_tokens", "kw tokens"), ("sig_keyword_overlap", "kw overlap"), ("sig_recency", "recency"), ("fuse", "fuse"), ("score_details", "explain")]
    lines = ["| N | query | cands | " + " | ".join(label for _, label in keys) + " | sum |", "|" + "---|" * (len(keys) + 4)]
    for run in report["runs"]:
        c = run["candidates_hydrated"] or 1
        b = run["buckets_ms"]
        vals = [b.get(k, 0.0) / c * 1000 for k, _ in keys]
        lines.append(f"| {run['n']} | {run['label']} | {c} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {sum(vals):.1f} |")
    return "\n".join(lines)


def cprofile_table(report: dict[str, Any], n: int, label: str, rows: int = 14) -> str:
    run = next(r for r in report["runs"] if r["n"] == n and r["label"] == label)
    lines = [f"cProfile, N={n}, query `{label}` (one search; cumulative ms / own ms / calls):", "", "| cum ms | own ms | calls | function |", "|---|---|---|---|"]
    for t in run["cprofile_top"][:rows]:
        func = t["func"].replace("/opt/anaconda3/lib/python3.14/site-packages/", "site-packages/").replace("/private/tmp/claude-501/-Users-lazy-Documents-Claude-Code/7477fb8f-767d-43d5-86b8-b84b581078ff/scratchpad/bench/", "bench/")
        lines.append(f"| {t['cumtime_ms']:.1f} | {t['tottime_ms']:.1f} | {t['ncalls']} | `{func}` |")
    return "\n".join(lines)


def milvus_probe_table(report: dict[str, Any]) -> str:
    lines = [
        f"milvus-lite, N={report['n']}, limit={report['limit']}, {report['repeat']} runs, median ms:",
        "",
        "| index | filter variant | expr bytes | rows matching | hits / recall gap | ids only | + vector payload | top_k=len(chunk) | top_k=len + vector |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in report["variants"]:
        lines.append(
            f"| {v['index_type']} | {v['variant']} | {v['expr_bytes']} | {v['matching_rows']} | {v['hits']} / {v['recall_gap']} | {v['median_ms']:.2f} | {v['median_ms_with_vector_payload']:.2f} | {v.get('median_ms_topk_eq_len', '-')} | {v.get('median_ms_topk_eq_len_with_vector', '-')} |"
        )
    return "\n".join(lines)


def fts_table(report: dict[str, Any]) -> str:
    lines = [f"FTS5 available: {report['fts5_available']} (sqlite {report['sqlite_version']}); K_kw = 30; 20 runs, median ms:", "", "| N | base insert s | FTS build s | query | matching docs | FTS5 MATCH+bm25 LIMIT 30 ms | LIKE+time index LIMIT 30 ms |", "|---|---|---|---|---|---|---|"]
    for size in report["sizes"]:
        for q in size["queries"]:
            lines.append(f"| {size['n']} | {size['base_insert_s']} | {size['fts_build_s']} | {q['label']} | {q['matching_docs']} | {q['fts5_ms']:.3f} | {q['like_ms']:.3f} |")
    sample = report["tokenizer_sample"]
    lines += ["", f"tokenize sample: `{sample['text']}` → `{sample['tokenize']}`; identifier_tokens → `{sample['identifier_tokens']}`"]
    return "\n".join(lines)


def semantic_table(report: dict[str, Any]) -> str:
    lines = [f"sqlite semantic channel prototype, dims={report['dims']}, numpy {report['numpy']}, median ms (whole scope gated in):", "", "| N | current (load 500-chunks + per-row np.dot, top_k=N) | B: VALUES-CTE join + matmul + argpartition K=60/80/240 | D: B + exact np.dot rescore of the K | C: scope JOIN in SQL, K=60/80/240 | E: cached float64 matrix, K=60/80/240 | rows where matmul ≠ dot (max abs diff) | order differs | D bit-equal |", "|---|---|---|---|---|---|---|---|---|"]
    for s in report["sizes"]:
        m = s["matmul_vs_dot"]
        fmt = lambda d: "/".join(f"{d[k]:.2f}" for k in ("60", "80", "240"))  # noqa: E731
        lines.append(f"| {s['n']} | {s['current_path_ms']:.2f} | {fmt(s['prototype_ms'])} | {fmt(s['prototype_exact_rescore_ms'])} | {fmt(s['prototype_scope_join_ms'])} | {fmt(s['prototype_cached_matrix_ms'])} | {m['rows_differing']} ({m['max_abs_diff']:.1e}) | {m['order_differs']} | {s['exact_rescore_bit_equal']} |")
    return "\n".join(lines)


def main() -> None:
    out: list[str] = []
    capped = load("profile_search.json")
    uncapped = load("profile_search_uncapped.json")
    milvus = load("profile_search_milvus.json")
    milvus_unc = load("profile_search_milvus_uncapped.json")
    if capped:
        out += ["### T1 sqlite, max_candidates=2000 (shipped default)", "", attribution_table(capped), "", "### T2 per-candidate cost (µs / hydrated candidate), sqlite capped run", "", per_candidate_table(capped), ""]
    if uncapped:
        out += ["### T3 sqlite, max_candidates=1000000 (true O(N) cost, no truncation)", "", attribution_table(uncapped), "", cprofile_table(uncapped, 30000, "identifier"), ""]
    if milvus:
        out += ["### T4 milvus-lite (HNSW), N=3000, max_candidates=2000", "", attribution_table(milvus, milvus=True), ""]
    if milvus_unc:
        out += ["### T5 milvus-lite (HNSW), N=3000, max_candidates=1000000", "", attribution_table(milvus_unc, milvus=True), "", cprofile_table(milvus_unc, 3000, "identifier", rows=10), ""]
    probe = load("milvus_scalar_probe.json")
    if probe:
        out += ["### T6 milvus-lite scalar-filter ANN probe", "", milvus_probe_table(probe), "", f"expression draft: `{probe['expression_draft']}`; VARCHAR instant comparison count check: {probe['varchar_instant_compare']}", ""]
    fts = load("fts5_probe.json")
    if fts:
        out += ["### T7 FTS5 keyword channel probe", "", fts_table(fts), ""]
    sem = load("sqlite_semantic_probe.json")
    if sem:
        out += ["### T8 sqlite semantic channel prototype", "", semantic_table(sem), ""]
    text = "\n".join(out)
    with open(os.path.join(RESULTS, "tables.md"), "w", encoding="utf-8") as handle:
        handle.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
