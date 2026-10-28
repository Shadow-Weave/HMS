"""Probe SQLite FTS5 availability and measure keyword-channel latency (FTS5 MATCH vs LIKE fallback) at N up to 30000.
探测 SQLite FTS5 可用性，并测量关键词通道延迟（FTS5 MATCH 与 LIKE 退路）在 N 最高 30000 时的表现。
"""

# Purpose (EN): Evidence for the K_kw channel design: pre-tokenized FTS5 over the project tokenizer, bounded LIMIT.
# 目的（中）：为 K_kw 通道设计提供证据：按项目分词器预分词写入 FTS5，带有界 LIMIT。

# Input (EN): The bench corpus generator (profile_search.gen_texts) and hms_elite.utils.text.tokenize.
# 输入（中）：基准语料生成器（profile_search.gen_texts）与 hms_elite.utils.text.tokenize。

# Output (EN): results/fts5_probe.json (availability, build time, per-query medians for FTS5 and LIKE).
# 输出（中）：results/fts5_probe.json（可用性、建表耗时、FTS5 与 LIKE 的逐查询中位数）。

# State effect (EN): In-memory SQLite only.
# 状态影响（中）：仅内存 SQLite。

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
import time
from typing import Any

PROJECT = "/Users/lazy/Documents/Claude Code/hms-elite"
sys.path.insert(0, PROJECT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_search import QUERIES, gen_texts  # noqa: E402

from hms_elite.utils.text import (  # noqa: E402
    content_tokens,
    identifier_tokens,
    tokenize,
)


def fts5_available() -> tuple[bool, str]:
    """The probe the store's capabilities() will run.
    存储 capabilities() 将要运行的探测。
    """
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True, sqlite3.sqlite_version
    except sqlite3.OperationalError as exc:
        return False, f"{sqlite3.sqlite_version}: {exc}"


def build(conn: sqlite3.Connection, texts: list[tuple[str, str]]) -> dict[str, float]:
    """Base table + pre-tokenized FTS5 (tokenize='unicode61' over space-joined project tokens) + LIKE column.
    基表 + 预分词 FTS5（对空格连接的项目词元使用 tokenize='unicode61'）+ LIKE 列。
    """
    conn.execute("CREATE TABLE memory_atoms (atom_id TEXT PRIMARY KEY, user_id TEXT, text TEXT, time_anchor TEXT, tokens TEXT)")
    conn.execute("CREATE INDEX ix_scope_time ON memory_atoms(user_id, time_anchor)")
    t0 = time.perf_counter()
    rows = []
    for i, (text, ts) in enumerate(texts):
        # CJK bigrams from tokenize() are joined with spaces so unicode61 sees them as separate terms.
        # tokenize() 产出的 CJK 二元组用空格连接，使 unicode61 把它们当作独立词项。
        toks = " ".join(tokenize(text))
        rows.append((f"atom_{i:06d}", "alice", text, ts, toks))
    conn.executemany("INSERT INTO memory_atoms VALUES (?,?,?,?,?)", rows)
    t1 = time.perf_counter()
    conn.execute("CREATE VIRTUAL TABLE atom_fts USING fts5(atom_id UNINDEXED, tokens, tokenize='unicode61')")
    conn.execute("INSERT INTO atom_fts(atom_id, tokens) SELECT atom_id, tokens FROM memory_atoms")
    t2 = time.perf_counter()
    conn.commit()
    return {"base_insert_s": round(t1 - t0, 3), "fts_build_s": round(t2 - t1, 3)}


def fts_query(query: str) -> str:
    """OR of content tokens plus identifier tokens; every term quoted so '.', '/' and CJK survive the FTS parser.
    内容词元与标识符词元的 OR；每个词项加引号，使 '.'、'/' 与 CJK 能通过 FTS 解析器。
    """
    terms = list(dict.fromkeys(tokenize(query)))
    terms = [t for t in terms if t] or ["zzz"]
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def timed(fn: Any, repeat: int) -> tuple[float, Any]:
    fn()
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000, out


def main() -> None:
    ok, version = fts5_available()
    report: dict[str, Any] = {"fts5_available": ok, "sqlite_version": version, "sizes": []}
    print(f"fts5 available={ok} sqlite={version}", file=sys.stderr)
    for n in (300, 3000, 30000):
        conn = sqlite3.connect(":memory:")
        texts = gen_texts(n)
        build_stats = build(conn, texts)
        entry: dict[str, Any] = {"n": n, **build_stats, "queries": []}
        for label, q in QUERIES:
            match = fts_query(q)
            k = 30

            def run_fts() -> list:
                return conn.execute(
                    "SELECT f.atom_id, bm25(atom_fts) AS rank FROM atom_fts f JOIN memory_atoms m ON m.atom_id = f.atom_id "
                    "WHERE atom_fts MATCH ? AND m.user_id = ? ORDER BY rank, f.atom_id LIMIT ?",
                    (match, "alice", k),
                ).fetchall()

            like_terms = [t for t in dict.fromkeys(tokenize(q)) if len(t) >= 2][:6]
            like_sql = " OR ".join("tokens LIKE ?" for _ in like_terms)
            like_params = [f"%{t}%" for t in like_terms]

            def run_like() -> list:
                return conn.execute(
                    f"SELECT atom_id FROM memory_atoms WHERE user_id = ? AND ({like_sql}) ORDER BY time_anchor DESC, atom_id LIMIT ?",
                    ["alice", *like_params, k],
                ).fetchall()

            def run_count() -> int:
                return conn.execute("SELECT count(*) FROM atom_fts WHERE atom_fts MATCH ?", (match,)).fetchone()[0]

            fts_ms, fts_rows = timed(run_fts, 20)
            like_ms, like_rows = timed(run_like, 20)
            _, matching = timed(run_count, 3)
            entry["queries"].append(
                {
                    "label": label,
                    "match": match,
                    "matching_docs": matching,
                    "fts5_ms": round(fts_ms, 3),
                    "fts5_rows": len(fts_rows),
                    "like_ms": round(like_ms, 3),
                    "like_rows": len(like_rows),
                }
            )
            print(f"N={n:6d} {label:10s} fts5={fts_ms:7.3f} ms ({len(fts_rows)} rows of {matching}) like={like_ms:7.3f} ms", file=sys.stderr)
        report["sizes"].append(entry)
        conn.close()
    report["tokenizer_sample"] = {
        "text": "Sarah patched src/billing/retry.ts 今天部署了支付服务",
        "tokenize": tokenize("Sarah patched src/billing/retry.ts 今天部署了支付服务"),
        "content_tokens": content_tokens("Sarah patched src/billing/retry.ts 今天部署了支付服务"),
        "identifier_tokens": identifier_tokens("Sarah patched src/billing/retry.ts 今天部署了支付服务"),
    }
    out = os.path.join(HERE, "results", "fts5_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
