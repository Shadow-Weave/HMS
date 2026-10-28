"""Measure the sqlite semantic channel prototype: gated-id vector load + one numpy matmul + top-K, no hydration.
测量 sqlite 语义通道原型：门控 id 的向量加载 + 一次 numpy 矩阵乘 + top-K，不做水合。
"""

# Purpose (EN): Decide K_sem for the sqlite / memory providers from measured cost per gated id.
# 目的（中）：依据每个门控 id 的实测成本，为 sqlite / memory 提供者确定 K_sem。

# Input (EN): N ∈ {300, 3000, 30000} float32 64-dim vectors in an atom_vectors-shaped table; K ∈ {60, 80, 240}.
# 输入（中）：N ∈ {300, 3000, 30000} 个 float32 64 维向量，存于 atom_vectors 形状的表；K ∈ {60, 80, 240}。

# Output (EN): results/sqlite_semantic_probe.json: load / decode / matmul / topk medians, and the id-gated JOIN variant.
# 输出（中）：results/sqlite_semantic_probe.json：加载 / 解码 / 矩阵乘 / topk 的中位数，以及 id 门控 JOIN 变体。

# State effect (EN): In-memory SQLite only.
# 状态影响（中）：仅内存 SQLite。

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import struct
import sys
import time
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = 64


def vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, DIMS)).astype("float32")
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def median_ms(fn: Any, repeat: int = 20) -> float:
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return round(statistics.median(times) * 1000, 3)


def main() -> None:
    report: dict[str, Any] = {"dims": DIMS, "numpy": np.__version__, "sizes": []}
    for n in (300, 3000, 30000):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memory_atoms (atom_id TEXT PRIMARY KEY, user_id TEXT, valid_to TEXT, archived_reason TEXT, time_anchor TEXT)")
        conn.execute("CREATE TABLE atom_vectors (atom_id TEXT PRIMARY KEY, embedder TEXT, dims INTEGER, embedding BLOB)")
        conn.execute("CREATE INDEX ix_scope ON memory_atoms(user_id)")
        mat = vectors(n, seed=1)
        ids = [f"atom_{i:06d}" for i in range(n)]
        conn.executemany("INSERT INTO memory_atoms VALUES (?,?,?,?,?)", [(i, "alice", None, None, "2026-01-01T00:00:00Z") for i in ids])
        conn.executemany(
            "INSERT INTO atom_vectors VALUES (?,?,?,?)",
            [(ids[i], "hash", DIMS, sqlite3.Binary(struct.pack(f"<{DIMS}f", *mat[i].tolist()))) for i in range(n)],
        )
        conn.commit()
        query = vectors(1, seed=5)[0]
        gated = ids  # the whole scope is gated in / 整个范围都被门控放行

        # Variant A (current path): load_vectors(ids) in 500-id chunks, struct.unpack per row, per-row np.dot.
        # 变体 A（现有路径）：按 500 个 id 分块 load_vectors，逐行 struct.unpack，逐行 np.dot。
        def current_path() -> list[tuple[str, float]]:
            out: dict[str, list[float]] = {}
            for start in range(0, len(gated), 500):
                chunk = gated[start : start + 500]
                rows = conn.execute(
                    f"SELECT atom_id, dims, embedding FROM atom_vectors WHERE atom_id IN ({','.join('?' * len(chunk))})", chunk
                ).fetchall()
                for atom_id, dims, blob in rows:
                    out[atom_id] = list(struct.unpack(f"<{dims}f", bytes(blob)))
            scored = [(a, max(0.0, float(np.dot(np.asarray(v, dtype="float64"), query.astype("float64"))))) for a, v in out.items()]
            scored.sort(key=lambda h: (-h[1], h[0]))
            return scored[: len(gated)]

        # Variant B (prototype): one JOIN against a VALUES CTE of the gated ids, bulk frombuffer, one matmul, argpartition K.
        # 变体 B（原型）：对门控 id 的 VALUES CTE 做一次 JOIN，整体 frombuffer，一次矩阵乘，argpartition 取 K。
        def prototype(k: int) -> list[tuple[str, float]]:
            id_list: list[str] = []
            blobs: list[bytes] = []
            for start in range(0, len(gated), 500):
                chunk = gated[start : start + 500]
                rows = conn.execute(
                    f"WITH ids(atom_id) AS (VALUES {','.join(['(?)'] * len(chunk))}) "
                    "SELECT v.atom_id, v.embedding FROM ids JOIN atom_vectors v ON v.atom_id = ids.atom_id",
                    chunk,
                ).fetchall()
                for atom_id, blob in rows:
                    id_list.append(atom_id)
                    blobs.append(blob)
            if not blobs:
                return []
            matrix = np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), DIMS).astype("float64")
            scores = matrix @ query.astype("float64")
            scores = np.maximum(scores, 0.0)
            k = min(k, len(id_list))
            top = np.argpartition(-scores, k - 1)[:k] if k < len(id_list) else np.arange(len(id_list))
            hits = [(id_list[i], float(scores[i])) for i in top]
            hits.sort(key=lambda h: (-h[1], h[0]))
            return hits

        # Variant C (no id gate; scope gate pushed into SQL, scan the whole scope's vectors in one statement).
        # 变体 C（无 id 门；范围门下推到 SQL，一条语句扫描整个范围的向量）。
        def prototype_scope_join(k: int) -> list[tuple[str, float]]:
            rows = conn.execute(
                "SELECT v.atom_id, v.embedding FROM atom_vectors v JOIN memory_atoms m ON m.atom_id = v.atom_id "
                "WHERE m.user_id = ? AND m.archived_reason IS NULL AND m.valid_to IS NULL",
                ("alice",),
            ).fetchall()
            id_list = [r[0] for r in rows]
            matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4").reshape(len(rows), DIMS).astype("float64")
            scores = np.maximum(matrix @ query.astype("float64"), 0.0)
            k = min(k, len(id_list))
            top = np.argpartition(-scores, k - 1)[:k]
            hits = [(id_list[i], float(scores[i])) for i in top]
            hits.sort(key=lambda h: (-h[1], h[0]))
            return hits

        # Variant D: matmul only chooses the K recalled ids; their authoritative score is the exact per-row np.dot the
        # current rank_hits() uses, so pool scores stay bit-identical to the old path.
        # 变体 D：矩阵乘只负责挑出 K 个召回 id；它们的权威分数用现有 rank_hits() 的逐行 np.dot 精确重算，
        # 使池内分数与旧路径逐位一致。
        def prototype_exact(k: int) -> list[tuple[str, float]]:
            picked = prototype(k)
            q64 = query.astype("float64")
            out = []
            for atom_id, _score in picked:
                blob = conn.execute("SELECT embedding FROM atom_vectors WHERE atom_id = ?", (atom_id,)).fetchone()[0]
                vec = list(struct.unpack(f"<{DIMS}f", bytes(blob)))
                out.append((atom_id, max(0.0, float(np.dot(np.asarray(vec, dtype="float64"), q64)))))
            out.sort(key=lambda h: (-h[1], h[0]))
            return out

        # Variant E: an in-process float32 matrix cache (no SQL at all), the shape a MemoryIndex-style cache would have.
        # 变体 E：进程内 float32 矩阵缓存（完全不走 SQL），即 MemoryIndex 风格缓存的形态。
        cached = mat.astype("float64")

        def prototype_cached(k: int) -> list[tuple[str, float]]:
            scores = np.maximum(cached @ query.astype("float64"), 0.0)
            top = np.argpartition(-scores, k - 1)[:k]
            hits = [(ids[i], float(scores[i])) for i in top]
            hits.sort(key=lambda h: (-h[1], h[0]))
            return hits

        # Bit-equality check: matmul (dgemv) vs per-row np.dot (ddot) differ in accumulation order; count how often.
        # 逐位相等检查：矩阵乘（dgemv）与逐行 np.dot（ddot）的累加顺序不同；统计差异频率。
        cur = dict(current_path())
        proto = dict(prototype(len(gated)))
        diffs = [abs(cur[a] - proto[a]) for a in cur]
        exact = dict(prototype_exact(len(gated)))
        entry = {
            "n": n,
            "current_path_ms": median_ms(current_path, 10 if n < 30000 else 5),
            "prototype_ms": {str(k): median_ms(lambda k=k: prototype(k)) for k in (60, 80, 240)},
            "prototype_exact_rescore_ms": {str(k): median_ms(lambda k=k: prototype_exact(k)) for k in (60, 80, 240)},
            "prototype_scope_join_ms": {str(k): median_ms(lambda k=k: prototype_scope_join(k)) for k in (60, 80, 240)},
            "prototype_cached_matrix_ms": {str(k): median_ms(lambda k=k: prototype_cached(k)) for k in (60, 80, 240)},
            "matmul_vs_dot": {
                "rows_differing": int(sum(1 for d in diffs if d > 0)),
                "max_abs_diff": max(diffs),
                "order_differs": [a for a, _ in sorted(cur.items(), key=lambda h: (-h[1], h[0]))] != [a for a, _ in sorted(proto.items(), key=lambda h: (-h[1], h[0]))],
            },
            "exact_rescore_bit_equal": all(cur[a] == exact[a] for a in cur) and set(cur) == set(exact),
        }
        report["sizes"].append(entry)
        print(json.dumps(entry), file=sys.stderr)
        conn.close()
    out = os.path.join(HERE, "results", "sqlite_semantic_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
