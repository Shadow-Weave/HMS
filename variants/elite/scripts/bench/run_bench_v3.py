"""Scale sweep N ∈ {300, 3000, 30000} for mem0+FAISS / elite_sqlite / elite_milvus: add segments, search@N, oracle overlap.
规模扫描 N ∈ {300, 3000, 30000}，三臂 mem0+FAISS / elite_sqlite / elite_milvus：add 分段、search@N、oracle 重叠率。
"""

# Purpose (EN): The acceptance benchmark of the bounded-recall retrieval (RETRIEVAL_BENCH_2026-09-10.md): one arm per
#               process, the same hash@64 embedder everywhere, infer=False, top_k=20, threshold=0.1, 5 queries × 20
#               timed runs, and the top-20 overlap of the new path with the exhaustive reference (_exhaustive_search).
# 目的（中）：有界召回检索的验收基准（RETRIEVAL_BENCH_2026-09-10.md）：每个进程一臂，各处使用同一 hash@64 嵌入器，
#             infer=False，top_k=20，threshold=0.1，5 个查询 × 20 次计时，以及新路径 top-20 与穷举参考
#             （_exhaustive_search）的重叠率。

# Input (EN): --arm mem0|elite_sqlite|elite_milvus, --sizes, --out, --work; scripts/bench/profile_search.py's corpus.
# 输入（中）：--arm mem0|elite_sqlite|elite_milvus、--sizes、--out、--work；scripts/bench/profile_search.py 的语料。

# Output (EN): One JSON file per arm (merged by --merge into scale_v3.json plus a markdown table).
# 输出（中）：每臂一个 JSON 文件（--merge 合并为 scale_v3.json 并生成 markdown 表）。

# State effect (EN): Throwaway SQLite / milvus-lite / FAISS files under --work only; no network.
# 状态影响（中）：只在 --work 下写入一次性 SQLite / milvus-lite / FAISS 文件；不联网。

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT)

from profile_search import AS_OF, QUERIES, USER, gen_texts  # noqa: E402

TOP_K = 20
THRESHOLD = 0.1
REPEAT = 20
SEGMENT = 10


def _median(values: list[float]) -> float:
    """Median in ms of a list of seconds.
    秒列表的中位数（毫秒）。
    """
    return round(statistics.median(values) * 1000, 3) if values else 0.0


def _p90(values: list[float]) -> float:
    """p90 in ms of a list of seconds (nearest-rank).
    秒列表的 p90（毫秒，最近秩）。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(0.9 * len(ordered))) - 1)] * 1000, 3)


def _segments(latencies: list[float]) -> dict[str, float]:
    """Median add latency of the first / middle / last SEGMENT adds.
    前 / 中 / 末 SEGMENT 次 add 的中位延迟。
    """
    n = len(latencies)
    mid = max(0, n // 2 - SEGMENT // 2)
    return {
        "first10_ms": _median(latencies[:SEGMENT]),
        "mid10_ms": _median(latencies[mid : mid + SEGMENT]),
        "last10_ms": _median(latencies[-SEGMENT:]),
        "all_median_ms": _median(latencies),
        "total_s": round(sum(latencies), 3),
    }


def _overlap(new_ids: list[str], oracle_ids: list[str]) -> float:
    """|new ∩ oracle| / |oracle| (0.0 when the oracle is empty).
    |new ∩ oracle| / |oracle|（oracle 为空时为 0.0）。
    """
    if not oracle_ids:
        return 0.0
    return round(len(set(new_ids) & set(oracle_ids)) / len(oracle_ids), 4)


# ---------------------------------------------------------------- elite arms
# ---------------------------------------------------------------- elite 两臂
def run_elite(provider: str, sizes: list[int], work: str) -> dict[str, Any]:
    """Build an HMS per N (sqlite or milvus provider), time adds and searches, compute the exhaustive oracle overlap.
    为每个 N 构建一个 HMS（sqlite 或 milvus 提供者），为 add 与 search 计时，计算穷举 oracle 的重叠率。
    """
    from hms_elite import HMS
    from hms_elite.configs.base import HMSConfig, ScoringConfig
    from hms_elite.models import AxisQuery
    from hms_elite.mvmi.planner import plan_query
    from tests.test_retrieval import _exhaustive_search

    report: dict[str, Any] = {"arm": f"elite_{provider}", "sizes": {}}
    for n in sizes:
        path = os.path.join(work, f"hms_{provider}_{n}.db")
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        vector_store: dict[str, Any] = {"provider": provider}
        if provider == "milvus":
            muri = os.path.join(work, f"milvus_{n}.db")
            shutil.rmtree(muri, ignore_errors=True)
            if os.path.exists(muri):
                os.remove(muri)
            vector_store["config"] = {"uri": muri, "index_type": "HNSW"}
        hms = HMS(HMSConfig(vector_store=vector_store, store={"path": path}))
        latencies: list[float] = []
        for text, ts in gen_texts(n):
            t0 = time.perf_counter()
            hms.add([{"role": "user", "content": text}], user_id=USER, infer=False, timestamp=ts)
            latencies.append(time.perf_counter() - t0)
        entry: dict[str, Any] = {"add": _segments(latencies), "queries": {}}
        filters = {"user_id": USER}
        all_runs: list[float] = []
        overlaps: list[float] = []
        for label, text in QUERIES:
            hms.search(text, filters=filters, top_k=TOP_K, threshold=THRESHOLD, reference_date=AS_OF)
            runs: list[float] = []
            for _ in range(REPEAT):
                t0 = time.perf_counter()
                res = hms.search(text, filters=filters, top_k=TOP_K, threshold=THRESHOLD, reference_date=AS_OF)
                runs.append(time.perf_counter() - t0)
            explained = hms.search(
                text, filters=filters, top_k=TOP_K, threshold=THRESHOLD, reference_date=AS_OF, explain=True
            )
            # Oracle rows are atom (version) ids; the public row's `id` is the memory_id, `atom_id` the version.
            # oracle 行是原子（版本）id；公开行的 `id` 是 memory_id，`atom_id` 才是版本。
            new_ids = [row["atom_id"] for row in res["results"]]
            details = explained["results"][0]["score_details"] if explained["results"] else {}
            plan = plan_query(text, as_of=AS_OF)
            aq = AxisQuery(
                filters=filters, text=text, limit=TOP_K, as_of=AS_OF, aggregation="topk", plan=plan.as_dict()
            )
            t0 = time.perf_counter()
            oracle = _exhaustive_search(
                hms.store,
                hms.index,
                hms.embedder,
                aq,
                scoring=ScoringConfig(),
                capabilities=hms.store.capabilities,
                threshold=THRESHOLD,
                explain=False,
                max_candidates=10**6,
            )
            oracle_s = time.perf_counter() - t0
            oracle_ids = [atom_id for atom_id, _score, _details in oracle]
            overlap = _overlap(new_ids, oracle_ids)
            overlaps.append(overlap)
            all_runs.extend(runs)
            entry["queries"][label] = {
                "median_ms": _median(runs),
                "p90_ms": _p90(runs),
                "rows": len(new_ids),
                "overlap_top20": overlap,
                "oracle_rows": len(oracle_ids),
                "oracle_ms": round(oracle_s * 1000, 1),
                "missed": [i for i in oracle_ids if i not in set(new_ids)][:20],
                "pool_size": details.get("pool_size"),
                "channel_sizes": details.get("channel_sizes"),
                "primary_axis": details.get("primary_axis"),
            }
        entry["search"] = {
            "median_of_query_medians_ms": round(statistics.median([q["median_ms"] for q in entry["queries"].values()]), 3),
            "median_all_runs_ms": _median(all_runs),
            "p90_all_runs_ms": _p90(all_runs),
            "overlap_mean": round(statistics.mean(overlaps), 4),
            "overlap_min": min(overlaps),
        }
        report["sizes"][str(n)] = entry
        hms.close()
        print(json.dumps({"arm": report["arm"], "N": n, **entry["search"], "add": entry["add"]}), flush=True)
    return report


# ---------------------------------------------------------------- mem0 arm
# ---------------------------------------------------------------- mem0 臂
def run_mem0(sizes: list[int], work: str) -> dict[str, Any]:
    """mem0 + FAISS with elite's HashEmbedder injected; FAISS _save kept only on the timed segments (see report).
    注入 elite 的 HashEmbedder 的 mem0 + FAISS；FAISS 的 _save 只在计时段保留（见报告）。
    """
    os.environ["MEM0_TELEMETRY"] = "false"
    import numpy as np
    from mem0.memory import main as mem0_main
    from mem0.vector_stores import faiss as faiss_mod

    from hms_elite.embeddings.hash import HashEmbedder

    hash_embedder = HashEmbedder({"embedding_dims": 64})

    class Shared:
        """mem0 embedder facade over elite's HashEmbedder (same vectors as the elite arms).
        elite HashEmbedder 之上的 mem0 嵌入器门面（与 elite 两臂向量相同）。
        """

        config = type("C", (), {"embedding_dims": 64})()

        def embed(self, text: str, memory_action: Any = None) -> list[float]:
            """Embed one text.
            嵌入一段文本。
            """
            return hash_embedder.embed(text, "add")

        def embed_batch(self, texts: list[str], memory_action: Any = None) -> list[list[float]]:
            """Embed several texts.
            嵌入多段文本。
            """
            return [hash_embedder.embed(t, "add") for t in texts]

    mem0_main.LlmFactory.create = staticmethod(lambda *a, **k: None)
    mem0_main.EmbedderFactory.create = staticmethod(lambda *a, **k: Shared())
    original_save = faiss_mod.FAISS._save
    state = {"save": True}

    def gated_save(self: Any) -> None:
        """Run the real _save only while a timed segment is active.
        只在计时段内执行真实的 _save。
        """
        if state["save"]:
            original_save(self)

    faiss_mod.FAISS._save = gated_save
    report: dict[str, Any] = {"arm": "mem0", "sizes": {}, "note": "FAISS._save (O(N) pickle per add) ran only during the three timed add segments and once at the end; other adds skipped it."}
    for n in sizes:
        root = os.path.join(work, f"mem0_{n}")
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        memory = mem0_main.Memory.from_config(
            {
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "path": os.path.join(root, "faiss"),
                        "embedding_model_dims": 64,
                        "collection_name": "bench",
                        "distance_strategy": "cosine",
                    },
                },
                "llm": {"provider": "openai", "config": {"api_key": "x"}},
                "embedder": {"provider": "openai", "config": {"api_key": "x", "embedding_dims": 64}},
                "history_db_path": os.path.join(root, "history.db"),
            }
        )
        texts = gen_texts(n)
        mid = max(0, n // 2 - SEGMENT // 2)
        timed = set(range(SEGMENT)) | set(range(mid, mid + SEGMENT)) | set(range(max(0, n - SEGMENT), n))
        latencies: list[float] = []
        for i, (text, ts) in enumerate(texts):
            state["save"] = i in timed
            t0 = time.perf_counter()
            # The OSS mem0 SDK rejects `timestamp` (the elite arms pass it); mem0 stamps created_at = now.
            # 开源 mem0 SDK 拒绝 `timestamp`（elite 两臂传入它）；mem0 以 now 作为 created_at。
            memory.add(text, user_id=USER, infer=False)
            latencies.append(time.perf_counter() - t0)
        state["save"] = True
        original_save(memory.vector_store)
        entry: dict[str, Any] = {"add": _segments(latencies), "queries": {}}
        store = memory.vector_store
        total = store.index.ntotal
        matrix = store.index.reconstruct_n(0, total)
        ids = [store.index_to_id[i] for i in range(total)]
        all_runs: list[float] = []
        overlaps: list[float] = []
        for label, text in QUERIES:
            memory.search(text, filters={"user_id": USER}, top_k=TOP_K, threshold=THRESHOLD)
            runs: list[float] = []
            for _ in range(REPEAT):
                t0 = time.perf_counter()
                res = memory.search(text, filters={"user_id": USER}, top_k=TOP_K, threshold=THRESHOLD)
                runs.append(time.perf_counter() - t0)
            new_ids = [row["id"] for row in res["results"]]
            q = np.asarray(hash_embedder.embed(text, "search"), dtype=np.float32)
            scores = matrix @ q
            order = np.argsort(-scores, kind="stable")[:TOP_K]
            oracle_ids = [ids[i] for i in order if scores[i] >= THRESHOLD]
            overlap = _overlap(new_ids, oracle_ids)
            overlaps.append(overlap)
            all_runs.extend(runs)
            entry["queries"][label] = {
                "median_ms": _median(runs),
                "p90_ms": _p90(runs),
                "rows": len(new_ids),
                "overlap_top20": overlap,
                "oracle_rows": len(oracle_ids),
            }
        entry["search"] = {
            "median_of_query_medians_ms": round(statistics.median([q["median_ms"] for q in entry["queries"].values()]), 3),
            "median_all_runs_ms": _median(all_runs),
            "p90_all_runs_ms": _p90(all_runs),
            "overlap_mean": round(statistics.mean(overlaps), 4),
            "overlap_min": min(overlaps),
            "oracle": "brute-force cosine over the FAISS vectors (same metric as IndexFlat), not the elite fused score",
        }
        report["sizes"][str(n)] = entry
        print(json.dumps({"arm": "mem0", "N": n, **entry["search"], "add": entry["add"]}), flush=True)
    return report


def merge(paths: list[str], out: str) -> None:
    """Merge per-arm JSON files into `out` and print a markdown table.
    把各臂 JSON 合并到 `out` 并打印 markdown 表。
    """
    arms = [json.load(open(p)) for p in paths]
    merged = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "repeat": REPEAT,
        "arms": {a["arm"]: a for a in arms},
    }
    json.dump(merged, open(out, "w"), indent=1)
    print("| arm | N | add first10 | add mid10 | add last10 | search median (ms) | search p90 (ms) | overlap mean | overlap min |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        for n, e in a["sizes"].items():
            s, ad = e["search"], e["add"]
            print(
                f"| {a['arm']} | {n} | {ad['first10_ms']} | {ad['mid10_ms']} | {ad['last10_ms']} | "
                f"{s['median_of_query_medians_ms']} | {s['p90_all_runs_ms']} | {s['overlap_mean']} | {s['overlap_min']} |"
            )


def main() -> None:
    """CLI entry.
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["mem0", "elite_sqlite", "elite_milvus"])
    parser.add_argument("--sizes", default="300,3000,30000")
    parser.add_argument("--work", default=os.path.join(HERE, "work_v3"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--merge", nargs="*", default=None)
    args = parser.parse_args()
    if args.merge is not None:
        merge(args.merge, args.out or os.path.join(HERE, "results", "scale_v3.json"))
        return
    os.makedirs(args.work, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",")]
    if args.arm == "mem0":
        report = run_mem0(sizes, args.work)
    else:
        report = run_elite(args.arm.split("_", 1)[1], sizes, args.work)
    report["python"] = platform.python_version()
    out = args.out or os.path.join(HERE, "results", f"scale_v3_{args.arm}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
