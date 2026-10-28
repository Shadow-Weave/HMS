"""Per-channel ranks of the oracle atoms the bounded pool missed (RETRIEVAL_BENCH §4 companion of run_bench_v3.py).
有界池漏掉的 oracle 原子在各通道中的名次（run_bench_v3.py 的 RETRIEVAL_BENCH §4 伴随脚本）。
"""

# Purpose (EN): For every (N, query) of an elite arm's scale_v3 JSON whose overlap is below 0.9, locate each missed
#               oracle atom in the semantic order (cosine over G), the keyword order (global bm25 and the newest-match
#               window), the recency order (axis order inside G) and the entity order (time order among the linked
#               atoms), so the loss is attributable to a channel bound rather than to an implementation slip.
# 目的（中）：对某 elite 臂 scale_v3 JSON 中重叠率低于 0.9 的每个 (N, 查询)，把每个漏掉的 oracle 原子定位到语义序
#             （G 上的余弦）、关键词序（全局 bm25 与最新匹配窗口）、新近度序（G 内的轴序）与实体序（链接原子的时间序），
#             使损失可归因于通道上限而不是实现失误。

# Input (EN): --json (the arm's scale_v3 JSON), --work (the run's --work directory holding hms_<provider>_<N>.db).
# 输入（中）：--json（该臂的 scale_v3 JSON）、--work（该次运行的 --work 目录，含 hms_<provider>_<N>.db）。

# Output (EN): A markdown table on stdout (N, query, missed count, median rank per channel) and one row per missed atom.
# 输出（中）：stdout 上的 markdown 表（N、查询、漏掉数、各通道名次中位）与每个漏掉原子一行。

# State effect (EN): Read-only over the bench databases; no network.
# 状态影响（中）：只读基准数据库；不联网。

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT)

from profile_search import AS_OF, QUERIES, USER  # noqa: E402


def _rank_of(order: list[str], atom_id: str) -> int | None:
    """1-based position of an id in an ordered list, None when absent.
    id 在有序列表中的 1 起位置；不在其中时为 None。
    """
    try:
        return order.index(atom_id) + 1
    except ValueError:
        return None


def analyse(provider: str, n: int, label: str, missed: list[str], work: str) -> list[dict[str, Any]]:
    """Ranks of the missed atoms of one (N, query) in every channel order.
    一个 (N, 查询) 的漏掉原子在每个通道顺序中的名次。
    """
    from hms_elite import HMS
    from hms_elite.configs.base import HMSConfig
    from hms_elite.memory.storage import fts_match_query
    from hms_elite.models import Scope
    from hms_elite.mvmi.planner import plan_query
    from hms_elite.mvmi.query import AXIS_ORDER, _plan_gates, _plan_keyword_tokens

    path = os.path.join(work, f"hms_{provider}_{n}.db")
    vector_store: dict[str, Any] = {"provider": provider}
    if provider == "milvus":
        vector_store["config"] = {"uri": os.path.join(work, f"milvus_{n}.db"), "index_type": "HNSW"}
    hms = HMS(HMSConfig(vector_store=vector_store, store={"path": path}))
    try:
        store, index, embedder = hms.store, hms.index, hms.embedder
        scope = Scope(user_id=USER)
        text = dict(QUERIES)[label]
        plan = plan_query(text, as_of=AS_OF).as_dict()
        clauses, params, _names = _plan_gates(plan, scope, store.capabilities)
        where = " AND ".join(f"({c})" for c in clauses)
        gate = store.axis_ids(scope, where_sql=where, params=params, as_of=AS_OF, order=None)
        if not gate:
            gate = store.axis_ids(scope, as_of=AS_OF, order=None)
        vector = embedder.embed(text, "search")
        semantic = [a for a, _s in index.search(vector, candidate_ids=gate, top_k=len(gate))]
        tokens = _plan_keyword_tokens(plan, text)
        bm25: list[str] = []
        window: list[str] = []
        if tokens and store.capabilities.get("fts5"):
            match = fts_match_query(tokens)
            bm25 = [
                r["atom_id"]
                for r in store._execute(
                    "SELECT atom_id FROM atom_fts WHERE atom_fts MATCH ? ORDER BY rank, atom_id", [match]
                ).fetchall()
            ]
            window = [
                r["atom_id"]
                for r in store._execute(
                    "SELECT atom_id FROM atom_fts WHERE atom_fts MATCH ? ORDER BY rowid DESC", [match]
                ).fetchall()
            ]
        recency = store.axis_ids(scope, where_sql=where, params=params, as_of=AS_OF, order=AXIS_ORDER)
        keys = [str(h.get("key")) for h in (plan.get("entity_hints") or []) if h.get("key")]
        entity = store.entity_candidates(scope, keys, as_of=AS_OF, limit=None) if keys else []
        rows = []
        for atom_id in missed:
            rows.append(
                {
                    "atom_id": atom_id,
                    "in_G": atom_id in set(gate),
                    "semantic": _rank_of(semantic, atom_id),
                    "bm25": _rank_of(bm25, atom_id),
                    "newest_window": _rank_of(window, atom_id),
                    "recency": _rank_of(recency, atom_id),
                    "entity": _rank_of(entity, atom_id),
                }
            )
        return rows
    finally:
        hms.close()


def main() -> None:
    """CLI entry.
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--work", default=os.path.join(HERE, "work_v3"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = json.load(open(args.json))
    provider = report["arm"].split("_", 1)[1]
    out: dict[str, Any] = {}
    print("| N | query | overlap | missed | semantic | bm25 | newest window | recency | entity |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, entry in report["sizes"].items():
        for label, q in entry["queries"].items():
            if q["overlap_top20"] >= args.threshold or not q.get("missed"):
                continue
            rows = analyse(provider, int(n), label, q["missed"], args.work)
            out[f"{n}:{label}"] = rows

            def med(name: str) -> str:
                values = [r[name] for r in rows if r[name] is not None]
                return str(int(statistics.median(values))) if values else "-"

            print(
                f"| {n} | {label} | {q['overlap_top20']} | {len(rows)}/{q['oracle_rows']} | {med('semantic')} | "
                f"{med('bm25')} | {med('newest_window')} | {med('recency')} | {med('entity')} |"
            )
            for r in rows:
                print(f"|  | {r['atom_id']} | in_G={r['in_G']} |  | {r['semantic']} | {r['bm25']} | {r['newest_window']} | {r['recency']} | {r['entity']} |")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
