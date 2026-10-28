"""Measure milvus-lite filter+ANN latency with pushed-down scalars (scope, validity, archived, scale, obs_type) vs id lists.
测量 milvus-lite 在标量下推（范围、有效期、归档、尺度、obs_type）过滤 + ANN 与 id 列表过滤两种方式下的延迟。
"""

# Purpose (EN): Evidence for the Milvus native-filter design in RETRIEVAL_DIAGNOSIS.md (expression draft + medians).
# 目的（中）：为 RETRIEVAL_DIAGNOSIS.md 中的 Milvus 原生过滤设计提供证据（表达式草案 + 中位延迟）。

# Input (EN): N rows (default 3000) of seeded random unit vectors with synthetic scalar columns; index types to test.
# 输入（中）：N 行（默认 3000）带种子的随机单位向量与合成标量列；待测索引类型。

# Output (EN): results/milvus_scalar_probe.json and a stdout table (median / p90 ms per variant).
# 输出（中）：results/milvus_scalar_probe.json 与 stdout 表（每个变体的中位 / p90 毫秒）。

# State effect (EN): Writes one milvus-lite .db under the bench work directory; removed at the end.
# 状态影响（中）：在 bench 工作目录写一个 milvus-lite .db；结束时删除。

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = 64
AS_OF = "2026-06-15T09:00:00Z"
# ISO-8601 UTC strings with a fixed layout compare lexicographically like instants; NULL valid_to is stored as "".
# 固定版式的 ISO-8601 UTC 字符串按字典序比较等价于按时刻比较；NULL 的 valid_to 存为 ""。
NULL_TO = ""
SCALES = ["state", "event", "L1_micro", "L2_meso", "L3_macro", "entity", "relation"]
OBS_TYPES = ["", "preference_event", "habit", "constraint", "fact"]
# Which scalar index kind each probe collection ended up with (None = vector index only).
# 每个探测集合最终使用的标量索引类型（None = 只有向量索引）。
SCALAR_INDEX_USED: dict[str, str | None] = {}


def unit_vectors(n: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        raw = [rng.gauss(0.0, 1.0) for _ in range(DIMS)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        out.append([x / norm for x in raw])
    return out


def rows_for(n: int) -> list[dict[str, Any]]:
    """Synthetic rows: 80% user alice, 10% archived, 10% expired, timestamps over 18 months.
    合成行：80% 属于用户 alice，10% 已归档，10% 已过期，时间戳跨 18 个月。
    """
    vectors = unit_vectors(n, seed=7)
    rows = []
    for i in range(n):
        m = i % 18
        ts = f"{2025 + m // 12:04d}-{m % 12 + 1:02d}-{1 + (i * 7) % 28:02d}T{(i * 3) % 24:02d}:00:00Z"
        rows.append(
            {
                "atom_id": f"atom_{i:06d}",
                "vector": vectors[i],
                "embedder": "hash:64",
                "user_id": "alice" if i % 10 else "bob",
                "agent_id": "",
                "run_id": f"run-{i % 7}",
                "time_anchor": ts,
                "valid_from": ts,
                "valid_to": ("2026-01-01T00:00:00Z" if i % 10 == 3 else NULL_TO),
                "archived": (i % 10 == 5),
                "scale_level": SCALES[i % len(SCALES)],
                "obs_type": OBS_TYPES[i % len(OBS_TYPES)],
            }
        )
    return rows


def build(client: Any, name: str, rows: list[dict[str, Any]], index_type: str) -> None:
    from pymilvus import DataType

    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("atom_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMS)
    schema.add_field("embedder", DataType.VARCHAR, max_length=128)
    for scalar in ("user_id", "agent_id", "run_id"):
        schema.add_field(scalar, DataType.VARCHAR, max_length=256)
    for instant in ("time_anchor", "valid_from", "valid_to"):
        schema.add_field(instant, DataType.VARCHAR, max_length=40)
    schema.add_field("archived", DataType.BOOL)
    schema.add_field("scale_level", DataType.VARCHAR, max_length=16)
    schema.add_field("obs_type", DataType.VARCHAR, max_length=32)
    params = {"M": 16, "efConstruction": 200} if index_type == "HNSW" else {}
    # Scalar indexes on the gate columns: try INVERTED first, then no scalar index (milvus-lite rejects some kinds).
    # 门控列上的标量索引：先试 INVERTED，再退到不建标量索引（milvus-lite 会拒绝部分类型）。
    last_error: Exception | None = None
    for scalar_kind in ("INVERTED", None):
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type=index_type, metric_type="COSINE", params=params)
        if scalar_kind is not None:
            for scalar in ("user_id", "run_id", "scale_level", "valid_to", "archived"):
                index_params.add_index(field_name=scalar, index_type=scalar_kind)
        try:
            client.create_collection(name, schema=schema, index_params=index_params, consistency_level="Strong")
            SCALAR_INDEX_USED[name] = scalar_kind
            break
        except Exception as exc:  # noqa: BLE001 - probe records the failure and tries the next ladder step
            last_error = exc
            if client.has_collection(name):
                client.drop_collection(name)
    else:
        raise RuntimeError(f"could not create {name}: {last_error}")
    for start in range(0, len(rows), 512):
        client.upsert(name, rows[start : start + 512])
    client.flush(name) if hasattr(client, "flush") else None


def scope_validity_expr(*, user_id: str, as_of: str, scales: list[str] | None, obs_types: list[str] | None) -> str:
    """The native-filter draft: scope AND live(as_of) AND optional scale / obs_type gates.
    原生过滤表达式草案：范围 AND 有效期(as_of) AND 可选的尺度 / obs_type 门控。
    """
    parts = [
        f'user_id == {json.dumps(user_id)}',
        "archived == false",
        f'valid_from <= {json.dumps(as_of)}',
        f'(valid_to == "" or valid_to > {json.dumps(as_of)})',
    ]
    if scales:
        parts.append(f"scale_level in [{', '.join(json.dumps(s) for s in scales)}]")
    if obs_types:
        parts.append(f"obs_type in [{', '.join(json.dumps(s) for s in obs_types)}]")
    return " and ".join(parts)


def bench(client: Any, name: str, query: list[float], expr: str, limit: int, repeat: int, index_type: str, *, output_vector: bool = False) -> dict[str, Any]:
    search_params: dict[str, Any] = {"metric_type": "COSINE"}
    if index_type == "HNSW":
        search_params["params"] = {"ef": max(64, limit)}
    # The current provider asks Milvus to return the stored vector of every hit (for re-scoring); measure that payload.
    # 现有提供者让 Milvus 返回每个命中的已存向量（用于重算分数）；测量这一载荷的代价。
    fields = ["vector"] if output_vector else ["atom_id"]
    times = []
    hits = 0
    for _ in range(repeat + 1):
        t0 = time.perf_counter()
        result = client.search(name, data=[query], filter=expr, limit=limit, output_fields=fields, search_params=search_params)
        dt = time.perf_counter() - t0
        hits = len(result[0]) if result else 0
        times.append(dt)
    times = times[1:]
    # How many rows satisfy the filter at all: hits < min(limit, matching) means the filtered ANN lost recall.
    # 满足过滤条件的总行数：hits < min(limit, matching) 表示过滤式 ANN 丢了召回。
    matching = int(client.query(name, filter=expr or "", output_fields=["count(*)"])[0]["count(*)"])
    return {
        "median_ms": round(statistics.median(times) * 1000, 3),
        "p90_ms": round(sorted(times)[int(0.9 * (len(times) - 1))] * 1000, 3),
        "hits": hits,
        "matching_rows": matching,
        "recall_gap": max(0, min(limit, matching) - hits),
        "expr_bytes": len(expr.encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--index-types", default="AUTOINDEX,HNSW")
    parser.add_argument("--out", default=os.path.join(HERE, "results", "milvus_scalar_probe.json"))
    args = parser.parse_args()
    from pymilvus import MilvusClient

    workdir = os.path.join(HERE, "work", "milvus_probe")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    client = MilvusClient(uri=os.path.join(workdir, "probe.db"))
    rows = rows_for(args.n)
    query = unit_vectors(1, seed=99)[0]
    report: dict[str, Any] = {"n": args.n, "limit": args.limit, "repeat": args.repeat, "variants": []}
    alice_live_ids = [r["atom_id"] for r in rows if r["user_id"] == "alice" and not r["archived"] and r["valid_to"] == NULL_TO]
    for index_type in args.index_types.split(","):
        name = f"probe_{index_type.lower()}"
        t0 = time.perf_counter()
        build(client, name, rows, index_type)
        build_s = time.perf_counter() - t0
        variants = {
            "no_filter": "",
            "scope_validity": scope_validity_expr(user_id="alice", as_of=AS_OF, scales=None, obs_types=None),
            "scope_validity_scale": scope_validity_expr(user_id="alice", as_of=AS_OF, scales=SCALES[:4], obs_types=None),
            "scope_validity_scale_obs": scope_validity_expr(user_id="alice", as_of=AS_OF, scales=["state"], obs_types=["preference_event", "habit"]),
            "scope_validity_time_gate": scope_validity_expr(user_id="alice", as_of=AS_OF, scales=None, obs_types=None)
            + ' and time_anchor >= "2026-03-01T00:00:00Z" and time_anchor <= "2026-03-31T23:59:59Z"',
            "id_list_300": "atom_id in [" + ", ".join(json.dumps(i) for i in alice_live_ids[:300]) + "]",
            "id_list_1024": "atom_id in [" + ", ".join(json.dumps(i) for i in alice_live_ids[:1024]) + "]",
            "id_list_2000": "atom_id in [" + ", ".join(json.dumps(i) for i in alice_live_ids[:2000]) + "]",
        }
        for label, expr in variants.items():
            limit = args.limit if not label.startswith("id_list") else min(args.limit, 1024)
            entry = bench(client, name, query, expr, limit, args.repeat, index_type)
            entry.update({"index_type": index_type, "variant": label, "build_s": round(build_s, 2), "scalar_index": SCALAR_INDEX_USED.get(name)})
            entry["median_ms_with_vector_payload"] = bench(client, name, query, expr, limit, args.repeat, index_type, output_vector=True)["median_ms"]
            if label.startswith("id_list"):
                # The current provider asks for top_k=len(ids) per chunk AND the stored vectors; measure both forms.
                # 当前提供者对每块请求 top_k=len(ids) **并**取回已存向量；两种形式都测。
                chunk_len = min(int(label.split("_")[-1]), 1024)
                entry["median_ms_topk_eq_len"] = bench(client, name, query, expr, chunk_len, args.repeat, index_type)["median_ms"]
                entry["median_ms_topk_eq_len_with_vector"] = bench(client, name, query, expr, chunk_len, args.repeat, index_type, output_vector=True)["median_ms"]
            report["variants"].append(entry)
            print(f"{index_type:9s} {label:26s} median={entry['median_ms']:7.2f} ms p90={entry['p90_ms']:7.2f} hits={entry['hits']}/{entry['matching_rows']} gap={entry['recall_gap']} expr={entry['expr_bytes']}B scalar_idx={entry['scalar_index']}", file=sys.stderr)
    # Verify correctness of the string comparison on VARCHAR instants: count via query must equal Python filter.
    # 校验 VARCHAR 时刻的字符串比较正确性：query 计数必须等于 Python 侧过滤结果。
    name = f"probe_{args.index_types.split(',')[0].lower()}"
    expr = scope_validity_expr(user_id="alice", as_of=AS_OF, scales=None, obs_types=None)
    got = client.query(name, filter=expr, output_fields=["count(*)"])[0]["count(*)"]
    expected = sum(1 for r in rows if r["user_id"] == "alice" and not r["archived"] and r["valid_from"] <= AS_OF and (r["valid_to"] == NULL_TO or r["valid_to"] > AS_OF))
    report["varchar_instant_compare"] = {"milvus_count": int(got), "python_count": expected, "equal": int(got) == expected}
    report["expression_draft"] = expr
    print(f"varchar instant compare: milvus={got} python={expected}", file=sys.stderr)
    client.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
