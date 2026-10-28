# RETRIEVAL_DIAGNOSIS — 检索路径 O(N) 缺陷的数据诊断与施工图

日期：2026-09-10。对象：`hms_elite` 的 `search()` / `compile()` 候选阶段。基线：310 passed / 1 skipped。
本文件是根治方案的**证据层**：每个设计参数都要能回指到下面的一张表。所有原始数据、脚本与本文的表格都在
`scripts/bench/`（脚本）与 `docs/design/bench_results/`（JSON + `tables.md`）。

---

## 0. 方法与环境

| 项目 | 取值 |
|---|---|
| 解释器 | `/opt/anaconda3/bin/python3` = CPython 3.14；numpy 2.4.6；pymilvus 3.0.1（milvus-lite 嵌入）；sqlite 3.53.2 |
| 嵌入器 / LLM | `hash`（64 维）/ `rule`；`add(..., infer=False)`，每条消息一个原始 message 原子 |
| 存储 | 文件库（WAL），`vector_store.provider ∈ {sqlite, milvus(HNSW)}` |
| 语料 | `scripts/bench/profile_search.py::gen_texts(N)`：5 种模板轮转（事件 / 带文件标识符 / 偏好 / 中文 / 站会），10 个人名，18 个月的 `timestamp`（2025-01 … 2026-06），每条含唯一 `item{i}` 词元；N ∈ {300, 3000, 30000} |
| 查询 | `who is Sarah`（entity 路由，硬实体门）；`retry.ts backoff patch`（recall，标识符）；`what happened in March`（temporal，硬时间门 = 2026-03）；`what coffee does alice like`（preference，`obs_type` 类型门→原始原子无 `obs_type`，走 4 级回退阶梯）；`部署 回滚 测试`（recall，CJK） |
| 调用 | `hms.search(q, filters={'user_id':'alice'}, top_k=20, threshold=0.0, explain=True, reference_date='2026-06-15T09:00:00Z')`，预热 1 次，计时 20 次取中位；另跑 1 次 cProfile |
| 归因方法 | 在实例/模块上包装：`StateStore._select_atoms`（拆成 `_execute().fetchall()` = axis SQL 与 `_hydrate_atoms` = hydrate）、`index.search`、`scoring.atom_keyword_tokens / keyword_score / entity_score / recency_score / support_score`、`query.fuse / score_details / atom_to_row / plan_query`、`embedder.embed`、`store.resolve_entities / entity_degree`；Milvus 另包 `MilvusIndex._ann`（gRPC + 响应解析 + `quantize`）、`_in_expr`、`rank_hits`。未被任何桶覆盖的时间记为 residual（= 排序 + 循环开销 + facade） |

**偏差说明**：任务给出的 `scratchpad/bench/`（`run_bench_v2.py`、`harness.py`、mem0 venv、`scale_after.json`）在本轮开始时已不存在（scratchpad 被清空，见 memory），因此规模扫描生成器按上述规则重建；mem0+FAISS 对照组本轮未重跑，引用值 0.16–0.21 ms 来自上一轮记录。为避免再次丢失，本轮脚本与结果都复制进了项目目录。

---

## 1. 现有数据流（读码结论，行号对应当前基线）

```
HMS.search()                              memory/main.py:442
└─ mvmi.query.search()                    mvmi/query.py:496
   ├─ plan_query(query, as_of)            mvmi/planner.py:96      # route / hard gates / content_tokens / entity_hints
   └─ execute_axis_query(aq, ...)         mvmi/query.py:224
      ├─ parse_filters / plan_filters / lower_to_sql            # scope + filter AST → SQL
      ├─ _plan_gates(plan)                mvmi/query.py:78       # time BETWEEN / entity EXISTS / obs_type IN
      ├─ 回退阶梯 _fallback_ladder         mvmi/query.py:114      # 最多 4 级，每级一次 _select
      │   └─ store.axis_select(..., order=AXIS_ORDER, limit=scoring.max_candidates)   memory/storage.py:2331
      │        ├─ SELECT memory_atoms.* … ORDER BY time_anchor DESC, created_at DESC, atom_id ASC LIMIT 2000
      │        └─ _hydrate_atoms(rows)    memory/storage.py:1703 # _links_for（VALUES CTE × 3 个 UNION）+ N 次 MemoryAtom(...)
      ├─ semantic_scores(index, qv, ids)  utils/scoring.py:112   # index.search(qv, candidate_ids=ids, top_k=len(ids))
      │   ├─ sqlite: store.load_vectors(ids)（500 个 id 一块）→ rank_hits：N 次 np.dot   index/sqlite.py:161, index/base.py:117
      │   └─ milvus: 每 1024 个 id 一块 `atom_id in [...]`，limit=len(chunk)，取回向量再 rank_hits   index/milvus.py:500
      ├─ 对每个候选 compute_signals()     utils/scoring.py:255
      │   ├─ atom_keyword_tokens(atom)    utils/scoring.py:84    # 每次查询重新 tokenize + stem 原子文本
      │   ├─ keyword_score / entity_score / recency_score / support_score
      │   └─ fuse()                        utils/scoring.py:311
      ├─ 三次稳定排序（atom_id, 时间, final）
      └─ AtomHit → atom_to_row            # 只对 top_k 行
compile() 候选阶段                       compiler/compile.py:342-361   # 同一 execute_axis_query，limit=candidate_k=50，threshold=0
```

三个结构性事实：

1. `index.search(..., top_k=len(ids))`（scoring.py:138）把索引降级成"给全部候选算分"，ANN 无剪枝可言；Milvus 提供者为了保持逐位一致还要把每个候选的向量传回来再算一遍。
2. `limit=scoring.max_candidates`（query.py:321，默认 2000）在 SQL 里按 `time_anchor DESC` 截断：N > 2000 时较旧原子无论相关性如何都进不了打分——**这是正确性缺陷**。T1 中 `who is Sarah` 在 N=30000 时硬实体门命中 3000 行（T3 uncapped 可见），capped 只拿到最新 2000 行。
3. 五路信号对**每个门控候选**逐个计算，其中 keyword 信号每次查询都重新对原子文本分词、词干化（`content_tokens` 2001 次 / 查询，见 T3 cProfile）。

---

## 2. 时间归因

### T1 sqlite, max_candidates=2000 (shipped default)

| N | query | cands | selects | total ms | axis SQL | hydrate | index.search | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 1 | 1.65 | 0.24 | 0.45 | 0.13 | 0.38 | 0.03 | 0.01 | 0.04 | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.24 |
| 300 | identifier | 292 | 1 | 12.93 | 0.87 | 4.49 | 1.17 | 3.19 | 0.49 | 0.07 | 0.34 | 0.03 | 0.18 | 0.24 | 0.03 | 0.07 | 1.76 |
| 300 | month | 16 | 1 | 0.86 | 0.10 | 0.24 | 0.07 | 0.18 | 0.02 | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.14 |
| 300 | preference | 292 | 4 | 12.39 | 0.99 | 4.27 | 1.12 | 3.13 | 0.52 | 0.00 | 0.34 | 0.03 | 0.16 | 0.23 | 0.03 | 0.06 | 1.51 |
| 300 | cjk | 292 | 1 | 11.99 | 0.81 | 4.27 | 1.13 | 3.12 | 0.46 | 0.00 | 0.33 | 0.03 | 0.16 | 0.23 | 0.03 | 0.04 | 1.39 |
| 3000 | entity | 300 | 1 | 14.95 | 2.29 | 4.63 | 1.33 | 3.71 | 0.27 | 0.14 | 0.34 | 0.03 | 0.18 | 0.36 | 0.03 | 0.07 | 1.56 |
| 3000 | identifier | 2000 | 1 | 89.47 | 7.27 | 30.82 | 8.78 | 21.47 | 3.35 | 0.44 | 2.26 | 0.20 | 1.18 | 1.69 | 0.04 | 0.10 | 11.88 |
| 3000 | month | 166 | 1 | 7.70 | 0.97 | 2.55 | 0.74 | 1.80 | 0.23 | 0.00 | 0.19 | 0.02 | 0.09 | 0.14 | 0.03 | 0.07 | 0.87 |
| 3000 | preference | 2000 | 4 | 89.52 | 7.93 | 30.86 | 8.86 | 21.70 | 3.50 | 0.00 | 2.29 | 0.21 | 1.08 | 1.78 | 0.04 | 0.08 | 11.19 |
| 3000 | cjk | 2000 | 1 | 88.32 | 7.34 | 30.97 | 8.77 | 21.63 | 3.12 | 0.00 | 2.28 | 0.20 | 1.07 | 1.72 | 0.04 | 0.06 | 11.11 |
| 30000 | entity | 2000 | 1 | 119.85 | 26.86 | 35.94 | 11.01 | 25.23 | 1.98 | 0.89 | 2.28 | 0.20 | 1.19 | 1.81 | 0.04 | 0.16 | 12.27 |
| 30000 | identifier | 2000 | 1 | 110.02 | 20.96 | 35.03 | 10.66 | 21.85 | 3.27 | 0.45 | 2.26 | 0.20 | 1.19 | 1.73 | 0.04 | 0.14 | 12.23 |
| 30000 | month | 1666 | 1 | 83.19 | 10.54 | 29.27 | 8.93 | 18.08 | 2.22 | 0.00 | 1.90 | 0.17 | 0.89 | 1.73 | 0.04 | 0.09 | 9.32 |
| 30000 | preference | 2000 | 4 | 122.15 | 34.33 | 34.98 | 10.63 | 21.71 | 3.62 | 0.00 | 2.29 | 0.20 | 1.07 | 1.78 | 0.04 | 0.08 | 11.42 |
| 30000 | cjk | 2000 | 1 | 108.49 | 21.03 | 34.74 | 10.81 | 21.72 | 3.22 | 0.00 | 2.29 | 0.21 | 1.07 | 1.75 | 0.04 | 0.06 | 11.54 |

### T2 per-candidate cost (µs / hydrated candidate), sqlite capped run

| N | query | cands | axis SQL | hydrate | index.search | kw tokens | kw overlap | recency | fuse | explain | sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 8.0 | 15.1 | 4.3 | 12.7 | 1.0 | 1.2 | 0.6 | 0.9 | 43.8 |
| 300 | identifier | 292 | 3.0 | 15.4 | 4.0 | 10.9 | 1.7 | 1.2 | 0.6 | 0.8 | 37.5 |
| 300 | month | 16 | 6.4 | 14.8 | 4.3 | 11.2 | 1.4 | 1.2 | 0.6 | 0.9 | 40.9 |
| 300 | preference | 292 | 3.4 | 14.6 | 3.8 | 10.7 | 1.8 | 1.1 | 0.5 | 0.8 | 36.9 |
| 300 | cjk | 292 | 2.8 | 14.6 | 3.9 | 10.7 | 1.6 | 1.1 | 0.5 | 0.8 | 36.0 |
| 3000 | entity | 300 | 7.6 | 15.4 | 4.4 | 12.4 | 0.9 | 1.1 | 0.6 | 1.2 | 43.7 |
| 3000 | identifier | 2000 | 3.6 | 15.4 | 4.4 | 10.7 | 1.7 | 1.1 | 0.6 | 0.8 | 38.4 |
| 3000 | month | 166 | 5.9 | 15.4 | 4.5 | 10.8 | 1.4 | 1.2 | 0.5 | 0.8 | 40.4 |
| 3000 | preference | 2000 | 4.0 | 15.4 | 4.4 | 10.8 | 1.8 | 1.1 | 0.5 | 0.9 | 39.0 |
| 3000 | cjk | 2000 | 3.7 | 15.5 | 4.4 | 10.8 | 1.6 | 1.1 | 0.5 | 0.9 | 38.5 |
| 30000 | entity | 2000 | 13.4 | 18.0 | 5.5 | 12.6 | 1.0 | 1.1 | 0.6 | 0.9 | 53.1 |
| 30000 | identifier | 2000 | 10.5 | 17.5 | 5.3 | 10.9 | 1.6 | 1.1 | 0.6 | 0.9 | 48.5 |
| 30000 | month | 1666 | 6.3 | 17.6 | 5.4 | 10.9 | 1.3 | 1.1 | 0.5 | 1.0 | 44.2 |
| 30000 | preference | 2000 | 17.2 | 17.5 | 5.3 | 10.9 | 1.8 | 1.1 | 0.5 | 0.9 | 55.2 |
| 30000 | cjk | 2000 | 10.5 | 17.4 | 5.4 | 10.9 | 1.6 | 1.1 | 0.5 | 0.9 | 48.3 |

### T3 sqlite, max_candidates=1000000 (true O(N) cost, no truncation)

| N | query | cands | selects | total ms | axis SQL | hydrate | index.search | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | entity | 30 | 1 | 1.72 | 0.25 | 0.46 | 0.13 | 0.39 | 0.03 | 0.01 | 0.04 | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.28 |
| 300 | identifier | 292 | 1 | 12.45 | 0.83 | 4.39 | 1.17 | 3.19 | 0.48 | 0.07 | 0.34 | 0.03 | 0.17 | 0.24 | 0.03 | 0.07 | 1.43 |
| 300 | month | 16 | 1 | 0.89 | 0.10 | 0.24 | 0.07 | 0.18 | 0.02 | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.15 |
| 300 | preference | 292 | 4 | 12.46 | 0.99 | 4.29 | 1.14 | 3.17 | 0.52 | 0.00 | 0.34 | 0.03 | 0.16 | 0.24 | 0.03 | 0.06 | 1.50 |
| 300 | cjk | 292 | 1 | 12.05 | 0.81 | 4.28 | 1.14 | 3.16 | 0.46 | 0.00 | 0.34 | 0.03 | 0.16 | 0.23 | 0.03 | 0.04 | 1.37 |
| 3000 | entity | 300 | 1 | 15.06 | 2.29 | 4.64 | 1.36 | 3.78 | 0.27 | 0.13 | 0.34 | 0.03 | 0.18 | 0.36 | 0.03 | 0.07 | 1.56 |
| 3000 | identifier | 2917 | 1 | 132.92 | 10.14 | 46.03 | 12.95 | 32.29 | 4.93 | 0.65 | 3.35 | 0.31 | 1.73 | 2.58 | 0.04 | 0.12 | 17.80 |
| 3000 | month | 166 | 1 | 7.82 | 1.01 | 2.55 | 0.76 | 1.83 | 0.23 | 0.00 | 0.19 | 0.02 | 0.09 | 0.14 | 0.03 | 0.07 | 0.91 |
| 3000 | preference | 2917 | 4 | 132.23 | 10.67 | 46.01 | 12.85 | 32.12 | 5.25 | 0.00 | 3.31 | 0.30 | 1.58 | 3.59 | 0.04 | 0.08 | 16.42 |
| 3000 | cjk | 2917 | 1 | 130.89 | 10.25 | 46.30 | 12.64 | 31.91 | 4.68 | 0.00 | 3.32 | 0.30 | 1.58 | 2.60 | 0.04 | 0.06 | 17.22 |
| 30000 | entity | 3000 | 1 | 170.65 | 29.81 | 54.53 | 16.48 | 38.44 | 2.92 | 1.30 | 3.39 | 0.31 | 1.78 | 2.80 | 0.04 | 0.16 | 18.67 |
| 30000 | identifier | 29167 | 1 | 1612.90 | 145.20 | 580.45 | 163.80 | 320.04 | 50.16 | 6.33 | 32.48 | 3.04 | 17.19 | 30.78 | 0.05 | 0.17 | 263.19 |
| 30000 | month | 1666 | 1 | 83.67 | 10.70 | 29.20 | 8.87 | 18.46 | 2.25 | 0.00 | 1.88 | 0.17 | 0.90 | 1.69 | 0.04 | 0.09 | 9.41 |
| 30000 | preference | 29167 | 4 | 1640.51 | 158.66 | 602.36 | 181.59 | 318.97 | 54.37 | 0.00 | 32.58 | 3.07 | 15.71 | 67.33 | 0.05 | 0.10 | 205.73 |
| 30000 | cjk | 29167 | 1 | 1698.74 | 151.97 | 618.59 | 193.50 | 325.06 | 59.22 | 0.00 | 33.26 | 3.11 | 16.08 | 34.97 | 0.05 | 0.08 | 262.83 |

cProfile, N=30000, query `identifier` (one search; cumulative ms / own ms / calls):

| cum ms | own ms | calls | function |
|---|---|---|---|
| 3185.9 | 0.1 | 1 | `hms_elite/memory/main.py:442(search)` |
| 3185.7 | 0.0 | 1 | `hms_elite/mvmi/query.py:496(search)` |
| 3185.7 | 149.0 | 233363 | `bench/profile_search.py:122(wrapper)` |
| 3154.4 | 70.1 | 1 | `hms_elite/mvmi/query.py:224(execute_axis_query)` |
| 1477.5 | 77.8 | 29167 | `hms_elite/utils/scoring.py:255(compute_signals)` |
| 1125.5 | 0.0 | 1 | `hms_elite/mvmi/query.py:309(_select)` |
| 1125.5 | 2.9 | 1 | `hms_elite/memory/storage.py:2331(axis_select)` |
| 1122.6 | 1.9 | 1 | `bench/profile_search.py:142(select_atoms)` |
| 975.5 | 149.3 | 1 | `hms_elite/memory/storage.py:1703(_hydrate_atoms)` |
| 908.2 | 47.8 | 29167 | `hms_elite/utils/scoring.py:84(atom_keyword_tokens)` |
| 738.8 | 35.8 | 29168 | `hms_elite/utils/text.py:453(content_tokens)` |
| 426.4 | 7.8 | 58359 | `site-packages/pydantic/main.py:253(__init__)` |
| 418.6 | 172.2 | 58359 | `~:0(<method 'validate_python' of 'pydantic_core._pydantic_core.SchemaValidator' objects>)` |
| 406.9 | 128.2 | 64168 | `hms_elite/utils/text.py:397(tokenize)` |

### T4 milvus-lite (HNSW), N=3000, max_candidates=2000

| N | query | cands | selects | total ms | axis SQL | hydrate | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | expr build | ANN rpc+parse | rescore | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | entity | 300 | 1 | 60.27 | 2.75 | 5.32 | 4.29 | 0.32 | 0.16 | 0.39 | 0.04 | 0.20 | 0.27 | 0.04 | 0.04 | 43.51 | 0.79 | 0.12 | 2.05 |
| 3000 | identifier | 2000 | 1 | 218.74 | 8.30 | 35.46 | 24.45 | 3.83 | 0.51 | 2.57 | 0.23 | 1.29 | 1.89 | 0.04 | 0.29 | 119.99 | 5.51 | 0.14 | 14.24 |
| 3000 | month | 166 | 1 | 72.31 | 1.81 | 4.50 | 3.20 | 0.40 | 0.00 | 0.34 | 0.03 | 0.15 | 0.23 | 0.06 | 0.04 | 58.93 | 0.68 | 0.12 | 1.82 |
| 3000 | preference | 2000 | 4 | 223.49 | 9.20 | 35.87 | 24.94 | 4.10 | 0.00 | 2.57 | 0.23 | 1.21 | 2.05 | 0.04 | 0.29 | 117.98 | 5.53 | 0.09 | 19.39 |
| 3000 | cjk | 2000 | 1 | 225.87 | 8.68 | 36.55 | 25.29 | 3.83 | 0.00 | 2.65 | 0.24 | 1.23 | 1.90 | 0.05 | 0.30 | 122.95 | 5.71 | 0.07 | 16.43 |

### T5 milvus-lite (HNSW), N=3000, max_candidates=1000000

| N | query | cands | selects | total ms | axis SQL | hydrate | kw tokens | kw overlap | entity | recency | support | fuse | explain | rows | expr build | ANN rpc+parse | rescore | plan+embed+ent | sort/loop residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | entity | 300 | 1 | 68.39 | 3.06 | 6.11 | 4.86 | 0.36 | 0.18 | 0.45 | 0.04 | 0.23 | 0.31 | 0.05 | 0.05 | 49.14 | 0.88 | 0.12 | 2.55 |
| 3000 | identifier | 2917 | 1 | 316.94 | 11.19 | 50.21 | 33.91 | 5.31 | 0.72 | 3.63 | 0.33 | 1.85 | 2.64 | 0.05 | 0.42 | 173.00 | 7.58 | 0.15 | 25.95 |
| 3000 | month | 166 | 1 | 47.67 | 1.26 | 2.88 | 2.00 | 0.24 | 0.00 | 0.22 | 0.02 | 0.10 | 0.14 | 0.04 | 0.03 | 39.12 | 0.42 | 0.08 | 1.11 |
| 3000 | preference | 2917 | 4 | 317.53 | 11.88 | 50.42 | 34.51 | 5.82 | 0.00 | 3.68 | 0.33 | 1.75 | 3.39 | 0.05 | 0.42 | 175.20 | 7.51 | 0.09 | 22.49 |
| 3000 | cjk | 2917 | 1 | 317.81 | 11.26 | 51.74 | 34.14 | 5.20 | 0.00 | 3.64 | 0.33 | 1.74 | 2.72 | 0.04 | 0.41 | 176.76 | 7.70 | 0.07 | 22.06 |

cProfile, N=3000, query `identifier` (one search; cumulative ms / own ms / calls):

| cum ms | own ms | calls | function |
|---|---|---|---|
| 578.4 | 0.1 | 2 | `hms_elite/memory/main.py:442(search)` |
| 578.4 | 17.2 | 23370 | `bench/profile_search.py:122(wrapper)` |
| 578.3 | 0.0 | 1 | `hms_elite/mvmi/query.py:496(search)` |
| 472.1 | 6.5 | 1 | `hms_elite/mvmi/query.py:224(execute_axis_query)` |
| 282.0 | 0.4 | 1 | `hms_elite/utils/scoring.py:112(semantic_scores)` |
| 279.8 | 0.0 | 1 | `hms_elite/index/milvus.py:500(search)` |
| 267.4 | 1.0 | 3 | `hms_elite/index/milvus.py:342(_ann)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/milvus_client/milvus_client.py:420(search)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/decorators.py:514(handler)` |
| 226.7 | 0.0 | 3 | `site-packages/pymilvus/decorators.py:560(handler)` |

### T6 milvus-lite scalar-filter ANN probe

milvus-lite, N=3000, limit=80, 20 runs, median ms:

| index | filter variant | expr bytes | rows matching | hits / recall gap | ids only | + vector payload | top_k=len(chunk) | top_k=len + vector |
|---|---|---|---|---|---|---|---|---|
| FLAT | no_filter | 0 | 3000 | 80 / 0 | 2.57 | 2.76 | - | - |
| FLAT | scope_validity | 139 | 2050 | 80 / 0 | 2.66 | 2.84 | - | - |
| FLAT | scope_validity_scale | 200 | 1174 | 80 / 0 | 2.69 | 2.87 | - | - |
| FLAT | scope_validity_scale_obs | 214 | 167 | 80 / 0 | 2.65 | 2.96 | - | - |
| FLAT | scope_validity_time_gate | 223 | 133 | 80 / 0 | 2.71 | 2.85 | - | - |
| FLAT | id_list_300 | 4511 | 300 | 80 / 0 | 3.15 | 3.23 | 7.469 | 8.17 |
| FLAT | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.19 | 4.36 | 22.986 | 25.876 |
| FLAT | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.68 | 6.01 | 24.36 | 28.158 |
| AUTOINDEX | no_filter | 0 | 3000 | 80 / 0 | 2.98 | 2.74 | - | - |
| AUTOINDEX | scope_validity | 139 | 2050 | 80 / 0 | 2.67 | 3.09 | - | - |
| AUTOINDEX | scope_validity_scale | 200 | 1174 | 80 / 0 | 2.87 | 2.94 | - | - |
| AUTOINDEX | scope_validity_scale_obs | 214 | 167 | 67 / 13 | 2.54 | 2.73 | - | - |
| AUTOINDEX | scope_validity_time_gate | 223 | 133 | 51 / 29 | 2.12 | 2.23 | - | - |
| AUTOINDEX | id_list_300 | 4511 | 300 | 80 / 0 | 3.12 | 3.33 | 3.967 | 4.374 |
| AUTOINDEX | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.26 | 4.42 | 11.753 | 12.759 |
| AUTOINDEX | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.92 | 6.11 | 21.845 | 24.202 |
| HNSW | no_filter | 0 | 3000 | 80 / 0 | 2.75 | 2.66 | - | - |
| HNSW | scope_validity | 139 | 2050 | 80 / 0 | 2.91 | 3.25 | - | - |
| HNSW | scope_validity_scale | 200 | 1174 | 80 / 0 | 3.06 | 3.44 | - | - |
| HNSW | scope_validity_scale_obs | 214 | 167 | 80 / 0 | 2.97 | 3.04 | - | - |
| HNSW | scope_validity_time_gate | 223 | 133 | 60 / 20 | 2.37 | 2.49 | - | - |
| HNSW | id_list_300 | 4511 | 300 | 80 / 0 | 3.14 | 3.40 | 7.063 | 7.897 |
| HNSW | id_list_1024 | 15371 | 1024 | 80 / 0 | 4.21 | 4.43 | 24.13 | 27.439 |
| HNSW | id_list_2000 | 30011 | 2000 | 80 / 0 | 5.84 | 6.12 | 25.547 | 28.648 |

expression draft: `user_id == "alice" and archived == false and valid_from <= "2026-06-15T09:00:00Z" and (valid_to == "" or valid_to > "2026-06-15T09:00:00Z")`; VARCHAR instant comparison count check: {'milvus_count': 2050, 'python_count': 2050, 'equal': True}

### T7 FTS5 keyword channel probe

FTS5 available: True (sqlite 3.53.2); K_kw = 30; 20 runs, median ms:

| N | base insert s | FTS build s | query | matching docs | FTS5 MATCH+bm25 LIMIT 30 ms | LIKE+time index LIMIT 30 ms |
|---|---|---|---|---|---|---|
| 300 | 0.002 | 0.0 | entity | 30 | 0.036 | 0.076 |
| 300 | 0.002 | 0.0 | identifier | 120 | 0.086 | 0.025 |
| 300 | 0.002 | 0.0 | month | 60 | 0.054 | 0.024 |
| 300 | 0.002 | 0.0 | preference | 60 | 0.059 | 0.059 |
| 300 | 0.002 | 0.0 | cjk | 36 | 0.040 | 0.067 |
| 3000 | 0.018 | 0.002 | entity | 300 | 0.202 | 0.132 |
| 3000 | 0.018 | 0.002 | identifier | 1200 | 0.741 | 0.028 |
| 3000 | 0.018 | 0.002 | month | 600 | 0.401 | 0.033 |
| 3000 | 0.018 | 0.002 | preference | 600 | 0.415 | 0.066 |
| 3000 | 0.018 | 0.002 | cjk | 360 | 0.254 | 0.073 |
| 30000 | 0.184 | 0.027 | entity | 3000 | 2.206 | 0.625 |
| 30000 | 0.184 | 0.027 | identifier | 12000 | 7.986 | 0.124 |
| 30000 | 0.184 | 0.027 | month | 6000 | 4.223 | 0.145 |
| 30000 | 0.184 | 0.027 | preference | 6000 | 4.297 | 0.161 |
| 30000 | 0.184 | 0.027 | cjk | 3600 | 2.697 | 0.127 |

tokenize sample: `Sarah patched src/billing/retry.ts 今天部署了支付服务` → `['sarah', 'patched', 'src/billing/retry.ts', '今天', '天部', '部署', '署了', '了支', '支付', '付服', '服务']`; identifier_tokens → `['src/billing/retry.ts']`

### T8 sqlite semantic channel prototype

sqlite semantic channel prototype, dims=64, numpy 2.4.6, median ms (whole scope gated in):

| N | current (load 500-chunks + per-row np.dot, top_k=N) | B: VALUES-CTE join + matmul + argpartition K=60/80/240 | D: B + exact np.dot rescore of the K | C: scope JOIN in SQL, K=60/80/240 | E: cached float64 matrix, K=60/80/240 | rows where matmul ≠ dot (max abs diff) | order differs | D bit-equal |
|---|---|---|---|---|---|---|---|---|
| 300 | 0.76 | 0.12/0.18/0.15 | 0.33/0.92/0.81 | 0.12/0.12/0.15 | 0.01/0.02/0.05 | 81 (5.6e-17) | False | True |
| 3000 | 8.56 | 1.21/1.21/1.22 | 1.37/1.42/1.98 | 1.11/1.14/1.15 | 0.04/0.04/0.07 | 828 (6.9e-17) | False | True |
| 30000 | 97.06 | 27.43/30.24/29.02 | 28.00/26.83/29.97 | 28.16/30.16/28.64 | 1.62/1.14/0.94 | 8349 (1.1e-16) | False | True |

### 2.1 结论（sqlite 提供者）

以 N=3000、门控 2917 行（T3 uncapped，identifier 查询，132.9 ms）为例，各阶段占比：

| 阶段 | ms | 占比 | 每候选 µs（T2） | 性质 |
|---|---|---|---|---|
| hydrate（`_links_for` + `MemoryAtom(...)`） | 46.0 | 35% | 15–18 | O(候选)。T3 cProfile（N=30k）：pydantic `validate_python` 58359 次 = 419 ms（每原子 1 个 MemoryAtom + 1 个 Scope） |
| keyword 分词（`atom_keyword_tokens`） | 32.3 | 24% | 11 | O(候选)。`content_tokens` 每查询对每个候选重跑（N=30k：29168 次 = 739 ms，含 profiler 开销） |
| `index.search`（load_vectors + 逐行 np.dot） | 13.0 | 10% | 4.4–5.5 | O(候选)，与 ANN 无关 |
| axis SQL（scope + LIVE + ORDER BY + LIMIT） | 10.1 | 8% | 3.6（N=3k）→ 10–17（N=30k） | 对整个 scope 排序，随 N 增长 |
| keyword overlap + recency + support + entity | 9.2 | 7% | 3.2 | O(候选) |
| fuse + explain | 4.3 | 3% | 1.5 | O(候选) |
| residual（三次排序 + 循环 + facade） | 17.8 | 13% | ~6 | O(候选 log 候选) |
| plan + embed + entity lookup | 0.1 | <1% | — | 常数 |

- **总成本 ≈ 45 µs × 门控候选数**（explain=True；T2 各行 36–55 µs，含 axis SQL 的摊分）。N=30000 不截断：1.61–1.70 s（29167 候选）；截断到 2000：108–122 ms，但截断改变结果集（`who is Sarah` 的硬实体门在 N=30000 命中 3000 行，capped 只打分最新 2000 行）。
- 门控本身（axis SQL）在 N=30000 时 10–33 ms（`ORDER BY time_anchor DESC LIMIT 2000` 要对整个 scope 排序；preference 路由走 4 级阶梯，每级一次 SQL）。这是新架构里"recency 通道 K_rec"和"硬门先 COUNT"要处理的部分。
- 索引搜索并不是最大头（10%）：把 `index.search` 换成 ANN 而不改打分范围，最多省 10%。**瓶颈是"对多少候选做 hydrate + 分词"，所以设计核心必须是把打分范围从"门控集合"缩到"池"。**
- `entity` 查询在 N=3000（门控 300 行）只要 15 ms、`month` 查询（166 行）7.6 ms：现有硬门已经证明"小池 = 快"，新架构要做的是让无硬门的查询也得到一个有界池。

### 2.2 结论（Milvus 提供者，T4/T5）

- `_ann`（gRPC + 响应解析 + 每块 ≤1024 个向量的 `quantize`）：3 块 = 173 ms（uncapped，2917 候选）/ 2 块 = 120 ms（capped，2000 候选），即每块 ~58 ms；`entity` 查询只有 300 个 id 也要 43–49 ms。T6 探针里同一形状的请求（`atom_id in [1024 ids]`, `limit=1024`, 取回向量）为 26–28 ms，多出的部分是 Python 侧对每个命中 `hit["entity"]["vector"]` 的解析与 `quantize`。
- id 列表表达式：300 个 id = 4.5 KB，1024 个 = 15.4 KB，2000 个 = 30 KB；T6 显示同一集合上标量过滤（139–223 B）为 2.5–3.1 ms，id 列表 3.1–5.9 ms（`limit=80`），而 `limit=len(chunk)` 时 7–28 ms——**成本主要来自 limit 而不是表达式长度**，HNSW 的 ef 被抬到 1024/2000 后等于暴力扫描并回传全部向量。
- rescore（`rank_hits` 对取回向量再算 np.dot）5.5–7.6 ms；`_in_expr` 构造 ≤0.4 ms，可忽略。
- 因此 Milvus 路径 = sqlite 路径的全部成本 + 每块 ~58 ms 的 RPC：capped 219 ms vs sqlite 89 ms；uncapped 317 ms vs 133 ms。ANN 在这条路径上没有任何价值。

### 2.3 过滤式 ANN 的召回缺口（T6，设计上必须处理）

milvus-lite 上 AUTOINDEX / HNSW 带选择性过滤（匹配行 ≤ 5%：`time_gate` 133/3000、`scale_obs` 167/3000）时，`limit=80` 只拿到 51–67 行（缺口 13–29），FLAT 无缺口。这是过滤式 HNSW 的已知行为（图遍历在过滤后候选稀疏时提前耗尽 ef）。含义：**硬门（时间/实体/类型）选出的小集合不能交给 ANN 过滤召回**，要先在 SQLite 里 `COUNT(*)`，不超过池上限就整体入池；只有门控集合大于 K_sem 时才让 Milvus 做原生过滤 ANN，并且 `ef` 至少设为 `max(cfg.ef, 4·K_sem)`。

---

## 3. 目标架构参数的取值依据

有界多通道召回 → 池内全量打分。所有 K 进 `RetrievalConfig`（新增于 `configs/base.py`，挂在 `HMSConfig.retrieval`；`ScoringConfig` 保持 8 个字段不变以维持 §5.1 config dict 兼容，`max_candidates` 语义改为"安全上限"）。

| 参数 | 默认 | 依据 |
|---|---|---|
| `k_semantic = max(4·top_k, 60)` | top_k=20 → 80 | Mem0 的过取规则；T8：sqlite 语义通道在 K=60/80/240 之间成本无差别（1.2 ms@3k，27–30 ms@30k，成本在向量加载 ≈ 0.9 µs/门控 id，不在 K）；T6：Milvus 原生过滤 `limit=80` 2.5–3.0 ms |
| `k_keyword = max(2·top_k, 30)` | 40 | T7：FTS5 `MATCH … ORDER BY bm25 LIMIT 30` 0.04–0.7 ms@≤3k，2.2–8.0 ms@30k（成本随匹配文档数线性，与 K 无关）；CJK 二元组经 unicode61 可检索（`部署` 命中 36/300） |
| `k_entity = max(2·top_k, 30)` | 40 | `atom_entity(entity_id)` 有 `ix_entity_atom`；`ORDER BY time_anchor DESC, atom_id ASC LIMIT 40` 是索引扫描 + 小排序；实体查询在 T1 中 300 行门控只需 15 ms，说明 40 行远低于可承受量 |
| `k_recency = max(2·top_k, 30)` | 40 | `ix_atoms_time`（time_anchor）+ scope 过滤 LIMIT 40；替代现在的 `ORDER BY … LIMIT 2000`（N=30k 时 10–33 ms） |
| `pool_max = 4·k_semantic` | 320 | 每池成员 ≈ 46 µs（T2：hydrate 15–17 + 分词 11 + 信号 3 + fuse/explain 1.5 + 排序/循环 ~8 + index 4.4）→ 320 行 ≈ 15 ms 上限；典型池（4 通道并集去重）100–150 行 ≈ 5–7 ms。与现有 `entity`（300 行 15 ms）一致 |
| 硬门整体入池阈值 | `pool_max` | T6 召回缺口 + T1：门控 ≤ 300 行时直接打分本来就快；`COUNT(*)` 一条 SQL |
| `max_candidates`（安全上限） | 50000 | 只对"硬门集合整体入池"与"通道并集"生效；触发时 `logger.warning`，绝不静默截断。50000 × 46 µs ≈ 2.3 s，是明确可见的告警而非隐藏丢失 |

**关键推论**：池内打分固定为 O(pool)，与 N 无关；N 只影响各通道的索引查找（sqlite 语义通道仍是 O(N) 的向量加载 0.9 µs/id，但不 hydrate、不分词；N=30000 时 ~27 ms，比现状 1.6 s 低 60 倍；T8 变体 E 说明进程内矩阵缓存可再降到 1–1.6 ms，列为第二阶段可选项）。

### 3.1 sqlite / memory 语义通道原型（T8）

- 变体 B（`VALUES` CTE JOIN `atom_vectors` → `np.frombuffer` → 一次 `matrix @ q` → `argpartition(K)`）：0.12 ms@300、1.2 ms@3k、27 ms@30k；现状（500 块 + 逐行 `np.dot` + 全排序）0.76 / 8.6–11.6 / 97 ms。
- **逐位性**：`matrix @ q`（dgemv）与逐行 `np.dot`（ddot）累加顺序不同，28% 的行末位不同（max |Δ| = 1.1e-16），本轮三个规模上排序未变但不能依赖。变体 D 用矩阵乘只**挑选** K 个 id，再对这 K 个用现有 `dot()` 精确重算 → 与现状逐位相等（`exact_rescore_bit_equal = True`）。原型 D 为了简单对 K 行各发一条单行 SELECT（+0.2–0.8 ms）；实现时直接复用已解码的矩阵行，额外成本是 K 次 `np.dot` ≈ K × 2 µs ≤ 0.5 ms。**实现规则：通道用 matmul 选人，池内语义分用 `index/base.py::dot` 算。**
- memory 提供者同样处理：对 `_vectors` 中门控 id 做一次 matmul（numpy 可用时），否则退回现有 `rank_hits` 但 `top_k=K_sem`。

### 3.2 FTS5 可用性与设计（T7）

- 探测 `sqlite3.connect(':memory:').execute("CREATE VIRTUAL TABLE t USING fts5(x)")`：**可用**（sqlite 3.53.2）。写入 `capabilities()['fts5']`；缺失时 keyword 通道退回 `tokens LIKE ? … ORDER BY time_anchor DESC LIMIT K_kw`（T7 的 LIKE 列：0.03–0.6 ms，靠 `ix_atoms_time` 提前停止；无排名但有界）。
- 虚拟表 `atom_fts(atom_id UNINDEXED, tokens, tokenize='unicode61')`，`tokens` = `' '.join(tokenize(text) + topic_tags + entity keys 及其 tokenize)`（与 `atom_keyword_tokens` 同词表，写入时一次算好；CJK 二元组以空格分隔后 unicode61 视为独立词项，已验证）。查询：`'"t1" OR "t2" …'`（每个词项加引号，保护 `.`、`/`、`-`），标识符词元的权重 ×2 在**池内**由现有 `keyword_score` 实现，FTS5 只负责召回顺序（bm25，`ORDER BY rank, atom_id LIMIT K_kw`），并 JOIN `memory_atoms` 施加 scope + LIVE。
- `content_json` 变化不影响 FTS（原子文本不可变，§7 触发器保证）；归档不删 FTS 行（LIVE 在 JOIN 中过滤）。
- 影子表：`atom_fts_data / _idx / _content / _docsize / _config`。`TABLES` 只列基表（22），新增 `VIRTUAL_TABLES = ('atom_fts',)`；`test_schema_migrations_and_capabilities` 改为"22 张基表 + 1 张虚拟表"，`stats()` 仍只数基表；CONTRACT §7 中英同步。`drop_all` 需在基表之前 `DROP TABLE IF EXISTS atom_fts`（影子表随之删除）。
- 迁移：schema v3 = `CREATE VIRTUAL TABLE atom_fts …` + `INSERT INTO atom_fts SELECT …`（回填 30000 行 0.2 s，T7 `fts_build_s`）。`build_ddl(3)` 与 `upgrade_3()` 都要产生逐行相同的 `sqlite_master`（沿用组 T 的 `_MASTER` 比对）。

### 3.3 Milvus 标量过滤表达式草案（T6）

集合新增标量字段（全部 VARCHAR 除 `archived BOOL`）：`user_id, agent_id, run_id`（未设置存 `""`）、`time_anchor, valid_from, valid_to`（UTC ISO，`valid_to` 为 NULL 时存 `""`）、`archived`、`scale_level`、`obs_type`（NULL 存 `""`）。标量索引 `INVERTED`（milvus-lite 接受；`AUTOINDEX` 对标量字段被拒绝，报 code=6）。

```
user_id == "alice" and archived == false
and valid_from <= "2026-06-15T09:00:00Z" and (valid_to == "" or valid_to > "2026-06-15T09:00:00Z")
[and scale_level in ["state","event","L1_micro","L3_macro"]] [and obs_type in ["preference_event","habit"]]
[and time_anchor >= "2026-03-01T00:00:00Z" and time_anchor <= "2026-03-31T23:59:59Z"]
```

- 固定版式 ISO-8601 UTC 字符串的字典序 = 时刻序：`query(count(*))` 与 Python 侧过滤计数一致（2050 = 2050）。`to_utc_iso` 输出含微秒的固定版式，写入 Milvus 前统一规范化为同一版式。
- 实测（N=3000，limit=80，20 次中位）：无过滤 2.6–3.0 ms；scope+有效期 2.7–2.9 ms；加尺度/类型门 2.5–3.1 ms；时间门 2.1–2.7 ms。对比现状每块 22–28 ms（`limit=len(chunk)` + 取回向量）。
- `include_expired=True` 时去掉有效期子句；`scope` 中未设置的键不产生谓词（与 `_scope_where` 一致）。
- 一致性：SQLite 仍是唯一真源。`StateStore.archive_atom / restore_atom / set_valid_to` 通过新增的 `StateStore.attr_listeners`（`list[Callable[[str, dict], None]]`，`HMS.__init__` 注册 `index.update_attrs`）同步 `{'archived', 'valid_to'}`；hydrate 后仍按 SQL 校验有效期，所以 Milvus 标量过期只会**漏检**（被同步更新排除）不会**误检**。`IndexBase.update_attrs(atom_id, attrs)` 为非抽象空实现，sqlite/memory 不覆盖。

---

## 4. 等价性定理（精确表述）

记 G 为旧路径 `axis_select` 在阶梯首个非空级别返回的门控集合（含 `py_expr` 后过滤），`K = min(k_semantic, k_keyword, k_entity, k_recency)`，`P` 为新路径的池。

**定理**：若 `|G| ≤ K` 且 `|G| ≤ pool_max`，则对同一 `(store, index, embedder, aq, scoring, threshold, explain)`，新旧路径返回的 `AxisResult.hits` 序列中每一项的 `(atom.atom_id, score)` 逐位相等，且 `explain=True` 时 `score_details` 的 7 个 Mem0 键（`semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold`）以及 `recency_score, support_score, keyword_backend, axis_gates, primary_axis, route, signals` 逐位相等；`primary_axis`、`projection`、`summary`、`count` 不变。

**成立条件（实现必须保证的 5 条）**：

1. **池 = G**。每个通道在 G 上取前 K_x：`|G| ≤ K_x` 时通道返回整个 G（语义通道对没有向量的 id 也要返回——旧路径对缺向量的候选给 0.0 分并保留；因此语义通道定义为"G 中全部 id 按 (score DESC, atom_id ASC) 取前 K_sem，无向量者 score=0.0 排在尾部"，与 `rank_hits` 现有的 0 分尾部语义一致），并集去重后 `P = G`。硬门整体入池条件 `|G| ≤ pool_max` 同样给出 `P = G`。
2. **语义分逐位相同**。池内语义分用 `index/base.py::dot(vec, query, use_numpy=index.uses_numpy)` 对同一 float32 量化向量计算（3.1 节变体 D），不用 matmul 的结果；Milvus 池内分同样由 `rank_hits` 对取回的 float32 向量重算（现状已如此）。
3. **其余四路信号只依赖 hydrate 后的原子与 plan**：`compute_signals` 不改；`degrees` 仍来自 `resolve_entities + entity_degree`；`n_anchors = len(atom.anchor_ids)`。
4. **融合、阈值、排序键不变**：`fuse` 不改；`final < threshold` 在池内施加；排序仍是三次稳定排序 `(atom_id ASC) → (time_anchor DESC, created_at DESC) → (final DESC)`。
5. **explain 只追加键**：`recalled_by`、`pool_size`、`channel_sizes` 追加在 `signals` 之后，不改变前 14 个键的顺序与值。

**非等价情形（`|G| > K`）**：新路径的池是 G 的子集（各通道前 K 的并集），因此可能丢掉旧路径里排名靠前但在所有通道都排不进前 K 的原子。这类原子在五路信号中每一路都弱于至少 K 个同门候选，其 `final` 必然不高于池内至少 K 个原子（每路 K 个上界，加权和不超过各路上界之和），因此 top_k ≤ K/4 时结果集合不变、只有分数尾部可能变化——这是通道设计的"排名不变性"论证，写成测试 `test_pool_recall_bounds_top_k`（随机语料，N=500, top_k=20，断言新旧 top_k 集合相同）。**现有测试全部满足 `|G| ≤ K`**：`tests/test_mvmi.py`、`test_facade.py`、`test_scenarios.py`、`test_compiler.py` 的语料都 < 60 行（最大的是 `test_mvmi.py` 的 alice_all，约 20 行），因此黄金场景无需改动；实现后用 `store.count_atoms` 断言逐个测试的 |G| ≤ 60 作为守卫。

**测试形态**（`tests/test_retrieval.py::test_equivalence_theorem`）：同一语料（N=50，含实体/标识符/CJK/多月），对 12 个查询 × `explain=True` × 三提供者（milvus importorskip），把 `hms_elite.mvmi.query._legacy_exhaustive()`（保留旧穷举实现作为测试专用参考函数）与新 `execute_axis_query` 的 `(atom_id, score, score_details 前 14 键)` 用 `==` 比较（浮点逐位）。

---

## 5. 施工图

### 阶段 1：有界多通道召回 + 池内打分（sqlite / memory 提供者，FTS5，schema v3）

| 文件 | 函数 / 对象 | 改动 |
|---|---|---|
| `hms_elite/configs/base.py` | 新增 `class RetrievalConfig(HMSModel)`：`k_semantic: int | None = None`（None → `max(4·top_k, 60)`）、`k_keyword`、`k_entity`、`k_recency`（None → `max(2·top_k, 30)`）、`pool_max_factor: int = 4`、`hard_gate_pool: bool = True`、`safety_limit: int = 50000`；`HMSConfig.retrieval: RetrievalConfig`；`ScoringConfig.max_candidates` 文档改为"deprecated alias of retrieval.safety_limit" | 配置 |
| `hms_elite/memory/schema.py` | `SCHEMA_VERSION = 3`；`VIRTUAL_TABLES = ('atom_fts',)`；`build_ddl(version)` 在 version ≥ 3 追加 FTS DDL；新增 `upgrade_3()`（CREATE VIRTUAL TABLE + 回填 INSERT … SELECT）；`MIGRATIONS[3]`；`detect_capabilities()` 加 `'fts5': bool`（每进程探测一次）；`drop_all()` 先删虚拟表 | 模式 |
| `hms_elite/memory/storage.py` | `insert_atom()` 末尾写 `atom_fts`（`fts_tokens(atom)`）；新增 `fts_tokens(atom) -> str`（模块级，复用 `scoring.atom_keyword_tokens` 的词表）；新增 `axis_ids(scope, *, where_sql, params, joins, as_of, include_expired, scale_constraint, atom_types) -> list[str]`（只取 `atom_id`，不 hydrate，供语义通道的门控集合）；新增 `recall_keyword(scope, *, gate_sql, params, terms, as_of, include_expired, limit) -> list[str]`（FTS5 或 LIKE 退路）；新增 `recall_entity(scope, *, gate_sql, params, entity_keys, as_of, include_expired, limit) -> list[str]`；新增 `recall_recent(scope, *, gate_sql, params, as_of, include_expired, limit) -> list[str]`；新增 `count_axis(...)`（`_count_rows` 已有的 fallback 提升为方法）；新增 `attr_listeners: list` 与 `_notify_attrs(atom_id, attrs)`，在 `archive_atom / restore_atom / set_valid_to` 末尾调用；`stats()` 只数 `TABLES` | 存储 |
| `hms_elite/index/base.py` | `IndexBase.search()` 签名扩展：`search(vector, *, candidate_ids=None, top_k, filter: IndexFilter | None = None)`；新增 `@dataclass(frozen=True) IndexFilter(scope, as_of, include_expired, scale_levels, obs_types, time_gate)`；新增非抽象 `update_attrs(atom_id, attrs) -> None`（空实现）与 `recall(vector, *, candidate_ids, top_k, filter) -> list[str]`（默认调 `search` 取 id）；`rank_hits()` 保留（池内精确分）；新增 `topk_ids(query, vectors, *, top_k, use_numpy)`：numpy 时 matmul + argpartition，选出 id 后**再用 `dot()` 精确打分**排序 | 索引契约 |
| `hms_elite/index/sqlite.py` | `search()` 支持 `filter`：`store.axis_ids(...)` 得门控 id → `store.load_vectors_matrix(ids)`（新增于 storage：VALUES CTE JOIN，返回 `(ids, np.ndarray)`，无 numpy 时返回 dict）→ `topk_ids` | sqlite |
| `hms_elite/index/memory.py` | 同上，矩阵从 `_vectors` 现算 | memory |
| `hms_elite/mvmi/query.py` | 新增 `_recall_pool(store, index, embedder, scope, *, gate_sql, params, joins, as_of, include_expired, scale_constraint, atom_types, plan, query_vector, ks) -> tuple[list[str], dict[str, list[str]], dict[str, int]]`（返回池 id、每 id 的 `recalled_by`、`channel_sizes`）；`execute_axis_query()` 的 `_select` 拆为 `_gate_count`（硬门/阶梯用 COUNT）与 `_pool`；`aq.text is None` 路径不变（`get_all` 仍按 `limit` 走 `axis_select`，`limit=aq.limit` 而不是 `max_candidates`）；`ranked` 分支改为只对池 hydrate（`store.get_atoms(pool_ids)` 保持池顺序）+ `semantic_scores(index, qv, pool_ids)`（不变：`top_k=len(pool)`，但 pool ≤ 320）；`_legacy_exhaustive()` 保留为测试参考（私有，不导出）；`score_details` 调用传 `recalled_by / pool_size / channel_sizes` | 查询 |
| `hms_elite/utils/scoring.py` | `score_details(..., recalled_by: list[str] | None = None, pool_size: int | None = None, channel_sizes: dict | None = None)` 在 `signals` 之后追加三键；`atom_keyword_tokens` 不改（池内仍现算，11 µs × ≤320） | 打分 |
| `hms_elite/compiler/compile.py` | 第 2 步 `AxisQuery(limit=cfg.candidate_k)` 不改，天然复用新召回（`k_semantic = max(4·50, 60) = 200`）；删除任何本地穷举（现无） | 编译 |
| `hms_elite/memory/main.py` | `HMS.__init__` 注册 `self.store.attr_listeners.append(self.index.update_attrs)`；`search()` 把 `self.config.retrieval` 传入 `mvmi_search(..., retrieval=...)` | 门面 |
| `hms_elite/hms4eye/`、`server/`、`async_main.py` | 只透传配置，不改签名 | — |
| `docs/design/CONTRACT.md` / `.zh.md` | §5.1 加 `retrieval`；§7 "22 tables" → "22 base tables + 1 virtual table (`atom_fts`, FTS5 shadow tables excluded)"，`SCHEMA_VERSION = 3`，`upgrade_3`；§10.3 `IndexFilter / update_attrs / recall`；§13.3 第 3 步改为召回池算法并写出等价性定理；§13.4 `score_details` 追加三键；§18 测试清单加 `test_retrieval.py` | 契约 |
| `tests/test_retrieval.py`（新） | `test_equivalence_theorem`、`test_pool_recall_bounds_top_k`、`test_no_newest_first_truncation`（N=2500，最旧的实体原子必须被 `who is X` 召回——没有它会失败）、`test_fts5_capability_and_like_fallback`（monkeypatch capabilities 后走 LIKE，结果集合相同）、`test_cjk_keyword_channel`、`test_channel_sizes_and_recalled_by`、`test_safety_limit_warns`（caplog）、`test_schema_v3_fresh_equals_upgraded`（`_MASTER` 逐行比较 v1→v2→v3 与全新 v3）、`test_hard_gate_whole_set_enters_pool` | 测试 |
| `tests/test_foundation.py` | `test_schema_migrations_and_capabilities`：`SCHEMA_VERSION == 3`，`len(TABLES) == 22`，`sqlite_master` 中 `type='table'` 的名字集合 = `TABLES ∪ VIRTUAL_TABLES ∪ 影子表`，`'fts5' in capabilities` | 测试 |
| `tests/test_perf_paths.py` | 语句计数断言若含 `insert_atom` 的 SQL 条数，+1（FTS 写入） | 测试 |

### 阶段 2：Milvus 原生过滤 ANN + 标量一致性

| 文件 | 函数 / 对象 | 改动 |
|---|---|---|
| `hms_elite/index/milvus.py` | `_ensure_collection()`：新增 9 个标量字段 + `INVERTED` 标量索引（`user_id, run_id, scale_level, valid_to, archived`）；`_check_schema()` 要求这些字段存在（旧集合 → `CONFIG_001`，建议改集合名或 `consolidate()` 重建）；`_rows()` 接受可选 `attrs`；新增 `upsert_with_attrs(atom_id, vector, *, embedder, attrs)`；`update_attrs(atom_id, attrs)`（`client.upsert` 只改标量：先 `get` 向量再整行 upsert，或用 `client.update` 若可用）；新增 `_filter_expr(filter: IndexFilter) -> str`（3.3 节草案；`json.dumps` 转义）；`search(..., filter=)`：`filter` 给定时一次 `_ann(expr=_filter_expr, limit=top_k, ef=max(cfg.ef, 4·top_k))`，不再传 id 列表；`candidate_ids` 路径保留（池内重算与旧测试用）；`MilvusIndexConfig` 新增 `ef_recall_factor: int = 4` | Milvus |
| `hms_elite/ingest/atoms.py` | `create_atom()`：`index.upsert` 改为传 `attrs=atom_index_attrs(stored)`（新增于 `index/base.py`：从 MemoryAtom 提取 9 个标量，NULL → `""`） | 摄取 |
| `hms_elite/ops/consolidation.py` | `collect` 阶段的 `index.upsert` 同样传 `attrs` | 巩固 |
| `hms_elite/memory/storage.py` | 阶段 1 的 `attr_listeners` 已覆盖 `archive_atom / restore_atom / set_valid_to`（rollback、feedback、revise、delete、prune 都经这三处） | 存储 |
| `hms_elite/mvmi/query.py` | `_recall_pool()` 的语义通道：`index.recall(vector, filter=IndexFilter(...), top_k=k_semantic)`；硬门 `COUNT ≤ pool_max` 时不调用 ANN | 查询 |
| `tests/test_index_milvus.py` | 新增 `test_milvus_native_filter_matches_sqlite_pool`（三提供者同池同分）、`test_archive_hides_from_all_providers_and_rollback_restores`（archive 后三提供者 search 不再返回；`rollback` 后再次返回）、`test_milvus_scalar_schema_guard`、`test_milvus_filter_expr_escaping`（含引号/反斜杠的 run_id） | 测试 |
| `docs/design/CONTRACT.md` / `.zh.md` | §10.3 Milvus 段：标量字段、`_filter_expr`、`update_attrs`、`ef` 规则、召回缺口说明 | 契约 |

两阶段都要过：ruff、`scan_en.py == 0`、`pytest`（< 60 s，本轮基线 14 s）、黄金场景原样。

---

## 6. 数据文件

- `docs/design/bench_results/profile_search.json`（sqlite，cap 2000）、`profile_search_uncapped.json`、`profile_search_milvus.json`、`profile_search_milvus_uncapped.json`、`milvus_scalar_probe.json`、`fts5_probe.json`、`sqlite_semantic_probe.json`、`tables.md`（本文第 2 节的表由 `scripts/bench/render_tables.py` 生成）。
- 复现：`/opt/anaconda3/bin/python3 scripts/bench/profile_search.py --provider sqlite --sizes 300,3000,30000`（约 2 分钟）；`--provider milvus --sizes 3000`；`milvus_scalar_probe.py --index-types FLAT,AUTOINDEX,HNSW`；`fts5_probe.py`；`sqlite_semantic_probe.py`；`render_tables.py`。
