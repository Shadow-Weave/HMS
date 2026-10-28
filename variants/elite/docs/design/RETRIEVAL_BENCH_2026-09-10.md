# RETRIEVAL_BENCH — 有界多通道召回的规模基准（2026-09-10）

对象：`hms_elite` 的 `search()`，池化召回落地后的验收。对照组 mem0ai 2.0.20 + FAISS（IndexFlat，cosine）。
原始数据：`docs/design/bench_results/scale_v3.json`（三臂合并）、`miss_analysis_3000.json` / `miss_analysis_30000.json`（漏检归因）；
脚本：`scripts/bench/run_bench_v3.py`（副本在 scratchpad/bench/run_bench_v3.py）。数字都是本机实测，不做修饰。

## 0. 方法

| 项目 | 取值 |
|---|---|
| 机器 / 解释器 | Apple Silicon（darwin 27.0）；elite 两臂 CPython 3.14.6（anaconda，numpy 2.4.6，pymilvus 3.0.1 + milvus-lite，sqlite 3.53.2）；mem0 臂 CPython 3.12.14（mem0ai 2.0.20 + faiss，同一 `hms_elite.embeddings.hash.HashEmbedder` 注入） |
| 嵌入器 | `hash@64`，三臂共用同一实现 |
| 语料 | `scripts/bench/profile_search.py::gen_texts(N)`，N ∈ {300, 3000, 30000}；5 种模板轮转、10 个人名、18 个月时间戳、每条含唯一 `item{i}` |
| 写入 | `add(..., infer=False)`；elite 传 `timestamp`，mem0 的开源 SDK 拒绝 `timestamp`（`ValueError`），因此 mem0 的 created_at = now |
| mem0 落盘 | FAISS `_save`（每次 add 把整个 docstore 重新 pickle，O(N)）只在三个计时段（前 10 / 中 10 / 末 10）内真实执行，其余 add 跳过，末尾再保存一次；这是对 mem0 有利的让步，否则 N=30000 的装载要 ~30 分钟 |
| 查询 | `who is Sarah`（entity 路由、硬实体门）、`retry.ts backoff patch`（recall、标识符）、`what happened in March`（temporal、硬时间门）、`what coffee does alice like`（preference，4 级回退阶梯）、`部署 回滚 测试`（recall、CJK） |
| 调用 | `search(q, filters={'user_id':'alice'}, top_k=20, threshold=0.1, reference_date='2026-06-15T09:00:00Z')`，预热 1 次，计时 20 次取中位与 p90；explain 另调一次取池溯源 |
| oracle | elite：`tests/test_retrieval.py::_exhaustive_search(max_candidates=10**6)`——旧穷举路径（门控集合全部水合、五路信号融合）的 top-20；重叠率 = \|新 ∩ oracle\| / \|oracle\|。mem0：对 FAISS 内全部向量做暴力 cosine 的 top-20（与 IndexFlat 同度量，**不是** elite 的融合分），只用来说明 FAISS 自身的召回，不能与 elite 的重叠率横向比较 |

## 1. 结果表

下表是 CONTRACT 变更日志 90–96 的精简（`axis_lite` 轻量打分行 + 只水合最终 top_k、单条 `UNION ALL` 召回语句、FTS5 最新窗口、`entity_gate_sql` 两种形式、`select_top_k` 行内 `dot`、页缓存、Milvus 主键读块）落地之后的复测（elite 两臂 CPython 3.14.6；mem0 臂 3.12.14；三臂同日重跑）。池化召回刚落地时的数字保留在 §5 作对照；§3 的阶段计时与 §4 的漏检归因取自那一次测量，精简没有改变通道顺序，只改变了池内打分与水合的成本，因此 §4 的结论不变。

| arm | N | add first10 (ms) | add mid10 | add last10 | search 中位 (ms) | search p90 | 重叠率均值 | 重叠率最小 |
|---|---|---|---|---|---|---|---|---|
| mem0 | 300 | 0.419 | 0.845 | 1.437 | 0.204 | 0.21 | 0.98 | 0.95 |
| mem0 | 3000 | 0.383 | 5.205 | 9.971 | 0.225 | 0.23 | 0.99 | 0.95 |
| mem0 | 30000 | 0.335 | 47.887 | 95.615 | 0.323 | 0.334 | 0.96 | 0.9 |
| elite_sqlite | 300 | 0.926 | 0.691 | 0.758 | 2.71 | 5.92 | 1.0 | 1.0 |
| elite_sqlite | 3000 | 0.914 | 0.789 | 0.825 | 2.055 | 3.401 | 0.8 | 0.45 |
| elite_sqlite | 30000 | 0.942 | 0.819 | 0.803 | 3.406 | 7.856 | 0.62 | 0.2 |
| elite_milvus | 300 | 2.387 | 0.993 | 1.078 | 11.617 | 11.807 | 1.0 | 1.0 |
| elite_milvus | 3000 | 2.542 | 1.151 | 1.129 | 49.202 | 50.261 | 0.81 | 0.5 |
| elite_milvus | 30000 | 1.666 | 1.233 | 1.237 | 14.38 | 53.175 | 0.65 | 0.2 |

search 中位 = 5 个查询各自中位数的中位；p90 = 全部 100 次计时运行的 p90。add 三段中位：mem0 的 mid/last 段随 N 线性增长（`_save` 的 O(N) pickle：N=30000 时 48–96 ms/条），elite 两臂与 N 无关（0.7–1.2 ms/条）。

逐查询（中位 ms / 重叠率 / 池大小与通道）：

| N | 查询 | elite_sqlite | elite_milvus | mem0 |
|---|---|---|---|---|
| 300 | entity | 0.90 / 1.0 / 30 gate | 2.04 / 1.0 / 30 gate | 0.20 / 0.95 |
| 300 | identifier | 4.09 / 1.0 / 292 gate | 11.7 / 1.0 / 292 gate | 0.20 / 1.0 |
| 300 | month | 0.87 / 1.0 / 16 gate | 1.15 / 1.0 / 16 gate | 0.20 / 1.0 |
| 300 | preference | 4.12 / 1.0 / 292 gate | 11.7 / 1.0 / 292 gate | 0.21 / 1.0 |
| 300 | cjk | 2.71 / 1.0 / 292 gate | 11.6 / 1.0 / 292 gate | 0.20 / 0.95 |
| 3000 | entity | 3.41 / 1.0 / 300 gate | 13.4 / 1.0 / 300 gate | 0.22 / 1.0 |
| 3000 | identifier | 2.06 / 1.0 / 94 (sem 80, kw 40, ent 40) | 49.2 / 1.0 / 94 | 0.23 / 1.0 |
| 3000 | month | 1.83 / 1.0 / 166 gate | 7.55 / 1.0 / 166 gate | 0.23 / 1.0 |
| 3000 | preference | 2.01 / 0.55 / 117 (sem 80, kw 40) | 49.4 / 0.55 / 117 | 0.23 / 1.0 |
| 3000 | cjk | 2.06 / 0.45 / 120 (sem 80, kw 40) | 49.4 / 0.5 / 120 | 0.22 / 0.95 |
| 30000 | entity | 6.65 / 0.55 / 149 (sem 80, kw 40, ent 40) | 53.2 / 0.7 / 148 | 0.32 / 0.95 |
| 30000 | identifier | 7.13 / 1.0 / 155 (sem 80, kw 40, ent 40) | 17.0 / 1.0 / 153 | 0.32 / 0.95 |
| 30000 | month | 2.07 / 1.0 / 118 (sem 80, rec 40) | 13.2 / 1.0 / 118 | 0.32 / 0.9 |
| 30000 | preference | 2.73 / 0.2 / 120 (sem 80, kw 40) | 13.9 / 0.2 / 120 | 0.32 / 1.0 |
| 30000 | cjk | 3.41 / 0.35 / 120 (sem 80, kw 40) | 14.4 / 0.35 / 120 | 0.31 / 1.0 |

## 2. 判据（逐条，达标与否）

1. **search@30000 三臂都在个位数毫秒量级或更低**：**两臂达标，milvus 臂未达标**。mem0 0.32 ms；elite_sqlite 中位 3.41 ms，五个查询 2.07–7.13 ms（精简前中位 15.9 ms、只有时间门查询是个位数）；elite_milvus 中位 14.4 ms（13.2–53.2 ms，精简前 27.0 ms），成本仍是 milvus-lite 的 RPC 固定开销。
2. **Elite 两臂 3000 → 30000 的增长显著低于线性**（N ×10）：**达标**。elite_sqlite 2.06 → 3.41 ms = ×1.7；elite_milvus 49.2 → 14.4 ms = ×0.29（N=3000 时 identifier / preference / cjk 三个查询的池内 `candidate_ids` 读取各约 49 ms，所以 3000 反而比 30000 慢——这说明 Milvus 臂的成本是 RPC 次数与每次 RPC 的固定开销，不是 N）。对照改造前（RETRIEVAL_DIAGNOSIS T3 / bench_pool）：sqlite 30000 从 172–210 ms（上限 2000）/ 1.6 s（不截断）降到 2.1–7.1 ms，即 ×24–100 / ×470；对照池化刚落地时（§5）：13.6–19.6 → 2.1–7.1 ms（中位 15.9 → 3.41，×4.7）。
3. **重叠率 ≥ 0.9**：**N=300 达标（1.0）；N=3000 与 30000 未达标**（sqlite 0.80 / 0.62，milvus 0.81 / 0.65；逐查询：identifier 与 month 始终 1.0，entity 在 3000 为 1.0、30000 为 0.55 / 0.7，preference 0.55 / 0.2，cjk 0.45–0.5 / 0.35）。精简只改池内打分与水合，不改通道顺序，重叠率的变动来自 hash 嵌入并列分下的排序抖动；原因见 §4，判据不改。
4. init 模板克隆（组 P 探针）：文件库 `apply_migrations` 2.13 ms → `clone_schema`（backup API）1.13 ms；内存库 1.41 → 0.28 ms；`StateStore(文件)` 端到端 1.43 ms、`StateStore(':memory:')` 0.29 ms（15 次中位）。克隆结果与迁移阶梯的 sqlite_master 逐行、`user_version`、`page_count` 相同（`test_file_store_migrates_before_wal_switch`）。

## 3. elite_sqlite@30000 的成本构成（stage 计时，10 次均值，ms）

| 查询 | 总计 | 语义通道（索引缓存掩码 + 矩阵乘） | 关键词（FTS5 两阶段） | 实体（G 枚举 / 通道） | 探测 / 新近度 | 水合（~120–160 原子） | 五路信号 |
|---|---|---|---|---|---|---|---|
| entity | 17.7 | 1.2 | 4.2 | 6.3（G=3000 条链接，一次 CROSS JOIN） | — | 4.3 | 0.8 |
| identifier | 18.1 | 1.5 | 4.6 | 2.7 | 0.3 | 5.4 | 2.0 |
| month | 9.3 | 0.8 | 0.1 | — | 1.3（time 索引探测 + 新近度） | 4.8 | 1.1 |
| preference | 15.3 | 1.5 | 8.0（6000 个匹配的 bm25 排序） | — | 0.3（三个空级别各 0.1） | 3.5 | 1.0 |
| cjk | 14.6 | 1.3 | 6.2 | — | 0.2 | 4.3 | 1.6 |

固定的 O(pool) 部分（水合 + 信号 + 融合 + 解释）约 6–8 ms/查询：水合每原子 ~19 µs（pydantic 校验 2 次/原子 + JSON 列 + 一条 VALUES-CTE 链接查询），信号每原子 ~7 µs。通道部分：FTS5 的 bm25 排序随匹配文档数线性（3000 个匹配 1.7 ms，6000 个 4.9 ms，12000 个 9 ms）；实体门的 G 枚举随链接数线性（3000 条 ~4–6 ms，每行一次主键查找 + 有效期判定）；语义通道经进程内索引缓存（掩码 0.5 ms + 30000×64 float64 矩阵乘 1 ms）与 N 弱相关。要进入个位数毫秒，需要把水合从 pydantic 全校验换成 `model_construct`（本轮未做，改变 §8 读路径的校验语义）并把 FTS 排序改为不带 `atom_id` 次序键（快 40%，但失去确定性）。

这一轮对 SQLite 规划器的实测结论（决定了 `index_hint` 而不是 ANALYZE）：无统计时 SQLite 对带时间 / obs_type / 尺度门控的探测一律走 scope 索引并扫完整个 scope（每级 5–6 ms；preference 阶梯的三个空级别合计 17–22 ms）；运行 `ANALYZE` 后规划器改用 `ix_atoms_validity` 的范围（valid_from<?）驱动**所有**无门控查询，无门控新近度从 4.7 ms 变成 21 ms、COUNT 从 ~2 ms 变成 17 ms（ENABLE_STAT4 未编译，范围估计没有直方图）。因此不收集统计，改由 mvmi 按级别门控指名索引（`schema.TIME_INDEX / OBS_TYPE_INDEX / SCALE_INDEX`）。

elite_milvus 的成本是 milvus-lite 的 RPC：ungated 过滤 HNSW 一次 16–18 ms（ef=320，valid_from 无标量索引）；实体门 3000 个 id 分 3 块、每块 ef=1024 的过滤搜索 ~37 ms（不把 ef 提到块大小时 30000 行上 1024 个 id 的过滤只返回 5 个命中——这是本轮修掉的召回缺口）；带过滤的 `count(*)` 是全量扫描（556 ms@30000），已从热路径移除（`gate_count=-1`）。

## 4. 重叠率低于 0.9 的归因（`miss_analysis_*.json`）

对每个漏掉的 oracle 原子，按各通道自身顺序定位它的名次（K_sem=80、K_kw=40、K_ent=40、K_rec=40，new 路径 hard_gate_pool=True 下 recall / preference / entity 路由**不运行**新近度通道）：

| N | 查询 | 漏掉 | 语义名次中位 | bm25 名次中位 | 新近度名次中位 | 漏掉原子的信号 |
|---|---|---|---|---|---|---|
| 30000 | entity（G=3000） | 10/20 | 194 | 1898 | 89 | keyword 1.0、entity 1.0、semantic 0.45、recency 0.49——与命中的原子在 keyword / entity 上完全并列 |
| 30000 | preference（G=29167，kw 匹配 5833） | 16/20 | 466 | 2929 | 491 | keyword 0.667、semantic 0.45、recency 0.85 |
| 30000 | cjk（kw 匹配 3501） | 14/20 | 6620 | 883 | 179 | keyword 0.333、semantic 0.15–0.18、recency 0.86 |
| 3000 | preference | 13/20 | 270 | 248 | 43 | keyword 0.667、semantic 0.40、recency 0.85 |
| 3000 | cjk | 12/20 | 604 | 150 | 42 | keyword 0.333、semantic 0.07、recency 0.85 |

结论：漏检的原子全部在 G 内（`in_G: True`），它们在词汇信号上与命中的原子**完全并列**（模板语料：3000–6000 个原子的 keyword / entity 分相同），oracle 的 top-20 由并列之外的两个弱信号决定——hash 嵌入的语义分（分布很窄，0.40–0.46 之间的差异决定名次）与新近度（0.25 权重）。四条通道各自的顺序都看不到这些原子：语义通道按 hash 点积取前 80，漏掉的在第 100–600 名（30000 时 cjk 在第 6000 名以后，因为 CJK 查询的 hash 向量与全体语料的点积几乎不可分）；关键词通道按 bm25 取前 40，而 bm25 偏爱短文档与稀有词，与池内的词元重叠分无关，漏掉的在第 150–3000 名；新近度通道没有运行，且即使运行，漏掉的原子在 30000 时也在第 89–491 名（K_rec=40 之外），只有 3000 时（第 41–43 名）勉强够到。identifier（`retry.ts` 独有词元 + 标识符 ×2）与 month（时间门把 G 缩到 1667 个，语义通道在 G 内取前 80）两类查询重叠率始终 1.0——当某一路信号能把候选区分开时，有界召回与穷举一致；当 top-20 由数千个并列原子之间的微小语义差异决定时，任何 K=40–80 的有界通道都不能复现穷举的排序，这是有界召回的固有损失，不是通道实现的缺陷。改用有语义区分度的嵌入器（fastembed / openai）会直接改变这一结论；本基准按任务要求固定 hash@64，判据不改。

## 5. 与改造前的对照与本轮改动清单

- 改造前（RETRIEVAL_DIAGNOSIS）：sqlite search@30000 = 172–210 ms（`max_candidates=2000` 截断，且 N > 2000 时静默丢旧记忆）/ 1.6 s（不截断）；Milvus 2 块以上 id 列表 ANN 每块 ~58 ms。
- 池化召回刚落地时（本文件首版，三臂同一天上午测量）：sqlite 8.21 / 4.54 / 15.9 ms（N = 300 / 3000 / 30000，p90 10.1 / 9.30 / 21.9，重叠率 1.0 / 0.75 / 0.60），Milvus 20.8 / 87.0 / 27.0 ms（p90 24.0 / 103 / 73.1，重叠率 1.0 / 0.81 / 0.46），mem0 0.20 / 0.22 / 0.35 ms；sqlite@30000 逐查询 13.6–19.6 ms，Milvus 21–73 ms；无截断、无静默丢失（`test_no_newest_first_truncation`）。
- 精简（变更日志 90–96）之后：§1 的表；sqlite 2.71 / 2.06 / 3.41 ms，Milvus 11.6 / 49.2 / 14.4 ms。
- 代码改动（每项都有回归测试，见 CONTRACT §18.3 64e）：`axis_ids(order=None)` 不排序探测；`axis_select(atom_ids=)` 用 VALUES-CTE CROSS JOIN 水合（之前 `atom_id IN (...)` 让规划器扫 scope 索引，160 个原子 20 ms → 0.35 ms SQL）；`keyword_candidates / entity_candidates` 接受级别门控并在 G 内搜索，关键词通道两阶段（AND 后 OR）且先在 atom_fts 内排序再对窗口施加门控；实体门降级为非相关 IN 子查询并经 atom_entity 一次枚举 G；`index_hint`（TIME / OBS_TYPE / SCALE 索引）；SQLite 索引提供者的进程内矩阵 + 标量缓存（以 `data_generation / rebuild_generation / data_version` 为键、按 rowid 增量、`supports_filter=True`）；新近度通道只在 temporal / 硬时间门 / hard_gate_pool=False / 兜底时运行；池内语义分复用通道分数；`AxisResult.retrieval`；`HMS.search` 传递 `config.retrieval`；摄取 `upsert(attrs=atom_index_attrs(...))`；`IndexFilter.atom_type_in` 与 Milvus `atom_type` 标量；Milvus 重开集合时 `load_collection`、id 块 ef 提到块大小、热路径不再 `count(filter)`；文件库也经模板克隆初始化。
