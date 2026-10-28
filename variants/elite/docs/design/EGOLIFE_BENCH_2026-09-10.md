# EGOLIFE_BENCH — EgoLife 流式检索对照（2026-09-10，final）

对象：`hms_elite`（三种配置）对官方 mem0ai 2.0.20（两种向量库），全离线。结果目录 `benchmarks/egolife/results/final/`：`summary.json` / `summary.md`（全部汇总数）、`questions_<arm>.jsonl`（逐题记录）、`arm_<arm>.json`（逐臂计数与每步保留集摘要）、`manifest.json`（数据哈希、规则、共享变量探针）、`../final.log`（运行日志）。代码 `benchmarks/egolife/`（`run_final_split.sh` 复现；`doc_tables.py` 生成本文三张表）。上一版（`results/full/`）与审计（`results/audit1/`）保留，本版修掉审计判定的全部 major：gold 改为包含定义、并列计错、补 mem0 自带 BM25 的混合臂。

## 方法

1. 数据：`data/EgoLifeQA_A1_JAKE_TEXT_ONLY.json` 204 题（sha256 `3ad50376…`，manifest.json `qa_sha256`）；`Transcript/A1_JAKE/DAY1..7` 148 个 SRT，cue 绝对时刻 = 文件名 HHMMSScc + 相对偏移（毫秒 // 10），公式逐字复制自 `hms_lite-main/benchmarks/egolife.py`（loader.py）。
2. 窗口：每文件按 cue 顺序折叠为 ≤400 字符 / ≤60 s、只含英文行的窗口，共 4851 个（windows.jsonl，sha256 `b5684575…`）；五臂 retain 的是同一份窗口与 window_id。
3. 流式规则：题目按（日，query_time）排序；前几日全量（对方的 23595999 哨兵），同日只释放 `window.end ≤ query_time` 的窗口（比对方的 `cue.start ≤ query_time` 保守，跨越提问瞬间的窗口最多晚 60 s 才可见；manifest `cutoff_rule`）。审计核对：无未来泄漏（0 个更晚日 / 0 个 end > query）。
4. 共享变量（summary.json `shared`）：同一 HashEmbedder(64)（mem0 侧用 EmbeddingBase 薄壳调同一函数，20 段探针 == 且 sha256 相同）、同一时间线（五臂每步保留集 sha256 逐位相同，`retained_digest_per_question_identical: true`）、`top_k=20`、`threshold=0.1`、同一阅读器。每臂独立进程，mem0 侧 LLM 为一调即抛的桩（`llm_calls=0`）。
5. 臂：`mem0_faiss`（FAISS IndexFlatIP，仅语义；faiss 无 keyword_search）、`mem0_milvus`（milvus-lite 本地文件，mem0 自带 BM25 混合：`(semantic + bm25)/2`）、`elite_matched`（w_keyword = w_entity = w_recency = w_support = 0，仅语义）、`elite_full`（默认融合权重，infer=False）、`elite_infer`（默认权重 + 离线 RuleLLM，infer=True）。除 elite_infer 外全部 `infer=False`，两边存的都是原始转写窗口。
6. 主指标 = 证据命中：gold 窗口 = **包含** target_time 的窗口（每题 1–4 个，中位 1；多时刻题取任一），hit@k = 任一 gold 进入去重后的 top-k；`nb-hit@20` 是 ±60 s 邻域的次要口径（上一版的口径）。204 题的 gold 在提问时全部已保留，分母 204，无剔除。
7. 阅读器代理：四选一由五臂共用的确定性词法阅读器（选项 token 计数 × IDF + 0.5 × 同窗题干 token；证据 4000 字符）产生；IDF 只在提问时已保留的窗口上计算（每步与 retained_count 相等）。并列或全零 = 无效 = **计错**（对方规范：invalid 计错，另报 valid-only）。oracle 上限 = 同一阅读器只读 gold 窗口文本。随机 25%。
8. 阈值说明：0.1 对 mem0 两臂与 elite_matched 作用在语义分上；对 elite_full / elite_infer 作用在融合分上，其 recency + support 下限已高于 0.1，因此不删任何行（manifest `threshold_note`）。
9. 延迟：`search 首次` = retain 之后的第一次调用（流式口径）；另有 3 次连发中位（预热）。retain 每窗口成本按调度四分位给出，因为 mem0 的 FAISS 存储每次 add 都整库落盘（O(n)）。

## 主表（all：分母 204；first-50：文件顺序前 50 题，分母 50；数值 %；来源 summary.json `arms.<arm>.all / .first50`）

| 臂 | hit@1 | hit@5 | hit@20 | MRR | nb-hit@20 (±60 s) | reader acc first-50 (valid-only, n) | reader acc all (valid-only, n) | oracle 上限 all (valid-only, n) | oracle first-50 | 随机 |
|---|---|---|---|---|---|---|---|---|---|---|
| mem0_faiss | 2.0 | 6.4 | 10.3 | 0.037 | 14.2 | 26.0 (34.2, 38) | 23.5 (29.3, 164) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| mem0_milvus | 8.3 | 15.2 | 18.1 | 0.117 | 27.9 | 34.0 (37.8, 45) | 27.9 (32.4, 176) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_matched | 2.0 | 6.4 | 10.8 | 0.037 | 14.7 | 26.0 (34.2, 38) | 23.5 (29.3, 164) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_full | 7.8 | 20.1 | 31.4 | 0.134 | 39.2 | 28.0 (34.2, 41) | 25.5 (29.9, 174) | 65.2 (76.9, 173) | 70.0 | 25.0 |
| elite_infer | 10.3 | 20.6 | 33.8 | 0.154 | 40.7 | 32.0 (45.7, 35) | 23.5 (35.6, 135) | 65.2 (76.9, 173) | 70.0 | 25.0 |

读数：仅语义的两臂（mem0_faiss、elite_matched）在同一嵌入上几乎相同（top-20 列表 204 题里 200 题逐位相同，差异来自 float32/float64 累加与 Elite 规划器的硬时间门；`results/audit1/timestamp.md`）。词法通道是差距来源：mem0 自带 BM25（milvus）把 hit@20 从 10.3 提到 18.1，Elite 的 FTS5 + 实体 + 新近度融合提到 31.4；infer=True 再 +2.4（hit@20）。阅读器一列并列数（`summary.md` ties 列：40 / 28 / 40 / 30 / 69）说明 elite_infer 的原子证据更短、更常无效，其 valid-only 35.6% 与 all 23.5% 的差就是这 69 题。

## 分类型（hit@5，%，all；来源 summary.json `arms.<arm>.by_type.<type>.hit@5`）

| 题型 | n | mem0_faiss | mem0_milvus | elite_matched | elite_full | elite_infer |
|---|---|---|---|---|---|---|
| EntityLog | 33 | 6.1 | 24.2 | 6.1 | 30.3 | 27.3 |
| EventRecall | 62 | 9.7 | 14.5 | 9.7 | 22.6 | 22.6 |
| HabitInsight | 23 | 8.7 | 13.0 | 8.7 | 13.0 | 13.0 |
| RelationMap | 39 | 2.6 | 12.8 | 2.6 | 17.9 | 17.9 |
| TaskMaster | 47 | 4.3 | 12.8 | 4.3 | 14.9 | 19.1 |

## 成本（来源 summary.json `arms.<arm>.timing / .stats`；库字节 = close 后磁盘文件，mem0 = 索引 + JSON docstore + history.db，elite = 单个 SQLite 文件含向量、FTS5、图表与历史）

| 臂 | 窗口数 | 存储行 | retain 总 ms | retain ms/窗口 均值 | retain ms/窗口 题中位 | Q1..Q4 ms/窗口 | search 首次 中位 ms | search 首次 p90 ms | search 3 次中位 ms | embedding 调用 | LLM 调用 | LLM token (est.) | 库字节 (close 后) | 文件明细 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem0_faiss | 4851 | 4,851 | 72014.5 | 14.85 | 13.62 | 4.73 / 11.41 / 16.83 / 24.52 | 0.41 | 0.49 | 0.29 | 5,463 | 0 | 0 | 10,357,331 | egolife.faiss 1,241,901; egolife.json 6,125,350; history.db 2,990,080 |
| mem0_milvus | 4851 | 4,851 | 4566.7 | 0.94 | 0.99 | 0.93 / 0.96 / 0.97 / 0.93 | 79.22 | 137.61 | 71.83 | 5,463 | 0 | 0 | 16,452,110 | milvus.db 13,462,030; history.db 2,990,080 |
| elite_matched | 4851 | 4,851 | 7572.1 | 1.56 | 1.54 | 1.40 / 1.55 / 1.59 / 1.69 | 5.39 | 9.16 | 4.00 | 5,463 | 0 | 0 | 31,793,152 | hms.db 31,793,152 |
| elite_full | 4851 | 4,851 | 7582.7 | 1.56 | 1.53 | 1.37 / 1.58 / 1.62 / 1.69 | 5.45 | 9.20 | 3.94 | 5,463 | 0 | 0 | 31,813,632 | hms.db 31,813,632 |
| elite_infer | 4851 | 30,351 | 84658.4 | 17.45 | 16.78 | 8.89 / 14.91 / 19.19 / 25.39 | 4.75 | 10.68 | 3.35 | 10,488 | 4,851 | 20,222,089 | 96,993,280 | hms.db 96,993,280 |

读数：mem0_faiss 的 retain 成本随库大小线性增长（Q1 4.7 → Q4 24.5 ms/窗口，`_save` 每次重写整个 docstore），elite 两个 infer=False 臂与库大小无关（1.4 → 1.7）；mem0_milvus 写入最快但每次 search 要两次 milvus-lite RPC（语义 + BM25，JSON 标量过滤），首次调用中位 79 ms。elite_infer 的 30,351 行是 RuleLLM 从 4851 个窗口派生的原子，LLM token 是估算值（离线规则 LLM，`usage.estimated`）。

## 不能得出的结论

本机无任何 LLM API，所有臂的四选一准确率都是同一个词法阅读器的代理值，与真实模型的 EgoLife 成绩无可比性；hash@64 嵌入是词法哈希而非语义嵌入，"仅语义" 两臂的低命中反映的是嵌入器而不是引擎；`infer=False` 让两边都只存原始窗口，这对 Elite（其设计以 infer=True 的原子 / 状态为主）是不利方向，而 elite_infer 的 +2.4 hit@20 又是在离线规则 LLM 上得到的，不能外推到真实 LLM；mem0 在 milvus-lite 上的混合检索没有 spaCy 词形还原（未安装，退回原文），也没有实体加成，是其 2.0.20 离线安装的原样行为。
