# HMS Elite × EgoLife 真实 LLM 评测（deepseek-flash，仅 Elite，2026-09-11 r3）

运行目录 `results/real/elite_full204_r3`；run_id `20260911T054117Z`；开始 2026-09-11T05:41:17.556125+00:00；模型 `deepseek-flash` @ `https://api.deepseek.com`；并发 1；Python 3.12.14；macOS-27.0-arm64-arm-64bit。

本报告全部数字由 `questions_*.jsonl` / `calls.jsonl` / `retain_elite.jsonl` 重算（`report_real.py`），`summary.json` 只做交叉核对；差异列在文末。凡未实测的量明确标「未实测」，不用估算冒充。

## 0. 实验条件

- 数据：EgoLife A1_JAKE，204 题，5 类题型；窗口 ≤400 字符 / ≤60 s，共 4,851 个窗口，290 个批次（每批 ≤25 窗口，一次 `HMS.add(infer=True)`）。
- 流式协议：按题目 query_time 推进时间线；同一天只放行 query_time 之前的窗口，此前各天全部放行；写入发生在题目之前，题目只能检索到此刻已入库的记忆。
- 两个回答上下文共用同一份物理写入：`elite_rows`（检索命中的记忆行）与 `elite_card`（HMS 状态卡片）；写入成本在两臂之间不重复计费。
- 写入抽取：deepseek-flash 单次复合抽取（`memory_extraction`）；输出 finish_reason=length 的尝试被拒绝，其窗口切片对半重提（递归到单窗口），每次子请求都是计费请求。
- 回答：deepseek-flash，temperature 0，max_tokens 8，与 hms_lite-main 相同的选择题提示词；无效字母 / 空输出 / API 失败一律记为答错。
- 嵌入：本地 fastembed（无 API token，只有本地耗时）。检索：`HMS.search(limit=20)`。
- 计价（deepseek-flash，USD / 1M token，来源 https://api-docs.deepseek.com/quick_start/pricing (captured 2026-09-10)）：非高峰 缓存命中 0.003 / 未命中 0.15 / 输出 0.6；高峰 缓存命中 0.006 / 未命中 0.3 / 输出 1.2。每个请求按其发起时刻所属档位计价；失败与重试请求同样计入。
- 基线对照：`results/real/elite_full204`（首轮，修复前：截断输出未重提、`attributed_to` 校验拒绝整条观察、裸列表回复整批丢弃）。

## 1. 准确率

分母恒为 204 题；检索命中率（hit@20，批次粒度）单列，不当作问答准确率。

「gold 批次全部完整入库的题」由 retain 记录判定（该题全部 gold 批次 windows_lost == 0）；「gold 全丢」= 全部 gold 批次一个窗口都没存进去；「部分」= 介于两者之间。

| 臂 | 答对 / 204 | 准确率 | gold 批次全部完整入库的题 (n) 上的准确率 | gold 全丢的题 | gold 部分丢的题 | hit@20（批次） | 无效输出 | 空输出 | API 失败 |
|---|---|---|---|---|---|---|---|---|---|
| elite_rows | 125 / 204 | 61.3% | 61.3% (n=204) | 0 | 0 | 78.9% | 0 | 0 | 0 |
| elite_card | 126 / 204 | 61.8% | 61.8% (n=204) | 0 | 0 | 77.9% | 0 | 0 | 0 |
| elite_rows（首轮基线，丢失以 added==0 代理） | 109 / 204 | 53.4% | 56.4% (n=156) | 46 | 2 | 63.7% | 0 | 0 | 0 |
| elite_card（首轮基线，丢失以 added==0 代理） | 123 / 204 | 60.3% | 65.4% (n=156) | 46 | 2 | 62.3% | 0 | 0 | 0 |

### 1.1 分题型准确率（答对 / n）

| 题型 | elite_rows | elite_card | elite_rows（基线） | elite_card（基线） |
|---|---|---|---|---|
| EventRecall | 39 / 62 = 62.9% | 41 / 62 = 66.1% | 39 / 62 = 62.9% | 42 / 62 = 67.7% |
| TaskMaster | 27 / 47 = 57.4% | 27 / 47 = 57.4% | 24 / 47 = 51.1% | 29 / 47 = 61.7% |
| RelationMap | 17 / 39 = 43.6% | 19 / 39 = 48.7% | 16 / 39 = 41.0% | 16 / 39 = 41.0% |
| EntityLog | 21 / 33 = 63.6% | 20 / 33 = 60.6% | 15 / 33 = 45.5% | 20 / 33 = 60.6% |
| HabitInsight | 21 / 23 = 91.3% | 19 / 23 = 82.6% | 15 / 23 = 65.2% | 16 / 23 = 69.6% |

### 1.2 失败题归类

`no_evidence_retrieved`：检索结果中没有 gold 批次；`evidence_present_wrong_answer`：gold 批次在上下文里但模型选错；`invalid_output` / `api_error` / `timeout`：输出不是 A–D / 请求失败 / 超时。

| 臂 | evidence_present_wrong_answer | no_evidence_retrieved |
|---|---|---|
| elite_rows | 55 | 24 |
| elite_card | 47 | 31 |

### 1.3 失败题清单（id / 题型 / gold → 预测 / 归类 / gold 批次是否在返回结果中）

<details><summary>elite_rows：79 题</summary>

| id | 题型 | gold → pred | 原始输出 | 归类 | hit@20 |
|---|---|---|---|---|---|
| 6 | TaskMaster | D → A | `A` | evidence_present_wrong_answer | True |
| 18 | RelationMap | A → D | `D` | evidence_present_wrong_answer | True |
| 38 | TaskMaster | D → C | `C` | evidence_present_wrong_answer | True |
| 40 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 61 | RelationMap | B → D | `D` | no_evidence_retrieved | False |
| 64 | TaskMaster | C → D | `D` | evidence_present_wrong_answer | True |
| 84 | RelationMap | B → D | `D` | evidence_present_wrong_answer | True |
| 95 | TaskMaster | A → B | `B` | no_evidence_retrieved | False |
| 101 | TaskMaster | D → B | `B` | evidence_present_wrong_answer | True |
| 104 | EntityLog | A → D | `D` | no_evidence_retrieved | False |
| 105 | RelationMap | C → D | `D` | evidence_present_wrong_answer | True |
| 121 | TaskMaster | C → B | `B` | evidence_present_wrong_answer | True |
| 125 | TaskMaster | B → A | `A` | evidence_present_wrong_answer | True |
| 128 | RelationMap | C → D | `D` | no_evidence_retrieved | False |
| 143 | RelationMap | C → B | `B` | evidence_present_wrong_answer | True |
| 148 | RelationMap | D → A | `A` | no_evidence_retrieved | False |
| 150 | TaskMaster | C → A | `A` | no_evidence_retrieved | False |
| 161 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 164 | TaskMaster | C → B | `B` | evidence_present_wrong_answer | True |
| 186 | RelationMap | C → D | `D` | evidence_present_wrong_answer | True |
| 188 | RelationMap | C → A | `A` | evidence_present_wrong_answer | True |
| 191 | TaskMaster | A → D | `D` | no_evidence_retrieved | False |
| 199 | RelationMap | C → B | `B` | evidence_present_wrong_answer | True |
| 213 | TaskMaster | B → D | `D` | evidence_present_wrong_answer | True |
| 229 | TaskMaster | C → A | `A` | evidence_present_wrong_answer | True |
| 236 | RelationMap | D → A | `A` | evidence_present_wrong_answer | True |
| 244 | RelationMap | A → D | `D` | no_evidence_retrieved | False |
| 252 | TaskMaster | B → D | `D` | no_evidence_retrieved | False |
| 254 | TaskMaster | D → C | `C` | evidence_present_wrong_answer | True |
| 255 | TaskMaster | D → B | `B` | evidence_present_wrong_answer | True |
| 256 | RelationMap | D → B | `B` | evidence_present_wrong_answer | True |
| 267 | TaskMaster | A → C | `C` | evidence_present_wrong_answer | True |
| 268 | TaskMaster | A → D | `D` | evidence_present_wrong_answer | True |
| 271 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 273 | TaskMaster | A → D | `D` | evidence_present_wrong_answer | True |
| 277 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 284 | EventRecall | C → B | `B` | evidence_present_wrong_answer | True |
| 292 | RelationMap | B → D | `D` | no_evidence_retrieved | False |
| 300 | TaskMaster | D → C | `C` | no_evidence_retrieved | False |
| 311 | EventRecall | C → A | `A` | evidence_present_wrong_answer | True |
| 313 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 314 | RelationMap | A → B | `B` | evidence_present_wrong_answer | True |
| 316 | HabitInsight | C → A | `A` | evidence_present_wrong_answer | True |
| 319 | RelationMap | B → C | `C` | evidence_present_wrong_answer | True |
| 321 | RelationMap | B → C | `C` | no_evidence_retrieved | False |
| 330 | RelationMap | A → D | `D` | no_evidence_retrieved | False |
| 331 | RelationMap | C → D | `D` | evidence_present_wrong_answer | True |
| 332 | EntityLog | D → A | `A` | no_evidence_retrieved | False |
| 341 | TaskMaster | B → A | `A` | no_evidence_retrieved | False |
| 356 | EntityLog | B → D | `D` | evidence_present_wrong_answer | True |
| 359 | EntityLog | C → A | `A` | evidence_present_wrong_answer | True |
| 362 | EventRecall | B → A | `A` | no_evidence_retrieved | False |
| 364 | EntityLog | C → B | `B` | evidence_present_wrong_answer | True |
| 367 | EntityLog | A → D | `D` | evidence_present_wrong_answer | True |
| 373 | EntityLog | C → D | `D` | evidence_present_wrong_answer | True |
| 378 | EntityLog | C → D | `D` | evidence_present_wrong_answer | True |
| 388 | EntityLog | B → D | `D` | no_evidence_retrieved | False |
| 393 | EventRecall | B → A | `A` | evidence_present_wrong_answer | True |
| 403 | HabitInsight | B → D | `D` | evidence_present_wrong_answer | True |
| 404 | RelationMap | D → B | `B` | evidence_present_wrong_answer | True |
| 409 | EntityLog | D → C | `C` | evidence_present_wrong_answer | True |
| 414 | EventRecall | B → D | `D` | no_evidence_retrieved | False |
| 416 | EntityLog | C → D | `D` | no_evidence_retrieved | False |
| 419 | EventRecall | D → C | `C` | no_evidence_retrieved | False |
| 426 | EventRecall | B → A | `A` | evidence_present_wrong_answer | True |
| 434 | EventRecall | D → B | `B` | no_evidence_retrieved | False |
| 442 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 449 | EventRecall | C → B | `B` | evidence_present_wrong_answer | True |
| 452 | EventRecall | A → C | `C` | no_evidence_retrieved | False |
| 463 | RelationMap | D → C | `C` | evidence_present_wrong_answer | True |
| 464 | RelationMap | D → B | `B` | no_evidence_retrieved | False |
| 473 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 474 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 475 | EntityLog | A → C | `C` | evidence_present_wrong_answer | True |
| 488 | EventRecall | C → B | `B` | no_evidence_retrieved | False |
| 495 | EventRecall | D → B | `B` | evidence_present_wrong_answer | True |
| 496 | EventRecall | B → A | `A` | evidence_present_wrong_answer | True |
| 499 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 500 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |

</details>

<details><summary>elite_card：78 题</summary>

| id | 题型 | gold → pred | 原始输出 | 归类 | hit@20 |
|---|---|---|---|---|---|
| 6 | TaskMaster | D → A | `A` | evidence_present_wrong_answer | True |
| 18 | RelationMap | A → D | `D` | evidence_present_wrong_answer | True |
| 38 | TaskMaster | D → C | `C` | evidence_present_wrong_answer | True |
| 52 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 61 | RelationMap | B → D | `D` | no_evidence_retrieved | False |
| 64 | TaskMaster | C → D | `D` | evidence_present_wrong_answer | True |
| 68 | EventRecall | A → C | `C` | evidence_present_wrong_answer | True |
| 78 | RelationMap | B → D | `D` | evidence_present_wrong_answer | True |
| 95 | TaskMaster | A → B | `B` | no_evidence_retrieved | False |
| 101 | TaskMaster | D → B | `B` | evidence_present_wrong_answer | True |
| 104 | EntityLog | A → D | `D` | no_evidence_retrieved | False |
| 105 | RelationMap | C → D | `D` | evidence_present_wrong_answer | True |
| 121 | TaskMaster | C → B | `B` | evidence_present_wrong_answer | True |
| 128 | RelationMap | C → D | `D` | no_evidence_retrieved | False |
| 134 | EntityLog | B → C | `C` | evidence_present_wrong_answer | True |
| 143 | RelationMap | C → D | `D` | no_evidence_retrieved | False |
| 148 | RelationMap | D → A | `A` | no_evidence_retrieved | False |
| 150 | TaskMaster | C → D | `D` | no_evidence_retrieved | False |
| 154 | EntityLog | B → D | `D` | evidence_present_wrong_answer | True |
| 161 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 164 | TaskMaster | C → B | `B` | evidence_present_wrong_answer | True |
| 168 | RelationMap | B → D | `D` | no_evidence_retrieved | False |
| 186 | RelationMap | C → D | `D` | evidence_present_wrong_answer | True |
| 191 | TaskMaster | A → C | `C` | no_evidence_retrieved | False |
| 199 | RelationMap | C → B | `B` | evidence_present_wrong_answer | True |
| 203 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 204 | EventRecall | C → B | `B` | evidence_present_wrong_answer | True |
| 213 | TaskMaster | B → D | `D` | evidence_present_wrong_answer | True |
| 217 | TaskMaster | A → B | `B` | no_evidence_retrieved | False |
| 225 | TaskMaster | A → B | `B` | evidence_present_wrong_answer | True |
| 229 | TaskMaster | C → A | `A` | evidence_present_wrong_answer | True |
| 244 | RelationMap | A → D | `D` | no_evidence_retrieved | False |
| 250 | HabitInsight | A → B | `B` | evidence_present_wrong_answer | True |
| 252 | TaskMaster | B → D | `D` | no_evidence_retrieved | False |
| 253 | TaskMaster | B → C | `C` | evidence_present_wrong_answer | True |
| 254 | TaskMaster | D → C | `C` | evidence_present_wrong_answer | True |
| 267 | TaskMaster | A → C | `C` | evidence_present_wrong_answer | True |
| 271 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 273 | TaskMaster | A → D | `D` | no_evidence_retrieved | False |
| 277 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 292 | RelationMap | B → D | `D` | evidence_present_wrong_answer | True |
| 300 | TaskMaster | D → C | `C` | no_evidence_retrieved | False |
| 313 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 314 | RelationMap | A → D | `D` | evidence_present_wrong_answer | True |
| 317 | RelationMap | B → D | `D` | evidence_present_wrong_answer | True |
| 319 | RelationMap | B → C | `C` | evidence_present_wrong_answer | True |
| 321 | RelationMap | B → C | `C` | no_evidence_retrieved | False |
| 330 | RelationMap | A → B | `B` | no_evidence_retrieved | False |
| 331 | RelationMap | C → D | `D` | no_evidence_retrieved | False |
| 332 | EntityLog | D → A | `A` | no_evidence_retrieved | False |
| 350 | EntityLog | C → D | `D` | no_evidence_retrieved | False |
| 356 | EntityLog | B → D | `D` | evidence_present_wrong_answer | True |
| 359 | EntityLog | C → D | `D` | evidence_present_wrong_answer | True |
| 362 | EventRecall | B → D | `D` | no_evidence_retrieved | False |
| 367 | EntityLog | A → D | `D` | evidence_present_wrong_answer | True |
| 373 | EntityLog | C → D | `D` | evidence_present_wrong_answer | True |
| 378 | EntityLog | C → D | `D` | no_evidence_retrieved | False |
| 384 | RelationMap | C → D | `D` | no_evidence_retrieved | False |
| 403 | HabitInsight | B → D | `D` | no_evidence_retrieved | False |
| 408 | TaskMaster | D → B | `B` | evidence_present_wrong_answer | True |
| 414 | EventRecall | B → D | `D` | no_evidence_retrieved | False |
| 416 | EntityLog | C → D | `D` | no_evidence_retrieved | False |
| 417 | EntityLog | D → B | `B` | no_evidence_retrieved | False |
| 426 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 434 | EventRecall | D → B | `B` | no_evidence_retrieved | False |
| 442 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 449 | EventRecall | C → B | `B` | evidence_present_wrong_answer | True |
| 452 | EventRecall | A → C | `C` | no_evidence_retrieved | False |
| 464 | RelationMap | D → B | `B` | no_evidence_retrieved | False |
| 469 | HabitInsight | A → D | `D` | no_evidence_retrieved | False |
| 471 | HabitInsight | A → D | `D` | evidence_present_wrong_answer | True |
| 473 | EventRecall | A → B | `B` | evidence_present_wrong_answer | True |
| 474 | EventRecall | B → C | `C` | evidence_present_wrong_answer | True |
| 475 | EntityLog | A → C | `C` | evidence_present_wrong_answer | True |
| 488 | EventRecall | C → B | `B` | no_evidence_retrieved | False |
| 496 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 499 | EventRecall | B → D | `D` | evidence_present_wrong_answer | True |
| 500 | EventRecall | A → D | `D` | evidence_present_wrong_answer | True |

</details>

## 2. 延迟（毫秒，中位 / P95 / 最大，n）

模型请求耗时与本地处理耗时分开报；并发固定为 1；冷启动（首个请求）与重试单独注明。写入延迟按「时间线步」统计：一步 = 某题之前放行的全部批次。

| 阶段 | 口径 | elite_rows | elite_card |
|---|---|---|---|
| 写入（含抽取模型） | 墙钟，每步 | 18677.5 / 174248.3 / 269464.0 (n=157) | 18677.5 / 174248.3 / 269464.0 (n=157) |
| 写入·模型 | 抽取尝试耗时之和（含重试 / 拆分子请求） | 18381.2 / 172243.2 / 266548.7 (n=157) | 18381.2 / 172243.2 / 266548.7 (n=157) |
| 写入·本地 | 嵌入 + 落库 + 编译 | 296.7 / 2005.2 / 2915.3 (n=157) | 296.7 / 2005.2 / 2915.3 (n=157) |
| 检索·本地 | HMS.search(limit=20) | 11.4 / 28.4 / 61.6 (n=204) | 17.6 / 27.4 / 78.9 (n=204) |
| 上下文构建·本地 | 拼接记忆行 / 状态卡片 | 0.0 / 0.0 / 0.0 (n=204) | 0.0 / 0.0 / 0.0 (n=204) |
| 回答·模型 | deepseek-flash 一次请求 | 780.8 / 1278.2 / 1626.5 (n=204) | 819.4 / 1392.1 / 1812.5 (n=204) |
| 回答·模型（去冷启动） | 剔除首个请求 | 780.8 / 1278.2 / 1626.5 (n=204) | 819.4 / 1392.1 / 1812.5 (n=204) |
| 回答·本地 | 解析字母 | 0.0 / 0.0 / 0.1 (n=204) | 0.0 / 0.0 / 0.2 (n=204) |
| 端到端（不含写入） | 检索 + 上下文 + 回答 | 791.4 / 1305.5 / 1637.5 (n=204) | 837.0 / 1415.8 / 1837.3 (n=204) |
| 端到端（含本步写入） | 写入 + 检索 + 上下文 + 回答 | 12272.9 / 148250.9 / 270207.4 (n=204) | 12625.1 / 148539.5 / 270072.3 (n=204) |

冷启动题数：elite_rows 1, elite_card 1；含写入的步数：157 / 204。

### 2.1 按请求的模型耗时（来自调用账本，逐次请求）

| 阶段:状态 | 中位 / P95 / 最大 (n) |
|---|---|
| answer:ok | 806.5 / 1362.4 / 1812.5 (n=408) |
| extract:finish=length | 21775.2 / 23222.9 / 27434.8 (n=94) |
| extract:ok | 13602.4 / 22214.0 / 27434.8 (n=478) |

写入批次粒度（每次 `add` 批次，n=290）：墙钟 15126.2 / 73195.2 / 121489.9 (n=290)；模型 14900.8 / 72395.7 / 120308.0 (n=290)；本地 231.5 / 706.6 / 1181.8 (n=290)。

## 3. Token 成本（API 返回的实际 usage；无 usage 的请求单列，不估算）

| 阶段 | 臂 | 请求数 | 重试数 | 输入 token | 其中缓存命中 | 其中缓存未命中 | 输出 token | 推理 token | 无 usage 的请求 |
|---|---|---|---|---|---|---|---|---|---|
| answer | elite_card | 204 | 0 | 329,558 | 0 | 329,558 | 204 | 未实测（API 未返回） | 0 |
| answer | elite_rows | 204 | 0 | 68,813 | 0 | 68,813 | 204 | 未实测（API 未返回） | 0 |
| extract | elite | 478 | 0 | 2,721,973 | 857,344 | 1,864,629 | 2,401,278 | 未实测（API 未返回） | 0 |
| organize（整理 / 编译） | elite | 0 | 0 | 本地（无 API token） | — | — | — | — | — |
| embedding | elite | 1269 次 / 9352 段文本 | — | 本地（无 API token） | — | — | — | — | 本地总耗时 73805 ms |

> 抽取阶段的输入 token 中，缓存命中部分是系统提示词与重复前缀在 DeepSeek 侧命中 prompt cache 的量；缓存命中与未命中按不同单价计费（见 §0）。

## 4. 金额与效率（USD，同一计价口径，含失败与重试请求）

| 臂 | 写入（抽取）费用 | 回答费用 | 总费用 | 每题费用 | 每答对一题的费用 | 写入摊销：每窗口 | 写入摊销：每批次 | 未计价请求 |
|---|---|---|---|---|---|---|---|---|
| elite_rows | $3.172024 | $0.019752 | $3.191776 | $0.015646 | $0.025534 | $0.000654 | $0.010938 | 0 |
| elite_card | $3.172024 | $0.094354 | $3.266378 | $0.016012 | $0.025924 | $0.000654 | $0.010938 | 0 |

写入费用两臂共享一次（同一物理库），上表在每臂各计一次以便得到该臂独立部署时的总价；两臂合并跑一次的真实总账 = 写入 $3.172024 + 两臂回答之和。
抽取请求按档位：{'offpeak': 71, 'peak': 407}（peak = 工作日 UTC 01–04 与 06–10）。

### 4.1 同一计价口径（把全部请求按同一档位单价重算；实际账单见上表）

| 轮次 | 实际账单（写入 + 两臂回答） | 全按非高峰：总 / 写入 | 全按高峰：总 / 写入 | 撞 max_tokens 被丢弃的尝试 | 其烧掉的输出 token | 该浪费的费用（非高峰 / 高峰） |
|---|---|---|---|---|---|---|
| 本轮 | $3.286130 | $1.783034 / $1.723033 | $3.566067 / $3.446066 | 94 | 770,048 | $0.462029 / $0.924058 |
| 首轮基线 | $1.110288 | $1.085243 / $1.028661 | $2.170485 / $2.057322 | 64 | 524,288 | $0.314573 / $0.629146 |

> 被截断的尝试按合同必须丢弃再对半重提（截断的 JSON 不可解析），因此这部分输出 token 是纯浪费；它随批次大小和抽取输出的冗长度上升，是压缩抽取输出契约要解决的对象。

| 对照 | 写入费用 | 写入请求数 | 已存窗口 | 每已存窗口的写入费用 |
|---|---|---|---|---|
| 本轮 | $3.172024 | 478 | 4851 / 4851 | $0.000654 |
| 首轮基线 | $1.053422 | 290 | 3332 / 4851 | $0.000316 |

## 5. 稳定性（失败请求保留在结果中）

| 阶段 | 臂 | 请求数 | ok | timeout | http_error | empty | 重试数 | finish_reason=length |
|---|---|---|---|---|---|---|---|---|
| answer | elite_card | 204 | 204 | 0 | 0 | 0 | 0 | 0 |
| answer | elite_rows | 204 | 204 | 0 | 0 | 0 | 0 | 0 |
| extract | elite | 478 | 478 | 0 | 0 | 0 | 0 | 94 |

| 写入指标 | 值 |
|---|---|
| 批次数 / add 请求数（含拆分子请求） | 290 / 478 |
| 因输出截断而拆分的批次 / 截断尝试次数 | 73 / 94 |
| 窗口：计划 / 已入库 / 丢失 | 4851 / 4851 / 0（0.0%） |
| 丢窗口的批次数 / add 失败的批次数 | 0 / 0 |
| 抽出的原子数 | 8,473 |
| 库端告警：JSON 解析失败（含截断） | 94 |
| 库端告警：裸 JSON 列表回复（已按 observations 读取） | 94 |
| 库端告警：其它告警 | 1 |
| elite_rows：空答率 / 无效输出率 / 回答请求失败率 | 0.0% / 0.0% / 0.0% |
| elite_card：空答率 / 无效输出率 / 回答请求失败率 | 0.0% / 0.0% / 0.0% |
| 首轮基线：窗口 计划 / 已入库 / 丢失 | 4851 / 3332 / 1519（31.3%） |
| 首轮基线：截断尝试 / 丢窗口批次 | 0 / 68 |

## 6. 与 summary.json 的交叉核对

- 答对数、总费用、请求数与 summary.json 一致。

## 7. 未实测 / 口径说明

- 推理 token：deepseek-flash 的 usage 若未返回 reasoning_tokens，则标「未实测」，不从输出 token 中反推。
- 嵌入：本地 fastembed，无 API token；只有本地耗时。
- 并发固定 1，因此 P95 反映串行请求的尾部，不含排队。
- 高峰 / 非高峰按请求发起时刻的 UTC 判定；同一轮内可能跨档位，费用按各请求实际档位累加。
- mem0 真实 LLM 臂未运行（按用户指示先只测 HMS Elite）。
- r3 所用库代码快照在 results/real/elite_full204_r3/code_snapshot/（prompts.py、extraction.py）；r3 启动后库里又落地了「规整而不拒绝」与 speaker→actor_id 改动，r3 未使用它们。
- r3 的 94 条「裸 JSON 列表回复」告警与 94 条「JSON 解析失败」一一对应：r3 所用代码版本把解析失败返回的空列表也记为列表回复，二者是同一批被截断的尝试，不是额外的失败。
- r3 抽取请求 773/886 落在高峰档（UTC 06–10 工作日），首轮几乎全在非高峰；跨轮比较费用请看 §4.1 同一计价口径。
- 紧凑抽取契约（`extraction_output = 'compact'`，可选；2026-09-11 上午曾为库默认，同日晚因真实 204 题运行记忆行 56.4% 对 `full` 61.3% 超出 2 个点容忍而回退，见 `EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`）与 r3 前 18 批的同批次对照、204 题零 API 回放 A/B 及默认值决定的依据见 `COMPACT_EXTRACTION_2026-09-11.md`；`results/real/elite_compact_entities_smoke5` 是一次未完成的真实 API 冒烟（18 批只记录了 12 批，无 summary.json，目录内有 `INCOMPLETE.md`），不作为测量结果引用，完成的重跑是 `results/real/elite_compact_entities_smoke5_r2`（run_id `20260911T090326Z`，18/18 批）。
