# 紧凑抽取契约（`ingest.extraction_output = 'compact'`）：契约、测量与默认值决定（2026-09-11，同日晚回退）

本文记录紧凑输出契约的最终形态（含 `entities` 提示，CONTRACT §6.5 变更记录第 99–105 行）、三组测量（真实 API 同批次冒烟、204 题零 API 回放 A/B、本地推导与 LLM 输出的一致率）、compact 相对 `full` 丢失的内容与在本数据集上未产生影响的原因、默认值的决定及其规则、剩余注意事项与复现命令。每个数字后括注来源文件；路径相对 `benchmarks/egolife/`，除非另有说明。凡未实测的量写「未测」，不用估算冒充。

## 0. 结论

- `ingest.extraction_output` 默认为 `full`。2026-09-11 上午按两条件规则切为 `compact`（CONTRACT 变更记录第 102 行），同日晚按追加的第三个条件（真实 204 题运行的记忆行准确率与 `full` 相差不超过 2 个点）回退为 `full`（第 105 行，§6）：compact 初版记忆行 53.9%（根因：模型把观察 i 与候选 i 配对判 NONE，1,685 个 NONE），DECISIONS 措辞重写加 `verdict_min_similarity` 守卫后的修订版（`results/real/elite_compact_v2_full204`）56.4%（115/204）/ 卡片 62.7%（128/204），hit@20 77.9% / 79.4%，NONE 15，窗口丢失 0，空闲价 ¥4.35，每批 7.3 s；`full` r3 61.3%（125/204）/ 61.8%，hit@20 78.9%，¥11.56，14.9 s；mem0 56.9%（`EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`）。前两个条件仍满足：(a) 回放 A/B 两臂 hit@20 均 161/204（§3）；(b) 带 `entities` 提示的最终提示词完成了同 18 批的真实 API 冒烟——`results/real/elite_compact_entities_smoke5_r2`（run_id `20260911T090326Z`，有 `summary.json`）：每次请求输出 token 中位 2,793 / 最大 4,312，同批次 `full` 为 5,631 / 8,192（中位 −50%）；触顶 0/18 vs 8/34 次；每次请求模型耗时中位 8.7 s vs 17.1 s（−49%）；窗口丢失 0；5 题答对 rows 3/5、card 4/5（`full` 3/5、3/5）；账单 $0.089961（§2）。第一次尝试 `results/real/elite_compact_entities_smoke5` 在 12/18 批处中断，由 `_r2` 取代。（2026-09-18 起默认 lines，见变更记录第 108 行）
- `compact` 仍完整支持，推荐用于成本敏感的批量灌入（写入费用约为 `full` 的 0.38、每批模型耗时 0.49）与以状态卡片上下文为阅读器输入的场景（卡片 62.7% 对 61.8%）；读记忆行上下文、或需要 LLM 直接给出 `confidence` / `topic_tags` / `sentiment` 的场景保持默认的 `full`（compact 把置信度推导为 0.7，§1.2；§5 列出 compact 丢失的内容）。两种情况下提示词都逐字节不变。

## 1. 契约

### 1.1 LLM 输出的键（`hms_elite/configs/prompts.py::COMPACT_OBSERVATION_KEYS`，9 个）

| 键 | 必要性 | 规则（两个紧凑提示词 `OBSERVATION_EXTRACTION_COMPACT_PROMPT` / `MEMORY_EXTRACTION_COMPACT_PROMPT` 的 OUTPUT FORMAT 与 guideline） |
|---|---|---|
| `id` | 必需 | 观察在本次回复内的编号，供 `supersedes_hint_ids` 引用 |
| `event_idx` | 必需 | 见证消息在批内的下标；越界或不可读 → 该条跳过并记一条警告（§12.3 见证规则不变） |
| `text` | 必需 | 第三人称、单一陈述、≤25 词；同一事实不在多条中重复 |
| `type` | 必需 | `fact / preference_event / habit / constraint / temporal_fact` |
| `speaker` | 必需（可为 null） | 具名说话人，否则 null；`speaker → actor_id` 映射与 `full` 相同（消息 `name` 优先） |
| `entities` | 可选 | 陈述点名的事物（人、地、物、工具、组织），0–6 个短字符串；没有则省略。只会加宽实体轴，绝不收窄 |
| `supersedes_hint_ids` | 可选 | 仅在出现取代线索时给出 |
| `when` | 可选 | 仅当陈述自带与消息日期不同的日期 / 时间 |
| `valid_to` | 可选 | 仅当陈述给出失效时间 |

OUTPUT FORMAT 中的示例条目（两个提示词逐字相同，CONTRACT §6.5 引用）：

```json
{"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": null,
 "entities": ["black coffee"], "supersedes_hint_ids": ["0"], "when": "2026-04-18", "valid_to": null}
```

提示词长度（字符，`tests/test_extraction_compact.py` 断言 compact < 0.75 × full；`full` 的两个提示词 sha256 钉死不变）：

| 任务 | compact | full | 比值 |
|---|---|---|---|
| `observation_extraction` | 2,978 | 4,105 | 0.7255 |
| `memory_extraction` | 5,195 | 6,936 | 0.7490 |

### 1.2 本地推导表（`hms_elite/ingest/derive.py::expand_compact_item`，由 `ingest/extraction.py::_compact_observation_from_item` 调用；推导值再经 `full` 契约同一套规整函数，词表不会拒绝）

| 字段 | 推导 |
|---|---|
| `attributed_to` | 见证消息的角色侧（assistant 消息 → `assistant`，否则 `user`） |
| `subject` | `llms.rule.statement_triple`：`User / The user / 用户` → `user`，`Assistant / 助手` → `assistant`，`The team / Project / 团队 / 项目` → `project`，句首专名 → 该名（`Sarah joined …` → `Sarah`）；否则具名 `speaker`；constraint 无主语 → `project`；再否则按角色 |
| `predicate` | `statement_triple` 按类型用规则 LLM R1–R9 的线索表：preference_event → `likes / dislikes / prefers`（否定与 `no longer` 翻转极性；切换动词 → relation `supersession`；`over / rather than` → `prefers` + `comparison`；无线索 → `likes`，情感为负 → `dislikes`）；constraint → `must_not` / `must`；habit → `does`；temporal_fact / fact → 动词词表命中，否则 `is` / `says` 类回退 |
| `objects` | 同一解析：偏好 / 身份 / 居住宾语，constraint 的 `constraint_objects`，habit 去掉习惯线索后的余部，temporal_fact 与回退 fact 取陈述中的实体减去主语 |
| `relation` | `supersedes_hint_ids` 非空 → `supersession`；否则陈述自身线索（切换 / 迁居动词、`no longer` → `supersession`；替代 → `comparison`）；否则 habit → `habit`，constraint → `constraint`，其余 `single_fact` |
| `confidence` | 0.7（`derive.COMPACT_CONFIDENCE`，即 `full` 契约缺省值） |
| `date_observed` | `when` 可解析为日期时取之（datetime 取 UTC 日期），否则见证消息自身日期，时间戳不可读时才用批次 `observation_date` |
| `valid_from` | `when` 为完整 datetime 时取之，否则 None，由 `atom_from_observation` 按 `full` 模式填时间锚 |
| `valid_to` | 条目 `valid_to` 经 `_datetime`，下限处理同 `full` |
| `sentiment` | `utils.text.sentiment(text)`（规则 LLM 词表） |
| `topic_tags` | `llms.rule.observation_topics(text, type)`：`TOPIC_TABLE` 命中加该类型的 `preference` / `constraint`，最多 4 个 |
| `entity_mentions` | `derive.union_mentions(entities, llms.rule.mention_texts(text, entity_extractor), [speaker])`：LLM 的 `entities` 提示在前（经 `extraction._entities_hint` → `_str_list` 规整：None → `[]`，字符串 → 单元素，dict 取文本；列表位置出现数字 / dict 时记警告并忽略，绝不跳过该条），再取实体抽取器在 `text` 中的命中，最后取 `speaker`；不分大小写去重、保留首次写法 |
| `speaker → actor_id` | 不变（§12.5） |

## 2. 真实 API 冒烟：相同 18 批（5 题 `6, 18, 20, 29, 32`，395 窗口）

条件：deepseek-flash，temperature 0，`max_tokens` 8,192，thinking 关闭，并发 1，`extraction_mode = single`，本地 fastembed bge-small-en-v1.5，`HMS.search(top_k=20, threshold=0.1)`。`full` 臂取 `results/real/elite_full204_r3`（290 批）的前 18 批——其 `batch_plan` 前 18 项（batch_id 与 window_ids）与两次 compact 冒烟的 18 批计划逐项相同（各目录 `manifest.json`）。`finish_reason = length` 的尝试被拒绝并把窗口切片对半重提（递归到单窗口），每次子请求都计费。

### 2.1 三臂对照（18 批）

| 臂 | 目录 | 批 | 请求次数 | 输出 token 中位 / 最大（每次请求） | 输出 token 合计 | 触顶（length）次数 | 对半重提批数 | 每批模型耗时中位 / 最大（ms） | prompt token 合计（缓存命中） | 存入原子（事件） | 5 题答对 rows / card |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full（r3 前 18 批） | `results/real/elite_full204_r3` | 18 | 34 | 5,631 / 8,192 | 194,811 | 8 | 6 | 17,147 / 94,879 | 203,189（59,648） | 655（ADD 655，UPDATE 5） | 3/5 / 3/5 |
| compact（第 99 行提示词，无 `entities`） | `results/real/elite_compact_smoke5` | 18 | 18 | 2,002.5 / 5,106 | 42,233 | 0 | 0 | 7,141 / 13,750 | 110,527（21,760） | 556（ADD 556，NONE 12） | 3/5 / 3/5 |
| compact+entities（第 100 行提示词，最终；`_r2`，run_id `20260911T090326Z`） | `results/real/elite_compact_entities_smoke5_r2` | 18 | 18 | 2,793 / 4,312 | 51,395 | 0 | 0 | 8,675 / 11,949 | 111,082（27,392） | 606（ADD 606，NONE 65，UPDATE 1；672 次 add 结果，395 窗口丢失 0） | 3/5 / 4/5 |

来源：`extract_raw_elite.jsonl`（请求次数、finish_reason、completion_tokens）、`retain_elite.jsonl`（model_ms、added、events、bisected）、`calls.jsonl`（prompt_tokens、cache_hit_tokens、latency_ms、cost_usd、rate_tier）、`questions_elite_rows.jsonl` / `questions_elite_card.jsonl`（答对）。中位数按全部请求（含对半重提的子请求）计；「每批模型耗时」列取 `calls.jsonl` 每次抽取请求的 `latency_ms`（`_r2` 的 `summary.json` 另给 `batch_model_ms` 中位 8,253.8，口径为 retain 行的 model_ms）。`_r2` 的第一次尝试 `results/real/elite_compact_entities_smoke5`（run_id `20260911T083650Z`）在 12/18 批处中断、无 `summary.json`，目录内 `INCOMPLETE.md` 记录了这一状态，本文不再引用其数字。

费用：`full` 前 18 批的 34 次请求全部落在非高峰档，账单 $0.1386；compact 的 18 次请求全部落在高峰档，账单 $0.0774；compact+entities（`_r2`）的 18 次抽取请求全部落在高峰档，抽取账单 $0.086945，加 10 次答题请求后运行合计 $0.089961（28 次计费请求）。按同一档位重算（deepseek-flash 计价 USD/1M：非高峰 缓存命中 0.003 / 未命中 0.15 / 输出 0.6；高峰 0.006 / 0.3 / 1.2）：非高峰 `full` $0.1386 vs compact $0.0387 vs compact+entities $0.0435；高峰 `full` $0.2772 vs compact $0.0774 vs compact+entities $0.0869。输入侧缓存未命中：`full` 143,541、compact 88,767、compact+entities 83,690。

### 2.2 逐题（gold / 预测；来源各目录 `questions_elite_*.jsonl`）

| id | 题型 | gold | full rows | full card | compact rows | compact card | compact+entities rows | compact+entities card | hit@20（六臂） |
|---|---|---|---|---|---|---|---|---|---|
| 6 | TaskMaster | D | A（错） | A（错） | D | D | D | D | 均命中 |
| 18 | RelationMap | A | D（错） | D（错） | D（错） | D（错） | D（错） | D（错） | 均命中 |
| 20 | EventRecall | D | D | D | D | B（错） | D | D | 均命中 |
| 29 | EventRecall | C | C | C | B（错） | C | C | C | 均命中 |
| 32 | TaskMaster | D | D | D | D | D | A（错） | D | 均命中 |

`full` 与 compact 各错两题但错的题不同；compact+entities rows 错两题（18、32）、card 错一题（18）。5 题不足以区分各臂的准确率。

### 2.3 三臂在前 12 批（b0000–b0011，278 窗口）上的对照

| 臂 | 请求次数 | 输出 token 中位 / 最大 | 输出 token 合计 | 触顶次数 | 每批模型耗时中位 / 最大（ms） | add 事件 |
|---|---|---|---|---|---|---|
| full | 14 | 5,631 / 8,192 | 74,371 | 1 | 15,948 / 41,010 | ADD 326，UPDATE 4 |
| compact | 12 | 2,002.5 / 3,111 | 25,368 | 0 | 6,479 / 8,926 | ADD 347，NONE 2 |
| compact+entities（`_r2`） | 12 | 2,364 / 4,224 | 32,659 | 0 | 7,643 / 11,949 | ADD 363，NONE 35，UPDATE 1 |

compact+entities（`_r2`）的 18 条回复共 672 条观察，其中 555 条带非空 `entities`（819 个字符串）；668 条回复的键集合为 `{id, event_idx, text, type, speaker, entities}`，4 条另带 `when`。相比第 99 行提示词，18 批上的输出 token 中位从 2,002.5 升到 2,793（+39.5%），合计从 42,233 升到 51,395（+21.7%），仍无触顶；相对 `full` 的 5,631 / 194,811 仍为 −50.4% / −73.6%。NONE 事件从 12 升到 65：这是修订决策把新观察判为与已存原子重复的次数，随 `entities` 提示而变；其对 204 题检索的影响未在真实 API 上测（零 API 回放见 §3）。

第一次尝试 `results/real/elite_compact_entities_smoke5`（run_id `20260911T083650Z`，开始 2026-09-11T08:36:50Z）在 12/18 批处中断：`manifest.json` 无 `finished_utc`，无 `summary.json`，两个 questions 文件为空；目录内的 `INCOMPLETE.md` 记录了这一状态。它不是一次测量，已由 `_r2` 取代，本文不再引用其数字。

## 3. 零 API 回放 A/B（`replay_ab.py`，204 题，无 API 请求）

方法：两臂消费同一批记录的 deepseek-flash 回复（`results/real/elite_full204_r3/extract_raw_elite.jsonl`，478 行，290/290 批有回复，384 个叶子可用，0 缺失），经拼接进 HMS 的 ReplayLLM 回放。`full` 臂按原样喂入；compact 臂把每条观察削减为紧凑键后以 `extraction_output = compact` 喂入，其余字段由 `derive.py` 本地推导。`entities` 提示 = 记录中该观察的 `entity_mentions`（非空时；`--no-entities-from-mentions` 关闭）。时间线与 hit@20 / MRR 口径同 EGOLIFE_REAL_LLM_ELITE_2026-09-11 §0；`--no-answer`，API 费用 0。

### 3.1 三臂检索（来源 `results/replay_ab_fix/summary.json`，`results/replay_ab_entities/summary.json`；两目录的 `full` 臂逐项相同）

| 臂 | 结果目录 | 命中 / 204 | hit@20 | MRR | MRR（行序） | 存活原子 | 有实体轴的原子 | 每原子 entity_ids |
|---|---|---|---|---|---|---|---|---|
| full | `replay_ab_fix` = `replay_ab_entities` | 161 | 78.9% | 0.4973 | 0.4884 | 8,439 | 8,421 | 1.3619 |
| compact，无 `entities` 提示 | `replay_ab_fix` | 160 | 78.4% | 0.4966 | 0.4894 | 8,440 | 未记录（该版脚本未输出） | 未记录 |
| compact + `entities` 提示 | `replay_ab_entities` | 161 | 78.9% | 0.5000 | 0.4924 | 8,440 | 8,428 | 1.3650 |

按题型 hit@20（命中 / n）：

| 题型 | n | full | compact，无提示 | compact + 提示 |
|---|---|---|---|---|
| EntityLog | 33 | 23（69.7%） | 24（72.7%） | 24（72.7%） |
| EventRecall | 62 | 51（82.3%） | 50（80.6%） | 50（80.6%） |
| HabitInsight | 23 | 23（100%） | 23（100%） | 23（100%） |
| RelationMap | 39 | 28（71.8%） | 26（66.7%） | 27（69.2%） |
| TaskMaster | 47 | 36（76.6%） | 37（78.7%） | 37（78.7%） |

（`summary.md` 把 EventRecall 的 50/62 打印为 80.7%：脚本先把比例四舍五入到 4 位小数 0.8065 再以一位小数格式化；精确值 80.645%。）

配对表（同一题两臂的命中情况）：

| 对照 | 两臂都中 | 仅 full 中 | 仅 compact 中 | 都不中 | 返回批次集合完全相同的题数 |
|---|---|---|---|---|---|
| full vs compact，无提示 | 158 | 3 | 2 | 41 | 120 |
| full vs compact + 提示 | 159 | 2 | 2 | 41 | 129 |

### 3.2 实体轴覆盖（`replay_ab_entities/entity_axis_full.jsonl` / `entity_axis_compact.jsonl`，存活原子按 (batch_id, text) 配对）

| 量 | 值 |
|---|---|
| 配对原子 | 8,439（未配对：full 0，compact 1） |
| `entity_ids` 完全相同 | 8,415（99.72%） |
| compact ⊇ full | 8,439（100%），其中严格超集 24 |
| compact 收窄（⊉ full） | 0 |
| 配对原子中实体轴非空 | full 8,421，compact 8,427 |

24 个严格超集对新增的 26 个实体 id：`event`、`dog`、`office work`、`here`、`puppy` 各 2 次，`neighbor`、`wifi`、`small dog`、`love`、`goodbye mr`、`loser`、`voice`、`western`、`english`、`fly`、`delivery person`、`panda`、`arctic ocean`、`product`、`pending`、`alice` 各 1 次。全部来自 `entities` 提示中实体抽取器抓不到的小写普通名词或名字变体——即第 100 行修的那个缺口。

注意：回放中的 `entities` 提示是 `full` 提示词诱导出的 `entity_mentions`（不设上限），不是紧凑提示词（0–6 个）会诱导出的列表，因此 §3 衡量的是推导规则，不是紧凑提示词本身。

## 4. 本地推导 vs LLM 输出（compact 臂，全部 8,470 条回放观察；来源 `results/replay_ab_entities/summary.json` 的 `derived_vs_llm`）

| 字段 | 一致率 | 平均 Jaccard | 两侧皆空 | LLM 空 | 推导空 |
|---|---|---|---|---|---|
| sentiment | 71.8% | – | – | – | – |
| relation | 99.1% | – | – | – | – |
| predicate | 82.7% | – | – | – | – |
| topic_tags | 2.6% | 0.0149 | 8 | 8 | 8,064 |
| entity_mentions | 92.2% | 0.8725 | 12 | 676 | 12 |

无 `entities` 提示时（`results/replay_ab_fix/summary.md`）entity_mentions 一致率 89.2%，平均 Jaccard 0.8142，推导空 30，两侧皆空 18；其余四行相同。`speaker`：记录回复无 `speaker` 字段，8,280 条由具名 `subject` 提升而来（`speaker_rule`），显式 0。

不一致的主要形态（LLM → 推导）：sentiment `positive → neutral` 1,280、`negative → neutral` 784、`neutral → positive` 237；relation `single_fact → habit` 57、`single_fact → supersession` 7、`supersession → single_fact` 7、`single_fact → constraint` 6；predicate `states → says` 576、`wants → likes` 31、`says → states` 31、`plans → says` 24、`requests → asks` 24、`prefers → likes` 23、`dislikes → likes` 19。比较对象是原始记录字符串（小写；列表经 `_str_list`），枚举外的 LLM 值算不一致。

## 5. compact 丢了什么，为何在这里没有表现

丢失项（相对 `full` 契约由 LLM 直接给出的字段）：

| 项 | 表现 | 该字段在库里的去向 |
|---|---|---|
| 普通事实的 `objects` | 回退 fact 的宾语 = 陈述实体减主语，无实体则为空 | 修订决策与图边（第 99 行用 `statement_triple` 修复后，六个聊天夹具的原子三元组、`history`、`graph_edges` 与 `full` 逐行相同，`tests/test_extraction_compact.py`） |
| LLM 的 `topic_tags` | 推导为空 8,064/8,470，一致率 2.6% | 主题轴（`atom_topic`，mvmi 过滤器）；关键词信号的词表（`hms_elite/utils/scoring.py::_atom_terms` 把 topic_tags 并入词元） |
| LLM 的 `sentiment` | 一致率 71.8%，主要是 LLM 的正 / 负被词表判为中性 | 情感轴；不在检索评分的五个信号内 |
| LLM 的 `predicate` | 一致率 82.7%，差异多为同义（`states/says`、`wants/likes`） | 修订表匹配、图边标签 |
| 记录回复中的 `valid_from` | 削减时丢弃，时间锚由见证消息给出 | 有效期轴 |

为何未影响 EgoLife 检索：`HMS.search` 的评分只有五个信号（`w_semantic 1.0`、`w_keyword 0.5`、`w_entity 0.5`、`w_recency 0.25`、`w_support 0.25`，`hms_elite/utils/scoring.py`），语义与关键词信号主要来自 `text`，而 `text` 在两臂完全相同；实体信号在 `entities` 提示下不比 `full` 窄（§3.2）；topic_tags 对关键词词表的贡献在 204 题上没有把命中数改变（161 = 161，129 题返回批次集合相同，各有 2 题互有胜负）。答题阶段只看 `text`（rows）或状态卡片，不看 objects / sentiment / topic_tags。这些结论限于 EgoLife 的题型与本次检索配置；依赖主题过滤或情感轴的查询未测。

## 6. 决定与规则

决定：`ingest.extraction_output` 默认值于 2026-09-11 上午切换为 `compact`（CONTRACT 变更记录第 102 行），并于同日晚回退为 `full`（`hms_elite/configs/base.py`；CONTRACT §5、§6.5，变更记录第 105 行）。`compact` 仍可用，提示词逐字节不变。

`compact` 保持默认须同时满足的三个条件；前两个满足，第三个不满足：

| 条件 | 判据 | 状态 |
|---|---|---|
| (a) 检索不退化 | 零 API 回放 A/B 中 compact 在 204 题上的 hit@20 不低于 `full` | 满足：161 = 161，MRR 0.5000 vs 0.4973，实体轴 8,439/8,439 配对原子 compact ⊇ full（24 个严格超集、0 个收窄；`results/replay_ab_entities`）；无提示版本为 160 < 161（`results/replay_ab_fix`） |
| (b) 输出 token 降幅经最终提示词确认 | 带 `entities` 提示的提示词在真实 API 上完成同 18 批冒烟（有 `summary.json`），输出 token 中位仍明显低于 `full` | 满足：`results/real/elite_compact_entities_smoke5_r2` 完成 18/18 批、395 窗口、丢失 0；每次请求输出 token 中位 2,793 vs 5,631（−50%）、触顶 0/18 vs 8/34 次、每次请求模型耗时中位 8,675 ms vs 17,147 ms（−49%）、prompt token 中位 6,643.5、672 次 add 结果（存入 606）、5 题答对 rows 3/5 card 4/5、账单 $0.089961（§2.1） |
| (c) 真实 204 题准确率不退化（2026-09-11 晚加入） | 同 204 题真实运行中 compact 在记忆行上下文上的准确率与 `full` 相差不超过 2 个点 | 不满足：compact 初版（`results/real/elite_compact_full204`）53.9%（110/204），根因是旧 DECISIONS 措辞下按下标配对的 NONE 裁决（1,685 个 NONE 对 `full` 的 23 个，22% 陈述未入库；`results/replay_diag`）；措辞重写加 `verdict_min_similarity`=0.5 守卫后的修订版（`results/real/elite_compact_v2_full204`）56.4%（115/204）/ 卡片 62.7%（128/204），hit@20 77.9% / 79.4%，裁决 ADD 7,604 / NONE 15 / UPDATE 9，窗口丢失 0，空闲价 ¥4.35，每批模型耗时 7.3 s；`full` r3 61.3%（125/204）/ 61.8%，hit@20 78.9%，¥11.56，14.9 s；mem0 56.9%。56.4 比 61.3 低 4.9 个点（`EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md`） |

回退后的默认行为：默认配置下每条观察的 `confidence` 由 LLM 给出（规则 LLM 的 R1 偏好为 0.8、R2 约束为 0.9），`topic_tags` / `sentiment` / `predicate` / `objects` 亦由 LLM 给出；README §3 的两处输出字面与 `hms.feedback` 的 `set_confidence` 示例重新打印 `full` 的值（0.8、`[0.9, 1.0]`）。显式设置 `extraction_output: compact` 时 `confidence` 为推导值 0.7，其余按 §1.2 推导。CONTRACT §19 的黄金场景与 `tests/test_ingest.py`、`tests/test_extraction_single.py` 的夹具仍显式钉住 `extraction_output='full'`（无害，它们陈述的是完整契约）；`tests/test_extraction_compact.py` 断言默认为 `full`、默认配置的 `HMS` 调用 `memory_extraction`，且钉在 `compact` 的配置到达不传参构造的 `Ingestor`。建议：成本敏感的批量灌入（写入费用 0.38 倍、每批耗时 0.49 倍）与状态卡片阅读器用 `compact`；记忆行阅读器、或需要 LLM 直接给出 `confidence` / `topic_tags` / `sentiment` 的调用方用 `full`。（2026-09-18 起默认 lines，见变更记录第 108 行：默认配置下 `confidence` 为 0.7 的推导值）

## 7. 剩余注意事项

- 冒烟只有 5 题、18 批；准确率差异（两臂各错两题、错题不同）无统计意义。
- 相同 18 批的对照中 `full` 为 r3 的开头（非高峰档，冷启动 4.8 s 计入首批），compact 为独立运行（高峰档）；模型耗时受档位与时段影响，费用已按同一档位重算。
- compact+entities 提示词的输出 token 比第 99 行提示词高 39.5%（中位 2,793 vs 2,002.5，18 批），仍比 `full` 低 50%；`entities` 提示带来的 NONE 事件增加（65 vs 12）对 204 题真实 API 检索的影响未测（零 API 回放中 hit@20 161 = 161）。
- 回放 A/B 的 `entities` 提示来自 `full` 提示词的输出，不是紧凑提示词的输出（§3.2 末）。
- 全部数字来自 EgoLife A1_JAKE（英文、多说话人、日常对话），检索配置固定为 `top_k=20, threshold=0.1`；其他语料、依赖主题 / 情感轴的查询、`staged` 抽取模式的 compact 行为未在真实 API 上测。
- 回放的 `speaker` 由具名 `subject` 提升而来；真实紧凑提示词要求 LLM 直接给出 `speaker`，两者的差异未量化。

## 8. 复现命令

在仓库根目录 `/Users/lazy/Documents/Claude Code/hms-elite`：

```bash
# 契约测试（17 个用例）与全套
/opt/anaconda3/bin/python3 -m pytest tests/test_extraction_compact.py -p no:warnings
/opt/anaconda3/bin/python3 -m pytest tests/ -p no:warnings
```

在 `benchmarks/egolife/`（`<venv>` 为 benchmarks 的虚拟环境解释器；回放不发 API 请求）：

```bash
# 零 API 回放 A/B，带 entities 提示（= results/replay_ab_entities；除 widened_examples 顺序外 summary.json 逐项相同）
<venv>/bin/python replay_ab.py --no-answer --out results/replay_ab_entities
# 无 entities 提示的对照（与 results/replay_ab_fix 同一推导规则：entity_mentions = extractor(text) ∪ speaker）
<venv>/bin/python replay_ab.py --no-answer --no-entities-from-mentions --out results/replay_ab_fix
```

真实 API 冒烟（发 deepseek 请求，计费；`.env` 里的 `DEEPSEEK_API_KEY` 由脚本读取，不要打印）：

```bash
# 5 题 / 18 批紧凑冒烟（= results/real/elite_compact_entities_smoke5_r2 的调用方式；输出目录由 --out 与 run_id 决定；
# --extraction-output 的默认值即库默认（2026-09-11 晚回退后为 full，run_real.py 从 IngestConfig 读取），紧凑臂须显式写出）（2026-09-18 起默认 lines，见变更记录第 108 行）
<venv>/bin/python run_real.py --limit 5 --engines elite --extraction-output compact --out results/real
# 完整契约的 204 题运行是 results/real/elite_full204_r3；其前 18 批即本文的 full 臂，不必重跑；重跑 full 臂可省略 --extraction-output（默认）或显式 --extraction-output full
# 紧凑修订版的 204 题运行是 results/real/elite_compact_v2_full204（§6 条件 (c) 的依据；同样显式 --extraction-output compact）
```

§2 的表由各目录的 `extract_raw_elite.jsonl` / `retain_elite.jsonl` / `calls.jsonl` / `questions_elite_*.jsonl` 按 `batch_id <= b0017`（§2.3 为 `<= b0011`）筛选后重算；§3、§4 的表直接取 `summary.json`。

## 9. 来源文件

- 代码：`hms_elite/configs/prompts.py`（`COMPACT_OBSERVATION_KEYS`、两个紧凑提示词）、`hms_elite/ingest/derive.py`（`expand_compact_item`、`union_mentions`）、`hms_elite/ingest/extraction.py`（`_compact_observation_from_item`、`_entities_hint`）、`hms_elite/llms/rule.py`（`statement_triple`、`observation_topics`、`mention_texts`、`_compact_observations`）、`hms_elite/utils/scoring.py`（五个信号）。
- 契约：`docs/design/CONTRACT.md` / `CONTRACT.zh.md` §5、§6.5、§11.8、§12.3，变更记录第 99–102 行；`DESIGN.md` / `DESIGN.zh.md` §9；`README.md` / `README.zh.md` §3、§4.4、§8。
- 测试：`tests/test_extraction_compact.py`（12 个测试、17 个用例）。
- 结果：`benchmarks/egolife/results/real/elite_full204_r3`、`results/real/elite_compact_smoke5`、`results/real/elite_compact_entities_smoke5_r2`（最终提示词的完成冒烟）、`results/real/elite_compact_entities_smoke5`（首次尝试，未完成，见 `INCOMPLETE.md`）、`results/replay_ab_fix`、`results/replay_ab_entities`。
