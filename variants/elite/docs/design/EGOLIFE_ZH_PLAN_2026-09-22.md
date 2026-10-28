# EgoLife-ZH：中文数据上的准确率损失与中文支持的挽回（2026-09-22）

目标：量化 HMS Elite 在中文数据上比英文基线少多少准确率，以及一套中文感知的文本流水线能挽回多少。基线已测：EgoLife A1，lines 写入 + hybrid 读取，70.1%（143/204），bge-small-en-v1.5（`docs/design/READER_AB_2026-09-18.md`）。

本文档记录协议、数据事实、库开关、成本预估与两次全量运行的命令。冒烟数字（W5）在最后一节填写。

## 1. 两次运行

| 运行 | 数据 | 嵌入器 | `text.cjk_tokens` | `ingest.zh_rules` | 度量对象 |
|---|---|---|---|---|---|
| 基线（已测） | `data/`（英文） | `BAAI/bge-small-en-v1.5`@384 | 库默认（bigram） | 库默认（False） | 英文准确率 |
| A | `data_zh/`（中文） | `BAAI/bge-small-zh-v1.5`@512 | `none` | False | 英文中心的关键词 / 实体通道在中文上丢多少 |
| B | `data_zh/`（中文） | `BAAI/bge-small-zh-v1.5`@512 | `bigram` | True | 中文支持挽回多少 |

两次运行共用：lines 契约、`elite_rows` + `elite_hybrid` 两个阅读标签（`run_real.py` 默认）、deepseek-flash、同一批次与时间线（290 批 / 4,851 窗口 / 204 题）、并发 1。`--cjk off` 对应 A，`--cjk on` 对应 B；不给 `--cjk` 即库默认，英文运行的 manifest 只多 `data_root` / `text_lang` / `cjk` / `library_switches` / `entity_names` 几个键。

嵌入器事实（bench venv，fastembed 0.8.0）：`BAAI/bge-small-zh-v1.5` 在注册表内，`TextEmbedding.get_embedding_size("BAAI/bge-small-zh-v1.5") == 512`，HF 源 `Qdrant/bge-small-zh-v1.5`，模型文件 90 MB，输出单位向量。`run_real.py --embed-model` 给出而 `--embed-dims` 未给时按该注册表解析，manifest 记 `embedding.dims_source`。库不校验 `embedding_dims` 与模型是否一致：sqlite 索引在不一致时只是走慢路径，milvus 与 mem0 直接失败，因此两处都必须写 512。

## 2. 数据事实（2026-09-22 在代码与数据上核对）

- F1 库本来就在关键词通道两侧发出 CJK 字符二元组（`hms_elite/utils/text.py _plain_tokens`，`tests/test_foundation.py` 钉住 `"我喜欢" → ["我喜", "喜欢"]`）。所以"当前行为" = bigram；新增的是 `none`（每个 CJK 连续串一个词元，unicode61 一类分词器的行为），它才是英文中心的基线，运行 A 用它。
- F2 QA 文件已带官方中文：204/204 条 `question_chinese`，204/204 条四个 `choice_*_chinese` 非空。四个选项全为拉丁字母的条目 24 条（都是人名），至少一个选项为拉丁字母的条目 56 条。**与任务书的偏离**：任务书要求题目与选项也由 deepseek-flash 翻译；本协议默认 `--questions-source native`，取官方中文字段再套术语表（8 个术语表人名在这些字段里都以拉丁形式出现，已核对）。代价是转写词汇（机器翻译）与题目词汇（人工翻译）来自不同译者，是关键词通道上一个未受控的词汇错配变量；理由是官方中文题目是人类基准的原文、且不花钱。`translate_report.json` 的 `questions_source` / `questions_source_note` 记录该选择；要消除该变量，用 `--questions-source translate`（204 条 ≈ 9 个请求，约 ¥0.05），代码路径已存在。
- F3 SRT 已带原生中文 ASR 行：56,640 条 cue 中 615 条只有英文、2,092 条只有中文；4,849/4,851 个窗口有中文；英文 1,742,449 字符、中文 769,214 字符（比 0.44）。用户决定转写走翻译（`--transcript-source translate`）；`native` 模式只翻 615 条纯英文 cue。原生数据下有 6 个窗口的全部 cue 都是纯英文（`A1_JAKE_DAY2_15000000#0010`、`DAY2_17000000#0000/#0001`、`DAY5_15000000#0000/#0001`、`DAY7_19000000#0050`），`build_windows(text_lang="zh")` 对这类窗口抛 ValueError；翻译后它们有中文行。
- F4 加载器按"含 CJK 字符即中文行"分类（`loader.py`），窗口边界按英文字符数计（`windows.py`）。因此 data_zh 的 SRT 保留英文行原文、中文槽位放译文；`build_windows(..., text_lang="zh")` 交换两个槽位：`Window.text` = 带英文 cue 的中文行，`text_zh` = 英文行；边界、id、时间戳、cue 区间与英文运行完全相同。2,092 条只有中文的 cue 两边都不计入，两次运行检索的内容集合相同。
- F5 `estimate_tokens = (utf8_len + 3) // 4 + 8`：CJK 字 3 字节 → 每字 0.75 个估算 token，高于中文分词器约 0.67 的实际值。它高估而不是低估，编译器预算在中文上偏保守；不改。
- F6 术语表人名长度 2–4（`卡特里娜`、`乔伊斯特` 都是 4 字）；`_CJK_PROPER_RE` 的 `{2,4}` 不变，新规则也用 2–4。
- F7 人民币单价：`reader_ab.PRICES_CNY_PER_M`（空闲 未命中 ¥1/M、输出 ¥4/M；高峰 ×2）。`translate_zh.py` 保留同一张表（测试钉住相等），`--cost-cap-cny` 按它核算。

## 3. data_zh 构建协议（`benchmarks/egolife/translate_zh.py`）

- 请求：每请求 25 个窗口，绝不跨 SRT 文件，共 233 个请求（74 个文件）。每窗口发送带英文的 cue 的英文行（含 `Speaker:` 标签）。`temperature 0`、`max_tokens 8192`、`response_format json_object`，经 `real_arms.DeepSeekClient`（台账、档位、费用与基准脚本一致；标签 `arm=translate, stage=translate, batch_id=<请求 id>`）。
- 指令（`TRANSLATE_SYSTEM`，中文）：逐行译成简体中文；标签原样保留；人名、品牌、译不出的菜名保留拉丁拼写；九个名字（Jake、Alice、Tasha、Katrina、Lucia、Shure、Nicous、Choiszt、Me）必须原样；JSON 输出，窗口 id、行数、顺序与输入一致。
- 校验（`validate_window`）：为空、行数不同、源有文本译行为空、正文长度比在 `LENGTH_RATIO = (0.12, 1.5)` 之外、拉丁字母 > 50%，任一命中即拒绝。长度比不是任务书写的 [0.3, 3.0]：按原生行逐窗口实测（n = 4,845）zh/en 字符比 min 0.148、p01 0.203、p50 0.279、p95 0.368、max 0.843，0.3 的下限会拒绝 70% 的真实窗口。
- 截断：`finish_reason == "length"` 的请求对半二分直到单窗口；单窗口仍截断记 `truncated`。批次轮被拒的窗口单独重试一次；仍失败的窗口回退到原生中文行并写入报告。
- 缓存：`data_zh/translate_cache.jsonl`，按窗口 id 一行，续跑只请求缺失 / 未通过校验的窗口。末行被截断（写到一半被杀）时告警跳过；其它位置的畸形行报错。单窗口重试（attempt 1）仍失败的窗口为最终状态：续跑不再为它付费，只在报告的 `coverage.windows_failed` 列出；`--retry-failed` 重新打开。被上限截断的行（reason `cap`）从未翻译过，总是重发。
- 术语表（`apply_glossary`）：整词、长者优先、`Name's → 值的`、`Name:` 保留冒号；套在写出的每一条中文行（译文或原生）与 `question` / `choice_*` 上，不碰隐藏字段。其它名字保持拉丁并列入 `latin_names_kept`（Others 1,906、Jack 374、Shure's GF 242、Male Guest 195、Liu Ting 176、Violet 38 …）。**Jack 的决定**：A1 没有叫 Jack 的参与者，374 处 'Jack' 很可能是佩戴者 Jake 的 ASR / 译文变体；任务书只规定九个名字，脚本不擅自合并，data_zh 因而会同时含 374 处拉丁 'Jack' 与中文 杰克（关键词与实体通道视为两个实体，`variant_rates` 只按术语表键统计、看不到它）。要合并，由用户在翻译前给 `--glossary-add Jack=杰克`（可重复；整词、长者优先，写入报告的 `glossary_added`，`latin_names_note` 说明该规则）。Liu Ting（176）与 Violet（38）是真实的其他人，保持拉丁。
- 写出：SRT 块保留序号行、时间行、英文行原文；中文行 = 译文（套术语表，须含 ≥1 个 CJK 字符，否则回退原生行，再否则不写）。QA 文件逐项原样，只替换 `question` / `choice_*`。报告的 `coverage.windows_native_fallback` / `windows_no_chinese` 用 `build_windows(text_lang='zh')` 的同一谓词（cue 须同时带英文与中文；只有纯中文 cue 有原生行的窗口记 `no_chinese`），`questions_translated` / `questions_native_fallback` 记题目来源（后者只在 `--questions-source translate` 下计数）。`--questions-source translate` 时题目单元的拉丁字母 > 50% 规则只看题干行（选项多是指令要求保留拉丁形式的裸人名；官方中文字段模拟下 204 题有 45 题会因选项被误拒）。
- 自检：`load_questions(data_zh)` id、答案、目标时刻与源一致；`build_windows(load_transcripts(data_zh))` 在 id / 时钟 / cue 区间 / 英文文本上与源逐窗口相同；`build_windows(..., text_lang="zh")` 4,851 个窗口都有中文文本。
- 报告 `translate_report.json`：覆盖率（窗口 4,851、题 204）、术语表命中、每人名的变体率（英文提到而译文既无拉丁形式也无术语表形式的窗口占比）、每窗口拉丁字母率（min / mean / p95 与 > 0.5 的列表）、调用 / 重试 / 截断、按类 token、人民币（实际按档位、全部按空闲、全部按高峰）、美元、自检结果。
- 费用上限：`--cost-cap-cny 6`（默认），**跨续跑累计**：缓存旁的 `translate_spent.json`（`SpendLedger`）记录每次调用脚本的实际人民币，每次计费调用后重写（被杀的运行至多丢一次调用的金额）；剩余上限 = 上限 − 既往花费。首次调用前按预估 + 既往花费与当前档位判断；预估超过剩余上限的 80%（`TIGHT_CAP_FRACTION`）时拒绝运行（退出码 2），除非给 `--allow-tight-cap`；运行中每次调用后按累计台账判断，超过即不再发起新请求，二分的两半也不再发出（飞行中的请求完成后落地）。

预估（`--dry-run`，2026-09-22）：233 个请求，输入 ≈ 516k token、输出 ≈ 586k token，空闲 ¥2.86 / 高峰 ¥5.72（$0.43 / $0.86）。两个档位都在 ¥6 之内，但高峰价占上限 95%，输出 token 估少就会半途停在 CostCapExceeded 并留下不完整的数据集，因此高峰时段脚本会拒绝启动（需 `--allow-tight-cap`）；应在空闲时段跑（北京时间工作日 09–12、14–18 之外），此时占 48%。

## 4. 库开关（CONTRACT §9.6，变更记录第 109 行）

- `text.cjk_tokens: "bigram" | "none"`（默认 bigram）。写侧 `atom_fts` / `atom_keyword_tokens` 与读侧查询词元同用一个 ContextVar 作用域（`cjk_tokens_scope`），由 `HMS` 的 35 个公共方法、`Ingestor` / `StateCompiler` / `StateScheduler` 的入口设置。存储上次以另一模式写入时，`HMS.__init__` 一次性重建 `atom_fts`（meta 键 `text_cjk_tokens`，缺失即 bigram；默认存储不多一行 meta）。
- `ingest.zh_rules: bool`（默认 False）。紧凑 / 行式推导识别中文陈述：开头人名作主语（`杰克说…` → 杰克；开头的 今天 / 下午 之类停用词 / 时间词先跳过）、开头 `我` → user、开头动词经 `VERB_LEXICON_ZH` 映射谓词（主语与动词之间的 也 / 一起 / 晚上 之类副词 / 时间词跳过；和 / 跟 / 给 之后取下钻到达的动词：爱丽丝和塔莎去了超市 → went_to，给了杰克一杯咖啡 → gives）、动词之后的子句作宾语；中文情感 / 主题词表；规则实体抽取器加中文人名规则（陈述动词之前或 和 / 跟 / 给 之后的 2–4 字，代词开头、停用词与以动词 / 副词 / 时间词开头或结尾的片段排除，和 / 跟 / 给 之后的 了 跳过，今天 之类前缀剥离），身份变为 `rule@entities-zh-v1`；中文主语经行式说话人规则到达 `Observation.speaker` → `actor_id`。开头的 我们 / 咱们 → project，他 / 她 / 他们 / 大家 → 主语 None 但动词照读（她喜欢吃披萨 → likes / 吃披萨）。中文正面词表不含裸字 爱（爱丽丝 不是喜爱）。已记录的限制：末字是动词 / 副词字的人名（李刚）不成为 ZH 候选，只有 `names` 表能接住；A1 术语表的八个人名不受影响。**读侧同一抽取器**：`plan_query` / `HMS.search` / `context` / `compile` / `coverage` 用门面的 `entity_extractor`（zh_rules + names）取实体线索，因此 B 的中文提问能激活实体通道（`entity_boost` > 0、`recalled_by` 含 entity）；A 与英文运行默认下逐字节不变。
- `RuleEntityExtractorConfig.names`：封闭人名表，任何位置精确匹配、优先级高于全部 CJK 规则。`run_real.py` 只在 `--text-lang zh` **且** `--cjk on` 时默认传入术语表的中文人名（爱丽丝、卡特里娜、乔伊斯特、塔莎、尼克斯、杰克、舒尔、露西亚）；运行 A（`--cjk off`）不带人名表，保持未改动的 `rule@entities-v1` 抽取器（用户定义的 A 是"全部 CJK 开关关闭的英文中心文本流水线"；人名表属于受测的中文支持）。要单独度量人名表的贡献，可用 `--entity-names 杰克,塔莎,…` 跑第三个臂（A + 人名表）；`--entity-names ""` 表示不带。
- 行式提示词加一句：陈述必须用源消息的语言写、绝不翻译。对英文运行无害。已记录的 `results/real/elite_lines_full204` 回复早于这句话；回放按原样提供记录的回复，可比性不受影响。
- 未改：`estimate_tokens`（见 F5）、compile / render / context。

度量口径：`cjk_tokens="none"` 是全局开关。中文陈述在 `none` 下成为一个整串词元，`statement_tokens` / `verdict_similarity` 几乎不重叠，多数 NONE / REVISE 裁决被守卫降为 ADD；重叠重排、覆盖选择、整合去重、规则句子门也随之改变。因此 A − B 混合了关键词通道损失与修订守卫损失。数据来源：`HMS.last_downgraded`（最近一次 add 的 `IngestResult.downgraded`）→ `EliteRealArm._add` 的信封键 `downgraded: {裁决: 数量}` → `retain_with_bisect` 逐子 add / 逐批次记录（`retain_elite.jsonl` 每行的 `downgraded` 与 `subs[*].downgraded`）→ `summarize()` 的 `engines.elite.retain.downgraded`。W5 直接读 summary 的这一项。

## 5. 基准脚本改动

- `windows.build_windows(..., text_lang="en" | "zh")`（默认 en）。
- `real_arms.EliteRealArm(..., embed_model, embed_dims, cjk, fastembed_cache_dir, entity_names)`；`Mem0RealArm(..., embed_model, embed_dims)`；`stats()` 多记 `cjk` / `text_cjk_tokens` / `zh_rules` / `entity_extractor`。`stats()["embedder"]`（`real_arms.embedder_identity`）：默认嵌入器仍为 `fastembed:BAAI/bge-small-en-v1.5`（与已记录的 `results/real/mem0_full204/summary.json` 逐字节相同），其它嵌入器为 `fastembed:<model>@<dims>`（本实验：`fastembed:BAAI/bge-small-zh-v1.5@512`）。
- `EliteRealArm.recall` 以 `explain=True` 检索（顺序与分数不变），每行多带 `keyword`（bm25_score）/ `entity`（`score_details['entity_boost']`，它是 `scoring.score_details` 唯一的实体键；第一版读了不存在的 `entity_score`，`entity_hits` 恒为 0，已修）/ `recalled_by` / `channel_sizes`；`run_real.py` 把它们汇成 `questions_elite_rows.jsonl` 每题的 `channels = {keyword_hits, entity_hits, channel_sizes, recalled_by}`（`keyword_hits` = 返回行里关键词分量 > 0 的行数）。`elite_hybrid` 走 `HMS.context()`，不带该信号；"关键词命中非零"以 `elite_rows` 为准。
- `retain_with_bisect` 每条记录带 `downgraded`（守卫降级的裁决计数）；`summarize()` 按运行求和到 `engines.elite.retain.downgraded`。
- `run_real.py --text-lang {en,zh} --embed-model --embed-dims --cjk {off,on} --fastembed-cache-dir --entity-names`；manifest 记 `embedding.{model,dims,dims_source,cache_dir,fastembed_cache_path_env}`、`data_root`、`text_lang`、`cjk`、`library_switches`、`entity_names`，summary 顶层重复 `embedding` / `text_lang` / `cjk`。`windows_sha256` 在 zh 运行下与英文运行不同（文本不同，设计如此）。`--out` 是结果目录本身（manifest.json、windows.jsonl、retain_elite.jsonl、questions_*.jsonl、summary.json 直接写在里面；`--run-id` 只是 manifest 字段），所以每次运行必须给自己的目录，与 `results/real/elite_lines_full204` 的惯例一致。
- fastembed 缓存：默认目录在 `tempfile.gettempdir()` 下，macOS 会清理（英文模型的缓存已坏）。`run_real.py` 默认 `--fastembed-cache-dir ~/.cache/hms-elite-bench/fastembed_cache`，并无条件把它写入 `FASTEMBED_CACHE_PATH`（覆盖 shell 里已有的值；manifest 记录 `embedding.fastembed_cache_path_env`），mem0 的提供者只吃这个环境变量。明确说明：默认的英文调用其进程环境因此与已记录的英文运行（用 fastembed 自带的临时目录缓存）不同；输出不受影响（同一份 ONNX 权重，en 模型已在该目录下），这不是逐字节一致性问题。`manifest.data_root` 记为绝对路径（`resolve()`；默认值本就是绝对路径，英文 manifest 不变），从别处读结果目录时不再有 `data_zh` 这种相对值。冒烟前把探针下载的 zh 快照 `cp -RL` 进该目录可免二次下载。
- `compare_real.py` / `report_real.py` 从 manifest 读嵌入器身份、文本语言与 CJK 开关；两次运行控制变量不同时逐运行列出（`unique_run_roots`：每个结果目录一句，不按臂重复），不再声称"同一嵌入器"。

## 6. W5 步骤（bench venv；只有这里调用 API）

1. `venv translate_zh.py --dry-run`，看预估；空闲时段 `venv translate_zh.py --workers 3 --cost-cap-cny 6`（高峰时段会因 80% 规则拒绝启动；续跑时上限按 `translate_spent.json` 累计）；核对 `translate_report.json`：窗口 4851/4851、题 204/204、自检三项为真、变体率、拉丁字母率 > 0.5 的窗口列表、`coverage.windows_failed` / `windows_capped` 为空、`spent_cny.total ≤ 6`。
2. 预热 zh 模型到持久缓存（一次）。
3. 冒烟（各 5 题，实际抽取，约 ¥0.2 一次）：

```
cd /Users/lazy/Documents/Claude\ Code/hms-elite/benchmarks/egolife
V=/Users/lazy/.cache/hms-elite-bench/venv-ort1290/bin/python
$V run_real.py --engines elite --limit 5 --data-root data_zh --text-lang zh --embed-model BAAI/bge-small-zh-v1.5 --embed-dims 512 --cjk off --out results/real_zh/zhA_smoke5 --run-id zhA_smoke5
$V run_real.py --engines elite --limit 5 --data-root data_zh --text-lang zh --embed-model BAAI/bge-small-zh-v1.5 --embed-dims 512 --cjk on  --out results/real_zh/zhB_smoke5 --run-id zhB_smoke5
```

每次运行一个目录（`--out` 就是结果目录；同一 `--out` 跑第二次会覆盖第一次的所有文件）。

检查项：`retain_elite.jsonl` 的陈述含 CJK；`summary.json` 的 `engines.elite.stats.embedder == "fastembed:BAAI/bge-small-zh-v1.5@512"`、`text_cjk_tokens` / `zh_rules` 与运行一致、`entity_extractor` 在 A 为 `rule@entities-v1`、在 B 为 `rule@entities-zh-v1`；标签 `elite_rows` + `elite_hybrid`；B 的 `questions_elite_rows.jsonl` 每题的 `channels.keyword_hits > 0`（A 预期多为 0）、`channels.channel_sizes` 的 keyword 分量 > 0，`retain_elite.jsonl` 的行里 `entity_ids` 含 杰克 / 塔莎 等；B 的 `questions_elite_rows.jsonl` 至少一题 `channels.entity_hits > 0` 且 `channels.recalled_by` 含 entity（读侧实体轴；A 预期为 0）；两次运行 `summary.json` 的 `engines.elite.retain.downgraded`；hit@20 与 correct/5。

4. 全量（204 题）：

```
$V run_real.py --engines elite --data-root data_zh --text-lang zh --embed-model BAAI/bge-small-zh-v1.5 --embed-dims 512 --cjk off --out results/real_zh/zhA_full204 --run-id zhA_full204
$V run_real.py --engines elite --data-root data_zh --text-lang zh --embed-model BAAI/bge-small-zh-v1.5 --embed-dims 512 --cjk on  --out results/real_zh/zhB_full204 --run-id zhB_full204
```

费用按冒烟的每批费用 × 290 批 + 204 次答题外推，时间按冒烟的每批模型耗时 × 290 外推。

## 7. 冒烟数字（W5 填写）

| 运行 | 窗口 / 批次 | 陈述数 | 含 CJK 的陈述 | `entity_extractor` | `channels.keyword_hits > 0` 的题（elite_rows） | `retain.downgraded` | hit@20 | correct/5 | 费用（¥） | 每批模型耗时（ms） |
|---|---|---|---|---|---|---|---|---|---|---|
| A（cjk off，无人名表） | 待填 | 待填 | 待填 | rule@entities-v1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| B（cjk on，术语表人名） | 待填 | 待填 | 待填 | rule@entities-zh-v1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

全量预估（费用 / 时间）：待填。

## 8. 相关文件

- `hms_elite/utils/text.py`、`hms_elite/utils/scoring.py`、`hms_elite/memory/main.py`、`hms_elite/memory/storage.py`、`hms_elite/configs/base.py`、`hms_elite/entities/rule.py`、`hms_elite/llms/rule.py`、`hms_elite/ingest/derive.py`、`hms_elite/ingest/extraction.py`、`hms_elite/ingest/pipeline.py`、`hms_elite/configs/prompts.py`。
- `benchmarks/egolife/translate_zh.py`、`windows.py`、`real_arms.py`、`run_real.py`、`compare_real.py`、`report_real.py`。
- 测试：`tests/test_text_cjk.py`、`tests/test_zh_rules.py`、`tests/test_egolife_zh.py`；`tests/test_wiring.py`、`tests/test_foundation.py`、`tests/test_extraction_lines.py` 的钉住。
- 文档：CONTRACT §5.1 / §5.2 / §9.2 / §9.6 / §10.5 / §11.1（中英）、README §3。
