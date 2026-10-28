"""HMSConfig and its sub-configs: a Mem0-shaped dict plus HMS knobs, validated once, frozen after.
HMSConfig 及其子配置：Mem0 形状的字典加 HMS 参数，一次校验、随后冻结。
"""

# Purpose (EN): Accept Mem0 config dicts (renamed/removed keys handled) and expose typed HMS settings.
# 目的（中）：接受 Mem0 配置字典（处理改名/移除的键），并暴露带类型的 HMS 设置。

# Input (EN): Config dicts or keyword arguments.
# 输入（中）：配置字典或关键字参数。

# Output (EN): HMSConfig instances; ConfigError with CONFIG_001/002/003 codes on bad input.
# 输出（中）：HMSConfig 实例；输入非法时抛出 CONFIG_001/002/003 的 ConfigError。

# State effect (EN): None (warnings are emitted for deprecated keys).
# 状态影响（中）：无（对弃用键发出警告）。

from __future__ import annotations

import copy
import difflib
import logging
import os
import types
import typing
import warnings
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hms_elite.configs.registry import REGISTRY, supported
from hms_elite.enums import CompileMode, Phase, Visibility
from hms_elite.exceptions import ConfigError, unwrap_hms_error
from hms_elite.models import HMSModel

logger = logging.getLogger(__name__)

RENAMED: dict[str, str] = {
    "custom_fact_extraction_prompt": "custom_instructions",
    "custom_update_memory_prompt": "custom_instructions",
    "index": "vector_store",
    "db_path": "store.path",
}
REMOVED: dict[str, str] = {
    "graph_store": "the support graph lives in the same SQLite file; remove the key",
    "enable_graph": "the graph is always on",
}
_ENVELOPE_KEYS: tuple[tuple[str, str], ...] = (
    ("index", "vector_store"),
    ("llm", "llm"),
    ("embedder", "embedder"),
    ("reranker", "reranker"),
)
_INDEX_SUGGESTION = "HMS Elite ships 'sqlite' and 'memory' index providers ('milvus' via the optional [milvus] extra); use {'provider': 'sqlite'}"


def default_db_path() -> str:
    """Default SQLite path: $HMS_DIR/hms.db or ~/.hms_elite/hms.db (no directory is created).
    默认 SQLite 路径：$HMS_DIR/hms.db 或 ~/.hms_elite/hms.db（不创建目录）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Path string.
        路径字符串。

    State effect / 状态影响：
        None (reads the HMS_DIR environment variable).
        无（读取 HMS_DIR 环境变量）。
    """
    return os.path.join(os.environ.get("HMS_DIR") or os.path.expanduser("~/.hms_elite"), "hms.db")


class ProviderEnvelope(HMSModel):
    """{provider, config} envelope; the provider must be registered for the envelope's kind.
    {provider, config} 信封；provider 必须已在该种类下注册。
    """

    kind: ClassVar[str] = ""
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config", mode="before")
    @classmethod
    def _none_to_dict(cls, value: Any) -> Any:
        """None → {} (Mem0 accepts a missing config).
        None → {}（Mem0 允许缺省 config）。
        """
        return {} if value is None else value

    @model_validator(mode="after")
    def _known_provider(self) -> "ProviderEnvelope":
        """provider ∈ REGISTRY[kind] else ValueError (pre-checked as CONFIG_002 by from_dict).
        provider ∈ REGISTRY[kind]，否则 ValueError（from_dict 预检查为 CONFIG_002）。
        """
        if self.provider not in REGISTRY.get(self.kind, {}):
            raise ValueError(f"unknown {self.kind} provider {self.provider!r}")
        return self


_VISION_KEYS: tuple[str, ...] = ("enable_vision", "vision_details")


class LlmConfig(ProviderEnvelope):
    """LLM envelope (default provider 'rule') plus the vision switch shared by every provider.
    LLM 信封（默认提供者 'rule'）以及所有提供者共享的视觉开关。
    """

    kind: ClassVar[str] = "llm"
    provider: str = "rule"
    # enable_vision=True lets add() describe image parts through llm.describe_image before Txn A; providers without
    # vision (rule, scripted by default) return None and the parts stay in dropped_parts, exactly as with False.
    # enable_vision=True 让 add() 在事务 A 之前通过 llm.describe_image 描述图片部件；无视觉的提供者
    # （rule、默认的 scripted）返回 None，部件仍计入 dropped_parts，与 False 时完全一致。
    enable_vision: bool = False
    # Mem0 `vision_details`: the detail hint sent with every image unless the part carries its own.
    # Mem0 的 `vision_details`：随每张图片发送的细节提示，除非部件自带。
    vision_details: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_vision_keys(cls, data: Any) -> Any:
        """Mem0 places enable_vision / vision_details inside `config`; lift them to the envelope (explicit envelope keys win).
        Mem0 把 enable_vision / vision_details 放在 `config` 里；提升到信封（信封显式键优先）。
        """
        if not isinstance(data, dict):
            return data
        config = data.get("config")
        if not isinstance(config, dict) or not any(key in config for key in _VISION_KEYS):
            return data
        lifted = dict(data)
        inner = dict(config)
        for key in _VISION_KEYS:
            if key in inner:
                value = inner.pop(key)
                if key not in lifted:
                    lifted[key] = value
        lifted["config"] = inner
        return lifted


class EmbedderConfig(ProviderEnvelope):
    """Embedder envelope (default provider 'hash').
    嵌入器信封（默认提供者 'hash'）。
    """

    kind: ClassVar[str] = "embedder"
    provider: str = "hash"


class IndexConfig(ProviderEnvelope):
    """Index envelope; the dict key is 'vector_store' (Mem0 name kept).
    索引信封；字典键为 'vector_store'（保留 Mem0 名称）。
    """

    kind: ClassVar[str] = "index"
    provider: str = "sqlite"


class RerankerEnvelope(ProviderEnvelope):
    """Reranker envelope (Mem0 key kept; default provider 'overlap'); None on HMSConfig means "not configured".
    重排器信封（保留 Mem0 键名；默认提供者 'overlap'）；HMSConfig 上为 None 表示"未配置"。
    """

    kind: ClassVar[str] = "reranker"
    provider: str = "overlap"


class StoreConfig(HMSModel):
    """SQLite store settings.
    SQLite 存储设置。
    """

    path: str = Field(default_factory=default_db_path)
    busy_timeout_ms: int = 5000
    journal_mode: Literal["wal", "delete"] = "wal"
    # File databases only: False leaves the WAL on disk at close() (durable; recovered on the next open) when the
    # runtime offers SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python >= 3.12); ':memory:' has no WAL either way.
    # 仅文件库：False 让 close() 把 WAL 留在磁盘上（仍耐久，下次打开时恢复），前提是运行时提供
    # SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE（Python >= 3.12）；':memory:' 无论如何都没有 WAL。
    checkpoint_on_close: bool = True


class IngestConfig(HMSModel):
    """Ingest pipeline settings.
    摄取流水线设置。
    """

    revision_mode: Literal["inline", "deferred", "off"] = "inline"
    neighbourhood_k: int = 10
    last_events_k: int = 10
    emit_relation_atoms: bool = True
    closure_max_gap_hours: float = 24.0
    # Entity extractor provider name (REGISTRY['entity_extractor']) and its config dict; 'rule' is the offline default.
    # 实体抽取器提供者名（REGISTRY['entity_extractor']）及其配置字典；'rule' 是离线默认。
    entity_extractor: str = "rule"
    entity_extractor_config: dict[str, Any] = Field(default_factory=dict)
    # 'single' = one memory_extraction call per message batch (default); 'staged' = observation → ep → revision
    # (up to three calls). Forwarded verbatim to Ingestor(extraction_mode=...) by HMS.__init__.
    # 'single' = 每批消息一次 memory_extraction 调用（默认）；'staged' = 观察 → EP → 修订（最多三次调用）。
    # 由 HMS.__init__ 原样透传给 Ingestor(extraction_mode=...)。
    extraction_mode: Literal["single", "staged"] = "single"
    # 'full' (the default until 2026-09-18) = the LLM emits every observation field; 'compact' = the LLM emits only id / event_idx / text /
    # type / speaker (+ optional entities / supersedes_hint_ids / when / valid_to) and ingest.derive fills subject,
    # predicate, objects, relation, dates, confidence, attributed_to, sentiment, topic_tags and entity_mentions locally
    # (CONTRACT §6.5; both prompts byte-identical either way). History of the default: switched to 'compact' on
    # 2026-09-11 (morning) under the two-condition rule of CONTRACT §6.5 (zero-API replay hit@20 161 = 161; same-18-batch
    # smoke median 2,793 vs 5,631 output tokens per request), reverted to 'full' on 2026-09-11 (evening) under the
    # accuracy rule added to §6.5 the same day: compact stays the default only if its rows accuracy on the real
    # 204-question EgoLife run is within 2 points of full. Real deepseek-flash runs on the same 204 questions
    # (docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md): full r3 rows 61.3% (125/204) / card 61.8%, hit@20
    # 78.9%; compact v1 rows 53.9% (root cause: index-paired NONE verdicts, 1,685 NONE); compact v2 (rewritten
    # DECISIONS wording + verdict_min_similarity guard, results/real/elite_compact_v2_full204) rows 56.4% (115/204) /
    # card 62.7% (128/204), hit@20 77.9% / 79.4%, NONE 15, windows lost 0 — 56.4 vs 61.3 is 4.9 points below, so
    # the rule failed and the default stayed 'full' until 2026-09-18 (below). 'compact' stays fully supported and is the recommendation for
    # cost-sensitive bulk ingestion (write cost 0.38× and 7.3 s vs 14.9 s per batch, ¥4.35 vs ¥11.56 at off-peak
    # prices) and wherever the state-card context is the reader input (card 62.7% vs 61.8%). Works with both
    # extraction modes. Forwarded verbatim to Ingestor(extraction_output=...) by HMS.__init__.
    # 'full'（2026-09-18 之前的默认）= LLM 输出每个观察字段；'compact' = LLM 只输出 id / event_idx / text / type / speaker（+ 可选的
    # entities / supersedes_hint_ids / when / valid_to），subject、predicate、objects、relation、日期、confidence、
    # attributed_to、sentiment、topic_tags、entity_mentions 由 ingest.derive 在本地补齐（CONTRACT §6.5；两种情况下
    # 提示词都逐字节不变）。默认值的历史：2026-09-11（上午）按 CONTRACT §6.5 的两条件规则切换为 'compact'（零 API 回放
    # hit@20 161 = 161；同 18 批冒烟每次请求输出 token 中位 2,793 vs 5,631），2026-09-11（晚）按同日加入 §6.5 的准确率
    # 规则回退为 'full'：compact 只有在真实 204 题 EgoLife 运行的记忆行准确率与 full 相差不超过 2 个点时才保持默认。
    # 同 204 题的真实 deepseek-flash 运行（docs/design/EGOLIFE_REAL_LLM_MEM0_VS_ELITE_2026-09-11.md）：full r3 记忆行
    # 61.3%（125/204）/ 状态卡片 61.8%，hit@20 78.9%；compact 初版记忆行 53.9%（根因：按下标配对的 NONE 裁决，1,685 个
    # NONE）；compact 修订版（DECISIONS 措辞重写 + verdict_min_similarity 守卫，results/real/elite_compact_v2_full204）
    # 记忆行 56.4%（115/204）/ 卡片 62.7%（128/204），hit@20 77.9% / 79.4%，NONE 15，窗口丢失 0——56.4 比 61.3 低 4.9 个点，
    # 规则不满足，默认直到 2026-09-18 仍为 'full'（见下）。'compact' 仍完整支持，并推荐用于成本敏感的批量灌入（写入费用 0.38 倍、每批 7.3 s 对
    # 14.9 s、空闲价 ¥4.35 对 ¥11.56）与以状态卡片上下文为阅读器输入的场景（卡片 62.7% 对 61.8%）。两种抽取模式都适用。
    # 由 HMS.__init__ 原样透传给 Ingestor(extraction_output=...)。
    # 'lines' (CONTRACT §6.6, 2026-09-12) = the LLM answers memory_extraction_lines with tab-separated OBS / EP / DEC
    # text instead of JSON (no json mode); ingest.lines parses it and the rest is derived exactly as in 'compact'.
    # Why: on the 4,851-window EgoLife run output tokens were 83% of the full contract's write cost (2.40M of ¥11.5;
    # compact v2 0.76M of ¥4.2; mem0 0.22M of ¥1.85, input-miss already at parity) and a JSON envelope costs 2–4×
    # the text it carries, while retrieval uses only the texts and the decisions; the compact prompt's texts scored
    # below the full prompt's (rows 56.4% vs 61.3%), so the lines prompt keeps the full prompt's statement rules and
    # changes only the output format. extraction_mode='single' only (the validator below).
    # 'lines'（CONTRACT §6.6，2026-09-12）= LLM 以 tab 分隔的 OBS / EP / DEC 文本而非 JSON 回答 memory_extraction_lines
    # （不用 json 模式）；由 ingest.lines 解析，其余与 'compact' 完全一样在本地推导。动机：4,851 窗口的 EgoLife 运行中，
    # 输出 token 占完整契约写入成本的 83%（¥11.5 中 2.40M；compact v2 ¥4.2 中 0.76M；mem0 ¥1.85 中 0.22M，输入未命中侧
    # 已持平），JSON 信封的开销是其承载文本的 2–4 倍，而检索只用到文本与决策；紧凑提示词的文本低于完整提示词（记忆行
    # 56.4% 对 61.3%），因此行式提示词保留完整提示词的陈述规则，只改输出格式。仅支持 extraction_mode='single'（下方校验器）。
    # 2026-09-18: default 'full' → 'lines' (CONTRACT changelog row 108). Evidence docs/design/READER_AB_2026-09-18.md
    # §六: the lines write costs ¥2.24 per 204 questions (full ¥11.56), and the hybrid read — the memoized session card
    # as a prefix plus timestamped relevant memory rows, HMS.context() — answers 70.1% (143/204) on EgoLife, sign test
    # against the plain rows context p < 0.0001. extraction_mode stays 'single' by default; 'staged' now needs an
    # explicit extraction_output ('full' | 'compact') because _lines_needs_single is unchanged and rejects
    # lines + staged (CONFIG_001 through from_dict). The 09-11 history above is kept as is.
    # 2026-09-18：默认 'full' → 'lines'（CONTRACT 变更记录第 108 行）。证据见 docs/design/READER_AB_2026-09-18.md §六：
    # 行式写入每 204 题 ¥2.24（full ¥11.56），混合读——记忆化的会话卡片作前缀加带时间戳的相关记忆行，即
    # HMS.context()——在 EgoLife 上答对 70.1%（143/204），与纯记忆行上下文的符号检验 p < 0.0001。extraction_mode 默认
    # 仍为 'single'；'staged' 现在必须显式给出 extraction_output（'full' | 'compact'），因为 _lines_needs_single 未变、
    # 仍拒绝 lines + staged（经 from_dict 为 CONFIG_001）。上方 09-11 的历史原样保留。
    extraction_output: Literal["full", "compact", "lines"] = "lines"
    # Messages longer than this (characters) are cut by segmentation.segment into extraction units before the LLM
    # sees them; every unit keeps the parent message's anchor (one event, one anchor). Shorter messages are untouched.
    # 超过此长度（字符数）的消息在进入 LLM 之前由 segmentation.segment 切成抽取单元；每个单元保留父消息的锚点
    # （一事一锚）。更短的消息原样不动。
    segment_max_chars: int = Field(default=400, ge=1)
    # Lexical bar for honouring a NONE / REVISE verdict (the §12.6 step-6 guard, both contracts, both modes): the
    # containment |A ∩ B| / min(|A|, |B|) of the stemmed content tokens of the new statement and its target after the
    # head is removed — each side's subject, and the predicate when both share it (ingest.extraction.verdict_similarity)
    # — must be at or above this value, else the verdict is downgraded to ADD (DEBUG log; one entry in
    # IngestResult.downgraded). Before the bar: a verdict whose two atoms carry different subjects is always downgraded
    # (NONE / REVISE / CONFLICT all assert a relation between statements about one subject); a REVISE is corroborated
    # and passes regardless when the observation itself names the target in supersedes_hint_ids or the two atoms share
    # subject and predicate (the exclusive-attribute case); a CONFLICT with agreeing subjects passes (a different value
    # is what it asserts). 0.0 disables the guard (pre-2026-09-11 behaviour). Motivation: under the compact prompt
    # deepseek-flash paired observation i with candidate i and returned NONE for 1,685 of 7,315 observations (23 of
    # 8,470 under the full prompt), each NONE discarding the new statement
    # (benchmarks/egolife/results/replay_diag/summary.md). The first cut counted the subject instead of removing the
    # head, so a short statement sharing only its speaker and a reporting verb scored exactly 0.5 and 112 index-paired
    # NONE survived (replay hit@20 151/204 against the force-ADD ceiling 154); with the head removed the replay reaches
    # 154/204 (results/replay_diag2/summary.md, G1v2) and the full run's guarded replay keeps 161/204.
    # 采纳 NONE / REVISE 裁决的词法门槛（§12.6 第 6 步守卫，两种契约、两种模式都适用）：新陈述与其目标去掉头部——各自的
    # 主语，以及双方共同的谓词——之后的词干化内容词元包含度 |A ∩ B| / min(|A|, |B|)（ingest.extraction.verdict_similarity）
    # 须不低于此值，否则裁决降级为 ADD（DEBUG 日志；IngestResult.downgraded 记一条）。门槛之前：两原子主语不同的裁决一律
    # 降级（NONE / REVISE / CONFLICT 都断言两条关于同一主语的陈述之间的关系）；观察自身在 supersedes_hint_ids 里点名了
    # 目标、或两原子主语与谓词相同（互斥属性情形）的 REVISE 视为已佐证、无条件通过；主语一致的 CONFLICT 通过（它断言的
    # 正是不同取值）。0.0 关闭守卫（2026-09-11 之前的行为）。动机：紧凑提示词下 deepseek-flash 把观察 i 与候选 i 配对，
    # 7,315 条观察中 1,685 条判为 NONE（完整提示词下 8,470 条中 23 条），每个 NONE 都丢掉新陈述
    # （benchmarks/egolife/results/replay_diag/summary.md）。第一版把主语计入而非去掉头部，只与目标共享说话人和一个
    # 转述动词的短陈述恰好得 0.5，112 条按下标配对的 NONE 越线（回放 hit@20 151/204，强制 ADD 的上限 154）；去掉头部
    # 后回放达到 154/204（results/replay_diag2/summary.md，G1v2），完整运行的守卫回放保持 161/204。
    verdict_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    # Chinese statement rules for the compact / lines derivation (llms.rule.statement_triple, predicate_of, the rule
    # entity extractor's proper-name rules, sentiment / topic lists, Observation.speaker from a Chinese subject).
    # False (default) keeps every English and legacy Chinese output byte-identical; True is the EgoLife-ZH run B switch
    # (CONTRACT §9.6). When True and entity_extractor == 'rule', HMS.__init__ passes zh_rules=True to the extractor.
    # 紧凑 / 行式推导的中文陈述规则（llms.rule.statement_triple、predicate_of、规则实体抽取器的人名规则、情感 / 主题
    # 词表、由中文主语得到 Observation.speaker）。False（默认）保持全部英文与既有中文输出逐字节不变；True 是
    # EgoLife-ZH 运行 B 的开关（CONTRACT §9.6）。为 True 且 entity_extractor == 'rule' 时，HMS.__init__ 把
    # zh_rules=True 传给抽取器。
    zh_rules: bool = False

    @field_validator("entity_extractor_config", mode="before")
    @classmethod
    def _none_to_dict(cls, value: Any) -> Any:
        """None → {} (same leniency as the provider envelopes).
        None → {}（与提供者信封同样宽松）。
        """
        return {} if value is None else value

    @model_validator(mode="after")
    def _known_extractor(self) -> "IngestConfig":
        """entity_extractor ∈ REGISTRY['entity_extractor'] else ValueError (pre-checked as CONFIG_002 by from_dict).
        entity_extractor ∈ REGISTRY['entity_extractor']，否则 ValueError（from_dict 预检查为 CONFIG_002）。
        """
        if self.entity_extractor not in REGISTRY.get("entity_extractor", {}):
            raise ValueError(f"unknown entity_extractor provider {self.entity_extractor!r}")
        return self

    @model_validator(mode="after")
    def _lines_needs_single(self) -> IngestConfig:
        """extraction_output='lines' requires extraction_mode='single' (the lines contract has no staged prompts), else ValueError.
        extraction_output='lines' 要求 extraction_mode='single'（行式契约没有分段提示词），否则 ValueError。
        """
        if self.extraction_output == "lines" and self.extraction_mode != "single":
            raise ValueError(
                "ingest.extraction_output='lines' supports extraction_mode='single' only (CONTRACT §6.6) / "
                "ingest.extraction_output='lines' 只支持 extraction_mode='single'（CONTRACT §6.6）"
            )
        return self


class ScoringConfig(HMSModel):
    """Search fusion weights (the eight Mem0-era keys; the candidate bound now lives in RetrievalConfig).
    搜索融合权重（Mem0 时代的八个键；候选上限已移到 RetrievalConfig）。
    """

    w_semantic: float = 1.0
    w_keyword: float = 0.5
    w_entity: float = 0.5
    w_recency: float = 0.25
    w_support: float = 0.25
    half_life_days: float = 30.0
    # Deprecated alias: kept for §5.1 dict compatibility only; the newest-first truncation it once drove is gone
    # (RETRIEVAL_DIAGNOSIS §1) and the pool safety cap is RetrievalConfig.safety_limit. The default is the None
    # sentinel so a model_dump() round-trip stays silent; only a non-None value warns.
    # 已弃用别名：仅为 §5.1 字典兼容保留；它曾驱动的最新优先截断已移除（RETRIEVAL_DIAGNOSIS §1），
    # 池安全上限是 RetrievalConfig.safety_limit。默认值是 None 哨兵，model_dump() 往返不出声；只有非 None 的值会告警。
    max_candidates: int | None = None
    identifier_weight: float = 2.0

    @model_validator(mode="before")
    @classmethod
    def _deprecated_max_candidates(cls, data: Any) -> Any:
        """A non-None max_candidates → DeprecationWarning (the value is accepted and ignored by retrieval).
        非 None 的 max_candidates → DeprecationWarning（值被接受，但检索不再使用它）。
        """
        if isinstance(data, dict) and data.get("max_candidates") is not None:
            warnings.warn(
                "ScoringConfig.max_candidates is deprecated and no longer truncates candidates; "
                "the pool safety cap is RetrievalConfig.safety_limit",
                DeprecationWarning,
                stacklevel=2,
            )
        return data


class RetrievalConfig(HMSModel):
    """Bounded multi-channel recall (RETRIEVAL_DIAGNOSIS §3): per-channel K, pool bound, hard-gate rule, safety cap.
    有界多通道召回（RETRIEVAL_DIAGNOSIS §3）：各通道 K、池上限、硬门规则、安全上限。

    None for a K means the top_k-derived default: k_semantic → max(4·top_k, 60) (Mem0's over-fetch rule),
    k_keyword / k_entity / k_recency → max(2·top_k, 30); pool_max = pool_max_factor · k_semantic.
    K 为 None 表示按 top_k 推导的默认值：k_semantic → max(4·top_k, 60)（Mem0 的过取规则），
    k_keyword / k_entity / k_recency → max(2·top_k, 30)；pool_max = pool_max_factor · k_semantic。
    """

    k_semantic: int | None = Field(default=None, ge=1)
    k_keyword: int | None = Field(default=None, ge=1)
    k_entity: int | None = Field(default=None, ge=1)
    k_recency: int | None = Field(default=None, ge=1)
    pool_max_factor: int = Field(default=4, ge=1)
    # A hard time / entity gate whose gated set fits the pool enters it whole (no ANN recall gap, T6).
    # 门控集合能装进池的硬时间 / 实体门整体入池（无 ANN 召回缺口，T6）。
    hard_gate_pool: bool = True
    # The only remaining cap on a pool: far above any sane pool; crossing it is logged, never silent.
    # 池上唯一保留的上限：远高于任何合理的池；越过时记录日志，绝不静默。
    safety_limit: int = Field(default=50000, ge=1)

    def k_for(self, top_k: int) -> dict[str, int]:
        """Resolve the four channel bounds and the pool bound for one query's top_k.
        为一次查询的 top_k 解析四个通道上限与池上限。

        Input / 输入：
            top_k: The caller's top_k (>= 1 assumed).
            top_k：调用方的 top_k（假定 >= 1）。

        Output / 输出：
            {'semantic', 'keyword', 'entity', 'recency', 'pool'} → ints.
            {'semantic', 'keyword', 'entity', 'recency', 'pool'} → 整数。

        State effect / 状态影响：
            None.
            无。
        """
        k = max(int(top_k), 1)
        semantic = self.k_semantic or max(4 * k, 60)
        small = max(2 * k, 30)
        return {
            "semantic": semantic,
            "keyword": self.k_keyword or small,
            "entity": self.k_entity or small,
            "recency": self.k_recency or small,
            "pool": self.pool_max_factor * semantic,
        }


class TextConfig(HMSModel):
    """Text pipeline switches shared by the write side (atom_fts, keyword vocabulary) and the read side (query tokens).
    写侧（atom_fts、关键词词表）与读侧（查询词元）共用的文本流水线开关。
    """

    # CJK tokenizer mode of utils.text.tokenize / content_tokens on BOTH the index and the query side (CONTRACT §9.6):
    # 'bigram' = overlapping character bigrams (the default and the historical behaviour: English stores are
    # byte-identical); 'none' = one token per CJK run, the English-centric baseline the EgoLife-ZH run A measures
    # (a Chinese keyword query then hits only an identical whole run). A store reopened under another mode gets its
    # atom_fts rebuilt once by HMS.__init__ (StateStore.rebuild_fts; meta key 'text_cjk_tokens', absent == bigram).
    # utils.text.tokenize / content_tokens 在索引侧与查询侧共用的 CJK 分词模式（CONTRACT §9.6）：'bigram' = 重叠字符
    # 二元组（默认，也是历史行为：英文存储逐字节不变）；'none' = 每个 CJK 连续串一个词元，即 EgoLife-ZH 运行 A 度量的
    # 英文中心基线（中文关键词查询只在整串完全相同时命中）。在另一模式下重新打开的存储由 HMS.__init__ 一次性重建
    # atom_fts（StateStore.rebuild_fts；meta 键 'text_cjk_tokens'，缺失即 bigram）。
    cjk_tokens: Literal["bigram", "none"] = "bigram"


class GraphConfig(HMSModel):
    """Support Graph / coarsening settings.
    支持图 / 粗化设置。
    """

    min_cohesion: float = 0.3
    coarsen_min_nodes: int = 2
    coarsen_max_nodes: int = 12
    coarsen_gap_hours: float = 48.0
    temporal_window_hours: float = 24.0
    coarsen_policy: Literal["rule", "llm"] = "rule"
    causal_proposer: Literal["rule", "llm"] = "rule"
    coarsen_l3: bool = True


class CompilerConfig(HMSModel):
    """StateCompiler budget and slot settings.
    状态编译器预算与槽位设置。
    """

    budget_tokens: int = 1200
    candidate_k: int = 50
    graph_hops: int = 1
    require_witness: bool = True
    slot_shares: dict[str, float] = Field(
        default_factory=lambda: {
            "facts": 0.30,
            "episodes": 0.25,
            "constraints": 0.15,
            "preferences": 0.10,
            "skills": 0.05,
            "audit": 0.15,
        }
    )
    min_constraints: int = 3
    max_why_nodes: int = 60
    default_mode: CompileMode = CompileMode.auto
    # 'shares' = the §15.2 step-8 slot-share packer (default, byte-identical cards); 'coverage' = requirement
    # extraction + greedy set cover + hierarchical L3 → L2 → L1 refinement (compiler.coverage / compiler.scc), also
    # selectable per call with compile(mode='coverage').
    # 'shares' = §15.2 第 8 步的槽位份额打包器（默认，卡片逐字节不变）；'coverage' = 需求抽取 + 贪心集合覆盖 +
    # L3 → L2 → L1 分层下钻（compiler.coverage / compiler.scc），也可按次调用 compile(mode='coverage') 选择。
    selection: Literal["shares", "coverage"] = "shares"
    # HMS.context() — the hybrid read context (CONTRACT §6.7, 2026-09-18): a session card compiled from the scope's
    # latest extractable event at context_session_budget_tokens, memoized per scope until any write touches it, then
    # context_rows_header on its own line and the search rows (top context_rows_top_k, deduplicated, time-sorted)
    # as '- [YYYY-MM-DD HH:MM] text' lines when context_timestamps is True ('- text' otherwise). The header default
    # equals hms_elite.compiler.context.ROWS_HEADER, the string the READER_AB harness measured with.
    # HMS.context()——混合读上下文（CONTRACT §6.7，2026-09-18）：以范围内最新可抽取事件、按 context_session_budget_tokens
    # 编译的会话卡片（按范围记忆化，直到任何写入触及该范围），随后单独一行的 context_rows_header，再是检索行
    # （前 context_rows_top_k 条，去重、按时间排序），context_timestamps 为 True 时渲染为 '- [YYYY-MM-DD HH:MM] 文本' 行
    # （否则 '- 文本'）。标题默认值等于 hms_elite.compiler.context.ROWS_HEADER，即 READER_AB 基准测量所用的字符串。
    context_session_budget_tokens: int = Field(default=1200, ge=1)
    context_rows_top_k: int = Field(default=20, ge=1)
    context_rows_header: str = "Recent relevant memories:"
    context_timestamps: bool = True


class SchedulerConfig(HMSModel):
    """StateScheduler settings.
    状态调度器设置。
    """

    phases: list[Phase] = Field(default_factory=lambda: [Phase.orient, Phase.collect, Phase.consolidate, Phase.prune])
    staleness_tau: float = 0.8
    confidence_min: float = 0.4
    use_recency_half_life_days: float = 30.0
    orient_window_days: int = 30
    decay_rate: float = 0.1
    compile_state: bool = True
    # Default interval of HMS.start_daemon() in seconds; None keeps the daemon opt-in (start_daemon() without an
    # explicit interval then raises VALIDATION_009). Must be > 0 when set.
    # HMS.start_daemon() 的默认间隔（秒）；None 表示守护进程仍需显式选择（此时不带间隔调用 start_daemon() 抛出
    # VALIDATION_009）。设置时必须 > 0。
    daemon_interval_s: float | None = Field(default=None, gt=0)


class RetentionConfig(HMSModel):
    """Retention escape hatch (§16.6): physical purge of long-archived atom versions, OFF by default.
    保留策略逃生舱（§16.6）：物理清理长期归档的原子版本，默认关闭。
    """

    # enabled=False keeps HMS.purge(dry_run=False) a ConfigError: revision-instead-of-deletion stays the default path.
    # enabled=False 时 HMS.purge(dry_run=False) 抛 ConfigError：修订代替删除仍是默认路径。
    enabled: bool = False
    older_than_days: int = Field(default=90, ge=0)
    # run_in_prune=True (with enabled) lets the scheduler's prune phase call purge(dry_run=False, keep_witnessed=True).
    # run_in_prune=True（且 enabled）让调度器的 prune 阶段调用 purge(dry_run=False, keep_witnessed=True)。
    run_in_prune: bool = False


def normalize_mem0_keys(data: dict, *, strict: bool) -> dict:
    """Deep-copy a config dict and remap RENAMED Mem0 keys; strict=True also rejects REMOVED keys (CONFIG_003).
    深拷贝配置字典并重映射 RENAMED 的 Mem0 键；strict=True 时还拒绝 REMOVED 键（CONFIG_003）。

    Input / 输入：
        data: Caller dict (never mutated); non-dict input is returned unchanged.
        data：调用方字典（从不修改）；非字典输入原样返回。

        strict: Raise ConfigError(CONFIG_003) for REMOVED keys instead of leaving them for pydantic.
        strict：对 REMOVED 键抛出 ConfigError(CONFIG_003)，而非留给 pydantic 处理。

    Output / 输出：
        New dict; each RENAMED key emits one DeprecationWarning (stacklevel=2); dotted targets set nested keys.
        新字典；每个 RENAMED 键发出一次 DeprecationWarning（stacklevel=2）；点分目标写入嵌套键。

    State effect / 状态影响：
        None.
        无。
    """
    if not isinstance(data, dict):
        return data
    out = copy.deepcopy(data)
    for old, target in RENAMED.items():
        if old not in out:
            continue
        value = out.pop(old)
        warnings.warn(f"config key '{old}' is deprecated; use '{target}'", DeprecationWarning, stacklevel=2)
        if "." in target:
            parent, child = target.split(".", 1)
            sub = out.get(parent)
            if sub is None:
                sub = out[parent] = {}
            if isinstance(sub, dict) and sub.get(child) is None:
                sub[child] = value
        elif out.get(target) is None:
            out[target] = value
    if strict:
        for key, hint in REMOVED.items():
            if key in out:
                raise ConfigError(
                    f"config key '{key}' was removed in HMS Elite: {hint}",
                    "CONFIG_003",
                    details={"key": key, "hint": hint},
                    suggestion=hint,
                )
    return out


def check_providers(data: dict) -> None:
    """Plain-Python pre-check: every envelope's provider must be registered (CONFIG_002 otherwise).
    纯 Python 预检查：每个信封的 provider 必须已注册（否则 CONFIG_002）。

    Input / 输入：
        data: Config dict after normalize_mem0_keys.
        data：normalize_mem0_keys 之后的配置字典。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        None; raises ConfigError(CONFIG_002, details={'kind', 'provider', 'supported'}).
        无；抛出 ConfigError(CONFIG_002, details={'kind', 'provider', 'supported'})。
    """
    for kind, key in _ENVELOPE_KEYS:
        envelope = data.get(key)
        provider = envelope.get("provider") if isinstance(envelope, dict) else None
        if provider is None or provider in REGISTRY[kind]:
            continue
        names = supported(kind)
        suggestion = _INDEX_SUGGESTION if kind == "index" else f"supported {kind} providers: {', '.join(names)}"
        raise ConfigError(
            f"unknown {kind} provider {provider!r} (config key '{key}'); supported: {', '.join(names)}",
            "CONFIG_002",
            details={"kind": kind, "provider": provider, "supported": names},
            suggestion=suggestion,
        )
    # The entity extractor is a plain name under `ingest`, not an envelope; same CONFIG_002 pre-check.
    # 实体抽取器是 `ingest` 下的纯名称而非信封；做同样的 CONFIG_002 预检查。
    ingest = data.get("ingest")
    extractor = ingest.get("entity_extractor") if isinstance(ingest, dict) else None
    if extractor is not None and extractor not in REGISTRY["entity_extractor"]:
        names = supported("entity_extractor")
        raise ConfigError(
            f"unknown entity_extractor provider {extractor!r} (config key 'ingest.entity_extractor'); "
            f"supported: {', '.join(names)}",
            "CONFIG_002",
            details={"kind": "entity_extractor", "provider": extractor, "supported": names},
            suggestion=f"supported entity_extractor providers: {', '.join(names)}",
        )


def _model_from_annotation(annotation: Any) -> type[BaseModel] | None:
    """Pick the BaseModel subclass inside an annotation (unwrapping Optional/Union).
    从注解中挑出 BaseModel 子类（解开 Optional/Union）。
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(types, "UnionType", None):
        for arg in typing.get_args(annotation):
            found = _model_from_annotation(arg)
            if found is not None:
                return found
    return None


def _candidates_at(loc: tuple) -> list[str]:
    """Field names of the model that owns the last element of a pydantic error loc.
    拥有 pydantic 错误 loc 末元素的模型的字段名列表。
    """
    model: type[BaseModel] | None = HMSConfig
    for part in loc[:-1]:
        if model is None or not isinstance(part, str) or part not in model.model_fields:
            return []
        model = _model_from_annotation(model.model_fields[part].annotation)
    return list(model.model_fields) if model is not None else []


def _describe(errors: list[dict]) -> str:
    """Human message for the first pydantic error, naming the key and a difflib hint.
    为首个 pydantic 错误生成可读消息，指出键名并给出 difflib 提示。
    """
    if not errors:
        return "invalid configuration"
    first = errors[0]
    loc = tuple(first.get("loc") or ())
    where = ".".join(str(part) for part in loc)
    if first.get("type") == "extra_forbidden" and loc:
        key = str(loc[-1])
        hint = difflib.get_close_matches(key, _candidates_at(loc), n=1)
        parent = ".".join(str(part) for part in loc[:-1])
        message = f"Unknown config key '{key}'" + (f" under '{parent}'" if parent else "")
        return message + (f". Did you mean '{hint[0]}'?" if hint else ".")
    return f"Invalid config value at '{where}': {first.get('msg')}"


class HMSConfig(HMSModel):
    """Top-level configuration (frozen); Mem0 keys keep their names, HMS knobs live in sub-configs.
    顶层配置（冻结）；Mem0 键名保持不变，HMS 参数位于子配置中。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True, use_enum_values=True)

    vector_store: IndexConfig = Field(default_factory=IndexConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    history_db_path: str | None = None
    # None = no reranker configured: the facade uses the offline OverlapReranker only when search(rerank=True).
    # None = 未配置重排器：门面仅在 search(rerank=True) 时使用离线的 OverlapReranker。
    reranker: RerankerEnvelope | None = None
    version: Literal["v1.1", "v2", "lite-1"] = "lite-1"
    custom_instructions: str | None = None
    store: StoreConfig = Field(default_factory=StoreConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    compiler: CompilerConfig = Field(default_factory=CompilerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    visibility_default: Visibility = Visibility.private

    @model_validator(mode="before")
    @classmethod
    def _mem0_keys(cls, data: Any) -> Any:
        """RENAMED remap only (idempotent); never mutates the caller dict.
        仅重映射 RENAMED 键（幂等）；从不修改调用方字典。
        """
        return normalize_mem0_keys(data, strict=False)

    @model_validator(mode="after")
    def _history_path(self) -> "HMSConfig":
        """history_db_path → store.path (warn when both set and differ).
        history_db_path → store.path（两者都设置且不同时警告）。
        """
        if self.history_db_path is not None:
            if "path" in self.store.model_fields_set and self.store.path != self.history_db_path:
                warnings.warn(
                    f"store.path {self.store.path!r} differs from history_db_path {self.history_db_path!r}; "
                    "history_db_path wins",
                    UserWarning,
                    stacklevel=2,
                )
            object.__setattr__(self, "store", self.store.model_copy(update={"path": self.history_db_path}))
        return self

    def db_path(self) -> str:
        """The effective SQLite path.
        生效的 SQLite 路径。

        Input / 输入：
            none.
            无。

        Output / 输出：
            store.path.
            store.path。

        State effect / 状态影响：
            None.
            无。
        """
        return self.store.path

    @classmethod
    def from_dict(cls, d: dict) -> "HMSConfig":
        """Validate a dict: normalize (CONFIG_003) → check providers (CONFIG_002) → construct (CONFIG_001).
        校验字典：归一化（CONFIG_003）→ 检查提供者（CONFIG_002）→ 构造（CONFIG_001）。

        Input / 输入：
            d: Config dict (Mem0-shaped or HMS-shaped).
            d：配置字典（Mem0 形状或 HMS 形状）。

        Output / 输出：
            HMSConfig; an HMSError raised inside a validator is re-raised unchanged, other pydantic errors
            become ConfigError(CONFIG_001, details={'errors': e.errors()}) naming the key with a hint.
            HMSConfig；校验器内抛出的 HMSError 原样重抛，其余 pydantic 错误转为
            ConfigError(CONFIG_001, details={'errors': e.errors()}) 并指出键名与提示。

        State effect / 状态影响：
            None (the full pydantic error is logged before re-raising).
            无（重抛前记录完整的 pydantic 错误）。
        """
        if not isinstance(d, dict):
            raise ConfigError(
                f"config must be a dict, got {type(d).__name__}", "CONFIG_001", details={"type": type(d).__name__}
            )
        data = normalize_mem0_keys(d, strict=True)
        check_providers(data)
        try:
            return cls(**data)
        except pydantic.ValidationError as exc:
            logger.error("HMSConfig validation failed: %s", exc)
            inner = unwrap_hms_error(exc)
            if inner is not None:
                raise inner from exc
            errors = exc.errors()
            raise ConfigError(_describe(errors), "CONFIG_001", details={"errors": errors}) from exc


__all__ = [
    "REMOVED",
    "RENAMED",
    "CompilerConfig",
    "EmbedderConfig",
    "GraphConfig",
    "HMSConfig",
    "IndexConfig",
    "IngestConfig",
    "LlmConfig",
    "ProviderEnvelope",
    "RerankerEnvelope",
    "RetentionConfig",
    "RetrievalConfig",
    "SchedulerConfig",
    "ScoringConfig",
    "StoreConfig",
    "TextConfig",
    "check_providers",
    "default_db_path",
    "normalize_mem0_keys",
]
