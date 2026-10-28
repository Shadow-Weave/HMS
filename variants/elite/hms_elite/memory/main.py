"""HMS: the Mem0-compatible public API over the dual-plane store.
HMS：双平面存储之上、与 Mem0 兼容的公共 API。
"""

# Purpose (EN): The facade (CONTRACT §17): the ten Mem0 methods plus the HMS-native surface, each wired to one
#               group function (ingest, mvmi, graph, compiler, ops) without duplicating their logic; the §2 tail of
#               the data flow (StateAPI cards, telemetry, the background daemon) hangs off the same object.
# 目的（中）：门面（契约 §17）：十个 Mem0 方法加 HMS 原生接口，每个方法都直接接到对应的分组函数
#            （ingest、mvmi、graph、compiler、ops），不重复其逻辑；§2 数据流的尾部（StateAPI 状态卡、遥测、
#            后台守护进程）挂在同一个对象上。

# Input (EN): An HMSConfig (or Mem0-shaped dict), an optional injected StateStore, and the public call arguments.
# 输入（中）：HMSConfig（或 Mem0 形状的字典）、可选注入的 StateStore，以及公共调用参数。

# Output (EN): Mem0 envelopes ({'results': [...]}, {'message': ...}, bare history lists) and HMS objects
#              (ContextIR, AxisResult, StatePackage, report dicts).
# 输出（中）：Mem0 信封（{'results': [...]}、{'message': ...}、裸历史列表）与 HMS 对象
#            （ContextIR、AxisResult、StatePackage、报告字典）。

# State effect (EN): Every public write is one store.transaction() (add/add_events use Txn A + Txn B around an
#                    unlocked phase); every public read calls store.now() once for as_of and never writes.
# 状态影响（中）：每个公共写操作是一个 store.transaction()（add/add_events 在无锁阶段前后各有事务 A/B）；
#                每个公共读操作只调用一次 store.now() 作为 as_of，且从不写入。

from __future__ import annotations

import json
import logging
import threading
from contextlib import AbstractContextManager
from typing import Any

from hms_elite import telemetry
from hms_elite.compiler.compile import StateCompiler
from hms_elite.compiler.context import dedup_rows, hybrid_text, render_rows, sort_rows_by_time
from hms_elite.configs.base import HMSConfig
from hms_elite.daemon import StateDaemon
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import CompileMode, FeedbackType, HistoryEvent, Visibility
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.graph.explain import explain, resolve_target
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.index.base import IndexBase
from hms_elite.ingest.atoms import revise_atom
from hms_elite.ingest.events import events_from_dicts, system_event
from hms_elite.ingest.pipeline import Ingestor
from hms_elite.llms.base import LLMBase
from hms_elite.memory.base import MemoryBase
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import (
    _UNSET,
    build_scope_and_metadata,
    coerce_messages,
    normalize_expiration_date,
    reject_top_level_entity_params,
    strip_reserved_keys,
    validate_and_trim_entity_id,
    validate_and_trim_query,
    validate_search_params,
)
from hms_elite.models import AxisQuery, AxisResult, ContextIR, MemoryAtom, Scope, SourceEvent, StatePackage
from hms_elite.mvmi.filters import parse_filters
from hms_elite.mvmi.query import execute_axis_query
from hms_elite.mvmi.query import get_all as mvmi_get_all
from hms_elite.mvmi.query import search as mvmi_search
from hms_elite.ops.checkpoint import checkpoint as ops_checkpoint
from hms_elite.ops.checkpoint import rollback as ops_rollback
from hms_elite.ops.consolidation import StateScheduler
from hms_elite.ops.feedback import FeedbackProcessor
from hms_elite.ops.feedback import revoke as ops_revoke
from hms_elite.ops.handoff import build_state_package, import_state_package
from hms_elite.ops.retention import purge as ops_purge
from hms_elite.ops.transitions import export_transitions
from hms_elite.reranker.base import RerankerBase
from hms_elite.reranker.overlap import OverlapReranker
from hms_elite.state.api import StateAPI
from hms_elite.state.kinds import (
    AccountHandoffState,
    CustomerIssueState,
    DetectiveReport,
    EdgeState,
    GroupState,
    HandoffPackageState,
    IncidentState,
    MeetingDecisionState,
    PersonalState,
    RollbackPoint,
    WorkState,
)
from hms_elite.timeutil import next_day_midnight, to_utc_iso
from hms_elite.utils.factory import (
    EmbedderFactory,
    EntityExtractorFactory,
    IndexFactory,
    LlmFactory,
    RerankerFactory,
)
from hms_elite.utils.json import canonical_json
from hms_elite.utils.text import cjk_tokens_scope, entity_key, identifier_tokens, ordered_union, text_scoped

# The meta key recording the `text.cjk_tokens` mode atom_fts was last built under; absent == 'bigram' (the default),
# so English stores carry no extra meta row and stay table-for-table identical to earlier releases (CONTRACT §9.6).
# 记录 atom_fts 上次以哪种 `text.cjk_tokens` 模式构建的 meta 键；缺失即 'bigram'（默认），因此英文存储没有多余的
# meta 行，逐表与早先版本一致（CONTRACT §9.6）。
TEXT_CJK_META_KEY = "text_cjk_tokens"
# Public HMS methods whose bodies run inside cjk_tokens_scope(config.text.cjk_tokens) — every write path that builds
# atom_fts / keyword vocabularies and every read path that tokenizes a query (tests pin this list against dir(HMS)).
# Excluded on purpose: from_config (classmethod), project / state (properties), close, usage, trace, start_daemon /
# stop_daemon (the daemon thread calls consolidate, which is in the list), history, vacuum, get (no tokenizer).
# 方法体在 cjk_tokens_scope(config.text.cjk_tokens) 内运行的 HMS 公共方法——每条构建 atom_fts / 关键词词表的写路径与每条
# 对查询分词的读路径（测试把该列表与 dir(HMS) 对照钉住）。有意排除：from_config（类方法）、project / state（属性）、
# close、usage、trace、start_daemon / stop_daemon（守护线程调用的 consolidate 在列表内）、history、vacuum、get（不分词）。
TEXT_SCOPED_METHODS: tuple[str, ...] = (
    "account_handoff_state", "add", "add_events", "axis_query", "chat", "checkpoint", "compile", "consolidate",
    "context", "customer_issue_state", "delete", "delete_all", "detective_report", "edge_state", "feedback",
    "get_all", "graph_stats", "group_state", "handoff", "handoff_package", "import_package", "incident_state",
    "meeting_decision_state", "personal_state", "purge", "reset", "revoke", "rollback", "rollback_point", "search",
    "shared_card", "transitions", "update", "why", "work_state",
)  # fmt: skip

logger = logging.getLogger(__name__)

PROCEDURAL_MEMORY = "procedural_memory"
COVERAGE_SELECTION = "coverage"
# The five object types whose events carry user content (ObjectType, enums.py); HMS.context() compiles its session
# card from the newest of these — system / feedback / checkpoint / rollback / scheduler events never become the query.
# 事件承载用户内容的五种对象类型（ObjectType，enums.py）；HMS.context() 以其中最新者编译会话卡片——system / feedback /
# checkpoint / rollback / scheduler 事件绝不会成为查询。
EXTRACTABLE_OBJECT_TYPES = ("message", "tool_call", "file_diff", "test_log", "sensor_trace")
TRANSITION_FORMATS = ("tuples", "records", "jsonl")
TRANSITION_SCALES = ("L1_micro", "L2_meso", "L3_macro")
_DELETE_ALL_MESSAGE = (
    "At least one filter is required to delete all memories. "
    "If you want to delete all memories, use the `reset()` method."
)


class _OSSProject:
    """Mem0 OSS `project` shim: project settings are a platform feature, so every call raises ValueError.
    Mem0 OSS 的 `project` 占位：项目设置是平台功能，因此每个调用都抛出 ValueError。
    """

    def update(
        self,
        custom_instructions: Any = None,
        custom_categories: Any = None,
        multilingual: Any = None,
        decay: Any = None,
    ) -> None:
        """Raise ValueError (Mem0 OSS parity: project settings need the platform).
        抛出 ValueError（与 Mem0 OSS 一致：项目设置需要平台版本）。

        Input / 输入：
            custom_instructions: Ignored.
            custom_instructions：忽略。

            custom_categories: Ignored.
            custom_categories：忽略。

            multilingual: Ignored.
            multilingual：忽略。

            decay: Ignored.
            decay：忽略。

        Output / 输出：
            Never returns.
            从不返回。

        State effect / 状态影响：
            None.
            无。
        """
        raise ValueError("Project settings are not available in HMS Elite (Mem0 OSS parity: platform-only feature).")


class HMS(MemoryBase):
    """The Mem0-compatible memory API over the State Index + Support Graph store.
    基于状态索引 + 支持图存储、与 Mem0 兼容的记忆 API。
    """

    def __init__(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None:
        """Build every collaborator from the config (§17.1); an injected store is used as is.
        按配置构建全部协作对象（§17.1）；注入的 store 原样使用。

        Input / 输入：
            config: HMSConfig, a Mem0-shaped dict (HMSConfig.from_dict) or None for defaults.
            config：HMSConfig、Mem0 形状的字典（HMSConfig.from_dict）或 None（默认值）。

            store: Optional StateStore (tests inject a frozen clock and sequential ids through it).
            store：可选的 StateStore（测试通过它注入冻结时钟与顺序 id）。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            Opens (or creates) the SQLite store when none is injected; builds the configured reranker (None when
            config.reranker is None: search(rerank=True) then uses the offline OverlapReranker).
            未注入时打开（或创建）SQLite 存储；构造所配置的重排器（config.reranker 为 None 时为 None：
            此时 search(rerank=True) 使用离线的 OverlapReranker）。
        """
        if config is None:
            cfg = HMSConfig()
        elif isinstance(config, HMSConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = HMSConfig.from_dict(config)
        else:
            raise ConfigError(
                f"config must be an HMSConfig, a dict or None, got {type(config).__name__}",
                "CONFIG_001",
                details={"type": type(config).__name__},
            )
        self.config: HMSConfig = cfg
        self.store: StateStore = store or StateStore(
            cfg.store.path,
            busy_timeout_ms=cfg.store.busy_timeout_ms,
            journal_mode=cfg.store.journal_mode,
            checkpoint_on_close=cfg.store.checkpoint_on_close,
        )
        self.embedder: EmbedderBase = EmbedderFactory.create(cfg.embedder.provider, cfg.embedder.config)
        self.index: IndexBase = IndexFactory.create(
            cfg.vector_store.provider, cfg.vector_store.config, store=self.store, dims=self.embedder.dims
        )
        # Committed validity / archive changes reach the index (§10.3 update_attrs) so a provider that keeps scalar
        # gates (milvus) follows SQLite; sqlite / memory implement the hook as a no-op.
        # 已提交的有效期 / 归档变化到达索引（§10.3 update_attrs），使保存标量门控的提供者（milvus）跟随 SQLite；
        # sqlite / memory 把该钩子实现为空操作。
        # Registered exactly once per index (SQLiteIndex subscribes itself in its constructor) and removed by close().
        # 每个索引恰好注册一次（SQLiteIndex 在构造函数里自行订阅），由 close() 移除。
        if not any(getattr(cb, "__self__", None) is self.index for cb in self.store.attr_listeners):
            self.store.attr_listeners.append(self.index.update_attrs)
        # One entity extractor per facade: the ingest plane, the rule LLM (entity_mentions), feedback corrections and
        # package import all link entities through it, so 'spacy' and 'rule' never disagree inside one store.
        # 每个门面一个实体抽取器：摄取平面、规则 LLM（entity_mentions）、反馈纠正与交接包导入都经它链接实体，
        # 因此同一存储内 'spacy' 与 'rule' 永不打架。
        # ingest.zh_rules reaches the rule extractor's own switch (a copy: the config model is frozen).
        # ingest.zh_rules 传到规则抽取器自己的开关（复制一份：配置模型是冻结的）。
        extractor_config: dict[str, Any] = dict(cfg.ingest.entity_extractor_config)
        if cfg.ingest.zh_rules and cfg.ingest.entity_extractor == "rule":
            extractor_config.setdefault("zh_rules", True)
        self.entity_extractor: EntityExtractorBase = EntityExtractorFactory.create(
            cfg.ingest.entity_extractor, extractor_config
        )
        self.llm: LLMBase = LlmFactory.create(cfg.llm.provider, cfg.llm.config, entity_extractor=self.entity_extractor)
        self._sync_fts_mode()
        self.graph: SupportGraph = SupportGraph(self.store, cfg.graph)
        self.ingestor: Ingestor = Ingestor(
            self.store,
            self.index,
            self.embedder,
            self.llm,
            cfg,
            entity_extractor=self.entity_extractor,
            extraction_mode=str(cfg.ingest.extraction_mode),
            extraction_output=str(cfg.ingest.extraction_output),
        )
        self.compiler: StateCompiler = StateCompiler(
            self.store, self.index, self.embedder, self.graph, cfg, entity_extractor=self.entity_extractor
        )
        self.feedback_processor: FeedbackProcessor = FeedbackProcessor(
            self.store, self.index, self.embedder, self.graph, cfg, entity_extractor=self.entity_extractor
        )
        self.scheduler: StateScheduler = StateScheduler(
            self.store, self.index, self.embedder, self.llm, self.graph, self.ingestor, self.compiler, cfg
        )
        # The verdicts the §12.6 step-6 guard downgraded to ADD in the most recent add() / add_events() (IngestResult
        # .downgraded; the Mem0 envelope cannot carry them). Read by benchmarks that count guard rejections per run.
        # 最近一次 add() / add_events() 中被 §12.6 第 6 步守卫降为 ADD 的裁决（IngestResult.downgraded；Mem0 信封装不下
        # 它们）。供按运行统计守卫拒绝数的基准脚本读取。
        self.last_downgraded: list[dict] = []
        self.reranker: RerankerBase | None = None
        if cfg.reranker is not None:
            # The facade LLM is shared with an LLM-backed reranker unless its own config binds an `llm` envelope.
            # 除非重排器配置自带 `llm` 信封，否则 LLM 驱动的重排器共享门面的 LLM。
            bound_llm = None if cfg.reranker.config.get("llm") else self.llm
            self.reranker = RerankerFactory.create(cfg.reranker.provider, cfg.reranker.config, llm=bound_llm)
        self._default_reranker: RerankerBase | None = None
        self._project = _OSSProject()
        # Lazily built §11 state surface and the optional background daemon (start_daemon / stop_daemon).
        # 惰性构造的 §11 状态表面与可选的后台守护进程（start_daemon / stop_daemon）。
        self._state: StateAPI | None = None
        self._daemon: StateDaemon | None = None
        # HMS.context()'s session-card memo (§6.7): (scope.key(), audience) → the compiled card plus the
        # store.scope_version() it was built at; an entry is reused only while that version is unchanged.
        # HMS.context() 的会话卡片记忆（§6.7）：(scope.key(), audience) → 编译好的卡片及构建时的 store.scope_version()；
        # 只有该版本未变时条目才被复用。
        self._context_memo: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._context_lock = threading.Lock()

    def _sync_fts_mode(self) -> None:
        """Rebuild atom_fts once when the store was last written under another `text.cjk_tokens` mode (§9.6).
        当存储上次是在另一种 `text.cjk_tokens` 模式下写入时，一次性重建 atom_fts（§9.6）。

        The meta row is written only when the mode is not the default or the key already exists, so a default
        (bigram) store never gains a meta row; the rebuild and the meta write run inside unversioned().
        只有模式不是默认值、或该键已存在时才写 meta 行，因此默认（bigram）存储永远不会多出一行 meta；重建与 meta
        写入都在 unversioned() 内进行。
        """
        mode = str(self.config.text.cjk_tokens)
        stored = self.store.get_meta(TEXT_CJK_META_KEY)
        if stored is None and mode == "bigram":
            return
        if (stored or "bigram") != mode:
            with cjk_tokens_scope(mode):
                self.store.rebuild_fts()
        if stored != mode:
            with self.store.unversioned():
                self.store.set_meta(TEXT_CJK_META_KEY, mode)

    # ------------------------------------------------------------------ construction helpers
    # ------------------------------------------------------------------ 构造辅助
    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "HMS":
        """Mem0 parity constructor: validate the dict with HMSConfig.from_dict and build the facade.
        与 Mem0 一致的构造入口：用 HMSConfig.from_dict 校验字典并构建门面。

        Input / 输入：
            config_dict: Mem0-shaped or HMS-shaped config dict (§5.1).
            config_dict：Mem0 形状或 HMS 形状的配置字典（§5.1）。

        Output / 输出：
            An HMS instance; ConfigError propagates after the full error is logged.
            HMS 实例；ConfigError 在记录完整错误后向上传播。

        State effect / 状态影响：
            Opens (or creates) the configured store.
            打开（或创建）所配置的存储。
        """
        try:
            cfg = HMSConfig.from_dict(config_dict)
        except ConfigError as exc:
            logger.error("HMS.from_config failed [%s]: %s details=%s", exc.error_code, exc.message, exc.details)
            raise
        return cls(cfg)

    def __enter__(self) -> "HMS":
        """Context-manager entry (returns self).
        上下文管理器入口（返回自身）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The instance.
            实例自身。

        State effect / 状态影响：
            None.
            无。
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context-manager exit: close the store.
        上下文管理器退出：关闭存储。

        Input / 输入：
            exc: Exception triple (ignored).
            exc：异常三元组（忽略）。

        Output / 输出：
            None (exceptions propagate).
            无（异常继续传播）。

        State effect / 状态影响：
            Closes the SQLite handle.
            关闭 SQLite 句柄。
        """
        self.close()

    @property
    def project(self) -> "_OSSProject":
        """Mem0 `project` handle whose update() raises ValueError (OSS parity).
        Mem0 的 `project` 句柄，其 update() 抛出 ValueError（与 OSS 一致）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The _OSSProject shim.
            _OSSProject 占位对象。

        State effect / 状态影响：
            None.
            无。
        """
        return self._project

    def chat(self, query: str) -> None:
        """Mem0 parity stub: chat is not implemented in HMS Elite.
        与 Mem0 一致的占位：HMS Elite 未实现 chat。

        Input / 输入：
            query: Ignored.
            query：忽略。

        Output / 输出：
            Never returns (NotImplementedError).
            从不返回（NotImplementedError）。

        State effect / 状态影响：
            None.
            无。
        """
        raise NotImplementedError("Chat function not implemented yet.")

    # ------------------------------------------------------------------ Mem0-compatible surface
    # ------------------------------------------------------------------ Mem0 兼容接口
    def add(
        self,
        messages,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        timestamp: Any = None,
        expiration_date: Any = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Mem0 add(): anchor the messages, then extract (infer=True) or store raw message atoms (infer=False).
        Mem0 的 add()：为消息落锚，然后抽取（infer=True）或存原始消息原子（infer=False）。

        Input / 输入：
            messages: str, {'role','content'} dict or a list of such dicts (VALIDATION_003 otherwise).
            messages：字符串、{'role','content'} 字典或其列表（否则 VALIDATION_003）。

            user_id: Mem0 user id (at least one of the three ids is required, VALIDATION_001).
            user_id：Mem0 用户 id（三个 id 至少一个，否则 VALIDATION_001）。

            agent_id: Mem0 agent id.
            agent_id：Mem0 代理 id。

            run_id: Mem0 run id.
            run_id：Mem0 运行 id。

            metadata: Caller metadata (reserved keys stripped with a warning).
            metadata：调用方元数据（保留键被剥离并警告）。

            timestamp: Event time of the messages (any to_utc_iso form; VALIDATION_004 when invalid).
            timestamp：消息的事件时间（to_utc_iso 接受的任意形式；非法时 VALIDATION_004）。

            expiration_date: Mem0 expiration date → valid_to = next-day midnight UTC.
            expiration_date：Mem0 过期日期 → valid_to = 次日零点（UTC）。

            infer: False stores one raw message atom per message without any LLM call.
            infer：False 时每条消息存一个原始消息原子，不调用 LLM。

            memory_type: None or 'procedural_memory' (VALIDATION_002 otherwise).
            memory_type：None 或 'procedural_memory'（否则 VALIDATION_002）。

            prompt: Replacement system prompt for the procedural branch.
            prompt：过程记忆分支的替代系统提示词。

        Output / 输出：
            {'results': [row + 'event' ('ADD'|'UPDATE'|'NONE') (+ 'revises', 'conflicts_with', 'actor_id', 'role')]}.
            {'results': [行 + 'event'（'ADD'|'UPDATE'|'NONE'）（+ 'revises'、'conflicts_with'、'actor_id'、'role'）]}。

        State effect / 状态影响：
            Txn A writes source_events/anchors; Txn B writes atoms, axes, vectors, nodes, edges, history, transitions.
            事务 A 写入 source_events/anchors；事务 B 写入原子、轴、向量、节点、边、历史与转移。
        """
        scope, user_meta = build_scope_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata, context="add"
        )
        if memory_type is not None and memory_type != PROCEDURAL_MEMORY:
            raise ValidationError(
                "Invalid 'memory_type'. Please pass procedural_memory or None",
                "VALIDATION_002",
                details={"memory_type": memory_type, "allowed": [PROCEDURAL_MEMORY]},
            )
        items = coerce_messages(messages)
        event_time = to_utc_iso(timestamp) if timestamp is not None else None
        expiration = normalize_expiration_date(expiration_date)
        valid_to = next_day_midnight(expiration) if expiration else None
        result = self.ingestor.ingest_messages(
            items,
            scope=scope,
            user_meta=user_meta,
            infer=bool(infer),
            memory_type=memory_type,
            prompt=prompt,
            event_time=event_time,
            valid_to=valid_to,
        )
        self.last_downgraded = list(result.downgraded)
        return {"results": list(result.rows)}

    def get(self, memory_id: str) -> dict | None:
        """Mem0 get(): the head version of a lineage as a row, or None.
        Mem0 的 get()：谱系头版本的行，或 None。

        Input / 输入：
            memory_id: Mem0 memory id (lineage id).
            memory_id：Mem0 记忆 id（谱系 id）。

        Output / 输出：
            Row (§4.1) or None when unknown, deleted, or every version archived.
            行（§4.1）；未知、已删除或所有版本均已归档时为 None。

        State effect / 状态影响：
            None.
            无。
        """
        atom = self.store.resolve_head(str(memory_id).strip())
        return None if atom is None else atom.to_public_dict()

    def get_all(self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs) -> dict:
        """Mem0 get_all(): axis-only listing in time order.
        Mem0 的 get_all()：仅走轴的按时间顺序列表。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key (VALIDATION_001 otherwise).
            filters：至少含一个范围键的 Mem0 过滤字典（否则 VALIDATION_001）。

            top_k: Result cap (non-negative int, VALIDATION_009).
            top_k：结果上限（非负整数，否则 VALIDATION_009）。

            show_expired: Include archived / expired versions (with their valid_to).
            show_expired：包含已归档 / 已过期版本（带 valid_to）。

            kwargs: Top-level user_id/agent_id/run_id are rejected (VALIDATION_008, a ValueError).
            kwargs：顶层 user_id/agent_id/run_id 会被拒绝（VALIDATION_008，属 ValueError）。

        Output / 输出：
            {'results': [row...]} without 'score', ordered time_anchor DESC, created_at DESC, atom_id ASC.
            不带 'score' 的 {'results': [行...]}，按 time_anchor 降序、created_at 降序、atom_id 升序。

        State effect / 状态影响：
            None (one store.now() for as_of).
            无（调用一次 store.now() 作为 as_of）。
        """
        reject_top_level_entity_params(kwargs, "get_all")
        parse_filters(filters, require_scope=True, context="get_all")
        validate_search_params(None, top_k)
        as_of = self.store.now()
        rows = mvmi_get_all(
            self.store, filters=dict(filters or {}), top_k=int(top_k), as_of=as_of, include_expired=bool(show_expired)
        )
        return {"results": rows}

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Any = None,
        show_expired: bool = False,
        **kwargs,
    ) -> dict:
        """Mem0 search(): planned multi-signal retrieval over the State Index.
        Mem0 的 search()：基于状态索引的规划式多信号检索。

        Input / 输入：
            query: Non-empty query text (VALIDATION_010).
            query：非空查询文本（否则 VALIDATION_010）。

            top_k: Result cap (VALIDATION_009).
            top_k：结果上限（否则 VALIDATION_009）。

            filters: Mem0 filter dict with >= 1 scope key (VALIDATION_001).
            filters：至少含一个范围键的 Mem0 过滤字典（否则 VALIDATION_001）。

            threshold: Minimum fused score in [0, 1] (VALIDATION_009).
            threshold：融合后最低分数，范围 [0, 1]（否则 VALIDATION_009）。

            rerank: Re-order the fused rows with self.reranker (unconfigured → offline OverlapReranker, one info log); rows gain 'rerank_score'; a reranker error keeps the retrieval order.
            rerank：用 self.reranker 重排融合后的行（未配置 → 离线 OverlapReranker，一条 info 日志）；各行追加 'rerank_score'；重排器出错时保持检索顺序。

            explain: Attach 'score_details' to every row.
            explain：为每行附加 'score_details'。

            reference_date: Validity instant (as_of); None → store.now().
            reference_date：有效性时刻（as_of）；None → store.now()。

            show_expired: Drop the validity predicate.
            show_expired：去掉有效性谓词。

            kwargs: Top-level user_id/agent_id/run_id are rejected (VALIDATION_008, a ValueError).
            kwargs：顶层 user_id/agent_id/run_id 会被拒绝（VALIDATION_008，属 ValueError）。

        Output / 输出：
            {'results': [row + 'score' (+ 'score_details')]} in rank order; with rerank=True the rows are in
            rerank order and carry 'rerank_score' (explain=True also appends 'rerank_score' and 'reranker' to
            'score_details').
            按排名顺序的 {'results': [行 + 'score'（+ 'score_details'）]}；rerank=True 时各行按重排顺序排列并带
            'rerank_score'（explain=True 时 'score_details' 还追加 'rerank_score' 与 'reranker'）。

        State effect / 状态影响：
            None (one store.now() when reference_date is None); an LLM reranker calls its provider.
            无（reference_date 为 None 时调用一次 store.now()）；LLM 重排器会调用其提供者。
        """
        reject_top_level_entity_params(kwargs, "search")
        parse_filters(filters, require_scope=True, context="search")
        validate_search_params(threshold, top_k)
        text = validate_and_trim_query(query)
        as_of = to_utc_iso(reference_date) if reference_date is not None else self.store.now()
        rows = mvmi_search(
            self.store,
            self.index,
            self.embedder,
            query=text,
            filters=dict(filters or {}),
            top_k=int(top_k),
            threshold=float(threshold),
            explain=bool(explain),
            as_of=as_of,
            include_expired=bool(show_expired),
            scoring=self.config.scoring,
            retrieval=self.config.retrieval,
            capabilities=self.store.capabilities,
            extractor=self.entity_extractor,
        )
        if rerank and rows:
            rows = self._rerank_rows(text, rows, top_k=int(top_k), explain=bool(explain))
        return {"results": rows}

    def _active_reranker(self) -> RerankerBase:
        """The configured reranker, or a lazily built OverlapReranker announced once at info level.
        所配置的重排器，或惰性构造、以 info 级别宣告一次的 OverlapReranker。
        """
        if self.reranker is not None:
            return self.reranker
        if self._default_reranker is None:
            self._default_reranker = OverlapReranker()
            logger.info("rerank=True without config.reranker; using the default offline overlap reranker")
        return self._default_reranker

    def _rerank_rows(self, query: str, rows: list[dict], *, top_k: int, explain: bool) -> list[dict]:
        """Rerank fused rows; any reranker exception keeps the retrieval order (Mem0 discipline).
        重排融合后的行；重排器的任何异常都保持检索顺序（Mem0 同款纪律）。
        """
        reranker = self._active_reranker()
        # Mem0 parity: any provider failure (transport, exhausted script, malformed reply) degrades to the
        # retrieval order with a warning instead of failing the search.
        # 与 Mem0 一致：提供者的任何失败（传输、脚本耗尽、畸形回复）都退化为带警告的检索顺序，而不是让搜索失败。
        try:
            ranked = reranker.rerank(query, rows, top_k=top_k)
        except Exception as exc:
            logger.warning("rerank failed (%s: %s); returning the retrieval order", type(exc).__name__, exc)
            return rows
        if not explain:
            return ranked
        # score_details is shared with the pre-rerank row (rank_rows copies rows shallowly), so extend a copy.
        # score_details 与重排前的行共享（rank_rows 是浅拷贝），因此在其副本上追加。
        return [
            {
                **row,
                "score_details": {
                    **(row.get("score_details") or {}),
                    "rerank_score": row["rerank_score"],
                    "reranker": reranker.identity,
                },
            }
            for row in ranked
        ]

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict | None = None,
        expiration_date: Any = _UNSET,
        data: str | None = None,
    ) -> dict:
        """Mem0 update(): a new text is a new lineage version; metadata and expiration edit the head in place.
        Mem0 的 update()：新文本产生新的谱系版本；元数据与过期日期就地修改头版本。

        Input / 输入：
            memory_id: Mem0 memory id (unknown/deleted → NOT_FOUND_001, a ValueError).
            memory_id：Mem0 记忆 id（未知/已删除 → NOT_FOUND_001，属 ValueError）。

            text: New text (creates version n+1 with a revision edge and an UPDATE history row).
            text：新文本（创建版本 n+1，附 revision 边与 UPDATE 历史行）。

            metadata: Metadata patch merged over the current user_meta (identity keys stripped).
            metadata：合并到当前 user_meta 之上的元数据补丁（身份键被剥离）。

            expiration_date: _UNSET = untouched; None = clear valid_to; a date → next-day midnight.
            expiration_date：_UNSET 表示不变；None 表示清除 valid_to；日期 → 次日零点。

            data: Deprecated alias of text (warns).
            data：text 的弃用别名（警告）。

        Output / 输出：
            {'message': 'Memory updated successfully!'}.
            {'message': 'Memory updated successfully!'}。

        State effect / 状态影响：
            One transaction: a system event + anchor, then revise_atom / set_user_meta / set_valid_to + history.
            一个事务：系统事件 + 锚点，然后 revise_atom / set_user_meta / set_valid_to + 历史。
        """
        if data is not None:
            logger.warning("update(data=...) is deprecated; use update(text=...)")
            if text is None:
                text = data
        has_expiration = expiration_date is not _UNSET
        if text is None and metadata is None and not has_expiration:
            raise ValidationError(
                "update() needs at least one of text, metadata or expiration_date",
                "VALIDATION_011",
                details={"memory_id": memory_id},
            )
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValidationError(
                "update() text must be a non-empty string", "VALIDATION_011", details={"memory_id": memory_id}
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError(
                f"metadata must be a dict, got {type(metadata).__name__}",
                "VALIDATION_011",
                details={"memory_id": memory_id},
            )
        head = self._require_head(memory_id)
        expiration = normalize_expiration_date(expiration_date) if has_expiration else None
        with self.store.transaction():
            now = self.store.now()
            _ev, sys_anchor = system_event(
                self.store,
                scope=head.scope,
                object_type="system",
                event_type="updated",
                payload={
                    "op": "update",
                    "memory_id": head.memory_id,
                    "atom_id": head.atom_id,
                    "text": text,
                    "metadata": dict(metadata) if metadata is not None else None,
                    "expiration_date": expiration,
                },
            )
            user_meta = dict(head.user_meta)
            if metadata is not None:
                user_meta.update(strip_reserved_keys(metadata, context="update"))
            if text is not None:
                if has_expiration:
                    valid_to = next_day_midnight(expiration) if expiration else None
                else:
                    valid_to = head.valid_to if head.valid_to is not None and head.valid_to > now else None
                successor = self._successor(
                    head, text, user_meta, anchor_id=sys_anchor.anchor_id, now=now, valid_to=valid_to
                )
                revise_atom(
                    self.store,
                    self.index,
                    self.embedder,
                    old=head,
                    new=successor,
                    witness_anchor_ids=ordered_union([sys_anchor.anchor_id], list(head.anchor_ids)),
                    at=now,
                    reason="revised",
                    created_by="ops",
                    history_event=HistoryEvent.UPDATE.value,
                )
            else:
                if metadata is not None:
                    self.store.set_user_meta(head.atom_id, user_meta, at=now)
                    self.store.add_history(
                        memory_id=head.memory_id,
                        old_memory=head.text,
                        new_memory=head.text,
                        event=HistoryEvent.UPDATE.value,
                        created_at=head.created_at or now,
                        updated_at=now,
                        anchor_id=sys_anchor.anchor_id,
                        atom_id=head.atom_id,
                        details={"metadata_only": True},
                    )
                if has_expiration:
                    self.store.set_valid_to(head.atom_id, next_day_midnight(expiration) if expiration else None, at=now)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> dict:
        """Mem0 delete(): archive the head (reason 'deleted'); content, edges and history are kept.
        Mem0 的 delete()：归档头版本（原因 'deleted'）；内容、边与历史均保留。

        Input / 输入：
            memory_id: Mem0 memory id (unknown/deleted → NOT_FOUND_001, a ValueError).
            memory_id：Mem0 记忆 id（未知/已删除 → NOT_FOUND_001，属 ValueError）。

        Output / 输出：
            {'message': 'Memory deleted successfully!'}.
            {'message': 'Memory deleted successfully!'}。

        State effect / 状态影响：
            One transaction: system event + anchor, archive_atom, DELETE history row (is_deleted=1).
            一个事务：系统事件 + 锚点、archive_atom、DELETE 历史行（is_deleted=1）。
        """
        head = self._require_head(memory_id)
        with self.store.transaction():
            now = self.store.now()
            _ev, sys_anchor = system_event(
                self.store,
                scope=head.scope,
                object_type="system",
                event_type="deleted",
                payload={"op": "delete", "memory_id": head.memory_id, "atom_id": head.atom_id},
            )
            self._archive_deleted(head, sys_anchor.anchor_id, now)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> dict:
        """Mem0 delete_all(): archive every live head in the scope (reason 'deleted').
        Mem0 的 delete_all()：归档范围内每个存活头版本（原因 'deleted'）。

        Input / 输入：
            user_id: Mem0 user id filter.
            user_id：Mem0 用户 id 过滤。

            agent_id: Mem0 agent id filter.
            agent_id：Mem0 代理 id 过滤。

            run_id: Mem0 run id filter (at least one of the three is required, VALIDATION_001).
            run_id：Mem0 运行 id 过滤（三者至少一个，否则 VALIDATION_001）。

        Output / 输出：
            {'message': 'Memories deleted successfully!'}.
            {'message': 'Memories deleted successfully!'}。

        State effect / 状态影响：
            One transaction and one system event; every head archived with a DELETE history row.
            一个事务、一个系统事件；每个头版本归档并写入 DELETE 历史行。
        """
        ids = {
            "user_id": validate_and_trim_entity_id(user_id, "user_id"),
            "agent_id": validate_and_trim_entity_id(agent_id, "agent_id"),
            "run_id": validate_and_trim_entity_id(run_id, "run_id"),
        }
        if not any(ids.values()):
            raise ValidationError(_DELETE_ALL_MESSAGE, "VALIDATION_001", details={"provided_ids": ids})
        scope = Scope(**ids)
        with self.store.transaction():
            now = self.store.now()
            _ev, sys_anchor = system_event(
                self.store, scope=scope, object_type="system", event_type="deleted", payload={"op": "delete_all"}
            )
            heads = self.store.axis_select(
                scope, where_sql="memory_atoms.archived_reason IS NULL", include_expired=True, order="atom_id ASC"
            )
            for atom in heads:
                self._archive_deleted(atom, sys_anchor.anchor_id, now)
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id: str) -> list[dict]:
        """Mem0 history(): the lineage's history rows (Mem0's 10 keys + anchor_id) in seq order.
        Mem0 的 history()：谱系的历史行（Mem0 的 10 个键 + anchor_id），按 seq 顺序。

        Input / 输入：
            memory_id: Mem0 memory id (unknown → []).
            memory_id：Mem0 记忆 id（未知 → []）。

        Output / 输出：
            Bare list of history dicts.
            历史字典的裸列表。

        State effect / 状态影响：
            None.
            无。
        """
        return self.store.get_history(str(memory_id).strip())

    def reset(self) -> None:
        """Mem0 reset(): drop every table, re-create the schema and clear the index.
        Mem0 的 reset()：删除全部表、重建模式并清空索引。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Destroys all data in the store and the index.
            销毁存储与索引中的全部数据。
        """
        self.store.reset()
        self.index.reset()
        with self._context_lock:
            self._context_memo.clear()

    def close(self) -> None:
        """Close the index (when its provider has a close hook) and the store (idempotent).
        关闭索引（其提供者带 close 钩子时）与存储（幂等）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Stops a running daemon, then releases the SQLite handle and, for providers holding an external client
            (milvus), that client.
            先停止仍在运行的守护进程，再释放 SQLite 句柄；持有外部客户端的提供者（milvus）同时释放该客户端。
        """
        # A running daemon would call consolidate() on a closed store, so it is stopped (joined) before anything else.
        # 仍在运行的守护进程会对已关闭的存储调用 consolidate()，因此先于一切把它停掉（并等待线程退出）。
        self.stop_daemon()
        # IndexBase.close() is a no-op for sqlite (shares the store's connection) and memory (a dict); providers that
        # own an external client (milvus) release it there, and an explicit HMS.close() must reach it.
        # IndexBase.close() 对 sqlite（共用 store 的连接）与 memory（字典）是空操作；拥有外部客户端的提供者（milvus）
        # 在那里释放它，显式的 HMS.close() 必须调到它。
        # The listener leaves the store first: a shared StateStore outlives this facade and must not keep calling
        # a closed provider (milvus raises StoreError per change; memory / sqlite would just accumulate callbacks).
        # 监听器先离开 store：共享的 StateStore 比本门面活得久，绝不能继续调用已关闭的提供者（milvus 每次变化都抛
        # StoreError；memory / sqlite 则只是不断累积回调）。
        self.store.attr_listeners = [
            cb for cb in self.store.attr_listeners if getattr(cb, "__self__", None) is not self.index
        ]
        try:
            self.index.close()
        finally:
            self.store.close()

    # ------------------------------------------------------------------ HMS-native surface
    # ------------------------------------------------------------------ HMS 原生接口
    def add_events(
        self,
        events: list[SourceEvent | dict],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        infer: bool = True,
    ) -> dict:
        """Ingest source events (messages and structured tool/file/test/sensor events) through the closure builder.
        通过闭合构建器摄取源事件（消息以及结构化的工具/文件/测试/传感器事件）。

        Input / 输入：
            events: SourceEvents or dicts (object_type, payload, optional role/actor_id/timestamp/visibility/scope).
            events：SourceEvent 或字典（object_type、payload，以及可选的 role/actor_id/timestamp/visibility/范围）。

            user_id: Fallback user id for items without scope keys.
            user_id：无范围键条目的回退用户 id。

            agent_id: Fallback agent id.
            agent_id：回退代理 id。

            run_id: Fallback run id.
            run_id：回退运行 id。

            infer: Whether message events go through LLM extraction (structured events never call the LLM).
            infer：消息事件是否走 LLM 抽取（结构化事件从不调用 LLM）。

        Output / 输出：
            {'results': rows, 'events': [{'event_id','anchor_id','object_type','dedup'}], 'eps', 'edges', 'constraints'}.
            {'results': 行, 'events': [{'event_id','anchor_id','object_type','dedup'}], 'eps', 'edges', 'constraints'}。

        State effect / 状态影响：
            Txn A anchors every event; one Txn B per (scope, path) writes atoms, EPs, edges, transitions.
            事务 A 为每个事件落锚；每个（范围，路径）一个事务 B，写入原子、EP、边与转移。
        """
        fallback = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        items = events_from_dicts(
            list(events),
            scope_fallback=None if fallback.is_empty() else fallback,
            visibility_default=str(self.config.visibility_default),
            event_time=self.store.now(),
        )
        result = self.ingestor.ingest_events(items, infer=bool(infer))
        self.last_downgraded = list(result.downgraded)
        return {
            "results": list(result.rows),
            "events": list(result.events),
            "eps": list(result.ep_ids),
            "edges": list(result.edge_ids),
            "constraints": list(result.constraint_ids),
        }

    def compile(
        self,
        query: str,
        *,
        filters: dict,
        budget_tokens: int | None = None,
        mode: str | None = None,
        as_of: Any = None,
        require_witness: bool | None = None,
    ) -> ContextIR:
        """Compile the Work State for a query (§15.2): axis first, graph on demand, deterministic slots and budget.
        为查询编译工作态（§15.2）：先轴，按需图扩展，确定性槽位与预算。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            filters: Mem0 filter dict with >= 1 scope key; a scalar 'visibility' is the audience.
            filters：至少含一个范围键的 Mem0 过滤字典；标量 'visibility' 即受众。

            budget_tokens: Token budget (None → config.compiler.budget_tokens).
            budget_tokens：token 预算（None → config.compiler.budget_tokens）。

            mode: 'auto' | 'fast' | 'axis_graph' | 'deep' | 'coverage' (None → config default; VALIDATION_013 otherwise; 'coverage' = the default mode with compiler.selection switched to 'coverage').
            mode：'auto' | 'fast' | 'axis_graph' | 'deep' | 'coverage'（None → 配置默认值；否则 VALIDATION_013；'coverage' = 默认模式并把 compiler.selection 切换为 'coverage'）。

            as_of: Reference instant (None → store.now()).
            as_of：参考时刻（None → store.now()）。

            require_witness: Override config.compiler.require_witness for PC-IR lowering.
            require_witness：覆盖 config.compiler.require_witness（用于 PC-IR 降低）。

        Output / 输出：
            ContextIR (render with .to_text(), lower with .to_pcir()).
            ContextIR（用 .to_text() 渲染，用 .to_pcir() 降低）。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="compile")
        # 'coverage' is a selection, not a graph-expansion mode: it keeps the configured mode and switches the
        # step-8 packer to requirement coverage (CompilerConfig.selection).
        # 'coverage' 是一种选择方式而非图扩展模式：保留配置的模式，只把第 8 步的打包器切换为需求覆盖
        # （CompilerConfig.selection）。
        selection: str | None = None
        if str(getattr(mode, "value", mode)) == COVERAGE_SELECTION:
            selection, mode = COVERAGE_SELECTION, None
        mode_value = self._mode_value(mode)
        instant = to_utc_iso(as_of) if as_of is not None else self.store.now()
        data = dict(filters or {})
        audience = data.get("visibility") if isinstance(data.get("visibility"), str) else None
        return self.compiler.compile(
            query,
            scope=scope,
            filters=data,
            budget_tokens=budget_tokens,
            mode=mode_value,
            as_of=instant,
            require_witness=require_witness,
            audience=audience,
            selection=selection,
        )

    def axis_query(self, aq: AxisQuery | dict) -> AxisResult:
        """HMS-native seven-axis query (HMS §3.5) executed by mvmi.query.execute_axis_query.
        HMS 原生七轴查询（HMS §3.5），由 mvmi.query.execute_axis_query 执行。

        Input / 输入：
            aq: AxisQuery or its dict form; scope from aq.scope or the scope keys in aq.filters (VALIDATION_001).
            aq：AxisQuery 或其字典形式；范围来自 aq.scope 或 aq.filters 中的范围键（否则 VALIDATION_001）。

        Output / 输出：
            AxisResult (hits, projection, count, summary, primary_axis).
            AxisResult（hits、projection、count、summary、primary_axis）。

        State effect / 状态影响：
            None (one store.now() when aq.as_of is None).
            无（aq.as_of 为 None 时调用一次 store.now()）。
        """
        if isinstance(aq, dict):
            aq = AxisQuery(**aq)
        elif not isinstance(aq, AxisQuery):
            raise ValidationError(
                f"axis_query expects an AxisQuery or a dict, got {type(aq).__name__}",
                "VALIDATION_013",
                details={"type": type(aq).__name__},
            )
        if aq.scope is None or aq.scope.is_empty():
            fscope, _expr = parse_filters(aq.filters, require_scope=False, context="axis_query")
            fscope.require_any(context="axis_query")
        if aq.as_of is None:
            aq = aq.model_copy(update={"as_of": self.store.now()})
        return execute_axis_query(
            self.store,
            self.index,
            self.embedder,
            aq,
            scoring=self.config.scoring,
            capabilities=self.store.capabilities,
            extractor=self.entity_extractor,
        )

    def why(
        self,
        target_id: str,
        *,
        hops: int = 2,
        filters: dict | None = None,
        edge_types: list[str] | None = None,
        include_invalid: bool = False,
    ) -> dict:
        """Explain an object (or a question, resolved through a detective compile) by backward BFS over the graph.
        通过图上的反向 BFS 解释一个对象（或经侦探式编译解析出的问题）。

        Input / 输入：
            target_id: memory/atom/ep/obs/interface_ep/edge id, or a question when filters are given.
            target_id：memory/atom/ep/obs/interface_ep/edge 的 id；给定 filters 时也可以是一个问题。

            hops: Expansion levels in [0, 6] (VALIDATION_013 otherwise).
            hops：扩展层数，范围 [0, 6]（否则 VALIDATION_013）。

            filters: Scope filters enabling question mode (compile(mode='deep', budget_tokens=600)).
            filters：启用问题模式的范围过滤（compile(mode='deep', budget_tokens=600)）。

            edge_types: Incoming edge types to follow (None → WHY_EDGE_TYPES).
            edge_types：跟随的入边类型（None → WHY_EDGE_TYPES）。

            include_invalid: Also follow invalid edges.
            include_invalid：是否也沿失效边扩展。

        Output / 输出：
            {'target', 'nodes', 'edges', 'anchors', 'interface_eps', 'paths', 'unresolved_conflicts', 'refined_from', 'chain', 'explanation'}.
            {'target', 'nodes', 'edges', 'anchors', 'interface_eps', 'paths', 'unresolved_conflicts', 'refined_from', 'chain', 'explanation'}。

        State effect / 状态影响：
            None (read-only; NOT_FOUND_001 when nothing resolves).
            无（只读；无法解析时 NOT_FOUND_001）。
        """
        if isinstance(hops, bool) or not isinstance(hops, int) or not 0 <= hops <= 6:
            raise ValidationError(
                f"hops must be an integer in [0, 6], got {hops!r}", "VALIDATION_013", details={"hops": hops}
            )
        target = resolve_target(self.store, str(target_id))
        if target is None and filters:
            target = self._target_from_question(str(target_id), filters)
        if target is None:
            raise ValidationError(
                f"why(): no object or answer found for {target_id!r}", "NOT_FOUND_001", details={"target_id": target_id}
            )
        return explain(
            self.store,
            self.graph,
            target,
            hops=hops,
            edge_types=edge_types,
            include_invalid=bool(include_invalid),
            max_nodes=self.config.compiler.max_why_nodes,
        )

    def feedback(
        self,
        target_id: str | None,
        feedback: FeedbackType | str,
        *,
        reason: str | None = None,
        correction: str | dict | None = None,
        actor_id: str | None = None,
    ) -> dict:
        """Apply one of the eleven state feedback types (§16.1) to a target.
        对目标应用十一种状态反馈类型之一（§16.1）。

        Input / 输入：
            target_id: memory/atom/ep/obs/interface_ep/edge/anchor id, '<atom_id>#…', a Scope.key() or None (state_missing).
            target_id：memory/atom/ep/obs/interface_ep/edge/anchor 的 id、'<atom_id>#…'、Scope.key() 或 None（state_missing）。

            feedback: FeedbackType value (unknown → VALIDATION_006).
            feedback：FeedbackType 值（未知 → VALIDATION_006）。

            reason: Free-text reason.
            reason：自由文本原因。

            correction: Correction payload (semantics per feedback type; str or dict).
            correction：纠正载荷（语义随反馈类型而定；str 或 dict）。

            actor_id: Who gave the feedback.
            actor_id：反馈提供者。

        Output / 输出：
            {'message': 'Feedback applied', 'feedback_id', 'actions': [...]}.
            {'message': 'Feedback applied', 'feedback_id', 'actions': [...]}。

        State effect / 状态影响：
            One transaction witnessed by the feedback anchor.
            一个由反馈锚点见证的事务。
        """
        return self.feedback_processor.process(
            target_id=target_id,
            feedback=str(getattr(feedback, "value", feedback)),
            reason=reason,
            correction=correction,
            actor_id=actor_id,
            scope_hint=None,
        )

    def revoke(self, anchor_id: str, *, reason: str | None = None, actor_id: str | None = None) -> dict:
        """Revoke an evidence anchor and cascade to everything it witnessed (§16.1 revoke).
        撤销一个证据锚点并级联到其见证的一切（§16.1 revoke）。

        Input / 输入：
            anchor_id: The anchor to revoke (unknown → NOT_FOUND_001).
            anchor_id：要撤销的锚点（未知 → NOT_FOUND_001）。

            reason: Optional reason recorded on the system event.
            reason：记录在系统事件上的可选原因。

            actor_id: Optional actor of the system event.
            actor_id：系统事件的可选行动者。

        Output / 输出：
            {'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'} (+ 'noop').
            {'revoked', 'archived', 'edges_invalidated', 'interface_eps_invalidated', 'anchor_id'}（+ 'noop'）。

        State effect / 状态影响：
            One transaction: anchors.revoked_at, disputes, archives + history, edge/InterfaceEP invalidations.
            一个事务：anchors.revoked_at、争议、归档与历史、边/InterfaceEP 失效。
        """
        return ops_revoke(self.store, self.graph, anchor_id=str(anchor_id), reason=reason, actor_id=actor_id)

    def handoff(
        self,
        *,
        filters: dict,
        goal: str | None = None,
        budget_tokens: int | None = None,
        checkpoint: bool = True,
        include_edges: bool = True,
    ) -> StatePackage:
        """Build a StatePackage (deep Work State + witness closure), optionally after taking a checkpoint.
        构建 StatePackage（deep 工作态 + 见证闭包），可先建立检查点。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Explicit goal; None → the newest valid EP's u in the run, else None.
            goal：显式目标；None → 该运行中最新有效 EP 的 u，否则 None。

            budget_tokens: Token budget (None → config.compiler.budget_tokens).
            budget_tokens：token 预算（None → config.compiler.budget_tokens）。

            checkpoint: Take a checkpoint labelled 'handoff:<goal>' first and embed its id.
            checkpoint：先建立标签为 'handoff:<goal>' 的检查点并嵌入其 id。

            include_edges: Include the valid edges among selected nodes.
            include_edges：包含所选节点之间的有效边。

        Output / 输出：
            StatePackage (serialize with .to_json()).
            StatePackage（用 .to_json() 序列化）。

        State effect / 状态影响：
            Only the optional checkpoint writes; the package itself is read-only.
            仅可选的检查点会写入；交接包本身只读。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="handoff")
        cp = self.checkpoint(filters=filters, label=f"handoff:{goal or ''}") if checkpoint else None
        return build_state_package(
            self.store,
            self.graph,
            self.compiler,
            config=self.config,
            scope=scope,
            filters=dict(filters or {}),
            goal=goal,
            budget_tokens=int(budget_tokens or self.config.compiler.budget_tokens),
            checkpoint_id=cp,
            include_edges=bool(include_edges),
        )

    def import_package(
        self,
        pkg: StatePackage | str | dict,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Recreate a StatePackage in this store (HMS §9.4): ids kept, identity dedup, optional re-scoping.
        在本存储中重建 StatePackage（HMS §9.4）：保留 id、按身份去重、可选的范围替换。

        Input / 输入：
            pkg: StatePackage, its JSON string or its dict form.
            pkg：StatePackage、其 JSON 字符串或字典形式。

            user_id: Override user id (any override replaces the package scope for every imported object).
            user_id：覆盖的用户 id（任一覆盖都会替换每个导入对象的范围）。

            agent_id: Override agent id.
            agent_id：覆盖的代理 id。

            run_id: Override run id.
            run_id：覆盖的运行 id。

        Output / 输出：
            {'imported': {kind: n}, 'skipped': {kind: n}, 'id_map', 'anchor_id'}.
            {'imported': {kind: n}, 'skipped': {kind: n}, 'id_map', 'anchor_id'}。

        State effect / 状态影响：
            One transaction witnessed by the import system anchor.
            一个由导入系统锚点见证的事务。
        """
        if isinstance(pkg, str):
            package = StatePackage.from_json(pkg)
        elif isinstance(pkg, dict):
            package = StatePackage.model_validate(pkg)
        elif isinstance(pkg, StatePackage):
            package = pkg
        else:
            raise ValidationError(
                f"import_package expects a StatePackage, JSON string or dict, got {type(pkg).__name__}",
                "VALIDATION_013",
                details={"type": type(pkg).__name__},
            )
        overrides = {k: v for k, v in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)) if v}
        override = Scope(**{**package.scope.as_filters(), **overrides}) if overrides else None
        return import_state_package(
            self.store,
            self.index,
            self.embedder,
            self.graph,
            pkg=package,
            scope_override=override,
            entity_extractor=self.entity_extractor,
        )

    def checkpoint(self, *, filters: dict, label: str | None = None) -> str:
        """Snapshot the validity of every atom, edge, InterfaceEP and EP of a scope.
        快照某范围内每个原子、边、InterfaceEP 与 EP 的有效性。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            label: Optional human label.
            label：可选的人类可读标签。

        Output / 输出：
            The checkpoint id (prefix 'ckp').
            检查点 id（前缀 'ckp'）。

        State effect / 状态影响：
            One transaction writing source_events, anchors and checkpoints.
            一个事务，写入 source_events、anchors 与 checkpoints。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="checkpoint")
        return ops_checkpoint(self.store, scope=scope, label=label)

    def rollback(self, checkpoint_id: str, *, reason: str | None = None) -> dict:
        """Restore a scope to a checkpoint (archives, restores, edges, InterfaceEPs, rollback edges + transition).
        将范围恢复到检查点（归档、恢复、边、InterfaceEP、rollback 边与转移）。

        Input / 输入：
            checkpoint_id: The checkpoint to restore (unknown → NOT_FOUND_001, a ValueError).
            checkpoint_id：要恢复的检查点（未知 → NOT_FOUND_001，属 ValueError）。

            reason: Optional reason recorded on the rollback event and transition.
            reason：记录在回滚事件与转移上的可选原因。

        Output / 输出：
            {'message', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}.
            {'message', 'checkpoint_id', 'restored', 'archived', 'edges_invalidated', 'edges_revalidated', 'edges_created', 'anchor_id'}。

        State effect / 状态影响：
            One transaction witnessed by the rollback anchor; content never touched.
            一个由回滚锚点见证的事务；从不修改内容。
        """
        return ops_rollback(self.store, self.graph, checkpoint_id=str(checkpoint_id), reason=reason)

    def transitions(
        self,
        *,
        filters: dict,
        since: Any = None,
        scale: str | None = None,
        fmt: str = "tuples",
        kinds: list[str] | None = None,
    ) -> list | str:
        """Export the (S_t, a_t, S_t+1, r_t, κ_t, w_t) transitions of a scope (§16.4), idempotently.
        幂等地导出某范围的 (S_t, a_t, S_t+1, r_t, κ_t, w_t) 转移（§16.4）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            since: Inclusive lower bound on time_t1 (any to_utc_iso form).
            since：time_t1 的下界（含；to_utc_iso 接受的任意形式）。

            scale: 'L1_micro' | 'L2_meso' | 'L3_macro' or None (VALIDATION_013 otherwise).
            scale：'L1_micro' | 'L2_meso' | 'L3_macro' 或 None（否则 VALIDATION_013）。

            fmt: 'tuples' | 'records' | 'jsonl' (VALIDATION_013 otherwise).
            fmt：'tuples' | 'records' | 'jsonl'（否则 VALIDATION_013）。

            kinds: Optional source_kind whitelist ('ep', 'interface_ep', 'causal', 'revision', 'conflict', 'rollback').
            kinds：可选的 source_kind 白名单（'ep'、'interface_ep'、'causal'、'revision'、'conflict'、'rollback'）。

        Output / 输出：
            [t.to_tuple()] | [t.model_dump()] | one sorted-key JSON object per line (ensure_ascii=False).
            [t.to_tuple()] | [t.model_dump()] | 每行一个键排序的 JSON 对象（ensure_ascii=False）。

        State effect / 状态影响：
            One transaction writing only new transitions rows (INSERT OR IGNORE).
            一个事务，仅写入新的 transitions 行（INSERT OR IGNORE）。
        """
        if fmt not in TRANSITION_FORMATS:
            raise ValidationError(
                f"unknown transitions fmt {fmt!r}",
                "VALIDATION_013",
                details={"fmt": fmt, "supported": list(TRANSITION_FORMATS)},
            )
        if scale is not None and str(scale) not in TRANSITION_SCALES:
            raise ValidationError(
                f"unknown transitions scale {scale!r}",
                "VALIDATION_013",
                details={"scale": scale, "supported": list(TRANSITION_SCALES)},
            )
        scope, _expr = parse_filters(filters, require_scope=True, context="transitions")
        rows = export_transitions(
            self.store,
            scope=scope,
            since=to_utc_iso(since) if since is not None else None,
            scale=str(scale) if scale is not None else None,
            kinds=[str(k) for k in kinds] if kinds else None,
        )
        if fmt == "tuples":
            return [t.to_tuple() for t in rows]
        if fmt == "records":
            return [t.model_dump() for t in rows]
        return "\n".join(json.dumps(t.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for t in rows)

    def consolidate(self, *, filters: dict | None = None, phases: list[str] | None = None) -> dict:
        """Run the synchronous StateScheduler phases (orient/collect/consolidate/prune) on the matched scopes.
        在匹配的范围上同步运行 StateScheduler 阶段（orient/collect/consolidate/prune）。

        Input / 输入：
            filters: Full or partial scope filters (None → every distinct scope in memory_atoms).
            filters：完整或部分范围过滤（None → memory_atoms 中的每个独立范围）。

            phases: Phase names in run order (None → config.scheduler.phases; unknown → VALIDATION_013).
            phases：按运行顺序的阶段名（None → config.scheduler.phases；未知 → VALIDATION_013）。

        Output / 输出：
            {'anchor_id': <last scheduler anchor>, 'phases': {phase: report}}.
            {'anchor_id': <最后一个调度器锚点>, 'phases': {phase: report}}。

        State effect / 状态影响：
            One scheduler event + anchor per scope and the phases' writes (entity atoms, edges, InterfaceEPs, skills, transitions, compiled states, decay). Only the phases' row changes advance a scope's write version: a run with nothing to do leaves HMS.context()'s memo of every scope reused (§6.7).
            每个范围一个调度器事件 + 锚点，以及各阶段的写入（实体原子、边、InterfaceEP、技能、转移、编译态、衰减）。只有各阶段真正改变的行才推进范围的写入版本：无事可做的运行让每个范围的 HMS.context() 记忆保持复用（§6.7）。
        """
        return self.scheduler.run(scope=self._optional_scope(filters, "consolidate"), phases=phases)

    def purge(self, *, filters: dict, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict:
        """Retention escape hatch (§16.6): report (default) or physically remove long-archived versions nothing needs.
        保留策略逃生舱（§16.6）：报告（默认）或物理删除再无对象需要的长期归档版本。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key (partial scopes sweep every matching run; VALIDATION_001 otherwise).
            filters：至少含一个范围键的 Mem0 过滤字典（部分范围扫过每个匹配的 run；否则 VALIDATION_001）。

            older_than_days: Only versions whose valid_to is before now − older_than_days qualify (VALIDATION_009 if negative).
            older_than_days：只有 valid_to 早于 now − older_than_days 的版本才符合（负数 → VALIDATION_009）。

            dry_run: True (default) returns the plan and byte estimate without writing; False needs config.retention.enabled (ConfigError CONFIG_001 otherwise).
            dry_run：True（默认）只返回计划与字节估算、不写入；False 需要 config.retention.enabled（否则 ConfigError CONFIG_001）。

            keep_witnessed: True (default) never removes anchors or source events; False also drops the purged rows' anchors that nothing cites any more.
            keep_witnessed：True（默认）从不删除锚点与源事件；False 还会删除被清行的、再无任何引用的锚点。

        Output / 输出：
            The ops.retention.purge report: {'dry_run', 'scope', 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable', 'protected', 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}.
            ops.retention.purge 的报告：{'dry_run', 'scope', 'older_than_days', 'cutoff', 'keep_witnessed', 'examined', 'purgeable', 'protected', 'estimated_bytes', 'purged', 'anchors_purged', 'events_purged', 'anchor_id'}。

        State effect / 状态影响：
            dry_run=True: none. dry_run=False: one transaction (purge system event + anchor, PURGE history rows, physical deletes under the trigger gate) and index vector removal; the protections (edge endpoints, checkpoint-live versions, InterfaceEP support, EP references, multi-version heads) are never overridable.
            dry_run=True：无。dry_run=False：一个事务（purge 系统事件 + 锚点、PURGE 历史行、触发器门下的物理删除）与索引向量删除；各项保护（边端点、检查点存活版本、InterfaceEP 支持、EP 引用、多版本头）不可绕过。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="purge")
        if not dry_run and not self.config.retention.enabled:
            raise ConfigError(
                "purge(dry_run=False) requires config.retention.enabled=True (revision-instead-of-deletion is the default)",
                "CONFIG_001",
                details={"retention.enabled": False, "dry_run": False},
                suggestion="HMS({'retention': {'enabled': True}}) or keep dry_run=True to only inspect",
            )
        return ops_purge(
            self.store,
            self.index,
            scope=scope,
            older_than_days=older_than_days,
            dry_run=bool(dry_run),
            keep_witnessed=bool(keep_witnessed),
        )

    def vacuum(self) -> dict:
        """Run SQLite VACUUM on the store (outside any transaction) and report the size before and after.
        对存储执行 SQLite VACUUM（在任何事务之外）并报告前后大小。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'} (page_count * page_size; a file database checkpoints and truncates its WAL first).
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'}（page_count * page_size；文件库先 checkpoint 并截断 WAL）。

        State effect / 状态影响：
            Rewrites the database file; no row changes.
            重写数据库文件；不改变任何行。
        """
        return self.store.vacuum()

    def graph_stats(self, *, filters: dict | None = None) -> dict:
        """Support Graph statistics for a scope (None → whole DB), counting nodes/edges regardless of validity.
        某范围（None → 整库）的支持图统计，节点/边计数不区分有效性。

        Input / 输入：
            filters: Full or partial scope filters, or None.
            filters：完整或部分范围过滤，或 None。

        Output / 输出：
            {'nodes', 'edges', 'valid_edges', 'invalid_edges', 'boundary_edges', 'interface_eps', 'unresolved_conflicts', 'witness_coverage', 'projection_coverage', 'traceable_high_edges', 'transitions'}.
            {'nodes', 'edges', 'valid_edges', 'invalid_edges', 'boundary_edges', 'interface_eps', 'unresolved_conflicts', 'witness_coverage', 'projection_coverage', 'traceable_high_edges', 'transitions'}。

        State effect / 状态影响：
            None.
            无。
        """
        return self.graph.stats(self._optional_scope(filters, "graph_stats"))

    def context(
        self,
        query: str,
        *,
        filters: dict,
        top_k: int | None = None,
        as_of: Any = None,
        budget_tokens: int | None = None,
    ) -> dict:
        """Hybrid read context (§6.7, READER_AB 2026-09-18): the memoized session card of the scope followed by the query's deduplicated, time-sorted, timestamped memory rows; put the text BEFORE the user's question ('Memories:\\n<text>\\n\\n<question>').
        混合读上下文（§6.7，READER_AB 2026-09-18）：范围的记忆化会话卡片，随后是本次查询去重、按时间排序、带时间戳的记忆行；把文本放在用户问题**之前**（'Memories:\\n<text>\\n\\n<question>'）。

        Input / 输入：
            query: The question / goal text (VALIDATION_010 when blank).
            query：问题 / 目标文本（空白时 VALIDATION_010）。

            filters: Mem0 filter dict with >= 1 scope key (VALIDATION_001); a scalar 'visibility' is the audience of both parts.
            filters：至少含一个范围键的 Mem0 过滤字典（否则 VALIDATION_001）；标量 'visibility' 是两部分共同的受众。

            top_k: Search rows before deduplication (None → config.compiler.context_rows_top_k; an explicit 0 reaches search unchanged and the rows part renders '(none)'; VALIDATION_009 when negative).
            top_k：去重前的检索行数（None → config.compiler.context_rows_top_k；显式的 0 原样传给 search、记忆行部分渲染为 '(none)'；为负时 VALIDATION_009）。

            as_of: Validity instant of the ROWS (search reference_date; None → store.now()). The session card is always compiled from the scope's latest extractable event at that event's own timestamp, whatever as_of says (streaming use; the memo is keyed on the scope, not on as_of).
            as_of：记忆行的有效性时刻（search 的 reference_date；None → store.now()）。会话卡片始终以范围内最新可抽取事件、在该事件自身时间戳处编译，与 as_of 无关（面向流式使用；记忆按范围而非 as_of 作键）。

            budget_tokens: Session-card budget (None → config.compiler.context_session_budget_tokens); a different budget recomputes the card.
            budget_tokens：会话卡片预算（None → config.compiler.context_session_budget_tokens）；预算不同则重算卡片。

        Output / 输出：
            {'text', 'session_card', 'session_key', 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids', 'rows', 'rows_text', 'rows_header', 'search_rows', 'scope'} — text = '<session card or (none)>\\n\\n<rows_header>\\n<rows_text or (none)>' (compiler.context.hybrid_text); rows = the kept search rows in time order; search_rows = the raw rows in rank order; session_key = the event_id the card was compiled from (None while the scope holds no extractable event).
            {'text', 'session_card', 'session_key', 'session_as_of', 'session_query', 'session_reused', 'session_memory_ids', 'rows', 'rows_text', 'rows_header', 'search_rows', 'scope'}——text = '<会话卡片或 (none)>\\n\\n<rows_header>\\n<rows_text 或 (none)>'（compiler.context.hybrid_text）；rows = 按时间排序的保留检索行；search_rows = 按排名的原始行；session_key = 卡片所据事件的 event_id（范围内无可抽取事件时为 None）。

        State effect / 状态影响：
            None in the store (one search, at most one compile); updates the in-process session-card memo, which every committed transaction that changed versioned rows of the scope (add / update / delete / delete_all / feedback / rollback / a consolidate that created or revised something / reset …) invalidates through store.scope_version(); a checkpoint, an idle consolidate, the daemon's pass over a scope with nothing to do and a transaction that changed no row leave it reused (§6.7).
            存储无变化（一次 search，至多一次 compile）；更新进程内的会话卡片记忆，每个改变了该范围带版本行的已提交事务（add / update / delete / delete_all / feedback / rollback / 创建或修订了内容的 consolidate / reset……）都经 store.scope_version() 使其失效；检查点、空转的 consolidate、守护进程对无事可做范围的扫描与没有改变任何行的事务都保持复用（§6.7）。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="context")
        data = dict(filters or {})
        audience = data.get("visibility") if isinstance(data.get("visibility"), str) else None
        cfg = self.config.compiler
        k = int(cfg.context_rows_top_k) if top_k is None else top_k
        budget = int(cfg.context_session_budget_tokens) if budget_tokens is None else budget_tokens
        card = self._session_card(scope, audience=audience, budget=budget)
        search_kwargs: dict[str, Any] = {"filters": data, "top_k": k}
        if as_of is not None:
            search_kwargs["reference_date"] = as_of
        raw = self.search(query, **search_kwargs)["results"]
        kept = sort_rows_by_time(dedup_rows(raw))
        rows_block = render_rows(kept, timestamps=bool(cfg.context_timestamps))
        header = str(cfg.context_rows_header)
        return {
            "text": hybrid_text(card["text"], rows_block, header=header),
            "session_card": card["text"],
            "session_key": card["event_id"],
            "session_as_of": card["as_of"],
            "session_query": card["query"],
            "session_reused": card["reused"],
            "session_memory_ids": list(card["memory_ids"]),
            "rows": kept,
            "rows_text": rows_block,
            "rows_header": header,
            "search_rows": raw,
            "scope": scope.key(),
        }

    def _session_card(self, scope: Scope, *, audience: str | None, budget: int) -> dict[str, Any]:
        """The memoized session card of a scope: compile(latest extractable event's text, as_of=its timestamp, budget) reused verbatim while store.scope_version(scope) and the budget are unchanged.
        范围的记忆化会话卡片：compile(最新可抽取事件的文本, as_of=其时间戳, 预算)，在 store.scope_version(scope) 与预算不变期间原样复用。
        """
        key = (scope.key(), audience)
        # The version is captured BEFORE compiling: a write racing the compile moves the store past it, so the
        # entry stored below is not reused next time. The compile itself runs outside the lock.
        # 版本在编译**之前**捕获：与编译竞争的写入会把存储推过它，因此下面存入的条目下次不会被复用。编译本身在锁外进行。
        version = self.store.scope_version(scope)
        with self._context_lock:
            hit = self._context_memo.get(key)
            if hit is not None and hit["version"] == version and hit["budget"] == budget:
                return dict(hit, reused=True)
        ev = self.store.latest_event(scope, object_types=list(EXTRACTABLE_OBJECT_TYPES))
        if ev is None:
            entry: dict[str, Any] = {"text": "", "event_id": None, "as_of": None, "query": None, "memory_ids": []}
        else:
            text = ev.payload.get("text")
            question = text if isinstance(text, str) and text.strip() else canonical_json(ev.payload)
            card_filters = {**scope.as_filters(), **({"visibility": audience} if audience else {})}
            ir = self.compile(question, filters=card_filters, as_of=ev.timestamp, budget_tokens=budget)
            entry = {
                "text": ir.to_text(),
                "event_id": ev.event_id,
                "as_of": ev.timestamp,
                "query": question,
                "memory_ids": [it.memory_id for it in ir.items() if it.memory_id],
            }
        entry.update(version=version, budget=budget, reused=False)
        with self._context_lock:
            self._context_memo[key] = entry
        return dict(entry)

    # ------------------------------------------------------------------ telemetry (§2 tail: usage / trace)
    # ------------------------------------------------------------------ 遥测（§2 尾部：usage / trace）
    def usage(self) -> telemetry.Usage:
        """Merged llm + embedder usage of this facade (telemetry.usage_of).
        本门面 llm + 嵌入器的合并用量（telemetry.usage_of）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A telemetry.Usage snapshot (the meters are read, never reset).
            telemetry.Usage 快照（只读取计量器，从不重置）。

        State effect / 状态影响：
            None.
            无。
        """
        return telemetry.usage_of(self)

    def trace(self, operation: str, request_id: str | None = None) -> AbstractContextManager[telemetry.RequestTrace]:
        """telemetry.trace(operation, request_id): a RequestTrace that provider calls inside the block append stages to.
        telemetry.trace(operation, request_id)：一个 RequestTrace，块内的提供者调用向其追加阶段。

        Input / 输入：
            operation: Name of the traced operation (e.g. 'add', 'search').
            operation：被追踪操作的名称（如 'add'、'search'）。

            request_id: Caller-supplied id; a uuid4 hex when None.
            request_id：调用方给定的 id；None 时为 uuid4 十六进制串。

        Output / 输出：
            The context manager from telemetry.trace (enter it with `with`).
            telemetry.trace 返回的上下文管理器（用 `with` 进入）。

        State effect / 状态影响：
            Sets the context-local active trace for the block and restores the previous one on exit.
            在块内设置上下文局部的活动追踪，退出时恢复之前的追踪。
        """
        return telemetry.trace(operation, request_id)

    # ------------------------------------------------------------------ background daemon (HMS_arch §3.9)
    # ------------------------------------------------------------------ 后台守护进程（HMS_arch §3.9）
    def start_daemon(
        self,
        *,
        interval_s: float | None = None,
        phases: list[str] | None = None,
        filters: dict | None = None,
    ) -> StateDaemon:
        """Start (or return the running) StateDaemon that calls consolidate() every interval on a background thread.
        启动（或返回正在运行的）StateDaemon，它在后台线程上每个间隔调用一次 consolidate()。

        Input / 输入：
            interval_s: Seconds between runs (None → config.scheduler.daemon_interval_s; both None → VALIDATION_009).
            interval_s：两次运行之间的秒数（None → config.scheduler.daemon_interval_s；两者皆 None → VALIDATION_009）。

            phases: Phase names in run order (None → orient/collect/consolidate/prune; unknown → VALIDATION_013).
            phases：按运行顺序的阶段名（None → orient/collect/consolidate/prune；未知 → VALIDATION_013）。

            filters: Full or partial scope filters (None → every scope in the store).
            filters：完整或部分范围过滤（None → 存储中的每个范围）。

        Output / 输出：
            The StateDaemon (already running); a daemon that is still running is returned as is.
            StateDaemon（已在运行）；仍在运行的守护进程原样返回。

        State effect / 状态影响：
            Spawns the 'hms-daemon' thread; every store write happens inside consolidate().
            派生 'hms-daemon' 线程；所有存储写入都发生在 consolidate() 内。
        """
        if self._daemon is not None and self._daemon.is_running:
            return self._daemon
        interval = interval_s if interval_s is not None else self.config.scheduler.daemon_interval_s
        if interval is None:
            raise ValidationError(
                "start_daemon() needs interval_s or config.scheduler.daemon_interval_s",
                "VALIDATION_009",
                details={"interval_s": None, "daemon_interval_s": None},
                suggestion="pass interval_s=<seconds> or set scheduler.daemon_interval_s in the config",
            )
        daemon = StateDaemon(self, interval_s=interval, phases=phases, filters=filters)
        daemon.start()
        self._daemon = daemon
        return daemon

    def stop_daemon(self, timeout: float | None = None) -> bool:
        """Stop the daemon started by start_daemon() and wait for its thread (idempotent; True when none is running).
        停止 start_daemon() 启动的守护进程并等待其线程（幂等；没有运行中的守护进程时也返回 True）。

        Input / 输入：
            timeout: Max seconds to wait for the thread (None → until it exits).
            timeout：等待线程的最长秒数（None → 直到退出）。

        Output / 输出：
            True when no daemon thread is alive afterwards.
            之后没有存活的守护线程时为 True。

        State effect / 状态影响：
            Sets the daemon's stop event; no store write.
            设置守护进程的停止事件；不写存储。
        """
        daemon = self._daemon
        if daemon is None:
            return True
        stopped = daemon.stop(timeout)
        if stopped:
            self._daemon = None
        return stopped

    # ------------------------------------------------------------------ §11 state surface (StateAPI delegates)
    # ------------------------------------------------------------------ §11 状态表面（StateAPI 委托）
    @property
    def state(self) -> StateAPI:
        """The StateAPI bound to this facade (built on first access).
        绑定到本门面的 StateAPI（首次访问时构造）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            StateAPI.
            StateAPI。

        State effect / 状态影响：
            Caches the instance on the facade.
            把实例缓存在门面上。
        """
        if self._state is None:
            self._state = StateAPI(self)
        return self._state

    def work_state(
        self, query: str, *, filters: dict, mode: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[WorkState, str]:
        """StateAPI.work_state: the Work State Card (§11.1).
        StateAPI.work_state：工作状态卡（§11.1）。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            mode: Compile mode (None → config default).
            mode：编译模式（None → 配置默认值）。

            as_of: Reference instant.
            as_of：参考时刻。

            budget_tokens: Token budget.
            budget_tokens：token 预算。

        Output / 输出：
            (WorkState, card text).
            (WorkState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.work_state(query, filters=filters, mode=mode, as_of=as_of, budget_tokens=budget_tokens)

    def handoff_package(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[HandoffPackageState, str]:
        """StateAPI.handoff_package: the Handoff Package (§11.1).
        StateAPI.handoff_package：交接包（§11.1）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Handoff goal (None → newest valid EP's u).
            goal：交接目标（None → 最新有效 EP 的 u）。

            checkpoint: Take a checkpoint first.
            checkpoint：先建立检查点。

            budget_tokens: Token budget.
            budget_tokens：token 预算。

        Output / 输出：
            (HandoffPackageState, package text).
            (HandoffPackageState, 交接包文本)。

        State effect / 状态影响：
            Writes the checkpoint when `checkpoint` is True.
            `checkpoint` 为 True 时写入检查点。
        """
        return self.state.handoff_package(
            filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens
        )

    def detective_report(
        self, target: str, *, filters: dict | None = None, hops: int = 2, as_of: Any = None
    ) -> tuple[DetectiveReport, str]:
        """StateAPI.detective_report: the Detective Report (§11.1).
        StateAPI.detective_report：侦探报告（§11.1）。

        Input / 输入：
            target: Object id or question text.
            target：对象 id 或问题文本。

            filters: Scope filters enabling question mode.
            filters：启用问题模式的范围过滤。

            hops: why() expansion levels.
            hops：why() 扩展层数。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (DetectiveReport, report text).
            (DetectiveReport, 报告文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.detective_report(target, filters=filters, hops=hops, as_of=as_of)

    def rollback_point(
        self, *, filters: dict, label: str | None = None, create: bool = False
    ) -> tuple[RollbackPoint | None, str]:
        """StateAPI.rollback_point: the Rollback Point (§11.1).
        StateAPI.rollback_point：回滚点（§11.1）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            label: Checkpoint label when creating.
            label：创建时的检查点标签。

            create: Take a checkpoint now.
            create：现在建立检查点。

        Output / 输出：
            (RollbackPoint or None, card text or '(none)').
            (RollbackPoint 或 None, 卡片文本或 '(none)')。

        State effect / 状态影响：
            Writes a checkpoint when create is True.
            create 为 True 时写入检查点。
        """
        return self.state.rollback_point(filters=filters, label=label, create=create)

    def personal_state(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str | None = None,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """StateAPI.personal_state: PersonalState + Personal Card (§11.2).
        StateAPI.personal_state：PersonalState 与个人卡（§11.2）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            subject: 'user' | 'assistant' | 'project' | entity text.
            subject：'user' | 'assistant' | 'project' | 实体文本。

            audience: Visibility floor (None → no gate).
            audience：可见性下限（None → 不门控）。

            as_of: Reference instant.
            as_of：参考时刻。

            timeline_limit: Newest N timeline entries.
            timeline_limit：时间线最新的 N 条。

        Output / 输出：
            (PersonalState, card text).
            (PersonalState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.personal_state(
            filters=filters, subject=subject, audience=audience, as_of=as_of, timeline_limit=timeline_limit
        )

    def shared_card(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str = Visibility.team,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """StateAPI.shared_card: the Shared Card (§11.2).
        StateAPI.shared_card：共享卡 Shared Card（§11.2）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            subject: Subject selector.
            subject：主体选择器。

            audience: Visibility floor.
            audience：可见性下限。

            as_of: Reference instant.
            as_of：参考时刻。

            timeline_limit: Newest N timeline entries.
            timeline_limit：时间线最新的 N 条。

        Output / 输出：
            (PersonalState with kind 'shared_card', card text).
            (kind 为 'shared_card' 的 PersonalState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.shared_card(
            filters=filters, subject=subject, audience=audience, as_of=as_of, timeline_limit=timeline_limit
        )

    def edge_state(
        self, a: str | dict, b: str | dict, *, filters: dict | None = None, as_of: Any = None
    ) -> tuple[EdgeState, str]:
        """StateAPI.edge_state: EdgeState + Edge Card (§11.2).
        StateAPI.edge_state：EdgeState 与关系卡（§11.2）。

        Input / 输入：
            a: Entity text or scope dict.
            a：实体文本或范围字典。

            b: Entity text or scope dict (same form as `a`).
            b：实体文本或范围字典（与 `a` 同形）。

            filters: Scope filters (entity mode).
            filters：范围过滤（实体模式）。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (EdgeState, card text).
            (EdgeState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.edge_state(a, b, filters=filters, as_of=as_of)

    def group_state(self, *, members: list[dict], as_of: Any = None) -> tuple[GroupState, str]:
        """StateAPI.group_state: GroupState + Group Card (§11.2).
        StateAPI.group_state：GroupState 与群体卡（§11.2）。

        Input / 输入：
            members: Scope dicts, one per member.
            members：范围字典，每个成员一个。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (GroupState, card text).
            (GroupState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.group_state(members=members, as_of=as_of)

    def customer_issue_state(
        self, *, filters: dict, issue: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[CustomerIssueState, str]:
        """StateAPI.customer_issue_state: CustomerIssueState + card (§11.3).
        StateAPI.customer_issue_state：CustomerIssueState + 卡片（§11.3）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            issue: Issue text (None → newest EP goal).
            issue：问题文本（None → 最新 EP 的目标）。

            as_of: Reference instant.
            as_of：参考时刻。

            budget_tokens: Compile budget.
            budget_tokens：编译预算。

        Output / 输出：
            (CustomerIssueState, card text).
            (CustomerIssueState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.customer_issue_state(filters=filters, issue=issue, as_of=as_of, budget_tokens=budget_tokens)

    def incident_state(
        self, *, filters: dict, question: str | None = None, as_of: Any = None, hops: int = 2
    ) -> tuple[IncidentState, str]:
        """StateAPI.incident_state: IncidentState + card (§11.3).
        StateAPI.incident_state：IncidentState + 卡片（§11.3）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            question: Detective question (None → newest negative EP).
            question：侦探式问题（None → 最新负回报 EP）。

            as_of: Reference instant.
            as_of：参考时刻。

            hops: why() expansion levels.
            hops：why() 扩展层数。

        Output / 输出：
            (IncidentState, card text).
            (IncidentState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.incident_state(filters=filters, question=question, as_of=as_of, hops=hops)

    def meeting_decision_state(self, *, filters: dict, as_of: Any = None) -> tuple[MeetingDecisionState, str]:
        """StateAPI.meeting_decision_state: MeetingDecisionState + card (§11.3).
        StateAPI.meeting_decision_state：MeetingDecisionState + 卡片（§11.3）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            as_of: Reference instant.
            as_of：参考时刻。

        Output / 输出：
            (MeetingDecisionState, card text).
            (MeetingDecisionState, 卡片文本)。

        State effect / 状态影响：
            None.
            无。
        """
        return self.state.meeting_decision_state(filters=filters, as_of=as_of)

    def account_handoff_state(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[AccountHandoffState, str]:
        """StateAPI.account_handoff_state: AccountHandoffState + card (§11.3).
        StateAPI.account_handoff_state：AccountHandoffState + 卡片（§11.3）。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Handoff goal.
            goal：交接目标。

            checkpoint: Take a checkpoint first.
            checkpoint：先建立检查点。

            budget_tokens: Compile budget.
            budget_tokens：编译预算。

        Output / 输出：
            (AccountHandoffState, card text).
            (AccountHandoffState, 卡片文本)。

        State effect / 状态影响：
            Writes the checkpoint when `checkpoint` is True.
            `checkpoint` 为 True 时写入检查点。
        """
        return self.state.account_handoff_state(
            filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens
        )

    # ------------------------------------------------------------------ private helpers
    # ------------------------------------------------------------------ 私有辅助
    def _optional_scope(self, filters: dict | None, context: str) -> Scope | None:
        """Scope from an optional filter dict; None (or no scope keys) → None.
        由可选过滤字典得到范围；None（或无范围键）→ None。
        """
        if not filters:
            return None
        scope, _expr = parse_filters(filters, require_scope=False, context=context)
        return None if scope.is_empty() else scope

    def _mode_value(self, mode: Any) -> str:
        """CompileMode value from the argument or the config default (VALIDATION_013 when unknown).
        由参数或配置默认值得到 CompileMode 值（未知时 VALIDATION_013）。
        """
        cfg_mode = self.config.compiler.default_mode
        value = str(getattr(mode, "value", mode) if mode is not None else getattr(cfg_mode, "value", cfg_mode))
        if value not in CompileMode.values():
            raise ValidationError(
                f"unknown compile mode {value!r}",
                "VALIDATION_013",
                details={"mode": value, "supported": CompileMode.values()},
            )
        return value

    def _require_head(self, memory_id: Any) -> MemoryAtom:
        """The live head of a lineage or NOT_FOUND_001 (a ValueError) with Mem0's message.
        谱系的存活头版本，否则抛出带 Mem0 消息的 NOT_FOUND_001（属 ValueError）。
        """
        head = self.store.resolve_head(str(memory_id).strip()) if memory_id is not None else None
        if head is None:
            raise ValidationError(
                f"Memory with id {memory_id} not found", "NOT_FOUND_001", details={"memory_id": memory_id}
            )
        return head

    def _archive_deleted(self, atom: MemoryAtom, anchor_id: str, now: str) -> None:
        """archive_atom(reason='deleted') plus the DELETE history row (is_deleted=1) witnessed by the system anchor.
        archive_atom(reason='deleted') 并写入由系统锚点见证的 DELETE 历史行（is_deleted=1）。
        """
        self.store.archive_atom(atom.atom_id, reason="deleted", at=now)
        self.store.add_history(
            memory_id=atom.memory_id,
            old_memory=atom.text,
            new_memory=None,
            event=HistoryEvent.DELETE.value,
            created_at=atom.created_at or now,
            updated_at=now,
            is_deleted=1,
            actor_id=atom.actor_id,
            role=atom.role if atom.atom_type == "message" else None,
            anchor_id=anchor_id,
            atom_id=atom.atom_id,
        )

    @staticmethod
    def _successor(
        head: MemoryAtom, text: str, user_meta: dict, *, anchor_id: str, now: str, valid_to: str | None
    ) -> MemoryAtom:
        """The update() successor: same coordinates, new text/hash, entities re-linked, [sys_anchor] + old anchors.
        update() 的后继版本：坐标相同、新文本/哈希、实体重新链接、[系统锚点] + 旧锚点。
        """
        content = dict(head.content) if isinstance(head.content, dict) else {}
        content["text"] = text
        data = head.model_dump()
        data.update(
            {
                "atom_id": "",
                "memory_id": "",
                "content": content,
                "text": text,
                "hash": "",
                "valid_from": now,
                "valid_to": valid_to,
                "entity_ids": [],
                "user_meta": user_meta,
                "anchor_ids": ordered_union([anchor_id], list(head.anchor_ids)),
                "archived_reason": None,
                "created_at": None,
                "updated_at": None,
            }
        )
        return MemoryAtom(**data)

    def _target_from_question(self, question: str, filters: dict) -> dict | None:
        """why() question mode: a detective compile; prefer an item whose entity keys hit the question's identifiers.
        why() 的问题模式：侦探式编译；优先选择实体键命中问题标识符的项。
        """
        ir = self.compile(question, filters=filters, mode=CompileMode.deep.value, budget_tokens=600)
        candidates = [
            item for item in ir.constraints + ir.episodes + ir.facts if item.atom_id and item.item_id == item.atom_id
        ]
        keys = {entity_key(token) for token in identifier_tokens(question)}
        chosen: MemoryAtom | None = None
        if keys:
            for item in candidates:
                atom = self.store.get_atom(str(item.atom_id))
                if atom is not None and keys.intersection(atom.entity_ids):
                    chosen = atom
                    break
        if chosen is None and candidates:
            chosen = self.store.get_atom(str(candidates[0].atom_id))
        if chosen is None:
            return None
        return {
            "kind": "atom",
            "id": chosen.atom_id,
            "node_id": chosen.atom_id,
            "text": chosen.text,
            "resolved_from": "query",
        }


# Apply the text scope to every method in TEXT_SCOPED_METHODS (one place, asserted by tests/test_text_cjk.py).
# 把文本作用域套到 TEXT_SCOPED_METHODS 中的每个方法上（只此一处，由 tests/test_text_cjk.py 断言）。
for _name in TEXT_SCOPED_METHODS:
    setattr(HMS, _name, text_scoped(getattr(HMS, _name)))
del _name

__all__ = ["HMS", "PROCEDURAL_MEMORY", "TEXT_CJK_META_KEY", "TEXT_SCOPED_METHODS", "TRANSITION_FORMATS", "TRANSITION_SCALES"]
