"""AsyncHMS: the asyncio twin of the HMS facade, one awaitable per public method.
AsyncHMS：HMS 门面的 asyncio 孪生，每个公共方法对应一个可等待对象。
"""

# Purpose (EN): Mirror every public HMS method as `async def` with the identical signature, running the synchronous
#               facade in the default executor through asyncio.to_thread so the event loop never blocks.
# 目的（中）：把 HMS 的每个公共方法镜像为签名完全相同的 `async def`，通过 asyncio.to_thread 在默认执行器里
#            运行同步门面，使事件循环永不阻塞。

# Input (EN): The same HMSConfig / Mem0-shaped dict / injected StateStore as HMS, and the public call arguments.
# 输入（中）：与 HMS 相同的 HMSConfig / Mem0 形状字典 / 注入的 StateStore，以及公共调用参数。

# Output (EN): Awaitables resolving to exactly what the synchronous method returns (same envelopes, same objects).
# 输出（中）：可等待对象，其结果与同步方法的返回值完全一致（相同信封、相同对象）。

# State effect (EN): None of its own; every write is the wrapped HMS write under the StateStore re-entrant lock.
#                    usage() / trace() are the two synchronous extras: a meter snapshot and a contextvar-driven trace
#                    that must be entered in the calling task (never inside to_thread) so awaited calls inherit it.
# 状态影响（中）：自身无状态影响；所有写入都是被封装的 HMS 写入，受 StateStore 可重入锁保护。
#                usage() / trace() 是两个同步附加项：计量快照与由 contextvar 驱动的追踪，后者必须在调用任务里
#                进入（绝不在 to_thread 内），这样被 await 的调用才能继承它。

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from typing import Any, Callable, TypeVar

from hms_elite import telemetry
from hms_elite.configs.base import HMSConfig
from hms_elite.daemon import StateDaemon
from hms_elite.enums import FeedbackType, Visibility
from hms_elite.memory.main import HMS, _OSSProject
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import _UNSET
from hms_elite.models import AxisQuery, AxisResult, ContextIR, SourceEvent, StatePackage
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

T = TypeVar("T")


class AsyncHMS:
    """Awaitable HMS: the 11 Mem0 methods, the 18 HMS-native methods and the 12 §11 state cards, each off the loop.
    可等待的 HMS：11 个 Mem0 方法、18 个 HMS 原生方法与 12 个 §11 状态卡，每个都在事件循环之外执行。
    """

    def __init__(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None:
        """Build the wrapped synchronous HMS (blocking: opens the store); use `await AsyncHMS.create()` to avoid that.
        构建被封装的同步 HMS（阻塞：会打开存储）；若要避免阻塞请用 `await AsyncHMS.create()`。

        Input / 输入：
            config: HMSConfig, a Mem0-shaped dict (HMSConfig.from_dict) or None for the offline defaults.
            config：HMSConfig、Mem0 形状的字典（HMSConfig.from_dict）或 None（离线默认值）。

            store: Optional StateStore (tests inject a frozen clock and sequential ids through it).
            store：可选的 StateStore（测试通过它注入冻结时钟与顺序 id）。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            Opens (or creates) the SQLite store when none is injected, exactly as HMS.__init__ does.
            未注入时打开（或创建）SQLite 存储，与 HMS.__init__ 完全一致。
        """
        self._sync: HMS = HMS(config, store=store)

    # ------------------------------------------------------------------ construction helpers
    # ------------------------------------------------------------------ 构造辅助
    @classmethod
    def _wrap(cls, sync: HMS) -> "AsyncHMS":
        """Wrap an already-built HMS without constructing a second one.
        封装一个已构建的 HMS，而不再构造第二个。
        """
        inst = cls.__new__(cls)
        inst._sync = sync
        return inst

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "AsyncHMS":
        """Mem0 parity constructor: HMS.from_config on the dict, then wrap the result.
        与 Mem0 一致的构造入口：对字典调用 HMS.from_config，再封装其结果。

        Input / 输入：
            config_dict: Mem0-shaped or HMS-shaped config dict (§5.1).
            config_dict：Mem0 形状或 HMS 形状的配置字典（§5.1）。

        Output / 输出：
            An AsyncHMS instance; ConfigError propagates exactly as from HMS.from_config.
            AsyncHMS 实例；ConfigError 与 HMS.from_config 一样向上传播。

        State effect / 状态影响：
            Opens (or creates) the configured store (blocking, like HMS.from_config).
            打开（或创建）所配置的存储（阻塞，与 HMS.from_config 相同）。
        """
        return cls._wrap(HMS.from_config(config_dict))

    @classmethod
    async def create(cls, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> "AsyncHMS":
        """Non-blocking constructor: build the synchronous HMS in a worker thread, then wrap it.
        非阻塞构造：在工作线程中构建同步 HMS，再封装它。

        Input / 输入：
            config: Same as AsyncHMS.__init__.
            config：同 AsyncHMS.__init__。

            store: Same as AsyncHMS.__init__.
            store：同 AsyncHMS.__init__。

        Output / 输出：
            An AsyncHMS instance.
            AsyncHMS 实例。

        State effect / 状态影响：
            Opens (or creates) the store off the event loop.
            在事件循环之外打开（或创建）存储。
        """
        return cls._wrap(await asyncio.to_thread(HMS, config, store=store))

    async def _call(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Run one synchronous facade call in the default executor and return its result.
        在默认执行器中运行一次同步门面调用并返回其结果。
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    # ------------------------------------------------------------------ async context manager
    # ------------------------------------------------------------------ 异步上下文管理器
    async def __aenter__(self) -> "AsyncHMS":
        """Async context-manager entry (returns self).
        异步上下文管理器入口（返回自身）。

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

    async def __aexit__(self, *exc: Any) -> None:
        """Async context-manager exit: close the store.
        异步上下文管理器退出：关闭存储。

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
        await self.aclose()

    async def aclose(self) -> None:
        """Close the wrapped store off the event loop (idempotent; the async-idiomatic name for close()).
        在事件循环之外关闭被封装的存储（幂等；close() 的异步惯用名）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Releases the SQLite handle.
            释放 SQLite 句柄。
        """
        await self._call(self._sync.close)

    # ------------------------------------------------------------------ attribute pass-through
    # ------------------------------------------------------------------ 属性透传
    @property
    def sync(self) -> HMS:
        """The wrapped synchronous HMS (for callers that need a blocking handle).
        被封装的同步 HMS（供需要阻塞句柄的调用方使用）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The HMS instance.
            HMS 实例。

        State effect / 状态影响：
            None.
            无。
        """
        return self._sync

    @property
    def config(self) -> HMSConfig:
        """The effective HMSConfig of the wrapped HMS.
        被封装 HMS 的生效 HMSConfig。

        Input / 输入：
            none.
            无。

        Output / 输出：
            HMSConfig.
            HMSConfig。

        State effect / 状态影响：
            None.
            无。
        """
        return self._sync.config

    @property
    def store(self) -> StateStore:
        """The StateStore of the wrapped HMS.
        被封装 HMS 的 StateStore。

        Input / 输入：
            none.
            无。

        Output / 输出：
            StateStore.
            StateStore。

        State effect / 状态影响：
            None.
            无。
        """
        return self._sync.store

    @property
    def project(self) -> _OSSProject:
        """Mem0 `project` handle of the wrapped HMS (update() raises ValueError, OSS parity).
        被封装 HMS 的 Mem0 `project` 句柄（update() 抛出 ValueError，与 OSS 一致）。

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
        return self._sync.project

    # ------------------------------------------------------------------ Mem0-compatible surface
    # ------------------------------------------------------------------ Mem0 兼容接口
    async def chat(self, query: str) -> None:
        """Awaitable HMS.chat(): Mem0 parity stub that raises NotImplementedError.
        可等待的 HMS.chat()：与 Mem0 一致的占位，抛出 NotImplementedError。

        Input / 输入：
            query: Ignored.
            query：忽略。

        Output / 输出：
            Never resolves (NotImplementedError).
            从不完成（NotImplementedError）。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(self._sync.chat, query)

    async def add(
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
        """Awaitable HMS.add(): anchor the messages, then extract (infer=True) or store raw message atoms.
        可等待的 HMS.add()：为消息落锚，然后抽取（infer=True）或存原始消息原子。

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
            The HMS.add() envelope {'results': [...]}.
            HMS.add() 的信封 {'results': [...]}。

        State effect / 状态影响：
            Same as HMS.add() (Txn A + Txn B), executed in a worker thread.
            与 HMS.add() 相同（事务 A + 事务 B），在工作线程中执行。
        """
        return await self._call(
            self._sync.add,
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            timestamp=timestamp,
            expiration_date=expiration_date,
            infer=infer,
            memory_type=memory_type,
            prompt=prompt,
        )

    async def get(self, memory_id: str) -> dict | None:
        """Awaitable HMS.get(): the head version of a lineage as a row, or None.
        可等待的 HMS.get()：谱系头版本的行，或 None。

        Input / 输入：
            memory_id: Mem0 memory id (lineage id).
            memory_id：Mem0 记忆 id（谱系 id）。

        Output / 输出：
            The public row dict, or None when unknown/deleted.
            公共行字典；未知/已删除时为 None。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(self._sync.get, memory_id)

    async def get_all(
        self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs
    ) -> dict:
        """Awaitable HMS.get_all(): axis-only listing in time order.
        可等待的 HMS.get_all()：仅走轴的按时间顺序列表。

        Input / 输入：
            filters: Mem0 filter dict (scope keys plus §13.2 operators).
            filters：Mem0 过滤字典（范围键加 §13.2 运算符）。

            top_k: Maximum rows (VALIDATION_009 when < 1).
            top_k：最大行数（< 1 时 VALIDATION_009）。

            show_expired: Include rows whose valid_to has passed.
            show_expired：是否包含 valid_to 已过的行。

            kwargs: Extra Mem0 keyword arguments, forwarded unchanged.
            kwargs：额外的 Mem0 关键字参数，原样转发。

        Output / 输出：
            {'results': [row, ...]}.
            {'results': [行, ...]}。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(self._sync.get_all, filters=filters, top_k=top_k, show_expired=show_expired, **kwargs)

    async def search(
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
        """Awaitable HMS.search(): planned multi-signal retrieval over the State Index.
        可等待的 HMS.search()：基于状态索引的规划式多信号检索。

        Input / 输入：
            query: Query text (VALIDATION_010 when empty).
            query：查询文本（为空时 VALIDATION_010）。

            top_k: Maximum hits (VALIDATION_009 when < 1).
            top_k：最大命中数（< 1 时 VALIDATION_009）。

            filters: Mem0 filter dict.
            filters：Mem0 过滤字典。

            threshold: Minimum fused score (VALIDATION_009 outside [0, 1]).
            threshold：最小融合分数（超出 [0, 1] 时 VALIDATION_009）。

            rerank: Re-order the fused rows with the configured reranker (default: offline overlap).
            rerank：用所配置的重排器重排融合后的行（默认：离线覆盖率重排器）。

            explain: Attach score details to each row.
            explain：为每行附加分数明细。

            reference_date: Reference instant for temporal signals.
            reference_date：时间信号的参考时刻。

            show_expired: Include rows whose valid_to has passed.
            show_expired：是否包含 valid_to 已过的行。

            kwargs: Extra Mem0 keyword arguments, forwarded unchanged.
            kwargs：额外的 Mem0 关键字参数，原样转发。

        Output / 输出：
            {'results': [row + 'score' (+ 'score_details'), ...]}.
            {'results': [行 + 'score'（+ 'score_details'）, ...]}。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(
            self._sync.search,
            query,
            top_k=top_k,
            filters=filters,
            threshold=threshold,
            rerank=rerank,
            explain=explain,
            reference_date=reference_date,
            show_expired=show_expired,
            **kwargs,
        )

    async def update(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict | None = None,
        expiration_date: Any = _UNSET,
        data: str | None = None,
    ) -> dict:
        """Awaitable HMS.update(): a new text is a new lineage version; metadata/expiration edit the head in place.
        可等待的 HMS.update()：新文本产生新的谱系版本；元数据/过期日期就地修改头版本。

        Input / 输入：
            memory_id: Mem0 memory id (unknown → NOT_FOUND_001).
            memory_id：Mem0 记忆 id（未知 → NOT_FOUND_001）。

            text: New memory text (creates a new version).
            text：新的记忆文本（产生新版本）。

            metadata: Replacement caller metadata.
            metadata：替换的调用方元数据。

            expiration_date: New expiration date; the _UNSET sentinel leaves it untouched.
            expiration_date：新的过期日期；_UNSET 哨兵表示保持不变。

            data: Mem0 legacy alias of text.
            data：Mem0 旧版的 text 别名。

        Output / 输出：
            {'message': 'Memory updated successfully!'}.
            {'message': 'Memory updated successfully!'}。

        State effect / 状态影响：
            Same as HMS.update(), executed in a worker thread.
            与 HMS.update() 相同，在工作线程中执行。
        """
        return await self._call(
            self._sync.update, memory_id, text=text, metadata=metadata, expiration_date=expiration_date, data=data
        )

    async def delete(self, memory_id: str) -> dict:
        """Awaitable HMS.delete(): archive the head (reason 'deleted'); content, edges and history are kept.
        可等待的 HMS.delete()：归档头版本（原因 'deleted'）；内容、边与历史均保留。

        Input / 输入：
            memory_id: Mem0 memory id (unknown/deleted → NOT_FOUND_001, a ValueError).
            memory_id：Mem0 记忆 id（未知/已删除 → NOT_FOUND_001，属 ValueError）。

        Output / 输出：
            {'message': 'Memory deleted successfully!'}.
            {'message': 'Memory deleted successfully!'}。

        State effect / 状态影响：
            Same as HMS.delete() (archive, never a physical delete), executed in a worker thread.
            与 HMS.delete() 相同（归档，绝不物理删除），在工作线程中执行。
        """
        return await self._call(self._sync.delete, memory_id)

    async def delete_all(
        self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict:
        """Awaitable HMS.delete_all(): archive every live head in the scope (reason 'deleted').
        可等待的 HMS.delete_all()：归档范围内每个存活头版本（原因 'deleted'）。

        Input / 输入：
            user_id: Mem0 user id filter (at least one id is required).
            user_id：Mem0 用户 id 过滤（至少需要一个 id）。

            agent_id: Mem0 agent id filter.
            agent_id：Mem0 代理 id 过滤。

            run_id: Mem0 run id filter.
            run_id：Mem0 运行 id 过滤。

        Output / 输出：
            {'message': 'Memories deleted successfully!'}.
            {'message': 'Memories deleted successfully!'}。

        State effect / 状态影响：
            Same as HMS.delete_all(), executed in a worker thread.
            与 HMS.delete_all() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.delete_all, user_id, agent_id, run_id)

    async def history(self, memory_id: str) -> list[dict]:
        """Awaitable HMS.history(): the lineage's history rows in seq order.
        可等待的 HMS.history()：谱系的历史行，按 seq 顺序。

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
        return await self._call(self._sync.history, memory_id)

    async def reset(self) -> None:
        """Awaitable HMS.reset(): drop every table, re-create the schema and clear the index.
        可等待的 HMS.reset()：删除全部表、重建模式并清空索引。

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
        return await self._call(self._sync.reset)

    async def close(self) -> None:
        """Awaitable HMS.close(): close the index (when it has a close hook) and the store (idempotent).
        可等待的 HMS.close()：关闭索引（其带 close 钩子时）与存储（幂等）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Releases the SQLite handle and, for index providers holding an external client (milvus), that client.
            释放 SQLite 句柄；索引提供者持有外部客户端时（milvus）同时释放该客户端。
        """
        return await self._call(self._sync.close)

    # ------------------------------------------------------------------ HMS-native surface
    # ------------------------------------------------------------------ HMS 原生接口
    async def add_events(
        self,
        events: list[SourceEvent | dict],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        infer: bool = True,
    ) -> dict:
        """Awaitable HMS.add_events(): ingest source events through the closure builder.
        可等待的 HMS.add_events()：通过闭合构建器摄取源事件。

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
            The HMS.add_events() report ({'results', 'events', 'eps', 'edges', ...}).
            HMS.add_events() 的报告（{'results', 'events', 'eps', 'edges', ...}）。

        State effect / 状态影响：
            Same as HMS.add_events(), executed in a worker thread.
            与 HMS.add_events() 相同，在工作线程中执行。
        """
        return await self._call(
            self._sync.add_events, events, user_id=user_id, agent_id=agent_id, run_id=run_id, infer=infer
        )

    async def compile(
        self,
        query: str,
        *,
        filters: dict,
        budget_tokens: int | None = None,
        mode: str | None = None,
        as_of: Any = None,
        require_witness: bool | None = None,
    ) -> ContextIR:
        """Awaitable HMS.compile(): the Work State for a query (axis first, graph on demand, deterministic budget).
        可等待的 HMS.compile()：查询的工作态（先轴，按需图扩展，确定性预算）。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            filters: Scope filters (at least one scope id).
            filters：范围过滤（至少一个范围 id）。

            budget_tokens: Token budget (config default when None).
            budget_tokens：token 预算（None 时取配置默认值）。

            mode: CompileMode literal ('auto' | 'fast' | 'axis_graph' | 'deep'); VALIDATION_013 otherwise.
            mode：CompileMode 字面量（'auto' | 'fast' | 'axis_graph' | 'deep'）；否则 VALIDATION_013。

            as_of: Validity instant (store.now() when None).
            as_of：有效性时刻（None 时为 store.now()）。

            require_witness: Override of compiler.require_witness.
            require_witness：覆盖 compiler.require_witness。

        Output / 输出：
            ContextIR.
            ContextIR。

        State effect / 状态影响：
            None (one store.now() call), executed in a worker thread.
            无（调用一次 store.now()），在工作线程中执行。
        """
        return await self._call(
            self._sync.compile,
            query,
            filters=filters,
            budget_tokens=budget_tokens,
            mode=mode,
            as_of=as_of,
            require_witness=require_witness,
        )

    async def axis_query(self, aq: AxisQuery | dict) -> AxisResult:
        """Awaitable HMS.axis_query(): the HMS-native seven-axis query.
        可等待的 HMS.axis_query()：HMS 原生七轴查询。

        Input / 输入：
            aq: AxisQuery model or its dict form.
            aq：AxisQuery 模型或其字典形式。

        Output / 输出：
            AxisResult.
            AxisResult。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(self._sync.axis_query, aq)

    async def why(
        self,
        target_id: str,
        *,
        hops: int = 2,
        filters: dict | None = None,
        edge_types: list[str] | None = None,
        include_invalid: bool = False,
    ) -> dict:
        """Awaitable HMS.why(): explain an object (or a question) by backward BFS over the Support Graph.
        可等待的 HMS.why()：通过支持图上的反向 BFS 解释一个对象（或问题）。

        Input / 输入：
            target_id: Object id, or a question resolved through a detective compile.
            target_id：对象 id，或经侦探式编译解析的问题。

            hops: BFS depth.
            hops：BFS 深度。

            filters: Scope filters (required when target_id is a question).
            filters：范围过滤（target_id 为问题时必需）。

            edge_types: Restrict to these edge types (WHY_EDGE_TYPES when None).
            edge_types：限定这些边类型（None 时为 WHY_EDGE_TYPES）。

            include_invalid: Traverse invalidated edges too.
            include_invalid：是否也遍历已失效的边。

        Output / 输出：
            The explanation dict ({'target', 'chain', 'anchors', ...}).
            解释字典（{'target', 'chain', 'anchors', ...}）。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(
            self._sync.why,
            target_id,
            hops=hops,
            filters=filters,
            edge_types=edge_types,
            include_invalid=include_invalid,
        )

    async def feedback(
        self,
        target_id: str | None,
        feedback: FeedbackType | str,
        *,
        reason: str | None = None,
        correction: str | dict | None = None,
        actor_id: str | None = None,
    ) -> dict:
        """Awaitable HMS.feedback(): apply one of the eleven state feedback types (§16.1) to a target.
        可等待的 HMS.feedback()：对目标应用十一种状态反馈类型之一（§16.1）。

        Input / 输入：
            target_id: Target object id (None only for state_missing).
            target_id：目标对象 id（仅 state_missing 可为 None）。

            feedback: FeedbackType or its literal (VALIDATION_006 otherwise).
            feedback：FeedbackType 或其字面量（否则 VALIDATION_006）。

            reason: Free-text reason recorded on the feedback event.
            reason：记录在反馈事件上的自由文本原因。

            correction: Correction text/dict (required by some types, VALIDATION_007).
            correction：修正文本/字典（某些类型必需，VALIDATION_007）。

            actor_id: Who gave the feedback.
            actor_id：反馈的给出者。

        Output / 输出：
            The feedback report dict.
            反馈报告字典。

        State effect / 状态影响：
            Same as HMS.feedback(), executed in a worker thread.
            与 HMS.feedback() 相同，在工作线程中执行。
        """
        return await self._call(
            self._sync.feedback, target_id, feedback, reason=reason, correction=correction, actor_id=actor_id
        )

    async def revoke(self, anchor_id: str, *, reason: str | None = None, actor_id: str | None = None) -> dict:
        """Awaitable HMS.revoke(): revoke an evidence anchor and cascade to everything it witnessed.
        可等待的 HMS.revoke()：撤销一个证据锚点并级联到其见证的一切。

        Input / 输入：
            anchor_id: Anchor id (unknown → NOT_FOUND_001).
            anchor_id：锚点 id（未知 → NOT_FOUND_001）。

            reason: Free-text reason.
            reason：自由文本原因。

            actor_id: Who revoked.
            actor_id：撤销者。

        Output / 输出：
            The revoke report dict.
            撤销报告字典。

        State effect / 状态影响：
            Same as HMS.revoke(), executed in a worker thread.
            与 HMS.revoke() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.revoke, anchor_id, reason=reason, actor_id=actor_id)

    async def handoff(
        self,
        *,
        filters: dict,
        goal: str | None = None,
        budget_tokens: int | None = None,
        checkpoint: bool = True,
        include_edges: bool = True,
    ) -> StatePackage:
        """Awaitable HMS.handoff(): build a StatePackage, optionally after taking a checkpoint.
        可等待的 HMS.handoff()：构建 StatePackage，可选地先做检查点。

        Input / 输入：
            filters: Scope filters.
            filters：范围过滤。

            goal: Goal text compiled into the package.
            goal：编译进包的目标文本。

            budget_tokens: Token budget for the compiled state.
            budget_tokens：编译状态的 token 预算。

            checkpoint: Take a checkpoint before building.
            checkpoint：构建前是否做检查点。

            include_edges: Include the witness-closure edges.
            include_edges：是否包含见证闭包的边。

        Output / 输出：
            StatePackage.
            StatePackage。

        State effect / 状态影响：
            Same as HMS.handoff() (a checkpoint row when checkpoint=True), executed in a worker thread.
            与 HMS.handoff() 相同（checkpoint=True 时写一行检查点），在工作线程中执行。
        """
        return await self._call(
            self._sync.handoff,
            filters=filters,
            goal=goal,
            budget_tokens=budget_tokens,
            checkpoint=checkpoint,
            include_edges=include_edges,
        )

    async def import_package(
        self,
        pkg: StatePackage | str | dict,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Awaitable HMS.import_package(): recreate a StatePackage in this store (ids kept, identity dedup).
        可等待的 HMS.import_package()：在本存储中重建 StatePackage（保留 id、按身份去重）。

        Input / 输入：
            pkg: StatePackage, its JSON text or its dict form.
            pkg：StatePackage、其 JSON 文本或字典形式。

            user_id: Re-scope target user id.
            user_id：重定范围的目标用户 id。

            agent_id: Re-scope target agent id.
            agent_id：重定范围的目标代理 id。

            run_id: Re-scope target run id.
            run_id：重定范围的目标运行 id。

        Output / 输出：
            The import report dict.
            导入报告字典。

        State effect / 状态影响：
            Same as HMS.import_package(), executed in a worker thread.
            与 HMS.import_package() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.import_package, pkg, user_id=user_id, agent_id=agent_id, run_id=run_id)

    async def checkpoint(self, *, filters: dict, label: str | None = None) -> str:
        """Awaitable HMS.checkpoint(): snapshot the validity of every object of a scope.
        可等待的 HMS.checkpoint()：快照某范围内每个对象的有效性。

        Input / 输入：
            filters: Scope filters.
            filters：范围过滤。

            label: Optional human label.
            label：可选的人类可读标签。

        Output / 输出：
            The checkpoint id.
            检查点 id。

        State effect / 状态影响：
            Same as HMS.checkpoint(), executed in a worker thread.
            与 HMS.checkpoint() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.checkpoint, filters=filters, label=label)

    async def rollback(self, checkpoint_id: str, *, reason: str | None = None) -> dict:
        """Awaitable HMS.rollback(): restore a scope to a checkpoint.
        可等待的 HMS.rollback()：将范围恢复到检查点。

        Input / 输入：
            checkpoint_id: Checkpoint id (unknown → NOT_FOUND_001).
            checkpoint_id：检查点 id（未知 → NOT_FOUND_001）。

            reason: Free-text reason.
            reason：自由文本原因。

        Output / 输出：
            The rollback report dict.
            回滚报告字典。

        State effect / 状态影响：
            Same as HMS.rollback(), executed in a worker thread.
            与 HMS.rollback() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.rollback, checkpoint_id, reason=reason)

    async def transitions(
        self,
        *,
        filters: dict,
        since: Any = None,
        scale: str | None = None,
        fmt: str = "tuples",
        kinds: list[str] | None = None,
    ) -> list | str:
        """Awaitable HMS.transitions(): export the (S_t, a_t, S_t+1, r_t, κ_t, w_t) transitions of a scope.
        可等待的 HMS.transitions()：导出某范围的 (S_t, a_t, S_t+1, r_t, κ_t, w_t) 转移。

        Input / 输入：
            filters: Scope filters.
            filters：范围过滤。

            since: Only transitions at or after this instant.
            since：仅此时刻及之后的转移。

            scale: Scale filter ('L1_micro' | 'L2_meso' | 'L3_macro').
            scale：尺度过滤（'L1_micro' | 'L2_meso' | 'L3_macro'）。

            fmt: 'tuples' | 'records' | 'jsonl' (VALIDATION_013 otherwise).
            fmt：'tuples' | 'records' | 'jsonl'（否则 VALIDATION_013）。

            kinds: TransitionKind filter.
            kinds：TransitionKind 过滤。

        Output / 输出：
            A list (tuples/records) or a JSONL string.
            列表（tuples/records）或 JSONL 字符串。

        State effect / 状态影响：
            Same as HMS.transitions() (idempotent export), executed in a worker thread.
            与 HMS.transitions() 相同（幂等导出），在工作线程中执行。
        """
        return await self._call(
            self._sync.transitions, filters=filters, since=since, scale=scale, fmt=fmt, kinds=kinds
        )

    async def consolidate(self, *, filters: dict | None = None, phases: list[str] | None = None) -> dict:
        """Awaitable HMS.consolidate(): run the StateScheduler phases on the matched scopes.
        可等待的 HMS.consolidate()：在匹配的范围上运行 StateScheduler 阶段。

        Input / 输入：
            filters: Scope filters (None → every scope).
            filters：范围过滤（None → 全部范围）。

            phases: Subset of orient/collect/consolidate/prune (VALIDATION_013 otherwise).
            phases：orient/collect/consolidate/prune 的子集（否则 VALIDATION_013）。

        Output / 输出：
            The consolidation report dict.
            整合报告字典。

        State effect / 状态影响：
            Same as HMS.consolidate(), executed in a worker thread.
            与 HMS.consolidate() 相同，在工作线程中执行。
        """
        return await self._call(self._sync.consolidate, filters=filters, phases=phases)

    async def graph_stats(self, *, filters: dict | None = None) -> dict:
        """Awaitable HMS.graph_stats(): Support Graph statistics for a scope (None → whole DB).
        可等待的 HMS.graph_stats()：某范围（None → 整库）的支持图统计。

        Input / 输入：
            filters: Scope filters (None → whole DB).
            filters：范围过滤（None → 整库）。

        Output / 输出：
            The statistics dict.
            统计字典。

        State effect / 状态影响：
            None.
            无。
        """
        return await self._call(self._sync.graph_stats, filters=filters)

    async def context(
        self,
        query: str,
        *,
        filters: dict,
        top_k: int | None = None,
        as_of: Any = None,
        budget_tokens: int | None = None,
    ) -> dict:
        """Awaitable HMS.context(): the hybrid read context (memoized session card + timestamped relevant rows, §6.7).
        可等待的 HMS.context()：混合读上下文（记忆化会话卡片 + 带时间戳的相关记忆行，§6.7）。

        Input / 输入：
            query: The question / goal text.
            query：问题 / 目标文本。

            filters: Scope filters (>= 1 scope key; scalar 'visibility' = audience).
            filters：范围过滤（至少一个范围键；标量 'visibility' = 受众）。

            top_k: Search rows before deduplication (None → config.compiler.context_rows_top_k; an explicit 0 reaches search unchanged and the rows part renders '(none)'; VALIDATION_009 when negative).
            top_k：去重前的检索行数（None → config.compiler.context_rows_top_k；显式的 0 原样传给 search、记忆行部分渲染为 '(none)'；为负时 VALIDATION_009）。

            as_of: Validity instant of the rows (None → store.now()).
            as_of：记忆行的有效性时刻（None → store.now()）。

            budget_tokens: Session-card budget (None → config default).
            budget_tokens：会话卡片预算（None → 配置默认值）。

        Output / 输出：
            The context dict of HMS.context() ('text' first).
            HMS.context() 的上下文字典（'text' 为首）。

        State effect / 状态影响：
            Same as HMS.context(), executed in a worker thread.
            与 HMS.context() 相同，在工作线程中执行。
        """
        return await self._call(
            self._sync.context, query, filters=filters, top_k=top_k, as_of=as_of, budget_tokens=budget_tokens
        )

    async def purge(
        self, *, filters: dict, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True
    ) -> dict:
        """Awaitable HMS.purge(): the retention escape hatch (§16.6), dry run by default.
        可等待的 HMS.purge()：保留策略逃生舱（§16.6），默认试运行。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            older_than_days: Age threshold on valid_to in days.
            older_than_days：以天计的 valid_to 年龄阈值。

            dry_run: True reports only; False materializes (needs config.retention.enabled).
            dry_run：True 只报告；False 真正执行（需要 config.retention.enabled）。

            keep_witnessed: True never removes anchors or source events.
            keep_witnessed：True 从不删除锚点与源事件。

        Output / 输出：
            The purge report dict.
            清理报告字典。

        State effect / 状态影响：
            The wrapped HMS.purge write (none on a dry run).
            被包装的 HMS.purge 写入（试运行时无）。
        """
        return await self._call(
            self._sync.purge,
            filters=filters,
            older_than_days=older_than_days,
            dry_run=dry_run,
            keep_witnessed=keep_witnessed,
        )

    async def vacuum(self) -> dict:
        """Awaitable HMS.vacuum(): VACUUM the store and report the size before and after.
        可等待的 HMS.vacuum()：对存储执行 VACUUM 并报告前后大小。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'}.
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'}。

        State effect / 状态影响：
            Rewrites the database file.
            重写数据库文件。
        """
        return await self._call(self._sync.vacuum)

    # ------------------------------------------------------------------ background daemon
    # ------------------------------------------------------------------ 后台守护进程
    async def start_daemon(
        self,
        *,
        interval_s: float | None = None,
        phases: list[str] | None = None,
        filters: dict | None = None,
    ) -> StateDaemon:
        """Awaitable HMS.start_daemon(): start (or return the running) background consolidation daemon.
        可等待的 HMS.start_daemon()：启动（或返回正在运行的）后台固化守护进程。

        Input / 输入：
            interval_s: Seconds between runs (None → config.scheduler.daemon_interval_s; both None → VALIDATION_009).
            interval_s：两次运行之间的秒数（None → config.scheduler.daemon_interval_s；两者皆 None → VALIDATION_009）。

            phases: Phase names in run order (None → all four).
            phases：按运行顺序的阶段名（None → 全部四个）。

            filters: Full or partial scope filters (None → every scope).
            filters：完整或部分范围过滤（None → 全部范围）。

        Output / 输出：
            The StateDaemon.
            StateDaemon。

        State effect / 状态影响：
            Spawns the daemon thread (its writes happen inside consolidate()).
            派生守护线程（其写入发生在 consolidate() 内）。
        """
        return await self._call(self._sync.start_daemon, interval_s=interval_s, phases=phases, filters=filters)

    async def stop_daemon(self, timeout: float | None = None) -> bool:
        """Awaitable HMS.stop_daemon(): stop the daemon and join its thread off the event loop.
        可等待的 HMS.stop_daemon()：在事件循环之外停止守护进程并等待其线程。

        Input / 输入：
            timeout: Max seconds to wait for the thread (None → until it exits).
            timeout：等待线程的最长秒数（None → 直到退出）。

        Output / 输出：
            True when no daemon thread is alive afterwards.
            之后没有存活的守护线程时为 True。

        State effect / 状态影响：
            Sets the daemon's stop event.
            设置守护进程的停止事件。
        """
        return await self._call(self._sync.stop_daemon, timeout)

    # ------------------------------------------------------------------ §11 state surface
    # ------------------------------------------------------------------ §11 状态表面
    @property
    def state(self) -> StateAPI:
        """The synchronous StateAPI of the wrapped HMS (a plain attribute, like `sync`).
        被封装 HMS 的同步 StateAPI（普通属性，与 `sync` 一样）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            StateAPI.
            StateAPI。

        State effect / 状态影响：
            None.
            无。
        """
        return self._sync.state

    async def work_state(
        self, query: str, *, filters: dict, mode: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[WorkState, str]:
        """Awaitable HMS.work_state(): the Work State Card.
        可等待的 HMS.work_state()：工作状态卡。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            mode: Compile mode.
            mode：编译模式。

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
        return await self._call(
            self._sync.work_state, query, filters=filters, mode=mode, as_of=as_of, budget_tokens=budget_tokens
        )

    async def handoff_package(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[HandoffPackageState, str]:
        """Awaitable HMS.handoff_package(): the Handoff Package.
        可等待的 HMS.handoff_package()：交接包。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            goal: Handoff goal.
            goal：交接目标。

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
        return await self._call(
            self._sync.handoff_package, filters=filters, goal=goal, checkpoint=checkpoint, budget_tokens=budget_tokens
        )

    async def detective_report(
        self, target: str, *, filters: dict | None = None, hops: int = 2, as_of: Any = None
    ) -> tuple[DetectiveReport, str]:
        """Awaitable HMS.detective_report(): the Detective Report.
        可等待的 HMS.detective_report()：侦探报告。

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
        return await self._call(self._sync.detective_report, target, filters=filters, hops=hops, as_of=as_of)

    async def rollback_point(
        self, *, filters: dict, label: str | None = None, create: bool = False
    ) -> tuple[RollbackPoint | None, str]:
        """Awaitable HMS.rollback_point(): the Rollback Point.
        可等待的 HMS.rollback_point()：回滚点。

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
        return await self._call(self._sync.rollback_point, filters=filters, label=label, create=create)

    async def personal_state(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str | None = None,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """Awaitable HMS.personal_state(): PersonalState + Personal Card.
        可等待的 HMS.personal_state()：PersonalState 与个人卡。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            subject: Subject selector.
            subject：主体选择器。

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
        return await self._call(
            self._sync.personal_state,
            filters=filters,
            subject=subject,
            audience=audience,
            as_of=as_of,
            timeline_limit=timeline_limit,
        )

    async def shared_card(
        self,
        *,
        filters: dict,
        subject: str = "user",
        audience: Visibility | str = Visibility.team,
        as_of: Any = None,
        timeline_limit: int = 20,
    ) -> tuple[PersonalState, str]:
        """Awaitable HMS.shared_card(): the Shared Card.
        可等待的 HMS.shared_card()：Shared Card。

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
        return await self._call(
            self._sync.shared_card,
            filters=filters,
            subject=subject,
            audience=audience,
            as_of=as_of,
            timeline_limit=timeline_limit,
        )

    async def edge_state(
        self, a: str | dict, b: str | dict, *, filters: dict | None = None, as_of: Any = None
    ) -> tuple[EdgeState, str]:
        """Awaitable HMS.edge_state(): EdgeState + Edge Card.
        可等待的 HMS.edge_state()：EdgeState 与关系卡。

        Input / 输入：
            a: Entity text or scope dict.
            a：实体文本或范围字典。

            b: Entity text or scope dict.
            b：实体文本或范围字典。

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
        return await self._call(self._sync.edge_state, a, b, filters=filters, as_of=as_of)

    async def group_state(self, *, members: list[dict], as_of: Any = None) -> tuple[GroupState, str]:
        """Awaitable HMS.group_state(): GroupState + Group Card.
        可等待的 HMS.group_state()：GroupState 与群体卡。

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
        return await self._call(self._sync.group_state, members=members, as_of=as_of)

    async def customer_issue_state(
        self, *, filters: dict, issue: str | None = None, as_of: Any = None, budget_tokens: int | None = None
    ) -> tuple[CustomerIssueState, str]:
        """Awaitable HMS.customer_issue_state(): CustomerIssueState + card.
        可等待的 HMS.customer_issue_state()：CustomerIssueState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            issue: Issue text.
            issue：问题文本。

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
        return await self._call(
            self._sync.customer_issue_state, filters=filters, issue=issue, as_of=as_of, budget_tokens=budget_tokens
        )

    async def incident_state(
        self, *, filters: dict, question: str | None = None, as_of: Any = None, hops: int = 2
    ) -> tuple[IncidentState, str]:
        """Awaitable HMS.incident_state(): IncidentState + card.
        可等待的 HMS.incident_state()：IncidentState + 卡片。

        Input / 输入：
            filters: Mem0 filter dict with >= 1 scope key.
            filters：至少含一个范围键的 Mem0 过滤字典。

            question: Detective question.
            question：侦探式问题。

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
        return await self._call(self._sync.incident_state, filters=filters, question=question, as_of=as_of, hops=hops)

    async def meeting_decision_state(self, *, filters: dict, as_of: Any = None) -> tuple[MeetingDecisionState, str]:
        """Awaitable HMS.meeting_decision_state(): MeetingDecisionState + card.
        可等待的 HMS.meeting_decision_state()：MeetingDecisionState + 卡片。

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
        return await self._call(self._sync.meeting_decision_state, filters=filters, as_of=as_of)

    async def account_handoff_state(
        self, *, filters: dict, goal: str | None = None, checkpoint: bool = True, budget_tokens: int | None = None
    ) -> tuple[AccountHandoffState, str]:
        """Awaitable HMS.account_handoff_state(): AccountHandoffState + card.
        可等待的 HMS.account_handoff_state()：AccountHandoffState + 卡片。

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
        return await self._call(
            self._sync.account_handoff_state,
            filters=filters,
            goal=goal,
            checkpoint=checkpoint,
            budget_tokens=budget_tokens,
        )

    # ------------------------------------------------------------------ telemetry (synchronous, no executor hop)
    # ------------------------------------------------------------------ 遥测（同步，不经执行器）
    def usage(self) -> telemetry.Usage:
        """Merged llm + embedder usage of the wrapped HMS (HMS.usage()).
        被封装 HMS 的 llm + embedder 合并用量（HMS.usage()）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A telemetry.Usage snapshot; reading a meter never blocks, so this is a plain method, not a coroutine.
            telemetry.Usage 快照；读取计量器从不阻塞，因此这是普通方法而非协程。

        State effect / 状态影响：
            None.
            无。
        """
        return self._sync.usage()

    def trace(self, operation: str, request_id: str | None = None) -> AbstractContextManager[telemetry.RequestTrace]:
        """telemetry.trace(operation, request_id): open a RequestTrace that the awaited facade calls append stages to.
        telemetry.trace(operation, request_id)：打开一个 RequestTrace，被 await 的门面调用会向其追加阶段。

        Input / 输入：
            operation: Name of the traced operation (e.g. 'add', 'search').
            operation：被追踪操作的名称（如 'add'、'search'）。

            request_id: Caller-supplied id; a uuid4 hex when None.
            request_id：调用方给定的 id；None 时为 uuid4 十六进制串。

        Output / 输出：
            The context manager from telemetry.trace, to be entered with `with` in the calling task (not in to_thread,
            because asyncio.to_thread copies the task's context into the worker only at call time).
            telemetry.trace 返回的上下文管理器，在调用任务里用 `with` 进入（不要在 to_thread 内，因为
            asyncio.to_thread 只在调用时把任务上下文复制进工作线程）。

        State effect / 状态影响：
            Sets the context-local active trace for the block and restores the previous one on exit.
            在块内设置上下文局部的活动追踪，退出时恢复之前的追踪。
        """
        return telemetry.trace(operation, request_id)
