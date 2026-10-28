"""AsyncHMS tests: signature parity, verbatim forwarding, non-blocking loop, concurrency, context manager, defaults.
AsyncHMS 测试：签名一致、逐字转发、不阻塞事件循环、并发、上下文管理器与默认值。
"""

# Purpose (EN): Pin the asyncio twin to the synchronous facade: same 41 signatures (11 Mem0 + 18 native + 12 state
#               cards), same arguments forwarded, same results on the same deterministic store, and real off-loop
#               execution under concurrency.
# 目的（中）：把 asyncio 孪生钉在同步门面上：相同的 41 个签名（11 个 Mem0 + 18 个原生 + 12 个状态卡）、相同的参数
#            转发、相同确定性存储上的相同结果，以及并发下真正的循环外执行。

# Input (EN): The conftest fixtures (clock, ids, events, scope4, index_provider) and tests/fixtures/events/*.json.
# 输入（中）：conftest 夹具（clock、ids、events、scope4、index_provider）与 tests/fixtures/events/*.json。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory stores and files under tmp_path (HMS_DIR); no network (the guard stays).
# 状态影响（中）：除内存存储与 tmp_path（HMS_DIR）下的文件外无副作用；无网络（守卫保持生效）。

from __future__ import annotations

import asyncio
import inspect
import socket
import time
from typing import Any, Coroutine

import pytest

import hms_elite
from hms_elite import HMS, AsyncHMS, HMSConfig
from hms_elite.exceptions import ConfigError, StoreError, ValidationError
from hms_elite.memory.main import _OSSProject
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import _UNSET
from tests.conftest import FrozenClock, SequentialIds

U = "alice"
LATTE = "I like oat-milk lattes in the morning."
S4_QUERY = "fix checkout webhook retry"
S4_AS_OF = "2026-06-15T12:00:00Z"
S6_CORRECTION = "retry.ts may change once billing_e2e is isolated."
MEM0_METHODS = (
    "add",
    "get",
    "get_all",
    "search",
    "update",
    "delete",
    "delete_all",
    "history",
    "reset",
    "close",
    "chat",
)
NATIVE_METHODS = (
    "add_events",
    "compile",
    "axis_query",
    "why",
    "feedback",
    "revoke",
    "handoff",
    "import_package",
    "checkpoint",
    "rollback",
    "transitions",
    "consolidate",
    "graph_stats",
    "context",
    "purge",
    "vacuum",
    "start_daemon",
    "stop_daemon",
)
# The §11 state cards delegated by HMS to its StateAPI (signatures verbatim on both facades).
# HMS 委托给其 StateAPI 的 §11 状态卡（两个门面上签名逐字相同）。
STATE_METHODS = (
    "work_state",
    "handoff_package",
    "detective_report",
    "rollback_point",
    "personal_state",
    "shared_card",
    "edge_state",
    "group_state",
    "customer_issue_state",
    "incident_state",
    "meeting_decision_state",
    "account_handoff_state",
)
ALL_METHODS = MEM0_METHODS + NATIVE_METHODS + STATE_METHODS

# The real socket class, captured before the autouse no_network guard swaps it in (collection precedes fixtures).
# 真实的 socket 类，在 autouse 的 no_network 守卫替换它之前捕获（收集阶段先于夹具执行）。
_REAL_SOCKET: type = socket.socket
while _REAL_SOCKET.__name__ == "_BlockedSocket":
    _REAL_SOCKET = _REAL_SOCKET.__bases__[0]


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine on a fresh loop whose self-pipe is built with the real socket class; the guard stays otherwise.
    在新事件循环上运行协程，其自管道用真实 socket 类构建；其余时间守卫保持生效。
    """
    guard = socket.socket
    socket.socket = _REAL_SOCKET  # type: ignore[misc]
    try:
        loop = asyncio.new_event_loop()
    finally:
        socket.socket = guard  # type: ignore[misc]
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _memory_config(index_provider: str = "memory") -> HMSConfig:
    """An in-memory HMSConfig for the given index provider.
    给定索引提供者的内存 HMSConfig。
    """
    return HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"})


class _Recorder:
    """Stand-in for the wrapped HMS: every attribute is a method that records (name, args, kwargs).
    被封装 HMS 的替身：每个属性都是记录 (name, args, kwargs) 的方法。
    """

    def __init__(self) -> None:
        """Start with an empty call log.
        以空的调用日志开始。
        """
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str) -> Any:
        """Return a recording method for any name.
        为任意名称返回一个记录方法。
        """

        def method(*args: Any, **kwargs: Any) -> tuple[str, str]:
            """Record the call and return a marker.
            记录调用并返回标记。
            """
            self.calls.append((name, args, kwargs))
            return ("ok", name)

        return method


class _Awaitable:
    """Wrap a synchronous HMS so the parity script can `await` it exactly like an AsyncHMS.
    封装同步 HMS，使一致性脚本能像 AsyncHMS 一样 `await` 它。
    """

    def __init__(self, sync: HMS) -> None:
        """Keep the synchronous facade.
        保存同步门面。
        """
        self._sync = sync

    def __getattr__(self, name: str) -> Any:
        """Return an async wrapper that calls the synchronous method inline.
        返回内联调用同步方法的异步包装。
        """
        func = getattr(self._sync, name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            """Call the synchronous method.
            调用同步方法。
            """
            return func(*args, **kwargs)

        return method


def _probe_args(name: str) -> tuple[list[Any], dict[str, Any]]:
    """Distinct probe values for every parameter of HMS.<name>: positional for POSITIONAL_OR_KEYWORD, keyword otherwise.
    为 HMS.<name> 的每个参数生成可区分的探测值：POSITIONAL_OR_KEYWORD 用位置传递，其余用关键字。
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in list(inspect.signature(getattr(HMS, name)).parameters.values())[1:]:
        value = f"<{name}.{param.name}>"
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(value)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs["extra_kw"] = f"<{name}.extra_kw>"
    return args, kwargs


async def _script(api: Any, clock: FrozenClock, events: Any, scope4: dict) -> list[Any]:
    """Drive the 24 data-path public methods in one deterministic flow and collect JSON-comparable results.
    用一条确定性流程驱动 24 个数据路径公共方法并收集可比较的 JSON 结果。
    """
    out: list[Any] = []
    fixture = events("structured_billing")
    r_add = await api.add(LATTE, user_id=U, run_id="run-coffee")
    out.append(r_add)
    mem = r_add["results"][0]["id"]
    out.append(await api.get(mem))
    out.append(await api.search("oat milk latte", filters={"user_id": U}, top_k=5))
    out.append(await api.get_all(filters={"user_id": U}, top_k=10))
    clock.tick(seconds=60)
    out.append(await api.update(mem, text="User likes oat-milk lattes in the afternoon."))
    out.append(await api.history(mem))
    clock.tick(seconds=60)
    r_uc = await api.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    out.append(r_uc)
    clock.tick(seconds=60)
    ra = await api.add_events(fixture.events_fail, **scope4)
    out.append(ra)
    clock.tick(seconds=60)
    ir = await api.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    out.append(ir.to_text())
    aq = await api.axis_query({"filters": {**scope4, "entity": "retry.ts"}, "aggregation": "count"})
    out.append(aq.model_dump(mode="json"))
    e2 = [row["atom_id"] for row in ra["results"] if row["atom_type"] == "event"][1]
    out.append(await api.why(e2, filters=scope4))
    out.append(await api.graph_stats(filters=scope4))
    clock.tick(seconds=60)
    cp = await api.checkpoint(filters=scope4, label="before-fix")
    out.append(cp)
    clock.tick(seconds=60)
    out.append(await api.add_events(fixture.events_fix, **scope4))
    clock.tick(seconds=60)
    out.append(await api.feedback(r_uc["results"][0]["id"], "state_outdated", correction=S6_CORRECTION))
    clock.tick(seconds=60)
    out.append(await api.rollback(cp, reason="test"))
    out.append(await api.transitions(filters=scope4, fmt="records"))
    pkg = await api.handoff(filters=scope4, goal=S4_QUERY, checkpoint=False)
    out.append(pkg.to_json())
    clock.tick(seconds=60)
    out.append(await api.import_package(pkg, user_id="bob", agent_id="coder", run_id="run-import"))
    out.append(await api.consolidate(filters=scope4))
    clock.tick(seconds=60)
    out.append(await api.delete(mem))
    out.append(await api.get(mem))
    clock.tick(seconds=60)
    out.append(await api.revoke(r_uc["results"][0]["anchor_ids"][0], reason="test"))
    out.append(await api.delete_all(user_id=U, run_id="run-coffee"))
    out.append(await api.get_all(filters={"user_id": U}, top_k=10, show_expired=True))
    await api.reset()
    out.append(await api.get_all(filters={"user_id": U}, top_k=10))
    await api.close()
    return out


def test_async_signatures_mirror_hms():
    """Every one of the 41 public methods is a coroutine function whose inspect.signature equals HMS's verbatim.
    41 个公共方法每一个都是协程函数，且其 inspect.signature 与 HMS 逐字相同。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over signatures, coroutine-ness, __init__/from_config/create shapes and the package export.
        对签名、协程属性、__init__/from_config/create 形状与包导出的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert len(ALL_METHODS) == 41 and len(set(ALL_METHODS)) == 41
    assert len(STATE_METHODS) == 12 and len(NATIVE_METHODS) == 18
    for name in ALL_METHODS:
        async_method = getattr(AsyncHMS, name)
        assert inspect.iscoroutinefunction(async_method), name
        assert not inspect.iscoroutinefunction(getattr(HMS, name)), name
        assert inspect.signature(async_method) == inspect.signature(getattr(HMS, name)), name
    assert inspect.signature(AsyncHMS.update).parameters["expiration_date"].default is _UNSET
    assert inspect.signature(AsyncHMS.__init__) == inspect.signature(HMS.__init__)
    from_config_params = inspect.signature(AsyncHMS.from_config).parameters.keys()
    assert from_config_params == inspect.signature(HMS.from_config).parameters.keys()
    assert not inspect.iscoroutinefunction(AsyncHMS.from_config)
    assert inspect.iscoroutinefunction(AsyncHMS.create)
    assert list(inspect.signature(AsyncHMS.create).parameters) == ["config", "store"]
    assert inspect.iscoroutinefunction(AsyncHMS.aclose)
    assert inspect.iscoroutinefunction(AsyncHMS.__aenter__) and inspect.iscoroutinefunction(AsyncHMS.__aexit__)
    assert "AsyncHMS" in hms_elite.__all__ and hms_elite.AsyncHMS is AsyncHMS
    # `state` is a synchronous attribute on both facades (the StateAPI itself), never a coroutine.
    # `state` 在两个门面上都是同步属性（StateAPI 本身），绝不是协程。
    assert isinstance(AsyncHMS.state, property) and isinstance(HMS.state, property)


def test_async_forwards_every_argument_verbatim():
    """Each async method hands its bound arguments to the same-named HMS method unchanged and returns its result.
    每个异步方法把绑定的参数原样交给同名 HMS 方法并返回其结果。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions comparing the bound arguments seen by the recorder with those given to the async method.
        比较记录器看到的绑定参数与传给异步方法的参数的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    api = AsyncHMS(_memory_config())
    api.sync.close()
    recorder = _Recorder()
    api._sync = recorder  # type: ignore[assignment]
    for name in ALL_METHODS:
        args, kwargs = _probe_args(name)
        result = _run(getattr(api, name)(*args, **kwargs))
        assert result == ("ok", name), name
        recorded_name, recorded_args, recorded_kwargs = recorder.calls[-1]
        assert recorded_name == name
        signature = inspect.signature(getattr(HMS, name))
        expected = signature.bind(None, *args, **kwargs)
        actual = signature.bind(None, *recorded_args, **recorded_kwargs)
        expected.apply_defaults()
        actual.apply_defaults()
        assert actual.arguments == expected.arguments, name
    assert [call[0] for call in recorder.calls] == list(ALL_METHODS)
    # aclose forwards to close() with no arguments / aclose 无参转发到 close()
    _run(api.aclose())
    assert recorder.calls[-1] == ("close", (), {})


def test_async_does_not_block_the_event_loop():
    """A slow synchronous call runs off the loop: a 10 ms ticker keeps advancing while search() sleeps 250 ms.
    缓慢的同步调用在循环之外运行：search() 睡眠 250 ms 期间，10 ms 的计时器持续推进。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the awaited result and the number of ticks observed during the call.
        对等待结果与调用期间观测到的计时次数的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """

    async def main() -> tuple[dict, int]:
        """Race the slow search against a ticker.
        让慢速搜索与计时器竞赛。
        """
        async with AsyncHMS(_memory_config()) as api:
            ticks = 0

            def slow_search(*args: Any, **kwargs: Any) -> dict:
                """Block the worker thread for 250 ms.
                阻塞工作线程 250 ms。
                """
                time.sleep(0.25)
                return {"results": []}

            async def ticker() -> None:
                """Count loop iterations every 10 ms.
                每 10 ms 计一次循环迭代。
                """
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.01)

            api.sync.search = slow_search  # type: ignore[method-assign]
            task = asyncio.create_task(ticker())
            result = await api.search("anything", filters={"user_id": U})
            task.cancel()
            return result, ticks

    result, ticks = _run(main())
    assert result == {"results": []}
    assert ticks >= 5, ticks


def test_async_concurrent_add_and_search(index_provider: str):
    """asyncio.gather of 20 add() + 20 search() raises nothing and persists exactly 20 atoms.
    asyncio.gather 并发 20 次 add() + 20 次 search() 不抛异常，且恰好持久化 20 个原子。

    Input / 输入：
        index_provider: 'sqlite' or 'memory' (conftest parametrization).
        index_provider：'sqlite' 或 'memory'（conftest 参数化）。

    Output / 输出：
        Assertions on the gathered results, the listing size, the store row counts and the index size.
        对 gather 结果、列表大小、存储行数与索引大小的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """

    async def main() -> None:
        """Run the concurrent workload.
        运行并发工作负载。
        """
        async with await AsyncHMS.create(_memory_config(index_provider)) as api:
            adds = [
                api.add(f"Note {i}: the user prefers tool number {i} for task {i}.", user_id=U, infer=False)
                for i in range(20)
            ]
            searches = [api.search(f"tool number {i}", filters={"user_id": U}, top_k=5) for i in range(20)]
            results = await asyncio.gather(*adds, *searches, return_exceptions=True)
            errors = [r for r in results if isinstance(r, BaseException)]
            assert errors == [], errors
            add_rows = results[:20]
            assert all(len(r["results"]) == 1 and r["results"][0]["event"] == "ADD" for r in add_rows)
            assert len({r["results"][0]["id"] for r in add_rows}) == 20
            assert all(isinstance(r["results"], list) for r in results[20:])
            listing = await api.get_all(filters={"user_id": U}, top_k=100)
            assert len(listing["results"]) == 20
            stats = api.store.stats()
            assert (stats["memory_atoms"], stats["source_events"], stats["anchors"], stats["history"]) == (20,) * 4
            assert api.sync.index.count() == 20
            hits = await api.search("tool number 7", filters={"user_id": U}, top_k=3)
            assert hits["results"] and hits["results"][0]["memory"].startswith("Note 7:")

    _run(main())


def test_async_context_manager_and_close_idempotent():
    """`async with` closes the store on exit; later calls raise STORE_001; aclose()/close() stay idempotent.
    `async with` 退出时关闭存储；之后的调用抛出 STORE_001；aclose()/close() 保持幂等。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the yielded object, the post-exit error code and repeated closes.
        对产出对象、退出后错误码与重复关闭的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """

    async def main() -> AsyncHMS:
        """Use the context manager once and return the closed instance.
        使用一次上下文管理器并返回已关闭的实例。
        """
        async with AsyncHMS(_memory_config()) as api:
            assert isinstance(api, AsyncHMS)
            added = await api.add(LATTE, user_id=U, infer=False)
            assert added["results"][0]["memory"] == LATTE
        with pytest.raises(StoreError) as exc_info:
            await api.get_all(filters={"user_id": U})
        assert exc_info.value.error_code == "STORE_001"
        await api.aclose()
        await api.close()
        return api

    api = _run(main())
    assert api.store.in_transaction is False


def test_async_offline_defaults_passthrough_and_errors():
    """AsyncHMS() is rule + hash + sqlite under HMS_DIR; .sync/.config/.store/.project pass through; errors propagate.
    AsyncHMS() 是 HMS_DIR 下的 rule + hash + sqlite；.sync/.config/.store/.project 透传；错误向上传播。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on providers, pass-through identity, from_config, ConfigError, chat() and VALIDATION/NOT_FOUND codes.
        对提供者、透传身份、from_config、ConfigError、chat() 与 VALIDATION/NOT_FOUND 错误码的断言。

    State effect / 状态影响：
        Creates the default database under HMS_DIR (tmp_path).
        在 HMS_DIR（tmp_path）下创建默认数据库。
    """
    api = AsyncHMS()
    try:
        assert isinstance(api.sync, HMS) and api.config is api.sync.config and api.store is api.sync.store
        assert (api.config.llm.provider, api.config.embedder.provider, api.config.vector_store.provider) == (
            "rule",
            "hash",
            "sqlite",
        )
        assert isinstance(api.project, _OSSProject) and api.project is api.sync.project
        with pytest.raises(ValueError):
            api.project.update(custom_instructions="x")
        with pytest.raises(NotImplementedError):
            _run(api.chat("hello"))
        with pytest.raises(ValidationError) as exc_info:
            _run(api.get_all())
        assert exc_info.value.error_code.startswith("VALIDATION_")
        with pytest.raises(ValueError):
            _run(api.delete("mem_000000000000"))
    finally:
        api.sync.close()
    with pytest.raises(ConfigError) as bad:
        AsyncHMS("not-a-config")  # type: ignore[arg-type]
    assert bad.value.error_code == "CONFIG_001"
    with pytest.raises(ConfigError):
        AsyncHMS.from_config({"vector_store": {"provider": "no-such-provider"}, "store": {"path": ":memory:"}})
    from_dict = AsyncHMS.from_config({"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}})
    assert isinstance(from_dict, AsyncHMS) and from_dict.config.vector_store.provider == "memory"
    from_dict.sync.close()
    created = _run(AsyncHMS.create(_memory_config(), store=StateStore(":memory:")))
    assert isinstance(created, AsyncHMS) and created.store.path == ":memory:"
    created.sync.close()


def test_async_results_equal_sync_results(events, scope4):
    """The same 24-method script on two identical deterministic stores yields identical results, sync and async.
    同一 24 方法脚本在两个相同的确定性存储上，同步与异步产出完全相同的结果。

    Input / 输入：
        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Equality of the two collected result lists and a length check.
        两个结果列表相等以及长度检查。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    clock_a, clock_b = FrozenClock(), FrozenClock()
    store_a = StateStore(":memory:", now=clock_a.now, id_factory=SequentialIds())
    store_b = StateStore(":memory:", now=clock_b.now, id_factory=SequentialIds())
    sync_api = _Awaitable(HMS(_memory_config(), store=store_a))
    async_api = AsyncHMS(_memory_config(), store=store_b)
    expected = _run(_script(sync_api, clock_a, events, scope4))
    actual = _run(_script(async_api, clock_b, events, scope4))
    assert len(expected) == 26
    assert actual == expected
    assert clock_a.now() == clock_b.now()


def test_async_usage_and_trace_are_synchronous_extras():
    """usage() returns the merged meter snapshot without awaiting; trace() entered in the task is seen by awaited calls.
    usage() 无需 await 即返回合并的计量快照；在任务里进入的 trace() 能被 await 的调用看到。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over Usage, RequestTrace stages and the hms4eye-style SimpleNamespace(stage=...) recorder.
        对 Usage、RequestTrace 阶段与 hms4eye 风格 SimpleNamespace(stage=...) 记录器的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    from hms_elite import telemetry

    async def scenario() -> tuple[telemetry.Usage, telemetry.Usage, telemetry.RequestTrace]:
        """Add one memory inside a trace and snapshot usage before and after.
        在追踪内添加一条记忆，并在前后各取一次用量快照。
        """
        async with AsyncHMS(_memory_config()) as api:
            assert not inspect.iscoroutinefunction(AsyncHMS.usage)
            assert not inspect.iscoroutinefunction(AsyncHMS.trace)
            before = api.usage()
            with api.trace("add", "req-async-1") as record:
                await api.add(LATTE, user_id=U)
            after = api.usage()
            assert telemetry.current_trace() is None
            return before, after, record

    before, after, record = _run(scenario())
    assert isinstance(before, telemetry.Usage) and before == telemetry.Usage()
    assert isinstance(after, telemetry.Usage)
    assert after.llm_calls + after.embedding_calls > 0 and after.total_tokens > 0
    assert record.request_id == "req-async-1" and record.operation == "add"
    assert record.duration_ms >= 0 and record.error is None
    assert record.stages and all(stage.duration_ms >= 0 for stage in record.stages)
    # The module-level stage() satisfies the hms4eye adapter's `stage(name)` protocol.
    # 模块级 stage() 满足 hms4eye 适配器的 `stage(name)` 协议。
    import types

    recorder = types.SimpleNamespace(stage=telemetry.stage)
    with telemetry.trace("outer") as outer, recorder.stage("mem0.add") as inner:
        assert inner is not None
    assert [stage.operation for stage in outer.stages] == ["mem0.add"]
