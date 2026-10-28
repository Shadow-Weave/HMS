"""Store ↔ index consistency under concurrency: commit-ordered listener dispatch, lock order, listener balance, FTS follow-up.
并发下的存储 ↔ 索引一致性：按提交顺序派发监听器、锁序、监听器数量平衡、FTS 跟进。
"""

# Purpose (EN): Regression tests for the review findings on index synchronization — two commits on one atom from two
#               threads must reach every provider in commit order with the full committed snapshot; a search inside a
#               transaction must never deadlock against a concurrent search; a facade registers its index listener
#               exactly once and removes it on close(); a topic or entity linked after insert is keyword-recallable.
# 目的（中）：索引同步相关审查发现的回归测试——两个线程对同一原子的两次提交必须按提交顺序、携带完整已提交快照
#             到达每个提供者；事务内的搜索绝不能与并发搜索死锁；门面恰好注册一次索引监听器并在 close() 时移除；
#             插入后链接的主题或实体能被关键词通道召回。

# Input (EN): conftest fixtures (store, clock, ids, index_provider, milvus_uri, tmp_path) and small hand-built corpora.
# 输入（中）：conftest 夹具（store、clock、ids、index_provider、milvus_uri、tmp_path）与手工构造的小语料。

# Output (EN): Assertions; every test fails on the code before the fix it guards.
# 输出（中）：断言；每个测试在其守护的修复之前的代码上都会失败。

# State effect (EN): In-memory stores, one tmp_path SQLite file and one milvus-lite file; no network, nothing under ~.
# 状态影响（中）：内存 store、一个 tmp_path 下的 SQLite 文件与一个 milvus-lite 文件；不联网，不写 ~。

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from hms_elite.configs.base import HMSConfig
from hms_elite.index import IndexFilter
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.memory.main import HMS
from hms_elite.memory.storage import StateStore
from hms_elite.models import MemoryAtom, Scope
from hms_elite.utils.scoring import atom_keyword_tokens
from tests.conftest import FrozenClock, SequentialIds
from tests.test_retrieval import _anchor

U = "alice"
SCOPE = Scope(user_id=U)
TEXTS = [f"Note {i}: the user prefers tool number {i} for task {i}." for i in range(4)]


def _hms(provider: str, tmp_path: Any) -> HMS:
    """An HMS on the given index provider over a fresh in-memory store (milvus-lite under tmp_path).
    在给定索引提供者上、基于全新内存 store 的 HMS（milvus-lite 放在 tmp_path 下）。
    """
    env: dict[str, Any] = {}
    if provider == "milvus":
        pytest.importorskip("pymilvus")
        env = {"config": {"uri": str(tmp_path / "sync.db"), "index_type": "FLAT", "scalar_index": "none"}}
    return HMS(
        HMSConfig(vector_store={"provider": provider, **env}, store={"path": ":memory:"}),
        store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()),
    )


def _recalled(hms: HMS, query: list[float]) -> set[str]:
    """The semantic channel's view of the live scope (filter pushdown or the axis_ids gate).
    语义通道眼中的存活范围（过滤下推或 axis_ids 门控）。
    """
    as_of = hms.store.now()
    if hms.index.supports_filter:
        hits = hms.index.search(query, filter=IndexFilter(scope=SCOPE, as_of=as_of), top_k=10)
    else:
        hits = hms.index.search(query, candidate_ids=hms.store.axis_ids(SCOPE, as_of=as_of), top_k=10)
    return {hit for hit, _score in hits}


@pytest.mark.parametrize("provider", ["sqlite", "memory", "milvus"])
def test_listener_dispatch_follows_commit_order(provider: str, tmp_path: Any) -> None:
    """archive (slow listener) then restore from two threads: listeners run in commit order, the provider ends live.
    两个线程先归档（监听器慢）再恢复：监听器按提交顺序运行，提供者以存活告终。

    Input / 输入：
        provider: 'sqlite' | 'memory' | 'milvus' (milvus skipped without pymilvus).
        provider：'sqlite' | 'memory' | 'milvus'（缺少 pymilvus 时跳过）。

        tmp_path: pytest temporary directory for the milvus-lite file.
        tmp_path：存放 milvus-lite 文件的 pytest 临时目录。

    Output / 输出：
        Assertions: listener call order == commit order, every call carries the full INDEX_ATTR_KEYS snapshot, and
        (providers with scalars) the semantic channel returns the atom again.
        断言：监听器调用顺序 == 提交顺序，每次调用都携带完整的 INDEX_ATTR_KEYS 快照，且（有标量的提供者）语义通道
        再次返回该原子。

    State effect / 状态影响：
        None outside the fixtures.
        夹具之外无副作用。
    """
    hms = _hms(provider, tmp_path)
    store, index = hms.store, hms.index
    try:
        mems = [hms.add(t, user_id=U, infer=False)["results"][0]["id"] for t in TEXTS]
        target = store.resolve_head(mems[1]).atom_id
        query = hms.embedder.embed("tool number 1", "search")
        assert target in _recalled(hms, query)
        calls: list[tuple[str, dict]] = []
        original = index.update_attrs

        def slow(atom_id: str, attrs: dict) -> None:
            """The provider round trip of the archive change is slow (remote index, GC pause).
            归档变化的提供者往返很慢（远程索引、GC 停顿）。
            """
            if attrs.get("archived") is True:
                time.sleep(0.25)
            calls.append((atom_id, dict(attrs)))
            original(atom_id, attrs)

        store.attr_listeners = [slow]
        now = store.now()
        t1 = threading.Thread(target=lambda: store.archive_atom(target, reason="deleted", at=now))
        t2 = threading.Thread(target=lambda: store.restore_atom(target, at=now))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(5)
        t2.join(5)
        assert not t1.is_alive() and not t2.is_alive()
        store.attr_listeners = [original]
        # Commit order is archive → restore; without ticketed dispatch the fast restore call overtakes the slow one.
        # 提交顺序是 归档 → 恢复；没有票号派发时，快的恢复调用会超过慢的归档调用。
        assert [c[1]["archived"] for c in calls] == [True, False]
        assert all(c[0] == target for c in calls)
        assert all(set(c[1]) == set(store.atom_attrs([target])[target]) for c in calls)
        assert calls[-1][1] == store.atom_attrs([target])[target]
        assert target in store.axis_ids(SCOPE, as_of=store.now())
        assert target in _recalled(hms, query)
    finally:
        hms.close()


def test_search_inside_transaction_never_deadlocks(clock: FrozenClock, ids: SequentialIds, tmp_path: Any) -> None:
    """A search inside store.transaction() and a concurrent plain search both finish (lock order store → index).
    store.transaction() 内的搜索与并发的普通搜索都能完成（锁序 store → index）。

    Input / 输入：
        clock, ids: conftest fixtures.
        clock、ids：conftest 夹具。

        tmp_path: A file database so the WAL / data_version path is exercised.
        tmp_path：文件库，使 WAL / data_version 路径被执行。

    Output / 输出：
        Assertions with a 5 s join timeout: both threads are done and the searches agree.
        带 5 s join 超时的断言：两个线程都完成，且搜索结果一致。

    State effect / 状态影响：
        Creates tmp_path/lock.db.
        创建 tmp_path/lock.db。
    """
    store = StateStore(str(tmp_path / "lock.db"), now=clock.now, id_factory=ids)
    index = SQLiteIndex({}, store=store, dims=4)
    try:
        for i in range(5):
            anchor = _anchor(store, f"t{i}", "2026-01-01T00:00:00Z")
            store.insert_atom(
                MemoryAtom(
                    atom_id=f"atm_{i}",
                    memory_id=f"mem_{i}",
                    atom_type="message",
                    content={"text": f"t{i}"},
                    text=f"t{i}",
                    time_anchor="2026-01-01T00:00:00Z",
                    scale_level="L1_micro",
                    source_kind="message",
                    source_id=anchor,
                    extracted_by="rule@obs-v1",
                    scope=SCOPE,
                    anchor_ids=[anchor],
                )
            )
            index.upsert(f"atm_{i}", [1.0, 0.0, 0.0, 0.0], embedder="x")
        query = [1.0, 0.0, 0.0, 0.0]
        expected = index.search(query, top_k=2)
        results: dict[str, Any] = {}

        def in_transaction() -> None:
            """Hold the store lock (an open transaction), then search — needs the index lock.
            持有 store 锁（打开的事务），再搜索——需要索引锁。
            """
            with store.transaction():
                store.set_meta("k", "v")
                time.sleep(0.3)
                results["B"] = index.search(query, top_k=2)

        def plain() -> None:
            """A plain search: index lock, then (cache sync) the store.
            普通搜索：索引锁，再（缓存同步）store。
            """
            time.sleep(0.1)
            results["A"] = index.search(query, top_k=2)

        tb = threading.Thread(target=in_transaction, daemon=True)
        ta = threading.Thread(target=plain, daemon=True)
        tb.start()
        ta.start()
        tb.join(5)
        ta.join(5)
        deadlocked = tb.is_alive() or ta.is_alive()
        assert not deadlocked, "lock-order inversion between StateStore and SQLiteIndex"
        assert results["A"] == results["B"] == expected
    finally:
        # A deadlocked pair still holds both locks: closing would hang the run, so leave the daemon threads to die
        # with the process (the assertion above has already failed the test).
        # 死锁的一对线程仍持有两把锁：关闭会挂起整个运行，因此把守护线程留给进程结束（上面的断言已经让测试失败）。
        if not (tb.is_alive() or ta.is_alive()):
            index.close()
            store.close()


@pytest.mark.parametrize("provider", ["sqlite", "memory", "milvus"])
def test_listener_registered_once_and_removed_on_close(provider: str, tmp_path: Any) -> None:
    """Two facades on one store register one listener each; HMS.close() of the first leaves exactly the other's.
    同一 store 上的两个门面各注册一个监听器；第一个的 HMS.close() 后恰好只剩另一个的。

    Input / 输入：
        provider: 'sqlite' | 'memory' | 'milvus' (milvus skipped without pymilvus).
        provider：'sqlite' | 'memory' | 'milvus'（缺少 pymilvus 时跳过）。

        tmp_path: pytest temporary directory for the milvus-lite file.
        tmp_path：存放 milvus-lite 文件的 pytest 临时目录。

    Output / 输出：
        Assertions on store.attr_listeners after construction, after the first close() and after the second.
        对构造后、第一次 close() 后与第二次 close() 后 store.attr_listeners 的断言。

    State effect / 状态影响：
        None outside the fixtures (the shared store is closed by the first facade; close() is idempotent).
        夹具之外无副作用（共享 store 由第一个门面关闭；close() 幂等）。
    """
    first = _hms(provider, tmp_path)
    store = first.store
    env: dict[str, Any] = {}
    if provider == "milvus":
        env = {"config": {"uri": str(tmp_path / "sync.db"), "index_type": "FLAT", "scalar_index": "none"}}
    second = HMS(HMSConfig(vector_store={"provider": provider, **env}, store={"path": ":memory:"}), store=store)
    owners = [getattr(cb, "__self__", None) for cb in store.attr_listeners]
    # SQLiteIndex subscribes itself in its constructor; HMS must not add a second copy of the same bound method.
    # SQLiteIndex 在构造函数里自行订阅；HMS 不得再加同一个绑定方法的第二份。
    assert owners == [first.index, second.index]
    first.close()
    assert [getattr(cb, "__self__", None) for cb in store.attr_listeners] == [second.index]
    second.close()
    assert store.attr_listeners == []


def test_late_topic_and_entity_reach_the_keyword_channel(store: StateStore) -> None:
    """A topic (feedback 'object_important') or entity linked after insert is recalled by keyword_candidates.
    插入后链接的主题（反馈 'object_important'）或实体能被 keyword_candidates 召回。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

    Output / 输出：
        Assertions: the token is in atom_keyword_tokens AND in the keyword channel (no vocabulary asymmetry).
        断言：词元既在 atom_keyword_tokens 中也在关键词通道中（没有词表不对称）。

    State effect / 状态影响：
        None outside the fixture.
        夹具之外无副作用。
    """
    if not store.capabilities.get("fts5"):
        pytest.skip("FTS5 unavailable")
    hms = HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}), store=store)
    try:
        memory_id = hms.add("Sarah patched retry.ts and the backoff regression is fixed.", user_id=U, infer=False)[
            "results"
        ][0]["id"]
        hms.feedback(memory_id, "object_important", reason="keep")
        atom = store.latest_version(memory_id)
        assert "important" in atom.topic_tags and "important" in atom_keyword_tokens(atom)
        as_of = store.now()
        recalled = [a for a, _s in store.keyword_candidates(SCOPE, ["important"], as_of=as_of, limit=5)]
        assert recalled == [atom.atom_id]
        entity_id = store.get_or_create_entity(SCOPE, normalized="zephyr", canonical="Zephyr", entity_type="PROPER")
        store.link_atom_entity(atom.atom_id, entity_id)
        recalled = [a for a, _s in store.keyword_candidates(SCOPE, ["zephyr"], as_of=as_of, limit=5)]
        assert recalled == [atom.atom_id]
        # An archived version never regains an FTS row through a late link.
        # 已归档版本不会因迟到的链接重新获得 FTS 行。
        store.archive_atom(atom.atom_id, reason="deleted", at=as_of)
        store.add_atom_topic(atom.atom_id, "later")
        assert store.keyword_candidates(SCOPE, ["later"], as_of=as_of, include_expired=True, limit=5) == []
    finally:
        store.attr_listeners = []
        hms.index.close()
