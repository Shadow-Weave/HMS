"""Regression tests for the small-N latency work: schema clone, migration order, hydration, witness memo, trace memo, checkpoint knob.
小规模延迟优化的回归测试：模式克隆、迁移顺序、水合、见证备忘、追踪备忘、checkpoint 旋钮。
"""

# Purpose (EN): Pin every behaviour the group-P latency changes rely on so that a regression fails loudly.
# 目的（中）：钉住 P 组延迟改动所依赖的每个行为，使回归立即失败。

# Input (EN): The shared clock/ids/store fixtures, tmp_path file databases and raw sqlite3 connections.
# 输入（中）：共享的 clock/ids/store 夹具、tmp_path 文件库与原始 sqlite3 连接。

# Output (EN): Assertions over statement counts, schema equality, hydration order, query plans and config plumbing.
# 输出（中）：对语句计数、模式相等、水合顺序、查询计划与配置贯通的断言。

# State effect (EN): Only temporary in-memory and tmp_path databases are written.
# 状态影响（中）：只写临时内存库与 tmp_path 文件库。

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from typing import Iterator

import pytest

from hms_elite import exceptions, models
from hms_elite.configs.base import HMSConfig
from hms_elite.memory import schema
from hms_elite.memory import storage as mstorage
from hms_elite.memory.storage import StateStore

_MASTER = "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master ORDER BY rowid"


def _scope(**kw) -> models.Scope:
    """Scope with user_id 'alice' unless overridden.
    默认 user_id 为 'alice' 的范围。
    """
    return models.Scope(**({"user_id": "alice"} | kw))


def _seed_event(store: StateStore, text: str, ts: str, scope: models.Scope | None = None):
    """Insert one message event and its anchor; return (event, anchor).
    插入一条消息事件及其锚点；返回 (event, anchor)。
    """
    ev = models.SourceEvent(scope=scope or _scope(), role="user", payload={"text": text}, timestamp=ts)
    stored, _ = store.insert_source_event(ev)
    return stored, store.get_or_create_anchor(stored)


def _atom(text: str, anchor_ids: list[str], **overrides) -> models.MemoryAtom:
    """A state atom witnessed by anchor_ids (not yet inserted).
    由 anchor_ids 见证的 state 原子（尚未插入）。
    """
    base = dict(
        atom_type="state",
        content={"text": text},
        text=text,
        scale_level="state",
        source_kind="observation",
        source_id="obs_" + text[:4],
        extracted_by="rule@obs-v1",
        scope=_scope(),
        anchor_ids=anchor_ids,
        obs_type="constraint",
        subject="project",
        predicate="must_not",
        time_anchor="2026-06-15T09:00:00Z",
    )
    base.update(overrides)
    return models.MemoryAtom(**base)


@contextlib.contextmanager
def _statements(store: StateStore) -> Iterator[list[str]]:
    """Collect the expanded SQL of every statement SQLite runs on the store's connection.
    收集 SQLite 在该 store 连接上执行的每条语句的展开 SQL。
    """
    seen: list[str] = []
    conn = store._conn
    assert conn is not None
    conn.set_trace_callback(seen.append)
    try:
        yield seen
    finally:
        conn.set_trace_callback(None)


# 1. Private ':memory:' databases clone the schema template; the result equals the DDL path and leaks nothing.
# 1. 私有 ':memory:' 库克隆模式模板；结果与 DDL 路径相同且不泄漏任何状态。
def test_memory_store_clones_template_identically(clock, ids, tmp_path, monkeypatch):
    """The clone is byte-for-byte the DDL schema, triggers fire, and instances share no rows.
    克隆与 DDL 模式逐字节相同、触发器生效、实例之间不共享任何行。

    Input / 输入：
        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

        ids: Sequential id factory fixture.
        ids：顺序 id 工厂夹具。

        tmp_path: Scratch directory for the file-database control.
        tmp_path：文件库对照所用的临时目录。

        monkeypatch: Used to prove which path each constructor takes.
        monkeypatch：用于证明每个构造函数走的是哪条路径。

    Output / 输出：
        Assertions over sqlite_master, user_version, meta, triggers, isolation and the template itself.
        对 sqlite_master、user_version、meta、触发器、隔离性与模板本身的断言。

    State effect / 状态影响：
        None beyond temporary databases.
        除临时库外无。
    """
    first = StateStore(":memory:", now=clock.now, id_factory=ids)
    raw = sqlite3.connect(":memory:", isolation_level=None)
    raw.execute("PRAGMA foreign_keys = ON")
    schema.apply_migrations(raw)
    assert [tuple(r) for r in first._conn.execute(_MASTER).fetchall()] == raw.execute(_MASTER).fetchall()
    assert first._conn.execute("PRAGMA page_count").fetchone()[0] == raw.execute("PRAGMA page_count").fetchone()[0]
    assert schema.current_version(first._conn) == schema.SCHEMA_VERSION
    assert first.capabilities["schema_version"] == schema.SCHEMA_VERSION
    assert first.get_meta("schema_created_at") == clock.now()
    assert first.stats()["meta"] == 1 and first.stats()["memory_atoms"] == 0
    raw.close()
    # Triggers travel with the clone: the append-only invariant still fires.
    # 触发器随克隆而来：只追加不变量依然触发。
    _ev, anc = _seed_event(first, "hello", "2026-06-15T09:00:00Z")
    with pytest.raises(exceptions.StoreError) as info:
        first._execute("DELETE FROM anchors WHERE anchor_id = ?", (anc.anchor_id,))
    assert info.value.error_code == "STORE_003"
    first.set_meta("k", "v")
    # Isolation: a second memory store starts empty and the template itself stays schema-only.
    # 隔离性：第二个内存 store 从空开始，模板本身始终只含模式。
    second = StateStore(":memory:", now=clock.now, id_factory=ids)
    assert second.get_meta("k") is None and second.stats()["anchors"] == 0 and second.stats()["meta"] == 1
    template = schema._TEMPLATE[2]
    assert template.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 0
    assert template.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 0
    # Which path runs: with the store-side DDL entry point disabled, ':memory:' and a FRESH file still open (clone)
    # while an existing file does not (the migration ladder decides its version).
    # 走的是哪条路径：禁用 store 侧的 DDL 入口后，':memory:' 与**全新**文件仍能打开（克隆），已有文件则不能
    # （由迁移阶梯判定其版本）。
    existing = tmp_path / "existing.db"
    StateStore(str(existing), now=clock.now, id_factory=ids).close()

    def boom(conn, *, created_at=None):  # noqa: ARG001 - signature parity / 签名一致
        raise AssertionError("apply_migrations must not run for a fresh database")

    monkeypatch.setattr(mstorage, "apply_migrations", boom)
    monkeypatch.setattr(schema, "apply_migrations", boom)
    third = StateStore(":memory:", now=clock.now, id_factory=ids)
    assert third.stats()["memory_atoms"] == 0 and third.get_meta("schema_created_at") == clock.now()
    fresh_file = StateStore(str(tmp_path / "file.db"), now=clock.now, id_factory=ids)
    assert fresh_file.stats()["memory_atoms"] == 0 and fresh_file.get_meta("schema_created_at") == clock.now()
    fresh_file.close()
    with pytest.raises(AssertionError):
        StateStore(str(existing), now=clock.now, id_factory=ids)
    for store in (first, second, third):
        store.close()


# 2. File databases: migrations run before the WAL switch and the creation stamp rides in the migration txn.
# 2. 文件库：迁移先于 WAL 切换执行，创建时间戳搭乘迁移事务。
def test_file_store_migrates_before_wal_switch(tmp_path, clock, ids):
    """A fresh file store ends in WAL mode with no schema pages in the WAL; created_at is written once.
    新文件 store 最终处于 WAL 模式且 WAL 中没有模式页；created_at 只写一次。

    Input / 输入：
        tmp_path: Scratch directory.
        tmp_path：临时目录。

        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

        ids: Sequential id factory fixture.
        ids：顺序 id 工厂夹具。

    Output / 输出：
        Assertions over journal_mode, WAL size, meta and apply_migrations(created_at=...).
        对 journal_mode、WAL 大小、meta 与 apply_migrations(created_at=...) 的断言。

    State effect / 状态影响：
        Writes tmp_path databases.
        写入 tmp_path 数据库。
    """
    path = tmp_path / "hms.db"
    store = StateStore(str(path), now=clock.now, id_factory=ids)
    assert store.capabilities["wal"] is True and store.capabilities["schema_version"] == schema.SCHEMA_VERSION
    assert store.get_meta("schema_created_at") == clock.now()
    # The cloned file equals a file built by the migration ladder: sqlite_master rows, user_version, page_count.
    # 克隆出的文件与迁移阶梯建出的文件相同：sqlite_master 行、user_version、page_count。
    raw = sqlite3.connect(str(tmp_path / "ladder.db"), isolation_level=None)
    raw.execute("PRAGMA foreign_keys = ON")
    schema.apply_migrations(raw, created_at=clock.now())
    assert [tuple(r) for r in store._conn.execute(_MASTER).fetchall()] == raw.execute(_MASTER).fetchall()
    assert schema.current_version(store._conn) == schema.current_version(raw) == schema.SCHEMA_VERSION
    assert store._conn.execute("PRAGMA page_count").fetchone()[0] == raw.execute("PRAGMA page_count").fetchone()[0]
    raw.close()
    wal = tmp_path / "hms.db-wal"
    # The 97 schema pages went straight into hms.db: the WAL is absent or holds no schema pages (< 8 frames).
    # 97 个模式页直接进入 hms.db：WAL 不存在或不含模式页（少于 8 帧）。
    assert (not wal.exists()) or wal.stat().st_size < 8 * 4096
    assert path.stat().st_size >= 90 * 4096
    store.set_meta("k", "v")
    store.close()
    reopened = StateStore(str(path), now=clock.now, id_factory=ids)
    assert reopened.get_meta("k") == "v" and reopened.get_meta("schema_created_at") == clock.now()
    reopened.close()
    # apply_migrations(created_at=...) stamps a fresh database inside migration 1 and never re-stamps.
    # apply_migrations(created_at=...) 在迁移 1 内为新库打时间戳，且从不重复打戳。
    raw = sqlite3.connect(str(tmp_path / "raw.db"), isolation_level=None)
    raw.execute("PRAGMA foreign_keys = ON")
    assert schema.apply_migrations(raw, created_at="X") == schema.SCHEMA_VERSION
    assert raw.execute("SELECT value FROM meta WHERE key = 'schema_created_at'").fetchone()[0] == "X"
    assert raw.in_transaction is False
    schema.apply_migrations(raw, created_at="Y")
    assert raw.execute("SELECT value FROM meta WHERE key = 'schema_created_at'").fetchone()[0] == "X"
    raw.close()


# 3. TableTrace memoizes the table set per statement text; behaviour is unchanged.
# 3. TableTrace 按语句文本记忆表集合；行为不变。
def test_table_trace_memo(store):
    """_tables_of returns the cached frozenset for repeated text and the trace still reports every table.
    _tables_of 对重复文本返回缓存的 frozenset，追踪仍报告每张表。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

    Output / 输出：
        Identity of the memoized value, trace contents and the bound on the memo.
        备忘值的同一性、追踪内容与备忘上限。

    State effect / 状态影响：
        Clears and refills the module memo.
        清空并重新填充模块备忘。
    """
    sql = "SELECT ae.atom_id FROM atom_entity ae JOIN entities e ON e.entity_id = ae.entity_id"
    first = mstorage._tables_of(sql)
    assert first == frozenset({"atom_entity", "entities"}) and mstorage._tables_of(sql) is first
    store.trace.reset()
    store.trace.record(sql)
    store.trace.record("INSERT INTO meta(key, value) VALUES ('a','b')")
    assert store.trace.tables() == {"atom_entity", "entities", "meta"}
    mstorage._TABLES_OF.clear()
    for i in range(mstorage._TABLES_OF_LIMIT):
        mstorage._tables_of(f"SELECT {i} FROM meta")
    assert len(mstorage._TABLES_OF) == mstorage._TABLES_OF_LIMIT
    mstorage._tables_of("SELECT 1 FROM anchors")
    assert len(mstorage._TABLES_OF) == 1


# 4. Hydration reads the three axis links in one statement and keeps the contract order.
# 4. 水合用一条语句读取三种轴链接并保持契约顺序。
def test_hydration_single_statement_and_order(store, clock):
    """get_atom runs exactly two statements; anchors primary-first, topics insertion order, entities sorted.
    get_atom 恰好执行两条语句；锚点 primary 在前、主题按插入顺序、实体排序。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

    Output / 输出：
        Statement counts and hydrated lists.
        语句计数与水合后的列表。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    _, a1 = _seed_event(store, "one", "2026-06-15T09:00:00Z")
    _, a2 = _seed_event(store, "two", "2026-06-15T09:01:00Z")
    _, a3 = _seed_event(store, "three", "2026-06-15T09:02:00Z")
    keys = [("zeta", "Zeta", "PROPER", ()), ("alpha", "Alpha", "PROPER", ("Alp",)), ("zeta", "Zeta", "PROPER", ())]
    stored = store.insert_atom(
        _atom("links", [a2.anchor_id, a1.anchor_id, a3.anchor_id, a1.anchor_id], topic_tags=["t0"]),
        entity_keys=keys,
        topics=["t2", "t1", "t2"],
    )
    other = store.insert_atom(_atom("plain", [a1.anchor_id]))
    with _statements(store) as seen:
        atom = store.get_atom(stored.atom_id)
    assert len(seen) == 2 and seen[0].startswith("SELECT * FROM memory_atoms") and seen[1].startswith("WITH ids")
    assert atom.anchor_ids == [a2.anchor_id, a1.anchor_id, a3.anchor_id]
    assert atom.entity_ids == ["alpha", "zeta"] and atom.topic_tags == ["t2", "t1", "t0"]
    with _statements(store) as seen:
        both = store.get_atoms([stored.atom_id, other.atom_id, stored.atom_id])
    assert [a.atom_id for a in both] == [stored.atom_id, other.atom_id]
    assert sum(1 for s in seen if s.startswith("WITH ids")) == 1
    assert both[1].anchor_ids == [a1.anchor_id] and both[1].entity_ids == [] and both[1].topic_tags == []
    # Disputed witnesses drop out; when every witness is disputed the full list comes back (fallback query).
    # 被争议的见证被排除；全部见证被争议时回退查询返回完整列表。
    store.dispute_atom_anchor(stored.atom_id, a1.anchor_id, at=clock.now())
    assert store.get_atom(stored.atom_id).anchor_ids == [a2.anchor_id, a3.anchor_id]
    store.dispute_atom_anchor(other.atom_id, a1.anchor_id, at=clock.now())
    assert store.get_atom(other.atom_id).anchor_ids == [a1.anchor_id]


# 5. insert_atom returns exactly what get_atom would read, without re-reading.
# 5. insert_atom 返回的正是 get_atom 会读到的内容，而无需重新读取。
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"content": {"text": "mirror", "objects": ("b", "a"), "nested": {"z": 1, "a": [1.0, 2]}}},
        {"user_meta": {"zeta": 1, "alpha": {"k": (1, 2)}}, "source_ep_ids": ["ep_2", "ep_1"]},
        {"valid_to": "2026-12-31T00:00:00Z", "confidence": 1, "archived_reason": "revised", "version": 3},
        {"topic_tags": ["shared", "own"], "role": "user", "actor_id": "u1", "attributed_to": "user"},
    ],
)
def test_insert_atom_mirrors_read_path(store, overrides):
    """insert_atom == get_atom for every shape, and insert_atom issues no memory_atoms SELECT.
    对每种形状 insert_atom == get_atom，且 insert_atom 不发出任何 memory_atoms SELECT。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        overrides: Atom field overrides for this case.
        overrides：本用例的原子字段覆盖。

    Output / 输出：
        Model equality, public-row equality and the absence of read-back statements.
        模型相等、公开行相等以及不存在回读语句。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    _, a1 = _seed_event(store, "one", "2026-06-15T09:00:00Z")
    _, a2 = _seed_event(store, "two", "2026-06-15T09:01:00Z")
    atom = _atom("mirror", [a1.anchor_id, a2.anchor_id, a1.anchor_id, a2.anchor_id], **overrides)
    keys = [("beta", "Beta", "PROPER", ()), ("alpha", "Alpha", "IDENTIFIER", ("A",)), ("beta", "Beta", "PROPER", ())]
    with _statements(store) as seen:
        stored = store.insert_atom(atom, entity_keys=keys, topics=["shared", "extra"])
    assert not any(s.startswith(("SELECT * FROM memory_atoms", "WITH ids")) for s in seen)
    read = store.get_atom(stored.atom_id)
    assert read is not None and stored == read
    assert stored.to_public_dict() == read.to_public_dict()
    assert stored.anchor_ids == [a1.anchor_id, a2.anchor_id] and stored.entity_ids == ["alpha", "beta"]
    assert stored.model_dump() == read.model_dump()


# 6. The witness memo lives for one outermost transaction only.
# 6. 见证备忘只在一个最外层事务内存活。
def test_witness_memo_scoped_to_transaction(store, clock):
    """A second pre-check of the same anchor inside one transaction issues no SELECT; rollback clears the memo.
    同一事务内对同一锚点的第二次预检不发出 SELECT；回滚清空备忘。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

    Output / 输出：
        SELECT counts inside and outside transactions and the post-rollback WitnessMissingError.
        事务内外的 SELECT 计数以及回滚后的 WitnessMissingError。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    _, anc = _seed_event(store, "one", "2026-06-15T09:00:00Z")

    def anchor_selects(seen: list[str]) -> int:
        return sum(1 for s in seen if s.startswith("SELECT * FROM anchors WHERE anchor_id IN"))

    with _statements(store) as seen:
        store._require_anchors([anc.anchor_id], obj="x", obj_id=None)
        store._require_anchors([anc.anchor_id], obj="x", obj_id=None)
    assert anchor_selects(seen) == 2 and store._verified_anchors == set()
    with _statements(store) as seen, store.transaction():
        store._require_anchors([anc.anchor_id], obj="x", obj_id=None)
        store._require_anchors([anc.anchor_id, anc.anchor_id], obj="y", obj_id=None)
        with pytest.raises(exceptions.WitnessMissingError) as info:
            store._require_anchors([anc.anchor_id, "anc_missing"], obj="z", obj_id=None)
        assert info.value.details["missing"] == ["anc_missing"]
    assert anchor_selects(seen) == 2 and store._verified_anchors == set()
    # An anchor written in a rolled-back transaction is not proven afterwards.
    # 在被回滚事务中写入的锚点事后不视为已证明。
    with pytest.raises(RuntimeError):
        with store.transaction():
            _, gone = _seed_event(store, "rolled back", "2026-06-15T10:00:00Z")
            assert gone.anchor_id in store._verified_anchors
            raise RuntimeError("abort")
    assert store._verified_anchors == set()
    with store.transaction():
        with pytest.raises(exceptions.WitnessMissingError):
            store._require_anchors([gone.anchor_id], obj="x", obj_id=None)
    # A history row after an insert in the same transaction re-uses the proof (one SELECT for both).
    # 同一事务内插入后再写历史行复用证明（两者只有一次 SELECT）。
    with _statements(store) as seen, store.transaction():
        stored = store.insert_atom(_atom("memo", [anc.anchor_id]))
        store.add_history(
            memory_id=stored.memory_id, old_memory=None, new_memory=stored.text, event="ADD",
            created_at=stored.created_at, updated_at=stored.created_at, anchor_id=anc.anchor_id,
            atom_id=stored.atom_id,
        )
    assert anchor_selects(seen) == 1


# 6b. Only the thread that owns the transaction may read the memo; other threads keep their own pre-check.
# 6b. 只有拥有事务的线程能读备忘；其它线程保留自己的预检。
def test_witness_memo_is_owner_thread_only(store):
    """A second thread never trusts an uncommitted witness: it blocks on the lock and then re-checks.
    第二个线程从不信任未提交的见证：它阻塞在锁上，之后重新检查。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

    Output / 输出：
        The checker thread neither returns early nor accepts the rolled-back witness.
        检查线程既不会提前返回，也不会接受被回滚的见证。

    State effect / 状态影响：
        Writes (and rolls back) the in-memory store from two threads.
        从两个线程写入（并回滚）内存 store。
    """
    started, release = threading.Event(), threading.Event()
    outcome: dict[str, str] = {}

    def writer() -> None:
        try:
            with store.transaction():
                _, gone = _seed_event(store, "uncommitted", "2026-06-15T11:00:00Z")
                outcome["id"] = gone.anchor_id
                started.set()
                release.wait(5)
                raise RuntimeError("abort")
        except RuntimeError:
            pass

    def checker() -> None:
        started.wait(5)
        try:
            store._require_anchors([outcome["id"]], obj="x", obj_id=None)
            outcome["checker"] = "accepted"
        except exceptions.WitnessMissingError:
            outcome["checker"] = "missing"

    threads = [threading.Thread(target=writer), threading.Thread(target=checker)]
    for t in threads:
        t.start()
    started.wait(5)
    time.sleep(0.2)
    # Still blocked on the lock: the writer's memo was not consulted from another thread.
    # 仍阻塞在锁上：另一线程没有读取写入者的备忘。
    assert "checker" not in outcome
    release.set()
    for t in threads:
        t.join(5)
    assert outcome["checker"] == "missing" and store._verified_anchors == set()


# 7. StoreConfig.checkpoint_on_close reaches the store and controls the WAL at close().
# 7. StoreConfig.checkpoint_on_close 贯通到 store 并控制 close() 时的 WAL。
def test_checkpoint_on_close_knob(tmp_path, clock, ids):
    """Default True removes the WAL on close; False keeps it (when the runtime can) and the data stays readable.
    默认 True 在关闭时移除 WAL；False（运行时支持时）保留它且数据仍可读。

    Input / 输入：
        tmp_path: Scratch directory.
        tmp_path：临时目录。

        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

        ids: Sequential id factory fixture.
        ids：顺序 id 工厂夹具。

    Output / 输出：
        Config plumbing and WAL file presence after close().
        配置贯通与 close() 后 WAL 文件的存在情况。

    State effect / 状态影响：
        Writes tmp_path databases.
        写入 tmp_path 数据库。
    """
    assert HMSConfig().store.checkpoint_on_close is True
    cfg = HMSConfig.from_dict({"store": {"path": str(tmp_path / "cfg.db"), "checkpoint_on_close": False}})
    assert cfg.store.checkpoint_on_close is False
    from hms_elite import HMS

    hms = HMS(cfg)
    assert hms.store._checkpoint_on_close is False
    hms.close()
    keep = StateStore(str(tmp_path / "a.db"), now=clock.now, id_factory=ids)
    keep.set_meta("k", "v")
    keep.close()
    assert not (tmp_path / "a.db-wal").exists()
    memory = StateStore(":memory:", now=clock.now, id_factory=ids, checkpoint_on_close=False)
    memory.set_meta("k", "v")
    memory.close()
    if not hasattr(sqlite3.Connection, "setconfig") or not hasattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE"):
        pytest.skip("SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE needs Python >= 3.12")
    skip = StateStore(str(tmp_path / "b.db"), now=clock.now, id_factory=ids, checkpoint_on_close=False)
    skip.set_meta("k", "v")
    skip.close()
    assert (tmp_path / "b.db-wal").exists() and (tmp_path / "b.db-wal").stat().st_size > 0
    again = StateStore(str(tmp_path / "b.db"), now=clock.now, id_factory=ids)
    assert again.get_meta("k") == "v"
    again.close()


# 8. Scope predicates name only the set keys so the planner can use ix_atoms_scope; results are unchanged.
# 8. 范围谓词只包含已设置的键，规划器因此能使用 ix_atoms_scope；结果不变。
def test_scope_where_uses_scope_index(store):
    """_scope_where emits `col = ?` per set key and axis_select plans through ix_atoms_scope.
    _scope_where 为每个已设置键产生 `col = ?`，axis_select 经 ix_atoms_scope 规划。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

    Output / 输出：
        SQL fragments, query plans and scoped results.
        SQL 片段、查询计划与按范围的结果。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    assert mstorage._scope_where(models.Scope(user_id="u"), "memory_atoms.") == ("memory_atoms.user_id = ?", ["u"])
    assert mstorage._scope_where(models.Scope(user_id="u", run_id="r")) == ("user_id = ? AND run_id = ?", ["u", "r"])
    assert mstorage._scope_where(models.Scope()) == ("1=1", [])
    assert mstorage._scope_where(None) == ("1=1", [])
    _, a_run1 = _seed_event(store, "r1", "2026-06-15T09:00:00Z", _scope(run_id="run-1"))
    _, a_run2 = _seed_event(store, "r2", "2026-06-15T09:01:00Z", _scope(run_id="run-2"))
    _, a_bob = _seed_event(store, "bob", "2026-06-15T09:02:00Z", models.Scope(user_id="bob"))
    one = store.insert_atom(_atom("in run 1", [a_run1.anchor_id], scope=_scope(run_id="run-1")))
    two = store.insert_atom(_atom("in run 2", [a_run2.anchor_id], scope=_scope(run_id="run-2")))
    store.insert_atom(_atom("bob's", [a_bob.anchor_id], scope=models.Scope(user_id="bob")))
    with _statements(store) as seen:
        alice = store.axis_select(_scope(), order="atom_id ASC")
    assert [a.atom_id for a in alice] == [one.atom_id, two.atom_id]
    assert [a.atom_id for a in store.axis_select(_scope(run_id="run-2"))] == [two.atom_id]
    assert store.count_atoms(models.Scope(user_id="bob")) == 1 and store.count_atoms(models.Scope(run_id="run-1")) == 1
    plan = " ".join(r[3] for r in store._conn.execute("EXPLAIN QUERY PLAN " + seen[0]).fetchall())
    assert "ix_atoms_scope" in plan
