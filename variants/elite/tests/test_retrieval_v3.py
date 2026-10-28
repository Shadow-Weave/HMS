"""Retrieval round 3 regressions: unordered probes, gated channels, index cache, semantic reuse, hints, clones, config flow.
检索第三轮回归：不排序探测、门控内通道、索引缓存、语义分复用、索引提示、模板克隆、配置流转。
"""

# Purpose (EN): Every behaviour change of RETRIEVAL_BENCH_2026-09-10 has a test that fails without it: the store's
#               write generations and index cache, the channels searching inside G, the recency rule, the two-phase
#               keyword channel, the entity-gate enumeration, the time/obs_type/scale index hints, the in-pool reuse
#               of channel scores, AxisResult.retrieval, the facade passing RetrievalConfig, attrs on ingest upserts,
#               atom_type pushdown and the Milvus reopen.
# 目的（中）：RETRIEVAL_BENCH_2026-09-10 的每项行为改动都有一条"没有它就会失败"的测试：store 的写入代数与索引缓存、
#             在 G 内搜索的通道、新近度规则、两阶段关键词通道、实体门枚举、时间/obs_type/尺度索引提示、
#             池内复用通道分数、AxisResult.retrieval、门面传递 RetrievalConfig、摄取 upsert 携带标量、
#             atom_type 下推与 Milvus 重新打开。

# Input (EN): conftest fixtures (store, clock, ids, hash_embedder, milvus_uri, tmp_path) and test_retrieval's corpus.
# 输入（中）：conftest 夹具（store、clock、ids、hash_embedder、milvus_uri、tmp_path）与 test_retrieval 的语料。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory / tmp databases (no network, no home writes).
# 状态影响（中）：除内存 / tmp 数据库外无副作用（无网络、不写用户目录）。

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest

from hms_elite.configs.base import HMSConfig, RetrievalConfig, ScoringConfig
from hms_elite.exceptions import ValidationError
from hms_elite.index import IndexFilter, atom_index_attrs
from hms_elite.index.base import rank_hits, select_top_k
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.memory import schema
from hms_elite.memory.storage import StateStore
from hms_elite.models import AxisQuery, Scope
from hms_elite.mvmi import query as mquery
from hms_elite.mvmi.filters import In, lower_to_sql
from hms_elite.mvmi.planner import plan_query
from hms_elite.mvmi.query import GATE_CHANNEL, execute_axis_query, recall_pool
from hms_elite.utils.factory import IndexFactory
from hms_elite.utils.scoring import semantic_scores
from tests.test_retrieval import SCOPE, U, _aq, _new_rows, seed_corpus

ROOT = pathlib.Path(__file__).resolve().parent.parent
AS_OF_2026 = "2026-06-15T09:00:00Z"


def _traced(store: StateStore) -> tuple[list[str], Any]:
    """Attach a trace callback collecting statement texts; returns (list, detach).
    挂上收集语句文本的追踪回调；返回 (列表, 解除函数)。
    """
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    return seen, lambda: store._conn.set_trace_callback(None)


def _plan(store: StateStore, sql: str) -> str:
    """EXPLAIN QUERY PLAN text of one traced statement (the trace already expanded its bound parameters).
    一条追踪到的语句的 EXPLAIN QUERY PLAN 文本（追踪已展开其绑定参数）。
    """
    return " | ".join(r[3] for r in store._conn.execute("EXPLAIN QUERY PLAN " + sql))


# ---------------------------------------------------------------- store: probes, hints, channels inside G
# ---------------------------------------------------------------- store：探测、提示、G 内通道
def test_axis_ids_unordered_and_index_hints(store, clock, index_provider, hash_embedder):
    """order=None emits no ORDER BY; a hint drives memory_atoms through the named index; a bad hint is rejected.
    order=None 不产生 ORDER BY；提示让 memory_atoms 经指名索引读取；非法提示被拒绝。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on the traced SQL and its plans.
        对追踪到的 SQL 及其执行计划的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 40)
    scope = Scope(**SCOPE)
    as_of = store.now()
    seen, detach = _traced(store)
    try:
        probe = store.axis_ids(scope, order=None, limit=5, as_of=as_of)
        ordered = store.axis_ids(scope, limit=5, as_of=as_of)
    finally:
        detach()
    assert len(probe) == 5 and len(ordered) == 5 and set(ordered) <= set(store.axis_ids(scope, as_of=as_of))
    probe_sql = [s for s in seen if s.startswith("SELECT memory_atoms.atom_id") and "ORDER BY" not in s]
    assert probe_sql and all("LIMIT" in s for s in probe_sql)
    assert any("ORDER BY" in s for s in seen)
    # The time hint: the plan names ix_atoms_time and the result equals the unhinted gate.
    # 时间提示：执行计划指名 ix_atoms_time，且结果与不带提示的门控相同。
    where = "memory_atoms.time_anchor BETWEEN ? AND ?"
    params = ["2026-03-01T00:00:00.000000+00:00", "2026-03-31T23:59:59.999999+00:00"]
    hinted = store.axis_ids(scope, where_sql=where, params=params, as_of=as_of, index_hint=schema.TIME_INDEX)
    assert hinted == store.axis_ids(scope, where_sql=where, params=params, as_of=as_of)
    assert store.count_axis(scope, where_sql=where, params=params, as_of=as_of, index_hint=schema.TIME_INDEX) == len(hinted)
    seen, detach = _traced(store)
    try:
        store.axis_ids(scope, where_sql=where, params=params, as_of=as_of, index_hint=schema.TIME_INDEX)
    finally:
        detach()
    hinted_sql = [s for s in seen if "INDEXED BY ix_atoms_time" in s]
    assert hinted_sql and "ix_atoms_time" in _plan(store, hinted_sql[0])
    with pytest.raises(ValidationError):
        store.axis_ids(scope, as_of=as_of, index_hint="ix_atoms_time; DROP TABLE meta")


def test_channels_search_inside_the_level_gate(store, clock, index_provider, hash_embedder):
    """keyword / entity candidates honour the level gate: only gated atoms, found even when the scope-wide top-K misses them.
    关键词 / 实体候选遵守级别门控：只返回门控内原子，即使 scope 范围的前 K 个漏掉它们也能找到。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on channel membership under a March time gate with |G| > K.
        三月时间门（|G| > K）下对通道成员的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 300)
    scope = Scope(**SCOPE)
    as_of = store.now()
    plan = plan_query("what happened in March 2026", as_of=as_of).as_dict()
    clauses, params, names = mquery._plan_gates(plan, scope, store.capabilities)
    where = " AND ".join(f"({c})" for c in clauses)
    gate = set(store.axis_ids(scope, where_sql=where, params=params, as_of=as_of))
    assert names == ["time"] and len(gate) > 3
    for tokens in (["sarah"], ["retry.ts", "backoff"], ["billing"]):
        scope_wide = [a for a, _s in store.keyword_candidates(scope, tokens, as_of=as_of, limit=3)]
        gated = [a for a, _s in store.keyword_candidates(scope, tokens, as_of=as_of, limit=3, where_sql=where, params=params)]
        assert set(gated) <= gate
        matches_in_gate = [a for a, _s in store.keyword_candidates(scope, tokens, as_of=as_of, limit=10**6) if a in gate]
        assert len(gated) == min(3, len(matches_in_gate))
        # The scope-wide top-3 is not a substitute: with |G| > K it usually lies outside the gate entirely.
        # scope 范围的前 3 个不能替代：|G| > K 时它通常完全落在门控之外。
        assert [a for a in scope_wide if a in gate] != gated or len(matches_in_gate) <= 3
    wide = store.entity_candidates(scope, ["sarah"], as_of=as_of, limit=3)
    gated_entity = store.entity_candidates(scope, ["sarah"], as_of=as_of, limit=3, where_sql=where, params=params)
    all_gated = store.entity_candidates(scope, ["sarah"], as_of=as_of, limit=None, where_sql=where, params=params)
    assert set(gated_entity) <= gate and gated_entity == all_gated[:3] and len(set(all_gated)) == len(all_gated)
    assert all_gated == [a for a in store.entity_candidates(scope, ["sarah"], as_of=as_of, limit=None) if a in gate]
    assert set(wide) - gate or len(all_gated) <= 3
    # Pool level: a tiny channel-only config still recalls keyword / entity atoms from inside the gate.
    # 池层面：极小的仅通道配置仍能从门控内召回关键词 / 实体原子。
    tiny = RetrievalConfig(k_semantic=2, k_keyword=3, k_entity=3, k_recency=2, pool_max_factor=1, hard_gate_pool=False)
    pool = recall_pool(
        store,
        index,
        hash_embedder,
        plan_query("what did Sarah do in March 2026", as_of=as_of).as_dict(),
        scope,
        as_of=as_of,
        include_expired=False,
        cfg=tiny,
        top_k=3,
        text="what did Sarah do in March 2026",
        where_sql=where,
        params=params,
    )
    assert pool.ids and set(pool.ids) <= gate and not pool.whole
    assert "keyword" in pool.channel_sizes or "entity" in pool.channel_sizes


def test_entity_gate_is_a_non_correlated_subquery(store, clock, index_provider, hash_embedder):
    """The plan's hard entity gate lowers to `atom_id IN (subquery)` and selects exactly the filter DSL's EXISTS set.
    规划的硬实体门降级为 `atom_id IN (子查询)`，选出的集合与过滤 DSL 的 EXISTS 完全相同。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on the clause shape, its plan and the id set.
        对子句形状、执行计划与 id 集合的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 120)
    scope = Scope(**SCOPE)
    as_of = store.now()
    plan = plan_query("who is Sarah", as_of=as_of).as_dict()
    assert plan.get("hard_entity_gate")
    clauses, params, names = mquery._plan_gates(plan, scope, store.capabilities)
    assert names == ["entity"] and clauses[0].startswith("memory_atoms.atom_id IN (SELECT ae.atom_id")
    via_gate = store.axis_ids(scope, where_sql=clauses[0], params=params, as_of=as_of)
    sql, dsl_params, _ = lower_to_sql(
        In(field="entity", values=[str(k) for k in plan["hard_entity_gate"]]), capabilities=store.capabilities, scope=scope
    )
    assert via_gate == store.axis_ids(scope, where_sql=sql, params=dsl_params, as_of=as_of) and via_gate
    assert "sarah" in store.get_atom(via_gate[0]).entity_ids
    # G of an entity-gated level is enumerated once through atom_entity and feeds the entity channel.
    # 实体门级别的 G 经 atom_entity 一次枚举，并供实体通道使用。
    # pool_max (2) < |G|: channels with the whole-gate rule on, so recency stays off (no probe, no ordered LIMIT).
    # pool_max（2）< |G|：整体入池规则开启下走通道，因此新近度不运行（无探测、无排序 LIMIT）。
    tiny = RetrievalConfig(k_semantic=2, k_keyword=2, k_entity=4, k_recency=2, pool_max_factor=1)
    seen, detach = _traced(store)
    try:
        pool = recall_pool(
            store,
            index,
            hash_embedder,
            plan,
            scope,
            as_of=as_of,
            include_expired=False,
            cfg=tiny,
            top_k=2,
            text="who is Sarah",
            where_sql=clauses[0],
            params=params,
            gate_entity_keys=[str(k) for k in plan["hard_entity_gate"]],
        )
    finally:
        detach()
    assert pool.gate_count == len(via_gate) and set(pool.ids) <= set(via_gate)
    assert pool.channel_sizes["entity"] == min(4, len(via_gate))
    assert sum(1 for s in seen if s.startswith("SELECT memory_atoms.atom_id AS atom_id FROM atom_entity ae")) == 1
    assert not any(s.startswith("SELECT memory_atoms.atom_id FROM memory_atoms") and "LIMIT" in s for s in seen)


def test_keyword_channel_two_phases(store, clock):
    """Atoms matching every token come first (AND phase); the OR phase only runs when the AND phase cannot fill K.
    匹配全部词元的原子在前（AND 阶段）；只有 AND 阶段填不满 K 时才运行 OR 阶段。

    Input / 输入：
        store, clock: conftest fixtures.
        store、clock：conftest 夹具。

    Output / 输出：
        Assertions on order, on the traced MATCH statements and on the LIKE fallback parity.
        对顺序、追踪到的 MATCH 语句与 LIKE 退路一致性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    pytest.importorskip("numpy")
    if not store.capabilities.get("fts5"):
        pytest.skip("FTS5 unavailable")
    from hms_elite.models import MemoryAtom
    from tests.test_retrieval import _anchor

    scope = Scope(**SCOPE)
    texts = {
        "atm_all": "backoff patch retry",
        "atm_two": "backoff patch",
        "atm_one1": "backoff backoff backoff backoff",
        "atm_one2": "patch patch patch",
    }
    for i, (atom_id, text) in enumerate(texts.items()):
        clock.tick(seconds=1)
        anchor = _anchor(store, text, f"2026-01-{10 + i:02d}T09:00:00Z")
        store.insert_atom(
            MemoryAtom(
                atom_id=atom_id,
                memory_id=atom_id.replace("atm", "mem"),
                atom_type="message",
                content={"text": text},
                text=text,
                time_anchor=f"2026-01-{10 + i:02d}T09:00:00Z",
                scale_level="L1_micro",
                source_kind="message",
                source_id=anchor,
                extracted_by="rule@obs-v1",
                scope=scope,
                anchor_ids=[anchor],
            )
        )
    as_of = store.now()
    seen, detach = _traced(store)
    try:
        two = [a for a, _s in store.keyword_candidates(scope, ["backoff", "patch", "retry"], as_of=as_of, limit=1)]
        four = [a for a, _s in store.keyword_candidates(scope, ["backoff", "patch", "retry"], as_of=as_of, limit=4)]
    finally:
        detach()
    assert two == ["atm_all"] and four[0] == "atm_all" and set(four) == set(texts)
    matches = [s for s in seen if "atom_fts MATCH" in s]
    assert len(matches) == 3 and '"backoff" AND "patch" AND "retry"' in matches[0] and " OR " in matches[-1]
    # Without FTS5 the bounded LIKE fallback still ranks the all-token atom first.
    # 没有 FTS5 时有界 LIKE 退路仍把匹配全部词元的原子排在最前。
    store.capabilities["fts5"] = False
    try:
        assert [a for a, _s in store.keyword_candidates(scope, ["backoff", "patch", "retry"], as_of=as_of, limit=1)] == ["atm_all"]
    finally:
        store.capabilities["fts5"] = True


def test_write_generations_and_data_version(store, clock, tmp_path):
    """data_generation counts every memory_atoms / atom_vectors write; rebuild_generation the removals; data_version other connections.
    data_generation 统计每次 memory_atoms / atom_vectors 写入；rebuild_generation 统计删除类写入；data_version 统计其它连接。

    Input / 输入：
        store, clock, tmp_path: conftest / pytest fixtures.
        store、clock、tmp_path：conftest / pytest 夹具。

    Output / 输出：
        Assertions on the counters across INSERT / UPDATE / DELETE / ROLLBACK / reset and a second connection.
        对 INSERT / UPDATE / DELETE / ROLLBACK / reset 与第二个连接下计数器的断言。

    State effect / 状态影响：
        Writes a tmp file database.
        写入 tmp 文件数据库。
    """
    from hms_elite.models import MemoryAtom
    from tests.test_retrieval import _anchor

    scope = Scope(**SCOPE)
    gen, rebuild = store.data_generation, store.rebuild_generation
    store.set_meta("k", "v")
    assert (store.data_generation, store.rebuild_generation) == (gen, rebuild)
    anchor = _anchor(store, "hello", AS_OF_2026)
    atom = store.insert_atom(
        MemoryAtom(
            atom_id="atm_g1", memory_id="mem_g1", atom_type="message", content={"text": "hello"}, text="hello",
            time_anchor=AS_OF_2026, scale_level="L1_micro", source_kind="message", source_id=anchor,
            extracted_by="rule@obs-v1", scope=scope, anchor_ids=[anchor],
        )
    )  # fmt: skip
    assert store.data_generation > gen and store.rebuild_generation == rebuild
    gen = store.data_generation
    store.upsert_vector(atom.atom_id, [1.0, 0.0], embedder="hash@2")
    assert store.data_generation == gen + 1 and store.rebuild_generation == rebuild
    store.archive_atom(atom.atom_id, reason="deleted", at=AS_OF_2026)
    assert store.data_generation > gen + 1 and store.rebuild_generation == rebuild
    gen = store.data_generation
    store.delete_vector(atom.atom_id)
    assert store.data_generation == gen + 1 and store.rebuild_generation == rebuild + 1
    gen, rebuild = store.data_generation, store.rebuild_generation
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.upsert_vector(atom.atom_id, [0.0, 1.0], embedder="hash@2")
            raise RuntimeError("abort")
    assert store.data_generation > gen and store.rebuild_generation == rebuild + 1
    rebuild = store.rebuild_generation
    store.reset()
    assert store.rebuild_generation == rebuild + 1
    # data_version only moves for another connection's commit.
    # data_version 只在其它连接提交时变化。
    path = tmp_path / "dv.db"
    first = StateStore(str(path), now=clock.now)
    before = first.data_version()
    first.set_meta("a", "1")
    assert first.data_version() == before
    second = StateStore(str(path), now=clock.now)
    second.set_meta("b", "2")
    second.close()
    assert first.data_version() != before
    first.close()


def test_memory_atoms_updates_stay_listener_covered():
    """Every UPDATE memory_atoms in storage.py touches only listener-covered or non-gate columns (the cache's premise).
    storage.py 中每条 UPDATE memory_atoms 只触及监听器覆盖的列或非门控列（缓存的前提）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the source text.
        对源码文本的断言。

    State effect / 状态影响：
        None.
        无。
    """
    source = (ROOT / "hms_elite" / "memory" / "storage.py").read_text(encoding="utf-8")
    gate_columns = {"user_id", "agent_id", "run_id", "time_anchor", "valid_from", "scale_level", "obs_type", "atom_type"}
    statements = re.findall(r"UPDATE memory_atoms SET ([^\"]+?) WHERE", source)
    assert statements
    for stmt in statements:
        columns = {part.split("=")[0].strip() for part in stmt.split(",")}
        assert not (columns & gate_columns), stmt
        if columns & {"valid_to", "archived_reason"}:
            assert "_queue_attrs" in source


# ---------------------------------------------------------------- sqlite index cache
# ---------------------------------------------------------------- sqlite 索引缓存
def _sqlite_parity(index: SQLiteIndex, store: StateStore, query: list[float], scope: Scope, as_of: str) -> None:
    """The cache equals the exhaustive ranking and the SQL gate, bit for bit.
    缓存与穷举排序、与 SQL 门控逐位相同。
    """
    everything = store.load_vectors(None)
    assert index.search(query, top_k=10**6) == rank_hits(query, everything, top_k=10**6)
    gate = store.axis_ids(scope, as_of=as_of)
    spec = IndexFilter(scope=scope, as_of=as_of)
    assert index.search(query, filter=spec, top_k=10**6) == index.search(query, candidate_ids=gate, top_k=10**6)
    assert index.count(spec) == len([i for i in gate if i in everything])


def test_sqlite_cache_follows_every_write(clock, ids, tmp_path):
    """Upsert / replace / delete / archive / restore / set_valid_to / rollback / clear / reset / a foreign commit all reach the cache.
    upsert / 替换 / 删除 / 归档 / 恢复 / set_valid_to / 回滚 / 清空 / reset / 其它连接的提交都到达缓存。

    Input / 输入：
        clock, ids, tmp_path: conftest / pytest fixtures (a file database so a second connection can commit).
        clock、ids、tmp_path：conftest / pytest 夹具（文件库，使第二个连接能提交）。

    Output / 输出：
        Parity assertions after every step, plus the incremental / rebuild accounting.
        每一步之后的一致性断言，以及增量 / 重建的核算。

    State effect / 状态影响：
        Writes a tmp file database.
        写入 tmp 文件数据库。
    """
    pytest.importorskip("numpy")
    from hms_elite.embeddings.hash import HashEmbedder
    from hms_elite.models import MemoryAtom
    from tests.test_retrieval import _anchor

    path = tmp_path / "cache.db"
    store = StateStore(str(path), now=clock.now, id_factory=ids)
    embedder = HashEmbedder({"embedding_dims": 8})
    index = SQLiteIndex({}, store=store, dims=8)
    scope = Scope(**SCOPE)
    atoms = []
    for i in range(12):
        clock.tick(seconds=1)
        text = f"note {i} about {'sarah' if i % 2 else 'billing'} item{i}"
        anchor = _anchor(store, text, f"2026-0{1 + i % 6}-{10 + i:02d}T09:00:00Z")
        atom = store.insert_atom(
            MemoryAtom(
                atom_id=f"atm_c{i:02d}", memory_id=f"mem_c{i:02d}", atom_type="message", content={"text": text},
                text=text, time_anchor=f"2026-0{1 + i % 6}-{10 + i:02d}T09:00:00Z", scale_level="L1_micro",
                source_kind="message", source_id=anchor, extracted_by="rule@obs-v1", scope=scope, anchor_ids=[anchor],
            )
        )  # fmt: skip
        atoms.append(atom)
        if i != 5:
            index.upsert(atom.atom_id, embedder.embed(text, "add"), embedder=embedder.identity)
    query = embedder.embed("sarah billing note", "search")
    as_of = store.now()
    _sqlite_parity(index, store, query, scope, as_of)
    cache = index._caches[8]
    assert cache.n == 11 and cache.key is not None
    # Late vector (append) and a replaced vector (INSERT OR REPLACE → new rowid, same position).
    # 迟到的向量（追加）与被替换的向量（INSERT OR REPLACE → 新 rowid、同一位置）。
    index.upsert("atm_c05", embedder.embed("late", "add"), embedder=embedder.identity)
    index.upsert("atm_c01", embedder.embed("replaced text", "add"), embedder=embedder.identity)
    _sqlite_parity(index, store, query, scope, as_of)
    assert cache.n == 12 and cache.positions["atm_c01"] == 1
    # archive / restore / set_valid_to through the store listener; no HMS involved.
    # 经 store 监听器的归档 / 恢复 / set_valid_to；不经 HMS。
    store.archive_atom("atm_c03", reason="deleted", at=as_of)
    _sqlite_parity(index, store, query, scope, as_of)
    assert "atm_c03" not in {a for a, _s in index.search(query, filter=IndexFilter(scope=scope, as_of=as_of), top_k=100)}
    store.restore_atom("atm_c03", at=as_of)
    _sqlite_parity(index, store, query, scope, as_of)
    assert "atm_c03" in {a for a, _s in index.search(query, filter=IndexFilter(scope=scope, as_of=as_of), top_k=100)}
    store.set_valid_to("atm_c04", "2026-06-01T00:00:00Z", at=as_of)  # valid_from is its time_anchor 2026-05-14 / valid_from 即其 time_anchor 2026-05-14
    _sqlite_parity(index, store, query, scope, as_of)
    _sqlite_parity(index, store, query, scope, "2026-05-20T00:00:00Z")
    # A rolled-back upsert that was searched inside its transaction never survives.
    # 在事务内被搜索过、随后回滚的 upsert 绝不残留。
    with pytest.raises(RuntimeError):
        with store.transaction():
            index.upsert("atm_c00", embedder.embed("phantom", "add"), embedder=embedder.identity)
            index.search(query, top_k=3)
            raise RuntimeError("abort")
    _sqlite_parity(index, store, query, scope, as_of)
    assert index.get("atm_c00") == store.load_vectors(["atm_c00"])["atm_c00"]
    # Deletes and clears rebuild; a second connection's commit is seen through data_version.
    # 删除与清空触发重建；第二个连接的提交经 data_version 被察觉。
    index.delete("atm_c02")
    _sqlite_parity(index, store, query, scope, as_of)
    assert cache.n == 11
    other = StateStore(str(path), now=clock.now, id_factory=ids)
    other.upsert_vector("atm_c02", embedder.embed("from another connection", "add"), embedder=embedder.identity)
    other.close()
    _sqlite_parity(index, store, query, scope, as_of)
    assert cache.n == 12
    index.reset()
    assert index.search(query, top_k=5) == [] and index.count() == 0
    _sqlite_parity(index, store, query, scope, as_of)
    index.close()
    assert all(getattr(cb, "__self__", None) is not index for cb in store.attr_listeners)
    store.close()


def test_select_top_k_rows_and_scores_are_bit_equal(store):
    """select_top_k with rows= / scores= equals rank_hits over the same rows, including the zero tail.
    带 rows= / scores= 的 select_top_k 与同一批行上的 rank_hits 相同，包括零分尾部。

    Input / 输入：
        store: conftest fixture (unused; keeps the module offline-only).
        store：conftest 夹具（未使用；保持模块离线）。

    Output / 输出：
        Equality assertions over random matrices.
        随机矩阵上的相等断言。

    State effect / 状态影响：
        None.
        无。
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((200, 8)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [f"atm_{i:03d}" for i in range(200)]
    query = (rng.standard_normal(8) / 3).tolist()
    rows = np.asarray(sorted(rng.choice(200, 90, replace=False)), dtype="int64")
    scores = matrix.astype("float64")[rows] @ np.asarray(query)
    subset = {ids[r]: matrix[r].tolist() for r in rows}
    for top_k in (1, 5, 30, 89, 90, 500):
        assert select_top_k(query, ids, matrix.astype("float64"), top_k=top_k, rows=rows, scores=scores) == rank_hits(
            query, subset, top_k=top_k
        )
        assert select_top_k(query, ids, matrix, top_k=top_k) == rank_hits(query, dict(zip(ids, matrix.tolist())), top_k=top_k)


# ---------------------------------------------------------------- pool: recency rule, semantic reuse, retrieval field
# ---------------------------------------------------------------- 池：新近度规则、语义分复用、retrieval 字段
def test_recency_channel_rule(store, clock, index_provider, hash_embedder):
    """Recency runs for temporal routes, when the whole-gate rule is off, or as a fallback — not for an ordinary recall.
    新近度通道在 temporal 路由、整体入池规则关闭或作为兜底时运行——普通 recall 不运行。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on channel_sizes of the four cases.
        对四种情形 channel_sizes 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 400)
    as_of = store.now()
    small = RetrievalConfig(k_semantic=8, k_keyword=8, k_entity=8, k_recency=8, pool_max_factor=1)
    recall = _new_rows(store, index, hash_embedder, _aq("retry.ts backoff patch", as_of, 3), retrieval=small)
    assert recall and "recency" not in recall[0][2]["channel_sizes"] and "semantic" in recall[0][2]["channel_sizes"]
    temporal = _new_rows(store, index, hash_embedder, _aq("what happened in March 2026", as_of, 3), retrieval=small)
    assert temporal and temporal[0][2]["channel_sizes"].get("recency") == 8
    no_whole = RetrievalConfig(k_semantic=8, k_keyword=8, k_entity=8, k_recency=8, pool_max_factor=1, hard_gate_pool=False)
    channels = _new_rows(store, index, hash_embedder, _aq("retry.ts backoff patch", as_of, 3), retrieval=no_whole)
    assert channels and channels[0][2]["channel_sizes"].get("recency") == 8
    # Fallback: without an index and without tokens the other channels recall nothing, so recency fills the pool.
    # 兜底：没有索引也没有词元时其它通道一无所获，新近度填满池。
    pool = recall_pool(
        store, None, None, {"route": "recall"}, Scope(**SCOPE), as_of=as_of, include_expired=False, cfg=small, top_k=3
    )
    assert pool.channel_sizes == {"recency": 8} and pool.size == 8 and not pool.whole


def test_semantic_channel_scores_are_reused_in_pool(store, clock, index_provider, hash_embedder):
    """The in-pool semantic score of a channel hit is the channel's score; index.search is not asked for it again.
    通道命中的池内语义分就是通道分数；不再向 index.search 索取。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on the recorded index.search calls and on bit-equality with a full recompute.
        对记录到的 index.search 调用与全量重算逐位相等的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 200)
    as_of = store.now()
    calls: list[dict[str, Any]] = []
    original = index.search

    def spy(vector, *, candidate_ids=None, filter=None, top_k):
        """Record every search.
        记录每次搜索。
        """
        hits = original(vector, candidate_ids=candidate_ids, filter=filter, top_k=top_k)
        calls.append({"candidate_ids": candidate_ids, "filter": filter, "top_k": top_k, "hits": hits})
        return hits

    index.search = spy  # type: ignore[method-assign]
    cfg = RetrievalConfig(k_semantic=10, k_keyword=10, k_entity=10, k_recency=10, pool_max_factor=1, hard_gate_pool=False)
    rows = _new_rows(store, index, hash_embedder, _aq("retry.ts backoff patch", as_of, 5), retrieval=cfg)
    index.search = original  # type: ignore[method-assign]
    assert rows and len(calls) in (1, 2)
    channel = calls[0]
    assert channel["top_k"] == 10 and (channel["candidate_ids"] is None) == (channel["filter"] is not None)
    channel_ids = {atom_id for atom_id, _score in channel["hits"]}
    if len(calls) == 2:
        assert not (set(calls[1]["candidate_ids"]) & channel_ids) and calls[1]["filter"] is None
    pool_ids = [atom_id for atom_id, _score, _details in rows]
    query_vector = hash_embedder.embed("retry.ts backoff patch", "search")
    full = semantic_scores(index, query_vector, pool_ids)
    for atom_id, _score, details in rows:
        assert details["semantic_score"] == full[atom_id]
        if atom_id in channel_ids:
            assert details["semantic_score"] == max(0.0, dict(channel["hits"])[atom_id])


def test_axis_result_carries_retrieval_stats(store, clock, index_provider, hash_embedder):
    """AxisResult.retrieval is Pool.stats() for ranked queries and {} for axis-only ones; the out-parameter agrees.
    排序查询的 AxisResult.retrieval 是 Pool.stats()，仅轴查询为 {}；出参与之一致。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 20)
    as_of = store.now()
    stats: dict[str, Any] = {}
    ranked = execute_axis_query(
        store, index, hash_embedder, _aq("who is Sarah", as_of, 5), scoring=ScoringConfig(),
        capabilities=store.capabilities, retrieval_stats=stats,
    )  # fmt: skip
    assert ranked.retrieval == stats and set(ranked.retrieval) == {"pool_size", "channel_sizes", "channels_used", "gate_count", "whole"}
    assert ranked.retrieval["channels_used"] == [GATE_CHANNEL] and ranked.retrieval["whole"] is True
    listing = execute_axis_query(
        store, None, None, AxisQuery(filters=dict(SCOPE), text=None, limit=5, as_of=as_of, aggregation="list"),
        scoring=ScoringConfig(), capabilities=store.capabilities,
    )  # fmt: skip
    assert listing.retrieval == {} and listing.hits


def test_atom_type_pushdown_matches_sql(store, clock, index_provider, hash_embedder):
    """IndexFilter.atom_type_in equals the SQL atom_types gate on the sqlite provider, and the ranked path uses it.
    sqlite 提供者上 IndexFilter.atom_type_in 与 SQL 的 atom_types 门控相同，且排序路径使用它。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    pytest.importorskip("numpy")
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 80)
    as_of = store.now()
    scope = Scope(**SCOPE)
    query = hash_embedder.embed("deploy", "search")
    spec = IndexFilter(scope=scope, as_of=as_of, atom_type_in=("state",))
    gate = store.axis_ids(scope, as_of=as_of, atom_types=["state"])
    if index.supports_filter:
        assert index.search(query, filter=spec, top_k=50) == index.search(query, candidate_ids=gate, top_k=50)
        assert index.search(query, filter=IndexFilter(scope=scope, as_of=as_of, atom_type_in=("episode",)), top_k=50) == []
    aq = AxisQuery(
        filters=dict(SCOPE), text="deploy", limit=5, as_of=as_of, aggregation="topk",
        plan={**plan_query("deploy", as_of=as_of).as_dict(), "atom_types": ["state"]},
    )  # fmt: skip
    cfg = RetrievalConfig(k_semantic=6, k_keyword=6, k_entity=6, k_recency=6, pool_max_factor=1, hard_gate_pool=False)
    result = execute_axis_query(
        store, index, hash_embedder, aq, scoring=ScoringConfig(), capabilities=store.capabilities, retrieval=cfg, explain=True
    )
    assert result.hits and all(h.atom.atom_type == "state" for h in result.hits)
    assert result.retrieval["channel_sizes"].get("semantic") == 6


# ---------------------------------------------------------------- facade, ingest, milvus
# ---------------------------------------------------------------- 门面、摄取、milvus
def test_facade_search_passes_retrieval_config(store):
    """HMS.search hands config.retrieval to mvmi.search: custom channel bounds show in the explain block.
    HMS.search 把 config.retrieval 交给 mvmi.search：自定义通道上限出现在解释块中。

    Input / 输入：
        store: conftest fixture.
        store：conftest 夹具。

    Output / 输出：
        Assertions on score_details['channel_sizes'].
        对 score_details['channel_sizes'] 的断言。

    State effect / 状态影响：
        Ingests 12 messages into the in-memory store.
        向内存 store 摄取 12 条消息。
    """
    from hms_elite.memory.main import HMS

    config = HMSConfig(
        store={"path": ":memory:"},
        retrieval={"k_semantic": 4, "k_keyword": 3, "k_entity": 3, "k_recency": 3, "pool_max_factor": 1, "hard_gate_pool": False},
    )
    hms = HMS(config, store=store)
    for i in range(12):
        hms.add(f"Sarah patched retry.ts item{i} and the backoff regression is fixed.", user_id=U, timestamp=f"2026-03-{1 + i:02d}T09:00:00Z")
    rows = hms.search("retry.ts backoff patch", filters={"user_id": U}, top_k=3, threshold=0.0, explain=True)["results"]
    assert rows and rows[0]["score_details"]["channel_sizes"]["semantic"] == 4
    assert rows[0]["score_details"]["channel_sizes"].get("keyword") == 3 and rows[0]["score_details"]["pool_size"] <= 4
    assert GATE_CHANNEL not in rows[0]["score_details"]["channel_sizes"]


def test_ingest_upserts_carry_index_attrs(hms, store):
    """create_atom passes attrs=atom_index_attrs(stored) to index.upsert (no provider-side SELECT needed).
    create_atom 把 attrs=atom_index_attrs(stored) 传给 index.upsert（提供者无需自行 SELECT）。

    Input / 输入：
        hms, store: conftest fixtures.
        hms、store：conftest 夹具。

    Output / 输出：
        Assertions on the recorded upsert keyword arguments.
        对记录到的 upsert 关键字参数的断言。

    State effect / 状态影响：
        Ingests one message.
        摄取一条消息。
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    original = hms.index.upsert

    def spy(atom_id, vector, *, embedder, attrs=None):
        """Record the upsert.
        记录 upsert。
        """
        calls.append((atom_id, dict(attrs or {})))
        return original(atom_id, vector, embedder=embedder, attrs=attrs)

    hms.index.upsert = spy  # type: ignore[method-assign]
    try:
        hms.add("Sarah reviewed the billing retry patch.", user_id=U, timestamp="2026-03-02T09:00:00Z")
    finally:
        hms.index.upsert = original  # type: ignore[method-assign]
    assert calls
    for atom_id, attrs in calls:
        assert attrs == atom_index_attrs(store.get_atom(atom_id)) and attrs["user_id"] == U and attrs["atom_type"]


def test_milvus_reopen_loads_collection(milvus_uri, store, hash_embedder):
    """A released / reopened collection is loaded by _ensure_collection: the second MilvusIndex can search at once.
    已 release / 重新打开的集合由 _ensure_collection 加载：第二个 MilvusIndex 能立即搜索。

    Input / 输入：
        milvus_uri, store, hash_embedder: conftest fixtures (skipped without pymilvus).
        milvus_uri、store、hash_embedder：conftest 夹具（缺少 pymilvus 时跳过）。

    Output / 输出：
        Assertions on search results across the reopen and on the atom_type scalar.
        对重新打开前后搜索结果与 atom_type 标量的断言。

    State effect / 状态影响：
        Creates the 'reopen' collection.
        创建 'reopen' 集合。
    """
    pytest.importorskip("pymilvus")
    from hms_elite.index.milvus import MilvusIndex

    first = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "reopen", "index_type": "FLAT"}, store=store, dims=64)
    vectors = [(f"atm_r{i}", hash_embedder.embed(f"text {i}", "add")) for i in range(5)]
    first.upsert_batch(vectors, embedder="hash@64", attrs={f"atm_r{i}": {"atom_type": "message", "user_id": U} for i in range(5)})
    query = hash_embedder.embed("text 2", "search")
    expected = first.search(query, top_k=3)
    first._client.release_collection("reopen")
    first.close()
    second = MilvusIndex({"uri": milvus_uri, "scalar_index": "none", "collection": "reopen", "index_type": "FLAT"}, store=store, dims=64)
    assert second.search(query, top_k=3) == expected and second.count() == 5
    spec = IndexFilter(scope=Scope(user_id=U), include_expired=True, atom_type_in=("message",))
    assert second.search(query, filter=spec, top_k=3) == expected and second.count(spec) == 5
    assert second.search(query, filter=IndexFilter(include_expired=True, atom_type_in=("episode",)), top_k=3) == []
    second.close()
