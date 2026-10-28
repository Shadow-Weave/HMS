"""Group D tests: filter DSL conformance, query planner, seven-axis execution, search fusion, get_all and rows.
D 组测试：过滤 DSL 一致性、查询规划器、七轴执行、搜索融合、get_all 与行投影。
"""

# Purpose (EN): Assert CONTRACT §13 (tests 33–39 of §18.3) directly against the mvmi plane and utils/scoring.
# 目的（中）：直接针对 mvmi 平面与 utils/scoring 断言契约 §13（§18.3 的测试 33–39）。

# Input (EN): The store/clock/ids/index_provider/hash_embedder fixtures of conftest.py; atoms seeded through StateStore.
# 输入（中）：conftest.py 的 store/clock/ids/index_provider/hash_embedder 夹具；通过 StateStore 播种的原子。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside the in-memory SQLite database (no network, no home writes, no ingest, no graph).
# 状态影响（中）：除内存 SQLite 数据库外无副作用（无网络、不写用户目录、不经 ingest、不触图）。

from __future__ import annotations

import json
from typing import Any

import pytest

from hms_elite.configs.base import ScoringConfig
from hms_elite.exceptions import ValidationError, unwrap_hms_error
from hms_elite.models import RESERVED_ATOM_KEYS, AxisQuery, MemoryAtom, Scope, SourceEvent
from hms_elite.mvmi import filters as mfilters
from hms_elite.mvmi import query as mquery
from hms_elite.mvmi.filters import And, Eq, Exists, In, Not, Or, evaluate, lower_to_sql, parse_filters, plan_filters
from hms_elite.mvmi.planner import plan_query
from hms_elite.mvmi.query import atom_to_row, count_axis_query, execute_axis_query, get_all, search
from hms_elite.utils import scoring as mscoring
from hms_elite.utils.factory import IndexFactory

U = "alice"
S1 = {"user_id": U, "run_id": "run-coffee"}
SCOPE4 = {"user_id": U, "agent_id": "coder", "run_id": "run-billing"}
ALLOWED_TABLES = {"memory_atoms", "atom_entity", "entities", "atom_topic", "atom_anchor", "atom_vectors"}
EXPLAIN_KEYS = [
    "semantic_score",
    "bm25_score",
    "entity_boost",
    "raw_score",
    "max_possible_score",
    "final_score",
    "threshold",
    "recency_score",
    "support_score",
    "keyword_backend",
    "axis_gates",
    "primary_axis",
    "route",
    "signals",
    "recalled_by",
    "pool_size",
    "channel_sizes",
]
ROW_KEYS = [
    "id",
    "memory",
    "hash",
    "metadata",
    "created_at",
    "updated_at",
    "user_id",
    "run_id",
    "atom_id",
    "version",
    "atom_type",
    "scale_level",
    "anchor_ids",
    "valid_from",
    "valid_to",
    "confidence",
    "hms",
]
RETRY = ("retry.ts", "src/billing/retry.ts", "FILE", ("src/billing/retry.ts",))
IDEMP = ("idempotency.ts", "src/billing/idempotency.ts", "FILE", ("src/billing/idempotency.ts",))
E2E = ("billing_e2e", "billing_e2e", "IDENTIFIER", ())
REGRESSION = ("subscription_state_regression", "subscription_state_regression", "IDENTIFIER", ())
SARAH = ("sarah", "Sarah", "PROPER", ())


def _anchor(store, scope: dict, text: str, ts: str, *, role: str = "user", **kw) -> str:
    """Insert one message event in `scope` at `ts` and return its anchor id.
    在 `scope` 于 `ts` 插入一条消息事件并返回锚点 id。
    """
    ev = SourceEvent(scope=Scope(**scope), role=role, payload={"text": text}, timestamp=ts, **kw)
    stored, _ = store.insert_source_event(ev)
    return store.get_or_create_anchor(stored).anchor_id


def _insert(store, clock, *, name: str, text: str, ts: str, scope: dict, anchors: list[str], **overrides) -> MemoryAtom:
    """Insert an atom (state by default) witnessed by `anchors`; entity_keys/topics go to the store; clock ticks.
    插入由 `anchors` 见证的原子（默认 state）；entity_keys/topics 交给 store；时钟前进。
    """
    entity_keys = overrides.pop("entity_keys", [])
    topics = overrides.pop("topics", [])
    content = {**overrides.pop("content", {}), "text": text}
    base: dict[str, Any] = dict(
        atom_type="state",
        content=content,
        text=text,
        scale_level="state",
        source_kind="observation",
        source_id=f"obs_{name}",
        extracted_by="rule@obs-v1",
        scope=Scope(**scope),
        anchor_ids=anchors,
        time_anchor=ts,
        subject="user",
        predicate="likes",
    )
    base.update(overrides)
    atom = store.insert_atom(MemoryAtom(**base), entity_keys=list(entity_keys), topics=list(topics))
    clock.tick(seconds=1)
    return atom


@pytest.fixture
def index(index_provider: str, store):
    """The parametrized index provider bound to the store fixture (64 dims).
    绑定到 store 夹具的参数化索引提供者（64 维）。

    Input / 输入：
        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        store: The in-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        An IndexBase instance.
        IndexBase 实例。

    State effect / 状态影响：
        None until vectors are upserted.
        写入向量前无副作用。
    """
    return IndexFactory.create(index_provider, {}, store=store, dims=64)


@pytest.fixture
def seeded(store, clock, index, hash_embedder) -> dict[str, MemoryAtom]:
    """Seed S1/S2/S3/S4-shaped atoms directly through StateStore and index every vector; returns atoms by name.
    直接通过 StateStore 播种 S1/S2/S3/S4 形状的原子并索引全部向量；按名称返回原子。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        clock: FrozenClock (ticked one second per atom so created_at differs).
        clock：FrozenClock（每个原子前进一秒，使 created_at 不同）。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: 64-dim HashEmbedder.
        hash_embedder：64 维 HashEmbedder。

    Output / 输出：
        {name: MemoryAtom} — v1, v2, habit, vac, sarah1..3, sarah_ent, uc, e1, e2, e3, e4, ep1, ep2, dc, bob.
        {名称: MemoryAtom}——v1、v2、habit、vac、sarah1..3、sarah_ent、uc、e1、e2、e3、e4、ep1、ep2、dc、bob。

    State effect / 状态影响：
        Writes memory_atoms and the axis tables of the in-memory store; upserts vectors into the index.
        写入内存 store 的 memory_atoms 与轴表；向索引写入向量。
    """
    atoms: dict[str, MemoryAtom] = {}
    a1 = _anchor(store, S1, "I like oat-milk lattes in the morning.", "2026-04-02T08:00:00Z")
    a2 = _anchor(store, S1, "Actually I switched to black coffee now.", "2026-04-20T08:00:00Z")
    a3 = _anchor(store, S1, "I usually run every morning.", "2026-05-02T08:00:00Z")
    a4 = _anchor(store, S1, "I am on vacation until June.", "2026-05-01T09:00:00Z")
    atoms["v1"] = _insert(
        store,
        clock,
        name="v1",
        text="User likes oat-milk lattes in the morning.",
        ts="2026-04-02T08:00:00Z",
        scope=S1,
        anchors=[a1],
        obs_type="preference_event",
        sentiment="positive",
        content={"objects": ["oat-milk lattes in the morning"], "obs_type": "preference_event"},
        topics=["coffee", "preference"],
        user_meta={"source": "chat", "priority": 1},
        archived_reason="revised",
        valid_to="2026-04-20T08:00:00Z",
        updated_at="2026-04-20T08:00:00Z",
    )
    atoms["v2"] = _insert(
        store,
        clock,
        name="v2",
        text="User likes black coffee.",
        ts="2026-04-20T08:00:00Z",
        scope=S1,
        anchors=[a2],
        memory_id=atoms["v1"].memory_id,
        version=2,
        obs_type="preference_event",
        sentiment="positive",
        content={"objects": ["black coffee"], "obs_type": "preference_event"},
        topics=["coffee", "preference"],
        user_meta={"source": "chat", "priority": 2, "tags": ["morning"]},
    )
    atoms["habit"] = _insert(
        store,
        clock,
        name="habit",
        text="User usually runs every morning.",
        ts="2026-05-02T08:00:00Z",
        scope=S1,
        anchors=[a3],
        obs_type="habit",
        predicate="usually",
        content={"objects": ["runs every morning"]},
        user_meta={"source": "CHAT-import"},
        visibility="team",
    )
    atoms["vac"] = _insert(
        store,
        clock,
        name="vac",
        text="User is on vacation until June.",
        ts="2026-05-01T09:00:00Z",
        scope=S1,
        anchors=[a4],
        obs_type="fact",
        predicate="is",
        content={"objects": ["on vacation until June"]},
        topics=["travel"],
        valid_to="2026-06-01T00:00:00Z",
        visibility="public",
    )
    b1 = _anchor(
        store, {"user_id": U, "run_id": "run-a"}, "Sarah joined the billing team in March.", "2026-03-02T09:00:00Z"
    )
    b2 = _anchor(
        store, {"user_id": U, "run_id": "run-b"}, "Sarah shipped the webhook retry design.", "2026-05-06T09:00:00Z"
    )
    b3 = _anchor(
        store, {"user_id": U, "run_id": "run-c"}, "Sarah is reviewing the checkout refactor.", "2026-06-11T09:00:00Z"
    )
    atoms["sarah1"] = _insert(
        store,
        clock,
        name="sarah1",
        text="Sarah joined the billing team in March.",
        ts="2026-03-01T00:00:00Z",
        scope={"user_id": U, "run_id": "run-a"},
        anchors=[b1],
        atom_type="event",
        scale_level="event",
        obs_type="temporal_fact",
        subject="Sarah",
        predicate="did",
        content={"event_time": "2026-03-01T00:00:00.000000+00:00"},
        entity_keys=[SARAH],
        topics=["work", "billing"],
    )
    atoms["sarah2"] = _insert(
        store,
        clock,
        name="sarah2",
        text="Sarah shipped the webhook retry design on May 6.",
        ts="2026-05-06T00:00:00Z",
        scope={"user_id": U, "run_id": "run-b"},
        anchors=[b2],
        atom_type="event",
        scale_level="event",
        obs_type="temporal_fact",
        subject="Sarah",
        predicate="did",
        entity_keys=[SARAH],
        topics=["work", "engineering"],
    )
    atoms["sarah3"] = _insert(
        store,
        clock,
        name="sarah3",
        text="Sarah is reviewing the checkout refactor this week.",
        ts="2026-06-11T09:00:00Z",
        scope={"user_id": U, "run_id": "run-c"},
        anchors=[b3],
        obs_type="fact",
        subject="Sarah",
        predicate="is",
        entity_keys=[SARAH],
        topics=["engineering", "billing"],
    )
    atoms["sarah_ent"] = _insert(
        store,
        clock,
        name="sarah_ent",
        text="Sarah",
        ts="2026-06-11T09:00:00Z",
        scope={"user_id": U},
        anchors=[b1, b2, b3],
        atom_type="entity",
        scale_level="entity",
        source_kind="entity",
        source_id="ent_sarah",
        subject="Sarah",
        predicate="is_entity",
        content={"entity_id": "ent_sarah", "entity_type": "PROPER", "aliases": [], "mentions": 3},
        entity_keys=[SARAH],
    )
    c0 = _anchor(store, SCOPE4, "Do not modify the subscription state machine.", "2026-06-15T09:00:00Z")
    c1 = _anchor(store, SCOPE4, "file_diff src/billing/retry.ts", "2026-06-15T10:00:00Z", role="tool", actor_id="coder")
    c2 = _anchor(store, SCOPE4, "test_log billing_e2e failed", "2026-06-15T10:04:00Z", role="tool", actor_id="coder")
    c3 = _anchor(store, SCOPE4, "file_diff src/billing/idempotency.ts", "2026-06-15T10:10:00Z", role="tool")
    c4 = _anchor(store, SCOPE4, "test_log billing_e2e passed", "2026-06-15T10:14:00Z", role="tool")
    atoms["uc"] = _insert(
        store,
        clock,
        name="uc",
        text="Do not modify the subscription state machine.",
        ts="2026-06-15T09:00:00Z",
        scope=SCOPE4,
        anchors=[c0],
        obs_type="constraint",
        subject="project",
        predicate="must_not",
        content={"objects": ["subscription state machine"]},
        topics=["billing", "constraint"],
        confidence=0.9,
    )
    atoms["e1"] = _insert(
        store,
        clock,
        name="e1",
        text="file_diff src/billing/retry.ts: add exponential backoff",
        ts="2026-06-15T10:00:00Z",
        scope=SCOPE4,
        anchors=[c1],
        atom_type="event",
        scale_level="event",
        source_kind="event",
        role="tool",
        actor_id="coder",
        sentiment="neutral",
        subject=None,
        predicate=None,
        entity_keys=[RETRY],
        topics=["engineering", "billing"],
    )
    atoms["e2"] = _insert(
        store,
        clock,
        name="e2",
        text="test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression",
        ts="2026-06-15T10:04:00Z",
        scope=SCOPE4,
        anchors=[c2],
        atom_type="event",
        scale_level="event",
        source_kind="event",
        role="tool",
        actor_id="coder",
        sentiment="negative",
        subject=None,
        predicate=None,
        entity_keys=[E2E, REGRESSION],
        topics=["engineering", "billing"],
    )
    atoms["ep1"] = _insert(
        store,
        clock,
        name="ep1",
        text=(
            "[negative] fix checkout webhook retry → patch src/billing/retry.ts: add exponential backoff → "
            "billing_e2e failed: subscription_state_regression"
        ),
        ts="2026-06-15T10:04:00Z",
        scope=SCOPE4,
        anchors=[c2, c1],
        atom_type="episode",
        scale_level="L1_micro",
        source_kind="ep",
        source_id="ep_1",
        sentiment="negative",
        subject=None,
        predicate=None,
        content={"ep_id": "ep_1", "slots": {"u": "fix checkout webhook retry", "r": "negative"}},
        entity_keys=[RETRY, E2E, REGRESSION],
        topics=["engineering", "billing"],
    )
    atoms["dc"] = _insert(
        store,
        clock,
        name="dc",
        text=(
            "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': "
            "billing_e2e failed: subscription_state_regression"
        ),
        ts="2026-06-15T10:04:00Z",
        scope=SCOPE4,
        anchors=[c2, c1],
        obs_type="constraint",
        subject="project",
        predicate="must_not",
        content={"objects": ["patch src/billing/retry.ts: add exponential backoff"]},
        entity_keys=[RETRY, E2E],
        topics=["engineering", "billing"],
        confidence=0.9,
    )
    atoms["e3"] = _insert(
        store,
        clock,
        name="e3",
        text="file_diff src/billing/idempotency.ts: add idempotency key",
        ts="2026-06-15T10:10:00Z",
        scope=SCOPE4,
        anchors=[c3],
        atom_type="event",
        scale_level="event",
        source_kind="event",
        role="tool",
        sentiment="neutral",
        subject=None,
        predicate=None,
        entity_keys=[IDEMP],
        topics=["engineering", "billing"],
    )
    atoms["e4"] = _insert(
        store,
        clock,
        name="e4",
        text="test_log billing_e2e: passed (14 passed, 0 failed)",
        ts="2026-06-15T10:14:00Z",
        scope=SCOPE4,
        anchors=[c4],
        atom_type="event",
        scale_level="event",
        source_kind="event",
        role="tool",
        sentiment="positive",
        subject=None,
        predicate=None,
        entity_keys=[E2E],
        topics=["engineering", "billing"],
    )
    atoms["ep2"] = _insert(
        store,
        clock,
        name="ep2",
        text=(
            "[positive] fix checkout webhook retry → patch src/billing/idempotency.ts: add idempotency key → "
            "billing_e2e passed"
        ),
        ts="2026-06-15T10:14:00Z",
        scope=SCOPE4,
        anchors=[c4, c3],
        atom_type="episode",
        scale_level="L1_micro",
        source_kind="ep",
        source_id="ep_2",
        sentiment="positive",
        subject=None,
        predicate=None,
        content={"ep_id": "ep_2", "slots": {"u": "fix checkout webhook retry", "r": "positive"}},
        entity_keys=[IDEMP, E2E],
        topics=["engineering", "billing"],
    )
    bob = _anchor(store, {"user_id": "bob"}, "I like tea.", "2026-06-01T09:00:00Z")
    atoms["bob"] = _insert(
        store,
        clock,
        name="bob",
        text="Bob likes tea.",
        ts="2026-06-01T09:00:00Z",
        scope={"user_id": "bob"},
        anchors=[bob],
        obs_type="preference_event",
        content={"objects": ["tea"]},
        topics=["coffee"],
    )
    for atom in atoms.values():
        index.upsert(atom.atom_id, hash_embedder.embed(atom.text), embedder=hash_embedder.identity)
    return atoms


def _ids(atoms: list[MemoryAtom]) -> list[str]:
    """Atom ids in order.
    按顺序的原子 id。
    """
    return [a.atom_id for a in atoms]


def _run(store, index, embedder, **aq_kwargs) -> Any:
    """execute_axis_query with the default ScoringConfig and the store's capabilities.
    使用默认 ScoringConfig 与 store 能力执行 execute_axis_query。
    """
    explain = aq_kwargs.pop("explain", False)
    threshold = aq_kwargs.pop("threshold", 0.0)
    return execute_axis_query(
        store,
        index,
        embedder,
        AxisQuery(**aq_kwargs),
        scoring=ScoringConfig(),
        capabilities=store.capabilities,
        threshold=threshold,
        explain=explain,
    )


def _search(store, index, embedder, query: str, filters: dict, **kw) -> list[dict]:
    """mvmi.query.search with test defaults (top_k=20, threshold=0.0, as_of=now).
    带测试默认值（top_k=20、threshold=0.0、as_of=now）的 mvmi.query.search。
    """
    params: dict[str, Any] = dict(
        top_k=20, threshold=0.0, explain=False, as_of=store.now(), include_expired=False, scoring=ScoringConfig()
    )
    params.update(kw)
    return search(store, index, embedder, query=query, filters=filters, capabilities=store.capabilities, **params)


# ---------------------------------------------------------------------------
# 33. Filter DSL conformance
# 33. 过滤 DSL 一致性
# ---------------------------------------------------------------------------
def test_filter_dsl_conformance(store, index, hash_embedder, seeded, monkeypatch):
    """Fourteen Mem0 filter cases give identical ids through lower_to_sql and evaluate; JSON1 fallback agrees.
    十四个 Mem0 过滤用例经 lower_to_sql 与 evaluate 得到相同 id；JSON1 回退结果一致。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index (seeding dependency).
        index：参数化索引（播种依赖）。

        hash_embedder: HashEmbedder (seeding dependency).
        hash_embedder：HashEmbedder（播种依赖）。

        seeded: Atoms by name.
        seeded：按名称的原子。

        monkeypatch: pytest monkeypatch for the capabilities dict.
        monkeypatch：用于能力字典的 pytest monkeypatch。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    caps = store.capabilities
    assert caps["json1"] is True

    def both(filters: dict) -> list[str]:
        scope, expr = parse_filters(filters)
        sql, params, joins = lower_to_sql(expr, capabilities=caps, scope=scope)
        via_sql = _ids(store.axis_select(scope, where_sql=sql, params=params, joins=list(joins), include_expired=True))
        via_py = _ids(
            [
                atom
                for atom in store.axis_select(scope, include_expired=True)
                if evaluate(expr, atom, entity_keys=set(atom.entity_ids), anchor_ids=set(atom.anchor_ids))
            ]
        )
        assert via_sql == via_py, filters
        return via_sql

    e1, ep1, dc = a["e1"].atom_id, a["ep1"].atom_id, a["dc"].atom_id
    alice_all = _ids(store.axis_select(Scope(user_id=U), include_expired=True))
    # 1. scalar eq
    # 1. 标量相等
    assert set(both({"user_id": U, "obs_type": "preference_event"})) == {a["v1"].atom_id, a["v2"].atom_id}
    # 2. list in
    # 2. 列表 in
    assert set(both({"user_id": U, "scale": ["event", "L1_micro"]})) == {
        a[n].atom_id for n in ("sarah1", "sarah2", "e1", "e2", "e3", "e4", "ep1", "ep2")
    }
    # 3. '*'
    # 3. '*' 通配（字段存在）
    assert set(both({"user_id": U, "sentiment": "*"})) == {
        a[n].atom_id for n in ("v1", "v2", "e1", "e2", "ep1", "e3", "e4", "ep2")
    }
    # 4. {gte} on created_at (clock ticks between inserts → proper, non-empty subset)
    # 4. created_at 上的 {gte}（两次插入之间时钟推进 → 得到非空的真子集）
    since = a["uc"].created_at
    got = both({"user_id": U, "created_at": {"gte": since}})
    assert got and set(got) < set(alice_all) and a["uc"].atom_id in got and a["v1"].atom_id not in got
    # 5. {lte} with a date-only literal → end of that day
    # 5. {lte} 搭配纯日期字面量 → 当天的结束时刻
    assert set(both({"user_id": U, "time": {"lte": "2026-05-06"}})) == {
        a[n].atom_id for n in ("v1", "v2", "habit", "vac", "sarah1", "sarah2")
    }
    # 6. {icontains} on a metadata key
    # 6. metadata 键上的 {icontains}
    assert set(both({"user_id": U, "source": {"icontains": "CHAT"}})) == {a[n].atom_id for n in ("v1", "v2", "habit")}
    # 7. meta.<key>
    # 7. meta.<key> 形式的键
    assert both({"user_id": U, "meta.priority": {"gte": 2}}) == [a["v2"].atom_id]
    # 混合类型之间永不比较
    assert both({"user_id": U, "meta.priority": {"gte": "2"}}) == []  # mixed types never compare
    # 8. OR
    # 8. OR（或）
    assert set(both({"user_id": U, "OR": [{"topic": "coffee"}, {"entity": "Sarah"}]})) == {
        a[n].atom_id for n in ("v1", "v2", "sarah1", "sarah2", "sarah3", "sarah_ent")
    }
    # 9. NOT
    # 9. NOT（非）
    not_state = both({"user_id": U, "NOT": [{"atom_type": "state"}]})
    assert set(not_state) == {
        i for i in alice_all if i not in {a[n].atom_id for n in ("v1", "v2", "habit", "vac", "sarah3", "uc", "dc")}
    }
    # 10. nested AND(OR)
    # 10. 嵌套的 AND(OR)
    assert set(
        both({"user_id": U, "AND": [{"OR": [{"topic": "billing"}, {"topic": "coffee"}]}, {"scale": "state"}]})
    ) == {a[n].atom_id for n in ("v1", "v2", "sarah3", "uc", "dc")}
    # 11. entity axis (FILE paths keyed by basename; scope pins user_id only)
    # 11. 实体轴（FILE 路径以 basename 为键；scope 只固定 user_id）
    assert set(both({"user_id": U, "entity": "src/billing/retry.ts"})) == {e1, ep1, dc}
    assert set(both({"user_id": U, "run_id": "run-b", "entities": ["sarah"]})) == {a["sarah2"].atom_id}
    # 12. topic axis
    # 12. 主题轴
    assert set(both({"user_id": U, "topics": ["travel", "preference"]})) == {a[n].atom_id for n in ("v1", "v2", "vac")}
    # 13. anchor_id
    # 13. anchor_id（锚点 id）
    assert set(both({"user_id": U, "anchor_id": a["e1"].anchor_ids[0]})) == {e1, ep1, dc}
    # 14. unknown operator → ValueError (VALIDATION_012, message verbatim)
    # 14. 未知操作符 → ValueError（VALIDATION_012，消息逐字一致）
    with pytest.raises(ValueError) as info:
        parse_filters({"user_id": U, "topic": {"like": "coffee"}})
    assert isinstance(info.value, ValidationError) and info.value.error_code == "VALIDATION_012"
    assert (
        str(info.value)
        == "Unsupported operator 'like'. Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"
    )
    assert mfilters.has_advanced_operators({"user_id": U, "topic": {"eq": "x"}}) is True
    assert mfilters.has_advanced_operators({"user_id": U, "OR": []}) is True
    assert mfilters.has_advanced_operators({"user_id": U, "topic": "*"}) is True
    assert mfilters.has_advanced_operators({"user_id": U, "topic": "coffee"}) is False

    # AST shapes and the visibility lattice gate
    # 语法树形状与可见性格（lattice）门控
    scope, expr = parse_filters({"user_id": U, "visibility": "team", "NOT": [{"scale": ["event"]}]})
    assert scope == Scope(user_id=U)
    assert expr == And(
        items=[
            In(field="visibility", values=["team", "event", "public"]),
            Not(item=In(field="scale_level", values=["event"])),
        ]
    )
    _, meta_only = parse_filters({"user_id": U, "priority": 2}, require_scope=False, force_meta=True)
    assert meta_only == And(items=[Eq(field="meta.user_id", value=U), Eq(field="meta.priority", value=2)])
    _, star = parse_filters({"user_id": U, "topic": "*", "OR": [{"topic": "a"}, {"topic": "b"}]})
    assert star == And(
        items=[Exists(field="topic"), Or(items=[Eq(field="topic", value="a"), Eq(field="topic", value="b")])]
    )
    with pytest.raises(ValidationError) as info:
        parse_filters({"topic": "coffee"})
    assert info.value.error_code == "VALIDATION_001"

    # JSON1 fallback: meta leaves move to Python and the executed result is identical
    # JSON1 回退：meta 叶子下放到 Python 求值，执行结果完全一致
    fallback = {**caps, "json1": False}
    for filters in (
        {"user_id": U, "source": {"icontains": "CHAT"}},
        {"user_id": U, "meta.priority": {"gte": 2}},
        {"user_id": U, "OR": [{"meta.priority": 1}, {"topic": "travel"}]},
        {"user_id": U, "NOT": [{"meta.source": "chat"}], "scale": "state"},
    ):
        scope, expr = parse_filters(filters)
        sql_part, py_part = plan_filters(expr, capabilities=fallback)
        assert py_part is not None and mfilters._has_meta(py_part)
        assert sql_part is None or not mfilters._has_meta(sql_part)
        assert plan_filters(expr, capabilities=caps) == (expr, None)
        with_json1 = _run(store, index, hash_embedder, filters=filters, include_expired=True, limit=100)
        monkeypatch.setattr(store, "capabilities", fallback)
        without = _run(store, index, hash_embedder, filters=filters, include_expired=True, limit=100)
        monkeypatch.setattr(store, "capabilities", caps)
        assert _ids([h.atom for h in with_json1.hits]) == _ids([h.atom for h in without.hits])
        assert with_json1.hits, filters
    with pytest.raises(ValidationError) as info:
        lower_to_sql(Eq(field="meta.priority", value=1), capabilities=fallback, scope=Scope(user_id=U))
    assert info.value.error_code == "VALIDATION_012"


# ---------------------------------------------------------------------------
# 34. Planner
# 34. 计划器
# ---------------------------------------------------------------------------
def test_plan_query_routes_and_gates(clock):
    """Routes, time windows, entity/identifier hints and hard gates of plan_query.
    plan_query 的路由、时间窗、实体/标识符线索与硬门控。

    Input / 输入：
        clock: FrozenClock (as_of base date 2026-09-02).
        clock：FrozenClock（as_of 基准日期 2026-09-02）。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    now = clock.now()
    may = ("2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00")
    en = plan_query("what happened in May", as_of=now)
    assert en.route == "temporal" and en.hard_time_gate == may and en.time_windows == [("month_en", *may)]
    assert en.scale_default == ["event", "L1_micro", "state", "L3_macro"]
    zh = plan_query("5月发生了什么", as_of=now)
    assert zh.route == "temporal" and zh.hard_time_gate == may
    ent = plan_query("everything about Sarah", as_of=now)
    assert ent.route == "entity" and ent.hard_entity_gate == ["sarah"] and ent.entity_hints[0].key == "sarah"
    assert ent.scale_default == ["state", "event", "L1_micro", "L2_meso", "L3_macro", "entity", "relation"]
    why = plan_query("why must retry.ts not be modified", as_of=now)
    assert why.route == "detective" and why.identifier_tokens == ["retry.ts"] and why.scale_default is None
    assert why.hard_entity_gate is None and why.hard_type_gate is None
    plain = plan_query("retry.ts may change", as_of=now)
    assert plain.route == "recall" and plain.time_windows == [] and plain.hard_time_gate is None
    assert plain.content_tokens == ["retry.ts", "change"]
    pref = plan_query("what coffee does the user like", as_of=now)
    assert pref.route == "preference"
    assert pref.hard_type_gate == ("obs_type", ["preference_event", "habit"]) and pref.scale_default == ["state"]
    assert pref.content_tokens == ["coffee", "like"] and pref.entity_hints == []
    iso = plan_query("what did we ship on 2026-05-06", as_of=now)
    assert iso.hard_time_gate == ("2026-05-06T00:00:00.000000+00:00", "2026-05-06T23:59:59.999999+00:00")
    rel = plan_query("yesterday's failures", as_of=now)
    assert rel.route == "temporal" and rel.hard_time_gate == (
        "2026-09-01T00:00:00.000000+00:00",
        "2026-09-01T23:59:59.999999+00:00",
    )
    assert plan_query("where were we on the billing work", as_of=now).route == "work_state"
    assert plan_query("coffee", as_of=now).route == "recall"
    assert plan_query("coffee", as_of=now, mode="deep").route == "work_state"
    assert plan_query("what coffee does the user like", as_of=now, mode="deep").route == "preference"
    as_dict = pref.as_dict()
    assert as_dict["hard_type_gate"] == ["obs_type", ["preference_event", "habit"]] and as_dict["route"] == "preference"
    assert json.loads(json.dumps(ent.as_dict()))["entity_hints"][0]["key"] == "sarah"


# ---------------------------------------------------------------------------
# 35. Seven axes and every aggregation / projection
# 35. 七个轴与全部聚合 / 投影
# ---------------------------------------------------------------------------
def test_axis_query_seven_axes_and_aggregations(store, index, hash_embedder, seeded):
    """Time/validity/entity/topic/role/sentiment/scale gates plus HMS §3.5 projections and aggregations.
    时间/有效性/实体/主题/角色/情感/尺度门控，以及 HMS §3.5 的投影与聚合。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: HashEmbedder.
        hash_embedder：HashEmbedder。

        seeded: Atoms by name.
        seeded：按名称的原子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    run = lambda **kw: _run(store, index, hash_embedder, **kw)  # noqa: E731 - local shorthand
    ids = lambda res: _ids([h.atom for h in res.hits])  # noqa: E731 - local shorthand

    # time axis
    # 时间轴
    res = run(filters={"user_id": U, "time": {"gte": "2026-05-01", "lte": "2026-05-31"}}, include_expired=True)
    assert ids(res) == [a["sarah2"].atom_id, a["habit"].atom_id, a["vac"].atom_id] and res.primary_axis == "time"
    # validity axis (as_of / include_expired)
    # 有效性轴（as_of / include_expired）
    assert ids(run(filters={"user_id": U, "topic": "travel"}, as_of="2026-05-15T00:00:00Z")) == [a["vac"].atom_id]
    assert ids(run(filters={"user_id": U, "topic": "travel"}, as_of="2026-07-01T00:00:00Z")) == []
    assert ids(run(filters={"user_id": U, "topic": "travel"}, include_expired=True)) == [a["vac"].atom_id]
    # v1 已归档
    assert ids(run(filters={"user_id": U, "topic": "coffee"}, as_of="2026-04-10T00:00:00Z")) == []  # v1 archived
    # entity axis
    # 实体轴
    assert set(ids(run(filters={"user_id": U, "entity": "Sarah"}))) == {
        a[n].atom_id for n in ("sarah1", "sarah2", "sarah3", "sarah_ent")
    }
    # topic axis
    # 主题轴
    assert ids(run(filters={"user_id": U, "topic": "coffee"})) == [a["v2"].atom_id]
    # role axis
    # 角色轴
    assert set(ids(run(filters={**SCOPE4, "role": "tool"}))) == {a[n].atom_id for n in ("e1", "e2", "e3", "e4")}
    # sentiment axis
    # 情感轴
    assert set(ids(run(filters={**SCOPE4, "sentiment": "negative"}))) == {a["e2"].atom_id, a["ep1"].atom_id}
    # scale axis
    # 尺度轴
    assert set(ids(run(filters=SCOPE4, scale_constraint=["L1_micro"]))) == {a["ep1"].atom_id, a["ep2"].atom_id}
    assert ids(run(filters=SCOPE4, scale_constraint=["L1_micro"])) == [a["ep2"].atom_id, a["ep1"].atom_id]
    # scope: bob never leaks into alice
    # 范围：bob 的数据绝不会泄漏进 alice
    assert a["bob"].atom_id not in ids(run(filters={"user_id": U}, include_expired=True, limit=100))

    # HMS §3.5 preference query: value projection + latest per (subject, predicate)
    # HMS §3.5 偏好查询：value 投影 + 每个 (subject, predicate) 取最新
    res = run(
        filters={"user_id": U, "topic": "coffee"},
        project_axis="value",
        scale_constraint=["state"],
        aggregation="latest",
    )
    assert res.projection == [["black coffee"]] and len(res.hits) == 1 and res.hits[0].atom.atom_id == a["v2"].atom_id
    assert res.aggregation == "latest" and res.project_axis == "value" and res.count is None
    res = run(
        filters={"user_id": U, "subject": "user"},
        project_axis="value",
        scale_constraint=["state"],
        aggregation="latest",
    )
    assert res.projection == [["runs every morning"], ["black coffee"]]
    res = run(
        filters={"user_id": U, "topic": "coffee"}, project_axis="value", include_expired=True, aggregation="latest"
    )
    assert res.projection == [["black coffee"]]
    res = run(filters={"user_id": U, "topic": "coffee"}, project_axis="value", include_expired=True)
    assert res.projection == [["black coffee"], ["oat-milk lattes in the morning"]]
    # time projection: ascending (time_anchor, atom_id)
    # time 投影：按 (time_anchor, atom_id) 升序
    res = run(filters={"user_id": U, "entity": "sarah"}, project_axis="time")
    assert res.projection == [
        (a["sarah1"].time_anchor, a["sarah1"].atom_id),
        (a["sarah2"].time_anchor, a["sarah2"].atom_id),
        (a["sarah3"].time_anchor, a["sarah3"].atom_id),
        (a["sarah_ent"].time_anchor, a["sarah_ent"].atom_id),
    ]
    # entity projection: {key: [atom_ids in hit order]}, keys sorted
    # entity 投影：{key: [按命中顺序排列的 atom_id]}，键已排序
    res = run(filters={**SCOPE4, "entity": "retry.ts"}, project_axis="entity")
    order = ids(res)
    assert list(res.projection) == ["billing_e2e", "retry.ts", "subscription_state_regression"]
    assert res.projection["retry.ts"] == order == [a["dc"].atom_id, a["ep1"].atom_id, a["e1"].atom_id]
    assert res.projection["billing_e2e"] == [a["dc"].atom_id, a["ep1"].atom_id]
    # topic projection
    # topic 投影
    res = run(filters={"user_id": U, "topic": "coffee"}, project_axis="topic", include_expired=True)
    assert res.projection == {
        "coffee": [a["v2"].atom_id, a["v1"].atom_id],
        "preference": [a["v2"].atom_id, a["v1"].atom_id],
    }
    # count: one memory_atoms COUNT, no rows loaded
    # count：只对 memory_atoms 做一次 COUNT，不加载任何行
    listed = run(filters={"user_id": U, "scale": "state"}, limit=100)
    store.trace.reset()
    res = run(filters={"user_id": U, "scale": "state"}, aggregation="count")
    assert store.trace.tables() == {"memory_atoms"}
    assert res.count == len(listed.hits) == 5 and res.hits == [] and res.projection is None
    assert count_axis_query(store, AxisQuery(filters={"user_id": U, "scale": "state"})) == 5
    assert count_axis_query(store, AxisQuery(filters={"user_id": U, "scale": "state"}, include_expired=True)) == 7
    # summary
    # summary（摘要）
    res = run(filters=S1, aggregation="summary")
    assert res.summary == "2 atoms; latest: User usually runs every morning." and len(res.hits) == 2
    assert run(filters={"user_id": U, "topic": "nothing"}, aggregation="summary").summary == "0 atoms"
    # topk without text → VALIDATION_013
    # topk 缺少 text → VALIDATION_013
    # 模型校验器抛出的 HMS 错误会被 pydantic 包装
    with pytest.raises(Exception) as info:  # the model validator's HMS error is wrapped by pydantic
        AxisQuery(filters=S1, aggregation="topk")
    wrapped = info.value if isinstance(info.value, ValidationError) else unwrap_hms_error(info.value)
    assert wrapped is not None and wrapped.error_code == "VALIDATION_013"
    # scope derived from filters; explicit scope wins; neither → VALIDATION_001
    # scope 由 filters 推导；显式 scope 优先；两者都没有 → VALIDATION_001
    assert ids(run(filters={"user_id": U, "run_id": "run-b"})) == [a["sarah2"].atom_id]
    assert ids(run(filters={"user_id": U, "run_id": "run-b"}, scope=Scope(user_id=U, run_id="run-a"))) == [
        a["sarah1"].atom_id
    ]
    with pytest.raises(ValidationError) as info:
        run(filters={"topic": "coffee"})
    assert info.value.error_code == "VALIDATION_001"
    # multi-session scale constraint returns the entity atom plus the event atoms (HMS §4.2)
    # 跨会话的 scale 约束返回实体原子加事件原子（HMS §4.2）
    res = run(filters={"user_id": U, "entity": "Sarah"}, scale_constraint=["event", "entity", "L1_micro"])
    assert set(ids(res)) == {a["sarah1"].atom_id, a["sarah2"].atom_id, a["sarah_ent"].atom_id}
    assert [h.atom.atom_type for h in res.hits if h.atom.atom_id == a["sarah_ent"].atom_id] == ["entity"]
    # visibility lattice gate
    # 可见性格门控
    assert set(ids(run(filters={"user_id": U, "visibility": "team"}, include_expired=True))) == {
        a["habit"].atom_id,
        a["vac"].atom_id,
    }
    assert ids(run(filters={"user_id": U, "visibility": "public"}, include_expired=True)) == [a["vac"].atom_id]
    assert ids(run(filters={"user_id": U, "visibility": {"eq": "team"}}, include_expired=True)) == [a["habit"].atom_id]
    # meta_filters escape hatch and axis-only scores
    # meta_filters 逃生通道与纯轴命中的分数
    res = run(filters=S1, meta_filters={"priority": {"gte": 2}}, include_expired=True)
    assert ids(res) == [a["v2"].atom_id] and res.hits[0].score == 0.0 and res.hits[0].matched_axes == []
    # ranked latest/summary over a text query
    # 在文本查询之上做排序的 latest/summary
    res = run(
        filters={"user_id": U, "topic": "coffee"},
        text="coffee",
        aggregation="latest",
        project_axis="value",
        scale_constraint=["state"],
    )
    assert res.projection == [["black coffee"]] and res.hits[0].score > 0.0
    res = run(filters={"user_id": U}, text="coffee", aggregation="latest", project_axis="value", limit=3)
    groups = [(h.atom.subject, h.atom.predicate) for h in res.hits]
    # 每个 (subject, predicate) 分组取最新的一条……
    assert len(groups) == len(set(groups)) == 3  # newest per (subject, predicate) group ...
    assert [h.atom.time_anchor for h in res.hits] == sorted((h.atom.time_anchor for h in res.hits), reverse=True)
    # ……并按时间倒序的分组顺序排列
    assert res.hits[0].atom.time_anchor == "2026-06-15T10:14:00.000000+00:00"  # ... in time-desc group order
    res = run(filters=SCOPE4, text="retry.ts", aggregation="summary", limit=2)
    assert res.summary.startswith("8 atoms; latest: ") and len(res.hits) == 2 and res.primary_axis == "semantic"


# ---------------------------------------------------------------------------
# 36. Search fusion / explain / threshold
# 36. 搜索融合 / explain / 阈值
# ---------------------------------------------------------------------------
def test_search_fusion_explain_threshold_after_fusion(store, index, hash_embedder, seeded):
    """S4b identifier search math, exact explain keys, hub damping, active-signal max, determinism, preference gate.
    S4b 标识符搜索算术、精确的解释键、枢纽阻尼、活跃信号上限、确定性、偏好门控。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: HashEmbedder.
        hash_embedder：HashEmbedder。

        seeded: Atoms by name.
        seeded：按名称的原子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    expected = {a["e1"].atom_id, a["ep1"].atom_id, a["dc"].atom_id}
    store.trace.reset()
    hits = _search(store, index, hash_embedder, "retry.ts", SCOPE4, top_k=3, explain=True)
    touched = store.trace.tables()
    assert not any(t.startswith("graph_") for t in touched) and touched <= ALLOWED_TABLES
    assert {h["atom_id"] for h in hits} == expected
    for h in hits:
        sd = h["score_details"]
        assert "retry.ts" in h["hms"]["entity_ids"]
        assert list(sd) == EXPLAIN_KEYS
        assert sd["entity_boost"] == 1.0 and "entity" in sd["signals"] and "keyword" in sd["signals"]
        assert sd["signals"] == ["semantic", "keyword", "entity", "recency", "support"]
        assert sd["bm25_score"] == 1.0 and sd["keyword_backend"] == "overlap"
        assert sd["max_possible_score"] == pytest.approx(2.5) and sd["final_score"] == pytest.approx(
            sd["raw_score"] / 2.5
        )
        assert sd["final_score"] == h["score"] and sd["threshold"] == 0.0
        assert sd["axis_gates"] == {
            "time": None,
            "entity": None,
            "obs_type": None,
            "scope": SCOPE4,
            "validity": store.now(),
        }
        assert sd["primary_axis"] == "semantic" and sd["route"] == "recall"
        assert 0.0 < sd["recency_score"] < 1.0 and sd["support_score"] > 0.0
    # threshold applies AFTER fusion: the identifier hit with weak semantic survives threshold=0.3
    # 阈值在融合之后生效：语义分很弱的标识符命中仍能通过 threshold=0.3
    kept = _search(store, index, hash_embedder, "retry.ts", SCOPE4, top_k=10, threshold=0.3, explain=True)
    assert {h["atom_id"] for h in kept} == expected
    assert min(h["score_details"]["semantic_score"] for h in kept) < 0.3
    assert all(h["score"] >= 0.3 for h in kept)
    # hub damping: billing_e2e has degree 5 → entity_boost < 1.0 (exact formula)
    # 枢纽阻尼：billing_e2e 度数为 5 → entity_boost < 1.0（精确公式）
    hub = _search(store, index, hash_embedder, "billing_e2e", SCOPE4, top_k=5, explain=True)
    assert {h["atom_id"] for h in hub} == {a[n].atom_id for n in ("e2", "ep1", "dc", "e4", "ep2")}
    assert all(h["score_details"]["entity_boost"] == pytest.approx(1 / (1 + 0.001 * 4)) for h in hub)
    assert all(h["score_details"]["entity_boost"] < 1.0 for h in hub)
    assert store.entity_degree(store.resolve_entity(Scope(**SCOPE4), "billing_e2e")) == 5
    assert store.entity_degree(store.resolve_entity(Scope(**SCOPE4), "retry.ts")) == 3
    # max_possible reflects the active signals: no entity hint → 2.0; no content tokens either → 1.5
    # max_possible 反映当前活跃的信号：无实体线索 → 2.0；连内容词也没有 → 1.5
    plain = _search(store, index, hash_embedder, "coffee", {"user_id": U}, explain=True)
    assert plain and plain[0]["memory"] == "User likes black coffee."
    assert plain[0]["score_details"]["max_possible_score"] == pytest.approx(2.0)
    assert plain[0]["score_details"]["signals"] == ["semantic", "keyword", "recency", "support"]
    bare = _search(store, index, hash_embedder, "the", {"user_id": U}, explain=True)
    assert bare[0]["score_details"]["max_possible_score"] == pytest.approx(1.5)
    assert bare[0]["score_details"]["signals"] == ["semantic", "recency", "support"]
    # deterministic: two calls byte-identical; ordering is (final DESC, time_anchor DESC, atom_id ASC)
    # 确定性：两次调用逐字节相同；排序为 (final DESC, time_anchor DESC, atom_id ASC)
    first = json.dumps(_search(store, index, hash_embedder, "retry.ts", SCOPE4, explain=True), ensure_ascii=False)
    second = json.dumps(_search(store, index, hash_embedder, "retry.ts", SCOPE4, explain=True), ensure_ascii=False)
    assert first == second
    scores = [h["score"] for h in json.loads(first)]
    assert scores == sorted(scores, reverse=True)
    # preference route: obs_type gate (primary_axis 'type'), and fallback when nothing matches
    # preference 路由：obs_type 门控（primary_axis 为 'type'），无匹配时回退
    pref = _search(store, index, hash_embedder, "what coffee does the user like", {"user_id": U}, explain=True)
    assert pref[0]["memory"] == "User likes black coffee." and all("latte" not in h["memory"] for h in pref)
    assert {h["hms"]["obs_type"] for h in pref} <= {"preference_event", "habit"}
    sd = pref[0]["score_details"]
    assert sd["route"] == "preference" and sd["primary_axis"] == "type"
    assert sd["axis_gates"]["obs_type"] == ["preference_event", "habit"] and sd["axis_gates"]["scope"] == {"user_id": U}
    fallback = _search(store, index, hash_embedder, "what does the user like", SCOPE4, explain=True)
    assert fallback and fallback[0]["score_details"]["primary_axis"] == "fallback"
    assert fallback[0]["score_details"]["route"] == "preference"
    # explain=False rows carry score but no score_details; the result is Mem0-shaped
    # explain=False 的行带 score 但没有 score_details；结果是 Mem0 形状
    rows = _search(store, index, hash_embedder, "retry.ts", SCOPE4, top_k=1)
    assert len(rows) == 1 and "score" in rows[0] and "score_details" not in rows[0]
    assert list(rows[0])[:5] == ["id", "memory", "hash", "metadata", "score"]
    # scoring helpers directly
    # 直接测试打分辅助函数
    assert (
        mscoring.support_score(0) == 0.0 and mscoring.recency_score(None, as_of=store.now(), half_life_days=30) == 0.0
    )
    assert mscoring.keyword_score(["retry.ts", "change"], {"retry.ts"}, identifier_weight=2.0) == pytest.approx(2 / 3)
    assert mscoring.keyword_score([], {"x"}, identifier_weight=2.0) == 0.0
    assert (
        mscoring.entity_score({"a"}, {"b"}, degrees={}) == 0.0
        and mscoring.entity_score(set(), {"b"}, degrees={}) == 0.0
    )
    assert mscoring.entity_score({"a", "b"}, {"a"}, degrees={"a": 3}) == 0.5
    assert mscoring.recency_score(
        "2026-08-03T10:00:00Z", as_of="2026-09-02T10:00:00Z", half_life_days=30
    ) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 37. Hard time gate and fallback
# 37. 硬时间门控与回退
# ---------------------------------------------------------------------------
def test_search_hard_time_gate_and_fallback(store, index, hash_embedder, seeded):
    """A May query returns only May atoms (EN and ZH); an empty window falls back with primary_axis='fallback'.
    五月查询只返回五月原子（中英）；空窗口回退，primary_axis 为 'fallback'。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: HashEmbedder.
        hash_embedder：HashEmbedder。

        seeded: Atoms by name.
        seeded：按名称的原子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    may = ["2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00"]
    for query in ("what happened in May", "5月发生了什么"):
        rows = _search(store, index, hash_embedder, query, {"user_id": U}, explain=True)
        assert rows and {r["atom_id"] for r in rows} == {a["sarah2"].atom_id, a["habit"].atom_id}
        assert all(r["hms"]["time_anchor"].startswith("2026-05") for r in rows)
        sd = rows[0]["score_details"]
        assert sd["axis_gates"]["time"] == may and sd["primary_axis"] == "time" and sd["route"] == "temporal"
    rows = _search(
        store, index, hash_embedder, "what happened in May", {"user_id": U}, explain=True, include_expired=True
    )
    assert {r["atom_id"] for r in rows} == {a["sarah2"].atom_id, a["habit"].atom_id, a["vac"].atom_id}
    jan = _search(store, index, hash_embedder, "what happened in January", {"user_id": U}, explain=True)
    assert jan and jan[0]["score_details"]["primary_axis"] == "fallback"
    assert jan[0]["score_details"]["axis_gates"]["time"][0].startswith("2026-01-01")
    assert all(r["scale_level"] in {"event", "L1_micro", "state", "L3_macro"} for r in jan)
    # the temporal scale default excludes entity atoms; an explicit AxisQuery scale_constraint overrides it
    # temporal 路由的默认 scale 排除实体原子；显式的 AxisQuery scale_constraint 会覆盖它
    assert a["sarah_ent"].atom_id not in {r["atom_id"] for r in jan}
    res = _run(
        store,
        index,
        hash_embedder,
        filters={"user_id": U},
        text="what happened in May",
        aggregation="topk",
        plan=plan_query("what happened in May", as_of=store.now()).as_dict(),
        scale_constraint=["entity"],
    )
    assert res.primary_axis == "fallback" and _ids([h.atom for h in res.hits]) == [a["sarah_ent"].atom_id]


# ---------------------------------------------------------------------------
# 37b. Scale-gate fallback ladder (group Q1)
# 37b. 尺度门控回退阶梯（Q1 组）
# ---------------------------------------------------------------------------
def test_search_scale_gate_fallback_ladder(store, clock, index, hash_embedder, seeded):
    """A raw infer=False atom is found by a preference query; the ladder records scale_fallback / fallback honestly.
    infer=False 的原始原子能被偏好查询找到；阶梯如实记录 scale_fallback / fallback。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        clock: FrozenClock (ticked by _insert).
        clock：FrozenClock（由 _insert 推进）。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: HashEmbedder.
        hash_embedder：HashEmbedder。

        seeded: Atoms by name (the level-1 preference path must stay byte-identical).
        seeded：按名称的原子（第一级偏好路径必须保持逐字节相同）。

    Output / 输出：
        Assertions on every ladder level, the count aggregation, the explicit scale_constraint and level 1.
        对每个阶梯级别、计数聚合、显式 scale_constraint 与第一级的断言。

    State effect / 状态影响：
        Inserts atoms in three fresh user scopes.
        在三个新的用户范围内插入原子。
    """
    query = "What drink does Alice prefer?"
    plan = plan_query(query, as_of=store.now()).as_dict()
    assert plan["route"] == "preference" and plan["scale_default"] == ["state"]
    assert plan["hard_type_gate"] == ["obs_type", ["preference_event", "habit"]]
    # level 4: a raw message atom (L1_micro, obs_type None) is what add(infer=False) stores
    # 第四级：原始消息原子（L1_micro、obs_type 为 None）正是 add(infer=False) 所存的内容
    raw_scope = {"user_id": "carol"}
    raw_text = "Alice prefers jasmine tea and avoids coffee after 3 PM."
    raw = _insert(
        store,
        clock,
        name="raw",
        text=raw_text,
        ts="2026-08-01T08:00:00Z",
        scope=raw_scope,
        anchors=[_anchor(store, raw_scope, raw_text, "2026-08-01T08:00:00Z")],
        atom_type="message",
        scale_level="L1_micro",
        source_kind="message",
        subject=None,
        predicate=None,
    )
    index.upsert(raw.atom_id, hash_embedder.embed(raw.text), embedder=hash_embedder.identity)
    rows = _search(store, index, hash_embedder, query, raw_scope, explain=True)
    assert [r["atom_id"] for r in rows] == [raw.atom_id] and rows[0]["scale_level"] == "L1_micro"
    assert rows[0]["score_details"]["primary_axis"] == "fallback" and rows[0]["score"] > 0.0
    assert rows[0]["score_details"]["axis_gates"]["obs_type"] == ["preference_event", "habit"]
    # the facade default threshold (0.1) keeps the hit: the defect was an empty candidate set, not a low score
    # 门面默认阈值（0.1）保留该命中：缺陷在于候选集为空，而不是分数过低
    assert _search(store, index, hash_embedder, query, raw_scope, threshold=0.1) == _search(
        store, index, hash_embedder, query, raw_scope
    )
    res = _run(store, index, hash_embedder, filters=raw_scope, text=query, aggregation="topk", plan=plan)
    assert res.primary_axis == "fallback" and "type" not in res.hits[0].matched_axes
    count = _run(store, index, hash_embedder, filters=raw_scope, text=query, aggregation="count", plan=plan)
    assert (count.count, count.primary_axis) == (1, "fallback")
    # an explicit AxisQuery scale_constraint is a hard axis: the ladder relaxes the gates but never the scale
    # 显式的 AxisQuery scale_constraint 是硬轴：阶梯放宽门控但从不放宽尺度
    strict = _run(
        store, index, hash_embedder, filters=raw_scope, text=query, aggregation="topk", plan=plan, scale_constraint=["state"]
    )
    assert strict.hits == [] and strict.primary_axis == "fallback"
    # level 2: the type gate holds at a relaxed scale (a raw preference event beats an unrelated state fact)
    # 第二级：类型门控在放宽的尺度上成立（原始偏好事件胜过不相干的 state 事实）
    ev_scope = {"user_id": "dave"}
    fact_text = "Dave lives in Lisbon."
    fact = _insert(
        store,
        clock,
        name="dave_fact",
        text=fact_text,
        ts="2026-08-02T08:00:00Z",
        scope=ev_scope,
        anchors=[_anchor(store, ev_scope, fact_text, "2026-08-02T08:00:00Z")],
        obs_type="fact",
        predicate="lives_in",
    )
    pref_text = "Dave prefers oolong tea."
    pref = _insert(
        store,
        clock,
        name="dave_pref",
        text=pref_text,
        ts="2026-08-03T08:00:00Z",
        scope=ev_scope,
        anchors=[_anchor(store, ev_scope, pref_text, "2026-08-03T08:00:00Z")],
        atom_type="event",
        scale_level="L1_micro",
        source_kind="event",
        obs_type="preference_event",
    )
    for atom in (fact, pref):
        index.upsert(atom.atom_id, hash_embedder.embed(atom.text), embedder=hash_embedder.identity)
    rows = _search(store, index, hash_embedder, "What tea does Dave prefer?", ev_scope, explain=True)
    assert [r["atom_id"] for r in rows] == [pref.atom_id] and rows[0]["scale_level"] == "L1_micro"
    assert rows[0]["score_details"]["primary_axis"] == "scale_fallback"
    dave_plan = plan_query("What tea does Dave prefer?", as_of=store.now()).as_dict()
    res = _run(store, index, hash_embedder, filters=ev_scope, text="What tea does Dave prefer?", aggregation="topk", plan=dave_plan)
    assert res.primary_axis == "scale_fallback" and res.hits[0].matched_axes[0] == "type"
    count = _run(store, index, hash_embedder, filters=ev_scope, text="What tea does Dave prefer?", aggregation="count", plan=dave_plan)
    assert (count.count, count.primary_axis) == (1, "scale_fallback")
    # level 3 (the historical fallback): gates dropped, scale kept, so the entity atom stays excluded
    # 第三级（原有回退）：放弃门控、保留尺度，因此实体原子仍被排除
    ent_scope = {"user_id": "erin"}
    erin_text = "Erin works in Oslo."
    erin = _insert(
        store,
        clock,
        name="erin_fact",
        text=erin_text,
        ts="2026-08-04T08:00:00Z",
        scope=ent_scope,
        anchors=[_anchor(store, ent_scope, erin_text, "2026-08-04T08:00:00Z")],
        obs_type="fact",
        predicate="works_in",
    )
    erin_ent = _insert(
        store,
        clock,
        name="erin_ent",
        text="Erin",
        ts="2026-08-04T08:00:00Z",
        scope=ent_scope,
        anchors=[erin.anchor_ids[0]],
        atom_type="entity",
        scale_level="entity",
        source_kind="entity",
        source_id="ent_erin",
        subject="Erin",
        predicate="is_entity",
        content={"entity_id": "ent_erin", "entity_type": "PROPER", "aliases": [], "mentions": 1},
        entity_keys=[("erin", "Erin", "PROPER", ())],
    )
    for atom in (erin, erin_ent):
        index.upsert(atom.atom_id, hash_embedder.embed(atom.text), embedder=hash_embedder.identity)
    rows = _search(store, index, hash_embedder, "What does Erin prefer?", ent_scope, explain=True)
    assert [r["atom_id"] for r in rows] == [erin.atom_id]
    assert rows[0]["score_details"]["primary_axis"] == "fallback"
    # level 1 is untouched: the infer=True preference path keeps primary_axis 'type' and its exact scores
    # 第一级不受影响：infer=True 的偏好路径保持 primary_axis 为 'type' 及其精确分数
    pref_query = "what coffee does the user like"
    ranked = _search(store, index, hash_embedder, pref_query, {"user_id": U}, explain=True)
    assert ranked[0]["memory"] == "User likes black coffee." and ranked[0]["score_details"]["primary_axis"] == "type"
    level1 = _run(
        store,
        index,
        hash_embedder,
        filters={"user_id": U},
        text=pref_query,
        aggregation="topk",
        plan=plan_query(pref_query, as_of=store.now()).as_dict(),
        scale_constraint=["state"],
    )
    assert level1.primary_axis == "type"
    assert [(h.atom.atom_id, h.score) for h in level1.hits] == [(r["atom_id"], r["score"]) for r in ranked]


# ---------------------------------------------------------------------------
# 38. get_all
# 38. get_all（全量列出）
# ---------------------------------------------------------------------------
def test_get_all_show_expired_and_order(store, index, hash_embedder, seeded):
    """get_all returns live rows in axis order without score; include_expired shows archived/expired versions.
    get_all 按轴序返回存活行且无 score；include_expired 显示已归档/过期版本。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index (seeding dependency).
        index：参数化索引（播种依赖）。

        hash_embedder: HashEmbedder (seeding dependency).
        hash_embedder：HashEmbedder（播种依赖）。

        seeded: Atoms by name.
        seeded：按名称的原子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    now = store.now()
    rows = get_all(store, filters=S1, top_k=20, as_of=now, include_expired=False)
    assert [r["atom_id"] for r in rows] == [a["habit"].atom_id, a["v2"].atom_id]
    assert all("score" not in r and "score_details" not in r for r in rows)
    expired = get_all(store, filters=S1, top_k=20, as_of=now, include_expired=True)
    assert [r["atom_id"] for r in expired] == [a["habit"].atom_id, a["vac"].atom_id, a["v2"].atom_id, a["v1"].atom_id]
    archived = expired[-1]
    assert (
        archived["valid_to"] == "2026-04-20T08:00:00.000000+00:00" and archived["hms"]["archived_reason"] == "revised"
    )
    assert archived["id"] == expired[-2]["id"] and archived["version"] == 1 and expired[-2]["version"] == 2
    assert [
        r["atom_id"] for r in get_all(store, filters=S1, top_k=20, as_of="2026-05-15T00:00:00Z", include_expired=False)
    ] == [
        a["habit"].atom_id,
        a["vac"].atom_id,
        a["v2"].atom_id,
    ]
    assert get_all(store, filters=S1, top_k=1, as_of=now, include_expired=False)[0]["atom_id"] == a["habit"].atom_id
    assert get_all(store, filters=S1, top_k=0, as_of=now, include_expired=False) == []
    everything = get_all(store, filters={"user_id": U}, top_k=100, as_of=now, include_expired=True)
    keys = [(r["hms"]["time_anchor"], r["created_at"], r["atom_id"]) for r in everything]
    assert keys == sorted(keys, key=lambda k: (k[0], k[1]), reverse=True) and a["bob"].atom_id not in {
        r["atom_id"] for r in everything
    }
    assert [
        r["atom_id"]
        for r in get_all(store, filters={"user_id": U, "entity": "Sarah"}, top_k=20, as_of=now, include_expired=False)
    ] == [
        # 与 sarah3 的 time_anchor 相同 → 由 created_at DESC 打破平局
        a["sarah_ent"].atom_id,  # same time_anchor as sarah3 → created_at DESC breaks the tie
        a["sarah3"].atom_id,
        a["sarah2"].atom_id,
        a["sarah1"].atom_id,
    ]
    assert (
        len(
            get_all(
                store,
                filters={"user_id": U, "run_id": "run-b", "entity": "sarah"},
                top_k=20,
                as_of=now,
                include_expired=False,
            )
        )
        == 1
    )
    with pytest.raises(ValidationError) as info:
        get_all(store, filters={"topic": "coffee"}, top_k=20, as_of=now, include_expired=False)
    assert info.value.error_code == "VALIDATION_001"


# ---------------------------------------------------------------------------
# 39. Row shape
# 39. 行结构
# ---------------------------------------------------------------------------
def test_atom_to_row_shape_and_no_metadata_leak(store, index, hash_embedder, seeded):
    """§4.1 key order, metadata carries only caller keys, RESERVED_ATOM_KEYS covers every column, wrapper parity.
    §4.1 键序、metadata 仅含调用方键、RESERVED_ATOM_KEYS 覆盖全部列、包装函数与模型一致。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index (seeding dependency).
        index：参数化索引（播种依赖）。

        hash_embedder: HashEmbedder (seeding dependency).
        hash_embedder：HashEmbedder（播种依赖）。

        seeded: Atoms by name.
        seeded：按名称的原子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    a = seeded
    v2 = store.get_atom(a["v2"].atom_id)
    row = atom_to_row(v2)
    assert list(row) == ROW_KEYS and row == v2.to_public_dict()
    assert row["metadata"] == {"source": "chat", "priority": 2, "tags": ["morning"]}
    assert row["id"] == v2.memory_id and row["atom_id"] == v2.atom_id and row["version"] == 2
    assert set(row["hms"]) == {
        "time_anchor",
        "entity_ids",
        "topic_tags",
        "sentiment",
        "subject",
        "predicate",
        "obs_type",
        "source_kind",
        "source_id",
        "source_ep_ids",
        "extracted_by",
        "visibility",
        "archived_reason",
    }
    assert not set(row["metadata"]) & RESERVED_ATOM_KEYS
    columns = {r[1] for r in store._conn.execute("PRAGMA table_info(memory_atoms)")}
    assert "source_ep_ids_json" in columns and columns <= RESERVED_ATOM_KEYS
    e1 = store.get_atom(a["e1"].atom_id)
    row = atom_to_row(e1, score=0.5, score_details={"final_score": 0.5}, anchor_ids=["anc_x"])
    assert row == e1.to_public_dict(score=0.5, score_details={"final_score": 0.5}, anchor_ids=["anc_x"])
    assert list(row)[:7] == ["id", "memory", "hash", "metadata", "score", "created_at", "updated_at"]
    assert (
        row["metadata"] is None and row["actor_id"] == "coder" and "role" not in row and row["anchor_ids"] == ["anc_x"]
    )
    assert list(row)[-1] == "score_details" and row["hms"]["entity_ids"] == ["retry.ts"]
    assert mquery.atom_to_row(v2) == v2.to_public_dict()


# ---------------------------------------------------------------------------
# 33b. Scope keys in list / operator-dict / '*' form (Mem0 grammar on user_id / agent_id / run_id)
# 33b. 范围键的列表 / 操作符字典 / '*' 形式（user_id / agent_id / run_id 上的 Mem0 语法）
# ---------------------------------------------------------------------------
def test_scope_keys_accept_list_and_operator_forms(store, index, hash_embedder, seeded):
    """A top-level scope key with a list / operator dict / '*' is a column filter, not a Scope pin (Mem0 grammar).
    顶层范围键取列表 / 操作符字典 / '*' 时是列过滤而非 Scope 固定值（Mem0 语法）。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: 64-dim HashEmbedder.
        hash_embedder：64 维 HashEmbedder。

        seeded: Atoms by name (sarah1..3 live in run-a / run-b / run-c).
        seeded：按名称的原子（sarah1..3 位于 run-a / run-b / run-c）。

    Output / 输出：
        Assertions on the parse_filters AST, get_all / search rows and the HMS error codes (never a raw pydantic error).
        对 parse_filters 语法树、get_all / search 行与 HMS 错误码的断言（绝不是原始 pydantic 错误）。

    State effect / 状态影响：
        None.
        无。
    """
    scope, expr = parse_filters({"user_id": U, "run_id": {"in": ["run-a", "run-b"]}})
    assert scope == Scope(user_id=U) and expr == In(field="run_id", values=["run-a", "run-b"])
    assert parse_filters({"user_id": U, "run_id": ["run-a", "run-b"]})[1] == expr
    assert parse_filters({"user_id": U, "run_id": "*"}) == (Scope(user_id=U), Exists(field="run_id"))
    with pytest.raises(ValidationError) as info:
        parse_filters({"user_id": U, "run_id": {"like": "run-%"}})
    assert info.value.error_code == "VALIDATION_012" and isinstance(info.value, ValueError)
    # Scope 校验器的错误已从 pydantic 的包装中解包出来
    with pytest.raises(ValidationError) as info:  # the Scope validator's error is unwrapped from pydantic's wrapper
        parse_filters({"user_id": "a b"})
    assert info.value.error_code == "VALIDATION_001" and isinstance(info.value, ValueError)
    # 被过滤的范围键不是固定值：仍然需要一个纯 id
    with pytest.raises(ValidationError) as info:  # a filtered scope key is not a pin: a plain id is still required
        parse_filters({"run_id": {"in": ["run-a"]}})
    assert info.value.error_code == "VALIDATION_001"

    def rows(filters: dict) -> set[str]:
        return {r["id"] for r in get_all(store, filters=filters, top_k=50, as_of=store.now(), include_expired=False)}

    both = rows({"user_id": U, "run_id": {"in": ["run-a", "run-b"]}})
    assert both and both == rows({"user_id": U, "run_id": "run-a"}) | rows({"user_id": U, "run_id": "run-b"})
    assert rows({"user_id": U, "run_id": ["run-a", "run-b"]}) == both
    assert rows({"user_id": U, "run_id": {"nin": ["run-a", "run-b"]}}).isdisjoint(both)
    assert both <= rows({"user_id": U, "run_id": "*"}) <= rows({"user_id": U})
    hits = _search(store, index, hash_embedder, "Sarah", {"user_id": U, "run_id": ["run-a", "run-b"]})
    assert hits and {h["id"] for h in hits} <= both


# ---------------------------------------------------------------------------
# 33c. Scalar scope keys inside the top-level AND chain (Mem0 flattens AND before reading the scope)
# 33c. 顶层 AND 链中的标量范围键（Mem0 在读取 scope 前会展平 AND）
# ---------------------------------------------------------------------------
def test_scope_hoisted_from_top_level_and(store, index, hash_embedder, seeded):
    """A scalar scope key written inside the top-level `AND` pins the Scope and leaves every other leaf in the AST.
    写在顶层 `AND` 里的标量范围键会固定 Scope，AST 中其余叶节点保持不变。

    Input / 输入：
        store: The in-memory StateStore.
        store：内存 StateStore。

        index: The parametrized index.
        index：参数化索引。

        hash_embedder: 64-dim HashEmbedder.
        hash_embedder：64 维 HashEmbedder。

        seeded: Atoms by name (sarah1..3 live in run-a / run-b / run-c).
        seeded：按名称的原子（sarah1..3 位于 run-a / run-b / run-c）。

    Output / 输出：
        Assertions that parse_filters hoists the pin (nested AND included, OR/NOT untouched, caller dict unmutated)
        and that get_all / search return the same rows for the AND form and the equivalent flat form.
        断言 parse_filters 会上提固定值（含嵌套 AND，不动 OR/NOT，且不修改调用方字典），
        并断言 AND 写法与等价扁平写法在 get_all / search 下返回相同的行。

    State effect / 状态影响：
        None.
        无。
    """
    scope, expr = parse_filters({"AND": [{"user_id": U}, {"source": "chat"}]})
    # 固定值被上提；source 叶仍留在 AND 里
    assert scope == Scope(user_id=U)  # the pin is hoisted; the source leaf stays inside the AND
    assert expr == And(items=[Eq(field="meta.source", value="chat")])
    # 上提后 AND 变空：既不报错，也不留下空节点
    assert parse_filters({"AND": [{"user_id": U}]}) == (Scope(user_id=U), None)  # emptied AND: no error, no leftover
    # 裸字典等价于只有一个子句的 AND
    assert parse_filters({"AND": {"user_id": U}}) == (Scope(user_id=U), None)  # a bare dict is a one-clause AND
    # 嵌套在 AND 里的 AND 同样是必然约束
    nested = parse_filters({"AND": [{"AND": [{"user_id": U}]}, {"source": "chat"}]})  # an AND inside an AND
    assert nested == (Scope(user_id=U), And(items=[Eq(field="meta.source", value="chat")]))
    # 多个范围键可以分散在不同子句里
    multi = parse_filters({"AND": [{"user_id": U}, {"run_id": "run-a"}]})  # scope keys may sit in different clauses
    assert multi == (Scope(user_id=U, run_id="run-a"), None)
    # 顶层已定义该键 → 不上提，AND 里的写法仍是列过滤
    pinned_twice = parse_filters({"user_id": U, "AND": [{"user_id": "bob"}]})  # top level wins; the AND stays a leaf
    assert pinned_twice == (Scope(user_id=U), And(items=[Eq(field="user_id", value="bob")]))
    # A list / operator-dict / '*' scope form is not a pin anywhere: it stays a column-filter leaf.
    # 列表 / 操作符字典 / '*' 形式在任何位置都不是固定值，仍作为列过滤叶节点。
    kept = parse_filters({"AND": [{"user_id": U}, {"run_id": ["run-a", "run-b"]}, {"agent_id": "*"}]})
    assert kept == (
        Scope(user_id=U),
        And(items=[In(field="run_id", values=["run-a", "run-b"]), Exists(field="agent_id")]),
    )
    # parse_filters 绝不修改调用方的过滤字典
    caller = {"AND": [{"user_id": U}, {"source": "chat"}]}  # parse_filters never mutates the caller's filter dict
    parse_filters(caller)
    assert caller == {"AND": [{"user_id": U}, {"source": "chat"}]}

    def rows(filters: dict) -> set[str]:
        return {r["id"] for r in get_all(store, filters=filters, top_k=50, as_of=store.now(), include_expired=False)}

    flat = {"user_id": U, "run_id": {"in": ["run-a", "run-b"]}}
    conjunction = {"AND": [{"user_id": U}, {"run_id": {"in": ["run-a", "run-b"]}}]}
    assert rows(flat) and rows(conjunction) == rows(flat)
    assert rows({"AND": [{"user_id": U}, {"run_id": "run-a"}]}) == rows({"user_id": U, "run_id": "run-a"})
    hits = _search(store, index, hash_embedder, "Sarah", conjunction)
    assert hits and [h["id"] for h in hits] == [h["id"] for h in _search(store, index, hash_embedder, "Sarah", flat)]


# ---------------------------------------------------------------------------
# 33d. Scope keys under OR / NOT are not necessary constraints → still VALIDATION_001
# 33d. OR / NOT 下的范围键不是必然约束 → 仍抛 VALIDATION_001
# ---------------------------------------------------------------------------
def test_or_scope_still_rejected():
    """Only the top-level AND chain is hoisted: a scope key under OR / NOT stays a column leaf and pins nothing.
    只有顶层 AND 链会被上提：OR / NOT 下的范围键仍是列叶节点，不固定任何 Scope。

    Input / 输入：
        None (pure parse_filters calls).
        无（纯 parse_filters 调用）。

    Output / 输出：
        Assertions that OR / NOT / AND-nested-OR scope keys raise VALIDATION_001 with require_scope=True and remain
        column leaves with require_scope=False.
        断言 OR / NOT / AND 内嵌 OR 中的范围键在 require_scope=True 时抛 VALIDATION_001，
        在 require_scope=False 时仍为列叶节点。

    State effect / 状态影响：
        None.
        无。
    """
    for filters in (
        {"OR": [{"user_id": "u"}, {"user_id": "v"}]},
        {"NOT": [{"user_id": "u"}]},
        {"AND": [{"OR": [{"user_id": "u"}, {"user_id": "v"}]}]},
    ):
        with pytest.raises(ValidationError) as info:
            parse_filters(filters)
        assert info.value.error_code == "VALIDATION_001" and isinstance(info.value, ValueError)
    # 没有 require_scope 时，OR 分支保留 user_id 列叶节点
    scope, expr = parse_filters(  # without require_scope the OR branch keeps its user_id column leaves
        {"OR": [{"user_id": "u"}, {"user_id": "v"}]}, require_scope=False
    )
    assert scope == Scope() and expr == Or(items=[Eq(field="user_id", value="u"), Eq(field="user_id", value="v")])
