"""Group T golden scenarios: CONTRACT §19 S1–S6 (+ S4b and the epilogue) run in sequence on ONE HMS per index provider.
T 组黄金场景：契约 §19 的 S1–S6（含 S4b 与尾声）在每种索引提供者的同一个 HMS 上按顺序运行。
"""

# Purpose (EN): Assert every `expect` block of §19.1–§19.7 exactly (cards and chain byte-for-byte, transitions masked, ids symbolic).
# 目的（中）：逐一精确断言 §19.1–§19.7 的每个 `expect` 块（卡片与链逐字节、转移掩码比对、id 用符号解析）。

# Input (EN): tests/fixtures/events/*.json, tests/fixtures/golden/* and the conftest fixtures (frozen clock, sequential ids).
# 输入（中）：tests/fixtures/events/*.json、tests/fixtures/golden/* 与 conftest 夹具（冻结时钟、顺序 id）。

# Output (EN): One parametrized pytest result per index provider.
# 输出（中）：每种索引提供者一个参数化的 pytest 结果。

# State effect (EN): None outside the in-memory store and index; --update-golden rewrites golden files.
# 状态影响（中）：除内存存储与索引外无副作用；--update-golden 会重写黄金文件。

from __future__ import annotations

import hashlib
from typing import Any, Callable

import pytest

from hms_elite.models import Scope


@pytest.fixture
def config(index_provider: str):
    """HMSConfig for the §19 goldens: parametrized index, in-memory store, and extraction_output pinned to 'full'.
    §19 黄金场景的 HMSConfig：参数化索引、内存存储，并把 extraction_output 钉在 'full'。

    Input / 输入：
        index_provider: The conftest index provider parameter.
        index_provider：conftest 的索引提供者参数。

    Output / 输出：
        An HMSConfig; overrides the conftest `config` fixture for this module only.
        一个 HMSConfig；仅在本模块覆盖 conftest 的 `config` 夹具。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.configs.base import HMSConfig

    # CONTRACT §19 states its expected outputs on the rule LLM's full observations (R2 constraint confidence 0.9, R1
    # preference 0.8, the §19.8 payload hashes); the goldens pin the full contract explicitly rather than relying on
    # the library default ('lines' since 2026-09-18, §6.6 / changelog row 108 — the pin keeps these goldens on the full
    # contract; 'compact' / 'lines' would derive confidence 0.7 for every observation), so a change of the default
    # cannot rewrite them.
    # CONTRACT §19 的预期输出是在规则 LLM 的完整观察上陈述的（R2 约束置信度 0.9、R1 偏好 0.8、§19.8 的载荷哈希）；
    # 黄金场景显式钉住完整契约而不依赖库默认（2026-09-18 起为 'lines'，§6.6 / 变更记录第 108 行——钉住使黄金场景留在
    # 完整契约上；'compact' / 'lines' 会把每条观察的置信度推导为 0.7），这样默认值再变也不会改写黄金值。
    return HMSConfig(
        vector_store={"provider": index_provider}, store={"path": ":memory:"}, ingest={"extraction_output": "full"}
    )


U = "alice"
S1 = {"user_id": U, "run_id": "run-coffee"}
LATTE_MSG = "I like oat-milk lattes in the morning."
BLACK_MSG = "Actually I switched to black coffee now."
LATTE = "User likes oat-milk lattes in the morning."
BLACK = "User likes black coffee."
BLACK_TS = "2026-04-20T08:00:00.000000+00:00"
SARAH_1 = "Sarah joined the billing team in March."
SARAH_2 = "Sarah shipped the webhook retry design on May 6."
SARAH_3 = "Sarah is reviewing the checkout refactor this week."
S4_QUERY = "fix checkout webhook retry"
S4_AS_OF = "2026-06-15T12:00:00Z"
UC_TEXT = "Do not modify the subscription state machine."
EP1_A = "patch src/billing/retry.ts: add exponential backoff"
EP1_Y = "billing_e2e failed: subscription_state_regression"
EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
EP2_Y = "billing_e2e passed"
DC_TEXT = f"Do not re-apply '{EP1_A}': {EP1_Y}"
EP1_TEXT = f"[negative] {S4_QUERY} → {EP1_A} → {EP1_Y}"
E1_TEXT = "file_diff src/billing/retry.ts: add exponential backoff"
E2_TEXT = "test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression"
CORRECTION = "retry.ts may change once billing_e2e is isolated."
SKILL_TEXT = f"Skill: {UC_TEXT}"
UC_SOURCE_REF = "chat://agent_id=coder&run_id=run-billing&user_id=alice/0"
HASHES = {
    "uc": "3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc",
    "e1": "a27b81718fae86837895813746ec188afca9087544df8950c3b5832742ebfa8d",
    "e2": "e51db27bfdf98d59d0abccc1fb59f7413e1f9bbaf1260406fdc2154e4b7ce1e2",
    "e3": "cae8b909c617c029c16e12f5ef04e0906be8a797b07eca94132db1654344fb2a",
    "e4": "ea5edcbd2c3f0b79ef01e3768590081cf1a024f8081b4c2b2d17155dd68ff272",
}
SOURCE_REFS = {
    "e1": "file://src/billing/retry.ts",
    "e2": "test://billing_e2e",
    "e3": "file://src/billing/idempotency.ts",
    "e4": "test://billing_e2e",
}
KAPPA_KEYS = ["constraints", "visibility", "permission_scope", "provenance"]
CONSOLIDATE_REPORT = {
    "entity_atoms": 6,
    "temporal_edges": 0,
    "causal_edges": 0,
    "conflicts": 0,
    "interface_eps": 1,
    "projected_edges": 3,
    "interface_eps_l3": 0,
    "skills": 1,
    "transitions": 1,
    "compiled_states": 1,
}
GRAPH_STATS = {
    "nodes": {"message": 0, "observation": 3, "event": 4, "entity": 0, "ep": 2, "interface_ep": 1, "skill": 1},
    "edges": {"temporal": 1, "causal": 1, "use": 5, "distill": 3, "revision": 2, "conflict": 0},
    "valid_edges": 11,
    "invalid_edges": 1,
    "boundary_edges": 3,
    "interface_eps": 1,
    "unresolved_conflicts": 0,
    "witness_coverage": 1.0,
    "projection_coverage": 1.0,
    "traceable_high_edges": 1.0,
    "transitions": 6,
}


class _Runner:
    """Scenario runner: ticks the frozen clock 60 s before every public call and keeps the resolved symbolic ids.
    场景运行器：每次公开调用前将冻结时钟推进 60 秒，并保存解析出的符号 id。
    """

    def __init__(
        self, hms: Any, clock: Any, store: Any, golden: Any, events: Callable[[str], Any], scope4: dict
    ) -> None:
        """Bind the fixtures.
        绑定夹具。

        Input / 输入：
            hms: The HMS under test (rule LLM, hash embedder, parametrized index).
            hms：被测 HMS（规则 LLM、哈希嵌入器、参数化索引）。

            clock: FrozenClock fixture.
            clock：FrozenClock 夹具。

            store: The StateStore shared with `hms`.
            store：与 `hms` 共享的 StateStore。

            golden: Golden helper.
            golden：黄金比对辅助。

            events: Fixture loader `name -> FixtureDict`.
            events：夹具加载器 `name -> FixtureDict`。

            scope4: The S4 scope dict.
            scope4：S4 范围字典。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.hms = hms
        self.clock = clock
        self.store = store
        self.golden = golden
        self.events = events
        self.scope4 = scope4
        self.scope4_obj = Scope(**scope4)
        self.ids: dict[str, Any] = {}

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Advance the clock by 60 s, then invoke one public HMS method.
        将时钟推进 60 秒，然后调用一个公开的 HMS 方法。

        Input / 输入：
            fn: The bound HMS method.
            fn：绑定的 HMS 方法。

            args: Positional arguments.
            args：位置参数。

            kwargs: Keyword arguments.
            kwargs：关键字参数。

        Output / 输出：
            Whatever the method returns.
            方法的返回值。

        State effect / 状态影响：
            Mutates the frozen clock (and whatever the method mutates).
            修改冻结时钟（以及方法本身的副作用）。
        """
        self.clock.tick(seconds=60)
        return fn(*args, **kwargs)

    def edge(self, edge_type: str, src: str, dst: str) -> Any:
        """The valid L1 edge `src -type-> dst` (asserted to exist).
        有效的 L1 边 `src -type-> dst`（断言其存在）。

        Input / 输入：
            edge_type: Edge type name.
            edge_type：边类型名。

            src: Source node id.
            src：源节点 id。

            dst: Destination node id.
            dst：目标节点 id。

        Output / 输出：
            The GraphEdge.
            GraphEdge。

        State effect / 状态影响：
            None.
            无。
        """
        edge = self.store.find_edge(src, dst, edge_type)
        assert edge is not None, f"missing edge {edge_type} {src} -> {dst}"
        return edge

    def witnesses(self, edge_id: str) -> list[str]:
        """Witness anchor ids of an edge in witness order.
        某条边按见证顺序排列的锚点 id。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

        Output / 输出：
            Anchor id list.
            锚点 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return list(self.store.edge_witnesses(edge_id))

    def transitions_index(self) -> list[tuple[str, str]]:
        """`(source_kind, source_id)` of every transition row in scope4, export order.
        scope4 中每一条转移行的 `(source_kind, source_id)`，按导出顺序。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of pairs.
            成对列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [(t.source_kind, t.source_id) for t in self.store.list_transitions(self.scope4_obj)]


def _s1_preference_recall(run: _Runner) -> None:
    """§19.1 S1: preference recall with an ingest-time REVISE.
    §19.1 S1：偏好回忆与摄取期的 REVISE。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts; stores mem_coffee/atm_v1/atm_v2/anc_1/anc_2 in run.ids).
        无（断言；将 mem_coffee/atm_v1/atm_v2/anc_1/anc_2 存入 run.ids）。

    State effect / 状态影响：
        Adds two S1 messages to the HMS.
        向 HMS 添加两条 S1 消息。
    """
    hms, store, ids = run.hms, run.store, run.ids
    r1 = run.call(hms.add, LATTE_MSG, timestamp="2026-04-02T08:00:00Z", **S1)
    r2 = run.call(hms.add, BLACK_MSG, timestamp="2026-04-20T08:00:00Z", **S1)
    assert set(r1) == {"results"} and len(r1["results"]) == 1
    row1 = r1["results"][0]
    mem, v1 = row1["id"], row1["atom_id"]
    assert len(row1["anchor_ids"]) == 1
    anc1 = row1["anchor_ids"][0]
    assert (row1["memory"], row1["event"], row1["version"], row1["atom_type"], row1["scale_level"]) == (
        LATTE, "ADD", 1, "state", "state",
    )  # fmt: skip
    assert set(r2) == {"results"} and len(r2["results"]) == 1
    row2 = r2["results"][0]
    v2 = row2["atom_id"]
    assert (row2["id"], row2["memory"], row2["event"], row2["revises"], row2["version"]) == (
        mem,
        BLACK,
        "UPDATE",
        v1,
        2,
    )
    assert v2 != v1 and len(row2["anchor_ids"]) == 1 and row2["anchor_ids"][0] != anc1
    anc2 = row2["anchor_ids"][0]
    ids.update(mem_coffee=mem, atm_v1=v1, atm_v2=v2, anc_1=anc1, anc_2=anc2)

    got = run.call(hms.get, mem)
    assert got["memory"] == BLACK and got["version"] == 2 and got["valid_to"] is None
    hist = run.call(hms.history, mem)
    assert [h["event"] for h in hist] == ["ADD", "UPDATE"]
    update_row = hist[1]
    assert update_row["old_memory"] == LATTE and update_row["new_memory"] == BLACK
    assert update_row["anchor_id"] == anc2 and update_row["updated_at"] == BLACK_TS
    assert len(run.call(hms.get_all, filters=S1)["results"]) == 1
    all_rows = run.call(hms.get_all, filters=S1, show_expired=True)["results"]
    assert len(all_rows) == 2
    archived = [r for r in all_rows if r["atom_id"] == v1]
    assert len(archived) == 1
    assert archived[0]["valid_to"] == BLACK_TS and archived[0]["hms"]["archived_reason"] == "revised"

    hits = run.call(hms.search, "what coffee does the user like", filters={"user_id": U})["results"]
    assert hits[0]["memory"] == BLACK and all(h["memory"] != LATTE for h in hits)
    explained = run.call(hms.search, "what coffee does the user like", filters={"user_id": U}, explain=True)["results"]
    details = explained[0]["score_details"]
    assert details["route"] == "preference" and details["primary_axis"] == "type"
    assert details["axis_gates"]["obs_type"] == ["preference_event", "habit"]
    run.golden.masked(
        "search_explain_keys.json",
        {"score_details": list(details.keys()), "axis_gates": list(details["axis_gates"].keys())},
    )

    aq = {"filters": {"user_id": U, "topic": "coffee"}, "project_axis": "value", "scale_constraint": ["state"]}
    latest = run.call(hms.axis_query, {**aq, "aggregation": "latest"})
    assert latest.projection == [["black coffee"]] and len(latest.hits) == 1
    assert run.call(hms.axis_query, {**aq, "aggregation": "count"}).count == 1

    revisions = [e for e in store.list_edges(Scope(**S1), edge_types=["revision"]) if e.valid]
    assert len(revisions) == 1
    rev = revisions[0]
    assert (rev.src_id, rev.dst_id, rev.created_by) == (v1, v2, "ingest")
    assert rev.witness_anchor_ids == [anc2] and run.witnesses(rev.edge_id) == [anc2]

    ir = run.call(hms.compile, "coffee", filters=S1)
    assert [i.text for i in ir.preferences] == [BLACK] and ir.facts == []
    assert [i for i in ir.audit if i.explain.get("reason") == "revised"] == []
    deep = run.call(hms.compile, "coffee", filters=S1, mode="deep")
    revised = [i for i in deep.audit if i.explain.get("reason") == "revised"]
    assert len(revised) == 1
    assert revised[0].text == LATTE and revised[0].explain["superseded_by_text"] == BLACK


def _s2_s3_temporal_and_entity(run: _Runner) -> None:
    """§19.2 S2 temporal recall + S3 multi-session entity.
    §19.2 S2 时间回忆 + S3 多会话实体。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts).
        无（断言）。

    State effect / 状态影响：
        Adds the three Sarah messages under run-a/run-b/run-c.
        在 run-a/run-b/run-c 下添加三条 Sarah 消息。
    """
    hms, store = run.hms, run.store
    expected = [
        (SARAH_1, "2026-03-02T09:00:00Z", "run-a", "2026-03-01T00:00:00.000000+00:00", "event", "temporal_fact", "did", ["work", "billing"]),
        (SARAH_2, "2026-05-06T09:00:00Z", "run-b", "2026-05-06T00:00:00.000000+00:00", "event", "temporal_fact", "did", ["work", "engineering"]),
        (SARAH_3, "2026-06-11T09:00:00Z", "run-c", "2026-06-11T09:00:00.000000+00:00", "state", "fact", "is", ["engineering", "billing"]),
    ]  # fmt: skip
    for text, ts, run_id, time_anchor, atom_type, obs_type, predicate, topics in expected:
        res = run.call(hms.add, text, timestamp=ts, user_id=U, run_id=run_id)
        assert len(res["results"]) == 1
        row = res["results"][0]
        assert (row["memory"], row["event"], row["atom_type"]) == (text, "ADD", atom_type)
        assert row["hms"]["time_anchor"] == time_anchor
        assert (row["hms"]["obs_type"], row["hms"]["predicate"], row["hms"]["subject"]) == (
            obs_type,
            predicate,
            "Sarah",
        )
        assert row["hms"]["topic_tags"] == topics and "sarah" in row["hms"]["entity_ids"]

    may = run.call(hms.search, "what happened in May", filters={"user_id": U})["results"]
    assert [r["memory"] for r in may] == [SARAH_2]
    may_x = run.call(hms.search, "what happened in May", filters={"user_id": U}, explain=True)["results"]
    details = may_x[0]["score_details"]
    assert details["axis_gates"]["time"] == ["2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00"]
    assert details["primary_axis"] == "time" and details["route"] == "temporal"
    run.golden.masked(
        "search_explain_keys.json",
        {"score_details": list(details.keys()), "axis_gates": list(details["axis_gates"].keys())},
    )

    by_entity = run.call(hms.get_all, filters={"user_id": U, "entity": "Sarah"})["results"]
    assert [r["memory"] for r in by_entity] == [SARAH_3, SARAH_2, SARAH_1]
    run_b = run.call(hms.get_all, filters={"user_id": U, "run_id": "run-b", "entity": "sarah"})["results"]
    assert len(run_b) == 1 and run_b[0]["memory"] == SARAH_2
    sarah_rows = [e for e in store.entities_in_scope(Scope(user_id=U)) if e["normalized_text"] == "sarah"]
    assert len(sarah_rows) == 1 and sarah_rows[0]["user_id"] == U and sarah_rows[0]["agent_id"] is None
    entity_id = store.resolve_entity(Scope(user_id=U), "sarah")
    assert entity_id == sarah_rows[0]["entity_id"] and store.entity_degree(entity_id) == 3


def _check_anchor(run: _Runner, anchor_id: str, *, source_ref: str, key: str) -> None:
    """Assert an anchor's source_ref and full §19.8 payload hash.
    断言锚点的 source_ref 与 §19.8 的完整载荷哈希。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

        anchor_id: The anchor to check.
        anchor_id：待检查的锚点。

        source_ref: Expected `anchors.source_ref`.
        source_ref：期望的 `anchors.source_ref`。

        key: HASHES key.
        key：HASHES 的键。

    Output / 输出：
        None (asserts).
        无（断言）。

    State effect / 状态影响：
        None.
        无。
    """
    anchor = run.store.get_anchor(anchor_id)
    assert anchor is not None and anchor.source_ref == source_ref and anchor.hash == HASHES[key]
    assert anchor.revoked_at is None


def _s4_work_state_card(run: _Runner) -> None:
    """§19.3 S4: constraint + events_fail → card1, events_fix → card2, ablation at the card2 stage.
    §19.3 S4：约束 + events_fail → card1，events_fix → card2，以及 card2 阶段的消融。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts; stores uc/mem_uc/anc_uc/e1..e4/anc_e1..e4/ep1/ep2/ep1_atom/ep2_atom/dc/card2 in run.ids).
        无（断言；将 uc/mem_uc/anc_uc/e1..e4/anc_e1..e4/ep1/ep2/ep1_atom/ep2_atom/dc/card2 存入 run.ids）。

    State effect / 状态影响：
        Ingests the S4 fixture into scope4.
        将 S4 夹具摄取到 scope4。
    """
    hms, store, ids, scope4 = run.hms, run.store, run.ids, run.scope4
    fixture = run.events("structured_billing")
    assert fixture.scope == scope4 and fixture.constraint["content"] == UC_TEXT

    r0 = run.call(hms.add, UC_TEXT, timestamp=fixture.constraint["timestamp"], **scope4)
    assert len(r0["results"]) == 1
    uc_row = r0["results"][0]
    assert uc_row["event"] == "ADD" and uc_row["memory"] == UC_TEXT and uc_row["atom_type"] == "state"
    assert (uc_row["hms"]["obs_type"], uc_row["hms"]["subject"], uc_row["hms"]["predicate"]) == ("constraint", "project", "must_not")  # fmt: skip
    assert uc_row["hms"]["topic_tags"] == ["billing", "constraint"] and uc_row["confidence"] == 0.9
    uc, mem_uc, anc_uc = uc_row["atom_id"], uc_row["id"], uc_row["anchor_ids"][0]
    assert uc_row["anchor_ids"] == [anc_uc]
    _check_anchor(run, anc_uc, source_ref=UC_SOURCE_REF, key="uc")
    ids.update(uc=uc, mem_uc=mem_uc, anc_uc=anc_uc)

    # --- events_fail ---------------------------------------------------------------------------------------------
    # --- events_fail（失败事件批次） -----------------------------------------------------------------------------
    ra = run.call(hms.add_events, fixture.events_fail, **scope4)
    assert set(ra) == {"results", "events", "eps", "edges", "constraints"}
    assert [(e["object_type"], e["dedup"]) for e in ra["events"]] == [("file_diff", False), ("test_log", False)]
    anc_e1, anc_e2 = ra["events"][0]["anchor_id"], ra["events"][1]["anchor_id"]
    _check_anchor(run, anc_e1, source_ref=SOURCE_REFS["e1"], key="e1")
    _check_anchor(run, anc_e2, source_ref=SOURCE_REFS["e2"], key="e2")
    event_rows = [r for r in ra["results"] if r["atom_type"] == "event"]
    assert [r["memory"] for r in event_rows] == [E1_TEXT, E2_TEXT]
    assert [r["anchor_ids"] for r in event_rows] == [[anc_e1], [anc_e2]]
    e1, e2 = event_rows[0]["atom_id"], event_rows[1]["atom_id"]
    assert len(ra["eps"]) == 1 and len(ra["edges"]) == 2 and len(ra["constraints"]) == 1
    ep1 = store.get_ep(ra["eps"][0])
    assert ep1 is not None and ep1.ep_id == ra["eps"][0]
    assert (ep1.slots.u, ep1.slots.b, ep1.slots.a, ep1.slots.y, ep1.slots.r) == (S4_QUERY, UC_TEXT, EP1_A, EP1_Y, "negative")  # fmt: skip
    assert ep1.slots.kappa.model_dump(mode="json") == {"constraints": [UC_TEXT], "visibility": "private", "evidence": [anc_e2, anc_e1]}  # fmt: skip
    assert ep1.anchor_ids == [anc_e2, anc_e1] and ep1.time_anchor == "2026-06-15T10:04:00.000000+00:00"
    ep1_atom = store.get_atom(ep1.atom_id)
    assert ep1_atom is not None and ep1_atom.text == EP1_TEXT and ep1_atom.atom_type == "episode"
    assert set(ep1_atom.entity_ids) == {"retry.ts", "billing_e2e", "subscription_state_regression"}
    ep_row = [r for r in ra["results"] if r["atom_id"] == ep1_atom.atom_id]
    assert len(ep_row) == 1 and ep_row[0]["memory"] == EP1_TEXT
    dc = store.get_atom(ra["constraints"][0])
    assert dc is not None and dc.text == DC_TEXT
    assert (dc.obs_type, dc.subject, dc.predicate) == ("constraint", "project", "must_not")
    assert dc.anchor_ids == [anc_e2, anc_e1] and "retry.ts" in dc.entity_ids
    use_ep1_uc = run.edge("use", ep1_atom.atom_id, uc)
    distill_ep1_dc = run.edge("distill", ep1_atom.atom_id, dc.atom_id)
    assert run.witnesses(use_ep1_uc.edge_id) == [anc_e1] and use_ep1_uc.confidence == 0.8
    assert run.witnesses(distill_ep1_dc.edge_id) == [anc_e2] and distill_ep1_dc.confidence == 0.9
    assert set(ra["edges"]) == {use_ep1_uc.edge_id, distill_ep1_dc.edge_id}
    assert all(e.created_by == "ingest" for e in (use_ep1_uc, distill_ep1_dc))
    assert run.transitions_index() == [("ep", ep1.ep_id)]
    ids.update(
        e1=e1, e2=e2, anc_e1=anc_e1, anc_e2=anc_e2, ep1=ep1.ep_id, ep1_atom=ep1_atom.atom_id, dc=dc.atom_id,
        edg_use_ep1_uc=use_ep1_uc.edge_id, edg_distill_ep1_dc=distill_ep1_dc.edge_id,
    )  # fmt: skip

    # --- card1 -----------------------------------------------------------------------------------------------------
    # --- card1（第一张工作状态卡） ---------------------------------------------------------------------------------
    ir1 = run.call(hms.compile, S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    card1 = ir1.to_text()
    run.golden.text("work_state_card_failed.txt", card1)
    assert [c.item_id for c in ir1.constraints] == [uc, dc.atom_id]
    assert [e.item_id for e in ir1.episodes] == [ep1_atom.atom_id]
    assert ir1.facts == [] and ir1.preferences == [] and ir1.skills == []
    assert ir1.retrieval["dropped_covered"] == 2
    assert len(ir1.audit) == 3 and all(i.explain.get("kind") == "evidence" for i in ir1.audit)
    assert ir1.unresolved_conflicts == [] and ir1.route == "recall"
    assert ir1.retrieval["graph_hops"] == 1 and ir1.retrieval["ablation"] == "axis_graph_support"
    assert ir1.used_tokens <= 800 and ir1.mode == "auto"

    # --- events_fix ----------------------------------------------------------------------------------------------
    # --- events_fix（修复事件批次） ------------------------------------------------------------------------------
    rb = run.call(hms.add_events, fixture.events_fix, **scope4)
    assert [(e["object_type"], e["dedup"]) for e in rb["events"]] == [("file_diff", False), ("test_log", False)]
    anc_e3, anc_e4 = rb["events"][0]["anchor_id"], rb["events"][1]["anchor_id"]
    _check_anchor(run, anc_e3, source_ref=SOURCE_REFS["e3"], key="e3")
    _check_anchor(run, anc_e4, source_ref=SOURCE_REFS["e4"], key="e4")
    fix_rows = [r for r in rb["results"] if r["atom_type"] == "event"]
    assert [r["anchor_ids"] for r in fix_rows] == [[anc_e3], [anc_e4]]
    e3, e4 = fix_rows[0]["atom_id"], fix_rows[1]["atom_id"]
    assert len(rb["eps"]) == 1 and len(rb["edges"]) == 4 and rb["constraints"] == []
    ep2 = store.get_ep(rb["eps"][0])
    assert ep2 is not None
    assert (ep2.slots.u, ep2.slots.b, ep2.slots.a, ep2.slots.y, ep2.slots.r) == (
        S4_QUERY, f"{UC_TEXT} ; {DC_TEXT}", EP2_A, EP2_Y, "positive",
    )  # fmt: skip
    assert ep2.slots.kappa.constraints == [UC_TEXT, DC_TEXT] and ep2.anchor_ids == [anc_e4, anc_e3]
    ep2_atom = store.get_atom(ep2.atom_id)
    assert ep2_atom is not None and ep2_atom.text == f"[positive] {S4_QUERY} → {EP2_A} → {EP2_Y}"
    use_ep2_uc = run.edge("use", ep2_atom.atom_id, uc)
    use_ep2_dc = run.edge("use", ep2_atom.atom_id, dc.atom_id)
    temporal = run.edge("temporal", ep1_atom.atom_id, ep2_atom.atom_id)
    causal = run.edge("causal", ep1_atom.atom_id, ep2_atom.atom_id)
    assert run.witnesses(use_ep2_uc.edge_id) == [anc_e3] and run.witnesses(use_ep2_dc.edge_id) == [anc_e3]
    assert run.witnesses(temporal.edge_id) == [anc_e4] and temporal.confidence == 0.5
    assert run.witnesses(causal.edge_id) == [anc_e3, anc_e2] and causal.confidence == 0.7
    assert set(rb["edges"]) == {use_ep2_uc.edge_id, use_ep2_dc.edge_id, temporal.edge_id, causal.edge_id}
    assert run.transitions_index() == [("ep", ep1.ep_id), ("ep", ep2.ep_id)]
    ids.update(
        e3=e3, e4=e4, anc_e3=anc_e3, anc_e4=anc_e4, ep2=ep2.ep_id, ep2_atom=ep2_atom.atom_id,
        edg_use_ep2_uc=use_ep2_uc.edge_id, edg_use_ep2_dc=use_ep2_dc.edge_id,
        edg_temporal=temporal.edge_id, edg_causal=causal.edge_id,
    )  # fmt: skip

    # --- card2 + ablation ------------------------------------------------------------------------------------------
    # --- card2 + ablation（第二张卡与消融对比） --------------------------------------------------------------------
    ir2 = run.call(hms.compile, S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    card2 = ir2.to_text()
    run.golden.text("work_state_card_fixed.txt", card2)
    assert ir2.retrieval["dropped_covered"] == 4
    assert [e.item_id for e in ir2.episodes] == [ep1_atom.atom_id, ep2_atom.atom_id] and ir2.facts == []
    pcir = ir2.to_pcir()
    assert pcir.ok() is True and pcir.human_gloss == card2
    statuses = {ob.name: ob.status for ob in pcir.proof_obligations}
    assert statuses["governance"] == "skipped"
    assert all(status == "satisfied" for name, status in statuses.items() if name != "governance")
    assert {"evidence_coverage", "no_unresolved_conflict", "governance", "not_revoked"} <= set(statuses)
    deep = run.call(hms.compile, S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="deep")
    assert deep.to_text().replace("Mode: deep", "Mode: auto") == card2
    axis_graph = run.call(hms.compile, S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="axis_graph")
    assert axis_graph.to_text().replace("Mode: axis_graph", "Mode: auto") == card2
    assert axis_graph.retrieval["ablation"] == "axis_graph"
    run.clock.tick(seconds=60)
    store.trace.reset()
    fast = hms.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="fast")
    assert not {t for t in store.trace.tables() if t.startswith("graph_")}
    assert fast.retrieval["ablation"] == "axis" and fast.retrieval["graph_hops"] == 0
    ids.update(card1=card1, card2=card2)


def _s4b_search_by_identifier(run: _Runner) -> None:
    """§19.4 S4b: identifier search hits the event atom, the EP atom and the derived constraint.
    §19.4 S4b：标识符搜索命中事件原子、EP 原子与派生约束。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts).
        无（断言）。

    State effect / 状态影响：
        None (read-only calls).
        无（只读调用）。
    """
    hms, ids = run.hms, run.ids
    hits = run.call(hms.search, "retry.ts", filters=run.scope4, top_k=3)["results"]
    assert {h["atom_id"] for h in hits} == {ids["e1"], ids["ep1_atom"], ids["dc"]}
    assert all("retry.ts" in h["hms"]["entity_ids"] for h in hits)
    assert all("score_details" not in h for h in hits)
    explained = run.call(hms.search, "retry.ts", filters=run.scope4, top_k=3, explain=True)["results"]
    assert {h["atom_id"] for h in explained} == {ids["e1"], ids["ep1_atom"], ids["dc"]}
    for hit in explained:
        details = hit["score_details"]
        assert details["entity_boost"] == 1.0 and "entity" in details["signals"]
        assert list(details.keys()) == list(explained[0]["score_details"].keys())


def _s5_why(run: _Runner) -> None:
    """§19.5 S5: detective why() in question mode, by id, and the NOT_FOUND path.
    §19.5 S5：问题模式、按 id 的侦探式 why() 与 NOT_FOUND 路径。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts; chain compared byte-for-byte with why_chain.txt).
        无（断言；链与 why_chain.txt 逐字节比对）。

    State effect / 状态影响：
        None (read-only calls).
        无（只读调用）。
    """
    hms, ids = run.hms, run.ids
    trace = run.call(hms.why, "why must retry.ts not be modified", filters=run.scope4)
    assert set(trace) == {
        "target", "nodes", "edges", "anchors", "interface_eps", "paths", "unresolved_conflicts", "refined_from",
        "chain", "explanation",
    }  # fmt: skip
    assert trace["target"] == {"kind": "atom", "id": ids["dc"], "node_id": ids["dc"], "text": DC_TEXT, "resolved_from": "query"}  # fmt: skip
    assert {n["node_id"] for n in trace["nodes"]} == {ids["dc"], ids["ep1_atom"], ids["ep2_atom"], ids["uc"]}
    assert {(e["src_id"], e["dst_id"], e["edge_type"]) for e in trace["edges"]} == {
        (ids["ep1_atom"], ids["dc"], "distill"),
        (ids["ep2_atom"], ids["dc"], "use"),
        (ids["ep1_atom"], ids["uc"], "use"),
        (ids["ep1_atom"], ids["ep2_atom"], "causal"),
        (ids["ep2_atom"], ids["uc"], "use"),
    }
    anchor_ids = {a["anchor_id"] for a in trace["anchors"]}
    assert all(e["witness_anchor_ids"] and set(e["witness_anchor_ids"]) <= anchor_ids for e in trace["edges"])
    assert trace["unresolved_conflicts"] == [] and trace["refined_from"] == [] and trace["interface_eps"] == []
    assert set(trace["paths"]) == {n["node_id"] for n in trace["nodes"]} and trace["paths"][ids["dc"]] == []
    run.golden.text("why_chain.txt", "\n".join(trace["chain"]))
    assert trace["explanation"] == "\n".join(trace["chain"]) and len(trace["chain"]) == 3
    by_id = run.call(hms.why, ids["ep1"])
    assert by_id["target"]["resolved_from"] == "id" and by_id["target"]["node_id"] == ids["ep1_atom"]
    run.clock.tick(seconds=60)
    with pytest.raises(ValueError) as exc_info:
        hms.why("nonsense-id")
    assert getattr(exc_info.value, "error_code", None) == "NOT_FOUND_001"


def _s6_checkpoint_feedback_rollback(run: _Runner) -> None:
    """§19.6 S6: checkpoint → state_outdated feedback → mid compile → rollback → transitions.
    §19.6 S6：检查点 → state_outdated 反馈 → 中间编译 → 回滚 → 转移。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts; stores cp/uc2/anc_fb/edg_rev/anc_rb/edg_rb in run.ids).
        无（断言；将 cp/uc2/anc_fb/edg_rev/anc_rb/edg_rb 存入 run.ids）。

    State effect / 状态影响：
        Writes checkpoint, feedback, rollback and transition rows.
        写入检查点、反馈、回滚与转移行。
    """
    hms, store, ids, scope4 = run.hms, run.store, run.ids, run.scope4
    uc, mem_uc, dc = ids["uc"], ids["mem_uc"], ids["dc"]

    cp = run.call(hms.checkpoint, filters=scope4, label="before-feedback")
    assert isinstance(cp, str) and store.get_checkpoint(cp).label == "before-feedback"
    fb = run.call(hms.feedback, mem_uc, "state_outdated", correction=CORRECTION)
    feedback_instant = run.clock.now()
    assert fb["message"] == "Feedback applied" and set(fb) == {"message", "feedback_id", "actions"}
    assert len(fb["actions"]) == 3
    assert fb["actions"][0] == {"action": "archive", "atom_id": uc}
    assert fb["actions"][1]["action"] == "create" and fb["actions"][1]["atom_id"] != uc
    uc2 = fb["actions"][1]["atom_id"]
    assert set(fb["actions"][2]) == {"action", "edge_id"} and fb["actions"][2]["action"] == "revision_edge"
    edg_rev = fb["actions"][2]["edge_id"]
    # §16.1 action dicts are {'action', ...ids}: the successor's create action also names its (unchanged) lineage id.
    # §16.1 action 字典形如 {'action', ...ids}：后继的 create 动作同样给出其（未变的）lineage id。
    assert fb["actions"] == [
        {"action": "archive", "atom_id": uc},
        {"action": "create", "atom_id": uc2, "memory_id": mem_uc},
        {"action": "revision_edge", "edge_id": edg_rev},
    ]
    rev_edge = store.get_edge(edg_rev)
    assert (rev_edge.edge_type, rev_edge.src_id, rev_edge.dst_id, rev_edge.created_by) == ("revision", uc, uc2, "feedback")  # fmt: skip
    feedback_rows = store.list_feedback(mem_uc)
    assert len(feedback_rows) == 1 and feedback_rows[0].feedback_id == fb["feedback_id"]
    anc_fb = feedback_rows[0].anchor_id
    got = run.call(hms.get, mem_uc)
    assert got["memory"] == CORRECTION and got["version"] == 2 and got["atom_id"] == uc2
    successor = store.get_atom(uc2)
    assert {"retry.ts", "billing_e2e"} <= set(successor.entity_ids) and successor.obs_type == "constraint"
    assert successor.anchor_ids[0] == anc_fb and successor.anchor_ids == [anc_fb, ids["anc_uc"]]
    assert successor.valid_from == feedback_instant and successor.time_anchor == feedback_instant

    mid = run.call(hms.compile, S4_QUERY, filters=scope4)
    assert [c.text for c in mid.constraints] == [DC_TEXT, CORRECTION]
    assert [c.item_id for c in mid.constraints] == [dc, uc2]
    revised = [i for i in mid.audit if i.explain.get("reason") == "revised"]
    assert len(revised) == 1 and revised[0].text == UC_TEXT
    assert mid.as_of == store.now() == run.clock.now()

    rb = run.call(hms.rollback, cp)
    assert set(rb) == {
        "message", "checkpoint_id", "restored", "archived", "edges_invalidated", "edges_revalidated", "edges_created",
        "anchor_id",
    }  # fmt: skip
    assert len(rb["edges_created"]) == 1
    edg_rb, anc_rb = rb["edges_created"][0], rb["anchor_id"]
    assert rb == {
        "message": "Rolled back to before-feedback",
        "checkpoint_id": cp,
        "restored": [uc],
        "archived": [uc2],
        "edges_invalidated": [edg_rev],
        "edges_revalidated": [],
        "edges_created": [edg_rb],
        "anchor_id": anc_rb,
    }
    got = run.call(hms.get, mem_uc)
    assert got["memory"] == UC_TEXT and got["version"] == 1 and got["atom_id"] == uc
    rb_edge = store.get_edge(edg_rb)
    assert (rb_edge.edge_type, rb_edge.src_id, rb_edge.dst_id) == ("revision", uc2, uc)
    assert rb_edge.label == f"rollback:{cp}" and rb_edge.created_by == "ops" and rb_edge.valid
    assert rb_edge.witness_anchor_ids == [anc_rb] and run.witnesses(edg_rb) == [anc_rb]
    assert store.get_edge(edg_rev).valid is False
    assert [h["event"] for h in run.call(hms.history, mem_uc)] == ["ADD", "FEEDBACK", "UPDATE", "ROLLBACK", "RESTORE"]
    ids.update(cp=cp, uc2=uc2, anc_fb=anc_fb, edg_rev=edg_rev, anc_rb=anc_rb, edg_rb=edg_rb)

    tr = run.call(hms.transitions, filters=scope4)
    assert len(tr) == 5 and all(isinstance(t, tuple) and len(t) == 6 and t[5] for t in tr)
    assert [set(t[4].keys()) for t in tr] == [set(KAPPA_KEYS)] * 5
    s_t, a_t, s_t1, r_t, kappa, w_t = tr[0]
    assert (a_t, r_t, w_t) == (EP1_A, "negative", [ids["anc_e2"], ids["anc_e1"]])
    assert s_t["u"] == S4_QUERY and s_t["ep_id"] == ids["ep1"] and s_t1["y"] == EP1_Y
    assert kappa["constraints"] == [UC_TEXT]
    assert kappa["provenance"] == {"evidence": [ids["anc_e2"], ids["anc_e1"]], "ep_id": ids["ep1"]}
    s_t, a_t, s_t1, r_t, kappa, w_t = tr[1]
    assert (a_t, r_t, w_t) == (EP2_A, "positive", [ids["anc_e4"], ids["anc_e3"]])
    assert s_t["ep_id"] == ids["ep2"] and s_t1["ep_id"] == ids["ep2"]
    assert kappa["provenance"] == {"evidence": [ids["anc_e4"], ids["anc_e3"]], "ep_id": ids["ep2"]}
    s_t, a_t, s_t1, r_t, kappa, w_t = tr[2]
    assert s_t["ep_id"] == ids["ep1"] and s_t["y"] == EP1_Y and s_t1["ep_id"] == ids["ep2"]
    assert (a_t, r_t, w_t) == (EP2_A, "positive", [ids["anc_e3"], ids["anc_e2"], ids["anc_e4"]])
    assert kappa["constraints"] == [UC_TEXT, DC_TEXT] and kappa["provenance"]["edge_id"] == ids["edg_causal"]
    s_t, a_t, s_t1, r_t, kappa, w_t = tr[3]
    assert (s_t["text"], s_t["atom_id"], a_t, s_t1["text"], s_t1["atom_id"], r_t) == (
        UC_TEXT, uc, "revise", CORRECTION, uc2, "negative",
    )  # fmt: skip
    assert w_t == [anc_fb, ids["anc_uc"]]
    assert kappa == {
        "constraints": [DC_TEXT, CORRECTION],
        "visibility": "private",
        "permission_scope": "private",
        "provenance": {"label": "", "created_by": "feedback", "proposer": "rule"},
    }
    s_t, a_t, s_t1, r_t, kappa, w_t = tr[4]
    assert (s_t["text"], s_t["atom_id"], a_t, s_t1["text"], s_t1["atom_id"], r_t) == (
        CORRECTION, uc2, "rollback", UC_TEXT, uc, "neutral",
    )  # fmt: skip
    assert w_t == [anc_rb]
    assert kappa == {
        "constraints": [UC_TEXT, DC_TEXT],
        "visibility": "private",
        "permission_scope": "private",
        "provenance": {"checkpoint_id": cp, "label": "before-feedback", "reason": None},
    }
    assert run.transitions_index() == [
        ("ep", ids["ep1"]), ("ep", ids["ep2"]), ("causal", ids["edg_causal"]), ("revision", edg_rev), ("rollback", edg_rb),
    ]  # fmt: skip
    jsonl = run.call(hms.transitions, filters=scope4, fmt="jsonl")
    assert isinstance(jsonl, str) and jsonl.count("\n") == 4
    run.golden.masked("transitions_s6.jsonl", jsonl)
    assert run.call(hms.transitions, filters=scope4) == tr and store.stats()["transitions"] == 5


def _epilogue_consolidate_and_stats(run: _Runner) -> None:
    """§19.7 epilogue: consolidate(), the InterfaceEP/skill/transition it creates, graph_stats and the ablation block.
    §19.7 尾声：consolidate()、其创建的 InterfaceEP/技能/转移、graph_stats 与消融块。

    Input / 输入：
        run: The scenario runner.
        run：场景运行器。

    Output / 输出：
        None (asserts).
        无（断言）。

    State effect / 状态影响：
        Runs the scheduler over scope4 (entity atoms, IEP, projections, skill, transition, compiled state).
        在 scope4 上运行调度器（实体原子、IEP、投影、技能、转移、编译状态）。
    """
    hms, store, ids, scope4 = run.hms, run.store, run.ids, run.scope4
    uc, dc, ep1_atom, ep2_atom = ids["uc"], ids["dc"], ids["ep1_atom"], ids["ep2_atom"]
    evidence = sorted({ids["anc_e1"], ids["anc_e2"], ids["anc_e3"], ids["anc_e4"]})

    rep = run.call(hms.consolidate, filters=scope4)
    assert rep["phases"]["consolidate"] == CONSOLIDATE_REPORT
    assert store.get_edge(ids["edg_temporal"]).mentions == 2 and store.get_edge(ids["edg_causal"]).mentions == 2
    entity_atoms = run.call(hms.get_all, filters={"user_id": U, "agent_id": "coder", "scale": "entity"})["results"]
    assert len(entity_atoms) == 6 and all(len(r["hms"]["entity_ids"]) == 1 for r in entity_atoms)
    assert sorted(r["hms"]["entity_ids"][0] for r in entity_atoms) == sorted(
        ["retry.ts", "file_diff", "billing_e2e", "test_log", "subscription_state_regression", "idempotency.ts"]
    )
    assert all("run_id" not in r and r["agent_id"] == "coder" and r["user_id"] == U for r in entity_atoms)
    assert all(r["atom_type"] == "entity" and r["hms"]["source_kind"] == "entity" for r in entity_atoms)
    assert all(store.get_atom(r["atom_id"]).content["text"] == r["memory"] for r in entity_atoms)

    ieps = store.list_interface_eps(run.scope4_obj)
    assert len(ieps) == 1
    iep = ieps[0]
    assert iep.support_node_ids == [ep1_atom, ep2_atom] and iep.scale == "L2_meso" and iep.valid
    assert iep.cohesion_score == pytest.approx(2 / 3, abs=1e-4)
    assert (iep.slots.u, iep.slots.a, iep.slots.b, iep.slots.y, iep.slots.r) == (
        S4_QUERY, f"{EP1_A} → {EP2_A}", f"{UC_TEXT} ; {DC_TEXT}", EP2_Y, "positive",
    )  # fmt: skip
    assert iep.slots.kappa.constraints == [UC_TEXT, DC_TEXT]
    assert iep.slots.kappa.evidence == iep.evidence_anchor_ids == evidence
    assert [(i.text, i.kind, i.support_node_ids, i.anchor_ids, i.confidence) for i in iep.invariants] == [
        (UC_TEXT, "constraint", [ep1_atom, ep2_atom], evidence, 0.8),
        (f"do not: {EP1_A}", "failed_path", [ep1_atom], sorted({ids["anc_e1"], ids["anc_e2"]}), 0.8),
    ]
    # §14.2: invariant confidence == round(mean(support atom confidence), 4); both EP atoms carry 0.8.
    # §14.2：不变量 confidence == round(mean(支撑原子 confidence), 4)；两个 EP 原子都是 0.8。
    assert [store.get_atom(n).confidence for n in (ep1_atom, ep2_atom)] == [0.8, 0.8]
    assert iep.residuals == []
    iep_atom = store.get_atom(iep.atom_id)
    assert iep_atom is not None and iep_atom.source_ep_ids == [ids["ep1"], ids["ep2"]]
    assert iep_atom.scale_level == "L2_meso" and iep_atom.atom_type == "episode"
    high_use_uc = store.find_edge(iep.atom_id, uc, "use", "L2_meso")
    high_use_dc = store.find_edge(iep.atom_id, dc, "use", "L2_meso")
    high_distill_dc = store.find_edge(iep.atom_id, dc, "distill", "L2_meso")
    assert all(e is not None and e.is_boundary and e.valid for e in (high_use_uc, high_use_dc, high_distill_dc))
    assert sorted(store.projections_of(high_use_uc.edge_id)) == sorted([ids["edg_use_ep1_uc"], ids["edg_use_ep2_uc"]])
    assert store.projections_of(high_use_dc.edge_id) == [ids["edg_use_ep2_dc"]]
    assert store.projections_of(high_distill_dc.edge_id) == [ids["edg_distill_ep1_dc"]]

    skill_rows = run.call(hms.get_all, filters={**scope4, "scale": "L3_macro"})["results"]
    skill_rows = [r for r in skill_rows if r["hms"]["source_kind"] == "skill"]
    assert [r["memory"] for r in skill_rows] == [SKILL_TEXT]
    skill = store.get_atom(skill_rows[0]["atom_id"])
    assert skill.scale_level == "L3_macro" and skill.anchor_ids == evidence and skill.confidence == 0.8
    skill_edge = store.find_edge(iep.atom_id, skill.atom_id, "distill", "L3_macro")
    assert skill_edge is not None and skill_edge.is_boundary is False and skill_edge.valid
    assert skill_edge.witness_edge_id == ids["edg_use_ep1_uc"]
    assert store.projections_of(skill_edge.edge_id) == [ids["edg_use_ep1_uc"], ids["edg_use_ep2_uc"]]

    iep_rows = [t for t in store.list_transitions(run.scope4_obj) if t.source_kind == "interface_ep"]
    assert len(iep_rows) == 1
    row = iep_rows[0]
    assert row.source_id == iep.interface_ep_id and row.scale == "L2_meso"
    assert row.a_t == iep.slots.a and row.r_t == "positive" and row.w_t == evidence
    assert row.s_t1["invariants"] == [UC_TEXT, f"do not: {EP1_A}"] and row.s_t1["residuals"] == []
    assert row.time_t == "2026-06-15T10:04:00.000000+00:00" and row.time_t1 == "2026-06-15T10:14:00.000000+00:00"
    assert set(row.kappa_t.keys()) == set(KAPPA_KEYS)
    compiled = store.last_compiled_state(run.scope4_obj)
    assert compiled is not None and compiled["card"].startswith("# Work State Card")
    # The scheduler compiled at the consolidate instant; a deep compile at that same `as_of` reproduces the card.
    # 调度器在 consolidate 那一刻完成编译；对同一个 `as_of` 做 deep 编译可复现该卡片。
    assert compiled["card"] == run.call(hms.compile, S4_QUERY, filters=scope4, mode="deep", as_of=compiled["as_of"]).to_text()  # fmt: skip
    assert compiled["hash"] == hashlib.sha256(compiled["card"].encode("utf-8")).hexdigest()
    assert rep["phases"]["orient"]["last_compiled"] is None
    assert compiled["hash"] in (store.get_meta(f"work_state:{run.scope4_obj.key()}") or "")

    st = run.call(hms.graph_stats, filters=scope4)
    assert st == GRAPH_STATS
    assert run.call(hms.graph_stats, filters={"user_id": U, "agent_id": "coder"})["nodes"]["entity"] == 6

    deep = run.call(hms.compile, S4_QUERY, filters=scope4, mode="deep")
    assert [s.text for s in deep.skills] == [SKILL_TEXT] and deep.retrieval["ablation"] == "axis_graph_support"
    assert any(i.explain.get("reason") == "invariant:failed_path" for i in deep.audit)
    axis_graph = run.call(hms.compile, S4_QUERY, filters=scope4, mode="axis_graph")
    assert [s.text for s in axis_graph.skills] == [SKILL_TEXT] and axis_graph.retrieval["ablation"] == "axis_graph"
    assert not any("#inv:" in i.item_id for i in axis_graph.items())
    assert not any(str(i.explain.get("reason", "")).startswith(("invariant:", "residual:")) for i in axis_graph.audit)
    fast = run.call(hms.compile, S4_QUERY, filters=scope4, mode="fast")
    assert [s.text for s in fast.skills] == [SKILL_TEXT] and fast.retrieval["ablation"] == "axis"
    assert not any("#inv:" in i.item_id for i in fast.items())
    assert not any(str(i.explain.get("reason", "")).startswith("invariant:") for i in fast.audit)


# ---------------------------------------------------------------------------
# 63
# ---------------------------------------------------------------------------
def test_scenarios_end_to_end(hms, clock, store, events, scope4, golden, index_provider):
    """S1–S6 (§19) in sequence on ONE hms, every expect block asserted exactly, then consolidate() + graph_stats() (§19.7).
    在同一个 hms 上按顺序运行 S1–S6（§19），精确断言每个 expect 块，然后 consolidate() + graph_stats()（§19.7）。

    Input / 输入：
        hms: The HMS fixture (rule LLM + hash embedder + parametrized index).
        hms：HMS 夹具（规则 LLM + 哈希嵌入器 + 参数化索引）。

        clock: FrozenClock fixture (ticked 60 s before every public call).
        clock：FrozenClock 夹具（每次公开调用前推进 60 秒）。

        store: The StateStore shared with `hms`.
        store：与 `hms` 共享的 StateStore。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        golden: Golden helper (text and masked comparisons).
        golden：黄金比对辅助（文本与掩码比对）。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        None (asserts).
        无（断言）。

    State effect / 状态影响：
        None outside the in-memory store and index.
        除内存存储与索引外无副作用。
    """
    assert hms.config.vector_store.provider == index_provider and hms.store is store
    assert clock.now() == "2026-09-02T10:00:00.000000+00:00"
    run = _Runner(hms, clock, store, golden, events, scope4)
    _s1_preference_recall(run)
    _s2_s3_temporal_and_entity(run)
    _s4_work_state_card(run)
    _s4b_search_by_identifier(run)
    _s5_why(run)
    _s6_checkpoint_feedback_rollback(run)
    _epilogue_consolidate_and_stats(run)
