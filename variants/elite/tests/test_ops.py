"""Group G tests: feedback matrix, checkpoint/rollback, handoff/import, transitions, scheduler phases, revoke, entity atoms, retention.
G 组测试：反馈矩阵、检查点/回滚、交接/导入、状态转移、调度器阶段、撤销、实体原子、保留策略。
"""

# Purpose (EN): Assert CONTRACT §16 (and §18.3 tests 52–56f) over the ops plane on S1–S6 ingests.
# 目的（中）：在 S1–S6 摄取结果上断言契约 §16（以及 §18.3 测试 52–56f）。

# Input (EN): tests/fixtures/events/*.json, tests/fixtures/golden/transitions_s6.jsonl and the conftest fixtures.
# 输入（中）：tests/fixtures/events/*.json、tests/fixtures/golden/transitions_s6.jsonl 与 conftest 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores (--update-golden rewrites the transitions golden).
# 状态影响（中）：除内存 SQLite 存储外无副作用（--update-golden 会重写转移黄金文件）。

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

from hms_elite.compiler import StateCompiler
from hms_elite.configs.base import HMSConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.exceptions import StoreError, ValidationError
from hms_elite.graph import SupportGraph
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.ingest.atoms import create_atom
from hms_elite.ingest.events import anchor_events, events_from_dicts
from hms_elite.ingest.pipeline import Ingestor, IngestResult
from hms_elite.llms.rule import RuleLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import AxisQuery, MemoryAtom, Scope, StatePackage
from hms_elite.mvmi.filters import parse_filters
from hms_elite.mvmi.query import execute_axis_query
from hms_elite.ops import (
    FeedbackProcessor,
    StateScheduler,
    build_state_package,
    checkpoint,
    export_transitions,
    import_state_package,
    purge,
    revoke,
    rollback,
)
from hms_elite.ops.feedback import scope_from_key
from hms_elite.ops.retention import PROTECTED_REASONS
from hms_elite.utils.text import sha256_hex
from tests.conftest import FrozenClock, SequentialIds

U = "alice"
S4_QUERY = "fix checkout webhook retry"
S4_AS_OF = "2026-06-15T12:00:00Z"
S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP1_Y = "billing_e2e failed: subscription_state_regression"
S4_EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
S4_DC = f"Do not re-apply '{S4_EP1_A}': {S4_EP1_Y}"
S6_CORRECTION = "retry.ts may change once billing_e2e is isolated."
KAPPA_KEYS = ["constraints", "visibility", "permission_scope", "provenance"]
CONSOLIDATE_S4 = {
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


class _Harness:
    """Store + index + embedder + rule LLM wired into the ingest, graph, compiler and ops planes (the facade minus H).
    将 store + 索引 + 嵌入器 + 规则 LLM 接入摄取、图、编译器与运维平面（去掉 H 组门面）。
    """

    def __init__(self, store: StateStore, clock: FrozenClock, index_provider: str, **config: Any) -> None:
        """Wire the collaborators.
        接线协作对象。
        """
        self.store = store
        self.clock = clock
        self.embedder = HashEmbedder({"embedding_dims": 64})
        self.config = HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}, **config)
        if index_provider == "sqlite":
            self.index: Any = SQLiteIndex(store=store, dims=64)
        else:
            self.index = MemoryIndex(dims=64)
        self.llm = RuleLLM()
        self.ingestor = Ingestor(store, self.index, self.embedder, self.llm, self.config)
        self.graph = SupportGraph(store, self.config.graph)
        self.compiler = StateCompiler(store, self.index, self.embedder, self.graph, self.config)
        self.processor = FeedbackProcessor(store, self.index, self.embedder, self.graph, self.config)
        self.scheduler = StateScheduler(
            store, self.index, self.embedder, self.llm, self.graph, self.ingestor, self.compiler, self.config
        )

    def add(self, messages: Any, *, timestamp: str | None = None, **scope: str) -> IngestResult:
        """Mem0 add() minus the facade.
        去掉门面的 Mem0 add()。
        """
        self.clock.tick(seconds=60)
        return self.ingestor.ingest_messages(
            coerce_messages(messages),
            scope=Scope(**scope),
            user_meta={},
            infer=True,
            memory_type=None,
            prompt=None,
            event_time=timestamp,
            valid_to=None,
        )

    def add_events(self, items: list, **scope: str) -> IngestResult:
        """HMS add_events() minus the facade.
        去掉门面的 HMS add_events()。
        """
        self.clock.tick(seconds=60)
        events = events_from_dicts(
            items, scope_fallback=Scope(**scope), visibility_default="private", event_time=self.clock.now()
        )
        return self.ingestor.ingest_events(events, infer=True)

    def compile(self, query: str, filters: dict, **kw: Any):
        """HMS compile() minus the facade (ticks the clock like the scenario runner).
        去掉门面的 HMS compile()（像场景运行器一样推进时钟）。
        """
        self.clock.tick(seconds=60)
        scope, _expr = parse_filters(filters, require_scope=True, context="compile")
        return self.compiler.compile(query, scope=scope, filters=filters, **kw)

    def feedback(self, target_id: str | None, feedback: str, **kw: Any) -> dict:
        """HMS feedback() minus the facade.
        去掉门面的 HMS feedback()。
        """
        self.clock.tick(seconds=60)
        return self.processor.process(
            target_id=target_id,
            feedback=feedback,
            reason=kw.get("reason"),
            correction=kw.get("correction"),
            actor_id=kw.get("actor_id"),
            scope_hint=kw.get("scope_hint"),
        )

    def checkpoint(self, filters: dict, label: str | None = None) -> str:
        """HMS checkpoint() minus the facade.
        去掉门面的 HMS checkpoint()。
        """
        self.clock.tick(seconds=60)
        return checkpoint(self.store, scope=Scope(**filters), label=label)

    def rollback(self, cp: str, reason: str | None = None) -> dict:
        """HMS rollback() minus the facade.
        去掉门面的 HMS rollback()。
        """
        self.clock.tick(seconds=60)
        return rollback(self.store, self.graph, checkpoint_id=cp, reason=reason)

    def transitions(self, filters: dict, **kw: Any) -> list:
        """HMS transitions() minus the facade (tuples by default).
        去掉门面的 HMS transitions()（默认元组）。
        """
        self.clock.tick(seconds=60)
        return export_transitions(
            self.store, scope=Scope(**filters), since=kw.get("since"), scale=kw.get("scale"), kinds=kw.get("kinds")
        )

    def consolidate(self, filters: dict | None = None, phases: list[str] | None = None) -> dict:
        """HMS consolidate() minus the facade.
        去掉门面的 HMS consolidate()。
        """
        self.clock.tick(seconds=60)
        return self.scheduler.run(scope=Scope(**filters) if filters else None, phases=phases)

    def handoff(self, filters: dict, *, goal: str | None = None, budget: int = 800, take_checkpoint: bool = True):
        """HMS handoff() minus the facade: optional checkpoint first, then build_state_package.
        去掉门面的 HMS handoff()：可选地先建检查点，再 build_state_package。
        """
        cp = self.checkpoint(filters, label="handoff") if take_checkpoint else None
        self.clock.tick(seconds=60)
        scope = Scope(**filters)
        return build_state_package(
            self.store,
            self.graph,
            self.compiler,
            config=self.config,
            scope=scope,
            filters=filters,
            goal=goal,
            budget_tokens=budget,
            checkpoint_id=cp,
        )

    def import_package(self, pkg_json: str, **scope: str) -> dict:
        """HMS import_package() minus the facade.
        去掉门面的 HMS import_package()。
        """
        self.clock.tick(seconds=60)
        pkg = StatePackage.from_json(pkg_json)
        override = None
        if scope:
            merged = {**pkg.scope.as_filters(), **scope}
            override = Scope(**merged)
        return import_state_package(self.store, self.index, self.embedder, self.graph, pkg=pkg, scope_override=override)

    def revoke(self, anchor_id: str, reason: str | None = None) -> dict:
        """HMS revoke() minus the facade.
        去掉门面的 HMS revoke()。
        """
        self.clock.tick(seconds=60)
        return revoke(self.store, self.graph, anchor_id=anchor_id, reason=reason)


def _anchor_of(res: IngestResult, object_type: str) -> str:
    """The anchor id of the first event of a type in an IngestResult.
    IngestResult 中某类型首个事件的锚点 id。
    """
    return [e["anchor_id"] for e in res.events if e["object_type"] == object_type][0]


def _build_s4(h: _Harness, events: Any, scope4: dict) -> dict[str, Any]:
    """Ingest the §19.3 S4 fixture (constraint + events_fail + events_fix) and return the ids a test needs.
    摄取 §19.3 的 S4 夹具（约束 + events_fail + events_fix）并返回测试所需的 id。
    """
    fixture = events("structured_billing")
    r0 = h.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    ra = h.add_events(fixture.events_fail, **scope4)
    rb = h.add_events(fixture.events_fix, **scope4)
    uc = h.store.get_atom(r0.rows[0]["atom_id"])
    fail_events = [row["atom_id"] for row in ra.rows if row["atom_type"] == "event"]
    fix_events = [row["atom_id"] for row in rb.rows if row["atom_type"] == "event"]
    return {
        "uc": uc.atom_id,
        "mem_uc": uc.memory_id,
        "dc": ra.constraint_ids[0],
        "ep1": h.store.get_ep(ra.ep_ids[0]),
        "ep2": h.store.get_ep(rb.ep_ids[0]),
        "anc_uc": uc.anchor_ids[0],
        "anc_e1": _anchor_of(ra, "file_diff"),
        "anc_e2": _anchor_of(ra, "test_log"),
        "anc_e3": _anchor_of(rb, "file_diff"),
        "anc_e4": _anchor_of(rb, "test_log"),
        "e1": fail_events[0],
        "e2": fail_events[1],
        "e3": fix_events[0],
        "e4": fix_events[1],
        "edges": {(e.edge_type, e.src_id, e.dst_id): e for e in h.store.list_edges(Scope(**scope4))},
    }


def _run_s6(h: _Harness, ids: dict, scope4: dict) -> dict[str, Any]:
    """The §19.6 S6 flow: checkpoint → state_outdated feedback → mid compile → rollback → transitions.
    §19.6 的 S6 流程：检查点 → state_outdated 反馈 → 中间编译 → 回滚 → 状态转移。
    """
    cp = h.checkpoint(scope4, label="before-feedback")
    fb = h.feedback(ids["mem_uc"], "state_outdated", correction=S6_CORRECTION)
    mid = h.compile(S4_QUERY, scope4)
    rb = h.rollback(cp)
    tr = h.transitions(scope4)
    return {"cp": cp, "fb": fb, "mid": mid, "rb": rb, "tr": tr}


def _seed_state(h: _Harness, *, scope: Scope, ts: str, text: str, **overrides: Any) -> MemoryAtom:
    """Insert a state atom (+node) witnessed by a fresh message anchor.
    插入一个由新消息锚点见证的 state 原子（+节点）。
    """
    ((ev, anc, _dedup),) = anchor_events(
        h.store,
        events_from_dicts(
            [{"object_type": "message", "role": "user", "timestamp": ts, "payload": {"text": text}}],
            scope_fallback=scope,
            visibility_default="private",
        ),
    )
    base: dict[str, Any] = dict(
        atom_type="state",
        content={"text": text, "objects": overrides.pop("objects", [])},
        text=text,
        time_anchor=ts,
        scale_level="state",
        subject="user",
        predicate="is",
        obs_type="fact",
        source_kind="observation",
        source_id=f"obs_{ev.event_id}",
        extracted_by="test",
        scope=scope,
        anchor_ids=[anc.anchor_id],
    )
    base.update(overrides)
    return create_atom(h.store, h.index, h.embedder, MemoryAtom(**base))


def _names(actions: list[dict]) -> list[str]:
    """The action names of a feedback result, in order.
    反馈结果中的动作名，按顺序。
    """
    return [a["action"] for a in actions]


# ---------------------------------------------------------------------------
# 52
# ---------------------------------------------------------------------------
_FEEDBACK_CASES = [
    "state_wrong",
    "state_outdated",
    "state_missing",
    "privacy_violation",
    "next_action_wrong",
    "relationship_not_useful",
    "connection_reason_wrong",
    "evidence_wrong",
    "render_not_like_me",
    "render_like_me",
    "object_important",
    "state_missing_no_target",
    "connection_reason_wrong_same_type",
    "evidence_wrong_revoke",
]


@pytest.mark.parametrize("feedback_type", _FEEDBACK_CASES)
def test_feedback_matrix(store, clock, events, scope4, index_provider, feedback_type):
    """Every FeedbackType applies its §16.1 actions in order, writes a FEEDBACK history row and only privacy changes visibility.
    每种 FeedbackType 按序执行其 §16.1 动作、写入 FEEDBACK 历史行，且只有 privacy_violation 改变可见性。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        feedback_type: The FeedbackType (or sub-case name) under test.
        feedback_type：被测的 FeedbackType（或子用例名）。

    Output / 输出：
        Assertions on actions, history, visibility, witnesses and validation codes.
        对动作、历史、可见性、见证与校验码的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    ep1_atom = ids["ep1"].atom_id
    causal = ids["edges"][("causal", ep1_atom, ids["ep2"].atom_id)]
    visibility_before = {a.atom_id: a.visibility for a in store.get_atoms([ids["uc"], ids["dc"], ids["e1"], ep1_atom])}

    def fb(target: str | None, kind: str, **kw: Any) -> dict:
        """Apply feedback and assert the common prologue effects.
        应用反馈并断言共同的前奏效果。
        """
        out = h.feedback(target, kind, **kw)
        assert out["message"] == "Feedback applied" and out["feedback_id"].startswith("fb_")
        if target and "&" in target:
            lookup = scope_from_key(target).key()
        else:
            lookup = target or Scope(**scope4).key()
        rows = store.list_feedback(lookup)
        assert any(r.feedback_id == out["feedback_id"] and r.actions == out["actions"] for r in rows)
        return out

    def history_events(memory_id: str) -> list[str]:
        """History event names of a lineage.
        某谱系的历史事件名。
        """
        return [row["event"] for row in store.get_history(memory_id)]

    if feedback_type == "state_wrong":
        out = fb(ids["mem_uc"], "state_wrong", correction="Never touch the subscription state machine.")
        assert _names(out["actions"]) == ["archive", "create", "revision_edge"]
        head = store.resolve_head(ids["mem_uc"])
        assert head.version == 2 and head.text == "Never touch the subscription state machine."
        assert (
            head.obs_type == "constraint" and head.anchor_ids[0] != ids["anc_uc"] and ids["anc_uc"] in head.anchor_ids
        )
        edge = store.get_edge(out["actions"][2]["edge_id"])
        assert edge.edge_type == "revision" and edge.confidence == 0.95 and edge.created_by == "feedback"
        assert edge.witness_anchor_ids == head.anchor_ids
        assert history_events(ids["mem_uc"]) == ["ADD", "FEEDBACK", "UPDATE"]
        assert store.get_atom(ids["uc"]).archived_reason == "feedback"
        # a second state_wrong without a correction archives only and invalidates outgoing use/distill edges
        # 第二次不带 correction 的 state_wrong 只做归档，并使出向的 use/distill 边失效
        out2 = fb(ep1_atom, "state_wrong")
        assert _names(out2["actions"])[0] == "archive" and "invalidate_edge" in _names(out2["actions"])
        assert store.get_atom(ep1_atom).archived_reason == "feedback"
        assert not [e for e in store.edges_touching([ep1_atom], valid_only=True, direction="out") if e.edge_type in ("use", "distill")]  # fmt: skip
    elif feedback_type == "state_outdated":
        out = fb(ids["mem_uc"], "state_outdated", correction=S6_CORRECTION)
        assert _names(out["actions"]) == ["archive", "create", "revision_edge"]
        head = store.resolve_head(ids["mem_uc"])
        assert head.text == S6_CORRECTION and head.version == 2 and head.obs_type == "constraint"
        assert {"retry.ts", "billing_e2e"} <= set(head.entity_ids)
        assert head.valid_from == clock.now() and head.time_anchor == clock.now()
        assert store.get_atom(ids["uc"]).valid_to == clock.now()
        assert history_events(ids["mem_uc"]) == ["ADD", "FEEDBACK", "UPDATE"]
        # dict correction with an explicit valid_to and no text: archive only
        # correction 为带显式 valid_to 且无 text 的 dict：只做归档
        out2 = fb(ids["dc"], "state_outdated", correction={"valid_to": "2026-06-16T00:00:00Z"})
        assert _names(out2["actions"]) == ["archive"]
        assert store.get_atom(ids["dc"]).valid_to == "2026-06-16T00:00:00.000000+00:00"
    elif feedback_type == "state_missing":
        with pytest.raises(ValidationError) as exc:
            h.feedback(ids["ep1"].ep_id, "state_missing")
        assert exc.value.error_code == "VALIDATION_007"
        out = fb(ids["ep1"].ep_id, "state_missing", correction="The webhook retries are capped at five.")
        assert _names(out["actions"]) == ["create", "insert_edge"]
        atom = store.get_atom(out["actions"][0]["atom_id"])
        assert atom.atom_type == "state" and atom.obs_type == "fact" and atom.source_kind == "feedback"
        assert atom.subject == "user" and atom.predicate == "states" and atom.source_id == out["feedback_id"]
        assert atom.anchor_ids[0] not in (ids["anc_e1"], ids["anc_e2"]) and set(ids["ep1"].anchor_ids) <= set(atom.anchor_ids)  # fmt: skip
        edge = store.get_edge(out["actions"][1]["edge_id"])
        assert (edge.edge_type, edge.src_id, edge.dst_id) == ("use", atom.atom_id, ep1_atom)
        assert history_events(atom.memory_id) == ["ADD"]
        assert history_events(store.get_atom(ep1_atom).memory_id) == ["ADD", "FEEDBACK"]
    elif feedback_type == "privacy_violation":
        store.set_atom_visibility(ids["e1"], "team", at=clock.now())
        store.set_atom_visibility(ep1_atom, "team", at=clock.now())
        out = fb(ids["e1"], "privacy_violation")
        assert set(_names(out["actions"])) == {"set_visibility"}
        touched = {a["atom_id"] for a in out["actions"] if "atom_id" in a}
        # 共享 <anc_e1> 的每一个原子
        assert touched == {ids["e1"], ep1_atom, ids["dc"]}  # every atom sharing <anc_e1>
        assert all(store.get_atom(a).visibility == "private" for a in touched)
        assert store.get_anchor(ids["anc_e1"]).permission_scope == "private"
        assert store.list_edges(Scope(**scope4)) and all(e.valid for e in store.list_edges(Scope(**scope4)))
    elif feedback_type == "next_action_wrong":
        out = fb(ids["ep1"].ep_id, "next_action_wrong")
        assert _names(out["actions"]) == ["create", "set_confidence", "insert_edge"]
        constraint = store.get_atom(out["actions"][0]["atom_id"])
        assert constraint.text == f"Do not: {S4_EP1_A}" and constraint.obs_type == "constraint"
        assert constraint.subject == "project" and constraint.predicate == "must_not"
        assert constraint.content["objects"] == [S4_EP1_A] and "next_action" in constraint.topic_tags
        assert constraint.anchor_ids[1:] == ids["ep1"].anchor_ids and constraint.source_kind == "feedback"
        assert out["actions"][1]["confidence"] == [0.8, 0.6] and store.get_atom(ep1_atom).confidence == 0.6
        edge = store.get_edge(out["actions"][2]["edge_id"])
        assert (edge.edge_type, edge.src_id, edge.dst_id) == ("distill", ep1_atom, constraint.atom_id)
    elif feedback_type == "relationship_not_useful":
        with pytest.raises(ValidationError) as exc:
            h.feedback(ids["uc"], "relationship_not_useful")
        assert exc.value.error_code == "VALIDATION_007"
        out = fb(causal.edge_id, "relationship_not_useful")
        assert _names(out["actions"]) == ["invalidate_edge"]
        edge = store.get_edge(causal.edge_id)
        assert edge.valid is False and edge.invalidated_reason == "feedback:not_useful"
    elif feedback_type == "connection_reason_wrong":
        out = fb(
            causal.edge_id, "connection_reason_wrong", correction={"edge_type": "use", "label": "ep2 relies on ep1"}
        )
        assert _names(out["actions"]) == ["invalidate_edge", "insert_edge"]
        assert store.get_edge(causal.edge_id).valid is False
        new = store.get_edge(out["actions"][1]["edge_id"])
        assert new.edge_id != causal.edge_id and new.edge_type == "use" and new.label == "ep2 relies on ep1"
        assert new.created_by == "feedback" and (new.src_id, new.dst_id) == (causal.src_id, causal.dst_id)
        assert new.witness_anchor_ids[: len(causal.witness_anchor_ids)] == causal.witness_anchor_ids
        assert len(new.witness_anchor_ids) == len(causal.witness_anchor_ids) + 1
    elif feedback_type == "evidence_wrong":
        out = fb(ids["dc"], "evidence_wrong")
        assert _names(out["actions"]) == ["dispute_anchor", "set_confidence"]
        assert out["actions"][0]["anchor_id"] == ids["anc_e1"]
        assert store.atom_anchor_ids(ids["dc"]) == [ids["anc_e2"]] and store.get_atom(ids["dc"]).archived_reason is None
        assert out["actions"][1]["confidence"] == [0.85, 0.55]
        out2 = fb(ids["dc"], "evidence_wrong", correction=[ids["anc_e2"]])
        assert _names(out2["actions"]) == ["dispute_anchor", "archive", "set_confidence"]
        assert store.get_atom(ids["dc"]).archived_reason == "feedback:evidence_wrong"
        # 没有未争议的见证就绝不有效
        assert store.atom_anchor_ids(ids["dc"]) == []  # never valid without an undisputed witness
    elif feedback_type in ("render_not_like_me", "render_like_me"):
        out = fb(ids["e1"], feedback_type, reason=None)
        assert _names(out["actions"]) == ["set_confidence", "create"]
        pref = store.get_atom(out["actions"][1]["atom_id"])
        verb = "likes" if feedback_type == "render_like_me" else "dislikes"
        assert pref.obs_type == "preference_event" and pref.predicate == verb
        assert pref.content["objects"] == [f"render:{ids['e1']}"] and "render_style" in pref.topic_tags
        assert pref.text.startswith(f"User {verb} how '")
        delta = 0.1 if feedback_type == "render_like_me" else -0.1
        assert out["actions"][0]["confidence"] == [0.7, round(0.7 + delta, 4)]
    elif feedback_type == "object_important":
        out = fb(ids["e1"], "object_important")
        assert _names(out["actions"]) == ["set_confidence", "pin"]
        atom = store.get_atom(ids["e1"])
        assert atom.confidence == 0.9 and "important" in atom.topic_tags
        assert store.get_node(ids["e1"]).mentions == 4
        out2 = fb(ids["e1"], "object_important")
        # 已被上限截断
        assert out2["actions"][0]["confidence"] == [0.9, 1.0]  # clamped
    elif feedback_type == "state_missing_no_target":
        out = fb(None, "state_missing", correction={"scope": scope4, "text": "Billing uses idempotency keys."})
        atom = store.get_atom(out["actions"][0]["atom_id"])
        assert atom.atom_type == "state" and atom.scope == Scope(**scope4)
        assert store.list_feedback(Scope(**scope4).key())[0].target_kind == "scope"
        key = "user_id=alice&run_id=run-a"
        out2 = fb(key, "state_missing", correction="Alice prefers async reviews.")
        atom2 = store.get_atom(out2["actions"][0]["atom_id"])
        assert atom2.scope == Scope(user_id="alice", run_id="run-a") and atom2.text == "Alice prefers async reviews."
        assert store.list_feedback(Scope(user_id="alice", run_id="run-a").key())[0].target_kind == "scope"
        assert scope_from_key("user_id=a%26b&run_id=r%3D1") == Scope(user_id="a&b", run_id="r=1")
        with pytest.raises(ValidationError) as exc:
            h.feedback(None, "state_missing", correction="no scope anywhere")
        assert exc.value.error_code == "VALIDATION_001"
        with pytest.raises(ValidationError) as exc:
            h.feedback("nonsense-id", "state_wrong")
        assert exc.value.error_code == "NOT_FOUND_001"
        with pytest.raises(ValidationError) as exc:
            h.feedback(ids["uc"], "not_a_feedback_type")
        assert exc.value.error_code == "VALIDATION_006" and "state_wrong" in exc.value.details["allowed"]
    elif feedback_type == "connection_reason_wrong_same_type":
        out = fb(causal.edge_id, "connection_reason_wrong", correction="retry failure triggered the idempotency fix")
        assert out["actions"] == [{"action": "insert_edge", "edge_id": causal.edge_id, "merged": True}]
        edge = store.get_edge(causal.edge_id)
        assert edge.valid and edge.label == "retry failure triggered the idempotency fix"
        fb_anchor = store.list_feedback(causal.edge_id)[0].anchor_id
        assert fb_anchor in store.edge_witnesses(causal.edge_id)
        assert store.edge_witnesses(causal.edge_id)[: len(causal.witness_anchor_ids)] == causal.witness_anchor_ids
    elif feedback_type == "evidence_wrong_revoke":
        with pytest.raises(ValidationError) as exc:
            h.feedback(ids["anc_e2"], "evidence_wrong")
        assert exc.value.error_code == "VALIDATION_007"
        out = fb(ids["anc_e2"], "evidence_wrong", correction={"revoke": True})
        assert _names(out["actions"]) == ["revoke"]
        assert out["actions"][0]["revoked"] == ids["anc_e2"] and out["actions"][0]["archived"] == [ids["e2"]]
        assert store.get_anchor(ids["anc_e2"]).revoked_at == clock.now()
        assert store.list_feedback(ids["anc_e2"])[0].target_kind == "anchor"

    # common: history FEEDBACK on atom targets; visibility untouched except for privacy_violation
    # 公共部分：对原子目标写入 FEEDBACK 历史；除 privacy_violation 外 visibility 保持不变
    if feedback_type in ("state_wrong", "state_outdated", "evidence_wrong", "object_important"):
        target_mem = ids["mem_uc"] if feedback_type.startswith("state") else store.get_atom(ids["dc"] if feedback_type == "evidence_wrong" else ids["e1"]).memory_id  # fmt: skip
        assert "FEEDBACK" in history_events(target_mem)
    if feedback_type != "privacy_violation":
        for atom_id, visibility in visibility_before.items():
            assert store.get_atom(atom_id).visibility == visibility


# ---------------------------------------------------------------------------
# 53
# ---------------------------------------------------------------------------
def test_checkpoint_rollback_symmetric(store, clock, events, scope4, index_provider):
    """S6: feedback then rollback restores the original head, edge validity, EP validity and history is append-only.
    S6：反馈后回滚恢复原头版本、边有效性、EP 有效性，历史只追加。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the §19.6 rollback report, lineage, history and ep_state restoration.
        对 §19.6 回滚报告、谱系、历史与 ep_state 恢复的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    scope = Scope(**scope4)
    # an atom archived before the checkpoint and restored after it is re-archived by the rollback
    # 在检查点之前归档、之后又被恢复的原子会被回滚重新归档
    store.archive_atom(ids["e1"], reason="deleted", at=clock.now())
    s6 = _run_s6_with_restore(h, ids, scope4)
    fb, mid, rb, cp = s6["fb"], s6["mid"], s6["rb"], s6["cp"]
    uc2 = fb["actions"][1]["atom_id"]
    edg_rev = fb["actions"][2]["edge_id"]
    assert fb["actions"] == [
        {"action": "archive", "atom_id": ids["uc"]},
        {"action": "create", "atom_id": uc2, "memory_id": ids["mem_uc"]},
        {"action": "revision_edge", "edge_id": edg_rev},
    ]
    assert [c.text for c in mid.constraints] == [S4_DC, S6_CORRECTION]
    revised = [a for a in mid.audit if a.explain.get("reason") == "revised"]
    assert [a.text for a in revised] == [S4_CONSTRAINT] and mid.as_of == s6["mid_now"]
    assert rb == {
        "message": "Rolled back to before-feedback",
        "checkpoint_id": cp,
        "restored": [ids["uc"]],
        "archived": sorted([ids["e1"], uc2]),
        "edges_invalidated": [edg_rev],
        "edges_revalidated": [],
        "edges_created": rb["edges_created"],
        "anchor_id": rb["anchor_id"],
    }
    assert len(rb["edges_created"]) == 1 and rb["anchor_id"].startswith("anc_")
    head = store.resolve_head(ids["mem_uc"])
    assert head.atom_id == ids["uc"] and head.text == S4_CONSTRAINT and head.version == 1
    edge = store.get_edge(rb["edges_created"][0])
    assert (edge.edge_type, edge.src_id, edge.dst_id) == ("revision", uc2, ids["uc"])
    assert edge.label == f"rollback:{cp}" and edge.witness_anchor_ids == [rb["anchor_id"]] and edge.created_by == "ops"
    assert [row["event"] for row in store.get_history(ids["mem_uc"])] == ["ADD", "FEEDBACK", "UPDATE", "ROLLBACK", "RESTORE"]  # fmt: skip
    e1 = store.get_atom(ids["e1"])
    assert e1.archived_reason == f"rollback:{cp}" and e1.valid_to == s6["e1_valid_to"]
    assert store.get_checkpoint(cp).history_seq < store.history_seq()

    # ep_state: deleting the episodes after a checkpoint changes handoff's goal; rollback restores eps.valid_to
    # ep_state：检查点之后删除 episodes 会改变 handoff 的 goal；回滚恢复 eps.valid_to
    cp2 = h.checkpoint(scope4, label="before-delete")
    assert h.handoff(scope4, take_checkpoint=False).goal == S4_QUERY
    for ep in (ids["ep1"], ids["ep2"]):
        store.archive_atom(ep.atom_id, reason="deleted", at=clock.now())
    assert store.list_eps(scope) == [] and h.handoff(scope4, take_checkpoint=False).goal is None
    rb2 = h.rollback(cp2)
    assert sorted(rb2["restored"]) == sorted([ids["ep1"].atom_id, ids["ep2"].atom_id])
    assert [ep.ep_id for ep in store.list_eps(scope)] == [ids["ep1"].ep_id, ids["ep2"].ep_id]
    assert all(v is None for v in store.scope_ep_state(scope).values())
    assert h.handoff(scope4, take_checkpoint=False).goal == S4_QUERY
    with pytest.raises(ValueError):
        h.rollback("ckp_nonsense")


def _run_s6_with_restore(h: _Harness, ids: dict, scope4: dict) -> dict[str, Any]:
    """S6 flow where <atm_e1> was archived before the checkpoint and is restored after it.
    <atm_e1> 在检查点前归档、检查点后恢复的 S6 流程。
    """
    cp = h.checkpoint(scope4, label="before-feedback")
    e1_valid_to = h.store.get_atom(ids["e1"]).valid_to
    h.store.restore_atom(ids["e1"], at=h.clock.now())
    fb = h.feedback(ids["mem_uc"], "state_outdated", correction=S6_CORRECTION)
    mid = h.compile(S4_QUERY, scope4)
    mid_now = h.clock.now()
    rb = h.rollback(cp)
    return {"cp": cp, "fb": fb, "mid": mid, "mid_now": mid_now, "rb": rb, "e1_valid_to": e1_valid_to}


# ---------------------------------------------------------------------------
# 54 / 54b
# ---------------------------------------------------------------------------
def test_handoff_state_package_roundtrip(store, clock, events, scope4, index_provider):
    """handoff() → JSON → from_json equal; witness/evidence closure; checkpoint_id set; rollback(pkg.checkpoint_id) works.
    handoff() → JSON → from_json 相等；见证/证据闭包；checkpoint_id 已设置；rollback(pkg.checkpoint_id) 可用。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the StatePackage closures and the checkpoint it carries.
        对 StatePackage 闭包及其携带的检查点的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    pkg = h.handoff(scope4, goal=S4_QUERY)
    assert pkg.goal == S4_QUERY and pkg.scope == Scope(**scope4) and pkg.checkpoint_id.startswith("ckp_")
    assert StatePackage.from_json(pkg.to_json()) == pkg
    assert pkg.pcir.ok() and pkg.pcir.human_gloss == pkg.context.to_text()
    atom_ids = {a.atom_id for a in pkg.atoms}
    assert {ids["uc"], ids["dc"], ids["ep1"].atom_id, ids["ep2"].atom_id} <= atom_ids
    assert {i.atom_id for i in pkg.context.items() if i.atom_id} <= atom_ids
    assert [ep.ep_id for ep in pkg.eps] == sorted([ids["ep1"].ep_id, ids["ep2"].ep_id])
    anchor_ids = {a.anchor_id for a in pkg.anchors}
    assert anchor_ids == {ids["anc_uc"], ids["anc_e1"], ids["anc_e2"], ids["anc_e3"], ids["anc_e4"]}
    assert {e.event_id for e in pkg.source_events} == {a.source_event_id for a in pkg.anchors}
    assert len(pkg.source_events) == len(pkg.anchors)
    assert {(e.edge_type, e.src_id, e.dst_id) for e in pkg.edges} == set(ids["edges"])
    assert pkg.interface_eps == []
    h.feedback(ids["mem_uc"], "state_wrong", correction="x")
    rb = h.rollback(pkg.checkpoint_id)
    assert rb["restored"] == [ids["uc"]] and store.resolve_head(ids["mem_uc"]).text == S4_CONSTRAINT
    # goal=None takes the newest valid EP's u
    # goal=None 时取最新有效 EP 的 u
    assert h.handoff(scope4, take_checkpoint=False).goal == S4_QUERY


def test_handoff_import_roundtrip(store, clock, events, scope4, index_provider):
    """A fresh receiver imports the package and renders the same card; re-import skips; run_id override re-scopes.
    新接收方导入交接包并渲染同一张卡片；重复导入被跳过；run_id 覆盖会重新划定范围。

    Input / 输入：
        store: In-memory StateStore fixture (the sender).
        store：内存 StateStore 夹具（发送方）。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the import report, the receiver's card and the re-scoped import.
        对导入报告、接收方卡片与重新划定范围导入的断言。

    State effect / 状态影响：
        None outside the in-memory stores.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    card2 = h.compile(S4_QUERY, scope4, budget_tokens=800, as_of=S4_AS_OF).to_text()
    pkg = h.handoff(scope4, take_checkpoint=False)
    b = _Harness(StateStore(":memory:", now=clock.now, id_factory=SequentialIds()), clock, index_provider)
    rep = b.import_package(pkg.to_json())
    assert rep["imported"]["atoms"] == len(pkg.atoms) and rep["imported"]["anchors"] == len(pkg.anchors)
    assert rep["imported"]["eps"] == len(pkg.eps) and rep["imported"]["edges"] == len(pkg.edges)
    assert rep["imported"]["source_events"] == len(pkg.source_events) and rep["imported"]["transitions"] == 2
    assert rep["anchor_id"].startswith("anc_") and all(v == 0 for v in rep["skipped"].values())
    assert b.compile(S4_QUERY, scope4, budget_tokens=800, as_of=S4_AS_OF).to_text() == card2
    assert b.store.resolve_head(ids["mem_uc"]).text == S4_CONSTRAINT
    rep2 = b.import_package(pkg.to_json())
    assert all(v == 0 for v in rep2["imported"].values())
    assert rep2["skipped"]["atoms"] == len(pkg.atoms) and rep2["skipped"]["anchors"] == len(pkg.anchors)
    rep3 = b.import_package(pkg.to_json(), run_id="run-import")
    assert rep3["imported"]["atoms"] == len(pkg.atoms)
    rescoped = b.store.axis_select(Scope(user_id=U, run_id="run-import"))
    assert rescoped and all(a.scope.run_id == "run-import" for a in rescoped)
    # id 保持稳定
    assert b.store.resolve_head(ids["mem_uc"]).text == S4_CONSTRAINT  # ids stay stable
    assert b.compile(S4_QUERY, {"user_id": U, "run_id": "run-import"}, budget_tokens=800, as_of=S4_AS_OF).to_text().count(S4_DC) == 1  # fmt: skip


# ---------------------------------------------------------------------------
# 55
# ---------------------------------------------------------------------------
def test_transitions_export_golden_and_idempotent(store, clock, events, scope4, index_provider, golden):
    """S6 transitions == §19.6 tuples, golden JSONL (masked), uniform κ_t keys, idempotent, filters, InterfaceEP row.
    S6 转移 == §19.6 元组、黄金 JSONL（掩码）、统一 κ_t 键、幂等、过滤、InterfaceEP 行。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        golden: Golden helper (masked JSONL).
        golden：黄金比对辅助（掩码 JSONL）。

    Output / 输出：
        Assertions on the five S6 tuples and the export contract.
        对五个 S6 元组与导出契约的断言。

    State effect / 状态影响：
        Rewrites transitions_s6.jsonl under --update-golden.
        --update-golden 下重写 transitions_s6.jsonl。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    s6 = _run_s6(h, ids, scope4)
    tr, cp, fb, rb = s6["tr"], s6["cp"], s6["fb"], s6["rb"]
    uc2 = fb["actions"][1]["atom_id"]
    anc_fb = store.list_feedback(ids["mem_uc"])[0].anchor_id
    tuples = [t.to_tuple() for t in tr]
    assert len(tuples) == 5 and all(t[5] for t in tuples)
    s_t, a_t, s_t1, r_t, kappa, w_t = tuples[0]
    assert (a_t, r_t, w_t) == (S4_EP1_A, "negative", [ids["anc_e2"], ids["anc_e1"]])
    assert s_t["u"] == S4_QUERY and s_t["ep_id"] == ids["ep1"].ep_id and s_t1["y"] == S4_EP1_Y
    assert kappa["constraints"] == [S4_CONSTRAINT] and kappa["provenance"] == {"evidence": [ids["anc_e2"], ids["anc_e1"]], "ep_id": ids["ep1"].ep_id}  # fmt: skip
    s_t, a_t, s_t1, r_t, kappa, w_t = tuples[1]
    assert (a_t, r_t, w_t) == (S4_EP2_A, "positive", [ids["anc_e4"], ids["anc_e3"]]) and s_t["ep_id"] == ids[
        "ep2"
    ].ep_id
    s_t, a_t, s_t1, r_t, kappa, w_t = tuples[2]
    causal = ids["edges"][("causal", ids["ep1"].atom_id, ids["ep2"].atom_id)]
    assert s_t["ep_id"] == ids["ep1"].ep_id and s_t["y"] == S4_EP1_Y and s_t1["ep_id"] == ids["ep2"].ep_id
    assert (a_t, r_t) == (S4_EP2_A, "positive") and w_t == [ids["anc_e3"], ids["anc_e2"], ids["anc_e4"]]
    assert kappa["constraints"] == [S4_CONSTRAINT, S4_DC] and kappa["provenance"]["edge_id"] == causal.edge_id
    s_t, a_t, s_t1, r_t, kappa, w_t = tuples[3]
    assert (s_t["text"], s_t["atom_id"], a_t, s_t1["text"], s_t1["atom_id"], r_t) == (S4_CONSTRAINT, ids["uc"], "revise", S6_CORRECTION, uc2, "negative")  # fmt: skip
    assert w_t == [anc_fb, ids["anc_uc"]]
    assert kappa == {
        "constraints": [S4_DC, S6_CORRECTION],
        "visibility": "private",
        "permission_scope": "private",
        "provenance": {"label": "", "created_by": "feedback", "proposer": "rule"},
    }
    s_t, a_t, s_t1, r_t, kappa, w_t = tuples[4]
    assert (s_t["text"], s_t["atom_id"], a_t, s_t1["text"], s_t1["atom_id"], r_t) == (S6_CORRECTION, uc2, "rollback", S4_CONSTRAINT, ids["uc"], "neutral")  # fmt: skip
    assert w_t == [rb["anchor_id"]]
    assert kappa == {
        "constraints": [S4_CONSTRAINT, S4_DC],
        "visibility": "private",
        "permission_scope": "private",
        "provenance": {"checkpoint_id": cp, "label": "before-feedback", "reason": None},
    }
    assert [sorted(t[4]) for t in tuples] == [sorted(KAPPA_KEYS)] * 5
    assert [t.source_kind for t in tr] == ["ep", "ep", "causal", "revision", "rollback"]
    golden.masked("transitions_s6.jsonl", [t.model_dump(mode="json") for t in tr])
    again = h.transitions(scope4)
    assert store.stats()["transitions"] == 5 and [t.transition_id for t in again] == [t.transition_id for t in tr]
    assert [t.source_kind for t in h.transitions(scope4, kinds=["ep"])] == ["ep", "ep"]
    assert [t.source_kind for t in h.transitions(scope4, since=tr[3].time_t1)] == ["revision", "rollback"]
    assert h.transitions(scope4, scale="L2_meso") == []
    rep = h.consolidate(scope4)
    assert rep["phases"]["consolidate"]["transitions"] == 1
    rows = h.transitions(scope4, kinds=["interface_ep"], scale="L2_meso")
    assert len(rows) == 1 and rows[0].s_t1["invariants"] == [S4_CONSTRAINT, f"do not: {S4_EP1_A}"]
    iep = store.list_interface_eps(Scope(**scope4))[0]
    assert rows[0].a_t == iep.slots.a and rows[0].r_t == "positive" and rows[0].w_t == iep.evidence_anchor_ids
    assert (rows[0].time_t, rows[0].time_t1) == ("2026-06-15T10:04:00.000000+00:00", "2026-06-15T10:14:00.000000+00:00")
    assert store.stats()["transitions"] == 6


# ---------------------------------------------------------------------------
# 56
# ---------------------------------------------------------------------------
def test_consolidate_phases_reports_and_decay(store, clock, events, scope4, index_provider):
    """consolidate() after S4/S6 == §19.7; MERGE keeps mentions; compiled card persisted; prune decays and archives.
    S4/S6 之后的 consolidate() == §19.7；MERGE 保持 mentions；编译卡片持久化；prune 衰减并归档。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on every phase report, graph stats, compiled state, decay/archive and collect re-routing.
        对每个阶段报告、图统计、编译态、衰减/归档与 collect 重分流的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    _run_s6(h, ids, scope4)
    scope = Scope(**scope4)
    with pytest.raises(ValidationError) as exc:
        h.consolidate(scope4, phases=["orient", "bogus"])
    assert exc.value.error_code == "VALIDATION_013"

    rep = h.consolidate(scope4)
    consolidated_at = clock.now()
    assert rep["phases"]["consolidate"] == CONSOLIDATE_S4
    assert list(rep["phases"]) == ["orient", "collect", "consolidate", "prune"]
    assert rep["phases"]["collect"] == {"re_embedded": 0, "re_routed": 0}
    assert rep["phases"]["orient"]["coarsen_candidates"] == [[ids["ep1"].atom_id, ids["ep2"].atom_id]]
    assert rep["phases"]["orient"]["last_compiled"] is None and rep["phases"]["orient"]["open_failures"] == []
    assert rep["phases"]["prune"]["decayed"] == 0 and rep["phases"]["prune"]["archived"] == []
    temporal = store.find_edge(ids["ep1"].atom_id, ids["ep2"].atom_id, "temporal")
    causal = store.find_edge(ids["ep1"].atom_id, ids["ep2"].atom_id, "causal")
    assert temporal.mentions == 2 and causal.mentions == 2 and temporal.confidence == 0.5 and causal.confidence == 0.7
    stats = h.graph.stats(scope)
    assert stats["nodes"] == {"message": 0, "observation": 3, "event": 4, "entity": 0, "ep": 2, "interface_ep": 1, "skill": 1}  # fmt: skip
    assert stats["edges"] == {"temporal": 1, "causal": 1, "use": 5, "distill": 3, "revision": 2, "conflict": 0}
    assert (stats["valid_edges"], stats["invalid_edges"], stats["boundary_edges"], stats["transitions"]) == (
        11,
        1,
        3,
        6,
    )
    assert stats["traceable_high_edges"] == 1.0 and stats["projection_coverage"] == 1.0
    assert h.graph.stats(Scope(user_id=U, agent_id="coder"))["nodes"]["entity"] == 6
    entities = store.axis_select(Scope(user_id=U, agent_id="coder"), where_sql="source_kind = ?", params=["entity"])
    keys = {"retry.ts", "file_diff", "billing_e2e", "test_log", "subscription_state_regression", "idempotency.ts"}
    assert len(entities) == 6 and {a.entity_ids[0] for a in entities} == keys
    assert all(a.scope == Scope(user_id=U, agent_id="coder") and len(a.entity_ids) == 1 for a in entities)
    assert all(a.content["mentions"] >= 1 and a.source_id.startswith("ent_") for a in entities)
    last = store.last_compiled_state(scope)
    assert last["as_of"] == consolidated_at and last["used_tokens"] > 0
    assert last["card"] == h.compile(S4_QUERY, scope4, mode="deep", as_of=consolidated_at).to_text()
    assert "## Skills\n- Skill: Do not modify the subscription state machine." in last["card"]
    assert store.get_meta(f"work_state:{scope.key()}") is not None
    assert store.get_meta(f"last_consolidation:{scope.key()}") == consolidated_at
    orient = h.consolidate(scope4, phases=["orient"])["phases"]["orient"]
    assert orient["last_compiled"]["compiled_id"] == last["compiled_id"] and "card" not in orient["last_compiled"]
    assert rep["anchor_id"].startswith("anc_") and store.get_anchor(rep["anchor_id"]) is not None
    again = h.consolidate(scope4, phases=["consolidate"])["phases"]["consolidate"]
    assert {k: v for k, v in again.items() if k != "compiled_states"} == {k: 0 for k in CONSOLIDATE_S4 if k != "compiled_states"}  # fmt: skip

    # prune: decay and archive
    # prune：衰减与归档
    pinned = _seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text="Pinned note about billing.", topic_tags=["important"], confidence=0.2)  # fmt: skip
    left = _seed_state(h, scope=scope, ts="2026-06-15T11:01:00Z", text="Alice lives in Paris.", predicate="lives_in", objects=["Paris"], confidence=0.2)  # fmt: skip
    right = _seed_state(h, scope=scope, ts="2026-06-15T11:02:00Z", text="Alice lives in Berlin.", predicate="lives_in", objects=["Berlin"], confidence=0.2)  # fmt: skip
    h.graph.insert_edge(src_id=left.atom_id, dst_id=right.atom_id, edge_type="conflict", witness_anchor_ids=right.anchor_ids, created_by="ingest")  # fmt: skip
    store.set_atom_confidence(ids["e1"], 0.2, at=clock.now())
    before = {a: store.get_atom(a).confidence for a in (ids["uc"], ids["dc"], ids["ep1"].atom_id, ids["ep2"].atom_id)}
    clock.tick(days=400)
    rep2 = h.consolidate(scope4, phases=["prune"])
    prune = rep2["phases"]["prune"]
    assert prune["decayed"] == 3 and prune["archived"] == [ids["e1"]] and prune["skipped_conflict"] == 2
    assert prune["examined"] == 6 and prune["expired"] >= 2
    for atom_id in (ids["e2"], ids["e3"], ids["e4"]):
        atom = store.get_atom(atom_id)
        assert atom.confidence == 0.61 and atom.archived_reason is None
        rows = [r for r in store.get_history(atom.memory_id) if r["event"] == "UPDATE"]
        assert rows and rows[-1]["anchor_id"] == rep2["anchor_id"]
    e1 = store.get_atom(ids["e1"])
    assert e1.archived_reason == "decayed" and e1.confidence == 0.2
    archive_rows = [r for r in store.get_history(e1.memory_id) if r["event"] == "ARCHIVE"]
    assert archive_rows and archive_rows[-1]["anchor_id"] == rep2["anchor_id"]
    assert store.get_atom(pinned.atom_id).confidence == 0.2 and store.get_atom(pinned.atom_id).archived_reason is None
    assert (
        store.get_atom(left.atom_id).archived_reason is None and store.get_atom(right.atom_id).archived_reason is None
    )
    for atom_id, confidence in before.items():
        assert store.get_atom(atom_id).confidence == confidence and store.get_atom(atom_id).archived_reason is None
    skill = store.axis_select(scope, where_sql="source_kind = ?", params=["skill"])
    assert len(skill) == 1 and skill[0].confidence == 0.8
    assert all(store.get_atom(a.atom_id).archived_reason is None for a in entities)

    # collect re-routes an event anchored by Txn A whose Txn B never ran (or was rolled back)
    # collect 会重新接管由 Txn A 锚定、而 Txn B 从未执行（或已回滚）的事件
    orphan = events_from_dicts(
        [
            {
                "object_type": "test_log",
                "role": "tool",
                "timestamp": "2026-06-15T10:30:00Z",
                "payload": {"suite": "billing_smoke", "passed": 1, "failed": 0},
            }
        ],  # fmt: skip
        scope_fallback=scope,
        visibility_default="private",
    )
    ((_ev, orphan_anchor, _d),) = anchor_events(store, orphan)
    assert store.atoms_for_anchor(orphan_anchor.anchor_id) == []
    rep3 = h.consolidate(scope4, phases=["collect"])
    assert rep3["phases"]["collect"]["re_routed"] == 1
    assert len(store.atoms_for_anchor(orphan_anchor.anchor_id)) == 1
    assert h.consolidate(scope4, phases=["collect"])["phases"]["collect"]["re_routed"] == 0


# ---------------------------------------------------------------------------
# 56b
# ---------------------------------------------------------------------------
def test_revoke_cascade(store, clock, events, scope4, index_provider):
    """revoke(<anc_e2>) archives its only-witnessed atom, invalidates its only-witnessed edge, keeps shared objects valid.
    revoke(<anc_e2>) 归档仅由它见证的原子、使仅由它见证的边失效，共享对象保持有效。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the revoke report, disputes, history, PC-IR not_revoked and idempotence.
        对撤销报告、争议、历史、PC-IR not_revoked 与幂等性的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    distill = ids["edges"][("distill", ids["ep1"].atom_id, ids["dc"])]
    with pytest.raises(ValueError):
        h.revoke("anc_nonsense")
    rv = h.revoke(ids["anc_e2"], reason="log was from the wrong branch")
    assert rv == {
        "revoked": ids["anc_e2"],
        "archived": [ids["e2"]],
        "edges_invalidated": [distill.edge_id],
        "interface_eps_invalidated": [],
        "anchor_id": rv["anchor_id"],
    }
    assert rv["anchor_id"].startswith("anc_") and store.get_anchor(ids["anc_e2"]).revoked_at == clock.now()
    assert store.get_atom(ids["e2"]).archived_reason == "revoked"
    for atom_id in (ids["ep1"].atom_id, ids["dc"]):
        atom = store.get_atom(atom_id)
        assert atom.archived_reason is None and ids["anc_e2"] not in store.atom_anchor_ids(atom_id)
        assert ids["anc_e2"] in store.atom_anchor_ids(atom_id, include_disputed=True)
    assert store.get_edge(distill.edge_id).invalidated_reason == "witness_revoked"
    # anc_e3 仍然见证着这条边
    assert store.find_edge(ids["ep1"].atom_id, ids["ep2"].atom_id, "causal").valid  # anc_e3 still witnesses it
    e2_history = store.get_history(store.get_atom(ids["e2"]).memory_id)
    assert e2_history[-1]["event"] == "ARCHIVE" and e2_history[-1]["anchor_id"] == rv["anchor_id"]
    # 证据永远不会被删除
    assert store.stats()["transitions"] == 2  # evidence is never deleted
    ir = h.compile(S4_QUERY, scope4, mode="deep")
    assert ir.to_pcir().ok() and not [i for i in ir.items() if i.atom_id == ids["e2"]]
    assert h.revoke(ids["anc_e2"]) == {
        "revoked": ids["anc_e2"],
        "archived": [],
        "edges_invalidated": [],
        "interface_eps_invalidated": [],
        "anchor_id": None,
        "noop": True,
    }
    # an InterfaceEP whose evidence is entirely revoked is invalidated too
    # 证据被全部撤销的 InterfaceEP 也会被置为失效
    h.consolidate(scope4, phases=["consolidate"])
    iep = store.list_interface_eps(Scope(**scope4))[0]
    reports = [h.revoke(a) for a in iep.evidence_anchor_ids if a != ids["anc_e2"]]
    assert reports[-1]["interface_eps_invalidated"] == [iep.interface_ep_id]
    assert store.get_interface_ep(iep.interface_ep_id).valid is False


# ---------------------------------------------------------------------------
# 56e / 56f — witness semantics (docs/design/WITNESS_SEMANTICS.md)
# 56e / 56f —— 见证语义（docs/design/WITNESS_SEMANTICS.md）
# ---------------------------------------------------------------------------
def _assert_existence_rule(store: StateStore, scope: Scope) -> None:
    """Every valid edge keeps an unrevoked witness and (when projected) a valid source; revoke-invalidated edges lack one.
    每条有效边都保有未撤销见证且（若为投影边）保有有效来源；被撤销级联失效的边缺其一。
    """
    for edge in store.list_edges(scope, valid_only=False):
        witnesses = store.get_anchors(store.edge_witnesses(edge.edge_id))
        witnessed = any(a.revoked_at is None for a in witnesses)
        sources = [store.get_edge(s) for s in store.projections_of(edge.edge_id)]
        sourced = not sources or any(s is not None and s.valid for s in sources)
        if edge.valid:
            assert witnessed and sourced, (edge.edge_id, edge.edge_type, str(edge.scale))
        elif edge.invalidated_reason in ("witness_revoked", "projection_revoked"):
            assert not (witnessed and sourced), (edge.edge_id, edge.invalidated_reason)


def _s4_projection_ids(h: _Harness, ids: dict, scope4: dict) -> dict[str, Any]:
    """After S4 + consolidate: the two low `use` edges to <atm_uc>, their L2 projection and the L3 skill distill edge.
    S4 + consolidate 之后：指向 <atm_uc> 的两条低层 `use` 边、它们的 L2 投影与 L3 技能 distill 边。
    """
    store, scope = h.store, Scope(**scope4)
    iep = store.list_interface_eps(scope)[0]
    low1 = store.find_edge(ids["ep1"].atom_id, ids["uc"], "use")
    low2 = store.find_edge(ids["ep2"].atom_id, ids["uc"], "use")
    high = store.find_edge(iep.atom_id, ids["uc"], "use", "L2_meso")
    distill = [e for e in store.list_edges(scope) if e.edge_type == "distill" and str(e.scale) == "L3_macro"]
    assert low1 is not None and low2 is not None and high is not None and len(distill) == 1
    assert set(store.projections_of(high.edge_id)) == {low1.edge_id, low2.edge_id}
    assert set(store.projections_of(distill[0].edge_id)) == {low1.edge_id, low2.edge_id}
    return {"iep": iep, "low1": low1, "low2": low2, "high": high, "distill": distill[0]}


def test_revoke_existence_semantics_and_projection_sources(store, clock, events, scope4, index_provider):
    """Revoking one of several witnesses keeps atoms, IEPs and projected edges valid; a projected edge falls with its last source.
    撤销多个见证之一时原子、IEP 与投影边保持有效；投影边随其最后一个有效来源一起倒下。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on primary-witness revocation, multi-source projected edges, reasons, PC-IR and the global rule.
        对主见证撤销、多来源投影边、失效原因、PC-IR 与全局规则的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    h.consolidate(scope4)
    scope = Scope(**scope4)
    p = _s4_projection_ids(h, ids, scope4)
    ep1, ep2, iep = ids["ep1"], ids["ep2"], p["iep"]
    # EP 原子的见证是 [outcome, action]，outcome 锚点是主见证 / EP atoms are witnessed by [outcome, action]; outcome is primary
    assert list(ep1.anchor_ids) == [ids["anc_e2"], ids["anc_e1"]] and list(ep2.anchor_ids) == [
        ids["anc_e4"],
        ids["anc_e3"],
    ]
    _assert_existence_rule(store, scope)

    # (1) 撤销 EP2 的主见证：原子靠剩下的 action 锚点存活 / revoke EP2's PRIMARY witness: the atom survives on its action anchor
    rv1 = h.revoke(ids["anc_e4"], reason="wrong test run")
    assert rv1["archived"] == [ids["e4"]] and rv1["interface_eps_invalidated"] == []
    assert store.get_atom(ep2.atom_id).archived_reason is None
    assert store.atom_anchor_ids(ep2.atom_id) == [ids["anc_e3"]]
    assert store.atom_anchor_ids(ep2.atom_id, include_disputed=True) == [ids["anc_e4"], ids["anc_e3"]]
    assert store.get_interface_ep(iep.interface_ep_id).valid
    for e in (p["low1"], p["low2"], p["high"], p["distill"]):
        assert store.get_edge(e.edge_id).valid
    _assert_existence_rule(store, scope)

    # (2) 撤销 EP1 的 action 锚点：low1 倒下，但 high / distill 还有 low2 这个有效来源
    # revoke EP1's action anchor: low1 falls; high / distill keep low2 as a valid source
    rv2 = h.revoke(ids["anc_e1"], reason="patch was never applied")
    assert rv2["archived"] == [ids["e1"]] and p["low1"].edge_id in rv2["edges_invalidated"]
    assert store.get_edge(p["low1"].edge_id).invalidated_reason == "witness_revoked"
    assert p["high"].edge_id not in rv2["edges_invalidated"] and p["distill"].edge_id not in rv2["edges_invalidated"]
    assert store.get_edge(p["high"].edge_id).valid and store.get_edge(p["distill"].edge_id).valid
    assert store.get_atom(ep1.atom_id).archived_reason is None  # anc_e2 still witnesses it / anc_e2 仍见证它
    _assert_existence_rule(store, scope)
    ir = h.compile(S4_QUERY, scope4, mode="deep")
    assert ir.to_pcir().ok()  # not_revoked evaluates the undisputed subset / not_revoked 只看未争议子集

    # (3) 给 high 加一个与任何低层边无关的见证，再撤销最后一个来源的锚点：high 与 distill 因来源全失效而倒下
    # add a witness to `high` that no low edge shares, then revoke the last source's anchor:
    # high and distill fall because EVERY source is invalid, although their own witnesses survive
    store.add_edge_witness(p["high"].edge_id, ids["anc_uc"])
    rv3 = h.revoke(ids["anc_e3"], reason="idempotency patch reverted")
    assert (
        ids["e3"] in rv3["archived"] and ep2.atom_id in rv3["archived"]
    )  # no undisputed witness left / 已无未争议见证
    assert store.get_atom(ep2.atom_id).archived_reason == "revoked"
    assert p["low2"].edge_id in rv3["edges_invalidated"]
    assert store.get_edge(p["low2"].edge_id).invalidated_reason == "witness_revoked"
    for e in (p["high"], p["distill"]):
        assert e.edge_id in rv3["edges_invalidated"]
        stored = store.get_edge(e.edge_id)
        assert stored.valid is False and stored.invalidated_reason == "projection_revoked"
        alive = [a for a in store.get_anchors(store.edge_witnesses(e.edge_id)) if a.revoked_at is None]
        assert alive  # fell by provenance, not by witnesses / 因来源而非见证倒下
    # 投影边排在见证规则失效的边之后 / projected edges are reported after the witness-rule ones
    assert rv3["edges_invalidated"].index(p["low2"].edge_id) < rv3["edges_invalidated"].index(p["high"].edge_id)
    assert store.get_interface_ep(iep.interface_ep_id).valid  # anc_e2 still in its evidence / anc_e2 仍在证据里
    _assert_existence_rule(store, scope)
    assert h.revoke(ids["anc_e3"])["noop"] is True


def test_relationship_not_useful_keeps_high_edge_with_other_sources(store, clock, events, scope4, index_provider):
    """`relationship_not_useful` on a low edge drops its projections only once no other valid source remains.
    对低层边的 `relationship_not_useful` 只在没有其他有效来源时才连带其投影边。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the feedback action lists and edge validity / reasons.
        对反馈动作列表与边的有效性 / 原因的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    h.consolidate(scope4)
    p = _s4_projection_ids(h, ids, scope4)
    out1 = h.feedback(p["low1"].edge_id, "relationship_not_useful")
    assert out1["actions"] == [
        {"action": "invalidate_edge", "edge_id": p["low1"].edge_id, "reason": "feedback:not_useful"}
    ]
    assert store.get_edge(p["high"].edge_id).valid and store.get_edge(p["distill"].edge_id).valid
    out2 = h.feedback(p["low2"].edge_id, "relationship_not_useful")
    assert _names(out2["actions"]) == ["invalidate_edge"] * 3
    assert [a["edge_id"] for a in out2["actions"]] == [p["low2"].edge_id, p["high"].edge_id, p["distill"].edge_id]
    for e in (p["low2"], p["high"], p["distill"]):
        stored = store.get_edge(e.edge_id)
        assert stored.valid is False and stored.invalidated_reason == "feedback:not_useful"
    _assert_existence_rule(store, Scope(**scope4))


# ---------------------------------------------------------------------------
# 56c
# ---------------------------------------------------------------------------
def test_entity_atoms_after_consolidate(store, clock, events, index_provider):
    """S2/S3 + consolidate(filters={'user_id': U}) materializes one 'Sarah' entity atom; idempotent; revised on growth.
    S2/S3 + consolidate(filters={'user_id': U}) 物化一个 'Sarah' 实体原子；幂等；增长时修订。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the entity atom, the entity-axis query and the revision after a fourth mention.
        对实体原子、实体轴查询与第四次提及后修订的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    for message in events("chat_multisession").messages:
        h.add(message["content"], timestamp=message["timestamp"], user_id=U, run_id=message["run_id"])
    h.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z", user_id=U, run_id="run-coffee")
    rep = h.consolidate({"user_id": U})
    assert rep["phases"]["consolidate"]["entity_atoms"] == 1 and rep["phases"]["consolidate"]["compiled_states"] == 4
    entity_scope = Scope(user_id=U)
    rows = store.axis_select(entity_scope, scale_constraint=["entity"])
    assert len(rows) == 1
    sarah = rows[0]
    assert sarah.text == "Sarah" and sarah.atom_type == "entity" and sarah.source_kind == "entity"
    assert sarah.content["mentions"] == 3 and sarah.scope == entity_scope and sarah.predicate == "is_entity"
    sarah_atoms = store.axis_select(entity_scope, where_sql="source_kind <> ?", params=["entity"])
    sarah_atoms = [a for a in sarah_atoms if "sarah" in a.entity_ids]
    assert sarah.anchor_ids == sorted({a.anchor_ids[0] for a in sarah_atoms}) and len(sarah.anchor_ids) == 3
    assert "sarah" in sarah.entity_ids
    result = execute_axis_query(
        store,
        h.index,
        h.embedder,
        AxisQuery(filters={"user_id": U, "entity": "Sarah"}, scale_constraint=["event", "entity", "L1_micro"]),
        scoring=h.config.scoring,
        capabilities=store.capabilities,
    )
    hit_ids = {hit.atom.atom_id for hit in result.hits}
    assert sarah.atom_id in hit_ids and len(hit_ids) == 3
    assert {hit.atom.atom_type for hit in result.hits} == {"entity", "event"}
    rep2 = h.consolidate({"user_id": U})
    assert rep2["phases"]["consolidate"]["entity_atoms"] == 0
    assert len(store.axis_select(entity_scope, scale_constraint=["entity"])) == 1
    h.add("Sarah presented the retry design to the team.", timestamp="2026-06-20T09:00:00Z", user_id=U, run_id="run-d")
    rep3 = h.consolidate({"user_id": U})
    assert rep3["phases"]["consolidate"]["entity_atoms"] == 1
    head = store.resolve_head(sarah.memory_id)
    assert head.version == 2 and head.content["mentions"] == 4 and len(head.anchor_ids) == 4
    edge = store.find_edge(sarah.atom_id, head.atom_id, "revision")
    assert edge is not None and edge.created_by == "consolidation" and len(edge.witness_anchor_ids) == 1
    witness_event = store.get_source_event(store.get_anchor(edge.witness_anchor_ids[0]).source_event_id)
    assert witness_event.object_type == "scheduler" and witness_event.timestamp == clock.now()
    assert store.get_anchor(rep3["anchor_id"]).timestamp == clock.now()
    assert store.get_atom(sarah.atom_id).archived_reason == "revised"


# ---------------------------------------------------------------------------
# extra: L3 coarsening across runs, the LLM edge-proposal path, coarsen_l3 off
# 补充：跨 run 的 L3 粗化、LLM 提边路径、coarsen_l3 关闭
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("proposer", ["rule", "llm"])
def test_consolidate_l3_across_runs_and_llm_proposer(store, clock, events, index_provider, proposer):
    """Two S4-like runs joined by a causal edge: consolidate coarsens each run to L2 and both L2s to one L3 InterfaceEP.
    由 causal 边相连的两个 S4 式运行：consolidate 将每个运行粗化为 L2，并将两个 L2 粗化为一个 L3 InterfaceEP。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        proposer: graph.causal_proposer ('rule' in-process, 'llm' through the edge_proposal task of the rule LLM).
        proposer：graph.causal_proposer（'rule' 进程内；'llm' 经规则 LLM 的 edge_proposal 任务）。

    Output / 输出：
        Assertions on interface_eps / interface_eps_l3 / skills counts and the L3 support set.
        对 interface_eps / interface_eps_l3 / skills 计数与 L3 支持集的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider, graph={"causal_proposer": proposer, "coarsen_l3": True})
    scope_a = {"user_id": U, "agent_id": "coder", "run_id": "run-a"}
    scope_b = {"user_id": U, "agent_id": "coder", "run_id": "run-b"}
    ids_a = _build_s4(h, events, scope_a)
    ids_b = _build_s4(h, events, scope_b)
    link = h.graph.insert_edge(
        src_id=ids_a["ep2"].atom_id,
        dst_id=ids_b["ep1"].atom_id,
        edge_type="causal",
        witness_anchor_ids=[ids_b["anc_e1"]],
        created_by="ingest",
        proposer="closure",
    )
    joint = Scope(user_id=U, agent_id="coder")
    rep = h.consolidate({"user_id": U, "agent_id": "coder"})["phases"]["consolidate"]
    assert rep["interface_eps"] == 2 and rep["interface_eps_l3"] == 1 and rep["temporal_edges"] == 0
    assert rep["causal_edges"] == 0 and rep["skills"] >= 1 and rep["compiled_states"] == 2
    assert store.find_edge(ids_a["ep1"].atom_id, ids_a["ep2"].atom_id, "causal").mentions == 2
    l2 = store.list_interface_eps(joint, scale="L2_meso")
    l3 = store.list_interface_eps(joint, scale="L3_macro")
    assert len(l2) == 2 and len(l3) == 1 and l3[0].scope == joint
    assert sorted(l3[0].support_node_ids) == sorted(i.atom_id for i in l2)
    joined = store.find_edge(l2[0].atom_id, l2[1].atom_id, "causal", "L2_meso")
    assert joined is not None and link in store.projections_of(joined.edge_id)
    assert [inv.text for inv in l3[0].invariants if inv.kind == "constraint"] == [S4_CONSTRAINT, S4_DC]
    again = h.consolidate({"user_id": U, "agent_id": "coder"})["phases"]["consolidate"]
    assert again["interface_eps"] == 0 and again["interface_eps_l3"] == 0 and again["skills"] == 0
    off = _Harness(StateStore(":memory:", now=clock.now, id_factory=SequentialIds()), clock, index_provider, graph={"coarsen_l3": False})  # fmt: skip
    _build_s4(off, events, scope_a)
    assert off.consolidate(scope_a)["phases"]["consolidate"]["interface_eps_l3"] == 0
    assert off.store.list_interface_eps(joint, scale="L3_macro") == []


# ---------------------------------------------------------------------------
# 56d
# ---------------------------------------------------------------------------
def test_consolidate_llm_proposer_drops_malformed_proposals(
    store, clock, events, scope4, index_provider, scripted_llm, caplog
):
    """causal_proposer='llm': malformed `witness_refs` are dropped with a warning and a non-finite `confidence` falls back to 0.7; the run completes.
    causal_proposer='llm'：畸形的 `witness_refs` 告警丢弃、非有限的 `confidence` 回退到 0.7；本次运行正常完成。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        scripted_llm: ScriptedLLM factory (replaces the scheduler's rule LLM).
        scripted_llm：ScriptedLLM 工厂（替换调度器的规则 LLM）。

        caplog: pytest log capture (one warning per dropped proposal / invalid confidence).
        caplog：pytest 日志捕获（每条丢弃的提议 / 无效置信度各一条告警）。

    Output / 输出：
        Assertions on the consolidate report, the persisted edges and the warnings.
        对 consolidate 报告、落库的边与告警的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider, graph={"causal_proposer": "llm"})
    ids = _build_s4(h, events, scope4)
    ep1, ep2 = ids["ep1"].atom_id, ids["ep2"].atom_id
    before = store.find_edge(ep1, ep2, "causal")
    assert before is not None and before.confidence == 0.7 and before.mentions == 1
    reply = {
        "edges": [
            {"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal", "witness_refs": 7},
            {"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal", "witness_refs": [["w1"]]},
            {"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal", "witness_refs": [{"ref": "w1"}]},
            {"src_ref": "n0", "dst_ref": "n1", "edge_type": "causal", "witness_refs": ["w1"], "confidence": "nan"},
            {"src_ref": "n1", "dst_ref": "n0", "edge_type": "use", "witness_refs": ["w0"], "confidence": float("nan")},
        ]
    }
    h.scheduler.llm = scripted_llm([reply])
    with caplog.at_level(logging.WARNING, logger="hms_elite.ops.consolidation"):
        rep = h.consolidate(scope4, phases=["consolidate"])["phases"]["consolidate"]
    assert [c["task"] for c in h.scheduler.llm.calls] == ["edge_proposal"]
    assert rep["causal_edges"] == 1 and rep["interface_eps"] == 1 and rep["compiled_states"] == 1
    assert store.get_meta(f"last_consolidation:{Scope(**scope4).key()}") is not None
    # neither the NaN literal nor the 'nan' string may launder into 1.0; MERGE takes MAX(confidence), so 0.7 must stay
    # NaN 字面量与 'nan' 字符串都不得"洗白"成 1.0；MERGE 取 MAX(confidence)，故已有边必须保持 0.7
    merged = store.find_edge(ep1, ep2, "causal")
    assert merged.confidence == 0.7 and merged.mentions == 2
    use = store.find_edge(ep2, ep1, "use")
    assert use is not None and use.confidence == 0.7 and use.proposer == "scripted@v1"
    assert use.created_by == "consolidation"
    messages = [r.getMessage() for r in caplog.records]
    assert sum("witness_refs must be a list" in m for m in messages) == 3
    assert sum("invalid confidence" in m for m in messages) == 2


# ---------------------------------------------------------------------------
# 56g–56k: retention escape hatch (§16.6)
# 56g–56k：保留策略逃生舱（§16.6）
# ---------------------------------------------------------------------------
def _seed_raw(h: _Harness, *, scope: Scope, ts: str, text: str) -> MemoryAtom:
    """Insert a state atom through store.insert_atom only (no node, no history row): its anchor is cited by nothing else.
    只经 store.insert_atom 插入一个 state 原子（无节点、无历史行）：其锚点不被任何其它对象引用。
    """
    ((ev, anc, _dedup),) = anchor_events(
        h.store,
        events_from_dicts(
            [{"object_type": "message", "role": "user", "timestamp": ts, "payload": {"text": text}}],
            scope_fallback=scope,
            visibility_default="private",
        ),
    )
    return h.store.insert_atom(
        MemoryAtom(
            atom_type="state",
            content={"text": text, "objects": []},
            text=text,
            time_anchor=ts,
            scale_level="state",
            subject="user",
            predicate="is",
            obs_type="fact",
            source_kind="observation",
            source_id=f"obs_{ev.event_id}",
            extracted_by="test",
            scope=scope,
            anchor_ids=[anc.anchor_id],
        )
    )


def _tombstone(h: _Harness, atom: MemoryAtom, reason: str = "deleted") -> str:
    """Archive an atom the way delete()/prune do (archive_atom + history row) and return its id.
    像 delete()/prune 那样归档原子（archive_atom + 历史行）并返回其 id。
    """
    now = h.clock.tick(seconds=60)
    h.store.archive_atom(atom.atom_id, reason=reason, at=now)
    h.store.add_history(
        memory_id=atom.memory_id,
        old_memory=atom.text,
        new_memory=None,
        event="DELETE" if reason == "deleted" else "ARCHIVE",
        created_at=atom.created_at or now,
        updated_at=now,
        is_deleted=1 if reason == "deleted" else 0,
        anchor_id=atom.anchor_ids[0],
        atom_id=atom.atom_id,
    )
    return atom.atom_id


def test_purge_dry_run_reports_and_changes_nothing(store, clock, events, scope4, index_provider):
    """purge(dry_run=True) lists the purgeable tombstones with sha256 + byte estimate and writes no row at all.
    purge(dry_run=True) 列出可清理墓碑（含 sha256 与字节估算），且一行都不写。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the report, on store.stats() equality and on the argument validation codes.
        对报告、store.stats() 相等性与参数校验错误码的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    _build_s4(h, events, scope4)
    scope = Scope(**scope4)
    gone = _seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text="Old note that was deleted.")
    live = _seed_state(h, scope=scope, ts="2026-06-15T11:01:00Z", text="Live note that stays.")
    _tombstone(h, gone)
    clock.tick(days=100)
    before = store.stats()
    history_before = store.get_history(gone.memory_id)
    rep = purge(store, h.index, scope=scope, older_than_days=90, dry_run=True)
    assert rep["dry_run"] is True and rep["anchor_id"] is None and rep["purged"] == 0
    assert rep["scope"] == scope.key() and rep["older_than_days"] == 90 and rep["cutoff"] < clock.now()
    assert [p["atom_id"] for p in rep["purgeable"]] == [gone.atom_id]
    summary = rep["purgeable"][0]
    assert summary["memory_id"] == gone.memory_id and summary["version"] == 1
    assert summary["text_sha256"] == sha256_hex(gone.text) and "text" not in summary
    assert summary["archived_reason"] == "deleted" and summary["valid_to"] == store.get_atom(gone.atom_id).valid_to
    assert summary["bytes"] > len(gone.text) and rep["estimated_bytes"] == summary["bytes"]
    assert set(rep["protected"]) == set(PROTECTED_REASONS)
    # Nothing moved: same row counts, same history, the tombstone and its vector are still there.
    # 什么都没动：行数相同、历史相同、墓碑及其向量仍在。
    assert store.stats() == before and store.get_history(gone.memory_id) == history_before
    assert store.get_atom(gone.atom_id) is not None and h.index.get(gone.atom_id) is not None
    assert store.get_atom(live.atom_id).archived_reason is None
    # A young tombstone is not examined; the defaults are dry_run=True / keep_witnessed=True.
    # 年轻的墓碑不在受检范围内；默认 dry_run=True / keep_witnessed=True。
    young = purge(store, h.index, scope=scope, older_than_days=200)
    assert young["examined"] == 0 and young["purgeable"] == [] and young["dry_run"] and young["keep_witnessed"]
    with pytest.raises(ValidationError) as exc:
        purge(store, h.index, scope=scope, older_than_days=-1)
    assert exc.value.error_code == "VALIDATION_009"
    with pytest.raises(ValidationError) as exc:
        purge(store, h.index, scope=Scope(), older_than_days=90)
    assert exc.value.error_code == "VALIDATION_001"
    assert store.stats() == before


def test_purge_protections_keep_needed_rows(store, clock, events, scope4, index_provider):
    """Multi-version heads, edge endpoints, checkpoint-live versions, InterfaceEP support and shared anchors survive a purge.
    多版本头、边端点、检查点存活版本、InterfaceEP 支持节点与共享锚点都能在清理中幸存。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions: each protected id is reported under its reason and still present after purge(dry_run=False).
        断言：每个受保护 id 都在其原因下被报告，且 purge(dry_run=False) 之后仍然存在。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4)
    h.consolidate(scope4, phases=["consolidate"])
    scope = Scope(**scope4)
    # (a) head_version: v1 → feedback successor v2 → delete v2; the lineage keeps both (v2 head, v1 edge endpoint).
    # (a) head_version：v1 → 反馈后继 v2 → 删除 v2；谱系两者都保留（v2 为头、v1 为边端点）。
    v1 = _seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text="Alice works in Paris.", predicate="works_in", objects=["Paris"])  # fmt: skip
    h.feedback(v1.memory_id, "state_outdated", correction="Alice works in Berlin.")
    v2 = store.latest_version(v1.memory_id)
    assert v2.version == 2 and v2.atom_id != v1.atom_id
    _tombstone(h, v2)
    # (b) edge_endpoint: a tombstone that is the src of a valid use edge.
    # (b) edge_endpoint：作为有效 use 边源点的墓碑。
    src = _seed_state(h, scope=scope, ts="2026-06-15T11:02:00Z", text="Retry policy note.")
    dst = _seed_state(h, scope=scope, ts="2026-06-15T11:03:00Z", text="Backoff policy note.")
    h.graph.insert_edge(src_id=src.atom_id, dst_id=dst.atom_id, edge_type="use", witness_anchor_ids=dst.anchor_ids, created_by="ingest")  # fmt: skip
    _tombstone(h, src)
    # (c) checkpoint_live: recorded live in a snapshot, deleted afterwards.
    # (c) checkpoint_live：快照记录为存活、之后被删除。
    kept = _seed_state(h, scope=scope, ts="2026-06-15T11:04:00Z", text="Note recorded by a checkpoint.")
    cp = h.checkpoint(scope4, label="before-delete")
    _tombstone(h, kept)
    # (d) anchors: a tombstone whose anchor is also cited by a live atom; keep_witnessed=True never touches anchors.
    # (d) 锚点：其锚点同时被存活原子引用的墓碑；keep_witnessed=True 从不触碰锚点。
    shared = _seed_state(h, scope=scope, ts="2026-06-15T11:05:00Z", text="Shared-evidence note.")
    other = _seed_state(h, scope=scope, ts="2026-06-15T11:06:00Z", text="Other note citing the same evidence.")
    assert store.add_atom_anchor(other.atom_id, shared.anchor_ids[0], role="support") is True
    _tombstone(h, shared)
    # (e) interface_ep_support: the EP atoms are support nodes of the consolidated InterfaceEP.
    # (e) interface_ep_support：EP 原子是固化出的 InterfaceEP 的支持节点。
    ep1 = ids["ep1"].atom_id
    _tombstone(h, store.get_atom(ep1))
    clock.tick(days=100)
    anchors_before, events_before = store.stats()["anchors"], store.stats()["source_events"]
    rep = purge(store, h.index, scope=scope, older_than_days=90, dry_run=False)
    protected = rep["protected"]
    assert v2.atom_id in protected["head_version"] and v1.atom_id not in protected["head_version"]
    assert {v1.atom_id, v2.atom_id, src.atom_id} <= set(protected["edge_endpoint"])
    # ep1 was live at checkpoint time too: it is protected twice (checkpoint_live + interface_ep_support).
    # ep1 在检查点时也是存活的：它受到双重保护（checkpoint_live + interface_ep_support）。
    assert set(protected["checkpoint_live"]) == {kept.atom_id, ep1}
    assert ep1 in protected["interface_ep_support"] and ep1 in protected["edge_endpoint"]
    assert rep["purged"] == 1 and [p["atom_id"] for p in rep["purgeable"]] == [shared.atom_id]
    for atom_id in (v1.atom_id, v2.atom_id, src.atom_id, kept.atom_id, ep1):
        assert store.get_atom(atom_id) is not None, atom_id
    assert store.get_atom(shared.atom_id) is None
    # Evidence untouched: the only new anchor / event is the purge's own witness.
    # 证据未动：唯一新增的锚点 / 事件是清理自身的见证。
    assert store.stats()["anchors"] == anchors_before + 1 and store.stats()["source_events"] == events_before + 1
    assert store.get_anchor(shared.anchor_ids[0]) is not None and store.get_atom(other.atom_id).anchor_ids[1:] == [shared.anchor_ids[0]]  # fmt: skip
    # The checkpoint-live version is still restorable: rollback brings it back.
    # 检查点存活版本仍可恢复：回滚把它带回来。
    rb = h.rollback(cp)
    assert kept.atom_id in rb["restored"] and store.get_atom(kept.atom_id).archived_reason is None
    # keep_witnessed=False removes only anchors nothing cites: the shared one stays, a raw atom's private one goes.
    # keep_witnessed=False 只删除无人引用的锚点：共享锚点保留，裸原子的私有锚点被删除。
    lone = _seed_raw(h, scope=scope, ts="2026-06-15T11:07:00Z", text="Lone raw note.")
    lone_anchor = lone.anchor_ids[0]
    lone_event = store.get_anchor(lone_anchor).source_event_id
    store.archive_atom(lone.atom_id, reason="deleted", at=clock.tick(seconds=60))
    clock.tick(days=100)
    rep2 = purge(store, h.index, scope=scope, older_than_days=90, dry_run=False, keep_witnessed=False)
    # `other` was created after the checkpoint, so the rollback archived it: now old enough, edge-free → purged too;
    # its anchor stays (cited by its ADD history row), exactly like the shared one.
    # `other` 创建于检查点之后，因此回滚归档了它：如今足够旧、无边 → 一并清理；
    # 它的锚点保留（被其 ADD 历史行引用），与共享锚点一样。
    assert rep2["purged"] == 2 and sorted(p["atom_id"] for p in rep2["purgeable"]) == sorted([lone.atom_id, other.atom_id])  # fmt: skip
    assert rep2["anchors_purged"] == [lone_anchor] and rep2["events_purged"] == [lone_event]
    assert store.get_anchor(other.anchor_ids[0]) is not None
    assert store.get_anchor(lone_anchor) is None and store.get_source_event(lone_event) is None
    assert store.get_anchor(shared.anchor_ids[0]) is not None
    # +purge, +rollback, +lone, +purge, −lone: only the lone anchor went; both purge anchors are cited by history.
    # +purge、+rollback、+lone、+purge、−lone：只有 lone 的锚点被删；两个清理锚点都被 history 引用。
    assert store.stats()["anchors"] == anchors_before + 3


def test_purge_materializes_with_audit_and_keeps_survivors_intact(store, clock, events, scope4, index_provider):
    """purge(dry_run=False) removes tombstones + vectors under one anchor, leaves a PURGE history row and changes no survivor.
    purge(dry_run=False) 在一个锚点下删除墓碑与向量，留下 PURGE 历史行，且不改变任何幸存对象。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on rows, vectors, history, the purge anchor, an unchanged compiled card, a later rollback and idempotence.
        对行、向量、历史、清理锚点、不变的编译卡片、随后的回滚与幂等性的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    _build_s4(h, events, scope4)
    scope = Scope(**scope4)
    t1 = _seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text="First deleted note.")
    t2 = _seed_state(h, scope=scope, ts="2026-06-15T11:01:00Z", text="Second decayed note.")
    live = _seed_state(h, scope=scope, ts="2026-06-15T11:02:00Z", text="Live note.")
    _tombstone(h, t1)
    _tombstone(h, t2, reason="decayed")
    cp = h.checkpoint(scope4, label="after-tombstones")
    clock.tick(days=100)
    card_before = h.compile(S4_QUERY, scope4, mode="deep", as_of=S4_AS_OF).to_text()
    stats_before = store.stats()
    rep = purge(store, h.index, scope=scope, older_than_days=90, dry_run=False)
    assert rep["purged"] == 2 and rep["dry_run"] is False and rep["anchor_id"].startswith("anc_")
    assert [p["atom_id"] for p in rep["purgeable"]] == sorted([t1.atom_id, t2.atom_id])
    purge_anchor = store.get_anchor(rep["anchor_id"])
    purge_event = store.get_source_event(purge_anchor.source_event_id)
    assert purge_event.object_type == "system" and purge_event.payload["op"] == "purge"
    assert purge_event.payload["metadata"] == {"older_than_days": 90, "cutoff": rep["cutoff"], "keep_witnessed": True, "atoms": 2}  # fmt: skip
    for atom in (t1, t2):
        assert store.get_atom(atom.atom_id) is None and store.get_node(atom.atom_id) is None
        assert h.index.get(atom.atom_id) is None and store.atom_anchor_ids(atom.atom_id) == []
        rows = store.get_history(atom.memory_id)
        assert [r["event"] for r in rows][-2:] == ["DELETE" if atom is t1 else "ARCHIVE", "PURGE"]
        last = rows[-1]
        assert last["anchor_id"] == rep["anchor_id"] and last["is_deleted"] is True
        assert last["old_memory"] is None and last["new_memory"] is None
        details = store._execute("SELECT details_json FROM history WHERE id = ?", (last["id"],)).fetchone()
        assert f'"text_sha256":"{sha256_hex(atom.text)}"' in details["details_json"]
        assert f'"cutoff":"{rep["cutoff"]}"' in details["details_json"]
    after = store.stats()
    assert after["memory_atoms"] == stats_before["memory_atoms"] - 2 and after["graph_nodes"] == stats_before["graph_nodes"] - 2  # fmt: skip
    # Only the sqlite index keeps its vectors in atom_vectors; the memory index answered index.get() above.
    # 只有 sqlite 索引把向量放在 atom_vectors；memory 索引已由上面的 index.get() 覆盖。
    vectors_gone = 2 if index_provider == "sqlite" else 0
    assert after["atom_vectors"] == stats_before["atom_vectors"] - vectors_gone
    assert after["observations"] == stats_before["observations"]
    assert after["anchors"] == stats_before["anchors"] + 1 and after["source_events"] == stats_before["source_events"] + 1  # fmt: skip
    assert after["history"] == stats_before["history"] + 2 and after["graph_edges"] == stats_before["graph_edges"]
    # Survivors: same compiled card, the live note untouched, rollback to the later checkpoint still symmetric.
    # 幸存对象：编译卡片相同、存活笔记未变、回滚到后来的检查点仍然对称。
    assert h.compile(S4_QUERY, scope4, mode="deep", as_of=S4_AS_OF).to_text() == card_before
    assert store.get_atom(live.atom_id).archived_reason is None
    rb = h.rollback(cp)
    assert rb["restored"] == [] and rb["archived"] == [] and store.get_atom(live.atom_id).archived_reason is None
    again = purge(store, h.index, scope=scope, older_than_days=90, dry_run=False)
    assert again["examined"] == 0 and again["purged"] == 0 and again["anchor_id"] is None
    # The gate closed again: a raw DELETE on a state table is still an invariant violation.
    # 门重新关上：对状态表的裸 DELETE 仍然违反不变量。
    assert store.get_meta("purge_in_progress") is None
    with pytest.raises(StoreError) as exc:
        store._execute("DELETE FROM memory_atoms WHERE atom_id = ?", (live.atom_id,))
    assert exc.value.error_code == "STORE_003"


def test_prune_runs_retention_only_when_both_switches_are_on(store, clock, events, scope4, index_provider):
    """prune reports purged=0 and keeps tombstones by default; retention.enabled + run_in_prune purge them.
    prune 默认报告 purged=0 并保留墓碑；retention.enabled + run_in_prune 时才清理它们。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the prune report key 'purged' and on row presence under both configurations.
        对 prune 报告键 'purged' 与两种配置下行是否存在的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    _build_s4(h, events, scope4)
    scope = Scope(**scope4)
    gone = _seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text="Old deleted note.")
    _tombstone(h, gone)
    clock.tick(days=60)
    default = h.consolidate(scope4, phases=["prune"])["phases"]["prune"]
    assert default["purged"] == 0 and store.get_atom(gone.atom_id) is not None
    enabled_only = _Harness(store, clock, index_provider, retention={"enabled": True, "older_than_days": 30})
    assert enabled_only.consolidate(scope4, phases=["prune"])["phases"]["prune"]["purged"] == 0
    assert store.get_atom(gone.atom_id) is not None
    both = _Harness(store, clock, index_provider, retention={"enabled": True, "run_in_prune": True, "older_than_days": 30})  # fmt: skip
    rep = both.consolidate(scope4, phases=["prune"])["phases"]["prune"]
    assert rep["purged"] == 1 and store.get_atom(gone.atom_id) is None
    assert [r["event"] for r in store.get_history(gone.memory_id)][-1] == "PURGE"
    assert both.consolidate(scope4, phases=["prune"])["phases"]["prune"]["purged"] == 0


def test_vacuum_shrinks_file_after_purge(tmp_path, clock, ids, events, scope4):
    """After purging bulky tombstones, StateStore.vacuum() shrinks the file; inside a transaction it is STORE_001.
    清理体积大的墓碑之后，StateStore.vacuum() 缩小文件；在事务内调用则 STORE_001。

    Input / 输入：
        tmp_path: Scratch directory for the file database.
        tmp_path：文件库所用的临时目录。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        ids: SequentialIds fixture.
        ids：SequentialIds 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions on before/after byte counts, the on-disk size and the transaction guard.
        对前后字节数、磁盘大小与事务守卫的断言。

    State effect / 状态影响：
        Creates a SQLite file under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    path = str(tmp_path / "retention.db")
    file_store = StateStore(path, now=clock.now, id_factory=ids)
    h = _Harness(file_store, clock, "sqlite")
    scope = Scope(**scope4)
    bulky = "x" * 4000
    tombs = [_seed_state(h, scope=scope, ts="2026-06-15T11:00:00Z", text=f"{bulky} {i}") for i in range(40)]
    for atom in tombs:
        _tombstone(h, atom)
    clock.tick(days=100)
    rep = purge(file_store, h.index, scope=scope, older_than_days=90, dry_run=False)
    assert rep["purged"] == 40 and rep["estimated_bytes"] > 40 * 4000
    with file_store.transaction():
        with pytest.raises(StoreError) as exc:
            file_store.vacuum()
        assert exc.value.error_code == "STORE_001"
    disk_before = os.path.getsize(path)
    out = file_store.vacuum()
    assert out["path"] == path and out["after_bytes"] < out["before_bytes"]
    assert out["freed_bytes"] == out["before_bytes"] - out["after_bytes"] and out["freed_bytes"] >= 40 * 4000
    assert os.path.getsize(path) < disk_before and os.path.getsize(path) == out["after_bytes"]
    assert file_store.stats()["memory_atoms"] == 0 and file_store.stats()["history"] == 120
    file_store.close()


def test_feedback_successor_keeps_shared_observation_live(store, clock, index_provider):
    """A feedback successor shares its predecessor's observation: it stays open for v2 and purge protects v1 (`source_shared`).
    反馈后继共享前驱的观察：它为 v2 保持打开，且清理保护 v1（`source_shared`）。

    The protection does not depend on the revision edge.
    该保护不依赖 revision 边。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions: shared source, observation `valid_to` None after the correction, `source_shared` in the purge
        report, the observation row intact after purge(dry_run=False), and `shared_source_atom_ids` semantics.
        断言：来源共享、纠正后观察 `valid_to` 为 None、清理报告含 `source_shared`、purge(dry_run=False) 后观察行完好，
        以及 `shared_source_atom_ids` 的语义。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    from hms_elite.memory.main import HMS

    hms = HMS(
        HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}, retention={"enabled": True}),
        store=store,
    )
    try:
        hms.add("I like black coffee in the morning.", user_id="alice", timestamp="2026-03-01T09:00:00Z", infer=True)
        rows = hms.get_all(filters={"user_id": "alice"}, top_k=10)["results"]
        memory_id = next(r["id"] for r in rows if "coffee" in r["memory"])
        v1 = store.resolve_head(memory_id)
        assert str(v1.source_kind) == "observation"
        hms.feedback(memory_id, "state_wrong", correction="I like green tea in the morning.")
        v2 = store.resolve_head(memory_id)
        assert v2.atom_id != v1.atom_id and (str(v2.source_kind), v2.source_id) == (str(v1.source_kind), v1.source_id)
        assert store.get_atom(v1.atom_id).archived_reason == "feedback"
        # The cascade of archive_atom(v1) must not end the observation the live head still cites.
        # archive_atom(v1) 的级联不得结束存活头版本仍在引用的观察。
        assert store.get_observation(v2.source_id).valid_to is None
        assert store.shared_source_atom_ids([v1.atom_id]) == {v1.atom_id}
        assert store.shared_source_atom_ids([v1.atom_id, v2.atom_id]) == set()
        clock.tick(1)
        report = hms.purge(filters={"user_id": "alice"}, older_than_days=0, dry_run=False)
        assert v1.atom_id in report["protected"]["source_shared"] and report["purged"] == 0
        assert store.get_atom(v1.atom_id) is not None
        assert store.get_observation(v2.source_id) is not None and store.get_atom(v2.atom_id).archived_reason is None
        # A plain delete of the head ends the observation: nothing live cites it any more.
        # 直接删除头版本会结束观察：再没有存活对象引用它。
        hms.delete(memory_id)
        assert store.get_observation(v2.source_id).valid_to is not None
    finally:
        store.attr_listeners = []
        hms.index.close()
