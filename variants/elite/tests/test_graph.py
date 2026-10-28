"""Group E tests: cohesion, coarsen/project/refine round trip, InterfaceEP contract, skills, explain chain, refine callers, L3.
E 组测试：内聚度、coarsen/project/refine 往返、InterfaceEP 契约、技能、解释链、refine 调用方、L3。
"""

# Purpose (EN): Assert CONTRACT §14 (and §18.3 tests 40–45b) over the Support Graph plane on hand-built and S4 graphs.
# 目的（中）：在手工构造图与 S4 图上断言契约 §14（以及 §18.3 测试 40–45b）。

# Input (EN): tests/fixtures/events/structured_billing.json, tests/fixtures/golden/why_chain.txt and the conftest fixtures.
# 输入（中）：tests/fixtures/events/structured_billing.json、tests/fixtures/golden/why_chain.txt 与 conftest 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import importlib.util
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pydantic
import pytest

from hms_elite.configs.base import GraphConfig, HMSConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.exceptions import ValidationError, WitnessMissingError
from hms_elite.graph import (
    SupportGraph,
    coarsen,
    explain,
    project,
    promote_skills,
    refine,
    render_chain,
    resolve_target,
    round_trip_check,
)
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.ingest.atoms import atom_from_ep, create_atom
from hms_elite.ingest.events import events_from_dicts, system_event
from hms_elite.ingest.pipeline import Ingestor, IngestResult
from hms_elite.llms.rule import RuleLLM
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import (
    EP,
    GraphEdge,
    InterfaceEP,
    Invariant,
    Kappa,
    MemoryAtom,
    Residual,
    Scope,
    Slots,
    SourceEvent,
)
from hms_elite.utils.text import ordered_union, topic_tags
from tests.conftest import FrozenClock

ROOT = pathlib.Path(__file__).resolve().parents[1]
S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP1_Y = "billing_e2e failed: subscription_state_regression"
S4_EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
S4_DC = f"Do not re-apply '{S4_EP1_A}': {S4_EP1_Y}"


@dataclass
class _Rig:
    """Everything a test needs from an ingested S4 graph.
    测试从已摄取的 S4 图中所需的一切。
    """

    store: StateStore
    clock: FrozenClock
    index: Any
    embedder: Any
    graph: SupportGraph
    scope: Scope
    uc: MemoryAtom
    dc: MemoryAtom
    ep1: EP
    ep2: EP
    anc: dict[str, str] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], GraphEdge] = field(default_factory=dict)

    @property
    def support(self) -> list[str]:
        """The S4 support set [ep1_atom, ep2_atom].
        S4 支持集 [ep1_atom, ep2_atom]。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The two EP node ids.
            两个 EP 节点 id。

        State effect / 状态影响：
            None.
            无。
        """
        return [self.ep1.atom_id, self.ep2.atom_id]


def _anchor_of(res: IngestResult, object_type: str) -> str:
    """The anchor id of the first event of a type in an IngestResult.
    IngestResult 中某类型首个事件的锚点 id。
    """
    return [e["anchor_id"] for e in res.events if e["object_type"] == object_type][0]


def _index_for(store: StateStore, index_provider: str, dims: int = 64) -> Any:
    """An index of the parametrized provider.
    参数化提供者对应的索引。
    """
    if index_provider == "sqlite":
        return SQLiteIndex(store=store, dims=dims)
    return MemoryIndex(dims=dims)


def _build_s4(
    store: StateStore,
    clock: FrozenClock,
    events: Any,
    scope_dict: dict,
    index_provider: str,
    *,
    rig: _Rig | None = None,
) -> _Rig:
    """Ingest the §19.3 S4 fixture (constraint via the rule LLM, structured events with zero LLM calls) into `store`.
    将 §19.3 的 S4 夹具摄取进 `store`（约束经规则 LLM，结构化事件零 LLM 调用）。
    """
    embedder = rig.embedder if rig is not None else HashEmbedder({"embedding_dims": 64})
    index = rig.index if rig is not None else _index_for(store, index_provider)
    config = HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"})
    scope = Scope(**scope_dict)
    fixture = events("structured_billing")
    clock.tick(seconds=60)
    r0 = Ingestor(store, index, embedder, RuleLLM(), config).ingest_messages(
        coerce_messages(fixture.constraint["content"]),
        scope=scope,
        user_meta={},
        infer=True,
        memory_type=None,
        prompt=None,
        event_time=fixture.constraint["timestamp"],
        valid_to=None,
    )
    uc = store.get_atom(r0.rows[0]["atom_id"])
    scripted = ScriptedLLM({"responses": []})
    ingestor = Ingestor(store, index, embedder, scripted, config)
    clock.tick(seconds=60)
    ra = ingestor.ingest_events(
        events_from_dicts(
            fixture.events_fail, scope_fallback=scope, visibility_default="private", event_time=clock.now()
        ),
        infer=True,
    )
    clock.tick(seconds=60)
    rb = ingestor.ingest_events(
        events_from_dicts(
            fixture.events_fix, scope_fallback=scope, visibility_default="private", event_time=clock.now()
        ),
        infer=True,
    )
    assert scripted.calls == []
    anc = {
        "uc": uc.anchor_ids[0],
        "e1": _anchor_of(ra, "file_diff"),
        "e2": _anchor_of(ra, "test_log"),
        "e3": _anchor_of(rb, "file_diff"),
        "e4": _anchor_of(rb, "test_log"),
    }
    return _Rig(
        store=store,
        clock=clock,
        index=index,
        embedder=embedder,
        graph=SupportGraph(store, GraphConfig()),
        scope=scope,
        uc=uc,
        dc=store.get_atom(ra.constraint_ids[0]),
        ep1=store.get_ep(ra.ep_ids[0]),
        ep2=store.get_ep(rb.ep_ids[0]),
        anc=anc,
        edges={(e.edge_type, e.src_id, e.dst_id): e for e in store.list_edges(scope)},
    )


def _seed_event(store: StateStore, text: str, ts: str, scope: Scope) -> tuple[SourceEvent, str]:
    """Insert one message event and its anchor; return (event, anchor_id).
    插入一条消息事件及其锚点；返回 (event, anchor_id)。
    """
    ev = SourceEvent(scope=scope, role="user", payload={"text": text}, timestamp=ts)
    stored, _dedup = store.insert_source_event(ev)
    return stored, store.get_or_create_anchor(stored).anchor_id


def _seed_ep(
    store: StateStore,
    index: Any,
    embedder: Any,
    *,
    scope: Scope,
    ts: str,
    u: str,
    a: str,
    y: str,
    r: str,
    constraints: list[str] = (),
) -> EP:
    """Insert an EP (event + anchor + eps row + L1 episode atom/node) for hand-built graphs.
    为手工构造图插入一个 EP（事件 + 锚点 + eps 行 + L1 情节原子/节点）。
    """
    _ev, anchor = _seed_event(store, f"{a} -> {y or '?'}", ts, scope)
    ep = EP(
        slots=Slots(u=u, a=a, y=y, r=r, kappa=Kappa(constraints=list(constraints), evidence=[anchor])),
        time_anchor=ts,
        topic_tags=topic_tags(f"{u} {a} {y}"),
        confidence=0.8,
        scope=scope,
        anchor_ids=[anchor],
    )
    with store.transaction():
        ep = store.insert_ep(ep)
        atom = create_atom(store, index, embedder, atom_from_ep(ep, scope=scope))
        store.set_ep_atom(ep.ep_id, atom.atom_id)
    return store.get_ep(ep.ep_id)


def _seed_state(store: StateStore, index: Any, embedder: Any, *, scope: Scope, ts: str, text: str) -> MemoryAtom:
    """Insert a constraint-like state atom (+node) witnessed by a fresh message anchor.
    插入一个由新消息锚点见证的约束型 state 原子（+节点）。
    """
    _ev, anchor = _seed_event(store, text, ts, scope)
    atom = MemoryAtom(
        atom_type="state",
        content={"text": text},
        text=text,
        time_anchor=ts,
        scale_level="state",
        subject="project",
        predicate="must_not",
        obs_type="constraint",
        source_kind="observation",
        source_id="obs_hand",
        extracted_by="test",
        scope=scope,
        anchor_ids=[anchor],
    )
    return create_atom(store, index, embedder, atom)


def _sched_anchor(store: StateStore, scope: Scope) -> str:
    """A scheduler system event + anchor used to witness skill history rows.
    用于见证技能历史行的调度器系统事件 + 锚点。
    """
    _ev, anc = system_event(
        store, scope=scope, object_type="scheduler", event_type="scheduler", payload={"phases": ["consolidate"]}
    )
    return anc.anchor_id


def _coarsen_s4(rig: _Rig, *, policy: str = "rule", llm: Any = None) -> InterfaceEP:
    """coarsen(S4 support) with the rig's collaborators.
    用 rig 的协作对象执行 coarsen(S4 支持集)。
    """
    iep = coarsen(
        rig.store,
        rig.index,
        rig.embedder,
        rig.graph,
        rig.support,
        target_scale="L2_meso",
        policy=policy,
        llm=llm,
        at=rig.store.now(),
        scope=rig.scope,
        min_cohesion=rig.graph.config.min_cohesion,
    )
    assert iep is not None
    return iep


# ---------------------------------------------------------------------------
# 40
# ---------------------------------------------------------------------------
def test_cohesion_formula_values(store, clock, events, scope4, index_provider):
    """Chain of 3 with no boundary → 1.0; isolated node → 0.0; S4 {EP1, EP2} → 2/3 (unrounded); components find the chain.
    无边界的三节点链 → 1.0；孤立节点 → 0.0；S4 {EP1, EP2} → 2/3（不取整）；components 找到该链。

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
        Assertions on SupportGraph.cohesion / internal_edges / boundary / components.
        对 SupportGraph.cohesion / internal_edges / boundary / components 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    embedder = HashEmbedder({"embedding_dims": 64})
    index = _index_for(store, index_provider)
    graph = SupportGraph(store, GraphConfig())
    scope = Scope(user_id="hand", run_id="chain")
    eps = [
        _seed_ep(store, index, embedder, scope=scope, ts=f"2026-05-01T1{i}:00:00Z", u=f"retry step {i}", a=f"patch retry step {i}", y="tests passed", r="positive")
        for i in range(3)
    ]  # fmt: skip
    ids = [ep.atom_id for ep in eps]
    for a, b in zip(ids, ids[1:]):
        graph.insert_edge(
            src_id=a,
            dst_id=b,
            edge_type="causal",
            witness_anchor_ids=store.get_ep(eps[1].ep_id).anchor_ids,
            created_by="ingest",
        )
    assert [e.edge_type for e in graph.internal_edges(ids)] == ["causal", "causal"] and graph.boundary(ids) == []
    assert graph.cohesion(ids) == pytest.approx(1.0)
    lone = _seed_ep(
        store,
        index,
        embedder,
        scope=scope,
        ts="2026-05-02T10:00:00Z",
        u="alone",
        a="patch nothing",
        y="unknown outcome",
        r="unknown",
    )
    assert graph.cohesion([lone.atom_id]) == 0.0 and graph.cohesion([]) == 0.0
    # 按 time_anchor 排序；孤立的那条 EP 低于 min_size
    assert graph.components(scope) == [ids]  # sorted by time_anchor; the lone EP is below min_size
    # 相邻两点均相隔 1 小时 → 单点分量被丢弃
    assert graph.components(scope, gap_hours=0.5) == []  # every consecutive pair is 1h apart → singletons dropped
    with pytest.raises(WitnessMissingError):
        graph.insert_edge(src_id=ids[0], dst_id=ids[2], edge_type="use", witness_anchor_ids=[], created_by="ingest")
    with pytest.raises(ValidationError):
        graph.insert_edge(
            src_id="atm_nope",
            dst_id=ids[2],
            edge_type="use",
            witness_anchor_ids=[eps[0].anchor_ids[0]],
            created_by="ingest",
        )

    rig = _build_s4(store, clock, events, scope4, index_provider)
    e_in = rig.graph.internal_edges(rig.support)
    e_b = rig.graph.boundary(rig.support)
    assert {e.edge_type for e in e_in} == {"temporal", "causal"} and len(e_b) == 4
    assert {(b.direction, b.edge_type) for b in e_b} == {("out", "use"), ("out", "distill")}
    assert rig.graph.cohesion(rig.support) == pytest.approx(2 / 3, abs=1e-6)
    assert rig.graph.components(rig.scope) == [rig.support]


# ---------------------------------------------------------------------------
# hand-built round trip (the law on a graph with in- and out-boundary edges)
# 手工构造的往返（在同时含入向与出向边界边的图上验证该定律）
# ---------------------------------------------------------------------------
def test_round_trip_law_hand_built(store, index_provider):
    """refine(project(coarsen(S)))['boundary_edges'] == E_b(S) on a hand-built chain with an incoming and two outgoing boundary edges.
    在带一条入向、两条出向边界边的手工链上验证 refine(project(coarsen(S)))['boundary_edges'] == E_b(S)。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on coarsen, project (projection rows per high edge), refine, round_trip_check and neighbors.
        对 coarsen、project（每条高层边的投影行）、refine、round_trip_check 与 neighbors 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    embedder = HashEmbedder({"embedding_dims": 64})
    index = _index_for(store, index_provider)
    graph = SupportGraph(store, GraphConfig())
    scope = Scope(user_id="hand", run_id="law")
    e1, e2, e3 = [
        _seed_ep(store, index, embedder, scope=scope, ts=f"2026-05-01T1{i}:00:00Z", u=f"retry step {i}", a=f"patch retry step {i}", y="deploy failed" if i == 1 else "tests passed", r="negative" if i == 1 else "positive", constraints=["Never deploy on Friday."])
        for i in range(3)
    ]  # fmt: skip
    outsider = _seed_ep(
        store,
        index,
        embedder,
        scope=scope,
        ts="2026-05-01T09:00:00Z",
        u="setup retry",
        a="patch bootstrap",
        y="tests passed",
        r="positive",
    )
    c1 = _seed_state(store, index, embedder, scope=scope, ts="2026-05-01T08:00:00Z", text="Never deploy on Friday.")
    c2 = _seed_state(store, index, embedder, scope=scope, ts="2026-05-01T08:30:00Z", text="Keep retries idempotent.")
    w = {ep.atom_id: ep.anchor_ids for ep in (e1, e2, e3, outsider)}
    ids = [e1.atom_id, e2.atom_id, e3.atom_id]
    causal_a = graph.insert_edge(
        src_id=e1.atom_id, dst_id=e2.atom_id, edge_type="causal", witness_anchor_ids=w[e2.atom_id], created_by="ingest"
    )
    causal_b = graph.insert_edge(
        src_id=e2.atom_id, dst_id=e3.atom_id, edge_type="causal", witness_anchor_ids=w[e3.atom_id], created_by="ingest"
    )
    use_1 = graph.insert_edge(
        src_id=e1.atom_id,
        dst_id=c1.atom_id,
        edge_type="use",
        witness_anchor_ids=w[e1.atom_id],
        created_by="ingest",
        confidence=0.8,
    )
    use_3 = graph.insert_edge(
        src_id=e3.atom_id,
        dst_id=c2.atom_id,
        edge_type="use",
        witness_anchor_ids=w[e3.atom_id],
        created_by="ingest",
        confidence=0.6,
    )
    use_2 = graph.insert_edge(
        src_id=e2.atom_id,
        dst_id=c1.atom_id,
        edge_type="use",
        witness_anchor_ids=w[e2.atom_id],
        created_by="ingest",
        confidence=0.7,
    )
    inbound = graph.insert_edge(
        src_id=outsider.atom_id,
        dst_id=e2.atom_id,
        edge_type="causal",
        witness_anchor_ids=w[e2.atom_id],
        created_by="ingest",
    )
    e_b = graph.boundary(ids)
    assert {b.edge_id for b in e_b} == {use_1, use_2, use_3, inbound}
    assert {b.edge_id: b.direction for b in e_b}[inbound] == "in"
    assert graph.cohesion(ids) == pytest.approx(0.5 * min(1.0, 2 / 2) + 0.5 * 2 / 6)

    iep = coarsen(store, index, embedder, graph, ids, llm=None, at=store.now(), scope=scope, min_cohesion=0.3)
    assert iep is not None and iep.support_node_ids == ids and set(iep.support_edge_ids) == {causal_a, causal_b}
    assert {b.edge_id for b in iep.boundary_edges} == {use_1, use_2, use_3, inbound}
    assert iep.slots.u == "retry step 0" and iep.slots.r == "positive" and iep.slots.b == "Never deploy on Friday."
    assert [inv.kind for inv in iep.invariants] == ["constraint", "failed_path"]
    assert iep.invariants[0].support_node_ids == ids and iep.invariants[1].support_node_ids == [e2.atom_id]
    assert iep.evidence_anchor_ids == sorted({a for ep in (e1, e2, e3) for a in ep.anchor_ids})
    node = store.get_node(iep.atom_id)
    assert node.node_kind == "interface_ep" and node.ref_id == iep.interface_ep_id and node.scale == "L2_meso"

    high = project(graph, iep, at=store.now())
    # (c1, use) 合并了 use_1 + use_2；(c2, use)；(outsider → iep, causal)
    assert len(high) == 3  # (c1, use) merges use_1 + use_2; (c2, use); (outsider → iep, causal)
    projections = {h: store.projections_of(h) for h in high}
    assert all(projections[h] for h in high) and sorted(sum(projections.values(), [])) == sorted(
        [use_1, use_2, use_3, inbound]
    )
    merged = store.find_edge(iep.atom_id, c1.atom_id, "use", "L2_meso")
    assert merged is not None and merged.is_boundary and set(store.projections_of(merged.edge_id)) == {use_1, use_2}
    assert merged.witness_edge_id == use_1 and merged.mentions == 2 and merged.confidence == 0.8
    incoming = store.find_edge(outsider.atom_id, iep.atom_id, "causal", "L2_meso")
    assert incoming is not None and incoming.created_by == "project" and incoming.proposer == "project-v1"
    for e in store.list_edges(scope, scale="L2_meso"):
        assert e.is_boundary and store.projections_of(e.edge_id)

    opened = refine(store, graph, iep.interface_ep_id)
    assert [n["node_id"] for n in opened["nodes"]] == ids and {e["edge_id"] for e in opened["edges"]} == {
        causal_a,
        causal_b,
    }
    assert {b["edge_id"] for b in opened["boundary_edges"]} == {b.edge_id for b in graph.boundary(ids)}
    assert round_trip_check(graph, iep)
    assert set(opened["projections"]) == set(high) and {a["anchor_id"] for a in opened["anchors"]} >= set(
        iep.evidence_anchor_ids
    )
    wider = refine(store, graph, iep.interface_ep_id, hops=1)
    assert {n["node_id"] for n in wider["nodes"]} == set(ids) | {c1.atom_id, c2.atom_id, outsider.atom_id}
    # 幂等
    assert project(graph, iep, at=store.now()) == high  # idempotent
    with pytest.raises(ValueError):
        refine(store, graph, "iep_nope")

    nodes, edges, paths = graph.neighbors([e2.atom_id], hops=1, direction="in")
    assert [n.node_id for n in nodes][0] == e2.atom_id and {n.node_id for n in nodes} == {
        e2.atom_id,
        e1.atom_id,
        outsider.atom_id,
    }
    assert (
        paths[e1.atom_id] == [causal_a]
        and paths[outsider.atom_id] == [inbound]
        and {e.edge_id for e in edges} == {causal_a, inbound}
    )
    nodes, _edges, _paths = graph.neighbors([e2.atom_id], hops=2, max_nodes=2)
    assert len(nodes) == 2


# ---------------------------------------------------------------------------
# 41
# ---------------------------------------------------------------------------
def test_coarsen_project_refine_roundtrip(store, clock, events, scope4, index_provider):
    """S4: cohesion 2/3, §11.5 slots, completed invariants, IEP atom axes, 3 projected edges / 4 projection rows, round trip, idempotent re-coarsen.
    S4：内聚度 2/3、§11.5 槽位、补全的不变量、IEP 原子轴、3 条投影边 / 4 行投影、往返定律、再粗化幂等。

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
        Assertions on coarsen / project / refine / round_trip_check over the S4 graph.
        对 S4 图上 coarsen / project / refine / round_trip_check 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    ep1_atom, ep2_atom = rig.support
    iep = _coarsen_s4(rig)
    assert iep.scale == "L2_meso" and iep.cohesion_score == pytest.approx(2 / 3, abs=1e-4)
    assert iep.support_node_ids == [ep1_atom, ep2_atom] and iep.coarsen_op_version == "rule-coarsen-v1"
    slots = iep.slots
    assert slots.u == "fix checkout webhook retry" and slots.a == f"{S4_EP1_A} → {S4_EP2_A}"
    assert slots.y == "billing_e2e passed" and slots.r == "positive"
    assert slots.b == f"{S4_CONSTRAINT} ; {S4_DC}" and slots.b.count(S4_CONSTRAINT) == 1 and slots.b.count(S4_DC) == 1
    assert slots.kappa.constraints == [S4_CONSTRAINT, S4_DC] and slots.kappa.visibility == "private"
    assert slots.kappa.evidence == iep.evidence_anchor_ids
    evidence = sorted({rig.anc["e1"], rig.anc["e2"], rig.anc["e3"], rig.anc["e4"]})
    witnesses = {a for e in rig.graph.internal_edges(rig.support) for a in e.witness_anchor_ids}
    assert iep.evidence_anchor_ids == evidence == sorted(set(rig.ep1.anchor_ids) | set(rig.ep2.anchor_ids) | witnesses)
    for name in ("u", "a", "y", "r", "b", "kappa"):
        assert iep.slot_support[name] and set(iep.slot_support[name]) <= set(iep.support_node_ids)
    assert (
        iep.slot_support["u"] == [ep1_atom]
        and iep.slot_support["y"] == [ep2_atom]
        and iep.slot_support["a"] == rig.support
    )
    assert [inv.kind for inv in iep.invariants] == ["constraint", "failed_path"]
    constraint_inv, failed = iep.invariants
    assert constraint_inv.text == S4_CONSTRAINT and constraint_inv.support_node_ids == [ep1_atom, ep2_atom]
    assert constraint_inv.anchor_ids == evidence and constraint_inv.confidence == 0.8
    assert failed.text == f"do not: {S4_EP1_A}" and failed.support_node_ids == [ep1_atom]
    assert failed.anchor_ids == sorted({rig.anc["e1"], rig.anc["e2"]}) and failed.confidence == 0.8
    assert iep.residuals == []
    assert set(iep.support_edge_ids) == {
        rig.edges[("temporal", ep1_atom, ep2_atom)].edge_id,
        rig.edges[("causal", ep1_atom, ep2_atom)].edge_id,
    }

    atom = store.get_atom(iep.atom_id)
    assert (
        atom.source_kind == "interface_ep" and atom.scale_level == "L2_meso" and atom.source_id == iep.interface_ep_id
    )
    assert (
        atom.text == slots.gloss()
        and atom.anchor_ids == evidence
        and atom.source_ep_ids == [rig.ep1.ep_id, rig.ep2.ep_id]
    )
    ep1_atom_row, ep2_atom_row = store.get_atoms(rig.support)
    assert atom.topic_tags == ordered_union(ep1_atom_row.topic_tags, ep2_atom_row.topic_tags)
    assert {"retry.ts", "idempotency.ts", "billing_e2e"} <= set(atom.entity_ids)
    assert (
        atom.time_anchor == "2026-06-15T10:14:00.000000+00:00"
        and atom.content["interface_ep_id"] == iep.interface_ep_id
    )
    assert store.support_members(iep.interface_ep_id)["node"] == rig.support

    high = project(rig.graph, iep, at=store.now())
    assert len(high) == 3
    rows = {h: store.projections_of(h) for h in high}
    assert all(rows.values()) and sum(len(v) for v in rows.values()) == 4
    use_uc = store.find_edge(iep.atom_id, rig.uc.atom_id, "use", "L2_meso")
    use_dc = store.find_edge(iep.atom_id, rig.dc.atom_id, "use", "L2_meso")
    distill_dc = store.find_edge(iep.atom_id, rig.dc.atom_id, "distill", "L2_meso")
    assert use_uc and use_dc and distill_dc and {use_uc.edge_id, use_dc.edge_id, distill_dc.edge_id} == set(high)
    low_use_1 = rig.edges[("use", ep1_atom, rig.uc.atom_id)].edge_id
    low_use_2 = rig.edges[("use", ep2_atom, rig.uc.atom_id)].edge_id
    assert store.projections_of(use_uc.edge_id) == [low_use_1, low_use_2] and use_uc.witness_edge_id == low_use_1
    assert store.projections_of(distill_dc.edge_id) == [rig.edges[("distill", ep1_atom, rig.dc.atom_id)].edge_id]
    assert store.projections_of(use_dc.edge_id) == [rig.edges[("use", ep2_atom, rig.dc.atom_id)].edge_id]
    assert store.projections_from(low_use_1) == [use_uc.edge_id]
    for e in (use_uc, use_dc, distill_dc):
        assert e.is_boundary and e.created_by == "project" and e.proposer == "project-v1" and e.scale == "L2_meso"
    assert use_uc.witness_anchor_ids == [rig.anc["e1"], rig.anc["e3"]] and use_uc.confidence == 0.8

    opened = refine(store, rig.graph, iep.interface_ep_id)
    assert {b["edge_id"] for b in opened["boundary_edges"]} == {b.edge_id for b in rig.graph.boundary(rig.support)}
    assert round_trip_check(rig.graph, iep)
    assert [n["node_id"] for n in opened["nodes"]] == rig.support and len(opened["edges"]) == 2
    assert opened["interface_ep"]["interface_ep_id"] == iep.interface_ep_id and set(opened["projections"]) == set(high)

    again = _coarsen_s4(rig)
    assert again.interface_ep_id == iep.interface_ep_id and again.support_node_ids == iep.support_node_ids
    assert len(store.list_interface_eps(rig.scope, valid_only=False)) == 1
    reversed_order = coarsen(
        store,
        rig.index,
        rig.embedder,
        rig.graph,
        [ep2_atom, ep1_atom],
        llm=None,
        at=store.now(),
        scope=rig.scope,
        min_cohesion=0.3,
    )
    # support_hash 基于排序后的 id 计算
    assert reversed_order.interface_ep_id == iep.interface_ep_id  # support_hash over sorted ids
    # 已存在者胜出
    assert (
        coarsen(
            store,
            rig.index,
            rig.embedder,
            rig.graph,
            rig.support,
            llm=None,
            at=store.now(),
            scope=rig.scope,
            min_cohesion=0.9,
        )
        is not None
    )  # existing wins / 已存在者优先
    with pytest.raises(ValidationError):
        coarsen(
            store,
            rig.index,
            rig.embedder,
            rig.graph,
            [ep1_atom],
            llm=None,
            at=store.now(),
            scope=rig.scope,
            min_cohesion=0.3,
        )
    with pytest.raises(ValidationError):
        coarsen(
            store,
            rig.index,
            rig.embedder,
            rig.graph,
            [ep1_atom, rig.uc.atom_id],
            llm=None,
            at=store.now(),
            scope=rig.scope,
            min_cohesion=0.3,
        )


# ---------------------------------------------------------------------------
# 42
# ---------------------------------------------------------------------------
def test_interface_ep_is_not_a_summary(store, clock, events, scope4, index_provider):
    """InterfaceEP needs ≥2 support nodes and evidence; every slot cites support; Slots is shared; Invariant needs anchors; Residual anchors default to [].
    InterfaceEP 需要 ≥2 支持节点与证据；每个槽位引用支持；Slots 共用；Invariant 需要锚点；Residual 锚点默认为 []。

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
        Assertions on the InterfaceEP / Invariant / Residual / Slots models and the persisted S4 InterfaceEP.
        对 InterfaceEP / Invariant / Residual / Slots 模型与已持久化 S4 InterfaceEP 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scope = Scope(**scope4)
    slots = Slots(u="fix retry", a="patch retry.ts", y="tests passed", r="positive")
    good = dict(
        slots=slots,
        slot_support={"u": ["n0"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"]},
        support_node_ids=["n0", "n1"],
        evidence_anchor_ids=["anc_1"],
        scope=scope,
    )
    assert InterfaceEP(**good).support_node_ids == ["n0", "n1"]
    with pytest.raises(pydantic.ValidationError):
        InterfaceEP(**{**good, "support_node_ids": ["n0"]})
    with pytest.raises(pydantic.ValidationError):
        InterfaceEP(**{**good, "evidence_anchor_ids": []})
    with pytest.raises(pydantic.ValidationError, match="cites no support node"):
        InterfaceEP(**{**good, "slot_support": {"u": ["n0"]}})
    with pytest.raises(pydantic.ValidationError, match="outside support_node_ids"):
        InterfaceEP(**{**good, "slot_support": {**good["slot_support"], "a": ["n7"]}})
    assert EP.model_fields["slots"].annotation is Slots and InterfaceEP.model_fields["slots"].annotation is Slots
    with pytest.raises(pydantic.ValidationError):
        Invariant(text="x", support_node_ids=["n0"])
    with pytest.raises(pydantic.ValidationError):
        Invariant(text="x", support_node_ids=["n0"], anchor_ids=[])
    assert Invariant(text="x", support_node_ids=["n0"], anchor_ids=["anc_1"]).confidence == 0.7
    residual = Residual(text="open?", kind="open_question")
    assert residual.anchor_ids == [] and residual.support_node_ids == [] and residual.edge_ids == []
    with pytest.raises(pydantic.ValidationError):
        # 既没有 outcome 也没有 constraint
        Slots(u="u", a="a")  # neither outcome nor constraint

    rig = _build_s4(store, clock, events, scope4, index_provider)
    iep = _coarsen_s4(rig)
    stored = store.get_interface_ep(iep.interface_ep_id)
    for name, value in (("u", stored.slots.u), ("a", stored.slots.a), ("y", stored.slots.y), ("b", stored.slots.b)):
        assert value and stored.slot_support[name] and set(stored.slot_support[name]) <= set(stored.support_node_ids)
    assert stored.slot_support["kappa"] and stored.slot_support["r"]
    assert isinstance(stored.slots, Slots) and isinstance(rig.ep1.slots, Slots)
    assert stored.evidence_anchor_ids and stored.cohesion_score == iep.cohesion_score
    for inv in stored.invariants:
        assert inv.anchor_ids and set(inv.support_node_ids) <= set(stored.support_node_ids)


# ---------------------------------------------------------------------------
# 43
# ---------------------------------------------------------------------------
def test_promote_skills_from_one_interface_ep(store, clock, events, scope4, index_provider):
    """A constraint invariant with ≥2 support nodes becomes an L3_macro skill atom with a traceable distill edge; idempotent on re-run.
    支持节点 ≥2 的约束不变量成为带可追溯 distill 边的 L3_macro 技能原子；重复运行幂等。

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
        Assertions on promote_skills, the skill atom, the distill edge, projections and graph stats.
        对 promote_skills、技能原子、distill 边、投影与图统计的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    iep = _coarsen_s4(rig)
    high = project(rig.graph, iep, at=store.now())
    sched = _sched_anchor(store, rig.scope)
    created = promote_skills(
        store, rig.index, rig.embedder, rig.graph, scope=rig.scope, at=store.now(), anchor_id=sched
    )
    assert len(created) == 1
    skill = store.get_atom(created[0])
    inv = iep.invariants[0]
    assert skill.text == f"Skill: {S4_CONSTRAINT}" and skill.scale_level == "L3_macro" and skill.atom_type == "state"
    assert skill.source_kind == "skill" and skill.source_id == iep.interface_ep_id
    assert skill.anchor_ids == inv.anchor_ids == sorted({rig.anc["e1"], rig.anc["e2"], rig.anc["e3"], rig.anc["e4"]})
    assert skill.confidence == inv.confidence == 0.8 and skill.subject == "project" and skill.predicate == "skill"
    assert skill.topic_tags[0] == "skill" and skill.content["kind"] == "skill" and skill.extracted_by == "rule@skill-v1"
    assert store.get_node(skill.atom_id).node_kind == "skill"
    history = store.get_history(skill.memory_id)
    assert [h["event"] for h in history] == ["ADD"] and history[0]["anchor_id"] == sched
    distill = store.find_edge(iep.atom_id, skill.atom_id, "distill", "L3_macro")
    assert distill is not None and distill.scale == "L3_macro" and distill.is_boundary is False
    assert (
        distill.created_by == "consolidation"
        and distill.confidence == 0.9
        and distill.witness_anchor_ids == inv.anchor_ids
    )
    use_ep1_uc = rig.edges[("use", rig.ep1.atom_id, rig.uc.atom_id)].edge_id
    use_ep2_uc = rig.edges[("use", rig.ep2.atom_id, rig.uc.atom_id)].edge_id
    assert distill.witness_edge_id == use_ep1_uc == min(use_ep1_uc, use_ep2_uc)
    assert store.projections_of(distill.edge_id) == [use_ep1_uc, use_ep2_uc]
    stats = rig.graph.stats(rig.scope)
    assert stats["traceable_high_edges"] == 1.0 and stats["projection_coverage"] == 1.0
    assert stats["nodes"]["skill"] == 1 and stats["edges"]["distill"] == 3 and stats["boundary_edges"] == len(high)

    again = promote_skills(store, rig.index, rig.embedder, rig.graph, scope=rig.scope, at=store.now(), anchor_id=sched)
    assert again == []
    assert (
        store.count_atoms(rig.scope) == store.count_atoms(rig.scope)
        and len(store.list_edges(rig.scope, edge_types=["distill"])) == 3
    )
    assert store.projections_of(distill.edge_id) == [use_ep1_uc, use_ep2_uc]
    assert [a.atom_id for a in store.find_atoms_by_hash(rig.scope, skill.hash)] == [skill.atom_id]


# ---------------------------------------------------------------------------
# 44
# ---------------------------------------------------------------------------
def test_explain_backward_bfs_and_chain(store, clock, events, scope4, index_provider, golden):
    """why() core on the derived constraint: §19.5 nodes/edges/paths/anchors, golden chain, hops=0 → target only, target resolution order.
    在派生约束上运行 why() 核心：§19.5 的节点/边/路径/锚点、黄金链、hops=0 → 仅目标、目标解析顺序。

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

        golden: Golden helper.
        golden：黄金比对辅助。

    Output / 输出：
        Assertions on resolve_target, explain and render_chain.
        对 resolve_target、explain 与 render_chain 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    ep1_atom, ep2_atom = rig.support
    dc, uc = rig.dc, rig.uc
    target = resolve_target(store, dc.atom_id)
    assert target == {"kind": "atom", "id": dc.atom_id, "node_id": dc.atom_id, "text": S4_DC, "resolved_from": "id"}
    assert resolve_target(store, dc.memory_id) == target
    by_ep = resolve_target(store, rig.ep1.ep_id)
    assert by_ep["node_id"] == ep1_atom and by_ep["resolved_from"] == "id" and by_ep["kind"] == "atom"
    assert resolve_target(store, dc.content["obs_id"])["node_id"] == dc.atom_id
    distill_id = rig.edges[("distill", ep1_atom, dc.atom_id)].edge_id
    assert resolve_target(store, distill_id) == {
        "kind": "edge", "id": distill_id, "node_ids": [ep1_atom, dc.atom_id], "text": store.get_edge(distill_id).label or f"distill: {ep1_atom} → {dc.atom_id}", "resolved_from": "id",
    }  # fmt: skip
    assert resolve_target(store, "nonsense-id") is None and resolve_target(store, "") is None

    store.trace.reset()
    trace = explain(store, rig.graph, target)
    assert "graph_edges" in store.trace.tables()
    assert trace["target"] == target
    assert {n["node_id"] for n in trace["nodes"]} == {dc.atom_id, ep1_atom, ep2_atom, uc.atom_id}
    assert {(e["src_id"], e["dst_id"], e["edge_type"]) for e in trace["edges"]} == {
        (ep1_atom, dc.atom_id, "distill"),
        (ep2_atom, dc.atom_id, "use"),
        (ep1_atom, uc.atom_id, "use"),
        (ep1_atom, ep2_atom, "causal"),
        (ep2_atom, uc.atom_id, "use"),
    }
    anchor_ids = {a["anchor_id"] for a in trace["anchors"]}
    assert all(set(e["witness_anchor_ids"]) <= anchor_ids for e in trace["edges"])
    assert anchor_ids == {rig.anc["e1"], rig.anc["e2"], rig.anc["e3"], rig.anc["e4"], rig.anc["uc"]}
    assert trace["unresolved_conflicts"] == [] and trace["refined_from"] == [] and trace["interface_eps"] == []
    use_ep2_dc = rig.edges[("use", ep2_atom, dc.atom_id)].edge_id
    use_ep1_uc = rig.edges[("use", ep1_atom, uc.atom_id)].edge_id
    assert trace["paths"] == {
        dc.atom_id: [],
        ep1_atom: [distill_id],
        ep2_atom: [use_ep2_dc],
        uc.atom_id: [distill_id, use_ep1_uc],
    }
    assert [n["node_id"] for n in trace["nodes"]] == [dc.atom_id, ep1_atom, ep2_atom, uc.atom_id]
    assert trace["nodes"][0]["text"] == S4_DC
    assert trace["chain"] == [
        S4_DC,
        f"<-distill[test://billing_e2e]- [negative] fix checkout webhook retry → {S4_EP1_A} → {S4_EP1_Y}",
        f"-use[file://src/billing/retry.ts]-> {S4_CONSTRAINT}",
    ]
    golden.text("why_chain.txt", "\n".join(trace["chain"]))
    assert trace["explanation"] == "\n".join(trace["chain"])
    edges = store.get_edges([e["edge_id"] for e in trace["edges"]])
    assert render_chain(store, dc.atom_id, edges, hops=2) == trace["chain"]
    assert render_chain(store, dc.atom_id, edges, hops=1) == trace["chain"][:2]
    assert render_chain(store, dc.atom_id, [], hops=2) == [S4_DC]

    only = explain(store, rig.graph, target, hops=0)
    assert (
        [n["node_id"] for n in only["nodes"]] == [dc.atom_id]
        and only["edges"] == []
        and only["paths"] == {dc.atom_id: []}
    )
    assert only["chain"] == [S4_DC] and {a["anchor_id"] for a in only["anchors"]} == set(dc.anchor_ids)
    one = explain(store, rig.graph, target, hops=1)
    assert {n["node_id"] for n in one["nodes"]} == {dc.atom_id, ep1_atom, ep2_atom}
    assert {e["edge_id"] for e in one["edges"]} == {distill_id, use_ep2_dc}
    capped = explain(store, rig.graph, target, max_nodes=2)
    assert len(capped["nodes"]) == 2 and all(
        e["src_id"] in capped["paths"] and e["dst_id"] in capped["paths"] for e in capped["edges"]
    )
    typed = explain(store, rig.graph, target, edge_types=["use"])
    assert {n["node_id"] for n in typed["nodes"]} == {dc.atom_id, ep2_atom, uc.atom_id}

    # An InterfaceEP target is refined on the way: support nodes, internal edges and boundary join the trace.
    # InterfaceEP 目标会在途中被细化：支撑节点、内部边与边界都加入追踪。
    iep = _coarsen_s4(rig)
    project(rig.graph, iep, at=store.now())
    iep_target = resolve_target(store, iep.interface_ep_id)
    assert iep_target == {
        "kind": "interface_ep",
        "id": iep.interface_ep_id,
        "node_id": iep.atom_id,
        "text": iep.slots.gloss(),
        "resolved_from": "id",
    }
    deep = explain(store, rig.graph, iep_target, hops=1)
    assert deep["refined_from"] == [iep.interface_ep_id] and [i["interface_ep_id"] for i in deep["interface_eps"]] == [
        iep.interface_ep_id
    ]
    assert {iep.atom_id, ep1_atom, ep2_atom, uc.atom_id, dc.atom_id} <= {n["node_id"] for n in deep["nodes"]}
    assert deep["chain"][0] == iep.slots.gloss()
    via_uc = explain(store, rig.graph, resolve_target(store, uc.atom_id), hops=1)
    assert iep.interface_ep_id in via_uc["refined_from"] and {ep1_atom, ep2_atom} <= {
        n["node_id"] for n in via_uc["nodes"]
    }


# ---------------------------------------------------------------------------
# 45
# ---------------------------------------------------------------------------
def test_refine_called_only_by_explain_and_compile(store, clock, events, scope4, index_provider):
    """`refine(` appears only in graph/operators.py, graph/explain.py and compiler/compile.py; fast/axis_graph compile modes never refine.
    `refine(` 仅出现在 graph/operators.py、graph/explain.py 与 compiler/compile.py；fast/axis_graph 编译模式从不细化。

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
        Source-inspection assertions, plus compile-mode trace assertions once the facade exists.
        源码检查断言，以及门面存在后的编译模式跟踪断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    allowed = {"hms_elite/graph/operators.py", "hms_elite/graph/explain.py", "hms_elite/compiler/compile.py"}
    callers = {
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "hms_elite").rglob("*.py"))
        if "refine(" in path.read_text(encoding="utf-8")
    }
    assert callers <= allowed, f"unexpected refine( callers: {sorted(callers - allowed)}"
    assert {"hms_elite/graph/operators.py", "hms_elite/graph/explain.py"} <= callers
    rig = _build_s4(store, clock, events, scope4, index_provider)
    iep = _coarsen_s4(rig)
    project(rig.graph, iep, at=store.now())
    store.trace.reset()
    explain(store, rig.graph, resolve_target(store, rig.dc.atom_id))
    # explain 按需细化
    assert {"graph_edges", "interface_eps"} <= store.trace.tables()  # explain refines on demand
    store.trace.reset()
    rig.graph.neighbors(rig.support, hops=1)
    assert not any(t.startswith("interface_ep") for t in store.trace.tables())
    if (
        importlib.util.find_spec("hms_elite.memory.main") is None
        or importlib.util.find_spec("hms_elite.compiler") is None
    ):
        # F/H 两组存在之后才会走到编译模式的追踪
        return  # the compile-mode traces are exercised once groups F/H exist
    from hms_elite.memory.main import HMS

    hms = HMS(HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}), store=store)
    store.trace.reset()
    hms.compile("fix checkout webhook retry", filters=scope4, mode="fast")
    assert not any(t.startswith(("graph_", "interface_ep")) for t in store.trace.tables())
    store.trace.reset()
    hms.compile("fix checkout webhook retry", filters=scope4, mode="axis_graph")
    touched = store.trace.tables()
    assert any(t.startswith("graph_") for t in touched) and not any(t.startswith("interface_ep") for t in touched)


# ---------------------------------------------------------------------------
# 45b
# ---------------------------------------------------------------------------
def test_coarsen_l3_from_two_interface_eps(store, clock, events, index_provider):
    """Two L2 InterfaceEPs (two S4-like runs joined by a projected causal edge) coarsen into one L3 InterfaceEP whose refine returns exactly them.
    两个 L2 InterfaceEP（由投影 causal 边相连的两个 S4 式运行）粗化为一个 L3 InterfaceEP，其 refine 恰好返回这两个节点。

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
        Assertions on components(kinds=('interface_ep',)), coarsen(target_scale='L3_macro'), project and refine at L3.
        对 components(kinds=('interface_ep',))、coarsen(target_scale='L3_macro')、L3 的 project 与 refine 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scope_a = {"user_id": "alice", "agent_id": "coder", "run_id": "run-a"}
    scope_b = {"user_id": "alice", "agent_id": "coder", "run_id": "run-b"}
    rig_a = _build_s4(store, clock, events, scope_a, index_provider)
    rig_b = _build_s4(store, clock, events, scope_b, index_provider, rig=rig_a)
    graph = rig_a.graph
    joint = Scope(user_id="alice", agent_id="coder")
    link = graph.insert_edge(
        src_id=rig_a.ep2.atom_id, dst_id=rig_b.ep1.atom_id, edge_type="causal", witness_anchor_ids=[rig_b.anc["e1"]], created_by="ingest", proposer="closure"
    )  # fmt: skip
    iep_a = _coarsen_s4(rig_a)
    project(graph, iep_a, at=store.now())
    iep_b = _coarsen_s4(rig_b)
    project(graph, iep_b, at=store.now())
    assert {i.interface_ep_id for i in store.list_interface_eps(joint, scale="L2_meso")} == {
        iep_a.interface_ep_id,
        iep_b.interface_ep_id,
    }
    joined = store.find_edge(iep_a.atom_id, iep_b.atom_id, "causal", "L2_meso")
    assert joined is not None and joined.is_boundary and link in store.projections_of(joined.edge_id)
    assert graph.components(joint, kinds=("interface_ep",), scale="L2_meso", gap_hours=48.0 * 4) == [
        [iep_a.atom_id, iep_b.atom_id]
    ]
    assert graph.components(joint, kinds=("interface_ep",), scale="L1_micro") == []

    l3 = coarsen(
        store,
        rig_a.index,
        rig_a.embedder,
        graph,
        [iep_b.atom_id, iep_a.atom_id],
        target_scale="L3_macro",
        llm=None,
        at=store.now(),
        scope=joint,
        min_cohesion=0.3,
    )
    assert l3 is not None and l3.scale == "L3_macro" and l3.support_node_ids == [iep_a.atom_id, iep_b.atom_id]
    assert l3.cohesion_score == pytest.approx(0.5 + 0.5 * 1 / 8, abs=1e-6) and l3.support_edge_ids == [joined.edge_id]
    assert (
        l3.slots.u == iep_a.slots.u and l3.slots.a == f"{iep_a.slots.a} → {iep_b.slots.a}" and l3.slots.r == "positive"
    )
    assert l3.slots.b.count(S4_CONSTRAINT) == 1 and l3.slots.b.count(S4_DC) == 1
    assert [inv.text for inv in l3.invariants if inv.kind == "constraint"] == [S4_CONSTRAINT, S4_DC]
    assert l3.invariants[0].support_node_ids == [iep_a.atom_id, iep_b.atom_id] and l3.invariants[
        0
    ].anchor_ids == sorted(set(iep_a.evidence_anchor_ids) | set(iep_b.evidence_anchor_ids))
    assert l3.evidence_anchor_ids == sorted(
        set(iep_a.evidence_anchor_ids) | set(iep_b.evidence_anchor_ids) | set(joined.witness_anchor_ids)
    )
    atom = store.get_atom(l3.atom_id)
    assert atom.scale_level == "L3_macro" and atom.source_kind == "interface_ep" and atom.scope == joint
    assert atom.source_ep_ids == [rig_a.ep1.ep_id, rig_a.ep2.ep_id, rig_b.ep1.ep_id, rig_b.ep2.ep_id]
    assert store.get_node(l3.atom_id).scale == "L3_macro"
    high = project(graph, l3, at=store.now())
    assert high and all(store.get_edge(h).scale == "L3_macro" and store.projections_of(h) for h in high)
    assert {store.get_edge(h).dst_id for h in high} >= {
        rig_a.uc.atom_id,
        rig_b.uc.atom_id,
        rig_a.dc.atom_id,
        rig_b.dc.atom_id,
    }
    opened = refine(store, graph, l3.interface_ep_id, hops=0)
    assert [n["node_id"] for n in opened["nodes"]] == [iep_a.atom_id, iep_b.atom_id] and round_trip_check(graph, l3)
    assert (
        coarsen(
            store,
            rig_a.index,
            rig_a.embedder,
            graph,
            [iep_a.atom_id, iep_b.atom_id],
            target_scale="L3_macro",
            llm=None,
            at=store.now(),
            scope=joint,
            min_cohesion=0.3,
        ).interface_ep_id
        == l3.interface_ep_id
    )
    with pytest.raises(ValidationError):
        coarsen(
            store,
            rig_a.index,
            rig_a.embedder,
            graph,
            [iep_a.atom_id, iep_b.atom_id],
            target_scale="L2_meso",
            llm=None,
            at=store.now(),
            scope=joint,
            min_cohesion=0.3,
        )
    stats = graph.stats(joint)
    assert stats["nodes"]["interface_ep"] == 3 and stats["interface_eps"] == 3 and stats["traceable_high_edges"] == 1.0
    assert graph.stats(None)["nodes"] == stats["nodes"]
    trace = explain(store, graph, resolve_target(store, l3.interface_ep_id), hops=0)
    assert trace["refined_from"] == [l3.interface_ep_id, iep_a.interface_ep_id, iep_b.interface_ep_id]


# ---------------------------------------------------------------------------
# extra: supersession, LLM policy, stats
# 补充：取代关系、LLM 策略、统计
# ---------------------------------------------------------------------------
def test_recoarsen_superset_invalidates_predecessor(store, clock, events, scope4, index_provider):
    """A later coarsen over a strict superset invalidates the old InterfaceEP and links it by a revision edge witnessed by the new members.
    对严格超集的再粗化使旧 InterfaceEP 失效，并以新成员见证的 revision 边相连。

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
        Assertions on invalidation, the revision edge and low-cohesion residuals.
        对失效、revision 边与低内聚残差的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    first = _coarsen_s4(rig)
    config = HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"})
    ingestor = Ingestor(store, rig.index, rig.embedder, ScriptedLLM({"responses": []}), config)
    clock.tick(seconds=60)
    rc = ingestor.ingest_events(
        events_from_dicts(
            [
                {"object_type": "file_diff", "timestamp": "2026-06-15T11:00:00Z", "payload": {"path": "src/billing/x.ts", "patch": "+ y"}},
                {"object_type": "test_log", "timestamp": "2026-06-15T11:30:00Z", "payload": {"suite": "billing_e2e", "passed": 1, "failed": 0}},
            ],
            scope_fallback=rig.scope,
            visibility_default="private",
            event_time=clock.now(),
        ),
        infer=True,
    )  # fmt: skip
    ep3 = store.get_ep(rc.ep_ids[0])
    superset = rig.support + [ep3.atom_id]
    assert rig.graph.components(rig.scope) == [superset]
    clock.tick(seconds=60)
    second = coarsen(
        store, rig.index, rig.embedder, rig.graph, superset, llm=None, at=store.now(), scope=rig.scope, min_cohesion=0.3
    )
    assert (
        second is not None and second.interface_ep_id != first.interface_ep_id and second.support_node_ids == superset
    )
    old = store.get_interface_ep(first.interface_ep_id)
    assert old.valid is False and old.valid_to == store.now() and store.get_node(first.atom_id).valid is False
    assert store.get_atom(first.atom_id).archived_reason == "revised"
    revision = store.find_edge(first.atom_id, second.atom_id, "revision", "L2_meso")
    assert revision is not None and revision.created_by == "consolidation" and revision.is_boundary is False
    assert revision.witness_anchor_ids == store.atom_anchor_ids(ep3.atom_id)
    assert [i.interface_ep_id for i in store.list_interface_eps(rig.scope)] == [second.interface_ep_id]
    assert rig.graph.owner_interface_ep(rig.ep1.atom_id, scale="L2_meso").interface_ep_id == second.interface_ep_id
    assert rig.graph.owner_interface_ep(rig.ep1.atom_id, scale="L3_macro") is None
    weak = coarsen(
        store,
        rig.index,
        rig.embedder,
        rig.graph,
        [rig.ep1.atom_id, ep3.atom_id],
        llm=None,
        at=store.now(),
        scope=rig.scope,
        min_cohesion=0.3,
    )
    # ep1 与 ep3 之间没有内部边 → 内聚度 0.0
    assert weak is None  # no internal edge between ep1 and ep3 → cohesion 0.0
    # Low-cohesion residual on a hand-built pair: one internal edge, two boundary edges → cohesion 2/3.
    # 手工构造的一对节点上的低内聚残余：一条内部边、两条边界边 → 内聚度 2/3。
    hand = Scope(user_id="hand", run_id="low")
    p1 = _seed_ep(
        store,
        rig.index,
        rig.embedder,
        scope=hand,
        ts="2026-05-03T10:00:00Z",
        u="lower latency",
        a="patch cache",
        y="tests passed",
        r="positive",
    )
    p2 = _seed_ep(
        store,
        rig.index,
        rig.embedder,
        scope=hand,
        ts="2026-05-03T11:00:00Z",
        u="lower latency",
        a="patch pool",
        y="tests passed",
        r="positive",
    )
    c = _seed_state(store, rig.index, rig.embedder, scope=hand, ts="2026-05-03T09:00:00Z", text="Keep p99 under 200ms.")
    rig.graph.insert_edge(
        src_id=p1.atom_id, dst_id=p2.atom_id, edge_type="causal", witness_anchor_ids=p2.anchor_ids, created_by="ingest"
    )
    for ep in (p1, p2):
        rig.graph.insert_edge(
            src_id=ep.atom_id, dst_id=c.atom_id, edge_type="use", witness_anchor_ids=ep.anchor_ids, created_by="ingest"
        )
    pair = [p1.atom_id, p2.atom_id]
    assert rig.graph.cohesion(pair) == pytest.approx(2 / 3)
    assert (
        coarsen(store, rig.index, rig.embedder, rig.graph, pair, llm=None, at=store.now(), scope=hand, min_cohesion=0.7)
        is None
    )
    assert store.list_interface_eps(hand, valid_only=False) == []
    low = coarsen(
        store, rig.index, rig.embedder, rig.graph, pair, llm=None, at=store.now(), scope=hand, min_cohesion=0.6
    )
    assert (
        low is not None
        and [r.kind for r in low.residuals] == ["low_cohesion"]
        and low.cohesion_score == pytest.approx(2 / 3)
    )
    assert low.residuals[0].support_node_ids == pair and low.residuals[0].anchor_ids == sorted(
        set(p1.anchor_ids) | set(p2.anchor_ids)
    )
    assert low.residuals[0].edge_ids == low.support_edge_ids and round_trip_check(rig.graph, low)


def test_coarsen_llm_policy_validates_refs(store, clock, events, scope4, index_provider, scripted_llm):
    """policy='llm': slots with valid support refs are kept, slots with unknown refs fall back to the rule, unusable output falls back entirely.
    policy='llm'：支持引用有效的槽位保留，引用未知的槽位回退到规则值，不可用输出整体回退。

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

        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

    Output / 输出：
        Assertions on the merged slots, slot_support, invariants and coarsen_op_version.
        对合并后的槽位、slot_support、不变量与 coarsen_op_version 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    ep1_atom, ep2_atom = rig.support
    proposal = {
        "interface_ep": {
            "slots": {"u": "stabilise the checkout webhook", "b": "", "a": "backoff then idempotency", "y": "green", "r": "positive", "kappa": {"constraints": [S4_CONSTRAINT], "visibility": "private"}},
            "slot_support": {"u": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n9"], "r": ["n1"], "kappa": ["n0", "n1"]},
            "invariants": [
                {"text": S4_CONSTRAINT, "kind": "constraint", "support_refs": ["n0", "n1"]},
                {"text": "ghost", "kind": "constraint", "support_refs": ["n5"]},
                {"text": "bad kind", "kind": "nope", "support_refs": ["n0"]},
            ],
            "residuals": [{"text": "why did n0 fail?", "kind": "open_question", "support_refs": ["n0"]}],
        }
    }  # fmt: skip
    llm = scripted_llm([proposal])
    iep = _coarsen_s4(rig, policy="llm", llm=llm)
    assert llm.calls and llm.calls[0]["task"] == "coarsen"
    assert iep.coarsen_op_version == "llm-coarsen-v1" and store.get_atom(iep.atom_id).extracted_by.endswith(
        "@coarsen-v1"
    )
    assert iep.slots.u == "stabilise the checkout webhook" and iep.slot_support["u"] == [ep1_atom, ep2_atom]
    # y：引用未知 → 回退到 rule
    assert iep.slots.a == "backoff then idempotency" and iep.slots.y == "billing_e2e passed"  # y: unknown ref → rule
    # b：没有引用 → 回退到 rule
    assert iep.slot_support["y"] == [ep2_atom] and iep.slots.b == f"{S4_CONSTRAINT} ; {S4_DC}"  # b: no refs → rule
    assert iep.slots.kappa.constraints == [S4_CONSTRAINT] and iep.slots.kappa.evidence == iep.evidence_anchor_ids
    assert [(i.text, i.support_node_ids) for i in iep.invariants] == [(S4_CONSTRAINT, [ep1_atom, ep2_atom])]
    assert iep.invariants[0].anchor_ids == iep.evidence_anchor_ids and iep.invariants[0].confidence == 0.8
    assert [(r.kind, r.support_node_ids, r.anchor_ids) for r in iep.residuals] == [
        ("open_question", [ep1_atom], sorted(rig.ep1.anchor_ids))
    ]
    assert round_trip_check(rig.graph, iep)

    fresh_store = StateStore(":memory:", now=clock.now, id_factory=lambda p: f"{p}_{abs(hash(p)) % 10**6:06d}")
    assert fresh_store.stats()["graph_nodes"] == 0
    other = _build_s4(
        store, clock, events, {"user_id": "alice", "agent_id": "coder", "run_id": "run-llm"}, index_provider, rig=rig
    )
    garbage = scripted_llm(["not json at all"])
    fallback = _coarsen_s4(other, policy="llm", llm=garbage)
    assert fallback.coarsen_op_version == "rule-coarsen-v1" and fallback.slots.u == "fix checkout webhook retry"
    with pytest.raises(ValidationError):
        coarsen(
            store,
            rig.index,
            rig.embedder,
            rig.graph,
            rig.support,
            policy="magic",
            llm=None,
            at=store.now(),
            scope=rig.scope,
            min_cohesion=0.3,
        )


# ---------------------------------------------------------------------------
# 45c
# ---------------------------------------------------------------------------
def test_coarsen_llm_policy_validates_enums(store, clock, events, scope4, index_provider, scripted_llm, caplog):
    """policy='llm': an off-enum `r` or `kappa.visibility` falls back to the rule value with a warning instead of raising.
    policy='llm'：枚举外的 `r` 或 `kappa.visibility` 回退到规则值并告警，而不是抛异常。

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

        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

        caplog: pytest log capture (the two fallback warnings).
        caplog：pytest 日志捕获（两条回退告警）。

    Output / 输出：
        Assertions on the persisted slots, slot_support, coarsen_op_version and the warnings.
        对持久化的槽位、slot_support、coarsen_op_version 与告警的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    ep1_atom, ep2_atom = rig.support
    proposal = {
        "interface_ep": {
            "slots": {"u": "stabilise the checkout webhook", "b": "", "a": "backoff then idempotency", "y": "green", "r": "great", "kappa": {"constraints": [S4_CONSTRAINT], "visibility": "everyone"}},
            "slot_support": {"u": ["n0", "n1"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n0"], "kappa": ["n0", "n1"]},
            "invariants": [],
            "residuals": [],
        }
    }  # fmt: skip
    llm = scripted_llm([proposal])
    with caplog.at_level(logging.WARNING, logger="hms_elite.graph.operators"):
        iep = _coarsen_s4(rig, policy="llm", llm=llm)
    assert llm.calls and llm.calls[0]["task"] == "coarsen"
    # valid slots stay LLM-authored; r and kappa.visibility fall back to the rule values (S4 rule r = 'positive' ← EP2)
    # 有效槽位仍来自 LLM；r 与 kappa.visibility 落回规则值（S4 的规则 r = 'positive'，来自 EP2）
    assert iep.coarsen_op_version == "llm-coarsen-v1" and iep.slots.u == "stabilise the checkout webhook"
    assert iep.slots.r == "positive" and iep.slot_support["r"] == [ep2_atom]
    assert iep.slots.kappa.visibility == "private" and iep.slots.kappa.constraints == [S4_CONSTRAINT]
    assert iep.slot_support["kappa"] == [ep1_atom, ep2_atom] and iep.slots.y == "green"
    messages = [r.getMessage() for r in caplog.records]
    assert sum("slot 'r'" in m for m in messages) == 1 and sum("kappa.visibility" in m for m in messages) == 1
    assert round_trip_check(rig.graph, iep)


def test_graph_stats_after_coarsen_project_promote(store, clock, events, scope4, index_provider):
    """graph.stats(scope) after the S4 pipeline + coarsen + project + promote_skills counts every kind/type and reports full coverage.
    S4 流水线 + coarsen + project + promote_skills 之后 graph.stats(scope) 计数每种节点/边并报告完整覆盖率。

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
        Assertions on the exact stats dict, scope filtering and the whole-DB variant.
        对精确统计字典、范围过滤与整库变体的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rig = _build_s4(store, clock, events, scope4, index_provider)
    empty = rig.graph.stats(Scope(user_id="nobody"))
    assert empty["nodes"] == {
        k: 0 for k in ("message", "observation", "event", "entity", "ep", "interface_ep", "skill")
    }
    assert empty["edges"] == {k: 0 for k in ("temporal", "causal", "revision", "conflict", "use", "distill")}
    assert (
        empty["witness_coverage"],
        empty["projection_coverage"],
        empty["traceable_high_edges"],
        empty["transitions"],
    ) == (1.0, 1.0, 1.0, 0)
    iep = _coarsen_s4(rig)
    project(rig.graph, iep, at=store.now())
    promote_skills(
        store,
        rig.index,
        rig.embedder,
        rig.graph,
        scope=rig.scope,
        at=store.now(),
        anchor_id=_sched_anchor(store, rig.scope),
    )
    assert rig.graph.stats(rig.scope) == {
        "nodes": {"message": 0, "observation": 2, "event": 4, "entity": 0, "ep": 2, "interface_ep": 1, "skill": 1},
        "edges": {"temporal": 1, "causal": 1, "revision": 0, "conflict": 0, "use": 5, "distill": 3},
        "valid_edges": 10,
        "invalid_edges": 0,
        "boundary_edges": 3,
        "interface_eps": 1,
        "unresolved_conflicts": 0,
        "witness_coverage": 1.0,
        "projection_coverage": 1.0,
        "traceable_high_edges": 1.0,
        "transitions": 2,
    }
    assert rig.graph.stats(None) == rig.graph.stats(rig.scope) == rig.graph.stats(Scope(user_id="alice"))
    low_use = rig.edges[("use", rig.ep1.atom_id, rig.uc.atom_id)].edge_id
    store.invalidate_edge(low_use, reason="test", at=store.now())
    after = rig.graph.stats(rig.scope)
    assert (after["valid_edges"], after["invalid_edges"], after["edges"]["use"]) == (9, 1, 5)
