"""Group F tests: slot rules, Work State Card goldens, budget packing, compile modes, support layer and PC-IR obligations.
F 组测试：槽位规则、Work State Card 黄金比对、预算打包、编译模式、支持层与 PC-IR 义务。
"""

# Purpose (EN): Assert CONTRACT §15 (and §18.3 tests 46–51) over the compiler plane on S1/S4 ingests and hand-built IRs.
# 目的（中）：在 S1/S4 摄取结果与手工构造的 IR 上断言契约 §15（以及 §18.3 测试 46–51）。

# Input (EN): tests/fixtures/events/*.json, tests/fixtures/golden/work_state_card_*.txt and the conftest fixtures.
# 输入（中）：tests/fixtures/events/*.json、tests/fixtures/golden/work_state_card_*.txt 与 conftest 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

from typing import Any

import pytest

from hms_elite.compiler import (
    SLOT_RULES,
    ProofObligationError,
    StateCompiler,
    assign_slot,
    derive_next_action,
    evidence_items,
    fit_budget,
    invariant_items,
    item_from_atom,
    kappa_items,
    lower,
    render_card,
)
from hms_elite.configs.base import HMSConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.enums import Slot
from hms_elite.exceptions import ValidationError
from hms_elite.graph import SupportGraph, coarsen, project, promote_skills
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.ingest.atoms import create_atom
from hms_elite.ingest.events import events_from_dicts, system_event
from hms_elite.ingest.pipeline import Ingestor, IngestResult
from hms_elite.llms.rule import RuleLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages, estimate_tokens
from hms_elite.models import (
    Anchor,
    ContextIR,
    ContextItem,
    InterfaceEP,
    Invariant,
    Kappa,
    MemoryAtom,
    Residual,
    Scope,
    Slots,
    SourceEvent,
)
from hms_elite.mvmi.filters import parse_filters
from tests.conftest import FrozenClock

S4_QUERY = "fix checkout webhook retry"
S4_AS_OF = "2026-06-15T12:00:00Z"
S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP1_Y = "billing_e2e failed: subscription_state_regression"
S4_EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
S4_DC = f"Do not re-apply '{S4_EP1_A}': {S4_EP1_Y}"
S1_SCOPE = {"user_id": "alice", "run_id": "run-coffee"}
LATTE = "User likes oat-milk lattes in the morning."
BLACK = "User likes black coffee."


class _Harness:
    """Store + index + embedder + rule LLM + config wired into an Ingestor, a SupportGraph and a StateCompiler.
    将 store + 索引 + 嵌入器 + 规则 LLM + 配置接入 Ingestor、SupportGraph 与 StateCompiler。
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
        self.ingestor = Ingestor(store, self.index, self.embedder, RuleLLM(), self.config)
        self.graph = SupportGraph(store, self.config.graph)
        self.compiler = StateCompiler(store, self.index, self.embedder, self.graph, self.config)

    def add(self, messages: Any, *, timestamp: str | None = None, **scope: str) -> IngestResult:
        """Mem0 add() minus the facade: coerce, tick, ingest_messages with the rule LLM.
        去掉门面的 Mem0 add()：规整、推进时钟、用规则 LLM 执行 ingest_messages。
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
        """HMS add_events() minus the facade: events_from_dicts, tick, ingest_events (no LLM call happens).
        去掉门面的 HMS add_events()：events_from_dicts、推进时钟、ingest_events（不会调用 LLM）。
        """
        self.clock.tick(seconds=60)
        events = events_from_dicts(
            items, scope_fallback=Scope(**scope), visibility_default="private", event_time=self.clock.now()
        )
        return self.ingestor.ingest_events(events, infer=True)

    def compile(self, query: str, filters: dict, **kw: Any) -> ContextIR:
        """HMS compile() minus the facade: scope from the filters, then StateCompiler.compile.
        去掉门面的 HMS compile()：由过滤器得到范围，再调用 StateCompiler.compile。
        """
        scope, _expr = parse_filters(filters, require_scope=True, context="compile")
        return self.compiler.compile(query, scope=scope, filters=filters, **kw)


def _anchor_of(res: IngestResult, object_type: str) -> str:
    """The anchor id of the first event of a type in an IngestResult.
    IngestResult 中某类型首个事件的锚点 id。
    """
    return [e["anchor_id"] for e in res.events if e["object_type"] == object_type][0]


def _build_s4(h: _Harness, events: Any, scope4: dict, *, fix: bool) -> dict[str, Any]:
    """Ingest the §19.3 S4 fixture (constraint + events_fail [+ events_fix]); return the ids a test needs.
    摄取 §19.3 的 S4 夹具（约束 + events_fail [+ events_fix]）；返回测试所需的 id。
    """
    fixture = events("structured_billing")
    r0 = h.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    ra = h.add_events(fixture.events_fail, **scope4)
    out: dict[str, Any] = {
        "uc": r0.rows[0]["atom_id"],
        "dc": ra.constraint_ids[0],
        "ep1": h.store.get_ep(ra.ep_ids[0]),
        "anc_uc": h.store.get_atom(r0.rows[0]["atom_id"]).anchor_ids[0],
        "anc_e1": _anchor_of(ra, "file_diff"),
        "anc_e2": _anchor_of(ra, "test_log"),
        "events_fail": [row["atom_id"] for row in ra.rows if row["atom_type"] == "event"],
    }
    if fix:
        rb = h.add_events(fixture.events_fix, **scope4)
        out["ep2"] = h.store.get_ep(rb.ep_ids[0])
        out["anc_e3"] = _anchor_of(rb, "file_diff")
        out["anc_e4"] = _anchor_of(rb, "test_log")
    return out


def _seed_event(store: StateStore, text: str, ts: str, scope: Scope, *, visibility: str = "private") -> Anchor:
    """Insert one message event and its anchor (permission_scope = the event visibility); return the Anchor.
    插入一条消息事件及其锚点（permission_scope = 事件可见性）；返回 Anchor。
    """
    ev = SourceEvent(scope=scope, role="user", payload={"text": text}, timestamp=ts, visibility=visibility)
    stored, _dedup = store.insert_source_event(ev)
    return store.get_or_create_anchor(stored)


def _seed_state(
    h: _Harness,
    *,
    scope: Scope,
    ts: str,
    text: str,
    obs_type: str = "fact",
    visibility: str = "private",
) -> MemoryAtom:
    """Insert a state atom (+node) witnessed by a fresh message anchor of the same visibility.
    插入一个由同可见性的新消息锚点见证的 state 原子（+节点）。
    """
    anchor = _seed_event(h.store, text, ts, scope, visibility=visibility)
    atom = MemoryAtom(
        atom_type="state",
        content={"text": text},
        text=text,
        time_anchor=ts,
        scale_level="state",
        subject="user",
        predicate="is",
        obs_type=obs_type,
        source_kind="observation",
        source_id="obs_hand",
        extracted_by="test",
        visibility=visibility,
        scope=scope,
        anchor_ids=[anchor.anchor_id],
    )
    return create_atom(h.store, h.index, h.embedder, atom)


def _atom(**overrides: Any) -> MemoryAtom:
    """An unsaved valid state atom with overrides (slot-rule unit tests).
    应用覆盖后的未保存合法 state 原子（槽位规则单元测试）。
    """
    base: dict[str, Any] = dict(
        atom_id="atm_unit",
        memory_id="mem_unit",
        atom_type="state",
        content={"text": "User lives in Berlin."},
        text="User lives in Berlin.",
        scale_level="state",
        source_kind="observation",
        source_id="obs_unit",
        obs_type="fact",
        extracted_by="test",
        scope=Scope(user_id="alice"),
        anchor_ids=["anc_unit"],
        time_anchor="2026-06-01T00:00:00Z",
        valid_from="2026-06-01T00:00:00Z",
    )
    base.update(overrides)
    return MemoryAtom(**base)


def _episode_atom(atom_id: str, *, a: str, y: str, r: str, constraints: list[str], ts: str) -> MemoryAtom:
    """An unsaved L1 episode atom whose content carries the slots (κ constraints included).
    内容携带槽位（含 κ 约束）的未保存 L1 情节原子。
    """
    slots = Slots(u="goal", a=a, y=y, r=r, kappa=Kappa(constraints=constraints, evidence=[f"anc_{atom_id}"]))
    text = slots.gloss()
    return _atom(
        atom_id=atom_id,
        memory_id=f"mem_{atom_id}",
        atom_type="episode",
        content={"text": text, "ep_id": f"ep_{atom_id}", "slots": slots.model_dump()},
        text=text,
        scale_level="L1_micro",
        source_kind="ep",
        source_id=f"ep_{atom_id}",
        obs_type=None,
        anchor_ids=[f"anc_{atom_id}"],
        time_anchor=ts,
        valid_from=ts,
    )


def _hand_anchor(anchor_id: str, *, ts: str = "2026-06-01T00:00:00Z", permission_scope: str = "private") -> Anchor:
    """A hand-built Anchor row for store-free budget / PC-IR tests.
    用于无存储预算 / PC-IR 测试的手工 Anchor 行。
    """
    return Anchor(
        anchor_id=anchor_id,
        source_event_id=f"evt_{anchor_id}",
        hash=("ab" * 32),
        source_ref=f"chat://user_id=alice/{anchor_id}",
        timestamp=ts,
        ingested_at=ts,
        permission_scope=permission_scope,
        scope=Scope(user_id="alice"),
    )


def _item(
    item_id: str,
    slot: str,
    text: str,
    *,
    score: float = 0.0,
    time_anchor: str | None = None,
    anchors: list[str] | None = None,
    explain: dict | None = None,
    visibility: str = "private",
) -> ContextItem:
    """A hand-built ContextItem with tokens == estimate_tokens(text).
    tokens == estimate_tokens(text) 的手工 ContextItem。
    """
    return ContextItem(
        item_id=item_id,
        slot=slot,
        text=text,
        atom_id=item_id,
        score=score,
        time_anchor=time_anchor,
        visibility=visibility,
        tokens=estimate_tokens(text),
        explain=explain or {},
        anchor_ids=anchors or ["anc_a"],
    )


def _assert_witnessed_and_disjoint(ir: ContextIR) -> None:
    """Every item has >= 1 anchor and item ids are pairwise disjoint across the six slots.
    每个项至少一个锚点，且六个槽位间 item_id 两两互斥。
    """
    ids = [item.item_id for item in ir.items()]
    assert len(ids) == len(set(ids))
    for item in ir.items():
        assert item.anchor_ids


# ---------------------------------------------------------------------------
# 46
# ---------------------------------------------------------------------------
def test_slot_rules_total_and_disjoint(store, clock, events, scope4, index_provider):
    """Every atom kind lands in exactly one slot; future valid_to stays natural; expired → audit; κ dedup; covered events; default_mode.
    每类原子恰好落入一个槽位；未来 valid_to 保持自然槽位；已过期 → audit；κ 去重；被覆盖事件丢弃；default_mode。

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
        Assertions over assign_slot, item_from_atom, kappa_items and compile().
        对 assign_slot、item_from_atom、kappa_items 与 compile() 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    as_of = "2026-06-15T12:00:00Z"
    assert [name for name, _p, _s in SLOT_RULES] == [
        "archived_or_expired",
        "skill",
        "constraint",
        "preference",
        "episode",
        "fact",
    ]
    cases = {
        "message": (_atom(atom_type="message", scale_level="L1_micro", source_kind="message", obs_type=None), "facts"),
        "fact": (_atom(), "facts"),
        "temporal": (_atom(atom_type="event", scale_level="event", obs_type="temporal_fact"), "facts"),
        "event": (_atom(atom_type="event", scale_level="event", source_kind="event", obs_type=None), "facts"),
        "preference": (_atom(obs_type="preference_event"), "preferences"),
        "habit": (_atom(obs_type="habit"), "preferences"),
        "constraint": (_atom(obs_type="constraint"), "constraints"),
        "episode": (_episode_atom("atm_ep", a="a", y="y", r="negative", constraints=[], ts=as_of), "episodes"),
        "procedural": (
            _atom(atom_type="episode", scale_level="L1_micro", source_kind="procedural", obs_type=None),
            "episodes",
        ),
        "skill": (_atom(scale_level="L3_macro", source_kind="skill", obs_type=None), "skills"),
        "skill_constraint": (_atom(scale_level="L3_macro", source_kind="skill", obs_type="constraint"), "skills"),
        "archived": (_atom(archived_reason="revised", valid_to=as_of), "audit"),
        "expired": (_atom(valid_to="2026-06-15T11:59:59Z"), "audit"),
        "expiring_now": (_atom(valid_to=as_of), "audit"),
        "future_expiry_pref": (_atom(obs_type="preference_event", valid_to="2027-01-01T00:00:00Z"), "preferences"),
        "future_expiry_fact": (_atom(valid_to="2026-06-15T12:00:01Z"), "facts"),
    }
    for name, (atom, expected) in cases.items():
        slot = assign_slot(atom, as_of=as_of)
        assert isinstance(slot, Slot) and slot == Slot(expected), name
    assert assign_slot(_atom(archived_reason="revised"), as_of=as_of, reason="revised") == Slot.audit
    assert assign_slot(_atom(), as_of=as_of, reason="revised") == Slot.audit

    expired = item_from_atom(
        cases["expired"][0], slot="audit", score=0.0, anchor_ids=["anc_unit"], source="axis", hops=0, explain=None
    )
    assert expired.explain["reason"] == "archived:expired" and expired.tokens == estimate_tokens(expired.text)
    archived = item_from_atom(
        cases["archived"][0], slot="audit", score=0.0, anchor_ids=["anc_unit"], source="axis", hops=0, explain=None
    )
    assert archived.explain["reason"] == "archived:revised"
    ep_item = item_from_atom(
        cases["episode"][0], slot="episodes", score=0.5, anchor_ids=["anc_atm_ep"], source="axis", hops=0, explain=None
    )
    assert ep_item.item_id == "atm_ep" and ep_item.ep_id == "ep_atm_ep"
    assert {"r", "a", "y", "u", "ep_id", "kappa"} <= set(ep_item.explain)

    # κ items: deduplicated against selected constraints AND audit items, and across episodes (first wins).
    # κ 项：既对已选 constraints 也对 audit 项去重，并跨 episodes 去重（先出现者胜）。
    ep_a = item_from_atom(
        _episode_atom("atm_a", a="a1", y="y1", r="negative", constraints=["Keep X.", "Keep Y.", "Keep Z."], ts=as_of),
        slot="episodes",
        score=0.0,
        anchor_ids=["anc_atm_a"],
        source="axis",
        hops=0,
        explain=None,
    )
    ep_b = item_from_atom(
        _episode_atom("atm_b", a="a2", y="y2", r="positive", constraints=["keep y", "Keep W."], ts=as_of),
        slot="episodes",
        score=0.0,
        anchor_ids=["anc_atm_b"],
        source="axis",
        hops=0,
        explain=None,
    )
    # 'Keep X.' 已被选为 constraint，'Keep Z.' 作为 audit 项存在
    selected = {"keep x", "keep z"}  # 'Keep X.' selected as a constraint, 'Keep Z.' present as an audit item
    items = kappa_items([ep_a, ep_b], selected)
    assert [i.item_id for i in items] == ["atm_a#kappa:1", "atm_b#kappa:1"]
    assert [i.text for i in items] == ["Keep Y.", "Keep W."]
    assert all(i.slot == "constraints" and i.source == "system" for i in items)
    assert items[0].anchor_ids == ["anc_atm_a"] and items[1].anchor_ids == ["anc_atm_b"]
    assert (
        kappa_items([ep_a, ep_b], set()) and [i.text for i in kappa_items([ep_a], {"keep x", "keep y", "keep z"})] == []
    )

    # Covered event atoms are dropped by compile(); the compiled slots are disjoint.
    # 被覆盖的 event 原子会被 compile() 丢弃；编译出的各 slot 互不相交。
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4, fix=False)
    ir = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    _assert_witnessed_and_disjoint(ir)
    assert ir.retrieval["dropped_covered"] == 2 and ir.facts == []
    assert set(ids["events_fail"]).isdisjoint({i.atom_id for i in ir.items()})
    assert [c.text for c in ir.constraints] == [S4_CONSTRAINT, S4_DC]
    assert not [i for i in ir.constraints if "#kappa:" in i.item_id]

    # mode=None reads config.compiler.default_mode.
    # mode=None 时读取 config.compiler.default_mode。
    fast = _Harness(
        StateStore(":memory:", now=clock.now, id_factory=store.new_id),
        clock,
        index_provider,
        compiler={"default_mode": "fast"},
    )
    _build_s4(fast, events, scope4, fix=False)
    ir_fast = fast.compile(S4_QUERY, filters=scope4, mode=None)
    assert ir_fast.retrieval["ablation"] == "axis" and ir_fast.retrieval["graph_hops"] == 0
    assert str(ir_fast.mode) == "fast" and ir_fast.retrieval["mode"] == "fast"
    with pytest.raises(ValidationError) as exc:
        h.compile(S4_QUERY, filters=scope4, mode="bogus")
    assert exc.value.error_code == "VALIDATION_013"


# ---------------------------------------------------------------------------
# 47
# ---------------------------------------------------------------------------
def test_work_state_card_failed_golden(store, clock, events, scope4, index_provider, golden):
    """After constraint + events_fail the card equals work_state_card_failed.txt byte-for-byte; slots per §19.3.
    约束 + events_fail 之后卡片与 work_state_card_failed.txt 逐字节相等；槽位符合 §19.3。

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

        golden: Golden helper (text mode).
        golden：黄金比对辅助（文本模式）。

    Output / 输出：
        Golden and slot assertions.
        黄金比对与槽位断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4, fix=False)
    before = store.stats()
    ir = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    assert store.stats() == before, "compile() must be read-only"
    card = ir.to_text()
    golden.text("work_state_card_failed.txt", card)
    assert card == render_card(ir) and not card.endswith("\n")
    _assert_witnessed_and_disjoint(ir)
    assert ir.used_tokens <= 800 and ir.used_tokens == sum(i.tokens for i in ir.items())
    assert ir.retrieval["dropped_covered"] == 2 and ir.retrieval["dropped"] == 0
    assert [c.item_id for c in ir.constraints] == [ids["uc"], ids["dc"]]
    assert [e.item_id for e in ir.episodes] == [ids["ep1"].atom_id]
    assert ir.facts == [] and ir.preferences == [] and ir.skills == []
    assert len(ir.audit) == 3 and all(a.explain["kind"] == "evidence" for a in ir.audit)
    assert [a.anchor_ids[0] for a in ir.audit] == [ids["anc_uc"], ids["anc_e1"], ids["anc_e2"]]
    assert ir.unresolved_conflicts == [] and str(ir.route) == "recall"
    assert ir.retrieval["graph_hops"] == 1 and ir.retrieval["ablation"] == "axis_graph_support"
    assert ir.retrieval["mode"] == "auto" and str(ir.mode) == "auto"
    assert {a["anchor_id"] for a in ir.retrieval["anchors"]} == {ids["anc_uc"], ids["anc_e1"], ids["anc_e2"]}
    assert ir.episodes[0].explain["kappa"] == [S4_CONSTRAINT] and ir.episodes[0].ep_id == ids["ep1"].ep_id
    assert ir.as_of == "2026-06-15T12:00:00.000000+00:00" and ir.goal == S4_QUERY
    # Deterministic: a second compile yields the identical card and IR.
    # 确定性：第二次编译产出完全相同的卡片与 IR。
    again = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    assert again.to_text() == card and again.model_dump() == ir.model_dump()


# ---------------------------------------------------------------------------
# 48
# ---------------------------------------------------------------------------
def test_work_state_card_fixed_golden(store, clock, events, scope4, index_provider, golden):
    """After events_fix the card equals work_state_card_fixed.txt; deep/axis_graph give the same card; fast is axis-only.
    events_fix 之后卡片与 work_state_card_fixed.txt 相等；deep/axis_graph 给出同一卡片；fast 仅走轴。

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

        golden: Golden helper (text mode).
        golden：黄金比对辅助（文本模式）。

    Output / 输出：
        Golden, ablation and PC-IR assertions.
        黄金比对、消融与 PC-IR 断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4, fix=True)
    ir = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF)
    card = ir.to_text()
    golden.text("work_state_card_fixed.txt", card)
    _assert_witnessed_and_disjoint(ir)
    assert ir.retrieval["dropped_covered"] == 4
    assert [e.item_id for e in ir.episodes] == [ids["ep1"].atom_id, ids["ep2"].atom_id]
    assert len(ir.audit) == 5
    pcir = ir.to_pcir()
    assert pcir.ok() is True and pcir.human_gloss == card
    statuses = {str(o.name): str(o.status) for o in pcir.proof_obligations}
    assert statuses == {
        "evidence_coverage": "satisfied",
        "no_unresolved_conflict": "satisfied",
        "governance": "skipped",
        "not_revoked": "satisfied",
    }
    assert pcir.witness["_all"] == ir.anchor_ids() and set(pcir.witness["items"]) == {i.item_id for i in ir.items()}
    assert [e["a"] for e in pcir.payload["episodes"]] == [S4_EP1_A, S4_EP2_A]
    assert pcir.payload["next_action"] == ir.next_action and pcir.payload["scope"] == scope4

    deep = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="deep")
    assert deep.to_text().replace("Mode: deep", "Mode: auto") == card
    assert deep.retrieval["ablation"] == "axis_graph_support"
    axis_graph = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="axis_graph")
    assert axis_graph.to_text().replace("Mode: axis_graph", "Mode: auto") == card
    assert axis_graph.retrieval["ablation"] == "axis_graph" and axis_graph.retrieval["graph_hops"] == 1
    store.trace.reset()
    fast = h.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF, mode="fast")
    assert not any(t.startswith(("graph_", "interface_ep")) for t in store.trace.tables())
    assert fast.retrieval["ablation"] == "axis" and fast.retrieval["graph_hops"] == 0
    assert fast.to_text().replace("Mode: fast", "Mode: auto") == card


# ---------------------------------------------------------------------------
# 49
# ---------------------------------------------------------------------------
def test_budget_shares_min_constraints_and_spill(store, clock, events, scope4, index_provider):
    """fit_budget keeps >= min_constraints constraints under a tiny budget and spills the unused facts share to episodes.
    fit_budget 在极小预算下仍保留 >= min_constraints 条约束，并将未用的 facts 份额溢出给 episodes。

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
        Assertions over fit_budget and a tiny-budget compile().
        对 fit_budget 与极小预算 compile() 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    shares = HMSConfig().compiler.slot_shares
    anchors = {"anc_a": _hand_anchor("anc_a"), "anc_b": _hand_anchor("anc_b", ts="2026-06-02T00:00:00Z")}
    # 25 个 token
    text = "Never deploy to production on Fridays without a rollback plan"  # 25 tokens
    constraints = [
        _item(
            f"atm_c{i}", "constraints", f"{text} #{i}.", score=1.0 - i * 0.1, time_anchor=f"2026-06-0{i + 1}T00:00:00Z"
        )
        for i in range(5)
    ]
    packed, info = fit_budget(
        {"constraints": constraints}, budget=60, shares=shares, min_constraints=3, anchors=anchors
    )
    assert len(packed["constraints"]) == 3 and info["caps"]["constraints"] == 9
    # 先取分数最高的若干项
    assert {c.item_id for c in packed["constraints"]} == {"atm_c0", "atm_c1", "atm_c2"}  # best scores first
    # 按 time_anchor 升序排列
    assert [c.item_id for c in packed["constraints"]] == ["atm_c0", "atm_c1", "atm_c2"]  # time_anchor ASC
    assert [a.item_id for a in packed["audit"]] == ["anc:anc_a"] and info["dropped"] == 2
    assert info["used_tokens"] == sum(i.tokens for slot in packed.values() for i in slot)

    # 23 个 token
    episode_text = "[positive] goal → patch retry.ts → billing_e2e passed"  # 23 tokens
    episodes = [
        _item(
            f"atm_e{i}",
            "episodes",
            f"{episode_text} {i}",
            score=0.5,
            time_anchor=f"2026-06-1{i}T00:00:00Z",
            anchors=["anc_a" if i % 2 else "anc_b"],
        )
        for i in range(6)
    ]
    packed, info = fit_budget({"episodes": episodes}, budget=250, shares=shares, min_constraints=3, anchors=anchors)
    # 仅 episodes 份额本身只放得下两条 episode……
    assert info["caps"]["episodes"] == 62  # two episodes fit the episodes share alone ...
    # ……未用完的 facts 份额溢出给排序后列表中的其余项
    assert len(packed["episodes"]) > 2  # ... the unused facts share spills to the rest of the sorted list
    assert info["used_tokens"] <= 250 and info["over_budget_evidence"] == 0
    assert [e.item_id for e in packed["episodes"]] == sorted(e.item_id for e in packed["episodes"])
    assert {a.anchor_ids[0] for a in packed["audit"]} == {e.anchor_ids[0] for e in packed["episodes"]}
    no_spill, _info = fit_budget(
        {"episodes": episodes},
        budget=250,
        shares={**shares, "facts": 0.0, "audit": 0.0, "preferences": 0.0, "skills": 0.0, "constraints": 0.0},
        min_constraints=3,
        anchors=anchors,
    )
    # 溢出只会增加条目，不会减少
    assert len(no_spill["episodes"]) >= len(packed["episodes"])  # spill only ever adds items

    # Through compile(): a tiny budget keeps both S4 constraints (min_constraints=3 floor) and every evidence line.
    # 经由 compile()：极小预算仍保留两条 S4 constraints（min_constraints=3 下限）与每一行证据。
    h = _Harness(store, clock, index_provider)
    _build_s4(h, events, scope4, fix=False)
    ir = h.compile(S4_QUERY, filters=scope4, budget_tokens=40, as_of=S4_AS_OF)
    assert [c.text for c in ir.constraints] == [S4_CONSTRAINT, S4_DC]
    assert ir.episodes == [] and ir.retrieval["dropped"] == 1
    assert len(ir.audit) == 3 and ir.used_tokens > 40 and ir.retrieval["over_budget_evidence"] == ir.used_tokens - 40
    # 在预算裁剪之前的集合上计算
    assert ir.next_action.startswith("Do not re-apply")  # computed on the pre-budget set


# ---------------------------------------------------------------------------
# 50
# ---------------------------------------------------------------------------
def test_compile_modes(store, clock, events, scope4, index_provider):
    """fast never touches graph tables; deep on S1 yields the revised latte in audit; auto without EPs does not expand.
    fast 从不触及图表；S1 上的 deep 在 audit 中给出被修订的拿铁项；无 EP 的 auto 不扩展。

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
        Trace and slot assertions per mode.
        每种模式的跟踪与槽位断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    _build_s4(h, events, scope4, fix=True)
    store.trace.reset()
    fast = h.compile(S4_QUERY, filters=scope4, mode="fast")
    assert not any(t.startswith(("graph_", "interface_ep", "eps", "edge_")) for t in store.trace.tables())
    assert fast.retrieval["ablation"] == "axis" and fast.retrieval["graph_hops"] == 0
    assert [c.text for c in fast.constraints] == [S4_CONSTRAINT, S4_DC]

    msg1, msg2 = events("chat_preference").messages
    r1 = h.add(msg1["content"], timestamp=msg1["timestamp"], **S1_SCOPE)
    r2 = h.add(msg2["content"], timestamp=msg2["timestamp"], **S1_SCOPE)
    assert r1.rows[0]["event"] == "ADD" and r2.rows[0]["event"] == "UPDATE"
    auto = h.compile("coffee", filters=S1_SCOPE)
    assert [i.text for i in auto.preferences] == [BLACK] and auto.facts == []
    assert not [a for a in auto.audit if a.explain.get("reason") == "revised"]
    assert auto.retrieval["ablation"] == "axis" and auto.retrieval["graph_hops"] == 0
    assert "## Preferences\n- User likes black coffee." in auto.to_text()
    assert "## Skills" not in auto.to_text() and "## Invalidated Assumptions\n- (none)" in auto.to_text()
    deep = h.compile("coffee", filters=S1_SCOPE, mode="deep")
    revised = [a for a in deep.audit if a.explain.get("reason") == "revised"]
    assert len(revised) == 1
    assert revised[0].text == LATTE and revised[0].explain["superseded_by_text"] == BLACK
    assert revised[0].atom_id == r1.rows[0]["atom_id"] and revised[0].source == "graph"
    assert [i.text for i in deep.preferences] == [BLACK]
    assert deep.retrieval["ablation"] == "axis_graph_support" and deep.retrieval["graph_hops"] == 1
    assert f"## Invalidated Assumptions\n- {LATTE} -> superseded by: {BLACK}" in deep.to_text()
    assert {a.explain.get("kind") for a in deep.audit} == {"evidence", None}
    assert deep.to_pcir().ok() is True
    assert "## Preferences" not in h.compile(S4_QUERY, filters=scope4, mode="fast").to_text()


# ---------------------------------------------------------------------------
# 50b — support layer (refined InterfaceEP invariants) distinguishes deep from axis_graph / fast
# 50b —— 支撑层（细化后的 InterfaceEP 不变量）区分 deep 与 axis_graph / fast
# ---------------------------------------------------------------------------
def test_support_layer_after_coarsen(store, clock, events, scope4, index_provider):
    """After coarsen/project/promote_skills: deep adds '#inv:' items and 'invariant:' audit, axis_graph/fast do not.
    coarsen/project/promote_skills 之后：deep 增加 '#inv:' 项与 'invariant:' audit，axis_graph/fast 则不会。

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
        Ablation assertions of §19.7.
        §19.7 的消融断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4, fix=True)
    scope = Scope(**scope4)
    support = [ids["ep1"].atom_id, ids["ep2"].atom_id]
    iep = coarsen(
        store,
        h.index,
        h.embedder,
        h.graph,
        support,
        target_scale="L2_meso",
        policy="rule",
        llm=None,
        at=store.now(),
        scope=scope,
        min_cohesion=h.config.graph.min_cohesion,
    )
    assert iep is not None
    project(h.graph, iep, at=store.now())
    _ev, sched = system_event(
        store, scope=scope, object_type="scheduler", event_type="scheduler", payload={"phases": ["consolidate"]}
    )
    skills = promote_skills(store, h.index, h.embedder, h.graph, scope=scope, at=store.now(), anchor_id=sched.anchor_id)
    assert len(skills) == 1
    skill_text = f"Skill: {S4_CONSTRAINT}"

    store.trace.reset()
    deep = h.compile(S4_QUERY, filters=scope4, mode="deep")
    assert any(t.startswith("interface_ep") for t in store.trace.tables())
    assert [s.text for s in deep.skills] == [skill_text]
    assert deep.retrieval["ablation"] == "axis_graph_support"
    failed_paths = [a for a in deep.audit if a.explain.get("reason") == "invariant:failed_path"]
    assert (
        failed_paths and failed_paths[0].item_id == f"{iep.atom_id}#inv:1" and failed_paths[0].source == "interface_ep"
    )
    assert failed_paths[0].anchor_ids == sorted({ids["anc_e1"], ids["anc_e2"]})
    # constraint 类不变量已经被选中了
    assert not [c for c in deep.constraints if "#inv:" in c.item_id]  # the constraint invariant is already selected
    assert [c.text for c in deep.constraints] == [S4_CONSTRAINT, S4_DC]
    assert {e.item_id for e in deep.episodes} == {*support, iep.atom_id}
    assert deep.to_pcir().ok() is True
    _assert_witnessed_and_disjoint(deep)

    store.trace.reset()
    axis_graph = h.compile(S4_QUERY, filters=scope4, mode="axis_graph")
    assert not any(t.startswith("interface_ep") for t in store.trace.tables())
    assert [s.text for s in axis_graph.skills] == [skill_text] and axis_graph.retrieval["ablation"] == "axis_graph"
    assert not [i for i in axis_graph.items() if "#inv:" in i.item_id]
    assert not [a for a in axis_graph.audit if str(a.explain.get("reason", "")).startswith(("invariant:", "residual:"))]

    store.trace.reset()
    fast = h.compile(S4_QUERY, filters=scope4, mode="fast")
    assert not any(t.startswith(("graph_", "interface_ep")) for t in store.trace.tables())
    assert [s.text for s in fast.skills] == [skill_text] and fast.retrieval["ablation"] == "axis"
    assert not [i for i in fast.items() if "#inv:" in i.item_id]
    assert "## Skills\n- Skill: Do not modify the subscription state machine." in fast.to_text()


# ---------------------------------------------------------------------------
# 51
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case", ["disputed", "conflict", "governance_selection", "governance_handbuilt", "revoked", "no_raise"]
)
def test_pcir_obligations(store, clock, index_provider, case):
    """Each obligation trips exactly as §15.4 says; require_witness=False reports instead of raising.
    每条义务恰按 §15.4 触发；require_witness=False 只报告而不抛出。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

        case: The obligation scenario.
        case：义务场景。

    Output / 输出：
        Assertions over compile() output and lower()/to_pcir().
        对 compile() 输出与 lower()/to_pcir() 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    scope_dict = {"user_id": "alice", "run_id": f"run-{case}"}
    scope = Scope(**scope_dict)

    if case in ("disputed", "no_raise"):
        atom = _seed_state(h, scope=scope, ts="2026-05-01T09:00:00Z", text="User lives in Paris.")
        store.dispute_atom_anchor(atom.atom_id, atom.anchor_ids[0], at=clock.now())
        ir = h.compile("where does the user live", filters=scope_dict)
        item = next(i for i in ir.facts if i.atom_id == atom.atom_id)
        assert item.anchor_ids == [atom.anchor_ids[0]]
        assert item.explain["disputed_anchor_ids"] == [atom.anchor_ids[0]]
        if case == "no_raise":
            pcir = ir.to_pcir(require_witness=False)
            assert pcir.ok() is False
            assert [str(o.name) for o in pcir.proof_obligations if str(o.status) == "violated"] == ["evidence_coverage"]
            assert lower(ir, require_witness=False).model_dump() == pcir.model_dump()
            return
        with pytest.raises(ProofObligationError) as exc:
            ir.to_pcir()
        assert exc.value.error_code == "PCIR_001"
        assert exc.value.details["violated"] == ["evidence_coverage"]
        assert exc.value.details["missing_witness"] == [item.item_id]
        assert set(exc.value.details) == {
            "violated",
            "missing_witness",
            "unresolved_conflicts",
            "revoked",
            "governance",
        }
        assert "require_witness=False" in exc.value.suggestion
        return

    if case == "conflict":
        paris = _seed_state(h, scope=scope, ts="2026-05-01T09:00:00Z", text="User lives in Paris.")
        berlin = _seed_state(h, scope=scope, ts="2026-04-01T09:00:00Z", text="User lives in Berlin.")
        edge_id = h.graph.insert_edge(
            src_id=paris.atom_id,
            dst_id=berlin.atom_id,
            edge_type="conflict",
            witness_anchor_ids=[berlin.anchor_ids[0]],
            created_by="ingest",
        )
        ir = h.compile("where does the user live", filters=scope_dict, mode="deep")
        assert ir.unresolved_conflicts == [edge_id]
        assert ir.next_action == "Resolve conflict between 'User lives in Paris.' and 'User lives in Berlin.'"
        with pytest.raises(ProofObligationError) as exc:
            ir.to_pcir()
        assert exc.value.details["violated"] == ["no_unresolved_conflict"]
        assert exc.value.details["unresolved_conflicts"] == [edge_id]
        assert h.compile("where does the user live", filters=scope_dict, mode="fast").unresolved_conflicts == []
        return

    if case == "governance_selection":
        private = _seed_state(h, scope=scope, ts="2026-05-01T09:00:00Z", text="User lives in Paris.")
        team = _seed_state(
            h, scope=scope, ts="2026-05-02T09:00:00Z", text="The team ships on Tuesdays.", visibility="team"
        )
        ir = h.compile("where does the user live", filters={**scope_dict, "visibility": "team"})
        assert ir.retrieval["audience"] == "team"
        assert {i.atom_id for i in ir.items() if i.atom_id} == {team.atom_id}
        assert private.atom_id not in {i.atom_id for i in ir.items()}
        pcir = ir.to_pcir()
        assert pcir.ok() is True
        assert {str(o.name): str(o.status) for o in pcir.proof_obligations}["governance"] == "satisfied"
        explicit = h.compile("where does the user live", filters=scope_dict, audience="team")
        assert {i.atom_id for i in explicit.items() if i.atom_id} == {team.atom_id}
        assert explicit.retrieval["dropped_governance"] >= 1 and explicit.to_pcir().ok() is True
        with pytest.raises(ValidationError):
            h.compile("q", filters=scope_dict, audience="bogus")
        return

    if case == "governance_handbuilt":
        anchor = _hand_anchor("anc_x", permission_scope="team")
        item = _item("atm_x", "facts", "private fact", anchors=["anc_x"], visibility="private")
        ir = ContextIR(
            query="q",
            as_of="2026-06-15T12:00:00Z",
            scope=Scope(user_id="alice"),
            facts=[item],
            audit=evidence_items([anchor]),
            retrieval={"audience": "team", "anchors": [anchor.model_dump()]},
        )
        pcir = lower(ir, require_witness=False)
        by_name = {str(o.name): o for o in pcir.proof_obligations}
        assert str(by_name["governance"].status) == "violated" and by_name["governance"].details["governance"] == [
            "atm_x"
        ]
        assert str(by_name["evidence_coverage"].status) == "satisfied"
        with pytest.raises(ProofObligationError) as exc:
            lower(ir)
        assert exc.value.details["violated"] == ["governance"] and exc.value.details["governance"] == ["atm_x"]
        # A κ constraint omitted from constraints ∪ audit also violates governance; a private anchor too.
        # constraints ∪ audit 中遗漏的 κ 约束同样违反 governance；私有锚点亦然。
        ep = item_from_atom(
            _episode_atom("atm_ep", a="a", y="y", r="positive", constraints=["Keep X."], ts="2026-06-01T00:00:00Z"),
            slot="episodes",
            score=0.0,
            anchor_ids=["anc_x"],
            source="axis",
            hops=0,
            explain=None,
        ).model_copy(update={"visibility": "team"})
        ir2 = ContextIR(
            query="q",
            as_of="2026-06-15T12:00:00Z",
            scope=Scope(user_id="alice"),
            episodes=[ep],
            retrieval={"audience": "team", "anchors": [anchor.model_dump()]},
        )
        assert lower(ir2, require_witness=False).ok() is False
        ir3 = ir2.model_copy(
            update={"constraints": [_item("atm_k", "constraints", "Keep X.", anchors=["anc_x"], visibility="team")]}
        )
        assert lower(ir3, require_witness=False).ok() is True
        private_anchor = _hand_anchor("anc_x", permission_scope="private")
        ir4 = ir3.model_copy(update={"retrieval": {"audience": "team", "anchors": [private_anchor.model_dump()]}})
        assert [
            str(o.name) for o in lower(ir4, require_witness=False).proof_obligations if str(o.status) == "violated"
        ] == ["governance"]
        assert (
            str(
                lower(ir3.model_copy(update={"retrieval": {"anchors": [anchor.model_dump()]}}), require_witness=False)
                .proof_obligations[2]
                .status
            )
            == "skipped"
        )
        return

    assert case == "revoked"
    atom = _seed_state(h, scope=scope, ts="2026-05-01T09:00:00Z", text="User lives in Paris.")
    store.revoke_anchor(atom.anchor_ids[0], at=clock.now())
    ir = h.compile("where does the user live", filters=scope_dict)
    evidence = [a for a in ir.audit if a.explain.get("kind") == "evidence"]
    assert evidence and evidence[0].explain["revoked_at"] == clock.now()
    with pytest.raises(ProofObligationError) as exc:
        ir.to_pcir()
    assert exc.value.details["violated"] == ["not_revoked"]
    assert exc.value.details["revoked"] == [atom.anchor_ids[0]]
    # Disputing the revoked anchor takes the item out of the card entirely (no undisputed witness → coverage).
    # 对已撤销锚点提出争议会使该条目彻底离开卡片（无未争议见证 → coverage）。
    store.dispute_atom_anchor(atom.atom_id, atom.anchor_ids[0], at=clock.now())
    with pytest.raises(ProofObligationError) as exc2:
        h.compile("where does the user live", filters=scope_dict).to_pcir()
    assert exc2.value.details["violated"] == ["evidence_coverage"] and exc2.value.details["revoked"] == []


# ---------------------------------------------------------------------------
# Renderer / next-action / invariant unit tests
# 渲染器 / 下一步动作 / 不变量的单元测试
# ---------------------------------------------------------------------------
def test_render_card_and_next_action_rules():
    """render_card emits optional sections only when non-empty and '- (none)' otherwise; next-action rules (1)-(5).
    render_card 仅在非空时输出可选块、否则输出 '- (none)'；下一步规则 (1)-(5)。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over render_card, derive_next_action and invariant_items.
        对 render_card、derive_next_action 与 invariant_items 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    anchor = _hand_anchor("anc_a", ts="2026-06-01T00:00:00Z")
    ir = ContextIR(query="what now", as_of="2026-06-15T12:00:00Z", scope=Scope(user_id="alice", agent_id="bot"))
    assert render_card(ir) == "\n".join(
        [
            "# Work State Card",
            "Scope: user_id=alice agent_id=bot | As of: 2026-06-15T12:00:00.000000+00:00 | Mode: auto",
            "## Current Goal",
            "what now",
            "## Valid Facts",
            "- (none)",
            "## Invalidated Assumptions",
            "- (none)",
            "## Failed Paths",
            "- (none)",
            "## Constraints",
            "- (none)",
            "## Next Action",
            "none pending",
            "## Evidence",
            "- (none)",
        ]
    )
    neg = _item(
        "atm_neg",
        "episodes",
        "[negative] g → a1 → y1",
        time_anchor="2026-06-01T00:00:00Z",
        explain={"r": "negative", "a": "a1", "y": "y1", "u": "g", "kappa": []},
    )
    pos = _item(
        "atm_pos",
        "episodes",
        "[positive] g → a2 → y2",
        time_anchor="2026-06-02T00:00:00Z",
        explain={"r": "positive", "a": "a2", "y": "y2", "u": "g", "kappa": []},
    )
    unknown = _item(
        "atm_unk",
        "episodes",
        "[unknown] g → a3 → ?",
        time_anchor="2026-06-03T00:00:00Z",
        explain={"r": "unknown", "a": "a3", "y": "", "u": "g", "kappa": []},
    )
    procedural = _item("atm_proc", "episodes", "# How to deploy", time_anchor=None)
    archived = _item("atm_old", "audit", "Old fact.", explain={"reason": "archived:deleted"})
    revised = _item("atm_rev", "audit", "Older fact.", explain={"reason": "revised", "superseded_by_text": "New fact."})
    full = ContextIR(
        query="q",
        goal="ship it",
        as_of="2026-06-15T12:00:00Z",
        scope=Scope(user_id="alice"),
        mode="deep",
        facts=[_item("atm_f", "facts", "A fact.")],
        episodes=[neg, pos, unknown, procedural],
        preferences=[_item("atm_p", "preferences", "Likes tea.")],
        constraints=[
            _item("atm_c", "constraints", "Do not re-apply 'x': y"),
            _item("atm_c2", "constraints", "Keep calm."),
        ],
        skills=[_item("atm_s", "skills", "Skill: Keep calm.")],
        audit=[*evidence_items([anchor]), archived, revised],
        next_action="Continue from 'a2' (last outcome: y2) while keeping constraints: Keep calm.",
    )
    card = render_card(full)
    assert card.split("\n") == [
        "# Work State Card",
        "Scope: user_id=alice | As of: 2026-06-15T12:00:00.000000+00:00 | Mode: deep",
        "## Current Goal",
        "ship it",
        "## Valid Facts",
        "- A fact.",
        "- a2 → y2 [positive]",
        "- a3 → ? [unknown]",
        "- # How to deploy",
        "## Invalidated Assumptions",
        "- Old fact. (archived: deleted)",
        "- Older fact. -> superseded by: New fact.",
        "## Failed Paths",
        "- a1 → y1",
        "## Constraints",
        "- Do not re-apply 'x': y",
        "- Keep calm.",
        "## Preferences",
        "- Likes tea.",
        "## Skills",
        "- Skill: Keep calm.",
        "## Next Action",
        "Continue from 'a2' (last outcome: y2) while keeping constraints: Keep calm.",
        "## Evidence",
        f"- {anchor.evidence_line()}",
    ]
    assert full.to_text() == card
    pcir = lower(full, require_witness=False)
    assert [e["text"] for e in pcir.payload["invalidated"]] == ["Old fact.", "Older fact."]
    assert pcir.payload["invalidated"][1]["superseded_by_text"] == "New fact."

    kept = "while keeping constraints: Keep calm."
    cons = full.constraints
    assert derive_next_action(episodes=[neg, pos], constraints=cons, residuals=[], conflicts=[], goal="g") == (
        f"Continue from 'a2' (last outcome: y2) {kept}"
    )
    assert derive_next_action(episodes=[neg], constraints=cons, residuals=[], conflicts=[], goal="g") == (
        f"Do not re-apply 'a1'; investigate 'y1' {kept}"
    )
    newer_neg = neg.model_copy(update={"time_anchor": "2026-06-09T00:00:00.000000+00:00"})
    assert derive_next_action(episodes=[pos, newer_neg], constraints=[], residuals=[], conflicts=[], goal="g") == (
        "Do not re-apply 'a1'; investigate 'y1'"
    )
    assert (
        derive_next_action(episodes=[unknown], constraints=[], residuals=["open question"], conflicts=[], goal="g")
        == "Resolve: open question"
    )
    assert derive_next_action(episodes=[], constraints=[], residuals=[], conflicts=[("x", "y")], goal="g") == (
        "Resolve conflict between 'x' and 'y'"
    )
    assert (
        derive_next_action(episodes=[], constraints=[], residuals=[], conflicts=[], goal="ship it")
        == "Continue: ship it"
    )
    many = [_item(f"atm_k{i}", "constraints", f"Rule {i}.") for i in range(5)]
    assert derive_next_action(episodes=[], constraints=many, residuals=[], conflicts=[], goal="g") == "Continue: g"
    assert derive_next_action(episodes=[pos], constraints=many, residuals=[], conflicts=[], goal="g").endswith(
        "while keeping constraints: Rule 0.; Rule 1.; Rule 2."
    )

    iep = InterfaceEP(
        interface_ep_id="iep_x",
        slots=Slots(u="g", a="a1 → a2", y="y2", r="positive", kappa=Kappa(constraints=["Keep calm."])),
        slot_support={"u": ["n1"], "a": ["n1", "n2"], "y": ["n2"], "r": ["n2"], "kappa": ["n1"]},
        support_node_ids=["n1", "n2"],
        invariants=[
            Invariant(text="Keep calm.", kind="constraint", support_node_ids=["n1", "n2"], anchor_ids=["anc_a"]),
            Invariant(text="Be kind.", kind="constraint", support_node_ids=["n1", "n2"], anchor_ids=["anc_a"]),
            Invariant(text="do not: a1", kind="failed_path", support_node_ids=["n1"], anchor_ids=["anc_a"]),
        ],
        residuals=[Residual(text="outcome missing", kind="missing_outcome", support_node_ids=["n2"])],
        evidence_anchor_ids=["anc_a"],
        scope=Scope(user_id="alice"),
    )
    constraints, audit = invariant_items(iep, "atm_iep", {"keep calm"})
    assert [c.item_id for c in constraints] == ["atm_iep#inv:1"] and constraints[0].text == "Be kind."
    assert [a.item_id for a in audit] == ["atm_iep#inv:2", "atm_iep#inv:3"]
    assert [a.explain["reason"] for a in audit] == ["invariant:failed_path", "residual:missing_outcome"]
    # 没有锚点的 residual 回退到 InterfaceEP 的证据
    assert audit[1].anchor_ids == ["anc_a"]  # residual without anchors falls back to the IEP evidence
    assert all(i.source == "interface_ep" and i.explain["interface_ep_id"] == "iep_x" for i in constraints + audit)
    ev = evidence_items([_hand_anchor("anc_z", ts="2026-06-03T00:00:00Z"), anchor, anchor])
    assert [e.item_id for e in ev] == ["anc:anc_a", "anc:anc_z"] and ev[0].slot == "audit"


# ---------------------------------------------------------------------------
# §19.6 `mid` by slot content: a revised κ constraint is an audit item, never re-synthesized as '#kappa:'
# §19.6 `mid` 按 slot 内容判定：被修订的 κ 约束是 audit 项，绝不会重新合成为 '#kappa:'
# ---------------------------------------------------------------------------
def test_revised_constraint_suppresses_kappa(store, clock, events, scope4, index_provider):
    """After archiving the background constraint with a successor + revision edge, compile() without as_of matches §19.6 `mid`.
    将背景约束归档并给出后继 + revision 边后，不带 as_of 的 compile() 与 §19.6 的 `mid` 一致。

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
        Slot-content assertions (never a golden: the card carries the clock instant).
        槽位内容断言（绝不比对黄金：卡片携带时钟时刻）。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, index_provider)
    ids = _build_s4(h, events, scope4, fix=True)
    scope = Scope(**scope4)
    clock.tick(seconds=60)
    now = clock.now()
    successor_text = "retry.ts may change once billing_e2e is isolated."
    fb_event = SourceEvent(
        scope=scope,
        object_type="feedback",
        event_type="feedback",
        payload={"target_id": ids["uc"], "feedback": "state_outdated", "reason": None, "correction": successor_text},
        timestamp=now,
    )
    stored, _dedup = store.insert_source_event(fb_event)
    fb_anchor = store.get_or_create_anchor(stored)
    old = store.get_atom(ids["uc"])
    with store.transaction():
        store.archive_atom(old.atom_id, reason="feedback", at=now, valid_to=now)
        new = create_atom(
            store,
            h.index,
            h.embedder,
            MemoryAtom(
                memory_id=old.memory_id,
                version=2,
                atom_type="state",
                content={"text": successor_text},
                text=successor_text,
                time_anchor=now,
                valid_from=now,
                scale_level="state",
                subject="project",
                predicate="must_not",
                obs_type="constraint",
                source_kind="feedback",
                source_id="fb_hand",
                extracted_by="test",
                scope=scope,
                anchor_ids=[fb_anchor.anchor_id, old.anchor_ids[0]],
            ),
        )
        h.graph.insert_edge(
            src_id=old.atom_id,
            dst_id=new.atom_id,
            edge_type="revision",
            witness_anchor_ids=[fb_anchor.anchor_id],
            created_by="feedback",
        )
    clock.tick(seconds=60)
    mid = h.compile(S4_QUERY, filters=scope4)
    assert mid.as_of == store.now()
    assert [c.text for c in mid.constraints] == [S4_DC, successor_text]
    revised = [a for a in mid.audit if a.explain.get("reason") == "revised"]
    assert len(revised) == 1 and revised[0].text == S4_CONSTRAINT
    assert revised[0].explain["superseded_by_text"] == successor_text and revised[0].atom_id == old.atom_id
    assert not [c for c in mid.constraints if "#kappa:" in c.item_id]
    assert mid.episodes[0].explain["kappa"] == [S4_CONSTRAINT]
    assert f"- {S4_CONSTRAINT} -> superseded by: {successor_text}" in mid.to_text()
    assert mid.to_pcir().ok() is True
    _assert_witnessed_and_disjoint(mid)
    # Once the successor is gone too, the κ text is synthesized exactly once for the two episodes.
    # 当后继也消失之后，κ 文本会为这两条 episodes 恰好合成一次。
    with store.transaction():
        store.archive_atom(new.atom_id, reason="deleted", at=clock.now(), valid_to=clock.now())
    clock.tick(seconds=60)
    later = h.compile(S4_QUERY, filters=scope4, mode="fast")
    kappa = [c for c in later.constraints if "#kappa:" in c.item_id]
    assert [c.text for c in kappa] == [S4_CONSTRAINT] and kappa[0].item_id == f"{ids['ep1'].atom_id}#kappa:0"
    assert kappa[0].anchor_ids == later.episodes[0].anchor_ids and kappa[0].source == "system"
