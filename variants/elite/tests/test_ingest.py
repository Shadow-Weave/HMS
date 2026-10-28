"""Group C tests: events/anchors, payload normalization, routing, structured closure, raw/procedural/LLM paths, revisions.
C 组测试：事件/锚点、载荷归一化、分流、结构化闭包、原始/过程/LLM 路径与修订。
"""

# Purpose (EN): Assert CONTRACT §12 (and §18.3 tests 24–32) over the ingest pipeline driven directly through Ingestor.
# 目的（中）：直接通过 Ingestor 驱动摄取流水线，断言契约 §12（以及 §18.3 测试 24–32）。

# Input (EN): tests/fixtures/events/*.json and the shared conftest fixtures (store, clock, ids, index_provider, ...).
# 输入（中）：tests/fixtures/events/*.json 与共享的 conftest 夹具（store、clock、ids、index_provider……）。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any

import pytest

from hms_elite import ingest
from hms_elite.configs.base import HMSConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.enums import Decision
from hms_elite.exceptions import EmbeddingError, LLMError, ValidationError
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.ingest.atoms import atom_from_observation, create_atom
from hms_elite.ingest.events import anchor_events, events_from_dicts, events_from_messages
from hms_elite.ingest.pipeline import Ingestor, IngestResult
from hms_elite.ingest.router import route_event
from hms_elite.llms.rule import RuleLLM
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import Observation, Scope
from hms_elite.timeutil import next_day_midnight
from tests.conftest import FrozenClock, SequentialIds

S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP1_Y = "billing_e2e failed: subscription_state_regression"
S4_EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
PLANE_TABLES = (
    "observations",
    "eps",
    "memory_atoms",
    "atom_anchor",
    "entities",
    "atom_entity",
    "atom_topic",
    "atom_vectors",
    "graph_nodes",
    "graph_edges",
    "history",
    "transitions",
)


class _Harness:
    """One store + index + embedder + LLM + config wired into an Ingestor, with add()/add_events() helpers.
    一组 store + 索引 + 嵌入器 + LLM + 配置接入 Ingestor，并提供 add()/add_events() 辅助方法。
    """

    def __init__(
        self,
        store: StateStore,
        clock: FrozenClock,
        llm: Any,
        *,
        index_provider: str = "memory",
        embedder: Any = None,
        llm_config: dict | None = None,
        **config: Any,
    ) -> None:
        """Wire the collaborators (`llm_config` is the HMSConfig `llm` envelope, e.g. the vision switch).
        接线协作对象（`llm_config` 是 HMSConfig 的 `llm` 信封，如视觉开关）。
        """
        self.store = store
        self.clock = clock
        self.llm = llm
        self.embedder = embedder or HashEmbedder({"embedding_dims": 64})
        envelope = {"llm": llm_config} if llm_config is not None else {}
        self.config = HMSConfig(
            vector_store={"provider": index_provider}, store={"path": ":memory:"}, **envelope, **config
        )
        if index_provider == "sqlite":
            self.index = SQLiteIndex(store=store, dims=self.embedder.dims)
        else:
            self.index = MemoryIndex(dims=self.embedder.dims)
        # This harness exercises the full extraction contract (scripted replies carry every observation field, the
        # call log is asserted against `memory_extraction`), so it pins extraction_output='full' explicitly rather
        # than relying on the library default ('lines' since 2026-09-18, CONTRACT §6.6 / changelog row 108 — the pin
        # keeps this test on the full contract);
        # the compact contract is covered by tests/test_extraction_compact.py.
        # 本夹具练习的是完整抽取契约（脚本化回复带全部观察字段，调用日志按 `memory_extraction` 断言），因此显式钉住
        # extraction_output='full' 而不依赖库默认（2026-09-18 起为 'lines'，CONTRACT §6.6 / 变更记录第 108 行——钉住使本
        # 测试留在完整契约上）；紧凑契约由
        # tests/test_extraction_compact.py 覆盖。
        self.ingestor = Ingestor(store, self.index, self.embedder, llm, self.config, extraction_output="full")

    def add(
        self,
        messages: Any,
        *,
        timestamp: str | None = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
        metadata: dict | None = None,
        expiration_date: str | None = None,
        **scope: str,
    ) -> IngestResult:
        """Mem0 add() minus the facade: coerce, tick the clock, ingest_messages.
        去掉门面的 Mem0 add()：规整消息、推进时钟、ingest_messages。
        """
        self.clock.tick(seconds=60)
        return self.ingestor.ingest_messages(
            coerce_messages(messages),
            scope=Scope(**scope),
            user_meta=metadata or {},
            infer=infer,
            memory_type=memory_type,
            prompt=prompt,
            event_time=timestamp,
            valid_to=next_day_midnight(expiration_date) if expiration_date else None,
        )

    def add_events(self, items: list, *, infer: bool = True, **scope: str) -> IngestResult:
        """HMS add_events() minus the facade: events_from_dicts, tick the clock, ingest_events.
        去掉门面的 HMS add_events()：events_from_dicts、推进时钟、ingest_events。
        """
        self.clock.tick(seconds=60)
        events = events_from_dicts(
            items, scope_fallback=Scope(**scope), visibility_default="private", event_time=self.clock.now()
        )
        return self.ingestor.ingest_events(events, infer=infer)


def _fresh() -> tuple[StateStore, FrozenClock]:
    """A fresh in-memory store with its own frozen clock and sequential ids.
    带独立冻结时钟与顺序 id 的全新内存 store。
    """
    clock = FrozenClock()
    return StateStore(":memory:", now=clock.now, id_factory=SequentialIds()), clock


def _anchor_of(res: IngestResult, object_type: str, nth: int = 0) -> str:
    """The anchor id of the nth event of a type in an IngestResult.
    IngestResult 中某类型第 nth 个事件的锚点 id。
    """
    return [e["anchor_id"] for e in res.events if e["object_type"] == object_type][nth]


def _seed_constraint(store: StateStore, index: Any, embedder: Any, scope4: dict) -> Any:
    """The test-27 setup: anchor the constraint message and create its constraint atom without any LLM.
    测试 27 的准备：为约束消息落锚并在不使用 LLM 的情况下创建其约束原子。
    """
    scope = Scope(**scope4)
    ((ev, anc, _dedup),) = anchor_events(
        store,
        events_from_messages(
            [{"role": "user", "content": S4_CONSTRAINT}],
            scope=scope,
            event_time="2026-06-15T09:00:00Z",
            user_meta={},
            visibility="private",
        ),
    )
    assert anc.source_ref == "chat://agent_id=coder&run_id=run-billing&user_id=alice/0"
    observation = Observation(
        type="constraint",
        subject="project",
        predicate="must_not",
        objects=["subscription state machine"],
        relation="constraint",
        text=S4_CONSTRAINT,
        date_observed="2026-06-15",
        anchor_ids=[anc.anchor_id],
        scope=scope,
    )
    return create_atom(
        store, index, embedder, atom_from_observation(observation, scope=scope, time_anchor=ev.timestamp)
    )


class _FlakyEmbedder(HashEmbedder):
    """A hash embedder that raises EmbeddingError for one poisoned text.
    对某个特定文本抛出 EmbeddingError 的哈希嵌入器。
    """

    poison = "POISON"

    def embed(self, text: str, memory_action: Any = None) -> list[float]:
        """Embed, or raise when the text is the poisoned one.
        嵌入；文本为毒化文本时抛出。
        """
        if self.poison in text:
            raise EmbeddingError("poisoned text", "EMBED_001", details={"text": text})
        return super().embed(text, memory_action)


# ---------------------------------------------------------------------------
# 24
# ---------------------------------------------------------------------------
def test_one_event_one_anchor_and_dedup_at_same_event_time(store, clock, events, index_provider):
    """N messages → N events/anchors; identical re-add at the same event time dedups (NONE, mentions bump); later time adds a support anchor.
    N 条消息 → N 个事件/锚点；同一事件时间的相同重加去重（NONE，提及计数增加）；更晚时间则新增支持锚点。

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
        Assertions over events, anchors, rows, atom_anchor rows and graph_nodes.mentions.
        对事件、锚点、行、atom_anchor 行与 graph_nodes.mentions 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    h = _Harness(store, clock, RuleLLM(), index_provider=index_provider)
    scope = {"user_id": "alice", "run_id": "run-coffee"}
    both = h.add(events("chat_preference").messages, **scope)
    assert len(both.events) == 2 and store.stats()["source_events"] == 2 and store.stats()["anchors"] == 2
    assert [e["dedup"] for e in both.events] == [False, False]
    assert [e["event"] for e in both.rows] == ["ADD", "ADD"]

    latte = "I like oat-milk lattes in the morning."
    s2 = {"user_id": "bob", "run_id": "run-dedup"}
    first = h.add(latte, timestamp="2026-04-02T08:00:00Z", **s2)
    assert first.rows[0]["event"] == "ADD"
    atom_id = first.rows[0]["atom_id"]
    anc_1 = first.events[0]["anchor_id"]
    assert store.atom_anchor_ids(atom_id) == [anc_1]
    anchors_before = store.stats()["anchors"]
    atom_anchor_rows_before = store.stats()["atom_anchor"]

    again = h.add(latte, timestamp="2026-04-02T08:00:00Z", **s2)
    assert again.events[0]["dedup"] is True and again.events[0]["anchor_id"] == anc_1
    assert store.stats()["anchors"] == anchors_before
    assert [r["event"] for r in again.rows] == ["NONE"] and again.rows[0]["atom_id"] == atom_id
    assert store.atom_anchor_ids(atom_id) == [anc_1]
    assert store.stats()["atom_anchor"] == atom_anchor_rows_before
    assert store.get_node(atom_id).mentions == 2

    later = h.add(latte, timestamp="2026-04-03T08:00:00Z", **s2)
    anc_2 = later.events[0]["anchor_id"]
    assert later.events[0]["dedup"] is False and anc_2 != anc_1
    assert store.stats()["anchors"] == anchors_before + 1
    assert [r["event"] for r in later.rows] == ["NONE"] and later.rows[0]["atom_id"] == atom_id
    assert store.atom_anchor_ids(atom_id) == [anc_1, anc_2]
    assert store.get_node(atom_id).mentions == 3
    assert store.count_atoms(Scope(**s2)) == 1


# ---------------------------------------------------------------------------
# 25
# ---------------------------------------------------------------------------
def test_payload_normalization_and_internal_types_rejected():
    """§4.3 shapes through events_from_dicts: aliases, derived status, defaults, sorting, VALIDATION_007 / _005 / _001.
    通过 events_from_dicts 检验 §4.3 形状：别名、派生状态、默认值、排序以及 VALIDATION_007 / _005 / _001。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over normalized SourceEvents and error codes.
        对归一化 SourceEvent 与错误码的断言。

    State effect / 状态影响：
        None.
        无。
    """
    fallback = Scope(user_id="alice", agent_id="coder", run_id="run-billing")
    items = [
        {
            "object_type": "test_log",
            "timestamp": "2026-06-15T10:04:00Z",
            "payload": {
                "suite": "billing_e2e",
                "passed": 11,
                "failed": 3,
                "failed_tests": ["subscription_state_regression"],
            },
        },
        {
            "object_type": "tool_call",
            "timestamp": "2026-06-15T10:00:00Z",
            "payload": {"tool": "pytest", "arguments": {"k": "billing"}, "result": {"ok": True}, "status": "ok"},
        },
        {
            "object_type": "sensor_trace",
            "timestamp": "2026-06-15T09:00:00Z",
            "payload": {"sensor": "temp-1", "samples": [{"t": "2026-06-15T08:59:00Z", "v": 21.5}]},
        },
        {"object_type": "file_diff", "timestamp": "2026-06-15T09:30:00Z", "payload": {"path": "a.py", "patch": "+x"}},
        {
            "object_type": "message",
            "role": "user",
            "timestamp": "2026-06-15T09:30:00Z",
            "payload": {"text": "hi there"},
        },
    ]
    out = events_from_dicts(items, scope_fallback=fallback, visibility_default="team")
    assert [e.object_type for e in out] == ["sensor_trace", "file_diff", "message", "tool_call", "test_log"]
    by_type = {e.object_type: e for e in out}
    log = by_type["test_log"]
    assert log.payload == {
        "suite": "billing_e2e",
        "status": "failed",
        "passed": 11,
        "failed": 3,
        "failures": [{"name": "subscription_state_regression", "message": None}],
        "log": None,
        "related_paths": [],
    }
    assert log.event_type == "tool_result" and log.role == "tool" and log.raw_ref == "test://billing_e2e"
    assert log.scope == fallback and log.visibility == "team"
    call = by_type["tool_call"]
    assert call.payload["args"] == {"k": "billing"} and call.payload["result"] == '{"ok": true}'
    assert call.event_type == "tool_result" and call.raw_ref == "tool://pytest"
    trace = by_type["sensor_trace"]
    assert trace.event_type == "tool_result" and trace.raw_ref == "sensor://temp-1"
    assert trace.payload["samples"] == [{"t": "2026-06-15T08:59:00.000000+00:00", "v": 21.5}]
    diff = by_type["file_diff"]
    assert diff.event_type == "created" and diff.payload["constraints"] == [] and diff.raw_ref == "file://a.py"
    assert by_type["message"].raw_ref == "chat://agent_id=coder&run_id=run-billing&user_id=alice/4"
    # explicit event_type and scope keys on the dict are kept
    # dict 上显式给出的 event_type 与 scope 键会被保留
    own = events_from_dicts(
        [{"object_type": "test_log", "event_type": "created", "user_id": "zoe", "payload": {"suite": "s"}}],
        scope_fallback=None,
        visibility_default="private",
        event_time="2026-06-15T09:00:00Z",
    )[0]
    assert own.event_type == "created" and own.scope == Scope(user_id="zoe") and own.payload["status"] == "passed"
    assert own.timestamp == "2026-06-15T09:00:00.000000+00:00"

    with pytest.raises(ValidationError) as internal:
        events_from_dicts(
            [{"object_type": "feedback", "payload": {"target_id": "x", "feedback": "state_wrong"}}],
            scope_fallback=fallback,
            visibility_default="private",
        )
    assert internal.value.error_code == "VALIDATION_007"
    assert internal.value.details["index"] == 0 and internal.value.details["object_type"] == "feedback"
    with pytest.raises(ValidationError) as malformed:
        events_from_dicts(
            [
                {"object_type": "test_log", "payload": {"suite": "s"}},
                {"object_type": "file_diff", "payload": {"patch": "+x"}},
            ],
            scope_fallback=fallback,
            visibility_default="private",
        )
    assert malformed.value.error_code == "VALIDATION_005" and malformed.value.details["index"] == 1
    assert malformed.value.details["field"] == "path"
    with pytest.raises(ValidationError) as unscoped:
        events_from_dicts(
            [{"object_type": "test_log", "payload": {"suite": "s"}}], scope_fallback=None, visibility_default="private"
        )
    assert unscoped.value.error_code == "VALIDATION_001"


# ---------------------------------------------------------------------------
# 26
# ---------------------------------------------------------------------------
def test_router_observation_vs_state(store, clock, events):
    """chat_preference → observations only; chat_action_closure → an observation and an EP; system role → anchored, not extracted.
    chat_preference → 仅观察；chat_action_closure → 一个观察加一个 EP；system 角色 → 落锚但不抽取。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

    Output / 输出：
        Assertions over route_event and the atoms/EPs produced.
        对 route_event 与产出的原子/EP 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scope = Scope(user_id="alice", run_id="run-route")
    pref = events_from_messages(
        events("chat_preference").messages,
        scope=scope,
        event_time="2026-04-02T08:00:00Z",
        user_meta={},
        visibility="private",
    )
    assert [route_event(ev) for ev in pref] == [{"observation"}, {"observation"}]
    closure = events_from_messages(
        events("chat_action_closure").messages,
        scope=scope,
        event_time="2026-06-15T10:04:00Z",
        user_meta={},
        visibility="private",
    )
    assert route_event(closure[0]) == {"observation", "state"}
    system = events_from_messages(
        [{"role": "system", "content": "You are a helpful assistant."}],
        scope=scope,
        event_time="2026-06-15T10:04:00Z",
        user_meta={},
        visibility="private",
    )
    assert route_event(system[0]) == set()
    forced = events_from_messages(
        [{"role": "user", "content": "Please remember this."}], scope=scope, event_time="2026-06-15T10:04:00Z",
        user_meta={"route": "state"}, visibility="private",
    )  # fmt: skip
    assert route_event(forced[0]) == {"observation", "state"}
    structured = events_from_dicts(
        [{"object_type": "file_diff", "payload": {"path": "a.py", "patch": "+x"}}],
        scope_fallback=scope,
        visibility_default="private",
    )
    assert route_event(structured[0]) == {"structured"}
    assert ingest.route_event is route_event and ingest.has_action_closure("I patched retry.ts and the tests failed.")
    assert not ingest.has_action_closure("I like oat-milk lattes in the morning.")

    h = _Harness(store, clock, RuleLLM())
    res = h.add(events("chat_preference").messages, user_id="alice", run_id="run-route")
    assert res.ep_ids == [] and store.list_eps(scope) == []
    assert {r["atom_type"] for r in res.rows} == {"state"}
    res2 = h.add(
        events("chat_action_closure").messages, timestamp="2026-06-15T10:04:00Z", user_id="alice", run_id="run-route"
    )
    kinds = [(r["atom_type"], r["hms"]["source_kind"]) for r in res2.rows]
    assert ("state", "observation") in kinds and ("episode", "ep") in kinds
    assert len(res2.ep_ids) == 1
    ep = store.get_ep(res2.ep_ids[0])
    assert (ep.slots.a, ep.slots.y, ep.slots.r, ep.slots.u) == (
        "patched retry.ts to add backoff",
        "billing_e2e failed",
        "negative",
        "the webhook kept timing out",
    )
    assert ep.atom_id and store.get_atom(ep.atom_id).text.startswith("[negative] the webhook kept timing out")
    assert ep.time_anchor == "2026-06-15T10:04:00.000000+00:00"
    assert [t.source_id for t in store.list_transitions(scope)] == [ep.ep_id]
    observation = next(r for r in res2.rows if r["atom_type"] == "state")
    assert (
        observation["memory"].startswith("Assistant patched retry.ts") and observation["attributed_to"] == "assistant"
    )
    before = store.stats()
    res3 = h.add([{"role": "system", "content": "You are a helpful assistant."}], user_id="alice", run_id="run-route")
    after = store.stats()
    assert res3.rows == [] and len(res3.events) == 1
    assert after["source_events"] == before["source_events"] + 1 and after["anchors"] == before["anchors"] + 1
    assert after["memory_atoms"] == before["memory_atoms"] and after["observations"] == before["observations"]


# ---------------------------------------------------------------------------
# 27
# ---------------------------------------------------------------------------
def test_structured_closure_builds_eps_without_llm(store, clock, events, scope4, index_provider):
    """Structured events close into EPs, derived constraints and use/distill/temporal/causal edges with zero LLM calls.
    结构化事件在零 LLM 调用下闭合为 EP、派生约束以及 use/distill/temporal/causal 边。

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
        Assertions over EPs, derived constraints, edges, transitions, cross-batch closure and the empty LLM call log.
        对 EP、派生约束、边、转移、跨批闭包以及空 LLM 调用日志的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scripted = ScriptedLLM({"responses": []})
    h = _Harness(store, clock, scripted, index_provider=index_provider)
    scope = Scope(**scope4)
    uc = _seed_constraint(store, h.index, h.embedder, scope4)
    fixture = events("structured_billing")

    ra = h.add_events(fixture.events_fail, **scope4)
    assert scripted.calls == []
    anc_e1, anc_e2 = _anchor_of(ra, "file_diff"), _anchor_of(ra, "test_log")
    assert [e["dedup"] for e in ra.events] == [False, False]
    assert len(ra.ep_ids) == 1 and len(ra.edge_ids) == 2 and len(ra.constraint_ids) == 1
    ep1 = store.get_ep(ra.ep_ids[0])
    assert (ep1.slots.u, ep1.slots.b, ep1.slots.a, ep1.slots.y, ep1.slots.r) == (
        "fix checkout webhook retry",
        S4_CONSTRAINT,
        S4_EP1_A,
        S4_EP1_Y,
        "negative",
    )
    assert ep1.anchor_ids == [anc_e2, anc_e1] and ep1.slots.kappa.evidence == ep1.anchor_ids
    assert ep1.slots.kappa.constraints == [S4_CONSTRAINT] and ep1.slots.kappa.visibility == "private"
    assert ep1.time_anchor == "2026-06-15T10:04:00.000000+00:00" and ep1.extracted_by == "rule@closure-v1"
    ep1_atom = store.get_atom(ep1.atom_id)
    assert ep1_atom.text == f"[negative] fix checkout webhook retry → {S4_EP1_A} → {S4_EP1_Y}"
    assert set(ep1_atom.entity_ids) == {"retry.ts", "billing_e2e", "subscription_state_regression"}
    texts = {r["memory"]: r for r in ra.rows}
    assert "file_diff src/billing/retry.ts: add exponential backoff" in texts
    assert "test_log billing_e2e: failed (11 passed, 3 failed): subscription_state_regression" in texts
    dc = store.get_atom(ra.constraint_ids[0])
    assert dc.text == f"Do not re-apply '{S4_EP1_A}': {S4_EP1_Y}"
    assert (dc.obs_type, dc.subject, dc.predicate) == ("constraint", "project", "must_not")
    assert dc.anchor_ids == [anc_e2, anc_e1] and "retry.ts" in dc.entity_ids and dc.time_anchor == ep1.time_anchor
    edges = {(e.edge_type, e.src_id, e.dst_id): e for e in store.list_edges(scope)}
    use = edges[("use", ep1.atom_id, uc.atom_id)]
    assert use.witness_anchor_ids == [anc_e1] and use.confidence == 0.8 and use.proposer == "closure"
    distill = edges[("distill", ep1.atom_id, dc.atom_id)]
    assert distill.witness_anchor_ids == [anc_e2] and distill.confidence == 0.9 and distill.created_by == "ingest"
    assert [t.source_id for t in store.list_transitions(scope)] == [ep1.ep_id]

    rb = h.add_events(fixture.events_fix, **scope4)
    assert scripted.calls == []
    anc_e3, anc_e4 = _anchor_of(rb, "file_diff"), _anchor_of(rb, "test_log")
    assert len(rb.ep_ids) == 1 and len(rb.edge_ids) == 4 and rb.constraint_ids == []
    ep2 = store.get_ep(rb.ep_ids[0])
    assert (ep2.slots.a, ep2.slots.y, ep2.slots.r) == (S4_EP2_A, "billing_e2e passed", "positive")
    assert ep2.slots.b == f"{S4_CONSTRAINT} ; {dc.text}" and ep2.slots.kappa.constraints == [S4_CONSTRAINT, dc.text]
    assert ep2.anchor_ids == [anc_e4, anc_e3]
    edges = {(e.edge_type, e.src_id, e.dst_id): e for e in store.list_edges(scope)}
    assert edges[("use", ep2.atom_id, uc.atom_id)].witness_anchor_ids == [anc_e3]
    assert edges[("use", ep2.atom_id, dc.atom_id)].witness_anchor_ids == [anc_e3]
    temporal = edges[("temporal", ep1.atom_id, ep2.atom_id)]
    assert temporal.witness_anchor_ids == [anc_e4] and temporal.confidence == 0.5
    causal = edges[("causal", ep1.atom_id, ep2.atom_id)]
    assert causal.witness_anchor_ids == [anc_e3, anc_e2] and causal.confidence == 0.7
    assert [t.source_id for t in store.list_transitions(scope)] == [ep1.ep_id, ep2.ep_id]

    # a later batch closes an earlier open file_diff through last_open_action ...
    # 后续批次通过 last_open_action 闭合更早的未闭合 file_diff……
    rc = h.add_events(
        [{"object_type": "file_diff", "timestamp": "2026-06-15T11:00:00Z", "payload": {"path": "src/billing/x.ts", "patch": "+ y"}}],
        **scope4,
    )  # fmt: skip
    assert rc.ep_ids == [] and len(rc.rows) == 1 and rc.rows[0]["atom_type"] == "event"
    rd = h.add_events(
        [{"object_type": "test_log", "timestamp": "2026-06-15T11:30:00Z", "payload": {"suite": "billing_e2e", "passed": 1, "failed": 0}}],
        **scope4,
    )  # fmt: skip
    assert len(rd.ep_ids) == 1
    ep3 = store.get_ep(rd.ep_ids[0])
    assert ep3.slots.a == "patch src/billing/x.ts: + y" and ep3.slots.u == "billing_e2e passed"
    assert ep3.anchor_ids == [_anchor_of(rd, "test_log"), _anchor_of(rc, "file_diff")]
    # ... but not one older than closure_max_gap_hours
    # ……但间隔超过 closure_max_gap_hours 的旧动作不会被闭合
    h.add_events(
        [{"object_type": "file_diff", "timestamp": "2026-06-17T10:00:00Z", "payload": {"path": "src/old.ts", "patch": "+ z"}}],
        **scope4,
    )  # fmt: skip
    stale = h.add_events(
        [{"object_type": "test_log", "timestamp": "2026-06-19T10:00:00Z", "payload": {"suite": "billing_e2e", "passed": 2, "failed": 0}}],
        **scope4,
    )  # fmt: skip
    assert stale.ep_ids == [] and len(stale.rows) == 1 and stale.rows[0]["atom_type"] == "event"

    # tool_call status=error self-closes as a negative EP witnessed by its single anchor
    # status=error 的 tool_call 自闭合为 negative EP，由其唯一锚点见证
    err = h.add_events(
        [{"object_type": "tool_call", "timestamp": "2026-06-20T10:00:00Z",
          "payload": {"tool": "deploy", "args": {"env": "prod"}, "status": "error", "result": "timeout", "intent": "ship it"}}],
        **scope4,
    )  # fmt: skip
    assert len(err.ep_ids) == 1 and len(err.constraint_ids) == 1
    ep_err = store.get_ep(err.ep_ids[0])
    assert ep_err.anchor_ids == [_anchor_of(err, "tool_call")]
    assert (ep_err.slots.a, ep_err.slots.y, ep_err.slots.r, ep_err.slots.u) == (
        "deploy(env=prod)",
        "timeout",
        "negative",
        "ship it",
    )
    assert store.get_atom(err.constraint_ids[0]).text == "Do not re-apply 'deploy(env=prod)': timeout"

    # an open tool_call whose payload carries constraints self-closes with y='' / r='unknown'
    # payload 携带 constraints 的未闭合 tool_call 以 y='' / r='unknown' 自闭合
    tools = {"user_id": "alice", "agent_id": "coder", "run_id": "run-tools"}
    open_call = h.add_events(
        [{"object_type": "tool_call", "timestamp": "2026-06-21T10:00:00Z",
          "payload": {"tool": "lint", "args": {}, "constraints": ["Never touch the generated files."]}}],
        **tools,
    )  # fmt: skip
    assert len(open_call.ep_ids) == 1
    ep_open = store.get_ep(open_call.ep_ids[0])
    assert (ep_open.slots.y, ep_open.slots.r, ep_open.slots.u, ep_open.slots.a) == (
        "",
        "unknown",
        "call lint",
        "lint()",
    )
    assert ep_open.slots.kappa.constraints == ["Never touch the generated files."]
    assert ep_open.anchor_ids == [_anchor_of(open_call, "tool_call")]
    # an open tool_call without constraints and a test_log without any action yield event atoms only
    # 不带 constraints 的未闭合 tool_call 与没有任何动作的 test_log 只产生 event 原子
    quiet = {"user_id": "alice", "agent_id": "coder", "run_id": "run-quiet"}
    lonely = h.add_events(
        [{"object_type": "test_log", "timestamp": "2026-06-22T10:00:00Z", "payload": {"suite": "unit", "passed": 3, "failed": 0}},
         {"object_type": "tool_call", "timestamp": "2026-06-22T11:00:00Z", "payload": {"tool": "fmt", "args": {}}}],
        **quiet,
    )  # fmt: skip
    assert lonely.ep_ids == [] and [r["atom_type"] for r in lonely.rows] == ["event", "event"]
    # sensor traces produce an event atom only
    # sensor_trace 只产生一个 event 原子
    sensed = h.add_events(
        [{"object_type": "sensor_trace", "timestamp": "2026-06-22T12:00:00Z",
          "payload": {"sensor": "temp-1", "samples": [{"t": "2026-06-22T11:59:00Z", "v": 20}]}}],
        **quiet,
    )  # fmt: skip
    assert (
        sensed.ep_ids == [] and len(sensed.rows) == 1 and sensed.rows[0]["memory"] == "sensor_trace temp-1: 1 samples"
    )
    sensor_event = store.get_source_event(sensed.events[0]["event_id"])
    assert sensor_event.event_type == "tool_result" and sensor_event.raw_ref == "sensor://temp-1"
    assert scripted.calls == []


# ---------------------------------------------------------------------------
# 28
# ---------------------------------------------------------------------------
def test_infer_false_raw_atoms_link_entities(store, clock):
    """infer=False: one message atom per message with actor_id/role rows, entity links and an ADD history row.
    infer=False：每条消息一个消息原子，行带 actor_id/role，链接实体并写一条 ADD 历史行。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over rows, atom_entity rows and history.
        对行、atom_entity 行与历史的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scripted = ScriptedLLM({"responses": []})
    h = _Harness(store, clock, scripted)
    res = h.add(
        [
            {"role": "user", "name": "alice", "content": "Sarah joined the billing team in March."},
            {"role": "assistant", "content": "Noted: I patched retry.ts."},
        ],
        infer=False,
        timestamp="2026-03-02T09:00:00Z",
        metadata={"source": "slack"},
        user_id="alice",
        run_id="run-raw",
    )
    assert scripted.calls == []
    assert [(r["event"], r["role"], r["actor_id"]) for r in res.rows] == [
        ("ADD", "user", "alice"),
        ("ADD", "assistant", None),
    ]
    assert all(r["atom_type"] == "message" and r["scale_level"] == "L1_micro" for r in res.rows)
    assert res.rows[0]["metadata"] == {"source": "slack"} and res.rows[0]["hms"]["extracted_by"] == "raw"
    first, second = (store.get_atom(r["atom_id"]) for r in res.rows)
    assert [e["normalized_text"] for e in store.entities_for_atom(first.atom_id)] == ["sarah"]
    assert "retry.ts" in second.entity_ids and first.topic_tags == ["work", "billing"]
    assert store.stats()["atom_entity"] == len(first.entity_ids) + len(second.entity_ids) >= 2
    history = store.get_history(first.memory_id)
    assert [(h_["event"], h_["role"], h_["actor_id"], h_["anchor_id"]) for h_ in history] == [
        ("ADD", "user", "alice", res.events[0]["anchor_id"])
    ]
    assert store.stats()["observations"] == 0 and store.stats()["eps"] == 0


# ---------------------------------------------------------------------------
# 29
# ---------------------------------------------------------------------------
def test_procedural_branch_is_l1_episode(store, clock):
    """agent_id + memory_type='procedural_memory' → one L1_micro episode atom; prompt= replaces the system prompt.
    agent_id + memory_type='procedural_memory' → 一个 L1_micro 情节原子；prompt= 替换系统提示词。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over the atom, its row and the scripted call log.
        对原子、其行与脚本化调用日志的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    conversation = [
        {"role": "user", "content": "Fix the flaky billing test."},
        {"role": "assistant", "content": "I patched retry.ts to add backoff."},
        {"role": "user", "content": "Tests passed afterwards."},
    ]
    markdown = (
        "## Summary of the agent's execution history\n\n**Task Objective**: fix billing\n**Progress Status**: completed"
    )
    scripted = ScriptedLLM({"responses": [markdown]})
    h = _Harness(store, clock, scripted)
    res = h.add(
        conversation, memory_type="procedural_memory", prompt="CUSTOM PROMPT", user_id="alice", agent_id="coder"
    )
    assert len(res.rows) == 1 and len(res.events) == 3
    row = res.rows[0]
    assert (row["event"], row["scale_level"], row["atom_type"], row["memory_type"]) == (
        "ADD",
        "L1_micro",
        "episode",
        "procedural_memory",
    )
    assert row["memory"] == markdown and row["hms"]["source_kind"] == "procedural"
    assert row["anchor_ids"] == [e["anchor_id"] for e in res.events]
    assert len(scripted.calls) == 1 and scripted.calls[0]["task"] is None
    assert scripted.calls[0]["messages"][0] == {"role": "system", "content": "CUSTOM PROMPT"}
    assert scripted.calls[0]["messages"][-1]["content"] == "Create procedural memory of the above conversation."
    atom = store.get_atom(row["atom_id"])
    assert (
        atom.memory_type == "procedural_memory"
        and atom.role == "assistant"
        and atom.content["memory_type"] == "procedural_memory"
    )
    assert store.stats()["observations"] == 0 and store.stats()["eps"] == 0

    rule = _Harness(store, clock, RuleLLM())
    res2 = rule.add(conversation, memory_type="procedural_memory", user_id="alice", agent_id="coder", run_id="run-rule")
    assert res2.rows[0]["memory"].startswith("## Summary of the agent's execution history")
    assert rule.llm.calls[-1]["task"] == "procedural_summary"

    empty = _Harness(store, clock, ScriptedLLM({"responses": [""]}))
    with pytest.raises(ValueError, match="Procedural memory generation failed"):
        empty.add(conversation, memory_type="procedural_memory", user_id="alice", agent_id="coder", run_id="run-empty")
    # without agent_id the memory_type is not a procedural branch: ordinary extraction runs
    # 没有 agent_id 时 memory_type 不构成程序性分支：执行普通抽取
    plain = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": []}]}))
    plain_res = plain.add(conversation, memory_type="procedural_memory", user_id="alice", run_id="run-plain")
    assert plain_res.rows == [] and plain.llm.calls[0]["task"] == "memory_extraction"


# ---------------------------------------------------------------------------
# 30
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["inline", "deferred", "off"])
def test_revision_inline_deferred_off(events, mode):
    """chat_preference under each ingest.revision_mode: inline → UPDATE (one valid atom); deferred/off → two valid atoms.
    在每种 ingest.revision_mode 下的 chat_preference：inline → UPDATE（一个有效原子）；deferred/off → 两个有效原子。

    Input / 输入：
        events: Fixture loader.
        events：夹具加载器。

        mode: The revision mode under test.
        mode：被测的修订模式。

    Output / 输出：
        Assertions over rows, valid atom counts, lineage and history.
        对行、有效原子数、谱系与历史的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    store, clock = _fresh()
    h = _Harness(store, clock, RuleLLM(), ingest={"revision_mode": mode})
    scope = {"user_id": "alice", "run_id": "run-coffee"}
    msg1, msg2 = events("chat_preference").messages
    r1 = h.add(msg1["content"], timestamp=msg1["timestamp"], **scope)
    r2 = h.add(msg2["content"], timestamp=msg2["timestamp"], **scope)
    assert r1.rows[0]["event"] == "ADD" and r1.rows[0]["memory"] == "User likes oat-milk lattes in the morning."
    mem = r1.rows[0]["id"]
    if mode == "inline":
        assert r2.rows[0]["event"] == "UPDATE" and r2.rows[0]["id"] == mem
        assert r2.rows[0]["revises"] == r1.rows[0]["atom_id"] and r2.rows[0]["version"] == 2
        assert store.count_atoms(Scope(**scope)) == 1
        head = store.resolve_head(mem)
        assert head.text == "User likes black coffee." and head.version == 2 and head.valid_to is None
        archived = store.get_atom(r1.rows[0]["atom_id"])
        assert archived.archived_reason == "revised" and archived.valid_to == "2026-04-20T08:00:00.000000+00:00"
        history = store.get_history(mem)
        assert [row["event"] for row in history] == ["ADD", "UPDATE"]
        assert history[1]["old_memory"] == "User likes oat-milk lattes in the morning."
        assert history[1]["new_memory"] == "User likes black coffee."
        assert history[1]["anchor_id"] == r2.events[0]["anchor_id"]
        assert history[1]["updated_at"] == "2026-04-20T08:00:00.000000+00:00"
        edges = store.list_edges(Scope(**scope), edge_types=["revision"])
        assert len(edges) == 1 and edges[0].witness_anchor_ids == [r2.events[0]["anchor_id"]]
        assert edges[0].src_id == r1.rows[0]["atom_id"] and edges[0].dst_id == r2.rows[0]["atom_id"]
        assert edges[0].created_by == "ingest"
        # single extraction mode: the revision judgement travelled inside the one memory_extraction call
        # 单次抽取模式：修订判定随唯一的 memory_extraction 调用一起返回
        assert [c["task"] for c in h.llm.calls] == ["memory_extraction", "memory_extraction"]
        assert [c["id"] for c in h.llm.calls[-1]["sections"]["Candidate Existing Memories"]] == ["0"]
    else:
        assert r2.rows[0]["event"] == "ADD" and r2.rows[0]["id"] != mem and r2.rows[0]["version"] == 1
        assert store.count_atoms(Scope(**scope)) == 2
        assert store.resolve_head(mem).text == "User likes oat-milk lattes in the morning."
        assert store.list_edges(Scope(**scope)) == []
        assert all(call["task"] != "revision_decision" for call in h.llm.calls)


# ---------------------------------------------------------------------------
# 30b. Relocation supersedes the previous residence end to end (§11.1 R5 + §11.3 rule 6)
# 30b. 搬迁端到端取代此前的居住地（§11.1 R5 + §11.3 规则 6）
# ---------------------------------------------------------------------------
def test_relocation_revises_previous_residence():
    """'I live in Paris.' then 'I moved to Berlin.' → UPDATE with predicate lives_in, the old atom archived, one revision edge.
    先 'I live in Paris.' 再 'I moved to Berlin.' → UPDATE 且谓词为 lives_in，旧原子归档，产生一条 revision 边。

    Input / 输入：
        none (a fresh in-memory harness is built inside).
        无（内部自建全新内存夹具）。

    Output / 输出：
        Assertions over the two ingest results, the archived atom, the head and the revision edge.
        对两次摄取结果、被归档原子、头部版本与 revision 边的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    store, clock = _fresh()
    h = _Harness(store, clock, RuleLLM())
    scope = {"user_id": "alice", "run_id": "run-move"}
    r1 = h.add("I live in Paris.", timestamp="2026-04-02T08:00:00Z", **scope)
    r2 = h.add("I moved to Berlin.", timestamp="2026-04-20T08:00:00Z", **scope)
    assert r1.rows[0]["event"] == "ADD" and r1.rows[0]["hms"]["predicate"] == "lives_in"
    # 关键回归：第二句是搬家事实（R5），不是偏好切换（R1）
    assert r2.rows[0]["event"] == "UPDATE"  # the key regression: R5 relocation, not an R1 preference switch
    assert r2.rows[0]["memory"] == "User moved to Berlin." and r2.rows[0]["hms"]["predicate"] == "lives_in"
    assert r2.rows[0]["hms"]["obs_type"] == "fact" and r2.rows[0]["hms"]["topic_tags"] == []
    assert r2.rows[0]["revises"] == r1.rows[0]["atom_id"] and r2.rows[0]["version"] == 2
    assert r2.rows[0]["id"] == r1.rows[0]["id"] and store.count_atoms(Scope(**scope)) == 1
    archived = store.get_atom(r1.rows[0]["atom_id"])
    assert archived.archived_reason == "revised" and archived.valid_to == "2026-04-20T08:00:00.000000+00:00"
    head = store.resolve_head(r1.rows[0]["id"])
    assert head.text == "User moved to Berlin." and head.predicate == "lives_in" and head.valid_to is None
    edges = store.list_edges(Scope(**scope), edge_types=["revision"])
    assert len(edges) == 1 and edges[0].src_id == r1.rows[0]["atom_id"]
    assert edges[0].dst_id == r2.rows[0]["atom_id"] and edges[0].created_by == "ingest"


# ---------------------------------------------------------------------------
# 31
# ---------------------------------------------------------------------------
def test_lock_discipline_llm_runs_unlocked(store, clock, events):
    """The LLM is invoked with no transaction open (Txn A committed, Txn B not yet started).
    LLM 在没有打开事务时被调用（事务 A 已提交，事务 B 尚未开始）。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

    Output / 输出：
        Assertions over store.in_transaction at call time.
        对调用时 store.in_transaction 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    seen: list[bool] = []

    def reply(messages: list[dict]) -> dict:
        """Record the lock state and answer with no observations.
        记录锁状态并回复无观察。
        """
        seen.append(store.in_transaction)
        assert store.in_transaction is False
        assert store.stats()["source_events"] == 1 and store.stats()["anchors"] == 1
        return {"observations": []}

    scripted = ScriptedLLM({"responses": [reply]})
    h = _Harness(store, clock, scripted)
    res = h.add(
        events("chat_preference").messages[0]["content"],
        timestamp="2026-04-02T08:00:00Z",
        user_id="alice",
        run_id="run-lock",
    )
    assert seen == [False] and res.rows == [] and len(res.events) == 1
    assert scripted.calls[0]["task"] == "memory_extraction"
    assert store.in_transaction is False


# ---------------------------------------------------------------------------
# 32
# ---------------------------------------------------------------------------
def test_create_atom_rolls_back_on_embedding_error(store, clock, scope4):
    """An embedder failing on the second text leaves no partial rows: pipeline (before Txn B) and create_atom (rollback).
    嵌入器在第二段文本上失败时不留下部分行：流水线（事务 B 之前）与 create_atom（回滚）。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions over table counts after each failure.
        对每次失败后各表行数的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    embedder = _FlakyEmbedder({"embedding_dims": 64})
    h = _Harness(store, clock, ScriptedLLM({"responses": []}), embedder=embedder)
    with pytest.raises(EmbeddingError):
        h.add(
            [
                {"role": "user", "content": "Sarah joined the billing team."},
                {"role": "user", "content": "POISON text here."},
            ],
            infer=False,
            user_id="alice",
            run_id="run-embed",
        )
    stats = store.stats()
    assert stats["source_events"] == 2 and stats["anchors"] == 2
    assert all(stats[table] == 0 for table in PLANE_TABLES), stats
    assert h.index.count() == 0

    scope = Scope(**scope4)
    (ev, anc, _dedup), = anchor_events(
        store,
        events_from_messages([{"role": "user", "content": S4_CONSTRAINT}], scope=scope, event_time="2026-06-15T09:00:00Z",
                             user_meta={}, visibility="private"),
    )  # fmt: skip
    good = atom_from_observation(
        Observation(type="constraint", subject="project", predicate="must_not", objects=["subscription state machine"],
                    relation="constraint", text=S4_CONSTRAINT, date_observed="2026-06-15", anchor_ids=[anc.anchor_id], scope=scope),
        scope=scope,
        time_anchor=ev.timestamp,
    )  # fmt: skip
    bad = good.model_copy(
        update={"text": "POISON constraint.", "content": {**good.content, "text": "POISON constraint."}, "hash": ""}
    )
    with pytest.raises(EmbeddingError):
        with store.transaction():
            create_atom(store, h.index, embedder, good)
            assert store.stats()["memory_atoms"] == 1
            create_atom(store, h.index, embedder, bad)
    stats = store.stats()
    assert all(stats[table] == 0 for table in PLANE_TABLES), stats
    assert store.in_transaction is False
    # the primitive still works afterwards and the decision plane is intact
    # 此后该原语仍能正常工作，决策平面完好无损
    stored = create_atom(store, h.index, embedder, good)
    assert store.stats()["memory_atoms"] == 1 and store.get_history(stored.memory_id)[0]["event"] == "ADD"
    assert stored.entity_ids == [] and stored.topic_tags == ["billing"]
    assert Decision.ADD.value == "ADD"


# ---------------------------------------------------------------------------
# 32b. Malformed LLM items: a string in a list field is one element; non-string text fields skip the item
# 32b. 畸形 LLM 条目：列表字段里的字符串算一个元素；文本字段不是字符串则跳过该条目
# ---------------------------------------------------------------------------
def test_extraction_coerces_string_lists_and_skips_non_string_fields(store, clock, caplog):
    """`objects: "black coffee"` persists as ['black coffee'] (never its characters); non-string subject/predicate/text/u/a skip the item; a blank constraint string cannot close an EP.
    `objects: "black coffee"` 以 ['black coffee'] 持久化（绝不是逐个字符）；subject/predicate/text/u/a 不是字符串则跳过该条目；空白的约束字符串无法闭合 EP。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        caplog: pytest log capture (the skip warnings).
        caplog：pytest 日志捕获（跳过警告）。

    Output / 输出：
        Assertions over the persisted atom content, the EP κ.constraints and the skip warnings.
        对持久化的原子内容、EP 的 κ.constraints 与跳过警告的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.extraction")
    day = "2026-06-15"
    base = {"event_idx": 0, "type": "fact", "subject": "user", "predicate": "is", "objects": ["x"], "date_observed": day}
    observations = [
        {**base, "id": "0", "type": "preference_event", "predicate": "likes", "objects": "black coffee",
         "text": "User likes black coffee.", "topic_tags": "coffee", "entity_mentions": None},
        {**base, "id": "1", "subject": 42, "text": "User is 42."},
        {**base, "id": "2", "predicate": ["likes"], "text": "User likes tea."},
        {**base, "id": "3", "text": {"nested": "oops"}},
    ]  # fmt: skip
    constraint = "Do not touch the state machine."
    eps = [
        {"id": "0", "event_idx": 0, "u": "fix retry", "a": "patched retry.ts", "y": "", "r": "unknown",
         "kappa": {"constraints": constraint, "visibility": "private"}, "entity_mentions": "retry.ts"},
        {"id": "1", "event_idx": 0, "u": {"x": 1}, "a": ["patched"], "y": "tests failed", "r": "negative"},
        {"id": "2", "event_idx": 0, "u": "fix retry again", "a": "patched again", "y": "", "kappa": {"constraints": "   "}},
    ]  # fmt: skip
    scripted = ScriptedLLM({"responses": [{"observations": observations, "eps": eps}]})
    h = _Harness(store, clock, scripted)
    scope = Scope(user_id="alice", run_id="run-coerce")
    res = h.add(
        "I switched to black coffee, then I patched retry.ts and the tests failed.",
        timestamp="2026-06-15T10:04:00Z",
        user_id="alice",
        run_id="run-coerce",
    )
    assert [c["task"] for c in scripted.calls] == ["memory_extraction"]
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(scope)["atoms"])}
    coffee = atoms["User likes black coffee."]
    assert coffee.content["objects"] == ["black coffee"] and coffee.topic_tags == ["coffee"]
    assert not any(t in atoms for t in ("User is 42.", "User likes tea.", "{'nested': 'oops'}"))
    assert len(res.ep_ids) == 1 and len(store.list_eps(scope)) == 1
    ep = store.get_ep(res.ep_ids[0])
    assert ep.slots.kappa.constraints == [constraint] and ep.entity_mentions == ["retry.ts"] and ep.slots.y == ""
    skipped = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert sum(m.startswith("memory_extraction.observations") for m in skipped) == 3
    assert sum(m.startswith("memory_extraction.eps") for m in skipped) == 2


# ---------------------------------------------------------------------------
# 32c. A scalar in a list field skips that item with a warning instead of raising out of the unlocked phase
# 32c. 列表字段出现标量时告警并跳过该条目，而不是从无锁阶段抛出异常
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["objects", "topic_tags", "entity_mentions", "supersedes_hint_ids"])
def test_extraction_scalar_list_field_skips_item_instead_of_raising(store, clock, caplog, field):
    """An item with `<field>: 5` is skipped with a warning while the other item of the same reply still becomes an atom.
    `<field>: 5` 的条目告警并跳过，同一回复中的另一条目仍然生成原子。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        caplog: pytest log capture (the skip warning).
        caplog：pytest 日志捕获（跳过警告）。

        field: The list field that receives the scalar.
        field：收到标量的列表字段。

    Output / 输出：
        Assertions: add() returns normally with exactly the valid row; one warning names the field.
        断言：add() 正常返回且只含有效行；恰有一条警告点名该字段。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.extraction")
    day = "2026-06-15"
    bad = {"id": "0", "event_idx": 0, "type": "fact", "subject": "user", "predicate": "is", "objects": ["x"],
           "text": f"User is {field}.", "date_observed": day, field: 5}  # fmt: skip
    good = {"id": "1", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
            "objects": ["tea"], "text": "User likes tea.", "date_observed": day}  # fmt: skip
    scripted = ScriptedLLM({"responses": [{"observations": [bad, good]}]})
    h = _Harness(store, clock, scripted)
    scope = Scope(user_id="alice", run_id=f"run-{field}")
    # Before the fix this raised a raw TypeError out of the unlocked phase (Txn A committed, Txn B never run).
    # 修复前这里会从无锁阶段抛出原生 TypeError（事务 A 已提交，事务 B 从未执行）。
    res = h.add(f"I like tea ({field}).", timestamp="2026-06-15T10:00:00Z", user_id="alice", run_id=f"run-{field}")
    assert [(r["event"], r["memory"]) for r in res.rows] == [("ADD", "User likes tea.")]
    assert [a.text for a in store.get_atoms(store.valid_ids(scope)["atoms"])] == ["User likes tea."]
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("memory_extraction.observations")]
    assert len(messages) == 1 and "skipped" in messages[0] and field in messages[0]


# ---------------------------------------------------------------------------
# Group V: multipart messages and the vision hook
# V 组：多部件消息与视觉钩子
# ---------------------------------------------------------------------------
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"not really a png" * 8
_PNG_URI = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode("ascii")
_PNG_SHA = hashlib.sha256(_PNG_BYTES).hexdigest()
_IMAGE_MESSAGE = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this."},
            {"type": "image_url", "image_url": {"url": _PNG_URI, "detail": "low"}},
        ],
    }
]
_UC_HASH = "3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc"


def _no_base64(store: StateStore, res: IngestResult) -> None:
    """Assert that no anchored payload of `res` carries the base64 body (only sha256 / mime reach the store).
    断言 `res` 落锚的载荷都不含 base64 正文（只有 sha256 / mime 进入存储）。
    """
    body = _PNG_URI.split(",", 1)[1]
    for entry in res.events:
        ev = store.get_source_event(entry["event_id"])
        blob = json.dumps(ev.payload, ensure_ascii=False)
        assert body not in blob and "data:" not in blob


def test_multipart_message_without_vision_keeps_current_behaviour(store, clock):
    """enable_vision=False (default): no describe_image call, image parts stay in dropped_parts, sha256/mime recorded, no base64 stored; text-only hashes unchanged.
    enable_vision=False（默认）：不调用 describe_image，图片部件仍计入 dropped_parts，记录 sha256/mime，不存 base64；纯文本哈希不变。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over scripted.calls, the anchored payload, the raw message atom and the §19 payload hash.
        对 scripted.calls、落锚载荷、原始消息原子与 §19 载荷哈希的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scripted = ScriptedLLM({"responses": [], "image_description": "a red bicycle"})
    h = _Harness(store, clock, scripted)
    assert h.config.llm.enable_vision is False and h.config.llm.vision_details is None
    res = h.add(_IMAGE_MESSAGE, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    # Vision off → the scripted description is never requested even though it is configured.
    # 视觉关闭 → 即便配置了描述，也从不请求 scripted 的描述。
    assert scripted.calls == []
    ev = store.get_source_event(res.events[0]["event_id"])
    assert ev.payload == {
        "text": "Look at this.",
        "dropped_parts": 1,
        "parts": [{"kind": "image", "source": "data", "sha256": _PNG_SHA, "mime": "image/png", "url": None, "detail": "low"}],
    }
    assert "described_parts" not in ev.payload
    _no_base64(store, res)
    anchor = store.get_anchor(res.events[0]["anchor_id"])
    assert anchor.hash == ev.payload_hash
    atom = store.get_atom(res.rows[0]["atom_id"])
    assert atom.text == "Look at this." and atom.content["dropped_parts"] == 1
    # A text-only message keeps its historical payload shape and hash (CONTRACT §19 anchor table).
    # 纯文本消息保持历史载荷形状与哈希（契约 §19 锚点表）。
    plain = h.add(S4_CONSTRAINT, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    plain_ev = store.get_source_event(plain.events[0]["event_id"])
    assert plain_ev.payload == {"text": S4_CONSTRAINT, "dropped_parts": 0} and plain_ev.payload_hash == _UC_HASH


def test_vision_describes_images_unlocked_before_txn_a(store, clock):
    """enable_vision=True: describe_image runs with no transaction and no event written; the description joins the text, parts/described_parts are anchored, dropped_parts excludes described parts, identical re-add dedups.
    enable_vision=True：describe_image 在无事务、未写事件时运行；描述并入文本，parts/described_parts 落锚，dropped_parts 不含已描述部件，相同重加去重。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over the lock state at describe time, the anchored payload, the raw atom, the extraction prompt and dedup.
        对描述时的锁状态、落锚载荷、原始原子、抽取提示词与去重的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    seen: list[tuple[bool, int]] = []

    class _VisionLLM(ScriptedLLM):
        """Scripted LLM that records the store state at describe time.
        在描述时记录 store 状态的脚本化 LLM。
        """

        def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
            """Record (in_transaction, source_events) then reply with the fixed description.
            记录 (in_transaction, source_events) 后回复固定描述。
            """
            seen.append((store.in_transaction, store.stats()["source_events"]))
            return super().describe_image(url_or_data_uri, detail=detail)

    def reply(messages: list[dict]) -> dict:
        """The extraction prompt must already carry the merged image description.
        抽取提示词必须已包含并入的图片描述。
        """
        assert "[Image: a red bicycle leaning on a wall]" in messages[-1]["content"]
        return {"observations": []}

    scripted = _VisionLLM({"responses": [reply], "image_description": "a red bicycle leaning on a wall"})
    h = _Harness(store, clock, scripted, llm_config={"provider": "scripted", "enable_vision": True, "vision_details": "high"})
    res = h.add(_IMAGE_MESSAGE, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    # One describe call, unlocked, before Txn A wrote anything; the part-level detail wins over vision_details.
    # 一次描述调用、无锁、在事务 A 写入任何东西之前；部件级 detail 优先于 vision_details。
    assert seen == [(False, 0)]
    assert scripted.calls[0]["task"] == "describe_image" and scripted.calls[0]["kwargs"] == {"url": _PNG_URI, "detail": "low"}
    ev = store.get_source_event(res.events[0]["event_id"])
    assert ev.payload == {
        "text": "Look at this.\n[Image: a red bicycle leaning on a wall]",
        "dropped_parts": 0,
        "parts": [{"kind": "image", "source": "data", "sha256": _PNG_SHA, "mime": "image/png", "url": None, "detail": "low"}],
        "described_parts": [
            {"index": 0, "sha256": _PNG_SHA, "text": "a red bicycle leaning on a wall", "described_by": "scripted@v1"}
        ],
    }
    _no_base64(store, res)
    assert store.get_anchor(res.events[0]["anchor_id"]).hash == ev.payload_hash
    atom = store.get_atom(res.rows[0]["atom_id"])
    assert atom.text == ev.payload["text"] and atom.content["dropped_parts"] == 0
    assert atom.anchor_ids == [res.events[0]["anchor_id"]]
    # The same message at the same event time dedups: the anchored identity covers the image sha256 + description.
    # 同一事件时间的相同消息去重：落锚身份覆盖图片 sha256 + 描述。
    again = h.add(_IMAGE_MESSAGE, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    assert again.events[0]["dedup"] is True and again.rows[0]["event"] == "NONE"
    # infer=True: the description reaches the extraction prompt; the describe call precedes the extraction call.
    # infer=True：描述进入抽取提示词；描述调用先于抽取调用。
    before = len(scripted.calls)
    ext = h.add(_IMAGE_MESSAGE, timestamp="2026-06-16T09:00:00Z", user_id="alice", run_id="run-v")
    assert [c["task"] for c in scripted.calls[before:]] == ["describe_image", "memory_extraction"]
    assert ext.rows == [] and store.in_transaction is False
    # A message without a part-level detail falls back to config.llm.vision_details; an http URL is kept verbatim.
    # 没有部件级 detail 的消息回退到 config.llm.vision_details；http URL 原样保留。
    remote = [{"role": "user", "content": [{"type": "image_url", "image_url": "https://img.example/x.jpg"}]}]
    res2 = h.add(remote, infer=False, timestamp="2026-06-17T09:00:00Z", user_id="alice", run_id="run-v")
    assert scripted.calls[-1]["kwargs"] == {"url": "https://img.example/x.jpg", "detail": "high"}
    ev2 = store.get_source_event(res2.events[0]["event_id"])
    assert ev2.payload["text"] == "[Image: a red bicycle leaning on a wall]" and ev2.payload["dropped_parts"] == 0
    assert ev2.payload["parts"][0]["source"] == "url" and ev2.payload["parts"][0]["url"] == "https://img.example/x.jpg"
    assert ev2.payload["parts"][0]["mime"] == "image/jpeg"


def test_vision_without_perception_keeps_parts_dropped(store, clock):
    """enable_vision=True with a provider whose describe_image returns None (rule): parts recorded, still dropped, no error, no described_parts.
    enable_vision=True 但提供者的 describe_image 返回 None（rule）：记录 parts、仍计丢弃、不报错、无 described_parts。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over the anchored payload and the raw atom.
        对落锚载荷与原始原子的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    rule = RuleLLM()
    assert rule.describe_image(_PNG_URI, detail="high") is None
    h = _Harness(store, clock, rule, llm_config={"provider": "rule", "enable_vision": True})
    res = h.add(_IMAGE_MESSAGE, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    ev = store.get_source_event(res.events[0]["event_id"])
    assert ev.payload["text"] == "Look at this." and ev.payload["dropped_parts"] == 1
    assert ev.payload["parts"][0]["sha256"] == _PNG_SHA and "described_parts" not in ev.payload
    assert store.get_atom(res.rows[0]["atom_id"]).content["dropped_parts"] == 1
    _no_base64(store, res)


def test_vision_failure_raises_llm_error_before_any_write(store, clock):
    """A provider exception inside describe_image becomes LLMError(LLM_001, task='describe_image') and nothing is written.
    describe_image 内的提供者异常变为 LLMError(LLM_001, task='describe_image')，且什么都不写入。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over the error code / details and store.stats().
        对错误码 / 详情与 store.stats() 的断言。

    State effect / 状态影响：
        None (the store stays empty).
        无（store 保持为空）。
    """

    class _BrokenVision(ScriptedLLM):
        """Scripted LLM whose vision transport fails.
        视觉传输失败的脚本化 LLM。
        """

        def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
            """Raise a transport error.
            抛出传输错误。
            """
            raise RuntimeError("boom")

    h = _Harness(store, clock, _BrokenVision({"responses": []}), llm_config={"provider": "scripted", "enable_vision": True})
    with pytest.raises(LLMError) as info:
        h.add(_IMAGE_MESSAGE, infer=False, timestamp="2026-06-15T09:00:00Z", user_id="alice", run_id="run-v")
    assert info.value.error_code == "LLM_001" and isinstance(info.value.__cause__, RuntimeError)
    assert info.value.details == {"task": "describe_image", "provider": "scripted@v1", "index": 0, "part": 0}
    assert store.stats()["source_events"] == 0 and store.stats()["anchors"] == 0 and store.in_transaction is False


def test_procedural_branch_sees_image_descriptions(store, clock):
    """The procedural summary receives the merged text (description included) and the anchored payload matches it.
    过程摘要收到并入描述后的文本，且落锚载荷与之一致。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over the procedural_summary call messages and the episode atom.
        对 procedural_summary 调用消息与情节原子的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    scripted = ScriptedLLM({"responses": ["# Procedure\n1. looked at the bicycle"], "image_description": "a red bicycle"})
    h = _Harness(store, clock, scripted, llm_config={"provider": "scripted", "enable_vision": True})
    res = h.add(
        _IMAGE_MESSAGE,
        memory_type="procedural_memory",
        timestamp="2026-06-15T09:00:00Z",
        user_id="alice",
        agent_id="bot",
        run_id="run-v",
    )
    assert [c["task"] for c in scripted.calls] == ["describe_image", "procedural_summary"]
    sent = scripted.calls[1]["messages"]
    assert sent[1] == {"role": "user", "content": "Look at this.\n[Image: a red bicycle]"}
    ev = store.get_source_event(res.events[0]["event_id"])
    assert ev.payload["text"] == "Look at this.\n[Image: a red bicycle]" and ev.payload["dropped_parts"] == 0
    assert res.rows[0]["memory"] == "# Procedure\n1. looked at the bicycle"


def test_message_payload_parts_reject_data_uri_and_malformed_items(store, clock):
    """A message payload part carrying a data: URL or a malformed part / described_part is VALIDATION_005 (field 'parts' / 'described_parts').
    消息载荷部件带 data: URL，或部件 / described_part 畸形 → VALIDATION_005（字段 'parts' / 'described_parts'）。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions over ValidationError codes and details for add_events dicts and validate_payload.
        对 add_events 字典与 validate_payload 的 ValidationError 错误码与详情的断言。

    State effect / 状态影响：
        None (nothing is anchored).
        无（没有任何东西落锚）。
    """
    good = {"kind": "image", "source": "url", "sha256": _PNG_SHA, "mime": "image/png", "url": "https://x/a.png"}
    h = _Harness(store, clock, ScriptedLLM({"responses": []}))
    ok = h.add_events(
        [{"object_type": "message", "payload": {"text": "t", "dropped_parts": 1, "parts": [good]}}],
        infer=False,
        user_id="alice",
    )
    ev = store.get_source_event(ok.events[0]["event_id"])
    assert ev.payload["parts"] == [{**good, "detail": None}] and "described_parts" not in ev.payload
    bad_cases = [
        ({"text": "t", "parts": [{**good, "url": _PNG_URI}]}, "parts", 0),
        ({"text": "t", "parts": [{**good, "source": "blob"}]}, "parts", 0),
        ({"text": "t", "parts": [{"kind": "image", "source": "url"}]}, "parts", 0),
        # A non-list `parts` has no part index: events_from_dicts fills `index` with the event position (0).
        # 非列表的 `parts` 没有部件序号：events_from_dicts 用事件位置（0）填充 `index`。
        ({"text": "t", "parts": {"kind": "image"}}, "parts", 0),
        ({"text": "t", "parts": [good], "described_parts": [{"index": True, "sha256": _PNG_SHA, "text": "d"}]}, "described_parts", 0),
        ({"text": "t", "parts": [good], "described_parts": [{"index": 0, "sha256": _PNG_SHA, "text": ""}]}, "described_parts", 0),
    ]
    for payload, field_name, index in bad_cases:
        with pytest.raises(ValidationError) as info:
            h.add_events([{"object_type": "message", "payload": payload}], infer=False, user_id="alice")
        assert info.value.error_code == "VALIDATION_005", payload
        assert info.value.details["field"] == field_name and info.value.details.get("index") == index, payload
    assert store.stats()["source_events"] == 1


# ---------------------------------------------------------------------------
# 32e. A speaker name in `attributed_to` is kept as an entity mention; the side falls back to the message role
# 32e. `attributed_to` 里的说话人名保留为实体提及；侧别回退为消息角色
# ---------------------------------------------------------------------------
def test_attributed_to_speaker_name_is_coerced_not_rejected(store, clock, caplog):
    """A speaker name in attributed_to is kept as an entity mention and the side falls back to the message role.
    attributed_to 里的说话人名保留为实体提及，侧别回退为消息角色。

    Input / 输入：
        A scripted memory_extraction reply whose observations carry attributed_to "Katrina" (a speaker in a
        multi-party transcript), "assistant" (legal) and "" (empty).
        脚本化的 memory_extraction 回复，其观察的 attributed_to 分别为 "Katrina"（多方转录中的说话人）、
        "assistant"（合法）与 ""（空）。

    Output / 输出：
        All three atoms persist; the named one is attributed to the user side with "Katrina" appended to its
        entity mentions; no "skipped" warning is logged.
        三个原子全部持久化；带人名的那条归到 user 侧且实体提及追加了 "Katrina"；没有 "skipped" 告警。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.extraction")
    day = "2026-06-15"
    base = {"event_idx": 0, "type": "fact", "predicate": "plans", "objects": ["pressed flowers"], "date_observed": day}
    observations = [
        {**base, "id": "0", "subject": "Katrina", "text": "Katrina plans to make pressed flowers.",
         "attributed_to": "Katrina", "entity_mentions": ["Katrina"]},
        {**base, "id": "1", "subject": "Shure", "text": "Shure bought acrylic glass.", "attributed_to": " Shure ",
         "entity_mentions": []},
        {**base, "id": "2", "subject": "assistant", "text": "Assistant summarised the plan.",
         "attributed_to": "assistant"},
        {**base, "id": "3", "subject": "user", "text": "User agreed to the plan.", "attributed_to": ""},
    ]  # fmt: skip
    scripted = ScriptedLLM({"responses": [{"observations": observations, "eps": []}]})
    h = _Harness(store, clock, scripted)
    scope = Scope(user_id="alice", run_id="run-speaker")
    h.add(
        "Katrina: I'll make pressed flowers. Shure: I bought acrylic glass.",
        timestamp="2026-06-15T10:04:00Z",
        user_id="alice",
        run_id="run-speaker",
    )
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(scope)["atoms"])}
    assert len(atoms) == 4
    katrina = atoms["Katrina plans to make pressed flowers."]
    assert katrina.attributed_to == "user" and any("katrina" in e.lower() for e in katrina.entity_ids)
    shure = atoms["Shure bought acrylic glass."]
    assert shure.attributed_to == "user" and any("shure" in e.lower() for e in shure.entity_ids)
    assert atoms["Assistant summarised the plan."].attributed_to == "assistant"
    assert atoms["User agreed to the plan."].attributed_to == "user"
    assert not [r for r in caplog.records if "skipped" in r.getMessage()]


# ---------------------------------------------------------------------------
# 32f. A bare JSON list reply is read as the observations array instead of dropping the batch
# 32f. 裸 JSON 列表回复按 observations 数组读取，而不是丢掉整个批次
# ---------------------------------------------------------------------------
def test_bare_list_reply_is_read_as_observations(store, clock, caplog):
    """A bare JSON list reply is read as the observations array instead of dropping the batch.
    裸 JSON 列表回复按 observations 数组读取，而不是丢掉整个批次。

    Input / 输入：
        A scripted memory_extraction reply that is a JSON list of observation items (no {"observations": ...}
        envelope), as deepseek-flash produced on long batches.
        脚本化的 memory_extraction 回复是观察条目的 JSON 列表（没有 {"observations": ...} 信封），
        deepseek-flash 在长批次上曾如此输出。

    Output / 输出：
        The items persist as atoms; one warning notes the envelope was missing; nothing is reported as skipped.
        条目持久化为原子；一条告警指出缺少信封；没有条目被报告为跳过。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.extraction")
    day = "2026-06-15"
    items = [
        {"id": "0", "event_idx": 0, "type": "fact", "subject": "user", "predicate": "likes", "objects": ["tea"],
         "text": "User likes tea.", "date_observed": day},
        {"id": "1", "event_idx": 0, "type": "fact", "subject": "user", "predicate": "owns", "objects": ["a kettle"],
         "text": "User owns a kettle.", "date_observed": day},
    ]  # fmt: skip
    scripted = ScriptedLLM({"responses": [items]})
    h = _Harness(store, clock, scripted)
    scope = Scope(user_id="alice", run_id="run-list")
    h.add("I like tea and I own a kettle.", timestamp="2026-06-15T10:04:00Z", user_id="alice", run_id="run-list")
    texts = sorted(a.text for a in store.get_atoms(store.valid_ids(scope)["atoms"]))
    assert texts == ["User likes tea.", "User owns a kettle."]
    messages = [r.getMessage() for r in caplog.records]
    assert any("reply is a JSON list" in m for m in messages)
    assert not any("skipped" in m for m in messages)

