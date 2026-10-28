"""Group K tests: HMS §11 state kinds, builders, deterministic renderers and the StateAPI over the offline facade.
K 组测试：HMS §11 的状态种类、构建器、确定性渲染器与离线门面之上的 StateAPI。
"""

# Purpose (EN): Assert every state kind is non-empty and witnessed, renders its template titles, orders time, groups
# invalidated preferences, matches why() chains and renders byte-identically twice.
# 目的（中）：断言每种状态非空且有见证、渲染出模板标题、按时间排序、失效偏好分组、与 why() 链一致、两次渲染逐字节相同。

# Input (EN): tests/fixtures/events/* plus a few structured events, on the conftest fixtures (frozen clock, ids).
# 输入（中）：tests/fixtures/events/* 与少量结构化事件，基于 conftest 夹具（冻结时钟、顺序 id）。

# Output (EN): pytest results, parametrized over the two index providers by the `hms` fixture.
# 输出（中）：pytest 结果，由 `hms` 夹具按两种索引提供者参数化。

# State effect (EN): None outside the in-memory store and index.
# 状态影响（中）：除内存存储与索引外无副作用。

from __future__ import annotations

from typing import Any, Callable

import pytest

from hms_elite.models import Scope
from hms_elite.state import (
    STATE_SPECS,
    StateAPI,
    StateKind,
    build_edge_state,
    render_detective_report,
    render_state,
    spec_of,
)

U = "alice"
AS_OF = "2026-09-03T00:00:00Z"
LATTE = "User likes oat-milk lattes in the morning."
BLACK = "User likes black coffee."
BLACK_TS = "2026-04-20T08:00:00.000000+00:00"
UC_TEXT = "Do not modify the subscription state machine."
S4_QUERY = "fix checkout webhook retry"
FRIDAY = "Do not deploy on Fridays."
TEAM_MSG = "I work at Acme Robotics."
TEAM_FACT = "User work at Acme Robotics."
MEETING = [
    ("We decided to ship the webhook retry design by June 30.", "2026-06-01T09:00:00Z"),
    ("Sarah will own the checkout refactor.", "2026-06-01T09:05:00Z"),
    ("The team must review the rollout plan before Friday.", "2026-06-01T09:10:00Z"),
    ("Bob and Sarah agreed to postpone the migration.", "2026-06-01T09:15:00Z"),
    ("Bob met Sarah in Berlin on May 6.", "2026-05-07T09:15:00Z"),
]


class _Seeded:
    """One HMS seeded with every fixture the state tests need, plus a clock-ticking `call`.
    用状态测试所需的全部夹具灌入的一个 HMS，外加推动时钟的 `call`。
    """

    def __init__(self, hms: Any, clock: Any, store: Any, events: Callable[[str], Any], scope4: dict) -> None:
        """Ingest chat fixtures (alice + bob), meeting messages, one team-visible fact and the S4 billing run.
        摄取聊天夹具（alice + bob）、会议消息、一条 team 可见事实与 S4 计费运行。

        Input / 输入：
            hms: The HMS fixture.
            hms：HMS 夹具。

            clock: FrozenClock fixture.
            clock：FrozenClock 夹具。

            store: The StateStore shared with `hms`.
            store：与 `hms` 共享的 StateStore。

            events: Fixture loader.
            events：夹具加载器。

            scope4: The S4 scope dict.
            scope4：S4 范围字典。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            Writes atoms, EPs and edges into the in-memory store.
            向内存存储写入原子、EP 与边。
        """
        self.hms, self.clock, self.store, self.scope4 = hms, clock, store, scope4
        self.api = StateAPI(hms)
        for m in events("chat_preference").messages:
            self.call(hms.add, m["content"], timestamp=m["timestamp"], user_id=U, run_id="run-coffee")
        for m in events("chat_conflict").messages:
            self.call(hms.add, m["content"], timestamp=m["timestamp"], user_id=U, run_id="run-home")
        for m in events("chat_habit_temporal").messages:
            self.call(hms.add, m["content"], timestamp=m.get("timestamp"), user_id=U)
        for m in events("chat_multisession").messages:
            self.call(hms.add, m["content"], timestamp=m["timestamp"], user_id=U, run_id=m["run_id"])
        for text, ts in MEETING:
            self.call(hms.add, text, timestamp=ts, user_id=U, run_id="run-meeting")
        self.call(
            hms.add_events,
            [
                {
                    "object_type": "message",
                    "role": "user",
                    "payload": {"text": TEAM_MSG},
                    "visibility": "team",
                    "timestamp": "2026-07-01T09:00:00Z",
                }
            ],  # fmt: skip
            user_id=U,
            run_id="run-profile",
        )
        self.call(hms.add, "I like black coffee.", timestamp="2026-04-21T08:00:00Z", user_id="bob")
        self.call(hms.add, "Sarah joined the billing team in March.", timestamp="2026-03-02T09:00:00Z", user_id="bob")
        self.call(hms.add, FRIDAY, timestamp="2026-03-03T09:00:00Z", user_id="bob")
        self.call(hms.add, FRIDAY, timestamp="2026-03-03T09:00:00Z", user_id=U)
        self.call(hms.add, "I live in Paris.", timestamp="2026-05-02T09:00:00Z", user_id="bob")
        fixture = events("structured_billing")
        self.call(hms.add, fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
        self.call(hms.add_events, fixture.events_fail, **scope4)
        self.call(hms.add_events, fixture.events_fix, **scope4)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Advance the clock 60 s, then call.
        推进时钟 60 秒后调用。

        Input / 输入：
            fn: The callable.
            fn：可调用对象。

            args: Positional arguments.
            args：位置参数。

            kwargs: Keyword arguments.
            kwargs：关键字参数。

        Output / 输出：
            The callable's result.
            可调用对象的返回值。

        State effect / 状态影响：
            Mutates the frozen clock.
            修改冻结时钟。
        """
        self.clock.tick(seconds=60)
        return fn(*args, **kwargs)


@pytest.fixture
def seeded(hms, clock, store, events, scope4) -> _Seeded:
    """The seeded HMS for every test of this module.
    本模块每个测试使用的已灌入 HMS。
    """
    return _Seeded(hms, clock, store, events, scope4)


def _assert_witnessed(state: Any) -> None:
    """Every item of a state has >= 1 anchor id and the state's witness union is non-empty.
    状态的每一项至少有一个锚点 id，且状态的见证并集非空。
    """
    items = state.items()
    assert items, f"{state.kind} carries no items"
    assert all(item.anchor_ids for item in items)
    assert state.witness_ids()


def _titles(kind: StateKind, text: str) -> None:
    """Every template title of the kind appears in the rendered text.
    该种类模板的每个标题都出现在渲染文本中。
    """
    for title in spec_of(kind).template:
        assert title in text, f"{kind}: missing section {title!r}"


def test_state_specs_cover_every_kind():
    """Every StateKind has a StateSpec with a slot rule, a template and fields; render_state dispatches by kind.
    每个 StateKind 都有带槽位规则、模板与字段的 StateSpec；render_state 按种类分派。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over STATE_SPECS.
        对 STATE_SPECS 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert set(STATE_SPECS) == set(StateKind.values()) and len(StateKind.values()) == 12
    for kind in StateKind:
        spec = spec_of(kind)
        assert spec.kind == kind and spec.slot_rule and spec.template and spec.fields and spec.title


def test_work_state_handoff_package_and_rollback_point(seeded):
    """Work State Card == ContextIR.to_text(); the Handoff Package renders the §3.8 sections + rollback/checkpoint.
    Work State Card == ContextIR.to_text()；交接包渲染 §3.8 各节 + 回滚点/检查点。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes two checkpoints.
        写入两个检查点。
    """
    api, s4 = seeded.api, seeded.scope4
    ws, card = seeded.call(api.work_state, S4_QUERY, filters=s4, as_of=AS_OF)
    assert ws.kind == "work_state" and card == ws.context.to_text() and card.startswith("# Work State Card")
    assert ws.goal == S4_QUERY and ws.next_action and ws.context.anchor_ids() and ws.witness_ids()
    _titles(StateKind.work_state, card.replace("## ", ""))

    point, point_text = seeded.call(api.rollback_point, filters=s4, label="before-handoff", create=True)
    assert point is not None and point.checkpoint_id.startswith("ckp_") and point.anchor_ids
    assert point.label == "before-handoff" and point.atoms_live > 0
    _titles(StateKind.rollback_point, point_text.replace("## ", ""))
    latest, _ = api.rollback_point(filters=s4)
    assert latest == point

    pkg_state, text = seeded.call(api.handoff_package, filters=s4, goal=S4_QUERY)
    pkg = pkg_state.package
    assert pkg.goal == S4_QUERY and pkg.checkpoint_id and pkg_state.rollback_point is not None
    assert pkg_state.rollback_point.checkpoint_id == pkg.checkpoint_id
    for title in spec_of(StateKind.handoff_package).template:
        assert f"{title}:" in text
    assert pkg.checkpoint_id in text and text.startswith("# Handoff Package")
    assert (
        "- patch src/billing/retry.ts: add exponential backoff → billing_e2e failed: subscription_state_regression"
        in text
    )
    assert f"- {UC_TEXT}" in text and all(a.anchor_id in text for a in pkg.anchors)
    assert set(pkg_state.witness_ids()) == {a.anchor_id for a in pkg.anchors} | set(pkg_state.rollback_point.anchor_ids)
    assert render_state(pkg_state) == text == render_state(pkg_state)


def test_detective_report_follows_the_why_chain(seeded):
    """DetectiveReport.chain == why()['chain']; evidence rows carry anchor/time/source; dict and state render equal.
    DetectiveReport.chain == why()['chain']；证据行带锚点/时间/来源；字典与状态渲染相同。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api, hms, s4 = seeded.api, seeded.hms, seeded.scope4
    question = "why must retry.ts not be modified"
    report, text = seeded.call(api.detective_report, question, filters=s4, as_of=AS_OF)
    why = hms.why(question, filters=s4)
    assert report.chain == why["chain"] and len(report.chain) == 3
    assert report.question == question and report.conclusion.text == why["target"]["text"]
    assert report.conclusion.atom_id == why["target"]["node_id"] and report.conclusion.anchor_ids
    assert {s.atom_id for s in report.steps} == {n["node_id"] for n in why["nodes"]}
    expected = sorted(why["anchors"], key=lambda a: (a["timestamp"], a["anchor_id"]))
    assert [e.anchor_ids for e in report.evidence] == [[a["anchor_id"]] for a in expected]
    assert all(e.time and e.extra["source_ref"] for e in report.evidence)
    assert report.open_conflicts == [] and "Unresolved Conflicts:\n- (none)" in text
    _titles(StateKind.detective_report, text)
    assert text.split("\n")[1:3] == ["Question:", question]
    assert "1. " + why["chain"][0] in text and "3. " + why["chain"][2] in text
    assert render_detective_report(why, question=question) == text
    _assert_witnessed(report)


def test_personal_state_groups_valid_and_invalidated_preferences(seeded):
    """The revised latte preference lands in the invalidated group with its valid_to and successor; timeline is ordered.
    被修订的拿铁偏好进入失效组并带 valid_to 与后继；时间线有序。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api = seeded.api
    state, text = api.personal_state(filters={"user_id": U}, as_of=AS_OF)
    assert state.kind == "personal_state" and state.subject == "user" and state.audience is None
    assert [p.text for p in state.preferences_valid] == [BLACK]
    assert [p.text for p in state.preferences_invalid] == [LATTE]
    latte = state.preferences_invalid[0]
    assert latte.valid_to == BLACK_TS and latte.extra["invalidated_at"] == BLACK_TS
    assert latte.extra["archived_reason"] == "revised" and latte.extra["superseded_by"] == BLACK
    assert [h.text for h in state.habits] == ["User usually run every morning."]
    assert any(c.text == FRIDAY for c in state.constraints)
    assert any("Paris" in f.text for f in state.profile_facts)
    times = [t.time for t in state.timeline]
    assert times == sorted(times) and len(times) >= 3
    _assert_witnessed(state)
    _titles(StateKind.personal_state, text.replace("## ", ""))
    assert f"until {BLACK_TS}; superseded by: {BLACK}" in text and text.startswith("# Personal Card")
    # as_of before the switch: black coffee is not yet valid and the archived latte version stays invalidated
    # (LIVE_SQL semantics: an archived version never comes back through as_of).
    # as_of 早于切换：黑咖啡尚未生效，而已归档的拿铁版本仍在失效组（LIVE_SQL 语义：归档版本不会因 as_of 复活）。
    before, _ = api.personal_state(filters={"user_id": U}, as_of="2026-04-10T00:00:00Z")
    assert before.preferences_valid == [] and [p.text for p in before.preferences_invalid] == [LATTE]
    # Subject filter: Sarah's facts only.
    # 主体过滤：仅 Sarah 的事实。
    sarah, _ = api.personal_state(filters={"user_id": U}, subject="Sarah", as_of=AS_OF)
    assert sarah.preferences_valid == [] and sarah.profile_facts and all("Sarah" in f.text for f in sarah.profile_facts)


def test_shared_card_gates_visibility(seeded):
    """The Shared Card keeps only atoms visible to the audience (team): private preferences drop, the team fact stays.
    Shared Card 只保留受众（team）可见的原子：私有偏好被丢弃，team 事实保留。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api = seeded.api
    card, text = api.shared_card(filters={"user_id": U}, as_of=AS_OF)
    assert card.kind == "shared_card" and card.audience == "team" and text.startswith("# Shared Card")
    assert card.preferences_valid == [] and card.preferences_invalid == []
    assert [f.text for f in card.profile_facts] == [TEAM_FACT]
    assert all(str(i.visibility) != "private" for i in card.items())
    _assert_witnessed(card)
    _titles(StateKind.shared_card, text.replace("## ", ""))
    # A private audience is lifted to team: a shared card never leaks private atoms.
    # 私有受众被提升为 team：共享卡绝不泄露私有原子。
    assert api.shared_card(filters={"user_id": U}, audience="private", as_of=AS_OF)[0] == card


def test_edge_state_between_entities_and_between_scopes(seeded):
    """EdgeState: shared facts, time-ordered interactions, last contact, open conflicts; both endpoint forms.
    EdgeState：共享事实、按时间排序的互动、最近联系、未解冲突；两种端点形式。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api, hms = seeded.api, seeded.hms
    edge, text = api.edge_state("Bob", "Sarah", filters={"user_id": U}, as_of=AS_OF)
    assert edge.endpoint_kind == "entity" and (edge.a, edge.b) == ("Bob", "Sarah")
    assert [f.text for f in edge.shared_facts] == ["Bob and Sarah agreed to postpone the migration."]
    assert [i.text for i in edge.interactions] == ["Bob met Sarah in Berlin on May 6."]
    assert edge.last_contact == "2026-06-01T09:15:00.000000+00:00"
    _assert_witnessed(edge)
    _titles(StateKind.edge_state, text.replace("## ", ""))
    assert "Endpoints: Bob ⟷ Sarah (entity)" in text

    pair, pair_text = api.edge_state({"user_id": U}, {"user_id": "bob"}, as_of=AS_OF)
    assert pair.endpoint_kind == "scope" and (pair.a, pair.b) == ("alice", "bob")
    times = [i.time for i in pair.interactions]
    assert times == sorted(times) and len(times) >= 3
    assert {i.extra["side"] for i in pair.interactions} == {"alice", "bob"}
    assert any(f.text == FRIDAY for f in pair.shared_facts)
    cross = [c for c in pair.open_conflicts if c.extra.get("members") == ["alice", "bob"]]
    assert len(cross) == 1 and cross[0].text == "User live in Berlin. ⟂ User live in Paris."
    assert cross[0].extra["reason"].startswith("exclusive predicate 'lives_in'")
    # The unresolved conflict edge inside alice's own scope also surfaces, witnessed by the edge.
    # alice 自己范围内的未解冲突边也会浮现，由该边见证。
    edge_conflicts = [c for c in pair.open_conflicts if "edge_id" in c.extra]
    assert (
        len(edge_conflicts) == 1 and seeded.store.get_edge(edge_conflicts[0].extra["edge_id"]).edge_type == "conflict"
    )
    _assert_witnessed(pair)
    assert "Endpoints: alice ⟷ bob (scope)" in pair_text
    with pytest.raises(ValueError):
        build_edge_state(hms, "Bob", {"user_id": "bob"}, filters={"user_id": U})


def test_group_state_shared_constraints_decisions_and_conflicts(seeded):
    """GroupState: a constraint held by both members is shared, decisions are collected, conflicts are paired.
    GroupState：两名成员共同持有的约束成为共享约束，决策被收集，冲突成对出现。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api = seeded.api
    group, text = api.group_state(members=[{"user_id": U}, {"user_id": "bob"}], as_of=AS_OF)
    assert group.members == ["alice", "bob"]
    assert [c.text for c in group.shared_constraints] == [FRIDAY]
    shared = group.shared_constraints[0]
    assert shared.extra["members"] == ["alice", "bob"] and len(shared.anchor_ids) == 2
    assert any("decided" in d.text for d in group.decisions) and any("agreed" in d.text for d in group.decisions)
    assert all(d.extra["member"] in group.members for d in group.decisions)
    assert len(group.conflicts_between_members) == 1
    conflict = group.conflicts_between_members[0]
    assert conflict.members == ["alice", "bob"] and (conflict.left.text, conflict.right.text) == (
        "User live in Berlin.",
        "User live in Paris.",
    )
    assert set(conflict.anchor_ids) == set(conflict.left.anchor_ids) | set(conflict.right.anchor_ids)
    _assert_witnessed(group)
    _titles(StateKind.group_state, text.replace("## ", ""))
    assert f"- {FRIDAY} (held by: alice, bob)" in text


def test_customer_issue_state(seeded):
    """CustomerIssueState: issue / status / owner / failed_paths / next_action from the S4 run, all witnessed.
    CustomerIssueState：来自 S4 运行的 issue / status / owner / failed_paths / next_action，全部有见证。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api, s4 = seeded.api, seeded.scope4
    state, text = api.customer_issue_state(filters=s4, as_of=AS_OF)
    assert (
        state.issue.text == S4_QUERY and state.status.text == "resolved" and state.status.extra["reward"] == "positive"
    )
    assert state.owner is not None and state.owner.text == "coder"
    assert len(state.failed_paths) == 1 and state.failed_paths[0].extra["r"] == "negative"
    assert [c.text for c in state.constraints][0] == UC_TEXT
    assert state.next_action is not None and state.next_action.text == state.context.next_action
    assert state.next_action.text.startswith("Continue from 'patch src/billing/idempotency.ts")
    _assert_witnessed(state)
    _titles(StateKind.customer_issue_state, text.replace("## ", ""))
    assert "- resolved [" in text and "- coder [" in text
    # At the failing instant the issue is blocked and the next action says not to re-apply the patch.
    # 在失败时刻问题处于 blocked，下一步说明不要重复应用该补丁。
    blocked, _ = api.customer_issue_state(filters=s4, as_of="2026-06-15T10:05:00Z")
    assert blocked.status.text == "blocked" and blocked.next_action.text.startswith("Do not re-apply")
    with pytest.raises(ValueError):
        api.customer_issue_state(filters={"user_id": "nobody"})


def test_incident_state_root_cause_matches_why(seeded):
    """IncidentState.root_cause.chain equals HMS.why() for both the implicit (newest failed EP) and question targets.
    IncidentState.root_cause.chain 在隐式目标（最新失败 EP）与问题目标两种情况下都等于 HMS.why()。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes one checkpoint.
        写入一个检查点。
    """
    api, hms, store, s4 = seeded.api, seeded.hms, seeded.store, seeded.scope4
    cp = seeded.call(hms.checkpoint, filters=s4, label="incident")
    state, text = api.incident_state(filters=s4, as_of=AS_OF)
    times = [t.time for t in state.timeline]
    assert times == sorted(times) and len(state.timeline) == 4
    assert all(t.atom_type == "event" for t in state.timeline)
    failed = [ep for ep in store.list_eps(Scope(**s4)) if ep.slots.r == "negative"]
    assert state.root_cause is not None and state.root_cause.chain == hms.why(failed[-1].atom_id)["chain"]
    assert state.root_cause.question == failed[-1].slots.gloss()
    assert [c.text for c in state.constraints][0] == UC_TEXT
    assert state.rollback_point is not None and state.rollback_point.checkpoint_id == cp
    _assert_witnessed(state)
    _titles(StateKind.incident_state, text.replace("## ", ""))
    assert cp in text and "Question: " in text

    question = "why must retry.ts not be modified"
    asked, asked_text = api.incident_state(filters=s4, question=question, as_of=AS_OF)
    assert asked.root_cause.chain == hms.why(question, filters=s4)["chain"]
    assert f"Question: {question}" in asked_text


def test_meeting_decision_state(seeded):
    """MeetingDecisionState: decisions, named owners, dated deadlines and superseded statements with successors.
    MeetingDecisionState：决策、具名负责人、带日期的截止项与带后继的被取代陈述。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    api = seeded.api
    state, text = api.meeting_decision_state(filters={"user_id": U, "run_id": "run-meeting"}, as_of=AS_OF)
    assert [d.text for d in state.decisions] == [
        "Bob and Sarah agreed to postpone the migration.",
        "The team decided to ship the webhook retry design by June 30.",
    ]
    assert {o.extra["owner"] for o in state.owners} == {"Sarah", "Bob"}
    assert [d.text for d in state.deadlines] == ["The team decided to ship the webhook retry design by June 30."]
    assert state.deadlines[0].extra["due"] == "2026-06-30T00:00:00.000000+00:00"
    assert state.superseded == []
    whole, _ = api.meeting_decision_state(filters={"user_id": U}, as_of=AS_OF)
    assert [s.text for s in whole.superseded] == [LATTE] and whole.superseded[0].extra["superseded_by"] == BLACK
    _assert_witnessed(state)
    _titles(StateKind.meeting_decision_state, text.replace("## ", ""))
    assert "(owner: Sarah)" in text and "(due: 2026-06-30T00:00:00.000000+00:00)" in text


def test_account_handoff_state(seeded):
    """AccountHandoffState = Handoff Package + customer constraints; the checkpoint becomes its rollback point.
    AccountHandoffState = 交接包 + 客户约束；检查点成为其回滚点。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes one checkpoint.
        写入一个检查点。
    """
    api, s4 = seeded.api, seeded.scope4
    state, text = seeded.call(api.account_handoff_state, filters=s4, goal=S4_QUERY)
    assert state.package.goal == S4_QUERY and state.package.pcir.ok()
    assert [c.text for c in state.customer_constraints][0] == UC_TEXT
    assert state.rollback_point is not None and state.rollback_point.checkpoint_id == state.package.checkpoint_id
    _assert_witnessed(state)
    _titles(StateKind.account_handoff_state, text.replace("## ", ""))
    assert "# Handoff Package" in text and text.startswith("# Account Handoff State")
    assert text.count(state.package.checkpoint_id) >= 3
    plain, plain_text = seeded.call(api.account_handoff_state, filters=s4, goal=S4_QUERY, checkpoint=False)
    assert plain.rollback_point is None and "Checkpoint id:\n(none)" in plain_text


def test_renderers_are_deterministic_and_states_witnessed(seeded):
    """Every kind: build → render twice byte-identical; read-only kinds rebuild to identical text; all items witnessed.
    每种状态：构建 → 两次渲染逐字节相同；只读种类重建后文本相同；全部项有见证。

    Input / 输入：
        seeded: The seeded HMS.
        seeded：已灌入的 HMS。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the checkpoints of handoff_package / account_handoff_state.
        写入 handoff_package / account_handoff_state 的检查点。
    """
    api, s4 = seeded.api, seeded.scope4
    members = [{"user_id": U}, {"user_id": "bob"}]
    read_only: dict[str, Callable[[], tuple[Any, str]]] = {
        "work_state": lambda: api.work_state(S4_QUERY, filters=s4, as_of=AS_OF),
        "detective_report": lambda: api.detective_report("why must retry.ts not be modified", filters=s4, as_of=AS_OF),
        "personal_state": lambda: api.personal_state(filters={"user_id": U}, as_of=AS_OF),
        "shared_card": lambda: api.shared_card(filters={"user_id": U}, as_of=AS_OF),
        "edge_state": lambda: api.edge_state("Bob", "Sarah", filters={"user_id": U}, as_of=AS_OF),
        "group_state": lambda: api.group_state(members=members, as_of=AS_OF),
        "customer_issue_state": lambda: api.customer_issue_state(filters=s4, as_of=AS_OF),
        "incident_state": lambda: api.incident_state(filters=s4, as_of=AS_OF),
        "meeting_decision_state": lambda: api.meeting_decision_state(filters={"user_id": U}, as_of=AS_OF),
    }
    for kind, build in read_only.items():
        state, text = build()
        assert str(state.kind) == kind and render_state(state) == text == render_state(state)
        rebuilt, text2 = build()
        assert text2 == text and rebuilt == state
        assert text.startswith("# " + spec_of(kind).title)
        _assert_witnessed(state)
    writing: dict[str, Callable[[], tuple[Any, str]]] = {
        "handoff_package": lambda: seeded.call(api.handoff_package, filters=s4, goal=S4_QUERY),
        "rollback_point": lambda: seeded.call(api.rollback_point, filters=s4, label="det", create=True),
        "account_handoff_state": lambda: seeded.call(api.account_handoff_state, filters=s4, goal=S4_QUERY),
    }
    for kind, build in writing.items():
        state, text = build()
        assert str(state.kind) == kind and render_state(state) == text == render_state(state)
        assert text.startswith("# " + spec_of(kind).title) and state.witness_ids()
