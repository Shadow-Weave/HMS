"""Tests for hms_elite.runtime: ActionRegistry mapping and the AgentRuntime act → outcome → EP → Work State Card loop.
hms_elite.runtime 的测试：ActionRegistry 映射与 AgentRuntime 的 act → outcome → EP → Work State Card 闭环。
"""

# Purpose (EN): Prove the HMS_arch §5.1 runtime adapter only needs the public HMS facade: registry normalization,
#               closure into EPs with derived constraints on negative results, the card, why / handoff / checkpoint /
#               rollback / feedback delegation and the error codes.
# 目的（中）：证明 HMS_arch §5.1 的运行时适配器只需要公共 HMS 门面：注册表归一化、负结果闭合成 EP 并派生约束、
#            卡片、why / handoff / checkpoint / rollback / feedback 的委托以及错误码。

# Input (EN): The conftest store fixture (frozen clock, sequential ids) and an HMS on rule LLM + hash + memory index.
# 输入（中）：conftest 的 store 夹具（冻结时钟、顺序 id）以及 rule LLM + hash + memory 索引上的 HMS。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): In-memory stores only.
# 状态影响（中）：仅内存存储。

from __future__ import annotations

import pytest

from hms_elite.configs.base import HMSConfig
from hms_elite.exceptions import ValidationError
from hms_elite.memory.main import HMS
from hms_elite.models import ContextIR, StatePackage
from hms_elite.runtime import (
    DEFAULT_ACTIONS,
    ActionRegistry,
    AgentRuntime,
    generic_outcome,
    outcome_from_test_result,
)

GOAL = "fix checkout webhook retry"
BACKGROUND = "Do not modify the subscription state machine."
TEST_RESULT = {
    "suite": "billing_e2e",
    "passed": 11,
    "failed": 3,
    "failures": [{"name": "subscription_state_regression", "message": "state machine changed"}],
}


@pytest.fixture
def runtime_hms(store):
    """HMS on the rule LLM + hash embedder + memory index sharing the frozen-clock store.
    共享冻结时钟 store 的 rule LLM + hash 嵌入 + memory 索引上的 HMS。

    Input / 输入：
        store: The conftest StateStore fixture.
        store：conftest 的 StateStore 夹具。

    Output / 输出：
        An HMS instance.
        一个 HMS 实例。

    State effect / 状态影响：
        None beyond the in-memory store.
        除内存存储外无影响。
    """
    return HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}), store=store)


@pytest.fixture
def runtime(runtime_hms):
    """An AgentRuntime bound to alice / coder / run-billing with the default registry.
    绑定到 alice / coder / run-billing、使用默认注册表的 AgentRuntime。

    Input / 输入：
        runtime_hms: The HMS fixture.
        runtime_hms：HMS 夹具。

    Output / 输出：
        An AgentRuntime.
        一个 AgentRuntime。

    State effect / 状态影响：
        None.
        无。
    """
    return AgentRuntime(
        runtime_hms, user_id="alice", agent_id="coder", run_id="run-billing", goal=GOAL, source_system="claude_code"
    )


def _failed_patch(rt: AgentRuntime) -> dict:
    """Background constraint, then patch retry.ts and fail billing_e2e; returns the outcome report.
    背景约束，然后修改 retry.ts 并让 billing_e2e 失败；返回结果报告。
    """
    rt.observe(BACKGROUND, timestamp="2026-06-15T09:00:00Z")
    action_id = rt.act(
        "patch",
        path="src/billing/retry.ts",
        patch="+ backoff = 2 ** attempt",
        summary="add exponential backoff",
        intent=GOAL,
        timestamp="2026-06-15T10:00:00Z",
    )
    return rt.outcome(action_id, TEST_RESULT, status="error", timestamp="2026-06-15T10:04:00Z")


def test_registry_defaults_map_calls_to_action_events():
    """The four default actions map their calls onto normalized tool_call / file_diff payloads.
    四个默认动作把调用映射为归一化的 tool_call / file_diff 载荷。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the normalized event dicts.
        对归一化事件字典的断言。

    State effect / 状态影响：
        None.
        无。
    """
    reg = ActionRegistry()
    assert reg.names() == list(DEFAULT_ACTIONS) and "patch" in reg and "nope" not in reg
    patch = reg.call_event("patch", {"path": "a.py", "patch": "+x", "intent": "fix", "constraints": "keep api"})
    assert patch == {
        "object_type": "file_diff",
        "payload": {
            "path": "a.py",
            "patch": "+x",
            "summary": None,
            "intent": "fix",
            "commit": None,
            "constraints": ["keep api"],
        },
    }
    shell = reg.call_event("shell", {"cmd": "pytest -q", "cwd": "/repo", "intent": "run"})
    assert shell["object_type"] == "tool_call"
    assert shell["payload"] == {
        "tool": "shell",
        "args": {"cmd": "pytest -q", "cwd": "/repo"},
        "result": None,
        "status": None,
        "target": "/repo",
        "intent": "run",
        "constraints": [],
    }
    http = reg.call_event("http", {"url": "https://x.test/a", "body": {"k": 1}})
    assert http["payload"]["args"] == {"body": '{"k": 1}', "method": "GET", "url": "https://x.test/a"}
    assert http["payload"]["target"] == "https://x.test/a"
    tests = reg.call_event("run_tests", {"suite": "billing_e2e", "k": "billing"})
    assert tests["payload"]["tool"] == "run_tests" and tests["payload"]["target"] == "billing_e2e"
    assert tests["payload"]["args"] == {"k": "billing", "suite": "billing_e2e"}


def test_registry_maps_results_to_test_log_events():
    """Results become test_log outcomes: 'error' → failed with one failure, test-shaped dicts keep their structure.
    结果变成 test_log 结果事件：'error' → failed 并带一条失败项，测试形状的字典保留结构。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the normalized outcome dicts.
        对归一化结果字典的断言。

    State effect / 状态影响：
        None.
        无。
    """
    reg = ActionRegistry()
    shell = reg.outcome_event("shell", {"cmd": "make"}, "exit 1: boom\nmore lines", "error")
    assert shell == {
        "object_type": "test_log",
        "payload": {
            "suite": "shell",
            "status": "failed",
            "passed": 0,
            "failed": 1,
            "failures": [{"name": "exit 1: boom", "message": "exit 1: boom\nmore lines"}],
            "log": "exit 1: boom\nmore lines",
            "related_paths": [],
        },
    }
    ok = reg.outcome_event("http", {"url": "https://x.test/a", "method": "post"}, {"code": 200}, "ok")
    assert ok["payload"]["suite"] == "http POST https://x.test/a" and ok["payload"]["status"] == "passed"
    assert ok["payload"]["passed"] == 1 and ok["payload"]["failed"] == 0 and ok["payload"]["failures"] == []
    tests = reg.outcome_event("run_tests", {"suite": "unit"}, {"passed": 11, "failed": 3, "failures": ["x"]}, "ok")
    assert tests["payload"]["suite"] == "unit" and tests["payload"]["status"] == "failed"
    assert tests["payload"]["failures"] == [{"name": "x", "message": None}] and tests["payload"]["passed"] == 11
    log_only = reg.outcome_event("run_tests", {}, "42 passed", "ok")
    assert log_only["payload"] == {
        "suite": "tests",
        "status": "passed",
        "passed": None,
        "failed": None,
        "failures": [],
        "log": "42 passed",
        "related_paths": [],
    }
    patch = reg.outcome_event("patch", {"path": "a.py", "patch": "+x"}, TEST_RESULT, "error")
    assert patch["payload"]["suite"] == "billing_e2e" and patch["payload"]["status"] == "failed"
    assert patch["payload"]["failures"] == TEST_RESULT["failures"]
    plain = generic_outcome("patch a.py", {"path": "a.py"}, None, "ok")
    assert plain["payload"]["related_paths"] == ["a.py"] and plain["payload"]["log"] is None
    assert outcome_from_test_result("s", {}, {"status": "failed"}, "ok")["payload"]["status"] == "failed"


def test_registry_errors_and_custom_actions():
    """Unknown names / statuses / wrong object types raise VALIDATION_013; custom actions get generic outcomes.
    未知名称 / 状态 / 错误对象类型抛出 VALIDATION_013；自定义动作获得通用结果映射。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over error codes and a custom registration.
        对错误码与一次自定义注册的断言。

    State effect / 状态影响：
        None.
        无。
    """
    reg = ActionRegistry(defaults=False)
    assert reg.names() == []
    with pytest.raises(ValidationError) as exc:
        reg.get("patch")
    assert exc.value.error_code == "VALIDATION_013" and exc.value.details["action"] == "patch"
    spec = reg.register("deploy", call=lambda a: {"object_type": "tool_call", "payload": {"tool": "deploy", "args": a}})
    assert spec.name == "deploy" and reg.names() == ["deploy"]
    call = reg.call_event("deploy", {"env": "staging"})
    assert call["payload"]["args"] == {"env": "staging"} and call["payload"]["status"] is None
    out = reg.outcome_event("deploy", {"env": "staging"}, "rolled back", "error")
    assert out["payload"]["suite"] == "deploy" and out["payload"]["failures"][0]["name"] == "rolled back"
    with pytest.raises(ValidationError) as bad_status:
        reg.outcome_event("deploy", {}, None, "failed")
    assert bad_status.value.error_code == "VALIDATION_013"
    reg.register("say", call=lambda a: {"object_type": "message", "payload": {"text": "hi"}})
    with pytest.raises(ValidationError) as wrong_type:
        reg.call_event("say", {})
    assert wrong_type.value.error_code == "VALIDATION_013" and wrong_type.value.details["object_type"] == "message"
    reg.register(
        "odd",
        call=lambda a: {"object_type": "tool_call", "payload": {"tool": "odd"}},
        outcome=lambda a, r, s: {"object_type": "tool_call", "payload": {"tool": "odd", "status": "error"}},
    )
    with pytest.raises(ValidationError):
        reg.outcome_event("odd", {}, None, "ok")
    reg.register("broken", call=lambda a: "not a dict")
    with pytest.raises(ValidationError):
        reg.call_event("broken", {})
    with pytest.raises(ValidationError):
        reg.register("", call=lambda a: {})
    with pytest.raises(ValidationError):
        reg.register("x", call="not callable")
    with pytest.raises(ValidationError) as missing:
        ActionRegistry().call_event("patch", {"path": "a.py"})
    assert missing.value.error_code == "VALIDATION_005" and missing.value.details["field"] == "patch"
    assert reg.unregister("odd") is True and reg.unregister("odd") is False


def test_act_outcome_closes_into_ep_and_derives_constraint(runtime, store):
    """act() opens a file_diff, outcome(status='error') closes it into a negative EP with a derived constraint.
    act() 打开一个 file_diff，outcome(status='error') 把它闭合成负向 EP 并派生约束。

    Input / 输入：
        runtime: The AgentRuntime fixture.
        runtime：AgentRuntime 夹具。

        store: The shared store (public reads only).
        store：共享的存储（仅公开读取）。

    Output / 输出：
        Assertions over the pending table, the EP slots and the derived constraint atom.
        对未闭合表、EP 槽位与派生约束原子的断言。

    State effect / 状态影响：
        Writes to the in-memory store through the facade.
        通过门面写入内存存储。
    """
    runtime.observe(BACKGROUND, timestamp="2026-06-15T09:00:00Z")
    action_id = runtime.act(
        "patch",
        path="src/billing/retry.ts",
        patch="+ backoff = 2 ** attempt",
        summary="add exponential backoff",
        intent=GOAL,
        timestamp="2026-06-15T10:00:00Z",
    )
    pending = runtime.open_actions()
    assert action_id.startswith("evt_") and len(pending) == 1
    assert pending[0]["action"] == "patch" and pending[0]["object_type"] == "file_diff"
    assert pending[0]["action_id"] == action_id and pending[0]["anchor_id"].startswith("anc_")
    action_event = store.get_source_event(action_id)
    assert action_event is not None and action_event.source_system == "claude_code"
    assert action_event.actor_id == "coder" and str(action_event.role) == "tool"

    report = runtime.outcome(action_id, TEST_RESULT, status="error", timestamp="2026-06-15T10:04:00Z")
    assert report["closed"] is True and report["status"] == "error" and report["action"] == "patch"
    assert len(report["eps"]) == 1 and len(report["constraints"]) == 1 and report["edges"]
    assert runtime.open_actions() == []
    ep = store.get_ep(report["eps"][0])
    assert ep is not None and str(ep.slots.r) == "negative"
    assert ep.slots.a == "patch src/billing/retry.ts: add exponential backoff"
    assert ep.slots.y == "billing_e2e failed: subscription_state_regression"
    assert ep.slots.u == GOAL and BACKGROUND in ep.slots.kappa.constraints
    assert pending[0]["anchor_id"] in ep.anchor_ids and report["anchor_id"] in ep.anchor_ids
    derived = store.get_atom(report["constraints"][0])
    assert derived is not None and derived.text == (
        "Do not re-apply 'patch src/billing/retry.ts: add exponential backoff': "
        "billing_e2e failed: subscription_state_regression"
    )
    assert str(derived.obs_type) == "constraint"

    shell_id = runtime.act("shell", cmd="npm run build", timestamp="2026-06-15T10:10:00Z")
    ok = runtime.outcome(shell_id, "built in 3s", status="ok", timestamp="2026-06-15T10:11:00Z")
    assert ok["closed"] is True and len(ok["eps"]) == 1 and ok["constraints"] == []
    assert str(store.get_ep(ok["eps"][0]).slots.r) == "positive"
    assert store.get_ep(ok["eps"][0]).slots.y == "shell passed"


def test_state_card_lists_failed_paths_and_context_returns_ir(runtime):
    """state() renders the Work State Card with the failed path, the goal and the constraints; context() is a ContextIR.
    state() 渲染含失败路径、目标与约束的 Work State Card；context() 返回 ContextIR。

    Input / 输入：
        runtime: The AgentRuntime fixture.
        runtime：AgentRuntime 夹具。

    Output / 输出：
        Assertions over the card text and the ContextIR.
        对卡片文本与 ContextIR 的断言。

    State effect / 状态影响：
        Writes to the in-memory store through the facade.
        通过门面写入内存存储。
    """
    _failed_patch(runtime)
    card = runtime.state(as_of="2026-06-15T12:00:00Z")
    assert card.startswith("# Work State Card\nScope: user_id=alice agent_id=coder run_id=run-billing | As of: ")
    assert "## Current Goal\nfix checkout webhook retry\n" in card
    assert "## Failed Paths\n- patch src/billing/retry.ts: add exponential backoff → " in card
    assert "billing_e2e failed: subscription_state_regression\n" in card
    assert f"- {BACKGROUND}" in card and "- Do not re-apply 'patch src/billing/retry.ts" in card
    ir = runtime.context(600, as_of="2026-06-15T12:00:00Z", mode="deep")
    assert isinstance(ir, ContextIR) and ir.goal == GOAL and ir.budget_tokens == 600 and str(ir.mode) == "deep"
    assert [str(item.explain.get("r")) for item in ir.episodes] == ["negative"]
    assert runtime.state(query="what failed", as_of="2026-06-15T12:00:00Z").count("## Failed Paths") == 1
    runtime.set_goal(None)
    assert "## Current Goal\ncurrent work state\n" in runtime.state(as_of="2026-06-15T12:00:00Z")


def test_why_handoff_checkpoint_rollback_feedback_delegate(runtime, runtime_hms, store):
    """why / handoff / checkpoint / rollback / feedback go through the facade with this runtime's scope and actor.
    why / handoff / checkpoint / rollback / feedback 以本运行时的范围与 actor 经过门面。

    Input / 输入：
        runtime: The AgentRuntime fixture.
        runtime：AgentRuntime 夹具。

        runtime_hms: The HMS fixture.
        runtime_hms：HMS 夹具。

        store: The shared store (public reads only).
        store：共享的存储（仅公开读取）。

    Output / 输出：
        Assertions over the facade reports.
        对门面报告的断言。

    State effect / 状态影响：
        Writes to the in-memory store through the facade.
        通过门面写入内存存储。
    """
    report = _failed_patch(runtime)
    why = runtime.why("why must retry.ts not be modified")
    assert why["explanation"] and why["nodes"] and "chain" in why
    by_id = runtime.why(report["eps"][0], hops=1)
    assert by_id["target"]["id"].startswith(("atm_", "ep_")) and by_id["nodes"]

    package = runtime.handoff()
    assert isinstance(package, StatePackage) and package.goal == GOAL and package.checkpoint_id is not None
    assert package.scope.as_filters() == runtime.filters
    plain = runtime.handoff(goal="ship it", checkpoint=False)
    assert plain.goal == "ship it" and plain.checkpoint_id is None

    cp = runtime.checkpoint("before shell")
    assert cp.startswith("ckp_")
    shell_id = runtime.act("shell", cmd="rm -rf build", timestamp="2026-06-15T10:20:00Z")
    runtime.outcome(shell_id, "exit 1: permission denied", status="error", timestamp="2026-06-15T10:21:00Z")
    runtime.act("http", url="https://api.test/x", timestamp="2026-06-15T10:30:00Z")
    assert len(runtime.open_actions()) == 1
    rolled = runtime.rollback(cp, reason="undo shell")
    assert rolled["checkpoint_id"] == cp and rolled["archived"] and runtime.open_actions() == []

    fb = runtime.feedback(report["constraints"][0], "object_important", reason="keep")
    assert fb["message"] == "Feedback applied" and fb["feedback_id"].startswith("fb_")
    feedback_events = store.list_events(runtime.scope, object_types=["feedback"])
    assert feedback_events and feedback_events[-1].actor_id == "coder"
    assert runtime_hms.get_all(filters=runtime.filters)["results"]


def test_observe_records_message_and_errors(runtime, runtime_hms, store):
    """observe() records a message event attributed to the user; bad scopes / unknown actions raise contract codes.
    observe() 记录归属于用户的消息事件；错误范围 / 未知动作抛出契约错误码。

    Input / 输入：
        runtime: The AgentRuntime fixture.
        runtime：AgentRuntime 夹具。

        runtime_hms: The HMS fixture.
        runtime_hms：HMS 夹具。

        store: The shared store (public reads only).
        store：共享的存储（仅公开读取）。

    Output / 输出：
        Assertions over the recorded event and error codes.
        对记录的事件与错误码的断言。

    State effect / 状态影响：
        Writes to the in-memory store through the facade.
        通过门面写入内存存储。
    """
    out = runtime.observe("I prefer black coffee.", timestamp="2026-06-15T08:00:00Z")
    assert out["results"] and out["events"][0]["object_type"] == "message"
    event = store.get_source_event(out["events"][0]["event_id"])
    assert event.actor_id == "alice" and str(event.role) == "user" and event.payload["text"] == "I prefer black coffee."
    raw = runtime.observe("tool said hi", role="tool", infer=False, timestamp="2026-06-15T08:01:00Z")
    assert store.get_source_event(raw["events"][0]["event_id"]).actor_id == "coder"
    assert runtime.actor_id == "coder" and runtime.filters == {
        "user_id": "alice",
        "agent_id": "coder",
        "run_id": "run-billing",
    }

    with pytest.raises(ValidationError) as no_scope:
        AgentRuntime(runtime_hms)
    assert no_scope.value.error_code == "VALIDATION_001"
    with pytest.raises(ValidationError) as unknown:
        runtime.act("teleport", where="mars")
    assert unknown.value.error_code == "VALIDATION_013"
    with pytest.raises(ValidationError) as closed:
        runtime.outcome("evt_nope", "x")
    assert closed.value.error_code == "NOT_FOUND_001"
    action_id = runtime.act("shell", cmd="ls", timestamp="2026-06-15T08:02:00Z")
    assert runtime.discard(action_id) is True and runtime.discard(action_id) is False
    with pytest.raises(ValidationError):
        runtime.outcome(action_id, "gone")
    solo = AgentRuntime(runtime_hms, user_id="bob", registry=ActionRegistry(defaults=False))
    assert solo.actor_id == "bob" and solo.registry.names() == [] and solo.filters == {"user_id": "bob"}
