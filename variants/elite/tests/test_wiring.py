"""Tests for the §2 main-line wiring: config knobs, HMS.usage / trace, StateAPI delegates, the daemon and root exports.
§2 主线接线的测试：配置开关、HMS.usage / trace、StateAPI 委托、守护进程与根包导出。
"""

# Purpose (EN): Prove that the three streams (telemetry, state cards, background daemon) hang off the HMS facade at
#               their HMS_arch §2 positions and that the new config fields reach the collaborators they configure.
# 目的（中）：证明三条流（遥测、状态卡、后台守护进程）按 HMS_arch §2 的位置挂在 HMS 门面上，且新的配置字段
#            到达了它们所配置的协作对象。

# Input (EN): The conftest fixtures (hms, clock, events, scope4) and in-memory facades built here.
# 输入（中）：conftest 夹具（hms、clock、events、scope4）与本文件构造的内存门面。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): In-memory stores and short-lived daemon threads only.
# 状态影响（中）：仅内存存储与短生命周期的守护线程。

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

import hms_elite
from hms_elite import HMS, AgentRuntime, HMSConfig, StateAPI, StateDaemon, StateKind, Usage, telemetry
from hms_elite.configs.base import REMOVED, RENAMED, CompilerConfig, IngestConfig, SchedulerConfig, TextConfig
from hms_elite.daemon import THREAD_NAME
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.state.kinds import WorkState
from hms_elite.telemetry import usage_of

AS_OF = "2026-06-15T12:00:00Z"
S4_QUERY = "fix checkout webhook retry"
SCOPE = {"user_id": "alice", "agent_id": "coder", "run_id": "run-wiring"}
STATE_METHODS = (
    "work_state",
    "handoff_package",
    "detective_report",
    "rollback_point",
    "personal_state",
    "shared_card",
    "edge_state",
    "group_state",
    "customer_issue_state",
    "incident_state",
    "meeting_decision_state",
    "account_handoff_state",
)


def _memory_hms(**overrides: Any) -> HMS:
    """An offline in-memory facade (rule + hash + memory index) with config overrides.
    带配置覆盖的离线内存门面（rule + hash + memory 索引）。
    """
    return HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, **overrides))


def _wait_until(predicate: Any, *, timeout: float = 5.0) -> bool:
    """Poll `predicate` every 5 ms until true or `timeout` seconds elapsed.
    每 5 ms 轮询一次 `predicate`，直到为真或超过 `timeout` 秒。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _seed_s4(hms: HMS, clock: Any, events: Any, scope4: dict) -> None:
    """Constraint + events_fail + events_fix, one minute apart.
    约束 + events_fail + events_fix，各隔一分钟。
    """
    fixture = events("structured_billing")
    clock.tick(seconds=60)
    hms.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fail, **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fix, **scope4)
    clock.tick(seconds=60)


def test_config_fields_reach_their_collaborators():
    """ingest.extraction_mode reaches the Ingestor, compiler.selection / scheduler.daemon_interval_s validate, and the Mem0 RENAMED / REMOVED tables are unchanged.
    ingest.extraction_mode 到达 Ingestor，compiler.selection / scheduler.daemon_interval_s 受校验，Mem0 的 RENAMED / REMOVED 表保持不变。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over defaults, pass-through and rejection of bad values.
        对默认值、透传与非法值拒绝的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    assert IngestConfig().extraction_mode == "single" and IngestConfig().segment_max_chars == 400
    assert CompilerConfig().selection == "shares" and SchedulerConfig().daemon_interval_s is None
    assert (CompilerConfig().context_session_budget_tokens, CompilerConfig().context_rows_top_k) == (1200, 20)
    assert CompilerConfig().context_rows_header == "Recent relevant memories:" and CompilerConfig().context_timestamps is True
    # CJK switches (CONTRACT §9.6): bigram tokens and no Chinese statement rules keep English stores byte-identical.
    # CJK 开关（CONTRACT §9.6）：二元组分词与不启用中文陈述规则，使英文存储逐字节不变。
    assert TextConfig().cjk_tokens == "bigram" and IngestConfig().zh_rules is False
    default = _memory_hms()
    assert default.ingestor.extraction_mode == "single"
    default.close()
    # staged has no lines prompts: since the 2026-09-18 default flip it needs an explicit full / compact output.
    # staged 没有行式提示词：2026-09-18 默认值切换后需显式给出 full / compact 输出。
    staged = _memory_hms(ingest={"extraction_mode": "staged", "extraction_output": "full", "segment_max_chars": 120})
    assert staged.ingestor.extraction_mode == "staged" and staged.config.ingest.segment_max_chars == 120
    staged.close()
    for bad in (
        {"ingest": {"extraction_mode": "triple"}},
        {"ingest": {"segment_max_chars": 0}},
        {"compiler": {"selection": "budget"}},
        {"scheduler": {"daemon_interval_s": 0}},
    ):
        with pytest.raises(ConfigError) as exc:
            HMSConfig.from_dict({**bad, "store": {"path": ":memory:"}})
        assert exc.value.error_code == "CONFIG_001", bad
    assert HMSConfig.from_dict({"scheduler": {"daemon_interval_s": 0.5}}).scheduler.daemon_interval_s == 0.5
    assert RENAMED == {
        "custom_fact_extraction_prompt": "custom_instructions",
        "custom_update_memory_prompt": "custom_instructions",
        "index": "vector_store",
        "db_path": "store.path",
    }
    assert set(REMOVED) == {"graph_store", "enable_graph"}


def test_facade_usage_and_trace(hms):
    """HMS.usage() is telemetry.usage_of(hms) and HMS.trace() is telemetry.trace: provider stages land in the trace.
    HMS.usage() 即 telemetry.usage_of(hms)，HMS.trace() 即 telemetry.trace：提供者阶段落入该追踪。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

    Output / 输出：
        Assertions over the Usage snapshot before / after one add and over the RequestTrace.
        对一次 add 前后的 Usage 快照与 RequestTrace 的断言。

    State effect / 状态影响：
        One add() on the in-memory store.
        对内存存储做一次 add()。
    """
    before = hms.usage()
    assert isinstance(before, Usage) and before == usage_of(hms) and before.llm_calls == 0
    with hms.trace("add", "req-wiring-1") as record:
        hms.add("I like oat-milk lattes in the morning.", user_id="alice")
    assert telemetry.current_trace() is None
    assert record.request_id == "req-wiring-1" and record.operation == "add" and record.duration_ms >= 0.0
    stages = {stage.name for stage in record.stages}
    assert telemetry.STAGE_LLM_GENERATE in stages and stages & {telemetry.STAGE_EMBED, telemetry.STAGE_EMBED_BATCH}
    after = hms.usage()
    assert after == usage_of(hms) and after.llm_calls >= 1 and after.embedding_calls >= 1 and after.estimated is True
    assert after.total_tokens == after.input_tokens + after.output_tokens > 0


def test_state_property_and_delegates_mirror_state_api(hms, clock, events, scope4):
    """hms.state is one cached StateAPI bound to the facade; the twelve delegates carry StateAPI's signatures verbatim and return exactly what StateAPI returns.
    hms.state 是绑定到门面的同一个缓存 StateAPI；十二个委托方法逐字携带 StateAPI 的签名，且返回值与 StateAPI 完全相同。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions over signatures, identity of the cached StateAPI and three delegated calls.
        对签名、缓存 StateAPI 的同一性与三次委托调用的断言。

    State effect / 状态影响：
        Seeds S4 into the in-memory store.
        把 S4 灌入内存存储。
    """
    assert isinstance(HMS.state, property)
    api = hms.state
    assert isinstance(api, StateAPI) and api.hms is hms and hms.state is api
    for name in STATE_METHODS:
        assert inspect.signature(getattr(HMS, name)) == inspect.signature(getattr(StateAPI, name)), name
    _seed_s4(hms, clock, events, scope4)
    state, text = hms.work_state(S4_QUERY, filters=scope4, as_of=AS_OF)
    via_api, text_api = api.work_state(S4_QUERY, filters=scope4, as_of=AS_OF)
    assert isinstance(state, WorkState) and state.kind == StateKind.work_state.value
    assert state.model_dump(mode="json") == via_api.model_dump(mode="json") and text == text_api
    assert text == hms.state.work_state(S4_QUERY, filters=scope4, as_of=AS_OF)[1]
    report, report_text = hms.detective_report("why must retry.ts not be modified", filters=scope4, as_of=AS_OF)
    assert report.kind == StateKind.detective_report.value and report_text.startswith("# Detective Report")
    assert report_text == api.detective_report("why must retry.ts not be modified", filters=scope4, as_of=AS_OF)[1]
    point, point_text = hms.rollback_point(filters=scope4)
    assert point is None and point_text == "(none)"


def test_start_and_stop_daemon_on_the_facade():
    """start_daemon() needs an interval (argument or scheduler.daemon_interval_s), runs consolidate() on 'hms-daemon', returns the running daemon on a second call, and close() stops it before the store closes.
    start_daemon() 需要一个间隔（参数或 scheduler.daemon_interval_s），在 'hms-daemon' 上运行 consolidate()，第二次调用返回正在运行的守护进程，close() 在关闭存储之前把它停掉。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the error code, the daemon object, its counters and the thread state after close().
        对错误码、守护进程对象、其计数器与 close() 之后线程状态的断言。

    State effect / 状态影响：
        Short-lived daemon threads over in-memory stores.
        内存存储上的短生命周期守护线程。
    """
    hms = _memory_hms()
    with pytest.raises(ValidationError) as exc:
        hms.start_daemon()
    assert exc.value.error_code == "VALIDATION_009"
    assert hms.stop_daemon() is True  # nothing running / 没有运行中的守护进程
    hms.add("I like black coffee.", **SCOPE)
    daemon = hms.start_daemon(interval_s=0.02, filters=SCOPE)
    assert isinstance(daemon, StateDaemon) and daemon.is_running and daemon.thread_name == THREAD_NAME
    assert hms.start_daemon(interval_s=5.0) is daemon
    assert _wait_until(lambda: daemon.runs >= 2) and daemon.failures == 0
    assert daemon.last_report and set(daemon.last_report["phases"]) == {"orient", "collect", "consolidate", "prune"}
    assert hms.stop_daemon() is True and daemon.is_running is False
    assert hms.start_daemon(interval_s=0.02, filters=SCOPE) is not daemon
    hms.close()  # stops the second daemon first / 先停掉第二个守护进程
    assert hms.stop_daemon() is True

    configured = _memory_hms(scheduler={"daemon_interval_s": 0.02})
    configured.add("I like black coffee.", **SCOPE)
    daemon = configured.start_daemon(filters=SCOPE)
    assert daemon.interval_s == 0.02 and _wait_until(lambda: daemon.runs >= 1)
    configured.close()
    assert daemon.is_running is False and daemon.failures == 0


def test_root_exports_the_main_line_entry_points():
    """hms_elite exports AgentRuntime, StateDaemon, StateAPI, StateKind and Usage beside HMS / AsyncHMS.
    hms_elite 在 HMS / AsyncHMS 之外导出 AgentRuntime、StateDaemon、StateAPI、StateKind 与 Usage。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over __all__ and object identity.
        对 __all__ 与对象同一性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.daemon import StateDaemon as daemon_cls
    from hms_elite.runtime import AgentRuntime as runtime_cls
    from hms_elite.state.api import StateAPI as state_cls
    from hms_elite.state.kinds import StateKind as kind_cls
    from hms_elite.telemetry import Usage as usage_cls

    assert {"AgentRuntime", "StateDaemon", "StateAPI", "StateKind", "Usage"} <= set(hms_elite.__all__)
    assert hms_elite.__all__ == sorted(hms_elite.__all__, key=str.lower) or "HMS" in hms_elite.__all__
    assert (AgentRuntime, StateDaemon, StateAPI, StateKind, Usage) == (
        runtime_cls,
        daemon_cls,
        state_cls,
        kind_cls,
        usage_cls,
    )
