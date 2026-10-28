"""Tests for hms_elite.daemon: StateDaemon start / stop / run_once / error survival / foreground concurrency.
hms_elite.daemon 的测试：StateDaemon 的 start / stop / run_once / 异常存活 / 与前台并发。
"""

# Purpose (EN): Prove the HMS_arch §3.9 background loop: phases validated, reports delivered, the thread named
#               hms-daemon, stop() returning within one interval, exceptions logged without killing the loop, and a
#               foreground add/search racing consolidate for three seconds without errors or lost rows.
# 目的（中）：证明 HMS_arch §3.9 的后台循环：阶段校验、报告投递、线程名 hms-daemon、stop() 在一个间隔内返回、
#            异常被记录而不杀死循环、前台 add/search 与 consolidate 并发三秒无异常且不丢行。

# Input (EN): A stub facade recording consolidate() calls, and a real offline HMS (rule + hash + memory index).
# 输入（中）：记录 consolidate() 调用的桩门面，以及真实的离线 HMS（rule + hash + memory 索引）。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): In-memory stores and short-lived threads only.
# 状态影响（中）：仅内存存储与短生命周期线程。

from __future__ import annotations

import logging
import threading
import time

import pytest

from hms_elite.daemon import DEFAULT_PHASES, THREAD_NAME, StateDaemon
from hms_elite.enums import Phase
from hms_elite.exceptions import ValidationError
from hms_elite.memory.main import HMS

SCOPE = {"user_id": "alice", "agent_id": "coder", "run_id": "run-daemon"}


class _StubHMS:
    """A facade stand-in recording consolidate() calls and optionally failing the first N of them.
    记录 consolidate() 调用、可让前 N 次失败的门面替身。
    """

    def __init__(self, *, fail_first: int = 0, sleep_s: float = 0.0) -> None:
        """Configure the stub.
        配置桩对象。
        """
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.sleep_s = sleep_s
        self.lock = threading.Lock()

    def consolidate(self, *, filters=None, phases=None) -> dict:
        """Record the call; raise RuntimeError while fail_first > 0.
        记录调用；fail_first > 0 时抛出 RuntimeError。
        """
        with self.lock:
            self.calls.append({"filters": filters, "phases": phases, "thread": threading.current_thread().name})
            n = len(self.calls)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if n <= self.fail_first:
            raise RuntimeError(f"boom {n}")
        return {"anchor_id": f"anc_{n:06d}", "phases": {p: {"n": n} for p in (phases or [])}}


def _wait_until(predicate, *, timeout: float = 5.0) -> bool:
    """Poll `predicate` every 5 ms until it is true or `timeout` elapses.
    每 5 毫秒轮询一次 `predicate`，直到为真或超时。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_daemon_validates_interval_and_phases():
    """interval_s must be positive (VALIDATION_009) and phases must be Phase values (VALIDATION_013).
    interval_s 必须为正（VALIDATION_009），phases 必须是 Phase 的值（VALIDATION_013）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over error codes and the normalized phase list.
        对错误码与归一化阶段列表的断言。

    State effect / 状态影响：
        None.
        无。
    """
    stub = _StubHMS()
    for bad in (0, -1.0, True, "5"):
        with pytest.raises(ValidationError) as exc:
            StateDaemon(stub, interval_s=bad)
        assert exc.value.error_code == "VALIDATION_009"
    with pytest.raises(ValidationError) as unknown:
        StateDaemon(stub, interval_s=1, phases=["orient", "nope"])
    assert unknown.value.error_code == "VALIDATION_013" and unknown.value.details["phases"] == ["nope"]
    d = StateDaemon(stub, interval_s=1, phases=[Phase.prune, "orient"], filters=SCOPE)
    assert d.phases == ["prune", "orient"] and d.filters == SCOPE and d.is_running is False
    assert (
        StateDaemon(stub, interval_s=1).phases == list(DEFAULT_PHASES) == ["orient", "collect", "consolidate", "prune"]
    )
    assert StateDaemon(stub, interval_s=1, filters={}).filters is None


def test_run_once_reports_and_counts():
    """run_once() calls consolidate(filters, phases) synchronously, stores the report and notifies on_report.
    run_once() 同步调用 consolidate(filters, phases)，保存报告并通知 on_report。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the stub's recorded call, the counters and the hook.
        对桩记录的调用、计数器与回调的断言。

    State effect / 状态影响：
        None.
        无。
    """
    stub = _StubHMS()
    seen: list[dict] = []
    d = StateDaemon(stub, interval_s=10, phases=["orient", "collect"], filters=SCOPE, on_report=seen.append)
    report = d.run_once()
    assert stub.calls == [
        {"filters": SCOPE, "phases": ["orient", "collect"], "thread": threading.current_thread().name}
    ]
    assert report["anchor_id"] == "anc_000001" and set(report["phases"]) == {"orient", "collect"}
    assert seen == [report] and d.runs == 1 and d.failures == 0 and d.last_report is report and d.last_error is None
    assert d.is_running is False
    failing = StateDaemon(_StubHMS(fail_first=1), interval_s=10)
    with pytest.raises(RuntimeError):
        failing.run_once()
    assert failing.runs == 0 and failing.failures == 1 and isinstance(failing.last_error, RuntimeError)


def test_start_stop_loop_runs_on_named_thread():
    """start() spawns 'hms-daemon', runs immediately and every interval, start() twice is a no-op, stop() joins it.
    start() 派生 'hms-daemon'，立即运行并每个间隔运行一次，重复 start() 无操作，stop() 等待其结束。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over thread name, run count and running state.
        对线程名、运行次数与运行状态的断言。

    State effect / 状态影响：
        Short-lived background thread.
        短生命周期的后台线程。
    """
    stub = _StubHMS()
    reports: list[dict] = []
    with StateDaemon(stub, interval_s=0.02, filters=SCOPE, on_report=reports.append) as d:
        assert d.is_running is True and d.start() is False
        assert _wait_until(lambda: d.runs >= 3)
        names = {t.name for t in threading.enumerate()}
        assert THREAD_NAME in names and d.thread_name == THREAD_NAME
    assert d.is_running is False
    with stub.lock:
        calls = list(stub.calls)
    assert len(calls) >= 3 and {c["thread"] for c in calls} == {THREAD_NAME}
    assert all(c["filters"] == SCOPE and c["phases"] == list(DEFAULT_PHASES) for c in calls)
    assert len(reports) == d.runs and d.failures == 0
    assert d.start() is True and d.is_running is True and d.stop() is True and d.is_running is False
    assert d.stop() is True
    lazy = StateDaemon(_StubHMS(), interval_s=0.01, run_at_start=False, thread_name="hms-daemon-lazy")
    lazy.start()
    assert _wait_until(lambda: lazy.runs >= 1) and lazy.stop() is True


def test_stop_returns_within_one_interval():
    """stop() does not wait for the next tick: with a 5 s interval it returns well under a second.
    stop() 不等待下一次触发：间隔 5 秒时也在远小于一秒内返回。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertion over the elapsed stop() time.
        对 stop() 耗时的断言。

    State effect / 状态影响：
        Short-lived background thread.
        短生命周期的后台线程。
    """
    d = StateDaemon(_StubHMS(), interval_s=5.0)
    d.start()
    assert _wait_until(lambda: d.runs >= 1)
    t0 = time.monotonic()
    assert d.stop() is True
    assert time.monotonic() - t0 < 1.0 and d.is_running is False and d.runs == 1


def test_exception_is_logged_and_loop_continues(caplog):
    """A consolidate() exception is logged with logger.exception, handed to on_error, and the loop keeps running.
    consolidate() 抛出的异常经 logger.exception 记录、交给 on_error，循环继续运行。

    Input / 输入：
        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions over counters, the log record and the thread state.
        对计数器、日志记录与线程状态的断言。

    State effect / 状态影响：
        Short-lived background thread.
        短生命周期的后台线程。
    """
    errors: list[BaseException] = []
    stub = _StubHMS(fail_first=2)
    d = StateDaemon(stub, interval_s=0.01, on_error=errors.append)
    with caplog.at_level(logging.ERROR, logger="hms_elite.daemon"):
        d.start()
        assert _wait_until(lambda: d.runs >= 2)
        assert d.is_running is True
        d.stop()
    assert d.failures == 2 and [str(e) for e in errors] == ["boom 1", "boom 2"]
    assert isinstance(d.last_error, RuntimeError) and d.last_report is not None
    records = [r for r in caplog.records if r.name == "hms_elite.daemon" and r.levelno == logging.ERROR]
    assert len(records) == 2 and all("consolidate failed" in r.getMessage() and r.exc_info for r in records)


def test_foreground_add_search_race_with_background_consolidate():
    """Three seconds of foreground add()/search() racing the daemon's consolidate(): no errors, no lost rows.
    前台 add()/search() 与守护进程的 consolidate() 并发三秒：无异常、不丢行。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over exception lists, daemon counters and the message-row count.
        对异常列表、守护进程计数器与消息行数的断言。

    State effect / 状态影响：
        In-memory HMS store written from two threads.
        由两个线程写入的内存 HMS 存储。
    """
    hms = HMS({"store": {"path": ":memory:"}, "vector_store": {"provider": "memory"}})
    hms.add("Background: keep the billing API stable.", **SCOPE)
    foreground_errors: list[BaseException] = []
    added = 0
    daemon = StateDaemon(hms, interval_s=0.01, filters=SCOPE)
    daemon.start()
    deadline = time.monotonic() + 3.0
    try:
        while time.monotonic() < deadline:
            try:
                hms.add(f"Note {added}: retry.ts step {added} of the billing run.", infer=False, **SCOPE)
                added += 1
                if added % 5 == 0:
                    hits = hms.search("retry.ts billing", filters=SCOPE, top_k=5)["results"]
                    assert isinstance(hits, list)
            except Exception as exc:
                # The test collects every foreground failure instead of aborting mid-race.
                # 测试收集每个前台失败，而不是在并发过程中中止。
                foreground_errors.append(exc)
                break
    finally:
        assert daemon.stop(timeout=10) is True
    assert foreground_errors == [] and daemon.failures == 0 and daemon.last_error is None
    assert daemon.runs >= 1 and added >= 10
    assert daemon.last_report is not None and set(daemon.last_report["phases"]) == set(DEFAULT_PHASES)
    rows = hms.get_all(filters=SCOPE, top_k=added + 1000)["results"]
    notes = sorted(int(r["memory"].split(":")[0].split()[1]) for r in rows if r["memory"].startswith("Note "))
    assert notes == list(range(added))
    final = daemon.run_once()
    assert final["anchor_id"].startswith("anc_") and daemon.runs >= 2
    hms.close()
