"""Background consolidation loop: the HMS_arch §3.9 / §5.1 `daemon.py` running the StateScheduler off the foreground.
后台固化循环：HMS_arch §3.9 / §5.1 的 `daemon.py`，在前台之外运行 StateScheduler。
"""

# Purpose (EN): A thread named 'hms-daemon' that calls hms.consolidate(filters, phases) every interval so the four
#               phases Orient / Collect / Consolidate / Prune run asynchronously and never block foreground queries.
# 目的（中）：一个名为 'hms-daemon' 的线程，每个间隔调用一次 hms.consolidate(filters, phases)，使
#            Orient / Collect / Consolidate / Prune 四阶段异步运行，从不阻塞前台查询。

# Input (EN): An HMS facade, the interval in seconds, the phase list, optional scope filters and report / error hooks.
# 输入（中）：HMS 门面、以秒计的间隔、阶段列表、可选的范围过滤与报告 / 错误回调。

# Output (EN): Consolidation reports ({'anchor_id', 'phases'}) delivered to on_report and kept as last_report; counters
#              runs / failures / last_error.
# 输出（中）：交给 on_report 并保存在 last_report 的固化报告（{'anchor_id', 'phases'}）；计数器 runs / failures /
#            last_error。

# State effect (EN): Every store write happens inside hms.consolidate (the StateStore lock serializes it against the
#                    foreground); the daemon itself only owns its thread, stop event and counters.
# 状态影响（中）：所有存储写入都发生在 hms.consolidate 内（StateStore 的锁把它与前台串行化）；守护进程自身只拥有
#                线程、停止事件与计数器。

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from hms_elite.enums import Phase
from hms_elite.exceptions import ValidationError

logger = logging.getLogger(__name__)

THREAD_NAME = "hms-daemon"
DEFAULT_PHASES = (Phase.orient.value, Phase.collect.value, Phase.consolidate.value, Phase.prune.value)


class StateDaemon:
    """Periodic StateScheduler runner on a background thread (start / stop / run_once / is_running).
    在后台线程上周期运行 StateScheduler（start / stop / run_once / is_running）。
    """

    def __init__(
        self,
        hms: Any,
        *,
        interval_s: float = 60.0,
        phases: list[str] | tuple[str, ...] | None = None,
        filters: dict | None = None,
        on_report: Callable[[dict], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        run_at_start: bool = True,
        thread_name: str = THREAD_NAME,
    ) -> None:
        """Configure the loop (nothing runs until start()).
        配置循环（start() 之前不运行任何东西）。

        Input / 输入：
            hms: The HMS facade (only consolidate() is called).
            hms：HMS 门面（只调用 consolidate()）。

            interval_s: Seconds between two runs (> 0, VALIDATION_009 otherwise).
            interval_s：两次运行之间的秒数（> 0，否则 VALIDATION_009）。

            phases: Phase names in run order (None → orient/collect/consolidate/prune; unknown → VALIDATION_013).
            phases：按运行顺序的阶段名（None → orient/collect/consolidate/prune；未知 → VALIDATION_013）。

            filters: Full or partial scope filters (None → every scope in the store).
            filters：完整或部分范围过滤（None → 存储中的每个范围）。

            on_report: Called with each consolidate() report on the daemon thread.
            on_report：在守护线程上以每个 consolidate() 报告调用。

            on_error: Called with each exception on the daemon thread (after logger.exception).
            on_error：在守护线程上以每个异常调用（在 logger.exception 之后）。

            run_at_start: Run once immediately after start(), then every interval.
            run_at_start：start() 后立即运行一次，之后每个间隔运行一次。

            thread_name: Name of the background thread.
            thread_name：后台线程名。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or interval_s <= 0:
            raise ValidationError(
                f"interval_s must be a positive number, got {interval_s!r}",
                "VALIDATION_009",
                details={"interval_s": interval_s},
            )
        wanted = [str(getattr(p, "value", p)) for p in (phases if phases is not None else DEFAULT_PHASES)]
        unknown = [p for p in wanted if p not in Phase.values()]
        if unknown:
            raise ValidationError(
                f"unknown phases {unknown}", "VALIDATION_013", details={"phases": unknown, "supported": Phase.values()}
            )
        self.hms = hms
        self.interval_s = float(interval_s)
        self.phases: list[str] = wanted
        self.filters = dict(filters) if filters else None
        self.on_report = on_report
        self.on_error = on_error
        self.run_at_start = bool(run_at_start)
        self.thread_name = thread_name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._runs = 0
        self._failures = 0
        self._last_report: dict | None = None
        self._last_error: BaseException | None = None

    # ------------------------------------------------------------------ counters
    # ------------------------------------------------------------------ 计数器
    @property
    def runs(self) -> int:
        """Completed consolidate() calls (loop and run_once).
        已完成的 consolidate() 调用次数（循环与 run_once）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Count.
            计数。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return self._runs

    @property
    def failures(self) -> int:
        """consolidate() calls that raised.
        抛出异常的 consolidate() 调用次数。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Count.
            计数。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return self._failures

    @property
    def last_report(self) -> dict | None:
        """The most recent consolidate() report.
        最近一次 consolidate() 的报告。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'anchor_id', 'phases'} or None.
            {'anchor_id', 'phases'} 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return self._last_report

    @property
    def last_error(self) -> BaseException | None:
        """The most recent exception raised by consolidate().
        consolidate() 最近抛出的异常。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The exception or None.
            异常或 None。

        State effect / 状态影响：
            None.
            无。
        """
        with self._lock:
            return self._last_error

    @property
    def is_running(self) -> bool:
        """Whether the background thread is alive.
        后台线程是否存活。

        Input / 输入：
            none.
            无。

        Output / 输出：
            True while the thread runs.
            线程运行期间为 True。

        State effect / 状态影响：
            None.
            无。
        """
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------ control
    # ------------------------------------------------------------------ 控制
    def run_once(self) -> dict:
        """Run the configured phases once, synchronously, on the calling thread.
        在调用线程上同步运行一次配置的阶段。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The consolidate() report ({'anchor_id', 'phases'}).
            consolidate() 的报告（{'anchor_id', 'phases'}）。

        State effect / 状态影响：
            The phases' writes; counters updated; exceptions propagate (the loop catches them, run_once does not).
            各阶段的写入；更新计数器；异常向上传播（循环会捕获，run_once 不会）。
        """
        try:
            report = self.hms.consolidate(filters=self.filters, phases=list(self.phases))
        except BaseException as exc:
            with self._lock:
                self._failures += 1
                self._last_error = exc
            raise
        with self._lock:
            self._runs += 1
            self._last_report = report
        if self.on_report is not None:
            self.on_report(report)
        return report

    def start(self) -> bool:
        """Start the background thread.
        启动后台线程。

        Input / 输入：
            none.
            无。

        Output / 输出：
            True when a thread was started, False when one is already running.
            启动了线程时为 True，已在运行时为 False。

        State effect / 状态影响：
            Spawns a daemon thread named thread_name.
            派生一个名为 thread_name 的守护线程。
        """
        if self.is_running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.thread_name, daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float | None = None) -> bool:
        """Ask the loop to stop and wait for the thread (returns within one interval plus the running phase).
        请求循环停止并等待线程（在一个间隔加当前运行阶段的时间内返回）。

        Input / 输入：
            timeout: Max seconds to wait for the thread (None → until it exits).
            timeout：等待线程的最长秒数（None → 直到退出）。

        Output / 输出：
            True when the thread has exited.
            线程已退出时为 True。

        State effect / 状态影响：
            Sets the stop event; no store write.
            设置停止事件；不写存储。
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _loop(self) -> None:
        """Thread body: run, wait interval, repeat; an exception is logged and the loop continues.
        线程主体：运行、等待间隔、重复；异常被记录后循环继续。
        """
        if not self.run_at_start and self._stop.wait(self.interval_s):
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                # A failed pass must never kill the loop: log, notify, and try again after the interval.
                # 一次失败的运行绝不能杀死循环：记录、通知，然后在下一个间隔重试。
                logger.exception("hms-daemon consolidate failed: %s", exc)
                if self.on_error is not None:
                    try:
                        self.on_error(exc)
                    except Exception:
                        logger.exception("hms-daemon on_error hook failed")
            if self._stop.wait(self.interval_s):
                break

    def __enter__(self) -> "StateDaemon":
        """start() on entry.
        进入时 start()。
        """
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """stop() on exit.
        退出时 stop()。
        """
        self.stop()
