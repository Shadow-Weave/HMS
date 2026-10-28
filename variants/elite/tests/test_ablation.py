"""Tests for the HMS §12 Phase 5 ablation framework: four ablations × nine §10 tasks, the §10 ordering, the report.
HMS §12 第五阶段消融框架的测试：四种消融 × §10 的九个任务、§10 表的序关系、报告。
"""

# Purpose (EN): Assert that every ablation runs every task offline with a 0–1 score, that Axis + Graph + Support is
#   never below Axis only on handoff / rollback / conflict / evidence audit (HMS §10), and that rendering is stable.
# 目的（中）：断言每种消融都能离线跑完每个任务并给出 0–1 分、Axis + Graph + Support 在交接 / 回滚 / 冲突 / 证据审计上
#   不低于 Axis only（HMS §10），以及报告渲染稳定。

# Input (EN): hms_elite.eval, tests/fixtures/events/*.json, the conftest index_provider fixture and a ticking clock.
# 输入（中）：hms_elite.eval、tests/fixtures/events/*.json、conftest 的 index_provider 夹具与一个递增时钟。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory stores and tmp_path.
# 状态影响（中）：除内存存储与 tmp_path 外无副作用。

from __future__ import annotations

import json
import os
import pathlib
from datetime import timedelta
from typing import Any, Callable

import pytest

from hms_elite.configs.base import HMSConfig
from hms_elite.enums import ABLATION_OF_MODE, CompileMode
from hms_elite.eval import (
    TASK_BY_NAME,
    TASKS,
    Ablation,
    AblationReport,
    Check,
    Probe,
    TaskScore,
    longmemeval_available,
    longmemeval_tasks,
    render,
    run_ablation,
    task_from_longmemeval,
    to_json,
)
from hms_elite.memory.main import HMS
from hms_elite.memory.storage import StateStore
from hms_elite.timeutil import parse_iso, to_utc_iso
from tests.conftest import SequentialIds

TASK_NAMES = [
    "preference_recall",
    "temporal",
    "multi_session",
    "handoff",
    "rollback",
    "conflict_revision",
    "evidence_audit",
    "skill_distillation",
    "world_model_transition",
]
# HMS §10: the rows where "只用 Graph" and "HMS 双平面" are 强 while "只用 Observation" is 弱 / 中.
# HMS §10：「只用 Graph」与「HMS 双平面」为强、而「只用 Observation」为弱 / 中的那些行。
STATE_TASKS = ["handoff", "rollback", "conflict_revision", "evidence_audit"]
_REPORTS: dict[str, AblationReport] = {}


class _TickingClock:
    """A deterministic clock that advances one second per now() call (start 2026-09-02T10:00:00+00:00).
    每次 now() 前进一秒的确定性时钟（起点 2026-09-02T10:00:00+00:00）。
    """

    def __init__(self) -> None:
        """Start at the §19 instant.
        从 §19 的起点时刻开始。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self._t = parse_iso("2026-09-02T10:00:00+00:00")

    def now(self) -> str:
        """Advance by one second and return the instant.
        前进一秒并返回时刻。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Fixed-width UTC ISO string.
            定宽 UTC ISO 字符串。

        State effect / 状态影响：
            Mutates the clock.
            修改时钟。
        """
        self._t = self._t + timedelta(seconds=1)
        return to_utc_iso(self._t)


def _factory(index_provider: str) -> Callable[[], HMS]:
    """Zero-argument HMS factory on the rule LLM + hash embedder + the given index, with deterministic clock and ids.
    基于规则 LLM + 哈希嵌入器 + 给定索引、带确定性时钟与 id 的无参 HMS 工厂。
    """

    def make() -> HMS:
        """Build one fresh HMS.
        构建一个新的 HMS。
        """
        store = StateStore(":memory:", now=_TickingClock().now, id_factory=SequentialIds())
        return HMS(HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}), store=store)

    return make


@pytest.fixture
def report(index_provider: str) -> AblationReport:
    """The full nine × four matrix for an index provider (computed once per provider, then reused).
    某索引提供者的完整九 × 四矩阵（每个提供者只计算一次，之后复用）。
    """
    if index_provider not in _REPORTS:
        _REPORTS[index_provider] = run_ablation(_factory(index_provider))
    return _REPORTS[index_provider]


def _misses(report: AblationReport, task: str, ablation: Ablation) -> list[str]:
    """Readable miss lines of one cell.
    某单元格可读的未命中行。
    """
    cell = report.cell(task, ablation)
    return [f"{c['name']}: {c['detail']}" for c in cell.checks if not c["hit"]] + ([cell.error] if cell.error else [])


def test_ablation_enum_aligns_with_compile_modes():
    """Ablation values are the ABLATION_OF_MODE labels; switches map to fast / axis_graph / deep and nothing else.
    Ablation 的取值就是 ABLATION_OF_MODE 的标签；开关映射到 fast / axis_graph / deep，别无其他。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert [a.name for a in Ablation] == ["axis_only", "graph_only", "axis_graph", "axis_graph_support"]
    assert set(ABLATION_OF_MODE.values()) < set(Ablation.values())
    assert Ablation.of_mode(CompileMode.fast) is Ablation.axis_only
    assert Ablation.of_mode("axis_graph") is Ablation.axis_graph
    assert Ablation.of_mode(CompileMode.deep) is Ablation.axis_graph_support
    assert Ablation.axis_only.compile_mode == "fast"
    assert Ablation.axis_graph.compile_mode == "axis_graph"
    assert Ablation.axis_graph_support.compile_mode == "deep"
    assert Ablation.graph_only.compile_mode is None
    for ablation in Ablation:
        sw = ablation.switches
        if sw.compile_mode is not None:
            assert ABLATION_OF_MODE[sw.compile_mode] == ablation.value
        assert sw.use_support <= sw.use_graph
        assert sw.use_axis or sw.use_graph
    assert Ablation.graph_only.switches.use_axis is False and Ablation.graph_only.switches.use_graph is True
    with pytest.raises(KeyError):
        Ablation.of_mode("auto")


def test_task_table_matches_hms_section_10():
    """TASKS carries exactly the nine §10 rows, in order, each with a build and a run callable.
    TASKS 恰好按顺序包含 §10 的九行，每行都有 build 与 run 可调用对象。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert [t.name for t in TASKS] == TASK_NAMES
    assert set(TASK_BY_NAME) == set(TASK_NAMES)
    assert all(callable(t.build) and callable(t.run) and t.description for t in TASKS)


def test_matrix_runs_every_task_under_every_ablation(report, index_provider):
    """Nine tasks × four ablations all complete offline with a 0–1 score, explained checks and no error cell.
    九个任务 × 四种消融全部离线完成，给出 0–1 分、可解释检查项且没有错误单元格。

    Input / 输入：
        report: The matrix fixture.
        report：矩阵夹具。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert report.tasks == TASK_NAMES and report.ablations == Ablation.values()
    assert len(report.cells) == 9 * 4
    errors = [(c.task, c.ablation, c.error) for c in report.cells if c.error]
    assert errors == [], errors
    for cell in report.cells:
        assert 0.0 <= cell.score <= 1.0
        assert cell.checks and all(c["detail"] for c in cell.checks)
        assert cell.score == round(sum(c["score"] for c in cell.checks) / len(cell.checks), 4)
        assert cell.latency_ms is not None and cell.latency_ms >= 0.0
        assert cell.llm_calls == 0, (cell.task, cell.ablation, cell.llm_calls)
        assert cell.tokens == 0, (cell.task, cell.ablation, cell.tokens)
    # Every ablation is strong somewhere and the full system dominates every ablation on average (HMS §10).
    # 每种消融都有其强项，而完整系统在均值上支配每种消融（HMS §10）。
    full = report.mean(Ablation.axis_graph_support)
    assert all(full >= report.mean(a) for a in Ablation), {a.value: report.mean(a) for a in Ablation}
    assert report.score("preference_recall", Ablation.axis_only) == 1.0
    assert report.score("temporal", Ablation.axis_only) > report.score("temporal", Ablation.graph_only)
    assert report.score("rollback", Ablation.graph_only) > report.score("rollback", Ablation.axis_only)


@pytest.mark.parametrize("task", STATE_TASKS)
def test_support_never_below_axis_only_on_state_tasks(report, task):
    """HMS §10: Axis + Graph + Support >= Axis only on handoff / rollback / conflict-revision / evidence audit.
    HMS §10：在交接 / 回滚 / 冲突修订 / 证据审计上 Axis + Graph + Support >= Axis only。

    Input / 输入：
        report: The matrix fixture.
        report：矩阵夹具。

        task: One of the four state-inheritance tasks.
        task：四个状态继承任务之一。

    Output / 输出：
        Assertions reporting both numbers and the misses on failure.
        失败时报出两个数字与未命中项的断言。

    State effect / 状态影响：
        None.
        无。
    """
    full = report.score(task, Ablation.axis_graph_support)
    axis = report.score(task, Ablation.axis_only)
    assert full >= axis, (
        f"{task}: axis_graph_support={full} < axis_only={axis}; "
        f"support misses={_misses(report, task, Ablation.axis_graph_support)}; "
        f"axis misses={_misses(report, task, Ablation.axis_only)}"
    )
    # The same ordering holds along the ladder Axis → Axis + Graph → Axis + Graph + Support.
    # 同样的序关系沿 Axis → Axis + Graph → Axis + Graph + Support 的阶梯成立。
    graph = report.score(task, Ablation.axis_graph)
    assert axis <= graph <= full, (task, axis, graph, full)


def test_probe_switches_are_read_only_and_honoured(index_provider):
    """axis_only never reads graph tables and has no why(); graph_only never runs the axis planner; support adds PC-IR.
    axis_only 从不读图表且没有 why()；graph_only 从不运行轴规划器；support 增加 PC-IR。

    Input / 输入：
        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None outside the in-memory HMS.
        除内存 HMS 外无副作用。
    """
    hms = _factory(index_provider)()
    try:
        ctx = TASK_BY_NAME["handoff"].build(hms)
        before = hms.store.stats()
        for ablation in Ablation:
            probe = Probe(hms, ablation)
            ir = probe.compile("fix checkout webhook retry", filters=ctx["filters"])
            assert ir.retrieval["ablation"] == ablation.value
            assert (probe.pcir(ir) is None) == (not ablation.switches.use_support)
            assert (probe.why(ctx["ep1"]) is None) == (not ablation.switches.use_graph)
            assert (probe.axis_query({"filters": ctx["filters"]}) is None) == (not ablation.switches.use_axis)
            assert ir.items() and all(i.anchor_ids for i in ir.items())
        assert hms.store.stats() == before
        trace = getattr(hms.store, "trace", None)
        if trace is not None:
            trace.reset()
            Probe(hms, Ablation.axis_only).compile("fix checkout webhook retry", filters=ctx["filters"])
            assert not {t for t in trace.tables() if t.startswith("graph_")}
            trace.reset()
            graph_ir = Probe(hms, Ablation.graph_only).compile("fix checkout webhook retry", filters=ctx["filters"])
            assert "atom_vectors" not in trace.tables()
            assert all(i.source in {"graph", "system"} and i.score == 0.0 for i in graph_ir.items())
    finally:
        hms.close()


def test_report_render_is_deterministic(index_provider):
    """Two independent runs render byte-identical markdown and JSON (timing excluded), with every task and column.
    两次独立运行渲染出逐字节相同的 markdown 与 JSON（不含计时），且包含每个任务与每一列。

    Input / 输入：
        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None outside in-memory HMS instances.
        除内存 HMS 实例外无副作用。
    """
    first = run_ablation(_factory(index_provider))
    second = run_ablation(_factory(index_provider))
    assert render(first) == render(second)
    assert render(first, details=True) == render(second, details=True)
    assert to_json(first, timing=False) == to_json(second, timing=False)
    text = render(first)
    lines = text.split("\n")
    assert lines[0] == "# HMS §12 Phase 5 ablation"
    assert lines[2] == "| task | Axis only | Graph only | Axis + Graph | Axis + Graph + Support |"
    assert lines[-1].startswith("| mean |")
    assert all(any(line.startswith(f"| {task} |") for line in lines) for task in TASK_NAMES)
    assert "## Timing" not in text and "## Timing" in render(first, timing=True)
    data = json.loads(to_json(first, timing=False))
    assert data["tasks"] == TASK_NAMES and len(data["cells"]) == 36
    assert all(cell["latency_ms"] is None for cell in data["cells"])
    assert all(cell["latency_ms"] is not None for cell in json.loads(to_json(first))["cells"])


def test_subset_and_error_cells():
    """A task and ablation subset runs alone; a task whose build raises becomes error cells, not a crash.
    任务与消融子集可单独运行；构建抛错的任务变成错误单元格而非崩溃。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None outside in-memory HMS instances.
        除内存 HMS 实例外无副作用。
    """

    def boom(_hms: Any) -> dict:
        """A build that always fails.
        总是失败的构建。
        """
        raise RuntimeError("fixture missing")

    broken = type(TASKS[0])("broken", "always fails", boom, TASKS[0].run)
    rep = run_ablation(_factory("memory"), tasks=[TASK_BY_NAME["preference_recall"], broken], ablations=["axis"])
    assert rep.ablations == ["axis"] and rep.tasks == ["preference_recall", "broken"]
    assert rep.score("preference_recall", "axis") == 1.0
    cell = rep.cell("broken", Ablation.axis_only)
    assert cell.score == 0.0 and cell.error is not None and "fixture missing" in cell.error
    assert "ERR" in render(rep) and "error: build failed" in render(rep, details=True)
    with pytest.raises(KeyError):
        rep.cell("preference_recall", "graph")


def test_task_score_arithmetic():
    """TaskScore averages partial checks and exposes misses; an empty list scores 0.
    TaskScore 对部分分检查项取均值并暴露未命中项；空列表得 0 分。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    score = TaskScore.from_checks([Check("a", 1.0, "ok"), Check("b", 0.5, "half"), Check("c", 0.0, "no")])
    assert score.value == 0.5 and [c.name for c in score.misses()] == ["b", "c"]
    assert Check("a", 1.0, "ok").hit and not Check("b", 0.5, "half").hit
    assert TaskScore.from_checks([]).value == 0.0
    assert TaskScore.from_checks([Check("x", 7.0, "clamped")]).value == 1.0


def test_longmemeval_adapter_skeleton(tmp_path: pathlib.Path):
    """A synthetic LongMemEval entry converts to a Task and scores by answer coverage; a missing file yields no tasks.
    合成的 LongMemEval 条目可转为 Task 并按答案覆盖率打分；文件缺失时不产生任务。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes one JSON file under tmp_path.
        在 tmp_path 下写入一个 JSON 文件。
    """
    missing = tmp_path / "longmemeval_absent.json"
    assert not longmemeval_available(missing) and longmemeval_tasks(missing) == []
    entry = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "what coffee does the user like",
        "answer": "black coffee",
        "question_date": "2026/06/01 (Mon) 09:00",
        "haystack_dates": ["2026/04/02 (Thu) 08:00", "2026/04/20 (Mon) 08:00"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I like oat-milk lattes in the morning."}],
            [{"role": "user", "content": "Actually I switched to black coffee now."}],
        ],
        "answer_session_ids": ["s2"],
    }
    path = tmp_path / "longmemeval_tiny.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    tasks = longmemeval_tasks(path, limit=5)
    assert [t.name for t in tasks] == ["longmemeval:q1"] and tasks[0].description == "single-session-user"
    assert task_from_longmemeval(entry).name == "longmemeval:q1"
    rep = run_ablation(_factory("memory"), tasks=tasks, ablations=[Ablation.axis_only, Ablation.axis_graph_support])
    for ablation in ("axis", "axis_graph_support"):
        cell = rep.cell("longmemeval:q1", ablation)
        assert cell.error is None and cell.score == 1.0
        assert [c["name"] for c in cell.checks] == ["answer_coverage"] and "2/2" in cell.checks[0]["detail"]
    assert longmemeval_tasks(path, limit=0) == []


def test_longmemeval_real_dataset_when_present():
    """Run the adapter on a real LongMemEval file only when HMS_LONGMEMEVAL_PATH points at one (never downloaded).
    仅当 HMS_LONGMEMEVAL_PATH 指向真实的 LongMemEval 文件时运行适配器（从不下载）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions, or a skip.
        断言，或跳过。

    State effect / 状态影响：
        None outside in-memory HMS instances.
        除内存 HMS 实例外无副作用。
    """
    path = os.environ.get("HMS_LONGMEMEVAL_PATH")
    if not path or not longmemeval_available(path):
        pytest.skip("LongMemEval dataset not available (set HMS_LONGMEMEVAL_PATH to a local JSON file)")
    tasks = longmemeval_tasks(path, limit=3)
    assert tasks
    rep = run_ablation(_factory("memory"), tasks=tasks, ablations=[Ablation.axis_only])
    assert all(c.error is None and 0.0 <= c.score <= 1.0 for c in rep.cells)
