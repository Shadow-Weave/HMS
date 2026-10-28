"""Deterministic rendering of an AblationReport: a task × ablation markdown table and a JSON export.
AblationReport 的确定性渲染：任务 × 消融的 markdown 表与 JSON 导出。
"""

# Purpose (EN): Turn the run_ablation matrix into text a human (markdown) or a tool (JSON) can read.
# 目的（中）：把 run_ablation 的矩阵变成人（markdown）或工具（JSON）可读的文本。

# Input (EN): An AblationReport.
# 输入（中）：一个 AblationReport。

# Output (EN): Strings; render() is byte-stable across runs unless timing=True is requested.
# 输出（中）：字符串；除非请求 timing=True，render() 跨运行逐字节稳定。

# State effect (EN): None (pure formatting).
# 状态影响（中）：无（纯格式化）。

from __future__ import annotations

from hms_elite.eval.ablation import AblationReport, Cell

# Column titles of the four §12 ablations (HMS §12 Phase 5 wording).
# 四种 §12 消融的列标题（HMS §12 第五阶段的措辞）。
ABLATION_TITLES: dict[str, str] = {
    "axis": "Axis only",
    "graph": "Graph only",
    "axis_graph": "Axis + Graph",
    "axis_graph_support": "Axis + Graph + Support",
}


def _fmt(cell: Cell) -> str:
    """Score cell text ('ERR' when the cell errored).
    分数单元格文本（出错时为 'ERR'）。
    """
    return "ERR" if cell.error else f"{cell.score:.2f}"


def _row(name: str, values: list[str]) -> str:
    """One markdown table row.
    一行 markdown 表格。
    """
    return "| " + " | ".join([name, *values]) + " |"


def render(report: AblationReport, *, details: bool = False, timing: bool = False) -> str:
    """Render the task × ablation table (scores in [0, 1], a mean row) with optional per-cell misses and timing.
    渲染任务 × 消融表（[0, 1] 分数、一行均值），可选每单元格的未命中项与计时。

    Input / 输入：
        report: The AblationReport.
        report：AblationReport。

        details: Append a section listing every missed check per cell (and errors).
        details：追加一节，列出每个单元格未命中的检查项（以及错误）。

        timing: Append latency / LLM-call / token rows (latency is not deterministic).
        timing：追加延迟 / LLM 调用 / token 行（延迟不具确定性）。

    Output / 输出：
        Markdown text without a trailing newline.
        不带尾随换行的 markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """
    columns = [ABLATION_TITLES.get(a, a) for a in report.ablations]
    lines = ["# HMS §12 Phase 5 ablation", "", _row("task", columns), _row("---", ["---"] * len(columns))]
    for task in report.tasks:
        lines.append(_row(task, [_fmt(report.cell(task, a)) for a in report.ablations]))
    lines.append(_row("mean", [f"{report.mean(a):.2f}" for a in report.ablations]))
    if timing:
        lines.extend(["", "## Timing", "", _row("metric", columns), _row("---", ["---"] * len(columns))])
        for task in report.tasks:
            cells = [report.cell(task, a) for a in report.ablations]
            lines.append(
                _row(f"{task} latency_ms", [f"{c.latency_ms:.1f}" if c.latency_ms is not None else "-" for c in cells])
            )
            lines.append(
                _row(f"{task} llm_calls", [str(c.llm_calls) if c.llm_calls is not None else "-" for c in cells])
            )
            lines.append(_row(f"{task} tokens", [str(c.tokens) if c.tokens is not None else "-" for c in cells]))
    if details:
        lines.extend(["", "## Misses"])
        for task in report.tasks:
            for ablation in report.ablations:
                cell = report.cell(task, ablation)
                misses = [c for c in cell.checks if not c.get("hit")]
                if cell.error is None and not misses:
                    continue
                lines.append(f"- {task} / {ABLATION_TITLES.get(ablation, ablation)}")
                if cell.error:
                    lines.append(f"  - error: {cell.error}")
                for miss in misses:
                    lines.append(f"  - {miss['name']} ({miss['score']:.2f}): {miss['detail']}")
    return "\n".join(lines)


def to_json(report: AblationReport, *, timing: bool = True) -> str:
    """JSON export of the report (AblationReport.to_json).
    报告的 JSON 导出（AblationReport.to_json）。

    Input / 输入：
        report: The AblationReport.
        report：AblationReport。

        timing: Keep latency_ms values (False blanks them for byte-stable output).
        timing：保留 latency_ms（False 时抹去以得到逐字节稳定的输出）。

    Output / 输出：
        JSON string (sorted keys, indent 2, ensure_ascii=False).
        JSON 字符串（键排序、缩进 2、ensure_ascii=False）。

    State effect / 状态影响：
        None.
        无。
    """
    return report.to_json(timing=timing)


__all__ = ["ABLATION_TITLES", "render", "to_json"]
