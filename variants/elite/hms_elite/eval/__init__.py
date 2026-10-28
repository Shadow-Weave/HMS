"""HMS §12 Phase 5 benchmark and ablation package: ablations, the nine §10 tasks, the runner and the report.
HMS §12 第五阶段基准与消融包：消融、§10 的九个任务、运行器与报告。
"""

# Purpose (EN): Re-export the public surface of the evaluation plane.
# 目的（中）：重新导出评估平面的公开接口。

# Input (EN): Nothing at import time.
# 输入（中）：导入时不接收任何输入。

# Output (EN): The names listed in __all__.
# 输出（中）：__all__ 中列出的名称。

# State effect (EN): None; importing has no side effects.
# 状态影响（中）：无；导入没有副作用。

from hms_elite.eval.ablation import (
    MODE_OF_ABLATION,
    Ablation,
    AblationReport,
    AblationSwitches,
    Cell,
    Check,
    Probe,
    Task,
    TaskScore,
    run_ablation,
)
from hms_elite.eval.report import ABLATION_TITLES, render, to_json
from hms_elite.eval.tasks import TASK_BY_NAME, TASKS, longmemeval_available, longmemeval_tasks, task_from_longmemeval

__all__ = [
    "ABLATION_TITLES",
    "MODE_OF_ABLATION",
    "TASKS",
    "TASK_BY_NAME",
    "Ablation",
    "AblationReport",
    "AblationSwitches",
    "Cell",
    "Check",
    "Probe",
    "Task",
    "TaskScore",
    "longmemeval_available",
    "longmemeval_tasks",
    "render",
    "run_ablation",
    "task_from_longmemeval",
    "to_json",
]
