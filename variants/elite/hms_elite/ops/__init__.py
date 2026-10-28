"""Operations plane: feedback, checkpoint/rollback, handoff, transition export and the synchronous StateScheduler.
运维平面：反馈、检查点/回滚、交接、状态转移导出与同步调度器。
"""

# Purpose (EN): Re-export the public entry points of hms_elite.ops (§16).
# 目的（中）：重新导出 hms_elite.ops 的公开入口（§16）。

# Input (EN): Nothing (package init).
# 输入（中）：无（包初始化）。

# Output (EN): FeedbackProcessor, revoke, checkpoint, rollback, build_state_package, import_state_package,
#              export_transitions, ep_transition, interface_ep_transition, StateScheduler, purge.
# 输出（中）：FeedbackProcessor、revoke、checkpoint、rollback、build_state_package、import_state_package、
#            export_transitions、ep_transition、interface_ep_transition、StateScheduler、purge。

# State effect (EN): None at import time.
# 状态影响（中）：导入时无副作用。

from hms_elite.ops.checkpoint import checkpoint, rollback
from hms_elite.ops.consolidation import StateScheduler
from hms_elite.ops.feedback import FeedbackProcessor, revoke
from hms_elite.ops.handoff import build_state_package, import_state_package
from hms_elite.ops.retention import purge
from hms_elite.ops.transitions import ep_transition, export_transitions, interface_ep_transition

__all__ = [
    "FeedbackProcessor",
    "StateScheduler",
    "build_state_package",
    "checkpoint",
    "ep_transition",
    "export_transitions",
    "import_state_package",
    "interface_ep_transition",
    "purge",
    "revoke",
    "rollback",
]
