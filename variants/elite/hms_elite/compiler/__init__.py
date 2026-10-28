"""State compiler: axis candidates first, graph on demand, deterministic slots, budget, Work State Card and PC-IR.
状态编译器：先轴候选，按需展开图，确定性槽位、预算、Work State Card 与 PC-IR。
"""

# Purpose (EN): Re-export the public surface of the compiler plane (slots, StateCompiler, renderer, PC-IR lowering,
#               the hybrid read context builders of HMS.context()).
# 目的（中）：再导出编译器平面的公开接口（槽位、StateCompiler、渲染器、PC-IR 降低、HMS.context() 的混合读上下文构造器）。

# Input (EN): Nothing at import time.
# 输入（中）：导入时不接收任何输入。

# Output (EN): The names listed in __all__.
# 输出（中）：__all__ 中列出的名称。

# State effect (EN): None; importing has no side effects and the whole plane is read-only.
# 状态影响（中）：无；导入没有副作用，整个平面只读。

from hms_elite.compiler.compile import StateCompiler, fit_budget
from hms_elite.compiler.context import (
    NONE_TEXT,
    ROWS_HEADER,
    dedup_rows,
    format_stamp,
    hybrid_text,
    normalize_text,
    render_rows,
    row_times,
    rows_text,
    sort_rows_by_time,
)
from hms_elite.compiler.pc_ir import ProofObligationError, lower, obligations
from hms_elite.compiler.render import render_card
from hms_elite.compiler.slots import (
    SLOT_RULES,
    assign_slot,
    audit_reason,
    derive_next_action,
    evidence_items,
    invariant_items,
    item_from_atom,
    kappa_items,
    normalized_texts,
)

__all__ = [
    "NONE_TEXT",
    "ROWS_HEADER",
    "SLOT_RULES",
    "ProofObligationError",
    "StateCompiler",
    "assign_slot",
    "audit_reason",
    "dedup_rows",
    "derive_next_action",
    "evidence_items",
    "fit_budget",
    "format_stamp",
    "hybrid_text",
    "invariant_items",
    "item_from_atom",
    "kappa_items",
    "lower",
    "normalize_text",
    "normalized_texts",
    "obligations",
    "render_card",
    "render_rows",
    "row_times",
    "rows_text",
    "sort_rows_by_time",
]
