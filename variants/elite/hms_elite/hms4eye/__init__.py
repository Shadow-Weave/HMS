"""hms4eye v0.6.1 compatibility layer: the engine seam plus the row-to-response field mapping.
hms4eye v0.6.1 兼容层：引擎接缝，以及行到响应的字段映射。
"""

# Purpose (EN): Sub-package root; exposes HMSEngineAdapter and the pure mapping helpers of this layer.
# 目的（中）：子包根模块；暴露 HMSEngineAdapter 与本层的纯映射辅助函数。

# Input (EN): Nothing at import time.
# 输入（中）：导入时不接收任何输入。

# Output (EN): HMSEngineAdapter, DEFAULT_STAGE_PREFIX, fact_type_of, split_reserved, to_memory_unit, to_recall_result.
# 输出（中）：HMSEngineAdapter、DEFAULT_STAGE_PREFIX、fact_type_of、split_reserved、to_memory_unit、to_recall_result。

# State effect (EN): None; importing has no side effects.
# 状态影响（中）：无；导入没有副作用。

from hms_elite.hms4eye.adapter import DEFAULT_STAGE_PREFIX, HMSEngineAdapter
from hms_elite.hms4eye.mapping import (
    DEFAULT_FACT_TYPE,
    FACT_TYPES,
    RESERVED_PREFIX,
    fact_type_of,
    split_reserved,
    to_memory_unit,
    to_recall_result,
)

__all__ = [
    "DEFAULT_FACT_TYPE",
    "DEFAULT_STAGE_PREFIX",
    "FACT_TYPES",
    "HMSEngineAdapter",
    "RESERVED_PREFIX",
    "fact_type_of",
    "split_reserved",
    "to_memory_unit",
    "to_recall_result",
]
