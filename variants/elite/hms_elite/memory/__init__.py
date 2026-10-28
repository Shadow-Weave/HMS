"""Memory package: the Mem0 contract, input helpers, the SQLite schema and the StateStore seam.
记忆包：Mem0 契约、输入辅助、SQLite 模式与 StateStore 接缝。
"""

# Purpose (EN): Re-export MemoryBase, the memory.utils helpers, schema constants and StateStore; never import main.
# 目的（中）：再导出 MemoryBase、memory.utils 辅助函数、模式常量与 StateStore；绝不导入 main。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The re-exported names.
# 输出（中）：再导出的名称。

# State effect (EN): None.
# 状态影响（中）：无。

from hms_elite.memory.base import MemoryBase
from hms_elite.memory.schema import MIGRATIONS, SCHEMA_VERSION, TABLES, apply_migrations, detect_capabilities, drop_all
from hms_elite.memory.storage import LIVE_SQL, EntityKey, StateStore, TableTrace
from hms_elite.memory.utils import (
    _UNSET,
    build_scope_and_metadata,
    coerce_messages,
    estimate_tokens,
    mask_ids,
    normalize_expiration_date,
    parse_messages,
    reject_top_level_entity_params,
    strip_reserved_keys,
    validate_and_trim_entity_id,
    validate_and_trim_query,
    validate_search_params,
)

__all__ = [
    "LIVE_SQL",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "TABLES",
    "EntityKey",
    "MemoryBase",
    "StateStore",
    "TableTrace",
    "apply_migrations",
    "detect_capabilities",
    "drop_all",
    "_UNSET",
    "build_scope_and_metadata",
    "coerce_messages",
    "estimate_tokens",
    "mask_ids",
    "normalize_expiration_date",
    "parse_messages",
    "reject_top_level_entity_params",
    "strip_reserved_keys",
    "validate_and_trim_entity_id",
    "validate_and_trim_query",
    "validate_search_params",
]
