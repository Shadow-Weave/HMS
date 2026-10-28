"""Utility package: text lexicons/tokenizers and JSON helpers shared across planes.
工具包：跨平面共享的文本词表/分词器与 JSON 辅助函数。
"""

# Purpose (EN): Re-export the public names of utils.text and utils.json (factory/scoring stay in their modules).
# 目的（中）：再导出 utils.text 与 utils.json 的公开名称（factory/scoring 留在各自模块）。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The re-exported names.
# 输出（中）：再导出的名称。

# State effect (EN): None.
# 状态影响（中）：无。

from hms_elite.utils.json import canonical_json, extract_json, parse_llm_json, remove_code_blocks
from hms_elite.utils.text import (
    EntityCandidate,
    cjk_tokens_mode,
    cjk_tokens_scope,
    clean_object,
    content_tokens,
    entity_key,
    extract_dates,
    extract_entities,
    has_action_closure,
    identifier_tokens,
    is_first_person,
    is_identifier,
    md5_hex,
    normalize_entity,
    ordered_union,
    resolve_date,
    rewrite_first_person,
    sentences,
    sentiment,
    sha256_hex,
    stem,
    tokenize,
    topic_tags,
)

__all__ = [
    "EntityCandidate",
    "canonical_json",
    "cjk_tokens_mode",
    "cjk_tokens_scope",
    "clean_object",
    "content_tokens",
    "entity_key",
    "extract_dates",
    "extract_entities",
    "extract_json",
    "has_action_closure",
    "identifier_tokens",
    "is_first_person",
    "is_identifier",
    "md5_hex",
    "normalize_entity",
    "ordered_union",
    "parse_llm_json",
    "remove_code_blocks",
    "resolve_date",
    "rewrite_first_person",
    "sentences",
    "sentiment",
    "sha256_hex",
    "stem",
    "tokenize",
    "topic_tags",
]
