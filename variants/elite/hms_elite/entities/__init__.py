"""Entity extractor package: the provider contract, the rule default and the optional spaCy provider.
实体抽取器包：提供者契约、规则默认实现与可选的 spaCy 提供者。
"""

# Purpose (EN): Re-export the public names; the spacy module imports its SDK lazily so this package is offline-safe.
# 目的（中）：再导出公开名称；spacy 模块惰性导入 SDK，因此本包离线可安全导入。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The re-exported names.
# 输出（中）：再导出的名称。

# State effect (EN): None.
# 状态影响（中）：无。

from hms_elite.entities.base import BaseEntityExtractorConfig, EntityExtractorBase, Span, merge_candidates
from hms_elite.entities.rule import RuleEntityExtractor, RuleEntityExtractorConfig
from hms_elite.entities.spacy import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    SpacyEntityExtractor,
    SpacyEntityExtractorConfig,
    load_pipeline,
)

__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_MODEL",
    "BaseEntityExtractorConfig",
    "EntityExtractorBase",
    "RuleEntityExtractor",
    "RuleEntityExtractorConfig",
    "Span",
    "SpacyEntityExtractor",
    "SpacyEntityExtractorConfig",
    "load_pipeline",
    "merge_candidates",
]
