"""LLM package: the provider contract, the offline rule provider, the scripted test provider and the rule tables.
LLM 包：提供者契约、离线规则提供者、脚本化测试提供者与规则表。
"""

# Purpose (EN): Re-export the offline-safe public names (openai/anthropic stay behind their import guards).
# 目的（中）：再导出离线可用的公开名称（openai/anthropic 留在各自的导入守卫之后）。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The re-exported names.
# 输出（中）：再导出的名称。

# State effect (EN): None.
# 状态影响（中）：无。

from hms_elite.llms.base import ROOT_KEY, BaseLlmConfig, LLMBase
from hms_elite.llms.rule import (
    EXCLUSIVE,
    PREDICATE_FAMILIES,
    RuleLLM,
    RuleLlmConfig,
    coarsen_rule,
    decide_revision,
    family,
    propose_edges,
)
from hms_elite.llms.scripted import ScriptedLLM, ScriptedLlmConfig

__all__ = [
    "EXCLUSIVE",
    "PREDICATE_FAMILIES",
    "ROOT_KEY",
    "BaseLlmConfig",
    "LLMBase",
    "RuleLLM",
    "RuleLlmConfig",
    "ScriptedLLM",
    "ScriptedLlmConfig",
    "coarsen_rule",
    "decide_revision",
    "family",
    "propose_edges",
]
