"""HMS Elite: a Mem0-compatible memory API over an experience-to-state dual-plane store.
HMS Elite：基于经验到状态双平面存储的 Mem0 兼容记忆 API。
"""

# Purpose (EN): Package root; exposes HMS, AsyncHMS, HMSConfig, the reranker classes, the §2 tail entry points
#               (AgentRuntime, StateDaemon, StateAPI, StateKind, Usage) and the package version (§17.4).
# 目的（中）：包根模块；暴露 HMS、AsyncHMS、HMSConfig、重排器类、§2 尾部入口（AgentRuntime、StateDaemon、StateAPI、
#            StateKind、Usage）与包版本号（§17.4）。

# Input (EN): Nothing at import time.
# 输入（中）：导入时不接收任何输入。

# Output (EN): HMS, AsyncHMS, HMSConfig, RerankerBase, OverlapReranker, LLMReranker, AgentRuntime, StateDaemon,
#              StateAPI, StateKind, Usage, __version__.
# 输出（中）：HMS、AsyncHMS、HMSConfig、RerankerBase、OverlapReranker、LLMReranker、AgentRuntime、StateDaemon、
#            StateAPI、StateKind、Usage、__version__。

# State effect (EN): None; importing has no side effects.
# 状态影响（中）：无；导入没有副作用。

from importlib import metadata as _metadata

from hms_elite.configs.base import HMSConfig
from hms_elite.daemon import StateDaemon
from hms_elite.memory.async_main import AsyncHMS
from hms_elite.memory.main import HMS
from hms_elite.reranker.base import RerankerBase
from hms_elite.reranker.llm import LLMReranker
from hms_elite.reranker.overlap import OverlapReranker
from hms_elite.runtime import AgentRuntime
from hms_elite.state.api import StateAPI
from hms_elite.state.kinds import StateKind
from hms_elite.telemetry import Usage

try:
    __version__ = _metadata.version("hms-elite")
except _metadata.PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"

__all__ = [
    "HMS",
    "AgentRuntime",
    "AsyncHMS",
    "HMSConfig",
    "LLMReranker",
    "OverlapReranker",
    "RerankerBase",
    "StateAPI",
    "StateDaemon",
    "StateKind",
    "Usage",
    "__version__",
]
