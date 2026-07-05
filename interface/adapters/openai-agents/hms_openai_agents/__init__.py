"""HMS-OpenAI-Agents: Persistent memory tools for OpenAI Agents SDK.

Provides ``FunctionTool`` instances that give OpenAI agents long-term memory
via HMS's retain/recall/reflect APIs.

Basic usage::

    from hms_client import HMS
    from hms_openai_agents import create_hms_tools

    client = HMS(base_url="http://localhost:8888")
    tools = create_hms_tools(client=client, bank_id="user-123")

    agent = Agent(name="assistant", tools=tools)
"""

from ._version import __version__
from .config import (
    HMSOpenAIAgentsConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import create_hms_tools, memory_instructions

__all__ = [
    "__version__",
    "configure",
    "get_config",
    "reset_config",
    "HMSOpenAIAgentsConfig",
    "HMSError",
    "create_hms_tools",
    "memory_instructions",
]
