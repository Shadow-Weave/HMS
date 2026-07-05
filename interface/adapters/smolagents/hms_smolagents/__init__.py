"""HMS-SmolAgents: Persistent memory tools for AI agents.

Provides HMS-backed Tool subclasses for SmolAgents agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from smolagents import CodeAgent, HfApiModel
    from hms_smolagents import create_hms_tools

    tools = create_hms_tools(
        bank_id="user-123",
        hms_api_url="http://localhost:8888",
    )

    agent = CodeAgent(
        tools=tools,
        model=HfApiModel(),
    )

    agent.run("Remember that I prefer dark mode")
"""

from .config import (
    HMSSmolAgentsConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import (
    HMSRecallTool,
    HMSReflectTool,
    HMSRetainTool,
    create_hms_tools,
    memory_instructions,
)

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSSmolAgentsConfig",
    "HMSError",
    "HMSRetainTool",
    "HMSRecallTool",
    "HMSReflectTool",
    "create_hms_tools",
    "memory_instructions",
]
