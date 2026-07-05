"""HMS-Strands: Persistent memory tools for AI agents.

Provides HMS-backed tool functions for Strands agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from strands import Agent
    from hms_strands import create_hms_tools

    tools = create_hms_tools(
        bank_id="user-123",
        hms_api_url="http://localhost:8888",
    )

    agent = Agent(tools=tools)
    agent("Remember that I prefer dark mode")
"""

from .config import (
    HMSStrandsConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import create_hms_tools, memory_instructions

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSStrandsConfig",
    "HMSError",
    "create_hms_tools",
    "memory_instructions",
]
