"""HMS-Pydantic AI: Persistent memory tools for AI agents.

Provides HMS-backed tools and instructions for Pydantic AI agents,
giving them long-term memory across runs.

Basic usage::

    from hms_client import HMS
    from hms_pydantic_ai import create_hms_tools, memory_instructions
    from pydantic_ai import Agent

    client = HMS(base_url="http://localhost:8888")

    agent = Agent(
        "openai:gpt-4o",
        tools=create_hms_tools(client=client, bank_id="user-123"),
        instructions=[memory_instructions(client=client, bank_id="user-123")],
    )

    result = await agent.run("What do you remember about my preferences?")
"""

from .config import (
    HMSPydanticAIConfig,
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
    "HMSPydanticAIConfig",
    "HMSError",
    "create_hms_tools",
    "memory_instructions",
]
