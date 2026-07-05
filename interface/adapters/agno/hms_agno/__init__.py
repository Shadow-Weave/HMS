"""HMS-Agno: Persistent memory tools for AI agents.

Provides a HMS-backed Toolkit for Agno agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from hms_agno import HMSTools, memory_instructions

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[HMSTools(
            bank_id="user-123",
            hms_api_url="http://localhost:8888",
        )],
        instructions=[memory_instructions(
            bank_id="user-123",
            hms_api_url="http://localhost:8888",
        )],
    )

    agent.print_response("What do you remember about my preferences?")
"""

from .config import (
    HMSAgnoConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import HMSTools, memory_instructions

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSAgnoConfig",
    "HMSError",
    "HMSTools",
    "memory_instructions",
]
