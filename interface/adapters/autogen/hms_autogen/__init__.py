"""HMS-AutoGen: Persistent memory tools for AutoGen agents.

Provides ``FunctionTool`` instances that give AutoGen agents long-term memory
via HMS's retain/recall/reflect APIs.

Basic usage::

    from hms_client import HMS
    from hms_autogen import create_hms_tools

    client = HMS(base_url="http://localhost:8888")
    tools = create_hms_tools(client=client, bank_id="user-123")

    # Use with an AutoGen AssistantAgent
    agent = AssistantAgent(name="assistant", model_client=model, tools=tools)
"""

from .config import (
    HMSAutoGenConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import create_hms_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSAutoGenConfig",
    "HMSError",
    "create_hms_tools",
]
