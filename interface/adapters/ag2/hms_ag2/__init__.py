"""HMS-AG2: Persistent memory tools for AG2 agents.

Provides HMS-backed tool functions that give AG2 agents long-term
memory across conversations via retain/recall/reflect operations.

Basic usage::

    from hms_ag2 import register_hms_tools

    register_hms_tools(
        assistant, user_proxy,
        bank_id="user-123",
        hms_api_url="http://localhost:8888",
    )

Manual registration::

    from hms_ag2 import create_hms_tools

    tools = create_hms_tools(
        bank_id="user-123",
        hms_api_url="http://localhost:8888",
    )
    for tool_fn in tools:
        assistant.register_for_llm(description=tool_fn.__doc__)(tool_fn)
        user_proxy.register_for_execution()(tool_fn)
"""

from .config import (
    HMSAG2Config,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .tools import create_hms_tools, register_hms_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSAG2Config",
    "HMSError",
    "create_hms_tools",
    "register_hms_tools",
]
