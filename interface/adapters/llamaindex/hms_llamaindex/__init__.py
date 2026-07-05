"""HMS memory integration for LlamaIndex agents.

Provides two complementary patterns:

- **Tools** (``HMSToolSpec``, ``create_hms_tools``):
  Agent-driven memory via LlamaIndex's ``BaseToolSpec``.
  The agent decides when to retain/recall/reflect.

- **Memory** (``HMSMemory``):
  Automatic memory via LlamaIndex's ``BaseMemory`` interface.
  Messages are stored on ``put()`` and recalled on ``get()``.

Usage::

    from hms_llamaindex import HMSToolSpec, create_hms_tools
    from hms_llamaindex import HMSMemory
"""

from .config import (
    HMSLlamaIndexConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .memory import HMSMemory
from .tools import HMSToolSpec, create_hms_tools

__version__ = "0.1.2"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSLlamaIndexConfig",
    "HMSError",
    "HMSToolSpec",
    "create_hms_tools",
    "HMSMemory",
]
