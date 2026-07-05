"""HMS-CrewAI: Persistent memory for AI agent crews.

Provides a HMS-backed Storage implementation for CrewAI's
ExternalMemory system, giving your crews long-term memory across runs.

Basic usage::

    from hms_crewai import configure, HMSStorage
    from crewai.memory.external.external_memory import ExternalMemory
    from crewai import Crew

    configure(hms_api_url="http://localhost:8888")

    crew = Crew(
        agents=[...],
        tasks=[...],
        external_memory=ExternalMemory(
            storage=HMSStorage(bank_id="my-crew")
        ),
    )

Per-agent banks::

    storage = HMSStorage(
        bank_id="crew-shared",
        per_agent_banks=True,
    )
"""

from .config import (
    HMSCrewAIConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSError
from .storage import HMSStorage
from .tools import HMSReflectTool

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSCrewAIConfig",
    "HMSStorage",
    "HMSReflectTool",
    "HMSError",
]
