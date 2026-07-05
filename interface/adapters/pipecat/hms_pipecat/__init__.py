"""HMS-Pipecat: Persistent memory for voice AI pipelines.

Provides a HMS-backed FrameProcessor for Pipecat pipelines,
giving them long-term memory via automatic recall and retain.

Basic usage::

    from pipecat.pipeline.pipeline import Pipeline
    from hms_pipecat import HMSMemoryService

    memory = HMSMemoryService(
        bank_id="user-123",
        hms_api_url="http://localhost:8888",
    )

    pipeline = Pipeline([
        transport.input(),
        stt_service,
        user_aggregator,
        memory,            # recall before LLM, retain after each turn
        llm_service,
        assistant_aggregator,
        tts_service,
        transport.output(),
    ])
"""

from .config import (
    HMSPipecatConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HMSPipecatError
from .memory import HMSMemoryService

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HMSPipecatConfig",
    "HMSPipecatError",
    "HMSMemoryService",
]
