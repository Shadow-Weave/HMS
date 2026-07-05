"""Global configuration for HMS-Pipecat integration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HMS_API_URL = "https://api.hms.local"
HMS_API_KEY_ENV = "HMS_API_KEY"


@dataclass
class HMSPipecatConfig:
    """Connection and default settings for the Pipecat integration.

    Attributes:
        hms_api_url: URL of the HMS API server.
        api_key: API key for HMS authentication.
        recall_budget: Default recall budget level (low/mid/high).
        recall_max_tokens: Default maximum tokens for recall results.
    """

    hms_api_url: str = DEFAULT_HMS_API_URL
    api_key: str | None = None
    recall_budget: str = "mid"
    recall_max_tokens: int = 4096


_global_config: HMSPipecatConfig | None = None


def configure(
    hms_api_url: str | None = None,
    api_key: str | None = None,
    recall_budget: str = "mid",
    recall_max_tokens: int = 4096,
) -> HMSPipecatConfig:
    """Configure HMS connection and default settings.

    Args:
        hms_api_url: HMS API URL (default: production).
        api_key: API key. Falls back to HMS_API_KEY env var.
        recall_budget: Default recall budget (low/mid/high).
        recall_max_tokens: Default max tokens for recall.

    Returns:
        The configured HMSPipecatConfig.
    """
    global _global_config

    resolved_url = hms_api_url or DEFAULT_HMS_API_URL
    resolved_key = api_key or os.environ.get(HMS_API_KEY_ENV)

    _global_config = HMSPipecatConfig(
        hms_api_url=resolved_url,
        api_key=resolved_key,
        recall_budget=recall_budget,
        recall_max_tokens=recall_max_tokens,
    )

    return _global_config


def get_config() -> HMSPipecatConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
