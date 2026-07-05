"""
Configuration for hms-agentcore.

Supports global configuration (via configure()) and per-adapter overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_HMS_API_URL = "https://api.hms.local"
HMS_API_URL_ENV = "HMS_API_URL"
HMS_API_KEY_ENV = "HMS_API_KEY"
HMS_API_TOKEN_ENV = "HMS_API_TOKEN"


@dataclass
class HMSAgentCoreConfig:
    """Global configuration for the AgentCore–HMS adapter."""

    hms_api_url: str = DEFAULT_HMS_API_URL
    """HMS server URL."""

    api_key: str | None = None
    """API key for HMS Cloud. Reads HMS_API_KEY or HMS_API_TOKEN if not set."""

    recall_budget: str = "mid"
    """Recall search depth: 'low', 'mid', or 'high'."""

    recall_max_tokens: int = 1500
    """Maximum tokens in the recalled memory block."""

    retain_async: bool = True
    """If True, retain is fire-and-forget (does not block the turn response)."""

    timeout: float = 15.0
    """HTTP timeout in seconds for HMS API calls."""

    tags: list[str] = field(default_factory=list)
    """Default tags added to all retained memories."""

    verbose: bool = False
    """Log memory operations."""


_global_config: HMSAgentCoreConfig | None = None


def configure(
    hms_api_url: str | None = None,
    api_key: str | None = None,
    recall_budget: str = "mid",
    recall_max_tokens: int = 1500,
    retain_async: bool = True,
    timeout: float = 15.0,
    tags: list[str] | None = None,
    verbose: bool = False,
) -> HMSAgentCoreConfig:
    """Set the global HMS configuration for AgentCore adapters.

    Call once at application startup, before creating any adapters.

    Example:
        from hms_agentcore import configure

        configure(
            hms_api_url="https://api.hms.local",
            api_key=os.environ["HMS_API_KEY"],
        )
    """
    global _global_config

    resolved_url = hms_api_url or os.environ.get(HMS_API_URL_ENV) or DEFAULT_HMS_API_URL
    resolved_key = api_key or os.environ.get(HMS_API_KEY_ENV) or os.environ.get(HMS_API_TOKEN_ENV)

    _global_config = HMSAgentCoreConfig(
        hms_api_url=resolved_url,
        api_key=resolved_key,
        recall_budget=recall_budget,
        recall_max_tokens=recall_max_tokens,
        retain_async=retain_async,
        timeout=timeout,
        tags=tags or [],
        verbose=verbose,
    )
    return _global_config


def get_config() -> HMSAgentCoreConfig | None:
    """Return the current global config, or None if not yet configured."""
    return _global_config


def reset_config() -> None:
    """Reset the global config to None. Useful in tests."""
    global _global_config
    _global_config = None
