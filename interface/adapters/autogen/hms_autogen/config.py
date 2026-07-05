"""Global configuration for HMS-AutoGen integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_HMS_API_URL = "https://api.hms.local"
HMS_API_KEY_ENV = "HMS_API_KEY"

DEFAULT_BUDGET: Literal["low", "mid", "high"] = "mid"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RECALL_TAGS_MATCH: Literal["any", "all", "any_strict", "all_strict"] = "any"

Budget = Literal["low", "mid", "high"]
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


@dataclass
class HMSAutoGenConfig:
    """Connection and default settings for the AutoGen integration.

    Attributes:
        hms_api_url: URL of the HMS API server.
        api_key: API key for HMS authentication.
        budget: Default recall budget level (low/mid/high).
        max_tokens: Default maximum tokens for recall results.
        tags: Default tags applied when storing memories.
        recall_tags: Default tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
    """

    hms_api_url: str = DEFAULT_HMS_API_URL
    api_key: str | None = None
    budget: Budget = DEFAULT_BUDGET
    max_tokens: int = DEFAULT_MAX_TOKENS
    tags: list[str] | None = None
    recall_tags: list[str] | None = None
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH


_global_config: HMSAutoGenConfig | None = None


def configure(
    hms_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget = DEFAULT_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH,
) -> HMSAutoGenConfig:
    """Configure HMS connection and default settings.

    Args:
        hms_api_url: HMS API URL (default: production).
        api_key: API key. Falls back to HMS_API_KEY env var.
        budget: Default recall budget (low/mid/high).
        max_tokens: Default max tokens for recall.
        tags: Default tags for retain operations.
        recall_tags: Default tags to filter recall/search.
        recall_tags_match: Tag matching mode.

    Returns:
        The configured HMSAutoGenConfig.
    """
    global _global_config

    resolved_url = hms_api_url or DEFAULT_HMS_API_URL
    resolved_key = api_key or os.environ.get(HMS_API_KEY_ENV)

    _global_config = HMSAutoGenConfig(
        hms_api_url=resolved_url,
        api_key=resolved_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
    )

    return _global_config


def get_config() -> HMSAutoGenConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
