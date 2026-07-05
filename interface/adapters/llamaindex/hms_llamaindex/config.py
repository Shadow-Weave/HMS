"""Global configuration for HMS-LlamaIndex integration."""

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_HMS_API_URL = "https://api.hms.local"
HMS_API_KEY_ENV = "HMS_API_KEY"


@dataclass
class HMSLlamaIndexConfig:
    """Connection and default settings for the LlamaIndex integration.

    Attributes:
        hms_api_url: URL of the HMS API server.
        api_key: API key for HMS authentication.
        budget: Default recall budget level (low/mid/high).
        max_tokens: Default maximum tokens for recall results.
        tags: Default tags applied when storing memories.
        recall_tags: Default tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        context: Source label for retain operations (default: "llamaindex").
        mission: Bank mission for fact extraction context.
        verbose: Enable verbose logging.
    """

    hms_api_url: str = DEFAULT_HMS_API_URL
    api_key: Optional[str] = None
    budget: str = "mid"
    max_tokens: int = 4096
    tags: Optional[list[str]] = None
    recall_tags: Optional[list[str]] = None
    recall_tags_match: str = "any"
    context: str = "llamaindex"
    mission: Optional[str] = None
    verbose: bool = False


_global_config: Optional[HMSLlamaIndexConfig] = None


def configure(
    hms_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: Optional[list[str]] = None,
    recall_tags: Optional[list[str]] = None,
    recall_tags_match: str = "any",
    context: str = "llamaindex",
    mission: Optional[str] = None,
    verbose: bool = False,
) -> HMSLlamaIndexConfig:
    """Configure HMS connection and default settings.

    Args:
        hms_api_url: HMS API URL (default: production).
        api_key: API key. Falls back to HMS_API_KEY env var.
        budget: Default recall budget (low/mid/high).
        max_tokens: Default max tokens for recall.
        tags: Default tags for retain operations.
        recall_tags: Default tags to filter recall/search.
        recall_tags_match: Tag matching mode.
        context: Source label for retain operations.
        mission: Bank mission for fact extraction context.
        verbose: Enable verbose logging.

    Returns:
        The configured HMSLlamaIndexConfig.
    """
    global _global_config

    resolved_url = hms_api_url or DEFAULT_HMS_API_URL
    resolved_key = api_key or os.environ.get(HMS_API_KEY_ENV)

    _global_config = HMSLlamaIndexConfig(
        hms_api_url=resolved_url,
        api_key=resolved_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        context=context,
        mission=mission,
        verbose=verbose,
    )

    return _global_config


def get_config() -> Optional[HMSLlamaIndexConfig]:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
