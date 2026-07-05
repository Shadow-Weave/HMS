"""Shared HMS client resolution logic."""

from __future__ import annotations

from typing import Any

from hms_client import HMS

from ._version import __version__
from .config import get_config
from .errors import HMSError

_USER_AGENT = f"hms-openai-agents/{__version__}"


def resolve_client(
    client: HMS | None,
    hms_api_url: str | None,
    api_key: str | None,
) -> HMS:
    """Resolve a HMS client from explicit args or global config."""
    if client is not None:
        return client

    config = get_config()
    url = hms_api_url or (config.hms_api_url if config else None)
    key = api_key or (config.api_key if config else None)

    if url is None:
        raise HMSError(
            "No HMS API URL configured. Pass client= or hms_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return HMS(**kwargs)
