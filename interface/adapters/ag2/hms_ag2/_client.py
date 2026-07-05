"""Shared HMS client resolution logic."""

from importlib import metadata
from typing import Any, Optional

from hms_client import HMS

from .config import get_config
from .errors import HMSError

try:
    _VERSION = metadata.version("hms-ag2")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hms-ag2/{_VERSION}"


def resolve_client(
    client: Optional[HMS],
    hms_api_url: Optional[str],
    api_key: Optional[str],
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
