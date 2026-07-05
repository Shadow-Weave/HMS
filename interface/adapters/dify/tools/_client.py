"""Shared helper for building a HMS client from Dify tool credentials."""

from __future__ import annotations

from typing import Any

from hms_client import HMS


def build_client(credentials: dict[str, Any]) -> HMS:
    """Construct a HMS client from the provider credentials dict."""
    api_url = (credentials.get("api_url") or "").rstrip("/")
    api_key = credentials.get("api_key") or None

    kwargs: dict[str, Any] = {"base_url": api_url, "timeout": 30.0}
    if api_key:
        kwargs["api_key"] = api_key
    return HMS(**kwargs)


def parse_tags(value: str | None) -> list[str] | None:
    """Parse a comma-separated tag string into a list. Returns None for empty input."""
    if not value:
        return None
    tags = [t.strip() for t in value.split(",") if t.strip()]
    return tags or None
