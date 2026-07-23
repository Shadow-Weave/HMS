"""Pure encoded-media guards shared by admission and durable checkpoints."""

from __future__ import annotations

import base64
import binascii
import re

_DATA_URL_RE = re.compile(r"data:[^,\s]*;base64,", re.IGNORECASE)
_LONG_BASE64_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{256,}={0,2}(?![A-Za-z0-9+/_=-])")
_BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")


def _is_wrapped_long_base64(value: str) -> bool:
    """Recognize MIME-style line-wrapped base64 without folding prose."""

    lines = value.splitlines()
    if len(lines) < 2:
        return False
    tokens = [line.strip(" \t") for line in lines]
    if any(not token or not _BASE64_LINE_RE.fullmatch(token) for token in tokens):
        return False
    # Encoders conventionally wrap base64 into long, four-byte-aligned lines.
    # Requiring that shape avoids classifying ordinary multi-line prose as a
    # binary payload merely because whitespace was removed.
    if any(len(token) < 32 or len(token) > 128 or len(token) % 4 for token in tokens[:-1]):
        return False
    compact = "".join(tokens)
    if len(compact) < 256 or "=" in compact.rstrip("="):
        return False
    try:
        padded = compact + "=" * (-len(compact) % 4)
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(decoded) >= 192


def contains_encoded_media_payload(value: str, *, reject_long_token: bool = True) -> bool:
    """Return whether one user-controlled string resembles encoded media."""

    if _DATA_URL_RE.search(value):
        return True
    if not reject_long_token:
        return False
    return bool(_LONG_BASE64_TOKEN_RE.search(value) or _is_wrapped_long_base64(value))


__all__ = ["contains_encoded_media_payload"]
