"""Constant-time media magic hints for bounded upload admission.

This module intentionally has no Pillow/PyAV imports.  It is safe to use on
the legacy text path and only inspects the first container signature bytes.
Full decoder validation remains in the multimodal parser.
"""

from __future__ import annotations

from typing import Literal

MediaKindHint = Literal["image", "video"]

_IMAGE_EXTENSIONS = frozenset({"gif", "jpeg", "jpg", "png", "webp"})
_VIDEO_EXTENSIONS = frozenset({"avi", "m4v", "mkv", "mov", "mp4", "webm"})


def classify_media_hint(*, filename: str, content_type: str | None) -> MediaKindHint | None:
    """Classify untrusted MIME/extension hints without treating them as facts."""

    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if normalized_type.startswith("image/") or suffix in _IMAGE_EXTENSIONS:
        return "image"
    if normalized_type.startswith("video/") or suffix in _VIDEO_EXTENSIONS:
        return "video"
    return None


def classify_media_magic_prefix(prefix: bytes) -> MediaKindHint | None:
    """Classify a supported media family when its magic header is complete."""

    if prefix.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")):
        return "image"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "image"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return "video"
    if prefix.startswith(b"\x1aE\xdf\xa3"):
        return "video"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"AVI ":
        return "video"
    return None


def initial_multimodal_probe_bytes(*, image_limit: int, video_limit: int, chunk_size: int) -> int:
    """Bound the first read tightly enough to discover magic before buffering."""

    if min(image_limit, video_limit) <= 0 or chunk_size <= 0:
        raise ValueError("multimodal admission limits must be positive")
    # All supported signatures are <= 12 bytes.  Probe the fixed header first;
    # once the actual family is known, the caller switches to that family's
    # strict byte limit.  Hints must never select a larger limit before this
    # probe, because a spoofed ``video/*`` hint could otherwise widen an image
    # upload from the image budget to the video budget.
    return min(chunk_size, 12)
