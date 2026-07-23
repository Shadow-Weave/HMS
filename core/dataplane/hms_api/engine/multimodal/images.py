"""Deterministic, budgeted image validation and normalization."""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError, features
from PIL import __version__ as _pillow_version

from .errors import AnimatedImageNotSupportedError, MediaBudgetExceededError, MediaValidationError
from .models import MediaAsset, VisualEvidence

_EXTENSION_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_normalizer_identity() -> dict[str, str]:
    """Versions that can change normalized bytes and therefore cache output."""

    identity = {"pillow": _pillow_version}
    for feature_name in ("jpg", "jpg_2000", "webp", "zlib"):
        version = features.version(feature_name)
        if version:
            identity[f"codec_{feature_name}"] = version
    return identity


@dataclass(frozen=True)
class ImageNormalizationConfig:
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_dimension: int = 2048
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        if self.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")


@dataclass(frozen=True)
class NormalizedImage:
    asset: MediaAsset
    evidence: VisualEvidence


def detect_image_mime(data: bytes) -> str:
    """Detect the supported image type from magic bytes, never extension."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise MediaValidationError("media.unsupported_image", "Unsupported or unrecognized image format")


def image_extension_mime_hint(filename: str) -> str | None:
    """Return the exact extension constraint enforced by image validation."""

    return _EXTENSION_MIMES.get(Path(filename).suffix.lower())


def _validate_type_hints(*, detected_mime: str, declared_mime: str | None, filename: str) -> None:
    normalized_declared = (declared_mime or "").split(";", 1)[0].strip().lower()
    if (
        normalized_declared
        and normalized_declared != "application/octet-stream"
        and normalized_declared != detected_mime
    ):
        raise MediaValidationError("media.mime_mismatch", "Declared content type does not match decoded media")

    extension_mime = image_extension_mime_hint(filename)
    if extension_mime and extension_mime != detected_mime:
        raise MediaValidationError("media.extension_mismatch", "Filename extension does not match decoded media")


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    return image.mode == "P" and "transparency" in image.info


def normalize_image(
    *,
    file_data: bytes,
    filename: str,
    declared_mime: str | None,
    config: ImageNormalizationConfig,
    asset_id: str | None = None,
) -> NormalizedImage:
    """Validate and normalize one supported still image.

    The output has no EXIF/profile/application metadata and is encoded with
    deterministic parameters.  Animated GIFs are rejected rather than silently
    described from a single frame.
    """

    if not file_data:
        raise MediaValidationError("media.empty", "Uploaded media is empty")
    if len(file_data) > config.max_bytes:
        raise MediaBudgetExceededError("media.image_bytes_exceeded", "Image exceeds the configured byte budget")

    detected_mime = detect_image_mime(file_data)
    _validate_type_hints(detected_mime=detected_mime, declared_mime=declared_mime, filename=filename)
    raw_sha256 = hashlib.sha256(file_data).hexdigest()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(file_data)) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0:
                    raise MediaValidationError("media.invalid_dimensions", "Image dimensions must be positive")
                if width * height > config.max_pixels:
                    raise MediaBudgetExceededError(
                        "media.image_pixels_exceeded", "Image exceeds the configured decoded pixel budget"
                    )
                if detected_mime == "image/gif" and getattr(opened, "n_frames", 1) != 1:
                    raise AnimatedImageNotSupportedError(
                        "media.animated_image_unsupported", "Animated images require an explicit video pipeline"
                    )
                opened.seek(0)
                image = ImageOps.exif_transpose(opened).copy()
    except (AnimatedImageNotSupportedError, MediaBudgetExceededError, MediaValidationError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaBudgetExceededError(
            "media.image_decompression_bomb", "Image exceeds the safe decoded pixel budget"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaValidationError("media.corrupt_image", "Image could not be decoded safely") from exc

    image.thumbnail((config.max_dimension, config.max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    if _has_alpha(image):
        normalized = image.convert("RGBA")
        output_mime = "image/png"
        normalized.save(output, format="PNG", compress_level=9, optimize=False)
    else:
        normalized = image.convert("RGB")
        output_mime = "image/jpeg"
        normalized.save(
            output,
            format="JPEG",
            quality=config.jpeg_quality,
            optimize=False,
            progressive=False,
            subsampling=0,
        )

    normalized_bytes = output.getvalue()
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
    width, height = normalized.size
    effective_asset_id = asset_id or f"sha256:{raw_sha256}"
    asset = MediaAsset(
        asset_id=effective_asset_id,
        sha256=raw_sha256,
        media_kind="image",
        detected_mime=detected_mime,
        original_filename=filename or "upload",
        byte_size=len(file_data),
        width=width,
        height=height,
        duration_ms=None,
        audio_presence="absent",
        audio_processing="not_requested",
    )
    evidence = VisualEvidence(
        evidence_id=f"image-000-{normalized_sha256[:12]}",
        timestamp_ms=None,
        sha256=normalized_sha256,
        mime_type=output_mime,
        width=width,
        height=height,
        encoded_bytes=normalized_bytes,
    )
    return NormalizedImage(asset=asset, evidence=evidence)
