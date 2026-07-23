"""Local image validation and normalization tests (no network or database)."""

import io

import pytest
from PIL import Image

from hms_api.engine.multimodal import (
    AnimatedImageNotSupportedError,
    ImageNormalizationConfig,
    MediaBudgetExceededError,
    MediaValidationError,
    normalize_image,
)


def _image_bytes(*, format: str, mode: str = "RGB", size: tuple[int, int] = (32, 16)) -> bytes:
    image = Image.new(mode, size, (10, 20, 30, 128) if "A" in mode else (10, 20, 30))
    output = io.BytesIO()
    image.save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize(
    ("format", "filename", "declared"),
    [
        ("PNG", "screen.png", "image/png"),
        ("JPEG", "screen.jpg", "image/jpeg"),
        ("WEBP", "screen.webp", "image/webp"),
        ("GIF", "screen.gif", "image/gif"),
    ],
)
def test_supported_still_images_normalize_deterministically(format, filename, declared):
    payload = _image_bytes(format=format)
    config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=10_000, max_dimension=64)

    first = normalize_image(file_data=payload, filename=filename, declared_mime=declared, config=config)
    second = normalize_image(file_data=payload, filename=filename, declared_mime=declared, config=config)

    assert first.asset.sha256 == second.asset.sha256
    assert first.evidence.sha256 == second.evidence.sha256
    assert first.evidence.encoded_bytes == second.evidence.encoded_bytes
    assert first.evidence.mime_type in {"image/jpeg", "image/png"}
    assert first.evidence.width == 32
    assert first.evidence.height == 16


def test_transparent_image_is_preserved_as_png():
    payload = _image_bytes(format="PNG", mode="RGBA")
    result = normalize_image(
        file_data=payload,
        filename="transparent.png",
        declared_mime="image/png",
        config=ImageNormalizationConfig(),
    )
    assert result.evidence.mime_type == "image/png"


def test_jpeg_exif_orientation_is_applied_and_removed_deterministically():
    image = Image.new("RGB", (40, 20), "white")
    for x in range(20):
        for y in range(20):
            image.putpixel((x, y), (220, 20, 20))
    exif = Image.Exif()
    exif[274] = 6  # Rotate the stored 40x20 pixels 90 degrees clockwise.
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, exif=exif)
    payload = output.getvalue()
    config = ImageNormalizationConfig(max_bytes=1_000_000, max_pixels=10_000, max_dimension=64)

    first = normalize_image(
        file_data=payload,
        filename="rotated.jpg",
        declared_mime="image/jpeg",
        config=config,
    )
    second = normalize_image(
        file_data=payload,
        filename="rotated.jpg",
        declared_mime="image/jpeg",
        config=config,
    )

    assert (first.asset.width, first.asset.height) == (20, 40)
    assert (first.evidence.width, first.evidence.height) == (20, 40)
    assert first.evidence.encoded_bytes == second.evidence.encoded_bytes
    with Image.open(io.BytesIO(first.evidence.encoded_bytes)) as normalized:
        assert normalized.getexif().get(274) is None


def test_animated_gif_is_not_silently_truncated():
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")
    output = io.BytesIO()
    first.save(output, format="GIF", save_all=True, append_images=[second], duration=50, loop=0)

    with pytest.raises(AnimatedImageNotSupportedError) as exc_info:
        normalize_image(
            file_data=output.getvalue(),
            filename="animated.gif",
            declared_mime="image/gif",
            config=ImageNormalizationConfig(),
        )
    assert exc_info.value.code == "media.animated_image_unsupported"


def test_mime_and_extension_spoofing_are_rejected():
    payload = _image_bytes(format="PNG")
    with pytest.raises(MediaValidationError) as mime_error:
        normalize_image(
            file_data=payload,
            filename="screen.png",
            declared_mime="image/jpeg",
            config=ImageNormalizationConfig(),
        )
    assert mime_error.value.code == "media.mime_mismatch"

    with pytest.raises(MediaValidationError) as extension_error:
        normalize_image(
            file_data=payload,
            filename="screen.jpg",
            declared_mime="application/octet-stream",
            config=ImageNormalizationConfig(),
        )
    assert extension_error.value.code == "media.extension_mismatch"


def test_corrupt_and_over_budget_images_fail_before_provider():
    with pytest.raises(MediaValidationError) as corrupt:
        normalize_image(
            file_data=b"not an image",
            filename="screen.png",
            declared_mime="image/png",
            config=ImageNormalizationConfig(),
        )
    assert corrupt.value.code == "media.unsupported_image"

    payload = _image_bytes(format="PNG", size=(100, 100))
    with pytest.raises(MediaBudgetExceededError) as pixels:
        normalize_image(
            file_data=payload,
            filename="large.png",
            declared_mime="image/png",
            config=ImageNormalizationConfig(max_pixels=9_999),
        )
    assert pixels.value.code == "media.image_pixels_exceeded"

    with pytest.raises(MediaBudgetExceededError) as byte_budget:
        normalize_image(
            file_data=payload,
            filename="large.png",
            declared_mime="image/png",
            config=ImageNormalizationConfig(max_bytes=len(payload) - 1),
        )
    assert byte_budget.value.code == "media.image_bytes_exceeded"
