"""Bounded local video decoding and deterministic scene/coverage sampling.

Raw video is deliberately never a provider input.  This module validates and
decodes an uploaded container with the optional PyAV dependency, selects a
bounded set of timestamped frames, and emits deterministic JPEG
``VisualEvidence`` objects for the existing image-capable provider path.

The selector is kept pure and independent from PyAV so its budget, coverage,
novelty, and tie-breaking contracts can be tested without a decoder.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, Sequence

from PIL import Image

from .errors import (
    MediaBudgetExceededError,
    MediaValidationError,
    MultimodalError,
    VideoDecodeError,
    VideoDecoderUnavailableError,
)
from .models import MediaAsset, VisualEvidence

try:  # The decoder is an explicit project extra: ``multimodal-video``.
    import av as _av
except (ImportError, OSError):  # pragma: no cover - environment dependent
    _av = None

SelectionReason = Literal["coverage", "scene", "both"]

_EXTENSION_FAMILIES = {
    ".avi": "avi",
    ".m4v": "isobmff",
    ".mkv": "matroska",
    ".mov": "isobmff",
    ".mp4": "isobmff",
    ".webm": "matroska",
}
_DECLARED_MIME_FAMILIES = {
    "video/avi": "avi",
    "video/mp4": "isobmff",
    "video/quicktime": "isobmff",
    "video/webm": "matroska",
    "video/x-matroska": "matroska",
    "video/x-msvideo": "avi",
}
_CONTAINER_FORMAT_TOKENS = {
    "avi": {"avi"},
    "isobmff": {"3g2", "3gp", "m4a", "mj2", "mov", "mp4"},
    "matroska": {"matroska", "webm"},
}

_PROCESS_TERMINATE_GRACE_SECONDS = 0.25
_PROCESS_KILL_GRACE_SECONDS = 1.0
_IPC_THREAD_JOIN_GRACE_SECONDS = 1.0
_SUPERVISOR_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class VideoProcessingConfig:
    """Static security, work, and sampling budgets for one video."""

    max_bytes: int = 200 * 1024 * 1024
    max_duration_seconds: float = 300.0
    max_pixels_per_frame: int = 8_847_360
    max_decoded_frames: int = 36_000
    max_decoded_work_pixels: int = 40_000_000_000
    max_probe_frames: int = 602
    max_candidate_encoded_bytes: int = 160 * 1024 * 1024
    max_frame_rate: float = 240.0
    decode_timeout_seconds: float = 120.0
    probe_interval_seconds: float = 1.0
    max_frames: int = 12
    coverage_ratio: float = 0.67
    min_scene_interval_seconds: float = 0.75
    scene_change_threshold: float = 0.06
    luma_weight: float = 0.7
    color_histogram_weight: float = 0.3
    novelty_weight: float = 0.8
    signature_size: int = 24
    histogram_bins: int = 16
    jpeg_max_dimension: int = 2048
    jpeg_quality: int = 90
    frame_count_tolerance_ratio: float = 0.10
    duration_tolerance_ratio: float = 0.10
    sampling_version: str = "scene-coverage-v1"

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "max_bytes",
            "max_pixels_per_frame",
            "max_decoded_frames",
            "max_decoded_work_pixels",
            "max_probe_frames",
            "max_candidate_encoded_bytes",
            "signature_size",
            "histogram_bins",
            "jpeg_max_dimension",
        )
        for name in positive_integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        positive_number_fields = (
            "max_duration_seconds",
            "max_frame_rate",
            "decode_timeout_seconds",
            "probe_interval_seconds",
        )
        for name in positive_number_fields:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_frames < 4:
            raise ValueError("max_frames must be at least 4 so one novelty slot remains")
        if self.max_probe_frames < self.max_frames:
            raise ValueError("max_probe_frames must be greater than or equal to max_frames")
        if not 0 < self.coverage_ratio < 1:
            raise ValueError("coverage_ratio must be strictly between zero and one")
        if self.min_scene_interval_seconds < 0 or not math.isfinite(self.min_scene_interval_seconds):
            raise ValueError("min_scene_interval_seconds must be finite and non-negative")
        if not 0 <= self.scene_change_threshold <= 1:
            raise ValueError("scene_change_threshold must be between zero and one")
        if self.luma_weight < 0 or self.color_histogram_weight < 0:
            raise ValueError("visual change weights must be non-negative")
        if self.luma_weight + self.color_histogram_weight <= 0:
            raise ValueError("at least one visual change weight must be positive")
        if not 0 <= self.novelty_weight <= 1:
            raise ValueError("novelty_weight must be between zero and one")
        if self.histogram_bins > 256 or 256 % self.histogram_bins:
            raise ValueError("histogram_bins must be a positive divisor of 256")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")
        if not 0 <= self.frame_count_tolerance_ratio <= 1:
            raise ValueError("frame_count_tolerance_ratio must be between zero and one")
        if not 0 <= self.duration_tolerance_ratio <= 1:
            raise ValueError("duration_tolerance_ratio must be between zero and one")
        if not self.sampling_version or len(self.sampling_version) > 128:
            raise ValueError("sampling_version must be non-empty and at most 128 characters")

    @property
    def coverage_slots(self) -> int:
        """The configured K formula, always leaving a novelty slot."""

        return min(self.max_frames - 1, max(3, math.floor(self.max_frames * self.coverage_ratio)))

    @property
    def sampling_config_fingerprint(self) -> str:
        """Stable identity for all budgets and algorithm parameters."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class VideoMagic:
    family: Literal["isobmff", "matroska", "avi"]
    detected_mime: str


@dataclass(frozen=True)
class VideoProbe:
    """Sanitized metadata cross-checked against decoded frames."""

    detected_mime: str
    container_format: str
    codec_name: str
    video_stream_index: int
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    duration_ms: int
    container_duration_ms: int | None
    stream_duration_ms: int | None
    decoded_duration_ms: int
    declared_frame_count: int | None
    decoded_frame_count: int
    average_frame_rate: float | None
    decoded_average_frame_rate: float | None
    rotation_degrees: int
    audio_presence: Literal["absent", "present"]
    decoded_work_pixels: int
    candidate_frame_count: int


@dataclass(frozen=True)
class VideoFrameCandidate:
    """One probe-interval candidate and its deterministic visual signature."""

    timestamp_ms: int
    frame_sha256: str
    width: int
    height: int
    luma_signature: tuple[int, ...]
    color_histogram: tuple[int, ...]
    signature_size: int
    histogram_pixels: int
    encoded_bytes: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class SamplingCandidateDiagnostic:
    timestamp_ms: int
    frame_sha256: str
    change_score: float
    is_scene_candidate: bool
    selected_reason: SelectionReason | None


@dataclass(frozen=True)
class SelectedFrameDiagnostic:
    timestamp_ms: int
    frame_sha256: str
    change_score: float
    reason: SelectionReason


@dataclass(frozen=True)
class VideoSamplingDiagnostics:
    sampling_version: str
    sampling_config_fingerprint: str
    frame_budget: int
    coverage_slots: int
    novelty_slots: int
    candidate_count: int
    scene_candidate_count: int
    candidates: tuple[SamplingCandidateDiagnostic, ...]
    selected: tuple[SelectedFrameDiagnostic, ...]


@dataclass(frozen=True)
class VideoSamplingResult:
    frames: tuple[VideoFrameCandidate, ...]
    diagnostics: VideoSamplingDiagnostics


@dataclass(frozen=True)
class VideoProcessingTimings:
    """Non-semantic wall timings for the three bounded local video stages."""

    decode_seconds: float = 0.0
    normalization_seconds: float = 0.0
    sample_seconds: float = 0.0


@dataclass(frozen=True)
class NormalizedVideo:
    asset: MediaAsset
    evidence: tuple[VisualEvidence, ...]
    probe: VideoProbe
    sampling: VideoSamplingDiagnostics
    # Timing is operational evidence, not part of deterministic media
    # identity, so repeated conversions remain semantically comparable.
    timings: VideoProcessingTimings = field(default_factory=VideoProcessingTimings, compare=False)


def video_decoder_available() -> bool:
    """Return whether the release-qualified H.264 decoder is available.

    This only inspects the local codec registry. It does not open user media
    or contact an external provider.
    """

    if _av is None:
        return False
    try:
        _av.Codec("h264", "r")
    except (AttributeError, ValueError, RuntimeError):
        return False
    return True


def video_decoder_identity() -> dict[str, str]:
    """Return decoder/library versions that can affect sampled frame bytes."""

    if _av is None:
        return {"pyav": "unavailable"}
    identity = {"pyav": str(getattr(_av, "__version__", "unknown"))}
    for library_name, version in sorted(getattr(_av, "library_versions", {}).items()):
        if isinstance(version, tuple):
            identity[library_name] = ".".join(str(part) for part in version)
        else:
            identity[library_name] = str(version)
    return identity


def require_video_decoder() -> None:
    """Fail before provider work when the explicit PyAV extra is absent."""

    if _av is None:
        raise VideoDecoderUnavailableError(
            "media.video_decoder_unavailable",
            "Video decoding requires the optional multimodal-video dependency",
        )


def detect_video_magic(data: bytes) -> VideoMagic:
    """Recognize a bounded set of video containers from bytes, not filename."""

    if len(data) >= 12 and data[4:8] == b"ftyp":
        box_size = int.from_bytes(data[:4], "big")
        if box_size not in {0, 1} and box_size < 12:
            raise MediaValidationError("media.invalid_video_magic", "Invalid ISO media container header")
        major_brand = data[8:12]
        mime = "video/quicktime" if major_brand == b"qt  " else "video/mp4"
        return VideoMagic(family="isobmff", detected_mime=mime)
    if data.startswith(b"\x1aE\xdf\xa3"):
        return VideoMagic(family="matroska", detected_mime="video/x-matroska")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return VideoMagic(family="avi", detected_mime="video/x-msvideo")
    raise MediaValidationError("media.unsupported_video", "Unsupported or unrecognized video container")


def video_extension_family_hint(filename: str) -> str | None:
    """Return the container-family constraint enforced by video validation."""

    return _EXTENSION_FAMILIES.get(Path(filename).suffix.lower())


def _validate_type_hints(*, magic: VideoMagic, declared_mime: str | None, filename: str) -> None:
    normalized_declared = (declared_mime or "").split(";", 1)[0].strip().lower()
    if normalized_declared and normalized_declared != "application/octet-stream":
        declared_family = _DECLARED_MIME_FAMILIES.get(normalized_declared)
        if declared_family != magic.family:
            raise MediaValidationError("media.mime_mismatch", "Declared content type does not match video bytes")

    extension_family = video_extension_family_hint(filename)
    if extension_family and extension_family != magic.family:
        raise MediaValidationError("media.extension_mismatch", "Filename extension does not match video bytes")


def normalize_rotation_degrees(value: object) -> int:
    """Normalize decoder display rotation to deterministic counterclockwise degrees."""

    if value is None or value == "":
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoDecodeError("media.video_rotation_invalid", "Video rotation metadata is invalid") from exc
    if not math.isfinite(numeric):
        raise VideoDecodeError("media.video_rotation_invalid", "Video rotation metadata is invalid")
    rounded = round(numeric)
    if not math.isclose(numeric, rounded, abs_tol=1e-6):
        raise VideoDecodeError("media.video_rotation_unsupported", "Fractional video rotation is unsupported")
    return int(rounded) % 360


def _apply_rotation(image: Image.Image, rotation_degrees: int) -> Image.Image:
    if rotation_degrees == 0:
        return image
    if rotation_degrees == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if rotation_degrees == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if rotation_degrees == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image.rotate(rotation_degrees, resample=Image.Resampling.BICUBIC, expand=True)


def _display_dimensions(width: int, height: int, rotation_degrees: int) -> tuple[int, int]:
    if rotation_degrees in {0, 180}:
        return width, height
    if rotation_degrees in {90, 270}:
        return height, width
    radians = math.radians(rotation_degrees)
    display_width = math.ceil(abs(width * math.cos(radians)) + abs(height * math.sin(radians)))
    display_height = math.ceil(abs(width * math.sin(radians)) + abs(height * math.cos(radians)))
    return display_width, display_height


def _color_histogram(image: Image.Image, bins: int) -> tuple[int, ...]:
    histogram = image.histogram()
    stride = 256 // bins
    return tuple(
        sum(histogram[channel * 256 + offset : channel * 256 + offset + stride])
        for channel in range(3)
        for offset in range(0, 256, stride)
    )


def build_video_frame_candidate(
    *,
    image: Image.Image,
    timestamp_ms: int,
    rotation_degrees: int,
    config: VideoProcessingConfig,
) -> VideoFrameCandidate:
    """Normalize a decoded frame and build its low-resolution visual signature."""

    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    oriented = _apply_rotation(image, rotation_degrees).convert("RGB")
    normalized = oriented.copy()
    normalized.thumbnail((config.jpeg_max_dimension, config.jpeg_max_dimension), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    normalized.save(
        output,
        format="JPEG",
        quality=config.jpeg_quality,
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    encoded = output.getvalue()
    frame_sha256 = hashlib.sha256(encoded).hexdigest()

    signature_image = oriented.resize((config.signature_size, config.signature_size), Image.Resampling.BILINEAR)
    luma = tuple(signature_image.convert("L").get_flattened_data())
    histogram = _color_histogram(signature_image, config.histogram_bins)
    return VideoFrameCandidate(
        timestamp_ms=timestamp_ms,
        frame_sha256=frame_sha256,
        width=normalized.width,
        height=normalized.height,
        luma_signature=luma,
        color_histogram=histogram,
        signature_size=config.signature_size,
        histogram_pixels=config.signature_size * config.signature_size,
        encoded_bytes=encoded,
    )


def _luma_structure_distance(left: VideoFrameCandidate, right: VideoFrameCandidate) -> float:
    if left.signature_size != right.signature_size or len(left.luma_signature) != len(right.luma_signature):
        raise ValueError("candidate luma signatures are incompatible")
    size = left.signature_size
    pixel_count = size * size
    if len(left.luma_signature) != pixel_count:
        raise ValueError("candidate luma signature has an invalid shape")

    pixel_distance = sum(abs(a - b) for a, b in zip(left.luma_signature, right.luma_signature, strict=True))
    pixel_distance /= 255 * pixel_count

    gradient_distance = 0
    gradient_count = 0
    for y in range(size):
        row = y * size
        for x in range(size - 1):
            index = row + x
            left_gradient = left.luma_signature[index + 1] - left.luma_signature[index]
            right_gradient = right.luma_signature[index + 1] - right.luma_signature[index]
            gradient_distance += abs(left_gradient - right_gradient)
            gradient_count += 1
    for y in range(size - 1):
        row = y * size
        next_row = row + size
        for x in range(size):
            left_gradient = left.luma_signature[next_row + x] - left.luma_signature[row + x]
            right_gradient = right.luma_signature[next_row + x] - right.luma_signature[row + x]
            gradient_distance += abs(left_gradient - right_gradient)
            gradient_count += 1
    normalized_gradient_distance = gradient_distance / (510 * gradient_count) if gradient_count else 0.0
    return 0.5 * pixel_distance + 0.5 * normalized_gradient_distance


def _histogram_distance(left: VideoFrameCandidate, right: VideoFrameCandidate) -> float:
    if len(left.color_histogram) != len(right.color_histogram):
        raise ValueError("candidate color histograms are incompatible")
    if left.histogram_pixels <= 0 or left.histogram_pixels != right.histogram_pixels:
        raise ValueError("candidate histogram pixel counts are incompatible")
    # Total-variation distance, averaged across the three RGB channels.
    total = sum(abs(a - b) for a, b in zip(left.color_histogram, right.color_histogram, strict=True))
    return total / (2 * 3 * left.histogram_pixels)


def _change_scores(candidates: Sequence[VideoFrameCandidate], config: VideoProcessingConfig) -> list[float]:
    weight_sum = config.luma_weight + config.color_histogram_weight
    scores = [0.0]
    for previous, current in zip(candidates, candidates[1:], strict=False):
        luma = _luma_structure_distance(previous, current)
        histogram = _histogram_distance(previous, current)
        score = (config.luma_weight * luma + config.color_histogram_weight * histogram) / weight_sum
        scores.append(round(score, 12))
    return scores


def _scene_candidate_indices(
    candidates: Sequence[VideoFrameCandidate], scores: Sequence[float], config: VideoProcessingConfig
) -> set[int]:
    local_peaks: list[int] = []
    for index in range(1, len(candidates)):
        score = scores[index]
        previous_score = scores[index - 1]
        next_score = scores[index + 1] if index + 1 < len(scores) else -1.0
        # Strict on the left and inclusive on the right selects the earliest
        # item on a flat peak and therefore makes plateau handling stable.
        if score >= config.scene_change_threshold and score > previous_score and score >= next_score:
            local_peaks.append(index)

    minimum_distance_ms = round(config.min_scene_interval_seconds * 1000)
    accepted: set[int] = set()
    for index in sorted(
        local_peaks,
        key=lambda item: (-scores[item], candidates[item].timestamp_ms, candidates[item].frame_sha256),
    ):
        timestamp = candidates[index].timestamp_ms
        if all(abs(timestamp - candidates[other].timestamp_ms) >= minimum_distance_ms for other in accepted):
            accepted.add(index)
    return accepted


def _coverage_indices(candidates: Sequence[VideoFrameCandidate], *, duration_ms: int, slots: int) -> list[int]:
    if not candidates or slots <= 0:
        return []

    selected: list[int] = []
    selected_set: set[int] = set()
    effective_duration = max(duration_ms, candidates[-1].timestamp_ms, 0)

    for stratum in range(slots):
        available = [index for index in range(len(candidates)) if index not in selected_set]
        if not available:
            break
        start = effective_duration * stratum / slots
        end = effective_duration * (stratum + 1) / slots
        in_stratum = [
            index
            for index in available
            if candidates[index].timestamp_ms >= start
            and (candidates[index].timestamp_ms < end or stratum == slots - 1)
        ]

        if stratum == 0 and in_stratum:
            chosen = min(in_stratum, key=lambda item: (candidates[item].timestamp_ms, candidates[item].frame_sha256))
        elif stratum == slots - 1 and in_stratum:
            chosen = min(
                in_stratum,
                key=lambda item: (-candidates[item].timestamp_ms, candidates[item].frame_sha256),
            )
        else:
            target = (start + end) / 2
            pool = in_stratum or available
            chosen = min(
                pool,
                key=lambda item: (
                    abs(candidates[item].timestamp_ms - target),
                    candidates[item].timestamp_ms,
                    candidates[item].frame_sha256,
                ),
            )
        selected.append(chosen)
        selected_set.add(chosen)
    return selected


def sample_video_candidates(
    candidates: Sequence[VideoFrameCandidate],
    *,
    duration_ms: int,
    config: VideoProcessingConfig,
) -> VideoSamplingResult:
    """Select scene novelty plus temporal coverage with stable tie-breaking."""

    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    ordered = list(candidates)
    if len(ordered) > config.max_probe_frames:
        raise MediaBudgetExceededError(
            "media.video_probe_frames_exceeded", "Video exceeds the configured probe-frame budget"
        )
    if any(current.timestamp_ms <= previous.timestamp_ms for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("candidate timestamps must be unique and strictly increasing")
    if not ordered:
        raise VideoDecodeError("media.video_no_frames", "Video did not contain a decodable visual frame")

    scores = _change_scores(ordered, config)
    scene_indices = _scene_candidate_indices(ordered, scores, config)
    coverage_indices = _coverage_indices(ordered, duration_ms=duration_ms, slots=config.coverage_slots)

    selected_reasons: dict[int, SelectionReason] = {}
    for index in coverage_indices:
        selected_reasons[index] = "both" if index in scene_indices else "coverage"

    target_count = min(config.max_frames, len(ordered))
    effective_duration = max(duration_ms, ordered[-1].timestamp_ms, 1)
    while len(selected_reasons) < target_count:
        remaining_scene = [index for index in scene_indices if index not in selected_reasons]
        pool = remaining_scene or [index for index in range(len(ordered)) if index not in selected_reasons]
        if not pool:
            break

        def novelty_key(index: int) -> tuple[float, int, str]:
            minimum_distance = min(
                abs(ordered[index].timestamp_ms - ordered[chosen].timestamp_ms) for chosen in selected_reasons
            )
            temporal_diversity = min(1.0, minimum_distance / effective_duration)
            combined = config.novelty_weight * scores[index] + (1 - config.novelty_weight) * temporal_diversity
            return (-round(combined, 12), ordered[index].timestamp_ms, ordered[index].frame_sha256)

        chosen = min(pool, key=novelty_key)
        selected_reasons[chosen] = "scene"

    selected_indices = sorted(selected_reasons, key=lambda index: ordered[index].timestamp_ms)
    selected_diagnostics = tuple(
        SelectedFrameDiagnostic(
            timestamp_ms=ordered[index].timestamp_ms,
            frame_sha256=ordered[index].frame_sha256,
            change_score=scores[index],
            reason=selected_reasons[index],
        )
        for index in selected_indices
    )
    candidate_diagnostics = tuple(
        SamplingCandidateDiagnostic(
            timestamp_ms=candidate.timestamp_ms,
            frame_sha256=candidate.frame_sha256,
            change_score=scores[index],
            is_scene_candidate=index in scene_indices,
            selected_reason=selected_reasons.get(index),
        )
        for index, candidate in enumerate(ordered)
    )
    diagnostics = VideoSamplingDiagnostics(
        sampling_version=config.sampling_version,
        sampling_config_fingerprint=config.sampling_config_fingerprint,
        frame_budget=config.max_frames,
        coverage_slots=config.coverage_slots,
        novelty_slots=config.max_frames - config.coverage_slots,
        candidate_count=len(ordered),
        scene_candidate_count=len(scene_indices),
        candidates=candidate_diagnostics,
        selected=selected_diagnostics,
    )
    return VideoSamplingResult(frames=tuple(ordered[index] for index in selected_indices), diagnostics=diagnostics)


def _milliseconds(value: int | None, time_base: object | None) -> int | None:
    if value is None or time_base is None:
        return None
    fraction = Fraction(value) * Fraction(time_base)
    if fraction < 0:
        return None
    numerator = fraction.numerator * 1000
    return (numerator + fraction.denominator // 2) // fraction.denominator


def _container_duration_ms(container: object) -> int | None:
    duration = getattr(container, "duration", None)
    if duration is None or duration < 0:
        return None
    # PyAV exposes container duration in AV_TIME_BASE (microsecond) units.
    return int((duration + 500) // 1000)


def _stream_duration_ms(stream: object) -> int | None:
    return _milliseconds(getattr(stream, "duration", None), getattr(stream, "time_base", None))


def _resolve_verified_duration_ms(
    *,
    decoded_duration_ms: int,
    stream_duration_ms: int | None,
    container_duration_ms: int | None,
    tolerance_ratio: float,
) -> int:
    """Cross-check every declared timeline before choosing a duration.

    Container metadata is attacker-controlled and must not become the sampling
    or provenance duration merely because stream metadata is absent. Each
    available declaration is independently checked against the decoded
    timeline before a conservative maximum is selected.
    """

    if decoded_duration_ms < 0:
        raise VideoDecodeError("media.video_duration_mismatch", "Decoded video duration is invalid")
    declared_durations = [value for value in (stream_duration_ms, container_duration_ms) if value is not None]
    for declared_duration_ms in declared_durations:
        if declared_duration_ms < 0:
            raise VideoDecodeError("media.video_duration_mismatch", "Declared video duration is invalid")
        tolerance = max(1_000, math.ceil(declared_duration_ms * tolerance_ratio))
        if abs(declared_duration_ms - decoded_duration_ms) > tolerance:
            raise VideoDecodeError(
                "media.video_duration_mismatch",
                "Declared and decoded video durations disagree",
            )
    return max(decoded_duration_ms, *declared_durations)


def _average_rate(stream: object) -> Fraction | None:
    rate = getattr(stream, "average_rate", None)
    if rate is None:
        return None
    try:
        fraction = Fraction(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return fraction if fraction > 0 else None


def _frame_relative_timestamp_ms(frame: object, *, first_time: Fraction | None, index: int, rate: Fraction | None):
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    raw_time = Fraction(pts) * Fraction(time_base) if pts is not None and time_base is not None else None
    if raw_time is None:
        raw_time = Fraction(index, 1) / rate if rate is not None else Fraction(index, 1)
    origin = raw_time if first_time is None else first_time
    relative = raw_time - origin
    numerator = relative.numerator * 1000
    milliseconds = (numerator + relative.denominator // 2) // relative.denominator
    return max(0, milliseconds), origin


def _format_tokens(container_format: str) -> set[str]:
    return {token.strip().lower() for token in container_format.split(",") if token.strip()}


def _validate_container_family(*, family: str, container_format: str) -> None:
    if not (_format_tokens(container_format) & _CONTAINER_FORMAT_TOKENS[family]):
        raise MediaValidationError(
            "media.video_container_mismatch", "Decoded container does not match the uploaded file signature"
        )


def _has_disposition(stream: object, name: str) -> bool:
    """Test a PyAV IntFlag without treating the flag constant as a value."""

    disposition = getattr(stream, "disposition", 0)
    flag = getattr(type(disposition), name, 0)
    return bool(disposition & flag)


def _decode_and_sample_video_locally(
    *,
    file_data: bytes,
    filename: str,
    declared_mime: str | None,
    config: VideoProcessingConfig,
    asset_id: str | None = None,
) -> NormalizedVideo:
    """Decode in the disposable worker process after repeating all validation."""

    if not file_data:
        raise MediaValidationError("media.empty", "Uploaded media is empty")
    if len(file_data) > config.max_bytes:
        raise MediaBudgetExceededError("media.video_bytes_exceeded", "Video exceeds the configured byte budget")
    magic = detect_video_magic(file_data)
    _validate_type_hints(magic=magic, declared_mime=declared_mime, filename=filename)
    require_video_decoder()
    assert _av is not None

    started_at = time.monotonic()
    candidates: list[VideoFrameCandidate] = []
    candidate_timestamps: set[int] = set()
    encoded_candidate_bytes = 0
    normalization_seconds = 0.0
    decoded_frames = 0
    decoded_work_pixels = 0
    decoded_duration_ms = 0
    first_raw_time: Fraction | None = None
    previous_frame = None
    previous_timestamp_ms: int | None = None
    last_frame = None
    last_timestamp_ms: int | None = None
    next_probe_timestamp_ms = 0
    probe_interval_ms = max(1, round(config.probe_interval_seconds * 1000))
    resolved_rotation: int | None = None
    coded_width = 0
    coded_height = 0

    def check_elapsed() -> None:
        if time.monotonic() - started_at > config.decode_timeout_seconds:
            raise MediaBudgetExceededError("media.video_decode_timeout", "Video decode exceeded the time budget")

    def add_candidate(frame: object, timestamp_ms: int) -> None:
        nonlocal encoded_candidate_bytes, normalization_seconds
        if timestamp_ms in candidate_timestamps:
            return
        if len(candidates) >= config.max_probe_frames:
            raise MediaBudgetExceededError(
                "media.video_probe_frames_exceeded", "Video exceeds the configured probe-frame budget"
            )
        normalization_started = time.monotonic()
        image = frame.to_image()
        candidate = build_video_frame_candidate(
            image=image,
            timestamp_ms=timestamp_ms,
            rotation_degrees=resolved_rotation or 0,
            config=config,
        )
        normalization_seconds += time.monotonic() - normalization_started
        encoded_candidate_bytes += len(candidate.encoded_bytes)
        if encoded_candidate_bytes > config.max_candidate_encoded_bytes:
            raise MediaBudgetExceededError(
                "media.video_candidate_bytes_exceeded", "Video candidates exceed the configured memory budget"
            )
        candidates.append(candidate)
        candidate_timestamps.add(timestamp_ms)

    try:
        source = io.BytesIO(file_data)
        with _av.open(source, mode="r", timeout=config.decode_timeout_seconds) as container:
            container_format = str(container.format.name or "")
            _validate_container_family(family=magic.family, container_format=container_format)
            video_streams = [
                stream
                for stream in container.streams
                if stream.type == "video" and not _has_disposition(stream, "attached_pic")
            ]
            if not video_streams:
                raise VideoDecodeError("media.video_stream_missing", "Container does not contain a video stream")
            stream = min(
                video_streams,
                key=lambda item: (not _has_disposition(item, "default"), item.index),
            )
            video_stream_index = int(stream.index)
            audio_presence: Literal["absent", "present"] = (
                "present" if any(item.type == "audio" for item in container.streams) else "absent"
            )
            codec_context = stream.codec_context
            codec_name = str(getattr(codec_context, "name", "") or "")
            if not codec_name:
                raise VideoDecodeError("media.video_codec_missing", "Video stream codec could not be identified")
            try:
                codec_context.thread_count = 1
            except (AttributeError, RuntimeError, ValueError):
                pass

            coded_width = int(getattr(codec_context, "width", 0) or 0)
            coded_height = int(getattr(codec_context, "height", 0) or 0)
            if coded_width < 0 or coded_height < 0:
                raise VideoDecodeError("media.video_dimensions_invalid", "Video dimensions are invalid")
            if coded_width and coded_height and coded_width * coded_height > config.max_pixels_per_frame:
                raise MediaBudgetExceededError(
                    "media.video_pixels_exceeded", "Video exceeds the configured pixels-per-frame budget"
                )

            container_duration_ms = _container_duration_ms(container)
            stream_duration_ms = _stream_duration_ms(stream)
            declared_duration_ms = stream_duration_ms or container_duration_ms
            if declared_duration_ms is not None and declared_duration_ms > config.max_duration_seconds * 1000:
                raise MediaBudgetExceededError(
                    "media.video_duration_exceeded", "Video exceeds the configured duration budget"
                )
            declared_frame_count = int(getattr(stream, "frames", 0) or 0) or None
            if declared_frame_count is not None and declared_frame_count > config.max_decoded_frames:
                raise MediaBudgetExceededError(
                    "media.video_frame_count_exceeded", "Video exceeds the configured decoded-frame budget"
                )
            rate = _average_rate(stream)
            average_frame_rate = float(rate) if rate is not None else None
            if average_frame_rate is not None and average_frame_rate > config.max_frame_rate:
                raise MediaBudgetExceededError(
                    "media.video_frame_rate_exceeded", "Video exceeds the configured frame-rate budget"
                )
            metadata_rotation = normalize_rotation_degrees(stream.metadata.get("rotate"))

            for frame in container.decode(stream):
                check_elapsed()
                decoded_frames += 1
                if decoded_frames > config.max_decoded_frames:
                    raise MediaBudgetExceededError(
                        "media.video_frame_count_exceeded", "Video exceeds the configured decoded-frame budget"
                    )
                frame_width = int(frame.width)
                frame_height = int(frame.height)
                if frame_width <= 0 or frame_height <= 0:
                    raise VideoDecodeError("media.video_dimensions_invalid", "Decoded video dimensions are invalid")
                frame_pixels = frame_width * frame_height
                if frame_pixels > config.max_pixels_per_frame:
                    raise MediaBudgetExceededError(
                        "media.video_pixels_exceeded", "Video exceeds the configured pixels-per-frame budget"
                    )
                decoded_work_pixels += frame_pixels
                if decoded_work_pixels > config.max_decoded_work_pixels:
                    raise MediaBudgetExceededError(
                        "media.video_decoded_work_exceeded", "Video exceeds the configured decoded-work budget"
                    )
                if coded_width == 0 or coded_height == 0:
                    coded_width, coded_height = frame_width, frame_height
                elif (frame_width, frame_height) != (coded_width, coded_height):
                    raise VideoDecodeError(
                        "media.video_resolution_changed", "Video changes resolution within the selected stream"
                    )

                frame_rotation = normalize_rotation_degrees(getattr(frame, "rotation", 0))
                effective_rotation = frame_rotation or metadata_rotation
                if resolved_rotation is None:
                    resolved_rotation = effective_rotation
                elif resolved_rotation != effective_rotation:
                    raise VideoDecodeError(
                        "media.video_rotation_changed", "Video changes display rotation within the selected stream"
                    )

                timestamp_ms, first_raw_time = _frame_relative_timestamp_ms(
                    frame,
                    first_time=first_raw_time,
                    index=decoded_frames - 1,
                    rate=rate,
                )
                if last_timestamp_ms is not None and timestamp_ms < last_timestamp_ms:
                    raise VideoDecodeError("media.video_timeline_invalid", "Decoded video timestamps are not monotonic")

                while next_probe_timestamp_ms <= timestamp_ms:
                    if previous_frame is not None and previous_timestamp_ms is not None:
                        previous_distance = abs(previous_timestamp_ms - next_probe_timestamp_ms)
                        current_distance = abs(timestamp_ms - next_probe_timestamp_ms)
                        if previous_distance <= current_distance:
                            add_candidate(previous_frame, previous_timestamp_ms)
                        else:
                            add_candidate(frame, timestamp_ms)
                    else:
                        add_candidate(frame, timestamp_ms)
                    next_probe_timestamp_ms += probe_interval_ms

                frame_duration_ms = _milliseconds(getattr(frame, "duration", None), getattr(frame, "time_base", None))
                decoded_duration_ms = max(decoded_duration_ms, timestamp_ms + (frame_duration_ms or 0))
                previous_frame = frame
                previous_timestamp_ms = timestamp_ms
                last_frame = frame
                last_timestamp_ms = timestamp_ms

            check_elapsed()
            if last_frame is None or last_timestamp_ms is None:
                raise VideoDecodeError("media.video_no_frames", "Video did not contain a decodable visual frame")
            add_candidate(last_frame, last_timestamp_ms)

            if declared_frame_count is not None:
                tolerance = max(2, math.ceil(declared_frame_count * config.frame_count_tolerance_ratio))
                if abs(declared_frame_count - decoded_frames) > tolerance:
                    raise VideoDecodeError(
                        "media.video_frame_count_mismatch", "Declared and decoded video frame counts disagree"
                    )

            decoded_duration_ms = max(decoded_duration_ms, last_timestamp_ms)
            duration_ms = _resolve_verified_duration_ms(
                decoded_duration_ms=decoded_duration_ms,
                stream_duration_ms=stream_duration_ms,
                container_duration_ms=container_duration_ms,
                tolerance_ratio=config.duration_tolerance_ratio,
            )
            if duration_ms > config.max_duration_seconds * 1000:
                raise MediaBudgetExceededError(
                    "media.video_duration_exceeded", "Video exceeds the configured duration budget"
                )
            decoded_average_frame_rate = (
                decoded_frames / (decoded_duration_ms / 1000) if decoded_duration_ms > 0 else None
            )
            if decoded_average_frame_rate is not None and decoded_average_frame_rate > config.max_frame_rate * 1.05:
                raise MediaBudgetExceededError(
                    "media.video_frame_rate_exceeded", "Video exceeds the configured frame-rate budget"
                )

    except MultimodalError:
        raise
    except (MemoryError, OverflowError) as exc:
        raise MediaBudgetExceededError(
            "media.video_decode_resources_exceeded", "Video decode exceeded a local resource budget"
        ) from exc
    except Exception as exc:
        raise VideoDecodeError("media.video_decode_failed", "Video could not be decoded safely") from exc

    decode_finished_at = time.monotonic()
    sampling_started_at = time.monotonic()
    sampling = sample_video_candidates(candidates, duration_ms=duration_ms, config=config)
    sample_seconds = time.monotonic() - sampling_started_at
    evidence = tuple(
        VisualEvidence(
            evidence_id=f"frame-{frame.timestamp_ms:012d}-{frame.frame_sha256[:12]}",
            timestamp_ms=frame.timestamp_ms,
            sha256=frame.frame_sha256,
            mime_type="image/jpeg",
            width=frame.width,
            height=frame.height,
            encoded_bytes=frame.encoded_bytes,
        )
        for frame in sampling.frames
    )
    raw_sha256 = hashlib.sha256(file_data).hexdigest()
    rotation = resolved_rotation or 0
    display_width, display_height = _display_dimensions(coded_width, coded_height, rotation)
    if display_width * display_height > config.max_pixels_per_frame:
        raise MediaBudgetExceededError(
            "media.video_display_pixels_exceeded", "Oriented video exceeds the configured pixels-per-frame budget"
        )
    asset = MediaAsset(
        asset_id=asset_id or f"sha256:{raw_sha256}",
        sha256=raw_sha256,
        media_kind="video",
        detected_mime=magic.detected_mime,
        original_filename=filename or "upload",
        byte_size=len(file_data),
        width=display_width,
        height=display_height,
        duration_ms=duration_ms,
        audio_presence=audio_presence,
        audio_processing="not_requested",
    )
    probe = VideoProbe(
        detected_mime=magic.detected_mime,
        container_format=container_format,
        codec_name=codec_name,
        video_stream_index=video_stream_index,
        coded_width=coded_width,
        coded_height=coded_height,
        display_width=display_width,
        display_height=display_height,
        duration_ms=duration_ms,
        container_duration_ms=container_duration_ms,
        stream_duration_ms=stream_duration_ms,
        decoded_duration_ms=decoded_duration_ms,
        declared_frame_count=declared_frame_count,
        decoded_frame_count=decoded_frames,
        average_frame_rate=average_frame_rate,
        decoded_average_frame_rate=decoded_average_frame_rate,
        rotation_degrees=rotation,
        audio_presence=audio_presence,
        decoded_work_pixels=decoded_work_pixels,
        candidate_frame_count=len(candidates),
    )
    return NormalizedVideo(
        asset=asset,
        evidence=evidence,
        probe=probe,
        sampling=sampling.diagnostics,
        timings=VideoProcessingTimings(
            decode_seconds=max(decode_finished_at - started_at - normalization_seconds, 0.0),
            normalization_seconds=normalization_seconds,
            sample_seconds=sample_seconds,
        ),
    )


_VideoWorkerTarget = Callable[[Connection, Connection], None]

_WORKER_ERROR_TYPES: dict[str, type[MultimodalError]] = {
    "MediaBudgetExceededError": MediaBudgetExceededError,
    "MediaValidationError": MediaValidationError,
    "MultimodalError": MultimodalError,
    "VideoDecodeError": VideoDecodeError,
    "VideoDecoderUnavailableError": VideoDecoderUnavailableError,
}
_WORKER_ERROR_MESSAGES = {
    "MediaBudgetExceededError": "Video exceeded a configured local resource budget",
    "MediaValidationError": "Video failed deterministic local validation",
    "MultimodalError": "Video processing failed in the isolated decoder",
    "VideoDecodeError": "Video could not be decoded safely",
    "VideoDecoderUnavailableError": "The isolated video decoder is unavailable",
}
_ERROR_CODE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


def _video_decode_timeout_error() -> MediaBudgetExceededError:
    return MediaBudgetExceededError("media.video_decode_timeout", "Video decode exceeded the time budget")


def _video_decode_process_error() -> VideoDecodeError:
    return VideoDecodeError("media.video_decode_failed", "Video could not be decoded safely")


def _send_worker_message(connection: Connection, message: object) -> bool:
    """Send one sanitized worker envelope, tolerating a parent that timed out."""

    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


def _close_connection(connection: Connection) -> None:
    """Close an IPC endpoint safely when a supervisor thread may race cleanup."""

    try:
        connection.close()
    except OSError:
        pass


def _isolated_video_decode_worker(request_connection: Connection, result_connection: Connection) -> None:
    """Receive bytes, decode locally, and emit only a bounded typed envelope.

    This target must stay at module scope so ``multiprocessing`` can import it
    with the ``spawn`` start method.  In particular, it never receives a path
    and never invokes a shell.  The parent owns the wall-clock deadline and can
    terminate this entire process even while FFmpeg is stuck in native code.
    """

    try:
        if not _send_worker_message(result_connection, ("ready",)):
            return
        request = request_connection.recv()
        if not isinstance(request, tuple) or len(request) != 4:
            raise VideoDecodeError("media.video_decode_failed", "Video could not be decoded safely")
        filename, declared_mime, config, asset_id = request
        if (
            not isinstance(filename, str)
            or (declared_mime is not None and not isinstance(declared_mime, str))
            or not isinstance(config, VideoProcessingConfig)
            or (asset_id is not None and not isinstance(asset_id, str))
        ):
            raise VideoDecodeError("media.video_decode_failed", "Video could not be decoded safely")
        file_data = request_connection.recv_bytes(maxlength=config.max_bytes)
        result = _decode_and_sample_video_locally(
            file_data=file_data,
            filename=filename,
            declared_mime=declared_mime,
            config=config,
            asset_id=asset_id,
        )
        message: object = ("ok", result)
    except MultimodalError as exc:
        message = (
            "error",
            type(exc).__name__,
            exc.code,
            exc.retryable,
            exc.logical_calls,
            exc.physical_attempts,
        )
    except BaseException:
        # Never cross the process boundary with a traceback, repr, filename,
        # media fragment, or decoder/library diagnostic.
        message = ("error", "VideoDecodeError", "media.video_decode_failed", False, 0, 0)
    finally:
        _close_connection(request_connection)

    _send_worker_message(result_connection, message)
    _close_connection(result_connection)


def _stop_video_worker(process: BaseProcess) -> bool:
    """Best-effort terminate, escalate to kill, and reap without an unbounded wait."""

    if not process.is_alive():
        process.join(timeout=0)
        return True
    process.terminate()
    process.join(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_PROCESS_KILL_GRACE_SECONDS)
    return not process.is_alive()


def _worker_error_from_message(message: object) -> MultimodalError:
    """Reconstruct only explicitly allowed error metadata from the worker."""

    if not isinstance(message, tuple) or len(message) != 6 or message[0] != "error":
        return _video_decode_process_error()
    _, error_name, code, retryable, logical_calls, physical_attempts = message
    if (
        error_name not in _WORKER_ERROR_TYPES
        or not isinstance(code, str)
        or not code.startswith("media.")
        or len(code) > 128
        or not code
        or any(character not in _ERROR_CODE_CHARACTERS for character in code)
        or not isinstance(retryable, bool)
        or type(logical_calls) is not int
        or type(physical_attempts) is not int
        or logical_calls < 0
        or physical_attempts < 0
    ):
        return _video_decode_process_error()
    error_type = _WORKER_ERROR_TYPES[error_name]
    return error_type(
        code,
        _WORKER_ERROR_MESSAGES[error_name],
        retryable=retryable,
        logical_calls=logical_calls,
        physical_attempts=physical_attempts,
    )


def _run_video_decode_in_subprocess(
    *,
    file_data: bytes,
    filename: str,
    declared_mime: str | None,
    config: VideoProcessingConfig,
    asset_id: str | None,
    worker_target: _VideoWorkerTarget = _isolated_video_decode_worker,
) -> NormalizedVideo:
    """Run one decoder in a fresh spawn process under a hard parent deadline.

    The request uses a one-way byte pipe rather than a ``Process`` argument or
    queue: ``send_bytes`` avoids building another full-size pickle of an upload,
    while a readiness handshake avoids copying it before the clean interpreter
    is ready.  Daemon I/O threads let the supervisor retain the deadline if a
    pipe stalls while moving bounded upload or JPEG evidence bytes.
    """

    context = multiprocessing.get_context("spawn")
    request_receive, request_send = context.Pipe(duplex=False)
    result_receive, result_send = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(request_receive, result_send),
        name="hms-video-decoder",
        daemon=True,
    )
    deadline = time.monotonic() + config.decode_timeout_seconds
    process_started = False
    sender: threading.Thread | None = None
    receiver: threading.Thread | None = None
    worker_ready = threading.Event()
    request_done = threading.Event()
    result_done = threading.Event()
    request_failed: list[bool] = []
    result_failed: list[bool] = []
    received_messages: list[object] = []

    def send_request() -> None:
        try:
            request_send.send((filename, declared_mime, config, asset_id))
            request_send.send_bytes(file_data)
        except BaseException:
            request_failed.append(True)
        finally:
            _close_connection(request_send)
            request_done.set()

    def receive_result() -> None:
        try:
            while True:
                message = result_receive.recv()
                if message == ("ready",):
                    worker_ready.set()
                    continue
                received_messages.append(message)
                return
        except BaseException:
            result_failed.append(True)
        finally:
            _close_connection(result_receive)
            result_done.set()

    try:
        try:
            process.start()
            process_started = True
        except Exception:
            raise VideoDecoderUnavailableError(
                "media.video_decoder_isolation_unavailable",
                "The isolated video decoder could not be started",
            ) from None
        finally:
            # These endpoint copies belong only to the spawned child.  Closing
            # them in the parent is required for EOF and broken-pipe cleanup.
            _close_connection(request_receive)
            _close_connection(result_send)

        receiver = threading.Thread(target=receive_result, name="hms-video-output", daemon=True)
        receiver.start()

        # Do not start copying raw upload bytes until the clean interpreter has
        # imported the fixed worker target and acknowledged the private pipe.
        while not worker_ready.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _video_decode_timeout_error()
            worker_ready.wait(timeout=min(remaining, _SUPERVISOR_POLL_SECONDS))
            if result_done.is_set() and not worker_ready.is_set():
                raise _video_decode_process_error()

        sender = threading.Thread(target=send_request, name="hms-video-input", daemon=True)
        sender.start()

        while not result_done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _video_decode_timeout_error()
            result_done.wait(timeout=min(remaining, _SUPERVISOR_POLL_SECONDS))
            if request_done.is_set() and request_failed:
                raise _video_decode_process_error()
            if process.exitcode is not None and not result_done.is_set():
                # Allow the receiver one scheduler turn to consume a final
                # envelope already present in the pipe before classifying EOF.
                result_done.wait(timeout=min(remaining, _SUPERVISOR_POLL_SECONDS))

        if result_failed or not received_messages:
            raise _video_decode_process_error()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _video_decode_timeout_error()
        process.join(timeout=remaining)
        if process.is_alive():
            raise _video_decode_timeout_error()
        if process.exitcode != 0:
            raise _video_decode_process_error()

        message = received_messages[0]
        if isinstance(message, tuple) and len(message) == 2 and message[0] == "ok":
            result = message[1]
            if isinstance(result, NormalizedVideo):
                return result
            raise _video_decode_process_error()
        raise _worker_error_from_message(message)
    finally:
        if process_started and process.is_alive():
            _stop_video_worker(process)
        # Reaping the child closes its pipe ends and normally wakes both I/O
        # threads.  Join before closing a descriptor from another thread so a
        # recycled POSIX file descriptor can never receive a stale write.
        if sender is not None:
            sender.join(timeout=_IPC_THREAD_JOIN_GRACE_SECONDS)
        if receiver is not None:
            receiver.join(timeout=_IPC_THREAD_JOIN_GRACE_SECONDS)
        _close_connection(request_send)
        _close_connection(result_receive)
        if process_started and not process.is_alive():
            process.join(timeout=0)
            process.close()


def decode_and_sample_video(
    *,
    file_data: bytes,
    filename: str,
    declared_mime: str | None,
    config: VideoProcessingConfig,
    asset_id: str | None = None,
) -> NormalizedVideo:
    """Validate and decode one video inside a disposable hard-timeout worker."""

    supervisor_started_at = time.monotonic()
    if not file_data:
        raise MediaValidationError("media.empty", "Uploaded media is empty")
    if len(file_data) > config.max_bytes:
        raise MediaBudgetExceededError("media.video_bytes_exceeded", "Video exceeds the configured byte budget")
    magic = detect_video_magic(file_data)
    _validate_type_hints(magic=magic, declared_mime=declared_mime, filename=filename)
    require_video_decoder()
    result = _run_video_decode_in_subprocess(
        file_data=file_data,
        filename=filename,
        declared_mime=declared_mime,
        config=config,
        asset_id=asset_id,
    )
    # Attribute spawn/IPC/supervision overhead to decode so the three emitted
    # stage durations add up to the caller-visible preprocessing wall time.
    elapsed = time.monotonic() - supervisor_started_at
    normalization_seconds = result.timings.normalization_seconds
    sample_seconds = result.timings.sample_seconds
    return replace(
        result,
        timings=VideoProcessingTimings(
            decode_seconds=max(elapsed - normalization_seconds - sample_seconds, 0.0),
            normalization_seconds=normalization_seconds,
            sample_seconds=sample_seconds,
        ),
    )


__all__ = [
    "NormalizedVideo",
    "SamplingCandidateDiagnostic",
    "SelectedFrameDiagnostic",
    "VideoFrameCandidate",
    "VideoMagic",
    "VideoProbe",
    "VideoProcessingTimings",
    "VideoProcessingConfig",
    "VideoSamplingDiagnostics",
    "VideoSamplingResult",
    "build_video_frame_candidate",
    "decode_and_sample_video",
    "detect_video_magic",
    "normalize_rotation_degrees",
    "require_video_decoder",
    "sample_video_candidates",
    "video_decoder_available",
]
