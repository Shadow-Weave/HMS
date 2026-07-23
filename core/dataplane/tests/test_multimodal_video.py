from __future__ import annotations

import hashlib
import io
import multiprocessing
import os
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from hms_api.engine.multimodal import (
    MediaBudgetExceededError,
    MediaValidationError,
    VideoDecodeError,
    VideoDecoderUnavailableError,
    VideoProcessingConfig,
    build_video_frame_candidate,
    decode_and_sample_video,
    detect_video_magic,
    normalize_rotation_degrees,
    sample_video_candidates,
    video_decoder_available,
)


def _stalling_spawn_video_worker(request_connection, result_connection) -> None:
    """Act like native decoder code that never reaches a Python checkpoint."""

    if os.name == "posix":
        import signal

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    result_connection.send(("ready",))
    filename, declared_mime, config, marker_path = request_connection.recv()
    del filename, declared_mime, config
    request_connection.recv_bytes()
    Path(marker_path).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(60)


def _screen_frame(*, transient: bool = False, size: tuple[int, int] = (160, 96)) -> Image.Image:
    image = Image.new("RGB", size, "#171a21")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, size[0] - 5, size[1] - 5), outline="#6c7480", width=2)
    draw.text((12, 14), "def stable_state():", fill="#d8dee9")
    draw.text((18, 30), "return memory", fill="#88c0d0")
    if transient:
        draw.rectangle((8, 52, size[0] - 8, size[1] - 8), fill="#8b1e2d")
        draw.text((14, 62), "TEST FAILED", fill="white")
    else:
        draw.rectangle((8, 52, size[0] - 8, size[1] - 8), fill="#243044")
        draw.text((14, 62), "terminal idle", fill="#a3be8c")
    return image


def _pure_candidates(config: VideoProcessingConfig):
    return tuple(
        build_video_frame_candidate(
            image=_screen_frame(transient=timestamp_ms == 1500),
            timestamp_ms=timestamp_ms,
            rotation_degrees=0,
            config=config,
        )
        for timestamp_ms in range(0, 6001, 500)
    )


def _make_mp4(
    *,
    frame_count: int = 30,
    rate: int = 5,
    size: tuple[int, int] = (160, 96),
    transient_indices: frozenset[int] = frozenset({7}),
    include_audio: bool = False,
) -> bytes:
    av = pytest.importorskip("av", reason="install the multimodal-video extra to run required decoder tests")
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=rate)
        video.width, video.height = size
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18", "preset": "ultrafast", "tune": "zerolatency"}
        audio = None
        audio_frame_count = 0
        audio_index = 0
        if include_audio:
            audio = container.add_stream("aac", rate=48_000)
            audio.layout = "mono"
            duration_seconds = frame_count / rate
            audio_frame_count = int(duration_seconds * 48_000 / 1024)
        for index in range(frame_count):
            frame = av.VideoFrame.from_image(_screen_frame(transient=index in transient_indices, size=size))
            for packet in video.encode(frame):
                container.mux(packet)
            audio_target = min(audio_frame_count, int((index + 1) / rate * 48_000 / 1024))
            while audio is not None and audio_index < audio_target:
                frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
                frame.sample_rate = 48_000
                frame.pts = audio_index * 1024
                frame.time_base = Fraction(1, 48_000)
                frame.planes[0].update(bytes(frame.planes[0].buffer_size))
                for packet in audio.encode(frame):
                    container.mux(packet)
                audio_index += 1
        for packet in video.encode():
            container.mux(packet)

        if audio is not None:
            while audio_index < audio_frame_count:
                frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
                frame.sample_rate = 48_000
                frame.pts = audio_index * 1024
                frame.time_base = Fraction(1, 48_000)
                frame.planes[0].update(bytes(frame.planes[0].buffer_size))
                for packet in audio.encode(frame):
                    container.mux(packet)
                audio_index += 1
            for packet in audio.encode():
                container.mux(packet)
    return output.getvalue()


def _config(**overrides) -> VideoProcessingConfig:
    defaults = {
        "max_duration_seconds": 20,
        "max_pixels_per_frame": 1_000_000,
        "max_decoded_frames": 1_000,
        "max_decoded_work_pixels": 1_000_000_000,
        "max_probe_frames": 100,
        "probe_interval_seconds": 0.5,
        "max_frames": 5,
        "coverage_ratio": 0.7,
        "min_scene_interval_seconds": 0.75,
        "scene_change_threshold": 0.03,
    }
    defaults.update(overrides)
    return VideoProcessingConfig(**defaults)


def test_video_config_enforces_coverage_formula_and_novelty_slot():
    config = VideoProcessingConfig(max_frames=10, coverage_ratio=0.8)
    assert config.coverage_slots == 8
    assert config.max_frames - config.coverage_slots >= 1
    assert len(config.sampling_config_fingerprint) == 64

    with pytest.raises(ValueError, match="at least 4"):
        VideoProcessingConfig(max_frames=3)
    with pytest.raises(ValueError, match="strictly between"):
        VideoProcessingConfig(coverage_ratio=1)


def test_container_and_stream_durations_are_each_cross_checked_against_decoded_timeline():
    import hms_api.engine.multimodal.video as video_module

    assert (
        video_module._resolve_verified_duration_ms(
            decoded_duration_ms=6_000,
            stream_duration_ms=6_000,
            container_duration_ms=6_200,
            tolerance_ratio=0.1,
        )
        == 6_200
    )

    for stream_duration, container_duration in ((None, 60_000), (6_000, 60_000), (60_000, 6_000)):
        with pytest.raises(VideoDecodeError) as exc_info:
            video_module._resolve_verified_duration_ms(
                decoded_duration_ms=6_000,
                stream_duration_ms=stream_duration,
                container_duration_ms=container_duration,
                tolerance_ratio=0.1,
            )
        assert exc_info.value.code == "media.video_duration_mismatch"


def test_video_magic_and_type_are_derived_from_bytes():
    mp4_header = b"\x00\x00\x00\x18ftypisom" + bytes(12)
    assert detect_video_magic(mp4_header).detected_mime == "video/mp4"
    assert detect_video_magic(b"\x1aE\xdf\xa3" + bytes(20)).family == "matroska"
    assert detect_video_magic(b"RIFF" + bytes(4) + b"AVI " + bytes(12)).family == "avi"
    with pytest.raises(MediaValidationError, match="Unsupported"):
        detect_video_magic(b"not a video")


def test_scene_coverage_sampler_is_deterministic_and_captures_short_change():
    config = _config()
    candidates = _pure_candidates(config)

    first = sample_video_candidates(candidates, duration_ms=6000, config=config)
    second = sample_video_candidates(candidates, duration_ms=6000, config=config)

    assert first == second
    timestamps = [frame.timestamp_ms for frame in first.frames]
    assert len(timestamps) == config.max_frames
    assert timestamps == sorted(set(timestamps))
    assert timestamps[0] == 0
    assert any(2500 <= value <= 3500 for value in timestamps)
    assert timestamps[-1] == 6000
    assert 1500 in timestamps
    transient = next(item for item in first.diagnostics.selected if item.timestamp_ms == 1500)
    assert transient.reason in {"scene", "both"}
    assert first.diagnostics.coverage_slots == 3
    assert first.diagnostics.novelty_slots == 2
    assert first.diagnostics.scene_candidate_count >= 1


def test_short_video_never_duplicates_frames_to_fill_budget():
    config = _config(max_frames=8, max_probe_frames=8)
    candidates = tuple(
        build_video_frame_candidate(
            image=_screen_frame(transient=index == 1),
            timestamp_ms=index * 200,
            rotation_degrees=0,
            config=config,
        )
        for index in range(2)
    )
    result = sample_video_candidates(candidates, duration_ms=400, config=config)
    assert len(result.frames) == 2
    assert [frame.timestamp_ms for frame in result.frames] == [0, 200]


def test_sampler_rejects_duplicate_or_unbounded_candidates():
    config = _config(max_frames=4, max_probe_frames=4)
    candidate = build_video_frame_candidate(image=_screen_frame(), timestamp_ms=0, rotation_degrees=0, config=config)
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_video_candidates((candidate, candidate), duration_ms=1000, config=config)
    with pytest.raises(MediaBudgetExceededError) as exc_info:
        sample_video_candidates((candidate,) * 5, duration_ms=1000, config=config)
    assert exc_info.value.code == "media.video_probe_frames_exceeded"


def test_frame_jpeg_normalization_applies_rotation_and_has_stable_hash():
    config = _config()
    image = _screen_frame(size=(160, 96))
    first = build_video_frame_candidate(image=image, timestamp_ms=1250, rotation_degrees=90, config=config)
    second = build_video_frame_candidate(image=image, timestamp_ms=1250, rotation_degrees=90, config=config)
    assert (first.width, first.height) == (96, 160)
    assert first.frame_sha256 == second.frame_sha256
    assert first.encoded_bytes == second.encoded_bytes
    assert first.encoded_bytes.startswith(b"\xff\xd8\xff")
    assert normalize_rotation_degrees(-90) == 270
    with pytest.raises(VideoDecodeError):
        normalize_rotation_degrees(22.5)


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_mp4_h264_decode_probe_sampling_and_evidence_are_deterministic():
    video_bytes = _make_mp4()
    config = _config()
    first = decode_and_sample_video(
        file_data=video_bytes,
        filename="coding-session.mp4",
        declared_mime="video/mp4",
        config=config,
    )
    second = decode_and_sample_video(
        file_data=video_bytes,
        filename="coding-session.mp4",
        declared_mime="video/mp4",
        config=config,
    )

    assert first.asset.sha256 == hashlib.sha256(video_bytes).hexdigest()
    assert first.asset.media_kind == "video"
    assert first.asset.audio_presence == "absent"
    assert first.asset.audio_processing == "not_requested"
    assert first.probe.codec_name == "h264"
    assert first.probe.container_format.split(",")[0] == "mov"
    assert first.probe.video_stream_index == 0
    assert first.probe.decoded_frame_count == 30
    assert first.probe.declared_frame_count == 30
    assert first.probe.decoded_work_pixels == 30 * 160 * 96
    assert first.probe.candidate_frame_count <= config.max_probe_frames
    assert first.timings.decode_seconds > 0
    assert first.timings.normalization_seconds > 0
    assert first.timings.sample_seconds > 0

    timeline = [(item.timestamp_ms, item.sha256) for item in first.evidence]
    assert timeline == [(item.timestamp_ms, item.sha256) for item in second.evidence]
    timestamps = [timestamp for timestamp, _ in timeline]
    assert len(timestamps) <= config.max_frames
    assert timestamps == sorted(set(timestamps))
    assert timestamps[0] == 0
    assert any(2500 <= value <= 3500 for value in timestamps)
    assert timestamps[-1] >= 5500
    assert any(abs(timestamp - 1400) <= 300 for timestamp in timestamps)
    assert all(
        item.mime_type == "image/jpeg" and item.encoded_bytes.startswith(b"\xff\xd8\xff") for item in first.evidence
    )


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_probe_records_audio_presence_separately_from_audio_processing():
    video_bytes = _make_mp4(frame_count=10, include_audio=True)
    result = decode_and_sample_video(
        file_data=video_bytes,
        filename="with-audio.mp4",
        declared_mime="video/mp4",
        config=_config(),
    )
    assert result.probe.audio_presence == "present"
    assert result.asset.audio_presence == "present"
    assert result.asset.audio_processing == "not_requested"


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_video_validation_rejects_mime_extension_and_corrupt_container():
    video_bytes = _make_mp4(frame_count=5)
    with pytest.raises(MediaValidationError) as mime_error:
        decode_and_sample_video(
            file_data=video_bytes,
            filename="video.mp4",
            declared_mime="video/webm",
            config=_config(),
        )
    assert mime_error.value.code == "media.mime_mismatch"

    with pytest.raises(MediaValidationError) as extension_error:
        decode_and_sample_video(
            file_data=video_bytes,
            filename="video.mkv",
            declared_mime="video/mp4",
            config=_config(),
        )
    assert extension_error.value.code == "media.extension_mismatch"

    corrupt = video_bytes[:32]
    with pytest.raises(VideoDecodeError) as decode_error:
        decode_and_sample_video(
            file_data=corrupt,
            filename="video.mp4",
            declared_mime="video/mp4",
            config=_config(),
        )
    assert decode_error.value.code == "media.video_decode_failed"


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"max_duration_seconds": 1}, "media.video_duration_exceeded"),
        ({"max_pixels_per_frame": 160 * 96 - 1}, "media.video_pixels_exceeded"),
        ({"max_decoded_frames": 4, "max_frames": 4}, "media.video_frame_count_exceeded"),
        ({"max_decoded_work_pixels": 160 * 96 * 2}, "media.video_decoded_work_exceeded"),
        ({"max_probe_frames": 4, "max_frames": 4}, "media.video_probe_frames_exceeded"),
        ({"max_candidate_encoded_bytes": 1}, "media.video_candidate_bytes_exceeded"),
        ({"decode_timeout_seconds": 1e-12}, "media.video_decode_timeout"),
    ],
)
def test_video_decode_work_budgets_fail_with_stable_codes(overrides, expected_code):
    video_bytes = _make_mp4(frame_count=30)
    with pytest.raises(MediaBudgetExceededError) as exc_info:
        decode_and_sample_video(
            file_data=video_bytes,
            filename="budget.mp4",
            declared_mime="video/mp4",
            config=_config(**overrides),
        )
    assert exc_info.value.code == expected_code


def test_spawned_decoder_hard_timeout_kills_and_reaps_a_stalled_child(tmp_path):
    import hms_api.engine.multimodal.video as video_module

    marker = tmp_path / "decoder-worker.pid"
    existing_child_pids = {child.pid for child in multiprocessing.active_children()}
    started_at = time.monotonic()

    with pytest.raises(MediaBudgetExceededError) as exc_info:
        video_module._run_video_decode_in_subprocess(
            file_data=b"bounded in-memory request",
            filename="stall.mp4",
            declared_mime="video/mp4",
            config=_config(decode_timeout_seconds=3),
            asset_id=str(marker),
            worker_target=_stalling_spawn_video_worker,
        )

    elapsed = time.monotonic() - started_at
    assert marker.exists(), "the spawned target must enter its simulated native stall"
    stalled_pid = int(marker.read_text(encoding="utf-8"))
    assert exc_info.value.code == "media.video_decode_timeout"
    assert elapsed < 6
    assert stalled_pid not in {
        child.pid for child in multiprocessing.active_children() if child.pid not in existing_child_pids
    }
    assert not any(
        thread.is_alive() and thread.name in {"hms-video-input", "hms-video-output"} for thread in threading.enumerate()
    )


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_video_byte_budget_is_checked_before_decoder_work():
    video_bytes = _make_mp4(frame_count=5)
    with pytest.raises(MediaBudgetExceededError) as exc_info:
        decode_and_sample_video(
            file_data=video_bytes,
            filename="budget.mp4",
            declared_mime="video/mp4",
            config=_config(max_bytes=len(video_bytes) - 1),
        )
    assert exc_info.value.code == "media.video_bytes_exceeded"


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_missing_decoder_is_an_early_typed_failure(monkeypatch):
    import hms_api.engine.multimodal.video as video_module

    video_bytes = _make_mp4(frame_count=5)
    monkeypatch.setattr(video_module, "_av", None)
    with pytest.raises(VideoDecoderUnavailableError) as exc_info:
        decode_and_sample_video(
            file_data=video_bytes,
            filename="decoder.mp4",
            declared_mime="video/mp4",
            config=_config(),
        )
    assert exc_info.value.code == "media.video_decoder_unavailable"
