"""Offline image-to-canonical-text parser integration tests."""

import hashlib
import io
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from hms_api.engine.multimodal import (
    FakeMultimodalProvider,
    GroundedStatement,
    ImageNormalizationConfig,
    MediaAsset,
    ModelMultimodalDescription,
    ModelTemporalSegment,
    ProviderUnavailableError,
    VideoProcessingConfig,
    VideoSegmentCheckpoint,
    VisualEvidence,
    decode_and_sample_video,
    derive_video_segment_identity,
    normalize_image,
    video_decoder_available,
)
from hms_api.engine.parsers import (
    ConversionInput,
    MultimodalParserConfig,
    OpenAIMultimodalParser,
    ParserNotApplicableError,
    ParserProcessingError,
    create_openai_multimodal_parser,
)


class _RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.in_flight: list[tuple[str, str, int]] = []

    def record_multimodal_pipeline(self, **event) -> None:
        self.events.append(event)

    @contextmanager
    def record_multimodal_in_flight(self, *, media_kind: str, stage: str):
        self.in_flight.append((media_kind, stage, 1))
        try:
            yield
        finally:
            self.in_flight.append((media_kind, stage, -1))


def _png_bytes() -> bytes:
    image = Image.new("RGB", (48, 24), (245, 245, 245))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _description(evidence_id: str) -> ModelMultimodalDescription:
    return ModelMultimodalDescription.model_validate(
        {
            "summary": [
                {
                    "text": "A code editor displays a Python function.",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "low",
                }
            ],
            "entities": [
                {
                    "name": "Python",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "low",
                },
                {
                    "name": "ambiguous package name",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "high",
                },
            ],
            "observations": [
                {
                    "text": "The editor and terminal panes are side by side.",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "medium",
                    "kind": "spatial",
                }
            ],
            "visible_text": [
                {
                    "text": "def retain():",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "low",
                }
            ],
            "temporal_segments": [],
            "limitations": [
                {
                    "text": "Small text may be unreadable.",
                    "evidence_ids": [evidence_id],
                    "uncertainty": "high",
                }
            ],
        }
    )


def test_pipeline_fingerprint_tracks_normalizer_runtime(monkeypatch):
    import hms_api.engine.parsers.openai_multimodal as parser_module

    provider = FakeMultimodalProvider(_description("image-000-placeholder"))
    parser = OpenAIMultimodalParser(provider)
    baseline = parser.pipeline_fingerprint()
    monkeypatch.setattr(parser_module, "image_normalizer_identity", lambda: {"pillow": "changed"})

    assert parser.pipeline_fingerprint() != baseline


def test_pipeline_fingerprint_tracks_canonical_chunk_contract(monkeypatch):
    import hms_api.engine.parsers.openai_multimodal as parser_module

    provider = FakeMultimodalProvider(_description("image-000-placeholder"))
    parser = OpenAIMultimodalParser(provider)
    baseline = parser.pipeline_fingerprint()
    monkeypatch.setattr(parser_module, "CANONICAL_CHUNK_CONTRACT", "provenance-closed-test")

    assert parser.pipeline_fingerprint() != baseline


def test_direct_parser_config_rejects_schema_incompatible_frame_plans() -> None:
    with pytest.raises(ValueError, match="max_frames_per_call must be <= 64"):
        MultimodalParserConfig(max_frames_per_call=65)

    video = VideoProcessingConfig(max_frames=257)
    with pytest.raises(ValueError, match=r"ceil\(video.max_frames / max_frames_per_call\).*<= 256"):
        MultimodalParserConfig(video=video, max_frames_per_call=1)


def _parser_for(data: bytes) -> tuple[OpenAIMultimodalParser, FakeMultimodalProvider]:
    image_config = ImageNormalizationConfig()
    evidence_id = normalize_image(
        file_data=data,
        filename="editor.png",
        declared_mime="image/png",
        config=image_config,
    ).evidence.evidence_id
    provider = FakeMultimodalProvider(_description(evidence_id))
    parser = OpenAIMultimodalParser(provider, MultimodalParserConfig(image=image_config))
    return parser, provider


@pytest.mark.asyncio
async def test_image_parser_produces_grounded_chunks_result_without_payload_leakage(monkeypatch) -> None:
    import hms_api.metrics as metrics_module

    data = _png_bytes()
    parser, provider = _parser_for(data)
    metrics = _RecordingMetrics()
    monkeypatch.setattr(metrics_module, "get_metrics_collector", lambda: metrics)

    result = await parser.convert_input(
        ConversionInput(
            file_data=data,
            filename="editor.png",
            content_type="image/png",
            source_available=False,
        )
    )

    assert provider.calls == 1
    assert result.retain_extraction_mode == "chunks"
    assert '"def retain():"' in result.content
    assert "[evidence: image-000-" in result.content
    assert result.metadata["media_kind"] == "image"
    assert result.metadata["media_descriptor_model"] == "gpt-5-mini"
    assert result.metadata["media_pipeline_version"] == "hms-multimodal-v1"
    assert result.metadata["media_audio_presence"] == "absent"
    assert result.metadata["media_audio_processing"] == "not_requested"
    assert result.metadata["media_source_available"] == "false"
    assert result.entities == [{"text": "Python", "type": "CONCEPT"}]
    assert result.pipeline_metadata["stage"] == "normalized"
    assert len(result.metadata["media_pipeline_fingerprint"]) == 64

    serialized_surfaces = repr((result.content, result.metadata, result.entities, result.pipeline_metadata))
    assert "base64," not in serialized_surfaces
    assert data.hex() not in serialized_surfaces
    assert {event["stage"] for event in metrics.events if event["success"]} >= {
        "validation",
        "normalize",
        "preprocess",
        "describe",
        "complete",
    }
    assert [event.get("asset_outcome") for event in metrics.events].count("accepted") == 1
    assert metrics.in_flight == [("image", "describe", 1), ("image", "describe", -1)]


@pytest.mark.asyncio
async def test_grounding_failure_is_terminal_and_sanitized() -> None:
    data = _png_bytes()
    invalid_provider = FakeMultimodalProvider(_description("unknown-evidence"))
    parser = OpenAIMultimodalParser(invalid_provider)

    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(ConversionInput(file_data=data, filename="editor.png", content_type="image/png"))

    assert exc_info.value.code == "grounding.unknown_evidence"
    assert "unknown-evidence" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_non_media_can_fallback_but_corrupt_declared_image_is_terminal() -> None:
    parser = OpenAIMultimodalParser(FakeMultimodalProvider(_description("unused")))

    with pytest.raises(ParserNotApplicableError):
        await parser.convert_input(
            ConversionInput(file_data=b"plain text", filename="notes.txt", content_type="text/plain")
        )

    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(
            ConversionInput(file_data=b"not really png", filename="screen.png", content_type="image/png")
        )
    assert exc_info.value.code == "media.unsupported_image"


@pytest.mark.asyncio
async def test_asset_hash_mismatch_fails_before_provider() -> None:
    data = _png_bytes()
    parser, provider = _parser_for(data)

    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(
            ConversionInput(
                file_data=data,
                filename="editor.png",
                content_type="image/png",
                asset_sha256="0" * 64,
            )
        )

    assert exc_info.value.code == "media.asset_hash_mismatch"
    assert provider.calls == 0


def _production_config(**overrides):
    values = {
        "multimodal_enabled": True,
        "multimodal_image_enabled": True,
        "multimodal_capability_responses_api": True,
        "multimodal_capability_image_input": True,
        "multimodal_capability_structured_outputs": True,
        "multimodal_api_key": "sentinel-secret-key",
        "multimodal_base_url": "https://api.openai.com/v1",
        "multimodal_model": "gpt-5-mini",
        "multimodal_image_detail": "auto",
        "multimodal_request_timeout_seconds": 30.0,
        "multimodal_max_output_tokens": 1024,
        "multimodal_max_retries": 2,
        "multimodal_max_schema_repairs": 1,
        "multimodal_max_concurrency": 2,
        "multimodal_max_image_bytes": 2_000_000,
        "multimodal_max_image_pixels": 4_000_000,
        "multimodal_max_frames_per_call": 4,
        "multimodal_model_behavior_version": "alias-v1",
        "multimodal_prompt_version": "prompt-v1",
        "multimodal_schema_version": "schema-v1",
        "multimodal_sampling_version": "sampling-v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_production_factory_is_opt_in_and_capability_gated() -> None:
    assert create_openai_multimodal_parser(_production_config(multimodal_enabled=False)) is None
    with pytest.raises(ValueError, match="structured-output"):
        create_openai_multimodal_parser(_production_config(multimodal_capability_structured_outputs=False))

    parser = create_openai_multimodal_parser(_production_config())
    assert parser is not None
    assert parser.name() == "openai_multimodal"
    assert parser.supports("screen.png", "image/png")
    assert "sentinel-secret-key" not in repr(parser)
    await parser.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
async def test_video_parser_runs_bounded_map_reduce_and_renders_system_timeline(monkeypatch) -> None:
    import hms_api.metrics as metrics_module
    from tests.test_multimodal_video import _config as video_config
    from tests.test_multimodal_video import _make_mp4

    data = _make_mp4(frame_count=30)
    processing_config = video_config(max_frames=5)
    decoded = decode_and_sample_video(
        file_data=data,
        filename="coding-session.mp4",
        declared_mime="video/mp4",
        config=processing_config,
    )
    evidence = list(decoded.evidence)
    mapped_segments: list[ModelTemporalSegment] = []
    for index, offset in enumerate(range(0, len(evidence), 2)):
        batch = evidence[offset : offset + 2]
        ids = [item.evidence_id for item in batch]
        mapped_segments.append(
            ModelTemporalSegment(
                segment_id=f"segment-{index:03d}",
                summary=[
                    GroundedStatement(
                        text=f"Editor state {index} is visible.",
                        evidence_ids=ids,
                        uncertainty="low",
                    )
                ],
                observations=[],
                visible_text=[],
                evidence_ids=ids,
            )
        )
    all_ids = [item.evidence_id for item in evidence]
    description = ModelMultimodalDescription(
        summary=[
            GroundedStatement(
                text="The coding screencast shows changing editor states.",
                evidence_ids=all_ids,
                uncertainty="low",
            )
        ],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=mapped_segments,
        limitations=[],
    )
    provider = FakeMultimodalProvider(
        description,
        segments={segment.segment_id: segment for segment in mapped_segments},
    )
    parser = OpenAIMultimodalParser(
        provider,
        MultimodalParserConfig(
            video=processing_config,
            max_frames_per_call=2,
            sampling_version=processing_config.sampling_version,
        ),
    )
    metrics = _RecordingMetrics()
    monkeypatch.setattr(metrics_module, "get_metrics_collector", lambda: metrics)

    result = await parser.convert_input(
        ConversionInput(
            file_data=data,
            filename="coding-session.mp4",
            content_type="video/mp4",
        )
    )

    assert provider.calls == len(mapped_segments) + 1
    assert result.metadata["media_kind"] == "video"
    assert result.metadata["media_pipeline_version"] == "hms-multimodal-v1"
    assert result.metadata["media_audio_presence"] == "absent"
    assert result.metadata["media_audio_processing"] == "not_requested"
    assert result.retain_extraction_mode == "chunks"
    assert "Processing scope: visual-only; audio not processed" in result.content
    assert "## Timeline" in result.content
    assert "segment=segment-000" in result.content
    assert "00:00:" in result.content
    assert result.pipeline_metadata["logical_calls"] == len(mapped_segments) + 1
    assert "base64," not in repr(result)
    successful_by_stage = {event["stage"]: event for event in metrics.events if event["success"]}
    assert {"validation", "decode", "normalize", "sample", "describe", "complete"} <= successful_by_stage.keys()
    assert successful_by_stage["sample"]["candidate_frames"] >= len(evidence)
    assert successful_by_stage["sample"]["selected_frames"] == len(evidence)
    assert successful_by_stage["decode"]["duration"] > 0
    assert successful_by_stage["normalize"]["duration"] > 0
    assert successful_by_stage["sample"]["duration"] > 0
    assert successful_by_stage["describe"]["duration"] > 0
    assert [event.get("asset_outcome") for event in metrics.events].count("accepted") == 1


@pytest.mark.asyncio
async def test_video_retry_reuses_durable_validated_map_checkpoint(monkeypatch) -> None:
    import hms_api.engine.parsers.openai_multimodal as parser_module
    import hms_api.metrics as metrics_module

    source = b"synthetic-video-checkpoint-source"
    source_sha = hashlib.sha256(source).hexdigest()
    evidence = tuple(
        VisualEvidence(
            evidence_id=f"frame-{index:03d}",
            timestamp_ms=index * 1_000,
            sha256=hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
            mime_type="image/jpeg",
            width=16,
            height=8,
            encoded_bytes=f"ephemeral-frame-{index}".encode(),
        )
        for index in range(3)
    )
    asset = MediaAsset(
        asset_id="asset-checkpoint",
        sha256=source_sha,
        media_kind="video",
        detected_mime="video/mp4",
        original_filename="checkpoint.mp4",
        byte_size=len(source),
        width=16,
        height=8,
        duration_ms=2_000,
        audio_presence="absent",
        audio_processing="not_requested",
    )
    mapped = {
        f"segment-{index:03d}": ModelTemporalSegment(
            segment_id=f"segment-{index:03d}",
            summary=[
                GroundedStatement(
                    text=f"Editor state {index} is visible.",
                    evidence_ids=[evidence[index].evidence_id],
                    uncertainty="low",
                )
            ],
            observations=[],
            visible_text=[],
            evidence_ids=[evidence[index].evidence_id],
        )
        for index in range(3)
    }
    description = ModelMultimodalDescription(
        summary=[mapped["segment-000"].summary[0]],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=list(mapped.values()),
        limitations=[],
    )

    class _FailSecondSegmentOnce(FakeMultimodalProvider):
        def __init__(self):
            super().__init__(description, segments=mapped)
            self.segment_calls = {segment_id: 0 for segment_id in mapped}
            self.reduce_calls = 0

        async def describe_video_segment(self, segment_id, segment_evidence):
            self.segment_calls[segment_id] += 1
            if segment_id == "segment-001" and self.segment_calls[segment_id] == 1:
                raise ProviderUnavailableError(
                    "provider.unavailable",
                    "Provider remained unavailable after retries",
                    retryable=True,
                    logical_calls=1,
                    physical_attempts=2,
                )
            return await super().describe_video_segment(segment_id, segment_evidence)

        async def reduce_video(self, segments):
            self.reduce_calls += 1
            return await super().reduce_video(segments)

    def unsupported_image(**_kwargs):
        from hms_api.engine.multimodal import MediaValidationError

        raise MediaValidationError("media.unsupported_image", "Unsupported image")

    monkeypatch.setattr(parser_module, "normalize_image", unsupported_image)
    monkeypatch.setattr(parser_module, "detect_video_magic", lambda _data: object())
    monkeypatch.setattr(
        parser_module,
        "decode_and_sample_video",
        lambda **_kwargs: SimpleNamespace(asset=asset, evidence=evidence),
    )
    metrics = _RecordingMetrics()
    monkeypatch.setattr(metrics_module, "get_metrics_collector", lambda: metrics)
    provider = _FailSecondSegmentOnce()
    parser = OpenAIMultimodalParser(
        provider,
        MultimodalParserConfig(video=VideoProcessingConfig(max_frames=4), max_frames_per_call=1),
    )
    checkpoints = {}

    async def load_checkpoint(identity):
        return checkpoints.get(identity.segment_key)

    async def save_checkpoint(checkpoint):
        checkpoints[checkpoint.segment_key] = checkpoint

    request = ConversionInput(
        file_data=source,
        filename="checkpoint.mp4",
        content_type="video/mp4",
        asset_sha256=source_sha,
        load_video_segment_checkpoint=load_checkpoint,
        save_video_segment_checkpoint=save_checkpoint,
    )

    with pytest.raises(ParserProcessingError) as first_failure:
        await parser.convert_input(request)
    assert first_failure.value.code == "provider.unavailable"
    assert len(checkpoints) == 1
    # A map-stage failure is terminal for this attempt: no reducer call and no
    # canonical output/child-retain input can be produced from a prefix.
    assert provider.reduce_calls == 0

    result = await parser.convert_input(request)

    assert provider.segment_calls == {"segment-000": 1, "segment-001": 2, "segment-002": 1}
    assert provider.reduce_calls == 1
    assert len(checkpoints) == 3
    assert result.pipeline_metadata["segment_checkpoint_hits"] == 1
    assert result.pipeline_metadata["logical_calls"] == 4
    successful_describe = [event for event in metrics.events if event["stage"] == "describe" and event["success"]]
    assert successful_describe[-1]["logical_calls"] == 3
    assert successful_describe[-1]["physical_attempts"] == 3
    assert successful_describe[-1]["deduplicated"] is True
    assert "ephemeral-frame" not in repr(checkpoints)
    assert "base64," not in repr(checkpoints)


@pytest.mark.asyncio
async def test_video_cached_checkpoint_must_match_system_evidence_identity(monkeypatch) -> None:
    """A stale/wrong-window checkpoint is rejected before reduce or provider I/O."""

    import hms_api.engine.parsers.openai_multimodal as parser_module

    source = b"synthetic-video-checkpoint-identity"
    source_sha = hashlib.sha256(source).hexdigest()
    evidence = (
        VisualEvidence(
            evidence_id="frame-000",
            timestamp_ms=0,
            sha256=hashlib.sha256(b"frame-000").hexdigest(),
            mime_type="image/jpeg",
            width=16,
            height=8,
            encoded_bytes=b"ephemeral-frame-000",
        ),
    )
    asset = MediaAsset(
        asset_id="asset-checkpoint-identity",
        sha256=source_sha,
        media_kind="video",
        detected_mime="video/mp4",
        original_filename="checkpoint-identity.mp4",
        byte_size=len(source),
        width=16,
        height=8,
        duration_ms=0,
        audio_presence="absent",
        audio_processing="not_requested",
    )
    mapped = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[
            GroundedStatement(
                text="A bounded editor state is visible.",
                evidence_ids=["frame-000"],
                uncertainty="low",
            )
        ],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-000"],
    )
    description = ModelMultimodalDescription(
        summary=list(mapped.summary),
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[mapped],
        limitations=[],
    )
    provider = FakeMultimodalProvider(description, segments={"segment-000": mapped})

    def unsupported_image(**_kwargs):
        from hms_api.engine.multimodal import MediaValidationError

        raise MediaValidationError("media.unsupported_image", "Unsupported image")

    monkeypatch.setattr(parser_module, "normalize_image", unsupported_image)
    monkeypatch.setattr(parser_module, "detect_video_magic", lambda _data: object())
    monkeypatch.setattr(
        parser_module,
        "decode_and_sample_video",
        lambda **_kwargs: SimpleNamespace(asset=asset, evidence=evidence),
    )

    identity = derive_video_segment_identity("segment-000", evidence)
    # The segment key is stable, but the cached row belongs to a different
    # system evidence window.  A durable adapter must not silently reuse it.
    stale_checkpoint = VideoSegmentCheckpoint(
        segment_key=identity.segment_key,
        segment_id=identity.segment_id,
        evidence_fingerprint="0" * 64,
        value=mapped,
        provider="fake",
        configured_model="gpt-5-mini",
        resolved_model="fake-gpt-5-mini",
        request_id="cached-stale",
        input_tokens=0,
        output_tokens=0,
        logical_calls=1,
        physical_attempts=1,
    )
    seen_identities = []
    saved = []

    async def load_checkpoint(requested_identity):
        seen_identities.append(requested_identity)
        return stale_checkpoint

    async def save_checkpoint(checkpoint):
        saved.append(checkpoint)

    parser = OpenAIMultimodalParser(
        provider,
        MultimodalParserConfig(video=VideoProcessingConfig(max_frames=4), max_frames_per_call=1),
    )
    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(
            ConversionInput(
                file_data=source,
                filename="checkpoint-identity.mp4",
                content_type="video/mp4",
                asset_sha256=source_sha,
                load_video_segment_checkpoint=load_checkpoint,
                save_video_segment_checkpoint=save_checkpoint,
            )
        )

    assert exc_info.value.code == "grounding.segment_checkpoint_identity"
    assert seen_identities == [identity]
    assert provider.calls == 0
    assert saved == []


@pytest.mark.asyncio
@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
async def test_video_reducer_cannot_mutate_validated_map_segments() -> None:
    from tests.test_multimodal_video import _config as video_config
    from tests.test_multimodal_video import _make_mp4

    data = _make_mp4(frame_count=10)
    processing_config = video_config(max_frames=4)
    decoded = decode_and_sample_video(
        file_data=data,
        filename="coding-session.mp4",
        declared_mime="video/mp4",
        config=processing_config,
    )
    ids = [item.evidence_id for item in decoded.evidence]
    mapped = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[GroundedStatement(text="Mapped fact.", evidence_ids=ids, uncertainty="low")],
        observations=[],
        visible_text=[],
        evidence_ids=ids,
    )
    mutated = mapped.model_copy(
        update={"summary": [GroundedStatement(text="New reducer fact.", evidence_ids=ids, uncertainty="low")]}
    )
    description = ModelMultimodalDescription(
        summary=[GroundedStatement(text="Summary.", evidence_ids=ids, uncertainty="low")],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[mutated],
        limitations=[],
    )

    class MutatingReducer(FakeMultimodalProvider):
        async def reduce_video(self, segments):
            result = await super().reduce_video(segments)
            value = result.value.model_copy(update={"temporal_segments": [mutated]})
            return replace(result, value=value)

    provider = MutatingReducer(description, segments={"segment-000": mapped})
    parser = OpenAIMultimodalParser(
        provider,
        MultimodalParserConfig(
            video=processing_config,
            max_frames_per_call=processing_config.max_frames,
        ),
    )

    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(
            ConversionInput(file_data=data, filename="coding-session.mp4", content_type="video/mp4")
        )

    assert exc_info.value.code == "grounding.reducer_segment_mutation"


@pytest.mark.asyncio
@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
async def test_video_reducer_cannot_add_unproven_top_level_fact() -> None:
    from tests.test_multimodal_video import _config as video_config
    from tests.test_multimodal_video import _make_mp4

    data = _make_mp4(frame_count=10)
    processing_config = video_config(max_frames=4)
    decoded = decode_and_sample_video(
        file_data=data,
        filename="coding-session.mp4",
        declared_mime="video/mp4",
        config=processing_config,
    )
    ids = [item.evidence_id for item in decoded.evidence]
    mapped = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[GroundedStatement(text="Mapped fact.", evidence_ids=ids, uncertainty="low")],
        observations=[],
        visible_text=[],
        evidence_ids=ids,
    )
    description = ModelMultimodalDescription(
        summary=[GroundedStatement(text="Mapped fact.", evidence_ids=ids, uncertainty="low")],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[mapped],
        limitations=[],
    )

    class HallucinatingReducer(FakeMultimodalProvider):
        async def reduce_video(self, segments):
            result = await super().reduce_video(segments)
            value = result.value.model_copy(
                update={
                    "summary": [
                        GroundedStatement(
                            text="HALLUCINATED_REDUCER_FACT",
                            evidence_ids=ids,
                            uncertainty="low",
                        )
                    ]
                }
            )
            return replace(result, value=value)

    parser = OpenAIMultimodalParser(
        HallucinatingReducer(description, segments={"segment-000": mapped}),
        MultimodalParserConfig(
            video=processing_config,
            max_frames_per_call=processing_config.max_frames,
        ),
    )

    with pytest.raises(ParserProcessingError) as exc_info:
        await parser.convert_input(
            ConversionInput(file_data=data, filename="coding-session.mp4", content_type="video/mp4")
        )

    assert exc_info.value.code == "grounding.reducer_unproven_statement"
