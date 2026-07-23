"""Offline contract tests for grounded multimodal descriptions."""

import hashlib

import pytest
from pydantic import ValidationError

from hms_api.engine.multimodal import (
    GroundedEntity,
    GroundedStatement,
    GroundingError,
    MediaAsset,
    ModelMultimodalDescription,
    ModelTemporalSegment,
    SegmentWindow,
    SystemProvenance,
    VisibleText,
    VisualEvidence,
    VisualObservation,
    flat_provenance_metadata,
    normalize_description,
    render_canonical_markdown,
)
from hms_api.engine.multimodal.serialization import (
    CANONICAL_CHUNK_CONTRACT,
    DEFAULT_CANONICAL_ATOM_MAX_CHARS,
)


def _asset(kind: str = "image") -> MediaAsset:
    return MediaAsset(
        asset_id="asset-test",
        sha256="a" * 64,
        media_kind=kind,
        detected_mime="image/png" if kind == "image" else "video/mp4",
        original_filename="synthetic.png" if kind == "image" else "synthetic.mp4",
        byte_size=123,
        width=64,
        height=32,
        duration_ms=None if kind == "image" else 4_000,
        audio_presence="absent",
        audio_processing="not_requested",
    )


def _evidence(evidence_id: str, timestamp_ms: int | None = None) -> VisualEvidence:
    payload = f"bytes:{evidence_id}".encode()
    return VisualEvidence(
        evidence_id=evidence_id,
        timestamp_ms=timestamp_ms,
        sha256=hashlib.sha256(payload).hexdigest(),
        mime_type="image/png",
        width=64,
        height=32,
        encoded_bytes=payload,
    )


def _statement(text: str, evidence_id: str) -> GroundedStatement:
    return GroundedStatement(text=text, evidence_ids=[evidence_id], uncertainty="low")


def _provenance() -> SystemProvenance:
    return SystemProvenance(
        provider="fake",
        configured_model="gpt-5-mini",
        resolved_model="gpt-5-mini-test",
        pipeline_version="hms-multimodal-v1",
        prompt_version="mm-prompt-v1",
        schema_version="mm-schema-v1",
        sampling_version="image-v1",
        pipeline_fingerprint="b" * 64,
        provider_request_id="req-test",
        input_tokens=10,
        output_tokens=20,
        logical_calls=1,
        physical_attempts=1,
    )


def test_visual_evidence_never_serializes_encoded_bytes():
    evidence = _evidence("image-000")

    assert "encoded_bytes" not in evidence.model_dump()
    assert "encoded_bytes" not in evidence.model_dump_json()
    assert b"bytes:image-000" not in repr(evidence).encode()


def test_every_semantic_statement_requires_evidence():
    with pytest.raises(ValidationError, match="at least 1 item"):
        GroundedStatement(text="visible fact", evidence_ids=[], uncertainty="low")


def test_image_rejects_unknown_evidence_and_timeline():
    evidence = _evidence("image-000")
    unknown = ModelMultimodalDescription(
        summary=[_statement("editor is open", "missing")],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[],
        limitations=[],
    )
    with pytest.raises(GroundingError) as exc_info:
        normalize_description(
            media_kind="image",
            asset=_asset(),
            description=unknown,
            evidence=[evidence],
            segment_windows=[],
            provenance=_provenance(),
        )
    assert exc_info.value.code == "grounding.unknown_evidence"

    segment = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[_statement("state", "image-000")],
        observations=[],
        visible_text=[],
        evidence_ids=["image-000"],
    )
    with pytest.raises(GroundingError, match="Image descriptions"):
        normalize_description(
            media_kind="image",
            asset=_asset(),
            description=unknown.model_copy(
                update={"summary": [_statement("editor", "image-000")], "temporal_segments": [segment]}
            ),
            evidence=[evidence],
            segment_windows=[],
            provenance=_provenance(),
        )


def test_video_time_ranges_are_system_owned_and_render_deterministically():
    first = _evidence("frame-000", 500)
    second = _evidence("frame-001", 2_500)
    video_asset = _asset("video").model_copy(update={"audio_presence": "present", "audio_processing": "not_requested"})
    segment = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[_statement("tests changed from failing to passing", "frame-001")],
        observations=[
            VisualObservation(
                text="terminal shows tests passed",
                evidence_ids=["frame-001"],
                uncertainty="low",
                kind="ui",
            )
        ],
        visible_text=[
            VisibleText(text="2 passed", evidence_ids=["frame-001"], uncertainty="low"),
        ],
        evidence_ids=["frame-000", "frame-001"],
    )
    output = ModelMultimodalDescription(
        summary=[_statement("a coding test run is shown", "frame-000")],
        entities=[GroundedEntity(name="pytest", evidence_ids=["frame-001"], uncertainty="low")],
        observations=[],
        visible_text=[],
        temporal_segments=[segment],
        limitations=[_statement("small text may be incomplete", "frame-000")],
    )

    normalized = normalize_description(
        media_kind="video",
        asset=video_asset,
        description=output,
        evidence=[first, second],
        segment_windows=[
            SegmentWindow(
                segment_id="segment-000",
                start_ms=500,
                end_ms=2_500,
                evidence_ids=["frame-000", "frame-001"],
            )
        ],
        provenance=_provenance(),
    )

    assert normalized.temporal_segments[0].start_ms == 500
    assert normalized.temporal_segments[0].end_ms == 2_500
    rendered_once = render_canonical_markdown(normalized)
    rendered_twice = render_canonical_markdown(normalized)
    assert rendered_once == rendered_twice
    assert "[00:00:00.500–00:00:02.500]" in rendered_once
    assert "Pipeline version: hms-multimodal-v1" in rendered_once
    assert "2 passed" in rendered_once
    assert "base64" not in rendered_once

    metadata = flat_provenance_metadata(normalized, source_available=False)
    assert metadata["media_source_available"] == "false"
    assert metadata["media_pipeline_version"] == "hms-multimodal-v1"
    assert metadata["media_audio_presence"] == "present"
    assert metadata["media_audio_processing"] == "not_requested"
    assert all(isinstance(value, str) for value in metadata.values())


def test_long_video_atoms_remain_provenance_closed_after_real_chunks_splitter():
    """No generic retain chunk may contain timeline body without its envelope."""

    from hms_api.engine.retain.fact_extraction import chunk_text

    first = _evidence("frame-000", 500)
    second = _evidence("frame-001", 2_500)
    # Include Markdown-looking text from the model. JSON scalar rendering must
    # keep it data rather than allowing it to create a synthetic segment header.
    alpha_text = ("ALPHA_BODY " * 900) + "\n### [99:99:99.999] segment=segment-injected"
    beta_text = "BETA_BODY " * 900
    first_segment = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[_statement(alpha_text, "frame-000")],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-000"],
    )
    second_segment = ModelTemporalSegment(
        segment_id="segment-001",
        summary=[_statement(beta_text, "frame-001")],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-001"],
    )
    normalized = normalize_description(
        media_kind="video",
        asset=_asset("video"),
        description=ModelMultimodalDescription(
            summary=[_statement("A long synthetic coding video.", "frame-000")],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[first_segment, second_segment],
            limitations=[],
        ),
        evidence=[first, second],
        segment_windows=[
            SegmentWindow(
                segment_id="segment-000",
                start_ms=0,
                end_ms=1_000,
                evidence_ids=["frame-000"],
            ),
            SegmentWindow(
                segment_id="segment-001",
                start_ms=2_000,
                end_ms=3_000,
                evidence_ids=["frame-001"],
            ),
        ],
        provenance=_provenance(),
    )

    rendered = render_canonical_markdown(normalized)
    assert rendered == render_canonical_markdown(normalized)
    assert f"Canonical chunk contract: {CANONICAL_CHUNK_CONTRACT}" in rendered
    assert "\n### [99:99:99.999] segment=segment-injected" not in rendered

    semantic_blocks = [block for block in rendered.rstrip().split("\n\n") if block.startswith("[canonical-atom:")]
    timeline_blocks = [block for block in semantic_blocks if "\n## Timeline\n" in block]
    assert len(timeline_blocks) > 2  # Both oversized statements were fragmented.
    assert all(len(block) <= DEFAULT_CANONICAL_ATOM_MAX_CHARS for block in semantic_blocks)
    assert all(f"[canonical-atom: {CANONICAL_CHUNK_CONTRACT}]" in block for block in semantic_blocks)
    assert all(f"Asset: sha256:{normalized.asset.sha256}" in block for block in semantic_blocks)
    assert all("Media kind: video" in block for block in semantic_blocks)
    assert all("[evidence:" in block and "[uncertainty:" in block for block in semantic_blocks)
    assert all("segment=" in block and "Time: [" in block for block in timeline_blocks)
    assert all("Atom: sha256:" in block and " part=" in block for block in timeline_blocks)

    # Exercise the exact generic splitter used by chunks retain mode. It may
    # pack several complete atoms together, but it must not split any one atom.
    retain_chunks = chunk_text(rendered, max_chars=3_000)
    assert all(len(chunk) <= 3_000 for chunk in retain_chunks)
    alpha_chunks = [chunk for chunk in retain_chunks if "ALPHA_BODY" in chunk]
    beta_chunks = [chunk for chunk in retain_chunks if "BETA_BODY" in chunk]
    assert len(alpha_chunks) > 1
    assert len(beta_chunks) > 1
    assert all("[00:00:00.000–00:00:01.000]" in chunk for chunk in alpha_chunks)
    assert all("segment=segment-000" in chunk for chunk in alpha_chunks)
    assert all("[evidence: frame-000]" in chunk for chunk in alpha_chunks)
    assert all("[00:00:02.000–00:00:03.000]" in chunk for chunk in beta_chunks)
    assert all("segment=segment-001" in chunk for chunk in beta_chunks)
    assert all("[evidence: frame-001]" in chunk for chunk in beta_chunks)


def test_provenance_closed_renderer_fails_when_envelope_cannot_fit():
    evidence = _evidence("image-000")
    normalized = normalize_description(
        media_kind="image",
        asset=_asset(),
        description=ModelMultimodalDescription(
            summary=[_statement("visible state", "image-000")],
            entities=[],
            observations=[],
            visible_text=[],
            temporal_segments=[],
            limitations=[],
        ),
        evidence=[evidence],
        segment_windows=[],
        provenance=_provenance(),
    )

    with pytest.raises(ValueError, match="envelope exceeds"):
        render_canonical_markdown(normalized, max_atom_chars=64)


def test_video_rejects_unknown_segments_and_cross_window_evidence():
    first = _evidence("frame-000", 500)
    second = _evidence("frame-001", 2_500)
    first_window = SegmentWindow(
        segment_id="segment-000",
        start_ms=0,
        end_ms=1_000,
        evidence_ids=["frame-000"],
    )
    second_window = SegmentWindow(
        segment_id="segment-001",
        start_ms=2_000,
        end_ms=3_000,
        evidence_ids=["frame-001"],
    )
    first_segment = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[_statement("first state", "frame-000")],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-000"],
    )
    second_segment = ModelTemporalSegment(
        segment_id="segment-001",
        summary=[_statement("second state", "frame-001")],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-001"],
    )

    unknown_segment = ModelMultimodalDescription(
        summary=[_statement("video", "frame-000")],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[
            first_segment.model_copy(update={"segment_id": "segment-unknown"}),
            second_segment,
        ],
        limitations=[],
    )
    with pytest.raises(GroundingError) as unknown_error:
        normalize_description(
            media_kind="video",
            asset=_asset("video"),
            description=unknown_segment,
            evidence=[first, second],
            segment_windows=[first_window, second_window],
            provenance=_provenance(),
        )
    assert unknown_error.value.code == "grounding.segment_coverage"

    cross_window = ModelMultimodalDescription(
        summary=[_statement("video", "frame-000")],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[
            first_segment.model_copy(update={"summary": [_statement("future state", "frame-001")]}),
            second_segment,
        ],
        limitations=[],
    )
    with pytest.raises(GroundingError) as evidence_error:
        normalize_description(
            media_kind="video",
            asset=_asset("video"),
            description=cross_window,
            evidence=[first, second],
            segment_windows=[first_window, second_window],
            provenance=_provenance(),
        )
    assert evidence_error.value.code == "grounding.segment_statement_mismatch"


def test_video_rejects_segment_window_outside_system_duration():
    evidence = _evidence("frame-000", 2_500)
    segment = ModelTemporalSegment(
        segment_id="segment-000",
        summary=[_statement("state", "frame-000")],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-000"],
    )

    with pytest.raises(GroundingError) as exc_info:
        normalize_description(
            media_kind="video",
            asset=_asset("video"),
            description=ModelMultimodalDescription(
                summary=[_statement("video", "frame-000")],
                entities=[],
                observations=[],
                visible_text=[],
                temporal_segments=[segment],
                limitations=[],
            ),
            evidence=[evidence],
            segment_windows=[
                SegmentWindow(
                    segment_id="segment-000",
                    start_ms=2_000,
                    end_ms=5_000,
                    evidence_ids=["frame-000"],
                )
            ],
            provenance=_provenance(),
        )

    assert exc_info.value.code == "grounding.segment_out_of_range"


def test_model_cannot_smuggle_system_time_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelTemporalSegment.model_validate(
            {
                "segment_id": "segment-000",
                "start_ms": 0,
                "end_ms": 1_000,
                "summary": [
                    {"text": "state", "evidence_ids": ["frame-000"], "uncertainty": "low"},
                ],
                "observations": [],
                "visible_text": [],
                "evidence_ids": ["frame-000"],
            }
        )
