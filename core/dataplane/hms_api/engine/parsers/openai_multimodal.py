"""Strict multimodal parser feeding canonical text into the existing retain path."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hms_api.engine.multimodal import (
    GroundingError,
    ImageNormalizationConfig,
    MediaValidationError,
    MultimodalDescriptionProvider,
    MultimodalError,
    SegmentWindow,
    SystemProvenance,
    VideoProcessingConfig,
    VideoSegmentCheckpoint,
    VisualEvidence,
    decode_and_sample_video,
    derive_video_segment_identity,
    detect_image_mime,
    detect_video_magic,
    estimate_provider_budget,
    flat_provenance_metadata,
    image_normalizer_identity,
    normalize_description,
    normalize_image,
    render_canonical_markdown,
    video_decoder_identity,
)
from hms_api.engine.multimodal.provider import OpenAIProviderConfig, OpenAIResponsesMultimodalProvider
from hms_api.engine.multimodal.serialization import CANONICAL_CHUNK_CONTRACT, DEFAULT_CANONICAL_ATOM_MAX_CHARS

from .base import ConversionInput, FileParser, ParserNotApplicableError, ParserOutput, ParserProcessingError

_IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
_IMAGE_MIMES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_VIDEO_MIMES = {"video/avi", "video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-msvideo"}
_SCHEMA_MAX_EVIDENCE_IDS = 64
_SCHEMA_MAX_TEMPORAL_SEGMENTS = 256


@dataclass(frozen=True)
class MultimodalParserConfig:
    """Versioned, non-secret inputs to canonical image conversion."""

    image: ImageNormalizationConfig = field(default_factory=ImageNormalizationConfig)
    video: VideoProcessingConfig | None = None
    max_frames_per_call: int = 4
    pipeline_version: str = "hms-multimodal-v1"
    model_behavior_version: str = "gpt-5-mini-alias-v1"
    prompt_version: str = "openai-mm-v2"
    schema_version: str = "hms-multimodal-description-v1"
    sampling_version: str = "still-image-v1"

    def __post_init__(self) -> None:
        for name in (
            "pipeline_version",
            "model_behavior_version",
            "prompt_version",
            "schema_version",
            "sampling_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.max_frames_per_call <= 0:
            raise ValueError("max_frames_per_call must be positive")
        if self.max_frames_per_call > _SCHEMA_MAX_EVIDENCE_IDS:
            raise ValueError(
                f"max_frames_per_call must be <= {_SCHEMA_MAX_EVIDENCE_IDS} for the configured multimodal schema"
            )
        if self.video is not None and self.max_frames_per_call > self.video.max_frames:
            raise ValueError("max_frames_per_call cannot exceed the video frame budget")
        if self.video is not None:
            maximum_segment_count = (self.video.max_frames + self.max_frames_per_call - 1) // self.max_frames_per_call
            if maximum_segment_count > _SCHEMA_MAX_TEMPORAL_SEGMENTS:
                raise ValueError(
                    "ceil(video.max_frames / max_frames_per_call) must be "
                    f"<= {_SCHEMA_MAX_TEMPORAL_SEGMENTS} for the configured multimodal schema"
                )


def _declared_mime(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _looks_like_image(request: ConversionInput) -> bool:
    return (
        _declared_mime(request.content_type) in _IMAGE_MIMES
        or Path(request.filename).suffix.lower() in _IMAGE_EXTENSIONS
    )


def _looks_like_video(request: ConversionInput) -> bool:
    return (
        _declared_mime(request.content_type) in _VIDEO_MIMES
        or Path(request.filename).suffix.lower() in _VIDEO_EXTENSIONS
    )


class OpenAIMultimodalParser(FileParser):
    """Convert supported media to evidence-grounded canonical Markdown.

    Despite the stable public parser name, the provider is injected.  Tests and
    offline deployments can therefore exercise the full parser seam without a
    network request, while production can use the isolated Responses transport.
    """

    def __init__(
        self,
        provider: MultimodalDescriptionProvider,
        config: MultimodalParserConfig | None = None,
    ) -> None:
        self._provider = provider
        self._config = config or MultimodalParserConfig()

    @property
    def terminal_processing_failures(self) -> bool:
        return True

    def name(self) -> str:
        return "openai_multimodal"

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        image_supported = (
            _declared_mime(content_type) in _IMAGE_MIMES or Path(filename).suffix.lower() in _IMAGE_EXTENSIONS
        )
        video_supported = self._config.video is not None and (
            _declared_mime(content_type) in _VIDEO_MIMES or Path(filename).suffix.lower() in _VIDEO_EXTENSIONS
        )
        return image_supported or video_supported

    async def convert(self, file_data: bytes, filename: str) -> str:
        output = await self.convert_input(ConversionInput(file_data=file_data, filename=filename))
        return output.content

    async def close(self) -> None:
        await self._provider.close()

    def pipeline_fingerprint(self) -> str:
        """Return the complete non-secret descriptor cache fingerprint."""

        identity = {
            "canonical_atom_max_chars": DEFAULT_CANONICAL_ATOM_MAX_CHARS,
            "canonical_chunk_contract": CANONICAL_CHUNK_CONTRACT,
            "image_normalization": asdict(self._config.image),
            "image_runtime": image_normalizer_identity(),
            "model_behavior_version": self._config.model_behavior_version,
            "pipeline_version": self._config.pipeline_version,
            "prompt_version": self._config.prompt_version,
            "provider": self._provider.pipeline_identity(),
            "sampling_version": self._config.sampling_version,
            "schema_version": self._config.schema_version,
            "video_processing": asdict(self._config.video) if self._config.video is not None else None,
            "video_runtime": video_decoder_identity() if self._config.video is not None else None,
            "video_max_frames_per_call": self._config.max_frames_per_call,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _provider_budget(self, evidence: list[VisualEvidence], *, logical_calls: int):
        identity = self._provider.pipeline_identity()
        return estimate_provider_budget(
            evidence,
            logical_calls=logical_calls,
            max_retries=int(identity.get("max_retries", 0)),
            max_schema_repairs=int(identity.get("max_schema_repairs", 0)),
            max_output_tokens=int(identity.get("max_output_tokens", 0)),
        )

    @staticmethod
    def _statement_atom(value: Any) -> str:
        """Canonical closed-world identity for one evidence-grounded text atom.

        The reducer is allowed to select, deduplicate, and reorder map-stage
        atoms, but it cannot introduce a paraphrase that merely reuses a valid
        evidence ID.  Exact canonical atoms make that boundary mechanically
        auditable instead of relying on another model to judge entailment.
        """

        return json.dumps(
            {
                "text": value.text,
                "evidence_ids": sorted(value.evidence_ids),
                "uncertainty": value.uncertainty,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def _validate_closed_world_reducer(
        cls,
        mapped_segments: list[Any],
        reduced: Any,
    ) -> None:
        """Reject reducer semantics that are not exact map-stage atoms."""

        source_summary_atoms: set[str] = set()
        source_observation_atoms: set[str] = set()
        source_visible_text_atoms: set[str] = set()
        for segment in mapped_segments:
            source_summary_atoms.update(cls._statement_atom(item) for item in segment.summary)
            source_summary_atoms.update(cls._statement_atom(item) for item in segment.observations)
            source_summary_atoms.update(cls._statement_atom(item) for item in segment.visible_text)
            source_observation_atoms.update(
                json.dumps(
                    {
                        "statement": cls._statement_atom(item),
                        "kind": item.kind,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                for item in segment.observations
            )
            source_visible_text_atoms.update(cls._statement_atom(item) for item in segment.visible_text)

        if any(cls._statement_atom(item) not in source_summary_atoms for item in reduced.summary):
            raise GroundingError(
                "grounding.reducer_unproven_statement",
                "Video reducer introduced a summary atom absent from validated map output",
            )
        if any(
            json.dumps(
                {"statement": cls._statement_atom(item), "kind": item.kind},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            not in source_observation_atoms
            for item in reduced.observations
        ):
            raise GroundingError(
                "grounding.reducer_unproven_observation",
                "Video reducer introduced an observation absent from validated map output",
            )
        if any(cls._statement_atom(item) not in source_visible_text_atoms for item in reduced.visible_text):
            raise GroundingError(
                "grounding.reducer_unproven_visible_text",
                "Video reducer introduced visible text absent from validated map output",
            )
        if reduced.entities or reduced.limitations:
            # ModelTemporalSegment currently has no entity/limitation atoms.
            # Accepting either at reduce time would therefore have no map-stage
            # provenance.  A future schema may add typed source atom IDs.
            raise GroundingError(
                "grounding.reducer_unproven_semantics",
                "Video reducer introduced semantics unavailable in the map schema",
            )

    @staticmethod
    async def _before_provider_call(request: ConversionInput) -> None:
        """Cross the durable provider-start boundary immediately before I/O."""

        if request.before_provider_call is not None:
            await request.before_provider_call()

    async def convert_input(self, request: ConversionInput) -> ParserOutput:
        from hms_api.metrics import get_metrics_collector

        metrics = get_metrics_collector()
        pipeline_started = time.monotonic()
        validation_started = pipeline_started
        result = None
        try:
            detect_image_mime(request.file_data)
        except MediaValidationError as exc:
            if exc.code == "media.unsupported_image" and not _looks_like_image(request):
                return await self._convert_possible_video(request, image_error=exc)
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="validation",
                duration=time.monotonic() - validation_started,
                success=False,
                reason=exc.code,
                asset_outcome="rejected",
            )
            raise ParserProcessingError(exc.code, "Uploaded image failed local validation") from None

        metrics.record_multimodal_pipeline(
            media_kind="image",
            stage="validation",
            duration=time.monotonic() - validation_started,
            success=True,
        )

        normalization_started = time.monotonic()
        try:
            normalized_image = normalize_image(
                file_data=request.file_data,
                filename=request.filename,
                declared_mime=request.content_type,
                config=self._config.image,
                asset_id=request.asset_id,
            )
        except MediaValidationError as exc:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="normalize",
                duration=time.monotonic() - normalization_started,
                success=False,
                reason=exc.code,
                asset_outcome="rejected",
            )
            raise ParserProcessingError(exc.code, "Uploaded image failed local validation") from None

        metrics.record_multimodal_pipeline(
            media_kind="image",
            stage="normalize",
            duration=time.monotonic() - normalization_started,
            success=True,
            selected_frames=1,
        )
        metrics.record_multimodal_pipeline(
            media_kind="image",
            stage="preprocess",
            duration=time.monotonic() - pipeline_started,
            success=True,
        )

        if request.asset_sha256 and request.asset_sha256 != normalized_image.asset.sha256:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="normalize",
                duration=time.monotonic() - normalization_started,
                success=False,
                reason="media.asset_hash_mismatch",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "media.asset_hash_mismatch", "Uploaded asset identity changed before conversion"
            )

        describe_started = time.monotonic()
        capabilities = self._provider.capabilities()
        if not capabilities.image_input or not capabilities.structured_outputs:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                reason="provider.capability_unavailable",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "provider.capability_unavailable",
                "Configured provider lacks required image or structured-output capability",
            )
        if normalized_image.evidence.mime_type not in capabilities.accepted_image_mimes:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                reason="provider.image_mime_unsupported",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "provider.image_mime_unsupported",
                "Configured provider does not accept the normalized image type",
            )

        try:
            provider_budget = self._provider_budget([normalized_image.evidence], logical_calls=1)
            await self._before_provider_call(request)
            with metrics.record_multimodal_in_flight(media_kind="image", stage="describe"):
                result = await self._provider.describe_image(normalized_image.evidence)
            provenance = SystemProvenance(
                provider=result.provider,
                configured_model=result.configured_model,
                resolved_model=result.resolved_model,
                pipeline_version=self._config.pipeline_version,
                prompt_version=self._config.prompt_version,
                schema_version=self._config.schema_version,
                sampling_version=self._config.sampling_version,
                pipeline_fingerprint=self.pipeline_fingerprint(),
                provider_request_id=result.request_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                logical_calls=result.logical_calls,
                physical_attempts=result.physical_attempts,
            )
            normalized = normalize_description(
                media_kind="image",
                asset=normalized_image.asset,
                description=result.value,
                evidence=[normalized_image.evidence],
                segment_windows=[],
                provenance=provenance,
            )
        except asyncio.CancelledError:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                reason="operation.cancelled",
                cancelled=True,
                asset_outcome="rejected",
            )
            raise
        except ParserProcessingError as exc:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                reason=exc.code,
                cancelled=exc.code == "operation.cancelled",
                asset_outcome="rejected",
            )
            raise
        except MultimodalError as exc:
            logical_calls = exc.logical_calls or (result.logical_calls if result is not None else 0)
            physical_attempts = exc.physical_attempts or (result.physical_attempts if result is not None else 0)
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                logical_calls=logical_calls,
                physical_attempts=physical_attempts,
                reason=exc.code,
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                exc.code,
                "Multimodal description failed",
                retryable=exc.retryable,
                logical_calls=logical_calls,
                physical_attempts=physical_attempts,
            ) from None
        except Exception:
            metrics.record_multimodal_pipeline(
                media_kind="image",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=1,
                reason="multimodal.processing_failed",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "multimodal.processing_failed",
                "Multimodal description failed unexpectedly",
            ) from None

        metrics.record_multimodal_pipeline(
            media_kind="image",
            stage="describe",
            duration=time.monotonic() - describe_started,
            success=True,
            frames=1,
            logical_calls=result.logical_calls,
            physical_attempts=result.physical_attempts,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        metrics.record_multimodal_pipeline(
            media_kind="image",
            stage="complete",
            duration=time.monotonic() - pipeline_started,
            success=True,
            asset_outcome="accepted",
        )

        entities = [
            {"text": entity.name, "type": "CONCEPT"}
            for entity in normalized.model_output.entities
            if entity.uncertainty == "low"
        ]
        return ParserOutput(
            content=render_canonical_markdown(normalized),
            metadata=flat_provenance_metadata(normalized, source_available=request.source_available),
            entities=entities,
            retain_extraction_mode="chunks",
            pipeline_metadata={
                "asset_id": normalized.asset.asset_id,
                "asset_sha256": normalized.asset.sha256,
                "media_kind": normalized.media_kind,
                "pipeline_version": self._config.pipeline_version,
                "descriptor_model": normalized.provenance.configured_model,
                "resolved_model": normalized.provenance.resolved_model,
                "stage": "normalized",
                "provider_request_id": normalized.provenance.provider_request_id,
                "input_tokens": normalized.provenance.input_tokens,
                "output_tokens": normalized.provenance.output_tokens,
                "logical_calls": normalized.provenance.logical_calls,
                "physical_attempts": normalized.provenance.physical_attempts,
                "physical_attempts_upper_bound": provider_budget.physical_attempts_upper_bound,
                "output_tokens_upper_bound": provider_budget.output_tokens_upper_bound,
                "estimated_image_transport_bytes": provider_budget.estimated_image_transport_bytes,
            },
        )

    async def _convert_possible_video(
        self,
        request: ConversionInput,
        *,
        image_error: MediaValidationError,
    ) -> ParserOutput:
        from hms_api.metrics import get_metrics_collector

        metrics = get_metrics_collector()
        pipeline_started = time.monotonic()
        validation_started = pipeline_started
        try:
            detect_video_magic(request.file_data)
        except MediaValidationError as video_magic_error:
            if _looks_like_video(request):
                metrics.record_multimodal_pipeline(
                    media_kind="video",
                    stage="validation",
                    duration=time.monotonic() - validation_started,
                    success=False,
                    reason=video_magic_error.code,
                    asset_outcome="rejected",
                )
                raise ParserProcessingError(
                    video_magic_error.code,
                    "Uploaded video failed local validation",
                ) from None
            raise ParserNotApplicableError("Input is not supported visual media") from None

        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="validation",
            duration=time.monotonic() - validation_started,
            success=True,
        )

        if self._config.video is None:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="decode",
                duration=0.0,
                success=False,
                reason="media.video_disabled",
                asset_outcome="rejected",
            )
            raise ParserProcessingError("media.video_disabled", "Video ingestion is not enabled on this deployment")

        decode_started = time.monotonic()
        try:
            normalized_video = decode_and_sample_video(
                file_data=request.file_data,
                filename=request.filename,
                declared_mime=request.content_type,
                config=self._config.video,
                asset_id=request.asset_id,
            )
        except MultimodalError as exc:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="decode",
                duration=time.monotonic() - decode_started,
                success=False,
                reason=exc.code,
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                exc.code, "Uploaded video failed local processing", retryable=exc.retryable
            ) from None

        preprocess_duration = time.monotonic() - decode_started
        timings = getattr(normalized_video, "timings", None)
        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="decode",
            duration=float(getattr(timings, "decode_seconds", preprocess_duration)),
            success=True,
        )
        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="normalize",
            duration=float(getattr(timings, "normalization_seconds", 0.0)),
            success=True,
        )
        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="sample",
            duration=float(getattr(timings, "sample_seconds", 0.0)),
            success=True,
            candidate_frames=getattr(getattr(normalized_video, "sampling", None), "candidate_count", 0),
            selected_frames=len(normalized_video.evidence),
        )

        if request.asset_sha256 and request.asset_sha256 != normalized_video.asset.sha256:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="normalize",
                duration=preprocess_duration,
                success=False,
                reason="media.asset_hash_mismatch",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "media.asset_hash_mismatch", "Uploaded asset identity changed before conversion"
            )
        if not normalized_video.evidence:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="sample",
                duration=preprocess_duration,
                success=False,
                reason="media.video_no_evidence",
                asset_outcome="rejected",
            )
            raise ParserProcessingError("media.video_no_evidence", "Video sampling produced no visual evidence")

        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="preprocess",
            duration=preprocess_duration,
            success=True,
        )

        describe_started = time.monotonic()
        capabilities = self._provider.capabilities()
        if not capabilities.image_input or not capabilities.structured_outputs:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=len(normalized_video.evidence),
                reason="provider.capability_unavailable",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "provider.capability_unavailable",
                "Configured provider lacks required image or structured-output capability",
            )

        evidence = list(normalized_video.evidence)
        map_calls = (len(evidence) + self._config.max_frames_per_call - 1) // self._config.max_frames_per_call
        provider_budget = self._provider_budget(evidence, logical_calls=map_calls + 1)
        segment_results = []
        segment_windows = []
        provider_results = []
        current_logical_calls = 0
        current_physical_attempts = 0
        current_input_tokens = 0
        current_output_tokens = 0
        reused_segment_checkpoints = 0
        try:
            for segment_index, offset in enumerate(range(0, len(evidence), self._config.max_frames_per_call)):
                segment_evidence = evidence[offset : offset + self._config.max_frames_per_call]
                segment_id = f"segment-{segment_index:03d}"
                segment_identity = derive_video_segment_identity(segment_id, segment_evidence)
                cached_segment = None
                if request.load_video_segment_checkpoint is not None:
                    cached_segment = await request.load_video_segment_checkpoint(segment_identity)
                if cached_segment is not None:
                    if (
                        cached_segment.segment_key != segment_identity.segment_key
                        or cached_segment.segment_id != segment_identity.segment_id
                        or cached_segment.evidence_fingerprint != segment_identity.evidence_fingerprint
                    ):
                        raise GroundingError(
                            "grounding.segment_checkpoint_identity",
                            "Durable segment checkpoint does not match its system evidence window",
                        )
                    segment_result = cached_segment.to_provider_result()
                    reused_segment_checkpoints += 1
                else:
                    await self._before_provider_call(request)
                    with metrics.record_multimodal_in_flight(media_kind="video", stage="describe"):
                        segment_result = await self._provider.describe_video_segment(segment_id, segment_evidence)
                    current_logical_calls += segment_result.logical_calls
                    current_physical_attempts += segment_result.physical_attempts
                    current_input_tokens += segment_result.input_tokens
                    current_output_tokens += segment_result.output_tokens
                self._validate_mapped_segment(segment_id, segment_evidence, segment_result.value)
                if cached_segment is None and request.save_video_segment_checkpoint is not None:
                    await request.save_video_segment_checkpoint(
                        VideoSegmentCheckpoint.from_provider_result(segment_identity, segment_result)
                    )
                segment_results.append(segment_result.value)
                provider_results.append(segment_result)
                timestamps = [item.timestamp_ms for item in segment_evidence]
                if any(timestamp is None for timestamp in timestamps):
                    raise GroundingError("grounding.video_timestamp", "Video evidence lacks a system timestamp")
                segment_windows.append(
                    SegmentWindow(
                        segment_id=segment_id,
                        start_ms=min(timestamps),
                        end_ms=max(timestamps),
                        evidence_ids=[item.evidence_id for item in segment_evidence],
                    )
                )

            await self._before_provider_call(request)
            with metrics.record_multimodal_in_flight(media_kind="video", stage="describe"):
                reduce_result = await self._provider.reduce_video(segment_results)
            current_logical_calls += reduce_result.logical_calls
            current_physical_attempts += reduce_result.physical_attempts
            current_input_tokens += reduce_result.input_tokens
            current_output_tokens += reduce_result.output_tokens
            provider_results.append(reduce_result)
            if reduce_result.value.temporal_segments != segment_results:
                raise GroundingError(
                    "grounding.reducer_segment_mutation",
                    "Video reducer changed a validated segment",
                )
            self._validate_closed_world_reducer(segment_results, reduce_result.value)

            first_result = provider_results[0]
            if any(
                item.provider != first_result.provider or item.configured_model != first_result.configured_model
                for item in provider_results
            ):
                raise GroundingError(
                    "grounding.provider_identity_changed", "Provider identity changed within one video"
                )
            resolved_models = {item.resolved_model for item in provider_results if item.resolved_model}
            provenance = SystemProvenance(
                provider=first_result.provider,
                configured_model=first_result.configured_model,
                resolved_model=next(iter(resolved_models)) if len(resolved_models) == 1 else None,
                pipeline_version=self._config.pipeline_version,
                prompt_version=self._config.prompt_version,
                schema_version=self._config.schema_version,
                sampling_version=self._config.sampling_version,
                pipeline_fingerprint=self.pipeline_fingerprint(),
                provider_request_id=reduce_result.request_id,
                input_tokens=sum(item.input_tokens for item in provider_results),
                output_tokens=sum(item.output_tokens for item in provider_results),
                logical_calls=sum(item.logical_calls for item in provider_results),
                physical_attempts=sum(item.physical_attempts for item in provider_results),
            )
            normalized = normalize_description(
                media_kind="video",
                asset=normalized_video.asset,
                description=reduce_result.value,
                evidence=evidence,
                segment_windows=segment_windows,
                provenance=provenance,
            )
        except asyncio.CancelledError:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=len(evidence),
                logical_calls=current_logical_calls,
                physical_attempts=current_physical_attempts,
                input_tokens=current_input_tokens,
                output_tokens=current_output_tokens,
                deduplicated=reused_segment_checkpoints > 0,
                reason="operation.cancelled",
                cancelled=True,
                asset_outcome="rejected",
            )
            raise
        except ParserProcessingError as exc:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=len(evidence),
                logical_calls=current_logical_calls + exc.logical_calls,
                physical_attempts=current_physical_attempts + exc.physical_attempts,
                input_tokens=current_input_tokens,
                output_tokens=current_output_tokens,
                deduplicated=reused_segment_checkpoints > 0,
                reason=exc.code,
                cancelled=exc.code == "operation.cancelled",
                asset_outcome="rejected",
            )
            raise
        except MultimodalError as exc:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=len(evidence),
                logical_calls=current_logical_calls + exc.logical_calls,
                physical_attempts=current_physical_attempts + exc.physical_attempts,
                input_tokens=current_input_tokens,
                output_tokens=current_output_tokens,
                deduplicated=reused_segment_checkpoints > 0,
                reason=exc.code,
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                exc.code,
                "Multimodal video description failed",
                retryable=exc.retryable,
                logical_calls=current_logical_calls + exc.logical_calls,
                physical_attempts=current_physical_attempts + exc.physical_attempts,
            ) from None
        except Exception:
            metrics.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=time.monotonic() - describe_started,
                success=False,
                frames=len(evidence),
                logical_calls=current_logical_calls,
                physical_attempts=current_physical_attempts,
                input_tokens=current_input_tokens,
                output_tokens=current_output_tokens,
                deduplicated=reused_segment_checkpoints > 0,
                reason="multimodal.video_processing_failed",
                asset_outcome="rejected",
            )
            raise ParserProcessingError(
                "multimodal.video_processing_failed",
                "Multimodal video description failed unexpectedly",
            ) from None

        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="describe",
            duration=time.monotonic() - describe_started,
            success=True,
            frames=len(evidence),
            logical_calls=current_logical_calls,
            physical_attempts=current_physical_attempts,
            input_tokens=current_input_tokens,
            output_tokens=current_output_tokens,
            deduplicated=reused_segment_checkpoints > 0,
        )
        metrics.record_multimodal_pipeline(
            media_kind="video",
            stage="complete",
            duration=time.monotonic() - pipeline_started,
            success=True,
            asset_outcome="accepted",
        )

        entities = [
            {"text": entity.name, "type": "CONCEPT"}
            for entity in normalized.model_output.entities
            if entity.uncertainty == "low"
        ]
        return ParserOutput(
            content=render_canonical_markdown(normalized),
            metadata=flat_provenance_metadata(normalized, source_available=request.source_available),
            entities=entities,
            retain_extraction_mode="chunks",
            pipeline_metadata={
                "asset_id": normalized.asset.asset_id,
                "asset_sha256": normalized.asset.sha256,
                "media_kind": normalized.media_kind,
                "pipeline_version": self._config.pipeline_version,
                "descriptor_model": normalized.provenance.configured_model,
                "resolved_model": normalized.provenance.resolved_model,
                "stage": "normalized",
                "provider_request_id": normalized.provenance.provider_request_id,
                "input_tokens": normalized.provenance.input_tokens,
                "output_tokens": normalized.provenance.output_tokens,
                "logical_calls": normalized.provenance.logical_calls,
                "physical_attempts": normalized.provenance.physical_attempts,
                "segment_checkpoint_hits": reused_segment_checkpoints,
                "physical_attempts_upper_bound": provider_budget.physical_attempts_upper_bound,
                "output_tokens_upper_bound": provider_budget.output_tokens_upper_bound,
                "estimated_image_transport_bytes": provider_budget.estimated_image_transport_bytes,
            },
        )

    @staticmethod
    def _validate_mapped_segment(segment_id: str, evidence: list[VisualEvidence], segment) -> None:
        if segment.segment_id != segment_id:
            raise GroundingError("grounding.unknown_segment", "Provider returned the wrong system segment ID")
        known_evidence = {item.evidence_id for item in evidence}
        if not set(segment.evidence_ids).issubset(known_evidence):
            raise GroundingError(
                "grounding.segment_evidence_mismatch",
                "Segment output referenced evidence outside its system window",
            )
        statements = [*segment.summary, *segment.observations, *segment.visible_text]
        if any(not set(statement.evidence_ids).issubset(known_evidence) for statement in statements):
            raise GroundingError(
                "grounding.segment_statement_mismatch",
                "Segment statement referenced evidence outside its system window",
            )


def create_openai_multimodal_parser(config: Any) -> OpenAIMultimodalParser | None:
    """Build the opt-in production parser from server-static configuration."""

    if getattr(config, "multimodal_enabled", False) is not True:
        return None
    if getattr(config, "multimodal_image_enabled", False) is not True:
        return None
    required_capabilities = (
        "multimodal_capability_responses_api",
        "multimodal_capability_image_input",
        "multimodal_capability_structured_outputs",
    )
    if not all(getattr(config, name, False) is True for name in required_capabilities):
        raise ValueError("Multimodal parser requires Responses, image-input, and structured-output capabilities")

    api_key = getattr(config, "multimodal_api_key", None)
    if not api_key:
        raise ValueError("Multimodal parser requires an independently configured API key")

    provider = OpenAIResponsesMultimodalProvider(
        OpenAIProviderConfig(
            api_key=api_key,
            base_url=config.multimodal_base_url,
            model=config.multimodal_model,
            image_detail=config.multimodal_image_detail,
            timeout_seconds=config.multimodal_request_timeout_seconds,
            max_output_tokens=config.multimodal_max_output_tokens,
            max_retries=config.multimodal_max_retries,
            max_schema_repairs=config.multimodal_max_schema_repairs,
            max_concurrency=config.multimodal_max_concurrency,
        )
    )
    parser_config = MultimodalParserConfig(
        image=ImageNormalizationConfig(
            max_bytes=config.multimodal_max_image_bytes,
            max_pixels=config.multimodal_max_image_pixels,
        ),
        video=(
            VideoProcessingConfig(
                max_bytes=config.multimodal_max_video_bytes,
                max_duration_seconds=config.multimodal_max_video_duration_seconds,
                probe_interval_seconds=config.multimodal_video_probe_interval_seconds,
                max_frames=config.multimodal_video_max_frames,
                coverage_ratio=config.multimodal_video_coverage_ratio,
                sampling_version=config.multimodal_sampling_version,
            )
            if getattr(config, "multimodal_video_enabled", False) is True
            else None
        ),
        max_frames_per_call=config.multimodal_max_frames_per_call,
        pipeline_version=getattr(config, "multimodal_pipeline_version", "hms-multimodal-v1"),
        model_behavior_version=config.multimodal_model_behavior_version,
        prompt_version=config.multimodal_prompt_version,
        schema_version=config.multimodal_schema_version,
        sampling_version=config.multimodal_sampling_version,
    )
    return OpenAIMultimodalParser(provider, parser_config)
