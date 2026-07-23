"""Focused multimodal privacy, failure-boundary, and cancellation tests.

The sentinels in this module are generated at runtime.  No base64 media fixture
or golden request body is committed to the repository.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import hms_api.engine.memory_engine as memory_engine_module
import hms_api.engine.parsers.openai_multimodal as parser_module
from hms_api.engine.memory_engine import MemoryEngine
from hms_api.engine.multimodal import (
    GroundedStatement,
    MediaAsset,
    ModelMultimodalDescription,
    ModelTemporalSegment,
    MultimodalCapabilities,
    NormalizedImage,
    OpenAIProviderConfig,
    OpenAIResponsesMultimodalProvider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderIncompleteError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
    VideoProcessingConfig,
    VisualEvidence,
    video_decoder_available,
)
from hms_api.engine.parsers import (
    ConversionInput,
    FileParserRegistry,
    MultimodalParserConfig,
    OpenAIMultimodalParser,
)
from hms_api.engine.retain.orchestrator import _RetainLogBuffer
from hms_api.metrics import MetricsCollector

_FRAME_BYTES = b"HMS_FRAME_BYTES_SENTINEL_ignore_previous_instructions"
_FRAME_BASE64 = base64.b64encode(_FRAME_BYTES).decode("ascii")
_DATA_URL = f"data:image/png;base64,{_FRAME_BASE64}"
_SOURCE_BYTES = b"HMS_RAW_ASSET_BYTES_SENTINEL"
_API_KEY = "HMS_MULTIMODAL_API_KEY_SENTINEL"
_PROVIDER_ECHO = f"provider-echo::{_DATA_URL}"


def _evidence(*, evidence_id: str = "image-000-security", timestamp_ms: int | None = None) -> VisualEvidence:
    return VisualEvidence(
        evidence_id=evidence_id,
        timestamp_ms=timestamp_ms,
        sha256=hashlib.sha256(_FRAME_BYTES).hexdigest(),
        mime_type="image/png",
        width=16,
        height=8,
        encoded_bytes=_FRAME_BYTES,
    )


def _image_normalization() -> NormalizedImage:
    evidence = _evidence()
    return NormalizedImage(
        asset=MediaAsset(
            asset_id="asset-security",
            sha256=hashlib.sha256(_SOURCE_BYTES).hexdigest(),
            media_kind="image",
            detected_mime="image/png",
            original_filename="security.png",
            byte_size=len(_SOURCE_BYTES),
            width=evidence.width,
            height=evidence.height,
            duration_ms=None,
            audio_presence="absent",
            audio_processing="not_requested",
        ),
        evidence=evidence,
    )


def _description(evidence_id: str = "image-000-security") -> ModelMultimodalDescription:
    return ModelMultimodalDescription(
        summary=[
            GroundedStatement(
                text="A synthetic editor pane is visible.",
                evidence_ids=[evidence_id],
                uncertainty="low",
            )
        ],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[],
        limitations=[],
    )


def _success_response() -> dict:
    return {
        "id": "resp-security",
        "model": "gpt-5-mini-test-revision",
        "status": "completed",
        "output_text": _description().model_dump_json(),
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _provider_config(**overrides) -> OpenAIProviderConfig:
    values = {
        "api_key": _API_KEY,
        "base_url": "https://api.openai.test/v1",
        "model": "gpt-5-mini",
        "initial_backoff_seconds": 0,
        "max_retries": 0,
        "max_schema_repairs": 0,
    }
    values.update(overrides)
    return OpenAIProviderConfig(**values)


class _AsyncContext:
    def __init__(self, value=None) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc_value, tb):
        return False


class _Connection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []

    def transaction(self):
        return _AsyncContext()

    async def fetchrow(self, _sql, *_args):
        return {"status": "processing"}

    async def execute(self, sql, *args):
        self.executions.append((sql, args))
        return "UPDATE 1"


class _Storage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def retrieve(self, key: str) -> bytes:
        assert key == "immutable/security/source"
        return _SOURCE_BYTES

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


class _TaskBackend:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def submit_task(self, payload: dict) -> None:
        self.payloads.append(payload)


class _AuditLogger:
    def __init__(self) -> None:
        self.entries = []

    def is_enabled(self, _action: str) -> bool:
        return True

    def log_fire_and_forget(self, entry) -> None:
        self.entries.append(entry)


class _Validator:
    def __init__(self) -> None:
        self.file_results = []

    async def on_file_convert_complete(self, result) -> None:
        self.file_results.append(result)


class _RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_multimodal_pipeline(self, **event) -> None:
        self.events.append(event)

    @contextmanager
    def record_multimodal_in_flight(self, **_attributes):
        yield


class _SpanRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_llm_call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _WebhookRecorder:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def fire_event_with_conn(self, event, _conn, *, schema: str | None = None) -> None:
        self.payloads.append(
            {
                "schema": schema,
                "event": event.model_dump(mode="json"),
            }
        )


def _task() -> dict:
    return {
        "type": "file_convert_retain",
        "bank_id": "bank-security",
        "storage_key": "immutable/security/source",
        "document_id": "doc-security",
        "operation_id": "00000000-0000-0000-0000-000000000101",
        "original_filename": "security.png",
        "content_type": "image/png",
        "asset_id": "asset-security",
        "asset_sha256": hashlib.sha256(_SOURCE_BYTES).hexdigest(),
        "parser": ["openai_multimodal"],
        "context": "synthetic security test",
        "metadata": {"customer_key": "preserved"},
        "tags": ["security"],
        "document_tags": ["synthetic"],
        "timestamp": "unset",
        "_tenant_id": "tenant-security",
    }


def _engine(parser: OpenAIMultimodalParser, connection: _Connection) -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._file_storage = _Storage()
    registry = FileParserRegistry()
    registry.register(parser)
    engine._parser_registry = registry
    engine._operation_validator = _Validator()
    engine._task_backend = _TaskBackend()
    engine._audit_logger = _AuditLogger()

    async def get_backend():
        return SimpleNamespace()

    engine._get_backend = get_backend
    return engine


def _assert_sentinels_absent(surface: object) -> None:
    serialized = json.dumps(surface, default=str, sort_keys=True)
    rendered = f"{serialized}\n{surface!r}"
    for sentinel in (_FRAME_BYTES.decode(), _FRAME_BASE64, _DATA_URL, _SOURCE_BYTES.decode(), _API_KEY):
        assert sentinel not in rendered


def test_retain_log_buffer_redacts_identifiers_only_when_enabled() -> None:
    bank_id = "bank-private-security-sentinel"
    document_id = "document-private-security-sentinel"

    sanitized = _RetainLogBuffer(sanitized=True, secrets=(bank_id,))
    sanitized.append(f"bank={bank_id}")
    sanitized.add_secret(document_id)
    sanitized.append(f"bank={bank_id} document={document_id}")

    sanitized_output = "\n".join(sanitized)
    assert bank_id not in sanitized_output
    assert document_id not in sanitized_output
    assert sanitized_output.count("<redacted>") == 3

    legacy = _RetainLogBuffer(sanitized=False, secrets=(bank_id,))
    legacy.add_secret(document_id)
    legacy.append(f"bank={bank_id} document={document_id}")

    assert legacy == [f"bank={bank_id} document={document_id}"]


@pytest.mark.asyncio
async def test_transport_base64_is_ephemeral_across_actual_observable_surfaces(monkeypatch, caplog) -> None:
    """The one allowed data URL is redacted at the HTTP boundary and never escapes."""

    transport_checked = False
    redacted_wire_shape: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_checked, redacted_wire_shape
        payload = json.loads(request.content)
        image_part = payload["input"][0]["content"][1]
        assert image_part["image_url"] == _DATA_URL
        assert _FRAME_BYTES.decode() not in payload["input"][0]["content"][0]["text"]
        assert "untrusted data" in payload["input"][0]["content"][0]["text"]
        transport_checked = True

        # The test transport keeps only the same redacted request shape that a
        # safe request-shape snapshot is allowed to retain.
        image_part["image_url"] = "<redacted-data-url>"
        redacted_wire_shape = payload
        return httpx.Response(200, headers={"x-request-id": "req-security"}, json=_success_response())

    monkeypatch.setattr(parser_module, "normalize_image", lambda **_kwargs: _image_normalization())
    monkeypatch.setattr(parser_module, "detect_image_mime", lambda _data: "image/png")
    metrics = _RecordingMetrics()
    monkeypatch.setattr("hms_api.metrics.get_metrics_collector", lambda: metrics)

    import hms_api.config
    import hms_api.tracing as tracing_module

    config = replace(hms_api.config._get_raw_config(), file_delete_after_retain=False)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    span_recorder = _SpanRecorder()
    monkeypatch.setattr(tracing_module, "_span_recorder", span_recorder)

    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    caplog.set_level(logging.DEBUG)

    provider_config = _provider_config()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=provider_config.base_url + "/"
    ) as client:
        provider = OpenAIResponsesMultimodalProvider(provider_config, client=client)
        parser = OpenAIMultimodalParser(provider)
        engine = _engine(parser, connection)
        task_payload = _task()
        await MemoryEngine.execute_task(engine, task_payload)

    assert transport_checked is True
    assert redacted_wire_shape["input"][0]["content"][1]["image_url"] == "<redacted-data-url>"
    assert len(engine._task_backend.payloads) == 1
    [child_payload] = engine._task_backend.payloads
    assert child_payload["type"] == "batch_retain"
    assert child_payload["_retain_extraction_mode"] == "chunks"
    assert all(payload["type"] != "webhook_delivery" for payload in engine._task_backend.payloads)

    # Exercise the real retain webhook event builder.  Conversion does not
    # enqueue retain.completed itself; the child retain transaction invokes
    # this callback after publication.  Capturing the typed event here proves
    # that the webhook payload surface contains identifiers/tags only, never
    # canonical text, metadata, source bytes, or provider transport data.
    webhook_recorder = _WebhookRecorder()
    engine._webhook_manager = webhook_recorder
    webhook_callback = engine._build_retain_outbox_callback(
        "bank-security",
        child_payload["contents"],
        child_payload["operation_id"],
        schema="tenant-security",
    )
    assert webhook_callback is not None
    await webhook_callback(connection)
    assert len(webhook_recorder.payloads) == 1
    assert set(webhook_recorder.payloads[0]["event"]["data"]) == {"document_id", "tags"}

    [audit_entry] = engine._audit_logger.entries
    [hook_result] = engine._operation_validator.file_results

    evidence = _image_normalization().evidence
    conversion_input = ConversionInput(file_data=_SOURCE_BYTES, filename="security.png", content_type="image/png")
    assert "encoded_bytes" not in evidence.model_dump()
    assert "encoded_bytes" not in evidence.model_dump_json()
    _assert_sentinels_absent(repr(evidence))
    _assert_sentinels_absent(repr(conversion_input))
    _assert_sentinels_absent(repr(provider_config))

    # These are the real serialization seams feeding operations, audit, parser
    # hooks, documents/content_hash, chunks, and downstream memory metadata.
    observable_surfaces = {
        "redacted_wire_shape": redacted_wire_shape,
        "operation_sql_arguments": connection.executions,
        "child_task_payload": child_payload,
        "document_and_chunk_input": child_payload["contents"],
        "audit": asdict(audit_entry),
        "parser_hook": asdict(hook_result),
        "logs": [record.getMessage() for record in caplog.records],
        "traces": span_recorder.calls,
        "metrics": metrics.events,
        "webhook_payloads": webhook_recorder.payloads,
    }
    _assert_sentinels_absent(observable_surfaces)
    assert span_recorder.calls == []  # never route image parts through full-message LLM tracing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_type", "expected_code", "expected_calls"),
    [
        ("timeout", ProviderUnavailableError, "provider.network_unavailable", 2),
        ("authentication", ProviderAuthenticationError, "provider.authentication", 1),
        ("rate_limit", ProviderRateLimitError, "provider.rate_limited", 2),
        ("server", ProviderUnavailableError, "provider.unavailable", 2),
        ("unsupported_model", ProviderError, "provider.unsupported_model", 1),
        ("refusal", ProviderRefusalError, "provider.refusal", 1),
        ("incomplete", ProviderIncompleteError, "provider.incomplete.content_filter", 1),
        ("schema", ProviderSchemaError, "provider.schema_invalid", 2),
    ],
)
async def test_provider_failures_are_bounded_classified_and_traceback_sanitized(
    failure, expected_type, expected_code, expected_calls, caplog
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout(_PROVIDER_ECHO, request=request)
        if failure == "authentication":
            return httpx.Response(401, json={"error": {"message": _PROVIDER_ECHO}})
        if failure == "rate_limit":
            return httpx.Response(429, json={"error": {"message": _PROVIDER_ECHO}})
        if failure == "server":
            return httpx.Response(503, json={"error": {"message": _PROVIDER_ECHO}})
        if failure == "unsupported_model":
            return httpx.Response(
                400,
                json={"error": {"code": "model_not_found", "message": _PROVIDER_ECHO}},
            )
        if failure == "refusal":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "refusal", "refusal": _PROVIDER_ECHO}]}],
                },
            )
        if failure == "incomplete":
            return httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter", "echo": _PROVIDER_ECHO},
                    "output": [],
                },
            )
        return httpx.Response(200, json={"status": "completed", "output_text": _PROVIDER_ECHO})

    caplog.set_level(logging.DEBUG)
    config = _provider_config(max_retries=1, max_schema_repairs=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        with pytest.raises(expected_type) as exc_info:
            await provider.describe_image(_evidence())

    assert calls == expected_calls
    assert exc_info.value.code == expected_code
    assert exc_info.value.logical_calls == 1
    assert exc_info.value.physical_attempts == expected_calls
    formatted_traceback = "".join(
        traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.value.__traceback__)
    )
    _assert_sentinels_absent(
        {
            "exception": str(exc_info.value),
            "traceback": formatted_traceback,
            "logs": [record.getMessage() for record in caplog.records],
        }
    )


def _failure_response(failure: str) -> dict:
    if failure == "refusal":
        return {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": _PROVIDER_ECHO}]}],
        }
    if failure == "incomplete":
        return {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens", "echo": _PROVIDER_ECHO},
            "output": [],
        }
    if failure == "schema":
        return {"status": "completed", "output_text": _PROVIDER_ECHO}
    return {"error": {"message": _PROVIDER_ECHO}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "http_status", "expected_code"),
    [
        ("provider", 503, "provider.unavailable"),
        ("refusal", 200, "provider.refusal"),
        ("incomplete", 200, "provider.incomplete.max_output_tokens"),
        ("schema", 200, "provider.schema_invalid"),
    ],
)
async def test_terminal_description_failure_records_metric_and_never_enqueues_child(
    monkeypatch, caplog, failure, http_status, expected_code
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(http_status, json=_failure_response(failure))

    monkeypatch.setattr(parser_module, "normalize_image", lambda **_kwargs: _image_normalization())
    monkeypatch.setattr(parser_module, "detect_image_mime", lambda _data: "image/png")
    metrics = _RecordingMetrics()
    monkeypatch.setattr("hms_api.metrics.get_metrics_collector", lambda: metrics)

    import hms_api.config

    config = replace(hms_api.config._get_raw_config(), file_delete_after_retain=False)
    monkeypatch.setattr(hms_api.config, "_get_raw_config", lambda: config)
    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    caplog.set_level(logging.DEBUG)

    provider_config = _provider_config()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=provider_config.base_url + "/"
    ) as client:
        parser = OpenAIMultimodalParser(OpenAIResponsesMultimodalProvider(provider_config, client=client))
        engine = _engine(parser, connection)
        with pytest.raises(RuntimeError) as exc_info:
            await MemoryEngine._handle_file_convert_retain(engine, _task())

    assert engine._task_backend.payloads == []
    assert all("INSERT INTO" not in sql for sql, _args in connection.executions)
    failure_updates = [args for sql, args in connection.executions if "SET result_metadata" in sql]
    assert len(failure_updates) == 1
    public_metadata = json.loads(failure_updates[0][1])
    assert public_metadata["multimodal"]["stage"] == "failed"
    assert public_metadata["multimodal"]["sanitized_error_code"] == expected_code
    failure_metric = next(
        event for event in metrics.events if event["stage"] == "describe" and event["success"] is False
    )
    assert failure_metric["logical_calls"] == 1
    assert failure_metric["physical_attempts"] == 1

    formatted_traceback = "".join(
        traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.value.__traceback__)
    )
    _assert_sentinels_absent(
        {
            "operation_updates": connection.executions,
            "task_payloads": engine._task_backend.payloads,
            "exception": str(exc_info.value),
            "traceback": formatted_traceback,
            "logs": [record.getMessage() for record in caplog.records],
            "metrics": metrics.events,
        }
    )


@pytest.mark.asyncio
async def test_cancellation_propagates_to_inflight_transport_and_never_enqueues_child(monkeypatch) -> None:
    request_started = asyncio.Event()
    transport_cancelled = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            transport_cancelled.set()
        raise AssertionError("cancelled request unexpectedly resumed")

    monkeypatch.setattr(parser_module, "normalize_image", lambda **_kwargs: _image_normalization())
    monkeypatch.setattr(parser_module, "detect_image_mime", lambda _data: "image/png")
    metrics = _RecordingMetrics()
    monkeypatch.setattr("hms_api.metrics.get_metrics_collector", lambda: metrics)
    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))

    config = _provider_config()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        parser = OpenAIMultimodalParser(OpenAIResponsesMultimodalProvider(config, client=client))
        engine = _engine(parser, connection)
        conversion = asyncio.create_task(MemoryEngine._handle_file_convert_retain(engine, _task()))
        await asyncio.wait_for(request_started.wait(), timeout=1)
        conversion.cancel()
        with pytest.raises(asyncio.CancelledError):
            await conversion
        await asyncio.wait_for(transport_cancelled.wait(), timeout=1)

    assert engine._task_backend.payloads == []
    assert connection.executions == []
    assert any(event["stage"] == "describe" and event["success"] is False for event in metrics.events)


class _SecondSegmentFailureProvider:
    def __init__(self, failure: ProviderError) -> None:
        self.failure = failure
        self.map_calls = 0
        self.reduce_calls = 0

    def capabilities(self) -> MultimodalCapabilities:
        return MultimodalCapabilities(
            image_input=True,
            video_input=False,
            structured_outputs=True,
            accepted_image_mimes=["image/png"],
            image_detail_levels=["auto"],
        )

    def pipeline_identity(self) -> dict[str, str | int]:
        return {"provider": "failing-fake", "model": "gpt-5-mini"}

    async def describe_image(self, evidence):
        raise AssertionError("video fixture must not use the image path")

    async def describe_video_segment(self, segment_id, evidence):
        self.map_calls += 1
        if self.map_calls == 2:
            raise self.failure
        evidence_ids = [item.evidence_id for item in evidence]
        segment = ModelTemporalSegment(
            segment_id=segment_id,
            summary=[
                GroundedStatement(
                    text="UNCOMMITTED_PARTIAL_SEGMENT",
                    evidence_ids=evidence_ids,
                    uncertainty="low",
                )
            ],
            observations=[],
            visible_text=[],
            evidence_ids=evidence_ids,
        )
        return ProviderResult(
            value=segment,
            provider="failing-fake",
            configured_model="gpt-5-mini",
            resolved_model=None,
            request_id="map-1",
            input_tokens=1,
            output_tokens=1,
            logical_calls=1,
            physical_attempts=1,
            latency_seconds=0,
        )

    async def reduce_video(self, segments):
        self.reduce_calls += 1
        raise AssertionError("reduce must not run after a required map failure")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_logical_calls", "expected_physical_attempts"),
    [
        pytest.param(
            ProviderUnavailableError(
                "provider.network_unavailable",
                "Provider network request failed",
                retryable=True,
                logical_calls=1,
                physical_attempts=3,
            ),
            "provider.network_unavailable",
            2,
            4,
            id="timeout-or-unavailable",
        ),
        pytest.param(
            ProviderRefusalError(
                "provider.refusal",
                "Provider refused the visual description request",
                logical_calls=1,
                physical_attempts=1,
            ),
            "provider.refusal",
            2,
            2,
            id="refusal",
        ),
        pytest.param(
            ProviderIncompleteError(
                "provider.incomplete.max_output_tokens",
                "Provider response was incomplete",
                logical_calls=1,
                physical_attempts=1,
            ),
            "provider.incomplete.max_output_tokens",
            2,
            2,
            id="incomplete",
        ),
        pytest.param(
            ProviderSchemaError(
                "provider.schema_invalid",
                "Provider output failed the multimodal schema",
                logical_calls=1,
                physical_attempts=2,
            ),
            "provider.schema_invalid",
            2,
            3,
            id="schema-invalid-after-repair",
        ),
    ],
)
async def test_second_video_segment_failure_discards_partial_map_and_skips_child(
    monkeypatch,
    caplog,
    failure: ProviderError,
    expected_code: str,
    expected_logical_calls: int,
    expected_physical_attempts: int,
) -> None:
    source_sha = hashlib.sha256(_SOURCE_BYTES).hexdigest()
    evidence = (
        _evidence(evidence_id="frame-000", timestamp_ms=0),
        _evidence(evidence_id="frame-001", timestamp_ms=1000),
    )
    video_asset = MediaAsset(
        asset_id="asset-security",
        sha256=source_sha,
        media_kind="video",
        detected_mime="video/mp4",
        original_filename="security.mp4",
        byte_size=len(_SOURCE_BYTES),
        width=16,
        height=8,
        duration_ms=1000,
        audio_presence="absent",
        audio_processing="not_requested",
    )

    def unsupported_image(**_kwargs):
        from hms_api.engine.multimodal import MediaValidationError

        raise MediaValidationError("media.unsupported_image", "Unsupported image")

    monkeypatch.setattr(parser_module, "normalize_image", unsupported_image)
    monkeypatch.setattr(parser_module, "detect_video_magic", lambda _data: object())
    monkeypatch.setattr(
        parser_module,
        "decode_and_sample_video",
        lambda **_kwargs: SimpleNamespace(asset=video_asset, evidence=evidence),
    )
    metrics = _RecordingMetrics()
    monkeypatch.setattr("hms_api.metrics.get_metrics_collector", lambda: metrics)
    connection = _Connection()
    monkeypatch.setattr(memory_engine_module, "acquire_with_retry", lambda _backend: _AsyncContext(connection))
    caplog.set_level(logging.DEBUG)

    provider = _SecondSegmentFailureProvider(failure)
    parser = OpenAIMultimodalParser(
        provider,
        MultimodalParserConfig(video=VideoProcessingConfig(max_frames=4), max_frames_per_call=1),
    )
    engine = _engine(parser, connection)
    task = _task()
    task.update(
        {
            "original_filename": "security.mp4",
            "content_type": "video/mp4",
            "asset_sha256": source_sha,
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        await MemoryEngine._handle_file_convert_retain(engine, task)

    assert provider.map_calls == 2
    assert provider.reduce_calls == 0
    assert engine._task_backend.payloads == []
    assert "UNCOMMITTED_PARTIAL_SEGMENT" not in str(exc_info.value)
    assert "UNCOMMITTED_PARTIAL_SEGMENT" not in json.dumps(connection.executions, default=str)
    assert "UNCOMMITTED_PARTIAL_SEGMENT" not in "\n".join(record.getMessage() for record in caplog.records)
    failure_event = next(event for event in metrics.events if event["stage"] == "describe" and not event["success"])
    assert failure_event["reason"] == expected_code
    assert failure_event["logical_calls"] == expected_logical_calls
    assert failure_event["physical_attempts"] == expected_physical_attempts


def test_failed_metric_uses_only_bounded_non_media_labels() -> None:
    meter = MagicMock()
    # Return a fresh instrument without coupling this security assertion to
    # the collector's current instrument count.
    meter.create_histogram.side_effect = lambda **_kwargs: MagicMock()
    meter.create_counter.side_effect = lambda **_kwargs: MagicMock()
    meter.create_up_down_counter.side_effect = lambda **_kwargs: MagicMock()
    config = SimpleNamespace(metrics_include_bank_id=False)
    with (
        patch("hms_api.metrics.get_meter", return_value=meter),
        patch("hms_api.config.get_config", return_value=config),
    ):
        collector = MetricsCollector()

    collector.record_multimodal_pipeline(
        media_kind="image",
        stage="describe",
        duration=0.25,
        success=False,
        frames=1,
    )

    attributes = collector.multimodal_stage_duration.record.call_args.args[1]
    assert attributes == {
        "media_kind": "image",
        "stage": "describe",
        "outcome": "failed",
        "reason": "other",
    }
    assert not (
        {"tenant", "schema", "filename", "document_id", "asset_sha256", "bank_id", "data_url"} & attributes.keys()
    )


@pytest.mark.skipif(not video_decoder_available(), reason="install the multimodal-video extra")
def test_video_decoder_uses_bytesio_not_filename_shell_or_tempfile(monkeypatch, tmp_path) -> None:
    import hms_api.engine.multimodal.video as video_module
    from tests.test_multimodal_video import _config as video_config
    from tests.test_multimodal_video import _make_mp4

    video_bytes = _make_mp4(frame_count=5)
    marker = tmp_path / "shell-injection-marker"
    malicious_filename = f"$(touch {marker}).mp4"
    original_open = video_module._av.open
    opened_sources = []

    def checked_open(source, *args, **kwargs):
        opened_sources.append(source)
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(video_module._av, "open", checked_open)
    # Inspect the child-local decoder directly: monkeypatch state is
    # intentionally not inherited by the production ``spawn`` worker.
    result = video_module._decode_and_sample_video_locally(
        file_data=video_bytes,
        filename=malicious_filename,
        declared_mime="video/mp4",
        config=video_config(max_frames=4),
    )

    assert result.evidence
    assert opened_sources and all(isinstance(source, io.BytesIO) for source in opened_sources)
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == []
