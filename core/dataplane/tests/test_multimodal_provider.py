"""Transport-level OpenAI Responses tests with an in-memory HTTP server."""

import base64
import json

import httpx
import pytest

from hms_api.engine.multimodal import (
    GroundedStatement,
    ModelMultimodalDescription,
    ModelTemporalSegment,
    OpenAIProviderConfig,
    OpenAIResponsesMultimodalProvider,
    ProviderIncompleteError,
    ProviderRefusalError,
    ProviderSchemaError,
    VisualEvidence,
    estimate_provider_budget,
)


class _ChunkedResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _evidence(payload: bytes = b"synthetic-image-bytes") -> VisualEvidence:
    import hashlib

    return VisualEvidence(
        evidence_id="image-000-deadbeef",
        timestamp_ms=None,
        sha256=hashlib.sha256(payload).hexdigest(),
        mime_type="image/png",
        width=32,
        height=16,
        encoded_bytes=payload,
    )


def _description() -> ModelMultimodalDescription:
    return ModelMultimodalDescription(
        summary=[
            GroundedStatement(
                text="a synthetic terminal is visible",
                evidence_ids=["image-000-deadbeef"],
                uncertainty="low",
            )
        ],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[],
        limitations=[],
    )


def _video_evidence(index: int) -> VisualEvidence:
    import hashlib

    payload = f"synthetic-frame-{index}".encode()
    return VisualEvidence(
        evidence_id=f"frame-{index:03d}",
        timestamp_ms=index * 1_000,
        sha256=hashlib.sha256(payload).hexdigest(),
        mime_type="image/jpeg",
        width=64,
        height=32,
        encoded_bytes=payload,
    )


def _video_segment() -> ModelTemporalSegment:
    return ModelTemporalSegment(
        segment_id="segment-000",
        summary=[
            GroundedStatement(
                text="the editor state changes",
                evidence_ids=["frame-000", "frame-001"],
                uncertainty="low",
            )
        ],
        observations=[],
        visible_text=[],
        evidence_ids=["frame-000", "frame-001"],
    )


def _video_description() -> ModelMultimodalDescription:
    segment = _video_segment()
    return ModelMultimodalDescription(
        summary=[
            GroundedStatement(
                text="the coding session changes state",
                evidence_ids=["frame-000", "frame-001"],
                uncertainty="low",
            )
        ],
        entities=[],
        observations=[],
        visible_text=[],
        temporal_segments=[segment],
        limitations=[],
    )


def _assert_visual_inference_policy(prompt: str) -> None:
    normalized = prompt.lower()
    assert "do not identify or verify any person" in normalized
    assert "face recognition" in normalized
    assert "protected or sensitive attributes" in normalized
    assert "visible text" in normalized
    assert "verified" in normalized


def test_provider_budget_is_bounded_before_io():
    evidence = [_video_evidence(index) for index in range(24)]
    budget = estimate_provider_budget(
        evidence,
        logical_calls=4,
        max_retries=2,
        max_schema_repairs=1,
        max_output_tokens=4096,
    )

    assert budget.image_count == 24
    assert budget.logical_calls == 4
    assert budget.physical_attempts_upper_bound == 24
    assert budget.output_tokens_upper_bound == 98_304
    assert budget.estimated_image_transport_bytes == sum(
        len(f"data:{item.mime_type};base64,") + 4 * ((len(item.encoded_bytes) + 2) // 3) for item in evidence
    )


@pytest.mark.asyncio
async def test_pipeline_identity_hashes_endpoint_without_persisting_url():
    identities = []
    for base_url in ("https://api.openai.com/v1", "https://private-gateway.invalid/v1"):
        config = _config(base_url=base_url)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
            provider = OpenAIResponsesMultimodalProvider(config, client=client)
            identity = provider.pipeline_identity()
            identities.append(identity)
            assert base_url not in json.dumps(identity)
            assert len(identity["endpoint_fingerprint"]) == 64
            assert identity["retry_backoff"] == "full-jitter-v1"

    assert identities[0]["endpoint_fingerprint"] != identities[1]["endpoint_fingerprint"]


def _success_response(description: ModelMultimodalDescription) -> dict:
    return {
        "id": "resp_test",
        "model": "gpt-5-mini-test-revision",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": description.model_dump_json()}],
            }
        ],
        "usage": {"input_tokens": 123, "output_tokens": 45},
    }


def _config(**overrides) -> OpenAIProviderConfig:
    values = {
        "api_key": "unit-test-secret",
        "base_url": "https://api.openai.test/v1",
        "model": "gpt-5-mini",
        "initial_backoff_seconds": 0,
    }
    values.update(overrides)
    return OpenAIProviderConfig(**values)


@pytest.mark.asyncio
async def test_responses_wire_shape_and_ephemeral_data_url():
    evidence = _evidence()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer unit-test-secret"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5-mini"
        assert payload["store"] is False
        assert "response_format" not in payload
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
        schema = payload["text"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        serialized_schema = json.dumps(schema)
        assert '"minLength"' not in serialized_schema
        assert '"maxLength"' not in serialized_schema
        # Array bounds are supported for non-fine-tuned Structured Outputs
        # models and remain useful for controlling response size.
        assert '"minItems"' in serialized_schema
        assert '"maxItems"' in serialized_schema
        content = payload["input"][0]["content"]
        assert content[0]["type"] == "input_text"
        _assert_visual_inference_policy(content[0]["text"])
        image_url = content[1]["image_url"]
        prefix, encoded = image_url.split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded) == evidence.encoded_bytes
        assert content[1]["detail"] == "auto"
        return httpx.Response(200, headers={"x-request-id": "req_test"}, json=_success_response(_description()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=_config().base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(_config(), client=client)
        result = await provider.describe_image(evidence)

    assert calls == 1
    assert result.value.summary[0].text == "a synthetic terminal is visible"
    assert result.request_id == "req_test"
    assert result.resolved_model == "gpt-5-mini-test-revision"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.physical_attempts == 1
    assert "unit-test-secret" not in repr(provider.config)


@pytest.mark.asyncio
async def test_transient_http_failure_is_bounded_and_counted():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "do not persist this body"}})
        return httpx.Response(200, json=_success_response(_description()))

    config = _config(max_retries=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        result = await provider.describe_image(_evidence())

    assert calls == 2
    assert result.physical_attempts == 2


@pytest.mark.asyncio
async def test_transport_retries_use_injectable_bounded_full_jitter():
    calls = 0
    jitter_bounds: list[float] = []
    slept: list[float] = []
    fractions = iter((0.25, 0.5, 0.75))

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 4:
            status = (503, 429, 502)[calls - 1]
            return httpx.Response(status, json={"error": {"message": "transient"}})
        return httpx.Response(200, json=_success_response(_description()))

    def deterministic_jitter(upper_bound: float) -> float:
        jitter_bounds.append(upper_bound)
        return upper_bound * next(fractions)

    async def record_sleep(delay: float) -> None:
        slept.append(delay)

    config = _config(max_retries=3, initial_backoff_seconds=2.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(
            config,
            client=client,
            sleep=record_sleep,
            jitter=deterministic_jitter,
        )
        result = await provider.describe_image(_evidence())

    assert calls == 4
    assert result.physical_attempts == 4
    assert jitter_bounds == [2.0, 4.0, 8.0]
    assert slept == [0.5, 2.0, 6.0]
    assert all(0.0 <= delay <= upper_bound for delay, upper_bound in zip(slept, jitter_bounds, strict=True))


@pytest.mark.parametrize("initial_backoff", [-1.0, float("inf"), float("nan")])
def test_retry_backoff_rejects_invalid_initial_delay(initial_backoff):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _config(initial_backoff_seconds=initial_backoff)


def test_response_byte_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_response_bytes must be positive"):
        _config(max_response_bytes=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 400])
async def test_provider_response_body_is_streamed_with_a_hard_byte_limit(status_code):
    sentinel = "OVERSIZED_PROVIDER_BODY_SENTINEL"
    body = (sentinel * 20).encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body)

    config = _config(max_retries=0, max_response_bytes=128)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        with pytest.raises(ProviderSchemaError) as exc_info:
            await provider.describe_image(_evidence())

    assert exc_info.value.code == "provider.response_too_large"
    assert exc_info.value.physical_attempts == 1
    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_response_cumulative_limit_does_not_trust_content_length():
    sentinel = b"CHUNKED_OVERSIZED_SENTINEL"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ChunkedResponseStream([sentinel] * 8))

    config = _config(max_retries=0, max_response_bytes=64)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        with pytest.raises(ProviderSchemaError) as exc_info:
            await provider.describe_image(_evidence())

    assert exc_info.value.code == "provider.response_too_large"
    assert exc_info.value.physical_attempts == 1
    assert sentinel.decode() not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_usage", [True, -1, "many", [], 2**63])
async def test_provider_usage_counts_are_typed_non_negative_and_bounded(invalid_usage):
    response_body = _success_response(_description())
    response_body["usage"] = {"input_tokens": invalid_usage, "output_tokens": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    config = _config(max_retries=0, max_schema_repairs=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        with pytest.raises(ProviderSchemaError) as exc_info:
            await provider.describe_image(_evidence())

    assert exc_info.value.code == "provider.invalid_usage"
    assert exc_info.value.physical_attempts == 1


@pytest.mark.asyncio
async def test_schema_repair_is_distinct_from_transport_retry():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            invalid = {
                "summary": [],
                "entities": [],
                "observations": [],
                "visible_text": [],
                "temporal_segments": [],
                "limitations": [],
            }
            response = _success_response(_description())
            response["output"][0]["content"][0]["text"] = json.dumps(invalid)
            return httpx.Response(200, json=response)
        return httpx.Response(200, json=_success_response(_description()))

    config = _config(max_retries=0, max_schema_repairs=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        result = await provider.describe_image(_evidence())

    assert calls == 2
    assert result.physical_attempts == 2


@pytest.mark.asyncio
async def test_video_map_and_evidence_only_reduce_wire_contract():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "hms_multimodal_video_segment":
            output = _video_segment()
        else:
            output = _video_description()
        response = _success_response(_description())
        response["output"][0]["content"][0]["text"] = output.model_dump_json()
        return httpx.Response(200, json=response)

    config = _config(max_retries=0, max_schema_repairs=0)
    evidence = [_video_evidence(0), _video_evidence(1)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        segment_result = await provider.describe_video_segment("segment-000", evidence)
        reduce_result = await provider.reduce_video([segment_result.value])

    assert segment_result.value.segment_id == "segment-000"
    assert reduce_result.value.temporal_segments == [segment_result.value]
    map_content = requests[0]["input"][0]["content"]
    assert [part["type"] for part in map_content] == ["input_text", "input_image", "input_image"]
    assert "frame-000@0ms" in map_content[0]["text"]
    assert "frame-001@1000ms" in map_content[0]["text"]
    _assert_visual_inference_policy(map_content[0]["text"])
    assert requests[0]["store"] is False

    reduce_content = requests[1]["input"][0]["content"]
    assert [part["type"] for part in reduce_content] == ["input_text"]
    assert "data:image" not in reduce_content[0]["text"]
    assert "validated segments" in reduce_content[0]["text"].lower()
    _assert_visual_inference_policy(reduce_content[0]["text"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_body", "expected_error"),
    [
        (
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "policy"}]}],
            },
            ProviderRefusalError,
        ),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
            ProviderIncompleteError,
        ),
    ],
)
async def test_refusal_and_incomplete_are_terminal_not_schema_repairs(response_body, expected_error):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=response_body)

    config = _config(max_retries=2, max_schema_repairs=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=config.base_url + "/") as client:
        provider = OpenAIResponsesMultimodalProvider(config, client=client)
        with pytest.raises(expected_error) as exc_info:
            await provider.describe_image(_evidence(b"base64-sentinel-secret"))

    assert calls == 1
    assert "base64-sentinel-secret" not in str(exc_info.value)
    assert "data:image" not in str(exc_info.value)
