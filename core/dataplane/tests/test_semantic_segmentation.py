"""Focused offline contracts for Retain semantic boundary planning."""

from __future__ import annotations

import json

import pytest

from hms_api.engine.ingestion.chunking import compute_content_hash, split_text
from hms_api.engine.ingestion.domain import (
    ChunkPolicy,
    ContentItem,
    EventDateState,
    EventDateValue,
    UpdateMode,
    freeze_json,
)
from hms_api.engine.ingestion.segmentation import (
    BoundaryResponse,
    EffectiveSegmentationStrategy,
    SegmentationFailurePolicy,
    SegmentationManifest,
    SegmentationMode,
    SegmentationReuseError,
    SemanticBoundaryValidationError,
    SemanticSegmentationError,
    SemanticSegmentationPolicy,
    SemanticSegmenter,
    build_chunk_plans_from_segmentation,
    materialize_semantic_boundaries,
    parse_conversation,
    validate_boundary_response,
)
from hms_api.engine.response_models import TokenUsage


def _conversation(turns: list[dict]) -> str:
    return json.dumps(turns, ensure_ascii=False)


def _policy(**overrides) -> SemanticSegmentationPolicy:
    values = {
        "max_chars": 10_000,
        "provider": "mock",
        "model": "boundary-model",
    }
    values.update(overrides)
    return SemanticSegmentationPolicy(**values)


class _FakeLLM:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def test_parse_conversation_groups_complete_user_exchanges_and_preserves_turns() -> None:
    turns = [
        {"role": "system", "content": "System context", "extra": {"b": 2, "a": 1}},
        {"role": "user", "content": "Basketball question"},
        {"role": "assistant", "content": "Basketball answer"},
        {"role": "tool", "content": "Box score"},
        {"role": "user", "content": "Travel question"},
        {"role": "assistant", "content": "Travel answer"},
    ]

    parsed = parse_conversation(_conversation(turns))

    assert parsed is not None
    assert tuple((item.start_turn, item.end_turn) for item in parsed.exchanges) == ((0, 3), (4, 5))
    assert json.loads(parsed.canonical_text) == turns
    assert parsed.canonical_text == (
        '[{"content":"System context","extra":{"a":1,"b":2},"role":"system"},'
        '{"content":"Basketball question","role":"user"},'
        '{"content":"Basketball answer","role":"assistant"},'
        '{"content":"Box score","role":"tool"},'
        '{"content":"Travel question","role":"user"},'
        '{"content":"Travel answer","role":"assistant"}]'
    )


@pytest.mark.parametrize(
    "indices",
    (
        [],
        [1, 0, 2],
        [0, 1],
        [-1, 2],
        [0, 3],
        [0, 2, 2],
    ),
)
def test_boundary_validation_rejects_incomplete_or_unordered_coverage(indices: list[int]) -> None:
    with pytest.raises((SemanticBoundaryValidationError, ValueError)):
        validate_boundary_response(
            {"end_exchange_indices": indices},
            exchange_count=3,
        )


def test_materialization_uses_boundaries_only_and_preserves_original_turn_data() -> None:
    turns = [
        {"role": "user", "content": "Basketball"},
        {"role": "assistant", "content": "Warriors"},
        {"role": "user", "content": "Still basketball"},
        {"role": "assistant", "content": "Playoffs"},
        {"role": "user", "content": "Now travel"},
        {"role": "assistant", "content": "Kyoto"},
    ]
    parsed = parse_conversation(_conversation(turns))
    assert parsed is not None

    segments = materialize_semantic_boundaries(parsed, (1, 2), max_chars=10_000)

    assert tuple((item.start_exchange, item.end_exchange) for item in segments) == ((0, 1), (2, 2))
    reconstructed = [turn for segment in segments for turn in json.loads(segment.text)]
    assert reconstructed == turns


def test_hard_limit_splits_only_between_exchanges_and_retains_topic_identity() -> None:
    turns = [{"role": "user", "content": f"Question {index} " + ("x" * 24)} for index in range(3)]
    parsed = parse_conversation(_conversation(turns))
    assert parsed is not None
    one_exchange_lengths = [len(parsed.render_exchange_range(index, index)) for index in range(3)]
    max_chars = max(one_exchange_lengths) + 1

    first = materialize_semantic_boundaries(parsed, (2,), max_chars=max_chars)
    second = materialize_semantic_boundaries(parsed, (2,), max_chars=max_chars)

    assert first == second
    assert len(first) == 3
    assert all(item.semantic_segment_index == 0 for item in first)
    assert all(len(item.text) <= max_chars for item in first)
    assert [json.loads(item.text)[0]["content"] for item in first] == [turn["content"] for turn in turns]


def test_oversized_atomic_exchange_is_preserved_and_marked() -> None:
    parsed = parse_conversation(
        _conversation(
            [
                {"role": "user", "content": "q" * 100},
                {"role": "assistant", "content": "a" * 100},
            ]
        )
    )
    assert parsed is not None

    segments = materialize_semantic_boundaries(parsed, (0,), max_chars=80)

    assert len(segments) == 1
    assert segments[0].oversized_atomic is True
    assert json.loads(segments[0].text) == [
        {"role": "user", "content": "q" * 100},
        {"role": "assistant", "content": "a" * 100},
    ]


@pytest.mark.asyncio
async def test_semantic_segmenter_calls_structured_provider_and_builds_stable_manifest() -> None:
    text = _conversation(
        [
            {"role": "user", "content": "Basketball"},
            {"role": "assistant", "content": "Warriors"},
            {"role": "user", "content": "Travel"},
            {"role": "assistant", "content": "Kyoto"},
        ]
    )
    usage = TokenUsage(input_tokens=50, output_tokens=2)
    llm = _FakeLLM((BoundaryResponse(end_exchange_indices=[0, 1]), usage))
    segmenter = SemanticSegmenter(llm_config=llm, policy=_policy(max_chars=100))

    first = await segmenter.segment(text)
    second = await segmenter.segment(text)

    assert first.manifest.effective_strategy is EffectiveSegmentationStrategy.SEMANTIC
    assert first.manifest.end_exchange_indices == (0, 1)
    assert first.manifest.plan_digest == second.manifest.plan_digest
    assert first.usage == usage
    assert len(llm.calls) == 2
    assert llm.calls[0]["response_format"] is BoundaryResponse
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["strict_schema"] is True
    assert llm.calls[0]["return_usage"] is True
    assert "summary" not in BoundaryResponse.model_json_schema()["properties"]
    assert "label" not in BoundaryResponse.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_invalid_boundaries_use_exact_fixed_chunker_fallback() -> None:
    text = _conversation(
        [
            {"role": "user", "content": "Basketball " + ("x" * 80)},
            {"role": "assistant", "content": "Answer " + ("y" * 80)},
            {"role": "user", "content": "Travel " + ("z" * 80)},
        ]
    )
    policy = _policy(max_chars=150)
    usage = TokenUsage(input_tokens=20, output_tokens=1)
    llm = _FakeLLM((BoundaryResponse(end_exchange_indices=[0]), usage))

    result = await SemanticSegmenter(llm_config=llm, policy=policy).segment(text)
    expected = split_text(
        text,
        ChunkPolicy(
            version="test-fixed",
            max_chars=policy.max_chars,
            conversation_mode=True,
            overlap=0,
        ),
    )

    assert tuple(item.text for item in result.segments) == expected
    assert result.manifest.effective_strategy is EffectiveSegmentationStrategy.FIXED_FALLBACK
    assert result.manifest.fallback_reason == "invalid_boundaries"
    assert result.manifest.end_exchange_indices == ()
    assert result.usage == usage


@pytest.mark.asyncio
async def test_non_conversation_uses_fixed_fallback_without_calling_provider() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    llm = _FakeLLM(error=AssertionError("provider must not be called"))

    result = await SemanticSegmenter(
        llm_config=llm,
        policy=_policy(max_chars=20),
    ).segment(text)

    assert llm.calls == []
    assert result.manifest.effective_strategy is EffectiveSegmentationStrategy.FIXED_FALLBACK
    assert result.manifest.fallback_reason == "not_conversation"


@pytest.mark.asyncio
async def test_provider_failure_can_fail_closed() -> None:
    text = _conversation(
        [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
            {"role": "user", "content": "Three"},
        ]
    )
    segmenter = SemanticSegmenter(
        llm_config=_FakeLLM(error=TimeoutError("provider timeout")),
        policy=_policy(max_chars=60, failure_policy=SegmentationFailurePolicy.RAISE),
    )

    with pytest.raises(SemanticSegmentationError, match="semantic boundary planning failed"):
        await segmenter.segment(text)


def test_policy_fingerprint_and_plan_digest_change_with_behavior() -> None:
    baseline = _policy()
    same = _policy()
    changed_prompt = _policy(prompt_version="semantic-boundary-prompt-v2")
    changed_model = _policy(model="different-model")
    changed_limit = _policy(max_chars=9_999)
    changed_completion_limit = _policy(max_completion_tokens=2_048)
    changed_retries = _policy(max_retries=2)

    assert baseline.fingerprint == same.fingerprint
    assert len(baseline.fingerprint) == 64
    assert (
        len(
            {
                baseline.fingerprint,
                changed_prompt.fingerprint,
                changed_model.fingerprint,
                changed_limit.fingerprint,
                changed_completion_limit.fingerprint,
                changed_retries.fingerprint,
            }
        )
        == 6
    )


@pytest.mark.asyncio
async def test_canonical_input_produces_same_digest_across_json_formatting() -> None:
    compact = '[{"role":"user","content":"One","metadata":{"b":2,"a":1}},{"role":"user","content":"Two"}]'
    formatted = json.dumps(
        [
            {"metadata": {"a": 1, "b": 2}, "content": "One", "role": "user"},
            {"content": "Two", "role": "user"},
        ],
        indent=2,
    )
    usage = TokenUsage()
    first = await SemanticSegmenter(
        llm_config=_FakeLLM((BoundaryResponse(end_exchange_indices=[1]), usage)),
        policy=_policy(max_chars=60),
    ).segment(compact)
    second = await SemanticSegmenter(
        llm_config=_FakeLLM((BoundaryResponse(end_exchange_indices=[1]), usage)),
        policy=_policy(max_chars=60),
    ).segment(formatted)

    assert first.manifest.input_hash == second.manifest.input_hash
    assert first.manifest.plan_digest == second.manifest.plan_digest


@pytest.mark.asyncio
async def test_short_content_is_byte_exact_passthrough_without_provider_call() -> None:
    text = '[ { "role": "user", "content": "raw formatting" } ]'
    llm = _FakeLLM(error=AssertionError("provider must not be called"))

    result = await SemanticSegmenter(llm_config=llm, policy=_policy(max_chars=len(text))).plan_document(text)

    assert llm.calls == []
    assert result.segments[0].text == text
    assert result.manifest.effective_strategy is EffectiveSegmentationStrategy.PASSTHROUGH


@pytest.mark.asyncio
async def test_explicit_fixed_bypass_never_calls_provider() -> None:
    text = _conversation(
        [
            {"role": "user", "content": "One " + ("x" * 100)},
            {"role": "assistant", "content": "Two " + ("y" * 100)},
        ]
    )
    llm = _FakeLLM(error=AssertionError("provider must not be called"))

    result = await SemanticSegmenter(
        llm_config=llm,
        policy=_policy(max_chars=80),
    ).plan_document(text, mode=SegmentationMode.FIXED_BYPASS)

    assert llm.calls == []
    assert result.manifest.effective_strategy is EffectiveSegmentationStrategy.FIXED_BYPASS
    assert result.manifest.fallback_reason is None


@pytest.mark.asyncio
async def test_manifest_round_trip_and_reuse_are_text_free_and_provider_free() -> None:
    private_text = _conversation(
        [
            {"role": "user", "content": "private-topic-" + ("x" * 50)},
            {"role": "assistant", "content": "private-answer-" + ("y" * 50)},
            {"role": "user", "content": "second-private-topic-" + ("z" * 50)},
        ]
    )
    usage = TokenUsage(input_tokens=10, output_tokens=1)
    planning_llm = _FakeLLM((BoundaryResponse(end_exchange_indices=[0, 1]), usage))
    policy = _policy(max_chars=100)
    planned = await SemanticSegmenter(llm_config=planning_llm, policy=policy).plan_document(private_text)
    serialized = planned.manifest.as_dict()

    assert SegmentationManifest.from_dict(serialized) == planned.manifest
    assert "private-topic" not in json.dumps(serialized)

    reuse_llm = _FakeLLM(error=AssertionError("provider must not be called"))
    reused = SemanticSegmenter(llm_config=reuse_llm, policy=policy).reuse(private_text, serialized)

    assert reuse_llm.calls == []
    assert reused.segments == planned.segments
    assert reused.manifest == planned.manifest
    assert reused.usage == TokenUsage()


@pytest.mark.asyncio
async def test_reuse_rejects_source_policy_and_manifest_tampering() -> None:
    text = "short private content"
    policy = _policy(max_chars=100)
    segmenter = SemanticSegmenter(llm_config=_FakeLLM(), policy=policy)
    result = await segmenter.plan_document(text)

    with pytest.raises(SegmentationReuseError, match="source input hash"):
        segmenter.reuse(text + " changed", result.manifest)
    with pytest.raises(SegmentationReuseError, match="policy fingerprint"):
        SemanticSegmenter(
            llm_config=_FakeLLM(),
            policy=_policy(max_chars=101),
        ).reuse(text, result.manifest)

    serialized = result.manifest.as_dict()
    serialized["unexpected"] = True
    with pytest.raises(SegmentationReuseError, match="manifest is invalid"):
        segmenter.reuse(text, serialized)


@pytest.mark.asyncio
async def test_chunk_plan_adapter_assigns_stable_document_indices() -> None:
    policy = _policy(max_chars=10)
    segmenter = SemanticSegmenter(llm_config=_FakeLLM(), policy=policy)
    items = (
        ContentItem(
            content="first item content",
            context="",
            event_date=EventDateValue(EventDateState.TIMELESS, None),
            metadata=freeze_json({}),
            entities=(),
            tags=(),
            observation_scopes=None,
            document_id=None,
            update_mode=UpdateMode.REPLACE,
            source_index=4,
        ),
        ContentItem(
            content="second",
            context="",
            event_date=EventDateValue(EventDateState.TIMELESS, None),
            metadata=freeze_json({}),
            entities=(),
            tags=(),
            observation_scopes=None,
            document_id=None,
            update_mode=UpdateMode.REPLACE,
            source_index=9,
        ),
    )
    results = (
        await segmenter.plan_document(items[0].content, mode=SegmentationMode.FIXED_BYPASS),
        await segmenter.plan_document(items[1].content, mode=SegmentationMode.FIXED_BYPASS),
    )

    plans = build_chunk_plans_from_segmentation("doc:one", items, results)

    assert tuple(plan.global_index for plan in plans) == tuple(range(len(plans)))
    assert tuple(plan.local_index for plan in plans) == (0, 1, 0)
    assert tuple(plan.source_index for plan in plans) == (4, 4, 9)
    assert all(plan.content_hash == compute_content_hash(plan.text) for plan in plans)
    assert all(plan.chunk_key.startswith("chunk:7:doc:one:") for plan in plans)
