"""Retain-only affect recognition tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hms_api.engine.response_models import TokenUsage
from hms_api.engine.retain.affect import AffectSignals, parse_affect
from hms_api.engine.retain.fact_extraction import (
    _build_extraction_prompt_and_schema,
    _extract_facts_from_chunk,
)


def _config(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
        retain_custom_instructions=None,
        entity_labels=None,
        entities_allow_free_form=True,
        retain_affect_enabled=enabled,
        retain_affect_version="affect-v1",
        retain_batch_enabled=False,
        retain_llm_max_retries=1,
        llm_max_retries=1,
        retain_llm_initial_backoff=None,
        llm_initial_backoff=0.0,
        retain_llm_max_backoff=None,
        llm_max_backoff=0.0,
        retain_max_completion_tokens=8192,
    )


def test_affect_schema_is_opt_in_and_strict() -> None:
    disabled_prompt, disabled_schema = _build_extraction_prompt_and_schema(_config(enabled=False))
    assert "AFFECT RECOGNITION" not in disabled_prompt
    assert "AffectFact" not in disabled_schema.model_json_schema().get("$defs", {})

    enabled_prompt, enabled_schema = _build_extraction_prompt_and_schema(_config(enabled=True))
    schema = enabled_schema.model_json_schema()
    fact_schema = schema["$defs"]["AffectFact"]
    affect_schema = schema["$defs"]["AffectAnnotation"]

    assert "AFFECT RECOGNITION" in enabled_prompt
    assert "Do not attribute the assistant" in enabled_prompt
    assert "empathetic tone to the user" in enabled_prompt
    assert "affect" in fact_schema["required"]
    assert set(affect_schema["properties"]["sentiment"]["enum"]) == {
        "positive",
        "negative",
        "neutral",
    }
    assert set(affect_schema["properties"]["emotion"]["enum"]) == {
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "neutral",
    }
    assert affect_schema["properties"]["intensity"]["minimum"] == 0.0
    assert affect_schema["properties"]["intensity"]["maximum"] == 1.0


def test_parse_affect_is_versioned_and_fail_soft() -> None:
    affect = parse_affect(
        {"sentiment": "negative", "emotion": "sadness", "intensity": 0.85},
        version="affect-v1",
    )

    assert affect == AffectSignals(
        sentiment="negative",
        emotion="sadness",
        intensity=0.85,
        version="affect-v1",
    )
    assert (
        parse_affect(
            {"sentiment": "negative", "emotion": "nostalgia", "intensity": 0.5},
            version="affect-v1",
        )
        is None
    )
    assert (
        parse_affect(
            {"sentiment": "positive", "emotion": "joy", "intensity": 1.1},
            version="affect-v1",
        )
        is None
    )


@pytest.mark.asyncio
async def test_extract_chunk_attaches_affect_when_enabled() -> None:
    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=(
            {
                "facts": [
                    {
                        "what": "The user is delighted about shipping the release",
                        "when": None,
                        "who": "the user",
                        "why": "the release shipped",
                        "fact_type": "world",
                        "affect": {
                            "sentiment": "positive",
                            "emotion": "joy",
                            "intensity": 0.9,
                        },
                    }
                ]
            },
            TokenUsage(),
        )
    )

    facts, _usage = await _extract_facts_from_chunk(
        chunk="I am thrilled that we finally shipped!",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="chat message",
        llm_config=llm,
        config=_config(enabled=True),
        agent_name="chatbot",
    )

    assert len(facts) == 1
    assert facts[0].affect == AffectSignals(
        sentiment="positive",
        emotion="joy",
        intensity=0.9,
        version="affect-v1",
    )


@pytest.mark.asyncio
async def test_extract_chunk_ignores_affect_when_disabled() -> None:
    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=(
            {
                "facts": [
                    {
                        "what": "The user is angry",
                        "fact_type": "world",
                        "affect": {
                            "sentiment": "negative",
                            "emotion": "anger",
                            "intensity": 1.0,
                        },
                    }
                ]
            },
            TokenUsage(),
        )
    )

    facts, _usage = await _extract_facts_from_chunk(
        chunk="I am furious.",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="chat message",
        llm_config=llm,
        config=_config(enabled=False),
        agent_name="chatbot",
    )

    assert len(facts) == 1
    assert facts[0].affect is None


@pytest.mark.asyncio
async def test_affect_survives_ingestion_projection_pipeline() -> None:
    from hms_api.engine.ingestion.chunking import compute_content_hash
    from hms_api.engine.ingestion.domain import ChunkPlan, freeze_json
    from hms_api.engine.ingestion.extraction import (
        ExtractionMode,
        ExtractionPolicy,
        FactExtractorAdapter,
        build_prechunked_extraction_layout,
    )
    from hms_api.engine.ingestion.normalization import normalize_contents
    from hms_api.engine.ingestion.projection.records import MemoryRecord, to_processed_fact
    from hms_api.engine.retain.types import ChunkMetadata, ExtractedFact

    affect = AffectSignals(
        sentiment="negative",
        emotion="fear",
        intensity=0.7,
        version="affect-v1",
    )
    item = normalize_contents(
        (
            {
                "content": "I am worried the deployment might fail.",
                "context": "chat message",
                "document_id": "document-1",
                "event_date": None,
            },
        )
    )[0]
    chunk = ChunkPlan(
        chunk_key="chunk-key",
        source_index=item.source_index,
        global_index=0,
        local_index=0,
        text=item.content,
        content_hash=compute_content_hash(item.content),
    )
    request = build_prechunked_extraction_layout((item,), (chunk,)).extraction_request(
        ExtractionPolicy(mode=ExtractionMode.CONCISE)
    )

    async def primitive(**_kwargs):
        return (
            [
                ExtractedFact(
                    fact_text="The user is worried the deployment might fail.",
                    fact_type="world",
                    affect=affect,
                    content_index=0,
                    chunk_index=0,
                    context="chat message",
                )
            ],
            [
                ChunkMetadata(
                    chunk_text=item.content,
                    fact_count=1,
                    content_index=0,
                    chunk_index=0,
                )
            ],
            TokenUsage(),
        )

    result = await FactExtractorAdapter(
        llm_config=object(),
        config=SimpleNamespace(
            retain_extraction_mode="concise",
            retain_batch_enabled=False,
            retain_chunk_size=3000,
        ),
        agent_name="chatbot",
        sync_primitive=primitive,
        batch_primitive=primitive,
    ).extract(request)

    candidate = result.candidates[0]
    record = MemoryRecord.from_candidate(
        candidate,
        embedding=None,
        projection=freeze_json({"extraction": {"v": "test"}}),
    )
    processed = to_processed_fact(
        record,
        document_id="document-1",
        chunk_id="chunk-1",
        content_index=0,
    )

    assert candidate.affect is affect
    assert record.affect is affect
    assert processed.affect is affect
