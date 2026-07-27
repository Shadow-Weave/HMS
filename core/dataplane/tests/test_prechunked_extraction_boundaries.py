"""Contracts for preserving planner-owned Fact Extraction boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from hms_api.engine.ingestion.chunking import compute_content_hash
from hms_api.engine.ingestion.domain import ChunkPlan
from hms_api.engine.ingestion.extraction import (
    ExtractionMode,
    ExtractionPolicy,
    FactExtractorAdapter,
    build_prechunked_extraction_layout,
)
from hms_api.engine.ingestion.normalization import normalize_contents
from hms_api.engine.response_models import TokenUsage


def _oversized_complete_exchange() -> str:
    return json.dumps(
        [
            {"role": "user", "content": "u" * 184},
            {"role": "assistant", "content": "a" * 3024},
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@pytest.mark.asyncio
async def test_prechunked_layout_prevents_resplitting_of_complete_exchange() -> None:
    """A semantic exchange remains one extraction chunk above the fixed limit."""

    from hms_api.engine.retain.fact_extraction import chunk_text

    text = _oversized_complete_exchange()
    assert len(text) > 3000
    item = normalize_contents(
        (
            {
                "content": text,
                "document_id": "document",
                "event_date": None,
            },
        )
    )[0]
    chunk = ChunkPlan(
        chunk_key="chunk-key",
        source_index=item.source_index,
        global_index=0,
        local_index=0,
        text=text,
        content_hash=compute_content_hash(text),
    )
    layout = build_prechunked_extraction_layout((item,), (chunk,))
    request = layout.extraction_request(ExtractionPolicy(mode=ExtractionMode.CONCISE))
    original_config = SimpleNamespace(
        retain_extraction_mode="concise",
        retain_batch_enabled=False,
        retain_chunk_size=3000,
    )
    observed: dict[str, Any] = {}

    async def splitting_primitive(**kwargs: Any):
        observed["config"] = kwargs["config"]
        extracted_chunks = []
        global_index = 0
        for content_index, content in enumerate(kwargs["contents"]):
            for extracted_text in chunk_text(
                content.content,
                kwargs["config"].retain_chunk_size,
            ):
                extracted_chunks.append(
                    SimpleNamespace(
                        chunk_text=extracted_text,
                        fact_count=0,
                        content_index=content_index,
                        chunk_index=global_index,
                    )
                )
                global_index += 1
        return [], extracted_chunks, TokenUsage()

    result = await FactExtractorAdapter(
        llm_config=object(),
        config=original_config,
        agent_name="agent",
        sync_primitive=splitting_primitive,
        batch_primitive=splitting_primitive,
    ).extract(request)

    assert request.preserve_chunk_boundaries is True
    assert len(result.chunk_fact_counts) == 1
    assert result.chunk_fact_counts[0].fact_count == 0
    assert original_config.retain_chunk_size == 3000
    assert observed["config"] is not original_config
    assert observed["config"].retain_chunk_size == len(text)


def test_boundary_preservation_does_not_copy_config_when_limit_is_sufficient() -> None:
    item = normalize_contents(
        (
            {
                "content": "short",
                "document_id": "document",
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
    config = SimpleNamespace(retain_chunk_size=3000)

    preserved = FactExtractorAdapter._boundary_preserving_config(request, config)

    assert preserved is config
