"""Lightweight tests for projection manifest plumbing."""

from __future__ import annotations

from datetime import UTC, datetime

from hms_api.engine.retain.types import ExtractedFact, ProcessedFact
from hms_api.engine.search.types import RetrievalResult


def test_processed_fact_projection_records_ready_channels():
    extracted = ExtractedFact(
        fact_text="Alice joined LaunchPlan.",
        fact_type="world",
        entities=["Alice", "LaunchPlan"],
        occurred_start=datetime(2026, 7, 15, tzinfo=UTC),
    )

    processed = ProcessedFact.from_extracted_fact(
        extracted,
        embedding=[0.1, 0.2, 0.3],
        extraction_prompt_version="prompt-v2",
        embedding_model_version="embed-model:3",
    )

    assert processed.projection == {
        "embedding": {"v": "embed-model:3", "ok": True},
        "tsvector": {"v": 1, "ok": True},
        "temporal": {"v": 1, "grade": "resolved"},
        "entities": {"v": 1, "ok": True},
        "extraction": {"v": "prompt-v2"},
    }


def test_processed_fact_projection_records_missing_materials():
    extracted = ExtractedFact(
        fact_text="A fact without extracted structure.",
        fact_type="world",
        entities=[],
        occurred_start=None,
        mentioned_at=None,
    )

    processed = ProcessedFact.from_extracted_fact(extracted, embedding=None)

    assert processed.embedding is None
    assert processed.projection["embedding"]["ok"] is False
    assert processed.projection["temporal"]["grade"] == "unresolved"
    assert processed.projection["entities"]["ok"] is False


def test_retrieval_result_parses_projection_json_string():
    result = RetrievalResult.from_db_row(
        {
            "id": "unit-1",
            "text": "Alice joined LaunchPlan.",
            "fact_type": "world",
            "projection": '{"temporal": {"grade": "resolved"}}',
        }
    )

    assert result.projection == {"temporal": {"grade": "resolved"}}
