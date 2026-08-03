import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.evidence_bundles import RenderedEvidence
from benchmarks.longmemeval.longmemeval_benchmark import LongMemEvalAnswerGenerator


def _generator(context_format: str = "structured_source") -> LongMemEvalAnswerGenerator:
    """Create a formatter without constructing a real LLM client."""

    generator = object.__new__(LongMemEvalAnswerGenerator)
    generator.context_format = context_format
    generator.evidence_mode = None
    return generator


class _FakeConnection:
    def __init__(self, documents, chunks):
        self.documents = documents
        self.chunks = chunks
        self.closed = False

    async def fetch(self, query, *_args):
        if "original_text" in query:
            return self.documents
        return self.chunks

    async def close(self):
        self.closed = True


class _FakeAsyncpg:
    def __init__(self, connection):
        self.connection = connection

    async def connect(self, _database_url):
        return self.connection


def test_source_centric_formatter_reports_only_visible_document_coverage():
    generator = _generator()
    chunk_text = json.dumps([{"role": "user", "content": "The preferred color is green."}])
    rendered = generator._format_context_source_centric(
        {
            "results": [
                {
                    "id": "fact-1",
                    "text": "The preferred color is green.",
                    "fact_type": "preference",
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                }
            ],
            "chunks": {"chunk-1": {"chunk_text": chunk_text}},
        },
        "What color do I prefer?",
    )

    assert isinstance(rendered, RenderedEvidence)
    assert rendered.text
    assert rendered.covered_by_document["doc-1"]
    assert rendered.covered_by_document["doc-1"][0] in rendered.text


@pytest.mark.asyncio
async def test_backfill_recovers_original_text_when_raw_chunk_misses_bundle_budget(monkeypatch):
    generator = _generator()
    results = []
    chunks = {}
    # The first fourteen bundles nearly fill the source-centric cap.  The
    # target chunk is still part of recall, but its bundle is not admitted.
    for index in range(15):
        chunk_id = f"chunk-{index}"
        document_id = f"doc-{index}"
        results.append(
            {
                "id": f"fact-{index}",
                "text": f"noise {index} " + ("x" * 5000),
                "fact_type": "note",
                "document_id": document_id,
                "chunk_id": chunk_id,
            }
        )
        chunks[chunk_id] = {"chunk_text": json.dumps([{"role": "user", "content": f"noise {index}"}])}
    results.append(
        {
            "id": "target-fact",
            "text": "The target code is blue.",
            "fact_type": "preference",
            "document_id": "target-doc",
            "chunk_id": "target-chunk",
        }
    )
    chunks["target-chunk"] = {"chunk_text": json.dumps([{"role": "user", "content": "The target code is blue."}])}
    recall_result = {"results": results, "chunks": chunks}
    rendered = generator._format_context_source_centric(recall_result, "What is the target code?")
    assert "target-doc" not in rendered.covered_by_document

    target_document = json.dumps(
        [
            {"role": "user", "content": "The target code is blue and must be copied exactly."},
        ]
    )
    connection = _FakeConnection(
        documents=[{"id": "target-doc", "original_text": target_document}],
        chunks=[],
    )
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))
    monkeypatch.setenv("HMS_API_DATABASE_URL", "postgresql://fake")

    source_block = await generator._format_source_document_backfill(
        "What is the target code?",
        recall_result,
        "bank-1",
        rendered_coverage=rendered.covered_by_document,
    )

    assert "The target code is blue and must be copied exactly." in source_block
    assert "document=target-doc" in source_block
    assert connection.closed


@pytest.mark.asyncio
async def test_generate_answer_passes_rendered_coverage_to_backfill(monkeypatch):
    generator = _generator()
    expected_coverage = {"doc-1": ("visible source excerpt",)}
    generator._format_context_source_centric = lambda _result, _query: RenderedEvidence(
        text="compact context", covered_by_document=expected_coverage
    )
    captured = {}

    async def fake_backfill(question, recall_result, bank_id, question_type=None, rendered_coverage=None):
        captured.update(
            question=question,
            recall_result=recall_result,
            bank_id=bank_id,
            question_type=question_type,
            rendered_coverage=rendered_coverage,
        )
        return ""

    generator._format_source_document_backfill = fake_backfill

    class _FakeLLM:
        async def call(self, **_kwargs):
            return SimpleNamespace(answer="ok", reasoning="")

    generator.llm_config = _FakeLLM()
    answer, _reasoning, _memories = await generator.generate_answer(
        "Which code?",
        {"results": [], "chunks": {}},
        question_type="single-session-user",
        bank_id="bank-1",
    )

    assert answer == "ok"
    assert captured["rendered_coverage"] == expected_coverage
    assert captured["bank_id"] == "bank-1"
