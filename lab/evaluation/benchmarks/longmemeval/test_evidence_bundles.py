import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.longmemeval.evidence_bundles import (
    build_evidence_bundles,
    render_evidence_bundles,
    render_evidence_with_coverage,
)


def _fact(text, *, chunk_id="chunk-1", document_id="doc-1", fact_type="world"):
    return {
        "id": text,
        "text": text,
        "fact_type": fact_type,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "occurred_start": None,
        "occurred_end": None,
        "mentioned_at": None,
    }


def test_bundles_render_each_source_chunk_once_and_keep_distinct_facts():
    results = [
        _fact("A generic recommendation about travel."),
        _fact("The train cost $50 and was booked on May 20."),
        _fact("A second distinct event in the same source.", chunk_id="chunk-2"),
    ]
    chunks = {
        "chunk-1": {"chunk_text": "raw chunk one", "chunk_index": 0},
        "chunk-2": {"chunk_text": "raw chunk two", "chunk_index": 1},
    }

    bundles = build_evidence_bundles(results, chunks, "How much did the train cost?")
    rendered = render_evidence_with_coverage(bundles)

    assert len(bundles) == 2
    assert rendered.text.count("raw chunk one") == 1
    assert rendered.text.count("raw chunk two") == 1
    assert "The train cost $50" in rendered.text
    assert rendered.covered_by_document == {"doc-1": ("raw chunk one", "raw chunk two")}


def test_bundle_selection_is_stable_for_missing_chunk_and_observation_rows():
    results = [
        _fact("First fact", chunk_id=None, document_id="doc-1"),
        _fact("Second fact", chunk_id=None, document_id="doc-1"),
        _fact("Third fact", chunk_id=None, document_id="doc-1"),
    ]
    bundles = build_evidence_bundles(results, {}, "What happened?", max_facts_per_bundle=2)

    assert len(bundles) == 1
    assert [fact["text"] for fact in bundles[0].facts] == ["First fact", "Second fact"]


def test_chunk_rendering_preserves_late_state_update_after_long_turn():
    long_assistant = "synthetic robotics workshop venue recommendations " * 180
    results = [_fact("User selected the Copper Finch Inn", chunk_id="chunk-1")]
    chunks = {
        "chunk-1": {
            "chunk_text": json.dumps(
                [
                    {"role": "user", "content": "I am planning a fictional robotics workshop."},
                    {"role": "assistant", "content": long_assistant},
                    {
                        "role": "user",
                        "content": "I selected the Copper Finch Inn as the Northbridge lodging for the robotics workshop.",
                    },
                ]
            )
        }
    }

    query = "What lodging did I select for the Northbridge robotics workshop?"
    bundles = build_evidence_bundles(results, chunks, query)
    rendered = render_evidence_with_coverage(
        bundles,
        max_chunk_chars=700,
        query=query,
    )

    assert "Copper Finch Inn" in rendered.text


def test_render_bound_keeps_whole_bundles_and_stops_before_budget():
    results = [_fact(f"fact {index}", chunk_id=f"chunk-{index}") for index in range(8)]
    chunks = {f"chunk-{index}": {"chunk_text": "x" * 200} for index in range(8)}

    bundles = build_evidence_bundles(results, chunks, "What happened?", max_bundles=8)
    rendered = render_evidence_with_coverage(bundles, max_chunk_chars=80, max_total_chars=500)

    assert len(rendered.text) <= 500
    assert "Bundle 1" in rendered.text
    # A budget cut must not emit a partial bundle header without its fact.
    assert rendered.text.count("Bundle ") == rendered.text.count("- Fact ")


def test_render_coverage_contains_only_admitted_compact_excerpts_at_exact_cap():
    results = [
        _fact("first fact", chunk_id="chunk-1", document_id="doc-1"),
        _fact("second fact", chunk_id="chunk-2", document_id="doc-2"),
    ]
    raw_chunks = {
        "chunk-1": {"chunk_text": json.dumps([{"role": "user", "content": "alpha " + "x " * 100}])},
        "chunk-2": {"chunk_text": json.dumps([{"role": "user", "content": "beta " + "y " * 100}])},
    }
    bundles = build_evidence_bundles(results, raw_chunks, "alpha beta")
    first_only = render_evidence_with_coverage(bundles[:1], max_chunk_chars=80, query="alpha beta")

    rendered = render_evidence_with_coverage(
        bundles,
        max_chunk_chars=80,
        max_total_chars=len(first_only.text),
        query="alpha beta",
    )

    assert rendered.text == first_only.text
    assert len(rendered.text) == len(first_only.text)
    assert set(rendered.covered_by_document) == {"doc-1"}
    excerpt = rendered.covered_by_document["doc-1"][0]
    assert excerpt in rendered.text
    assert excerpt != raw_chunks["chunk-1"]["chunk_text"]
    assert "doc-2" not in rendered.covered_by_document


def test_bundle_renderer_keeps_string_return_contract():
    bundles = build_evidence_bundles(
        [_fact("visible fact", chunk_id="chunk-1", document_id="doc-1")],
        {},
        "visible",
    )

    rendered = render_evidence_bundles(bundles)

    assert isinstance(rendered, str)
    assert "visible fact" in rendered


def test_bundle_order_uses_source_first_rank_even_when_later_facts_are_selected():
    results = [
        _fact("generic", chunk_id="chunk-a", document_id="doc-a"),
        _fact("beta match", chunk_id="chunk-b", document_id="doc-b"),
        _fact("beta amount $10", chunk_id="chunk-a", document_id="doc-a"),
        _fact("beta amount $20", chunk_id="chunk-a", document_id="doc-a"),
    ]

    bundles = build_evidence_bundles(results, {}, "beta amount", max_facts_per_bundle=2)

    assert [bundle.source_key for bundle in bundles] == ["chunk-a", "chunk-b"]
    assert bundles[0].first_rank == 1
    assert [fact["text"] for fact in bundles[0].facts] == ["beta amount $10", "beta amount $20"]


def test_long_turn_anchor_is_stable_across_hash_seeds_and_uses_latest_match():
    evaluation_root = Path(__file__).resolve().parents[2]
    script = """
import json
from benchmarks.longmemeval.evidence_bundles import _compact_chunk_text, _terms

raw = json.dumps([{
    "role": "user",
    "content": "alpha ANSWER_EARLY " + ("x " * 300) + " beta ANSWER_LATE",
}])
print(_compact_chunk_text(raw, _terms("alpha beta"), 80))
"""
    outputs = set()
    for seed in ("1", "2", "3", "4", "5", "6", "7", "8"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(path for path in (str(evaluation_root), env.get("PYTHONPATH", "")) if path)
        outputs.add(subprocess.check_output([sys.executable, "-c", script], env=env, text=True).strip())

    assert len(outputs) == 1
    assert "beta ANSWER_LATE" in outputs.pop()
