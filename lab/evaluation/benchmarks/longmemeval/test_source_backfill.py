import json

from benchmarks.longmemeval.source_backfill import (
    render_source_chunk_snippets,
    render_source_snippets,
    select_missing_chunk_snippets,
    select_source_snippets,
)


def test_source_selection_keeps_answer_window_and_document_provenance():
    documents = {
        "doc-a": json.dumps(
            [
                {"role": "user", "content": "We discussed several fictional puzzle trials."},
                {"role": "assistant", "content": "Your best FableGrid total is 731 markers."},
                {"role": "user", "content": "Now let us discuss a different synthetic trial."},
            ]
        ),
        "doc-b": json.dumps([{"role": "user", "content": "A generic unrelated note."}]),
    }

    snippets = select_source_snippets(documents, "What is my best FableGrid total?", ["doc-a", "doc-b"])
    rendered = render_source_snippets(snippets)

    assert snippets
    assert snippets[0].document_id == "doc-a"
    assert "731 markers" in rendered
    assert "document=doc-a" in rendered


def test_source_selection_is_bounded_and_handles_non_json_documents():
    documents = {"doc-a": "plain retained text with 42", "doc-b": ""}
    snippets = select_source_snippets(
        documents,
        "What number was recorded?",
        ["doc-a", "doc-b"],
        max_snippets=1,
        max_chars_per_snippet=20,
    )

    assert len(snippets) == 1
    assert snippets[0].document_id == "doc-a"
    assert len(snippets[0].text) <= 20


def test_source_window_keeps_later_turn_after_long_assistant_turn():
    long_answer = "synthetic robotics workshop venue planning details " * 260
    documents = {
        "doc-a": json.dumps(
            [
                {"role": "user", "content": "I am planning a fictional robotics workshop."},
                {"role": "assistant", "content": long_answer},
                {
                    "role": "user",
                    "content": "I selected the Copper Finch Inn as the Northbridge lodging for the robotics workshop.",
                },
            ]
        )
    }

    snippets = select_source_snippets(
        documents,
        "What lodging did I select for the Northbridge robotics workshop?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=900,
    )

    assert snippets
    assert "Copper Finch Inn" in snippets[0].text
    assert "assistant:" in snippets[0].text


def test_covered_query_turn_does_not_trigger_unrelated_fallback():
    documents = {
        "doc-a": json.dumps(
            [
                {
                    "role": "user",
                    "content": "I selected the Copper Finch Inn as the Northbridge lodging for the robotics workshop.",
                },
                {"role": "assistant", "content": "The total cost was 42 dollars."},
            ]
        )
    }

    snippets = select_source_snippets(
        documents,
        "What lodging did I select for the Northbridge robotics workshop?",
        ["doc-a"],
        covered_chunks={
            "doc-a": [
                json.dumps(
                    [
                        {
                            "role": "user",
                            "content": "I selected the Copper Finch Inn as the Northbridge lodging for the robotics workshop.",
                        }
                    ]
                )
            ]
        },
    )

    assert snippets == []


def test_long_turn_compaction_prefers_dense_query_cluster_over_late_signal():
    long_turn = (
        "alpha appears once near the beginning. "
        + ("unrelated context " * 80)
        + "beta gamma beta gamma establish the relevant cluster. "
        + ("archive detail " * 80)
        + "The record was archived in 2025."
    )
    documents = {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])}

    snippets = select_source_snippets(
        documents,
        "What did alpha beta gamma establish?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=180,
    )

    assert snippets
    assert "beta gamma" in snippets[0].text
    assert "2025" not in snippets[0].text


def test_long_turn_compaction_prefers_query_anchor_over_unrelated_late_signal():
    long_turn = "alpha and omega started at $11. " + ("background detail " * 120) + "The resolved value was $73."
    documents = {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])}

    first = select_source_snippets(
        documents,
        "What did alpha and omega establish?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=180,
    )
    second = select_source_snippets(
        documents,
        "What did alpha and omega establish?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=180,
    )

    assert first and second
    assert first[0].text == second[0].text
    assert "$11" in first[0].text
    assert "$73" not in first[0].text


def test_long_turn_compaction_keeps_query_amount_before_late_year():
    long_turn = (
        "The fence repair cost $42 and was completed promptly. "
        + ("unrelated archive detail " * 120)
        + "The archive was migrated in 2025."
    )
    snippets = select_source_snippets(
        {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])},
        "How much did the fence repair cost?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=180,
    )

    assert snippets
    assert "$42" in snippets[0].text
    assert "2025" not in snippets[0].text


def test_long_turn_compaction_without_query_overlap_keeps_first_signal():
    long_turn = (
        "The recorded value was $42. " + ("unrelated archive detail " * 120) + "The archive was migrated in 2025."
    )
    snippets = select_source_snippets(
        {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])},
        "What was the amount?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=180,
    )

    assert snippets
    assert "$42" in snippets[0].text
    assert "2025" not in snippets[0].text


def test_long_turn_compaction_keeps_long_query_term_inside_window_boundary():
    long_term = "supercalifragilistic"
    long_turn = "alpha " + ("background " * 32) + long_term
    snippets = select_source_snippets(
        {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])},
        f"What is the {long_term} value?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=100,
    )

    assert snippets
    assert long_term in snippets[0].text


def test_long_turn_compaction_handles_budget_shorter_than_query_token():
    snippets = select_source_snippets(
        {"doc-a": json.dumps([{"role": "assistant", "content": "supercalifragilistic"}])},
        "supercalifragilistic?",
        ["doc-a"],
        max_snippets=1,
        max_chars_per_snippet=18,
    )

    assert snippets
    assert len(snippets[0].text) <= 18


def test_missing_chunk_recovery_keeps_exact_chunk_provenance():
    snippets = select_missing_chunk_snippets(
        [
            {
                "chunk_id": "chunk-7",
                "document_id": "doc-a",
                "chunk_index": 7,
                "chunk_text": json.dumps([{"role": "user", "content": "The repaired fence cost 42 dollars."}]),
            }
        ],
        "How much did the fence repair cost?",
    )

    rendered = render_source_chunk_snippets(snippets)
    assert snippets and snippets[0].chunk_id == "chunk-7"
    assert "42 dollars" in rendered
    assert "chunk_index=7" in rendered


def test_long_turn_prefix_is_not_mistaken_for_complete_coverage():
    long_turn = (
        "The fictional exhibit is a clockwork kelpie with " + ("background detail " * 80) + "an amber mosaic tail."
    )
    snippets = select_source_snippets(
        {"doc-a": json.dumps([{"role": "assistant", "content": long_turn}])},
        "What description was given for the clockwork kelpie's tail?",
        ["doc-a"],
        covered_chunks={"doc-a": [json.dumps([{"role": "assistant", "content": long_turn[:180]}])]},
        max_snippets=1,
    )

    assert snippets
    assert "amber mosaic tail" in snippets[0].text
