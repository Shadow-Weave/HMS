from __future__ import annotations

import pytest
from rsi_harness import EvaluationComparisonError, compare_evaluation_reports


def test_compare_strict_frozen_records_without_reading_answer_text():
    comparison = compare_evaluation_reports(
        {
            "benchmark": "longmemeval",
            "dataset_revision": "rev-1",
            "records": [
                {"query_id": "q-1", "passed": False, "answer": "gold must stay opaque"},
                {"query_id": "q-2", "passed": True},
            ],
        },
        {
            "benchmark": "longmemeval",
            "dataset_revision": "rev-1",
            "records": [
                {"query_id": "q-1", "passed": True},
                {"query_id": "q-2", "passed": True},
            ],
        },
    )
    assert comparison.delta == 0.5
    assert comparison.improved == 1
    assert comparison.regressed == 0
    assert comparison.unchanged == 1


def test_compare_existing_longmem_item_results():
    report = {
        "run_manifest": {"dataset": {"revision": "rev-1"}},
        "item_results": [
            {
                "item_id": "q-1",
                "metrics": {"detailed_results": [{"question_index": 0, "is_correct": True}]},
            },
            {
                "item_id": "q-2",
                "metrics": {"detailed_results": [{"question_index": 0, "is_correct": False}]},
            },
        ],
    }
    comparison = compare_evaluation_reports(
        report,
        {
            "records": [
                {"query_id": "q-1", "passed": True},
                {"query_id": "q-2", "passed": True},
            ]
        },
    )
    assert comparison.baseline_accuracy == 0.5
    assert comparison.evolved_accuracy == 1.0


def test_compare_rejects_mismatched_revision_and_duplicates():
    with pytest.raises(EvaluationComparisonError, match="dataset revisions"):
        compare_evaluation_reports(
            {"dataset_revision": "rev-a", "records": [{"query_id": "q", "passed": True}]},
            {"dataset_revision": "rev-b", "records": [{"query_id": "q", "passed": True}]},
        )
    with pytest.raises(EvaluationComparisonError, match="duplicate"):
        compare_evaluation_reports(
            {
                "records": [
                    {"query_id": "q", "passed": True},
                    {"query_id": "q", "passed": False},
                ]
            },
            {"records": [{"query_id": "q", "passed": True}]},
        )
