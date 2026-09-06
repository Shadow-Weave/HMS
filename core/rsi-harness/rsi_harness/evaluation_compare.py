"""Leakage-neutral comparison of two frozen evaluation result artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class EvaluationComparisonError(ValueError):
    """Raised when two result artifacts cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Paired H0/H1 metrics without answer text or evaluator explanations."""

    benchmark: str
    dataset_revision: str | None
    baseline_accuracy: float
    evolved_accuracy: float
    delta: float
    baseline_total: int
    evolved_total: int
    paired_total: int
    improved: int
    regressed: int
    unchanged: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_evaluation_reports(
    baseline: Mapping[str, Any],
    evolved: Mapping[str, Any],
) -> EvaluationComparison:
    """Compare H0 and H1 artifacts over their common question IDs.

    The parser accepts the strict frozen runner's ``records`` format and the
    existing LongMemEval runner's ``item_results`` format.  It only consumes
    boolean correctness fields; answer, gold-answer, reasoning, and retrieved
    text are intentionally ignored.
    """

    benchmark = _shared_value(baseline, evolved, "benchmark") or "longmemeval"
    baseline_revision = _dataset_revision(baseline)
    evolved_revision = _dataset_revision(evolved)
    if baseline_revision and evolved_revision and baseline_revision != evolved_revision:
        raise EvaluationComparisonError("baseline and evolved artifacts use different dataset revisions")
    baseline_records = _records(baseline)
    evolved_records = _records(evolved)
    if not baseline_records or not evolved_records:
        raise EvaluationComparisonError("both artifacts need per-question correctness records")
    common_ids = sorted(set(baseline_records) & set(evolved_records))
    if not common_ids:
        raise EvaluationComparisonError("baseline and evolved artifacts have no common question IDs")

    improved = sum(not baseline_records[key] and evolved_records[key] for key in common_ids)
    regressed = sum(baseline_records[key] and not evolved_records[key] for key in common_ids)
    unchanged = len(common_ids) - improved - regressed
    baseline_accuracy = sum(baseline_records[key] for key in common_ids) / len(common_ids)
    evolved_accuracy = sum(evolved_records[key] for key in common_ids) / len(common_ids)
    return EvaluationComparison(
        benchmark=str(benchmark),
        dataset_revision=evolved_revision or baseline_revision,
        baseline_accuracy=baseline_accuracy,
        evolved_accuracy=evolved_accuracy,
        delta=evolved_accuracy - baseline_accuracy,
        baseline_total=len(baseline_records),
        evolved_total=len(evolved_records),
        paired_total=len(common_ids),
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
    )


def _shared_value(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> Any:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value and right_value and left_value != right_value:
        raise EvaluationComparisonError(f"artifacts use different {key} values")
    return left_value or right_value


def _dataset_revision(report: Mapping[str, Any]) -> str | None:
    direct = report.get("dataset_revision")
    if direct:
        return str(direct)
    manifest = report.get("run_manifest")
    if isinstance(manifest, Mapping):
        dataset = manifest.get("dataset")
        if isinstance(dataset, Mapping) and dataset.get("revision"):
            return str(dataset["revision"])
    return None


def _records(report: Mapping[str, Any]) -> dict[str, bool]:
    records = report.get("records")
    if isinstance(records, list):
        return _strict_records(records)
    item_results = report.get("item_results")
    if not isinstance(item_results, list):
        return {}
    output: dict[str, bool] = {}
    for item in item_results:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("item_id") or item.get("question_id") or "").strip()
        metrics = item.get("metrics")
        details = metrics.get("detailed_results") if isinstance(metrics, Mapping) else None
        if not item_id or not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            index = detail.get("question_index")
            key = item_id if index in (None, 0) else f"{item_id}:{index}"
            _insert_unique(output, key, detail.get("is_correct") is True)
    return output


def _strict_records(records: list[Any]) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        key = str(record.get("query_id") or record.get("question_id") or "").strip()
        if key:
            _insert_unique(output, key, record.get("passed") is True or record.get("is_correct") is True)
    return output


def _insert_unique(output: dict[str, bool], key: str, value: bool) -> None:
    if key in output:
        raise EvaluationComparisonError(f"duplicate result ID: {key}")
    output[key] = value
