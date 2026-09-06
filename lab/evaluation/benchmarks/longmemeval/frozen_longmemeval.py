"""Read-only LongMemEval runner backed by frozen RSI snapshots.

Unlike the historical Retain -> Recall -> Answer -> Judge runner, this
command never opens a writable HMS memory engine and never reads documents,
chunks, live memory units, candidates, or verifier receipts.  A snapshot map
must be materialized by an evolution/development process before this command
starts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping

from rsi_harness import (
    EvaluationSnapshotBuilder,
    FrozenBenchmarkRunner,
    HarnessSnapshot,
    LongMemEvalAdapter,
)

from benchmarks.common.benchmark_runner import LLMAnswerEvaluator
from benchmarks.longmemeval.frozen_snapshot_backend import PostgresFrozenSnapshotBackend
from benchmarks.longmemeval.longmemeval_benchmark import LongMemEvalAnswerGenerator


def _load_snapshot_map(path: Path | None) -> dict[str, Mapping[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot map must be a JSON object")
    result: dict[str, Mapping[str, str]] = {}
    for question_id, value in payload.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"snapshot map entry {question_id!r} must be an object")
        bank_id = str(value.get("bank_id") or "").strip()
        snapshot_id = str(value.get("snapshot_id") or "").strip()
        if not bank_id or not snapshot_id:
            raise ValueError(f"snapshot map entry {question_id!r} needs bank_id and snapshot_id")
        result[str(question_id)] = {"bank_id": bank_id, "snapshot_id": snapshot_id}
    return result


def _snapshot_ref(
    question_id: str,
    snapshot_map: Mapping[str, Mapping[str, str]],
    *,
    snapshot_id: str | None,
    bank_prefix: str,
) -> tuple[str, str]:
    mapped = snapshot_map.get(question_id)
    if mapped is not None:
        return mapped["bank_id"], mapped["snapshot_id"]
    if not snapshot_id:
        raise ValueError(f"no frozen snapshot for {question_id!r}; provide --snapshot-map or --snapshot-id")
    return f"{bank_prefix}{question_id}", snapshot_id


async def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter = LongMemEvalAdapter()
    items = adapter.load(args.dataset_path, max_items=args.max_items)
    snapshot_map = _load_snapshot_map(args.snapshot_map)
    generator = LongMemEvalAnswerGenerator(context_format=args.context_format)
    evaluator = LLMAnswerEvaluator()
    judge_semaphore = asyncio.Semaphore(args.judge_concurrency)
    records: list[dict[str, Any]] = []

    for item in items:
        question_id = str(item["question_id"])
        bank_id, memory_snapshot_id = _snapshot_ref(
            question_id,
            snapshot_map,
            snapshot_id=args.snapshot_id,
            bank_prefix=args.bank_prefix,
        )
        backend = PostgresFrozenSnapshotBackend(args.database_url, bank_id)
        memory_snapshot = await backend.get_snapshot(memory_snapshot_id)
        runtime = adapter.runtime(item, evaluation_id=args.evaluation_id)
        skill_snapshot = HarnessSnapshot(
            args.skill_snapshot_id,
            None,
            manifest={"skill_complete": True, "skill_snapshot_id": args.skill_snapshot_id},
        )
        evaluation_snapshot = EvaluationSnapshotBuilder(benchmark_name="longmemeval").build(
            memory_snapshot,
            skill_snapshot,
            evaluation_id=f"{args.evaluation_id}:{question_id}",
            source_dataset_revision=args.dataset_revision,
            stream_id=runtime.query.stream_id,
            last_session_id=runtime.sessions[-1].session_id if runtime.sessions else None,
            last_source_position=runtime.sessions[-1].source_position if runtime.sessions else -1,
        )
        runner = FrozenBenchmarkRunner(
            args.firewall_factory(),
            backend,
            evaluation_snapshot,
            "longmemeval",
            token_budget=args.token_budget,
        )

        async def answerer(query, hits, view):
            del view
            recall_result = {
                "results": [
                    {
                        "id": hit.memory_id,
                        "text": hit.text,
                        "score": hit.score,
                        **dict(hit.metadata),
                    }
                    for hit in hits
                ],
                "chunks": {},
                "entities": {},
            }
            answer, _reasoning, _override = await generator.generate_answer(
                query.query,
                recall_result,
                query.asked_at,
                item.get("question_type"),
                bank_id=None,
            )
            return answer

        async def scorer(query, answer, hits):
            del hits
            correct, _reasoning, _duration = await evaluator.judge_answer(
                query.query,
                str(item.get("answer") or ""),
                answer,
                judge_semaphore,
                item.get("question_type"),
            )
            return correct

        report = await runner.run((runtime.query,), answerer=answerer, scorer=scorer)
        record = report.records[0]
        records.append(
            {
                "question_id": question_id,
                "evaluation_id": report.result.evaluation_id,
                "snapshot_id": memory_snapshot_id,
                "skill_snapshot_id": args.skill_snapshot_id,
                "passed": record.passed,
                "retrieved_memory_ids": list(record.retrieved_memory_ids),
            }
        )

    correct = sum(bool(record["passed"]) for record in records)
    return {
        "evaluation_id": args.evaluation_id,
        "benchmark": "longmemeval",
        "dataset_revision": args.dataset_revision,
        "skill_snapshot_id": args.skill_snapshot_id,
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("HMS_API_DATABASE_URL"))
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--snapshot-map", type=Path)
    parser.add_argument("--snapshot-id", default=os.environ.get("RSI_BENCHMARK_MEMORY_SNAPSHOT_ID"))
    parser.add_argument("--bank-prefix", default="longmemeval_")
    parser.add_argument("--skill-snapshot-id", default=os.environ.get("RSI_BENCHMARK_SKILL_SNAPSHOT_ID"))
    parser.add_argument("--evaluation-id", default=os.environ.get("RSI_BENCHMARK_EVALUATION_ID"))
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--context-format", choices=("json", "structured"), default="structured")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for field in ("database_url", "skill_snapshot_id", "evaluation_id"):
        if not getattr(args, field):
            raise SystemExit(f"{field.replace('_', '-')} is required")
    if args.judge_concurrency < 1 or args.token_budget < 1:
        raise SystemExit("judge-concurrency and token-budget must be positive")
    # A fresh firewall is intentionally created for each invocation.  The
    # command never receives an evolution store or a writable backend.
    from rsi_harness import BenchmarkFirewall

    args.firewall_factory = lambda: BenchmarkFirewall(benchmark_name="longmemeval")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
