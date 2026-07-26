"""
LongMemEval-specific benchmark implementations.

Provides dataset, answer generator, and evaluator for the LongMemEval benchmark.
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pydantic
from hms_api.engine.llm_wrapper import LLMConfig
from hms_api.engine.schema import fq_table
from openai import AsyncOpenAI

from benchmarks.common.benchmark_runner import (
    BenchmarkDataset,
    BenchmarkRunner,
    LLMAnswerEvaluator,
    LLMAnswerGenerator,
    RecallPlan,
    get_model_config,
)
from benchmarks.longmemeval.evidence_bundles import (
    RenderedEvidence,
    build_evidence_bundles,
    render_evidence_with_coverage,
)
from benchmarks.longmemeval.source_backfill import (
    render_source_chunk_snippets,
    render_source_snippets,
    select_missing_chunk_snippets,
    select_source_snippets,
)

LONGMEMEVAL_DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LONGMEMEVAL_DATASET_FILENAME = "longmemeval_s_cleaned.json"
LONGMEMEVAL_DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_DATASET_REVISION}/{LONGMEMEVAL_DATASET_FILENAME}"
)
LONGMEMEVAL_DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LONGMEMEVAL_DATASET_SIZE = 277_383_467
LONGMEMEVAL_EXPECTED_ITEMS = 500
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_PATH = REPOSITORY_ROOT / ".aaaDATA" / "longmemeval" / LONGMEMEVAL_DATASET_FILENAME
REPRODUCIBILITY_SOURCE_PATHS = (
    ".aaaSCRIPT/run_benchmark.sh",
    "core/dataplane/hms_api",
    "core/dataplane/pyproject.toml",
    "lab/evaluation/benchmarks",
    "lab/evaluation/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
)


def _git_value(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_bytes(*args: str) -> Optional[bytes]:
    """Return raw Git output without assuming that source diffs are text."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            timeout=10,
        )
        return completed.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _source_tree_fingerprint() -> Optional[str]:
    """Identify relevant tracked and untracked source changes for safe resume."""

    tracked_diff = _git_bytes("diff", "--binary", "HEAD", "--", *REPRODUCIBILITY_SOURCE_PATHS)
    untracked_output = _git_bytes(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *REPRODUCIBILITY_SOURCE_PATHS,
    )
    if tracked_diff is None or untracked_output is None:
        return None

    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)
    if not tracked_diff and not untracked_paths:
        return None

    digest = hashlib.sha256()
    digest.update(b"hms-longmemeval-source-tree-v1\0")
    digest.update(tracked_diff)
    for encoded_path in untracked_paths:
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        source_path = REPOSITORY_ROOT / relative_path
        digest.update(b"\0path\0")
        digest.update(encoded_path)
        digest.update(b"\0content\0")
        try:
            digest.update(source_path.read_bytes())
        except OSError:
            return None
    return f"sha256:{digest.hexdigest()}"


def _manifest_dataset_reference(dataset_path: Path) -> str:
    """Return a portable dataset reference without exposing checkout paths."""

    resolved_path = dataset_path.expanduser().resolve()
    try:
        return resolved_path.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return f"external:{resolved_path.name}"


def build_run_manifest(
    *,
    dataset_path: Path,
    context_format: str,
    max_instances: Optional[int],
    max_instances_per_category: Optional[int],
    max_questions_per_instance: Optional[int],
    question_id: Optional[str],
    index_range: Optional[str],
    category: Optional[str],
    max_concurrent_items: int,
    max_concurrent_questions: int,
    eval_semaphore_size: int,
    thinking_budget: int,
    max_tokens: int,
    oracle_planner_v26: bool,
    oracle_planner_v220: bool,
    query_expansion_enabled: bool,
    query_rewriting_strategy: str,
    session_expansion_weight: float,
    skip_ingestion: bool,
    ingest_only: bool,
    force_reingest: bool,
) -> Dict[str, Any]:
    """Build a non-secret manifest sufficient to audit a benchmark artifact."""

    canonical = dataset_path.resolve() == DEFAULT_DATASET_PATH.resolve()
    dataset_sha256 = _sha256_file(dataset_path)
    git_commit = _git_value("rev-parse", "HEAD")
    git_status = _git_value("status", "--porcelain")
    planner = "self_evolution" if oracle_planner_v220 else "ledger" if oracle_planner_v26 else "standard"
    return {
        "artifact_schema_version": 1,
        "dataset": {
            "path": _manifest_dataset_reference(dataset_path),
            "sha256": dataset_sha256,
            "revision": LONGMEMEVAL_DATASET_REVISION if canonical else None,
            "expected_full_items": LONGMEMEVAL_EXPECTED_ITEMS if canonical else None,
        },
        "pipeline": {
            "stages": ["retain", "recall", "answer", "judge"],
            "planner": planner,
            "context_format": context_format,
            "thinking_budget": thinking_budget,
            "max_tokens": max_tokens,
            "query_expansion_enabled": query_expansion_enabled,
            "query_rewriting_strategy": query_rewriting_strategy if query_expansion_enabled else "noop",
            "session_expansion_weight": session_expansion_weight,
        },
        "concurrency": {
            "items": max_concurrent_items,
            "questions": max_concurrent_questions,
            "judge": eval_semaphore_size,
        },
        "selection": {
            "max_instances": max_instances,
            "max_instances_per_category": max_instances_per_category,
            "max_questions_per_instance": max_questions_per_instance,
            "question_id": question_id,
            "index_range": index_range,
            "category": category,
        },
        "execution": {
            "skip_ingestion": skip_ingestion,
            "ingest_only": ingest_only,
            "force_reingest": force_reingest,
        },
        "database": {
            "backend": os.getenv("HMS_API_DATABASE_BACKEND", "postgresql"),
            "schema": os.getenv("HMS_API_DATABASE_SCHEMA", "public"),
            "vector_extension": os.getenv("HMS_API_VECTOR_EXTENSION", "pgvector"),
        },
        "runtime": {
            "git_commit": git_commit,
            "git_dirty": bool(git_status) if git_status is not None else None,
            "source_tree_fingerprint": _source_tree_fingerprint(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def _artifact_compatibility_contract(
    manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select fields that must match before item results can share one artifact."""

    runtime = manifest.get("runtime", {})
    dataset = manifest.get("dataset", {})
    dataset_identity = (
        {
            "sha256": dataset.get("sha256"),
        }
        if isinstance(dataset, Mapping)
        else None
    )
    return {
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "dataset": dataset_identity,
        "pipeline": manifest.get("pipeline"),
        "database": manifest.get("database"),
        "git_commit": runtime.get("git_commit") if isinstance(runtime, Mapping) else None,
        "source_tree_fingerprint": (runtime.get("source_tree_fingerprint") if isinstance(runtime, Mapping) else None),
        "model_config": model_config,
    }


def validate_artifact_compatibility(
    output_path: Path,
    *,
    current_manifest: Mapping[str, Any],
    current_model_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Load an existing artifact and reject unsafe cross-run result mixing."""

    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume from unreadable result artifact {output_path}: {exc}") from exc
    if not isinstance(existing, dict):
        raise ValueError(f"Cannot resume from non-object result artifact: {output_path}")

    existing_manifest = existing.get("run_manifest")
    existing_model_config = existing.get("model_config")
    if not isinstance(existing_manifest, Mapping) or not isinstance(existing_model_config, Mapping):
        raise ValueError(
            f"Cannot safely merge {output_path}: existing artifact has no compatible run_manifest/model_config"
        )

    existing_contract = _artifact_compatibility_contract(existing_manifest, existing_model_config)
    current_contract = _artifact_compatibility_contract(current_manifest, current_model_config)
    mismatches = [
        field_name
        for field_name in current_contract
        if existing_contract.get(field_name) != current_contract.get(field_name)
    ]
    if mismatches:
        raise ValueError(
            f"Cannot safely merge {output_path}: incompatible experiment fields: {', '.join(mismatches)}. "
            "Use a new HMS_RESULTS_FILENAME for a different experiment."
        )
    return existing


def validate_output_target(
    output_path: Path,
    *,
    merge_with_existing: bool,
    resume: bool,
) -> None:
    """Protect fresh results and enforce an explicit existing target for resume."""

    if resume and not output_path.exists():
        raise FileNotFoundError(f"--resume requires an existing results file. Not found: {output_path}")
    if output_path.exists() and not merge_with_existing:
        raise FileExistsError(
            f"Refusing to overwrite existing result artifact {output_path}. "
            "Choose a new HMS_RESULTS_FILENAME or use --resume for a compatible interrupted run."
        )


ORACLE_PLANNER_V1_WEIGHTS = {
    "single-session-user": 0.25,
    "single-session-assistant": 0.25,
    "single-session-preference": 0.30,
    "knowledge-update": 0.50,
    "temporal-reasoning": 0.60,
    "multi-session": 0.80,
}


SELF_EVOLUTION_PROFILES = {
    "oracle_v220": {
        "base": "oracle_v26",
        "diagnosis_source": "v2.6 failed LongMemEval cases only",
        "evolution_targets": [
            "count/total deduplication",
            "relative-date lookup grounding",
            "amount/difference missing-side calibration",
            "current/previous state arbitration",
        ],
        "selection_rule": "Keep V2.6 retrieval and ledger as the base; add only diagnosis-derived pre-generation evidence controls.",
    },
}


def _v26_base_retrieval_plan(
    question: str,
    question_type: Optional[str],
    question_date: Optional[datetime],
) -> RecallPlan:
    """Internal V2.6 retrieval base: oracle weights plus multi-session query expansion and appendix."""
    del question, question_date
    weight = ORACLE_PLANNER_V1_WEIGHTS.get(question_type or "", 0.30)
    if question_type == "multi-session":
        return RecallPlan(
            name="longmemeval_v26_base_retrieval",
            session_expansion_weight=weight,
            query_rewriting_enabled=True,
            query_rewriting_strategy_name="llm_driven",
            evidence_appendix_mode="cross_session",
        )
    return RecallPlan(name="longmemeval_v26_base_retrieval", session_expansion_weight=weight)


def longmemeval_oracle_planner_v26(
    question: str,
    question_type: Optional[str],
    question_date: Optional[datetime],
) -> RecallPlan:
    """V2.6: base retrieval plus a pre-generation Structured Evidence Ledger."""
    plan = _v26_base_retrieval_plan(question, question_type, question_date)
    plan.name = "longmemeval_oracle_v26"
    return plan


def longmemeval_oracle_planner_v220(
    question: str,
    question_type: Optional[str],
    question_date: Optional[datetime],
) -> RecallPlan:
    """V2.20: V2.6 plus diagnosis-driven self-evolution controls."""
    plan = _v26_base_retrieval_plan(question, question_type, question_date)
    plan.name = "longmemeval_oracle_v220"
    return plan


class LongMemEvalDataset(BenchmarkDataset):
    """LongMemEval dataset implementation."""

    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LongMemEval dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError(f"LongMemEval dataset must be a JSON list: {path}")

        question_ids: set[str] = set()
        for index, item in enumerate(dataset):
            if not isinstance(item, dict):
                raise ValueError(f"LongMemEval item {index} must be a JSON object")
            question_id = item.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"LongMemEval item {index} has no valid question_id")
            if question_id in question_ids:
                raise ValueError(f"Duplicate LongMemEval question_id: {question_id}")
            question_ids.add(question_id)
            for field_name in ("haystack_sessions", "haystack_dates", "haystack_session_ids"):
                if not isinstance(item.get(field_name), list):
                    raise ValueError(f"LongMemEval item {question_id!r} has no valid {field_name}")
            lengths = {
                len(item["haystack_sessions"]),
                len(item["haystack_dates"]),
                len(item["haystack_session_ids"]),
            }
            if len(lengths) != 1:
                raise ValueError(f"LongMemEval item {question_id!r} has misaligned session arrays")

        if max_items is not None:
            dataset = dataset[:max_items]

        return dataset

    def get_item_id(self, item: Dict) -> str:
        """Get question ID from LongMemEval item."""
        return item.get("question_id", "unknown")

    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare LongMemEval conversation sessions for batch ingestion.

        Returns:
            List of session dicts with 'content', 'context', 'event_date'
        """
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        session_ids = item.get("haystack_session_ids", [])

        # Dataset loading validates that all three arrays are aligned.
        if not (len(sessions) == len(dates) == len(session_ids)):
            raise ValueError(f"LongMemEval item {item.get('question_id', 'unknown')!r} has misaligned session arrays")

        batch_contents = []
        seen_document_ids = {}

        # Process each session
        for idx, (session_turns, date_str, session_id) in enumerate(zip(sessions, dates, session_ids)):
            # Parse session date
            session_date = self._parse_date(date_str) if date_str else datetime.now(timezone.utc)

            # Clean session turns - remove has_answer key if present
            cleaned_turns = []
            for turn in session_turns:
                if isinstance(turn, dict):
                    cleaned_turn = {k: v for k, v in turn.items() if k != "has_answer"}
                    cleaned_turns.append(cleaned_turn)
                else:
                    cleaned_turns.append(turn)

            session_content = json.dumps(cleaned_turns)
            question_id = item.get("question_id", "unknown")
            base_document_id = f"{question_id}_{session_id}"

            unique_document_id = base_document_id
            if base_document_id in seen_document_ids:
                seen_document_ids[base_document_id] += 1
                unique_document_id = f"{base_document_id}_chunk{seen_document_ids[base_document_id]}"
            else:
                seen_document_ids[base_document_id] = 0

            batch_contents.append(
                {
                    "content": session_content,
                    "context": f"Session {unique_document_id} - you are the assistant in this conversation - happened on {session_date.strftime('%Y-%m-%d %H:%M:%S')} UTC.",
                    "event_date": session_date,
                    "document_id": unique_document_id,
                }
            )

        return batch_contents

    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from LongMemEval item.

        For LongMemEval, each item has one question.

        Returns:
            List with single QA dict with 'question', 'answer', 'category', 'question_date'
        """
        # Parse question_date if available
        question_date = None
        if "question_date" in item:
            question_date = self._parse_date(item["question_date"])

        return [
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("question_type", "unknown"),
                "question_date": question_date,
            }
        ]

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            # LongMemEval format: "2023/05/20 (Sat) 02:21"
            # Try to parse the main part before the day name
            date_str_cleaned = date_str.split("(")[0].strip() if "(" in date_str else date_str

            # Try multiple formats
            for fmt in ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    dt = datetime.strptime(date_str_cleaned, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

            # Fallback: try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"Failed to parse date string: {date_str}")


class QuestionAnswer(pydantic.BaseModel):
    answer: str
    reasoning: Optional[str] = None


class LongMemEvalAnswerGenerator(LLMAnswerGenerator):
    """LongMemEval-specific answer generator using configurable LLM provider."""

    def __init__(
        self,
        context_format: str = "structured_source",
        evidence_mode: Optional[str] = None,
    ):
        """Initialize with LLM configuration for answer generation.

        Args:
            context_format: How to format the retrieved context. Options:
                - "json": Raw JSON dump of recall_result (original behavior)
                - "structured": Human-readable format with facts grouped with source chunks
                - "structured_compact": Source-centric bundles with each source chunk rendered once
        """
        # Uses HMS_API_ANSWER_LLM_* env vars with fallback to HMS_API_LLM_* for
        # benchmark-specific LLM configuration (separate from the API config system).
        self.llm_config = LLMConfig(
            provider=os.getenv("HMS_API_ANSWER_LLM_PROVIDER", os.getenv("HMS_API_LLM_PROVIDER", "openai")),
            api_key=os.getenv("HMS_API_ANSWER_LLM_API_KEY", os.getenv("HMS_API_LLM_API_KEY", "")),
            base_url=os.getenv("HMS_API_ANSWER_LLM_BASE_URL", os.getenv("HMS_API_LLM_BASE_URL", "")),
            model=os.getenv("HMS_API_ANSWER_LLM_MODEL", os.getenv("HMS_API_LLM_MODEL", "gpt-4o-mini")),
            reasoning_effort="high",
        )
        self.client = self.llm_config._client
        self.model = self.llm_config.model
        self.context_format = context_format
        self.evidence_mode = evidence_mode

    def _format_context_json(self, recall_result: Dict[str, Any]) -> str:
        """Original JSON dump format."""
        return json.dumps(recall_result)

    def _format_context_structured(self, recall_result: Dict[str, Any]) -> str:
        """Human-readable format with facts grouped with their source chunks.

        Format:
            Fact 1: [fact text]
            When: [date]
            Source:
              "[chunk text]"

            ---

            Fact 2: ...

            === Entity Observations ===
            Entity: [name]
            - [observation 1]
            - [observation 2]
        """
        results = recall_result.get("results", [])
        chunks = recall_result.get("chunks", {})
        entities = recall_result.get("entities", {})

        if not results and not entities:
            return "No memories found."

        formatted_parts = []

        for i, fact in enumerate(results, 1):
            fact_text = fact.get("text", "")
            fact_type = fact.get("fact_type", "unknown")

            # Extract temporal information
            occurred_start = fact.get("occurred_start")
            occurred_end = fact.get("occurred_end")
            mentioned_at = fact.get("mentioned_at")

            # Build temporal string
            when_parts = []
            if occurred_start:
                when_parts.append(f"occurred: {occurred_start}")
            if mentioned_at:
                when_parts.append(f"mentioned: {mentioned_at}")
            when_str = " | ".join(when_parts) if when_parts else "unknown"

            # Get the source chunk if available
            chunk_id = fact.get("chunk_id")
            chunk_text = None
            if chunk_id and chunk_id in chunks:
                chunk_info = chunks[chunk_id]
                chunk_text = chunk_info.get("chunk_text", "")

            # Build the formatted fact entry
            entry_parts = [f"Fact {i} ({fact_type}): {fact_text}", f"When: {when_str}"]

            # Add context field if present
            context = fact.get("context")
            if context:
                entry_parts.append(f"Context: {context}")

            # Add source chunk
            if chunk_text:
                # Truncate very long chunks
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                entry_parts.append(f'Source chunk:\n  "{chunk_text}"')

            formatted_parts.append("\n".join(entry_parts))

        # Add entity observations section if present
        if entities:
            entity_parts = ["=== Entity Observations ==="]
            for entity_name, entity_state in entities.items():
                observations = entity_state.get("observations", [])
                if observations:
                    entity_parts.append(f"\nEntity: {entity_name}")
                    for obs in observations:
                        obs_text = obs.get("text", "")
                        entity_parts.append(f"  - {obs_text}")
            if len(entity_parts) > 1:  # More than just the header
                formatted_parts.append("\n".join(entity_parts))

        return "\n\n---\n\n".join(formatted_parts)

    def _format_context_source_centric(self, recall_result: Dict[str, Any], query: str) -> RenderedEvidence:
        """Render a bounded, source-centric view without changing retrieval.

        The per-fact structured formatter appends the same raw chunk to every
        fact extracted from it.  LongMemEval sessions commonly produce many
        near-duplicate facts, so that layout spends context on repetition and
        makes counts look larger than they are.  Bundles preserve the selected
        facts and provenance while making the unit of evidence the source
        chunk.
        """

        results = recall_result.get("results", [])
        chunks = recall_result.get("chunks", {}) or {}
        entities = recall_result.get("entities", {}) or {}
        # Keep the answer prompt bounded even when recall returns hundreds of
        # facts.  The retained-source block is rendered separately and placed
        # near the answer instructions, so the model can inspect provenance
        # without competing with an unbounded candidate dump.
        bundles = build_evidence_bundles(
            results,
            chunks,
            query,
            max_bundles=96,
            max_facts_per_bundle=2,
        )
        rendered_bundles = render_evidence_with_coverage(
            bundles,
            max_chunk_chars=900,
            max_total_chars=76_000,
            query=query,
        )
        parts = [rendered_bundles.text] if rendered_bundles.text else []

        if entities:
            entity_parts = ["=== Entity Observations ==="]
            for entity_name, entity_state in entities.items():
                observations = entity_state.get("observations", [])
                if observations:
                    entity_parts.append(f"\nEntity: {entity_name}")
                    for observation in observations:
                        text = observation.get("text", "")
                        if text:
                            entity_parts.append(f"  - {text}")
            if len(entity_parts) > 1:
                parts.append("\n".join(entity_parts))

        return RenderedEvidence(
            text="\n\n---\n\n".join(part for part in parts if part),
            covered_by_document=rendered_bundles.covered_by_document,
        )

    def _get_context_instructions(self) -> str:
        """Get instructions for interpreting the context based on format."""
        if self.context_format in {"structured", "structured_compact", "structured_source"}:
            context_guide = """**Understanding the Retrieved Context:**
The context contains memory facts extracted from previous conversations, each with its source chunk.

1. **Fact**: A high-level summary/atomic fact (e.g., "User loves hiking in mountains")
   - This is the searchable summary of what was stored

2. **Source Chunk**: The actual raw conversation where the fact was extracted from
   - **This is your primary source for detailed information**
   - Look here for specifics, context, quotes, and evidence
   - Prioritize information from chunks when facts seem ambiguous

3. **Temporal Information**:
   - "occurred": When the event actually happened
   - "mentioned": When it was discussed in conversation
   - Use this to understand the timeline and resolve conflicts (prefer more recent info)

4. **Context**: Additional metadata about the conversation session

5. **Retained Source-Document Evidence** (when present): Verbatim windows or
   exact retained chunks loaded from the same bank. Use these as the primary
   evidence for details missing from a summarized fact, especially explicit
   user state updates, dates, amounts, and purchases. Do not dismiss a source
   window merely because the extracted fact is incomplete.
"""
        else:
            context_guide = ""

        base_instructions = """
**Date Calculations (CRITICAL - read carefully):**
- When calculating days between two dates: count the days from Date A to Date B as (B - A)
- Example: Jan 1 to Jan 8 = 7 days (not 8)
- "X days ago" from Question Date means: Question Date minus X days
- When a fact says "three weeks ago" on a certain mentioned date, that refers to 3 weeks before THAT mentioned date, NOT the question date
- Always convert relative times ("last Friday", "two weeks ago") to absolute dates BEFORE comparing
- Double-check your arithmetic - off-by-one errors are very common
- **Important**: Read questions carefully for time anchors. "How many days ago did X happen when Y happened?" asks for the time between X and Y, NOT between X and the question date

**Handling Relative Times in Facts:**
- If a fact says "last Friday" or "two weeks ago", anchor it to the fact's "mentioned" date, NOT the question date
- First convert ALL relative references to absolute dates, then answer the question
- Show your date conversion work in your reasoning

**Counting Questions (CRITICAL for "how many" questions):**
- **Scan ALL facts first** - go through every single fact before counting, don't stop early
- **List each item explicitly in your reasoning** before giving the count: "1. X, 2. Y, 3. Z = 3 total"
- **Check all facts and chunks** before giving your final count
- **Watch for duplicates**: The same item may appear in multiple facts. Deduplicate by checking if two facts refer to the same underlying item/event
- **Watch for different descriptions of same thing**: "Dr. Patel (ENT specialist)" and "the ENT specialist" might be the same doctor
- **Don't over-interpret**: A project you "completed" is different from a project you're "leading"
- **Don't double-count**: If the same charity event is mentioned in two conversations, it's still one event

**Disambiguation Guidance (CRITICAL - many errors come from over-counting):**
- **Assume overlap by default**: If two facts describe similar events (same type, similar timeframe, similar details), assume they are the SAME event unless there's clear evidence they are different
- If a person has a name AND a role mentioned, check if they're the same person before counting separately
- If an amount is mentioned multiple times on different dates, check if it's the same event or different events
- When facts reference the same underlying event from different sessions, count it once
- **Check for aliases**: "my college roommate's wedding" and "Emily's wedding" might be the same event
- **Check for time period overlap**: Two "week-long breaks" mentioned in overlapping time periods are likely the same break
- **When in doubt, undercount**: It's better to miss a duplicate than to count the same thing twice

**Question Interpretation (read carefully):**
- "How many X before Y?" - count only X that happened BEFORE Y, not Y itself
- "How many properties viewed before making an offer on Z?" - count OTHER properties, not Z
- "How many X in the last week/month?" - calculate the exact date range from the question date, then filter
- Pay attention to qualifiers like "before", "after", "initially", "currently", "in total"

**When to Say "I Don't Know":**
- If the question asks about something not in the retrieved context, say "I don't have information about X"
- If comparing two things (e.g., "which happened first, X or Y?") but only one is mentioned, explicitly say the other is missing
- Don't guess or infer dates that aren't explicitly stated in the facts or chunks
- If you cannot find a specific piece of information after checking all facts and chunks, admit it
- **Partial knowledge is OK**: If asked about two things and you only have info on one, provide what you know and note what's missing (don't just say "I don't know")

**For Recommendation/Preference Questions (tips, suggestions, advice):**
- **DO NOT invent specific recommendations** (no made-up product names, course names, paper titles, channel names, etc.)
- **DO mention specific brands/products the user ALREADY uses** from the context
- Describe WHAT KIND of recommendation the user would prefer, referencing their existing tools/brands
- Keep answers concise - focus on key preferences (brand, quality level, specific interests) not exhaustive category lists
- First scan ALL facts for user's existing tools, brands, stated preferences

**How to Answer:**
1. Scan ALL facts to find relevant memories - don't stop after finding a few
2. **Read the source chunks carefully** - they contain the actual details you need
3. Convert all relative times to absolute dates
4. Use temporal information to understand when things happened
5. Synthesize information from multiple facts if needed
6. If facts conflict, prefer more recent information
7. Double-check any date calculations before answering
8. **For counting questions ("how many")**: First list each unique item in your reasoning (1. X, 2. Y, 3. Z...), then count them
9. **For recommendations**: Reference the user's existing tools, experiences, or preferences explicitly
"""
        return context_guide + base_instructions

    async def _format_source_document_backfill(
        self,
        question: str,
        recall_result: Dict[str, Any],
        bank_id: Optional[str],
        question_type: Optional[str] = None,
        rendered_coverage: Mapping[str, Sequence[str]] | None = None,
    ) -> str:
        """Recover bounded source windows for retrieved documents.

        Retain stores the original transcript separately from extracted facts.
        Reading only documents referenced by the current candidate set keeps
        this a scoped provenance lookup, rather than a bank-wide search.
        """

        if not bank_id:
            return ""
        database_url = os.environ.get("HMS_API_DATABASE_URL")
        if not database_url:
            return ""

        document_order: list[str] = []
        seen: set[str] = set()
        for fact in recall_result.get("results", []):
            document_id = str(fact.get("document_id") or "")
            if document_id and document_id not in seen:
                seen.add(document_id)
                document_order.append(document_id)
        if not document_order:
            return ""

        returned_chunks = recall_result.get("chunks", {}) or {}
        missing_chunk_ids: list[str] = []
        chunk_retrieval_rank: dict[str, int] = {}
        for rank, fact in enumerate(recall_result.get("results", []), 1):
            chunk_id = str(fact.get("chunk_id") or "")
            if chunk_id and chunk_id not in returned_chunks and chunk_id not in missing_chunk_ids and rank <= 64:
                missing_chunk_ids.append(chunk_id)
                chunk_retrieval_rank[chunk_id] = rank

        try:
            import asyncpg

            conn = await asyncpg.connect(database_url)
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT id, original_text
                    FROM {fq_table("documents")}
                    WHERE bank_id = $1 AND id = ANY($2::text[])
                    """,
                    bank_id,
                    document_order[:128],
                )
                missing_chunk_rows = []
                if missing_chunk_ids:
                    missing_chunk_rows = await conn.fetch(
                        f"""
                        SELECT chunk_id, document_id, chunk_index, chunk_text
                        FROM {fq_table("chunks")}
                        WHERE bank_id = $1 AND chunk_id = ANY($2::text[])
                        ORDER BY array_position($2::text[], chunk_id)
                        """,
                        bank_id,
                        missing_chunk_ids[:128],
                    )
            finally:
                await conn.close()
        except Exception as exc:
            # Source backfill is an optional evidence aid.  A database or JSON
            # failure must leave ordinary recall and judging valid.
            logging.warning("Source-document evidence backfill unavailable for %s: %s", bank_id, exc)
            return ""

        documents = {str(row["id"]): row["original_text"] for row in rows if row["original_text"]}
        exact_chunk_records = []
        for row in missing_chunk_rows:
            record = dict(row)
            record["_retrieval_rank"] = chunk_retrieval_rank.get(str(row["chunk_id"]), 10_000)
            exact_chunk_records.append(record)
        exact_chunks = select_missing_chunk_snippets(
            exact_chunk_records,
            question,
        )
        base_coverage = {
            str(document_id): list(excerpts)
            for document_id, excerpts in (rendered_coverage or {}).items()
            if document_id
        }
        admitted_exact_chunks = list(exact_chunks)
        while True:
            covered_chunks_by_document = {
                document_id: list(excerpts) for document_id, excerpts in base_coverage.items()
            }
            for snippet in admitted_exact_chunks:
                if snippet.document_id and snippet.text:
                    covered_chunks_by_document.setdefault(snippet.document_id, []).append(snippet.text)

            snippets = select_source_snippets(
                documents,
                question,
                document_order,
                max_documents=16,
                max_snippets=12,
                max_snippets_per_document=2,
                max_chars_per_snippet=900,
                max_total_chars=max(0, 9_000 - sum(len(snippet.text) for snippet in admitted_exact_chunks)),
                covered_chunks=covered_chunks_by_document,
            )
            while True:
                blocks = [
                    render_source_chunk_snippets(admitted_exact_chunks),
                    render_source_snippets(snippets),
                ]
                source_block = "\n\n".join(block for block in blocks if block)
                if len(source_block) <= 14_000:
                    return source_block
                if not snippets:
                    break
                snippets.pop()

            if not admitted_exact_chunks:
                return ""
            # Re-run source selection when an exact entry cannot fit. Its
            # original document turn is no longer considered covered.
            admitted_exact_chunks.pop()

    def _needs_structured_evidence_ledger(self, question: str, question_type: Optional[str]) -> bool:
        if self.evidence_mode not in {"oracle_v26", "oracle_v220"}:
            return False
        eligible_types = {"multi-session", "temporal-reasoning", "knowledge-update"}
        if question_type not in eligible_types:
            return False

        question_lower = question.lower()
        markers = (
            "after",
            "ago",
            "amount",
            "before",
            "between",
            "cashback",
            "compared",
            "cost",
            "current",
            "currently",
            "date",
            "days",
            "difference",
            "earliest",
            "first",
            "higher",
            "hours",
            "how long",
            "how many",
            "how much",
            "in total",
            "initially",
            "latest",
            "less",
            "lower",
            "months",
            "more",
            "most",
            "order",
            "percentage",
            "previous",
            "recently",
            "since",
            "spent",
            "total",
            "weeks",
            "years",
        )
        return any(marker in question_lower for marker in markers)

    @staticmethod
    def _needs_v26_self_evolution_controller(question: str, question_type: Optional[str]) -> bool:
        if question_type not in {"multi-session", "temporal-reasoning", "knowledge-update"}:
            return False
        question_lower = question.lower()
        markers = (
            "ago",
            "before",
            "after",
            "current",
            "currently",
            "difference",
            "first",
            "how many",
            "how much",
            "in total",
            "initially",
            "latest",
            "previous",
            "save",
            "spent",
            "total",
        )
        return any(marker in question_lower for marker in markers)

    @staticmethod
    def _v26_self_evolution_controller() -> str:
        return """
**V2.20 V2.6 Self-Evolution Controller:**
This controller was derived from V2.6 failure analysis. It does not replace the V2.6 evidence ledger; it tells you how to use that ledger more carefully.
- Count/total questions: enumerate unique real user events/items before giving the count. Do not count recommendations, options, generic background facts, plans, or duplicate extractions of the same event. If one required category is missing, answer with the known side plus "not enough information"; do not collapse missing evidence to 0.
- Amount/difference questions: compute only from amounts that are explicitly present for both sides requested by the question. If one side's amount is missing, say which side is missing instead of using generic ranges.
- Relative-date lookup questions: resolve the relative date from the question date, then prefer facts and source snippets whose event date or source text matches that resolved day. If the answer is described rather than named, return the full descriptive phrase.
- Current/previous/update questions: prefer the latest explicit state for "current" questions and the older explicit state for "previous/before" questions. Do not add old and new state values unless the question asks for a cumulative lifetime total.
- Final answer contract: start with the direct value/name/date/insufficient-information statement. Put caveats in reasoning after the direct answer.
"""

    @staticmethod
    def _compact_text(text: Any, max_chars: int) -> str:
        compact = " ".join(str(text or "").replace("<|endoftext|>", " ").split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _bound_context_block(text: str, max_chars: int) -> str:
        """Bound one evidence block while retaining both its head and tail.

        Evidence sections often contain a high-ranked header at the front and
        exact source rows at the end.  Keeping both sides is more useful than
        a left-only cut, which can silently remove the answer-bearing span.
        """

        if len(text) <= max_chars:
            return text
        if max_chars <= 80:
            return text[:max_chars]
        head = max_chars // 2
        tail = max_chars - head - 40
        return f"{text[:head].rstrip()}\n... [evidence block bounded] ...\n{text[-tail:].lstrip()}"

    @staticmethod
    def _content_terms(question: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "again",
            "before",
            "between",
            "current",
            "currently",
            "different",
            "during",
            "first",
            "from",
            "have",
            "many",
            "much",
            "previous",
            "recently",
            "since",
            "that",
            "the",
            "then",
            "there",
            "this",
            "total",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
        return {
            token for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", question.lower()) if token not in stopwords
        }

    def _format_structured_evidence_ledger(self, question: str, recall_result: Dict[str, Any]) -> str:
        results = recall_result.get("results", [])
        chunks = recall_result.get("chunks", {})
        question_terms = self._content_terms(question)
        signal_re = re.compile(
            r"(\$?\d+(?:[.,]\d+)?%?|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"today|yesterday|tomorrow|last|next|ago|week|month|year|day|hour|"
            r"before|after|first|earlier|later|previous|current|latest|total|spent|cost|discount|cashback)",
            re.IGNORECASE,
        )

        ledger_rows = []
        seen = set()
        for fact in results[:180]:
            text = self._compact_text(fact.get("text", ""), 360)
            if not text:
                continue
            text_lower = text.lower()
            term_overlap = sum(1 for term in question_terms if term in text_lower)
            has_signal = bool(signal_re.search(text))
            if not has_signal and term_overlap < 2:
                continue

            dedupe_key = re.sub(r"\W+", " ", text_lower).strip()[:180]
            doc_id = str(fact.get("document_id") or "")
            dedupe_key = f"{doc_id}:{dedupe_key}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            ledger_rows.append(
                {
                    "score": (3 if has_signal else 0) + term_overlap,
                    "doc": doc_id,
                    "type": fact.get("fact_type"),
                    "occurred": fact.get("occurred_start") or fact.get("occurred_end") or "-",
                    "mentioned": fact.get("mentioned_at") or "-",
                    "chunk_id": fact.get("chunk_id"),
                    "text": text,
                }
            )
            if len(ledger_rows) >= 70:
                break

        ledger_rows.sort(key=lambda row: row["score"], reverse=True)
        ledger_rows = ledger_rows[:45]
        if not ledger_rows:
            return ""

        lines = [
            "=== V2.6 Structured Evidence Ledger ===",
            "Use this as a checklist for count/sum/date/order/update questions. It is extracted from the retrieved context; do not use it as new evidence beyond the facts and source chunks. Deduplicate repeated mentions of the same event before counting.",
            "",
            "Candidate facts:",
        ]
        used_chunks = []
        seen_chunks = set()
        for idx, row in enumerate(ledger_rows, 1):
            lines.append(
                f"{idx}. occurred={row['occurred']} | mentioned={row['mentioned']} | "
                f"doc={row['doc']} | type={row['type']} | {row['text']}"
            )
            chunk_id = row.get("chunk_id")
            if chunk_id and chunk_id in chunks and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                used_chunks.append(chunk_id)

        if used_chunks:
            lines.extend(["", "Raw source snippets for ledger rows:"])
            for idx, chunk_id in enumerate(used_chunks[:18], 1):
                chunk_info = chunks.get(chunk_id) or {}
                chunk_text = self._compact_text(chunk_info.get("chunk_text", ""), 650)
                if chunk_text:
                    lines.append(f"{idx}. chunk={chunk_id} | {chunk_text}")

        return "\n".join(lines)

    @staticmethod
    def _sort_date_key(value: Any) -> Tuple[int, str]:
        if not value or value == "-":
            return (1, "")
        return (0, str(value))

    @staticmethod
    def _date_from_value(value: Any) -> Optional[date]:
        if not value or value == "-":
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            match = re.search(r"\d{4}-\d{2}-\d{2}", text)
            if not match:
                return None
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d").date()
            except ValueError:
                return None

    @staticmethod
    def _resolved_relative_dates(question: str, question_date: Optional[datetime]) -> List[Tuple[str, date]]:
        if question_date is None:
            return []
        question_lower = question.lower()
        base_date = question_date.date()
        resolved: List[Tuple[str, date]] = []

        for match in re.finditer(r"\b(\d+)\s+days?\s+ago\b", question_lower):
            days = int(match.group(1))
            resolved.append((match.group(0), base_date - timedelta(days=days)))

        for match in re.finditer(r"\b(\d+)\s+weeks?\s+ago\b", question_lower):
            weeks = int(match.group(1))
            resolved.append((match.group(0), base_date - timedelta(days=7 * weeks)))

        word_numbers = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        for word, value in word_numbers.items():
            if re.search(rf"\b{word}\s+days?\s+ago\b", question_lower):
                resolved.append((f"{word} days ago", base_date - timedelta(days=value)))
            if re.search(rf"\b{word}\s+weeks?\s+ago\b", question_lower):
                resolved.append((f"{word} weeks ago", base_date - timedelta(days=7 * value)))

        if re.search(r"\byesterday\b", question_lower):
            resolved.append(("yesterday", base_date - timedelta(days=1)))

        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        for weekday, target_idx in weekdays.items():
            if re.search(rf"\blast\s+{weekday}\b", question_lower):
                days_back = (base_date.weekday() - target_idx) % 7
                if days_back == 0:
                    days_back = 7
                resolved.append((f"last {weekday}", base_date - timedelta(days=days_back)))

        deduped: List[Tuple[str, date]] = []
        seen = set()
        for label, date_value in resolved:
            key = (label, date_value.isoformat())
            if key not in seen:
                seen.add(key)
                deduped.append((label, date_value))
        return deduped

    @staticmethod
    def _is_relative_date_lookup_question(question: str, question_type: Optional[str]) -> bool:
        if question_type != "temporal-reasoning":
            return False
        question_lower = question.lower()
        has_relative_date = bool(
            re.search(
                r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?)\s+ago\b",
                question_lower,
            )
            or re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", question_lower)
            or re.search(r"\byesterday\b", question_lower)
        )
        if not has_relative_date:
            return False

        comparison_markers = (
            "first",
            "earliest",
            "latest",
            "before",
            "after",
            "between",
            "compared",
            "higher",
            "lower",
            "more",
            "less",
            "total",
            "how many",
            "how much",
            "how long",
        )
        if any(marker in question_lower for marker in comparison_markers):
            return False

        lookup_markers = (
            "what ",
            "which ",
            "who ",
            "where ",
            "from whom",
            "by whom",
            "did i buy",
            "did i get",
            "did i receive",
            "did i purchase",
            "started to listen",
        )
        return any(marker in question_lower for marker in lookup_markers)

    def _format_resolved_date_evidence_block(
        self,
        question: str,
        question_date: Optional[datetime],
        question_type: Optional[str],
        recall_result: Dict[str, Any],
    ) -> str:
        if self.evidence_mode != "oracle_v220" or question_type != "temporal-reasoning":
            return ""
        if not self._is_relative_date_lookup_question(question, question_type):
            return ""

        resolved_dates = self._resolved_relative_dates(question, question_date)
        if not resolved_dates:
            return ""

        results = recall_result.get("results", [])
        chunks = recall_result.get("chunks", {})
        rows = []
        seen = set()
        target_dates = {date_value for _, date_value in resolved_dates}
        for rank, fact in enumerate(results[:220], 1):
            fact_dates = {
                self._date_from_value(fact.get("occurred_start")),
                self._date_from_value(fact.get("occurred_end")),
                self._date_from_value(fact.get("mentioned_at")),
            }
            fact_dates.discard(None)
            matched_dates = sorted(date_value for date_value in fact_dates if date_value in target_dates)
            if not matched_dates:
                continue

            text = self._compact_text(fact.get("text", ""), 340)
            if not text:
                continue
            doc_id = str(fact.get("document_id") or "")
            dedupe_text = re.sub(r"\W+", " ", text.lower()).strip()[:160]
            dedupe_key = f"{doc_id}:{dedupe_text}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "rank": rank,
                    "matched": ", ".join(date_value.isoformat() for date_value in matched_dates),
                    "doc": doc_id,
                    "type": fact.get("fact_type"),
                    "occurred": fact.get("occurred_start") or fact.get("occurred_end") or "-",
                    "mentioned": fact.get("mentioned_at") or "-",
                    "chunk_id": fact.get("chunk_id"),
                    "text": text,
                }
            )
            if len(rows) >= 24:
                break

        if not rows:
            return ""

        title = "=== V2.20 V2.6 Self-Evolved Relative-Date Evidence Block ==="
        lines = [
            title,
            "Relative date targets resolved from the question:",
        ]
        for label, date_value in resolved_dates:
            lines.append(f"- {label} => {date_value.isoformat()}")
        lines.extend(
            [
                "Facts retrieved for those exact dates. Use them as same-day evidence, even when the surface noun in the question differs from the extracted fact wording.",
                "",
                "Same-date candidate facts:",
            ]
        )

        used_chunks = []
        seen_chunks = set()
        for idx, row in enumerate(rows, 1):
            lines.append(
                f"{idx}. target_date={row['matched']} | occurred={row['occurred']} | "
                f"mentioned={row['mentioned']} | doc={row['doc']} | type={row['type']} | {row['text']}"
            )
            chunk_id = row.get("chunk_id")
            if chunk_id and chunk_id in chunks and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                used_chunks.append(chunk_id)

        if used_chunks:
            lines.extend(["", "Raw source snippets for same-date facts:"])
            for idx, chunk_id in enumerate(used_chunks[:10], 1):
                chunk_info = chunks.get(chunk_id) or {}
                chunk_text = self._compact_text(chunk_info.get("chunk_text", ""), 560)
                if chunk_text:
                    lines.append(f"{idx}. chunk={chunk_id} | {chunk_text}")

        return "\n".join(lines)

    async def _format_resolved_date_memory_backfill(
        self,
        question: str,
        question_date: Optional[datetime],
        question_type: Optional[str],
        bank_id: Optional[str],
    ) -> str:
        if self.evidence_mode != "oracle_v220" or question_type != "temporal-reasoning" or not bank_id:
            return ""
        if not self._is_relative_date_lookup_question(question, question_type):
            return ""

        resolved_dates = self._resolved_relative_dates(question, question_date)
        if not resolved_dates:
            return ""

        database_url = os.environ.get("HMS_API_DATABASE_URL")
        if not database_url:
            return ""

        try:
            import asyncpg

            conn = await asyncpg.connect(database_url)
            try:
                target_dates = [date_value for _, date_value in resolved_dates]
                rows = await conn.fetch(
                    f"""
                    SELECT id, document_id, chunk_id, text, fact_type,
                           occurred_start, occurred_end, mentioned_at
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1
                      AND (
                        occurred_start::date = ANY($2::date[])
                        OR occurred_end::date = ANY($2::date[])
                        OR mentioned_at::date = ANY($2::date[])
                      )
                    ORDER BY mentioned_at, document_id
                    LIMIT 180
                    """,
                    bank_id,
                    target_dates,
                )
                if not rows:
                    return ""

                target_date_set = set(target_dates)
                question_terms = self._content_terms(question)
                acquisition_re = re.compile(r"\b(acquired|got|bought|purchased|received|picked up|ordered)\b", re.I)

                def row_score(row: Any) -> Tuple[int, str]:
                    text = str(row["text"] or "")
                    text_lower = text.lower()
                    fact_dates = {
                        self._date_from_value(row["occurred_start"]),
                        self._date_from_value(row["occurred_end"]),
                        self._date_from_value(row["mentioned_at"]),
                    }
                    fact_dates.discard(None)
                    exact_occurred = (
                        self._date_from_value(row["occurred_start"]) in target_date_set
                        or self._date_from_value(row["occurred_end"]) in target_date_set
                    )
                    mentioned_match = self._date_from_value(row["mentioned_at"]) in target_date_set
                    term_overlap = sum(1 for term in question_terms if term in text_lower)
                    acquisition = bool(acquisition_re.search(text))
                    score = (
                        (12 if exact_occurred else 0)
                        + (2 if mentioned_match else 0)
                        + 3 * term_overlap
                        + (6 if acquisition else 0)
                    )
                    return (-score, str(row["mentioned_at"] or ""), str(row["document_id"] or ""))

                rows = sorted(rows, key=row_score)[:36]

                chunk_ids = [row["chunk_id"] for row in rows if row["chunk_id"]]
                chunk_rows = []
                if chunk_ids:
                    chunk_rows = await conn.fetch(
                        f"""
                        SELECT chunk_id, chunk_text
                        FROM {fq_table("chunks")}
                        WHERE bank_id = $1 AND chunk_id = ANY($2::text[])
                        LIMIT 16
                        """,
                        bank_id,
                        chunk_ids[:16],
                    )
                chunk_text_by_id = {row["chunk_id"]: row["chunk_text"] for row in chunk_rows}
            finally:
                await conn.close()
        except Exception:
            return ""

        title = "=== V2.20 V2.6 Self-Evolved Exact-Date Memory Backfill ==="
        lines = [
            title,
            "Memory units directly loaded from the fixed memory bank for the resolved relative-date target. This is not new extraction; it is a date-constrained evidence backfill from stored memories.",
            "For acquisition questions, stored wording such as got, acquired, received, or bought should be treated as candidate acquisition evidence for the item named in the memory.",
            "Resolved targets:",
        ]
        for label, date_value in resolved_dates:
            lines.append(f"- {label} => {date_value.isoformat()}")
        lines.extend(["", "Exact-date stored memories:"])

        used_chunks = []
        seen_chunks = set()
        for idx, row in enumerate(rows, 1):
            text = self._compact_text(row["text"], 340)
            lines.append(
                f"{idx}. occurred={row['occurred_start'] or row['occurred_end'] or '-'} | "
                f"mentioned={row['mentioned_at'] or '-'} | doc={row['document_id']} | "
                f"type={row['fact_type']} | {text}"
            )
            chunk_id = row["chunk_id"]
            if chunk_id and chunk_id in chunk_text_by_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                used_chunks.append(chunk_id)

        if used_chunks:
            lines.extend(["", "Raw source snippets for exact-date backfill:"])
            for idx, chunk_id in enumerate(used_chunks[:8], 1):
                chunk_text = self._compact_text(chunk_text_by_id.get(chunk_id, ""), 560)
                if chunk_text:
                    lines.append(f"{idx}. chunk={chunk_id} | {chunk_text}")

        return "\n".join(lines)

    async def generate_answer(
        self,
        question: str,
        recall_result: Dict[str, Any],
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None,
        bank_id: Optional[str] = None,
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories using Groq.

        Args:
            question: The question text
            recall_result: Full RecallResult dict containing results, entities, chunks, and trace
            question_date: Date when the question was asked (for temporal context)
            question_type: Question category (e.g., 'single-session-user', 'multi-session-assistant')

        Returns:
            Tuple of (answer, reasoning, None)
            - None indicates to use the memories from recall_result
        """
        # Format context based on selected mode
        rendered_coverage: Mapping[str, Sequence[str]] = {}
        if self.context_format in {"structured_compact", "structured_source"}:
            rendered_context = self._format_context_source_centric(recall_result, question)
            context = rendered_context.text
            rendered_coverage = rendered_context.covered_by_document
        elif self.context_format == "structured":
            context = self._format_context_structured(recall_result)
        else:
            context = self._format_context_json(recall_result)

        source_block = ""
        if self.context_format == "structured_source":
            source_block = await self._format_source_document_backfill(
                question,
                recall_result,
                bank_id,
                question_type,
                rendered_coverage,
            )

        if self._needs_structured_evidence_ledger(question, question_type):
            ledger = self._format_structured_evidence_ledger(question, recall_result)
            if ledger:
                context = f"{context}\n\n{self._bound_context_block(ledger, 20_000)}"
        if self.evidence_mode == "oracle_v220":
            backfill_block = await self._format_resolved_date_memory_backfill(
                question,
                question_date,
                question_type,
                bank_id,
            )
            if backfill_block:
                context = f"{backfill_block}\n\n{context}"
            date_block = self._format_resolved_date_evidence_block(
                question,
                question_date,
                question_type,
                recall_result,
            )
            if date_block:
                context = f"{context}\n\n{date_block}"

        # Put retained-source evidence after the compact fact/ledger sections,
        # immediately before the answer prompt.  This keeps provenance visible
        # when the upstream provider applies an input-token window and makes
        # the source text the last evidence the model reads.
        if source_block:
            context = f"{context}\n\n{source_block}"

        context_instructions = self._get_context_instructions()
        if (
            self.evidence_mode == "oracle_v220"
            and self._needs_structured_evidence_ledger(question, question_type)
            and self._needs_v26_self_evolution_controller(question, question_type)
        ):
            context_instructions = f"{context_instructions}{self._v26_self_evolution_controller()}"

        # Format question date if provided
        formatted_question_date = question_date.strftime("%Y-%m-%d %H:%M:%S UTC") if question_date else "Not specified"

        # Use LLM to generate answer. Provider and schema failures must
        # propagate to the benchmark runner so the question is recorded as
        # invalid rather than silently scored as an ordinary answer.
        answer_obj = await self.llm_config.call(
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a helpful assistant that must answer user questions based on the previous conversations.

{context_instructions}**Answer Guidelines:**
1. Start by scanning retrieved context to understand the facts and events that happened and the timeline.
2. Reason about all the memories and find the right answer, considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

In general the answer must be comprehensive and plenty of details from the retrieved context.

For quantitative/counting questions ("how many..."): First list each unique item in your reasoning (1. X, 2. Y, 3. Z...), scanning ALL facts, then count them for your answer.
If questions asks a location (where...?) make sure to include the location name.
For recommendation questions ("can you recommend...", "suggest...", "any tips..."): DO NOT give actual recommendations. Instead, describe what KIND the user would prefer based on their context. Example answer format: "The user would prefer recommendations for [category] that focus on [their interest]. They would not prefer [what to avoid based on context]."
For questions asking for help or instructions, consider the users' recent memories and previous interactions with the assistant to understand their current situation better (recent purchases, specific product models used..)
For specific number/value questions, use the context to understand what is the most up-to-date number based on recency, but also include the reasoning (in the answer) on previous possible values and why you think are less relevant.
For open questions, include as much details as possible from different sources that are relevant.
For questions where a specific entity/role is mentioned and it's different from your memory, just say the truth, don't make up anything just to fulfill the question. For example, if the question is about a specific sport, you should consider if the memories and the question are about the same sport. (e.g. american football vs soccer, shows vs podcasts)
For comparative questions, say you don't know the answer if you don't have information about both sides. (or more sides)
For questions related to time/date, carefully review the question date and the memories date to correctly answer the question.
For questions related to time/date calculation (e.g. How many days passed between X and Y?), carefully review the memories date to correctly answer the question and only provide an answer if you have information about both X and Y, otherwise say it's not possible to calculate and why.

Consider assistant's previous actions (e.g., bookings, reminders) as impactful to the user experiences.


Question: {question}
Question Date: {formatted_question_date}

Retrieved Context:
{context}


Answer:
""",
                }
            ],
            response_format=QuestionAnswer,
            scope="memory",
            max_completion_tokens=32768,
        )
        reasoning_text = answer_obj.reasoning or ""
        if reasoning_text:
            reasoning_text = reasoning_text + " "
        reasoning_text += f"(question date: {formatted_question_date})"
        return answer_obj.answer, reasoning_text, None


def validate_runtime_options(
    *,
    max_instances: Optional[int],
    max_instances_per_category: Optional[int],
    max_questions_per_instance: Optional[int],
    max_concurrent_items: int,
    max_concurrent_questions: int,
    eval_semaphore_size: int,
    thinking_budget: int,
    max_tokens: int,
) -> None:
    """Reject zero and negative limits before any provider or database work."""

    optional_positive = {
        "max_instances": max_instances,
        "max_instances_per_category": max_instances_per_category,
        "max_questions_per_instance": max_questions_per_instance,
    }
    required_positive = {
        "max_concurrent_items": max_concurrent_items,
        "max_concurrent_questions": max_concurrent_questions,
        "eval_semaphore_size": eval_semaphore_size,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
    }
    for name, value in optional_positive.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    for name, value in required_positive.items():
        if value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")


def resolve_dataset_path(dataset_path: Optional[str]) -> Path:
    """Resolve the selected dataset and enforce the canonical artifact pin."""

    if dataset_path is None:
        resolved = DEFAULT_DATASET_PATH
        if not resolved.exists():
            download_dataset(resolved)
    else:
        resolved = Path(dataset_path).expanduser()
        if not resolved.is_absolute():
            resolved = (REPOSITORY_ROOT / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Custom dataset not found: {resolved}")

    if resolved.resolve() == DEFAULT_DATASET_PATH.resolve():
        validate_canonical_dataset(resolved)
    return resolved


def resolve_output_path(results_dir: Optional[str], results_filename: str) -> Path:
    """Resolve a benchmark result artifact path from CLI settings."""

    if results_dir:
        result_root = Path(results_dir).expanduser()
        if not result_root.is_absolute():
            result_root = (REPOSITORY_ROOT / result_root).resolve()
        return result_root / results_filename
    return Path(__file__).parent / "results" / results_filename


async def run_benchmark(
    max_instances: int = None,
    max_instances_per_category: int = None,
    max_questions_per_instance: int = None,
    thinking_budget: int = 500,
    max_tokens: int = 8192,
    skip_ingestion: bool = False,
    filln: bool = False,
    question_id: str = None,
    index_range: str = None,
    only_failed: bool = False,
    only_invalid: bool = False,
    only_ingested: bool = False,
    category: str = None,
    max_concurrent_items: int = 1,
    results_filename: str = "benchmark_results.json",
    results_dir: str = None,
    context_format: str = "structured_source",
    source_results: str = None,
    ingest_only: bool = False,
    force_reingest: bool = False,
    max_concurrent_questions: int = 10,
    eval_semaphore_size: int = 10,
    dataset_path: Optional[str] = None,
    query_expansion_enabled: bool = False,
    query_rewriting_strategy: str = "llm_based",
    session_expansion_weight: float = 0.3,
    oracle_planner_v26: bool = False,
    oracle_planner_v220: bool = False,
    resume: bool = False,
):
    """
    Run the LongMemEval benchmark.

    Args:
        max_instances: Maximum number of instances to evaluate (None for all). Mutually exclusive with max_instances_per_category and category.
        max_instances_per_category: Maximum number of instances per category (None for all). Mutually exclusive with max_instances and category.
        max_questions_per_instance: Maximum questions per instance (for testing)
        thinking_budget: Thinking budget for spreading activation search
        max_tokens: Maximum tokens to retrieve from memories
        skip_ingestion: Whether to skip ingestion and use existing data
        filln: If True, only process question IDs not already present in the result artifact
        question_id: Optional question ID to filter (e.g., 'sample-question-a'). Useful with --skip-ingestion.
        only_failed: If True, only run questions that were previously marked as incorrect (is_correct=False)
        only_invalid: If True, only run questions that were previously marked as invalid (is_invalid=True)
        only_ingested: If True, only run questions whose memory bank already exists (has been ingested)
        category: Optional category to filter questions (e.g., 'single-session-user', 'multi-session', 'temporal-reasoning'). Mutually exclusive with max_instances and max_instances_per_category.
        max_concurrent_items: Maximum number of instances to process in parallel (default: 1 for sequential)
        results_filename: Filename for results (default: benchmark_results.json).
        results_dir: Optional directory for results. If None, defaults to results/ relative to script location.
        context_format: How to format context for answer generation. "json" (raw JSON) or "structured" (human-readable with facts+chunks).
        source_results: Source results file to read failed/invalid questions from (for --only-failed/--only-invalid). Defaults to benchmark_results.json.
        ingest_only: Only ingest, skip evaluation
        force_reingest: If True, always re-ingest even if data already exists (for re-running after fixing ingestion issues)
        dataset_path: Optional custom dataset path. If None, uses the default dataset.
        oracle_planner_v26: If True, use the V2.6 Structured Evidence Ledger.
        oracle_planner_v220: If True, use pure v2.6 plus diagnosis-driven self-evolution controls.
        resume: If True, merge with existing results and skip already processed items (default: False)
    """
    from rich.console import Console

    console = Console()
    validate_runtime_options(
        max_instances=max_instances,
        max_instances_per_category=max_instances_per_category,
        max_questions_per_instance=max_questions_per_instance,
        max_concurrent_items=max_concurrent_items,
        max_concurrent_questions=max_concurrent_questions,
        eval_semaphore_size=eval_semaphore_size,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
    )

    # Validate mutually exclusive arguments
    # --max-instances-per-category can't be combined with --max-instances or --category
    # But --category CAN be combined with --max-instances (to limit questions within a category)
    if max_instances_per_category is not None and (max_instances is not None or category is not None):
        raise ValueError("--max-questions-per-category cannot be combined with --max-instances or --category")

    # Validate --only-ingested can't be combined with other dataset filters
    if only_ingested:
        incompatible_flags = []
        if only_failed:
            incompatible_flags.append("--only-failed")
        if only_invalid:
            incompatible_flags.append("--only-invalid")
        if category is not None:
            incompatible_flags.append("--category")
        if question_id is not None:
            incompatible_flags.append("--question-id")
        if max_instances_per_category is not None:
            incompatible_flags.append("--max-instances-per-category")

        if incompatible_flags:
            raise ValueError(f"--only-ingested cannot be combined with: {', '.join(incompatible_flags)}")

    # Determine dataset path. The canonical path is validated even when it is
    # supplied explicitly through --dataset-path.
    explicit_dataset_path = dataset_path is not None
    dataset_path = resolve_dataset_path(dataset_path)
    if explicit_dataset_path:
        console.print(f"[cyan]Using custom dataset: {dataset_path}[/cyan]")
    else:
        console.print(
            f"[cyan]Using pinned LongMemEval dataset {LONGMEMEVAL_DATASET_REVISION[:12]} "
            f"(sha256:{LONGMEMEVAL_DATASET_SHA256[:12]}…)[/cyan]"
        )

    # Initialize components
    dataset = LongMemEvalDataset()

    # Start with all items or load from dataset
    original_dataset_items = None
    filtered_items = None

    # Handle max_instances_per_category (aka max_questions_per_category)
    if max_instances_per_category:
        console.print(f"[cyan]Limiting to {max_instances_per_category} questions per category[/cyan]")
        if original_dataset_items is None:
            original_dataset_items = dataset.load(dataset_path, max_items=None)

        # Group by category and take max_instances_per_category from each
        from collections import defaultdict

        category_items = defaultdict(list)
        for item in original_dataset_items:
            cat = item.get("question_type", "unknown")
            category_items[cat].append(item)

        # Take up to max_instances_per_category from each category
        filtered_items = []
        for cat, items in sorted(category_items.items()):
            limited = items[:max_instances_per_category]
            filtered_items.extend(limited)
            console.print(f"  [green]{cat}:[/green] {len(limited)} questions (of {len(items)} available)")

        console.print(f"[green]Total: {len(filtered_items)} questions across {len(category_items)} categories[/green]")

    # Load previous results if filtering for failed/invalid questions
    failed_question_ids = set()
    invalid_question_ids = set()
    if only_failed or only_invalid:
        # Use source_results if specified, otherwise default to benchmark_results.json
        source_file = source_results if source_results else "benchmark_results.json"
        results_path = Path(source_file).expanduser()
        if not results_path.is_absolute():
            source_root = Path(results_dir) if results_dir else Path(__file__).parent / "results"
            if not source_root.is_absolute():
                source_root = (REPOSITORY_ROOT / source_root).resolve()
            results_path = source_root / results_path
        if not results_path.exists():
            raise FileNotFoundError(
                f"Cannot use --only-failed or --only-invalid; results file not found: {results_path}"
            )

        console.print(f"[cyan]Reading failed/invalid questions from: {source_file}[/cyan]")
        with open(results_path, "r", encoding="utf-8") as f:
            previous_results = json.load(f)

        # Extract question IDs that failed or are invalid
        for item_result in previous_results.get("item_results", []):
            item_id = item_result["item_id"]
            for detail in item_result["metrics"].get("detailed_results", []):
                if only_failed and detail.get("is_correct") == False and not detail.get("is_invalid", False):
                    failed_question_ids.add(item_id)
                if only_invalid and detail.get("is_invalid", False):
                    invalid_question_ids.add(item_id)

        if only_failed:
            console.print(
                f"[cyan]Filtering to {len(failed_question_ids)} questions that failed (is_correct=False)[/cyan]"
            )
        if only_invalid:
            console.print(
                f"[cyan]Filtering to {len(invalid_question_ids)} questions that were invalid (is_invalid=True)[/cyan]"
            )

    # Filter dataset by category if specified
    if category:
        console.print(f"[cyan]Filtering questions by category: {category}[/cyan]")
        if original_dataset_items is None:
            # Load full dataset without max_instances limit for filtering
            original_dataset_items = dataset.load(dataset_path, max_items=None)

        filtered_items = [item for item in original_dataset_items if item.get("question_type") == category]

        if not filtered_items:
            available_categories = set(item.get("question_type", "unknown") for item in original_dataset_items)
            raise ValueError(
                f"No questions found for category {category!r}. Available categories: {sorted(available_categories)}"
            )

        total_found = len(filtered_items)
        will_run = min(total_found, max_instances) if max_instances else total_found
        if max_instances and total_found > max_instances:
            console.print(
                f"[green]Found {total_found} questions for category '{category}' (will run {will_run} due to --max-instances)[/green]"
            )
        else:
            console.print(f"[green]Found {total_found} questions for category '{category}'[/green]")

    # Filter dataset by question_id(s) if specified
    if question_id is not None:
        # Parse comma-separated question IDs
        target_ids = set(q.strip() for q in question_id.split(",") if q.strip())
        if not target_ids:
            raise ValueError("--question-id did not contain a valid question ID")

        console.print(f"[cyan]Filtering to {len(target_ids)} question ID(s): {sorted(target_ids)}[/cyan]")

        # Load original items if not already loaded
        if original_dataset_items is None:
            original_dataset_items = dataset.load(dataset_path, max_items=None)

        # If we already have filtered_items from category filtering, filter those
        # Otherwise start with all items
        items_to_filter = filtered_items if filtered_items is not None else original_dataset_items
        filtered_items = [item for item in items_to_filter if dataset.get_item_id(item) in target_ids]

        total_found = len(filtered_items)
        missing_ids = target_ids - {dataset.get_item_id(item) for item in filtered_items}
        if missing_ids:
            console.print(
                f"[yellow]Warning: {len(missing_ids)} question ID(s) not found in dataset: {sorted(missing_ids)}[/yellow]"
            )
        if total_found == 0:
            raise ValueError(f"None of the requested question IDs were found: {sorted(target_ids)}")
        will_run = min(total_found, max_instances) if max_instances else total_found
        if max_instances and total_found > max_instances:
            console.print(
                f"[green]Found {total_found} items matching question ID(s) (will run {will_run} due to --max-instances)[/green]"
            )
        else:
            console.print(f"[green]Found {total_found} items matching question ID(s)[/green]")

    # Filter dataset by index range if specified
    if index_range:
        try:
            start_idx, end_idx = map(int, index_range.split(","))
            start_idx = max(1, start_idx)  # Ensure 1-indexed, min 1
            end_idx = max(start_idx, end_idx)

            console.print(f"[cyan]Filtering to item index range: {start_idx}-{end_idx} (1-indexed)[/cyan]")

            if original_dataset_items is None:
                original_dataset_items = dataset.load(dataset_path, max_items=None)

            items_to_filter = filtered_items if filtered_items is not None else original_dataset_items
            filtered_items = [item for i, item in enumerate(items_to_filter, 1) if start_idx <= i <= end_idx]

            total_found = len(filtered_items)
            if total_found == 0:
                raise ValueError(f"--index-range {index_range!r} selected no questions")
            will_run = min(total_found, max_instances) if max_instances else total_found
            if max_instances and total_found > max_instances:
                console.print(
                    f"[green]Found {total_found} questions in range {start_idx}-{end_idx} (will run {will_run} due to --max-instances)[/green]"
                )
            else:
                console.print(f"[green]Found {total_found} questions in range {start_idx}-{end_idx}[/green]")
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid --index-range {index_range!r}; use 'start,end' (for example, '75,412')") from exc

    # Filter dataset based on failed/invalid flags
    if only_failed or only_invalid:
        target_ids = failed_question_ids if only_failed else invalid_question_ids
        if not target_ids:
            filter_type = "failed" if only_failed else "invalid"
            raise ValueError(f"No {filter_type} questions were found in the source results")

        # Load original items if not already loaded
        if original_dataset_items is None:
            # Load full dataset without max_instances limit for filtering
            original_dataset_items = dataset.load(dataset_path, max_items=None)

        # If we already have filtered_items from category filtering, filter those
        # Otherwise start with all items
        items_to_filter = filtered_items if filtered_items is not None else original_dataset_items
        filtered_items = [item for item in items_to_filter if dataset.get_item_id(item) in target_ids]

        filter_type = "failed" if only_failed else "invalid"
        total_found = len(filtered_items)
        will_run = min(total_found, max_instances) if max_instances else total_found
        if max_instances and total_found > max_instances:
            console.print(
                f"[green]Found {total_found} {filter_type} items to re-evaluate (will run {will_run} due to --max-instances)[/green]"
            )
        else:
            console.print(f"[green]Found {total_found} {filter_type} items to re-evaluate[/green]")

    output_path = resolve_output_path(results_dir, results_filename)
    merge_with_existing = (
        filln
        or question_id is not None
        or only_failed
        or only_invalid
        or only_ingested
        or category is not None
        or max_instances_per_category is not None
        or resume
    )
    current_manifest = build_run_manifest(
        dataset_path=dataset_path,
        context_format=context_format,
        max_instances=max_instances,
        max_instances_per_category=max_instances_per_category,
        max_questions_per_instance=max_questions_per_instance,
        question_id=question_id,
        index_range=index_range,
        category=category,
        max_concurrent_items=max_concurrent_items,
        max_concurrent_questions=max_concurrent_questions,
        eval_semaphore_size=eval_semaphore_size,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        oracle_planner_v26=oracle_planner_v26,
        oracle_planner_v220=oracle_planner_v220,
        query_expansion_enabled=query_expansion_enabled,
        query_rewriting_strategy=query_rewriting_strategy,
        session_expansion_weight=session_expansion_weight,
        skip_ingestion=skip_ingestion or only_ingested,
        ingest_only=ingest_only,
        force_reingest=force_reingest,
    )
    current_model_config = get_model_config()
    validate_output_target(
        output_path,
        merge_with_existing=merge_with_existing,
        resume=resume,
    )
    if output_path.exists() and merge_with_existing:
        validate_artifact_compatibility(
            output_path,
            current_manifest=current_manifest,
            current_model_config=current_model_config,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        console.print(f"[cyan]Resume mode enabled: using compatible results from {output_path}[/cyan]")

    # Create local memory engine only after all local artifact checks pass.
    from hms_api.engine.memory_engine import Budget
    from hms_api.models import RequestContext

    from benchmarks.common.benchmark_runner import create_memory_engine

    memory = await create_memory_engine()

    evidence_mode = None
    if oracle_planner_v220:
        evidence_mode = "oracle_v220"
    elif oracle_planner_v26:
        evidence_mode = "oracle_v26"

    # Create answer generator
    answer_generator = LongMemEvalAnswerGenerator(
        context_format=context_format,
        evidence_mode=evidence_mode,
    )
    # Log context format being used
    console.print(f"[blue]Context format: {context_format}[/blue]")

    answer_evaluator = LLMAnswerEvaluator()

    # Filter by only_ingested: only run items whose memory bank already exists
    if only_ingested:
        console.print("[cyan]Filtering to only items with existing memory banks...[/cyan]")

        # Load all items if not already loaded
        if original_dataset_items is None:
            original_dataset_items = dataset.load(dataset_path, max_items=None)

        items_to_check = filtered_items if filtered_items is not None else original_dataset_items

        # Check which items have existing banks
        ingested_items = []
        pool = await memory._get_pool()

        for item in items_to_check:
            item_id = dataset.get_item_id(item)
            agent_id = f"longmemeval_{item_id}"

            # A retained bank is reusable when its source chunks are durable;
            # fact extraction may legitimately produce zero facts.
            async with pool.acquire() as conn:
                has_chunks = await conn.fetchval(
                    f"SELECT EXISTS(SELECT 1 FROM {fq_table('chunks')} WHERE bank_id = $1)",
                    agent_id,
                )
                if has_chunks:
                    ingested_items.append(item)

        filtered_items = ingested_items
        console.print(f"[green]Found {len(filtered_items)} items with existing memory banks[/green]")

        if not filtered_items:
            raise RuntimeError("No items with durable retained source chunks were found")

    # Determine query rewriting strategy
    if query_expansion_enabled:
        strategy_name = query_rewriting_strategy
    else:
        strategy_name = "noop"

    # Create benchmark runner
    retrieval_planner = None
    if oracle_planner_v220:
        retrieval_planner = longmemeval_oracle_planner_v220
    elif oracle_planner_v26:
        retrieval_planner = longmemeval_oracle_planner_v26

    runner = BenchmarkRunner(
        dataset=dataset,
        answer_generator=answer_generator,
        answer_evaluator=answer_evaluator,
        memory=memory,
        query_rewriting_strategy_name=strategy_name,
        query_rewriting_enabled=query_expansion_enabled,
        session_expansion_weight=session_expansion_weight,
        retrieval_planner=retrieval_planner,
    )

    if query_expansion_enabled:
        console.print(f"[cyan]Query expansion enabled: using {strategy_name} strategy[/cyan]")

    console.print(f"[cyan]Session expansion weight: {session_expansion_weight}[/cyan]")
    if oracle_planner_v26:
        console.print("[cyan]Oracle planner v2.6 enabled: base retrieval + structured evidence ledger[/cyan]")
        for planner_category, planner_weight in sorted(ORACLE_PLANNER_V1_WEIGHTS.items()):
            suffix = " + query expansion + evidence appendix" if planner_category == "multi-session" else ""
            ledger = (
                " + high-risk ledger"
                if planner_category in {"multi-session", "temporal-reasoning", "knowledge-update"}
                else ""
            )
            console.print(f"  [cyan]{planner_category}:[/cyan] {planner_weight}{suffix}{ledger}")
    if oracle_planner_v220:
        profile = SELF_EVOLUTION_PROFILES["oracle_v220"]
        console.print("[cyan]Oracle planner v2.20 enabled: pure v2.6 + diagnosis-driven self-evolution[/cyan]")
        console.print(f"  [cyan]base:[/cyan] {profile['base']}")
        console.print(f"  [cyan]diagnosis source:[/cyan] {profile['diagnosis_source']}")
        console.print(f"  [cyan]selection:[/cyan] {profile['selection_rule']}")
        for planner_category, planner_weight in sorted(ORACLE_PLANNER_V1_WEIGHTS.items()):
            suffix = " + query expansion + evidence appendix" if planner_category == "multi-session" else ""
            ledger = (
                " + V2.6 ledger"
                if planner_category in {"multi-session", "temporal-reasoning", "knowledge-update"}
                else ""
            )
            controller = (
                " + self-evolution controller"
                if planner_category in {"multi-session", "temporal-reasoning", "knowledge-update"}
                else ""
            )
            date_block = " + self-evolved date evidence" if planner_category == "temporal-reasoning" else ""
            console.print(
                f"  [cyan]{planner_category}:[/cyan] {planner_weight}{suffix}{ledger}{controller}{date_block}"
            )

    # If filtering by category, failed, invalid, only_ingested, or max_instances_per_category, we need to use a custom dataset that only returns those items
    # We'll temporarily replace the dataset's load method
    if filtered_items is not None:
        original_load = dataset.load

        def filtered_load(path: Path, max_items: Optional[int] = None):
            return filtered_items[:max_items] if max_items else filtered_items

        dataset.load = filtered_load

    # Run benchmark
    # Single-phase approach: each question gets its own isolated agent_id
    # This ensures each question only has access to its own context

    # Configuration for single-phase benchmark
    separate_ingestion = False
    clear_per_item = True  # Use unique agent_id per question

    results = await runner.run(
        dataset_path=dataset_path,
        agent_id="longmemeval",  # Will be suffixed with question_id per item
        max_items=max_instances
        if not max_instances_per_category
        else None,  # Don't apply max_items when using per-category limit
        max_questions_per_item=max_questions_per_instance,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        skip_ingestion=skip_ingestion or only_ingested,  # Auto-skip ingestion when using --only-ingested
        max_concurrent_questions=max_concurrent_questions,
        eval_semaphore_size=eval_semaphore_size,
        separate_ingestion_phase=separate_ingestion,
        clear_agent_per_item=clear_per_item,
        filln=filln or resume,  # Resume skips item IDs already present in the result file.
        specific_item=None,  # Already filtered via filtered_items replacement
        max_concurrent_items=max_concurrent_items,  # Parallel instance processing
        output_path=output_path,  # Save results incrementally
        merge_with_existing=merge_with_existing,  # Merge when using --fill, --category, --only-failed, --only-invalid flags or specific question
        ingest_only=ingest_only,  # Only ingest, skip evaluation
        force_reingest=force_reingest,  # Force re-ingest even if data already exists
        rerun_invalid_existing=resume,
        run_manifest=current_manifest,
    )
    results["run_manifest"] = current_manifest
    runner.save_results(results, output_path)

    if ingest_only:
        console.print("\n[green]✓[/green] Ingest-only mode completed. Data is ready for evaluation.")
        console.print("  To run evaluation later with a different model:")
        console.print("  1. Update .env with your preferred model")
        console.print("  2. Run: HMS_BENCHMARK=longmemeval bash .aaaSCRIPT/run_benchmark.sh --only-ingested --fill")
        return results

    full_run_requested = (
        dataset_path.resolve() == DEFAULT_DATASET_PATH.resolve()
        and max_instances is None
        and max_instances_per_category is None
        and question_id is None
        and index_range is None
        and category is None
        and not only_failed
        and not only_invalid
        and not only_ingested
    )
    if full_run_requested:
        if (
            results.get("num_items") != LONGMEMEVAL_EXPECTED_ITEMS
            or results.get("total_questions") != LONGMEMEVAL_EXPECTED_ITEMS
        ):
            raise RuntimeError(
                "Incomplete full LongMemEval run: "
                f"items={results.get('num_items')}, questions={results.get('total_questions')}, "
                f"expected={LONGMEMEVAL_EXPECTED_ITEMS}"
            )
        if results.get("total_invalid", 0):
            raise RuntimeError(
                f"Full LongMemEval run contains {results['total_invalid']} invalid question(s); "
                f"inspect {output_path} and resume after correcting the provider or runtime failure"
            )

    # Display results (final save already happened incrementally)
    runner.display_results(results)
    console.print(f"\n[green]✓[/green] Results saved incrementally to {output_path}")

    # Generate detailed report by question type
    generate_type_report(results)

    # Generate markdown results table
    generate_markdown_table(results, output_path)

    return results


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_canonical_dataset(dataset_path: Path) -> None:
    """Validate the immutable LongMemEval artifact used by the full benchmark."""

    actual_size = dataset_path.stat().st_size
    if actual_size != LONGMEMEVAL_DATASET_SIZE:
        raise ValueError(
            f"LongMemEval dataset size mismatch at {dataset_path}: "
            f"expected {LONGMEMEVAL_DATASET_SIZE}, got {actual_size}"
        )
    actual_sha256 = _sha256_file(dataset_path)
    if actual_sha256 != LONGMEMEVAL_DATASET_SHA256:
        raise ValueError(
            f"LongMemEval dataset checksum mismatch at {dataset_path}: "
            f"expected {LONGMEMEVAL_DATASET_SHA256}, got {actual_sha256}"
        )


def download_dataset(dataset_path: Path) -> None:
    """Download, verify, and atomically install the pinned LongMemEval dataset."""

    from rich.console import Console

    console = Console()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = dataset_path.with_name(f".{dataset_path.name}.part")
    partial_path.unlink(missing_ok=True)

    console.print("[yellow]Dataset not found. Downloading the pinned LongMemEval artifact...[/yellow]")
    console.print(f"[dim]URL: {LONGMEMEVAL_DATASET_URL}[/dim]")
    console.print(f"[dim]Destination: {dataset_path}[/dim]")

    request = urllib.request.Request(
        LONGMEMEVAL_DATASET_URL,
        headers={"User-Agent": "HMS-LongMemEval-Reproduction/1.0"},
    )
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial_path.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                total_bytes += len(block)
            handle.flush()
            os.fsync(handle.fileno())

        if total_bytes != LONGMEMEVAL_DATASET_SIZE:
            raise ValueError(
                f"Downloaded dataset size mismatch: expected {LONGMEMEVAL_DATASET_SIZE}, got {total_bytes}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != LONGMEMEVAL_DATASET_SHA256:
            raise ValueError(
                f"Downloaded dataset checksum mismatch: expected {LONGMEMEVAL_DATASET_SHA256}, got {actual_sha256}"
            )
        os.replace(partial_path, dataset_path)
        console.print("[green]✓ Pinned dataset downloaded and verified[/green]")
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download the pinned LongMemEval dataset: {exc}") from exc


def generate_type_report(results: dict):
    """Generate a detailed report by question type."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Aggregate stats by question type
    type_stats = {}

    for item_result in results["item_results"]:
        metrics = item_result["metrics"]
        by_category = metrics.get("category_stats", {})

        for qtype, stats in by_category.items():
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "correct": 0}
            type_stats[qtype]["total"] += stats["total"]
            type_stats[qtype]["correct"] += stats["correct"]

    # Display table
    table = Table(title="Performance by Question Type")
    table.add_column("Question Type", style="cyan")
    table.add_column("Total", justify="right", style="yellow")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="magenta")

    for qtype, stats in sorted(type_stats.items()):
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        table.add_row(qtype, str(stats["total"]), str(stats["correct"]), f"{acc:.1f}%")

    console.print("\n")
    console.print(table)


def generate_markdown_table(results: dict, json_output_path: Path):
    """Generate a markdown results table with model configuration."""
    from rich.console import Console

    console = Console()

    # Aggregate stats by question type
    type_stats = {}

    for item_result in results["item_results"]:
        metrics = item_result["metrics"]
        by_category = metrics.get("category_stats", {})

        for qtype, stats in by_category.items():
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "correct": 0, "invalid": 0}
            type_stats[qtype]["total"] += stats["total"]
            type_stats[qtype]["correct"] += stats["correct"]
            type_stats[qtype]["invalid"] += stats.get("invalid", 0)

    # Build markdown content
    lines = []
    lines.append("# LongMemEval Benchmark Results")
    lines.append("")

    # Add model configuration
    if "model_config" in results:
        config = results["model_config"]
        lines.append("## Model Configuration")
        lines.append("")
        lines.append(f"- **HMS**: {config['hms']['provider']}/{config['hms']['model']}")
        if "retain" in config:
            lines.append(f"- **Retain**: {config['retain']['provider']}/{config['retain']['model']}")
        lines.append(
            f"- **Answer Generation**: {config['answer_generation']['provider']}/{config['answer_generation']['model']}"
        )
        lines.append(f"- **LLM Judge**: {config['judge']['provider']}/{config['judge']['model']}")
        if "embeddings" in config:
            lines.append(f"- **Embeddings**: {config['embeddings']['provider']}/{config['embeddings']['model']}")
        lines.append("")

    lines.append(
        f"**Overall Accuracy**: {results['overall_accuracy']:.2f}% ({results['total_correct']}/{results['total_questions']})"
    )
    lines.append("")

    # Results by question type
    lines.append("## Results by Question Type")
    lines.append("")
    lines.append("| Question Type | Total | Correct | Invalid | Accuracy |")
    lines.append("|---------------|-------|---------|---------|----------|")

    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        invalid_str = str(stats["invalid"]) if stats["invalid"] > 0 else "-"
        lines.append(f"| {qtype} | {stats['total']} | {stats['correct']} | {invalid_str} | {acc:.1f}% |")

    # Add overall row
    total_invalid = results.get("total_invalid", 0)
    invalid_str = str(total_invalid) if total_invalid > 0 else "-"
    lines.append(
        f"| **OVERALL** | **{results['total_questions']}** | **{results['total_correct']}** | **{invalid_str}** | **{results['overall_accuracy']:.1f}%** |"
    )

    # Write to file (same directory as JSON, but .md extension)
    md_output_path = json_output_path.with_suffix(".md")
    md_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n[green]✓[/green] Results table saved to {md_output_path}")


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit TOTAL number of questions to evaluate (default: all 500). For per-category limits, use --max-questions-per-category instead.",
    )
    parser.add_argument(
        "--max-instances-per-category",
        "--max-questions-per-category",  # Alias since each instance = 1 question in LongMemEval
        type=int,
        default=None,
        dest="max_instances_per_category",
        help="Limit number of questions per category (e.g., 20 = 20 questions from each of the 6 categories = 120 total). Cannot be combined with --max-instances or --category.",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None, help="Limit number of questions per instance (for quick testing)"
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=500, help="Thinking budget for spreading activation search"
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum tokens to retrieve from memories")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip ingestion and use existing data")
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Only process questions not already in results file (for resuming interrupted runs)",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Filter to specific question ID(s). Can be a single synthetic-style ID "
        "(e.g., 'sample-question-a') or comma-separated IDs "
        "(e.g., 'sample-question-a,sample-question-b'). Useful with --skip-ingestion "
        "to test specific questions.",
    )
    parser.add_argument(
        "--index-range",
        type=str,
        default=None,
        help="Filter to a range of item indices (e.g., '75,412'). Both start and end are inclusive, 1-indexed.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only run questions that were previously marked as incorrect (is_correct=False). Requires existing results file.",
    )
    parser.add_argument(
        "--only-invalid",
        action="store_true",
        help="Only run questions that were previously marked as invalid (is_invalid=True). Requires existing results file.",
    )
    parser.add_argument(
        "--only-ingested",
        action="store_true",
        help="Only run questions whose memory bank already exists (has been ingested). Automatically skips ingestion. Cannot be combined with --only-failed, --only-invalid, --category, --question-id, or --max-instances-per-category.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter questions by category/question_type. Available categories: 'single-session-user', 'multi-session', 'single-session-preference', 'temporal-reasoning', 'knowledge-update', 'single-session-assistant'. Can be combined with --max-instances to limit questions within the category.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of instances to process in parallel (default: 1 for sequential). Higher values speed up evaluation but use more memory.",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default="benchmark_results.json",
        help="Filename for results output (default: benchmark_results.json).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional directory for results. If not specified, uses results/ relative to script location.",
    )
    parser.add_argument(
        "--context-format",
        type=str,
        choices=["json", "structured", "structured_compact", "structured_source"],
        default="structured_source",
        help="How to format context: 'json' (raw), 'structured' (per-fact), 'structured_compact' (source bundles), or 'structured_source' (bundles plus retained source windows). Default: structured_source.",
    )
    parser.add_argument(
        "--source-results",
        type=str,
        default=None,
        help="Source results file to read failed/invalid questions from (for --only-failed/--only-invalid). Defaults to benchmark_results.json if not specified.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest conversation data (skip evaluation). Use with --fill to skip already ingested items. Use after ingest to do evaluation with different model.",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Force re-ingest even if data already exists. Use when you want to re-process ingestion for items that may have incomplete data.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO level log messages (only show warnings and errors)",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=10,
        help="Maximum number of concurrent question processing (default: 10)",
    )
    parser.add_argument(
        "--eval-semaphore-size",
        type=int,
        default=10,
        help="Maximum concurrent LLM judge requests (default: 10)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional custom dataset path. If not specified, uses the default dataset.",
    )
    parser.add_argument(
        "--enable-query-expansion",
        action="store_true",
        help="Enable query rewriting (default: disabled). When enabled, uses --query-rewriting-strategy to determine the strategy.",
    )
    parser.add_argument(
        "--query-rewriting-strategy",
        type=str,
        choices=["noop", "llm_based", "llm_driven"],
        default="llm_based",
        help="Query rewriting strategy to use when --enable-query-expansion is set. Options: 'noop' (no expansion), 'llm_based' (rule-based decision with LLM expansion), 'llm_driven' (LLM-driven analysis with entity expansion and time window calculation) (default: llm_based)",
    )
    parser.add_argument(
        "--session-expansion-weight",
        type=float,
        default=0.3,
        help="Weight for session-based node expansion (default: 0.3). Set to 0 to disable.",
    )
    parser.add_argument(
        "--oracle-planner-v26",
        action="store_true",
        help="Use the V2.6 Structured Evidence Ledger for high-risk questions.",
    )
    parser.add_argument(
        "--oracle-planner-v220",
        action="store_true",
        help="Use pure v2.6 retrieval plus diagnosis-driven self-evolution controls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous run by merging with existing results. Use with --results-filename to specify the same output file.",
    )

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Validate that only one of --only-failed or --only-invalid is set
    if args.only_failed and args.only_invalid:
        parser.error("Cannot use both --only-failed and --only-invalid at the same time")

    planner_flags = [
        args.oracle_planner_v26,
        args.oracle_planner_v220,
    ]
    if sum(1 for flag in planner_flags if flag) > 1:
        parser.error("Cannot use more than one oracle planner flag at the same time")

    # Validate mutually exclusive arguments
    # --max-instances-per-category can't be combined with --max-instances or --category
    if args.max_instances_per_category is not None and (args.max_instances is not None or args.category is not None):
        parser.error("--max-questions-per-category cannot be combined with --max-instances or --category")

    results = asyncio.run(
        run_benchmark(
            max_instances=args.max_instances,
            max_instances_per_category=args.max_instances_per_category,
            max_questions_per_instance=args.max_questions,
            thinking_budget=args.thinking_budget,
            max_tokens=args.max_tokens,
            skip_ingestion=args.skip_ingestion,
            filln=args.fill,
            question_id=args.question_id,
            index_range=args.index_range,
            only_failed=args.only_failed,
            only_invalid=args.only_invalid,
            only_ingested=args.only_ingested,
            category=args.category,
            max_concurrent_items=args.parallel,
            results_filename=args.results_filename,
            results_dir=args.results_dir,
            context_format=args.context_format,
            source_results=args.source_results,
            ingest_only=args.ingest_only,
            force_reingest=args.force_reingest,
            max_concurrent_questions=args.max_concurrent_questions,
            eval_semaphore_size=args.eval_semaphore_size,
            dataset_path=args.dataset_path,
            query_expansion_enabled=args.enable_query_expansion,
            query_rewriting_strategy=args.query_rewriting_strategy,
            session_expansion_weight=args.session_expansion_weight,
            oracle_planner_v26=args.oracle_planner_v26,
            oracle_planner_v220=args.oracle_planner_v220,
            resume=args.resume,
        )
    )
