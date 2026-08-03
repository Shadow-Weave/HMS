"""
Common benchmark runner framework.

This module provides a unified interface for running memory benchmarks with:
- Batch ingestion for speed
- Parallel question processing with semaphores
- Parallel LLM judging with rate limiting
- Progress tracking with Rich
- Comprehensive metrics collection
- Support for both traditional (search + LLM) and integrated (think API) approaches

The framework supports two answer generation patterns:
1. Traditional: Benchmark runner performs search, then passes results to answer generator
2. Integrated: Answer generator performs its own retrieval (e.g., think API)
   - Indicated by needs_external_search() returning False
   - Skips the search step for efficiency
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pydantic
from hms_api import MemoryEngine
from hms_api.config import DEFAULT_LLM_MODEL, PROVIDER_DEFAULT_MODELS, get_config

# Configure logging from environment variable
get_config().configure_logging()
from hms_api.engine.ingestion.adapters.storage_records import compute_document_hash, retain_document_metadata
from hms_api.engine.ingestion.normalization import normalize_content_item
from hms_api.engine.memory_engine import Budget
from hms_api.engine.schema import fq_table
from hms_api.models import RequestContext
from openai import AsyncOpenAI
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def _endpoint_fingerprint(value: Optional[str]) -> str:
    """Return a non-secret identity for an endpoint used in result compatibility checks."""

    normalized = (value or "<provider-default>").strip().rstrip("/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _embedding_runtime_config() -> Dict[str, str]:
    """Describe the effective embedding backend without exposing credentials."""

    provider = os.getenv("HMS_API_EMBEDDINGS_PROVIDER", "local").lower()
    model_settings = {
        "local": ("HMS_API_EMBEDDINGS_LOCAL_MODEL", "BAAI/bge-small-en-v1.5"),
        "tei": (None, "<server-defined>"),
        "openai": ("HMS_API_EMBEDDINGS_OPENAI_MODEL", "text-embedding-3-small"),
        "openrouter": ("HMS_API_EMBEDDINGS_OPENROUTER_MODEL", "perplexity/pplx-embed-v1-0.6b"),
        "cohere": ("HMS_API_EMBEDDINGS_COHERE_MODEL", "embed-english-v3.0"),
        "litellm": ("HMS_API_EMBEDDINGS_LITELLM_MODEL", "text-embedding-3-small"),
        "litellm-sdk": ("HMS_API_EMBEDDINGS_LITELLM_SDK_MODEL", "cohere/embed-english-v3.0"),
        "google": ("HMS_API_EMBEDDINGS_GEMINI_MODEL", "gemini-embedding-001"),
    }
    endpoint_settings = {
        "tei": ("HMS_API_EMBEDDINGS_TEI_URL", None),
        "openai": ("HMS_API_EMBEDDINGS_OPENAI_BASE_URL", None),
        "openrouter": (None, "https://openrouter.ai/api/v1"),
        "cohere": ("HMS_API_EMBEDDINGS_COHERE_BASE_URL", None),
        "litellm": ("HMS_API_EMBEDDINGS_LITELLM_API_BASE", "http://localhost:4000"),
        "litellm-sdk": ("HMS_API_EMBEDDINGS_LITELLM_SDK_API_BASE", None),
    }

    model_env, model_default = model_settings.get(provider, (None, "<unknown>"))
    model = os.getenv(model_env, model_default) if model_env else model_default
    endpoint_env, endpoint_default = endpoint_settings.get(provider, (None, None))
    endpoint = os.getenv(endpoint_env, endpoint_default) if endpoint_env else endpoint_default
    if provider == "google":
        endpoint = "|".join(
            (
                "google",
                os.getenv("HMS_API_EMBEDDINGS_VERTEXAI_PROJECT_ID", "<developer-api>"),
                os.getenv("HMS_API_EMBEDDINGS_VERTEXAI_REGION", "us-central1"),
            )
        )

    return {
        "provider": provider,
        "model": model,
        "fingerprint_policy": os.getenv("HMS_API_EMBEDDING_FINGERPRINT_POLICY", "strict"),
        "endpoint_fingerprint": _endpoint_fingerprint(endpoint),
    }


def _reranker_runtime_config() -> Dict[str, str]:
    """Describe the effective reranker backend without exposing credentials."""

    provider = os.getenv("HMS_API_RERANKER_PROVIDER", "local").lower()
    model_settings = {
        "local": ("HMS_API_RERANKER_LOCAL_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "tei": (None, "<server-defined>"),
        "cohere": ("HMS_API_RERANKER_COHERE_MODEL", "rerank-english-v3.0"),
        "openrouter": ("HMS_API_RERANKER_OPENROUTER_MODEL", "cohere/rerank-v3.5"),
        "flashrank": ("HMS_API_RERANKER_FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2"),
        "litellm": ("HMS_API_RERANKER_LITELLM_MODEL", "cohere/rerank-english-v3.0"),
        "litellm-sdk": ("HMS_API_RERANKER_LITELLM_SDK_MODEL", "cohere/rerank-english-v3.0"),
        "zeroentropy": ("HMS_API_RERANKER_ZEROENTROPY_MODEL", "zerank-2"),
        "siliconflow": ("HMS_API_RERANKER_SILICONFLOW_MODEL", "BAAI/bge-reranker-v2-m3"),
        "google": ("HMS_API_RERANKER_GOOGLE_MODEL", "semantic-ranker-default-004"),
        "qwen3-reranker": ("HMS_API_RERANKER_QWEN3_MODEL_PATH", "<provider-default>"),
        "rrf": (None, "<none>"),
        "jina-mlx": (None, "<built-in>"),
    }
    endpoint_settings = {
        "tei": ("HMS_API_RERANKER_TEI_URL", None),
        "cohere": ("HMS_API_RERANKER_COHERE_BASE_URL", None),
        "openrouter": (None, "https://openrouter.ai/api/v1/rerank"),
        "litellm": ("HMS_API_RERANKER_LITELLM_API_BASE", "http://localhost:4000"),
        "litellm-sdk": ("HMS_API_RERANKER_LITELLM_SDK_API_BASE", None),
        "siliconflow": ("HMS_API_RERANKER_SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    }

    model_env, model_default = model_settings.get(provider, (None, "<unknown>"))
    model = os.getenv(model_env, model_default) if model_env else model_default
    endpoint_env, endpoint_default = endpoint_settings.get(provider, (None, None))
    endpoint = os.getenv(endpoint_env, endpoint_default) if endpoint_env else endpoint_default
    if provider == "google":
        endpoint = f"google|{os.getenv('HMS_API_RERANKER_GOOGLE_PROJECT_ID', '<unknown-project>')}"

    return {
        "provider": provider,
        "model": model,
        "endpoint_fingerprint": _endpoint_fingerprint(endpoint),
    }


def _default_model_for_provider(provider: str) -> str:
    """Mirror the core configuration's provider-specific model defaults."""

    return PROVIDER_DEFAULT_MODELS.get(provider.lower(), DEFAULT_LLM_MODEL)


class IngestionIntegrityError(RuntimeError):
    """Raised when retained source documents are not durably queryable."""

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        item_id = report.get("item_id", "unknown")
        failure_fields = (
            "missing_documents",
            "documents_without_chunks",
            "unexpected_documents",
            "inflight_documents",
            "content_hash_mismatches",
            "context_mismatches",
            "event_date_mismatches",
            "unverifiable_documents",
            "invalid_retain_params",
            "failed_banks",
        )
        failures = {field: report.get(field) for field in failure_fields if report.get(field)}
        super().__init__(f"Durable ingestion audit failed for item {item_id!r}: {failures}")


def _write_json_atomic(payload: Dict[str, Any], output_path: Path) -> None:
    """Write a JSON result without exposing a partially written resume file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _result_is_resume_complete(result: Mapping[str, Any]) -> bool:
    """Return whether an existing item can be safely skipped by resume mode."""

    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    if int(metrics.get("total", 0) or 0) < 1 or int(metrics.get("invalid", 0) or 0) > 0:
        return False

    details = metrics.get("detailed_results", [])
    if not isinstance(details, list) or not details:
        return False
    for detail in details:
        if not isinstance(detail, Mapping):
            return False
        if detail.get("is_invalid") or detail.get("error"):
            return False
        predicted = str(detail.get("predicted_answer", ""))
        reasoning = str(detail.get("correctness_reasoning", ""))
        if predicted.startswith("Error generating answer:") or reasoning.startswith("Error:"):
            return False
    return True


def get_model_config() -> Dict[str, Dict[str, str]]:
    """
    Get the non-secret model configuration for the benchmark runtime.

    Reads directly from environment variables without instantiating LLM clients.

    Returns:
        Provider and model identifiers for each runtime role.
    """
    # Memory/HMS config (base config)
    memory_provider = os.getenv("HMS_API_LLM_PROVIDER", "groq")
    memory_model = os.getenv("HMS_API_LLM_MODEL", "openai/gpt-oss-120b")

    # Retain config (falls back to memory config).
    retain_provider = os.getenv("HMS_API_RETAIN_LLM_PROVIDER", memory_provider)
    retain_model = os.getenv("HMS_API_RETAIN_LLM_MODEL")
    if retain_model is None:
        retain_model = (
            _default_model_for_provider(retain_provider)
            if "HMS_API_RETAIN_LLM_PROVIDER" in os.environ
            else memory_model
        )
    memory_base_url = os.getenv("HMS_API_LLM_BASE_URL") or None
    retain_base_url = os.getenv("HMS_API_RETAIN_LLM_BASE_URL") or memory_base_url

    # Answer generation config (falls back to memory config)
    answer_provider = os.getenv("HMS_API_ANSWER_LLM_PROVIDER", memory_provider)
    answer_model = os.getenv("HMS_API_ANSWER_LLM_MODEL", memory_model)
    answer_base_url = os.getenv("HMS_API_ANSWER_LLM_BASE_URL") or memory_base_url

    # Judge config (falls back to memory config)
    judge_provider = os.getenv("HMS_API_JUDGE_LLM_PROVIDER", memory_provider)
    judge_model = os.getenv("HMS_API_JUDGE_LLM_MODEL", memory_model)
    judge_base_url = os.getenv("HMS_API_JUDGE_LLM_BASE_URL") or memory_base_url

    return {
        "hms": {
            "provider": memory_provider,
            "model": memory_model,
            "endpoint_fingerprint": _endpoint_fingerprint(memory_base_url),
        },
        "retain": {
            "provider": retain_provider,
            "model": retain_model,
            "endpoint_fingerprint": _endpoint_fingerprint(retain_base_url),
        },
        "answer_generation": {
            "provider": answer_provider,
            "model": answer_model,
            "endpoint_fingerprint": _endpoint_fingerprint(answer_base_url),
        },
        "judge": {
            "provider": judge_provider,
            "model": judge_model,
            "endpoint_fingerprint": _endpoint_fingerprint(judge_base_url),
        },
        "embeddings": _embedding_runtime_config(),
        "reranker": _reranker_runtime_config(),
    }


def get_artifact_model_config(
    *,
    retain_executed: bool = True,
    retain_execution: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Return model identities that truthfully describe stages executed in this run.

    A reused memory bank predates the current benchmark process. Its Retain
    model identity cannot be reconstructed from durable document rows, so the
    current Retain environment must not be presented as the bank creator.
    """

    execution = retain_execution or ("executed" if retain_executed else "not_executed")
    if execution not in {"executed", "not_executed", "partial_or_skipped"}:
        raise ValueError(f"Unsupported Retain execution mode: {execution}")

    config = get_model_config()
    if execution == "not_executed":
        config["retain"] = {
            "execution": "not_executed",
            "bank_creator_identity": "unverifiable",
        }
    elif execution == "partial_or_skipped":
        config["retain"] = {
            "execution": "partial_or_skipped",
            "bank_creator_identity": "mixed_or_unverifiable",
        }
    return config


def format_retain_model_config(config: Mapping[str, str]) -> str:
    """Render an executed, reused, or mixed Retain identity without overclaiming."""

    if "provider" in config and "model" in config:
        return f"{config['provider']}/{config['model']}"
    if config.get("execution") == "partial_or_skipped":
        return "partially executed or skipped; bank creator identity is mixed or unverifiable"
    return "not executed; reused-bank creator identity is unverifiable"


def print_model_config(config: Optional[Mapping[str, Mapping[str, str]]] = None):
    """Print the model configuration to console."""
    config = config or get_model_config()
    retain_config = config.get("retain", {})

    console.print("\n[bold cyan]Model Configuration:[/bold cyan]")
    console.print(f"  HMS:         {config['hms']['provider']}/{config['hms']['model']}")
    console.print(f"  Retain:      {format_retain_model_config(retain_config)}")
    console.print(
        f"  Answer Generation: {config['answer_generation']['provider']}/{config['answer_generation']['model']}"
    )
    console.print(f"  LLM Judge:         {config['judge']['provider']}/{config['judge']['model']}")
    console.print()


async def create_memory_engine() -> MemoryEngine:
    """
    Create and initialize a MemoryEngine instance from environment variables.

    Reads configuration from:
    - HMS_API_DATABASE_URL (default: "pg0")
    - HMS_API_LLM_PROVIDER (default: "groq")
    - HMS_API_LLM_API_KEY
    - HMS_API_LLM_MODEL (default: "openai/gpt-oss-120b")
    - HMS_API_LLM_BASE_URL (optional)

    Returns:
        Initialized MemoryEngine instance
    """
    memory = MemoryEngine(
        db_url=os.getenv("HMS_API_DATABASE_URL", "pg0"),
        memory_llm_provider=os.getenv("HMS_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HMS_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HMS_API_LLM_MODEL", "openai/gpt-oss-120b"),
        memory_llm_base_url=os.getenv("HMS_API_LLM_BASE_URL") or None,  # Use None to get provider defaults
    )
    await memory.initialize()
    return memory


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load dataset from file.

        Returns:
            List of dataset items
        """
        pass

    @abstractmethod
    def get_item_id(self, item: Dict) -> str:
        """Get unique identifier for an item."""
        pass

    @abstractmethod
    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare conversation sessions for batch ingestion.

        Returns:
            List of session dicts with keys: 'content', 'context', 'event_date'
        """
        pass

    @abstractmethod
    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from an item.

        Returns:
            List of QA dicts with keys: 'question', 'answer', 'category' (optional)
        """
        pass


class LLMAnswerGenerator(ABC):
    """Abstract base class for LLM-based answer generation."""

    def needs_external_search(self) -> bool:
        """
        Whether this generator needs external search to be performed.

        Returns:
            True if the benchmark runner should perform search before calling generate_answer.
            False if the generator does its own retrieval (e.g., integrated think API).
        """
        return True

    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        recall_result: Dict[str, Any],
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None,
        bank_id: Optional[str] = None,
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories.

        Args:
            question: The question text
            recall_result: Full RecallResult dict containing results, entities, chunks, and trace
            question_date: Optional date when the question was asked (for temporal context)
            question_type: Optional question category/type (e.g., 'multi-session', 'temporal-reasoning')
            bank_id: Optional bank ID for generators that need it (e.g., ReflectAnswerGenerator)

        Returns:
            Tuple of (answer, reasoning, retrieved_memories_override)
            - answer: The generated answer text
            - reasoning: Explanation of how the answer was derived
            - retrieved_memories_override: Optional list of memories to include in results
              - None: Use memories from recall_result (traditional mode)
              - List: Use these memories instead (integrated mode like think API)
        """
        pass


class JudgeResponse(pydantic.BaseModel):
    """Judge response format."""

    correct: bool
    reasoning: str


class LLMAnswerEvaluator:
    """LLM-based answer evaluator with configurable provider."""

    def __init__(self):
        """Initialize with LLM configuration for judge/evaluator."""
        import os

        from hms_api.engine.llm_wrapper import LLMConfig

        self.llm_config = LLMConfig(
            provider=os.getenv("HMS_API_JUDGE_LLM_PROVIDER", os.getenv("HMS_API_LLM_PROVIDER", "openai")),
            api_key=os.getenv("HMS_API_JUDGE_LLM_API_KEY", os.getenv("HMS_API_LLM_API_KEY", "")),
            base_url=os.getenv("HMS_API_JUDGE_LLM_BASE_URL", os.getenv("HMS_API_LLM_BASE_URL", "")),
            model=os.getenv("HMS_API_JUDGE_LLM_MODEL", os.getenv("HMS_API_LLM_MODEL", "gpt-4o-mini")),
            reasoning_effort="high",
        )
        self.client = self.llm_config._client
        self.model = self.llm_config.model

    async def judge_answer(
        self,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        semaphore: asyncio.Semaphore,
        category: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[bool, str, float]:
        """
        Evaluate predicted answer using LLM-as-judge with category-specific prompts.

        Args:
            question: The question
            correct_answer: Gold/correct answer
            predicted_answer: Predicted answer
            semaphore: Semaphore for rate limiting
            category: Question category for LongMemEval-specific evaluation
            max_retries: Maximum retry attempts for validation errors

        Returns:
            Tuple of (is_correct, reasoning, judge_time)
        """
        async with semaphore:
            import time

            judge_start_time = time.time()

            for attempt in range(max_retries):
                try:
                    # LongMemEval-specific evaluation prompts
                    if category in ["single-session-user", "single-session-assistant", "multi-session"]:
                        prompt_content = f"""Evaluate if the model response contains the correct answer to the question.

I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=no.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, set correct=false

Question: {question}

Correct Answer: {correct_answer}

Model Response: {predicted_answer}

Evaluation criteria:
- Set correct=true if the response contains the correct answer
- Set correct=true if the response is equivalent to the correct answer or contains intermediate steps
- Set correct=false if the response is incorrect or missing key information

Provide your evaluation as JSON with:
- reasoning: One sentence explanation
- correct: true or false"""

                    elif category == "temporal-reasoning":
                        prompt_content = """
I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true.
If the response only contains a subset of the information required by the answer, answer correct=false.
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.
"""

                    elif category == "knowledge-update":
                        prompt_content = """
I will give you a question, a correct answer, and a response from a model.
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false.
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.
"""

                    elif category == "single-session-preference":
                        prompt_content = """
I will give you a question, a answer for desired personalized response, and a response from a model.
Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false.
The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly.
"""

                    else:
                        # Default short-form answer evaluation.
                        prompt_content = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember which keepsake I chose at the fictional Harborlight fair?
    Gold answer: A copper compass pin
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.
"""

                    judgement = await self.llm_config.call(
                        messages=[
                            {
                                "role": "user",
                                "content": f"""{prompt_content}


Question: {question}
Gold answer: {correct_answer}
Generated answer: {predicted_answer}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.
""",
                            }
                        ],
                        response_format=JudgeResponse,
                        scope="judge",
                        temperature=0,
                        max_completion_tokens=4096,
                    )

                    judge_time = time.time() - judge_start_time
                    console.print(f"      [cyan]Answer judged in {judge_time:.1f}s[/cyan]")

                    return judgement.correct, judgement.reasoning, judge_time

                except Exception as e:
                    # Check if it's a validation error (LLM returned malformed JSON)
                    error_str = str(e)
                    is_validation_error = "ValidationError" in error_str or "Field required" in error_str

                    # Retry on validation errors, fail immediately on other errors
                    if is_validation_error and attempt < max_retries - 1:
                        print(f"Judge validation error on attempt {attempt + 1}/{max_retries}, retrying...")
                        await asyncio.sleep(0.5)  # Small delay before retry
                        continue

                    # Provider and parsing failures are not benchmark judgments.
                    # Propagate them so the caller records this question as
                    # invalid instead of silently scoring it as an ordinary
                    # incorrect answer.
                    raise RuntimeError(f"Judge failed after {attempt + 1} attempt(s): {e}") from e


@dataclass
class CoarseSearchCandidate:
    """Candidate document produced by the coarse retrieval stage."""

    rank: int
    document_id: str
    text: str
    context: Optional[str] = None
    occurred_start: Optional[str] = None
    fact_type: str = "unknown"
    rrf_score: float = 0.0
    proof_count: Optional[int] = None


@dataclass
class CoarseSearchResults:
    """Results produced by the coarse retrieval stage."""

    total_candidates: int
    candidates: list[CoarseSearchCandidate]


@dataclass
class RerankedCandidate:
    """Candidate document after reranking."""

    original_rank: int
    document_id: str
    text: str
    cross_encoder_score: float
    combined_score: float
    final_rank: int


@dataclass
class RerankedResults:
    """Results produced by reranking."""

    reranker_model: str
    reranker_provider: str
    reranked_candidates: list[RerankedCandidate]


@dataclass
class RecallPlan:
    """Per-question retrieval controls selected by a planner."""

    name: str = "default"
    session_expansion_weight: Optional[float] = None
    query_rewriting_enabled: Optional[bool] = None
    query_rewriting_strategy_name: Optional[str] = None
    max_tokens: Optional[int] = None
    include_chunks: Optional[bool] = None
    max_chunk_tokens: Optional[int] = None
    evidence_appendix_mode: Optional[str] = None


RetrievalPlanner = Callable[[str, Optional[str], Optional[datetime]], RecallPlan]


def _appendix_session_key(fact: Dict[str, Any]) -> str:
    document_id = fact.get("document_id")
    if document_id:
        return str(document_id)

    context = fact.get("context") or ""
    match = re.search(r"Session\s+([^\s]+)", context)
    if match:
        return match.group(1)

    return "__unknown_session__"


def _truncate_appendix_text(text: str, max_chars: int = 360) -> str:
    normalized = " ".join((text or "").replace("<|endoftext|>", " ").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def add_cross_session_evidence_appendix(
    recall_result: Dict[str, Any],
    *,
    max_sessions: int = 6,
    per_session_facts: int = 3,
    max_chars: int = 360,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach grouped cross-session evidence without changing recall ordering."""
    results = recall_result.get("results")
    if not isinstance(results, list) or not results:
        return recall_result

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    session_order: list[str] = []
    for fact in results:
        if not isinstance(fact, dict):
            continue
        session_key = _appendix_session_key(fact)
        if session_key not in grouped:
            grouped[session_key] = []
            session_order.append(session_key)
        if len(grouped[session_key]) < per_session_facts:
            grouped[session_key].append(fact)

    if len(session_order) <= 1:
        return recall_result

    sessions = []
    for session_key in session_order[:max_sessions]:
        evidence = []
        for fact in grouped.get(session_key, []):
            evidence.append(
                {
                    "id": fact.get("id"),
                    "text": _truncate_appendix_text(fact.get("text") or "", max_chars=max_chars),
                    "fact_type": fact.get("fact_type"),
                    "occurred_start": fact.get("occurred_start"),
                    "mentioned_at": fact.get("mentioned_at"),
                    "document_id": fact.get("document_id"),
                    "entities": (fact.get("entities") or [])[:8],
                }
            )
        if evidence:
            sessions.append({"session_id": session_key, "evidence": evidence})

    if not sessions:
        return recall_result

    updated = dict(recall_result)
    updated["cross_session_evidence"] = {
        "mode": "appendix",
        "instruction": instruction
        or (
            "Supplementary grouped evidence for cross-session comparison. "
            "Use this only to compare facts across sessions; preserve the main retrieved results as primary evidence."
        ),
        "max_sessions": max_sessions,
        "per_session_facts": per_session_facts,
        "sessions": sessions,
    }
    return updated


@dataclass
class RetrievalCache:
    """Diagnostic retrieval cache for an incorrectly answered question."""

    question_id: str
    question: str
    question_date: Optional[str]
    category: str
    correct_answer: str
    generated_answer: str
    is_correct: bool
    judge_reasoning: str
    retrieval_timestamp: str
    coarse_search_results: CoarseSearchResults
    reranked_results: RerankedResults


class BenchmarkRunner:
    """
    Common benchmark runner for retain, recall, answer, and judge evaluation.

    Optimizations:
    - Batch ingestion (put_batch_async)
    - Parallel question processing with rate limiting
    - Parallel LLM judging with rate limiting
    - Progress tracking
    """

    def __init__(
        self,
        dataset: BenchmarkDataset,
        answer_generator: LLMAnswerGenerator,
        answer_evaluator: LLMAnswerEvaluator,
        memory: Optional[MemoryEngine] = None,
        query_rewriting_strategy_name: str = "noop",
        query_rewriting_enabled: bool = False,
        session_expansion_weight: float = 0.3,
        retrieval_planner: Optional[RetrievalPlanner] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            dataset: Dataset implementation
            answer_generator: Answer generator implementation
            answer_evaluator: Answer evaluator implementation
            memory: Memory system instance (creates new if None)
            query_rewriting_strategy_name: Name of query rewriting strategy to use
            query_rewriting_enabled: Whether to enable query rewriting
            session_expansion_weight: Weight for session-based node expansion (default 0.3)
            retrieval_planner: Optional per-question planner for dynamic retrieval controls
        """
        import os

        self.dataset = dataset
        self.answer_generator = answer_generator
        self.answer_evaluator = answer_evaluator
        self.template_path: Optional[str] = None
        self.query_rewriting_strategy_name = query_rewriting_strategy_name
        self.query_rewriting_enabled = query_rewriting_enabled
        self.session_expansion_weight = session_expansion_weight
        self.retrieval_planner = retrieval_planner
        self._diagnostic_cache_dir: Optional[Path] = None
        self.memory = memory or MemoryEngine(
            db_url=os.getenv("HMS_API_DATABASE_URL", "pg0"),
            memory_llm_provider=os.getenv("HMS_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HMS_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HMS_API_LLM_MODEL", "openai/gpt-oss-20b"),
            memory_llm_base_url=os.getenv("HMS_API_LLM_BASE_URL") or None,
        )

    def _save_retrieval_cache(self, cache: RetrievalCache, cache_dir: Path) -> bool:
        """Persist diagnostic retrieval data without affecting judge validity."""
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"retrieval_cache_wrong_{cache.question_id}_{timestamp}.json"
            filepath = cache_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(cache), f, ensure_ascii=False, indent=2)
            return True
        except OSError as exc:
            logging.warning(
                "Could not write optional retrieval cache for %s to %s: %s",
                cache.question_id,
                cache_dir,
                exc,
            )
            return False

    def calculate_data_stats(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the data to be ingested.

        Returns:
            Dict with statistics: total_sessions, total_chars, avg_session_length, etc.
        """
        total_sessions = 0
        total_chars = 0
        session_lengths = []

        for item in items:
            batch_contents = self.dataset.prepare_sessions_for_ingestion(item)
            total_sessions += len(batch_contents)

            for session in batch_contents:
                content_len = len(session["content"])
                total_chars += content_len
                session_lengths.append(content_len)

        avg_length = total_chars / total_sessions if total_sessions > 0 else 0

        return {
            "total_sessions": total_sessions,
            "total_chars": total_chars,
            "total_items": len(items),
            "avg_session_length": avg_length,
            "min_session_length": min(session_lengths) if session_lengths else 0,
            "max_session_length": max(session_lengths) if session_lengths else 0,
        }

    async def apply_template(self, bank_id: str, manifest_path: str) -> None:
        """Apply a bank template manifest to a bank before ingestion.

        Reads the manifest JSON file and applies config overrides, creates
        mental models and directives — same logic as the /import API endpoint.
        """
        from hms_api.api.http import BankTemplateManifest
        from hms_api.models import RequestContext

        raw = json.loads(Path(manifest_path).read_text())
        manifest = BankTemplateManifest.model_validate(raw)

        request_context = RequestContext()
        await self.memory.get_bank_profile(bank_id, request_context=request_context)

        # Apply bank config overrides
        if manifest.bank:
            config_updates = manifest.bank.get_config_updates()
            if config_updates:
                await self.memory._config_resolver.update_bank_config(bank_id, config_updates, request_context)

        # Create directives
        for directive in manifest.directives or []:
            await self.memory.create_directive(
                bank_id=bank_id,
                name=directive.name,
                content=directive.content,
                priority=directive.priority,
                is_active=directive.is_active,
                tags=directive.tags if directive.tags else None,
                request_context=request_context,
            )

        # Create mental models (async content generation)
        for mm in manifest.mental_models or []:
            mental_model = await self.memory.create_mental_model(
                bank_id=bank_id,
                name=mm.name,
                source_query=mm.source_query,
                content="Generating content...",
                mental_model_id=mm.id,
                tags=mm.tags if mm.tags else None,
                max_tokens=mm.max_tokens,
                trigger=mm.trigger.model_dump() if mm.trigger else None,
                request_context=request_context,
            )
            await self.memory.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model["id"],
                request_context=request_context,
            )

    async def ingest_conversation(
        self, item: Dict[str, Any], agent_id: str, wait_for_consolidation: bool = False
    ) -> int:
        """
        Ingest conversation into memory using batch ingestion.

        Uses put_batch_async for maximum efficiency.

        Args:
            item: Dataset item to ingest
            agent_id: Agent/bank ID to ingest into
            wait_for_consolidation: If True, wait for consolidation to complete after ingestion

        Returns:
            Number of sessions ingested
        """
        batch_contents = self.dataset.prepare_sessions_for_ingestion(item)

        if batch_contents:
            max_retries = 3
            retry_delay = 2.0

            for attempt in range(max_retries):
                try:
                    await self.memory.retain_batch_async(
                        bank_id=agent_id,
                        contents=batch_contents,
                        request_context=RequestContext(),
                    )
                    break
                except ValueError as e:
                    error_msg = str(e)
                    if "Batch contains duplicate document_ids" in error_msg:
                        console.print(
                            f"      [yellow]⚠[/yellow] Duplicate document_id error on attempt {attempt + 1}, "
                            f"deduplicating batch contents..."
                        )
                        unique_contents = []
                        seen_doc_ids = set()
                        for content in batch_contents:
                            doc_id = content.get("document_id")
                            if doc_id and doc_id in seen_doc_ids:
                                continue
                            unique_contents.append(content)
                            if doc_id:
                                seen_doc_ids.add(doc_id)
                        batch_contents = unique_contents
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        raise
                except Exception as e:
                    console.print(f"      [yellow]⚠[/yellow] Ingestion error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5

        if wait_for_consolidation and batch_contents:
            await self._wait_for_consolidation(agent_id)

        return len(batch_contents)

    async def _get_pending_consolidation_count(self, bank_id: str) -> int:
        """
        Get the count of memories pending consolidation.

        Returns:
            Number of memories not yet processed by the consolidation job
        """
        pool = await self.memory._get_pool()
        from hms_api.engine.memory_engine import fq_table

        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as count
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND consolidated_at IS NULL AND fact_type IN ('experience', 'world')
                """,
                bank_id,
            )
            return result["count"] if result else 0

    async def _wait_for_consolidation(self, bank_id: str, poll_interval: float = 2.0, timeout: float = 3000.0) -> None:
        """
        Wait for consolidation to complete (pending_consolidation reaches 0).

        Args:
            bank_id: Bank ID to check
            poll_interval: Seconds between polls
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If consolidation doesn't complete within timeout
        """
        import time

        start_time = time.time()
        console.print("      [yellow]Waiting for consolidation to complete...[/yellow]")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Consolidation did not complete within {timeout}s")

            pending = await self._get_pending_consolidation_count(bank_id)
            if pending == 0:
                console.print("      [green]✓[/green] Consolidation complete")
                return

            # Still pending, wait and poll again
            await asyncio.sleep(poll_interval)

    async def answer_question(
        self,
        agent_id: str,
        question: str,
        thinking_budget: int = 500,
        max_tokens: int = 4096,
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None,
    ) -> Tuple[str, str, List[Dict], Dict[str, Dict], Optional[Dict[str, Any]]]:
        """
        Answer a question using memory retrieval.

        Args:
            agent_id: Agent ID
            question: Question text
            thinking_budget: Thinking budget for search
            max_tokens: Maximum tokens to retrieve
            question_date: Date when the question was asked (for temporal filtering)
            question_type: Question category/type (e.g., 'multi-session', 'temporal-reasoning')

        Returns:
            Tuple of (answer, reasoning, retrieved_memories, chunks, retrieval_details)
            - retrieval_details: Optional dict with coarse_search_results and reranked_results
        """
        retrieval_details = None

        # Check if generator needs external search
        if self.answer_generator.needs_external_search():
            # Traditional flow: search then generate
            # Use MemoryEngine directly
            # Map thinking_budget to budget level
            budget = Budget.LOW if thinking_budget <= 30 else Budget.MID if thinking_budget <= 70 else Budget.HIGH

            import time

            recall_start_time = time.time()
            plan = (
                self.retrieval_planner(question, question_type, question_date)
                if self.retrieval_planner is not None
                else RecallPlan()
            )
            recall_max_tokens = plan.max_tokens if plan.max_tokens is not None else max_tokens
            recall_include_chunks = plan.include_chunks if plan.include_chunks is not None else True
            recall_max_chunk_tokens = plan.max_chunk_tokens if plan.max_chunk_tokens is not None else 8192
            recall_query_rewriting_enabled = (
                plan.query_rewriting_enabled
                if plan.query_rewriting_enabled is not None
                else self.query_rewriting_enabled
            )
            recall_query_rewriting_strategy = (
                plan.query_rewriting_strategy_name
                if plan.query_rewriting_strategy_name is not None
                else self.query_rewriting_strategy_name
            )
            recall_session_expansion_weight = (
                plan.session_expansion_weight
                if plan.session_expansion_weight is not None
                else self.session_expansion_weight
            )

            # Use default fact types (no filtering)
            search_result = await self.memory.recall_async(
                bank_id=agent_id,
                query=question,
                budget=budget,
                max_tokens=recall_max_tokens,
                question_date=question_date,
                include_entities=True,
                max_entity_tokens=2048,
                include_chunks=recall_include_chunks,
                max_chunk_tokens=recall_max_chunk_tokens,
                request_context=RequestContext(),
                query_rewriting_strategy_name=recall_query_rewriting_strategy,
                query_rewriting_enabled=recall_query_rewriting_enabled,
                session_expansion_weight=recall_session_expansion_weight,
                enable_trace=True,
            )
            recall_time = time.time() - recall_start_time

            # Log recall stats
            num_results = len(search_result.results) if search_result.results else 0
            num_chunks = len(search_result.chunks) if search_result.chunks else 0
            num_entities = len(search_result.entities) if search_result.entities else 0

            # Keep the detailed trace local to benchmark diagnostics. It can be
            # large and contains retrieval internals that must not influence the
            # answer generator.
            recall_result_dict = search_result.model_dump(exclude={"trace"})
            if plan.evidence_appendix_mode == "cross_session":
                recall_result_dict = add_cross_session_evidence_appendix(recall_result_dict)
            elif plan.evidence_appendix_mode == "cross_session_compact":
                recall_result_dict = add_cross_session_evidence_appendix(
                    recall_result_dict,
                    max_sessions=4,
                    per_session_facts=2,
                    max_chars=240,
                    instruction=(
                        "Compact supplementary evidence for questions that require counting, ordering, or comparing "
                        "facts across sessions. Use it only when it directly supports a multi-session calculation; "
                        "do not override more specific main evidence, and answer that information is insufficient "
                        "when neither source directly supports the answer."
                    ),
                )

            # Extract retrieval details from trace for caching
            trace = search_result.trace or {}
            rrf_results = trace.get("rrf_merged", [])
            reranked_results = trace.get("reranked", [])

            # Extract cross-encoder config info
            cross_encoder_config = {}
            if hasattr(self.memory, "_cross_encoder") and self.memory._cross_encoder:
                ce = self.memory._cross_encoder
                cross_encoder_config["model"] = getattr(ce, "model_name", "unknown")
                cross_encoder_config["provider"] = getattr(ce, "provider_name", "unknown")
            else:
                cross_encoder_config["model"] = "unknown"
                cross_encoder_config["provider"] = "unknown"

            # Build coarse search results from RRF merge
            coarse_candidates = []
            for rrf_item in rrf_results:
                if isinstance(rrf_item, dict):
                    node_id = rrf_item.get("node_id", "")
                    text = rrf_item.get("text", "")
                    # Find matching result for context and metadata
                    matching_result = None
                    if search_result.results:
                        for result in search_result.results:
                            if result.id == node_id or (result.document_id and node_id in result.document_id):
                                matching_result = result
                                break

                    coarse_candidates.append(
                        CoarseSearchCandidate(
                            rank=rrf_item.get("final_rrf_rank", 0),
                            document_id=node_id,
                            text=text,
                            context=matching_result.context if matching_result else None,
                            occurred_start=matching_result.occurred_start if matching_result else None,
                            fact_type=matching_result.fact_type if matching_result else "unknown",
                            rrf_score=rrf_item.get("rrf_score", 0.0),
                            proof_count=matching_result.metadata.get("proof_count")
                            if matching_result and matching_result.metadata
                            else None,
                        )
                    )

            coarse_search_results = CoarseSearchResults(
                total_candidates=len(coarse_candidates),
                candidates=coarse_candidates,
            )

            # Build reranked results
            reranked_candidates = []
            for rerank_item in reranked_results:
                if isinstance(rerank_item, dict):
                    node_id = rerank_item.get("node_id", "")
                    text = rerank_item.get("text", "")
                    score_components = rerank_item.get("score_components", {})

                    reranked_candidates.append(
                        RerankedCandidate(
                            original_rank=rerank_item.get("rrf_rank", 0),
                            document_id=node_id,
                            text=text,
                            cross_encoder_score=rerank_item.get("rerank_score", 0.0),
                            combined_score=score_components.get("combined_score", 0.0),
                            final_rank=rerank_item.get("rerank_rank", 0),
                        )
                    )

            reranked_results_obj = RerankedResults(
                reranker_model=cross_encoder_config["model"],
                reranker_provider=cross_encoder_config["provider"],
                reranked_candidates=reranked_candidates,
            )

            retrieval_details = {
                "coarse_search_results": coarse_search_results,
                "reranked_results": reranked_results_obj,
            }

            # Extract chunks from search result
            chunks = {}
            if search_result.chunks:
                for chunk_key, chunk_info in search_result.chunks.items():
                    chunks[chunk_key] = chunk_info.model_dump()

            # Check if we have any results
            if not search_result.results:
                return (
                    "I don't have enough information to answer that question.",
                    "No relevant memories found.",
                    [],
                    {},
                    None,
                )

            # Generate answer using LLM - pass entire recall result
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(
                question, recall_result_dict, question_date, question_type, bank_id=agent_id
            )

            # Use override if provided, otherwise use the results from recall
            final_memories = (
                memories_override
                if memories_override is not None
                else recall_result_dict.get("results", [fact.model_dump() for fact in search_result.results])
            )

            return answer, reasoning, final_memories, chunks, retrieval_details
        else:
            # Integrated flow: generator does its own search (e.g., reflect API)
            # Pass empty recall result since generator doesn't need them
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(
                question, {"results": []}, question_date, question_type, bank_id=agent_id
            )

            # Use memories from generator (should not be None for integrated mode)
            final_memories = memories_override if memories_override is not None else []

            return answer, reasoning, final_memories, {}, None

    async def evaluate_qa_task(
        self,
        agent_id: str,
        qa_pairs: List[Dict],
        item_id: str,
        thinking_budget: int,
        max_tokens: int,
        max_questions: Optional[int] = None,
        semaphore: asyncio.Semaphore = None,
    ) -> List[Dict]:
        """
        Evaluate QA task with parallel question processing.

        Args:
            semaphore: Semaphore to limit concurrent question processing

        Returns:
            List of QA results
        """
        # Filter out questions without answers (category 5)
        # First, identify and log category 5 questions that will be skipped
        category_5_questions = [pair for pair in qa_pairs if pair.get("category") == 5]
        if category_5_questions:
            logging.info(f"Skipping {len(category_5_questions)} category=5 questions for {item_id}")
            for q in category_5_questions:
                logging.debug(f"  Skipped category=5 question: {q.get('question', 'N/A')[:100]}")

        # Filter out category 5 and questions without answers, preserving original indices
        indexed_pairs = [
            (orig_idx, pair)
            for orig_idx, pair in enumerate(qa_pairs)
            if pair.get("category") != 5 and pair.get("answer")
        ]
        indexed_pairs_to_eval = indexed_pairs[:max_questions] if max_questions else indexed_pairs

        # Progress output disabled for cleaner logs
        async def process_question(orig_idx: int, qa: dict):
            async with semaphore:
                question = qa["question"]
                correct_answer = qa["answer"]
                category = qa.get("category", 0)
                question_date = qa.get("question_date")

                import time

                start_time = time.time()

                try:
                    (
                        predicted_answer,
                        reasoning,
                        retrieved_memories,
                        chunks,
                        retrieval_details,
                    ) = await self.answer_question(
                        agent_id,
                        question,
                        thinking_budget,
                        max_tokens,
                        question_date,
                        category,
                    )

                    answer_time = time.time() - start_time

                    memories_without_embeddings = [
                        {k: v for k, v in mem.items() if k != "embedding"} for mem in retrieved_memories
                    ]

                    return {
                        "question_index": orig_idx,
                        "question": question,
                        "correct_answer": correct_answer,
                        "predicted_answer": predicted_answer,
                        "reasoning": reasoning,
                        "category": category,
                        "retrieved_memories": memories_without_embeddings,
                        "is_invalid": False,
                        "error": None,
                        "answer_time": answer_time,
                        "retrieval_details": retrieval_details,
                    }
                except Exception as e:
                    logging.exception(f"Failed to answer question: {question[:100]}")
                    return {
                        "question_index": orig_idx,
                        "question": question,
                        "correct_answer": correct_answer,
                        "predicted_answer": "ERROR: Failed to generate answer",
                        "reasoning": f"Error: {str(e)}",
                        "category": category,
                        "retrieved_memories": [],
                        "is_invalid": True,
                        "error": str(e),
                        "retrieval_details": None,
                    }

        question_tasks = [process_question(orig_idx, qa) for orig_idx, qa in indexed_pairs_to_eval]

        results = await asyncio.gather(*question_tasks, return_exceptions=True)
        results = [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]

        return results

    async def calculate_metrics(
        self,
        results: List[Dict],
        eval_semaphore: asyncio.Semaphore,
        item_id: Optional[str] = None,
    ) -> Dict:
        """
        Calculate evaluation metrics using parallel LLM-as-judge.

        Args:
            results: QA results to evaluate
            eval_semaphore: Run-scoped semaphore for all LLM judge requests
            item_id: Optional item ID for retrieval cache naming

        Returns:
            Dict with evaluation metrics
        """
        total = len(results)

        # Progress output disabled for cleaner logs
        async def judge_single(result):
            # Skip judging if already marked as invalid
            if result.get("is_invalid", False):
                result["is_correct"] = None
                result["correctness_reasoning"] = (
                    f"Question invalid due to error: {result.get('error', 'Unknown error')}"
                )
                return result

            try:
                is_correct, eval_reasoning, judge_time = await self.answer_evaluator.judge_answer(
                    result["question"],
                    result["correct_answer"],
                    result["predicted_answer"],
                    eval_semaphore,
                    category=result.get("category"),
                )
                result["is_correct"] = is_correct
                result["correctness_reasoning"] = eval_reasoning
                result["judge_time"] = judge_time

                if not is_correct:
                    retrieval_details = result.get("retrieval_details")
                    if retrieval_details:
                        cache = RetrievalCache(
                            question_id=f"{item_id}_{result.get('question_index', 0)}",
                            question=result["question"],
                            question_date=result.get("question_date"),
                            category=str(result.get("category", "unknown")),
                            correct_answer=result["correct_answer"],
                            generated_answer=result["predicted_answer"],
                            is_correct=is_correct,
                            judge_reasoning=eval_reasoning,
                            retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                            coarse_search_results=retrieval_details.get("coarse_search_results"),
                            reranked_results=retrieval_details.get("reranked_results"),
                        )
                        if self._diagnostic_cache_dir is not None:
                            self._save_retrieval_cache(cache, self._diagnostic_cache_dir)

                return result
            except Exception as e:
                logging.exception(f"Failed to judge answer for question: {result.get('question', 'unknown')[:100]}")
                result["is_invalid"] = True
                result["is_correct"] = None
                result["correctness_reasoning"] = f"Judge error: {str(e)}"
                result["error"] = str(e)
                return result

        judgment_tasks = [judge_single(result) for result in results]
        judged_results = await asyncio.gather(*judgment_tasks, return_exceptions=True)
        judged_results = [r if not isinstance(r, Exception) else {"error": str(r)} for r in judged_results]

        # Calculate stats
        correct = sum(1 for r in judged_results if r.get("is_correct", False))
        invalid = sum(1 for r in judged_results if r.get("is_invalid", False))
        valid_total = total - invalid
        category_stats = {}

        for result in judged_results:
            category = result.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0, "invalid": 0}
            category_stats[category]["total"] += 1
            if result.get("is_invalid", False):
                category_stats[category]["invalid"] += 1
            elif result.get("is_correct", False):
                category_stats[category]["correct"] += 1

        # Provider and processing failures count as incorrect in the primary
        # benchmark metric. Keep valid-only accuracy as a diagnostic.
        accuracy = (correct / total * 100) if total > 0 else 0
        valid_accuracy = (correct / valid_total * 100) if valid_total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "invalid": invalid,
            "valid_total": valid_total,
            "valid_accuracy": valid_accuracy,
            "category_stats": category_stats,
            "detailed_results": judged_results,
        }

    def _expected_document_snapshots(
        self,
        item: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Optional[str]]], List[str]]:
        """Build the same content and metadata identities used by Retain."""

        prepared = self.dataset.prepare_sessions_for_ingestion(item)
        normalized = tuple(
            normalize_content_item(content, source_index=source_index) for source_index, content in enumerate(prepared)
        )
        explicit_ids = tuple(
            dict.fromkeys(content.document_id for content in normalized if content.document_id is not None)
        )
        grouped: Dict[str, List[Any]] = {}
        unverifiable: List[str] = []

        if len(explicit_ids) == 1:
            # Retain assigns missing-ID items to the sole explicit document.
            grouped[explicit_ids[0]] = list(normalized)
        elif len(explicit_ids) > 1:
            for content in normalized:
                if content.document_id is None:
                    unverifiable.append(f"source_index:{content.source_index}")
                    continue
                grouped.setdefault(content.document_id, []).append(content)
        elif normalized:
            # Retain generates a random document ID when the whole batch omits
            # IDs. Such a bank cannot be safely matched to a later dataset item.
            unverifiable.extend(f"source_index:{content.source_index}" for content in normalized)

        snapshots: Dict[str, Dict[str, Optional[str]]] = {}
        for document_id, contents in grouped.items():
            if any(content.update_mode.value == "append" for content in contents):
                # The final content hash also depends on pre-existing text,
                # which is not part of the submitted benchmark item.
                unverifiable.append(document_id)
                continue
            retain_params, _ = retain_document_metadata(tuple(contents))
            snapshots[document_id] = {
                "content_hash": compute_document_hash("\n".join(content.content for content in contents)),
                "context": retain_params.get("context"),
                "event_date": retain_params.get("event_date"),
            }
        return snapshots, unverifiable

    @staticmethod
    def _retain_params_mapping(value: Any) -> Dict[str, Any]:
        """Normalize PostgreSQL/Oracle JSON representations to a mapping."""

        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        raise ValueError(f"retain_params must be a JSON object, got {type(value).__name__}")

    async def _audit_durable_ingestion(
        self,
        item: Dict[str, Any],
        agent_id: str,
        *,
        reject_unexpected_documents: bool = True,
        allowed_document_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Verify a bank is the exact durable Retain output expected for an item.

        Zero extracted facts remain valid because source chunks are recallable.
        Missing/empty documents, stale content or Retain metadata, unexpected
        documents, and in-flight writes make an item-scoped bank unsafe to
        reuse.
        """

        expected, unverifiable = self._expected_document_snapshots(item)
        expected_document_ids = list(expected)
        report: Dict[str, Any] = {
            "item_id": self.dataset.get_item_id(item),
            "bank_id": agent_id,
            "expected_documents": len(expected_document_ids),
            "durable_documents": 0,
            "missing_documents": [],
            "documents_without_chunks": [],
            "documents_without_facts": [],
            "unexpected_documents": [],
            "additional_documents": [],
            "inflight_documents": [],
            "content_hash_mismatches": [],
            "context_mismatches": [],
            "event_date_mismatches": [],
            "unverifiable_documents": unverifiable,
            "invalid_retain_params": [],
        }

        pool = await self.memory._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT d.id,
                       d.content_hash,
                       d.retain_params,
                       (
                           SELECT COUNT(*)
                           FROM {fq_table("chunks")} AS c
                           WHERE c.bank_id = d.bank_id AND c.document_id = d.id
                       ) AS chunk_count,
                       (
                           SELECT COUNT(*)
                           FROM {fq_table("memory_units")} AS m
                           WHERE m.bank_id = d.bank_id AND m.document_id = d.id
                       ) AS fact_count
                FROM {fq_table("documents")} AS d
                WHERE d.bank_id = $1
                """,
                agent_id,
            )

        by_id = {str(row["id"]): row for row in rows}
        observed_ids = set(by_id)
        expected_ids = set(expected_document_ids)
        report["missing_documents"] = sorted(expected_ids - observed_ids)
        allowed_ids = set(allowed_document_ids) if allowed_document_ids is not None else expected_ids
        unexpected_documents = sorted(observed_ids - allowed_ids)
        if reject_unexpected_documents or allowed_document_ids is not None:
            report["unexpected_documents"] = unexpected_documents
        else:
            report["additional_documents"] = unexpected_documents

        for document_id, row in by_id.items():
            content_hash = str(row["content_hash"] or "")
            if content_hash.startswith("retain-inflight:"):
                report["inflight_documents"].append(document_id)
            if document_id not in expected:
                continue

            if int(row["chunk_count"] or 0) == 0:
                report["documents_without_chunks"].append(document_id)
            if int(row["fact_count"] or 0) == 0:
                report["documents_without_facts"].append(document_id)

            snapshot = expected[document_id]
            if content_hash != snapshot["content_hash"]:
                report["content_hash_mismatches"].append(
                    {
                        "document_id": document_id,
                        "expected": snapshot["content_hash"],
                        "actual": content_hash,
                    }
                )

            try:
                retain_params = self._retain_params_mapping(row["retain_params"])
            except (TypeError, ValueError, json.JSONDecodeError):
                report["invalid_retain_params"].append(document_id)
                continue

            if retain_params.get("context") != snapshot["context"]:
                report["context_mismatches"].append(
                    {
                        "document_id": document_id,
                        "expected": snapshot["context"],
                        "actual": retain_params.get("context"),
                    }
                )
            if retain_params.get("event_date") != snapshot["event_date"]:
                report["event_date_mismatches"].append(
                    {
                        "document_id": document_id,
                        "expected": snapshot["event_date"],
                        "actual": retain_params.get("event_date"),
                    }
                )

        invalid_document_ids = {
            *report["missing_documents"],
            *report["documents_without_chunks"],
            *report["inflight_documents"],
            *report["invalid_retain_params"],
        }
        for mismatch_field in ("content_hash_mismatches", "context_mismatches", "event_date_mismatches"):
            invalid_document_ids.update(mismatch["document_id"] for mismatch in report[mismatch_field])
        report["durable_documents"] = sum(
            1 for document_id in expected_document_ids if document_id not in invalid_document_ids
        )

        failure_fields = (
            "missing_documents",
            "documents_without_chunks",
            "unexpected_documents",
            "inflight_documents",
            "content_hash_mismatches",
            "context_mismatches",
            "event_date_mismatches",
            "unverifiable_documents",
            "invalid_retain_params",
        )
        if any(report[field] for field in failure_fields):
            raise IngestionIntegrityError(report)
        return report

    async def _preflight_reusable_items(
        self,
        items: Iterable[Dict[str, Any]],
        agent_id: str,
        *,
        clear_agent_per_item: bool,
        require_all: bool,
    ) -> set[str]:
        """Return item IDs with an exact reusable bank, before any QA starts."""

        reusable_item_ids: set[str] = set()
        failures: List[IngestionIntegrityError] = []
        item_list = list(items)
        shared_allowed_document_ids: Optional[set[str]] = None
        if not clear_agent_per_item:
            shared_allowed_document_ids = set()
            for item in item_list:
                snapshots, _ = self._expected_document_snapshots(item)
                shared_allowed_document_ids.update(snapshots)
        for item in item_list:
            item_id = self.dataset.get_item_id(item)
            item_agent_id = f"{agent_id}_{item_id}" if clear_agent_per_item else agent_id
            try:
                await self._audit_durable_ingestion(
                    item,
                    item_agent_id,
                    reject_unexpected_documents=True,
                    allowed_document_ids=shared_allowed_document_ids,
                )
            except IngestionIntegrityError as exc:
                failures.append(exc)
            else:
                reusable_item_ids.add(item_id)

        if failures and require_all:
            failed_banks = [str(exc.report.get("bank_id", "unknown")) for exc in failures]
            raise IngestionIntegrityError(
                {
                    "item_id": "reuse-preflight",
                    "bank_id": agent_id,
                    "failed_banks": failed_banks,
                    "bank_reports": [exc.report for exc in failures],
                }
            )
        return reusable_item_ids

    async def process_single_item(
        self,
        item: Dict,
        agent_id: str,
        i: int,
        total_items: int,
        thinking_budget: int,
        max_tokens: int,
        max_questions_per_item: Optional[int],
        skip_ingestion: bool,
        question_semaphore: asyncio.Semaphore,
        eval_semaphore: asyncio.Semaphore,
        clear_this_agent: bool = True,
        wait_consolidation: bool = False,
        ingest_only: bool = False,
        skip_if_already_ingested: bool = False,
        force_reingest: bool = False,
        bank_is_item_scoped: bool = True,
    ) -> Dict:
        """
        Process a single item (ingest + evaluate).

        Args:
            clear_this_agent: Whether to clear this agent's data before ingesting.
                             Set to False to skip clearing (e.g., when agent_id is shared and already cleared)
            wait_consolidation: If True, wait for consolidation to complete before evaluating QA.
            skip_if_already_ingested: If True and agent already has data, skip ingestion entirely.

        Returns:
            Result dict with metrics
        """
        item_id = self.dataset.get_item_id(item)

        console.print(f"\n[bold blue]Item {i}/{total_items}[/bold blue] (ID: {item_id})")

        step = 1
        num_sessions = 0
        if not skip_ingestion:
            # Check if already ingested (for smart resume)
            already_ingested = False
            if skip_if_already_ingested and not force_reingest:
                try:
                    await self._audit_durable_ingestion(
                        item,
                        agent_id,
                        reject_unexpected_documents=bank_is_item_scoped,
                    )
                except IngestionIntegrityError:
                    already_ingested = False
                else:
                    already_ingested = True
                    console.print(f"  [{step}] [yellow]⊘[/yellow] Skipping - already ingested")

            if not already_ingested or force_reingest:
                # Clear agent data before ingesting (always clear when force_reingest)
                if clear_this_agent or force_reingest:
                    console.print(f"  [{step}] Clearing previous agent data...")
                    await self.memory.delete_bank(agent_id, request_context=RequestContext())
                    console.print(f"      [green]✓[/green] Cleared '{agent_id}' agent data")

                # Apply template if configured
                if self.template_path:
                    step += 1
                    console.print(f"  [{step}] Applying bank template...")
                    await self.apply_template(agent_id, self.template_path)
                    console.print("      [green]✓[/green] Template applied")

                # Ingest conversation
                step += 1
                console.print(f"  [{step}] Ingesting conversation (batch mode)...")
                num_sessions = await self.ingest_conversation(item, agent_id, wait_for_consolidation=False)
                console.print(f"      [green]✓[/green] Ingested {num_sessions} sessions")

        step += 1
        console.print(f"  [{step}] Auditing durable ingestion...")
        ingestion_audit = await self._audit_durable_ingestion(
            item,
            agent_id,
            reject_unexpected_documents=bank_is_item_scoped,
        )
        console.print(
            "      [green]✓[/green] "
            f"{ingestion_audit['durable_documents']}/{ingestion_audit['expected_documents']} "
            "documents have durable chunks"
        )

        # Wait for consolidation before evaluating if requested
        if wait_consolidation:
            step += 1
            console.print(f"  [{step}] Waiting for consolidation...")
            await self._wait_for_consolidation(agent_id)

        # Ingest-only mode: skip evaluation
        if ingest_only:
            console.print("  [green]✓[/green] Ingest complete (skipping evaluation)")
            return {
                "item_id": item_id,
                "metrics": {"correct": 0, "total": 0, "invalid": 0, "accuracy": 0.0},
                "num_sessions": num_sessions,
                "ingest_only": True,
                "ingestion_audit": ingestion_audit,
            }

        # Evaluate QA
        step += 1
        qa_pairs = self.dataset.get_qa_pairs(item)
        console.print(f"  [{step}] Evaluating {len(qa_pairs)} QA pairs (parallel)...")
        qa_results = await self.evaluate_qa_task(
            agent_id,
            qa_pairs,
            item_id,
            thinking_budget,
            max_tokens,
            max_questions_per_item,
            question_semaphore,
        )

        # Calculate metrics
        step += 1
        console.print(f"  [{step}] Calculating metrics...")
        metrics = await self.calculate_metrics(qa_results, eval_semaphore, item_id)

        console.print(
            f"      [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})"
        )

        return {
            "item_id": item_id,
            "metrics": metrics,
            "num_sessions": num_sessions,
            "ingestion_audit": ingestion_audit,
        }

    async def run(
        self,
        dataset_path: Path,
        agent_id: str,
        max_items: Optional[int] = None,
        max_questions_per_item: Optional[int] = None,
        thinking_budget: int = 500,
        max_tokens: int = 4096,
        skip_ingestion: bool = False,
        max_concurrent_questions: int = 10,
        eval_semaphore_size: int = 10,
        clear_agent_per_item: bool = False,
        specific_item: Optional[Union[str, Iterable[str]]] = None,
        separate_ingestion_phase: bool = False,
        filln: bool = False,
        max_concurrent_items: int = 1,  # Max concurrent items (conversations) to process in parallel
        output_path: Optional[Path] = None,  # Path to save results incrementally
        merge_with_existing: bool = False,  # Whether to merge with existing results
        wait_consolidation: bool = False,  # Wait for consolidation to complete before evaluating QA
        template_path: Optional[str] = None,  # Path to a bank template manifest to apply before ingestion
        ingest_only: bool = False,  # Only ingest, skip evaluation
        force_reingest: bool = False,  # If True, always re-ingest even if data already exists
        rerun_invalid_existing: bool = False,  # Resume mode: rerun invalid existing item results
        run_manifest: Optional[Dict[str, Any]] = None,  # Stable metadata included in every checkpoint
        model_config: Optional[Mapping[str, Mapping[str, str]]] = None,  # Executed-stage model identities
    ) -> Dict[str, Any]:
        """
        Run the full benchmark evaluation.

        Args:
            dataset_path: Path to dataset file
            agent_id: Agent ID to use
            max_items: Maximum number of items to evaluate
            max_questions_per_item: Maximum questions per item
            thinking_budget: Thinking budget for search
            max_tokens: Maximum tokens to retrieve from memories
            skip_ingestion: Skip ingestion and use existing data
            max_concurrent_questions: Max concurrent question processing
            eval_semaphore_size: Max concurrent LLM judge requests
            clear_agent_per_item: Use unique agent ID per item for isolation (deprecated when separate_ingestion_phase=True)
            specific_item: If provided, only run this specific item ID (e.g., conversation)
            separate_ingestion_phase: If True, ingest all data first, then evaluate all questions (single agent)
            filln: If True, skip item IDs already complete in the result artifact
            max_concurrent_items: Max concurrent items to process in parallel (requires clear_agent_per_item=True)

        Returns:
            Dict with complete benchmark results
        """
        console.print("\n[bold cyan]Benchmark Evaluation[/bold cyan]")
        console.print("=" * 80)

        for name, value in (
            ("max_concurrent_items", max_concurrent_items),
            ("max_concurrent_questions", max_concurrent_questions),
            ("eval_semaphore_size", eval_semaphore_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")

        self._run_manifest = run_manifest
        selected_model_config = model_config or get_model_config()
        self._model_config = {role: dict(settings) for role, settings in selected_model_config.items()}
        self._diagnostic_cache_dir = output_path.parent / "retrieval_cache" if output_path is not None else None

        # Print model configuration
        print_model_config(self._model_config)

        # Load dataset
        console.print(f"\n[1] Loading dataset from {dataset_path}...")
        items = self.dataset.load(dataset_path, max_items)

        # Filter for specific item(s) if requested
        if specific_item is not None:
            target_ids = {specific_item} if isinstance(specific_item, str) else set(specific_item)
            items = [item for item in items if self.dataset.get_item_id(item) in target_ids]
            if not items:
                console.print(f"    [red]✗[/red] No item found with ID(s): {sorted(target_ids)}")
                raise ValueError(f"No items matching ID(s) {sorted(target_ids)} found in dataset")
            console.print(f"    [green]✓[/green] Filtering to {len(items)} item(s): {sorted(target_ids)}")

        console.print(f"    [green]✓[/green] Loaded {len(items)} items")

        # Initialize memory system
        console.print("\n[2] Initializing memory system...")
        if template_path:
            self.template_path = template_path
            console.print(f"    Bank template: {template_path}")
        console.print("    [green]✓[/green] Memory system initialized")

        # Start a background worker poller when we need to wait for consolidation.
        # Consolidation is submitted as an async task by retain_batch_async, but
        # without a running worker those tasks sit in the queue forever.
        poller_task = None
        poller = None
        if wait_consolidation:
            from hms_api.worker.poller import WorkerPoller

            poller = WorkerPoller(
                backend=self.memory._backend,
                worker_id="benchmark-runner-worker",
                executor=self.memory.execute_task,
                poll_interval_ms=500,
                max_slots=4,
            )
            poller_task = asyncio.create_task(poller.run())
            console.print("    [green]✓[/green] Background worker started (for consolidation)")

        try:
            return await self._run_inner(
                items,
                agent_id,
                thinking_budget,
                max_tokens,
                skip_ingestion,
                max_questions_per_item,
                max_concurrent_questions,
                eval_semaphore_size,
                clear_agent_per_item,
                specific_item,
                separate_ingestion_phase,
                filln,
                max_concurrent_items,
                output_path,
                merge_with_existing,
                wait_consolidation,
                ingest_only,
                force_reingest,
                rerun_invalid_existing,
            )
        finally:
            if poller and poller_task:
                await poller.shutdown_graceful(timeout=60.0)
                poller_task.cancel()
                try:
                    await poller_task
                except asyncio.CancelledError:
                    pass
                console.print("    [green]✓[/green] Background worker stopped")

    async def _run_inner(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        max_concurrent_questions: int,
        eval_semaphore_size: int,
        clear_agent_per_item: bool,
        specific_item: Any,
        separate_ingestion_phase: bool,
        filln: bool,
        max_concurrent_items: int,
        output_path: Optional[Path],
        merge_with_existing: bool,
        wait_consolidation: bool,
        ingest_only: bool = False,
        force_reingest: bool = False,
        rerun_invalid_existing: bool = False,
    ) -> Dict[str, Any]:
        if separate_ingestion_phase:
            # New two-phase approach: ingest all, then evaluate all
            return await self._run_two_phase(
                items,
                agent_id,
                thinking_budget,
                max_tokens,
                skip_ingestion,
                max_questions_per_item,
                max_concurrent_questions,
                eval_semaphore_size,
                output_path,
                merge_with_existing,
            )
        else:
            # Original approach: process each item independently
            return await self._run_single_phase(
                items,
                agent_id,
                thinking_budget,
                max_tokens,
                skip_ingestion,
                max_questions_per_item,
                max_concurrent_questions,
                eval_semaphore_size,
                clear_agent_per_item,
                filln,
                max_concurrent_items,
                output_path,
                merge_with_existing,
                wait_consolidation,
                ingest_only,
                force_reingest,
                rerun_invalid_existing,
            )

    async def _run_single_phase(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        max_concurrent_questions: int,
        eval_semaphore_size: int,
        clear_agent_per_item: bool,
        filln: bool = False,
        max_concurrent_items: int = 1,
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
        wait_consolidation: bool = False,
        ingest_only: bool = False,
        force_reingest: bool = False,
        rerun_invalid_existing: bool = False,
    ) -> Dict[str, Any]:
        """Original single-phase approach: process each item independently."""
        # Create semaphore for question processing
        question_semaphore = asyncio.Semaphore(max_concurrent_questions)
        # This semaphore is deliberately shared by every item in the run.
        eval_semaphore = asyncio.Semaphore(eval_semaphore_size)

        # Process items - either in parallel or sequentially
        if max_concurrent_items > 1 and clear_agent_per_item:
            # Parallel item processing (requires unique agent IDs)
            all_results = await self._process_items_parallel(
                items,
                agent_id,
                thinking_budget,
                max_tokens,
                skip_ingestion,
                max_questions_per_item,
                question_semaphore,
                eval_semaphore,
                filln,
                max_concurrent_items,
                output_path,
                merge_with_existing,
                wait_consolidation,
                ingest_only,
                force_reingest,
                rerun_invalid_existing,
            )
        else:
            # Sequential item processing (original behavior)
            all_results = await self._process_items_sequential(
                items,
                agent_id,
                thinking_budget,
                max_tokens,
                skip_ingestion,
                max_questions_per_item,
                question_semaphore,
                eval_semaphore,
                clear_agent_per_item,
                filln,
                output_path,
                merge_with_existing,
                wait_consolidation,
                ingest_only,
                force_reingest,
                rerun_invalid_existing,
            )

        # Calculate overall metrics
        total_correct = sum(r["metrics"]["correct"] for r in all_results)
        total_questions = sum(r["metrics"]["total"] for r in all_results)
        total_invalid = sum(r["metrics"].get("invalid", 0) for r in all_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        valid_only_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "total_invalid": total_invalid,
            "total_valid": total_valid,
            "valid_only_accuracy": valid_only_accuracy,
            "num_items": len(all_results),
            "model_config": getattr(self, "_model_config", None) or get_model_config(),
            "item_results": all_results,
        }

    async def _process_items_sequential(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        question_semaphore: asyncio.Semaphore,
        eval_semaphore: asyncio.Semaphore,
        clear_agent_per_item: bool,
        filln: bool,
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
        wait_consolidation: bool = False,
        ingest_only: bool = False,
        force_reingest: bool = False,
        rerun_invalid_existing: bool = False,
    ) -> List[Dict]:
        """Process items sequentially (original behavior)."""
        all_results = []
        existing_item_ids = set()
        resume_complete_item_ids = set()

        # Load existing results if merge_with_existing is True
        if merge_with_existing and output_path and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if "item_results" in existing_data:
                    all_results = existing_data["item_results"]
                    existing_item_ids = {r["item_id"] for r in all_results}
                    resume_complete_item_ids = {r["item_id"] for r in all_results if _result_is_resume_complete(r)}
                    console.print(f"[cyan]Loaded {len(all_results)} existing results from {output_path}[/cyan]")

        # Pre-load durable banks for reuse and ingest-only fill modes.
        ingested_item_ids = set()
        if skip_ingestion or ingest_only:
            console.print("[cyan]Auditing retained banks against selected dataset items...[/cyan]")
            ingested_item_ids = await self._preflight_reusable_items(
                items,
                agent_id,
                clear_agent_per_item=clear_agent_per_item,
                require_all=skip_ingestion,
            )
            if ingested_item_ids:
                console.print(f"[cyan]Found {len(ingested_item_ids)} exact reusable item banks[/cyan]")
            else:
                console.print("[cyan]No exact reusable item banks found[/cyan]")
            if ingest_only and not skip_ingestion and not clear_agent_per_item:
                requested_item_ids = {self.dataset.get_item_id(item) for item in items}
                if ingested_item_ids != requested_item_ids:
                    # Repairing any item in a shared bank may clear or replace
                    # state needed by otherwise reusable items. Rebuild the
                    # selected union together instead of skipping a stale mix.
                    ingested_item_ids.clear()

        for i, item in enumerate(items, 1):
            item_id = self.dataset.get_item_id(item)
            # Use unique agent ID per item if requested (for isolation in benchmarks like LongMemEval)
            # This avoids deadlocks from deleting agent data
            if clear_agent_per_item:
                item_agent_id = f"{agent_id}_{item_id}"
                # Always clear for unique agents (each agent_id is used only once)
                clear_this_agent = True
            else:
                item_agent_id = agent_id
                # Only clear on first item for shared agent_id
                clear_this_agent = i == 1

            # For ingest_only mode, skip already ingested items
            if ingest_only and not skip_ingestion and not force_reingest and item_id in ingested_item_ids:
                console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                console.print("  [yellow]⊘[/yellow] Skipping - already ingested")
                continue

            # Then check fill status (results file)
            if filln and not force_reingest:
                skip_item_ids = resume_complete_item_ids if rerun_invalid_existing else existing_item_ids
                if item_id in skip_item_ids:
                    console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                    console.print("  [yellow]⊘[/yellow] Skipping - already has results in output file")
                    continue

            result = await self.process_single_item(
                item,
                item_agent_id,
                i,
                len(items),
                thinking_budget,
                max_tokens,
                max_questions_per_item,
                skip_ingestion,
                question_semaphore,
                eval_semaphore,
                clear_this_agent,
                wait_consolidation,
                ingest_only,
                skip_if_already_ingested=False,
                force_reingest=force_reingest,
                bank_is_item_scoped=clear_agent_per_item,
            )

            # Replace existing result or append new one
            result_item_id = result["item_id"]
            if result_item_id in existing_item_ids:
                # Replace existing result
                all_results = [r for r in all_results if r["item_id"] != result_item_id]
                console.print(f"  [cyan]↻[/cyan] Updating existing result for {result_item_id}")
            all_results.append(result)
            existing_item_ids.add(result_item_id)

            # Save results incrementally after each item
            if output_path:
                self._save_incremental_results(all_results, output_path)

        return all_results

    async def _process_items_parallel(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        question_semaphore: asyncio.Semaphore,
        eval_semaphore: asyncio.Semaphore,
        filln: bool,
        max_concurrent_items: int,
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
        wait_consolidation: bool = False,
        ingest_only: bool = False,
        force_reingest: bool = False,
        rerun_invalid_existing: bool = False,
    ) -> List[Dict]:
        """Process items in parallel (requires unique agent IDs per item)."""
        # Load existing results if merge_with_existing is True
        all_results = []
        existing_item_ids = set()
        resume_complete_item_ids = set()
        result_lock = asyncio.Lock()  # Lock for thread-safe updates to all_results

        if merge_with_existing and output_path and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if "item_results" in existing_data:
                    all_results = existing_data["item_results"]
                    existing_item_ids = {r["item_id"] for r in all_results}
                    resume_complete_item_ids = {r["item_id"] for r in all_results if _result_is_resume_complete(r)}
                    console.print(f"[cyan]Loaded {len(all_results)} existing results from {output_path}[/cyan]")

        # Pre-load durable banks when evaluating retained data or filling an
        # ingest-only run.
        ingested_item_ids = set()
        if skip_ingestion or ingest_only:
            console.print("[cyan]Auditing retained banks against selected dataset items...[/cyan]")
            ingested_item_ids = await self._preflight_reusable_items(
                items,
                agent_id,
                clear_agent_per_item=True,
                require_all=skip_ingestion,
            )
            if ingested_item_ids:
                console.print(f"[cyan]Found {len(ingested_item_ids)} exact reusable item banks[/cyan]")
            else:
                console.print("[cyan]No exact reusable item banks found[/cyan]")

        # Create semaphore for item-level parallelism
        item_semaphore = asyncio.Semaphore(max_concurrent_items)

        async def process_item_wrapper(i: int, item: Dict) -> Optional[Dict]:
            """Wrapper to process a single item with semaphore control."""
            async with item_semaphore:
                item_id = self.dataset.get_item_id(item)
                item_agent_id = f"{agent_id}_{item_id}"

                # For ingest_only mode, skip already ingested items
                if ingest_only and not skip_ingestion and not force_reingest and item_id in ingested_item_ids:
                    console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                    console.print("  [yellow]⊘[/yellow] Skipping - already ingested")
                    return None

                # Then check fill status (results file)
                if filln and not force_reingest:
                    skip_item_ids = resume_complete_item_ids if rerun_invalid_existing else existing_item_ids
                    if item_id in skip_item_ids:
                        console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                        console.print("  [yellow]⊘[/yellow] Skipping - already has results in output file")
                        return None

                # Process the item
                result = await self.process_single_item(
                    item,
                    item_agent_id,
                    i,
                    len(items),
                    thinking_budget,
                    max_tokens,
                    max_questions_per_item,
                    skip_ingestion,
                    question_semaphore,
                    eval_semaphore,
                    clear_this_agent=True,
                    wait_consolidation=wait_consolidation,
                    ingest_only=ingest_only,
                    skip_if_already_ingested=False,
                    force_reingest=force_reingest,
                    bank_is_item_scoped=True,
                )
                return result

        # Create all tasks
        tasks = [process_item_wrapper(i, item) for i, item in enumerate(items, 1)]

        # Run in parallel and collect results incrementally
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result is not None:
                async with result_lock:
                    # Replace existing result or append new one
                    result_item_id = result["item_id"]
                    if result_item_id in existing_item_ids:
                        # Replace existing result
                        all_results = [r for r in all_results if r["item_id"] != result_item_id]
                        console.print(f"  [cyan]↻[/cyan] Updating existing result for {result_item_id}")
                    all_results.append(result)
                    existing_item_ids.add(result_item_id)

                    # Save results incrementally after each item completes
                    if output_path:
                        self._save_incremental_results(all_results, output_path)

        return all_results

    async def _run_two_phase(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        max_concurrent_questions: int,
        eval_semaphore_size: int,
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Two-phase approach: ingest all data into single agent, then evaluate all questions.

        More realistic scenario where agent accumulates memories over time.
        """
        # Phase 1: Ingestion
        if not skip_ingestion:
            # Calculate and display data statistics
            console.print("\n[3] Analyzing data to be ingested...")
            stats = self.calculate_data_stats(items)
            console.print(f"    [cyan]Total items:[/cyan] {stats['total_items']}")
            console.print(f"    [cyan]Total sessions:[/cyan] {stats['total_sessions']}")
            console.print(f"    [cyan]Total characters:[/cyan] {stats['total_chars']:,}")
            console.print(f"    [cyan]Avg session length:[/cyan] {stats['avg_session_length']:.0f} chars")
            console.print(
                f"    [cyan]Session length range:[/cyan] {stats['min_session_length']}-{stats['max_session_length']} chars"
            )

            console.print(f"\n[4] Phase 1: Ingesting all data into agent '{agent_id}'...")
            console.print("    [yellow]Clearing previous agent data...[/yellow]")
            await self.memory.delete_bank(agent_id, request_context=RequestContext())
            console.print("    [green]✓[/green] Cleared agent data")

            # Apply template if configured
            if self.template_path:
                console.print("    [yellow]Applying bank template...[/yellow]")
                await self.apply_template(agent_id, self.template_path)
                console.print("    [green]✓[/green] Template applied")

            # Collect all sessions and send in one batch (with auto-chunking)
            console.print("    [yellow]Collecting sessions from all items...[/yellow]")
            all_sessions = []
            for item in items:
                item_sessions = self.dataset.prepare_sessions_for_ingestion(item)
                all_sessions.extend(item_sessions)

            console.print(f"    [cyan]Collected {len(all_sessions)} sessions from {len(items)} items[/cyan]")
            console.print("    [yellow]Ingesting in one batch (auto-chunks if needed)...[/yellow]")

            max_retries = 3
            retry_delay = 2.0
            ingest_success = False

            for attempt in range(max_retries):
                try:
                    await self.memory.retain_batch_async(
                        bank_id=agent_id, contents=all_sessions, request_context=RequestContext()
                    )
                    ingest_success = True
                    break
                except ValueError as e:
                    error_msg = str(e)
                    if "Batch contains duplicate document_ids" in error_msg:
                        console.print(
                            f"    [yellow]⚠[/yellow] Duplicate document_id error on attempt {attempt + 1}, "
                            f"deduplicating session batch..."
                        )
                        unique_sessions = []
                        seen_doc_ids = set()
                        for session in all_sessions:
                            doc_id = session.get("document_id")
                            if doc_id and doc_id in seen_doc_ids:
                                continue
                            unique_sessions.append(session)
                            if doc_id:
                                seen_doc_ids.add(doc_id)
                        all_sessions = unique_sessions
                        console.print(f"    [cyan]Deduplicated to {len(all_sessions)} sessions[/cyan]")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        raise
                except Exception as e:
                    console.print(f"    [yellow]⚠[/yellow] Ingestion error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5

            if not ingest_success:
                raise RuntimeError("Ingestion exhausted all retry attempts without completing")
            console.print(f"    [green]✓[/green] Ingested {len(all_sessions)} sessions from {len(items)} items")
        else:
            console.print("\n[3] Skipping ingestion (using existing data)")

        console.print("    [cyan]Auditing retained documents against all selected items...[/cyan]")
        await self._preflight_reusable_items(
            items,
            agent_id,
            clear_agent_per_item=False,
            require_all=True,
        )

        # Phase 2: Evaluation
        console.print("\n[5] Phase 2: Evaluating all questions...")

        # Create semaphore for question processing
        question_semaphore = asyncio.Semaphore(max_concurrent_questions)
        eval_semaphore = asyncio.Semaphore(eval_semaphore_size)

        all_results = []
        for i, item in enumerate(items, 1):
            item_id = self.dataset.get_item_id(item)
            console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")

            # Get QA pairs
            qa_pairs = self.dataset.get_qa_pairs(item)
            console.print(f"  Evaluating {len(qa_pairs)} QA pairs (parallel)...")

            qa_results = await self.evaluate_qa_task(
                agent_id,
                qa_pairs,
                item_id,
                thinking_budget,
                max_tokens,
                max_questions_per_item,
                question_semaphore,
            )

            # Calculate metrics
            metrics = await self.calculate_metrics(qa_results, eval_semaphore, item_id)
            console.print(
                f"  [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})"
            )

            all_results.append(
                {
                    "item_id": item_id,
                    "metrics": metrics,
                    "num_sessions": -1,  # Not tracked in two-phase mode
                }
            )

        # Calculate overall metrics
        total_correct = sum(r["metrics"]["correct"] for r in all_results)
        total_questions = sum(r["metrics"]["total"] for r in all_results)
        total_invalid = sum(r["metrics"].get("invalid", 0) for r in all_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        valid_only_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "total_invalid": total_invalid,
            "total_valid": total_valid,
            "valid_only_accuracy": valid_only_accuracy,
            "num_items": len(all_results),
            "model_config": getattr(self, "_model_config", None) or get_model_config(),
            "item_results": all_results,
        }

    def display_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table."""
        console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")

        # Display model configuration
        if "model_config" in results:
            config = results["model_config"]
            console.print("[bold cyan]Model Configuration:[/bold cyan]")
            console.print(f"  HMS:         {config['hms']['provider']}/{config['hms']['model']}")
            if "retain" in config:
                retain_config = config["retain"]
                console.print(f"  Retain:      {format_retain_model_config(retain_config)}")
            console.print(
                f"  Answer Generation: {config['answer_generation']['provider']}/{config['answer_generation']['model']}"
            )
            console.print(f"  LLM Judge:         {config['judge']['provider']}/{config['judge']['model']}")
            console.print()

        # Display results table
        table = Table(title="Benchmark Results", box=box.ROUNDED)
        table.add_column("Item ID", style="cyan")
        table.add_column("Sessions", justify="right", style="yellow")
        table.add_column("Questions", justify="right", style="blue")
        table.add_column("Correct", justify="right", style="green")
        table.add_column("Invalid", justify="right", style="red")
        table.add_column("Accuracy", justify="right", style="magenta")

        for result in results["item_results"]:
            metrics = result["metrics"]
            invalid_count = metrics.get("invalid", 0)
            invalid_str = str(invalid_count) if invalid_count > 0 else "-"
            table.add_row(
                result["item_id"],
                str(result["num_sessions"]),
                str(metrics["total"]),
                str(metrics["correct"]),
                invalid_str,
                f"{metrics['accuracy']:.1f}%",
            )

        overall_invalid = results.get("total_invalid", 0)
        invalid_str = str(overall_invalid) if overall_invalid > 0 else "-"
        table.add_row(
            "[bold]OVERALL[/bold]",
            "-",
            f"[bold]{results['total_questions']}[/bold]",
            f"[bold]{results['total_correct']}[/bold]",
            f"[bold]{invalid_str}[/bold]",
            f"[bold]{results['overall_accuracy']:.1f}%[/bold]",
        )

        console.print(table)

        # Display note about invalid questions if any
        if overall_invalid > 0:
            console.print(
                f"\n[yellow]Note: {overall_invalid} question(s) failed during processing and count as incorrect "
                f"in the primary accuracy. Valid-only accuracy: {results.get('valid_only_accuracy', 0.0):.1f}%.[/yellow]"
            )

    def merge_results(self, new_results: Dict[str, Any], existing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new results into existing results.

        Updates or adds item results, then recalculates overall metrics.

        Args:
            new_results: New results to merge (typically from a specific item run)
            existing_results: Existing results to merge into

        Returns:
            Merged results with updated overall metrics
        """
        # Start with existing item results
        merged_item_results = existing_results.get("item_results", [])

        # Update or add new item results
        for new_item in new_results["item_results"]:
            item_id = new_item["item_id"]

            # Find if item already exists
            found = False
            for i, existing_item in enumerate(merged_item_results):
                if existing_item["item_id"] == item_id:
                    # Replace existing item result
                    merged_item_results[i] = new_item
                    found = True
                    console.print(f"    [yellow]→[/yellow] Updated results for item: {item_id}")
                    break

            if not found:
                # Add new item result
                merged_item_results.append(new_item)
                console.print(f"    [green]+[/green] Added results for item: {item_id}")

        # Recalculate overall metrics from all item results
        total_correct = sum(r["metrics"]["correct"] for r in merged_item_results)
        total_questions = sum(r["metrics"]["total"] for r in merged_item_results)
        total_invalid = sum(r["metrics"].get("invalid", 0) for r in merged_item_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        valid_only_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "total_invalid": total_invalid,
            "total_valid": total_valid,
            "valid_only_accuracy": valid_only_accuracy,
            "num_items": len(merged_item_results),
            "item_results": merged_item_results,
        }

    def _save_incremental_results(self, all_results: List[Dict], output_path: Path):
        """
        Save results incrementally to JSON file.

        Args:
            all_results: Current list of all item results
            output_path: Path to save results to
        """
        ordered_results = sorted(all_results, key=lambda result: str(result.get("item_id", "")))
        total_correct = sum(r["metrics"]["correct"] for r in ordered_results)
        total_questions = sum(r["metrics"]["total"] for r in ordered_results)
        total_invalid = sum(r["metrics"].get("invalid", 0) for r in ordered_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        valid_only_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        results_dict = {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "total_invalid": total_invalid,
            "total_valid": total_valid,
            "valid_only_accuracy": valid_only_accuracy,
            "num_items": len(ordered_results),
            "model_config": getattr(self, "_model_config", None) or get_model_config(),
            "item_results": ordered_results,
        }
        run_manifest = getattr(self, "_run_manifest", None)
        if run_manifest is not None:
            results_dict["run_manifest"] = run_manifest

        _write_json_atomic(results_dict, output_path)

    def save_results(self, results: Dict[str, Any], output_path: Path, merge_with_existing: bool = False):
        """
        Save results to JSON file.

        Args:
            results: Results to save
            output_path: Path to save results to
            merge_with_existing: If True, merge with existing results file if it exists
        """
        if merge_with_existing and output_path.exists():
            # Load existing results
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

            console.print(f"\n[cyan]Merging with existing results from {output_path}...[/cyan]")
            results = self.merge_results(results, existing_results)

        if "item_results" in results:
            results["item_results"] = sorted(
                results["item_results"],
                key=lambda result: str(result.get("item_id", "")),
            )
        _write_json_atomic(results, output_path)
        console.print(f"\n[green]✓[/green] Results saved to {output_path}")
