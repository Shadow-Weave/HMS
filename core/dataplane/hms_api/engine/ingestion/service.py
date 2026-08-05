"""Retain application service for database-backed ingestion.

All documents are planned from one read-only snapshot before any bank,
checkpoint, document, chunk, fact, link, or outbox write begins.  Each document
then executes hash-guarded writes selected from the pure change plan:
fresh/existing full replacement (possibly split into bounded windows), delta,
or metadata-only.  Stale plans perform no write at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, TypeVar

from ...config import DEFAULT_RETAIN_SEMANTIC_CHUNKING_ENABLED
from ..db_utils import acquire_with_retry
from ..embedding_fingerprint import EmbeddingFingerprintError, ensure_bank_embedding_fingerprint
from ..response_models import TokenUsage
from ..retain import bank_utils
from .adapters.embedding_model import EmbeddingModelAdapter
from .adapters.postgres_fresh_ownership import FreshDocumentOwnershipConflict
from .adapters.storage_records import (
    chunks_to_storage,
    compute_document_hash,
    content_positions,
    content_to_storage,
    record_to_extracted_fact,
    record_to_processed_fact,
    retain_document_metadata,
)
from .change_detection import detect_document_change
from .chunking import build_chunk_plans, compute_content_hash
from .contracts import RetainExecutionContext, RetainInvocation, RetainOutcome
from .document_planner import plan_documents, prepend_existing_document
from .domain import (
    ChunkPlan,
    ChunkPolicy,
    DocumentChangeKind,
    DocumentChangePlan,
    DocumentIntent,
    ExistingChunkFingerprint,
    UpdateMode,
)
from .execution import merge_window_unit_ids, plan_full_write_windows
from .extraction import (
    ExtractionMode,
    ExtractionPolicy,
    FactExtractorAdapter,
    build_prechunked_extraction_layout,
    extract_passthrough,
)
from .normalization import normalize_contents
from .persistence.backend import RetainBackendAdapters, retain_backend_adapters
from .persistence.models import CommittedUnitBinding, ExistingDocument, OperationCheckpoint
from .persistence.unit_of_work import (
    AtomicRetainUnitOfWork,
    AtomicWriteOwnershipLost,
    AtomicWriteStep,
    ChunkWrite,
    CoreGraphWrite,
    CoreWriteResult,
    DeltaWriteRequest,
    ExistingChunkWrite,
    FactWrite,
    FirstFullWriteWindow,
    LaterFullWriteWindow,
    MetadataOnlyWriteRequest,
    OwnershipDisposition,
    PostCommitStatus,
    RetainUnitOfWork,
    RetainWriteRequest,
    WriteWindowRequest,
)
from .persistence.writer import PersistenceWriter
from .projection import EmbeddingFailurePolicy, MemoryRecord, project_embeddings
from .redaction import IdentifierSanitizer
from .runtime import (
    count_tokens,
    embedding_model_version,
    pre_resolve_entities,
    run_final_semantic_ann,
)
from .segmentation import (
    EffectiveSegmentationStrategy,
    SegmentationFailurePolicy,
    SegmentationManifest,
    SegmentationReuseError,
    SemanticSegmentationPolicy,
    SemanticSegmenter,
    build_chunk_plans_from_segmentation,
    parse_conversation,
)

logger = logging.getLogger(__name__)

_INFLIGHT_CONTENT_HASH_PREFIX = "retain-inflight:"
_DEFAULT_PROJECTION_PIPELINE_CONCURRENCY = 4
_SEMANTIC_PLAN_METADATA_KEY = "_hms_ingestion"
_SEMANTIC_DOCUMENT_PLAN_SCHEMA = "retain-semantic-document-plan-v1"
_PlanningResult = TypeVar("_PlanningResult")


class RetainError(RuntimeError):
    """Base class for Retain application-boundary failures."""


class RetainUnsupportedError(RetainError):
    """The Retain pipeline does not safely implement this request."""


class RetainExtractionModeUnsupportedError(RetainUnsupportedError):
    """The configured extraction strategy is not a supported Retain mode."""


class RetainDatabaseUnsupportedError(RetainUnsupportedError):
    """The configured database backend is not implemented by Retain."""


class RetainResultMappingError(RetainError):
    """A persistence result cannot be mapped one-to-one to request inputs."""


class RetainPublicationAborted(RetainError):
    """The request no longer owns a document state that it can publish.

    The message must remain identifier-free because asynchronous workers may
    persist it as part of a terminal operation result.
    """


class RetainOwnershipLostError(RetainPublicationAborted):
    """The document changed after planning and this attempt must be retried."""


class RetainCheckpointRecoveryError(RetainError):
    """A durable operation checkpoint cannot be reconciled with Retain rows."""


def _request_sanitizer(
    invocation: RetainInvocation,
    *identifiers: Any,
) -> IdentifierSanitizer:
    return IdentifierSanitizer.from_values(
        enabled=invocation.sanitize_log_identifiers,
        values=(
            invocation.bank_id,
            invocation.operation_id,
            *identifiers,
        ),
    )


def _log_identifier(invocation: RetainInvocation, value: Any) -> str:
    return _request_sanitizer(invocation, value).identifier(value)


def _log_warning(
    invocation: RetainInvocation,
    message: str,
    *args: Any,
    identifiers: tuple[Any, ...] = (),
    exc_info: bool = False,
) -> None:
    """Log a warning without exposing trusted request identifiers."""

    sanitizer = _request_sanitizer(invocation, *identifiers)
    if not sanitizer.enabled:
        logger.warning(message, *args, exc_info=exc_info)
        return

    safe_args = tuple(sanitizer.text(value) for value in args)
    if exc_info:
        exception = sys.exc_info()[1]
        if exception is not None:
            message = f"{message}: %s"
            safe_args = (*safe_args, sanitizer.text(exception))
    logger.warning(message, *safe_args)


@dataclass(frozen=True, slots=True)
class _DocumentExecutionPlan:
    intent: DocumentIntent
    chunks: tuple[ChunkPlan, ...]
    combined_content: str
    existing: ExistingDocument | None
    existing_chunks: tuple[ExistingChunkFingerprint, ...]
    change: DocumentChangePlan
    recovered_unit_bindings: tuple[CommittedUnitBinding, ...] | None = None
    recovered_chunk_sources: tuple[tuple[int, int | None], ...] | None = None
    final_ann_pending: bool = False
    segmentation_metadata: dict[str, Any] | None = None
    segmentation_usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def recovered_unit_ids(self) -> tuple[str, ...] | None:
        if self.recovered_unit_bindings is None:
            return None
        return tuple(binding.unit_id for binding in self.recovered_unit_bindings)


@dataclass(frozen=True, slots=True)
class _FactPayload:
    storage_contents: tuple[Any, ...]
    chunks: tuple[ChunkWrite, ...]
    facts: tuple[FactWrite, ...]
    graph: CoreGraphWrite


@dataclass(frozen=True, slots=True)
class _DocumentOutcome:
    unit_ids_by_content: tuple[tuple[str, ...], ...]
    usage: TokenUsage
    processed_tokens: int | None


@dataclass(frozen=True, slots=True)
class _ProjectedChunkOutcome:
    records: tuple[MemoryRecord, ...]
    usage: TokenUsage
    extraction_seconds: float
    embedding_seconds: float


@dataclass(frozen=True, slots=True)
class _AtomicFullPublication:
    unit_ids_by_content: tuple[tuple[str, ...], ...]
    committed_unit_ids: tuple[str, ...]


def _validate_atomic_full_publication(
    document_source_indices: Sequence[int | None],
    prepared_records: Sequence[Sequence[MemoryRecord]],
    cores: Sequence[CoreWriteResult],
) -> _AtomicFullPublication:
    """Validate every persisted FULL result before checkpointing or commit."""

    record_windows = tuple(tuple(records) for records in prepared_records)
    core_results = tuple(cores)
    if len(core_results) != len(record_windows):
        raise RetainResultMappingError(
            "Atomic FULL publication returned a different number of write results than prepared windows"
        )

    window_results: list[tuple[Sequence[MemoryRecord], Sequence[tuple[str, str]]]] = []
    seen_bucket_unit_ids: set[str] = set()
    for window_index, (records, core) in enumerate(zip(record_windows, core_results, strict=True)):
        bindings = tuple(core.unit_ids_by_fact_key)
        bucket_unit_ids = tuple(unit_id for bucket in core.unit_ids_by_content for unit_id in bucket)
        binding_unit_ids = tuple(unit_id for _fact_key, unit_id in bindings)
        if any(not isinstance(unit_id, str) or not unit_id for unit_id in binding_unit_ids):
            raise RetainResultMappingError(
                f"Atomic FULL window {window_index} returned an invalid fact-key binding unit ID"
            )
        if any(not isinstance(unit_id, str) or not unit_id for unit_id in bucket_unit_ids):
            raise RetainResultMappingError(
                f"Atomic FULL window {window_index} returned an invalid content-bucket unit ID"
            )
        if len(bucket_unit_ids) != len(set(bucket_unit_ids)):
            raise RetainResultMappingError(
                f"Atomic FULL window {window_index} returned duplicate content-bucket unit IDs"
            )
        duplicate_across_windows = seen_bucket_unit_ids.intersection(bucket_unit_ids)
        if duplicate_across_windows:
            raise RetainResultMappingError("Atomic FULL publication returned duplicate unit IDs across windows")
        seen_bucket_unit_ids.update(bucket_unit_ids)
        if len(bucket_unit_ids) != len(binding_unit_ids) or set(bucket_unit_ids) != set(binding_unit_ids):
            raise RetainResultMappingError(
                f"Atomic FULL window {window_index} returned inconsistent content and fact-key unit IDs"
            )
        window_results.append((records, bindings))

    sources = tuple(document_source_indices)
    try:
        public_buckets = merge_window_unit_ids(sources, tuple(window_results))
    except (TypeError, ValueError) as exc:
        raise RetainResultMappingError("Atomic FULL publication returned an invalid fact-key mapping") from exc

    committed_unit_ids: list[str] = []
    for records, bindings in window_results:
        units_by_key = dict(bindings)
        committed_unit_ids.extend(units_by_key[record.fact_key] for record in records)

    public_iterator = iter(public_buckets)
    unit_ids_by_content = tuple(() if source_index is None else next(public_iterator) for source_index in sources)
    return _AtomicFullPublication(
        unit_ids_by_content=unit_ids_by_content,
        committed_unit_ids=tuple(committed_unit_ids),
    )


@dataclass(frozen=True, slots=True)
class _DocumentPreflightSnapshot:
    submitted_intent: DocumentIntent
    existing: ExistingDocument | None
    existing_chunks: tuple[ExistingChunkFingerprint, ...]
    recovered_unit_bindings: tuple[CommittedUnitBinding, ...] | None = None
    expected_unit_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _SemanticDocumentPlan:
    chunks: tuple[ChunkPlan, ...]
    metadata: dict[str, Any]
    usage: TokenUsage
    layout_signature: tuple[Any, ...]


def _projection_pipeline_concurrency(config: Any) -> int:
    """Resolve a finite positive producer width without trusting bool-as-int."""

    configured = getattr(config, "retain_llm_max_concurrent", None)
    if configured is None:
        configured = getattr(config, "llm_max_concurrent", None)
    if configured is None:
        return _DEFAULT_PROJECTION_PIPELINE_CONCURRENCY
    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        logger.warning(
            "Invalid Retain projection pipeline concurrency %r; using fallback=%d",
            configured,
            _DEFAULT_PROJECTION_PIPELINE_CONCURRENCY,
        )
        return _DEFAULT_PROJECTION_PIPELINE_CONCURRENCY
    return configured


def _log_window_producer_summary(
    *,
    path: str,
    chunks: int,
    facts: int,
    wall_seconds: float,
    extraction_seconds: Sequence[float],
    embedding_seconds: Sequence[float],
    configured_concurrency: int,
) -> None:
    """Emit payload-free timing telemetry for one extraction window."""

    logger.info(
        "Retain window producer: path=%s chunks=%d facts=%d wall_seconds=%.3f "
        "extraction_sum_seconds=%.3f extraction_max_seconds=%.3f "
        "embedding_sum_seconds=%.3f embedding_max_seconds=%.3f configured_concurrency=%d",
        path,
        chunks,
        facts,
        wall_seconds,
        sum(extraction_seconds),
        max(extraction_seconds, default=0.0),
        sum(embedding_seconds),
        max(embedding_seconds, default=0.0),
        configured_concurrency,
    )


def _backend_type(execution: RetainExecutionContext) -> str:
    backend_type = getattr(execution.pool, "backend_type", None)
    if backend_type is None:
        backend_type = getattr(execution.resolved_config, "database_backend", "postgresql")
    if not isinstance(backend_type, str):
        raise RetainDatabaseUnsupportedError("Retain could not determine the database backend")
    return backend_type.lower()


def _require_supported_route(execution: RetainExecutionContext) -> None:
    backend_type = _backend_type(execution)
    if backend_type not in {"postgresql", "oracle"}:
        raise RetainDatabaseUnsupportedError(
            f"Retain supports PostgreSQL and Oracle; configured backend is {backend_type!r}"
        )
    mode = getattr(execution.resolved_config, "retain_extraction_mode", None)
    try:
        ExtractionMode(mode)
    except (TypeError, ValueError) as exc:
        raise RetainExtractionModeUnsupportedError(f"Retain does not support retain_extraction_mode={mode!r}.") from exc


def _semantic_planning_active(
    invocation: RetainInvocation,
    execution: RetainExecutionContext,
) -> bool:
    """Return whether this invocation may call the semantic boundary model."""

    if (
        getattr(
            execution.resolved_config,
            "retain_semantic_chunking_enabled",
            DEFAULT_RETAIN_SEMANTIC_CHUNKING_ENABLED,
        )
        is not True
    ):
        return False
    if invocation.trusted_prechunked_input:
        return False
    if getattr(execution.resolved_config, "retain_extraction_mode", None) == ExtractionMode.CHUNKS.value:
        return False
    return getattr(execution.llm_config, "provider", None) != "none"


def _semantic_policy(
    execution: RetainExecutionContext,
    chunk_policy: ChunkPolicy,
) -> SemanticSegmentationPolicy:
    config = execution.resolved_config
    return SemanticSegmentationPolicy(
        max_chars=chunk_policy.max_chars,
        provider=str(getattr(execution.llm_config, "provider", "unknown")),
        model=str(getattr(execution.llm_config, "model", "unknown")),
        failure_policy=SegmentationFailurePolicy(
            getattr(
                config,
                "retain_semantic_chunking_failure_policy",
                SegmentationFailurePolicy.FIXED_FALLBACK.value,
            )
        ),
        max_completion_tokens=getattr(
            config,
            "retain_semantic_chunking_max_completion_tokens",
            1024,
        ),
        max_retries=getattr(
            config,
            "retain_semantic_chunking_max_retries",
            1,
        ),
    )


def _semantic_manifest_layout(manifest: SegmentationManifest) -> tuple[Any, ...]:
    """Return only boundary fields whose drift requires a conservative FULL."""

    return (
        manifest.effective_strategy.value,
        manifest.end_exchange_indices,
        tuple(
            (
                chunk.semantic_segment_index,
                chunk.start_exchange,
                chunk.end_exchange,
                chunk.oversized_atomic,
            )
            for chunk in manifest.chunks
        ),
    )


def _semantic_document_metadata(
    *,
    policy: SemanticSegmentationPolicy,
    items: Sequence[Any],
    manifests: Sequence[SegmentationManifest],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if len(items) != len(manifests):
        raise ValueError("semantic plan items and manifests must have the same length")
    item_payloads = [
        {
            "position": position,
            "source_index": item.source_index,
            "manifest": manifest.as_dict(),
        }
        for position, (item, manifest) in enumerate(zip(items, manifests, strict=True))
    ]
    digest_payload = {
        "schema_version": _SEMANTIC_DOCUMENT_PLAN_SCHEMA,
        "policy_fingerprint": policy.fingerprint,
        "items": item_payloads,
    }
    encoded = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = {
        **digest_payload,
        "plan_digest": hashlib.sha256(encoded).hexdigest(),
    }
    layout = tuple(_semantic_manifest_layout(manifest) for manifest in manifests)
    return metadata, layout


def _existing_semantic_metadata(existing: ExistingDocument | None) -> dict[str, Any] | None:
    if existing is None or not isinstance(existing.retain_params, dict):
        return None
    value = existing.retain_params.get(_SEMANTIC_PLAN_METADATA_KEY)
    return value if isinstance(value, dict) else None


def _stored_manifest_payloads(
    metadata: dict[str, Any] | None,
    *,
    policy_fingerprint: str,
) -> tuple[dict[str, Any], ...] | None:
    """Validate the document envelope and return ordered text-free manifests."""

    if not isinstance(metadata, dict):
        return None
    if metadata.get("schema_version") != _SEMANTIC_DOCUMENT_PLAN_SCHEMA:
        return None
    if metadata.get("policy_fingerprint") != policy_fingerprint:
        return None
    items = metadata.get("items")
    if not isinstance(items, list):
        return None
    manifests: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict) or item.get("position") != position:
            return None
        manifest = item.get("manifest")
        if not isinstance(manifest, dict):
            return None
        manifests.append(manifest)
    digest_payload = {
        "schema_version": metadata.get("schema_version"),
        "policy_fingerprint": metadata.get("policy_fingerprint"),
        "items": items,
    }
    encoded = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if metadata.get("plan_digest") != hashlib.sha256(encoded).hexdigest():
        return None
    return tuple(manifests)


def _stored_semantic_layout(
    metadata: dict[str, Any] | None,
    *,
    policy_fingerprint: str,
) -> tuple[Any, ...] | None:
    payloads = _stored_manifest_payloads(
        metadata,
        policy_fingerprint=policy_fingerprint,
    )
    if payloads is None:
        return None
    try:
        manifests = tuple(SegmentationManifest.from_dict(payload) for payload in payloads)
    except (TypeError, ValueError):
        return None
    return tuple(_semantic_manifest_layout(manifest) for manifest in manifests)


def _stored_recovery_manifest_items(
    metadata: dict[str, Any],
    *,
    document_id: str,
) -> tuple[tuple[int | None, SegmentationManifest], ...]:
    """Load a durable semantic plan without applying the current policy.

    A committed recovery must reproduce the layout that wrote the durable
    chunks. Policy drift is therefore expected here: the stored envelope and
    each manifest are validated against their own persisted fingerprint.
    """

    policy_fingerprint = metadata.get("policy_fingerprint")
    if not isinstance(policy_fingerprint, str):
        raise RetainCheckpointRecoveryError(
            f"Committed document {document_id!r} has an invalid semantic plan policy fingerprint"
        )
    payloads = _stored_manifest_payloads(
        metadata,
        policy_fingerprint=policy_fingerprint,
    )
    raw_items = metadata.get("items")
    if payloads is None or not isinstance(raw_items, list):
        raise RetainCheckpointRecoveryError(f"Committed document {document_id!r} has an invalid semantic plan envelope")

    items: list[tuple[int | None, SegmentationManifest]] = []
    for position, (raw_item, payload) in enumerate(zip(raw_items, payloads, strict=True)):
        if not isinstance(raw_item, dict) or set(raw_item) != {"position", "source_index", "manifest"}:
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} has invalid semantic plan item {position}"
            )
        source_index = raw_item.get("source_index")
        if source_index is not None and (
            isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0
        ):
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} has invalid semantic source index at item {position}"
            )
        try:
            manifest = SegmentationManifest.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} has an invalid semantic plan manifest"
            ) from exc
        if manifest.policy_fingerprint != policy_fingerprint:
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} has inconsistent semantic plan fingerprints"
            )
        items.append((source_index, manifest))
    return tuple(items)


def _manifest_input_hash(
    text: str,
    manifest: SegmentationManifest,
    *,
    document_id: str,
) -> str:
    """Recompute the source identity encoded by one durable manifest."""

    if manifest.effective_strategy in {
        EffectiveSegmentationStrategy.PASSTHROUGH,
        EffectiveSegmentationStrategy.FIXED_BYPASS,
    }:
        return compute_content_hash(text)

    conversation = parse_conversation(text)
    if conversation is None:
        if manifest.effective_strategy is EffectiveSegmentationStrategy.SEMANTIC:
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} has a semantic plan for non-conversation input"
            )
        return compute_content_hash(text)
    if manifest.effective_strategy is EffectiveSegmentationStrategy.SEMANTIC and not conversation.exchanges:
        raise RetainCheckpointRecoveryError(
            f"Committed document {document_id!r} has a semantic plan for an empty conversation"
        )
    return conversation.input_hash


def _validated_recovery_chunk_sources(
    durable_chunks: Sequence[ExistingChunkFingerprint],
    expected_chunks: Sequence[tuple[str, int | None]],
    *,
    document_id: str,
) -> tuple[tuple[int, int | None], ...]:
    """Verify a complete durable layout before exposing source ownership."""

    indices = tuple(chunk.chunk_index for chunk in durable_chunks)
    if indices != tuple(range(len(durable_chunks))):
        raise RetainCheckpointRecoveryError(
            f"Committed document {document_id!r} has non-contiguous chunk indices {indices!r}"
        )
    if len(durable_chunks) != len(expected_chunks):
        raise RetainCheckpointRecoveryError(
            f"Committed document {document_id!r} chunk count does not match its recovery layout"
        )

    sources: list[tuple[int, int | None]] = []
    for durable, (expected_hash, source_index) in zip(durable_chunks, expected_chunks, strict=True):
        if durable.content_hash != expected_hash:
            raise RetainCheckpointRecoveryError(
                f"Committed document {document_id!r} does not match its recovery layout "
                f"at durable chunk index {durable.chunk_index}"
            )
        sources.append((durable.chunk_index, source_index))
    return tuple(sources)


def _semantic_recovery_chunk_sources(
    metadata: dict[str, Any],
    intent: DocumentIntent,
    durable_chunks: Sequence[ExistingChunkFingerprint],
) -> tuple[tuple[int, int | None], ...]:
    """Map durable semantic chunks to the retry payload without an LLM call."""

    stored_items = _stored_recovery_manifest_items(
        metadata,
        document_id=intent.document_id,
    )
    if intent.update_mode is UpdateMode.APPEND:
        if len(stored_items) < len(intent.items):
            raise RetainCheckpointRecoveryError(
                f"Committed append document {intent.document_id!r} has fewer semantic items than the retry payload"
            )
        prefix_count = len(stored_items) - len(intent.items)
        if any(source_index is not None for source_index, _manifest in stored_items[:prefix_count]):
            raise RetainCheckpointRecoveryError(
                f"Committed append document {intent.document_id!r} has an ambiguous semantic prefix"
            )
    else:
        if len(stored_items) != len(intent.items):
            raise RetainCheckpointRecoveryError(
                f"Committed document {intent.document_id!r} semantic item count does not match the retry payload"
            )
        prefix_count = 0

    for position, (item, (stored_source_index, manifest)) in enumerate(
        zip(intent.items, stored_items[prefix_count:], strict=True)
    ):
        if stored_source_index != item.source_index:
            raise RetainCheckpointRecoveryError(
                f"Committed document {intent.document_id!r} semantic source mapping changed at item {position}"
            )
        if (
            _manifest_input_hash(
                item.content,
                manifest,
                document_id=intent.document_id,
            )
            != manifest.input_hash
        ):
            raise RetainCheckpointRecoveryError(
                f"Committed document {intent.document_id!r} retry input does not match its semantic plan "
                f"at item {position}"
            )

    expected_chunks: list[tuple[str, int | None]] = []
    for position, (stored_source_index, manifest) in enumerate(stored_items):
        source_index = None if position < prefix_count else stored_source_index
        expected_chunks.extend((chunk.content_hash, source_index) for chunk in manifest.chunks)
    return _validated_recovery_chunk_sources(
        durable_chunks,
        expected_chunks,
        document_id=intent.document_id,
    )


def _committed_recovery_chunk_sources(
    existing: ExistingDocument,
    intent: DocumentIntent,
    durable_chunks: Sequence[ExistingChunkFingerprint],
    chunk_policy: ChunkPolicy,
) -> tuple[tuple[int, int | None], ...]:
    """Recover one committed source map without semantic replanning."""

    retain_params = existing.retain_params
    if isinstance(retain_params, dict) and _SEMANTIC_PLAN_METADATA_KEY in retain_params:
        metadata = retain_params[_SEMANTIC_PLAN_METADATA_KEY]
        if not isinstance(metadata, dict):
            raise RetainCheckpointRecoveryError(
                f"Committed document {intent.document_id!r} has invalid semantic plan metadata"
            )
        return _semantic_recovery_chunk_sources(
            metadata,
            intent,
            durable_chunks,
        )

    try:
        submitted_chunks = build_chunk_plans(
            intent.document_id,
            intent.items,
            chunk_policy,
        )
    except Exception as exc:
        raise RetainCheckpointRecoveryError(
            f"Committed document {intent.document_id!r} fixed recovery planning failed"
        ) from exc
    if intent.update_mode is UpdateMode.APPEND:
        return _append_recovery_chunk_sources(
            durable_chunks,
            submitted_chunks,
            document_id=intent.document_id,
        )
    return _validated_recovery_chunk_sources(
        durable_chunks,
        tuple((chunk.content_hash, chunk.source_index) for chunk in submitted_chunks),
        document_id=intent.document_id,
    )


def _retain_metadata_with_semantic_plan(
    plan: _DocumentExecutionPlan,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    retain_params, document_tags = retain_document_metadata(plan.intent.items)
    if plan.segmentation_metadata is not None:
        retain_params[_SEMANTIC_PLAN_METADATA_KEY] = plan.segmentation_metadata
    return retain_params, document_tags


def _recovered_id_factory(recovered_ids: Sequence[str], explicit_ids: set[str]) -> Iterator[str]:
    for document_id in recovered_ids:
        if document_id not in explicit_ids:
            yield document_id
    while True:
        yield str(uuid.uuid4())


def _append_recovery_chunk_sources(
    durable_chunks: Sequence[ExistingChunkFingerprint],
    submitted_chunks: Sequence[ChunkPlan],
    *,
    document_id: str,
) -> tuple[tuple[int, int | None], ...]:
    """Map a completed append's durable chunk suffix to submitted sources.

    The document row now contains the *post-append* combined text, so
    re-splitting it as one synthetic item cannot reliably reproduce the chunk
    boundaries used when the operation committed.  Append planning always
    places submitted chunks after the synthetic pre-append chunks.  Recover
    that stable suffix from the durable chunk table and verify every hash
    before exposing any operation-local unit ID.
    """

    indices = tuple(chunk.chunk_index for chunk in durable_chunks)
    if indices != tuple(range(len(durable_chunks))):
        raise RetainCheckpointRecoveryError(
            f"Committed append document {document_id!r} has non-contiguous chunk indices {indices!r}"
        )
    if len(submitted_chunks) > len(durable_chunks):
        raise RetainCheckpointRecoveryError(
            f"Committed append document {document_id!r} has fewer durable chunks than the retry payload"
        )

    submitted_count = len(submitted_chunks)
    suffix_start = len(durable_chunks) - submitted_count
    if submitted_count:
        durable_suffix = durable_chunks[suffix_start:]
        for durable, submitted in zip(durable_suffix, submitted_chunks, strict=True):
            if durable.content_hash != submitted.content_hash:
                raise RetainCheckpointRecoveryError(
                    f"Committed append document {document_id!r} does not match the retry payload "
                    f"at durable chunk index {durable.chunk_index}"
                )

    sources: list[tuple[int, int | None]] = []
    for position, durable in enumerate(durable_chunks):
        source_index = None
        if position >= suffix_start:
            source_index = submitted_chunks[position - suffix_start].source_index
        sources.append((durable.chunk_index, source_index))
    return tuple(sources)


def _graph_write(phase1: Any) -> CoreGraphWrite:
    if phase1 is None:
        return CoreGraphWrite()
    entity_read_plan = getattr(phase1, "entity_read_plan", None)
    if entity_read_plan is not None:
        return CoreGraphWrite(
            semantic_ann_links=tuple(tuple(link) for link in phase1.semantic_ann_links),
            entity_read_plan=entity_read_plan,
        )
    return CoreGraphWrite(
        resolved_entity_ids=tuple(phase1.entities.resolved_entity_ids),
        entity_to_unit=tuple(tuple(binding) for binding in phase1.entities.entity_to_unit),
        unit_to_entity_ids=tuple(
            (unit_id, tuple(entity_ids)) for unit_id, entity_ids in phase1.entities.unit_to_entity_ids.items()
        ),
        semantic_ann_links=tuple(tuple(link) for link in phase1.semantic_ann_links),
    )


def _merge_processed_tokens(current: int | None, document: int | None) -> int | None:
    """Preserve the public token-accounting contract across documents."""

    if current is None or document is None:
        return None
    return current + document


@asynccontextmanager
async def _database_budget(semaphore: Any):
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield


async def _gather_planning_tasks(
    coroutines: Sequence[Coroutine[Any, Any, _PlanningResult]],
) -> tuple[_PlanningResult, ...]:
    """Cancel and await sibling planning work after the first task failure."""

    tasks = tuple(asyncio.create_task(coroutine) for coroutine in coroutines)
    if not tasks:
        return ()

    task_positions = {task: position for position, task in enumerate(tasks)}
    pending = set(tasks)
    try:
        while pending:
            completed, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            failures: list[tuple[int, BaseException]] = []
            for task in completed:
                if task.cancelled():
                    failures.append(
                        (
                            task_positions[task],
                            asyncio.CancelledError("semantic planning task was cancelled"),
                        )
                    )
                    continue
                error = task.exception()
                if error is not None:
                    failures.append((task_positions[task], error))
            if not failures:
                continue

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            failures.sort(key=lambda item: item[0])
            raise failures[0][1]

        return tuple(task.result() for task in tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _backend_adapters(execution: RetainExecutionContext) -> RetainBackendAdapters:
    """Select the persistence contracts for the request's configured backend."""

    try:
        return retain_backend_adapters(_backend_type(execution))
    except ValueError as exc:  # Defensive parity with the route guard.
        raise RetainDatabaseUnsupportedError(str(exc)) from exc


class RetainPipelineService:
    """Hash-guarded chunks-mode Retain pipeline."""

    async def retain(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
    ) -> RetainOutcome:
        """Run the pipeline with one schema shared by every Retain stage.

        Planning receives the schema explicitly, while low-level storage
        helpers resolve table names through ``memory_engine``'s task-local
        schema context. Scoping both to the same value prevents a request from
        reading one tenant and writing another.
        """

        from ..memory_engine import _current_schema, get_current_schema

        effective_schema = execution.schema or get_current_schema()
        scoped_execution = replace(execution, schema=effective_schema)
        schema_token = _current_schema.set(effective_schema)
        try:
            return await self._retain_in_schema(invocation, scoped_execution)
        finally:
            _current_schema.reset(schema_token)

    async def _retain_in_schema(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
    ) -> RetainOutcome:
        _require_supported_route(execution)
        request_started_at = datetime.now(UTC)
        normalized = normalize_contents(invocation.raw_contents, document_tags=invocation.document_tags)
        if not normalized:
            return RetainOutcome([], TokenUsage(), None)

        checkpoint = await self._recover_checkpoint(invocation, execution)
        # Core checkpoints atomically carry document IDs. The secondary field
        # recovers checkpoints whose early document-ID write did not complete.
        recovered_ids = checkpoint.document_ids or checkpoint.core_committed_document_ids
        explicit_ids = {item.document_id for item in normalized if item.document_id is not None}
        generated_ids = _recovered_id_factory(recovered_ids, explicit_ids)
        intents = plan_documents(
            normalized,
            batch_document_id=invocation.batch_document_id,
            recovered_document_id=recovered_ids[0] if not explicit_ids and recovered_ids else None,
            id_factory=lambda: next(generated_ids),
        )
        policy = ChunkPolicy(
            version="retain-chunker-v1",
            max_chars=getattr(execution.resolved_config, "retain_chunk_size", 3000),
            conversation_mode=True,
            overlap=0,
        )
        if _semantic_planning_active(invocation, execution) and getattr(
            execution.resolved_config,
            "retain_batch_enabled",
            False,
        ):
            all_documents_core_committed = all(checkpoint.is_core_committed(intent.document_id) for intent in intents)
            if not all_documents_core_committed:
                raise RetainUnsupportedError(
                    "Semantic Retain chunking does not support provider Batch extraction because "
                    "the existing Batch checkpoint does not bind results to a semantic plan digest."
                )

        # This is the all-document read barrier. No bank auto-create,
        # checkpoint update, semantic write, or outbox call occurs before it.
        plans = await self._preflight_documents(
            invocation,
            execution,
            intents,
            policy,
            checkpoint=checkpoint,
            request_started_at=request_started_at,
        )

        unpublished_stale_plans = tuple(
            plan
            for plan in plans
            if plan.change.kind is DocumentChangeKind.STALE_SKIP and plan.recovered_unit_ids is None
        )
        if unpublished_stale_plans and invocation.outbox_callback is not None:
            raise RetainPublicationAborted("Retain was superseded before publication")

        bank_profile = await bank_utils.get_bank_profile(execution.pool, invocation.bank_id)
        agent_name = bank_profile["name"]
        try:
            async with acquire_with_retry(execution.pool) as fingerprint_connection:
                await ensure_bank_embedding_fingerprint(
                    fingerprint_connection,
                    invocation.bank_id,
                    execution.embeddings_model,
                    policy=getattr(
                        execution.resolved_config,
                        "embedding_fingerprint_policy",
                        "strict",
                    ),
                    for_write=False,
                    legacy_attestation=getattr(
                        execution.resolved_config,
                        "embedding_fingerprint_legacy_attestation",
                        None,
                    ),
                    log_sanitizer=_request_sanitizer(invocation),
                )
        except EmbeddingFingerprintError as exc:
            sanitizer = _request_sanitizer(invocation)
            if sanitizer.enabled:
                exc.args = (sanitizer.text(exc),)
            raise
        await self._record_document_ids(invocation, execution, intents)

        unit_ids_by_input: list[list[str]] = [[] for _ in invocation.raw_contents]
        total_usage = TokenUsage()
        total_processed_tokens: int | None = 0
        commit_positions = [
            position
            for position, plan in enumerate(plans)
            if plan.change.kind is not DocumentChangeKind.STALE_SKIP and plan.recovered_unit_ids is None
        ]
        last_commit_position = commit_positions[-1] if commit_positions else None

        for position, plan in enumerate(plans):
            total_usage = total_usage + getattr(plan, "segmentation_usage", TokenUsage())
            if plan.recovered_unit_ids is not None:
                document_outcome = await self._resume_committed_document(
                    invocation,
                    execution,
                    plan,
                )
            elif plan.change.kind is DocumentChangeKind.STALE_SKIP:
                document_outcome = _DocumentOutcome(
                    unit_ids_by_content=tuple(() for _ in plan.intent.items),
                    usage=TokenUsage(),
                    processed_tokens=0,
                )
            else:
                document_outcome = await self._execute_document(
                    invocation,
                    execution,
                    plan,
                    agent_name=agent_name,
                    outbox_callback=(invocation.outbox_callback if position == last_commit_position else None),
                )
            total_usage = total_usage + document_outcome.usage
            total_processed_tokens = _merge_processed_tokens(
                total_processed_tokens,
                document_outcome.processed_tokens,
            )
            self._merge_document_result(
                unit_ids_by_input,
                plan.intent,
                document_outcome.unit_ids_by_content,
            )

        return RetainOutcome(unit_ids_by_input, total_usage, total_processed_tokens)

    async def _recover_checkpoint(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
    ) -> OperationCheckpoint:
        if invocation.operation_id is None:
            return OperationCheckpoint()
        try:
            async with acquire_with_retry(execution.pool) as connection:
                return (
                    await _backend_adapters(execution)
                    .checkpoint_store(
                        connection,
                        schema=execution.schema,
                    )
                    .recover(invocation.operation_id)
                )
        except Exception as exc:
            # Once an operation ID exists, an unreadable checkpoint is an
            # unknown commit state.  Treating it as empty can allocate a new
            # document ID and duplicate an already-committed retry.
            operation_id = _log_identifier(invocation, invocation.operation_id)
            raise RetainCheckpointRecoveryError(
                f"Retain could not safely recover operation checkpoint {operation_id}"
            ) from exc

    async def _record_document_ids(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        intents: Sequence[DocumentIntent],
    ) -> None:
        if invocation.operation_id is None:
            return
        try:
            async with acquire_with_retry(execution.pool) as connection:
                store = _backend_adapters(execution).checkpoint_store(
                    connection,
                    schema=execution.schema,
                )
                for intent in intents:
                    await store.record_document_id(invocation.operation_id, intent.document_id)
        except Exception:
            # This early checkpoint is best effort. The core-commit checkpoint
            # below is strict because it shares the memory write transaction.
            _log_warning(
                invocation,
                "Retain could not record document IDs for operation %s",
                _log_identifier(invocation, invocation.operation_id),
                identifiers=tuple(intent.document_id for intent in intents),
                exc_info=True,
            )

    @staticmethod
    async def _build_semantic_document_plan(
        execution: RetainExecutionContext,
        intent: DocumentIntent,
        chunk_policy: ChunkPolicy,
        *,
        stored_metadata: dict[str, Any] | None,
        planning_semaphore: asyncio.Semaphore,
        reuse_trailing_items: bool = False,
    ) -> _SemanticDocumentPlan:
        """Plan and materialize one document after the database snapshot closes."""

        semantic_policy = _semantic_policy(execution, chunk_policy)
        segmenter = SemanticSegmenter(
            llm_config=execution.llm_config,
            policy=semantic_policy,
        )
        stored_manifests = _stored_manifest_payloads(
            stored_metadata,
            policy_fingerprint=semantic_policy.fingerprint,
        )
        if stored_manifests is not None:
            if reuse_trailing_items and len(stored_manifests) >= len(intent.items):
                stored_manifests = stored_manifests[-len(intent.items) :] if intent.items else ()
            elif len(stored_manifests) != len(intent.items):
                stored_manifests = None

        async def plan_item(position: int):
            item = intent.items[position]
            if stored_manifests is not None:
                try:
                    return segmenter.reuse(item.content, stored_manifests[position])
                except SegmentationReuseError:
                    pass
            async with planning_semaphore:
                return await segmenter.plan_document(item.content)

        results = await _gather_planning_tasks(tuple(plan_item(position) for position in range(len(intent.items))))
        chunks = build_chunk_plans_from_segmentation(
            intent.document_id,
            intent.items,
            results,
        )
        manifests = tuple(result.manifest for result in results)
        metadata, layout_signature = _semantic_document_metadata(
            policy=semantic_policy,
            items=intent.items,
            manifests=manifests,
        )
        usage = TokenUsage()
        for result in results:
            usage = usage + result.usage
        return _SemanticDocumentPlan(
            chunks=chunks,
            metadata=metadata,
            usage=usage,
            layout_signature=layout_signature,
        )

    async def _preflight_documents(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        intents: Sequence[DocumentIntent],
        policy: ChunkPolicy,
        *,
        checkpoint: OperationCheckpoint,
        request_started_at: datetime,
    ) -> tuple[_DocumentExecutionPlan, ...]:
        adapters = _backend_adapters(execution)
        snapshots: list[_DocumentPreflightSnapshot] = []

        # Read every database dependency first, then release the connection
        # before any semantic boundary provider call begins.
        async with acquire_with_retry(execution.pool) as connection, adapters.planning_snapshot(connection):
            repository = adapters.planning_repository(connection, schema=execution.schema)
            for submitted_intent in intents:
                existing = await repository.load_document(
                    invocation.bank_id,
                    submitted_intent.document_id,
                )
                existing_chunks = (
                    await repository.load_chunks(invocation.bank_id, submitted_intent.document_id)
                    if existing is not None
                    else ()
                )
                recovered_unit_bindings = None
                expected_unit_ids = None
                if checkpoint.is_core_committed(submitted_intent.document_id):
                    if existing is None:
                        raise RetainCheckpointRecoveryError(
                            "Operation checkpoint says document "
                            f"{submitted_intent.document_id!r} committed, but no document row exists"
                        )
                    expected_unit_ids = checkpoint.unit_ids_for_document(submitted_intent.document_id)
                    if expected_unit_ids is None:
                        raise RetainCheckpointRecoveryError(
                            "Operation checkpoint says document "
                            f"{submitted_intent.document_id!r} committed, but does not contain "
                            "operation-local unit IDs"
                        )
                    try:
                        recovered_unit_bindings = await repository.load_document_unit_bindings(
                            invocation.bank_id,
                            submitted_intent.document_id,
                            expected_unit_ids=expected_unit_ids,
                        )
                    except (TypeError, ValueError) as exc:
                        raise RetainCheckpointRecoveryError(
                            "Operation checkpoint unit IDs cannot be reconciled for document "
                            f"{submitted_intent.document_id!r}"
                        ) from exc
                snapshots.append(
                    _DocumentPreflightSnapshot(
                        submitted_intent=submitted_intent,
                        existing=existing,
                        existing_chunks=existing_chunks,
                        recovered_unit_bindings=recovered_unit_bindings,
                        expected_unit_ids=expected_unit_ids,
                    )
                )

        semantic_active = _semantic_planning_active(invocation, execution)
        planning_semaphore = asyncio.Semaphore(_projection_pipeline_concurrency(execution.resolved_config))

        async def materialize(snapshot: _DocumentPreflightSnapshot) -> _DocumentExecutionPlan:
            submitted_intent = snapshot.submitted_intent
            existing = snapshot.existing
            committed = snapshot.recovered_unit_bindings is not None
            intent = submitted_intent
            append_suffix_recovery = committed and submitted_intent.update_mode is UpdateMode.APPEND
            if (
                not append_suffix_recovery
                and submitted_intent.update_mode is UpdateMode.APPEND
                and existing is not None
                and existing.original_text
            ):
                intent = prepend_existing_document(submitted_intent, existing.original_text)

            stored_metadata = _existing_semantic_metadata(existing)
            combined_content = "\n".join(item.content for item in intent.items)
            if committed:
                if existing is None:  # pragma: no cover - snapshot invariant
                    raise RetainCheckpointRecoveryError(
                        f"Committed document {submitted_intent.document_id!r} disappeared after planning"
                    )
                if intent.update_mode is not UpdateMode.APPEND and combined_content != existing.original_text:
                    raise RetainCheckpointRecoveryError(
                        f"Committed document {submitted_intent.document_id!r} retry input "
                        "does not match its durable document"
                    )
                bindings = snapshot.recovered_unit_bindings
                single_replacement = len(intent.items) == 1 and intent.update_mode is not UpdateMode.APPEND
                has_semantic_metadata = (
                    isinstance(existing.retain_params, dict) and _SEMANTIC_PLAN_METADATA_KEY in existing.retain_params
                )
                all_bindings_chunkless = bool(bindings) and all(binding.chunk_index is None for binding in bindings)
                mapping_required = intent.update_mode is UpdateMode.APPEND or (
                    bool(bindings)
                    and not (single_replacement and (not has_semantic_metadata or all_bindings_chunkless))
                )
                recovered_chunk_sources = (
                    _committed_recovery_chunk_sources(
                        existing,
                        intent,
                        snapshot.existing_chunks,
                        policy,
                    )
                    if mapping_required
                    else ()
                )
                fallback_checkpoint = checkpoint.unscoped_facts_committed and not checkpoint.core_committed_document_ids
                return _DocumentExecutionPlan(
                    intent=intent,
                    chunks=(),
                    combined_content=combined_content,
                    existing=existing,
                    existing_chunks=(),
                    change=DocumentChangePlan(
                        kind=DocumentChangeKind.METADATA_ONLY,
                        reason="operation core commit recovered",
                    ),
                    recovered_unit_bindings=snapshot.recovered_unit_bindings,
                    recovered_chunk_sources=recovered_chunk_sources,
                    final_ann_pending=(
                        submitted_intent.document_id in checkpoint.final_ann_pending_document_ids or fallback_checkpoint
                    ),
                    segmentation_metadata=stored_metadata,
                    segmentation_usage=TokenUsage(),
                )

            semantic_plan = None
            if semantic_active:
                semantic_plan = await self._build_semantic_document_plan(
                    execution,
                    intent,
                    policy,
                    stored_metadata=stored_metadata,
                    planning_semaphore=planning_semaphore,
                    reuse_trailing_items=append_suffix_recovery,
                )
                chunks = semantic_plan.chunks
            else:
                chunks = build_chunk_plans(intent.document_id, intent.items, policy)

            policy_compatible = not (
                existing is not None
                and isinstance(existing.content_hash, str)
                and existing.content_hash.startswith(_INFLIGHT_CONTENT_HASH_PREFIX)
            )
            if semantic_plan is not None:
                semantic_policy = _semantic_policy(execution, policy)
                stored_layout = _stored_semantic_layout(
                    stored_metadata,
                    policy_fingerprint=semantic_policy.fingerprint,
                )
                policy_compatible = (
                    policy_compatible
                    and stored_layout is not None
                    and stored_layout == semantic_plan.layout_signature
                    and submitted_intent.update_mode is not UpdateMode.APPEND
                )
            elif stored_metadata is not None:
                # Switching from semantic chunks back to the deterministic
                # fixed-chunk policy is a policy migration, never a partial Delta.
                policy_compatible = False

            change = detect_document_change(
                chunks,
                snapshot.existing_chunks,
                document_exists=existing is not None,
                existing_document_content_hash=existing.content_hash if existing is not None else None,
                new_document_content_hash=compute_document_hash(combined_content),
                updated_at=existing.updated_at if existing is not None else None,
                request_started_at=request_started_at,
                policy_compatible=policy_compatible,
            )
            return _DocumentExecutionPlan(
                intent=intent,
                chunks=chunks,
                combined_content=combined_content,
                existing=existing,
                existing_chunks=snapshot.existing_chunks,
                change=change,
                segmentation_metadata=(semantic_plan.metadata if semantic_plan is not None else None),
                segmentation_usage=(semantic_plan.usage if semantic_plan is not None else TokenUsage()),
            )

        plans = await _gather_planning_tasks(tuple(materialize(snapshot) for snapshot in snapshots))
        if semantic_active:
            strategies: dict[str, int] = {}
            semantic_items = 0
            input_tokens = 0
            output_tokens = 0
            for plan in plans:
                metadata = plan.segmentation_metadata or {}
                for item in metadata.get("items", []):
                    manifest = item.get("manifest", {}) if isinstance(item, dict) else {}
                    strategy = manifest.get("effective_strategy")
                    if isinstance(strategy, str):
                        strategies[strategy] = strategies.get(strategy, 0) + 1
                        if strategy == EffectiveSegmentationStrategy.SEMANTIC.value:
                            semantic_items += 1
                input_tokens += plan.segmentation_usage.input_tokens
                output_tokens += plan.segmentation_usage.output_tokens
            logger.info(
                "Retain semantic planning: documents=%d strategies=%s semantic_items=%d "
                "input_tokens=%d output_tokens=%d",
                len(plans),
                json.dumps(strategies, sort_keys=True, separators=(",", ":")),
                semantic_items,
                input_tokens,
                output_tokens,
            )
        return plans

    async def _execute_document(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        *,
        agent_name: str,
        outbox_callback: Any,
    ) -> _DocumentOutcome:
        if plan.change.kind is DocumentChangeKind.FULL:
            return await self._execute_full_document_windows(
                invocation,
                execution,
                plan,
                agent_name=agent_name,
                outbox_callback=outbox_callback,
            )

        selected_chunks = self._chunks_for_extraction(plan)
        processed_tokens = self._processed_chunk_tokens(plan, selected_chunks)
        records, extraction_usage = await self._extract_and_project_selected_chunks(
            invocation,
            execution,
            plan,
            selected_chunks,
            agent_name=agent_name,
        )
        checkpoint_callback = self._compose_checkpoint_callback(
            invocation,
            execution,
            plan,
            expected_unit_ids_count=len(records),
        )

        ownership = _backend_adapters(execution).document_ownership(
            schema=execution.schema,
            fresh=plan.existing is None,
        )
        adapter = PersistenceWriter(
            pool=execution.pool,
            embeddings_model=execution.embeddings_model,
            entity_resolver=execution.entity_resolver,
            config=execution.resolved_config,
            ownership=ownership,
            operation_activity=_backend_adapters(execution).operation_activity_fence(
                invocation.operation_id,
                schema=execution.schema,
            ),
            schema=execution.schema,
            sanitize_log_identifiers=invocation.sanitize_log_identifiers,
        )
        unit_of_work = RetainUnitOfWork(
            connection_scope=lambda: acquire_with_retry(execution.pool),
            adapter=adapter,
        )

        try:
            async with _database_budget(execution.db_semaphore):
                request = await self._build_write_request(
                    invocation,
                    execution,
                    plan,
                    selected_chunks,
                    records,
                    processed_tokens=processed_tokens,
                    checkpoint_callback=checkpoint_callback,
                    outbox_callback=outbox_callback,
                )
                result = await unit_of_work.execute(request)
        except FreshDocumentOwnershipConflict as exc:
            execution.entity_resolver.discard_pending_stats()
            raise RetainOwnershipLostError(
                "Retain lost fresh-document ownership for "
                f"{_log_identifier(invocation, plan.intent.document_id)!r}; retry"
            ) from exc
        except BaseException:
            # Phase 1 accumulates resolver stats before the core transaction.
            # Never retain those task-local entries after rollback, commit
            # failure, cancellation, or an adapter contract error.
            execution.entity_resolver.discard_pending_stats()
            raise

        if result.core.ownership is OwnershipDisposition.LOST:
            # Phase 1 may have accumulated resolver statistics even though the
            # hash guard prevented every core write. They must never leak into
            # a later successful request's post-commit flush.
            execution.entity_resolver.discard_pending_stats()
            raise RetainOwnershipLostError(
                f"Retain lost document ownership for {_log_identifier(invocation, plan.intent.document_id)!r}; retry"
            )
        self._log_post_commit_failure(invocation, plan, result)
        return _DocumentOutcome(
            unit_ids_by_content=result.core.unit_ids_by_content,
            usage=extraction_usage,
            processed_tokens=(
                result.core.processed_tokens if result.core.processed_tokens is not None else processed_tokens
            ),
        )

    async def _execute_full_document_windows(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        *,
        agent_name: str,
        outbox_callback: Any,
    ) -> _DocumentOutcome:
        """Prepare ordered FULL windows, then publish them atomically."""

        windows = plan_full_write_windows(
            plan.chunks,
            getattr(execution.resolved_config, "retain_chunk_batch_size", 100),
        )
        final_content_hash = compute_document_hash(plan.combined_content)
        inflight_content_hash = f"{_INFLIGHT_CONTENT_HASH_PREFIX}{uuid.uuid4()}" if len(windows) > 1 else None
        prepared_steps: list[AtomicWriteStep] = []
        prepared_records: list[tuple[MemoryRecord, ...]] = []
        total_usage = TokenUsage()

        # Phase 1 may keep task-local resolver statistics. Reset once before
        # preparing the complete atomic publication, then let every window
        # contribute to the same post-commit flush.
        execution.entity_resolver.discard_pending_stats()
        try:
            for window in windows:
                records, extraction_usage = await self._extract_and_project_selected_chunks(
                    invocation,
                    execution,
                    plan,
                    window.chunks,
                    agent_name=agent_name,
                    fact_position_offset=(window.global_indices[0] if window.global_indices else 0),
                )
                total_usage = total_usage + extraction_usage
                request = await self._build_full_window_request(
                    invocation,
                    execution,
                    plan,
                    window.chunks,
                    records,
                    is_first=window.is_first,
                    is_last=window.is_last,
                    inflight_content_hash=inflight_content_hash,
                    final_content_hash=final_content_hash,
                    reset_pending_stats=False,
                )

                ownership = _backend_adapters(execution).document_ownership(
                    schema=execution.schema,
                    fresh=window.is_first and plan.existing is None,
                )
                prepared_steps.append(
                    AtomicWriteStep(
                        adapter=PersistenceWriter(
                            pool=execution.pool,
                            embeddings_model=execution.embeddings_model,
                            entity_resolver=execution.entity_resolver,
                            config=execution.resolved_config,
                            ownership=ownership,
                            operation_activity=_backend_adapters(execution).operation_activity_fence(
                                invocation.operation_id,
                                schema=execution.schema,
                            ),
                            schema=execution.schema,
                            sanitize_log_identifiers=invocation.sanitize_log_identifiers,
                        ),
                        request=request,
                    )
                )
                prepared_records.append(tuple(records))

            checkpoint_callback = self._compose_checkpoint_callback(
                invocation,
                execution,
                plan,
                expected_unit_ids_count=sum(len(records) for records in prepared_records),
            )
            validated_publication: _AtomicFullPublication | None = None

            def validate_publication(cores: tuple[CoreWriteResult, ...]) -> None:
                nonlocal validated_publication
                validated_publication = _validate_atomic_full_publication(
                    tuple(item.source_index for item in plan.intent.items),
                    prepared_records,
                    cores,
                )

            async def finalize_publication(
                connection: Any,
                _cores: tuple[CoreWriteResult, ...],
            ) -> None:
                if validated_publication is None:  # pragma: no cover - UoW callback order invariant
                    raise AssertionError("Atomic FULL publication was not validated")
                if checkpoint_callback is not None:
                    await checkpoint_callback(
                        connection,
                        (validated_publication.committed_unit_ids,),
                    )
                if outbox_callback is not None:
                    await outbox_callback(connection)

            atomic_unit_of_work = AtomicRetainUnitOfWork(
                connection_scope=lambda: acquire_with_retry(execution.pool),
            )
            async with _database_budget(execution.db_semaphore):
                results = await atomic_unit_of_work.execute(
                    prepared_steps,
                    validation_callback=validate_publication,
                    commit_callback=finalize_publication,
                )
        except FreshDocumentOwnershipConflict as exc:
            execution.entity_resolver.discard_pending_stats()
            raise RetainOwnershipLostError(
                "Retain lost fresh-document ownership for "
                f"{_log_identifier(invocation, plan.intent.document_id)!r}; retry"
            ) from exc
        except AtomicWriteOwnershipLost as exc:
            execution.entity_resolver.discard_pending_stats()
            raise RetainOwnershipLostError(
                "Retain lost document ownership for "
                f"{_log_identifier(invocation, plan.intent.document_id)!r} "
                f"at FULL window {exc.window_index}; retry"
            ) from exc
        except BaseException:
            execution.entity_resolver.discard_pending_stats()
            raise

        if validated_publication is None:  # pragma: no cover - UoW callback order invariant
            raise AssertionError("Atomic FULL publication committed without validation")
        for window, result in zip(windows, results, strict=True):
            if result.core.ownership is OwnershipDisposition.LOST:  # pragma: no cover - atomic UoW raises
                raise RetainOwnershipLostError(
                    "Retain lost document ownership for "
                    f"{_log_identifier(invocation, plan.intent.document_id)!r} "
                    f"at FULL window {window.window_index}; retry"
                )
            self._log_post_commit_failure(invocation, plan, result)

        if validated_publication.committed_unit_ids:
            final_ann_completed = await self._run_full_semantic_ann_best_effort(
                invocation,
                execution,
                plan,
                validated_publication.committed_unit_ids,
            )
            if final_ann_completed:
                await self._record_final_ann_completed_best_effort(
                    invocation,
                    execution,
                    plan.intent.document_id,
                )
        return _DocumentOutcome(
            unit_ids_by_content=validated_publication.unit_ids_by_content,
            usage=total_usage,
            processed_tokens=None,
        )

    @staticmethod
    def _log_post_commit_failure(
        invocation: RetainInvocation,
        plan: _DocumentExecutionPlan,
        result: Any,
    ) -> None:
        if result.post_commit.status is not PostCommitStatus.FAILED:
            return
        failure = result.post_commit.failure
        sanitizer = _request_sanitizer(invocation, plan.intent.document_id)
        _log_warning(
            invocation,
            "Retain post-commit stage failed for document %s at %s: %s",
            sanitizer.identifier(plan.intent.document_id),
            failure.stage.value if failure is not None else "unknown",
            (sanitizer.text(failure.exception) if failure is not None else "unknown failure"),
            identifiers=(plan.intent.document_id,),
        )

    async def _resume_committed_document(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
    ) -> _DocumentOutcome:
        """Resume only the idempotent post-commit work for a durable core write."""

        if plan.recovered_unit_bindings is None:  # pragma: no cover - caller invariant
            raise RetainCheckpointRecoveryError("Recovery plan has no committed unit-ID snapshot")
        recovered_unit_ids = tuple(binding.unit_id for binding in plan.recovered_unit_bindings)
        # Validate the durable unit/chunk/source mapping before running ANN or
        # clearing its retry marker.  Corrupt recovery state must have no
        # post-commit side effects.
        buckets = self._recovery_result_buckets(plan)
        if plan.final_ann_pending:
            final_ann_completed = True
            if recovered_unit_ids:
                final_ann_completed = await self._run_full_semantic_ann_best_effort(
                    invocation,
                    execution,
                    plan,
                    list(recovered_unit_ids),
                )
            if final_ann_completed:
                await self._record_final_ann_completed_best_effort(
                    invocation,
                    execution,
                    plan.intent.document_id,
                )
        return _DocumentOutcome(
            unit_ids_by_content=buckets,
            usage=TokenUsage(),
            processed_tokens=0,
        )

    @staticmethod
    def _recovery_result_buckets(
        plan: _DocumentExecutionPlan,
    ) -> tuple[tuple[str, ...], ...]:
        """Restore committed unit IDs to their original public content buckets.

        Units with a durable chunk association can be mapped back to their
        immutable input source. Rows without a chunk association are safe only
        for an exact, single-input replacement; ambiguous requests fail closed.
        """

        bindings = plan.recovered_unit_bindings
        if bindings is None:  # pragma: no cover - caller invariant
            raise RetainCheckpointRecoveryError("Recovery plan has no committed unit bindings")
        if not plan.intent.items:
            raise RetainCheckpointRecoveryError("Recovery plan has no content items")
        if not bindings:
            return tuple(() for _ in plan.intent.items)

        recovered_unit_ids = tuple(binding.unit_id for binding in bindings)
        if len(plan.intent.items) == 1 and plan.intent.update_mode is not UpdateMode.APPEND:
            return (recovered_unit_ids,)
        if any(binding.chunk_index is None for binding in bindings):
            raise RetainCheckpointRecoveryError(
                f"Committed recovery for document {plan.intent.document_id!r} "
                "contains units without an unambiguous chunk source"
            )

        sources_by_chunk_index: dict[int, int | None] = {}
        if plan.recovered_chunk_sources is not None:
            for chunk_index, source_index in plan.recovered_chunk_sources:
                if chunk_index in sources_by_chunk_index:
                    raise RetainCheckpointRecoveryError(f"Recovery plan has duplicate chunk index {chunk_index}")
                sources_by_chunk_index[chunk_index] = source_index
        else:
            for chunk in plan.chunks:
                if chunk.global_index in sources_by_chunk_index:
                    raise RetainCheckpointRecoveryError(f"Recovery plan has duplicate chunk index {chunk.global_index}")
                sources_by_chunk_index[chunk.global_index] = chunk.source_index

        try:
            positions = content_positions(plan.intent.items)
        except (TypeError, ValueError) as exc:
            raise RetainCheckpointRecoveryError(
                f"Recovery plan for document {plan.intent.document_id!r} has invalid content sources"
            ) from exc

        buckets: list[list[str]] = [[] for _ in plan.intent.items]
        for binding in bindings:
            chunk_index = binding.chunk_index
            if chunk_index is None:  # pragma: no cover - handled by fallback above
                raise AssertionError("unmapped recovery binding escaped fallback")
            try:
                source_index = sources_by_chunk_index[chunk_index]
            except KeyError as exc:
                raise RetainCheckpointRecoveryError(
                    f"Committed unit {binding.unit_id!r} references unknown chunk index "
                    f"{chunk_index} for document {plan.intent.document_id!r}"
                ) from exc
            if source_index is None:
                # An append may regenerate units for changed synthetic
                # pre-append chunks.  They remain part of post-commit ANN but
                # never occupy a caller-visible input bucket.
                continue
            try:
                content_position = positions[source_index]
            except KeyError as exc:  # pragma: no cover - chunk planner owns this invariant
                raise RetainCheckpointRecoveryError(
                    f"Recovery chunk index {chunk_index} references unknown source {source_index!r}"
                ) from exc
            buckets[content_position].append(binding.unit_id)
        return tuple(tuple(bucket) for bucket in buckets)

    @staticmethod
    def _compose_checkpoint_callback(
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        *,
        expected_unit_ids_count: int,
        prior_unit_ids: tuple[str, ...] = (),
    ) -> Any:
        """Build the strict in-transaction core checkpoint callback."""

        if invocation.operation_id is None:
            return None

        async def record_exact_core_commit(
            connection: Any,
            current_unit_ids_by_content: tuple[tuple[str, ...], ...],
        ) -> None:
            current_unit_ids = tuple(unit_id for bucket in current_unit_ids_by_content for unit_id in bucket)
            unit_ids = (*prior_unit_ids, *current_unit_ids)
            if len(unit_ids) != expected_unit_ids_count:
                raise RetainCheckpointRecoveryError(
                    f"Core write for document {plan.intent.document_id!r} returned "
                    f"{len(unit_ids)} unit IDs; expected {expected_unit_ids_count}"
                )
            if len(unit_ids) != len(set(unit_ids)):
                raise RetainCheckpointRecoveryError(
                    f"Core write for document {plan.intent.document_id!r} returned duplicate unit IDs"
                )
            await (
                _backend_adapters(execution)
                .checkpoint_store(
                    connection,
                    schema=execution.schema,
                )
                .record_core_committed(
                    invocation.operation_id,
                    plan.intent.document_id,
                    unit_ids=unit_ids,
                    requires_final_ann=(plan.change.kind is DocumentChangeKind.FULL),
                )
            )

        return record_exact_core_commit

    @staticmethod
    async def _record_final_ann_completed_best_effort(
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        document_id: str,
    ) -> None:
        if invocation.operation_id is None:
            return
        try:
            async with acquire_with_retry(execution.pool) as connection:
                await (
                    _backend_adapters(execution)
                    .checkpoint_store(
                        connection,
                        schema=execution.schema,
                    )
                    .record_final_ann_completed(
                        invocation.operation_id,
                        document_id,
                    )
                )
        except Exception:
            # The final ANN pass is idempotent and best effort.  Leaving its
            # marker in place causes a safe retry rather than a duplicate core
            # write or outbox event.
            _log_warning(
                invocation,
                "Retain could not clear final ANN checkpoint for operation %s, document %s",
                _log_identifier(invocation, invocation.operation_id),
                _log_identifier(invocation, document_id),
                identifiers=(document_id,),
                exc_info=True,
            )

    @staticmethod
    async def _run_full_semantic_ann_best_effort(
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        committed_unit_ids: list[str],
    ) -> bool:
        """Run the deferred semantic-link pass and report whether it completed."""

        try:
            await run_final_semantic_ann(
                execution.pool,
                invocation.bank_id,
                committed_unit_ids,
                execution.resolved_config,
                [],
            )
            return True
        except Exception:
            # Facts and the retrieval-critical core graph are already committed,
            # so semantic ANN remains a post-commit best-effort operation. The
            # caller must preserve the durable retry marker after a failure.
            _log_warning(
                invocation,
                "Retain final semantic ANN failed for document %s",
                _log_identifier(invocation, plan.intent.document_id),
                identifiers=(plan.intent.document_id,),
                exc_info=True,
            )
            return False

    @staticmethod
    def _chunks_for_extraction(plan: _DocumentExecutionPlan) -> tuple[ChunkPlan, ...]:
        if plan.change.kind is DocumentChangeKind.FULL:
            return plan.chunks
        if plan.change.kind is DocumentChangeKind.DELTA:
            selected = set(plan.change.chunks_to_process)
            return tuple(chunk for chunk in plan.chunks if chunk.global_index in selected)
        return ()

    @staticmethod
    def _processed_chunk_tokens(
        plan: _DocumentExecutionPlan,
        selected_chunks: Sequence[ChunkPlan],
    ) -> int:
        items_by_source = {item.source_index: item for item in plan.intent.items}
        total = 0
        for chunk in selected_chunks:
            try:
                item = items_by_source[chunk.source_index]
            except KeyError as exc:  # pragma: no cover - chunk planning owns this invariant
                raise RetainResultMappingError(f"Chunk {chunk.chunk_key!r} references an unknown source item") from exc
            total += count_tokens(chunk.text)
            total += count_tokens(item.context)
        return total

    async def _extract_and_project_selected_chunks(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        selected_chunks: Sequence[ChunkPlan],
        *,
        agent_name: str,
        fact_position_offset: int = 0,
    ) -> tuple[tuple[MemoryRecord, ...], TokenUsage]:
        mode = ExtractionMode(execution.resolved_config.retain_extraction_mode)
        if not selected_chunks:
            return (), TokenUsage()
        configured_concurrency = _projection_pipeline_concurrency(execution.resolved_config)
        window_started = time.perf_counter()
        extraction_seconds: list[float] = []
        embedding_seconds: list[float] = []
        embedder = EmbeddingModelAdapter(execution.embeddings_model)
        embedding_model_version_value = embedding_model_version(execution.embeddings_model)
        extraction_version = getattr(execution.resolved_config, "extraction_prompt_version", "5w-v1")

        async def project(candidates) -> tuple[tuple[MemoryRecord, ...], float]:
            started = time.perf_counter()
            records = await project_embeddings(
                candidates,
                embedder=embedder,
                format_date=execution.format_date_fn,
                embedding_model_version=embedding_model_version_value,
                extraction_version=extraction_version,
                failure_policy=getattr(
                    execution.resolved_config,
                    "retain_embedding_failure_policy",
                    EmbeddingFailurePolicy.STORE_WITHOUT_EMBEDDING,
                ),
            )
            return records, time.perf_counter() - started

        async def extract_structured(chunks: Sequence[ChunkPlan]):
            layout = build_prechunked_extraction_layout(
                plan.intent.items,
                chunks,
            )
            extraction = await FactExtractorAdapter(
                llm_config=execution.llm_config,
                config=execution.resolved_config,
                agent_name=agent_name,
                pool=execution.pool,
                operation_id=invocation.operation_id,
                schema=execution.schema,
                batch_checkpoint_clearer=self._provider_batch_checkpoint_clearer(
                    invocation,
                    execution,
                ),
            ).extract(
                layout.extraction_request(
                    ExtractionPolicy(
                        mode=mode,
                        fact_type_override=invocation.fact_type_override,
                    )
                )
            )
            return layout.remap_result(extraction)

        chunk_batch_size = getattr(execution.resolved_config, "retain_chunk_batch_size", 100)
        pipeline_selected_chunks = (
            mode is not ExtractionMode.CHUNKS
            and len(selected_chunks) > 1
            and not getattr(execution.resolved_config, "retain_batch_enabled", False)
            and isinstance(chunk_batch_size, int)
            and not isinstance(chunk_batch_size, bool)
            and chunk_batch_size > 0
            and len(selected_chunks) <= chunk_batch_size
        )

        if pipeline_selected_chunks:
            extraction_semaphore = asyncio.Semaphore(configured_concurrency)
            embedding_semaphore = asyncio.Semaphore(configured_concurrency)

            async def extract_and_project_one(chunk: ChunkPlan) -> _ProjectedChunkOutcome:
                async with extraction_semaphore:
                    extract_started = time.perf_counter()
                    extraction = await extract_structured((chunk,))
                    extract_elapsed = time.perf_counter() - extract_started

                async with embedding_semaphore:
                    records, embed_elapsed = await project(extraction.candidates)
                return _ProjectedChunkOutcome(
                    records=records,
                    usage=extraction.usage,
                    extraction_seconds=extract_elapsed,
                    embedding_seconds=embed_elapsed,
                )

            results = await asyncio.gather(
                *(extract_and_project_one(chunk) for chunk in selected_chunks),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    # gather preserves input order and waits for every sibling.
                    # Raise the earliest source-chunk failure only after all
                    # provider work has reached a terminal state.
                    raise result

            records: list[MemoryRecord] = []
            extraction_usage = TokenUsage()
            for result in results:
                # Every BaseException was rejected in the loop above.
                assert isinstance(result, _ProjectedChunkOutcome)
                records.extend(result.records)
                extraction_usage = extraction_usage + result.usage
                extraction_seconds.append(result.extraction_seconds)
                embedding_seconds.append(result.embedding_seconds)
            frozen_records = tuple(records)
            _log_window_producer_summary(
                path="pipelined",
                chunks=len(selected_chunks),
                facts=len(frozen_records),
                wall_seconds=time.perf_counter() - window_started,
                extraction_seconds=extraction_seconds,
                embedding_seconds=embedding_seconds,
                configured_concurrency=configured_concurrency,
            )
            return frozen_records, extraction_usage

        if mode is ExtractionMode.CHUNKS:
            candidates = extract_passthrough(
                selected_chunks,
                plan.intent.items,
                fact_type_override=invocation.fact_type_override,
                fact_position_offset=fact_position_offset,
            )
            extraction_usage = TokenUsage()
        else:
            extract_started = time.perf_counter()
            extraction = await extract_structured(selected_chunks)
            extraction_seconds.append(time.perf_counter() - extract_started)
            candidates = extraction.candidates
            extraction_usage = extraction.usage

        records, embed_elapsed = await project(candidates)
        embedding_seconds.append(embed_elapsed)
        _log_window_producer_summary(
            path="window_batch",
            chunks=len(selected_chunks),
            facts=len(records),
            wall_seconds=time.perf_counter() - window_started,
            extraction_seconds=extraction_seconds,
            embedding_seconds=embedding_seconds,
            configured_concurrency=configured_concurrency,
        )
        return records, extraction_usage

    @staticmethod
    def _provider_batch_checkpoint_clearer(
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
    ) -> Any:
        if invocation.operation_id is None:
            return None

        async def clear_completed_provider_batch() -> None:
            async with acquire_with_retry(execution.pool) as connection:
                await (
                    _backend_adapters(execution)
                    .checkpoint_store(
                        connection,
                        schema=execution.schema,
                    )
                    .clear_provider_batch(invocation.operation_id)
                )

        return clear_completed_provider_batch

    async def _build_full_window_request(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        selected_chunks: Sequence[ChunkPlan],
        records: Sequence[MemoryRecord],
        *,
        is_first: bool,
        is_last: bool,
        inflight_content_hash: str | None,
        final_content_hash: str,
        checkpoint_callback: Any = None,
        outbox_callback: Any = None,
        reset_pending_stats: bool = True,
    ) -> WriteWindowRequest:
        retain_params, document_tags = _retain_metadata_with_semantic_plan(plan)
        payload = await self._build_fact_payload(
            invocation,
            execution,
            plan,
            selected_chunks,
            records,
            reset_pending_stats=reset_pending_stats,
        )
        if is_first:
            document_window = FirstFullWriteWindow(
                combined_content=plan.combined_content,
                is_first_batch=True,
                retain_params=retain_params,
                document_tags=document_tags,
                recovery=False,
                expected_existing_content_hash=(plan.existing.content_hash if plan.existing is not None else None),
                expects_unhashed_existing_document=(plan.existing is not None and not plan.existing.content_hash),
                continuation_content_hash=(inflight_content_hash if not is_last else None),
            )
        else:
            if inflight_content_hash is None:  # pragma: no cover - window planner invariant
                raise RetainError("Later FULL window requires an in-flight ownership hash")
            document_window = LaterFullWriteWindow(
                expected_content_hash=inflight_content_hash,
                completed_content_hash=final_content_hash if is_last else None,
            )
        return WriteWindowRequest(
            bank_id=invocation.bank_id,
            document_id=plan.intent.document_id,
            document_window=document_window,
            contents=payload.storage_contents,
            chunks=payload.chunks,
            facts=payload.facts,
            graph=payload.graph,
            skip_semantic_links=True,
            checkpoint_callback=checkpoint_callback,
            outbox_callback=outbox_callback,
            log_buffer=[],
        )

    async def _build_write_request(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        selected_chunks: Sequence[ChunkPlan],
        records: Sequence[MemoryRecord],
        *,
        processed_tokens: int | None,
        checkpoint_callback: Any = None,
        outbox_callback: Any = None,
    ) -> RetainWriteRequest:
        retain_params, document_tags = _retain_metadata_with_semantic_plan(plan)
        if plan.change.kind is DocumentChangeKind.METADATA_ONLY:
            if plan.existing is None or not plan.existing.content_hash:  # pragma: no cover - classifier invariant
                raise RetainError("Metadata-only change requires an existing hash snapshot")
            return MetadataOnlyWriteRequest(
                bank_id=invocation.bank_id,
                document_id=plan.intent.document_id,
                expected_content_hash=plan.existing.content_hash,
                combined_content=plan.combined_content,
                input_slot_count=len(plan.intent.items),
                retain_params=retain_params,
                document_tags=document_tags,
                checkpoint_callback=checkpoint_callback,
                outbox_callback=outbox_callback,
            )

        payload = await self._build_fact_payload(
            invocation,
            execution,
            plan,
            selected_chunks,
            records,
        )
        if plan.change.kind is not DocumentChangeKind.DELTA:
            raise RetainError(f"Cannot build a write request for {plan.change.kind.value}")
        if plan.existing is None or not plan.existing.content_hash:  # pragma: no cover - classifier invariant
            raise RetainError("Delta change requires an existing hash snapshot")

        existing_by_index = {chunk.chunk_index: chunk for chunk in plan.existing_chunks}

        def existing_write(index: int) -> ExistingChunkWrite:
            try:
                chunk = existing_by_index[index]
            except KeyError as exc:
                raise RetainError(f"Change plan references missing existing chunk index {index}") from exc
            return ExistingChunkWrite(chunk_id=chunk.chunk_id, chunk_index=index)

        return DeltaWriteRequest(
            bank_id=invocation.bank_id,
            document_id=plan.intent.document_id,
            expected_content_hash=plan.existing.content_hash,
            combined_content=plan.combined_content,
            contents=payload.storage_contents,
            unchanged_chunk_indices=plan.change.unchanged,
            changed_chunks=tuple(existing_write(index) for index in plan.change.changed),
            added_chunk_indices=plan.change.added,
            removed_chunks=tuple(existing_write(index) for index in plan.change.removed),
            chunks=payload.chunks,
            facts=payload.facts,
            graph=payload.graph,
            processed_tokens=processed_tokens or 0,
            retain_params=retain_params,
            document_tags=document_tags,
            skip_semantic_links=False,
            checkpoint_callback=checkpoint_callback,
            outbox_callback=outbox_callback,
            log_buffer=(),
        )

    async def _build_fact_payload(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
        plan: _DocumentExecutionPlan,
        selected_chunks: Sequence[ChunkPlan],
        records: Sequence[MemoryRecord],
        *,
        reset_pending_stats: bool = True,
    ) -> _FactPayload:
        storage_contents = tuple(content_to_storage(item) for item in plan.intent.items)
        positions = content_positions(plan.intent.items)
        fact_positions = {record.fact_key: position for position, record in enumerate(records)}
        fact_writes: list[FactWrite] = []
        processed_facts = []
        for record in records:
            try:
                content_index = positions[record.source_index]
            except KeyError as exc:  # pragma: no cover - passthrough validates this
                raise RetainResultMappingError(
                    f"Record {record.fact_key!r} references unknown source index {record.source_index!r}"
                ) from exc
            extracted = record_to_extracted_fact(record, content_index=content_index)
            processed = record_to_processed_fact(
                record,
                document_id=plan.intent.document_id,
                content_index=content_index,
                fact_positions=fact_positions,
            )
            fact_writes.append(
                FactWrite(
                    fact_key=record.fact_key,
                    chunk_key=record.chunk_key,
                    extracted=extracted,
                    processed=processed,
                )
            )
            processed_facts.append(processed)

        phase1 = None
        if processed_facts:
            if reset_pending_stats:
                execution.entity_resolver.discard_pending_stats()
            phase1_kwargs = {
                "skip_semantic_ann": (
                    plan.change.kind is DocumentChangeKind.FULL
                    or not getattr(execution.resolved_config, "write_semantic_links", True)
                )
            }
            phase1 = await pre_resolve_entities(
                execution.pool,
                execution.entity_resolver,
                invocation.bank_id,
                list(storage_contents),
                [fact.fact_key for fact in fact_writes],
                processed_facts,
                execution.resolved_config,
                [],
                **phase1_kwargs,
            )
        chunk_metadata = chunks_to_storage(selected_chunks, plan.intent.items, records)
        return _FactPayload(
            storage_contents=storage_contents,
            chunks=tuple(
                ChunkWrite(chunk_key=chunk.chunk_key, metadata=metadata)
                for chunk, metadata in zip(selected_chunks, chunk_metadata, strict=True)
            ),
            facts=tuple(fact_writes),
            graph=_graph_write(phase1),
        )

    @staticmethod
    def _merge_document_result(
        result_by_input: list[list[str]],
        intent: DocumentIntent,
        document_result: Sequence[Sequence[str]],
    ) -> None:
        if len(document_result) != len(intent.items):
            raise RetainResultMappingError(
                f"Document {intent.document_id!r} returned {len(document_result)} result buckets "
                f"for {len(intent.items)} content items"
            )
        for item, unit_ids in zip(intent.items, document_result, strict=True):
            if item.source_index is None:
                continue
            if not 0 <= item.source_index < len(result_by_input):
                raise RetainResultMappingError(
                    f"Document {intent.document_id!r} references out-of-range input slot {item.source_index}"
                )
            if result_by_input[item.source_index]:
                raise RetainResultMappingError(
                    f"Input slot {item.source_index} received results from more than one document"
                )
            bucket = list(unit_ids)
            if any(not isinstance(unit_id, str) or not unit_id for unit_id in bucket):
                raise RetainResultMappingError(f"Document {intent.document_id!r} returned an invalid unit ID")
            result_by_input[item.source_index] = bucket
