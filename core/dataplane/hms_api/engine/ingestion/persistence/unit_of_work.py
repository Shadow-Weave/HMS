"""Semantic write-window contracts and transaction orchestration for Retain.

The application layer submits one :class:`WriteWindowRequest`; a persistence
adapter owns all backend-specific work performed inside the transaction, while
this unit-of-work wrapper owns the transaction and post-commit failure boundary.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeAlias

from ...entity_resolution_contracts import EntityResolutionReadPlan
from ...retain.types import ChunkMetadata, ExtractedFact, ProcessedFact, RetainContent
from ..contracts import CoreCommitCallback, OutboxCallback
from ..domain import FrozenObject, freeze_json


class PersistenceContractError(ValueError):
    """A planned write cannot be mapped one-to-one to durable records."""


@dataclass(frozen=True, slots=True)
class FirstFullWriteWindow:
    """Document work that must occur in the first full-retain write window."""

    combined_content: str
    is_first_batch: bool = True
    retain_params: Mapping[str, Any] | None = None
    document_tags: tuple[str, ...] = ()
    recovery: bool = False
    # A full replacement planned from an existing document must lock the row
    # and prove that this exact hash is still current before deleting anything.
    # ``None`` is reserved for a genuinely fresh document claim.
    expected_existing_content_hash: str | None = None
    # Databases upgraded from releases that did not populate ``content_hash``
    # need a distinct ownership state. The writer must lock the existing row
    # and prove that its hash is still absent before replacing any data.
    expects_unhashed_existing_document: bool = False
    # Multi-window FULL writes replace the final document hash with a unique
    # in-flight ownership token after tracking the first window.  A single
    # window leaves this unset and publishes the final hash immediately.
    continuation_content_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.combined_content, str):
            raise TypeError("combined_content must be a string")
        if not isinstance(self.is_first_batch, bool):
            raise TypeError("is_first_batch must be a bool")
        if self.retain_params is not None and not isinstance(self.retain_params, Mapping):
            raise TypeError("retain_params must be a mapping or None")
        if not isinstance(self.document_tags, tuple) or any(not isinstance(tag, str) for tag in self.document_tags):
            raise TypeError("document_tags must be a tuple of strings")
        if not isinstance(self.recovery, bool):
            raise TypeError("recovery must be a bool")
        if self.expected_existing_content_hash is not None and (
            not isinstance(self.expected_existing_content_hash, str) or not self.expected_existing_content_hash
        ):
            raise ValueError("expected_existing_content_hash must be a non-empty string or None")
        if not isinstance(self.expects_unhashed_existing_document, bool):
            raise TypeError("expects_unhashed_existing_document must be a bool")
        if self.expected_existing_content_hash is not None and self.expects_unhashed_existing_document:
            raise ValueError("hashed and unhashed existing-document ownership are mutually exclusive")
        if self.recovery and (
            self.expected_existing_content_hash is not None or self.expects_unhashed_existing_document
        ):
            raise ValueError("recovery and existing-document full replacement are mutually exclusive")
        if self.continuation_content_hash is not None and (
            not isinstance(self.continuation_content_hash, str) or not self.continuation_content_hash
        ):
            raise ValueError("continuation_content_hash must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class LaterFullWriteWindow:
    """A later full-retain window that may write only while it owns the document."""

    expected_content_hash: str
    # Only the last window sets this to the final document content hash.  An
    # intermediate window keeps the in-flight token unchanged.
    completed_content_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expected_content_hash, str) or not self.expected_content_hash:
            raise ValueError("expected_content_hash must be a non-empty string")
        if self.completed_content_hash is not None and (
            not isinstance(self.completed_content_hash, str) or not self.completed_content_hash
        ):
            raise ValueError("completed_content_hash must be a non-empty string or None")


DocumentWindow: TypeAlias = FirstFullWriteWindow | LaterFullWriteWindow


@dataclass(frozen=True, slots=True)
class ChunkWrite:
    """Stable chunk identity paired with its storage DTO."""

    chunk_key: str
    metadata: ChunkMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_key, str) or not self.chunk_key:
            raise ValueError("chunk_key must be a non-empty string")
        if not isinstance(self.metadata, ChunkMetadata):
            raise TypeError("metadata must be ChunkMetadata")


@dataclass(frozen=True, slots=True)
class FactWrite:
    """Stable fact identity and its one-to-one storage representations."""

    fact_key: str
    chunk_key: str
    extracted: ExtractedFact
    processed: ProcessedFact

    def __post_init__(self) -> None:
        if not isinstance(self.fact_key, str) or not self.fact_key:
            raise ValueError("fact_key must be a non-empty string")
        if not isinstance(self.chunk_key, str) or not self.chunk_key:
            raise ValueError("chunk_key must be a non-empty string")
        if not isinstance(self.extracted, ExtractedFact):
            raise TypeError("extracted must be ExtractedFact")
        if not isinstance(self.processed, ProcessedFact):
            raise TypeError("processed must be ProcessedFact")


@dataclass(frozen=True, slots=True)
class ExistingChunkWrite:
    """Durable identity for an existing chunk affected by a delta write."""

    chunk_id: str
    chunk_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")


@dataclass(frozen=True, slots=True)
class CoreGraphWrite:
    """Entity-resolution and precomputed-link inputs for the core write."""

    resolved_entity_ids: tuple[str, ...] = ()
    entity_to_unit: tuple[tuple[Any, ...], ...] = ()
    unit_to_entity_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    semantic_ann_links: tuple[tuple[Any, ...], ...] = ()
    entity_read_plan: EntityResolutionReadPlan | None = None

    def __post_init__(self) -> None:
        unit_keys = [unit_id for unit_id, _entity_ids in self.unit_to_entity_ids]
        if len(unit_keys) != len(set(unit_keys)):
            raise PersistenceContractError("unit_to_entity_ids contains duplicate unit keys")
        if self.entity_read_plan is not None and (
            self.resolved_entity_ids or self.entity_to_unit or self.unit_to_entity_ids
        ):
            raise PersistenceContractError(
                "entity_read_plan cannot be combined with already-finalized entity graph data"
            )


@dataclass(frozen=True, slots=True)
class WriteWindowRequest:
    """All semantic writes that belong to one full-retain transaction."""

    bank_id: str
    document_id: str
    document_window: DocumentWindow
    contents: tuple[RetainContent, ...]
    chunks: tuple[ChunkWrite, ...] = ()
    facts: tuple[FactWrite, ...] = ()
    graph: CoreGraphWrite = field(default_factory=CoreGraphWrite)
    skip_semantic_links: bool = False
    checkpoint_callback: CoreCommitCallback | None = None
    outbox_callback: OutboxCallback | None = None
    log_buffer: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.bank_id, str) or not self.bank_id:
            raise ValueError("bank_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.document_window, (FirstFullWriteWindow, LaterFullWriteWindow)):
            raise TypeError("document_window must describe a first or later full write window")
        if not isinstance(self.contents, tuple) or any(not isinstance(item, RetainContent) for item in self.contents):
            raise TypeError("contents must be a tuple of RetainContent values")
        if not self.contents:
            raise PersistenceContractError("a write window must contain at least one content item")
        if not isinstance(self.chunks, tuple) or any(not isinstance(item, ChunkWrite) for item in self.chunks):
            raise TypeError("chunks must be a tuple of ChunkWrite values")
        if not isinstance(self.facts, tuple) or any(not isinstance(item, FactWrite) for item in self.facts):
            raise TypeError("facts must be a tuple of FactWrite values")
        if not isinstance(self.graph, CoreGraphWrite):
            raise TypeError("graph must be CoreGraphWrite")
        if not isinstance(self.skip_semantic_links, bool):
            raise TypeError("skip_semantic_links must be a bool")
        if self.checkpoint_callback is not None and not callable(self.checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable or None")
        if self.outbox_callback is not None and not callable(self.outbox_callback):
            raise TypeError("outbox_callback must be callable or None")
        if not isinstance(self.log_buffer, list):
            raise TypeError("log_buffer must be a list")
        _validate_identity_bindings(self.contents, self.chunks, self.facts)


def _freeze_retain_params(value: FrozenObject | Mapping[str, Any] | None) -> FrozenObject | None:
    if value is None or isinstance(value, FrozenObject):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("retain_params must be a mapping, FrozenObject, or None")
    frozen = freeze_json(dict(value))
    if not isinstance(frozen, FrozenObject):  # pragma: no cover - dicts freeze to objects
        raise AssertionError("retain_params must freeze to an object")
    return frozen


def _validate_identity_bindings(
    contents: Sequence[RetainContent],
    chunks: Sequence[ChunkWrite],
    facts: Sequence[FactWrite],
) -> None:
    chunk_keys = [item.chunk_key for item in chunks]
    if len(chunk_keys) != len(set(chunk_keys)):
        raise PersistenceContractError("chunks contain duplicate chunk_key values")

    chunk_indices = [item.metadata.chunk_index for item in chunks]
    if len(chunk_indices) != len(set(chunk_indices)):
        raise PersistenceContractError("chunks contain duplicate chunk_index values")

    fact_keys = [item.fact_key for item in facts]
    if len(fact_keys) != len(set(fact_keys)):
        raise PersistenceContractError("facts contain duplicate fact_key values")

    chunks_by_key = {item.chunk_key: item for item in chunks}
    fact_count_by_chunk = dict.fromkeys(chunk_keys, 0)
    content_count = len(contents)
    for fact in facts:
        chunk = chunks_by_key.get(fact.chunk_key)
        if chunk is None:
            raise PersistenceContractError(f"fact_key={fact.fact_key!r} refers to unknown chunk_key={fact.chunk_key!r}")
        fact_count_by_chunk[fact.chunk_key] += 1
        if fact.extracted.chunk_index != chunk.metadata.chunk_index:
            raise PersistenceContractError(
                f"fact_key={fact.fact_key!r} chunk_index does not match chunk_key={fact.chunk_key!r}"
            )
        if fact.extracted.content_index != fact.processed.content_index:
            raise PersistenceContractError(f"fact_key={fact.fact_key!r} extracted/processed content_index mismatch")
        if not 0 <= fact.processed.content_index < content_count:
            raise PersistenceContractError(f"fact_key={fact.fact_key!r} has an out-of-range content_index")

    for chunk in chunks:
        if not 0 <= chunk.metadata.content_index < content_count:
            raise PersistenceContractError(f"chunk_key={chunk.chunk_key!r} has an out-of-range content_index")
        actual_count = fact_count_by_chunk[chunk.chunk_key]
        if chunk.metadata.fact_count != actual_count:
            raise PersistenceContractError(
                f"chunk_key={chunk.chunk_key!r} declares {chunk.metadata.fact_count} facts, "
                f"but {actual_count} fact bindings were supplied"
            )


def _validate_request_header(*, bank_id: str, document_id: str, expected_content_hash: str) -> None:
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError("bank_id must be a non-empty string")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(expected_content_hash, str) or not expected_content_hash:
        raise ValueError("expected_content_hash must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MetadataOnlyWriteRequest:
    """Immutable metadata-only delta operation.

    ``input_slot_count`` preserves the public one-bucket-per-input outcome even
    though no content enters extraction and the processed-token outcome is
    therefore exactly zero.
    """

    bank_id: str
    document_id: str
    expected_content_hash: str
    combined_content: str
    input_slot_count: int
    retain_params: FrozenObject | Mapping[str, Any] | None = None
    document_tags: tuple[str, ...] = ()
    checkpoint_callback: CoreCommitCallback | None = None
    outbox_callback: OutboxCallback | None = None

    def __post_init__(self) -> None:
        _validate_request_header(
            bank_id=self.bank_id,
            document_id=self.document_id,
            expected_content_hash=self.expected_content_hash,
        )
        if not isinstance(self.combined_content, str):
            raise TypeError("combined_content must be a string")
        if isinstance(self.input_slot_count, bool) or not isinstance(self.input_slot_count, int):
            raise TypeError("input_slot_count must be an integer")
        if self.input_slot_count < 0:
            raise ValueError("input_slot_count must be non-negative")
        if not isinstance(self.document_tags, tuple) or any(not isinstance(tag, str) for tag in self.document_tags):
            raise TypeError("document_tags must be a tuple of strings")
        if self.checkpoint_callback is not None and not callable(self.checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable or None")
        if self.outbox_callback is not None and not callable(self.outbox_callback):
            raise TypeError("outbox_callback must be callable or None")
        object.__setattr__(self, "retain_params", _freeze_retain_params(self.retain_params))


@dataclass(frozen=True, slots=True)
class DeltaWriteRequest:
    """Immutable partial-document transaction planned from a hash snapshot."""

    bank_id: str
    document_id: str
    expected_content_hash: str
    combined_content: str
    contents: tuple[RetainContent, ...]
    unchanged_chunk_indices: tuple[int, ...]
    changed_chunks: tuple[ExistingChunkWrite, ...]
    added_chunk_indices: tuple[int, ...]
    removed_chunks: tuple[ExistingChunkWrite, ...]
    chunks: tuple[ChunkWrite, ...] = ()
    facts: tuple[FactWrite, ...] = ()
    graph: CoreGraphWrite = field(default_factory=CoreGraphWrite)
    processed_tokens: int = 0
    retain_params: FrozenObject | Mapping[str, Any] | None = None
    document_tags: tuple[str, ...] = ()
    skip_semantic_links: bool = False
    checkpoint_callback: CoreCommitCallback | None = None
    outbox_callback: OutboxCallback | None = None
    log_buffer: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_request_header(
            bank_id=self.bank_id,
            document_id=self.document_id,
            expected_content_hash=self.expected_content_hash,
        )
        if not isinstance(self.combined_content, str):
            raise TypeError("combined_content must be a string")
        if not isinstance(self.contents, tuple) or any(not isinstance(item, RetainContent) for item in self.contents):
            raise TypeError("contents must be a tuple of RetainContent values")
        if not isinstance(self.chunks, tuple) or any(not isinstance(item, ChunkWrite) for item in self.chunks):
            raise TypeError("chunks must be a tuple of ChunkWrite values")
        if not isinstance(self.facts, tuple) or any(not isinstance(item, FactWrite) for item in self.facts):
            raise TypeError("facts must be a tuple of FactWrite values")
        if not isinstance(self.graph, CoreGraphWrite):
            raise TypeError("graph must be CoreGraphWrite")
        if isinstance(self.processed_tokens, bool) or not isinstance(self.processed_tokens, int):
            raise TypeError("processed_tokens must be an integer")
        if self.processed_tokens < 0:
            raise ValueError("processed_tokens must be non-negative")
        if not isinstance(self.document_tags, tuple) or any(not isinstance(tag, str) for tag in self.document_tags):
            raise TypeError("document_tags must be a tuple of strings")
        if not isinstance(self.skip_semantic_links, bool):
            raise TypeError("skip_semantic_links must be a bool")
        if self.checkpoint_callback is not None and not callable(self.checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable or None")
        if self.outbox_callback is not None and not callable(self.outbox_callback):
            raise TypeError("outbox_callback must be callable or None")
        if not isinstance(self.log_buffer, tuple) or any(not isinstance(line, str) for line in self.log_buffer):
            raise TypeError("log_buffer must be a tuple of strings")
        object.__setattr__(self, "retain_params", _freeze_retain_params(self.retain_params))
        self._validate_chunk_sets()
        _validate_identity_bindings(self.contents, self.chunks, self.facts)

    def _validate_chunk_sets(self) -> None:
        def validate_indices(values: tuple[int, ...], *, field_name: str) -> set[int]:
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple of integers")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise TypeError(f"{field_name} must be a tuple of integers")
            if any(value < 0 for value in values):
                raise ValueError(f"{field_name} must contain only non-negative integers")
            if len(values) != len(set(values)):
                raise PersistenceContractError(f"{field_name} contains duplicate chunk indices")
            return set(values)

        unchanged = validate_indices(self.unchanged_chunk_indices, field_name="unchanged_chunk_indices")
        added = validate_indices(self.added_chunk_indices, field_name="added_chunk_indices")
        if not unchanged:
            raise PersistenceContractError("delta writes require at least one unchanged chunk")

        for field_name, values in (("changed_chunks", self.changed_chunks), ("removed_chunks", self.removed_chunks)):
            if not isinstance(values, tuple) or any(not isinstance(value, ExistingChunkWrite) for value in values):
                raise TypeError(f"{field_name} must be a tuple of ExistingChunkWrite values")
        changed_indices = [chunk.chunk_index for chunk in self.changed_chunks]
        removed_indices = [chunk.chunk_index for chunk in self.removed_chunks]
        if len(changed_indices) != len(set(changed_indices)):
            raise PersistenceContractError("changed_chunks contains duplicate chunk indices")
        if len(removed_indices) != len(set(removed_indices)):
            raise PersistenceContractError("removed_chunks contains duplicate chunk indices")
        affected_ids = [chunk.chunk_id for chunk in (*self.changed_chunks, *self.removed_chunks)]
        if len(affected_ids) != len(set(affected_ids)):
            raise PersistenceContractError("changed/removed chunks contain duplicate chunk IDs")

        named_sets = {
            "unchanged": unchanged,
            "changed": set(changed_indices),
            "added": added,
            "removed": set(removed_indices),
        }
        names = tuple(named_sets)
        for position, left_name in enumerate(names):
            for right_name in names[position + 1 :]:
                overlap = named_sets[left_name] & named_sets[right_name]
                if overlap:
                    raise PersistenceContractError(
                        f"{left_name}/{right_name} chunk sets overlap at indices {sorted(overlap)}"
                    )

        if not (changed_indices or added or removed_indices):
            raise PersistenceContractError("delta writes require at least one changed, added, or removed chunk")
        supplied_indices = {chunk.metadata.chunk_index for chunk in self.chunks}
        expected_indices = set(changed_indices) | added
        if supplied_indices != expected_indices:
            missing = sorted(expected_indices - supplied_indices)
            unexpected = sorted(supplied_indices - expected_indices)
            raise PersistenceContractError(
                f"delta chunk bindings do not match changed/added sets (missing={missing}, unexpected={unexpected})"
            )


RetainWriteRequest: TypeAlias = WriteWindowRequest | DeltaWriteRequest | MetadataOnlyWriteRequest


class OperationActivityPort(Protocol):
    """Lock and validate the tracked operation chain for one core transaction."""

    async def assert_active(self, connection: Any, *, bank_id: str) -> None: ...


class DocumentOwnershipPort(Protocol):
    """Own row-lock and document-hash transition semantics."""

    async def prepare_first_window(self, connection: Any, *, bank_id: str, document_id: str) -> None: ...

    async def validate_unhashed_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
    ) -> bool: ...

    async def validate_later_window(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
        expected_content_hash: str,
    ) -> bool: ...

    async def transition_content_hash(
        self,
        connection: Any,
        *,
        bank_id: str,
        document_id: str,
        expected_content_hash: str,
        new_content_hash: str,
    ) -> bool: ...


class OwnershipDisposition(StrEnum):
    OWNED = "owned"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class CoreWriteResult:
    """Result produced inside the transaction before commit is attempted."""

    ownership: OwnershipDisposition
    unit_ids_by_content: tuple[tuple[str, ...], ...] = ()
    unit_ids_by_fact_key: tuple[tuple[str, str], ...] = ()
    phase3_payload: Any = None
    post_commit_required: bool = False
    processed_tokens: int | None = None

    def __post_init__(self) -> None:
        fact_keys = [fact_key for fact_key, _unit_id in self.unit_ids_by_fact_key]
        if len(fact_keys) != len(set(fact_keys)):
            raise PersistenceContractError("unit_ids_by_fact_key contains duplicate fact keys")
        if any(not fact_key or not unit_id for fact_key, unit_id in self.unit_ids_by_fact_key):
            raise PersistenceContractError("unit_ids_by_fact_key contains an empty identity")
        if self.processed_tokens is not None:
            if isinstance(self.processed_tokens, bool) or not isinstance(self.processed_tokens, int):
                raise TypeError("processed_tokens must be an integer or None")
            if self.processed_tokens < 0:
                raise ValueError("processed_tokens must be non-negative")


class PostCommitStage(StrEnum):
    ENTITY_STATS = "entity_stats"
    DISPLAY_ENTITY_LINKS = "display_entity_links"


class PostCommitStatus(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PostCommitSkipReason(StrEnum):
    OWNERSHIP_LOST = "ownership_lost"
    NO_FACTS = "no_facts"


@dataclass(frozen=True, slots=True)
class PostCommitFailure:
    """A best-effort failure after the core transaction has committed."""

    stage: PostCommitStage
    exception: Exception


@dataclass(frozen=True, slots=True)
class PostCommitReport:
    status: PostCommitStatus
    failure: PostCommitFailure | None = None
    skip_reason: PostCommitSkipReason | None = None

    def __post_init__(self) -> None:
        if self.status is PostCommitStatus.FAILED and self.failure is None:
            raise ValueError("a failed post-commit report requires failure details")
        if self.status is not PostCommitStatus.FAILED and self.failure is not None:
            raise ValueError("only a failed post-commit report may contain failure details")
        if self.status is PostCommitStatus.SKIPPED and self.skip_reason is None:
            raise ValueError("a skipped post-commit report requires a reason")
        if self.status is not PostCommitStatus.SKIPPED and self.skip_reason is not None:
            raise ValueError("only a skipped post-commit report may contain a reason")


@dataclass(frozen=True, slots=True)
class UnitOfWorkResult:
    """Committed core result plus independently classified best-effort work."""

    core: CoreWriteResult
    post_commit: PostCommitReport


class PersistenceAdapter(Protocol):
    """Backend adapter invoked by :class:`RetainUnitOfWork`."""

    async def write_core(self, connection: Any, request: RetainWriteRequest) -> CoreWriteResult: ...

    async def flush_entity_stats(self) -> None: ...

    async def write_display_entity_links(self, request: RetainWriteRequest, phase3_payload: Any) -> None: ...


ConnectionScope: TypeAlias = Callable[[], AbstractAsyncContextManager[Any]]
AtomicCommitCallback: TypeAlias = Callable[[Any, tuple[CoreWriteResult, ...]], Awaitable[None]]
AtomicValidationCallback: TypeAlias = Callable[[tuple[CoreWriteResult, ...]], None]


@dataclass(frozen=True, slots=True)
class AtomicWriteStep:
    """One prepared core write and the adapter that owns its backend semantics."""

    adapter: PersistenceAdapter
    request: RetainWriteRequest


class AtomicWriteOwnershipLost(RuntimeError):
    """A prepared write lost ownership before the atomic batch could publish."""

    def __init__(self, window_index: int) -> None:
        self.window_index = window_index
        super().__init__(f"Atomic Retain write lost ownership at window {window_index}")


class AtomicRetainUnitOfWork:
    """Publish prepared Retain windows in one transaction.

    Core writes are applied in order on one connection. Any exception, inactive
    operation fence, ownership loss, checkpoint failure, outbox failure, or
    commit failure rolls every window back. Resolver statistics are flushed
    once after commit, followed by independent best-effort display-link work
    for each fact-bearing window.
    """

    def __init__(self, *, connection_scope: ConnectionScope) -> None:
        if not callable(connection_scope):
            raise TypeError("connection_scope must be callable")
        self._connection_scope = connection_scope

    async def execute(
        self,
        steps: Sequence[AtomicWriteStep],
        *,
        validation_callback: AtomicValidationCallback | None = None,
        commit_callback: AtomicCommitCallback | None = None,
    ) -> tuple[UnitOfWorkResult, ...]:
        prepared = tuple(steps)
        if not prepared:
            raise ValueError("Atomic Retain execution requires at least one write step")
        if any(not isinstance(step, AtomicWriteStep) for step in prepared):
            raise TypeError("steps must contain only AtomicWriteStep values")
        if validation_callback is not None and not callable(validation_callback):
            raise TypeError("validation_callback must be callable or None")
        if commit_callback is not None and not callable(commit_callback):
            raise TypeError("commit_callback must be callable or None")

        cores: list[CoreWriteResult] = []
        async with self._connection_scope() as connection:
            async with connection.transaction():
                for window_index, step in enumerate(prepared):
                    core = await step.adapter.write_core(connection, step.request)
                    if core.ownership is OwnershipDisposition.LOST:
                        raise AtomicWriteOwnershipLost(window_index)
                    cores.append(core)
                immutable_cores = tuple(cores)
                if validation_callback is not None:
                    validation_callback(immutable_cores)
                if commit_callback is not None:
                    await commit_callback(connection, immutable_cores)

        return await self._post_commit(prepared, tuple(cores))

    @staticmethod
    async def _post_commit(
        steps: tuple[AtomicWriteStep, ...],
        cores: tuple[CoreWriteResult, ...],
    ) -> tuple[UnitOfWorkResult, ...]:
        fact_window_indices = tuple(index for index, core in enumerate(cores) if core.post_commit_required)
        reports: list[PostCommitReport | None] = [
            (
                None
                if core.post_commit_required
                else PostCommitReport(
                    status=PostCommitStatus.SKIPPED,
                    skip_reason=PostCommitSkipReason.NO_FACTS,
                )
            )
            for core in cores
        ]

        if fact_window_indices:
            try:
                await steps[fact_window_indices[0]].adapter.flush_entity_stats()
            except Exception as exc:
                failure = PostCommitFailure(stage=PostCommitStage.ENTITY_STATS, exception=exc)
                for index in fact_window_indices:
                    reports[index] = PostCommitReport(
                        status=PostCommitStatus.FAILED,
                        failure=failure,
                    )
            else:
                for index in fact_window_indices:
                    try:
                        await steps[index].adapter.write_display_entity_links(
                            steps[index].request,
                            cores[index].phase3_payload,
                        )
                    except Exception as exc:
                        reports[index] = PostCommitReport(
                            status=PostCommitStatus.FAILED,
                            failure=PostCommitFailure(
                                stage=PostCommitStage.DISPLAY_ENTITY_LINKS,
                                exception=exc,
                            ),
                        )
                    else:
                        reports[index] = PostCommitReport(status=PostCommitStatus.COMPLETED)

        if any(report is None for report in reports):  # pragma: no cover - exhaustive classification invariant
            raise AssertionError("Atomic Retain post-commit report is incomplete")
        return tuple(
            UnitOfWorkResult(core=core, post_commit=report)
            for core, report in zip(cores, reports, strict=True)
            if report is not None
        )


class RetainUnitOfWork:
    """Run one semantic Retain write window."""

    def __init__(self, *, connection_scope: ConnectionScope, adapter: PersistenceAdapter) -> None:
        if not callable(connection_scope):
            raise TypeError("connection_scope must be callable")
        self._connection_scope = connection_scope
        self._adapter = adapter

    async def execute(self, request: RetainWriteRequest) -> UnitOfWorkResult:
        """Commit core writes, then classify rather than raise best-effort failures.

        Any exception raised while acquiring a connection, running the core
        transaction, invoking the outbox callback, or committing is propagated
        unchanged.  Only work that begins after a successful transaction exit is
        converted into a :class:`PostCommitFailure`.
        """

        async with self._connection_scope() as connection:
            async with connection.transaction():
                core = await self._adapter.write_core(connection, request)

        if core.ownership is OwnershipDisposition.LOST:
            return UnitOfWorkResult(
                core=core,
                post_commit=PostCommitReport(
                    status=PostCommitStatus.SKIPPED,
                    skip_reason=PostCommitSkipReason.OWNERSHIP_LOST,
                ),
            )
        if not core.post_commit_required:
            return UnitOfWorkResult(
                core=core,
                post_commit=PostCommitReport(
                    status=PostCommitStatus.SKIPPED,
                    skip_reason=PostCommitSkipReason.NO_FACTS,
                ),
            )

        try:
            await self._adapter.flush_entity_stats()
        except Exception as exc:
            return UnitOfWorkResult(
                core=core,
                post_commit=PostCommitReport(
                    status=PostCommitStatus.FAILED,
                    failure=PostCommitFailure(stage=PostCommitStage.ENTITY_STATS, exception=exc),
                ),
            )

        try:
            await self._adapter.write_display_entity_links(request, core.phase3_payload)
        except Exception as exc:
            return UnitOfWorkResult(
                core=core,
                post_commit=PostCommitReport(
                    status=PostCommitStatus.FAILED,
                    failure=PostCommitFailure(stage=PostCommitStage.DISPLAY_ENTITY_LINKS, exception=exc),
                ),
            )

        return UnitOfWorkResult(
            core=core,
            post_commit=PostCommitReport(status=PostCommitStatus.COMPLETED),
        )
