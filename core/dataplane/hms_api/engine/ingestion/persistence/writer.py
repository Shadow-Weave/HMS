"""Durable writer for semantic ingestion records.

This module coordinates the storage and graph helpers behind one persistence
boundary.
"""

from __future__ import annotations

from typing import Any

from ...embedding_fingerprint import EmbeddingFingerprintError, ensure_bank_embedding_fingerprint
from ...retain import chunk_storage, fact_storage
from ...schema import fq_table_explicit
from .. import runtime
from ..adapters.storage_records import compute_document_hash
from ..domain import FrozenObject, thaw_json
from ..redaction import IdentifierSanitizer
from .unit_of_work import (
    CoreGraphWrite,
    CoreWriteResult,
    DeltaWriteRequest,
    DocumentOwnershipPort,
    FirstFullWriteWindow,
    LaterFullWriteWindow,
    MetadataOnlyWriteRequest,
    OperationActivityPort,
    OwnershipDisposition,
    PersistenceContractError,
    RetainWriteRequest,
    WriteWindowRequest,
)


class PersistenceWriter:
    """Coordinate semantic persistence operations."""

    def __init__(
        self,
        *,
        pool: Any,
        embeddings_model: Any,
        entity_resolver: Any,
        config: Any,
        ownership: DocumentOwnershipPort,
        operation_activity: OperationActivityPort | None = None,
        ops: Any = None,
        schema: str | None = None,
        sanitize_log_identifiers: bool = False,
    ) -> None:
        self._pool = pool
        self._embeddings_model = embeddings_model
        self._entity_resolver = entity_resolver
        self._config = config
        self._ownership = ownership
        self._operation_activity = operation_activity
        self._schema = schema
        self._sanitize_log_identifiers = sanitize_log_identifiers
        self._ops = ops if ops is not None else getattr(pool, "ops", None)
        if self._ops is None:
            raise ValueError("PersistenceWriter requires backend data-access ops")

    async def write_core(self, connection: Any, request: RetainWriteRequest) -> CoreWriteResult:
        """Write one Retain transaction through the storage helpers.

        Full and delta writes still invoke ``_insert_facts_and_links`` when
        ``facts`` is empty.  That lets an outbox callback run in the same durable
        transaction as document/chunk tracking.  Metadata-only writes invoke the
        callback directly after their final core mutation.  Post-commit entity
        work remains skipped when no facts were inserted.
        """

        if self._operation_activity is not None:
            await self._operation_activity.assert_active(connection, bank_id=request.bank_id)

        sanitizer = IdentifierSanitizer.from_values(
            enabled=self._sanitize_log_identifiers,
            values=(request.bank_id, request.document_id),
        )
        try:
            await ensure_bank_embedding_fingerprint(
                connection,
                request.bank_id,
                self._embeddings_model,
                policy=getattr(self._config, "embedding_fingerprint_policy", "strict"),
                for_write=True,
                legacy_attestation=getattr(
                    self._config,
                    "embedding_fingerprint_legacy_attestation",
                    None,
                ),
                log_sanitizer=sanitizer,
            )
        except EmbeddingFingerprintError as exc:
            if sanitizer.enabled:
                exc.args = (sanitizer.text(exc),)
            raise

        if isinstance(request, MetadataOnlyWriteRequest):
            return await self._write_metadata_only_core(connection, request)
        if isinstance(request, DeltaWriteRequest):
            return await self._write_delta_core(connection, request)
        if not isinstance(request, WriteWindowRequest):  # pragma: no cover - closed request union
            raise TypeError(f"Unsupported Retain write request: {type(request).__name__}")

        document_window = request.document_window
        if isinstance(document_window, FirstFullWriteWindow):
            if document_window.expects_unhashed_existing_document:
                owns_document = await self._ownership.validate_unhashed_window(
                    connection,
                    bank_id=request.bank_id,
                    document_id=request.document_id,
                )
                if not owns_document:
                    return CoreWriteResult(ownership=OwnershipDisposition.LOST)
            elif document_window.expected_existing_content_hash is None:
                await self._ownership.prepare_first_window(
                    connection,
                    bank_id=request.bank_id,
                    document_id=request.document_id,
                )
            else:
                owns_document = await self._ownership.validate_later_window(
                    connection,
                    bank_id=request.bank_id,
                    document_id=request.document_id,
                    expected_content_hash=document_window.expected_existing_content_hash,
                )
                if not owns_document:
                    return CoreWriteResult(ownership=OwnershipDisposition.LOST)
            if document_window.recovery:
                await fact_storage.upsert_document_metadata(
                    connection,
                    request.bank_id,
                    request.document_id,
                    document_window.combined_content,
                    dict(document_window.retain_params) if document_window.retain_params is not None else None,
                    list(document_window.document_tags),
                )
            else:
                await fact_storage.handle_document_tracking(
                    connection,
                    request.bank_id,
                    request.document_id,
                    document_window.combined_content,
                    document_window.is_first_batch,
                    dict(document_window.retain_params) if document_window.retain_params is not None else None,
                    list(document_window.document_tags),
                    ops=self._ops,
                    log_sanitizer=sanitizer,
                )
            if document_window.continuation_content_hash is not None:
                transitioned = await self._ownership.transition_content_hash(
                    connection,
                    bank_id=request.bank_id,
                    document_id=request.document_id,
                    expected_content_hash=compute_document_hash(document_window.combined_content),
                    new_content_hash=document_window.continuation_content_hash,
                )
                if not transitioned:  # pragma: no cover - row is locked in this transaction
                    raise PersistenceContractError(
                        "first FULL window could not publish its continuation ownership hash"
                    )
        elif isinstance(document_window, LaterFullWriteWindow):
            owns_document = await self._ownership.validate_later_window(
                connection,
                bank_id=request.bank_id,
                document_id=request.document_id,
                expected_content_hash=document_window.expected_content_hash,
            )
            if not owns_document:
                return CoreWriteResult(ownership=OwnershipDisposition.LOST)
            if document_window.completed_content_hash is not None:
                transitioned = await self._ownership.transition_content_hash(
                    connection,
                    bank_id=request.bank_id,
                    document_id=request.document_id,
                    expected_content_hash=document_window.expected_content_hash,
                    new_content_hash=document_window.completed_content_hash,
                )
                if not transitioned:  # pragma: no cover - validate holds the row lock
                    raise PersistenceContractError("final FULL window could not publish its completed content hash")
        else:  # pragma: no cover - WriteWindowRequest validates this boundary
            raise TypeError(f"Unsupported document window: {type(document_window).__name__}")

        graph = await self._finalize_entity_graph(connection, request)
        chunk_ids_by_index: dict[int, str] = {}
        if request.chunks:
            chunk_ids_by_index = await chunk_storage.store_chunks_batch(
                connection,
                request.bank_id,
                request.document_id,
                [chunk.metadata for chunk in request.chunks],
                ops=self._ops,
            )
        self._assert_complete_chunk_result(request, chunk_ids_by_index)

        chunk_ids_by_key = {chunk.chunk_key: chunk_ids_by_index[chunk.metadata.chunk_index] for chunk in request.chunks}
        for fact in request.facts:
            # Stable fact/chunk keys were validated before the transaction and
            # are used here instead of relying on zip/list completion order.
            fact.processed.document_id = request.document_id
            fact.processed.chunk_id = chunk_ids_by_key[fact.chunk_key]

        unit_ids_by_content, phase3_payload = await runtime.insert_facts_and_links(
            connection,
            self._entity_resolver,
            request.bank_id,
            list(request.contents),
            [fact.processed for fact in request.facts],
            self._config,
            request.log_buffer,
            resolved_entity_ids=list(graph.resolved_entity_ids),
            entity_to_unit=list(graph.entity_to_unit),
            unit_to_entity_ids={unit_id: list(entity_ids) for unit_id, entity_ids in graph.unit_to_entity_ids},
            semantic_ann_links=list(graph.semantic_ann_links),
            skip_semantic_links=request.skip_semantic_links,
            # When a strict checkpoint is present it must observe the exact
            # IDs returned by this core write before the external outbox runs.
            # Delay both callbacks until the result contract is validated.
            outbox_callback=(request.outbox_callback if request.checkpoint_callback is None else None),
            ops=self._ops,
        )
        self._assert_complete_fact_result(request, unit_ids_by_content)
        if request.checkpoint_callback is not None:
            immutable_buckets = tuple(tuple(ids) for ids in unit_ids_by_content)
            await request.checkpoint_callback(connection, immutable_buckets)
            if request.outbox_callback is not None:
                await request.outbox_callback(connection)
        return CoreWriteResult(
            ownership=OwnershipDisposition.OWNED,
            unit_ids_by_content=tuple(tuple(ids) for ids in unit_ids_by_content),
            unit_ids_by_fact_key=self._map_unit_ids_by_fact_key(request, unit_ids_by_content),
            phase3_payload=phase3_payload,
            post_commit_required=bool(request.facts),
        )

    async def _write_metadata_only_core(
        self,
        connection: Any,
        request: MetadataOnlyWriteRequest,
    ) -> CoreWriteResult:
        owns_document = await self._ownership.validate_later_window(
            connection,
            bank_id=request.bank_id,
            document_id=request.document_id,
            expected_content_hash=request.expected_content_hash,
        )
        if not owns_document:
            return CoreWriteResult(
                ownership=OwnershipDisposition.LOST,
                processed_tokens=0,
            )

        await fact_storage.upsert_document_metadata(
            connection,
            request.bank_id,
            request.document_id,
            request.combined_content,
            self._thaw_retain_params(request.retain_params),
            list(request.document_tags),
        )
        await fact_storage.update_memory_units_tags(
            connection,
            request.bank_id,
            request.document_id,
            list(request.document_tags),
        )
        if request.checkpoint_callback is not None:
            await request.checkpoint_callback(
                connection,
                tuple(() for _ in range(request.input_slot_count)),
            )
        if request.outbox_callback is not None:
            await request.outbox_callback(connection)
        return CoreWriteResult(
            ownership=OwnershipDisposition.OWNED,
            unit_ids_by_content=tuple(() for _ in range(request.input_slot_count)),
            processed_tokens=0,
        )

    async def _write_delta_core(self, connection: Any, request: DeltaWriteRequest) -> CoreWriteResult:
        owns_document = await self._ownership.validate_later_window(
            connection,
            bank_id=request.bank_id,
            document_id=request.document_id,
            expected_content_hash=request.expected_content_hash,
        )
        if not owns_document:
            return CoreWriteResult(
                ownership=OwnershipDisposition.LOST,
                processed_tokens=request.processed_tokens,
            )

        graph = await self._finalize_entity_graph(connection, request)
        await fact_storage.upsert_document_metadata(
            connection,
            request.bank_id,
            request.document_id,
            request.combined_content,
            self._thaw_retain_params(request.retain_params),
            list(request.document_tags),
        )

        chunk_ids_to_delete = [chunk.chunk_id for chunk in (*request.changed_chunks, *request.removed_chunks)]
        await chunk_storage.delete_chunks_by_ids(connection, chunk_ids_to_delete)
        await fact_storage.update_memory_units_tags(
            connection,
            request.bank_id,
            request.document_id,
            list(request.document_tags),
        )

        chunk_ids_by_index: dict[int, str] = {}
        if request.chunks:
            chunk_ids_by_index = await chunk_storage.store_chunks_batch(
                connection,
                request.bank_id,
                request.document_id,
                [chunk.metadata for chunk in request.chunks],
                ops=self._ops,
            )
        self._assert_complete_chunk_result(request, chunk_ids_by_index)

        chunk_ids_by_key = {chunk.chunk_key: chunk_ids_by_index[chunk.metadata.chunk_index] for chunk in request.chunks}
        for fact in request.facts:
            fact.processed.document_id = request.document_id
            fact.processed.chunk_id = chunk_ids_by_key[fact.chunk_key]

        log_buffer = list(request.log_buffer)
        unit_ids_by_content, phase3_payload = await runtime.insert_facts_and_links(
            connection,
            self._entity_resolver,
            request.bank_id,
            list(request.contents),
            [fact.processed for fact in request.facts],
            self._config,
            log_buffer,
            resolved_entity_ids=list(graph.resolved_entity_ids),
            entity_to_unit=list(graph.entity_to_unit),
            unit_to_entity_ids={unit_id: list(entity_ids) for unit_id, entity_ids in graph.unit_to_entity_ids},
            semantic_ann_links=list(graph.semantic_ann_links),
            skip_semantic_links=request.skip_semantic_links,
            outbox_callback=(request.outbox_callback if request.checkpoint_callback is None else None),
            ops=self._ops,
        )
        self._assert_complete_fact_result(request, unit_ids_by_content)
        if request.checkpoint_callback is not None:
            immutable_buckets = tuple(tuple(ids) for ids in unit_ids_by_content)
            await request.checkpoint_callback(connection, immutable_buckets)
            if request.outbox_callback is not None:
                await request.outbox_callback(connection)
        return CoreWriteResult(
            ownership=OwnershipDisposition.OWNED,
            unit_ids_by_content=tuple(tuple(ids) for ids in unit_ids_by_content),
            unit_ids_by_fact_key=self._map_unit_ids_by_fact_key(request, unit_ids_by_content),
            phase3_payload=phase3_payload,
            post_commit_required=bool(request.facts),
            processed_tokens=request.processed_tokens,
        )

    async def flush_entity_stats(self) -> None:
        """Flush resolver statistics after, and never inside, the core commit."""

        await self._entity_resolver.flush_pending_stats()

    async def _finalize_entity_graph(
        self,
        connection: Any,
        request: WriteWindowRequest | DeltaWriteRequest,
    ) -> CoreGraphWrite:
        """Finalize missing canonical rows after ownership, inside the core UoW."""

        graph = request.graph
        if graph.entity_read_plan is None:
            return graph
        finalized = await self._entity_resolver.finalize_entity_read_plan(
            connection,
            request.bank_id,
            graph.entity_read_plan,
            entities_table=fq_table_explicit("entities", self._schema),
        )
        fact_position = {fact.fact_key: str(index) for index, fact in enumerate(request.facts)}
        planned_unit_keys = {occurrence.unit_key for occurrence in graph.entity_read_plan.occurrences}
        if not planned_unit_keys <= set(fact_position):
            unexpected = sorted(planned_unit_keys - set(fact_position))
            raise PersistenceContractError(f"entity read plan references unknown stable fact keys: {unexpected!r}")
        entity_to_unit = tuple(
            (fact_position[unit_key], local_index, event_date)
            for unit_key, local_index, event_date in finalized.entity_to_unit
        )
        unit_to_entity_ids = tuple(
            (fact_position[unit_key], tuple(entity_ids)) for unit_key, entity_ids in finalized.unit_to_entity_ids
        )
        return CoreGraphWrite(
            resolved_entity_ids=tuple(finalized.resolved_entity_ids),
            entity_to_unit=entity_to_unit,
            unit_to_entity_ids=unit_to_entity_ids,
            semantic_ann_links=graph.semantic_ann_links,
        )

    async def write_display_entity_links(self, request: RetainWriteRequest, phase3_payload: Any) -> None:
        """Delegate best-effort Phase 3 display-link work after commit."""

        if isinstance(request, MetadataOnlyWriteRequest):  # pragma: no cover - metadata never requests phase 3
            raise PersistenceContractError("metadata-only writes cannot require display entity links")
        log_buffer = request.log_buffer if isinstance(request, WriteWindowRequest) else list(request.log_buffer)

        await runtime.build_and_insert_entity_links(
            self._pool,
            self._entity_resolver,
            request.bank_id,
            phase3_payload,
            self._config,
            log_buffer,
        )

    @staticmethod
    def _assert_complete_chunk_result(
        request: WriteWindowRequest | DeltaWriteRequest,
        chunk_ids_by_index: dict[int, str],
    ) -> None:
        expected_indices = {chunk.metadata.chunk_index for chunk in request.chunks}
        actual_indices = set(chunk_ids_by_index)
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            unexpected = sorted(actual_indices - expected_indices)
            raise PersistenceContractError(
                f"chunk upsert returned an invalid key set (missing={missing}, unexpected={unexpected})"
            )
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids_by_index.values()):
            raise PersistenceContractError("chunk upsert returned an empty or non-string chunk_id")

    @staticmethod
    def _assert_complete_fact_result(
        request: WriteWindowRequest | DeltaWriteRequest,
        unit_ids_by_content: Any,
    ) -> None:
        if not isinstance(unit_ids_by_content, list) or len(unit_ids_by_content) != len(request.contents):
            raise PersistenceContractError("core write did not return one unit-id bucket per content item")
        if any(not isinstance(bucket, list) for bucket in unit_ids_by_content):
            raise PersistenceContractError("core write returned a non-list unit-id bucket")
        flattened = [unit_id for bucket in unit_ids_by_content for unit_id in bucket]
        if len(flattened) != len(request.facts):
            raise PersistenceContractError(
                f"core write returned {len(flattened)} unit IDs for {len(request.facts)} fact bindings"
            )
        expected_per_content = [0] * len(request.contents)
        for fact in request.facts:
            expected_per_content[fact.processed.content_index] += 1
        actual_per_content = [len(bucket) for bucket in unit_ids_by_content]
        if actual_per_content != expected_per_content:
            raise PersistenceContractError(
                "core write returned invalid per-content fact cardinality "
                f"(expected={expected_per_content}, actual={actual_per_content})"
            )
        if any(not isinstance(unit_id, str) or not unit_id for unit_id in flattened):
            raise PersistenceContractError("core write returned an empty or non-string unit_id")

    @staticmethod
    def _map_unit_ids_by_fact_key(
        request: WriteWindowRequest | DeltaWriteRequest,
        unit_ids_by_content: list[list[str]],
    ) -> tuple[tuple[str, str], ...]:
        mappings: list[tuple[str, str]] = []
        for content_index, unit_ids in enumerate(unit_ids_by_content):
            facts = [fact for fact in request.facts if fact.processed.content_index == content_index]
            mappings.extend((fact.fact_key, unit_id) for fact, unit_id in zip(facts, unit_ids, strict=True))
        return tuple(mappings)

    @staticmethod
    def _thaw_retain_params(retain_params: FrozenObject | None) -> dict[str, Any] | None:
        if retain_params is None:
            return None
        value = thaw_json(retain_params)
        if not isinstance(value, dict):  # pragma: no cover - request normalization guarantees an object
            raise PersistenceContractError("retain_params did not thaw to an object")
        return value
