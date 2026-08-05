"""Fact extractor adapter for the ingestion pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ...response_models import TokenUsage
from ..domain import ChunkPlan, ContentItem, FrozenJson, freeze_json, thaw_json
from .models import CausalFactRelation, FactCandidate, compute_fact_key
from .passthrough import build_content_position_map, extract_passthrough
from .ports import (
    BatchExtractionUnsupportedError,
    ChunkFactCount,
    ExtractionContractError,
    ExtractionMode,
    ExtractionModeMismatchError,
    ExtractionRequest,
    ExtractionResult,
)

ExtractionPrimitive = Callable[..., Awaitable[tuple[Any, Any, Any]]]
BatchCheckpointClearer = Callable[[], Awaitable[None]]


def _frozen_metadata(value: Any, *, fact_index: int) -> FrozenJson:
    if not isinstance(value, Mapping):
        raise ExtractionContractError(f"fact[{fact_index}].metadata must be a mapping")
    try:
        return freeze_json(deepcopy(dict(value)))
    except TypeError as exc:
        raise ExtractionContractError(f"fact[{fact_index}].metadata is not JSON-compatible") from exc


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExtractionContractError(f"{field_name} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise ExtractionContractError(f"{field_name} must contain only strings")
    return result


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ExtractionContractError(f"{field_name} must be a string or None")
    return value


def _aware_datetime(value: Any, *, field_name: str) -> datetime | None:
    """Interpret naive datetimes as UTC while preserving aware time zones."""

    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ExtractionContractError(f"{field_name} must be a datetime or None")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value


def _observation_scopes(value: Any, *, field_name: str):
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"per_tag", "combined", "all_combinations"}:
            raise ExtractionContractError(f"{field_name} contains an unsupported named scope")
        return value
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ExtractionContractError(f"{field_name} must be a named scope or nested string sequences")

    scopes: list[tuple[str, ...]] = []
    for scope_index, scope in enumerate(value):
        if isinstance(scope, (str, bytes)) or not isinstance(scope, Sequence):
            raise ExtractionContractError(f"{field_name}[{scope_index}] must be a sequence of strings")
        frozen_scope = tuple(scope)
        if any(not isinstance(tag, str) for tag in frozen_scope):
            raise ExtractionContractError(f"{field_name}[{scope_index}] must contain only strings")
        scopes.append(frozen_scope)
    return tuple(scopes)


def _storage_content(item: ContentItem):
    metadata = thaw_json(item.metadata)
    if not isinstance(metadata, dict):
        raise ExtractionContractError("ContentItem.metadata must thaw to an object")

    entities = []
    for index, frozen_entity in enumerate(item.entities):
        entity = thaw_json(frozen_entity)
        if not isinstance(entity, dict):
            raise ExtractionContractError(f"ContentItem.entities[{index}] must thaw to an object")
        entities.append(entity)

    if isinstance(item.observation_scopes, tuple):
        observation_scopes = [list(scope) for scope in item.observation_scopes]
    else:
        observation_scopes = item.observation_scopes

    from ...retain.types import RetainContent

    return RetainContent(
        content=item.content,
        context=item.context,
        event_date=item.event_date.value,
        metadata=metadata,
        entities=entities,
        tags=list(item.tags),
        observation_scopes=observation_scopes,
    )


def _validate_planned_chunks(
    chunks: tuple[ChunkPlan, ...],
    items: tuple[ContentItem, ...],
) -> dict[int | None, int]:
    try:
        positions = build_content_position_map(items)
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(str(exc)) from exc

    if bool(items) != bool(chunks):
        raise ExtractionContractError("items and planned chunks must either both be empty or both be non-empty")

    seen_keys: set[str] = set()
    seen_indices: set[int] = set()
    planned_sources: set[int | None] = set()
    for position, chunk in enumerate(chunks):
        if chunk.chunk_key in seen_keys:
            raise ExtractionContractError(f"planned chunk[{position}] duplicates chunk_key={chunk.chunk_key!r}")
        if chunk.global_index in seen_indices:
            raise ExtractionContractError(f"planned chunk[{position}] duplicates global_index={chunk.global_index}")
        if chunk.source_index not in positions:
            raise ExtractionContractError(
                f"planned chunk[{position}] references unknown source_index={chunk.source_index!r}"
            )
        seen_keys.add(chunk.chunk_key)
        seen_indices.add(chunk.global_index)
        planned_sources.add(chunk.source_index)

    missing_sources = set(positions) - planned_sources
    if missing_sources:
        raise ExtractionContractError(f"No planned chunk for source indices: {sorted(missing_sources, key=str)!r}")
    return positions


def _validate_extracted_chunks(
    extracted_chunks: Sequence[Any],
    planned_chunks: tuple[ChunkPlan, ...],
    content_positions: Mapping[int | None, int],
) -> tuple[int, ...]:
    if len(extracted_chunks) != len(planned_chunks):
        raise ExtractionContractError(
            f"Extractor returned {len(extracted_chunks)} chunks for {len(planned_chunks)} planned chunks"
        )

    declared_counts: list[int] = []
    for position, (extracted_chunk, planned_chunk) in enumerate(zip(extracted_chunks, planned_chunks, strict=True)):
        chunk_index = getattr(extracted_chunk, "chunk_index", None)
        content_index = getattr(extracted_chunk, "content_index", None)
        fact_count = getattr(extracted_chunk, "fact_count", None)
        chunk_text = getattr(extracted_chunk, "chunk_text", None)

        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index != position:
            raise ExtractionContractError(
                f"chunk[{position}].chunk_index must equal its zero-based position; got {chunk_index!r}"
            )
        expected_content_index = content_positions[planned_chunk.source_index]
        if (
            isinstance(content_index, bool)
            or not isinstance(content_index, int)
            or content_index != expected_content_index
        ):
            raise ExtractionContractError(
                f"chunk[{position}].content_index={content_index!r} does not match "
                f"source_index={planned_chunk.source_index!r} at content position {expected_content_index}"
            )
        if chunk_text != planned_chunk.text:
            raise ExtractionContractError(f"chunk[{position}] text does not match its planned chunk")
        if isinstance(fact_count, bool) or not isinstance(fact_count, int) or fact_count < 0:
            raise ExtractionContractError(f"chunk[{position}].fact_count must be a non-negative integer")
        declared_counts.append(fact_count)
    return tuple(declared_counts)


class FactExtractorAdapter:
    """Expose the configured extractor through the ingestion extraction port."""

    def __init__(
        self,
        *,
        llm_config: Any,
        config: Any,
        agent_name: str,
        pool: Any = None,
        operation_id: str | None = None,
        schema: str | None = None,
        batch_checkpoint_clearer: BatchCheckpointClearer | None = None,
        sync_primitive: ExtractionPrimitive | None = None,
        batch_primitive: ExtractionPrimitive | None = None,
    ) -> None:
        if sync_primitive is None or batch_primitive is None:
            from ...retain import fact_extraction

            sync_primitive = sync_primitive or fact_extraction.extract_facts_from_contents
            batch_primitive = batch_primitive or fact_extraction.extract_facts_from_contents_batch_api
        self._llm_config = llm_config
        self._config = config
        self._agent_name = agent_name
        self._pool = pool
        self._operation_id = operation_id
        self._schema = schema
        self._batch_checkpoint_clearer = batch_checkpoint_clearer
        self._sync_primitive = sync_primitive
        self._batch_primitive = batch_primitive

    def _sync_fallback_config(self) -> Any:
        """Disable Batch API on a shallow config copy for one safe fallback.

        The batch entry point falls back by calling the sync entry point
        with ``retain_batch_enabled`` still true, which immediately routes back
        to Batch API and recurses forever. This adapter makes the one-shot
        boundary explicit and never mutates the resolved bank configuration
        shared by the request.
        """

        fallback = copy(self._config)
        setattr(fallback, "retain_batch_enabled", False)
        return fallback

    @staticmethod
    def _boundary_preserving_config(request: ExtractionRequest, config: Any) -> Any:
        """Prevent the extraction primitive from re-splitting planned chunks.

        A pre-chunked layout represents every planned chunk as one temporary
        content item. The extraction primitive still applies
        ``retain_chunk_size`` to each item, so a semantic segment containing a
        complete oversized exchange could otherwise be divided between its
        user and assistant turns. Raising the limit on a request-local shallow
        copy preserves the planner boundary without mutating shared bank
        configuration. Output-overflow recovery inside the primitive remains
        available because it operates after this initial split.
        """

        if not request.preserve_chunk_boundaries or not request.items:
            return config

        required_chunk_size = max(len(item.content) for item in request.items)
        configured_chunk_size = getattr(config, "retain_chunk_size", None)
        if (
            isinstance(configured_chunk_size, int)
            and not isinstance(configured_chunk_size, bool)
            and configured_chunk_size >= required_chunk_size
        ):
            return config

        boundary_config = copy(config)
        setattr(boundary_config, "retain_chunk_size", required_chunk_size)
        return boundary_config

    async def _batch_primitive_if_supported(
        self,
        mode: ExtractionMode,
    ) -> tuple[ExtractionPrimitive, Any]:
        if mode is ExtractionMode.VERBATIM:
            # The Batch result parser requires ``what`` while verbatim
            # deliberately omits it.  Running the established sync primitive
            # is lossless and avoids silently producing zero facts.
            return self._sync_primitive, self._sync_fallback_config()

        provider = getattr(self._llm_config, "_provider_impl", None)
        supports_batch_api = getattr(provider, "supports_batch_api", None)
        if not callable(supports_batch_api):
            return self._sync_primitive, self._sync_fallback_config()
        try:
            supported = await supports_batch_api()
        except Exception as exc:
            raise BatchExtractionUnsupportedError("Failed to determine provider Batch API capability") from exc
        if supported is not True:
            return self._sync_primitive, self._sync_fallback_config()
        return self._batch_primitive, self._config

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        configured_mode = getattr(self._config, "retain_extraction_mode", None)
        if configured_mode != request.policy.mode.value:
            raise ExtractionModeMismatchError(
                f"Extraction policy mode={request.policy.mode.value!r} does not match "
                f"resolved config mode={configured_mode!r}"
            )

        content_positions = _validate_planned_chunks(request.chunks, request.items)
        if request.policy.mode is ExtractionMode.CHUNKS:
            candidates = extract_passthrough(
                request.chunks,
                request.items,
                fact_type_override=request.policy.fact_type_override,
            )
            return ExtractionResult(
                candidates=candidates,
                chunk_fact_counts=tuple(ChunkFactCount(chunk.chunk_key, 1) for chunk in request.chunks),
                usage=TokenUsage(),
            )

        primitive = self._sync_primitive
        primitive_config = self._config
        if getattr(self._config, "retain_batch_enabled", False):
            primitive, primitive_config = await self._batch_primitive_if_supported(request.policy.mode)
        primitive_config = self._boundary_preserving_config(request, primitive_config)

        storage_contents = [_storage_content(item) for item in request.items]
        primitive_result = await primitive(
            contents=storage_contents,
            llm_config=self._llm_config,
            agent_name=self._agent_name,
            config=primitive_config,
            pool=self._pool,
            operation_id=self._operation_id,
            schema=self._schema,
        )
        if not isinstance(primitive_result, tuple) or len(primitive_result) != 3:
            raise ExtractionContractError("Extractor primitive must return a three-tuple of facts, chunks, and usage")

        extracted_facts, extracted_chunks, usage = primitive_result
        if isinstance(extracted_facts, (str, bytes)) or not isinstance(extracted_facts, Sequence):
            raise ExtractionContractError("Extractor facts must be a sequence")
        if isinstance(extracted_chunks, (str, bytes)) or not isinstance(extracted_chunks, Sequence):
            raise ExtractionContractError("Extractor chunks must be a sequence")
        if not isinstance(usage, TokenUsage):
            raise ExtractionContractError("Extractor usage must be a TokenUsage")

        declared_counts = _validate_extracted_chunks(extracted_chunks, request.chunks, content_positions)
        if request.policy.mode is ExtractionMode.VERBATIM:
            # Sync verbatim collapses multiple metadata candidates to one fact
            # per non-empty chunk but leaves the pre-collapse ChunkMetadata
            # count unchanged.  Normalize that known compatibility artifact at
            # this boundary; all other modes retain strict count equality.
            declared_counts = tuple(1 if count else 0 for count in declared_counts)
        candidates, actual_counts = self._convert_facts(
            tuple(extracted_facts),
            tuple(extracted_chunks),
            request,
            content_positions,
        )
        if actual_counts != declared_counts:
            raise ExtractionContractError(
                f"Chunk fact-count mismatch: metadata={declared_counts!r}, actual={actual_counts!r}"
            )

        causal_relations = self._convert_causal_relations(tuple(extracted_facts), candidates)
        relations_by_source: dict[str, list[CausalFactRelation]] = {}
        for relation in causal_relations:
            relations_by_source.setdefault(relation.source_fact_key, []).append(relation)
        candidates = tuple(
            replace(candidate, causal_relations=tuple(relations_by_source.get(candidate.fact_key, ())))
            for candidate in candidates
        )
        result = ExtractionResult(
            candidates=candidates,
            chunk_fact_counts=tuple(
                ChunkFactCount(chunk.chunk_key, actual_counts[position])
                for position, chunk in enumerate(request.chunks)
            ),
            usage=usage,
            causal_relations=causal_relations,
        )
        if primitive is self._batch_primitive and self._batch_checkpoint_clearer is not None:
            # A single async operation may contain multiple documents/windows.
            # Retire this completed provider job before the next extraction so
            # it cannot accidentally resume results belonging to another
            # chunk set.  Crashes while polling still retain the checkpoint.
            await self._batch_checkpoint_clearer()
        return result

    def _convert_facts(
        self,
        extracted_facts: tuple[Any, ...],
        extracted_chunks: tuple[Any, ...],
        request: ExtractionRequest,
        content_positions: Mapping[int | None, int],
    ) -> tuple[tuple[FactCandidate, ...], tuple[int, ...]]:
        actual_counts = [0] * len(request.chunks)
        per_chunk_ordinals: Counter[int] = Counter()
        candidates: list[FactCandidate] = []
        seen_fact_keys: set[str] = set()

        for fact_index, fact in enumerate(extracted_facts):
            chunk_index = getattr(fact, "chunk_index", None)
            content_index = getattr(fact, "content_index", None)
            if (
                isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or not 0 <= chunk_index < len(request.chunks)
            ):
                raise ExtractionContractError(
                    f"fact[{fact_index}].chunk_index={chunk_index!r} is outside the returned chunk range"
                )

            planned_chunk = request.chunks[chunk_index]
            expected_content_index = content_positions[planned_chunk.source_index]
            if (
                isinstance(content_index, bool)
                or not isinstance(content_index, int)
                or content_index != expected_content_index
            ):
                raise ExtractionContractError(
                    f"fact[{fact_index}].content_index={content_index!r} does not match its chunk content position "
                    f"{expected_content_index}"
                )
            if getattr(extracted_chunks[chunk_index], "content_index", None) != content_index:
                raise ExtractionContractError(f"fact[{fact_index}] and chunk metadata disagree on content_index")

            text = getattr(fact, "fact_text", None)
            raw_fact_type = getattr(fact, "fact_type", None)
            context = getattr(fact, "context", None)
            if not isinstance(text, str):
                raise ExtractionContractError(f"fact[{fact_index}].fact_text must be a string")
            if not isinstance(raw_fact_type, str) or not raw_fact_type:
                raise ExtractionContractError(f"fact[{fact_index}].fact_type must be a non-empty string")
            if not isinstance(context, str):
                raise ExtractionContractError(f"fact[{fact_index}].context must be a string")

            fact_type = request.policy.fact_type_override or raw_fact_type
            extractor_local_index = per_chunk_ordinals[chunk_index]
            per_chunk_ordinals[chunk_index] += 1
            fact_key = compute_fact_key(
                chunk_key=planned_chunk.chunk_key,
                source_index=planned_chunk.source_index,
                global_index=planned_chunk.global_index,
                extractor_local_index=extractor_local_index,
                text=text,
                fact_type=fact_type,
            )
            if fact_key in seen_fact_keys:
                raise ExtractionContractError(f"fact[{fact_index}] produced a duplicate stable fact key")
            seen_fact_keys.add(fact_key)

            item = request.items[content_index]
            try:
                candidate = FactCandidate(
                    fact_key=fact_key,
                    chunk_key=planned_chunk.chunk_key,
                    source_index=planned_chunk.source_index,
                    global_index=planned_chunk.global_index,
                    extractor_local_index=extractor_local_index,
                    text=text,
                    fact_type=fact_type,
                    context=context,
                    where=_optional_string(
                        getattr(fact, "where", None),
                        field_name=f"fact[{fact_index}].where",
                    ),
                    occurred_start=_aware_datetime(
                        getattr(fact, "occurred_start", None),
                        field_name=f"fact[{fact_index}].occurred_start",
                    ),
                    occurred_end=_aware_datetime(
                        getattr(fact, "occurred_end", None),
                        field_name=f"fact[{fact_index}].occurred_end",
                    ),
                    mentioned_at=_aware_datetime(
                        getattr(fact, "mentioned_at", None),
                        field_name=f"fact[{fact_index}].mentioned_at",
                    ),
                    metadata=_frozen_metadata(getattr(fact, "metadata", None), fact_index=fact_index),
                    declared_entities=item.entities,
                    tags=_string_tuple(getattr(fact, "tags", None), field_name=f"fact[{fact_index}].tags"),
                    observation_scopes=_observation_scopes(
                        getattr(fact, "observation_scopes", None),
                        field_name=f"fact[{fact_index}].observation_scopes",
                    ),
                    entity_mentions=_string_tuple(
                        getattr(fact, "entities", None),
                        field_name=f"fact[{fact_index}].entities",
                    ),
                    causal_relations=(),
                )
            except ExtractionContractError:
                raise
            except (TypeError, ValueError) as exc:
                raise ExtractionContractError(f"fact[{fact_index}] violates FactCandidate invariants: {exc}") from exc

            candidates.append(candidate)
            actual_counts[chunk_index] += 1

        return tuple(candidates), tuple(actual_counts)

    @staticmethod
    def _convert_causal_relations(
        extracted_facts: tuple[Any, ...],
        candidates: tuple[FactCandidate, ...],
    ) -> tuple[CausalFactRelation, ...]:
        relations: list[CausalFactRelation] = []
        for source_index, fact in enumerate(extracted_facts):
            raw_relations = getattr(fact, "causal_relations", None) or ()
            if isinstance(raw_relations, (str, bytes)) or not isinstance(raw_relations, Sequence):
                raise ExtractionContractError(f"fact[{source_index}].causal_relations must be a sequence")
            for relation_index, relation in enumerate(raw_relations):
                target_index = getattr(relation, "target_fact_index", None)
                relation_type = getattr(relation, "relation_type", None)
                if (
                    isinstance(target_index, bool)
                    or not isinstance(target_index, int)
                    or not 0 <= target_index < source_index
                ):
                    raise ExtractionContractError(
                        f"fact[{source_index}].causal_relations[{relation_index}] target index "
                        f"must reference an earlier fact; got {target_index!r}"
                    )
                if relation_type != "caused_by":
                    raise ExtractionContractError(
                        f"fact[{source_index}].causal_relations[{relation_index}] has unsupported relation_type"
                    )
                relations.append(
                    CausalFactRelation(
                        source_fact_key=candidates[source_index].fact_key,
                        target_fact_key=candidates[target_index].fact_key,
                        relation_type=relation_type,
                    )
                )
        return tuple(relations)
