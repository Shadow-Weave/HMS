"""Runtime operations shared by the final Retain application service.

This module contains the database-facing entity, graph, and post-commit
operations used by the single Retain pipeline.  Planning remains in the pure
domain modules, while write orchestration remains in ``persistence``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ...worker.stage import set_stage
from ..db_utils import acquire_with_retry
from ..embedding_fingerprint import embedding_model_version
from ..memory_engine import count_tokens, fq_table
from ..retain import entity_processing, fact_storage, link_creation
from ..retain.link_utils import _bulk_insert_links, compute_semantic_links_ann
from ..retain.types import (
    EntityReadPlanPhase1Result,
    Phase3Context,
    ProcessedFact,
    RetainContent,
)
from .adapters.oracle_semantic import compute_oracle_semantic_links_ann
from .persistence.backend import retain_backend_adapters

logger = logging.getLogger(__name__)

_ANN_CHUNK_SIZE = 1000
_ANN_PARALLELISM = 4
_ORACLE_IN_CHUNK_SIZE = 900


async def pre_resolve_entities(
    pool: Any,
    entity_resolver: Any,
    bank_id: str,
    contents: list[RetainContent],
    fact_keys: list[str],
    processed_facts: list[ProcessedFact],
    config: Any,
    log_buffer: list[str],
    *,
    skip_semantic_ann: bool = False,
) -> EntityReadPlanPhase1Result:
    """Build a read-only entity and semantic-link plan before the write UoW."""

    set_stage("retain.phase1.resolve")
    if len(fact_keys) != len(processed_facts):
        raise ValueError("Retain requires one stable fact key per processed fact")

    user_entities_per_content = {index: content.entities for index, content in enumerate(contents) if content.entities}
    placeholder_unit_ids = [str(index) for index in range(len(processed_facts))]
    embeddings = [fact.embedding for fact in processed_facts]
    backend_adapters = retain_backend_adapters(getattr(pool, "backend_type", "postgresql"))
    backend_type = backend_adapters.backend_type

    async with acquire_with_retry(pool) as resolve_conn, backend_adapters.planning_snapshot(resolve_conn):
        entity_read_plan = await entity_processing.plan_entities(
            entity_resolver,
            resolve_conn,
            bank_id,
            fact_keys,
            processed_facts,
            log_buffer,
            user_entities_per_content=user_entities_per_content,
            entity_labels=getattr(config, "entity_labels", None),
        )
        semantic_ann_links = []
        if not skip_semantic_ann and all(embedding is not None for embedding in embeddings):
            fact_types = [fact.fact_type for fact in processed_facts]
            if backend_type == "oracle":
                semantic_ann_links = await compute_oracle_semantic_links_ann(
                    resolve_conn,
                    bank_id,
                    placeholder_unit_ids,
                    embeddings,
                    fact_types=fact_types,
                )
            else:
                semantic_ann_links = await compute_semantic_links_ann(
                    resolve_conn,
                    bank_id,
                    placeholder_unit_ids,
                    embeddings,
                    fact_types=fact_types,
                    log_buffer=log_buffer,
                    read_only=True,
                )
        elif not skip_semantic_ann:
            log_buffer.append("  Semantic ANN precompute: skipped (missing embeddings)")

    return EntityReadPlanPhase1Result(
        entity_read_plan=entity_read_plan,
        semantic_ann_links=semantic_ann_links,
    )


def _remap_phase1_results(
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple[Any, ...]],
    unit_to_entity_ids: dict[str, list[str]],
    semantic_ann_links: list[tuple[Any, ...]],
    actual_unit_ids: list[str],
) -> tuple[list[tuple[Any, ...]], dict[str, list[str]], list[tuple[Any, ...]]]:
    """Replace read-plan placeholder unit IDs with committed database IDs."""

    placeholder_to_actual = {str(index): actual_id for index, actual_id in enumerate(actual_unit_ids)}
    remapped_entity_to_unit = [
        (
            placeholder_to_actual.get(unit_id, unit_id),
            local_index,
            fact_date,
        )
        for unit_id, local_index, fact_date in entity_to_unit
    ]
    remapped_unit_to_entity_ids = {
        placeholder_to_actual.get(placeholder_id, placeholder_id): entity_ids
        for placeholder_id, entity_ids in unit_to_entity_ids.items()
    }
    remapped_semantic = [
        (
            placeholder_to_actual.get(link[0], link[0]),
            link[1],
            link[2],
            link[3],
            link[4],
        )
        for link in semantic_ann_links
    ]
    return (
        remapped_entity_to_unit,
        remapped_unit_to_entity_ids,
        remapped_semantic,
    )


async def insert_facts_and_links(
    conn: Any,
    entity_resolver: Any,
    bank_id: str,
    contents: list[RetainContent],
    processed_facts: list[ProcessedFact],
    config: Any,
    log_buffer: list[str],
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple[Any, ...]],
    unit_to_entity_ids: dict[str, list[str]],
    semantic_ann_links: list[tuple[Any, ...]],
    *,
    skip_semantic_links: bool = False,
    outbox_callback: Any = None,
    ops: Any = None,
) -> tuple[list[list[str]], Phase3Context]:
    """Insert facts and retrieval-critical graph edges in one transaction."""

    set_stage("retain.phase2.insert_facts")
    step_start = time.time()
    unit_ids = await fact_storage.insert_facts_batch(
        conn,
        bank_id,
        processed_facts,
        ops=ops,
    )
    log_buffer.append(f"  Insert facts: {len(unit_ids)} units in {time.time() - step_start:.3f}s")

    phase3_context = Phase3Context()
    if unit_ids:
        step_start = time.time()
        (
            remapped_entity_to_unit,
            remapped_unit_to_entity_ids,
            semantic_ann_links,
        ) = _remap_phase1_results(
            resolved_entity_ids,
            entity_to_unit,
            unit_to_entity_ids,
            semantic_ann_links or [],
            unit_ids,
        )
        unit_entity_pairs = [
            (unit_id, resolved_entity_ids[index], fact_date)
            for index, (unit_id, _local_index, fact_date) in enumerate(remapped_entity_to_unit)
        ]
        await entity_resolver.link_units_to_entities_batch(
            unit_entity_pairs,
            conn=conn,
        )
        log_buffer.append(f"  Insert unit_entities: {len(unit_entity_pairs)} pairs in {time.time() - step_start:.3f}s")
        phase3_context = Phase3Context(
            unit_ids=unit_ids,
            resolved_entity_ids=resolved_entity_ids,
            entity_to_unit=remapped_entity_to_unit,
            unit_to_entity_ids=remapped_unit_to_entity_ids,
        )

        step_start = time.time()
        temporal_link_count = await link_creation.create_temporal_links_batch(
            conn,
            bank_id,
            unit_ids,
            ops=ops,
            write_temporal_links=getattr(config, "write_temporal_links", True),
        )
        log_buffer.append(f"  Temporal links: {temporal_link_count} links in {time.time() - step_start:.3f}s")

        if not getattr(config, "write_semantic_links", True):
            log_buffer.append("  Semantic links: skipped (mode=ann)")
        elif skip_semantic_links:
            log_buffer.append("  Semantic links: skipped (deferred to final ANN pass)")
        else:
            step_start = time.time()
            embeddings_for_links = [fact.embedding for fact in processed_facts]
            if not all(embedding is not None for embedding in embeddings_for_links):
                log_buffer.append("  Semantic links: skipped (missing embeddings)")
            else:
                semantic_link_count = await link_creation.create_semantic_links_batch(
                    conn,
                    bank_id,
                    unit_ids,
                    embeddings_for_links,
                    pre_computed_ann_links=semantic_ann_links,
                    ops=ops,
                    write_semantic_links=getattr(
                        config,
                        "write_semantic_links",
                        True,
                    ),
                )
                log_buffer.append(f"  Semantic links: {semantic_link_count} links in {time.time() - step_start:.3f}s")

        step_start = time.time()
        causal_link_count = await link_creation.create_causal_links_batch(
            conn,
            bank_id,
            unit_ids,
            processed_facts,
            ops=ops,
        )
        log_buffer.append(f"  Causal links: {causal_link_count} links in {time.time() - step_start:.3f}s")

    result_unit_ids: list[list[str]] = [[] for _ in contents]
    for processed_fact, unit_id in zip(processed_facts, unit_ids, strict=True):
        content_index = processed_fact.content_index
        if content_index < 0 or content_index >= len(contents):
            raise ValueError(f"Fact content index {content_index} is outside the request")
        result_unit_ids[content_index].append(unit_id)

    if outbox_callback:
        await outbox_callback(conn)
    return result_unit_ids, phase3_context


async def build_and_insert_entity_links(
    pool: Any,
    entity_resolver: Any,
    bank_id: str,
    phase3_context: Phase3Context,
    config: Any,
    log_buffer: list[str],
) -> None:
    """Build visualization-only entity links after the core transaction."""

    set_stage("retain.phase3.entity_links")
    if not getattr(config, "write_entity_links", True):
        log_buffer.append("  Entity links (viz): skipped (write_entity_links=false)")
        return
    if not phase3_context.unit_ids or not phase3_context.resolved_entity_ids:
        return

    async with acquire_with_retry(pool) as conn:
        step_start = time.time()
        entity_links = await entity_processing.build_entity_links(
            entity_resolver,
            conn,
            bank_id,
            phase3_context.unit_ids,
            phase3_context.resolved_entity_ids,
            phase3_context.entity_to_unit,
            phase3_context.unit_to_entity_ids,
            log_buffer,
            skip_unit_entities_insert=True,
            ops=pool.ops,
        )
        if entity_links:
            await entity_processing.insert_entity_links_batch(
                conn,
                entity_links,
                bank_id,
                ops=pool.ops,
            )
        log_buffer.append(f"  Entity links (viz): {len(entity_links)} links in {time.time() - step_start:.3f}s")


async def run_final_semantic_ann(
    pool: Any,
    bank_id: str,
    unit_ids: list[str],
    config: Any,
    log_buffer: list[str],
) -> None:
    """Create semantic links for all committed units in bounded ANN batches."""

    if not getattr(config, "write_semantic_links", True):
        log_buffer.append("[streaming] Final ANN: semantic links skipped")
        return
    if not unit_ids:
        return

    backend_type = retain_backend_adapters(getattr(pool, "backend_type", "postgresql")).backend_type
    load_start = time.time()
    async with acquire_with_retry(pool) as conn:
        if backend_type == "oracle":
            rows = []
            for start in range(0, len(unit_ids), _ORACLE_IN_CHUNK_SIZE):
                rows.extend(
                    await conn.fetch(
                        f"""
                        SELECT id, embedding, fact_type
                        FROM {fq_table("memory_units")}
                        WHERE bank_id = $1 AND id = ANY($2::uuid[])
                        ORDER BY id
                        """,
                        bank_id,
                        unit_ids[start : start + _ORACLE_IN_CHUNK_SIZE],
                    )
                )
        else:
            rows = await conn.fetch(
                f"""
                SELECT id::text, embedding::text, fact_type
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND id = ANY($2::uuid[])
                ORDER BY id
                """,
                bank_id,
                unit_ids,
            )
    if not rows:
        log_buffer.append("[streaming] Final ANN: no committed units found")
        return

    unit_map = {str(row["id"]): (row["embedding"], row["fact_type"]) for row in rows}
    ann_unit_ids: list[str] = []
    ann_embeddings: list[Any] = []
    ann_fact_types: list[str] = []
    for unit_id in unit_ids:
        stored = unit_map.get(unit_id)
        if stored is not None and stored[0] is not None:
            ann_unit_ids.append(unit_id)
            ann_embeddings.append(stored[0])
            ann_fact_types.append(stored[1])

    log_buffer.append(
        f"[streaming] Final ANN: loaded {len(ann_unit_ids)} units with embeddings in {time.time() - load_start:.3f}s"
    )
    if not ann_unit_ids:
        return

    chunk_count = (len(ann_unit_ids) + _ANN_CHUNK_SIZE - 1) // _ANN_CHUNK_SIZE
    semaphore = asyncio.Semaphore(_ANN_PARALLELISM)
    link_counts = [0] * chunk_count

    async def process_chunk(chunk_index: int) -> None:
        start = chunk_index * _ANN_CHUNK_SIZE
        end = min(start + _ANN_CHUNK_SIZE, len(ann_unit_ids))
        async with semaphore:
            started_at = time.time()
            async with acquire_with_retry(pool) as conn:
                if backend_type == "oracle":
                    ann_links = await compute_oracle_semantic_links_ann(
                        conn,
                        bank_id,
                        ann_unit_ids[start:end],
                        ann_embeddings[start:end],
                        fact_types=ann_fact_types[start:end],
                        top_k=20,
                    )
                else:
                    ann_links = await compute_semantic_links_ann(
                        conn,
                        bank_id,
                        ann_unit_ids[start:end],
                        ann_embeddings[start:end],
                        fact_types=ann_fact_types[start:end],
                        top_k=20,
                        log_buffer=log_buffer,
                    )
                if ann_links:
                    await _bulk_insert_links(
                        conn,
                        ann_links,
                        bank_id=bank_id,
                        ops=pool.ops,
                    )
                link_counts[chunk_index] = len(ann_links)
            logger.info(
                "Final ANN chunk %d/%d: %d links in %.3fs",
                chunk_index + 1,
                chunk_count,
                len(ann_links),
                time.time() - started_at,
            )

    await asyncio.gather(*(process_chunk(index) for index in range(chunk_count)))
    log_buffer.append(f"[streaming] Final ANN: {sum(link_counts)} total semantic links")


__all__ = [
    "build_and_insert_entity_links",
    "count_tokens",
    "embedding_model_version",
    "insert_facts_and_links",
    "pre_resolve_entities",
    "run_final_semantic_ann",
]
