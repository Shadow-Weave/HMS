"""Oracle semantic-neighbor reads used while retaining memories."""

from __future__ import annotations

import json
from array import array
from collections.abc import Sequence
from typing import Any

from ...memory_engine import fq_table


def _vector_bind(embedding: Any) -> array:
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError as exc:
            raise ValueError("Oracle semantic planning received an invalid vector string") from exc
    if not isinstance(embedding, Sequence):
        raise TypeError("Oracle semantic planning requires a vector sequence")
    if not embedding:
        raise ValueError("Oracle semantic planning requires a non-empty vector")
    try:
        return array("f", (float(value) for value in embedding))
    except (TypeError, ValueError) as exc:
        raise ValueError("Oracle semantic planning requires numeric vector values") from exc


async def compute_oracle_semantic_links_ann(
    connection: Any,
    bank_id: str,
    unit_ids: Sequence[str],
    embeddings: Sequence[Any],
    *,
    fact_types: Sequence[str] | None = None,
    top_k: int = 50,
    threshold: float = 0.7,
) -> list[tuple[Any, ...]]:
    """Find Oracle VECTOR neighbors without PostgreSQL temp tables or arrays."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not unit_ids or not embeddings:
        return []
    if len(unit_ids) != len(embeddings):
        raise ValueError("Oracle semantic planning requires one embedding per unit")
    if fact_types is None:
        fact_types = ("world",) * len(unit_ids)
    if len(fact_types) != len(unit_ids):
        raise ValueError("Oracle semantic planning requires one fact type per unit")

    links: list[tuple[Any, ...]] = []
    for unit_id, embedding, fact_type in zip(unit_ids, embeddings, fact_types, strict=True):
        vector = _vector_bind(embedding)
        rows = await connection.fetch(
            f"""
            SELECT id AS to_id,
                   1 - VECTOR_DISTANCE(embedding, $3, COSINE) AS similarity
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND fact_type = $2
              AND embedding IS NOT NULL
            ORDER BY VECTOR_DISTANCE(embedding, $3, COSINE)
            FETCH FIRST {top_k} ROWS ONLY
            """,
            bank_id,
            fact_type,
            vector,
        )
        for row in rows:
            similarity = float(min(1.0, max(0.0, row["similarity"])))
            if similarity >= threshold:
                links.append((unit_id, str(row["to_id"]), "semantic", similarity, None))
    return links


__all__ = ["compute_oracle_semantic_links_ann"]
