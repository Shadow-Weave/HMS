"""Tests for entity IDF graph weighting SQL."""

from __future__ import annotations

import pytest

from hms_api.engine.db.ops_oracle import OracleOps
from hms_api.engine.db.ops_postgresql import PostgreSQLOps


def test_postgresql_entity_expansion_legacy_shape():
    cte = PostgreSQLOps().build_entity_expansion_cte(
        "memory_units",
        "unit_entities",
        "entities",
        per_entity_limit=200,
        entity_fanout_hard_cap=3,
        entity_idf_weighting=False,
    )

    assert "COUNT(DISTINCT se.entity_id)::float AS score" in cte
    assert "fact_count" not in cte
    assert "LIMIT 200" in cte


def test_postgresql_entity_expansion_idf_shape_and_hard_cap():
    cte = PostgreSQLOps().build_entity_expansion_cte(
        "memory_units",
        "unit_entities",
        "entities",
        per_entity_limit=25,
        entity_fanout_hard_cap=3,
        entity_idf_weighting=True,
    )

    assert "JOIN entities e ON e.id = ue.entity_id" in cte
    assert "e.fact_count <= 3" in cte
    assert "1.0 / ln(2.718281828 + GREATEST(e.fact_count, 1))" in cte
    assert "SUM(se.entity_weight)::float AS score" in cte
    assert "LIMIT 25" in cte


def test_oracle_entity_expansion_idf_shape_and_hard_cap():
    cte = OracleOps().build_entity_expansion_cte(
        "memory_units",
        "unit_entities",
        "entities",
        per_entity_limit=25,
        entity_fanout_hard_cap=3,
        entity_idf_weighting=True,
    )

    assert "JOIN entities e ON e.id = ue.entity_id" in cte
    assert "e.fact_count <= 3" in cte
    assert "1.0 / LN(2.718281828 + GREATEST(e.fact_count, 1))" in cte
    assert "SUM(se.entity_weight) AS score" in cte
    assert "FETCH FIRST 25 ROWS ONLY" in cte


@pytest.mark.asyncio
async def test_postgresql_refresh_entity_fact_counts_updates_only_requested_entities():
    class FakeConn:
        def __init__(self):
            self.calls = []

        async def execute(self, query, *args):
            self.calls.append((query, args))

    conn = FakeConn()
    entity_ids = ["00000000-0000-0000-0000-000000000001"]
    await PostgreSQLOps().refresh_entity_fact_counts(conn, "entities", "unit_entities", entity_ids)

    assert len(conn.calls) == 1
    query, args = conn.calls[0]
    assert "UPDATE entities e" in query
    assert "LEFT JOIN unit_entities ue ON ue.entity_id = ids.entity_id" in query
    assert args == (entity_ids,)
