"""Tests for semantic ANN graph expansion mode switches."""

from __future__ import annotations

import pytest

from hms_api.engine.db.ops_oracle import OracleOps
from hms_api.engine.db.ops_postgresql import PostgreSQLOps
from hms_api.engine.retain import link_creation


def test_postgresql_semantic_ann_cte_shape_and_filters():
    cte = PostgreSQLOps().build_semantic_ann_cte("memory_units", 4, 5, 6)

    assert "semantic_expanded AS" in cte
    assert "mu_inner.bank_id = $6" in cte
    assert "mu_inner.fact_type = $2" in cte
    assert "mu_inner.embedding IS NOT NULL" in cte
    assert "mu_inner.id != ALL($1::uuid[])" in cte
    assert "LIMIT $5" in cte
    assert ">= $4" in cte
    assert "MAX(score) AS score" in cte
    assert "'semantic'::text AS source" in cte


def test_oracle_semantic_ann_mode_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError):
        OracleOps().build_semantic_ann_cte("memory_units", 4, 5, 6)


@pytest.mark.asyncio
async def test_write_semantic_links_switch_short_circuits():
    class ExplodingConn:
        async def execute(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("semantic link skip should not touch the database")

    count = await link_creation.create_semantic_links_batch(
        ExplodingConn(),
        "bank",
        ["00000000-0000-0000-0000-000000000001"],
        [[0.0, 0.0, 0.0]],
        write_semantic_links=False,
    )
    assert count == 0
