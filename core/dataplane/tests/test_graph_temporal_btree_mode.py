"""Tests for temporal B-tree graph expansion mode switches."""

from __future__ import annotations

import pytest

from hms_api.engine.db.ops_oracle import OracleOps
from hms_api.engine.db.ops_postgresql import PostgreSQLOps
from hms_api.engine.retain import link_creation


def test_postgresql_temporal_btree_lateral_shape():
    lateral = PostgreSQLOps().build_temporal_btree_spreading_lateral(
        "memory_units",
        "memory_links",
        limit_param_idx=5,
        bank_param_idx=6,
        fact_type_param_idx=3,
        window_seconds_param_idx=7,
        sigma_seconds_param_idx=8,
    )

    assert "mu_t.bank_id = $6" in lateral
    assert "mu_t.fact_type = $3" in lateral
    assert "mu_t.event_date >= src_mu.event_date" in lateral
    assert "mu_t.event_date < src_mu.event_date" in lateral
    assert "abs(extract(epoch from (mu_t.event_date - src_mu.event_date))) <= $7" in lateral
    assert "exp(-abs(extract(epoch from (mu_t.event_date - src_mu.event_date))) / $8)" in lateral
    assert "ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')" in lateral


def test_oracle_temporal_btree_mode_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError):
        OracleOps().build_temporal_btree_spreading_lateral(
            "memory_units",
            "memory_links",
            limit_param_idx=5,
            bank_param_idx=6,
            fact_type_param_idx=3,
            window_seconds_param_idx=7,
            sigma_seconds_param_idx=8,
        )


@pytest.mark.asyncio
async def test_write_temporal_links_switch_short_circuits():
    class ExplodingConn:
        async def execute(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("temporal link skip should not touch the database")

    count = await link_creation.create_temporal_links_batch(
        ExplodingConn(),
        "bank",
        ["00000000-0000-0000-0000-000000000001"],
        write_temporal_links=False,
    )
    assert count == 0
