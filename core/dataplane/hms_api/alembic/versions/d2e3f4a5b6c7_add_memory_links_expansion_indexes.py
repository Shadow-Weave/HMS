"""Add covering and composite indexes to speed up link expansion graph retrieval.

Two indexes target common link-expansion query-plan bottlenecks:

1. idx_memory_links_to_type_weight  (to_unit_id, link_type, weight DESC)
   The semantic incoming direction — finding facts that consider seeds as their
   nearest neighbour — currently hits an expensive BitmapAnd of two separate
   bitmap scans (to_unit_id bitmap ∩ link_type bitmap).  A composite index
   on (to_unit_id, link_type) turns this into a single index scan.

2. idx_memory_links_entity_covering  (from_unit_id) INCLUDE (to_unit_id, entity_id)
   WHERE link_type = 'entity'
   The entity co-occurrence expansion uses COUNT(DISTINCT ml.entity_id) and
   joins on ml.to_unit_id.  Without a covering index the planner must read
   heap pages to fetch entity_id and to_unit_id after the bitmap index scan.
   INCLUDE adds those two columns to the
   index leaf pages so the entire query can be served from the index (index-only
   scan), eliminating the heap reads entirely.
   A partial predicate limits the index to entity links.

Both indexes are created with CONCURRENTLY so the migration does not block
concurrent reads or writes on memory_links.  CONCURRENTLY requires running
outside a transaction block, so the migration emits an explicit COMMIT before
each statement and uses IF NOT EXISTS for idempotency.

Revision ID: d2e3f4a5b6c7
Revises: b3c4d5e6f7g8
Create Date: 2026-03-02
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "d2e3f4a5b6c7"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7g8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    # Commit the current Alembic transaction, then issue each CONCURRENTLY
    # statement in its own implicit autocommit transaction.
    # IF NOT EXISTS makes each statement idempotent if the migration is retried.

    # Index for the semantic *incoming* direction in link_expansion_retrieval.py.
    # Replaces the BitmapAnd of idx_memory_links_to_unit ∩ idx_memory_links_link_type
    # with a single composite index scan.
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_to_type_weight "
        f"ON {schema}memory_links(to_unit_id, link_type, weight DESC)"
    )

    # Covering index for entity co-occurrence expansion.
    # Enables an index-only scan: entity_id and to_unit_id are read from the
    # index leaf pages instead of the heap.
    op.execute("COMMIT")
    op.execute(
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_entity_covering "
        f"ON {schema}memory_links(from_unit_id) "
        f"INCLUDE (to_unit_id, entity_id) "
        f"WHERE link_type = 'entity'"
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity_covering")
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_to_type_weight")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
