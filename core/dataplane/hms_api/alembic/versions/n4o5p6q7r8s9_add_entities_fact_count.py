"""Add fact_count to entities for graph IDF weighting

Revision ID: n4o5p6q7r8s9
Revises: m3rg3h3ad5f6
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "n4o5p6q7r8s9"
down_revision: str | Sequence[str] | None = "m3rg3h3ad5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}entities ADD COLUMN IF NOT EXISTS fact_count integer NOT NULL DEFAULT 0")
    op.execute(f"""
        UPDATE {schema}entities e
        SET fact_count = counts.fact_count
        FROM (
            SELECT e2.id, COUNT(ue.unit_id)::int AS fact_count
            FROM {schema}entities e2
            LEFT JOIN {schema}unit_entities ue ON ue.entity_id = e2.id
            GROUP BY e2.id
        ) counts
        WHERE e.id = counts.id
    """)


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}entities DROP COLUMN IF EXISTS fact_count")


def _oracle_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}entities ADD (fact_count NUMBER(10) DEFAULT 0 NOT NULL)';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN
                RAISE;
            END IF;
    END;
    """)
    op.execute(f"""
        UPDATE {schema}entities e
        SET fact_count = (
            SELECT COUNT(*)
            FROM {schema}unit_entities ue
            WHERE ue.entity_id = e.id
        )
    """)


def _oracle_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}entities DROP COLUMN fact_count';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -904 THEN
                RAISE;
            END IF;
    END;
    """)


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
