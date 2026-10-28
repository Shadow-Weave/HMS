"""Add Retain affect metadata to memory units.

Revision ID: s7t8u9v0w1x2
Revises: r6s7t8u9v0w1
Create Date: 2026-09-02
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "s7t8u9v0w1x2"
down_revision: str | Sequence[str] | None = "r6s7t8u9v0w1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ADD COLUMN IF NOT EXISTS affect JSONB
    """)
    op.execute(f"""
        DO $$
        BEGIN
            ALTER TABLE {schema}memory_units
            ADD CONSTRAINT ck_memory_units_affect_object
            CHECK (affect IS NULL OR jsonb_typeof(affect) = 'object');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END
        $$
    """)


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS affect")


def _oracle_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}memory_units ADD (
            affect CLOB
            CONSTRAINT ck_mu_affect_json CHECK (affect IS NULL OR affect IS JSON)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN
                RAISE;
            END IF;
    END;
    """)


def _oracle_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}memory_units DROP COLUMN affect';
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
