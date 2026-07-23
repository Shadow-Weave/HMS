"""Add a dedicated embedding fingerprint to banks.

Revision ID: q5r6s7t8u9v0
Revises: p4q5r6s7t8u9
Create Date: 2026-07-18

Existing rows intentionally remain NULL.  The legacy projection migration
wrote ``embedding.v = 1`` and therefore cannot establish which model produced
the vectors already present in a bank.
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "q5r6s7t8u9v0"
down_revision: str | Sequence[str] | None = "p4q5r6s7t8u9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}banks ADD COLUMN IF NOT EXISTS embedding_fingerprint jsonb")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS embedding_fingerprint")


def _oracle_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}banks ADD (
            embedding_fingerprint CLOB
            CONSTRAINT banks_embedding_fp_json_ck CHECK (embedding_fingerprint IS JSON)
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
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}banks DROP COLUMN embedding_fingerprint';
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
