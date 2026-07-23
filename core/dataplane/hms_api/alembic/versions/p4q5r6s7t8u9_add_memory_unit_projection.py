"""Add projection manifest to memory_units

Revision ID: p4q5r6s7t8u9
Revises: n4o5p6q7r8s9
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "p4q5r6s7t8u9"
down_revision: str | Sequence[str] | None = "n4o5p6q7r8s9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS projection jsonb NOT NULL DEFAULT '{{}}'::jsonb"
    )
    op.execute(f"""
        UPDATE {schema}memory_units mu
        SET projection = jsonb_build_object(
            'embedding', jsonb_build_object('v', 1, 'ok', mu.embedding IS NOT NULL),
            'tsvector', jsonb_build_object('v', 1, 'ok', true),
            'temporal', jsonb_build_object(
                'v', 1,
                'grade',
                CASE
                    WHEN mu.occurred_start IS NOT NULL OR mu.event_date IS NOT NULL THEN 'resolved'
                    ELSE 'unresolved'
                END
            ),
            'entities', jsonb_build_object(
                'v', 1,
                'ok', EXISTS (
                    SELECT 1 FROM {schema}unit_entities ue WHERE ue.unit_id = mu.id
                )
            ),
            'extraction', jsonb_build_object('v', 'legacy')
        )
        WHERE mu.projection = '{{}}'::jsonb
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_projection
        ON {schema}memory_units
        USING gin(projection)
    """)


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_projection")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS projection")


def _oracle_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}memory_units ADD (
            projection CLOB DEFAULT ''{{}}'' NOT NULL
            CONSTRAINT mu_projection_json CHECK (projection IS JSON)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN
                RAISE;
            END IF;
    END;
    """)
    op.execute(f"""
        UPDATE {schema}memory_units mu
        SET projection =
            '{{"embedding":{{"v":1,"ok":' ||
            CASE WHEN mu.embedding IS NOT NULL THEN 'true' ELSE 'false' END ||
            '}},"tsvector":{{"v":1,"ok":true}},"temporal":{{"v":1,"grade":"' ||
            CASE
                WHEN mu.occurred_start IS NOT NULL OR mu.event_date IS NOT NULL THEN 'resolved'
                ELSE 'unresolved'
            END ||
            '"}},"entities":{{"v":1,"ok":' ||
            CASE
                WHEN EXISTS (SELECT 1 FROM {schema}unit_entities ue WHERE ue.unit_id = mu.id) THEN 'true'
                ELSE 'false'
            END ||
            '}},"extraction":{{"v":"legacy"}}}}'
        WHERE projection = '{{}}'
    """)


def _oracle_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"""
    BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE {schema}memory_units DROP COLUMN projection';
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
