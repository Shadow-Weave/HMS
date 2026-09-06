"""Add versioned RSI Harness snapshots and candidate state.

The first implementation is PostgreSQL-only.  Oracle remains fail-closed until
the same immutable snapshot semantics are implemented for its backend.
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "j5k6l7m8n9o0"
# The repository currently has two historical heads (the v0.5.3 merge chain
# and the later multimodal chain).  RSI is deliberately the first migration
# after both so `alembic upgrade head` remains unambiguous for fresh and
# upgraded installations.
down_revision: str | Sequence[str] | None = ("8c6fa6f7230b", "r6s7t8u9v0w1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}rsi_harness_snapshots (
            bank_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            parent_snapshot_id TEXT,
            status TEXT NOT NULL DEFAULT 'building',
            manifest JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            frozen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bank_id, snapshot_id),
            CONSTRAINT fk_rsi_snapshot_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_rsi_snapshot_parent FOREIGN KEY (bank_id, parent_snapshot_id)
                REFERENCES {schema}rsi_harness_snapshots(bank_id, snapshot_id),
            CONSTRAINT ck_rsi_snapshot_status CHECK (status IN ('building', 'frozen', 'deprecated')),
            CONSTRAINT ck_rsi_snapshot_manifest_object CHECK (jsonb_typeof(manifest) = 'object')
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}rsi_memory_candidates (
            bank_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            kind TEXT NOT NULL,
            trigger TEXT NOT NULL,
            content TEXT NOT NULL,
            preconditions JSONB NOT NULL DEFAULT '[]'::jsonb,
            anti_patterns JSONB NOT NULL DEFAULT '[]'::jsonb,
            evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
            source_commit TEXT,
            created_before_verifier BOOLEAN NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            invalidation_reason TEXT,
            promoted_snapshot_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bank_id, candidate_id),
            CONSTRAINT fk_rsi_candidate_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_rsi_candidate_snapshot FOREIGN KEY (bank_id, promoted_snapshot_id)
                REFERENCES {schema}rsi_harness_snapshots(bank_id, snapshot_id),
            CONSTRAINT ck_rsi_candidate_namespace CHECK (namespace IN ('candidate', 'project', 'general')),
            CONSTRAINT ck_rsi_candidate_status CHECK (
                status IN ('candidate', 'promoted', 'quarantined', 'invalidated')
            ),
            CONSTRAINT ck_rsi_candidate_before_verifier CHECK (created_before_verifier IS TRUE),
            CONSTRAINT ck_rsi_candidate_json CHECK (
                jsonb_typeof(preconditions) = 'array'
                AND jsonb_typeof(anti_patterns) = 'array'
                AND jsonb_typeof(evidence_refs) = 'array'
                AND jsonb_typeof(metadata) = 'object'
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}rsi_snapshot_memory_units (
            bank_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            text TEXT NOT NULL,
            context TEXT,
            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            event_date TIMESTAMPTZ,
            occurred_start TIMESTAMPTZ,
            occurred_end TIMESTAMPTZ,
            mentioned_at TIMESTAMPTZ,
            source_memory_ids UUID[],
            copied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bank_id, snapshot_id, memory_id),
            CONSTRAINT fk_rsi_snapshot_memory_snapshot FOREIGN KEY (bank_id, snapshot_id)
                REFERENCES {schema}rsi_harness_snapshots(bank_id, snapshot_id) ON DELETE CASCADE,
            CONSTRAINT ck_rsi_snapshot_memory_namespace CHECK (namespace IN ('project', 'general')),
            CONSTRAINT ck_rsi_snapshot_memory_json CHECK (
                jsonb_typeof(tags) = 'array'
                AND jsonb_typeof(metadata) = 'object'
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}rsi_verifier_receipts (
            bank_id TEXT NOT NULL,
            receipt_hash CHAR(64) NOT NULL,
            run_id TEXT NOT NULL,
            candidate_digest CHAR(64) NOT NULL,
            passed BOOLEAN NOT NULL,
            snapshot_id TEXT NOT NULL,
            consumed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bank_id, receipt_hash),
            CONSTRAINT fk_rsi_receipt_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_rsi_receipt_snapshot FOREIGN KEY (bank_id, snapshot_id)
                REFERENCES {schema}rsi_harness_snapshots(bank_id, snapshot_id) ON DELETE CASCADE,
            CONSTRAINT ck_rsi_receipt_passed CHECK (passed IS TRUE),
            CONSTRAINT ck_rsi_receipt_hash CHECK (receipt_hash ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_rsi_receipt_digest CHECK (candidate_digest ~ '^[0-9a-f]{{64}}$')
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}rsi_snapshot_candidates (
            bank_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            verifier_receipt_hash CHAR(64) NOT NULL,
            promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bank_id, snapshot_id, candidate_id),
            CONSTRAINT fk_rsi_sc_snapshot FOREIGN KEY (bank_id, snapshot_id)
                REFERENCES {schema}rsi_harness_snapshots(bank_id, snapshot_id) ON DELETE CASCADE,
            CONSTRAINT fk_rsi_sc_candidate FOREIGN KEY (bank_id, candidate_id)
                REFERENCES {schema}rsi_memory_candidates(bank_id, candidate_id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_rsi_candidates_bank_status ON {schema}rsi_memory_candidates (bank_id, status)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_rsi_candidates_run ON {schema}rsi_memory_candidates (bank_id, run_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_rsi_snapshot_candidates_lookup ON {schema}rsi_snapshot_candidates (bank_id, snapshot_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_rsi_snapshot_memory_lookup ON {schema}rsi_snapshot_memory_units (bank_id, snapshot_id, namespace)"
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}rsi_snapshot_candidates")
    op.execute(f"DROP TABLE IF EXISTS {schema}rsi_verifier_receipts")
    op.execute(f"DROP TABLE IF EXISTS {schema}rsi_snapshot_memory_units")
    op.execute(f"DROP TABLE IF EXISTS {schema}rsi_memory_candidates")
    op.execute(f"DROP TABLE IF EXISTS {schema}rsi_harness_snapshots")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
