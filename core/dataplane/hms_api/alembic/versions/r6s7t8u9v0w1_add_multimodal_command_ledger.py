"""Add the durable multimodal descriptor and document-command ledger.

The descriptor cache and document command use deliberately different keys:
one identifies reusable model work, while the other identifies a caller's
logical document update.  A per-document head assigns admission order and
provides the compare-and-swap state needed to prevent an older, slower media
conversion from overwriting a newer accepted command.

Only derived text, bounded JSON metadata, hashes, identifiers, and lifecycle
state are stored here.  Raw media, normalized frame bytes, base64 data URLs,
and provider request bodies have no columns in this schema.

Revision ID: r6s7t8u9v0w1
Revises: q5r6s7t8u9v0
Create Date: 2026-07-22
"""

from collections.abc import Sequence

from alembic import context, op

from hms_api.alembic._dialect import run_for_dialect

revision: str = "r6s7t8u9v0w1"
down_revision: str | Sequence[str] | None = "q5r6s7t8u9v0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # Dependency order is descriptor cache -> document heads -> commands.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}multimodal_descriptor_cache (
            bank_id TEXT NOT NULL,
            descriptor_key VARCHAR(64) NOT NULL,
            asset_sha256 CHAR(64) NOT NULL,
            pipeline_fingerprint CHAR(64) NOT NULL,
            status VARCHAR(24) NOT NULL DEFAULT 'pending',
            claim_token UUID,
            lease_expires_at TIMESTAMPTZ,
            provider_started_at TIMESTAMPTZ,
            possible_duplicate_provider_attempt BOOLEAN NOT NULL DEFAULT FALSE,
            canonical_markdown TEXT,
            provenance_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            entities JSONB NOT NULL DEFAULT '[]'::jsonb,
            checkpointed_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT pk_mm_descriptor_cache PRIMARY KEY (bank_id, descriptor_key),
            CONSTRAINT fk_mm_desc_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT ck_mm_desc_status CHECK (
                status IN ('pending', 'processing', 'completed', 'failed')
            ),
            CONSTRAINT ck_mm_desc_asset CHECK (asset_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_mm_desc_pipeline CHECK (pipeline_fingerprint ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_mm_desc_claim CHECK (
                (status = 'processing' AND claim_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR (status <> 'processing' AND claim_token IS NULL AND lease_expires_at IS NULL)
            ),
            CONSTRAINT ck_mm_desc_checkpoint CHECK (
                status <> 'completed'
                OR (canonical_markdown IS NOT NULL AND checkpointed_at IS NOT NULL)
            ),
            CONSTRAINT ck_mm_desc_provenance CHECK (
                jsonb_typeof(provenance_metadata) = 'object'
            ),
            CONSTRAINT ck_mm_desc_entities CHECK (jsonb_typeof(entities) = 'array')
        )
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_desc_claim
        ON {schema}multimodal_descriptor_cache (bank_id, status, lease_expires_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_desc_expiry
        ON {schema}multimodal_descriptor_cache (bank_id, expires_at)
        WHERE expires_at IS NOT NULL
    """)

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}multimodal_segment_checkpoints (
            bank_id TEXT NOT NULL,
            descriptor_key VARCHAR(64) NOT NULL,
            segment_key VARCHAR(64) NOT NULL,
            segment_id VARCHAR(160) NOT NULL,
            evidence_fingerprint CHAR(64) NOT NULL,
            segment_json JSONB NOT NULL,
            provider VARCHAR(128) NOT NULL,
            configured_model VARCHAR(256) NOT NULL,
            resolved_model VARCHAR(256),
            provider_request_id VARCHAR(256),
            input_tokens BIGINT NOT NULL,
            output_tokens BIGINT NOT NULL,
            logical_calls BIGINT NOT NULL,
            physical_attempts BIGINT NOT NULL,
            checkpointed_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT pk_mm_segment_checkpoints
                PRIMARY KEY (bank_id, descriptor_key, segment_key),
            CONSTRAINT fk_mm_segment_descriptor
                FOREIGN KEY (bank_id, descriptor_key)
                REFERENCES {schema}multimodal_descriptor_cache(bank_id, descriptor_key)
                ON DELETE CASCADE,
            CONSTRAINT ck_mm_segment_key CHECK (segment_key ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_mm_segment_evidence CHECK (
                evidence_fingerprint ~ '^[0-9a-f]{{64}}$'
            ),
            CONSTRAINT ck_mm_segment_json CHECK (jsonb_typeof(segment_json) = 'object'),
            CONSTRAINT ck_mm_segment_usage CHECK (
                input_tokens >= 0 AND output_tokens >= 0
                AND logical_calls >= 1
                AND physical_attempts >= logical_calls
            ),
            CONSTRAINT ck_mm_segment_expiry CHECK (expires_at > checkpointed_at)
        )
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_segment_expiry
        ON {schema}multimodal_segment_checkpoints (bank_id, expires_at)
    """)

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}multimodal_document_heads (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            next_sequence BIGINT NOT NULL DEFAULT 1,
            published_sequence BIGINT NOT NULL DEFAULT 0,
            active_sequence BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT pk_mm_document_heads PRIMARY KEY (bank_id, document_id),
            CONSTRAINT fk_mm_heads_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT ck_mm_heads_sequence CHECK (
                next_sequence >= 1
                AND published_sequence >= 0
                AND published_sequence < next_sequence
                AND (active_sequence IS NULL OR (
                    active_sequence >= 1 AND active_sequence < next_sequence
                ))
            )
        )
    """)

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}multimodal_document_commands (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            command_key VARCHAR(64) NOT NULL,
            sequence BIGINT NOT NULL,
            operation_id UUID NOT NULL,
            source_storage_key TEXT NOT NULL,
            asset_sha256 CHAR(64) NOT NULL,
            descriptor_key VARCHAR(64) NOT NULL,
            retain_input_fingerprint CHAR(64) NOT NULL,
            status VARCHAR(24) NOT NULL DEFAULT 'pending',
            child_retain_operation_id UUID,
            source_delete_after_retain BOOLEAN NOT NULL,
            source_deleted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            CONSTRAINT pk_mm_document_commands
                PRIMARY KEY (bank_id, document_id, command_key),
            CONSTRAINT uq_mm_doc_cmd_sequence UNIQUE (bank_id, document_id, sequence),
            CONSTRAINT uq_mm_doc_cmd_operation UNIQUE (bank_id, operation_id),
            CONSTRAINT fk_mm_cmd_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_mm_cmd_head FOREIGN KEY (bank_id, document_id)
                REFERENCES {schema}multimodal_document_heads(bank_id, document_id)
                ON DELETE CASCADE,
            CONSTRAINT ck_mm_cmd_status CHECK (
                status IN (
                    'pending', 'processing', 'retaining', 'completed',
                    'failed', 'superseded', 'cancelled'
                )
            ),
            CONSTRAINT ck_mm_cmd_sequence CHECK (sequence >= 1),
            CONSTRAINT ck_mm_cmd_source_key CHECK (char_length(source_storage_key) <= 512),
            CONSTRAINT ck_mm_cmd_asset CHECK (asset_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_mm_cmd_descriptor CHECK (descriptor_key ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_mm_cmd_retain_fp CHECK (
                retain_input_fingerprint ~ '^[0-9a-f]{{64}}$'
            ),
            CONSTRAINT ck_mm_cmd_source_delete CHECK (
                source_deleted_at IS NULL OR source_delete_after_retain
            ),
            CONSTRAINT ck_mm_cmd_completed CHECK (
                status <> 'completed' OR completed_at IS NOT NULL
            )
        )
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_cmd_status
        ON {schema}multimodal_document_commands (bank_id, status, updated_at)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_cmd_descriptor
        ON {schema}multimodal_document_commands (bank_id, descriptor_key)
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_mm_cmd_child_op
        ON {schema}multimodal_document_commands (bank_id, child_retain_operation_id)
        WHERE child_retain_operation_id IS NOT NULL
    """)


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    # Reverse dependency order: commands reference heads and segment rows
    # reference descriptor rows.
    op.execute(f"DROP TABLE IF EXISTS {schema}multimodal_document_commands")
    op.execute(f"DROP TABLE IF EXISTS {schema}multimodal_document_heads")
    op.execute(f"DROP TABLE IF EXISTS {schema}multimodal_segment_checkpoints")
    op.execute(f"DROP TABLE IF EXISTS {schema}multimodal_descriptor_cache")


def _oracle_create_ignoring_955(sql: str) -> None:
    """Execute one CREATE and tolerate only the already-exists condition."""

    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_drop_ignoring_942(sql: str) -> None:
    """Execute one DROP and tolerate only a missing table."""

    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -942 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    schema = _get_schema_prefix()

    tables = (
        f"""
        CREATE TABLE {schema}multimodal_descriptor_cache (
            bank_id VARCHAR2(256) NOT NULL,
            descriptor_key VARCHAR2(64) NOT NULL,
            asset_sha256 CHAR(64) NOT NULL,
            pipeline_fingerprint CHAR(64) NOT NULL,
            status VARCHAR2(24) DEFAULT 'pending' NOT NULL,
            claim_token RAW(16),
            lease_expires_at TIMESTAMP WITH TIME ZONE,
            provider_started_at TIMESTAMP WITH TIME ZONE,
            possible_duplicate_provider_attempt NUMBER(1) DEFAULT 0 NOT NULL,
            canonical_markdown CLOB,
            provenance_metadata CLOB DEFAULT '{{}}' NOT NULL,
            entities CLOB DEFAULT '[]' NOT NULL,
            checkpointed_at TIMESTAMP WITH TIME ZONE,
            expires_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_mm_descriptor_cache PRIMARY KEY (bank_id, descriptor_key),
            CONSTRAINT fk_mm_desc_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT ck_mm_desc_status CHECK (
                status IN ('pending', 'processing', 'completed', 'failed')
            ),
            CONSTRAINT ck_mm_desc_asset CHECK (
                REGEXP_LIKE(asset_sha256, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_desc_pipeline CHECK (
                REGEXP_LIKE(pipeline_fingerprint, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_desc_claim CHECK (
                (status = 'processing' AND claim_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR (status <> 'processing' AND claim_token IS NULL AND lease_expires_at IS NULL)
            ),
            CONSTRAINT ck_mm_desc_checkpoint CHECK (
                status <> 'completed'
                OR (canonical_markdown IS NOT NULL AND checkpointed_at IS NOT NULL)
            ),
            CONSTRAINT ck_mm_desc_dup CHECK (
                possible_duplicate_provider_attempt IN (0, 1)
            ),
            CONSTRAINT ck_mm_desc_provenance CHECK (provenance_metadata IS JSON),
            CONSTRAINT ck_mm_desc_entities CHECK (entities IS JSON)
        )
        """,
        f"""
        CREATE TABLE {schema}multimodal_segment_checkpoints (
            bank_id VARCHAR2(256) NOT NULL,
            descriptor_key VARCHAR2(64) NOT NULL,
            segment_key VARCHAR2(64) NOT NULL,
            segment_id VARCHAR2(160) NOT NULL,
            evidence_fingerprint CHAR(64) NOT NULL,
            segment_json CLOB NOT NULL,
            provider VARCHAR2(128) NOT NULL,
            configured_model VARCHAR2(256) NOT NULL,
            resolved_model VARCHAR2(256),
            provider_request_id VARCHAR2(256),
            input_tokens NUMBER(19) NOT NULL,
            output_tokens NUMBER(19) NOT NULL,
            logical_calls NUMBER(19) NOT NULL,
            physical_attempts NUMBER(19) NOT NULL,
            checkpointed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_mm_segment_checkpoints
                PRIMARY KEY (bank_id, descriptor_key, segment_key),
            CONSTRAINT fk_mm_segment_descriptor
                FOREIGN KEY (bank_id, descriptor_key)
                REFERENCES {schema}multimodal_descriptor_cache(bank_id, descriptor_key)
                ON DELETE CASCADE,
            CONSTRAINT ck_mm_segment_key CHECK (
                REGEXP_LIKE(segment_key, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_segment_evidence CHECK (
                REGEXP_LIKE(evidence_fingerprint, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_segment_json CHECK (segment_json IS JSON),
            CONSTRAINT ck_mm_segment_usage CHECK (
                input_tokens >= 0 AND output_tokens >= 0
                AND logical_calls >= 1
                AND physical_attempts >= logical_calls
            ),
            CONSTRAINT ck_mm_segment_expiry CHECK (expires_at > checkpointed_at)
        )
        """,
        f"""
        CREATE TABLE {schema}multimodal_document_heads (
            bank_id VARCHAR2(256) NOT NULL,
            document_id VARCHAR2(512) NOT NULL,
            next_sequence NUMBER(19) DEFAULT 1 NOT NULL,
            published_sequence NUMBER(19) DEFAULT 0 NOT NULL,
            active_sequence NUMBER(19),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_mm_document_heads PRIMARY KEY (bank_id, document_id),
            CONSTRAINT fk_mm_heads_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT ck_mm_heads_sequence CHECK (
                next_sequence >= 1
                AND published_sequence >= 0
                AND published_sequence < next_sequence
                AND (active_sequence IS NULL OR (
                    active_sequence >= 1 AND active_sequence < next_sequence
                ))
            )
        )
        """,
        f"""
        CREATE TABLE {schema}multimodal_document_commands (
            bank_id VARCHAR2(256) NOT NULL,
            document_id VARCHAR2(512) NOT NULL,
            command_key VARCHAR2(64) NOT NULL,
            sequence NUMBER(19) NOT NULL,
            operation_id RAW(16) NOT NULL,
            source_storage_key VARCHAR2(512) NOT NULL,
            asset_sha256 CHAR(64) NOT NULL,
            descriptor_key VARCHAR2(64) NOT NULL,
            retain_input_fingerprint CHAR(64) NOT NULL,
            status VARCHAR2(24) DEFAULT 'pending' NOT NULL,
            child_retain_operation_id RAW(16),
            source_delete_after_retain NUMBER(1) NOT NULL,
            source_deleted_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            completed_at TIMESTAMP WITH TIME ZONE,
            CONSTRAINT pk_mm_document_commands
                PRIMARY KEY (bank_id, document_id, command_key),
            CONSTRAINT uq_mm_doc_cmd_sequence UNIQUE (bank_id, document_id, sequence),
            CONSTRAINT uq_mm_doc_cmd_operation UNIQUE (bank_id, operation_id),
            CONSTRAINT fk_mm_cmd_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_mm_cmd_head FOREIGN KEY (bank_id, document_id)
                REFERENCES {schema}multimodal_document_heads(bank_id, document_id)
                ON DELETE CASCADE,
            CONSTRAINT ck_mm_cmd_status CHECK (
                status IN (
                    'pending', 'processing', 'retaining', 'completed',
                    'failed', 'superseded', 'cancelled'
                )
            ),
            CONSTRAINT ck_mm_cmd_sequence CHECK (sequence >= 1),
            CONSTRAINT ck_mm_cmd_asset CHECK (
                REGEXP_LIKE(asset_sha256, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_cmd_descriptor CHECK (
                REGEXP_LIKE(descriptor_key, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_cmd_retain_fp CHECK (
                REGEXP_LIKE(retain_input_fingerprint, '^[0-9a-f]{{64}}$')
            ),
            CONSTRAINT ck_mm_cmd_delete_flag CHECK (source_delete_after_retain IN (0, 1)),
            CONSTRAINT ck_mm_cmd_source_delete CHECK (
                source_deleted_at IS NULL OR source_delete_after_retain = 1
            ),
            CONSTRAINT ck_mm_cmd_completed CHECK (
                status <> 'completed' OR completed_at IS NOT NULL
            )
        )
        """,
    )

    indexes = (
        f"""CREATE INDEX {schema}idx_mm_desc_claim
            ON {schema}multimodal_descriptor_cache (bank_id, status, lease_expires_at)""",
        f"""CREATE INDEX {schema}idx_mm_desc_expiry
            ON {schema}multimodal_descriptor_cache (bank_id, expires_at)""",
        f"""CREATE INDEX {schema}idx_mm_segment_expiry
            ON {schema}multimodal_segment_checkpoints (bank_id, expires_at)""",
        f"""CREATE INDEX {schema}idx_mm_cmd_status
            ON {schema}multimodal_document_commands (bank_id, status, updated_at)""",
        f"""CREATE INDEX {schema}idx_mm_cmd_descriptor
            ON {schema}multimodal_document_commands (bank_id, descriptor_key)""",
        f"""CREATE INDEX {schema}idx_mm_cmd_child_op
            ON {schema}multimodal_document_commands (bank_id, child_retain_operation_id)""",
    )

    for ddl in tables:
        _oracle_create_ignoring_955(ddl)
    for ddl in indexes:
        _oracle_create_ignoring_955(ddl)


def _oracle_downgrade() -> None:
    schema = _get_schema_prefix()
    # Reverse dependency order.  CASCADE CONSTRAINTS also removes each table's
    # indexes, so no independent DROP INDEX race is introduced.
    _oracle_drop_ignoring_942(f"DROP TABLE {schema}multimodal_document_commands CASCADE CONSTRAINTS")
    _oracle_drop_ignoring_942(f"DROP TABLE {schema}multimodal_document_heads CASCADE CONSTRAINTS")
    _oracle_drop_ignoring_942(f"DROP TABLE {schema}multimodal_segment_checkpoints CASCADE CONSTRAINTS")
    _oracle_drop_ignoring_942(f"DROP TABLE {schema}multimodal_descriptor_cache CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
