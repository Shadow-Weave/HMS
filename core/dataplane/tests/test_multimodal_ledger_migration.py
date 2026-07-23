"""Static shape tests for the durable multimodal command ledger migration."""

from __future__ import annotations

import re

from hms_api.alembic.versions import r6s7t8u9v0w1_add_multimodal_command_ledger as migration


def _compact(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def test_revision_extends_the_previous_single_head() -> None:
    assert migration.revision == "r6s7t8u9v0w1"
    assert migration.down_revision == "q5r6s7t8u9v0"


def test_postgresql_shape_has_four_scoped_tables_and_no_payload_column(monkeypatch) -> None:
    statements: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"tenant".')
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration._pg_upgrade()

    compact = [_compact(statement) for statement in statements]
    creates = [statement for statement in compact if statement.startswith("CREATE TABLE")]
    indexes = [statement for statement in compact if statement.startswith("CREATE INDEX")]
    assert len(creates) == 4
    assert len(indexes) == 6
    assert '"tenant".multimodal_descriptor_cache' in creates[0]
    assert '"tenant".multimodal_segment_checkpoints' in creates[1]
    assert '"tenant".multimodal_document_heads' in creates[2]
    assert '"tenant".multimodal_document_commands' in creates[3]

    descriptor, segments, heads, commands = creates
    assert "PRIMARY KEY (bank_id, descriptor_key)" in descriptor
    assert "claim_token UUID" in descriptor
    assert "lease_expires_at TIMESTAMPTZ" in descriptor
    assert "provider_started_at TIMESTAMPTZ" in descriptor
    assert "possible_duplicate_provider_attempt BOOLEAN" in descriptor
    assert "canonical_markdown TEXT" in descriptor
    assert "provenance_metadata JSONB" in descriptor
    assert "entities JSONB" in descriptor
    assert "expires_at TIMESTAMPTZ" in descriptor
    assert "jsonb_typeof(provenance_metadata) = 'object'" in descriptor
    assert "jsonb_typeof(entities) = 'array'" in descriptor

    assert "PRIMARY KEY (bank_id, descriptor_key, segment_key)" in segments
    assert "FOREIGN KEY (bank_id, descriptor_key)" in segments
    assert "segment_json JSONB" in segments
    assert "jsonb_typeof(segment_json) = 'object'" in segments
    assert "expires_at > checkpointed_at" in segments
    assert "physical_attempts >= logical_calls" in segments

    assert "PRIMARY KEY (bank_id, document_id)" in heads
    assert "next_sequence BIGINT" in heads
    assert "published_sequence BIGINT" in heads
    assert "active_sequence BIGINT" in heads

    assert "PRIMARY KEY (bank_id, document_id, command_key)" in commands
    assert "UNIQUE (bank_id, document_id, sequence)" in commands
    assert "UNIQUE (bank_id, operation_id)" in commands
    assert "operation_id UUID" in commands
    assert "source_storage_key TEXT" in commands
    assert "char_length(source_storage_key) <= 512" in commands
    assert "asset_sha256 CHAR(64)" in commands
    assert "descriptor_key VARCHAR(64)" in commands
    assert "retain_input_fingerprint CHAR(64)" in commands
    assert "child_retain_operation_id UUID" in commands

    all_sql = " ".join(compact).lower()
    assert all_sql.count('references "tenant".banks(bank_id) on delete cascade') == 3
    assert "bytea" not in all_sql
    assert "blob" not in all_sql
    assert "base64" not in all_sql
    assert "provider_request_body" not in all_sql
    assert "provider_response" not in all_sql


def test_postgresql_downgrade_is_reverse_dependency_order(monkeypatch) -> None:
    statements: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"tenant".')
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration._pg_downgrade()

    assert statements == [
        'DROP TABLE IF EXISTS "tenant".multimodal_document_commands',
        'DROP TABLE IF EXISTS "tenant".multimodal_document_heads',
        'DROP TABLE IF EXISTS "tenant".multimodal_segment_checkpoints',
        'DROP TABLE IF EXISTS "tenant".multimodal_descriptor_cache',
    ]


def test_oracle_shape_is_dialect_equivalent_and_json_checked(monkeypatch) -> None:
    creates: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"TENANT".')
    monkeypatch.setattr(migration, "_oracle_create_ignoring_955", creates.append)

    migration._oracle_upgrade()

    compact = [_compact(statement) for statement in creates]
    tables = [statement for statement in compact if statement.startswith("CREATE TABLE")]
    indexes = [statement for statement in compact if statement.startswith("CREATE INDEX")]
    assert len(tables) == 4
    assert len(indexes) == 6

    descriptor, segments, heads, commands = tables
    assert "claim_token RAW(16)" in descriptor
    assert "TIMESTAMP WITH TIME ZONE" in descriptor
    assert "possible_duplicate_provider_attempt NUMBER(1)" in descriptor
    assert "canonical_markdown CLOB" in descriptor
    assert "provenance_metadata CLOB" in descriptor
    assert "provenance_metadata IS JSON" in descriptor
    assert "entities CLOB" in descriptor
    assert "entities IS JSON" in descriptor

    assert "segment_json CLOB" in segments
    assert "segment_json IS JSON" in segments
    assert "FOREIGN KEY (bank_id, descriptor_key)" in segments
    assert "physical_attempts >= logical_calls" in segments

    assert "next_sequence NUMBER(19)" in heads
    assert "published_sequence NUMBER(19)" in heads
    assert "active_sequence NUMBER(19)" in heads

    assert "operation_id RAW(16)" in commands
    assert "source_storage_key VARCHAR2(512)" in commands
    assert "child_retain_operation_id RAW(16)" in commands
    assert "source_delete_after_retain NUMBER(1)" in commands
    assert "source_delete_after_retain IN (0, 1)" in commands
    assert "UNIQUE (bank_id, document_id, sequence)" in commands

    all_sql = " ".join(compact).lower()
    assert all_sql.count('references "tenant".banks(bank_id) on delete cascade') == 3
    assert " bytea" not in all_sql
    assert " blob" not in all_sql
    assert "base64" not in all_sql
    assert "provider_request_body" not in all_sql
    assert "provider_response" not in all_sql


def test_oracle_downgrade_is_reverse_dependency_order(monkeypatch) -> None:
    statements: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"TENANT".')
    monkeypatch.setattr(migration, "_oracle_drop_ignoring_942", statements.append)

    migration._oracle_downgrade()

    assert statements == [
        'DROP TABLE "TENANT".multimodal_document_commands CASCADE CONSTRAINTS',
        'DROP TABLE "TENANT".multimodal_document_heads CASCADE CONSTRAINTS',
        'DROP TABLE "TENANT".multimodal_segment_checkpoints CASCADE CONSTRAINTS',
        'DROP TABLE "TENANT".multimodal_descriptor_cache CASCADE CONSTRAINTS',
    ]


def test_oracle_helpers_swallow_only_the_audited_error_codes(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class Bind:
        def exec_driver_sql(self, sql: str, params: dict[str, str]) -> None:
            calls.append((sql, params))

    monkeypatch.setattr(migration.op, "get_bind", lambda: Bind())
    migration._oracle_create_ignoring_955("CREATE TABLE example (id NUMBER)")
    migration._oracle_drop_ignoring_942("DROP TABLE example")

    assert "SQLCODE = -955" in calls[0][0]
    assert "SQLCODE = -942" in calls[1][0]
    assert calls[0][1] == {"stmt": "CREATE TABLE example (id NUMBER)"}
    assert calls[1][1] == {"stmt": "DROP TABLE example"}
