"""Real PostgreSQL concurrency and crash-window qualification.

The test is opt-in because it creates rows in an explicitly supplied
disposable database.  It never calls a model provider and stores no media.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest
from alembic import command
from alembic.config import Config

from hms_api.engine.db.postgresql import PostgresConnection
from hms_api.engine.multimodal.checkpoints import VideoSegmentCheckpoint, VideoSegmentIdentity
from hms_api.engine.multimodal.ledger import (
    DescriptorIdentity,
    DocumentCommandSpec,
    LedgerConflictError,
    MultimodalLedger,
    PublishDecision,
    derive_descriptor_key,
    derive_document_command_key,
    derive_retain_input_fingerprint,
)
from hms_api.engine.multimodal.models import GroundedStatement, ModelTemporalSegment
from hms_api.migrations import run_migrations


def _alembic_config(database_url: str, schema: str) -> Config:
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).parents[2] / "hms_api" / "alembic"),
    )
    config.set_main_option("sqlalchemy.url", database_url)
    config.set_main_option("prepend_sys_path", ".")
    config.set_main_option("path_separator", "os")
    config.set_main_option("target_schema", schema)
    return config


@pytest.mark.asyncio
async def test_real_postgres_multimodal_migration_round_trip() -> None:
    """Prove upgrade -> downgrade -> upgrade in an isolated tenant schema."""

    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    schema = f"mm_migration_{uuid.uuid4().hex[:16]}"
    config = _alembic_config(database_url, schema)
    tables = (
        "multimodal_descriptor_cache",
        "multimodal_segment_checkpoints",
        "multimodal_document_heads",
        "multimodal_document_commands",
    )

    async def table_presence() -> dict[str, bool]:
        connection = await asyncpg.connect(database_url)
        try:
            return {
                table: bool(
                    await connection.fetchval(
                        "SELECT to_regclass($1)::text",
                        f'"{schema}"."{table}"',
                    )
                )
                for table in tables
            }
        finally:
            await connection.close()

    try:
        command.upgrade(config, "r6s7t8u9v0w1")
        assert all((await table_presence()).values())

        command.downgrade(config, "q5r6s7t8u9v0")
        assert not any((await table_presence()).values())

        command.upgrade(config, "r6s7t8u9v0w1")
        assert all((await table_presence()).values())
    finally:
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


@pytest.mark.asyncio
async def test_real_postgres_video_segment_checkpoint_is_claim_and_ttl_scoped() -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    run_migrations(database_url)
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    bank_id = f"multimodal-segment-{uuid.uuid4().hex}"
    asset_sha256 = "5" * 64
    pipeline_fingerprint = "6" * 64
    descriptor_key = derive_descriptor_key(
        tenant_scope="public",
        bank_id=bank_id,
        asset_sha256=asset_sha256,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    token = uuid.uuid4()
    segment_identity = VideoSegmentIdentity(
        segment_key="7" * 64,
        segment_id="segment-000",
        evidence_fingerprint="8" * 64,
    )
    checkpoint = VideoSegmentCheckpoint(
        **segment_identity.model_dump(),
        value=ModelTemporalSegment(
            segment_id="segment-000",
            summary=[
                GroundedStatement(
                    text="A validated editor state is visible.",
                    evidence_ids=["frame-000"],
                    uncertainty="low",
                )
            ],
            observations=[],
            visible_text=[],
            evidence_ids=["frame-000"],
        ),
        provider="fake",
        configured_model="gpt-5-mini",
        resolved_model="fake-gpt-5-mini",
        request_id="segment-request-safe",
        input_tokens=12,
        output_tokens=8,
        logical_calls=1,
        physical_attempts=1,
    )

    try:
        async with pool.acquire() as raw:
            await raw.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, "Segment test")
            conn = PostgresConnection(raw)
            ledger = MultimodalLedger()
            now = datetime.now(UTC)
            claimed = await ledger.claim_descriptor(
                conn,
                DescriptorIdentity(
                    bank_id=bank_id,
                    descriptor_key=descriptor_key,
                    asset_sha256=asset_sha256,
                    pipeline_fingerprint=pipeline_fingerprint,
                ),
                claim_token=token,
                now=now,
                lease_expires_at=now + timedelta(minutes=5),
            )
            assert claimed is not None
            await ledger.checkpoint_video_segment(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=token,
                checkpoint=checkpoint,
                now=now,
                expires_at=now + timedelta(hours=1),
            )

            restored = await ledger.get_video_segment_checkpoint(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=token,
                identity=segment_identity,
                now=now + timedelta(seconds=1),
            )
            assert restored == checkpoint
            assert (
                await raw.fetchval(
                    "SELECT COUNT(*) FROM multimodal_segment_checkpoints WHERE bank_id = $1 AND descriptor_key = $2",
                    bank_id,
                    descriptor_key,
                )
                == 1
            )

            with pytest.raises(LedgerConflictError, match="lost its active descriptor claim"):
                await ledger.get_video_segment_checkpoint(
                    conn,
                    bank_id=bank_id,
                    descriptor_key=descriptor_key,
                    claim_token=uuid.uuid4(),
                    identity=segment_identity,
                    now=now + timedelta(seconds=1),
                )

            await raw.execute(
                "UPDATE multimodal_segment_checkpoints "
                "SET checkpointed_at = NOW() - INTERVAL '2 seconds', "
                "expires_at = NOW() - INTERVAL '1 second' "
                "WHERE bank_id = $1 AND descriptor_key = $2",
                bank_id,
                descriptor_key,
            )
            assert (
                await ledger.get_video_segment_checkpoint(
                    conn,
                    bank_id=bank_id,
                    descriptor_key=descriptor_key,
                    claim_token=token,
                    identity=segment_identity,
                    now=datetime.now(UTC),
                )
                is None
            )
    finally:
        async with pool.acquire() as raw:
            await raw.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await pool.close()


@pytest.mark.asyncio
async def test_real_postgres_serializes_claims_sequences_and_crash_recovery() -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    run_migrations(database_url)
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=12)
    bank_id = f"multimodal-ledger-{uuid.uuid4().hex}"
    document_id = f"document-{uuid.uuid4().hex}"
    asset_sha256 = "a" * 64
    pipeline_fingerprint = "b" * 64
    descriptor_key = derive_descriptor_key(
        tenant_scope="public",
        bank_id=bank_id,
        asset_sha256=asset_sha256,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    retain_fingerprint = derive_retain_input_fingerprint(
        context={"project": "concurrency"},
        normalized_tags=["runtime"],
        timestamp=None,
        explicit_strategy=None,
        update_intent="replace",
    )
    command_key = derive_document_command_key(
        tenant_scope="public",
        bank_id=bank_id,
        document_id=document_id,
        descriptor_key=descriptor_key,
        retain_input_fingerprint=retain_fingerprint,
    )

    try:
        async with pool.acquire() as raw:
            await raw.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, "Runtime test")

        async def admit_same(attempt: int):
            async with pool.acquire() as raw:
                conn = PostgresConnection(raw)
                return await MultimodalLedger().admit_document_command(
                    conn,
                    DocumentCommandSpec(
                        bank_id=bank_id,
                        document_id=document_id,
                        command_key=command_key,
                        operation_id=uuid.uuid4(),
                        source_storage_key=f"immutable/source/retry-{attempt}",
                        asset_sha256=asset_sha256,
                        descriptor_key=descriptor_key,
                        retain_input_fingerprint=retain_fingerprint,
                        source_delete_after_retain=True,
                    ),
                    now=datetime.now(UTC),
                )

        admissions = await asyncio.gather(*(admit_same(index) for index in range(10)))
        assert sum(item.created for item in admissions) == 1
        assert {item.command.sequence for item in admissions} == {1}
        assert len({item.command.operation_id for item in admissions}) == 1

        identity = DescriptorIdentity(
            bank_id=bank_id,
            descriptor_key=descriptor_key,
            asset_sha256=asset_sha256,
            pipeline_fingerprint=pipeline_fingerprint,
        )
        claim_tokens = [uuid.uuid4() for _ in range(10)]

        async def claim(token: uuid.UUID):
            async with pool.acquire() as raw:
                conn = PostgresConnection(raw)
                now = datetime.now(UTC)
                return await MultimodalLedger().claim_descriptor(
                    conn,
                    identity,
                    claim_token=token,
                    now=now,
                    lease_expires_at=now + timedelta(minutes=5),
                )

        claims = await asyncio.gather(*(claim(token) for token in claim_tokens))
        assert sum(item is not None for item in claims) == 1
        winning_index = next(index for index, item in enumerate(claims) if item is not None)
        winning_token = claim_tokens[winning_index]

        # Simulate a worker crash after the external provider accepted work but
        # before the canonical checkpoint committed.  Reclaim must expose the
        # at-least-once billing boundary rather than claiming exactly-once.
        async with pool.acquire() as raw:
            conn = PostgresConnection(raw)
            ledger = MultimodalLedger()
            await ledger.mark_provider_started(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=winning_token,
                now=datetime.now(UTC),
            )
            await raw.execute(
                "UPDATE multimodal_descriptor_cache SET lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE bank_id = $1 AND descriptor_key = $2",
                bank_id,
                descriptor_key,
            )

        replacement_token = uuid.uuid4()
        async with pool.acquire() as raw:
            conn = PostgresConnection(raw)
            ledger = MultimodalLedger()
            now = datetime.now(UTC)
            reclaimed = await ledger.claim_descriptor(
                conn,
                identity,
                claim_token=replacement_token,
                now=now,
                lease_expires_at=now + timedelta(minutes=5),
            )
            assert reclaimed is not None
            assert reclaimed.possible_duplicate_provider_attempt is True
            checkpoint = await ledger.checkpoint_descriptor(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=replacement_token,
                canonical_markdown="# Media memory\n\nGrounded runtime checkpoint.\n",
                provenance_metadata={"media_kind": "image"},
                entities=[],
                now=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )
            assert checkpoint.status == "completed"
            assert checkpoint.possible_duplicate_provider_attempt is True

        async def admit_distinct(index: int):
            distinct_retain = derive_retain_input_fingerprint(
                context={"revision": index},
                normalized_tags=["runtime"],
                timestamp=None,
                explicit_strategy=None,
                update_intent="replace",
            )
            distinct_key = derive_document_command_key(
                tenant_scope="public",
                bank_id=bank_id,
                document_id=document_id,
                descriptor_key=descriptor_key,
                retain_input_fingerprint=distinct_retain,
            )
            async with pool.acquire() as raw:
                conn = PostgresConnection(raw)
                admission = await MultimodalLedger().admit_document_command(
                    conn,
                    DocumentCommandSpec(
                        bank_id=bank_id,
                        document_id=document_id,
                        command_key=distinct_key,
                        operation_id=uuid.uuid4(),
                        source_storage_key=f"immutable/source/revision-{index}",
                        asset_sha256=asset_sha256,
                        descriptor_key=descriptor_key,
                        retain_input_fingerprint=distinct_retain,
                        source_delete_after_retain=False,
                    ),
                    now=datetime.now(UTC),
                )
                return admission.command

        distinct_commands = await asyncio.gather(*(admit_distinct(index) for index in range(5)))
        assert sorted(command.sequence for command in distinct_commands) == [2, 3, 4, 5, 6]
        newest = max(distinct_commands, key=lambda command: command.sequence)
        older = min(distinct_commands, key=lambda command: command.sequence)

        async with pool.acquire() as raw:
            conn = PostgresConnection(raw)
            ledger = MultimodalLedger()
            await ledger.mark_document_processing(
                conn,
                bank_id=bank_id,
                document_id=document_id,
                command_key=newest.command_key,
                now=datetime.now(UTC),
            )
            await ledger.attach_child_retain(
                conn,
                bank_id=bank_id,
                document_id=document_id,
                command_key=newest.command_key,
                child_retain_operation_id=uuid.uuid4(),
                now=datetime.now(UTC),
            )
            async with conn.transaction():
                decision, locked = await ledger.lock_for_publish(
                    conn,
                    bank_id=bank_id,
                    document_id=document_id,
                    command_key=newest.command_key,
                )
                assert decision is PublishDecision.PUBLISH
                completed = await ledger.complete_publish(conn, command=locked, now=datetime.now(UTC))
                assert completed.status == "completed"

            older_decision, _ = await ledger.lock_for_publish(
                conn,
                bank_id=bank_id,
                document_id=document_id,
                command_key=older.command_key,
            )
            assert older_decision is PublishDecision.SUPERSEDED

            head = await ledger.get_document_head(conn, bank_id=bank_id, document_id=document_id)
            assert head is not None
            assert head.published_sequence == 6
            assert head.active_sequence is None
    finally:
        async with pool.acquire() as raw:
            await raw.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await pool.close()


@pytest.mark.asyncio
async def test_real_postgres_descriptor_ttl_cleanup_is_bounded_and_non_destructive() -> None:
    database_url = os.getenv("HMS_TEST_MULTIMODAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set HMS_TEST_MULTIMODAL_DATABASE_URL to a disposable PostgreSQL database")

    run_migrations(database_url)
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=3)
    bank_id = f"multimodal-ttl-{uuid.uuid4().hex}"
    other_bank_id = f"multimodal-ttl-other-{uuid.uuid4().hex}"
    document_id = f"document-{uuid.uuid4().hex}"
    operation_id = uuid.uuid4()
    expired_keys = ["1" * 64, "2" * 64]
    live_key = "3" * 64
    other_bank_expired_key = "4" * 64
    asset_sha256 = "a" * 64
    pipeline_fingerprint = "b" * 64
    retain_fingerprint = "c" * 64
    command_key = "d" * 64
    now = datetime.now(UTC)

    try:
        async with pool.acquire() as raw:
            await raw.executemany(
                "INSERT INTO banks (bank_id, name) VALUES ($1, $2)",
                [(bank_id, "TTL test"), (other_bank_id, "TTL other-bank test")],
            )
            for descriptor_bank, descriptor_key, expires_at in [
                (bank_id, expired_keys[0], now - timedelta(minutes=2)),
                (bank_id, expired_keys[1], now - timedelta(minutes=1)),
                (bank_id, live_key, now + timedelta(days=1)),
                (other_bank_id, other_bank_expired_key, now - timedelta(minutes=3)),
            ]:
                await raw.execute(
                    """INSERT INTO multimodal_descriptor_cache
                       (bank_id, descriptor_key, asset_sha256, pipeline_fingerprint,
                        status, canonical_markdown, checkpointed_at, expires_at)
                       VALUES ($1, $2, $3, $4, 'completed', $5, $6, $7)""",
                    descriptor_bank,
                    descriptor_key,
                    asset_sha256,
                    pipeline_fingerprint,
                    "# Derived descriptor\n",
                    now,
                    expires_at,
                )

            conn = PostgresConnection(raw)
            admission = await MultimodalLedger().admit_document_command(
                conn,
                DocumentCommandSpec(
                    bank_id=bank_id,
                    document_id=document_id,
                    command_key=command_key,
                    operation_id=operation_id,
                    source_storage_key="immutable/source/ttl-command",
                    asset_sha256=asset_sha256,
                    descriptor_key=expired_keys[0],
                    retain_input_fingerprint=retain_fingerprint,
                    source_delete_after_retain=False,
                ),
                now=now,
            )
            assert admission.created is True
            await raw.execute(
                """INSERT INTO async_operations
                   (operation_id, bank_id, operation_type, status, task_payload, result_metadata)
                   VALUES ($1, $2, 'file_convert_retain', 'pending', '{}'::jsonb, '{}'::jsonb)""",
                operation_id,
                bank_id,
            )

            ledger = MultimodalLedger()
            first_deleted = await ledger.purge_expired_descriptors(
                conn,
                bank_id=bank_id,
                now=now,
                limit=1,
            )
            assert first_deleted == 1
            first_remaining = await raw.fetch(
                "SELECT descriptor_key FROM multimodal_descriptor_cache WHERE bank_id = $1 ORDER BY descriptor_key",
                bank_id,
            )
            assert [row["descriptor_key"] for row in first_remaining] == [expired_keys[1], live_key]

            second_deleted = await ledger.purge_expired_descriptors(
                conn,
                bank_id=bank_id,
                now=now,
                limit=100,
            )
            assert second_deleted == 1
            assert (
                await raw.fetchval(
                    "SELECT COUNT(*) FROM multimodal_descriptor_cache WHERE bank_id = $1 AND descriptor_key = $2",
                    bank_id,
                    live_key,
                )
                == 1
            )
            assert (
                await raw.fetchval(
                    "SELECT COUNT(*) FROM multimodal_descriptor_cache WHERE bank_id = $1 AND descriptor_key = $2",
                    other_bank_id,
                    other_bank_expired_key,
                )
                == 1
            )
            assert (
                await raw.fetchval(
                    "SELECT COUNT(*) FROM multimodal_document_commands "
                    "WHERE bank_id = $1 AND document_id = $2 AND command_key = $3",
                    bank_id,
                    document_id,
                    command_key,
                )
                == 1
            )
            assert (
                await raw.fetchval(
                    "SELECT COUNT(*) FROM async_operations WHERE operation_id = $1 AND bank_id = $2",
                    operation_id,
                    bank_id,
                )
                == 1
            )
    finally:
        async with pool.acquire() as raw:
            await raw.execute("DELETE FROM banks WHERE bank_id = ANY($1::text[])", [bank_id, other_bank_id])
        await pool.close()
