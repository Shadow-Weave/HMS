"""Pure unit tests for multimodal ledger identities and SQL CAS primitives."""

from __future__ import annotations

import base64
import json
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from hms_api.engine.multimodal.checkpoints import VideoSegmentCheckpoint, VideoSegmentIdentity
from hms_api.engine.multimodal.ledger import (
    DescriptorIdentity,
    DocumentCommandSpec,
    LedgerConflictError,
    LedgerInvariantError,
    LedgerTables,
    MultimodalLedger,
    PublishDecision,
    derive_descriptor_key,
    derive_document_command_key,
    derive_retain_input_fingerprint,
)
from hms_api.engine.multimodal.models import GroundedStatement, ModelTemporalSegment

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
ASSET_SHA = "a" * 64
PIPELINE_FP = "b" * 64
DESCRIPTOR_KEY = "c" * 64
RETAIN_FP = "d" * 64
COMMAND_KEY = "e" * 64
SEGMENT_KEY = "f" * 64
EVIDENCE_FP = "1" * 64


class _Transaction(AbstractAsyncContextManager):
    def __init__(self, conn: "FakeConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "FakeConnection":
        self.conn.transaction_depth += 1
        self.conn.max_transaction_depth = max(self.conn.max_transaction_depth, self.conn.transaction_depth)
        return self.conn

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.conn.transaction_depth -= 1


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any] | None], *, backend_type: str = "postgresql") -> None:
        self.backend_type = backend_type
        self.rows = list(rows)
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.transaction_depth = 0
        self.max_transaction_depth = 0

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        self.calls.append(("execute", query, args))
        command = query.lstrip().split(maxsplit=1)[0].upper()
        return "INSERT 0 1" if command == "INSERT" else f"{command} 1"

    async def fetchrow(self, query: str, *args: Any, timeout: float | None = None) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", query, args))
        if not self.rows:
            raise AssertionError(f"no scripted row remains for query: {query}")
        return self.rows.pop(0)

    def parse_json(self, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


class FakeCleanupConnection(FakeConnection):
    async def fetch(self, query: str, *args: Any, timeout: float | None = None) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query, args))
        return [{"descriptor_key": "1" * 64}, {"descriptor_key": "2" * 64}]


def _descriptor_row(
    *,
    status: str = "pending",
    claim_token: UUID | bytes | None = None,
    possible_duplicate: bool | int = False,
    canonical_markdown: str | None = None,
) -> dict[str, Any]:
    if status == "completed" and canonical_markdown is None:
        canonical_markdown = "# media\n"
    return {
        "bank_id": "bank-a",
        "descriptor_key": DESCRIPTOR_KEY,
        "asset_sha256": ASSET_SHA,
        "pipeline_fingerprint": PIPELINE_FP,
        "status": status,
        "claim_token": claim_token,
        "lease_expires_at": NOW + timedelta(minutes=5) if status == "processing" else None,
        "provider_started_at": None,
        "possible_duplicate_provider_attempt": possible_duplicate,
        "canonical_markdown": canonical_markdown,
        "provenance_metadata": "{}",
        "entities": "[]",
        "checkpointed_at": NOW if status == "completed" else None,
        "expires_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _head_row(*, active: int | None = 1, published: int = 0, next_sequence: int = 2) -> dict[str, Any]:
    return {
        "bank_id": "bank-a",
        "document_id": "doc-a",
        "next_sequence": next_sequence,
        "published_sequence": published,
        "active_sequence": active,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _command_row(
    *,
    status: str = "pending",
    sequence: int = 1,
    operation_id: UUID | None = None,
    command_key: str = COMMAND_KEY,
) -> dict[str, Any]:
    return {
        "bank_id": "bank-a",
        "document_id": "doc-a",
        "command_key": command_key,
        "sequence": sequence,
        "operation_id": operation_id or UUID("00000000-0000-0000-0000-000000000001"),
        "source_storage_key": "immutable/source/asset-a",
        "asset_sha256": ASSET_SHA,
        "descriptor_key": DESCRIPTOR_KEY,
        "retain_input_fingerprint": RETAIN_FP,
        "status": status,
        "child_retain_operation_id": (
            UUID("00000000-0000-0000-0000-000000000002") if status in {"retaining", "completed"} else None
        ),
        "source_delete_after_retain": True,
        "source_deleted_at": None,
        "created_at": NOW,
        "updated_at": NOW,
        "completed_at": NOW if status == "completed" else None,
    }


def _identity() -> DescriptorIdentity:
    return DescriptorIdentity(
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        asset_sha256=ASSET_SHA,
        pipeline_fingerprint=PIPELINE_FP,
    )


def _segment_checkpoint(*, text: str = "A grounded editor state is visible.") -> VideoSegmentCheckpoint:
    return VideoSegmentCheckpoint(
        segment_key=SEGMENT_KEY,
        segment_id="segment-000",
        evidence_fingerprint=EVIDENCE_FP,
        value=ModelTemporalSegment(
            segment_id="segment-000",
            summary=[
                GroundedStatement(
                    text=text,
                    evidence_ids=["frame-000"],
                    uncertainty="low",
                )
            ],
            observations=[],
            visible_text=[],
            evidence_ids=["frame-000"],
        ),
        provider="openai",
        configured_model="gpt-5-mini",
        resolved_model="gpt-5-mini-test",
        request_id="request-safe",
        input_tokens=11,
        output_tokens=7,
        logical_calls=1,
        physical_attempts=1,
    )


def _segment_row() -> dict[str, Any]:
    checkpoint = _segment_checkpoint()
    return {
        "bank_id": "bank-a",
        "descriptor_key": DESCRIPTOR_KEY,
        **checkpoint.model_dump(exclude={"value", "request_id"}),
        "segment_json": checkpoint.value.model_dump_json(),
        "provider_request_id": checkpoint.request_id,
        "checkpointed_at": NOW,
        "expires_at": NOW + timedelta(days=1),
        "created_at": NOW,
        "updated_at": NOW,
    }


@pytest.mark.asyncio
async def test_expired_descriptor_cleanup_is_bounded_and_bank_scoped() -> None:
    conn = FakeCleanupConnection([])
    ledger = MultimodalLedger()

    deleted = await ledger.purge_expired_descriptors(
        conn,
        bank_id="bank-a",
        now=NOW,
        limit=2,
    )

    assert deleted == 2
    fetch_call = next(call for call in conn.calls if call[0] == "fetch")
    assert "bank_id = $1" in fetch_call[1]
    assert "expires_at <= $2" in fetch_call[1]
    assert "NOT EXISTS" in fetch_call[1]
    assert "source_deleted_at IS NOT NULL" in fetch_call[1]
    assert "'failed', 'cancelled'" in fetch_call[1]
    assert "LIMIT $3" in fetch_call[1]
    assert fetch_call[2] == ("bank-a", NOW, 2)
    delete_calls = [call for call in conn.calls if call[0] == "execute" and "DELETE FROM" in call[1]]
    assert len(delete_calls) == 2
    assert all("NOT EXISTS" in call[1] and "source_deleted_at IS NOT NULL" in call[1] for call in delete_calls)

    with pytest.raises(ValueError, match="between 1 and 10000"):
        await ledger.purge_expired_descriptors(conn, bank_id="bank-a", now=NOW, limit=0)


@pytest.mark.asyncio
async def test_video_segment_checkpoint_is_claim_scoped_typed_and_expiring() -> None:
    token = uuid4()
    checkpoint = _segment_checkpoint()
    write_conn = FakeConnection([{"descriptor_key": DESCRIPTOR_KEY}, None])
    ledger = MultimodalLedger()

    await ledger.checkpoint_video_segment(
        write_conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        claim_token=token,
        checkpoint=checkpoint,
        now=NOW,
        expires_at=NOW + timedelta(days=1),
    )

    insert = next(call for call in write_conn.calls if call[0] == "execute")
    assert "multimodal_segment_checkpoints" in insert[1]
    assert "segment_json" in insert[1]
    serialized_calls = repr(write_conn.calls)
    assert "base64," not in serialized_calls
    assert "encoded_bytes" not in serialized_calls

    identity = VideoSegmentIdentity(
        segment_key=SEGMENT_KEY,
        segment_id="segment-000",
        evidence_fingerprint=EVIDENCE_FP,
    )
    read_conn = FakeConnection([{"descriptor_key": DESCRIPTOR_KEY}, _segment_row()])
    restored = await ledger.get_video_segment_checkpoint(
        read_conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        claim_token=token,
        identity=identity,
        now=NOW,
    )

    assert restored == checkpoint
    read_query = read_conn.calls[1][1]
    assert "expires_at > $6" in read_query
    assert "evidence_fingerprint = $5" in read_query


@pytest.mark.asyncio
async def test_video_segment_checkpoint_rejects_payload_and_lost_claim() -> None:
    unsafe = _segment_checkpoint(text="data:image/png;base64," + "A" * 300)
    conn = FakeConnection([])
    with pytest.raises(ValueError, match="encoded media payload"):
        await MultimodalLedger().checkpoint_video_segment(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            checkpoint=unsafe,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
    assert conn.calls == []

    lost_claim = FakeConnection([None])
    with pytest.raises(LedgerConflictError, match="lost its active descriptor claim"):
        await MultimodalLedger().get_video_segment_checkpoint(
            lost_claim,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            identity=VideoSegmentIdentity(
                segment_key=SEGMENT_KEY,
                segment_id="segment-000",
                evidence_fingerprint=EVIDENCE_FP,
            ),
            now=NOW,
        )


@pytest.mark.asyncio
async def test_reusable_descriptor_requires_future_expiry_or_current_deleted_source_command() -> None:
    fresh = _descriptor_row(status="completed")
    fresh["expires_at"] = NOW + timedelta(seconds=1)
    conn = FakeConnection([fresh])

    record = await MultimodalLedger().get_reusable_descriptor(
        conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        document_id="doc-a",
        command_key=COMMAND_KEY,
        now=NOW,
    )

    assert record is not None
    query_call = conn.calls[0]
    assert query_call[2] == ("bank-a", DESCRIPTOR_KEY, "doc-a", COMMAND_KEY, NOW)
    assert "d.status = 'completed'" in query_call[1]
    assert "d.expires_at > $5" in query_call[1]
    assert "cmd.document_id = $3" in query_call[1]
    assert "cmd.command_key = $4" in query_call[1]
    assert "cmd.source_deleted_at IS NOT NULL" in query_call[1]
    assert "cmd.descriptor_key = d.descriptor_key" in query_call[1]

    # An expired physical row left behind by bounded cleanup is not returned
    # unless the database proves the command-scoped recovery exception.
    expired_unpinned = FakeConnection([None])
    assert (
        await MultimodalLedger().get_reusable_descriptor(
            expired_unpinned,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            document_id="doc-b",
            command_key="f" * 64,
            now=NOW,
        )
        is None
    )


@pytest.mark.asyncio
async def test_failed_latest_command_can_restart_but_not_after_newer_admission() -> None:
    restarted_row = _command_row(status="processing")
    conn = FakeConnection([_head_row(active=None), _command_row(status="failed"), restarted_row])
    ledger = MultimodalLedger()

    restarted = await ledger.restart_document_command(
        conn,
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        now=NOW + timedelta(minutes=1),
    )

    assert restarted.status == "processing"
    assert any("SET active_sequence = $3" in query for kind, query, _args in conn.calls if kind == "execute")

    newer_exists = FakeConnection(
        [
            _head_row(active=2, published=0, next_sequence=3),
            _command_row(status="failed", sequence=1),
        ]
    )
    with pytest.raises(LedgerConflictError, match="newer admission"):
        await ledger.restart_document_command(
            newer_exists,
            bank_id="bank-a",
            document_id="doc-a",
            command_key=COMMAND_KEY,
            now=NOW + timedelta(minutes=1),
        )


def _spec(*, operation_id: UUID | None = None) -> DocumentCommandSpec:
    return DocumentCommandSpec(
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        operation_id=operation_id or UUID("00000000-0000-0000-0000-000000000001"),
        source_storage_key="immutable/source/asset-a",
        asset_sha256=ASSET_SHA,
        descriptor_key=DESCRIPTOR_KEY,
        retain_input_fingerprint=RETAIN_FP,
        source_delete_after_retain=True,
    )


def test_descriptor_and_document_keys_are_stable_separate_and_scoped() -> None:
    descriptor = derive_descriptor_key(
        tenant_scope="tenant-a",
        bank_id="bank-a",
        asset_sha256=ASSET_SHA,
        pipeline_fingerprint=PIPELINE_FP,
    )
    repeat = derive_descriptor_key(
        tenant_scope="tenant-a",
        bank_id="bank-a",
        asset_sha256=ASSET_SHA,
        pipeline_fingerprint=PIPELINE_FP,
    )
    other_tenant = derive_descriptor_key(
        tenant_scope="tenant-b",
        bank_id="bank-a",
        asset_sha256=ASSET_SHA,
        pipeline_fingerprint=PIPELINE_FP,
    )
    other_bank = derive_descriptor_key(
        tenant_scope="tenant-a",
        bank_id="bank-b",
        asset_sha256=ASSET_SHA,
        pipeline_fingerprint=PIPELINE_FP,
    )
    retain = derive_retain_input_fingerprint(
        context={"project": "sample"},
        normalized_tags=["ui", "coding"],
        timestamp=NOW,
        explicit_strategy="chunks",
        update_intent="replace",
    )
    command = derive_document_command_key(
        tenant_scope="tenant-a",
        bank_id="bank-a",
        document_id="doc-a",
        descriptor_key=descriptor,
        retain_input_fingerprint=retain,
    )
    other_document = derive_document_command_key(
        tenant_scope="tenant-a",
        bank_id="bank-a",
        document_id="doc-b",
        descriptor_key=descriptor,
        retain_input_fingerprint=retain,
    )

    assert descriptor == repeat
    assert len(descriptor) == 64
    assert descriptor != other_tenant
    assert descriptor != other_bank
    assert command != descriptor
    assert command != other_document


def test_descriptor_key_separates_validator_hints_and_parser_policy() -> None:
    common = {
        "tenant_scope": "tenant-a",
        "bank_id": "bank-a",
        "asset_sha256": ASSET_SHA,
        "pipeline_fingerprint": PIPELINE_FP,
    }
    valid_png = derive_descriptor_key(
        **common,
        validator_hints={"declared_mime": "image/png", "extension_family": "image:image/png"},
        parser_policy={"chain": ["openai_multimodal"], "fallback_policy": "typed-not-applicable-only-v1"},
    )
    wrong_mime = derive_descriptor_key(
        **common,
        validator_hints={"declared_mime": "image/jpeg", "extension_family": "image:image/png"},
        parser_policy={"chain": ["openai_multimodal"], "fallback_policy": "typed-not-applicable-only-v1"},
    )
    legacy_first = derive_descriptor_key(
        **common,
        validator_hints={"declared_mime": "image/png", "extension_family": "image:image/png"},
        parser_policy={
            "chain": ["markitdown", "openai_multimodal"],
            "fallback_policy": "typed-not-applicable-only-v1",
        },
    )

    assert valid_png != wrong_mime
    assert valid_png != legacy_first


def test_retain_fingerprint_normalizes_tags_but_preserves_context_semantics() -> None:
    common = {
        "timestamp": NOW,
        "explicit_strategy": "chunks",
        "update_intent": "replace",
    }
    first = derive_retain_input_fingerprint(context={"project": "sample"}, normalized_tags=["b", "a", "a"], **common)
    reordered = derive_retain_input_fingerprint(context={"project": "sample"}, normalized_tags=["a", "b"], **common)
    changed_context = derive_retain_input_fingerprint(
        context={"project": "other"}, normalized_tags=["a", "b"], **common
    )

    assert first == reordered
    assert first != changed_context


def test_fingerprints_reject_binary_payloads_and_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="binary payloads are forbidden"):
        derive_retain_input_fingerprint(
            context={"payload": b"raw-frame"},
            normalized_tags=[],
            timestamp=NOW,
            explicit_strategy=None,
            update_intent="replace",
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        derive_retain_input_fingerprint(
            context=None,
            normalized_tags=[],
            timestamp=datetime(2026, 7, 22),
            explicit_strategy=None,
            update_intent="replace",
        )


def test_table_names_quote_tenant_scope_and_keep_names_constant() -> None:
    tables = LedgerTables.for_schema('tenant"blue')
    assert tables.descriptor_cache == '"tenant""blue".multimodal_descriptor_cache'
    assert tables.segment_checkpoints == '"tenant""blue".multimodal_segment_checkpoints'
    assert tables.document_heads == '"tenant""blue".multimodal_document_heads'
    assert tables.document_commands == '"tenant""blue".multimodal_document_commands'
    with pytest.raises(ValueError, match="NUL"):
        LedgerTables.for_schema("tenant\x00bad")


def test_ledger_connection_factory_uses_oracle_current_schema_contract() -> None:
    oracle_ledger = MultimodalLedger.for_connection(FakeConnection([], backend_type="oracle"), schema="TENANT")
    pg_ledger = MultimodalLedger.for_connection(FakeConnection([]), schema="tenant")

    assert oracle_ledger.tables.descriptor_cache == "multimodal_descriptor_cache"
    assert pg_ledger.tables.descriptor_cache == '"tenant".multimodal_descriptor_cache'


def test_oracle_marks_plain_canonical_markdown_bind_as_clob(monkeypatch) -> None:
    from hms_api.engine.db import oracle

    clob_type = object()

    class OracleTypes:
        DB_TYPE_CLOB = clob_type
        DB_TYPE_TIMESTAMP_TZ = object()

    class Cursor:
        sizes: dict[str, Any] | None = None

        def setinputsizes(self, **sizes: Any) -> None:
            self.sizes = sizes

    monkeypatch.setattr(oracle, "_import_oracledb", lambda: OracleTypes())
    cursor = Cursor()

    oracle.OracleConnection._apply_clob_input_sizes(
        cursor,
        "UPDATE multimodal_descriptor_cache SET canonical_markdown = :4 WHERE descriptor_key = :1",
        {"1": DESCRIPTOR_KEY, "4": "ordinary Markdown that does not begin with JSON punctuation"},
    )

    assert cursor.sizes == {"4": clob_type}


def test_oracle_deserializes_multimodal_checkpoint_json_columns() -> None:
    from hms_api.engine.db.oracle import _convert_row_from_oracle

    converted = _convert_row_from_oracle(
        ["segment_json", "provenance_metadata", "entities"],
        ('{"segment_id":"segment-000"}', '{"media_kind":"video"}', '["Editor"]'),
    )

    assert converted == (
        {"segment_id": "segment-000"},
        {"media_kind": "video"},
        ["Editor"],
    )


@pytest.mark.asyncio
async def test_descriptor_claim_is_transactional_lease_cas_with_crash_marker() -> None:
    token = uuid4()
    pending = _descriptor_row()
    processing = _descriptor_row(status="processing", claim_token=token, possible_duplicate=True)
    conn = FakeConnection([pending, processing])
    ledger = MultimodalLedger.for_schema("tenant")

    record = await ledger.claim_descriptor(
        conn,
        _identity(),
        claim_token=token,
        now=NOW,
        lease_expires_at=NOW + timedelta(minutes=5),
    )

    assert record is not None
    assert record.claim_token == token
    assert record.possible_duplicate_provider_attempt is True
    assert conn.max_transaction_depth == 1
    sql = "\n".join(query for _, query, _ in conn.calls)
    assert "ON CONFLICT (bank_id, descriptor_key) DO NOTHING" in sql
    assert "status = 'processing' AND lease_expires_at <= $5" in sql
    assert "status IN ('processing', 'failed')" in sql
    assert "provider_started_at IS NOT NULL" in sql
    assert "possible_duplicate_provider_attempt" in sql
    assert "RETURNING" in sql
    assert "raw-frame" not in repr(conn.calls)


@pytest.mark.asyncio
async def test_oracle_claim_uses_number_boolean_and_decodes_raw_claim_token() -> None:
    token = uuid4()
    pending = _descriptor_row()
    processing = _descriptor_row(status="processing", claim_token=token.bytes, possible_duplicate=1)
    conn = FakeConnection([pending, processing], backend_type="oracle")

    record = await MultimodalLedger().claim_descriptor(
        conn,
        _identity(),
        claim_token=token,
        now=NOW,
        lease_expires_at=NOW + timedelta(minutes=5),
    )

    assert record is not None and record.claim_token == token
    claim_sql = next(
        query for kind, query, _ in conn.calls if kind == "execute" and query.lstrip().startswith("UPDATE")
    )
    assert "THEN 1" in claim_sql
    assert "THEN TRUE" not in claim_sql
    assert "RETURNING" not in claim_sql
    assert conn.calls[-1][0] == "fetchrow"
    assert conn.calls[-1][1].lstrip().startswith("SELECT")


@pytest.mark.asyncio
async def test_oracle_checkpoint_selects_clob_after_cas_instead_of_returning_it() -> None:
    token = uuid4()
    markdown = "visible grounded statement with spaces\n" * 1_200
    completed = _descriptor_row(status="completed", canonical_markdown=markdown)
    completed["provenance_metadata"] = '{"media_kind":"image"}'
    completed["entities"] = '["Editor"]'
    conn = FakeConnection([completed], backend_type="oracle")

    record = await MultimodalLedger().checkpoint_descriptor(
        conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        claim_token=token,
        canonical_markdown=markdown,
        provenance_metadata={"media_kind": "image"},
        entities=["Editor"],
        now=NOW,
        expires_at=NOW + timedelta(days=1),
    )

    assert record.canonical_markdown == markdown
    update = next(query for kind, query, _ in conn.calls if kind == "execute")
    assert "canonical_markdown = $4" in update
    assert "RETURNING" not in update
    assert any(kind == "fetchrow" and query.lstrip().startswith("SELECT") for kind, query, _ in conn.calls)
    assert conn.calls[-1][1].lstrip().startswith("DELETE FROM multimodal_segment_checkpoints")


@pytest.mark.asyncio
async def test_checkpoint_accepts_only_bounded_flat_metadata_and_entity_names() -> None:
    conn = FakeConnection([])
    ledger = MultimodalLedger()
    with pytest.raises(ValueError, match="flat JSON scalars"):
        await ledger.checkpoint_descriptor(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            canonical_markdown="# media\n",
            provenance_metadata={"nested": {"forbidden": True}},
            entities=[],
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="entity names"):
        await ledger.checkpoint_descriptor(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            canonical_markdown="# media\n",
            provenance_metadata={},
            entities=[b"frame"],  # type: ignore[list-item]
            now=NOW,
            expires_at=None,
        )
    with pytest.raises(ValueError, match="encoded media payload"):
        await ledger.checkpoint_descriptor(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            canonical_markdown="visible text data:image/png;base64," + "A" * 300,
            provenance_metadata={},
            entities=[],
            now=NOW,
            expires_at=None,
        )


def _wrapped_base64_payload() -> str:
    encoded = base64.b64encode(b"wrapped-ledger-payload" * 20).decode("ascii")
    return "\n".join(encoded[offset : offset + 76] for offset in range(0, len(encoded), 76))


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["canonical_markdown", "provenance", "entity"])
@pytest.mark.parametrize(
    "payload",
    ["data:;base64,AAAA", _wrapped_base64_payload()],
    ids=["empty-media-type-data-url", "mime-wrapped-base64"],
)
async def test_checkpoint_rejects_noncanonical_encoded_payload_variants(surface: str, payload: str) -> None:
    values = {
        "canonical_markdown": "# Safe canonical descriptor",
        "provenance_metadata": {"media_kind": "image"},
        "entities": ["Editor"],
    }
    if surface == "canonical_markdown":
        values["canonical_markdown"] = payload
    elif surface == "provenance":
        values["provenance_metadata"] = {"unsafe": payload}
    else:
        values["entities"] = [payload]

    conn = FakeConnection([])
    with pytest.raises(ValueError, match="encoded media payload") as exc_info:
        await MultimodalLedger().checkpoint_descriptor(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            canonical_markdown=values["canonical_markdown"],
            provenance_metadata=values["provenance_metadata"],
            entities=values["entities"],
            now=NOW,
            expires_at=None,
        )

    assert payload not in str(exc_info.value)
    assert conn.calls == []


@pytest.mark.asyncio
async def test_checkpoint_allows_ordinary_long_text_on_all_string_surfaces() -> None:
    prose = "Readable project notes contain spaces, punctuation, and normal sentences for reviewers. " * 8
    completed = _descriptor_row(status="completed", canonical_markdown=prose)
    completed["provenance_metadata"] = json.dumps({"notes": prose})
    completed["entities"] = json.dumps([prose])
    conn = FakeConnection([completed])

    record = await MultimodalLedger().checkpoint_descriptor(
        conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        claim_token=uuid4(),
        canonical_markdown=prose,
        provenance_metadata={"notes": prose},
        entities=[prose],
        now=NOW,
        expires_at=None,
    )

    assert record.canonical_markdown == prose
    assert conn.calls


def test_document_command_source_key_rejects_data_urls() -> None:
    with pytest.raises(ValueError, match="encoded media payload"):
        DocumentCommandSpec(
            bank_id="bank-a",
            document_id="doc-a",
            command_key=COMMAND_KEY,
            operation_id=uuid4(),
            source_storage_key="data:image/png;base64,AAAA",
            asset_sha256=ASSET_SHA,
            descriptor_key=DESCRIPTOR_KEY,
            retain_input_fingerprint=RETAIN_FP,
            source_delete_after_retain=True,
        )


def test_document_command_source_key_rejects_empty_media_type_data_url() -> None:
    with pytest.raises(ValueError, match="encoded media payload"):
        DocumentCommandSpec(
            bank_id="bank-a",
            document_id="doc-a",
            command_key=COMMAND_KEY,
            operation_id=uuid4(),
            source_storage_key="data:;base64,AAAA",
            asset_sha256=ASSET_SHA,
            descriptor_key=DESCRIPTOR_KEY,
            retain_input_fingerprint=RETAIN_FP,
            source_delete_after_retain=True,
        )


@pytest.mark.asyncio
async def test_typed_record_repr_redacts_checkpoint_and_storage_locator() -> None:
    command = _command_row()
    descriptor = _descriptor_row(status="completed", canonical_markdown="private derived description")
    descriptor["provenance_metadata"] = '{"private":"metadata"}'
    descriptor["entities"] = '["private entity"]'
    command_record = await _record_from_row(command)
    descriptor_record = await MultimodalLedger().get_descriptor(
        FakeConnection([descriptor]),
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
    )
    assert descriptor_record is not None

    assert "immutable/source/asset-a" not in repr(_spec())
    assert "immutable/source/asset-a" not in repr(command_record)
    assert "private derived description" not in repr(descriptor_record)
    assert "private entity" not in repr(descriptor_record)


@pytest.mark.asyncio
async def test_document_admission_allocates_sequence_under_head_lock() -> None:
    operation_id = UUID("00000000-0000-0000-0000-000000000001")
    command = _command_row(operation_id=operation_id)
    conn = FakeConnection([_head_row(), None, command, _head_row(active=1, next_sequence=2)])
    ledger = MultimodalLedger.for_schema("tenant")

    admitted = await ledger.admit_document_command(conn, _spec(operation_id=operation_id), now=NOW)

    assert admitted.created is True
    assert admitted.command.sequence == 1
    assert conn.max_transaction_depth == 1
    sql = "\n".join(query for _, query, _ in conn.calls)
    assert "multimodal_document_heads" in sql
    assert "FOR UPDATE" in sql
    assert "next_sequence = $4" in sql
    assert "active_sequence = $3" in sql
    assert "RETURNING" in sql


@pytest.mark.asyncio
async def test_oracle_document_admission_uses_rowcount_then_select() -> None:
    command = _command_row()
    conn = FakeConnection(
        [_head_row(), None, command, _head_row(active=1, next_sequence=2)],
        backend_type="oracle",
    )

    admitted = await MultimodalLedger().admit_document_command(conn, _spec(), now=NOW)

    assert admitted.created is True
    mutation_sql = [query for kind, query, _ in conn.calls if kind == "execute"]
    assert any("INSERT INTO multimodal_document_commands" in query for query in mutation_sql)
    assert any("SET next_sequence = $4" in query for query in mutation_sql)
    assert all("RETURNING" not in query for query in mutation_sql)


@pytest.mark.asyncio
async def test_same_document_command_retry_returns_original_without_reallocation() -> None:
    existing = _command_row()
    conn = FakeConnection([_head_row(), existing])

    admitted = await MultimodalLedger().admit_document_command(conn, _spec(), now=NOW)

    assert admitted.created is False
    assert admitted.command.operation_id == existing["operation_id"]
    sql = "\n".join(query for _, query, _ in conn.calls)
    assert "INSERT INTO multimodal_document_commands" not in sql
    assert "SET next_sequence" not in sql


@pytest.mark.asyncio
async def test_same_command_retry_with_new_operation_and_source_reuses_original() -> None:
    existing = _command_row()
    conn = FakeConnection([_head_row(), existing])
    retry = DocumentCommandSpec(
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        operation_id=uuid4(),
        source_storage_key="immutable/source/retry-upload",
        asset_sha256=ASSET_SHA,
        descriptor_key=DESCRIPTOR_KEY,
        retain_input_fingerprint=RETAIN_FP,
        source_delete_after_retain=True,
    )

    admitted = await MultimodalLedger().admit_document_command(conn, retry, now=NOW)

    assert admitted.created is False
    assert admitted.command.operation_id == existing["operation_id"]
    assert admitted.command.source_storage_key == existing["source_storage_key"]
    assert "must not treat it as a blanket no-op for request metadata or tags" in (
        MultimodalLedger.admit_document_command.__doc__ or ""
    )


@pytest.mark.asyncio
async def test_same_command_key_with_different_hashed_identity_is_rejected() -> None:
    existing = _command_row()
    conn = FakeConnection([_head_row(), existing])
    collision = DocumentCommandSpec(
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        operation_id=uuid4(),
        source_storage_key="immutable/source/collision",
        asset_sha256="f" * 64,
        descriptor_key=DESCRIPTOR_KEY,
        retain_input_fingerprint=RETAIN_FP,
        source_delete_after_retain=True,
    )

    with pytest.raises(LedgerInvariantError, match="collision"):
        await MultimodalLedger().admit_document_command(conn, collision, now=NOW)


@pytest.mark.asyncio
async def test_publish_lock_allows_only_current_retaining_sequence() -> None:
    retaining = _command_row(status="retaining")
    conn = FakeConnection([_head_row(active=1), retaining])

    decision, command = await MultimodalLedger().lock_for_publish(
        conn,
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
    )

    assert decision is PublishDecision.PUBLISH
    assert command.sequence == 1
    assert all("FOR UPDATE" in query for _, query, _ in conn.calls)


@pytest.mark.asyncio
async def test_older_command_is_superseded_after_newer_admission() -> None:
    older = _command_row(status="retaining", sequence=1)
    conn = FakeConnection([_head_row(active=2, next_sequence=3), older])

    decision, _ = await MultimodalLedger().lock_for_publish(
        conn,
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
    )

    assert decision is PublishDecision.SUPERSEDED


@pytest.mark.asyncio
async def test_publish_completion_is_compare_and_swap() -> None:
    retaining_row = _command_row(status="retaining")
    completed_row = _command_row(status="completed")
    conn = FakeConnection([_head_row(active=None, published=1), completed_row])
    ledger = MultimodalLedger()
    retaining = ledger_command = await _record_from_row(retaining_row)

    completed = await ledger.complete_publish(conn, command=ledger_command, now=NOW)

    assert completed.status == "completed"
    head_update = conn.calls[0][1]
    assert "active_sequence = $3" in head_update
    assert "published_sequence < $3" in head_update


@pytest.mark.asyncio
async def test_publish_completion_rejects_lost_cas() -> None:
    conn = FakeConnection([None])
    retaining = await _record_from_row(_command_row(status="retaining"))
    with pytest.raises(LedgerConflictError, match="newer document command"):
        await MultimodalLedger().complete_publish(conn, command=retaining, now=NOW)


@pytest.mark.asyncio
async def test_terminal_transition_locks_head_before_command_to_avoid_inversion() -> None:
    failed = _command_row(status="failed")
    conn = FakeConnection([_head_row(), failed])

    command = await MultimodalLedger().mark_document_terminal(
        conn,
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        status="failed",
        now=NOW,
    )

    assert command.status == "failed"
    assert "multimodal_document_heads" in conn.calls[0][1]
    assert "FOR UPDATE" in conn.calls[0][1]
    assert "UPDATE multimodal_document_commands" in conn.calls[1][1]
    assert "UPDATE multimodal_document_heads" in conn.calls[2][1]


@pytest.mark.asyncio
async def test_terminal_transition_can_cas_expected_child_identity() -> None:
    child_id = UUID("00000000-0000-0000-0000-000000000002")
    failed = _command_row(status="failed")
    failed["child_retain_operation_id"] = child_id
    conn = FakeConnection([_head_row(), failed])

    command = await MultimodalLedger().mark_document_terminal(
        conn,
        bank_id="bank-a",
        document_id="doc-a",
        command_key=COMMAND_KEY,
        status="failed",
        now=NOW,
        expected_sequence=1,
        expected_child_retain_operation_id=child_id,
    )

    assert command.status == "failed"
    command_update = conn.calls[1]
    assert "sequence = $6" in command_update[1]
    assert "child_retain_operation_id = $7" in command_update[1]
    assert command_update[2][-2:] == (1, child_id)


@pytest.mark.asyncio
async def test_descriptor_failure_and_command_terminalization_share_claim_fence() -> None:
    claim_token = uuid4()
    operation_id = UUID("00000000-0000-0000-0000-000000000001")
    conn = FakeConnection(
        [
            # Active descriptor lock, descriptor failure RETURNING row,
            # then the head/command rows locked by mark_document_terminal and
            # its terminal CAS RETURNING row.
            {"descriptor_key": DESCRIPTOR_KEY},
            _descriptor_row(status="failed"),
            _head_row(active=1),
            _command_row(status="failed", operation_id=operation_id),
        ]
    )

    command = await MultimodalLedger().fail_descriptor_and_mark_document_terminal(
        conn,
        bank_id="bank-a",
        descriptor_key=DESCRIPTOR_KEY,
        claim_token=claim_token,
        document_id="doc-a",
        command_key=COMMAND_KEY,
        status="failed",
        now=NOW,
        expected_sequence=1,
        expected_operation_id=operation_id,
    )

    assert command.status == "failed"
    assert conn.transaction_depth == 0
    assert conn.max_transaction_depth >= 2
    assert "lease_expires_at > $4" in conn.calls[0][1]
    command_update = next(
        query
        for kind, query, _ in conn.calls
        if kind == "fetchrow" and query.lstrip().startswith("UPDATE multimodal_document_commands")
    )
    assert "sequence = $6" in command_update
    assert "operation_id = $7" in command_update


@pytest.mark.asyncio
async def test_descriptor_failure_and_command_terminalization_reject_lost_claim() -> None:
    conn = FakeConnection([None])
    with pytest.raises(LedgerConflictError, match="active descriptor claim"):
        await MultimodalLedger().fail_descriptor_and_mark_document_terminal(
            conn,
            bank_id="bank-a",
            descriptor_key=DESCRIPTOR_KEY,
            claim_token=uuid4(),
            document_id="doc-a",
            command_key=COMMAND_KEY,
            status="failed",
            now=NOW,
            expected_sequence=1,
            expected_operation_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    assert conn.transaction_depth == 0
    assert len(conn.calls) == 1


async def _record_from_row(row: dict[str, Any]):
    """Use the public read primitive to build a typed command for CAS tests."""

    conn = FakeConnection([row])
    record = await MultimodalLedger().get_document_command(
        conn,
        bank_id=row["bank_id"],
        document_id=row["document_id"],
        command_key=row["command_key"],
    )
    assert record is not None
    return record
