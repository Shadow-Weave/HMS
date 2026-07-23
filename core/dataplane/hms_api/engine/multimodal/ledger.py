"""Durable multimodal idempotency and document-ordering primitives.

This module is intentionally independent from ``MemoryEngine``.  It owns only
deterministic key derivation, typed ledger records, and transaction-friendly
SQL primitives.  Callers remain responsible for storing immutable source
objects, running the provider/parser, and performing the existing retain.

Security boundary: no public type or SQL statement accepts media bytes, base64
data URLs, or provider request/response bodies.  The descriptor checkpoint is
limited to canonical Markdown plus bounded, flat provenance and entity names.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from .checkpoints import VideoSegmentCheckpoint, VideoSegmentIdentity
from .security import contains_encoded_media_payload

if TYPE_CHECKING:
    from hms_api.engine.db.base import DatabaseConnection
    from hms_api.engine.db.result import ResultRow

DESCRIPTOR_CACHE_TABLE = "multimodal_descriptor_cache"
SEGMENT_CHECKPOINTS_TABLE = "multimodal_segment_checkpoints"
DOCUMENT_HEADS_TABLE = "multimodal_document_heads"
DOCUMENT_COMMANDS_TABLE = "multimodal_document_commands"

KEY_VERSION = "multimodal-ledger-v1"
MAX_CANONICAL_MARKDOWN_CHARS = 2_000_000
MAX_FLAT_METADATA_KEYS = 128
MAX_FLAT_METADATA_BYTES = 65_536
MAX_ENTITY_COUNT = 256
MAX_ENTITY_CHARS = 1_024
MAX_SOURCE_STORAGE_KEY_CHARS = 512
MAX_BANK_ID_CHARS = 256
MAX_DOCUMENT_ID_CHARS = 512
MAX_SEGMENT_CHECKPOINT_JSON_BYTES = 512_000

DescriptorStatus = Literal["pending", "processing", "completed", "failed"]
DocumentCommandStatus = Literal[
    "pending",
    "processing",
    "retaining",
    "completed",
    "failed",
    "superseded",
    "cancelled",
]


class LedgerError(RuntimeError):
    """Base class for sanitized ledger failures."""


class LedgerInvariantError(LedgerError):
    """A persisted row or caller-supplied identity violates the ledger contract."""


class LedgerConflictError(LedgerError):
    """A compare-and-swap transition lost to another command or claim."""


class PublishDecision(str, Enum):
    """Result of locking a logical document before the existing retain writes it."""

    PUBLISH = "publish"
    ALREADY_PUBLISHED = "already_published"
    SUPERSEDED = "superseded"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class LedgerTables:
    """Schema-qualified immutable table names.

    The schema is quoted as one identifier.  Table names are constants rather
    than caller input, preventing SQL identifier injection at this boundary.
    """

    descriptor_cache: str
    segment_checkpoints: str
    document_heads: str
    document_commands: str

    @classmethod
    def for_schema(cls, schema: str | None = None) -> "LedgerTables":
        prefix = ""
        if schema:
            if "\x00" in schema:
                raise ValueError("schema name cannot contain a NUL character")
            prefix = f'"{schema.replace(chr(34), chr(34) * 2)}".'
        return cls(
            descriptor_cache=f"{prefix}{DESCRIPTOR_CACHE_TABLE}",
            segment_checkpoints=f"{prefix}{SEGMENT_CHECKPOINTS_TABLE}",
            document_heads=f"{prefix}{DOCUMENT_HEADS_TABLE}",
            document_commands=f"{prefix}{DOCUMENT_COMMANDS_TABLE}",
        )


@dataclass(frozen=True)
class DescriptorIdentity:
    """System-derived identity for reusable descriptor work."""

    bank_id: str
    descriptor_key: str
    asset_sha256: str
    pipeline_fingerprint: str

    def __post_init__(self) -> None:
        _require_text("bank_id", self.bank_id, max_chars=MAX_BANK_ID_CHARS)
        _require_digest("descriptor_key", self.descriptor_key)
        _require_digest("asset_sha256", self.asset_sha256)
        _require_digest("pipeline_fingerprint", self.pipeline_fingerprint)


@dataclass(frozen=True)
class DescriptorRecord:
    bank_id: str
    descriptor_key: str
    asset_sha256: str
    pipeline_fingerprint: str
    status: DescriptorStatus
    claim_token: UUID | None
    lease_expires_at: datetime | None
    provider_started_at: datetime | None
    possible_duplicate_provider_attempt: bool
    canonical_markdown: str | None = field(repr=False)
    provenance_metadata: dict[str, str | int | float | bool | None] = field(repr=False)
    entities: list[str] = field(repr=False)
    checkpointed_at: datetime | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DocumentHeadRecord:
    bank_id: str
    document_id: str
    next_sequence: int
    published_sequence: int
    active_sequence: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DocumentCommandSpec:
    """Immutable admission fields for one logical document update.

    ``source_storage_key`` is an opaque pointer to an immutable, operation-
    scoped object.  It must never contain the media payload itself.
    """

    bank_id: str
    document_id: str
    command_key: str
    operation_id: UUID
    source_storage_key: str = field(repr=False)
    asset_sha256: str
    descriptor_key: str
    retain_input_fingerprint: str
    source_delete_after_retain: bool

    def __post_init__(self) -> None:
        _require_text("bank_id", self.bank_id, max_chars=MAX_BANK_ID_CHARS)
        _require_text("document_id", self.document_id, max_chars=MAX_DOCUMENT_ID_CHARS)
        _require_text("source_storage_key", self.source_storage_key)
        if len(self.source_storage_key) > MAX_SOURCE_STORAGE_KEY_CHARS:
            raise ValueError("source_storage_key exceeds the ledger limit")
        if "\x00" in self.source_storage_key or "\r" in self.source_storage_key or "\n" in self.source_storage_key:
            raise ValueError("source_storage_key contains a forbidden control character")
        _reject_encoded_payload("source_storage_key", self.source_storage_key, reject_long_token=False)
        _require_digest("command_key", self.command_key)
        _require_digest("asset_sha256", self.asset_sha256)
        _require_digest("descriptor_key", self.descriptor_key)
        _require_digest("retain_input_fingerprint", self.retain_input_fingerprint)


@dataclass(frozen=True)
class DocumentCommandRecord:
    bank_id: str
    document_id: str
    command_key: str
    sequence: int
    operation_id: UUID
    source_storage_key: str = field(repr=False)
    asset_sha256: str
    descriptor_key: str
    retain_input_fingerprint: str
    status: DocumentCommandStatus
    child_retain_operation_id: UUID | None
    source_delete_after_retain: bool
    source_deleted_at: datetime | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class DocumentAdmission:
    """Result of command-row admission.

    ``created=False`` means only that the durable command identity and sequence
    were reused.  It is not permission to discard the current request's
    metadata/tags.  Because the public contract excludes arbitrary metadata
    from ``retain_input_fingerprint``, the integration must still coordinate
    an idempotent document metadata refresh (or follow the in-flight owner).
    Descriptor reuse and document-update completion are separate decisions.
    """

    command: DocumentCommandRecord
    created: bool


def _require_text(name: str, value: str, *, max_chars: int | None = None) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if max_chars is not None and len(value) > max_chars:
        raise ValueError(f"{name} exceeds the ledger limit")


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _canonical_value(value: Any) -> Any:
    """Convert supported values to a deterministic, JSON-safe tree."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint values cannot contain NaN or infinity")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fingerprint datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError("binary payloads are forbidden in ledger fingerprints")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("fingerprint mapping keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        canonical = [_canonical_value(item) for item in value]
        return sorted(canonical, key=_canonical_json)
    if isinstance(value, Sequence):
        return [_canonical_value(item) for item in value]
    raise ValueError(f"unsupported fingerprint value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _domain_digest(domain: str, payload: Mapping[str, Any]) -> str:
    envelope = {
        "domain": f"{KEY_VERSION}:{domain}",
        "payload": _canonical_value(payload),
    }
    return hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()


def derive_descriptor_key(
    *,
    tenant_scope: str | None,
    bank_id: str,
    asset_sha256: str,
    pipeline_fingerprint: str,
    validator_hints: Mapping[str, Any] | None = None,
    parser_policy: Mapping[str, Any] | None = None,
) -> str:
    """Derive a tenant/bank-scoped cache identity for model descriptor work.

    Validation hints and ordered parser policy are part of descriptor identity,
    not merely document-command identity.  A completed checkpoint produced
    under a valid PNG hint or a multimodal-first chain must never satisfy a
    later command whose hint/policy could select a different validator or
    fallback winner.  The values are canonicalized and contain no media bytes.
    """

    _require_text("bank_id", bank_id, max_chars=MAX_BANK_ID_CHARS)
    _require_digest("asset_sha256", asset_sha256)
    _require_digest("pipeline_fingerprint", pipeline_fingerprint)
    return _domain_digest(
        "descriptor",
        {
            "tenant_scope": tenant_scope if tenant_scope is not None else {"default": True},
            "bank_id": bank_id,
            "asset_sha256": asset_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "validator_hints": validator_hints or {},
            "parser_policy": parser_policy or {},
        },
    )


def derive_retain_input_fingerprint(
    *,
    context: Any,
    normalized_tags: Sequence[str],
    timestamp: datetime | str | None,
    explicit_strategy: str | None,
    update_intent: str,
) -> str:
    """Hash document-update semantics without storing raw context or tags.

    Tags are set-like after normalization, so their order and duplicates do
    not create a new command.  Context remains structured and order-sensitive
    for arrays, matching JSON semantics.
    """

    _require_text("update_intent", update_intent)
    if isinstance(normalized_tags, str):
        raise ValueError("normalized_tags must be a sequence of strings")
    if any(not isinstance(tag, str) or not tag for tag in normalized_tags):
        raise ValueError("normalized_tags must contain only non-empty strings")
    return _domain_digest(
        "retain-input",
        {
            "context": context,
            "normalized_tags": sorted(set(normalized_tags)),
            "timestamp": timestamp,
            "explicit_strategy": explicit_strategy,
            "update_intent": update_intent,
        },
    )


def derive_document_command_key(
    *,
    tenant_scope: str | None,
    bank_id: str,
    document_id: str,
    descriptor_key: str,
    retain_input_fingerprint: str,
) -> str:
    """Derive a document update identity distinct from descriptor caching."""

    _require_text("bank_id", bank_id, max_chars=MAX_BANK_ID_CHARS)
    _require_text("document_id", document_id, max_chars=MAX_DOCUMENT_ID_CHARS)
    _require_digest("descriptor_key", descriptor_key)
    _require_digest("retain_input_fingerprint", retain_input_fingerprint)
    return _domain_digest(
        "document-command",
        {
            "tenant_scope": tenant_scope if tenant_scope is not None else {"default": True},
            "bank_id": bank_id,
            "document_id": document_id,
            "descriptor_key": descriptor_key,
            "retain_input_fingerprint": retain_input_fingerprint,
        },
    )


def _db_bool(conn: DatabaseConnection, value: bool) -> bool | int:
    return int(value) if conn.backend_type == "oracle" else value


def _sql_bool(conn: DatabaseConnection, value: bool) -> str:
    if conn.backend_type == "oracle":
        return "1" if value else "0"
    return "TRUE" if value else "FALSE"


def _json_for_db(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _db_bool_value(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false"}:
            return False
        if normalized in {"1", "true"}:
            return True
    return bool(value)


def _affected_rows(status: str) -> int:
    """Parse the PG-compatible status returned by both database wrappers."""

    try:
        return int(status.rsplit(" ", 1)[-1])
    except (TypeError, ValueError) as exc:
        raise LedgerInvariantError("database mutation returned an unknown row-count status") from exc


def _reject_encoded_payload(label: str, value: str, *, reject_long_token: bool = True) -> None:
    if contains_encoded_media_payload(value, reject_long_token=reject_long_token):
        raise ValueError(f"{label} cannot contain an encoded media payload")


def _flat_metadata(value: Mapping[str, Any]) -> dict[str, str | int | float | bool | None]:
    if len(value) > MAX_FLAT_METADATA_KEYS:
        raise ValueError("descriptor provenance metadata has too many keys")
    result: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("descriptor provenance keys must be non-empty strings")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("descriptor provenance values cannot contain NaN or infinity")
        if item is not None and not isinstance(item, str | int | float | bool):
            raise ValueError("descriptor provenance must be flat JSON scalars")
        if isinstance(item, str):
            _reject_encoded_payload("descriptor provenance", item)
        result[key] = item
    if len(_json_for_db(result).encode("utf-8")) > MAX_FLAT_METADATA_BYTES:
        raise ValueError("descriptor provenance metadata exceeds the byte limit")
    return result


def _entity_names(value: Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raise ValueError("descriptor entities must be a sequence of names")
    if len(value) > MAX_ENTITY_COUNT:
        raise ValueError("descriptor entity count exceeds the limit")
    result: list[str] = []
    seen: set[str] = set()
    for entity in value:
        if not isinstance(entity, str) or not entity or len(entity) > MAX_ENTITY_CHARS:
            raise ValueError("descriptor entity names must be non-empty bounded strings")
        _reject_encoded_payload("descriptor entity", entity)
        if entity not in seen:
            seen.add(entity)
            result.append(entity)
    return result


def _row_json(conn: DatabaseConnection, row: ResultRow, key: str, default: Any) -> Any:
    value = row.get(key)
    if value is None:
        return default
    parsed = conn.parse_json(value)
    return default if parsed is None else parsed


def _uuid_or_none(value: Any) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    if isinstance(value, bytes) and len(value) == 16:
        return UUID(bytes=value)
    return UUID(str(value))


def _required_uuid(value: Any, field: str) -> UUID:
    parsed = _uuid_or_none(value)
    if parsed is None:
        raise LedgerInvariantError(f"{field} cannot be null")
    return parsed


def _descriptor_from_row(conn: DatabaseConnection, row: ResultRow) -> DescriptorRecord:
    metadata = _row_json(conn, row, "provenance_metadata", {})
    entities = _row_json(conn, row, "entities", [])
    if not isinstance(metadata, dict) or not isinstance(entities, list):
        raise LedgerInvariantError("descriptor checkpoint JSON has an invalid shape")
    return DescriptorRecord(
        bank_id=str(row["bank_id"]),
        descriptor_key=str(row["descriptor_key"]),
        asset_sha256=str(row["asset_sha256"]),
        pipeline_fingerprint=str(row["pipeline_fingerprint"]),
        status=row["status"],
        claim_token=_uuid_or_none(row.get("claim_token")),
        lease_expires_at=row.get("lease_expires_at"),
        provider_started_at=row.get("provider_started_at"),
        possible_duplicate_provider_attempt=_db_bool_value(row["possible_duplicate_provider_attempt"]),
        canonical_markdown=row.get("canonical_markdown"),
        provenance_metadata=metadata,
        entities=entities,
        checkpointed_at=row.get("checkpointed_at"),
        expires_at=row.get("expires_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _segment_checkpoint_from_row(
    conn: DatabaseConnection,
    row: ResultRow,
) -> VideoSegmentCheckpoint:
    value = _row_json(conn, row, "segment_json", None)
    if not isinstance(value, dict):
        raise LedgerInvariantError("video segment checkpoint JSON has an invalid shape")
    try:
        return VideoSegmentCheckpoint(
            segment_key=str(row["segment_key"]),
            segment_id=str(row["segment_id"]),
            evidence_fingerprint=str(row["evidence_fingerprint"]),
            value=value,
            provider=str(row["provider"]),
            configured_model=str(row["configured_model"]),
            resolved_model=str(row["resolved_model"]) if row.get("resolved_model") is not None else None,
            request_id=str(row["provider_request_id"]) if row.get("provider_request_id") is not None else None,
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            logical_calls=int(row["logical_calls"]),
            physical_attempts=int(row["physical_attempts"]),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerInvariantError("video segment checkpoint failed local validation") from exc


def _head_from_row(row: ResultRow) -> DocumentHeadRecord:
    return DocumentHeadRecord(
        bank_id=str(row["bank_id"]),
        document_id=str(row["document_id"]),
        next_sequence=int(row["next_sequence"]),
        published_sequence=int(row["published_sequence"]),
        active_sequence=int(row["active_sequence"]) if row.get("active_sequence") is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _command_from_row(row: ResultRow) -> DocumentCommandRecord:
    return DocumentCommandRecord(
        bank_id=str(row["bank_id"]),
        document_id=str(row["document_id"]),
        command_key=str(row["command_key"]),
        sequence=int(row["sequence"]),
        operation_id=_required_uuid(row["operation_id"], "operation_id"),
        source_storage_key=str(row["source_storage_key"]),
        asset_sha256=str(row["asset_sha256"]),
        descriptor_key=str(row["descriptor_key"]),
        retain_input_fingerprint=str(row["retain_input_fingerprint"]),
        status=row["status"],
        child_retain_operation_id=_uuid_or_none(row.get("child_retain_operation_id")),
        source_delete_after_retain=_db_bool_value(row["source_delete_after_retain"]),
        source_deleted_at=row.get("source_deleted_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
    )


_DESCRIPTOR_COLUMNS = """
    bank_id, descriptor_key, asset_sha256, pipeline_fingerprint, status,
    claim_token, lease_expires_at, provider_started_at,
    possible_duplicate_provider_attempt, canonical_markdown,
    provenance_metadata, entities, checkpointed_at, expires_at,
    created_at, updated_at
""".strip()

_SEGMENT_CHECKPOINT_COLUMNS = """
    bank_id, descriptor_key, segment_key, segment_id, evidence_fingerprint,
    segment_json, provider, configured_model, resolved_model,
    provider_request_id, input_tokens, output_tokens, logical_calls,
    physical_attempts, checkpointed_at, expires_at, created_at, updated_at
""".strip()

_HEAD_COLUMNS = """
    bank_id, document_id, next_sequence, published_sequence, active_sequence,
    created_at, updated_at
""".strip()

_COMMAND_COLUMNS = """
    bank_id, document_id, command_key, sequence, operation_id,
    source_storage_key, asset_sha256, descriptor_key,
    retain_input_fingerprint, status, child_retain_operation_id,
    source_delete_after_retain, source_deleted_at, created_at, updated_at,
    completed_at
""".strip()


class MultimodalLedger:
    """Database primitives for durable descriptor claims and ordered commands.

    Every method that performs multiple statements opens a transaction (or a
    backend savepoint when the caller already owns one).  ``lock_for_publish``
    and ``complete_publish`` are the intentional exception: callers must invoke
    both inside the *same outer transaction* as the existing document retain
    write, so the row lock and published sequence commit atomically with it.
    """

    def __init__(self, tables: LedgerTables | None = None) -> None:
        self.tables = tables or LedgerTables.for_schema()

    @classmethod
    def for_schema(cls, schema: str | None = None) -> "MultimodalLedger":
        return cls(LedgerTables.for_schema(schema))

    @classmethod
    def for_connection(
        cls,
        conn: DatabaseConnection,
        *,
        schema: str | None = None,
    ) -> "MultimodalLedger":
        """Build table names using the repository's backend schema contract.

        PostgreSQL queries are explicitly schema-qualified.  Oracle tenant
        routing is established by ``ALTER SESSION SET CURRENT_SCHEMA`` in the
        backend, so its runtime SQL deliberately uses bare table names.
        """

        return cls.for_schema(None if conn.backend_type == "oracle" else schema)

    async def get_descriptor(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
    ) -> DescriptorRecord | None:
        row = await conn.fetchrow(
            f"""SELECT {_DESCRIPTOR_COLUMNS}
                FROM {self.tables.descriptor_cache}
                WHERE bank_id = $1 AND descriptor_key = $2""",
            bank_id,
            descriptor_key,
        )
        return _descriptor_from_row(conn, row) if row is not None else None

    async def get_reusable_descriptor(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        document_id: str,
        command_key: str,
        now: datetime,
    ) -> DescriptorRecord | None:
        """Return a completed checkpoint only while it is safe to reuse.

        ``purge_expired_descriptors`` is intentionally bounded, so physical row
        presence is not cache validity.  An ordinary cache hit therefore
        requires a strictly future ``expires_at`` value.  The sole exception is
        the current recoverable document command after its immutable source was
        verifiably deleted: for that command, the descriptor is the durable
        checkpoint needed to retry a failed child retain without another
        provider call or inaccessible source bytes.

        This exception is command-scoped; an expired row pinned for one failed
        command cannot become a general cache hit for another document update.
        """

        row = await conn.fetchrow(
            f"""SELECT {_DESCRIPTOR_COLUMNS}
                FROM {self.tables.descriptor_cache} d
                WHERE d.bank_id = $1
                  AND d.descriptor_key = $2
                  AND d.status = 'completed'
                  AND (
                      d.expires_at > $5
                      OR EXISTS (
                          SELECT 1
                          FROM {self.tables.document_commands} cmd
                          WHERE cmd.bank_id = d.bank_id
                            AND cmd.document_id = $3
                            AND cmd.command_key = $4
                            AND cmd.descriptor_key = d.descriptor_key
                            AND cmd.source_deleted_at IS NOT NULL
                            AND cmd.status IN (
                                'pending', 'processing', 'retaining',
                                'failed', 'cancelled'
                            )
                      )
                  )""",
            bank_id,
            descriptor_key,
            document_id,
            command_key,
            now,
        )
        return _descriptor_from_row(conn, row) if row is not None else None

    async def purge_expired_descriptors(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        now: datetime,
        limit: int = 100,
    ) -> int:
        """Delete a bounded batch of expired derived-text checkpoints.

        Cleanup is deliberately scoped to the current bank and invoked during
        ordinary multimodal admission, so it requires no global tenant scan or
        separate privileged scheduler.  Bank deletion remains the complete
        lifecycle backstop through the migration's cascading foreign key.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > 10_000:
            raise ValueError("descriptor cleanup limit must be between 1 and 10000")
        deleted = 0
        async with conn.transaction():
            rows = await conn.fetch(
                f"""SELECT d.descriptor_key
                    FROM {self.tables.descriptor_cache} d
                    WHERE d.bank_id = $1
                      AND d.status = 'completed'
                      AND d.expires_at IS NOT NULL
                      AND d.expires_at <= $2
                      AND NOT EXISTS (
                          SELECT 1
                          FROM {self.tables.document_commands} cmd
                          WHERE cmd.bank_id = d.bank_id
                            AND cmd.descriptor_key = d.descriptor_key
                            AND cmd.source_deleted_at IS NOT NULL
                            AND cmd.status IN (
                                'pending', 'processing', 'retaining',
                                'failed', 'cancelled'
                            )
                      )
                    ORDER BY d.expires_at, d.descriptor_key
                    LIMIT $3""",
                bank_id,
                now,
                limit,
            )
            for row in rows:
                status = await conn.execute(
                    f"""DELETE FROM {self.tables.descriptor_cache} d
                        WHERE d.bank_id = $1
                          AND d.descriptor_key = $2
                          AND d.status = 'completed'
                          AND d.expires_at IS NOT NULL
                          AND d.expires_at <= $3
                          AND NOT EXISTS (
                              SELECT 1
                              FROM {self.tables.document_commands} cmd
                              WHERE cmd.bank_id = d.bank_id
                                AND cmd.descriptor_key = d.descriptor_key
                                AND cmd.source_deleted_at IS NOT NULL
                                AND cmd.status IN (
                                    'pending', 'processing', 'retaining',
                                    'failed', 'cancelled'
                                )
                          )""",
                    bank_id,
                    str(row["descriptor_key"]),
                    now,
                )
                deleted += _affected_rows(status)
        return deleted

    async def _mutate_descriptor(
        self,
        conn: DatabaseConnection,
        query: str,
        *args: Any,
        bank_id: str,
        descriptor_key: str,
    ) -> DescriptorRecord | None:
        """Run descriptor DML without returning Oracle CLOBs into VARCHAR."""

        if conn.backend_type != "oracle":
            row = await conn.fetchrow(f"{query} RETURNING {_DESCRIPTOR_COLUMNS}", *args)
            return _descriptor_from_row(conn, row) if row is not None else None
        status = await conn.execute(query, *args)
        if _affected_rows(status) != 1:
            return None
        return await self.get_descriptor(conn, bank_id=bank_id, descriptor_key=descriptor_key)

    async def _ensure_descriptor(
        self,
        conn: DatabaseConnection,
        identity: DescriptorIdentity,
        *,
        now: datetime,
    ) -> DescriptorRecord:
        await conn.execute(
            f"""INSERT INTO {self.tables.descriptor_cache}
                (bank_id, descriptor_key, asset_sha256, pipeline_fingerprint,
                 status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, 'pending', $5, $5)
                ON CONFLICT (bank_id, descriptor_key) DO NOTHING""",
            identity.bank_id,
            identity.descriptor_key,
            identity.asset_sha256,
            identity.pipeline_fingerprint,
            now,
        )
        record = await self.get_descriptor(
            conn,
            bank_id=identity.bank_id,
            descriptor_key=identity.descriptor_key,
        )
        if record is None:
            raise LedgerInvariantError("descriptor row disappeared during admission")
        if record.asset_sha256 != identity.asset_sha256 or record.pipeline_fingerprint != identity.pipeline_fingerprint:
            raise LedgerInvariantError("descriptor key collision with different immutable identity")
        return record

    async def claim_descriptor(
        self,
        conn: DatabaseConnection,
        identity: DescriptorIdentity,
        *,
        claim_token: UUID,
        now: datetime,
        lease_expires_at: datetime,
    ) -> DescriptorRecord | None:
        """Acquire the sole active descriptor claim, or return ``None``.

        An expired lease is reclaimable.  If its worker had crossed the
        provider-start checkpoint, the durable duplicate-attempt bit is set
        before any retry; this accurately models the external call as
        at-least-once without persisting its request body.
        """

        _require_aware_order(now, lease_expires_at, "descriptor lease")
        duplicate_true = _sql_bool(conn, True)
        async with conn.transaction():
            await self._ensure_descriptor(conn, identity, now=now)
            record = await self._mutate_descriptor(
                conn,
                f"""UPDATE {self.tables.descriptor_cache}
                    SET status = 'processing',
                        claim_token = $3,
                        lease_expires_at = $4,
                        provider_started_at = NULL,
                        possible_duplicate_provider_attempt = CASE
                            WHEN status IN ('processing', 'failed')
                                 AND provider_started_at IS NOT NULL
                                 AND checkpointed_at IS NULL
                            THEN {duplicate_true}
                            ELSE possible_duplicate_provider_attempt
                        END,
                        canonical_markdown = NULL,
                        provenance_metadata = '{{}}'::jsonb,
                        entities = '[]'::jsonb,
                        checkpointed_at = NULL,
                        expires_at = NULL,
                        updated_at = $5
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND (
                          status IN ('pending', 'failed')
                          OR (status = 'processing' AND lease_expires_at <= $5)
                          OR (status = 'completed' AND expires_at IS NOT NULL AND expires_at <= $5)
                      )
                    """,
                identity.bank_id,
                identity.descriptor_key,
                claim_token,
                lease_expires_at,
                now,
                bank_id=identity.bank_id,
                descriptor_key=identity.descriptor_key,
            )
            return record

    async def mark_provider_started(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        now: datetime,
    ) -> DescriptorRecord:
        async with conn.transaction():
            record = await self._mutate_descriptor(
                conn,
                f"""UPDATE {self.tables.descriptor_cache}
                    SET provider_started_at = COALESCE(provider_started_at, $4),
                        updated_at = $4
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND status = 'processing' AND claim_token = $3
                      AND lease_expires_at > $4""",
                bank_id,
                descriptor_key,
                claim_token,
                now,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
            )
            if record is None:
                raise LedgerConflictError("descriptor provider-start checkpoint lost its active claim")
            return record

    async def renew_descriptor_lease(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        now: datetime,
        lease_expires_at: datetime,
    ) -> DescriptorRecord:
        _require_aware_order(now, lease_expires_at, "descriptor lease")
        async with conn.transaction():
            record = await self._mutate_descriptor(
                conn,
                f"""UPDATE {self.tables.descriptor_cache}
                    SET lease_expires_at = $5, updated_at = $4
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND status = 'processing' AND claim_token = $3
                      AND lease_expires_at > $4""",
                bank_id,
                descriptor_key,
                claim_token,
                now,
                lease_expires_at,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
            )
            if record is None:
                raise LedgerConflictError("descriptor lease can no longer be renewed")
            return record

    async def _lock_active_descriptor_claim(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        now: datetime,
    ) -> None:
        row = await conn.fetchrow(
            f"""SELECT descriptor_key
                FROM {self.tables.descriptor_cache}
                WHERE bank_id = $1 AND descriptor_key = $2
                  AND status = 'processing' AND claim_token = $3
                  AND lease_expires_at > $4
                FOR UPDATE""",
            bank_id,
            descriptor_key,
            claim_token,
            now,
        )
        if row is None:
            raise LedgerConflictError("video segment checkpoint lost its active descriptor claim")

    async def get_video_segment_checkpoint(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        identity: VideoSegmentIdentity,
        now: datetime,
    ) -> VideoSegmentCheckpoint | None:
        """Load one unexpired map result for the current descriptor owner."""

        _require_text("bank_id", bank_id, max_chars=MAX_BANK_ID_CHARS)
        _require_digest("descriptor_key", descriptor_key)
        async with conn.transaction():
            await self._lock_active_descriptor_claim(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=claim_token,
                now=now,
            )
            row = await conn.fetchrow(
                f"""SELECT {_SEGMENT_CHECKPOINT_COLUMNS}
                    FROM {self.tables.segment_checkpoints}
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND segment_key = $3 AND segment_id = $4
                      AND evidence_fingerprint = $5 AND expires_at > $6""",
                bank_id,
                descriptor_key,
                identity.segment_key,
                identity.segment_id,
                identity.evidence_fingerprint,
                now,
            )
        return _segment_checkpoint_from_row(conn, row) if row is not None else None

    async def checkpoint_video_segment(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        checkpoint: VideoSegmentCheckpoint,
        now: datetime,
        expires_at: datetime,
    ) -> None:
        """Durably store one validated map result without any frame payload."""

        _require_text("bank_id", bank_id, max_chars=MAX_BANK_ID_CHARS)
        _require_digest("descriptor_key", descriptor_key)
        _require_aware_order(now, expires_at, "video segment checkpoint expiry")
        for label, value in (
            ("segment provider", checkpoint.provider),
            ("segment configured model", checkpoint.configured_model),
            ("segment resolved model", checkpoint.resolved_model),
            ("segment provider request ID", checkpoint.request_id),
        ):
            if value is not None:
                _reject_encoded_payload(label, value)
        segment_value = checkpoint.value.model_dump(mode="json")
        segment_json = _json_for_db(segment_value)
        if len(segment_json.encode("utf-8")) > MAX_SEGMENT_CHECKPOINT_JSON_BYTES:
            raise ValueError("video segment checkpoint exceeds the JSON byte limit")
        _reject_encoded_payload("video segment checkpoint", segment_json)

        async with conn.transaction():
            await self._lock_active_descriptor_claim(
                conn,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
                claim_token=claim_token,
                now=now,
            )
            existing = await conn.fetchrow(
                f"""SELECT segment_id, evidence_fingerprint
                    FROM {self.tables.segment_checkpoints}
                    WHERE bank_id = $1 AND descriptor_key = $2 AND segment_key = $3
                    FOR UPDATE""",
                bank_id,
                descriptor_key,
                checkpoint.segment_key,
            )
            values = (
                checkpoint.segment_id,
                checkpoint.evidence_fingerprint,
                segment_json,
                checkpoint.provider,
                checkpoint.configured_model,
                checkpoint.resolved_model,
                checkpoint.request_id,
                checkpoint.input_tokens,
                checkpoint.output_tokens,
                checkpoint.logical_calls,
                checkpoint.physical_attempts,
                now,
                expires_at,
            )
            if existing is None:
                await conn.execute(
                    f"""INSERT INTO {self.tables.segment_checkpoints}
                        (bank_id, descriptor_key, segment_key, segment_id,
                         evidence_fingerprint, segment_json, provider,
                         configured_model, resolved_model, provider_request_id,
                         input_tokens, output_tokens, logical_calls,
                         physical_attempts, checkpointed_at, expires_at,
                         created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8,
                                $9, $10, $11, $12, $13, $14, $15, $16,
                                $15, $15)""",
                    bank_id,
                    descriptor_key,
                    checkpoint.segment_key,
                    *values,
                )
            else:
                if (
                    str(existing["segment_id"]) != checkpoint.segment_id
                    or str(existing["evidence_fingerprint"]) != checkpoint.evidence_fingerprint
                ):
                    raise LedgerInvariantError("video segment key collision with different immutable identity")
                await conn.execute(
                    f"""UPDATE {self.tables.segment_checkpoints}
                        SET segment_json = $4::jsonb, provider = $5,
                            configured_model = $6, resolved_model = $7,
                            provider_request_id = $8, input_tokens = $9,
                            output_tokens = $10, logical_calls = $11,
                            physical_attempts = $12, checkpointed_at = $13,
                            expires_at = $14, updated_at = $13
                        WHERE bank_id = $1 AND descriptor_key = $2
                          AND segment_key = $3""",
                    bank_id,
                    descriptor_key,
                    checkpoint.segment_key,
                    segment_json,
                    checkpoint.provider,
                    checkpoint.configured_model,
                    checkpoint.resolved_model,
                    checkpoint.request_id,
                    checkpoint.input_tokens,
                    checkpoint.output_tokens,
                    checkpoint.logical_calls,
                    checkpoint.physical_attempts,
                    now,
                    expires_at,
                )

    async def purge_expired_video_segment_checkpoints(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        now: datetime,
        limit: int = 100,
    ) -> int:
        """Delete a bounded, bank-scoped batch of expired derived segments."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > 10_000:
            raise ValueError("segment checkpoint cleanup limit must be between 1 and 10000")
        deleted = 0
        async with conn.transaction():
            rows = await conn.fetch(
                f"""SELECT descriptor_key, segment_key
                    FROM {self.tables.segment_checkpoints}
                    WHERE bank_id = $1 AND expires_at <= $2
                    ORDER BY expires_at, descriptor_key, segment_key
                    LIMIT $3""",
                bank_id,
                now,
                limit,
            )
            for row in rows:
                status = await conn.execute(
                    f"""DELETE FROM {self.tables.segment_checkpoints}
                        WHERE bank_id = $1 AND descriptor_key = $2
                          AND segment_key = $3 AND expires_at <= $4""",
                    bank_id,
                    str(row["descriptor_key"]),
                    str(row["segment_key"]),
                    now,
                )
                deleted += _affected_rows(status)
        return deleted

    async def checkpoint_descriptor(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        canonical_markdown: str,
        provenance_metadata: Mapping[str, Any],
        entities: Sequence[str],
        now: datetime,
        expires_at: datetime | None,
    ) -> DescriptorRecord:
        """Commit the only reusable provider output allowed in the ledger."""

        if not canonical_markdown or len(canonical_markdown) > MAX_CANONICAL_MARKDOWN_CHARS:
            raise ValueError("canonical descriptor Markdown is empty or exceeds the limit")
        _reject_encoded_payload("canonical descriptor Markdown", canonical_markdown)
        metadata_value = _flat_metadata(provenance_metadata)
        entity_value = _entity_names(entities)
        if expires_at is not None:
            _require_aware_order(now, expires_at, "descriptor expiry")
        async with conn.transaction():
            record = await self._mutate_descriptor(
                conn,
                f"""UPDATE {self.tables.descriptor_cache}
                    SET status = 'completed',
                        claim_token = NULL,
                        lease_expires_at = NULL,
                        canonical_markdown = $4,
                        provenance_metadata = $5::jsonb,
                        entities = $6::jsonb,
                        checkpointed_at = $7,
                        expires_at = $8,
                        updated_at = $7
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND status = 'processing' AND claim_token = $3
                      AND lease_expires_at > $7""",
                bank_id,
                descriptor_key,
                claim_token,
                canonical_markdown,
                _json_for_db(metadata_value),
                _json_for_db(entity_value),
                now,
                expires_at,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
            )
            if record is None:
                raise LedgerConflictError("descriptor checkpoint lost its active claim")
            await conn.execute(
                f"""DELETE FROM {self.tables.segment_checkpoints}
                    WHERE bank_id = $1 AND descriptor_key = $2""",
                bank_id,
                descriptor_key,
            )
            return record

    async def fail_descriptor(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID,
        now: datetime,
    ) -> DescriptorRecord:
        """Release a known failed attempt without persisting provider error text."""

        async with conn.transaction():
            record = await self._mutate_descriptor(
                conn,
                f"""UPDATE {self.tables.descriptor_cache}
                    SET status = 'failed', claim_token = NULL,
                        lease_expires_at = NULL, updated_at = $4
                    WHERE bank_id = $1 AND descriptor_key = $2
                      AND status = 'processing' AND claim_token = $3""",
                bank_id,
                descriptor_key,
                claim_token,
                now,
                bank_id=bank_id,
                descriptor_key=descriptor_key,
            )
            if record is None:
                raise LedgerConflictError("descriptor failure transition lost its active claim")
            return record

    async def fail_descriptor_and_mark_document_terminal(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        descriptor_key: str,
        claim_token: UUID | None,
        document_id: str,
        command_key: str,
        status: Literal["failed", "cancelled"],
        now: datetime,
        expected_sequence: int | None = None,
        expected_operation_id: UUID | None = None,
    ) -> DocumentCommandRecord:
        """Atomically release a descriptor claim and close its command.

        A conversion worker can discover a failure after its descriptor lease
        has expired.  If descriptor release and command terminalization happen
        in separate transactions, a replacement worker may acquire the claim
        between those statements and the stale worker can then clear the new
        owner's command/head state.  This method keeps the descriptor row lock
        until the command CAS commits and, when supplied, verifies the original
        operation/sequence as an additional ownership fence.

        ``claim_token`` may be ``None`` for paths that never acquired a
        descriptor (for example a cancellation observed before claim); the
        command ownership predicates still apply in that case.
        """

        if status not in {"failed", "cancelled"}:
            raise ValueError("invalid descriptor failure terminal status")
        if claim_token is None and expected_operation_id is None:
            raise ValueError("a descriptor claim or expected operation owner is required")
        if expected_sequence is not None and (
            isinstance(expected_sequence, bool) or not isinstance(expected_sequence, int) or expected_sequence < 1
        ):
            raise ValueError("expected_sequence must be a positive integer")
        async with conn.transaction():
            if claim_token is not None:
                # This lock includes the lease predicate.  An expired/lost
                # claim is a conflict, not permission to terminalize a shared
                # command; the caller must defer and let the current owner
                # finish or retry from its checkpoint.
                await self._lock_active_descriptor_claim(
                    conn,
                    bank_id=bank_id,
                    descriptor_key=descriptor_key,
                    claim_token=claim_token,
                    now=now,
                )
                await self.fail_descriptor(
                    conn,
                    bank_id=bank_id,
                    descriptor_key=descriptor_key,
                    claim_token=claim_token,
                    now=now,
                )

            return await self.mark_document_terminal(
                conn,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
                status=status,
                now=now,
                expected_sequence=expected_sequence,
                expected_operation_id=expected_operation_id,
            )

    async def get_document_head(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
    ) -> DocumentHeadRecord | None:
        row = await conn.fetchrow(
            f"""SELECT {_HEAD_COLUMNS}
                FROM {self.tables.document_heads}
                WHERE bank_id = $1 AND document_id = $2""",
            bank_id,
            document_id,
        )
        return _head_from_row(row) if row is not None else None

    async def _mutate_head(
        self,
        conn: DatabaseConnection,
        query: str,
        *args: Any,
        bank_id: str,
        document_id: str,
    ) -> DocumentHeadRecord | None:
        if conn.backend_type != "oracle":
            row = await conn.fetchrow(f"{query} RETURNING {_HEAD_COLUMNS}", *args)
            return _head_from_row(row) if row is not None else None
        status = await conn.execute(query, *args)
        if _affected_rows(status) != 1:
            return None
        return await self.get_document_head(conn, bank_id=bank_id, document_id=document_id)

    async def get_document_command(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
    ) -> DocumentCommandRecord | None:
        row = await conn.fetchrow(
            f"""SELECT {_COMMAND_COLUMNS}
                FROM {self.tables.document_commands}
                WHERE bank_id = $1 AND document_id = $2 AND command_key = $3""",
            bank_id,
            document_id,
            command_key,
        )
        return _command_from_row(row) if row is not None else None

    async def _mutate_command(
        self,
        conn: DatabaseConnection,
        query: str,
        *args: Any,
        bank_id: str,
        document_id: str,
        command_key: str,
    ) -> DocumentCommandRecord | None:
        if conn.backend_type != "oracle":
            row = await conn.fetchrow(f"{query} RETURNING {_COMMAND_COLUMNS}", *args)
            return _command_from_row(row) if row is not None else None
        status = await conn.execute(query, *args)
        if _affected_rows(status) != 1:
            return None
        return await self.get_document_command(
            conn,
            bank_id=bank_id,
            document_id=document_id,
            command_key=command_key,
        )

    async def admit_document_command(
        self,
        conn: DatabaseConnection,
        spec: DocumentCommandSpec,
        *,
        now: datetime,
    ) -> DocumentAdmission:
        """Idempotently admit a command and allocate its monotonic sequence.

        Locking the head serializes both same-key retries and different updates
        for one logical document.  A retry returns the original operation and
        sequence; it never allocates another visible command.  Returning an
        existing row deduplicates descriptor/command work only: the caller
        must not treat it as a blanket no-op for request metadata or tags.
        Completed retries require an idempotent metadata refresh; in-flight
        retries must follow the owner that will perform that refresh.
        """

        async with conn.transaction():
            await conn.execute(
                f"""INSERT INTO {self.tables.document_heads}
                    (bank_id, document_id, next_sequence, published_sequence,
                     active_sequence, created_at, updated_at)
                    VALUES ($1, $2, 1, 0, NULL, $3, $3)
                    ON CONFLICT (bank_id, document_id) DO NOTHING""",
                spec.bank_id,
                spec.document_id,
                now,
            )
            head_row = await conn.fetchrow(
                f"""SELECT {_HEAD_COLUMNS}
                    FROM {self.tables.document_heads}
                    WHERE bank_id = $1 AND document_id = $2
                    FOR UPDATE""",
                spec.bank_id,
                spec.document_id,
            )
            if head_row is None:
                raise LedgerInvariantError("document head disappeared during admission")

            existing = await self.get_document_command(
                conn,
                bank_id=spec.bank_id,
                document_id=spec.document_id,
                command_key=spec.command_key,
            )
            if existing is not None:
                _verify_command_identity(existing, spec)
                return DocumentAdmission(command=existing, created=False)

            sequence = int(head_row["next_sequence"])
            command = await self._mutate_command(
                conn,
                f"""INSERT INTO {self.tables.document_commands}
                    (bank_id, document_id, command_key, sequence, operation_id,
                     source_storage_key, asset_sha256, descriptor_key,
                     retain_input_fingerprint, status,
                     source_delete_after_retain, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                            'pending', $10, $11, $11)""",
                spec.bank_id,
                spec.document_id,
                spec.command_key,
                sequence,
                spec.operation_id,
                spec.source_storage_key,
                spec.asset_sha256,
                spec.descriptor_key,
                spec.retain_input_fingerprint,
                _db_bool(conn, spec.source_delete_after_retain),
                now,
                bank_id=spec.bank_id,
                document_id=spec.document_id,
                command_key=spec.command_key,
            )
            if command is None:
                raise LedgerInvariantError("document command insert returned no row")

            updated_head = await self._mutate_head(
                conn,
                f"""UPDATE {self.tables.document_heads}
                    SET next_sequence = $4, active_sequence = $3, updated_at = $5
                    WHERE bank_id = $1 AND document_id = $2
                      AND next_sequence = $3""",
                spec.bank_id,
                spec.document_id,
                sequence,
                sequence + 1,
                now,
                bank_id=spec.bank_id,
                document_id=spec.document_id,
            )
            if updated_head is None:
                raise LedgerConflictError("document admission sequence compare-and-swap failed")
            return DocumentAdmission(command=command, created=True)

    async def mark_document_processing(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
        now: datetime,
    ) -> DocumentCommandRecord:
        async with conn.transaction():
            command = await self._mutate_command(
                conn,
                f"""UPDATE {self.tables.document_commands}
                    SET status = 'processing', updated_at = $4
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                      AND status = 'pending'""",
                bank_id,
                document_id,
                command_key,
                now,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
            )
            if command is None:
                raise LedgerConflictError("document command is not pending")
            return command

    async def restart_document_command(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
        now: datetime,
    ) -> DocumentCommandRecord:
        """Reactivate the latest failed command for an explicit operation retry.

        A command can be restarted only when no later command was admitted or
        published.  The original sequence, operation and immutable source are
        reused, so retrying never appends a second visible document command.
        """

        async with conn.transaction():
            head_row = await conn.fetchrow(
                f"""SELECT {_HEAD_COLUMNS}
                    FROM {self.tables.document_heads}
                    WHERE bank_id = $1 AND document_id = $2
                    FOR UPDATE""",
                bank_id,
                document_id,
            )
            command_row = await conn.fetchrow(
                f"""SELECT {_COMMAND_COLUMNS}
                    FROM {self.tables.document_commands}
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                    FOR UPDATE""",
                bank_id,
                document_id,
                command_key,
            )
            if head_row is None or command_row is None:
                raise LedgerInvariantError("document retry state does not exist")
            head = _head_from_row(head_row)
            command = _command_from_row(command_row)
            is_latest_unpublished = (
                command.sequence > head.published_sequence
                and head.next_sequence == command.sequence + 1
                and head.active_sequence in {None, command.sequence}
            )
            if command.status not in {"failed", "cancelled"} or not is_latest_unpublished:
                raise LedgerConflictError("document command cannot be restarted after a newer admission")

            await conn.execute(
                f"""UPDATE {self.tables.document_heads}
                    SET active_sequence = $3, updated_at = $4
                    WHERE bank_id = $1 AND document_id = $2""",
                bank_id,
                document_id,
                command.sequence,
                now,
            )
            restarted = await self._mutate_command(
                conn,
                f"""UPDATE {self.tables.document_commands}
                    SET status = 'processing', completed_at = NULL,
                        child_retain_operation_id = NULL, updated_at = $4
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                      AND status IN ('failed', 'cancelled')""",
                bank_id,
                document_id,
                command_key,
                now,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
            )
            if restarted is None:
                raise LedgerConflictError("document retry lost its command compare-and-swap")
            return restarted

    async def attach_child_retain(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
        child_retain_operation_id: UUID,
        now: datetime,
    ) -> DocumentCommandRecord:
        async with conn.transaction():
            command = await self._mutate_command(
                conn,
                f"""UPDATE {self.tables.document_commands}
                    SET status = 'retaining', child_retain_operation_id = $4,
                        updated_at = $5
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                      AND (
                          status = 'processing'
                          OR (status = 'retaining' AND child_retain_operation_id = $4)
                      )""",
                bank_id,
                document_id,
                command_key,
                child_retain_operation_id,
                now,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
            )
            if command is None:
                raise LedgerConflictError("child retain cannot be attached to this command state")
            return command

    async def mark_source_deleted(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
        now: datetime,
    ) -> DocumentCommandRecord:
        async with conn.transaction():
            command = await self._mutate_command(
                conn,
                f"""UPDATE {self.tables.document_commands}
                    SET source_deleted_at = COALESCE(source_deleted_at, $4),
                        updated_at = $4
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                      AND source_delete_after_retain = {_sql_bool(conn, True)}""",
                bank_id,
                document_id,
                command_key,
                now,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
            )
            if command is None:
                raise LedgerConflictError("source deletion is not enabled for this command")
            return command

    async def lock_for_publish(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
    ) -> tuple[PublishDecision, DocumentCommandRecord]:
        """Lock a document head and decide whether this command may publish.

        The caller **must** already be inside the same database transaction
        that will write the existing HMS document.  The lock must remain held
        until ``complete_publish`` and that document write commit together.
        """

        head_row = await conn.fetchrow(
            f"""SELECT {_HEAD_COLUMNS}
                FROM {self.tables.document_heads}
                WHERE bank_id = $1 AND document_id = $2
                FOR UPDATE""",
            bank_id,
            document_id,
        )
        command_row = await conn.fetchrow(
            f"""SELECT {_COMMAND_COLUMNS}
                FROM {self.tables.document_commands}
                WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                FOR UPDATE""",
            bank_id,
            document_id,
            command_key,
        )
        if head_row is None or command_row is None:
            raise LedgerInvariantError("document publish state does not exist")
        head = _head_from_row(head_row)
        command = _command_from_row(command_row)

        # A completed command may be replayed only for an idempotent metadata
        # refresh while it is still the latest accepted document state.  Once
        # a newer sequence is active, the older replay must not race or
        # overwrite that command even though its original publish completed.
        if (
            command.status == "completed"
            and command.sequence == head.published_sequence
            and head.active_sequence is None
        ):
            return PublishDecision.ALREADY_PUBLISHED, command
        if command.sequence <= head.published_sequence or head.active_sequence != command.sequence:
            return PublishDecision.SUPERSEDED, command
        if command.status != "retaining":
            return PublishDecision.NOT_READY, command
        return PublishDecision.PUBLISH, command

    async def complete_publish(
        self,
        conn: DatabaseConnection,
        *,
        command: DocumentCommandRecord,
        now: datetime,
    ) -> DocumentCommandRecord:
        """Advance the publish CAS after the document write, in the same tx."""

        head = await self._mutate_head(
            conn,
            f"""UPDATE {self.tables.document_heads}
                SET published_sequence = $3, active_sequence = NULL,
                    updated_at = $4
                WHERE bank_id = $1 AND document_id = $2
                  AND active_sequence = $3 AND published_sequence < $3""",
            command.bank_id,
            command.document_id,
            command.sequence,
            now,
            bank_id=command.bank_id,
            document_id=command.document_id,
        )
        if head is None:
            raise LedgerConflictError("a newer document command won the publish compare-and-swap")
        completed = await self._mutate_command(
            conn,
            f"""UPDATE {self.tables.document_commands}
                SET status = 'completed', completed_at = $4, updated_at = $4
                WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                  AND status = 'retaining'""",
            command.bank_id,
            command.document_id,
            command.command_key,
            now,
            bank_id=command.bank_id,
            document_id=command.document_id,
            command_key=command.command_key,
        )
        if completed is None:
            raise LedgerConflictError("document command could not complete after publish")
        return completed

    async def mark_document_terminal(
        self,
        conn: DatabaseConnection,
        *,
        bank_id: str,
        document_id: str,
        command_key: str,
        status: Literal["failed", "superseded", "cancelled"],
        now: datetime,
        expected_sequence: int | None = None,
        expected_child_retain_operation_id: UUID | None = None,
        expected_operation_id: UUID | None = None,
    ) -> DocumentCommandRecord:
        """Close a non-published command and release its active head slot.

        When a terminal transition is driven by a child operation, callers may
        provide both expected identity fields.  The compare-and-swap then
        proves that the child still owns the command before releasing the
        active head slot; a late failure from an older child cannot close a
        newer child attached to the same logical document.  Conversion workers
        may additionally provide ``expected_operation_id`` to fence a stale
        file-convert task before it closes its command.
        """

        if status not in {"failed", "superseded", "cancelled"}:
            raise ValueError("invalid terminal document status")
        if expected_child_retain_operation_id is not None and expected_sequence is None:
            raise ValueError("expected_sequence is required with expected_child_retain_operation_id")
        if (
            expected_sequence is not None
            and expected_child_retain_operation_id is None
            and expected_operation_id is None
        ):
            raise ValueError("expected_sequence and expected_child_retain_operation_id must be provided together")
        async with conn.transaction():
            # All paths that touch both rows take the head before the command.
            # This matches admission/publish order and avoids a head<->command
            # lock inversion under concurrent cancellation and publication.
            head_row = await conn.fetchrow(
                f"""SELECT {_HEAD_COLUMNS}
                    FROM {self.tables.document_heads}
                    WHERE bank_id = $1 AND document_id = $2
                    FOR UPDATE""",
                bank_id,
                document_id,
            )
            if head_row is None:
                raise LedgerInvariantError("document head does not exist for terminal transition")
            command_predicate = "AND status IN ('pending', 'processing', 'retaining')"
            command_args: list[Any] = [bank_id, document_id, command_key, status, now]
            if expected_sequence is not None:
                sequence_placeholder = len(command_args) + 1
                command_predicate += f" AND sequence = ${sequence_placeholder}"
                command_args.append(expected_sequence)
            if expected_child_retain_operation_id is not None:
                child_placeholder = len(command_args) + 1
                command_predicate += f" AND child_retain_operation_id = ${child_placeholder}"
                command_args.append(expected_child_retain_operation_id)
            if expected_operation_id is not None:
                operation_placeholder = len(command_args) + 1
                command_predicate += f" AND operation_id = ${operation_placeholder}"
                command_args.append(expected_operation_id)
            command = await self._mutate_command(
                conn,
                f"""UPDATE {self.tables.document_commands}
                    SET status = $4, completed_at = $5, updated_at = $5
                    WHERE bank_id = $1 AND document_id = $2 AND command_key = $3
                      {command_predicate}""",
                *command_args,
                bank_id=bank_id,
                document_id=document_id,
                command_key=command_key,
            )
            if command is None:
                raise LedgerConflictError("document command is already terminal")
            await conn.execute(
                f"""UPDATE {self.tables.document_heads}
                    SET active_sequence = NULL, updated_at = $4
                    WHERE bank_id = $1 AND document_id = $2
                      AND active_sequence = $3""",
                bank_id,
                document_id,
                command.sequence,
                now,
            )
            return command


def _require_aware_order(start: datetime, end: datetime, label: str) -> None:
    if start.tzinfo is None or start.utcoffset() is None or end.tzinfo is None or end.utcoffset() is None:
        raise ValueError(f"{label} timestamps must be timezone-aware")
    if end <= start:
        raise ValueError(f"{label} end must be after start")


def _verify_command_identity(existing: DocumentCommandRecord, spec: DocumentCommandSpec) -> None:
    # A transport/worker retry legitimately has a new operation UUID and a new
    # immutable upload key.  Those identify attempts, not the logical command.
    # Returning the original row is what prevents the retry from allocating a
    # second sequence or publishing a duplicate document; the caller can then
    # clean up only its own unused source object.
    expected = (
        spec.asset_sha256,
        spec.descriptor_key,
        spec.retain_input_fingerprint,
        spec.source_delete_after_retain,
    )
    actual = (
        existing.asset_sha256,
        existing.descriptor_key,
        existing.retain_input_fingerprint,
        existing.source_delete_after_retain,
    )
    if actual != expected:
        raise LedgerInvariantError("document command key collision with different immutable admission fields")


__all__ = [
    "DESCRIPTOR_CACHE_TABLE",
    "DOCUMENT_COMMANDS_TABLE",
    "DOCUMENT_HEADS_TABLE",
    "DescriptorIdentity",
    "DescriptorRecord",
    "DocumentAdmission",
    "DocumentCommandRecord",
    "DocumentCommandSpec",
    "DocumentHeadRecord",
    "LedgerConflictError",
    "LedgerError",
    "LedgerInvariantError",
    "LedgerTables",
    "MultimodalLedger",
    "PublishDecision",
    "derive_descriptor_key",
    "derive_document_command_key",
    "derive_retain_input_fingerprint",
]
