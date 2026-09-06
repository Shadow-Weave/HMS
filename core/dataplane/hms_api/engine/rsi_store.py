"""PostgreSQL persistence for versioned RSI Harness state.

This module deliberately keeps RSI state separate from ordinary memory units.
It is invoked by the HTTP/MCP boundary only after normal HMS tenant and bank
authorization has been applied.  Oracle is fail-closed until an equivalent
snapshot implementation is available for that backend.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from typing import Any, Mapping, Sequence

from .memory_engine import fq_table


class RSIStoreError(RuntimeError):
    """Base error for RSI persistence operations."""


class RSINotSupportedError(RSIStoreError):
    """Raised when the configured HMS backend cannot provide RSI guarantees."""


class RSIConflictError(RSIStoreError):
    """Raised when an immutable RSI object would be changed or duplicated."""


class RSINotFoundError(RSIStoreError):
    """Raised when a requested RSI object does not exist in the bank."""


_FORBIDDEN_CANDIDATE_METADATA_KEYS = frozenset(
    {
        "benchmark_output",
        "benchmark_score",
        "gold_patch",
        "hidden_test",
        "reference_answer",
        "verifier_output",
        "verifier_stdout",
        "verifier_stderr",
    }
)
_FORBIDDEN_CANDIDATE_TEXT = re.compile(
    r"(?:gold[_ -]?patch|hidden[_ -]?test|reference[_ -]?answer|verifier[_ -]?(?:output|stdout|stderr))",
    re.IGNORECASE,
)
_SNAPSHOT_NAMESPACES = frozenset({"project", "general"})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def hash_verifier_receipt(token: str) -> str:
    if not token or not token.strip():
        raise ValueError("verifier receipt token must not be empty")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _candidate_digest(candidate_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest()


def _verify_receipt(*, token: str, run_id: str, candidate_digest: str) -> str:
    secret_value = os.environ.get("HMS_API_RSI_VERIFIER_SECRET") or os.environ.get("HMS_RSI_VERIFIER_SECRET")
    secret = secret_value.encode("utf-8") if secret_value else b""
    if not secret:
        raise RSINotSupportedError("HMS_RSI_VERIFIER_SECRET is required for candidate promotion")
    expected = hmac.new(
        secret,
        f"development:{run_id}:{candidate_digest}:0".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, token):
        raise RSIConflictError("verifier receipt is invalid, expired, or not for this candidate set")
    return hash_verifier_receipt(token)


def _validate_candidate_boundary(candidate: Mapping[str, Any]) -> None:
    metadata = {str(key).lower() for key in (candidate.get("metadata") or {})}
    forbidden_metadata = sorted(metadata & _FORBIDDEN_CANDIDATE_METADATA_KEYS)
    if forbidden_metadata:
        raise ValueError(f"candidate contains verifier or benchmark metadata: {forbidden_metadata}")
    fields = [candidate.get("trigger", ""), candidate.get("content", "")]
    fields.extend(str(ref) for ref in candidate.get("evidence_refs", ()))
    fields.append(_json(candidate.get("metadata", {})))
    if any(_FORBIDDEN_CANDIDATE_TEXT.search(str(value)) for value in fields):
        raise ValueError("candidate contains verifier output or benchmark answer material")


def _validate_snapshot_memory_boundary(text: Any, metadata: Mapping[str, Any]) -> None:
    """Reject obvious evaluator material before live memory is snapshot-tagged."""
    metadata_keys = {str(key).lower() for key in metadata}
    if metadata_keys & _FORBIDDEN_CANDIDATE_METADATA_KEYS:
        raise ValueError("memory contains verifier or benchmark metadata")
    inspected = f"{text}\n{_json(metadata)}"
    if _FORBIDDEN_CANDIDATE_TEXT.search(inspected):
        raise ValueError("memory contains verifier output or benchmark answer material")


async def _authorized_backend(memory: Any, bank_id: str, request_context: Any):
    await memory._authenticate_tenant(request_context)
    profile = await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False)
    if profile is None:
        raise RSINotFoundError(f"bank '{bank_id}' not found")
    backend = await memory._get_backend()
    if backend.backend_type != "postgresql":
        raise RSINotSupportedError("RSI snapshots currently require PostgreSQL")
    return backend


async def tag_memory_namespaces(
    memory: Any,
    *,
    bank_id: str,
    memory_ids: Sequence[str],
    namespace: str,
    only_unclassified: bool,
    request_context: Any,
) -> dict[str, Any]:
    """Explicitly classify an allowlisted set of memory units.

    There is intentionally no "classify the whole bank" default: unknown
    memories are excluded from snapshots until an operator or an upstream
    retention policy assigns a safe namespace.
    """
    if namespace not in _SNAPSHOT_NAMESPACES:
        raise ValueError("namespace must be project or general")
    unique_ids = list(dict.fromkeys(str(memory_id).strip() for memory_id in memory_ids if str(memory_id).strip()))
    if not unique_ids:
        raise ValueError("memory_ids must not be empty")
    backend = await _authorized_backend(memory, bank_id, request_context)
    updated: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    async with backend.transaction() as conn:
        rows = await conn.fetch(
            f"SELECT id::text AS id, text, metadata FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id::text = ANY($2::text[]) FOR UPDATE",
            bank_id,
            unique_ids,
        )
        by_id = {row["id"]: row for row in rows}
        missing = [memory_id for memory_id in unique_ids if memory_id not in by_id]
        for memory_id in unique_ids:
            row = by_id.get(memory_id)
            if row is None:
                continue
            metadata = _row_json(conn, row["metadata"])
            _validate_snapshot_memory_boundary(row["text"], metadata)
            current = metadata.get("rsi_namespace")
            if current and current != namespace:
                if only_unclassified:
                    skipped.append(memory_id)
                    continue
                raise RSIConflictError(f"memory '{memory_id}' already has namespace '{current}'")
            if current == namespace:
                skipped.append(memory_id)
                continue
            metadata["rsi_namespace"] = namespace
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET metadata = $1::jsonb, updated_at = NOW() WHERE bank_id = $2 AND id = $3::uuid",
                _json(metadata),
                bank_id,
                memory_id,
            )
            updated.append(memory_id)
    return {"bank_id": bank_id, "namespace": namespace, "updated": updated, "skipped": skipped, "missing": missing}


def _row_json(conn: Any, value: Any) -> Any:
    return conn.parse_json(value) or {}


def _snapshot_payload(row: Any, conn: Any) -> dict[str, Any]:
    return {
        "bank_id": row["bank_id"],
        "snapshot_id": row["snapshot_id"],
        "parent_snapshot_id": row["parent_snapshot_id"],
        "status": row["status"],
        "manifest": _row_json(conn, row["manifest"]),
        "created_at": row["created_at"],
        "frozen_at": row["frozen_at"],
    }


def _candidate_payload(row: Any, conn: Any) -> dict[str, Any]:
    return {
        "bank_id": row["bank_id"],
        "candidate_id": row["candidate_id"],
        "run_id": row["run_id"],
        "namespace": row["namespace"],
        "kind": row["kind"],
        "trigger": row["trigger"],
        "content": row["content"],
        "preconditions": _row_json(conn, row["preconditions"]),
        "anti_patterns": _row_json(conn, row["anti_patterns"]),
        "evidence_refs": _row_json(conn, row["evidence_refs"]),
        "source_commit": row["source_commit"],
        "created_before_verifier": row["created_before_verifier"],
        "status": row["status"],
        "metadata": _row_json(conn, row["metadata"]),
        "invalidation_reason": row["invalidation_reason"],
        "promoted_snapshot_id": row["promoted_snapshot_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def _copy_snapshot_memory_units(conn: Any, *, bank_id: str, snapshot_id: str) -> int:
    """Copy only explicitly classified project/general memories.

    Unknown or episode-scoped HMS memories are intentionally excluded. This
    fail-closed default prevents a task transcript from entering a benchmark
    snapshot merely because it happened to exist in the bank.
    """
    result = await conn.execute(
        f"""
        INSERT INTO {fq_table('rsi_snapshot_memory_units')}
            (bank_id, snapshot_id, memory_id, namespace, fact_type, text, context,
             tags, metadata, event_date, occurred_start, occurred_end, mentioned_at,
             source_memory_ids)
        SELECT mu.bank_id, $2, mu.id::text,
               mu.metadata->>'rsi_namespace', mu.fact_type, mu.text, mu.context,
               COALESCE(mu.tags, '[]'::jsonb), COALESCE(mu.metadata, '{{}}'::jsonb),
               mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at,
               mu.source_memory_ids
        FROM {fq_table('memory_units')} mu
        WHERE mu.bank_id = $1
          AND mu.metadata->>'rsi_namespace' IN ('project', 'general')
        ON CONFLICT (bank_id, snapshot_id, memory_id) DO NOTHING
        """,
        bank_id,
        snapshot_id,
    )
    try:
        return int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


async def _inherit_parent_snapshot(conn: Any, *, bank_id: str, parent_snapshot_id: str, snapshot_id: str) -> None:
    """Materialize parent contents so every frozen snapshot is self-contained."""
    await conn.execute(
        f"""
        INSERT INTO {fq_table('rsi_snapshot_memory_units')}
            (bank_id, snapshot_id, memory_id, namespace, fact_type, text, context,
             tags, metadata, event_date, occurred_start, occurred_end, mentioned_at,
             source_memory_ids)
        SELECT bank_id, $3, memory_id, namespace, fact_type, text, context,
               tags, metadata, event_date, occurred_start, occurred_end, mentioned_at,
               source_memory_ids
        FROM {fq_table('rsi_snapshot_memory_units')}
        WHERE bank_id = $1 AND snapshot_id = $2
        ON CONFLICT (bank_id, snapshot_id, memory_id) DO NOTHING
        """,
        bank_id,
        parent_snapshot_id,
        snapshot_id,
    )
    await conn.execute(
        f"""
        INSERT INTO {fq_table('rsi_snapshot_candidates')}
            (bank_id, snapshot_id, candidate_id, verifier_receipt_hash)
        SELECT bank_id, $3, candidate_id, verifier_receipt_hash
        FROM {fq_table('rsi_snapshot_candidates')}
        WHERE bank_id = $1 AND snapshot_id = $2
        ON CONFLICT (bank_id, snapshot_id, candidate_id) DO NOTHING
        """,
        bank_id,
        parent_snapshot_id,
        snapshot_id,
    )


async def _refresh_snapshot_completeness(
    conn: Any,
    *,
    bank_id: str,
    snapshot_id: str,
    existing_manifest: Mapping[str, Any] | None = None,
    require_stable_source: bool,
) -> dict[str, Any]:
    """Record and validate the safe-source fingerprint for a snapshot."""
    source_rows = await conn.fetch(
        f"""
        SELECT id::text AS memory_id, COALESCE(updated_at, created_at)::text AS version,
               md5(text) AS text_hash
        FROM {fq_table('memory_units')}
        WHERE bank_id = $1 AND metadata->>'rsi_namespace' IN ('project', 'general')
        ORDER BY id
        """,
        bank_id,
    )
    source_lines = [f"{row['memory_id']}|{row['version']}|{row['text_hash']}" for row in source_rows]
    source_checksum = hashlib.sha256("\n".join(source_lines).encode("utf-8")).hexdigest()
    previous = dict(existing_manifest or {})
    previous_checksum = previous.get("memory_source_checksum")
    if require_stable_source and previous_checksum and previous_checksum != source_checksum:
        raise RSIConflictError("source memories changed after snapshot creation; create a new snapshot")

    snapshot_rows = await conn.fetch(
        f"""
        SELECT memory_id, md5(text) AS text_hash
        FROM {fq_table('rsi_snapshot_memory_units')}
        WHERE bank_id = $1 AND snapshot_id = $2
        """,
        bank_id,
        snapshot_id,
    )
    snapshot_hashes = {row["memory_id"]: row["text_hash"] for row in snapshot_rows}
    covered = sum(1 for row in source_rows if snapshot_hashes.get(row["memory_id"]) == row["text_hash"])
    complete = covered == len(source_rows)
    manifest = {
        **previous,
        "memory_namespace_policy": "explicit_project_or_general",
        "memory_source_count": len(source_rows),
        "memory_covered_count": covered,
        "memory_source_checksum": source_checksum,
        "memory_complete": complete,
    }
    await conn.execute(
        f"UPDATE {fq_table('rsi_harness_snapshots')} SET manifest = $1::jsonb WHERE bank_id = $2 AND snapshot_id = $3",
        _json(manifest),
        bank_id,
        snapshot_id,
    )
    return manifest


async def create_snapshot(
    memory: Any,
    *,
    bank_id: str,
    snapshot_id: str,
    parent_snapshot_id: str | None,
    manifest: Mapping[str, Any],
    request_context: Any,
) -> dict[str, Any]:
    if not snapshot_id.strip() or len(snapshot_id) > 256:
        raise ValueError("snapshot_id must be 1-256 characters")
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.transaction() as conn:
        if parent_snapshot_id:
            parent = await conn.fetchrow(
                f"SELECT status FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2",
                bank_id,
                parent_snapshot_id,
            )
            if parent is None:
                raise RSINotFoundError(f"parent snapshot '{parent_snapshot_id}' not found")
            if parent["status"] != "frozen":
                raise RSIConflictError("parent snapshot must be frozen")
        existing = await conn.fetchrow(
            f"SELECT * FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2",
            bank_id,
            snapshot_id,
        )
        if existing is not None:
            existing_payload = _snapshot_payload(existing, conn)
            existing_manifest = existing_payload["manifest"]
            # The first create call enriches the manifest with source
            # fingerprints and completeness.  Retries must be idempotent while
            # still rejecting a caller that changes any originally supplied
            # field.
            manifest_matches = all(existing_manifest.get(key) == value for key, value in dict(manifest).items())
            if not manifest_matches or existing_payload["parent_snapshot_id"] != parent_snapshot_id:
                raise RSIConflictError(f"snapshot '{snapshot_id}' already exists with different content")
            return existing_payload
        row = await conn.fetchrow(
            f"""
            INSERT INTO {fq_table('rsi_harness_snapshots')}
                (bank_id, snapshot_id, parent_snapshot_id, manifest)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING *
            """,
            bank_id,
            snapshot_id,
            parent_snapshot_id,
            _json(dict(manifest)),
        )
        if parent_snapshot_id:
            await _inherit_parent_snapshot(
                conn,
                bank_id=bank_id,
                parent_snapshot_id=parent_snapshot_id,
                snapshot_id=snapshot_id,
            )
        await _copy_snapshot_memory_units(conn, bank_id=bank_id, snapshot_id=snapshot_id)
        manifest = await _refresh_snapshot_completeness(
            conn,
            bank_id=bank_id,
            snapshot_id=snapshot_id,
            existing_manifest=manifest,
            require_stable_source=False,
        )
        payload = _snapshot_payload(row, conn)
        payload["manifest"] = manifest
        return payload


async def get_snapshot(memory: Any, *, bank_id: str, snapshot_id: str, request_context: Any) -> dict[str, Any]:
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT * FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2",
            bank_id,
            snapshot_id,
        )
        if row is None:
            raise RSINotFoundError(f"snapshot '{snapshot_id}' not found")
        payload = _snapshot_payload(row, conn)
        members = await conn.fetch(
            f"""
            SELECT candidate_id
            FROM {fq_table('rsi_snapshot_candidates')}
            WHERE bank_id = $1 AND snapshot_id = $2
            ORDER BY promoted_at ASC, candidate_id ASC
            """,
            bank_id,
            snapshot_id,
        )
        payload["candidate_ids"] = [member["candidate_id"] for member in members]
        return payload


async def freeze_snapshot(memory: Any, *, bank_id: str, snapshot_id: str, request_context: Any) -> dict[str, Any]:
    """Seal an empty/base building snapshot before any evolution round."""
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.transaction() as conn:
        row = await conn.fetchrow(
            f"SELECT * FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2 FOR UPDATE",
            bank_id,
            snapshot_id,
        )
        if row is None:
            raise RSINotFoundError(f"snapshot '{snapshot_id}' not found")
        if row["status"] != "building":
            raise RSIConflictError("snapshot is already frozen or deprecated")
        manifest = await _refresh_snapshot_completeness(
            conn,
            bank_id=bank_id,
            snapshot_id=snapshot_id,
            existing_manifest=_row_json(conn, row["manifest"]),
            require_stable_source=True,
        )
        row = await conn.fetchrow(
            f"""
            UPDATE {fq_table('rsi_harness_snapshots')}
            SET status = 'frozen', frozen_at = NOW(), manifest = $3::jsonb
            WHERE bank_id = $1 AND snapshot_id = $2
            RETURNING *
            """,
            bank_id,
            snapshot_id,
            _json(manifest),
        )
        payload = _snapshot_payload(row, conn)
        members = await conn.fetch(
            f"SELECT candidate_id FROM {fq_table('rsi_snapshot_candidates')} WHERE bank_id = $1 AND snapshot_id = $2",
            bank_id,
            snapshot_id,
        )
        payload["candidate_ids"] = [member["candidate_id"] for member in members]
        return payload


async def search_snapshot(
    memory: Any,
    *,
    bank_id: str,
    snapshot_id: str,
    namespace: str,
    query: str,
    limit: int,
    request_context: Any,
) -> list[dict[str, Any]]:
    if namespace not in {"project", "general"}:
        raise ValueError("snapshot search only supports project and general namespaces")
    if not query.strip():
        raise ValueError("snapshot search query must not be empty")
    limit = max(1, min(limit, 100))
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.acquire() as conn:
        snapshot = await conn.fetchrow(
            f"SELECT status, manifest FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2",
            bank_id,
            snapshot_id,
        )
        if snapshot is None:
            raise RSINotFoundError(f"snapshot '{snapshot_id}' not found")
        if snapshot["status"] != "frozen":
            raise RSIConflictError("snapshot is not frozen")
        if not _row_json(conn, snapshot["manifest"]).get("memory_complete", False):
            raise RSIConflictError("snapshot completeness check failed")
        rows = await conn.fetch(
            f"""
            SELECT memory_id, namespace, fact_type, text, metadata,
                   1.0::double precision AS score
            FROM {fq_table('rsi_snapshot_memory_units')}
            WHERE bank_id = $1 AND snapshot_id = $2 AND namespace = $3
              AND text ILIKE '%' || $4 || '%'
            LIMIT $5
            """,
            bank_id,
            snapshot_id,
            namespace,
            query,
            limit,
        )
        return [
            {
                "id": row["memory_id"],
                "text": row["text"],
                "namespace": row["namespace"],
                "fact_type": row["fact_type"],
                "score": row["score"],
                "metadata": _row_json(conn, row["metadata"]),
            }
            for row in rows
        ]


async def fetch_snapshot_memory(
    memory: Any,
    *,
    bank_id: str,
    snapshot_id: str,
    memory_id: str,
    request_context: Any,
) -> dict[str, Any]:
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.acquire() as conn:
        snapshot = await conn.fetchrow(
            f"SELECT status, manifest FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2",
            bank_id,
            snapshot_id,
        )
        if snapshot is None:
            raise RSINotFoundError(f"snapshot '{snapshot_id}' not found")
        if snapshot["status"] != "frozen":
            raise RSIConflictError("snapshot is not frozen")
        if not _row_json(conn, snapshot["manifest"]).get("memory_complete", False):
            raise RSIConflictError("snapshot completeness check failed")
        row = await conn.fetchrow(
            f"""
            SELECT memory_id, namespace, fact_type, text, context,
                   tags, metadata, event_date, occurred_start, occurred_end, mentioned_at,
                   source_memory_ids
            FROM {fq_table('rsi_snapshot_memory_units')}
            WHERE bank_id = $1 AND snapshot_id = $2 AND memory_id = $3
            """,
            bank_id,
            snapshot_id,
            memory_id,
        )
        if row is None:
            raise RSINotFoundError(f"memory '{memory_id}' not found in snapshot '{snapshot_id}'")
        return {
            "id": row["memory_id"],
            "namespace": row["namespace"],
            "fact_type": row["fact_type"],
            "text": row["text"],
            "context": row["context"],
            "tags": _row_json(conn, row["tags"]),
            "metadata": _row_json(conn, row["metadata"]),
            "event_date": row["event_date"],
            "occurred_start": row["occurred_start"],
            "occurred_end": row["occurred_end"],
            "mentioned_at": row["mentioned_at"],
            "source_memory_ids": [str(value) for value in (row["source_memory_ids"] or ())],
        }


async def stage_candidate(
    memory: Any,
    *,
    bank_id: str,
    candidate: Mapping[str, Any],
    request_context: Any,
) -> dict[str, Any]:
    required = ("candidate_id", "run_id", "namespace", "kind", "trigger", "content", "evidence_refs")
    missing = [key for key in required if not str(candidate.get(key, "")).strip()]
    if missing:
        raise ValueError(f"missing candidate fields: {', '.join(missing)}")
    if candidate.get("namespace") not in {"candidate", "project", "general"}:
        raise ValueError("candidate namespace must be candidate, project, or general")
    if candidate.get("created_before_verifier") is not True:
        raise ValueError("candidate must be created before verifier output")
    _validate_candidate_boundary(candidate)
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.transaction() as conn:
        existing = await conn.fetchrow(
            f"SELECT * FROM {fq_table('rsi_memory_candidates')} WHERE bank_id = $1 AND candidate_id = $2",
            bank_id,
            candidate["candidate_id"],
        )
        if existing is not None:
            existing_payload = _candidate_payload(existing, conn)
            immutable_fields = (
                "run_id",
                "namespace",
                "kind",
                "trigger",
                "content",
                "preconditions",
                "anti_patterns",
                "evidence_refs",
                "source_commit",
            )
            if any(existing_payload[field] != candidate.get(field) for field in immutable_fields):
                raise RSIConflictError(f"candidate '{candidate['candidate_id']}' already exists with different content")
            return existing_payload
        row = await conn.fetchrow(
            f"""
            INSERT INTO {fq_table('rsi_memory_candidates')}
                (bank_id, candidate_id, run_id, namespace, kind, trigger, content,
                 preconditions, anti_patterns, evidence_refs, source_commit,
                 created_before_verifier, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10::jsonb,
                    $11, $12, $13::jsonb)
            RETURNING *
            """,
            bank_id,
            candidate["candidate_id"],
            candidate["run_id"],
            candidate["namespace"],
            candidate["kind"],
            candidate["trigger"],
            candidate["content"],
            _json(candidate.get("preconditions", [])),
            _json(candidate.get("anti_patterns", [])),
            _json(candidate["evidence_refs"]),
            candidate.get("source_commit"),
            True,
            _json(candidate.get("metadata", {})),
        )
        return _candidate_payload(row, conn)


async def list_candidates(
    memory: Any,
    *,
    bank_id: str,
    status: str | None,
    run_id: str | None,
    request_context: Any,
) -> list[dict[str, Any]]:
    backend = await _authorized_backend(memory, bank_id, request_context)
    clauses = ["bank_id = $1"]
    args: list[Any] = [bank_id]
    if status:
        clauses.append(f"status = ${len(args) + 1}")
        args.append(status)
    if run_id:
        clauses.append(f"run_id = ${len(args) + 1}")
        args.append(run_id)
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM {fq_table('rsi_memory_candidates')} WHERE {' AND '.join(clauses)} ORDER BY created_at ASC",
            *args,
        )
        return [_candidate_payload(row, conn) for row in rows]


async def promote_candidates(
    memory: Any,
    *,
    bank_id: str,
    snapshot_id: str,
    candidate_ids: Sequence[str],
    run_id: str,
    candidate_digest: str,
    verifier_receipt_token: str,
    request_context: Any,
) -> dict[str, Any]:
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must not contain duplicates")
    normalized_ids = [str(candidate_id).strip() for candidate_id in candidate_ids]
    if any(not candidate_id for candidate_id in normalized_ids):
        raise ValueError("candidate_ids must not contain empty values")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("candidate_ids must not contain duplicates")
    candidate_ids = normalized_ids
    if not run_id.strip() or not candidate_digest.strip():
        raise ValueError("run_id and candidate_digest are required")
    expected_digest = _candidate_digest(candidate_ids)
    if not hmac.compare_digest(expected_digest, candidate_digest):
        raise RSIConflictError("candidate_digest does not match candidate_ids")
    receipt_hash = _verify_receipt(
        token=verifier_receipt_token,
        run_id=run_id,
        candidate_digest=candidate_digest,
    )
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.transaction() as conn:
        snapshot = await conn.fetchrow(
            f"SELECT * FROM {fq_table('rsi_harness_snapshots')} WHERE bank_id = $1 AND snapshot_id = $2 FOR UPDATE",
            bank_id,
            snapshot_id,
        )
        if snapshot is None:
            raise RSINotFoundError(f"snapshot '{snapshot_id}' not found")
        if snapshot["status"] != "building":
            raise RSIConflictError("only building snapshots can receive candidates")
        rows = await conn.fetch(
            f"""
            SELECT candidate_id, status
            FROM {fq_table('rsi_memory_candidates')}
            WHERE bank_id = $1 AND candidate_id = ANY($2::text[])
            FOR UPDATE
            """,
            bank_id,
            list(candidate_ids),
        )
        found = {row["candidate_id"]: row["status"] for row in rows}
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in found]
        if missing:
            raise RSINotFoundError(f"candidates not found: {missing}")
        invalid = [candidate_id for candidate_id, status in found.items() if status != "candidate"]
        if invalid:
            raise RSIConflictError(f"candidates are not promotable: {invalid}")
        run_rows = await conn.fetch(
            f"""
            SELECT candidate_id, run_id
            FROM {fq_table('rsi_memory_candidates')}
            WHERE bank_id = $1 AND candidate_id = ANY($2::text[])
            """,
            bank_id,
            list(candidate_ids),
        )
        mismatched_run = [row["candidate_id"] for row in run_rows if row["run_id"] != run_id]
        if mismatched_run:
            raise RSIConflictError(f"candidates belong to a different run: {mismatched_run}")
        try:
            await conn.execute(
                f"""
                INSERT INTO {fq_table('rsi_verifier_receipts')}
                    (bank_id, receipt_hash, run_id, candidate_digest, passed, snapshot_id)
                VALUES ($1, $2, $3, $4, TRUE, $5)
                """,
                bank_id,
                receipt_hash,
                run_id,
                candidate_digest,
                snapshot_id,
            )
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                raise RSIConflictError("verifier receipt has already been consumed") from exc
            raise
        await _refresh_snapshot_completeness(
            conn,
            bank_id=bank_id,
            snapshot_id=snapshot_id,
            existing_manifest=_row_json(conn, snapshot["manifest"]),
            require_stable_source=True,
        )
        await conn.execute(
            f"""
            UPDATE {fq_table('rsi_memory_candidates')}
            SET status = 'promoted', promoted_snapshot_id = $3, updated_at = NOW()
            WHERE bank_id = $1 AND candidate_id = ANY($2::text[])
            """,
            bank_id,
            list(candidate_ids),
            snapshot_id,
        )
        await conn.executemany(
            f"""
            INSERT INTO {fq_table('rsi_snapshot_candidates')}
                (bank_id, snapshot_id, candidate_id, verifier_receipt_hash)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (bank_id, snapshot_id, candidate_id) DO NOTHING
            """,
            [(bank_id, snapshot_id, candidate_id, receipt_hash) for candidate_id in candidate_ids],
        )
        await conn.execute(
            f"""
            INSERT INTO {fq_table('rsi_snapshot_memory_units')}
                (bank_id, snapshot_id, memory_id, namespace, fact_type, text, metadata)
            SELECT bank_id, $2, 'candidate:' || candidate_id, namespace, kind, content, metadata
            FROM {fq_table('rsi_memory_candidates')}
            WHERE bank_id = $1 AND candidate_id = ANY($3::text[])
              AND namespace IN ('project', 'general')
            ON CONFLICT (bank_id, snapshot_id, memory_id) DO NOTHING
            """,
            bank_id,
            snapshot_id,
            list(candidate_ids),
        )
        await conn.execute(
            f"""
            UPDATE {fq_table('rsi_harness_snapshots')}
            SET status = 'frozen', frozen_at = NOW()
            WHERE bank_id = $1 AND snapshot_id = $2
            """,
            bank_id,
            snapshot_id,
        )
        return {
            "bank_id": bank_id,
            "snapshot_id": snapshot_id,
            "promoted_candidate_ids": list(candidate_ids),
            "verifier_receipt_accepted": True,
        }


async def invalidate_candidate(
    memory: Any,
    *,
    bank_id: str,
    candidate_id: str,
    reason: str,
    request_context: Any,
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("invalidation reason must not be empty")
    backend = await _authorized_backend(memory, bank_id, request_context)
    async with backend.transaction() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE {fq_table('rsi_memory_candidates')}
            SET status = 'invalidated', invalidation_reason = $3, updated_at = NOW()
            WHERE bank_id = $1 AND candidate_id = $2
              AND status IN ('candidate', 'promoted', 'quarantined')
            RETURNING *
            """,
            bank_id,
            candidate_id,
            reason,
        )
        if row is None:
            exists = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('rsi_memory_candidates')} WHERE bank_id = $1 AND candidate_id = $2",
                bank_id,
                candidate_id,
            )
            if exists is None:
                raise RSINotFoundError(f"candidate '{candidate_id}' not found")
            raise RSIConflictError(f"candidate '{candidate_id}' cannot be invalidated")
        return _candidate_payload(row, conn)
