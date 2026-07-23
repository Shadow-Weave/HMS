"""Focused tests for embedding-space fingerprints and bank guards."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hms_api.engine.embedding_fingerprint import (
    EmbeddingFingerprintMismatchError,
    EmbeddingFingerprintUnknownError,
    build_embedding_fingerprint,
    canonical_endpoint,
    embedding_model_version,
    fingerprint_matches,
    validate_bank_embedding_fingerprint,
)


def _model(
    *,
    provider: str,
    model: str,
    dimension: int = 384,
    endpoint: str | None = None,
    normalization: bool | str | None = True,
):
    return SimpleNamespace(
        provider_name=provider,
        model=model,
        dimension=dimension,
        base_url=endpoint,
        normalization=normalization,
    )


class _FingerprintConn:
    backend_type = "postgresql"

    def __init__(self, *, stored=None, nonempty: bool = False, row_exists: bool = True):
        self.stored = stored
        self.nonempty = nonempty
        self.row_exists = row_exists
        self.queries: list[tuple[str, tuple]] = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        if not self.row_exists:
            return None
        return {"embedding_fingerprint": self.stored}

    async def fetchval(self, query, *args):
        self.queries.append((query, args))
        return self.nonempty

    async def execute(self, query, *args):
        self.queries.append((query, args))
        self.stored = json.loads(args[1])
        return "UPDATE 1"


def test_bge_registry_aliases_are_transport_compatible():
    local = _model(provider="local", model="BAAI/bge-small-en-v1.5")
    proxy = _model(
        provider="openai",
        model="bge-small-en-v1.5",
        endpoint="https://adapter.example/v1",
    )

    local_fp = build_embedding_fingerprint(local)
    proxy_fp = build_embedding_fingerprint(proxy)

    assert local_fp["provider"] == "local"
    assert proxy_fp["provider"] == "openai"
    assert local_fp["model"] == proxy_fp["model"] == "bge-small-en-v1.5"
    assert local_fp["hash"] != proxy_fp["hash"]  # transport stays diagnostic
    assert local_fp["compatibility_hash"] == proxy_fp["compatibility_hash"]
    assert fingerprint_matches(local_fp, proxy_fp)
    assert embedding_model_version(local) == f"fp:{local_fp['compatibility_hash']}"


def test_same_dimension_different_models_are_incompatible():
    bge = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    openai = build_embedding_fingerprint(_model(provider="openai", model="text-embedding-3-small", dimension=384))

    assert bge["dimension"] == openai["dimension"] == 384
    assert not fingerprint_matches(bge, openai)


def test_endpoint_canonicalization_removes_credentials_and_query_secrets():
    endpoint = canonical_endpoint(
        "https://operator:password@EXAMPLE.com:8443/v1/key/provider-secret-value?api-key=query-secret#fragment"
    )

    assert endpoint == "https://example.com:8443/v1/key/_redacted_"
    for secret in ("operator", "password", "provider-secret-value", "query-secret", "api-key"):
        assert secret not in endpoint


@pytest.mark.asyncio
async def test_empty_bank_initializes_fingerprint_under_write_lock():
    current = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    conn = _FingerprintConn(nonempty=False)

    result = await validate_bank_embedding_fingerprint(
        conn,
        "bank-empty",
        current,
        for_write=True,
    )

    assert result["compatibility_hash"] == current["compatibility_hash"]
    assert conn.stored["hash"] == current["hash"]
    assert "FOR UPDATE" in conn.queries[0][0]
    assert any("UPDATE" in query and "embedding_fingerprint" in query for query, _ in conn.queries)


@pytest.mark.asyncio
async def test_nonempty_legacy_bank_requires_explicit_attestation():
    current = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    conn = _FingerprintConn(nonempty=True)

    with pytest.raises(EmbeddingFingerprintUnknownError, match="has no embedding fingerprint"):
        await validate_bank_embedding_fingerprint(conn, "bank-legacy", current, for_write=True)

    assert conn.stored is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attestation_factory",
    [
        lambda fp: json.dumps({"compatibility_hash": fp["compatibility_hash"]}),
        lambda fp: "BAAI/bge-small-en-v1.5:384",
    ],
)
async def test_explicit_legacy_attestation_is_atomically_persisted(attestation_factory):
    current = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    conn = _FingerprintConn(nonempty=True)

    await validate_bank_embedding_fingerprint(
        conn,
        "bank-legacy",
        current,
        for_write=True,
        legacy_attestation=attestation_factory(current),
    )

    assert conn.stored["compatibility_hash"] == current["compatibility_hash"]
    assert "FOR UPDATE" in conn.queries[0][0]


@pytest.mark.asyncio
async def test_strict_policy_rejects_a_mismatch():
    stored = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    current = build_embedding_fingerprint(_model(provider="openai", model="text-embedding-3-small", dimension=384))
    conn = _FingerprintConn(stored=stored, nonempty=True)

    with pytest.raises(EmbeddingFingerprintMismatchError, match="fingerprint mismatch"):
        await validate_bank_embedding_fingerprint(conn, "bank-mismatch", current, policy="strict")


@pytest.mark.asyncio
async def test_warn_policy_is_explicit_and_does_not_rewrite(caplog):
    stored = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    current = build_embedding_fingerprint(_model(provider="openai", model="text-embedding-3-small", dimension=384))
    conn = _FingerprintConn(stored=stored, nonempty=True)

    await validate_bank_embedding_fingerprint(conn, "bank-mismatch", current, policy="warn")

    assert "fingerprint mismatch" in caplog.text.lower()
    assert conn.stored == stored
    assert not any(query.lstrip().startswith("UPDATE") for query, _ in conn.queries)


@pytest.mark.asyncio
async def test_off_policy_skips_database_access():
    current = build_embedding_fingerprint(_model(provider="openai", model="text-embedding-3-small"))
    conn = _FingerprintConn(stored="not-json", nonempty=True)

    await validate_bank_embedding_fingerprint(conn, "bank-disabled", current, policy="off")

    assert conn.queries == []


@pytest.mark.asyncio
async def test_oracle_bank_emptiness_query_uses_dual():
    current = build_embedding_fingerprint(_model(provider="local", model="BAAI/bge-small-en-v1.5"))
    conn = _FingerprintConn(nonempty=False)
    conn.backend_type = "oracle"

    await validate_bank_embedding_fingerprint(conn, "bank-oracle", current)

    emptiness_query = conn.queries[1][0]
    assert "CASE WHEN EXISTS" in emptiness_query
    assert "FROM DUAL" in emptiness_query


def test_migration_postgresql_sql_is_nullable_and_has_no_legacy_backfill(monkeypatch):
    from hms_api.alembic.versions import q5r6s7t8u9v0_add_bank_embedding_fingerprint as migration

    statements: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"tenant".')
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration._pg_upgrade()
    migration._pg_downgrade()

    assert statements == [
        'ALTER TABLE "tenant".banks ADD COLUMN IF NOT EXISTS embedding_fingerprint jsonb',
        'ALTER TABLE "tenant".banks DROP COLUMN IF EXISTS embedding_fingerprint',
    ]
    assert all("UPDATE" not in statement for statement in statements)


def test_migration_oracle_sql_uses_clob_json_check_and_downgrade(monkeypatch):
    from hms_api.alembic.versions import q5r6s7t8u9v0_add_bank_embedding_fingerprint as migration

    statements: list[str] = []
    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"TENANT".')
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration._oracle_upgrade()
    migration._oracle_downgrade()

    assert "embedding_fingerprint CLOB" in statements[0]
    assert "embedding_fingerprint IS JSON" in statements[0]
    assert 'ALTER TABLE "TENANT".banks DROP COLUMN embedding_fingerprint' in statements[1]
