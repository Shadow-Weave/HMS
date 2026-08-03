"""Offline Oracle contracts for the Retain ingestion pipeline.

These tests exercise adapter selection and Oracle-specific SQL boundaries with
fakes. They do not replace validation against an Oracle 23ai instance.
"""

from __future__ import annotations

import copy
import json
import uuid
from array import array
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from hms_api.engine.db.ops_oracle import OracleOps
from hms_api.engine.db.oracle import _convert_arg, _rewrite_pg_to_oracle
from hms_api.engine.entity_resolution_contracts import (
    EntityOccurrenceBinding,
    EntityResolutionReadPlan,
    ExistingEntityBinding,
)
from hms_api.engine.entity_resolver import EntityResolver
from hms_api.engine.ingestion import runtime as runtime_module
from hms_api.engine.ingestion import service as service_module
from hms_api.engine.ingestion.adapters.oracle_semantic import (
    compute_oracle_semantic_links_ann,
)
from hms_api.engine.ingestion.adapters.postgres_fresh_ownership import (
    FreshDocumentOwnershipConflict,
    FreshPostgresDocumentOwnership,
)
from hms_api.engine.ingestion.persistence.backend import retain_backend_adapters
from hms_api.engine.ingestion.persistence.operation_fence import OperationActivityFence
from hms_api.engine.ingestion.persistence.oracle import (
    FreshOracleDocumentOwnership,
    OracleCheckpointStore,
    OracleDocumentOwnership,
    OraclePlanningRepository,
)
from hms_api.engine.ingestion.persistence.postgres import (
    PostgresCheckpointStore,
    PostgresDocumentOwnership,
    PostgresPlanningRepository,
)
from hms_api.engine.retain import chunk_storage


def test_projection_migration_executes_literal_json_as_raw_oracle_sql(monkeypatch) -> None:
    """Literal JSON colons must not be parsed as SQLAlchemy bind parameters."""

    from hms_api.alembic.versions import p4q5r6s7t8u9_add_memory_unit_projection as migration

    alembic_statements: list[str] = []
    driver_statements: list[str] = []
    bind = SimpleNamespace(exec_driver_sql=driver_statements.append)

    monkeypatch.setattr(migration, "_get_schema_prefix", lambda: '"TENANT".')
    monkeypatch.setattr(migration.op, "execute", alembic_statements.append)
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)

    migration._oracle_upgrade()

    assert len(alembic_statements) == 1
    assert 'ALTER TABLE "TENANT".memory_units ADD' in alembic_statements[0]
    assert len(driver_statements) == 1
    assert 'UPDATE "TENANT".memory_units mu' in driver_statements[0]
    assert '"embedding":{"v":1,"ok":' in driver_statements[0]
    assert "DBMS_LOB.COMPARE(projection, TO_CLOB('{}')) = 0" in driver_statements[0]
    assert "WHERE projection = '{}'" not in driver_statements[0]


class _Transaction:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self):
        self._events.append("begin")
        return self

    async def __aexit__(self, exc_type, _exc, _traceback):
        self._events.append("rollback" if exc_type is not None else "commit")
        return False


class _SnapshotConnection:
    def __init__(self) -> None:
        self.events: list[str] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self.events)

    async def execute(self, query: str, *_args):
        self.events.append(" ".join(query.split()))
        return "OK 0"


def test_backend_selector_exposes_matching_retain_adapters() -> None:
    postgres = retain_backend_adapters("postgresql")
    oracle = retain_backend_adapters("ORACLE")

    assert isinstance(postgres.planning_repository(object()), PostgresPlanningRepository)
    assert isinstance(postgres.checkpoint_store(object()), PostgresCheckpointStore)
    assert isinstance(postgres.document_ownership(), PostgresDocumentOwnership)
    assert isinstance(postgres.document_ownership(fresh=True), FreshPostgresDocumentOwnership)
    assert isinstance(postgres.operation_activity_fence(str(uuid.uuid4())), OperationActivityFence)
    assert postgres.operation_activity_fence(None) is None

    assert isinstance(oracle.planning_repository(object()), OraclePlanningRepository)
    assert isinstance(oracle.checkpoint_store(object()), OracleCheckpointStore)
    assert isinstance(oracle.document_ownership(), OracleDocumentOwnership)
    assert isinstance(oracle.document_ownership(fresh=True), FreshOracleDocumentOwnership)
    assert isinstance(oracle.operation_activity_fence(str(uuid.uuid4())), OperationActivityFence)
    assert oracle.operation_activity_fence(None) is None

    with pytest.raises(ValueError, match="Unsupported Retain database backend"):
        retain_backend_adapters("sqlite")


@pytest.mark.asyncio
async def test_backend_snapshots_preserve_database_transaction_rules() -> None:
    postgres_connection = _SnapshotConnection()
    async with retain_backend_adapters("postgresql").planning_snapshot(postgres_connection):
        postgres_connection.events.append("body")
    assert postgres_connection.events == [
        "begin",
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY",
        "body",
        "commit",
    ]

    oracle_connection = _SnapshotConnection()
    async with retain_backend_adapters("oracle").planning_snapshot(oracle_connection):
        oracle_connection.events.append("body")
    assert oracle_connection.events == [
        "SET TRANSACTION READ ONLY",
        "body",
    ]


def test_service_route_accepts_oracle_and_rejects_unknown_backends() -> None:
    oracle_execution = SimpleNamespace(
        pool=SimpleNamespace(backend_type="oracle"),
        resolved_config=SimpleNamespace(
            database_backend="oracle",
            retain_extraction_mode="chunks",
        ),
    )
    service_module._require_supported_route(oracle_execution)

    unknown_execution = SimpleNamespace(
        pool=SimpleNamespace(backend_type="sqlite"),
        resolved_config=SimpleNamespace(
            database_backend="sqlite",
            retain_extraction_mode="chunks",
        ),
    )
    with pytest.raises(service_module.RetainDatabaseUnsupportedError):
        service_module._require_supported_route(unknown_execution)


def test_oracle_rewrites_json_uuid_text_comparisons_to_raw_uuid_comparisons() -> None:
    parent_operation_id = uuid.uuid4()

    rewritten, ignore_duplicate, returning_columns = _rewrite_pg_to_oracle(
        """
        SELECT operation_id
        FROM async_operations
        WHERE bank_id = $1
          AND (result_metadata->>'parent_operation_id')::uuid = $2
        """
    )

    assert "HEXTORAW(REPLACE(JSON_VALUE(result_metadata, '$.parent_operation_id'), '-', '')) = :2" in rewritten
    assert "->>" not in rewritten
    assert "::uuid" not in rewritten
    assert ignore_duplicate is False
    assert returning_columns is None
    assert _convert_arg(parent_operation_id) == parent_operation_id.bytes
    assert _convert_arg(str(parent_operation_id)) == parent_operation_id.bytes


def test_oracle_preserves_numeric_ordering_for_json_text_casts() -> None:
    rewritten, _, _ = _rewrite_pg_to_oracle(
        "SELECT operation_id FROM async_operations ORDER BY (result_metadata->>'sub_batch_index')::int"
    )

    assert "ORDER BY TO_NUMBER(JSON_VALUE(result_metadata, '$.sub_batch_index'))" in rewritten
    assert "->>" not in rewritten
    assert "::int" not in rewritten


class _PlanningConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.query = ""
        self.args: tuple = ()

    async def fetch(self, query: str, *args):
        self.query = query
        self.args = args
        return self.rows


@pytest.mark.asyncio
async def test_oracle_planning_reorders_checkpoint_bindings_in_python() -> None:
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    connection = _PlanningConnection(
        [
            {"unit_id": first_id, "chunk_index": 0},
            {"unit_id": second_id, "chunk_index": 1},
        ]
    )
    repository = OraclePlanningRepository(connection, schema="tenant")

    bindings = await repository.load_document_unit_bindings(
        "bank",
        "document",
        expected_unit_ids=(str(second_id), str(first_id)),
    )

    assert [binding.unit_id for binding in bindings] == [str(second_id), str(first_id)]
    assert [binding.chunk_index for binding in bindings] == [1, 0]
    assert connection.args == ("bank", "document")
    assert "array_position" not in connection.query
    assert "ANY(" not in connection.query
    assert '"tenant".memory_units' in connection.query

    with pytest.raises(ValueError, match="checkpoint unit IDs are missing"):
        await repository.load_document_unit_bindings(
            "bank",
            "document",
            expected_unit_ids=(str(uuid.uuid4()),),
        )


class _OwnershipConnection:
    def __init__(self, *, insert_status: str = "INSERT 0 1", locked_value: str | None = "document") -> None:
        self.insert_status = insert_status
        self.locked_value = locked_value
        self.queries: list[str] = []

    async def execute(self, query: str, *_args):
        self.queries.append(query)
        if query.lstrip().startswith("INSERT"):
            return self.insert_status
        return "UPDATE 1"

    async def fetchval(self, query: str, *_args):
        self.queries.append(query)
        return self.locked_value


class _UnhashedOwnershipConnection:
    def __init__(self, row) -> None:
        self.row = row
        self.query = ""
        self.args: tuple = ()

    async def fetchrow(self, query: str, *args):
        self.query = query
        self.args = args
        return self.row


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership_type", [PostgresDocumentOwnership, OracleDocumentOwnership])
@pytest.mark.parametrize(
    "row,expected",
    [
        ({"content_hash": None}, True),
        ({"content_hash": ""}, True),
        ({"content_hash": "new-hash"}, False),
        (None, False),
    ],
)
async def test_unhashed_ownership_requires_a_locked_existing_row(ownership_type, row, expected) -> None:
    connection = _UnhashedOwnershipConnection(row)

    result = await ownership_type(schema="tenant").validate_unhashed_window(
        connection,
        bank_id="bank",
        document_id="document",
    )

    assert result is expected
    assert "FOR UPDATE" in connection.query
    assert '"tenant".documents' in connection.query
    assert connection.args == ("document", "bank")


@pytest.mark.asyncio
async def test_oracle_fresh_ownership_claim_and_transition_use_row_counts() -> None:
    connection = _OwnershipConnection()
    ownership = FreshOracleDocumentOwnership(schema="tenant")

    await ownership.prepare_first_window(
        connection,
        bank_id="bank",
        document_id="document",
    )
    transitioned = await ownership.transition_content_hash(
        connection,
        bank_id="bank",
        document_id="document",
        expected_content_hash="old",
        new_content_hash="new",
    )

    assert transitioned is True
    assert "FOR UPDATE" in connection.queries[1]
    assert "RETURNING" not in connection.queries[2]

    conflict_connection = _OwnershipConnection(insert_status="INSERT 0 0")
    with pytest.raises(FreshDocumentOwnershipConflict):
        await ownership.prepare_first_window(
            conflict_connection,
            bank_id="bank",
            document_id="document",
        )


class _CheckpointConnection:
    def __init__(self, metadata: dict) -> None:
        self.metadata = copy.deepcopy(metadata)
        self.events: list[str] = []
        self.queries: list[str] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self.events)

    async def fetchrow(self, query: str, *_args):
        self.queries.append(query)
        return {"result_metadata": json.dumps(self.metadata)}

    async def execute(self, query: str, *args):
        self.queries.append(query)
        self.metadata = json.loads(args[0])
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_oracle_checkpoint_read_modify_write_preserves_resume_contract() -> None:
    operation_id = str(uuid.uuid4())
    connection = _CheckpointConnection(
        {
            "keep": {"caller": "value"},
            "batch_id": "provider-job",
            "batch_provider": "provider",
            "chunk_count": 2,
        }
    )
    store = OracleCheckpointStore(connection, schema="tenant")

    await store.record_document_id(operation_id, "document-a")
    await store.record_document_id(operation_id, "document-a")
    await store.record_core_committed(
        operation_id,
        "document-a",
        unit_ids=("unit-b", "unit-a"),
        requires_final_ann=True,
    )
    checkpoint = await store.recover(operation_id)

    assert checkpoint.document_ids == ("document-a",)
    assert checkpoint.core_committed_document_ids == ("document-a",)
    assert checkpoint.final_ann_pending_document_ids == ("document-a",)
    assert checkpoint.unit_ids_for_document("document-a") == ("unit-b", "unit-a")
    assert checkpoint.unscoped_facts_committed is True
    assert connection.metadata["keep"] == {"caller": "value"}

    await store.record_core_committed(
        operation_id,
        "document-empty",
        unit_ids=(),
        requires_final_ann=True,
    )
    checkpoint = await store.recover(operation_id)
    assert checkpoint.unit_ids_for_document("document-empty") == ()
    assert checkpoint.final_ann_pending_document_ids == ("document-a",)

    await store.record_final_ann_completed(operation_id, "document-a")
    await store.clear_provider_batch(operation_id)
    checkpoint = await store.recover(operation_id)

    assert checkpoint.final_ann_pending_document_ids == ()
    assert connection.metadata["keep"] == {"caller": "value"}
    assert "batch_id" not in connection.metadata
    assert "batch_provider" not in connection.metadata
    assert "chunk_count" not in connection.metadata
    assert all("jsonb_" not in query.lower() for query in connection.queries)
    assert any("FOR UPDATE" in query for query in connection.queries)


class _SemanticConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *args):
        self.calls.append((query, args))
        return [
            {"to_id": uuid.uuid4(), "similarity": 0.82},
            {"to_id": uuid.uuid4(), "similarity": 0.64},
        ]


@pytest.mark.asyncio
async def test_oracle_semantic_ann_uses_vector_selects_without_postgres_ddl() -> None:
    connection = _SemanticConnection()

    links = await compute_oracle_semantic_links_ann(
        connection,
        "bank",
        ["seed"],
        [array("f", [0.1, 0.2, 0.3])],
        fact_types=["world"],
        top_k=20,
        threshold=0.7,
    )

    assert len(links) == 1
    assert links[0][0] == "seed"
    assert isinstance(links[0][1], str)
    query, args = connection.calls[0]
    assert "VECTOR_DISTANCE" in query
    assert "FETCH FIRST 20 ROWS ONLY" in query
    assert args[:2] == ("bank", "world")
    assert isinstance(args[2], array)
    assert list(args[2]) == pytest.approx([0.1, 0.2, 0.3])
    assert all(token not in query for token in ("SET LOCAL", "unnest", "TEMP TABLE", "LATERAL"))


@pytest.mark.asyncio
async def test_oracle_entity_lookup_preserves_original_input_name() -> None:
    entity_id = uuid.uuid4()

    class _EntityConnection:
        async def fetchrow(self, _query: str, _bank_id: str, input_name: str):
            return {
                "id": entity_id,
                "name_lower": input_name.lower(),
            }

    rows = await OracleOps().fetch_missing_entity_ids(
        _EntityConnection(),
        "entities",
        "bank",
        ["Mixed Case"],
    )

    assert len(rows) == 1
    assert rows[0]["id"] == entity_id
    assert rows[0]["name_lower"] == "mixed case"
    assert rows[0]["input_name"] == "Mixed Case"


class _ExistingEntityValidationConnection:
    def __init__(self, backend_type: str) -> None:
        self.backend_type = backend_type
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *args):
        self.calls.append((query, args))
        if self.backend_type == "oracle":
            assert len(args[1]) <= 900
        return [{"id": entity_id} for entity_id in args[1]]


@pytest.mark.parametrize(
    ("backend_type", "expected_batch_sizes"),
    (("oracle", [900, 101]), ("postgresql", [1001])),
)
@pytest.mark.asyncio
async def test_existing_entity_validation_chunks_only_oracle_bind_lists(
    backend_type: str,
    expected_batch_sizes: list[int],
) -> None:
    entity_ids = [str(uuid.uuid4()) for _ in range(1001)]
    occurrences = tuple(
        EntityOccurrenceBinding(
            occurrence_key=f"occurrence-{index}",
            unit_key=f"unit-{index}",
            local_index=0,
            event_date=None,
        )
        for index in range(len(entity_ids))
    )
    plan = EntityResolutionReadPlan(
        bank_id="bank",
        occurrences=occurrences,
        existing_bindings=tuple(
            ExistingEntityBinding(
                occurrence_key=occurrence.occurrence_key,
                entity_id=entity_id,
            )
            for occurrence, entity_id in zip(occurrences, entity_ids, strict=True)
        ),
    )
    connection = _ExistingEntityValidationConnection(backend_type)
    resolver = EntityResolver(SimpleNamespace(ops=OracleOps()))

    finalized = await resolver.finalize_entity_read_plan(
        connection,
        "bank",
        plan,
        entities_table="entities",
    )

    assert [len(call_args[1]) for _query, call_args in connection.calls] == expected_batch_sizes
    assert list(finalized.resolved_entity_ids) == entity_ids


class _ChunkDeletionConnection:
    def __init__(self, backend_type: str) -> None:
        self.backend_type = backend_type
        self.calls: list[tuple[str, tuple]] = []

    async def execute(self, query: str, *args):
        self.calls.append((query, args))
        if self.backend_type == "oracle":
            assert len(args[0]) <= 900
        return f"DELETE {len(args[0])}"


@pytest.mark.parametrize(
    ("backend_type", "expected_batch_sizes"),
    (("oracle", [900, 101]), ("postgresql", [1001])),
)
@pytest.mark.asyncio
async def test_delta_chunk_deletion_chunks_only_oracle_bind_lists(
    backend_type: str,
    expected_batch_sizes: list[int],
) -> None:
    connection = _ChunkDeletionConnection(backend_type)

    await chunk_storage.delete_chunks_by_ids(
        connection,
        [f"chunk-{index}" for index in range(1001)],
    )

    assert [len(call_args[0]) for _query, call_args in connection.calls] == expected_batch_sizes
    assert all("ANY($1::text[])" in query for query, _call_args in connection.calls)


@pytest.mark.asyncio
async def test_runtime_oracle_planning_uses_read_only_snapshot_and_oracle_ann(monkeypatch) -> None:
    connection = _SnapshotConnection()
    oracle_ann_calls: list[tuple] = []

    @asynccontextmanager
    async def acquire(_pool):
        yield connection

    async def plan_entities(*_args, **_kwargs):
        return SimpleNamespace(occurrences=())

    async def oracle_ann(*args, **kwargs):
        oracle_ann_calls.append((args, kwargs))
        return [("0", "neighbor", "semantic", 0.9, None)]

    async def postgres_ann(*_args, **_kwargs):
        raise AssertionError("PostgreSQL ANN must not run for Oracle")

    monkeypatch.setattr(runtime_module, "acquire_with_retry", acquire)
    monkeypatch.setattr(runtime_module.entity_processing, "plan_entities", plan_entities)
    monkeypatch.setattr(runtime_module, "compute_oracle_semantic_links_ann", oracle_ann)
    monkeypatch.setattr(runtime_module, "compute_semantic_links_ann", postgres_ann)

    result = await runtime_module.pre_resolve_entities(
        SimpleNamespace(backend_type="oracle"),
        object(),
        "bank",
        [SimpleNamespace(entities=[])],
        ["fact-key"],
        [SimpleNamespace(embedding=[0.1, 0.2], fact_type="world")],
        SimpleNamespace(entity_labels=None),
        [],
    )

    assert connection.events == ["SET TRANSACTION READ ONLY"]
    assert len(oracle_ann_calls) == 1
    assert result.semantic_ann_links == [("0", "neighbor", "semantic", 0.9, None)]


@pytest.mark.asyncio
async def test_runtime_final_oracle_ann_maps_driver_uuid_rows_to_string_ids(monkeypatch) -> None:
    unit_id = str(uuid.uuid4())
    inserted_links: list[tuple] = []

    class _FinalAnnConnection:
        async def fetch(self, query: str, *_args):
            if "VECTOR_DISTANCE" in query:
                return [{"to_id": uuid.uuid4(), "similarity": 0.91}]
            return [
                {
                    "id": uuid.UUID(unit_id),
                    "embedding": array("f", [0.1, 0.2]),
                    "fact_type": "world",
                }
            ]

    connection = _FinalAnnConnection()

    @asynccontextmanager
    async def acquire(_pool):
        yield connection

    async def insert_links(_connection, links, **_kwargs):
        inserted_links.extend(links)

    monkeypatch.setattr(runtime_module, "acquire_with_retry", acquire)
    monkeypatch.setattr(runtime_module, "_bulk_insert_links", insert_links)

    await runtime_module.run_final_semantic_ann(
        SimpleNamespace(backend_type="oracle", ops=object()),
        "bank",
        [unit_id],
        SimpleNamespace(write_semantic_links=True),
        [],
    )

    assert len(inserted_links) == 1
    assert inserted_links[0][0] == unit_id
    assert isinstance(inserted_links[0][1], str)
