"""Real Milvus Lite tests for the optional semantic vector index."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter

from hms_api.engine.search.tags import TagGroup
from hms_api.engine.vector_index import VectorIndexRecord
from hms_api.engine.vector_index.milvus import MilvusVectorIndex, build_filter_expression

requires_pymilvus = pytest.mark.skipif(
    importlib.util.find_spec("pymilvus") is None,
    reason="pymilvus is not installed",
)


def _record(
    record_id: int,
    embedding: list[float],
    *,
    namespace: str = "public",
    bank_id: str = "bank-a",
    fact_type: str = "world",
    document_id: str | None = None,
    tags: list[str] | None = None,
    updated_at: datetime | None = None,
) -> VectorIndexRecord:
    return VectorIndexRecord(
        id=f"00000000-0000-0000-0000-{record_id:012d}",
        namespace=namespace,
        bank_id=bank_id,
        fact_type=fact_type,
        embedding=embedding,
        document_id=document_id,
        tags=tags or [],
        updated_at=updated_at or datetime.now(UTC),
    )


@pytest.mark.asyncio
@requires_pymilvus
async def test_milvus_lite_search_filters_isolation_and_deletes(tmp_path):
    index = MilvusVectorIndex(
        uri=str(tmp_path / "hms.db"),
        collection_name="hms_test_filters",
    )
    await index.initialize(3)
    now = datetime.now(UTC)
    await index.upsert(
        [
            _record(1, [1.0, 0.0, 0.0], document_id="doc-a", tags=["user:alice", "priority:high"], updated_at=now),
            _record(2, [0.8, 0.2, 0.0], document_id="doc-b", tags=["user:bob"], updated_at=now - timedelta(days=2)),
            _record(3, [0.0, 1.0, 0.0], document_id="doc-c", tags=[], fact_type="experience", updated_at=now),
            _record(4, [1.0, 0.0, 0.0], document_id="doc-d", tags=["user:alice"], bank_id="bank-b", updated_at=now),
            _record(1, [1.0, 0.0, 0.0], document_id="doc-e", tags=["user:alice"], namespace="tenant-b", updated_at=now),
        ]
    )

    hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
    )
    assert [hit.id for hit in hits["world"]] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    assert hits["world"][0].similarity > hits["world"][1].similarity

    strict_hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        tags=["user:alice", "priority:high"],
        tags_match="all_strict",
    )
    assert [hit.id for hit in strict_hits["world"]] == ["00000000-0000-0000-0000-000000000001"]

    inclusive_hits = await index.search(
        [0.0, 1.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["experience"],
        limit=10,
        tags=["unrelated"],
        tags_match="any",
    )
    assert [hit.id for hit in inclusive_hits["experience"]] == ["00000000-0000-0000-0000-000000000003"]

    adapter = TypeAdapter(TagGroup)
    tag_groups = [
        adapter.validate_python(
            {
                "and": [
                    {"tags": ["user:alice"], "match": "all_strict"},
                    {"not": {"tags": ["archived"], "match": "any_strict"}},
                ]
            }
        )
    ]
    grouped_hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        tag_groups=tag_groups,
    )
    assert [hit.id for hit in grouped_hits["world"]] == ["00000000-0000-0000-0000-000000000001"]

    recent_hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        created_after=now - timedelta(days=1),
    )
    assert [hit.id for hit in recent_hits["world"]] == ["00000000-0000-0000-0000-000000000001"]

    historical_hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        created_before=now - timedelta(days=1),
    )
    assert [hit.id for hit in historical_hits["world"]] == ["00000000-0000-0000-0000-000000000002"]

    await index.delete_document("public", "bank-a", "doc-a")
    hits = await index.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
    )
    assert [hit.id for hit in hits["world"]] == ["00000000-0000-0000-0000-000000000002"]

    await index.delete_units("public", ["00000000-0000-0000-0000-000000000002"])
    assert not (
        await index.search(
            [1.0, 0.0, 0.0],
            namespace="public",
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
        )
    )["world"]

    await index.delete_bank("public", "bank-a", fact_type="experience")
    assert not (
        await index.search(
            [0.0, 1.0, 0.0],
            namespace="public",
            bank_id="bank-a",
            fact_types=["experience"],
            limit=10,
        )
    )["experience"]

    await index.delete_namespace("tenant-b")
    assert not (
        await index.search(
            [1.0, 0.0, 0.0],
            namespace="tenant-b",
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
        )
    )["world"]
    await index.close()


@pytest.mark.asyncio
@requires_pymilvus
async def test_milvus_lite_reuses_compatible_schema_and_rejects_dimension_mismatch(tmp_path):
    from pymilvus import MilvusClient

    uri = str(tmp_path / "reuse.db")
    first = MilvusVectorIndex(uri=uri, collection_name="hms_test_reuse")
    await first.initialize(3)
    await first.upsert([_record(10, [1.0, 0.0, 0.0])])
    await first.close()

    client = MilvusClient(uri=uri)
    client.release_collection(collection_name="hms_test_reuse")
    client.close()

    second = MilvusVectorIndex(uri=uri, collection_name="hms_test_reuse")
    await second.initialize(3)
    hits = await second.search(
        [1.0, 0.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
    )
    assert [hit.id for hit in hits["world"]] == ["00000000-0000-0000-0000-000000000010"]
    await second.close()

    mismatched = MilvusVectorIndex(uri=uri, collection_name="hms_test_reuse")
    with pytest.raises(ValueError, match="embedding dimension 3"):
        await mismatched.initialize(4)


@pytest.mark.asyncio
@requires_pymilvus
async def test_milvus_lite_upsert_replaces_metadata_and_vector(tmp_path):
    index = MilvusVectorIndex(uri=str(tmp_path / "upsert.db"), collection_name="hms_test_upsert")
    await index.initialize(3)
    record = _record(20, [1.0, 0.0, 0.0], document_id="before", tags=["old"])
    await index.upsert([record])
    await index.upsert(
        [
            _record(
                20,
                [0.0, 1.0, 0.0],
                document_id="after",
                tags=["new"],
                updated_at=datetime.now(UTC) + timedelta(seconds=1),
            )
        ]
    )

    old_tag_hits = await index.search(
        [0.0, 1.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        tags=["old"],
        tags_match="any_strict",
    )
    assert not old_tag_hits["world"]

    new_tag_hits = await index.search(
        [0.0, 1.0, 0.0],
        namespace="public",
        bank_id="bank-a",
        fact_types=["world"],
        limit=10,
        tags=["new"],
        tags_match="any_strict",
    )
    assert [hit.id for hit in new_tag_hits["world"]] == ["00000000-0000-0000-0000-000000000020"]
    assert new_tag_hits["world"][0].similarity == pytest.approx(1.0)

    await index.delete_document("public", "bank-a", "after")
    assert not (
        await index.search(
            [0.0, 1.0, 0.0],
            namespace="public",
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
        )
    )["world"]
    await index.close()


@pytest.mark.asyncio
@requires_pymilvus
async def test_milvus_lite_supports_all_tag_match_modes(tmp_path):
    index = MilvusVectorIndex(uri=str(tmp_path / "tag-modes.db"), collection_name="hms_test_tag_modes")
    await index.initialize(3)
    await index.upsert(
        [
            _record(21, [1.0, 0.0, 0.0], tags=["a", "b"]),
            _record(22, [0.9, 0.1, 0.0], tags=[]),
        ]
    )

    expected = {
        "any": {"00000000-0000-0000-0000-000000000021", "00000000-0000-0000-0000-000000000022"},
        "any_strict": {"00000000-0000-0000-0000-000000000021"},
        "all": {"00000000-0000-0000-0000-000000000022"},
        "all_strict": set(),
    }
    for match, expected_ids in expected.items():
        hits = await index.search(
            [1.0, 0.0, 0.0],
            namespace="public",
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
            tags=["a", "c"],
            tags_match=match,
        )
        assert {hit.id for hit in hits["world"]} == expected_ids

    await index.close()


@pytest.mark.asyncio
@requires_pymilvus
async def test_milvus_lite_rejects_incompatible_index_and_vector_dimensions(tmp_path):
    from pymilvus import MilvusClient

    uri = str(tmp_path / "incompatible.db")
    collection_name = "hms_test_incompatible"
    index = MilvusVectorIndex(uri=uri, collection_name=collection_name)
    await index.initialize(3)
    await index.close()

    client = MilvusClient(uri=uri)
    client.release_collection(collection_name=collection_name)
    client.drop_index(collection_name=collection_name, index_name="embedding")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="L2")
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.close()

    mismatched_index = MilvusVectorIndex(uri=uri, collection_name=collection_name)
    with pytest.raises(ValueError, match="must use COSINE"):
        await mismatched_index.initialize(3)

    fresh = MilvusVectorIndex(uri=str(tmp_path / "dimensions.db"), collection_name="hms_test_dimensions")
    await fresh.initialize(3)
    with pytest.raises(ValueError, match="embedding dimension 2"):
        await fresh.upsert([_record(30, [1.0, 0.0])])
    with pytest.raises(ValueError, match="Query embedding dimension 2"):
        await fresh.search(
            [1.0, 0.0],
            namespace="public",
            bank_id="bank-a",
            fact_types=["world"],
            limit=10,
        )
    await fresh.close()


def test_milvus_filter_expression_escapes_user_controlled_values():
    expression = build_filter_expression(
        namespace='tenant" or true or "',
        bank_id='bank" or true or "',
        fact_type='world" or true or "',
        document_id='doc" or true or "',
        tags=['tag" or true or "'],
        tags_match="any_strict",
    )

    assert '\\" or true or \\"' in expression
    assert 'fact_type == "world\\" or true or \\""' in expression
    assert 'document_id == "doc\\" or true or \\""' in expression
