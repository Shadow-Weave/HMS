"""Milvus semantic vector-index provider using ``MilvusClient``."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from typing import Any
from urllib.parse import urlparse

from ..search.tags import (
    TagGroup,
    TagGroupAnd,
    TagGroupLeaf,
    TagGroupNot,
    TagGroupOr,
    TagsMatch,
)
from .base import VectorIndex, VectorIndexRecord, VectorSearchHit

logger = logging.getLogger(__name__)

_EXPECTED_FIELDS = {
    "pk",
    "memory_id",
    "namespace",
    "tenant_key",
    "bank_id",
    "fact_type",
    "document_id",
    "tags",
    "updated_at_ms",
    "embedding",
}

_VARCHAR_LIMITS = {
    "memory_id": 64,
    "namespace": 512,
    "bank_id": 65535,
    "fact_type": 64,
    "document_id": 65535,
}


def _quoted(value: str) -> str:
    """Encode a string literal safely for a Milvus filter expression."""

    return json.dumps(value, ensure_ascii=False)


def _quoted_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _tenant_key(namespace: str, bank_id: str) -> str:
    return hashlib.sha256(f"{namespace}\0{bank_id}".encode()).hexdigest()


def _record_key(namespace: str, memory_id: str) -> str:
    return hashlib.sha256(f"{namespace}\0{memory_id}".encode()).hexdigest()


def _datetime_to_millis(value: datetime | None) -> int:
    if value is None:
        return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _build_tag_leaf_filter(tags: list[str], match: TagsMatch) -> str:
    if not tags:
        return ""

    strict = match in ("any_strict", "all_strict")
    any_match = match in ("any", "any_strict")
    function = "array_contains_any" if any_match else "array_contains_all"
    match_expr = f"{function}(tags, {_quoted_list(tags)})"
    if strict:
        return match_expr
    return f"(array_length(tags) == 0 or {match_expr})"


def _build_tag_group_filter(group: TagGroup) -> str:
    if isinstance(group, TagGroupLeaf):
        return _build_tag_leaf_filter(group.tags, group.match)
    if isinstance(group, TagGroupAnd):
        parts = [part for child in group.filters if (part := _build_tag_group_filter(child))]
        return f"({' and '.join(parts)})" if parts else ""
    if isinstance(group, TagGroupOr):
        parts = [part for child in group.filters if (part := _build_tag_group_filter(child))]
        return f"({' or '.join(parts)})" if parts else ""
    if isinstance(group, TagGroupNot):
        child = _build_tag_group_filter(group.filter)
        return f"not ({child})" if child else ""
    raise TypeError(f"Unsupported tag group type: {type(group).__name__}")


def build_filter_expression(
    *,
    namespace: str,
    bank_id: str,
    fact_type: str | None = None,
    document_id: str | None = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> str:
    """Build a validated Milvus scalar-filter expression."""

    parts = [f"tenant_key == {_quoted(_tenant_key(namespace, bank_id))}"]
    if fact_type is not None:
        parts.append(f"fact_type == {_quoted(fact_type)}")
    if document_id is not None:
        parts.append(f"document_id == {_quoted(document_id)}")
    if tags:
        parts.append(_build_tag_leaf_filter(tags, tags_match))
    if tag_groups:
        parts.extend(part for group in tag_groups if (part := _build_tag_group_filter(group)))
    if created_after is not None:
        parts.append(f"updated_at_ms > {_datetime_to_millis(created_after)}")
    if created_before is not None:
        parts.append(f"updated_at_ms < {_datetime_to_millis(created_before)}")
    return " and ".join(parts)


def _is_local_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return not parsed.scheme or parsed.scheme == "file"


def _has_lite_3_cosine_distance_bug(uri: str) -> bool:
    if not _is_local_uri(uri):
        return False
    try:
        version = importlib.metadata.version("milvus-lite")
    except importlib.metadata.PackageNotFoundError:
        return False
    normalized = version.split("+", 1)[0]
    return normalized in {"3.0", "3.0.0"}


class MilvusVectorIndex(VectorIndex):
    """Rebuildable Milvus projection for dense semantic recall."""

    def __init__(
        self,
        *,
        uri: str,
        token: str | None = None,
        db_name: str | None = None,
        collection_name: str = "hms_memory_units",
        consistency_level: str = "Session",
    ) -> None:
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self._collection_name = collection_name
        self._consistency_level = consistency_level
        self._is_local = _is_local_uri(uri)
        self._client: Any | None = None
        self._dimension: int | None = None
        self._initialize_lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._lite_cosine_distance_bug = _has_lite_3_cosine_distance_bug(uri)

    @property
    def provider_name(self) -> str:
        return "milvus"

    @property
    def is_external(self) -> bool:
        return True

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _import_pymilvus(self):
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise ImportError(
                "Milvus vector indexing requires the optional dependency. Install HMS with the 'milvus' extra."
            ) from exc
        return MilvusClient, DataType

    async def initialize(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Milvus embedding dimension must be greater than zero")
        async with self._initialize_lock:
            if self._client is not None:
                if self._dimension != dimension:
                    raise ValueError(
                        f"Milvus vector index is already initialized with dimension {self._dimension}, not {dimension}"
                    )
                return
            try:
                # Milvus Lite owns native resources whose construction and
                # destruction must stay on the process main thread. Startup is
                # infrequent, so initialize synchronously here and use the
                # dedicated executor only for steady-state operations.
                self._initialize_sync(dimension)
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hms-milvus")
            except Exception:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                self._executor = None
                if self._client is not None:
                    self._client.close()
                self._client = None
                self._dimension = None
                raise

    async def _run_sync(self, function, *args):
        if self._executor is None:
            raise RuntimeError("Milvus vector index executor is not initialized")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(function, *args))

    def _initialize_sync(self, dimension: int) -> None:
        MilvusClient, DataType = self._import_pymilvus()
        client_kwargs: dict[str, Any] = {"uri": self._uri}
        if self._token:
            client_kwargs["token"] = self._token
        if self._db_name:
            client_kwargs["db_name"] = self._db_name

        client = MilvusClient(**client_kwargs)
        try:
            if client.has_collection(collection_name=self._collection_name):
                self._validate_collection(client, DataType, dimension)
                client.load_collection(collection_name=self._collection_name)
            else:
                schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
                schema.add_field(field_name="memory_id", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="namespace", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(
                    field_name="tenant_key",
                    datatype=DataType.VARCHAR,
                    max_length=64,
                    is_partition_key=True,
                )
                schema.add_field(field_name="bank_id", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="fact_type", datatype=DataType.VARCHAR, max_length=64)
                schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(
                    field_name="tags",
                    datatype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=1024,
                    max_length=4096,
                )
                schema.add_field(field_name="updated_at_ms", datatype=DataType.INT64)
                schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)

                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="embedding",
                    index_type="AUTOINDEX",
                    metric_type="COSINE",
                )
                create_kwargs: dict[str, Any] = {
                    "collection_name": self._collection_name,
                    "schema": schema,
                    "index_params": index_params,
                }
                # Milvus Lite always provides strong local visibility and does
                # not implement the timestamp allocation RPC used by explicit
                # consistency settings.
                if not self._is_local:
                    create_kwargs["consistency_level"] = self._consistency_level
                client.create_collection(
                    **create_kwargs,
                )
        except Exception:
            client.close()
            raise

        self._client = client
        self._dimension = dimension
        if self._lite_cosine_distance_bug:
            logger.warning(
                "Milvus Lite 3.0 reports COSINE distance instead of similarity; "
                "HMS will normalize scores for this known version"
            )
        logger.info(
            "Milvus vector index initialized: collection=%s, dimension=%d",
            self._collection_name,
            dimension,
        )

    def _validate_collection(self, client: Any, data_type: Any, dimension: int) -> None:
        description = client.describe_collection(collection_name=self._collection_name)
        fields = description.get("fields") or description.get("schema", {}).get("fields") or []
        fields_by_name = {field.get("name") or field.get("field_name"): field for field in fields}
        missing = sorted(_EXPECTED_FIELDS - set(fields_by_name))
        if missing:
            raise ValueError(
                f"Milvus collection '{self._collection_name}' has an incompatible schema; "
                f"missing fields: {', '.join(missing)}"
            )
        if description.get("auto_id"):
            raise ValueError(f"Milvus collection '{self._collection_name}' must disable automatic primary keys")

        expected_types = {
            "pk": data_type.VARCHAR,
            "memory_id": data_type.VARCHAR,
            "namespace": data_type.VARCHAR,
            "tenant_key": data_type.VARCHAR,
            "bank_id": data_type.VARCHAR,
            "fact_type": data_type.VARCHAR,
            "document_id": data_type.VARCHAR,
            "tags": data_type.ARRAY,
            "updated_at_ms": data_type.INT64,
            "embedding": data_type.FLOAT_VECTOR,
        }
        incompatible_types = [
            name
            for name, expected_type in expected_types.items()
            if fields_by_name[name].get("type", fields_by_name[name].get("datatype")) != expected_type
        ]
        if incompatible_types:
            raise ValueError(
                f"Milvus collection '{self._collection_name}' has incompatible field types: "
                f"{', '.join(sorted(incompatible_types))}"
            )

        if not fields_by_name["pk"].get("is_primary"):
            raise ValueError(f"Milvus collection '{self._collection_name}' must use 'pk' as its primary key")
        if not fields_by_name["tenant_key"].get("is_partition_key"):
            raise ValueError(f"Milvus collection '{self._collection_name}' must use 'tenant_key' as its partition key")
        if fields_by_name["tags"].get("element_type") != data_type.VARCHAR:
            raise ValueError(
                f"Milvus collection '{self._collection_name}' must store 'tags' as an array of VARCHAR values"
            )
        for field_name, required_length in _VARCHAR_LIMITS.items():
            params = fields_by_name[field_name].get("params") or fields_by_name[field_name].get("type_params") or {}
            actual_length = params.get("max_length")
            if actual_length is None or int(actual_length) < required_length:
                raise ValueError(
                    f"Milvus collection '{self._collection_name}' field '{field_name}' must have "
                    f"max_length >= {required_length}"
                )
        pk_params = fields_by_name["pk"].get("params") or fields_by_name["pk"].get("type_params") or {}
        if int(pk_params.get("max_length", 0)) < 64:
            raise ValueError(f"Milvus collection '{self._collection_name}' field 'pk' must have max_length >= 64")
        tags_params = fields_by_name["tags"].get("params") or fields_by_name["tags"].get("type_params") or {}
        if int(tags_params.get("max_capacity", 0)) < 1024:
            raise ValueError(f"Milvus collection '{self._collection_name}' field 'tags' must have max_capacity >= 1024")

        embedding_field = fields_by_name["embedding"]
        params = embedding_field.get("params") or embedding_field.get("type_params") or {}
        existing_dimension = params.get("dim") or embedding_field.get("dim")
        if existing_dimension is None:
            raise ValueError(f"Milvus collection '{self._collection_name}' does not expose the embedding dimension")
        if int(existing_dimension) != dimension:
            raise ValueError(
                f"Milvus collection '{self._collection_name}' uses embedding dimension "
                f"{existing_dimension}, but HMS is configured for {dimension}. "
                "Use a new collection name or recreate and rebuild the collection."
            )

        embedding_indexes = []
        for index_name in client.list_indexes(collection_name=self._collection_name):
            index = client.describe_index(collection_name=self._collection_name, index_name=index_name)
            if index.get("field_name") == "embedding":
                embedding_indexes.append(index)
        if not embedding_indexes:
            raise ValueError(
                f"Milvus collection '{self._collection_name}' does not have an index on the embedding field"
            )
        if not any(str(index.get("metric_type", "")).upper() == "COSINE" for index in embedding_indexes):
            raise ValueError(f"Milvus collection '{self._collection_name}' must use COSINE for the embedding index")

    def _require_client(self) -> Any:
        if self._client is None or self._dimension is None:
            raise RuntimeError("Milvus vector index is not initialized")
        return self._client

    def _record_to_entity(self, record: VectorIndexRecord) -> dict[str, Any]:
        if self._dimension is None:
            raise RuntimeError("Milvus vector index is not initialized")
        if len(record.embedding) != self._dimension:
            raise ValueError(
                f"Memory unit {record.id} has embedding dimension {len(record.embedding)}, expected {self._dimension}"
            )
        values = {
            "memory_id": record.id,
            "namespace": record.namespace,
            "bank_id": record.bank_id,
            "fact_type": record.fact_type,
            "document_id": record.document_id or "",
        }
        for field_name, value in values.items():
            max_length = _VARCHAR_LIMITS[field_name]
            if len(value) > max_length:
                raise ValueError(f"Memory unit {record.id} has {field_name} longer than {max_length} characters")
        tags = record.tags or []
        if len(tags) > 1024:
            raise ValueError(f"Memory unit {record.id} has more than 1024 tags")
        if any(len(tag) > 4096 for tag in tags):
            raise ValueError(f"Memory unit {record.id} has a tag longer than 4096 characters")

        return {
            "pk": _record_key(record.namespace, record.id),
            "memory_id": record.id,
            "namespace": record.namespace,
            "tenant_key": _tenant_key(record.namespace, record.bank_id),
            "bank_id": record.bank_id,
            "fact_type": record.fact_type,
            "document_id": record.document_id or "",
            "tags": tags,
            "updated_at_ms": _datetime_to_millis(record.updated_at),
            "embedding": record.embedding,
        }

    async def upsert(self, records: list[VectorIndexRecord]) -> None:
        if not records:
            return
        entities = [self._record_to_entity(record) for record in records]
        await self._run_sync(self._upsert_sync, entities)

    def _upsert_sync(self, entities: list[dict[str, Any]]) -> None:
        client = self._require_client()
        for start in range(0, len(entities), 1000):
            client.upsert(
                collection_name=self._collection_name,
                data=entities[start : start + 1000],
            )

    async def search(
        self,
        query_vector: list[float],
        *,
        namespace: str,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list[VectorSearchHit]] | None:
        if self._dimension is None:
            raise RuntimeError("Milvus vector index is not initialized")
        if len(query_vector) != self._dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_vector)} does not match Milvus dimension {self._dimension}"
            )
        if limit <= 0:
            return {fact_type: [] for fact_type in fact_types}
        return await self._run_sync(
            self._search_sync,
            query_vector,
            namespace,
            bank_id,
            fact_types,
            limit,
            tags,
            tags_match,
            tag_groups,
            created_after,
            created_before,
        )

    def _search_sync(
        self,
        query_vector: list[float],
        namespace: str,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        tags: list[str] | None,
        tags_match: TagsMatch,
        tag_groups: list[TagGroup] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> dict[str, list[VectorSearchHit]]:
        client = self._require_client()
        results: dict[str, list[VectorSearchHit]] = {fact_type: [] for fact_type in fact_types}
        for fact_type in fact_types:
            expression = build_filter_expression(
                namespace=namespace,
                bank_id=bank_id,
                fact_type=fact_type,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
            )
            search_kwargs: dict[str, Any] = {}
            if not self._is_local:
                search_kwargs["consistency_level"] = self._consistency_level
            response = client.search(
                collection_name=self._collection_name,
                data=[query_vector],
                filter=expression,
                limit=limit,
                output_fields=["memory_id", "fact_type"],
                search_params={"metric_type": "COSINE", "params": {}},
                **search_kwargs,
            )
            hits = response[0] if response else []
            for hit in hits:
                entity = hit.get("entity") or {}
                memory_id = entity.get("memory_id") or hit.get("memory_id")
                hit_fact_type = entity.get("fact_type") or hit.get("fact_type") or fact_type
                raw_score = hit.get("distance", hit.get("score"))
                if memory_id is None or raw_score is None:
                    continue
                similarity = 1.0 - float(raw_score) if self._lite_cosine_distance_bug else float(raw_score)
                results[fact_type].append(
                    VectorSearchHit(
                        id=str(memory_id),
                        fact_type=str(hit_fact_type),
                        similarity=similarity,
                    )
                )
        return results

    async def delete_units(self, namespace: str, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        keys = [_record_key(namespace, unit_id) for unit_id in unit_ids]
        await self._delete(f"pk in {_quoted_list(keys)}")

    async def delete_document(self, namespace: str, bank_id: str, document_id: str) -> None:
        await self._delete(
            build_filter_expression(
                namespace=namespace,
                bank_id=bank_id,
                document_id=document_id,
            )
        )

    async def delete_bank(self, namespace: str, bank_id: str, fact_type: str | None = None) -> None:
        await self._delete(
            build_filter_expression(
                namespace=namespace,
                bank_id=bank_id,
                fact_type=fact_type,
            )
        )

    async def delete_namespace(self, namespace: str) -> None:
        await self._delete(f"namespace == {_quoted(namespace)}")

    async def _delete(self, expression: str) -> None:
        await self._run_sync(self._delete_sync, expression)

    def _delete_sync(self, expression: str) -> None:
        client = self._require_client()
        client.delete(collection_name=self._collection_name, filter=expression)

    async def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        client = self._client
        self._client = None
        self._dimension = None
        if client is not None:
            client.close()


__all__ = ["MilvusVectorIndex", "build_filter_expression"]
