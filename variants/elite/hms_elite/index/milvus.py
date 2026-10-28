"""Milvus vector index: milvus-lite embedded by default (or a remote server by uri), scored like the sqlite provider.
Milvus 向量索引：默认走 milvus-lite 嵌入模式（也可按 uri 连远端），打分方式与 sqlite 提供者完全一致。
"""

# Purpose (EN): An optional IndexBase provider keeping atom vectors plus the INDEX_ATTR_KEYS scalars in one Milvus
#               collection, so an IndexFilter (scope, validity at as_of, scale / obs_type, time) becomes a native
#               filtered ANN instead of an id list; SQLite stays the source of truth and feeds the scalars.
# 目的（中）：可选的 IndexBase 提供者，把原子向量与 INDEX_ATTR_KEYS 标量一起存放在一个 Milvus 集合中，使 IndexFilter
#             （scope、as_of 时刻的有效性、尺度 / obs_type、时间）成为原生过滤 ANN 而非 id 列表；SQLite 仍是真源并提供标量。

# Input (EN): MilvusIndexConfig (uri, token, db_name, collection, index_type, HNSW params, chunk sizes, ef_recall_factor),
#             dims from the factory, atom ids, unit vectors, optional attrs (looked up in the store when omitted).
# 输入（中）：MilvusIndexConfig（uri、token、db_name、collection、index_type、HNSW 参数、分块大小、ef_recall_factor）、
#             工厂传入的 dims、原子 id、单位向量、可选的 attrs（省略时从 store 查询）。

# Output (EN): (atom_id, score) hits re-scored by rank_hits over the stored float32 vectors, counts, info.
# 输出（中）：由 rank_hits 基于已存 float32 向量重算分数的 (atom_id, score) 命中、计数、信息。

# State effect (EN): Creates / drops / writes one Milvus collection; DependencyError(DEPS_001) when pymilvus is missing.
# 状态影响（中）：创建 / 删除 / 写入一个 Milvus 集合；缺少 pymilvus 时在构造处抛 DependencyError(DEPS_001)。

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Literal

from pydantic import Field, field_validator

from hms_elite.exceptions import ConfigError, DependencyError, HMSError, StoreError, ValidationError
from hms_elite.index.base import (
    INDEX_ATTR_KEYS,
    BaseIndexConfig,
    IndexBase,
    IndexFilter,
    normalize_attrs,
    quantize,
    rank_hits,
)
from hms_elite.timeutil import now_iso, to_utc_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hms_elite.memory.storage import StateStore

logger = logging.getLogger(__name__)

# Field names of the HMS atom collection; the primary key is the atom (version) id, like atom_vectors.atom_id.
# HMS 原子集合的字段名；主键是原子（版本）id，与 atom_vectors.atom_id 一致。
ID_FIELD = "atom_id"
VECTOR_FIELD = "vector"
EMBEDDER_FIELD = "embedder"
DEFAULT_COLLECTION = "hms_atoms"
# Scalar (gate) fields beside the vector: VARCHAR except the BOOL `archived`; NULL strings are stored as "" so the
# fixed-width UTC ISO instants compare lexically as instants. Scalar indexes go on the equality / range gates.
# 向量旁的标量（门控）字段：除 BOOL 的 `archived` 外都是 VARCHAR；字符串 NULL 存为 ""，使定宽 UTC ISO 时刻的字典序
# 比较即时刻比较。标量索引建在等值 / 范围门控字段上。
SCALAR_FIELDS: tuple[tuple[str, int], ...] = (
    ("user_id", 256),
    ("agent_id", 256),
    ("run_id", 256),
    ("time_anchor", 40),
    ("valid_from", 40),
    ("valid_to", 40),
    ("scale_level", 16),
    ("obs_type", 32),
    ("atom_type", 16),
)
ARCHIVED_FIELD = "archived"
SCALAR_INDEXED: tuple[str, ...] = (
    "user_id",
    "agent_id",
    "run_id",
    "scale_level",
    "obs_type",
    "atom_type",
    "valid_to",
    ARCHIVED_FIELD,
)
ATTR_DEFAULTS: dict[str, Any] = {key: (False if key == ARCHIVED_FIELD else "") for key in INDEX_ATTR_KEYS}
# A uri with one of these prefixes is a server address; anything else is a local milvus-lite `.db` path.
# 带这些前缀的 uri 是服务器地址；其余都是本地 milvus-lite 的 `.db` 路径。
REMOTE_PREFIXES = ("http://", "https://", "grpc://", "grpcs://", "tcp://", "unix:")
# Milvus servers cap the per-search `limit` (topk) and query pagination at 16384.
# Milvus 服务器把单次 search 的 `limit`（topk）与 query 分页上限定为 16384。
MAX_LIMIT = 16384


def default_milvus_uri() -> str:
    """Default milvus-lite path: $HMS_DIR/milvus.db or ~/.hms_elite/milvus.db (no directory is created here).
    默认 milvus-lite 路径：$HMS_DIR/milvus.db 或 ~/.hms_elite/milvus.db（此处不创建目录）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Path string ending with '.db'.
        以 '.db' 结尾的路径字符串。

    State effect / 状态影响：
        None (reads the HMS_DIR environment variable).
        无（读取 HMS_DIR 环境变量）。
    """
    return os.path.join(os.environ.get("HMS_DIR") or os.path.expanduser("~/.hms_elite"), "milvus.db")


def is_remote_uri(uri: str) -> bool:
    """Whether a uri addresses a Milvus server rather than a local milvus-lite file.
    判断 uri 指向 Milvus 服务器还是本地 milvus-lite 文件。

    Input / 输入：
        uri: The configured uri.
        uri：配置的 uri。

    Output / 输出：
        True for http/https/grpc/grpcs/tcp/unix addresses, False for local paths.
        http/https/grpc/grpcs/tcp/unix 地址为 True，本地路径为 False。

    State effect / 状态影响：
        None.
        无。
    """
    return uri.lower().startswith(REMOTE_PREFIXES)


def _load_pymilvus() -> Any:
    """Import pymilvus lazily (§0.4: no SDK at module import); missing → DependencyError(DEPS_001, extra='milvus').
    惰性导入 pymilvus（契约 §0.4：模块导入时不引入 SDK）；缺失 → DependencyError(DEPS_001, extra='milvus')。
    """
    try:
        import pymilvus  # noqa: PLC0415 - optional dependency imported at construction time

        return pymilvus
    except ImportError as exc:
        raise DependencyError(
            "The 'pymilvus' package is required for the milvus index provider",
            "DEPS_001",
            details={"package": "pymilvus", "extra": "milvus"},
        ) from exc


@contextlib.contextmanager
def _milvus_errors(op: str, collection: str) -> Iterator[None]:
    """Re-raise HMS errors untouched and wrap every other client failure as StoreError(STORE_001).
    HMS 错误原样抛出，其余客户端失败一律包装为 StoreError(STORE_001)。
    """
    try:
        yield
    except HMSError:
        raise
    except Exception as exc:
        raise StoreError(
            f"milvus {op} failed on collection {collection!r}: {exc}",
            "STORE_001",
            details={"provider": "milvus", "op": op, "collection": collection},
        ) from exc


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive slices of at most `size` items.
    按最多 `size` 个元素切出连续片段。
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _in_expr(ids: Iterable[str]) -> str:
    """Build the `atom_id in ["a", "b"]` filter expression (json.dumps escapes quotes and backslashes).
    构造 `atom_id in ["a", "b"]` 过滤表达式（json.dumps 负责转义引号与反斜杠）。
    """
    return f"{ID_FIELD} in [{', '.join(json.dumps(i) for i in ids)}]"


def _quoted(value: Any) -> str:
    """A Milvus string literal (json.dumps escapes quotes and backslashes the way the expression parser expects).
    Milvus 字符串字面量（json.dumps 按表达式解析器的要求转义引号与反斜杠）。
    """
    return json.dumps(str(value))


def filter_expr(spec: IndexFilter | None) -> str:
    """Render an IndexFilter as a Milvus boolean expression over the scalar fields ("" when nothing is gated).
    把 IndexFilter 渲染为作用于标量字段的 Milvus 布尔表达式（无门控时为 ""）。

    Input / 输入：
        spec: The filter (None → "").
        spec：过滤器（None → ""）。

    Output / 输出：
        `user_id == "alice" and archived == false and valid_from <= "T" and (valid_to == "" or valid_to > "T") and
        scale_level in [...] and obs_type in [...] and atom_type in [...] and time_anchor >= "A" and
        time_anchor <= "B"`, each part only when gated; unset scope keys add nothing (Scope.matches); `as_of`
        defaults to now() like the SQL gate.
        `user_id == "alice" and archived == false and valid_from <= "T" and (valid_to == "" or valid_to > "T") and
        scale_level in [...] and obs_type in [...] and atom_type in [...] and time_anchor >= "A" and
        time_anchor <= "B"`，每部分只在有门控时出现；未设置的 scope 键不产生任何谓词（Scope.matches）；`as_of` 与
        SQL 门控一样默认 now()。

    State effect / 状态影响：
        None.
        无。
    """
    if spec is None:
        return ""
    parts: list[str] = []
    if spec.scope is not None:
        for key in ("user_id", "agent_id", "run_id"):
            value = getattr(spec.scope, key)
            if value is not None:
                parts.append(f"{key} == {_quoted(value)}")
    if not spec.include_expired:
        as_of = _quoted(to_utc_iso(spec.as_of) if spec.as_of else now_iso())
        parts.append(f"{ARCHIVED_FIELD} == false")
        parts.append(f"valid_from <= {as_of}")
        parts.append(f'(valid_to == "" or valid_to > {as_of})')
    if spec.scale_in:
        parts.append(f"scale_level in [{', '.join(_quoted(v) for v in spec.scale_in)}]")
    if spec.obs_type_in:
        parts.append(f"obs_type in [{', '.join(_quoted(v) for v in spec.obs_type_in)}]")
    if spec.atom_type_in:
        parts.append(f"atom_type in [{', '.join(_quoted(v) for v in spec.atom_type_in)}]")
    if spec.time_from:
        parts.append(f"time_anchor >= {_quoted(to_utc_iso(spec.time_from))}")
    if spec.time_to:
        parts.append(f"time_anchor <= {_quoted(to_utc_iso(spec.time_to))}")
    return " and ".join(parts)


def _and(*exprs: str) -> str:
    """Join non-empty expressions with `and`, each parenthesized.
    用 `and` 连接非空表达式，每个加括号。
    """
    parts = [f"({e})" for e in exprs if e]
    return " and ".join(parts)


def _hit_id(hit: Any) -> str:
    """Primary key of a search hit (`hit.id` on every pymilvus line; the entity dict as a fallback).
    搜索命中的主键（各版本 pymilvus 都有 `hit.id`；退而取 entity 字典）。
    """
    value = getattr(hit, "id", None)
    if value is None:
        value = hit["entity"].get(ID_FIELD)
    return str(value)


class MilvusIndexConfig(BaseIndexConfig):
    """Milvus index config; `uri` is a local milvus-lite `.db` path (default $HMS_DIR/milvus.db) or a server address.
    Milvus 索引配置；`uri` 为本地 milvus-lite 的 `.db` 路径（默认 $HMS_DIR/milvus.db）或服务器地址。
    """

    uri: str = Field(default_factory=default_milvus_uri)
    token: str | None = None
    db_name: str | None = None
    collection: str = DEFAULT_COLLECTION
    metric: Literal["COSINE"] = "COSINE"
    index_type: Literal["AUTOINDEX", "HNSW", "FLAT"] = "AUTOINDEX"
    M: int = Field(default=16, ge=2)
    efConstruction: int = Field(default=200, ge=1)
    ef: int = Field(default=64, ge=1)
    consistency_level: Literal["Strong", "Session", "Bounded", "Eventually"] = "Strong"
    batch_size: int = Field(default=512, ge=1)
    candidate_chunk: int = Field(default=1024, ge=1, le=MAX_LIMIT)
    # Filtered HNSW exhausts `ef` early on selective gates (RETRIEVAL_DIAGNOSIS T6): a filtered search raises ef to
    # ef_recall_factor × limit.
    # 选择性门控下过滤式 HNSW 会提前耗尽 `ef`（RETRIEVAL_DIAGNOSIS T6）：过滤搜索把 ef 提高到 ef_recall_factor × limit。
    ef_recall_factor: int = Field(default=4, ge=1)
    # Scalar index kind on the gate fields (SCALAR_INDEXED). INVERTED costs ~0.5 s per field at collection creation on
    # milvus-lite (once per collection) and trims filtered HNSW latency (8.8 vs 12.9 ms at N=10k, 50 % selectivity);
    # 'none' skips them (tests, throwaway collections).
    # 门控字段（SCALAR_INDEXED）上的标量索引种类。INVERTED 在 milvus-lite 上每字段约 0.5 s 建集合成本（每集合一次），
    # 并降低过滤 HNSW 延迟（N=10k、50 % 选择率下 8.8 对 12.9 ms）；'none' 跳过它们（测试、一次性集合）。
    scalar_index: Literal["INVERTED", "none"] = "INVERTED"
    # A candidate list longer than candidate_chunk is searched as several id-filtered RPCs; > 1 runs them on that
    # many threads. The default stays sequential: milvus-lite serializes them and the threads only add contention
    # (3 × 1024-id chunks at N=30k: 41 ms sequential, 63 ms on 4 threads, RETRIEVAL_BENCH §3); a server may gain.
    # 长于 candidate_chunk 的候选列表拆成多个 id 过滤 RPC；> 1 时在那么多线程上运行。默认保持顺序：milvus-lite
    # 会把它们串行化，线程只增加争用（N=30k 下 3 × 1024 个 id 的块：顺序 41 ms，4 线程 63 ms，
    # RETRIEVAL_BENCH §3）；服务器可能受益。
    search_workers: int = Field(default=1, ge=1)

    @field_validator("uri", mode="before")
    @classmethod
    def _fspath(cls, value: Any) -> Any:
        """Accept os.PathLike (tmp_path / 'm.db') for the uri.
        uri 接受 os.PathLike（如 tmp_path / 'm.db'）。
        """
        return os.fspath(value) if isinstance(value, os.PathLike) else value

    @field_validator("uri")
    @classmethod
    def _local_needs_db_suffix(cls, value: str) -> str:
        """Local paths must end with '.db' (milvus-lite rule); a clear ConfigError beats the client's message.
        本地路径必须以 '.db' 结尾（milvus-lite 规则）；清晰的 ConfigError 好过客户端的报错。
        """
        value = value.strip()
        if not value:
            raise ConfigError("milvus uri must not be empty", "CONFIG_001", details={"provider": "milvus"})
        if not is_remote_uri(value) and not value.endswith(".db"):
            raise ConfigError(
                f"local milvus-lite uri must end with '.db', got {value!r}",
                "CONFIG_001",
                details={"provider": "milvus", "uri": value},
                suggestion="use a path such as '$HMS_DIR/milvus.db' or a server address such as 'http://host:19530'",
            )
        return value


class MilvusIndex(IndexBase):
    """Vector index over one Milvus collection; hits are re-scored with rank_hits so scores equal the sqlite provider's.
    面向单个 Milvus 集合的向量索引；命中经 rank_hits 重新打分，使分数与 sqlite 提供者相等。
    """

    config_class: ClassVar[type[BaseIndexConfig]] = MilvusIndexConfig
    provider_name: ClassVar[str] = "milvus"

    def __init__(
        self, config: MilvusIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64
    ) -> None:
        """Open the client (DependencyError without pymilvus) and create or validate the collection.
        打开客户端（缺少 pymilvus 时抛 DependencyError），并创建或校验集合。
        """
        self._client_handle: Any = None
        super().__init__(config, store=store, dims=dims)
        self._pymilvus = _load_pymilvus()
        cfg: MilvusIndexConfig = self.config  # type: ignore[assignment]
        if self.dims <= 0:
            raise ConfigError(
                f"milvus index needs dims >= 1, got {self.dims}", "CONFIG_001", details={"provider": "milvus"}
            )
        self._remote = is_remote_uri(cfg.uri)
        if not self._remote:
            # milvus-lite needs the parent directory; the `.db` entry itself is created by the client.
            # milvus-lite 需要父目录存在；`.db` 条目本身由客户端创建。
            os.makedirs(os.path.dirname(os.path.abspath(cfg.uri)), exist_ok=True)
        kwargs: dict[str, Any] = {"uri": cfg.uri}
        if cfg.token:
            kwargs["token"] = cfg.token
        if cfg.db_name:
            kwargs["db_name"] = cfg.db_name
        with _milvus_errors("connect", cfg.collection):
            self._client_handle = self._pymilvus.MilvusClient(**kwargs)
        self._ensure_collection()

    # ------------------------------------------------------------------ client plumbing
    # ------------------------------------------------------------------ 客户端管道
    @property
    def _client(self) -> Any:
        """The open MilvusClient; StoreError once close() ran.
        已打开的 MilvusClient；close() 之后抛 StoreError。
        """
        if self._client_handle is None:
            raise StoreError(
                "MilvusIndex is closed",
                "STORE_001",
                details={"provider": "milvus", "collection": self.config.collection},  # type: ignore[attr-defined]
            )
        return self._client_handle

    @property
    def _cfg(self) -> MilvusIndexConfig:
        """The typed config.
        带类型的配置。
        """
        return self.config  # type: ignore[return-value]

    def _index_params(self) -> dict:
        """Build-time index params: M / efConstruction for HNSW, nothing for AUTOINDEX / FLAT.
        建索引参数：HNSW 用 M / efConstruction，AUTOINDEX / FLAT 无参数。
        """
        cfg = self._cfg
        if cfg.index_type == "HNSW":
            return {"M": cfg.M, "efConstruction": cfg.efConstruction}
        return {}

    def _search_params(self, limit: int, *, filtered: bool = False, ef_floor: int = 0) -> dict:
        """Search params; Milvus requires ef >= limit for HNSW, a filtered search raises ef further (T6 recall gap)
        and an id-list chunk raises it to the chunk size (a 1024-id filter over 30k rows otherwise returns a handful).
        搜索参数；Milvus 要求 HNSW 的 ef >= limit，过滤搜索进一步提高 ef（T6 召回缺口），id 列表分块则把它提到分块
        大小（否则 30k 行上 1024 个 id 的过滤只返回寥寥几行）。
        """
        cfg = self._cfg
        params: dict[str, Any] = {"metric_type": cfg.metric}
        if cfg.index_type == "HNSW":
            ef = max(cfg.ef, limit, cfg.ef_recall_factor * limit if filtered else 0, int(ef_floor))
            params["params"] = {"ef": min(ef, 32768)}
        return params

    def _ensure_collection(self) -> None:
        """Create the (atom_id, vector, embedder, scalars) collection, or validate an existing one against dims.
        创建 (atom_id, vector, embedder, 标量) 集合，或按 dims 校验已有集合。
        """
        cfg = self._cfg
        client = self._client
        with _milvus_errors("describe", cfg.collection):
            exists = bool(client.has_collection(cfg.collection))
        if exists:
            with _milvus_errors("describe", cfg.collection):
                description = client.describe_collection(cfg.collection)
            self._check_schema(description)
            # A collection reopened by a new process (or released) is not loaded: search() would raise code 101
            # "call load() before search"; load_collection is idempotent on a loaded one.
            # 由新进程重新打开（或已 release）的集合未加载：search() 会抛 code 101 "call load() before search"；
            # 对已加载的集合 load_collection 是幂等的。
            with _milvus_errors("load_collection", cfg.collection):
                client.load_collection(cfg.collection)
            return
        data_type = self._pymilvus.DataType
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(ID_FIELD, data_type.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(VECTOR_FIELD, data_type.FLOAT_VECTOR, dim=self.dims)
        schema.add_field(EMBEDDER_FIELD, data_type.VARCHAR, max_length=128)
        for name, max_length in SCALAR_FIELDS:
            schema.add_field(name, data_type.VARCHAR, max_length=max_length)
        schema.add_field(ARCHIVED_FIELD, data_type.BOOL)
        # Scalar indexes: the configured kind first, then none (some servers reject an index kind on some types).
        # 标量索引：先试配置的种类，再退到不建（部分服务器拒绝某些类型上的索引种类）。
        last_error: Exception | None = None
        kinds = (cfg.scalar_index, None) if cfg.scalar_index != "none" else (None,)
        for scalar_kind in kinds:
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=VECTOR_FIELD, index_type=cfg.index_type, metric_type=cfg.metric, params=self._index_params()
            )
            if scalar_kind is not None:
                for name in SCALAR_INDEXED:
                    index_params.add_index(field_name=name, index_type=scalar_kind)
            try:
                with _milvus_errors("create_collection", cfg.collection):
                    client.create_collection(
                        cfg.collection, schema=schema, index_params=index_params, consistency_level=cfg.consistency_level
                    )
                return
            except StoreError as exc:
                last_error = exc
                with contextlib.suppress(Exception):
                    if client.has_collection(cfg.collection):
                        client.drop_collection(cfg.collection)
                if scalar_kind is not None:
                    logger.warning("milvus rejected %s scalar indexes on %r: %s", scalar_kind, cfg.collection, exc)
        raise last_error  # type: ignore[misc]

    def _check_schema(self, description: dict) -> None:
        """An existing collection must carry the HMS fields (scalars included) and the factory's dims, else CONFIG_001.
        已有集合必须带有 HMS 字段（含标量）且维数与工厂一致，否则抛 CONFIG_001。
        """
        cfg = self._cfg
        fields = {str(f.get("name")): f for f in description.get("fields", [])}
        vector = fields.get(VECTOR_FIELD)
        required = {ID_FIELD, EMBEDDER_FIELD, ARCHIVED_FIELD, *(name for name, _length in SCALAR_FIELDS)}
        missing = sorted(required - set(fields))
        if vector is None or missing:
            raise ConfigError(
                f"milvus collection {cfg.collection!r} at {cfg.uri!r} is not an HMS atom collection with scalar "
                f"gates (missing: {missing}; fields: {sorted(fields)})",
                "CONFIG_001",
                details={
                    "provider": "milvus",
                    "collection": cfg.collection,
                    "uri": cfg.uri,
                    "fields": sorted(fields),
                    "missing": missing,
                },
                suggestion="point vector_store.config.collection at a new collection name, or drop the collection "
                "and re-embed with consolidate()",
            )
        existing = int((vector.get("params") or {}).get("dim") or 0)
        if existing != self.dims:
            raise ConfigError(
                f"milvus collection {cfg.collection!r} at {cfg.uri!r} stores {existing}-dim vectors but the embedder "
                f"produces {self.dims}-dim vectors",
                "CONFIG_001",
                details={
                    "provider": "milvus",
                    "collection": cfg.collection,
                    "uri": cfg.uri,
                    "collection_dims": existing,
                    "embedder_dims": self.dims,
                },
                suggestion="keep the embedder that built the collection, choose a new vector_store.config.collection "
                "name, or drop the collection and re-embed with consolidate()",
            )

    def _rows(
        self,
        items: Iterable[tuple[str, list[float]]],
        embedder: str,
        attrs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict]:
        """Turn (atom_id, vector) pairs into float32-quantized Milvus rows with their scalars, checking the dims.
        把 (atom_id, vector) 对转为带标量、float32 取整的 Milvus 行，并检查维数。

        Scalars are layered: defaults, then the store's row (StateStore.atom_attrs, one batched SELECT when a store is
        bound), then the caller's `attrs[atom_id]` on top.
        标量分层叠加：默认值，再是 store 的行（StateStore.atom_attrs，绑定 store 时一次批量 SELECT），最后是调用方的
        `attrs[atom_id]`。
        """
        pairs = [(str(atom_id), vector) for atom_id, vector in items]
        given = {str(k): normalize_attrs(v) for k, v in (attrs or {}).items()}
        lookup: dict[str, dict[str, Any]] = {}
        if pairs and self.store is not None:
            lookup = self.store.atom_attrs([atom_id for atom_id, _vector in pairs])
        rows: list[dict] = []
        for atom_id, vector in pairs:
            values = quantize(vector)
            if len(values) != self.dims:
                raise ValidationError(
                    f"vector for {atom_id!r} has {len(values)} components; collection {self._cfg.collection!r} "
                    f"stores {self.dims}",
                    "VALIDATION_000",
                    details={"atom_id": atom_id, "dims": len(values), "expected": self.dims},
                )
            scalars = {**ATTR_DEFAULTS, **normalize_attrs(lookup.get(atom_id)), **given.get(atom_id, {})}
            rows.append({ID_FIELD: atom_id, VECTOR_FIELD: values, EMBEDDER_FIELD: str(embedder), **scalars})
        return rows

    def _ann(
        self, query: list[float], *, expr: str, limit: int, filtered: bool = False, ef_floor: int = 0
    ) -> dict[str, list[float]]:
        """One Milvus search returning {atom_id: stored float32 vector} for at most `limit` nearest rows.
        一次 Milvus 搜索，返回最多 `limit` 个最近行的 {atom_id: 已存 float32 向量}。
        """
        cfg = self._cfg
        limit = min(limit, MAX_LIMIT)
        with _milvus_errors("search", cfg.collection):
            result = self._client.search(
                cfg.collection,
                data=[query],
                filter=expr,
                limit=limit,
                output_fields=[VECTOR_FIELD],
                search_params=self._search_params(limit, filtered=filtered, ef_floor=ef_floor),
            )
        hits: dict[str, list[float]] = {}
        for hit in result[0] if result else []:
            hits[_hit_id(hit)] = quantize(hit["entity"][VECTOR_FIELD])
        return hits

    def _vectors_of(self, ids: list[str], expr: str = "") -> dict[str, list[float]]:
        """{atom_id: stored float32 vector} of the given ids that satisfy `expr` (a primary-key `query`, no ANN).
        给定 id 中满足 `expr` 者的 {atom_id: 已存 float32 向量}（主键 `query`，不走 ANN）。
        """
        cfg = self._cfg
        with _milvus_errors("query", cfg.collection):
            rows = self._client.query(
                cfg.collection, filter=_and(_in_expr(ids), expr), output_fields=[VECTOR_FIELD], limit=len(ids)
            )
        return {str(row[ID_FIELD]): quantize(row[VECTOR_FIELD]) for row in rows}

    def _existing_ids(self, wanted: list[str] | None, expr: str = "") -> set[str]:
        """Ids that have a vector and satisfy `expr`: within `wanted` (chunked id filters) or the whole collection.
        拥有向量且满足 `expr` 的 id：在 `wanted` 内（分块 id 过滤）或整个集合。
        """
        cfg = self._cfg
        client = self._client
        found: set[str] = set()
        if wanted is not None:
            for chunk in _chunks(wanted, cfg.candidate_chunk):
                with _milvus_errors("query", cfg.collection):
                    rows = client.query(
                        cfg.collection, filter=_and(_in_expr(chunk), expr), output_fields=[ID_FIELD], limit=len(chunk)
                    )
                found.update(str(row[ID_FIELD]) for row in rows)
            return found
        iterator_factory = getattr(client, "query_iterator", None)
        if iterator_factory is not None:
            with _milvus_errors("query", cfg.collection):
                iterator = iterator_factory(
                    cfg.collection, batch_size=cfg.candidate_chunk, filter=expr, output_fields=[ID_FIELD]
                )
                try:
                    while True:
                        batch = iterator.next()
                        if not batch:
                            break
                        found.update(str(row[ID_FIELD]) for row in batch)
                finally:
                    iterator.close()
            return found
        # Old clients without query_iterator: page with offset, which Milvus caps at MAX_LIMIT rows in total.
        # 没有 query_iterator 的旧客户端：按 offset 分页，Milvus 把总量限制在 MAX_LIMIT 行。
        offset = 0
        while offset + cfg.candidate_chunk <= MAX_LIMIT:
            with _milvus_errors("query", cfg.collection):
                rows = client.query(
                    cfg.collection, filter=expr, output_fields=[ID_FIELD], limit=cfg.candidate_chunk, offset=offset
                )
            found.update(str(row[ID_FIELD]) for row in rows)
            if len(rows) < cfg.candidate_chunk:
                break
            offset += cfg.candidate_chunk
        return found

    # ------------------------------------------------------------------ IndexBase contract
    # ------------------------------------------------------------------ IndexBase 契约
    def upsert(
        self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping[str, Any] | None = None
    ) -> None:
        """Insert or replace one atom vector with its scalars.
        插入或替换一个原子向量及其标量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            vector: Unit vector (stored float32-rounded, like atom_vectors).
            vector：单位向量（与 atom_vectors 一样按 float32 取整存储）。

            embedder: Embedder identity recorded beside the vector.
            embedder：随向量记录的嵌入器身份。

            attrs: INDEX_ATTR_KEYS scalars; None → read from the store (atom_attrs), defaults when unknown there.
            attrs：INDEX_ATTR_KEYS 标量；None → 从 store 读取（atom_attrs），store 中未知时用默认值。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            One Milvus upsert.
            一次 Milvus upsert。
        """
        self.upsert_batch([(atom_id, vector)], embedder=embedder, attrs=None if attrs is None else {str(atom_id): attrs})

    def upsert_batch(
        self,
        items: list[tuple[str, list[float]]],
        *,
        embedder: str,
        attrs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Upsert several vectors (with scalars) in batches of config.batch_size.
        按 config.batch_size 分批写入多个向量（含标量）。

        Input / 输入：
            items: `[(atom_id, vector), ...]`.
            items：`[(atom_id, vector), ...]`。

            embedder: Embedder identity.
            embedder：嵌入器身份。

            attrs: `{atom_id: attrs}`; ids absent from it are looked up in the store in one batch.
            attrs：`{atom_id: attrs}`；不在其中的 id 一次性从 store 批量查询。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            One Milvus upsert per batch.
            每批一次 Milvus upsert。
        """
        cfg = self._cfg
        rows = self._rows(items, embedder, attrs)
        for start in range(0, len(rows), cfg.batch_size):
            with _milvus_errors("upsert", cfg.collection):
                self._client.upsert(cfg.collection, rows[start : start + cfg.batch_size])

    def update_attrs(self, atom_id: str, attrs: Mapping[str, Any]) -> None:
        """Overwrite some scalars of one row (read the row, merge, upsert: Milvus has no in-place scalar update).
        覆盖一行的部分标量（读回该行、合并、upsert：Milvus 没有原地标量更新）。

        Input / 输入：
            atom_id: Atom id (a row without a vector is ignored).
            atom_id：原子 id（没有向量的行被忽略）。

            attrs: The scalars to overwrite (normalized like atom_index_attrs).
            attrs：要覆盖的标量（按 atom_index_attrs 的方式规范化）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            One Milvus get + one upsert.
            一次 Milvus get + 一次 upsert。
        """
        changes = normalize_attrs(attrs)
        if not changes:
            return
        cfg = self._cfg
        fields = [VECTOR_FIELD, EMBEDDER_FIELD, ARCHIVED_FIELD, *(name for name, _length in SCALAR_FIELDS)]
        with _milvus_errors("get", cfg.collection):
            rows = self._client.get(cfg.collection, ids=[str(atom_id)], output_fields=fields)
        if not rows:
            return
        current = rows[0]
        row = {
            ID_FIELD: str(atom_id),
            VECTOR_FIELD: quantize(current[VECTOR_FIELD]),
            EMBEDDER_FIELD: str(current.get(EMBEDDER_FIELD, "")),
            **{key: current.get(key, ATTR_DEFAULTS[key]) for key in INDEX_ATTR_KEYS},
            **changes,
        }
        with _milvus_errors("upsert", cfg.collection):
            self._client.upsert(cfg.collection, [row])

    def get(self, atom_id: str) -> list[float] | None:
        """Read one atom vector.
        读取一个原子向量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            The stored float32 vector or None.
            已存 float32 向量或 None。

        State effect / 状态影响：
            None.
            无。
        """
        cfg = self._cfg
        with _milvus_errors("get", cfg.collection):
            rows = self._client.get(cfg.collection, ids=[str(atom_id)], output_fields=[VECTOR_FIELD])
        if not rows:
            return None
        return quantize(rows[0][VECTOR_FIELD])

    def delete(self, atom_id: str) -> None:
        """Remove one atom vector (no error when absent).
        删除一个原子向量（不存在时不报错）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            One Milvus delete.
            一次 Milvus delete。
        """
        cfg = self._cfg
        with _milvus_errors("delete", cfg.collection):
            self._client.delete(cfg.collection, ids=[str(atom_id)])

    @property
    def supports_filter(self) -> bool:
        """True: an IndexFilter becomes a native scalar-filtered ANN (filter_expr).
        为真：IndexFilter 成为原生标量过滤 ANN（filter_expr）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            True.
            True。

        State effect / 状态影响：
            None.
            无。
        """
        return True

    def search(
        self,
        vector: list[float],
        *,
        candidate_ids: list[str] | None = None,
        filter: IndexFilter | None = None,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Nearest atoms by cosine (dot of unit vectors, clamped at 0), sorted (score DESC, atom_id ASC).
        按余弦（单位向量点积，下限 0）取最近原子，按（score 降序，atom_id 升序）排序。

        Input / 输入：
            vector: Query vector.
            vector：查询向量。

            candidate_ids: Restrict to these ids (None → no id restriction); searched in candidate_chunk-sized id filters, each chunk fetched whole when top_k covers it (exact, like the sqlite provider).
            candidate_ids：限定这些 id（None → 不限定 id），按 config.candidate_chunk 切块做 id 过滤搜索；top_k 覆盖整块时整块取回（精确，与 sqlite 提供者一致）。

            filter: Pushed down as filter_expr(filter): alone it is ONE filtered ANN with limit=top_k (approximate recall under HNSW / AUTOINDEX, exact under FLAT); with candidate_ids it is ANDed into every id chunk (the exact intersection).
            filter：以 filter_expr(filter) 下推：单独给出时是**一次** limit=top_k 的过滤 ANN（HNSW / AUTOINDEX 下召回近似，FLAT 下精确）；与 candidate_ids 同时给出时 AND 进每个 id 块（精确交集）。

            top_k: Maximum hits.
            top_k：最大命中数。

        Output / 输出：
            `[(atom_id, score), ...]` with scores recomputed by rank_hits over the stored float32 vectors, so the
            same vectors score bit-for-bit like the sqlite and memory providers.
            `[(atom_id, score), ...]`，分数由 rank_hits 基于已存 float32 向量重算，因此同一向量的分数与 sqlite、
            memory 提供者逐位一致。

        State effect / 状态影响：
            None (reads Milvus).
            无（读取 Milvus）。
        """
        if top_k <= 0 or (candidate_ids is not None and not candidate_ids):
            return []
        query = [float(v) for v in vector]
        cfg = self._cfg
        gate = filter_expr(filter)
        hits: dict[str, list[float]] = {}
        truncated = False
        wanted: list[str] | None = None
        if candidate_ids is None:
            limit = min(top_k, MAX_LIMIT)
            hits = self._ann(query, expr=gate, limit=limit, filtered=bool(gate))
            truncated = len(hits) >= limit
        else:
            wanted = list(dict.fromkeys(str(i) for i in candidate_ids))
            chunks = list(_chunks(wanted, cfg.candidate_chunk))

            def _chunk_search(chunk: list[str]) -> tuple[int, dict[str, list[float]]]:
                limit = min(top_k, len(chunk))
                if limit >= len(chunk):
                    # Every id of the chunk is wanted (the in-pool rescoring): a primary-key read of the vectors,
                    # no HNSW traversal — 2.4 ms against 6.6 ms for the id-filtered search at N=30k, exact.
                    # 块内每个 id 都要（池内重算）：按主键读取向量，不走 HNSW——N=30k 下 2.4 ms 对 id 过滤搜索的
                    # 6.6 ms，且精确。
                    return limit, self._vectors_of(chunk, gate)
                return limit, self._ann(
                    query, expr=_and(_in_expr(chunk), gate), limit=limit, filtered=bool(gate), ef_floor=len(chunk)
                )

            if len(chunks) > 1 and cfg.search_workers > 1:
                # Several id chunks: the RPCs are independent, so they run concurrently (RETRIEVAL_DIAGNOSIS T6: each
                # filtered chunk costs a fixed ~30 ms on milvus-lite, sequentially additive otherwise).
                # 多个 id 块：各 RPC 相互独立，因此并发执行（RETRIEVAL_DIAGNOSIS T6：milvus-lite 上每个过滤块固定约
                # 30 ms，否则按顺序累加）。
                with ThreadPoolExecutor(max_workers=min(cfg.search_workers, len(chunks))) as pool:
                    results = list(pool.map(_chunk_search, chunks))
            else:
                results = [_chunk_search(chunk) for chunk in chunks]
            for chunk, (limit, chunk_hits) in zip(chunks, results):
                truncated = truncated or (limit < len(chunk) and len(chunk_hits) >= limit)
                hits.update(chunk_hits)
        scored = rank_hits(query, hits, top_k=top_k, use_numpy=self.uses_numpy)
        # Milvus picks the `limit` nearest by raw cosine, so once negative-cosine rows enter a truncated result the
        # zero-score tail must be rebuilt from every candidate in atom_id order to match the exhaustive providers.
        # Milvus 按原始余弦取最近的 `limit` 行，一旦负余弦行进入被截断的结果，零分尾部必须按 atom_id 顺序从全部
        # 候选重建，才能与穷举式提供者一致。
        if truncated and len(scored) == top_k and scored[-1][1] == 0.0:
            positives = [(atom_id, score) for atom_id, score in scored if score > 0.0]
            taken = {atom_id for atom_id, _score in positives}
            zeros = sorted(self._existing_ids(wanted, gate) - taken)
            scored = (positives + [(atom_id, 0.0) for atom_id in zeros])[:top_k]
        return scored

    def count(self, filter: IndexFilter | None = None) -> int:
        """Number of stored vectors, optionally under a pushed-down gate (`count(*)` query with filter_expr).
        已存储向量数，可带下推门控（带 filter_expr 的 `count(*)` 查询）。

        Input / 输入：
            filter: An IndexFilter rendered by filter_expr (None → every row).
            filter：由 filter_expr 渲染的 IndexFilter（None → 全部行）。

        Output / 输出：
            Count.
            数量。

        State effect / 状态影响：
            None.
            无。
        """
        cfg = self._cfg
        with _milvus_errors("count", cfg.collection):
            rows = self._client.query(cfg.collection, filter=filter_expr(filter), output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) if rows else 0

    def reset(self) -> None:
        """Drop the collection and re-create it empty with the same schema and index.
        删除集合并以相同模式与索引重建为空集合。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Drops and re-creates the Milvus collection.
            删除并重建 Milvus 集合。
        """
        cfg = self._cfg
        with _milvus_errors("drop_collection", cfg.collection):
            if self._client.has_collection(cfg.collection):
                self._client.drop_collection(cfg.collection)
        self._ensure_collection()

    def info(self) -> dict:
        """Provider summary.
        提供者摘要。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `{'provider', 'dims', 'count', 'numpy', 'uri', 'collection', 'index_type'}`.
            `{'provider', 'dims', 'count', 'numpy', 'uri', 'collection', 'index_type'}`。

        State effect / 状态影响：
            None.
            无。
        """
        cfg = self._cfg
        return {**super().info(), "uri": cfg.uri, "collection": cfg.collection, "index_type": cfg.index_type}

    def close(self) -> None:
        """Close the Milvus client (idempotent); later calls raise StoreError.
        关闭 Milvus 客户端（幂等）；之后的调用抛 StoreError。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Releases the client handle.
            释放客户端句柄。
        """
        handle = getattr(self, "_client_handle", None)
        self._client_handle = None
        if handle is not None:
            with _milvus_errors("close", self._cfg.collection):
                handle.close()

    def __del__(self) -> None:
        """Best-effort close at garbage collection.
        垃圾回收时尽力关闭。
        """
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown ordering
            pass


__all__ = ["MilvusIndex", "MilvusIndexConfig", "default_milvus_uri", "filter_expr", "is_remote_uri"]
