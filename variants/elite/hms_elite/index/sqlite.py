"""SQLite vector index: atom_vectors BLOBs in the same database, mirrored into a generation-keyed in-process matrix.
SQLite 向量索引：atom_vectors BLOB 与主库同库，并镜像为以写入代数为键的进程内矩阵。
"""

# Purpose (EN): The default index provider; vectors live beside the atoms so rollback and reset stay atomic. search()
#               ranks through a process-local cache of the vectors and their gate scalars (INDEX_ATTR_KEYS), extended
#               incrementally by atom_vectors rowid and rebuilt after a DELETE / ROLLBACK / foreign commit, so an
#               IndexFilter is evaluated as a numpy mask (native pushdown, RETRIEVAL_DIAGNOSIS T8 variant E) and a
#               candidate list as a row gather — one matmul selects, `dot` rescores bit-for-bit, nothing hydrates.
# 目的（中）：默认索引提供者；向量与原子同库，使回滚与重置保持原子性。search() 通过一个进程内缓存排序：缓存镜像
#             向量及其门控标量（INDEX_ATTR_KEYS），按 atom_vectors rowid 增量扩展，在 DELETE / ROLLBACK / 其它连接提交后
#             重建；因此 IndexFilter 以 numpy 掩码求值（原生下推，RETRIEVAL_DIAGNOSIS T8 变体 E），候选列表以行聚集求值
#             ——一次矩阵乘挑选、`dot` 逐位重算，从不水合。

# Input (EN): SQLiteIndexConfig (use_numpy), a StateStore, atom ids, vectors, IndexFilter / candidate ids, attrs.
# 输入（中）：SQLiteIndexConfig（use_numpy）、StateStore、原子 id、向量、IndexFilter / 候选 id、标量。

# Output (EN): Ranked hits; counts (filtered or not); info; the SQL gate fallback when numpy is unavailable.
# 输出（中）：排序命中；计数（带或不带过滤）；信息；numpy 不可用时的 SQL 门控退路。

# State effect (EN): Writes atom_vectors through StateStore.upsert_vector / delete_vector / clear_vectors; keeps the
#                    cache in sync from store.data_generation / rebuild_generation / data_version() and the store's
#                    attr_listeners (archive / restore / set_valid_to). Lock order: store.lock, then the index lock.
# 状态影响（中）：通过 StateStore.upsert_vector / delete_vector / clear_vectors 写入 atom_vectors；依据
#               store.data_generation / rebuild_generation / data_version() 与 store 的 attr_listeners
#               （archive / restore / set_valid_to）保持缓存同步。锁序：先 store.lock，再索引锁。

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from hms_elite.exceptions import ConfigError
from hms_elite.index.base import (
    BaseIndexConfig,
    IndexBase,
    IndexFilter,
    normalize_attrs,
    numpy_module,
    rank_hits,
    select_top_k,
)
from hms_elite.timeutil import to_utc_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hms_elite.memory.storage import StateStore

# Scalars kept as integer codes (equality / whitelist gates) and as fixed-width bytes (lexical instant comparisons).
# 以整数码保存的标量（等值 / 白名单门控）与以定宽字节保存的标量（时刻的字典序比较）。
CODE_ATTRS: tuple[str, ...] = ("user_id", "agent_id", "run_id", "scale_level", "obs_type", "atom_type")
INSTANT_ATTRS: tuple[str, ...] = ("time_anchor", "valid_from", "valid_to")
_INSTANT_WIDTH = 40
_MIN_CAPACITY = 256


class SQLiteIndexConfig(BaseIndexConfig):
    """SQLite index config.
    SQLite 索引配置。
    """

    use_numpy: Literal["auto", "never"] = "auto"


class _VectorCache:
    """Process-local mirror of atom_vectors ⋈ memory_atoms gate scalars for one dimensionality.
    某一维数下 atom_vectors ⋈ memory_atoms 门控标量的进程内镜像。

    Rows are appended in atom_vectors rowid order (a replaced vector reappears with a new rowid and overwrites its
    row in place); the arrays grow by doubling so an append is amortized O(1).
    行按 atom_vectors 的 rowid 顺序追加（被替换的向量以新 rowid 再次出现并原地覆盖其行）；数组按倍增扩容，
    因此追加的摊还成本为 O(1)。
    """

    def __init__(self, np: Any, dims: int) -> None:
        """Start empty for `dims`-dimensional vectors.
        以空状态开始，面向 `dims` 维向量。
        """
        self.np = np
        self.dims = int(dims)
        self.n = 0
        self.capacity = 0
        self.ids: list[str] = []
        self.positions: dict[str, int] = {}
        self.matrix: Any = np.zeros((0, self.dims), dtype="float64")
        self.codes: dict[str, Any] = {name: np.zeros(0, dtype="int32") for name in CODE_ATTRS}
        self.code_of: dict[str, dict[str, int]] = {name: {} for name in CODE_ATTRS}
        self.instants: dict[str, Any] = {name: np.zeros(0, dtype=f"S{_INSTANT_WIDTH}") for name in INSTANT_ATTRS}
        self.archived: Any = np.zeros(0, dtype=bool)
        self.last_rowid = 0
        self.key: tuple[int, int, int] | None = None
        # False once a row of another dimensionality was seen: the provider then uses the SQL path.
        # 一旦见到其它维数的行即为 False：提供者此后走 SQL 路径。
        self.usable = True

    def clear(self) -> None:
        """Forget every row (a rebuild follows).
        忘记全部行（随后重建）。
        """
        self.__init__(self.np, self.dims)

    def _grow(self, extra: int) -> None:
        """Ensure capacity for `extra` more rows (doubling).
        确保还能容纳 `extra` 行（倍增）。
        """
        need = self.n + extra
        if need <= self.capacity:
            return
        np = self.np
        capacity = max(_MIN_CAPACITY, self.capacity)
        while capacity < need:
            capacity *= 2
        matrix = np.zeros((capacity, self.dims), dtype="float64")
        matrix[: self.n] = self.matrix[: self.n]
        self.matrix = matrix
        for name in CODE_ATTRS:
            grown = np.zeros(capacity, dtype="int32")
            grown[: self.n] = self.codes[name][: self.n]
            self.codes[name] = grown
        for name in INSTANT_ATTRS:
            grown = np.zeros(capacity, dtype=f"S{_INSTANT_WIDTH}")
            grown[: self.n] = self.instants[name][: self.n]
            self.instants[name] = grown
        archived = np.zeros(capacity, dtype=bool)
        archived[: self.n] = self.archived[: self.n]
        self.archived = archived
        self.capacity = capacity

    def _code(self, name: str, value: Any) -> int:
        """Integer code of a scalar value (new values get the next code).
        标量值的整数码（新值获得下一个码）。
        """
        table = self.code_of[name]
        key = "" if value is None else str(value)
        code = table.get(key)
        if code is None:
            code = len(table)
            table[key] = code
        return code

    def set_attrs(self, pos: int, attrs: Mapping[str, Any]) -> None:
        """Write the given scalars of one row (unknown keys ignored).
        写入一行的给定标量（未知键忽略）。
        """
        for name, value in attrs.items():
            if name in CODE_ATTRS:
                self.codes[name][pos] = self._code(name, value)
            elif name in INSTANT_ATTRS:
                self.instants[name][pos] = ("" if value is None else str(value)).encode("ascii", "replace")
            elif name == "archived":
                self.archived[pos] = bool(value)

    def ingest(self, rows: list[tuple[int, str, int, bytes, dict[str, Any]]]) -> None:
        """Append or overwrite the rows of store.load_vector_rows (rowid order).
        追加或覆盖 store.load_vector_rows 的行（rowid 顺序）。
        """
        np = self.np
        fresh = sum(1 for _rowid, atom_id, _dims, _blob, _attrs in rows if atom_id not in self.positions)
        self._grow(fresh)
        for rowid, atom_id, dims, blob, attrs in rows:
            self.last_rowid = max(self.last_rowid, int(rowid))
            if dims != self.dims:
                self.usable = False
                continue
            pos = self.positions.get(atom_id)
            if pos is None:
                pos = self.n
                self.n += 1
                self.ids.append(atom_id)
                self.positions[atom_id] = pos
            self.matrix[pos] = np.frombuffer(blob, dtype="<f4").astype("float64")
            self.set_attrs(pos, attrs)

    def mask(self, spec: IndexFilter, *, as_of_default: str) -> Any:
        """The boolean row mask of an IndexFilter (exactly the SQL gate of StateStore.axis_select).
        IndexFilter 的布尔行掩码（与 StateStore.axis_select 的 SQL 门控完全相同）。
        """
        np = self.np
        n = self.n
        mask = np.ones(n, dtype=bool)
        if spec.scope is not None:
            for name in ("user_id", "agent_id", "run_id"):
                value = getattr(spec.scope, name)
                if value is None:
                    continue
                code = self.code_of[name].get(str(value))
                if code is None:
                    return np.zeros(n, dtype=bool)
                mask &= self.codes[name][:n] == code
        if not spec.include_expired:
            as_of = to_utc_iso(spec.as_of) if spec.as_of else to_utc_iso(as_of_default)
            instant = as_of.encode("ascii", "replace")
            mask &= ~self.archived[:n]
            mask &= self.instants["valid_from"][:n] <= instant
            valid_to = self.instants["valid_to"][:n]
            mask &= (valid_to == b"") | (valid_to > instant)
        for name, values in (
            ("scale_level", spec.scale_in),
            ("obs_type", spec.obs_type_in),
            ("atom_type", spec.atom_type_in),
        ):
            if values:
                known = [self.code_of[name][v] for v in dict.fromkeys(str(v) for v in values) if v in self.code_of[name]]
                if not known:
                    return np.zeros(n, dtype=bool)
                mask &= np.isin(self.codes[name][:n], np.asarray(known, dtype="int32"))
        if spec.time_from or spec.time_to:
            anchors = self.instants["time_anchor"][:n]
            # A NULL time_anchor ("" here) never satisfies a bound, like SQL's BETWEEN / comparison with NULL.
            # NULL 的 time_anchor（此处为 ""）与 SQL 的 BETWEEN / NULL 比较一样，从不满足任何边界。
            mask &= anchors != b""
            if spec.time_from:
                mask &= anchors >= to_utc_iso(spec.time_from).encode("ascii", "replace")
            if spec.time_to:
                mask &= anchors <= to_utc_iso(spec.time_to).encode("ascii", "replace")
        return mask

    def rows_of(self, candidate_ids: list[str]) -> Any:
        """Row positions of the candidate ids that have a vector (first-seen order, duplicates dropped).
        拥有向量的候选 id 的行位置（首次出现顺序，去重）。
        """
        positions = self.positions
        return self.np.asarray(
            [positions[i] for i in dict.fromkeys(str(i) for i in candidate_ids) if i in positions], dtype="int64"
        )


class SQLiteIndex(IndexBase):
    """Vector index backed by the atom_vectors table of the shared StateStore, ranked through _VectorCache.
    以共享 StateStore 的 atom_vectors 表为后端、经 _VectorCache 排序的向量索引。
    """

    config_class: ClassVar[type[BaseIndexConfig]] = SQLiteIndexConfig
    provider_name: ClassVar[str] = "sqlite"

    def __init__(
        self, config: SQLiteIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64
    ) -> None:
        """Bind to a StateStore (required) and subscribe to its committed attribute changes.
        绑定到 StateStore（必需）并订阅其已提交的属性变化。
        """
        super().__init__(config, store=store, dims=dims)
        if store is None:
            raise ConfigError(
                "SQLiteIndex requires a StateStore (store=...)",
                "CONFIG_001",
                details={"provider": "sqlite", "missing": "store"},
                suggestion="pass store=StateStore(...) or use the 'memory' index provider",
            )
        self._lock = threading.RLock()
        self._caches: dict[int, _VectorCache] = {}
        self._pending: list[tuple[str, dict[str, Any]]] = []
        # The last evaluated (cache key, filter) → mask: search() and count() of one query share the gate.
        # 最近一次求值的 (缓存键, 过滤器) → 掩码：同一次查询的 search() 与 count() 共享门控。
        self._last_mask: tuple[Any, IndexFilter, Any] | None = None
        listeners = getattr(store, "attr_listeners", None)
        if isinstance(listeners, list) and not any(getattr(cb, "__self__", None) is self for cb in listeners):
            listeners.append(self.update_attrs)

    @property
    def uses_numpy(self) -> bool:
        """numpy is used when importable and config.use_numpy == 'auto'.
        当 numpy 可导入且 config.use_numpy == 'auto' 时使用 numpy。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            May probe numpy once.
            可能探测一次 numpy。
        """
        return self.config.use_numpy == "auto" and numpy_module() is not None  # type: ignore[attr-defined]

    @property
    def supports_filter(self) -> bool:
        """True: an IndexFilter is a numpy mask over the cache, or the SQL gate of the store without numpy.
        为真：IndexFilter 是缓存上的 numpy 掩码，无 numpy 时则是 store 的 SQL 门控。

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

    # ------------------------------------------------------------------ cache plumbing
    # ------------------------------------------------------------------ 缓存管道
    def _cache(self, np: Any, dims: int) -> _VectorCache | None:
        """The synchronized cache for `dims` (None when a stored row has another dimensionality); caller holds _lock.
        与 `dims` 对应、已同步的缓存（某已存行维数不同时为 None）；调用方持有 _lock。
        """
        store = self.store
        cache = self._caches.get(dims)
        if cache is None:
            cache = _VectorCache(np, dims)
            self._caches[dims] = cache
        key = (int(store.data_generation), int(store.rebuild_generation), int(store.data_version()))
        if cache.key != key or self._pending:
            if cache.key is None or cache.key[1] != key[1] or cache.key[2] != key[2]:
                # A DELETE / ROLLBACK / reset of this connection, or a commit by another connection: rebuild.
                # 本连接的 DELETE / ROLLBACK / reset，或其它连接的提交：重建。
                cache.clear()
            cache.ingest(store.load_vector_rows(cache.last_rowid))
            pending, self._pending = self._pending, []
            for atom_id, attrs in pending:
                pos = cache.positions.get(atom_id)
                if pos is not None:
                    cache.set_attrs(pos, attrs)
            cache.key = key
            # Any change of rows or scalars invalidates the memoized mask (its key would not see pending attrs).
            # 行或标量的任何变化都使记忆化的掩码失效（其键看不到待处理的标量变化）。
            self._last_mask = None
        return cache if cache.usable else None

    def _mask(self, cache: _VectorCache, spec: IndexFilter) -> Any:
        """The filter mask over `cache`, memoized for the last (cache key, filter) pair; caller holds _lock.
        `cache` 上的过滤掩码，对最近一次 (缓存键, 过滤器) 记忆化；调用方持有 _lock。
        """
        memo = self._last_mask
        if memo is not None and spec.as_of is not None and memo[0] == (id(cache), cache.key) and memo[1] == spec:
            return memo[2]
        mask = cache.mask(spec, as_of_default=self.store.now())
        if spec.as_of is not None:
            self._last_mask = ((id(cache), cache.key), spec, mask)
        return mask

    def _synced(self, query_dims: int) -> _VectorCache | None:
        """The usable cache for a query dimensionality, or None (numpy off / mixed dims); caller holds _lock.
        某查询维数下可用的缓存，或 None（numpy 关闭 / 维数混杂）；调用方持有 _lock。
        """
        np = numpy_module() if self.uses_numpy else None
        if np is None:
            return None
        return self._cache(np, query_dims)

    def update_attrs(self, atom_id: str, attrs: Mapping[str, Any]) -> None:
        """Queue a committed scalar change (archive / restore / set_valid_to); applied at the next synchronization.
        排队一次已提交的标量变化（archive / restore / set_valid_to）；在下次同步时应用。

        Input / 输入：
            atom_id: Atom id (ids without a cached vector are ignored when applied).
            atom_id：原子 id（应用时没有缓存向量的 id 被忽略）。

            attrs: The scalars to overwrite (normalized like atom_index_attrs).
            attrs：要覆盖的标量（按 atom_index_attrs 的方式规范化）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Appends to the pending queue.
            追加到待处理队列。
        """
        changes = normalize_attrs(attrs)
        if changes:
            with self._lock:
                self._pending.append((str(atom_id), changes))

    # ------------------------------------------------------------------ IndexBase contract
    # ------------------------------------------------------------------ IndexBase 契约
    def upsert(
        self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping[str, Any] | None = None
    ) -> None:
        """Insert or replace one atom vector (attrs are ignored: the scalars are read from memory_atoms).
        插入或替换一个原子向量（attrs 被忽略：标量从 memory_atoms 读取）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            vector: Unit vector.
            vector：单位向量。

            embedder: Embedder identity.
            embedder：嵌入器身份。

            attrs: Ignored.
            attrs：忽略。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            store.upsert_vector (atom_vectors row replaced; the cache picks it up by rowid).
            store.upsert_vector（替换 atom_vectors 行；缓存按 rowid 拾取它）。
        """
        self.store.upsert_vector(atom_id, vector, embedder=embedder)

    def upsert_batch(
        self,
        items: list[tuple[str, list[float]]],
        *,
        embedder: str,
        attrs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Upsert several vectors inside one store transaction.
        在一个 store 事务内写入多个向量。

        Input / 输入：
            items: `[(atom_id, vector), ...]`.
            items：`[(atom_id, vector), ...]`。

            embedder: Embedder identity.
            embedder：嵌入器身份。

            attrs: Ignored.
            attrs：忽略。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes atom_vectors.
            写入 atom_vectors。
        """
        with self.store.transaction():
            for atom_id, vector in items:
                self.store.upsert_vector(atom_id, vector, embedder=embedder)

    def get(self, atom_id: str) -> list[float] | None:
        """Read one atom vector.
        读取一个原子向量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            The float32-decoded vector or None.
            解码自 float32 的向量或 None。

        State effect / 状态影响：
            None.
            无。
        """
        return self.store.load_vectors([atom_id]).get(atom_id)

    def delete(self, atom_id: str) -> None:
        """Remove one atom vector.
        删除一个原子向量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            store.delete_vector (the cache rebuilds on the next search).
            store.delete_vector（缓存在下次搜索时重建）。
        """
        self.store.delete_vector(atom_id)

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

            candidate_ids: Restrict to these ids (None → all rows).
            candidate_ids：限定这些 id（None → 全部行）。

            filter: Gate evaluated as a numpy mask over the cached scalars (the SQL gate of the store without numpy); with candidate_ids the intersection.
            filter：作为缓存标量上的 numpy 掩码求值的门控（无 numpy 时为 store 的 SQL 门控）；与 candidate_ids 同时给出时取交集。

            top_k: Maximum hits.
            top_k：最大命中数。

        Output / 输出：
            `[(atom_id, score), ...]`, at most top_k, bit-for-bit what rank_hits over the same vectors returns.
            `[(atom_id, score), ...]`，至多 top_k 个，与 rank_hits 在同一批向量上的结果逐位相同。

        State effect / 状态影响：
            Synchronizes the cache (incremental rows, pending attrs, or a rebuild).
            同步缓存（增量行、待处理标量或重建）。
        """
        if top_k <= 0 or (candidate_ids is not None and not candidate_ids):
            return []
        query = [float(v) for v in vector]
        # Lock order: store.lock BEFORE self._lock. Synchronizing the cache reads the store (generations,
        # load_vector_rows, now()); a search issued inside store.transaction() already holds the store lock, so
        # taking it first here (re-entrant) keeps every thread on one order and no two searches can deadlock.
        # 锁序：先 store.lock 再 self._lock。同步缓存要读 store（代数、load_vector_rows、now()）；在
        # store.transaction() 内发起的搜索已经持有 store 锁，因此这里先取它（可重入）让所有线程遵循同一顺序，
        # 两次搜索绝不会死锁。
        with self.store.lock, self._lock:
            cache = self._synced(len(query))
            if cache is not None:
                return self._search_cached(cache, query, candidate_ids, filter, top_k)
        wanted = None if candidate_ids is None else list(dict.fromkeys(str(i) for i in candidate_ids))
        if filter is not None:
            gated = self.gate_ids(filter) or []
            wanted = gated if wanted is None else [i for i in wanted if i in set(gated)]
            if not wanted:
                return []
        vectors = self.store.load_vectors(wanted)
        return rank_hits(query, vectors, top_k=top_k, use_numpy=self.uses_numpy)

    def _search_cached(
        self,
        cache: _VectorCache,
        query: list[float],
        candidate_ids: list[str] | None,
        filter: IndexFilter | None,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Rank inside the cache: candidate rows ∩ filter mask, full-matrix product when most rows qualify.
        在缓存内排序：候选行 ∩ 过滤掩码，多数行合格时用整矩阵乘积。
        """
        np = cache.np
        n = cache.n
        if n == 0:
            return []
        rows = None if candidate_ids is None else cache.rows_of(candidate_ids)
        if filter is not None:
            mask = self._mask(cache, filter)
            rows = np.nonzero(mask)[0] if rows is None else rows[mask[rows]]
        if rows is None:
            rows = np.arange(n)
        if len(rows) == 0:
            return []
        q = np.asarray(query, dtype="float64")
        if len(rows) * 4 >= n:
            # Most rows qualify: one product over the whole matrix, then the masked slice (no row gather).
            # 多数行合格：对整个矩阵做一次乘积，再取掩码切片（不聚集行）。
            scores = (cache.matrix[:n] @ q)[rows]
        else:
            scores = cache.matrix[rows] @ q
        return select_top_k(query, cache.ids, cache.matrix, top_k=top_k, rows=rows, scores=scores)

    def count(self, filter: IndexFilter | None = None) -> int:
        """Number of atom_vectors rows, or of the cached rows passing a filter.
        atom_vectors 行数，或通过过滤器的缓存行数。

        Input / 输入：
            filter: An IndexFilter (None → every row, via the store).
            filter：IndexFilter（None → 经 store 计数全部行）。

        Output / 输出：
            Count.
            数量。

        State effect / 状态影响：
            May synchronize the cache.
            可能同步缓存。
        """
        if filter is None:
            return self.store.count_vectors()
        with self.store.lock, self._lock:
            cache = self._synced(self.dims)
            if cache is not None:
                return int(self._mask(cache, filter).sum())
        gated = self.gate_ids(filter) or []
        return len(self.store.load_vector_blobs(gated)) if gated else 0

    def reset(self) -> None:
        """Drop every atom_vectors row.
        删除 atom_vectors 全部行。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            store.clear_vectors (the cache rebuilds on the next search).
            store.clear_vectors（缓存在下次搜索时重建）。
        """
        self.store.clear_vectors()

    def close(self) -> None:
        """Unsubscribe from the store's attribute listeners and drop the cache (idempotent).
        取消订阅 store 的属性监听器并丢弃缓存（幂等）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Removes this index's listener from store.attr_listeners.
            从 store.attr_listeners 移除本索引的监听器。
        """
        listeners = getattr(self.store, "attr_listeners", None)
        if isinstance(listeners, list):
            self.store.attr_listeners = [cb for cb in listeners if getattr(cb, "__self__", None) is not self]
        with self._lock:
            self._caches.clear()
            self._pending.clear()


__all__ = ["CODE_ATTRS", "INSTANT_ATTRS", "SQLiteIndex", "SQLiteIndexConfig"]
