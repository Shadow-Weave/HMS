"""In-memory vector index: a dict of atom_id → vector with the same contract as the SQLite provider.
内存向量索引：atom_id → 向量的字典，与 SQLite 提供者契约一致。
"""

# Purpose (EN): A store-free index for tests and ephemeral sessions; parity with SQLiteIndex is asserted by tests.
#               search() ranks through one cached float32 matrix (rebuilt lazily after writes) and `select_top_k`.
# 目的（中）：面向测试与临时会话的无存储索引；测试断言其与 SQLiteIndex 的一致性。search() 通过一个缓存的 float32
#             矩阵（写入后惰性重建）与 `select_top_k` 排序。

# Input (EN): MemoryIndexConfig (no fields), atom ids and vectors; no IndexFilter (supports_filter=False).
# 输入（中）：MemoryIndexConfig（无字段）、原子 id 与向量；不接受 IndexFilter（supports_filter=False）。

# Output (EN): Ranked hits, counts, info.
# 输出（中）：排序命中、计数、信息。

# State effect (EN): Mutates only its own dict and matrix cache.
# 状态影响（中）：仅修改自身字典与矩阵缓存。

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from hms_elite.index.base import (
    BaseIndexConfig,
    IndexBase,
    IndexFilter,
    numpy_module,
    quantize,
    rank_hits,
    select_top_k,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hms_elite.memory.storage import StateStore


class MemoryIndexConfig(BaseIndexConfig):
    """Memory index config (no fields).
    内存索引配置（无字段）。
    """


class MemoryIndex(IndexBase):
    """Dict-backed vector index; vectors are float32-quantized on upsert so scores match the SQLite provider.
    以字典为后端的向量索引；写入时向量按 float32 取整，使打分与 SQLite 提供者一致。
    """

    config_class: ClassVar[type[BaseIndexConfig]] = MemoryIndexConfig
    provider_name: ClassVar[str] = "memory"

    def __init__(
        self, config: MemoryIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64
    ) -> None:
        """Start with an empty dict (the store handle is accepted and unused).
        以空字典开始（接受但不使用 store 句柄）。
        """
        super().__init__(config, store=store, dims=dims)
        self._vectors: dict[str, list[float]] = {}
        self._embedders: dict[str, str] = {}
        # (ids, row positions, matrix) snapshot of the dict; None until search() rebuilds it after a write.
        # 字典的 (id 列表, 行位置, 矩阵) 快照；写入后为 None，直到 search() 重建。
        self._matrix: tuple[list[str], dict[str, int], Any] | None = None

    def upsert(
        self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping[str, Any] | None = None
    ) -> None:
        """Insert or replace one atom vector (attrs are ignored: this provider keeps no scalars).
        插入或替换一个原子向量（attrs 被忽略：本提供者不保存标量）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            vector: Unit vector (stored float32-rounded).
            vector：单位向量（按 float32 取整存储）。

            embedder: Embedder identity (kept for info only).
            embedder：嵌入器身份（仅作记录）。

            attrs: Ignored.
            attrs：忽略。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes the dict and drops the matrix cache.
            写入字典并丢弃矩阵缓存。
        """
        self._vectors[str(atom_id)] = quantize(vector)
        self._embedders[str(atom_id)] = str(embedder)
        self._matrix = None

    def get(self, atom_id: str) -> list[float] | None:
        """Read one atom vector.
        读取一个原子向量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            A copy of the vector or None.
            向量副本或 None。

        State effect / 状态影响：
            None.
            无。
        """
        vector = self._vectors.get(str(atom_id))
        return list(vector) if vector is not None else None

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
            Writes the dict and drops the matrix cache.
            写入字典并丢弃矩阵缓存。
        """
        self._vectors.pop(str(atom_id), None)
        self._embedders.pop(str(atom_id), None)
        self._matrix = None

    def _snapshot_matrix(self, np: Any) -> tuple[list[str], dict[str, int], Any] | None:
        """The cached (ids, positions, matrix) over a snapshot of the dict; None when dims are mixed.
        基于字典快照缓存的 (id 列表, 位置, 矩阵)；维数不一致时为 None。
        """
        cached = self._matrix
        if cached is not None:
            return cached
        # Snapshot the live dict first: an upsert() from another thread (AsyncHMS worker pool) must not resize it
        # while the rows are being copied ("dictionary changed size during iteration").
        # 先对实时字典做快照：另一线程的 upsert()（AsyncHMS 工作线程池）不得在复制行时改变其大小
        # （"dictionary changed size during iteration"）。
        vectors = dict(self._vectors)
        ids = list(vectors)
        if not ids:
            return None
        dims = len(vectors[ids[0]])
        if any(len(vectors[i]) != dims for i in ids):
            return None
        matrix = np.asarray([vectors[i] for i in ids], dtype="<f4")
        cached = (ids, {atom_id: pos for pos, atom_id in enumerate(ids)}, matrix)
        self._matrix = cached
        return cached

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

            candidate_ids: Restrict to these ids (None → all).
            candidate_ids：限定这些 id（None → 全部）。

            filter: Not supported (ValidationError): gate in SQL and pass candidate_ids.
            filter：不支持（ValidationError）：先在 SQL 里门控再传 candidate_ids。

            top_k: Maximum hits.
            top_k：最大命中数。

        Output / 输出：
            `[(atom_id, score), ...]`, at most top_k, bit-for-bit what rank_hits over the same vectors returns.
            `[(atom_id, score), ...]`，至多 top_k 个，与 rank_hits 在同一批向量上的结果逐位相同。

        State effect / 状态影响：
            May rebuild the matrix cache.
            可能重建矩阵缓存。
        """
        self._reject_filter(filter)
        if top_k <= 0:
            return []
        query = [float(v) for v in vector]
        np = numpy_module() if self.uses_numpy else None
        if np is not None and len(self._vectors) > top_k:
            snapshot = self._snapshot_matrix(np)
            if snapshot is not None and snapshot[2].shape[1] == len(query):
                ids, positions, matrix = snapshot
                if candidate_ids is None:
                    return select_top_k(query, ids, matrix, top_k=top_k)
                rows = [positions[i] for i in dict.fromkeys(str(i) for i in candidate_ids) if i in positions]
                if len(rows) > top_k:
                    return select_top_k(query, [ids[r] for r in rows], matrix[rows], top_k=top_k)
        if candidate_ids is None:
            # Snapshot the live dict: an upsert() from another thread (AsyncHMS runs sync calls in a worker pool) must
            # not resize it while rank_hits iterates ("dictionary changed size during iteration").
            # 对实时字典做快照：另一线程的 upsert()（AsyncHMS 在工作线程池中运行同步调用）不得在 rank_hits 迭代时
            # 改变其大小（"dictionary changed size during iteration"）。
            vectors = dict(self._vectors)
        else:
            wanted = dict.fromkeys(str(i) for i in candidate_ids)
            vectors = {i: self._vectors[i] for i in wanted if i in self._vectors}
        return rank_hits(query, vectors, top_k=top_k, use_numpy=self.uses_numpy)

    def count(self, filter: IndexFilter | None = None) -> int:
        """Number of stored vectors (a filter is rejected: this provider keeps no scalars).
        已存储向量数（不接受过滤器：本提供者不保存标量）。

        Input / 输入：
            filter: Must be None (ValidationError otherwise).
            filter：必须为 None（否则抛 ValidationError）。

        Output / 输出：
            Count.
            数量。

        State effect / 状态影响：
            None.
            无。
        """
        self._reject_filter(filter)
        return len(self._vectors)

    def reset(self) -> None:
        """Clear the dict.
        清空字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Clears the dict and the matrix cache.
            清空字典与矩阵缓存。
        """
        self._vectors.clear()
        self._embedders.clear()
        self._matrix = None


__all__ = ["MemoryIndex", "MemoryIndexConfig"]
