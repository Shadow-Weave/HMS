"""Vector index contract: BaseIndexConfig, IndexFilter, IndexBase and the shared ranking helpers of every provider.
向量索引契约：BaseIndexConfig、IndexFilter、IndexBase 以及所有提供者共用的排序辅助函数。
"""

# Purpose (EN): One pluggable index interface (upsert / get / delete / search / count / reset / info / update_attrs /
#               close) over atom vectors, plus the bounded top-k selection every provider ranks with.
# 目的（中）：面向原子向量的统一可插拔索引接口（upsert / get / delete / search / count / reset / info / update_attrs /
#             close），以及所有提供者共用的有界 top-k 选择。

# Input (EN): Atom ids, unit vectors, optional candidate id lists, an optional IndexFilter, scalar attrs, top_k.
# 输入（中）：原子 id、单位向量、可选的候选 id 列表、可选的 IndexFilter、标量属性、top_k。

# Output (EN): (atom_id, score) hits sorted by score DESC then atom_id ASC, at most top_k of them.
# 输出（中）：按 score 降序、atom_id 升序排列的 (atom_id, score) 命中，至多 top_k 个。

# State effect (EN): Providers mutate their own storage (atom_vectors table, an in-memory dict, a Milvus collection).
# 状态影响（中）：提供者修改各自的存储（atom_vectors 表、内存字典、Milvus 集合）。

from __future__ import annotations

import heapq
import logging
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

from pydantic import model_validator

from hms_elite.exceptions import ValidationError
from hms_elite.models import HMSModel, MemoryAtom, Scope
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.factory import check_extra_fields, coerce_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hms_elite.memory.storage import StateStore

logger = logging.getLogger(__name__)

# Tri-state numpy sentinel: None = not probed yet, False = unavailable, module = available. Logged once.
# 三态 numpy 哨兵：None = 尚未探测，False = 不可用，module = 可用。只记录一次日志。
_NUMPY_STATE: dict[str, Any] = {"module": None}

# Scalar attributes a provider may keep beside each vector (Milvus pushes the gate down onto them); NULL is stored as
# "" for strings so lexical comparison on the fixed-width UTC ISO instants stays a time comparison.
# 提供者可随向量保存的标量属性（Milvus 把门控下推到它们上面）；字符串的 NULL 存为 ""，使定宽 UTC ISO 时刻的
# 字典序比较仍是时间比较。
INDEX_ATTR_KEYS: tuple[str, ...] = (
    "user_id",
    "agent_id",
    "run_id",
    "time_anchor",
    "valid_from",
    "valid_to",
    "archived",
    "scale_level",
    "obs_type",
    "atom_type",
)
_INSTANT_ATTRS: frozenset[str] = frozenset({"time_anchor", "valid_from", "valid_to"})
# Matrix-vs-row dot products differ in the last bits (~1e-16 on unit vectors); the selection margin is far wider, so
# the exact rescoring never misses a row of the true top-k and never has to look at more than a handful extra.
# 矩阵乘与逐行点积只在末位不同（单位向量上约 1e-16）；选择余量远大于此，因此精确重算既不会漏掉真实 top-k 的
# 任何一行，也不会多看几行以上。
_SELECT_EPS = 1e-12


def numpy_module() -> Any:
    """Return the numpy module when importable (probed once, logged once), else None.
    可导入时返回 numpy 模块（只探测一次、只记录一次日志），否则返回 None。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The numpy module or None.
        numpy 模块或 None。

    State effect / 状态影响：
        Caches the probe result in a module-level sentinel.
        将探测结果缓存到模块级哨兵。
    """
    state = _NUMPY_STATE["module"]
    if state is None:
        try:
            import numpy  # noqa: PLC0415 - optional dependency probed lazily

            _NUMPY_STATE["module"] = numpy
            logger.info("numpy %s available: vector dot products are vectorized", numpy.__version__)
        except ImportError:
            _NUMPY_STATE["module"] = False
            logger.info("numpy not installed: vector dot products use math.fsum")
        state = _NUMPY_STATE["module"]
    return state or None


def quantize(vector: Iterable[float]) -> list[float]:
    """Round a vector through little-endian float32 (the on-disk representation) so every provider scores alike.
    将向量经小端 float32（磁盘表示）往返取整，使所有提供者的打分一致。

    Input / 输入：
        vector: Floats.
        vector：浮点数。

    Output / 输出：
        The float32-rounded floats.
        float32 取整后的浮点数。

    State effect / 状态影响：
        None.
        无。
    """
    values = [float(v) for v in vector]
    if not values:
        return values
    return list(struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values)))


def dot(a: list[float], b: list[float], *, use_numpy: bool = True) -> float:
    """Dot product (numpy when allowed and available, else math.fsum).
    点积（允许且可用时用 numpy，否则用 math.fsum）。

    Input / 输入：
        a: Vector.
        a：向量。

        b: Vector of the same length (extra components are ignored).
        b：等长向量（多余分量忽略）。

        use_numpy: Whether numpy may be used.
        use_numpy：是否允许使用 numpy。

    Output / 输出：
        The dot product as a float.
        浮点点积。

    State effect / 状态影响：
        None.
        无。
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    np = numpy_module() if use_numpy else None
    if np is not None:
        return float(np.dot(np.asarray(a[:n], dtype="float64"), np.asarray(b[:n], dtype="float64")))
    return math.fsum(x * y for x, y in zip(a[:n], b[:n]))


def rank_hits(
    query: list[float], vectors: dict[str, list[float]], *, top_k: int, use_numpy: bool = True
) -> list[tuple[str, float]]:
    """Score every vector against the query (max(0, dot)) and sort by (score DESC, atom_id ASC).
    对每个向量与查询打分（max(0, dot)），按（score 降序，atom_id 升序）排序。

    Input / 输入：
        query: The query vector (unit length expected).
        query：查询向量（期望为单位长度）。

        vectors: {atom_id: vector} candidates.
        vectors：候选 {atom_id: 向量}。

        top_k: Maximum hits (<= 0 → []).
        top_k：最大命中数（<= 0 → []）。

        use_numpy: Whether numpy may be used.
        use_numpy：是否允许使用 numpy。

    Output / 输出：
        Sorted (atom_id, score) pairs.
        排序后的 (atom_id, score) 对。

    State effect / 状态影响：
        None.
        无。
    """
    if top_k <= 0 or not vectors:
        return []
    scored = [(atom_id, max(0.0, dot(vec, query, use_numpy=use_numpy))) for atom_id, vec in vectors.items()]
    if top_k < len(scored):
        # A bounded heap: O(n log k) instead of a full sort, same order as sorting by (score DESC, atom_id ASC).
        # 有界堆：O(n log k) 而非全排序，顺序与按（score 降序，atom_id 升序）排序相同。
        return heapq.nsmallest(top_k, scored, key=lambda hit: (-hit[1], hit[0]))
    scored.sort(key=lambda hit: (-hit[1], hit[0]))
    return scored


def select_top_k(
    query: list[float],
    ids: list[str],
    matrix: Any,
    *,
    top_k: int,
    rows: Any = None,
    scores: Any = None,
) -> list[tuple[str, float]]:
    """Bounded top-k over a float-valued matrix: one matmul picks the rows, `dot` rescores them bit-for-bit.
    对浮点矩阵的有界 top-k：一次矩阵乘挑出行，再用 `dot` 逐位精确重算分数。

    Input / 输入：
        query: The query vector (same dimensionality as the matrix columns).
        query：查询向量（与矩阵列数同维）。

        ids: Row ids, aligned with the matrix rows.
        ids：与矩阵行对齐的行 id。

        matrix: A numpy array of shape (len(ids), dims) holding the stored float32 values (float32 or float64 dtype).
        matrix：形状为 (len(ids), dims) 的 numpy 数组，存放已存 float32 值（float32 或 float64 dtype）。

        top_k: Maximum hits (<= 0 → []).
        top_k：最大命中数（<= 0 → []）。

        rows: Optional integer index array of the candidate rows within `ids` / `matrix` (None → every row).
        rows：可选的整数索引数组，给出 `ids` / `matrix` 中的候选行（None → 全部行）。

        scores: Optional float64 matmul scores aligned with `rows` (None → computed here; a caller holding a full-matrix product passes its masked slice so no row is gathered twice).
        scores：可选的、与 `rows` 对齐的 float64 矩阵乘分数（None → 在此计算；持有整矩阵乘积的调用方传入其掩码切片，任何行都不必再聚集一次）。

    Output / 输出：
        Exactly what `rank_hits(query, {ids[r]: matrix[r] for r in rows}, top_k=top_k, use_numpy=True)[:top_k]`
        returns: the scores are recomputed with `dot` on the selected rows, the zero-score tail follows atom_id order.
        与 `rank_hits(query, {ids[r]: matrix[r] for r in rows}, top_k=top_k, use_numpy=True)[:top_k]` 完全相同：
        分数在被选中的行上用 `dot` 重算，零分尾部按 atom_id 顺序。

    State effect / 状态影响：
        None.
        无。
    """
    np = numpy_module()
    if top_k <= 0 or np is None:
        return []
    if rows is None:
        rows = np.arange(len(ids))
    n = int(len(rows))
    if n == 0:
        return []
    q = np.asarray(query, dtype="float64")
    if scores is None:
        scores = matrix[rows].astype("float64", copy=False) @ q
    if top_k >= n:
        chosen = range(n)
        kth = -math.inf
    else:
        kth = float(np.partition(scores, n - top_k)[n - top_k])
        # Near the zero clamp every row with a non-negative product may enter the ranked prefix and the tail is
        # ordered by id: rescore everything that can be positive; otherwise only the rows within the margin.
        # 靠近零钳位时，任何积非负的行都可能进入排序前缀，且尾部按 id 排序：重算所有可能为正的行；
        # 否则只重算余量之内的行。
        threshold = -_SELECT_EPS if kth <= 4 * _SELECT_EPS else kth - 2 * _SELECT_EPS
        chosen = np.nonzero(scores >= threshold)[0].tolist()
    positives: list[tuple[str, float]] = []
    zero_ids: list[str] = []
    # `dot` is np.dot over two contiguous float64 arrays; a float64 matrix row IS that array already, so the
    # list → array round trip is skipped (same inputs, same BLAS call, same bits: test_select_top_k_row_dot_is_exact).
    # `dot` 就是对两个连续 float64 数组的 np.dot；float64 矩阵的一行本身就是那个数组，因此跳过 list → array 往返
    # （同样的输入、同一个 BLAS 调用、同样的位：test_select_top_k_row_dot_is_exact）。
    direct = matrix.dtype == np.float64 and q.shape[0] == matrix.shape[1]
    query_list = [float(v) for v in query]
    for i in chosen:
        r = int(rows[i])
        exact = float(np.dot(matrix[r], q)) if direct else dot(matrix[r].tolist(), query_list, use_numpy=True)
        if exact > 0.0:
            positives.append((ids[r], exact))
        else:
            zero_ids.append(ids[r])
    positives.sort(key=lambda hit: (-hit[1], hit[0]))
    if len(positives) >= top_k:
        return positives[:top_k]
    if kth > 4 * _SELECT_EPS:
        # Rows outside the margin all score below the k-th positive: they cannot enter the prefix.
        # 余量之外的行分数都低于第 k 个正分行：它们进不了前缀。
        return positives
    if top_k < n:
        # The rows never rescored are certainly non-positive: they join the zero tail in id order.
        # 从未重算的行必定非正：按 id 顺序加入零分尾部。
        chosen_set = set(chosen)
        zero_ids.extend(ids[int(rows[i])] for i in range(n) if i not in chosen_set)
    zero_ids.sort()
    return (positives + [(atom_id, 0.0) for atom_id in zero_ids])[:top_k]


def atom_index_attrs(atom: MemoryAtom) -> dict[str, Any]:
    """The INDEX_ATTR_KEYS scalars of an atom, NULL strings as "" and `archived` as a bool.
    原子的 INDEX_ATTR_KEYS 标量，字符串 NULL 为 ""，`archived` 为布尔值。

    Input / 输入：
        atom: A stored MemoryAtom (valid_from assigned).
        atom：已存储的 MemoryAtom（valid_from 已分配）。

    Output / 输出：
        {key: value} for every INDEX_ATTR_KEYS key.
        每个 INDEX_ATTR_KEYS 键的 {key: value}。

    State effect / 状态影响：
        None.
        无。
    """
    scope = atom.scope
    return normalize_attrs(
        {
            "user_id": scope.user_id,
            "agent_id": scope.agent_id,
            "run_id": scope.run_id,
            "time_anchor": atom.time_anchor,
            "valid_from": atom.valid_from,
            "valid_to": atom.valid_to,
            "archived": atom.archived_reason is not None,
            "scale_level": atom.scale_level,
            "obs_type": atom.obs_type,
            "atom_type": atom.atom_type,
        }
    )


def normalize_attrs(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only INDEX_ATTR_KEYS, coerce None → "", instants → fixed-width UTC ISO, `archived` → bool.
    只保留 INDEX_ATTR_KEYS，None → ""，时刻 → 定宽 UTC ISO，`archived` → 布尔值。

    Input / 输入：
        attrs: Any mapping (unknown keys are ignored; None → {}).
        attrs：任意映射（未知键忽略；None → {}）。

    Output / 输出：
        The normalized subset (only the keys present in `attrs`).
        规范化后的子集（只含 `attrs` 中出现的键）。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if key not in INDEX_ATTR_KEYS:
            continue
        if key == "archived":
            out[key] = bool(value)
        elif value is None or value == "":
            out[key] = ""
        elif key in _INSTANT_ATTRS:
            out[key] = to_utc_iso(value)
        else:
            out[key] = str(value)
    return out


@dataclass(frozen=True)
class IndexFilter:
    """A gate a provider may push down into its own search (scope, validity at `as_of`, scale / obs_type, time).
    提供者可下推到自身搜索里的门控（scope、`as_of` 时刻的有效性、尺度 / obs_type、时间）。

    Semantics equal the SQL gate of StateStore.axis_select: unset scope keys add no predicate (Scope.matches);
    `include_expired=False` requires `archived == false and valid_from <= as_of and (valid_to == "" or valid_to > as_of)`;
    `scale_in` / `obs_type_in` / `atom_type_in` are whitelists; `time_from` / `time_to` bound `time_anchor` inclusively
    (rows without a time_anchor never match a time bound, like SQL).
    语义与 StateStore.axis_select 的 SQL 门控相同：未设置的 scope 键不加谓词（Scope.matches）；
    `include_expired=False` 要求 `archived == false and valid_from <= as_of and (valid_to == "" or valid_to > as_of)`；
    `scale_in` / `obs_type_in` / `atom_type_in` 是白名单；`time_from` / `time_to` 对 `time_anchor` 做闭区间约束
    （没有 time_anchor 的行与 SQL 一样不匹配任何时间约束）。
    """

    scope: Scope | None = None
    as_of: str | None = None
    include_expired: bool = False
    scale_in: tuple[str, ...] | None = None
    obs_type_in: tuple[str, ...] | None = None
    time_from: str | None = None
    time_to: str | None = None
    atom_type_in: tuple[str, ...] | None = None


class BaseIndexConfig(HMSModel):
    """Base index config (no fields; providers add their own).
    基础索引配置（无字段；提供者自行扩展）。
    """

    @model_validator(mode="before")
    @classmethod
    def _extra_fields(cls, data: Any) -> Any:
        """Unknown keys → ConfigError(CONFIG_001) with a did-you-mean hint.
        未知键 → 带 did-you-mean 提示的 ConfigError(CONFIG_001)。
        """
        return check_extra_fields(cls, data)


class IndexBase(ABC):
    """Abstract vector index over atom ids.
    面向原子 id 的抽象向量索引。
    """

    config_class: ClassVar[type[BaseIndexConfig]] = BaseIndexConfig
    provider_name: ClassVar[str] = "base"

    def __init__(
        self, config: BaseIndexConfig | dict | None = None, *, store: StateStore | None = None, dims: int = 64
    ) -> None:
        """Coerce the config and remember the store handle and dimensionality.
        规整配置并记录 store 句柄与维数。
        """
        self.config: BaseIndexConfig = coerce_config(self.config_class, config)  # type: ignore[assignment]
        self.store = store
        self.dims = int(dims)

    @abstractmethod
    def upsert(
        self, atom_id: str, vector: list[float], *, embedder: str, attrs: Mapping[str, Any] | None = None
    ) -> None:
        """Insert or replace one atom vector (and, for providers that keep them, its scalar attrs).
        插入或替换一个原子向量（保存标量的提供者同时写入其标量属性）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            vector: Unit vector.
            vector：单位向量。

            embedder: Embedder identity that produced it.
            embedder：生成该向量的嵌入器身份。

            attrs: Optional INDEX_ATTR_KEYS scalars (see atom_index_attrs); ignored by providers without scalars.
            attrs：可选的 INDEX_ATTR_KEYS 标量（见 atom_index_attrs）；无标量的提供者忽略。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes the provider storage.
            写入提供者存储。
        """

    def upsert_batch(
        self,
        items: list[tuple[str, list[float]]],
        *,
        embedder: str,
        attrs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Upsert several vectors sequentially.
        顺序写入多个向量。

        Input / 输入：
            items: `[(atom_id, vector), ...]`.
            items：`[(atom_id, vector), ...]`。

            embedder: Embedder identity.
            embedder：嵌入器身份。

            attrs: Optional `{atom_id: attrs}` scalars per item.
            attrs：可选的每项标量 `{atom_id: attrs}`。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes the provider storage.
            写入提供者存储。
        """
        for atom_id, vector in items:
            self.upsert(atom_id, vector, embedder=embedder, attrs=(attrs or {}).get(str(atom_id)))

    @abstractmethod
    def get(self, atom_id: str) -> list[float] | None:
        """Read one atom vector.
        读取一个原子向量。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            The vector or None.
            向量或 None。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
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
            Writes the provider storage.
            写入提供者存储。
        """

    @abstractmethod
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

            candidate_ids: Restrict to these ids (None → no id restriction); ids without a vector are skipped.
            candidate_ids：限定这些 id（None → 不限定 id）；没有向量的 id 被跳过。

            filter: A gate pushed down by `supports_filter` providers (others raise ValidationError); with candidate_ids the result is the intersection.
            filter：由 `supports_filter` 的提供者下推的门控（其它提供者抛 ValidationError）；与 candidate_ids 同时给出时结果为交集。

            top_k: Maximum hits.
            top_k：最大命中数。

        Output / 输出：
            `[(atom_id, score), ...]`, at most top_k.
            `[(atom_id, score), ...]`，至多 top_k 个。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
    def count(self, filter: IndexFilter | None = None) -> int:
        """Number of stored vectors, or of the stored vectors passing a pushed-down gate.
        已存储向量数，或通过下推门控的已存储向量数。

        Input / 输入：
            filter: An IndexFilter for `supports_filter` providers (others raise ValidationError); None → all rows.
            filter：供 `supports_filter` 提供者使用的 IndexFilter（其它提供者抛 ValidationError）；None → 全部行。

        Output / 输出：
            Count (atoms without a vector are never counted).
            数量（没有向量的原子从不计入）。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
    def reset(self) -> None:
        """Drop every vector.
        清空全部向量。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Clears the provider storage.
            清空提供者存储。
        """

    def update_attrs(self, atom_id: str, attrs: Mapping[str, Any]) -> None:
        """Overwrite some INDEX_ATTR_KEYS scalars of one atom; a no-op for providers without scalars.
        覆盖一个原子的部分 INDEX_ATTR_KEYS 标量；无标量的提供者为空操作。

        Input / 输入：
            atom_id: Atom id (absent ids are ignored).
            atom_id：原子 id（不存在的 id 忽略）。

            attrs: The scalars to overwrite (e.g. `{'archived': True, 'valid_to': '...'}`).
            attrs：要覆盖的标量（如 `{'archived': True, 'valid_to': '...'}`）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Provider-specific (SQLite stays the source of truth; StateStore calls this after COMMIT).
            因提供者而异（SQLite 仍是唯一真源；StateStore 在 COMMIT 后调用它）。
        """
        return None

    def close(self) -> None:
        """Release external resources (idempotent); a no-op for providers without a client of their own.
        释放外部资源（幂等）；没有自有客户端的提供者为空操作。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Provider-specific.
            因提供者而异。
        """
        return None

    @property
    def supports_filter(self) -> bool:
        """Whether search() accepts an IndexFilter (native pushdown); False → callers gate in SQL first.
        search() 是否接受 IndexFilter（原生下推）；False → 调用方先在 SQL 里门控。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        return False

    def _reject_filter(self, filter: IndexFilter | None) -> None:
        """ValidationError(VALIDATION_000) when a filter reaches a provider without pushdown.
        过滤器到达不支持下推的提供者时抛 ValidationError(VALIDATION_000)。
        """
        if filter is not None and not self.supports_filter:
            raise ValidationError(
                f"the {self.provider_name!r} index provider does not push filters down; gate the candidate ids in "
                "SQL first (StateStore.axis_ids) and pass candidate_ids=",
                "VALIDATION_000",
                details={"provider": self.provider_name, "filter": "unsupported"},
            )

    @property
    def uses_numpy(self) -> bool:
        """Whether dot products go through numpy for this provider.
        本提供者的点积是否经由 numpy。

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
        return numpy_module() is not None

    def info(self) -> dict:
        """Provider summary.
        提供者摘要。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `{'provider', 'dims', 'count', 'numpy': bool}`.
            `{'provider', 'dims', 'count', 'numpy': bool}`。

        State effect / 状态影响：
            None.
            无。
        """
        return {"provider": self.provider_name, "dims": self.dims, "count": self.count(), "numpy": self.uses_numpy}

    def gate_ids(self, filter: IndexFilter) -> list[str] | None:
        """Evaluate an IndexFilter as a SQL gate on the bound store (the fallback for providers without pushdown).
        在绑定的 store 上把 IndexFilter 评估为 SQL 门控（无下推能力的提供者的退路）。

        Input / 输入：
            filter: The gate.
            filter：门控。

        Output / 输出：
            Gated atom ids in index order, or None when no store is bound.
            索引序的门控原子 id；未绑定 store 时为 None。

        State effect / 状态影响：
            None.
            无。
        """
        store = self.store
        if store is None:
            return None
        clauses: list[str] = []
        params: list[Any] = []
        if filter.obs_type_in:
            kinds = [str(v) for v in filter.obs_type_in]
            clauses.append(f"memory_atoms.obs_type IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)
        if filter.time_from:
            clauses.append("memory_atoms.time_anchor >= ?")
            params.append(to_utc_iso(filter.time_from))
        if filter.time_to:
            clauses.append("memory_atoms.time_anchor <= ?")
            params.append(to_utc_iso(filter.time_to))
        return store.axis_ids(
            filter.scope if filter.scope is not None else Scope(),
            where_sql=" AND ".join(clauses),
            params=params,
            as_of=filter.as_of,
            include_expired=bool(filter.include_expired),
            scale_constraint=list(filter.scale_in) if filter.scale_in else None,
            atom_types=list(filter.atom_type_in) if filter.atom_type_in else None,
            order=None,
        )


__all__ = [
    "INDEX_ATTR_KEYS",
    "BaseIndexConfig",
    "IndexBase",
    "IndexFilter",
    "atom_index_attrs",
    "dot",
    "normalize_attrs",
    "numpy_module",
    "quantize",
    "rank_hits",
    "select_top_k",
]
