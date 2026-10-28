"""One SQLite connection, one lock, typed writers for every table; the only place SQL is executed.
单一 SQLite 连接、单一锁、每张表的类型化写入器；唯一执行 SQL 的地方。
"""

# Purpose (EN): Guarded persistence for all planes (evidence, observation/state, State Index, Support Graph, ops).
# 目的（中）：全平面（证据、观察/状态、状态索引、支持图、运维）的受控持久化。

# Input (EN): Pydantic models, scopes, ids, raw SQL fragments from the filter planner.
# 输入（中）：pydantic 模型、范围、id，以及过滤规划器生成的 SQL 片段。

# Output (EN): Hydrated models, rows, ids and counts.
# 输出（中）：水合后的模型、行、id 与计数。

# State effect (EN): Writes the database inside transaction(); the raw cursor is private. purge_rows() is the ONLY
#                    physical DELETE path of state rows (retention escape hatch, §16.6); vacuum() rewrites the file.
# 状态影响（中）：在 transaction() 内写库；游标私有。purge_rows() 是状态行唯一的物理 DELETE 路径
#                （保留策略逃生舱，§16.6）；vacuum() 重写文件。

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import struct
import threading
from calendar import timegm
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from hms_elite.enums import KIND_ORDER, NodeKind, SourceKind
from hms_elite.exceptions import StoreError, ValidationError, WitnessMissingError
from hms_elite.memory.schema import (
    PURGE_FLAG_KEY,
    SCOPED_TABLES,
    TABLES,
    TIME_INDEX,
    apply_migrations,
    backfill_fts,
    clone_schema,
    detect_capabilities,
    drop_all,
    fts_tokens,
)
from hms_elite.models import (
    EP,
    Anchor,
    BoundaryEdge,
    Checkpoint,
    FeedbackEvent,
    GraphEdge,
    GraphNode,
    InterfaceEP,
    Invariant,
    Kappa,
    MemoryAtom,
    Observation,
    Residual,
    Scope,
    Slots,
    SourceEvent,
    Transition,
)
from hms_elite.timeutil import now_iso, parse_iso, to_utc_iso
from hms_elite.utils.json import canonical_json
from hms_elite.utils.text import ordered_union, sha256_hex

logger = logging.getLogger(__name__)

# (key, canonical, entity_type, aliases) (§12.5)
# （键、规范文本、实体类型、别名）（§12.5）
EntityKey = tuple[str, str, str, tuple[str, ...]]

# The ONE liveness predicate (§8): two positional params, both the `as_of` instant.
# 唯一的存活性谓词（§8）：两个位置参数，都是同一个 `as_of` 时刻。
LIVE_SQL = (
    "memory_atoms.archived_reason IS NULL AND memory_atoms.valid_from <= ? "
    "AND (memory_atoms.valid_to IS NULL OR memory_atoms.valid_to > ?)"
)
DEFAULT_ATOM_ORDER = "time_anchor DESC, created_at DESC, atom_id ASC"
NODE_KIND_OF_SOURCE: dict[str, str] = {
    SourceKind.message.value: NodeKind.message.value,
    SourceKind.observation.value: NodeKind.observation.value,
    SourceKind.event.value: NodeKind.event.value,
    SourceKind.entity.value: NodeKind.entity.value,
    SourceKind.ep.value: NodeKind.ep.value,
    SourceKind.interface_ep.value: NodeKind.interface_ep.value,
    SourceKind.skill.value: NodeKind.skill.value,
    SourceKind.procedural.value: NodeKind.ep.value,
    SourceKind.feedback.value: NodeKind.observation.value,
}
_TABLE_RE = re.compile(r"\b(" + "|".join(TABLES) + r")\b")
# Statement text → tables it names. A pure function of the SQL, so the memo carries no store state; bounded.
# 语句文本 → 其提及的表。它是 SQL 的纯函数，因此备忘不携带任何 store 状态；有上限。
_TABLES_OF: dict[str, frozenset[str]] = {}
_TABLES_OF_LIMIT = 4096
_NOT_NULL_RE = re.compile(r"NOT NULL constraint failed: (\w+)\.(\w+)")
_ATOM_COLS = (
    "atom_id|memory_id|version|atom_type|content_json|text|hash|time_anchor|valid_from|valid_to|scale_level|role|"
    "sentiment|subject|predicate|obs_type|source_kind|source_id|source_ep_ids_json|confidence|extracted_by|"
    "primary_anchor_id|visibility|user_id|agent_id|run_id|actor_id|attributed_to|memory_type|user_meta_json|"
    "archived_reason|created_at|updated_at"
)
_QUALIFY_RE = re.compile(r"(?<![\w.])(" + _ATOM_COLS + r")(?![\w.(])")
_IN_CHUNK = 500
_SCOPE_COLS = ("user_id", "agent_id", "run_id")
# Scalar attrs mirrored by index providers that keep them (Milvus); read by atom_attrs() with NULL → "".
# 保存标量的索引提供者（Milvus）所镜像的属性；由 atom_attrs() 读取，NULL → ""。
_ATTR_SQL = (
    "SELECT atom_id, user_id, agent_id, run_id, time_anchor, valid_from, valid_to, archived_reason, scale_level, "
    "obs_type, atom_type FROM memory_atoms WHERE atom_id IN ({marks})"
)
# The atom_vectors rows joined to their gate scalars, in rowid order: the SQLite index provider builds and extends its
# in-process matrix cache from this shape (`v.rowid > ?` reads only what was appended since the last refresh).
# atom_vectors 行连接其门控标量、按 rowid 排列：SQLite 索引提供者用这一形状构建并增量扩展进程内矩阵缓存
# （`v.rowid > ?` 只读取上次刷新之后追加的行）。
_VECTOR_ROWS_SQL = (
    "SELECT v.rowid AS rowid, v.atom_id AS atom_id, v.dims AS dims, v.embedding AS embedding, a.user_id, a.agent_id, "
    "a.run_id, a.time_anchor, a.valid_from, a.valid_to, a.archived_reason, a.scale_level, a.obs_type, a.atom_type "
    "FROM atom_vectors v JOIN memory_atoms a ON a.atom_id = v.atom_id WHERE v.rowid > ? ORDER BY v.rowid"
)
# Tables whose writes invalidate an index provider's cached view of the vectors and their gate scalars (§8 generations).
# 其写入会使索引提供者缓存的向量与门控标量视图失效的表（§8 代数）。
_CACHE_TABLES: frozenset[str] = frozenset({"memory_atoms", "atom_vectors"})
# Every column (and JSON id list) that can cite an anchor: anchor_reference_count() sums them (§16.6, keep_witnessed=False).
# 所有可能引用锚点的列（及 JSON id 列表）：anchor_reference_count() 对它们求和（§16.6，keep_witnessed=False）。
_ANCHOR_REF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("atom_anchor", "anchor_id"),
    ("observations", "anchor_id"),
    ("eps", "anchor_id"),
    ("memory_atoms", "primary_anchor_id"),
    ("graph_nodes", "primary_anchor_id"),
    ("graph_edges", "primary_witness_anchor_id"),
    ("graph_edges", "resolved_by_anchor_id"),
    ("edge_witnesses", "anchor_id"),
    ("interface_eps", "primary_anchor_id"),
    ("feedback_events", "anchor_id"),
    ("history", "anchor_id"),
    ("checkpoints", "anchor_id"),
    ("transitions", "w_t_primary_anchor_id"),
    ("compiled_states", "anchor_id"),
)
_ANCHOR_REF_JSON: tuple[tuple[str, str], ...] = (
    ("observations", "anchor_ids_json"),
    ("eps", "anchor_ids_json"),
    ("interface_eps", "evidence_anchor_ids_json"),
    ("transitions", "w_t_json"),
)
# Dependent rows of an atom version, deleted in this order by purge_rows() (children before the atom row).
# 原子版本的从属行，purge_rows() 按此顺序删除（先子行，后原子行）。
_PURGE_TABLES: tuple[tuple[str, str], ...] = (
    ("atom_vectors", "atom_id"),
    ("atom_topic", "atom_id"),
    ("atom_entity", "atom_id"),
    ("atom_anchor", "atom_id"),
    ("graph_nodes", "node_id"),
    ("memory_atoms", "atom_id"),
)


def _tables_of(sql: str) -> frozenset[str]:
    """Table names mentioned by a statement, memoized on the statement text.
    语句提及的表名，按语句文本记忆化。
    """
    tables = _TABLES_OF.get(sql)
    if tables is None:
        if len(_TABLES_OF) >= _TABLES_OF_LIMIT:
            _TABLES_OF.clear()
        tables = frozenset(match.group(1) for match in _TABLE_RE.finditer(sql))
        _TABLES_OF[sql] = tables
    return tables


def _scope_where(scope: Scope | None, prefix: str = "") -> tuple[str, list[Any]]:
    """Scope.matches semantics in SQL: `col = ?` for every set key, nothing for an unset key (`1=1` when none is set).
    Scope.matches 的 SQL 形式：每个已设置键 `col = ?`，未设置键不产生谓词（全未设置时为 `1=1`）。
    """
    if scope is None:
        return "1=1", []
    parts: list[str] = []
    params: list[Any] = []
    for col in _SCOPE_COLS:
        value = getattr(scope, col)
        # An unset key matches everything, so it emits no predicate at all: the literal `(col = ? OR ? IS NULL)`
        # form is equivalent but hides the bound value from the planner, which then cannot use ix_atoms_scope.
        # 未设置的键匹配一切，因此完全不产生谓词：字面的 `(col = ? OR ? IS NULL)` 形式等价，
        # 但会把绑定值藏起来，规划器因此无法使用 ix_atoms_scope。
        if value is not None:
            parts.append(f"{prefix}{col} = ?")
            params.append(value)
    return (" AND ".join(parts) or "1=1"), params


def _placeholders(n: int) -> str:
    """`?,?,?` for n values.
    n 个值的 `?,?,?`。
    """
    return ",".join("?" for _ in range(n))


def _chunks(items: Sequence[Any], size: int = _IN_CHUNK) -> Iterator[Sequence[Any]]:
    """Yield slices of at most `size` items.
    以至多 `size` 项切片。
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _loads(text: str | None, default: Any) -> Any:
    """json.loads with a default for NULL.
    对 NULL 使用默认值的 json.loads。
    """
    if text is None:
        return default
    return json.loads(text)


def _scope_of(row: sqlite3.Row) -> Scope:
    """Scope from the three denormalized columns.
    由三列反规范化字段构造 Scope。
    """
    return Scope(user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"])


def _indexed_by(index_hint: str | None) -> str:
    """`INDEXED BY <name>` for a memory_atoms FROM clause, "" when no hint.
    memory_atoms FROM 子句的 `INDEXED BY <名称>`，无提示时为 ""。
    """
    if not index_hint:
        return ""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", index_hint):
        raise ValidationError(f"invalid index hint {index_hint!r}", "VALIDATION_000", details={"index_hint": index_hint})
    return f"INDEXED BY {index_hint}"


def _qualify(order: str) -> str:
    """Prefix bare memory_atoms column names with the table name so joins stay unambiguous.
    为裸露的 memory_atoms 列名加表前缀，避免连接时歧义。
    """
    return _QUALIFY_RE.sub(r"memory_atoms.\1", order)


def fts_match_query(tokens: Iterable[str], *, all_terms: bool = False) -> str:
    """The FTS5 MATCH expression for some tokens: every token a quoted phrase, joined by OR (AND when all_terms).
    若干词元的 FTS5 MATCH 表达式：每个词元一个带引号短语，以 OR 连接（all_terms 时以 AND 连接）。

    Input / 输入：
        tokens: Query tokens (blank ones dropped, duplicates collapsed).
        tokens：查询词元（空白的丢弃，重复的合并）。

        all_terms: Join with AND (the strict phase of keyword_candidates) instead of OR.
        all_terms：以 AND 连接（keyword_candidates 的严格阶段）而非 OR。

    Output / 输出：
        `"t1" OR "t2" ...` (or `"t1" AND "t2" ...`; "" when no token); quotes protect `.`, `/`, `-` and CJK from the
        query grammar, inner quotes are doubled.
        `"t1" OR "t2" ...`（或 `"t1" AND "t2" ...`；无词元时为 ""）；引号保护 `.`、`/`、`-` 与 CJK 不被查询语法解析，
        内部引号加倍。

    State effect / 状态影响：
        None.
        无。
    """
    phrases = []
    for token in dict.fromkeys(str(t).strip() for t in tokens):
        if token:
            # Quotes protect `.`, `/`, `-` and CJK from the query grammar; an inner quote is doubled (FTS5 rule).
            # 引号保护 `.`、`/`、`-` 与 CJK 不被查询语法解析；内部引号按 FTS5 规则加倍。
            phrases.append('"' + token.replace('"', '""') + '"')
    return (" AND " if all_terms else " OR ").join(phrases)


def _like_pattern(token: str) -> str:
    """`%token%` with the LIKE wildcards and the escape character escaped (ESCAPE '\\').
    带 LIKE 通配符与转义字符转义的 `%token%`（ESCAPE '\\'）。
    """
    escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


# The scoring columns of memory_atoms (§13.4 signals + the axis sort keys), read by axis_lite instead of `*`.
# memory_atoms 中用于打分的列（§13.4 信号 + 轴排序键），axis_lite 读取它们而非 `*`。
_LITE_COLS = (
    "memory_atoms.atom_id AS atom_id, memory_atoms.memory_id AS memory_id, memory_atoms.text AS text, "
    "memory_atoms.time_anchor AS time_anchor, memory_atoms.created_at AS created_at, "
    "memory_atoms.confidence AS confidence, memory_atoms.valid_from AS valid_from, memory_atoms.valid_to AS valid_to, "
    "memory_atoms.scale_level AS scale_level, memory_atoms.obs_type AS obs_type, memory_atoms.subject AS subject, "
    "memory_atoms.predicate AS predicate"
)
# Exact integer microseconds from the atom instant (time_anchor, else created_at) to as_of, bound as (seconds,
# microseconds) of as_of: strftime('%s') gives the floor seconds of the 32-char UTC form and its micro field sits at
# offset 21, so the value equals timedelta.total_seconds() * 10**6 bit for bit (timeutil.age_days); NULL for any
# other form, where the reader falls back to Python.
# 从原子时刻（time_anchor，否则 created_at）到 as_of 的精确整数微秒，绑定 as_of 的（秒，微秒）：strftime('%s') 给出
# 32 字符 UTC 形式的整秒，微秒字段在偏移 21，因此该值与 timedelta.total_seconds() * 10**6 逐位相同
# （timeutil.age_days）；其它形式为 NULL，读取方退回 Python 计算。
_AGE_US_SQL = (
    "CASE WHEN length(COALESCE(memory_atoms.time_anchor, memory_atoms.created_at)) = 32 THEN "
    "(? - CAST(strftime('%s', COALESCE(memory_atoms.time_anchor, memory_atoms.created_at)) AS INTEGER)) * 1000000 "
    "+ (? - CAST(substr(COALESCE(memory_atoms.time_anchor, memory_atoms.created_at), 21, 6) AS INTEGER)) "
    "ELSE NULL END AS age_us"
)
# Recall channel labels of recall_channels: the two FTS5 phases share the keyword output.
# recall_channels 的召回通道标签：两个 FTS5 阶段共用关键词输出。
_KW_STRICT = "kw_and"
_KW_LOOSE = "kw_or"


@dataclass(slots=True)
class AtomLite:
    """The scoring view of one gated atom: the signal columns, its link aggregates and its exact age — no model.
    一个门控原子的打分视图：信号列、链接聚合与精确年龄——不是模型。

    Every attribute utils.scoring reads from a MemoryAtom (text, topic_tags, entity_ids, time_anchor, created_at)
    carries the same name and value here, so the in-pool signals are bit-identical; `n_anchors` equals
    `len(atom.anchor_ids)` of the hydrated atom and `age_us` is the integer microsecond age used by recency.
    utils.scoring 从 MemoryAtom 读取的每个属性（text、topic_tags、entity_ids、time_anchor、created_at）在此同名同值，
    因此池内信号逐位相同；`n_anchors` 等于水合原子的 `len(atom.anchor_ids)`，`age_us` 是新近度使用的整数微秒年龄。
    """

    atom_id: str
    memory_id: str
    text: str
    time_anchor: str | None
    created_at: str | None
    confidence: float
    valid_from: str | None
    valid_to: str | None
    scale_level: str
    obs_type: str | None
    subject: str | None
    predicate: str | None
    entity_ids: list[str]
    topic_tags: list[str]
    n_anchors: int
    age_us: int | None


class _Rows:
    """Rows of one statement, fetched under the store lock; mimics the cursor surface readers use (§8).
    一条语句的结果行，在 store 锁内取出；模仿读取方使用的游标接口（§8）。
    """

    __slots__ = ("_rows", "_pos", "rowcount", "lastrowid")

    def __init__(self, rows: list[sqlite3.Row], rowcount: int, lastrowid: int | None) -> None:
        """Wrap the fetched rows together with the cursor's rowcount / lastrowid.
        连同游标的 rowcount / lastrowid 一起包装已取出的行。
        """
        self._rows = rows
        self._pos = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self) -> sqlite3.Row | None:
        """The next row, or None when exhausted (sqlite3.Cursor.fetchone).
        下一行；耗尽时为 None（同 sqlite3.Cursor.fetchone）。
        """
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> list[sqlite3.Row]:
        """All remaining rows (sqlite3.Cursor.fetchall).
        其余所有行（同 sqlite3.Cursor.fetchall）。
        """
        rows = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rows

    def __iter__(self) -> Iterator[sqlite3.Row]:
        """Iterate the remaining rows (sqlite3.Cursor iteration).
        迭代其余的行（同 sqlite3.Cursor 的迭代）。
        """
        row = self.fetchone()
        while row is not None:
            yield row
            row = self.fetchone()


class TableTrace:
    """Records the table names touched by every statement (used by tests to assert plane isolation).
    记录每条语句触及的表名（测试用于断言平面隔离）。
    """

    enabled: bool = True

    def __init__(self) -> None:
        """Start with an empty, enabled trace.
        以空的、启用状态的追踪开始。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.enabled = True
        self._tables: set[str] = set()

    def record(self, sql: str) -> None:
        """Add every table name that appears in a statement.
        加入语句中出现的每个表名。

        Input / 输入：
            sql: The statement text.
            sql：语句文本。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Mutates the recorded set when enabled.
            启用时修改记录集合。
        """
        if not self.enabled:
            return
        self._tables.update(_tables_of(sql))

    def reset(self) -> None:
        """Clear the recorded table set.
        清空记录的表集合。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Empties the trace.
            清空追踪。
        """
        self._tables.clear()

    def tables(self) -> set[str]:
        """The tables touched since the last reset.
        自上次重置以来触及的表。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A copy of the recorded set.
            记录集合的副本。

        State effect / 状态影响：
            None.
            无。
        """
        return set(self._tables)


class StateStore:
    """The shared seam: one connection, one re-entrant lock, typed writers and readers for every plane.
    共享接缝：单一连接、单一可重入锁、各平面的类型化写入器与读取器。
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        now: Callable[[], str] | None = None,
        id_factory: Callable[[str], str] | None = None,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "wal",
        checkpoint_on_close: bool = True,
        page_cache_kib: int = 32768,
    ) -> None:
        """Open (or create) the database, apply migrations and probe capabilities.
        打开（或创建）数据库、应用迁移并探测能力。

        Input / 输入：
            path: SQLite path or ':memory:'.
            path：SQLite 路径或 ':memory:'。

            now: Injected clock returning fixed-width UTC ISO; defaults to timeutil.now_iso.
            now：注入的时钟，返回定宽 UTC ISO；默认 timeutil.now_iso。

            id_factory: Injected id maker `prefix -> id`; defaults to `prefix_<12 hex>`.
            id_factory：注入的 id 生成器 `prefix -> id`；默认 `prefix_<12 hex>`。

            busy_timeout_ms: PRAGMA busy_timeout.
            busy_timeout_ms：PRAGMA busy_timeout。

            journal_mode: 'wal' (file databases only) or 'delete'.
            journal_mode：'wal'（仅文件库）或 'delete'。

            checkpoint_on_close: File databases in WAL mode: True (default) lets SQLite checkpoint and remove the WAL on close(); False keeps the WAL on disk (still durable, recovered on the next open) when the runtime offers SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python >= 3.12), else it is ignored. Memory databases have no WAL.
            checkpoint_on_close：WAL 模式的文件库：True（默认）让 SQLite 在 close() 时做 checkpoint 并删除 WAL；False 在运行时提供 SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE（Python >= 3.12）时保留磁盘上的 WAL（仍然耐久，下次打开时恢复），否则被忽略。内存库没有 WAL。

            page_cache_kib: PRAGMA cache_size in KiB (SQLite's own default is 2048): the pool's primary-key reads land on pages scattered over the whole file, so a 30000-atom file (~100 MB) spends ~1 ms per search re-reading pages under the 2 MB default (RETRIEVAL_BENCH §3); 0 keeps SQLite's default.
            page_cache_kib：PRAGMA cache_size，单位 KiB（SQLite 自身默认 2048）：池的主键读取落在整个文件中零散的页上，30000 原子的文件（约 100 MB）在 2 MB 默认值下每次搜索要花约 1 ms 重读页（RETRIEVAL_BENCH §3）；0 保持 SQLite 默认。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            Creates the schema when missing (a fresh private ':memory:' or file database clones the process-wide schema template through the backup API before any WAL switch; an existing file runs the migration ladder) and writes meta.schema_created_at once.
            缺失时创建模式（全新的私有 ':memory:' 或文件库在切换 WAL 之前经 backup API 克隆进程级模式模板；已有文件执行迁移阶梯），并一次性写入 meta.schema_created_at。
        """
        self.path = path
        self._now = now
        self._id_factory = id_factory
        self.trace = TableTrace()
        self._lock = threading.RLock()
        self._depth = 0
        self._conn: sqlite3.Connection | None = None
        # Anchor ids proven to exist during the current outermost transaction (anchors are never deleted, so a
        # witness verified once stays verified until COMMIT/ROLLBACK clears the memo); per instance, never shared.
        # 当前最外层事务期间已证明存在的锚点 id（锚点从不删除，因此一次验证过的见证在 COMMIT/ROLLBACK
        # 清空备忘前始终有效）；按实例持有，从不共享。
        self._verified_anchors: set[str] = set()
        # The thread that opened the outermost transaction: only it may read the memo (other threads block on the
        # lock and must run their own pre-check after the transaction ends, exactly as before).
        # 打开最外层事务的线程：只有它能读备忘（其它线程阻塞在锁上，须在事务结束后自行预检，与从前完全一致）。
        self._txn_owner: int | None = None
        # Attribute listeners (§10.3): `callback(atom_id, attrs)` for every committed change of an atom's validity or
        # archived state (archive_atom / restore_atom / set_valid_to). HMS registers index.update_attrs so a provider
        # that keeps scalars (Milvus) follows SQLite, which stays the only source of truth. `attrs` is the FULL
        # atom_attrs snapshot read inside the transaction; calls are dispatched after COMMIT (a ROLLBACK discards
        # them), outside the store lock, in commit order.
        # 属性监听器（§10.3）：原子有效期或归档状态每次**已提交**的变化（archive_atom / restore_atom / set_valid_to）
        # 都会回调 `callback(atom_id, attrs)`。HMS 注册 index.update_attrs，使保存标量的提供者（Milvus）跟随 SQLite——
        # 它仍是唯一真源。`attrs` 是事务内读取的**完整** atom_attrs 快照；COMMIT 后在 store 锁之外按提交顺序派发
        # （ROLLBACK 则丢弃）。
        self.attr_listeners: list[Callable[[str, dict], None]] = []
        # Atom ids whose validity / archived state changed in the open transaction; the FULL scalar snapshot
        # (atom_attrs) is read just before COMMIT so listeners receive the committed truth, never a partial diff.
        # 打开的事务中有效期 / 归档状态发生变化的原子 id；COMMIT 前读取**完整**标量快照（atom_attrs），
        # 使监听器收到的是已提交的真值，而不是部分差异。
        self._pending_attrs: list[str] = []
        # Dispatch tickets: every commit that carries changes takes the next ticket under the store lock and the
        # listeners are called in ticket order (a Condition outside the store lock), so two commits on one atom from
        # two threads reach a provider in commit order — a later snapshot can never be overwritten by an earlier one.
        # 派发票号：每个携带变化的提交在 store 锁下领取下一个票号，监听器按票号顺序被调用（store 锁之外的 Condition），
        # 因此两个线程对同一原子的两次提交按提交顺序到达提供者——较晚的快照绝不会被较早的覆盖。
        self._dispatch_cond = threading.Condition()
        self._dispatch_ticket = 0
        self._dispatch_next = 0
        self._dispatch_done: set[int] = set()
        # Write generations (§8): data_generation counts every write statement of this connection touching
        # memory_atoms / atom_vectors; rebuild_generation counts the subset that can remove or rewrite rows
        # (DELETE / DROP / ROLLBACK / reset), after which a cached view must be rebuilt rather than extended.
        # Together with data_version() (other connections' commits) they key the SQLite index provider's cache.
        # 写入代数（§8）：data_generation 统计本连接触及 memory_atoms / atom_vectors 的每条写语句；
        # rebuild_generation 统计其中可能删除或改写行的子集（DELETE / DROP / ROLLBACK / reset），此后缓存视图必须重建
        # 而不能增量扩展。它们与 data_version()（其它连接的提交）一起作为 SQLite 索引提供者缓存的键。
        self.data_generation = 0
        self.rebuild_generation = 0
        # Per-scope write versions for HMS.context()'s session-card memo (§6.7): scope_write_versions maps
        # Scope.key() → (scope, n). A transaction attributes itself to a scope with touch_scope (insert_source_event,
        # insert_atom, the scheduler phases writing scope-less rows on behalf of a target scope); at the COMMIT of the
        # outermost transaction n advances for every attributed scope IF the transaction wrote versioned rows
        # (_txn_wrote: a write statement whose rowcount != 0 outside unversioned()), and write_generation advances
        # instead when versioned rows were written under no scope (edges, transitions, meta … with no scoped caller);
        # an autocommit versioned write, every rollback and reset advance write_generation as well. A transaction
        # that changed no row (an ignored INSERT OR IGNORE, an UPDATE of nothing, an idle scheduler phase) or only
        # unversioned rows (an internal event and its anchor, the checkpoints row, the scheduler's meta rows and
        # compiled-state archives — nothing a scope reads through HMS.context() depends on them) moves no version.
        # scope_version() folds both with data_version() (other connections' commits) into one comparable tuple;
        # nothing here is timer-based.
        # 供 HMS.context() 会话卡片记忆使用的按范围写入版本（§6.7）：scope_write_versions 把 Scope.key() 映射为
        # (scope, n)。事务通过 touch_scope 把自己归于某个范围（insert_source_event、insert_atom、代表目标范围写入
        # 无范围行的调度器各阶段）；最外层事务 COMMIT 时，**若**该事务写入了带版本的行（_txn_wrote：unversioned()
        # 之外、rowcount != 0 的写语句），每个被归属范围的 n 推进一次；若带版本的行是在没有任何范围下写入的（边、转移、
        # meta……且没有范围内调用方），则改为推进 write_generation；自动提交的带版本写入、每次回滚与 reset 同样推进
        # write_generation。没有改变任何行（被忽略的 INSERT OR IGNORE、未命中的 UPDATE、空转的调度器阶段）或只写了
        # 不带版本的行（内部事件及其锚点、checkpoints 行、调度器的 meta 行与编译态归档——范围经 HMS.context() 读到的
        # 内容都不依赖它们）的事务不推进任何版本。scope_version() 把两者与 data_version()（其它连接的提交）折成一个
        # 可比较的元组；这里没有任何基于计时器的逻辑。
        self.write_generation = 0
        self.scope_write_versions: dict[str, tuple[Scope, int]] = {}
        self._txn_scope_keys: dict[str, Scope] = {}
        self._txn_wrote = False
        # Thread-local depth of unversioned() blocks (a block on one thread must not hide another thread's writes).
        # unversioned() 块的线程局部深度（一个线程的块不能遮蔽另一个线程的写入）。
        self._unversioned = threading.local()
        is_private_memory = path in ("", ":memory:")
        is_memory = is_private_memory or path.startswith("file:")
        self._is_file = not is_memory
        self._checkpoint_on_close = bool(checkpoint_on_close)
        try:
            if not is_memory:
                parent = os.path.dirname(os.path.abspath(path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            if int(page_cache_kib) > 0:
                conn.execute(f"PRAGMA cache_size = -{int(page_cache_kib)}")
        except (sqlite3.Error, OSError) as exc:
            raise StoreError(f"cannot open database {path!r}: {exc}", "STORE_001", details={"path": path}) from exc
        self._conn = conn
        # The schema arrives before the WAL switch: a fresh database (private ':memory:' or a new file) clones the
        # per-process template through the backup API, so the 97 schema pages go straight into the file instead of
        # through the WAL once and the close-time checkpoint a second time; an existing file (and a `file:` URI,
        # which may be a shared-cache database with other connections) runs the migration ladder.
        # 模式在切换 WAL 之前到位：全新数据库（私有 ':memory:' 或新文件）经 backup API 克隆进程级模板，因此 97 个
        # 模式页直接写入文件，而不是先经 WAL 写一次、再在关闭时的 checkpoint 中写第二次；已有文件（以及可能是带
        # 其它连接的共享缓存库的 `file:` URI）执行迁移阶梯。
        if is_private_memory or not is_memory:
            clone_schema(conn, created_at=self.now())
        else:
            apply_migrations(conn, created_at=self.now())
        if not is_memory and str(journal_mode).lower() == "wal":
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error as exc:
                raise StoreError(f"cannot open database {path!r}: {exc}", "STORE_001", details={"path": path}) from exc
        self.capabilities: dict = detect_capabilities(conn)
        if self.get_meta("schema_created_at") is None:
            self.set_meta("schema_created_at", self.now())
        # A purge flag can only outlive its transaction if the file was edited by hand: clear it so the BEFORE DELETE
        # triggers are armed again before any statement of this connection runs.
        # 清理标志只有在文件被手工改过时才可能残留：先清掉它，让 BEFORE DELETE 触发器在本连接执行任何语句前重新武装。
        if self.get_meta(PURGE_FLAG_KEY) is not None:
            logger.warning("store %r opened with a stale %s flag; clearing it", path, PURGE_FLAG_KEY)
            with self.transaction():
                self._execute("DELETE FROM meta WHERE key = ?", (PURGE_FLAG_KEY,))

    # ------------------------------------------------------------------ infrastructure
    # ------------------------------------------------------------------ 基础设施
    def now(self) -> str:
        """The library clock: injected `now` or timeutil.now_iso.
        库时钟：注入的 `now` 或 timeutil.now_iso。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Fixed-width UTC ISO instant.
            定宽 UTC ISO 时刻。

        State effect / 状态影响：
            None.
            无。
        """
        return to_utc_iso(self._now()) if self._now is not None else now_iso()

    def new_id(self, prefix: str) -> str:
        """Mint an object id for a prefix.
        为前缀生成对象 id。

        Input / 输入：
            prefix: One of evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst cst.
            prefix：evt anc obs ep atm mem ent edg iep fb ckp trn pkg hst cst 之一。

        Output / 输出：
            `f"{prefix}_{secrets.token_hex(6)}"` or the injected factory's value.
            `f"{prefix}_{secrets.token_hex(6)}"` 或注入工厂的返回值。

        State effect / 状态影响：
            None (the injected factory may advance its counter).
            无（注入工厂可能推进计数器）。
        """
        if self._id_factory is not None:
            return self._id_factory(prefix)
        return f"{prefix}_{secrets.token_hex(6)}"

    def next_seq(self, key: str = "history_seq") -> int:
        """Increment and return a meta counter inside the current transaction.
        在当前事务内递增并返回 meta 计数器。

        Input / 输入：
            key: Meta key of the counter.
            key：计数器的 meta 键。

        Output / 输出：
            The new counter value (1 on first use).
            新的计数值（首次为 1）。

        State effect / 状态影响：
            Writes meta.
            写入 meta。
        """
        with self.transaction():
            row = self._execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            value = int(row["value"]) + 1 if row else 1
            self._execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))
            return value

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Re-entrant transaction: the outermost level does BEGIN IMMEDIATE / COMMIT / ROLLBACK under the lock.
        可重入事务：最外层在锁下执行 BEGIN IMMEDIATE / COMMIT / ROLLBACK。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A context manager yielding None.
            产出 None 的上下文管理器。

        State effect / 状态影响：
            Commits on normal exit, rolls back on exception (outermost level only).
            正常退出时提交，异常时回滚（仅最外层）。
        """
        committed: list[tuple[str, dict]] = []
        ticket: int | None = None
        self._lock.acquire()
        try:
            if self._depth == 0:
                self._txn_owner = threading.get_ident()
                self._raw("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    self._rollback()
                raise
            self._depth -= 1
            if self._depth == 0:
                try:
                    if self._pending_attrs:
                        # The snapshot is read INSIDE the transaction: it is exactly what COMMIT makes true.
                        # 快照在事务**内部**读取：它恰好就是 COMMIT 使之成立的内容。
                        committed = self._attr_snapshot(self._pending_attrs)
                    self._raw("COMMIT")
                except StoreError:
                    self._rollback()
                    raise
                finally:
                    # The witness memo is scoped to one outermost transaction.
                    # 见证备忘只在一个最外层事务内有效。
                    self._verified_anchors.clear()
                self._pending_attrs = []
                self._commit_versions()
                if committed:
                    # The ticket is taken under the store lock, so ticket order is commit order.
                    # 票号在 store 锁下领取，因此票号顺序就是提交顺序。
                    ticket = self._dispatch_ticket
                    self._dispatch_ticket += 1
        finally:
            self._lock.release()
            if ticket is not None:
                # Only committed attribute changes reach the listeners, after the lock is released, in commit order.
                # 只有已提交的属性变化才到达监听器，在释放锁之后，按提交顺序。
                self._dispatch_attrs(committed, ticket)

    @property
    def lock(self) -> threading.RLock:
        """The store's re-entrant lock, FIRST in the lock order (index providers take it before their own lock).
        存储的可重入锁，锁序中的第一把（索引提供者先取它再取自身的锁）。

        A component holding its own lock must never wait for this one; see SQLiteIndex.search.
        持有自身锁的组件绝不能等待它；见 SQLiteIndex.search。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The RLock.
            RLock。

        State effect / 状态影响：
            None.
            无。
        """
        return self._lock

    @property
    def in_transaction(self) -> bool:
        """True while inside transaction().
        处于 transaction() 内时为真。

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
        return self._depth > 0

    def reset(self) -> None:
        """Drop every table and re-create the schema (meta cleared).
        删除全部表并重建模式（meta 清空）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Destroys all data.
            销毁全部数据。
        """
        with self._lock:
            if self._depth:
                raise StoreError("reset() cannot run inside a transaction", "STORE_001")
            conn = self._connection()
            drop_all(conn)
            apply_migrations(conn)
            self.data_generation += 1
            self.rebuild_generation += 1
            self.scope_write_versions.clear()
            self.write_generation += 1
            self.capabilities = detect_capabilities(conn)
            self.set_meta("schema_created_at", self.now())

    def close(self) -> None:
        """Close the connection (idempotent).
        关闭连接（幂等）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Releases the SQLite handle; a file database checkpoints its WAL first unless checkpoint_on_close=False.
            释放 SQLite 句柄；文件库先 checkpoint 其 WAL，除非 checkpoint_on_close=False。
        """
        with self._lock:
            if self._conn is not None:
                try:
                    if self._is_file and not self._checkpoint_on_close:
                        # Best effort: the option exists in Python >= 3.12; older runtimes keep SQLite's default.
                        # 尽力而为：该选项自 Python 3.12 起可用；更老的运行时保持 SQLite 的默认行为。
                        option = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
                        setconfig = getattr(self._conn, "setconfig", None)
                        if option is not None and setconfig is not None:
                            try:
                                setconfig(option, True)
                            except sqlite3.Error:  # pragma: no cover - the checkpoint then simply happens
                                pass
                    self._conn.close()
                finally:
                    self._conn = None

    def stats(self, scope: Scope | None = None) -> dict[str, int]:
        """Row counts per table (scope-filtered where the table carries scope columns).
        每张表的行数（带范围列的表按范围过滤）。

        Input / 输入：
            scope: Optional scope filter.
            scope：可选的范围过滤。

        Output / 输出：
            {table: count} for all 22 tables.
            全部 22 张表的 {table: count}。

        State effect / 状态影响：
            None.
            无。
        """
        out: dict[str, int] = {}
        for table in TABLES:
            if scope is not None and table in SCOPED_TABLES:
                where, params = _scope_where(scope)
                row = self._execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params).fetchone()
            else:
                row = self._execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"])
        return out

    # ------------------------------------------------------------------ private plumbing
    # ------------------------------------------------------------------ 私有管道
    def _connection(self) -> sqlite3.Connection:
        """The open connection or STORE_001 when closed.
        已打开的连接；关闭后抛出 STORE_001。
        """
        if self._conn is None:
            raise StoreError("store is closed", "STORE_001", details={"path": self.path})
        return self._conn

    def data_version(self) -> int:
        """SQLite's `PRAGMA data_version`: changes whenever ANOTHER connection commits to this database file.
        SQLite 的 `PRAGMA data_version`：每当**其它**连接向本数据库文件提交时改变。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The integer data version (own-connection writes do not change it; see data_generation for those).
            整数数据版本（本连接的写入不改变它；本连接的写入见 data_generation）。

        State effect / 状态影响：
            None (not traced).
            无（不追踪）。
        """
        with self._lock:
            try:
                return int(self._connection().execute("PRAGMA data_version").fetchone()[0])
            except sqlite3.Error as exc:
                raise StoreError(f"PRAGMA data_version failed: {exc}", "STORE_001") from exc

    def touch_scope(self, scope: Scope) -> None:
        """Attribute the open transaction's rows to `scope` — a source event, an atom, or scope-less rows a scoped caller (the scheduler phases) writes on its behalf — so that its version, instead of the global write_generation, advances at COMMIT if the transaction wrote versioned rows (a transaction that changed no row, or only unversioned() rows, moves nothing); outside a transaction the version advances at once.
        把打开事务的行归于 `scope`——源事件、原子，或由范围内调用方（调度器各阶段）代表它写入的无范围行——使其版本（而非全局 write_generation）在 COMMIT 时推进，前提是该事务写入了带版本的行（没有改变任何行、或只写了 unversioned() 行的事务不推进任何东西）；不在事务内时立即推进。

        Input / 输入：
            scope: The written row's scope (SourceEvent.scope / MemoryAtom.scope, or the scheduler target).
            scope：被写入行的范围（SourceEvent.scope / MemoryAtom.scope，或调度器的目标范围）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            In-process counters only (scope_write_versions / _txn_scope_keys); nothing is written to SQLite.
            仅进程内计数（scope_write_versions / _txn_scope_keys）；不写 SQLite。
        """
        with self._lock:
            if self._depth > 0:
                self._txn_scope_keys[scope.key()] = scope
            else:
                self._bump_scope(scope)

    @contextmanager
    def unversioned(self) -> Iterator[None]:
        """Write statements inside the block move no write version (§6.7): for rows nothing a scope reads through HMS.context() depends on — an internal event and its anchor (system_event), the checkpoints row, the scheduler's last_consolidation / work_state meta rows and compiled-state archives — so a transaction made only of them leaves every session-card memo reused.
        块内的写语句不推进任何写入版本（§6.7）：用于范围经 HMS.context() 读到的内容都不依赖的行——内部事件及其锚点（system_event）、checkpoints 行、调度器的 last_consolidation / work_state meta 行与编译态归档——因此只由这些行组成的事务让每个会话卡片记忆保持复用。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A re-entrant, thread-local context manager yielding None.
            产出 None 的可重入、线程局部的上下文管理器。

        State effect / 状态影响：
            Only the version accounting changes: tracing, data_generation / rebuild_generation and the transaction itself are unaffected, and a touch_scope() inside the block still records its attribution (which bumps only if the transaction also wrote versioned rows). Never wrap a statement that changes what compile() can return.
            只改变版本记账：追踪、data_generation / rebuild_generation 与事务本身不受影响，块内的 touch_scope() 仍记录归属（只有事务同时写了带版本的行才推进）。绝不能包住会改变 compile() 可能返回内容的语句。
        """
        self._unversioned.depth = self._unversioned_depth() + 1
        try:
            yield
        finally:
            self._unversioned.depth = self._unversioned_depth() - 1

    def _unversioned_depth(self) -> int:
        """Depth of unversioned() blocks on the calling thread (0 outside any).
        调用线程上 unversioned() 块的深度（不在任何块内时为 0）。
        """
        return int(getattr(self._unversioned, "depth", 0))

    def scope_version(self, scope: Scope) -> tuple:
        """The comparable write version of everything a memo for `scope` depends on: (write_generation, data_version(), ((key, n) …)) over every recorded scope that covers or is covered by `scope`.
        某范围的记忆所依赖的一切的可比较写入版本：(write_generation, data_version(), ((key, n) …))，覆盖 `scope` 或被 `scope` 覆盖的每个已记录范围都计入。

        Input / 输入：
            scope: The memo's (full or partial) scope.
            scope：记忆的（完整或部分）范围。

        Output / 输出：
            A tuple that changes whenever a committed write could have changed what the scope reads (writes to a
            parent scope such as delete_all(user_id=u) or rollback of a partial-scope checkpoint count too).
            每当某次已提交写入可能改变该范围读到的内容时就会改变的元组（对父范围的写入——如 delete_all(user_id=u) 或
            对部分范围检查点的回滚——同样计入）。

        State effect / 状态影响：
            None (one PRAGMA data_version; not traced).
            无（一次 PRAGMA data_version；不追踪）。
        """
        with self._lock:
            related = tuple(
                sorted(
                    (key, n)
                    for key, (stored, n) in self.scope_write_versions.items()
                    if scope.matches(stored) or stored.matches(scope)
                )
            )
            return (self.write_generation, self.data_version(), related)

    def _bump_scope(self, scope: Scope) -> None:
        """Advance one scope's write version (caller holds the lock).
        推进一个范围的写入版本（调用方持锁）。
        """
        key = scope.key()
        n = self.scope_write_versions.get(key, (scope, 0))[1]
        self.scope_write_versions[key] = (scope, n + 1)

    def _commit_versions(self) -> None:
        """After COMMIT of the outermost transaction: when versioned rows were written, bump every attributed scope, or write_generation when none was attributed; a transaction that changed no row or only unversioned rows moves nothing (caller holds the lock).
        最外层事务 COMMIT 之后：若写入了带版本的行，推进每个被归属的范围，没有任何归属时推进 write_generation；没有改变任何行或只写了不带版本行的事务不推进任何东西（调用方持锁）。
        """
        if self._txn_wrote:
            for scope in self._txn_scope_keys.values():
                self._bump_scope(scope)
            if not self._txn_scope_keys:
                self.write_generation += 1
        self._txn_scope_keys = {}
        self._txn_wrote = False

    def _raw(self, sql: str) -> None:
        """Execute a transaction-control statement (not traced).
        执行事务控制语句（不追踪）。
        """
        try:
            self._connection().execute(sql)
        except sqlite3.Error as exc:
            raise StoreError(f"{sql} failed: {exc}", "STORE_001") from exc

    def _rollback(self) -> None:
        """Roll back if a transaction is open; never raises.
        若有事务则回滚；从不抛出。
        """
        self._verified_anchors.clear()
        self._pending_attrs.clear()
        # A rolled-back statement may already have bumped data_generation and been cached: force a rebuild.
        # 被回滚的语句可能已经推进了 data_generation 并被缓存：强制重建。
        self.data_generation += 1
        self.rebuild_generation += 1
        # Whatever the rolled-back transaction touched is unknown to a memo captured meanwhile: invalidate globally.
        # 被回滚的事务触及过什么，期间捕获的记忆无从得知：全局失效。
        self._txn_scope_keys = {}
        self._txn_wrote = False
        self.write_generation += 1
        conn = self._conn
        if conn is None:
            return
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - nothing to roll back
            pass

    def _queue_attrs(self, atom_id: str) -> None:
        """Remember that an atom's validity / archived state changed; its full snapshot is dispatched after COMMIT.
        记住某原子的有效期 / 归档状态发生了变化；其完整快照在 COMMIT 后派发。
        """
        if self.attr_listeners:
            self._pending_attrs.append(str(atom_id))

    def _attr_snapshot(self, atom_ids: Sequence[str]) -> list[tuple[str, dict]]:
        """The full INDEX_ATTR_KEYS snapshot of the changed atoms as the transaction leaves them (purged ids drop out).
        变化原子在事务结束时的完整 INDEX_ATTR_KEYS 快照（已被清理的 id 不出现）。
        """
        wanted = list(dict.fromkeys(atom_ids))
        attrs = self.atom_attrs(wanted)
        return [(atom_id, attrs[atom_id]) for atom_id in wanted if atom_id in attrs]

    def _dispatch_attrs(self, changes: list[tuple[str, dict]], ticket: int) -> None:
        """Call every listener for every committed change in commit (ticket) order; failures are logged, never raised.
        按提交（票号）顺序对每次已提交变化调用每个监听器；失败只记录日志、从不抛出。

        SQLite is the source of truth and a stale provider scalar can only hide a row, never surface a wrong one; the
        ticket is released even when a listener fails so later commits are never blocked.
        SQLite 是真源，提供者标量过期只会漏检、不会误检；即使监听器失败也释放票号，后续提交绝不会被阻塞。
        """
        try:
            with self._dispatch_cond:
                while self._dispatch_next != ticket:
                    self._dispatch_cond.wait()
            for listener in list(self.attr_listeners):
                for atom_id, attrs in changes:
                    try:
                        listener(atom_id, dict(attrs))
                    except Exception:  # noqa: BLE001 - the write is already committed
                        logger.warning(
                            "index attribute listener %r failed for %s %s", listener, atom_id, attrs, exc_info=True
                        )
        finally:
            # The ticket is retired even when this thread was interrupted while still waiting (a ticket ahead of
            # the sequence is parked in _dispatch_done and consumed when the sequence reaches it).
            # 即使本线程在等待时被中断，票号也会被注销（排在序列前面的票号先停在 _dispatch_done 里，序列到达时再消耗）。
            with self._dispatch_cond:
                self._dispatch_done.add(ticket)
                while self._dispatch_next in self._dispatch_done:
                    self._dispatch_done.discard(self._dispatch_next)
                    self._dispatch_next += 1
                self._dispatch_cond.notify_all()

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> _Rows:
        """Execute one statement and fetch its rows under the lock, with tracing and error mapping (the only cursor access).
        在锁内执行一条语句并取出其结果行，同时追踪、映射错误（唯一的游标访问）。
        """
        self.trace.record(sql)
        head = sql.lstrip()[:7].upper()
        is_write = not head.startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN"))
        if is_write and _tables_of(sql) & _CACHE_TABLES:
            # Every write to the cached tables advances data_generation; anything but an INSERT / UPDATE (DELETE,
            # DROP, ...) can remove rows and therefore advances rebuild_generation as well.
            # 对缓存表的每次写入推进 data_generation；除 INSERT / UPDATE 之外的语句（DELETE、DROP…）可能删除行，
            # 因此同时推进 rebuild_generation。
            self.data_generation += 1
            if not head.startswith(("INSERT", "UPDATE")):
                self.rebuild_generation += 1
        with self._lock:
            try:
                cursor = self._connection().execute(sql, tuple(params))
                # Fetch before releasing the lock: a live cursor read outside it races with any other thread running
                # the same SQL text, because CPython's per-connection statement cache re-binds and resets the shared
                # prepared statement underneath it (rows of the wrong statement, IndexError, ValidationError).
                # 在释放锁之前取完所有行：在锁外读取活动游标会与其它执行相同 SQL 文本的线程竞争，
                # 因为 CPython 的每连接语句缓存会在其下重新绑定并重置共享的预编译语句
                # （拿到别的语句的行、IndexError、ValidationError）。
                try:
                    rows = _Rows(cursor.fetchall(), cursor.rowcount, cursor.lastrowid)
                finally:
                    cursor.close()
                if is_write and rows.rowcount != 0 and not self._unversioned_depth():
                    # A versioned write statement that changed rows (rowcount -1 = DDL, counted too; an ignored
                    # INSERT OR IGNORE or an UPDATE of no row is 0; a statement inside unversioned() never counts):
                    # inside a transaction it is accounted at COMMIT (_commit_versions), an autocommit write counts now.
                    # 改变了行的带版本写语句（rowcount -1 = DDL，同样计数；被忽略的 INSERT OR IGNORE 或未命中行的 UPDATE 为 0；
                    # unversioned() 内的语句从不计数）：事务内在 COMMIT 时记账（_commit_versions），自动提交的写入立即计数。
                    if self._depth > 0:
                        self._txn_wrote = True
                    else:
                        self.write_generation += 1
                return rows
            except sqlite3.IntegrityError as exc:
                message = str(exc)
                if message.startswith("HMS_INVARIANT"):
                    raise StoreError(message, "STORE_003", details={"trigger": message}) from exc
                match = _NOT_NULL_RE.match(message)
                if match and match.group(2).endswith("anchor_id"):
                    raise WitnessMissingError(
                        message, details={"object": match.group(1), "id": None, "missing": []}
                    ) from exc
                raise StoreError(message, "STORE_001", details={"sql": sql[:160]}) from exc
            except sqlite3.Error as exc:
                raise StoreError(str(exc), "STORE_001", details={"sql": sql[:160]}) from exc

    def _require_anchors(self, ids: Sequence[str] | None, *, obj: str, obj_id: str | None) -> list[str]:
        """Witness pre-check (§3): reject an empty list, then every unknown id, before any write SQL.
        见证预检（§3）：在任何写 SQL 之前拒绝空列表，再拒绝未知 id。
        """
        ids = [str(i) for i in (ids or [])]
        if not ids:
            raise WitnessMissingError(
                f"{obj} {obj_id or ''} references no anchor".strip(),
                details={"object": obj, "id": obj_id or None, "missing": []},
            )
        wanted = list(dict.fromkeys(ids))
        # Inside a transaction (on its owning thread), ids proven earlier in it need no second SELECT.
        # 在事务内（且在其拥有线程上），同一事务中早先已证明的 id 无需第二次 SELECT。
        owner = self._depth > 0 and self._txn_owner == threading.get_ident()
        memo = self._verified_anchors if owner else None
        unknown = [i for i in wanted if memo is None or i not in memo]
        found = {a.anchor_id for a in self.get_anchors(unknown)} if unknown else set()
        missing = [i for i in unknown if i not in found]
        if missing:
            raise WitnessMissingError(
                f"{obj} {obj_id or ''} references unknown anchors {missing}".strip(),
                details={"object": obj, "id": obj_id or None, "missing": missing},
            )
        if memo is not None:
            memo.update(found)
        return ids

    def _live(self, as_of: str | None) -> tuple[str, list[str]]:
        """LIVE_SQL with its two params for `as_of` (defaults to now()).
        带两个参数的 LIVE_SQL（as_of 默认 now()）。
        """
        instant = to_utc_iso(as_of) if as_of else self.now()
        return LIVE_SQL, [instant, instant]

    # ------------------------------------------------------------------ events / anchors
    # ------------------------------------------------------------------ 事件 / 锚点
    def _event_from_row(self, row: sqlite3.Row) -> SourceEvent:
        """SourceEvent from a source_events row.
        由 source_events 行构造 SourceEvent。
        """
        return SourceEvent(
            event_id=row["event_id"],
            source_system=row["source_system"],
            object_type=row["object_type"],
            event_type=row["event_type"],
            actor_id=row["actor_id"],
            role=row["role"],
            timestamp=row["timestamp"],
            raw_ref=row["raw_ref"],
            visibility=row["visibility"],
            scope=_scope_of(row),
            payload=_loads(row["payload_json"], {}),
            payload_hash=row["payload_hash"],
            user_meta=_loads(row["user_meta_json"], {}),
        )

    def _anchor_from_row(self, row: sqlite3.Row) -> Anchor:
        """Anchor from an anchors row.
        由 anchors 行构造 Anchor。
        """
        return Anchor(
            anchor_id=row["anchor_id"],
            source_event_id=row["source_event_id"],
            hash=row["hash"],
            source_ref=row["source_ref"],
            timestamp=row["timestamp"],
            ingested_at=row["ingested_at"],
            permission_scope=row["permission_scope"],
            scope=_scope_of(row),
            revoked_at=row["revoked_at"],
        )

    def insert_source_event(self, ev: SourceEvent) -> tuple[SourceEvent, bool]:
        """Append a source event (INSERT OR IGNORE on the One Event, One Anchor identity).
        追加源事件（在“一事件一锚点”身份上 INSERT OR IGNORE）。

        Input / 输入：
            ev: Event; an empty event_id is assigned, ingested_at = now().
            ev：事件；event_id 为空时分配，ingested_at = now()。

        Output / 输出：
            (event, dedup) — the stored event and True when an identical row already existed.
            (event, dedup)——已存储的事件，以及已存在相同行时为 True。

        State effect / 状态影响：
            Writes source_events.
            写入 source_events。
        """
        event_id = ev.event_id or self.new_id("evt")
        stored = ev if ev.event_id == event_id else ev.model_copy(update={"event_id": event_id})
        scope_key = ev.scope.key()
        with self.transaction():
            cur = self._execute(
                "INSERT OR IGNORE INTO source_events(event_id, source_system, object_type, event_type, actor_id, role, "
                "timestamp, ingested_at, raw_ref, visibility, user_id, agent_id, run_id, scope_key, payload_json, "
                "payload_hash, user_meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    ev.source_system,
                    ev.object_type,
                    ev.event_type,
                    ev.actor_id,
                    ev.role,
                    ev.timestamp,
                    self.now(),
                    ev.raw_ref,
                    ev.visibility,
                    ev.scope.user_id,
                    ev.scope.agent_id,
                    ev.scope.run_id,
                    scope_key,
                    canonical_json(ev.payload),
                    ev.payload_hash,
                    canonical_json(ev.user_meta),
                ),
            )
            if cur.rowcount == 1:
                # Only a stored row attributes the transaction to the scope; a replayed identical event changes
                # nothing. An internal event written by system_event is an unversioned row: the attribution alone
                # moves no version — the witnessed operation's own row changes do.
                # 只有真正入库的行才把事务归于该范围；重放的相同事件不改变任何东西。system_event 写入的内部事件是不带版本
                # 的行：仅有归属不推进任何版本——被见证操作自身改变的行才推进。
                self.touch_scope(ev.scope)
                return stored, False
            row = self._execute(
                "SELECT * FROM source_events WHERE payload_hash = ? AND timestamp = ? AND scope_key = ?",
                (ev.payload_hash, ev.timestamp, scope_key),
            ).fetchone()
            if row is None:
                row = self._execute("SELECT * FROM source_events WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:  # pragma: no cover - IGNORE without a conflicting row is impossible
                raise StoreError("source_events insert was ignored without a conflicting row", "STORE_001")
            return self._event_from_row(row), True

    def get_source_event(self, event_id: str) -> SourceEvent | None:
        """Fetch one event by id.
        按 id 读取一个事件。

        Input / 输入：
            event_id: Event id.
            event_id：事件 id。

        Output / 输出：
            SourceEvent or None.
            SourceEvent 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM source_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._event_from_row(row) if row else None

    def list_events(
        self,
        scope: Scope,
        *,
        object_types: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        order: str = "ASC",
    ) -> list[SourceEvent]:
        """List events in a scope by event time.
        按事件时间列出某范围内的事件。

        Input / 输入：
            scope: Scope filter (Scope.matches semantics).
            scope：范围过滤（Scope.matches 语义）。

            object_types: Optional object_type whitelist.
            object_types：可选的 object_type 白名单。

            since: Inclusive lower bound on timestamp.
            since：timestamp 的下界（含）。

            until: Inclusive upper bound on timestamp.
            until：timestamp 的上界（含）。

            limit: Optional row cap.
            limit：可选的行数上限。

            order: 'ASC' or 'DESC' on (timestamp, event_id).
            order：按 (timestamp, event_id) 的 'ASC' 或 'DESC'。

        Output / 输出：
            Events in the requested order.
            按请求顺序的事件。

        State effect / 状态影响：
            None.
            无。
        """
        direction = str(order).upper()
        if direction not in ("ASC", "DESC"):
            raise ValidationError(
                f"order must be ASC or DESC, got {order!r}", "VALIDATION_013", details={"order": order}
            )
        where, params = _scope_where(scope)
        clauses = [where]
        if object_types:
            clauses.append(f"object_type IN ({_placeholders(len(object_types))})")
            params.extend(object_types)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(to_utc_iso(since))
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(to_utc_iso(until))
        sql = f"SELECT * FROM source_events WHERE {' AND '.join(clauses)} ORDER BY timestamp {direction}, event_id {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [self._event_from_row(r) for r in self._execute(sql, params).fetchall()]

    def last_events(self, scope: Scope, *, object_types: list[str] | None = None, limit: int = 10) -> list[SourceEvent]:
        """The newest `limit` events of a scope, returned chronologically.
        某范围最新的 `limit` 个事件，按时间顺序返回。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            object_types: Optional object_type whitelist.
            object_types：可选的 object_type 白名单。

            limit: How many newest events to keep.
            limit：保留多少个最新事件。

        Output / 输出：
            Events oldest → newest.
            由旧到新的事件。

        State effect / 状态影响：
            None.
            无。
        """
        newest = self.list_events(scope, object_types=object_types, limit=limit, order="DESC")
        newest.reverse()
        return newest

    def latest_event(self, scope: Scope, *, object_types: list[str] | None = None) -> SourceEvent | None:
        """The most recent event of a scope by (timestamp DESC, rowid DESC): among the N messages of one add(), which share event_time, the last WRITTEN wins — list_events / last_events tie-break on the random event_id instead.
        某范围按 (timestamp DESC, rowid DESC) 的最新事件：同一次 add() 的 N 条消息共享事件时间，其中最后**写入**的胜出——list_events / last_events 则以随机的 event_id 打破平局。

        Input / 输入：
            scope: Scope filter (Scope.matches semantics).
            scope：范围过滤（Scope.matches 语义）。

            object_types: Optional object_type whitelist (HMS.context() passes the five extractable types).
            object_types：可选的 object_type 白名单（HMS.context() 传五种可抽取类型）。

        Output / 输出：
            The event or None when the scope holds none.
            事件，范围内没有时为 None。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        clauses = [where]
        if object_types:
            clauses.append(f"object_type IN ({_placeholders(len(object_types))})")
            params.extend(object_types)
        sql = f"SELECT * FROM source_events WHERE {' AND '.join(clauses)} ORDER BY timestamp DESC, rowid DESC LIMIT 1"
        row = self._execute(sql, params).fetchone()
        return None if row is None else self._event_from_row(row)

    def get_or_create_anchor(self, ev: SourceEvent) -> Anchor:
        """Return the anchor of an inserted event, creating it once (UNIQUE source_event_id).
        返回已插入事件的锚点，仅创建一次（source_event_id 唯一）。

        Input / 输入：
            ev: An event whose event_id is set (already inserted).
            ev：event_id 已设置（已插入）的事件。

        Output / 输出：
            The Anchor (permission_scope = ev.visibility, source_ref = ev.raw_ref, hash = payload_hash).
            Anchor（permission_scope = ev.visibility，source_ref = ev.raw_ref，hash = payload_hash）。

        State effect / 状态影响：
            Writes anchors when absent.
            缺失时写入 anchors。
        """
        if not ev.event_id:
            raise StoreError("event must be inserted before it can be anchored", "STORE_001")
        with self.transaction():
            existing = self.anchor_for_event(ev.event_id)
            if existing is not None:
                self._verified_anchors.add(existing.anchor_id)
                return existing
            anchor = Anchor(
                anchor_id=self.new_id("anc"),
                source_event_id=ev.event_id,
                hash=ev.payload_hash,
                source_ref=ev.raw_ref,
                timestamp=ev.timestamp,
                ingested_at=self.now(),
                permission_scope=ev.visibility,
                scope=ev.scope,
            )
            self._execute(
                "INSERT INTO anchors(anchor_id, source_event_id, hash, source_ref, timestamp, ingested_at, "
                "permission_scope, user_id, agent_id, run_id, revoked_at) VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    anchor.anchor_id,
                    anchor.source_event_id,
                    anchor.hash,
                    anchor.source_ref,
                    anchor.timestamp,
                    anchor.ingested_at,
                    anchor.permission_scope,
                    anchor.scope.user_id,
                    anchor.scope.agent_id,
                    anchor.scope.run_id,
                ),
            )
            # A witness this transaction just wrote is proven for the rest of the transaction.
            # 本事务刚写入的见证在事务余下部分视为已证明。
            self._verified_anchors.add(anchor.anchor_id)
            return anchor

    def get_anchor(self, anchor_id: str) -> Anchor | None:
        """Fetch one anchor.
        读取一个锚点。

        Input / 输入：
            anchor_id: Anchor id.
            anchor_id：锚点 id。

        Output / 输出：
            Anchor or None.
            Anchor 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM anchors WHERE anchor_id = ?", (anchor_id,)).fetchone()
        return self._anchor_from_row(row) if row else None

    def get_anchors(self, anchor_ids: Iterable[str]) -> list[Anchor]:
        """Fetch anchors in the given order, skipping unknown ids.
        按给定顺序读取锚点，跳过未知 id。

        Input / 输入：
            anchor_ids: Ids to resolve.
            anchor_ids：待解析的 id。

        Output / 输出：
            Anchors in input order (duplicates collapsed).
            按输入顺序的 Anchor（重复折叠）。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in anchor_ids))
        if not wanted:
            return []
        found: dict[str, Anchor] = {}
        for chunk in _chunks(wanted):
            rows = self._execute(
                f"SELECT * FROM anchors WHERE anchor_id IN ({_placeholders(len(chunk))})", list(chunk)
            ).fetchall()
            for row in rows:
                found[row["anchor_id"]] = self._anchor_from_row(row)
        return [found[i] for i in wanted if i in found]

    def anchor_for_event(self, event_id: str) -> Anchor | None:
        """The anchor of an event, if any.
        事件的锚点（若有）。

        Input / 输入：
            event_id: Event id.
            event_id：事件 id。

        Output / 输出：
            Anchor or None.
            Anchor 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM anchors WHERE source_event_id = ?", (event_id,)).fetchone()
        return self._anchor_from_row(row) if row else None

    def revoke_anchor(self, anchor_id: str, *, at: str) -> None:
        """Mark an anchor revoked (never deleted).
        标记锚点已撤销（绝不删除）。

        Input / 输入：
            anchor_id: Anchor id.
            anchor_id：锚点 id。

            at: Revocation instant.
            at：撤销时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates anchors.revoked_at.
            更新 anchors.revoked_at。
        """
        with self.transaction():
            self._execute("UPDATE anchors SET revoked_at = ? WHERE anchor_id = ?", (to_utc_iso(at), anchor_id))

    def set_anchor_permission(self, anchor_id: str, visibility: str) -> None:
        """Change an anchor's permission_scope (the only other mutable anchor column).
        修改锚点的 permission_scope（另一个唯一可变列）。

        Input / 输入：
            anchor_id: Anchor id.
            anchor_id：锚点 id。

            visibility: New Visibility value.
            visibility：新的 Visibility 值。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates anchors.permission_scope.
            更新 anchors.permission_scope。
        """
        with self.transaction():
            self._execute("UPDATE anchors SET permission_scope = ? WHERE anchor_id = ?", (str(visibility), anchor_id))

    def events_without_atoms(self, scope: Scope, *, object_types: list[str]) -> list[SourceEvent]:
        """Events whose anchor is cited by no atom_anchor row (crash recovery / collect).
        其锚点未被任何 atom_anchor 行引用的事件（崩溃恢复 / collect）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            object_types: object_type whitelist (extractable types).
            object_types：object_type 白名单（可抽取类型）。

        Output / 输出：
            Events ordered by (timestamp, event_id).
            按 (timestamp, event_id) 排序的事件。

        State effect / 状态影响：
            None.
            无。
        """
        if not object_types:
            return []
        where, params = _scope_where(scope, "e.")
        params.extend(object_types)
        rows = self._execute(
            "SELECT e.* FROM source_events e LEFT JOIN anchors a ON a.source_event_id = e.event_id "
            f"WHERE {where} AND e.object_type IN ({_placeholders(len(object_types))}) "
            "AND (a.anchor_id IS NULL OR NOT EXISTS (SELECT 1 FROM atom_anchor aa WHERE aa.anchor_id = a.anchor_id)) "
            "ORDER BY e.timestamp, e.event_id",
            params,
        ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def last_open_action(
        self, scope: Scope, *, run_id: str | None, before: str, max_gap_hours: float
    ) -> tuple[SourceEvent, Anchor] | None:
        """Newest tool_call/file_diff event within the gap whose anchor no EP cites (cross-batch closure).
        间隔内最新的、锚点未被任何 EP 引用的 tool_call/file_diff 事件（跨批闭包）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            run_id: Restrict to this run when given, else the scope's run semantics.
            run_id：给定时限定该 run，否则沿用范围的 run 语义。

            before: Upper bound on timestamp (inclusive).
            before：timestamp 上界（含）。

            max_gap_hours: Maximum age of the action relative to `before`.
            max_gap_hours：动作相对 `before` 的最大时长。

        Output / 输出：
            (event, anchor) or None.
            (event, anchor) 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        upper = to_utc_iso(before)
        lower = to_utc_iso(parse_iso(upper) - timedelta(hours=float(max_gap_hours)))
        where, params = _scope_where(scope, "e.")
        clauses = [where, "e.object_type IN ('tool_call','file_diff')", "e.timestamp <= ?", "e.timestamp >= ?"]
        params.extend([upper, lower])
        if run_id is not None:
            clauses.append("e.run_id = ?")
            params.append(run_id)
        base = (
            "SELECT e.*, a.anchor_id AS _anchor_id FROM source_events e "
            "JOIN anchors a ON a.source_event_id = e.event_id WHERE " + " AND ".join(clauses)
        )
        order = " ORDER BY e.timestamp DESC, e.event_id DESC"
        if self.capabilities.get("json1"):
            row = self._execute(
                base
                + " AND NOT EXISTS (SELECT 1 FROM eps, json_each(eps.anchor_ids_json) je WHERE je.value = a.anchor_id)"
                + order
                + " LIMIT 1",
                params,
            ).fetchone()
        else:
            cited: set[str] = set()
            for ep_row in self._execute("SELECT anchor_ids_json FROM eps").fetchall():
                cited.update(_loads(ep_row["anchor_ids_json"], []))
            row = None
            for candidate in self._execute(base + order, params).fetchall():
                if candidate["_anchor_id"] not in cited:
                    row = candidate
                    break
        if row is None:
            return None
        anchor = self.get_anchor(row["_anchor_id"])
        if anchor is None:  # pragma: no cover - joined above
            return None
        return self._event_from_row(row), anchor

    # ------------------------------------------------------------------ observations
    # ------------------------------------------------------------------ 观察
    def _observation_from_row(self, row: sqlite3.Row) -> Observation:
        """Observation from an observations row.
        由 observations 行构造 Observation。
        """
        return Observation(
            obs_id=row["obs_id"],
            type=row["type"],
            subject=row["subject"],
            predicate=row["predicate"],
            objects=_loads(row["objects_json"], []),
            relation=row["relation"],
            text=row["text"],
            date_observed=row["date_observed"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            confidence=row["confidence"],
            visibility=row["visibility"],
            attributed_to=row["attributed_to"],
            sentiment=row["sentiment"],
            topic_tags=_loads(row["topic_tags_json"], []),
            entity_mentions=_loads(row["entity_mentions_json"], []),
            supersedes_hint_ids=_loads(row["supersedes_hint_ids_json"], []),
            extracted_by=row["extracted_by"],
            scope=_scope_of(row),
            atom_id=row["atom_id"],
            created_at=row["created_at"],
            anchor_ids=_loads(row["anchor_ids_json"], []) or [row["anchor_id"]],
        )

    def insert_observation(self, obs: Observation) -> Observation:
        """Persist an observation (witness pre-checked; obs_id/created_at assigned when empty).
        持久化观察（预检见证；obs_id/created_at 为空时分配）。

        Input / 输入：
            obs: The observation.
            obs：观察。

        Output / 输出：
            The stored observation with ids.
            带 id 的已存储观察。

        State effect / 状态影响：
            Writes observations.
            写入 observations。
        """
        anchor_ids = self._require_anchors(obs.anchor_ids, obj="observation", obj_id=obs.obs_id or None)
        obs_id = obs.obs_id or self.new_id("obs")
        created_at = obs.created_at or self.now()
        stored = obs.model_copy(update={"obs_id": obs_id, "created_at": created_at})
        with self.transaction():
            self._execute(
                "INSERT INTO observations(obs_id, anchor_id, anchor_ids_json, type, subject, predicate, objects_json, "
                "relation, text, date_observed, valid_from, valid_to, confidence, visibility, attributed_to, sentiment, "
                "topic_tags_json, entity_mentions_json, supersedes_hint_ids_json, extracted_by, atom_id, user_id, "
                "agent_id, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    obs_id,
                    anchor_ids[0],
                    canonical_json(anchor_ids),
                    obs.type,
                    obs.subject,
                    obs.predicate,
                    canonical_json(list(obs.objects)),
                    obs.relation,
                    obs.text,
                    obs.date_observed,
                    obs.valid_from,
                    obs.valid_to,
                    float(obs.confidence),
                    obs.visibility,
                    obs.attributed_to,
                    obs.sentiment,
                    canonical_json(list(obs.topic_tags)),
                    canonical_json(list(obs.entity_mentions)),
                    canonical_json(list(obs.supersedes_hint_ids)),
                    obs.extracted_by,
                    obs.atom_id,
                    obs.scope.user_id,
                    obs.scope.agent_id,
                    obs.scope.run_id,
                    created_at,
                ),
            )
        return stored

    def get_observation(self, obs_id: str) -> Observation | None:
        """Fetch one observation.
        读取一条观察。

        Input / 输入：
            obs_id: Observation id.
            obs_id：观察 id。

        Output / 输出：
            Observation or None.
            Observation 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM observations WHERE obs_id = ?", (obs_id,)).fetchone()
        return self._observation_from_row(row) if row else None

    def set_observation_atom(self, obs_id: str, atom_id: str) -> None:
        """Set the back-pointer from an observation to its atom.
        设置观察指向其原子的回指。

        Input / 输入：
            obs_id: Observation id.
            obs_id：观察 id。

            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates observations.atom_id.
            更新 observations.atom_id。
        """
        with self.transaction():
            self._execute("UPDATE observations SET atom_id = ? WHERE obs_id = ?", (atom_id, obs_id))

    def set_observation_valid_to(self, obs_id: str, valid_to: str | None) -> None:
        """Mirror the observation atom's valid_to onto the observation row.
        将观察原子的 valid_to 镜像到观察行。

        Input / 输入：
            obs_id: Observation id.
            obs_id：观察 id。

            valid_to: New valid_to or None.
            valid_to：新的 valid_to 或 None。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates observations.valid_to.
            更新 observations.valid_to。
        """
        with self.transaction():
            self._execute(
                "UPDATE observations SET valid_to = ? WHERE obs_id = ?",
                (to_utc_iso(valid_to) if valid_to else None, obs_id),
            )

    # ------------------------------------------------------------------ EPs
    # ------------------------------------------------------------------ EP（经验包）
    def _ep_from_row(self, row: sqlite3.Row) -> EP:
        """EP from an eps row.
        由 eps 行构造 EP。
        """
        kappa = Kappa(**_loads(row["kappa_json"], {}))
        slots = Slots(u=row["u"], b=row["b"], a=row["a"], y=row["y"], r=row["r"], kappa=kappa)
        return EP(
            ep_id=row["ep_id"],
            scale=row["scale"],
            slots=slots,
            time_anchor=row["time_anchor"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            entity_mentions=_loads(row["entity_mentions_json"], []),
            topic_tags=_loads(row["topic_tags_json"], []),
            uses_ep_ids=_loads(row["uses_ep_ids_json"], []),
            confidence=row["confidence"],
            visibility=row["visibility"],
            extracted_by=row["extracted_by"],
            scope=_scope_of(row),
            atom_id=row["atom_id"],
            created_at=row["created_at"],
            anchor_ids=_loads(row["anchor_ids_json"], []) or [row["anchor_id"]],
        )

    def insert_ep(self, ep: EP) -> EP:
        """Persist an L1 EP (witness pre-checked; valid_from defaults to time_anchor).
        持久化 L1 EP（预检见证；valid_from 默认 time_anchor）。

        Input / 输入：
            ep: The EP.
            ep：EP。

        Output / 输出：
            The stored EP with ids.
            带 id 的已存储 EP。

        State effect / 状态影响：
            Writes eps.
            写入 eps。
        """
        anchor_ids = self._require_anchors(ep.anchor_ids, obj="ep", obj_id=ep.ep_id or None)
        ep_id = ep.ep_id or self.new_id("ep")
        created_at = ep.created_at or self.now()
        valid_from = ep.valid_from or ep.time_anchor
        stored = ep.model_copy(update={"ep_id": ep_id, "created_at": created_at, "valid_from": valid_from})
        slots = ep.slots
        with self.transaction():
            self._execute(
                "INSERT INTO eps(ep_id, anchor_id, anchor_ids_json, scale, u, b, a, y, r, kappa_json, time_anchor, "
                "valid_from, valid_to, entity_mentions_json, topic_tags_json, uses_ep_ids_json, confidence, "
                "visibility, extracted_by, atom_id, user_id, agent_id, run_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ep_id,
                    anchor_ids[0],
                    canonical_json(anchor_ids),
                    ep.scale,
                    slots.u,
                    slots.b,
                    slots.a,
                    slots.y,
                    slots.r,
                    canonical_json(slots.kappa.model_dump()),
                    ep.time_anchor,
                    valid_from,
                    ep.valid_to,
                    canonical_json(list(ep.entity_mentions)),
                    canonical_json(list(ep.topic_tags)),
                    canonical_json(list(ep.uses_ep_ids)),
                    float(ep.confidence),
                    ep.visibility,
                    ep.extracted_by,
                    ep.atom_id,
                    ep.scope.user_id,
                    ep.scope.agent_id,
                    ep.scope.run_id,
                    created_at,
                ),
            )
        return stored

    def get_ep(self, ep_id: str) -> EP | None:
        """Fetch one EP.
        读取一个 EP。

        Input / 输入：
            ep_id: EP id.
            ep_id：EP id。

        Output / 输出：
            EP or None.
            EP 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM eps WHERE ep_id = ?", (ep_id,)).fetchone()
        return self._ep_from_row(row) if row else None

    def get_eps(self, ep_ids: Iterable[str]) -> list[EP]:
        """Fetch EPs in input order, skipping unknown ids.
        按输入顺序读取 EP，跳过未知 id。

        Input / 输入：
            ep_ids: EP ids.
            ep_ids：EP id。

        Output / 输出：
            EPs.
            EP 列表。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in ep_ids))
        found: dict[str, EP] = {}
        for chunk in _chunks(wanted):
            for row in self._execute(f"SELECT * FROM eps WHERE ep_id IN ({_placeholders(len(chunk))})", list(chunk)):
                found[row["ep_id"]] = self._ep_from_row(row)
        return [found[i] for i in wanted if i in found]

    def set_ep_atom(self, ep_id: str, atom_id: str) -> None:
        """Set the back-pointer from an EP to its episode atom.
        设置 EP 指向其情节原子的回指。

        Input / 输入：
            ep_id: EP id.
            ep_id：EP id。

            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates eps.atom_id.
            更新 eps.atom_id。
        """
        with self.transaction():
            self._execute("UPDATE eps SET atom_id = ? WHERE ep_id = ?", (atom_id, ep_id))

    def _ep_valid_clause(self) -> str:
        """Validity of an EP: eps.valid_to IS NULL and its node (when present) is valid.
        EP 的有效性：eps.valid_to 为空且其节点（若存在）有效。
        """
        return (
            "eps.valid_to IS NULL AND NOT EXISTS "
            "(SELECT 1 FROM graph_nodes n WHERE n.node_id = eps.atom_id AND n.valid = 0)"
        )

    def list_eps(self, scope: Scope, *, run_id: str | None = None, valid_only: bool = True) -> list[EP]:
        """EPs of a scope ordered by (time_anchor, ep_id).
        按 (time_anchor, ep_id) 排序的范围内 EP。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            run_id: Extra run restriction when given.
            run_id：给定时的额外 run 限制。

            valid_only: Keep only EPs with valid_to NULL and a valid node.
            valid_only：仅保留 valid_to 为空且节点有效的 EP。

        Output / 输出：
            EPs.
            EP 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope, "eps.")
        clauses = [where]
        if run_id is not None:
            clauses.append("eps.run_id = ?")
            params.append(run_id)
        if valid_only:
            clauses.append(self._ep_valid_clause())
        rows = self._execute(
            f"SELECT eps.* FROM eps WHERE {' AND '.join(clauses)} ORDER BY eps.time_anchor, eps.ep_id", params
        ).fetchall()
        return [self._ep_from_row(r) for r in rows]

    def last_ep(self, scope: Scope, *, run_id: str | None) -> EP | None:
        """Newest valid EP of a scope/run.
        范围/run 内最新的有效 EP。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            run_id: Extra run restriction when given.
            run_id：给定时的额外 run 限制。

        Output / 输出：
            EP or None, by (time_anchor DESC, created_at DESC, ep_id DESC).
            按 (time_anchor DESC, created_at DESC, ep_id DESC) 的 EP 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope, "eps.")
        clauses = [where, self._ep_valid_clause()]
        if run_id is not None:
            clauses.append("eps.run_id = ?")
            params.append(run_id)
        row = self._execute(
            f"SELECT eps.* FROM eps WHERE {' AND '.join(clauses)} "
            "ORDER BY eps.time_anchor DESC, eps.created_at DESC, eps.ep_id DESC LIMIT 1",
            params,
        ).fetchone()
        return self._ep_from_row(row) if row else None

    def archive_ep(self, ep_id: str, *, at: str) -> None:
        """Close an EP's validity window.
        关闭 EP 的有效窗口。

        Input / 输入：
            ep_id: EP id.
            ep_id：EP id。

            at: valid_to instant.
            at：valid_to 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates eps.valid_to.
            更新 eps.valid_to。
        """
        with self.transaction():
            self._execute("UPDATE eps SET valid_to = ? WHERE ep_id = ?", (to_utc_iso(at), ep_id))

    def restore_ep(self, ep_id: str) -> None:
        """Reopen an EP's validity window.
        重新打开 EP 的有效窗口。

        Input / 输入：
            ep_id: EP id.
            ep_id：EP id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Sets eps.valid_to to NULL.
            将 eps.valid_to 置为 NULL。
        """
        with self.transaction():
            self._execute("UPDATE eps SET valid_to = NULL WHERE ep_id = ?", (ep_id,))

    def scope_ep_state(self, scope: Scope) -> dict[str, str | None]:
        """Every EP in scope → its valid_to (checkpoint ep_state).
        范围内每个 EP → 其 valid_to（检查点 ep_state）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

        Output / 输出：
            {ep_id: valid_to}.
            {ep_id: valid_to}。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        rows = self._execute(f"SELECT ep_id, valid_to FROM eps WHERE {where} ORDER BY ep_id", params).fetchall()
        return {r["ep_id"]: r["valid_to"] for r in rows}

    # ------------------------------------------------------------------ atoms (State Index)
    # ------------------------------------------------------------------ 原子（State Index）
    def _links_for(
        self, atom_ids: Sequence[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """One statement per chunk for the three axis links: (entity keys sorted, topics in insertion order, undisputed anchors primary first).
        每块一条语句取三种轴链接：（排序后的实体键、按插入顺序的主题、无争议锚点且 primary 在前）。
        """
        entities: dict[str, list[str]] = {}
        topics: dict[str, list[tuple[int, str]]] = {}
        anchors: dict[str, list[tuple[int, int, str]]] = {}
        for chunk in _chunks(list(dict.fromkeys(atom_ids))):
            # The id list is bound once into a VALUES CTE and joined three times (each join probes the
            # (atom_id, …) primary-key index), replacing three round trips with one.
            # id 列表只绑定一次进 VALUES CTE，再连接三次（每次连接都探测 (atom_id, …) 主键索引），
            # 三次往返合并为一次。
            rows = self._execute(
                f"WITH ids(atom_id) AS (VALUES {','.join('(?)' for _ in chunk)}) "
                "SELECT 0 AS kind, ae.atom_id AS atom_id, e.normalized_text AS value, 0 AS ord "
                "FROM ids JOIN atom_entity ae ON ae.atom_id = ids.atom_id JOIN entities e ON e.entity_id = ae.entity_id "
                "UNION ALL SELECT 1, t.atom_id, t.topic, t.rowid FROM ids JOIN atom_topic t ON t.atom_id = ids.atom_id "
                "UNION ALL SELECT CASE a.role WHEN 'primary' THEN 2 ELSE 3 END, a.atom_id, a.anchor_id, a.rowid "
                "FROM ids JOIN atom_anchor a ON a.atom_id = ids.atom_id WHERE a.disputed_at IS NULL",
                list(chunk),
            ).fetchall()
            for row in rows:
                kind = row["kind"]
                if kind == 0:
                    entities.setdefault(row["atom_id"], []).append(row["value"])
                elif kind == 1:
                    topics.setdefault(row["atom_id"], []).append((row["ord"], row["value"]))
                else:
                    anchors.setdefault(row["atom_id"], []).append((kind, row["ord"], row["value"]))
        return (
            {k: sorted(dict.fromkeys(v)) for k, v in entities.items()},
            {k: [topic for _ord, topic in sorted(v)] for k, v in topics.items()},
            {k: [anchor for _kind, _ord, anchor in sorted(v)] for k, v in anchors.items()},
        )

    def _anchors_for(self, atom_ids: Sequence[str], *, include_disputed: bool = False) -> dict[str, list[str]]:
        """atom_id → anchor ids, primary first then insertion order.
        atom_id → 锚点 id，主锚点优先，其后按插入顺序。
        """
        out: dict[str, list[str]] = {}
        cond = "" if include_disputed else " AND disputed_at IS NULL"
        for chunk in _chunks(list(atom_ids)):
            rows = self._execute(
                "SELECT atom_id, anchor_id FROM atom_anchor "
                f"WHERE atom_id IN ({_placeholders(len(chunk))}){cond} "
                "ORDER BY atom_id, CASE role WHEN 'primary' THEN 0 ELSE 1 END, rowid",
                list(chunk),
            ).fetchall()
            for row in rows:
                out.setdefault(row["atom_id"], []).append(row["anchor_id"])
        return out

    def _hydrate_atoms(self, rows: Sequence[sqlite3.Row], *, trusted: bool = False) -> list[MemoryAtom]:
        """Build MemoryAtom models with entity_ids / topic_tags / anchor_ids filled (§8 Hydration).
        构造并填充 entity_ids / topic_tags / anchor_ids 的 MemoryAtom（§8 水合）。

        `trusted=True` builds the models with `model_construct` from the same field dict: every stored value was
        validated on its way in (instants normalized, enums stored as values, JSON canonical), so the result equals
        the validated model field for field (`test_trusted_hydration_equals_validated`) at a fraction of the cost.
        `trusted=True` 用同一字段字典经 `model_construct` 构造模型：每个已存值写入时都已校验（时刻已规范化、枚举按值
        存储、JSON 规范形式），因此结果与校验构造的模型逐字段相同（`test_trusted_hydration_equals_validated`），
        代价只是零头。
        """
        if not rows:
            return []
        ids = [r["atom_id"] for r in rows]
        entities, topics, anchors = self._links_for(ids)
        missing = [i for i in ids if not anchors.get(i)]
        # 全部见证都被争议：回退到完整锚点列表，以保证模型仍可构造
        if missing:  # every witness disputed: fall back to the full list so the model stays constructible
            anchors.update(self._anchors_for(missing, include_disputed=True))
        build = MemoryAtom.model_construct if trusted else MemoryAtom
        scope_of = Scope.model_construct if trusted else Scope
        atoms: list[MemoryAtom] = []
        # One positional tuple → dict per row: sqlite3.Row's by-name access costs ~0.3 µs per column, 33 times.
        # 每行一次位置元组 → 字典：sqlite3.Row 按名访问每列约 0.3 µs，共 33 次。
        keys = rows[0].keys()
        for raw in rows:
            row = dict(zip(keys, tuple(raw)))
            atom_id = row["atom_id"]
            atoms.append(
                build(
                    atom_id=atom_id,
                    memory_id=row["memory_id"],
                    version=row["version"],
                    atom_type=row["atom_type"],
                    content=_loads(row["content_json"], {}),
                    text=row["text"],
                    hash=row["hash"],
                    time_anchor=row["time_anchor"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    entity_ids=entities.get(atom_id, []),
                    topic_tags=topics.get(atom_id, []),
                    scale_level=row["scale_level"],
                    role=row["role"],
                    sentiment=row["sentiment"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    obs_type=row["obs_type"],
                    source_kind=row["source_kind"],
                    source_id=row["source_id"],
                    source_ep_ids=_loads(row["source_ep_ids_json"], []),
                    confidence=row["confidence"],
                    extracted_by=row["extracted_by"],
                    visibility=row["visibility"],
                    scope=scope_of(user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"]),
                    actor_id=row["actor_id"],
                    attributed_to=row["attributed_to"],
                    memory_type=row["memory_type"],
                    user_meta=_loads(row["user_meta_json"], {}),
                    archived_reason=row["archived_reason"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    anchor_ids=anchors.get(atom_id) or [row["primary_anchor_id"]],
                )
            )
        return atoms

    def _select_atoms(self, sql: str, params: Sequence[Any] = ()) -> list[MemoryAtom]:
        """Run a memory_atoms SELECT and hydrate.
        执行 memory_atoms 的 SELECT 并水合。
        """
        return self._hydrate_atoms(self._execute(sql, params).fetchall())

    def insert_atom(self, atom: MemoryAtom, *, entity_keys: list[EntityKey] = (), topics: list[str] = ()) -> MemoryAtom:
        """Persist an atom version with its support anchors, entity/topic links and graph node.
        持久化一个原子版本及其支持锚点、实体/主题链接与图节点。

        Input / 输入：
            atom: The atom; atom_id/memory_id (new lineage when empty), created_at/updated_at, valid_from assigned when empty.
            atom：原子；atom_id/memory_id（为空时新谱系）、created_at/updated_at、valid_from 为空时分配。

            entity_keys: (key, canonical, entity_type, aliases) tuples linked via get_or_create_entity.
            entity_keys：经 get_or_create_entity 链接的 (key, canonical, entity_type, aliases) 元组。

            topics: Topic tags (ordered union with atom.topic_tags).
            topics：主题标签（与 atom.topic_tags 做有序并集）。

        Output / 输出：
            The hydrated stored atom.
            水合后的已存储原子。

        State effect / 状态影响：
            Writes memory_atoms, atom_anchor, entities, atom_entity, atom_topic, graph_nodes.
            写入 memory_atoms、atom_anchor、entities、atom_entity、atom_topic、graph_nodes。
        """
        anchor_ids = self._require_anchors(atom.anchor_ids, obj="atom", obj_id=atom.atom_id or None)
        atom_id = atom.atom_id or self.new_id("atm")
        memory_id = atom.memory_id or self.new_id("mem")
        created_at = atom.created_at or self.now()
        updated_at = atom.updated_at or created_at
        valid_from = atom.valid_from or atom.time_anchor or created_at
        if atom.valid_to is not None and atom.valid_to < valid_from:
            raise ValidationError("valid_to must be >= valid_from", "VALIDATION_004", details={"atom_id": atom_id})
        node_kind = NODE_KIND_OF_SOURCE.get(str(atom.source_kind), NodeKind.observation.value)
        scope = atom.scope
        with self.transaction():
            self.touch_scope(scope)
            self._execute(
                "INSERT INTO memory_atoms(atom_id, memory_id, version, atom_type, content_json, text, hash, "
                "time_anchor, valid_from, valid_to, scale_level, role, sentiment, subject, predicate, obs_type, "
                "source_kind, source_id, source_ep_ids_json, confidence, extracted_by, primary_anchor_id, visibility, "
                "user_id, agent_id, run_id, actor_id, attributed_to, memory_type, user_meta_json, archived_reason, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    atom_id,
                    memory_id,
                    int(atom.version),
                    atom.atom_type,
                    canonical_json(atom.content),
                    atom.text,
                    atom.hash,
                    atom.time_anchor,
                    valid_from,
                    atom.valid_to,
                    atom.scale_level,
                    atom.role,
                    atom.sentiment,
                    atom.subject,
                    atom.predicate,
                    atom.obs_type,
                    atom.source_kind,
                    atom.source_id,
                    canonical_json(list(atom.source_ep_ids)),
                    float(atom.confidence),
                    atom.extracted_by,
                    anchor_ids[0],
                    atom.visibility,
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    atom.actor_id,
                    atom.attributed_to,
                    atom.memory_type,
                    canonical_json(atom.user_meta),
                    atom.archived_reason,
                    created_at,
                    updated_at,
                ),
            )
            for anchor_id in anchor_ids[1:]:
                self._execute(
                    "INSERT OR IGNORE INTO atom_anchor(atom_id, anchor_id, role) VALUES (?, ?, 'support')",
                    (atom_id, anchor_id),
                )
            for key, canonical, entity_type, aliases in entity_keys:
                entity_id = self.get_or_create_entity(
                    scope, normalized=key, canonical=canonical, entity_type=entity_type, aliases=tuple(aliases)
                )
                self._execute(
                    "INSERT OR IGNORE INTO atom_entity(atom_id, entity_id) VALUES (?, ?)", (atom_id, entity_id)
                )
            topic_list = ordered_union(list(topics), list(atom.topic_tags))
            for topic in topic_list:
                self._execute("INSERT OR IGNORE INTO atom_topic(atom_id, topic) VALUES (?, ?)", (atom_id, topic))
            archived = atom.archived_reason is not None
            if not archived and self.capabilities.get("fts5"):
                # The keyword channel row (archived versions carry none; archive_atom removes it, restore_atom rebuilds it).
                # 关键词通道行（已归档版本没有；archive_atom 删除它，restore_atom 重建它）。
                self._execute(
                    "INSERT INTO atom_fts(atom_id, tokens) VALUES (?, ?)",
                    (atom_id, fts_tokens(atom.text, topic_list, [key for key, _c, _t, _a in entity_keys])),
                )
            self._execute(
                "INSERT INTO graph_nodes(node_id, node_kind, ref_id, scale, primary_anchor_id, user_id, agent_id, "
                "run_id, valid, invalidated_at, mentions, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,1,?)",
                (
                    atom_id,
                    node_kind,
                    atom.source_id,
                    atom.scale_level,
                    anchor_ids[0],
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    0 if archived else 1,
                    updated_at if archived else None,
                    created_at,
                ),
            )
            return self._materialized(
                atom,
                atom_id=atom_id,
                memory_id=memory_id,
                created_at=created_at,
                updated_at=updated_at,
                valid_from=valid_from,
                anchor_ids=anchor_ids,
                entity_keys=entity_keys,
                topics=topic_list,
            )

    def _materialized(
        self,
        atom: MemoryAtom,
        *,
        atom_id: str,
        memory_id: str,
        created_at: str,
        updated_at: str,
        valid_from: str,
        anchor_ids: Sequence[str],
        entity_keys: Sequence[EntityKey],
        topics: Sequence[str],
    ) -> MemoryAtom:
        """The hydrated form of an atom insert_atom just wrote, rebuilt from the write itself instead of re-read.
        insert_atom 刚写入原子的水合形式，由写入本身重建而非重新读取。
        """
        # Mirrors the read path exactly: JSON columns round-trip through canonical_json, entity keys are the
        # sorted unique normalized texts, topics keep insertion order, anchors are primary first then the
        # OR-IGNORE-deduplicated support witnesses in insertion order (asserted equal to get_atom by tests).
        # 与读取路径严格一致：JSON 列经 canonical_json 往返，实体键为排序去重的 normalized 文本，
        # 主题保持插入顺序，锚点为 primary 在前、其后是 OR IGNORE 去重后按插入顺序的支持见证（测试断言与 get_atom 相等）。
        primary = anchor_ids[0]
        fields = atom.model_dump()
        fields.update(
            atom_id=atom_id,
            memory_id=memory_id,
            version=int(atom.version),
            content=_loads(canonical_json(atom.content), {}),
            valid_from=valid_from,
            entity_ids=sorted(dict.fromkeys(str(key) for key, _canonical, _type, _aliases in entity_keys)),
            topic_tags=[topic for topic in topics if topic is not None],
            source_ep_ids=_loads(canonical_json(list(atom.source_ep_ids)), []),
            confidence=float(atom.confidence),
            user_meta=_loads(canonical_json(atom.user_meta), {}),
            created_at=created_at,
            updated_at=updated_at,
            anchor_ids=[primary] + [a for a in dict.fromkeys(anchor_ids[1:]) if a != primary],
        )
        return MemoryAtom(**fields)

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        """Fetch one atom version (hydrated).
        读取一个原子版本（已水合）。

        Input / 输入：
            atom_id: Atom (version) id.
            atom_id：原子（版本）id。

        Output / 输出：
            MemoryAtom or None.
            MemoryAtom 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        atoms = self._select_atoms("SELECT * FROM memory_atoms WHERE atom_id = ?", (atom_id,))
        return atoms[0] if atoms else None

    def get_atoms(self, atom_ids: Iterable[str]) -> list[MemoryAtom]:
        """Fetch atoms in input order, skipping unknown ids.
        按输入顺序读取原子，跳过未知 id。

        Input / 输入：
            atom_ids: Atom ids.
            atom_ids：原子 id。

        Output / 输出：
            Hydrated atoms.
            水合后的原子。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        found: dict[str, MemoryAtom] = {}
        for chunk in _chunks(wanted):
            for atom in self._select_atoms(
                f"SELECT * FROM memory_atoms WHERE atom_id IN ({_placeholders(len(chunk))})", list(chunk)
            ):
                found[atom.atom_id] = atom
        return [found[i] for i in wanted if i in found]

    def resolve_head(self, memory_id: str) -> MemoryAtom | None:
        """The lineage head: the version with archived_reason NULL (expiring heads included).
        谱系头：archived_reason 为空的版本（含即将过期的头）。

        Input / 输入：
            memory_id: Lineage id.
            memory_id：谱系 id。

        Output / 输出：
            MemoryAtom or None.
            MemoryAtom 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        atoms = self._select_atoms(
            "SELECT * FROM memory_atoms WHERE memory_id = ? AND archived_reason IS NULL ORDER BY version DESC LIMIT 1",
            (memory_id,),
        )
        return atoms[0] if atoms else None

    def latest_version(self, memory_id: str) -> MemoryAtom | None:
        """The highest version of a lineage regardless of validity.
        谱系中版本号最高的原子，不论有效性。

        Input / 输入：
            memory_id: Lineage id.
            memory_id：谱系 id。

        Output / 输出：
            MemoryAtom or None.
            MemoryAtom 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        atoms = self._select_atoms(
            "SELECT * FROM memory_atoms WHERE memory_id = ? ORDER BY version DESC LIMIT 1", (memory_id,)
        )
        return atoms[0] if atoms else None

    def list_versions(self, memory_id: str) -> list[MemoryAtom]:
        """Every version of a lineage ordered by version.
        按版本排序的谱系全部版本。

        Input / 输入：
            memory_id: Lineage id.
            memory_id：谱系 id。

        Output / 输出：
            Atoms ordered by version.
            按版本排序的原子。

        State effect / 状态影响：
            None.
            无。
        """
        return self._select_atoms("SELECT * FROM memory_atoms WHERE memory_id = ? ORDER BY version", (memory_id,))

    def _atom_source(self, atom_id: str) -> tuple[str, str] | None:
        """(source_kind, source_id) of an atom.
        原子的 (source_kind, source_id)。
        """
        row = self._execute("SELECT source_kind, source_id FROM memory_atoms WHERE atom_id = ?", (atom_id,)).fetchone()
        return (row["source_kind"], row["source_id"]) if row else None

    def archive_atom(self, atom_id: str, *, reason: str, at: str, valid_to: str | None = None) -> None:
        """Archive an atom version (revision, never deletion) with cascade to its source row.
        归档原子版本（修订而非删除），并级联到其来源行。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            reason: archived_reason (revised|deleted|feedback|rollback|decayed|conflict|...).
            reason：archived_reason（revised|deleted|feedback|rollback|decayed|conflict|…）。

            at: updated_at / node invalidation instant.
            at：updated_at / 节点失效时刻。

            valid_to: Explicit validity end; defaults to `at`.
            valid_to：显式有效期终点；默认 `at`。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms, graph_nodes and cascades to eps / observations / interface_eps.
            更新 memory_atoms、graph_nodes，并级联到 eps / observations / interface_eps。
        """
        instant = to_utc_iso(at)
        end = to_utc_iso(valid_to) if valid_to else instant
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET valid_to = ?, archived_reason = ?, updated_at = ? WHERE atom_id = ?",
                (end, reason, instant, atom_id),
            )
            self._execute("UPDATE graph_nodes SET valid = 0, invalidated_at = ? WHERE node_id = ?", (instant, atom_id))
            if self.capabilities.get("fts5"):
                self._execute("DELETE FROM atom_fts WHERE atom_id = ?", (atom_id,))
            self._queue_attrs(atom_id)
            source = self._atom_source(atom_id)
            # The source row is ended only when no other live version still cites it (a feedback / update successor
            # shares its predecessor's observation; revise_atom re-opens it when the successor is created later).
            # 只有当没有其它存活版本仍引用来源行时才结束它（反馈 / 更新的后继共享前驱的观察；后继稍后创建时由
            # revise_atom 重新打开）。
            if source is not None and not self._source_shared(atom_id, *source):
                self._cascade_source(*source, end=end)

    def restore_atom(self, atom_id: str, *, at: str) -> None:
        """Reverse archive_atom (valid_to/archived_reason cleared, node valid again) with cascade.
        撤销 archive_atom（清空 valid_to/archived_reason，节点重新有效）并级联。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            at: updated_at instant.
            at：updated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms, graph_nodes, rebuilds the atom_fts row and cascades to eps / observations / interface_eps.
            更新 memory_atoms、graph_nodes，重建 atom_fts 行，并级联到 eps / observations / interface_eps。
        """
        instant = to_utc_iso(at)
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET valid_to = NULL, archived_reason = NULL, updated_at = ? WHERE atom_id = ?",
                (instant, atom_id),
            )
            self._execute("UPDATE graph_nodes SET valid = 1, invalidated_at = NULL WHERE node_id = ?", (atom_id,))
            if self.capabilities.get("fts5"):
                self._refresh_fts(atom_id)
            self._queue_attrs(atom_id)
            source = self._atom_source(atom_id)
            if source is not None:
                self._cascade_source(*source, end=None)

    def _source_shared(self, atom_id: str, kind: str, source_id: str) -> bool:
        """True when another non-archived atom cites the same (source_kind, source_id) as `atom_id`.
        当另一个未归档原子引用与 `atom_id` 相同的 (source_kind, source_id) 时为真。
        """
        row = self._execute(
            "SELECT 1 FROM memory_atoms WHERE source_kind = ? AND source_id = ? AND atom_id != ? "
            "AND archived_reason IS NULL LIMIT 1",
            (kind, source_id, atom_id),
        ).fetchone()
        return row is not None

    def _cascade_source(self, kind: str, source_id: str, *, end: str | None) -> None:
        """End (`end` set) or re-open (`end` None) the source row of an atom: eps / observations / interface_eps.
        结束（给出 `end`）或重新打开（`end` 为 None）原子的来源行：eps / observations / interface_eps。
        """
        if kind == SourceKind.ep.value:
            if end is None:
                self.restore_ep(source_id)
            else:
                self.archive_ep(source_id, at=end)
        elif kind == SourceKind.observation.value:
            self.set_observation_valid_to(source_id, end)
        elif kind == SourceKind.interface_ep.value:
            if end is None:
                self._execute(
                    "UPDATE interface_eps SET valid = 1, valid_to = NULL WHERE interface_ep_id = ?", (source_id,)
                )
            else:
                self._execute(
                    "UPDATE interface_eps SET valid = 0, valid_to = ? WHERE interface_ep_id = ?", (end, source_id)
                )

    def reopen_source(self, atom_id: str) -> None:
        """Re-open the source row cited by a live atom (undo the archive cascade a purged / archived predecessor caused).
        重新打开某存活原子引用的来源行（撤销已归档前驱造成的归档级联）。

        Input / 输入：
            atom_id: A non-archived atom whose source row may have been ended by archive_atom of another version.
            atom_id：其来源行可能已被另一版本的 archive_atom 结束的未归档原子。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Restores the eps / observations / interface_eps row (no-op for an archived atom or a missing source).
            恢复 eps / observations / interface_eps 行（原子已归档或来源缺失时为空操作）。
        """
        with self.transaction():
            row = self._execute(
                "SELECT source_kind, source_id FROM memory_atoms WHERE atom_id = ? AND archived_reason IS NULL",
                (atom_id,),
            ).fetchone()
            if row is not None:
                self._cascade_source(row["source_kind"], row["source_id"], end=None)

    def shared_source_atom_ids(self, atom_ids: Iterable[str]) -> set[str]:
        """Among `atom_ids`, those whose (source_kind, source_id) is also cited by an atom outside the list.
        `atom_ids` 中其 (source_kind, source_id) 也被列表之外的原子引用的那些。

        Input / 输入：
            atom_ids: Candidate atom ids (a purge set).
            atom_ids：候选原子 id（一个清理集合）。

        Output / 输出：
            The subset whose source row another atom still needs.
            其来源行仍被另一原子需要的子集。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        out: set[str] = set()
        for chunk in _chunks(wanted):
            marks = _placeholders(len(chunk))
            rows = self._execute(
                f"SELECT a.atom_id AS atom_id FROM memory_atoms AS a WHERE a.atom_id IN ({marks}) AND EXISTS ("
                "SELECT 1 FROM memory_atoms AS b WHERE b.source_kind = a.source_kind AND b.source_id = a.source_id "
                f"AND b.atom_id != a.atom_id AND b.atom_id NOT IN ({marks}))",
                [*chunk, *chunk],
            ).fetchall()
            out.update(r["atom_id"] for r in rows)
        return out

    def set_valid_to(self, atom_id: str, valid_to: str | None, *, at: str) -> None:
        """Set (or clear) an atom's valid_to without archiving it (expiration_date).
        设置（或清除）原子的 valid_to 而不归档（expiration_date）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            valid_to: New boundary or None (explicit NULL allowed).
            valid_to：新边界或 None（允许显式 NULL）。

            at: updated_at instant.
            at：updated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms.
            更新 memory_atoms。
        """
        end = to_utc_iso(valid_to) if valid_to else None
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET valid_to = ?, updated_at = ? WHERE atom_id = ?", (end, to_utc_iso(at), atom_id)
            )
            self._queue_attrs(atom_id)

    def rebuild_fts(self) -> int:
        """Rebuild every atom_fts row from the stored atoms under the tokenizer mode in force (utils.text.cjk_tokens_scope).
        在当前生效的分词模式（utils.text.cjk_tokens_scope）下，从已存原子重建全部 atom_fts 行。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Rows written (0 when the linked SQLite has no FTS5).
            写入的行数（所链接的 SQLite 没有 FTS5 时为 0）。

        State effect / 状态影响：
            DELETE FROM atom_fts + schema.backfill_fts in one transaction inside unversioned() (§6.7): the keyword
            channel changes, but no scope write version moves and no history row is written. Called by HMS.__init__
            when a store was last written under another `text.cjk_tokens` mode.
            在 unversioned()（§6.7）内的一个事务里 DELETE FROM atom_fts + schema.backfill_fts：关键词通道改变，但不推进
            任何范围写入版本，也不写 history 行。当存储上次是在另一种 `text.cjk_tokens` 模式下写入时由 HMS.__init__ 调用。
        """
        if not self.capabilities.get("fts5"):
            return 0
        with self.unversioned(), self.transaction():
            self._execute("DELETE FROM atom_fts")
            return backfill_fts(self._connection())

    def _refresh_fts(self, atom_id: str) -> None:
        """Rewrite the atom_fts row of one live atom from its stored text, topics and entity keys.
        由已存文本、主题与实体键重写一个存活原子的 atom_fts 行。
        """
        atom = self.get_atom(atom_id)
        self._execute("DELETE FROM atom_fts WHERE atom_id = ?", (atom_id,))
        if atom is not None and atom.archived_reason is None:
            self._execute(
                "INSERT INTO atom_fts(atom_id, tokens) VALUES (?, ?)",
                (atom_id, fts_tokens(atom.text, atom.topic_tags, atom.entity_ids)),
            )

    def set_atom_confidence(self, atom_id: str, value: float, *, at: str) -> None:
        """Set an atom's confidence (clamped to [0, 1]).
        设置原子置信度（钳制到 [0, 1]）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            value: New confidence.
            value：新置信度。

            at: updated_at instant.
            at：updated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms.
            更新 memory_atoms。
        """
        clamped = min(1.0, max(0.0, float(value)))
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET confidence = ?, updated_at = ? WHERE atom_id = ?",
                (clamped, to_utc_iso(at), atom_id),
            )

    def set_atom_visibility(self, atom_id: str, visibility: str, *, at: str) -> None:
        """Set an atom's visibility.
        设置原子可见性。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            visibility: New Visibility value.
            visibility：新的 Visibility 值。

            at: updated_at instant.
            at：updated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms.
            更新 memory_atoms。
        """
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET visibility = ?, updated_at = ? WHERE atom_id = ?",
                (str(visibility), to_utc_iso(at), atom_id),
            )

    def set_user_meta(self, atom_id: str, user_meta: dict, *, at: str) -> None:
        """Replace an atom's caller metadata.
        替换原子的调用方元数据。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            user_meta: New metadata dict.
            user_meta：新的元数据字典。

            at: updated_at instant.
            at：updated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates memory_atoms.
            更新 memory_atoms。
        """
        with self.transaction():
            self._execute(
                "UPDATE memory_atoms SET user_meta_json = ?, updated_at = ? WHERE atom_id = ?",
                (canonical_json(dict(user_meta or {})), to_utc_iso(at), atom_id),
            )

    def add_atom_anchor(self, atom_id: str, anchor_id: str, *, role: str = "support") -> bool:
        """Link an extra witness anchor to an atom (INSERT OR IGNORE on the PK).
        为原子链接额外见证锚点（主键上 INSERT OR IGNORE）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            anchor_id: Anchor id (pre-checked).
            anchor_id：锚点 id（预检）。

            role: 'support' | 'feedback' ('primary' is set by the insert trigger).
            role：'support' | 'feedback'（'primary' 由插入触发器设置）。

        Output / 输出：
            True iff a row was inserted (an existing primary witness adds nothing).
            仅当插入了新行为 True（已是主见证时不新增）。

        State effect / 状态影响：
            Writes atom_anchor.
            写入 atom_anchor。
        """
        self._require_anchors([anchor_id], obj="atom", obj_id=atom_id)
        with self.transaction():
            cur = self._execute(
                "INSERT OR IGNORE INTO atom_anchor(atom_id, anchor_id, role) VALUES (?, ?, ?)",
                (atom_id, anchor_id, role),
            )
            return cur.rowcount == 1

    def dispute_atom_anchor(self, atom_id: str, anchor_id: str, *, at: str) -> None:
        """Mark an atom↔anchor link disputed (never deleted).
        标记原子与锚点的链接为有争议（绝不删除）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            anchor_id: Anchor id.
            anchor_id：锚点 id。

            at: Dispute instant.
            at：争议时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates atom_anchor.disputed_at.
            更新 atom_anchor.disputed_at。
        """
        with self.transaction():
            self._execute(
                "UPDATE atom_anchor SET disputed_at = ? WHERE atom_id = ? AND anchor_id = ?",
                (to_utc_iso(at), atom_id, anchor_id),
            )

    def atom_anchor_ids(self, atom_id: str, *, include_disputed: bool = False) -> list[str]:
        """Anchor ids of an atom, primary first then insertion order.
        原子的锚点 id，主锚点优先，其后按插入顺序。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            include_disputed: Include disputed links.
            include_disputed：包含有争议的链接。

        Output / 输出：
            Anchor id list.
            锚点 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return self._anchors_for([atom_id], include_disputed=include_disputed).get(atom_id, [])

    def atoms_for_anchor(self, anchor_id: str) -> list[str]:
        """Atom ids linked to an anchor (disputed links included), ascending.
        链接到某锚点的原子 id（含争议链接），升序。

        Input / 输入：
            anchor_id: Anchor id.
            anchor_id：锚点 id。

        Output / 输出：
            Sorted atom ids.
            排序后的原子 id。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(
            "SELECT atom_id FROM atom_anchor WHERE anchor_id = ? ORDER BY atom_id", (anchor_id,)
        ).fetchall()
        return [r["atom_id"] for r in rows]

    def axis_select(
        self,
        scope: Scope,
        *,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        as_of: str | None = None,
        include_expired: bool = False,
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        order: str = "time_anchor DESC, created_at DESC, atom_id ASC",
        limit: int | None = None,
        atom_ids: Sequence[str] | None = None,
        index_hint: str | None = None,
        trusted: bool = False,
    ) -> list[MemoryAtom]:
        """The axis gate: scope AND validity(as_of) AND caller WHERE, always as SQL, never a post-filter.
        轴门控：scope AND 有效性(as_of) AND 调用方 WHERE，始终为 SQL，绝不后过滤。

        Input / 输入：
            scope: Scope filter (Scope.matches semantics).
            scope：范围过滤（Scope.matches 语义）。

            where_sql: Extra WHERE fragment over memory_atoms (positional '?').
            where_sql：针对 memory_atoms 的额外 WHERE 片段（位置参数 '?'）。

            params: Parameters of where_sql.
            params：where_sql 的参数。

            joins: JOIN clauses appended after FROM memory_atoms.
            joins：追加在 FROM memory_atoms 之后的 JOIN 子句。

            as_of: Validity instant; defaults to now().
            as_of：有效性时刻；默认 now()。

            include_expired: Drop the whole validity predicate.
            include_expired：去掉整个有效性谓词。

            scale_constraint: scale_level whitelist (empty/None = no constraint).
            scale_constraint：scale_level 白名单（空/None 表示不限制）。

            atom_types: atom_type whitelist.
            atom_types：atom_type 白名单。

            order: ORDER BY expression (bare atom columns are qualified automatically).
            order：ORDER BY 表达式（裸原子列名自动加表前缀）。

            limit: Optional row cap.
            limit：可选的行数上限。

            atom_ids: Restrict to these ids (the pool hydration): a VALUES CTE drives a CROSS JOIN so memory_atoms is read by primary key per id, never scanned through the scope index.
            atom_ids：限定这些 id（池的水合）：VALUES CTE 驱动 CROSS JOIN，memory_atoms 按每个 id 的主键读取，绝不经 scope 索引扫描。

            index_hint: `INDEXED BY` name for memory_atoms (schema.TIME_INDEX under a hard time gate); ignored with atom_ids.
            index_hint：memory_atoms 的 `INDEXED BY` 索引名（硬时间门下为 schema.TIME_INDEX）；给了 atom_ids 时忽略。

            trusted: Build the models with model_construct (see _hydrate_atoms); the pool's final top_k rows use it.
            trusted：用 model_construct 构造模型（见 _hydrate_atoms）；池的最终 top_k 行使用它。

        Output / 输出：
            Hydrated atoms in the requested order.
            按请求顺序的水合原子。

        State effect / 状态影响：
            None.
            无。
        """
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        distinct = "DISTINCT " if joins else ""
        if atom_ids is not None:
            wanted = list(dict.fromkeys(str(i) for i in atom_ids))
            if not wanted:
                return []
            # The id list drives: SQLite keeps the CROSS JOIN order, so each id is one primary-key lookup even when
            # the gate carries an indexed scope predicate the planner would otherwise prefer to scan.
            # id 列表驱动：SQLite 保持 CROSS JOIN 的顺序，因此即使门控带有规划器本会偏好扫描的 scope 索引谓词，
            # 每个 id 也只是一次主键查找。
            marks = ", ".join("(?)" for _ in wanted)
            sql = (
                f"WITH ids(atom_id) AS (VALUES {marks}) SELECT {distinct}memory_atoms.* FROM ids CROSS JOIN memory_atoms "
                f"ON memory_atoms.atom_id = ids.atom_id {' '.join(joins)} "
                f"WHERE {' AND '.join(clauses)} ORDER BY {_qualify(order or DEFAULT_ATOM_ORDER)}"
            )
            values = wanted + values
        else:
            sql = (
                f"SELECT {distinct}memory_atoms.* FROM memory_atoms {_indexed_by(index_hint)} {' '.join(joins)} "
                f"WHERE {' AND '.join(clauses)} ORDER BY {_qualify(order or DEFAULT_ATOM_ORDER)}"
            )
        if limit is not None:
            sql += " LIMIT ?"
            values.append(int(limit))
        return self._hydrate_atoms(self._execute(sql, values).fetchall(), trusted=trusted)

    def axis_lite(
        self,
        scope: Scope,
        atom_ids: Sequence[str],
        *,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        as_of: str | None = None,
        include_expired: bool = False,
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
    ) -> list[AtomLite]:
        """The scoring rows of some ids under the axis gate (pool ∩ G): two statements per 500 ids, no model built.
        轴门控下若干 id 的打分行（池 ∩ G）：每 500 个 id 两条语句，不构造模型。

        Input / 输入：
            scope, where_sql, params, joins, as_of, include_expired, scale_constraint, atom_types: As axis_select.
            scope、where_sql、params、joins、as_of、include_expired、scale_constraint、atom_types：同 axis_select。

            atom_ids: The pool ids (a VALUES CTE drives primary-key lookups, as axis_select(atom_ids=...)).
            atom_ids：池的 id（VALUES CTE 驱动主键查找，同 axis_select(atom_ids=...)）。

        Output / 输出：
            AtomLite rows in axis order (time_anchor DESC, created_at DESC, atom_id ASC) with entity_ids (sorted),
            topic_tags (insertion order), n_anchors (undisputed count; every witness disputed → all; none → 1, exactly
            len(anchor_ids) of the hydrated atom) and age_us (None when the instant is not the 32-char form).
            按轴序（time_anchor 降序，created_at 降序，atom_id 升序）的 AtomLite 行，含 entity_ids（排序）、topic_tags
            （插入顺序）、n_anchors（无争议数；全部争议 → 全部；没有 → 1，恰为水合原子的 len(anchor_ids)）与 age_us
            （时刻不是 32 字符形式时为 None）。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        if not wanted:
            return []
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        instant = parse_iso(as_of) if as_of else parse_iso(self.now())
        age_params = [timegm(instant.utctimetuple()), instant.microsecond]
        distinct = "DISTINCT " if joins else ""
        out: list[AtomLite] = []
        for chunk in _chunks(wanted):
            marks = ", ".join("(?)" for _ in chunk)
            rows = self._execute(
                f"WITH ids(atom_id) AS (VALUES {marks}) SELECT {distinct}{_LITE_COLS}, {_AGE_US_SQL} "
                f"FROM ids CROSS JOIN memory_atoms ON memory_atoms.atom_id = ids.atom_id {' '.join(joins)} "
                f"WHERE {' AND '.join(clauses)} ORDER BY {_qualify(DEFAULT_ATOM_ORDER)}",
                list(chunk) + age_params + values,
            ).fetchall()
            if not rows:
                continue
            # Positional columns in _LITE_COLS order (+ age_us): no by-name Row access on the hot path.
            # 按 _LITE_COLS 顺序（+ age_us）的位置列：热路径上不做按名的 Row 访问。
            tuples = [tuple(r) for r in rows]
            entities, topics, counts = self._link_aggregates([t[0] for t in tuples])
            for t in tuples:
                atom_id = t[0]
                undisputed, total = counts.get(atom_id, (0, 0))
                age = t[12]
                out.append(
                    AtomLite(
                        *t[:12],
                        entities.get(atom_id, []),
                        topics.get(atom_id, []),
                        undisputed or total or 1,
                        int(age) if age is not None else None,
                    )
                )
        if len(wanted) > _IN_CHUNK:
            out.sort(key=lambda a: a.atom_id)
            out.sort(key=lambda a: (a.time_anchor or "", a.created_at or ""), reverse=True)
        return out

    def _link_aggregates(
        self, atom_ids: Sequence[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, tuple[int, int]]]:
        """One statement for the scoring links of some ids: (entity keys sorted, topics in order, (undisputed, total) anchors).
        一条语句取若干 id 的打分链接：（排序后的实体键、按顺序的主题、(无争议, 全部) 锚点计数）。
        """
        entities: dict[str, list[str]] = {}
        topics: dict[str, list[tuple[int, str]]] = {}
        counts: dict[str, tuple[int, int]] = {}
        rows = self._execute(
            f"WITH ids(atom_id) AS (VALUES {','.join('(?)' for _ in atom_ids)}) "
            "SELECT 0 AS kind, ae.atom_id AS atom_id, e.normalized_text AS value, 0 AS ord "
            "FROM ids JOIN atom_entity ae ON ae.atom_id = ids.atom_id JOIN entities e ON e.entity_id = ae.entity_id "
            "UNION ALL SELECT 1, t.atom_id, t.topic, t.rowid FROM ids JOIN atom_topic t ON t.atom_id = ids.atom_id "
            "UNION ALL SELECT 2, a.atom_id, NULL, SUM(a.disputed_at IS NULL) * 1000000 + COUNT(*) "
            "FROM ids JOIN atom_anchor a ON a.atom_id = ids.atom_id GROUP BY a.atom_id",
            list(atom_ids),
        ).fetchall()
        for row in rows:
            kind, atom_id, value, ord_ = tuple(row)
            if kind == 0:
                entities.setdefault(atom_id, []).append(value)
            elif kind == 1:
                topics.setdefault(atom_id, []).append((ord_, value))
            else:
                packed = int(ord_)
                counts[atom_id] = (packed // 1000000, packed % 1000000)
        return (
            {k: sorted(dict.fromkeys(v)) for k, v in entities.items()},
            {k: [topic for _ord, topic in sorted(v)] for k, v in topics.items()},
            counts,
        )

    def _axis_where(
        self,
        scope: Scope,
        *,
        where_sql: str = "",
        params: Sequence = (),
        as_of: str | None = None,
        include_expired: bool = False,
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        obs_types: Sequence[str] | None = None,
    ) -> tuple[list[str], list[Any]]:
        """The axis gate as (clauses, params): scope AND validity(as_of) AND caller WHERE AND whitelists.
        以 (子句, 参数) 表示的轴门控：scope AND 有效性(as_of) AND 调用方 WHERE AND 白名单。
        """
        where, values = _scope_where(scope, "memory_atoms.")
        clauses = [where]
        if not include_expired:
            live, live_params = self._live(as_of)
            clauses.append(live)
            values.extend(live_params)
        if where_sql and where_sql.strip():
            clauses.append(f"({where_sql})")
            values.extend(list(params))
        if scale_constraint:
            scales = [str(s) for s in scale_constraint]
            clauses.append(f"memory_atoms.scale_level IN ({_placeholders(len(scales))})")
            values.extend(scales)
        if atom_types:
            types = [str(t) for t in atom_types]
            clauses.append(f"memory_atoms.atom_type IN ({_placeholders(len(types))})")
            values.extend(types)
        if obs_types:
            kinds = [str(t) for t in obs_types]
            clauses.append(f"memory_atoms.obs_type IN ({_placeholders(len(kinds))})")
            values.extend(kinds)
        return clauses, values

    def axis_ids(
        self,
        scope: Scope,
        *,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        as_of: str | None = None,
        include_expired: bool = False,
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        order: str | None = "time_anchor DESC, created_at DESC, atom_id ASC",
        limit: int | None = None,
        index_hint: str | None = None,
    ) -> list[str]:
        """The axis gate of axis_select as ids only: no hydration, one SELECT of memory_atoms.atom_id.
        axis_select 的轴门控只取 id：不水合，一条只选 memory_atoms.atom_id 的 SELECT。

        Input / 输入：
            scope, where_sql, params, joins, as_of, include_expired, scale_constraint, atom_types, limit: As axis_select.
            scope、where_sql、params、joins、as_of、include_expired、scale_constraint、atom_types、limit：同 axis_select。

            order: ORDER BY expression as axis_select, or None for NO ORDER BY (index order; a `limit` then stops the scan early instead of sorting the whole gated set — the pool probe and the semantic channel's G).
            order：同 axis_select 的 ORDER BY 表达式，或 None 表示**不排序**（索引序；此时 `limit` 让扫描提前停止，而不是对整个门控集合排序——池探测与语义通道的 G 使用它）。

            index_hint: `INDEXED BY` name for memory_atoms (schema.TIME_INDEX under a hard time gate).
            index_hint：memory_atoms 的 `INDEXED BY` 索引名（硬时间门下为 schema.TIME_INDEX）。

        Output / 输出：
            Atom ids in the requested order (the gated set G of the retrieval pool).
            按请求顺序的原子 id（检索池的门控集合 G）。

        State effect / 状态影响：
            None.
            无。
        """
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        distinct = "DISTINCT " if joins else ""
        sql = (
            f"SELECT {distinct}memory_atoms.atom_id FROM memory_atoms {_indexed_by(index_hint)} {' '.join(joins)} "
            f"WHERE {' AND '.join(clauses)}"
        )
        if order is not None:
            sql += f" ORDER BY {_qualify(order or DEFAULT_ATOM_ORDER)}"
        if limit is not None:
            sql += " LIMIT ?"
            values.append(int(limit))
        return [r["atom_id"] for r in self._execute(sql, values).fetchall()]

    def count_axis(
        self,
        scope: Scope,
        *,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        as_of: str | None = None,
        include_expired: bool = False,
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        index_hint: str | None = None,
    ) -> int:
        """COUNT(*) of the axis gate (COUNT(DISTINCT atom_id) when joins are present).
        轴门控的 COUNT(*)（有 joins 时为 COUNT(DISTINCT atom_id)）。

        Input / 输入：
            scope, where_sql, params, joins, as_of, include_expired, scale_constraint, atom_types, index_hint: As axis_select.
            scope、where_sql、params、joins、as_of、include_expired、scale_constraint、atom_types、index_hint：同 axis_select。

        Output / 输出：
            The number of gated atoms (|G|), which decides whether a hard gate enters the pool whole.
            门控原子数（|G|），决定硬门是否整体入池。

        State effect / 状态影响：
            None.
            无。
        """
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        counted = "COUNT(DISTINCT memory_atoms.atom_id)" if joins else "COUNT(*)"
        row = self._execute(
            f"SELECT {counted} AS n FROM memory_atoms {_indexed_by(index_hint)} {' '.join(joins)} "
            f"WHERE {' AND '.join(clauses)}",
            values,
        ).fetchone()
        return int(row["n"])

    def keyword_candidates(
        self,
        scope: Scope,
        tokens: Sequence[str],
        *,
        as_of: str | None = None,
        include_expired: bool = False,
        limit: int,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        index_hint: str | None = None,
    ) -> list[tuple[str, float]]:
        """The keyword recall channel: the `limit` best-matching live atoms of the gate for some query tokens.
        关键词召回通道：门控内若干查询词元的 `limit` 个最佳匹配存活原子。

        Two FTS5 phases, strict then loose: atoms matching EVERY token first, then atoms matching ANY token fill the
        remainder. Each phase reads a WINDOW of the newest matches (`ORDER BY rowid DESC LIMIT max(4·need, 64)`, a
        reverse posting-list walk that stops early instead of ranking every match — bm25 over 6000 matches costs
        4–8 ms at N=30000, RETRIEVAL_BENCH §3), re-applies the gate to the window and orders the survivors by
        (bm25 rank, time_anchor DESC, atom_id ASC); the window grows (offset paging, doubling) until `limit` gated
        rows are found or the phase is exhausted.
        两个 FTS5 阶段，先严后宽：先取匹配**全部**词元的原子，再由匹配**任一**词元的原子补足。每个阶段读取最新匹配的
        一个**窗口**（`ORDER BY rowid DESC LIMIT max(4·need, 64)`，倒序遍历倒排表并提前停止，而不是给每个匹配排名——
        N=30000 时 6000 个匹配的 bm25 排序要 4–8 ms，RETRIEVAL_BENCH §3），对窗口重新施加门控，并按
        （bm25 rank，time_anchor 降序，atom_id 升序）排列幸存者；窗口逐步扩大（offset 分页、倍增）直到找到 `limit` 个
        门控内的行或该阶段耗尽。

        Input / 输入：
            scope: Scope filter (Scope.matches semantics).
            scope：范围过滤（Scope.matches 语义）。

            tokens: Query tokens (content tokens + identifier tokens, already stemmed like the atom vocabulary).
            tokens：查询词元（内容词元 + 标识符词元，已与原子词表同样词干化）。

            as_of: Validity instant; defaults to now().
            as_of：有效性时刻；默认 now()。

            include_expired: Drop the validity predicate (archived versions carry no atom_fts row either way).
            include_expired：去掉有效性谓词（已归档版本无论如何都没有 atom_fts 行）。

            limit: Channel bound K_kw.
            limit：通道上限 K_kw。

            where_sql, params, joins, scale_constraint, atom_types: The level gate, as axis_select (the channel searches inside G, not the scope).
            where_sql、params、joins、scale_constraint、atom_types：该级别的门控，同 axis_select（通道在 G 内而非 scope 内搜索）。

            index_hint: Accepted for signature parity with axis_ids and ignored (memory_atoms is read by primary key here).
            index_hint：为与 axis_ids 签名一致而接受，忽略（这里 memory_atoms 按主键读取）。

        Output / 输出：
            `[(atom_id, score), ...]` in recall order: strict-phase rows (score = -bm25, DESC, then time_anchor DESC,
            atom_id ASC within a window) then loose-phase rows likewise; under the LIKE fallback
            (capabilities['fts5'] False) score = matched-token count, ordered (score DESC, axis order).
            按召回顺序的 `[(atom_id, score), ...]`：先是严格阶段的行（score = -bm25，降序，窗口内再按 time_anchor 降序、
            atom_id 升序），再是宽松阶段的行；LIKE 退路（capabilities['fts5'] 为 False）下 score = 匹配词元数，
            按（score 降序，轴序）。

        State effect / 状态影响：
            None.
            无。
        """
        terms = [t for t in dict.fromkeys(str(t).strip() for t in tokens) if t]
        if limit <= 0 or not terms:
            return []
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        gate = " AND ".join(clauses)
        join_sql = " ".join(joins)
        if self.capabilities.get("fts5"):
            out: list[tuple[str, float]] = []
            seen: set[str] = set()
            phases = [fts_match_query(terms, all_terms=True)] if len(terms) > 1 else []
            phases.append(fts_match_query(terms))
            for match in phases:
                if len(out) >= limit:
                    break
                out.extend(self._fts_phase(match, gate, join_sql, values, need=limit - len(out), seen=seen))
            return out[:limit]
        # Bounded fallback: LIKE per token (ASCII case-insensitive), ranked by matched-token count then recency.
        # 有界退路：每个词元一个 LIKE（ASCII 大小写不敏感），按匹配词元数、再按新近度排序。
        likes = ["(memory_atoms.text LIKE ? ESCAPE '\\')" for _t in terms]
        score = " + ".join(f"(CASE WHEN {like} THEN 1 ELSE 0 END)" for like in likes)
        distinct = "DISTINCT " if joins else ""
        sql = (
            f"SELECT {distinct}memory_atoms.atom_id AS atom_id, ({score}) AS score FROM memory_atoms {join_sql} "
            f"WHERE {gate} AND ({' OR '.join(likes)}) "
            "ORDER BY score DESC, memory_atoms.time_anchor DESC, memory_atoms.created_at DESC, memory_atoms.atom_id ASC "
            "LIMIT ?"
        )
        patterns = [_like_pattern(t) for t in terms]
        rows = self._execute(sql, patterns + values + patterns + [int(limit)]).fetchall()
        return [(r["atom_id"], float(r["score"])) for r in rows]

    @staticmethod
    def _fts_window_sql(gate: str, *, label: str) -> tuple[str, str]:
        """The gated window of one FTS5 phase as (materialized CTE, UNION ALL branch): every window row with its `live` flag.
        一个 FTS5 阶段的门控窗口，作为（物化 CTE，UNION ALL 分支）：窗口的每一行连同其 `live` 标志。

        The FTS5 read (`ORDER BY rowid DESC LIMIT ? OFFSET ?`) is a MATERIALIZED CTE: SQLite would otherwise push
        the join into the virtual table and fail on `rank` ("datatype mismatch"). The CTE drives a LEFT JOIN so the
        reader learns the window's true size (offset paging needs it) while the gate is evaluated as an expression;
        the filter-DSL joins are not supported here.
        FTS5 读取（`ORDER BY rowid DESC LIMIT ? OFFSET ?`）是一个 MATERIALIZED CTE：否则 SQLite 会把连接推进虚拟表并在
        `rank` 上失败（"datatype mismatch"）。CTE 驱动 LEFT JOIN，读取方因此知道窗口的真实大小（offset 分页需要它），
        门控则作为表达式求值；此处不支持过滤 DSL 的连接。
        """
        cte = (
            f"fts_{label} AS MATERIALIZED (SELECT atom_id, rank FROM atom_fts WHERE atom_fts MATCH ? "
            "ORDER BY rowid DESC LIMIT ? OFFSET ?)"
        )
        branch = (
            f"SELECT '{label}' AS ch, f.atom_id AS atom_id, f.rank AS score, memory_atoms.time_anchor AS time_anchor, "
            f"CASE WHEN memory_atoms.atom_id IS NOT NULL AND ({gate}) THEN 1 ELSE 0 END AS live "
            f"FROM fts_{label} f LEFT JOIN memory_atoms ON memory_atoms.atom_id = f.atom_id"
        )
        return cte, branch

    @staticmethod
    def _fts_rank(rows: list[tuple[str, float, str | None]]) -> list[tuple[str, float]]:
        """Order gated window rows by (bm25 rank ASC, time_anchor DESC nulls last, atom_id ASC) → (atom_id, -rank).
        把门控窗口行按（bm25 rank 升序，time_anchor 降序、空值最后，atom_id 升序）排列 → (atom_id, -rank)。
        """
        ordered = sorted(rows, key=lambda r: r[0])
        ordered.sort(key=lambda r: r[2] or "", reverse=True)
        ordered.sort(key=lambda r: r[1])
        return [(atom_id, -float(rank)) for atom_id, rank, _t in ordered]

    def _fts_phase(
        self,
        match: str,
        gate: str,
        join_sql: str,
        values: list[Any],
        *,
        need: int,
        seen: set[str],
        first: tuple[list[tuple[str, float, str | None, int]], int] | None = None,
    ) -> list[tuple[str, float]]:
        """One FTS5 phase: newest-match windows, the gate re-applied per window, grown until `need` rows are found.
        一个 FTS5 阶段：最新匹配的窗口，逐窗口重新施加门控，扩大直到找到 `need` 行。

        `first` hands in the rows (atom_id, rank, time_anchor, live) and the size of a window recall_channels already
        fetched, so paging continues after it instead of reading it again.
        `first` 传入 recall_channels 已取回的窗口行 (atom_id, rank, time_anchor, live) 及其大小，分页从它之后继续，
        而不是再读一次。
        """
        out: list[tuple[str, float]] = []
        window = max(4 * need, 64)
        offset = 0
        while len(out) < need:
            if first is not None:
                fetched, window = first
                first = None
                ranked = fetched
            elif join_sql:
                # A filter-DSL join cannot ride the LEFT JOIN branch: the window is gated through the CROSS JOIN
                # form (primary-key lookups) and its size is known from the FTS5 read itself.
                # 过滤 DSL 的连接无法搭 LEFT JOIN 分支：窗口经 CROSS JOIN 形式（主键查找）门控，其大小由 FTS5 读取本身给出。
                fts = self._execute(
                    "SELECT atom_id, rank FROM atom_fts WHERE atom_fts MATCH ? ORDER BY rowid DESC LIMIT ? OFFSET ?",
                    [match, window, offset],
                ).fetchall()
                ranks = {r["atom_id"]: float(r["rank"]) for r in fts}
                ranked = []
                for chunk in _chunks(list(ranks)):
                    marks = ", ".join("(?)" for _ in chunk)
                    for r in self._execute(
                        f"WITH ids(atom_id) AS (VALUES {marks}) SELECT DISTINCT memory_atoms.atom_id AS atom_id, "
                        "memory_atoms.time_anchor AS time_anchor FROM ids "
                        f"CROSS JOIN memory_atoms ON memory_atoms.atom_id = ids.atom_id {join_sql} WHERE {gate}",
                        list(chunk) + values,
                    ).fetchall():
                        ranked.append((r["atom_id"], ranks[r["atom_id"]], r["time_anchor"], 1))
                gated = {r[0] for r in ranked}
                ranked.extend((atom_id, 0.0, None, 0) for atom_id in ranks if atom_id not in gated)
            else:
                cte, branch = self._fts_window_sql(gate, label="kw")
                ranked = [
                    (t[1], float(t[2]), t[3], int(t[4]))
                    for t in map(tuple, self._execute(f"WITH {cte} {branch}", [match, window, offset] + values).fetchall())
                ]
            live = [(atom_id, rank, t) for atom_id, rank, t, flag in ranked if flag and atom_id not in seen]
            for atom_id, score in self._fts_rank(live):
                if atom_id not in seen:
                    seen.add(atom_id)
                    out.append((atom_id, score))
                    if len(out) >= need:
                        break
            if len(ranked) < window:
                break
            offset += window
            window *= 2
        return out

    def entity_candidates(
        self,
        scope: Scope,
        entity_keys: Sequence[str],
        *,
        as_of: str | None = None,
        include_expired: bool = False,
        limit: int | None,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        index_hint: str | None = None,
        order: str | None = "time_anchor DESC, atom_id ASC",
    ) -> list[str]:
        """The entity recall channel: the `limit` most recent gated atoms linked to any of the entity keys.
        实体召回通道：链接到任一实体键的 `limit` 个最新门控内原子。

        Input / 输入：
            scope: Scope filter; entity identity uses its user_id / agent_id like resolve_entities.
            scope：范围过滤；实体身份同 resolve_entities 使用其 user_id / agent_id。

            entity_keys: Normalized entity keys (entities.normalized_text).
            entity_keys：规范化实体键（entities.normalized_text）。

            as_of: Validity instant; defaults to now().
            as_of：有效性时刻；默认 now()。

            include_expired: Drop the validity predicate.
            include_expired：去掉有效性谓词。

            limit: Channel bound K_ent; None → every linked gated atom (the gated set G of a hard entity gate).
            limit：通道上限 K_ent；None → 全部链接的门控内原子（硬实体门的门控集合 G）。

            where_sql, params, joins, scale_constraint, atom_types: The level gate, as axis_select.
            where_sql、params、joins、scale_constraint、atom_types：该级别的门控，同 axis_select。

            index_hint: Accepted for signature parity with axis_ids and ignored (atom_entity drives the join).
            index_hint：为与 axis_ids 签名一致而接受，忽略（由 atom_entity 驱动连接）。

            order: ORDER BY over memory_atoms columns, or None for NO ORDER BY (link order; a `limit` then stops after that many gated links instead of sorting every link — the hard entity gate's pool probe).
            order：针对 memory_atoms 列的 ORDER BY，或 None 表示**不排序**（链接顺序；此时 `limit` 在找到那么多门控内链接后停止，而不是对全部链接排序——硬实体门的池探测）。

        Output / 输出：
            Atom ids ordered (time_anchor DESC, atom_id ASC), duplicates dropped in Python (an atom linked to several
            of the keys is fetched once per link, so `limit` over-fetches by the number of keys); the entity ids are
            resolved through ux_entities_norm_scope first, then atom_entity drives the join (ix_entity_atom →
            memory_atoms by primary key), so the cost is O(links of the keys), never a scan of the scope.
            按（time_anchor 降序，atom_id 升序）的原子 id，在 Python 中去重（链接到多个键的原子每个链接取回一次，
            因此 `limit` 按键数过取）；先经 ux_entities_norm_scope 解析实体 id，再由 atom_entity 驱动连接
            （ix_entity_atom → memory_atoms 主键），因此成本是 O(这些键的链接数)，绝不扫描 scope。

        State effect / 状态影响：
            None.
            无。
        """
        keys = [k for k in dict.fromkeys(str(k) for k in entity_keys) if k]
        if (limit is not None and limit <= 0) or not keys:
            return []
        entity_ids = sorted(self.resolve_entities(scope, keys).values())
        if not entity_ids:
            return []
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        # CROSS JOIN pins the loop order (SQLite honours it): atom_entity by ix_entity_atom drives, memory_atoms is
        # looked up by primary key — the planner would otherwise scan the scope and probe atom_entity per row.
        # CROSS JOIN 钉住循环顺序（SQLite 遵守它）：atom_entity 经 ix_entity_atom 驱动，memory_atoms 按主键查找——
        # 否则规划器会扫描整个 scope 并逐行探测 atom_entity。
        # No DISTINCT (its temp b-tree costs more than the links): each atom appears once per matching link, so a
        # limit over-fetches by len(entity_ids) and Python keeps the first occurrence in order.
        # 不用 DISTINCT（它的临时 b-tree 比链接本身更贵）：每个原子按匹配链接数出现，因此 limit 按 len(entity_ids)
        # 过取，Python 按顺序保留首次出现。
        sql = (
            "SELECT memory_atoms.atom_id AS atom_id "
            f"FROM atom_entity ae CROSS JOIN memory_atoms ON memory_atoms.atom_id = ae.atom_id {' '.join(joins)} "
            f"WHERE ae.entity_id IN ({_placeholders(len(entity_ids))}) AND {' AND '.join(clauses)}"
        )
        if order is not None:
            sql += f" ORDER BY {_qualify(order)}"
        bound = list(entity_ids) + values
        if limit is not None:
            sql += " LIMIT ?"
            bound.append(int(limit) * len(entity_ids))
        rows = self._execute(sql, bound).fetchall()
        ids = list(dict.fromkeys(r["atom_id"] for r in rows))
        return ids if limit is None else ids[: int(limit)]

    @staticmethod
    def entity_gate_sql(scope: Scope, entity_keys: Sequence[str], *, correlated: bool = False) -> tuple[str, list[Any]]:
        """The hard entity gate over memory_atoms as (clause, params): IN-list for scans, correlated EXISTS for id-driven reads.
        memory_atoms 上的硬实体门，以（子句，参数）表示：扫描用 IN 列表，按 id 驱动的读取用相关 EXISTS。

        Input / 输入：
            scope: The query scope; entity identity uses its user_id / agent_id like resolve_entities.
            scope：查询范围；实体身份同 resolve_entities 使用其 user_id / agent_id。

            entity_keys: Normalized entity keys of the gate.
            entity_keys：门控的规范化实体键。

            correlated: False → `atom_id IN (SELECT ae.atom_id …)`, materialized ONCE per statement (a hash probe per scanned row: right for a scope / time-index walk); True → `EXISTS (SELECT 1 FROM atom_entity ae WHERE ae.atom_id = memory_atoms.atom_id …)`, one primary-key probe per candidate row and nothing materialized (right for a VALUES-CTE / FTS-window read of a few hundred ids, where the list would cost more than the rows).
            correlated：False → `atom_id IN (SELECT ae.atom_id …)`，每条语句物化**一次**（每个扫描行一次哈希探测：适合 scope / 时间索引遍历）；True → `EXISTS (SELECT 1 FROM atom_entity ae WHERE ae.atom_id = memory_atoms.atom_id …)`，每个候选行一次主键探测、不物化任何东西（适合 VALUES-CTE / FTS 窗口对几百个 id 的读取，此时列表比行本身更贵）。

        Output / 输出：
            (clause, params); both forms select exactly the filter DSL's `entity IN` set.
            （子句，参数）；两种形式选出的集合与过滤 DSL 的 `entity IN` 完全相同。

        State effect / 状态影响：
            None.
            无。
        """
        keys = [str(k) for k in entity_keys]
        identity = ""
        identity_params: list[Any] = []
        for column in ("user_id", "agent_id"):
            value = getattr(scope, column)
            if value is not None:
                identity += f" AND COALESCE(e.{column},'') = COALESCE(?, '')"
                identity_params.append(value)
        entities = f"SELECT e.entity_id FROM entities e WHERE e.normalized_text IN ({_placeholders(len(keys))}){identity}"
        if correlated:
            clause = (
                "EXISTS (SELECT 1 FROM atom_entity ae WHERE ae.atom_id = memory_atoms.atom_id "
                f"AND ae.entity_id IN ({entities}))"
            )
        else:
            clause = f"memory_atoms.atom_id IN (SELECT ae.atom_id FROM atom_entity ae WHERE ae.entity_id IN ({entities}))"
        return clause, keys + identity_params

    def recall_channels(
        self,
        scope: Scope,
        *,
        as_of: str | None = None,
        include_expired: bool = False,
        where_sql: str = "",
        params: Sequence = (),
        joins: Sequence[str] = (),
        scale_constraint: Sequence[str] | None = None,
        atom_types: Sequence[str] | None = None,
        index_hint: str | None = None,
        keyword: tuple[Sequence[str], int] | None = None,
        entity: tuple[Sequence[str], int] | None = None,
        recency: int | None = None,
        links: Sequence[str] | None = None,
        entity_gate: Sequence[str] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """The SQL recall channels of one ladder level in ONE UNION ALL statement, each branch with its own LIMIT.
        一个阶梯级别的 SQL 召回通道合成**一条** UNION ALL 语句，每个分支各带自己的 LIMIT。

        Input / 输入：
            scope, as_of, include_expired, where_sql, params, joins, scale_constraint, atom_types: The level gate, as axis_select (WITHOUT the hard entity gate, which travels as `entity_gate`).
            scope、as_of、include_expired、where_sql、params、joins、scale_constraint、atom_types：该级别的门控，同 axis_select（**不含**硬实体门，它以 `entity_gate` 传入）。

            entity_gate: Entity keys of the level's hard entity gate, or None: the recency branch adds entity_gate_sql(correlated=False), the keyword window adds the correlated form, and the atom_entity-driven branches (entity / links, whose keys are then the gate keys) add nothing.
            entity_gate：该级别硬实体门的实体键，或 None：新近度分支加 entity_gate_sql(correlated=False)，关键词窗口加相关形式，由 atom_entity 驱动的分支（entity / links，此时其键就是门控键）不加任何东西。

            index_hint: `INDEXED BY` name for the recency branch; None without joins / WHERE → schema.TIME_INDEX (the axis order walks ix_atoms_time backwards and stops after K rows instead of sorting the scope).
            index_hint：新近度分支的 `INDEXED BY` 索引名；无 joins / WHERE 时 None → schema.TIME_INDEX（轴序倒着走 ix_atoms_time，K 行后停止，而不是对整个 scope 排序）。

            keyword: (query tokens, K_kw) → the keyword channel of keyword_candidates (first windows of both FTS5 phases ride this statement; paging and the LIKE / join forms run through keyword_candidates' own statements).
            keyword：（查询词元，K_kw）→ keyword_candidates 的关键词通道（两个 FTS5 阶段的首个窗口搭本语句；分页与 LIKE / 连接形式走 keyword_candidates 自己的语句）。

            entity: (entity keys, K_ent) → the entity channel of entity_candidates (time order, bounded).
            entity：（实体键，K_ent）→ entity_candidates 的实体通道（时间序、有界）。

            recency: K_rec → the recency channel (axis order, bounded).
            recency：K_rec → 新近度通道（轴序、有界）。

            links: Entity keys whose UNGATED atom_entity links are wanted (the semantic channel's candidate list under a hard entity gate; the provider applies the gate as its pushed-down filter).
            links：需要其**未门控** atom_entity 链接的实体键（硬实体门下语义通道的候选列表；提供者把门控作为下推过滤器施加）。

        Output / 输出：
            {'keyword': [(atom_id, -bm25)], 'entity': [(atom_id, 0.0)], 'recency': [(atom_id, 0.0)], 'links': [(atom_id, 0.0)]}
            for the requested channels only, each in its channel order and bit-identical to keyword_candidates /
            entity_candidates / axis_ids(order=AXIS_ORDER) over the same gate.
            仅含所请求通道的 {'keyword': [(atom_id, -bm25)], 'entity': [(atom_id, 0.0)], 'recency': [(atom_id, 0.0)],
            'links': [(atom_id, 0.0)]}，各按其通道顺序，并与同一门控上的 keyword_candidates / entity_candidates /
            axis_ids(order=AXIS_ORDER) 逐位相同。

        State effect / 状态影响：
            None.
            无。
        """
        clauses, values = self._axis_where(
            scope,
            where_sql=where_sql,
            params=params,
            as_of=as_of,
            include_expired=include_expired,
            scale_constraint=scale_constraint,
            atom_types=atom_types,
        )
        gate = " AND ".join(clauses)
        keyed_gate, keyed_values = gate, values
        scan_gate, scan_values = gate, values
        if entity_gate:
            exists, exists_params = self.entity_gate_sql(scope, entity_gate, correlated=True)
            keyed_gate, keyed_values = f"{gate} AND {exists}", values + exists_params
            listed, listed_params = self.entity_gate_sql(scope, entity_gate, correlated=False)
            scan_gate, scan_values = f"{gate} AND {listed}", values + listed_params
        join_sql = " ".join(joins)
        distinct = "DISTINCT " if joins else ""
        ctes: list[str] = []
        cte_params: list[Any] = []
        branches: list[str] = []
        bound: list[Any] = []
        out: dict[str, list[tuple[str, float]]] = {}
        kw_terms: list[str] = []
        kw_phases: list[tuple[str, str]] = []
        kw_window = 0
        keys: list[str] = []
        entity_ids: list[str] = []
        if keyword is not None:
            kw_terms = [t for t in dict.fromkeys(str(t).strip() for t in keyword[0]) if t]
            if kw_terms and keyword[1] > 0 and self.capabilities.get("fts5") and not joins:
                kw_window = max(4 * int(keyword[1]), 64)
                if len(kw_terms) > 1:
                    kw_phases.append((_KW_STRICT, fts_match_query(kw_terms, all_terms=True)))
                kw_phases.append((_KW_LOOSE, fts_match_query(kw_terms)))
                # Only the first phase rides the statement: when it fills K_kw (every term co-occurs) the loose
                # window would be a wasted bm25 read; when it is empty its read costs nothing and the loose phase
                # follows through _fts_phase.
                # 只有第一个阶段搭本语句：它填满 K_kw（各词元共现）时宽松窗口只是浪费的 bm25 读取；它为空时读取
                # 几乎不花钱，宽松阶段随后经 _fts_phase 执行。
                label, match = kw_phases[0]
                cte, branch = self._fts_window_sql(keyed_gate, label=label)
                ctes.append(cte)
                cte_params.extend([match, kw_window, 0])
                branches.append(branch)
                bound.extend(keyed_values)
        if entity is not None and int(entity[1]) > 0:
            keys = [k for k in dict.fromkeys(str(k) for k in entity[0]) if k]
            entity_ids = sorted(self.resolve_entities(scope, keys).values()) if keys else []
            if entity_ids:
                # As entity_candidates: atom_entity drives (ix_entity_atom), memory_atoms by primary key, no
                # DISTINCT (Python keeps first occurrences), the limit over-fetched by the number of keys. The
                # entity gate is implied when the channel keys are the gate keys, else it rides as the scan form.
                # 同 entity_candidates：atom_entity 驱动（ix_entity_atom），memory_atoms 按主键，不用 DISTINCT
                # （Python 保留首次出现），limit 按键数过取。通道键就是门控键时实体门已隐含，否则以扫描形式附上。
                branch_gate, branch_values = gate, values
                if entity_gate and set(keys) - set(str(k) for k in entity_gate):
                    branch_gate, branch_values = scan_gate, scan_values
                branches.append(
                    "SELECT 'entity' AS ch, atom_id, 0.0 AS score, time_anchor, 1 AS live FROM ("
                    "SELECT memory_atoms.atom_id AS atom_id, memory_atoms.time_anchor AS time_anchor "
                    f"FROM atom_entity ae CROSS JOIN memory_atoms ON memory_atoms.atom_id = ae.atom_id {join_sql} "
                    f"WHERE ae.entity_id IN ({_placeholders(len(entity_ids))}) AND {branch_gate} "
                    "ORDER BY memory_atoms.time_anchor DESC, memory_atoms.atom_id ASC LIMIT ?)"
                )
                bound.extend([*entity_ids, *branch_values, int(entity[1]) * len(entity_ids)])
            else:
                out["entity"] = []
        if recency is not None and int(recency) > 0:
            hint = index_hint if (index_hint or joins or (where_sql and where_sql.strip())) else TIME_INDEX
            branches.append(
                "SELECT 'recency' AS ch, atom_id, 0.0 AS score, time_anchor, 1 AS live FROM ("
                f"SELECT {distinct}memory_atoms.atom_id AS atom_id, memory_atoms.time_anchor AS time_anchor "
                f"FROM memory_atoms {_indexed_by(hint)} {join_sql} WHERE {scan_gate} "
                f"ORDER BY {_qualify(DEFAULT_ATOM_ORDER)} LIMIT ?)"
            )
            bound.extend([*scan_values, int(recency)])
        if links is not None:
            link_keys = [k for k in dict.fromkeys(str(k) for k in links) if k]
            if link_keys != keys:
                keys = link_keys
                entity_ids = sorted(self.resolve_entities(scope, keys).values()) if keys else []
            if entity_ids:
                branches.append(
                    "SELECT 'links' AS ch, ae.atom_id AS atom_id, 0.0 AS score, NULL AS time_anchor, 1 AS live "
                    f"FROM atom_entity ae WHERE ae.entity_id IN ({_placeholders(len(entity_ids))})"
                )
                bound.extend(entity_ids)
            else:
                out["links"] = []
        prefix = f"WITH {', '.join(ctes)} " if ctes else ""
        rows = self._execute(prefix + " UNION ALL ".join(branches), cte_params + bound).fetchall() if branches else []
        grouped: dict[str, list[tuple[Any, ...]]] = {}
        for t in map(tuple, rows):
            grouped.setdefault(t[0], []).append(t)
        for name in ("entity", "recency", "links"):
            if any(b.startswith(f"SELECT '{name}'") for b in branches):
                ids = list(dict.fromkeys(t[1] for t in grouped.get(name, [])))
                if name == "entity":
                    ids = ids[: int(entity[1])]  # type: ignore[index]
                out[name] = [(atom_id, 0.0) for atom_id in ids]
        if keyword is not None:
            if kw_phases:
                need = int(keyword[1])
                seen: set[str] = set()
                found: list[tuple[str, float]] = []
                for position, (label, match) in enumerate(kw_phases):
                    if len(found) >= need:
                        break
                    first = None
                    if position == 0:
                        fetched = [(t[1], float(t[2]), t[3], int(t[4])) for t in grouped.get(label, [])]
                        first = (fetched, kw_window)
                    found.extend(
                        self._fts_phase(
                            match, keyed_gate, join_sql, keyed_values, need=need - len(found), seen=seen, first=first
                        )
                    )
                out["keyword"] = found[:need]
            else:
                extra_sql, extra_params = "", []
                if entity_gate:
                    extra_sql, extra_params = self.entity_gate_sql(scope, entity_gate, correlated=False)
                out["keyword"] = self.keyword_candidates(
                    scope,
                    kw_terms,
                    as_of=as_of,
                    include_expired=include_expired,
                    limit=int(keyword[1]),
                    where_sql=" AND ".join(f"({c})" for c in (where_sql, extra_sql) if c and c.strip()),
                    params=list(params) + extra_params,
                    joins=joins,
                    scale_constraint=scale_constraint,
                    atom_types=atom_types,
                )
        return out

    def recent_candidates(
        self,
        scope: Scope,
        *,
        as_of: str | None = None,
        include_expired: bool = False,
        limit: int,
        scale_in: Sequence[str] | None = None,
        obs_type_in: Sequence[str] | None = None,
    ) -> list[str]:
        """The recency recall channel: the `limit` most recent live atoms of the scope (optionally by scale / obs_type).
        新近度召回通道：范围内 `limit` 个最新存活原子（可按尺度 / obs_type 限定）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            as_of: Validity instant; defaults to now().
            as_of：有效性时刻；默认 now()。

            include_expired: Drop the validity predicate.
            include_expired：去掉有效性谓词。

            limit: Channel bound K_rec.
            limit：通道上限 K_rec。

            scale_in: scale_level whitelist (None / empty = no constraint).
            scale_in：scale_level 白名单（None / 空 = 不限制）。

            obs_type_in: obs_type whitelist (None / empty = no constraint).
            obs_type_in：obs_type 白名单（None / 空 = 不限制）。

        Output / 输出：
            Atom ids ordered (time_anchor DESC, created_at DESC, atom_id ASC), via ix_atoms_scope.
            按（time_anchor 降序，created_at 降序，atom_id 升序）的原子 id，经 ix_atoms_scope。

        State effect / 状态影响：
            None.
            无。
        """
        if limit <= 0:
            return []
        clauses, values = self._axis_where(
            scope, as_of=as_of, include_expired=include_expired, scale_constraint=scale_in, obs_types=obs_type_in
        )
        sql = (
            f"SELECT memory_atoms.atom_id AS atom_id FROM memory_atoms WHERE {' AND '.join(clauses)} "
            f"ORDER BY {_qualify(DEFAULT_ATOM_ORDER)} LIMIT ?"
        )
        return [r["atom_id"] for r in self._execute(sql, values + [int(limit)]).fetchall()]

    def atom_attrs(self, atom_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """The index scalars (INDEX_ATTR_KEYS of §10.3) of some atoms, NULL strings as "" and `archived` as a bool.
        若干原子的索引标量（§10.3 的 INDEX_ATTR_KEYS），字符串 NULL 为 ""，`archived` 为布尔值。

        Input / 输入：
            atom_ids: Atom ids (unknown ids are absent from the result).
            atom_ids：原子 id（未知 id 不出现在结果中）。

        Output / 输出：
            {atom_id: {'user_id', 'agent_id', 'run_id', 'time_anchor', 'valid_from', 'valid_to', 'archived', 'scale_level', 'obs_type', 'atom_type'}}.
            {atom_id: {'user_id', 'agent_id', 'run_id', 'time_anchor', 'valid_from', 'valid_to', 'archived', 'scale_level', 'obs_type', 'atom_type'}}。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(wanted):
            for r in self._execute(_ATTR_SQL.format(marks=_placeholders(len(chunk))), list(chunk)).fetchall():
                out[r["atom_id"]] = {
                    "user_id": r["user_id"] or "",
                    "agent_id": r["agent_id"] or "",
                    "run_id": r["run_id"] or "",
                    "time_anchor": r["time_anchor"] or "",
                    "valid_from": r["valid_from"] or "",
                    "valid_to": r["valid_to"] or "",
                    "archived": r["archived_reason"] is not None,
                    "scale_level": r["scale_level"] or "",
                    "obs_type": r["obs_type"] or "",
                    "atom_type": r["atom_type"] or "",
                }
        return out

    def count_atoms(self, scope: Scope, *, valid_only: bool = True) -> int:
        """Count atoms in a scope.
        统计范围内的原子数。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            valid_only: Apply LIVE_SQL at now().
            valid_only：按 now() 应用 LIVE_SQL。

        Output / 输出：
            Row count.
            行数。

        State effect / 状态影响：
            None.
            无。
        """
        where, values = _scope_where(scope, "memory_atoms.")
        clauses = [where]
        if valid_only:
            live, live_params = self._live(None)
            clauses.append(live)
            values.extend(live_params)
        row = self._execute(f"SELECT COUNT(*) AS n FROM memory_atoms WHERE {' AND '.join(clauses)}", values).fetchone()
        return int(row["n"])

    def find_atoms_by_hash(self, scope: Scope, md5: str, *, valid_only: bool = True) -> list[MemoryAtom]:
        """Atoms in scope with a given text hash (Mem0 exact dedup).
        范围内具有给定文本哈希的原子（Mem0 精确去重）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            md5: md5(text) hex digest.
            md5：md5(text) 十六进制摘要。

            valid_only: Apply LIVE_SQL at now().
            valid_only：按 now() 应用 LIVE_SQL。

        Output / 输出：
            Hydrated atoms in axis order.
            按轴顺序的水合原子。

        State effect / 状态影响：
            None.
            无。
        """
        return self.axis_select(scope, where_sql="memory_atoms.hash = ?", params=(md5,), include_expired=not valid_only)

    def revision_candidates(
        self,
        scope: Scope,
        *,
        subject: str,
        predicate: str,
        obs_type: str,
        topics: Sequence[str],
        objects: Sequence[str],
    ) -> list[MemoryAtom]:
        """Valid state/event atoms with the same subject and obs_type sharing predicate, a topic or an object.
        同 subject 与 obs_type、且共享谓词、主题或宾语的有效 state/event 原子。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            subject: Observation subject.
            subject：观察主体。

            predicate: Observation predicate.
            predicate：观察谓词。

            obs_type: Observation type.
            obs_type：观察类型。

            topics: Topic tags of the new observation.
            topics：新观察的主题标签。

            objects: Objects of the new observation.
            objects：新观察的宾语。

        Output / 输出：
            Candidates ordered by time_anchor DESC.
            按 time_anchor DESC 排序的候选。

        State effect / 状态影响：
            None.
            无。
        """
        pool = self.axis_select(
            scope,
            where_sql="memory_atoms.subject = ? AND memory_atoms.obs_type = ?",
            params=(subject, str(obs_type)),
            atom_types=("state", "event"),
        )
        topic_set = {str(t) for t in topics}
        object_set = {str(o) for o in objects}
        out: list[MemoryAtom] = []
        for atom in pool:
            atom_objects = atom.content.get("objects") if isinstance(atom.content, dict) else None
            objs = {str(o) for o in atom_objects} if isinstance(atom_objects, list) else set()
            if atom.predicate == predicate or (topic_set & set(atom.topic_tags)) or (object_set & objs):
                out.append(atom)
        return out

    def scope_atom_state(self, scope: Scope) -> dict[str, dict[str, str | None]]:
        """Every atom in scope → {'valid_to', 'archived_reason'} (checkpoint atom_state).
        范围内每个原子 → {'valid_to', 'archived_reason'}（检查点 atom_state）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

        Output / 输出：
            {atom_id: {'valid_to': ..., 'archived_reason': ...}}.
            {atom_id: {'valid_to': ..., 'archived_reason': ...}}。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        rows = self._execute(
            f"SELECT atom_id, valid_to, archived_reason FROM memory_atoms WHERE {where} ORDER BY atom_id", params
        ).fetchall()
        return {r["atom_id"]: {"valid_to": r["valid_to"], "archived_reason": r["archived_reason"]} for r in rows}

    def valid_ids(self, scope: Scope, *, as_of: str | None = None) -> dict[str, list[str]]:
        """Live ids per plane (the ONLY candidate-id helper).
        各平面的存活 id（唯一的候选 id 辅助函数）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            as_of: Validity instant; defaults to now().
            as_of：有效性时刻；默认 now()。

        Output / 输出：
            {'atoms', 'eps', 'edges', 'interface_eps'} → sorted id lists.
            {'atoms', 'eps', 'edges', 'interface_eps'} → 排序 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        instant = to_utc_iso(as_of) if as_of else self.now()
        where_a, params_a = _scope_where(scope, "memory_atoms.")
        live, live_params = self._live(instant)
        atoms = [
            r["atom_id"]
            for r in self._execute(
                f"SELECT atom_id FROM memory_atoms WHERE {where_a} AND {live} ORDER BY atom_id",
                params_a + live_params,
            )
        ]
        where_e, params_e = _scope_where(scope, "eps.")
        eps = [
            r["ep_id"]
            for r in self._execute(
                f"SELECT ep_id FROM eps WHERE {where_e} AND eps.valid_from <= ? "
                "AND (eps.valid_to IS NULL OR eps.valid_to > ?) AND NOT EXISTS "
                "(SELECT 1 FROM graph_nodes n WHERE n.node_id = eps.atom_id AND n.valid = 0) ORDER BY ep_id",
                params_e + [instant, instant],
            )
        ]
        where_g, params_g = _scope_where(scope)
        edges = [
            r["edge_id"]
            for r in self._execute(
                f"SELECT edge_id FROM graph_edges WHERE {where_g} AND valid = 1 ORDER BY edge_id", params_g
            )
        ]
        ieps = [
            r["interface_ep_id"]
            for r in self._execute(
                f"SELECT interface_ep_id FROM interface_eps WHERE {where_g} AND valid = 1 "
                "AND (valid_to IS NULL OR valid_to > ?) ORDER BY interface_ep_id",
                params_g + [instant],
            )
        ]
        return {"atoms": atoms, "eps": eps, "edges": edges, "interface_eps": ieps}

    # ------------------------------------------------------------------ entities / topics / vectors
    # ------------------------------------------------------------------ 实体 / 主题 / 向量
    def get_or_create_entity(
        self, scope: Scope, *, normalized: str, canonical: str, entity_type: str, aliases: Sequence[str] = ()
    ) -> str:
        """Resolve or create an entity keyed by (normalized, user_id, agent_id); run_id is provenance only.
        按 (normalized, user_id, agent_id) 解析或创建实体；run_id 仅作出处。

        Input / 输入：
            scope: Scope whose user_id/agent_id form the identity.
            scope：其 user_id/agent_id 构成身份的范围。

            normalized: Entity key (entities.normalized_text).
            normalized：实体键（entities.normalized_text）。

            canonical: Display text.
            canonical：显示文本。

            entity_type: PROPER | IDENTIFIER | FILE | QUOTED | TOPIC.
            entity_type：PROPER | IDENTIFIER | FILE | QUOTED | TOPIC。

            aliases: Extra surface forms appended when new.
            aliases：新增时追加的其他表面形式。

        Output / 输出：
            The entity id.
            实体 id。

        State effect / 状态影响：
            Writes entities (insert or alias append).
            写入 entities（插入或追加别名）。
        """
        new_aliases = [a for a in ordered_union(list(aliases)) if a and a != normalized]
        with self.transaction():
            row = self._execute(
                "SELECT entity_id, aliases_json FROM entities WHERE normalized_text = ? "
                "AND COALESCE(user_id,'') = COALESCE(?, '') AND COALESCE(agent_id,'') = COALESCE(?, '')",
                (normalized, scope.user_id, scope.agent_id),
            ).fetchone()
            if row is not None:
                existing = _loads(row["aliases_json"], [])
                merged = ordered_union(existing, new_aliases)
                if merged != existing:
                    self._execute(
                        "UPDATE entities SET aliases_json = ? WHERE entity_id = ?",
                        (canonical_json(merged), row["entity_id"]),
                    )
                return row["entity_id"]
            entity_id = self.new_id("ent")
            self._execute(
                "INSERT INTO entities(entity_id, normalized_text, canonical_text, entity_type, aliases_json, user_id, "
                "agent_id, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    entity_id,
                    normalized,
                    canonical,
                    str(entity_type),
                    canonical_json(new_aliases),
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    self.now(),
                ),
            )
            return entity_id

    def resolve_entity(self, scope: Scope, normalized: str) -> str | None:
        """Entity id for a key in the scope's identity (user_id, agent_id).
        某键在范围身份 (user_id, agent_id) 下的实体 id。

        Input / 输入：
            scope: Scope (user_id/agent_id used).
            scope：范围（使用 user_id/agent_id）。

            normalized: Entity key.
            normalized：实体键。

        Output / 输出：
            Entity id or None.
            实体 id 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute(
            "SELECT entity_id FROM entities WHERE normalized_text = ? AND COALESCE(user_id,'') = COALESCE(?, '') "
            "AND COALESCE(agent_id,'') = COALESCE(?, '')",
            (normalized, scope.user_id, scope.agent_id),
        ).fetchone()
        return row["entity_id"] if row else None

    def resolve_entities(self, scope: Scope, normalized: Sequence[str]) -> dict[str, str]:
        """Resolve many entity keys at once.
        一次解析多个实体键。

        Input / 输入：
            scope: Scope (user_id/agent_id used).
            scope：范围（使用 user_id/agent_id）。

            normalized: Entity keys.
            normalized：实体键。

        Output / 输出：
            {key: entity_id} for the keys that exist.
            存在的键的 {key: entity_id}。

        State effect / 状态影响：
            None.
            无。
        """
        keys = list(dict.fromkeys(str(k) for k in normalized))
        out: dict[str, str] = {}
        for chunk in _chunks(keys):
            rows = self._execute(
                f"SELECT normalized_text, entity_id FROM entities WHERE normalized_text IN ({_placeholders(len(chunk))}) "
                "AND COALESCE(user_id,'') = COALESCE(?, '') AND COALESCE(agent_id,'') = COALESCE(?, '')",
                list(chunk) + [scope.user_id, scope.agent_id],
            ).fetchall()
            for row in rows:
                out[row["normalized_text"]] = row["entity_id"]
        return out

    def entity_degree(self, entity_id: str) -> int:
        """Number of atom_entity links of an entity (entity atoms included).
        实体的 atom_entity 链接数（含实体原子）。

        Input / 输入：
            entity_id: Entity id.
            entity_id：实体 id。

        Output / 输出：
            Link count.
            链接数。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT COUNT(*) AS n FROM atom_entity WHERE entity_id = ?", (entity_id,)).fetchone()
        return int(row["n"])

    def entities_in_scope(self, scope: Scope) -> list[dict]:
        """Entities linked to at least one live atom matching the scope.
        与至少一个匹配范围的存活原子相链接的实体。

        Input / 输入：
            scope: Scope filter applied to the linked atoms.
            scope：应用于被链接原子的范围过滤。

        Output / 输出：
            [{'entity_id','normalized_text','canonical_text','entity_type','aliases','user_id','agent_id'}] by (normalized_text, entity_id).
            按 (normalized_text, entity_id) 排序的 [{'entity_id','normalized_text','canonical_text','entity_type','aliases','user_id','agent_id'}]。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope, "memory_atoms.")
        live, live_params = self._live(None)
        rows = self._execute(
            "SELECT DISTINCT e.entity_id, e.normalized_text, e.canonical_text, e.entity_type, e.aliases_json, "
            "e.user_id, e.agent_id FROM entities e JOIN atom_entity ae ON ae.entity_id = e.entity_id "
            f"JOIN memory_atoms ON memory_atoms.atom_id = ae.atom_id WHERE {where} AND {live} "
            "ORDER BY e.normalized_text, e.entity_id",
            params + live_params,
        ).fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "normalized_text": r["normalized_text"],
                "canonical_text": r["canonical_text"],
                "entity_type": r["entity_type"],
                "aliases": _loads(r["aliases_json"], []),
                "user_id": r["user_id"],
                "agent_id": r["agent_id"],
            }
            for r in rows
        ]

    def entity_linked_atoms(
        self, entity_id: str, *, valid_only: bool = True, exclude_entity_atoms: bool = True
    ) -> list[str]:
        """Atom ids linked to an entity.
        链接到某实体的原子 id。

        Input / 输入：
            entity_id: Entity id.
            entity_id：实体 id。

            valid_only: Apply LIVE_SQL at now().
            valid_only：按 now() 应用 LIVE_SQL。

            exclude_entity_atoms: Drop atoms with source_kind 'entity'.
            exclude_entity_atoms：排除 source_kind 为 'entity' 的原子。

        Output / 输出：
            Sorted atom ids.
            排序后的原子 id。

        State effect / 状态影响：
            None.
            无。
        """
        clauses = ["ae.entity_id = ?"]
        params: list[Any] = [entity_id]
        if valid_only:
            live, live_params = self._live(None)
            clauses.append(live)
            params.extend(live_params)
        if exclude_entity_atoms:
            clauses.append("memory_atoms.source_kind <> 'entity'")
        rows = self._execute(
            "SELECT ae.atom_id FROM atom_entity ae JOIN memory_atoms ON memory_atoms.atom_id = ae.atom_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY ae.atom_id",
            params,
        ).fetchall()
        return [r["atom_id"] for r in rows]

    def entities_for_atom(self, atom_id: str) -> list[dict]:
        """Entities linked to an atom.
        链接到某原子的实体。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            [{'entity_id','normalized_text','canonical_text','entity_type'}] by (normalized_text, entity_id).
            按 (normalized_text, entity_id) 排序的 [{'entity_id','normalized_text','canonical_text','entity_type'}]。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(
            "SELECT e.entity_id, e.normalized_text, e.canonical_text, e.entity_type FROM atom_entity ae "
            "JOIN entities e ON e.entity_id = ae.entity_id WHERE ae.atom_id = ? ORDER BY e.normalized_text, e.entity_id",
            (atom_id,),
        ).fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "normalized_text": r["normalized_text"],
                "canonical_text": r["canonical_text"],
                "entity_type": r["entity_type"],
            }
            for r in rows
        ]

    def link_atom_entity(self, atom_id: str, entity_id: str) -> None:
        """Link an atom to an entity (idempotent).
        将原子链接到实体（幂等）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            entity_id: Entity id.
            entity_id：实体 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes atom_entity.
            写入 atom_entity。
        """
        with self.transaction():
            self._execute("INSERT OR IGNORE INTO atom_entity(atom_id, entity_id) VALUES (?, ?)", (atom_id, entity_id))
            if self.capabilities.get("fts5"):
                # The keyword channel vocabulary follows the in-pool vocabulary (atom_keyword_tokens includes entities).
                # 关键词通道词表跟随池内词表（atom_keyword_tokens 包含实体）。
                self._refresh_fts(atom_id)

    def add_atom_topic(self, atom_id: str, topic: str) -> None:
        """Tag an atom with a topic (idempotent).
        为原子打上主题标签（幂等）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            topic: Topic string.
            topic：主题字符串。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes atom_topic.
            写入 atom_topic。
        """
        with self.transaction():
            self._execute("INSERT OR IGNORE INTO atom_topic(atom_id, topic) VALUES (?, ?)", (atom_id, topic))
            if self.capabilities.get("fts5"):
                # A topic added after insert (feedback 'object_important') must be recalled by the keyword channel.
                # 插入后添加的主题（反馈 'object_important'）必须能被关键词通道召回。
                self._refresh_fts(atom_id)

    def upsert_vector(self, atom_id: str, vector: Sequence[float], *, embedder: str) -> None:
        """Store an atom's embedding as little-endian float32.
        以小端 float32 存储原子嵌入。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

            vector: Floats (L2-normalized by the embedder).
            vector：浮点数（由嵌入器做 L2 归一化）。

            embedder: EmbedderBase.identity, e.g. 'hash@64'.
            embedder：EmbedderBase.identity，如 'hash@64'。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes atom_vectors (replace).
            写入 atom_vectors（替换）。
        """
        values = [float(v) for v in vector]
        if not values:
            raise ValidationError("vector must be non-empty", "VALIDATION_000", details={"atom_id": atom_id})
        blob = struct.pack(f"<{len(values)}f", *values)
        with self.transaction():
            self._execute(
                "INSERT OR REPLACE INTO atom_vectors(atom_id, embedder, dims, embedding) VALUES (?, ?, ?, ?)",
                (atom_id, str(embedder), len(values), sqlite3.Binary(blob)),
            )

    def load_vectors(self, atom_ids: Iterable[str] | None) -> dict[str, list[float]]:
        """Load embeddings for some atoms (None → all).
        加载若干原子的嵌入（None → 全部）。

        Input / 输入：
            atom_ids: Atom ids or None.
            atom_ids：原子 id 或 None。

        Output / 输出：
            {atom_id: [float, ...]}.
            {atom_id: [float, ...]}。

        State effect / 状态影响：
            None.
            无。
        """
        out: dict[str, list[float]] = {}

        def _collect(rows: Iterable[sqlite3.Row]) -> None:
            for row in rows:
                dims = int(row["dims"])
                out[row["atom_id"]] = list(struct.unpack(f"<{dims}f", bytes(row["embedding"])))

        if atom_ids is None:
            _collect(self._execute("SELECT atom_id, dims, embedding FROM atom_vectors ORDER BY atom_id").fetchall())
            return out
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        for chunk in _chunks(wanted):
            _collect(
                self._execute(
                    f"SELECT atom_id, dims, embedding FROM atom_vectors WHERE atom_id IN ({_placeholders(len(chunk))})",
                    list(chunk),
                ).fetchall()
            )
        return out

    def load_vector_blobs(self, atom_ids: Iterable[str] | None) -> list[tuple[str, int, bytes]]:
        """Load raw little-endian float32 embeddings (no decoding) for some atoms (None → all, by atom_id).
        加载若干原子的原始小端 float32 嵌入（不解码）（None → 全部，按 atom_id）。

        Input / 输入：
            atom_ids: Atom ids (result keeps their first-seen order; ids without a row are skipped) or None.
            atom_ids：原子 id（结果保持首次出现的顺序；没有行的 id 被跳过）或 None。

        Output / 输出：
            `[(atom_id, dims, blob), ...]`; the caller builds one matrix with numpy.frombuffer.
            `[(atom_id, dims, blob), ...]`；调用方用 numpy.frombuffer 构造一个矩阵。

        State effect / 状态影响：
            None.
            无。
        """
        if atom_ids is None:
            rows = self._execute("SELECT atom_id, dims, embedding FROM atom_vectors ORDER BY atom_id").fetchall()
            return [(r["atom_id"], int(r["dims"]), bytes(r["embedding"])) for r in rows]
        wanted = list(dict.fromkeys(str(i) for i in atom_ids))
        found: dict[str, tuple[int, bytes]] = {}
        for chunk in _chunks(wanted):
            for r in self._execute(
                f"SELECT atom_id, dims, embedding FROM atom_vectors WHERE atom_id IN ({_placeholders(len(chunk))})",
                list(chunk),
            ).fetchall():
                found[r["atom_id"]] = (int(r["dims"]), bytes(r["embedding"]))
        return [(atom_id, found[atom_id][0], found[atom_id][1]) for atom_id in wanted if atom_id in found]

    def load_vector_rows(self, after_rowid: int = 0) -> list[tuple[int, str, int, bytes, dict[str, Any]]]:
        """Vectors appended after `after_rowid`, each with its atom's gate scalars (the index cache feed, §10.3).
        `after_rowid` 之后追加的向量，各带其原子的门控标量（索引缓存的数据源，§10.3）。

        Input / 输入：
            after_rowid: Read atom_vectors rows with rowid greater than this (0 → every row).
            after_rowid：读取 rowid 大于此值的 atom_vectors 行（0 → 全部行）。

        Output / 输出：
            `[(rowid, atom_id, dims, blob, attrs), ...]` in rowid order; attrs has the INDEX_ATTR_KEYS shape of
            atom_attrs (NULL strings as "", archived as a bool). An INSERT OR REPLACE gives the replaced row a new
            rowid, so a replaced vector reappears here with its atom_id.
            按 rowid 顺序的 `[(rowid, atom_id, dims, blob, attrs), ...]`；attrs 为 atom_attrs 的 INDEX_ATTR_KEYS 形状
            （字符串 NULL 为 ""，archived 为布尔值）。INSERT OR REPLACE 会给被替换的行一个新 rowid，因此被替换的向量会
            带着其 atom_id 再次出现在这里。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(_VECTOR_ROWS_SQL, [int(after_rowid)]).fetchall()
        return [
            (
                int(r["rowid"]),
                r["atom_id"],
                int(r["dims"]),
                bytes(r["embedding"]),
                {
                    "user_id": r["user_id"] or "",
                    "agent_id": r["agent_id"] or "",
                    "run_id": r["run_id"] or "",
                    "time_anchor": r["time_anchor"] or "",
                    "valid_from": r["valid_from"] or "",
                    "valid_to": r["valid_to"] or "",
                    "archived": r["archived_reason"] is not None,
                    "scale_level": r["scale_level"] or "",
                    "obs_type": r["obs_type"] or "",
                    "atom_type": r["atom_type"] or "",
                },
            )
            for r in rows
        ]

    def delete_vector(self, atom_id: str) -> None:
        """Remove an atom's embedding (vectors are derived data, not evidence).
        删除原子嵌入（向量是派生数据而非证据）。

        Input / 输入：
            atom_id: Atom id.
            atom_id：原子 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Deletes from atom_vectors.
            从 atom_vectors 删除。
        """
        with self.transaction():
            self._execute("DELETE FROM atom_vectors WHERE atom_id = ?", (atom_id,))

    def count_vectors(self) -> int:
        """Number of stored embeddings.
        已存储嵌入的数量。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Row count.
            行数。

        State effect / 状态影响：
            None.
            无。
        """
        return int(self._execute("SELECT COUNT(*) AS n FROM atom_vectors").fetchone()["n"])

    def atoms_without_vectors(self, scope: Scope | None, *, embedder: str) -> list[str]:
        """Atoms lacking an embedding row or embedded by a different embedder identity.
        缺少嵌入行、或由不同嵌入器身份生成嵌入的原子。

        Input / 输入：
            scope: Optional scope filter.
            scope：可选的范围过滤。

            embedder: The current embedder identity.
            embedder：当前嵌入器身份。

        Output / 输出：
            Sorted atom ids.
            排序后的原子 id。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope, "memory_atoms.")
        rows = self._execute(
            "SELECT memory_atoms.atom_id FROM memory_atoms LEFT JOIN atom_vectors v ON v.atom_id = memory_atoms.atom_id "
            f"WHERE {where} AND (v.atom_id IS NULL OR v.embedder <> ?) ORDER BY memory_atoms.atom_id",
            params + [str(embedder)],
        ).fetchall()
        return [r["atom_id"] for r in rows]

    def clear_vectors(self) -> None:
        """Drop every embedding (index reset).
        清空全部嵌入（索引重置）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Deletes all atom_vectors rows.
            删除 atom_vectors 全部行。
        """
        with self.transaction():
            self._execute("DELETE FROM atom_vectors")

    # ------------------------------------------------------------------ graph: nodes
    # ------------------------------------------------------------------ 图：节点
    def _node_from_row(self, row: sqlite3.Row) -> GraphNode:
        """GraphNode from a graph_nodes row.
        由 graph_nodes 行构造 GraphNode。
        """
        return GraphNode(
            node_id=row["node_id"],
            node_kind=row["node_kind"],
            ref_id=row["ref_id"],
            scale=row["scale"],
            scope=_scope_of(row),
            valid=bool(row["valid"]),
            invalidated_at=row["invalidated_at"],
            mentions=int(row["mentions"]),
            created_at=row["created_at"],
            anchor_ids=[row["primary_anchor_id"]],
        )

    def insert_node(self, node: GraphNode) -> str:
        """Insert a graph node or MERGE on node_id (mentions += 1).
        插入图节点，或按 node_id MERGE（mentions += 1）。

        Input / 输入：
            node: The node (node_id == atom_id; anchor_ids[0] is the primary witness).
            node：节点（node_id == atom_id；anchor_ids[0] 为主见证）。

        Output / 输出：
            The node id.
            节点 id。

        State effect / 状态影响：
            Writes graph_nodes.
            写入 graph_nodes。
        """
        anchor_ids = self._require_anchors(node.anchor_ids, obj="node", obj_id=node.node_id or None)
        with self.transaction():
            cur = self._execute("UPDATE graph_nodes SET mentions = mentions + 1 WHERE node_id = ?", (node.node_id,))
            if cur.rowcount == 1:
                return node.node_id
            self._execute(
                "INSERT INTO graph_nodes(node_id, node_kind, ref_id, scale, primary_anchor_id, user_id, agent_id, "
                "run_id, valid, invalidated_at, mentions, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node.node_id,
                    node.node_kind,
                    node.ref_id,
                    node.scale,
                    anchor_ids[0],
                    node.scope.user_id,
                    node.scope.agent_id,
                    node.scope.run_id,
                    1 if node.valid else 0,
                    node.invalidated_at,
                    int(node.mentions),
                    node.created_at or self.now(),
                ),
            )
            return node.node_id

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch one node.
        读取一个节点。

        Input / 输入：
            node_id: Node id (== atom id).
            node_id：节点 id（== 原子 id）。

        Output / 输出：
            GraphNode or None.
            GraphNode 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone()
        return self._node_from_row(row) if row else None

    def get_nodes(self, node_ids: Iterable[str]) -> list[GraphNode]:
        """Fetch nodes in input order, skipping unknown ids.
        按输入顺序读取节点，跳过未知 id。

        Input / 输入：
            node_ids: Node ids.
            node_ids：节点 id。

        Output / 输出：
            GraphNodes.
            GraphNode 列表。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in node_ids))
        found: dict[str, GraphNode] = {}
        for chunk in _chunks(wanted):
            for row in self._execute(
                f"SELECT * FROM graph_nodes WHERE node_id IN ({_placeholders(len(chunk))})", list(chunk)
            ):
                found[row["node_id"]] = self._node_from_row(row)
        return [found[i] for i in wanted if i in found]

    def list_nodes(
        self, scope: Scope, *, kinds: Sequence[str] | None = None, valid_only: bool = True, scale: str | None = None
    ) -> list[GraphNode]:
        """Nodes of a scope ordered by (created_at, node_id).
        按 (created_at, node_id) 排序的范围内节点。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            kinds: node_kind whitelist.
            kinds：node_kind 白名单。

            valid_only: Keep valid nodes only.
            valid_only：仅保留有效节点。

            scale: Optional graph_nodes.scale filter.
            scale：可选的 graph_nodes.scale 过滤。

        Output / 输出：
            GraphNodes.
            GraphNode 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        clauses = [where]
        if kinds:
            values = [str(k) for k in kinds]
            clauses.append(f"node_kind IN ({_placeholders(len(values))})")
            params.extend(values)
        if valid_only:
            clauses.append("valid = 1")
        if scale is not None:
            clauses.append("scale = ?")
            params.append(str(scale))
        rows = self._execute(
            f"SELECT * FROM graph_nodes WHERE {' AND '.join(clauses)} ORDER BY created_at, node_id", params
        ).fetchall()
        return [self._node_from_row(r) for r in rows]

    def set_node_valid(self, node_id: str, valid: bool, *, at: str) -> None:
        """Flip a node's validity.
        切换节点有效性。

        Input / 输入：
            node_id: Node id.
            node_id：节点 id。

            valid: New validity.
            valid：新的有效性。

            at: invalidated_at instant when invalidating.
            at：失效时的 invalidated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_nodes.
            更新 graph_nodes。
        """
        with self.transaction():
            self._execute(
                "UPDATE graph_nodes SET valid = ?, invalidated_at = ? WHERE node_id = ?",
                (1 if valid else 0, None if valid else to_utc_iso(at), node_id),
            )

    def bump_mentions(self, node_id: str, n: int = 1) -> None:
        """Increase a node's mention counter.
        增加节点的提及计数。

        Input / 输入：
            node_id: Node id.
            node_id：节点 id。

            n: Increment.
            n：增量。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_nodes.mentions.
            更新 graph_nodes.mentions。
        """
        with self.transaction():
            self._execute("UPDATE graph_nodes SET mentions = mentions + ? WHERE node_id = ?", (int(n), node_id))

    # ------------------------------------------------------------------ graph: edges
    # ------------------------------------------------------------------ 图：边
    def _witnesses_for(self, edge_ids: Sequence[str]) -> dict[str, list[str]]:
        """edge_id → witness anchor ids, primary first then insertion order.
        edge_id → 见证锚点 id，主见证优先，其后按插入顺序。
        """
        out: dict[str, list[str]] = {}
        for chunk in _chunks(list(edge_ids)):
            rows = self._execute(
                "SELECT ew.edge_id, ew.anchor_id FROM edge_witnesses ew JOIN graph_edges e ON e.edge_id = ew.edge_id "
                f"WHERE ew.edge_id IN ({_placeholders(len(chunk))}) "
                "ORDER BY ew.edge_id, CASE WHEN ew.anchor_id = e.primary_witness_anchor_id THEN 0 ELSE 1 END, ew.rowid",
                list(chunk),
            ).fetchall()
            for row in rows:
                out.setdefault(row["edge_id"], []).append(row["anchor_id"])
        return out

    def _edges_from_rows(self, rows: Sequence[sqlite3.Row]) -> list[GraphEdge]:
        """GraphEdge models from graph_edges rows (witnesses hydrated in batch).
        由 graph_edges 行构造 GraphEdge（批量水合见证）。
        """
        if not rows:
            return []
        witnesses = self._witnesses_for([r["edge_id"] for r in rows])
        edges: list[GraphEdge] = []
        for row in rows:
            edges.append(
                GraphEdge(
                    edge_id=row["edge_id"],
                    src_id=row["src_id"],
                    dst_id=row["dst_id"],
                    edge_type=row["edge_type"],
                    label=row["label"],
                    scale=row["scale"],
                    is_boundary=bool(row["is_boundary"]),
                    confidence=row["confidence"],
                    mentions=int(row["mentions"]),
                    witness_anchor_ids=witnesses.get(row["edge_id"]) or [row["primary_witness_anchor_id"]],
                    witness_edge_id=row["witness_edge_id"],
                    created_by=row["created_by"],
                    proposer=row["proposer"],
                    valid=bool(row["valid"]),
                    invalidated_at=row["invalidated_at"],
                    invalidated_reason=row["invalidated_reason"],
                    resolved_at=row["resolved_at"],
                    resolved_by_anchor_id=row["resolved_by_anchor_id"],
                    resolution_edge_id=row["resolution_edge_id"],
                    scope=_scope_of(row),
                    created_at=row["created_at"],
                )
            )
        return edges

    def _select_edges(self, sql: str, params: Sequence[Any] = ()) -> list[GraphEdge]:
        """Run a graph_edges SELECT and hydrate.
        执行 graph_edges 的 SELECT 并水合。
        """
        return self._edges_from_rows(self._execute(sql, params).fetchall())

    def insert_edge(self, edge: GraphEdge | dict) -> str:
        """Insert a witnessed edge or MERGE into the existing (src, dst, type, scale) row.
        插入带见证的边，或 MERGE 进已存在的 (src, dst, type, scale) 行。

        Input / 输入：
            edge: GraphEdge or spec dict; an empty witness list raises WitnessMissingError before any SQL.
            edge：GraphEdge 或规格字典；见证列表为空时在任何 SQL 之前抛出 WitnessMissingError。

        Output / 输出：
            The existing or new edge id (callers compare with find_edge() to detect "new").
            已存在或新建的边 id（调用方通过 find_edge() 比较判断是否新建）。

        State effect / 状态影响：
            Writes graph_edges, edge_witnesses (+ edge_projections by trigger / witness_edge_id); MERGE bumps mentions and confidence.
            写入 graph_edges、edge_witnesses（触发器/witness_edge_id 写 edge_projections）；MERGE 提升 mentions 与 confidence。
        """
        data: dict[str, Any] = dict(edge) if isinstance(edge, dict) else dict(getattr(edge, "__dict__", {}))
        witnesses = data.get("witness_anchor_ids")
        edge_id_hint = data.get("edge_id") or None
        if not witnesses:
            raise WitnessMissingError(
                "edge has no witness anchors", details={"object": "edge", "id": edge_id_hint, "missing": []}
            )
        witness_ids = self._require_anchors(list(witnesses), obj="edge", obj_id=edge_id_hint)
        if data.get("scope") is None:
            src = self.get_node(str(data.get("src_id", "")))
            if src is None:
                raise ValidationError(
                    f"unknown src node {data.get('src_id')!r}", "NOT_FOUND_001", details={"src_id": data.get("src_id")}
                )
            data["scope"] = src.scope
        if data.get("created_at") is None:
            data["created_at"] = self.now()
        data["witness_anchor_ids"] = witness_ids
        model = GraphEdge(**data)
        existing = self.find_edge(model.src_id, model.dst_id, model.edge_type, model.scale)
        with self.transaction():
            if existing is not None:
                self._execute(
                    "UPDATE graph_edges SET mentions = mentions + 1, confidence = MAX(confidence, ?) WHERE edge_id = ?",
                    (float(model.confidence), existing.edge_id),
                )
                for anchor_id in witness_ids:
                    self._execute(
                        "INSERT OR IGNORE INTO edge_witnesses(edge_id, anchor_id) VALUES (?, ?)",
                        (existing.edge_id, anchor_id),
                    )
                if model.witness_edge_id and model.witness_edge_id != existing.edge_id:
                    self._execute(
                        "INSERT OR IGNORE INTO edge_projections(high_edge_id, low_edge_id) VALUES (?, ?)",
                        (existing.edge_id, model.witness_edge_id),
                    )
                return existing.edge_id
            edge_id = model.edge_id or self.new_id("edg")
            self._execute(
                "INSERT INTO graph_edges(edge_id, src_id, dst_id, edge_type, label, scale, is_boundary, confidence, "
                "mentions, primary_witness_anchor_id, witness_edge_id, created_by, proposer, valid, invalidated_at, "
                "invalidated_reason, resolved_at, resolved_by_anchor_id, resolution_edge_id, user_id, agent_id, run_id, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    edge_id,
                    model.src_id,
                    model.dst_id,
                    model.edge_type,
                    model.label,
                    model.scale,
                    1 if model.is_boundary else 0,
                    float(model.confidence),
                    int(model.mentions),
                    witness_ids[0],
                    model.witness_edge_id,
                    model.created_by,
                    model.proposer,
                    1 if model.valid else 0,
                    model.invalidated_at,
                    model.invalidated_reason,
                    model.resolved_at,
                    model.resolved_by_anchor_id,
                    model.resolution_edge_id,
                    model.scope.user_id,
                    model.scope.agent_id,
                    model.scope.run_id,
                    model.created_at,
                ),
            )
            for anchor_id in witness_ids[1:]:
                self._execute(
                    "INSERT OR IGNORE INTO edge_witnesses(edge_id, anchor_id) VALUES (?, ?)", (edge_id, anchor_id)
                )
            return edge_id

    def set_edge_label(self, edge_id: str, label: str, *, at: str) -> None:
        """Relabel an edge (label is a mutable column).
        重设边标签（label 为可变列）。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

            label: New label.
            label：新标签。

            at: Instant of the change (recorded by the caller's history row; unused by the column).
            at：变更时刻（由调用方历史行记录；该列不使用）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_edges.label.
            更新 graph_edges.label。
        """
        # 边记录没有 updated_at 列
        del at  # the edge row has no updated_at column
        with self.transaction():
            self._execute("UPDATE graph_edges SET label = ? WHERE edge_id = ?", (label, edge_id))

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Fetch one edge.
        读取一条边。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

        Output / 输出：
            GraphEdge or None.
            GraphEdge 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        edges = self._select_edges("SELECT * FROM graph_edges WHERE edge_id = ?", (edge_id,))
        return edges[0] if edges else None

    def get_edges(self, edge_ids: Iterable[str]) -> list[GraphEdge]:
        """Fetch edges in input order, skipping unknown ids.
        按输入顺序读取边，跳过未知 id。

        Input / 输入：
            edge_ids: Edge ids.
            edge_ids：边 id。

        Output / 输出：
            GraphEdges.
            GraphEdge 列表。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = list(dict.fromkeys(str(i) for i in edge_ids))
        found: dict[str, GraphEdge] = {}
        for chunk in _chunks(wanted):
            for e in self._select_edges(
                f"SELECT * FROM graph_edges WHERE edge_id IN ({_placeholders(len(chunk))})", list(chunk)
            ):
                found[e.edge_id] = e
        return [found[i] for i in wanted if i in found]

    def find_edge(self, src_id: str, dst_id: str, edge_type: str, scale: str = "L1_micro") -> GraphEdge | None:
        """The edge with a given identity (src, dst, type, scale), any validity.
        具有给定身份 (src, dst, type, scale) 的边，不论有效性。

        Input / 输入：
            src_id: Source node id.
            src_id：源节点 id。

            dst_id: Target node id.
            dst_id：目标节点 id。

            edge_type: EdgeType value.
            edge_type：EdgeType 值。

            scale: Edge scale.
            scale：边尺度。

        Output / 输出：
            GraphEdge or None.
            GraphEdge 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        edges = self._select_edges(
            "SELECT * FROM graph_edges WHERE src_id = ? AND dst_id = ? AND edge_type = ? AND scale = ?",
            (src_id, dst_id, str(edge_type), str(scale)),
        )
        return edges[0] if edges else None

    def edges_touching(
        self,
        node_ids: Iterable[str],
        *,
        valid_only: bool = True,
        edge_types: Sequence[str] | None = None,
        direction: str = "both",
    ) -> list[GraphEdge]:
        """Edges with at least one endpoint in a node set.
        至少一个端点落在节点集合内的边。

        Input / 输入：
            node_ids: Node ids.
            node_ids：节点 id。

            valid_only: Keep valid edges only.
            valid_only：仅保留有效边。

            edge_types: edge_type whitelist.
            edge_types：edge_type 白名单。

            direction: 'both' | 'in' (dst in set) | 'out' (src in set).
            direction：'both' | 'in'（dst 在集合内）| 'out'（src 在集合内）。

        Output / 输出：
            Edges ordered by (created_at, edge_id).
            按 (created_at, edge_id) 排序的边。

        State effect / 状态影响：
            None.
            无。
        """
        ids = list(dict.fromkeys(str(i) for i in node_ids))
        if not ids:
            return []
        if direction not in ("both", "in", "out"):
            raise ValidationError(
                f"direction must be both|in|out, got {direction!r}", "VALIDATION_013", details={"direction": direction}
            )
        out: dict[str, GraphEdge] = {}
        for chunk in _chunks(ids):
            marks = _placeholders(len(chunk))
            if direction == "in":
                endpoint = f"dst_id IN ({marks})"
                params: list[Any] = list(chunk)
            elif direction == "out":
                endpoint = f"src_id IN ({marks})"
                params = list(chunk)
            else:
                endpoint = f"(src_id IN ({marks}) OR dst_id IN ({marks}))"
                params = list(chunk) + list(chunk)
            clauses = [endpoint]
            if valid_only:
                clauses.append("valid = 1")
            if edge_types:
                types = [str(t) for t in edge_types]
                clauses.append(f"edge_type IN ({_placeholders(len(types))})")
                params.extend(types)
            for e in self._select_edges(
                f"SELECT * FROM graph_edges WHERE {' AND '.join(clauses)} ORDER BY created_at, edge_id", params
            ):
                out[e.edge_id] = e
        return sorted(out.values(), key=lambda e: (e.created_at or "", e.edge_id))

    def edges_between(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[GraphEdge]:
        """Edges whose both endpoints lie in a node set.
        两个端点都在节点集合内的边。

        Input / 输入：
            node_ids: Node ids.
            node_ids：节点 id。

            valid_only: Keep valid edges only.
            valid_only：仅保留有效边。

        Output / 输出：
            Edges ordered by (created_at, edge_id).
            按 (created_at, edge_id) 排序的边。

        State effect / 状态影响：
            None.
            无。
        """
        ids = set(str(i) for i in node_ids)
        if not ids:
            return []
        touching = self.edges_touching(ids, valid_only=valid_only, direction="both")
        return [e for e in touching if e.src_id in ids and e.dst_id in ids]

    def list_edges(
        self,
        scope: Scope,
        *,
        edge_types: Sequence[str] | None = None,
        valid_only: bool = True,
        since: str | None = None,
        scale: str | None = None,
        label_prefix: str | None = None,
        include_boundary: bool = True,
    ) -> list[GraphEdge]:
        """Edges of a scope ordered by (created_at, edge_id).
        按 (created_at, edge_id) 排序的范围内边。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            edge_types: edge_type whitelist.
            edge_types：edge_type 白名单。

            valid_only: Keep valid edges only.
            valid_only：仅保留有效边。

            since: Inclusive lower bound on created_at.
            since：created_at 的下界（含）。

            scale: Optional scale filter.
            scale：可选的尺度过滤。

            label_prefix: Keep labels starting with this prefix.
            label_prefix：仅保留以该前缀开头的标签。

            include_boundary: False drops projected (boundary) edges.
            include_boundary：为 False 时排除投影（边界）边。

        Output / 输出：
            GraphEdges.
            GraphEdge 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        clauses = [where]
        if edge_types:
            types = [str(t) for t in edge_types]
            clauses.append(f"edge_type IN ({_placeholders(len(types))})")
            params.extend(types)
        if valid_only:
            clauses.append("valid = 1")
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(to_utc_iso(since))
        if scale is not None:
            clauses.append("scale = ?")
            params.append(str(scale))
        if label_prefix is not None:
            escaped = label_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("label LIKE ? ESCAPE '\\'")
            params.append(escaped + "%")
        if not include_boundary:
            clauses.append("is_boundary = 0")
        return self._select_edges(
            f"SELECT * FROM graph_edges WHERE {' AND '.join(clauses)} ORDER BY created_at, edge_id", params
        )

    def invalidate_edge(self, edge_id: str, *, reason: str, at: str) -> None:
        """Invalidate an edge (never deleted).
        使边失效（绝不删除）。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

            reason: invalidated_reason.
            reason：invalidated_reason。

            at: invalidated_at instant.
            at：invalidated_at 时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_edges.
            更新 graph_edges。
        """
        with self.transaction():
            self._execute(
                "UPDATE graph_edges SET valid = 0, invalidated_at = ?, invalidated_reason = ? WHERE edge_id = ?",
                (to_utc_iso(at), reason, edge_id),
            )

    def revalidate_edge(self, edge_id: str, *, at: str) -> None:
        """Make an edge valid again.
        使边重新有效。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

            at: Instant of the change (not stored on the row).
            at：变更时刻（不存储在行上）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_edges (valid = 1, invalidation fields cleared).
            更新 graph_edges（valid = 1，清空失效字段）。
        """
        del at
        with self.transaction():
            self._execute(
                "UPDATE graph_edges SET valid = 1, invalidated_at = NULL, invalidated_reason = NULL WHERE edge_id = ?",
                (edge_id,),
            )

    def resolve_conflict(
        self, edge_id: str, *, at: str, by_anchor_id: str, resolution_edge_id: str | None = None
    ) -> None:
        """Mark a conflict edge resolved (witnessed by an anchor, optionally by a resolving edge).
        标记冲突边已解决（由锚点见证，可选地由解决边见证）。

        Input / 输入：
            edge_id: Conflict edge id.
            edge_id：冲突边 id。

            at: resolved_at instant.
            at：resolved_at 时刻。

            by_anchor_id: Witness anchor of the resolution (pre-checked).
            by_anchor_id：解决的见证锚点（预检）。

            resolution_edge_id: Optional revision edge that resolved it.
            resolution_edge_id：可选的解决它的修订边。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates graph_edges (CHECK: conflict edges only).
            更新 graph_edges（CHECK：仅限冲突边）。
        """
        self._require_anchors([by_anchor_id], obj="edge", obj_id=edge_id)
        with self.transaction():
            self._execute(
                "UPDATE graph_edges SET resolved_at = ?, resolved_by_anchor_id = ?, resolution_edge_id = ? "
                "WHERE edge_id = ?",
                (to_utc_iso(at), by_anchor_id, resolution_edge_id, edge_id),
            )

    def edge_witnesses(self, edge_id: str) -> list[str]:
        """Witness anchor ids of an edge, primary first.
        边的见证锚点 id，主见证优先。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

        Output / 输出：
            Anchor ids.
            锚点 id。

        State effect / 状态影响：
            None.
            无。
        """
        return self._witnesses_for([edge_id]).get(edge_id, [])

    def add_edge_witness(self, edge_id: str, anchor_id: str) -> None:
        """Add a witness anchor to an edge (idempotent; pre-checked).
        为边追加见证锚点（幂等；预检）。

        Input / 输入：
            edge_id: Edge id.
            edge_id：边 id。

            anchor_id: Anchor id.
            anchor_id：锚点 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes edge_witnesses.
            写入 edge_witnesses。
        """
        self._require_anchors([anchor_id], obj="edge", obj_id=edge_id)
        with self.transaction():
            self._execute(
                "INSERT OR IGNORE INTO edge_witnesses(edge_id, anchor_id) VALUES (?, ?)", (edge_id, anchor_id)
            )

    def projections_of(self, high_edge_id: str) -> list[str]:
        """Low edges a high edge projects (insertion order: witness_edge_id first).
        某高层边所投影的低层边（插入顺序：witness_edge_id 优先）。

        Input / 输入：
            high_edge_id: High-level edge id.
            high_edge_id：高层边 id。

        Output / 输出：
            Low edge ids.
            低层边 id。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(
            "SELECT low_edge_id FROM edge_projections WHERE high_edge_id = ? ORDER BY rowid", (high_edge_id,)
        ).fetchall()
        return [r["low_edge_id"] for r in rows]

    def projections_from(self, low_edge_id: str) -> list[str]:
        """High edges projected from a low edge.
        由某低层边投影出的高层边。

        Input / 输入：
            low_edge_id: Low-level edge id.
            low_edge_id：低层边 id。

        Output / 输出：
            High edge ids.
            高层边 id。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(
            "SELECT high_edge_id FROM edge_projections WHERE low_edge_id = ? ORDER BY rowid", (low_edge_id,)
        ).fetchall()
        return [r["high_edge_id"] for r in rows]

    def add_projection(self, high_edge_id: str, low_edge_id: str) -> None:
        """Record that a high edge projects a low edge (idempotent).
        记录高层边投影某低层边（幂等）。

        Input / 输入：
            high_edge_id: High-level edge id.
            high_edge_id：高层边 id。

            low_edge_id: Low-level edge id.
            low_edge_id：低层边 id。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes edge_projections.
            写入 edge_projections。
        """
        with self.transaction():
            self._execute(
                "INSERT OR IGNORE INTO edge_projections(high_edge_id, low_edge_id) VALUES (?, ?)",
                (high_edge_id, low_edge_id),
            )

    # ------------------------------------------------------------------ graph: interface EPs
    # ------------------------------------------------------------------ 图：接口 EP
    def _iep_from_row(self, row: sqlite3.Row) -> InterfaceEP:
        """InterfaceEP from an interface_eps row.
        由 interface_eps 行构造 InterfaceEP。
        """
        slots_data = _loads(row["slots_json"], {})
        slots = Slots(
            u=slots_data.get("u", ""),
            b=slots_data.get("b", ""),
            a=slots_data.get("a", ""),
            y=slots_data.get("y", ""),
            r=slots_data.get("r", "unknown"),
            kappa=Kappa(**slots_data.get("kappa", {})),
        )
        return InterfaceEP(
            interface_ep_id=row["interface_ep_id"],
            scale=row["scale"],
            slots=slots,
            slot_support=_loads(row["slot_support_json"], {}),
            support_node_ids=_loads(row["support_node_ids_json"], []),
            support_edge_ids=_loads(row["support_edge_ids_json"], []),
            boundary_edges=[BoundaryEdge(**b) for b in _loads(row["boundary_edges_json"], [])],
            invariants=[Invariant(**i) for i in _loads(row["invariants_json"], [])],
            residuals=[Residual(**r) for r in _loads(row["residuals_json"], [])],
            evidence_anchor_ids=_loads(row["evidence_anchor_ids_json"], []),
            coarsen_op_version=row["coarsen_op_version"],
            cohesion_score=row["cohesion_score"],
            atom_id=row["atom_id"],
            scope=_scope_of(row),
            valid=bool(row["valid"]),
            valid_to=row["valid_to"],
            created_at=row["created_at"],
        )

    def _select_ieps(self, sql: str, params: Sequence[Any] = ()) -> list[InterfaceEP]:
        """Run an interface_eps SELECT and hydrate.
        执行 interface_eps 的 SELECT 并水合。
        """
        return [self._iep_from_row(r) for r in self._execute(sql, params).fetchall()]

    def insert_interface_ep(self, iep: InterfaceEP) -> InterfaceEP:
        """Persist an InterfaceEP and its support members (its episode atom must already exist).
        持久化 InterfaceEP 及其支持成员（其情节原子必须已存在）。

        Input / 输入：
            iep: The InterfaceEP with atom_id set.
            iep：atom_id 已设置的 InterfaceEP。

        Output / 输出：
            The stored InterfaceEP with ids.
            带 id 的已存储 InterfaceEP。

        State effect / 状态影响：
            Writes interface_eps, interface_ep_support.
            写入 interface_eps、interface_ep_support。
        """
        evidence = self._require_anchors(
            iep.evidence_anchor_ids, obj="interface_ep", obj_id=iep.interface_ep_id or None
        )
        if not iep.atom_id:
            raise ValidationError(
                "InterfaceEP.atom_id must reference its episode atom",
                "VALIDATION_005",
                details={"object": "interface_ep", "field": "atom_id"},
            )
        iep_id = iep.interface_ep_id or self.new_id("iep")
        created_at = iep.created_at or self.now()
        support_hash = sha256_hex(",".join(sorted(iep.support_node_ids)))
        stored = iep.model_copy(update={"interface_ep_id": iep_id, "created_at": created_at})
        with self.transaction():
            self._execute(
                "INSERT INTO interface_eps(interface_ep_id, scale, atom_id, slots_json, slot_support_json, "
                "support_node_ids_json, support_edge_ids_json, boundary_edges_json, invariants_json, residuals_json, "
                "evidence_anchor_ids_json, primary_support_node_id, primary_anchor_id, coarsen_op_version, "
                "cohesion_score, support_hash, user_id, agent_id, run_id, valid, valid_to, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    iep_id,
                    iep.scale,
                    iep.atom_id,
                    canonical_json(iep.slots.model_dump()),
                    canonical_json({k: list(v) for k, v in iep.slot_support.items()}),
                    canonical_json(list(iep.support_node_ids)),
                    canonical_json(list(iep.support_edge_ids)),
                    canonical_json([b.model_dump() for b in iep.boundary_edges]),
                    canonical_json([i.model_dump() for i in iep.invariants]),
                    canonical_json([r.model_dump() for r in iep.residuals]),
                    canonical_json(evidence),
                    iep.support_node_ids[0],
                    evidence[0],
                    iep.coarsen_op_version,
                    float(iep.cohesion_score),
                    support_hash,
                    iep.scope.user_id,
                    iep.scope.agent_id,
                    iep.scope.run_id,
                    1 if iep.valid else 0,
                    iep.valid_to,
                    created_at,
                ),
            )
            members = (
                [("node", n) for n in iep.support_node_ids]
                + [("edge", e) for e in iep.support_edge_ids]
                + [("boundary_edge", b.edge_id) for b in iep.boundary_edges]
            )
            for kind, member_id in members:
                self._execute(
                    "INSERT OR IGNORE INTO interface_ep_support(interface_ep_id, member_kind, member_id) VALUES (?,?,?)",
                    (iep_id, kind, member_id),
                )
        return stored

    def get_interface_ep(self, iep_id: str) -> InterfaceEP | None:
        """Fetch one InterfaceEP.
        读取一个 InterfaceEP。

        Input / 输入：
            iep_id: InterfaceEP id.
            iep_id：InterfaceEP id。

        Output / 输出：
            InterfaceEP or None.
            InterfaceEP 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        ieps = self._select_ieps("SELECT * FROM interface_eps WHERE interface_ep_id = ?", (iep_id,))
        return ieps[0] if ieps else None

    def find_interface_ep_by_support_hash(self, scope: Scope, support_hash: str) -> InterfaceEP | None:
        """The valid InterfaceEP of a scope with a given support hash (re-coarsen dedup).
        范围内具有给定支持哈希的有效 InterfaceEP（再粗化去重）。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            support_hash: sha256 of the sorted support node ids.
            support_hash：排序支持节点 id 的 sha256。

        Output / 输出：
            InterfaceEP or None (newest valid one).
            InterfaceEP 或 None（最新的有效者）。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        ieps = self._select_ieps(
            f"SELECT * FROM interface_eps WHERE {where} AND support_hash = ? AND valid = 1 "
            "ORDER BY created_at DESC, interface_ep_id DESC LIMIT 1",
            params + [support_hash],
        )
        return ieps[0] if ieps else None

    def list_interface_eps(
        self, scope: Scope, *, valid_only: bool = True, scale: str | None = None
    ) -> list[InterfaceEP]:
        """InterfaceEPs of a scope ordered by (created_at, interface_ep_id).
        按 (created_at, interface_ep_id) 排序的范围内 InterfaceEP。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            valid_only: Keep valid rows only.
            valid_only：仅保留有效行。

            scale: Optional scale filter.
            scale：可选的尺度过滤。

        Output / 输出：
            InterfaceEPs.
            InterfaceEP 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        clauses = [where]
        if valid_only:
            clauses.append("valid = 1")
        if scale is not None:
            clauses.append("scale = ?")
            params.append(str(scale))
        return self._select_ieps(
            f"SELECT * FROM interface_eps WHERE {' AND '.join(clauses)} ORDER BY created_at, interface_ep_id", params
        )

    def invalidate_interface_ep(self, iep_id: str, *, at: str) -> None:
        """Invalidate an InterfaceEP: valid = 0, valid_to = at, its atom archived ('revised'), node invalid.
        使 InterfaceEP 失效：valid = 0、valid_to = at、其原子归档（'revised'）、节点失效。

        Input / 输入：
            iep_id: InterfaceEP id.
            iep_id：InterfaceEP id。

            at: Instant of invalidation.
            at：失效时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates interface_eps, memory_atoms, graph_nodes.
            更新 interface_eps、memory_atoms、graph_nodes。
        """
        instant = to_utc_iso(at)
        with self.transaction():
            row = self._execute("SELECT atom_id FROM interface_eps WHERE interface_ep_id = ?", (iep_id,)).fetchone()
            self._execute(
                "UPDATE interface_eps SET valid = 0, valid_to = ? WHERE interface_ep_id = ?", (instant, iep_id)
            )
            if row is not None:
                atom = self._execute(
                    "SELECT archived_reason FROM memory_atoms WHERE atom_id = ?", (row["atom_id"],)
                ).fetchone()
                if atom is not None and atom["archived_reason"] is None:
                    self.archive_atom(row["atom_id"], reason="revised", at=instant)
                else:
                    self._execute(
                        "UPDATE graph_nodes SET valid = 0, invalidated_at = ? WHERE node_id = ?",
                        (instant, row["atom_id"]),
                    )

    def revalidate_interface_ep(self, iep_id: str, *, at: str) -> None:
        """Reverse invalidate_interface_ep.
        撤销 invalidate_interface_ep。

        Input / 输入：
            iep_id: InterfaceEP id.
            iep_id：InterfaceEP id。

            at: Instant of the change.
            at：变更时刻。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates interface_eps, memory_atoms, graph_nodes.
            更新 interface_eps、memory_atoms、graph_nodes。
        """
        instant = to_utc_iso(at)
        with self.transaction():
            row = self._execute("SELECT atom_id FROM interface_eps WHERE interface_ep_id = ?", (iep_id,)).fetchone()
            self._execute("UPDATE interface_eps SET valid = 1, valid_to = NULL WHERE interface_ep_id = ?", (iep_id,))
            if row is not None:
                self.restore_atom(row["atom_id"], at=instant)

    def support_members(self, iep_id: str) -> dict[str, list[str]]:
        """Support members of an InterfaceEP grouped by kind.
        按种类分组的 InterfaceEP 支持成员。

        Input / 输入：
            iep_id: InterfaceEP id.
            iep_id：InterfaceEP id。

        Output / 输出：
            {'node': [...], 'edge': [...], 'boundary_edge': [...]} in insertion order.
            按插入顺序的 {'node': [...], 'edge': [...], 'boundary_edge': [...]}。

        State effect / 状态影响：
            None.
            无。
        """
        out: dict[str, list[str]] = {"node": [], "edge": [], "boundary_edge": []}
        rows = self._execute(
            "SELECT member_kind, member_id FROM interface_ep_support WHERE interface_ep_id = ? ORDER BY rowid",
            (iep_id,),
        ).fetchall()
        for row in rows:
            out.setdefault(row["member_kind"], []).append(row["member_id"])
        return out

    def interface_eps_containing(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[InterfaceEP]:
        """InterfaceEPs whose support contains any of the nodes.
        支持集包含任一给定节点的 InterfaceEP。

        Input / 输入：
            node_ids: Node ids.
            node_ids：节点 id。

            valid_only: Keep valid rows only.
            valid_only：仅保留有效行。

        Output / 输出：
            InterfaceEPs ordered by (created_at, interface_ep_id).
            按 (created_at, interface_ep_id) 排序的 InterfaceEP。

        State effect / 状态影响：
            None.
            无。
        """
        ids = list(dict.fromkeys(str(i) for i in node_ids))
        if not ids:
            return []
        found: dict[str, InterfaceEP] = {}
        for chunk in _chunks(ids):
            clause = "" if not valid_only else " AND i.valid = 1"
            for iep in self._select_ieps(
                "SELECT DISTINCT i.* FROM interface_eps i JOIN interface_ep_support s "
                "ON s.interface_ep_id = i.interface_ep_id WHERE s.member_kind = 'node' "
                f"AND s.member_id IN ({_placeholders(len(chunk))}){clause} ORDER BY i.created_at, i.interface_ep_id",
                list(chunk),
            ):
                found[iep.interface_ep_id] = iep
        return sorted(found.values(), key=lambda i: (i.created_at or "", i.interface_ep_id))

    # ------------------------------------------------------------------ ops
    # ------------------------------------------------------------------ ops（运维）
    def insert_feedback(self, fb: FeedbackEvent) -> str:
        """Persist a feedback record (anchor pre-checked).
        持久化反馈记录（预检锚点）。

        Input / 输入：
            fb: The FeedbackEvent; target_id None stores the scope key.
            fb：FeedbackEvent；target_id 为 None 时存储范围键。

        Output / 输出：
            The feedback id.
            反馈 id。

        State effect / 状态影响：
            Writes feedback_events.
            写入 feedback_events。
        """
        self._require_anchors([fb.anchor_id], obj="feedback", obj_id=fb.feedback_id or None)
        feedback_id = fb.feedback_id or self.new_id("fb")
        with self.transaction():
            self._execute(
                "INSERT INTO feedback_events(feedback_id, target_id, target_kind, feedback, reason, correction_json, "
                "actor_id, anchor_id, actions_json, user_id, agent_id, run_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feedback_id,
                    fb.target_id if fb.target_id is not None else fb.scope.key(),
                    fb.target_kind,
                    fb.feedback,
                    fb.reason,
                    canonical_json(fb.correction) if fb.correction is not None else None,
                    fb.actor_id,
                    fb.anchor_id,
                    canonical_json(list(fb.actions)),
                    fb.scope.user_id,
                    fb.scope.agent_id,
                    fb.scope.run_id,
                    fb.created_at,
                ),
            )
        return feedback_id

    def list_feedback(self, target_id: str) -> list[FeedbackEvent]:
        """Feedback records for a target ordered by (created_at, feedback_id).
        按 (created_at, feedback_id) 排序的某目标的反馈记录。

        Input / 输入：
            target_id: Target id (or scope key).
            target_id：目标 id（或范围键）。

        Output / 输出：
            FeedbackEvents.
            FeedbackEvent 列表。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute(
            "SELECT * FROM feedback_events WHERE target_id = ? ORDER BY created_at, feedback_id", (target_id,)
        ).fetchall()
        return [
            FeedbackEvent(
                feedback_id=r["feedback_id"],
                target_id=r["target_id"],
                target_kind=r["target_kind"],
                feedback=r["feedback"],
                reason=r["reason"],
                correction=_loads(r["correction_json"], None),
                actor_id=r["actor_id"],
                anchor_id=r["anchor_id"],
                actions=_loads(r["actions_json"], []),
                scope=_scope_of(r),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def add_history(
        self,
        *,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        created_at: str,
        updated_at: str,
        is_deleted: int = 0,
        actor_id: str | None = None,
        role: str | None = None,
        anchor_id: str,
        atom_id: str | None = None,
        edge_id: str | None = None,
        successor_id: str | None = None,
        details: dict | None = None,
    ) -> str:
        """Append a witnessed history row (Mem0's columns plus HMS provenance; seq from next_seq()).
        追加带见证的历史行（Mem0 列加 HMS 出处；seq 来自 next_seq()）。

        Input / 输入：
            memory_id: Lineage id.
            memory_id：谱系 id。

            old_memory: Previous text or None.
            old_memory：先前文本或 None。

            new_memory: New text or None.
            new_memory：新文本或 None。

            event: HistoryEvent value.
            event：HistoryEvent 值。

            created_at: Mem0 created_at.
            created_at：Mem0 的 created_at。

            updated_at: Mem0 updated_at.
            updated_at：Mem0 的 updated_at。

            is_deleted: 1 for DELETE rows.
            is_deleted：DELETE 行为 1。

            actor_id: Actor id.
            actor_id：行动者 id。

            role: Message role.
            role：消息角色。

            anchor_id: Witness anchor (pre-checked).
            anchor_id：见证锚点（预检）。

            atom_id: Version written/affected.
            atom_id：写入/受影响的版本。

            edge_id: Related edge.
            edge_id：相关的边。

            successor_id: Successor atom id.
            successor_id：后继原子 id。

            details: Extra JSON details.
            details：额外 JSON 细节。

        Output / 输出：
            The history row id (prefix 'hst').
            历史行 id（前缀 'hst'）。

        State effect / 状态影响：
            Writes history and meta.history_seq.
            写入 history 与 meta.history_seq。
        """
        self._require_anchors([anchor_id], obj="history", obj_id=None)
        with self.transaction():
            row_id = self.new_id("hst")
            seq = self.next_seq("history_seq")
            self._execute(
                "INSERT INTO history(id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, "
                "actor_id, role, anchor_id, atom_id, edge_id, successor_id, details_json, seq) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id,
                    memory_id,
                    old_memory,
                    new_memory,
                    str(event),
                    to_utc_iso(created_at),
                    to_utc_iso(updated_at),
                    1 if is_deleted else 0,
                    actor_id,
                    role,
                    anchor_id,
                    atom_id,
                    edge_id,
                    successor_id,
                    canonical_json(details or {}),
                    seq,
                ),
            )
            return row_id

    def get_history(self, memory_id: str) -> list[dict]:
        """History rows of a lineage in seq order (Mem0's 10 keys plus anchor_id).
        按 seq 顺序的谱系历史行（Mem0 的 10 个键加 anchor_id）。

        Input / 输入：
            memory_id: Lineage id.
            memory_id：谱系 id。

        Output / 输出：
            List of row dicts; is_deleted is a bool.
            行字典列表；is_deleted 为布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        rows = self._execute("SELECT * FROM history WHERE memory_id = ? ORDER BY seq", (memory_id,)).fetchall()
        return [
            {
                "id": r["id"],
                "memory_id": r["memory_id"],
                "old_memory": r["old_memory"],
                "new_memory": r["new_memory"],
                "event": r["event"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "is_deleted": bool(r["is_deleted"]),
                "actor_id": r["actor_id"],
                "role": r["role"],
                "anchor_id": r["anchor_id"],
            }
            for r in rows
        ]

    def history_seq(self) -> int:
        """Current value of the history counter.
        历史计数器的当前值。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Integer (0 before the first row).
            整数（首行之前为 0）。

        State effect / 状态影响：
            None.
            无。
        """
        return int(self.get_meta("history_seq", "0") or 0)

    def insert_checkpoint(self, cp: Checkpoint) -> str:
        """Persist a checkpoint snapshot (anchor pre-checked).
        持久化检查点快照（预检锚点）。

        Input / 输入：
            cp: The Checkpoint.
            cp：Checkpoint。

        Output / 输出：
            The checkpoint id.
            检查点 id。

        State effect / 状态影响：
            Writes checkpoints.
            写入 checkpoints。
        """
        self._require_anchors([cp.anchor_id], obj="checkpoint", obj_id=cp.checkpoint_id or None)
        checkpoint_id = cp.checkpoint_id or self.new_id("ckp")
        snapshot = {
            "atom_state": cp.atom_state,
            "edge_state": cp.edge_state,
            "iep_state": cp.iep_state,
            "ep_state": cp.ep_state,
        }
        with self.transaction():
            self._execute(
                "INSERT INTO checkpoints(checkpoint_id, label, anchor_id, user_id, agent_id, run_id, snapshot_json, "
                "history_seq, snapshot_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    checkpoint_id,
                    cp.label,
                    cp.anchor_id,
                    cp.scope.user_id,
                    cp.scope.agent_id,
                    cp.scope.run_id,
                    canonical_json(snapshot),
                    int(cp.history_seq),
                    cp.snapshot_hash,
                    cp.created_at,
                ),
            )
        return checkpoint_id

    def _checkpoint_from_row(self, row: sqlite3.Row) -> Checkpoint:
        """Checkpoint from a checkpoints row.
        由 checkpoints 行构造 Checkpoint。
        """
        snapshot = _loads(row["snapshot_json"], {})
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            label=row["label"],
            anchor_id=row["anchor_id"],
            scope=_scope_of(row),
            atom_state=snapshot.get("atom_state", {}),
            edge_state=snapshot.get("edge_state", {}),
            iep_state=snapshot.get("iep_state", {}),
            ep_state=snapshot.get("ep_state", {}),
            history_seq=int(row["history_seq"]),
            snapshot_hash=row["snapshot_hash"],
            created_at=row["created_at"],
        )

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Fetch one checkpoint.
        读取一个检查点。

        Input / 输入：
            checkpoint_id: Checkpoint id.
            checkpoint_id：检查点 id。

        Output / 输出：
            Checkpoint or None.
            Checkpoint 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)).fetchone()
        return self._checkpoint_from_row(row) if row else None

    def list_checkpoints(self, scope: Scope) -> list[Checkpoint]:
        """Checkpoints of a scope ordered by (created_at, checkpoint_id).
        按 (created_at, checkpoint_id) 排序的范围内检查点。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

        Output / 输出：
            Checkpoints.
            Checkpoint 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        rows = self._execute(
            f"SELECT * FROM checkpoints WHERE {where} ORDER BY created_at, checkpoint_id", params
        ).fetchall()
        return [self._checkpoint_from_row(r) for r in rows]

    def _transition_from_row(self, row: sqlite3.Row) -> Transition:
        """Transition from a transitions row.
        由 transitions 行构造 Transition。
        """
        return Transition(
            transition_id=row["transition_id"],
            source_kind=row["source_kind"],
            source_id=row["source_id"],
            scale=row["scale"],
            s_t=_loads(row["s_t_json"], {}),
            a_t=row["a_t"],
            s_t1=_loads(row["s_t1_json"], {}),
            r_t=row["r_t"],
            kappa_t=_loads(row["kappa_t_json"], {}),
            w_t=_loads(row["w_t_json"], []) or [row["w_t_primary_anchor_id"]],
            time_t=row["time_t"],
            time_t1=row["time_t1"],
            scope=_scope_of(row),
            created_at=row["created_at"],
        )

    def upsert_transition(self, tr: Transition) -> str:
        """Insert a transition once per (source_kind, source_id); return the existing id when ignored.
        每个 (source_kind, source_id) 只插入一次转移；被忽略时返回已有 id。

        Input / 输入：
            tr: The Transition (w_t pre-checked).
            tr：Transition（预检 w_t）。

        Output / 输出：
            The transition id (existing or new).
            转移 id（已有或新建）。

        State effect / 状态影响：
            Writes transitions.
            写入 transitions。
        """
        w_t = self._require_anchors(tr.w_t, obj="transition", obj_id=tr.transition_id or None)
        with self.transaction():
            existing = self._execute(
                "SELECT transition_id FROM transitions WHERE source_kind = ? AND source_id = ?",
                (tr.source_kind, tr.source_id),
            ).fetchone()
            if existing is not None:
                return existing["transition_id"]
            transition_id = tr.transition_id or self.new_id("trn")
            cur = self._execute(
                "INSERT OR IGNORE INTO transitions(transition_id, source_kind, source_id, scale, s_t_json, a_t, "
                "s_t1_json, r_t, kappa_t_json, w_t_json, w_t_primary_anchor_id, time_t, time_t1, user_id, agent_id, "
                "run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    transition_id,
                    tr.source_kind,
                    tr.source_id,
                    tr.scale,
                    canonical_json(tr.s_t),
                    tr.a_t,
                    canonical_json(tr.s_t1),
                    tr.r_t,
                    canonical_json(tr.kappa_t),
                    canonical_json(w_t),
                    w_t[0],
                    tr.time_t,
                    tr.time_t1,
                    tr.scope.user_id,
                    tr.scope.agent_id,
                    tr.scope.run_id,
                    tr.created_at or self.now(),
                ),
            )
            if cur.rowcount == 1:
                return transition_id
            row = self._execute(
                "SELECT transition_id FROM transitions WHERE source_kind = ? AND source_id = ?",
                (tr.source_kind, tr.source_id),
            ).fetchone()
            return row["transition_id"] if row else transition_id

    def list_transitions(
        self,
        scope: Scope,
        *,
        since: str | None = None,
        scale: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[Transition]:
        """Transitions of a scope ordered by (time_t1, KIND_ORDER[source_kind], source_id).
        按 (time_t1, KIND_ORDER[source_kind], source_id) 排序的范围内转移。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            since: Inclusive lower bound on time_t1.
            since：time_t1 的下界（含）。

            scale: Optional scale filter.
            scale：可选的尺度过滤。

            kinds: source_kind whitelist.
            kinds：source_kind 白名单。

        Output / 输出：
            Transitions.
            Transition 列表。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope)
        clauses = [where]
        if since is not None:
            clauses.append("time_t1 >= ?")
            params.append(to_utc_iso(since))
        if scale is not None:
            clauses.append("scale = ?")
            params.append(str(scale))
        if kinds:
            values = [str(k) for k in kinds]
            clauses.append(f"source_kind IN ({_placeholders(len(values))})")
            params.extend(values)
        order_case = " ".join(f"WHEN '{kind}' THEN {rank}" for kind, rank in KIND_ORDER.items())
        rows = self._execute(
            f"SELECT * FROM transitions WHERE {' AND '.join(clauses)} "
            f"ORDER BY time_t1, CASE source_kind {order_case} ELSE 99 END, source_id",
            params,
        ).fetchall()
        return [self._transition_from_row(r) for r in rows]

    def insert_compiled_state(
        self, *, scope: Scope, as_of: str, mode: str, card: str, context_json: str, used_tokens: int, anchor_id: str
    ) -> str:
        """Persist a compiled Work State snapshot (consolidate step 7).
        持久化编译后的工作状态快照（consolidate 第 7 步）。

        Input / 输入：
            scope: Scope of the card.
            scope：卡片的范围。

            as_of: ContextIR.as_of.
            as_of：ContextIR.as_of。

            mode: Compile mode used.
            mode：所用编译模式。

            card: ContextIR.to_text().
            card：ContextIR.to_text()。

            context_json: ContextIR.model_dump_json().
            context_json：ContextIR.model_dump_json()。

            used_tokens: ContextIR.used_tokens.
            used_tokens：ContextIR.used_tokens。

            anchor_id: The scheduler anchor (pre-checked).
            anchor_id：调度器锚点（预检）。

        Output / 输出：
            The compiled state id (prefix 'cst').
            编译状态 id（前缀 'cst'）。

        State effect / 状态影响：
            Writes compiled_states.
            写入 compiled_states。
        """
        self._require_anchors([anchor_id], obj="compiled_state", obj_id=None)
        compiled_id = self.new_id("cst")
        with self.transaction():
            self._execute(
                "INSERT INTO compiled_states(compiled_id, scope_key, user_id, agent_id, run_id, as_of, mode, card, "
                "context_json, hash, used_tokens, anchor_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    compiled_id,
                    scope.key(),
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    to_utc_iso(as_of),
                    str(mode),
                    card,
                    context_json,
                    sha256_hex(card),
                    int(used_tokens),
                    anchor_id,
                    self.now(),
                ),
            )
        return compiled_id

    def last_compiled_state(self, scope: Scope) -> dict | None:
        """The newest compiled state of a scope key.
        某范围键最新的编译状态。

        Input / 输入：
            scope: Scope whose key() is matched exactly.
            scope：按其 key() 精确匹配的范围。

        Output / 输出：
            {'compiled_id','as_of','hash','used_tokens','created_at','card'} or None.
            {'compiled_id','as_of','hash','used_tokens','created_at','card'} 或 None。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute(
            "SELECT compiled_id, as_of, hash, used_tokens, created_at, card FROM compiled_states WHERE scope_key = ? "
            "ORDER BY created_at DESC, compiled_id DESC LIMIT 1",
            (scope.key(),),
        ).fetchone()
        if row is None:
            return None
        return {
            "compiled_id": row["compiled_id"],
            "as_of": row["as_of"],
            "hash": row["hash"],
            "used_tokens": int(row["used_tokens"]),
            "created_at": row["created_at"],
            "card": row["card"],
        }

    # ------------------------------------------------------------------ retention (§16.6)
    # ------------------------------------------------------------------ 保留策略（§16.6）
    def purge_candidates(self, scope: Scope, *, valid_to_before: str) -> list[MemoryAtom]:
        """Archived atom versions of a (possibly partial) scope whose valid_to ends before an instant.
        某（可为部分）范围内、valid_to 早于给定时刻的已归档原子版本。

        Input / 输入：
            scope: Scope filter with Scope.matches semantics (an empty scope matches every row).
            scope：按 Scope.matches 语义过滤的范围（空范围匹配所有行）。

            valid_to_before: Exclusive upper bound on valid_to (any to_utc_iso form).
            valid_to_before：valid_to 的开区间上界（to_utc_iso 接受的任意形式）。

        Output / 输出：
            Hydrated atoms with archived_reason set and valid_to < bound, ordered by atom_id.
            archived_reason 非空且 valid_to < 上界的水合原子，按 atom_id 排序。

        State effect / 状态影响：
            None.
            无。
        """
        return self.axis_select(
            scope,
            where_sql="memory_atoms.archived_reason IS NOT NULL AND memory_atoms.valid_to IS NOT NULL "
            "AND memory_atoms.valid_to < ?",
            params=[to_utc_iso(valid_to_before)],
            include_expired=True,
            order="atom_id ASC",
        )

    def edge_endpoint_ids(self, atom_ids: Iterable[str], *, valid_only: bool = False) -> set[str]:
        """The subset of node ids that are an endpoint of at least one edge (of any validity by default).
        节点 id 中至少作为一条边（默认不分有效性）端点的那部分。

        Input / 输入：
            atom_ids: Candidate node ids.
            atom_ids：候选节点 id。

            valid_only: Count valid edges only.
            valid_only：只统计有效边。

        Output / 输出：
            Set of ids that graph_edges references as src_id or dst_id.
            被 graph_edges 作为 src_id 或 dst_id 引用的 id 集合。

        State effect / 状态影响：
            None.
            无。
        """
        ids = list(dict.fromkeys(str(i) for i in atom_ids))
        found: set[str] = set()
        for chunk in _chunks(ids):
            marks = _placeholders(len(chunk))
            valid = " AND valid = 1" if valid_only else ""
            rows = self._execute(
                f"SELECT src_id, dst_id FROM graph_edges WHERE (src_id IN ({marks}) OR dst_id IN ({marks})){valid}",
                list(chunk) + list(chunk),
            ).fetchall()
            wanted = set(chunk)
            for r in rows:
                found.update(x for x in (r["src_id"], r["dst_id"]) if x in wanted)
        return found

    def interface_ep_member_ids(self, atom_ids: Iterable[str]) -> set[str]:
        """The subset of atom ids an InterfaceEP row cites (its own atom, primary support node or a support member).
        原子 id 中被某 InterfaceEP 行引用（自身原子、主支持节点或支持成员）的那部分。

        Input / 输入：
            atom_ids: Candidate atom / node ids.
            atom_ids：候选原子 / 节点 id。

        Output / 输出：
            Set of referenced ids.
            被引用的 id 集合。

        State effect / 状态影响：
            None.
            无。
        """
        ids = list(dict.fromkeys(str(i) for i in atom_ids))
        found: set[str] = set()
        for chunk in _chunks(ids):
            marks = _placeholders(len(chunk))
            rows = self._execute(
                f"SELECT atom_id AS id FROM interface_eps WHERE atom_id IN ({marks}) "
                f"UNION SELECT primary_support_node_id FROM interface_eps WHERE primary_support_node_id IN ({marks}) "
                f"UNION SELECT member_id FROM interface_ep_support WHERE member_kind = 'node' AND member_id IN ({marks})",
                list(chunk) * 3,
            ).fetchall()
            found.update(r["id"] for r in rows)
        return found

    def checkpoint_live_atom_ids(self, scope: Scope | None = None) -> set[str]:
        """Atom ids that any checkpoint snapshot records as live (archived_reason None at snapshot time).
        任一检查点快照记录为存活（快照时 archived_reason 为 None）的原子 id。

        Input / 输入：
            scope: Optional Scope.matches filter over checkpoints (None → every checkpoint).
            scope：对 checkpoints 的可选 Scope.matches 过滤（None → 所有检查点）。

        Output / 输出：
            Set of atom ids a rollback could still need to restore.
            回滚仍可能需要恢复的原子 id 集合。

        State effect / 状态影响：
            None.
            无。
        """
        where, params = _scope_where(scope) if scope is not None else ("1=1", [])
        rows = self._execute(f"SELECT snapshot_json FROM checkpoints WHERE {where}", params).fetchall()
        found: set[str] = set()
        for r in rows:
            atom_state = _loads(r["snapshot_json"], {}).get("atom_state", {})
            found.update(
                a for a, st in atom_state.items() if isinstance(st, dict) and st.get("archived_reason") is None
            )
        return found

    def ep_reference_ids(self, ep_ids: Iterable[str], *, excluding_atoms: Iterable[str] = ()) -> set[str]:
        """EP ids still cited by another EP's uses_ep_ids, a surviving atom's source_ep_ids or a transition.
        仍被其它 EP 的 uses_ep_ids、幸存原子的 source_ep_ids 或某条转移引用的 EP id。

        Input / 输入：
            ep_ids: Candidate EP ids.
            ep_ids：候选 EP id。

            excluding_atoms: Atom ids about to be purged (their own source_ep_ids do not count).
            excluding_atoms：即将清理的原子 id（它们自身的 source_ep_ids 不计）。

        Output / 输出：
            Set of referenced EP ids.
            被引用的 EP id 集合。

        State effect / 状态影响：
            None.
            无。
        """
        skip = set(str(i) for i in excluding_atoms)
        found: set[str] = set()
        for ep_id in sorted(set(str(i) for i in ep_ids)):
            needle = f'%"{ep_id}"%'
            row = self._execute(
                "SELECT (SELECT COUNT(*) FROM eps WHERE ep_id <> ? AND uses_ep_ids_json LIKE ?) "
                "+ (SELECT COUNT(*) FROM transitions WHERE source_kind = 'ep' AND source_id = ?) AS n",
                (ep_id, needle, ep_id),
            ).fetchone()
            count = int(row["n"]) if row else 0
            if count == 0:
                atoms = self._execute(
                    "SELECT atom_id FROM memory_atoms WHERE source_ep_ids_json LIKE ?", (needle,)
                ).fetchall()
                count = sum(1 for a in atoms if a["atom_id"] not in skip)
            if count:
                found.add(ep_id)
        return found

    def anchor_reference_count(self, anchor_id: str) -> int:
        """How many rows cite an anchor across every witness column and JSON id list of the schema.
        某锚点在模式的所有见证列与 JSON id 列表中被引用的行数。

        Input / 输入：
            anchor_id: Anchor id.
            anchor_id：锚点 id。

        Output / 输出：
            Total reference count (0 → nothing in the database cites the anchor any more).
            引用总数（0 → 库中再无任何对象引用该锚点）。

        State effect / 状态影响：
            None.
            无。
        """
        parts = [f"(SELECT COUNT(*) FROM {table} WHERE {col} = ?)" for table, col in _ANCHOR_REF_COLUMNS]
        params: list[Any] = [anchor_id] * len(_ANCHOR_REF_COLUMNS)
        for table, col in _ANCHOR_REF_JSON:
            parts.append(f"(SELECT COUNT(*) FROM {table} WHERE {col} LIKE ?)")
            params.append(f'%"{anchor_id}"%')
        row = self._execute("SELECT " + " + ".join(parts) + " AS n", params).fetchone()
        return int(row["n"]) if row else 0

    def purge_rows(self, atom_ids: Iterable[str], *, evidence: bool = False) -> dict:
        """Physically delete archived atom versions (the retention escape hatch): the ONLY DELETE path of state rows.
        物理删除已归档的原子版本（保留策略逃生舱）：状态行唯一的 DELETE 路径。

        Input / 输入：
            atom_ids: Versions already vetted by ops.retention.purge (archived, edge-free, not checkpoint-live, ...).
            atom_ids：已由 ops.retention.purge 审核过的版本（已归档、无边、非检查点存活……）。

            evidence: False keeps every anchor and source event; True also deletes the anchors of the purged rows that nothing in the database cites any more, together with their source events.
            evidence：False 保留全部锚点与源事件；True 还会删除被清行的、库中再无任何对象引用的锚点及其源事件。

        Output / 输出：
            {'atoms', 'observations', 'eps', 'anchors', 'events': [ids], 'nodes', 'vectors', 'links': counts}.
            {'atoms', 'observations', 'eps', 'anchors', 'events': [id 列表], 'nodes', 'vectors', 'links': 计数}。

        State effect / 状态影响：
            One transaction: writes meta.purge_in_progress = '1' (which opens the WHEN gate of the seven BEFORE DELETE triggers), deletes atom_vectors / atom_topic / atom_entity / atom_anchor / graph_nodes / memory_atoms rows and the atom's own observations / eps source row, optionally orphaned anchors + source_events, then removes the flag before COMMIT (a failure rolls everything back, flag included).
            一个事务：写入 meta.purge_in_progress = '1'（打开七个 BEFORE DELETE 触发器的 WHEN 门），删除 atom_vectors / atom_topic / atom_entity / atom_anchor / graph_nodes / memory_atoms 行与该原子自身的 observations / eps 来源行，可选地删除孤立锚点 + source_events，随后在 COMMIT 前移除标志（失败则全部回滚，标志一并回滚）。
        """
        ids = list(dict.fromkeys(str(i) for i in atom_ids))
        report: dict[str, Any] = {
            "atoms": [],
            "observations": [],
            "eps": [],
            "anchors": [],
            "events": [],
            "nodes": 0,
            "vectors": 0,
            "links": 0,
        }
        if not ids:
            return report
        with self.transaction():
            sources = {atom_id: self._atom_source(atom_id) for atom_id in ids}
            present = [atom_id for atom_id in ids if sources[atom_id] is not None]
            anchors_before = self._anchors_for(present, include_disputed=True)
            primaries = {
                r["atom_id"]: r["primary_anchor_id"]
                for chunk in _chunks(present)
                for r in self._execute(
                    f"SELECT atom_id, primary_anchor_id FROM memory_atoms WHERE atom_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall()
            }
            self._execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, '1')", (PURGE_FLAG_KEY,))
            try:
                for chunk in _chunks(present):
                    marks = _placeholders(len(chunk))
                    if self.capabilities.get("fts5"):
                        # The virtual table has no trigger gate; its row (if any) goes with the version.
                        # 虚拟表没有触发器门；其行（如有）随版本一起删除。
                        self._execute(f"DELETE FROM atom_fts WHERE atom_id IN ({marks})", chunk)
                    for table, col in _PURGE_TABLES:
                        rows = self._execute(f"DELETE FROM {table} WHERE {col} IN ({marks})", chunk)
                        if table == "atom_vectors":
                            report["vectors"] += max(0, rows.rowcount)
                        elif table == "graph_nodes":
                            report["nodes"] += max(0, rows.rowcount)
                        elif table in ("atom_topic", "atom_entity", "atom_anchor"):
                            report["links"] += max(0, rows.rowcount)
                report["atoms"] = list(present)
                for atom_id in present:
                    kind, source_id = sources[atom_id]
                    if kind == SourceKind.observation.value:
                        rows = self._execute(
                            "DELETE FROM observations WHERE obs_id = ? AND (atom_id IS NULL OR atom_id = ?)",
                            (source_id, atom_id),
                        )
                        if rows.rowcount > 0:
                            report["observations"].append(source_id)
                    elif kind == SourceKind.ep.value:
                        rows = self._execute(
                            "DELETE FROM eps WHERE ep_id = ? AND (atom_id IS NULL OR atom_id = ?)", (source_id, atom_id)
                        )
                        if rows.rowcount > 0:
                            report["eps"].append(source_id)
                if evidence:
                    cited: set[str] = set(primaries.values())
                    for linked in anchors_before.values():
                        cited.update(linked)
                    for anchor_id in sorted(cited):
                        if self.anchor_reference_count(anchor_id):
                            continue
                        row = self._execute(
                            "SELECT source_event_id FROM anchors WHERE anchor_id = ?", (anchor_id,)
                        ).fetchone()
                        if row is None:
                            continue
                        self._execute("DELETE FROM anchors WHERE anchor_id = ?", (anchor_id,))
                        self._execute("DELETE FROM source_events WHERE event_id = ?", (row["source_event_id"],))
                        self._verified_anchors.discard(anchor_id)
                        report["anchors"].append(anchor_id)
                        report["events"].append(row["source_event_id"])
            finally:
                self._execute("DELETE FROM meta WHERE key = ?", (PURGE_FLAG_KEY,))
        return report

    def vacuum(self) -> dict:
        """Run VACUUM outside any transaction and report the database size before and after.
        在任何事务之外执行 VACUUM，并报告前后的数据库大小。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'} measured as page_count * page_size (a file database additionally checkpoints and truncates its WAL first so the numbers match the file on disk).
            {'path', 'before_bytes', 'after_bytes', 'freed_bytes'}，按 page_count * page_size 计量（文件库会先 checkpoint 并截断 WAL，使数字与磁盘文件一致）。

        State effect / 状态影响：
            Rewrites the database file (STORE_001 when called inside transaction()).
            重写数据库文件（在 transaction() 内调用时抛 STORE_001）。
        """
        with self._lock:
            if self._depth:
                raise StoreError("vacuum() cannot run inside a transaction", "STORE_001")
            conn = self._connection()
            try:
                before = self._db_bytes()
                conn.execute("VACUUM")
                if self._is_file:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                after = self._db_bytes()
            except sqlite3.Error as exc:
                raise StoreError(f"VACUUM failed: {exc}", "STORE_001", details={"path": self.path}) from exc
        return {"path": self.path, "before_bytes": before, "after_bytes": after, "freed_bytes": max(0, before - after)}

    def _db_bytes(self) -> int:
        """page_count * page_size of the open database.
        已打开数据库的 page_count * page_size。
        """
        conn = self._connection()
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        return pages * size

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a meta key.
        写入/更新 meta 键。

        Input / 输入：
            key: Meta key.
            key：meta 键。

            value: Text value.
            value：文本值。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Writes meta.
            写入 meta。
        """
        with self.transaction():
            self._execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """Read a meta key.
        读取 meta 键。

        Input / 输入：
            key: Meta key.
            key：meta 键。

            default: Returned when absent.
            default：缺失时的返回值。

        Output / 输出：
            The stored text or the default.
            存储的文本或默认值。

        State effect / 状态影响：
            None.
            无。
        """
        row = self._execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


__all__ = [
    "AtomLite",
    "DEFAULT_ATOM_ORDER",
    "LIVE_SQL",
    "NODE_KIND_OF_SOURCE",
    "EntityKey",
    "StateStore",
    "TableTrace",
]
