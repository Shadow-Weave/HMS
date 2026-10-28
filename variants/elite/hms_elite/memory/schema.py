"""SQLite schema for all HMS planes in one file; invariants as CHECKs and triggers.
单文件多平面 SQLite 模式；不变量以 CHECK 与触发器表达。
"""

# Purpose (EN): Generate the authoritative DDL (CHECK lists from enums), own the migration ladder and capability probe.
# 目的（中）：生成权威 DDL（CHECK 列表来自枚举），拥有迁移阶梯与能力探测。

# Input (EN): A sqlite3 connection owned by StateStore.
# 输入（中）：由 StateStore 拥有的 sqlite3 连接。

# Output (EN): Applied tables/indexes/triggers, the schema version and a capabilities dict.
# 输出（中）：已应用的表/索引/触发器、模式版本与能力字典。

# State effect (EN): apply_migrations creates schema objects (migration 2 swaps seven triggers and rebuilds history,
#                    migration 3 adds the atom_fts FTS5 table and backfills it); drop_all removes every table
#                    (StateStore.reset only).
# 状态影响（中）：apply_migrations 创建模式对象（迁移 2 替换七个触发器并重建 history，迁移 3 新增 atom_fts FTS5 表并
#                回填）；drop_all 删除全部表（仅供 StateStore.reset）。

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable

from hms_elite.enums import (
    AtomType,
    EdgeType,
    EventType,
    FeedbackType,
    HistoryEvent,
    NodeKind,
    ObjectType,
    ObservationType,
    Relation,
    Reward,
    Role,
    Scale,
    Sentiment,
    SourceKind,
    TransitionKind,
    Visibility,
)
from hms_elite.exceptions import StoreError
from hms_elite.utils.text import content_tokens, tokenize

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

# Retention purge gate (§16.6): the seven BEFORE DELETE triggers below carry a WHEN clause that lets a DELETE through
# only while the meta row purge_in_progress = '1' exists. StateStore.purge_rows() writes that row inside its own
# transaction and removes it before COMMIT (a ROLLBACK drops it as well); StateStore.__init__ clears a stray row.
# Every other DELETE on these tables still raises STORE_003. UPDATE (immutability) triggers are never gated.
# 保留策略清理门（§16.6）：下面七个 BEFORE DELETE 触发器带 WHEN 子句，只有当 meta 行 purge_in_progress = '1'
# 存在时才放行 DELETE。StateStore.purge_rows() 在自己的事务内写入该行并在 COMMIT 前删除（ROLLBACK 同样会丢弃它）；
# StateStore.__init__ 会清掉残留行。这些表上的其它 DELETE 仍抛 STORE_003。UPDATE（不可变）触发器从不设门。
PURGE_FLAG_KEY = "purge_in_progress"
PURGE_GATE = f"WHEN NOT EXISTS (SELECT 1 FROM meta WHERE key = '{PURGE_FLAG_KEY}' AND value = '1') "
# (trigger name, table, RAISE message) of the gated triggers, in DDL order.
# 设门触发器的（触发器名、表、RAISE 消息），按 DDL 顺序。
PURGE_GATED_TRIGGERS: tuple[tuple[str, str, str], ...] = (
    ("trg_events_no_delete", "source_events", "HMS_INVARIANT evidence_preservation: source_events are append-only"),
    (
        "trg_anchors_no_delete",
        "anchors",
        "HMS_INVARIANT evidence_preservation: anchors are never deleted (use revoked_at)",
    ),
    ("trg_obs_no_delete", "observations", "HMS_INVARIANT revision_only: observations"),
    ("trg_eps_no_delete", "eps", "HMS_INVARIANT revision_only: eps"),
    (
        "trg_atoms_no_delete",
        "memory_atoms",
        "HMS_INVARIANT revision_only: atoms are archived (valid_to), never deleted",
    ),
    (
        "trg_atom_anchor_no_delete",
        "atom_anchor",
        "HMS_INVARIANT evidence_preservation: atom_anchor rows are disputed, never deleted",
    ),
    ("trg_nodes_no_delete", "graph_nodes", "HMS_INVARIANT revision_only: graph_nodes"),
)
# history.event vocabulary of schema version 1 (before HistoryEvent.PURGE existed); migration 2 rebuilds the CHECK.
# 模式版本 1 的 history.event 词表（HistoryEvent.PURGE 出现之前）；迁移 2 重建该 CHECK。
HISTORY_EVENTS_V1: tuple[str, ...] = tuple(e for e in HistoryEvent.values() if e != HistoryEvent.PURGE.value)

# Literal vocabularies that live on pydantic Literal fields rather than enums (§4).
# 承载在 pydantic Literal 字段（而非枚举）上的字面量词表（§4）。
CREATED_BY_VALUES: tuple[str, ...] = ("ingest", "consolidation", "feedback", "ops", "project")
ATTRIBUTED_TO_VALUES: tuple[str, ...] = ("user", "assistant")
TARGET_KIND_VALUES: tuple[str, ...] = ("atom", "edge", "interface_ep", "context_item", "anchor", "scope")
ATOM_ANCHOR_ROLE_VALUES: tuple[str, ...] = ("primary", "support", "feedback")
MEMBER_KIND_VALUES: tuple[str, ...] = ("node", "edge", "boundary_edge")
EDGE_SCALE_VALUES: tuple[str, ...] = ("L1_micro", "L2_meso", "L3_macro")
IEP_SCALE_VALUES: tuple[str, ...] = ("L2_meso", "L3_macro")
EVENT_ROLE_VALUES: tuple[str, ...] = tuple(r for r in Role.values() if r != Role.extracted.value)

# 共 22 张基表（虚拟表见下方 VIRTUAL_TABLES）
TABLES: tuple[str, ...] = (
    "source_events",
    "anchors",
    "observations",
    "eps",
    "memory_atoms",
    "atom_anchor",
    "entities",
    "atom_entity",
    "atom_topic",
    "atom_vectors",
    "graph_nodes",
    "graph_edges",
    "edge_witnesses",
    "edge_projections",
    "interface_eps",
    "interface_ep_support",
    "feedback_events",
    "history",
    "checkpoints",
    "transitions",
    "compiled_states",
    "meta",
)  # 22 base tables / 共 22 张基表

# The keyword channel's FTS5 virtual table (schema version 3): one row per non-archived atom version, `tokens` being
# the pre-tokenized keyword vocabulary of the atom (fts_tokens), so CJK bigrams and identifiers are searchable with the
# stock unicode61 tokenizer. FTS5 keeps its own shadow tables (FTS_SHADOW_TABLES); neither the virtual table nor its
# shadows appear in TABLES, stats() or TableTrace. The statement text is shared by build_ddl(3) and upgrade_3() so a
# fresh database and an upgraded one carry the same sqlite_master row.
# 关键词通道的 FTS5 虚拟表（模式版本 3）：每个未归档原子版本一行，`tokens` 为该原子预分词后的关键词词表
# （fts_tokens），因此 CJK 二元组与标识符用自带的 unicode61 分词器即可检索。FTS5 自有影子表（FTS_SHADOW_TABLES）；
# 虚拟表及其影子表都不进入 TABLES、stats() 或 TableTrace。语句文本由 build_ddl(3) 与 upgrade_3() 共用，因此全新库
# 与升级库的 sqlite_master 行相同。
VIRTUAL_TABLES: tuple[str, ...] = ("atom_fts",)  # 1 virtual table / 共 1 张虚拟表
FTS_SHADOW_TABLES: tuple[str, ...] = tuple(f"atom_fts_{suffix}" for suffix in ("data", "idx", "content", "docsize", "config"))
# The time_anchor index: a hard time gate names it (StateStore index_hint) because without ANALYZE statistics SQLite
# prefers the scope index and walks the whole scope; statistics are deliberately NOT collected (ANALYZE made the
# planner pick the validity range index for every ungated query, 4x slower — RETRIEVAL_BENCH_2026-09-10 §3).
# time_anchor 索引：硬时间门指名它（StateStore 的 index_hint），因为没有 ANALYZE 统计时 SQLite 偏好 scope 索引并
# 走遍整个 scope；这里有意**不**收集统计（ANALYZE 会让规划器给每个无门控查询选有效期范围索引，慢 4 倍——
# RETRIEVAL_BENCH_2026-09-10 §3）。
TIME_INDEX = "ix_atoms_time"
OBS_TYPE_INDEX = "ix_atoms_obs_type"
SCALE_INDEX = "ix_atoms_scale"
FTS_DDL = "CREATE VIRTUAL TABLE atom_fts USING fts5(atom_id UNINDEXED, tokens, tokenize = 'unicode61')"

# Tables that denormalize the three scope columns (stats() filters them by scope).
# 反规范化了三个 scope 列的表（stats() 按 scope 过滤这些表）。
SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "source_events",
        "anchors",
        "observations",
        "eps",
        "memory_atoms",
        "entities",
        "graph_nodes",
        "graph_edges",
        "interface_eps",
        "feedback_events",
        "checkpoints",
        "transitions",
        "compiled_states",
    }
)

# The §7 SCOPE macro: the three columns stay where the macro appears; its table-level CHECK must be the last
# constraint of the table (SQLite rejects table constraints between column definitions), so it is appended.
# §7 的 SCOPE 宏：三个列留在宏出现的位置；它的表级 CHECK 必须是该表的最后一条
# 约束（SQLite 不接受夹在列定义之间的表约束），因此它是被追加到末尾的。
SCOPE = "user_id TEXT, agent_id TEXT, run_id TEXT"
SCOPE_CHECK = "CHECK (user_id IS NOT NULL OR agent_id IS NOT NULL OR run_id IS NOT NULL)"


def _in(col: str, values) -> str:
    """Render `CHECK (col IN ('a','b'))` from a value sequence.
    由取值序列渲染 `CHECK (col IN ('a','b'))`。
    """
    return f"CHECK ({col} IN ({','.join(chr(39) + str(v) + chr(39) for v in values)}))"


def _in_or_null(col: str, values) -> str:
    """Render `CHECK (col IS NULL OR col IN (...))`.
    渲染 `CHECK (col IS NULL OR col IN (...))`。
    """
    return f"CHECK ({col} IS NULL OR {col} IN ({','.join(chr(39) + str(v) + chr(39) for v in values)}))"


def _vis(col: str) -> str:
    """The VIS macro of §7 for a column.
    §7 中 VIS 宏对某列的展开。
    """
    return _in(col, Visibility.values())


def _no_delete_trigger(name: str, table: str, message: str, *, gate: str) -> str:
    """One BEFORE DELETE RAISE trigger, with the purge WHEN gate when `gate` is non-empty.
    一条 BEFORE DELETE 的 RAISE 触发器；`gate` 非空时带清理门 WHEN 子句。
    """
    return f"CREATE TRIGGER {name} BEFORE DELETE ON {table} {gate}BEGIN SELECT RAISE(ABORT, '{message}'); END"


def _history_ddl(version: int) -> list[str]:
    """CREATE TABLE history (+ index + append-only triggers) with the event vocabulary of `version`.
    按 `version` 的事件词表生成 CREATE TABLE history（及索引与只追加触发器）。
    """
    events = HISTORY_EVENTS_V1 if version < 2 else HistoryEvent.values()
    return [
        "CREATE TABLE history (\n"
        "    id           TEXT PRIMARY KEY,\n"
        "    memory_id    TEXT NOT NULL,\n"
        "    old_memory   TEXT,\n"
        "    new_memory   TEXT,\n"
        f"    event        TEXT NOT NULL {_in('event', events)},\n"
        "    created_at   TEXT NOT NULL,\n"
        "    updated_at   TEXT NOT NULL,\n"
        "    is_deleted   INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0,1)),\n"
        "    actor_id     TEXT,\n"
        "    role         TEXT,\n"
        "    anchor_id    TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    atom_id      TEXT,\n"
        "    edge_id      TEXT,\n"
        "    successor_id TEXT,\n"
        "    details_json TEXT NOT NULL DEFAULT '{}',\n"
        "    seq          INTEGER NOT NULL UNIQUE\n"
        ")",
        "CREATE INDEX ix_history_memory ON history(memory_id, seq)",
        "CREATE TRIGGER trg_history_append_only BEFORE UPDATE ON history BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT: history is append-only'); END",
        "CREATE TRIGGER trg_history_no_delete BEFORE DELETE ON history BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT: history is append-only'); END",
    ]


def build_ddl(version: int = SCHEMA_VERSION) -> list[str]:
    """Generate the §7 DDL statements of one schema version in order (one statement per entry, never a script).
    按顺序生成某模式版本的 §7 DDL 语句（每项一条语句，绝非脚本）。

    Input / 输入：
        version: 1 (ungated triggers, history without PURGE), 2 (the retention gate) or 3 (adds atom_fts, the current schema); defaults to SCHEMA_VERSION.
        version：1（无门触发器、history 无 PURGE）、2（保留策略清理门）或 3（新增 atom_fts，当前模式）；默认 SCHEMA_VERSION。

    Output / 输出：
        List of CREATE TABLE / INDEX / TRIGGER statements; version 1 is exactly the schema the 0.1 release created.
        CREATE TABLE / INDEX / TRIGGER 语句列表；版本 1 与 0.1 发布版创建的模式完全一致。

    State effect / 状态影响：
        None.
        无。
    """
    gate = PURGE_GATE if version >= 2 else ""
    gated = {name: (table, message) for name, table, message in PURGE_GATED_TRIGGERS}
    stmts: list[str] = []
    # meta comes first: the purge gate of the BEFORE DELETE triggers below reads it (SQLite resolves trigger bodies at
    # run time, but creating the table first keeps the generated text self-explanatory).
    # meta 放在最前：下面 BEFORE DELETE 触发器的清理门要读它（SQLite 在运行时才解析触发器体，
    # 但先建这张表能让生成的文本一目了然）。
    stmts.append("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    # ===================== evidence plane =====================
    # ===================== 证据平面 =====================
    stmts.append(
        "CREATE TABLE source_events (\n"
        "    event_id        TEXT PRIMARY KEY,\n"
        "    source_system   TEXT NOT NULL,\n"
        f"    object_type     TEXT NOT NULL {_in('object_type', ObjectType.values())},\n"
        f"    event_type      TEXT NOT NULL {_in('event_type', EventType.values())},\n"
        "    actor_id        TEXT,\n"
        f"    role            TEXT {_in_or_null('role', EVENT_ROLE_VALUES)},\n"
        "    timestamp       TEXT NOT NULL,\n"
        "    ingested_at     TEXT NOT NULL,\n"
        "    raw_ref         TEXT,\n"
        f"    visibility      TEXT NOT NULL {_vis('visibility')},\n"
        f"    {SCOPE},\n"
        "    scope_key       TEXT NOT NULL,\n"
        "    payload_json    TEXT NOT NULL,\n"
        "    payload_hash    TEXT NOT NULL,\n"
        "    user_meta_json  TEXT NOT NULL DEFAULT '{}',\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_events_scope_time ON source_events(scope_key, timestamp)")
    stmts.append("CREATE INDEX ix_events_type_time  ON source_events(object_type, timestamp)")
    stmts.append("CREATE UNIQUE INDEX ux_events_identity ON source_events(payload_hash, timestamp, scope_key)")
    stmts.append(_no_delete_trigger("trg_events_no_delete", *gated["trg_events_no_delete"], gate=gate))
    stmts.append(
        "CREATE TRIGGER trg_events_immutable BEFORE UPDATE ON source_events BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: source_events are immutable'); END"
    )
    stmts.append(
        "CREATE TABLE anchors (\n"
        "    anchor_id        TEXT PRIMARY KEY,\n"
        "    source_event_id  TEXT NOT NULL UNIQUE REFERENCES source_events(event_id),\n"
        "    hash             TEXT NOT NULL,\n"
        "    source_ref       TEXT,\n"
        "    timestamp        TEXT NOT NULL,\n"
        "    ingested_at      TEXT NOT NULL,\n"
        f"    permission_scope TEXT NOT NULL {_vis('permission_scope')},\n"
        f"    {SCOPE},\n"
        "    revoked_at       TEXT,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_anchors_hash ON anchors(hash)")
    stmts.append("CREATE INDEX ix_anchors_scope_time ON anchors(user_id, agent_id, run_id, timestamp)")
    stmts.append(_no_delete_trigger("trg_anchors_no_delete", *gated["trg_anchors_no_delete"], gate=gate))
    stmts.append(
        "CREATE TRIGGER trg_anchors_immutable BEFORE UPDATE OF anchor_id, source_event_id, hash, source_ref, "
        "timestamp, ingested_at, user_id, agent_id, run_id ON anchors\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: anchors are immutable "
        "(only permission_scope/revoked_at may change)'); END"
    )
    # ===================== observation / state layers =====================
    # ===================== 观察 / 状态层 =====================
    stmts.append(
        "CREATE TABLE observations (\n"
        "    obs_id          TEXT PRIMARY KEY,\n"
        "    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    anchor_ids_json TEXT NOT NULL DEFAULT '[]',\n"
        f"    type            TEXT NOT NULL {_in('type', ObservationType.values())},\n"
        "    subject         TEXT NOT NULL,\n"
        "    predicate       TEXT NOT NULL,\n"
        "    objects_json    TEXT NOT NULL DEFAULT '[]',\n"
        f"    relation        TEXT NOT NULL {_in('relation', Relation.values())},\n"
        "    text            TEXT NOT NULL CHECK (length(text) > 0),\n"
        "    date_observed   TEXT NOT NULL,\n"
        "    valid_from      TEXT, valid_to TEXT,\n"
        "    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),\n"
        f"    visibility      TEXT NOT NULL {_vis('visibility')},\n"
        f"    attributed_to   TEXT NOT NULL {_in('attributed_to', ATTRIBUTED_TO_VALUES)},\n"
        f"    sentiment       TEXT NOT NULL {_in('sentiment', Sentiment.values())},\n"
        "    topic_tags_json TEXT NOT NULL DEFAULT '[]',\n"
        "    entity_mentions_json TEXT NOT NULL DEFAULT '[]',\n"
        "    supersedes_hint_ids_json TEXT NOT NULL DEFAULT '[]',\n"
        "    extracted_by    TEXT NOT NULL,\n"
        "    atom_id         TEXT,\n"
        f"    {SCOPE},\n"
        "    created_at      TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_obs_anchor ON observations(anchor_id)")
    stmts.append("CREATE INDEX ix_obs_subject_pred ON observations(user_id, agent_id, run_id, subject, predicate)")
    stmts.append(_no_delete_trigger("trg_obs_no_delete", *gated["trg_obs_no_delete"], gate=gate))
    stmts.append(
        "CREATE TRIGGER trg_obs_immutable BEFORE UPDATE OF obs_id, anchor_id, type, subject, predicate, objects_json, "
        "text, date_observed, created_at ON observations\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: observation content is immutable'); END"
    )
    stmts.append(
        "CREATE TABLE eps (\n"
        "    ep_id           TEXT PRIMARY KEY,\n"
        "    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    anchor_ids_json TEXT NOT NULL,\n"
        "    scale           TEXT NOT NULL CHECK (scale = 'L1_micro'),\n"
        "    u TEXT NOT NULL CHECK (length(u) > 0), b TEXT NOT NULL DEFAULT '', "
        "a TEXT NOT NULL CHECK (length(a) > 0), y TEXT NOT NULL DEFAULT '',\n"
        f"    r               TEXT NOT NULL {_in('r', Reward.values())},\n"
        "    kappa_json      TEXT NOT NULL,\n"
        "    time_anchor     TEXT NOT NULL,\n"
        "    valid_from      TEXT NOT NULL, valid_to TEXT,\n"
        "    entity_mentions_json TEXT NOT NULL DEFAULT '[]',\n"
        "    topic_tags_json TEXT NOT NULL DEFAULT '[]',\n"
        "    uses_ep_ids_json TEXT NOT NULL DEFAULT '[]',\n"
        "    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),\n"
        f"    visibility      TEXT NOT NULL {_vis('visibility')},\n"
        "    extracted_by    TEXT NOT NULL,\n"
        "    atom_id         TEXT,\n"
        f"    {SCOPE},\n"
        "    created_at      TEXT NOT NULL,\n"
        "    CHECK (valid_to IS NULL OR valid_to >= valid_from),\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_eps_scope_time ON eps(user_id, agent_id, run_id, time_anchor)")
    stmts.append(_no_delete_trigger("trg_eps_no_delete", *gated["trg_eps_no_delete"], gate=gate))
    stmts.append(
        "CREATE TRIGGER trg_eps_immutable BEFORE UPDATE OF ep_id, anchor_id, anchor_ids_json, scale, u, b, a, y, r, "
        "kappa_json, time_anchor, valid_from, created_at ON eps\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: ep slots are immutable'); END"
    )
    # ===================== State Index plane (MVMI) =====================
    # ===================== State Index 平面（MVMI） =====================
    stmts.append(
        "CREATE TABLE memory_atoms (\n"
        "    atom_id           TEXT PRIMARY KEY,\n"
        "    memory_id         TEXT NOT NULL,\n"
        "    version           INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),\n"
        f"    atom_type         TEXT NOT NULL {_in('atom_type', AtomType.values())},\n"
        "    content_json      TEXT NOT NULL,\n"
        "    text              TEXT NOT NULL CHECK (length(text) > 0),\n"
        "    hash              TEXT NOT NULL,\n"
        "    time_anchor       TEXT,\n"
        "    valid_from        TEXT NOT NULL,\n"
        "    valid_to          TEXT,\n"
        f"    scale_level       TEXT NOT NULL {_in('scale_level', Scale.values())},\n"
        f"    role              TEXT NOT NULL {_in('role', Role.values())},\n"
        f"    sentiment         TEXT {_in_or_null('sentiment', Sentiment.values())},\n"
        "    subject           TEXT,\n"
        "    predicate         TEXT,\n"
        f"    obs_type          TEXT {_in_or_null('obs_type', ObservationType.values())},\n"
        f"    source_kind       TEXT NOT NULL {_in('source_kind', SourceKind.values())},\n"
        "    source_id         TEXT NOT NULL,\n"
        "    source_ep_ids_json TEXT NOT NULL DEFAULT '[]',\n"
        "    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),\n"
        "    extracted_by      TEXT NOT NULL,\n"
        "    primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        f"    visibility        TEXT NOT NULL {_vis('visibility')},\n"
        f"    {SCOPE},\n"
        "    actor_id          TEXT,\n"
        f"    attributed_to     TEXT {_in_or_null('attributed_to', ATTRIBUTED_TO_VALUES)},\n"
        "    memory_type       TEXT,\n"
        "    user_meta_json    TEXT NOT NULL DEFAULT '{}',\n"
        "    archived_reason   TEXT,\n"
        "    created_at        TEXT NOT NULL,\n"
        "    updated_at        TEXT NOT NULL,\n"
        "    CHECK (valid_to IS NULL OR valid_to >= valid_from),\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE UNIQUE INDEX ux_atoms_lineage_version ON memory_atoms(memory_id, version)")
    stmts.append("CREATE UNIQUE INDEX ux_atoms_head ON memory_atoms(memory_id) WHERE archived_reason IS NULL")
    stmts.append("CREATE INDEX ix_atoms_time      ON memory_atoms(time_anchor)")
    stmts.append("CREATE INDEX ix_atoms_validity  ON memory_atoms(valid_from, valid_to)")
    stmts.append("CREATE INDEX ix_atoms_live      ON memory_atoms(valid_from) WHERE valid_to IS NULL")
    stmts.append("CREATE INDEX ix_atoms_scale     ON memory_atoms(scale_level)")
    stmts.append("CREATE INDEX ix_atoms_role      ON memory_atoms(role)")
    stmts.append("CREATE INDEX ix_atoms_scope     ON memory_atoms(user_id, agent_id, run_id)")
    stmts.append("CREATE INDEX ix_atoms_subject   ON memory_atoms(subject, predicate)")
    stmts.append("CREATE INDEX ix_atoms_hash      ON memory_atoms(hash)")
    stmts.append("CREATE INDEX ix_atoms_source    ON memory_atoms(source_kind, source_id)")
    stmts.append("CREATE INDEX ix_atoms_obs_type  ON memory_atoms(obs_type)")
    stmts.append(_no_delete_trigger("trg_atoms_no_delete", *gated["trg_atoms_no_delete"], gate=gate))
    stmts.append(
        "CREATE TRIGGER trg_atoms_immutable BEFORE UPDATE OF atom_id, memory_id, version, atom_type, content_json, "
        "text, hash, time_anchor, valid_from, scale_level, role, subject, predicate, obs_type, source_kind, source_id, "
        "primary_anchor_id, created_at, user_id, agent_id, run_id ON memory_atoms\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: atom content is immutable; "
        "create a successor version + revision edge'); END"
    )
    stmts.append(
        "CREATE TABLE atom_anchor (\n"
        "    atom_id     TEXT NOT NULL REFERENCES memory_atoms(atom_id),\n"
        "    anchor_id   TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        f"    role        TEXT NOT NULL DEFAULT 'primary' {_in('role', ATOM_ANCHOR_ROLE_VALUES)},\n"
        "    disputed_at TEXT,\n"
        "    PRIMARY KEY (atom_id, anchor_id)\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_atom_anchor_anchor ON atom_anchor(anchor_id)")
    stmts.append(
        "CREATE TRIGGER trg_atoms_primary_anchor AFTER INSERT ON memory_atoms\n"
        "BEGIN INSERT OR IGNORE INTO atom_anchor(atom_id, anchor_id, role) "
        "VALUES (NEW.atom_id, NEW.primary_anchor_id, 'primary'); END"
    )
    stmts.append(_no_delete_trigger("trg_atom_anchor_no_delete", *gated["trg_atom_anchor_no_delete"], gate=gate))
    stmts.append(
        "CREATE TABLE entities (\n"
        "    entity_id       TEXT PRIMARY KEY,\n"
        "    normalized_text TEXT NOT NULL,\n"
        "    canonical_text  TEXT NOT NULL,\n"
        "    entity_type     TEXT NOT NULL CHECK (entity_type IN ('PROPER','IDENTIFIER','FILE','QUOTED','TOPIC')),\n"
        "    aliases_json    TEXT NOT NULL DEFAULT '[]',\n"
        "    user_id TEXT, agent_id TEXT, run_id TEXT,\n"
        "    created_at      TEXT NOT NULL\n"
        ")"
    )
    stmts.append(
        "CREATE UNIQUE INDEX ux_entities_norm_scope ON entities(normalized_text, COALESCE(user_id,''), COALESCE(agent_id,''))"
    )
    stmts.append(
        "CREATE TABLE atom_entity (atom_id TEXT NOT NULL REFERENCES memory_atoms(atom_id), "
        "entity_id TEXT NOT NULL REFERENCES entities(entity_id), PRIMARY KEY (atom_id, entity_id))"
    )
    stmts.append("CREATE INDEX ix_entity_atom ON atom_entity(entity_id)")
    stmts.append(
        "CREATE TABLE atom_topic  (atom_id TEXT NOT NULL REFERENCES memory_atoms(atom_id), "
        "topic TEXT NOT NULL, PRIMARY KEY (atom_id, topic))"
    )
    stmts.append("CREATE INDEX ix_topic_atom ON atom_topic(topic)")
    stmts.append(
        "CREATE TABLE atom_vectors (\n"
        "    atom_id   TEXT PRIMARY KEY REFERENCES memory_atoms(atom_id),\n"
        "    embedder  TEXT NOT NULL,\n"
        "    dims      INTEGER NOT NULL CHECK (dims > 0),\n"
        "    embedding BLOB NOT NULL\n"
        ")"
    )
    # ===================== Support Graph plane =====================
    # ===================== Support Graph 平面 =====================
    stmts.append(
        "CREATE TABLE graph_nodes (\n"
        "    node_id           TEXT PRIMARY KEY REFERENCES memory_atoms(atom_id),\n"
        f"    node_kind         TEXT NOT NULL {_in('node_kind', NodeKind.values())},\n"
        "    ref_id            TEXT NOT NULL,\n"
        f"    scale             TEXT NOT NULL {_in('scale', Scale.values())},\n"
        "    primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        f"    {SCOPE},\n"
        "    valid             INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),\n"
        "    invalidated_at    TEXT,\n"
        "    mentions          INTEGER NOT NULL DEFAULT 1,\n"
        "    created_at        TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_nodes_scope_kind ON graph_nodes(user_id, agent_id, run_id, node_kind, valid)")
    stmts.append("CREATE INDEX ix_nodes_ref ON graph_nodes(ref_id)")
    stmts.append(_no_delete_trigger("trg_nodes_no_delete", *gated["trg_nodes_no_delete"], gate=gate))
    stmts.append(
        "CREATE TABLE graph_edges (\n"
        "    edge_id                   TEXT PRIMARY KEY,\n"
        "    src_id                    TEXT NOT NULL REFERENCES graph_nodes(node_id),\n"
        "    dst_id                    TEXT NOT NULL REFERENCES graph_nodes(node_id),\n"
        f"    edge_type                 TEXT NOT NULL {_in('edge_type', EdgeType.values())},\n"
        "    label                     TEXT NOT NULL DEFAULT '',\n"
        f"    scale                     TEXT NOT NULL {_in('scale', EDGE_SCALE_VALUES)},\n"
        "    is_boundary               INTEGER NOT NULL DEFAULT 0 CHECK (is_boundary IN (0,1)),\n"
        "    confidence                REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),\n"
        "    mentions                  INTEGER NOT NULL DEFAULT 1,\n"
        "    primary_witness_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    witness_edge_id           TEXT REFERENCES graph_edges(edge_id),\n"
        f"    created_by                TEXT NOT NULL {_in('created_by', CREATED_BY_VALUES)},\n"
        "    proposer                  TEXT NOT NULL,\n"
        "    valid                     INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),\n"
        "    invalidated_at            TEXT,\n"
        "    invalidated_reason        TEXT,\n"
        "    resolved_at               TEXT,\n"
        "    resolved_by_anchor_id     TEXT REFERENCES anchors(anchor_id),\n"
        "    resolution_edge_id        TEXT REFERENCES graph_edges(edge_id),\n"
        f"    {SCOPE},\n"
        "    created_at                TEXT NOT NULL,\n"
        "    CHECK (src_id <> dst_id),\n"
        "    CHECK (is_boundary = 0 OR witness_edge_id IS NOT NULL),\n"
        "    CHECK (edge_type = 'conflict' OR (resolved_at IS NULL AND resolution_edge_id IS NULL "
        "AND resolved_by_anchor_id IS NULL)),\n"
        "    CHECK ((resolved_at IS NULL) = (resolved_by_anchor_id IS NULL)),\n"
        "    UNIQUE (src_id, dst_id, edge_type, scale),\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_edges_src ON graph_edges(src_id, valid)")
    stmts.append("CREATE INDEX ix_edges_dst ON graph_edges(dst_id, valid)")
    stmts.append("CREATE INDEX ix_edges_type_scope ON graph_edges(edge_type, user_id, agent_id, run_id, valid)")
    stmts.append("CREATE INDEX ix_edges_label ON graph_edges(label)")
    stmts.append(
        "CREATE TRIGGER trg_edges_no_delete BEFORE DELETE ON graph_edges BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: graph_edges'); END"
    )
    stmts.append(
        "CREATE TRIGGER trg_edges_immutable BEFORE UPDATE OF edge_id, src_id, dst_id, edge_type, scale, is_boundary, "
        "primary_witness_anchor_id, witness_edge_id, created_by, created_at ON graph_edges\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: edge identity/witness columns are immutable'); END"
    )
    stmts.append(
        "CREATE TABLE edge_witnesses (edge_id TEXT NOT NULL REFERENCES graph_edges(edge_id), "
        "anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id), PRIMARY KEY (edge_id, anchor_id))"
    )
    stmts.append("CREATE INDEX ix_witness_anchor ON edge_witnesses(anchor_id)")
    stmts.append(
        "CREATE TRIGGER trg_edge_witnesses_no_delete BEFORE DELETE ON edge_witnesses BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT evidence_preservation: edge_witnesses'); END"
    )
    stmts.append(
        "CREATE TABLE edge_projections (\n"
        "    high_edge_id TEXT NOT NULL REFERENCES graph_edges(edge_id),\n"
        "    low_edge_id  TEXT NOT NULL REFERENCES graph_edges(edge_id),\n"
        "    PRIMARY KEY (high_edge_id, low_edge_id),\n"
        "    CHECK (high_edge_id <> low_edge_id)\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_projections_low ON edge_projections(low_edge_id)")
    stmts.append(
        "CREATE TRIGGER trg_edges_after_insert AFTER INSERT ON graph_edges\n"
        "BEGIN\n"
        "    INSERT OR IGNORE INTO edge_witnesses(edge_id, anchor_id) VALUES (NEW.edge_id, NEW.primary_witness_anchor_id);\n"
        "    INSERT OR IGNORE INTO edge_projections(high_edge_id, low_edge_id) "
        "SELECT NEW.edge_id, NEW.witness_edge_id WHERE NEW.witness_edge_id IS NOT NULL;\n"
        "END"
    )
    stmts.append(
        "CREATE TABLE interface_eps (\n"
        "    interface_ep_id         TEXT PRIMARY KEY,\n"
        f"    scale                   TEXT NOT NULL {_in('scale', IEP_SCALE_VALUES)},\n"
        "    atom_id                 TEXT NOT NULL UNIQUE REFERENCES memory_atoms(atom_id),\n"
        "    slots_json              TEXT NOT NULL,\n"
        "    slot_support_json       TEXT NOT NULL,\n"
        "    support_node_ids_json   TEXT NOT NULL,\n"
        "    support_edge_ids_json   TEXT NOT NULL DEFAULT '[]',\n"
        "    boundary_edges_json     TEXT NOT NULL DEFAULT '[]',\n"
        "    invariants_json         TEXT NOT NULL DEFAULT '[]',\n"
        "    residuals_json          TEXT NOT NULL DEFAULT '[]',\n"
        "    evidence_anchor_ids_json TEXT NOT NULL,\n"
        "    primary_support_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),\n"
        "    primary_anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    coarsen_op_version      TEXT NOT NULL,\n"
        "    cohesion_score          REAL NOT NULL CHECK (cohesion_score BETWEEN 0 AND 1),\n"
        "    support_hash            TEXT NOT NULL,\n"
        f"    {SCOPE},\n"
        "    valid                   INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0,1)),\n"
        "    valid_to                TEXT,\n"
        "    created_at              TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_iep_scope ON interface_eps(user_id, agent_id, run_id, valid)")
    stmts.append("CREATE INDEX ix_iep_support_hash ON interface_eps(support_hash)")
    stmts.append(
        "CREATE TABLE interface_ep_support (\n"
        "    interface_ep_id TEXT NOT NULL REFERENCES interface_eps(interface_ep_id),\n"
        f"    member_kind     TEXT NOT NULL {_in('member_kind', MEMBER_KIND_VALUES)},\n"
        "    member_id       TEXT NOT NULL,\n"
        "    PRIMARY KEY (interface_ep_id, member_kind, member_id)\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_iep_support_member ON interface_ep_support(member_kind, member_id)")
    stmts.append(
        "CREATE TRIGGER trg_iep_after_insert AFTER INSERT ON interface_eps\n"
        "BEGIN INSERT OR IGNORE INTO interface_ep_support VALUES (NEW.interface_ep_id, 'node', NEW.primary_support_node_id); END"
    )
    stmts.append(
        "CREATE TRIGGER trg_iep_no_delete BEFORE DELETE ON interface_eps BEGIN "
        "SELECT RAISE(ABORT, 'HMS_INVARIANT revision_only: interface_eps'); END"
    )
    stmts.append(
        "CREATE TRIGGER trg_iep_immutable BEFORE UPDATE OF interface_ep_id, scale, atom_id, slots_json, "
        "slot_support_json, support_node_ids_json, support_edge_ids_json, boundary_edges_json, "
        "evidence_anchor_ids_json, primary_support_node_id, primary_anchor_id, coarsen_op_version, support_hash, "
        "created_at ON interface_eps\n"
        "BEGIN SELECT RAISE(ABORT, 'HMS_INVARIANT: interface_ep support/boundary/evidence are immutable; "
        "re-coarsen creates a successor'); END"
    )
    # ===================== ops plane =====================
    # ===================== ops（运维）平面 =====================
    stmts.append(
        "CREATE TABLE feedback_events (\n"
        "    feedback_id     TEXT PRIMARY KEY,\n"
        "    target_id       TEXT NOT NULL,\n"
        f"    target_kind     TEXT NOT NULL {_in('target_kind', TARGET_KIND_VALUES)},\n"
        f"    feedback        TEXT NOT NULL {_in('feedback', FeedbackType.values())},\n"
        "    reason          TEXT,\n"
        "    correction_json TEXT,\n"
        "    actor_id        TEXT,\n"
        "    anchor_id       TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    actions_json    TEXT NOT NULL DEFAULT '[]',\n"
        f"    {SCOPE},\n"
        "    created_at      TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_feedback_target ON feedback_events(target_id)")
    stmts.extend(_history_ddl(version))
    stmts.append(
        "CREATE TABLE checkpoints (\n"
        "    checkpoint_id TEXT PRIMARY KEY,\n"
        "    label         TEXT,\n"
        "    anchor_id     TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        f"    {SCOPE},\n"
        "    snapshot_json TEXT NOT NULL,\n"
        "    history_seq   INTEGER NOT NULL,\n"
        "    snapshot_hash TEXT NOT NULL,\n"
        "    created_at    TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append(
        "CREATE TABLE transitions (\n"
        "    transition_id         TEXT PRIMARY KEY,\n"
        f"    source_kind           TEXT NOT NULL {_in('source_kind', TransitionKind.values())},\n"
        "    source_id             TEXT NOT NULL,\n"
        f"    scale                 TEXT NOT NULL {_in('scale', EDGE_SCALE_VALUES)},\n"
        "    s_t_json              TEXT NOT NULL,\n"
        "    a_t                   TEXT NOT NULL CHECK (length(a_t) > 0),\n"
        "    s_t1_json             TEXT NOT NULL,\n"
        f"    r_t                   TEXT NOT NULL {_in('r_t', Reward.values())},\n"
        "    kappa_t_json          TEXT NOT NULL,\n"
        "    w_t_json              TEXT NOT NULL,\n"
        "    w_t_primary_anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    time_t                TEXT NOT NULL,\n"
        "    time_t1               TEXT NOT NULL,\n"
        f"    {SCOPE},\n"
        "    created_at            TEXT NOT NULL,\n"
        "    UNIQUE (source_kind, source_id),\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_transitions_scope_t ON transitions(user_id, agent_id, run_id, time_t1)")
    stmts.append(
        "CREATE TABLE compiled_states (\n"
        "    compiled_id   TEXT PRIMARY KEY,\n"
        "    scope_key     TEXT NOT NULL,\n"
        f"    {SCOPE},\n"
        "    as_of         TEXT NOT NULL,\n"
        "    mode          TEXT NOT NULL,\n"
        "    card          TEXT NOT NULL,\n"
        "    context_json  TEXT NOT NULL,\n"
        "    hash          TEXT NOT NULL,\n"
        "    used_tokens   INTEGER NOT NULL,\n"
        "    anchor_id     TEXT NOT NULL REFERENCES anchors(anchor_id),\n"
        "    created_at    TEXT NOT NULL,\n"
        f"    {SCOPE_CHECK}\n"
        ")"
    )
    stmts.append("CREATE INDEX ix_compiled_scope ON compiled_states(scope_key, created_at)")
    if version >= 3:
        stmts.append(FTS_DDL)
    return stmts


def upgrade_2() -> list[str]:
    """Migration 1 → 2: swap the seven gated BEFORE DELETE triggers and rebuild history with the PURGE event.
    迁移 1 → 2：替换七个设门的 BEFORE DELETE 触发器，并以含 PURGE 事件的词表重建 history。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Statements applied in order inside one migration transaction; the result equals build_ddl(2).
        在一个迁移事务内按顺序执行的语句；结果等于 build_ddl(2)。

    State effect / 状态影响：
        None (a generator); applied by apply_migrations, which copies every history row into the rebuilt table.
        无（生成器）；由 apply_migrations 应用，它把每一行 history 复制进重建的表。
    """
    stmts: list[str] = []
    for name, table, message in PURGE_GATED_TRIGGERS:
        stmts.append(f"DROP TRIGGER IF EXISTS {name}")
        stmts.append(_no_delete_trigger(name, table, message, gate=PURGE_GATE))
    # SQLite cannot alter a CHECK: rename the old table (its triggers and index follow it), create the new one with
    # the same text build_ddl(2) emits, copy the rows, drop the old table (which drops its triggers), re-add the rest.
    # SQLite 不能修改 CHECK：先重命名旧表（它的触发器与索引随之改名），再按 build_ddl(2) 的相同文本建新表、
    # 复制行、删除旧表（其触发器随之删除），最后补回索引与触发器。
    history = _history_ddl(2)
    columns = (
        "id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role, "
        "anchor_id, atom_id, edge_id, successor_id, details_json, seq"
    )
    stmts.append("ALTER TABLE history RENAME TO history_v1")
    stmts.append(history[0])
    stmts.append(f"INSERT INTO history({columns}) SELECT {columns} FROM history_v1 ORDER BY seq")
    stmts.append("DROP TABLE history_v1")
    stmts.extend(history[1:])
    return stmts


def upgrade_3() -> list[str]:
    """Migration 2 → 3: create the atom_fts FTS5 table (apply_migrations backfills it from memory_atoms in Python).
    迁移 2 → 3：创建 atom_fts FTS5 表（apply_migrations 用 Python 从 memory_atoms 回填）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The single CREATE VIRTUAL TABLE statement (the same text build_ddl(3) appends).
        单条 CREATE VIRTUAL TABLE 语句（与 build_ddl(3) 追加的文本相同）。

    State effect / 状态影响：
        None (a generator); the backfill (POST_MIGRATIONS[3]) runs inside the same migration transaction.
        无（生成器）；回填（POST_MIGRATIONS[3]）在同一个迁移事务内执行。
    """
    return [FTS_DDL]


def fts_tokens(text: str, topic_tags: Iterable[str] = (), entity_keys: Iterable[str] = ()) -> str:
    """The atom_fts `tokens` column: content tokens (path basenames added), topic tags, entity keys and their tokens.
    atom_fts 的 `tokens` 列：内容词元（补充路径基名）、主题标签、实体键及其词元。

    Input / 输入：
        text: The atom text.
        text：原子文本。

        topic_tags: The atom's topic tags.
        topic_tags：原子的主题标签。

        entity_keys: The atom's entity keys (entities.normalized_text).
        entity_keys：原子的实体键（entities.normalized_text）。

    Output / 输出：
        Space-joined lowercase tokens: the content tokens in order (duplicates kept for term frequency), then the
        sorted extras — the same vocabulary utils.scoring.atom_keyword_tokens builds for the in-pool keyword score.
        以空格连接的小写词元：内容词元按顺序（保留重复以体现词频），随后是排序后的附加词元——与
        utils.scoring.atom_keyword_tokens 为池内关键词分构造的词表相同。

    State effect / 状态影响：
        None.
        无。
    """
    ordered: list[str] = []
    extras: set[str] = set()
    for token in content_tokens(text):
        ordered.append(token)
        if "/" in token:
            extras.add(token.rsplit("/", 1)[-1])
    extras.update(str(t).lower() for t in topic_tags)
    for key in entity_keys:
        extras.add(str(key))
        extras.update(tokenize(str(key)))
    seen = set(ordered)
    return " ".join(ordered + sorted(t for t in extras if t and t not in seen))


def backfill_fts(conn: sqlite3.Connection) -> int:
    """Rebuild atom_fts from every non-archived memory_atoms row (topics and entity keys included).
    从每个未归档的 memory_atoms 行重建 atom_fts（含主题与实体键）。

    Input / 输入：
        conn: Connection inside the migration transaction (atom_fts exists and is empty).
        conn：处于迁移事务内的连接（atom_fts 已存在且为空）。

    Output / 输出：
        Number of rows written.
        写入的行数。

    State effect / 状态影响：
        Inserts into atom_fts.
        向 atom_fts 插入行。
    """
    topics: dict[str, list[str]] = {}
    for atom_id, topic in conn.execute("SELECT atom_id, topic FROM atom_topic ORDER BY rowid"):
        topics.setdefault(atom_id, []).append(topic)
    keys: dict[str, list[str]] = {}
    for atom_id, key in conn.execute(
        "SELECT ae.atom_id, e.normalized_text FROM atom_entity ae JOIN entities e ON e.entity_id = ae.entity_id "
        "ORDER BY e.normalized_text"
    ):
        keys.setdefault(atom_id, []).append(key)
    rows = [
        (atom_id, fts_tokens(text, topics.get(atom_id, ()), keys.get(atom_id, ())))
        for atom_id, text in conn.execute(
            "SELECT atom_id, text FROM memory_atoms WHERE archived_reason IS NULL ORDER BY atom_id"
        ).fetchall()
    ]
    conn.executemany("INSERT INTO atom_fts(atom_id, tokens) VALUES (?, ?)", rows)
    return len(rows)


# Append-only ladder: 1 = the 0.1 schema, 2 = the retention gate (§16.6), 3 = the atom_fts keyword table. The ladder
# upgrades EXISTING files; a version-0 database is created directly from build_ddl(SCHEMA_VERSION) in one transaction
# (apply_migrations). POST_MIGRATIONS holds the Python steps a version needs after its statements (same transaction).
# 只追加的阶梯：1 = 0.1 版模式，2 = 保留策略清理门（§16.6），3 = atom_fts 关键词表。阶梯只用于升级**已有**文件；
# 版本 0 的库由 apply_migrations 直接按 build_ddl(SCHEMA_VERSION) 在一个事务内创建。POST_MIGRATIONS 存放某版本在其
# 语句之后还需要的 Python 步骤（同一事务）。
MIGRATIONS: dict[int, list[str]] = {1: build_ddl(1), 2: upgrade_2(), 3: upgrade_3()}
POST_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], object]] = {3: backfill_fts}


def ddl_text() -> str:
    """Render the full generated DDL as one printable text (statements terminated by ';').
    将生成的完整 DDL 渲染为可打印文本（语句以 ';' 结束）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The DDL text in migration order.
        按迁移顺序的 DDL 文本。

    State effect / 状态影响：
        None.
        无。
    """
    return "\n".join(stmt + ";" for version in sorted(MIGRATIONS) for stmt in MIGRATIONS[version])


def current_version(conn: sqlite3.Connection) -> int:
    """Read PRAGMA user_version.
    读取 PRAGMA user_version。

    Input / 输入：
        conn: Open sqlite3 connection.
        conn：已打开的 sqlite3 连接。

    Output / 输出：
        Integer schema version (0 for a fresh database).
        整数模式版本（新库为 0）。

    State effect / 状态影响：
        None.
        无。
    """
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def apply_migrations(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    """Create a fresh database directly at SCHEMA_VERSION, or upgrade an older one step by step, under BEGIN IMMEDIATE.
    在 BEGIN IMMEDIATE 下把新库直接建到 SCHEMA_VERSION，或把旧库逐级升级；由 PRAGMA user_version 控制。

    Input / 输入：
        conn: Connection in autocommit mode (isolation_level=None) with foreign keys enabled.
        conn：自动提交模式（isolation_level=None）且已启用外键的连接。

        created_at: When given and the database is fresh (version 0), meta.schema_created_at is written inside the single creation transaction.
        created_at：给定且数据库为全新（版本 0）时，meta.schema_created_at 在唯一的建库事务内写入。

    Output / 输出：
        The schema version after migration (== SCHEMA_VERSION); a newer database raises StoreError(STORE_002).
        迁移后的模式版本（== SCHEMA_VERSION）；库版本更新时抛出 StoreError(STORE_002)。

    State effect / 状态影响：
        Creates tables/indexes/triggers and bumps user_version; a failed step is rolled back and raised as STORE_001.
        创建表/索引/触发器并提升 user_version；失败步骤回滚并以 STORE_001 抛出。
    """
    current = current_version(conn)
    if current > SCHEMA_VERSION:
        raise StoreError(
            f"database schema version {current} is newer than this library supports ({SCHEMA_VERSION})",
            "STORE_002",
            details={"db_version": current, "library_version": SCHEMA_VERSION},
            suggestion="pip install -U hms-elite",
        )
    if current == 0:
        # A fresh database is built from build_ddl(SCHEMA_VERSION) in ONE transaction: one fsync'd commit, no
        # rename/copy/drop of the ladder and no freed pages (97 pages instead of 101, 1.8 ms instead of 3.4 ms on a
        # file). The result is the same sqlite_master row set the ladder produces (pinned by tests).
        # 新库按 build_ddl(SCHEMA_VERSION) 在**一个**事务内建成：只有一次带 fsync 的提交，没有阶梯的重命名/复制/删除，
        # 也没有释放页（文件上 97 页而非 101 页，1.8 ms 而非 3.4 ms）。sqlite_master 的行集合与阶梯结果相同（有测试钉住）。
        steps: list[tuple[int, list[str], bool]] = [(SCHEMA_VERSION, build_ddl(SCHEMA_VERSION), True)]
    else:
        # An existing file walks the append-only ladder from its version to SCHEMA_VERSION, one commit per step.
        # 已有文件从其版本沿只追加的阶梯升到 SCHEMA_VERSION，每级一次提交。
        steps = [(version, MIGRATIONS[version], False) for version in range(current + 1, SCHEMA_VERSION + 1)]
    fts5 = fts5_available()
    if not fts5:
        logger.warning("SQLite %s has no FTS5 module: atom_fts is skipped, keyword recall uses LIKE", sqlite3.sqlite_version)
    for version, statements, fresh in steps:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise StoreError(f"cannot begin migration {version}: {exc}", "STORE_001") from exc
        try:
            for stmt in statements:
                if not fts5 and stmt == FTS_DDL:
                    # Without the module the virtual table cannot exist; capabilities()['fts5'] reports it.
                    # 没有该模块虚拟表无法存在；capabilities()['fts5'] 会如实报告。
                    continue
                # 绝不使用 executescript：每个条目只执行一条语句
                conn.execute(stmt)  # never executescript: one statement per entry
            post = POST_MIGRATIONS.get(version)
            if post is not None and fts5 and not fresh:
                # Python-side steps (the FTS backfill) ride in the same transaction; a fresh database has no rows.
                # Python 侧步骤（FTS 回填）搭乘同一事务；全新库没有行可回填。
                post(conn)
            if fresh and created_at is not None:
                # The creation stamp rides in the creation transaction: one commit instead of two on a fresh file.
                # 创建时间戳搭乘建库事务：新文件库只提交一次而非两次。
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_created_at', ?)", (created_at,))
            conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - nothing left to roll back
                pass
            raise StoreError(
                f"migration {version} failed: {exc}", "STORE_001", details={"version": version, "error": str(exc)}
            ) from exc
    return SCHEMA_VERSION


# Process-wide schema-only template for fresh databases (private ':memory:' and new files): (pid, connection). Built
# once per process by apply_migrations on ':memory:', never written afterwards, never handed out; cloned page-for-page
# by clone_schema through the sqlite3 backup API.
# 进程级、仅含模式的模板，供全新数据库（私有 ':memory:' 与新文件）使用：(pid, 连接)。每个进程由 apply_migrations
# 在 ':memory:' 上构建一次，此后从不写入、从不外泄；由 clone_schema 经 sqlite3 backup API 逐页克隆。
_TEMPLATE: tuple[int, bool, sqlite3.Connection] | None = None
_TEMPLATE_LOCK = threading.Lock()


def _schema_template() -> sqlite3.Connection:
    """The per-process schema template connection, keyed on (pid, FTS5 availability); caller holds _TEMPLATE_LOCK.
    每进程的模式模板连接，以 (pid, FTS5 是否可用) 为键；调用方持有 _TEMPLATE_LOCK。
    """
    global _TEMPLATE
    key = (os.getpid(), fts5_available())
    if _TEMPLATE is None or _TEMPLATE[:2] != key:
        template = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        template.execute("PRAGMA foreign_keys = ON")
        apply_migrations(template)
        _TEMPLATE = (*key, template)
    return _TEMPLATE[2]


def clone_schema(conn: sqlite3.Connection, *, created_at: str | None = None) -> int:
    """Fresh database (private ':memory:' or a new file): copy the schema template page-for-page instead of re-running the migration ladder.
    全新数据库（私有 ':memory:' 或新文件）：逐页复制模式模板，而非重新执行迁移阶梯。

    Input / 输入：
        conn: A connection in autocommit mode with no open transaction; anything that is not fresh (user_version != 0 or any sqlite_master row) takes the apply_migrations path unchanged.
        conn：自动提交模式、无未结事务的连接；凡不是全新的（user_version != 0 或 sqlite_master 有任何行）都原样走 apply_migrations 路径。

        created_at: Written to meta.schema_created_at right after the copy (same value apply_migrations would write).
        created_at：复制完成后立即写入 meta.schema_created_at（与 apply_migrations 写入的值相同）。

    Output / 输出：
        SCHEMA_VERSION; the result is byte-for-byte the schema apply_migrations builds (same sqlite_master rows, root pages, user_version, page_count).
        SCHEMA_VERSION；结果与 apply_migrations 构建的模式逐字节相同（相同的 sqlite_master 行、根页、user_version、page_count）。

    State effect / 状态影响：
        Overwrites every page of a fresh `conn` with the template (a file database receives them before its WAL switch); a backup failure falls back to apply_migrations.
        用模板覆盖全新 `conn` 的每一页（文件库在切换 WAL 之前收到这些页）；备份失败时回退到 apply_migrations。
    """
    if current_version(conn) != 0 or conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] != 0:
        return apply_migrations(conn, created_at=created_at)
    try:
        with _TEMPLATE_LOCK:
            _schema_template().backup(conn, pages=-1)
    except sqlite3.Error:  # pragma: no cover - backup unsupported or interrupted: the slow path is always correct
        return apply_migrations(conn, created_at=created_at)
    if created_at is not None:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_created_at', ?)", (created_at,))
    return SCHEMA_VERSION


# json1 / FTS5 availability is a property of the linked SQLite library, not of a connection: probe once per process.
# json1 / FTS5 是否可用取决于链接的 SQLite 库而非某个连接：每进程只探测一次。
_JSON1: dict[str, bool] = {}
_FTS5: dict[str, bool] = {}


def fts5_available() -> bool:
    """Whether the linked SQLite can create FTS5 virtual tables (probed once per process on a scratch connection).
    所链接的 SQLite 能否创建 FTS5 虚拟表（每进程在一个临时连接上探测一次）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        Caches the probe result per sqlite_version.
        按 sqlite_version 缓存探测结果。
    """
    cached = _FTS5.get(sqlite3.sqlite_version)
    if cached is None:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(x)")
            cached = True
        except sqlite3.Error:
            cached = False
        finally:
            probe.close()
        _FTS5[sqlite3.sqlite_version] = cached
    return cached


def detect_capabilities(conn: sqlite3.Connection) -> dict:
    """Probe the connection for the features the planner and store adapt to.
    探测连接所具备的、规划器与存储需适配的特性。

    Input / 输入：
        conn: Open sqlite3 connection.
        conn：已打开的 sqlite3 连接。

    Output / 输出：
        {'sqlite_version': str, 'json1': bool, 'fts5': bool (the module exists AND atom_fts is present), 'wal': bool, 'schema_version': int}.
        {'sqlite_version': str, 'json1': bool, 'fts5': bool（模块存在且 atom_fts 已建）, 'wal': bool, 'schema_version': int}。

    State effect / 状态影响：
        None (read-only probes).
        无（只读探测）。
    """
    json1 = _JSON1.get(sqlite3.sqlite_version)
    if json1 is None:
        try:
            row = conn.execute("SELECT json_extract('{\"a\":1}','$.a')").fetchone()
            json1 = bool(row and row[0] == 1)
        except sqlite3.Error:
            json1 = False
        _JSON1[sqlite3.sqlite_version] = json1
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        wal = bool(mode and str(mode[0]).lower() == "wal")
    except sqlite3.Error:  # pragma: no cover - defensive
        wal = False
    fts5 = fts5_available()
    if fts5:
        try:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", VIRTUAL_TABLES).fetchone()
            fts5 = row is not None
        except sqlite3.Error:  # pragma: no cover - defensive
            fts5 = False
    return {
        "sqlite_version": sqlite3.sqlite_version,
        "json1": json1,
        "fts5": fts5,
        "wal": wal,
        "schema_version": current_version(conn),
    }


def drop_all(conn: sqlite3.Connection) -> None:
    """Drop every HMS table in reverse creation order (the only DELETE-like path; used by StateStore.reset).
    按创建逆序删除全部 HMS 表（唯一类似 DELETE 的路径；供 StateStore.reset 使用）。

    Input / 输入：
        conn: Open sqlite3 connection outside any transaction.
        conn：处于事务之外的已打开 sqlite3 连接。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Removes the virtual table (its shadow tables follow), then all base tables (triggers and indexes go with them), and resets user_version to 0.
        先删除虚拟表（影子表随之删除），再删除全部基表（触发器与索引随之删除），并将 user_version 重置为 0。
    """
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in reversed(VIRTUAL_TABLES):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for table in reversed(TABLES):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("PRAGMA user_version = 0")
    except sqlite3.Error as exc:
        raise StoreError(f"drop_all failed: {exc}", "STORE_001") from exc
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


__all__ = [
    "ATOM_ANCHOR_ROLE_VALUES",
    "ATTRIBUTED_TO_VALUES",
    "CREATED_BY_VALUES",
    "EDGE_SCALE_VALUES",
    "EVENT_ROLE_VALUES",
    "FTS_DDL",
    "FTS_SHADOW_TABLES",
    "OBS_TYPE_INDEX",
    "SCALE_INDEX",
    "TIME_INDEX",
    "HISTORY_EVENTS_V1",
    "IEP_SCALE_VALUES",
    "MEMBER_KIND_VALUES",
    "MIGRATIONS",
    "POST_MIGRATIONS",
    "PURGE_FLAG_KEY",
    "PURGE_GATE",
    "PURGE_GATED_TRIGGERS",
    "SCHEMA_VERSION",
    "SCOPE",
    "SCOPED_TABLES",
    "TABLES",
    "TARGET_KIND_VALUES",
    "VIRTUAL_TABLES",
    "apply_migrations",
    "backfill_fts",
    "build_ddl",
    "clone_schema",
    "current_version",
    "ddl_text",
    "detect_capabilities",
    "drop_all",
    "fts5_available",
    "fts_tokens",
    "upgrade_2",
    "upgrade_3",
]
