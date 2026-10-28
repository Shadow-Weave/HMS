"""Axis execution: gated candidate selection, ranking, aggregation and projection over the State Index.
轴执行：在状态索引上进行门控候选选择、排序、聚合与投影。
"""

# Purpose (EN): Execute AxisQuery semantics (HMS §3.5) — bounded multi-channel recall into a pool, then full
# scoring inside the pool — and expose Mem0-shaped search / get_all over them (RETRIEVAL_DIAGNOSIS §3–§5).
# 目的（中）：执行 AxisQuery 语义（HMS §3.5）——有界多通道召回入池、池内全量打分——并在其上暴露 Mem0 形状的
# search / get_all（RETRIEVAL_DIAGNOSIS §3–§5）。

# Input (EN): StateStore, an IndexBase provider, an EmbedderBase, AxisQuery models, ScoringConfig, RetrievalConfig, capabilities.
# 输入（中）：StateStore、IndexBase 提供者、EmbedderBase、AxisQuery 模型、ScoringConfig、RetrievalConfig、能力字典。

# Output (EN): AxisResult models and Mem0 rows (MemoryAtom.to_public_dict).
# 输出（中）：AxisResult 模型与 Mem0 行（MemoryAtom.to_public_dict）。

# State effect (EN): Read-only; touches only memory_atoms, atom_entity, entities, atom_topic, atom_anchor, atom_vectors.
# 状态影响（中）：只读；仅触及 memory_atoms、atom_entity、entities、atom_topic、atom_anchor、atom_vectors。

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hms_elite.configs.base import RetrievalConfig, ScoringConfig
from hms_elite.enums import Route
from hms_elite.index.base import IndexFilter
from hms_elite.memory.schema import OBS_TYPE_INDEX, SCALE_INDEX, TIME_INDEX
from hms_elite.memory.storage import LIVE_SQL, AtomLite, StateStore, _placeholders, _scope_where
from hms_elite.models import AtomHit, AxisQuery, AxisResult, MemoryAtom, Scope
from hms_elite.mvmi.filters import And, FilterExpr, evaluate, lower_to_sql, parse_filters, plan_filters
from hms_elite.mvmi.planner import ALL_SCALES, plan_query
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.scoring import (
    Signals,
    compute_signals,
    fuse,
    fusion_weights,
    prepare_query,
    score_details,
    semantic_scores,
)
from hms_elite.utils.text import content_tokens, extract_entities, identifier_tokens, ordered_union

logger = logging.getLogger(__name__)

# Shipped fusion defaults for the text-less paths (count / get_all never fuse); built once instead of per call.
# 无文本路径（count / get_all 从不融合）使用的出厂融合默认值；只构造一次而不是每次调用都构造。
_DEFAULT_SCORING = ScoringConfig()
# Shipped recall bounds used when the caller passes no RetrievalConfig (the facade passes config.retrieval).
# 调用方未传 RetrievalConfig 时使用的出厂召回上限（门面会传入 config.retrieval）。
_DEFAULT_RETRIEVAL = RetrievalConfig()

AXIS_ORDER = "time_anchor DESC, created_at DESC, atom_id ASC"
# Recall channel names in pool-merge precedence; 'gate' is the whole gated set entering the pool (§13.3 step 3).
# 召回通道名，按入池合并的优先级排列；'gate' 表示门控集合整体入池（§13.3 第 3 步）。
CHANNELS: tuple[str, ...] = ("semantic", "keyword", "entity", "recency")
GATE_CHANNEL = "gate"
_GATE_AXIS = {"time": "time", "entity": "entity", "type": "type"}
# primary_axis labels of the two relaxation levels: the route's scale default dropped, then the plan gates dropped.
# 两级放宽对应的 primary_axis 标签：先放弃路由的默认尺度，再放弃规划门控。
SCALE_FALLBACK = "scale_fallback"
FALLBACK = "fallback"


def _time_key(atom: Any) -> tuple[str, str, str]:
    """Sort key reproducing the axis order (time_anchor DESC nulls last, created_at DESC, atom_id ASC) when reversed.
    还原轴序（time_anchor 降序、空值最后，created_at 降序，atom_id 升序）的排序键。
    """
    return (atom.time_anchor or "", atom.created_at or "", atom.atom_id)


def _axis_sorted(atoms: list[Any]) -> list[Any]:
    """Atoms (MemoryAtom or AtomLite) in axis order: time_anchor DESC (None last), created_at DESC, atom_id ASC.
    按轴序排列的原子（MemoryAtom 或 AtomLite）：time_anchor 降序（None 最后）、created_at 降序、atom_id 升序。
    """
    ordered = sorted(atoms, key=lambda a: a.atom_id)
    ordered.sort(key=lambda a: (a.time_anchor or "", a.created_at or ""), reverse=True)
    return ordered


def _hint_key(hint: Any) -> str | None:
    """Entity key of a plan hint (dict or EntityCandidate).
    规划线索（字典或 EntityCandidate）的实体键。
    """
    value = hint.get("key") if isinstance(hint, dict) else getattr(hint, "key", None)
    return str(value) if value else None


def _plan_gates(
    plan: dict[str, Any], scope: Scope, capabilities: dict, *, entity_form: str = "in"
) -> tuple[list[str], list[Any], list[str]]:
    """Lower the plan's hard gates to SQL: (clauses, params, gate names in precedence order).
    将规划的硬门控降级为 SQL：（子句、参数、按优先级排列的门控名）。

    `entity_form` picks the entity gate's SQL: 'in' (StateStore.entity_gate_sql, materialized once: scans), 'exists'
    (correlated: id-driven reads of a few hundred rows) or 'none' (the caller drives the read through atom_entity
    itself; the gate name is still reported).
    `entity_form` 选择实体门的 SQL：'in'（StateStore.entity_gate_sql，物化一次：扫描）、'exists'（相关形式：按 id
    驱动的几百行读取）或 'none'（调用方自己经 atom_entity 驱动读取；门控名仍被报告）。
    """
    clauses: list[str] = []
    params: list[Any] = []
    names: list[str] = []
    time_gate = plan.get("hard_time_gate")
    if time_gate:
        # The callers add index_hint=TIME_INDEX for this level: without statistics SQLite would walk the scope index.
        # 调用方为该级别加上 index_hint=TIME_INDEX：没有统计时 SQLite 会改走 scope 索引。
        gte, lte = time_gate[0], time_gate[1]
        clauses.append("memory_atoms.time_anchor BETWEEN ? AND ?")
        params.extend([to_utc_iso(gte), to_utc_iso(lte)])
        names.append("time")
    entity_gate = plan.get("hard_entity_gate")
    if entity_gate:
        if entity_form != "none":
            clause, gate_params = StateStore.entity_gate_sql(
                scope, [str(k) for k in entity_gate], correlated=entity_form == "exists"
            )
            clauses.append(clause)
            params.extend(gate_params)
        names.append("entity")
    type_gate = plan.get("hard_type_gate")
    if type_gate:
        column, values = type_gate[0], [str(v) for v in type_gate[1]]
        if values:
            clauses.append(f"memory_atoms.{column} IN ({_placeholders(len(values))})")
            params.extend(values)
            names.append("type")
    return clauses, params, names


def _fallback_ladder(*, gated: bool, scale_relaxable: bool, primary_axis: str) -> list[tuple[bool, bool, str]]:
    """The retry ladder as (with_gates, with_scale, primary_axis) levels, strictest first.
    回退阶梯：按 (是否带门控, 是否带尺度, primary_axis) 排列的各级，从最严格开始。
    """
    ladder: list[tuple[bool, bool, str]] = [(True, True, primary_axis)]
    # Level 2 keeps the plan gates (cues found in the query text) and drops only the route's scale default:
    # a raw L1_micro preference event is a better answer to a preference query than an unrelated state atom.
    # 第二级保留规划门控（来自查询文本的线索），只放弃路由的默认尺度：
    # 对偏好查询而言，原始的 L1_micro 偏好事件比不相干的 state 原子更像答案。
    if scale_relaxable:
        ladder.append((True, False, SCALE_FALLBACK))
    # Level 3 is the historical fallback (gates dropped, scale kept); level 4 drops both.
    # 第三级是原有的回退（放弃门控、保留尺度）；第四级两者都放弃。
    if gated:
        ladder.append((False, True, FALLBACK))
        if scale_relaxable:
            ladder.append((False, False, FALLBACK))
    return ladder


def _count_rows(
    store: StateStore,
    scope: Scope,
    *,
    where_sql: str,
    params: list[Any],
    as_of: str,
    include_expired: bool,
    scale_constraint: list[str] | None,
    atom_types: list[str] | None,
    index_hint: str | None = None,
) -> int:
    """One SELECT COUNT(*) over the gated WHERE (store.count_axis when the store provides it).
    对门控后的 WHERE 执行一次 SELECT COUNT(*)（存储提供 count_axis 时优先使用）。
    """
    counter = getattr(store, "count_axis", None)
    if callable(counter):
        return int(
            counter(
                scope,
                where_sql=where_sql,
                params=params,
                as_of=as_of,
                include_expired=include_expired,
                scale_constraint=scale_constraint,
                atom_types=atom_types,
                index_hint=index_hint,
            )
        )
    where, values = _scope_where(scope, "memory_atoms.")
    clauses = [where]
    if not include_expired:
        instant = to_utc_iso(as_of)
        clauses.append(LIVE_SQL)
        values.extend([instant, instant])
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
    row = store._execute(f"SELECT COUNT(*) AS n FROM memory_atoms WHERE {' AND '.join(clauses)}", values).fetchone()
    return int(row["n"]) if row is not None else 0


@dataclass
class Pool:
    """The recall pool of one ladder level: ids in merge order, their channels of origin and the channel sizes.
    一个阶梯级别的召回池：按合并顺序的 id、各自的来源通道以及各通道大小。
    """

    ids: list[str] = field(default_factory=list)
    recalled_by: dict[str, list[str]] = field(default_factory=dict)
    channel_sizes: dict[str, int] = field(default_factory=dict)
    # |G| of the level (an exact SQL / id count before any Python post-filter) and whether G entered the pool whole;
    # -1 under native pushdown, where G is never enumerated (a filtered count is a full scan on Milvus) and the probe
    # only proved |G| > pool_max.
    # 该级别的 |G|（Python 后过滤之前的精确 SQL / id 计数）以及 G 是否整体入池；原生下推时为 -1——此时 G 从不被
    # 枚举（带过滤的计数在 Milvus 上是全量扫描），探测只证明了 |G| > pool_max。
    gate_count: int = 0
    whole: bool = False
    # Provider scores of the semantic channel's hits (bit-identical to an in-pool index.search over the same ids):
    # the in-pool semantic score reuses them and only scores the pool ids the channel did not return.
    # 语义通道命中的提供者分数（与对同一批 id 的池内 index.search 逐位相同）：池内语义分复用它们，
    # 只对通道未返回的池内 id 再打分。
    semantic: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of pooled ids.
        池内 id 数。

        Input / 输入：
            none.
            无。

        Output / 输出：
            len(ids).
            len(ids)。

        State effect / 状态影响：
            None.
            无。
        """
        return len(self.ids)

    def stats(self) -> dict[str, Any]:
        """The retrieval provenance of the pool (pool_size, channel_sizes, channels_used, gate_count, whole).
        池的检索溯源（pool_size、channel_sizes、channels_used、gate_count、whole）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'pool_size', 'channel_sizes', 'channels_used', 'gate_count', 'whole'}; channels_used lists the channels
            that recalled >= 0 ids, 'gate' first; gate_count is -1 when G was not enumerated (native pushdown).
            {'pool_size', 'channel_sizes', 'channels_used', 'gate_count', 'whole'}；channels_used 列出参与召回的通道，
            'gate' 在前；G 未被枚举（原生下推）时 gate_count 为 -1。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "pool_size": self.size,
            "channel_sizes": dict(self.channel_sizes),
            "channels_used": [name for name in (GATE_CHANNEL, *CHANNELS) if name in self.channel_sizes],
            "gate_count": int(self.gate_count),
            "whole": bool(self.whole),
        }


def _plan_keyword_tokens(plan: dict[str, Any], text: str | None) -> list[str]:
    """Query tokens of the keyword channel: content tokens + identifier tokens (from the plan, else from the text).
    关键词通道的查询词元：内容词元 + 标识符词元（取自规划，否则由文本派生）。
    """
    tokens = [str(t) for t in (plan.get("content_tokens") or [])]
    identifiers = [str(t) for t in (plan.get("identifier_tokens") or [])]
    if not tokens and not identifiers and text:
        tokens, identifiers = content_tokens(text), identifier_tokens(text)
    return ordered_union(tokens, identifiers)


def _merge_channels(lists: dict[str, list[str]], *, pool_max: int) -> tuple[list[str], dict[str, list[str]]]:
    """Round-robin union of the channel lists (CHANNELS order, duplicates dropped) cut at pool_max, plus origins.
    各通道列表的轮转并集（按 CHANNELS 顺序，去重）截到 pool_max，并附来源。
    """
    ids: list[str] = []
    seen: set[str] = set()
    longest = max((len(v) for v in lists.values()), default=0)
    # Round-robin keeps every channel's head when the union would overflow the pool.
    # 轮转合并保证并集溢出池时每个通道的头部都能保留。
    for position in range(longest):
        for name in CHANNELS:
            entries = lists.get(name) or []
            if position < len(entries) and entries[position] not in seen:
                seen.add(entries[position])
                ids.append(entries[position])
                if len(ids) >= pool_max:
                    break
        if len(ids) >= pool_max:
            break
    members = {name: set(entries) for name, entries in lists.items()}
    recalled_by = {atom_id: [name for name in CHANNELS if atom_id in members.get(name, ())] for atom_id in ids}
    return ids, recalled_by


def recall_pool(
    store: StateStore,
    index: Any,
    embedder: Any,
    plan: dict[str, Any],
    scope: Scope,
    *,
    as_of: str,
    include_expired: bool,
    cfg: RetrievalConfig,
    top_k: int,
    text: str | None = None,
    query_vector: list[float] | None = None,
    where_sql: str = "",
    params: Sequence[Any] = (),
    joins: Sequence[str] = (),
    scale_constraint: Sequence[str] | None = None,
    atom_types: Sequence[str] | None = None,
    gate_count: int | None = None,
    index_filter: IndexFilter | None = None,
    gate_entity_keys: Sequence[str] | None = None,
    index_hint: str | None = None,
) -> Pool:
    """Bounded multi-channel recall of one ladder level: G enters whole when it fits the pool, else the channel union.
    一个阶梯级别的有界多通道召回：门控集合 G 装得下池时整体入池，否则取各通道（各自截到 K）的轮转并集。

    The channels are semantic / keyword / entity / recency (CHANNELS order), merged round-robin and cut at pool_max;
    every channel searches inside G (the level gate travels into keyword_candidates / entity_candidates). Recency runs
    when the route is temporal or carries a hard time gate, when the whole-gate rule is off (so |G| <= K_rec still
    reaches the pool whole), or as a small fallback when the other channels together recalled fewer than K_rec ids.
    通道为语义 / 关键词 / 实体 / 新近度（CHANNELS 顺序），轮转合并并截到 pool_max；每个通道都在 G 内搜索（级别门控
    传入 keyword_candidates / entity_candidates）。新近度通道在路由为 temporal 或带硬时间门时运行，在整体入池规则
    关闭时运行（使 |G| <= K_rec 仍整体到达池），或在其它通道合计召回不足 K_rec 个 id 时作为小额兜底运行。

    Input / 输入：
        store: The StateStore (axis_ids / count_axis / keyword_candidates / entity_candidates).
        store：StateStore（axis_ids / count_axis / keyword_candidates / entity_candidates）。

        index: IndexBase provider for the semantic channel (None → no semantic channel).
        index：语义通道使用的 IndexBase 提供者（None → 无语义通道）。

        embedder: EmbedderBase used to embed `text` when query_vector is None (None → no semantic channel).
        embedder：query_vector 为 None 时用于嵌入 `text` 的 EmbedderBase（None → 无语义通道）。

        plan: The query plan dict (content_tokens, identifier_tokens, entity_hints).
        plan：查询规划字典（content_tokens、identifier_tokens、entity_hints）。

        scope: The query scope.
        scope：查询范围。

        as_of: Validity instant.
        as_of：有效性时刻。

        include_expired: Drop the validity predicate.
        include_expired：去掉有效性谓词。

        cfg: RetrievalConfig (channel K's, pool bound, hard-gate rule, safety limit).
        cfg：RetrievalConfig（各通道 K、池上限、硬门规则、安全上限）。

        top_k: The caller's top_k (drives cfg.k_for).
        top_k：调用方的 top_k（驱动 cfg.k_for）。

        text: Query text (embedded when query_vector is None; keyword tokens when the plan carries none).
        text：查询文本（query_vector 为 None 时嵌入；规划无词元时用于关键词）。

        query_vector: Pre-embedded query.
        query_vector：预先嵌入的查询向量。

        where_sql, params, joins: The level's gate WHERE over memory_atoms (filters + plan gates), as axis_select; when gate_entity_keys is given (and there are no joins) the entity gate must NOT be in it — recall_pool drives it through atom_entity and adds the right SQL form per statement.
        where_sql、params、joins：该级别针对 memory_atoms 的门控 WHERE（过滤 + 规划门控），同 axis_select；给了 gate_entity_keys（且无 joins）时它**不得**包含实体门——recall_pool 经 atom_entity 驱动实体门，并按语句加上合适的 SQL 形式。

        scale_constraint: scale_level whitelist of the level.
        scale_constraint：该级别的 scale_level 白名单。

        atom_types: atom_type whitelist.
        atom_types：atom_type 白名单。

        gate_count: |G| if already known; None → an unordered pool_max + 1 id probe decides whole vs channels (exact |G| from the id fetch; -1 under pushdown, where G is not enumerated).
        gate_count：已知的 |G|；None → 一次不排序的 pool_max + 1 个 id 探测决定整体入池还是走通道（精确 |G| 来自 id 读取；下推时为 -1，此时不枚举 G）。

        index_filter: Native pushdown gate for `index.supports_filter` providers (None → gate ids in SQL first).
        index_filter：供 `index.supports_filter` 提供者原生下推的门控（None → 先在 SQL 里门控 id）。

        gate_entity_keys: Entity keys of the level's hard entity gate; without joins G is then enumerated ONCE through entity_candidates (atom_entity-driven, time order), which replaces the probe and feeds the entity channel.
        gate_entity_keys：该级别硬实体门的实体键；无 joins 时 G 经 entity_candidates（由 atom_entity 驱动、时间序）**一次**枚举，取代探测并供实体通道使用。

        index_hint: `INDEXED BY` name handed to the store's memory_atoms reads (schema.TIME_INDEX under a hard time gate).
        index_hint：交给 store 读取 memory_atoms 时使用的 `INDEXED BY` 索引名（硬时间门下为 schema.TIME_INDEX）。

    Output / 输出：
        Pool: ids in merge order (index order when G enters whole; hydration restores axis order), recalled_by per id, channel_sizes, gate_count, whole, semantic (channel scores).
        Pool：按合并顺序的 id（G 整体入池时为索引序；水合恢复轴序）、每个 id 的 recalled_by、channel_sizes、gate_count、whole、semantic（通道分数）。

    State effect / 状态影响：
        Read-only (memory_atoms, atom_entity, entities, atom_fts, atom_vectors); one logger.warning when |G| or the
        pool exceeds cfg.safety_limit — nothing is ever truncated silently.
        只读（memory_atoms、atom_entity、entities、atom_fts、atom_vectors）；|G| 或池超过 cfg.safety_limit 时记一条
        logger.warning——绝不静默截断。
    """
    ks = cfg.k_for(top_k)
    gate_kwargs: dict[str, Any] = dict(
        where_sql=where_sql,
        params=list(params),
        joins=list(joins),
        as_of=as_of,
        include_expired=include_expired,
        scale_constraint=list(scale_constraint) if scale_constraint else None,
        atom_types=list(atom_types) if atom_types else None,
        index_hint=index_hint,
    )
    pool = Pool()
    entity_keys = [k for k in dict.fromkeys(str(k) for k in (gate_entity_keys or [])) if k]
    pushdown = index_filter is not None and getattr(index, "supports_filter", False)
    entity_gate = bool(entity_keys) and not gate_kwargs["joins"]
    gate_ids: list[str] | None = None
    if entity_gate:
        # A hard entity gate reads G through atom_entity ONCE (link order, never sorted): under pushdown a probe
        # that stops at pool_max + 1 gated links (when G fits it already is the pool; otherwise the provider gets
        # the ungated links plus the filter), else the whole unordered G (the provider's candidate list).
        # 硬实体门经 atom_entity **一次**读取 G（链接顺序，从不排序）：下推时是在 pool_max + 1 个门控内链接处停止的
        # 探测（G 装得下时它就是池；否则提供者拿到未门控链接加过滤器），否则是整个无序的 G（提供者的候选列表）。
        probe = store.entity_candidates(
            scope, entity_keys, limit=ks["pool"] + 1 if pushdown else None, order=None, **gate_kwargs
        )
        if not pushdown or len(probe) <= ks["pool"]:
            gate_count = len(probe)
            gate_ids = probe
    elif gate_count is None and cfg.hard_gate_pool:
        # One unordered probe of pool_max + 1 ids (index order, no sort, stops early) replaces a COUNT(*): it
        # decides whole vs channels and, when G fits, already is the pool.
        # 一次不排序的 pool_max + 1 个 id 探测（索引序、不排序、提前停止）取代 COUNT(*)：它决定整体入池还是走通道，
        # 且当 G 装得下时它本身就是池。
        probe = store.axis_ids(scope, order=None, limit=ks["pool"] + 1, **gate_kwargs)
        if len(probe) <= ks["pool"]:
            gate_count = len(probe)
            gate_ids = probe
    if gate_count is not None and int(gate_count) <= 0:
        return pool
    if cfg.hard_gate_pool and gate_count is not None and int(gate_count) <= ks["pool"]:
        # The whole gated set fits the pool: score all of it (no channel can lose an atom; RETRIEVAL_DIAGNOSIS §4).
        # 门控集合整体装得下池：全部打分（任何通道都不会丢原子；RETRIEVAL_DIAGNOSIS §4）。
        if gate_ids is None:
            if entity_gate:
                gate_ids = store.entity_candidates(scope, entity_keys, limit=None, order=None, **gate_kwargs)
            else:
                gate_ids = store.axis_ids(scope, order=None, **gate_kwargs)
        pool.ids = gate_ids
        pool.recalled_by = {atom_id: [GATE_CHANNEL] for atom_id in gate_ids}
        pool.channel_sizes = {GATE_CHANNEL: len(gate_ids)}
        pool.gate_count = int(gate_count)
        pool.whole = True
    else:
        lists: dict[str, list[str]] = {}
        if index is not None and query_vector is None and embedder is not None and text:
            query_vector = embedder.embed(text, "search")
        semantic_on = index is not None and query_vector is not None
        tokens = _plan_keyword_tokens(plan, text)
        keys = [k for k in (_hint_key(h) for h in (plan.get("entity_hints") or [])) if k]
        # Recency runs a priori for temporal routes / hard time gates / the whole-gate rule off; the small fallback
        # (other channels short of K_rec) is decided after them and costs one more bounded statement.
        # 新近度在 temporal 路由 / 硬时间门 / 整体入池规则关闭时先验运行；小额兜底（其它通道不足 K_rec）在它们
        # 之后决定，多付一条有界语句。
        recency_first = (
            bool(plan.get("hard_time_gate"))
            or str(plan.get("route") or "") == Route.temporal.value
            or not cfg.hard_gate_pool
        )
        # The SQL channels (keyword / entity / recency) and, under a pushed-down entity gate, the ungated links
        # feeding the semantic channel, in ONE statement (RETRIEVAL_DIAGNOSIS §5 / RETRIEVAL_BENCH §3).
        # SQL 通道（关键词 / 实体 / 新近度）以及下推的实体门下供语义通道使用的未门控链接，合成一条语句
        # （RETRIEVAL_DIAGNOSIS §5 / RETRIEVAL_BENCH §3）。
        channels = store.recall_channels(
            scope,
            keyword=(tokens, ks["keyword"]) if tokens else None,
            entity=(keys, ks["entity"]) if keys else None,
            recency=ks["recency"] if recency_first else None,
            links=entity_keys if (entity_gate and pushdown and semantic_on) else None,
            entity_gate=entity_keys if entity_gate else None,
            **gate_kwargs,
        )
        if semantic_on:
            vector = [float(v) for v in query_vector]
            if pushdown:
                # Native filtered ANN / masked matmul: the gate travels as scalars; an entity gate travels as the
                # ungated link ids intersected with the same filter (RETRIEVAL_DIAGNOSIS §3.3).
                # 原生过滤 ANN / 掩码矩阵乘：门控以标量下推；实体门以未门控链接 id 与同一过滤器求交的形式下推
                # （RETRIEVAL_DIAGNOSIS §3.3）。
                if entity_gate:
                    link_ids = [atom_id for atom_id, _score in channels.get("links", [])]
                    hits = (
                        index.search(vector, candidate_ids=link_ids, filter=index_filter, top_k=ks["semantic"])
                        if link_ids
                        else []
                    )
                else:
                    hits = index.search(vector, filter=index_filter, top_k=ks["semantic"])
            else:
                # ids only, no hydration: the provider does one matmul over G and returns the K_sem best.
                # 只取 id、不水合：提供者对 G 做一次矩阵乘并返回最优的 K_sem 个。
                if gate_ids is None:
                    if entity_gate:
                        gate_ids = store.entity_candidates(scope, entity_keys, limit=None, order=None, **gate_kwargs)
                    else:
                        gate_ids = store.axis_ids(scope, order=None, **gate_kwargs)
                hits = index.search(vector, candidate_ids=gate_ids, top_k=ks["semantic"]) if gate_ids else []
            lists["semantic"] = [str(atom_id) for atom_id, _score in hits]
            pool.semantic = {str(atom_id): float(score) for atom_id, score in hits}
        if "keyword" in channels:
            lists["keyword"] = [atom_id for atom_id, _score in channels["keyword"]]
        if "entity" in channels:
            lists["entity"] = [atom_id for atom_id, _score in channels["entity"]]
        if "recency" in channels:
            lists["recency"] = [atom_id for atom_id, _score in channels["recency"]]
        else:
            recalled = {atom_id for entries in lists.values() for atom_id in entries}
            if len(recalled) < ks["recency"]:
                lists["recency"] = [
                    atom_id
                    for atom_id, _score in store.recall_channels(
                        scope, recency=ks["recency"], entity_gate=entity_keys if entity_gate else None, **gate_kwargs
                    )["recency"]
                ]
        pool.ids, pool.recalled_by = _merge_channels(lists, pool_max=ks["pool"])
        pool.channel_sizes = {name: len(lists[name]) for name in CHANNELS if name in lists}
        if gate_count is None:
            if gate_ids is not None:
                gate_count = len(gate_ids)
            elif pushdown:
                gate_count = -1
            else:
                gate_count = store.count_axis(scope, **gate_kwargs)
        pool.gate_count = int(gate_count)
    if pool.gate_count > cfg.safety_limit:
        logger.warning(
            "retrieval: gated set of %d atoms exceeds retrieval.safety_limit=%d; recall stays bounded per channel",
            pool.gate_count,
            cfg.safety_limit,
        )
    if pool.size > cfg.safety_limit:
        logger.warning(
            "retrieval: pool of %d atoms exceeds retrieval.safety_limit=%d; scoring all of it",
            pool.size,
            cfg.safety_limit,
        )
    return pool


def _lite_pool(
    store: StateStore,
    scope: Scope,
    ids: Sequence[str],
    *,
    where_sql: str,
    params: Sequence[Any],
    joins: Sequence[str],
    as_of: str,
    include_expired: bool,
    scale_constraint: Sequence[str] | None,
    atom_types: Sequence[str] | None,
) -> list[AtomLite]:
    """The pool's scoring rows (the gate WHERE re-applied: pool ∩ G) in axis order, no models built.
    池的打分行（重新施加门控 WHERE：池 ∩ G），轴序，不构造模型。
    """
    return store.axis_lite(
        scope,
        list(ids),
        where_sql=where_sql,
        params=list(params),
        joins=list(joins),
        as_of=as_of,
        include_expired=include_expired,
        scale_constraint=scale_constraint,
        atom_types=atom_types,
    )


def _n_anchors(atom: Any) -> int:
    """Undisputed anchor count of a MemoryAtom (len(anchor_ids)) or an AtomLite row (n_anchors).
    MemoryAtom（len(anchor_ids)）或 AtomLite 行（n_anchors）的无争议锚点数。
    """
    return int(atom.n_anchors) if isinstance(atom, AtomLite) else len(atom.anchor_ids)


def _latest_per_group(atoms: list[Any], *, by_value: bool) -> list[Any]:
    """Newest atom overall, or newest per (subject, predicate) group in first-appearance order when by_value.
    整体最新的原子；by_value 时为每个 (subject, predicate) 组按首次出现顺序取最新。
    """
    ordered = _axis_sorted(atoms)
    if not ordered:
        return []
    if not by_value:
        return [ordered[0]]
    seen: set[tuple[str | None, str | None]] = set()
    out: list[Any] = []
    for atom in ordered:
        key = (atom.subject, atom.predicate)
        if key in seen:
            continue
        seen.add(key)
        out.append(atom)
    return out


def _projection(project_axis: str, hits: list[MemoryAtom]) -> list[Any] | dict[str, list[str]] | None:
    """Build the projection of §13.3 step 5 for the final hits.
    为最终命中构造 §13.3 第 5 步的投影。
    """
    if project_axis == "atom":
        return None
    if project_axis == "value":
        return [list(atom.content.get("objects") or [atom.text]) for atom in hits]
    if project_axis == "time":
        return sorted((atom.time_anchor or atom.created_at or "", atom.atom_id) for atom in hits)
    if project_axis == "entity":
        keyed: dict[str, list[str]] = {}
        for atom in hits:
            for key in atom.entity_ids:
                keyed.setdefault(key, []).append(atom.atom_id)
        return {key: keyed[key] for key in sorted(keyed)}
    keyed_topics: dict[str, list[str]] = {}
    for atom in hits:
        for topic in atom.topic_tags:
            keyed_topics.setdefault(topic, []).append(atom.atom_id)
    return {topic: keyed_topics[topic] for topic in sorted(keyed_topics)}


def _scoring_plan(aq: AxisQuery, as_of: str, extractor: Any | None = None) -> dict[str, Any]:
    """The plan used for signals: aq.plan, or tokens/hints derived from the text when no plan was given (hints from
    `extractor` when given, else utils.text.extract_entities).
    用于信号的规划：aq.plan，或在未给规划时由文本派生的词元/线索（给了 `extractor` 时线索来自它，否则来自
    utils.text.extract_entities）。
    """
    if aq.plan:
        return dict(aq.plan)
    if aq.text is None:
        return {}
    hints = extractor.extract(aq.text) if extractor is not None else extract_entities(aq.text)
    return {
        "route": Route.recall.value,
        "content_tokens": content_tokens(aq.text),
        "entity_hints": [dict(h._asdict()) for h in hints],
    }


def execute_axis_query(
    store: StateStore,
    index: Any,
    embedder: Any,
    aq: AxisQuery,
    *,
    scoring: ScoringConfig,
    capabilities: dict,
    threshold: float = 0.0,
    explain: bool = False,
    retrieval: RetrievalConfig | None = None,
    retrieval_stats: dict[str, Any] | None = None,
    extractor: Any | None = None,
) -> AxisResult:
    """Execute an AxisQuery: SQL gates, the fallback ladder, bounded recall into a pool, in-pool scoring, projection.
    执行 AxisQuery：SQL 门控，回退阶梯，有界召回入池，池内打分，投影。

    Input / 输入：
        store: The StateStore (axis_select / axis_ids / count_axis / channels / entity degrees).
        store：StateStore（axis_select / axis_ids / count_axis / 通道 / 实体度数）。

        index: IndexBase provider for semantic similarity (None → 0.0 everywhere).
        index：用于语义相似度的 IndexBase 提供者（None → 全为 0.0）。

        embedder: EmbedderBase used to embed aq.text (None → no semantic signal).
        embedder：用于嵌入 aq.text 的 EmbedderBase（None → 无语义信号）。

        aq: The AxisQuery (filters, scope, text, plan, projection and aggregation).
        aq：AxisQuery（过滤、范围、文本、规划、投影与聚合）。

        scoring: ScoringConfig (weights, half-life, identifier weight).
        scoring：ScoringConfig（权重、半衰期、标识符权重）。

        capabilities: store.capabilities (json1 decides SQL vs Python meta filters).
        capabilities：store.capabilities（json1 决定元数据过滤走 SQL 还是 Python）。

        threshold: Minimum fused score kept for ranked queries (applied after fusion).
        threshold：排序查询保留的最低融合分数（融合后施加）。

        explain: Attach score_details to every hit.
        explain：为每个命中附加 score_details。

        retrieval: RetrievalConfig bounding the recall channels and the pool (None → shipped defaults).
        retrieval：约束召回通道与池的 RetrievalConfig（None → 出厂默认）。

        retrieval_stats: When given, updated in place with Pool.stats() of the winning ladder level.
        retrieval_stats：给定时原地写入胜出阶梯级别的 Pool.stats()。

        extractor: The facade's entity extractor for the ad-hoc plan of a text query without aq.plan (None → utils.text.extract_entities).
        extractor：门面的实体抽取器，用于没有 aq.plan 的文本查询的临时规划（None → utils.text.extract_entities）。

    Output / 输出：
        AxisResult(hits, project_axis, aggregation, projection, count, summary, primary_axis, retrieval); retrieval is
        Pool.stats() of the winning level for ranked queries ({} for axis-only / count); primary_axis records
        the ladder level that produced the candidates: the gate axis / 'semantic' / 'time' when the plan gates and
        the route's scale default held, 'scale_fallback' when only the scale default was dropped, 'fallback' when
        the plan gates were dropped (an explicit aq.scale_constraint is never relaxed).
        AxisResult(hits, project_axis, aggregation, projection, count, summary, primary_axis, retrieval)；排序查询的
        retrieval 是胜出级别的 Pool.stats()（仅轴 / count 时为 {}）；primary_axis 记录
        产生候选的阶梯级别：规划门控与路由默认尺度都成立时为门控轴 / 'semantic' / 'time'，仅放弃默认尺度时为
        'scale_fallback'，放弃规划门控时为 'fallback'（显式的 aq.scale_constraint 从不放宽）。

    State effect / 状态影响：
        Read-only; touches only memory_atoms, atom_entity, entities, atom_topic, atom_anchor, atom_vectors, atom_fts.
        A pool or an axis-only read that crosses retrieval.safety_limit logs a warning (never a silent cut); a ranked
        level under a Python-side filter (no JSON1) logs a warning and scores the whole gated set instead of recalling.
        只读；仅触及 memory_atoms、atom_entity、entities、atom_topic、atom_anchor、atom_vectors、atom_fts。
        池或仅轴读取越过 retrieval.safety_limit 时记录警告（绝不静默截断）；Python 侧过滤（无 JSON1）下的排序级别
        记录警告并对整个门控集合打分而不做召回。
    """
    capabilities = capabilities or {}
    cfg = retrieval or _DEFAULT_RETRIEVAL
    fscope, expr = parse_filters(aq.filters, require_scope=False, context="axis_query")
    scope = aq.scope if (aq.scope is not None and not aq.scope.is_empty()) else fscope
    scope.require_any(context="axis_query")
    _, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context="meta", force_meta=True)
    parts: list[FilterExpr] = [e for e in (expr, meta_expr) if e is not None]
    combined: FilterExpr | None = None if not parts else (parts[0] if len(parts) == 1 else And(items=parts))
    sql_expr, py_expr = plan_filters(combined, capabilities=capabilities)
    where_sql, params, joins = lower_to_sql(sql_expr, capabilities=capabilities, scope=scope)

    plan = dict(aq.plan or {})
    as_of = aq.as_of or store.now()
    aggregation = aq.aggregation or "list"
    explicit_scale = [str(s) for s in aq.scale_constraint] if aq.scale_constraint else None
    plan_scale = [str(s) for s in (plan.get("scale_default") or [])] or None
    scale_constraint = explicit_scale or plan_scale
    # An explicit aq.scale_constraint is the scale axis itself (§3.5) and is never relaxed; only the route's
    # scale default is a soft preference, so only it takes part in the fallback ladder (and only when it
    # actually excludes a scale: the recall route lists every scale, so dropping it would change nothing).
    # 显式的 aq.scale_constraint 就是尺度轴本身（§3.5），从不放宽；只有路由的默认尺度是软偏好，
    # 因此只有它参与回退阶梯（且仅当它确实排除了某个尺度：recall 路由列出全部尺度，放弃它不会改变任何东西）。
    scale_relaxable = explicit_scale is None and plan_scale is not None and not set(ALL_SCALES) <= set(plan_scale)
    atom_types = plan.get("atom_types")
    gate_clauses, gate_params, gate_names = _plan_gates(plan, scope, capabilities)
    # The entity gate has three SQL forms (StateStore.entity_gate_sql): the materialized IN list for scans (axis-only
    # reads, counts), the correlated EXISTS for the id-driven pool reads, and none for the pool recall, which drives
    # the gate through atom_entity itself (gate_entity_keys) unless a filter join is present.
    # 实体门有三种 SQL 形式（StateStore.entity_gate_sql）：扫描（仅轴读取、计数）用物化的 IN 列表，按 id 驱动的池
    # 读取用相关 EXISTS，池召回则不用——它自己经 atom_entity 驱动门控（gate_entity_keys），除非存在过滤连接。
    forms = {
        form: _plan_gates(plan, scope, capabilities, entity_form=form)[:2] for form in ("in", "exists", "none")
    }

    def _where(with_gates: bool, entity_form: str = "in") -> tuple[str, list[Any]]:
        clauses = [where_sql] if where_sql else []
        values = list(params)
        if with_gates:
            form_clauses, form_params = forms[entity_form]
            clauses.extend(form_clauses)
            values.extend(form_params)
        return " AND ".join(f"({c})" for c in clauses), values

    def _hint(with_gates: bool, with_scale: bool) -> str | None:
        # The most selective gate of the level names the memory_atoms index (time gate > obs_type gate > the scale
        # whitelist); a join or an entity gate leaves the choice to the planner. Without statistics SQLite would
        # walk the scope index for every one of these and the empty ladder levels would each cost a scope scan.
        # 该级别中最具选择性的门控指名 memory_atoms 的索引（时间门 > obs_type 门 > 尺度白名单）；有连接或实体门时
        # 交给规划器。没有统计时 SQLite 会为它们逐个走遍 scope 索引，空的阶梯级别也各付一次 scope 扫描。
        if joins or (with_gates and "entity" in gate_names):
            return None
        if with_gates and "time" in gate_names:
            return TIME_INDEX
        type_gate = plan.get("hard_type_gate") if with_gates else None
        if type_gate and str(type_gate[0]) == "obs_type":
            return OBS_TYPE_INDEX
        if with_scale and scale_constraint:
            return SCALE_INDEX
        return None

    def _py_filter(rows: list[MemoryAtom]) -> list[MemoryAtom]:
        if py_expr is None:
            return rows
        return [a for a in rows if evaluate(py_expr, a, entity_keys=set(a.entity_ids), anchor_ids=set(a.anchor_ids))]

    def _select(with_gates: bool, with_scale: bool, *, limit: int | None) -> list[MemoryAtom]:
        # Axis-only reads: the caller's limit when SQL alone decides membership, else the safety limit (warned).
        # 仅轴读取：SQL 单独决定成员资格时用调用方的 limit，否则用安全上限（越过时告警）。
        sql, values = _where(with_gates)
        cap = limit if (limit is not None and py_expr is None) else cfg.safety_limit
        rows = store.axis_select(
            scope,
            where_sql=sql,
            params=values,
            joins=list(joins),
            as_of=as_of,
            include_expired=aq.include_expired,
            scale_constraint=scale_constraint if with_scale else None,
            atom_types=atom_types,
            order=AXIS_ORDER,
            limit=cap,
            index_hint=_hint(with_gates, with_scale),
        )
        if cap == cfg.safety_limit and len(rows) >= cap:
            logger.warning("axis read reached retrieval.safety_limit=%d rows; older rows were not loaded", cap)
        return _py_filter(rows)

    def _count(with_gates: bool, with_scale: bool) -> int:
        if py_expr is not None:
            return len(_select(with_gates, with_scale, limit=None))
        sql, values = _where(with_gates)
        return _count_rows(
            store,
            scope,
            where_sql=sql,
            params=values,
            as_of=as_of,
            include_expired=aq.include_expired,
            scale_constraint=scale_constraint if with_scale else None,
            atom_types=atom_types,
            index_hint=_hint(with_gates, with_scale),
        )

    def _index_filter(with_gates: bool, with_scale: bool) -> IndexFilter | None:
        # Pushdown covers scope, validity, scale, obs_type, atom_type and the time gate; an entity gate rides it
        # as candidate_ids (recall_pool passes the ungated links); any filter WHERE or join keeps the SQL-gated
        # candidate_ids path (the pool is intersected with G anyway).
        # 下推覆盖 scope、有效性、尺度、obs_type、atom_type 与时间门；实体门以 candidate_ids 形式搭乘它
        # （recall_pool 传入未门控链接）；任何过滤 WHERE 或连接都保留 SQL 门控的 candidate_ids 路径（池反正要与 G 求交）。
        if where_sql or joins:
            return None
        time_gate = plan.get("hard_time_gate") if with_gates else None
        type_gate = plan.get("hard_type_gate") if with_gates else None
        obs_types = tuple(str(v) for v in type_gate[1]) if type_gate and str(type_gate[0]) == "obs_type" else None
        if type_gate and obs_types is None:
            return None
        scales = tuple(scale_constraint) if (with_scale and scale_constraint) else None
        return IndexFilter(
            scope=scope,
            as_of=as_of,
            include_expired=bool(aq.include_expired),
            scale_in=scales,
            obs_type_in=obs_types,
            time_from=to_utc_iso(time_gate[0]) if time_gate else None,
            time_to=to_utc_iso(time_gate[1]) if time_gate else None,
            atom_type_in=tuple(str(t) for t in atom_types) if atom_types else None,
        )

    gated = bool(gate_names)
    primary_axis = _GATE_AXIS[gate_names[0]] if gate_names else ("semantic" if aq.text is not None else "time")
    matched_gates = list(gate_names)
    ladder = _fallback_ladder(gated=gated, scale_relaxable=scale_relaxable, primary_axis=primary_axis)

    # ---- count: one SQL statement per ladder level, no rows loaded --------------------------------------------
    # ---- 计数：每个阶梯级别一条 SQL 语句，不加载任何行 ----------------------------------------------------------
    if aggregation == "count":
        count = 0
        for with_gates, with_scale, label in ladder:
            count = _count(with_gates, with_scale)
            primary_axis = label
            if count:
                break
        return AxisResult(
            hits=[],
            project_axis=aq.project_axis,
            aggregation="count",
            projection=None,
            count=count,
            summary=None,
            primary_axis=primary_axis,
        )

    limit = max(0, int(aq.limit))
    by_value = aq.project_axis == "value"
    retrieval: dict[str, Any] = {}
    scores: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    signal_axes: dict[str, list[str]] = {}
    summary: str | None = None

    if aq.text is None:
        # ---- axis-only: the SQL path, candidates in axis order with the caller's limit, score 0.0 ----------------
        # ---- 仅轴：SQL 路径，候选按轴序、带调用方 limit，分数为 0.0 -----------------------------------------------
        # 'latest' over values needs every group and 'summary' counts the whole gated set: only 'list' / plain
        # 'latest' can stop at the caller's limit.
        # 按值的 'latest' 需要每个分组，'summary' 要数整个门控集合：只有 'list' / 普通 'latest' 能在调用方 limit 处停下。
        if aggregation == "latest":
            read_limit = None if by_value else 1
        else:
            read_limit = limit
        candidates: list[MemoryAtom] = []
        total = 0
        for with_gates, with_scale, label in ladder:
            candidates = _select(with_gates, with_scale, limit=read_limit)
            primary_axis = label
            matched_gates = list(gate_names) if with_gates else []
            if candidates:
                total = _count(with_gates, with_scale) if aggregation == "summary" else len(candidates)
                break
        ordered = list(candidates)
        if aggregation == "latest":
            final_atoms = _latest_per_group(ordered, by_value=by_value)[:limit]
        else:
            final_atoms = ordered[:limit]
        if aggregation == "summary":
            summary = f"{total} atoms; latest: {ordered[0].text}" if ordered else "0 atoms"
    else:
        # ---- ranked: bounded recall into a pool per ladder level, signals + fusion inside the pool -------------
        # ---- 排序：每个阶梯级别有界召回入池，池内计算信号并融合 ---------------------------------------------------
        sig_plan = _scoring_plan(aq, as_of, extractor)
        query_vector = embedder.embed(aq.text, "search") if embedder is not None else None
        pool = Pool()
        candidates = []
        for with_gates, with_scale, label in ladder:
            entity_driven = bool(with_gates and "entity" in gate_names and not joins)
            sql, values = _where(with_gates, "exists" if entity_driven else "in")
            level_scale = scale_constraint if with_scale else None
            primary_axis = label
            matched_gates = list(gate_names) if with_gates else []
            if py_expr is not None:
                # A Python-side predicate (metadata filter without JSON1) cannot travel into the channels: bounding
                # them on the SQL gate alone would recall K ids of which none need pass the predicate. The level is
                # enumerated in SQL (safety_limit, warned), filtered in Python and scored whole — the exhaustive
                # semantics, reported as the 'gate' channel.
                # Python 侧谓词（无 JSON1 的元数据过滤）无法进入通道：只按 SQL 门控约束通道会召回 K 个 id，而它们
                # 可能没有一个通过谓词。该级别在 SQL 中枚举（safety_limit，越过时告警）、在 Python 中过滤、整体打分
                # ——即穷举语义，以 'gate' 通道报告。
                candidates = _select(with_gates, with_scale, limit=None)
                pool = Pool()
                pool.ids = [a.atom_id for a in candidates]
                pool.recalled_by = {a.atom_id: [GATE_CHANNEL] for a in candidates}
                pool.channel_sizes = {GATE_CHANNEL: len(candidates)}
                pool.gate_count = len(candidates)
                pool.whole = True
                if candidates:
                    logger.warning(
                        "retrieval: a Python-side filter (no JSON1) disables bounded recall; %d gated atoms scored whole",
                        len(candidates),
                    )
                    break
                continue
            recall_sql, recall_values = _where(with_gates, "none") if entity_driven else (sql, values)
            pool = recall_pool(
                store,
                index,
                embedder,
                sig_plan,
                scope,
                as_of=as_of,
                include_expired=aq.include_expired,
                cfg=cfg,
                top_k=limit,
                text=aq.text,
                query_vector=query_vector,
                where_sql=recall_sql,
                params=recall_values,
                joins=joins,
                scale_constraint=level_scale,
                atom_types=atom_types,
                index_filter=_index_filter(with_gates, with_scale),
                gate_entity_keys=[str(k) for k in (plan.get("hard_entity_gate") or [])] if entity_driven else None,
                index_hint=_hint(with_gates, with_scale),
            )
            if not pool.ids:
                continue
            # The pool is scored on lite rows (pool ∩ G, no model built); only the final top_k is hydrated below.
            # 池在轻量行上打分（池 ∩ G，不构造模型）；只有最终 top_k 在下面水合。
            candidates = _lite_pool(
                store,
                scope,
                pool.ids,
                where_sql=sql,
                params=values,
                joins=joins,
                as_of=as_of,
                include_expired=aq.include_expired,
                scale_constraint=level_scale,
                atom_types=atom_types,
            )
            if candidates:
                break
        retrieval = pool.stats()
        if retrieval_stats is not None:
            retrieval_stats.update(retrieval)
        ids = [a.atom_id for a in candidates]
        # The semantic channel already scored its hits with the provider's `dot`: score only the rest of the pool.
        # 语义通道已用提供者的 `dot` 给其命中打过分：只对池的其余部分再打分。
        semantic = semantic_scores(index, query_vector, [i for i in ids if i not in pool.semantic])
        semantic.update({i: max(0.0, float(pool.semantic[i])) for i in ids if i in pool.semantic})
        prepared = prepare_query(sig_plan, scoring)
        weights = fusion_weights(scoring)
        degrees: dict[str, int] = {}
        if prepared.keys:
            for key, entity_id in store.resolve_entities(scope, sorted(prepared.keys)).items():
                degrees[key] = store.entity_degree(entity_id)
        ranked: list[tuple[Any, Signals, float, float, float]] = []
        for atom in candidates:
            signals = compute_signals(
                atom=atom,
                semantic=semantic.get(atom.atom_id, 0.0),
                plan=sig_plan,
                as_of=as_of,
                scoring=scoring,
                degrees=degrees,
                n_anchors=_n_anchors(atom),
                prepared=prepared,
                age_us=atom.age_us if isinstance(atom, AtomLite) else None,
            )
            final, raw, max_possible = fuse(signals, scoring, weights=weights)
            if final < threshold:
                continue
            ranked.append((atom, signals, final, raw, max_possible))
        ranked.sort(key=lambda r: r[0].atom_id)
        ranked.sort(key=lambda r: (r[0].time_anchor or "", r[0].created_at or ""), reverse=True)
        ranked.sort(key=lambda r: r[2], reverse=True)
        for atom, signals, final, raw, max_possible in ranked:
            scores[atom.atom_id] = final
            signal_axes[atom.atom_id] = [name for name in signals.active if float(getattr(signals, name)) > 0.0]
            if explain:
                details[atom.atom_id] = score_details(
                    signals,
                    final,
                    raw,
                    max_possible,
                    threshold=threshold,
                    plan=sig_plan,
                    primary_axis=primary_axis,
                    scope=scope.as_filters(),
                    as_of=as_of,
                    recalled_by=pool.recalled_by.get(atom.atom_id),
                    pool_size=pool.size,
                    channel_sizes=pool.channel_sizes,
                )
        ranked_atoms = [r[0] for r in ranked]
        if aggregation == "latest":
            final_atoms = _latest_per_group(ranked_atoms, by_value=by_value)[:limit]
        else:
            final_atoms = ranked_atoms[:limit]
        if aggregation == "summary":
            newest = _axis_sorted(ranked_atoms)
            summary = f"{len(ranked_atoms)} atoms; latest: {newest[0].text}" if newest else "0 atoms"
        if final_atoms and isinstance(final_atoms[0], AtomLite):
            # Hydrate only the returned rows (primary-key lookups under the same gate, models built from the
            # validated rows); a row archived meanwhile simply drops out, as it would have under the old hydration.
            # 只水合返回的行（同一门控下的主键查找，模型由已校验的行构造）；期间被归档的行直接落出，
            # 与旧的水合方式一致。
            hydrated = {
                atom.atom_id: atom
                for atom in store.axis_select(
                    scope,
                    where_sql=sql,
                    params=values,
                    joins=list(joins),
                    as_of=as_of,
                    include_expired=aq.include_expired,
                    scale_constraint=level_scale,
                    atom_types=atom_types,
                    atom_ids=[a.atom_id for a in final_atoms],
                    trusted=True,
                )
            }
            final_atoms = [hydrated[a.atom_id] for a in final_atoms if a.atom_id in hydrated]

    hits = [
        AtomHit(
            atom=atom,
            score=scores.get(atom.atom_id, 0.0),
            score_details=details.get(atom.atom_id, {}),
            matched_axes=ordered_union(matched_gates, signal_axes.get(atom.atom_id, [])),
        )
        for atom in final_atoms
    ]
    return AxisResult(
        hits=hits,
        project_axis=aq.project_axis,
        aggregation=aggregation,
        projection=_projection(aq.project_axis, final_atoms),
        count=None,
        summary=summary,
        primary_axis=primary_axis,
        retrieval=retrieval,
    )


def count_axis_query(store: StateStore, aq: AxisQuery) -> int:
    """Count the atoms an AxisQuery would gate (aggregation forced to 'count'; no ranking, no rows loaded).
    统计 AxisQuery 门控下的原子数（聚合强制为 'count'；不排序、不加载行）。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        aq: The AxisQuery.
        aq：AxisQuery。

    Output / 输出：
        Row count.
        行数。

    State effect / 状态影响：
        Read-only.
        只读。
    """
    data = aq.model_dump()
    data["aggregation"] = "count"
    result = execute_axis_query(
        store, None, None, AxisQuery(**data), scoring=_DEFAULT_SCORING, capabilities=store.capabilities
    )
    return int(result.count or 0)


def search(
    store: StateStore,
    index: Any,
    embedder: Any,
    *,
    query: str,
    filters: dict,
    top_k: int,
    threshold: float,
    explain: bool,
    as_of: str,
    include_expired: bool,
    scoring: ScoringConfig,
    capabilities: dict,
    retrieval: RetrievalConfig | None = None,
    extractor: Any | None = None,
) -> list[dict]:
    """Mem0-shaped search: plan the query, run a 'topk' AxisQuery (bounded pool recall) and project the hits to rows.
    Mem0 形状的搜索：规划查询，执行 'topk' AxisQuery（有界池召回），并把命中投影为行。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        index: IndexBase provider.
        index：IndexBase 提供者。

        embedder: EmbedderBase.
        embedder：EmbedderBase。

        query: Non-empty query text.
        query：非空查询文本。

        filters: Mem0 filter dict (scope keys required).
        filters：Mem0 过滤字典（需含范围键）。

        top_k: Maximum rows.
        top_k：最大行数。

        threshold: Minimum final score.
        threshold：最低最终分数。

        explain: Attach score_details to every row.
        explain：为每行附加 score_details。

        as_of: Validity instant (Mem0 reference_date).
        as_of：有效性时刻（Mem0 reference_date）。

        include_expired: Drop the validity predicate (Mem0 show_expired).
        include_expired：去掉有效性谓词（Mem0 show_expired）。

        scoring: ScoringConfig.
        scoring：ScoringConfig。

        capabilities: store.capabilities.
        capabilities：store.capabilities。

        retrieval: RetrievalConfig (channel K's / pool bound); None → shipped defaults.
        retrieval：RetrievalConfig（各通道 K / 池上限）；None → 出厂默认。

        extractor: The facade's entity extractor for plan_query's entity hints (CONTRACT §9.6); None → utils.text.extract_entities.
        extractor：门面的实体抽取器，供 plan_query 的实体线索使用（CONTRACT §9.6）；None → utils.text.extract_entities。

    Output / 输出：
        Rows (§4.1) with 'score' (+ 'score_details' when explain) in rank order.
        按排名顺序、带 'score'（explain 时另有 'score_details'）的行（§4.1）。

    State effect / 状态影响：
        Read-only; never reads a graph_* table.
        只读；从不读取 graph_* 表。
    """
    plan = plan_query(query, as_of=as_of, extractor=extractor)
    aq = AxisQuery(
        filters=dict(filters or {}),
        text=query,
        limit=top_k,
        as_of=as_of,
        include_expired=include_expired,
        aggregation="topk",
        plan=plan.as_dict(),
    )
    result = execute_axis_query(
        store,
        index,
        embedder,
        aq,
        scoring=scoring,
        capabilities=capabilities,
        threshold=threshold,
        explain=explain,
        retrieval=retrieval,
        extractor=extractor,
    )
    return [
        atom_to_row(hit.atom, score=hit.score, score_details=hit.score_details if explain else None)
        for hit in result.hits
    ]


def get_all(store: StateStore, *, filters: dict, top_k: int, as_of: str, include_expired: bool) -> list[dict]:
    """Mem0-shaped get_all: an axis-only AxisQuery (no plan, no scale constraint) in time order.
    Mem0 形状的 get_all：仅走轴的 AxisQuery（无规划、无尺度限制），按时间顺序。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        filters: Mem0 filter dict (scope keys required).
        filters：Mem0 过滤字典（需含范围键）。

        top_k: Maximum rows.
        top_k：最大行数。

        as_of: Validity instant.
        as_of：有效性时刻。

        include_expired: Drop the validity predicate (archived versions shown).
        include_expired：去掉有效性谓词（显示已归档版本）。

    Output / 输出：
        Rows (§4.1) without 'score', ordered time_anchor DESC, created_at DESC, atom_id ASC.
        不带 'score' 的行（§4.1），按 time_anchor 降序、created_at 降序、atom_id 升序。

    State effect / 状态影响：
        Read-only.
        只读。
    """
    aq = AxisQuery(
        filters=dict(filters or {}),
        text=None,
        limit=top_k,
        as_of=as_of,
        include_expired=include_expired,
        aggregation="list",
    )
    result = execute_axis_query(store, None, None, aq, scoring=_DEFAULT_SCORING, capabilities=store.capabilities)
    return [atom_to_row(hit.atom) for hit in result.hits]


def atom_to_row(
    atom: MemoryAtom,
    *,
    score: float | None = None,
    score_details: dict | None = None,
    anchor_ids: list[str] | None = None,
) -> dict:
    """The Mem0 row of an atom (§4.1) — a one-line wrapper over MemoryAtom.to_public_dict.
    原子的 Mem0 行（§4.1）——MemoryAtom.to_public_dict 的单行包装。

    Input / 输入：
        atom: The atom.
        atom：原子。

        score: Search score (included only when given).
        score：搜索分数（给定时才包含）。

        score_details: Explain block (included only when given).
        score_details：解释块（给定时才包含）。

        anchor_ids: Undisputed anchor list override.
        anchor_ids：无争议锚点列表覆盖值。

    Output / 输出：
        Dict in the exact §4.1 key order.
        严格按 §4.1 键序的字典。

    State effect / 状态影响：
        None.
        无。
    """
    return atom.to_public_dict(score=score, score_details=score_details, anchor_ids=anchor_ids)


__all__ = [
    "AXIS_ORDER",
    "CHANNELS",
    "FALLBACK",
    "GATE_CHANNEL",
    "Pool",
    "SCALE_FALLBACK",
    "atom_to_row",
    "count_axis_query",
    "execute_axis_query",
    "get_all",
    "recall_pool",
    "search",
]
