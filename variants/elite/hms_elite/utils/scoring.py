"""Search signals (semantic, keyword, entity, recency, support), their fusion and the explain block.
搜索信号（语义、关键词、实体、新近度、支持度）、其融合与解释块。
"""

# Purpose (EN): Deterministic scoring for the State Index hot path (CONTRACT §13.4 / DESIGN §8).
# 目的（中）：状态索引热路径的确定性打分（契约 §13.4 / 设计 §8）。

# Input (EN): Hydrated atoms or AtomLite scoring rows, the planner output (prepared once per query), index
#             similarities, entity degrees, anchor counts, exact ages, ScoringConfig.
# 输入（中）：已水合原子或 AtomLite 打分行、规划器输出（每次查询预处理一次）、索引相似度、实体度数、锚点计数、
#             精确年龄、ScoringConfig。

# Output (EN): Signals, (final, raw, max_possible) triples and score_details dicts with the Mem0 key order.
# 输出（中）：Signals、(final, raw, max_possible) 三元组以及按 Mem0 键序的 score_details 字典。

# State effect (EN): None (pure; the index is only read).
# 状态影响（中）：无（纯函数；仅读取索引）。

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from hms_elite.configs.base import ScoringConfig
from hms_elite.timeutil import age_days
from hms_elite.utils.text import cjk_tokens_mode, content_tokens, is_identifier, tokenize

SIGNAL_NAMES: tuple[str, ...] = ("semantic", "keyword", "entity", "recency", "support")
KEYWORD_BACKEND = "overlap"
_HUB_DAMPING = 0.001
_HUB_FREE_DEGREE = 3


@dataclass(frozen=True)
class Signals:
    """The five per-atom signals plus the tuple of active ones.
    每个原子的五个信号及活跃信号元组。
    """

    semantic: float
    keyword: float
    entity: float
    recency: float
    support: float
    active: tuple[str, ...]


def _plan_field(plan: Any, name: str) -> Any:
    """Read a plan field from a QueryPlan-like object or its as_dict() form.
    从 QueryPlan 类对象或其 as_dict() 形式读取字段。
    """
    if plan is None:
        return None
    if isinstance(plan, Mapping):
        return plan.get(name)
    return getattr(plan, name, None)


def _hint_key(hint: Any) -> str | None:
    """Entity key of a hint (EntityCandidate or its dict form).
    线索的实体键（EntityCandidate 或其字典形式）。
    """
    if isinstance(hint, Mapping):
        value = hint.get("key")
    else:
        value = getattr(hint, "key", None)
    return str(value) if value else None


def fusion_weights(scoring: ScoringConfig) -> dict[str, float]:
    """Signal name → fusion weight, derived once per query for fuse(weights=...).
    信号名 → 融合权重，每次查询派生一次供 fuse(weights=...) 使用。

    Input / 输入：
        scoring: ScoringConfig.
        scoring：ScoringConfig。

    Output / 输出：
        {'semantic', 'keyword', 'entity', 'recency', 'support'} → float.
        {'semantic', 'keyword', 'entity', 'recency', 'support'} → 浮点数。

    State effect / 状态影响：
        None.
        无。
    """
    return {
        "semantic": float(scoring.w_semantic),
        "keyword": float(scoring.w_keyword),
        "entity": float(scoring.w_entity),
        "recency": float(scoring.w_recency),
        "support": float(scoring.w_support),
    }


@lru_cache(maxsize=65536)
def _content_vocabulary(text: str, cjk: str = "bigram") -> frozenset[str]:
    """Content tokens of a text plus the basename of every path token (pure; memoized: pools repeat across queries;
    the CJK tokenizer mode is part of the key so two facades with different `text.cjk_tokens` never share entries).
    文本的内容词元加上每个路径词元的基名（纯函数；记忆化：池在多次查询间重复出现；CJK 分词模式是键的一部分，
    使 `text.cjk_tokens` 不同的两个门面绝不共享条目）。
    """
    tokens: set[str] = set()
    for token in content_tokens(text, cjk=cjk):
        tokens.add(token)
        if "/" in token:
            tokens.add(token.rsplit("/", 1)[-1])
    return frozenset(tokens)


@lru_cache(maxsize=65536)
def _key_tokens(key: str, cjk: str = "bigram") -> tuple[str, ...]:
    """The key itself plus its tokens (pure; memoized: the same entity keys recur across a pool; keyed by CJK mode).
    键本身加上其词元（纯函数；记忆化：同一批实体键在池内反复出现；按 CJK 模式分键）。
    """
    return (key, *tokenize(key, cjk=cjk))


def atom_keyword_tokens(atom: Any) -> set[str]:
    """Keyword vocabulary of an atom: content tokens (path basenames added), topic tags, entity keys and their tokens.
    原子的关键词词表：内容词元（补充路径基名）、主题标签、实体键及其词元。

    Input / 输入：
        atom: A hydrated MemoryAtom or a storage.AtomLite row (text, topic_tags, entity_ids).
        atom：已水合的 MemoryAtom 或 storage.AtomLite 行（text、topic_tags、entity_ids）。

    Output / 输出：
        Set of lowercase tokens.
        小写词元集合。

    State effect / 状态影响：
        None.
        无。
    """
    mode = cjk_tokens_mode()
    tokens: set[str] = set(_content_vocabulary(atom.text, mode))
    tokens.update(str(t).lower() for t in atom.topic_tags)
    for key in atom.entity_ids:
        tokens.update(_key_tokens(key, mode))
    return tokens


@dataclass(frozen=True)
class QueryTerms:
    """The query side of the keyword / entity signals, derived once per query instead of once per atom.
    关键词 / 实体信号的查询侧，每次查询派生一次而不是每个原子一次。

    `weights` keeps the unique content tokens in first-appearance order with their weights, so the overlap sum runs
    in exactly the order keyword_score uses (bit-identical fusion).
    `weights` 按首次出现顺序保存去重后的内容词元及其权重，因此重叠求和的顺序与 keyword_score 完全相同（融合逐位相同）。
    """

    weights: tuple[tuple[str, float], ...]
    total: float
    keys: frozenset[str]
    active: tuple[str, ...]


def prepare_query(plan: Any, scoring: ScoringConfig) -> QueryTerms:
    """Derive the QueryTerms of a plan (content tokens, identifier weights, entity keys, active signal names).
    由规划派生 QueryTerms（内容词元、标识符权重、实体键、活跃信号名）。

    Input / 输入：
        plan: QueryPlan or its as_dict() form (content_tokens, entity_hints).
        plan：QueryPlan 或其 as_dict() 形式（content_tokens、entity_hints）。

        scoring: ScoringConfig (identifier weight).
        scoring：ScoringConfig（标识符权重）。

    Output / 输出：
        QueryTerms.
        QueryTerms。

    State effect / 状态影响：
        None.
        无。
    """
    query_tokens = [str(t) for t in (_plan_field(plan, "content_tokens") or [])]
    keys = frozenset(k for k in (_hint_key(h) for h in (_plan_field(plan, "entity_hints") or [])) if k)
    unique = list(dict.fromkeys(query_tokens))
    weights = tuple((t, float(scoring.identifier_weight) if is_identifier(t) else 1.0) for t in unique)
    active = tuple(
        name
        for name in SIGNAL_NAMES
        if name in ("semantic", "recency", "support")
        or (name == "keyword" and bool(query_tokens))
        or (name == "entity" and bool(keys))
    )
    return QueryTerms(weights=weights, total=sum(w for _t, w in weights), keys=keys, active=active)


def _weighted_overlap(weights: tuple[tuple[str, float], ...], total: float, atom_tokens: set[str]) -> float:
    """Σ_{t∈Q∩A} w(t) / Σ_{t∈Q} w(t) in query order; 0.0 for an empty or zero-weight query.
    按查询顺序的 Σ_{t∈Q∩A} w(t) / Σ_{t∈Q} w(t)；空查询或零权重时为 0.0。
    """
    if not weights or total <= 0:
        return 0.0
    matched = sum(w for t, w in weights if t in atom_tokens)
    return matched / total


def semantic_scores(index: Any, query_vector: list[float] | None, candidate_ids: Iterable[str]) -> dict[str, float]:
    """Cosine similarity of the query against every candidate (max(0, cos)); missing vectors score 0.0.
    查询对每个候选的余弦相似度（max(0, cos)）；缺失向量得 0.0。

    Input / 输入：
        index: An IndexBase provider (None → every score 0.0).
        index：IndexBase 提供者（None → 全部 0.0）。

        query_vector: The embedded query (None → every score 0.0).
        query_vector：查询嵌入（None → 全部 0.0）。

        candidate_ids: Atom ids to score.
        candidate_ids：待打分的原子 id。

    Output / 输出：
        {atom_id: score} for every candidate id.
        每个候选 id 的 {atom_id: score}。

    State effect / 状态影响：
        None (reads the index).
        无（读取索引）。
    """
    ids = list(dict.fromkeys(candidate_ids))
    scores = {atom_id: 0.0 for atom_id in ids}
    if index is None or query_vector is None or not ids:
        return scores
    for atom_id, score in index.search(list(query_vector), candidate_ids=ids, top_k=len(ids)):
        if atom_id in scores:
            scores[atom_id] = max(0.0, float(score))
    return scores


def keyword_score(query_tokens: list[str], atom_tokens: set[str], *, identifier_weight: float) -> float:
    """Weighted token overlap Σ_{t∈Q∩A} w(t) / Σ_{t∈Q} w(t) with identifiers weighted identifier_weight.
    加权词元重叠 Σ_{t∈Q∩A} w(t) / Σ_{t∈Q} w(t)，标识符权重为 identifier_weight。

    Input / 输入：
        query_tokens: Content tokens of the query (duplicates ignored).
        query_tokens：查询的内容词元（忽略重复）。

        atom_tokens: The atom's keyword vocabulary (atom_keyword_tokens).
        atom_tokens：原子的关键词词表（atom_keyword_tokens）。

        identifier_weight: Weight of identifier tokens (plain tokens weigh 1.0).
        identifier_weight：标识符词元的权重（普通词元为 1.0）。

    Output / 输出：
        Score in [0, 1]; 0.0 for an empty query.
        [0, 1] 内的分数；空查询为 0.0。

    State effect / 状态影响：
        None.
        无。
    """
    unique = list(dict.fromkeys(query_tokens))
    weights = tuple((t, float(identifier_weight) if is_identifier(t) else 1.0) for t in unique)
    return _weighted_overlap(weights, sum(w for _t, w in weights), atom_tokens)


def entity_score(query_keys: set[str], atom_keys: set[str], *, degrees: dict[str, int]) -> float:
    """Entity overlap |Q∩A|/|Q| damped by 1/(1 + 0.001·max(0, deg−3)²) with deg the max degree among matches.
    实体重叠 |Q∩A|/|Q|，乘以 1/(1 + 0.001·max(0, deg−3)²)，deg 为匹配实体的最大度数。

    Input / 输入：
        query_keys: Entity keys mentioned by the query.
        query_keys：查询提及的实体键。

        atom_keys: Entity keys linked to the atom.
        atom_keys：链接到原子的实体键。

        degrees: Entity key → atom_entity link count.
        degrees：实体键 → atom_entity 链接数。

    Output / 输出：
        Score in [0, 1]; 0.0 for an empty query set.
        [0, 1] 内的分数；查询集合为空时为 0.0。

    State effect / 状态影响：
        None.
        无。
    """
    if not query_keys:
        return 0.0
    matched = set(query_keys) & set(atom_keys)
    if not matched:
        return 0.0
    degree = max(int(degrees.get(key, 0)) for key in matched)
    damping = 1.0 / (1.0 + _HUB_DAMPING * max(0, degree - _HUB_FREE_DEGREE) ** 2)
    return (len(matched) / len(query_keys)) * damping


def recency_score(t: str | None, *, as_of: str, half_life_days: float) -> float:
    """Exponential decay exp(−ln2 · age_days / half_life); None → 0.0.
    指数衰减 exp(−ln2 · age_days / half_life)；None → 0.0。

    Input / 输入：
        t: The atom instant (time_anchor or created_at).
        t：原子时刻（time_anchor 或 created_at）。

        as_of: Reference instant.
        as_of：参考时刻。

        half_life_days: Half-life in days (<= 0 → 0.0).
        half_life_days：半衰期天数（<= 0 → 0.0）。

    Output / 输出：
        Score in (0, 1].
        (0, 1] 内的分数。

    State effect / 状态影响：
        None.
        无。
    """
    if t is None or half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2.0) * age_days(t, as_of) / float(half_life_days))


def recency_from_age_us(age_us: int, *, half_life_days: float) -> float:
    """recency_score from an exact integer microsecond age (storage.AtomLite.age_us), bit-identical to recency_score.
    由精确整数微秒年龄（storage.AtomLite.age_us）计算 recency_score，与 recency_score 逐位相同。

    Input / 输入：
        age_us: (as_of − instant) in whole microseconds (negative → age 0).
        age_us：(as_of − 时刻) 的整数微秒（负值 → 年龄 0）。

        half_life_days: Half-life in days (<= 0 → 0.0).
        half_life_days：半衰期天数（<= 0 → 0.0）。

    Output / 输出：
        Score in (0, 1].
        (0, 1] 内的分数。

    State effect / 状态影响：
        None.
        无。
    """
    if half_life_days <= 0:
        return 0.0
    # timedelta.total_seconds() is `micros / 10**6`; age_days then divides by 86400.0 — the same two operations.
    # timedelta.total_seconds() 即 `micros / 10**6`；age_days 再除以 86400.0——完全相同的两步运算。
    age = max(0.0, int(age_us) / 10**6 / 86400.0)
    return math.exp(-math.log(2.0) * age / float(half_life_days))


def support_score(n_anchors: int) -> float:
    """tanh(n / 3) over the number of undisputed anchors.
    对无争议锚点数取 tanh(n / 3)。

    Input / 输入：
        n_anchors: Anchor count.
        n_anchors：锚点数。

    Output / 输出：
        Score in [0, 1).
        [0, 1) 内的分数。

    State effect / 状态影响：
        None.
        无。
    """
    return math.tanh(max(0, int(n_anchors)) / 3.0)


def compute_signals(
    *,
    atom: Any,
    semantic: float,
    plan: Any,
    as_of: str,
    scoring: ScoringConfig,
    degrees: dict[str, int],
    n_anchors: int,
    prepared: QueryTerms | None = None,
    age_us: int | None = None,
) -> Signals:
    """Compute the five signals of one atom; keyword/entity are active only when the plan carries tokens/hints.
    计算一个原子的五个信号；仅当规划带有词元/线索时 keyword/entity 才活跃。

    Input / 输入：
        atom: A hydrated MemoryAtom or a storage.AtomLite row (text, topic_tags, entity_ids, time_anchor, created_at).
        atom：已水合的 MemoryAtom 或 storage.AtomLite 行（text、topic_tags、entity_ids、time_anchor、created_at）。

        semantic: The atom's semantic similarity (semantic_scores).
        semantic：原子的语义相似度（semantic_scores）。

        plan: QueryPlan or its as_dict() form (content_tokens, entity_hints).
        plan：QueryPlan 或其 as_dict() 形式（content_tokens、entity_hints）。

        as_of: Reference instant for recency.
        as_of：新近度的参考时刻。

        scoring: ScoringConfig (half-life, identifier weight).
        scoring：ScoringConfig（半衰期、标识符权重）。

        degrees: Entity key → degree for the query's entity keys.
        degrees：查询实体键 → 度数。

        n_anchors: Undisputed anchor count of the atom.
        n_anchors：原子的无争议锚点数。

        prepared: prepare_query(plan, scoring), when the caller scores a whole pool (None → derived here).
        prepared：prepare_query(plan, scoring)，调用方给整个池打分时传入（None → 在此派生）。

        age_us: Exact microsecond age of the atom instant (storage.AtomLite.age_us); None → recency_score over the strings.
        age_us：原子时刻的精确微秒年龄（storage.AtomLite.age_us）；None → 对字符串调用 recency_score。

    Output / 输出：
        Signals.
        Signals。

    State effect / 状态影响：
        None.
        无。
    """
    terms = prepared if prepared is not None else prepare_query(plan, scoring)
    keyword = _weighted_overlap(terms.weights, terms.total, atom_keyword_tokens(atom)) if "keyword" in terms.active else 0.0
    entity = entity_score(terms.keys, set(atom.entity_ids), degrees=degrees) if terms.keys else 0.0
    if age_us is not None:
        recency = recency_from_age_us(age_us, half_life_days=scoring.half_life_days)
    else:
        recency = recency_score(atom.time_anchor or atom.created_at, as_of=as_of, half_life_days=scoring.half_life_days)
    support = support_score(n_anchors)
    return Signals(
        semantic=max(0.0, float(semantic)),
        keyword=keyword,
        entity=entity,
        recency=recency,
        support=support,
        active=terms.active,
    )


def fuse(
    signals: Signals, scoring: ScoringConfig, *, weights: Mapping[str, float] | None = None
) -> tuple[float, float, float]:
    """Weighted fusion over the active signals: (final, raw, max_possible) with final = raw / max_possible.
    对活跃信号加权融合：(final, raw, max_possible)，final = raw / max_possible。

    Input / 输入：
        signals: The atom's Signals.
        signals：原子的 Signals。

        scoring: ScoringConfig weights.
        scoring：ScoringConfig 权重。

        weights: fusion_weights(scoring), when the caller fuses a whole pool (None → derived here).
        weights：fusion_weights(scoring)，调用方融合整个池时传入（None → 在此派生）。

    Output / 输出：
        (final, raw, max_possible); final is 0.0 when no weight is active.
        (final, raw, max_possible)；无活跃权重时 final 为 0.0。

    State effect / 状态影响：
        None.
        无。
    """
    if weights is None:
        weights = fusion_weights(scoring)
    raw = 0.0
    max_possible = 0.0
    for name in signals.active:
        weight = weights.get(name, 0.0)
        raw += weight * float(getattr(signals, name))
        max_possible += weight
    final = raw / max_possible if max_possible > 0 else 0.0
    return final, raw, max_possible


def score_details(
    signals: Signals,
    final: float,
    raw: float,
    max_possible: float,
    *,
    threshold: float,
    plan: Any,
    primary_axis: str,
    scope: dict[str, Any] | None = None,
    as_of: str | None = None,
    recalled_by: list[str] | None = None,
    pool_size: int | None = None,
    channel_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The explain block in the exact CONTRACT §13.4 key order (Mem0 keys first, HMS keys, then the recall keys).
    严格按契约 §13.4 键序的解释块（先 Mem0 键，后 HMS 键，最后是召回键）。

    Input / 输入：
        signals: The atom's Signals.
        signals：原子的 Signals。

        final: Fused score.
        final：融合分数。

        raw: Σ w·s over the active signals.
        raw：活跃信号的 Σ w·s。

        max_possible: Σ w over the active signals.
        max_possible：活跃信号的 Σ w。

        threshold: The threshold applied to final.
        threshold：施加于 final 的阈值。

        plan: QueryPlan or its as_dict() form (gates and route).
        plan：QueryPlan 或其 as_dict() 形式（门控与路由）。

        primary_axis: 'time' | 'entity' | 'type' | 'semantic' | 'fallback'.
        primary_axis：'time' | 'entity' | 'type' | 'semantic' | 'fallback'。

        scope: Scope filter dict recorded under axis_gates.scope.
        scope：记录于 axis_gates.scope 的范围过滤字典。

        as_of: Validity instant recorded under axis_gates.validity.
        as_of：记录于 axis_gates.validity 的有效性时刻。

        recalled_by: The recall channels that put this atom into the pool (None → []).
        recalled_by：把该原子送入池的召回通道（None → []）。

        pool_size: Size of the scored pool (None → 0).
        pool_size：被打分的池大小（None → 0）。

        channel_sizes: Channel name → number of ids it recalled (None → {}).
        channel_sizes：通道名 → 其召回的 id 数（None → {}）。

    Output / 输出：
        Dict with keys semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold,
        recency_score, support_score, keyword_backend, axis_gates, primary_axis, route, signals, recalled_by,
        pool_size, channel_sizes (the first 14 keys are unchanged; the three recall keys are appended).
        含键 semantic_score、bm25_score、entity_boost、raw_score、max_possible_score、final_score、threshold、
        recency_score、support_score、keyword_backend、axis_gates、primary_axis、route、signals、recalled_by、
        pool_size、channel_sizes 的字典（前 14 个键不变；三个召回键追加在后）。

    State effect / 状态影响：
        None.
        无。
    """
    time_gate = _plan_field(plan, "hard_time_gate")
    entity_gate = _plan_field(plan, "hard_entity_gate")
    type_gate = _plan_field(plan, "hard_type_gate")
    return {
        "semantic_score": float(signals.semantic),
        "bm25_score": float(signals.keyword),
        "entity_boost": float(signals.entity),
        "raw_score": float(raw),
        "max_possible_score": float(max_possible),
        "final_score": float(final),
        "threshold": float(threshold),
        "recency_score": float(signals.recency),
        "support_score": float(signals.support),
        "keyword_backend": KEYWORD_BACKEND,
        "axis_gates": {
            "time": list(time_gate) if time_gate else None,
            "entity": list(entity_gate) if entity_gate is not None else None,
            "obs_type": list(type_gate[1]) if type_gate else None,
            "scope": dict(scope or {}),
            "validity": as_of,
        },
        "primary_axis": primary_axis,
        "route": _plan_field(plan, "route") or "recall",
        "signals": list(signals.active),
        "recalled_by": list(recalled_by or []),
        "pool_size": int(pool_size or 0),
        "channel_sizes": {str(k): int(v) for k, v in (channel_sizes or {}).items()},
    }


__all__ = [
    "KEYWORD_BACKEND",
    "SIGNAL_NAMES",
    "QueryTerms",
    "Signals",
    "atom_keyword_tokens",
    "compute_signals",
    "entity_score",
    "fuse",
    "fusion_weights",
    "keyword_score",
    "prepare_query",
    "recency_from_age_us",
    "recency_score",
    "score_details",
    "semantic_scores",
    "support_score",
]
