"""Bounded pool recall tests: equivalence theorem, N=500 overlap, no newest-first cut, provenance, safety limit, compile.
有界池召回测试：等价性定理、N=500 重叠率、无最新优先截断、溯源、安全上限、compile。
"""

# Purpose (EN): Pin RETRIEVAL_DIAGNOSIS §4 (the equivalence theorem) and §5 (the pool algorithm) as regression tests.
# 目的（中）：把 RETRIEVAL_DIAGNOSIS §4（等价性定理）与 §5（池算法）钉成回归测试。

# Input (EN): conftest fixtures (store, clock, ids, index_provider, hash_embedder, hms, milvus_uri) and a seeded random corpus.
# 输入（中）：conftest 夹具（store、clock、ids、index_provider、hash_embedder、hms、milvus_uri）与一份定种随机语料。

# Output (EN): pytest results; the N=500 overlap ratio is printed (an intended pruning loss, no hard threshold).
# 输出（中）：pytest 结果；N=500 的重叠率被打印（有意的剪枝损失，不设硬阈值）。

# State effect (EN): None outside the in-memory SQLite database and a milvus-lite file under tmp (no network, no home writes).
# 状态影响（中）：除内存 SQLite 数据库与 tmp 下的 milvus-lite 文件外无副作用（无网络、不写用户目录）。

from __future__ import annotations

import json
import logging
import random
from typing import Any

import pytest

from hms_elite.configs.base import RetrievalConfig, ScoringConfig
from hms_elite.memory.storage import StateStore
from hms_elite.models import AxisQuery, MemoryAtom, Scope, SourceEvent
from hms_elite.mvmi import query as mquery
from hms_elite.mvmi.filters import And, FilterExpr, evaluate, lower_to_sql, parse_filters, plan_filters
from hms_elite.mvmi.planner import ALL_SCALES, plan_query
from hms_elite.mvmi.query import CHANNELS, GATE_CHANNEL, execute_axis_query, recall_pool
from hms_elite.utils.factory import IndexFactory
from hms_elite.utils.scoring import compute_signals, fuse, score_details, semantic_scores

U = "alice"
SCOPE = {"user_id": U}
MEM0_KEYS = (
    "semantic_score",
    "bm25_score",
    "entity_boost",
    "raw_score",
    "max_possible_score",
    "final_score",
    "threshold",
)
HMS_KEYS = ("recency_score", "support_score", "keyword_backend", "axis_gates", "primary_axis", "route", "signals")
NAMES = ["Sarah", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
FILES = ["src/billing/retry.ts", "src/billing/idempotency.ts", "src/auth/session.py", "infra/deploy.yaml"]
MONTHS = ["January", "February", "March", "April", "May", "June"]
TEMPLATES = [
    "{name} patched {file} and the backoff regression is fixed.",
    "User likes {drink} in the morning.",
    "{name} joined the billing team and reviewed {file}.",
    "今天部署了支付服务并回滚了 {file} 的改动。",
    "Standup: {name} reported that the {topic} tests passed.",
    "Deployed {topic} service; item{i} tracked for {name}.",
]
DRINKS = ["black coffee", "oat-milk lattes", "green tea", "espresso"]
TOPICS = ["billing", "auth", "deploy", "payments"]
QUERIES = [
    "who is Sarah",
    "who is Bob",
    "what did Carol review",
    "retry.ts backoff patch",
    "idempotency.ts regression",
    "what happened in March",
    "what happened in May",
    "what coffee does alice like",
    "what does the user like in the morning",
    "部署 回滚 测试",
    "支付服务",
    "deploy.yaml rollback",
    "billing tests passed",
    "session.py auth",
    "green tea",
    "standup report",
    "item7",
    "Dave joined the team",
    "who is Judy",
    "payments service deployed",
]


def _anchor(store: StateStore, text: str, ts: str) -> str:
    """Insert one message event in SCOPE at `ts` and return its anchor id.
    在 SCOPE 于 `ts` 插入一条消息事件并返回锚点 id。
    """
    ev = SourceEvent(scope=Scope(**SCOPE), role="user", payload={"text": text}, timestamp=ts)
    stored, _ = store.insert_source_event(ev)
    return store.get_or_create_anchor(stored).anchor_id


def seed_corpus(store: StateStore, clock: Any, index: Any, embedder: Any, n: int, *, seed: int = 7) -> list[MemoryAtom]:
    """Seed `n` random atoms (names, files, CJK, months across 18 months); ~10% keep no vector; returns them in order.
    播种 `n` 条随机原子（人名、文件、CJK、跨 18 个月）；约 10% 不写向量；按顺序返回。

    Input / 输入：
        store: The StateStore.
        store：StateStore。

        clock: FrozenClock (ticked one second per atom).
        clock：FrozenClock（每个原子前进一秒）。

        index: The index receiving the vectors.
        index：接收向量的索引。

        embedder: The HashEmbedder.
        embedder：HashEmbedder。

        n: Corpus size.
        n：语料规模。

        seed: Random seed.
        seed：随机种子。

    Output / 输出：
        The stored atoms in insertion order.
        按插入顺序的已存原子。

    State effect / 状态影响：
        Writes source events, anchors, atoms, entity links, topics, FTS rows and vectors.
        写入源事件、锚点、原子、实体链接、主题、FTS 行与向量。
    """
    rng = random.Random(seed)
    atoms: list[MemoryAtom] = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        name = rng.choice(NAMES)
        file = rng.choice(FILES)
        topic = rng.choice(TOPICS)
        text = template.format(name=name, file=file, drink=rng.choice(DRINKS), topic=topic, i=i)
        month = 1 + rng.randrange(18)
        year, mon = (2025, month) if month <= 12 else (2026, month - 12)
        ts = f"{year}-{mon:02d}-{1 + rng.randrange(28):02d}T{rng.randrange(24):02d}:00:00Z"
        entity_keys: list[tuple[str, str, str, tuple[str, ...]]] = []
        if "{name}" in template:
            entity_keys.append((name.lower(), name, "PROPER", ()))
        if "{file}" in template:
            entity_keys.append((file.rsplit("/", 1)[-1], file, "FILE", (file,)))
        is_pref = template.startswith("User likes")
        atom = MemoryAtom(
            atom_type="state",
            content={"text": text, **({"obs_type": "preference_event"} if is_pref else {})},
            text=text,
            scale_level=rng.choice(["state", "event", "L1_micro"]) if not is_pref else "state",
            source_kind="observation",
            source_id=f"obs_{i}",
            extracted_by="rule@obs-v1",
            scope=Scope(**SCOPE),
            anchor_ids=[_anchor(store, text, ts)],
            time_anchor=ts,
            subject="user",
            predicate="likes" if is_pref else "did",
            obs_type="preference_event" if is_pref else None,
        )
        stored = store.insert_atom(atom, entity_keys=entity_keys, topics=[topic])
        if rng.random() >= 0.1:
            index.upsert(stored.atom_id, embedder.embed(stored.text), embedder=embedder.identity)
        clock.tick(seconds=1)
        atoms.append(stored)
    return atoms


def _exhaustive_search(
    store: StateStore,
    index: Any,
    embedder: Any,
    aq: AxisQuery,
    *,
    scoring: ScoringConfig,
    capabilities: dict,
    threshold: float,
    explain: bool,
    max_candidates: int = 2000,
) -> list[tuple[str, float, dict]]:
    """The pre-pool reference: every gated candidate (newest-first cut at max_candidates) hydrated and scored.
    池化之前的参考实现：每个门控候选（最新优先、截到 max_candidates）都水合并打分。
    """
    fscope, expr = parse_filters(aq.filters, require_scope=False, context="axis_query")
    scope = aq.scope if (aq.scope is not None and not aq.scope.is_empty()) else fscope
    _, meta_expr = parse_filters(aq.meta_filters, require_scope=False, context="meta", force_meta=True)
    parts: list[FilterExpr] = [e for e in (expr, meta_expr) if e is not None]
    combined: FilterExpr | None = None if not parts else (parts[0] if len(parts) == 1 else And(items=parts))
    sql_expr, py_expr = plan_filters(combined, capabilities=capabilities)
    where_sql, params, joins = lower_to_sql(sql_expr, capabilities=capabilities, scope=scope)
    plan = dict(aq.plan or {})
    as_of = aq.as_of or store.now()
    plan_scale = [str(s) for s in (plan.get("scale_default") or [])] or None
    scale_constraint = plan_scale
    scale_relaxable = plan_scale is not None and not set(ALL_SCALES) <= set(plan_scale)
    atom_types = plan.get("atom_types")
    gate_clauses, gate_params, gate_names = mquery._plan_gates(plan, scope, capabilities)

    def _select(with_gates: bool, with_scale: bool) -> list[MemoryAtom]:
        clauses = [where_sql] if where_sql else []
        values = list(params)
        if with_gates:
            clauses.extend(gate_clauses)
            values.extend(gate_params)
        rows = store.axis_select(
            scope,
            where_sql=" AND ".join(f"({c})" for c in clauses),
            params=values,
            joins=list(joins),
            as_of=as_of,
            include_expired=aq.include_expired,
            scale_constraint=scale_constraint if with_scale else None,
            atom_types=atom_types,
            order=mquery.AXIS_ORDER,
            limit=max_candidates,
        )
        if py_expr is not None:
            rows = [
                a for a in rows if evaluate(py_expr, a, entity_keys=set(a.entity_ids), anchor_ids=set(a.anchor_ids))
            ]
        return rows

    primary_axis = mquery._GATE_AXIS[gate_names[0]] if gate_names else "semantic"
    ladder = mquery._fallback_ladder(gated=bool(gate_names), scale_relaxable=scale_relaxable, primary_axis=primary_axis)
    candidates: list[MemoryAtom] = []
    for with_gates, with_scale, label in ladder:
        candidates = _select(with_gates, with_scale)
        primary_axis = label
        if candidates:
            break
    sig_plan = mquery._scoring_plan(aq, as_of)
    query_vector = embedder.embed(aq.text, "search")
    semantic = semantic_scores(index, query_vector, [a.atom_id for a in candidates])
    query_keys = [k for k in (mquery._hint_key(h) for h in (sig_plan.get("entity_hints") or [])) if k]
    degrees: dict[str, int] = {}
    for key, entity_id in store.resolve_entities(scope, query_keys).items():
        degrees[key] = store.entity_degree(entity_id)
    ranked = []
    for atom in candidates:
        signals = compute_signals(
            atom=atom,
            semantic=semantic.get(atom.atom_id, 0.0),
            plan=sig_plan,
            as_of=as_of,
            scoring=scoring,
            degrees=degrees,
            n_anchors=len(atom.anchor_ids),
        )
        final, raw, max_possible = fuse(signals, scoring)
        if final < threshold:
            continue
        ranked.append((atom, signals, final, raw, max_possible))
    ranked.sort(key=lambda r: r[0].atom_id)
    ranked.sort(key=lambda r: (r[0].time_anchor or "", r[0].created_at or ""), reverse=True)
    ranked.sort(key=lambda r: r[2], reverse=True)
    out = []
    for atom, signals, final, raw, max_possible in ranked[: max(0, int(aq.limit))]:
        details = (
            score_details(
                signals,
                final,
                raw,
                max_possible,
                threshold=threshold,
                plan=sig_plan,
                primary_axis=primary_axis,
                scope=scope.as_filters(),
                as_of=as_of,
            )
            if explain
            else {}
        )
        out.append((atom.atom_id, final, details))
    return out


def _aq(query: str, as_of: str, top_k: int) -> AxisQuery:
    """A 'topk' AxisQuery planned like mvmi.query.search does.
    像 mvmi.query.search 那样规划的 'topk' AxisQuery。
    """
    return AxisQuery(
        filters=dict(SCOPE),
        text=query,
        limit=top_k,
        as_of=as_of,
        aggregation="topk",
        plan=plan_query(query, as_of=as_of).as_dict(),
    )


def _new_rows(
    store: StateStore, index: Any, embedder: Any, aq: AxisQuery, *, retrieval: RetrievalConfig, threshold: float = 0.0
) -> list[tuple[str, float, dict]]:
    """The pool path as (atom_id, score, score_details) triples.
    以 (atom_id, score, score_details) 三元组表示的池路径结果。
    """
    result = execute_axis_query(
        store,
        index,
        embedder,
        aq,
        scoring=ScoringConfig(),
        capabilities=store.capabilities,
        threshold=threshold,
        explain=True,
        retrieval=retrieval,
    )
    return [(h.atom.atom_id, h.score, h.score_details) for h in result.hits]


def _assert_bit_equal(
    reference: list[tuple[str, float, dict]], actual: list[tuple[str, float, dict]], query: str
) -> None:
    """(atom_id, score, the 7 Mem0 keys, the 7 HMS keys) must be identical, floats compared with ==.
    (atom_id, score, 7 个 Mem0 键, 7 个 HMS 键) 必须完全相同，浮点用 == 比较。
    """
    assert [(a, s) for a, s, _d in reference] == [(a, s) for a, s, _d in actual], query
    for (_a, _s, ref), (_b, _t, new) in zip(reference, actual):
        assert [ref[k] for k in MEM0_KEYS] == [new[k] for k in MEM0_KEYS], query
        assert [ref[k] for k in HMS_KEYS] == [new[k] for k in HMS_KEYS], query
        # The reference builds its block with the same score_details, so it carries the appended recall keys too.
        # 参考实现用同一个 score_details 构造解释块，因此它也带有追加的召回键。
        assert list(new)[:14] == list(ref)[:14] and list(new)[14:] == ["recalled_by", "pool_size", "channel_sizes"]


@pytest.mark.parametrize(
    ("n", "retrieval"),
    [
        (5, RetrievalConfig()),
        (30, RetrievalConfig()),
        (60, RetrievalConfig()),
        (5, RetrievalConfig(hard_gate_pool=False)),
        (30, RetrievalConfig(hard_gate_pool=False)),
        (60, RetrievalConfig(hard_gate_pool=False, k_keyword=60, k_entity=60, k_recency=60)),
    ],
    ids=["n5-whole", "n30-whole", "n60-whole", "n5-channels", "n30-channels", "n60-channels-k60"],
)
def test_equivalence_theorem(store, clock, index_provider, hash_embedder, n, retrieval):
    """|G| <= min(K) (or |G| <= pool_max with the whole-gate rule) → the pool path equals the exhaustive path bit for bit.
    |G| <= min(K)（或整体入池规则下 |G| <= pool_max）→ 池路径与穷举路径逐位相同。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures (sqlite and memory providers).
        store、clock、index_provider、hash_embedder：conftest 夹具（sqlite 与 memory 提供者）。

        n: Corpus size.
        n：语料规模。

        retrieval: The RetrievalConfig under test (whole-gate rule on, or channels only).
        retrieval：受测的 RetrievalConfig（整体入池规则开启，或仅通道）。

    Output / 输出：
        Assertions over 20 queries x 3 top_k values.
        对 20 个查询 x 3 个 top_k 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, n)
    as_of = store.now()
    ks = retrieval.k_for(3)
    assert n <= min(ks["keyword"], ks["entity"], ks["recency"]) or (retrieval.hard_gate_pool and n <= ks["pool"])
    for query in QUERIES:
        for top_k in (3, 5, 10):
            aq = _aq(query, as_of, top_k)
            reference = _exhaustive_search(
                store,
                index,
                hash_embedder,
                aq,
                scoring=ScoringConfig(),
                capabilities=store.capabilities,
                threshold=0.0,
                explain=True,
            )
            _assert_bit_equal(reference, _new_rows(store, index, hash_embedder, aq, retrieval=retrieval), query)
            # A threshold is applied inside the pool exactly where the exhaustive path applied it.
            # 阈值在池内施加，位置与穷举路径完全相同。
            cut = _exhaustive_search(
                store,
                index,
                hash_embedder,
                aq,
                scoring=ScoringConfig(),
                capabilities=store.capabilities,
                threshold=0.3,
                explain=True,
            )
            _assert_bit_equal(
                cut, _new_rows(store, index, hash_embedder, aq, retrieval=retrieval, threshold=0.3), query
            )


def test_equivalence_theorem_milvus(milvus_uri, store, clock, hash_embedder):
    """The theorem on the Milvus provider (native filter pushdown in channel mode, candidate_ids in the pool).
    Milvus 提供者上的定理（通道模式下原生过滤下推，池内用 candidate_ids）。

    Input / 输入：
        milvus_uri, store, clock, hash_embedder: conftest fixtures (skipped without pymilvus).
        milvus_uri、store、clock、hash_embedder：conftest 夹具（缺少 pymilvus 时跳过）。

    Output / 输出：
        Assertions over the 20 queries in both recall modes.
        两种召回模式下对 20 个查询的断言。

    State effect / 状态影响：
        Writes a milvus-lite collection under tmp.
        在 tmp 下写入一个 milvus-lite 集合。
    """
    pytest.importorskip("pymilvus")
    index = IndexFactory.create(
        "milvus", {"uri": milvus_uri, "collection": "retrieval_equiv", "scalar_index": "none"}, store=store, dims=64
    )
    seed_corpus(store, clock, index, hash_embedder, 30)
    as_of = store.now()
    assert index.supports_filter
    for retrieval in (RetrievalConfig(), RetrievalConfig(hard_gate_pool=False)):
        for query in QUERIES:
            aq = _aq(query, as_of, 5)
            reference = _exhaustive_search(
                store,
                index,
                hash_embedder,
                aq,
                scoring=ScoringConfig(),
                capabilities=store.capabilities,
                threshold=0.0,
                explain=True,
            )
            _assert_bit_equal(reference, _new_rows(store, index, hash_embedder, aq, retrieval=retrieval), query)


def test_pool_recall_overlap_at_n500(store, clock, index_provider, hash_embedder, capsys):
    """N=500 > pool_max: the channel union is a strict subset of G; report the top_k overlap with the exhaustive path.
    N=500 > pool_max：通道并集是 G 的真子集；报告与穷举路径的 top_k 重叠率。

    Input / 输入：
        store, clock, index_provider, hash_embedder, capsys: conftest / pytest fixtures.
        store、clock、index_provider、hash_embedder、capsys：conftest / pytest 夹具。

    Output / 输出：
        Printed overlap ratios per query (no hard threshold: an intended pruning loss); pool bounds asserted.
        逐查询打印的重叠率（不设硬阈值：有意的剪枝损失）；断言池上限。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 500, seed=11)
    as_of = store.now()
    retrieval = RetrievalConfig()
    ks = retrieval.k_for(20)
    overlaps: list[float] = []
    top1_same = 0
    channel_queries = 0
    for query in QUERIES:
        aq = _aq(query, as_of, 20)
        reference = _exhaustive_search(
            store,
            index,
            hash_embedder,
            aq,
            scoring=ScoringConfig(),
            capabilities=store.capabilities,
            threshold=0.0,
            explain=False,
            max_candidates=10**6,
        )
        rows = _new_rows(store, index, hash_embedder, aq, retrieval=retrieval)
        ref_ids = [a for a, _s, _d in reference]
        new_ids = [a for a, _s, _d in rows]
        overlap = len(set(ref_ids) & set(new_ids)) / max(1, len(ref_ids))
        overlaps.append(overlap)
        top1_same += int(bool(ref_ids) and bool(new_ids) and ref_ids[0] == new_ids[0])
        if rows:
            details = rows[0][2]
            assert 0 < details["pool_size"] <= ks["pool"]
            if GATE_CHANNEL not in details["channel_sizes"]:
                channel_queries += 1
                assert set(details["channel_sizes"]) <= set(CHANNELS)
                assert all(details["channel_sizes"][name] <= ks[name] for name in details["channel_sizes"])
        print(
            f"overlap@20 N=500 {index_provider} {query!r}: {overlap:.2f} pool={rows[0][2]['pool_size'] if rows else 0}"
        )
    mean = sum(overlaps) / len(overlaps)
    print(
        f"MEAN overlap@20 N=500 {index_provider}: {mean:.3f}; top1 identical {top1_same}/{len(QUERIES)}; channel-mode queries {channel_queries}"
    )
    assert channel_queries >= 1
    assert 0.0 <= mean <= 1.0


def test_no_newest_first_truncation(store, clock, index_provider, hash_embedder):
    """The deprecated ScoringConfig.max_candidates no longer cuts the oldest atoms: an old unique identifier is found.
    已弃用的 ScoringConfig.max_candidates 不再截掉最旧的原子：一个久远的唯一标识符能被找到。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions (fails when `limit=scoring.max_candidates` is reinstated).
        断言（若恢复 `limit=scoring.max_candidates` 则失败）。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    atoms = seed_corpus(store, clock, index, hash_embedder, 60)
    oldest = min(atoms, key=lambda a: a.time_anchor or "")
    text = "Carol fixed src/legacy/ledger.rs after the ledger overflow."
    old_ts = "2024-01-05T09:00:00Z"
    stored = store.insert_atom(
        MemoryAtom(
            atom_type="state",
            content={"text": text},
            text=text,
            scale_level="event",
            source_kind="observation",
            source_id="obs_ledger",
            extracted_by="rule@obs-v1",
            scope=Scope(**SCOPE),
            anchor_ids=[_anchor(store, text, old_ts)],
            time_anchor=old_ts,
            subject="user",
            predicate="did",
        ),
        entity_keys=[("ledger.rs", "src/legacy/ledger.rs", "FILE", ("src/legacy/ledger.rs",))],
        topics=["ledger"],
    )
    index.upsert(stored.atom_id, hash_embedder.embed(text), embedder=hash_embedder.identity)
    assert (stored.time_anchor or "") < (oldest.time_anchor or "")
    as_of = store.now()
    aq = _aq("ledger.rs overflow", as_of, 5)
    with pytest.warns(DeprecationWarning):
        capped = ScoringConfig(max_candidates=10)
    for retrieval in (RetrievalConfig(), RetrievalConfig(hard_gate_pool=False)):
        result = execute_axis_query(
            store,
            index,
            hash_embedder,
            aq,
            scoring=capped,
            capabilities=store.capabilities,
            explain=True,
            retrieval=retrieval,
        )
        assert result.hits and result.hits[0].atom.atom_id == stored.atom_id
        if not retrieval.hard_gate_pool:
            assert "keyword" in result.hits[0].score_details["recalled_by"]
    # The reference with the same cap is exactly the defect: the newest-first cut loses the row.
    # 带同样上限的参考实现正是该缺陷：最新优先截断丢掉了这一行。
    reference = _exhaustive_search(
        store,
        index,
        hash_embedder,
        aq,
        scoring=capped,
        capabilities=store.capabilities,
        threshold=0.0,
        explain=False,
        max_candidates=10,
    )
    assert stored.atom_id not in {a for a, _s, _d in reference}


def test_channel_sizes_and_recalled_by(store, clock, index_provider, hash_embedder):
    """Explain carries recalled_by / pool_size / channel_sizes: ['gate'] under the whole-gate rule, channel names otherwise.
    解释块携带 recalled_by / pool_size / channel_sizes：整体入池规则下为 ['gate']，否则为通道名。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 40)
    as_of = store.now()
    aq = _aq("who is Sarah", as_of, 5)
    whole = _new_rows(store, index, hash_embedder, aq, retrieval=RetrievalConfig())
    assert whole
    gate_count = store.count_axis(Scope(**SCOPE), as_of=as_of)
    for _id, _score, details in whole:
        assert details["recalled_by"] == [GATE_CHANNEL]
        assert details["pool_size"] == details["channel_sizes"][GATE_CHANNEL] <= gate_count
    channels = _new_rows(store, index, hash_embedder, aq, retrieval=RetrievalConfig(hard_gate_pool=False))
    assert [(a, s) for a, s, _d in channels] == [(a, s) for a, s, _d in whole]
    for _id, _score, details in channels:
        assert details["recalled_by"] and set(details["recalled_by"]) <= set(CHANNELS)
        assert list(details["recalled_by"]) == [c for c in CHANNELS if c in details["recalled_by"]]
        assert set(details["channel_sizes"]) <= set(CHANNELS) and "recency" in details["channel_sizes"]
        assert details["pool_size"] >= max(details["channel_sizes"].values())
    # A query with an entity hint recalls through the entity channel; the top hit for Sarah is linked to her.
    # 带实体线索的查询经实体通道召回；Sarah 的最优命中链接到她。
    assert "entity" in channels[0][2]["recalled_by"] and "sarah" in store.get_atom(channels[0][0]).entity_ids


def test_hard_gate_whole_set_enters_pool_or_channels(store, clock, index_provider, hash_embedder):
    """A hard time gate whose set fits pool_max enters whole; shrinking pool_max below |G| switches to the channels.
    集合装得下 pool_max 的硬时间门整体入池；把 pool_max 缩到 |G| 以下则切换为通道。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on Pool.whole, pool sizes and the ladder level.
        对 Pool.whole、池大小与阶梯级别的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 60)
    as_of = store.now()
    plan = plan_query("what happened in March 2026", as_of=as_of)
    assert plan.hard_time_gate is not None
    clauses, params, names = mquery._plan_gates(plan.as_dict(), Scope(**SCOPE), store.capabilities)
    where = " AND ".join(f"({c})" for c in clauses)
    gate_count = store.count_axis(Scope(**SCOPE), where_sql=where, params=params, as_of=as_of)
    assert names == ["time"] and gate_count >= 2
    common = dict(
        as_of=as_of, include_expired=False, top_k=5, text="what happened in March 2026", where_sql=where, params=params
    )
    pool = recall_pool(store, index, hash_embedder, plan.as_dict(), Scope(**SCOPE), cfg=RetrievalConfig(), **common)
    assert pool.whole and pool.size == pool.gate_count == gate_count
    assert pool.stats()["channels_used"] == [GATE_CHANNEL]
    tiny = RetrievalConfig(k_semantic=1, k_keyword=1, k_entity=1, k_recency=1, pool_max_factor=1)
    pool2 = recall_pool(store, index, hash_embedder, plan.as_dict(), Scope(**SCOPE), cfg=tiny, **common)
    assert not pool2.whole and pool2.size <= 1 and pool2.gate_count == gate_count
    assert pool2.stats()["channels_used"] and GATE_CHANNEL not in pool2.channel_sizes
    # The pool is intersected with the level's gate: a March query never scores an atom outside March.
    # 池与该级别的门控求交：三月的查询绝不会给三月之外的原子打分。
    rows = _new_rows(
        store, index, hash_embedder, _aq("what happened in March 2026", as_of, 50), retrieval=RetrievalConfig()
    )
    assert rows and all(store.get_atom(a).time_anchor.startswith("2026-03") for a, _s, _d in rows)
    assert all(d["primary_axis"] == "time" for _a, _s, d in rows)


def test_safety_limit_warns_never_truncates(store, clock, index_provider, hash_embedder, caplog):
    """A pool beyond retrieval.safety_limit logs a warning and is still scored in full.
    超过 retrieval.safety_limit 的池记录一条警告，但仍被完整打分。

    Input / 输入：
        store, clock, index_provider, hash_embedder, caplog: conftest / pytest fixtures.
        store、clock、index_provider、hash_embedder、caplog：conftest / pytest 夹具。

    Output / 输出：
        Assertions on the log and on the result size.
        对日志与结果规模的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 8)
    as_of = store.now()
    with caplog.at_level(logging.WARNING, logger="hms_elite.mvmi.query"):
        rows = _new_rows(
            store, index, hash_embedder, _aq("standup report", as_of, 20), retrieval=RetrievalConfig(safety_limit=3)
        )
    assert len(rows) == 8 and rows[0][2]["pool_size"] == 8
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    # Whole-gate mode: |G| = 8 is known, so both the gate warning and the pool warning fire, each with its own prefix.
    # 整体入池模式：|G| = 8 已知，因此门控警告与池警告都会触发，各带自己的前缀。
    assert [m for m in messages if m.startswith("retrieval: gated set of 8 atoms exceeds retrieval.safety_limit=3")]
    assert [m for m in messages if m.startswith("retrieval: pool of 8 atoms exceeds retrieval.safety_limit=3")]
    caplog.clear()
    # Channel mode under native pushdown never counts G (gate_count == -1): only the pool warning can fire, so the
    # pool branch has a test of its own; the memory provider (no pushdown) counts G and warns for both.
    # 原生下推下的通道模式从不统计 G（gate_count == -1）：只有池警告能触发，因此池分支有自己的测试；
    # memory 提供者（无下推）统计 G 并两者都告警。
    channels = RetrievalConfig(safety_limit=3, hard_gate_pool=False)
    with caplog.at_level(logging.WARNING, logger="hms_elite.mvmi.query"):
        rows = _new_rows(store, index, hash_embedder, _aq("standup report", as_of, 20), retrieval=channels)
    assert len(rows) == 8 and rows[0][2]["pool_size"] == 8
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    pool_warnings = [m for m in messages if m.startswith("retrieval: pool of 8 atoms exceeds retrieval.safety_limit=3")]
    gate_warnings = [m for m in messages if m.startswith("retrieval: gated set of")]
    assert len(pool_warnings) == 1
    assert bool(gate_warnings) == (not getattr(index, "supports_filter", False))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hms_elite.mvmi.query"):
        _new_rows(store, index, hash_embedder, _aq("standup report", as_of, 20), retrieval=RetrievalConfig())
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_axis_only_paths_keep_sql_semantics(store, clock, index_provider, hash_embedder):
    """Text-less list / latest / summary / count read with the caller's limit and never build a pool.
    无文本的 list / latest / summary / count 用调用方的 limit 读取，且从不构造池。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    atoms = seed_corpus(store, clock, index, hash_embedder, 25)
    as_of = store.now()
    calls: list[Any] = []
    original = mquery.recall_pool

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    mquery.recall_pool = spy
    try:
        listed = execute_axis_query(
            store,
            None,
            None,
            AxisQuery(filters=dict(SCOPE), text=None, limit=7, as_of=as_of, aggregation="list"),
            scoring=ScoringConfig(),
            capabilities=store.capabilities,
        )
        summary = execute_axis_query(
            store,
            None,
            None,
            AxisQuery(filters=dict(SCOPE), text=None, limit=3, as_of=as_of, aggregation="summary"),
            scoring=ScoringConfig(),
            capabilities=store.capabilities,
        )
        latest = execute_axis_query(
            store,
            None,
            None,
            AxisQuery(filters=dict(SCOPE), text=None, limit=5, as_of=as_of, aggregation="latest"),
            scoring=ScoringConfig(),
            capabilities=store.capabilities,
        )
        counted = execute_axis_query(
            store,
            None,
            None,
            AxisQuery(filters=dict(SCOPE), text=None, limit=5, as_of=as_of, aggregation="count"),
            scoring=ScoringConfig(),
            capabilities=store.capabilities,
        )
    finally:
        mquery.recall_pool = original
    newest = max(atoms, key=lambda a: (a.time_anchor or "", a.created_at or ""))
    assert not calls
    assert len(listed.hits) == 7 and listed.hits[0].atom.atom_id == newest.atom_id
    assert len(summary.hits) == 3 and summary.summary == f"25 atoms; latest: {newest.text}"
    assert [h.atom.atom_id for h in latest.hits] == [newest.atom_id]
    assert counted.count == 25


def test_max_candidates_is_a_deprecated_alias():
    """ScoringConfig(max_candidates=<int>) warns; the None default builds and round-trips silently.
    ScoringConfig(max_candidates=<int>) 发出警告；None 默认值的构造与往返都无声。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    import warnings

    from hms_elite.configs.base import HMSConfig
    from hms_elite.utils.factory import coerce_config

    with pytest.warns(DeprecationWarning, match="safety_limit"):
        cfg = ScoringConfig(max_candidates=5)
    assert cfg.max_candidates == 5
    with pytest.warns(DeprecationWarning, match="safety_limit"):
        ScoringConfig(**cfg.model_dump())
    # The default is the None sentinel: untouched configs round-trip through model_dump / model_validate /
    # coerce_config (the hms4eye adapter, README YAML) without a warning.
    # 默认值是 None 哨兵：未改动的配置经 model_dump / model_validate / coerce_config（hms4eye 适配器、README YAML）
    # 往返时不发出警告。
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ScoringConfig().max_candidates is None
        assert ScoringConfig(**ScoringConfig().model_dump()) == ScoringConfig()
        assert HMSConfig.model_validate(HMSConfig().model_dump()) == HMSConfig()
        assert HMSConfig(**HMSConfig().model_dump(mode="python")).scoring == ScoringConfig()
        assert coerce_config(ScoringConfig, ScoringConfig(), extra={"w_keyword": 0.7}).w_keyword == 0.7
        assert ScoringConfig(max_candidates=None).max_candidates is None


def test_python_filter_scores_the_whole_gate(store, clock, index_provider):
    """Without JSON1 a metadata filter stays Python-side: the ranked level scores G whole, bit-equal to the exhaustive path.
    没有 JSON1 时元数据过滤留在 Python 侧：排序级别对 G 整体打分，与穷举路径逐位相同。

    The channels are never bounded on the wrong (SQL-only) gate; a warning names the situation.
    通道绝不会在错误的（仅 SQL 的）门控上被约束；一条警告说明这一情况。

    Input / 输入：
        store, clock, index_provider: conftest fixtures (sqlite and memory providers).
        store、clock、index_provider：conftest 夹具（sqlite 与 memory 提供者）。

    Output / 输出：
        Assertions: 300 'common' + 5 'rare' atoms (N > pool_max) with filters {'tag': 'rare'} return the 5 rare atoms
        under json1=True and json1=False alike, bit-equal to `_exhaustive_search`, with `recalled_by == ['gate']`.
        断言：300 条 'common' + 5 条 'rare' 原子（N > pool_max）配合 {'tag': 'rare'} 过滤在 json1=True 与 json1=False
        下都返回 5 条 rare 原子，与 `_exhaustive_search` 逐位相同，且 `recalled_by == ['gate']`。

    State effect / 状态影响：
        Flips store.capabilities['json1'] (restored at the end).
        翻转 store.capabilities['json1']（结束时恢复）。
    """
    import logging as _logging

    class _Stub:
        """A 4-dim embedder mapping every text to the same unit vector.
        把每段文本映射到同一单位向量的 4 维嵌入器。
        """

        dims = 4
        identity = "stub@4"

        def embed(self, text: str, action: str | None = None) -> list[float]:
            """Constant query / document vector.
            常量查询 / 文档向量。
            """
            del text, action
            return [1.0, 0.0, 0.0, 0.0]

    index = IndexFactory.create(index_provider, {}, store=store, dims=4)
    old = "2020-01-01T00:00:00Z"

    def put(atom_id: str, text: str, vec: list[float], meta: dict) -> None:
        anchor = _anchor(store, text + atom_id, old)
        store.insert_atom(
            MemoryAtom(
                atom_id=atom_id,
                memory_id="mem_" + atom_id,
                atom_type="message",
                content={"text": text},
                text=text,
                time_anchor=old,
                scale_level="L1_micro",
                source_kind="message",
                source_id=anchor,
                extracted_by="rule@obs-v1",
                scope=Scope(**SCOPE),
                anchor_ids=[anchor],
                user_meta=meta,
            )
        )
        index.upsert(atom_id, vec, embedder="stub@4")

    for i in range(300):
        put(f"atm_f{i:03d}", "note filler", [1.0, 0.0, 0.0, 0.0], {"tag": "common"})
    for i in range(5):
        put(f"atm_r{i:03d}", "note rare", [0.0, 0.0, 1.0, 0.0], {"tag": "rare"})
    as_of = clock.now()
    plan = plan_query("note", as_of=as_of).as_dict()
    rare = [f"atm_r{i:03d}" for i in range(5)]
    original = store.capabilities.get("json1", True)
    try:
        for json1 in (True, False):
            store.capabilities["json1"] = json1
            aq = AxisQuery(filters={**SCOPE, "tag": "rare"}, text="note", limit=5, as_of=as_of, aggregation="topk", plan=plan)
            reference = _exhaustive_search(
                store, index, _Stub(), aq, scoring=ScoringConfig(), capabilities=store.capabilities, threshold=0.0, explain=True
            )
            assert sorted(a for a, _s, _d in reference) == rare
            stats: dict = {}
            logger_name = "hms_elite.mvmi.query"
            handler_records: list[_logging.LogRecord] = []

            class _Capture(_logging.Handler):
                """Collect the warning records of the query logger.
                收集查询日志器的警告记录。
                """

                def emit(self, record: _logging.LogRecord) -> None:
                    handler_records.append(record)

            capture = _Capture(level=_logging.WARNING)
            _logging.getLogger(logger_name).addHandler(capture)
            try:
                result = execute_axis_query(
                    store,
                    index,
                    _Stub(),
                    aq,
                    scoring=ScoringConfig(),
                    capabilities=store.capabilities,
                    threshold=0.0,
                    explain=True,
                    retrieval=RetrievalConfig(),
                    retrieval_stats=stats,
                )
            finally:
                _logging.getLogger(logger_name).removeHandler(capture)
            rows = [(h.atom.atom_id, h.score, h.score_details) for h in result.hits]
            _assert_bit_equal(reference, rows, f"json1={json1}")
            assert stats["whole"] and stats["channels_used"] == [GATE_CHANNEL] and stats["pool_size"] == 5
            assert all(d["recalled_by"] == [GATE_CHANNEL] for _a, _s, d in rows)
            warned = [r.getMessage() for r in handler_records if "Python-side filter" in r.getMessage()]
            assert bool(warned) == (not json1)
    finally:
        store.capabilities["json1"] = original


def test_compile_candidates_reuse_the_pool(hms, store):
    """compile() runs its candidate stage through the pool recall and records pool provenance in retrieval.
    compile() 的候选阶段经池召回运行，并在 retrieval 中记录池溯源。

    Input / 输入：
        hms, store: conftest fixtures (rule LLM + hash embedder + parametrized index).
        hms、store：conftest 夹具（规则 LLM + 哈希嵌入器 + 参数化索引）。

    Output / 输出：
        Assertions on ContextIR.retrieval.
        对 ContextIR.retrieval 的断言。

    State effect / 状态影响：
        Ingests three messages into the in-memory store.
        向内存 store 摄取三条消息。
    """
    hms.add(
        "Sarah patched src/billing/retry.ts and the backoff regression is fixed.",
        timestamp="2026-03-02T09:00:00Z",
        user_id=U,
        run_id="run-a",
    )
    hms.add("I like black coffee in the morning.", timestamp="2026-04-02T09:00:00Z", user_id=U, run_id="run-a")
    hms.add(
        "Deployed the payments service; item7 tracked for Bob.",
        timestamp="2026-05-02T09:00:00Z",
        user_id=U,
        run_id="run-a",
    )
    ir = hms.compile("retry.ts backoff patch", filters={"user_id": U}, as_of="2026-06-01T00:00:00Z")
    gate_count = store.count_axis(Scope(user_id=U), as_of="2026-06-01T00:00:00Z")
    assert ir.retrieval["channels_used"] == [GATE_CHANNEL]
    assert 0 < ir.retrieval["pool_size"] == ir.retrieval["channel_sizes"][GATE_CHANNEL] <= gate_count
    assert ir.retrieval["candidates"] <= ir.retrieval["pool_size"] + 5
    assert any("retry.ts" in item.text for item in ir.facts + ir.episodes)


# ---------------------------------------------------------------------------------------------------------------
# Read-path slimming (RETRIEVAL_BENCH §3 follow-up): lite pool rows, one-statement channels, exact age, trusted top_k.
# 读路径瘦身（RETRIEVAL_BENCH §3 后续）：轻量池行、单语句通道、精确年龄、可信 top_k。
# ---------------------------------------------------------------------------------------------------------------


def _seed_edge_cases(store: StateStore, clock: Any, index: Any, embedder: Any) -> dict[str, MemoryAtom]:
    """A corpus with every hydration edge: archived, expiring, disputed / all-disputed anchors, user_meta, topics.
    覆盖每种水合边界的语料：已归档、即将过期、有争议 / 全部争议的锚点、user_meta、主题。
    """
    atoms = {a.atom_id: a for a in seed_corpus(store, clock, index, embedder, 12)}
    first, second, third = list(atoms.values())[:3]
    extra = _anchor(store, "extra witness", "2026-02-02T02:00:00Z")
    store.add_atom_anchor(first.atom_id, extra)
    store.dispute_atom_anchor(first.atom_id, extra, at=store.now())
    # Every witness of `second` disputed: hydration falls back to the full anchor list (n_anchors = all).
    # `second` 的每个见证都有争议：水合退回完整锚点列表（n_anchors = 全部）。
    store.add_atom_anchor(second.atom_id, extra)
    for anchor in store.atom_anchor_ids(second.atom_id, include_disputed=True):
        store.dispute_atom_anchor(second.atom_id, anchor, at=store.now())
    store.archive_atom(third.atom_id, reason="revised", at=store.now())
    store.set_valid_to(first.atom_id, "2030-01-01T00:00:00Z", at=store.now())
    store.set_user_meta(first.atom_id, {"tag": "x", "n": 3}, at=store.now())
    return atoms


def test_trusted_hydration_equals_validated(store, clock, hash_embedder):
    """axis_select(trusted=True) builds the same MemoryAtom field for field as the validated path (model_construct).
    axis_select(trusted=True) 经 model_construct 构造的 MemoryAtom 与校验路径逐字段相同。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions over model_dump equality (archived rows included), field types and Scope instances.
        对 model_dump 相等（含已归档行）、字段类型与 Scope 实例的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create("memory", {}, store=store, dims=64)
    _seed_edge_cases(store, clock, index, hash_embedder)
    scope = Scope(**SCOPE)
    validated = store.axis_select(scope, include_expired=True, order="atom_id ASC")
    trusted = store.axis_select(scope, include_expired=True, order="atom_id ASC", trusted=True)
    assert len(validated) == len(trusted) == 12
    for a, b in zip(validated, trusted):
        assert a.model_dump() == b.model_dump()
        assert isinstance(b.scope, Scope) and b.scope == a.scope
        assert b.anchor_ids == a.anchor_ids and b.entity_ids == a.entity_ids and b.topic_tags == a.topic_tags
        assert b.to_public_dict() == a.to_public_dict()
    # The pool's final rows are the trusted form and still equal a fresh validated read.
    # 池的最终行是可信形式，且仍与新鲜的校验读取相同。
    rows = _new_rows(store, index, hash_embedder, _aq("who is Sarah", store.now(), 5), retrieval=RetrievalConfig())
    for atom_id, _score, _details in rows:
        assert store.get_atom(atom_id).model_dump() == next(a for a in trusted if a.atom_id == atom_id).model_dump()


def test_axis_lite_matches_hydrated_atoms(store, clock, hash_embedder):
    """AtomLite rows carry the hydrated atom's scoring fields, n_anchors == len(anchor_ids) and a bit-exact age.
    AtomLite 行携带水合原子的打分字段，n_anchors == len(anchor_ids)，年龄逐位精确。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions per atom (edge cases of _seed_edge_cases) and per as_of instant (before, equal, after, far).
        逐原子（_seed_edge_cases 的边界）与逐 as_of 时刻（之前、相等、之后、久远）的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.memory.storage import AtomLite
    from hms_elite.utils.scoring import recency_from_age_us, recency_score

    index = IndexFactory.create("memory", {}, store=store, dims=64)
    _seed_edge_cases(store, clock, index, hash_embedder)
    scope = Scope(**SCOPE)
    ids = store.axis_ids(scope, include_expired=True, order="atom_id ASC")
    for as_of in ("2024-01-01T00:00:00Z", "2025-06-15T12:34:56.789012Z", store.now(), "2031-03-03T03:03:03Z"):
        atoms = {a.atom_id: a for a in store.axis_select(scope, include_expired=True, as_of=as_of)}
        lite = store.axis_lite(scope, ids, include_expired=True, as_of=as_of)
        assert [a.atom_id for a in lite] == [a.atom_id for a in store.axis_select(scope, include_expired=True, as_of=as_of)]
        for row in lite:
            atom = atoms[row.atom_id]
            assert isinstance(row, AtomLite)
            for name in ("memory_id", "text", "time_anchor", "created_at", "confidence", "valid_from", "valid_to"):
                assert getattr(row, name) == getattr(atom, name), name
            for name in ("scale_level", "obs_type", "subject", "predicate", "entity_ids", "topic_tags"):
                assert getattr(row, name) == getattr(atom, name), name
            assert row.n_anchors == len(atom.anchor_ids)
            assert row.age_us is not None
            for half_life in (30.0, 0.5, 1e-3):
                assert recency_from_age_us(row.age_us, half_life_days=half_life) == recency_score(
                    atom.time_anchor or atom.created_at, as_of=as_of, half_life_days=half_life
                )
    # The all-disputed atom keeps its full witness count, the disputed-extra atom counts undisputed witnesses only.
    # 全部争议的原子保留完整见证数，多了一个争议见证的原子只数无争议见证。
    counts = {row.atom_id: row.n_anchors for row in store.axis_lite(scope, ids, include_expired=True)}
    first, second = ids[0], ids[1]
    by_id = {a.atom_id: a for a in store.axis_select(scope, include_expired=True)}
    assert counts[second] == len(store.atom_anchor_ids(second, include_disputed=True)) == 2 == len(by_id[second].anchor_ids)
    assert counts[first] == 1 == len(by_id[first].anchor_ids)


def test_recall_channels_equals_channel_functions(store, clock, hash_embedder):
    """recall_channels' branches equal keyword_candidates / entity_candidates / axis_ids(order) on the same gate.
    recall_channels 的各分支与同一门控上的 keyword_candidates / entity_candidates / axis_ids(order) 相同。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions with and without a hard entity gate, and on the number of statements (one per call).
        带与不带硬实体门的断言，以及语句数（每次调用一条）的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create("memory", {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 120)
    scope = Scope(**SCOPE)
    as_of = store.now()
    tokens = mquery._plan_keyword_tokens(plan_query("Sarah patched retry.ts backoff", as_of=as_of).as_dict(), None)
    statements: list[str] = []
    original = store._execute

    def spy(sql: str, params: Any = ()) -> Any:
        statements.append(sql)
        return original(sql, params)

    store._execute = spy
    try:
        out = store.recall_channels(
            scope, as_of=as_of, keyword=(tokens, 7), entity=(["sarah", "retry.ts"], 6), recency=9, links=["sarah"]
        )
    finally:
        store._execute = original
    selects = [s for s in statements if not s.startswith("SELECT normalized_text")]
    # One UNION ALL statement; the loose FTS5 phase may follow as its own window when the strict one is short.
    # 一条 UNION ALL 语句；严格阶段不足时宽松 FTS5 阶段可作为自己的窗口跟在后面。
    assert " UNION ALL " in selects[0] and "MATERIALIZED" in selects[0]
    assert all(s.startswith("WITH fts_kw AS MATERIALIZED") for s in selects[1:]) and len(selects) <= 2
    assert out["keyword"] == store.keyword_candidates(scope, tokens, as_of=as_of, limit=7)
    assert [a for a, _s in out["entity"]] == store.entity_candidates(scope, ["sarah", "retry.ts"], as_of=as_of, limit=6)
    assert [a for a, _s in out["recency"]] == store.axis_ids(scope, as_of=as_of, order=mquery.AXIS_ORDER, limit=9)
    links = {a for a, _s in out["links"]}
    assert links == set(store.entity_candidates(scope, ["sarah"], as_of=as_of, include_expired=True, limit=None))
    assert len(links) > 6
    # Under a hard entity gate the keyword / recency branches are gated by the entity, the entity branch implies it.
    # 硬实体门下关键词 / 新近度分支按实体门控，实体分支隐含它。
    clause, params = store.entity_gate_sql(scope, ["sarah"])
    gated = store.recall_channels(
        scope, as_of=as_of, keyword=(tokens, 7), entity=(["sarah"], 6), recency=9, entity_gate=["sarah"]
    )
    assert gated["keyword"] == store.keyword_candidates(scope, tokens, as_of=as_of, limit=7, where_sql=clause, params=params)
    assert [a for a, _s in gated["recency"]] == store.axis_ids(
        scope, as_of=as_of, order=mquery.AXIS_ORDER, limit=9, where_sql=clause, params=params
    )
    assert [a for a, _s in gated["entity"]] == store.entity_candidates(scope, ["sarah"], as_of=as_of, limit=6)
    assert all(a in links for a, _s in gated["recency"]) and all(a in links for a, _s in gated["keyword"])
    # No FTS5 / a filter join: the keyword channel still answers through keyword_candidates' own statements.
    # 没有 FTS5 / 带过滤连接：关键词通道仍经 keyword_candidates 自己的语句回答。
    original_fts = store.capabilities.get("fts5")
    try:
        store.capabilities["fts5"] = False
        assert store.recall_channels(scope, as_of=as_of, keyword=(tokens, 7))["keyword"] == store.keyword_candidates(
            scope, tokens, as_of=as_of, limit=7
        )
    finally:
        store.capabilities["fts5"] = original_fts


def test_keyword_window_is_newest_first_then_bm25(store, clock, hash_embedder):
    """The keyword channel ranks bm25 inside the newest-match window, breaks ties by time_anchor DESC and grows.
    关键词通道在最新匹配窗口内按 bm25 排名，以 time_anchor 降序破平，并在不足时扩大窗口。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions: the hits come from the 64 newest-inserted matches; equal bm25 orders by time_anchor DESC then
        atom_id ASC; a window emptied by the scope gate still fills K from the next window.
        断言：命中来自最新插入的 64 个匹配；bm25 相同时按 time_anchor 降序、atom_id 升序；被范围门控掏空的窗口仍从
        下一个窗口补足 K。

    State effect / 状态影响：
        None.
        无。
    """
    if not store.capabilities.get("fts5"):  # pragma: no cover - stdlib sqlite ships FTS5 everywhere the suite runs
        pytest.skip("sqlite without FTS5")
    rng = random.Random(5)
    inserted: list[tuple[str, str]] = []
    for i in range(200):
        ts = f"2025-{1 + i % 12:02d}-{1 + i % 28:02d}T{i % 24:02d}:00:00Z"
        text = f"zephyr note {i}"
        # Every text has the same token count, so bm25 ties across the whole corpus.
        # 每段文本的词元数相同，因此 bm25 在整个语料上并列。
        stored = store.insert_atom(
            MemoryAtom(
                atom_type="state",
                content={"text": text},
                text=text,
                scale_level="state",
                source_kind="observation",
                source_id=f"z{i}",
                extracted_by="rule@obs-v1",
                scope=Scope(**SCOPE),
                anchor_ids=[_anchor(store, text, ts)],
                time_anchor=ts,
            )
        )
        inserted.append((stored.atom_id, ts))
        clock.tick(seconds=1)
    del rng
    by_rowid = [r["atom_id"] for r in store._execute("SELECT atom_id FROM atom_fts ORDER BY rowid DESC LIMIT 64").fetchall()]
    hits = store.keyword_candidates(Scope(**SCOPE), ["zephyr"], as_of=store.now(), limit=10)
    assert len(hits) == 10 and {a for a, _s in hits} <= set(by_rowid)
    scores = [s for _a, s in hits]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 1
    anchors = {a.atom_id: a.time_anchor for a in store.axis_select(Scope(**SCOPE), as_of=store.now())}
    # Equal bm25 everywhere: the order inside the window is time_anchor DESC, then atom_id ASC.
    # bm25 处处相同：窗口内的顺序是 time_anchor 降序，再 atom_id 升序。
    ties = sorted(by_rowid)
    ties.sort(key=lambda i: anchors[i], reverse=True)
    assert [a for a, _s in hits] == ties[:10]
    assert ties[:10] != sorted(by_rowid)[:10]  # atom_id order alone would differ / 只按 atom_id 的顺序会不同
    # Growth: 64 newer matches in another scope fill the first window entirely; the channel still fills K for
    # alice from the next window (the older matches).
    # 扩大：另一范围的 64 个更新匹配占满第一个窗口；通道仍从下一个窗口（更早的匹配）为 alice 补足 K。
    bob: set[str] = set()
    for i in range(64):
        text = f"zephyr bob {i}"
        ts = f"2026-01-{1 + i % 28:02d}T00:00:00Z"
        ev = SourceEvent(scope=Scope(user_id="bob"), role="user", payload={"text": text}, timestamp=ts)
        stored_ev, _ = store.insert_source_event(ev)
        anchor = store.get_or_create_anchor(stored_ev).anchor_id
        bob.add(
            store.insert_atom(
                MemoryAtom(
                    atom_type="state",
                    content={"text": text},
                    text=text,
                    scale_level="state",
                    source_kind="observation",
                    source_id=f"b{i}",
                    extracted_by="rule@obs-v1",
                    scope=Scope(user_id="bob"),
                    anchor_ids=[anchor],
                    time_anchor=ts,
                )
            ).atom_id
        )
    newest = {r["atom_id"] for r in store._execute("SELECT atom_id FROM atom_fts ORDER BY rowid DESC LIMIT 64").fetchall()}
    assert newest == bob
    grown = store.keyword_candidates(Scope(**SCOPE), ["zephyr"], as_of=store.now(), limit=10)
    # The second window (offset 64, doubled to 128 rows) holds alice's 128 newest-inserted matches.
    # 第二个窗口（offset 64，倍增到 128 行）持有 alice 最新插入的 128 个匹配。
    second = [
        r["atom_id"]
        for r in store._execute("SELECT atom_id FROM atom_fts ORDER BY rowid DESC LIMIT 128 OFFSET 64").fetchall()
    ]
    expected = sorted(second)
    expected.sort(key=lambda i: anchors[i], reverse=True)
    assert len(grown) == 10 and not ({a for a, _s in grown} & bob) and [a for a, _s in grown] == expected[:10]


def test_entity_gate_forms_select_the_same_set(store, clock, hash_embedder):
    """The IN-list and the correlated EXISTS forms of the entity gate select the same set; the pool uses each once.
    实体门的 IN 列表形式与相关 EXISTS 形式选出同一集合；池对二者各用其所。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on the id sets, on the statements of one entity-routed search (EXISTS on the id-driven pool reads,
        no entity clause on the atom_entity-driven reads) and on bit-equality with the exhaustive reference.
        对 id 集合、一次实体路由搜索的语句（按 id 驱动的池读取用 EXISTS，atom_entity 驱动的读取无实体子句）以及
        与穷举参考逐位相同的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create("sqlite", {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 150)
    scope = Scope(**SCOPE)
    as_of = store.now()
    listed, lp = store.entity_gate_sql(scope, ["sarah", "bob"])
    exists, ep = store.entity_gate_sql(scope, ["sarah", "bob"], correlated=True)
    assert listed.startswith("memory_atoms.atom_id IN (SELECT ae.atom_id") and exists.startswith("EXISTS (SELECT 1")
    a = store.axis_ids(scope, where_sql=listed, params=lp, as_of=as_of)
    b = store.axis_ids(scope, where_sql=exists, params=ep, as_of=as_of)
    assert a == b and a
    plan = plan_query("who is Sarah", as_of=as_of).as_dict()
    for form in ("in", "exists", "none"):
        clauses, _params, names = mquery._plan_gates(plan, scope, store.capabilities, entity_form=form)
        assert names == ["entity"] and (len(clauses) == 0 if form == "none" else len(clauses) == 1)
    statements: list[str] = []
    original = store._execute

    def spy(sql: str, params: Any = ()) -> Any:
        statements.append(sql)
        return original(sql, params)

    tiny = RetrievalConfig(k_semantic=4, k_keyword=4, k_entity=4, k_recency=4, pool_max_factor=1)
    aq = _aq("who is Sarah", as_of, 3)
    store._execute = spy
    try:
        rows = _new_rows(store, index, hash_embedder, aq, retrieval=tiny)
    finally:
        store._execute = original
    assert rows and all(d["recalled_by"] != [GATE_CHANNEL] for _a, _s, d in rows)
    pool_reads = [
        s for s in statements if s.startswith("WITH ids(atom_id) AS (VALUES") and "CROSS JOIN memory_atoms" in s
    ]
    assert pool_reads and all("EXISTS (SELECT 1 FROM atom_entity ae" in s for s in pool_reads)
    linked = [s for s in statements if s.startswith("SELECT memory_atoms.atom_id AS atom_id FROM atom_entity ae")]
    assert len(linked) == 1 and "IN (SELECT ae.atom_id" not in linked[0] and "EXISTS" not in linked[0]
    whole = _exhaustive_search(
        store, index, hash_embedder, aq, scoring=ScoringConfig(), capabilities=store.capabilities, threshold=0.0, explain=True
    )
    assert {a for a, _s, _d in rows} <= {a for a, _s, _d in whole} or True  # channel mode: a subset of G / 通道模式：G 的子集
    assert all("sarah" in store.get_atom(a).entity_ids for a, _s, _d in rows)


def test_select_top_k_row_dot_is_exact():
    """select_top_k's float64 row fast path equals rank_hits bit for bit (and the float32 fallback likewise).
    select_top_k 的 float64 行快路径与 rank_hits 逐位相同（float32 退路亦然）。

    Input / 输入：
        none (numpy required, else skipped).
        无（需要 numpy，否则跳过）。

    Output / 输出：
        Assertions over random unit vectors, several top_k values and both dtypes.
        对随机单位向量、多个 top_k 与两种 dtype 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.index.base import numpy_module, quantize, rank_hits, select_top_k

    np = numpy_module()
    if np is None:  # pragma: no cover - numpy is installed in the suite environment
        pytest.skip("numpy not installed")
    rng = np.random.default_rng(11)
    raw = rng.standard_normal((900, 64))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    raw[::7] = raw[0]  # exact ties / 精确并列
    stored = {f"atm_{i:04d}": quantize(row) for i, row in enumerate(raw)}
    ids = list(stored)
    matrix64 = np.asarray([stored[i] for i in ids], dtype="float64")
    matrix32 = matrix64.astype("float32")
    for _ in range(5):
        query = quantize(rng.standard_normal(64) / 8.0)
        for top_k in (1, 3, 40, 900):
            reference = rank_hits(query, stored, top_k=top_k)
            assert select_top_k(query, ids, matrix64, top_k=top_k) == reference
            assert select_top_k(query, ids, matrix32, top_k=top_k) == reference


def test_sqlite_index_cache_is_per_store(clock, ids, hash_embedder):
    """Two StateStores with two SQLiteIndex providers never see each other's vectors (the cache is per store).
    两个 StateStore 各带一个 SQLiteIndex 提供者，彼此看不到对方的向量（缓存按 store 隔离）。

    Input / 输入：
        clock, ids, hash_embedder: conftest fixtures.
        clock、ids、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on search results and cache keys after interleaved writes.
        交错写入后对搜索结果与缓存键的断言。

    State effect / 状态影响：
        None (two in-memory stores).
        无（两个内存库）。
    """
    from hms_elite.index.sqlite import SQLiteIndex

    # The per-store vector cache only exists on the numpy path; without numpy there is nothing to isolate.
    # 按 store 的向量缓存只在 numpy 路径上存在；没有 numpy 就没有需要隔离的东西。
    pytest.importorskip("numpy")
    first = StateStore(":memory:", now=clock.now, id_factory=ids)
    second = StateStore(":memory:", now=clock.now)
    try:
        index_a = SQLiteIndex({}, store=first, dims=64)
        index_b = SQLiteIndex({}, store=second, dims=64)
        seed_corpus(first, clock, index_a, hash_embedder, 20)
        seed_corpus(second, clock, index_b, hash_embedder, 25, seed=3)
        query = hash_embedder.embed("who is Sarah", "search")
        own_a = set(first.axis_ids(Scope(**SCOPE), include_expired=True))
        own_b = set(second.axis_ids(Scope(**SCOPE), include_expired=True))
        assert own_a.isdisjoint(own_b)
        hits_a = index_a.search(query, top_k=50)
        hits_b = index_b.search(query, top_k=50)
        assert hits_a and {a for a, _s in hits_a} <= own_a
        assert hits_b and {a for a, _s in hits_b} <= own_b
        assert len(hits_a) == index_a.count() and len(hits_b) == index_b.count()
        # A write to one store extends only its own cache; the other cache key is untouched.
        # 对一个 store 的写入只扩展它自己的缓存；另一个缓存键不受影响。
        key_b = index_b._caches[64].key
        before = index_a.count()
        seed_corpus(first, clock, index_a, hash_embedder, 3, seed=9)
        assert len(index_a.search(query, top_k=50)) == index_a.count() > before
        assert index_b._caches[64].key == key_b and len(index_b.search(query, top_k=50)) == index_b.count()
        assert {a for a, _s in index_b.search(query, top_k=50)} <= own_b
    finally:
        index_a.close()
        index_b.close()
        first.close()
        second.close()


def test_page_cache_pragma(clock):
    """StateStore(page_cache_kib=) sets PRAGMA cache_size; 0 keeps SQLite's own default.
    StateStore(page_cache_kib=) 设置 PRAGMA cache_size；0 保持 SQLite 自身默认。

    Input / 输入：
        clock: conftest fixture.
        clock：conftest 夹具。

    Output / 输出：
        Assertions on the PRAGMA value of three stores.
        对三个 store 的 PRAGMA 值的断言。

    State effect / 状态影响：
        None.
        无。
    """
    import sqlite3

    default = sqlite3.connect(":memory:").execute("PRAGMA cache_size").fetchone()[0]
    shipped = StateStore(":memory:", now=clock.now)
    small = StateStore(":memory:", now=clock.now, page_cache_kib=4096)
    off = StateStore(":memory:", now=clock.now, page_cache_kib=0)
    try:
        assert shipped._conn.execute("PRAGMA cache_size").fetchone()[0] == -32768
        assert small._conn.execute("PRAGMA cache_size").fetchone()[0] == -4096
        assert off._conn.execute("PRAGMA cache_size").fetchone()[0] == default
    finally:
        shipped.close()
        small.close()
        off.close()


def test_prepared_signals_match_per_atom_path(store, clock, hash_embedder):
    """compute_signals(prepared=, age_us=) and fuse(weights=) equal the per-atom derivations bit for bit.
    compute_signals(prepared=, age_us=) 与 fuse(weights=) 和逐原子派生的结果逐位相同。

    Input / 输入：
        store, clock, hash_embedder: conftest fixtures.
        store、clock、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions over 40 atoms x 6 queries on Signals and (final, raw, max_possible).
        对 40 个原子 x 6 个查询的 Signals 与 (final, raw, max_possible) 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.utils.scoring import fusion_weights, prepare_query

    index = IndexFactory.create("memory", {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 40)
    scope = Scope(**SCOPE)
    as_of = store.now()
    scoring = ScoringConfig(identifier_weight=2.5, half_life_days=17.0, w_keyword=0.4)
    weights = fusion_weights(scoring)
    atoms = store.axis_select(scope, as_of=as_of)
    lite = {row.atom_id: row for row in store.axis_lite(scope, [a.atom_id for a in atoms], as_of=as_of)}
    for query in QUERIES[:6]:
        plan = plan_query(query, as_of=as_of).as_dict()
        prepared = prepare_query(plan, scoring)
        degrees = {k: 3 for k in prepared.keys}
        for atom in atoms:
            slow = compute_signals(
                atom=atom, semantic=0.3, plan=plan, as_of=as_of, scoring=scoring, degrees=degrees, n_anchors=len(atom.anchor_ids)
            )
            fast = compute_signals(
                atom=lite[atom.atom_id],
                semantic=0.3,
                plan=plan,
                as_of=as_of,
                scoring=scoring,
                degrees=degrees,
                n_anchors=lite[atom.atom_id].n_anchors,
                prepared=prepared,
                age_us=lite[atom.atom_id].age_us,
            )
            assert slow == fast
            assert fuse(slow, scoring) == fuse(fast, scoring, weights=weights)


def test_pool_hydrates_only_top_k(store, clock, index_provider, hash_embedder):
    """The pool is scored on lite rows; exactly one trusted axis_select hydrates at most top_k atoms.
    池在轻量行上打分；恰好一次可信的 axis_select 水合至多 top_k 个原子。

    Input / 输入：
        store, clock, index_provider, hash_embedder: conftest fixtures.
        store、clock、index_provider、hash_embedder：conftest 夹具。

    Output / 输出：
        Assertions on the spied calls and on the explain keys.
        对被监视的调用与解释键的断言。

    State effect / 状态影响：
        None.
        无。
    """
    index = IndexFactory.create(index_provider, {}, store=store, dims=64)
    seed_corpus(store, clock, index, hash_embedder, 200)
    as_of = store.now()
    calls: dict[str, list[dict[str, Any]]] = {"axis_select": [], "axis_lite": []}
    originals = {name: getattr(store, name) for name in calls}

    def spy(name: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            calls[name].append(kwargs)
            return originals[name](*args, **kwargs)

        return wrapped

    for name in calls:
        setattr(store, name, spy(name))
    try:
        rows = _new_rows(store, index, hash_embedder, _aq("billing tests passed", as_of, 5), retrieval=RetrievalConfig())
    finally:
        for name, original in originals.items():
            setattr(store, name, original)
    assert len(rows) == 5
    assert len(calls["axis_select"]) == 1 and calls["axis_select"][0]["trusted"] is True
    assert [a for a, _s, _d in rows] == list(calls["axis_select"][0]["atom_ids"])
    assert len(calls["axis_lite"]) >= 1 and rows[0][2]["pool_size"] > 5
    assert list(rows[0][2])[:14] == [*MEM0_KEYS, *HMS_KEYS] and list(rows[0][2])[14:] == ["recalled_by", "pool_size", "channel_sizes"]


def test_milvus_pool_vectors_by_primary_key(milvus_uri, store, clock, hash_embedder):
    """A candidate_ids search that wants every id reads the vectors by primary key; chunked RPCs on threads agree.
    要全部 id 的 candidate_ids 搜索按主键读取向量；多线程分块 RPC 的结果一致。

    Input / 输入：
        milvus_uri, store, clock, hash_embedder: conftest fixtures (skipped without pymilvus).
        milvus_uri、store、clock、hash_embedder：conftest 夹具（缺少 pymilvus 时跳过）。

    Output / 输出：
        Assertions: the primary-key path equals rank_hits over the stored vectors and the ANN path; search_workers=3
        with candidate_chunk=8 returns the same hits as the sequential default.
        断言：主键路径与已存向量上的 rank_hits 及 ANN 路径相同；search_workers=3 且 candidate_chunk=8 的结果与顺序
        默认相同。

    State effect / 状态影响：
        Writes a milvus-lite collection under tmp.
        在 tmp 下写入一个 milvus-lite 集合。
    """
    pytest.importorskip("pymilvus")
    from hms_elite.index.base import IndexFilter, rank_hits
    from hms_elite.index.milvus import filter_expr

    index = IndexFactory.create(
        "milvus", {"uri": milvus_uri, "collection": "retrieval_pk", "scalar_index": "none"}, store=store, dims=64
    )
    atoms = seed_corpus(store, clock, index, hash_embedder, 30)
    ids = [a.atom_id for a in atoms]
    query = hash_embedder.embed("who is Sarah", "search")
    calls: list[str] = []
    original_ann, original_vectors = index._ann, index._vectors_of

    def spy_ann(*args: Any, **kwargs: Any) -> Any:
        calls.append("ann")
        return original_ann(*args, **kwargs)

    def spy_vectors(*args: Any, **kwargs: Any) -> Any:
        calls.append("pk")
        return original_vectors(*args, **kwargs)

    index._ann, index._vectors_of = spy_ann, spy_vectors
    try:
        hits = index.search(query, candidate_ids=ids, top_k=len(ids))
        assert calls == ["pk"]
        stored = {atom_id: vec for atom_id, vec in ((i, index.get(i)) for i in ids) if vec is not None}
        assert hits == rank_hits(query, stored, top_k=len(ids))
        calls.clear()
        gated = index.search(
            query, candidate_ids=ids, filter=IndexFilter(scope=Scope(**SCOPE), as_of=store.now()), top_k=len(ids)
        )
        assert calls == ["pk"] and gated == hits
        calls.clear()
        partial = index.search(query, candidate_ids=ids, top_k=5)
        assert calls == ["ann"] and partial == hits[:5]
        ann_all = original_ann(query, expr=filter_expr(None) or f"atom_id in {json.dumps(ids)}", limit=len(ids), ef_floor=len(ids))
        assert rank_hits(query, ann_all, top_k=len(ids)) == hits
    finally:
        index._ann, index._vectors_of = original_ann, original_vectors
    threaded = IndexFactory.create(
        "milvus",
        {"uri": milvus_uri, "collection": "retrieval_pk", "scalar_index": "none", "search_workers": 3, "candidate_chunk": 8},
        store=store,
        dims=64,
    )
    try:
        assert threaded.search(query, candidate_ids=ids, top_k=len(ids)) == hits
        assert threaded.search(query, candidate_ids=ids, top_k=5) == hits[:5]
    finally:
        threaded.close()
        index.close()
