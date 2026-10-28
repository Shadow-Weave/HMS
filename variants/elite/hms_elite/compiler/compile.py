"""StateCompiler: axis candidates → work-state augmentation → graph expansion → slots → κ/invariants → budget → ContextIR.
StateCompiler：轴候选 → 工作态增广 → 图扩展 → 槽位 → κ/不变量 → 预算 → ContextIR。
"""

# Purpose (EN): Implement the read-only §15.2 compile algorithm and the §15.2 step-8 budget packer; with
#               selection='coverage' step 8 is requirement coverage (compiler.coverage) refined layer by layer
#               (compiler.scc) instead of slot shares.
# 目的（中）：实现只读的 §15.2 编译算法与 §15.2 第 8 步的预算打包器；selection='coverage' 时第 8 步改为按需求
#            覆盖（compiler.coverage）并逐层下钻（compiler.scc），而非槽位份额。

# Input (EN): Query, scope/filters, budget, mode, as_of, audience; the StateStore, index, embedder and SupportGraph.
# 输入（中）：查询、范围/过滤、预算、模式、as_of、受众；StateStore、索引、嵌入器与 SupportGraph。

# Output (EN): A ContextIR with six disjoint witnessed slots and retrieval provenance.
# 输出（中）：带六个互斥且有见证槽位及检索出处的 ContextIR。

# State effect (EN): None — compile() is read-only (row counts identical before/after); mode 'fast' never reads graph tables.
# 状态影响（中）：无——compile() 只读（前后行数一致）；'fast' 模式从不读取图表。

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hms_elite.compiler.coverage import extract_requirements, select_for_coverage
from hms_elite.compiler.scc import compile_hierarchical
from hms_elite.compiler.slots import (
    assign_slot,
    audit_reason,
    derive_next_action,
    evidence_items,
    invariant_items,
    item_from_atom,
    kappa_items,
    normalized_texts,
)
from hms_elite.configs.base import HMSConfig
from hms_elite.enums import (
    ABLATION_OF_MODE,
    VISIBILITY_RANK,
    WHY_EDGE_TYPES,
    AtomType,
    CompileMode,
    EdgeType,
    NodeKind,
    ObservationType,
    Route,
    Slot,
    SourceKind,
    rank,
)
from hms_elite.exceptions import ValidationError
from hms_elite.graph.operators import refine
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.memory.storage import StateStore
from hms_elite.models import Anchor, AxisQuery, ContextIR, ContextItem, InterfaceEP, MemoryAtom, Scope
from hms_elite.mvmi.filters import parse_filters
from hms_elite.mvmi.planner import plan_query
from hms_elite.mvmi.query import execute_axis_query
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.scoring import SIGNAL_NAMES
from hms_elite.utils.text import text_scoped

SLOT_NAMES: tuple[str, ...] = ("facts", "episodes", "preferences", "constraints", "skills", "audit")
PACKED_SLOTS: tuple[str, ...] = ("facts", "episodes", "preferences", "constraints", "skills")
SPILL_ORDER: tuple[str, ...] = ("facts", "episodes")
# Step-8 packers: 'shares' = fit_budget (slot-share caps), 'coverage' = requirement cover + hierarchical refinement.
# 第 8 步打包器：'shares' = fit_budget（槽位份额上限），'coverage' = 需求覆盖 + 分层下钻。
SELECTIONS: tuple[str, ...] = ("shares", "coverage")
_EVIDENCE_PREFIX = "anc:"
_ALWAYS_ACTIVE = frozenset({"semantic", "recency", "support"})
_AUGMENT_ROUTES = frozenset({Route.work_state.value, Route.detective.value})
_AUGMENT_WHERE = "(atom_type = ? OR obs_type = ? OR source_kind = ?)"


@dataclass
class _Candidate:
    """One candidate atom with its retrieval provenance.
    带检索出处的一个候选原子。
    """

    atom: MemoryAtom
    score: float
    source: str
    hops: int
    edge_ids: list[str] = field(default_factory=list)


def _pack_order(items: list[ContextItem]) -> list[ContextItem]:
    """(score DESC, time_anchor DESC, item_id ASC) — the packing order of §15.2 step 8.
    (score DESC, time_anchor DESC, item_id ASC)——§15.2 第 8 步的打包顺序。
    """
    ordered = sorted(items, key=lambda i: i.item_id)
    ordered.sort(key=lambda i: (i.score, i.time_anchor or ""), reverse=True)
    return ordered


def _render_order(items: list[ContextItem]) -> list[ContextItem]:
    """(time_anchor ASC nulls last, item_id ASC) — the rendering order of every non-evidence slot.
    (time_anchor ASC 空值靠后, item_id ASC)——每个非证据槽位的渲染顺序。
    """
    return sorted(items, key=lambda i: (i.time_anchor is None, i.time_anchor or "", i.item_id))


def _evidence_for(items: list[ContextItem], anchors: dict[str, Anchor], seen: set[str]) -> list[ContextItem]:
    """Evidence items for the anchors of `items` not yet in `seen` (which is updated).
    为 `items` 中尚不在 `seen`（会被更新）的锚点构造证据项。
    """
    fresh: list[Anchor] = []
    for item in items:
        for anchor_id in item.anchor_ids:
            if anchor_id in seen:
                continue
            seen.add(anchor_id)
            anchor = anchors.get(anchor_id)
            if anchor is not None:
                fresh.append(anchor)
    return evidence_items(fresh)


def fit_budget(
    slots: dict[str, list[ContextItem]],
    *,
    budget: int,
    shares: dict[str, float],
    min_constraints: int,
    anchors: dict[str, Anchor],
) -> tuple[dict[str, list[ContextItem]], dict[str, Any]]:
    """Pack candidate items into per-slot share caps, keep every evidence line, spill unused capacity (§15.2 step 8).
    按槽位份额上限打包候选项、保留全部证据行、将未用容量溢出（§15.2 第 8 步）。

    Input / 输入：
        slots: Candidate items per slot name (audit = non-evidence audit candidates).
        slots：按槽位名的候选项（audit = 非证据的 audit 候选）。

        budget: Token budget.
        budget：token 预算。

        shares: Slot share fractions (config.compiler.slot_shares).
        shares：槽位份额比例（config.compiler.slot_shares）。

        min_constraints: Constraints kept even beyond their cap.
        min_constraints：即便超出上限也保留的约束数量。

        anchors: anchor_id → Anchor for every anchor cited by a candidate (evidence lines are built from them).
        anchors：候选项引用的每个锚点 anchor_id → Anchor（证据行由此构造）。

    Output / 输出：
        (kept items per slot in rendering order — audit = evidence first, then packed audit items; info dict with
        used_tokens, dropped, over_budget_evidence, caps).
        （按渲染顺序的各槽位保留项——audit 为证据优先、其后打包的 audit 项；info 字典含 used_tokens、dropped、
        over_budget_evidence、caps）。

    State effect / 状态影响：
        None.
        无。
    """
    caps = {name: int(math.floor(float(shares.get(name, 0.0)) * budget)) for name in SLOT_NAMES}
    kept: dict[str, list[ContextItem]] = {name: [] for name in SLOT_NAMES}
    remainders: dict[str, list[ContextItem]] = {}
    used = 0
    dropped = 0
    for name in PACKED_SLOTS:
        ordered = _pack_order(list(slots.get(name) or []))
        spent = 0
        index = 0
        while index < len(ordered):
            item = ordered[index]
            must = name == "constraints" and len(kept[name]) < min_constraints
            if not must and spent + item.tokens > caps[name]:
                break
            kept[name].append(item)
            spent += item.tokens
            index += 1
        remainders[name] = ordered[index:]
        used += spent

    # Audit: evidence for every kept item is ALWAYS kept, other audit items pack within the audit cap.
    # audit：每个保留项的证据始终保留，其余 audit 项在 audit 上限内装填。
    audit_candidates = _pack_order(list(slots.get("audit") or []))
    audit_kept: list[ContextItem] = []
    spent = 0
    for item in audit_candidates:
        if spent + item.tokens > caps["audit"]:
            break
        audit_kept.append(item)
        spent += item.tokens
    used += spent
    dropped += len(audit_candidates) - len(audit_kept)
    seen: set[str] = set()
    evidence = _evidence_for([i for name in PACKED_SLOTS for i in kept[name]] + audit_kept, anchors, seen)
    used += sum(item.tokens for item in evidence)

    # Spill: unused capacity continues the sorted facts list, then episodes (their evidence is always kept too).
    # 溢出装填：剩余容量继续装排序后的 facts 列表，然后是 episodes（其证据同样始终保留）。
    leftover = budget - used
    for name in SPILL_ORDER:
        while remainders[name]:
            item = remainders[name][0]
            if item.tokens > leftover:
                break
            remainders[name].pop(0)
            kept[name].append(item)
            leftover -= item.tokens
            used += item.tokens
            extra = _evidence_for([item], anchors, seen)
            extra_tokens = sum(e.tokens for e in extra)
            evidence.extend(extra)
            leftover -= extra_tokens
            used += extra_tokens
    for name in PACKED_SLOTS:
        dropped += len(remainders[name])

    packed: dict[str, list[ContextItem]] = {name: _render_order(kept[name]) for name in PACKED_SLOTS}
    packed["audit"] = evidence_items([anchors[i.anchor_ids[0]] for i in evidence]) + _render_order(audit_kept)
    info = {
        "used_tokens": used,
        "dropped": dropped,
        "over_budget_evidence": max(0, used - budget),
        "caps": caps,
    }
    return packed, info


def select_by_coverage(
    slots: dict[str, list[ContextItem]],
    *,
    query: str,
    plan: dict[str, Any],
    budget: int,
    anchors: dict[str, Anchor],
    ir_fields: dict[str, Any],
) -> tuple[dict[str, list[ContextItem]], dict[str, Any], dict[str, Any]]:
    """Coverage packer (§15.2 step 8, selection='coverage'): requirements → hierarchical cover → per-slot floors.
    覆盖式打包器（§15.2 第 8 步，selection='coverage'）：需求 → 分层覆盖 → 槽位保底。

    Input / 输入：
        slots: Candidate items per slot name (audit = non-evidence audit candidates), as handed to fit_budget.
        slots：按槽位名的候选项（audit = 非证据的 audit 候选），与交给 fit_budget 的相同。

        query: The goal / question text the requirements are extracted from.
        query：抽取需求所用的目标 / 问题文本。

        plan: The planner dict (QueryPlan.as_dict()) whose identifiers / entities / time windows become requirements.
        plan：规划器字典（QueryPlan.as_dict()），其标识符 / 实体 / 时间窗成为需求。

        budget: Token budget shared by the cover and the floors.
        budget：覆盖与保底共用的 token 预算。

        anchors: anchor_id → Anchor for every anchor cited by a candidate.
        anchors：候选项引用的每个锚点 anchor_id → Anchor。

        ir_fields: query / goal / as_of / scope / mode / route of the ContextIR under construction.
        ir_fields：构造中的 ContextIR 的 query / goal / as_of / scope / mode / route。

    Output / 输出：
        (kept items per slot in rendering order with evidence first in audit; info dict shaped like fit_budget's;
        coverage report with 'stats' for retrieval['coverage'] and 'hierarchy' for retrieval['hierarchy']).
        （按渲染顺序的各槽位保留项，audit 证据优先；形状同 fit_budget 的 info 字典；覆盖报告，'stats' 进
        retrieval['coverage']，'hierarchy' 进 retrieval['hierarchy']）。

    State effect / 状态影响：
        None.
        无。
    """
    requirements = extract_requirements(query, plan, as_of=ir_fields.get("as_of"))
    ordered = {name: _render_order(list(slots.get(name) or [])) for name in PACKED_SLOTS}
    audit_all = _render_order(list(slots.get("audit") or []))
    seen: set[str] = set()
    evidence_all = _evidence_for([i for name in PACKED_SLOTS for i in ordered[name]] + audit_all, anchors, seen)
    # The unbudgeted ContextIR every candidate sits in: scc covers it top-down (L3 skills, L2 invariants, L1 atoms).
    # 所有候选所在的、未裁预算的 ContextIR：scc 自上而下覆盖它（L3 技能、L2 不变量、L1 原子）。
    full = ContextIR(
        **ir_fields,
        facts=ordered["facts"],
        episodes=ordered["episodes"],
        preferences=ordered["preferences"],
        constraints=ordered["constraints"],
        skills=ordered["skills"],
        audit=evidence_all + audit_all,
        budget_tokens=budget,
        used_tokens=sum(i.tokens for name in PACKED_SLOTS for i in ordered[name])
        + sum(i.tokens for i in evidence_all + audit_all),
        retrieval={"plan": dict(plan)},
    )
    hir = compile_hierarchical(full, requirements=requirements, budget_tokens=budget)
    keep = {item.item_id for item in hir.ir.items() if not item.item_id.startswith(_EVIDENCE_PREFIX)}
    # Floors: leftover budget gives every empty non-audit slot its best-scored item (select_for_coverage with no
    # requirements degenerates to the per-slot floor fill), so a card never loses its constraints / preferences to
    # a query that names none of them.
    # 保底：剩余预算让每个空的非 audit 槽位得到其分数最高的项（无需求的 select_for_coverage 退化为逐槽位保底
    # 填充），因此不点名约束 / 偏好的查询也不会让卡片失去它们。
    leftover = budget - hir.used_tokens
    rest = [i for i in full.items() if i.item_id not in keep and not i.item_id.startswith(_EVIDENCE_PREFIX)]
    floors: list[str] = []
    if leftover > 0 and rest:
        floor = select_for_coverage(rest, [], budget_tokens=leftover, min_per_slot=1)
        floors = [i.item_id for i in floor.selected]
        keep.update(floors)
    packed: dict[str, list[ContextItem]] = {name: [i for i in ordered[name] if i.item_id in keep] for name in PACKED_SLOTS}
    audit_kept = [i for i in audit_all if i.item_id in keep]
    cited: set[str] = set()
    evidence = _evidence_for([i for name in PACKED_SLOTS for i in packed[name]] + audit_kept, anchors, cited)
    packed["audit"] = evidence_items([anchors[i.anchor_ids[0]] for i in evidence]) + audit_kept
    used = sum(item.tokens for name in SLOT_NAMES for item in packed[name])
    n_candidates = sum(len(ordered[name]) for name in PACKED_SLOTS) + len(audit_all)
    n_kept = sum(len(packed[name]) for name in PACKED_SLOTS) + len(audit_kept)
    info = {"used_tokens": used, "dropped": n_candidates - n_kept, "over_budget_evidence": max(0, used - budget)}
    n_req = len(requirements)
    stats = {
        "selection": "coverage",
        "requirements": [req.as_dict() for req in requirements],
        "covered": {k: list(v) for k, v in hir.covered.items()},
        "uncovered": [req.req_id for req in hir.uncovered],
        "coverage_ratio": 1.0 if n_req == 0 else round(len(hir.covered) / n_req, 4),
        "layers": [{"layer": r.layer, "selected": list(r.selected), "uncovered": list(r.uncovered)} for r in hir.layers],
        "kappa_closure": list(hir.kappa_closure),
        "floors": floors,
    }
    return packed, info, {"stats": stats, "hierarchy": hir.report()}


class StateCompiler:
    """The read-only state compiler bound to a store, index, embedder, SupportGraph and config.
    绑定到存储、索引、嵌入器、SupportGraph 与配置的只读状态编译器。
    """

    def __init__(
        self, store: StateStore, index: Any, embedder: Any, graph: SupportGraph, config: HMSConfig, *, entity_extractor: Any | None = None
    ) -> None:
        """Bind the collaborators.
        绑定协作对象。

        Input / 输入：
            store: The StateStore.
            store：StateStore。

            index: IndexBase provider (semantic similarity).
            index：IndexBase 提供者（语义相似度）。

            embedder: EmbedderBase (embeds the query).
            embedder：EmbedderBase（嵌入查询）。

            graph: The SupportGraph (neighbourhood expansion).
            graph：SupportGraph（邻域扩展）。

            config: HMSConfig (compiler + scoring sections are read).
            config：HMSConfig（读取 compiler 与 scoring 部分）。

            entity_extractor: The facade's EntityExtractorBase for the query plan's entity hints (CONTRACT §9.6); None → utils.text.extract_entities (the default rule extractor, byte-identical).
            entity_extractor：门面的 EntityExtractorBase，供查询规划的实体线索使用（CONTRACT §9.6）；None → utils.text.extract_entities（默认规则抽取器，逐字节一致）。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.store = store
        self.index = index
        self.embedder = embedder
        self.graph = graph
        self.config = config
        self.entity_extractor = entity_extractor

    @text_scoped
    def compile(
        self,
        query: str,
        *,
        scope: Scope,
        filters: dict,
        budget_tokens: int | None = None,
        mode: str | None = None,
        as_of: str | None = None,
        require_witness: bool | None = None,
        audience: str | None = None,
        selection: str | None = None,
    ) -> ContextIR:
        """Compile the Work State for a query (§15.2): axis first, graph on demand, deterministic slots and budget.
        为查询编译工作态（§15.2）：先轴，按需图扩展，确定性槽位与预算。

        Input / 输入：
            query: The goal / question text.
            query：目标 / 问题文本。

            scope: Session scope (Mem0 ids).
            scope：会话范围（Mem0 id）。

            filters: Mem0 filter dict (scope keys plus axis/meta filters; a scalar 'visibility' is the audience).
            filters：Mem0 过滤字典（范围键加轴/元过滤；标量 'visibility' 即受众）。

            budget_tokens: Token budget (None → config.compiler.budget_tokens).
            budget_tokens：token 预算（None → config.compiler.budget_tokens）。

            mode: CompileMode value (None → config.compiler.default_mode).
            mode：CompileMode 值（None → config.compiler.default_mode）。

            as_of: Reference instant (None → store.now()).
            as_of：参考时刻（None → store.now()）。

            require_witness: Reserved for the facade's PC-IR lowering (None → config.compiler.require_witness).
            require_witness：为门面的 PC-IR 降低预留（None → config.compiler.require_witness）。

            audience: Visibility audience gating every selection (None → the scalar 'visibility' filter, else no gate).
            audience：门控所有选择的可见性受众（None → 标量 'visibility' 过滤，否则不门控）。

            selection: Step-8 packer, 'shares' | 'coverage' (None → config.compiler.selection; VALIDATION_013 otherwise).
            selection：第 8 步打包器，'shares' | 'coverage'（None → config.compiler.selection；否则 VALIDATION_013）。

        Output / 输出：
            ContextIR (mode = requested mode; retrieval['ablation'] records what ran; under 'coverage' retrieval also
            carries 'selection', 'coverage' and 'hierarchy', and to_pcir() adds the requirements_covered obligation).
            ContextIR（mode = 请求的模式；retrieval['ablation'] 记录实际运行的层；'coverage' 下 retrieval 另带
            'selection'、'coverage' 与 'hierarchy'，to_pcir() 追加 requirements_covered 义务）。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        store = self.store
        cfg = self.config.compiler
        instant = to_utc_iso(as_of) if as_of else store.now()
        mode_value = str(getattr(mode, "value", mode) or getattr(cfg.default_mode, "value", cfg.default_mode))
        if mode_value not in CompileMode.values():
            raise ValidationError(
                f"unknown compile mode {mode_value!r}",
                "VALIDATION_013",
                details={"mode": mode_value, "supported": CompileMode.values()},
            )
        filters = dict(filters or {})
        if scope is None or scope.is_empty():
            scope, _expr = parse_filters(filters, require_scope=True, context="compile")
        if audience is None and isinstance(filters.get("visibility"), str):
            audience = filters["visibility"]
        if audience is not None:
            audience = str(getattr(audience, "value", audience))
            if audience not in VISIBILITY_RANK:
                raise ValidationError(
                    f"unknown audience {audience!r}",
                    "VALIDATION_013",
                    details={"audience": audience, "supported": sorted(VISIBILITY_RANK)},
                )
        _require = cfg.require_witness if require_witness is None else bool(require_witness)
        selection_value = str(selection or cfg.selection)
        if selection_value not in SELECTIONS:
            raise ValidationError(
                f"unknown compile selection {selection_value!r}",
                "VALIDATION_013",
                details={"selection": selection_value, "supported": list(SELECTIONS)},
            )
        budget = int(budget_tokens or cfg.budget_tokens)
        plan = plan_query(query, as_of=instant, mode=mode_value, extractor=self.entity_extractor)
        route = plan.route

        def gate(visibility: Any) -> bool:
            """True when `visibility` may be shown to the audience.
            当 `visibility` 可展示给受众时为真。
            """
            return audience is None or rank(str(getattr(visibility, "value", visibility))) >= rank(audience)

        dropped_governance = 0
        dropped_covered = 0
        cands: dict[str, _Candidate] = {}

        # ---- step 2: axis candidates through the same bounded pool recall as search (never a private scan) ----
        # ---- 第 2 步：轴查询候选走与 search 相同的有界池召回（绝不自行穷举） -----------------------------------
        text = query if query and query.strip() else None
        aq = AxisQuery(
            filters=filters,
            scope=scope,
            text=text,
            limit=cfg.candidate_k,
            as_of=instant,
            aggregation="topk" if text is not None else "list",
            plan={**plan.as_dict(), "atom_types": [t for t in AtomType.values() if t != AtomType.entity.value]},
        )
        result = execute_axis_query(
            store,
            self.index,
            self.embedder,
            aq,
            scoring=self.config.scoring,
            capabilities=store.capabilities,
            threshold=0.0,
            retrieval=self.config.retrieval,
            extractor=self.entity_extractor,
        )
        pool_stats: dict[str, Any] = dict(result.retrieval)
        for hit in result.hits:
            atom = hit.atom
            if str(atom.atom_type) == AtomType.entity.value:
                continue
            if not gate(atom.visibility):
                dropped_governance += 1
                continue
            cands[atom.atom_id] = _Candidate(atom=atom, score=float(hit.score), source="axis", hops=0)

        # ---- step 3: work-state augmentation -----------------------------------------------------------------
        # ---- 第 3 步：工作态增补 -----------------------------------------------------------------------------
        augmented = False
        if (
            route in _AUGMENT_ROUTES
            or mode_value in (CompileMode.deep.value, CompileMode.axis_graph.value)
            or (mode_value == CompileMode.auto.value and len(store.list_eps(scope)) >= 1)
        ):
            augmented = True
            extra = store.axis_select(
                scope,
                where_sql=_AUGMENT_WHERE,
                params=[AtomType.episode.value, ObservationType.constraint.value, SourceKind.skill.value],
                as_of=instant,
            )
            for atom in extra:
                if atom.atom_id in cands or str(atom.atom_type) == AtomType.entity.value:
                    continue
                if not gate(atom.visibility):
                    dropped_governance += 1
                    continue
                cands[atom.atom_id] = _Candidate(atom=atom, score=0.0, source="axis", hops=0)

        # ---- step 4: graph expansion (never in fast; auto iff augmented; always in deep / axis_graph) ---------
        # ---- 第 4 步：图扩展（fast 从不扩展；auto 当且仅当已增补时扩展；deep / axis_graph 总是扩展） ----------
        expand = mode_value != CompileMode.fast.value and (
            mode_value in (CompileMode.deep.value, CompileMode.axis_graph.value) or augmented
        )
        support_layer = expand and mode_value in (CompileMode.deep.value, CompileMode.auto.value)
        hops = 0
        revised: list[tuple[MemoryAtom, MemoryAtom, str]] = []
        conflict_edges: list[Any] = []
        refined: list[tuple[str, dict]] = []
        if expand:
            hops = int(cfg.graph_hops)
            seeds = list(cands)
            nodes, edges, paths = self.graph.neighbors(
                seeds, hops=hops, edge_types=WHY_EDGE_TYPES, valid_only=False, max_nodes=cfg.max_why_nodes
            )
            max_seed = max((c.score for c in cands.values()), default=0.0)
            fetched = {a.atom_id: a for a in store.get_atoms([n.node_id for n in nodes if n.node_id not in cands])}
            for node in nodes:
                if node.node_id in cands:
                    continue
                atom = fetched.get(node.node_id)
                if atom is None or str(atom.atom_type) == AtomType.entity.value or not atom.is_valid_at(instant):
                    continue
                if not gate(atom.visibility):
                    dropped_governance += 1
                    continue
                cands[atom.atom_id] = _Candidate(
                    atom=atom,
                    score=0.5 * max_seed,
                    source="graph",
                    hops=len(paths.get(node.node_id) or []),
                    edge_ids=list(paths.get(node.node_id) or []),
                )
            seen_old: set[str] = set()
            for edge in edges:
                if str(edge.edge_type) != EdgeType.revision.value or not edge.valid:
                    continue
                if edge.dst_id not in cands or edge.src_id in cands or edge.src_id in seen_old:
                    continue
                old = fetched.get(edge.src_id) or store.get_atom(edge.src_id)
                if old is None or old.is_valid_at(instant):
                    continue
                seen_old.add(old.atom_id)
                if not gate(old.visibility):
                    dropped_governance += 1
                    continue
                revised.append((old, cands[edge.dst_id].atom, edge.edge_id))
            for edge in edges:
                if (
                    str(edge.edge_type) == EdgeType.conflict.value
                    and edge.valid
                    and edge.resolved_at is None
                    and edge.src_id in cands
                    and edge.dst_id in cands
                ):
                    conflict_edges.append(edge)
            if support_layer:
                for node in nodes:
                    if str(node.node_kind) != NodeKind.interface_ep.value or not node.valid:
                        continue
                    refined.append((node.node_id, refine(store, self.graph, node.ref_id, hops=0)))

        # ---- step 5: dedup (dict) + drop event atoms covered by a selected episode ----------------------------
        # ---- 第 5 步：去重（dict）+ 丢弃已被选中 episode 覆盖的 event 原子 ------------------------------------
        n_candidates = len(cands)
        all_anchor_ids: dict[str, list[str]] = {}
        undisputed: dict[str, list[str]] = {}
        for atom_id in list(cands):
            all_anchor_ids[atom_id] = store.atom_anchor_ids(atom_id, include_disputed=True)
            undisputed[atom_id] = store.atom_anchor_ids(atom_id)
        for old, _new, _edge in revised:
            all_anchor_ids[old.atom_id] = store.atom_anchor_ids(old.atom_id, include_disputed=True)
            undisputed[old.atom_id] = store.atom_anchor_ids(old.atom_id)
        episode_anchor_ids: set[str] = set()
        for cand in cands.values():
            if str(cand.atom.atom_type) == AtomType.episode.value:
                episode_anchor_ids.update(all_anchor_ids.get(cand.atom.atom_id) or [])
        for atom_id in list(cands):
            atom = cands[atom_id].atom
            primary = (all_anchor_ids.get(atom_id) or atom.anchor_ids or [None])[0]
            if str(atom.atom_type) == AtomType.event.value and primary in episode_anchor_ids:
                del cands[atom_id]
                dropped_covered += 1

        # ---- step 6: slot assignment, anchor gate, κ and invariants ------------------------------------------
        # ---- 第 6 步：槽位分配、锚点门控、κ 与不变量 ----------------------------------------------------------
        wanted: set[str] = set()
        for ids in all_anchor_ids.values():
            wanted.update(ids)
        for _node_id, sub in refined:
            iep_dict = sub.get("interface_ep") or {}
            for inv in list(iep_dict.get("invariants") or []) + list(iep_dict.get("residuals") or []):
                wanted.update(inv.get("anchor_ids") or [])
            wanted.update(iep_dict.get("evidence_anchor_ids") or [])
        anchor_rows: dict[str, Anchor] = {a.anchor_id: a for a in store.get_anchors(sorted(wanted))}

        def gated_anchors(ids: list[str]) -> list[str]:
            """Known anchors whose permission_scope passes the audience gate (dropped ones are counted).
            permission_scope 通过受众门控的已知锚点（被丢弃者计数）。
            """
            nonlocal dropped_governance
            kept_ids: list[str] = []
            for anchor_id in ids:
                row = anchor_rows.get(anchor_id)
                if row is None:
                    continue
                if not gate(row.permission_scope):
                    dropped_governance += 1
                    continue
                kept_ids.append(anchor_id)
            return kept_ids

        slot_lists: dict[str, list[ContextItem]] = {name: [] for name in SLOT_NAMES}
        for atom_id in sorted(cands):
            cand = cands[atom_id]
            atom = cand.atom
            slot = assign_slot(atom, as_of=instant)
            ids = gated_anchors(all_anchor_ids.get(atom_id) or [])
            if not ids:
                dropped_governance += 1
                continue
            explain: dict[str, Any] = {
                "disputed_anchor_ids": [a for a in ids if a not in (undisputed.get(atom_id) or [])],
                "edge_ids": list(cand.edge_ids),
            }
            if slot == Slot.audit:
                explain["reason"] = audit_reason(atom)
            item = item_from_atom(
                atom, slot=slot, score=cand.score, anchor_ids=ids, source=cand.source, hops=cand.hops, explain=explain
            )
            slot_lists[str(getattr(slot, "value", slot))].append(item)
        for old, new, edge_id in revised:
            ids = gated_anchors(all_anchor_ids.get(old.atom_id) or [])
            if not ids:
                dropped_governance += 1
                continue
            slot = assign_slot(old, as_of=instant, reason="revised")
            item = item_from_atom(
                old,
                slot=slot,
                score=0.0,
                anchor_ids=ids,
                source="graph",
                hops=1,
                explain={
                    "reason": "revised",
                    "superseded_by_text": new.text,
                    "superseded_by": new.atom_id,
                    "edge_ids": [edge_id],
                    "disputed_anchor_ids": [a for a in ids if a not in (undisputed.get(old.atom_id) or [])],
                },
            )
            slot_lists["audit"].append(item)

        selected_texts = normalized_texts(slot_lists["constraints"]) | normalized_texts(slot_lists["audit"])
        residual_texts: list[str] = []
        for node_id, sub in refined:
            iep = InterfaceEP(**sub["interface_ep"])
            inv_constraints, inv_audit = invariant_items(iep, node_id, selected_texts)
            for item in inv_constraints + inv_audit:
                ids = gated_anchors(list(item.anchor_ids))
                if not ids:
                    dropped_governance += 1
                    continue
                gated_item = item.model_copy(update={"anchor_ids": ids})
                slot_lists[str(getattr(gated_item.slot, "value", gated_item.slot))].append(gated_item)
                selected_texts.add(normalized_texts([gated_item]).pop())
            residual_texts.extend(res.text for res in iep.residuals)
        slot_lists["constraints"].extend(kappa_items(slot_lists["episodes"], selected_texts))

        # ---- step 7: goal + next action on the pre-budget candidate set --------------------------------------
        # ---- 第 7 步：在预算裁剪前的候选集上求 goal 与 next action -------------------------------------------
        goal = query
        text_of = {atom_id: cand.atom.text for atom_id, cand in cands.items()}
        conflict_pairs = [(text_of.get(e.src_id, e.src_id), text_of.get(e.dst_id, e.dst_id)) for e in conflict_edges]
        next_action = derive_next_action(
            episodes=slot_lists["episodes"],
            constraints=slot_lists["constraints"],
            residuals=residual_texts,
            conflicts=conflict_pairs,
            goal=goal,
        )

        # ---- step 8: budget ---------------------------------------------------------------------------------
        # ---- 第 8 步：预算装填 ------------------------------------------------------------------------------
        coverage: dict[str, Any] | None = None
        if selection_value == "coverage":
            packed, info, coverage = select_by_coverage(
                slot_lists,
                query=query,
                plan=plan.as_dict(),
                budget=budget,
                anchors=anchor_rows,
                ir_fields={
                    "query": query,
                    "goal": goal,
                    "as_of": instant,
                    "scope": scope,
                    "mode": CompileMode(mode_value),
                    "route": Route(route),
                },
            )
        else:
            packed, info = fit_budget(
                slot_lists,
                budget=budget,
                shares=dict(cfg.slot_shares),
                min_constraints=int(cfg.min_constraints),
                anchors=anchor_rows,
            )
        kept_atom_ids = {item.atom_id for name in PACKED_SLOTS for item in packed[name] if item.atom_id}
        unresolved = [e.edge_id for e in conflict_edges if e.src_id in kept_atom_ids and e.dst_id in kept_atom_ids]
        cited: set[str] = set()
        for name in SLOT_NAMES:
            for item in packed[name]:
                cited.update(item.anchor_ids)

        # ---- step 9: retrieval provenance -------------------------------------------------------------------
        # ---- 第 9 步：检索溯源 ------------------------------------------------------------------------------
        if expand:
            ablation = ABLATION_OF_MODE[CompileMode.deep.value if support_layer else CompileMode.axis_graph.value]
        else:
            ablation = ABLATION_OF_MODE[CompileMode.fast.value]
        signals = [
            name
            for name in SIGNAL_NAMES
            if name in _ALWAYS_ACTIVE
            or (name == "keyword" and bool(plan.content_tokens))
            or (name == "entity" and bool(plan.entity_hints))
        ]
        retrieval: dict[str, Any] = {
            "signals": signals,
            "route": route,
            "plan": plan.as_dict(),
            "mode": mode_value,
            "ablation": ablation,
            "candidates": n_candidates,
            "graph_hops": hops,
            "dropped": info["dropped"],
            "dropped_covered": dropped_covered,
            "dropped_governance": dropped_governance,
            "over_budget_evidence": info["over_budget_evidence"],
            # Which axis-query ladder level produced the candidates ('scale_fallback' / 'fallback' when relaxed).
            # 轴查询阶梯的哪一级产生了候选（放宽时为 'scale_fallback' / 'fallback'）。
            "primary_axis": result.primary_axis,
            # Pool provenance of the candidate stage (empty for a text-less compile, which lists in axis order).
            # 候选阶段的池溯源（无文本编译按轴序列出，此时为空）。
            "pool_size": int(pool_stats.get("pool_size", 0)),
            "channel_sizes": dict(pool_stats.get("channel_sizes") or {}),
            "channels_used": list(pool_stats.get("channels_used") or []),
            "rerank": None,
            "anchors": [anchor_rows[a].model_dump() for a in sorted(cited) if a in anchor_rows],
            "audience": audience,
        }
        if coverage is not None:
            # Only the coverage packer adds keys: the 'shares' retrieval dict stays exactly as before.
            # 只有覆盖式打包器添加键：'shares' 的 retrieval 字典与之前完全一致。
            retrieval["selection"] = "coverage"
            retrieval["coverage"] = coverage["stats"]
            retrieval["hierarchy"] = coverage["hierarchy"]
        return ContextIR(
            query=query,
            goal=goal,
            as_of=instant,
            scope=scope,
            mode=CompileMode(mode_value),
            route=Route(route),
            facts=packed["facts"],
            episodes=packed["episodes"],
            preferences=packed["preferences"],
            constraints=packed["constraints"],
            skills=packed["skills"],
            audit=packed["audit"],
            next_action=next_action,
            budget_tokens=budget,
            used_tokens=int(info["used_tokens"]),
            retrieval=retrieval,
            unresolved_conflicts=unresolved,
        )
