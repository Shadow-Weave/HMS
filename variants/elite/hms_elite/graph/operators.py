"""The three inheritance operators (coarsen / project / refine) plus skill promotion and the round-trip check.
三个继承算子（coarsen / project / refine）、技能提升与往返校验。
"""

# Purpose (EN): Build InterfaceEPs from low subgraphs, project their boundary upward, open on demand, promote skills.
# 目的（中）：由低层子图构造 InterfaceEP、向上投影其边界边、按需细化、提升技能（§14.2）。

# Input (EN): Node id sets, a SupportGraph, a StateStore + index + embedder, an optional LLM, scope and instants.
# 输入（中）：节点 id 集合、SupportGraph、StateStore + 索引 + 嵌入器、可选 LLM、范围与时刻。

# Output (EN): Persisted InterfaceEPs, high edge ids, refine subgraph dicts, skill atom ids.
# 输出（中）：已持久化的 InterfaceEP、高层边 id、细化子图字典、技能原子 id。

# State effect (EN): coarsen/project/promote_skills write atoms, interface tables, edges, projections; refine reads.
# 状态影响（中）：coarsen/project/promote_skills 经 StateStore 写原子、接口表、边与投影；refine 只读。

from __future__ import annotations

import logging
from typing import Any

from hms_elite.configs.prompts import COARSEN_PROMPT, build_coarsen_prompt
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.enums import (
    AtomType,
    EdgeType,
    InvariantKind,
    NodeKind,
    ResidualKind,
    Reward,
    Role,
    Scale,
    Sentiment,
    SourceKind,
    Visibility,
)
from hms_elite.exceptions import StoreError, ValidationError
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.index.base import IndexBase
from hms_elite.ingest.atoms import create_atom
from hms_elite.llms.base import LLMBase
from hms_elite.llms.rule import coarsen_rule
from hms_elite.memory.storage import StateStore
from hms_elite.models import GraphEdge, GraphNode, InterfaceEP, Invariant, Kappa, MemoryAtom, Residual, Scope, Slots
from hms_elite.timeutil import to_utc_iso
from hms_elite.utils.text import md5_hex, ordered_union, sha256_hex, topic_tags

logger = logging.getLogger(__name__)

_SLOT_NAMES = ("u", "b", "a", "y", "r", "kappa")
_TARGET_KIND = {Scale.L2_meso.value: NodeKind.ep.value, Scale.L3_macro.value: NodeKind.interface_ep.value}
_SENTIMENT_OF_REWARD = {
    Reward.negative.value: Sentiment.negative.value,
    Reward.positive.value: Sentiment.positive.value,
}


def _time_of(atom: MemoryAtom | None) -> str:
    """Sort key of an atom: time_anchor, else created_at, else ''.
    原子的排序键：time_anchor，否则 created_at，否则空串。
    """
    if atom is None:
        return ""
    return atom.time_anchor or atom.created_at or ""


def _member_slots(store: StateStore, node: GraphNode, atom: MemoryAtom) -> Slots:
    """Slots of a support member: eps.slots for EP nodes, interface_eps.slots_json for InterfaceEP nodes.
    支持成员的槽位：EP 节点取 eps.slots，InterfaceEP 节点取 interface_eps.slots_json。
    """
    if str(node.node_kind) == NodeKind.ep.value:
        ep = store.get_ep(node.ref_id)
        if ep is not None:
            return ep.slots
    elif str(node.node_kind) == NodeKind.interface_ep.value:
        iep = store.get_interface_ep(node.ref_id)
        if iep is not None:
            return iep.slots
    raw = atom.content.get("slots") if isinstance(atom.content, dict) else None
    if isinstance(raw, dict):
        return Slots(**raw)
    raise ValidationError(
        f"support node {node.node_id!r} carries no slots", "VALIDATION_005", details={"node_id": node.node_id}
    )


def _mean_confidence(atoms: dict[str, MemoryAtom], ids: list[str]) -> float:
    """round(mean(atom.confidence), 4) over the given atom ids (0.7 when none).
    给定原子 id 上 round(mean(atom.confidence), 4)（无时为 0.7）。
    """
    values = [float(atoms[i].confidence) for i in ids if i in atoms]
    return round(sum(values) / len(values), 4) if values else 0.7


def _anchor_union(store: StateStore, ids: list[str]) -> list[str]:
    """Sorted union of the undisputed anchors of the given atoms.
    给定原子未争议锚点的排序并集。
    """
    union: set[str] = set()
    for i in ids:
        union.update(store.atom_anchor_ids(i))
    return sorted(union)


def _llm_proposal(
    llm: LLMBase, members: list[dict], internal: list[dict], boundary: list[dict], target_scale: str
) -> dict | None:
    """Run the `coarsen` prompt and return the raw interface_ep dict (None when unusable).
    执行 `coarsen` 提示词并返回原始 interface_ep 字典（不可用时为 None）。
    """
    user = build_coarsen_prompt(
        support_nodes=members, internal_edges=internal, boundary_edges=boundary, target_scale=target_scale
    )
    raw = llm.generate_json("coarsen", system=COARSEN_PROMPT, user=user)
    return raw if isinstance(raw, dict) and isinstance(raw.get("slots"), dict) else None


def _merge_proposal(proposal: dict, rule_out: dict, known: set[str]) -> dict:
    """Validate an LLM proposal slot by slot against the rule output (invalid refs → rule value with a warning).
    逐槽位用规则输出校验 LLM 提案（引用无效 → 采用规则值并告警）。
    """
    slots: dict[str, Any] = dict(rule_out["slots"])
    support: dict[str, list[str]] = {k: list(v) for k, v in rule_out["slot_support"].items()}
    llm_slots = proposal.get("slots") or {}
    llm_support = proposal.get("slot_support") or {}
    for name in _SLOT_NAMES:
        refs = llm_support.get(name)
        refs = [str(r) for r in refs] if isinstance(refs, list) else []
        value = llm_slots.get(name)
        valid_refs = bool(refs) and all(r in known for r in refs)
        if name == "kappa":
            usable = isinstance(value, dict) and isinstance(value.get("constraints"), list)
        elif name in ("u", "a"):
            usable = isinstance(value, str) and bool(value.strip())
        elif name == "r":
            # r is the Reward enum: an off-enum string would raise pydantic ValidationError inside the transaction
            # r 是 Reward 枚举：枚举外的字符串会让 Slots(...) 在事务内抛 pydantic ValidationError，必须在此拦下
            usable = isinstance(value, str) and value in Reward.values()
        else:
            usable = isinstance(value, str)
        if not (valid_refs and usable):
            logger.warning(
                "coarsen: slot %r of the LLM proposal has no valid support refs or usable value; using the rule value",
                name,
            )
            continue
        if name == "kappa":
            rule_visibility = rule_out["slots"]["kappa"]["visibility"]
            visibility = value.get("visibility") or rule_visibility
            if visibility not in Visibility.values():
                # kappa.visibility is the Visibility enum: an off-enum value → the rule visibility, LLM constraints kept
                # kappa.visibility 是 Visibility 枚举：枚举外取值 → 采用规则可见性，LLM 的约束保留
                logger.warning(
                    "coarsen: kappa.visibility %r of the LLM proposal is not a Visibility; using %r",
                    visibility,
                    rule_visibility,
                )
                visibility = rule_visibility
            slots["kappa"] = {"constraints": [str(c) for c in value["constraints"]], "visibility": str(visibility)}
        else:
            slots[name] = value
        support[name] = list(dict.fromkeys(refs))
    kinds_inv = set(InvariantKind.values())
    kinds_res = set(ResidualKind.values())

    def _filter(items: Any, kinds: set[str]) -> list[dict] | None:
        """Keep entries with a known kind and refs ⊂ known; None when the key is absent.
        保留 kind 已知且引用 ⊂ known 的条目；键缺失时为 None。
        """
        if not isinstance(items, list):
            return None
        kept: list[dict] = []
        for item in items:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            refs = item.get("support_refs")
            refs = [str(r) for r in refs] if isinstance(refs, list) else []
            if str(item.get("kind")) not in kinds or not refs or any(r not in known for r in refs):
                logger.warning("coarsen: dropping LLM item %r (unknown kind or refs)", item.get("text"))
                continue
            kept.append({"text": str(item["text"]), "kind": str(item["kind"]), "support_refs": refs})
        return kept

    invariants = _filter(proposal.get("invariants"), kinds_inv)
    residuals = _filter(proposal.get("residuals"), kinds_res)
    return {
        "slots": slots,
        "slot_support": support,
        "invariants": rule_out["invariants"] if invariants is None else invariants,
        "residuals": rule_out["residuals"] if residuals is None else residuals,
    }


def coarsen(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    graph: SupportGraph,
    node_ids: list[str],
    *,
    target_scale: str = "L2_meso",
    policy: str = "rule",
    llm: LLMBase | None,
    at: str,
    scope: Scope,
    min_cohesion: float,
) -> InterfaceEP | None:
    """Coarsen: low subgraph → InterfaceEP.
    Coarsen：低层子图 → 高层接口。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index (the InterfaceEP's episode atom is indexed).
        index：向量索引（InterfaceEP 的情节原子会被索引）。

        embedder: Embedder for the episode atom.
        embedder：情节原子的嵌入器。

        graph: The SupportGraph over `store`.
        graph：基于 `store` 的 SupportGraph。

        node_ids: ≥2 valid node ids of the same scope — all `ep` for L2_meso, all `interface_ep` for L3_macro.
        node_ids：≥2 个同范围的有效节点 id —— L2_meso 全为 `ep`，L3_macro 全为 `interface_ep`。

        target_scale: 'L2_meso' | 'L3_macro'.
        target_scale：'L2_meso' | 'L3_macro'。

        policy: 'rule' (coarsen_rule) | 'llm' (the coarsen prompt, refs validated against S).
        policy：'rule'（coarsen_rule）| 'llm'（coarsen 提示词，引用按 S 校验）。

        llm: The LLM used by policy 'llm' (ignored by 'rule').
        llm：policy 为 'llm' 时使用的 LLM（'rule' 忽略）。

        at: Instant recorded as the InterfaceEP's created_at and used for supersession.
        at：记录为 InterfaceEP created_at 并用于替代失效的时刻。

        scope: Scope of the InterfaceEP (members must match it).
        scope：InterfaceEP 的范围（成员必须与之匹配）。

        min_cohesion: Below this cohesion nothing is written and None is returned.
        min_cohesion：内聚度低于该值时不写入并返回 None。

    Output / 输出：
        The persisted InterfaceEP (an existing one with the same support_hash is returned unchanged), or None.
        已持久化的 InterfaceEP（support_hash 相同的既有 InterfaceEP 原样返回），或 None。

    State effect / 状态影响：
        One transaction: episode atom (+node, vector, history), interface_eps, interface_ep_support; subsets revised.
        一个事务：情节原子（+节点、向量、历史）、interface_eps、interface_ep_support；严格子集的前驱失效并连 revision 边。
    """
    if target_scale not in _TARGET_KIND:
        raise ValidationError(
            f"target_scale must be L2_meso|L3_macro, got {target_scale!r}",
            "VALIDATION_013",
            details={"target_scale": target_scale},
        )
    if policy not in ("rule", "llm"):
        raise ValidationError(f"policy must be rule|llm, got {policy!r}", "VALIDATION_013", details={"policy": policy})
    ids = list(dict.fromkeys(str(n) for n in node_ids))
    if len(ids) < 2:
        raise ValidationError("coarsen needs at least two support nodes", "VALIDATION_005", details={"node_ids": ids})
    nodes = {n.node_id: n for n in store.get_nodes(ids)}
    missing = [i for i in ids if i not in nodes]
    if missing:
        raise ValidationError(f"unknown support nodes {missing}", "NOT_FOUND_001", details={"missing": missing})
    expected = _TARGET_KIND[target_scale]
    bad = [i for i in ids if str(nodes[i].node_kind) != expected or not nodes[i].valid]
    if bad:
        raise ValidationError(
            f"coarsen to {target_scale} needs valid {expected} nodes; offending: {bad}",
            "VALIDATION_005",
            details={"node_ids": bad, "expected_kind": expected},
        )
    foreign = [i for i in ids if not scope.matches(nodes[i].scope)]
    if foreign:
        raise ValidationError(
            f"support nodes outside scope {scope.key()!r}: {foreign}", "VALIDATION_005", details={"node_ids": foreign}
        )
    atoms = {a.atom_id: a for a in store.get_atoms(ids)}
    support = sorted(ids, key=lambda i: (_time_of(atoms.get(i)), i))
    support_hash = sha256_hex(",".join(sorted(support)))
    existing = store.find_interface_ep_by_support_hash(scope, support_hash)
    if existing is not None and existing.scale == target_scale:
        return existing
    e_in = graph.internal_edges(support)
    e_b = graph.boundary(support)
    cohesion = graph.cohesion(support)
    if cohesion < min_cohesion:
        return None

    # Prompt-shaped member / edge dicts (refs n0.., x0.., b0..; anchors w0..) — shared by the rule and the LLM path.
    # 提示词形态的成员 / 边字典（引用 n0.., x0.., b0..；锚点 w0..）—— 规则路径与 LLM 路径共用。
    ref_of = {node_id: f"n{i}" for i, node_id in enumerate(support)}
    id_of = {ref: node_id for node_id, ref in ref_of.items()}
    member_slots: dict[str, Slots] = {}
    members: list[dict] = []
    anchor_ref = 0
    for node_id in support:
        slots = _member_slots(store, nodes[node_id], atoms[node_id])
        member_slots[node_id] = slots
        anchors = store.atom_anchor_ids(node_id)
        members.append(
            {
                "ref": ref_of[node_id],
                "kind": str(nodes[node_id].node_kind),
                "u": slots.u,
                "b": slots.b,
                "a": slots.a,
                "y": slots.y,
                "r": str(slots.r),
                "constraints": list(slots.kappa.constraints),
                "visibility": str(slots.kappa.visibility),
                "time": atoms[node_id].time_anchor or atoms[node_id].created_at or "",
                "anchor_refs": [f"w{anchor_ref + k}" for k in range(len(anchors))],
            }
        )
        anchor_ref += len(anchors)
    internal = [
        {"ref": f"x{i}", "src": ref_of[e.src_id], "dst": ref_of[e.dst_id], "type": str(e.edge_type)}
        for i, e in enumerate(e_in)
    ]
    boundary = [
        {
            "ref": f"b{i}",
            "inner": ref_of[b.inner_node_id],
            "outer": "outside",
            "direction": b.direction,
            "type": str(b.edge_type),
        }
        for i, b in enumerate(e_b)
    ]
    rule_out = coarsen_rule(members, internal, boundary, target_scale)
    if rule_out is None:
        return None
    op_version = "rule-coarsen-v1"
    extracted_by = "rule@coarsen-v1"
    out = rule_out
    if policy == "llm" and llm is not None:
        # 无锁：在事务开始之前完成。
        proposal = _llm_proposal(llm, members, internal, boundary, target_scale)  # unlocked: before the transaction
        if proposal is None:
            logger.warning("coarsen: unusable LLM proposal; falling back to the rule output")
        else:
            out = _merge_proposal(proposal, rule_out, set(id_of))
            op_version = "llm-coarsen-v1"
            extracted_by = f"{llm.identity}@coarsen-v1"

    with store.transaction():
        evidence: set[str] = set()
        for node_id in support:
            evidence.update(store.atom_anchor_ids(node_id))
        for e in e_in:
            evidence.update(e.witness_anchor_ids)
        evidence_anchor_ids = sorted(evidence)
        check = set().union(
            *[set(store.atom_anchor_ids(n)) for n in support], *[set(e.witness_anchor_ids) for e in e_in]
        )
        # 证据保持
        if set(evidence_anchor_ids) != check:  # evidence preservation
            raise StoreError("coarsen lost evidence anchors", "STORE_003", details={"expected": sorted(check)})
        kappa_raw = out["slots"].get("kappa") or {}
        slots = Slots(
            u=str(out["slots"].get("u") or ""),
            b=str(out["slots"].get("b") or ""),
            a=str(out["slots"].get("a") or ""),
            y=str(out["slots"].get("y") or ""),
            r=str(out["slots"].get("r") or Reward.unknown.value),
            kappa=Kappa(
                constraints=[str(c) for c in (kappa_raw.get("constraints") or [])],
                visibility=str(kappa_raw.get("visibility") or "private"),
                evidence=evidence_anchor_ids,
            ),
        )
        slot_support: dict[str, list[str]] = {}
        for name in _SLOT_NAMES:
            cited = [id_of[r] for r in (out["slot_support"].get(name) or []) if r in id_of]
            slot_support[name] = list(dict.fromkeys(cited)) or [support[0]]
        invariants: list[Invariant] = []
        for item in out["invariants"]:
            members_ids = [id_of[r] for r in item.get("support_refs", []) if r in id_of]
            if not members_ids:
                continue
            invariants.append(
                Invariant(
                    text=str(item["text"]),
                    kind=str(item.get("kind") or InvariantKind.constraint.value),
                    support_node_ids=members_ids,
                    anchor_ids=_anchor_union(store, members_ids),
                    confidence=_mean_confidence(atoms, members_ids),
                )
            )
        residuals: list[Residual] = []
        for item in out["residuals"]:
            members_ids = [id_of[r] for r in item.get("support_refs", []) if r in id_of]
            edge_ids: list[str] = []
            if str(item.get("kind")) == ResidualKind.unresolved_conflict.value:
                edge_ids = [
                    e.edge_id
                    for e in e_in
                    if str(e.edge_type) == EdgeType.conflict.value and {e.src_id, e.dst_id} <= set(members_ids)
                ]
            residuals.append(
                Residual(
                    text=str(item["text"]),
                    kind=str(item["kind"]),
                    support_node_ids=members_ids,
                    anchor_ids=_anchor_union(store, members_ids),
                    edge_ids=edge_ids,
                )
            )
        if cohesion < min_cohesion + 0.1:
            residuals.append(
                Residual(
                    text=f"low cohesion {cohesion:.4f} (threshold {min_cohesion + 0.1:.4f})",
                    kind=ResidualKind.low_cohesion.value,
                    support_node_ids=list(support),
                    anchor_ids=_anchor_union(store, support),
                    edge_ids=[e.edge_id for e in e_in],
                )
            )
        member_scopes = {n.scope.key() for n in nodes.values()}
        iep_scope = nodes[support[0]].scope if len(member_scopes) == 1 else scope
        iep_id = store.new_id("iep")
        if target_scale == Scale.L3_macro.value:
            source_ep_ids = ordered_union(*[list(atoms[n].source_ep_ids) for n in support])
        else:
            source_ep_ids = [nodes[n].ref_id for n in support if str(nodes[n].node_kind) == NodeKind.ep.value]
        topics = ordered_union(*[list(atoms[n].topic_tags) for n in support])
        entities = ordered_union(*[list(atoms[n].entity_ids) for n in support])
        text = slots.gloss()
        time_anchor = _time_of(atoms[support[-1]]) or None
        atom = MemoryAtom(
            atom_type=AtomType.episode,
            content={
                "text": text,
                "interface_ep_id": iep_id,
                "slots": slots.model_dump(),
                "cohesion_score": cohesion,
                "invariants": [inv.text for inv in invariants],
                "residuals": [res.text for res in residuals],
            },
            text=text,
            time_anchor=time_anchor,
            valid_from=time_anchor,
            entity_ids=list(entities),
            topic_tags=list(topics),
            scale_level=target_scale,
            role=Role.extracted,
            sentiment=_SENTIMENT_OF_REWARD.get(str(slots.r), Sentiment.neutral.value),
            source_kind=SourceKind.interface_ep,
            source_id=iep_id,
            source_ep_ids=list(source_ep_ids),
            confidence=_mean_confidence(atoms, support),
            extracted_by=extracted_by,
            visibility=str(slots.kappa.visibility),
            scope=iep_scope,
            anchor_ids=list(evidence_anchor_ids),
        )
        stored_atom = create_atom(store, index, embedder, atom, extra_mentions=list(entities))
        iep = InterfaceEP(
            interface_ep_id=iep_id,
            scale=target_scale,
            slots=slots,
            slot_support=slot_support,
            support_node_ids=list(support),
            support_edge_ids=[e.edge_id for e in e_in],
            boundary_edges=list(e_b),
            invariants=invariants,
            residuals=residuals,
            evidence_anchor_ids=list(evidence_anchor_ids),
            coarsen_op_version=op_version,
            cohesion_score=cohesion,
            atom_id=stored_atom.atom_id,
            scope=iep_scope,
            created_at=to_utc_iso(at),
        )
        stored = store.insert_interface_ep(iep)
        # Supersession: a valid InterfaceEP whose support is a strict subset of S is invalidated and linked by revision.
        # 取代：支撑集是 S 的真子集的有效 InterfaceEP 会被置为失效，并通过 revision 关联。
        support_set = set(support)
        for old in store.list_interface_eps(scope, valid_only=True, scale=target_scale):
            if old.interface_ep_id == stored.interface_ep_id or not old.atom_id:
                continue
            old_set = set(old.support_node_ids)
            if old_set and old_set < support_set:
                store.invalidate_interface_ep(old.interface_ep_id, at=to_utc_iso(at))
                new_members = [n for n in support if n not in old_set]
                witnesses = ordered_union(*[store.atom_anchor_ids(n) for n in new_members])
                graph.insert_edge(
                    src_id=old.atom_id,
                    dst_id=stored.atom_id,
                    edge_type=EdgeType.revision.value,
                    witness_anchor_ids=witnesses,
                    created_by="consolidation",
                    proposer="rule",
                    confidence=0.9,
                    label=f"coarsen: {old.interface_ep_id} → {stored.interface_ep_id}",
                    scale=target_scale,
                    is_boundary=False,
                )
        return stored


def project(graph: SupportGraph, iep: InterfaceEP, *, at: str) -> list[str]:
    """Project: boundary edges → high edges.
    Project：边界边 → 高层边。

    Input / 输入：
        graph: The SupportGraph.
        graph：SupportGraph。

        iep: The InterfaceEP whose live boundary is projected.
        iep：其当前边界被投影的 InterfaceEP。

        at: Instant of the operation (recorded for callers; edges are stamped by the store clock).
        at：操作时刻（供调用方记录；边由存储时钟打点）。

    Output / 输出：
        High edge ids (one per (outer', type), first-creation order).
        高层边 id（每个 (outer', type) 一条，按首次创建顺序）。

    State effect / 状态影响：
        Writes graph_edges(is_boundary=1, witness_edge_id=low); the trigger writes edge_projections; merges add rows.
        写 graph_edges(is_boundary=1, witness_edge_id=low)；触发器写 edge_projections；合并的低层边追加投影行。
    """
    store = graph.store
    if not iep.atom_id:
        raise ValidationError(
            "InterfaceEP has no episode atom", "VALIDATION_005", details={"interface_ep_id": iep.interface_ep_id}
        )
    high_ids: list[str] = []
    with store.transaction():
        for b in graph.boundary(iep.support_node_ids):
            low = store.get_edge(b.edge_id)
            if low is None:
                continue
            owner = graph.owner_interface_ep(b.outer_node_id, scale=str(iep.scale))
            outer = owner.atom_id if owner is not None and owner.atom_id else b.outer_node_id
            if outer == iep.atom_id:
                # 已被吸收：该外部节点本就属于这个界面
                continue  # absorbed: the outer node already belongs to this interface
            src, dst = (iep.atom_id, outer) if b.direction == "out" else (outer, iep.atom_id)
            existing = store.find_edge(src, dst, str(low.edge_type), str(iep.scale))
            edge_id = graph.insert_edge(
                src_id=src,
                dst_id=dst,
                edge_type=str(low.edge_type),
                witness_anchor_ids=list(low.witness_anchor_ids),
                created_by="project",
                proposer="project-v1",
                confidence=float(low.confidence),
                label=low.label,
                scale=str(iep.scale),
                is_boundary=True,
                witness_edge_id=low.edge_id,
            )
            if existing is not None:
                store.add_projection(edge_id, low.edge_id)
            if edge_id not in high_ids:
                high_ids.append(edge_id)
        # 边界保持后置条件
        for edge_id in high_ids:  # boundary preservation post-condition
            if not store.projections_of(edge_id):
                raise StoreError(
                    f"projected edge {edge_id} has no edge_projections row",
                    "STORE_003",
                    details={"edge_id": edge_id, "interface_ep_id": iep.interface_ep_id},
                )
    return high_ids


def refine(store: StateStore, graph: SupportGraph, interface_ep_id: str, *, hops: int = 0, edge_types=None) -> dict:
    """Refine on demand: high object → low support subgraph.
    按需细化：高层对象 → 低层支持子图。

    Input / 输入：
        store: The state store.
        store：状态存储。

        graph: The SupportGraph.
        graph：SupportGraph。

        interface_ep_id: The InterfaceEP to open.
        interface_ep_id：待打开的 InterfaceEP。

        hops: Extra BFS levels beyond the boundary (0 → support set only).
        hops：边界之外的额外 BFS 层数（0 → 仅支持集）。

        edge_types: Edge type whitelist for the extra hops.
        edge_types：额外层数使用的边类型白名单。

    Output / 输出：
        Keys: interface_ep, nodes, edges, boundary_edges, anchors, projections ({high_edge_id: [low_edge_id]}).
        键：interface_ep、nodes、edges、boundary_edges、anchors、projections（{high_edge_id: [low_edge_id]}），均为字典。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    iep = store.get_interface_ep(str(interface_ep_id))
    if iep is None:
        raise ValidationError(
            f"unknown interface_ep {interface_ep_id!r}", "NOT_FOUND_001", details={"interface_ep_id": interface_ep_id}
        )
    support = list(iep.support_node_ids)
    node_map = {n.node_id: n for n in store.get_nodes(support)}
    order = [i for i in support if i in node_map]
    edge_map = {e.edge_id: e for e in graph.internal_edges(support)}
    boundary = graph.boundary(support)
    if hops > 0:
        more_nodes, more_edges, _paths = graph.neighbors(support, hops=hops, edge_types=edge_types)
        for n in more_nodes:
            if n.node_id not in node_map:
                node_map[n.node_id] = n
                order.append(n.node_id)
        for e in more_edges:
            edge_map.setdefault(e.edge_id, e)
    edges = sorted(edge_map.values(), key=lambda e: (e.created_at or "", e.edge_id))
    anchor_ids: set[str] = set(iep.evidence_anchor_ids)
    for e in edges:
        anchor_ids.update(e.witness_anchor_ids)
    for b in boundary:
        anchor_ids.update(b.witness_anchor_ids)
    for node_id in order:
        anchor_ids.update(store.atom_anchor_ids(node_id))
    projections: dict[str, list[str]] = {}
    for b in boundary:
        for high in store.projections_from(b.edge_id):
            projections.setdefault(high, []).append(b.edge_id)
    return {
        "interface_ep": iep.model_dump(),
        "nodes": [node_map[i].model_dump() for i in order],
        "edges": [e.model_dump() for e in edges],
        "boundary_edges": [b.model_dump() for b in boundary],
        "anchors": [a.model_dump() for a in store.get_anchors(sorted(anchor_ids))],
        "projections": projections,
    }


def _low_use_edges(store: StateStore, inv: Invariant) -> list[GraphEdge]:
    """Valid `use` edges support_node → constraint atom whose text == inv.text, ordered by edge_id.
    从支持节点指向文本等于 inv.text 的约束原子的有效 `use` 边，按 edge_id 排序。
    """
    found: dict[str, GraphEdge] = {}
    for node_id in inv.support_node_ids:
        for e in store.edges_touching([node_id], valid_only=True, edge_types=[EdgeType.use.value], direction="out"):
            dst = store.get_atom(e.dst_id)
            if dst is not None and dst.text == inv.text:
                found[e.edge_id] = e
    return [found[k] for k in sorted(found)]


def promote_skills(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    graph: SupportGraph,
    *,
    scope: Scope,
    at: str,
    anchor_id: str,
) -> list[str]:
    """Promote constraint invariants (≥2 support nodes) of valid InterfaceEPs to L3_macro skills with distill edges.
    将有效 InterfaceEP 中支持节点 ≥2 的约束不变量提升为 L3_macro 技能原子，并以可追溯的 distill 边相连。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index.
        index：向量索引。

        embedder: Embedder for the skill atoms.
        embedder：技能原子的嵌入器。

        graph: The SupportGraph.
        graph：SupportGraph。

        scope: Scope whose valid InterfaceEPs are scanned.
        scope：扫描其有效 InterfaceEP 的范围。

        at: Instant of the operation.
        at：操作时刻。

        anchor_id: Witness anchor of the skill atoms' history rows (scheduler/system anchor).
        anchor_id：技能原子历史行的见证锚点（调度器/系统锚点）。

    Output / 输出：
        Ids of the skill atoms created by this call (reused skills are not repeated).
        本次调用创建的技能原子 id（复用的技能不重复计入）。

    State effect / 状态影响：
        Writes skill atoms (+nodes, vectors, history), distill edges (witness_edge_id = lowest low use edge) + rows.
        写技能原子（+节点、向量、历史）、distill 边（witness_edge_id 为 id 最小的低层 use 边）与 edge_projections。
    """
    created: list[str] = []
    with store.transaction():
        for iep in store.list_interface_eps(scope, valid_only=True):
            if not iep.atom_id:
                continue
            for inv in iep.invariants:
                if str(inv.kind) != InvariantKind.constraint.value or len(inv.support_node_ids) < 2:
                    continue
                text = f"Skill: {inv.text}"
                existing = [
                    a
                    for a in store.find_atoms_by_hash(iep.scope, md5_hex(text), valid_only=True)
                    if str(a.source_kind) == SourceKind.skill.value
                ]
                if existing:
                    skill = sorted(existing, key=lambda a: a.atom_id)[0]
                else:
                    atom = MemoryAtom(
                        atom_type=AtomType.state,
                        content={
                            "text": text,
                            "kind": "skill",
                            "invariant": inv.model_dump(),
                            "source_interface_ep_ids": [iep.interface_ep_id],
                            "support_node_ids": list(inv.support_node_ids),
                        },
                        text=text,
                        time_anchor=None,
                        topic_tags=["skill"] + topic_tags(inv.text),
                        scale_level=Scale.L3_macro,
                        role=Role.extracted,
                        subject="project",
                        predicate="skill",
                        source_kind=SourceKind.skill,
                        source_id=iep.interface_ep_id,
                        confidence=float(inv.confidence),
                        extracted_by="rule@skill-v1",
                        visibility=str(iep.slots.kappa.visibility),
                        scope=iep.scope,
                        anchor_ids=list(inv.anchor_ids),
                    )
                    skill = create_atom(store, index, embedder, atom, history_anchor_id=anchor_id)
                    created.append(skill.atom_id)
                low = _low_use_edges(store, inv)
                distill_id = graph.insert_edge(
                    src_id=iep.atom_id,
                    dst_id=skill.atom_id,
                    edge_type=EdgeType.distill.value,
                    witness_anchor_ids=list(inv.anchor_ids),
                    created_by="consolidation",
                    proposer="rule",
                    confidence=0.9,
                    label=f"skill: {inv.text[:40]}",
                    scale=Scale.L3_macro.value,
                    is_boundary=False,
                    witness_edge_id=low[0].edge_id if low else None,
                )
                for e in low[1:]:
                    store.add_projection(distill_id, e.edge_id)
    return created


def round_trip_check(graph: SupportGraph, iep: InterfaceEP) -> bool:
    """Round-trip law: the boundary edges returned by refine equal boundary(support_node_ids).
    往返定律：refine 返回的边界边等于 boundary(support_node_ids)。

    Input / 输入：
        graph: The SupportGraph.
        graph：SupportGraph。

        iep: A persisted InterfaceEP.
        iep：已持久化的 InterfaceEP。

    Output / 输出：
        True iff set(refine(...)['boundary_edges'] ids) == set(boundary(support) ids).
        当且仅当 set(refine(...)['boundary_edges'] 的 id) == set(boundary(support) 的 id) 时为 True。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    refined = refine(graph.store, graph, iep.interface_ep_id)
    left = {b["edge_id"] for b in refined["boundary_edges"]}
    right = {b.edge_id for b in graph.boundary(iep.support_node_ids)}
    return left == right
