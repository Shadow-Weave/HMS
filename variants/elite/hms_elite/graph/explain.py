"""The `why()` core: target resolution, backward BFS over witnessed edges, on-demand refine and chain rendering.
`why()` 核心：目标解析、沿带见证边的反向 BFS、按需细化与链式渲染。
"""

# Purpose (EN): Explain an atom / InterfaceEP / edge by its inheritance path (§14.3): nodes, edges, anchors, chain.
# 目的（中）：通过继承路径解释原子 / InterfaceEP / 边（§14.3）：节点、边、锚点、路径、链。

# Input (EN): A StateStore, a SupportGraph, a target id or a resolved target dict, hop/edge-type options.
# 输入（中）：StateStore、SupportGraph、目标 id 或已解析的目标字典、层数/边类型选项。

# Output (EN): The §17.3 `why` dict and deterministic chain lines.
# 输出（中）：§17.3 的 `why` 字典与确定性的链式文本行。

# State effect (EN): None — read-only (refine is called with hops=0 for InterfaceEP nodes).
# 状态影响（中）：无 —— 只读（对 InterfaceEP 节点以 hops=0 调用 refine）。

from __future__ import annotations

from typing import Iterable

from hms_elite.enums import WHY_EDGE_TYPES, EdgeType, NodeKind
from hms_elite.graph.operators import refine
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.memory.storage import StateStore
from hms_elite.models import GraphEdge, MemoryAtom


def _atom_target(atom: MemoryAtom, resolved_from: str) -> dict:
    """Target dict of an atom.
    原子的目标字典。
    """
    return {
        "kind": "atom",
        "id": atom.atom_id,
        "node_id": atom.atom_id,
        "text": atom.text,
        "resolved_from": resolved_from,
    }


def resolve_target(store: StateStore, target_id: str) -> dict | None:
    """Resolve an id to a why() target: memory_id → atom_id → ep_id → obs_id → interface_ep_id → edge_id.
    将 id 解析为 why() 目标：memory_id → atom_id → ep_id → obs_id → interface_ep_id → edge_id。

    Input / 输入：
        store: The state store.
        store：状态存储。

        target_id: Any HMS object id.
        target_id：任意 HMS 对象 id。

    Output / 输出：
        {'kind': 'atom'|'interface_ep'|'edge', 'id', 'node_id'|'node_ids', 'text', 'resolved_from': 'id'} or None.
        {'kind': 'atom'|'interface_ep'|'edge', 'id', 'node_id'|'node_ids', 'text', 'resolved_from': 'id'} 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    tid = str(target_id or "").strip()
    if not tid:
        return None
    atom = store.resolve_head(tid) or store.latest_version(tid)
    if atom is not None:
        return _atom_target(atom, "id")
    atom = store.get_atom(tid)
    if atom is not None:
        return _atom_target(atom, "id")
    ep = store.get_ep(tid)
    if ep is not None and ep.atom_id:
        atom = store.get_atom(ep.atom_id)
        if atom is not None:
            return _atom_target(atom, "id")
    obs = store.get_observation(tid)
    if obs is not None and obs.atom_id:
        atom = store.get_atom(obs.atom_id)
        if atom is not None:
            return _atom_target(atom, "id")
    iep = store.get_interface_ep(tid)
    if iep is not None and iep.atom_id:
        return {
            "kind": "interface_ep",
            "id": iep.interface_ep_id,
            "node_id": iep.atom_id,
            "text": iep.slots.gloss(),
            "resolved_from": "id",
        }
    edge = store.get_edge(tid)
    if edge is not None:
        return {
            "kind": "edge",
            "id": edge.edge_id,
            "node_ids": [edge.src_id, edge.dst_id],
            "text": edge.label or f"{edge.edge_type}: {edge.src_id} → {edge.dst_id}",
            "resolved_from": "id",
        }
    return None


def _refs(store: StateStore, edge: GraphEdge) -> str:
    """Witness citation of an edge: source_ref or 'sha256:<12>' per witness anchor, in witness order.
    边的见证引用：每个见证锚点的 source_ref 或 'sha256:<12>'，按见证顺序。
    """
    anchors = {a.anchor_id: a for a in store.get_anchors(edge.witness_anchor_ids)}
    parts: list[str] = []
    for anchor_id in edge.witness_anchor_ids:
        anchor = anchors.get(anchor_id)
        if anchor is None:
            parts.append(anchor_id)
        else:
            parts.append(anchor.source_ref or "sha256:" + anchor.hash[:12])
    return ", ".join(parts)


def _text_of(store: StateStore, node_id: str) -> str:
    """Atom text of a node (the node id when the atom is unknown).
    节点的原子文本（原子未知时为节点 id）。
    """
    atom = store.get_atom(node_id)
    return atom.text if atom is not None else node_id


def render_chain(store: StateStore, target_node_id: str, edges: list[GraphEdge], *, hops: int) -> list[str]:
    """Render the deterministic chain: target text, then ≤hops steps along the strongest incoming (or `use`) edge.
    渲染确定性的解释链：目标文本，然后沿最强的入边（或出向 `use` 边）走至多 hops 步。

    Input / 输入：
        store: The state store (atom texts, anchors).
        store：状态存储（原子文本、锚点）。

        target_node_id: The chain start node.
        target_node_id：链的起点节点。

        edges: The explain edge set to walk.
        edges：可行走的 explain 边集合。

        hops: Maximum number of steps.
        hops：最大步数。

    Output / 输出：
        Lines: target text, then `<-{type}[{refs}]- {src_text}` / `-use[{refs}]-> {dst_text}`.
        文本行：目标文本，然后 `<-{type}[{refs}]- {src_text}` / `-use[{refs}]-> {dst_text}`。

    State effect / 状态影响：
        None.
        无。
    """
    lines = [_text_of(store, target_node_id)]
    current = target_node_id
    visited = {target_node_id}

    def _rank(e: GraphEdge) -> tuple[float, str, str]:
        """Highest confidence first; ties by created_at then edge_id.
        置信度最高优先；平局按 created_at、edge_id。
        """
        return (-float(e.confidence), e.created_at or "", e.edge_id)

    for _step in range(max(0, int(hops))):
        incoming = sorted((e for e in edges if e.dst_id == current and e.src_id not in visited), key=_rank)
        if incoming:
            e = incoming[0]
            lines.append(f"<-{e.edge_type}[{_refs(store, e)}]- {_text_of(store, e.src_id)}")
            current = e.src_id
        else:
            outgoing = sorted(
                (
                    e
                    for e in edges
                    if e.src_id == current and str(e.edge_type) == EdgeType.use.value and e.dst_id not in visited
                ),
                key=_rank,
            )
            if not outgoing:
                break
            e = outgoing[0]
            lines.append(f"-use[{_refs(store, e)}]-> {_text_of(store, e.dst_id)}")
            current = e.dst_id
        visited.add(current)
    return lines


def explain(
    store: StateStore,
    graph: SupportGraph,
    target: dict,
    *,
    hops: int = 2,
    edge_types: Iterable[str] | None = None,
    include_invalid: bool = False,
    max_nodes: int = 60,
) -> dict:
    """Backward BFS from a resolved target over incoming typed edges and outgoing `use` edges, refining InterfaceEPs.
    从已解析目标沿入向类型化边与出向 `use` 边反向 BFS，途中对 InterfaceEP 节点按需细化。

    Input / 输入：
        store: The state store.
        store：状态存储。

        graph: The SupportGraph.
        graph：SupportGraph。

        target: A resolve_target() dict (kind atom | interface_ep | edge).
        target：resolve_target() 的字典（kind 为 atom | interface_ep | edge）。

        hops: Expansion levels (0 → the target only).
        hops：扩展层数（0 → 仅目标）。

        edge_types: Incoming edge types to follow (None → WHY_EDGE_TYPES).
        edge_types：跟随的入边类型（None → WHY_EDGE_TYPES）。

        include_invalid: Also follow invalid edges.
        include_invalid：是否也沿失效边扩展。

        max_nodes: Hard cap on visited nodes.
        max_nodes：访问节点数上限。

    Output / 输出：
        Keys: target, nodes, edges, anchors, interface_eps, paths, unresolved_conflicts, refined_from, chain.
        键：target、nodes、edges、anchors、interface_eps、paths、unresolved_conflicts、refined_from、chain、explanation。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    types = sorted(str(t) for t in (edge_types or WHY_EDGE_TYPES))
    valid_only = not include_invalid
    if target.get("kind") == "edge":
        seeds = [str(n) for n in target.get("node_ids") or []]
        chain_start = seeds[-1] if seeds else ""
    else:
        seeds = [str(target.get("node_id") or "")]
        chain_start = seeds[0]
    seeds = [s for s in dict.fromkeys(seeds) if s]
    paths: dict[str, list[str]] = {s: [] for s in seeds}
    order: list[str] = list(seeds)
    edges: dict[str, GraphEdge] = {}

    def _visit(node_id: str, path: list[str]) -> bool:
        """Register a node with its path; False when the node cap is reached.
        登记节点及其路径；达到节点上限时返回 False。
        """
        if node_id in paths:
            return True
        if len(paths) >= max_nodes:
            return False
        paths[node_id] = path
        order.append(node_id)
        return True

    frontier = sorted(seeds)
    for _level in range(max(0, int(hops))):
        if not frontier:
            break
        next_frontier: list[str] = []
        for node_id in frontier:
            incoming = store.edges_touching([node_id], valid_only=valid_only, edge_types=types, direction="in")
            for e in incoming:
                if _visit(e.src_id, paths[node_id] + [e.edge_id]):
                    edges[e.edge_id] = e
                    if (
                        e.src_id not in frontier
                        and e.src_id not in next_frontier
                        and paths[e.src_id] == paths[node_id] + [e.edge_id]
                    ):
                        next_frontier.append(e.src_id)
            outgoing = store.edges_touching(
                [node_id], valid_only=valid_only, edge_types=[EdgeType.use.value], direction="out"
            )
            for e in outgoing:
                if _visit(e.dst_id, paths[node_id] + [e.edge_id]):
                    edges[e.edge_id] = e
                    if (
                        e.dst_id not in frontier
                        and e.dst_id not in next_frontier
                        and paths[e.dst_id] == paths[node_id] + [e.edge_id]
                    ):
                        next_frontier.append(e.dst_id)
        frontier = sorted(set(next_frontier))

    # Refine every InterfaceEP node visited (support nodes/edges/boundary join the result, not the frontier).
    # 对访问到的每个 InterfaceEP 节点做细化（支撑节点/边/边界并入结果，但不进入前沿）。
    refined_from: list[str] = []
    interface_eps: list[dict] = []
    pending = list(order)
    seen_nodes: set[str] = set()
    while pending:
        node_id = pending.pop(0)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        node = store.get_node(node_id)
        if node is None or str(node.node_kind) != NodeKind.interface_ep.value:
            continue
        iep = store.get_interface_ep(node.ref_id)
        if iep is None or iep.interface_ep_id in refined_from or (valid_only and not iep.valid):
            continue
        sub = refine(store, graph, iep.interface_ep_id, hops=0)
        refined_from.append(iep.interface_ep_id)
        interface_eps.append(sub["interface_ep"])
        for n in sub["nodes"]:
            if _visit(n["node_id"], list(paths[node_id])):
                pending.append(n["node_id"])
        for e in sub["edges"]:
            if e["src_id"] in paths and e["dst_id"] in paths:
                edges.setdefault(e["edge_id"], GraphEdge(**e))
        for b in sub["boundary_edges"]:
            edge = store.get_edge(b["edge_id"])
            if edge is None or (valid_only and not edge.valid):
                continue
            outer = b["outer_node_id"]
            if _visit(outer, list(paths[node_id]) + [edge.edge_id]):
                edges.setdefault(edge.edge_id, edge)
                pending.append(outer)

    kept = sorted(
        (e for e in edges.values() if e.src_id in paths and e.dst_id in paths),
        key=lambda e: (e.created_at or "", e.edge_id),
    )
    node_map = {n.node_id: n for n in store.get_nodes(order)}
    atom_map = {a.atom_id: a for a in store.get_atoms(order)}
    nodes: list[dict] = []
    anchor_ids: set[str] = set()
    for node_id in order:
        node = node_map.get(node_id)
        if node is None:
            continue
        entry = node.model_dump()
        atom = atom_map.get(node_id)
        entry["text"] = atom.text if atom is not None else ""
        nodes.append(entry)
        anchor_ids.update(store.atom_anchor_ids(node_id))
    for e in kept:
        anchor_ids.update(e.witness_anchor_ids)
    anchors = [a.model_dump() for a in store.get_anchors(sorted(anchor_ids))]
    unresolved = [e.edge_id for e in kept if str(e.edge_type) == EdgeType.conflict.value and e.resolved_at is None]
    chain = render_chain(store, chain_start, kept, hops=hops) if chain_start else []
    return {
        "target": dict(target),
        "nodes": nodes,
        "edges": [e.model_dump() for e in kept],
        "anchors": anchors,
        "interface_eps": interface_eps,
        "paths": {node_id: list(path) for node_id, path in paths.items() if node_id in node_map},
        "unresolved_conflicts": unresolved,
        "refined_from": refined_from,
        "chain": chain,
        "explanation": "\n".join(chain),
    }
