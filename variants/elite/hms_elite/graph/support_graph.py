"""SupportGraph: witnessed typed edges, BFS neighbourhoods, boundary detection, cohesion, components and stats.
SupportGraph：带 witness 的类型化边、BFS 邻域、边界检测、内聚度、连通分量与统计。
"""

# Purpose (EN): The Support Graph plane's read/write facade over StateStore (§14.1): witnessed edges, S/E_in/E_b.
# 目的（中）：支持图平面基于 StateStore 的读写门面（§14.1）：带见证的边、S/E_in/E_b 结构。

# Input (EN): Node id sets, edge specs, scopes and a GraphConfig.
# 输入（中）：节点 id 集合、边规格、范围与 GraphConfig。

# Output (EN): Edge ids, GraphNode/GraphEdge lists, BoundaryEdges, cohesion scores, components and stats dicts.
# 输出（中）：边 id、GraphNode/GraphEdge 列表、BoundaryEdge、内聚度、连通分量与统计字典。

# State effect (EN): insert_edge writes graph_edges/edge_witnesses (+edge_projections by trigger); the rest reads.
# 状态影响（中）：insert_edge 经 StateStore 写 graph_edges/edge_witnesses（触发器写 edge_projections）；其余只读。

from __future__ import annotations

from typing import Iterable, Literal

from hms_elite.configs.base import GraphConfig
from hms_elite.enums import EdgeType, NodeKind, Scale
from hms_elite.exceptions import ValidationError, WitnessMissingError
from hms_elite.memory.storage import StateStore
from hms_elite.models import BoundaryEdge, GraphEdge, GraphNode, InterfaceEP, Scope
from hms_elite.timeutil import age_days

_HIGH_SCALES = {Scale.L2_meso.value, Scale.L3_macro.value}


def _time_of(atom) -> str:
    """Sort key of an atom: time_anchor, else created_at, else ''.
    原子的排序键：time_anchor，否则 created_at，否则空串。
    """
    if atom is None:
        return ""
    return atom.time_anchor or atom.created_at or ""


class SupportGraph:
    """Support Graph facade: witnessed edges, neighbourhoods, boundaries, cohesion, components and stats over a store.
    支持图门面：基于 StateStore 的带见证边、邻域、边界、内聚度、连通分量与统计。
    """

    def __init__(self, store: StateStore, config: GraphConfig) -> None:
        """Bind the graph to a store and its graph configuration.
        将图绑定到存储及其图配置。

        Input / 输入：
            store: The shared StateStore (the only SQL seam).
            store：共享的 StateStore（唯一的 SQL 接缝）。

            config: GraphConfig (cohesion threshold, coarsen sizes/gaps, policies).
            config：GraphConfig（内聚度阈值、粗化规模/间隔、策略）。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        self.store = store
        self.config = config

    # ------------------------------------------------------------------ writes
    # ------------------------------------------------------------------ 写入
    def insert_edge(
        self,
        *,
        src_id: str,
        dst_id: str,
        edge_type: str,
        witness_anchor_ids: list[str],
        created_by: str,
        proposer: str = "rule",
        confidence: float = 0.7,
        label: str = "",
        scale: str = "L1_micro",
        is_boundary: bool = False,
        witness_edge_id: str | None = None,
    ) -> str:
        """Insert one typed edge with at least one witness anchor.
        插入一条至少带一个 witness 的类型化边。

        Input / 输入：
            src_id: Source node id (its scope becomes the edge scope).
            src_id：源节点 id（其范围成为边的范围）。

            dst_id: Target node id.
            dst_id：目标节点 id。

            edge_type: One of the six EdgeType values.
            edge_type：六种 EdgeType 值之一。

            witness_anchor_ids: Witness anchors; an empty list raises WitnessMissingError before any model or SQL.
            witness_anchor_ids：见证锚点；空列表在构造模型或任何 SQL 之前抛出 WitnessMissingError。

            created_by: 'ingest' | 'consolidation' | 'feedback' | 'ops' | 'project'.
            created_by：'ingest' | 'consolidation' | 'feedback' | 'ops' | 'project'。

            proposer: Proposer tag ('rule', 'closure', 'timestamp', 'llm:<identity>', 'project-v1').
            proposer：提议者标签（'rule'、'closure'、'timestamp'、'llm:<identity>'、'project-v1'）。

            confidence: Edge confidence in [0, 1].
            confidence：边置信度，范围 [0, 1]。

            label: Free-text label.
            label：自由文本标签。

            scale: Edge scale ('L1_micro' | 'L2_meso' | 'L3_macro').
            scale：边尺度（'L1_micro' | 'L2_meso' | 'L3_macro'）。

            is_boundary: True only for projected edges (requires witness_edge_id).
            is_boundary：仅投影边为 True（需要 witness_edge_id）。

            witness_edge_id: The low-level edge a projected edge witnesses.
            witness_edge_id：投影边所见证的低层边。

        Output / 输出：
            The (existing or new) edge id; MERGE semantics on (src, dst, type, scale).
            （已存在或新建的）边 id；按 (src, dst, type, scale) MERGE。

        State effect / 状态影响：
            Writes graph_edges/edge_witnesses (+edge_projections by trigger) through StateStore.insert_edge.
            经 StateStore.insert_edge 写 graph_edges/edge_witnesses（触发器写 edge_projections）。
        """
        if not witness_anchor_ids:
            raise WitnessMissingError(
                "edge has no witness anchors", details={"object": "edge", "id": None, "missing": []}
            )
        src = self.store.get_node(str(src_id))
        if src is None:
            raise ValidationError(f"unknown src node {src_id!r}", "NOT_FOUND_001", details={"src_id": src_id})
        edge = GraphEdge(
            src_id=str(src_id),
            dst_id=str(dst_id),
            edge_type=str(edge_type),
            label=label,
            scale=str(scale),
            is_boundary=bool(is_boundary),
            confidence=float(confidence),
            witness_anchor_ids=[str(a) for a in witness_anchor_ids],
            witness_edge_id=witness_edge_id,
            created_by=str(created_by),
            proposer=str(proposer),
            scope=src.scope,
            created_at=self.store.now(),
        )
        return self.store.insert_edge(edge)

    # ------------------------------------------------------------------ neighbourhoods
    # ------------------------------------------------------------------ 邻域
    def neighbors(
        self,
        node_ids: Iterable[str],
        *,
        hops: int = 1,
        edge_types: Iterable[str] | None = None,
        valid_only: bool = True,
        direction: Literal["both", "in", "out"] = "both",
        max_nodes: int = 60,
    ) -> tuple[list[GraphNode], list[GraphEdge], dict[str, list[str]]]:
        """Deterministic BFS over typed edges from a seed set.
        从种子集合出发、沿类型化边的确定性 BFS。

        Input / 输入：
            node_ids: Seed node ids.
            node_ids：种子节点 id。

            hops: Number of expansion levels (0 → seeds only).
            hops：扩展层数（0 → 仅种子）。

            edge_types: Edge type whitelist (None → all six).
            edge_types：边类型白名单（None → 全部六种）。

            valid_only: Follow valid edges only (nodes reached keep their own `valid` flag).
            valid_only：仅沿有效边扩展（到达的节点保留各自的 `valid` 标志）。

            direction: 'both' | 'in' (follow edges into the frontier) | 'out' (follow edges out of the frontier).
            direction：'both' | 'in'（沿指向前沿的边）| 'out'（沿离开前沿的边）。

            max_nodes: Hard cap on the number of nodes returned (seeds included).
            max_nodes：返回节点数量上限（含种子）。

        Output / 输出：
            (nodes incl. seeds in discovery order, edges by (created_at, edge_id), paths node_id → [edge_id, ...]).
            （含种子、按发现顺序的节点；其间按 (created_at, edge_id) 排序的边；paths：node_id → 从最近种子出发的 [edge_id, ...]）。

        State effect / 状态影响：
            None.
            无。
        """
        seeds = sorted(dict.fromkeys(str(i) for i in node_ids))
        types = sorted(str(t) for t in edge_types) if edge_types else None
        paths: dict[str, list[str]] = {s: [] for s in seeds}
        order: list[str] = list(seeds)
        edges: dict[str, GraphEdge] = {}
        frontier = list(seeds)
        for _level in range(max(0, int(hops))):
            if not frontier:
                break
            frontier_set = set(frontier)
            touching = self.store.edges_touching(frontier, valid_only=valid_only, edge_types=types, direction=direction)
            next_frontier: list[str] = []
            for e in touching:
                if direction == "out":
                    if e.src_id not in frontier_set:
                        continue
                    here, other = e.src_id, e.dst_id
                elif direction == "in":
                    if e.dst_id not in frontier_set:
                        continue
                    here, other = e.dst_id, e.src_id
                elif e.src_id in frontier_set:
                    here, other = e.src_id, e.dst_id
                elif e.dst_id in frontier_set:
                    here, other = e.dst_id, e.src_id
                else:
                    continue
                if other not in paths:
                    if len(paths) >= max_nodes:
                        continue
                    paths[other] = paths[here] + [e.edge_id]
                    order.append(other)
                    next_frontier.append(other)
                edges[e.edge_id] = e
            frontier = sorted(set(next_frontier))
        found = {n.node_id: n for n in self.store.get_nodes(order)}
        nodes = [found[i] for i in order if i in found]
        kept = sorted(
            (e for e in edges.values() if e.src_id in paths and e.dst_id in paths),
            key=lambda e: (e.created_at or "", e.edge_id),
        )
        return nodes, kept, paths

    def internal_edges(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[GraphEdge]:
        """E_in: edges whose both endpoints lie in the node set.
        E_in：两个端点都在节点集合内的边。

        Input / 输入：
            node_ids: The support set S.
            node_ids：支持集 S。

            valid_only: Keep valid edges only.
            valid_only：仅保留有效边。

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
        return self.store.edges_between(ids, valid_only=valid_only)

    def boundary(self, node_ids: Iterable[str], *, valid_only: bool = True) -> list[BoundaryEdge]:
        """E_b: edges with exactly one endpoint in the node set ('out' when the source is inside).
        E_b：恰有一个端点在节点集合内的边（源在集合内时方向为 'out'）。

        Input / 输入：
            node_ids: The support set S.
            node_ids：支持集 S。

            valid_only: Keep valid edges only.
            valid_only：仅保留有效边。

        Output / 输出：
            BoundaryEdges ordered like edges_touching ((created_at, edge_id)).
            按 edges_touching 顺序（(created_at, edge_id)）的 BoundaryEdge。

        State effect / 状态影响：
            None.
            无。
        """
        ids = set(str(i) for i in node_ids)
        if not ids:
            return []
        out: list[BoundaryEdge] = []
        for e in self.store.edges_touching(ids, valid_only=valid_only, direction="both"):
            in_src, in_dst = e.src_id in ids, e.dst_id in ids
            if in_src == in_dst:
                continue
            out.append(
                BoundaryEdge(
                    edge_id=e.edge_id,
                    inner_node_id=e.src_id if in_src else e.dst_id,
                    outer_node_id=e.dst_id if in_src else e.src_id,
                    direction="out" if in_src else "in",
                    edge_type=e.edge_type,
                    witness_anchor_ids=list(e.witness_anchor_ids),
                )
            )
        return out

    def cohesion(self, node_ids: Iterable[str]) -> float:
        """Cohesion of a node set: 0.5·min(1, |E_in|/max(1, |S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|), 0.0 when |E_in| == 0.
        节点集合的内聚度：0.5·min(1, |E_in|/max(1, |S|−1)) + 0.5·|E_in|/(|E_in|+|E_b|)，|E_in| == 0 时为 0.0。

        Input / 输入：
            node_ids: The support set S.
            node_ids：支持集 S。

        Output / 输出：
            Unrounded cohesion in [0, 1].
            未取整的内聚度，范围 [0, 1]。

        State effect / 状态影响：
            None.
            无。
        """
        ids = list(dict.fromkeys(str(i) for i in node_ids))
        e_in = self.internal_edges(ids)
        if not e_in:
            return 0.0
        e_b = self.boundary(ids)
        n_in, n_b = len(e_in), len(e_b)
        return 0.5 * min(1.0, n_in / max(1, len(ids) - 1)) + 0.5 * n_in / (n_in + n_b)

    # ------------------------------------------------------------------ components
    # ------------------------------------------------------------------ 连通分量
    def components(
        self,
        scope: Scope,
        *,
        kinds: Iterable[str] = ("ep",),
        edge_types: Iterable[str] = ("causal", "use", "temporal"),
        gap_hours: float = 48.0,
        min_size: int = 2,
        max_size: int = 12,
        scale: str = "L1_micro",
    ) -> list[list[str]]:
        """Coarsen candidates: connected components of valid nodes of the given kinds, split by time gap and topics.
        粗化候选：给定种类有效节点的连通分量，按时间间隔与主题共享拆分。

        Input / 输入：
            scope: Scope filter.
            scope：范围过滤。

            kinds: Node kinds ('ep' for L2 coarsening, 'interface_ep' for L3).
            kinds：节点种类（L2 粗化为 'ep'，L3 为 'interface_ep'）。

            edge_types: Edge types that connect members.
            edge_types：连接成员的边类型。

            gap_hours: Consecutive members (by time_anchor) further apart than this start a new component.
            gap_hours：按 time_anchor 相邻成员的间隔超过该值时切分为新分量。

            min_size: Components smaller than this are dropped.
            min_size：小于该规模的分量被丢弃。

            max_size: Larger components keep only their newest max_size members.
            max_size：更大的分量仅保留最新的 max_size 个成员。

            scale: graph_nodes.scale of the members (and of the connecting edges).
            scale：成员（及连接边）的 graph_nodes.scale。

        Output / 输出：
            Lists of node ids, each sorted by time_anchor; the list ordered by (first member time, first member id).
            节点 id 列表，每个按 time_anchor 排序；整体按（首成员时间、首成员 id）排序。

        State effect / 状态影响：
            None.
            无。
        """
        nodes = self.store.list_nodes(scope, kinds=[str(k) for k in kinds], valid_only=True, scale=str(scale))
        ids = [n.node_id for n in nodes]
        if not ids:
            return []
        atoms = {a.atom_id: a for a in self.store.get_atoms(ids)}
        types = {str(t) for t in edge_types}
        parent: dict[str, str] = {i: i for i in ids}

        def _find(x: str) -> str:
            """Union-find root with path compression.
            带路径压缩的并查集查根。
            """
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self.store.edges_between(ids, valid_only=True):
            if e.edge_type not in types or e.scale != str(scale):
                continue
            ra, rb = _find(e.src_id), _find(e.dst_id)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        groups: dict[str, list[str]] = {}
        for i in ids:
            groups.setdefault(_find(i), []).append(i)

        def _key(i: str) -> tuple[str, str]:
            """(time, id) ordering key of a member.
            成员的 (time, id) 排序键。
            """
            return (_time_of(atoms.get(i)), i)

        result: list[list[str]] = []
        for members in groups.values():
            members = sorted(members, key=_key)
            segments: list[list[str]] = []
            current: list[str] = []
            for i in members:
                if current:
                    prev_t, cur_t = _time_of(atoms.get(current[-1])), _time_of(atoms.get(i))
                    if prev_t and cur_t and age_days(prev_t, cur_t) * 24.0 > gap_hours:
                        segments.append(current)
                        current = []
                current.append(i)
            if current:
                segments.append(current)
            for segment in segments:
                topics = {i: set(atoms[i].topic_tags) if i in atoms else set() for i in segment}
                kept = [i for i in segment if any(topics[i] & topics[j] for j in segment if j != i)]
                if len(kept) < min_size:
                    continue
                if len(kept) > max_size:
                    kept = kept[-max_size:]
                result.append(kept)
        return sorted(result, key=lambda comp: _key(comp[0]))

    def owner_interface_ep(self, node_id: str, *, scale: str) -> InterfaceEP | None:
        """The valid InterfaceEP of a scale whose support contains the node.
        支持集包含该节点的、给定尺度的有效 InterfaceEP。

        Input / 输入：
            node_id: A node id.
            node_id：节点 id。

            scale: 'L2_meso' | 'L3_macro'.
            scale：'L2_meso' | 'L3_macro'。

        Output / 输出：
            The newest matching InterfaceEP, or None.
            最新的匹配 InterfaceEP，或 None。

        State effect / 状态影响：
            None.
            无。
        """
        owners = [
            i for i in self.store.interface_eps_containing([str(node_id)], valid_only=True) if i.scale == str(scale)
        ]
        return owners[-1] if owners else None

    # ------------------------------------------------------------------ stats
    # ------------------------------------------------------------------ 统计
    def stats(self, scope: Scope | None) -> dict:
        """Graph statistics for a scope (None → whole DB): node/edge counts, validity split, coverage ratios.
        某范围（None → 整库）的图统计：节点/边计数、有效性划分、覆盖率。

        Input / 输入：
            scope: Scope filter or None for the whole database.
            scope：范围过滤，None 表示整库。

        Output / 输出：
            Keys: nodes, edges, valid_edges, invalid_edges, boundary_edges, interface_eps, unresolved_conflicts, ratios.
            键：nodes、edges、valid_edges、invalid_edges、boundary_edges、interface_eps、unresolved_conflicts 与覆盖率。

        State effect / 状态影响：
            None.
            无。
        """
        filt = scope if scope is not None else Scope()
        nodes = self.store.list_nodes(filt, valid_only=False)
        edges = self.store.list_edges(filt, valid_only=False)
        node_counts = {kind: 0 for kind in NodeKind.values()}
        for n in nodes:
            node_counts[str(n.node_kind)] = node_counts.get(str(n.node_kind), 0) + 1
        edge_counts = {t: 0 for t in EdgeType.values()}
        for e in edges:
            edge_counts[str(e.edge_type)] = edge_counts.get(str(e.edge_type), 0) + 1
        valid_edges = [e for e in edges if e.valid]
        boundary_edges = [e for e in edges if e.is_boundary]
        high_edges = [e for e in edges if str(e.scale) in _HIGH_SCALES]
        witnessed = sum(1 for e in edges if e.witness_anchor_ids)
        projected = sum(1 for e in boundary_edges if self.store.projections_of(e.edge_id))
        traceable = sum(1 for e in high_edges if self.store.projections_of(e.edge_id))
        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "valid_edges": len(valid_edges),
            "invalid_edges": len(edges) - len(valid_edges),
            "boundary_edges": len(boundary_edges),
            "interface_eps": len(self.store.list_interface_eps(filt, valid_only=True)),
            "unresolved_conflicts": sum(
                1 for e in valid_edges if str(e.edge_type) == EdgeType.conflict.value and e.resolved_at is None
            ),
            "witness_coverage": (witnessed / len(edges)) if edges else 1.0,
            "projection_coverage": (projected / len(boundary_edges)) if boundary_edges else 1.0,
            "traceable_high_edges": (traceable / len(high_edges)) if high_edges else 1.0,
            "transitions": int(self.store.stats(scope).get("transitions", 0)),
        }
