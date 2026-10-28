"""Handoff: build a self-contained StatePackage (context + PC-IR + evidence closure) and import one on a receiver.
交接：构建自包含的 StatePackage（上下文 + PC-IR + 证据闭包），并在接收方导入。
"""

# Purpose (EN): HMS §9.4 — the receiver imports the state package instead of reading history (§16.3).
# 目的（中）：HMS §9.4——接手方导入状态包而不是阅读历史（§16.3）。

# Input (EN): A scope, filters, goal and budget (build); a StatePackage and an optional scope override (import).
# 输入（中）：范围、过滤器、目标与预算（构建）；StatePackage 与可选的范围覆盖（导入）。

# Output (EN): A StatePackage, or the import report {'imported', 'skipped', 'id_map', 'anchor_id'}.
# 输出（中）：StatePackage，或导入报告 {'imported', 'skipped', 'id_map', 'anchor_id'}。

# State effect (EN): build is read-only; import writes events, anchors, eps, atoms, InterfaceEPs, edges and transitions
#                    in ONE transaction witnessed by a system anchor.
# 状态影响（中）：构建只读；导入在一个由系统锚点见证的事务内写入事件、锚点、EP、原子、InterfaceEP、边与转移。

from __future__ import annotations

import logging
from typing import Any

from hms_elite.compiler.pc_ir import lower
from hms_elite.configs.base import HMSConfig
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import SourceKind
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.ingest.atoms import create_atom
from hms_elite.ingest.events import system_event
from hms_elite.memory.storage import StateStore
from hms_elite.models import EP, Anchor, GraphEdge, InterfaceEP, Scope, SourceEvent, StatePackage, Transition
from hms_elite.ops.transitions import interface_ep_window, permission_scope_of

logger = logging.getLogger(__name__)

_KINDS = ("source_events", "anchors", "eps", "atoms", "interface_eps", "edges", "transitions")


def build_state_package(
    store: StateStore,
    graph: SupportGraph,
    compiler: Any,
    *,
    config: HMSConfig,
    scope: Scope,
    filters: dict,
    goal: str | None,
    budget_tokens: int,
    checkpoint_id: str | None,
    include_edges: bool = True,
) -> StatePackage:
    """Compile the deep Work State and gather its witness closure into a StatePackage.
    编译 deep 工作态并收集其见证闭包，构成 StatePackage。

    Input / 输入：
        store: The state store.
        store：状态存储。

        graph: The SupportGraph.
        graph：SupportGraph。

        compiler: The StateCompiler (compile(query, scope=..., filters=..., budget_tokens=..., mode='deep')).
        compiler：StateCompiler（compile(query, scope=..., filters=..., budget_tokens=..., mode='deep')）。

        config: HMSConfig (compiler.require_witness gates the PC-IR lowering).
        config：HMSConfig（compiler.require_witness 控制 PC-IR 降低）。

        scope: The handoff scope.
        scope：交接范围。

        filters: Mem0 filter dict passed to the compiler (scope keys + optional audience).
        filters：传给编译器的 Mem0 过滤字典（范围键 + 可选受众）。

        goal: Explicit goal; None → the newest valid EP's `u` in the scope's run, else None.
        goal：显式目标；None → 该范围运行中最新有效 EP 的 `u`，否则 None。

        budget_tokens: Token budget of the compiled context.
        budget_tokens：编译上下文的 token 预算。

        checkpoint_id: Checkpoint taken by the facade before the handoff (recorded, not created here).
        checkpoint_id：门面在交接前创建的检查点（仅记录，不在此创建）。

        include_edges: Include the valid edges among the selected nodes.
        include_edges：是否包含所选节点之间的有效边。

    Output / 输出：
        A StatePackage whose anchors/source_events close over every reference.
        锚点/源事件闭合覆盖全部引用的 StatePackage。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    if goal is None:
        last = store.last_ep(scope, run_id=scope.run_id)
        goal = last.slots.u if last is not None else None
    ir = compiler.compile(goal or "handoff", scope=scope, filters=filters, budget_tokens=budget_tokens, mode="deep")
    pcir = lower(ir, require_witness=bool(config.compiler.require_witness))

    item_atom_ids = sorted({item.atom_id for item in ir.items() if item.atom_id})
    selected_atoms = {a.atom_id: a for a in store.get_atoms(item_atom_ids)}

    ep_ids: list[str] = []
    iep_ids: list[str] = []
    for item in ir.episodes:
        atom = selected_atoms.get(item.atom_id or "")
        if atom is None:
            continue
        if str(atom.source_kind) == SourceKind.ep.value:
            ep_ids.append(atom.source_id)
        elif str(atom.source_kind) == SourceKind.interface_ep.value:
            iep_ids.append(atom.source_id)
            ep_ids.extend(atom.source_ep_ids)
    eps = [ep for ep in store.get_eps(list(dict.fromkeys(ep_ids)))]
    eps.sort(key=lambda e: e.ep_id)

    ieps: dict[str, InterfaceEP] = {}
    for iep_id in iep_ids:
        iep = store.get_interface_ep(iep_id)
        if iep is not None and iep.valid:
            ieps[iep.interface_ep_id] = iep
    for iep in store.interface_eps_containing(item_atom_ids, valid_only=True):
        ieps[iep.interface_ep_id] = iep
    interface_eps = [ieps[k] for k in sorted(ieps)]

    node_ids: set[str] = set(item_atom_ids)
    node_ids.update(ep.atom_id for ep in eps if ep.atom_id)
    for iep in interface_eps:
        if iep.atom_id:
            node_ids.add(iep.atom_id)
        node_ids.update(iep.support_node_ids)
    edges: list[GraphEdge] = []
    if include_edges:
        edges = sorted(store.edges_between(sorted(node_ids), valid_only=True), key=lambda e: e.edge_id)
        for edge in edges:
            node_ids.update((edge.src_id, edge.dst_id))
    atoms = sorted(store.get_atoms(sorted(node_ids)), key=lambda a: a.atom_id)

    anchor_ids: set[str] = set(ir.anchor_ids())
    for atom in atoms:
        anchor_ids.update(store.atom_anchor_ids(atom.atom_id, include_disputed=True) or atom.anchor_ids)
    for ep in eps:
        anchor_ids.update(ep.anchor_ids)
    for iep in interface_eps:
        anchor_ids.update(iep.evidence_anchor_ids)
        for inv in iep.invariants:
            anchor_ids.update(inv.anchor_ids)
        for res in iep.residuals:
            anchor_ids.update(res.anchor_ids)
    for edge in edges:
        anchor_ids.update(edge.witness_anchor_ids)
    anchors = store.get_anchors(sorted(anchor_ids))
    source_events: list[SourceEvent] = []
    seen_events: set[str] = set()
    for anchor in anchors:
        if anchor.source_event_id in seen_events:
            continue
        ev = store.get_source_event(anchor.source_event_id)
        if ev is not None:
            seen_events.add(ev.event_id)
            source_events.append(ev)
    return StatePackage(
        package_id=store.new_id("pkg"),
        created_at=store.now(),
        goal=goal,
        scope=scope,
        context=ir,
        pcir=pcir,
        atoms=atoms,
        eps=eps,
        interface_eps=interface_eps,
        edges=edges,
        anchors=anchors,
        source_events=source_events,
        checkpoint_id=checkpoint_id,
    )


class _Importer:
    """Stateful helper of import_state_package (id map, counters).
    import_state_package 的有状态辅助对象（id 映射、计数器）。
    """

    def __init__(
        self,
        store: StateStore,
        index: Any,
        embedder: Any,
        pkg: StatePackage,
        override: Scope | None,
        extractor: EntityExtractorBase | None = None,
    ):
        """Bind the receiver collaborators, the package and the receiver's entity extractor (rule when None).
        绑定接收方协作对象、交接包与接收方的实体抽取器（None 时为规则）。
        """
        self.store = store
        self.index = index
        self.embedder = embedder
        self.pkg = pkg
        self.override = override
        self.extractor = extractor
        self.id_map: dict[str, str] = {}
        self.imported = {k: 0 for k in _KINDS}
        self.skipped = {k: 0 for k in _KINDS}
        self.lineage_map: dict[str, str] = {}

    def fresh_id(self, prefix: str, exists: Any) -> str:
        """Mint an id of `prefix` that `exists(id)` does not know (preserved ids may collide with the receiver's counter).
        铸造一个 `exists(id)` 不认识的 `prefix` id（保留的 id 可能与接收方计数器冲突）。
        """
        while True:
            candidate = self.store.new_id(prefix)
            if not exists(candidate):
                return candidate

    def map_id(self, old: str) -> str:
        """Mapped id (identity when not remapped).
        映射后的 id（未重映射时为其自身）。
        """
        return self.id_map.get(old, old)

    def map_ids(self, ids: list[str]) -> list[str]:
        """Mapped id list, duplicates dropped in first-occurrence order.
        映射后的 id 列表，按首次出现顺序去重。
        """
        return list(dict.fromkeys(self.map_id(i) for i in ids))

    def scope_of(self, scope: Scope) -> Scope:
        """The receiver scope of an object.
        对象在接收方的范围。
        """
        return self.override if self.override is not None else scope

    def _map_kappa(self, kappa: dict) -> dict:
        """Kappa dict with mapped evidence ids.
        证据 id 已映射的 Kappa 字典。
        """
        out = dict(kappa)
        if isinstance(out.get("evidence"), list):
            out["evidence"] = self.map_ids([str(a) for a in out["evidence"]])
        return out

    # ------------------------------------------------------------------ steps
    # ------------------------------------------------------------------ 步骤
    def source_events(self) -> None:
        """(1) insert every source event (identity dedup; id collisions with a different identity are remapped).
        (1) 插入每个源事件（身份去重；与不同身份的 id 冲突则重映射）。
        """
        for ev in self.pkg.source_events:
            candidate = ev.model_copy(update={"scope": self.scope_of(ev.scope)})
            existing = self.store.get_source_event(ev.event_id) if ev.event_id else None
            if existing is not None:
                same = (
                    existing.payload_hash == candidate.payload_hash
                    and existing.timestamp == candidate.timestamp
                    and existing.scope.key() == candidate.scope.key()
                )
                if same:
                    self.id_map[ev.event_id] = existing.event_id
                    self.skipped["source_events"] += 1
                    continue
                fresh = self.fresh_id("evt", lambda i: self.store.get_source_event(i) is not None)
                candidate = candidate.model_copy(update={"event_id": fresh})
            stored, dedup = self.store.insert_source_event(candidate)
            self.id_map[ev.event_id] = stored.event_id
            if dedup:
                self.skipped["source_events"] += 1
            else:
                self.imported["source_events"] += 1

    def anchors(self) -> None:
        """(2) one anchor per event through get_or_create_anchor; package anchor ids map to the receiver's.
        (2) 经 get_or_create_anchor 每事件一个锚点；交接包锚点 id 映射到接收方锚点。
        """
        for anchor in self.pkg.anchors:
            event_id = self.map_id(anchor.source_event_id)
            ev = self.store.get_source_event(event_id)
            if ev is None:
                logger.warning("import: anchor %s references an unknown event %s", anchor.anchor_id, event_id)
                self.skipped["anchors"] += 1
                continue
            existed = self.store.anchor_for_event(event_id)
            stored = self.store.get_or_create_anchor(ev)
            self.id_map[anchor.anchor_id] = stored.anchor_id
            if existed is not None:
                self.skipped["anchors"] += 1
            else:
                self.imported["anchors"] += 1
            if anchor.revoked_at is not None and stored.revoked_at is None:
                self.store.revoke_anchor(stored.anchor_id, at=anchor.revoked_at)

    def eps(self) -> list[EP]:
        """(3) insert EPs (existing id → skip, or remap under a scope override).
        (3) 插入 EP（已存在的 id → 跳过；范围覆盖时重映射）。
        """
        written: list[EP] = []
        for ep in sorted(self.pkg.eps, key=lambda e: e.ep_id):
            existing = self.store.get_ep(ep.ep_id) if ep.ep_id else None
            ep_id = ep.ep_id
            if existing is not None:
                same = (existing.slots.u, existing.slots.a, existing.slots.y) == (ep.slots.u, ep.slots.a, ep.slots.y)
                if same and self.override is None:
                    self.id_map[ep.ep_id] = existing.ep_id
                    self.skipped["eps"] += 1
                    continue
                ep_id = self.fresh_id("ep", lambda i: self.store.get_ep(i) is not None)
            slots = ep.slots.model_copy(
                update={"kappa": ep.slots.kappa.model_copy(update=self._map_kappa(ep.slots.kappa.model_dump()))}
            )
            candidate = ep.model_copy(
                update={
                    "ep_id": ep_id,
                    "slots": slots,
                    "anchor_ids": self.map_ids(list(ep.anchor_ids)),
                    "uses_ep_ids": self.map_ids(list(ep.uses_ep_ids)),
                    "scope": self.scope_of(ep.scope),
                    "atom_id": None,
                }
            )
            stored = self.store.insert_ep(candidate)
            self.id_map[ep.ep_id] = stored.ep_id
            self.imported["eps"] += 1
            written.append(stored)
        return written

    def atoms(self, sys_anchor: Anchor) -> None:
        """(4) create atoms (existing id with the same hash → skip; otherwise remap), then EP back-pointers.
        (4) 创建原子（同 hash 的已存在 id → 跳过；否则重映射），随后写 EP 反向指针。
        """
        ordered = sorted(self.pkg.atoms, key=lambda a: (a.memory_id, a.version, a.atom_id))
        for atom in ordered:
            existing = self.store.get_atom(atom.atom_id) if atom.atom_id else None
            atom_id, memory_id = atom.atom_id, atom.memory_id
            if existing is not None:
                if existing.hash == atom.hash and self.override is None:
                    self.id_map[atom.atom_id] = existing.atom_id
                    self.skipped["atoms"] += 1
                    continue
                atom_id = self.fresh_id("atm", lambda i: self.store.get_atom(i) is not None)
            remapped = atom_id != atom.atom_id
            if memory_id in self.lineage_map:
                memory_id = self.lineage_map[memory_id]
            elif remapped or (self.override is not None and self.store.latest_version(memory_id) is not None):
                memory_id = self.fresh_id("mem", lambda i: self.store.latest_version(i) is not None)
                self.lineage_map[atom.memory_id] = memory_id
            content = dict(atom.content)
            slots = content.get("slots")
            if isinstance(slots, dict) and isinstance(slots.get("kappa"), dict):
                content["slots"] = {**slots, "kappa": self._map_kappa(slots["kappa"])}
            candidate = atom.model_copy(
                update={
                    "atom_id": atom_id,
                    "memory_id": memory_id,
                    "content": content,
                    "anchor_ids": self.map_ids(list(atom.anchor_ids)),
                    "entity_ids": [],
                    "source_ep_ids": self.map_ids(list(atom.source_ep_ids)),
                    "scope": self.scope_of(atom.scope),
                }
            )
            stored = create_atom(
                self.store,
                self.index,
                self.embedder,
                candidate,
                extra_mentions=list(atom.entity_ids),
                history_event="ADD",
                history_anchor_id=sys_anchor.anchor_id,
                history_details={"imported_from": self.pkg.package_id},
                extractor=self.extractor,
            )
            self.id_map[atom.atom_id] = stored.atom_id
            self.imported["atoms"] += 1
        for ep in self.pkg.eps:
            if not ep.atom_id:
                continue
            new_ep_id = self.map_id(ep.ep_id)
            new_atom_id = self.map_id(ep.atom_id)
            stored_ep = self.store.get_ep(new_ep_id)
            if stored_ep is None or stored_ep.atom_id or self.store.get_atom(new_atom_id) is None:
                continue
            self.store.set_ep_atom(new_ep_id, new_atom_id)

    def interface_eps(self) -> list[InterfaceEP]:
        """(5) insert InterfaceEPs over the imported support atoms.
        (5) 在已导入的支持原子之上插入 InterfaceEP。
        """
        written: list[InterfaceEP] = []
        for iep in sorted(self.pkg.interface_eps, key=lambda i: i.interface_ep_id):
            existing = self.store.get_interface_ep(iep.interface_ep_id) if iep.interface_ep_id else None
            iep_id = iep.interface_ep_id
            if existing is not None:
                if self.override is None:
                    self.id_map[iep.interface_ep_id] = existing.interface_ep_id
                    self.skipped["interface_eps"] += 1
                    continue
                iep_id = self.fresh_id("iep", lambda i: self.store.get_interface_ep(i) is not None)
            atom_id = self.map_id(iep.atom_id or "")
            if not atom_id or self.store.get_atom(atom_id) is None:
                logger.warning("import: InterfaceEP %s has no episode atom on the receiver", iep.interface_ep_id)
                self.skipped["interface_eps"] += 1
                continue
            slots = iep.slots.model_copy(
                update={"kappa": iep.slots.kappa.model_copy(update=self._map_kappa(iep.slots.kappa.model_dump()))}
            )
            candidate = iep.model_copy(
                update={
                    "interface_ep_id": iep_id,
                    "slots": slots,
                    "slot_support": {k: self.map_ids(list(v)) for k, v in iep.slot_support.items()},
                    "support_node_ids": self.map_ids(list(iep.support_node_ids)),
                    "support_edge_ids": self.map_ids(list(iep.support_edge_ids)),
                    "boundary_edges": [
                        b.model_copy(
                            update={
                                "edge_id": self.map_id(b.edge_id),
                                "inner_node_id": self.map_id(b.inner_node_id),
                                "outer_node_id": self.map_id(b.outer_node_id),
                                "witness_anchor_ids": self.map_ids(list(b.witness_anchor_ids)),
                            }
                        )
                        for b in iep.boundary_edges
                    ],
                    "invariants": [
                        inv.model_copy(
                            update={
                                "support_node_ids": self.map_ids(list(inv.support_node_ids)),
                                "anchor_ids": self.map_ids(list(inv.anchor_ids)),
                            }
                        )
                        for inv in iep.invariants
                    ],
                    "residuals": [
                        res.model_copy(
                            update={
                                "support_node_ids": self.map_ids(list(res.support_node_ids)),
                                "anchor_ids": self.map_ids(list(res.anchor_ids)),
                                "edge_ids": self.map_ids(list(res.edge_ids)),
                            }
                        )
                        for res in iep.residuals
                    ],
                    "evidence_anchor_ids": self.map_ids(list(iep.evidence_anchor_ids)),
                    "atom_id": atom_id,
                    "scope": self.scope_of(iep.scope),
                }
            )
            stored = self.store.insert_interface_ep(candidate)
            self.id_map[iep.interface_ep_id] = stored.interface_ep_id
            self.imported["interface_eps"] += 1
            written.append(stored)
        return written

    def edges(self) -> None:
        """(6) insert edges (MERGE); boundary edges keep their witness edge only when it exists on the receiver.
        (6) 插入边（MERGE）；边界边仅在其见证边存在于接收方时保留。
        """
        ordered = sorted(self.pkg.edges, key=lambda e: (1 if e.is_boundary else 0, e.edge_id))
        for edge in ordered:
            src, dst = self.map_id(edge.src_id), self.map_id(edge.dst_id)
            if self.store.get_node(src) is None or self.store.get_node(dst) is None or src == dst:
                logger.warning("import: edge %s skipped (endpoint missing on the receiver)", edge.edge_id)
                self.skipped["edges"] += 1
                continue
            existing = self.store.get_edge(edge.edge_id) if edge.edge_id else None
            edge_id = edge.edge_id
            if existing is not None:
                same = (existing.src_id, existing.dst_id, str(existing.edge_type)) == (src, dst, str(edge.edge_type))
                if same and self.override is None:
                    self.id_map[edge.edge_id] = existing.edge_id
                    self.skipped["edges"] += 1
                    continue
                edge_id = self.fresh_id("edg", lambda i: self.store.get_edge(i) is not None)
            merged = self.store.find_edge(src, dst, str(edge.edge_type), str(edge.scale))
            witness_edge = self.map_id(edge.witness_edge_id) if edge.witness_edge_id else None
            boundary = (
                bool(edge.is_boundary) and witness_edge is not None and self.store.get_edge(witness_edge) is not None
            )
            spec = {
                "edge_id": edge_id,
                "src_id": src,
                "dst_id": dst,
                "edge_type": str(edge.edge_type),
                "label": edge.label,
                "scale": str(edge.scale),
                "is_boundary": boundary,
                "confidence": float(edge.confidence),
                "witness_anchor_ids": self.map_ids(list(edge.witness_anchor_ids)),
                "witness_edge_id": witness_edge if boundary else None,
                "created_by": str(edge.created_by),
                "proposer": edge.proposer,
                "valid": bool(edge.valid),
                "invalidated_at": edge.invalidated_at,
                "invalidated_reason": edge.invalidated_reason,
                "scope": self.scope_of(edge.scope),
                "created_at": edge.created_at,
            }
            stored_id = self.store.insert_edge(spec)
            self.id_map[edge.edge_id] = stored_id
            if merged is not None:
                self.skipped["edges"] += 1
            else:
                self.imported["edges"] += 1

    def transitions(self, eps: list[EP], ieps: list[InterfaceEP]) -> None:
        """(7) upsert the ep / interface_ep transition rows of every imported EP and InterfaceEP.
        (7) 为每个导入的 EP 与 InterfaceEP upsert ep / interface_ep 转移行。
        """
        before = self.store.stats()["transitions"]
        total = 0
        for ep in eps:
            stored = self.store.get_ep(ep.ep_id) or ep
            self.store.upsert_transition(
                Transition.from_ep(
                    stored, scope=stored.scope, permission_scope=permission_scope_of(self.store, stored.anchor_ids[0])
                )
            )
            total += 1
        for iep in ieps:
            time_t, time_t1 = interface_ep_window(self.store, iep)
            self.store.upsert_transition(
                Transition.from_interface_ep(
                    iep,
                    scope=iep.scope,
                    time_t=time_t,
                    time_t1=time_t1,
                    permission_scope=permission_scope_of(self.store, iep.evidence_anchor_ids[0]),
                )
            )
            total += 1
        new_rows = self.store.stats()["transitions"] - before
        self.imported["transitions"] = new_rows
        self.skipped["transitions"] = max(0, total - new_rows)


def import_state_package(
    store: StateStore,
    index: Any,
    embedder: Any,
    graph: SupportGraph,
    *,
    pkg: StatePackage,
    scope_override: Scope | None,
    entity_extractor: EntityExtractorBase | None = None,
) -> dict:
    """Recreate a StatePackage on the receiver (HMS §9.4): ids preserved, identity dedup, optional re-scoping.
    在接收方重建 StatePackage（HMS §9.4）：保留 id、按身份去重、可选的范围替换。

    Input / 输入：
        store: The receiver's state store.
        store：接收方的状态存储。

        index: The receiver's vector index.
        index：接收方的向量索引。

        embedder: The receiver's embedder.
        embedder：接收方的嵌入器。

        graph: The receiver's SupportGraph (parity with the other ops entry points).
        graph：接收方的 SupportGraph（与其他运维入口保持一致）。

        pkg: The package (StatePackage.from_json on the facade).
        pkg：交接包（门面上经 StatePackage.from_json 得到）。

        scope_override: When given, replaces the scope of every imported object (ids kept unless they already exist).
        scope_override：给定时替换每个导入对象的范围（id 保留，除非已存在）。

        entity_extractor: The receiver's entity extractor used to re-link imported atoms; None means the rule provider.
        entity_extractor：接收方用于重新链接导入原子的实体抽取器；None 表示规则提供者。

    Output / 输出：
        {'imported': {kind: n}, 'skipped': {kind: n}, 'id_map': {old: new}, 'anchor_id': <system anchor>}.
        {'imported': {kind: n}, 'skipped': {kind: n}, 'id_map': {old: new}, 'anchor_id': <系统锚点>}。

    State effect / 状态影响：
        ONE transaction witnessed by the import system anchor; writes every plane table the package touches.
        一个由导入系统锚点见证的事务；写入交接包涉及的每张平面表。
    """
    scope = scope_override if scope_override is not None else pkg.scope
    scope.require_any(context="import_package")
    importer = _Importer(store, index, embedder, pkg, scope_override, extractor=entity_extractor)
    with store.transaction():
        _ev, sys_anchor = system_event(
            store,
            scope=scope,
            object_type="system",
            event_type="created",
            payload={"op": "import", "package_id": pkg.package_id},
        )
        importer.source_events()
        importer.anchors()
        eps = importer.eps()
        importer.atoms(sys_anchor)
        ieps = importer.interface_eps()
        importer.edges()
        importer.transitions(eps, ieps)
    return {
        "imported": dict(importer.imported),
        "skipped": dict(importer.skipped),
        "id_map": dict(importer.id_map),
        "anchor_id": sys_anchor.anchor_id,
    }


__all__ = ["build_state_package", "import_state_package"]
