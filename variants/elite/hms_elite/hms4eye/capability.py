"""Capability extension: the hms4eye endpoints that are no-ops upstream but real reads/writes on the HMS engine.
能力扩展：那些在上游是 no-op、但在 HMS 引擎上是真实读写的 hms4eye 接口。
"""

# Purpose (EN): Serve hms4eye's entity / graph / mental-model / directive / consolidation / observation endpoints
#               from the dual-plane store, so the compatibility shell stops answering them with a neutral stub; and
#               hand reflect an `organize`-shaped evidence packet built from the Detective Report (HMS_arch §11.1).
# 目的（中）：以双平面存储服务 hms4eye 的 entity / graph / 心智模型 / 指令 / 凝聚 / observation 接口，
#            使兼容壳不再用中性桩答复它们；并为 reflect 交付一个由侦探报告（HMS_arch §11.1）构造的
#            `organize` 形状证据包。

# Input (EN): One live HMS facade plus keyword-only scope arguments (`user_id`, and the object id of a get).
# 输入（中）：一个存活的 HMS 门面，加上仅限关键字的范围参数（`user_id`，以及 get 类调用的对象 id）。

# Output (EN): Plain JSON-shaped dicts matching the hms4eye v0.6.1 response schemas, extended with HMS provenance
#              (witness anchor ids, support node ids, atom ids) that mem0 has nowhere to put.
# 输出（中）：与 hms4eye v0.6.1 响应模式一致的普通 JSON 形状字典，并扩展了 mem0 无处安放的 HMS 出处
#            （见证锚点 id、支持节点 id、原子 id）。

# State effect (EN): Reads are pure; the two clearing calls and the two consolidation calls write - clearing
#                    archives atoms (valid_to + archived_reason) under a system event/anchor and never deletes.
# 状态影响（中）：读操作是纯的；两个清除调用与两个凝聚调用会写入 —— 清除是在一个系统事件/锚点之下
#                归档原子（valid_to + archived_reason），从不删除。

from __future__ import annotations

from typing import Any

from hms_elite.enums import HistoryEvent, ObservationType, SourceKind
from hms_elite.exceptions import ValidationError
from hms_elite.models import MemoryAtom, Scope

# Source kinds that clear_observations ARCHIVES: everything consolidation regenerates from world/experience
# rows, `entity` atoms included, because a later consolidate() rebuilds all of them.
# This is deliberately WIDER than mapping.OBSERVATION_SOURCE_KINDS, which decides which rows read back as
# the `observation` fact type and keeps `entity` at `world` (an entity's text is a name, not a statement).
# clear_observations 归档的来源种类：consolidation 能从 world/experience 行重建的一切，含 `entity` 原子，
# 因为后续一次 consolidate() 会把它们全部重建。
# 这里刻意比 mapping.OBSERVATION_SOURCE_KINDS 更宽——后者决定哪些行回读为 `observation` 事实类型，
# 并把 `entity` 保留为 `world`（实体的文本是名称，不是陈述）。
CONSOLIDATION_SOURCE_KINDS: frozenset[str] = frozenset(
    {SourceKind.interface_ep.value, SourceKind.skill.value, SourceKind.entity.value}
)

# Our three layers projected onto upstream's fact_type: Observation -> world, EP -> experience, and every
# consolidation product -> observation. A row with no mapped source kind stays `world`, as upstream's reader does.
# 我们的三层投影到上游的 fact_type：Observation -> world、EP -> experience，
# 而一切 consolidation 产物 -> observation。没有映射的来源种类保持 `world`，与上游读取器一致。
FACT_TYPE_OF_SOURCE_KIND: dict[str, str] = {
    SourceKind.message.value: "world",
    SourceKind.event.value: "world",
    SourceKind.observation.value: "world",
    SourceKind.feedback.value: "world",
    SourceKind.procedural.value: "world",
    SourceKind.ep.value: "experience",
    SourceKind.interface_ep.value: "observation",
    SourceKind.skill.value: "observation",
    SourceKind.entity.value: "observation",
}

# The prefix promote_skills puts in front of a distilled invariant; stripped to build a mental model's `name`.
# promote_skills 放在被蒸馏不变量前面的前缀；构造心智模型 `name` 时会被剥掉。
_SKILL_PREFIX = "Skill: "

# archived_reason written by the two clearing calls, so a cleared row is distinguishable from a decayed one.
# 两个清除调用写入的 archived_reason，使被清除的行可以与被衰减的行区分开。
_CLEARED_REASON = "cleared"

# The `mode` of the reflect evidence packet: upstream's organizer says 'ordered_recall'; ours is a why() chain.
# reflect 证据包的 `mode`：上游的 organizer 写 'ordered_recall'；我们的是一条 why() 链。
REFLECT_EVIDENCE_MODE = "detective_report"
# The why() question mode answers NOT_FOUND_001 when nothing resolves; reflect then hands back the empty packet.
# why() 的问题模式在无法解析时回答 NOT_FOUND_001；此时 reflect 交回空证据包。
_NOT_FOUND = "NOT_FOUND_001"


def _short_name(text: str, limit: int = 80) -> str:
    """First sentence of a text, trimmed to `limit` characters, as a stable display name.
    文本的第一句，截断到 `limit` 个字符，作为稳定的显示名。
    """
    head = text.strip().split("\n", 1)[0].strip()
    for stop in (". ", "。", "! ", "? "):
        if stop in head:
            head = head.split(stop, 1)[0] + stop.strip()
            break
    head = head.rstrip(".。")
    return head if len(head) <= limit else head[: limit - 1].rstrip() + "…"


def _instant_of(atom: MemoryAtom) -> str | None:
    """The atom's best event instant: its time anchor, else its ingest instant.
    原子最合适的事件时刻：其时间锚点，否则其摄取时刻。
    """
    return atom.time_anchor or atom.created_at


def _fact_type(atom: MemoryAtom) -> str:
    """The upstream fact_type of an atom (world | experience | observation).
    某原子对应的上游 fact_type（world | experience | observation）。
    """
    return FACT_TYPE_OF_SOURCE_KIND.get(str(atom.source_kind), "world")


def _is_live(atom: MemoryAtom, now: str) -> bool:
    """True when an atom is neither archived nor past its validity window at `now`.
    当原子在 `now` 时刻既未归档也未过有效期时为 True。
    """
    return atom.archived_reason is None and (atom.valid_to is None or atom.valid_to > now)


class HMSCapabilities:
    """hms4eye capability surface over one HMS instance; it never subclasses or mutates the facade.
    基于单个 HMS 实例的 hms4eye 能力接口；它既不继承也不修改门面。
    """

    def __init__(self, hms: Any) -> None:
        """Bind the capability surface to a live HMS facade (its store and scheduler are used as they are).
        将能力接口绑定到一个存活的 HMS 门面（原样使用其存储与调度器）。

        Input / 输入：
            hms: An `hms_elite.memory.main.HMS`; only its public API and `hms.store` are touched.
            hms：一个 `hms_elite.memory.main.HMS`；只会接触其公共 API 与 `hms.store`。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None; the constructor stores the reference and opens nothing.
            无；构造函数只保存引用，不打开任何资源。
        """
        self.hms = hms
        self.store = hms.store

    # ------------------------------------------------------------------ entities
    # ------------------------------------------------------------------ 实体
    def list_entities(self, *, user_id: str, limit: int = 100, offset: int = 0) -> dict:
        """hms4eye `list_entities`: the scope's entity aggregates with mention counts and their observations.
        hms4eye 的 `list_entities`：范围内的实体聚合体，带提及计数与其 observation。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope `Scope(user_id=...)`.
            user_id：Mem0 用户 id；转换为部分范围 `Scope(user_id=...)`。

            limit: Maximum number of entities in `items` (upstream default 100).
            limit：`items` 中的实体数上限（上游默认 100）。

            offset: Pagination offset applied before `limit`.
            offset：在 `limit` 之前应用的分页偏移。

        Output / 输出：
            {'items': [EntityDetailResponse-shaped dicts], 'total': int, 'limit': int, 'offset': int}.
            {'items': [EntityDetailResponse 形状的字典], 'total': int, 'limit': int, 'offset': int}。

        State effect / 状态影响：
            None; three read queries per entity and no writes.
            无；每个实体三次读查询，且不写入。
        """
        scope = self._scope(user_id)
        entities = self.store.entities_in_scope(scope)
        window = entities[max(0, int(offset)) : max(0, int(offset)) + max(0, int(limit))]
        return {
            "items": [self._entity_detail(entity) for entity in window],
            "total": len(entities),
            "limit": int(limit),
            "offset": int(offset),
        }

    def get_entity(self, entity_id: str, *, user_id: str) -> dict | None:
        """hms4eye `get_entity`: one entity aggregate of the scope, or None when the id is unknown there.
        hms4eye 的 `get_entity`：范围内的单个实体聚合体；该范围内 id 未知时返回 None。

        Input / 输入：
            entity_id: The `entities.entity_id` of the wanted entity.
            entity_id：目标实体的 `entities.entity_id`。

            user_id: Mem0 user id defining the scope the entity must be linked in.
            user_id：定义实体必须在其中被链接的范围的 Mem0 用户 id。

        Output / 输出：
            An EntityDetailResponse-shaped dict, or None.
            EntityDetailResponse 形状的字典，或 None。

        State effect / 状态影响：
            None.
            无。
        """
        wanted = str(entity_id).strip()
        for entity in self.store.entities_in_scope(self._scope(user_id)):
            if entity["entity_id"] == wanted:
                return self._entity_detail(entity)
        return None

    def get_entity_graph(self, *, user_id: str, limit: int = 100) -> dict:
        """hms4eye `get_entity_graph`: entities as nodes, weighted by how often two of them share an atom.
        hms4eye 的 `get_entity_graph`：以实体为节点，按两个实体共处同一原子的次数加权。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope.
            user_id：Mem0 用户 id；转换为部分范围。

            limit: Maximum number of co-occurrence edges returned (heaviest first).
            limit：返回的共现边数上限（权重大的优先）。

        Output / 输出：
            {'nodes', 'edges', 'total_entities', 'total_edges', 'limit'}; each edge carries its witness anchors.
            {'nodes', 'edges', 'total_entities', 'total_edges', 'limit'}；每条边都带其见证锚点。

        State effect / 状态影响：
            None.
            无。
        """
        entities = self.store.entities_in_scope(self._scope(user_id))
        nodes: list[dict] = []
        by_atom: dict[str, list[str]] = {}
        for entity in entities:
            linked = self._source_atoms_of(entity["entity_id"])
            nodes.append(
                {
                    "id": entity["entity_id"],
                    "label": entity["canonical_text"],
                    "canonical_name": entity["canonical_text"],
                    "type": entity["entity_type"],
                    "mention_count": len(linked),
                }
            )
            for atom in linked:
                by_atom.setdefault(atom.atom_id, []).append(entity["entity_id"])
        pairs: dict[tuple[str, str], dict] = {}
        for atom_id in sorted(by_atom):
            members = sorted(set(by_atom[atom_id]))
            witnesses = self.store.atom_anchor_ids(atom_id)
            for i, src in enumerate(members):
                for dst in members[i + 1 :]:
                    edge = pairs.setdefault(
                        (src, dst),
                        {"source": src, "target": dst, "cooccurrence_count": 0, "atom_ids": [], "witnesses": []},
                    )
                    edge["cooccurrence_count"] += 1
                    edge["atom_ids"].append(atom_id)
                    edge["witnesses"] = sorted(set(edge["witnesses"]) | set(witnesses))
        ordered = sorted(pairs.values(), key=lambda e: (-e["cooccurrence_count"], e["source"], e["target"]))
        for edge in ordered:
            edge["weight"] = edge["cooccurrence_count"]
        return {
            "nodes": nodes,
            "edges": ordered[: max(0, int(limit))],
            "total_entities": len(nodes),
            "total_edges": len(ordered),
            "limit": int(limit),
        }

    # ------------------------------------------------------------------ memory graph
    # ------------------------------------------------------------------ 记忆图
    def get_graph(self, *, user_id: str, limit: int = 200) -> dict:
        """hms4eye `get_graph`: our Support Graph - valid nodes plus the valid edges between them, with witnesses.
        hms4eye 的 `get_graph`：我们的支持图 —— 有效节点，加上它们之间带见证的有效边。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope.
            user_id：Mem0 用户 id；转换为部分范围。

            limit: Maximum number of nodes returned; edges are kept only between returned nodes.
            limit：返回的节点数上限；只保留返回节点之间的边。

        Output / 输出：
            {'nodes', 'edges', 'table_rows', 'total_units', 'limit'}; every edge has a non-empty `witnesses` list.
            {'nodes', 'edges', 'table_rows', 'total_units', 'limit'}；每条边的 `witnesses` 列表都非空。

        State effect / 状态影响：
            None.
            无。
        """
        scope = self._scope(user_id)
        all_nodes = self.store.list_nodes(scope, valid_only=True)
        kept = all_nodes[: max(0, int(limit))]
        atoms = {atom.atom_id: atom for atom in self.store.get_atoms([node.node_id for node in kept])}
        node_ids = {node.node_id for node in kept}
        nodes: list[dict] = []
        table_rows: list[dict] = []
        for node in kept:
            atom = atoms.get(node.node_id)
            text = atom.text if atom is not None else ""
            fact_type = _fact_type(atom) if atom is not None else "world"
            nodes.append(
                {
                    "id": node.node_id,
                    "kind": str(node.node_kind),
                    "scale": str(node.scale),
                    "text": text,
                    "label": text,
                    "type": fact_type,
                    "fact_type": fact_type,
                    "mentions": node.mentions,
                    "created_at": node.created_at,
                    "witnesses": list(node.anchor_ids),
                }
            )
            table_rows.append(
                {
                    "id": node.node_id,
                    "text": text,
                    "date": _instant_of(atom) if atom is not None else node.created_at,
                    "context": str(node.node_kind),
                    "entities": ", ".join(e["canonical_text"] for e in self.store.entities_for_atom(node.node_id)),
                }
            )
        edges: list[dict] = []
        for edge in self.store.list_edges(scope, valid_only=True):
            if edge.src_id not in node_ids or edge.dst_id not in node_ids:
                continue
            edges.append(
                {
                    "id": edge.edge_id,
                    "src": edge.src_id,
                    "dst": edge.dst_id,
                    "from": edge.src_id,
                    "to": edge.dst_id,
                    "type": str(edge.edge_type),
                    "label": edge.label,
                    "scale": str(edge.scale),
                    "confidence": float(edge.confidence),
                    "weight": float(edge.confidence),
                    "is_boundary": bool(edge.is_boundary),
                    # The differentiator against mem0: an edge without a witness anchor cannot exist here.
                    # 相对 mem0 的关键差异：没有见证锚点的边在这里不可能存在。
                    "witnesses": list(edge.witness_anchor_ids),
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "table_rows": table_rows,
            "total_units": len(all_nodes),
            "limit": int(limit),
        }

    # ------------------------------------------------------------------ mental models
    # ------------------------------------------------------------------ 心智模型
    def list_mental_models(self, *, user_id: str) -> dict:
        """hms4eye `list_mental_models`: our skill atoms, the invariants promote_skills distilled from InterfaceEPs.
        hms4eye 的 `list_mental_models`：我们的 skill 原子，即 promote_skills 从 InterfaceEP 蒸馏出的不变量。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope.
            user_id：Mem0 用户 id；转换为部分范围。

        Output / 输出：
            {'items': [MentalModelResponse-shaped dicts], 'total': int}.
            {'items': [MentalModelResponse 形状的字典], 'total': int}。

        State effect / 状态影响：
            None.
            无。
        """
        atoms = self.store.axis_select(
            self._scope(user_id),
            where_sql="source_kind = ?",
            params=[SourceKind.skill.value],
            order="created_at ASC, atom_id ASC",
        )
        items = [self._mental_model(atom, bank_id=user_id) for atom in atoms]
        return {"items": items, "total": len(items)}

    def get_mental_model(self, model_id: str, *, user_id: str) -> dict | None:
        """hms4eye `get_mental_model`: one skill atom addressed by its lineage id (its version id also resolves).
        hms4eye 的 `get_mental_model`：按谱系 id 定位的单个 skill 原子（版本 id 同样可解析）。

        Input / 输入：
            model_id: The mental model id, i.e. our `memory_id` (an `atom_id` is accepted too).
            model_id：心智模型 id，即我们的 `memory_id`（也接受 `atom_id`）。

            user_id: Mem0 user id the model must belong to.
            user_id：模型必须归属的 Mem0 用户 id。

        Output / 输出：
            A MentalModelResponse-shaped dict, or None when unknown, out of scope or not a skill.
            MentalModelResponse 形状的字典；未知、越界或并非 skill 时返回 None。

        State effect / 状态影响：
            None.
            无。
        """
        atom = self._resolve_atom(model_id, user_id=user_id, source_kind=SourceKind.skill.value)
        return None if atom is None else self._mental_model(atom, bank_id=user_id)

    def get_mental_model_history(self, model_id: str, *, user_id: str) -> list[dict]:
        """hms4eye `get_mental_model_history`: the skill lineage's witnessed history rows, oldest first.
        hms4eye 的 `get_mental_model_history`：该 skill 谱系带见证的历史行，最旧在前。

        Input / 输入：
            model_id: The mental model id, i.e. our `memory_id` (an `atom_id` is accepted too).
            model_id：心智模型 id，即我们的 `memory_id`（也接受 `atom_id`）。

            user_id: Mem0 user id the model must belong to.
            user_id：模型必须归属的 Mem0 用户 id。

        Output / 输出：
            [{'id', 'mental_model_id', 'event', 'content', 'previous_content', 'created_at', ...}]; [] when unknown.
            [{'id', 'mental_model_id', 'event', 'content', 'previous_content', 'created_at', ...}]；未知时为 []。

        State effect / 状态影响：
            None.
            无。
        """
        atom = self._resolve_atom(model_id, user_id=user_id, source_kind=SourceKind.skill.value)
        if atom is None:
            return []
        versions = {version.atom_id: version.version for version in self.store.list_versions(atom.memory_id)}
        rows: list[dict] = []
        for row in self.store.get_history(atom.memory_id):
            rows.append(
                {
                    "id": row["id"],
                    "mental_model_id": atom.memory_id,
                    "event": row["event"],
                    "content": row["new_memory"],
                    "previous_content": row["old_memory"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "is_deleted": row["is_deleted"],
                    "anchor_id": row["anchor_id"],
                    "versions": sorted(versions.values()),
                }
            )
        return rows

    # ------------------------------------------------------------------ directives
    # ------------------------------------------------------------------ 指令
    def list_directives(self, *, user_id: str, active_only: bool = True) -> dict:
        """hms4eye `list_directives`: our constraint atoms, ordered by derived priority then id.
        hms4eye 的 `list_directives`：我们的 constraint 原子，按派生优先级、再按 id 排序。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope.
            user_id：Mem0 用户 id；转换为部分范围。

            active_only: True keeps only live constraints; False also returns archived and expired ones.
            active_only：True 只保留存活约束；False 同时返回已归档与已过期的约束。

        Output / 输出：
            {'items': [DirectiveResponse-shaped dicts], 'total': int}.
            {'items': [DirectiveResponse 形状的字典], 'total': int}。

        State effect / 状态影响：
            None.
            无。
        """
        atoms = self.store.axis_select(
            self._scope(user_id),
            where_sql="obs_type = ?",
            params=[ObservationType.constraint.value],
            include_expired=not active_only,
            order="atom_id ASC",
        )
        now = self.store.now()
        items = [self._directive(atom, bank_id=user_id, now=now) for atom in atoms]
        items.sort(key=lambda d: (-d["priority"], d["atom_id"]))
        return {"items": items, "total": len(items)}

    def get_directive(self, directive_id: str, *, user_id: str) -> dict | None:
        """hms4eye `get_directive`: one constraint atom addressed by its lineage id (version id also resolves).
        hms4eye 的 `get_directive`：按谱系 id 定位的单个 constraint 原子（版本 id 同样可解析）。

        Input / 输入：
            directive_id: The directive id, i.e. our `memory_id` (an `atom_id` is accepted too).
            directive_id：指令 id，即我们的 `memory_id`（也接受 `atom_id`）。

            user_id: Mem0 user id the directive must belong to.
            user_id：指令必须归属的 Mem0 用户 id。

        Output / 输出：
            A DirectiveResponse-shaped dict, or None when unknown, out of scope or not a constraint.
            DirectiveResponse 形状的字典；未知、越界或并非约束时返回 None。

        State effect / 状态影响：
            None.
            无。
        """
        atom = self._resolve_atom(directive_id, user_id=user_id, obs_type=ObservationType.constraint.value)
        return None if atom is None else self._directive(atom, bank_id=user_id, now=self.store.now())

    # ------------------------------------------------------------------ consolidation
    # ------------------------------------------------------------------ 凝聚
    def trigger_consolidation(self, *, user_id: str, phases: list[str] | None = None) -> dict:
        """hms4eye `trigger_consolidation`: run HMS.consolidate synchronously and report what each phase did.
        hms4eye 的 `trigger_consolidation`：同步运行 HMS.consolidate，并报告每个阶段做了什么。

        Input / 输入：
            user_id: Mem0 user id; every scope under it is a scheduler target.
            user_id：Mem0 用户 id；其下的每个范围都是调度器目标。

            phases: Phase names in run order, or None for the configured default.
            phases：按运行顺序的阶段名，或 None 表示使用配置默认值。

        Output / 输出：
            {'operation_id', 'deduplicated', 'report'}; operation_id is the scheduler anchor witnessing the run.
            {'operation_id', 'deduplicated', 'report'}；operation_id 是见证本次运行的调度器锚点。

        State effect / 状态影响：
            Everything HMS.consolidate writes: entity atoms, edges, InterfaceEPs, skills, transitions, decay.
            HMS.consolidate 写入的一切：entity 原子、边、InterfaceEP、skill、转移、衰减。
        """
        report = self.hms.consolidate(filters={"user_id": user_id}, phases=phases)
        # The scheduler anchor is already a stable id from our own id factory and it addresses the finished run;
        # only an empty database has none, and then we mint one so the response always carries an operation id.
        # 调度器锚点本就是我们自己的 id 工厂产出的稳定 id，且能定位这次已完成的运行；
        # 只有空库才没有它，此时我们生成一个，使响应始终带有 operation id。
        operation_id = str(report.get("anchor_id") or self.store.new_id("anc"))
        return {"operation_id": operation_id, "deduplicated": False, "report": report}

    def recover_consolidation(self, *, user_id: str) -> dict:
        """hms4eye `recover_consolidation`: re-run the phases over whatever is still pending (we have no failed state).
        hms4eye 的 `recover_consolidation`：对仍待处理的部分重跑阶段（我们没有失败态）。

        Input / 输入：
            user_id: Mem0 user id; every scope under it is a scheduler target.
            user_id：Mem0 用户 id；其下的每个范围都是调度器目标。

        Output / 输出：
            {'retried_count', 'operation_id', 'report'}; retried_count is the pending EP count before the run.
            {'retried_count', 'operation_id', 'report'}；retried_count 是本次运行前待处理的 EP 数。

        State effect / 状态影响：
            Same as trigger_consolidation; nothing is retried that already succeeded.
            与 trigger_consolidation 相同；已成功的部分不会被重试。
        """
        pending = self._pending_consolidation(self._scope(user_id))
        triggered = self.trigger_consolidation(user_id=user_id)
        return {
            "retried_count": pending,
            "operation_id": triggered["operation_id"],
            "report": triggered["report"],
        }

    # --------------------------------------------- observations (consolidation output) / 观察（凝聚产物）
    # ------------------------------------------------------------------ observation
    def clear_observations(self, *, user_id: str) -> dict:
        """hms4eye `clear_observations`: archive every consolidation product of the scope under a system anchor.
        hms4eye 的 `clear_observations`：在系统锚点之下归档范围内的每个 consolidation 产物。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope.
            user_id：Mem0 用户 id；转换为部分范围。

        Output / 输出：
            {'deleted_count', 'success', 'message', 'anchor_id', 'archived'}; the rows stay, only valid_to is set.
            {'deleted_count', 'success', 'message', 'anchor_id', 'archived'}；行仍在，只是设置了 valid_to。

        State effect / 状态影响：
            One transaction: a system event + anchor, then archive_atom + an ARCHIVE history row per target.
            一个事务：一个系统事件 + 锚点，随后对每个目标执行 archive_atom 并写入一行 ARCHIVE 历史。
        """
        scope = self._scope(user_id)
        targets = self._observation_atoms(scope)
        anchor_id = self._archive_all(scope, targets, op="delete_all", capability="clear_observations")
        return {
            "deleted_count": len(targets),
            "success": True,
            "message": f"Archived {len(targets)} observation atoms; HMS never deletes, the rows remain queryable.",
            "anchor_id": anchor_id,
            "archived": [atom.atom_id for atom in targets],
        }

    def clear_memory_observations(self, memory_id: str, *, user_id: str) -> dict:
        """hms4eye `clear_memory_observations`: archive only the consolidation products derived from one memory.
        hms4eye 的 `clear_memory_observations`：只归档由某一条记忆派生出的 consolidation 产物。

        Input / 输入：
            memory_id: Mem0 memory id (lineage id); an `atom_id` is accepted too.
            memory_id：Mem0 记忆 id（谱系 id）；也接受 `atom_id`。

            user_id: Mem0 user id the memory must belong to.
            user_id：该记忆必须归属的 Mem0 用户 id。

        Output / 输出：
            {'deleted_count', 'success', 'message', 'anchor_id', 'archived'}; deleted_count is 0 for an unknown id.
            {'deleted_count', 'success', 'message', 'anchor_id', 'archived'}；id 未知时 deleted_count 为 0。

        State effect / 状态影响：
            Same as clear_observations, restricted to the derived atoms that name this memory's atom as support.
            与 clear_observations 相同，但限于把该记忆的原子列为支持的派生原子。
        """
        scope = self._scope(user_id)
        source = self._resolve_atom(memory_id, user_id=user_id)
        if source is None:
            return {
                "deleted_count": 0,
                "success": False,
                "message": f"Memory with id {memory_id} not found in this scope.",
                "anchor_id": None,
                "archived": [],
            }
        targets = self._derived_from(scope, source)
        anchor_id = self._archive_all(
            scope, targets, op="delete", capability="clear_memory_observations", memory_id=source.memory_id
        )
        return {
            "deleted_count": len(targets),
            "success": True,
            "message": f"Archived {len(targets)} observation atoms derived from {source.memory_id}.",
            "anchor_id": anchor_id,
            "archived": [atom.atom_id for atom in targets],
        }

    # ------------------------------------------------------------------ statistics
    # ------------------------------------------------------------------ 统计
    def bank_stats(self, *, user_id: str) -> dict:
        """hms4eye `get_agent_stats`: the BankStatsResponse fields filled with counted rows, not placeholders.
        hms4eye 的 `get_agent_stats`：用真实计数而非占位值填充的 BankStatsResponse 字段。

        Input / 输入：
            user_id: Mem0 user id; becomes the partial scope and the reported bank_id.
            user_id：Mem0 用户 id；转换为部分范围，并作为上报的 bank_id。

        Output / 输出：
            The BankStatsResponse fields, including total_observations / pending_consolidation / failed_consolidation.
            BankStatsResponse 的各字段，含 total_observations / pending_consolidation / failed_consolidation。

        State effect / 状态影响：
            None.
            无。
        """
        scope = self._scope(user_id)
        atoms = self.store.axis_select(scope, order="atom_id ASC")
        edges = self.store.list_edges(scope, valid_only=True)
        fact_of_node = {atom.atom_id: _fact_type(atom) for atom in atoms}
        nodes_by_fact_type: dict[str, int] = {}
        for atom in atoms:
            key = _fact_type(atom)
            nodes_by_fact_type[key] = nodes_by_fact_type.get(key, 0) + 1
        links_by_link_type: dict[str, int] = {}
        links_by_fact_type: dict[str, int] = {}
        links_breakdown: dict[str, dict[str, int]] = {}
        for edge in edges:
            link_type = str(edge.edge_type)
            fact = fact_of_node.get(edge.src_id, "world")
            links_by_link_type[link_type] = links_by_link_type.get(link_type, 0) + 1
            links_by_fact_type[fact] = links_by_fact_type.get(fact, 0) + 1
            links_breakdown.setdefault(link_type, {})
            links_breakdown[link_type][fact] = links_breakdown[link_type].get(fact, 0) + 1
        scheduler_events = self.store.last_events(scope, object_types=["scheduler"], limit=1)
        return {
            "bank_id": user_id,
            "total_nodes": len(atoms),
            "total_links": len(edges),
            "total_documents": 0,
            "nodes_by_fact_type": nodes_by_fact_type,
            "links_by_link_type": links_by_link_type,
            "links_by_fact_type": links_by_fact_type,
            "links_breakdown": links_breakdown,
            "pending_operations": 0,
            "failed_operations": 0,
            "operations_by_status": {},
            "last_consolidated_at": scheduler_events[0].timestamp if scheduler_events else None,
            # An EP that no valid InterfaceEP names as support has not been coarsened into an observation yet.
            # 没有被任何有效 InterfaceEP 列为支持的 EP，还没有被粗化成 observation。
            "pending_consolidation": self._pending_consolidation(scope),
            # HMS has no permanent failure state: a phase either writes its rows or raises, so nothing is stuck.
            # HMS 没有永久失败态：某阶段要么写入其行、要么抛出异常，因此不存在卡住的对象。
            "failed_consolidation": 0,
            "total_observations": nodes_by_fact_type.get("observation", 0),
        }

    # ------------------------------------------------------------------ reflect evidence
    # ------------------------------------------------------------------ reflect 证据
    def reflect_evidence(self, query: str, *, user_id: str) -> dict:
        """Evidence packet for hms4eye `reflect`, in the `organize` shape, built from the Detective Report of `query`.
        面向 hms4eye `reflect` 的证据包，采用 `organize` 形状，由 `query` 的侦探报告构造。

        Input / 输入：
            query: The reflect question (why()'s question mode resolves it through a detective compile).
            query：reflect 问题（why() 的问题模式经侦探式编译解析它）。

            user_id: Bank id → the partial scope (user_id only).
            user_id：记忆库 id → 部分范围（仅 user_id）。

        Output / 输出：
            {'question', 'question_date', 'mode', 'ledger_rows', 'controls', 'source_snippets', 'answer_ready_context',
            'conclusion', 'chain'}: ledger_rows are the why() chain nodes in report order, each with `anchor_ids`
            (never empty), `atom_id`, `type` (world | experience | observation), `occurred`, `mentioned`, `entities`;
            source_snippets are the cited anchors; the empty packet (ledger_rows == []) when nothing resolves.
            {'question', 'question_date', 'mode', 'ledger_rows', 'controls', 'source_snippets', 'answer_ready_context',
            'conclusion', 'chain'}：ledger_rows 是按报告顺序的 why() 链节点，每行带 `anchor_ids`（永不为空）、
            `atom_id`、`type`（world | experience | observation）、`occurred`、`mentioned`、`entities`；
            source_snippets 是被引用的锚点；无法解析时为空证据包（ledger_rows == []）。

        State effect / 状态影响：
            None (read-only; one store.now() for question_date).
            无（只读；一次 store.now() 作为 question_date）。
        """
        question = str(query)
        now = self.store.now()
        try:
            report, _text = self.hms.state.detective_report(question, filters=self._scope(user_id).as_filters(), as_of=now)
        except ValidationError as exc:
            if exc.error_code != _NOT_FOUND:
                raise
            return self._evidence_packet(question, now, rows=[], snippets=[], conclusion=None, chain=[])
        atoms = {a.atom_id: a for a in self.store.get_atoms([s.atom_id for s in report.steps if s.atom_id])}
        rows: list[dict] = []
        for index, step in enumerate(report.steps, start=1):
            atom = atoms.get(step.atom_id or "")
            rows.append(
                {
                    "index": index,
                    "score": 0,
                    "text": step.text,
                    "document_id": step.anchor_ids[0] if step.anchor_ids else None,
                    "type": _fact_type(atom) if atom is not None else None,
                    "occurred": (_instant_of(atom) if atom is not None else None) or step.time,
                    "mentioned": atom.created_at if atom is not None else step.time,
                    "chunk_id": None,
                    "entities": list(atom.entity_ids) if atom is not None and atom.entity_ids else None,
                    "anchor_ids": list(step.anchor_ids),
                    "atom_id": step.atom_id,
                }
            )
        snippets = [
            {
                "anchor_id": item.anchor_ids[0],
                "text": item.text,
                "timestamp": item.time,
                "source_ref": item.extra.get("source_ref"),
                "revoked_at": item.extra.get("revoked_at"),
            }
            for item in report.evidence
        ]
        conclusion = {
            "text": report.conclusion.text,
            "atom_id": report.conclusion.atom_id,
            "anchor_ids": list(report.conclusion.anchor_ids),
        }
        return self._evidence_packet(question, now, rows=rows, snippets=snippets, conclusion=conclusion, chain=report.chain)

    @staticmethod
    def _evidence_packet(
        question: str, question_date: str, *, rows: list[dict], snippets: list[dict], conclusion: dict | None, chain: list
    ) -> dict:
        """Assemble the `organize`-shaped packet (upstream's seven keys plus conclusion / chain) and its context text.
        组装 `organize` 形状的证据包（上游的七个键加 conclusion / chain）及其上下文文本。
        """
        lines = [
            "=== Retrieved Memory Evidence ===",
            f"Question: {question}",
            f"Question date: {question_date}",
            "",
            "Recalled facts (original order):",
        ]
        if rows:
            lines.extend(
                f"{row['index']}. occurred={row['occurred'] or '-'} | mentioned={row['mentioned'] or '-'} | "
                f"doc={row['document_id'] or '-'} | type={row['type'] or '-'} | {row['text']}"
                for row in rows
            )
        else:
            lines.append("- none")
        return {
            "question": question,
            "question_date": question_date,
            "mode": REFLECT_EVIDENCE_MODE,
            "ledger_rows": rows,
            "controls": [],
            "source_snippets": snippets,
            "answer_ready_context": "\n".join(lines),
            "conclusion": conclusion,
            "chain": [str(line) for line in chain],
        }

    # ------------------------------------------------------------------ private helpers
    # ------------------------------------------------------------------ 私有辅助
    def _scope(self, user_id: str) -> Scope:
        """The partial scope of a bank: only user_id is pinned, so every agent/run under it matches.
        某记忆库的部分范围：只钉住 user_id，因此其下的每个 agent/run 都匹配。
        """
        return Scope(user_id=str(user_id).strip())

    def _source_atoms_of(self, entity_id: str) -> list[MemoryAtom]:
        """Live atoms linked to an entity that are NOT consolidation products (the entity's real mentions).
        链接到某实体、且不是 consolidation 产物的存活原子（该实体真正的提及）。
        """
        linked = self.store.entity_linked_atoms(entity_id, valid_only=True, exclude_entity_atoms=True)
        return [a for a in self.store.get_atoms(linked) if str(a.source_kind) not in CONSOLIDATION_SOURCE_KINDS]

    def _observation_atoms_of(self, entity_id: str) -> list[MemoryAtom]:
        """Live consolidation products linked to an entity, excluding that entity's own entity atom.
        链接到某实体的存活 consolidation 产物，排除该实体自身的 entity 原子。
        """
        linked = self.store.entity_linked_atoms(entity_id, valid_only=True, exclude_entity_atoms=True)
        return [a for a in self.store.get_atoms(linked) if str(a.source_kind) in CONSOLIDATION_SOURCE_KINDS]

    def _entity_atom_id(self, entity_id: str) -> str | None:
        """The atom_id of the consolidation-minted entity atom of an entity, or None before the first consolidate.
        某实体由 consolidation 生成的 entity 原子的 atom_id；首次 consolidate 之前为 None。
        """
        rows = self.store.axis_select(
            Scope(),
            where_sql="source_kind = ? AND source_id = ?",
            params=[SourceKind.entity.value, entity_id],
            include_expired=True,
            order="atom_id ASC",
            limit=1,
        )
        return rows[0].atom_id if rows else None

    def _entity_detail(self, entity: dict) -> dict:
        """One EntityDetailResponse-shaped dict from an `entities_in_scope` row.
        由一行 `entities_in_scope` 结果构造 EntityDetailResponse 形状的字典。
        """
        mentions = self._source_atoms_of(entity["entity_id"])
        observations = self._observation_atoms_of(entity["entity_id"])
        instants = sorted(i for i in (_instant_of(a) for a in mentions) if i)
        return {
            "id": entity["entity_id"],
            "canonical_name": entity["canonical_text"],
            "mention_count": len(mentions),
            "first_seen": instants[0] if instants else None,
            "last_seen": instants[-1] if instants else None,
            "metadata": {
                "entity_type": entity["entity_type"],
                "normalized_text": entity["normalized_text"],
                "aliases": list(entity["aliases"]),
                "user_id": entity["user_id"],
                "agent_id": entity["agent_id"],
                "entity_atom_id": self._entity_atom_id(entity["entity_id"]),
                "mention_atom_ids": [a.atom_id for a in mentions],
            },
            "observations": [
                {
                    "text": atom.text,
                    "mentioned_at": _instant_of(atom),
                    "atom_id": atom.atom_id,
                    "fact_type": _fact_type(atom),
                    "witnesses": list(atom.anchor_ids),
                }
                for atom in observations
            ],
        }

    def _resolve_atom(
        self, object_id: Any, *, user_id: str, source_kind: str | None = None, obs_type: str | None = None
    ) -> MemoryAtom | None:
        """The atom behind a lineage id or a version id, gated by scope and optionally by source_kind / obs_type.
        谱系 id 或版本 id 背后的原子，受范围门控，并可按 source_kind / obs_type 进一步限定。
        """
        wanted = str(object_id).strip() if object_id is not None else ""
        if not wanted:
            return None
        atom = self.store.resolve_head(wanted) or self.store.get_atom(wanted)
        if atom is None:
            rows = self.store.list_versions(wanted)
            atom = rows[-1] if rows else None
        if atom is None or not self._scope(user_id).matches(atom.scope):
            return None
        if source_kind is not None and str(atom.source_kind) != source_kind:
            return None
        if obs_type is not None and str(atom.obs_type or "") != obs_type:
            return None
        return atom

    def _mental_model(self, atom: MemoryAtom, *, bank_id: str) -> dict:
        """One MentalModelResponse-shaped dict from a skill atom and the InterfaceEP it was distilled from.
        由一个 skill 原子与其被蒸馏自的 InterfaceEP 构造 MentalModelResponse 形状的字典。
        """
        content = atom.content if isinstance(atom.content, dict) else {}
        invariant = content.get("invariant") if isinstance(content.get("invariant"), dict) else {}
        support = [str(s) for s in (content.get("support_node_ids") or [])]
        iep = self.store.get_interface_ep(atom.source_id) if atom.source_id else None
        refreshed = atom.updated_at or atom.created_at
        body = str(invariant.get("text") or atom.text.removeprefix(_SKILL_PREFIX))
        lines = [
            f"# {_short_name(body)}",
            "",
            body,
            "",
            f"- Interface EP: `{atom.source_id}`",
            f"- Support nodes: {', '.join(f'`{s}`' for s in support) or '—'}",
            f"- Witness anchors: {', '.join(f'`{a}`' for a in atom.anchor_ids)}",
            f"- Confidence: {float(atom.confidence)}",
        ]
        return {
            "id": atom.memory_id,
            "bank_id": bank_id,
            "name": _short_name(body),
            "source_query": iep.slots.u if iep is not None else None,
            "content": "\n".join(lines),
            "tags": list(atom.topic_tags),
            "max_tokens": None,
            "trigger": None,
            "last_refreshed_at": refreshed,
            "created_at": atom.created_at,
            "reflect_response": {
                "answer": body,
                "based_on": {"support_node_ids": support, "anchor_ids": list(atom.anchor_ids)},
                "interface_ep_id": atom.source_id,
                "invariant_kind": str(invariant.get("kind") or ""),
            },
            "is_stale": self._is_stale(atom, support, refreshed),
            "atom_id": atom.atom_id,
            "scale": str(atom.scale_level),
            "confidence": float(atom.confidence),
            "witnesses": list(atom.anchor_ids),
        }

    def _is_stale(self, atom: MemoryAtom, support: list[str], refreshed: str | None) -> bool:
        """True when a valid EP that the skill's support does not cover entered the same scope after it was minted.
        当该 skill 生成之后、同范围内出现了其支持集未覆盖的有效 EP 时为 True。
        """
        if not refreshed:
            return False
        scope = Scope(user_id=atom.scope.user_id, agent_id=atom.scope.agent_id)
        newer = self.store.axis_select(
            scope,
            where_sql="source_kind = ? AND memory_atoms.created_at > ?",
            params=[SourceKind.ep.value, refreshed],
            order="atom_id ASC",
        )
        return any(candidate.atom_id not in set(support) for candidate in newer)

    def _directive(self, atom: MemoryAtom, *, bank_id: str, now: str) -> dict:
        """One DirectiveResponse-shaped dict from a constraint atom.
        由一个 constraint 原子构造 DirectiveResponse 形状的字典。
        """
        return {
            "id": atom.memory_id,
            "bank_id": bank_id,
            "name": _short_name(atom.text),
            "content": atom.text,
            # Upstream priority is an author-set integer; ours is derived from confidence so the ordering stays
            # evidence-driven - int(confidence * 100) keeps the 0..100 range and the full two-decimal resolution.
            # 上游的 priority 是作者设定的整数；我们的由 confidence 派生，以保持排序由证据驱动 ——
            # int(confidence * 100) 既保留 0..100 的量程，也保留两位小数的分辨率。
            "priority": int(round(float(atom.confidence) * 100)),
            "is_active": _is_live(atom, now),
            "tags": list(atom.topic_tags),
            "created_at": atom.created_at,
            "updated_at": atom.updated_at,
            "atom_id": atom.atom_id,
            "confidence": float(atom.confidence),
            "valid_from": atom.valid_from,
            "valid_to": atom.valid_to,
            "archived_reason": atom.archived_reason,
            "source_kind": str(atom.source_kind),
            "witnesses": list(atom.anchor_ids),
        }

    def _observation_atoms(self, scope: Scope) -> list[MemoryAtom]:
        """Live consolidation products of a scope (InterfaceEP, skill and entity atoms), by atom id.
        某范围内存活的 consolidation 产物（InterfaceEP、skill 与 entity 原子），按原子 id 排序。
        """
        kinds = sorted(CONSOLIDATION_SOURCE_KINDS)
        return self.store.axis_select(
            scope,
            where_sql=f"source_kind IN ({','.join('?' for _ in kinds)})",
            params=kinds,
            order="atom_id ASC",
        )

    def _derived_from(self, scope: Scope, source: MemoryAtom) -> list[MemoryAtom]:
        """Live consolidation products that name `source` as support, or aggregate an entity `source` mentions.
        把 `source` 列为支持、或聚合了 `source` 所提及实体的存活 consolidation 产物。
        """
        entity_ids = {e["entity_id"] for e in self.store.entities_for_atom(source.atom_id)}
        from_iep = {
            iep.atom_id
            for iep in self.store.list_interface_eps(scope, valid_only=True)
            if iep.atom_id and source.atom_id in iep.support_node_ids
        }
        out: list[MemoryAtom] = []
        for atom in self._observation_atoms(scope):
            content = atom.content if isinstance(atom.content, dict) else {}
            support = {str(s) for s in (content.get("support_node_ids") or [])}
            if atom.atom_id in from_iep or source.atom_id in support:
                out.append(atom)
            elif str(atom.source_kind) == SourceKind.entity.value and atom.source_id in entity_ids:
                out.append(atom)
        return out

    def _pending_consolidation(self, scope: Scope) -> int:
        """Count of valid EPs whose atom no valid InterfaceEP names as a support node.
        其原子未被任何有效 InterfaceEP 列为支持节点的有效 EP 数量。
        """
        covered: set[str] = set()
        for iep in self.store.list_interface_eps(scope, valid_only=True):
            covered.update(iep.support_node_ids)
        eps = self.store.list_eps(scope, valid_only=True)
        return sum(1 for ep in eps if not ep.atom_id or ep.atom_id not in covered)

    def _archive_all(
        self,
        scope: Scope,
        targets: list[MemoryAtom],
        *,
        op: str,
        capability: str,
        memory_id: str | None = None,
    ) -> str | None:
        """Archive atoms under one system event/anchor with an ARCHIVE history row each; returns the anchor id.
        在同一个系统事件/锚点之下归档这些原子，并各写一行 ARCHIVE 历史；返回锚点 id。
        """
        if not targets:
            return None
        # Imported here, not at module level: hms4eye is a service-facing seam and CONTRACT §0.4 keeps its
        # module-level import edges inside the foundation group.
        # 在此处而非模块级导入：hms4eye 是面向服务的接缝，契约 §0.4 要求其模块级导入边留在基础组内。
        from hms_elite.ingest.events import system_event

        payload: dict[str, Any] = {
            "op": op,
            "memory_id": memory_id,
            "metadata": {
                "capability": capability,
                "scope": scope.key(),
                "atom_ids": [atom.atom_id for atom in targets],
            },
        }
        with self.store.transaction():
            now = self.store.now()
            _event, anchor = system_event(
                self.store, scope=scope, object_type="system", event_type="deleted", payload=payload
            )
            for atom in targets:
                self.store.archive_atom(atom.atom_id, reason=_CLEARED_REASON, at=now)
                self.store.add_history(
                    memory_id=atom.memory_id,
                    old_memory=atom.text,
                    new_memory=None,
                    event=HistoryEvent.ARCHIVE.value,
                    created_at=atom.created_at or now,
                    updated_at=now,
                    anchor_id=anchor.anchor_id,
                    atom_id=atom.atom_id,
                    details={"capability": capability, "reason": _CLEARED_REASON},
                )
        return anchor.anchor_id


__all__ = [
    "CONSOLIDATION_SOURCE_KINDS",
    "FACT_TYPE_OF_SOURCE_KIND",
    "HMSCapabilities",
]
