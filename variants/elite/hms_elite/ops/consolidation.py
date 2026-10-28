"""StateScheduler: the synchronous orient / collect / consolidate / prune loop over every matched scope.
StateScheduler：对每个匹配范围同步执行 orient / collect / consolidate / prune 的循环。
"""

# Purpose (EN): HMS §3.9 consolidation without a daemon — entity atoms, edges, conflicts, coarsen L2/L3, skills,
#               transitions, compiled state, decay and archival (§16.5).
# 目的（中）：无守护进程的 HMS §3.9 固化——实体原子、建边、冲突、L2/L3 粗化、技能、转移、编译态、衰减与归档（§16.5）。

# Input (EN): An optional scope (None / partial → every matched distinct scope) and an optional phase list.
# 输入（中）：可选范围（None / 部分范围 → 每个匹配的独立范围）与可选的阶段列表。

# Output (EN): {'anchor_id', 'phases': {phase: report}} with counts summed and lists concatenated across scopes.
# 输出（中）：{'anchor_id', 'phases': {phase: report}}，跨范围计数求和、列表拼接。

# State effect (EN): orient is read-only; collect re-embeds and re-ingests; consolidate writes atoms, edges,
#                    InterfaceEPs, skills, transitions, compiled_states; prune lowers confidence and archives (and,
#                    only with retention.enabled + retention.run_in_prune, calls the §16.6 purge).
# 状态影响（中）：orient 只读；collect 重新嵌入与重新摄取；consolidate 写原子、边、InterfaceEP、技能、转移、
#                compiled_states；prune 降低置信度并归档（仅当 retention.enabled + retention.run_in_prune 时
#                才调用 §16.6 的清理）。

from __future__ import annotations

import logging
import math
from datetime import timedelta
from typing import Any

from hms_elite.configs.base import HMSConfig
from hms_elite.configs.prompts import EDGE_PROPOSAL_PROMPT, build_edge_prompt
from hms_elite.enums import (
    EXTRACTABLE_OBJECT_TYPES,
    AtomType,
    Decision,
    EdgeType,
    HistoryEvent,
    NodeKind,
    ObjectType,
    ObservationType,
    Phase,
    Reward,
    Role,
    Scale,
    SourceKind,
)
from hms_elite.exceptions import ValidationError
from hms_elite.graph.operators import coarsen, project, promote_skills
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.index.base import atom_index_attrs
from hms_elite.ingest.atoms import atom_from_entity, create_atom, revise_atom
from hms_elite.ingest.events import system_event
from hms_elite.llms.rule import EXCLUSIVE, decide_revision, normalize, propose_edges
from hms_elite.memory.storage import StateStore
from hms_elite.models import EP, MemoryAtom, Scope
from hms_elite.ops.retention import purge
from hms_elite.ops.transitions import export_transitions
from hms_elite.timeutil import age_days, parse_iso, to_utc_iso
from hms_elite.utils.json import canonical_json
from hms_elite.utils.scoring import recency_score
from hms_elite.utils.text import content_tokens, is_identifier, sha256_hex, text_scoped

logger = logging.getLogger(__name__)

CONSOLIDATE_KEYS = (
    "entity_atoms",
    "temporal_edges",
    "causal_edges",
    "conflicts",
    "interface_eps",
    "projected_edges",
    "interface_eps_l3",
    "skills",
    "transitions",
    "compiled_states",
)
_COARSEN_EDGE_TYPES = ("causal", "use", "temporal")
_PRUNE_TYPES = (AtomType.state.value, AtomType.event.value, AtomType.message.value)
_PRUNE_EXCLUDED_SOURCES = (SourceKind.skill.value, SourceKind.entity.value)
# Entity mentions count first-order atoms only: InterfaceEP / skill atoms are derived from them (no double count).
# 实体提及只统计一阶原子：InterfaceEP / skill 原子由它们派生而来（避免重复计数）。
_DERIVED_SOURCES = frozenset({SourceKind.interface_ep.value, SourceKind.skill.value, SourceKind.entity.value})
_EDGE_BATCH = 20


def _proposal_confidence(item: dict, default: float = 0.7) -> float:
    """Edge-proposal confidence clamped to [0, 1]; unparsable or non-finite (NaN / ±inf) → `default` with a warning.
    建边提议的置信度，钳位到 [0, 1]；无法解析或非有限（NaN / ±inf）→ `default` 并告警。
    """
    raw = item.get("confidence", default)
    try:
        value: float | None = float(raw)
    except (TypeError, ValueError):
        value = None
    # NaN compares False with everything, so max(0.0, min(1.0, nan)) keeps 1.0: reject non-finite before clamping
    # NaN 与任何值比较都为 False，max(0.0, min(1.0, nan)) 会保留 1.0：先拒绝非有限值再钳位
    if value is None or not math.isfinite(value):
        logger.warning(
            "consolidate: edge proposal %r → %r has an invalid confidence %r; using %s",
            item.get("src_ref"),
            item.get("dst_ref"),
            raw,
            default,
        )
        return default
    return max(0.0, min(1.0, value))


def _zero_report(phase: str) -> dict:
    """The empty report of a phase (merge identity).
    某阶段的空报告（合并单位元）。
    """
    if phase == Phase.orient.value:
        return {
            "active_topics": [],
            "active_entities": [],
            "open_failures": [],
            "pending_ep_nodes": [],
            "coarsen_candidates": [],
            "last_compiled": None,
        }
    if phase == Phase.collect.value:
        return {"re_embedded": 0, "re_routed": 0}
    if phase == Phase.consolidate.value:
        return {key: 0 for key in CONSOLIDATE_KEYS}
    return {"examined": 0, "decayed": 0, "archived": [], "live": 0, "expired": 0, "skipped_conflict": 0, "purged": 0}


def _merge(into: dict, report: dict) -> dict:
    """Merge one scope's report into the aggregate: ints summed, lists concatenated, other values last-wins.
    将单个范围的报告合并进汇总：整数求和、列表拼接、其他值以后者为准。
    """
    for key, value in report.items():
        current = into.get(key)
        if isinstance(value, bool):
            into[key] = value
        elif isinstance(value, (int, float)) and isinstance(current, (int, float)) and not isinstance(current, bool):
            into[key] = current + value
        elif isinstance(value, list) and isinstance(current, list):
            into[key] = current + value
        elif value is not None or key not in into:
            into[key] = value
    return into


def _objects_overlap(a: list[str], b: list[str]) -> bool:
    """§11.3 rule 5 overlap: equal normalized strings or token-Jaccard >= 0.5.
    §11.3 规则 5 的重叠：归一化字符串相等或 token-Jaccard >= 0.5。
    """
    for x in a:
        tx = set(content_tokens(x))
        for y in b:
            if normalize(x) == normalize(y):
                return True
            ty = set(content_tokens(y))
            if tx and ty and len(tx & ty) / len(tx | ty) >= 0.5:
                return True
    return False


def _atom_dict(atom: MemoryAtom) -> dict:
    """The revision_decision view of an atom.
    原子在 revision_decision 中的视图。
    """
    content = atom.content if isinstance(atom.content, dict) else {}
    date_observed = content.get("date_observed") or (atom.time_anchor or atom.created_at or "")[:10]
    return {
        "id": atom.atom_id,
        "type": str(atom.obs_type or content.get("obs_type") or ObservationType.fact.value),
        "subject": atom.subject or content.get("subject"),
        "predicate": atom.predicate or content.get("predicate"),
        "objects": list(content.get("objects") or []),
        "relation": content.get("relation") or "single_fact",
        "text": atom.text,
        "date_observed": date_observed,
        "valid_from": atom.valid_from,
    }


class StateScheduler:
    """Synchronous scheduler running the four HMS phases per matched scope.
    按匹配范围同步运行 HMS 四阶段的调度器。
    """

    def __init__(
        self,
        store: StateStore,
        index: Any,
        embedder: Any,
        llm: Any,
        graph: SupportGraph,
        ingestor: Any,
        compiler: Any,
        config: HMSConfig,
    ) -> None:
        """Bind the collaborators.
        绑定协作对象。

        Input / 输入：
            store: The state store.
            store：状态存储。

            index: The vector index.
            index：向量索引。

            embedder: The embedder (collect re-embeds with its identity).
            embedder：嵌入器（collect 以其身份重新嵌入）。

            llm: The LLM (edge_proposal / coarsen when configured as 'llm').
            llm：LLM（配置为 'llm' 时用于 edge_proposal / coarsen）。

            graph: The SupportGraph.
            graph：SupportGraph。

            ingestor: The Ingestor (collect re-routes events through ingest_events).
            ingestor：Ingestor（collect 经 ingest_events 重新分流事件）。

            compiler: The StateCompiler (compile_state).
            compiler：StateCompiler（compile_state）。

            config: HMSConfig.
            config：HMSConfig。

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
        self.llm = llm
        self.graph = graph
        self.ingestor = ingestor
        self.compiler = compiler
        self.config = config

    # ------------------------------------------------------------------ entry
    # ------------------------------------------------------------------ 入口
    @text_scoped
    def run(self, *, scope: Scope | None, phases: list[str] | None) -> dict:
        """Run the requested phases on every scope matched by `scope` (None → every distinct scope).
        对 `scope` 匹配的每个范围运行所请求的阶段（None → 每个独立范围）。

        Input / 输入：
            scope: None, a full scope or a partial scope (Scope.matches semantics over memory_atoms).
            scope：None、完整范围或部分范围（对 memory_atoms 的 Scope.matches 语义）。

            phases: Phase names in run order (None → config.scheduler.phases; unknown → VALIDATION_013).
            phases：按运行顺序的阶段名（None → config.scheduler.phases；未知 → VALIDATION_013）。

        Output / 输出：
            {'anchor_id': <last scheduler anchor>, 'phases': {phase: report}}.
            {'anchor_id': <最后一个调度器锚点>, 'phases': {phase: report}}。

        State effect / 状态影响：
            One scheduler event + anchor per scope, then the phases' writes; meta last_consolidation:<scope> set. The event, its anchor and the meta row are unversioned writes (§6.7): only rows the phases actually changed advance a scope's write version, so a run with nothing to do leaves HMS.context()'s memo of every scope reused.
            每个范围一个调度器事件 + 锚点，随后是各阶段的写入；设置 meta last_consolidation:<scope>。事件、其锚点与 meta 行都是不带版本的写入（§6.7）：只有各阶段真正改变的行才推进范围的写入版本，因此无事可做的运行让每个范围的 HMS.context() 记忆保持复用。
        """
        wanted = [str(getattr(p, "value", p)) for p in (phases or self.config.scheduler.phases)]
        unknown = [p for p in wanted if p not in Phase.values()]
        if unknown:
            raise ValidationError(
                f"unknown phases {unknown}", "VALIDATION_013", details={"phases": unknown, "supported": Phase.values()}
            )
        targets = self._target_scopes(scope)
        aggregate: dict[str, dict] = {p: _zero_report(p) for p in wanted}
        anchor_id: str | None = None
        anchors: dict[str, str] = {}
        for target in targets:
            _ev, sched = system_event(
                self.store,
                scope=target,
                object_type="scheduler",
                event_type="scheduler",
                payload={"phases": wanted},
            )
            anchors[target.key()] = sched.anchor_id
            anchor_id = sched.anchor_id
        # Phase-major order: every scope finishes a phase before any scope starts the next one, so prune never
        # archives atoms that a later scope's consolidate step still counts; inside consolidate, L3 coarsening runs
        # once per (user_id, agent_id) after every run's L2 InterfaceEPs exist.
        # 阶段主序：每个范围都走完当前阶段，才会有范围进入下一阶段，因此 prune 绝不会归档较晚范围的
        # consolidate 步骤仍在统计的原子；在 consolidate 内部，L3 粗化在本次运行的所有 L2 InterfaceEP
        # 都已存在之后，按 (user_id, agent_id) 各执行一次。
        for phase in wanted:
            if phase == Phase.consolidate.value:
                reports = {t.key(): self._consolidate_low(t, anchors[t.key()]) for t in targets}
                l3_done: set[str] = set()
                for target in targets:
                    l3_scope = self._l3_scope(target)
                    if self.config.graph.coarsen_l3 and l3_scope.key() not in l3_done:
                        l3_done.add(l3_scope.key())
                        self._coarsen_l3(l3_scope, reports[target.key()], anchors[target.key()])
                for target in targets:
                    self._consolidate_high(target, anchors[target.key()], reports[target.key()])
                    _merge(aggregate[phase], reports[target.key()])
                continue
            for target in targets:
                sched_anchor = anchors[target.key()]
                if phase == Phase.orient.value:
                    report = self.orient(target)
                elif phase == Phase.collect.value:
                    report = self.collect(target)
                else:
                    report = self.prune(target, sched_anchor)
                _merge(aggregate[phase], report)
        for target in targets:
            # The meta row is bookkeeping no scope's read depends on: an unversioned write (§6.7), so a run that
            # changed no row — an idle consolidate, the daemon's periodic pass — leaves every scope's HMS.context()
            # memo reused; the phases' own row changes (atoms, edges, InterfaceEPs, skills, decay) have already
            # advanced the target's write version.
            # meta 行是任何范围的读取都不依赖的簿记：不带版本的写入（§6.7），因此没有改变任何行的运行——空转的
            # consolidate、守护进程的周期性扫描——让每个范围的 HMS.context() 记忆保持复用；各阶段自身改变的行
            # （原子、边、InterfaceEP、技能、衰减）已经推进了目标范围的写入版本。
            with self.store.unversioned():
                self.store.set_meta(f"last_consolidation:{target.key()}", self.store.now())
        return {"anchor_id": anchor_id, "phases": aggregate}

    def _target_scopes(self, scope: Scope | None) -> list[Scope]:
        """Distinct first-order atom scopes matched by `scope` (a scope without atoms is processed as itself).
        `scope` 匹配的、拥有一阶原子的独立范围（无原子的范围按其自身处理）。
        """
        # Only first-order atoms define a run scope: entity / InterfaceEP / skill atoms are derived objects that live
        # at the (user_id, agent_id) level and must never turn that level into a scheduler target of its own.
        # 只有一阶原子定义运行范围：entity / InterfaceEP / skill 原子是派生对象，位于 (user_id, agent_id)
        # 层级，绝不能让该层级本身变成调度器自己的目标范围。
        atoms = self.store.axis_select(
            Scope(),
            where_sql="source_kind NOT IN (?,?,?)",
            params=sorted(_DERIVED_SOURCES),
            include_expired=True,
            order="atom_id ASC",
        )
        found: dict[str, Scope] = {}
        for atom in atoms:
            if scope is None or scope.matches(atom.scope):
                found.setdefault(atom.scope.key(), atom.scope)
        if not found and scope is not None and not scope.is_empty():
            return [scope]
        return [found[key] for key in sorted(found)]

    # ------------------------------------------------------------------ orient
    # ------------------------------------------------------------------ orient（定向）
    def orient(self, scope: Scope) -> dict:
        """Read-only plan: active topics/entities, open failures, pending EP nodes, coarsen candidates, last card.
        只读规划：活跃主题/实体、未解决失败、待处理 EP 节点、粗化候选、最近卡片。

        Input / 输入：
            scope: One concrete scope.
            scope：一个具体范围。

        Output / 输出：
            {'active_topics', 'active_entities', 'open_failures', 'pending_ep_nodes', 'coarsen_candidates', 'last_compiled'}.
            {'active_topics', 'active_entities', 'open_failures', 'pending_ep_nodes', 'coarsen_candidates', 'last_compiled'}。

        State effect / 状态影响：
            None.
            无。
        """
        store, cfg = self.store, self.config
        now = store.now()
        window_start = to_utc_iso(parse_iso(now) - timedelta(days=int(cfg.scheduler.orient_window_days)))
        recent = store.axis_select(
            scope,
            where_sql="time_anchor >= ?",
            params=[window_start],
            as_of=now,
            order="time_anchor DESC, created_at DESC, atom_id ASC",
        )
        topic_counts: dict[str, int] = {}
        entity_counts: dict[str, int] = {}
        for atom in recent:
            for topic in atom.topic_tags:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            for key in atom.entity_ids:
                entity_counts[key] = entity_counts.get(key, 0) + 1
        top_topics = [t for t, _n in sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
        top_entities = [e for e, _n in sorted(entity_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
        eps = store.list_eps(scope, valid_only=True)
        by_run: dict[str | None, list[EP]] = {}
        for ep in eps:
            by_run.setdefault(ep.scope.run_id, []).append(ep)
        open_failures: list[str] = []
        for run_eps in by_run.values():
            ordered = sorted(run_eps, key=lambda e: (e.time_anchor, e.ep_id))
            for i, ep in enumerate(ordered):
                if str(ep.slots.r) != Reward.negative.value:
                    continue
                later_positive = any(str(o.slots.r) == Reward.positive.value for o in ordered[i + 1 :])
                if not later_positive:
                    open_failures.append(ep.ep_id)
        pending: list[str] = []
        for ep in eps:
            if ep.atom_id and not store.edges_touching([ep.atom_id], valid_only=True, direction="out"):
                pending.append(ep.atom_id)
        candidates = self.graph.components(
            scope,
            kinds=(NodeKind.ep.value,),
            edge_types=_COARSEN_EDGE_TYPES,
            gap_hours=float(cfg.graph.coarsen_gap_hours),
            min_size=int(cfg.graph.coarsen_min_nodes),
            max_size=int(cfg.graph.coarsen_max_nodes),
            scale=Scale.L1_micro.value,
        )
        last = store.last_compiled_state(scope)
        if last is not None:
            last = {k: v for k, v in last.items() if k != "card"}
        return {
            "active_topics": top_topics,
            "active_entities": top_entities,
            "open_failures": sorted(open_failures),
            "pending_ep_nodes": sorted(pending),
            "coarsen_candidates": candidates,
            "last_compiled": last,
        }

    # ------------------------------------------------------------------ collect
    # ------------------------------------------------------------------ collect（收集）
    def collect(self, scope: Scope) -> dict:
        """Re-embed atoms without a current vector and re-route extractable events that never produced atoms.
        重新嵌入缺少当前向量的原子，并重新分流从未产生原子的可抽取事件。

        Input / 输入：
            scope: One concrete scope.
            scope：一个具体范围。

        Output / 输出：
            {'re_embedded': n, 're_routed': n}.
            {'re_embedded': n, 're_routed': n}。

        State effect / 状态影响：
            index.upsert per re-embedded atom; ingest_events(infer=True) for the re-routed events (unlocked LLM phase).
            每个重新嵌入的原子一次 index.upsert；重新分流的事件经 ingest_events(infer=True)（无锁 LLM 阶段）。
        """
        store = self.store
        identity = self.embedder.identity
        re_embedded = 0
        for atom_id in store.atoms_without_vectors(scope, embedder=identity):
            stale_row = bool(store.load_vectors([atom_id]))
            if self.index.get(atom_id) is not None and not stale_row:
                continue
            atom = store.get_atom(atom_id)
            if atom is None:
                continue
            self.index.upsert(
                atom_id, list(self.embedder.embed(atom.text, "add")), embedder=identity, attrs=atom_index_attrs(atom)
            )
            re_embedded += 1
        events = [
            ev
            for ev in store.events_without_atoms(scope, object_types=sorted(EXTRACTABLE_OBJECT_TYPES))
            if not (str(ev.object_type) == ObjectType.message.value and str(ev.role or "") == Role.system.value)
        ]
        if events:
            self.ingestor.ingest_events(events, infer=True)
        return {"re_embedded": re_embedded, "re_routed": len(events)}

    # ------------------------------------------------------------------ consolidate
    # ------------------------------------------------------------------ consolidate（固化）
    def consolidate(self, scope: Scope, sched_anchor: str) -> dict:
        """Steps 0–7 for one scope: entity atoms, temporal/causal edges, conflicts, coarsen L2 + L3, skills, transitions, card.
        单个范围的第 0–7 步：实体原子、temporal/causal 边、冲突、L2 + L3 粗化、技能、转移、卡片。

        Input / 输入：
            scope: One concrete scope.
            scope：一个具体范围。

            sched_anchor: The scheduler anchor witnessing every row created here.
            sched_anchor：见证此处创建的每一行的调度器锚点。

        Output / 输出：
            {'entity_atoms', 'temporal_edges', 'causal_edges', 'conflicts', 'interface_eps', 'projected_edges',
             'interface_eps_l3', 'skills', 'transitions', 'compiled_states'}.
            {'entity_atoms', 'temporal_edges', 'causal_edges', 'conflicts', 'interface_eps', 'projected_edges',
             'interface_eps_l3', 'skills', 'transitions', 'compiled_states'}。

        State effect / 状态影响：
            Each step runs in its own transaction; writes atoms, edges, InterfaceEPs, skills, transitions, compiled_states. A step that changes no row moves no write version, and the compiled-state archive is an unversioned write (§6.7).
            每一步在各自事务内运行；写入原子、边、InterfaceEP、技能、转移与 compiled_states。没有改变任何行的步骤不推进任何写入版本，编译态归档是不带版本的写入（§6.7）。
        """
        report = self._consolidate_low(scope, sched_anchor)
        if self.config.graph.coarsen_l3:
            self._coarsen_l3(self._l3_scope(scope), report, sched_anchor)
        self._consolidate_high(scope, sched_anchor, report)
        return report

    def _consolidate_low(self, scope: Scope, sched_anchor: str) -> dict:
        """Steps 0–4: entity atoms, temporal edges, causal proposals, conflicts, L2 coarsen + project.
        第 0–4 步：实体原子、temporal 边、causal 提议、冲突、L2 粗化 + 投影。
        """
        report = {key: 0 for key in CONSOLIDATE_KEYS}
        report["entity_atoms"] = self._entity_atoms(scope, sched_anchor)
        pairs = self._temporal_edges(scope, report)
        report["causal_edges"] = self._causal_edges(scope, pairs)
        report["conflicts"] = self._conflicts(scope)
        self._coarsen_l2(scope, report)
        return report

    def _consolidate_high(self, scope: Scope, sched_anchor: str, report: dict) -> None:
        """Steps 5–7: skills, transitions and the persisted Work State (counts added to `report`).
        第 5–7 步：技能、转移与持久化工作态（计数累加进 `report`）。
        """
        report["skills"] += len(
            promote_skills(
                self.store,
                self.index,
                self.embedder,
                self.graph,
                scope=scope,
                at=self.store.now(),
                anchor_id=sched_anchor,
            )
        )
        before = self.store.stats()["transitions"]
        with self.store.transaction():
            # export_transitions opens its own scope-less transaction: attribute it here so a first export advances
            # this scope's write version only (§6.7), never the global one (INSERT OR IGNORE: a repeat writes and
            # moves nothing).
            # export_transitions 自行打开不带范围的事务：在此归属范围，使首次导出只推进本范围的写入版本（§6.7）而非
            # 全局版本（INSERT OR IGNORE：重复导出不写任何行、不推进任何版本）。
            self.store.touch_scope(scope)
            export_transitions(self.store, scope=scope, since=None, scale=None, kinds=None)
        report["transitions"] += self.store.stats()["transitions"] - before
        if self.config.scheduler.compile_state:
            report["compiled_states"] += self._compile_state(scope, sched_anchor)

    def _l3_scope(self, scope: Scope) -> Scope:
        """The (user_id, agent_id) scope at which L2 InterfaceEPs of several runs are coarsened to L3.
        多个运行的 L2 InterfaceEP 被粗化为 L3 时所处的 (user_id, agent_id) 范围。
        """
        if scope.user_id or scope.agent_id:
            return Scope(user_id=scope.user_id, agent_id=scope.agent_id)
        return scope

    def _entity_atoms(self, scope: Scope, sched_anchor: str) -> int:
        """Step 0: one entity-scale atom per entity linked to live atoms of the scope (created or revised).
        第 0 步：为链接到范围内存活原子的每个实体建立（或修订）一个 entity 尺度原子。
        """
        store = self.store
        count = 0
        with store.transaction():
            # Scope-less rows (edges, history, decay) may be written here: attribute the transaction to the scope so
            # the commit advances this scope's write version only (HMS.context() memo, §6.7), not the global one —
            # and only if the phase actually changed rows (an idle phase moves no version).
            # 这里可能写入无范围的行（边、历史、衰减）：把事务归于该范围，使提交只推进本范围的写入版本
            # （HMS.context() 记忆，§6.7）而非全局版本——且仅当该阶段真的改变了行（空转的阶段不推进任何版本）。
            store.touch_scope(scope)
            for entity in store.entities_in_scope(scope):
                linked = [
                    a.atom_id
                    for a in store.get_atoms(store.entity_linked_atoms(entity["entity_id"]))
                    if str(a.source_kind) not in _DERIVED_SOURCES
                ]
                if not linked:
                    continue
                anchors: set[str] = set()
                for atom_id in linked:
                    anchors.update(store.atom_anchor_ids(atom_id))
                if not anchors:
                    continue
                anchor_ids = sorted(anchors)
                if entity.get("user_id") or entity.get("agent_id"):
                    entity_scope = Scope(user_id=entity.get("user_id"), agent_id=entity.get("agent_id"))
                else:
                    first = store.get_atom(linked[0])
                    entity_scope = Scope(run_id=scope.run_id or (first.scope.run_id if first else None))
                existing = store.axis_select(
                    Scope(),
                    where_sql="source_kind = ? AND source_id = ?",
                    params=[SourceKind.entity.value, entity["entity_id"]],
                    order="atom_id ASC",
                    limit=1,
                )
                fresh = atom_from_entity(entity, scope=entity_scope, anchor_ids=anchor_ids, mentions=len(linked))
                if not existing:
                    create_atom(
                        store,
                        self.index,
                        self.embedder,
                        fresh,
                        extra_mentions=[fresh.text],
                        history_anchor_id=sched_anchor,
                    )
                    count += 1
                    continue
                old = existing[0]
                content = old.content if isinstance(old.content, dict) else {}
                same_mentions = int(content.get("mentions") or 0) == len(linked)
                same_aliases = list(content.get("aliases") or []) == list(entity.get("aliases") or [])
                if same_mentions and same_aliases:
                    continue
                revise_atom(
                    store,
                    self.index,
                    self.embedder,
                    old=old,
                    new=fresh.model_copy(update={"scope": old.scope}),
                    witness_anchor_ids=[sched_anchor],
                    at=store.now(),
                    created_by="consolidation",
                    extra_mentions=[fresh.text],
                )
                count += 1
        return count

    def _valid_eps(self, scope: Scope) -> list[EP]:
        """Valid EPs of the scope that own a node, ordered by (time_anchor, ep_id).
        范围内拥有节点的有效 EP，按 (time_anchor, ep_id) 排序。
        """
        eps = [ep for ep in self.store.list_eps(scope, valid_only=True) if ep.atom_id]
        return sorted(eps, key=lambda e: (e.time_anchor, e.ep_id))

    def _temporal_edges(self, scope: Scope, report: dict) -> list[tuple[EP, EP]]:
        """Step 1: temporal edges between consecutive EPs per run within temporal_window_hours (MERGE-aware count).
        第 1 步：每个运行内 temporal_window_hours 之内相邻 EP 之间的 temporal 边（MERGE 感知计数）。
        """
        store = self.store
        window = float(self.config.graph.temporal_window_hours)
        by_run: dict[str | None, list[EP]] = {}
        for ep in self._valid_eps(scope):
            by_run.setdefault(ep.scope.run_id, []).append(ep)
        pairs: list[tuple[EP, EP]] = []
        created = 0
        with store.transaction():
            # Scope-less rows (edges, history, decay) may be written here: attribute the transaction to the scope so
            # the commit advances this scope's write version only (HMS.context() memo, §6.7), not the global one —
            # and only if the phase actually changed rows (an idle phase moves no version).
            # 这里可能写入无范围的行（边、历史、衰减）：把事务归于该范围，使提交只推进本范围的写入版本
            # （HMS.context() 记忆，§6.7）而非全局版本——且仅当该阶段真的改变了行（空转的阶段不推进任何版本）。
            store.touch_scope(scope)
            for run_id in sorted(by_run, key=lambda r: r or ""):
                ordered = by_run[run_id]
                for prev, nxt in zip(ordered, ordered[1:]):
                    if age_days(prev.time_anchor, nxt.time_anchor) * 24.0 > window:
                        continue
                    pairs.append((prev, nxt))
                    new = store.find_edge(prev.atom_id, nxt.atom_id, EdgeType.temporal.value) is None
                    self.graph.insert_edge(
                        src_id=prev.atom_id,
                        dst_id=nxt.atom_id,
                        edge_type=EdgeType.temporal.value,
                        witness_anchor_ids=[nxt.anchor_ids[0]],
                        created_by="consolidation",
                        proposer="timestamp",
                        confidence=0.5,
                        label=f"temporal: {prev.slots.a[:40]} → {nxt.slots.a[:40]}",
                    )
                    if new:
                        created += 1
        report["temporal_edges"] = created
        return pairs

    def _candidate_pairs(self, eps: list[EP], temporal: list[tuple[EP, EP]]) -> list[tuple[str, str]]:
        """Temporal neighbours ∪ pairs sharing an identifier entity ∪ uses_ep_ids pairs (as node id pairs).
        temporal 相邻对 ∪ 共享标识符实体的对 ∪ uses_ep_ids 对（以节点 id 对表示）。
        """
        store = self.store
        atoms = {a.atom_id: a for a in store.get_atoms([ep.atom_id for ep in eps])}
        by_ep = {ep.ep_id: ep for ep in eps}
        pairs: dict[tuple[str, str], None] = {}
        for prev, nxt in temporal:
            pairs.setdefault((prev.atom_id, nxt.atom_id), None)
        for edge in store.edges_between([ep.atom_id for ep in eps], valid_only=True):
            if str(edge.edge_type) == EdgeType.temporal.value:
                pairs.setdefault((edge.src_id, edge.dst_id), None)
        for i, x in enumerate(eps):
            keys_x = {k for k in atoms[x.atom_id].entity_ids if is_identifier(k)} if x.atom_id in atoms else set()
            if not keys_x:
                continue
            for y in eps[i + 1 :]:
                keys_y = {k for k in atoms[y.atom_id].entity_ids if is_identifier(k)} if y.atom_id in atoms else set()
                if keys_x & keys_y:
                    pairs.setdefault((x.atom_id, y.atom_id), None)
        for ep in eps:
            for used_id in ep.uses_ep_ids:
                used = by_ep.get(used_id)
                if used is not None and used.atom_id:
                    pairs.setdefault((used.atom_id, ep.atom_id), None)
        return list(pairs)

    def _causal_edges(self, scope: Scope, temporal: list[tuple[EP, EP]]) -> int:
        """Step 2: rule or LLM edge proposals over the candidate pairs; MERGE-aware count of new edges.
        第 2 步：对候选对做规则或 LLM 建边提议；MERGE 感知地统计新边。
        """
        store = self.store
        eps = self._valid_eps(scope)
        if len(eps) < 2:
            return 0
        pairs = self._candidate_pairs(eps, temporal)
        if not pairs:
            return 0
        atoms = {a.atom_id: a for a in store.get_atoms([ep.atom_id for ep in eps])}
        ref_of = {ep.atom_id: f"n{i}" for i, ep in enumerate(eps)}
        node_of_ref = {ref: node_id for node_id, ref in ref_of.items()}
        nodes: list[dict] = []
        witnesses: list[dict] = []
        anchor_of_ref: dict[str, str] = {}
        for ep in eps:
            atom = atoms.get(ep.atom_id)
            refs: list[str] = []
            for anchor_id in ep.anchor_ids:
                ref = f"w{len(anchor_of_ref)}"
                anchor_of_ref[ref] = anchor_id
                anchor = store.get_anchor(anchor_id)
                witnesses.append(
                    {
                        "ref": ref,
                        "node_ref": ref_of[ep.atom_id],
                        "source_ref": anchor.source_ref if anchor else None,
                        "time": anchor.timestamp if anchor else ep.time_anchor,
                    }
                )
                refs.append(ref)
            nodes.append(
                {
                    "ref": ref_of[ep.atom_id],
                    "kind": NodeKind.ep.value,
                    "u": ep.slots.u,
                    "b": ep.slots.b,
                    "a": ep.slots.a,
                    "y": ep.slots.y,
                    "r": str(ep.slots.r),
                    "subject": None,
                    "predicate": None,
                    "objects": [],
                    "time": ep.time_anchor,
                    "entities": list(atom.entity_ids) if atom else [],
                    "topics": list(atom.topic_tags) if atom else list(ep.topic_tags),
                    "anchor_refs": refs,
                }
            )
        ref_pairs = [[ref_of[x], ref_of[y]] for x, y in pairs if x in ref_of and y in ref_of]
        proposals: list[dict] = []
        if str(self.config.graph.causal_proposer) == "llm" and self.llm is not None:
            proposer = str(self.llm.identity)
            # 无锁阶段：此处没有打开事务
            for start in range(0, len(ref_pairs), _EDGE_BATCH):  # unlocked: no transaction is open here
                batch = ref_pairs[start : start + _EDGE_BATCH]
                raw = self.llm.generate_json(
                    "edge_proposal",
                    system=EDGE_PROPOSAL_PROMPT,
                    user=build_edge_prompt(nodes=nodes, pairs=batch, witnesses=witnesses),
                )
                proposals.extend(item for item in (raw if isinstance(raw, list) else []) if isinstance(item, dict))
        else:
            proposer = "rule"
            proposals = propose_edges(nodes, ref_pairs, witnesses)
        created = 0
        with store.transaction():
            # Scope-less rows (edges, history, decay) may be written here: attribute the transaction to the scope so
            # the commit advances this scope's write version only (HMS.context() memo, §6.7), not the global one —
            # and only if the phase actually changed rows (an idle phase moves no version).
            # 这里可能写入无范围的行（边、历史、衰减）：把事务归于该范围，使提交只推进本范围的写入版本
            # （HMS.context() 记忆，§6.7）而非全局版本——且仅当该阶段真的改变了行（空转的阶段不推进任何版本）。
            store.touch_scope(scope)
            for item in proposals:
                src = node_of_ref.get(str(item.get("src_ref")))
                dst = node_of_ref.get(str(item.get("dst_ref")))
                edge_type = str(item.get("edge_type") or "")
                refs = item.get("witness_refs")
                if not isinstance(refs, list) or not all(isinstance(w, str) for w in refs):
                    # §6.2: witness_refs must be a list of refs; scalar / object / nested list → dropped, never raised
                    # §6.2：witness_refs 必须是引用列表；标量 / 对象 / 嵌套列表一律丢弃，绝不抛错中止整次运行
                    logger.warning(
                        "consolidate: dropping edge proposal %r → %r: witness_refs must be a list of refs, got %r",
                        item.get("src_ref"),
                        item.get("dst_ref"),
                        refs,
                    )
                    continue
                witness_ids = [anchor_of_ref[w] for w in refs if w in anchor_of_ref]
                if not src or not dst or src == dst or edge_type not in EdgeType.values() or not witness_ids:
                    continue
                new = store.find_edge(src, dst, edge_type) is None
                self.graph.insert_edge(
                    src_id=src,
                    dst_id=dst,
                    edge_type=edge_type,
                    witness_anchor_ids=witness_ids,
                    created_by="consolidation",
                    proposer=proposer,
                    confidence=_proposal_confidence(item),
                    label=str(item.get("label") or ""),
                )
                if new:
                    created += 1
        return created

    def _conflicts(self, scope: Scope) -> int:
        """Step 3: conflict edges for exclusive-predicate contradictions (+ deferred revision decisions).
        第 3 步：为互斥谓词矛盾建立 conflict 边（+ 延迟的修订决策）。
        """
        store = self.store
        deferred = str(self.config.ingest.revision_mode) == "deferred"
        atoms = store.axis_select(
            scope,
            where_sql="atom_type = ? AND subject IS NOT NULL AND predicate IS NOT NULL",
            params=[AtomType.state.value],
            order="time_anchor ASC, created_at ASC, atom_id ASC",
        )
        atoms = [a for a in atoms if str(a.source_kind) != SourceKind.skill.value]
        count = 0
        with store.transaction():
            # Scope-less rows (edges, history, decay) may be written here: attribute the transaction to the scope so
            # the commit advances this scope's write version only (HMS.context() memo, §6.7), not the global one —
            # and only if the phase actually changed rows (an idle phase moves no version).
            # 这里可能写入无范围的行（边、历史、衰减）：把事务归于该范围，使提交只推进本范围的写入版本
            # （HMS.context() 记忆，§6.7）而非全局版本——且仅当该阶段真的改变了行（空转的阶段不推进任何版本）。
            store.touch_scope(scope)
            for i, older in enumerate(atoms):
                for newer in atoms[i + 1 :]:
                    if older.subject != newer.subject:
                        continue
                    if self._linked(older.atom_id, newer.atom_id):
                        continue
                    old_d, new_d = _atom_dict(older), _atom_dict(newer)
                    exclusive = (
                        older.predicate == newer.predicate
                        and older.predicate in EXCLUSIVE
                        and not _objects_overlap(old_d["objects"], new_d["objects"])
                    )
                    if exclusive:
                        self._conflict_edge(older, newer, "consolidation: exclusive attribute conflict")
                        count += 1
                        continue
                    if not deferred:
                        continue
                    decision, why = decide_revision(new_d, old_d)
                    if decision == Decision.REVISE:
                        self._deferred_revise(older, newer, why)
                        count += 1
                    elif decision == Decision.CONFLICT:
                        self._conflict_edge(older, newer, why)
                        count += 1
        return count

    def _linked(self, a: str, b: str) -> bool:
        """True when a revision or conflict edge (any validity) already joins the two atoms.
        两个原子之间已存在 revision 或 conflict 边（不论有效性）时为真。
        """
        for edge in self.store.edges_between([a, b], valid_only=False):
            if str(edge.edge_type) in (EdgeType.revision.value, EdgeType.conflict.value):
                return True
        return False

    def _conflict_edge(self, older: MemoryAtom, newer: MemoryAtom, label: str) -> None:
        """conflict edge older → newer witnessed by the newer primary anchor + CONFLICT history on the older lineage.
        由较新主锚点见证的 conflict 边 older → newer，加上旧谱系上的 CONFLICT 历史。
        """
        store = self.store
        now = store.now()
        edge_id = self.graph.insert_edge(
            src_id=older.atom_id,
            dst_id=newer.atom_id,
            edge_type=EdgeType.conflict.value,
            witness_anchor_ids=[newer.anchor_ids[0]],
            created_by="consolidation",
            proposer="rule",
            confidence=0.6,
            label=label,
        )
        store.add_history(
            memory_id=older.memory_id,
            old_memory=older.text,
            new_memory=newer.text,
            event=HistoryEvent.CONFLICT.value,
            created_at=older.created_at or now,
            updated_at=now,
            anchor_id=newer.anchor_ids[0],
            atom_id=older.atom_id,
            edge_id=edge_id,
            details={"conflicts_with": newer.memory_id},
        )

    def _deferred_revise(self, older: MemoryAtom, newer: MemoryAtom, why: str) -> None:
        """Deferred REVISE: archive the older atom, revision edge older → newer, UPDATE history (no new atom).
        延迟 REVISE：归档旧原子，revision 边 older → newer，UPDATE 历史（不新建原子）。
        """
        store = self.store
        at = max(newer.valid_from or "", older.valid_from or "") or store.now()
        store.archive_atom(older.atom_id, reason="revised", at=at, valid_to=at)
        edge_id = self.graph.insert_edge(
            src_id=older.atom_id,
            dst_id=newer.atom_id,
            edge_type=EdgeType.revision.value,
            witness_anchor_ids=list(newer.anchor_ids),
            created_by="consolidation",
            proposer="rule",
            confidence=0.9,
            label=why,
        )
        store.add_history(
            memory_id=older.memory_id,
            old_memory=older.text,
            new_memory=newer.text,
            event=HistoryEvent.UPDATE.value,
            created_at=older.created_at or at,
            updated_at=at,
            anchor_id=newer.anchor_ids[0],
            atom_id=older.atom_id,
            edge_id=edge_id,
            successor_id=newer.atom_id,
        )

    def _coarsen_components(
        self, scope: Scope, components: list[list[str]], *, target_scale: str, report: dict, key: str
    ) -> None:
        """coarsen + project every component; counts new InterfaceEPs under `key` and new high edges.
        对每个分量 coarsen + project；在 `key` 下统计新 InterfaceEP，并统计新高层边。
        """
        store, cfg = self.store, self.config
        for members in components:
            known_ieps = {i.interface_ep_id for i in store.list_interface_eps(scope, valid_only=False)}
            iep = coarsen(
                store,
                self.index,
                self.embedder,
                self.graph,
                members,
                target_scale=target_scale,
                policy=str(cfg.graph.coarsen_policy),
                llm=self.llm,
                at=store.now(),
                scope=scope,
                min_cohesion=float(cfg.graph.min_cohesion),
            )
            if iep is None:
                continue
            if iep.interface_ep_id not in known_ieps:
                report[key] += 1
            known_edges = {e.edge_id for e in store.list_edges(scope, valid_only=False, scale=target_scale)}
            high = project(self.graph, iep, at=store.now())
            report["projected_edges"] += len([h for h in high if h not in known_edges])

    def _coarsen_l2(self, scope: Scope, report: dict) -> None:
        """Step 4: EP components → L2 InterfaceEPs → projected edges.
        第 4 步：EP 分量 → L2 InterfaceEP → 投影边。
        """
        cfg = self.config
        components = self.graph.components(
            scope,
            kinds=(NodeKind.ep.value,),
            edge_types=_COARSEN_EDGE_TYPES,
            gap_hours=float(cfg.graph.coarsen_gap_hours),
            min_size=int(cfg.graph.coarsen_min_nodes),
            max_size=int(cfg.graph.coarsen_max_nodes),
            scale=Scale.L1_micro.value,
        )
        self._coarsen_components(
            scope, components, target_scale=Scale.L2_meso.value, report=report, key="interface_eps"
        )

    def _coarsen_l3(self, scope: Scope, report: dict, sched_anchor: str) -> None:
        """Step 4b: L2 InterfaceEP components → L3 InterfaceEPs → projected edges (+ skills of the L3 scope).
        第 4b 步：L2 InterfaceEP 分量 → L3 InterfaceEP → 投影边（+ L3 范围的技能）。
        """
        cfg = self.config
        components = self.graph.components(
            scope,
            kinds=(NodeKind.interface_ep.value,),
            edge_types=_COARSEN_EDGE_TYPES,
            gap_hours=float(cfg.graph.coarsen_gap_hours) * 4.0,
            min_size=int(cfg.graph.coarsen_min_nodes),
            max_size=int(cfg.graph.coarsen_max_nodes),
            scale=Scale.L2_meso.value,
        )
        if not components:
            return
        before = report["interface_eps_l3"]
        self._coarsen_components(
            scope, components, target_scale=Scale.L3_macro.value, report=report, key="interface_eps_l3"
        )
        if report["interface_eps_l3"] > before:
            report["skills"] += len(
                promote_skills(
                    self.store,
                    self.index,
                    self.embedder,
                    self.graph,
                    scope=scope,
                    at=self.store.now(),
                    anchor_id=sched_anchor,
                )
            )

    def _compile_state(self, scope: Scope, sched_anchor: str) -> int:
        """Step 7: persist the deep Work State of the scope (compile stays read-only; only the scheduler persists).
        第 7 步：持久化该范围的 deep 工作态（compile 本身只读；只有调度器持久化）。
        """
        store = self.store
        last = store.last_ep(scope, run_id=scope.run_id)
        goal = last.slots.u if last is not None and last.slots.u else "work state"
        ir = self.compiler.compile(goal, scope=scope, filters=scope.as_filters(), mode="deep")
        card = ir.to_text()
        with store.transaction(), store.unversioned():
            # The compiled-state archive and the work_state meta row are read by orient and the state surface, never
            # by compile(): unversioned rows (§6.7), so persisting the card does not by itself drop HMS.context()'s
            # memo of the scope.
            # 编译态归档与 work_state meta 行由 orient 与状态表面读取，compile() 从不读取：不带版本的行（§6.7），
            # 因此持久化卡片本身不会丢弃该范围的 HMS.context() 记忆。
            compiled_id = store.insert_compiled_state(
                scope=scope,
                as_of=ir.as_of,
                mode="deep",
                card=card,
                context_json=ir.model_dump_json(),
                used_tokens=int(ir.used_tokens),
                anchor_id=sched_anchor,
            )
            store.set_meta(
                f"work_state:{scope.key()}",
                canonical_json(
                    {
                        "hash": sha256_hex(card),
                        "as_of": ir.as_of,
                        "used_tokens": int(ir.used_tokens),
                        "compiled_id": compiled_id,
                    }
                ),
            )
        return 1

    # ------------------------------------------------------------------ prune
    # ------------------------------------------------------------------ prune（剪枝归档）
    def prune(self, scope: Scope, sched_anchor: str) -> dict:
        """decay_stale_state: lower the confidence of stale atoms, then archive atoms below confidence_min.
        decay_stale_state：降低陈旧原子的置信度，然后归档低于 confidence_min 的原子。

        Input / 输入：
            scope: One concrete scope.
            scope：一个具体范围。

            sched_anchor: The scheduler anchor witnessing the UPDATE / ARCHIVE history rows.
            sched_anchor：见证 UPDATE / ARCHIVE 历史行的调度器锚点。

        Output / 输出：
            {'examined', 'decayed', 'archived': [atom_ids], 'live', 'expired', 'skipped_conflict', 'purged'} ('purged' is 0 unless retention.enabled and retention.run_in_prune are both set).
            {'examined', 'decayed', 'archived': [atom_ids], 'live', 'expired', 'skipped_conflict', 'purged'}（除非 retention.enabled 与 retention.run_in_prune 同时开启，否则 'purged' 为 0）。

        State effect / 状态影响：
            One transaction: set_atom_confidence + UPDATE history, archive_atom + ARCHIVE history; then, only when retention.enabled and retention.run_in_prune, the §16.6 purge(dry_run=False, keep_witnessed=True) in its own transaction.
            一个事务：set_atom_confidence + UPDATE 历史、archive_atom + ARCHIVE 历史；随后仅当 retention.enabled 且 retention.run_in_prune 时，在独立事务内执行 §16.6 的 purge(dry_run=False, keep_witnessed=True)。
        """
        store, cfg = self.store, self.config
        now = store.now()
        tau = float(cfg.scheduler.staleness_tau)
        conf_min = float(cfg.scheduler.confidence_min)
        rate = float(cfg.scheduler.decay_rate)
        half_life = float(cfg.scoring.half_life_days)
        use_half_life = float(cfg.scheduler.use_recency_half_life_days)
        candidates = store.axis_select(
            scope,
            where_sql="atom_type IN (?,?,?) AND (obs_type IS NULL OR obs_type <> ?) AND source_kind NOT IN (?,?)",
            params=[*_PRUNE_TYPES, ObservationType.constraint.value, *_PRUNE_EXCLUDED_SOURCES],
            as_of=now,
            order="atom_id ASC",
        )
        examined = [a for a in candidates if "important" not in a.topic_tags]
        decayed = 0
        archived: list[str] = []
        skipped_conflict = 0
        with store.transaction():
            # Scope-less rows (edges, history, decay) may be written here: attribute the transaction to the scope so
            # the commit advances this scope's write version only (HMS.context() memo, §6.7), not the global one —
            # and only if the phase actually changed rows (an idle phase moves no version).
            # 这里可能写入无范围的行（边、历史、衰减）：把事务归于该范围，使提交只推进本范围的写入版本
            # （HMS.context() 记忆，§6.7）而非全局版本——且仅当该阶段真的改变了行（空转的阶段不推进任何版本）。
            store.touch_scope(scope)
            for atom in examined:
                recency = recency_score(atom.time_anchor or atom.created_at, as_of=now, half_life_days=half_life)
                boost = max(0.0, min(0.4, float(atom.confidence) - 0.6))
                touching = store.edges_touching(
                    [atom.atom_id], valid_only=True, edge_types=[EdgeType.use.value, EdgeType.causal.value]
                )
                newest = max((e.created_at or "" for e in touching), default="")
                use_recency = recency_score(newest or None, as_of=now, half_life_days=use_half_life)
                staleness = (1.0 - recency) * (1.0 - boost) * (1.0 - use_recency)
                confidence = float(atom.confidence)
                if staleness > tau:
                    new_conf = max(conf_min / 2.0, round(confidence - rate * staleness, 4))
                    if new_conf != confidence:
                        store.set_atom_confidence(atom.atom_id, new_conf, at=now)
                        store.add_history(
                            memory_id=atom.memory_id,
                            old_memory=atom.text,
                            new_memory=atom.text,
                            event=HistoryEvent.UPDATE.value,
                            created_at=atom.created_at or now,
                            updated_at=now,
                            anchor_id=sched_anchor,
                            atom_id=atom.atom_id,
                            details={"decay": round(staleness, 4), "confidence": [confidence, new_conf]},
                        )
                        decayed += 1
                        confidence = new_conf
                if confidence < conf_min:
                    conflicts = store.edges_touching(
                        [atom.atom_id], valid_only=True, edge_types=[EdgeType.conflict.value]
                    )
                    if any(e.resolved_at is None for e in conflicts):
                        skipped_conflict += 1
                        continue
                    store.archive_atom(atom.atom_id, reason="decayed", at=now)
                    store.add_history(
                        memory_id=atom.memory_id,
                        old_memory=atom.text,
                        new_memory=None,
                        event=HistoryEvent.ARCHIVE.value,
                        created_at=atom.created_at or now,
                        updated_at=now,
                        anchor_id=sched_anchor,
                        atom_id=atom.atom_id,
                        details={"staleness": round(staleness, 4), "confidence": confidence},
                    )
                    archived.append(atom.atom_id)
        purged = 0
        # Retention is an escape hatch, never a default: both switches must be on before prune materializes a purge.
        # 保留策略是逃生舱而非默认：两个开关都打开，prune 才会真正执行清理。
        if cfg.retention.enabled and cfg.retention.run_in_prune:
            purged = purge(
                store,
                self.index,
                scope=scope,
                older_than_days=cfg.retention.older_than_days,
                dry_run=False,
                keep_witnessed=True,
            )["purged"]
        live = store.count_atoms(scope, valid_only=True)
        total = store.count_atoms(scope, valid_only=False)
        return {
            "examined": len(examined),
            "decayed": decayed,
            "archived": archived,
            "live": live,
            "expired": max(0, total - live),
            "skipped_conflict": skipped_conflict,
            "purged": purged,
        }


__all__ = ["CONSOLIDATE_KEYS", "StateScheduler"]
