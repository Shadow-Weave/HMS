"""Ingest orchestration: Txn A (events + anchors) → unlocked extraction/embedding → Txn B (atoms, edges, transitions).
摄取编排：事务 A（事件 + 锚点）→ 无锁抽取/嵌入 → 事务 B（原子、边、转移）。
"""

# Purpose (EN): Implement ingest_messages / ingest_events end to end over the StateStore, index, embedder and LLM.
# 目的（中）：基于 StateStore、索引、嵌入器与 LLM 端到端实现 ingest_messages / ingest_events。

# Input (EN): Coerced messages or SourceEvents plus scope / metadata / infer / memory_type / prompt / event_time / valid_to hints.
# 输入（中）：规整后的消息或 SourceEvent，以及 scope / metadata / infer / memory_type / prompt / event_time / valid_to 提示。

# Output (EN): IngestResult (Mem0 rows with ADD|UPDATE|NONE events, event entries, ep/edge/constraint ids, dropped items).
# 输出（中）：IngestResult（带 ADD|UPDATE|NONE 事件的 Mem0 行、事件条目、ep/边/约束 id、被丢弃项）。

# State effect (EN): Writes every ingest plane table; LLM and embedding calls never run inside a transaction
# (extraction_mode 'single' = one memory_extraction call per message batch, 'staged' = up to three). Messages longer
# than config.ingest.segment_max_chars are cut into segment units before extraction; a unit never gets its own
# SourceEvent or Anchor (one event, one anchor), every observation anchors to the parent message.
# 状态影响（中）：写入摄取涉及的所有平面表；LLM 与嵌入调用绝不在事务内运行
# （extraction_mode 'single' = 每批消息一次 memory_extraction 调用，'staged' = 最多三次）。超过
# config.ingest.segment_max_chars 的消息在抽取前切成分段单元；单元绝不拥有自己的 SourceEvent 或 Anchor（一事一锚），
# 每条观察都锚到父消息。

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import pydantic

from hms_elite.configs.base import HMSConfig
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import AtomType, Decision, ObjectType, Relation, Role, Scale, SourceKind
from hms_elite.exceptions import EmbeddingError, LLMError, ValidationError
from hms_elite.index.base import IndexBase
from hms_elite.ingest.atoms import (
    atom_from_ep,
    atom_from_event,
    atom_from_message,
    atom_from_observation,
    conflict_atom,
    create_atom,
    relation_atom_from_observation,
    revise_atom,
)
from hms_elite.ingest.closure import EdgeSpec, close_structured_events
from hms_elite.ingest.events import anchor_events, events_from_messages, merge_image_descriptions
from hms_elite.ingest.extraction import (
    EXTRACTION_OUTPUTS,
    ExtractionBundle,
    decide_revisions,
    extract_all,
    extract_eps,
    extract_observations,
    neighbourhood,
    procedural_summary,
    verdict_similarity,
)
from hms_elite.ingest.router import route_event
from hms_elite.llms.base import LLMBase
from hms_elite.memory.storage import StateStore
from hms_elite.models import EP, Anchor, MemoryAtom, Observation, Scope, SourceEvent, Transition
from hms_elite.segmentation import segment
from hms_elite.timeutil import age_days, date_of, to_utc_iso
from hms_elite.utils.factory import EntityExtractorFactory
from hms_elite.utils.text import text_scoped

logger = logging.getLogger(__name__)

_STRUCTURED_TYPES = frozenset(
    {
        ObjectType.tool_call.value,
        ObjectType.file_diff.value,
        ObjectType.test_log.value,
        ObjectType.sensor_trace.value,
    }
)
_CONSTRAINT_ORDER = "time_anchor ASC, created_at ASC, atom_id ASC"
# 'single' = one memory_extraction call per batch (default); 'staged' = observation → ep → revision (up to 3 calls).
# 'single' = 每批一次 memory_extraction 调用（默认）；'staged' = 观察 → EP → 修订（最多 3 次调用）。
EXTRACTION_MODES = ("single", "staged")
# Revision candidates of the single call come from the neighbourhood, restricted to the same atom kinds that
# store.revision_candidates searches on the staged path (observation-derived state/event atoms).
# 单次调用的修订候选来自邻域，并限定为分段路径 store.revision_candidates 所搜索的同类原子（观察派生的 state/event 原子）。
_CANDIDATE_ATOM_TYPES = frozenset({"state", "event"})


@dataclass
class IngestResult:
    """What one ingest call produced: Mem0 rows plus the ids of every derived object.
    一次摄取调用的产出：Mem0 行以及每个派生对象的 id。
    """

    # Mem0 行 + 'event'（'ADD'|'UPDATE'|'NONE'）+ 可选 'revises' + 可选 'conflicts_with'
    rows: list[dict] = field(
        default_factory=list
    )  # Mem0 rows + 'event' ('ADD'|'UPDATE'|'NONE') + 'revises'? + 'conflicts_with'?
    # 事件条目形如 [{'event_id','anchor_id','object_type','dedup'}]
    events: list[dict] = field(default_factory=list)  # [{'event_id','anchor_id','object_type','dedup'}]
    ep_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    constraint_ids: list[str] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    # Verdicts the similarity guard (ingest.verdict_min_similarity) turned into ADD:
    # [{'kind': 'verdict', 'verdict': 'NONE'|'REVISE'|'CONFLICT', 'target_id', 'similarity', 'text'}].
    # 被相似度守卫（ingest.verdict_min_similarity）改为 ADD 的裁决：
    # [{'kind': 'verdict', 'verdict': 'NONE'|'REVISE'|'CONFLICT', 'target_id', 'similarity', 'text'}]。
    downgraded: list[dict] = field(default_factory=list)


@dataclass
class _Plan:
    """One atom to materialize in Txn B and how (duplicate → NONE, else decision).
    事务 B 中待物化的一个原子及其处理方式（重复 → NONE，否则按决策）。
    """

    kind: str
    atom: MemoryAtom
    ev: SourceEvent | None = None
    anc: Anchor | None = None
    obs: Observation | None = None
    ep: EP | None = None
    duplicate: MemoryAtom | None = None
    batch_duplicate: int | None = None
    vector: list[float] | None = None
    decision: tuple[Decision, str | None, str] | None = None
    stored: MemoryAtom | None = None


def _event_entry(ev: SourceEvent, anc: Anchor, dedup: bool) -> dict:
    """The IngestResult.events entry of an anchored event.
    已落锚事件在 IngestResult.events 中的条目。
    """
    return {
        "event_id": ev.event_id,
        "anchor_id": anc.anchor_id,
        "object_type": str(ev.object_type),
        "dedup": bool(dedup),
    }


def _same_statement_head(new: MemoryAtom, old: MemoryAtom) -> bool:
    """True when both atoms carry the same non-empty subject and predicate (case-insensitive, stripped).
    两个原子的主语与谓词都非空且相同（不分大小写、去空白）时为真。
    """
    head = tuple((v or "").strip().lower() for v in (new.subject, new.predicate))
    return all(head) and head == tuple((v or "").strip().lower() for v in (old.subject, old.predicate))


def _row(atom: MemoryAtom, event: str, **extra: Any) -> dict:
    """A Mem0 row with the ingest 'event' key and optional extras.
    带摄取 'event' 键与可选附加键的 Mem0 行。
    """
    row = atom.to_public_dict()
    row["event"] = event
    row.update(extra)
    return row


class Ingestor:
    """The ingest pipeline bound to one store / index / embedder / LLM / config / entity extractor.
    绑定到一组 store / 索引 / 嵌入器 / LLM / 配置 / 实体抽取器的摄取流水线。
    """

    def __init__(
        self,
        store: StateStore,
        index: IndexBase,
        embedder: EmbedderBase,
        llm: LLMBase,
        config: HMSConfig,
        entity_extractor: EntityExtractorBase | None = None,
        extraction_mode: str = "single",
        extraction_output: str | None = None,
    ) -> None:
        """Bind the collaborators; the entity extractor is built from config.ingest.entity_extractor when not injected, and extraction_output falls back to config.ingest.extraction_output (no hard-coded default of its own).
        绑定协作对象；未注入时按 config.ingest.entity_extractor 构造实体抽取器，extraction_output 未给出时回退到 config.ingest.extraction_output（自身没有硬编码默认值）。
        """
        if extraction_output is None:
            extraction_output = str(config.ingest.extraction_output)
        if extraction_mode not in EXTRACTION_MODES:
            raise ValidationError(
                f"unknown extraction_mode {extraction_mode!r}",
                "VALIDATION_013",
                details={"extraction_mode": extraction_mode, "supported": list(EXTRACTION_MODES)},
            )
        if extraction_output not in EXTRACTION_OUTPUTS:
            raise ValidationError(
                f"unknown extraction_output {extraction_output!r}",
                "VALIDATION_013",
                details={"extraction_output": extraction_output, "supported": list(EXTRACTION_OUTPUTS)},
            )
        if extraction_output == "lines" and extraction_mode != "single":
            raise ValidationError(
                "extraction_output='lines' supports extraction_mode='single' only (CONTRACT §6.6) / "
                "extraction_output='lines' 只支持 extraction_mode='single'（CONTRACT §6.6）",
                "VALIDATION_013",
                details={"extraction_output": extraction_output, "extraction_mode": extraction_mode, "supported": ["single"]},
            )
        self.extraction_mode = extraction_mode
        # 'full' = the LLM emits every observation field; 'compact' = COMPACT_OBSERVATION_KEYS only, the rest derived
        # locally (ingest.derive). Both apply to both extraction modes (memory_extraction_compact /
        # observation_extraction_compact). 'lines' = tab-separated text (memory_extraction_lines, CONTRACT §6.6),
        # single mode only — checked above.
        # 'full' = LLM 输出每个观察字段；'compact' = 只输出 COMPACT_OBSERVATION_KEYS，其余在本地推导（ingest.derive）。
        # 两者都适用于两种抽取模式（memory_extraction_compact / observation_extraction_compact）。'lines' = tab 分隔文本
        # （memory_extraction_lines，CONTRACT §6.6），仅 single 模式——上方已校验。
        self.extraction_output = extraction_output
        self.store = store
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.config = config
        # The facade injects one shared extractor (also bound to the rule LLM); standalone Ingestors build their own.
        # 门面注入同一个共享抽取器（也绑定到规则 LLM）；独立构造的 Ingestor 自行创建。
        self.entity_extractor: EntityExtractorBase = entity_extractor or EntityExtractorFactory.create(
            config.ingest.entity_extractor, config.ingest.entity_extractor_config
        )

    # ------------------------------------------------------------------ public API
    # ------------------------------------------------------------------ 公开 API
    @text_scoped
    def ingest_messages(
        self,
        messages: list[dict],
        *,
        scope: Scope,
        user_meta: dict,
        infer: bool,
        memory_type: str | None,
        prompt: str | None,
        event_time: str | None,
        valid_to: str | None,
    ) -> IngestResult:
        """Mem0 add(): anchor the messages (Txn A), then raw atoms / procedural summary / LLM extraction (Txn B).
        Mem0 add()：先为消息落锚（事务 A），再走原始原子 / 过程摘要 / LLM 抽取路径（事务 B）。

        Input / 输入：
            messages: Coerced messages (memory.utils.coerce_messages).
            messages：规整后的消息（memory.utils.coerce_messages）。

            scope: Session scope (>= 1 id).
            scope：会话范围（至少一个 id）。

            user_meta: Caller metadata stored on events and atoms.
            user_meta：存于事件与原子上的调用方元数据。

            infer: False → one raw message atom per message; True → extraction.
            infer：False → 每条消息一个原始消息原子；True → 抽取。

            memory_type: 'procedural_memory' (with agent_id) selects the procedural branch.
            memory_type：'procedural_memory'（且有 agent_id）选择过程记忆分支。

            prompt: Replacement system prompt for the procedural branch.
            prompt：过程记忆分支的替代系统提示词。

            event_time: Event time of messages without their own timestamp (None → store.now()).
            event_time：无自带时间戳的消息的事件时间（None → store.now()）。

            valid_to: Validity end applied to every created atom (from expiration_date).
            valid_to：应用于每个新建原子的有效期终点（来自 expiration_date）。

        Output / 输出：
            IngestResult; rows in extraction order.
            IngestResult；行按抽取顺序。

        State effect / 状态影响：
            Txn A writes source_events/anchors; Txn B writes observations, eps, atoms, axes, vectors, nodes, edges, history, transitions.
            事务 A 写入 source_events/anchors；事务 B 写入 observations、eps、原子、轴、向量、节点、边、历史与转移。
        """
        scope.require_any(context="add")
        base_time = to_utc_iso(event_time) if event_time else self.store.now()
        # Vision pre-pass (unlocked, before Txn A): image parts become text so the anchored payload already carries
        # parts (sha256/mime) + described_parts; a no-op unless config.llm.enable_vision.
        # 视觉前置阶段（无锁，在事务 A 之前）：图片部件变为文字，使落锚载荷已含 parts（sha256/mime）+
        # described_parts；除非 config.llm.enable_vision，否则为空操作。
        messages = self._describe_images(messages)
        events = events_from_messages(
            messages,
            scope=scope,
            event_time=base_time,
            user_meta=user_meta or {},
            visibility=str(self.config.visibility_default),
        )
        anchored = anchor_events(self.store, events)
        result = IngestResult(events=[_event_entry(ev, anc, dedup) for ev, anc, dedup in anchored])
        if not infer:
            self._ingest_raw_messages(
                anchored, scope=scope, user_meta=user_meta or {}, valid_to=valid_to, result=result
            )
            return result
        if scope.agent_id and memory_type == "procedural_memory":
            self._ingest_procedural(
                anchored,
                messages,
                scope=scope,
                prompt=prompt,
                user_meta=user_meta or {},
                valid_to=valid_to,
                result=result,
            )
            return result
        self._ingest_message_batch(anchored, scope=scope, user_meta=user_meta or {}, valid_to=valid_to, result=result)
        return result

    @text_scoped
    def ingest_events(self, events: list[SourceEvent], *, infer: bool) -> IngestResult:
        """HMS add_events(): anchor all events (Txn A), extract messages per scope, close structured events without an LLM.
        HMS add_events()：为全部事件落锚（事务 A），按范围抽取消息，并在无 LLM 的情况下闭合结构化事件。

        Input / 输入：
            events: Validated SourceEvents (ingest.events.events_from_dicts).
            events：已校验的 SourceEvent（ingest.events.events_from_dicts）。

            infer: Whether message events go through LLM extraction (False → raw message atoms).
            infer：消息事件是否走 LLM 抽取（False → 原始消息原子）。

        Output / 输出：
            IngestResult over every scope in the batch.
            覆盖批次中所有范围的 IngestResult。

        State effect / 状态影响：
            Txn A + one Txn B per (scope, path); zero LLM calls on the structured path.
            事务 A 加每个 (范围, 路径) 一个事务 B；结构化路径零 LLM 调用。
        """
        ordered = sorted(enumerate(events), key=lambda pair: (pair[1].timestamp, pair[0]))
        anchored = anchor_events(self.store, [ev for _index, ev in ordered])
        result = IngestResult(events=[_event_entry(ev, anc, dedup) for ev, anc, dedup in anchored])
        groups: dict[str, tuple[Scope, list[tuple[SourceEvent, Anchor, bool]]]] = {}
        for ev, anc, dedup in anchored:
            groups.setdefault(ev.scope.key(), (ev.scope, []))[1].append((ev, anc, dedup))
        for _key, (scope, items) in groups.items():
            messages = [t for t in items if str(t[0].object_type) == ObjectType.message.value]
            structured = [t for t in items if str(t[0].object_type) in _STRUCTURED_TYPES]
            if messages:
                user_meta = dict(messages[0][0].user_meta or {})
                if infer:
                    self._ingest_message_batch(messages, scope=scope, user_meta=user_meta, valid_to=None, result=result)
                else:
                    self._ingest_raw_messages(messages, scope=scope, user_meta=user_meta, valid_to=None, result=result)
            if structured:
                self._ingest_structured(structured, scope=scope, result=result)
        return result

    # ------------------------------------------------------------------ shared helpers
    # ------------------------------------------------------------------ 共享辅助函数
    def _describe_images(self, messages: list[dict]) -> list[dict]:
        """Unlocked vision pre-pass: annotate every image part with llm.describe_image (no-op unless enable_vision).
        无锁视觉前置阶段：用 llm.describe_image 标注每个图片部件（除非 enable_vision，否则为空操作）。
        """
        llm_cfg = self.config.llm
        if not getattr(llm_cfg, "enable_vision", False):
            return messages
        out: list[dict] = []
        for index, message in enumerate(messages):
            image_parts = message.get("image_parts") if isinstance(message, dict) else None
            if not image_parts:
                out.append(message)
                continue
            annotated: list[Any] = []
            for part_index, part in enumerate(image_parts):
                url = part.get("url") if isinstance(part, dict) else None
                if not isinstance(url, str) or not url:
                    annotated.append(part)
                    continue
                detail = part.get("detail") or getattr(llm_cfg, "vision_details", None)
                try:
                    description = self.llm.describe_image(url, detail=detail if isinstance(detail, str) else None)
                except Exception as exc:
                    # Nothing has been written yet (Txn A comes after), so the caller can simply retry the add().
                    # 此时尚未写入任何东西（事务 A 在后），调用方直接重试 add() 即可。
                    code = exc.error_code if isinstance(exc, LLMError) else "LLM_001"
                    raise LLMError(
                        f"describe_image failed for messages[{index}] part {part_index}: {exc}",
                        code,
                        details={
                            "task": "describe_image",
                            "provider": self.llm.identity,
                            "index": index,
                            "part": part_index,
                        },
                    ) from exc
                copy = dict(part)
                if isinstance(description, str) and description.strip():
                    copy["description"] = description.strip()
                    copy["described_by"] = self.llm.identity
                annotated.append(copy)
            out.append({**message, "image_parts": annotated})
        return out

    @contextmanager
    def _txn_b(self) -> Iterator[list[str]]:
        """Txn B wrapper: on failure roll back and drop the vectors of atoms created inside it.
        事务 B 包装：失败时回滚，并删除其中创建的原子的向量。
        """
        created: list[str] = []
        try:
            with self.store.transaction():
                yield created
        except BaseException:
            for atom_id in created:
                try:
                    self.index.delete(atom_id)
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug("index cleanup failed for %s", atom_id)
            raise

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed texts unlocked: batch first, per-item fallback; any remaining failure raises EmbeddingError.
        无锁嵌入文本：先批量，再逐项回退；仍失败则抛出 EmbeddingError。
        """
        if not texts:
            return []
        try:
            return self.embedder.embed_batch(list(texts), "add")
        except EmbeddingError:
            out: list[list[float]] = []
            for index, text in enumerate(texts):
                try:
                    out.append(self.embedder.embed(text, "add"))
                except EmbeddingError:
                    raise
                except Exception as exc:
                    raise EmbeddingError(
                        f"embedding failed for item {index}: {exc}", "EMBED_001", details={"index": index}
                    ) from exc
            return out

    def _hint(self, atom: MemoryAtom, *, user_meta: dict, valid_to: str | None) -> MemoryAtom:
        """Apply the caller metadata and the valid_to hint to a built atom.
        将调用方元数据与 valid_to 提示应用到已构造的原子。
        """
        update: dict[str, Any] = {}
        if user_meta:
            update["user_meta"] = dict(user_meta)
        if valid_to is not None:
            update["valid_to"] = to_utc_iso(valid_to)
        return atom.model_copy(update=update) if update else atom

    def _duplicate_of(self, scope: Scope, atom: MemoryAtom) -> MemoryAtom | None:
        """The valid atom in scope with the same text hash, if any (Mem0 exact dedup).
        范围内文本哈希相同的有效原子（若有）（Mem0 精确去重）。
        """
        found = self.store.find_atoms_by_hash(scope, atom.hash, valid_only=True)
        return found[0] if found else None

    def _mark_duplicates(self, plans: list[_Plan], scope: Scope) -> None:
        """Mark plans whose atom text already exists (in scope or earlier in the batch).
        标记原子文本已存在（范围内或本批次更早处）的计划。
        """
        seen: dict[str, int] = {}
        for index, plan in enumerate(plans):
            if plan.kind in ("ep", "procedural"):
                continue
            plan.duplicate = self._duplicate_of(scope, plan.atom)
            if plan.duplicate is None and plan.atom.hash in seen:
                plan.batch_duplicate = seen[plan.atom.hash]
            elif plan.duplicate is None:
                seen[plan.atom.hash] = index

    def _embed_plans(self, plans: list[_Plan]) -> None:
        """Compute vectors for every plan that will create an atom.
        为每个将创建原子的计划计算向量。
        """
        pending = [p for p in plans if p.duplicate is None and p.batch_duplicate is None]
        vectors = self._embed_all([p.atom.text for p in pending])
        for plan, vector in zip(pending, vectors):
            plan.vector = vector

    @staticmethod
    def _extraction_units(
        events: list[tuple[SourceEvent, Anchor]], *, max_chars: int
    ) -> list[tuple[SourceEvent, Anchor]]:
        """Split every message longer than max_chars into (segment copy, parent anchor) units; shorter ones pass through.
        把每条长于 max_chars 的消息切成（分段副本，父锚点）单元；更短的原样通过。
        """
        units: list[tuple[SourceEvent, Anchor]] = []
        for ev, anc in events:
            text = str((ev.payload or {}).get("text") or "")
            if len(text) <= max_chars:
                units.append((ev, anc))
                continue
            pieces = segment(text, max_chars=max_chars)
            if len(pieces) <= 1:
                units.append((ev, anc))
                continue
            for piece in pieces:
                payload = dict(ev.payload or {})
                payload["text"] = piece.text
                units.append((ev.model_copy(update={"payload": payload}), anc))
        return units

    def _constraints(self, scope: Scope) -> list[MemoryAtom]:
        """Valid constraint atoms in scope ordered by time_anchor.
        范围内按 time_anchor 排序的有效约束原子。
        """
        return self.store.axis_select(
            scope, where_sql="memory_atoms.obs_type = ?", params=("constraint",), order=_CONSTRAINT_ORDER
        )

    def _none_row(self, target_id: str, anchor_id: str, result: IngestResult) -> MemoryAtom:
        """NONE outcome: add the anchor as a support witness, bump mentions and emit a NONE row.
        NONE 结果：将锚点加为支持见证、增加提及计数并输出 NONE 行。
        """
        self.store.add_atom_anchor(target_id, anchor_id, role="support")
        self.store.bump_mentions(target_id)
        target = self.store.get_atom(target_id)
        if target is None:  # pragma: no cover - target was read moments ago
            raise RuntimeError(f"atom {target_id} vanished")
        result.rows.append(_row(target, "NONE"))
        return target

    def _anchor(self, anchor_id: str, cache: dict[str, Anchor]) -> Anchor | None:
        """Resolve an anchor through the batch cache, else the store.
        通过批次缓存解析锚点，否则查 store。
        """
        anchor = cache.get(anchor_id)
        if anchor is None:
            anchor = self.store.get_anchor(anchor_id)
            if anchor is not None:
                cache[anchor_id] = anchor
        return anchor

    # ------------------------------------------------------------------ infer=False
    # ------------------------------------------------------------------ infer=False 路径
    def _ingest_raw_messages(
        self,
        anchored: list[tuple[SourceEvent, Anchor, bool]],
        *,
        scope: Scope,
        user_meta: dict,
        valid_to: str | None,
        result: IngestResult,
    ) -> None:
        """Raw path: one message atom per non-system message (entities still linked, history ADD with actor/role).
        原始路径：每条非 system 消息一个消息原子（仍链接实体，历史 ADD 记录 actor/role）。
        """
        plans: list[_Plan] = []
        for ev, anc, _dedup in anchored:
            if ev.role is not None and str(ev.role) == Role.system.value:
                continue
            text = str((ev.payload or {}).get("text") or "")
            if not text.strip():
                result.dropped.append({"kind": "message", "event_id": ev.event_id, "reason": "empty_text"})
                continue
            atom = self._hint(atom_from_message(ev, anc, scope=scope), user_meta=user_meta, valid_to=valid_to)
            plans.append(_Plan(kind="message", atom=atom, ev=ev, anc=anc))
        self._mark_duplicates(plans, scope)
        self._embed_plans(plans)
        with self._txn_b() as created:
            for plan in plans:
                target = plan.duplicate
                if target is None and plan.batch_duplicate is not None:
                    target = plans[plan.batch_duplicate].stored
                if target is not None:
                    plan.stored = self._none_row(target.atom_id, plan.anc.anchor_id, result)
                    row = result.rows[-1]
                    row.setdefault("actor_id", plan.stored.actor_id)
                    row.setdefault("role", plan.stored.role)
                    continue
                stored = create_atom(
                    self.store,
                    self.index,
                    self.embedder,
                    plan.atom,
                    vector=plan.vector,
                    history_event="ADD",
                    actor_id=plan.ev.actor_id,
                    role=str(plan.ev.role) if plan.ev.role is not None else None,
                    extractor=self.entity_extractor,
                )
                created.append(stored.atom_id)
                plan.stored = stored
                row = _row(stored, "ADD")
                row.setdefault("actor_id", stored.actor_id)
                row.setdefault("role", stored.role)
                result.rows.append(row)

    # ------------------------------------------------------------------ procedural branch
    # ------------------------------------------------------------------ 程序性分支
    def _ingest_procedural(
        self,
        anchored: list[tuple[SourceEvent, Anchor, bool]],
        messages: list[dict],
        *,
        scope: Scope,
        prompt: str | None,
        user_meta: dict,
        valid_to: str | None,
        result: IngestResult,
    ) -> None:
        """Procedural branch: one L1 episode atom carrying the markdown summary, witnessed by every message anchor.
        过程记忆分支：一个承载 markdown 摘要的 L1 情节原子，由全部消息锚点见证。
        """
        # The summary sees the same text the anchors carry: image descriptions merged into the message content.
        # 摘要看到的文本与锚点承载的一致：图片描述已并入消息内容。
        merged = [{**m, "content": merge_image_descriptions(m)} if isinstance(m, dict) else m for m in messages]
        markdown = procedural_summary(self.llm, messages=merged, prompt=prompt)
        first_ev = anchored[0][0]
        last_ev = anchored[-1][0]
        atom = MemoryAtom(
            atom_type=AtomType.episode,
            content={"text": markdown, "memory_type": "procedural_memory"},
            text=markdown,
            time_anchor=last_ev.timestamp,
            valid_from=last_ev.timestamp,
            scale_level=Scale.L1_micro,
            role=Role.assistant,
            source_kind=SourceKind.procedural,
            source_id=first_ev.event_id,
            extracted_by=self.llm.identity,
            visibility=str(self.config.visibility_default),
            scope=scope,
            memory_type="procedural_memory",
            anchor_ids=[anc.anchor_id for _ev, anc, _dedup in anchored],
        )
        atom = self._hint(atom, user_meta=user_meta, valid_to=valid_to)
        vector = self._embed_all([atom.text])[0]
        with self._txn_b() as created:
            stored = create_atom(
                self.store,
                self.index,
                self.embedder,
                atom,
                vector=vector,
                history_event="ADD",
                role=Role.assistant.value,
                extractor=self.entity_extractor,
            )
            created.append(stored.atom_id)
            result.rows.append(_row(stored, "ADD"))

    # ------------------------------------------------------------------ LLM path (steps 4–7)
    # ------------------------------------------------------------------ LLM 路径（第 4–7 步）
    def _ingest_message_batch(
        self,
        anchored: list[tuple[SourceEvent, Anchor, bool]],
        *,
        scope: Scope,
        user_meta: dict,
        valid_to: str | None,
        result: IngestResult,
    ) -> None:
        """Steps 4–7 of §12.6: route, extract unlocked (one call or three), decide revisions, then materialize in Txn B.
        §12.6 第 4–7 步：分流、无锁抽取（一次或三次调用）、修订决策，然后在事务 B 中物化。
        """
        cfg = self.config.ingest
        obs_events = [(ev, anc) for ev, anc, _d in anchored if "observation" in route_event(ev)]
        ep_events = [(ev, anc) for ev, anc, _d in anchored if "state" in route_event(ev)]
        if not obs_events:
            return
        # Extraction units: long messages are segmented (HMS §5.1) so the LLM extracts per semantic atom; each unit is
        # a payload-only copy of its parent message keyed to the parent's anchor, so event_idx → parent anchor and
        # no new SourceEvent / Anchor ever appears. EP closure needs the whole message, so ep_events stay whole.
        # 抽取单元：长消息被切分（HMS §5.1），LLM 按语义原子抽取；每个单元只是父消息的载荷副本、键到父消息的锚点，
        # 因此 event_idx → 父锚点，绝不出现新的 SourceEvent / Anchor。EP 闭包需要整条消息，故 ep_events 保持整条。
        obs_units = self._extraction_units(obs_events, max_chars=int(cfg.segment_max_chars))
        anchors: dict[str, Anchor] = {anc.anchor_id: anc for _ev, anc, _d in anchored}
        events_by_anchor: dict[str, SourceEvent] = {anc.anchor_id: ev for ev, anc, _d in anchored}
        observation_date = date_of(obs_events[-1][0].timestamp)
        current_date = date_of(self.store.now())
        # ---- unlocked: neighbourhood, extraction, atoms, embeddings, revision decisions
        # ---- 无锁阶段：邻域查找、抽取、原子、嵌入、修订决策
        texts = [f"{ev.role}: {(ev.payload or {}).get('text') or ''}" for ev, _anc in obs_events]
        existing, last = neighbourhood(
            self.store,
            self.index,
            self.embedder,
            scope=scope,
            texts=texts,
            k=int(cfg.neighbourhood_k),
            last_k=int(cfg.last_events_k),
        )
        batch_event_ids = {ev.event_id for ev, _anc, _d in anchored}
        last = [ev for ev in last if ev.event_id not in batch_event_ids]
        id_map = {str(e["id"]): str(e["atom_id"]) for e in existing}
        previous_ep = self.store.last_ep(scope, run_id=scope.run_id)
        recent: list[EP] = []
        if ep_events:
            recent = self.store.list_eps(scope, valid_only=True)
            recent = recent[-int(cfg.neighbourhood_k) :] if cfg.neighbourhood_k > 0 else []
        bundle: ExtractionBundle | None = None
        if self.extraction_mode == "single":
            # One round trip: the neighbourhood doubles as the revision candidate set (same synthetic ids), shown only
            # when revisions are decided inline; deferred / off never present candidates, so every decision is ADD.
            # 单次往返：邻域同时充当修订候选集（同一套合成 id），只在 inline 修订时呈现；
            # deferred / off 从不呈现候选，因此所有决策都是 ADD。
            candidates: list[tuple[str, MemoryAtom]] = []
            if str(cfg.revision_mode) == "inline" and existing:
                by_id = {a.atom_id: a for a in self.store.get_atoms([str(e["atom_id"]) for e in existing])}
                for entry in existing:
                    atom = by_id.get(str(entry["atom_id"]))
                    if atom is not None and str(atom.atom_type) in _CANDIDATE_ATOM_TYPES and atom.obs_type:
                        candidates.append((str(entry["id"]), atom))
            bundle = extract_all(
                self.llm,
                events=obs_units,
                ep_events=ep_events,
                existing=existing,
                candidates=candidates,
                last_events=last,
                known_constraints=self._constraints(scope) if ep_events else [],
                recent_eps=recent,
                observation_date=observation_date,
                current_date=current_date,
                custom_instructions=self.config.custom_instructions,
                scope=scope,
                id_map=id_map,
                output=self.extraction_output,
                entity_extractor=self.entity_extractor,
                zh_rules=bool(cfg.zh_rules),
            )
            observations, eps = bundle.observations, bundle.eps
        else:
            observations = extract_observations(
                self.llm,
                events=obs_units,
                existing=existing,
                last_events=last,
                observation_date=observation_date,
                current_date=current_date,
                custom_instructions=self.config.custom_instructions,
                scope=scope,
                id_map=id_map,
                output=self.extraction_output,
                entity_extractor=self.entity_extractor,
                zh_rules=bool(cfg.zh_rules),
            )
            eps = []
            if ep_events:
                eps = extract_eps(
                    self.llm,
                    events=ep_events,
                    known_constraints=self._constraints(scope),
                    recent_eps=recent,
                    observation_date=observation_date,
                    scope=scope,
                )
        observations = [o.model_copy(update={"obs_id": self.store.new_id("obs")}) for o in observations]
        eps = [e.model_copy(update={"ep_id": self.store.new_id("ep")}) for e in eps]
        plans: list[_Plan] = []
        for position, obs in enumerate(observations):
            anc = self._anchor(obs.anchor_ids[0], anchors)
            ev = events_by_anchor.get(obs.anchor_ids[0])
            try:
                atom = atom_from_observation(
                    obs, scope=scope, time_anchor=anc.timestamp if anc else None, actor_id=ev.actor_id if ev else None
                )
            except pydantic.ValidationError as exc:
                # Second line of defence behind extraction's per-item guard: an observation whose atom would break a
                # MemoryAtom invariant is dropped alone, with a warning, instead of aborting the whole batch.
                # 抽取层逐条守卫之后的第二道防线：原子会破坏 MemoryAtom 不变量的观察单独丢弃并记警告，
                # 而不是让整个批次中止。
                logger.warning("ingest: observation %r dropped, its atom is invalid: %s", obs.text, exc)
                result.dropped.append({"kind": "observation", "text": obs.text, "reason": "invalid_atom"})
                continue
            plans.append(
                _Plan(
                    kind="observation",
                    atom=self._hint(atom, user_meta=user_meta, valid_to=valid_to),
                    ev=ev,
                    anc=anc,
                    obs=obs,
                    decision=bundle.decisions[position] if bundle is not None else None,
                )
            )
        for ep in eps:
            anc = self._anchor(ep.anchor_ids[0], anchors)
            atom = atom_from_ep(ep, scope=scope)
            plans.append(
                _Plan(kind="ep", atom=self._hint(atom, user_meta=user_meta, valid_to=valid_to), anc=anc, ep=ep)
            )
        self._mark_duplicates(plans, scope)
        relation_plans: dict[int, _Plan] = {}
        for index, plan in enumerate(plans):
            if (
                plan.kind == "observation"
                and plan.duplicate is None
                and plan.batch_duplicate is None
                and str(plan.obs.relation) == Relation.comparison.value
                and cfg.emit_relation_atoms
            ):
                rel = relation_atom_from_observation(
                    plan.obs, scope=scope, time_anchor=plan.atom.time_anchor, actor_id=plan.atom.actor_id
                )
                relation_plans[index] = _Plan(
                    kind="relation",
                    atom=self._hint(rel, user_meta=user_meta, valid_to=valid_to),
                    obs=plan.obs,
                    anc=plan.anc,
                )
        self._embed_plans(plans + list(relation_plans.values()))
        fresh = [p for p in plans if p.kind == "observation" and p.duplicate is None and p.batch_duplicate is None]
        if bundle is None:
            decisions = decide_revisions(
                self.store,
                self.llm,
                new=[(p.obs, p.atom) for p in fresh],
                scope=scope,
                current_date=current_date,
                revision_mode=str(cfg.revision_mode),
            )
            for plan, decision in zip(fresh, decisions):
                plan.decision = decision
        elif str(cfg.revision_mode) != "inline":
            # Same gate as decide_revisions: deferred / off never revise inline, whatever the LLM answered.
            # 与 decide_revisions 相同的门控：deferred / off 绝不在线修订，无论 LLM 回答了什么。
            for plan in fresh:
                plan.decision = (Decision.ADD, None, f"revision_mode={cfg.revision_mode}")
        # ---- Txn B
        # ---- 事务 B
        with self._txn_b() as created:
            for plan in plans:
                if plan.kind == "observation":
                    plan.obs = self.store.insert_observation(plan.obs)
            stored_eps: list[tuple[EP, _Plan]] = []
            for plan in plans:
                if plan.kind == "ep":
                    stored_eps.append((self.store.insert_ep(plan.ep), plan))
            for index, plan in enumerate(plans):
                if plan.kind != "observation":
                    continue
                self._apply_observation_plan(plan, plans, created, result)
                self.store.set_observation_atom(plan.obs.obs_id, plan.stored.atom_id)
                rel = relation_plans.get(index)
                if rel is not None and plan.duplicate is None and plan.batch_duplicate is None:
                    rel_atom = create_atom(
                        self.store,
                        self.index,
                        self.embedder,
                        rel.atom,
                        extra_mentions=plan.obs.entity_mentions,
                        vector=rel.vector,
                        history_event="ADD",
                        actor_id=plan.ev.actor_id if plan.ev else None,
                        role=str(plan.ev.role) if plan.ev and plan.ev.role is not None else None,
                        extractor=self.entity_extractor,
                    )
                    created.append(rel_atom.atom_id)
                    result.rows.append(_row(rel_atom, "ADD"))
            self._materialize_eps(stored_eps, previous_ep, scope=scope, anchors=anchors, created=created, result=result)

    def _apply_observation_plan(
        self, plan: _Plan, plans: list[_Plan], created: list[str], result: IngestResult
    ) -> None:
        """Apply the dedup outcome or the revision decision of one observation atom inside Txn B.
        在事务 B 内应用一个观察原子的去重结果或修订决策。
        """
        anchor_id = plan.atom.anchor_ids[0]
        actor_id = plan.ev.actor_id if plan.ev else None
        role = str(plan.ev.role) if plan.ev is not None and plan.ev.role is not None else None
        target = plan.duplicate
        if target is None and plan.batch_duplicate is not None:
            target = plans[plan.batch_duplicate].stored
        if target is not None:
            plan.stored = self._none_row(target.atom_id, anchor_id, result)
            return
        decision, target_id, reason = plan.decision or (Decision.ADD, None, "additive")
        old = self.store.get_atom(target_id) if target_id else None
        if decision != Decision.ADD and (old is None or old.archived_reason is not None):
            logger.warning("revision target %r is no longer valid; downgrading to ADD", target_id)
            decision, old = Decision.ADD, None
        if decision != Decision.ADD and old is not None:
            decision, old = self._guard_verdict(plan, decision, old, result)
        if decision == Decision.NONE and old is not None:
            plan.stored = self._none_row(old.atom_id, anchor_id, result)
            return
        if decision == Decision.REVISE and old is not None:
            at = max(plan.atom.valid_from or "", old.valid_from or "") or self.store.now()
            stored = revise_atom(
                self.store,
                self.index,
                self.embedder,
                old=old,
                new=plan.atom,
                witness_anchor_ids=list(plan.atom.anchor_ids),
                at=at,
                label=reason,
                extra_mentions=plan.obs.entity_mentions,
                vector=plan.vector,
                extractor=self.entity_extractor,
            )
            created.append(stored.atom_id)
            plan.stored = stored
            result.rows.append(_row(stored, "UPDATE", revises=old.atom_id))
            return
        if decision == Decision.CONFLICT and old is not None:
            stored = conflict_atom(
                self.store,
                self.index,
                self.embedder,
                old=old,
                new=plan.atom,
                witness_anchor_ids=list(plan.atom.anchor_ids),
                at=plan.atom.valid_from or self.store.now(),
                label=reason,
                extra_mentions=plan.obs.entity_mentions,
                vector=plan.vector,
                extractor=self.entity_extractor,
            )
            created.append(stored.atom_id)
            plan.stored = stored
            result.rows.append(_row(stored, "ADD", conflicts_with=[old.memory_id]))
            return
        stored = create_atom(
            self.store,
            self.index,
            self.embedder,
            plan.atom,
            extra_mentions=plan.obs.entity_mentions,
            vector=plan.vector,
            history_event="ADD",
            actor_id=actor_id,
            role=role,
            extractor=self.entity_extractor,
        )
        created.append(stored.atom_id)
        plan.stored = stored
        result.rows.append(_row(stored, "ADD"))

    def _guard_verdict(
        self, plan: _Plan, decision: Decision, old: MemoryAtom, result: IngestResult
    ) -> tuple[Decision, MemoryAtom | None]:
        """Honour a non-ADD verdict only when its pairing holds up. Every NONE / REVISE / CONFLICT asserts a relation
        between two statements about ONE subject, so when both atoms carry a subject and the subjects differ the
        verdict becomes ADD. With the subjects agreeing: a REVISE corroborated structurally passes regardless — the
        observation itself names the target in `supersedes_hint_ids` (the target is cited twice, independently of
        the decisions block) or the two atoms share subject and predicate (the exclusive-attribute case of the
        decision rules: `lives_in` Paris → Berlin has little text in common); a CONFLICT passes (it is by nature a
        different value about the same subject, so its texts are expected to differ); a NONE or an uncorroborated
        REVISE passes only when `verdict_similarity` — containment of the stemmed content tokens after the head
        (subject, shared predicate) is removed — is at or above `ingest.verdict_min_similarity` AND the two share at
        least two tokens or the shorter statement is contained entirely (one shared token out of two — `today`,
        `song`, `need` — is a coincidence, not a restatement, unless it is all the shorter statement has). Otherwise
        the verdict becomes ADD with a DEBUG line and one `result.downgraded` entry (similarity, `subjects_differ`,
        `shared`); threshold 0.0 disables the guard entirely. Same guard for both extraction outputs and both
        extraction modes (the verdict source does not matter, the pairing does).
        只有配对站得住时才采纳非 ADD 裁决。每个 NONE / REVISE / CONFLICT 都断言两条关于同一主语的陈述之间的关系，
        因此两原子都带主语且主语不同时裁决改为 ADD。主语一致时：结构上被佐证的 REVISE 无条件通过——观察自身在
        `supersedes_hint_ids` 里点名了目标（目标在决策块之外被独立引用了第二次），或两原子主语与谓词相同（决策规则
        里的互斥属性情形：`lives_in` 巴黎 → 柏林文本几乎不重合）；CONFLICT 通过（它本就是同一主语的不同取值，文本
        本该不同）；NONE 或未被佐证的 REVISE 只在 `verdict_similarity`——去掉头部（主语、共同谓词）后词干化内容词元的
        包含度——不低于 `ingest.verdict_min_similarity`、且双方至少共享两个词元或较短陈述被整体包含时通过（两个
        词元里共享一个——`today`、`song`、`need`——是巧合而非复述，除非那就是较短陈述的全部）。否则裁决改为 ADD，
        记一条 DEBUG 日志与一条 `result.downgraded`（相似度、`subjects_differ`、`shared`）；门槛 0.0 整体关闭守卫。
        两种抽取输出、两种抽取模式共用此守卫（裁决来源无关，配对是否成立才有关）。

        Input / 输入：
            plan: The observation plan (atom text / subject / predicate, the observation's resolved hints).
            plan：观察计划（原子文本 / 主语 / 谓词、观察解析后的提示）。

            decision: The verdict to check (never ADD).
            decision：待检查的裁决（绝不是 ADD）。

            old: The valid target atom.
            old：有效的目标原子。

            result: Receives the `downgraded` entry.
            result：接收 `downgraded` 条目。

        Output / 输出：
            (decision, old) unchanged, or (Decision.ADD, None).
            原样的 (decision, old)，或 (Decision.ADD, None)。

        State effect / 状态影响：
            Appends to result.downgraded; logs at DEBUG.
            向 result.downgraded 追加；DEBUG 日志。
        """
        threshold = float(self.config.ingest.verdict_min_similarity)
        if threshold <= 0.0:
            return decision, old
        similarity, shared = verdict_similarity(
            plan.atom.text, old.text,
            new_subject=plan.atom.subject, old_subject=old.subject,
            new_predicate=plan.atom.predicate, old_predicate=old.predicate,
            with_shared=True,
        )  # fmt: skip
        new_subject, old_subject = ((v or "").strip().lower() for v in (plan.atom.subject, old.subject))
        subjects_differ = bool(new_subject and old_subject and new_subject != old_subject)
        why = ""
        if decision == Decision.REVISE and (
            (plan.obs is not None and old.atom_id in plan.obs.supersedes_hint_ids)
            or _same_statement_head(plan.atom, old)
        ):
            keep = True
        elif subjects_differ:
            keep, why = False, ", subjects differ"
        elif decision == Decision.CONFLICT:
            keep = True
        elif similarity < threshold:
            keep = False
        else:
            keep = shared >= 2 or similarity == 1.0
            why = "" if keep else ", one shared token"
        if keep:
            return decision, old
        logger.debug(
            "verdict %s on %r downgraded to ADD: similarity %.3f (bar %.3f%s) against target %s %r",
            decision.value, plan.atom.text, similarity, threshold, why, old.atom_id, old.text,
        )  # fmt: skip
        result.downgraded.append(
            {
                "kind": "verdict",
                "verdict": decision.value,
                "target_id": old.atom_id,
                "similarity": round(similarity, 4),
                "shared": shared,
                "subjects_differ": subjects_differ,
                "text": plan.atom.text,
            }
        )
        return Decision.ADD, None

    def _materialize_eps(
        self,
        stored_eps: list[tuple[EP, _Plan]],
        previous_ep: EP | None,
        *,
        scope: Scope,
        anchors: dict[str, Anchor],
        created: list[str],
        result: IngestResult,
    ) -> None:
        """Create the EP atoms, back-pointers, inline temporal/use edges and ep transitions inside Txn B.
        在事务 B 内创建 EP 原子、回指、内联 temporal/use 边与 ep 转移。
        """
        window = float(self.config.graph.temporal_window_hours)
        prev = previous_ep if previous_ep is not None and previous_ep.scope.run_id == scope.run_id else None
        prev_node = prev.atom_id if prev is not None else None
        for stored_ep, plan in stored_eps:
            stored = create_atom(
                self.store,
                self.index,
                self.embedder,
                plan.atom,
                extra_mentions=stored_ep.entity_mentions,
                vector=plan.vector,
                history_event="ADD",
                extractor=self.entity_extractor,
            )
            created.append(stored.atom_id)
            plan.stored = stored
            self.store.set_ep_atom(stored_ep.ep_id, stored.atom_id)
            result.ep_ids.append(stored_ep.ep_id)
            result.rows.append(_row(stored, "ADD"))
            primary = stored_ep.anchor_ids[0]
            if prev is not None and prev_node and age_days(prev.time_anchor, stored_ep.time_anchor) * 24.0 <= window:
                edge_id = self.store.insert_edge(
                    {
                        "src_id": prev_node,
                        "dst_id": stored.atom_id,
                        "edge_type": "temporal",
                        "witness_anchor_ids": [primary],
                        "created_by": "ingest",
                        "proposer": "timestamp",
                        "confidence": 0.5,
                        "scope": scope,
                    }
                )
                if edge_id not in result.edge_ids:
                    result.edge_ids.append(edge_id)
            for used_id in stored_ep.uses_ep_ids:
                used = self.store.get_ep(used_id)
                if used is None or not used.atom_id or used.atom_id == stored.atom_id:
                    continue
                edge_id = self.store.insert_edge(
                    {
                        "src_id": stored.atom_id,
                        "dst_id": used.atom_id,
                        "edge_type": "use",
                        "witness_anchor_ids": [primary],
                        "created_by": "ingest",
                        "proposer": "rule",
                        "confidence": 0.7,
                        "label": f"use: {stored_ep.slots.a[:40]} → {used.slots.a[:40]}",
                        "scope": scope,
                    }
                )
                if edge_id not in result.edge_ids:
                    result.edge_ids.append(edge_id)
            anchor = self._anchor(primary, anchors)
            permission = str(anchor.permission_scope) if anchor is not None else "private"
            self.store.upsert_transition(Transition.from_ep(stored_ep, scope=scope, permission_scope=permission))
            prev, prev_node = stored_ep, stored.atom_id

    # ------------------------------------------------------------------ structured path
    # ------------------------------------------------------------------ 结构化路径
    def _ingest_structured(
        self,
        anchored: list[tuple[SourceEvent, Anchor, bool]],
        *,
        scope: Scope,
        result: IngestResult,
    ) -> None:
        """Structured path: event atoms for every event, closure EPs, derived constraints, closure edges, ep transitions.
        结构化路径：每个事件一个事件原子、闭包 EP、派生约束、闭包边与 ep 转移。
        """
        cfg = self.config
        anchors: dict[str, Anchor] = {anc.anchor_id: anc for _ev, anc, _d in anchored}
        background = self._constraints(scope)
        closure_input = [
            (ev, anc, dedup)
            for ev, anc, dedup in anchored
            if not (dedup and self.store.atoms_for_anchor(anc.anchor_id))
        ]
        batch = close_structured_events(
            self.store,
            anchored=closure_input,
            scope=scope,
            background=background,
            max_gap_hours=float(cfg.ingest.closure_max_gap_hours),
            temporal_window_hours=float(cfg.graph.temporal_window_hours),
            extractor=self.entity_extractor,
        )
        eps = [ep.model_copy(update={"ep_id": self.store.new_id("ep")}) for ep in batch.eps]
        derived = [obs.model_copy(update={"obs_id": self.store.new_id("obs")}) for obs in batch.derived]
        event_plans = [
            _Plan(kind="event", atom=atom_from_event(ev, anc, scope=scope), ev=ev, anc=anc) for ev, anc, _d in anchored
        ]
        ep_plans = [_Plan(kind="ep", atom=atom_from_ep(ep, scope=scope), ep=ep) for ep in eps]
        derived_plans = [
            _Plan(kind="derived", atom=atom_from_observation(obs, scope=scope, time_anchor=obs.valid_from), obs=obs)
            for obs in derived
        ]
        self._mark_duplicates(event_plans, scope)
        self._mark_duplicates(derived_plans, scope)
        self._embed_plans(event_plans + ep_plans + derived_plans)
        with self._txn_b() as created:
            for plan in event_plans:
                target = plan.duplicate
                if target is None and plan.batch_duplicate is not None:
                    target = event_plans[plan.batch_duplicate].stored
                if target is not None:
                    plan.stored = self._none_row(target.atom_id, plan.anc.anchor_id, result)
                    continue
                stored = create_atom(
                    self.store,
                    self.index,
                    self.embedder,
                    plan.atom,
                    vector=plan.vector,
                    history_event="ADD",
                    actor_id=plan.ev.actor_id,
                    role=str(plan.ev.role) if plan.ev.role is not None else Role.tool.value,
                    extractor=self.entity_extractor,
                )
                created.append(stored.atom_id)
                plan.stored = stored
                result.rows.append(_row(stored, "ADD"))
            stored_eps: list[EP] = []
            for plan in ep_plans:
                stored_ep = self.store.insert_ep(plan.ep)
                stored = create_atom(
                    self.store,
                    self.index,
                    self.embedder,
                    plan.atom,
                    extra_mentions=stored_ep.entity_mentions,
                    vector=plan.vector,
                    history_event="ADD",
                    extractor=self.entity_extractor,
                )
                created.append(stored.atom_id)
                plan.stored = stored
                self.store.set_ep_atom(stored_ep.ep_id, stored.atom_id)
                stored_eps.append(stored_ep)
                result.ep_ids.append(stored_ep.ep_id)
                result.rows.append(_row(stored, "ADD"))
            for plan in derived_plans:
                obs = self.store.insert_observation(plan.obs)
                plan.obs = obs
                target = plan.duplicate
                if target is None and plan.batch_duplicate is not None:
                    target = derived_plans[plan.batch_duplicate].stored
                if target is not None:
                    plan.stored = self._none_row(target.atom_id, obs.anchor_ids[0], result)
                    for extra in obs.anchor_ids[1:]:
                        self.store.add_atom_anchor(target.atom_id, extra, role="support")
                else:
                    stored = create_atom(
                        self.store,
                        self.index,
                        self.embedder,
                        plan.atom,
                        extra_mentions=obs.entity_mentions,
                        vector=plan.vector,
                        history_event="ADD",
                        extractor=self.entity_extractor,
                    )
                    created.append(stored.atom_id)
                    plan.stored = stored
                    result.rows.append(_row(stored, "ADD"))
                self.store.set_observation_atom(obs.obs_id, plan.stored.atom_id)
                if plan.stored.atom_id not in result.constraint_ids:
                    result.constraint_ids.append(plan.stored.atom_id)
            for spec in batch.edges:
                edge_id = self._write_edge_spec(spec, ep_plans, derived_plans, scope=scope)
                if edge_id is not None and edge_id not in result.edge_ids:
                    result.edge_ids.append(edge_id)
            for stored_ep in stored_eps:
                anchor = self._anchor(stored_ep.anchor_ids[0], anchors)
                permission = str(anchor.permission_scope) if anchor is not None else "private"
                self.store.upsert_transition(Transition.from_ep(stored_ep, scope=scope, permission_scope=permission))

    def _write_edge_spec(
        self, spec: EdgeSpec, ep_plans: list[_Plan], derived_plans: list[_Plan], *, scope: Scope
    ) -> str | None:
        """Resolve an EdgeSpec's endpoint refs to atom ids and insert the edge (created_by ingest, proposer closure).
        将 EdgeSpec 的端点引用解析为原子 id 并插入该边（created_by ingest，proposer closure）。
        """

        def resolve(ref: tuple[str, object]) -> str | None:
            kind, value = ref
            if kind == "ep":
                plan = ep_plans[int(value)]
                return plan.stored.atom_id if plan.stored else None
            if kind == "derived":
                plan = derived_plans[int(value)]
                return plan.stored.atom_id if plan.stored else None
            return str(value) if value else None

        src, dst = resolve(spec.src), resolve(spec.dst)
        if not src or not dst or src == dst:
            logger.warning("closure edge %s skipped: unresolved endpoints %r → %r", spec.edge_type, spec.src, spec.dst)
            return None
        return self.store.insert_edge(
            {
                "src_id": src,
                "dst_id": dst,
                "edge_type": spec.edge_type,
                "label": spec.label,
                "witness_anchor_ids": list(spec.witness_anchor_ids),
                "created_by": "ingest",
                "proposer": "closure",
                "confidence": float(spec.confidence),
                "scope": scope,
            }
        )


__all__ = ["EXTRACTION_MODES", "IngestResult", "Ingestor"]
