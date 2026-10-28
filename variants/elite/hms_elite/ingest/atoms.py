"""Atom builders and the single materialization primitive (create / revise / conflict) shared by every plane.
原子构造器与各平面共享的唯一物化原语（create / revise / conflict）。
"""

# Purpose (EN): Turn observations, EPs, events, messages and entities into MemoryAtoms and write them with axes, vectors, nodes and history.
# 目的（中）：把观察、EP、事件、消息与实体转为 MemoryAtom，并连同轴链接、向量、节点与历史一起写入。

# Input (EN): Observation / EP / SourceEvent+Anchor / entity rows, a StateStore, an index and an embedder.
# 输入（中）：Observation / EP / SourceEvent+Anchor / 实体行，以及 StateStore、索引与嵌入器。

# Output (EN): Stored MemoryAtoms (hydrated) and revision / conflict edges.
# 输出（中）：已存储（水合）的 MemoryAtom 以及 revision / conflict 边。

# State effect (EN): create_atom / revise_atom / conflict_atom write memory_atoms, atom_anchor, entities, atom_entity, atom_topic, atom_vectors, graph_nodes, graph_edges, history.
# 状态影响（中）：create_atom / revise_atom / conflict_atom 写入 memory_atoms、atom_anchor、entities、atom_entity、atom_topic、atom_vectors、graph_nodes、graph_edges、history。

from __future__ import annotations

from typing import Any, Sequence

from hms_elite.embeddings.base import EmbedderBase
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import (
    AtomType,
    EdgeType,
    HistoryEvent,
    ObservationType,
    Reward,
    Role,
    Scale,
    Sentiment,
    SourceKind,
)
from hms_elite.index.base import IndexBase, atom_index_attrs
from hms_elite.ingest.closure import event_atom_text
from hms_elite.memory.storage import EntityKey, StateStore
from hms_elite.models import EP, Anchor, GraphEdge, MemoryAtom, Observation, Scope, SourceEvent
from hms_elite.utils.text import (
    EntityCandidate,
    entity_key,
    extract_entities,
    normalize_entity,
    ordered_union,
    topic_tags,
)

_MIDNIGHT = "T00:00:00.000000+00:00"
_SENTIMENT_OF_REWARD = {
    Reward.negative.value: Sentiment.negative.value,
    Reward.positive.value: Sentiment.positive.value,
}


def _observation_kind(obs_type: str) -> tuple[str, str]:
    """(atom_type, scale_level) of an observation type.
    观察类型对应的 (atom_type, scale_level)。
    """
    if str(obs_type) == ObservationType.temporal_fact.value:
        return AtomType.event.value, Scale.event.value
    return AtomType.state.value, Scale.state.value


def atom_from_observation(
    obs: Observation, *, scope: Scope, time_anchor: str | None = None, actor_id: str | None = None
) -> MemoryAtom:
    """Build the state / event atom of an observation (§4.2 observation shape).
    构造观察对应的 state / event 原子（§4.2 观察形状）。

    Input / 输入：
        obs: The observation (obs_id already assigned when known).
        obs：观察（已知时 obs_id 已分配）。

        scope: Scope of the atom.
        scope：原子的范围。

        time_anchor: The witnessing event's timestamp; temporal facts use date_observed midnight instead.
        time_anchor：见证事件的时间戳；temporal_fact 改用 date_observed 的零点。

        actor_id: The witnessing event's actor_id (the message `name`); None → the observation's `speaker` is used.
        actor_id：见证事件的 actor_id（消息的 `name`）；None → 使用观察的 `speaker`。

    Output / 输出：
        MemoryAtom (state/state, or event/event for temporal facts) with subject/predicate/obs_type promoted and
        actor_id = event actor_id or obs.speaker.
        MemoryAtom（state/state，temporal_fact 为 event/event），提升 subject/predicate/obs_type，
        actor_id = 事件 actor_id 或 obs.speaker。

    State effect / 状态影响：
        None.
        无。
    """
    atom_type, scale = _observation_kind(obs.type)
    if str(obs.type) == ObservationType.temporal_fact.value:
        anchor_time = obs.date_observed + _MIDNIGHT
    else:
        anchor_time = time_anchor or obs.valid_from or (obs.date_observed + _MIDNIGHT)
    content: dict[str, Any] = {
        "text": obs.text,
        "obs_id": obs.obs_id,
        "obs_type": str(obs.type),
        "subject": obs.subject,
        "predicate": obs.predicate,
        "objects": list(obs.objects),
        "relation": str(obs.relation),
        "date_observed": obs.date_observed,
    }
    if str(obs.type) == ObservationType.temporal_fact.value:
        content["event_time"] = anchor_time
    return MemoryAtom(
        atom_type=atom_type,
        content=content,
        text=obs.text,
        time_anchor=anchor_time,
        valid_from=obs.valid_from or anchor_time,
        valid_to=obs.valid_to,
        entity_ids=[entity_key(m) for m in obs.entity_mentions if m.strip()],
        topic_tags=list(obs.topic_tags),
        scale_level=scale,
        role=Role.extracted,
        sentiment=str(obs.sentiment),
        subject=obs.subject,
        predicate=obs.predicate,
        obs_type=str(obs.type),
        source_kind=SourceKind.observation,
        source_id=obs.obs_id,
        confidence=obs.confidence,
        extracted_by=obs.extracted_by,
        visibility=obs.visibility,
        scope=scope,
        actor_id=_observation_actor(obs, actor_id),
        attributed_to=obs.attributed_to,
        anchor_ids=list(obs.anchor_ids),
    )


def _observation_actor(obs: Observation, actor_id: str | None) -> str | None:
    """actor_id of an observation atom: the witnessing event's actor_id when present, else the LLM-inferred speaker.
    观察原子的 actor_id：见证事件带 actor_id 时用它，否则用 LLM 推断的说话人。
    """
    # The event's actor_id is part of the witnessed payload hash (event_payload_hash) and comes from the caller, so
    # it is the authoritative identity; `speaker` is an extraction inference and only fills the gap.
    # 事件的 actor_id 属于被见证的载荷哈希（event_payload_hash）且来自调用方，因此是权威身份；
    # `speaker` 是抽取推断，只在缺口处补位。
    if actor_id:
        return actor_id
    return obs.speaker or None


def relation_atom_from_observation(
    obs: Observation, *, scope: Scope, time_anchor: str | None = None, actor_id: str | None = None
) -> MemoryAtom:
    """Build the second, relation-scale atom of a comparison observation (§4.2 relation row).
    构造比较类观察的第二个 relation 尺度原子（§4.2 relation 行）。

    Input / 输入：
        obs: An observation with relation 'comparison'.
        obs：relation 为 'comparison' 的观察。

        scope: Scope of the atom.
        scope：原子的范围。

        time_anchor: The witnessing event's timestamp.
        time_anchor：见证事件的时间戳。

        actor_id: The witnessing event's actor_id; None → obs.speaker (same rule as atom_from_observation).
        actor_id：见证事件的 actor_id；None → obs.speaker（与 atom_from_observation 同一规则）。

    Output / 输出：
        MemoryAtom(atom_type relation, scale relation, text 'Comparison: <text>', source_kind observation, same obs_id).
        MemoryAtom（atom_type relation、scale relation、文本 'Comparison: <text>'、source_kind observation、同一 obs_id）。

    State effect / 状态影响：
        None.
        无。
    """
    text = f"Comparison: {obs.text}"
    anchor_time = time_anchor or obs.valid_from or (obs.date_observed + _MIDNIGHT)
    return MemoryAtom(
        atom_type=AtomType.relation,
        content={
            "text": text,
            "subject": obs.subject,
            "predicate": obs.predicate,
            "objects": list(obs.objects),
            "relation": str(obs.relation),
        },
        text=text,
        time_anchor=anchor_time,
        valid_from=obs.valid_from or anchor_time,
        valid_to=obs.valid_to,
        entity_ids=[entity_key(m) for m in obs.entity_mentions if m.strip()],
        topic_tags=list(obs.topic_tags),
        scale_level=Scale.relation,
        role=Role.extracted,
        sentiment=str(obs.sentiment),
        subject=obs.subject,
        predicate=obs.predicate,
        obs_type=str(obs.type),
        source_kind=SourceKind.observation,
        source_id=obs.obs_id,
        confidence=obs.confidence,
        extracted_by=obs.extracted_by,
        visibility=obs.visibility,
        scope=scope,
        actor_id=_observation_actor(obs, actor_id),
        attributed_to=obs.attributed_to,
        anchor_ids=list(obs.anchor_ids),
    )


def atom_from_ep(ep: EP, *, scope: Scope) -> MemoryAtom:
    """Build the L1 episode atom of an EP (text = slots gloss).
    构造 EP 的 L1 情节原子（文本为槽位 gloss）。

    Input / 输入：
        ep: The EP (ep_id already assigned).
        ep：EP（ep_id 已分配）。

        scope: Scope of the atom.
        scope：原子的范围。

    Output / 输出：
        MemoryAtom(episode / L1_micro, content {'text', 'ep_id', 'slots'}, source_kind ep, source_ep_ids [ep_id]).
        MemoryAtom（episode / L1_micro，content 为 {'text', 'ep_id', 'slots'}，source_kind ep，source_ep_ids [ep_id]）。

    State effect / 状态影响：
        None.
        无。
    """
    text = ep.slots.gloss()
    return MemoryAtom(
        atom_type=AtomType.episode,
        content={"text": text, "ep_id": ep.ep_id, "slots": ep.slots.model_dump()},
        text=text,
        time_anchor=ep.time_anchor,
        valid_from=ep.valid_from or ep.time_anchor,
        valid_to=ep.valid_to,
        entity_ids=[entity_key(m) for m in ep.entity_mentions if m.strip()],
        topic_tags=list(ep.topic_tags),
        scale_level=Scale.L1_micro,
        role=Role.extracted,
        sentiment=_SENTIMENT_OF_REWARD.get(str(ep.slots.r), Sentiment.neutral.value),
        source_kind=SourceKind.ep,
        source_id=ep.ep_id,
        source_ep_ids=[ep.ep_id],
        confidence=ep.confidence,
        extracted_by=ep.extracted_by,
        visibility=ep.visibility,
        scope=scope,
        anchor_ids=list(ep.anchor_ids),
    )


def atom_from_event(ev: SourceEvent, anchor: Anchor, *, scope: Scope) -> MemoryAtom:
    """Build the raw event atom of a structured event.
    构造结构化事件的原始事件原子。

    Input / 输入：
        ev: The structured event.
        ev：结构化事件。

        anchor: Its anchor.
        anchor：其锚点。

        scope: Scope of the atom.
        scope：原子的范围。

    Output / 输出：
        MemoryAtom(event / event, role tool, text event_atom_text(ev), content {'text', 'event_id', 'object_type', 'payload'}, extracted_by 'raw').
        MemoryAtom（event / event、角色 tool、文本 event_atom_text(ev)、content {'text', 'event_id', 'object_type', 'payload'}、extracted_by 'raw'）。

    State effect / 状态影响：
        None.
        无。
    """
    text = event_atom_text(ev)
    return MemoryAtom(
        atom_type=AtomType.event,
        content={
            "text": text,
            "event_id": ev.event_id,
            "object_type": str(ev.object_type),
            "payload": dict(ev.payload),
        },
        text=text,
        time_anchor=ev.timestamp,
        valid_from=ev.timestamp,
        scale_level=Scale.event,
        role=Role.tool,
        source_kind=SourceKind.event,
        source_id=ev.event_id,
        extracted_by="raw",
        visibility=ev.visibility,
        scope=scope,
        actor_id=ev.actor_id,
        anchor_ids=[anchor.anchor_id],
    )


def atom_from_message(ev: SourceEvent, anchor: Anchor, *, scope: Scope) -> MemoryAtom:
    """Build the raw message atom of a chat message (infer=False path).
    构造聊天消息的原始消息原子（infer=False 路径）。

    Input / 输入：
        ev: The message event.
        ev：消息事件。

        anchor: Its anchor.
        anchor：其锚点。

        scope: Scope of the atom.
        scope：原子的范围。

    Output / 输出：
        MemoryAtom(message / L1_micro, role = message role, content {'text', 'role', 'actor_id', 'dropped_parts'}, extracted_by 'raw').
        MemoryAtom（message / L1_micro、角色为消息角色、content {'text', 'role', 'actor_id', 'dropped_parts'}、extracted_by 'raw'）。

    State effect / 状态影响：
        None.
        无。
    """
    payload = ev.payload or {}
    text = str(payload.get("text") or "")
    role = str(ev.role) if ev.role is not None else Role.user.value
    return MemoryAtom(
        atom_type=AtomType.message,
        content={
            "text": text,
            "role": role,
            "actor_id": ev.actor_id,
            "dropped_parts": int(payload.get("dropped_parts") or 0),
        },
        text=text,
        time_anchor=ev.timestamp,
        valid_from=ev.timestamp,
        scale_level=Scale.L1_micro,
        role=role,
        source_kind=SourceKind.message,
        source_id=ev.event_id,
        extracted_by="raw",
        visibility=ev.visibility,
        scope=scope,
        actor_id=ev.actor_id,
        anchor_ids=[anchor.anchor_id],
    )


def _candidate_key(candidate: EntityCandidate) -> EntityKey:
    """EntityKey tuple of a candidate: (key, canonical, type, aliases).
    候选实体的 EntityKey 元组：(key, canonical, type, aliases)。
    """
    aliases: tuple[str, ...] = (candidate.normalized,) if candidate.normalized != candidate.key else ()
    return (candidate.key, candidate.text, candidate.entity_type, aliases)


def link_atom_axes(
    atom: MemoryAtom, *, extra_mentions: Sequence[str] = (), extractor: EntityExtractorBase | None = None
) -> tuple[list[EntityKey], list[str]]:
    """Compute the entity keys and topic tags an atom links to (text entities ∪ re-classified extra mentions).
    计算原子链接的实体键与主题标签（文本实体 ∪ 重新分类的额外提及）。

    Input / 输入：
        atom: The atom (text and topic_tags are read).
        atom：原子（读取 text 与 topic_tags）。

        extra_mentions: Mention texts (observation/EP entity_mentions, entity canonical text) re-classified by the extractor.
        extra_mentions：由抽取器重新分类的提及文本（观察/EP 的 entity_mentions、实体 canonical 文本）。

        extractor: Entity extractor; None means utils.text.extract_entities (the rule provider, byte-identical).
        extractor：实体抽取器；None 表示 utils.text.extract_entities（规则提供者，逐字节一致）。

    Output / 输出：
        (entity keys deduplicated by key with the first kept, ordered_union(atom.topic_tags, topic_tags(text))).
        （按 key 去重且保留首个的实体键列表，ordered_union(atom.topic_tags, topic_tags(text))）。

    State effect / 状态影响：
        None.
        无。
    """
    extract = extractor.extract if extractor is not None else extract_entities
    keys: list[EntityKey] = []
    seen: set[str] = set()
    candidates: list[EntityCandidate] = list(extract(atom.text))
    for mention in extra_mentions:
        mention = str(mention or "").strip()
        if not mention:
            continue
        found = extract(mention)
        if not found:
            found = [EntityCandidate("PROPER", mention, normalize_entity(mention), entity_key(mention))]
        candidates.extend(found)
    for candidate in candidates:
        if candidate.key in seen or not candidate.key:
            continue
        seen.add(candidate.key)
        keys.append(_candidate_key(candidate))
    topics = ordered_union(list(atom.topic_tags), topic_tags(atom.text))
    return keys, topics


def atom_from_entity(entity: dict, *, scope: Scope, anchor_ids: list[str], mentions: int) -> MemoryAtom:
    """Build the entity-scale atom of an entities row (materialized by consolidate step 0).
    构造实体行对应的 entity 尺度原子（由 consolidate 第 0 步物化）。

    Input / 输入：
        entity: `{'entity_id', 'normalized_text', 'canonical_text', 'entity_type', 'aliases'}`.
        entity：`{'entity_id', 'normalized_text', 'canonical_text', 'entity_type', 'aliases'}`。

        scope: The entity identity scope (user_id/agent_id, run_id None).
        scope：实体身份范围（user_id/agent_id，run_id 为 None）。

        anchor_ids: Sorted union of the anchors of the linked atoms.
        anchor_ids：所链接原子锚点的排序并集。

        mentions: Number of linked valid atoms.
        mentions：所链接有效原子的数量。

    Output / 输出：
        MemoryAtom(entity / entity, text canonical_text, subject canonical_text, predicate 'is_entity', extracted_by 'rule@entity-v1').
        MemoryAtom（entity / entity、文本 canonical_text、主体 canonical_text、谓词 'is_entity'、extracted_by 'rule@entity-v1'）。

    State effect / 状态影响：
        None.
        无。
    """
    canonical = str(entity.get("canonical_text") or entity.get("normalized_text") or "")
    return MemoryAtom(
        atom_type=AtomType.entity,
        content={
            "text": canonical,
            "entity_id": entity.get("entity_id"),
            "entity_type": entity.get("entity_type"),
            "aliases": list(entity.get("aliases") or []),
            "mentions": int(mentions),
        },
        text=canonical,
        time_anchor=None,
        entity_ids=[str(entity.get("normalized_text") or entity_key(canonical))],
        topic_tags=[],
        scale_level=Scale.entity,
        role=Role.extracted,
        subject=canonical,
        predicate="is_entity",
        source_kind=SourceKind.entity,
        source_id=str(entity.get("entity_id") or ""),
        confidence=0.7,
        extracted_by="rule@entity-v1",
        scope=scope,
        anchor_ids=list(anchor_ids),
    )


def create_atom(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    atom: MemoryAtom,
    *,
    extra_mentions: Sequence[str] = (),
    vector: list[float] | None = None,
    history_event: str | None = "ADD",
    history_anchor_id: str | None = None,
    actor_id: str | None = None,
    role: str | None = None,
    history_details: dict | None = None,
    extractor: EntityExtractorBase | None = None,
) -> MemoryAtom:
    """Materialize one atom: axes, vector, graph node and (optionally) a history row, in one transaction.
    在一个事务内物化一个原子：轴链接、向量、图节点以及（可选的）历史行。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index.
        index：向量索引。

        embedder: Embedder used when `vector` is not given (and for the index identity).
        embedder：未给 `vector` 时用于计算向量（并提供索引身份）。

        atom: The atom to write (ids assigned by the store when empty).
        atom：待写入的原子（id 为空时由 store 分配）。

        extra_mentions: Extra entity mentions linked besides the text entities.
        extra_mentions：文本实体之外额外链接的实体提及。

        vector: Pre-computed embedding (computed unlocked by the caller whenever possible).
        vector：预先计算的向量（调用方尽量在无锁阶段计算）。

        history_event: History event name, or None for no history row (revise_atom uses None).
        history_event：历史事件名；None 表示不写历史行（revise_atom 使用 None）。

        history_anchor_id: Witness of the history row (defaults to the atom's primary anchor).
        history_anchor_id：历史行的见证锚点（默认原子的主锚点）。

        actor_id: actor_id recorded on the history row.
        actor_id：记录在历史行上的 actor_id。

        role: role recorded on the history row.
        role：记录在历史行上的角色。

        history_details: Extra details JSON of the history row.
        history_details：历史行的额外 details JSON。

        extractor: Entity extractor for link_atom_axes; None means the rule provider.
        extractor：link_atom_axes 使用的实体抽取器；None 表示规则提供者。

    Output / 输出：
        The stored, hydrated atom.
        已存储并水合的原子。

    State effect / 状态影响：
        Writes memory_atoms, atom_anchor, entities, atom_entity, atom_topic, atom_vectors, graph_nodes and history; an EmbeddingError rolls everything back.
        写入 memory_atoms、atom_anchor、entities、atom_entity、atom_topic、atom_vectors、graph_nodes 与 history；EmbeddingError 会回滚全部写入。
    """
    entity_keys, topics = link_atom_axes(atom, extra_mentions=extra_mentions, extractor=extractor)
    with store.transaction():
        stored = store.insert_atom(atom, entity_keys=entity_keys, topics=topics)
        vec = vector if vector is not None else embedder.embed(stored.text, "add")
        # The scalars ride along so a provider that keeps them (Milvus) needs no SELECT of its own per upsert.
        # 标量随向量一起传入，使保存标量的提供者（Milvus）每次 upsert 不必再自行 SELECT。
        index.upsert(stored.atom_id, list(vec), embedder=embedder.identity, attrs=atom_index_attrs(stored))
        if history_event is not None:
            store.add_history(
                memory_id=stored.memory_id,
                old_memory=None,
                new_memory=stored.text,
                event=str(history_event),
                created_at=stored.created_at,
                updated_at=stored.created_at,
                actor_id=actor_id,
                role=role,
                anchor_id=history_anchor_id or stored.anchor_ids[0],
                atom_id=stored.atom_id,
                details=history_details,
            )
        return stored


def revise_atom(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    *,
    old: MemoryAtom,
    new: MemoryAtom,
    witness_anchor_ids: list[str],
    at: str,
    reason: str = "revised",
    created_by: str = "ingest",
    history_event: str = "UPDATE",
    label: str = "",
    extra_mentions: Sequence[str] = (),
    vector: list[float] | None = None,
    edge_confidence: float = 0.9,
    extractor: EntityExtractorBase | None = None,
) -> MemoryAtom:
    """Archive `old`, insert `new` as the next version of the same lineage, add a revision edge and an UPDATE history row.
    归档 `old`，将 `new` 作为同一谱系的下一版本插入，添加 revision 边与 UPDATE 历史行。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index.
        index：向量索引。

        embedder: The embedder.
        embedder：嵌入器。

        old: The current head (valid) version.
        old：当前（有效）的头版本。

        new: The successor atom (memory_id / version are overwritten).
        new：后继原子（memory_id / version 会被覆盖）。

        witness_anchor_ids: Witnesses of the revision edge and the history row (primary first).
        witness_anchor_ids：revision 边与历史行的见证（主见证在前）。

        at: Archive instant: the new evidence's event time at ingest, store.now() for update()/feedback.
        at：归档时刻：摄取时为新证据的事件时间，update()/feedback 时为 store.now()。

        reason: archived_reason of the old version.
        reason：旧版本的 archived_reason。

        created_by: created_by of the revision edge.
        created_by：revision 边的 created_by。

        history_event: History event name (UPDATE by default).
        history_event：历史事件名（默认 UPDATE）。

        label: Label of the revision edge.
        label：revision 边的标签。

        extra_mentions: Extra entity mentions of the successor.
        extra_mentions：后继原子的额外实体提及。

        vector: Pre-computed embedding of the successor.
        vector：后继原子的预计算向量。

        edge_confidence: Confidence of the revision edge (0.9 at ingest; feedback handlers pass 0.95).
        edge_confidence：revision 边的置信度（摄取时 0.9；反馈处理器传 0.95）。

        extractor: Entity extractor for the successor's axes; None means the rule provider.
        extractor：后继原子轴链接使用的实体抽取器；None 表示规则提供者。

    Output / 输出：
        The stored successor (same memory_id, version + 1).
        已存储的后继原子（同一 memory_id，版本 + 1）。

    State effect / 状态影响：
        Archives the old atom (valid_to = at), writes the new version, one revision edge and one history row.
        归档旧原子（valid_to = at），写入新版本、一条 revision 边与一条历史行。
    """
    instant = at
    successor = new.model_copy(update={"memory_id": old.memory_id, "version": old.version + 1, "archived_reason": None})
    with store.transaction():
        store.archive_atom(old.atom_id, reason=reason, at=instant, valid_to=instant)
        stored = create_atom(
            store,
            index,
            embedder,
            successor,
            extra_mentions=extra_mentions,
            vector=vector,
            history_event=None,
            extractor=extractor,
        )
        if (str(stored.source_kind), stored.source_id) == (str(old.source_kind), old.source_id):
            # An update / feedback successor cites its predecessor's source row: the archive cascade above ended it,
            # the live head needs it open again (purge protects it through `source_shared`).
            # 更新 / 反馈的后继引用前驱的来源行：上面的归档级联结束了它，存活的头版本需要它重新打开
            # （清理通过 `source_shared` 保护它）。
            store.reopen_source(stored.atom_id)
        edge_id = store.insert_edge(
            GraphEdge(
                src_id=old.atom_id,
                dst_id=stored.atom_id,
                edge_type=EdgeType.revision,
                label=label,
                confidence=float(edge_confidence),
                witness_anchor_ids=list(witness_anchor_ids),
                created_by=created_by,
                proposer="rule",
                scope=stored.scope,
                created_at=store.now(),
            )
        )
        store.add_history(
            memory_id=stored.memory_id,
            old_memory=old.text,
            new_memory=stored.text,
            event=str(history_event),
            created_at=old.created_at or stored.created_at,
            updated_at=instant,
            anchor_id=witness_anchor_ids[0],
            atom_id=stored.atom_id,
            edge_id=edge_id,
            successor_id=stored.atom_id,
        )
        return stored


def conflict_atom(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    *,
    old: MemoryAtom,
    new: MemoryAtom,
    witness_anchor_ids: list[str],
    at: str,
    label: str,
    extra_mentions: Sequence[str] = (),
    vector: list[float] | None = None,
    extractor: EntityExtractorBase | None = None,
) -> MemoryAtom:
    """Create `new` as its own lineage, join it to `old` by a conflict edge and record CONFLICT on the old lineage.
    以新谱系创建 `new`，用 conflict 边连接 `old`，并在旧谱系记录 CONFLICT。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index.
        index：向量索引。

        embedder: The embedder.
        embedder：嵌入器。

        old: The existing atom the new statement contradicts.
        old：被新陈述所矛盾的已有原子。

        new: The new atom (new lineage).
        new：新原子（新谱系）。

        witness_anchor_ids: Witnesses of the conflict edge (the new atom's anchors).
        witness_anchor_ids：conflict 边的见证（新原子的锚点）。

        at: Instant recorded as updated_at on the CONFLICT history row.
        at：记录为 CONFLICT 历史行 updated_at 的时刻。

        label: Label of the conflict edge (the decision reason).
        label：conflict 边的标签（决策原因）。

        extra_mentions: Extra entity mentions of the new atom.
        extra_mentions：新原子的额外实体提及。

        vector: Pre-computed embedding of the new atom.
        vector：新原子的预计算向量。

        extractor: Entity extractor for the new atom's axes; None means the rule provider.
        extractor：新原子轴链接使用的实体抽取器；None 表示规则提供者。

    Output / 输出：
        The stored new atom.
        已存储的新原子。

    State effect / 状态影响：
        Writes the new atom (ADD history), one conflict edge and one CONFLICT history row on the old lineage.
        写入新原子（ADD 历史）、一条 conflict 边以及旧谱系上的一条 CONFLICT 历史行。
    """
    with store.transaction():
        stored = create_atom(
            store, index, embedder, new, extra_mentions=extra_mentions, vector=vector, extractor=extractor
        )
        edge_id = store.insert_edge(
            GraphEdge(
                src_id=old.atom_id,
                dst_id=stored.atom_id,
                edge_type=EdgeType.conflict,
                label=label,
                confidence=0.6,
                witness_anchor_ids=list(witness_anchor_ids),
                created_by="ingest",
                proposer="rule",
                scope=stored.scope,
                created_at=store.now(),
            )
        )
        store.add_history(
            memory_id=old.memory_id,
            old_memory=old.text,
            new_memory=stored.text,
            event=HistoryEvent.CONFLICT.value,
            created_at=old.created_at or stored.created_at,
            updated_at=at,
            anchor_id=witness_anchor_ids[0],
            atom_id=old.atom_id,
            edge_id=edge_id,
            details={"conflicts_with": stored.memory_id},
        )
        return stored


__all__ = [
    "EntityKey",
    "atom_from_entity",
    "atom_from_ep",
    "atom_from_event",
    "atom_from_message",
    "atom_from_observation",
    "conflict_atom",
    "create_atom",
    "link_atom_axes",
    "relation_atom_from_observation",
    "revise_atom",
]
