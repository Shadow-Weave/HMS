"""Compact extraction output contract: the LLM emits nine observation keys (eight fields plus an optional entities hint), ingest.derive fills the rest, the full contract is untouched.
紧凑抽取输出契约：LLM 只输出九个观察键（八个字段加可选的 entities 提示），其余由 ingest.derive 补齐，完整契约原样不动。
"""

# Purpose (EN): Pin IngestConfig.extraction_output='compact' end to end — a scripted compact reply through HMS.add,
# the derived fields on the atoms, the seven axes, the witness anchor, supersession hints through id_map, the speaker
# → actor_id mapping, the `entities` hint (union with the extractor's hits and the speaker, full-mode parity on the
# entity axis, malformed shapes ignored), the compact prompts' vocabulary, the rule LLM's compact replies, config
# validation — and pin the full contract as byte-identical (prompt hashes, golden observations).
# 目的（中）：端到端钉住 IngestConfig.extraction_output='compact'——脚本化紧凑回复经 HMS.add、原子上的推导字段、
# 七个轴、见证锚点、经 id_map 解析的取代提示、speaker → actor_id 映射、`entities` 提示（与抽取器命中及说话人的并集、
# 实体轴与完整模式等价、畸形形状被忽略）、紧凑提示词的词表、规则 LLM 的紧凑回复、配置校验——并钉住完整契约
# 逐字节不变（提示词哈希、黄金观察）。

# Input (EN): Hand-written compact items, the chat_* fixtures, ScriptedLLM / RuleLLM, the conftest clock / id helpers.
# 输入（中）：手写的紧凑条目、chat_* 夹具、ScriptedLLM / RuleLLM、conftest 的时钟 / id 辅助。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pytest

from hms_elite import HMS, HMSConfig
from hms_elite.configs import prompts
from hms_elite.configs.base import IngestConfig
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.ingest import derive, extraction
from hms_elite.ingest.events import anchor_events, events_from_messages
from hms_elite.ingest.pipeline import EXTRACTION_OUTPUTS, Ingestor
from hms_elite.llms.rule import RuleLLM, statement_triple
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.async_main import AsyncHMS
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import Anchor, Observation, Scope, SourceEvent
from hms_elite.utils import json as ujson
from tests.conftest import EVENTS_DIR, FrozenClock, SequentialIds, load_events

CHAT_FIXTURES = sorted(path.stem for path in EVENTS_DIR.glob("chat_*.json"))
EXTRACTION_LOGGER = "hms_elite.ingest.extraction"
DAY = "2026-06-15"
STAMP = "2026-06-15T10:04:00Z"
STAMP_ISO = "2026-06-15T10:04:00.000000+00:00"
SCOPE = Scope(user_id="alice", run_id="run-compact")
# sha256 of the full prompts before the compact contract was added: 'full' must stay byte-identical.
# 加入紧凑契约之前完整提示词的 sha256：'full' 必须逐字节不变。
FULL_PROMPT_SHA256 = {
    "MEMORY_EXTRACTION_PROMPT": "951ff8c977062e44a48aa029dd23055461d1ab7e37502cdaf7fb2340d106a526",
    "OBSERVATION_EXTRACTION_PROMPT": "e3baca1270fd7062633f7b7deba0be6918aa879609550d37bdd1971d9133742e",
}
# Observation fields the compact contract drops; none may appear in a compact observation item.
# 紧凑契约删掉的观察字段；紧凑观察条目里一个都不许出现。
DROPPED = (
    "subject", "predicate", "objects", "relation", "date_observed", "valid_from", "confidence", "attributed_to",
    "sentiment", "topic_tags", "entity_mentions",
)  # fmt: skip
COMPACT_REPLY = {
    "observations": [
        {"id": "0", "event_idx": 0, "text": "Katrina must not modify the billing state machine.",
         "type": "constraint", "speaker": "Katrina"},
        {"id": "1", "event_idx": 1, "text": "Assistant scheduled the dentist visit for Sarah.",
         "type": "temporal_fact", "speaker": None, "when": "2026-06-20T09:30:00+00:00"},
        {"id": "2", "event_idx": 2, "text": "User usually drinks black coffee after lunch.", "type": "habit"},
    ],
    "eps": [],
}  # fmt: skip
COMPACT_MESSAGES = [
    {"role": "user", "content": "Katrina said: do not touch the billing state machine."},
    {"role": "assistant", "content": "I scheduled the dentist visit for Sarah on June 20 at 9:30."},
    {"role": "user", "name": "Bob", "content": "I usually drink black coffee after lunch."},
]


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
def _facade(output: str, llm: dict | None = None, **ingest: Any) -> tuple[HMS, FrozenClock]:
    """An offline in-memory HMS (memory index, frozen clock, sequential ids) in one extraction output.
    某种抽取输出下的离线内存 HMS（memory 索引、冻结时钟、顺序 id）。
    """
    clock = FrozenClock()
    store = StateStore(":memory:", now=clock.now, id_factory=SequentialIds())
    config: dict[str, Any] = {
        "vector_store": {"provider": "memory"},
        "store": {"path": ":memory:"},
        "ingest": {"extraction_output": output, **ingest},
    }
    if llm is not None:
        config["llm"] = llm
    return HMS(HMSConfig(**config), store=store), clock


def _events(store: StateStore, messages: list[dict]) -> list[tuple[SourceEvent, Anchor]]:
    """Anchor chat messages in the store and return the (event, anchor) pairs the extractor sees.
    把聊天消息落锚到 store，返回抽取器看到的 (事件, 锚点) 对。
    """
    events = events_from_messages(
        coerce_messages(messages), scope=SCOPE, event_time=STAMP, user_meta={}, visibility="private"
    )
    return [(ev, anc) for ev, anc, _dedup in anchor_events(store, events)]


def _fresh() -> StateStore:
    """A fresh in-memory store with a frozen clock and sequential ids.
    带冻结时钟与顺序 id 的全新内存 store。
    """
    return StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds())


def _compact(
    item: dict, *, messages: list[dict] | None = None, id_map: dict[str, str] | None = None
) -> Observation | None:
    """Run one compact item through the compact coercion against anchored user messages.
    让一个紧凑条目经紧凑规整处理，见证为已落锚的 user 消息。
    """
    store = _fresh()
    events = _events(store, messages or [{"role": "user", "content": "hello there"}])
    return extraction._compact_observation_from_item(
        RuleLLM(), item, events=events, observation_date="2026-01-01", scope=SCOPE, id_map=id_map or {},
        extractor=None, what="memory_extraction_compact.observations",
    )  # fmt: skip


def _observation_items(prompt: str) -> list[dict]:
    """The observation items of a prompt's OUTPUT FORMAT block.
    提示词 OUTPUT FORMAT 块中的观察条目。
    """
    return json.loads(ujson.extract_json(prompts.output_format_section(prompt)))["observations"]


# Tables whose rows must be identical between a full and a compact facade fed the same fixture (§6.5 invariants).
# 同一夹具下完整门面与紧凑门面必须逐行相同的表（§6.5 不变量）。
IDENTICAL_TABLES = ("atom_anchor", "atom_entity", "atom_topic", "entities", "history", "graph_nodes", "graph_edges", "edge_witnesses")


def _tables(hms: HMS) -> dict[str, list]:
    """Every row of IDENTICAL_TABLES plus the (text, subject, predicate, objects, relation) triple of every atom.
    IDENTICAL_TABLES 的全部行，加上每个原子的 (text, subject, predicate, objects, relation) 五元组。
    """
    conn = hms.store._conn
    out: dict[str, list] = {}
    for table in IDENTICAL_TABLES:
        cur = conn.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
    out["atom_triples"] = [
        (text, subject, predicate, json.loads(content).get("objects"), json.loads(content).get("relation"))
        for text, subject, predicate, content in conn.execute(
            "SELECT text, subject, predicate, content_json FROM memory_atoms ORDER BY atom_id"
        )
    ]
    return out


def _rows(hms: HMS, name: str) -> list[dict]:
    """add() every message of a chat fixture in turn and collect the rows.
    依次 add() 聊天夹具的每条消息并收集行。
    """
    rows: list[dict] = []
    for message in load_events(name).messages:
        rows.extend(
            hms.add(
                [{"role": message["role"], "content": message["content"]}],
                user_id="alice",
                run_id=f"run-{name}",
                timestamp=message.get("timestamp"),
            )["results"]
        )
    return rows


# ---------------------------------------------------------------------------
# (1) A scripted compact reply through HMS.add
# (1) 脚本化紧凑回复经 HMS.add
# ---------------------------------------------------------------------------
def test_compact_reply_through_hms_add_derives_the_dropped_fields():
    """A compact reply yields atoms whose dropped fields are derived (subject / predicate / objects read from the text), all seven axes are filled, every atom carries exactly its witness anchor and the speaker reaches actor_id only when the message names nobody.
    紧凑回复产出的原子，其被删字段都被推导出来（subject / predicate / objects 从文本读出），七个轴全部填充，每个原子恰好带自己的见证锚点，且只有消息未署名时 speaker 才成为 actor_id。

    Input / 输入：
        COMPACT_REPLY (three items: a named-speaker constraint, an assistant temporal fact with a full `when`, a habit) over COMPACT_MESSAGES.
        COMPACT_REPLY（三个条目：具名说话人的约束、带完整 `when` 的助手时态事实、一个习惯）作用于 COMPACT_MESSAGES。

    Output / 输出：
        Assertions on the add() rows: subject / predicate / relation / sentiment / topic_tags / entity_ids / confidence / attributed_to / actor_id / time_anchor / valid_from / scale_level / anchor_ids.
        对 add() 行的断言：subject / predicate / relation / sentiment / topic_tags / entity_ids / confidence / attributed_to / actor_id / time_anchor / valid_from / scale_level / anchor_ids。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    hms, _clock = _facade(
        "compact", llm={"provider": "scripted", "config": {"responses": [json.dumps(COMPACT_REPLY)]}}
    )
    rows = hms.add(COMPACT_MESSAGES, user_id="alice", run_id="run-compact", timestamp=STAMP)["results"]
    assert [c["task"] for c in hms.llm.calls] == ["memory_extraction_compact"]
    assert [r["event"] for r in rows] == ["ADD", "ADD", "ADD"]
    constraint, temporal, habit = rows
    # Derived fields (ingest.derive) and the constant confidence.
    # 推导字段（ingest.derive）与常量置信度。
    assert (constraint["hms"]["subject"], constraint["hms"]["predicate"]) == ("Katrina", "must_not")
    assert (temporal["hms"]["subject"], temporal["hms"]["predicate"]) == ("assistant", "did")
    assert (habit["hms"]["subject"], habit["hms"]["predicate"]) == ("user", "does")
    assert [r["confidence"] for r in rows] == [derive.COMPACT_CONFIDENCE] * 3
    assert [r["attributed_to"] for r in rows] == ["user", "assistant", "user"]
    atoms = hms.store.get_atoms([r["atom_id"] for r in rows])
    assert [a.content["relation"] for a in atoms] == ["constraint", "single_fact", "habit"]
    assert [a.content["objects"] for a in atoms] == [["billing state machine"], ["Sarah"], ["drinks black coffee after lunch"]]
    # Seven axes: time / validity / entity / topic / role / sentiment / scale.
    # 七个轴：时间 / 有效期 / 实体 / 主题 / 角色 / 情感 / 尺度。
    assert constraint["hms"]["time_anchor"] == STAMP_ISO and constraint["valid_from"] == STAMP_ISO
    assert temporal["hms"]["time_anchor"] == "2026-06-20T00:00:00.000000+00:00"
    assert temporal["valid_from"] == "2026-06-20T09:30:00.000000+00:00"
    assert all(r["valid_to"] is None for r in rows)
    assert constraint["hms"]["entity_ids"] == ["katrina"] and temporal["hms"]["entity_ids"] == ["sarah"]
    assert constraint["hms"]["topic_tags"] == ["billing", "constraint"] and "coffee" in habit["hms"]["topic_tags"]
    assert all(a.role == "extracted" for a in atoms)
    assert [r["hms"]["sentiment"] for r in rows] == ["neutral", "neutral", "neutral"]
    assert [r["scale_level"] for r in rows] == ["state", "event", "state"]
    assert [r["atom_type"] for r in rows] == ["state", "event", "state"]
    assert [r["hms"]["obs_type"] for r in rows] == ["constraint", "temporal_fact", "habit"]
    # Witness: each atom carries exactly the anchor of the message its event_idx named.
    # 见证：每个原子恰好带 event_idx 所指消息的锚点。
    scope = Scope(user_id="alice", run_id="run-compact")
    by_text = {
        str(ev.payload["text"]): hms.store.anchor_for_event(ev.event_id).anchor_id
        for ev in hms.store.last_events(scope, object_types=["message"], limit=10)
    }
    assert [r["anchor_ids"] for r in rows] == [[by_text[m["content"]]] for m in COMPACT_MESSAGES]
    # speaker → actor_id: the message `name` wins, else the named speaker, else nothing.
    # speaker → actor_id：消息 `name` 优先，其次具名说话人，否则为空。
    assert constraint["actor_id"] == "Katrina" and "actor_id" not in temporal and habit["actor_id"] == "Bob"
    hms.close()


# ---------------------------------------------------------------------------
# (2) The full contract is untouched
# (2) 完整契约原样不动
# ---------------------------------------------------------------------------
def test_full_contract_is_byte_identical_and_never_derives(golden, events, rule_llm):
    """The full prompts hash to their pre-compact bytes, a full reply keeps the LLM's own field values, and the rule LLM's full observations still equal the golden file.
    完整提示词的哈希等于加入紧凑契约之前的字节，完整回复保留 LLM 自己给出的字段值，规则 LLM 的完整观察仍等于黄金文件。

    Input / 输入：
        The two full prompt constants, a scripted full reply with non-derivable values, chat_preference through the rule LLM.
        两个完整提示词常量、带不可推导取值的脚本化完整回复、经规则 LLM 的 chat_preference。

    Output / 输出：
        Hash equalities, the stored sentiment / predicate / confidence / objects, the golden comparison.
        哈希相等、存储的 sentiment / predicate / confidence / objects、黄金比对。

    State effect / 状态影响：
        Rewrites the golden file under --update-golden.
        --update-golden 下重写黄金文件。
    """
    for name, digest in FULL_PROMPT_SHA256.items():
        assert hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest() == digest, name
    # The library default is 'lines' since 2026-09-18 (CONTRACT §6.6, changelog row 108); the full contract below is unchanged.
    # 库默认自 2026-09-18 起为 'lines'（CONTRACT §6.6，变更记录第 108 行）；下面的完整契约原样不变。
    assert IngestConfig().extraction_output == "lines"
    reply = {
        "observations": [{
            "id": "0", "event_idx": 0, "type": "fact", "subject": "project", "predicate": "ships", "objects": ["v2"],
            "relation": "comparison", "text": "The project ships v2 next week.", "date_observed": "2026-06-01",
            "confidence": 0.35, "sentiment": "negative", "topic_tags": ["release"], "entity_mentions": ["v2"],
        }],
        "eps": [], "decisions": [{"id": "0", "decision": "ADD", "target_id": None, "reason": "no candidates"}],
    }  # fmt: skip
    hms, _clock = _facade("full", llm={"provider": "scripted", "config": {"responses": [json.dumps(reply)]}})
    row = hms.add("We ship v2 next week.", user_id="alice", run_id="run-full", timestamp=STAMP)["results"][0]
    assert [c["task"] for c in hms.llm.calls] == ["memory_extraction"]
    assert (row["hms"]["subject"], row["hms"]["predicate"], row["hms"]["sentiment"]) == ("project", "ships", "negative")
    assert row["confidence"] == 0.35 and row["hms"]["topic_tags"][0] == "release"
    assert hms.store.get_atom(row["atom_id"]).content["objects"] == ["v2"]
    hms.close()
    existing: list[dict] = []
    calls: list[list[dict]] = []
    for message in events("chat_preference").messages:
        user = prompts.build_observation_prompt(
            existing_memories=existing,
            new_messages=[{"idx": 0, "role": message["role"], "text": message["content"], "timestamp": message["timestamp"]}],
            last_messages=[],
            observation_date=message["timestamp"][:10],
            current_date="2026-09-02",
        )
        out = rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=user)
        calls.append(out)
        existing = existing + [{"id": str(len(existing) + i), "text": o["text"]} for i, o in enumerate(out)]
    golden.masked("observations_chat_preference.json", calls)


# ---------------------------------------------------------------------------
# (3) Optional keys and `when` variants
# (3) 可选键与 `when` 的各种形态
# ---------------------------------------------------------------------------
def test_missing_optional_keys_and_when_variants_are_coerced(caplog):
    """A bare item (id / event_idx / text / type) is a valid observation; `when` becomes date_observed (and valid_from only as a full datetime); hints resolve through id_map; a bad witness still skips.
    只有 id / event_idx / text / type 的条目是合法观察；`when` 变为 date_observed（仅完整日期时间才成为 valid_from）；提示经 id_map 解析；错误的见证仍然跳过。

    Input / 输入：
        Hand-written compact items against one anchored user message dated 2026-06-15.
        针对一条 2026-06-15 落锚 user 消息的手写紧凑条目。

    Output / 输出：
        Assertions on the resulting Observation fields and on the extraction log.
        对所得 Observation 字段与抽取日志的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    bare = _compact({"id": "0", "event_idx": 0, "text": "User lives in Berlin.", "type": "fact"})
    assert bare is not None
    assert (bare.subject, bare.predicate, bare.objects, str(bare.relation)) == ("user", "lives_in", ["Berlin"], "single_fact")
    assert bare.speaker is None and bare.attributed_to == "user" and bare.supersedes_hint_ids == []
    assert bare.date_observed == DAY and bare.valid_from is None and bare.valid_to is None
    assert bare.confidence == derive.COMPACT_CONFIDENCE and str(bare.sentiment) == "neutral"
    assert bare.entity_mentions == ["Berlin"] and bare.anchor_ids == ["anc_000001"]
    date_only = _compact({"id": "0", "event_idx": 0, "text": "Sarah joined the team.", "type": "temporal_fact", "when": "2026-03-01"})
    assert date_only is not None and date_only.date_observed == "2026-03-01" and date_only.valid_from is None
    assert date_only.subject == "Sarah" and date_only.predicate == "did" and date_only.entity_mentions == ["Sarah"]
    assert date_only.objects == [] and str(date_only.relation) == "single_fact"
    full = _compact({"id": "0", "event_idx": 0, "text": "User flies to Oslo.", "type": "fact", "when": "2026-07-01T09:00:00+00:00", "valid_to": "2026-07-02"})
    assert full is not None and full.date_observed == "2026-07-01"
    assert full.valid_from == "2026-07-01T09:00:00.000000+00:00" and full.valid_to == "2026-07-02T00:00:00.000000+00:00"
    vague = _compact({"id": "0", "event_idx": 0, "text": "User flies to Oslo.", "type": "fact", "when": "next spring"})
    assert vague is not None and vague.date_observed == DAY and vague.valid_from is None
    hinted = _compact(
        {"id": "0", "event_idx": 0, "text": "User likes black coffee.", "type": "preference", "supersedes_hint_ids": ["0", "9"]},
        id_map={"0": "atm_000001"},
    )
    assert hinted is not None and str(hinted.type) == "preference_event" and hinted.predicate == "likes"
    assert hinted.objects == ["black coffee"]
    assert hinted.supersedes_hint_ids == ["atm_000001"] and str(hinted.relation) == "supersession"
    assert str(hinted.sentiment) == "positive" and hinted.topic_tags == ["coffee", "preference"]
    named = _compact({"id": "0", "event_idx": 0, "text": "Katrina dislikes cold calls.", "type": "preference_event", "speaker": "Katrina"})
    assert named is not None and named.subject == "Katrina" and named.predicate == "dislikes"
    assert named.objects == ["cold calls"] and str(named.sentiment) == "negative"
    moved = _compact({"id": "0", "event_idx": 0, "text": "User moved to Berlin.", "type": "fact"})
    assert moved is not None and (moved.predicate, moved.objects, str(moved.relation)) == ("lives_in", ["Berlin"], "supersession")
    bare_constraint = _compact({"id": "0", "event_idx": 0, "text": "Never touch prod.", "type": "constraint"})
    assert bare_constraint is not None and (bare_constraint.subject, bare_constraint.predicate) == ("project", "must_not")
    assert named.entity_mentions == ["Katrina"] and named.speaker == "Katrina" and named.attributed_to == "user"
    assert _compact({"id": "0", "text": "User lives in Berlin.", "type": "fact"}) is not None
    with caplog.at_level(logging.WARNING, logger=EXTRACTION_LOGGER):
        assert _compact({"id": "0", "event_idx": 7, "text": "User lives in Berlin.", "type": "fact"}) is None
        assert _compact({"id": "0", "event_idx": 0, "text": "", "type": "fact"}) is None
        assert _compact({"id": "0", "event_idx": 0, "text": ["not", "a", "string"], "type": "fact"}) is None
    messages = [r.getMessage() for r in caplog.records if r.name == EXTRACTION_LOGGER]
    assert sum("invalid event_idx" in m for m in messages) == 1 and sum("skipped" in m for m in messages) == 2


# ---------------------------------------------------------------------------
# (3b) The `entities` hint: full-mode parity on the entity axis, ordered union, malformed shapes
# (3b) `entities` 提示：实体轴与完整模式等价、有序并集、畸形形状
# ---------------------------------------------------------------------------
def test_compact_entities_hint_reaches_the_entity_axis_like_full_mode():
    """A compact reply whose item carries `entities: ["python"]` stores the same entity_ids as a full reply carrying `entity_mentions: ["python"]`; without the hint the lower-case common noun is absent, so the hint is what closes the gap.
    条目带 `entities: ["python"]` 的紧凑回复存下的 entity_ids 与带 `entity_mentions: ["python"]` 的完整回复相同；没有该提示时小写普通名词缺席，因此正是该提示补上了差距。

    Input / 输入：
        Three scripted facades over the same message 'I prefer python for scripting.': compact with the hint, full with entity_mentions, compact without the hint.
        同一条消息 'I prefer python for scripting.' 上的三个脚本化门面：带提示的 compact、带 entity_mentions 的 full、不带提示的 compact。

    Output / 输出：
        Assertions on the add() rows' entity_ids and predicate.
        对 add() 行的 entity_ids 与 predicate 的断言。

    State effect / 状态影响：
        None outside three in-memory stores.
        除三个内存存储外无副作用。
    """
    text = "User prefers python for scripting."
    compact_item = {"id": "0", "event_idx": 0, "text": text, "type": "preference_event", "speaker": None, "entities": ["python"]}
    full_item = {
        "id": "0", "event_idx": 0, "text": text, "type": "preference_event", "subject": "user", "predicate": "prefers",
        "objects": ["python"], "relation": "single_fact", "date_observed": DAY, "confidence": 0.7, "attributed_to": "user",
        "speaker": None, "sentiment": "positive", "topic_tags": ["preference"], "entity_mentions": ["python"],
    }  # fmt: skip
    bare_item = {k: v for k, v in compact_item.items() if k != "entities"}
    rows: dict[str, dict] = {}
    for label, output, item in (("hinted", "compact", compact_item), ("full", "full", full_item), ("bare", "compact", bare_item)):
        reply = {"observations": [item], "eps": [], "decisions": [{"id": "0", "decision": "ADD", "target_id": None, "reason": "no candidates"}]}
        hms, _clock = _facade(output, llm={"provider": "scripted", "config": {"responses": [json.dumps(reply)]}})
        rows[label] = hms.add("I prefer python for scripting.", user_id="alice", run_id="run-python", timestamp=STAMP)["results"][0]
        hms.close()
    assert rows["full"]["hms"]["entity_ids"] == ["python"]
    assert rows["hinted"]["hms"]["entity_ids"] == rows["full"]["hms"]["entity_ids"]
    assert rows["hinted"]["hms"]["predicate"] == rows["full"]["hms"]["predicate"] == "prefers"
    assert rows["bare"]["hms"]["entity_ids"] == [] and rows["bare"]["hms"]["predicate"] == "prefers"


def test_entity_union_keeps_llm_order_and_folds_case():
    """derive.union_mentions and expand_compact_item keep the LLM's entities first, append the extractor's hits and the speaker only when new under case folding, and drop blanks.
    derive.union_mentions 与 expand_compact_item 把 LLM 的 entities 放在最前，抽取器命中与说话人仅在折叠大小写后仍是新的时才追加，并丢弃空白。

    Input / 输入：
        Hand-written mention groups; a compact item over 'Katrina lives in Berlin.' with entities ['python', 'berlin', 'katrina', 'PYTHON', ' '] and speaker 'Katrina', then the same without a hint.
        手写的提及分组；'Katrina lives in Berlin.' 上带 entities ['python', 'berlin', 'katrina', 'PYTHON', ' '] 与 speaker 'Katrina' 的紧凑条目，再是不带提示的同一条目。

    Output / 输出：
        Assertions on the returned lists.
        对返回列表的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert derive.union_mentions(["python", "Berlin"], ["berlin", "Oslo"], ["Katrina"]) == ["python", "Berlin", "Oslo", "Katrina"]
    assert derive.union_mentions(None, [" python ", "", "PYTHON"], None) == ["python"]
    assert derive.union_mentions() == [] and derive.union_mentions([], [], []) == []
    item = {"id": "0", "event_idx": 0, "text": "Katrina lives in Berlin.", "type": "fact", "speaker": "Katrina"}
    kwargs: dict[str, Any] = {"obs_type": "fact", "speaker": "Katrina", "role": "user", "message_date": DAY, "extractor": None}
    hinted = derive.expand_compact_item(item, entities=["python", "berlin", "katrina", "PYTHON", " "], **kwargs)
    assert hinted["entity_mentions"] == ["python", "berlin", "katrina"]
    plain = derive.expand_compact_item(item, **kwargs)
    assert plain["entity_mentions"] == ["Katrina", "Berlin"]
    other = derive.expand_compact_item(item, entities=["python"], **{**kwargs, "speaker": "Bob"})
    assert other["entity_mentions"] == ["python", "Katrina", "Berlin", "Bob"]


def test_malformed_entities_hint_is_ignored_not_rejected(caplog):
    """A missing, None, string or dict-item `entities` value is coerced through _str_list; a number or a dict where the list belongs is ignored with one warning and the observation is still stored with the derived mentions.
    缺失、None、字符串或 dict 项形式的 `entities` 经 _str_list 规整；列表位置出现数字或 dict 时记一条警告并忽略，观察仍以推导出的提及存下。

    Input / 输入：
        Compact items over 'User lives in Berlin.' whose `entities` is absent, None, 'python', [{'name': 'python'}], 7 and {'a': 1}.
        'User lives in Berlin.' 上 `entities` 分别为缺失、None、'python'、[{'name': 'python'}]、7 与 {'a': 1} 的紧凑条目。

    Output / 输出：
        Assertions on the resulting Observation.entity_mentions and on the extraction log.
        对所得 Observation.entity_mentions 与抽取日志的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    base = {"id": "0", "event_idx": 0, "text": "User lives in Berlin.", "type": "fact"}
    expected = {
        "absent": ["Berlin"], "none": ["Berlin"], "string": ["python", "Berlin"], "dicts": ["python", "Berlin"],
        "number": ["Berlin"], "dict": ["Berlin"],
    }  # fmt: skip
    values = {"none": None, "string": "python", "dicts": [{"name": "python"}], "number": 7, "dict": {"a": 1}}
    with caplog.at_level(logging.WARNING, logger=EXTRACTION_LOGGER):
        for label, mentions in expected.items():
            item = dict(base) if label == "absent" else {**base, "entities": values[label]}
            obs = _compact(item)
            assert obs is not None and obs.entity_mentions == mentions, label
    warnings = [r.getMessage() for r in caplog.records if r.name == EXTRACTION_LOGGER]
    assert len(warnings) == 2 and all("entities ignored" in m for m in warnings)
    assert not any("skipped" in m for m in warnings)


# ---------------------------------------------------------------------------
# (4) The compact prompts
# (4) 紧凑提示词
# ---------------------------------------------------------------------------
def test_compact_prompts_omit_the_derived_fields_and_are_registered():
    """Both compact prompts are registered like their full twins, list only COMPACT_OBSERVATION_KEYS (including the `entities` hint) in their observation items, state the type vocabulary, carry the entities guideline, keep the supersession example (with entities) and the questions rule, and stay under 0.75 of their full twins' length.
    两个紧凑提示词与其完整孪生同样注册，观察条目只列 COMPACT_OBSERVATION_KEYS（含 `entities` 提示），写明 type 词表，带 entities 准则，保留取代示例（含 entities）与问句规则，且长度低于完整孪生的 0.75。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on configs.prompts.
        对 configs.prompts 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert prompts.COMPACT_OBSERVATION_KEYS == (
        "id", "event_idx", "text", "type", "speaker", "supersedes_hint_ids", "when", "valid_to", "entities",
    )
    assert prompts.ROOT_KEY["observation_extraction_compact"] == "observations"
    assert prompts.PROMPTS["observation_extraction_compact"] is prompts.OBSERVATION_EXTRACTION_COMPACT_PROMPT
    assert prompts.COMPOSITE_PROMPTS["memory_extraction_compact"] is prompts.MEMORY_EXTRACTION_COMPACT_PROMPT
    assert "memory_extraction_compact" not in prompts.ROOT_KEY and "memory_extraction_compact" not in prompts.TASKS
    pairs = (
        (prompts.OBSERVATION_EXTRACTION_COMPACT_PROMPT, prompts.OBSERVATION_EXTRACTION_PROMPT, "observation_extraction_compact"),
        (prompts.MEMORY_EXTRACTION_COMPACT_PROMPT, prompts.MEMORY_EXTRACTION_PROMPT, "memory_extraction_compact"),
    )
    for compact, full, task in pairs:
        assert prompts.task_of(compact) == task
        for heading in ("# ROLE", "# INPUTS", "# GUIDELINES", "# EXAMPLES", "# OUTPUT FORMAT"):
            assert heading in compact, (task, heading)
        items = _observation_items(compact)
        assert items and all(set(item) <= set(prompts.COMPACT_OBSERVATION_KEYS) for item in items)
        assert not any(field in item for item in items for field in DROPPED)
        section = prompts.output_format_section(compact)
        assert section.endswith(prompts.JSON_TRAILER)
        for field in ("subject", "predicate", "objects", "relation", "date_observed", "attributed_to", "sentiment"):
            assert f'"{field}"' not in section, (task, field)
        assert "fact | preference_event | habit | constraint | temporal_fact" in compact
        # The OUTPUT FORMAT item cites hint id "1" for observation "0": no observation-id / candidate-id coincidence.
        # OUTPUT FORMAT 条目里观察 "0" 引用提示 id "1"：观察 id 与候选 id 不重合。
        assert '"supersedes_hint_ids": ["1"]' in prompts.output_format_section(compact) and "switched to" in compact
        # The `entities` hint: named in the OUTPUT FORMAT item, one guideline line, and in the supersession example.
        # `entities` 提示：出现在 OUTPUT FORMAT 条目里、一行准则里，以及取代示例里。
        assert all("entities" in item for item in items) and items[0]["entities"] == ["black coffee"]
        assert "- `entities`: the named things the statement mentions, 0–6 short strings; omit when none." in compact
        assert compact.count('"entities": ["black coffee"]') == 2 and '"entity_mentions"' not in prompts.output_format_section(compact).split('"eps"')[0]
        assert "Questions, greetings and assistant boilerplate yield nothing" in compact
        assert "25 words" in compact and "same fact twice" in compact
        assert len(compact) < 0.75 * len(full), (task, len(compact), len(full))
        assert prompts.output_format_section(full) != section
    roots = json.loads(ujson.extract_json(prompts.output_format_section(prompts.MEMORY_EXTRACTION_COMPACT_PROMPT)))
    assert list(roots) == list(prompts.MEMORY_EXTRACTION_ROOTS)
    # EPs and decisions keep their full shapes; the prompt says when each is emitted.
    # EP 与决策保持完整形状；提示词写明各自何时输出。
    full_roots = json.loads(ujson.extract_json(prompts.output_format_section(prompts.MEMORY_EXTRACTION_PROMPT)))
    assert set(roots["eps"][0]) == set(full_roots["eps"][0]) and set(roots["decisions"][0]) == set(full_roots["decisions"][0])
    assert "ONLY when a trigger AND an action AND (an outcome OR a constraint) are present" in prompts.MEMORY_EXTRACTION_COMPACT_PROMPT
    assert "ONLY when Candidate Existing Memories is non-empty" in prompts.MEMORY_EXTRACTION_COMPACT_PROMPT


# ---------------------------------------------------------------------------
# (5) Config validation and wiring
# (5) 配置校验与接线
# ---------------------------------------------------------------------------
def test_extraction_output_config_validation_and_wiring():
    """extraction_output defaults to 'lines' (since 2026-09-18, CONTRACT §6.6, changelog row 108) — in IngestConfig, through a default-config HMS and in an Ingestor built without the argument — rejects unknown values (CONFIG_001 / VALIDATION_013), reaches the Ingestor through HMS and AsyncHMS, and works in both extraction modes.
    extraction_output 默认 'lines'（2026-09-18 起，CONTRACT §6.6，变更记录第 108 行）——IngestConfig、默认配置的 HMS、不传该参数构造的 Ingestor 三处一致——拒绝未知值（CONFIG_001 / VALIDATION_013），经 HMS 与 AsyncHMS 到达 Ingestor，且两种抽取模式都可用。

    Input / 输入：
        Config dicts, an Ingestor constructed directly, extract_all with a bad output, compact and staged + compact facades.
        配置字典、直接构造的 Ingestor、带非法 output 的 extract_all、compact 与 staged + compact 的门面。

    Output / 输出：
        Assertions over defaults, pass-through, rejections and the task names called.
        对默认值、透传、拒绝与所调用任务名的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    assert EXTRACTION_OUTPUTS == ("full", "compact", "lines") and IngestConfig().extraction_output == "lines"
    with pytest.raises(ConfigError) as exc:
        HMSConfig.from_dict({"ingest": {"extraction_output": "tiny"}, "store": {"path": ":memory:"}})
    assert exc.value.error_code == "CONFIG_001"
    # A facade built without an `ingest` section is lines end to end (config → Ingestor → task name), and an
    # Ingestor constructed without extraction_output follows its config rather than a hard-coded default.
    # 不带 `ingest` 段构造的门面端到端都是行式契约（配置 → Ingestor → 任务名）；不传 extraction_output 构造的 Ingestor
    # 跟随其配置而非硬编码默认值。
    untouched = HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}))
    assert untouched.config.ingest.extraction_output == "lines" and untouched.ingestor.extraction_output == "lines"
    untouched.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")
    assert [c["task"] for c in untouched.llm.calls] == ["memory_extraction_lines"]
    assert Ingestor(untouched.store, untouched.index, untouched.embedder, untouched.llm, untouched.config).extraction_output == "lines"
    untouched.close()
    # A config pinned to 'compact' reaches an Ingestor built without the argument, and the task name follows.
    # 钉在 'compact' 的配置同样到达不传该参数构造的 Ingestor，任务名随之改变。
    compact, _clock = _facade("compact")
    assert compact.config.ingest.extraction_output == "compact" and compact.ingestor.extraction_output == "compact"
    assert Ingestor(compact.store, compact.index, compact.embedder, compact.llm, compact.config).extraction_output == "compact"
    compact.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")
    assert [c["task"] for c in compact.llm.calls] == ["memory_extraction_compact"]
    with pytest.raises(ValidationError) as bad:
        Ingestor(compact.store, compact.index, compact.embedder, compact.llm, compact.config, extraction_output="tiny")
    assert bad.value.error_code == "VALIDATION_013" and bad.value.details["supported"] == ["full", "compact", "lines"]
    with pytest.raises(ValidationError) as bad_call:
        extraction.extract_all(
            compact.llm, events=[], ep_events=[], existing=[], candidates=[], last_events=[], known_constraints=[],
            recent_eps=[], observation_date=DAY, current_date=DAY, custom_instructions=None, scope=SCOPE, id_map={},
            output="tiny",
        )  # fmt: skip
    assert bad_call.value.error_code == "VALIDATION_013"
    compact.close()
    mirrored = AsyncHMS(
        HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, ingest={"extraction_output": "compact"})
    )
    assert mirrored._sync.ingestor.extraction_output == "compact"
    mirrored._sync.close()
    staged, _clock = _facade("compact", extraction_mode="staged")
    rows = staged.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")["results"]
    assert [c["task"] for c in staged.llm.calls] == ["observation_extraction_compact"]
    assert rows[0]["memory"] == "User likes oat-milk lattes in the morning." and rows[0]["hms"]["predicate"] == "likes"
    staged.close()


# ---------------------------------------------------------------------------
# (6) The rule LLM's compact replies through the pipeline
# (6) 规则 LLM 的紧凑回复经流水线
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CHAT_FIXTURES)
def test_rule_llm_compact_replies_keep_texts_witnesses_and_revisions(name):
    """On every chat fixture the rule LLM's compact replies carry only the compact keys and the pipeline stores the same texts, events (ADD / UPDATE / CONFLICT / NONE), witness anchors, types, sides, entity_ids, atom triples and axis / history / graph tables as in full output.
    在每个聊天夹具上，规则 LLM 的紧凑回复只带紧凑键，流水线存下的文本、事件（ADD / UPDATE / CONFLICT / NONE）、见证锚点、类型、侧别、entity_ids、原子三元组以及轴 / 历史 / 图表都与完整输出相同。

    Input / 输入：
        name: A tests/fixtures/events/chat_*.json stem.
        name：tests/fixtures/events/chat_*.json 的文件名。

    Output / 输出：
        Assertions comparing the add() rows of a full and a compact facade, and on the raw compact reply.
        比较完整与紧凑门面 add() 行的断言，以及对原始紧凑回复的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    full, _c1 = _facade("full")
    compact, _c2 = _facade("compact")
    full_rows, compact_rows = _rows(full, name), _rows(compact, name)
    assert [c["task"] for c in compact.llm.calls] == ["memory_extraction_compact"] * len(load_events(name).messages)
    keys = ("memory", "event", "anchor_ids", "attributed_to", "actor_id", "atom_type", "scale_level")
    assert [tuple(r.get(k) for k in keys) for r in compact_rows] == [tuple(r.get(k) for k in keys) for r in full_rows]
    assert [r["hms"]["obs_type"] for r in compact_rows] == [r["hms"]["obs_type"] for r in full_rows]
    assert [r["hms"]["entity_ids"] for r in compact_rows] == [r["hms"]["entity_ids"] for r in full_rows]
    assert [r.get("revises") for r in compact_rows] == [r.get("revises") for r in full_rows]
    assert all(r["confidence"] == derive.COMPACT_CONFIDENCE for r in compact_rows if r["hms"]["obs_type"])
    # The derived subject / predicate / objects / relation equal the rule LLM's own, so revision decisions, history
    # rows and graph edges are identical too (chat_conflict keeps its CONFLICT row; chat_multisession keeps 'Sarah').
    # 推导出的 subject / predicate / objects / relation 与规则 LLM 自己的相同，因此修订决策、历史行与图边也相同
    # （chat_conflict 保留 CONFLICT 行；chat_multisession 保留 'Sarah'）。
    full_tables, compact_tables = _tables(full), _tables(compact)
    for table in ("atom_triples", *IDENTICAL_TABLES):
        assert compact_tables[table] == full_tables[table], table
    events = [row["event"] for row in compact_tables["history"]]
    if name == "chat_conflict":
        assert events == ["ADD", "ADD", "CONFLICT"]
    if name == "chat_multisession":
        assert {triple[1] for triple in compact_tables["atom_triples"]} == {"Sarah"}
    for message in load_events(name).messages:
        stamp = message.get("timestamp") or STAMP
        user = prompts.build_observation_prompt(
            existing_memories=[], last_messages=[], observation_date=stamp[:10], current_date="2026-09-02",
            new_messages=[{"idx": 0, "role": message["role"], "text": message["content"], "timestamp": stamp}],
        )
        items = compact.llm.generate_json(
            "observation_extraction_compact", system=prompts.OBSERVATION_EXTRACTION_COMPACT_PROMPT, user=user
        )
        assert all(set(item) <= set(prompts.COMPACT_OBSERVATION_KEYS) for item in items)
        assert all(set(prompts.COMPACT_OBSERVATION_KEYS[:5]) <= set(item) for item in items)
        for item in items:
            assert not any(field in item for field in DROPPED)
            assert "when" not in item or item["when"] != stamp[:10]
    full.close()
    compact.close()


def test_rule_llm_compact_when_carries_a_dated_statement(rule_llm):
    """The rule LLM emits `when` only for a statement whose date differs from the message date, `entities` from the full observation's entity_mentions, and omits the hint list when empty.
    规则 LLM 只对日期与消息日期不同的陈述输出 `when`，`entities` 取完整观察的 entity_mentions，提示列表为空时省略。

    Input / 输入：
        rule_llm: A RuleLLM; the chat_multisession fixture's first message (a dated event).
        rule_llm：一个 RuleLLM；chat_multisession 夹具的第一条消息（带日期的事件）。

    Output / 输出：
        Assertions on the compact reply items.
        对紧凑回复条目的断言。

    State effect / 状态影响：
        None.
        无。
    """
    message = load_events("chat_multisession").messages[0]
    user = prompts.build_observation_prompt(
        existing_memories=[], last_messages=[], observation_date=message["timestamp"][:10], current_date="2026-09-02",
        new_messages=[{"idx": 0, "role": message["role"], "text": message["content"], "timestamp": message["timestamp"]}],
    )
    items = rule_llm.generate_json(
        "observation_extraction_compact", system=prompts.OBSERVATION_EXTRACTION_COMPACT_PROMPT, user=user
    )
    assert rule_llm.calls[-1]["task"] == "observation_extraction_compact"
    assert [item["type"] for item in items] == ["temporal_fact"] and items[0]["when"] == "2026-03-01"
    assert items[0]["text"] == "Sarah joined the billing team in March." and "supersedes_hint_ids" not in items[0]
    assert set(items[0]) == {"id", "event_idx", "text", "type", "speaker", "when", "entities"}
    assert items[0]["entities"] == ["Sarah"]
    llm = ScriptedLLM({"responses": [json.dumps({"observations": items})]})
    assert llm.generate_json("observation_extraction_compact", system=prompts.OBSERVATION_EXTRACTION_COMPACT_PROMPT, user=user) == items


# ---------------------------------------------------------------------------
# (7) The third-person statement parser and the silent empty decisions root
# (7) 第三人称陈述解析器与不出声的空决策根
# ---------------------------------------------------------------------------
def test_statement_triple_reads_subject_predicate_objects_from_third_person_text():
    """llms.rule.statement_triple reads subject / predicate / objects / relation cue from a third-person statement with the R1–R9 cue tables, in English and Chinese, and derive.py defines no cue of its own.
    llms.rule.statement_triple 用 R1–R9 的线索表从第三人称陈述中读出 subject / predicate / objects / 关系线索（中英文），且 derive.py 不定义任何自己的线索。

    Input / 输入：
        A table of (text, type) pairs covering every branch: preference (switch / negation / alternative), constraint (must_not / must / imperative / 'does not' / 禁止), habit, temporal_fact, identity, live / work / move, comparison, the inflected-verb and lexicon fallbacks.
        覆盖每个分支的 (text, type) 表：偏好（切换 / 否定 / 备选）、约束（must_not / must / 祈使 / 'does not' / 禁止）、习惯、时态事实、身份、居住 / 工作 / 搬迁、比较、变形动词与词表兜底。

    Output / 输出：
        Assertions on the returned dicts and on derive's module namespace.
        对返回字典与 derive 模块命名空间的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cases = [
        ("User likes black coffee.", "preference_event", ("user", "likes", ["black coffee"], None)),
        ("User switched to black coffee.", "preference_event", ("user", "likes", ["black coffee"], "supersession")),
        ("User no longer likes tea.", "preference_event", ("user", "dislikes", ["tea"], "supersession")),
        ("User prefers tea over coffee.", "preference_event", ("user", "prefers", ["tea", "coffee"], "comparison")),
        ("Katrina dislikes cold calls.", "preference_event", ("Katrina", "dislikes", ["cold calls"], None)),
        ("用户喜欢黑咖啡。", "preference_event", ("user", "likes", ["黑咖啡"], None)),
        ("用户讨厌冷电话。", "preference_event", ("user", "dislikes", ["冷电话"], None)),
        ("User enjoys hiking.", "preference_event", ("user", "likes", ["hiking"], None)),
        ("Katrina must not modify the billing state machine.", "constraint", ("Katrina", "must_not", ["billing state machine"], None)),
        ("User does not touch prod.", "constraint", ("user", "must_not", ["prod"], None)),
        ("Never touch prod.", "constraint", (None, "must_not", ["prod"], None)),
        ("The team must run tests before deploy.", "constraint", ("project", "must", ["tests before deploy"], None)),
        ("团队禁止直接改生产库。", "constraint", ("project", "must_not", ["直接改生产库"], None)),
        ("User usually drinks black coffee after lunch.", "habit", ("user", "does", ["drinks black coffee after lunch"], None)),
        ("Sarah joined the billing team in March.", "temporal_fact", ("Sarah", "did", [], None)),
        ("Assistant scheduled the dentist visit for Sarah.", "temporal_fact", ("assistant", "did", ["Sarah"], None)),
        ("User will visit Oslo next week.", "temporal_fact", ("user", "plans", ["Oslo"], None)),
        ("User is named Alice.", "fact", ("user", "is_named", ["Alice"], None)),
        ("User is a backend engineer.", "fact", ("user", "is", ["backend engineer"], None)),
        ("User lives in Berlin.", "fact", ("user", "lives_in", ["Berlin"], None)),
        ("User live in Paris.", "fact", ("user", "lives_in", ["Paris"], None)),
        ("User works at Acme Corp.", "fact", ("user", "works_at", ["Acme Corp"], None)),
        ("User moved to Berlin.", "fact", ("user", "lives_in", ["Berlin"], "supersession")),
        ("用户搬到柏林。", "fact", ("user", "lives_in", ["柏林"], "supersession")),
        ("User compared Kafka and SQS; no exclusive final choice was stated.", "fact", ("user", "compares", ["Kafka", "SQS"], "comparison")),
        ("Lucia explains that the next assembly line connects to the computer.", "fact", ("Lucia", "explains", [], None)),
        ("User uses Python for scripting.", "fact", ("user", "uses", ["Python"], None)),
        ("User always drinks water.", "fact", ("user", "states", [], None)),
        ("This is the last day.", "fact", (None, "is", [], None)),
        ("", "fact", (None, "states", [], None)),
    ]  # fmt: skip
    for text, obs_type, expected in cases:
        got = statement_triple(text, obs_type)
        assert (got["subject"], got["predicate"], got["objects"], got["relation"]) == expected, text
    # derive.py composes: it imports statement_triple and compiles no cue regex of its own.
    # derive.py 只做组合：它 import statement_triple，且不编译任何自己的线索正则。
    assert derive.statement_triple is statement_triple
    assert [name for name in vars(derive) if name.endswith("_RE")] == ["_FULL_DATETIME_RE"]
    assert derive.subject_for(None, None, "constraint", "user") == "project"
    assert derive.subject_for(None, "Katrina", "constraint", "user") == "Katrina"
    assert derive.subject_for("Sarah", "Katrina", "fact", "assistant") == "Sarah"
    assert derive.subject_for(None, None, "fact", "assistant") == "assistant"
    assert derive.relation_of("fact", [], "comparison") == "comparison" and derive.relation_of("fact", ["0"], None) == "supersession"


def test_empty_decisions_without_candidates_are_silent(caplog):
    """In compact mode a reply with `"decisions": []` and no candidates shown stores every observation as ADD without a per-observation warning; the same reply with candidates shown still warns once per observation.
    紧凑模式下，未呈现候选时 `"decisions": []` 的回复把每条观察存为 ADD 且不逐条记录警告；呈现了候选的同一回复仍每条观察警告一次。

    Input / 输入：
        A scripted compact reply with two observations and an empty decisions list, through HMS.add on an empty store and then on a store holding a candidate.
        带两条观察与空决策列表的脚本化紧凑回复，先经空 store 的 HMS.add，再经持有候选的 store。

    Output / 输出：
        Assertions on the add() events and on the extraction log.
        对 add() 事件与抽取日志的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    reply = {
        "observations": [
            {"id": "0", "event_idx": 0, "text": "User lives in Berlin.", "type": "fact", "speaker": None},
            {"id": "1", "event_idx": 0, "text": "User likes black coffee.", "type": "preference_event", "speaker": None},
        ],
        "eps": [],
        "decisions": [],
    }  # fmt: skip
    hms, _clock = _facade("compact", llm={"provider": "scripted", "config": {"responses": [json.dumps(reply)] * 2}})
    with caplog.at_level(logging.WARNING, logger=EXTRACTION_LOGGER):
        first = hms.add("I live in Berlin and I like black coffee.", user_id="alice", run_id="run-silent", timestamp=STAMP)["results"]
    assert [r["event"] for r in first] == ["ADD", "ADD"]
    assert not [r for r in caplog.records if r.name == EXTRACTION_LOGGER]
    with caplog.at_level(logging.WARNING, logger=EXTRACTION_LOGGER):
        second = hms.add("I live in Berlin and I like black coffee.", user_id="alice", run_id="run-silent", timestamp=STAMP)["results"]
    warnings = [r.getMessage() for r in caplog.records if r.name == EXTRACTION_LOGGER]
    # The stored atoms are exact-text duplicates of the candidates, so the pipeline's own hash match makes them NONE;
    # the empty decisions list still drew one warning per observation because candidates were shown.
    # 存下的原子与候选文本完全相同，流水线自己的哈希匹配将其判为 NONE；因呈现了候选，空决策列表仍每条观察记一次警告。
    assert [r["event"] for r in second] == ["NONE", "NONE"]
    assert sum("no decision for observation" in m for m in warnings) == 2
    hms.close()

