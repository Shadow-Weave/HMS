"""Coerce-never-reject tests for the LLM extraction layer: vocabulary mismatches are folded, only witness integrity and malformed types skip.
LLM 抽取层"只规整、不拒绝"测试：词表不匹配被折叠，只有见证完整性与类型畸形才跳过。
"""

# Purpose (EN): Pin every coercion of ingest.extraction (type / relation / sentiment / confidence / instants / event_idx
# / speaker / EP fields / decision verdicts and shapes) and the speaker → actor_id mapping of ingest.atoms.
# 目的（中）：钉住 ingest.extraction 的每一项规整（type / relation / sentiment / confidence / 时刻 / event_idx / speaker /
# EP 字段 / 决策裁决与形状）以及 ingest.atoms 的 speaker → actor_id 映射。

# Input (EN): Hand-written LLM item dicts, the conftest clock / id helpers, the test_ingest harness and ScriptedLLM.
# 输入（中）：手写的 LLM 条目字典、conftest 的时钟 / id 辅助、test_ingest 的 harness 与 ScriptedLLM。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pytest

from hms_elite.configs import prompts
from hms_elite.enums import Decision, Reward
from hms_elite.graph.support_graph import SupportGraph
from hms_elite.ingest import extraction
from hms_elite.ingest import pipeline as ingest_pipeline
from hms_elite.ingest.atoms import atom_from_observation, relation_atom_from_observation
from hms_elite.ingest.events import anchor_events, events_from_messages
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import Anchor, MemoryAtom, Observation, Scope, SourceEvent
from hms_elite.mvmi.query import get_all
from hms_elite.ops.consolidation import StateScheduler
from tests.conftest import FrozenClock, SequentialIds
from tests.test_ingest import _Harness

EXTRACTION_LOGGER = "hms_elite.ingest.extraction"
DAY = "2026-06-15"
STAMP = "2026-06-15T10:04:00Z"
STAMP_ISO = "2026-06-15T10:04:00.000000+00:00"
SCOPE = Scope(user_id="alice", run_id="run-coerce")
BASE = {
    "id": "0", "event_idx": 0, "type": "fact", "subject": "user", "predicate": "likes", "objects": ["tea"],
    "text": "User likes tea.", "date_observed": DAY,
}  # fmt: skip


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
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
    clock = FrozenClock()
    return StateStore(":memory:", now=clock.now, id_factory=SequentialIds())


def _observe(item: dict, *, messages: list[dict] | None = None, store: StateStore | None = None) -> Observation | None:
    """Run one item through _observation_from_item against one anchored user message.
    让一个条目经 _observation_from_item 处理，见证为一条已落锚的 user 消息。
    """
    store = store or _fresh()
    events = _events(store, messages or [{"role": "user", "content": "I like tea."}])
    return extraction._observation_from_item(
        ScriptedLLM({"responses": []}), item, events=events, observation_date=DAY, scope=SCOPE, id_map={},
        what="memory_extraction.observations",
    )  # fmt: skip


def _ep(item: dict, *, ep_map: dict[str, str] | None = None) -> Any:
    """Run one EP item through _ep_from_item against one anchored user message.
    让一个 EP 条目经 _ep_from_item 处理，见证为一条已落锚的 user 消息。
    """
    store = _fresh()
    events = _events(store, [{"role": "user", "content": "I patched retry.ts and the tests failed."}])
    return extraction._ep_from_item(
        ScriptedLLM({"responses": []}), item, events=events, ep_map=ep_map or {}, scope=SCOPE,
        what="memory_extraction.eps",
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 1. Enum fields: unknown values fold to the default instead of skipping the item
# 1. 枚举字段：未知值折叠为默认值，而不是跳过条目
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("type", "plan", "fact"),
        ("type", "Fact", "fact"),
        ("type", None, "fact"),
        ("type", "", "fact"),
        ("type", "preference", "preference_event"),
        ("type", "Temporal Fact", "temporal_fact"),
        ("type", "rule", "constraint"),
        ("type", ["habit"], "habit"),
        ("relation", "update", "supersession"),
        ("relation", "Comparison", "comparison"),
        ("relation", "revision", "supersession"),
        ("relation", "weird", "single_fact"),
        ("relation", None, "single_fact"),
        ("sentiment", "happy", "positive"),
        ("sentiment", "Positive", "positive"),
        ("sentiment", "very positive", "positive"),
        ("sentiment", "ambivalent", "mixed"),
        ("sentiment", "none", "unknown"),
        ("sentiment", "bizarre", "unknown"),
        ("sentiment", "", "unknown"),
    ],
)
def test_enum_fields_coerce_instead_of_rejecting(field, value, expected, caplog):
    """An out-of-vocabulary type / relation / sentiment folds to the closest member (or the default) and the item survives.
    词表外的 type / relation / sentiment 折叠为最接近的成员（或默认值），条目得以保留。

    Input / 输入：
        field: The enum field under test.
        field：被测枚举字段。

        value: The raw LLM value (synonym, case variant, list, None, '' or garbage).
        value：LLM 的原始值（同义词、大小写变体、列表、None、'' 或垃圾值）。

        expected: The member value the observation must carry.
        expected：观察必须携带的成员值。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: the observation is built with the expected value; no WARNING is logged (a DEBUG line at most).
        断言：观察以期望值构造；没有 WARNING（至多一条 DEBUG）。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=EXTRACTION_LOGGER)
    obs = _observe({**BASE, field: value})
    assert obs is not None and str(getattr(obs, field)) == expected
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    ("field", "value", "expected", "warned"),
    [
        ("sentiment", "not positive", "unknown", True),
        ("sentiment", "non-negative", "unknown", True),
        ("sentiment", "not negative", "unknown", True),
        ("type", "not a constraint", "fact", True),
        ("type", "non-habit", "fact", True),
        ("type", "artifact", "fact", False),
        ("type", "inhabit", "fact", False),
        ("type", "must not", "constraint", False),
    ],
)
def test_negated_enum_values_fall_to_the_default_with_a_warning(field, value, expected, warned, caplog):
    """A negated value ('not positive', 'non-habit') never folds to the member it names; a member that is only a substring ('artifact', 'inhabit') does not match either.
    被否定的值（'not positive'、'non-habit'）绝不折叠到它点名的成员；仅作子串出现的成员（'artifact'、'inhabit'）也不匹配。

    Input / 输入：
        field: The enum field under test.
        field：被测枚举字段。

        value: A negated, substring-only or exact-alias value.
        value：被否定、仅子串、或精确别名的值。

        expected: The member the observation must carry (the default for negations and substrings).
        expected：观察必须携带的成员（否定与子串情形为默认值）。

        warned: Whether a WARNING naming the negated field is expected.
        warned：是否期望一条点名被否定字段的 WARNING。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: the value is the default, never the inverted member; a negation logs exactly one WARNING, a
        substring only a DEBUG line; the Reward enum of EP items follows the same rule.
        断言：值为默认值而非反转后的成员；否定恰好记一条 WARNING，子串只记 DEBUG；EP 条目的 Reward 枚举遵循同一规则。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=EXTRACTION_LOGGER)
    obs = _observe({**BASE, field: value})
    assert obs is not None and str(getattr(obs, field)) == expected
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == (1 if warned else 0)
    if warned:
        assert f"negated {field} {value!r}" in warnings[0]
    reward = extraction._enum_value(
        "not positive", Reward, default="unknown", aliases=extraction._REWARD_ALIASES, what="t", field="r"
    )
    assert reward == "unknown"


# ---------------------------------------------------------------------------
# 2. Confidence: numeric strings, percentages, word scales, defaults; out-of-range numbers keep their meaning
# 2. 置信度：数字串、百分比、词级刻度、默认值；越界数字不改变含义
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.9", 0.9), (85, 0.85), ("85%", 0.85), (1.2, 0.7), (150, 0.7), (-0.1, 0.7), ("high", 0.9), ("medium", 0.7),
        ("low", 0.4), (None, 0.7), ("", 0.7), (True, 0.7), ("n/a", 0.7), (0.65, 0.65), (100, 1.0), (5, 0.7),
        (10, 0.7), (11, 0.11), ("150%", 0.7), (1, 1.0), (0, 0.0),
    ],
)  # fmt: skip
def test_confidence_is_rescaled_and_clamped(value, expected):
    """Confidence values an LLM writes in other scales land in [0, 1]; numbers that are neither a probability nor a percentage take the default instead of a value near 0.
    LLM 以其它刻度写出的置信度落到 [0, 1] 内；既非概率也非百分比的数取默认值，而不是接近 0 的值。

    Input / 输入：
        value: The raw confidence (string, percent, word, out-of-range number, None, '', bool).
        value：原始置信度（字符串、百分比、词、越界数字、None、''、布尔）。

        expected: The stored probability (0.7 for 1.2 / -0.1 / 5 / 10 / 150 / '150%').
        expected：存储的概率（1.2 / -0.1 / 5 / 10 / 150 / '150%' 为 0.7）。

    Output / 输出：
        Assertions: `_prob` and the built observation agree on the expected value.
        断言：`_prob` 与构造出的观察都给出期望值。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    assert extraction._prob(value) == pytest.approx(expected)
    obs = _observe({**BASE, "confidence": value})
    assert obs is not None and obs.confidence == pytest.approx(expected)


@pytest.mark.parametrize("value", [1.2, -1, 5, 150, "150%"])
def test_out_of_range_confidence_is_traced_and_survives_prune(store, clock, value, caplog):
    """An out-of-range confidence is stored as the default with one WARNING naming the field, so the prune phase does not archive the atom as decayed.
    越界置信度以默认值存储并记一条点名字段的 WARNING，因此 prune 阶段不会把该原子当作衰减归档。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        value: A confidence that is neither a probability nor a percentage.
        value：既非概率也非百分比的置信度。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: the atom carries confidence 0.7; exactly one ingest-time WARNING names `confidence` and the raw
        value; a prune run one second later archives nothing and both atoms stay live.
        断言：原子的置信度为 0.7；恰好一条摄取时 WARNING 点名 `confidence` 与原始值；一秒后的 prune 不归档任何原子，
        两个原子都保持有效。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    bad = {**BASE, "confidence": value}
    good = {**BASE, "id": "1", "predicate": "lives_in", "objects": ["Paris"], "text": "User lives in Paris.",
            "confidence": 0.9}  # fmt: skip
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": [bad, good], "eps": []}]}))
    h.add("I like tea and I live in Paris.", timestamp=STAMP, user_id="alice", run_id="run-coerce")
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(SCOPE)["atoms"])}
    assert atoms["User likes tea."].confidence == pytest.approx(0.7)
    lines = [r.getMessage() for r in caplog.records if "confidence" in r.getMessage()]
    assert len(lines) == 1 and f"confidence {value!r}" in lines[0] and "default 0.7" in lines[0]
    scheduler = StateScheduler(
        store, h.index, h.embedder, h.llm, SupportGraph(store, h.config.graph), h.ingestor, None, h.config
    )
    clock.tick(seconds=1)
    report = scheduler.run(scope=SCOPE, phases=["prune"])
    assert report["phases"]["prune"]["archived"] == []
    assert [a.archived_reason for a in store.get_atoms([a.atom_id for a in atoms.values()])] == [None, None]


# ---------------------------------------------------------------------------
# 3. Instants: ISO and date-only accepted, everything else None, a bare year never becomes 1970
# 3. 时刻：接受 ISO 与仅日期，其它为 None，裸年份绝不变成 1970
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-01", "2024-01-01T00:00:00.000000+00:00"),
        ("2024-01-01T10:00:00Z", "2024-01-01T10:00:00.000000+00:00"),
        ("next week", None),
        ("", None),
        ("unknown", None),
        ("2024", None),
        (2024, None),
        (2024.0, None),
        (None, None),
    ],
)
def test_valid_from_accepts_iso_or_date_only_else_none(value, expected):
    """valid_from keeps ISO / date-only values and drops anything else to None (no epoch reading of a bare year).
    valid_from 保留 ISO / 仅日期的值，其它一律变 None（裸年份不按 epoch 读）。

    Input / 输入：
        value: The raw valid_from.
        value：原始 valid_from。

        expected: The normalised instant or None.
        expected：归一化后的时刻或 None。

    Output / 输出：
        Assertions: the observation is built and its valid_from equals expected.
        断言：观察被构造，其 valid_from 等于 expected。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    obs = _observe({**BASE, "valid_from": value})
    assert obs is not None and obs.valid_from == expected


@pytest.mark.parametrize(
    ("valid_from", "valid_to"),
    [("2026-07-01", "2026-06-01"), (None, "2020-01-01"), ("2026-06-16", 2030)],
)
def test_contradictory_valid_to_is_dropped_and_the_batch_survives(store, clock, valid_from, valid_to, caplog):
    """A valid_to before valid_from (or before the witnessed message) is dropped, so MemoryAtom's invariant cannot abort HMS.add().
    早于 valid_from（或早于被见证消息）的 valid_to 被丢弃，MemoryAtom 的不变量因此无法中止 HMS.add()。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        valid_from: The item's valid_from (may be None).
        valid_from：条目的 valid_from（可为 None）。

        valid_to: A valid_to that precedes valid_from / the message, or a bare year (unreadable → None).
        valid_to：早于 valid_from / 消息的 valid_to，或裸年份（不可读 → None）。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: both observations of the reply persist; the bad one has valid_to None; nothing is "skipped".
        断言：回复中的两条观察都持久化；有问题的那条 valid_to 为 None；没有条目被"跳过"。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    bad = {**BASE, "id": "0", "text": "User likes tea.", "valid_from": valid_from, "valid_to": valid_to}
    good = {**BASE, "id": "1", "predicate": "owns", "objects": ["a kettle"], "text": "User owns a kettle."}
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": [bad, good], "eps": []}]}))
    res = h.add("I like tea and I own a kettle.", timestamp=STAMP, user_id="alice", run_id="run-coerce")
    assert sorted(r["memory"] for r in res.rows) == ["User likes tea.", "User owns a kettle."]
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(SCOPE)["atoms"])}
    assert atoms["User likes tea."].valid_to is None
    assert not [r for r in caplog.records if "skipped" in r.getMessage()]


def test_temporal_fact_valid_to_before_date_observed_is_dropped_and_the_batch_survives(store, clock, caplog):
    """A temporal_fact whose valid_to lies between the message time and its later date_observed is floored at date_observed midnight (the valid_from atom_from_observation assigns), so the batch is not lost.
    valid_to 落在消息时刻与更晚的 date_observed 之间的 temporal_fact，以 date_observed 零点（atom_from_observation 赋的 valid_from）为下限，因此批次不会丢失。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: both observations persist; the event atom has valid_from = date_observed midnight and valid_to
        None; the warning names the midnight floor, not the message time.
        断言：两条观察都持久化；事件原子的 valid_from 为 date_observed 零点、valid_to 为 None；警告点名零点下限而非消息时刻。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    bad = {**BASE, "type": "temporal_fact", "predicate": "attends", "objects": ["the conference"],
           "text": "User attends the conference on June 20.", "date_observed": "2026-06-20", "valid_from": None,
           "valid_to": "2026-06-18T00:00:00Z"}  # fmt: skip
    good = {**BASE, "id": "1", "predicate": "owns", "objects": ["a kettle"], "text": "User owns a kettle."}
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": [bad, good], "eps": []}]}))
    res = h.add("I attend the conference on June 20; I own a kettle.", timestamp=STAMP, user_id="alice",
                run_id="run-coerce")  # fmt: skip
    assert sorted(r["memory"] for r in res.rows) == ["User attends the conference on June 20.", "User owns a kettle."]
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(SCOPE)["atoms"])}
    event = atoms["User attends the conference on June 20."]
    assert event.valid_from == "2026-06-20T00:00:00.000000+00:00" and event.valid_to is None
    lines = [r.getMessage() for r in caplog.records if "valid_to" in r.getMessage()]
    assert len(lines) == 1 and "precedes 2026-06-20T00:00:00.000000+00:00" in lines[0]


def test_an_invalid_observation_atom_is_dropped_alone_by_the_pipeline(store, clock, monkeypatch, caplog):
    """When an observation's atom still violates a MemoryAtom invariant, the pipeline drops that observation with a warning and keeps the rest of the batch.
    当某条观察的原子仍违反 MemoryAtom 不变量时，pipeline 记警告并只丢弃该观察，批次其余部分保留。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        monkeypatch: Replaces the pipeline's atom_from_observation so the tea atom carries valid_to < valid_from.
        monkeypatch：替换 pipeline 的 atom_from_observation，使 tea 原子的 valid_to < valid_from。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: add() returns only the kettle row; `dropped` records the observation; one pipeline WARNING.
        断言：add() 只返回 kettle 行；`dropped` 记录该观察；一条 pipeline WARNING。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    real = ingest_pipeline.atom_from_observation

    def broken(obs: Observation, **kwargs: Any) -> MemoryAtom:
        atom = real(obs, **kwargs)
        if obs.text == "User likes tea.":
            return MemoryAtom(**{**atom.model_dump(), "valid_to": "2020-01-01T00:00:00.000000+00:00"})
        return atom

    monkeypatch.setattr(ingest_pipeline, "atom_from_observation", broken)
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.pipeline")
    good = {**BASE, "id": "1", "predicate": "owns", "objects": ["a kettle"], "text": "User owns a kettle."}
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": [BASE, good], "eps": []}]}))
    res = h.add("I like tea and I own a kettle.", timestamp=STAMP, user_id="alice", run_id="run-coerce")
    assert [r["memory"] for r in res.rows] == ["User owns a kettle."]
    assert res.dropped == [{"kind": "observation", "text": "User likes tea.", "reason": "invalid_atom"}]
    assert [a.text for a in store.get_atoms(store.valid_ids(SCOPE)["atoms"])] == ["User owns a kettle."]
    lines = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(lines) == 1 and "valid_to must be >= valid_from" in lines[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2024/01/05", "2024-01-05"), ("2024.1.5", "2024-01-05"), ("next week", DAY), (None, DAY), ("2024-02-03", "2024-02-03")],
)
def test_date_observed_normalises_separators_and_falls_back(value, expected, caplog):
    """date_observed accepts '/' and '.' separators; an unreadable value falls back to the batch date with a DEBUG line.
    date_observed 接受 '/' 与 '.' 分隔符；不可读的值回退为批次日期并记一条 DEBUG 日志。

    Input / 输入：
        value: The raw date_observed.
        value：原始 date_observed。

        expected: The YYYY-MM-DD the observation carries.
        expected：观察携带的 YYYY-MM-DD。

        caplog: pytest log capture at DEBUG.
        caplog：DEBUG 级别的 pytest 日志捕获。

    Output / 输出：
        Assertions: the date matches; the fallback (and only the fallback) logs a DEBUG line naming date_observed.
        断言：日期一致；回退（且仅回退）记录一条点名 date_observed 的 DEBUG 日志。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=EXTRACTION_LOGGER)
    obs = _observe({**BASE, "date_observed": value})
    assert obs is not None and obs.date_observed == expected
    debug = [r for r in caplog.records if "date_observed" in r.getMessage()]
    assert bool(debug) == (value == "next week") and all(r.levelno == logging.DEBUG for r in debug)


# ---------------------------------------------------------------------------
# 4. event_idx: vocabulary variants are read, out-of-range still skips (witness integrity)
# 4. event_idx：词表变体可读，越界仍然跳过（见证完整性）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("value", "n", "expected"), [("0", 2, 0), (1.0, 2, 1), ("1", 2, 1), (None, 1, 0)])
def test_event_idx_reads_digit_strings_and_integral_floats(value, n, expected):
    """A digit string or an integral float indexes the witness; a missing index with exactly one event is that event.
    数字串或整数值浮点用于索引见证；恰有一个事件时缺失的索引即该事件。

    Input / 输入：
        value: The raw event_idx.
        value：原始 event_idx。

        n: Number of events in the prompt list.
        n：提示词事件列表的长度。

        expected: The resolved index.
        expected：解析出的索引。

    Output / 输出：
        Assertions: `_event_index` returns expected.
        断言：`_event_index` 返回 expected。

    State effect / 状态影响：
        None.
        无。
    """
    assert extraction._event_index({"event_idx": value}, n, what="t") == expected


@pytest.mark.parametrize("value", [5, -1, "x", None, True, 1.5])
def test_event_idx_out_of_range_or_unreadable_still_skips(value, caplog):
    """An out-of-range or unreadable event_idx (two events) still skips: the witness is never guessed (integrity, not vocabulary).
    越界或不可读的 event_idx（两个事件时）仍然跳过：见证从不猜测（这是完整性而非词表问题）。

    Input / 输入：
        value: The raw event_idx (5, -1, 'x', None, True, 1.5).
        value：原始 event_idx（5、-1、'x'、None、True、1.5）。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: the item yields None and one warning names the invalid event_idx.
        断言：条目返回 None，一条警告点名非法的 event_idx。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    messages = [{"role": "user", "content": "I like tea."}, {"role": "assistant", "content": "Noted."}]
    assert _observe({**BASE, "event_idx": value}, messages=messages) is None
    assert [r for r in caplog.records if "invalid event_idx" in r.getMessage()]


# ---------------------------------------------------------------------------
# 5. Speaker: attributed_to names, explicit speaker, side words, collective words, actor_id on the atom
# 5. 说话人：attributed_to 里的名字、显式 speaker、侧别词、集体词、原子上的 actor_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("item", "side", "speaker", "extra_mention"),
    [
        ({"attributed_to": "Katrina"}, "user", "Katrina", True),
        ({"attributed_to": "user", "speaker": "Shure"}, "user", "Shure", True),
        ({"attributed_to": "Katrina", "speaker": "Shure"}, "user", "Shure", True),
        ({"attributed_to": "User"}, "user", None, False),
        ({"attributed_to": "USER"}, "user", None, False),
        ({"attributed_to": "Assistant"}, "assistant", None, False),
        ({"attributed_to": "system"}, "assistant", None, False),
        ({"attributed_to": "group"}, "user", None, False),
        ({"attributed_to": ["user"]}, "user", None, False),
        ({"attributed_to": 5}, "user", None, False),
        ({"speaker": "group"}, "user", None, False),
        ({"speaker": ""}, "user", None, False),
        ({"speaker": "Lucia", "entity_mentions": ["lucia"]}, "user", "Lucia", False),
    ],
)
def test_speaker_is_a_first_class_field(item, side, speaker, extra_mention):
    """`speaker` is item['speaker'], else a name in attributed_to; side / collective words never are; appended to entity_mentions once.
    `speaker` 取 item['speaker']，否则取 attributed_to 里的名字；侧别词 / 集体词绝不是；追加进 entity_mentions 一次。

    Input / 输入：
        item: The attributed_to / speaker / entity_mentions keys of the LLM item.
        item：LLM 条目的 attributed_to / speaker / entity_mentions 键。

        side: The expected attributed_to side.
        side：期望的 attributed_to 侧别。

        speaker: The expected Observation.speaker.
        speaker：期望的 Observation.speaker。

        extra_mention: Whether the speaker was appended to entity_mentions (False when absent or already present).
        extra_mention：说话人是否被追加进 entity_mentions（不存在或已存在时为 False）。

    Output / 输出：
        Assertions over attributed_to, speaker and entity_mentions of the built observation.
        对构造出的观察的 attributed_to、speaker 与 entity_mentions 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    obs = _observe({**BASE, **item})
    assert obs is not None and obs.attributed_to == side and obs.speaker == speaker
    mentions = list(item.get("entity_mentions") or [])
    assert obs.entity_mentions == (mentions + [speaker] if extra_mention else mentions)


def test_speaker_maps_to_actor_id_unless_the_event_has_one():
    """atom_from_observation sets actor_id = the event's actor_id when given, else Observation.speaker, else None.
    atom_from_observation 设置 actor_id = 事件的 actor_id（若给出），否则 Observation.speaker，否则 None。

    Input / 输入：
        An observation with speaker 'Katrina', built into atoms with actor_id None / 'name-from-message'.
        speaker 为 'Katrina' 的观察，以 actor_id None / 'name-from-message' 分别构造原子。

    Output / 输出：
        Assertions: the state atom and the relation atom carry 'Katrina' without an event actor and the event's
        actor when present; a speaker-less observation yields actor_id None.
        断言：无事件行动者时 state 原子与 relation 原子都携带 'Katrina'，有事件行动者时携带它；
        没有说话人的观察 actor_id 为 None。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    obs = _observe({**BASE, "attributed_to": "Katrina", "relation": "comparison"})
    assert obs is not None and obs.speaker == "Katrina"
    assert atom_from_observation(obs, scope=SCOPE, time_anchor=STAMP_ISO).actor_id == "Katrina"
    assert relation_atom_from_observation(obs, scope=SCOPE, time_anchor=STAMP_ISO).actor_id == "Katrina"
    # The witnessed event's actor_id is caller-supplied and part of the payload hash, so it wins over the inference.
    # 见证事件的 actor_id 由调用方提供且属于载荷哈希，因此优先于推断值。
    assert atom_from_observation(obs, scope=SCOPE, time_anchor=STAMP_ISO, actor_id="name-from-message").actor_id == (
        "name-from-message"
    )
    plain = _observe(BASE)
    assert plain is not None and plain.speaker is None
    assert atom_from_observation(plain, scope=SCOPE, time_anchor=STAMP_ISO).actor_id is None


def test_speaker_reaches_the_stored_atom_and_is_filterable(store, clock):
    """Through the pipeline, a speaker becomes the atom's actor_id (a message `name` wins) and `actor_id` filters it.
    经流水线后，说话人成为原子的 actor_id（消息的 `name` 优先），并可用 `actor_id` 过滤。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions: the atom witnessed by an unnamed message carries the LLM speaker; the atom witnessed by a
        message with `name` carries that name; get_all(filters={'actor_id': ...}) returns exactly the right atom.
        断言：由无名消息见证的原子携带 LLM 说话人；由带 `name` 的消息见证的原子携带该名字；
        get_all(filters={'actor_id': ...}) 恰好返回对应原子。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    observations = [
        {**BASE, "id": "0", "event_idx": 0, "text": "Katrina likes tea.", "subject": "Katrina",
         "attributed_to": "user", "speaker": "Katrina"},
        {**BASE, "id": "1", "event_idx": 1, "text": "Shure likes coffee.", "subject": "Shure",
         "objects": ["coffee"], "attributed_to": "user", "speaker": "Shure"},
    ]  # fmt: skip
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": observations, "eps": []}]}))
    messages = [
        {"role": "user", "content": "Katrina: I like tea."},
        {"role": "user", "content": "I like coffee.", "name": "shure-device"},
    ]
    h.add(messages, timestamp=STAMP, user_id="alice", run_id="run-coerce")
    atoms = {a.text: a for a in store.get_atoms(store.valid_ids(SCOPE)["atoms"])}
    assert atoms["Katrina likes tea."].actor_id == "Katrina"
    assert atoms["Shure likes coffee."].actor_id == "shure-device"
    rows = get_all(
        store, filters={"user_id": "alice", "actor_id": "Katrina"}, top_k=10, as_of=store.now(), include_expired=False
    )
    assert [r["memory"] for r in rows] == ["Katrina likes tea."]


# ---------------------------------------------------------------------------
# 6. List fields: dict items, nested lists, comma-joined tags; malformed types still skip
# 6. 列表字段：dict 项、嵌套列表、逗号连接的标签；类型畸形仍然跳过
# ---------------------------------------------------------------------------
def test_list_fields_flatten_dict_items_and_split_tags():
    """Dict items yield their name / text, nested lists flatten, and a single 'a, b' tag string is split.
    dict 项取其 name / text，嵌套列表展平，单个 'a, b' 标签串被切分。

    Input / 输入：
        entity_mentions with a dict item, objects with a dict and a nested list, topic_tags as one string.
        带 dict 项的 entity_mentions、带 dict 与嵌套列表的 objects、单字符串的 topic_tags。

    Output / 输出：
        Assertions over the flattened lists (no repr strings such as "{'name': ...}").
        对展平后列表的断言（没有 "{'name': ...}" 之类的 repr 字符串）。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    obs = _observe(
        {
            **BASE,
            "entity_mentions": [{"name": "Katrina", "type": "person"}, "Shure", None, {"text": "Lucia"}],
            "objects": [{"name": "tea"}, ["milk", "sugar"], 3],
            "topic_tags": "coffee, preference; drinks",
        }
    )
    assert obs is not None
    assert obs.entity_mentions == ["Katrina", "Shure", "Lucia"]
    assert obs.objects == ["tea", "milk", "sugar", "3"]
    assert obs.topic_tags == ["coffee", "preference", "drinks"]


@pytest.mark.parametrize(("field", "value"), [("text", 123), ("text", {"en": "x"}), ("subject", {"name": "K"}), ("objects", {"a": 1})])
def test_malformed_types_still_skip(field, value, caplog):
    """A dict / number where a string or list belongs is a malformed type and still skips the item with a warning.
    字符串或列表位置出现 dict / 数字属于类型畸形，仍然记警告并跳过条目。

    Input / 输入：
        field: The field receiving the malformed value.
        field：收到畸形值的字段。

        value: The malformed value.
        value：畸形值。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: None is returned and one "skipped" warning names the field.
        断言：返回 None，一条 "skipped" 警告点名该字段。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    assert _observe({**BASE, field: value}) is None
    messages = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert len(messages) == 1 and field in messages[0]


# ---------------------------------------------------------------------------
# 7. EP fields: r synonyms, κ shapes, time_anchor, confidence, visibility, uses_ep_ids
# 7. EP 字段：r 同义词、κ 形状、time_anchor、置信度、可见性、uses_ep_ids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [("Negative", "negative"), ("failure", "negative"), ("success", "positive"), ("mixed", "neutral"), ("meh", "unknown"), (1, "unknown"), (["negative"], "negative"), (None, "unknown")],
)  # fmt: skip
def test_ep_reward_coerces(value, expected):
    """An out-of-vocabulary EP `r` folds to the reward member (synonyms, case) or to unknown; the EP survives.
    词表外的 EP `r` 折叠为回报成员（同义词、大小写）或 unknown；EP 得以保留。

    Input / 输入：
        value: The raw r.
        value：原始 r。

        expected: The Reward member value.
        expected：Reward 成员值。

    Output / 输出：
        Assertions: the EP is built with slots.r == expected.
        断言：EP 被构造且 slots.r == expected。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    ep = _ep({"id": "0", "event_idx": 0, "u": "fix retry", "a": "patched retry.ts", "y": "tests failed", "r": value})
    assert ep is not None and str(ep.slots.r) == expected


def test_ep_kappa_time_anchor_confidence_and_refs_coerce():
    """κ list / string → constraints, bad time_anchor → event time, null confidence → 0.7, bad visibility → event's, bare ref → 'e<i>'.
    κ 列表 / 字符串 → 约束，坏 time_anchor → 事件时间，null 置信度 → 0.7，坏可见性 → 事件的，裸引用 → 'e<i>'。

    Input / 输入：
        Three EP items with the shapes above (one κ list, one κ string with visibility inside, one plain).
        三个上述形状的 EP 条目（一个 κ 列表、一个 κ 字符串、一个普通）。

    Output / 输出：
        Assertions over kappa.constraints, visibility, time_anchor, confidence and uses_ep_ids.
        对 kappa.constraints、visibility、time_anchor、confidence 与 uses_ep_ids 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    base = {"id": "0", "event_idx": 0, "u": "fix retry", "a": "patched retry.ts", "y": ""}
    as_list = _ep({**base, "kappa": ["Do not touch the state machine"], "time_anchor": "day 1 morning"})
    assert as_list is not None and as_list.slots.kappa.constraints == ["Do not touch the state machine"]
    assert as_list.time_anchor == STAMP_ISO
    as_str = _ep({**base, "kappa": "Keep the API stable", "confidence": None})
    assert as_str is not None and as_str.slots.kappa.constraints == ["Keep the API stable"]
    assert as_str.confidence == 0.7
    plain = _ep(
        {**base, "y": "tests failed", "kappa": {"constraints": [], "visibility": "internal"}, "confidence": "high",
         "uses_ep_ids": [0, "e1", 7]},
        ep_map={"e0": "ep_real_0", "e1": "ep_real_1"},
    )  # fmt: skip
    assert plain is not None and str(plain.visibility) == "private" and plain.confidence == 0.9
    assert plain.uses_ep_ids == ["ep_real_0", "ep_real_1"]


# ---------------------------------------------------------------------------
# 8. Decisions: verdict synonyms, root shapes, id join
# 8. 决策：裁决同义词、根形状、id 连接
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [("UPDATE", "REVISE"), ("revise ", "REVISE"), ("duplicate", "NONE"), ("contradicts", "CONFLICT"), ("keep", "ADD"), (["REVISE"], "REVISE"), (None, "ADD"), ("MAYBE", "MAYBE"), ("skip", "SKIP"), ("NOOP", "NOOP"), ("ignore", "IGNORE")],
)  # fmt: skip
def test_verdict_synonyms_fold(value, expected):
    """Decision verdicts are stripped, upper-cased and synonym-folded; a truly unknown word — SKIP / NOOP / IGNORE included — stays unknown (→ ADD later).
    决策裁决去空白、大写并折叠同义词；真正未知的词——包括 SKIP / NOOP / IGNORE——保持未知（后续 → ADD）。

    Input / 输入：
        value: The raw decision value.
        value：原始 decision 值。

        expected: The folded verdict string.
        expected：折叠后的裁决字符串。

    Output / 输出：
        Assertions: `_verdict` returns expected.
        断言：`_verdict` 返回 expected。

    State effect / 状态影响：
        None.
        无。
    """
    assert extraction._verdict(value) == expected


def test_unknown_verdict_is_stored_without_revision():
    """An unknown verdict keeps the observation as a plain ADD (no revision, no target) with a warning.
    未知裁决把观察作为普通 ADD 保留（不修订、无目标）并记警告。

    Input / 输入：
        A decision {'decision': 'MAYBE', 'target_id': '0'} against one candidate ref '0'.
        对照一个候选引用 '0' 的决策 {'decision': 'MAYBE', 'target_id': '0'}。

    Output / 输出：
        Assertions: (Decision.ADD, None, 'unknown decision MAYBE').
        断言：(Decision.ADD, None, 'unknown decision MAYBE')。

    State effect / 状态影响：
        None.
        无。
    """
    result = extraction._parse_decision(
        {"decision": "MAYBE", "target_id": "0"}, obs_key="0", candidate_refs={"0": "atom_a"}, hint_ids=[], what="t"
    )
    assert result == (Decision.ADD, None, "unknown decision MAYBE")


@pytest.mark.parametrize(
    ("verdict", "events", "live", "warning"),
    [
        ("SKIP", ["ADD"], 2, "unknown decision 'SKIP'"),
        ("IGNORE", ["ADD"], 2, "unknown decision 'IGNORE'"),
        ("noop", ["ADD"], 2, "unknown decision 'NOOP'"),
        ("DUPLICATE", ["NONE"], 1, "verdict 'DUPLICATE' read as NONE"),
    ],
)
def test_skip_like_verdicts_store_the_observation_and_duplicate_is_traced(
    store, clock, verdict, events, live, warning, caplog
):
    """SKIP / IGNORE / NOOP against a valid target store the new observation as ADD with a warning; DUPLICATE is NONE, and the fold is traced with a WARNING.
    针对有效目标的 SKIP / IGNORE / NOOP 把新观察按 ADD 存储并记警告；DUPLICATE 为 NONE，且该折叠记一条 WARNING。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        verdict: The raw decision word of the second reply, targeting the first atom.
        verdict：第二次回复的原始裁决词，目标为首个原子。

        events: The expected row events of the second add().
        events：第二次 add() 期望的行事件。

        live: The expected count of valid atoms afterwards.
        live：之后期望的有效原子数。

        warning: A substring of the expected WARNING line.
        warning：期望的 WARNING 行的子串。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: row events, live atom count, and the presence of the warning line.
        断言：行事件、有效原子数、以及警告行的存在。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    first = {**BASE, "type": "preference_event"}
    second = {**BASE, "type": "preference_event", "objects": ["green tea"], "text": "User likes green tea."}
    scripted = ScriptedLLM({"responses": [
        {"observations": [first], "eps": [], "decisions": []},
        {"observations": [second], "eps": [],
         "decisions": [{"id": "0", "decision": verdict, "target_id": "0", "reason": "x"}]},
    ]})  # fmt: skip
    h = _Harness(store, clock, scripted)
    h.add("I like tea.", timestamp=STAMP, user_id="alice", run_id="run-coerce")
    res = h.add("I like green tea.", timestamp="2026-06-15T11:04:00Z", user_id="alice", run_id="run-coerce")
    assert [r["event"] for r in res.rows] == events
    assert store.count_atoms(SCOPE) == live
    assert any(warning in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("decisions", "expected_events"),
    [
        ([{"id": "0", "decision": "UPDATE", "target_id": "0"}], ["UPDATE"]),
        ({"0": {"decision": "REVISE", "target_id": "0"}}, ["UPDATE"]),
        ({"id": "0", "decision": "REVISE", "target_id": "0"}, ["UPDATE"]),
        ([{"id": "obs_0", "decision": "REVISE", "target_id": "0"}], ["UPDATE"]),
        ([{"decision": "REVISE", "target_id": "0"}], ["UPDATE"]),
        ([{"id": "0", "decision": "REVISE", "target_id": "atom_placeholder"}], ["UPDATE"]),
    ],
)
def test_decision_shapes_and_keys_are_recovered(store, clock, decisions, expected_events):
    """UPDATE verdicts, dict / single-object decision roots, 'obs_0' ids, id-less (positional) decisions and real atom_id targets still revise.
    UPDATE 裁决、dict / 单对象形式的 decisions 根、'obs_0' 形式的 id、无 id（按位置）的决策与真实 atom_id 目标仍然修订。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        decisions: The scripted decisions root of the second reply ('atom_placeholder' is replaced by the real id).
        decisions：第二次回复的脚本化 decisions 根（'atom_placeholder' 替换为真实 id）。

        expected_events: The row events of the second add().
        expected_events：第二次 add() 的行事件。

    Output / 输出：
        Assertions: the second add() revises the first atom (UPDATE, same lineage id, one valid atom).
        断言：第二次 add() 修订首个原子（UPDATE、同一谱系 id、一个有效原子）。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    latte = {**BASE, "type": "preference_event", "objects": ["oat-milk lattes"], "text": "User likes oat-milk lattes."}
    black = {**BASE, "type": "preference_event", "objects": ["black coffee"], "relation": "single_fact",
             "text": "User likes black coffee."}  # fmt: skip
    seed = {"observations": [latte], "eps": [], "decisions": [{"id": "0", "decision": "ADD", "target_id": None}]}
    scripted = ScriptedLLM({"responses": [seed, {"observations": [black], "eps": [], "decisions": decisions}]})
    h = _Harness(store, clock, scripted)
    first = h.add("I like oat-milk lattes.", timestamp="2026-04-02T08:00:00Z", user_id="alice", run_id="run-coerce")
    if isinstance(decisions, list) and decisions[0].get("target_id") == "atom_placeholder":
        decisions[0]["target_id"] = first.rows[0]["atom_id"]
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z", user_id="alice",
                   run_id="run-coerce")  # fmt: skip
    assert [r["event"] for r in second.rows] == expected_events
    assert second.rows[0]["id"] == first.rows[0]["id"] and store.count_atoms(SCOPE) == 1


def test_eps_root_single_object_is_read_as_a_list(store, clock, caplog):
    """An `eps` root that is one EP object (not a list) still yields that EP, with one warning about the shape.
    `eps` 根是单个 EP 对象（而非列表）时仍然得到该 EP，并记一条关于形状的警告。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: one EP persists; the warning names the eps root; nothing is "skipped".
        断言：一个 EP 持久化；警告点名 eps 根；没有条目被"跳过"。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    ep = {"id": "0", "event_idx": 0, "u": "the webhook kept timing out", "a": "patched retry.ts",
          "y": "billing_e2e failed", "r": "failed"}  # fmt: skip
    h = _Harness(store, clock, ScriptedLLM({"responses": [{"observations": [], "eps": ep, "decisions": []}]}))
    res = h.add(
        "I patched retry.ts to add backoff and then billing_e2e failed.", timestamp=STAMP, user_id="alice",
        run_id="run-coerce",
    )  # fmt: skip
    assert len(res.ep_ids) == 1 and str(store.get_ep(res.ep_ids[0]).slots.r) == "negative"
    messages = [r.getMessage() for r in caplog.records]
    assert any("eps root is a single item" in m for m in messages) and not any("skipped" in m for m in messages)


# ---------------------------------------------------------------------------
# 9. The contract quotes the observation OUTPUT FORMAT verbatim (speaker key included)
# 9. 契约逐字引用观察的 OUTPUT FORMAT（含 speaker 键）
# ---------------------------------------------------------------------------
def test_contract_quotes_the_observation_output_format_verbatim():
    """CONTRACT §6.3 (EN and ZH) carries the same observation_extraction JSON block as the prompt's OUTPUT FORMAT, `speaker` included.
    CONTRACT §6.3（中英）携带与提示词 OUTPUT FORMAT 相同的 observation_extraction JSON 块，含 `speaker`。

    Input / 输入：
        none (reads docs/design/CONTRACT.md, CONTRACT.zh.md and configs.prompts).
        无（读取 docs/design/CONTRACT.md、CONTRACT.zh.md 与 configs.prompts）。

    Output / 输出：
        Assertions: the parsed JSON of both contract blocks equals the prompt's; the block spells `"speaker": null`
        and the field rules define `speaker`.
        断言：两份契约块解析出的 JSON 与提示词的相同；块中写有 `"speaker": null`，字段规则定义了 `speaker`。

    State effect / 状态影响：
        None.
        无。
    """
    prompt_block = prompts.OBSERVATION_EXTRACTION_PROMPT.split("# OUTPUT FORMAT")[1].split("```")[1]
    prompt_block = prompt_block.removeprefix("json")
    root = pathlib.Path(__file__).resolve().parents[1] / "docs" / "design"
    for name in ("CONTRACT.md", "CONTRACT.zh.md"):
        section = (root / name).read_text().split("**`HMS_TASK: observation_extraction`**")[1]
        block = section.split("```")[1]
        assert json.loads(block) == json.loads(prompt_block)
        assert '"speaker": null' in block
        assert "`speaker`" in section.split("```")[2].splitlines()[1]
