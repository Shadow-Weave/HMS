"""The verdict similarity guard (ingest.verdict_min_similarity) and the per-observation DECISIONS rule of the compact prompt.
裁决相似度守卫（ingest.verdict_min_similarity）与紧凑提示词的"每条新观察一个决策"规则。
"""

# Purpose (EN): Pin the 2026-09-11 fix for index-paired NONE verdicts (CONTRACT changelog rows 103 / 104): a
# scripted compact reply whose three NONE decisions each name a real candidate keeps only the true restatement as
# NONE and stores the other two as ADD (DEBUG log + IngestResult.downgraded); the same under the full contract;
# short new facts that share only their head (subject + verb) with the target — "User has dogs." after "User has
# cats." — are stored, not discarded (row 104); verdicts across different subjects are downgraded while a same-subject
# CONFLICT passes; threshold 0.0 restores the old behaviour; every chat fixture through the rule LLM is row- and
# table-identical with the guard on and off; the canonical supersession cases (oat-milk lattes → black coffee,
# Paris → Berlin) still revise; verdict_similarity is containment over stemmed content tokens after the head
# (subject, shared predicate) is removed; the compact DECISIONS section and OUTPUT FORMAT are stated per observation
# with a target id that differs from the observation id.
# 目的（中）：钉住 2026-09-11 对按下标配对的 NONE 裁决的修复（CONTRACT 变更记录第 103 / 104 行）：脚本化紧凑回复的
# 三条 NONE 决策各自点名一个真实候选时，只有真正的复述保持 NONE，另外两条按 ADD 存入（DEBUG 日志 +
# IngestResult.downgraded）；完整契约下相同；只与目标共享头部（主语 + 动词）的短新事实——"User has cats." 之后的
# "User has dogs."——被存入而非丢弃（第 104 行）；跨主语的裁决被降级而同主语的 CONFLICT 通过；门槛 0.0 恢复旧行为；
# 每个聊天夹具经规则 LLM 在守卫开 / 关下逐行、逐表相同；经典取代用例（燕麦拿铁 → 黑咖啡、巴黎 → 柏林）仍然修订；
# verdict_similarity 是去掉头部（主语、共同谓词）后词干化内容词元的包含度；紧凑提示词的 DECISIONS 段与 OUTPUT FORMAT
# 按观察陈述，目标 id 与观察 id 不同。

# Input (EN): Hand-written compact / full replies, the chat_* fixtures, ScriptedLLM / RuleLLM, the conftest helpers.
# 输入（中）：手写的紧凑 / 完整回复、chat_* 夹具、ScriptedLLM / RuleLLM、conftest 辅助。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import logging
from typing import Any

import pytest

from hms_elite.configs import prompts
from hms_elite.configs.base import HMSConfig, IngestConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.exceptions import ConfigError
from hms_elite.index.memory import MemoryIndex
from hms_elite.ingest.extraction import statement_tokens, verdict_similarity
from hms_elite.ingest.pipeline import Ingestor, IngestResult
from hms_elite.llms.rule import RuleLLM
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import Scope
from tests.conftest import FrozenClock, SequentialIds
from tests.test_extraction_compact import CHAT_FIXTURES, IDENTICAL_TABLES, _facade, _rows, _tables

PIPELINE_LOGGER = "hms_elite.ingest.pipeline"
STAMP = "2026-06-15T10:04:00Z"
LATER = "2026-06-15T11:04:00Z"
SCOPE = Scope(user_id="alice", run_id="run-guard")
# Three seeded statements (one add) and three new ones (a second add): only the first new statement restates a seed.
# 三条种子陈述（第一次 add）与三条新陈述（第二次 add）：只有第一条新陈述是种子的复述。
SEED_TEXTS = (
    "Jake sets the table for dinner.",
    "Katrina bought peonies shipped from Yunnan.",
    "Tasha says a full set costs 25 yuan.",
)
NEW_TEXTS = (
    "Jake set the dinner table.",
    "Katrina pressed the flowers with desiccants.",
    "Tasha will invite one person for now.",
)
SPEAKERS = ("Jake", "Katrina", "Tasha")
SEED_MESSAGES = [
    {"role": "user", "content": "Jake: let me set the table for dinner."},
    {"role": "user", "content": "Katrina: I bought peonies, shipped from Yunnan."},
    {"role": "user", "content": "Tasha: a full set is 25 yuan."},
]
NEW_MESSAGES = [
    {"role": "user", "content": "Jake: I set the dinner table."},
    {"role": "user", "content": "Katrina: I pressed the flowers with desiccants."},
    {"role": "user", "content": "Tasha: I will invite one person for now."},
]
# Short first-person facts that share only subject + verb with their seed (the row-104 defect: with the subject
# counted, each pair scored exactly 0.5 and the NONE was honoured); the last pair is a genuine restatement.
# 只与种子共享主语 + 动词的第一人称短事实（第 104 行的缺陷：主语计入时每对恰得 0.5，NONE 被采纳）；最后一对是真正的
# 复述。
SHORT_SEED_TEXTS = ("User has cats.", "User is tired today.", "User likes hiking on weekends.", "User works at Acme.")
SHORT_NEW_TEXTS = ("User has dogs.", "User is happy today.", "User likes reading at night.", "User is working at Acme.")
SHORT_SEED_MESSAGES = [{"role": "user", "content": c} for c in ("I have cats.", "I am tired today.", "I like hiking on weekends.", "I work at Acme.")]  # fmt: skip
SHORT_NEW_MESSAGES = [{"role": "user", "content": c} for c in ("I have dogs.", "I am happy today.", "I like reading at night.", "I am working at Acme.")]  # fmt: skip
# The canonical supersession pair of the prompts and of §19.1 S1 (scripted, full contract).
# 提示词与 §19.1 S1 的经典取代对（脚本化，完整契约）。
LATTE = {
    "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
    "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
    "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02",
}  # fmt: skip
BLACK = {
    "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
    "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
    "date_observed": "2026-04-20", "supersedes_hint_ids": ["0"],
}  # fmt: skip


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
class _Harness:
    """Ingestor over a fresh in-memory store, memory index and hash embedder in one output / threshold.
    绑定全新内存 store、内存索引与哈希嵌入器、给定输出与门槛的 Ingestor。
    """

    def __init__(self, llm: Any, *, output: str, threshold: float | None = None, mode: str = "single") -> None:
        """Wire the collaborators; `threshold` None keeps the IngestConfig default.
        接线协作对象；`threshold` 为 None 时保留 IngestConfig 默认值。
        """
        self.clock = FrozenClock()
        self.store = StateStore(":memory:", now=self.clock.now, id_factory=SequentialIds())
        self.llm = llm
        self.embedder = HashEmbedder({"embedding_dims": 64})
        self.index = MemoryIndex(dims=self.embedder.dims)
        ingest: dict[str, Any] = {"extraction_output": output}
        if threshold is not None:
            ingest["verdict_min_similarity"] = threshold
        self.config = HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, ingest=ingest)
        self.ingestor = Ingestor(
            self.store, self.index, self.embedder, llm, self.config, extraction_mode=mode, extraction_output=output
        )

    def add(self, messages: Any, *, timestamp: str) -> IngestResult:
        """One inferred add() (the clock ticks a minute first).
        一次推断式 add()（时钟先前进一分钟）。
        """
        self.clock.tick(seconds=60)
        return self.ingestor.ingest_messages(
            coerce_messages(messages), scope=SCOPE, user_meta={}, infer=True, memory_type=None, prompt=None,
            event_time=timestamp, valid_to=None,
        )  # fmt: skip


def _item(index: int, text: str, *, output: str, speaker: str | None = "") -> dict:
    """One observation item of the given contract for a fact; `speaker` "" means SPEAKERS[index], None a plain
    user statement (subject `user` under the full contract).
    给定契约下一条事实的观察条目；`speaker` 为 "" 表示 SPEAKERS[index]，None 表示普通用户陈述（完整契约下主语
    `user`）。
    """
    who = SPEAKERS[index] if speaker == "" else speaker
    item = {"id": str(index), "event_idx": index, "text": text, "type": "fact", "speaker": who}
    if output == "compact":
        return item
    verb, rest = text.split(" ", 1)[1].split(" ", 1)
    return {
        **item, "subject": who or "user", "predicate": verb, "objects": [rest.rstrip(".")], "relation": "single_fact",
        "date_observed": "2026-06-15", "confidence": 0.9, "attributed_to": "user", "sentiment": "neutral",
        "topic_tags": [], "entity_mentions": [who] if who else [],
    }  # fmt: skip


def _seed_reply(output: str, texts: tuple[str, ...] = SEED_TEXTS, *, speaker: str | None = "") -> dict:
    """The first reply: the seed observations, no candidates shown, so `decisions` is empty.
    第一条回复：种子观察，未呈现候选，故 `decisions` 为空。
    """
    return {
        "observations": [_item(i, t, output=output, speaker=speaker) for i, t in enumerate(texts)],
        "eps": [], "decisions": [],
    }  # fmt: skip


def _index_paired_reply(
    output: str, seen: dict, seeds: tuple[str, ...] = SEED_TEXTS, news: tuple[str, ...] = NEW_TEXTS,
    *, speaker: str | None = "",
) -> Any:
    """A callable reply that answers NONE for every new observation, each naming the candidate whose text is the
    seed at the same position — the index-pairing artefact, but with real candidate ids, so only the guard can
    tell a restatement from a new fact.
    可调用回复：对每条新观察都回答 NONE，各自点名文本等于同位置种子的候选——正是按下标配对的伪影，但候选 id 是真实
    的，因此只有守卫能把复述与新事实区分开。
    """

    def reply(messages: list[dict]) -> dict:
        user = next(m["content"] for m in messages if m["role"] == "user")
        candidates = prompts.parse_sections(user).get("Candidate Existing Memories") or []
        by_text = {str(c.get("text")): str(c.get("id")) for c in candidates if isinstance(c, dict)}
        seen["n"] = len(candidates)
        return {
            "observations": [_item(i, t, output=output, speaker=speaker) for i, t in enumerate(news)],
            "eps": [],
            "decisions": [
                {"id": str(i), "decision": "NONE", "target_id": by_text[seeds[i]], "reason": "duplicate"}
                for i in range(len(news))
            ],
        }

    return reply


def _scripted_reply(observations: list[dict], decisions: list[dict], seen: dict) -> Any:
    """A callable full-contract reply whose decisions name their targets by candidate TEXT (`target_text`), resolved
    to the candidate ids the prompt actually showed.
    可调用的完整契约回复：决策用候选文本（`target_text`）点名目标，解析为提示词实际呈现的候选 id。
    """

    def reply(messages: list[dict]) -> dict:
        user = next(m["content"] for m in messages if m["role"] == "user")
        candidates = prompts.parse_sections(user).get("Candidate Existing Memories") or []
        by_text = {str(c.get("text")): str(c.get("id")) for c in candidates if isinstance(c, dict)}
        seen["n"] = len(candidates)
        return {
            "observations": observations,
            "eps": [],
            "decisions": [
                {"id": d["id"], "decision": d["decision"], "target_id": by_text[d["target_text"]], "reason": "scripted"}
                for d in decisions
            ],
        }

    return reply


def _full(index: int, text: str, subject: str, predicate: str, objects: list[str]) -> dict:
    """One full-contract observation item with an explicit head.
    一条带显式头部的完整契约观察条目。
    """
    return {
        "id": str(index), "event_idx": index, "text": text, "type": "fact", "speaker": None, "subject": subject,
        "predicate": predicate, "objects": objects, "relation": "single_fact", "date_observed": "2026-06-15",
        "confidence": 0.9, "attributed_to": "user", "sentiment": "neutral", "topic_tags": [], "entity_mentions": [],
    }  # fmt: skip


def _downgrade_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The DEBUG lines the guard wrote.
    守卫写下的 DEBUG 行。
    """
    return [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER and "downgraded to ADD" in r.getMessage()]


# ---------------------------------------------------------------------------
# (a) Index-paired NONE verdicts keep only the restatement — both contracts
# (a) 按下标配对的 NONE 裁决只保留复述——两种契约
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("output", ["compact", "full"])
def test_index_paired_none_verdicts_keep_only_the_restatement(caplog, output):
    """Three NONE decisions that each name a real candidate yield one honoured NONE (the restatement) and two ADD rows with a DEBUG line and an IngestResult.downgraded entry each, under the compact and the full contract alike.
    三条各自点名真实候选的 NONE 决策，得到一条被采纳的 NONE（复述）与两条 ADD 行，每条 ADD 附一行 DEBUG 日志与一条 IngestResult.downgraded 条目，紧凑与完整契约下相同。

    Input / 输入：
        caplog: pytest log capture (DEBUG on the pipeline logger).
        caplog：pytest 日志捕获（流水线 logger 的 DEBUG）。

        output: 'compact' | 'full' — the extraction contract of the scripted replies.
        output：'compact' | 'full'——脚本化回复所用的抽取契约。

    Output / 输出：
        Assertions on the row events, the NONE target, `downgraded`, the atom count and the DEBUG lines.
        对行事件、NONE 目标、`downgraded`、原子数与 DEBUG 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seen: dict = {}
    h = _Harness(ScriptedLLM({"responses": [_seed_reply(output), _index_paired_reply(output, seen)]}), output=output)
    first = h.add(SEED_MESSAGES, timestamp=STAMP)
    assert [r["event"] for r in first.rows] == ["ADD"] * 3 and first.downgraded == []
    second = h.add(NEW_MESSAGES, timestamp=LATER)
    assert seen["n"] == 3, "all three seeds were shown as candidates"
    assert [r["event"] for r in second.rows] == ["NONE", "ADD", "ADD"]
    assert second.rows[0]["atom_id"] == first.rows[0]["atom_id"]
    assert [r["memory"] for r in second.rows[1:]] == list(NEW_TEXTS[1:])
    assert h.store.count_atoms(SCOPE) == 5
    assert [(d["kind"], d["verdict"], d["target_id"], d["text"]) for d in second.downgraded] == [
        ("verdict", "NONE", first.rows[1]["atom_id"], NEW_TEXTS[1]),
        ("verdict", "NONE", first.rows[2]["atom_id"], NEW_TEXTS[2]),
    ]
    assert all(0.0 <= d["similarity"] < IngestConfig().verdict_min_similarity for d in second.downgraded)
    assert all(d["subjects_differ"] is False for d in second.downgraded)
    lines = _downgrade_records(caplog)
    assert len(lines) == 2 and all(line.startswith("verdict NONE on ") for line in lines)
    assert not any("subjects differ" in line or "one shared token" in line for line in lines)
    # The honoured NONE is a support witness on the seed, exactly as before the guard.
    # 被采纳的 NONE 与守卫之前一样，是种子上的一个支持见证。
    assert len(h.store.atom_anchor_ids(first.rows[0]["atom_id"])) == 2


@pytest.mark.parametrize("output", ["compact", "full"])
def test_short_new_facts_sharing_only_the_head_are_stored(caplog, output):
    """Four NONE decisions on short user facts that share only subject + verb with their seed ("User has dogs." after "User has cats.") store three ADD rows — similarity 0.0, 0.5 (one shared token, `today`, rejected by the single-token rule) and 0.0, a downgrade entry and a DEBUG line each — and keep only the genuine restatement ("User is working at Acme." after "User works at Acme.") as NONE, under both contracts; with the subject counted all three pairs scored exactly 0.5 and the facts were discarded.
    四条对短用户事实的 NONE 决策——只与种子共享主语 + 动词（"User has cats." 之后的 "User has dogs."）——存下三条 ADD 行（相似度 0.0、0.5——只共享一个词元 `today`，被单词元规则拒绝——与 0.0，各一条降级条目与一行 DEBUG），只把真正的复述（"User works at Acme." 之后的 "User is working at Acme."）保留为 NONE，两种契约下相同；主语计入时三对都恰得 0.5，事实被丢弃。

    Input / 输入：
        caplog: pytest log capture (DEBUG on the pipeline logger).
        caplog：pytest 日志捕获（流水线 logger 的 DEBUG）。

        output: 'compact' | 'full' — the extraction contract of the scripted replies.
        output：'compact' | 'full'——脚本化回复所用的抽取契约。

    Output / 输出：
        Assertions on the row events and texts, `downgraded`, the atom count and the DEBUG lines.
        对行事件与文本、`downgraded`、原子数与 DEBUG 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seen: dict = {}
    llm = ScriptedLLM({"responses": [
        _seed_reply(output, SHORT_SEED_TEXTS, speaker=None),
        _index_paired_reply(output, seen, SHORT_SEED_TEXTS, SHORT_NEW_TEXTS, speaker=None),
    ]})  # fmt: skip
    h = _Harness(llm, output=output)
    first = h.add(SHORT_SEED_MESSAGES, timestamp=STAMP)
    assert [r["event"] for r in first.rows] == ["ADD"] * 4 and first.downgraded == []
    second = h.add(SHORT_NEW_MESSAGES, timestamp=LATER)
    assert seen["n"] == 4, "all four seeds were shown as candidates"
    assert [r["event"] for r in second.rows] == ["ADD", "ADD", "ADD", "NONE"]
    assert [r["memory"] for r in second.rows[:3]] == list(SHORT_NEW_TEXTS[:3])
    assert second.rows[3]["atom_id"] == first.rows[3]["atom_id"]
    assert h.store.count_atoms(SCOPE) == 7
    similarities, shared = (0.0, 0.5, 0.0), (0, 1, 0)
    assert [(d["verdict"], d["target_id"], d["similarity"], d["shared"], d["subjects_differ"], d["text"]) for d in second.downgraded] == [
        ("NONE", first.rows[i]["atom_id"], similarities[i], shared[i], False, SHORT_NEW_TEXTS[i]) for i in range(3)
    ]
    lines = _downgrade_records(caplog)
    assert len(lines) == 3 and [", one shared token)" in line for line in lines] == [False, True, False]
    for new, old, verb, expected in zip(SHORT_NEW_TEXTS[:3], SHORT_SEED_TEXTS[:3], ("has", "is", "likes"), similarities):
        assert verdict_similarity(new, old, new_subject="user", old_subject="user", new_predicate=verb, old_predicate=verb) == expected
    assert verdict_similarity(SHORT_NEW_TEXTS[3], SHORT_SEED_TEXTS[3], new_subject="user", old_subject="user",
                              new_predicate="is", old_predicate="works") == 1.0  # fmt: skip


def test_cross_subject_verdicts_are_downgraded_and_same_subject_conflict_passes(caplog):
    """A NONE whose target is the same statement about ANOTHER subject (Lucia / Tasha going to Zhengzhou, containment 1.0), a REVISE across subjects (Katrina's 4:35 against Jake's 5 o'clock) and a CONFLICT across subjects (Bob / Alice living in Paris) all become ADD with `subjects_differ` and a DEBUG line saying so, while a same-subject CONFLICT on different values (user lives in Berlin / Paris, no shared token) is honoured.
    目标是关于另一主语的同一陈述的 NONE（Lucia / Tasha 去郑州，包含度 1.0）、跨主语的 REVISE（Katrina 的 4:35 对 Jake 的 5 点）与跨主语的 CONFLICT（Bob / Alice 住在巴黎）都改为 ADD，带 `subjects_differ` 与注明此事的 DEBUG 行；而同主语、不同取值的 CONFLICT（用户住在柏林 / 巴黎，无共同词元）被采纳。

    Input / 输入：
        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the rows (events, `conflicts_with`), `downgraded`, the atom count and the DEBUG lines.
        对行（事件、`conflicts_with`）、`downgraded`、原子数与 DEBUG 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seeds = [
        _full(0, "Tasha is going to Zhengzhou.", "Tasha", "going_to", ["Zhengzhou"]),
        _full(1, "Jake said they will eat at 5.", "Jake", "said", ["they will eat at 5"]),
        _full(2, "User lives in Paris.", "user", "lives_in", ["Paris"]),
        _full(3, "Alice lives in Paris.", "Alice", "lives_in", ["Paris"]),
    ]
    news = [
        _full(0, "Lucia is going to Zhengzhou.", "Lucia", "going_to", ["Zhengzhou"]),
        _full(1, "Katrina said the event starts at 4:35.", "Katrina", "said", ["the event starts at 4:35"]),
        _full(2, "User lives in Berlin.", "user", "lives_in", ["Berlin"]),
        _full(3, "Bob lives in Paris.", "Bob", "lives_in", ["Paris"]),
    ]
    decisions = [
        {"id": "0", "decision": "NONE", "target_text": seeds[0]["text"]},
        {"id": "1", "decision": "REVISE", "target_text": seeds[1]["text"]},
        {"id": "2", "decision": "CONFLICT", "target_text": seeds[2]["text"]},
        {"id": "3", "decision": "CONFLICT", "target_text": seeds[3]["text"]},
    ]
    seen: dict = {}
    seed_reply = {"observations": seeds, "eps": [], "decisions": []}
    h = _Harness(ScriptedLLM({"responses": [seed_reply, _scripted_reply(news, decisions, seen)]}), output="full")
    first = h.add([{"role": "user", "content": o["text"]} for o in seeds], timestamp=STAMP)
    assert [r["event"] for r in first.rows] == ["ADD"] * 4
    second = h.add([{"role": "user", "content": o["text"]} for o in news], timestamp=LATER)
    assert seen["n"] == 4
    assert [r["event"] for r in second.rows] == ["ADD"] * 4
    assert second.rows[2]["conflicts_with"] == [first.rows[2]["id"]] and "conflicts_with" not in second.rows[3]
    assert h.store.count_atoms(SCOPE) == 8
    assert [(d["verdict"], d["target_id"], d["similarity"], d["subjects_differ"]) for d in second.downgraded] == [
        ("NONE", first.rows[0]["atom_id"], 1.0, True),
        ("REVISE", first.rows[1]["atom_id"], 0.0, True),
        ("CONFLICT", first.rows[3]["atom_id"], 1.0, True),
    ]
    lines = _downgrade_records(caplog)
    assert len(lines) == 3 and all(", subjects differ)" in line for line in lines)
    assert verdict_similarity("User lives in Berlin.", "User lives in Paris.", new_subject="user", old_subject="user",
                              new_predicate="lives_in", old_predicate="lives_in") == 0.0  # fmt: skip


# ---------------------------------------------------------------------------
# (b) Threshold 0.0 restores the old behaviour
# (b) 门槛 0.0 恢复旧行为
# ---------------------------------------------------------------------------
def test_threshold_zero_disables_the_guard(caplog):
    """With verdict_min_similarity 0.0 every index-paired NONE is honoured (three NONE rows, three atoms, no downgrade, no DEBUG line); the field defaults to 0.5 and rejects values outside [0, 1].
    verdict_min_similarity 为 0.0 时每条按下标配对的 NONE 都被采纳（三条 NONE 行、三个原子、无降级、无 DEBUG 行）；该字段默认 0.5 且拒绝 [0, 1] 之外的值。

    Input / 输入：
        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the rows, the atom count, `downgraded`, the log, and the config validation.
        对行、原子数、`downgraded`、日志与配置校验的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seen: dict = {}
    llm = ScriptedLLM({"responses": [_seed_reply("compact"), _index_paired_reply("compact", seen)]})
    h = _Harness(llm, output="compact", threshold=0.0)
    first = h.add(SEED_MESSAGES, timestamp=STAMP)
    second = h.add(NEW_MESSAGES, timestamp=LATER)
    assert [r["event"] for r in second.rows] == ["NONE"] * 3
    assert [r["atom_id"] for r in second.rows] == [r["atom_id"] for r in first.rows]
    assert h.store.count_atoms(SCOPE) == 3 and second.downgraded == [] and _downgrade_records(caplog) == []
    assert IngestConfig().verdict_min_similarity == 0.5
    for bad in (-0.1, 1.5):
        with pytest.raises(ConfigError) as exc:
            HMSConfig.from_dict({"ingest": {"verdict_min_similarity": bad}, "store": {"path": ":memory:"}})
        assert exc.value.error_code == "CONFIG_001"


# ---------------------------------------------------------------------------
# (c) The full-contract fixtures are unchanged; the canonical revisions survive
# (c) 完整契约夹具不变；经典修订仍然成立
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CHAT_FIXTURES)
def test_rule_llm_fixtures_are_identical_with_the_guard_on_and_off(name):
    """On every chat fixture a full-contract facade with the default guard and one with the guard disabled store the same rows (events, texts, revises) and the same axis / history / graph tables — the rule LLM's NONE / REVISE / CONFLICT verdicts all pass the bar.
    在每个聊天夹具上，默认守卫的完整契约门面与关闭守卫的门面存下相同的行（事件、文本、revises）与相同的轴 / 历史 / 图表——规则 LLM 的 NONE / REVISE / CONFLICT 裁决全部过线。

    Input / 输入：
        name: A tests/fixtures/events/chat_*.json stem.
        name：tests/fixtures/events/chat_*.json 的文件名。

    Output / 输出：
        Row and table equalities between the two facades.
        两个门面之间的行与表相等断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    guarded, _c1 = _facade("full")
    unguarded, _c2 = _facade("full", verdict_min_similarity=0.0)
    assert guarded.config.ingest.verdict_min_similarity == 0.5 and unguarded.config.ingest.verdict_min_similarity == 0.0
    keys = ("memory", "event", "revises", "conflicts_with", "anchor_ids", "atom_type")
    guarded_rows, unguarded_rows = _rows(guarded, name), _rows(unguarded, name)
    assert [tuple(r.get(k) for k in keys) for r in guarded_rows] == [tuple(r.get(k) for k in keys) for r in unguarded_rows]
    guarded_tables, unguarded_tables = _tables(guarded), _tables(unguarded)
    for table in ("atom_triples", *IDENTICAL_TABLES):
        assert guarded_tables[table] == unguarded_tables[table], table
    if name == "chat_preference":
        assert [r["event"] for r in guarded_rows] == ["ADD", "UPDATE"]
    if name == "chat_conflict":
        assert [row["event"] for row in guarded_tables["history"]] == ["ADD", "ADD", "CONFLICT"]
    guarded.close()
    unguarded.close()


def test_canonical_revisions_survive_the_guard(caplog):
    """The scripted oat-milk lattes → black coffee REVISE (containment 0.0 once the shared head `user likes` is removed; corroborated by its supersedes hint and by the equal subject + predicate) and the rule LLM's Paris → Berlin relocation (containment 0.0, corroborated by the equal subject + predicate `lives_in`) both still yield UPDATE without a downgrade.
    脚本化的燕麦拿铁 → 黑咖啡 REVISE（去掉共同头部 `user likes` 后包含度 0.0；被其取代提示与相同的主语 + 谓词佐证）与规则 LLM 的巴黎 → 柏林搬家（包含度 0.0，被相同的主语 + 谓词 `lives_in` 佐证）都仍得到 UPDATE，无降级。

    Input / 输入：
        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        UPDATE rows, empty `downgraded`, the similarity values, no DEBUG line.
        UPDATE 行、空的 `downgraded`、相似度取值、无 DEBUG 行。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seed = {"observations": [LATTE], "eps": [], "decisions": [{"id": "0", "decision": "ADD", "target_id": None}]}
    revise = {"observations": [BLACK], "eps": [], "decisions": [{"id": "0", "decision": "REVISE", "target_id": "0"}]}
    h = _Harness(ScriptedLLM({"responses": [seed, revise]}), output="full")
    first = h.add("I like oat-milk lattes in the morning.", timestamp="2026-04-02T08:00:00Z")
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z")
    assert second.rows[0]["event"] == "UPDATE" and second.rows[0]["revises"] == first.rows[0]["atom_id"]
    assert second.downgraded == []
    assert verdict_similarity(BLACK["text"], LATTE["text"], new_subject="user", old_subject="user") == pytest.approx(1 / 3)
    assert verdict_similarity(BLACK["text"], LATTE["text"], new_subject="user", old_subject="user",
                              new_predicate="likes", old_predicate="likes") == 0.0  # fmt: skip
    assert verdict_similarity(BLACK["text"], LATTE["text"]) == pytest.approx(1 / 3)
    # A re-valued statement shares nothing beyond its head, so only the hint or the equal subject + predicate carry
    # a REVISE; the lexical bar is for NONE (and a REVISE with neither corroboration).
    # 改了取值的陈述除头部外没有共同词元，因此只有提示或相同的主语 + 谓词能让 REVISE 成立；词法门槛是给 NONE
    # （以及两种佐证都没有的 REVISE）的。
    relocation = _Harness(RuleLLM(), output="full")
    r1 = relocation.add("I live in Paris.", timestamp="2026-04-02T08:00:00Z")
    r2 = relocation.add("I moved to Berlin.", timestamp="2026-04-20T08:00:00Z")
    assert r1.rows[0]["hms"]["predicate"] == "lives_in" and r2.rows[0]["hms"]["predicate"] == "lives_in"
    assert r2.rows[0]["event"] == "UPDATE" and r2.downgraded == []
    assert verdict_similarity(r2.rows[0]["memory"], r1.rows[0]["memory"], new_subject="user", old_subject="user",
                              new_predicate="lives_in", old_predicate="lives_in") == 0.0  # fmt: skip
    assert _downgrade_records(caplog) == []


# ---------------------------------------------------------------------------
# (d) The compact DECISIONS rule and the similarity measure
# (d) 紧凑提示词的 DECISIONS 规则与相似度度量
# ---------------------------------------------------------------------------
def test_compact_prompt_states_decisions_per_observation():
    """The compact DECISIONS section is stated per NEW observation, forbids per-candidate and positional pairing, reserves NONE for a restatement with an example where a similar candidate still yields ADD, drops the old 'NONE = duplicate of the target' wording, shows a target id different from the observation id in both the EXAMPLES and the OUTPUT FORMAT block, keeps the observation keys, and stays under 0.75 of the full prompt (whose bytes are pinned elsewhere).
    紧凑提示词的 DECISIONS 段按每条新观察陈述、禁止按候选与按位置配对、把 NONE 留给复述并给出相似候选仍为 ADD 的示例、去掉旧的 'NONE = duplicate of the target' 措辞、EXAMPLES 与 OUTPUT FORMAT 块里的目标 id 都与观察 id 不同、观察键不变，且长度低于完整提示词的 0.75（其字节在别处钉住）。

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
    compact, full = prompts.MEMORY_EXTRACTION_COMPACT_PROMPT, prompts.MEMORY_EXTRACTION_PROMPT
    # Line wrapping is not part of the contract: the section is compared with whitespace collapsed.
    # 换行不属于契约：比较时折叠空白。
    decisions = " ".join(compact.split("- Decisions:", 1)[1].split("# EXAMPLES", 1)[0].split())
    assert "exactly one per NEW observation `id`, never one per candidate, never by position" in decisions
    assert "NONE ONLY when a candidate states the SAME fact about the SAME subject with the SAME value (a restatement)" in decisions
    assert "a new detail, person, time or a different value is ADD" in decisions
    assert 'Bad: NONE for "Jake sets the table." against "Jake brought plates to the kitchen."' in decisions
    assert 'Good: ADD for it; NONE only against "Jake set the dinner table."' in decisions
    assert "REVISE = the new statement supersedes the target; CONFLICT = incompatible" in decisions
    assert "temporal facts and habits are always ADD" in decisions
    assert "NONE = duplicate of the target" not in compact and "NONE = duplicate of the target" in full
    assert "one decision per candidate" not in compact
    example = compact.split("# EXAMPLES", 1)[1].split("# OUTPUT FORMAT", 1)[0]
    assert '"decisions": [{"id": "0", "decision": "REVISE", "target_id": "1"' in example
    assert '"supersedes_hint_ids": ["1"]' in example and example.count('"id": "0"') >= 2 and '"id": "1"' in example
    # The OUTPUT FORMAT block pairs observation "0" with target "1" too: no observation-id / candidate-id coincidence
    # is exemplified anywhere in the compact prompt.
    # OUTPUT FORMAT 块同样把观察 "0" 与目标 "1" 配对：紧凑提示词里任何地方都不再示范观察 id 与候选 id 的重合。
    shape = prompts.output_format_section(compact)
    assert '{"id": "0", "decision": "REVISE", "target_id": "1", "reason": "cue \'switched to\'"}' in shape
    assert '"supersedes_hint_ids": ["1"]' in shape and '"target_id": "0"' not in compact
    assert '"target_id": "0"' in prompts.output_format_section(full), "the full prompt is pinned byte-identical"
    assert prompts.COMPACT_OBSERVATION_KEYS == (
        "id", "event_idx", "text", "type", "speaker", "supersedes_hint_ids", "when", "valid_to", "entities",
    )  # fmt: skip
    assert len(compact) < 0.75 * len(full)
    # The rule LLM's compact decisions keep their shape: one dict per observation id with the four keys.
    # 规则 LLM 的紧凑决策形状不变：每个观察 id 一个带四个键的字典。
    user = prompts.build_memory_extraction_prompt(
        existing_memories=[{"id": "0", "text": "User likes oat-milk lattes in the morning."}],
        new_messages=[{"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.", "timestamp": STAMP}],
        last_messages=[], observation_date="2026-06-15", current_date="2026-06-15", custom_instructions=None,
        events=[], known_constraints=[], recent_eps=[],
        candidates=[{
            "id": "0", "type": "preference_event", "subject": "user", "predicate": "likes",
            "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
            "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02", "valid_from": None,
        }],
    )  # fmt: skip
    out = RuleLLM().generate_json("memory_extraction_compact", system=compact, user=user)
    assert [set(d) for d in out["decisions"]] == [{"id", "decision", "target_id", "reason"}]
    assert out["decisions"][0]["id"] == "0" and out["decisions"][0]["decision"] == "REVISE"


@pytest.mark.parametrize(
    ("new", "old", "expected"),
    [
        ("Jake suggests ordering takeout.", "Jake suggests ordering takeout or some food.", 1.0),
        ("Tasha says the first set is 25 yuan.", "Tasha says a full set costs 25 yuan.", 5 / 6),
        ("Jake asks whether they still have any coins left.", "Jake says they have one coin left.", 0.6),
        ("Jake repeats that customs took his tweezers.", "Jake asks if the glove is broken now.", 0.2),
        ("Jake sets the table for dinner.", "Jake brought plates to the kitchen.", 0.25),
        ("Shure checks the fridge for eggs.", "Jake picks up the guitar.", 0.0),
        ("", "Jake picks up the guitar.", 0.0),
    ],
)
def test_verdict_similarity_is_containment_over_content_tokens(new, old, expected):
    """Without a head to remove, verdict_similarity is |A ∩ B| / min(|A|, |B|) over stemmed content tokens (0.0 when a side is empty), so a restatement scores 1.0, a positionally paired unrelated statement at most a shared name, and the r3 pairs score their measured values.
    不去头部时，verdict_similarity 是词干化内容词元上的 |A ∩ B| / min(|A|, |B|)（一侧为空时 0.0），因此复述得 1.0、按位置误配的无关陈述至多共享一个人名，r3 的配对得到其实测值。

    Input / 输入：
        new: The new statement.
        new：新陈述。

        old: The target text.
        old：目标文本。

        expected: The containment value.
        expected：包含度取值。

    Output / 输出：
        An equality (approx).
        一个（近似）相等断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert verdict_similarity(new, old) == pytest.approx(expected)
    assert verdict_similarity(old, new) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("new", "old", "head", "expected"),
    [
        # The row-104 pairs: only subject + verb in common → nothing left in common.
        # 第 104 行的配对：只有主语 + 动词相同 → 去头后无共同词元。
        ("User has dogs.", "User has cats.", ("user", "user", "has", "has"), 0.0),
        ("User is tired.", "User is happy.", ("user", "user", "is", "is"), 0.0),
        ("Shure likes the green plate.", "Shure likes the blue cup.", ("Shure", "Shure", "likes", "likes"), 0.0),
        ("Jake said there are two pots.", "Jake said it was put in the fridge.", ("Jake", "Jake", "said", "said"), 0.0),
        ("Jake left.", "Jake arrived.", ("Jake", "Jake", "left", "arrived"), 0.0),
        # A differing predicate stays in both sets (discriminating content).
        # 谓词不同时两侧都保留（有区分度的内容）。
        ("Jake asked Alice to help him take apart the camera.", "Jake said he is going to take the camera apart.",
         ("Jake", "Jake", "asked", "said"), 0.6),
        ("Jake reminded Tasha to wear glasses.", "Jake asked where the glasses are.", ("Jake", "Jake", "reminded", "asked"), 0.5),
        # Restatements keep scoring at or near 1.0; a subject given as a phrase is split.
        # 复述仍得 1.0 或接近；短语形式的主语被切分。
        ("Jake said the device has power but is not on.", "Jake said the device has power.", ("Jake", "Jake", "said", "said"), 1.0),
        ("Jake set the dinner table.", "Jake sets the table for dinner.", ("Jake", "Jake", "set", "sets"), 1.0),
        ("Tasha says the first set is 25 yuan.", "Tasha says a full set costs 25 yuan.", ("Tasha", "Tasha", "says", "says"), 0.75),
        ("Shure's GF says the song is by Pu Shu.", "Shure's GF says the song is by Accusefive.",
         ("Shure's GF", "Shure's GF", "says", "says"), 0.5),
        # A statement that is nothing but its head has no token left → 0.0 (never a division by zero).
        # 只剩头部的陈述无词元 → 0.0（绝不除零）。
        ("User works.", "User works.", ("user", "user", "works", "works"), 0.0),
    ],
)  # fmt: skip
def test_verdict_similarity_removes_the_head(new, old, head, expected):
    """With subjects and predicates given, verdict_similarity removes each side's subject tokens and the predicate tokens when both predicates agree, then takes the containment of what is left — so statements that share only their speaker and verb score 0.0 (they scored exactly 0.5 with the subject counted) while restatements keep their score; the measure is symmetric.
    给定主语与谓词时，verdict_similarity 去掉各侧的主语词元、并在双方谓词一致时去掉谓词词元，再对剩余部分取包含度——因此只共享说话人和动词的陈述得 0.0（主语计入时它们恰得 0.5），复述保持原分；度量是对称的。

    Input / 输入：
        new: The new statement.
        new：新陈述。

        old: The target text.
        old：目标文本。

        head: (new_subject, old_subject, new_predicate, old_predicate).
        head：(new_subject, old_subject, new_predicate, old_predicate)。

        expected: The containment value.
        expected：包含度取值。

    Output / 输出：
        An equality (approx) in both directions.
        双向的（近似）相等断言。

    State effect / 状态影响：
        None.
        无。
    """
    ns, os_, np_, op = head
    assert verdict_similarity(new, old, new_subject=ns, old_subject=os_, new_predicate=np_, old_predicate=op) == pytest.approx(expected)
    assert verdict_similarity(old, new, new_subject=os_, old_subject=ns, new_predicate=op, old_predicate=np_) == pytest.approx(expected)


def test_statement_tokens_drop_the_head():
    """statement_tokens returns the stemmed content tokens minus the subject and predicate tokens (each lower-cased, split on blanks / underscores, stemmed); `user` is already a stopword, a named subject in the text is removed, None / '' removes nothing.
    statement_tokens 返回词干化内容词元减去主语与谓词词元（各自小写、按空白 / 下划线切分、词干化）；`user` 本就是停用词，文本里的具名主语被去掉，None / '' 不去除任何词元。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Set equalities.
        集合相等断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert statement_tokens("User likes black coffee.") == {"like", "black", "coffee"}
    assert statement_tokens("User likes black coffee.", "user") == {"like", "black", "coffee"}
    assert statement_tokens("User likes black coffee.", "user", "likes") == {"black", "coffee"}
    assert statement_tokens("Jake sets the table.", "Jake") == {"set", "table"}
    assert statement_tokens("Jake sets the table.", None, None) == {"jake", "set", "table"}
    assert statement_tokens("Sarah Connor shipped the design.", "Sarah Connor", "shipped") == {"design"}
    assert statement_tokens("User lives in Paris.", "user", "lives_in") == {"pari"}
    assert statement_tokens("", None) == set() and statement_tokens("", "user", "has") == set()
