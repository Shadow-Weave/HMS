"""Single-call extraction tests: single vs staged equivalence, LLM call counts, and the scripted degrade paths.
单次调用抽取测试：single 与 staged 的等价性、LLM 调用计数，以及脚本化 LLM 的降级路径。
"""

# Purpose (EN): Assert that Ingestor(extraction_mode='single') reproduces the staged trio with ONE LLM round trip.
# 目的（中）：断言 Ingestor(extraction_mode='single') 用一次 LLM 往返复现分段三件套的结果。

# Input (EN): tests/fixtures/events/chat_*.json, the rule LLM, scripted LLM replies and the conftest clock/id helpers.
# 输入（中）：tests/fixtures/events/chat_*.json、规则 LLM、脚本化 LLM 回复与 conftest 的时钟/id 辅助。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from hms_elite.configs import prompts
from hms_elite.configs.base import HMSConfig
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.exceptions import ValidationError
from hms_elite.index.memory import MemoryIndex
from hms_elite.ingest.extraction import ExtractionBundle
from hms_elite.ingest.pipeline import EXTRACTION_MODES, Ingestor
from hms_elite.llms.rule import RuleLLM
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.memory.main import HMS
from hms_elite.memory.storage import StateStore
from hms_elite.memory.utils import coerce_messages
from hms_elite.models import Scope
from hms_elite.utils import json as ujson
from tests.conftest import EVENTS_DIR, FrozenClock, SequentialIds, load_events

CHAT_FIXTURES = sorted(path.stem for path in EVENTS_DIR.glob("chat_*.json"))
EXTRACTION_LOGGER = "hms_elite.ingest.extraction"
LATTE = {
    "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
    "objects": ["oat-milk lattes"], "relation": "single_fact", "text": "User likes oat-milk lattes.",
    "date_observed": "2026-04-02",
}  # fmt: skip
BLACK = {
    "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
    "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
    "date_observed": "2026-04-20", "supersedes_hint_ids": ["0"],
}  # fmt: skip
# BLACK without its supersedes hint: the degrade cases below must not be rescued by the hint fallback.
# 去掉 supersedes 提示的 BLACK：下面的降级用例不能被提示回退所挽救。
BLACK_NO_HINT = {**BLACK, "supersedes_hint_ids": []}
SEED_REPLY = {
    "observations": [LATTE],
    "eps": [],
    "decisions": [{"id": "0", "decision": "ADD", "target_id": None, "reason": "no candidates"}],
}


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
def _facade(mode: str) -> tuple[HMS, FrozenClock, list[int]]:
    """A fresh in-memory HMS on the rule LLM in one extraction mode, counting generate_response calls per add().
    在某种抽取模式下、基于规则 LLM 的全新内存 HMS，按每次 add() 统计 generate_response 调用次数。
    """
    clock = FrozenClock()
    store = StateStore(":memory:", now=clock.now, id_factory=SequentialIds())
    # The single ≡ staged equivalence is stated on the full contract (staged has no lines prompts; the library
    # default is lines since 2026-09-18).
    # single ≡ staged 的等价性建立在完整契约上（staged 没有行式提示词；库默认自 2026-09-18 起为 lines）。
    hms = HMS(
        HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, ingest={"extraction_output": "full"}),
        store=store,
    )
    hms.ingestor.extraction_mode = mode
    counts: list[int] = []
    original = hms.llm.generate_response

    def counting(messages: list[dict], *args: Any, **kwargs: Any) -> Any:
        """Count the call, then delegate to the real provider.
        计数后再委托给真实提供者。
        """
        counts[-1] += 1
        return original(messages, *args, **kwargs)

    hms.llm.generate_response = counting  # type: ignore[method-assign]
    return hms, clock, counts


def _replay(hms: HMS, clock: FrozenClock, counts: list[int], name: str) -> list[dict]:
    """Feed one chat fixture into the facade message by message (own run_id when the message carries one).
    把一个聊天夹具逐条喂给门面（消息自带 run_id 时使用它）。
    """
    results: list[dict] = []
    for message in load_events(name).messages:
        clock.tick(seconds=60)
        counts.append(0)
        results.append(
            hms.add(
                message["content"],
                user_id="alice",
                run_id=message.get("run_id", f"run-{name}"),
                timestamp=message.get("timestamp"),
            )
        )
    return results


def _scrub(obj: Any) -> Any:
    """Drop every id-like and instant-like key recursively so two stores compare on content only.
    递归去掉所有 id 类与时刻类的键，使两个存储只按内容比较。
    """
    if isinstance(obj, dict):
        return {
            key: _scrub(value)
            for key, value in obj.items()
            if not (key == "id" or key.endswith(("_id", "_ids", "_at")))
        }
    if isinstance(obj, list):
        return [_scrub(item) for item in obj]
    return obj


def _snapshot(hms: HMS) -> dict:
    """Rows (scrubbed), history event sequences and graph stats of the alice scope.
    alice 范围的行（已去 id/时刻）、历史事件序列与图统计。
    """
    rows = hms.get_all(filters={"user_id": "alice"}, top_k=1000, show_expired=True)["results"]
    lineages: list[str] = []
    for row in rows:
        if row["id"] not in lineages:
            lineages.append(row["id"])
    history = [
        [(entry["event"], entry["old_memory"], entry["new_memory"], entry["role"]) for entry in hms.history(lineage)]
        for lineage in lineages
    ]
    return {
        "rows": _scrub(rows),
        "history": history,
        "graph_stats": hms.graph_stats(filters={"user_id": "alice"}),
    }


class _Harness:
    """Ingestor over a fresh in-memory store, memory index and hash embedder with a caller-supplied LLM.
    绑定全新内存 store、内存索引与哈希嵌入器、由调用方提供 LLM 的 Ingestor。
    """

    def __init__(self, llm: Any, *, mode: str = "single", **config: Any) -> None:
        """Wire the collaborators (extra kwargs are HMSConfig sections such as `ingest`).
        接线协作对象（额外关键字参数是 HMSConfig 的分节，如 `ingest`）。
        """
        self.clock = FrozenClock()
        self.store = StateStore(":memory:", now=self.clock.now, id_factory=SequentialIds())
        self.llm = llm
        self.embedder = HashEmbedder({"embedding_dims": 64})
        self.index = MemoryIndex(dims=self.embedder.dims)
        self.config = HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, **config)
        # The single-vs-staged equivalence here is stated on the full contract (`memory_extraction` against the
        # observation / ep / revision trio, scripted full-shaped replies), so extraction_output='full' is pinned
        # rather than relying on the library default ('lines' since 2026-09-18, CONTRACT §6.6 / changelog row 108 — the
        # pin keeps this test on the full contract); the compact single/staged twins are pinned in
        # tests/test_extraction_compact.py.
        # 这里的单次-分段等价是在完整契约上陈述的（`memory_extraction` 对观察 / EP / 修订三件套，脚本化的完整回复），
        # 因此钉住 extraction_output='full' 而不依赖库默认（2026-09-18 起为 'lines'，CONTRACT §6.6 / 变更记录第 108 行——
        # 钉住使本测试留在完整契约上）；紧凑的单次 / 分段孪生在 tests/test_extraction_compact.py 中钉住。
        self.ingestor = Ingestor(
            self.store, self.index, self.embedder, llm, self.config, extraction_mode=mode, extraction_output="full"
        )

    def add(self, text: str, *, timestamp: str, run_id: str = "run-single") -> Any:
        """One inferred add() of a single user message.
        对单条用户消息执行一次推断式 add()。
        """
        self.clock.tick(seconds=60)
        return self.ingestor.ingest_messages(
            coerce_messages(text),
            scope=Scope(user_id="alice", run_id=run_id),
            user_meta={},
            infer=True,
            memory_type=None,
            prompt=None,
            event_time=timestamp,
            valid_to=None,
        )

    def candidates_shown(self) -> list[dict]:
        """The `Candidate Existing Memories` section of the last prompt the LLM received.
        LLM 最近一次收到的提示词中的 `Candidate Existing Memories` 分节。
        """
        user = self.llm.calls[-1]["messages"][-1]["content"]
        return prompts.parse_sections(user)["Candidate Existing Memories"]


# ---------------------------------------------------------------------------
# (a) single ≡ staged on every chat fixture
# (a) 每个聊天夹具上 single ≡ staged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CHAT_FIXTURES)
def test_single_mode_matches_staged_mode(name):
    """Rows (minus ids/instants), history event sequences and graph stats are identical between the two modes.
    两种模式下的行（去掉 id/时刻）、历史事件序列与图统计逐项相同。

    Input / 输入：
        name: Chat fixture stem.
        name：聊天夹具文件名。

    Output / 输出：
        Assertions over canonical_json of both snapshots plus a sanity check that something non-trivial happened.
        对两份快照 canonical_json 的断言，外加"确实发生了非平凡事件"的合理性检查。

    State effect / 状态影响：
        None outside two fresh in-memory stores.
        除两个全新内存 store 外无副作用。
    """
    snapshots: dict[str, dict] = {}
    for mode in EXTRACTION_MODES:
        hms, clock, counts = _facade(mode)
        _replay(hms, clock, counts, name)
        snapshots[mode] = _snapshot(hms)
    single, staged = snapshots["single"], snapshots["staged"]
    assert ujson.canonical_json(single["rows"]) == ujson.canonical_json(staged["rows"])
    assert single["history"] == staged["history"]
    assert single["graph_stats"] == staged["graph_stats"]
    assert single["rows"], name
    events = {event for lineage in single["history"] for event, _old, _new, _role in lineage}
    if name in {"chat_preference", "chat_zh"}:
        assert "UPDATE" in events and single["graph_stats"]["edges"]["revision"] == 1
    if name == "chat_conflict":
        assert "CONFLICT" in events and single["graph_stats"]["unresolved_conflicts"] == 1


# ---------------------------------------------------------------------------
# (b) call counts: single = exactly one generate_response per add(); staged = one to three
# (b) 调用计数：single 每次 add() 恰一次 generate_response；staged 一到三次
# ---------------------------------------------------------------------------
def test_single_mode_makes_exactly_one_llm_call_per_add():
    """Every add() of every chat fixture costs one LLM call in single mode and 1–3 in staged mode (strictly more in total).
    每个聊天夹具的每次 add() 在 single 模式下花费一次 LLM 调用，staged 模式下 1–3 次（总数严格更多）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the per-add() call counts of both modes.
        对两种模式每次 add() 调用次数的断言。

    State effect / 状态影响：
        None outside fresh in-memory stores.
        除全新内存 store 外无副作用。
    """
    totals = {mode: 0 for mode in EXTRACTION_MODES}
    for name in CHAT_FIXTURES:
        for mode in EXTRACTION_MODES:
            hms, clock, counts = _facade(mode)
            results = _replay(hms, clock, counts, name)
            assert len(counts) == len(results) == len(load_events(name).messages)
            if mode == "single":
                assert counts == [1] * len(counts), (name, counts)
            else:
                assert all(1 <= count <= 3 for count in counts), (name, counts)
            totals[mode] += sum(counts)
    assert totals["staged"] > totals["single"]


# ---------------------------------------------------------------------------
# (c) scripted degrade paths: missing / malformed decisions and unknown targets fall back to ADD
# (c) 脚本化降级路径：decisions 缺失 / 畸形与未知目标都回退为 ADD
# ---------------------------------------------------------------------------
_DEGRADE_CASES = {
    "missing_root": ({"observations": [BLACK]}, "decisions root missing"),
    "root_not_a_list": ({"observations": [BLACK], "decisions": "REVISE"}, "decisions root missing"),
    "items_not_dicts": ({"observations": [BLACK], "decisions": ["REVISE", 3]}, "no decision for observation '0'"),
    "unknown_verdict": (
        {"observations": [BLACK], "decisions": [{"id": "0", "decision": "MAYBE", "target_id": "0"}]},
        "unknown decision 'MAYBE'",
    ),
    "unknown_target": (
        {"observations": [BLACK_NO_HINT], "decisions": [{"id": "0", "decision": "REVISE", "target_id": "7"}]},
        "unknown target '7'",
    ),
    "null_target": (
        {"observations": [BLACK_NO_HINT], "decisions": [{"id": "0", "decision": "REVISE", "target_id": None}]},
        "unknown target None",
    ),
}


@pytest.mark.parametrize("case", sorted(_DEGRADE_CASES))
def test_scripted_decisions_degrade_to_add(case, caplog):
    """A second add() whose reply lacks a usable decision creates a second valid atom (ADD) and logs one warning.
    第二次 add() 的回复没有可用决策时，新增第二个有效原子（ADD）并记录一条警告。

    The unknown_target / null_target cases use BLACK_NO_HINT: since the coerce-never-reject change, an unresolvable
    target falls back to the observation's own supersedes_hint_ids when they name exactly one candidate (see
    test_unknown_target_falls_back_to_the_supersedes_hint), so the pure degrade path needs an observation without hints.
    unknown_target / null_target 用例使用 BLACK_NO_HINT：自"只规整、不拒绝"改动起，无法解析的目标会回退到观察自身的
    supersedes_hint_ids（当它恰好指向一个候选时，见 test_unknown_target_falls_back_to_the_supersedes_hint），
    因此纯降级路径需要一条没有提示的观察。

    Input / 输入：
        case: Key of _DEGRADE_CASES (the scripted second reply and the expected warning fragment).
        case：_DEGRADE_CASES 的键（脚本化的第二次回复与期望的警告片段）。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions over the rows, the atom count, the candidate the LLM was shown and the warning.
        对行、原子数、LLM 看到的候选与警告的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    reply, fragment = _DEGRADE_CASES[case]
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    h = _Harness(ScriptedLLM({"responses": [SEED_REPLY, reply]}))
    first = h.add("I like oat-milk lattes.", timestamp="2026-04-02T08:00:00Z")
    assert first.rows[0]["event"] == "ADD" and h.candidates_shown() == []
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z")
    shown = h.candidates_shown()
    assert [c["id"] for c in shown] == ["0"] and shown[0]["text"] == "User likes oat-milk lattes."
    assert [c["task"] for c in h.llm.calls] == ["memory_extraction", "memory_extraction"]
    assert second.rows[0]["event"] == "ADD" and second.rows[0]["memory"] == "User likes black coffee."
    assert second.rows[0]["id"] != first.rows[0]["id"] and "revises" not in second.rows[0]
    assert h.store.count_atoms(Scope(user_id="alice", run_id="run-single")) == 2
    assert h.store.list_edges(Scope(user_id="alice", run_id="run-single")) == []
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("memory_extraction")]
    assert len(warnings) == 1 and fragment in warnings[0], warnings


@pytest.mark.parametrize("target_id", ["7", None])
def test_unknown_target_falls_back_to_the_supersedes_hint(target_id, caplog):
    """A REVISE with an unknown / null target_id is applied to the single candidate the supersedes_hint_ids resolve to.
    target_id 未知 / 为 null 的 REVISE 会对照 supersedes_hint_ids 解析出的唯一候选执行。

    The old atom is therefore archived instead of a duplicate ADD being stored.
    因此旧原子被归档，而不是存入一个重复的 ADD。

    Input / 输入：
        target_id: The unusable target ('7' or None); BLACK's supersedes_hint_ids ["0"] resolve to candidate '0'.
        target_id：不可用的目标（'7' 或 None）；BLACK 的 supersedes_hint_ids ["0"] 解析为候选 '0'。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions: the second row is UPDATE revising the first atom (one valid atom, the old one archived); one
        warning names the hint fallback.
        断言：第二行是修订首个原子的 UPDATE（一个有效原子，旧原子已归档）；一条警告指出使用了提示回退。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    reply = {"observations": [BLACK], "decisions": [{"id": "0", "decision": "REVISE", "target_id": target_id}]}
    h = _Harness(ScriptedLLM({"responses": [SEED_REPLY, reply]}))
    first = h.add("I like oat-milk lattes.", timestamp="2026-04-02T08:00:00Z")
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z")
    assert second.rows[0]["event"] == "UPDATE" and second.rows[0]["memory"] == "User likes black coffee."
    assert second.rows[0]["id"] == first.rows[0]["id"] and second.rows[0]["revises"] == first.rows[0]["atom_id"]
    assert h.store.count_atoms(Scope(user_id="alice", run_id="run-single")) == 1
    assert h.store.get_atom(first.rows[0]["atom_id"]).archived_reason == "revised"
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("memory_extraction")]
    assert len(warnings) == 1 and "using its supersedes hint" in warnings[0], warnings


def test_scripted_valid_decision_revises_the_shown_candidate(caplog):
    """The positive control: a REVISE against the shown candidate id '0' archives the old atom in one LLM call.
    正向对照：针对呈现的候选 id '0' 的 REVISE 在一次 LLM 调用内归档旧原子。

    Input / 输入：
        caplog: pytest log capture (must stay silent).
        caplog：pytest 日志捕获（必须保持安静）。

    Output / 输出：
        Assertions over the UPDATE row, lineage, revision edge and the absence of warnings.
        对 UPDATE 行、谱系、revision 边与无警告的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    reply = {
        "observations": [BLACK],
        "eps": [],
        "decisions": [{"id": "0", "decision": "REVISE", "target_id": "0", "reason": "supersession cue"}],
    }
    h = _Harness(ScriptedLLM({"responses": [SEED_REPLY, reply]}))
    first = h.add("I like oat-milk lattes.", timestamp="2026-04-02T08:00:00Z")
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z")
    scope = Scope(user_id="alice", run_id="run-single")
    assert second.rows[0]["event"] == "UPDATE" and second.rows[0]["id"] == first.rows[0]["id"]
    assert second.rows[0]["revises"] == first.rows[0]["atom_id"] and second.rows[0]["version"] == 2
    assert h.store.count_atoms(scope) == 1
    assert h.store.get_atom(first.rows[0]["atom_id"]).archived_reason == "revised"
    # supersedes_hint_ids resolved through the same synthetic id map as the decision target
    # supersedes_hint_ids 通过与决策目标相同的合成 id 映射解析
    head = h.store.get_atom(second.rows[0]["atom_id"])
    assert h.store.get_observation(head.source_id).supersedes_hint_ids == [first.rows[0]["atom_id"]]
    edges = h.store.list_edges(scope, edge_types=["revision"])
    assert len(edges) == 1 and edges[0].label == "supersession cue"
    assert len(h.llm.calls) == 2 and not [r for r in caplog.records if r.name == EXTRACTION_LOGGER]


def test_single_mode_never_shows_candidates_unless_revisions_are_inline():
    """With revision_mode deferred the candidate section is empty and a REVISE reply still lands as ADD.
    revision_mode 为 deferred 时候选分节为空，回复 REVISE 也照样落为 ADD。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the shown candidates, the ADD row and the plan reason.
        对呈现的候选、ADD 行与计划原因的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    reply = {
        "observations": [BLACK],
        "eps": [],
        "decisions": [{"id": "0", "decision": "REVISE", "target_id": "0", "reason": "supersession cue"}],
    }
    h = _Harness(ScriptedLLM({"responses": [SEED_REPLY, reply]}), ingest={"revision_mode": "deferred"})
    h.add("I like oat-milk lattes.", timestamp="2026-04-02T08:00:00Z")
    second = h.add("Actually I switched to black coffee now.", timestamp="2026-04-20T08:00:00Z")
    assert h.candidates_shown() == [] and second.rows[0]["event"] == "ADD"
    assert h.store.count_atoms(Scope(user_id="alice", run_id="run-single")) == 2


def test_non_object_reply_extracts_nothing_without_crashing(caplog):
    """A scalar JSON or garbage text reply yields no rows, one warning, and leaves the anchored events in place.
    标量 JSON 或垃圾文本回复不产生任何行，只记录一条警告，并保留已落锚的事件。

    A non-empty bare list is no longer "non-object": it is read as the observations array
    (test_ingest.test_bare_list_reply_is_read_as_observations).
    非空裸列表不再算"非对象"：按 observations 数组读取（见 test_ingest.test_bare_list_reply_is_read_as_observations）。

    Input / 输入：
        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions over rows, events and the warning.
        对行、事件与警告的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    caplog.set_level(logging.WARNING, logger=EXTRACTION_LOGGER)
    h = _Harness(ScriptedLLM({"responses": ["42", "not json at all"]}))
    for text in ("I like oat-milk lattes.", "I like tea."):
        res = h.add(text, timestamp="2026-04-02T08:00:00Z")
        assert res.rows == [] and len(res.events) == 1
    assert h.store.stats()["memory_atoms"] == 0 and h.store.stats()["source_events"] == 2
    warnings = [r.getMessage() for r in caplog.records if "not a JSON object" in r.getMessage()]
    assert len(warnings) == 2


def test_single_mode_extracts_eps_in_the_same_call():
    """The action-closure fixture yields its observation AND its EP from the one memory_extraction reply.
    行动闭包夹具在唯一一次 memory_extraction 回复中同时得到观察与 EP。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over ep_ids, the EP slots and the single call.
        对 ep_ids、EP 槽位与单次调用的断言。

    State effect / 状态影响：
        None outside a fresh in-memory store.
        除全新内存 store 外无副作用。
    """
    h = _Harness(RuleLLM())
    message = load_events("chat_action_closure").messages[0]
    h.clock.tick(seconds=60)
    res = h.ingestor.ingest_messages(
        coerce_messages([message]),
        scope=Scope(user_id="alice", run_id="run-closure"),
        user_meta={},
        infer=True,
        memory_type=None,
        prompt=None,
        event_time="2026-06-15T10:04:00Z",
        valid_to=None,
    )
    assert [c["task"] for c in h.llm.calls] == ["memory_extraction"]
    sections = h.llm.calls[0]["sections"]
    assert [e["idx"] for e in sections["Events"]] == [0] and sections["Known Constraints"] == []
    assert len(res.ep_ids) == 1
    ep = h.store.get_ep(res.ep_ids[0])
    assert ep.slots.a == "patched retry.ts to add backoff" and ep.slots.r == "negative"
    assert {row["event"] for row in res.rows} == {"ADD"} and len(res.rows) == 2


# ---------------------------------------------------------------------------
# Prompt contract, rule handler and scripted replay of the multi-root object
# 提示词契约、规则处理器与多根对象的脚本化回放
# ---------------------------------------------------------------------------
def test_memory_extraction_prompt_contract():
    """The composite prompt carries the marker and five headings; its OUTPUT FORMAT has exactly the three roots; the builder's sections are the union of the three staged builders.
    复合提示词带标记与五个标题；其 OUTPUT FORMAT 恰有三个根；构造器的分节是三个分段构造器的并集。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over MEMORY_EXTRACTION_PROMPT, COMPOSITE_PROMPTS and build_memory_extraction_prompt.
        对 MEMORY_EXTRACTION_PROMPT、COMPOSITE_PROMPTS 与 build_memory_extraction_prompt 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    prompt = prompts.MEMORY_EXTRACTION_PROMPT
    assert prompts.task_of(prompt) == "memory_extraction" and prompts.COMPOSITE_TASKS[0] == "memory_extraction"
    assert prompts.COMPOSITE_PROMPTS["memory_extraction"] == prompt
    assert "memory_extraction" not in prompts.ROOT_KEY and "memory_extraction" not in prompts.TASKS
    for heading in ("# ROLE", "# INPUTS", "# GUIDELINES", "# EXAMPLES", "# OUTPUT FORMAT"):
        assert heading in prompt, heading
    section = prompts.output_format_section(prompt)
    assert list(json.loads(ujson.extract_json(section))) == list(prompts.MEMORY_EXTRACTION_ROOTS)
    assert section.endswith(prompts.JSON_TRAILER)
    for title in ("## Events", "## Known Constraints", "## Recent EPs", "## Candidate Existing Memories"):
        assert title in prompt, title
    kwargs = dict(
        existing_memories=[{"id": "0", "text": "User likes oat-milk lattes in the morning."}],
        new_messages=[{"idx": 0, "role": "user", "text": "x", "timestamp": "2026-04-20T08:00:00+00:00"}],
        last_messages=[{"role": "user", "text": "earlier"}],
        observation_date="2026-04-20",
        current_date="2026-09-02",
        custom_instructions="Be brief.",
    )
    ep_kwargs = dict(events=[], known_constraints=[{"id": "c0", "text": "c"}], recent_eps=[], observation_date="2026-04-20")
    candidates = [{"id": "0", "type": "preference_event", "text": "User likes oat-milk lattes in the morning."}]
    built = prompts.build_memory_extraction_prompt(
        **kwargs, events=[], known_constraints=[{"id": "c0", "text": "c"}], recent_eps=[], candidates=candidates
    )
    sections = prompts.parse_sections(built)
    expected = set(prompts.parse_sections(prompts.build_observation_prompt(**kwargs)))
    expected |= set(prompts.parse_sections(prompts.build_ep_prompt(**ep_kwargs)))
    expected |= {"Candidate Existing Memories"}
    assert set(sections) == expected and built.endswith("# Output:")
    assert sections["Candidate Existing Memories"] == candidates and sections["Events"] == []
    assert sections["Custom Instructions"] == "Be brief." and sections["Last k Messages"] == "user: earlier"


def test_rule_llm_memory_extraction_matches_the_three_staged_handlers():
    """RuleLLM answers memory_extraction with the same observations / eps / decisions its staged handlers give.
    RuleLLM 对 memory_extraction 的回答与其三个分段处理器给出的观察 / EP / 决策一致。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over generate_json('memory_extraction') against the three staged generate_json calls.
        对 generate_json('memory_extraction') 与三次分段 generate_json 调用的比对断言。

    State effect / 状态影响：
        None.
        无。
    """
    llm = RuleLLM()
    new_messages = [
        {"idx": 0, "role": "user", "text": "Actually I switched to black coffee now.", "timestamp": "2026-04-20T08:00:00+00:00"},
        {"idx": 1, "role": "assistant", "text": "I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed.", "timestamp": "2026-04-20T08:05:00+00:00"},
    ]  # fmt: skip
    existing = [{"id": "0", "text": "User likes oat-milk lattes in the morning."}]
    candidates = [
        {"id": "0", "type": "preference_event", "subject": "user", "predicate": "likes",
         "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
         "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02",
         "valid_from": "2026-04-02T08:00:00+00:00"},
    ]  # fmt: skip
    events = [{"idx": 0, "role": "assistant", "object_type": "message", "text": new_messages[1]["text"], "timestamp": new_messages[1]["timestamp"]}]  # fmt: skip
    shared = dict(observation_date="2026-04-20", current_date="2026-09-02")
    combined = llm.generate_json(
        "memory_extraction",
        system=prompts.MEMORY_EXTRACTION_PROMPT,
        user=prompts.build_memory_extraction_prompt(
            existing_memories=existing, new_messages=new_messages, last_messages=[], **shared,
            events=events, known_constraints=[], recent_eps=[], candidates=candidates,
        ),  # fmt: skip
    )
    assert isinstance(combined, dict) and list(combined) == list(prompts.MEMORY_EXTRACTION_ROOTS)
    observations = llm.generate_json(
        "observation_extraction",
        system=prompts.OBSERVATION_EXTRACTION_PROMPT,
        user=prompts.build_observation_prompt(
            existing_memories=existing, new_messages=new_messages, last_messages=[], **shared
        ),
    )
    eps = llm.generate_json(
        "ep_extraction",
        system=prompts.EP_EXTRACTION_PROMPT,
        user=prompts.build_ep_prompt(events=events, known_constraints=[], recent_eps=[], observation_date="2026-04-20"),
    )
    new_items = [
        {k: o[k] for k in ("id", "type", "subject", "predicate", "objects", "relation", "text", "date_observed")}
        for o in observations
    ]
    decisions = llm.generate_json(
        "revision_decision",
        system=prompts.REVISION_DECISION_PROMPT,
        user=prompts.build_revision_prompt(new_items=new_items, candidates=candidates, current_date="2026-09-02"),
    )
    assert combined["observations"] == observations and len(observations) == 2
    assert combined["eps"] == eps and len(eps) == 1
    assert combined["decisions"] == [{"id": d["new_id"], **{k: d[k] for k in ("decision", "target_id", "reason")}} for d in decisions]
    assert combined["decisions"][0]["decision"] == "REVISE" and combined["decisions"][0]["target_id"] == "0"
    assert combined["decisions"][1]["decision"] == "ADD"
    # An empty Events list means the EP rules never run (the staged path skips ep_extraction the same way).
    # Events 为空列表时 EP 规则不运行（分段路径同样跳过 ep_extraction）。
    without_events = llm.generate_json(
        "memory_extraction",
        system=prompts.MEMORY_EXTRACTION_PROMPT,
        user=prompts.build_memory_extraction_prompt(
            existing_memories=[], new_messages=new_messages, last_messages=[], **shared,
            events=[], known_constraints=[], recent_eps=[], candidates=[],
        ),  # fmt: skip
    )
    assert without_events["eps"] == [] and [d["reason"] for d in without_events["decisions"]] == ["no candidates"] * 2


def test_scripted_llm_replays_the_multi_root_object_and_bad_modes_are_rejected():
    """ScriptedLLM hands the whole `{observations, eps, decisions}` object back through generate_json; an unknown extraction_mode is VALIDATION_013.
    ScriptedLLM 通过 generate_json 原样返回整个 `{observations, eps, decisions}` 对象；未知 extraction_mode 为 VALIDATION_013。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the replayed object, ExtractionBundle defaults and the constructor error.
        对回放对象、ExtractionBundle 默认值与构造错误的断言。

    State effect / 状态影响：
        None.
        无。
    """
    reply = {"observations": [LATTE], "eps": [], "decisions": SEED_REPLY["decisions"]}
    llm = ScriptedLLM({"responses": [reply]})
    assert llm.generate_json("memory_extraction", system=prompts.MEMORY_EXTRACTION_PROMPT, user="x") == reply
    assert llm.calls[0]["task"] == "memory_extraction"
    empty = ExtractionBundle()
    assert (empty.observations, empty.eps, empty.decisions) == ([], [], [])
    with pytest.raises(ValidationError) as excinfo:
        _Harness(llm, mode="bogus")
    assert excinfo.value.error_code == "VALIDATION_013" and excinfo.value.details["supported"] == list(EXTRACTION_MODES)
