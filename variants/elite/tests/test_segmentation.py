"""Tests for hms_elite.segmentation: EN/ZH sentence boundaries, code fences, lists, protected identifiers, messages.
hms_elite.segmentation 的测试：中英句子边界、代码围栏、列表、受保护的标识符/URL、多轮消息。
"""

# Purpose (EN): Pin the HMS §5.1 segmentation contract — exact spans, kinds, languages and the never-cut guarantee —
#               and its wiring into the ingest main line (long messages become extraction units on ONE anchor).
# 目的（中）：钉住 HMS §5.1 的切分契约——精确区间、种类、语言与"绝不切断"保证——及其接入摄取主线
#            （长消息成为同一锚点上的抽取单元）。

# Input (EN): Hand-written texts, the tests/fixtures/events/chat_zh.json messages and in-memory HMS facades.
# 输入（中）：手写文本、tests/fixtures/events/chat_zh.json 消息与内存 HMS 门面。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory stores (the segmenter itself is pure).
# 状态影响（中）：除内存存储外无副作用（切分器本身是纯函数）。

from __future__ import annotations

from typing import Any

import pytest

from hms_elite import HMS, HMSConfig
from hms_elite.configs.prompts import parse_sections
from hms_elite.memory.storage import StateStore
from hms_elite.segmentation import KINDS, Segment, detect_lang, protected_spans, segment, segment_messages
from hms_elite.utils.text import sentences, tokenize
from tests.conftest import FrozenClock, SequentialIds

LONG_EN = (
    "We patched src/billing/retry.ts and https://example.com/docs/retry?attempt=3,4 explains the backoff policy, "
    "but the billing_e2e suite still failed because subscription_state_regression appeared in `state_machine.ts` "
    "after the deploy, so we reverted /opt/app/releases/2026-06-15/ and re-ran everything"
)


def _spans_match(text: str, segs: list[Segment]) -> None:
    """Every segment is the exact substring of its span, indices are 0..n-1 and spans are ordered and disjoint.
    每个分段都是其区间的精确子串，序号为 0..n-1，区间有序且互不重叠。
    """
    assert [s.index for s in segs] == list(range(len(segs)))
    last = 0
    for seg in segs:
        start, end = seg.span
        assert text[start:end] == seg.text and seg.text and start >= last
        assert seg.kind in KINDS
        last = end


def test_sentence_boundaries_en_and_zh():
    """EN sentences split on terminal punctuation, ZH sentences split on 。！？ even without whitespace, spans exact.
    英文句子按终止标点切分，中文句子即便无空白也按 。！？ 切分，区间精确。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over kinds, texts, languages and spans.
        对种类、文本、语言与区间的断言。

    State effect / 状态影响：
        None.
        无。
    """
    text = "User likes black coffee. She moved to Berlin in May! Does she still drink it?"
    segs = segment(text)
    _spans_match(text, segs)
    assert [s.text for s in segs] == [
        "User likes black coffee.",
        "She moved to Berlin in May!",
        "Does she still drink it?",
    ]
    assert {s.kind for s in segs} == {"sentence"} and {s.lang for s in segs} == {"en"}

    zh = "我喜欢燕麦拿铁。我现在改喝黑咖啡了。“真的吗？”然后呢"
    zh_segs = segment(zh)
    _spans_match(zh, zh_segs)
    assert [s.text for s in zh_segs] == ["我喜欢燕麦拿铁。", "我现在改喝黑咖啡了。", "“真的吗？”", "然后呢"]
    assert {s.lang for s in zh_segs} == {"zh"}
    assert segment(zh, lang="mixed")[0].lang == "mixed"

    mixed = "我在用 Python。User likes 咖啡."
    assert [s.lang for s in segment(mixed)] == ["zh", "en"]
    assert detect_lang("...") == "und" and detect_lang("2026") == "und"
    assert segment("") == [] and segment("   \n\n  ") == []


def test_code_blocks_urls_and_identifiers_are_never_cut():
    """Fenced code is one `code` atom with its fences; URLs, paths, identifiers and inline code survive clause cutting.
    围栏代码是一个带围栏的 `code` 原子；URL、路径、标识符与行内代码在子句切分中保持完整。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the code atom, the protected spans and the pieces of a long sentence.
        对代码原子、受保护区间与长句片段的断言。

    State effect / 状态影响：
        None.
        无。
    """
    text = "Before the block.\n```python\nx = 1, 2\nprint(x)\n```\nAfter, the block."
    segs = segment(text)
    _spans_match(text, segs)
    code = [s for s in segs if s.kind == "code"]
    assert len(code) == 1 and code[0].text == "```python\nx = 1, 2\nprint(x)\n```" and code[0].lang == "code"
    assert [s.text for s in segs if s.kind != "code"] == ["Before the block.", "After, the block."]
    unterminated = segment("Intro\n~~~\nraw stuff, still code")
    assert unterminated[-1].kind == "code" and unterminated[-1].text.startswith("~~~")

    protected = [LONG_EN[a:b] for a, b in protected_spans(LONG_EN)]
    for needle in (
        "src/billing/retry.ts",
        "https://example.com/docs/retry?attempt=3,4",
        "billing_e2e",
        "subscription_state_regression",
        "`state_machine.ts`",
        "/opt/app/releases/2026-06-15/",
    ):
        assert any(needle in p for p in protected), needle
    for max_chars in (400, 80, 40, 20):
        pieces = segment(LONG_EN, max_chars=max_chars)
        _spans_match(LONG_EN, pieces)
        for needle in protected:
            assert any(needle in piece.text for piece in pieces), (max_chars, needle)
        assert [t for p in pieces for t in tokenize(p.text)] == tokenize(LONG_EN)
    short = segment(LONG_EN, max_chars=80)
    assert len(short) > 1 and all(len(p.text) <= 80 for p in short)
    assert all(p.kind == "sentence" for p in short)
    # A single protected token longer than max_chars is kept whole rather than cut.
    # 单个超过 max_chars 的受保护词元整体保留而不被切断。
    url = "see https://example.com/a/very/long/path/that/keeps/going/and/going?with=query&and=more ok"
    assert any(
        "https://example.com/a/very/long/path" in s.text and "and=more" in s.text for s in segment(url, max_chars=16)
    )


def test_paragraphs_and_list_items():
    """Blank lines separate paragraphs; bullet, numbered and CJK-numbered lines become list items with markers stripped.
    空行分隔段落；项目符号、编号与中文编号行成为去掉标记的列表项。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over kinds and texts.
        对种类与文本的断言。

    State effect / 状态影响：
        None.
        无。
    """
    text = (
        "# Release notes\n\n"
        "- fix retry.ts backoff\n- add idempotency key\n  continued on a second line\n"
        "1. first step\n2) second step\n一、中文条目\n\n"
        "Trailing fragment without punctuation"
    )
    segs = segment(text)
    _spans_match(text, segs)
    assert segs[0].kind == "paragraph" and segs[0].text == "# Release notes"
    items = [s for s in segs if s.kind == "list_item"]
    assert [s.text for s in items] == [
        "fix retry.ts backoff",
        "add idempotency key\n  continued on a second line",
        "first step",
        "second step",
        "中文条目",
    ]
    assert items[-1].lang == "zh"
    assert segs[-1].kind == "paragraph" and segs[-1].text == "Trailing fragment without punctuation"
    two = segment("First sentence here. Second one.\n\nOnly one sentence.")
    assert [s.kind for s in two] == ["sentence", "sentence", "sentence"]


def test_segment_messages_roles_and_indices():
    """Multi-turn input yields conversation-wide indices, per-message spans, roles and message_index; bad input raises.
    多轮输入得到全对话序号、按消息的区间、角色与 message_index；非法输入抛错。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over Segment fields and TypeError / ValueError paths.
        对 Segment 字段与 TypeError / ValueError 路径的断言。

    State effect / 状态影响：
        None.
        无。
    """
    messages = [
        {"role": "user", "content": "Fix retry.ts today. Do not touch the state machine."},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Patched src/billing/retry.ts."}, {"type": "image"}],
        },
        {"role": "user", "content": "我现在改喝黑咖啡了。谢谢！"},
    ]
    segs = segment_messages(messages)
    assert [s.index for s in segs] == [0, 1, 2, 3, 4]
    assert [(s.role, s.message_index) for s in segs] == [
        ("user", 0),
        ("user", 0),
        ("assistant", 1),
        ("user", 2),
        ("user", 2),
    ]
    for seg in segs:
        body = messages[seg.message_index]["content"]
        text = body if isinstance(body, str) else body[0]["text"]
        assert text[seg.span[0] : seg.span[1]] == seg.text
    assert segs[2].text == "Patched src/billing/retry.ts." and segs[3].lang == "zh"
    assert [s.as_dict()["span"] for s in segs][:2] == [[0, 19], [20, 51]]
    assert segment_messages("one. two.")[1].text == "two."
    assert segment_messages({"role": "tool", "content": "ran tests"})[0].role == "tool"
    assert segment_messages([{"role": "user", "content": None}]) == []
    with pytest.raises(TypeError):
        segment_messages(42)
    with pytest.raises(TypeError):
        segment_messages(["not a dict"])
    with pytest.raises(TypeError):
        segment(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        segment("x", max_chars=0)


def test_segmentation_is_deterministic_and_reuses_fixture(events):
    """Same input → identical output, and the chat_zh fixture segments into one ZH sentence per message.
    相同输入 → 相同输出，且 chat_zh 夹具每条消息切成一个中文句子。

    Input / 输入：
        events: Fixture loader.
        events：夹具加载器。

    Output / 输出：
        Equality and fixture assertions.
        相等性与夹具断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert segment(LONG_EN, max_chars=60) == segment(LONG_EN, max_chars=60)
    fixture = events("chat_zh")
    segs = segment_messages(fixture.messages)
    assert [(s.text, s.lang, s.message_index) for s in segs] == [
        ("我喜欢燕麦拿铁。", "zh", 0),
        ("我现在改喝黑咖啡了。", "zh", 1),
    ]


# ---------------------------------------------------------------------------
# Wiring into the ingest main line (HMS_arch §2: SourceEvent → Anchor → extraction units → Observation)
# 接入摄取主线（HMS_arch §2：SourceEvent → Anchor → 抽取单元 → Observation）
# ---------------------------------------------------------------------------
_FACTS = ("I live in Berlin.", "I work at Stripe.", "I drink oat-milk lattes every morning.")
_FILLER = (
    "The rest of this paragraph is padding about nothing in particular, added so that the message grows well past "
    "the segmentation threshold without stating another fact. More padding follows the same idea, and a third "
    "padding sentence closes the paragraph quietly without any first-person claim at all."
)


def _long_message() -> str:
    """Five paragraphs (three carry one fact each, two are padding only), 1500+ characters in total.
    五个段落（三个各含一个事实，两个只有填充），总长超过 1500 个字符。
    """
    paragraphs = [f"{fact} {_FILLER}" for fact in _FACTS] + [_FILLER, _FILLER]
    text = "\n\n".join(paragraphs)
    assert len(text) >= 1500 and all(len(p) <= 400 for p in paragraphs)
    return text


def _one_fact_per_unit(messages: list[dict]) -> dict:
    """A scripted LLM that extracts the first first-person sentence of EVERY `New Messages` row (one fact per unit).
    一个脚本化 LLM：对 `New Messages` 的每一行抽取其第一个第一人称句子（每个单元一个事实）。
    """
    user = next(m["content"] for m in messages if m["role"] == "user")
    observations = []
    for row in parse_sections(user)["New Messages"]:
        claim = next((s for s in sentences(str(row["text"])) if s.startswith("I ")), None)
        if claim is None:
            continue
        observations.append(
            {
                "id": str(len(observations)),
                "event_idx": row["idx"],
                "type": "fact",
                "subject": "user",
                "predicate": "states",
                "objects": [],
                "relation": "single_fact",
                "text": claim,
                "date_observed": str(row["timestamp"])[:10],
                "confidence": 0.9,
                "attributed_to": "user",
                "sentiment": "neutral",
                "topic_tags": [],
                "entity_mentions": [],
                "supersedes_hint_ids": [],
            }
        )
    return {"observations": observations, "eps": [], "decisions": [{"id": o["id"], "decision": "ADD"} for o in observations]}


def _facade(llm: dict, *, segment_max_chars: int) -> Any:
    """An offline HMS (memory index, frozen clock, sequential ids) with the given LLM envelope and threshold.
    给定 LLM 信封与阈值的离线 HMS（memory 索引、冻结时钟、顺序 id）。
    """
    store = StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds())
    config = HMSConfig(
        llm=llm,
        vector_store={"provider": "memory"},
        store={"path": ":memory:"},
        # The scripted replies are full-contract JSON; the library default is lines since 2026-09-18.
        # 脚本化回复是完整契约的 JSON；库默认自 2026-09-18 起为 lines。
        ingest={"segment_max_chars": segment_max_chars, "extraction_output": "full"},
    )
    return HMS(config, store=store)


def _new_message_rows(hms: Any) -> list[int]:
    """Number of `New Messages` rows the LLM saw in each recorded call.
    LLM 在每次记录的调用中看到的 `New Messages` 行数。
    """
    out = []
    for call in hms.llm.calls:
        # ScriptedLLM records the raw messages, RuleLLM the parsed sections; both expose the same section.
        # ScriptedLLM 记录原始消息，RuleLLM 记录解析后的分节；两者都暴露同一个分节。
        if "sections" in call:
            out.append(len(call["sections"]["New Messages"]))
            continue
        user = next(m["content"] for m in call["messages"] if m["role"] == "user")
        out.append(len(parse_sections(user)["New Messages"]))
    return out


def test_long_messages_become_extraction_units_anchored_to_the_parent_message():
    """A 1500-char message with three facts yields >= 3 observations on ONE anchor once segmented; unsegmented, one.
    一条含三个事实的 1500 字符消息在切分后抽出 >= 3 条观察且都在同一个锚点上；不切分时只有一条。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the rows, their anchor ids, the anchor / event tables and the prompt rows the LLM saw.
        对行、其锚点 id、锚点 / 事件表以及 LLM 看到的提示词行的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    scripted = {"provider": "scripted", "config": {"responses": [_one_fact_per_unit], "loop": True}}
    text = _long_message()

    wired = _facade(scripted, segment_max_chars=400)
    result = wired.add(text, user_id="alice", timestamp="2026-06-15T09:00:00Z")
    rows = [row for row in result["results"] if row["event"] == "ADD"]
    assert len(rows) >= 3 and [row["memory"] for row in rows] == list(_FACTS)
    # One event, one anchor: every observation anchors to the parent message and no segment minted its own anchor.
    # 一事一锚：每条观察都锚到父消息，没有任何分段铸造自己的锚点。
    anchor_id = rows[0]["anchor_ids"][0]
    assert all(row["anchor_ids"] == [anchor_id] for row in rows)
    assert wired.store._execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 1
    assert wired.store._execute("SELECT COUNT(*) FROM source_events").fetchone()[0] == 1
    assert sorted(wired.store.atoms_for_anchor(anchor_id)) == sorted(row["atom_id"] for row in rows)
    # The LLM saw ONE prompt whose message rows are exactly the segments of the text, not one prompt per segment.
    # LLM 看到的是一条提示词，其消息行恰好是文本的各分段，而不是每个分段一条提示词。
    assert _new_message_rows(wired) == [len(segment(text, max_chars=400))] and _new_message_rows(wired)[0] >= 3
    wired.close()

    # Threshold far above the message length: the same LLM sees one row and extracts only the first fact.
    # 阈值远高于消息长度：同一个 LLM 只看到一行，只抽出第一个事实。
    unwired = _facade(scripted, segment_max_chars=10**9)
    before = unwired.add(text, user_id="alice", timestamp="2026-06-15T09:00:00Z")
    before_rows = [row for row in before["results"] if row["event"] == "ADD"]
    assert [row["memory"] for row in before_rows] == [_FACTS[0]] and _new_message_rows(unwired) == [1]
    assert before_rows[0]["anchor_ids"] == [anchor_id]  # same content, same anchor id / 相同内容，相同锚点 id
    unwired.close()


def test_rule_llm_rows_are_unchanged_by_segmentation_and_short_messages_are_untouched():
    """The rule LLM already works per sentence, so segmenting a long message leaves its rows byte-identical; a short message never reaches the segmenter (one prompt row either way).
    规则 LLM 本就逐句工作，因此切分长消息后其行逐字节相同；短消息从不进入切分器（两种阈值下都只有一行）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions comparing the add() envelopes of two facades that differ only in segment_max_chars.
        比较两个仅 segment_max_chars 不同的门面的 add() 信封的断言。

    State effect / 状态影响：
        None outside four in-memory stores.
        除四个内存存储外无副作用。
    """
    rule = {"provider": "rule", "config": {}}
    text = _long_message()
    cut, whole = _facade(rule, segment_max_chars=400), _facade(rule, segment_max_chars=10**9)
    r_cut = cut.add(text, user_id="alice", timestamp="2026-06-15T09:00:00Z")
    r_whole = whole.add(text, user_id="alice", timestamp="2026-06-15T09:00:00Z")
    assert r_cut == r_whole and len([r for r in r_cut["results"] if r["event"] == "ADD"]) >= 3
    assert _new_message_rows(cut) == [len(segment(text, max_chars=400))] and _new_message_rows(whole) == [1]
    cut.close()
    whole.close()

    short = "I live in Berlin. I work at Stripe."
    a, b = _facade(rule, segment_max_chars=400), _facade(rule, segment_max_chars=10**9)
    assert a.add(short, user_id="alice") == b.add(short, user_id="alice")
    assert _new_message_rows(a) == [1] == _new_message_rows(b)
    a.close()
    b.close()
