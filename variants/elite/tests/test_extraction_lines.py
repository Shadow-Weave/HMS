"""Lines extraction output contract: the LLM answers with tab-separated OBS / EP / DEC text, ingest.lines parses it, the rest is derived exactly as in compact, and the full and compact contracts are untouched.
行式抽取输出契约：LLM 以 tab 分隔的 OBS / EP / DEC 文本回答，ingest.lines 解析，其余与 compact 完全一样推导，完整与紧凑契约原样不动。
"""

# Purpose (EN): Pin IngestConfig.extraction_output='lines' end to end — the parser's tolerance rules (well-formed,
# truncated last line, stray prose and fences, a missing tab, an unknown type letter, a DEC line out of range, an EP
# with k=), its numbering (id = the OBS line the model wrote, surviving a dropped, wrapped or prefix-only line) and
# its no-section warning, a scripted string reply through HMS.add (anchors, derived axes, a DEC REVISE creating a
# revision edge, the verdict guard still applying, no json mode on the call, a DEC target that survives a dropped
# line, an EP with locally derived entity / topic / b columns), the rule LLM's lines facade against the full facade
# on every chat fixture (nine identical tables, the derived columns pinned one by one), config validation (single
# mode only, bilingual message), prompt registration and the statement rules kept verbatim from the full prompt,
# the DeepSeek provider sending no response_format, and the benchmark harness's lines bookkeeping (source pins).
# 目的（中）：端到端钉住 IngestConfig.extraction_output='lines'——解析器的容错规则（规范输入、被截断的末行、散文与
# 围栏、缺 tab、未知类型字母、越界的 DEC 行、带 k= 的 EP）、编号（id = 模型所写的 OBS 行号，丢行 / 折行 / 只有前缀的行
# 之后仍然成立）与无分节警告、脚本化字符串回复经 HMS.add（锚点、推导轴、DEC REVISE 生成修订边、裁决守卫仍然生效、
# 调用不带 json 模式、DEC 目标在丢行之后仍然正确、EP 的实体 / 主题 / b 列在本地推导）、规则 LLM 的行式门面与完整门面
# 在每个聊天夹具上的对照（九张表相同，推导列逐一钉住）、配置校验（仅 single 模式，双语消息）、提示词注册与逐字保留自
# 完整提示词的陈述规则、DeepSeek 提供者不发送 response_format，以及基准脚本的行式记账（源码钉住）。

# Input (EN): Hand-written line replies, the chat_* fixtures, ScriptedLLM / RuleLLM, a fake openai SDK, the conftest clock / id helpers.
# 输入（中）：手写的行式回复、chat_* 夹具、ScriptedLLM / RuleLLM、替身 openai SDK、conftest 的时钟 / id 辅助。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory SQLite stores.
# 状态影响（中）：除内存 SQLite 存储外无副作用。

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

import pydantic
import pytest

from hms_elite import HMS, HMSConfig
from hms_elite.configs import prompts
from hms_elite.configs.base import IngestConfig
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.ingest import derive, extraction
from hms_elite.ingest.lines import NO_SECTION_MARK, parse_lines
from hms_elite.ingest.pipeline import EXTRACTION_OUTPUTS, Ingestor
from hms_elite.llms import deepseek as deepseek_module
from hms_elite.memory.async_main import AsyncHMS
from hms_elite.memory.storage import StateStore
from hms_elite.models import Scope
from hms_elite.utils.factory import LlmFactory
from tests.conftest import EVENTS_DIR, FrozenClock, SequentialIds, load_events
from tests.test_providers_extra import _fake_openai

CHAT_FIXTURES = sorted(path.stem for path in EVENTS_DIR.glob("chat_*.json"))
LINES_LOGGER = "hms_elite.ingest.lines"
PIPELINE_LOGGER = "hms_elite.ingest.pipeline"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = "2026-06-15"
STAMP = "2026-06-15T10:04:00Z"
STAMP_ISO = "2026-06-15T10:04:00.000000+00:00"
SCOPE = Scope(user_id="alice", run_id="run-lines")
# sha256 of the full and compact prompts before the lines contract was added: both must stay byte-identical.
# 加入行式契约之前完整与紧凑提示词的 sha256：两者都必须逐字节不变。
PINNED_PROMPT_SHA256 = {
    "MEMORY_EXTRACTION_PROMPT": "951ff8c977062e44a48aa029dd23055461d1ab7e37502cdaf7fb2340d106a526",
    "OBSERVATION_EXTRACTION_PROMPT": "e3baca1270fd7062633f7b7deba0be6918aa879609550d37bdd1971d9133742e",
    "MEMORY_EXTRACTION_COMPACT_PROMPT": "517cf7b0fb69c8ea52f7dc0a7764028b931e19899bae796f37f8c4d2b1a2aa92",
    "OBSERVATION_EXTRACTION_COMPACT_PROMPT": "272f500f95fb19dcea02e292fdaacc1900b16c6175bb0c539fce0172fa70d844",
}
# Statement rules the lines prompt must carry verbatim from the full prompt (the text-writing rules that scored
# 61.3% against the compact prompt's 56.4% on memory rows).
# 行式提示词必须逐字保留自完整提示词的陈述规则（这些写文本的规则在记忆行上得 61.3%，紧凑提示词得 56.4%）。
KEY_SENTENCES = (
    "Observations: one per self-contained statement, third person, 5–80 words; type by intent (preference_event,\n"
    "  constraint, habit, temporal_fact, fact)",
    "a supersession cue (now, switched\n  to, no longer, instead, actually)",
    'Bad: "I like oat-milk lattes"\n  Good: "User likes oat-milk lattes in the morning."',
)
WELL_FORMED = (
    "OBS\n"
    "0\tC\tKatrina must not modify the billing state machine.\n"
    "1\tT\tAssistant scheduled the dentist visit for Sarah.\t@2026-06-20T09:30:00+00:00\n"
    "2\tH\tUser usually drinks black coffee after lunch.\n"
    "EP\n"
    "0\tu=the webhook kept timing out\ta=patched retry.ts to add backoff\ty=billing_e2e failed\tr=negative\tk=never touch billing;keep idempotency keys\n"
    "DEC\n"
    "2\tREVISE\t1\tsupersession cue: 'now'\n"
    "3\tNONE\t0\trestatement\n"
)
LINES_MESSAGES = [
    {"role": "user", "content": "Katrina said: do not touch the billing state machine."},
    {"role": "assistant", "content": "I scheduled the dentist visit for Sarah on June 20 at 9:30."},
    {"role": "user", "name": "Bob", "content": "I usually drink black coffee after lunch."},
]


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
def _facade(output: str, llm: dict | None = None, **ingest: Any) -> HMS:
    """An offline in-memory HMS (memory index, frozen clock, sequential ids) in one extraction output.
    某种抽取输出下的离线内存 HMS（memory 索引、冻结时钟、顺序 id）。
    """
    store = StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds())
    config: dict[str, Any] = {
        "vector_store": {"provider": "memory"},
        "store": {"path": ":memory:"},
        "ingest": {"extraction_output": output, **ingest},
    }
    if llm is not None:
        config["llm"] = llm
    return HMS(HMSConfig(**config), store=store)


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


# Tables whose rows must be identical between a full and a lines facade fed the same fixture (§6.6 invariants).
# 同一夹具下完整门面与行式门面必须逐行相同的表（§6.6 不变量）。
IDENTICAL_TABLES = ("atom_anchor", "atom_entity", "atom_topic", "entities", "history", "graph_nodes", "graph_edges", "edge_witnesses", "eps")
# The columns the compact derivation owns in the two remaining row tables; every other column must be identical and
# these are pinned one by one in the identity test (§6.6 invariants).
# 其余两张行表中由紧凑推导所拥有的列；其它每一列都必须相同，这些列在一致性测试中逐一钉住（§6.6 不变量）。
DERIVED_COLUMNS = {
    "observations": ("confidence", "relation", "supersedes_hint_ids_json", "entity_mentions_json"),
    "memory_atoms": ("confidence", "actor_id", "content_json"),
}


def _tables(hms: HMS) -> dict[str, list]:
    """Every row of IDENTICAL_TABLES, the (text, subject, predicate, objects) tuple of every atom, and the observations /
    memory_atoms rows split into their non-derived columns (`<table>_modulo`) and their derived ones (`<table>_derived`).
    IDENTICAL_TABLES 的全部行、每个原子的 (text, subject, predicate, objects) 四元组，以及拆成非推导列（`<table>_modulo`）
    与推导列（`<table>_derived`）的 observations / memory_atoms 行。
    """
    conn = hms.store._conn
    out: dict[str, list] = {}
    for table in IDENTICAL_TABLES:
        cur = conn.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
    for table, derived in DERIVED_COLUMNS.items():
        cur = conn.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        out[f"{table}_modulo"] = [{k: v for k, v in row.items() if k not in derived} for row in rows]
        out[f"{table}_derived"] = [{k: row[k] for k in derived} for row in rows]
    out["atom_quads"] = [
        (text, subject, predicate, json.loads(content).get("objects"))
        for text, subject, predicate, content in conn.execute(
            "SELECT text, subject, predicate, content_json FROM memory_atoms ORDER BY atom_id"
        )
    ]
    return out


def _warnings(caplog: pytest.LogCaptureFixture, logger: str = LINES_LOGGER) -> list[str]:
    """The WARNING messages of one logger captured so far.
    到目前为止捕获的某个 logger 的 WARNING 消息。
    """
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# (1) Parser
# (1) 解析器
# ---------------------------------------------------------------------------
def test_parse_lines_well_formed_reply():
    """A well-formed reply parses into compact-shaped observations (id = 1-based line number, letter → type, `when` from @date), an EP with its k= constraints and decisions keyed by OBS line number.
    规范的回复解析为紧凑形状的观察（id = 1 起行号，字母 → 类型，`when` 取自 @date）、带 k= 约束的 EP，以及以 OBS 行号为键的决策。

    Input / 输入：
        WELL_FORMED (three OBS lines, one EP line with k=, two DEC lines).
        WELL_FORMED（三行 OBS、一行带 k= 的 EP、两行 DEC）。

    Output / 输出：
        Assertions on the three roots.
        对三个根的断言。

    State effect / 状态影响：
        None.
        无。
    """
    out = parse_lines(WELL_FORMED)
    assert list(out) == ["observations", "eps", "decisions"]
    assert out["observations"] == [
        {"id": "1", "event_idx": 0, "text": "Katrina must not modify the billing state machine.", "type": "constraint"},
        {"id": "2", "event_idx": 1, "text": "Assistant scheduled the dentist visit for Sarah.", "type": "temporal_fact",
         "when": "2026-06-20T09:30:00+00:00"},
        {"id": "3", "event_idx": 2, "text": "User usually drinks black coffee after lunch.", "type": "habit"},
    ]  # fmt: skip
    assert out["eps"] == [
        {"id": "0", "event_idx": 0, "u": "the webhook kept timing out", "b": "", "a": "patched retry.ts to add backoff",
         "y": "billing_e2e failed", "r": "negative",
         "kappa": {"constraints": ["never touch billing", "keep idempotency keys"]}},
    ]  # fmt: skip
    assert out["decisions"] == [
        {"id": "2", "decision": "REVISE", "target_id": "1", "reason": "supersession cue: 'now'"},
        {"id": "3", "decision": "NONE", "target_id": "0", "reason": "restatement"},
    ]
    # Every letter of the vocabulary and a full type name (mapped by its initial); a lone OBS header is empty.
    # 词表的每个字母与完整类型名（按首字母映射）；只有 OBS 标题时为空。
    letters = "\n".join(f"0\t{letter}\tstatement {letter}" for letter in "FPHCT") + "\n0\tpreference_event\tnamed\n"
    types = [o["type"] for o in parse_lines("OBS\n" + letters)["observations"]]
    assert types == ["fact", "preference_event", "habit", "constraint", "temporal_fact", "preference_event"]
    assert parse_lines("OBS\n") == {"observations": [], "eps": [], "decisions": []}
    assert parse_lines("") == {"observations": [], "eps": [], "decisions": []} and parse_lines(None)["observations"] == []


def test_parse_lines_tolerates_prose_fences_missing_tab_and_unknown_letter(caplog):
    """Stray prose before OBS and code fences are ignored, a `# OBS:` header and an implicit OBS (a tabbed line before any header) open the section, a line without a tab continues the previous statement (no new line number), a `@date` glued after a space is read, an unknown letter reads as fact, and a tab-less first line is skipped with one warning.
    OBS 之前的散文与代码围栏被忽略，`# OBS:` 标题与隐式 OBS（任何标题之前的 tab 分隔行）都能开启分节，无 tab 的行接续上一条陈述（不占新行号），以空格粘在末尾的 `@date` 也能读到，未知字母按 fact 读取，开头无 tab 的行记一条警告并跳过。

    Input / 输入：
        caplog: pytest log capture on the lines logger.
        caplog：行式解析器 logger 的 pytest 日志捕获。

    Output / 输出：
        Assertions on the parsed items and the warning count.
        对解析条目与警告数的断言。

    State effect / 状态影响：
        None.
        无。
    """
    caplog.set_level(logging.DEBUG, logger=LINES_LOGGER)
    reply = (
        "Here are the memories you asked for:\n"
        "```text\n"
        "# OBS:\n"
        "0\tP\tUser likes black coffee.\n"
        "User no longer drinks oat-milk lattes.\n"
        "1\tX\tSarah joined the billing team in March. @2026-03-01\n"
        "H\tUser usually runs at 7am.\n"
        "```\n"
    )
    out = parse_lines(reply)
    assert [(o["id"], o["event_idx"], o["type"], o["text"], o.get("when")) for o in out["observations"]] == [
        ("1", 0, "preference_event", "User likes black coffee. User no longer drinks oat-milk lattes.", None),
        ("2", 1, "fact", "Sarah joined the billing team in March.", "2026-03-01"),
        ("3", 1, "habit", "User usually runs at 7am.", None),
    ]
    assert out["eps"] == [] and out["decisions"] == [] and _warnings(caplog) == []
    assert any("unknown type letter 'X'" in r.getMessage() for r in caplog.records if r.name == LINES_LOGGER)
    # No header at all: a tabbed `<idx> <T> ...` line opens OBS implicitly; a bare statement first is skipped once.
    # 完全没有标题：tab 分隔的 `<idx> <T> ...` 行隐式开启 OBS；开头的裸陈述只跳过并警告一次。
    implicit = parse_lines("2\tF\tUser works at Acme.\n1\tSecond statement without a type.\n")
    assert [(o["event_idx"], o["type"], o["text"]) for o in implicit["observations"]] == [
        (2, "fact", "User works at Acme."), (1, "fact", "Second statement without a type."),
    ]  # fmt: skip
    caplog.clear()
    orphan = parse_lines("OBS\nA statement with no index at all.\n0\tF\tUser works at Acme.\n")
    assert [o["text"] for o in orphan["observations"]] == ["User works at Acme."]
    assert len(_warnings(caplog)) == 1 and "no event_idx and no previous line" in _warnings(caplog)[0]


def test_parse_lines_drops_a_truncated_trailing_line(caplog):
    """A last line without a trailing newline and with fewer than two tabs is a cut-off reply and is dropped with one warning; the same line with a newline is a continuation, and a complete last line without a newline is kept.
    没有尾随换行且少于两个 tab 的末行是被截断的回复，记一条警告并丢弃；同一行带换行时是续行，完整的末行没有换行也保留。

    Input / 输入：
        caplog: pytest log capture on the lines logger.
        caplog：行式解析器 logger 的 pytest 日志捕获。

    Output / 输出：
        Assertions on the observation texts and the warning.
        对观察文本与警告的断言。

    State effect / 状态影响：
        None.
        无。
    """
    caplog.set_level(logging.WARNING, logger=LINES_LOGGER)
    cut = "OBS\n0\tP\tUser likes black coffee.\n0\tF"
    assert [o["text"] for o in parse_lines(cut)["observations"]] == ["User likes black coffee."]
    assert len(_warnings(caplog)) == 1 and "truncated" in _warnings(caplog)[0]
    caplog.clear()
    # With a newline the same fragment is a line whose statement is a lone letter (nothing to store, no warning); a
    # cut-off statement that still has its two tabs is kept as it stands — witness and type intact, text short.
    # 带换行时同一片段是陈述只剩一个字母的行（无可存内容，不记警告）；被截断但仍有两个 tab 的陈述原样保留——
    # 见证与类型完好，只是文本偏短。
    assert [o["text"] for o in parse_lines(cut + "\n")["observations"]] == ["User likes black coffee."] and not _warnings(caplog)
    assert [o["text"] for o in parse_lines(cut + "\tUser works at Ac")["observations"]] == ["User likes black coffee.", "User works at Ac"]
    assert [o["text"] for o in parse_lines("OBS\n0\tP\tUser likes black coffee.\n0\tF\tUser works at Acme.")["observations"]] == [
        "User likes black coffee.", "User works at Acme.",
    ]  # fmt: skip
    # A section header or a fence as the last line is not a truncation.
    # 末行是分节标题或围栏时不算截断。
    assert parse_lines("OBS\n0\tP\tUser likes black coffee.\nDEC")["observations"] and not _warnings(caplog)
    assert parse_lines("```\nOBS\n0\tP\tUser likes black coffee.\n```")["observations"] and not _warnings(caplog)


def test_parse_lines_dec_out_of_range_and_ep_shapes(caplog):
    """A DEC line naming an OBS number outside 1..n (or not a number, or with fewer than three fields) is skipped with a warning while the others stay; an EP line keeps b=, drops fields without a slot prefix, defaults r to unknown and needs at least one slot.
    DEC 行点名的 OBS 行号不在 1..n 内（或不是数字、或不足三个字段）时记警告并跳过，其余保留；EP 行保留 b=、丢掉没有槽位前缀的字段、r 默认 unknown，且至少要有一个槽位。

    Input / 输入：
        caplog: pytest log capture on the lines logger.
        caplog：行式解析器 logger 的 pytest 日志捕获。

    Output / 输出：
        Assertions on the decisions, the EPs and the warnings.
        对决策、EP 与警告的断言。

    State effect / 状态影响：
        None.
        无。
    """
    caplog.set_level(logging.WARNING, logger=LINES_LOGGER)
    reply = (
        "OBS\n0\tF\tUser works at Acme.\n0\tP\tUser likes tea.\n"
        "EP\n0\tu=tests failed\tb=CI is slow\ta=reran the suite\tstray field\ty=passed\n0\n"
        "DEC\n5\tNONE\t0\tout of range\n2\tREVISE\t1\tcue\nx\tNONE\t0\tnot a number\n1\tNONE\n0\tCONFLICT\t0\tzero\n"
    )
    out = parse_lines(reply)
    assert out["decisions"] == [{"id": "2", "decision": "REVISE", "target_id": "1", "reason": "cue"}]
    assert out["eps"] == [
        {"id": "0", "event_idx": 0, "u": "tests failed", "b": "CI is slow", "a": "reran the suite", "y": "passed",
         "r": "unknown", "kappa": {"constraints": []}},
    ]  # fmt: skip
    warnings = _warnings(caplog)
    assert len(warnings) == 5
    assert sum("outside 1..2" in w for w in warnings) == 3 and sum("incomplete" in w for w in warnings) == 1
    assert sum("has no slots" in w for w in warnings) == 1


def test_parse_lines_numbers_observations_by_the_obs_line_the_model_wrote(caplog):
    """Every tabbed OBS line consumes one line number whether or not it stores anything, so a DEC line still names the line the model wrote after a dropped (empty-statement) line; a tab-less line is a continuation that consumes no number — it fills an immediately preceding prefix-only line or extends the previous statement — and a DEC line naming a line that stored nothing is skipped with a warning.
    每个带 tab 的 OBS 行都占用一个行号，无论是否存下内容，因此丢掉一行（空陈述）之后 DEC 行仍然指向模型所写的那一行；无 tab 的行是不占行号的续行——填入紧接其前的只有前缀的行，或延续上一条陈述——DEC 行点名没有存下观察的行时记警告并跳过。

    Input / 输入：
        caplog: pytest log capture on the lines logger.
        caplog：行式解析器 logger 的 pytest 日志捕获。

    Output / 输出：
        Assertions on the observation ids / texts, the decisions and the warning.
        对观察 id / 文本、决策与警告的断言。

    State effect / 状态影响：
        None.
        无。
    """
    caplog.set_level(logging.WARNING, logger=LINES_LOGGER)
    # A dropped line keeps its number: DEC 3 / 4 still name Lucia / Shure; DEC 2 names the empty line and is skipped.
    # 被丢弃的行保留行号：DEC 3 / 4 仍指向 Lucia / Shure；DEC 2 指向空行，跳过。
    dropped = parse_lines(
        "OBS\n0\tF\tJake said hello.\n1\tF\t\n2\tF\tLucia laughed.\n3\tF\tShure sang.\n"
        "DEC\n3\tNONE\t0\tr\n4\tREVISE\t1\tc\n2\tNONE\t0\tempty line\n"
    )
    assert [(o["id"], o["text"]) for o in dropped["observations"]] == [("1", "Jake said hello."), ("3", "Lucia laughed."), ("4", "Shure sang.")]
    assert [(d["id"], d["decision"]) for d in dropped["decisions"]] == [("3", "NONE"), ("4", "REVISE")]
    assert len(_warnings(caplog)) == 1 and "OBS line 2, which stored no observation" in _warnings(caplog)[0]
    caplog.clear()
    # A wrapped statement is one observation with one number; the DEC meant for line 2 reaches the green-tea line.
    # 折行的陈述是一条观察、一个行号；本意指向第 2 行的 DEC 到达绿茶那一行。
    wrapped = parse_lines("OBS\n0\tF\tUser has two dogs, both adopted\nfrom the shelter last year.\n0\tP\tUser likes green tea.\nDEC\n2\tREVISE\t0\tcue\n")
    assert [(o["id"], o["text"]) for o in wrapped["observations"]] == [
        ("1", "User has two dogs, both adopted from the shelter last year."), ("2", "User likes green tea."),
    ]  # fmt: skip
    assert wrapped["decisions"] == [{"id": "2", "decision": "REVISE", "target_id": "0", "reason": "cue"}]
    # A prefix-only line takes the next bare line as its statement (its event_idx, type and number are kept).
    # 只有前缀的行把下一条裸行作为自己的陈述（保留它的 event_idx、类型与行号）。
    filled = parse_lines("OBS\n1\tH\t\nUser runs at 7am. @2026-03-01\n0\tP\tUser likes green tea.\nDEC\n1\tNONE\t0\tx\n")
    assert [(o["id"], o["event_idx"], o["type"], o["text"], o.get("when")) for o in filled["observations"]] == [
        ("1", 1, "habit", "User runs at 7am.", "2026-03-01"), ("2", 0, "preference_event", "User likes green tea.", None),
    ]  # fmt: skip
    assert [d["id"] for d in filled["decisions"]] == ["1"] and not _warnings(caplog)
    # The rule renderer's numbering (line = positional id + 1) is the same rule: every rendered line stores.
    # 规则渲染器的编号（行号 = 位置 id + 1）是同一条规则：渲染出的每一行都会存下。
    assert [o["id"] for o in parse_lines(WELL_FORMED)["observations"]] == ["1", "2", "3"]


def test_parse_lines_warns_once_when_the_reply_opens_no_section(caplog):
    """A reply that opens no section at all — prose, a JSON object, an empty string — logs exactly one warning containing NO_SECTION_MARK (the harness's unparsable-reply signal) and parses to empty roots; a bare `OBS` header is an empty extraction with no warning.
    完全没有开启任何分节的回复——散文、JSON 对象、空串——恰好记一条含 NO_SECTION_MARK（基准脚本的不可用回复信号）的警告并解析为空的各根；只有 `OBS` 标题是不记警告的空抽取。

    Input / 输入：
        caplog: pytest log capture on the lines logger.
        caplog：行式解析器 logger 的 pytest 日志捕获。

    Output / 输出：
        Assertions on the warnings and the parsed roots.
        对警告与解析结果的断言。

    State effect / 状态影响：
        None.
        无。
    """
    caplog.set_level(logging.WARNING, logger=LINES_LOGGER)
    assert NO_SECTION_MARK == "reply has no OBS / EP / DEC section"
    for reply in ("Sorry, I cannot help with that request.\n", '{"observations": [{"id": 0, "text": "x"}]}\n', "", None):
        caplog.clear()
        assert parse_lines(reply) == {"observations": [], "eps": [], "decisions": []}
        marked = [w for w in _warnings(caplog) if NO_SECTION_MARK in w]
        assert len(marked) == 1 and marked[0].startswith("memory_extraction_lines: ") and "nothing parsed" in marked[0]
    caplog.clear()
    assert parse_lines("OBS\n") == {"observations": [], "eps": [], "decisions": []} and not _warnings(caplog)
    assert parse_lines("```\nOBS\n0\tF\tUser works at Acme.\n```\n")["observations"] and not _warnings(caplog)


# ---------------------------------------------------------------------------
# (2) A scripted string reply through HMS.add
# (2) 脚本化字符串回复经 HMS.add
# ---------------------------------------------------------------------------
def test_lines_reply_through_hms_add_derives_axes_witnesses_and_revisions(caplog):
    """A scripted lines reply yields atoms with derived fields and all seven axes, each carrying exactly its witness anchor, the leading proper name reaching entity_mentions and actor_id (message `name` still wins), the call made without json mode; a later DEC REVISE creates the revision edge (UPDATE row, `revises`) while a NONE that is not a restatement is downgraded to ADD by the verdict guard, and a missing DEC line is ADD without a warning.
    脚本化行式回复产出带推导字段与七个轴的原子，每个恰好带自己的见证锚点，开头的专有名到达 entity_mentions 与 actor_id（消息 `name` 仍优先），调用不带 json 模式；随后的 DEC REVISE 生成修订边（UPDATE 行、`revises`），而非复述的 NONE 被裁决守卫降级为 ADD，缺少 DEC 行的观察是 ADD 且不记警告。

    Input / 输入：
        caplog: pytest log capture (DEBUG on the pipeline logger, WARNING on extraction).
        caplog：pytest 日志捕获（流水线 logger 的 DEBUG、抽取的 WARNING）。

    Output / 输出：
        Assertions on the add() rows, the atoms, the anchors, the call kwargs and the guard's DEBUG line.
        对 add() 行、原子、锚点、调用参数与守卫 DEBUG 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    caplog.set_level(logging.WARNING, logger="hms_elite.ingest.extraction")
    first = (
        "OBS\n"
        "0\tC\tKatrina must not modify the billing state machine.\n"
        "1\tT\tAssistant scheduled the dentist visit for Sarah.\t@2026-06-20T09:30:00+00:00\n"
        "2\tH\tUser usually drinks black coffee after lunch.\n"
    )
    second = "OBS\n0\tP\tUser likes black coffee.\n0\tF\tUser works at Acme.\nDEC\n1\tREVISE\t0\tsupersession cue: 'switched to'\n"
    hms = _facade("lines", llm={"provider": "scripted", "config": {"responses": [first, second]}})
    rows = hms.add(LINES_MESSAGES, user_id="alice", run_id="run-lines", timestamp=STAMP)["results"]
    assert [c["task"] for c in hms.llm.calls] == ["memory_extraction_lines"] and hms.llm.calls[0]["kwargs"] == {}
    assert [r["event"] for r in rows] == ["ADD", "ADD", "ADD"]
    constraint, temporal, habit = rows
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
    assert temporal["valid_from"] == "2026-06-20T09:30:00.000000+00:00" and all(r["valid_to"] is None for r in rows)
    assert constraint["hms"]["entity_ids"] == ["katrina"] and temporal["hms"]["entity_ids"] == ["sarah"]
    assert constraint["hms"]["topic_tags"] == ["billing", "constraint"] and "coffee" in habit["hms"]["topic_tags"]
    assert [r["hms"]["sentiment"] for r in rows] == ["neutral", "neutral", "neutral"]
    assert [r["scale_level"] for r in rows] == ["state", "event", "state"]
    assert [r["hms"]["obs_type"] for r in rows] == ["constraint", "temporal_fact", "habit"]
    # Witness: each atom carries exactly the anchor of the message its event_idx named.
    # 见证：每个原子恰好带 event_idx 所指消息的锚点。
    by_text = {
        str(ev.payload["text"]): hms.store.anchor_for_event(ev.event_id).anchor_id
        for ev in hms.store.last_events(SCOPE, object_types=["message"], limit=10)
    }
    assert [r["anchor_ids"] for r in rows] == [[by_text[m["content"]]] for m in LINES_MESSAGES]
    # The lines format has no speaker slot: the leading proper name is the speaker (§6.6), the message name wins.
    # 行式格式没有说话人槽位：开头的专有名即说话人（§6.6），消息的 name 优先。
    assert constraint["actor_id"] == "Katrina" and "actor_id" not in temporal and habit["actor_id"] == "Bob"
    # Second batch: one OBS line with a DEC REVISE against candidate '0' (the nearest atom, chosen by the LLM among
    # the candidates) and one without a DEC line, which is ADD by definition — no "no decision" warning.
    # 第二批：一行 OBS 带指向候选 '0'（最近的原子，由 LLM 在候选中选定）的 DEC REVISE，另一行没有 DEC 行，
    # 按定义就是 ADD——没有 "no decision" 警告。
    later = hms.add(
        [{"role": "user", "content": "Actually I switched to black coffee now. I work at Acme."}],
        user_id="alice", run_id="run-lines", timestamp="2026-06-16T10:04:00Z",
    )["results"]  # fmt: skip
    events = {r["memory"]: r["event"] for r in later}
    assert events["User works at Acme."] == "ADD"
    coffee = next(r for r in later if r["memory"] == "User likes black coffee.")
    assert coffee["event"] in {"ADD", "UPDATE"}
    assert not any("no decision" in r.getMessage() for r in caplog.records)
    hms.close()


def test_lines_dec_revise_creates_a_revision_edge_and_the_guard_still_applies(caplog):
    """Against a known candidate a DEC REVISE from a lines reply revises the target (UPDATE row with `revises`, one history UPDATE, a revision edge), a DEC NONE that restates it is honoured, and a DEC NONE on a different statement is downgraded to ADD by the verdict guard.
    对已知候选，行式回复中的 DEC REVISE 修订目标（带 `revises` 的 UPDATE 行、一条 UPDATE 历史、一条修订边），复述的 DEC NONE 被采纳，而针对不同陈述的 DEC NONE 被裁决守卫降级为 ADD。

    Input / 输入：
        caplog: pytest log capture (DEBUG on the pipeline logger).
        caplog：pytest 日志捕获（流水线 logger 的 DEBUG）。

    Output / 输出：
        Assertions on the rows, the history table, the graph edges and the guard's DEBUG line.
        对行、历史表、图边与守卫 DEBUG 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seed = "OBS\n0\tP\tUser likes oat-milk lattes in the morning.\n"
    revise = "OBS\n0\tP\tUser likes black coffee.\nDEC\n1\tREVISE\t0\tsupersession cue: 'switched to'\n"
    restate = "OBS\n0\tP\tUser likes black coffee.\n0\tF\tUser has two dogs.\nDEC\n1\tNONE\t0\trestatement\n2\tNONE\t0\tduplicate\n"
    hms = _facade("lines", llm={"provider": "scripted", "config": {"responses": [seed, revise, restate]}})
    old = hms.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")["results"][0]
    new = hms.add("Actually I switched to black coffee now.", user_id="alice", timestamp="2026-04-20T08:00:00Z")["results"][0]
    assert old["event"] == "ADD" and new["event"] == "UPDATE" and new["revises"] == old["atom_id"]
    assert hms.store.get_atom(old["atom_id"]).valid_to is not None
    conn = hms.store._conn
    assert [row[0] for row in conn.execute("SELECT event FROM history ORDER BY rowid")] == ["ADD", "UPDATE"]
    assert conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'revision'").fetchone()[0] == 1
    # The guard: a NONE restating the black-coffee atom is honoured (NONE row), a NONE on a new fact is stored as ADD.
    # 守卫：复述黑咖啡原子的 NONE 被采纳（NONE 行），针对新事实的 NONE 按 ADD 存入。
    third = hms.add("I still like black coffee. I have two dogs.", user_id="alice", timestamp="2026-05-01T08:00:00Z")["results"]
    assert {r["memory"]: r["event"] for r in third} == {"User likes black coffee.": "NONE", "User has two dogs.": "ADD"}
    downgraded = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER and "downgraded to ADD" in r.getMessage()]
    assert len(downgraded) == 1 and "'User has two dogs.'" in downgraded[0]
    assert [c["task"] for c in hms.llm.calls] == ["memory_extraction_lines"] * 3
    assert all(c["kwargs"] == {} for c in hms.llm.calls)
    hms.close()


def test_lines_dec_targets_survive_a_dropped_and_a_wrapped_line(caplog):
    """Through HMS.add a DEC line keeps naming the OBS line the model wrote: after an empty-statement line the REVISE on line 3 reaches 'two dogs' (and is downgraded by the guard) instead of 'green tea' (which shares the target's subject and predicate and would have revised the oat-milk atom), so the seed atom stays valid; a wrapped statement is stored as one atom, never as a fragment.
    经 HMS.add，DEC 行始终指向模型所写的 OBS 行：空陈述行之后，第 3 行的 REVISE 到达 'two dogs'（被守卫降级），而不是 'green tea'（与目标同主语同谓词，本会修订燕麦拿铁原子），种子原子因此保持有效；折行的陈述存为一条原子，绝不存为片段。

    Input / 输入：
        caplog: pytest log capture (DEBUG on the pipeline logger).
        caplog：pytest 日志捕获（流水线 logger 的 DEBUG）。

    Output / 输出：
        Assertions on the add() rows, the seed atom's validity, the history table and the stored texts.
        对 add() 行、种子原子有效期、历史表与所存文本的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    caplog.set_level(logging.DEBUG, logger=PIPELINE_LOGGER)
    seed = "OBS\n0\tP\tUser likes oat-milk lattes in the morning.\n"
    dropped = (
        "OBS\n0\tP\tUser likes black coffee.\n0\tF\t\n0\tF\tUser has two dogs.\n0\tP\tUser likes green tea.\n"
        "DEC\n3\tREVISE\t0\tmeant for the dogs line\n4\tNONE\t0\tmeant for the green-tea line\n"
    )
    wrapped = "OBS\n0\tF\tUser has a cat, adopted\nlast spring.\n0\tP\tUser likes green tea.\n"
    hms = _facade("lines", llm={"provider": "scripted", "config": {"responses": [seed, dropped, wrapped]}})
    old = hms.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")["results"][0]
    rows = hms.add("Black coffee now. I have two dogs. I like green tea.", user_id="alice", timestamp="2026-04-20T08:00:00Z")["results"]
    assert [(r["memory"], r["event"]) for r in rows] == [
        ("User likes black coffee.", "ADD"), ("User has two dogs.", "ADD"), ("User likes green tea.", "ADD"),
    ]  # fmt: skip
    assert hms.store.get_atom(old["atom_id"]).valid_to is None
    downgraded = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER and "downgraded to ADD" in r.getMessage()]
    assert len(downgraded) == 2 and "'User has two dogs.'" in downgraded[0] and "'User likes green tea.'" in downgraded[1]
    later = hms.add("I have a cat, adopted last spring. Green tea again.", user_id="alice", timestamp="2026-05-01T08:00:00Z")["results"]
    assert [r["memory"] for r in later] == ["User has a cat, adopted last spring.", "User likes green tea."]
    # The repeated green-tea statement is the pipeline's own restatement (no new atom); no atom was ever revised.
    # 重复的绿茶陈述由流水线自己识别为复述（不新增原子）；从未有原子被修订。
    texts = [row[0] for row in hms.store._conn.execute("SELECT text FROM memory_atoms ORDER BY atom_id")]
    assert texts == [
        "User likes oat-milk lattes in the morning.", "User likes black coffee.", "User has two dogs.", "User likes green tea.",
        "User has a cat, adopted last spring.",
    ]  # fmt: skip
    assert "UPDATE" not in {row[0] for row in hms.store._conn.execute("SELECT event FROM history")}
    hms.close()


def test_lines_ep_stores_derived_entities_topics_and_background():
    """expand_lines_ep fills what the EP wire format does not carry — entity_mentions and topic_tags from '<a> <y> <u>', b from the k= constraints when empty, nothing overwritten when given — and a scripted EP line through HMS.add stores those columns in the eps table.
    expand_lines_ep 补齐 EP 线上格式不携带的字段——entity_mentions 与 topic_tags 取自 '<a> <y> <u>'，b 为空时取自 k= 约束，已给出的不覆盖——脚本化 EP 行经 HMS.add 把这些列存入 eps 表。

    Input / 输入：
        The WELL_FORMED EP item; the chat_action_closure fixture's first message with a scripted lines reply.
        WELL_FORMED 的 EP 条目；chat_action_closure 夹具的第一条消息配脚本化行式回复。

    Output / 输出：
        Assertions on the expanded item and the eps row.
        对扩展后的条目与 eps 行的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    item = parse_lines(WELL_FORMED)["eps"][0]
    expanded = derive.expand_lines_ep(item, extractor=None)
    assert expanded["b"] == "never touch billing ; keep idempotency keys"
    assert expanded["entity_mentions"] == ["retry.ts", "billing_e2e"] and expanded["topic_tags"] == ["engineering", "billing"]
    assert {k: v for k, v in expanded.items() if k not in ("b", "entity_mentions", "topic_tags")} == {k: v for k, v in item.items() if k != "b"}
    given = derive.expand_lines_ep({**item, "b": "CI is slow", "entity_mentions": ["x"], "topic_tags": ["t"]}, extractor=None)
    assert (given["b"], given["entity_mentions"], given["topic_tags"]) == ("CI is slow", ["x"], ["t"])
    assert derive.expand_lines_ep({"event_idx": 0, "u": "", "a": "", "y": "", "kappa": {}}, extractor=None)["entity_mentions"] == []
    message = load_events("chat_action_closure").messages[0]
    reply = (
        "OBS\n0\tF\tUser patched retry.ts to add backoff.\n"
        "EP\n0\tu=the webhook kept timing out\ta=patched retry.ts to add backoff\ty=billing_e2e failed\tr=negative\tk=never touch billing\n"
    )
    hms = _facade("lines", llm={"provider": "scripted", "config": {"responses": [reply]}})
    hms.add([{"role": message["role"], "content": message["content"]}], user_id="alice", run_id="run-ep", timestamp=message.get("timestamp") or STAMP)
    eps = [tuple(row) for row in hms.store._conn.execute("SELECT u, b, a, y, r, entity_mentions_json, topic_tags_json, confidence FROM eps")]
    assert eps == [(
        "the webhook kept timing out", "never touch billing", "patched retry.ts to add backoff", "billing_e2e failed", "negative",
        '["retry.ts","billing_e2e"]', '["engineering","billing"]', derive.COMPACT_CONFIDENCE,
    )]  # fmt: skip
    hms.close()


# ---------------------------------------------------------------------------
# (3) The rule LLM's lines replies through the pipeline
# (3) 规则 LLM 的行式回复经流水线
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CHAT_FIXTURES)
def test_rule_llm_lines_replies_match_the_full_facade(name):
    """On every chat fixture the rule LLM's lines replies (tab-separated, no JSON, parsed back to the same texts) make the pipeline store the same texts, events, witness anchors, sides, types, entity_ids, revises, atom subject / predicate / objects and the same nine axis / history / graph / eps tables as the full facade, and observations / memory_atoms rows identical outside the derived columns, which are pinned one by one: confidence 0.7, relation single_fact with no hint where full says supersession (chat_preference, chat_zh), actor_id only where the lines speaker rule names the statement's leading proper name, and chat_zh's Chinese side-word mention.
    在每个聊天夹具上，规则 LLM 的行式回复（tab 分隔、无 JSON、解析回相同文本）让流水线存下与完整门面相同的文本、事件、见证锚点、侧别、类型、entity_ids、revises、原子的 subject / predicate / objects 以及相同的九张轴 / 历史 / 图 / eps 表，observations / memory_atoms 行在推导列之外相同，推导列逐一钉住：confidence 0.7、full 为 supersession 处（chat_preference、chat_zh）为不带提示的 single_fact、只有在行式说话人规则点名陈述开头专有名处 actor_id 不同，以及 chat_zh 的 用户 提及。

    Input / 输入：
        name: A tests/fixtures/events/chat_*.json stem.
        name：tests/fixtures/events/chat_*.json 的文件名。

    Output / 输出：
        Assertions comparing the add() rows and tables of a full and a lines facade, and on the raw lines reply.
        比较完整与行式门面 add() 行与表的断言，以及对原始行式回复的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    full, lines = _facade("full"), _facade("lines")
    full_rows, lines_rows = _rows(full, name), _rows(lines, name)
    assert [c["task"] for c in lines.llm.calls] == ["memory_extraction_lines"] * len(load_events(name).messages)
    keys = ("memory", "event", "anchor_ids", "attributed_to", "atom_type", "scale_level", "revises")
    assert [tuple(r.get(k) for k in keys) for r in lines_rows] == [tuple(r.get(k) for k in keys) for r in full_rows]
    assert [r["hms"]["obs_type"] for r in lines_rows] == [r["hms"]["obs_type"] for r in full_rows]
    assert [r["hms"]["entity_ids"] for r in lines_rows] == [r["hms"]["entity_ids"] for r in full_rows]
    assert all(r["confidence"] == derive.COMPACT_CONFIDENCE for r in lines_rows if r["hms"]["obs_type"])
    for full_row, lines_row in zip(full_rows, lines_rows):
        subject = lines_row["hms"].get("subject")
        if full_row.get("actor_id") is not None or not lines_row["hms"].get("obs_type") or subject in derive.SIDE_SUBJECTS:
            assert lines_row.get("actor_id") == full_row.get("actor_id")
        else:
            assert lines_row["actor_id"] == subject
    full_tables, lines_tables = _tables(full), _tables(lines)
    for table in ("atom_quads", *IDENTICAL_TABLES, "observations_modulo", "memory_atoms_modulo"):
        assert lines_tables[table] == full_tables[table], table
    # The derived columns, one by one: the lines side is the compact derivation, the full side the rule LLM's own.
    # 推导列逐一对照：行式一侧是紧凑推导，完整一侧是规则 LLM 自己的值。
    relations: list[tuple[str, str]] = []
    for full_d, lines_d in zip(full_tables["observations_derived"], lines_tables["observations_derived"]):
        assert lines_d["confidence"] == derive.COMPACT_CONFIDENCE and 0.0 < full_d["confidence"] <= 1.0
        assert lines_d["supersedes_hint_ids_json"] == "[]"
        assert lines_d["relation"] == full_d["relation"] or (full_d["relation"], lines_d["relation"]) == ("supersession", "single_fact")
        relations.append((full_d["relation"], lines_d["relation"]))
        assert lines_d["entity_mentions_json"] == full_d["entity_mentions_json"] or (
            name == "chat_zh" and (full_d["entity_mentions_json"], lines_d["entity_mentions_json"]) == ("[]", '["用户"]')
        )
    assert (("supersession", "single_fact") in relations) == (name in ("chat_preference", "chat_zh")), relations
    for full_d, lines_d in zip(full_tables["memory_atoms_derived"], lines_tables["memory_atoms_derived"]):
        assert lines_d["confidence"] == derive.COMPACT_CONFIDENCE
        full_content, lines_content = json.loads(full_d["content_json"]), json.loads(lines_d["content_json"])
        assert {k: v for k, v in full_content.items() if k != "relation"} == {k: v for k, v in lines_content.items() if k != "relation"}
        pair = (full_content.get("relation"), lines_content.get("relation"))
        assert pair[0] == pair[1] or pair == ("supersession", "single_fact"), pair
        assert lines_d["actor_id"] == full_d["actor_id"] or (full_d["actor_id"] is None and lines_d["actor_id"] not in derive.SIDE_SUBJECTS)
    if name == "chat_conflict":
        assert [row["event"] for row in lines_tables["history"]] == ["ADD", "ADD", "CONFLICT"]
    if name == "chat_multisession":
        assert {quad[1] for quad in lines_tables["atom_quads"]} == {"Sarah"} and {r["actor_id"] for r in lines_rows} == {"Sarah"}
    if name == "chat_action_closure":
        ep = lines_tables["eps"][0]
        assert (ep["entity_mentions_json"], ep["topic_tags_json"]) == ('["retry.ts","billing_e2e"]', '["engineering","billing"]')
    # The raw reply: tab-separated sections, no JSON, and parse_lines reads the stored texts back out of it.
    # 原始回复：tab 分隔的分节、无 JSON，parse_lines 从中读回所存的文本。
    for message in load_events(name).messages:
        stamp = message.get("timestamp") or STAMP
        user = prompts.build_memory_extraction_prompt(
            existing_memories=[], new_messages=[{"idx": 0, "role": message["role"], "text": message["content"], "timestamp": stamp}],
            last_messages=[], observation_date=stamp[:10], current_date="2026-09-02", custom_instructions=None,
            events=[], known_constraints=[], recent_eps=[], candidates=[],
        )  # fmt: skip
        reply = lines.llm.generate_lines("memory_extraction_lines", system=prompts.MEMORY_EXTRACTION_LINES_PROMPT, user=user)
        assert reply.startswith("OBS\n") and reply.endswith("\n") and "{" not in reply and "DEC" not in reply
        parsed = parse_lines(reply)
        assert all(set(item) <= {"id", "event_idx", "text", "type", "when"} for item in parsed["observations"])
        assert all(item["type"] in prompts.LINE_TYPES.values() and item["event_idx"] == 0 for item in parsed["observations"])
        assert all("when" not in item or item["when"] != stamp[:10] for item in parsed["observations"])
    full.close()
    lines.close()


def test_rule_llm_renders_eps_dates_and_non_add_decisions(rule_llm):
    """The rule LLM's lines reply carries an EP section only for an action closure (u= a= y= r= and k= for matched constraints), an @date only for a statement dated away from its message, and a DEC section only for a non-ADD verdict naming the 1-based OBS line.
    规则 LLM 的行式回复只在有行动闭包时带 EP 分节（u= a= y= r=，匹配的约束进 k=），只对日期偏离消息的陈述带 @date，只对非 ADD 裁决带 DEC 分节并以 1 起的 OBS 行号点名。

    Input / 输入：
        rule_llm: A RuleLLM; the chat_action_closure, chat_multisession and chat_preference fixtures.
        rule_llm：一个 RuleLLM；chat_action_closure、chat_multisession 与 chat_preference 夹具。

    Output / 输出：
        Assertions on the rendered lines.
        对渲染出的行的断言。

    State effect / 状态影响：
        None.
        无。
    """

    def reply(name: str, index: int, *, existing: list[dict], events: bool = False, constraints: list[dict] = ()) -> str:
        message = load_events(name).messages[index]
        stamp = message.get("timestamp") or STAMP
        row = {"idx": 0, "role": message["role"], "text": message["content"], "timestamp": stamp}
        user = prompts.build_memory_extraction_prompt(
            existing_memories=existing, new_messages=[row], last_messages=[], observation_date=stamp[:10],
            current_date="2026-09-02", custom_instructions=None,
            events=[{**row, "object_type": "message"}] if events else [], known_constraints=list(constraints),
            recent_eps=[],
            candidates=[
                {"id": e["id"], "type": "preference_event", "subject": "user", "predicate": "likes",
                 "objects": ["oat-milk lattes in the morning"], "relation": "single_fact", "text": e["text"],
                 "date_observed": "2026-04-02", "valid_from": None}
                for e in existing
            ],
        )  # fmt: skip
        return rule_llm.generate_lines("memory_extraction_lines", system=prompts.MEMORY_EXTRACTION_LINES_PROMPT, user=user)

    closure = reply("chat_action_closure", 0, existing=[], events=True, constraints=[{"id": "c0", "text": "never touch billing"}])
    obs, ep = closure.rstrip("\n").split("\nEP\n")
    assert obs.startswith("OBS\n0\tF\t") and "\t@" not in obs
    fields = ep.split("\t")
    assert fields[0] == "0" and [f.split("=")[0] for f in fields[1:5]] == ["u", "a", "y", "r"] and fields[4] == "r=negative"
    assert "DEC" not in closure and parse_lines(closure)["eps"][0]["a"] == "patched retry.ts to add backoff"
    dated = reply("chat_multisession", 0, existing=[])
    assert dated == "OBS\n0\tT\tSarah joined the billing team in March.\t@2026-03-01\n"
    revised = reply("chat_preference", 1, existing=[{"id": "0", "text": "User likes oat-milk lattes in the morning."}])
    assert revised == "OBS\n0\tP\tUser likes black coffee.\nDEC\n1\tREVISE\t0\tsupersession cue\n"
    assert rule_llm.calls[-1]["task"] == "memory_extraction_lines"


# ---------------------------------------------------------------------------
# (4) Config validation and wiring
# (4) 配置校验与接线
# ---------------------------------------------------------------------------
def test_lines_config_validation_and_wiring():
    """'lines' is a third extraction_output (the default since 2026-09-18, CONTRACT §6.6 / changelog row 108), reaches the Ingestor through HMS and AsyncHMS with the memory_extraction_lines task, and is rejected with a bilingual message under extraction_mode='staged' by IngestConfig (CONFIG_001 / pydantic), Ingestor (VALIDATION_013) and extract_observations (VALIDATION_013); the benchmark CLI takes its choices from EXTRACTION_OUTPUTS and the elite arm forwards the value.
    'lines' 是第三种 extraction_output（2026-09-18 起为默认，CONTRACT §6.6 / 变更记录第 108 行），经 HMS 与 AsyncHMS 以 memory_extraction_lines 任务到达 Ingestor，在 extraction_mode='staged' 下被 IngestConfig（CONFIG_001 / pydantic）、Ingestor（VALIDATION_013）与 extract_observations（VALIDATION_013）以双语消息拒绝；基准 CLI 的选项取自 EXTRACTION_OUTPUTS，elite 臂原样转发该值。

    Input / 输入：
        Config dicts, a lines facade, an Ingestor and extract_observations under 'staged', the benchmark sources.
        配置字典、行式门面、'staged' 下的 Ingestor 与 extract_observations、基准脚本源码。

    Output / 输出：
        Assertions over the default, the pass-through, the rejections and their messages.
        对默认值、透传、拒绝及其消息的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    assert EXTRACTION_OUTPUTS == ("full", "compact", "lines") and IngestConfig().extraction_output == "lines"
    assert IngestConfig(extraction_output="lines").extraction_mode == "single"
    with pytest.raises(pydantic.ValidationError) as direct:
        IngestConfig(extraction_output="lines", extraction_mode="staged")
    assert "extraction_mode='single' only" in str(direct.value) and "只支持 extraction_mode='single'" in str(direct.value)
    with pytest.raises(ConfigError) as via_dict:
        HMSConfig.from_dict({"ingest": {"extraction_output": "lines", "extraction_mode": "staged"}, "store": {"path": ":memory:"}})
    assert via_dict.value.error_code == "CONFIG_001"
    hms = _facade("lines")
    assert hms.config.ingest.extraction_output == "lines" and hms.ingestor.extraction_output == "lines"
    assert Ingestor(hms.store, hms.index, hms.embedder, hms.llm, hms.config).extraction_output == "lines"
    rows = hms.add("I like oat-milk lattes in the morning.", user_id="alice", timestamp="2026-04-02T08:00:00Z")["results"]
    assert [c["task"] for c in hms.llm.calls] == ["memory_extraction_lines"]
    assert rows[0]["memory"] == "User likes oat-milk lattes in the morning." and rows[0]["hms"]["predicate"] == "likes"
    with pytest.raises(ValidationError) as staged:
        Ingestor(hms.store, hms.index, hms.embedder, hms.llm, hms.config, extraction_mode="staged", extraction_output="lines")
    assert staged.value.error_code == "VALIDATION_013" and staged.value.details["supported"] == ["single"]
    assert "只支持" in str(staged.value)
    with pytest.raises(ValidationError) as call:
        extraction.extract_observations(
            hms.llm, events=[], existing=[], last_events=[], observation_date=DAY, current_date=DAY,
            custom_instructions=None, scope=SCOPE, id_map={}, output="lines",
        )  # fmt: skip
    assert call.value.error_code == "VALIDATION_013" and "CONTRACT §6.6" in str(call.value)
    hms.close()
    mirrored = AsyncHMS(
        HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, ingest={"extraction_output": "lines"})
    )
    assert mirrored._sync.ingestor.extraction_output == "lines"
    mirrored._sync.close()
    # The benchmark harness: the CLI choices follow EXTRACTION_OUTPUTS and the elite arm forwards the value verbatim.
    # 基准脚本：CLI 选项跟随 EXTRACTION_OUTPUTS，elite 臂原样转发该值。
    run_real = (ROOT / "benchmarks" / "egolife" / "run_real.py").read_text()
    assert 'parser.add_argument("--extraction-output", choices=EXTRACTION_OUTPUTS' in run_real
    real_arms = (ROOT / "benchmarks" / "egolife" / "real_arms.py").read_text()
    assert '{"extraction_output": extraction_output}' in real_arms


def test_lines_harness_bookkeeping_source_pins():
    """The benchmark harness accounts for the lines contract: an unparsable lines reply (NO_SECTION_MARK) counts as a parse error and enters the invalid-JSON retry / bisect ladder, the lines parser's own warnings are counted apart as lines_dropped (never as schema rejections), a finish=length lines reply is handed to the library as '' before the bisect, and the manifest's response_format names none for the lines arm.
    基准脚本为行式契约记账：不可解析的行式回复（NO_SECTION_MARK）计为解析错误并进入非法 JSON 的重试 / 二分阶梯，行式解析器自己的警告另计为 lines_dropped（绝不计为模式拒绝），finish=length 的行式回复在二分之前以 '' 交给库，manifest 的 response_format 对行式臂写为 none。

    Input / 输入：
        The benchmark sources (they import mem0 / openai, so they are read as text, not imported).
        基准脚本源码（它们导入 mem0 / openai，因此按文本读取而不导入）。

    Output / 输出：
        Assertions over source substrings and the parser's logger name.
        对源码子串与解析器 logger 名的断言。

    State effect / 状态影响：
        None.
        无。
    """
    real_arms = (ROOT / "benchmarks" / "egolife" / "real_arms.py").read_text()
    assert "from hms_elite.ingest.lines import NO_SECTION_MARK" in real_arms
    assert '("Error parsing" in w and "response" in w) or NO_SECTION_MARK in w)' in real_arms
    assert 'LINES_LOGGER = "hms_elite.ingest.lines:"' in real_arms and LINES_LOGGER == "hms_elite.ingest.lines"
    assert "w.startswith(LINES_LOGGER) and NO_SECTION_MARK not in w)" in real_arms
    assert "and not w.startswith(LINES_LOGGER)" in real_arms and '"lines_dropped": lines_dropped' in real_arms
    assert "def _reject_truncated_lines" in real_arms and "last.finish_reason == TRUNCATED" in real_arms
    assert 'if self.config.ingest.extraction_output == "lines":\n            self._reject_truncated_lines()' in real_arms
    run_real = (ROOT / "benchmarks" / "egolife" / "run_real.py").read_text()
    assert 'if args.extraction_output == "lines" else "json_object (both libraries\' own prompts)"' in run_real
    assert 'sum(k.get("lines_dropped", 0) for k in kinds)' in run_real and '"lines_dropped": lines_dropped' in run_real
    assert "- lines_dropped -" in run_real


# ---------------------------------------------------------------------------
# (5) Prompt registration and the full / compact contracts untouched
# (5) 提示词注册与完整 / 紧凑契约不变
# ---------------------------------------------------------------------------
def test_lines_prompt_is_registered_keeps_the_statement_rules_and_has_no_json_output():
    """The lines prompt is a composite task like its twins, carries the full prompt's statement rules verbatim, the compact prompt's decision rules, an OUTPUT FORMAT without JSON braces or the JSON trailer, one short example, and leaves the full and compact prompts byte-identical.
    行式提示词与其孪生一样是复合任务，逐字带有完整提示词的陈述规则、紧凑提示词的决策规则、不含 JSON 花括号与 JSON 尾句的 OUTPUT FORMAT、一个短示例，且让完整与紧凑提示词逐字节不变。

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
    lines, full = prompts.MEMORY_EXTRACTION_LINES_PROMPT, prompts.MEMORY_EXTRACTION_PROMPT
    assert prompts.task_of(lines) == "memory_extraction_lines"
    assert prompts.COMPOSITE_PROMPTS["memory_extraction_lines"] is lines and "memory_extraction_lines" in prompts.COMPOSITE_TASKS
    assert "memory_extraction_lines" not in prompts.ROOT_KEY and "memory_extraction_lines" not in prompts.TASKS
    assert "memory_extraction_lines" not in prompts.PROMPTS
    assert prompts.LINE_TYPES == {"F": "fact", "P": "preference_event", "H": "habit", "C": "constraint", "T": "temporal_fact"}
    assert prompts.TYPE_LETTERS == {name: letter for letter, name in prompts.LINE_TYPES.items()}
    for heading in ("# ROLE", "# INPUTS", "# GUIDELINES", "# EXAMPLES", "# OUTPUT FORMAT"):
        assert heading in lines, heading
    for sentence in KEY_SENTENCES:
        assert sentence in full and sentence in lines, sentence
    # The source-language rule (CONTRACT §9.6 (c)) lives in the lines prompt only; the four hashed prompts are untouched.
    # 源语言规则（CONTRACT §9.6 (c)）只在行式提示词中；四个带哈希钉住的提示词不动。
    assert "Write every statement in\n  the language of the source messages" in lines and "never translate." in lines
    assert "never translate" not in full
    assert "SAME fact about the SAME subject with the SAME value" in lines and "temporal facts and habits are always" in lines
    assert "ONLY for a non-ADD verdict" in lines and "never one\n  per candidate, never by position" in lines
    assert "ONLY when Candidate Existing Memories is non-empty" in lines
    section = prompts.output_format_section(lines)
    assert "{" not in section and "}" not in section and "```" not in section
    assert not section.endswith(prompts.JSON_TRAILER) and section.endswith("No JSON, no reasoning, no explanations, no wrappers.")
    assert "\nOBS\n" in section and "\nEP\n" in section and "\nDEC\n" in section and section.count("\t") >= 8
    assert "F fact, P preference_event, H habit, C constraint, T temporal_fact" in section
    # One short example: two OBS lines, no EP, one DEC line, all tab-separated.
    # 一个短示例：两行 OBS、无 EP、一行 DEC，全部 tab 分隔。
    example = lines.split("# EXAMPLES", 1)[1].split("# OUTPUT FORMAT", 1)[0].split("→", 1)[1].strip()
    parsed = parse_lines(example + "\n")
    assert [o["type"] for o in parsed["observations"]] == ["preference_event", "temporal_fact"] and parsed["eps"] == []
    assert parsed["observations"][1]["when"] == "2026-06-20T09:30"
    assert parsed["decisions"] == [{"id": "1", "decision": "REVISE", "target_id": "1", "reason": "supersession cue: 'switched to'"}]
    assert "\nEP\n" not in example
    for name, digest in PINNED_PROMPT_SHA256.items():
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == digest, name


# ---------------------------------------------------------------------------
# (6) Providers: generate_lines never uses json mode
# (6) 提供者：generate_lines 绝不使用 json 模式
# ---------------------------------------------------------------------------
def test_deepseek_generate_lines_sends_no_response_format(monkeypatch):
    """LLMBase.generate_lines through the DeepSeek provider calls chat.completions.create without response_format and returns the reply text raw, while generate_json on the same provider still sends json_object.
    经 DeepSeek 提供者的 LLMBase.generate_lines 调用 chat.completions.create 时不带 response_format 并原样返回回复文本，而同一提供者的 generate_json 仍发送 json_object。

    Input / 输入：
        monkeypatch: pytest monkeypatch (the fake openai SDK and env vars).
        monkeypatch：pytest monkeypatch（替身 openai SDK 与环境变量）。

    Output / 输出：
        Assertions over the recorded chat.completions.create kwargs and the returned text.
        对记录的 chat.completions.create 参数与返回文本的断言。

    State effect / 状态影响：
        Temporarily patches the deepseek module (restored by monkeypatch).
        临时给 deepseek 模块打补丁（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    calls: list[dict] = []
    monkeypatch.setattr(deepseek_module, "_IMPORT_ERROR", None)
    monkeypatch.setattr(deepseek_module, "OpenAI", _fake_openai(calls).OpenAI)
    llm = LlmFactory.create("deepseek", {"max_tokens": 77})
    text = llm.generate_lines("memory_extraction_lines", system=prompts.MEMORY_EXTRACTION_LINES_PROMPT, user="u")
    sent = calls[-1]
    assert sent["api"] == "chat" and "response_format" not in sent and "tools" not in sent
    assert sent["messages"][0] == {"role": "system", "content": prompts.MEMORY_EXTRACTION_LINES_PROMPT}
    assert sent["messages"][1] == {"role": "user", "content": "u"} and sent["max_tokens"] == 77
    assert text == '{"observations": []}'
    llm.generate_json("memory_extraction", system=prompts.MEMORY_EXTRACTION_PROMPT, user="u")
    assert calls[-1]["response_format"] == {"type": "json_object"}
