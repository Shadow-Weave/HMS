"""HMS.context() tests: the hybrid read text shape, the session-card memo and its per-scope invalidation, timestamps, dedup / sort parity with the EgoLife harness and the config fields.
HMS.context() 测试：混合读文本形状、会话卡片记忆及其按范围失效、时间戳、与 EgoLife 基准脚本的去重 / 排序一致性以及配置字段。
"""

# Purpose (EN): Pin CONTRACT §6.7 offline (hash embedder, rule LLM): text == hybrid_text(session card, rows_text),
#               '(none)' parts, the memo reused across calls and dropped by every transaction that changed versioned
#               rows of the scope (add / update / delete / delete_all / a consolidate that created or decayed
#               something / rollback / reset, a parent-scope delete_all) but not by a write to another scope, a
#               checkpoint, an idle consolidate, the daemon-shaped consolidate() or a zero-row transaction,
#               StateStore.scope_version's commit-time and unversioned() semantics, the '[YYYY-MM-DD HH:MM]' stamps,
#               and that benchmarks/egolife/reader_ab.py's builders ARE the library's functions.
# 目的（中）：离线钉住 CONTRACT §6.7（哈希嵌入器、规则 LLM）：text == hybrid_text(会话卡片, rows_text)、'(none)' 部分、
#            记忆在多次调用间复用且被每个改变了该范围带版本行的事务（add / update / delete / delete_all / 创建或衰减了
#            内容的 consolidate / rollback / reset、父范围的 delete_all）丢弃，但不被其它范围的写入、检查点、空转的
#            consolidate、守护进程形态的 consolidate() 或零行事务丢弃、StateStore.scope_version 的提交时与 unversioned()
#            语义、'[YYYY-MM-DD HH:MM]' 时间戳，以及 benchmarks/egolife/reader_ab.py 的构造器就是库的函数。

# Input (EN): The conftest `hms` / `config` fixtures, hand-written rows and the harness module imported the way
#             tests/test_reader_ab.py imports it.
# 输入（中）：conftest 的 `hms` / `config` 夹具、手写的行，以及按 tests/test_reader_ab.py 的方式导入的基准模块。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): In-memory stores only.
# 状态影响（中）：仅内存存储。

from __future__ import annotations

from typing import Any

import pytest

from hms_elite.compiler import context as ctx_mod
from hms_elite.compiler.context import (
    NONE_TEXT,
    ROWS_HEADER,
    dedup_rows,
    format_stamp,
    hybrid_text,
    render_rows,
    rows_text,
    sort_rows_by_time,
)
from hms_elite.configs.base import CompilerConfig, HMSConfig
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.memory.main import HMS
from hms_elite.memory.storage import StateStore
from hms_elite.models import MemoryAtom, Scope, SourceEvent
from tests.test_reader_ab import reader_ab

SCOPE = {"user_id": "alice", "agent_id": "bot"}
OTHER = {"user_id": "carol"}
T1, T2, T3 = "2026-04-02T08:00:00Z", "2026-04-09T08:00:00Z", "2026-04-16T08:00:00Z"
# One day before the conftest FrozenClock (2026-09-02): atoms anchored here are too fresh for prune to decay, so a
# consolidate over them has nothing to do (the April atoms above DO decay under that clock).
# conftest FrozenClock（2026-09-02）的前一天：锚定于此的原子太新、prune 不会衰减，因此对它们的 consolidate 无事可做
# （上面的四月原子在该时钟下**会**衰减）。
FRESH = "2026-09-01T08:00:00Z"
QUESTION = "what coffee do I drink?"


def _row(text: str, anchor: str | None, created: str | None, bid: str = "b0000") -> dict[str, Any]:
    """A public-row stand-in with the keys the builders read (the shape of tests/test_reader_ab.py's hand rows).
    只带构造器所读键的公开行替身（tests/test_reader_ab.py 手写行的形状）。
    """
    return {"id": text[:4], "memory": text, "metadata": {"batch_id": bid}, "score": 0.5, "created_at": created, "hms": {"time_anchor": anchor}}


# The seven hand rows of tests/test_reader_ab.py::test_rows_td_dedup_substring_and_time_sort_and_rows_ts_stamps.
# tests/test_reader_ab.py::test_rows_td_dedup_substring_and_time_sort_and_rows_ts_stamps 的七条手写行。
HAND_ROWS = [
    _row("Jake bought a hard drive.", "2024-01-01T12:00:00.000000+00:00", "2024-01-01T12:00:00.000000+00:00"),
    _row("  jake BOUGHT a hard   drive. ", "2024-01-01T11:00:00.000000+00:00", "2024-01-01T11:00:00.000000+00:00"),
    _row("a hard drive", "2024-01-01T09:00:00.000000+00:00", "2024-01-01T09:00:00.000000+00:00"),
    _row("Shure fixed the coffee machine.", None, "2024-01-01T10:30:00.000000+00:00"),
    _row("Alice arrived.", None, None),
    _row("Bob left.", None, None),
    _row("Jake bought a hard drive and a USB.", "2024-01-01T08:00:00.000000+00:00", "2024-01-01T08:00:00.000000+00:00"),
]
HAND_TS = (
    "- [2024-01-01 08:00] Jake bought a hard drive and a USB.\n- Shure fixed the coffee machine.\n"
    "- [2024-01-01 12:00] Jake bought a hard drive.\n- Alice arrived.\n- Bob left."
)
HAND_TD = "- Jake bought a hard drive and a USB.\n- Shure fixed the coffee machine.\n- Jake bought a hard drive.\n- Alice arrived.\n- Bob left."


def _seed(hms: HMS) -> list[str]:
    """Two coffee memories under SCOPE at T1 / T2; returns their memory ids.
    在 SCOPE 下于 T1 / T2 写入两条咖啡记忆；返回其 memory id。
    """
    ids = []
    for text, stamp in (("I drink oat-milk lattes every morning.", T1), ("I switched to black coffee last week.", T2)):
        ids.extend(r["id"] for r in hms.add(text, timestamp=stamp, **SCOPE)["results"] if r.get("id"))
    return ids


def test_context_shape_and_none_parts(hms: HMS) -> None:
    """Empty scope → '(none)' head and tail with session_key None; after two adds the text is exactly hybrid_text(session_card, rows_text) under the configured header, the rows are the deduplicated time-sorted search rows and the session card comes from the newest message event at its own timestamp.
    空范围 → 首尾都是 '(none)' 且 session_key 为 None；两次 add 之后文本恰好是配置标题下的 hybrid_text(session_card, rows_text)，rows 是去重并按时间排序的检索行，会话卡片来自最新的消息事件、以其自身时间戳编译。

    Input / 输入：
        hms: The conftest facade.
        hms：conftest 门面。

    Output / 输出：
        Assertions over keys, text and provenance.
        对键、文本与出处的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    empty = hms.context(QUESTION, filters=SCOPE)
    assert empty["text"] == f"{NONE_TEXT}\n\n{ROWS_HEADER}\n{NONE_TEXT}" == "(none)\n\nRecent relevant memories:\n(none)"
    assert empty["session_key"] is None and empty["session_card"] == "" and empty["rows"] == [] and empty["rows_text"] == ""
    assert empty["session_reused"] is False and empty["session_memory_ids"] == [] and empty["scope"] == "agent_id=bot&user_id=alice"
    _seed(hms)
    out = hms.context(QUESTION, filters=SCOPE)
    assert list(out) == [
        "text", "session_card", "session_key", "session_as_of", "session_query", "session_reused", "session_memory_ids",
        "rows", "rows_text", "rows_header", "search_rows", "scope",
    ]  # fmt: skip
    assert out["rows_header"] == ROWS_HEADER == hms.config.compiler.context_rows_header
    assert out["text"] == hybrid_text(out["session_card"], out["rows_text"], header=out["rows_header"])
    assert out["text"] == out["session_card"] + "\n\n" + ROWS_HEADER + "\n" + out["rows_text"]
    assert out["session_card"].startswith("# Work State Card") and "(none)" not in out["text"].split(ROWS_HEADER)[1]
    raw = hms.search(QUESTION, filters=SCOPE, top_k=hms.config.compiler.context_rows_top_k)["results"]
    assert out["search_rows"] == raw and out["rows"] == sort_rows_by_time(dedup_rows(raw))
    assert out["rows_text"] == rows_text(raw) == render_rows(out["rows"])
    # The session card is compiled from the newest message event: its text is the query, its timestamp the as_of.
    # 会话卡片以最新的消息事件编译：其文本是查询，其时间戳是 as_of。
    newest = hms.store.latest_event(Scope(**SCOPE), object_types=["message"])
    assert newest is not None and out["session_key"] == newest.event_id and out["session_as_of"] == newest.timestamp
    assert out["session_query"] == "I switched to black coffee last week." == newest.payload["text"]
    assert f"As of: {newest.timestamp}" in out["session_card"]
    assert all(hms.get(mid) is not None for mid in out["session_memory_ids"])
    # A missing scope key is VALIDATION_001, a blank query VALIDATION_010, a negative top_k VALIDATION_009 and an
    # explicit top_k=0 reaches search unchanged (an empty rows part), never silently the default.
    # 缺少范围键为 VALIDATION_001，空白查询为 VALIDATION_010，负的 top_k 为 VALIDATION_009，显式的 top_k=0 原样到达
    # search（记忆行部分为空），绝不悄悄变成默认值。
    zero = hms.context(QUESTION, filters=SCOPE, top_k=0)
    assert zero["search_rows"] == [] and zero["text"].endswith(ROWS_HEADER + "\n(none)") and zero["session_reused"] is True
    for bad_kwargs, code in (({"filters": {}}, "VALIDATION_001"), ({"filters": SCOPE, "top_k": -1}, "VALIDATION_009")):
        with pytest.raises(ValidationError) as exc:
            hms.context(QUESTION, **bad_kwargs)
        assert exc.value.error_code == code
    with pytest.raises(ValidationError) as blank:
        hms.context("   ", filters=SCOPE)
    assert blank.value.error_code == "VALIDATION_010"


def test_context_memo_reuse_and_invalidation(hms: HMS) -> None:
    """The session card is reused byte for byte across calls and queries until a transaction changes versioned rows of the scope: add, update, delete, delete_all on the scope itself, delete_all on the PARENT scope, a consolidate that decays the April atoms (the conftest clock is 2026-09-02), rollback and reset each force a recompute, while a write to an unrelated scope (add, consolidate, feedback — no global write_generation bump), a checkpoint (unversioned rows) and a different query leave it reused.
    会话卡片在多次调用与不同查询间逐字节复用，直到某个事务改变了该范围带版本的行：范围自身的 add、update、delete、delete_all，**父范围**的 delete_all，衰减了四月原子的 consolidate（conftest 时钟为 2026-09-02），rollback 与 reset 各自强制重算，而对无关范围的写入（add、consolidate、feedback——不推进全局 write_generation）、检查点（不带版本的行）与不同的查询都保持复用。

    Input / 输入：
        hms: The conftest facade.
        hms：conftest 门面。

    Output / 输出：
        Assertions over session_reused / session_key / session_card across the write sequence.
        对写入序列中 session_reused / session_key / session_card 的断言。

    State effect / 状态影响：
        None outside the in-memory store (reset at the end).
        除内存存储外无副作用（末尾 reset）。
    """
    ids = _seed(hms)
    first = hms.context(QUESTION, filters=SCOPE)
    again = hms.context("something unrelated to coffee", filters=SCOPE)
    assert first["session_reused"] is False and again["session_reused"] is True
    assert again["session_card"] == first["session_card"] and again["session_key"] == first["session_key"]
    # A write to another scope does not touch this memo (per-scope versions, not a global counter).
    # 对其它范围的写入不触及本记忆（按范围的版本，而非全局计数器）。
    hms.add("Carol likes tea.", timestamp=T1, **OTHER)
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True
    assert hms.context("tea", filters=OTHER)["session_reused"] is False
    # Scope-less rows written on behalf of another scope (consolidate's phase transactions and its meta row, a
    # feedback revision) advance that scope's version only, never the global write_generation.
    # 代表其它范围写入的无范围行（consolidate 的各阶段事务及其 meta 行、一次反馈修订）只推进该范围的版本，
    # 绝不推进全局 write_generation。
    hms.add("I like tea in the afternoon.", timestamp=T1, **OTHER)
    other_id = hms.get_all(filters=OTHER)["results"][0]["id"]
    generation = hms.store.write_generation
    hms.consolidate(filters=OTHER)
    hms.feedback(other_id, "object_important")
    assert hms.store.write_generation == generation
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True
    assert hms.context("tea", filters=OTHER)["session_reused"] is False

    def recomputed(label: str) -> dict[str, Any]:
        """Assert the next call recomputes, then that the one after reuses it.
        断言下一次调用重算，随后一次复用。
        """
        out = hms.context(QUESTION, filters=SCOPE)
        assert out["session_reused"] is False, label
        assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True, label
        return out

    hms.add("I take my coffee with no sugar.", timestamp=T3, **SCOPE)
    after_add = recomputed("add")
    assert after_add["session_key"] != first["session_key"] and after_add["session_query"] == "I take my coffee with no sugar."
    hms.update(ids[0], text="I drink oat-milk lattes every afternoon.")
    recomputed("update")
    hms.delete(ids[-1])
    recomputed("delete")
    # prune decays the April atoms (their confidence changes): a versioned row of the scope, hence a recompute.
    # prune 衰减四月的原子（其置信度改变）：该范围带版本的行，因此重算。
    assert hms.consolidate(filters=SCOPE)["phases"]["prune"]["decayed"] > 0
    recomputed("consolidate")
    # A checkpoint writes only unversioned rows (its event, anchor and checkpoints row): nothing to recompute.
    # 检查点只写不带版本的行（其事件、锚点与 checkpoints 行）：无需重算。
    cp = hms.checkpoint(filters=SCOPE, label="before")
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True
    hms.add("I bought a new grinder.", timestamp=T3, **SCOPE)
    recomputed("add after checkpoint")
    hms.rollback(cp, reason="test")
    recomputed("rollback")
    # delete_all on the scope itself and on the parent scope (user_id only) both invalidate the (user, agent) memo.
    # 对范围自身与对父范围（仅 user_id）的 delete_all 都使 (user, agent) 的记忆失效。
    hms.delete_all(**SCOPE)
    recomputed("delete_all(scope)")
    _seed(hms)
    recomputed("re-seed")
    hms.delete_all(user_id=SCOPE["user_id"])
    after_parent = recomputed("delete_all(parent)")
    assert after_parent["rows"] == []
    _seed(hms)
    assert hms.context(QUESTION, filters=SCOPE)["session_key"] is not None
    hms.reset()
    after_reset = hms.context(QUESTION, filters=SCOPE)
    assert after_reset["session_reused"] is False and after_reset["session_key"] is None
    assert after_reset["text"] == "(none)\n\nRecent relevant memories:\n(none)"


def test_context_idle_writes_keep_the_memo(hms: HMS) -> None:
    """Writes that cannot change the card leave the session card reused and both counters untouched: a consolidate(A) with nothing to do (its scheduler event, compiled-state archive and meta rows are unversioned), the daemon-shaped consolidate() over every scope, a checkpoint, an attributed transaction whose UPDATE hits no row, a NOT_FOUND delete and a replayed identical event; a consolidate that creates an entity atom in A recomputes A only (B stays reused), its repeat is idle again, and rollback to the checkpoint (versioned archives) recomputes.
    不可能改变卡片的写入让会话卡片保持复用、两个计数器不动：无事可做的 consolidate(A)（其调度器事件、编译态归档与 meta 行都不带版本）、守护进程形态的对全部范围的 consolidate()、检查点、UPDATE 未命中任何行的已归属事务、NOT_FOUND 的 delete 与重放的相同事件；在 A 中创建了实体原子的 consolidate 只重算 A（B 保持复用），其重复运行再次空转，回滚到检查点（带版本的归档）则重算。

    Input / 输入：
        hms: The conftest facade (FrozenClock 2026-09-02; FRESH-anchored atoms do not decay).
        hms：conftest 门面（FrozenClock 2026-09-02；锚定于 FRESH 的原子不衰减）。

    Output / 输出：
        Assertions over session_reused, scope_version and write_generation across the write sequence.
        对写入序列中 session_reused、scope_version 与 write_generation 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    scope = Scope(**SCOPE)
    for text in ("I drink oat-milk lattes every morning.", "I switched to black coffee last week."):
        hms.add(text, timestamp=FRESH, **SCOPE)
    hms.add("I like tea in the afternoon.", timestamp=FRESH, **OTHER)
    first = hms.context(QUESTION, filters=SCOPE)
    assert first["session_reused"] is False and hms.context("tea", filters=OTHER)["session_reused"] is False
    generation, version = hms.store.write_generation, hms.store.scope_version(scope)

    def untouched(label: str) -> None:
        """Assert both memos are reused and neither counter moved.
        断言两个记忆都被复用且两个计数器都没动。
        """
        assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True, label
        assert hms.context("tea", filters=OTHER)["session_reused"] is True, label
        assert hms.store.scope_version(scope) == version and hms.store.write_generation == generation, label

    # (i) consolidate(A) with nothing to do: the compiled-state archive is written, the memo survives.
    # (i) 无事可做的 consolidate(A)：写入了编译态归档，记忆仍然存活。
    report = hms.consolidate(filters=SCOPE)["phases"]
    assert report["consolidate"]["compiled_states"] == 1 and report["prune"]["decayed"] == 0
    assert all(n == 0 for key, n in report["consolidate"].items() if key != "compiled_states")
    assert hms.store.get_meta(f"last_consolidation:{scope.key()}") is not None
    untouched("idle consolidate(A)")
    # (ii) the daemon's shape: consolidate() over every scope, nothing to do anywhere.
    # (ii) 守护进程的形态：对全部范围的 consolidate()，处处无事可做。
    hms.consolidate()
    untouched("idle consolidate()")
    cp = hms.checkpoint(filters=SCOPE, label="idle")
    untouched("checkpoint")
    # (iv) transactions that change no row: an attributed UPDATE of nothing, a NOT_FOUND delete, a replayed event.
    # (iv) 没有改变任何行的事务：已归属但未命中的 UPDATE、NOT_FOUND 的 delete、重放的事件。
    with hms.store.transaction():
        hms.store.touch_scope(scope)
        hms.store.archive_atom("atm_missing", reason="deleted", at=hms.store.now())
    untouched("zero-row update")
    with pytest.raises(ValueError):
        hms.delete("mem_missing")
    untouched("delete of a missing id")
    newest = hms.store.latest_event(scope, object_types=["message"])
    assert newest is not None and hms.store.insert_source_event(newest)[1] is True
    untouched("replayed event")
    # (iii) a consolidate that creates an entity atom in A recomputes A only; the same run again is idle.
    # (iii) 在 A 中创建实体原子的 consolidate 只重算 A；同样的运行再来一次则空转。
    # No date in the sentence: the rule LLM anchors it at FRESH ("… in March" would anchor it in March and let
    # prune decay it every run — a genuine confidence change, not an idle pass).
    # 句中不含日期：规则 LLM 把它锚定在 FRESH（"… in March" 会锚定到三月，让 prune 每次运行都衰减它——那是真实的
    # 置信度变化，而非空转）。
    hms.add("Sarah leads the billing team.", timestamp=FRESH, **SCOPE)
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is False
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is True
    generation, version = hms.store.write_generation, hms.store.scope_version(scope)
    assert hms.consolidate(filters=SCOPE)["phases"]["consolidate"]["entity_atoms"] == 1
    after = hms.context(QUESTION, filters=SCOPE)
    assert after["session_reused"] is False and hms.store.scope_version(scope) != version
    assert hms.context("tea", filters=OTHER)["session_reused"] is True and hms.store.write_generation == generation
    generation, version = hms.store.write_generation, hms.store.scope_version(scope)
    assert hms.consolidate(filters=SCOPE)["phases"]["consolidate"]["entity_atoms"] == 0
    untouched("repeat consolidate(A)")
    hms.rollback(cp, reason="test")
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is False
    assert hms.context("tea", filters=OTHER)["session_reused"] is True and hms.store.write_generation == generation


def test_store_scope_version_moves_at_commit(clock, ids) -> None:
    """StateStore.scope_version: unchanged inside an open transaction and bumped after its COMMIT (so a context() built during add()'s unlocked phase is invalidated by Txn B), bumped by an insert_atom-only transaction, by a scope-less write (write_generation) and by reset; a replayed identical event (INSERT OR IGNORE dedup) leaves it unchanged; an attributed transaction that changed no row and unversioned() rows (attributed, scope-less or autocommit) move neither counter, while a versioned row next to unversioned ones bumps once; a partial query scope sees writes to the scopes it covers and to the scopes covering it.
    StateStore.scope_version：打开的事务内不变、COMMIT 之后推进（因此 add() 无锁阶段构建的 context() 会被事务 B 失效），只 insert_atom 的事务、无范围的写入（write_generation）与 reset 都推进；重放的相同事件（INSERT OR IGNORE 去重）不改变它；没有改变任何行的已归属事务与 unversioned() 的行（已归属、无范围或自动提交）都不推进任一计数器，而与不带版本行同处一个事务的带版本行推进一次；部分查询范围能看到对其覆盖的范围与覆盖它的范围的写入。

    Input / 输入：
        clock: Frozen clock fixture.
        clock：冻结时钟夹具。

        ids: Sequential id fixture.
        ids：顺序 id 夹具。

    Output / 输出：
        Assertions over scope_version tuples.
        对 scope_version 元组的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    store = StateStore(":memory:", now=clock.now, id_factory=ids)
    scope = Scope(user_id="alice", agent_id="bot")
    ev = SourceEvent(scope=scope, payload={"text": "hello", "role": "user"}, timestamp=T1)
    v0 = store.scope_version(scope)
    assert v0 == (0, store.data_version(), ())
    with store.transaction():
        stored, dedup = store.insert_source_event(ev)
        assert dedup is False and store.scope_version(scope) == v0
    v1 = store.scope_version(scope)
    assert v1 != v0 and v1[2] == (("agent_id=bot&user_id=alice", 1),) and v1[0] == 0
    # The same event again is INSERT OR IGNORE'd: nothing stored, nothing bumped.
    # 同一事件再来一次会被 INSERT OR IGNORE：什么也没存，什么也没推进。
    assert store.insert_source_event(ev)[1] is True and store.scope_version(scope) == v1
    # The anchor row alone (no event / atom insert) is a scope-less write: write_generation moves, the scope entry not.
    # 单独的锚点行（没有事件 / 原子插入）是无范围写入：write_generation 推进，范围条目不动。
    anchor = store.get_or_create_anchor(stored)
    va = store.scope_version(scope)
    assert va[0] == 1 and va[2] == v1[2]
    atom = MemoryAtom(
        atom_type="state", content={"text": "hello"}, text="hello", scale_level="state", source_kind="observation",
        source_id=stored.event_id, extracted_by="rule@obs-v1", scope=scope, anchor_ids=[anchor.anchor_id], time_anchor=T1,
    )  # fmt: skip
    with store.transaction():
        store.insert_atom(atom)
    v2 = store.scope_version(scope)
    assert v2[2] == (("agent_id=bot&user_id=alice", 2),) and v2[0] == va[0]
    # A scope-less write moves write_generation: inside a transaction at COMMIT, outside it at once.
    # 无范围的写入推进 write_generation：事务内在 COMMIT 时，事务外立即。
    with store.transaction():
        store.set_meta("probe", "1")
        assert store.scope_version(scope) == v2
    v3 = store.scope_version(scope)
    assert v3[0] == v2[0] + 1 and v3[2] == v2[2]
    store.set_meta("probe", "2")
    assert store.scope_version(scope)[0] == v3[0] + 1
    # Attribution alone moves nothing: a transaction whose UPDATE hits no row, or whose only rows are unversioned
    # (an internal event's shape), leaves both counters where they were; so does an autocommit unversioned write.
    # 仅有归属不推进任何东西：UPDATE 未命中任何行、或只有不带版本行（内部事件的形状）的事务让两个计数器原地不动；
    # 自动提交的不带版本写入亦然。
    v4 = store.scope_version(scope)
    with store.transaction():
        store.touch_scope(scope)
        store.archive_atom("atm_missing", reason="deleted", at=T1)
    assert store.scope_version(scope) == v4
    with store.transaction():
        store.touch_scope(scope)
        assert store.insert_source_event(ev)[1] is True
    assert store.scope_version(scope) == v4
    with store.transaction(), store.unversioned():
        store.touch_scope(scope)
        store.set_meta("probe", "3")
    assert store.scope_version(scope) == v4 and store.get_meta("probe") == "3"
    with store.unversioned():
        store.set_meta("probe", "4")
    assert store.scope_version(scope) == v4 and store.get_meta("probe") == "4"
    # A versioned row in the same transaction as unversioned ones counts exactly as it would alone.
    # 与不带版本行同处一个事务的带版本行，计数与它单独存在时完全一样。
    with store.transaction():
        store.touch_scope(scope)
        store.set_meta("probe", "5")
        with store.unversioned():
            store.set_meta("probe", "6")
    v5 = store.scope_version(scope)
    assert v5[0] == v4[0] and v5[2] == (("agent_id=bot&user_id=alice", 3),) and store.get_meta("probe") == "6"
    # Coverage both ways: the parent scope sees the child's writes and the child sees the parent's.
    # 双向覆盖：父范围看到子范围的写入，子范围看到父范围的写入。
    parent = Scope(user_id="alice")
    assert store.scope_version(parent)[2] == v5[2]
    with store.transaction():
        store.insert_source_event(SourceEvent(scope=parent, payload={"text": "parent"}, timestamp=T2))
    assert store.scope_version(scope)[2] == (("agent_id=bot&user_id=alice", 3), ("user_id=alice", 1))
    assert store.scope_version(Scope(user_id="zoe"))[2] == ()
    # A rolled-back transaction invalidates globally; reset clears the per-scope table.
    # 回滚的事务全局失效；reset 清空按范围的表。
    before = store.scope_version(scope)
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.insert_source_event(SourceEvent(scope=scope, payload={"text": "boom"}, timestamp=T3))
            raise RuntimeError("abort")
    rolled = store.scope_version(scope)
    assert rolled[0] == before[0] + 1 and rolled[2] == before[2]
    store.reset()
    assert store.scope_version(scope)[2] == () and store.scope_version(scope)[0] > rolled[0]
    store.close()


def test_context_timestamps_from_time_anchor(hms: HMS) -> None:
    """Row lines carry '[YYYY-MM-DD HH:MM]' from add(timestamp=...) in UTC; a row with no time_anchor renders without the bracket even when created_at is set; context_timestamps=False renders '- text' lines in the same order.
    记忆行带 add(timestamp=...) 的 UTC '[YYYY-MM-DD HH:MM]'；没有 time_anchor 的行即使有 created_at 也不带方括号；context_timestamps=False 时以同样顺序渲染 '- 文本' 行。

    Input / 输入：
        hms: The conftest facade.
        hms：conftest 门面。

    Output / 输出：
        Assertions over the rendered lines.
        对渲染行的断言。

    State effect / 状态影响：
        None outside in-memory stores.
        除内存存储外无副作用。
    """
    hms.add("I drink oat-milk lattes every morning.", timestamp="2026-04-02T10:30:00+02:00", **SCOPE)
    out = hms.context(QUESTION, filters=SCOPE)
    lines = out["rows_text"].split("\n")
    assert lines and all(line.startswith("- [2026-04-02 08:30] ") for line in lines), lines
    assert all(r["hms"]["time_anchor"] == "2026-04-02T08:30:00.000000+00:00" for r in out["rows"])
    # The builders on hand rows: no anchor → no bracket (created_at is a sort key only); timestamps off → plain lines.
    # 手写行上的构造器：无锚定 → 无方括号（created_at 只作排序键）；关闭时间戳 → 纯文本行。
    assert rows_text(HAND_ROWS) == HAND_TS and rows_text(HAND_ROWS, timestamps=False) == HAND_TD
    assert render_rows(sort_rows_by_time(dedup_rows(HAND_ROWS))) == HAND_TS
    assert render_rows([_row("  ", None, None), _row("x", "bad", None)]) == "- x"
    assert format_stamp(None) is None and format_stamp("not a date") is None
    assert format_stamp("2024-01-01T11:57:36.060000Z") == "2024-01-01 11:57" and format_stamp("2024-01-01T11:57:36+02:00") == "2024-01-01 09:57"
    plain = HMS(
        HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}, compiler={"context_timestamps": False}),
    )
    try:
        plain.add("I drink oat-milk lattes every morning.", timestamp="2026-04-02T10:30:00+02:00", **SCOPE)
        no_stamp = plain.context(QUESTION, filters=SCOPE)
        assert no_stamp["rows_text"] and "[" not in no_stamp["rows_text"]
        assert no_stamp["rows_text"] == render_rows(no_stamp["rows"], timestamps=False)
    finally:
        plain.close()


def test_context_builders_are_the_harness_builders() -> None:
    """benchmarks/egolife/reader_ab.py's dedup_rows / sort_rows_by_time / format_stamp / row_times / hybrid_context ARE the library functions, rows_ts_context equals rows_text and HYBRID_ROWS_HEADER equals the config default and ROWS_HEADER, so the harness's 70.1% context and HMS.context() cannot drift apart.
    benchmarks/egolife/reader_ab.py 的 dedup_rows / sort_rows_by_time / format_stamp / row_times / hybrid_context 就是库函数，rows_ts_context 等于 rows_text，HYBRID_ROWS_HEADER 等于配置默认值与 ROWS_HEADER，因此基准脚本的 70.1% 上下文与 HMS.context() 不可能分叉。

    Input / 输入：
        none (the seven hand rows).
        无（七条手写行）。

    Output / 输出：
        Identity and equality assertions.
        同一性与相等性断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert reader_ab.dedup_rows is ctx_mod.dedup_rows and reader_ab.sort_rows_by_time is ctx_mod.sort_rows_by_time
    assert reader_ab.format_stamp is ctx_mod.format_stamp and reader_ab.row_times is ctx_mod.row_times
    assert reader_ab.normalize_text is ctx_mod.normalize_text and reader_ab.hybrid_context is ctx_mod.hybrid_text
    assert reader_ab.HYBRID_ROWS_HEADER == CompilerConfig().context_rows_header == ROWS_HEADER == "Recent relevant memories:"
    assert reader_ab.rows_ts_context(HAND_ROWS) == rows_text(HAND_ROWS) == HAND_TS
    assert reader_ab.rows_td_context(HAND_ROWS) == rows_text(HAND_ROWS, timestamps=False) == HAND_TD
    assert [r["memory"] for r in dedup_rows(HAND_ROWS)] == [
        "Jake bought a hard drive.", "Shure fixed the coffee machine.", "Alice arrived.", "Bob left.", "Jake bought a hard drive and a USB.",
    ]  # fmt: skip
    assert reader_ab.hybrid_context("", "") == hybrid_text("", "") == "(none)\n\nRecent relevant memories:\n(none)"
    assert hybrid_text("# Card", " ") == "# Card\n\nRecent relevant memories:\n(none)"
    assert hybrid_text("  ", "- [2024-01-01 11:00] x", header="H:") == "(none)\n\nH:\n- [2024-01-01 11:00] x"
    assert reader_ab.build_prompt.__doc__ and "memories_first" in reader_ab.build_prompt.__doc__


def test_context_config_fields_and_overrides(hms: HMS) -> None:
    """CompilerConfig carries context_session_budget_tokens 1200 / context_rows_top_k 20 / context_rows_header / context_timestamps True, loads them from a dict, rejects an unknown key (CONFIG_001) and a non-positive budget; per-call top_k reaches search and a different budget_tokens recomputes the card instead of reusing it.
    CompilerConfig 带 context_session_budget_tokens 1200 / context_rows_top_k 20 / context_rows_header / context_timestamps True，可从字典加载，拒绝未知键（CONFIG_001）与非正预算；按次调用的 top_k 到达 search，不同的 budget_tokens 重算卡片而非复用。

    Input / 输入：
        hms: The conftest facade.
        hms：conftest 门面。

    Output / 输出：
        Assertions over defaults, from_dict, rejections and the overrides.
        对默认值、from_dict、拒绝与覆盖参数的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    defaults = CompilerConfig()
    assert (defaults.context_session_budget_tokens, defaults.context_rows_top_k) == (1200, 20)
    assert defaults.context_rows_header == ROWS_HEADER and defaults.context_timestamps is True
    loaded = HMSConfig.from_dict(
        {"compiler": {"context_session_budget_tokens": 600, "context_rows_top_k": 5, "context_rows_header": "Rows:", "context_timestamps": False}}
    )
    assert (loaded.compiler.context_session_budget_tokens, loaded.compiler.context_rows_top_k) == (600, 5)
    assert loaded.compiler.context_rows_header == "Rows:" and loaded.compiler.context_timestamps is False
    with pytest.raises(ConfigError) as unknown:
        HMSConfig.from_dict({"compiler": {"context_rows_topk": 5}})
    assert unknown.value.error_code == "CONFIG_001"
    with pytest.raises(ConfigError):
        HMSConfig.from_dict({"compiler": {"context_session_budget_tokens": 0}})
    _seed(hms)
    hms.add("Carol likes tea.", timestamp=T1, **SCOPE)
    full = hms.context(QUESTION, filters=SCOPE)
    one = hms.context(QUESTION, filters=SCOPE, top_k=1)
    assert len(full["search_rows"]) > 1 and len(one["search_rows"]) == 1 and one["session_reused"] is True
    assert one["rows_text"] == render_rows(one["rows"]) and one["rows"] == one["search_rows"]
    small = hms.context(QUESTION, filters=SCOPE, budget_tokens=200)
    assert small["session_reused"] is False and small["session_key"] == full["session_key"]
    assert hms.context(QUESTION, filters=SCOPE, budget_tokens=200)["session_reused"] is True
    assert hms.context(QUESTION, filters=SCOPE)["session_reused"] is False
    # as_of bounds the rows only: a reference instant before the T2 memory hides that row, the card is unchanged.
    # as_of 只约束记忆行：早于 T2 记忆的参考时刻隐藏该行，卡片不变。
    early = hms.context(QUESTION, filters=SCOPE, as_of="2026-04-05T00:00:00Z")
    assert all(r["hms"]["time_anchor"] <= "2026-04-05" for r in early["rows"]) and early["session_card"] == full["session_card"]
    assert early["session_reused"] is True
