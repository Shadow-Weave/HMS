"""Offline tests of the reader-side A/B harness (benchmarks/egolife/reader_ab.py + the lines contract of replay_ab.py): lines replay through ReplayLLM.generate_lines, the rows_td / rows_ts builders, the session-card memo, the two prompt orders, the card_prefix / hybrid_prefix memories-first probes, the baseline builders' delegation to real_arms / hms.compile, the locked meter and the parity check.
阅读端 A/B 基准脚本（benchmarks/egolife/reader_ab.py + replay_ab.py 的行式契约）的离线测试：经 ReplayLLM.generate_lines 的行式回放、rows_td / rows_ts 构造器、会话卡片记忆化、两种提示词顺序、card_prefix / hybrid_prefix 记忆在前探针、基线构造器对 real_arms / hms.compile 的委托、带锁台账与一致性检查。
"""

# Purpose (EN): Pin the harness semantics the smoke phase relies on without any network, dataset or fastembed model:
#               hash embedder, a three-window / three-question timeline built inline, recorded lines replies built inline,
#               and main() driven end to end offline (loaders monkeypatched to the inline fixture) for the CLI defaults,
#               the --no-answer ledger, the exit-4 cleanup and the deterministic atom ids.
# 目的（中）：在不依赖网络、数据集或 fastembed 模型的前提下钉住冒烟阶段所依赖的基准脚本语义：hash 嵌入器、
#            内联构造的三窗口 / 三题时间线、内联构造的行式记录回复，以及离线端到端驱动 main()（加载器打补丁指向内联夹具）
#            以检验命令行默认值、--no-answer 台账、退出码 4 的清理与确定性原子 id。

# Input (EN): benchmarks/egolife/{reader_ab, replay_ab, real_arms, timeline, windows, loader}.py; hms_elite.
# 输入（中）：benchmarks/egolife/{reader_ab, replay_ab, real_arms, timeline, windows, loader}.py；hms_elite。

# Output (EN): Assertions only.
# 输出（中）：仅断言。

# State effect (EN): Stubs `mem0` / `openai` in sys.modules when they are not installed (real_arms imports them at
#                    module import; nothing here calls them); the answer-pass test replaces DeepSeekClient / ask / load_api_key
#                    with fakes (no network, no .env read); writes only under tmp_path.
# 状态影响（中）：mem0 / openai 未安装时在 sys.modules 中放入桩模块（real_arms 在导入时引用它们；此处不调用）；答题阶段
#                测试用替身替换 DeepSeekClient / ask / load_api_key（无网络、不读 .env）；只写 tmp_path。

from __future__ import annotations

import functools
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EGOLIFE = ROOT / "benchmarks" / "egolife"


def _stub_bench_deps() -> None:
    """Put minimal `openai` / `mem0` stand-ins in sys.modules when the packages are absent.
    openai / mem0 缺失时在 sys.modules 放入最小替身。
    """
    try:
        import openai  # noqa: F401
    except ImportError:
        m = types.ModuleType("openai")
        for name in ("APITimeoutError", "APIStatusError", "APIConnectionError"):
            setattr(m, name, type(name, (Exception,), {}))
        m.OpenAI = type("OpenAI", (), {})  # type: ignore[attr-defined]
        sys.modules["openai"] = m
    try:
        import mem0  # noqa: F401
    except ImportError:
        tree = {
            "mem0": {}, "mem0.llms": {}, "mem0.llms.deepseek": {"DeepSeekLLM": type("DeepSeekLLM", (), {})},
            "mem0.memory": {}, "mem0.memory.main": {"Memory": type("Memory", (), {})},
            "mem0.vector_stores": {}, "mem0.vector_stores.base": {"VectorStoreBase": type("VectorStoreBase", (), {"keyword_search": None})},
        }  # fmt: skip
        for name, attrs in tree.items():
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m


_stub_bench_deps()
if str(EGOLIFE) not in sys.path:
    sys.path.insert(0, str(EGOLIFE))

import reader_ab  # noqa: E402
import replay_ab  # noqa: E402
from accounting import ApiCall, Meter  # noqa: E402
from loader import Question  # noqa: E402
from real_arms import BANK, AnswerResult, build_question_prompt, plan_batches, rows_to_context  # noqa: E402
from timeline import schedule  # noqa: E402
from windows import Window, clock_to_iso  # noqa: E402

from hms_elite.configs import prompts  # noqa: E402
from hms_elite.exceptions import LLMError  # noqa: E402

HASH = {"provider": "hash"}
CS = 100  # centiseconds per second / 每秒厘秒数


def _cs(h: int, m: int, s: int = 0) -> int:
    return (h * 3600 + m * 60 + s) * CS


def _window(i: int, start: int, end: int, text: str) -> Window:
    return Window(window_id=f"A1_JAKE_DAY1_11000000#{i:04d}", source="A1_JAKE_DAY1_11000000", day=1, start_cs=start, end_cs=end,
                  cue_from=i, cue_to=i, text=text, text_zh="")  # fmt: skip


def _question(qid: str, order: int, text: str, query_cs: int, target_cs: int, answer: str = "A") -> Question:
    options = {"A": "four USBs and a hard drive", "B": "a stopwatch", "C": "a coffee machine", "D": "nothing"}
    raw = {"question": text, **{f"choice_{k.lower()}": v for k, v in options.items()}}
    return Question(id=qid, order=order, type="EventRecall", question=text, options=options, query_day=1, query_cs=query_cs,
                    query_hhmmsscc=f"{query_cs // (3600 * CS):02d}{(query_cs // (60 * CS)) % 60:02d}0000", answer=answer, target_day=1,
                    targets=((1, target_cs),), raw=raw)  # fmt: skip


WINDOWS = [
    _window(0, _cs(11, 0), _cs(11, 0, 30), "Jake said there are four more USBs and a hard drive."),
    _window(1, _cs(11, 1), _cs(11, 1, 30), "Jake asked everyone to mark a timestamp with a stopwatch."),
    _window(2, _cs(11, 10), _cs(11, 10, 30), "Shure told Jake the coffee machine is broken."),
]
QUESTIONS = [
    _question("1", 0, "What did Jake say about the USBs?", _cs(11, 5), _cs(11, 0, 10)),
    _question("2", 1, "What did Jake ask everyone to mark?", _cs(11, 6), _cs(11, 1, 10), answer="B"),
    _question("3", 2, "What did Shure tell Jake about the coffee machine?", _cs(11, 20), _cs(11, 10, 10), answer="C"),
]
REPLIES = {
    "b0000": "OBS\n0\tF\tJake said there are four more USBs and a hard drive.\n1\tF\tJake asked everyone to mark a timestamp with a stopwatch.",
    "b0001": "OBS\n0\tF\tShure told Jake the coffee machine is broken.",
}


def _timeline():
    """steps, batches, recorded of the inline fixture (two ingesting steps around one that ingests nothing).
    内联夹具的 steps、batches、recorded（两个摄入步骤夹着一个不摄入的步骤）。
    """
    steps = schedule(QUESTIONS, WINDOWS)
    batches = plan_batches(steps, 25)
    recorded = replay_ab.Recorded()
    for bid, content in REPLIES.items():
        recorded.rows[(bid, None)].append({"batch_id": bid, "split": None, "attempt": 0, "finish_reason": "stop", "completion_tokens": 9, "content": content})
        recorded.batches_seen.add(bid)
        recorded.lines += 1
    return steps, batches, recorded


def _fake_run(tmp_path: Path, replies: dict[str, str]) -> Path:
    """A run directory main() accepts: batches.json of the inline timeline (batch size 25) and extract_raw_elite.jsonl
    holding `replies` as recorded lines rows.
    main() 能接受的 run 目录：内联时间线（批大小 25）的 batches.json，以及把 `replies` 写成记录行式行的 extract_raw_elite.jsonl。
    """
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    batches = plan_batches(schedule(QUESTIONS, WINDOWS), 25)
    (run / "batches.json").write_text(
        json.dumps([{"batch_id": b.batch_id, "position": b.position, "window_ids": b.window_ids} for b in batches]), encoding="utf-8"
    )
    rows = [{"arm": "elite", "stage": "extract", "batch_id": bid, "split": None, "attempt": 0, "finish_reason": "stop", "completion_tokens": 9,
             "content": content} for bid, content in replies.items()]  # fmt: skip
    (run / "extract_raw_elite.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run


def _offline_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point main() at the inline fixture: the loaders return QUESTIONS / WINDOWS and run_arm uses the hash embedder.
    把 main() 指向内联夹具：加载器返回 QUESTIONS / WINDOWS，run_arm 使用 hash 嵌入器。
    """
    monkeypatch.setattr(reader_ab, "load_questions", lambda path: list(QUESTIONS))
    monkeypatch.setattr(reader_ab, "load_transcripts", lambda root: [])
    monkeypatch.setattr(reader_ab, "build_windows", lambda transcripts: list(WINDOWS))
    monkeypatch.setattr(reader_ab, "run_arm", functools.partial(replay_ab.run_arm, embedder=HASH))


STALE = {
    "calls.jsonl": '{"stale": true}\n', "summary.json": '{"stale": true}', "summary.md": "# stale\n", "parity.json": "{}",
    "ingest_gaps.json": "{}", "per_question_rows.jsonl": '{"stale": true}\n',
}  # fmt: skip


def _seed_stale(out: Path) -> None:
    """Write an earlier run's outputs (STALE) into `out`.
    把先前 run 的输出（STALE）写进 `out`。
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, body in STALE.items():
        (out / name).write_text(body, encoding="utf-8")


def _main_args(run: Path, tmp_path: Path, *extra: str) -> list[str]:
    """The common main() argv of the offline runs.
    离线运行 main() 的公共 argv。
    """
    return ["--run", str(run), "--data-root", str(tmp_path), "--batch-size", "25", "--limit", "3", *extra]


def _extraction_user_prompt(texts: list[str]) -> str:
    return prompts.build_memory_extraction_prompt(
        existing_memories=[], new_messages=[{"idx": i, "role": "user", "text": t, "timestamp": "2024-01-01T11:00:00Z"} for i, t in enumerate(texts)],
        last_messages=[], observation_date="2024-01-01", current_date="2024-01-01", custom_instructions=None, events=[],
        known_constraints=[], recent_eps=[], candidates=[],
    )  # fmt: skip


def test_lines_contract_and_generate_lines_replay():
    """'lines' is a replay arm / contract / extraction task; ReplayLLM.generate_lines returns the registered raw text byte-identical (no trailing newline added) and counts the call; an unregistered slice raises LLMError(LLM_003) and counts a miss.
    'lines' 是回放的臂 / 契约 / 抽取任务；ReplayLLM.generate_lines 逐字节返回登记的原始文本（不补尾随换行）并计数；
    未登记的切片抛出 LLMError(LLM_003) 并计一次未命中。

    Input / 输入：
        A ReplayLLM('lines') with one registered slice; the library's lines system prompt and user prompt builder.
        登记了一个切片的 ReplayLLM('lines')；库的行式系统提示词与用户提示词构造器。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert "lines" in replay_ab.ARMS and "lines" in replay_ab.REPLIES_CONTRACTS
    assert replay_ab.ANSWER_LABEL["lines"] == "replay_lines"
    assert "memory_extraction_lines" in replay_ab.EXTRACTION_TASKS
    llm = replay_ab.ReplayLLM("lines")
    texts = [w.text for w in WINDOWS[:2]]
    llm.register(texts, REPLIES["b0000"], ("b0000", None))
    llm.expect = ("b0000", None)
    out = llm.generate_lines("memory_extraction_lines", system=prompts.MEMORY_EXTRACTION_LINES_PROMPT, user=_extraction_user_prompt(texts))
    assert out == REPLIES["b0000"] and not out.endswith("\n")
    assert (llm.lines_calls, llm.lines_misses, llm.mismatches) == (1, 0, 0)
    assert llm.tasks == {"memory_extraction_lines": 1}
    with pytest.raises(LLMError) as err:
        llm.generate_lines("memory_extraction_lines", system=prompts.MEMORY_EXTRACTION_LINES_PROMPT, user=_extraction_user_prompt(["never registered"]))
    assert err.value.error_code == "LLM_003"
    assert (llm.lines_calls, llm.lines_misses) == (2, 1)


def test_rows_td_dedup_substring_and_time_sort_and_rows_ts_stamps():
    """rows_td drops case / whitespace-insensitive duplicates (first survives) and rows whose normalized text is a proper substring of another surviving row, then sorts by time_anchor ascending with created_at as fallback and unknown times last in input order, without timestamps; rows_ts adds '[YYYY-MM-DD HH:MM] ' from time_anchor only.
    rows_td 去掉大小写 / 空白不敏感的重复（首个保留）与归一化文本是另一存活行真子串的行，再按 time_anchor 升序
    （回退 created_at；未知时间按输入顺序排最后）排序且不带时间戳；rows_ts 只从 time_anchor 加 '[YYYY-MM-DD HH:MM] '。

    Input / 输入：
        Seven hand-written search rows.
        七条手写检索行。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """

    def row(text: str, anchor: str | None, created: str | None, bid: str = "b0000") -> dict:
        return {"id": text[:4], "memory": text, "metadata": {"batch_id": bid}, "score": 0.5, "created_at": created, "hms": {"time_anchor": anchor}}

    raw = [
        row("Jake bought a hard drive.", "2024-01-01T12:00:00.000000+00:00", "2024-01-01T12:00:00.000000+00:00"),
        row("  jake BOUGHT a hard   drive. ", "2024-01-01T11:00:00.000000+00:00", "2024-01-01T11:00:00.000000+00:00"),
        row("a hard drive", "2024-01-01T09:00:00.000000+00:00", "2024-01-01T09:00:00.000000+00:00"),
        row("Shure fixed the coffee machine.", None, "2024-01-01T10:30:00.000000+00:00"),
        row("Alice arrived.", None, None),
        row("Bob left.", None, None),
        row("Jake bought a hard drive and a USB.", "2024-01-01T08:00:00.000000+00:00", "2024-01-01T08:00:00.000000+00:00"),
    ]
    kept = reader_ab.dedup_rows(raw)
    assert [r["memory"] for r in kept] == [
        "Jake bought a hard drive.", "Shure fixed the coffee machine.", "Alice arrived.", "Bob left.", "Jake bought a hard drive and a USB.",
    ]
    assert reader_ab.rows_td_context(raw) == (
        "- Jake bought a hard drive and a USB.\n- Shure fixed the coffee machine.\n- Jake bought a hard drive.\n- Alice arrived.\n- Bob left."
    )
    assert "[" not in reader_ab.rows_td_context(raw)
    assert reader_ab.rows_ts_context(raw) == (
        "- [2024-01-01 08:00] Jake bought a hard drive and a USB.\n- Shure fixed the coffee machine.\n"
        "- [2024-01-01 12:00] Jake bought a hard drive.\n- Alice arrived.\n- Bob left."
    )
    assert reader_ab.format_stamp(None) is None and reader_ab.format_stamp("not a date") is None
    assert reader_ab.format_stamp("2024-01-01T11:57:36.060000Z") == "2024-01-01 11:57"
    # The baseline delegates to real_arms.rows_to_context over the recall projection, rank order untouched.
    # 基线委托给召回投影上的 real_arms.rows_to_context，排名顺序不动。
    assert reader_ab.rows_context(raw) == rows_to_context(reader_ab.project_rows(raw))
    assert reader_ab.rows_context(raw).split("\n")[0] == "- Jake bought a hard drive."
    assert reader_ab.project_rows(raw)[0] == {"batch_id": "b0000", "text": "Jake bought a hard drive.", "score": 0.5, "memory_id": "Jake"}


def test_prompt_orders_share_prefix_only_when_memories_first():
    """'question_first' is real_arms.answer's prompt verbatim (question prompt, then 'Memories:'); 'memories_first' puts the Memories block first and the question prompt last so two questions with one card share the prefix; blank context renders '(none)'; an unknown order raises.
    'question_first' 是 real_arms.answer 的提示词原文（题目提示词在前，再 'Memories:'）；'memories_first' 把 Memories 段放最前、
    题目提示词放最后，使同一卡片的两道题共享前缀；空白上下文渲染为 '(none)'；未知顺序抛出异常。

    Input / 输入：
        Two inline questions and one card text.
        两道内联题目与一段卡片文本。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    q1, q2 = QUESTIONS[0], QUESTIONS[1]
    card = "# Work State Card\n## Valid Facts\n- Jake bought a hard drive."
    qf = reader_ab.build_prompt(q1, card, "question_first")
    assert qf == build_question_prompt(q1.raw) + "\n\nMemories:\n" + card
    mf1 = reader_ab.build_prompt(q1, card, "memories_first")
    mf2 = reader_ab.build_prompt(q2, card, "memories_first")
    prefix = "Memories:\n" + card + "\n\n"
    assert mf1.startswith(prefix) and mf2.startswith(prefix) and mf1 != mf2
    assert mf1 == prefix + build_question_prompt(q1.raw) and mf1.endswith("D. nothing")
    assert reader_ab.build_prompt(q1, "   ", "question_first").endswith("Memories:\n(none)")
    assert reader_ab.build_prompt(q1, "", "memories_first").startswith("Memories:\n(none)\n\n")
    assert reader_ab.PROMPT_ORDER["session_prefix"] == "memories_first"
    assert reader_ab.PROMPT_ORDER["card_prefix"] == reader_ab.PROMPT_ORDER["hybrid_prefix"] == "memories_first"
    assert reader_ab.MEMORIES_FIRST_VARIANTS == ("session_prefix", "card_prefix", "hybrid_prefix")
    assert all(reader_ab.PROMPT_ORDER[v] == "question_first" for v in reader_ab.VARIANTS if v not in reader_ab.MEMORIES_FIRST_VARIANTS)
    with pytest.raises(ValueError):
        reader_ab.build_prompt(q1, card, "sideways")


def test_run_arm_lines_hook_memo_and_baseline_delegation(tmp_path):
    """run_arm('lines', contract 'lines', hash embedder) replays the two recorded replies through generate_lines with no miss and stores three atoms; the hook captures all eleven variants per question at the step's store state: 'rows' equals run_arm's own rows_to_context text, the session card is compiled once per ingesting step (twice) and reused verbatim by the question asked before the next ingest (once), session_prefix carries the same text, and reopening the store reproduces card_1200 / card_400 / rows via hms.compile(...).to_text() and hms.search + real_arms.rows_to_context.
    run_arm('lines'，契约 'lines'，hash 嵌入器）经 generate_lines 无未命中地回放两条记录回复并存下三个原子；钩子在每步的库状态下
    为每道题捕获全部十一种变体：'rows' 等于 run_arm 自己的 rows_to_context 文本，会话卡片每个摄入步骤编译一次（两次）并被下次
    摄入前提出的题目原样复用（一次），session_prefix 携带相同文本，重开库后 hms.compile(...).to_text() 与 hms.search +
    real_arms.rows_to_context 复现 card_1200 / card_400 / rows。

    Input / 输入：
        The inline timeline and recorded replies; tmp_path for the store.
        内联时间线与记录回复；tmp_path 作库目录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes a SQLite store, retain_lines.jsonl and entity_axis_lines.jsonl under tmp_path.
        在 tmp_path 下写 SQLite 库、retain_lines.jsonl 与 entity_axis_lines.jsonl。
    """
    steps, batches, recorded = _timeline()
    assert [len(s.new_windows) for s in steps] == [2, 0, 1]
    assert [(b.batch_id, b.position) for b in batches] == [("b0000", 0), ("b0001", 2)]
    hook = reader_ab.ReaderContexts()
    res = replay_ab.run_arm("lines", steps, batches, recorded, work_root=tmp_path / "work", out=tmp_path, speaker_from_subject=True,
                            entities_from_mentions=True, contract="lines", context_hook=hook, embedder=HASH)  # fmt: skip
    assert res["replay"]["tasks"] == {"memory_extraction_lines": 2}
    assert (res["replay"]["lines_calls"], res["replay"]["lines_misses"], res["replay"]["mismatches"]) == (2, 0, 0)
    assert [r["added"] for r in res["retain"]] == [2, 1] and res["atoms"] == 3
    assert [r["id"] for r in res["records"]] == ["1", "2", "3"]
    assert set(hook.by_question) == {"1", "2", "3"}
    for rec in res["records"]:
        ctx = hook.by_question[rec["id"]]
        assert set(ctx) == set(reader_ab.VARIANTS)
        assert ctx["rows"]["context"] == rec["context"] and ctx["rows"]["context_chars"] == rec["context_chars"]
        assert ctx["rows"]["hit_batch"] == rec["hit_batch"] and ctx["rows"]["returned_batches"] == rec["returned_batches"]
        assert ctx["card_1200"]["context"].startswith("# Work State Card") and ctx["card_1200"]["card"]["budget_tokens"] == 1200
        assert ctx["card_400"]["card"]["budget_tokens"] == 400
        assert ctx["session_prefix"]["context"] == ctx["session_1200"]["context"]
        assert ctx["session_1200"]["est_tokens"] > 0 and ctx["rows"]["est_tokens"] > 0
    assert hook.rows_baseline_mismatch == 0
    # Memo: compiled at steps 0 and 2, reused at step 1; the reused text is byte-identical and keyed by the last ingested window.
    # 记忆化：第 0、2 步编译，第 1 步复用；复用文本逐字节相同，键为最后摄入的窗口。
    assert (hook.session_compiles, hook.session_reuses, hook.session_without_ingest) == (2, 1, 0)
    s1, s2, s3 = (hook.by_question[q]["session_1200"] for q in ("1", "2", "3"))
    assert s1["context"] == s2["context"] and s1["session"]["key"] == s2["session"]["key"] == WINDOWS[1].window_id
    assert (s1["session"]["reused"], s2["session"]["reused"], s3["session"]["reused"]) == (False, True, False)
    assert s3["session"]["key"] == WINDOWS[2].window_id and s3["context"] != s1["context"]
    assert s1["session"]["as_of"] == WINDOWS[1].start_iso and WINDOWS[1].text in s1["context"]
    # Baseline builders delegate to the same calls as real_arms: reopen the closed store (its final state is the state at question 3).
    # 基线构造器委托给与 real_arms 相同的调用：重开已关闭的库（其最终状态即第 3 题时的状态）。
    q3 = QUESTIONS[2]
    iso3 = clock_to_iso(q3.query_day, q3.query_cs)
    hms = replay_ab.build_hms(tmp_path / "work" / "lines", "lines", embedder=HASH)
    try:
        assert hms.compile(q3.question, filters={"user_id": BANK}, as_of=iso3).to_text() == hook.by_question["3"]["card_1200"]["context"]
        assert hms.compile(q3.question, filters={"user_id": BANK}, as_of=iso3, budget_tokens=400).to_text() == hook.by_question["3"]["card_400"]["context"]
        assert reader_ab.compile_card(hms, q3.question, iso3, None)["text"] == hook.by_question["3"]["card_1200"]["context"]
        raw = hms.search(q3.question, filters={"user_id": BANK}, top_k=20, threshold=0.1, reference_date=iso3)["results"]
        assert rows_to_context([{"text": r["memory"]} for r in raw]) == hook.by_question["3"]["rows"]["context"]
        assert reader_ab.rows_context(raw) == hook.by_question["3"]["rows"]["context"]
        assert "[2024-01-01 " in hook.by_question["3"]["rows_ts"]["context"]
    finally:
        hms.close()
    # Variant records and an unanswered summary; parity against a self-consistent oracle, then one perturbed field.
    # 变体记录与未答题汇总；先对自洽 oracle 做一致性检查，再扰动一个字段。
    records = reader_ab.variant_records(reader_ab.VARIANTS, res["records"], hook)
    assert {v: len(r) for v, r in records.items()} == {v: 3 for v in reader_ab.VARIANTS}
    assert records["session_prefix"][0]["prompt_order"] == "memories_first" and records["rows"][0]["gold"] == "A"
    summary = reader_ab.variant_summary("card_800", records["card_800"], Meter())
    assert summary["answered_n"] == 0 and summary["answer_accuracy"] is None and summary["cny_per_204"] == {"offpeak": None, "peak": None}
    assert summary["card"]["budget_tokens"] == 800 and summary["hit_batch"] is not None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows_oracle = [{"id": qid, **b} for qid, b in hook.baseline.items()]
    card_oracle = [
        {"id": qid, "returned_batches": c["card_1200"]["returned_batches"], "context_chars": c["card_1200"]["context_chars"],
         "card": {k: c["card_1200"]["card"][k] for k in ("used_tokens", "items", "mode")}}
        for qid, c in hook.by_question.items()
    ]  # fmt: skip
    (run_dir / "questions_elite_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows_oracle) + "\n", encoding="utf-8")
    (run_dir / "questions_elite_card.jsonl").write_text("\n".join(json.dumps(r) for r in card_oracle) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"engines": {"elite": {"stats": {"stored_count": res["atoms"]}}}}), encoding="utf-8")
    retain_oracle = [{"batch_id": r["batch_id"], "added": r["added"], "events": {"ADD": r["added"]}} for r in res["retain"]]
    (run_dir / "retain_elite.jsonl").write_text("\n".join(json.dumps(r) for r in retain_oracle) + "\n", encoding="utf-8")
    parity = reader_ab.parity_check(run_dir, hook, atoms=res["atoms"], complete=True, retain=res["retain"])
    assert parity["ok"] and parity["rows"]["compared"] == 3 and parity["card_1200"]["compared"] == 3 and parity["stored_count"]["match"] is True
    assert parity["retain"]["compared"] == 2 and parity["retain"]["mismatches"] == 0 and parity["rows"]["top_scores_exact_mismatches"] == 0
    info = parity["retain"]["informational"]
    assert info["warnings_count_diff_batches"] == 0 and info["items_skipped_diff_batches"] == 0 and info["first"] == []
    # A hook that never ran card_1200 skips the card section instead of reporting every question as a missing record.
    # 没跑 card_1200 的钩子跳过卡片节，而不是把每道题报成缺记录。
    rows_only = reader_ab.ReaderContexts(variants=("rows",))
    rows_only.baseline = dict(hook.baseline)
    rows_only.by_question = {qid: {"rows": ctx["rows"]} for qid, ctx in hook.by_question.items()}
    p_rows = reader_ab.parity_check(run_dir, rows_only, atoms=res["atoms"], complete=True, retain=res["retain"])
    assert p_rows["ok"] and p_rows["card_1200"] == {"compared": 0, "mismatches": 0, "questions_mismatched": 0, "first": [], "by_field": {},
                                                    "skipped": "card_1200 not run"}  # fmt: skip
    assert parity["oracle_note"].startswith("necessary, not sufficient") and parity["card_1200"]["next_action"]["questions"] == 0
    # No ingest gap in this replay; a synthetic api_error sub or a lines miss is fatal and names the batch.
    # 本次回放没有摄入缺口；合成的 api_error 子批次或行式未命中即致命并点名批次。
    gaps = reader_ab.ingest_gaps(res)
    assert gaps == {"fatal": False, "lines_misses": 0, "api_error": [], "missing": [], "batch_ids": []}
    broken = {"replay": {"lines_misses": 1}, "retain": [{"batch_id": "b0001", "subs": [{"split": None, "outcome": "api_error", "error": "LLMError: x"}]},
                                                       {"batch_id": "b0000", "subs": [{"split": 0, "outcome": "missing"}]}]}  # fmt: skip
    g = reader_ab.ingest_gaps(broken)
    assert g["fatal"] and g["lines_misses"] == 1 and g["batch_ids"] == ["b0000", "b0001"] and g["api_error"][0]["error"] == "LLMError: x"
    # Memo recompute is keyed on store changes (ADD + UPDATE events), not len(rows): a scheduled step whose retain
    # changed nothing reuses the card; run_arm's records carry the histogram and the last stored window.
    # 记忆化重算以库的变化（ADD + UPDATE 事件）为准而非 len(rows)：安排了窗口但 retain 未改变库的步骤复用卡片；run_arm
    # 的记录携带事件直方图与最后入库的窗口。
    assert [r["retain_events"] for r in res["records"]] == [{"ADD": 2}, {}, {"ADD": 1}]
    assert [r["retain_last_stored_window"] for r in res["records"]] == [WINDOWS[1].window_id, None, WINDOWS[2].window_id]
    assert [reader_ab.store_changes(r) for r in res["records"]] == [2, 0, 1] and reader_ab.store_changes({"retain_added": 4}) == 4
    assert reader_ab.store_changes({"retain_events": {"NONE": 3}, "retain_added": 3}) == 0
    assert s1["session"]["store_changes"] == 2 and hook.session_query_fallback == 0
    before = (hook.session_compiles, hook.session_reuses)
    skipped = hook._session(None, steps[2], set(), changed=0, last_window=None)  # type: ignore[arg-type]
    assert skipped["session"]["reused"] is True and skipped["session"]["key"] == WINDOWS[2].window_id and skipped["context"] == s3["context"]
    assert hook.session_ingest_skipped == 1 and (hook.session_compiles, hook.session_reuses) == (before[0], before[1] + 1)
    assert reader_ab.parity_caveats(parity) == [] and reader_ab.parity_caveats(None)[0].startswith("No --parity-check")
    # A 4th-decimal score drift stays within tolerance (informational); a card field, a batch count or the stored_count fails.
    # 第 4 位小数的分数漂移在容差内（仅信息）；卡片字段、批次计数或 stored_count 的差异则判为失败。
    rows_oracle[0]["top_scores"] = [round(x + 0.0001, 4) for x in rows_oracle[0]["top_scores"]]
    (run_dir / "questions_elite_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows_oracle) + "\n", encoding="utf-8")
    card_oracle[1]["card"]["items"] += 1
    (run_dir / "questions_elite_card.jsonl").write_text("\n".join(json.dumps(r) for r in card_oracle) + "\n", encoding="utf-8")
    retain_oracle[1]["added"] += 1
    (run_dir / "retain_elite.jsonl").write_text("\n".join(json.dumps(r) for r in retain_oracle) + "\n", encoding="utf-8")
    parity = reader_ab.parity_check(run_dir, hook, atoms=res["atoms"] + 1, complete=True, retain=res["retain"])
    assert not parity["ok"] and parity["rows"]["mismatches"] == 0 and parity["rows"]["top_scores_exact_mismatches"] == 1
    assert parity["card_1200"]["mismatches"] == 1 and parity["card_1200"]["first"][0]["id"] == "2"
    assert parity["card_1200"]["first"][0]["field"] == "items" and parity["stored_count"]["match"] is False
    assert parity["retain"]["mismatches"] == 1 and parity["retain"]["first"][0] == {"id": "b0001", "position": None, "field": "added", "recorded": 2, "replayed": 1}
    assert reader_ab.parity_check(run_dir, hook, atoms=res["atoms"] + 1, complete=False)["stored_count"]["match"] is None
    caveats = reader_ab.parity_caveats(parity)
    assert len(caveats) == 3 and caveats[0].startswith("PARITY: card_1200 differs") and "1/3 questions" in caveats[0]
    versions = reader_ab.package_versions()
    assert set(versions) == set(reader_ab.VERSIONED_PACKAGES) and all(v is None or isinstance(v, str) for v in versions.values())


def test_rows_td_hit_and_returned_batches_come_from_kept_rows():
    """rows_td / rows_ts describe the rows that survive dedup: when the substring rule drops every row of the gold batch, hit_batch is False and returned_batches / returned_count shrink to the kept rows, while hit_batch_top20 and returned_batches_top20 carry the shared top-20 facts; 'rows' keeps the top-20 facts unchanged.
    rows_td / rows_ts 描述去重后存活的行：子串规则丢掉 gold 批次的全部行时 hit_batch 为 False，returned_batches / returned_count
    缩为保留行，而 hit_batch_top20 与 returned_batches_top20 携带共享的 top-20 事实；'rows' 的 top-20 事实不变。

    Input / 输入：
        A rows-only hook called directly with two hand-written search rows (gold row a substring of the other).
        直接以两条手写检索行（gold 行是另一行的子串）调用只含行式变体的钩子。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    steps = schedule(QUESTIONS, WINDOWS)
    hook = reader_ab.ReaderContexts(variants=("rows", "rows_td", "rows_ts"))
    raw = [
        {"id": "g", "memory": "a hard drive", "metadata": {"batch_id": "b0000"}, "score": 0.9, "created_at": None, "hms": {"time_anchor": None}},
        {"id": "o", "memory": "Jake bought a hard drive.", "metadata": {"batch_id": "b0001"}, "score": 0.8, "created_at": None, "hms": {"time_anchor": None}},
    ]
    rec = {"gold_batches": ["b0000"], "hit_batch": True, "returned_batches": ["b0000", "b0001"], "returned_count": 2, "top_scores": [0.9, 0.8],
           "context_chars": len(rows_to_context([{"text": r["memory"]} for r in raw])), "context": rows_to_context([{"text": r["memory"]} for r in raw]),
           "retain_added": 0}  # fmt: skip
    hook(None, steps[0], "2024-01-01T11:05:00+00:00", raw, rec)  # type: ignore[arg-type]
    out = hook.by_question["1"]
    assert out["rows"]["hit_batch"] is True and out["rows"]["returned_batches"] == ["b0000", "b0001"] and out["rows"]["returned_count"] == 2
    for v in ("rows_td", "rows_ts"):
        assert out[v]["hit_batch"] is False and out[v]["returned_batches"] == ["b0001"] and out[v]["returned_count"] == 1
        assert out[v]["hit_batch_top20"] is True and out[v]["returned_batches_top20"] == ["b0000", "b0001"]
        assert {k: out[v]["rows"][k] for k in ("rows_in", "rows_kept", "rows_dropped", "stamps_distinct", "batches_distinct")} == {
            "rows_in": 2, "rows_kept": 1, "rows_dropped": 1, "stamps_distinct": 0, "batches_distinct": 1,
        }  # fmt: skip
        assert "a hard drive" in out[v]["context"] and "- a hard drive" not in out[v]["context"]
    assert hook.rows_baseline_mismatch == 0
    summary = reader_ab.variant_summary("rows_td", [dict(out["rows_td"], id="1", excluded_future=False)], Meter())
    assert summary["hit_batch"] == 0.0 and summary["hit_batch_top20"] == 1.0 and "kept" in summary["hit_note"]
    assert summary["rows"] == {"rows_in_total": 2, "rows_kept_total": 1, "rows_dropped_total": 1, "questions_with_drops": 1,
                               "stamps_distinct_mean": 0.0, "batches_distinct_mean": 1.0}  # fmt: skip
    assert reader_ab.variant_summary("rows", [dict(out["rows"], id="1", excluded_future=False)], Meter())["hit_batch_top20"] is None


def test_cny_per_question_uses_priced_records_and_session_hit_split():
    """cny_per_question divides the ledger CNY by priced_n (records whose last attempt carried usage), the same denominator as prompt_tokens_mean, so a request whose every attempt failed does not dilute it; session variants split the cache-hit share by first-at-step vs reused-card questions.
    cny_per_question 以 priced_n（最后一次尝试带 usage 的记录）除台账人民币，与 prompt_tokens_mean 同分母，全部尝试失败的
    请求不稀释它；会话变体把缓存命中占比按"该步首题 / 复用卡片题"拆分。

    Input / 输入：
        Three session_prefix records (two priced, one failed) and a meter holding their calls.
        三条 session_prefix 记录（两条已计价、一条失败）与持有其调用的台账。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    meter = Meter()
    calls = [
        ApiCall(arm="reader_session_prefix", stage="answer", question_id="1", attempt=0, started_utc="t", latency_ms=1.0, status="ok",
                prompt_tokens=1000, completion_tokens=1, cache_hit_tokens=0, cache_miss_tokens=1000, rate_tier="offpeak"),
        ApiCall(arm="reader_session_prefix", stage="answer", question_id="2", attempt=0, started_utc="t", latency_ms=1.0, status="ok",
                prompt_tokens=1000, completion_tokens=1, cache_hit_tokens=900, cache_miss_tokens=100, rate_tier="offpeak"),
        ApiCall(arm="reader_session_prefix", stage="answer", question_id="3", attempt=0, started_utc="t", latency_ms=1.0, status="timeout"),
    ]  # fmt: skip
    for c in calls:
        meter.add(c)
    base = {"context": "x", "context_chars": 1, "est_tokens": 1, "hit_batch": True, "excluded_future": False, "gold": "A"}
    recs = [
        dict(base, id="1", session={"key": "w1", "reused": False}), dict(base, id="2", session={"key": "w1", "reused": True}),
        dict(base, id="3", session={"key": "w1", "reused": True}),
    ]
    reader_ab.apply_answer(recs[0], AnswerResult(letter="A", raw="A", status="ok", prompt_chars=5, calls=[calls[0]], model_ms=1.0, wall_ms=1.0))
    reader_ab.apply_answer(recs[1], AnswerResult(letter="B", raw="B", status="ok", prompt_chars=5, calls=[calls[1]], model_ms=1.0, wall_ms=1.0))
    reader_ab.apply_answer(recs[2], AnswerResult(letter=None, raw=None, status="timeout", prompt_chars=5, calls=[calls[2]], model_ms=1.0, wall_ms=1.0))
    s = reader_ab.variant_summary("session_prefix", recs, meter)
    assert s["answered_n"] == 3 and s["priced_n"] == 2 and s["prompt_tokens_mean"] == 1000.0
    total = reader_ab.cny_cost(900, 1100, 2, "offpeak")
    assert s["cny_total"]["offpeak"] == round(total, 6) and s["cny_per_question"]["offpeak"] == round(total / 2, 6)
    assert s["cache_hit_share"] == round(900 / 2000, 4)
    assert s["session"] == {"reused": 2, "recomputed": 1, "cache_hit_share_first_at_step": 0.0, "cache_hit_share_reused_card": 0.9}
    md = reader_ab.markdown({
        "run_dir": "r", "started_utc": "t", "questions": 3, "steps_run": 3, "batches": 1, "replies_replayed": 1, "replies_missing": 0, "atoms": 1,
        "replay": {"lines_calls": 1, "lines_misses": 0}, "session_memo": {"compiles": 1, "reuses": 2, "without_ingest": 0, "ingest_skipped": 0, "rows_baseline_mismatch": 0},
        "config": {"package_versions": {"fastembed": "0.8.0"}}, "variants_run": ["session_prefix"], "variants": {"session_prefix": s},
        "answer_pass": {}, "parity": None, "method": "m", "caveats": ["c"],
    })  # fmt: skip
    assert "| session_prefix | memories_first | 33.3% (1/3) | 100.0% | - | 1 | 1000.0 | 45.0% | 1 / 2; 0.0% / 90.0% | 0.2048 / 0.4096 |" in md
    assert "priced_n" in md and '"fastembed": "0.8.0"' in md and "cache-neutral" in md
    # Cache-neutral CNY prices every prompt token as a miss; a partial variant (answered < n) withholds cny_per_204.
    # 缓存中性人民币把所有提示词 token 按未命中计；部分回答的变体（answered < n）不给出 cny_per_204。
    assert s["cny_total_no_cache"]["offpeak"] == round(reader_ab.cny_cost(0, 2000, 2, "offpeak"), 6) and s["partial"] is False
    assert s["cny_per_204"]["offpeak"] == round(total / 2 * 204, 4)
    partial = reader_ab.variant_summary("session_prefix", recs + [dict(base, id="4", session={"key": "w1", "reused": True})], meter)
    assert partial["partial"] is True and partial["cny_per_204"] == {"offpeak": None, "peak": None} and partial["cny_per_204_no_cache"]["peak"] is None
    assert partial["cny_per_question"]["offpeak"] == s["cny_per_question"]["offpeak"]
    caveats = reader_ab.truncation_caveats({"session_prefix": partial, "rows": dict(s, answered_n=0, n=3, partial=False)}, {"stopped": True, "reason": "cap"})
    assert len(caveats) == 2 and "session_prefix answered 3/4" in caveats[0] and "no answers for rows" in caveats[1]
    assert reader_ab.truncation_caveats({"session_prefix": partial}, {"stopped": True, "reason": "--no-answer"}) == []


def test_cache_probe_picks_shortest_distinct_questions():
    """cache_probe.pick_questions returns the n shortest pairwise distinct question texts (stable by file order on ties), honours max_chars, and never repeats a normalized text.
    cache_probe.pick_questions 返回 n 道最短且两两不同的题干（同长按文件顺序），遵守 max_chars，且不重复归一化文本。

    Input / 输入：
        Five inline questions with one duplicate text.
        五道内联题目，含一个重复题干。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    import cache_probe

    qs = [
        _question("a", 0, "What did Jake say about the USBs?", _cs(11, 5), _cs(11, 0)),
        _question("b", 1, "Who arrived?", _cs(11, 5), _cs(11, 0)),
        _question("c", 2, "  who ARRIVED? ", _cs(11, 5), _cs(11, 0)),
        _question("d", 3, "Who left?", _cs(11, 5), _cs(11, 0)),
        _question("e", 4, "Where is it?", _cs(11, 5), _cs(11, 0)),
    ]
    assert [q.id for q in cache_probe.pick_questions(qs, 3)] == ["d", "b", "e"]
    assert [q.id for q in cache_probe.pick_questions(qs, 10)] == ["d", "b", "e", "a"]
    assert [q.id for q in cache_probe.pick_questions(qs, 10, max_chars=12)] == ["d", "b", "e"]
    assert cache_probe.DEFAULT_N == 12 and cache_probe.PASSES[0].startswith("a_memories_first")
    # Pass c takes the (N+1)-th question so its first call is cold for the question suffix; with only N distinct
    # texts it falls back to qs[0] and says the first call is expected warm.
    # pass c 取第 N+1 道题，使其首次调用对题目后缀是冷的；只有 N 道不同题干时退回 qs[0] 并说明首次调用预期为热。
    picked = cache_probe.pick_questions(qs, 3 + 1)
    head, rep_q, cold = cache_probe.split_questions(picked, 3)
    assert [q.id for q in head] == ["d", "b", "e"] and rep_q.id == "a" and cold is True and rep_q.id not in {q.id for q in head}
    head, rep_q, cold = cache_probe.split_questions(cache_probe.pick_questions(qs, 4 + 1), 4)
    assert [q.id for q in head] == ["d", "b", "e", "a"] and rep_q.id == "d" and cold is False


def test_apply_answer_and_locked_meter():
    """apply_answer copies the last attempt's usage / tier into the record and judges correctness against gold; LockedMeter serialises add / next_is_cold across threads (every call kept, one cold start per arm) and reports the spent USD.
    apply_answer 把最后一次尝试的 usage / 档位写入记录并按 gold 判对错；LockedMeter 跨线程串行化 add / next_is_cold
    （每次调用都保留、每臂一次冷启动）并报告已花费美元。

    Input / 输入：
        A synthetic AnswerResult with two attempts; 8 threads x 50 ApiCalls.
        含两次尝试的合成 AnswerResult；8 线程 x 50 条 ApiCall。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    first = ApiCall(arm="reader_rows", stage="answer", question_id="1", attempt=0, started_utc="t", latency_ms=10.0, status="timeout")
    last = ApiCall(arm="reader_rows", stage="answer", question_id="1", attempt=1, started_utc="t", latency_ms=20.0, status="ok",
                   prompt_tokens=450, completion_tokens=1, cache_hit_tokens=64, cache_miss_tokens=386, rate_tier="peak")  # fmt: skip
    meter = reader_ab.LockedMeter()
    meter.add(first)
    meter.add(last)
    rec = {"id": "1", "gold": "B", "context": "- x"}
    reader_ab.apply_answer(rec, AnswerResult(letter="B", raw="B", status="ok", prompt_chars=12, calls=[first, last], model_ms=30.0, wall_ms=31.0))
    assert rec["correct"] is True and rec["answer_attempts"] == 2 and rec["tier"] == "peak"
    assert rec["answer_usage"]["cache_hit_tokens"] == 64 and rec["answer_usage"]["unpriced_attempts"] == 1
    assert rec["answer_usage"]["cost_usd_all_attempts"] == round(last.cost_usd, 8) and rec["latency_ms"] == {"model": 30.0, "wall": 31.0}
    assert meter.spent_usd() == pytest.approx(last.cost_usd)
    assert reader_ab.cny_cost(64, 386, 1, "offpeak") == pytest.approx((64 * 0.02 + 386 * 1.0 + 1 * 4.0) / 1e6)
    assert reader_ab.cny_cost(None, 1, 1, "peak") is None

    colds: list[bool] = []

    def worker(k: int) -> None:
        for i in range(50):
            colds.append(meter.next_is_cold("arm"))
            meter.add(ApiCall(arm="arm", stage="answer", question_id=str(i), attempt=0, started_utc="t", latency_ms=1.0, status="ok"))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(meter.calls) == 2 + 400 and colds.count(True) == 1


def test_build_plan_truncates_per_variant_and_stale_outputs_and_id_factory(tmp_path):
    """build_plan splits --max-answer-requests evenly across variants (each keeps its leading records; the remainder is unused; below one per variant the plan is empty) so no variant is left unanswered by a variant-major cut; remove_stale_outputs deletes an earlier run's per_question_*.jsonl, calls.jsonl, summary.json, summary.md and, unless kept, parity.json / ingest_gaps.json, never the retain file; counter_id_factory mints zero-padded sequential ids of the library's width; baseline_status words the method text by the parity outcome.
    build_plan 把 --max-answer-requests 均分到各变体（各保留其前若干条；余数弃置；不足每变体一条时计划为空），使变体优先的截断
    不再让末尾变体无答案；remove_stale_outputs 删除先前 run 的 per_question_*.jsonl、calls.jsonl、summary.json、summary.md 以及
    （未保留时的）parity.json / ingest_gaps.json，从不删 retain 文件；counter_id_factory 铸造与库等宽的零填充顺序 id；
    baseline_status 按一致性结果措辞方法文本。

    Input / 输入：
        Three synthetic variants x four records; a tmp dir with stale files.
        三个合成变体 x 四条记录；含过期文件的临时目录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes and deletes files under tmp_path.
        在 tmp_path 下写入并删除文件。
    """
    records = {v: [{"id": str(i), "context": "x"} for i in range(4)] for v in ("rows", "card_1200", "session_prefix")}
    plan, per_variant = reader_ab.build_plan(records, 8)
    assert per_variant == {"rows": 2, "card_1200": 2, "session_prefix": 2} and len(plan) == 6
    assert [(v, r["id"]) for v, r in plan] == [("rows", "0"), ("rows", "1"), ("card_1200", "0"), ("card_1200", "1"), ("session_prefix", "0"), ("session_prefix", "1")]
    assert reader_ab.build_plan(records, 2000)[1] == {v: 4 for v in records} and len(reader_ab.build_plan(records, 2000)[0]) == 12
    assert reader_ab.build_plan(records, 2) == ([], {v: 0 for v in records}) and reader_ab.build_plan({}, 10) == ([], {})
    for name in ("per_question_rows.jsonl", "per_question_card_800.jsonl", "parity.json", "ingest_gaps.json", "summary.json", "summary.md",
                 "calls.jsonl", "retain_lines.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    removed = reader_ab.remove_stale_outputs(tmp_path, keep_parity=True, keep_gaps=False)
    assert sorted(removed) == ["calls.jsonl", "ingest_gaps.json", "per_question_card_800.jsonl", "per_question_rows.jsonl", "summary.json", "summary.md"]
    assert (tmp_path / "parity.json").exists() and (tmp_path / "retain_lines.jsonl").exists() and not (tmp_path / "summary.json").exists()
    assert reader_ab.remove_stale_outputs(tmp_path, keep_parity=False, keep_gaps=False) == ["parity.json"]
    assert reader_ab.REWRITTEN_OUTPUTS == ("calls.jsonl", "summary.json", "summary.md")
    make = reader_ab.counter_id_factory()
    assert (make("mem"), make("ent")) == ("mem_000000000001", "ent_000000000002") and len(make("x")) == len("x_") + 12
    ok = {"rows": {"compared": 3, "mismatches": 0}, "card_1200": {"compared": 3, "mismatches": 0}}
    assert "parity ok" in reader_ab.baseline_status("rows", ok) and "not verifiable" in reader_ab.baseline_status("card_1200", ok)
    bad = {"card_1200": {"compared": 204, "mismatches": 56, "questions_mismatched": 56}}
    assert "PARITY FAILED on 56/204" in reader_ab.baseline_status("card_1200", bad) and "no --parity-check" in reader_ab.baseline_status("rows", None)


def test_parity_aggregates_mismatches_per_field_and_next_action_diagnostic(tmp_path):
    """parity_check reports, per field, the mismatch count, the histogram of replayed - recorded, the first / last timeline position and whether the positions are contiguous, and for card_1200 whether each char delta fits inside the replayed card's '## Next Action' section; parity_caveats carries the aggregate; next_action_section_chars measures that section.
    parity_check 按字段报告不一致计数、replayed - recorded 直方图、首 / 末时间线位置与位置是否连续，并对 card_1200 报告每个字符差
    是否落在回放卡片 '## Next Action' 节内；parity_caveats 携带该汇总；next_action_section_chars 度量该节长度。

    Input / 输入：
        A hand-filled hook (four questions) and recorded jsonl files where the last two cards are 66 chars longer.
        手工填好的钩子（四道题）与后两张卡片长 66 字符的记录 jsonl。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the oracle files under tmp_path.
        在 tmp_path 下写 oracle 文件。
    """
    na_text = "## Next Action\nDo not re-apply 'a'; investigate 'b' while keeping constraints: c1; c2; c3.\n"
    card = "# Work State Card\n## Valid Facts\n- x\n" + na_text + "## Evidence\n- e\n"
    na = reader_ab.next_action_section_chars(card)
    assert na == len(na_text) and card[card.find("## Next Action"):][:na] == na_text
    assert reader_ab.next_action_section_chars("no such section") == 0 and reader_ab.next_action_section_chars("## Next Action\ntail") == len("## Next Action\ntail")
    hook = reader_ab.ReaderContexts(variants=("rows", "card_1200"))
    ids = ("10", "11", "12", "13")
    for i, qid in enumerate(ids):
        hook.positions[qid] = i
        hook.baseline[qid] = {"returned_batches": ["b0000"], "top_scores": [0.5], "returned_count": 1, "context_chars": 10}
        hook.by_question[qid] = {"card_1200": {"context": card, "context_chars": len(card), "returned_batches": ["b0000"],
                                               "card": {"used_tokens": 50, "items": 1, "mode": "full"}}}  # fmt: skip
    rows_oracle = [{"id": qid, "position": i, **hook.baseline[qid]} for i, qid in enumerate(ids)]
    card_oracle = [{"id": qid, "position": i, "returned_batches": ["b0000"], "context_chars": len(card) + (66 if i >= 2 else 0),
                    "card": {"used_tokens": 50, "items": 1, "mode": "full"}} for i, qid in enumerate(ids)]  # fmt: skip
    (tmp_path / "questions_elite_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows_oracle) + "\n", encoding="utf-8")
    (tmp_path / "questions_elite_card.jsonl").write_text("\n".join(json.dumps(r) for r in card_oracle) + "\n", encoding="utf-8")
    parity = reader_ab.parity_check(tmp_path, hook, atoms=None, complete=False)
    assert not parity["ok"] and parity["rows"]["mismatches"] == 0 and parity["rows"]["by_field"] == {}
    c = parity["card_1200"]
    assert c["mismatches"] == 2 and c["questions_mismatched"] == 2 and c["first"][0]["position"] == 2
    assert c["by_field"] == {"context_chars": {"mismatches": 2, "delta_histogram": {"-66": 2}, "first_position": 2, "last_position": 3, "contiguous": True}}
    assert c["next_action"] == {"questions": 2, "delta_within_next_action": 2, "delta_exceeds_next_action": 0,
                                "first": [{"id": "12", "delta_chars": 66, "next_action_chars": na}, {"id": "13", "delta_chars": 66, "next_action_chars": na}]}  # fmt: skip
    caveat = reader_ab.parity_caveats(parity)
    assert len(caveat) == 1 and "context_chars: 2 (positions 2..3, contiguous; deltas {'-66': 2})" in caveat[0] and "Next Action" in caveat[0]
    # A non-contiguous, mixed-delta divergence is reported as such.
    # 不连续、差值不一的分歧按原样报告。
    card_oracle[0]["context_chars"] += 5
    (tmp_path / "questions_elite_card.jsonl").write_text("\n".join(json.dumps(r) for r in card_oracle) + "\n", encoding="utf-8")
    c = reader_ab.parity_check(tmp_path, hook, atoms=None, complete=False)["card_1200"]
    assert c["by_field"]["context_chars"] == {"mismatches": 3, "delta_histogram": {"-5": 1, "-66": 2}, "first_position": 0, "last_position": 3, "contiguous": False}


def test_session_memo_uses_last_stored_window_and_falls_back(tmp_path):
    """The session memo is keyed on the last window of the last leaf that produced an ADD / UPDATE: when the step's final window was never stored the query is that earlier window; an id outside the step's windows falls back to the step's last window and is counted; two replays with counter_id_factory mint the same atom ids and render identical cards.
    会话记忆以最后一个产生 ADD / UPDATE 的叶子的最后一个窗口为键：该步末窗口未入库时查询取更早的那个窗口；id 不在该步窗口内时
    退回该步最后一个窗口并计数；两次使用 counter_id_factory 的回放铸出相同的原子 id 并渲染相同的卡片。

    Input / 输入：
        The inline timeline; a hook driven by hand for the memo rule; two run_arm replays with deterministic ids.
        内联时间线；手工驱动记忆规则的钩子；两次使用确定性 id 的 run_arm 回放。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes two SQLite stores under tmp_path.
        在 tmp_path 下写两个 SQLite 库。
    """
    steps, batches, recorded = _timeline()

    class FakeHMS:
        """compile stub returning a card naming its query.
        返回点名查询的卡片的 compile 桩。
        """

        def compile(self, query, **kw):
            class IR:
                used_tokens = 1
                mode = "full"

                def to_text(self_inner):
                    return f"# Card\nquery={query}\nas_of={kw['as_of']}"

                def items(self_inner):
                    return []

            return IR()

    hook = reader_ab.ReaderContexts(variants=("session_1200",))
    first = hook._session(FakeHMS(), steps[0], set(), changed=1, last_window=WINDOWS[0].window_id)  # type: ignore[arg-type]
    assert first["session"]["query_window"] == WINDOWS[0].window_id and WINDOWS[0].text in first["context"] and first["session"]["as_of"] == WINDOWS[0].start_iso
    assert hook.session_query_fallback == 0 and hook.session_compiles == 1
    fallback = hook._session(FakeHMS(), steps[2], set(), changed=1, last_window="not-a-window-of-this-step")  # type: ignore[arg-type]
    assert fallback["session"]["query_window"] == WINDOWS[2].window_id and hook.session_query_fallback == 1 and hook.session_compiles == 2
    none_only = hook._session(FakeHMS(), steps[2], set(), changed=0, last_window=None)  # type: ignore[arg-type]
    assert none_only["session"]["reused"] is True and hook.session_ingest_skipped == 1 and hook.session_compiles == 2
    cards = []
    for k in range(2):
        h = reader_ab.ReaderContexts(variants=("card_1200",))
        replay_ab.run_arm("lines", steps, batches, recorded, work_root=tmp_path / f"w{k}", out=tmp_path / f"w{k}", speaker_from_subject=True,
                          entities_from_mentions=True, contract="lines", context_hook=h, embedder=HASH, id_factory=reader_ab.counter_id_factory())  # fmt: skip
        cards.append([h.by_question[q]["card_1200"]["context"] for q in ("1", "2", "3")])
    axis = [[json.loads(line)["id"] for line in (tmp_path / f"w{k}" / "entity_axis_lines.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for k in range(2)]  # fmt: skip
    assert cards[0] == cards[1] and axis[0] == axis[1] and len(axis[0]) == 3 and all(len(i.split("_")[-1]) == 12 and i.split("_")[-1].isdigit() for i in axis[0])


def test_cli_counter_ids_by_default_random_opt_out_and_baseline_note():
    """Counter atom ids are the CLI default: an empty command line and --deterministic-ids (a compatibility no-op) both select counter_id_factory, --random-ids selects None (the library's secrets.token_hex ids); the help text and CARD_BASELINE_NOTE say card_* / session_* are a re-derived baseline; markdown() prints the atom-id mode and the note.
    计数器原子 id 是命令行默认：空命令行与 --deterministic-ids（兼容用空操作）都选 counter_id_factory，--random-ids 选 None
    （库的 secrets.token_hex id）；帮助文本与 CARD_BASELINE_NOTE 说明 card_* / session_* 是重新导出的基线；markdown() 打印
    原子 id 模式与该说明。

    Input / 输入：
        build_parser() and three argument lists; a minimal summary dict.
        build_parser() 与三组参数；一个最小的汇总字典。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    parser = reader_ab.build_parser()
    for argv in ([], ["--deterministic-ids"]):
        args = parser.parse_args(argv)
        mode, factory = reader_ab.select_id_factory(args)
        assert mode == "deterministic" and factory is not None and factory("mem") == "mem_000000000001" and factory("ent") == "ent_000000000002"
    args = parser.parse_args(["--random-ids"])
    assert reader_ab.select_id_factory(args) == ("random", None) and args.random_ids is True and args.deterministic_ids is False
    assert parser.parse_args([]).random_ids is False
    helps = {opt: parser._option_string_actions[opt].help for opt in ("--deterministic-ids", "--random-ids", "--workers")}
    assert "no-op" in helps["--deterministic-ids"] and "default" in helps["--deterministic-ids"]
    assert "secrets.token_hex" in helps["--random-ids"] and "re-derived baseline" in helps["--random-ids"]
    assert "in flight" in helps["--workers"] and "interrupt" in helps["--workers"]
    note = reader_ab.CARD_BASELINE_NOTE
    assert note.startswith("card_* / session_* contexts are a re-derived baseline")
    for phrase in ("random atom ids", "tie-breaks", "cannot be byte-identical", "--random-ids", "fastembed 0.8.0 + onnxruntime 1.29.0"):
        assert phrase in note
    md = reader_ab.markdown({
        "run_dir": "r", "started_utc": "t", "questions": 3, "steps_run": 3, "batches": 1, "replies_replayed": 1, "replies_missing": 0, "atoms": 1,
        "replay": {"lines_calls": 1, "lines_misses": 0}, "session_memo": {"compiles": 1, "reuses": 2, "without_ingest": 0, "ingest_skipped": 0, "rows_baseline_mismatch": 0},
        "config": {"package_versions": {"fastembed": "0.8.0"}, "atom_ids": "deterministic"}, "variants_run": [], "variants": {},
        "answer_pass": {}, "parity": None, "method": "m. " + note, "caveats": [],
    })  # fmt: skip
    assert "atom ids: deterministic (card_* / session_* are re-derived baselines, see Method)" in md and note in md


def test_main_offline_no_answer_empties_ledger_and_mints_counter_ids_by_default(tmp_path, monkeypatch):
    """main() --no-answer over the inline fixture: an earlier run's calls.jsonl / summary / parity / ingest_gaps / per_question files are gone, calls.jsonl is rewritten empty, summary.json records atom_ids='deterministic' and carries CARD_BASELINE_NOTE in its method text; a second run (with the no-op --deterministic-ids) renders byte-identical card_1200 texts and the same 12-digit entity ids; --random-ids mints different ids and records atom_ids='random'; both flags together exit 2 before writing anything.
    对内联夹具运行 main() --no-answer：先前 run 的 calls.jsonl / 汇总 / 一致性 / 缺口 / 逐题文件都被清除，calls.jsonl 重写为空，
    summary.json 记录 atom_ids='deterministic' 且方法文本含 CARD_BASELINE_NOTE；第二次运行（带空操作 --deterministic-ids）渲染
    逐字节相同的 card_1200 文本与相同的 12 位实体 id；--random-ids 铸出不同 id 并记录 atom_ids='random'；两个标志同时给出时
    在写任何内容前退出码 2。

    Input / 输入：
        A fake run dir (both recorded replies), loaders monkeypatched to the inline fixture, four --out dirs.
        伪造的 run 目录（两条记录回复）、打补丁指向内联夹具的加载器、四个 --out 目录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes under tmp_path only.
        只写 tmp_path。
    """
    _offline_main(monkeypatch)
    run = _fake_run(tmp_path, REPLIES)
    outs = [tmp_path / f"out{k}" for k in range(3)]
    _seed_stale(outs[0])
    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--out", str(outs[0]))) == 0
    assert (outs[0] / "calls.jsonl").read_text(encoding="utf-8") == ""
    for name in ("parity.json", "ingest_gaps.json", "work"):
        assert not (outs[0] / name).exists()
    expected = {"calls.jsonl", "summary.json", "summary.md", "per_question_lines.jsonl", "retain_lines.jsonl", "entity_axis_lines.jsonl"}
    expected |= {f"per_question_{v}.jsonl" for v in reader_ab.VARIANTS}
    assert {p.name for p in outs[0].iterdir()} == expected
    summary = json.loads((outs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["answer_pass"]["reason"] == "--no-answer" and summary["config"]["atom_ids"] == "deterministic" and summary["api_cost_usd"] == 0.0
    assert summary["method"].endswith(reader_ab.CARD_BASELINE_NOTE) and "this replay's are deterministic (counter_id_factory)" in summary["caveats"][0]
    assert summary["ingest_gaps"]["fatal"] is False and summary["variants"]["rows"]["n"] == 3
    md = (outs[0] / "summary.md").read_text(encoding="utf-8")
    assert "atom ids: deterministic" in md and reader_ab.CARD_BASELINE_NOTE in md and "# stale" not in md
    assert len((outs[0] / "per_question_rows.jsonl").read_text(encoding="utf-8").splitlines()) == 3

    def cards(out: Path) -> list[str]:
        return [json.loads(line)["context"] for line in (out / "per_question_card_1200.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def axis(out: Path) -> list[str]:
        return [json.loads(line)["id"] for line in (out / "entity_axis_lines.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--deterministic-ids", "--out", str(outs[1]))) == 0
    assert cards(outs[0]) == cards(outs[1]) and axis(outs[0]) == axis(outs[1]) and len(axis(outs[0])) == 3
    assert all(len(i.split("_")[-1]) == 12 and i.split("_")[-1].isdigit() for i in axis(outs[0]))
    assert all(c.startswith("# Work State Card") for c in cards(outs[0]))
    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--random-ids", "--out", str(outs[2]))) == 0
    assert axis(outs[2]) != axis(outs[0]) and len(axis(outs[2])) == 3
    random_summary = json.loads((outs[2] / "summary.json").read_text(encoding="utf-8"))
    assert random_summary["config"]["atom_ids"] == "random" and "this replay's are random (secrets.token_hex, --random-ids)" in random_summary["caveats"][0]
    never = tmp_path / "never"
    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--deterministic-ids", "--random-ids", "--out", str(never))) == 2
    assert not never.exists()


def test_main_offline_exit_4_leaves_only_gap_outputs(tmp_path, monkeypatch):
    """A run dir without batch b0001's reply makes its leaf unresolved: main() exits 4 after removing every earlier output — summary.md, calls.jsonl, parity.json (kept for --parity-check, which never runs here) and per_question_*.jsonl — and leaves ingest_gaps.json plus a minimal summary.json (stopped=True, exit_code=4, the gaps, no variants, zero cost) beside the retain / entity_axis files; --tolerate-ingest-gaps continues to a full summary and the empty ledger.
    缺少 b0001 回复的 run 目录使其叶子未解析：main() 删除全部先前输出——summary.md、calls.jsonl、parity.json（原为 --parity-check
    保留，此处不会运行）与 per_question_*.jsonl——后退出码 4，只在 retain / entity_axis 文件旁留下 ingest_gaps.json 与最小的
    summary.json（stopped=True、exit_code=4、缺口、无变体、零费用）；--tolerate-ingest-gaps 则继续产出完整汇总与空台账。

    Input / 输入：
        A fake run dir with only b0000's reply; a seeded stale --out dir.
        只有 b0000 回复的伪造 run 目录；预置了过期文件的 --out 目录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes under tmp_path only.
        只写 tmp_path。
    """
    _offline_main(monkeypatch)
    run = _fake_run(tmp_path, {"b0000": REPLIES["b0000"]})
    out = tmp_path / "out"
    _seed_stale(out)
    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--parity-check", "--out", str(out))) == 4
    gaps = json.loads((out / "ingest_gaps.json").read_text(encoding="utf-8"))
    assert gaps["fatal"] is True and gaps["missing"] == [{"batch_id": "b0001", "split": None}] and gaps["batch_ids"] == ["b0001"]
    assert gaps["lines_misses"] == 0 and gaps["api_error"] == []
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped"] is True and summary["exit_code"] == 4 and summary["ingest_gaps"] == gaps and summary["variants"] == {}
    assert summary["api_cost_usd"] == 0.0 and summary["answer_pass"]["requests"] == 0 and summary["config"]["atom_ids"] == "deterministic"
    assert summary["variants_run"] == list(reader_ab.VARIANTS) and summary["parity"] is None and "stale" not in summary
    assert sorted(p.name for p in out.iterdir()) == ["entity_axis_lines.jsonl", "ingest_gaps.json", "retain_lines.jsonl", "summary.json"]
    tolerated = tmp_path / "tolerated"
    _seed_stale(tolerated)
    assert reader_ab.main(_main_args(run, tmp_path, "--no-answer", "--tolerate-ingest-gaps", "--out", str(tolerated))) == 0
    full = json.loads((tolerated / "summary.json").read_text(encoding="utf-8"))
    assert full["ingest_gaps"]["fatal"] is True and "exit_code" not in full and full["variants"]["card_1200"]["n"] == 3
    assert (tolerated / "calls.jsonl").read_text(encoding="utf-8") == "" and not (tolerated / "ingest_gaps.json").exists()


def test_answer_pass_workers_settles_finished_futures_before_reraising(monkeypatch):
    """With workers > 1 an exception surfacing from one in-flight request (a KeyboardInterrupt here) no longer discards the chunk: the other workers' finished answers are written into their records before it is re-raised, so the records carrying `pred` are exactly the ledger's calls; without the exception every planned request is answered once at workers 2 and 1 alike.
    workers > 1 时，某个飞行中请求冒出的异常（此处为 KeyboardInterrupt）不再丢掉整块：其他工作线程已完成的答案先写入记录再重新
    抛出，因此带 `pred` 的记录恰好等于台账中的调用；无异常时 workers 为 2 与 1 都把每个计划请求恰好回答一次。

    Input / 输入：
        Fake DeepSeekClient / load_api_key / ask (no network, no .env); two variants x three questions.
        替身 DeepSeekClient / load_api_key / ask（无网络、不读 .env）；两个变体 x 三道题。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None outside the fakes.
        替身之外无。
    """

    class FakeClient:
        """Holds the meter the fake ask records into.
        持有替身 ask 记账所用的台账。
        """

        def __init__(self, meter: Meter) -> None:
            self.meter = meter

    boom: set[tuple[str, str]] = {("reader_rows", "2")}

    def fake_ask(client: FakeClient, question: Question, prompt: str, *, arm: str) -> AnswerResult:
        if (arm, question.id) in boom:
            time.sleep(0.05)
            raise KeyboardInterrupt
        call = ApiCall(arm=arm, stage="answer", question_id=question.id, attempt=0, started_utc="t", latency_ms=1.0, status="ok",
                       prompt_tokens=10, completion_tokens=1, cache_hit_tokens=0, cache_miss_tokens=10, rate_tier="offpeak")  # fmt: skip
        client.meter.add(call)
        return AnswerResult(letter="A", raw="A", status="ok", prompt_chars=len(prompt), calls=[call], model_ms=1.0, wall_ms=1.0)

    monkeypatch.setattr(reader_ab, "load_api_key", lambda path: None)
    monkeypatch.setattr(reader_ab, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(reader_ab, "ask", fake_ask)
    questions = {q.id: q for q in QUESTIONS}

    def fresh() -> dict[str, list[dict]]:
        return {v: [{"id": q.id, "context": "- x", "gold": "A"} for q in QUESTIONS] for v in ("rows", "card_1200")}

    records = fresh()
    meter = reader_ab.LockedMeter()
    with pytest.raises(KeyboardInterrupt):
        reader_ab.answer_pass(records, questions, meter, max_requests=100, cost_cap=1.0, workers=3)
    answered = sorted((v, r["id"]) for v, recs in records.items() for r in recs if "pred" in r)
    assert answered == [("rows", "1"), ("rows", "3")]
    assert sorted((c.arm, c.question_id) for c in meter.calls) == [("reader_rows", "1"), ("reader_rows", "3")]
    assert all(r["correct"] for r in records["rows"] if "pred" in r)
    boom.clear()
    for workers in (2, 1):
        records = fresh()
        meter = reader_ab.LockedMeter()
        status = reader_ab.answer_pass(records, questions, meter, max_requests=100, cost_cap=1.0, workers=workers)
        assert status["requests"] == 6 and status["stopped"] is False and status["workers"] == workers and len(meter.calls) == 6
        assert all("pred" in r and r["answer_attempts"] == 1 for recs in records.values() for r in recs)


def test_card_prefix_and_hybrid_prefix_reuse_texts_and_answer_memories_first(tmp_path):
    """card_prefix carries card_1200's text, measurements and hit for every question (one compile shared when both run) and its prompt is 'Memories:\\n<card>\\n\\n<question prompt>'; hybrid_prefix's context is session_1200's memoized text, a blank line, 'Recent relevant memories:' and rows_ts's text, so its prompt has exactly the two-block memories-first shape, it reuses the session memo (no extra compile: 2 compiles / 1 reuse as before), carries session_1200's `session` / `card` fields and rows_ts's `rows` facts, and hit = session hit OR rows hit with both components recorded; a hook running only the two probes builds the same texts; hybrid_context renders a blank part as '(none)'; the variant summaries expose the memo column and the component hits.
    card_prefix 对每道题携带 card_1200 的文本、度量与命中（两者同时运行时共享一次 compile），提示词为 'Memories:\\n<卡片>\\n\\n<题目提示词>'；
    hybrid_prefix 的上下文是 session_1200 的记忆化文本、空行、'Recent relevant memories:' 与 rows_ts 的文本，故其提示词恰为两段式记忆在前
    形状，复用会话记忆（不多 compile：仍为 2 次编译 / 1 次复用），携带 session_1200 的 `session` / `card` 字段与 rows_ts 的 `rows` 事实，
    命中 = 会话命中 OR 行式命中且两部分都记录；只运行两个探针的钩子构造相同文本；hybrid_context 把空白部分渲染为 '(none)'；变体汇总
    暴露记忆列与各部分命中。

    Input / 输入：
        The inline timeline and recorded replies; two run_arm replays with counter ids under tmp_path.
        内联时间线与记录回复；tmp_path 下两次使用计数器 id 的 run_arm 回放。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes two SQLite stores under tmp_path.
        在 tmp_path 下写两个 SQLite 库。
    """
    steps, batches, recorded = _timeline()
    hook = reader_ab.ReaderContexts()
    res = replay_ab.run_arm("lines", steps, batches, recorded, work_root=tmp_path / "w0", out=tmp_path / "w0", speaker_from_subject=True,
                            entities_from_mentions=True, contract="lines", context_hook=hook, embedder=HASH, id_factory=reader_ab.counter_id_factory())  # fmt: skip
    assert "card_prefix" in reader_ab.VARIANTS and "hybrid_prefix" in reader_ab.VARIANTS and len(reader_ab.VARIANTS) == 11
    assert reader_ab.SESSION_MEMO_VARIANTS == ("session_1200", "session_prefix", "hybrid_prefix")
    assert (hook.session_compiles, hook.session_reuses, hook.session_without_ingest) == (2, 1, 0)
    by_id = {q.id: q for q in QUESTIONS}
    header = reader_ab.HYBRID_ROWS_HEADER
    for rec in res["records"]:
        q = by_id[rec["id"]]
        ctx = hook.by_question[rec["id"]]
        cp, c12 = ctx["card_prefix"], ctx["card_1200"]
        assert cp["context"] == c12["context"] and cp["card"] == c12["card"] and cp["context_chars"] == c12["context_chars"]
        assert cp["returned_batches"] == c12["returned_batches"] and cp["hit_batch"] == c12["hit_batch"] and cp["est_tokens"] == c12["est_tokens"]
        prompt = reader_ab.build_prompt(q, cp["context"], reader_ab.PROMPT_ORDER["card_prefix"])
        assert prompt == "Memories:\n" + c12["context"] + "\n\n" + build_question_prompt(q.raw)
        assert prompt.startswith("Memories:\n# Work State Card") and prompt.endswith("D. nothing")
        assert reader_ab.build_prompt(q, c12["context"], reader_ab.PROMPT_ORDER["card_1200"]).startswith(build_question_prompt(q.raw))
        hy, s12, ts = ctx["hybrid_prefix"], ctx["session_1200"], ctx["rows_ts"]
        assert hy["context"] == s12["context"] + "\n\n" + header + "\n" + ts["context"] and header == "Recent relevant memories:"
        hp = reader_ab.build_prompt(q, hy["context"], reader_ab.PROMPT_ORDER["hybrid_prefix"])
        assert hp == "Memories:\n" + s12["context"] + "\n\n" + header + "\n" + ts["context"] + "\n\n" + build_question_prompt(q.raw)
        assert hp.startswith("Memories:\n# Work State Card") and hp.count(header) == 1
        assert hy["session"] == s12["session"] and hy["card"] == s12["card"] and hy["rows"] == ts["rows"]
        assert hy["context_chars"] == len(hy["context"]) and hy["est_tokens"] > 0 and hy["hybrid"]["rows_header"] == header
        assert hy["hybrid"]["session_chars"] == len(s12["context"]) and hy["hybrid"]["rows_chars"] == len(ts["context"])
        assert hy["hybrid"]["session_hit"] == s12["hit_batch"] and hy["hybrid"]["rows_hit"] == ts["hit_batch"]
        assert hy["hybrid"]["rows_hit_top20"] == ts["hit_batch_top20"] and hy["hybrid"]["rows_batches"] == ts["returned_batches"]
        assert hy["hybrid"]["session_batches"] == s12["returned_batches"] and hy["hybrid"]["rows_kept"] == ts["returned_count"]
        assert hy["hit_batch"] == (s12["hit_batch"] or ts["hit_batch"]) == any(b in set(rec["gold_batches"]) for b in hy["returned_batches"])
        assert hy["returned_batches"][: len(s12["returned_batches"])] == s12["returned_batches"]
        assert set(hy["returned_batches"]) == set(s12["returned_batches"]) | set(ts["returned_batches"])
        assert hy["returned_count"] == s12["returned_count"] + ts["returned_count"]
    s1, s2 = hook.by_question["1"]["hybrid_prefix"], hook.by_question["2"]["hybrid_prefix"]
    assert (s1["session"]["reused"], s2["session"]["reused"]) == (False, True) and s1["hybrid"]["session_chars"] == s2["hybrid"]["session_chars"]
    # The two probes alone: one compile per step / question as before, and the same texts as the full hook.
    # 只跑两个探针：每步 / 每题仍只 compile 一次，文本与完整钩子相同。
    alone = reader_ab.ReaderContexts(variants=("card_prefix", "hybrid_prefix"))
    replay_ab.run_arm("lines", steps, batches, recorded, work_root=tmp_path / "w1", out=tmp_path / "w1", speaker_from_subject=True,
                      entities_from_mentions=True, contract="lines", context_hook=alone, embedder=HASH, id_factory=reader_ab.counter_id_factory())  # fmt: skip
    assert (alone.session_compiles, alone.session_reuses) == (2, 1) and alone.rows_baseline_mismatch == 0
    for qid in ("1", "2", "3"):
        assert set(alone.by_question[qid]) == {"card_prefix", "hybrid_prefix"}
        for v in ("card_prefix", "hybrid_prefix"):
            assert alone.by_question[qid][v]["context"] == hook.by_question[qid][v]["context"]
        assert alone.by_question[qid]["hybrid_prefix"]["session"] == hook.by_question[qid]["hybrid_prefix"]["session"]
    assert reader_ab.hybrid_context("", "") == "(none)\n\n" + header + "\n(none)"
    assert reader_ab.hybrid_context("# Card", " ") == "# Card\n\n" + header + "\n(none)"
    assert reader_ab.hybrid_context("  ", "- [2024-01-01 11:00] x") == "(none)\n\n" + header + "\n- [2024-01-01 11:00] x"
    # Records, files' labels and the unanswered summaries: hybrid_prefix has the memo column and component hits, card_prefix neither.
    # 记录、台账标签与未答题汇总：hybrid_prefix 有记忆列与各部分命中，card_prefix 两者皆无。
    records = reader_ab.variant_records(reader_ab.VARIANTS, res["records"], hook)
    assert {v: len(r) for v, r in records.items()} == {v: 3 for v in reader_ab.VARIANTS}
    assert records["card_prefix"][0]["prompt_order"] == records["hybrid_prefix"][0]["prompt_order"] == "memories_first"
    assert reader_ab.ANSWER_LABEL["card_prefix"] == "reader_card_prefix" and reader_ab.ANSWER_LABEL["hybrid_prefix"] == "reader_hybrid_prefix"
    hs = reader_ab.variant_summary("hybrid_prefix", records["hybrid_prefix"], Meter())
    assert hs["session"]["recomputed"] == 2 and hs["session"]["reused"] == 1 and hs["prompt_order"] == "memories_first"
    assert hs["hybrid"]["hit_session"] == reader_ab.variant_summary("session_1200", records["session_1200"], Meter())["hit_batch"]
    assert hs["hybrid"]["hit_rows"] == reader_ab.variant_summary("rows_ts", records["rows_ts"], Meter())["hit_batch"]
    assert hs["hit_batch"] >= max(hs["hybrid"]["hit_session"], hs["hybrid"]["hit_rows"]) and "OR" in hs["hit_note"]
    assert hs["rows"]["rows_in_total"] == reader_ab.variant_summary("rows_ts", records["rows_ts"], Meter())["rows"]["rows_in_total"]
    assert hs["card"]["budget_tokens"] == 1200 and hs["answered_n"] == 0 and hs["cny_per_204"] == {"offpeak": None, "peak": None}
    cs = reader_ab.variant_summary("card_prefix", records["card_prefix"], Meter())
    assert cs["session"] is None and cs["hybrid"] is None and cs["card"]["budget_tokens"] == 1200
    assert cs["hit_batch"] == reader_ab.variant_summary("card_1200", records["card_1200"], Meter())["hit_batch"]
    assert cs["context_chars_mean"] == reader_ab.variant_summary("card_1200", records["card_1200"], Meter())["context_chars_mean"]


def test_hybrid_prefix_summary_reports_component_hits_and_memo_column():
    """A priced hybrid_prefix variant: the CNY / cache-hit accounting reads the reader_hybrid_prefix ledger rows, the session memo splits the cache-hit share by first-at-step vs reused-card questions exactly as for session_prefix, the `hybrid` block gives the component hit rates and part sizes, and markdown() prints the memo column and the component hits next to the combined hit.
    已计价的 hybrid_prefix 变体：人民币 / 缓存命中记账读取 reader_hybrid_prefix 的台账行，会话记忆像 session_prefix 一样按"该步首题 /
    复用卡片题"拆分缓存命中占比，`hybrid` 块给出各部分命中率与大小，markdown() 在合并命中旁打印记忆列与各部分命中。

    Input / 输入：
        Two hybrid_prefix records answered through apply_answer with hand-written ApiCalls.
        两条经 apply_answer 以手写 ApiCall 答题的 hybrid_prefix 记录。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        None.
        无。
    """
    meter = Meter()
    calls = [
        ApiCall(arm="reader_hybrid_prefix", stage="answer", question_id="1", attempt=0, started_utc="t", latency_ms=1.0, status="ok",
                prompt_tokens=1000, completion_tokens=1, cache_hit_tokens=0, cache_miss_tokens=1000, rate_tier="offpeak"),
        ApiCall(arm="reader_hybrid_prefix", stage="answer", question_id="2", attempt=0, started_utc="t", latency_ms=1.0, status="ok",
                prompt_tokens=1000, completion_tokens=1, cache_hit_tokens=800, cache_miss_tokens=200, rate_tier="offpeak"),
    ]  # fmt: skip
    for c in calls:
        meter.add(c)
    body = reader_ab.hybrid_context("# Card", "- x")
    base = {"context": body, "context_chars": len(body), "est_tokens": 10, "hit_batch": True, "excluded_future": False, "gold": "A",
            "card": {"chars": 6, "used_tokens": 5, "items": 1, "mode": "full", "compile_ms": 1.0, "budget_tokens": 1200}}  # fmt: skip
    recs = [
        dict(base, id="1", session={"key": "w1", "reused": False}, hybrid={"session_chars": 6, "rows_chars": 3, "session_hit": True, "rows_hit": False}),
        dict(base, id="2", session={"key": "w1", "reused": True}, hybrid={"session_chars": 6, "rows_chars": 3, "session_hit": False, "rows_hit": True}),
    ]  # fmt: skip
    for rec, call in zip(recs, calls):
        reader_ab.apply_answer(rec, AnswerResult(letter="A", raw="A", status="ok", prompt_chars=5, calls=[call], model_ms=1.0, wall_ms=1.0))
    s = reader_ab.variant_summary("hybrid_prefix", recs, meter)
    assert s["answered_n"] == s["priced_n"] == 2 and s["answer_accuracy"] == 1.0 and s["cache_hit_share"] == 0.4
    assert s["session"] == {"reused": 1, "recomputed": 1, "cache_hit_share_first_at_step": 0.0, "cache_hit_share_reused_card": 0.8}
    assert s["hybrid"] == {"hit_session": 0.5, "hit_rows": 0.5, "hit_both": 0.0, "hit_rows_only": 0.5, "session_chars_mean": 6.0,
                           "rows_chars_mean": 3.0, "rows_header": "Recent relevant memories:"}  # fmt: skip
    assert s["cny_total"]["offpeak"] == round(reader_ab.cny_cost(800, 1200, 2, "offpeak"), 6) and s["card"]["budget_tokens"] == 1200
    md = reader_ab.markdown({
        "run_dir": "r", "started_utc": "t", "questions": 2, "steps_run": 2, "batches": 1, "replies_replayed": 1, "replies_missing": 0, "atoms": 1,
        "replay": {"lines_calls": 1, "lines_misses": 0}, "session_memo": {"compiles": 1, "reuses": 1, "without_ingest": 0, "ingest_skipped": 0, "rows_baseline_mismatch": 0},
        "config": {"package_versions": {"fastembed": "0.8.0"}}, "variants_run": ["hybrid_prefix"], "variants": {"hybrid_prefix": s},
        "answer_pass": {}, "parity": None, "method": "m", "caveats": ["c"],
    })  # fmt: skip
    assert (f"| hybrid_prefix | memories_first | 100.0% (2/2) | 100.0% (session 50.0% / rows 50.0%) | - | {len(body)} | 1000.0 | 40.0% | "
            "1 / 1; 0.0% / 80.0% |") in md
    assert "hybrid_prefix = session-card hit OR kept rows_ts hit" in md and "eleven answer contexts" in md
