"""EgoLife-ZH harness pieces, offline: windows.build_windows(text_lang), the translate_zh glossary / requests / validation / projection / dataset writer / self-check / bisecting round trip with a fake client, and the CNY price table parity.
EgoLife-ZH 基准脚本部件的离线测试：windows.build_windows(text_lang)、translate_zh 的术语表 / 请求 / 校验 / 预估 / 数据集写入 / 自检 / 用替身客户端的二分往返，以及人民币价目表一致性。
"""

# Purpose (EN): Exercise W1 / W2 without the API or the real dataset (a two-file synthetic data root under tmp_path).
# 目的（中）：不用 API 与真实数据集（tmp_path 下两文件的合成数据根）验证 W1 / W2。

# Input (EN): Synthetic SRT / QA files; a fake DeepSeek client.
# 输入（中）：合成的 SRT / QA 文件；替身 DeepSeek 客户端。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): Writes only under tmp_path.
# 状态影响（中）：只写 tmp_path。

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_reader_ab import _stub_bench_deps

ROOT = Path(__file__).resolve().parents[1]
EGOLIFE = ROOT / "benchmarks" / "egolife"
if str(EGOLIFE) not in sys.path:
    sys.path.insert(0, str(EGOLIFE))

import translate_zh  # noqa: E402
from loader import QA_FILENAME, TRANSCRIPT_SUBDIR, load_questions, load_transcripts  # noqa: E402
from translate_zh import Translation  # noqa: E402
from windows import build_windows  # noqa: E402

SRT = """1
00:00:00,000 --> 00:00:01,000
Jake: 好然后一个秒表
Jake: Okay, then we need a stopwatch.

2
00:00:01,500 --> 00:00:02,500
Shure: 时间戳吧
Shure: Mark a timestamp, right?

3
00:00:03,000 --> 00:00:04,000
Tasha: I like pizza.

4
00:00:04,500 --> 00:00:05,000
其他人: 只有中文的一句

5
00:00:05,500 --> 00:00:06,000
Alice: 我们走吧
Alice: Let's go, Shure's GF is waiting.
"""
QA = [
    {
        "ID": "6", "query_time": {"date": "DAY1", "time": "11000600"}, "type": "EventRecall", "question": "What did Jake need?",
        "question_chinese": "Jake 需要什么？", "choice_a": "A stopwatch", "choice_a_chinese": "秒表", "choice_b": "Pizza", "choice_b_chinese": "披萨",
        "choice_c": "Me", "choice_c_chinese": "我", "choice_d": "Tasha", "choice_d_chinese": "Tasha", "answer": "A",
        "target_time": {"date": "DAY1", "time": "11000000"},
    },
    {
        "ID": "7", "query_time": {"date": "DAY1", "time": "11000700"}, "type": "RelationMap", "question": "Who likes pizza?",
        "question_chinese": "谁喜欢披萨？", "choice_a": "Alice Katrina", "choice_a_chinese": "Alice Katrina", "choice_b": "Tasha", "choice_b_chinese": "Tasha",
        "choice_c": "Shure", "choice_c_chinese": "Shure", "choice_d": "Jake's friend", "choice_d_chinese": "Jake的朋友", "answer": "B",
        "target_time": {"date": "DAY1", "time": "11000300"},
    },
]  # fmt: skip


def _data_root(tmp_path: Path) -> Path:
    """A two-file synthetic dataset in the data/ layout.
    data/ 布局的两文件合成数据集。
    """
    root = tmp_path / "data"
    day = root / TRANSCRIPT_SUBDIR / "DAY1"
    day.mkdir(parents=True)
    (day / "A1_JAKE_DAY1_11000000.srt").write_text(SRT, encoding="utf-8")
    (root / QA_FILENAME).write_text(json.dumps(QA, ensure_ascii=False), encoding="utf-8")
    return root


def _fake_line(line: str, n: int) -> str:
    """The fake client's translation of one line: the label kept, '中文<n>' padded to a third of the source body.
    替身客户端对一行的译文：保留标签，'中文<n>' 填充到源正文长度的三分之一。
    """
    label = translate_zh.LABEL_RE.match(line)
    body = translate_zh.strip_label(line)
    text = f"中文{n}" + "字" * max(0, len(body) // 3 - 3)
    return f"{label.group(1)}: {text}" if label else text


class _FakeClient:
    """A DeepSeekClient stand-in: truncates any request with more than one item, translates single items line by line with _fake_line.
    DeepSeekClient 的替身：多于一个单元的请求一律截断，单个单元逐行用 _fake_line 翻译。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_calls: list[Any] = []
        self.labels: dict[str, Any] = {}

    @contextmanager
    def label(self, **labels: Any):
        self.labels = labels
        yield

    def create(self, **params: Any) -> Any:
        self.calls.append({"labels": dict(self.labels), **params})
        payload = json.loads(params["messages"][1]["content"])
        items = payload["items"]
        if len(items) > 1:
            self.last_calls = [SimpleNamespace(finish_reason="length")]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{\"items\": ["), finish_reason="length")])
        item = items[0]
        lines = [_fake_line(line, n) for n, line in enumerate(item["lines"])]
        content = json.dumps({"items": [{"window_id": item["window_id"], "lines": lines}]}, ensure_ascii=False)
        self.last_calls = [SimpleNamespace(finish_reason="stop")]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])


def test_windows_text_lang_twins_and_incomplete_dataset(tmp_path: Path):
    """build_windows(text_lang='zh') yields the same ids, clocks and cue ranges as the English build with the text slots swapped over the same English-bearing cues (a Chinese-only cue joins neither), and raises listing the windows whose Chinese text would be empty.
    build_windows(text_lang='zh') 给出与英文构建相同的 id、时钟与 cue 区间，只在同一批带英文的 cue 上交换文本槽位（只有中文的 cue 两边都不计入），中文文本为空的窗口则抛出并列出。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on the window lists.
        对窗口列表的断言。

    State effect / 状态影响：
        Writes the synthetic dataset under tmp_path.
        在 tmp_path 下写合成数据集。
    """
    transcripts = load_transcripts(_data_root(tmp_path))
    en = build_windows(transcripts, max_chars=80)
    zh = build_windows(transcripts, max_chars=80, text_lang="zh")
    assert len(en) == len(zh) == 2
    for a, b in zip(en, zh):
        assert (a.window_id, a.day, a.start_cs, a.end_cs, a.cue_from, a.cue_to) == (b.window_id, b.day, b.start_cs, b.end_cs, b.cue_from, b.cue_to)
        assert b.text_zh == a.text
    assert zh[0].text == "Jake: 好然后一个秒表\nShure: 时间戳吧" and "I like pizza" in en[0].text or "I like pizza" in en[1].text
    assert "只有中文的一句" not in zh[0].text + zh[1].text and "只有中文的一句" in en[0].text_zh + en[1].text_zh
    only_en = {"A1_JAKE_DAY1_11000000": [c for c in transcripts["A1_JAKE_DAY1_11000000"] if not c.text_zh]}
    with pytest.raises(ValueError, match="no Chinese text"):
        build_windows(only_en, text_lang="zh")
    with pytest.raises(ValueError):
        build_windows(transcripts, text_lang="fr")


def test_glossary_names_requests_validation_and_projection(tmp_path: Path):
    """apply_glossary is whole-word and longest-first (a possessive gets the Chinese particle, labels keep their colon, 'Jake' inside 'Jakes' untouched); collect_names counts labels and option tokens minus the stop list; build_requests never spans files and honours size / skip / mode; validate_window rejects the five failure shapes; project_cost scales with characters; the CNY table equals reader_ab's.
    apply_glossary 整词且长者优先（'Jake's' → 杰克的，标签保留冒号，'Jakes' 中的 'Jake' 不动）；collect_names 统计标签与去掉停用词的选项词元；build_requests 绝不跨文件并遵守 size / skip / mode；validate_window 拒绝五种失败形状；project_cost 随字符数缩放；人民币价目表与 reader_ab 一致。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on translate_zh's pure helpers.
        对 translate_zh 纯函数的断言。

    State effect / 状态影响：
        Writes the synthetic dataset under tmp_path.
        在 tmp_path 下写合成数据集。
    """
    g = translate_zh.apply_glossary
    assert g("Jake: 好") == "杰克: 好" and g("Jake's friend and Jakes") == "杰克的 friend and Jakes"
    assert g("Alice Katrina") == "爱丽丝 卡特里娜" and g("Me") == "我" and g("Meituan") == "Meituan" and g("Shure's GF: hi") == "舒尔的 GF: hi"
    assert g("no names here") == "no names here" and g("Nicous", {"Nicous": "尼克斯", "Nic": "尼"}) == "尼克斯"
    root = _data_root(tmp_path)
    transcripts = load_transcripts(root)
    questions = load_questions(root / QA_FILENAME)
    names = translate_zh.collect_names(transcripts, questions)
    assert names["Jake"] == 2 and names["Shure"] == 2 and names["Tasha"] == 3 and names["Alice"] == 2 and names["Katrina"] == 1
    assert "Pizza" not in names and names["Me"] == 1 and "Let" not in names and "Katrina" in names
    windows = build_windows(transcripts, max_chars=80)
    reqs = translate_zh.build_requests(windows, transcripts, size=25)
    assert [r.request_id for r in reqs] == ["r0000"] and [it.window_id for it in reqs[0].items] == [w.window_id for w in windows]
    assert reqs[0].items[0].lines == ["Jake: Okay, then we need a stopwatch.", "Shure: Mark a timestamp, right?"]
    assert [r.request_id for r in translate_zh.build_requests(windows, transcripts, size=1)] == ["r0000", "r0001"]
    assert translate_zh.build_requests(windows, transcripts, skip={windows[0].window_id})[0].items[0].window_id == windows[1].window_id
    native = translate_zh.build_requests(windows, transcripts, mode="native")
    assert [it.lines for r in native for it in r.items] == [["Tasha: I like pizza."]]
    src = ["Jake: Okay, then we need a stopwatch.", "Shure: Mark a timestamp, right?"]
    assert translate_zh.validate_window(src, ["Jake: 好然后一个秒表", "Shure: 时间戳吧"]) is None
    assert translate_zh.validate_window(src, []) == "empty" and translate_zh.validate_window(src, "x") == "empty"
    assert translate_zh.validate_window(src, ["Jake: 好"]).startswith("line_count")
    assert translate_zh.validate_window(src, ["Jake: 好", ""]) == "empty_line"
    assert translate_zh.validate_window(src, ["Jake: 好" * 40, "Shure: 时间戳吧" * 20]).startswith("length_ratio")
    assert translate_zh.validate_window(src, ["Jake: Okay then we need a stopwatch", "Shure: Mark a timestamp right"]).startswith("latin_rate")
    assert translate_zh.latin_letter_rate(["Jake: 好然后"]) == 0.0 and translate_zh.latin_letter_rate([]) == 0.0
    proj = translate_zh.project_cost(reqs)
    assert proj["requests"] == 1 and proj["windows"] == 2 and proj["en_chars"] == sum(len(line) for it in reqs[0].items for line in it.lines)
    assert proj["cny"]["peak"] == pytest.approx(2 * proj["cny"]["offpeak"]) and proj["usd"]["offpeak"] > 0
    assert translate_zh.cny_cost(0, 1_000_000, 1_000_000, "offpeak") == pytest.approx(5.0) and translate_zh.cny_cost(None, 1, 1, "peak") is None
    _stub_bench_deps()
    import reader_ab

    assert translate_zh.PRICES_CNY_PER_M == reader_ab.PRICES_CNY_PER_M
    assert translate_zh.cny_cost(10, 20, 30, "peak") == reader_ab.cny_cost(10, 20, 30, "peak")


def test_translate_round_trip_writes_a_loadable_twin_dataset(tmp_path: Path):
    """A batch request that the fake client truncates is bisected down to single windows and validated; write_dataset then produces data_zh with the English lines verbatim, glossary-applied Chinese lines (translated where validated, native otherwise, none when a cue has neither), the QA file with Chinese question / options and untouched answers; self_check reloads it through the unchanged loader as the exact twin, and the cache is resumable.
    替身客户端截断的批次请求被二分到单窗口并校验；write_dataset 随后产出 data_zh：英文行原样、套用术语表的中文行（通过校验处为译文，否则原生行，两者皆无时不写），QA 文件的题目 / 选项为中文而答案不变；self_check 经未改动的加载器重读为精确孪生，缓存可续跑。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on the written files, the cache rows, the ledger and the self-check.
        对写出的文件、缓存行、台账与自检的断言。

    State effect / 状态影响：
        Writes data/ and data_zh/ under tmp_path.
        在 tmp_path 下写 data/ 与 data_zh/。
    """
    root = _data_root(tmp_path)
    out = tmp_path / "data_zh"
    transcripts = load_transcripts(root)
    questions = load_questions(root / QA_FILENAME)
    windows = build_windows(transcripts, max_chars=80)
    reqs = translate_zh.build_requests(windows, transcripts, size=25)
    client = _FakeClient()
    meter = translate_zh.TranslateMeter()
    cache = out / "translate_cache.jsonl"
    results = translate_zh.translate_all(lambda: client, reqs, cache, meter, cap_cny=6.0, workers=2)
    assert sorted(results) == sorted(w.window_id for w in windows) and all(t.validated for t in results.values())
    assert [c["labels"]["batch_id"] for c in client.calls] == ["r0000", "r0000a", "r0000b"]
    assert client.calls[0]["response_format"] == {"type": "json_object"} and client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["messages"][0]["content"] == translate_zh.TRANSLATE_SYSTEM
    reloaded = translate_zh.load_cache(cache)
    assert {k: v.lines for k, v in reloaded.items()} == {k: v.lines for k, v in results.items()}
    assert translate_zh.build_requests(windows, transcripts, skip={w for w, t in reloaded.items() if t.validated}) == []
    assert meter.cost_summary()["calls"] == 0
    written = translate_zh.write_dataset(out, root, transcripts, windows, reloaded, QA)
    assert written["windows"] == {"translated": 2, "native_fallback": 0, "no_chinese": 0} and written["cues"] == {"translated": 4, "native": 1, "none": 0}
    assert written["glossary_used"]["Jake"] >= 2 and written["glossary_used"]["Tasha"] >= 2 and written["glossary_used"]["Me"] == 0 and written["questions"] == 2
    srt = (out / TRANSCRIPT_SUBDIR / "DAY1" / "A1_JAKE_DAY1_11000000.srt").read_text(encoding="utf-8")
    j0, s1 = _fake_line("Jake: Okay, then we need a stopwatch.", 0), _fake_line("Shure: Mark a timestamp, right?", 1)
    t0, a1 = _fake_line("Tasha: I like pizza.", 0), _fake_line("Alice: Let's go, Shure's GF is waiting.", 1)
    assert f"杰克{j0[4:]}\nJake: Okay, then we need a stopwatch." in srt and f"舒尔{s1[5:]}\nShure: Mark a timestamp, right?" in srt
    assert f"塔莎{t0[5:]}\nTasha: I like pizza." in srt and "其他人: 只有中文的一句" in srt and f"爱丽丝{a1[5:]}\nAlice: Let's go, Shure's GF is waiting." in srt
    zh_qa = json.loads((out / QA_FILENAME).read_text(encoding="utf-8"))
    assert [q["question"] for q in zh_qa] == ["杰克 需要什么？", "谁喜欢披萨？"] and zh_qa[0]["choice_c"] == "我" and zh_qa[1]["choice_a"] == "爱丽丝 卡特里娜"
    assert zh_qa[1]["choice_d"] == "杰克的朋友" and [q["answer"] for q in zh_qa] == ["A", "B"] and zh_qa[0]["target_time"] == QA[0]["target_time"]
    check = translate_zh.self_check(out, windows, questions, max_chars=80)
    assert check["questions_ok"] and check["windows_ok"] and check["zh_windows_ok"] and check["mismatches"] == []
    zh_windows = build_windows(load_transcripts(out), max_chars=80, text_lang="zh")
    assert [w.text for w in zh_windows] == [f"杰克{j0[4:]}\n舒尔{s1[5:]}", f"塔莎{t0[5:]}\n爱丽丝{a1[5:]}"]
    # A window whose translation failed validation falls back to the native lines and is reported.
    # 译文未通过校验的窗口回退到原生行并被报告。
    broken = dict(reloaded)
    broken[windows[0].window_id] = translate_zh.Translation(windows[0].window_id, [], "r0000", 1, False, "truncated")
    written = translate_zh.write_dataset(out, root, transcripts, windows, broken, QA)
    assert written["windows"]["native_fallback"] == 1 and written["windows_native_fallback"] == [windows[0].window_id]
    assert "杰克: 好然后一个秒表\nJake: Okay" in (out / TRANSCRIPT_SUBDIR / "DAY1" / "A1_JAKE_DAY1_11000000.srt").read_text(encoding="utf-8")
    rates = translate_zh.variant_rates(windows, transcripts, reloaded, "translate", translate_zh.NAME_GLOSSARY)
    assert rates["Jake"]["mentions"] == 1 and rates["Jake"]["missing"] == 0 and rates["Katrina"]["mentions"] == 0


def _priced_call(cny_offpeak: float) -> SimpleNamespace:
    """A ledger row whose off-peak price is `cny_offpeak` (output tokens only: ¥4 per 1M).
    空闲价为 `cny_offpeak` 元的台账行（只有输出 token：每 1M ¥4）。
    """
    return SimpleNamespace(cache_hit_tokens=0, cache_miss_tokens=0, completion_tokens=int(cny_offpeak * 250_000), rate_tier="offpeak", attempt=0, status="ok", finish_reason="stop", prompt_tokens=0, cost_usd=0.0)


class _BillingClient(_FakeClient):
    """_FakeClient that also books one priced ledger row per request into a shared meter.
    每次请求还向共用台账记一行计费的 _FakeClient。
    """

    def __init__(self, meter: translate_zh.TranslateMeter, cny_per_call: float) -> None:
        super().__init__()
        self.meter, self.cny_per_call = meter, cny_per_call

    def create(self, **params: Any) -> Any:
        self.meter.calls.append(_priced_call(self.cny_per_call))
        return super().create(**params)


def test_cache_resume_cost_cap_and_spend_ledger(tmp_path: Path):
    """load_cache skips a truncated LAST line with a warning but raises on a malformed line elsewhere; resume_skip_ids treats a failed attempt-1 row as final (skipped) unless retry_failed, and never a 'cap' row; SpendLedger sums earlier invocations and translate_all counts them against the cap — a tripped cap stops the bisect halves too (rows with reason 'cap', no further API call) and raises CostCapExceeded naming the earlier spend; translate_items with should_stop already true issues no request.
    load_cache 对被截断的末行告警跳过、对其它位置的畸形行报错；resume_skip_ids 把 attempt 1 失败的行视为最终状态（跳过），除非 retry_failed，且从不跳过 'cap' 行；SpendLedger 累加既往调用脚本的花费，translate_all 把它计入上限——上限触发后二分的两半也停止（reason 'cap' 的行，不再调用 API）并抛出点明既往花费的 CostCapExceeded；should_stop 已为真时 translate_items 不发请求。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on the cache reader, the resume sets, the ledger file and the cap behaviour.
        对缓存读取、续跑集合、台账文件与上限行为的断言。

    State effect / 状态影响：
        Writes under tmp_path.
        只写 tmp_path。
    """
    cache = tmp_path / "translate_cache.jsonl"
    good = Translation("w1", ["杰克: 好"], "r0000", 0, True).as_dict()
    cache.write_text(json.dumps(good, ensure_ascii=False) + "\n" + '{"window_id": "w2", "lines": ["x' , encoding="utf-8")
    assert list(translate_zh.load_cache(cache)) == ["w1"]
    cache.write_text('{"broken' + "\n" + json.dumps(good, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed cache line 1"):
        translate_zh.load_cache(cache)
    rows = {
        "ok": Translation("ok", ["杰克: 好"], "r0", 0, True),
        "retry_failed": Translation("retry_failed", [], "x0", 1, False, "truncated"),
        "batch_failed": Translation("batch_failed", [], "r0", 0, False, "latin_rate"),
        "capped": Translation("capped", [], "x1", 1, False, "cap"),
    }
    assert translate_zh.resume_skip_ids(rows) == ({"ok"}, {"retry_failed"})
    assert translate_zh.resume_skip_ids(rows, retry_failed=True) == ({"ok"}, set())
    # Spend ledger: an earlier run spent ¥5.5 of a ¥6 cap; this run's first priced call (¥1) trips it.
    # 花费台账：既往运行已花 ¥6 上限中的 ¥5.5；本次第一次计费调用（¥1）就触发上限。
    ledger_path = cache.with_name(translate_zh.SPENT_FILENAME)
    earlier = translate_zh.SpendLedger(ledger_path, "2026-09-22T00:00:00+00:00")
    assert earlier.spent_before == 0.0 and not ledger_path.exists()
    earlier.record(5.5, calls=3, finished=True)
    ledger = translate_zh.SpendLedger(ledger_path, "2026-09-22T01:00:00+00:00")
    assert ledger.spent_before == 5.5 and json.loads(ledger_path.read_text(encoding="utf-8"))["spent_cny"] == 5.5
    root = _data_root(tmp_path)
    transcripts = load_transcripts(root)
    windows = build_windows(transcripts, max_chars=80)
    reqs = translate_zh.build_requests(windows, transcripts, size=25)
    meter = translate_zh.TranslateMeter()
    client = _BillingClient(meter, 1.0)
    cache2 = tmp_path / "cap" / "translate_cache.jsonl"
    with pytest.raises(translate_zh.CostCapExceeded, match="5.5000 CNY from earlier runs"):
        translate_zh.translate_all(lambda: client, reqs, cache2, meter, cap_cny=6.0, workers=1, spent_before=ledger.spent_before, on_spend=ledger.record)
    # One request only: the truncated batch was NOT bisected into two more priced calls after the cap tripped.
    # 只有一次请求：上限触发后被截断的批次没有再二分成两次计费调用。
    assert [c["labels"]["batch_id"] for c in client.calls] == ["r0000"] and len(meter.calls) == 1
    rows_after = translate_zh.load_cache(cache2)
    assert sorted(rows_after) == sorted(w.window_id for w in windows) and all(not t.validated and t.reason == "cap" for t in rows_after.values())
    assert translate_zh.resume_skip_ids(rows_after) == (set(), set())
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["spent_cny"] == pytest.approx(6.5)
    assert ledger.total == pytest.approx(6.5)
    # should_stop already true → no request at all.
    # should_stop 已为真 → 完全不发请求。
    out = translate_zh.translate_items(client, reqs[0].items, "r9", 0, should_stop=lambda: True)
    assert [t.reason for t in out] == ["cap", "cap"] and len(client.calls) == 1
    # A cap with headroom translates as before (cumulative ¥0 + 3 calls × ¥1 < ¥6).
    # 有余量的上限照常翻译（累计 ¥0 + 3 次 × ¥1 < ¥6）。
    meter2, client2 = translate_zh.TranslateMeter(), None
    client2 = _BillingClient(meter2, 1.0)
    results = translate_zh.translate_all(lambda: client2, reqs, tmp_path / "ok" / "translate_cache.jsonl", meter2, cap_cny=6.0, workers=1)
    assert all(t.validated for t in results.values()) and len(meter2.calls) == 3


def test_harness_run_defaults_embedder_identity_and_guard_counts(tmp_path: Path):
    """run_real.default_entity_names gives the glossary names only for --text-lang zh with --cjk on (run B), none for run A / English, and honours --entity-names ('' → none); real_arms.embedder_identity keeps 'fastembed:<model>' for the default embedder and appends '@<dims>' otherwise; retain_with_bisect sums the 'downgraded' counts an arm's _add attaches (absent → {}).
    run_real.default_entity_names 只在 --text-lang zh 且 --cjk on（运行 B）时给出术语表人名，运行 A / 英文为空，并遵守 --entity-names（'' → 无）；real_arms.embedder_identity 对默认嵌入器保持 'fastembed:<model>'、其它追加 '@<dims>'；retain_with_bisect 汇总臂的 _add 附上的 'downgraded' 计数（缺失 → {}）。

    Input / 输入：
        tmp_path: pytest temporary directory (unused; keeps the signature uniform).
        tmp_path：pytest 临时目录（未用；保持签名一致）。

    Output / 输出：
        Assertions on run_real / real_arms helpers.
        对 run_real / real_arms 辅助函数的断言。

    State effect / 状态影响：
        None.
        无。
    """
    _stub_bench_deps()
    import real_arms
    import run_real

    glossary = run_real.glossary_names()
    assert "杰克" in glossary and "我" not in glossary
    assert run_real.default_entity_names("zh", "on", None) == glossary
    assert run_real.default_entity_names("zh", "off", None) == () and run_real.default_entity_names("zh", None, None) == ()
    assert run_real.default_entity_names("en", "on", None) == () and run_real.default_entity_names("en", None, None) == ()
    assert run_real.default_entity_names("zh", "off", "杰克, 塔莎") == ("杰克", "塔莎") and run_real.default_entity_names("zh", "on", "") == ()
    assert real_arms.embedder_identity(real_arms.EMBED_MODEL, real_arms.EMBEDDING_DIMS) == "fastembed:BAAI/bge-small-en-v1.5"
    assert real_arms.embedder_identity("BAAI/bge-small-zh-v1.5", 512) == "fastembed:BAAI/bge-small-zh-v1.5@512"
    assert real_arms.embedder_identity(real_arms.EMBED_MODEL, 512) == "fastembed:BAAI/bge-small-en-v1.5@512"

    class _Arm:
        name = "elite"

        def __init__(self, envelope: dict[str, Any]) -> None:
            self.meter = SimpleNamespace(calls=[])
            self.client = SimpleNamespace(label=_label)
            self.envelope = envelope

        def _add(self, windows: Any, metadata: Any, timestamp: Any) -> dict[str, Any]:
            return self.envelope

    @contextmanager
    def _label(**labels: Any):
        yield

    window = SimpleNamespace(window_id="w0", source="A1_JAKE_DAY1_11000000", day=1, start_cs=0, end_cs=100, cue_from=0, cue_to=0, start_iso="2024-01-01T00:00:00Z", text="x")
    batch = real_arms.Batch("b0", 0, [window])
    rows = [{"event": "ADD"}, {"event": "ADD"}, {"event": "NONE"}]
    rec = real_arms.retain_with_bisect(_Arm({"results": rows, "downgraded": {"NONE": 2, "REVISE": 1}}), batch)
    assert rec["events"] == {"ADD": 2, "NONE": 1} and rec["downgraded"] == {"NONE": 2, "REVISE": 1} and rec["subs"][0]["downgraded"] == {"NONE": 2, "REVISE": 1}
    rec = real_arms.retain_with_bisect(_Arm({"results": rows}), batch)
    assert rec["downgraded"] == {} and rec["subs"][0]["downgraded"] == {}


def test_review_fixes_question_validation_window_kind_progress_glossary_add_and_entity_signal(tmp_path: Path):
    """Review fixes of 2026-09-22: validate_window(question=True) judges the Latin rate on the question line only (options keep Latin names by instruction) and translate_items passes it for 'q:' items; write_dataset marks a window 'native_fallback' only when a cue carries English AND Chinese (the build_windows(zh) predicate) and reports questions_translated / questions_native_fallback; translate_all prints a progress line every ten completed requests whatever the worker count; parse_glossary_add reads NAME=<Chinese> pairs and rejects malformed ones; EliteRealArm.recall reads the entity signal from score_details['entity_boost'] (the key scoring.score_details writes); compare_real.unique_run_roots states each run's controls once.
    2026-09-22 的评审修正：validate_window(question=True) 只按题干行判断拉丁字母率（选项按指令保留拉丁人名），translate_items 对 'q:' 单元传入它；write_dataset 只在某 cue 同时带英文与中文时（build_windows(zh) 的谓词）把窗口记为 'native_fallback'，并报告 questions_translated / questions_native_fallback；translate_all 不论线程数每完成十个请求打印一行进度；parse_glossary_add 读 NAME=中文 对并拒绝畸形值；EliteRealArm.recall 从 score_details['entity_boost']（scoring.score_details 写的键）读实体信号；compare_real.unique_run_roots 对每次运行只陈述一次控制变量。

    Input / 输入：
        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on translate_zh, real_arms and compare_real helpers.
        对 translate_zh、real_arms 与 compare_real 辅助函数的断言。

    State effect / 状态影响：
        Writes data/ and data_zh/ under tmp_path.
        在 tmp_path 下写 data/ 与 data_zh/。
    """
    # (a) question items: Latin options pass, a Latin question line still fails; windows are unchanged.
    # (a) 题目单元：拉丁选项通过，题干仍是拉丁则失败；窗口不变。
    q_src = ["Who likes pizza?", "Alice Katrina", "Tasha", "Shure", "Jake's friend"]
    q_out = ["谁喜欢披萨？", "Alice Katrina", "Tasha", "Shure", "Jake的朋友"]
    assert translate_zh.validate_window(q_src, q_out).startswith("latin_rate") and translate_zh.validate_window(q_src, q_out, question=True) is None
    assert translate_zh.validate_window(q_src, ["Who likes pizza", "Alice Katrina", "Tasha", "Shure", "Jake的朋友"], question=True).startswith("latin_rate")
    src = ["Jake: Okay, then we need a stopwatch.", "Shure: Mark a timestamp, right?"]
    assert translate_zh.validate_window(src, ["Jake: Okay then we need a stopwatch", "Shure: 时间戳吧"], question=True).startswith("latin_rate")

    class _QClient(_FakeClient):
        def create(self, **params: Any) -> Any:
            payload = json.loads(params["messages"][1]["content"])
            items = [{"window_id": it["window_id"], "lines": q_out} for it in payload["items"]]
            self.last_calls = [SimpleNamespace(finish_reason="stop")]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"items": items}, ensure_ascii=False)), finish_reason="stop")])

    rows = translate_zh.translate_items(_QClient(), [translate_zh.Item("q:7", q_src), translate_zh.Item("w0", q_src)], "q0000", 0)
    assert [(r.window_id, r.validated) for r in rows] == [("q:7", True), ("w0", False)] and rows[1].reason.startswith("latin_rate")
    # (b) write_dataset: the question provenance counts and the window-builder predicate for 'native_fallback'.
    # (b) write_dataset：题目来源计数，以及 'native_fallback' 采用窗口构建器的谓词。
    root = _data_root(tmp_path)
    out = tmp_path / "data_zh"
    transcripts = load_transcripts(root)
    windows = build_windows(transcripts, max_chars=80)
    failed = {w.window_id: Translation(w.window_id, [], "r0000", 1, False, "truncated") for w in windows}
    written = translate_zh.write_dataset(out, root, transcripts, windows, failed, QA)
    assert written["questions_translated"] == 0 and written["questions_native_fallback"] == 0
    zh_windows = build_windows(load_transcripts(out), max_chars=80, text_lang="zh")
    assert written["windows"]["native_fallback"] == 2 and written["windows"]["no_chinese"] == 0 and [w.window_id for w in zh_windows] == [w.window_id for w in windows]
    # Drop the Chinese line of every English-bearing cue but the first: the second window then has native Chinese on a
    # Chinese-only cue alone, which build_windows(zh) cannot use — write_dataset must say 'no_chinese', not 'native_fallback'.
    # 去掉除第一条外每条带英文 cue 的中文行：第二个窗口只在纯中文 cue 上有原生中文，build_windows(zh) 用不上——
    # write_dataset 必须记 'no_chinese' 而非 'native_fallback'。
    first = min(c.index for cues in transcripts.values() for c in cues)
    stripped = {src_name: [type(c)(**{**c.__dict__, "text_zh": (c.text_zh if (c.index == first or not c.text_en) else "")}) for c in cues] for src_name, cues in transcripts.items()}
    written = translate_zh.write_dataset(tmp_path / "data_zh2", root, stripped, windows, failed, QA)
    assert written["windows"] == {"translated": 0, "native_fallback": 1, "no_chinese": 1} and written["windows_no_chinese"] == [windows[1].window_id]
    q_tr = {**failed, "q:6": Translation("q:6", ["杰克 需要什么？", "秒表", "披萨", "我", "Tasha"], "q0000", 0, True)}
    written = translate_zh.write_dataset(tmp_path / "data_zh3", root, transcripts, windows, q_tr, QA, questions_source="translate")
    assert written["questions_translated"] == 1 and written["questions_native_fallback"] == 1
    # (c) progress cadence: 12 single-window requests over 3 workers → lines at 10 (never 3/6/9/12).
    # (c) 进度节奏：3 个线程跑 12 个单窗口请求 → 在 10 处打印（绝不是 3/6/9/12）。
    lines: list[str] = []
    reqs = [translate_zh.Request(f"r{i:04d}", [translate_zh.Item(f"w{i}", ["Jake: Okay, then we need a stopwatch."])]) for i in range(12)]
    translate_zh.translate_all(lambda: _FakeClient(), reqs, tmp_path / "prog" / "translate_cache.jsonl", translate_zh.TranslateMeter(), cap_cny=6.0, workers=3, progress=lines.append)
    assert len(lines) == 1 and "requests 12/12" in lines[0]
    lines.clear()
    translate_zh.translate_all(lambda: _FakeClient(), reqs[:9], tmp_path / "prog2" / "translate_cache.jsonl", translate_zh.TranslateMeter(), cap_cny=6.0, workers=4, progress=lines.append)
    assert lines == []
    # (d) --glossary-add: NAME=中文 pairs on top of NAME_GLOSSARY, malformed values rejected.
    # (d) --glossary-add：叠加在 NAME_GLOSSARY 之上的 NAME=中文 对，畸形值被拒绝。
    assert translate_zh.parse_glossary_add(["Jack=杰克", " Violet = 维奥莱特 "]) == {"Jack": "杰克", "Violet": "维奥莱特"} and translate_zh.parse_glossary_add([]) == {}
    for bad in ("Jack", "=杰克", "Jack="):
        with pytest.raises(ValueError, match="NAME="):
            translate_zh.parse_glossary_add([bad])
    assert translate_zh.apply_glossary("Jack: hi Jake", {**translate_zh.NAME_GLOSSARY, "Jack": "杰克"}) == "杰克: hi 杰克"
    args = translate_zh.build_parser().parse_args(["--glossary-add", "Jack=杰克", "--dry-run"])
    assert args.glossary_add == ["Jack=杰克"]
    # (e) the harness reads the entity signal under the key score_details actually carries.
    # (e) 基准脚本按 score_details 实际携带的键读实体信号。
    _stub_bench_deps()
    import compare_real
    import real_arms

    details = {"bm25_score": 0.5, "entity_boost": 1.0, "recalled_by": ["semantic", "entity"], "channel_sizes": {"semantic": 3, "entity": 1}}
    fake = SimpleNamespace(name="elite", client=SimpleNamespace(label=_label_cm), hms=SimpleNamespace(search=lambda *a, **k: {"results": [{"memory": "杰克买了一把新吉他。", "score": 0.7, "id": "a1", "metadata": {"batch_id": "b0"}, "score_details": details}]}))
    row = real_arms.EliteRealArm.recall(fake, "杰克买了什么？", "2024-01-01T00:00:00Z")[0]
    assert row["entity"] == 1.0 and row["keyword"] == 0.5 and row["recalled_by"] == ["semantic", "entity"] and row["batch_id"] == "b0"
    # (f) compare_real states one control line per run root.
    # (f) compare_real 对每个运行目录只陈述一次控制变量。
    r1, r2 = SimpleNamespace(root=Path("/x/zhA")), SimpleNamespace(root=Path("/x/en"))
    assert [r.root.name for r in compare_real.unique_run_roots([(r1, "elite_rows"), (r1, "elite_hybrid"), (r2, "elite_rows")])] == ["zhA", "en"]


@contextmanager
def _label_cm(**labels: Any):
    """A no-op DeepSeekClient.label stand-in.
    DeepSeekClient.label 的空操作替身。
    """
    yield
