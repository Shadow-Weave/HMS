"""Zero-API replay A/B: the recorded deepseek-flash extraction replies of one real run are fed to HMS twice — verbatim
(full contract) and reduced to the compact key set (compact contract, the library derives the rest) — over the same
streaming timeline; retrieval is scored locally, then one cheap answer pass per arm.
零 API 回放 A/B：把一次真实 run 记录下来的 deepseek-flash 抽取回复送入 HMS 两次——原样（完整契约）与裁剪到紧凑键集
（紧凑契约，其余字段由库推导）——沿同一条流式时间线；检索在本地计分，随后每个臂做一次廉价的答题。
"""

# Purpose (EN): Measure what the compact extraction contract costs in retrieval / answer quality when the LLM-written
#               texts are held IDENTICAL (same recorded replies), so the only difference is which fields the library
#               derives locally. Also measure how often the local derivations agree with what the LLM had emitted.
# 目的（中）：在 LLM 写出的文本完全相同（同一批记录回复）的前提下，测量紧凑抽取契约在检索 / 答题质量上的代价，
#            差异只剩"哪些字段由库在本地推导"。同时测量本地推导与 LLM 原本输出的一致率。

# Input (EN): results/real/<run>/{extract_raw_elite.jsonl, batches.json, windows.jsonl}; the dataset under data/;
#             benchmarks/egolife/.env for the answer pass (DEEPSEEK_API_KEY; never printed).
# 输入（中）：results/real/<run>/{extract_raw_elite.jsonl, batches.json, windows.jsonl}；data/ 下的数据集；
#            答题阶段读取 benchmarks/egolife/.env（DEEPSEEK_API_KEY；绝不打印）。

# Output (EN): <out>/{summary.json, summary.md, per_question_full.jsonl, per_question_compact.jsonl, calls.jsonl,
#              retain_full.jsonl, retain_compact.jsonl, entity_axis_full.jsonl, entity_axis_compact.jsonl}.
# 输出（中）：<out>/{summary.json, summary.md, per_question_full.jsonl, per_question_compact.jsonl, calls.jsonl,
#            retain_full.jsonl, retain_compact.jsonl, entity_axis_full.jsonl, entity_axis_compact.jsonl}。

# State effect (EN): Reads the run directory once (never writes into it); builds two SQLite stores under <out>/work
#                    (removed unless --keep-db); the answer pass makes at most --max-answer-requests API calls and stops
#                    when the projected or running cost exceeds --cost-cap-usd.
# 状态影响（中）：只读一次 run 目录（绝不写入）；在 <out>/work 下建两个 SQLite 库（除非 --keep-db 否则删除）；
#                答题阶段至多 --max-answer-requests 次 API 调用，预计或累计费用超过 --cost-cap-usd 即停止。

# Variant mode (EN, added 2026-09-11 for the compact-loss diagnosis): --replies-run / --replies-contract choose whose
#               recorded replies are replayed and how their observation items are read (full = the r3 shape, reduced to
#               the compact keys for the compact arm as before; compact = a compact run's items served as recorded, only
#               the compact arm is meaningful). --force-type / --drop-entities / --drop-when / --keep-only-text rewrite
#               every compact item before it reaches HMS, so one axis (type, entities, when, everything but text) can be
#               removed while the texts are held fixed. --arms picks which arms run. --verdict-min-similarity FLOAT sets
#               ingest.verdict_min_similarity (0.0 = guard off) so the library's verdict guard can be replayed on and off;
#               the guard's DEBUG lines ('verdict X on ... downgraded to ADD') are counted per verdict (GuardCapture).
# 变体模式（中，2026-09-11 为紧凑损失诊断加入）：--replies-run / --replies-contract 选择回放谁的记录回复以及如何读其观察
#               条目（full = r3 的形状，紧凑臂照旧裁剪到紧凑键；compact = 紧凑 run 的条目按记录原样回放，只有紧凑臂有意义）。
#               --force-type / --drop-entities / --drop-when / --keep-only-text 在每条紧凑条目进入 HMS 前改写它，
#               从而在文本不变的前提下去掉某一个轴（类型、实体、时间、除文本外的一切）。--arms 选择运行哪些臂。
#               --verdict-min-similarity FLOAT 设置 ingest.verdict_min_similarity（0.0 = 关闭守卫），使库的裁决守卫可开可关地
#               回放；守卫的 DEBUG 行（'verdict X on ... downgraded to ADD'）按裁决计数（GuardCapture）。

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accounting import PRICES_USD_PER_M, Meter, rate_tier, utc_now  # noqa: E402
from loader import DEFAULT_DATA_ROOT, QA_FILENAME, Question, load_questions, load_transcripts  # noqa: E402
from real_arms import (  # noqa: E402
    ANSWER_MAX_TOKENS,
    BANK,
    BATCH_SIZE,
    EMBED_MODEL,
    EMBEDDING_DIMS,
    THRESHOLD,
    TOP_K,
    TRUNCATED,
    Batch,
    DeepSeekClient,
    LogCapture,
    answer,
    build_question_prompt,
    load_api_key,
    plan_batches,
    rows_to_context,
    sub_metadata,
)
from timeline import Step, schedule  # noqa: E402
from windows import Window, build_windows, clock_to_iso  # noqa: E402

from hms_elite import HMS, HMSConfig  # noqa: E402
from hms_elite.configs.prompts import COMPACT_OBSERVATION_KEYS, parse_sections, task_of  # noqa: E402
from hms_elite.enums import ObservationType  # noqa: E402
from hms_elite.exceptions import LLMError  # noqa: E402
from hms_elite.ingest import extraction  # noqa: E402
from hms_elite.ingest.derive import expand_compact_item  # noqa: E402
from hms_elite.llms.base import BaseLlmConfig, LLMBase  # noqa: E402
from hms_elite.memory.storage import StateStore  # noqa: E402
from hms_elite.timeutil import date_of  # noqa: E402
from hms_elite.utils.json import parse_llm_json  # noqa: E402

ENV_PATH = HERE / ".env"
DEFAULT_RUN = HERE / "results" / "real" / "elite_full204_r3"
DEFAULT_OUT = HERE / "results" / "replay_ab"
# 'lines' (added 2026-09-18 for the reader-side A/B): the recorded raw tab-text replies of a lines run are served
# verbatim through ReplayLLM.generate_lines under extraction_output='lines'.
# 'lines'（2026-09-18 为阅读端 A/B 加入）：行式 run 记录的原始 tab 文本回复经 ReplayLLM.generate_lines 在
# extraction_output='lines' 下原样回放。
ARMS = ("full", "compact", "lines")
ANSWER_LABEL = {"full": "replay_full", "compact": "replay_compact", "lines": "replay_lines"}
EXTRACTION_TASKS = ("memory_extraction", "memory_extraction_compact", "memory_extraction_lines")
MAX_ANSWER_REQUESTS = 408
COST_CAP_USD = 0.10
# Conservative chars-per-token for the pre-flight projection (r3 measured ~4.5 chars/token on these prompts).
# 预检投影用的保守字符/词元比（r3 在这类提示词上实测约 4.5 字符/词元）。
PROJECTION_CHARS_PER_TOKEN = 3.0
AGREEMENT_FIELDS = ("sentiment", "relation", "predicate", "topic_tags", "entity_mentions")
LIST_FIELDS = ("topic_tags", "entity_mentions")
REPLIES_CONTRACTS = ("full", "compact", "lines")
# Per-question hook of run_arm (the reader-side A/B seam): (hms, step, query_iso, raw search rows, record) -> None,
# called after the step's retain and the baseline search while the store is at that step's exact state; it may
# enrich the record in place.
# run_arm 的逐题钩子（阅读端 A/B 的接缝）：(hms, step, query_iso, 原始检索行, 记录) -> None，在该步 retain 与基线
# 检索之后、库恰处于该步状态时调用；可原地扩充记录。
ContextHook = Callable[[HMS, Step, str, list[dict[str, Any]], dict[str, Any]], None]
# The keys --keep-only-text preserves on a compact item / --keep-only-text 在紧凑条目上保留的键
TEXT_ONLY_KEYS = ("id", "event_idx", "text", "type")


@dataclass(frozen=True)
class Transforms:
    """Per-item rewrites applied to every compact observation item before it is served to HMS (compact arm only).
    Order: force_type, drop_entities, drop_when, keep_only_text (the last one subsumes the two drops).
    在每条紧凑观察条目送入 HMS 前施加的改写（仅紧凑臂）。顺序：force_type、drop_entities、drop_when、keep_only_text
    （最后一项涵盖前两项 drop）。
    """

    force_type: str | None = None  # every observation type := this value / 所有观察类型 := 此值
    drop_entities: bool = False  # entities := [] / 实体 := []
    drop_when: bool = False  # remove `when` and `valid_to` / 去掉 when 与 valid_to
    keep_only_text: bool = False  # keep TEXT_ONLY_KEYS only / 只保留 TEXT_ONLY_KEYS
    # Reply-level: every recorded decision := ADD (target_id null), so no observation is dropped as a NONE duplicate
    # or folded into a REVISE / CONFLICT; the observation items are untouched.
    # 回复级：每条记录的决策 := ADD（target_id 置空），观察不再因 NONE 重复被丢弃或被 REVISE / CONFLICT 折叠；观察条目不动。
    force_add: bool = False

    @property
    def active(self) -> bool:
        """Whether any rewrite is on. / 是否有任一改写打开。"""
        return bool(self.force_type or self.drop_entities or self.drop_when or self.keep_only_text or self.force_add)

    @property
    def item_active(self) -> bool:
        """Whether any per-item rewrite is on. / 是否有任一逐条改写打开。"""
        return bool(self.force_type or self.drop_entities or self.drop_when or self.keep_only_text)

    def describe(self) -> dict[str, Any]:
        """Plain dict for summary.json. / 写入 summary.json 的普通字典。"""
        return {"force_type": self.force_type, "drop_entities": self.drop_entities, "drop_when": self.drop_when,
                "keep_only_text": self.keep_only_text, "force_add": self.force_add}  # fmt: skip


def force_add_decisions(reply: dict[str, Any], counters: Counter) -> None:
    """Rewrite the reply's decisions in place: every dict item gets decision 'ADD' and target_id None; a decisions root
    that is not a list is left alone (the library already reads that as all-ADD).
    原地改写回复的决策：每个 dict 条目的 decision 置 'ADD'、target_id 置 None；决策根不是列表时不动（库本就按全 ADD 读取）。

    Input / 输入：
        reply: The parsed reply object.
        reply：解析后的回复对象。

        counters: Reply-level counters (non-ADD verdicts rewritten, by verdict).
        counters：回复级计数（被改写的非 ADD 裁决，按裁决分类）。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Mutates reply; updates counters.
        修改 reply；更新 counters。
    """
    decisions = reply.get("decisions")
    if not isinstance(decisions, list):
        return
    for d in decisions:
        if not isinstance(d, dict):
            continue
        verdict = str(d.get("decision") or "").strip().upper()
        if verdict != "ADD":
            counters[f"transform_force_add_from_{verdict or '<none>'}"] += 1
        d["decision"] = "ADD"
        d["target_id"] = None


def apply_transforms(item: dict[str, Any], t: Transforms, counters: Counter) -> dict[str, Any]:
    """Rewrite one compact item per the transforms; the input dict is not mutated.
    按变换改写一条紧凑条目；不修改输入字典。

    Input / 输入：
        item: A compact-key observation item (recorded or projected).
        item：紧凑键观察条目（记录的或投影的）。

        t: The transforms.
        t：变换。

        counters: Reply-level counters (what each transform actually changed).
        counters：回复级计数（每个变换实际改动了什么）。

    Output / 输出：
        The rewritten item.
        改写后的条目。

    State effect / 状态影响：
        Updates counters.
        更新 counters。
    """
    out = dict(item)
    if t.force_type:
        if str(out.get("type") or "") != t.force_type:
            counters["transform_type_changed"] += 1
        out["type"] = t.force_type
    if t.drop_entities:
        if out.get("entities"):
            counters["transform_entities_dropped_nonempty"] += 1
        out["entities"] = []
    if t.drop_when:
        if out.get("when") not in (None, ""):
            counters["transform_when_dropped"] += 1
        if out.get("valid_to") not in (None, ""):
            counters["transform_valid_to_dropped"] += 1
        out.pop("when", None)
        out.pop("valid_to", None)
    if t.keep_only_text:
        for k in out:
            if k not in TEXT_ONLY_KEYS and out.get(k) not in (None, "", []):
                counters[f"transform_keep_only_text_dropped_{k}"] += 1
        out = {k: out[k] for k in TEXT_ONLY_KEYS if k in out}
    return out


# ---------------------------------------------------------------------------
# Recorded replies
# 记录的回复
# ---------------------------------------------------------------------------
@dataclass
class Recorded:
    """The extraction replies of one run, indexed by (batch_id, split), plus the file-level counters.
    一次 run 的抽取回复，按 (batch_id, split) 索引，附文件级计数。
    """

    rows: dict[tuple[str, str | None], list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    lines: int = 0
    lines_truncated: int = 0
    batches_seen: set[str] = field(default_factory=set)

    def best(self, batch_id: str, split: str | None) -> dict[str, Any] | None:
        """The reply to serve for one slice: the last attempt whose finish_reason is not 'length' and whose content is
        non-empty; None when every recorded attempt of the slice is truncated or the slice was never recorded.
        某切片应回放的回复：最后一次 finish_reason 不为 'length' 且内容非空的尝试；该切片的所有尝试都被截断或从未记录时
        为 None。
        """
        usable = [
            r
            for r in self.rows.get((batch_id, split), [])
            if r.get("finish_reason") != TRUNCATED and isinstance(r.get("content"), str) and r["content"].strip()
        ]
        return usable[-1] if usable else None


def load_recorded(path: Path) -> Recorded:
    """Read extract_raw_elite.jsonl once; a partial last line (the run may still be writing) is counted, not parsed.
    只读一次 extract_raw_elite.jsonl；不完整的最后一行（run 可能仍在写）只计数不解析。

    Input / 输入：
        path: The JSONL file.
        path：JSONL 文件。

    Output / 输出：
        Recorded.
        Recorded。

    State effect / 状态影响：
        None.
        无。
    """
    rec = Recorded()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec.lines += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rec.lines_truncated += 1
            continue
        batch_id = str(row.get("batch_id"))
        split = row.get("split") if isinstance(row.get("split"), str) else None
        rec.rows[(batch_id, split)].append(row)
        rec.batches_seen.add(batch_id)
    return rec


@dataclass
class Leaf:
    """One add() of the replay: a window slice of a batch and the reply that answers it (None = unresolved).
    回放中的一次 add()：批次的一个窗口切片及其对应回复（None = 未解决）。
    """

    batch_id: str
    lo: int
    hi: int
    split: str | None
    row: dict[str, Any] | None


def resolve_leaves(batch: Batch, recorded: Recorded) -> list[Leaf]:
    """Re-walk retain_with_bisect's bisection on the recorded replies: a slice with a usable reply is a leaf; a slice
    whose recorded attempts are all truncated is split in two (mid = lo + (hi - lo) // 2) and the halves resolved
    recursively; a slice with no usable reply and no recorded halves is an unresolved leaf (row None).
    在记录的回复上重走 retain_with_bisect 的二分：有可用回复的切片是叶子；所有尝试都被截断的切片一分为二
    （mid = lo + (hi - lo) // 2）并递归解决两半；既无可用回复也没有记录的两半的切片是未解决叶子（row 为 None）。

    Input / 输入：
        batch: The planned batch.
        batch：计划中的批次。

        recorded: The recorded replies.
        recorded：记录的回复。

    Output / 输出：
        Leaves in submission order (the order retain_with_bisect used).
        按提交顺序（retain_with_bisect 使用的顺序）的叶子列表。

    State effect / 状态影响：
        None.
        无。
    """
    n = len(batch.windows)
    out: list[Leaf] = []
    queue: list[tuple[int, int]] = [(0, n)]
    while queue:
        lo, hi = queue.pop(0)
        split = None if (lo, hi) == (0, n) else f"{lo}-{hi}"
        row = recorded.best(batch.batch_id, split)
        if row is not None:
            out.append(Leaf(batch.batch_id, lo, hi, split, row))
            continue
        attempts = recorded.rows.get((batch.batch_id, split), [])
        truncated = attempts and all(r.get("finish_reason") == TRUNCATED for r in attempts)
        if truncated and hi - lo > 1:
            mid = lo + (hi - lo) // 2
            queue.insert(0, (mid, hi))
            queue.insert(0, (lo, mid))
            continue
        out.append(Leaf(batch.batch_id, lo, hi, split, None))
    return out


# ---------------------------------------------------------------------------
# Compact reduction and derived-vs-LLM agreement
# 紧凑裁剪与推导-LLM 一致性
# ---------------------------------------------------------------------------
def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets; two empty sets count as identical (1.0).
    两个集合的 Jaccard 相似度；两个空集视为相同（1.0）。
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _lower_set(values: Any) -> set[str]:
    """Lower-cased set of the strings in an LLM list field (the library's _str_list coercion first).
    LLM 列表字段中字符串的小写集合（先经库的 _str_list 规整）。
    """
    return {v.lower() for v in extraction._str_list(values, what="replay_ab", field="list")}


@dataclass
class Agreement:
    """Per-field tallies of locally derived values against the LLM-emitted values of the same observation.
    同一观察上本地推导值与 LLM 输出值的逐字段计数。
    """

    n: int = 0
    exact: Counter = field(default_factory=Counter)  # sentiment / relation / predicate equal / 相等
    jaccard_positive: Counter = field(default_factory=Counter)  # list fields with Jaccard > 0 / 列表字段 Jaccard > 0
    jaccard_sum: Counter = field(default_factory=Counter)
    both_empty: Counter = field(default_factory=Counter)
    llm_empty: Counter = field(default_factory=Counter)
    derived_empty: Counter = field(default_factory=Counter)
    sentiment_pairs: Counter = field(default_factory=Counter)  # 'llm->derived' / 'LLM->推导'
    relation_pairs: Counter = field(default_factory=Counter)
    predicate_pairs: Counter = field(default_factory=Counter)
    speaker_from_subject: int = 0
    speaker_explicit: int = 0

    def add(self, llm_item: dict[str, Any], derived: dict[str, Any]) -> None:
        """Tally one observation.
        计入一个观察。

        Input / 输入：
            llm_item: The recorded full-contract item.
            llm_item：记录的完整契约条目。

            derived: expand_compact_item's output for its compact reduction.
            derived：其紧凑裁剪经 expand_compact_item 的输出。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates the counters.
            更新计数。
        """
        self.n += 1
        for fld in ("sentiment", "relation", "predicate"):
            llm_value = str(llm_item.get(fld) or "").strip().lower()
            got = str(derived.get(fld) or "").strip().lower()
            if llm_value == got:
                self.exact[fld] += 1
            getattr(self, f"{fld}_pairs")[f"{llm_value or '<none>'}->{got or '<none>'}"] += 1
        for fld in LIST_FIELDS:
            a, b = _lower_set(llm_item.get(fld)), _lower_set(derived.get(fld))
            j = _jaccard(a, b)
            self.jaccard_sum[fld] += j
            if j > 0:
                self.jaccard_positive[fld] += 1
            if not a and not b:
                self.both_empty[fld] += 1
            if not a:
                self.llm_empty[fld] += 1
            if not b:
                self.derived_empty[fld] += 1

    def summary(self) -> dict[str, Any]:
        """Fractions per field plus the raw counts.
        逐字段比例与原始计数。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'n', 'agreement': {field: fraction}, 'mean_jaccard': {...}, 'counts': {...}, 'pairs': {...}}.
            {'n', 'agreement': {字段: 比例}, 'mean_jaccard': {...}, 'counts': {...}, 'pairs': {...}}。

        State effect / 状态影响：
            None.
            无。
        """
        n = self.n or 1
        agreement = {f: round(self.exact[f] / n, 4) for f in ("sentiment", "relation", "predicate")}
        agreement.update({f: round(self.jaccard_positive[f] / n, 4) for f in LIST_FIELDS})
        return {
            "n": self.n,
            "agreement": agreement,
            "mean_jaccard": {f: round(self.jaccard_sum[f] / n, 4) for f in LIST_FIELDS},
            "counts": {
                "exact": dict(self.exact),
                "jaccard_positive": dict(self.jaccard_positive),
                "both_empty": dict(self.both_empty),
                "llm_empty": dict(self.llm_empty),
                "derived_empty": dict(self.derived_empty),
                "speaker_explicit": self.speaker_explicit,
                "speaker_from_subject": self.speaker_from_subject,
            },
            "pairs": {
                "sentiment": dict(self.sentiment_pairs.most_common()),
                "relation": dict(self.relation_pairs.most_common()),
                "predicate_top40": dict(self.predicate_pairs.most_common(40)),
            },
        }


def compact_item(
    item: dict[str, Any], *, message_date: str, speaker_from_subject: bool, entities_from_mentions: bool
) -> tuple[dict[str, Any], str | None]:
    """Reduce one recorded full-contract observation to the compact key set: id, event_idx, text, type, speaker,
    supersedes_hint_ids (only when non-empty), when (date_observed only when it differs from the message date),
    valid_to (only when given), entities (the recorded `entity_mentions` through the library's _str_list, only when
    non-empty and entities_from_mentions is on — this is what the real compact prompt asks the LLM for). The recorded
    replies never carry `speaker`; with speaker_from_subject the subject is used when the library would accept it as a
    speaker (a named person, not a side or collective word).
    把一条记录的完整契约观察裁剪到紧凑键集：id、event_idx、text、type、speaker、supersedes_hint_ids（仅非空时）、
    when（仅当 date_observed 与消息日期不同）、valid_to（仅给定时）、entities（记录的 `entity_mentions` 经库的 _str_list
    规整，仅非空且 entities_from_mentions 打开时——真实紧凑提示词向 LLM 要的正是这个）。记录的回复从不携带 `speaker`；
    speaker_from_subject 打开时，若库会把 subject 当作说话人接受（具名人物，而非侧别词或集体词），则用 subject。

    Input / 输入：
        item: The recorded item.
        item：记录的条目。

        message_date: `YYYY-MM-DD` of the slice's messages (the add() timestamp).
        message_date：切片消息的 `YYYY-MM-DD`（add() 的 timestamp）。

        speaker_from_subject: Whether a named subject becomes the compact `speaker`.
        speaker_from_subject：具名 subject 是否成为紧凑 `speaker`。

        entities_from_mentions: Whether the recorded entity_mentions become the compact `entities` hint (a malformed
        recorded value — a number or dict where the list belongs — is dropped silently, as the library would ignore it).
        entities_from_mentions：记录的 entity_mentions 是否成为紧凑 `entities` 提示（记录值畸形——列表位置出现数字或
        dict——时静默丢弃，与库忽略它的行为一致）。

    Output / 输出：
        (compact item, where the speaker came from: 'explicit' | 'subject' | None).
        （紧凑条目，说话人来源：'explicit' | 'subject' | None）。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, Any] = {
        "id": item.get("id"),
        "event_idx": item.get("event_idx"),
        "text": item.get("text"),
        "type": item.get("type"),
    }
    speaker = extraction._speaker(item.get("speaker"))
    origin: str | None = "explicit" if speaker else None
    if speaker is None and speaker_from_subject:
        speaker = extraction._speaker(item.get("subject"))
        origin = "subject" if speaker else None
    out["speaker"] = speaker
    hints = item.get("supersedes_hint_ids")
    if (isinstance(hints, (list, tuple)) and any(h not in (None, "") for h in hints)) or (isinstance(hints, str) and hints.strip()):
        out["supersedes_hint_ids"] = list(hints) if isinstance(hints, (list, tuple)) else hints
    when = item.get("date_observed")
    if when not in (None, "") and str(when).strip() != message_date:
        out["when"] = when
    if item.get("valid_to") not in (None, ""):
        out["valid_to"] = item.get("valid_to")
    if entities_from_mentions:
        try:
            entities = extraction._str_list(item.get("entity_mentions"), what="replay_ab", field="entity_mentions")
        except TypeError:
            entities = []
        if entities:
            out["entities"] = entities
    assert set(out) <= set(COMPACT_OBSERVATION_KEYS)
    return out, origin


def reduce_reply(
    content: str,
    *,
    message_date: str,
    extractor: Any,
    agreement: Agreement,
    speaker_from_subject: bool,
    entities_from_mentions: bool,
    counters: Counter,
    transforms: Transforms | None = None,
) -> str:
    """Rewrite one recorded reply for the compact arm: observations reduced to the compact keys (then rewritten by
    `transforms` when given), eps / decisions verbatim; each observation's local derivation (with its `entities` hint,
    exactly as the library would coerce it) is tallied against the recorded values. A reply that does not parse as an
    object (or as a bare observation list) is served verbatim and counted.
    为紧凑臂改写一条记录的回复：观察裁剪到紧凑键（给定 `transforms` 时再改写），eps / decisions 原样；每条观察的本地推导
    （带其 `entities` 提示，规整方式与库一致）与记录值做比对计数。不能解析为对象（或裸观察列表）的回复原样回放并计数。

    Input / 输入：
        content: The recorded content.
        content：记录的内容。

        message_date: The slice's message date.
        message_date：切片的消息日期。

        extractor: The compact HMS's entity extractor (the one the library will use).
        extractor：紧凑 HMS 的实体抽取器（库将使用的那个）。

        agreement: Tallies.
        agreement：计数器。

        speaker_from_subject, entities_from_mentions: See compact_item.
        speaker_from_subject、entities_from_mentions：见 compact_item。

        counters: Reply-level counters.
        counters：回复级计数。

    Output / 输出：
        The JSON text to serve.
        要回放的 JSON 文本。

    State effect / 状态影响：
        Updates agreement and counters.
        更新 agreement 与 counters。
    """
    raw = parse_llm_json(content, key=None, task="replay_ab")
    if isinstance(raw, list) and raw:
        raw = {"observations": raw}
        counters["replies_bare_list"] += 1
    if not isinstance(raw, dict):
        counters["replies_not_reducible_served_verbatim"] += 1
        return content
    reply = dict(raw)
    observations = reply.get("observations")
    if not isinstance(observations, list):
        counters["replies_without_observation_list"] += 1
        return json.dumps(reply, ensure_ascii=False)
    reduced: list[Any] = []
    for item in observations:
        if not isinstance(item, dict):
            reduced.append(item)
            counters["observation_items_non_dict"] += 1
            continue
        small, origin = compact_item(
            item, message_date=message_date, speaker_from_subject=speaker_from_subject, entities_from_mentions=entities_from_mentions
        )
        if transforms is not None and transforms.item_active:
            small = apply_transforms(small, transforms, counters)
        reduced.append(small)
        if origin == "explicit":
            agreement.speaker_explicit += 1
        elif origin == "subject":
            agreement.speaker_from_subject += 1
        counters["observations_with_entities_hint" if "entities" in small else "observations_without_entities_hint"] += 1
        obs_type = extraction._enum_value(
            small.get("type"), ObservationType, default="fact", aliases=extraction._TYPE_ALIASES, what="replay_ab", field="type"
        )
        derived = expand_compact_item(
            small, obs_type=obs_type, speaker=small.get("speaker"), role="user", message_date=message_date, extractor=extractor,
            entities=extraction._entities_hint(small, what="replay_ab"),
        )  # fmt: skip
        agreement.add(item, derived)
    reply["observations"] = reduced
    counters["observations_reduced"] += len(reduced)
    if transforms is not None and transforms.force_add:
        force_add_decisions(reply, counters)
    return json.dumps(reply, ensure_ascii=False)


def prepare_compact_reply(content: str, *, transforms: Transforms, counters: Counter) -> str | None:
    """Serve one reply recorded under the COMPACT contract: the observation items are already compact-keyed, so they are
    passed through as recorded (rewritten by `transforms` when any is on); eps / decisions verbatim. A reply whose root
    is not a JSON object (the two compact-run batches whose reply the lenient parser reads as a bare list) is skipped —
    the real run stored nothing for them either ('reply is not a JSON object (list); nothing extracted').
    回放一条在紧凑契约下记录的回复：观察条目本就是紧凑键，按记录原样透传（有变换时改写）；eps / decisions 原样。
    根不是 JSON 对象的回复（紧凑 run 中两个被宽松解析器读成裸列表的批次）跳过——真实 run 对它们同样什么都没存
    （'reply is not a JSON object (list); nothing extracted'）。

    Input / 输入：
        content: The recorded content.
        content：记录的内容。

        transforms: The per-item rewrites.
        transforms：逐条改写。

        counters: Reply-level counters.
        counters：回复级计数。

    Output / 输出：
        The JSON text to serve, or None when the reply is skipped.
        要回放的 JSON 文本；跳过时为 None。

    State effect / 状态影响：
        Updates counters.
        更新 counters。
    """
    raw = parse_llm_json(content, key=None, task="replay_ab")
    if not isinstance(raw, dict):
        counters["compact_replies_not_object_skipped"] += 1
        return None
    reply = dict(raw)
    observations = reply.get("observations")
    if not isinstance(observations, list):
        counters["replies_without_observation_list"] += 1
        return json.dumps(reply, ensure_ascii=False)
    items: list[Any] = []
    for item in observations:
        if not isinstance(item, dict):
            items.append(item)
            counters["observation_items_non_dict"] += 1
            continue
        extra = set(item) - set(COMPACT_OBSERVATION_KEYS)
        if extra:
            counters["observation_items_with_extra_keys"] += 1
        counters["observations_with_entities_hint" if item.get("entities") else "observations_without_entities_hint"] += 1
        if item.get("when") not in (None, ""):
            counters["observations_with_when"] += 1
        counters[f"observation_type_{item.get('type')}"] += 1
        items.append(apply_transforms(item, transforms, counters) if transforms.item_active else dict(item))
    reply["observations"] = items
    counters["observations_served"] += len(items)
    if transforms.force_add:
        force_add_decisions(reply, counters)
    return json.dumps(reply, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Replay LLM
# 回放 LLM
# ---------------------------------------------------------------------------
class ReplayLlmConfig(BaseLlmConfig):
    """Config of the replay provider (no fields beyond BaseLlmConfig).
    回放提供者的配置（BaseLlmConfig 之外没有字段）。
    """


class ReplayLLM(LLMBase):
    """Serves recorded extraction replies: the user prompt's `New Messages` texts identify the slice; the reply text
    was registered for that ordered tuple of texts by the driver. Any other task or an unknown slice raises.
    回放记录的抽取回复：用户提示词中 `New Messages` 的文本标识切片；驱动器已为该有序文本元组登记回复文本。
    其它任务或未知切片抛出异常。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = ReplayLlmConfig
    provider_name: ClassVar[str] = "replay"

    def __init__(self, arm: str) -> None:
        """Empty table; the driver fills it slice by slice.
        空表；驱动器逐切片填充。
        """
        super().__init__(ReplayLlmConfig(model=f"replay-{arm}"))
        self.arm = arm
        self.table: dict[tuple[str, ...], str] = {}
        self.expect: tuple[str, str | None] | None = None
        self.served: dict[tuple[str, ...], tuple[str, str | None]] = {}
        self.calls = 0
        self.mismatches = 0
        self.tasks: Counter = Counter()
        # generate_lines traffic: calls and slices with no registered reply (each miss also raises).
        # generate_lines 流量：调用数与无登记回复的切片数（每次未命中同时抛出）。
        self.lines_calls = 0
        self.lines_misses = 0

    @property
    def identity(self) -> str:
        """'replay@<arm>'.
        'replay@<arm>'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The identity string.
            身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return f"replay@{self.arm}"

    def register(self, texts: list[str], content: str, slice_key: tuple[str, str | None]) -> None:
        """Register the reply for one slice (ordered window texts → content).
        为一个切片登记回复（有序窗口文本 → 内容）。

        Input / 输入：
            texts: The slice's window texts in order.
            texts：切片窗口文本，按序。

            content: The JSON text to serve.
            content：要回放的 JSON 文本。

            slice_key: (batch_id, split) for the mismatch check.
            slice_key：用于错配检查的 (batch_id, split)。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Updates the table; the driver sets `expect` before each add().
            更新表；驱动器在每次 add() 前设置 `expect`。
        """
        self.table[tuple(texts)] = content
        self.served[tuple(texts)] = slice_key

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Look the slice up by the `New Messages` texts and return its recorded (or reduced) content.
        按 `New Messages` 文本查找切片并返回其记录（或裁剪后）的内容。

        Input / 输入：
            messages: System + user messages built by the library.
            messages：库构造的 system + user 消息。

            tools, tool_choice, kwargs: Ignored.
            tools、tool_choice、kwargs：忽略。

        Output / 输出：
            The content string.
            内容字符串。

        State effect / 状态影响：
            Counts calls / mismatches; LLMError(LLM_003) for a non-extraction task or an unknown slice.
            计数调用 / 错配；非抽取任务或未知切片抛出 LLMError(LLM_003)。
        """
        system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
        task = task_of(system) if isinstance(system, str) else None
        self.tasks[str(task)] += 1
        self.calls += 1
        if task not in EXTRACTION_TASKS:
            raise LLMError(f"replay has no reply for task {task!r}", "LLM_003", details={"task": task})
        user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
        section = parse_sections(user if isinstance(user, str) else "").get("New Messages")
        texts = tuple(str(m.get("text") or "") for m in section) if isinstance(section, list) else ()
        content = self.table.get(texts)
        if content is None:
            raise LLMError(
                f"replay has no recorded reply for this slice (expected {self.expect})", "LLM_003",
                details={"task": task, "expect": self.expect, "n_messages": len(texts)},
            )  # fmt: skip
        if self.expect is not None and self.served.get(texts) != self.expect:
            self.mismatches += 1
        return content

    def generate_lines(self, task: str, *, system: str, user: str) -> str:
        """Serve the recorded raw text of the lines contract for the slice named by the user prompt, byte-identical
        (no strip, no added newline: ingest.lines.parse_lines' truncation heuristic reads the trailing-newline state).
        为用户提示词点名的切片回放行式契约记录的原始文本，逐字节相同（不 strip、不补换行：ingest.lines.parse_lines
        的截断启发式依赖尾随换行状态）。

        Input / 输入：
            task: 'memory_extraction_lines'.
            task：'memory_extraction_lines'。

            system, user: The prompts built by the library.
            system、user：库构造的提示词。

        Output / 输出：
            The registered content string.
            登记的内容字符串。

        State effect / 状态影响：
            Counts lines_calls; an unknown slice or task counts lines_misses and raises LLMError(LLM_003).
            计数 lines_calls；未知切片或任务计数 lines_misses 并抛出 LLMError(LLM_003)。
        """
        self.lines_calls += 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            raw = self._call(task, messages)
        except LLMError:
            self.lines_misses += 1
            raise
        if isinstance(raw, dict):
            raw = raw.get("content") or ""
        return "" if raw is None else str(raw)


class GuardCapture(logging.Handler):
    """Count the verdict guard's DEBUG lines ('verdict X on ... downgraded to ADD: similarity ...') of the
    'hms_elite.ingest.pipeline' logger while a block runs, per verdict. The facade's add() returns only the rows, so
    IngestResult.downgraded is not reachable from here; the DEBUG line is the guard's only public trace.
    在块运行期间按裁决计数 'hms_elite.ingest.pipeline' 日志器里裁决守卫的 DEBUG 行（'verdict X on ... downgraded to
    ADD: similarity ...'）。门面的 add() 只返回行，IngestResult.downgraded 在此不可达；DEBUG 行是守卫唯一的公开痕迹。
    """

    LOGGER = "hms_elite.ingest.pipeline"
    MARK = " downgraded to ADD: similarity "

    def __init__(self) -> None:
        """Start at DEBUG level with empty counters.
        以 DEBUG 级别与空计数开始。
        """
        super().__init__(level=logging.DEBUG)
        self.by_verdict: Counter = Counter()
        self.total = 0
        self._previous_level: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Count a guard line under its verdict (the second word of the message); ignore everything else.
        把守卫行按其裁决（消息的第二个词）计数；忽略其他记录。

        Input / 输入：
            record: The log record.
            record：日志记录。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Increments by_verdict / total.
            递增 by_verdict / total。
        """
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - never let logging break the run
            return
        if not msg.startswith("verdict ") or self.MARK not in msg:
            return
        self.by_verdict[msg.split(" ", 2)[1]] += 1
        self.total += 1

    def __enter__(self) -> GuardCapture:
        lg = logging.getLogger(self.LOGGER)
        self._previous_level = lg.level
        if lg.getEffectiveLevel() > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
        lg.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        lg = logging.getLogger(self.LOGGER)
        lg.removeHandler(self)
        if self._previous_level is not None:
            lg.setLevel(self._previous_level)


def build_hms(
    workdir: Path, output: str, verdict_min_similarity: float | None = None, *, embedder: dict[str, Any] | None = None,
    id_factory: Callable[[str], str] | None = None,
) -> HMS:  # fmt: skip
    """HMS with the real run's store / embedder / single-call extraction, `extraction_output` set per arm, and the
    replay LLM spliced into every collaborator that holds the facade LLM.
    与真实 run 相同的存储 / 嵌入器 / 单次抽取的 HMS，按臂设置 `extraction_output`，并把回放 LLM 接入所有持有门面 LLM
    的协作对象。

    Input / 输入：
        workdir: Holds hms.db.
        workdir：存放 hms.db。

        output: 'full' | 'compact'.
        output：'full' | 'compact'。

        verdict_min_similarity: ingest.verdict_min_similarity (None = the library default; 0.0 = guard off).
        verdict_min_similarity：ingest.verdict_min_similarity（None = 库默认；0.0 = 关闭守卫）。

        embedder: Embedder envelope override (None = the real run's fastembed model; tests pass the hash provider).
        embedder：嵌入器信封覆盖（None = 真实 run 的 fastembed 模型；测试传 hash 提供者）。

        id_factory: Optional atom-id maker `prefix -> id` injected through StateStore (None = the library's random
            secrets.token_hex ids, as the real run); a counter-based factory makes two replays mint the same ids, so
            compile()'s (time_anchor, item_id) tie-break renders the same within-slot order in both.
        id_factory：可选的原子 id 生成器 `prefix -> id`，经 StateStore 注入（None = 库的随机 secrets.token_hex id，
            与真实 run 相同）；基于计数器的工厂使两次回放铸出相同 id，从而 compile() 的 (time_anchor, item_id) 平局
            规则在两次回放里渲染相同的槽内顺序。

    Output / 输出：
        The facade (hms.llm is the ReplayLLM).
        门面（hms.llm 即 ReplayLLM）。

    State effect / 状态影响：
        Creates the SQLite file; loads the fastembed model.
        创建 SQLite 文件；加载 fastembed 模型。
    """
    workdir.mkdir(parents=True, exist_ok=True)
    ingest: dict[str, Any] = {"extraction_mode": "single", "extraction_output": output}
    if verdict_min_similarity is not None:
        ingest["verdict_min_similarity"] = float(verdict_min_similarity)
    config = HMSConfig(
        store={"path": str(workdir / "hms.db")},
        llm={"provider": "scripted", "config": {}},
        embedder=embedder or {"provider": "fastembed", "config": {"model": EMBED_MODEL, "embedding_dims": EMBEDDING_DIMS}},
        ingest=ingest,
    )
    store = None
    if id_factory is not None:
        store = StateStore(
            config.store.path, id_factory=id_factory, busy_timeout_ms=config.store.busy_timeout_ms,
            journal_mode=config.store.journal_mode, checkpoint_on_close=config.store.checkpoint_on_close,
        )  # fmt: skip
    hms = HMS(config, store=store)
    replay = ReplayLLM(output)
    for holder in (hms, hms.ingestor, hms.scheduler, hms.compiler, hms.feedback_processor):
        if hasattr(holder, "llm"):
            holder.llm = replay
    assert hms.config.ingest.extraction_output == output
    return hms


# ---------------------------------------------------------------------------
# One arm over the timeline
# 一个臂沿时间线
# ---------------------------------------------------------------------------
def run_arm(
    arm: str,
    steps: list[Step],
    batches: list[Batch],
    recorded: Recorded,
    *,
    work_root: Path,
    out: Path,
    speaker_from_subject: bool,
    entities_from_mentions: bool,
    contract: str = "full",
    transforms: Transforms = Transforms(),
    verdict_min_similarity: float | None = None,
    context_hook: ContextHook | None = None,
    embedder: dict[str, Any] | None = None,
    id_factory: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Stream every step through one arm: retain that position's leaves through HMS.add with the replay LLM (local
    only), then HMS.search(top_k=20) with the real run's query builder; score hit@20 and MRR at batch granularity.
    Under contract 'full' the compact arm reduces each recorded full item (reduce_reply); under contract 'compact' the
    recorded items are served as recorded (prepare_compact_reply) and a non-object reply is skipped; `transforms`
    rewrites the compact items in either case (compact arm only — the full and lines arms always serve the reply
    verbatim; the lines arm serves it through ReplayLLM.generate_lines).
    让每个步骤流过一个臂：把该位置的叶子经 HMS.add 与回放 LLM retain（仅本地），再用真实 run 的查询构造做
    HMS.search(top_k=20)；按批次粒度计 hit@20 与 MRR。契约 'full' 下紧凑臂裁剪每条记录的完整条目（reduce_reply）；
    契约 'compact' 下记录条目按原样回放（prepare_compact_reply），根非对象的回复跳过；两种情况下 `transforms` 都改写
    紧凑条目（仅紧凑臂——完整臂与行式臂始终原样回放；行式臂经 ReplayLLM.generate_lines 回放）。

    Input / 输入：
        arm: 'full' | 'compact' | 'lines'.
        arm：'full' | 'compact' | 'lines'。

        steps, batches: The shared timeline and batch plan.
        steps、batches：共享的时间线与批次计划。

        recorded: The recorded replies.
        recorded：记录的回复。

        work_root: Parent of the arm's store directory.
        work_root：臂存储目录的父目录。

        out: Result directory (retain_<arm>.jsonl is written here).
        out：结果目录（此处写 retain_<arm>.jsonl）。

        speaker_from_subject, entities_from_mentions: Compact reduction options (contract 'full' only).
        speaker_from_subject、entities_from_mentions：紧凑裁剪选项（仅契约 'full'）。

        contract: 'full' | 'compact' — the contract the recorded replies were written under.
        contract：'full' | 'compact'——记录回复所遵循的契约。

        transforms: Per-item rewrites for the compact arm.
        transforms：紧凑臂的逐条改写。

        verdict_min_similarity: ingest.verdict_min_similarity for this arm's HMS (None = library default).
        verdict_min_similarity：此臂 HMS 的 ingest.verdict_min_similarity（None = 库默认）。

        context_hook: Optional per-question callback (ContextHook) run after the baseline search, before the record
            is appended; the reader-side A/B captures its context variants here.
        context_hook：可选的逐题回调（ContextHook），在基线检索之后、记录追加之前运行；阅读端 A/B 在此捕获其上下文变体。

        embedder: Embedder envelope override for build_hms (None = fastembed).
        embedder：传给 build_hms 的嵌入器信封覆盖（None = fastembed）。

        id_factory: Atom-id maker forwarded to build_hms (None = random ids).
        id_factory：转交 build_hms 的原子 id 生成器（None = 随机 id）。

    Output / 输出：
        {'records', 'retain', 'replay', 'agreement', 'atoms', 'entity_axis'} — entity_axis = one row per live atom
        (id, batch_id, text, sorted entity_ids).
        {'records', 'retain', 'replay', 'agreement', 'atoms', 'entity_axis'}——entity_axis = 每个存活原子一行
        （id、batch_id、text、排序后的 entity_ids）。

    State effect / 状态影响：
        Builds and closes the arm's store; writes retain_<arm>.jsonl and entity_axis_<arm>.jsonl.
        建立并关闭臂的存储；写 retain_<arm>.jsonl 与 entity_axis_<arm>.jsonl。
    """
    t_init = time.perf_counter()
    hms = build_hms(work_root / arm, arm, verdict_min_similarity, embedder=embedder, id_factory=id_factory)
    replay: ReplayLLM = hms.llm  # type: ignore[assignment]
    init_ms = (time.perf_counter() - t_init) * 1000.0
    agreement = Agreement()
    counters: Counter = Counter()
    by_position: dict[int, list[Batch]] = defaultdict(list)
    for b in batches:
        by_position[b.position].append(b)
    batch_of_window = {w.window_id: b.batch_id for b in batches for w in b.windows}
    windows_stored_by_batch: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    retain_log: list[dict[str, Any]] = []
    retain_fh = (out / f"retain_{arm}.jsonl").open("w", encoding="utf-8")
    try:
        for step in steps:
            q = step.question
            query_iso = clock_to_iso(q.query_day, q.query_cs)
            t_pos = time.perf_counter()
            pos_added = 0
            pos_leaves = pos_missing = 0
            # Per-position event histogram and the last window of the last leaf that changed the store (ADD or
            # UPDATE); the reader-side hook keys its session-card memo on these rather than on len(rows), which also
            # counts NONE events.
            # 本位置的事件直方图与最后一个改变了库（ADD 或 UPDATE）的叶子的最后一个窗口；阅读端钩子据此而非 len(rows)
            # （它也把 NONE 事件计入）决定会话卡片记忆化。
            pos_events: Counter = Counter()
            pos_last_stored_window: str | None = None
            for batch in by_position.get(step.position, []):
                leaves = resolve_leaves(batch, recorded)
                stored = 0
                subs: list[dict[str, Any]] = []
                for leaf in leaves:
                    windows = batch.windows[leaf.lo : leaf.hi]
                    sub: dict[str, Any] = {"split": leaf.split, "lo": leaf.lo, "hi": leaf.hi, "windows": len(windows)}
                    if leaf.row is None:
                        sub.update({"outcome": "missing", "added": 0})
                        subs.append(sub)
                        pos_missing += 1
                        continue
                    message_date = date_of(windows[0].start_iso)
                    content: str | None = str(leaf.row.get("content") or "")
                    if arm == "compact" and contract == "compact":
                        content = prepare_compact_reply(content, transforms=transforms, counters=counters)
                    elif arm == "compact":
                        content = reduce_reply(
                            content, message_date=message_date, extractor=hms.entity_extractor, agreement=agreement,
                            speaker_from_subject=speaker_from_subject, entities_from_mentions=entities_from_mentions, counters=counters,
                            transforms=transforms,
                        )  # fmt: skip
                    if content is None:
                        # Non-object compact reply: nothing to serve; the real run stored nothing for it either.
                        # 根非对象的紧凑回复：无可回放；真实 run 对它同样什么都没存。
                        sub.update({"outcome": "invalid_json_skipped", "added": 0, "recorded_completion_tokens": leaf.row.get("completion_tokens")})
                        subs.append(sub)
                        pos_leaves += 1
                        continue
                    replay.register([w.text for w in windows], content, (batch.batch_id, leaf.split))
                    replay.expect = (batch.batch_id, leaf.split)
                    t_add = time.perf_counter()
                    with LogCapture() as logs, GuardCapture() as guard:
                        try:
                            result = hms.add(
                                [{"role": "user", "content": w.text} for w in windows], user_id=BANK,
                                metadata=sub_metadata(batch, windows, leaf.split), timestamp=windows[0].start_iso, infer=True,
                            )  # fmt: skip
                            rows = result.get("results", []) if isinstance(result, dict) else []
                            error = None
                        except Exception as exc:  # noqa: BLE001 - kept in the retain log
                            rows, error = [], f"{type(exc).__name__}: {str(exc)[:300]}"
                    add_ms = (time.perf_counter() - t_add) * 1000.0
                    events: Counter = Counter(str((r or {}).get("event", "?")) for r in rows if isinstance(r, dict))
                    skipped = sum(1 for w in logs.messages if " skipped: " in w or "invalid event_idx" in w)
                    for verdict, n in guard.by_verdict.items():
                        counters[f"guard_downgraded_from_{verdict}"] += n
                    if guard.total:
                        counters["guard_downgraded_total"] += guard.total
                    sub.update(
                        {
                            "outcome": "api_error" if error else ("ok" if rows else "empty"), "added": len(rows), "events": dict(events),
                            "error": error, "items_skipped": skipped, "warnings": len(logs.messages), "add_ms": round(add_ms, 3),
                            "downgraded": guard.total, "downgraded_by_verdict": dict(guard.by_verdict),
                            "recorded_completion_tokens": leaf.row.get("completion_tokens"),
                        }
                    )  # fmt: skip
                    subs.append(sub)
                    pos_leaves += 1
                    pos_added += len(rows)
                    pos_events.update(events)
                    if events.get("ADD", 0) + events.get("UPDATE", 0) > 0:
                        pos_last_stored_window = windows[-1].window_id
                    if rows:
                        stored += len(windows)
                windows_stored_by_batch[batch.batch_id] = stored
                row = {
                    "batch_id": batch.batch_id, "position": step.position, "question_id": q.id, "windows": len(batch.windows),
                    "windows_stored": stored, "leaves": len(leaves), "leaves_missing": sum(1 for s in subs if s["outcome"] == "missing"),
                    "leaves_invalid_json_skipped": sum(1 for s in subs if s["outcome"] == "invalid_json_skipped"),
                    "added": sum(s["added"] for s in subs), "subs": subs,
                }  # fmt: skip
                retain_log.append(row)
                retain_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            retain_ms = (time.perf_counter() - t_pos) * 1000.0
            t_s = time.perf_counter()
            result = hms.search(q.question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, reference_date=query_iso)
            search_ms = (time.perf_counter() - t_s) * 1000.0
            rows = [
                {"batch_id": ((r.get("metadata") or {}).get("batch_id")), "text": r.get("memory", ""), "score": float(r.get("score", 0.0))}
                for r in result.get("results", [])
            ]
            gold_batches = {batch_of_window[w] for w in step.gold_retained_ids if w in batch_of_window}
            returned_batches: list[str] = []
            for r in rows:
                if r["batch_id"] and r["batch_id"] not in returned_batches:
                    returned_batches.append(r["batch_id"])
            rank = next((i + 1 for i, b in enumerate(returned_batches) if b in gold_batches), None)
            row_rank = next((i + 1 for i, r in enumerate(rows) if r["batch_id"] in gold_batches), None)
            context = rows_to_context(rows)
            rec: dict[str, Any] = {
                "arm": arm, "id": q.id, "order": q.order, "position": step.position, "type": q.type,
                "query_day": q.query_day, "query_time": q.query_hhmmsscc,
                "retained_count": step.retained_count, "new_windows": len(step.new_windows),
                "retain_batches": len(by_position.get(step.position, [])), "retain_leaves": pos_leaves,
                "retain_leaves_missing": pos_missing, "retain_added": pos_added,
                "retain_events": dict(pos_events), "retain_last_stored_window": pos_last_stored_window,
                "excluded_future": step.excluded_future, "gold_window_count": len(step.gold_ids),
                "gold_retained_count": len(step.gold_retained_ids), "gold_batches": sorted(gold_batches),
                "gold_batches_stored": sorted(b for b in gold_batches if windows_stored_by_batch.get(b, 0) > 0),
                "returned_count": len(rows), "returned_batches": returned_batches,
                "top_scores": [round(r["score"], 4) for r in rows[:5]],
                "hit_batch": rank is not None, "hit_rank": rank, "rr": (1.0 / rank) if rank else 0.0,
                "hit_row_rank": row_rank, "rr_row": (1.0 / row_rank) if row_rank else 0.0,
                "context_chars": len(context), "context": context,
                "timing_ms": {"retain_local": round(retain_ms, 3), "search_local": round(search_ms, 3)},
                "gold": q.answer,
            }  # fmt: skip
            if context_hook is not None:
                context_hook(hms, step, query_iso, list(result.get("results", [])), rec)
            records.append(rec)
            if len(records) % 25 == 0 or len(records) == len(steps):
                hits = sum(1 for r in records if r["hit_batch"])
                print(f"  [{arm}] {len(records)}/{len(steps)} hit@20={hits} atoms_added={sum(r['added'] for r in retain_log)} "
                      f"leaves_missing={sum(r['leaves_missing'] for r in retain_log)} replay_calls={replay.calls}", flush=True)
    finally:
        retain_fh.close()
    t_c = time.perf_counter()
    live = hms.get_all(filters={"user_id": BANK}, top_k=1_000_000).get("results", [])
    atoms = len(live)
    entity_axis = [
        {
            "id": r.get("id"), "batch_id": ((r.get("metadata") or {}).get("batch_id")), "text": r.get("memory", ""),
            "entity_ids": sorted(str(e) for e in ((r.get("hms") or {}).get("entity_ids") or [])),
        }
        for r in live
    ]  # fmt: skip
    with (out / f"entity_axis_{arm}.jsonl").open("w", encoding="utf-8") as fh:
        for row in entity_axis:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    hms.close()
    close_ms = (time.perf_counter() - t_c) * 1000.0
    return {
        "records": records,
        "retain": retain_log,
        "replay": {
            "calls": replay.calls, "mismatches": replay.mismatches, "tasks": dict(replay.tasks), "slices_registered": len(replay.table),
            "lines_calls": replay.lines_calls, "lines_misses": replay.lines_misses,
            "reduction_counters": dict(counters), "init_ms": round(init_ms, 3), "close_ms": round(close_ms, 3),
            "verdict_min_similarity": float(hms.config.ingest.verdict_min_similarity),
            "downgraded_total": sum(s.get("downgraded", 0) for r in retain_log for s in r["subs"]),
        },  # fmt: skip
        "agreement": agreement.summary() if (arm == "compact" and contract == "full") else None,
        "atoms": atoms,
        "entity_axis": entity_axis,
    }


# ---------------------------------------------------------------------------
# Answer pass
# 答题阶段
# ---------------------------------------------------------------------------
def project_cost(records: list[dict[str, Any]], questions: dict[str, Question]) -> float:
    """Pre-flight USD projection of the answer pass at the current rate tier (chars / PROJECTION_CHARS_PER_TOKEN
    prompt tokens, all cache-miss, ANSWER_MAX_TOKENS output tokens per request).
    按当前档位预估答题阶段的美元费用（提示词按 chars / PROJECTION_CHARS_PER_TOKEN 计词元、全部按缓存未命中、
    每次请求 ANSWER_MAX_TOKENS 个输出词元）。

    Input / 输入：
        records: The per-question records of every arm to be answered.
        records：待答题的所有臂的逐题记录。

        questions: id → Question.
        questions：id → Question。

    Output / 输出：
        USD.
        美元。

    State effect / 状态影响：
        None.
        无。
    """
    price = PRICES_USD_PER_M[rate_tier(utc_now())]
    total = 0.0
    for rec in records:
        prompt = build_question_prompt(questions[rec["id"]].raw) + "\n\nMemories:\n" + (rec["context"] or "(none)")
        tokens = len(prompt) / PROJECTION_CHARS_PER_TOKEN
        total += (tokens * price["cache_miss"] + ANSWER_MAX_TOKENS * price["output"]) / 1_000_000.0
    return total


def answer_pass(
    arms: dict[str, dict[str, Any]], questions: dict[str, Question], meter: Meter, *, max_requests: int, cost_cap: float
) -> dict[str, Any]:
    """Ask deepseek-flash once per (arm, question) with real_arms.answer (their prompt, temperature 0, max_tokens 8);
    stops before the first call when the projection exceeds the cap, and mid-way when the running ledger does.
    对每个（臂，题目）用 real_arms.answer 向 deepseek-flash 提问一次（对方提示词、temperature 0、max_tokens 8）；
    预估超过上限则在第一次调用前停止，累计台账超过上限则中途停止。

    Input / 输入：
        arms: run_arm outputs by arm.
        arms：按臂的 run_arm 输出。

        questions: id → Question.
        questions：id → Question。

        meter: The ledger.
        meter：台账。

        max_requests: Hard cap on requests (attempt 0 calls).
        max_requests：请求数硬上限（attempt 0 的调用）。

        cost_cap: USD cap.
        cost_cap：美元上限。

    Output / 输出：
        {'projected_usd', 'stopped', 'reason', 'requests', 'cost_usd'}.
        {'projected_usd', 'stopped', 'reason', 'requests', 'cost_usd'}。

    State effect / 状态影响：
        Network calls; mutates the records in place (pred / correct / usage).
        网络调用；原地修改记录（pred / correct / usage）。
    """
    plan = [(arm, rec) for arm in arms for rec in arms[arm]["records"]][:max_requests]
    projected = project_cost([rec for _arm, rec in plan], questions)
    status: dict[str, Any] = {"projected_usd": round(projected, 6), "stopped": False, "reason": None, "requests": 0, "cost_usd": 0.0,
                              "tier_at_start": rate_tier(utc_now()), "planned_requests": len(plan)}  # fmt: skip
    if projected > cost_cap:
        status.update({"stopped": True, "reason": f"projected {projected:.4f} USD exceeds cap {cost_cap} USD; no request sent"})
        return status
    load_api_key(ENV_PATH)
    client = DeepSeekClient(meter)
    for i, (arm, rec) in enumerate(plan):
        spent = sum(c.cost_usd or 0.0 for c in meter.calls)
        if spent > cost_cap:
            status.update({"stopped": True, "reason": f"running cost {spent:.4f} USD exceeds cap {cost_cap} USD after {i} requests"})
            break
        res = answer(client, questions[rec["id"]], rec["context"], arm=ANSWER_LABEL[arm])
        last = res.calls[-1] if res.calls else None
        rec.update(
            {
                "pred": res.letter, "raw_output": res.raw, "correct": res.letter == rec["gold"], "answer_status": res.status,
                "answer_attempts": len(res.calls), "prompt_chars": res.prompt_chars,
                "answer_usage": {
                    "prompt_tokens": last.prompt_tokens if last else None, "completion_tokens": last.completion_tokens if last else None,
                    "cache_hit_tokens": last.cache_hit_tokens if last else None, "cache_miss_tokens": last.cache_miss_tokens if last else None,
                    "rate_tier": last.rate_tier if last else None,
                    "cost_usd_all_attempts": round(sum(c.cost_usd or 0.0 for c in res.calls), 8),
                    "unpriced_attempts": sum(1 for c in res.calls if c.cost_usd is None),
                },
                "timing_ms_answer": {"model": round(res.model_ms, 3), "wall": round(res.wall_ms, 3)},
            }
        )  # fmt: skip
        status["requests"] += 1
        if status["requests"] % 50 == 0:
            print(f"  [answer] {status['requests']}/{len(plan)} cost_usd={sum(c.cost_usd or 0.0 for c in meter.calls):.5f}", flush=True)
    status["cost_usd"] = round(sum(c.cost_usd or 0.0 for c in meter.calls), 6)
    return status


# ---------------------------------------------------------------------------
# Summary
# 汇总
# ---------------------------------------------------------------------------
def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def arm_summary(arm: str, res: dict[str, Any], meter: Meter) -> dict[str, Any]:
    """Retrieval / answer / timing figures of one arm over all 204 questions and the scored subsets.
    一个臂在全部 204 题及计分子集上的检索 / 答题 / 耗时数字。

    Input / 输入：
        arm: Arm name.
        arm：臂名。

        res: run_arm output (records answered in place).
        res：run_arm 输出（记录已原地答题）。

        meter: The ledger (answer cost).
        meter：台账（答题费用）。

    Output / 输出：
        The summary dict.
        汇总字典。

    State effect / 状态影响：
        None.
        无。
    """
    recs = res["records"]
    scored = [r for r in recs if not r["excluded_future"]]
    gold_stored = [r for r in scored if r["gold_batches_stored"]]
    answered = [r for r in recs if "pred" in r]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_type[r["type"]].append(r)
    retain_ms = [r["timing_ms"]["retain_local"] for r in recs if r["retain_leaves"] > 0]
    stage = meter.stage_summary(meter.calls_for(ANSWER_LABEL[arm], "answer"))
    return {
        "n": len(recs),
        "n_scored": len(scored),
        "n_gold_stored": len(gold_stored),
        "hit20": round(sum(1 for r in scored if r["hit_batch"]) / len(scored), 4) if scored else None,
        "hit20_gold_stored_subset": round(sum(1 for r in gold_stored if r["hit_batch"]) / len(gold_stored), 4) if gold_stored else None,
        "hit20_all204": round(sum(1 for r in recs if r["hit_batch"]) / len(recs), 4) if recs else None,
        "hits": sum(1 for r in scored if r["hit_batch"]),
        "mrr": round(sum(r["rr"] for r in scored) / len(scored), 4) if scored else None,
        "mrr_row_rank": round(sum(r["rr_row"] for r in scored) / len(scored), 4) if scored else None,
        "mrr_gold_stored_subset": round(sum(r["rr"] for r in gold_stored) / len(gold_stored), 4) if gold_stored else None,
        "by_type_hit20": {t: round(sum(1 for r in v if r["hit_batch"]) / len(v), 4) for t, v in sorted(by_type.items())},
        "by_type_n": {t: len(v) for t, v in sorted(by_type.items())},
        "returned_count_median": _median([float(r["returned_count"]) for r in recs]),
        "context_chars_median": _median([float(r["context_chars"]) for r in recs]),
        "answered_n": len(answered),
        "answer_correct": sum(1 for r in answered if r.get("correct")),
        "answer_accuracy": round(sum(1 for r in answered if r.get("correct")) / len(answered), 4) if answered else None,
        "answer_accuracy_all204_denominator": round(sum(1 for r in answered if r.get("correct")) / len(recs), 4) if recs else None,
        "answer_valid_n": sum(1 for r in answered if r.get("pred")),
        "answer_cost_usd": stage["cost_usd"],
        "answer_tokens": stage["tokens"],
        "answer_attempts": stage["attempts"],
        "answer_tier_attempts": stage["tier_attempts"],
        "atoms": res["atoms"],
        "atoms_with_entities": sum(1 for r in res["entity_axis"] if r["entity_ids"]),
        "entity_ids_per_atom_mean": round(sum(len(r["entity_ids"]) for r in res["entity_axis"]) / len(res["entity_axis"]), 4) if res["entity_axis"] else None,
        "atoms_added_sum": sum(r["added"] for r in res["retain"]),
        "items_skipped": sum(s.get("items_skipped", 0) for r in res["retain"] for s in r["subs"]),
        "leaves_replayed": sum(r["leaves"] - r["leaves_missing"] - r.get("leaves_invalid_json_skipped", 0) for r in res["retain"]),
        "leaves_missing": sum(r["leaves_missing"] for r in res["retain"]),
        "leaves_invalid_json_skipped": sum(r.get("leaves_invalid_json_skipped", 0) for r in res["retain"]),
        "retain_local_ms_median": _median(retain_ms),
        "retain_local_ms_total": round(sum(retain_ms), 1),
        "search_ms_median": _median([r["timing_ms"]["search_local"] for r in recs]),
        "replay": res["replay"],
    }


def paired_diff(full: list[dict[str, Any]], compact: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-question paired comparison of the two arms (same question order).
    两个臂的逐题配对比较（题目顺序相同）。

    Input / 输入：
        full, compact: The two record lists.
        full、compact：两份记录列表。

    Output / 输出：
        Counts of hit / answer agreement and disagreement.
        命中 / 答题一致与不一致的计数。

    State effect / 状态影响：
        None.
        无。
    """
    out: Counter = Counter()
    for a, b in zip(full, compact, strict=True):
        assert a["id"] == b["id"]
        if a["excluded_future"]:
            continue
        out["hit_both"] += a["hit_batch"] and b["hit_batch"]
        out["hit_full_only"] += a["hit_batch"] and not b["hit_batch"]
        out["hit_compact_only"] += b["hit_batch"] and not a["hit_batch"]
        out["hit_neither"] += not a["hit_batch"] and not b["hit_batch"]
        out["same_returned_batches"] += a["returned_batches"] == b["returned_batches"]
        if "pred" in a and "pred" in b:
            out["answered_pairs"] += 1
            out["correct_both"] += bool(a["correct"] and b["correct"])
            out["correct_full_only"] += bool(a["correct"] and not b["correct"])
            out["correct_compact_only"] += bool(b["correct"] and not a["correct"])
            out["same_pred"] += a.get("pred") == b.get("pred")
    return dict(out)


def entity_axis_diff(full: list[dict[str, Any]], compact: list[dict[str, Any]], *, examples: int = 25) -> dict[str, Any]:
    """Pair the live atoms of the two arms by (batch_id, text) and compare their entity_ids sets: identical, compact a
    superset of full (the §6.5 'never narrower' invariant), or narrowed (compact ⊉ full — each such pair is listed).
    Within one (batch_id, text) key the atoms of each arm are sorted by their entity_ids and zipped; leftovers are
    counted as unpaired.
    按 (batch_id, text) 配对两个臂的存活原子并比较其 entity_ids 集合：相同、紧凑为完整的超集（§6.5 的"绝不更窄"
    不变量）、或收窄（紧凑 ⊉ 完整——每个这样的配对都列出）。同一 (batch_id, text) 键内各臂的原子按 entity_ids 排序后
    逐位配对；余下的计为未配对。

    Input / 输入：
        full, compact: run_arm's entity_axis rows.
        full、compact：run_arm 的 entity_axis 行。

        examples: How many narrowed / widened pairs to list.
        examples：列出多少个收窄 / 加宽的配对。

    Output / 输出：
        Counts, fractions and example lists.
        计数、比例与示例列表。

    State effect / 状态影响：
        None.
        无。
    """
    def group(rows: list[dict[str, Any]]) -> dict[tuple[Any, str], list[list[str]]]:
        g: dict[tuple[Any, str], list[list[str]]] = defaultdict(list)
        for r in rows:
            g[(r["batch_id"], r["text"])].append(list(r["entity_ids"]))
        return {k: sorted(v) for k, v in g.items()}

    gf, gc = group(full), group(compact)
    out: Counter = Counter()
    narrowed: list[dict[str, Any]] = []
    widened: list[dict[str, Any]] = []
    extra_ids: Counter = Counter()
    for key in set(gf) | set(gc):
        a, b = gf.get(key, []), gc.get(key, [])
        out["unpaired_full"] += max(0, len(a) - len(b))
        out["unpaired_compact"] += max(0, len(b) - len(a))
        for fa, ca in zip(a, b, strict=False):
            sf, sc = set(fa), set(ca)
            out["paired"] += 1
            out["identical"] += sf == sc
            out["compact_superset_or_equal"] += sf <= sc
            out["compact_strict_superset"] += sf < sc
            out["narrowed"] += not sf <= sc
            out["paired_full_nonempty"] += bool(sf)
            out["paired_compact_nonempty"] += bool(sc)
            for e in sc - sf:
                extra_ids[e] += 1
            row = {"batch_id": key[0], "text": key[1], "full": fa, "compact": ca}
            if not sf <= sc and len(narrowed) < examples:
                narrowed.append({**row, "missing_in_compact": sorted(sf - sc)})
            elif sf < sc and len(widened) < examples:
                widened.append({**row, "extra_in_compact": sorted(sc - sf)})
    paired = out["paired"] or 1
    return {
        "pair_key": "(batch_id, text)",
        "paired": out["paired"],
        "unpaired_full": out["unpaired_full"],
        "unpaired_compact": out["unpaired_compact"],
        "identical": out["identical"],
        "identical_fraction": round(out["identical"] / paired, 4),
        "compact_superset_or_equal": out["compact_superset_or_equal"],
        "compact_superset_or_equal_fraction": round(out["compact_superset_or_equal"] / paired, 4),
        "compact_strict_superset": out["compact_strict_superset"],
        "narrowed": out["narrowed"],
        "paired_full_nonempty": out["paired_full_nonempty"],
        "paired_compact_nonempty": out["paired_compact_nonempty"],
        "extra_ids_in_compact_top40": dict(extra_ids.most_common(40)),
        "narrowed_examples": narrowed,
        "widened_examples": widened,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render summary.md.
    渲染 summary.md。

    Input / 输入：
        summary: The summary dict.
        summary：汇总字典。

    Output / 输出：
        Markdown text.
        Markdown 文本。

    State effect / 状态影响：
        None.
        无。
    """
    a = summary["arms"]
    arms_present = [arm for arm in ARMS if arm in a]

    def pct(v: Any) -> str:
        return "-" if v is None else f"{100 * v:.1f}%"

    variant = summary.get("variant") or {}
    lines = [
        "# Replay A/B: full vs compact extraction contract on identical recorded replies",
        "# 回放 A/B：同一批记录回复上的完整契约 vs 紧凑契约",
        "",
        f"run: `{summary['run_dir']}` (read once at {summary['started_utc']}; extract_raw lines={summary['recorded']['lines']}, "
        f"truncated last line={summary['recorded']['lines_truncated']}, batches with any reply={summary['recorded']['batches_seen']} of {summary['batches']})",
        "",
        f"replies: run=`{summary.get('replies_run_dir', summary['run_dir'])}` contract={variant.get('replies_contract', 'full')} "
        f"arms={arms_present} transforms={json.dumps(variant.get('transforms') or {})}",
        "",
        f"replies replayed (leaves with a usable reply) = {summary['replies_replayed']}; replies missing = {summary['replies_missing']} "
        f"(truncated-unresolved leaves = {summary['missing_breakdown']['truncated_unresolved']}, slices never recorded = {summary['missing_breakdown']['never_recorded']}; "
        f"batches with no recorded row = {summary['missing_breakdown']['batches_unrecorded']})",
        "",
        "## Retrieval (batch granularity, k = 20) / 检索（批次粒度，k = 20）",
        "",
        "| arm | atoms | atoms with entity axis | entity_ids / atom | hit@20 (scored n) | hits | hit@20 gold-stored subset (n) | MRR | MRR row-rank | leaves skipped (non-object reply) | retain local ms median | search ms median |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms_present:
        s = a[arm]
        lines.append(
            f"| {arm} | {s['atoms']} | {s['atoms_with_entities']} | {s['entity_ids_per_atom_mean']} | {pct(s['hit20'])} ({s['n_scored']}) | {s['hits']} | "
            f"{pct(s['hit20_gold_stored_subset'])} ({s['n_gold_stored']}) | "
            f"{s['mrr']} | {s['mrr_row_rank']} | {s.get('leaves_invalid_json_skipped', 0)} | {s['retain_local_ms_median']} | {s['search_ms_median']} |"
        )
    lines += ["", "### hit@20 by question type / 按题型", "", "| type | n | " + " | ".join(arms_present) + " |",
              "|---|---|" + "---|" * len(arms_present)]  # fmt: skip
    first = a[arms_present[0]]
    for t in sorted(first["by_type_hit20"]):
        cells = " | ".join(pct(a[arm]["by_type_hit20"].get(t)) for arm in arms_present)
        lines.append(f"| {t} | {first['by_type_n'][t]} | {cells} |")
    lines += ["", "## Answer pass (deepseek-flash, temperature 0, max_tokens 8) / 答题", "",
              f"status: {json.dumps(summary['answer_pass'], ensure_ascii=False)}", "",
              "| arm | answered | correct | accuracy | valid outputs | cost USD | prompt tokens | attempts (peak/offpeak) |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for arm in arms_present:
        s = a[arm]
        lines.append(
            f"| {arm} | {s['answered_n']} | {s['answer_correct']} | {pct(s['answer_accuracy'])} | {s['answer_valid_n']} | {s['answer_cost_usd']} | "
            f"{s['answer_tokens']['prompt']} | {s['answer_attempts']} ({s['answer_tier_attempts']['peak']}/{s['answer_tier_attempts']['offpeak']}) |"
        )
    if summary.get("paired") is not None:
        lines += ["", f"paired: {json.dumps(summary['paired'], ensure_ascii=False)}", ""]
    ea = summary.get("entity_axis")
    if ea is not None:
        lines += ["## Entity axis: live atoms paired by (batch_id, text) / 实体轴：按 (batch_id, text) 配对的存活原子", "",
                  f"paired = {ea['paired']} (unpaired full = {ea['unpaired_full']}, unpaired compact = {ea['unpaired_compact']}); "
                  f"identical entity_ids = {ea['identical']} ({100 * ea['identical_fraction']:.1f}%); compact ⊇ full = {ea['compact_superset_or_equal']} "
                  f"({100 * ea['compact_superset_or_equal_fraction']:.1f}%), of which strict superset = {ea['compact_strict_superset']}; "
                  f"narrowed (compact ⊉ full) = {ea['narrowed']}; paired atoms with a non-empty axis: full = {ea['paired_full_nonempty']}, "
                  f"compact = {ea['paired_compact_nonempty']}", "",
                  "extra ids in compact (top 40): " + json.dumps(ea["extra_ids_in_compact_top40"], ensure_ascii=False), "",
                  "narrowed examples: " + json.dumps(ea["narrowed_examples"], ensure_ascii=False), "",
                  "widened examples: " + json.dumps(ea["widened_examples"], ensure_ascii=False), ""]  # fmt: skip
    ag = summary.get("derived_vs_llm")
    if ag is not None:
        lines += ["## Derived vs LLM-emitted (compact arm, every replayed observation) / 本地推导 vs LLM 输出", "",
                  f"n = {ag['n']}; speaker explicit = {ag['counts']['speaker_explicit']}, speaker taken from subject = {ag['counts']['speaker_from_subject']}", "",
                  "| field | agreement | mean Jaccard | both empty | LLM empty | derived empty |", "|---|---|---|---|---|---|"]  # fmt: skip
        for f in AGREEMENT_FIELDS:
            lines.append(
                f"| {f} | {pct(ag['agreement'][f])} | {ag['mean_jaccard'].get(f, '-')} | {ag['counts']['both_empty'].get(f, '-')} | "
                f"{ag['counts']['llm_empty'].get(f, '-')} | {ag['counts']['derived_empty'].get(f, '-')} |"
            )
        lines += ["", "sentiment pairs (LLM->derived): " + json.dumps(ag["pairs"]["sentiment"]),
                  "relation pairs (LLM->derived): " + json.dumps(ag["pairs"]["relation"]),
                  "predicate pairs top 40 (LLM->derived): " + json.dumps(ag["pairs"]["predicate_top40"]), ""]  # fmt: skip
    for arm in arms_present:
        lines += [f"reply counters ({arm}): " + json.dumps(a[arm]["replay"].get("reduction_counters") or {}, ensure_ascii=False), ""]
        lines += [f"verdict guard ({arm}): verdict_min_similarity = {a[arm]['replay'].get('verdict_min_similarity')}; "
                  f"verdicts downgraded to ADD = {a[arm]['replay'].get('downgraded_total')}", ""]  # fmt: skip
    lines += ["## Method / 方法", "", summary["method"], "", "## Caveats / 注意", ""]
    lines += [f"- {c}" for c in summary["caveats"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entry.
    命令行入口。

    Input / 输入：
        argv: Arguments (None -> sys.argv[1:]).
        argv：参数（None -> sys.argv[1:]）。

    Output / 输出：
        Exit code 0; 2 when the batch plan does not match the run's batches.json.
        退出码 0；批次计划与 run 的 batches.json 不一致时为 2。

    State effect / 状态影响：
        Writes the result directory; the answer pass makes network calls.
        写结果目录；答题阶段有网络调用。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=204)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--no-answer", action="store_true", help="skip the API answer pass / 跳过 API 答题阶段")
    parser.add_argument("--max-answer-requests", type=int, default=MAX_ANSWER_REQUESTS)
    parser.add_argument("--cost-cap-usd", type=float, default=COST_CAP_USD)
    parser.add_argument("--steps", type=int, default=0, help="smoke: only the first N timeline steps (0 = all) / 冒烟：只跑前 N 步（0 = 全部）")
    parser.add_argument("--no-speaker-from-subject", action="store_true",
                        help="compact arm: never promote a named subject to speaker / 紧凑臂：不把具名 subject 提升为 speaker")  # fmt: skip
    parser.add_argument("--no-entities-from-mentions", action="store_true",
                        help="compact arm: do not carry the recorded entity_mentions as the `entities` hint / 紧凑臂：不把记录的 entity_mentions 作为 `entities` 提示")  # fmt: skip
    # ---- variant mode / 变体模式
    parser.add_argument("--replies-run", type=Path, default=None,
                        help="run directory whose extract_raw_elite.jsonl is replayed (default: --run); its batches.json must equal --run's / 回放其 extract_raw_elite.jsonl 的 run 目录（默认 --run）；其 batches.json 须与 --run 的相同")  # fmt: skip
    parser.add_argument("--replies-contract", choices=REPLIES_CONTRACTS, default="full",
                        help="how the recorded observation items are shaped: full (r3), compact (a compact run; only the compact arm is meaningful) or lines (a lines run; only the lines arm) / 记录的观察条目形状：full（r3）、compact（紧凑 run；只有紧凑臂有意义）或 lines（行式 run；只有行式臂）")  # fmt: skip
    parser.add_argument("--arms", default=None,
                        help="comma list of arms to run (default: full,compact for contract full; compact for contract compact) / 要运行的臂（默认：契约 full 为 full,compact；契约 compact 为 compact）")  # fmt: skip
    parser.add_argument("--force-type", default=None, metavar="TYPE",
                        help="compact arm: every observation type := TYPE (e.g. fact) / 紧凑臂：所有观察类型 := TYPE（如 fact）")  # fmt: skip
    parser.add_argument("--drop-entities", action="store_true", help="compact arm: entities := [] / 紧凑臂：entities := []")
    parser.add_argument("--drop-when", action="store_true", help="compact arm: remove when and valid_to / 紧凑臂：去掉 when 与 valid_to")
    parser.add_argument("--keep-only-text", action="store_true",
                        help="compact arm: keep id/event_idx/text/type only / 紧凑臂：只保留 id/event_idx/text/type")  # fmt: skip
    parser.add_argument("--force-add", action="store_true",
                        help="compact arm: every recorded decision := ADD (no NONE / REVISE / CONFLICT) / 紧凑臂：所有记录的决策 := ADD")  # fmt: skip
    parser.add_argument("--verdict-min-similarity", type=float, default=None, metavar="FLOAT",
                        help="ingest.verdict_min_similarity for both arms (0.0 = guard off; omitted = library default) / "
                             "两臂的 ingest.verdict_min_similarity（0.0 = 关闭守卫；省略 = 库默认）")  # fmt: skip
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    work_root = out / "work"
    replies_run: Path = args.replies_run if args.replies_run is not None else args.run
    contract: str = args.replies_contract
    transforms = Transforms(force_type=args.force_type, drop_entities=args.drop_entities, drop_when=args.drop_when,
                            keep_only_text=args.keep_only_text, force_add=args.force_add)  # fmt: skip
    default_arms = {"compact": "compact", "lines": "lines"}.get(contract, "full,compact")
    arms_to_run = [s.strip() for s in (args.arms or default_arms).split(",") if s.strip()]
    if any(arm not in ARMS for arm in arms_to_run) or not arms_to_run:
        print(f"unknown arm in --arms {args.arms!r}; choose from {ARMS} / --arms 含未知臂")
        return 2
    if contract == "compact" and "full" in arms_to_run:
        print("contract compact has no full-contract fields to serve to the full arm; use --arms compact / 紧凑契约无完整字段可供完整臂；请用 --arms compact")
        return 2
    if (contract == "lines") != (arms_to_run == ["lines"]):
        print("contract lines serves tab-text replies and only the lines arm can read them (and vice versa); use "
              "--replies-contract lines --arms lines / 行式契约回放 tab 文本回复，只有行式臂能读（反之亦然）；请用 --replies-contract lines --arms lines")  # fmt: skip
        return 2
    if transforms.active and "full" in arms_to_run:
        print("note: transforms act on the compact arm only; the full arm serves the reply verbatim / 注意：变换仅作用于紧凑臂，完整臂原样回放")

    # ---- timeline identical to the real run / 与真实 run 相同的时间线
    all_questions = load_questions(args.data_root / QA_FILENAME)
    questions = all_questions[: args.limit]
    windows: list[Window] = build_windows(load_transcripts(args.data_root))
    steps = schedule(questions, windows)
    batches = plan_batches(steps, args.batch_size)
    plan_recorded = json.loads((args.run / "batches.json").read_text(encoding="utf-8"))
    plan_here = [{"batch_id": b.batch_id, "position": b.position, "window_ids": b.window_ids} for b in batches]
    if plan_here != plan_recorded:
        print("batch plan differs from the run's batches.json; aborting / 批次计划与 run 的 batches.json 不一致；中止")
        return 2
    if replies_run != args.run:
        plan_replies = json.loads((replies_run / "batches.json").read_text(encoding="utf-8"))
        if plan_replies != plan_recorded:
            print("--replies-run batches.json differs from --run's; aborting / --replies-run 的 batches.json 与 --run 不一致；中止")
            return 2
    text_by_id: dict[str, str] = {}
    for line in (args.run / "windows.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text_by_id[str(row.get("window_id"))] = str(row.get("text") or "")
    text_mismatch = sum(1 for w in windows if text_by_id.get(w.window_id) != w.text)
    recorded = load_recorded(replies_run / "extract_raw_elite.jsonl")
    leaves_all = {b.batch_id: resolve_leaves(b, recorded) for b in batches}
    replayed = sum(1 for ls in leaves_all.values() for lf in ls if lf.row is not None)
    unresolved = [(lf.batch_id, lf.split) for ls in leaves_all.values() for lf in ls if lf.row is None]
    truncated_unresolved = sum(1 for bid, sp in unresolved if recorded.rows.get((bid, sp)))
    never_recorded = len(unresolved) - truncated_unresolved
    batches_unrecorded = sum(1 for b in batches if b.batch_id not in recorded.batches_seen)
    print(f"timeline: questions={len(questions)} windows={len(windows)} batches={len(batches)} text_mismatch={text_mismatch}")
    print(f"recorded: lines={recorded.lines} truncated_line={recorded.lines_truncated} batches_seen={len(recorded.batches_seen)} "
          f"leaves_replayable={replayed} unresolved={len(unresolved)} (truncated_unresolved={truncated_unresolved}, never_recorded={never_recorded})", flush=True)

    # ---- both arms, local only / 两个臂，仅本地
    run_steps = steps[: args.steps] if args.steps > 0 else steps
    arms: dict[str, dict[str, Any]] = {}
    for arm in arms_to_run:
        t0 = time.perf_counter()
        arms[arm] = run_arm(arm, run_steps, batches, recorded, work_root=work_root, out=out,
                            speaker_from_subject=not args.no_speaker_from_subject,
                            entities_from_mentions=not args.no_entities_from_mentions,
                            contract=contract, transforms=transforms,
                            verdict_min_similarity=args.verdict_min_similarity)  # fmt: skip
        arms[arm]["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        print(f"[{arm}] done in {arms[arm]['wall_ms'] / 1000:.1f} s; atoms={arms[arm]['atoms']} replay={arms[arm]['replay']}", flush=True)

    # ---- answer pass / 答题阶段
    meter = Meter()
    qmap = {q.id: q for q in questions}
    if args.no_answer:
        status: dict[str, Any] = {"stopped": True, "reason": "--no-answer", "requests": 0, "cost_usd": 0.0, "projected_usd": None}
    else:
        status = answer_pass(arms, qmap, meter, max_requests=args.max_answer_requests, cost_cap=args.cost_cap_usd)
    meter.write_jsonl(out / "calls.jsonl")

    # ---- files / 文件
    for arm in arms:
        with (out / f"per_question_{arm}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in arms[arm]["records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    speaker_rule = ("recorded replies carry no `speaker`; a named `subject` (one the library's _speaker accepts) was promoted to the compact "
                    "`speaker`" if not args.no_speaker_from_subject else "compact `speaker` is null everywhere (--no-speaker-from-subject)")
    entities_rule = ("the recorded `entity_mentions` (through the library's _str_list, non-empty only) was carried as the compact `entities` "
                     "hint, so compact entity_mentions = union(hint, extractor(text), speaker)" if not args.no_entities_from_mentions
                     else "no `entities` hint (--no-entities-from-mentions): compact entity_mentions = extractor(text) ∪ speaker only")
    summary: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run),
        "out_dir": str(out),
        "questions": len(questions),
        "steps_run": len(run_steps),
        "windows": len(windows),
        "windows_text_mismatch_vs_run": text_mismatch,
        "batches": len(batches),
        "recorded": {"lines": recorded.lines, "lines_truncated": recorded.lines_truncated, "batches_seen": len(recorded.batches_seen)},
        "replies_replayed": replayed,
        "replies_missing": len(unresolved),
        "missing_breakdown": {"truncated_unresolved": truncated_unresolved, "never_recorded": never_recorded, "batches_unrecorded": batches_unrecorded,
                              "unresolved_slices": [f"{b}:{s or 'whole'}" for b, s in unresolved]},  # fmt: skip
        "replies_run_dir": str(replies_run),
        "variant": {"replies_contract": contract, "arms": arms_to_run, "transforms": transforms.describe(),
                    "verdict_min_similarity_arg": args.verdict_min_similarity,
                    "verdict_min_similarity": {arm: arms[arm]["replay"]["verdict_min_similarity"] for arm in arms},
                    "argv": list(argv) if argv is not None else sys.argv[1:]},  # fmt: skip
        "arms": {arm: arm_summary(arm, arms[arm], meter) for arm in arms},
        "paired": paired_diff(arms["full"]["records"], arms["compact"]["records"]) if set(ARMS) <= set(arms) else None,
        "entity_axis": entity_axis_diff(arms["full"]["entity_axis"], arms["compact"]["entity_axis"]) if set(ARMS) <= set(arms) else None,
        "derived_vs_llm": arms["compact"]["agreement"] if "compact" in arms else None,
        "answer_pass": status,
        "api_cost_usd": round(sum(c.cost_usd or 0.0 for c in meter.calls), 6),
        "config": {
            "extraction_mode": "single", "embedder": f"fastembed:{EMBED_MODEL}", "index": "sqlite", "top_k": TOP_K, "threshold": THRESHOLD,
            "batch_size": args.batch_size, "speaker_rule": speaker_rule, "entities_rule": entities_rule,
            "projection_chars_per_token": PROJECTION_CHARS_PER_TOKEN,
            "max_answer_requests": args.max_answer_requests, "cost_cap_usd": args.cost_cap_usd,
        },  # fmt: skip
        "method": (
            "Both arms consume the SAME recorded deepseek-flash replies (extract_raw_elite.jsonl) through a ReplayLLM spliced into HMS "
            "(extraction_mode=single; store sqlite; embedder fastembed bge-small-en-v1.5). Arm full serves the reply verbatim under "
            "extraction_output=full; arm compact serves the reply with every observation reduced to the compact keys (id, event_idx, text, "
            "type, speaker, supersedes_hint_ids when non-empty, when = date_observed when it differs from the message date, valid_to when "
            "given, entities = the recorded entity_mentions when non-empty unless --no-entities-from-mentions) under "
            "extraction_output=compact, so subject/predicate/objects/relation/confidence/attributed_to/sentiment/topic_tags/"
            "date_observed/valid_from are derived by hms_elite.ingest.derive and entity_mentions = union(entities hint, extractor(text), "
            "speaker); eps and decisions are passed verbatim. The entity axis of the two arms is compared on the live atoms paired by "
            "(batch_id, text). The "
            "run's bisection is re-walked on the recorded attempts (a truncated slice is replaced by its two recorded halves; a slice with "
            "no usable reply is skipped). Timeline = timeline.schedule over the 204 questions; per question the position's slices are "
            "retained (HMS.add, local), then HMS.search(top_k=20, threshold=0.1, reference_date=query_time); hit@20 = a returned row's "
            "batch contains a gold window (window containing target_time); MRR = 1/rank of the first gold batch in the deduplicated "
            "returned-batch order (row-rank variant also given); scored = questions with a gold window retained at query time "
            "(excluded_future=False). Answer pass = real_arms.answer (their build_question_prompt + '- row' Memories block, deepseek-flash, "
            "temperature 0, max_tokens 8, thinking disabled), costed with accounting.Meter at the per-request rate tier."
        ),
        "caveats": [
            (f"Variant mode: replies from `{replies_run}` read under contract '{contract}'; arms run = {arms_to_run}; transforms = "
             f"{json.dumps(transforms.describe())} (compact arm only). Under contract 'compact' the recorded items are served as recorded "
             "(speaker / entities / when as the compact prompt elicited them; no reduction, no derived-vs-LLM tally) and a reply whose "
             "root the lenient parser reads as a non-object is skipped and counted (leaves_invalid_json_skipped), as the real run "
             "stored nothing for it."),
            f"{speaker_rule}; the real compact prompt asks the LLM for a speaker name, which the recorded full replies never contain.",
            f"{entities_rule}; the real compact prompt asks the LLM for 0-6 entities per statement, whereas the recorded full-contract "
            "entity_mentions lists are not bounded, so the hint replayed here is what the full prompt elicited, not what the compact prompt "
            "would.",
            "date_observed in the recorded replies almost always equals the message date, so `when` is rarely emitted; recorded valid_from "
            "values are dropped by the reduction and the compact arm's valid_from comes from the anchor (as in real compact mode).",
            "Derived-vs-LLM agreement compares expand_compact_item's output with the raw recorded strings (lower-cased; list fields after "
            "_str_list); two empty lists count as Jaccard 1. Coerced-enum differences (e.g. an out-of-vocabulary LLM sentiment) count as "
            "disagreement.",
            "Local timings are wall-clock on this machine with both arms run sequentially; the full arm ran first (cold fastembed cache).",
            "The run directory was read once; batches whose replies were not yet recorded (the run is still in progress) are skipped in "
            "both arms equally, so hit@20 on the 204 denominator is depressed identically for both arms; the gold-stored subset is the "
            "cleaner comparison.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    if not args.keep_db:
        shutil.rmtree(work_root, ignore_errors=True)
    for arm in arms:
        s = summary["arms"][arm]
        print(f"{arm}: atoms={s['atoms']} atoms_added={s['atoms_added_sum']} hits={s['hits']} hit20={s['hit20']} (n={s['n_scored']}) mrr={s['mrr']} "
              f"leaves_skipped={s['leaves_invalid_json_skipped']} acc={s['answer_accuracy']} ({s['answer_correct']}/{s['answered_n']}) cost={s['answer_cost_usd']} "
              f"guard={s['replay']['verdict_min_similarity']} downgraded={s['replay']['downgraded_total']}")
    print(f"derived_vs_llm={summary['derived_vs_llm']['agreement'] if summary['derived_vs_llm'] else None}")
    ea = summary["entity_axis"]
    if ea is not None:
        print(f"entity_axis: paired={ea['paired']} identical={ea['identical']} ({ea['identical_fraction']}) narrowed={ea['narrowed']} "
              f"compact_superset_or_equal={ea['compact_superset_or_equal']} atoms_with_entities full={summary['arms']['full']['atoms_with_entities']} "
              f"compact={summary['arms']['compact']['atoms_with_entities']}")
    print(f"api_cost_usd={summary['api_cost_usd']} written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
