"""Offline replay: every real deepseek-flash extraction reply through the current extraction code, counting what is
still rejected, which coercions fired, and how many speakers are recovered; BEFORE numbers come from retain logs.
离线回放：把每条真实的 deepseek-flash 抽取回复送入当前抽取代码，统计仍被拒绝的条目、触发的规整以及恢复的说话人数；
BEFORE 数字取自 retain 日志。
"""

# Purpose (EN): Replay results/real/<run>/extract_raw_elite.jsonl (batch_id, split, attempt, finish_reason, content)
#               through hms_elite.utils.json.parse_llm_json + hms_elite.ingest.extraction._observation_from_item /
#               _ep_from_item, with synthetic (SourceEvent, Anchor) witnesses rebuilt from batches.json + windows.jsonl.
#               Rejections are bucketed by the library's warning text; coercions are counted both from the library's
#               DEBUG/WARNING lines and from input-vs-output field diffs. BEFORE numbers are read from the same runs'
#               retain_elite.jsonl library_warnings (and per-sub `warnings`).
# 目的（中）：把 results/real/<run>/extract_raw_elite.jsonl（batch_id、split、attempt、finish_reason、content）
#            送入 hms_elite.utils.json.parse_llm_json 与 hms_elite.ingest.extraction._observation_from_item /
#            _ep_from_item，见证 (SourceEvent, Anchor) 由 batches.json + windows.jsonl 合成重建。
#            拒绝按库的警告文本分桶；规整同时从库的 DEBUG/WARNING 行与输入-输出字段差异两个来源计数。
#            BEFORE 数字取自同一批 run 的 retain_elite.jsonl 的 library_warnings（以及每个子批次的 `warnings`）。

# Input (EN): --runs (default elite_full204 elite_full204_r2 elite_full204_r3), --results-root (default results/real),
#             --json (optional output path), --verbatim (how many verbatim rejection lines to print per bucket).
# 输入（中）：--runs（默认 elite_full204 elite_full204_r2 elite_full204_r3）、--results-root（默认 results/real）、
#            --json（可选输出路径）、--verbatim（每个桶打印多少条原文拒绝行）。

# Output (EN): One table per run and a total on stdout (AFTER = replay, BEFORE = retain logs), the verdict line
#              'VERDICT=clean' or 'VERDICT=defects_remain', and the same figures as JSON when --json is given.
# 输出（中）：每个 run 一张表加合计（AFTER = 回放，BEFORE = retain 日志）、判定行 'VERDICT=clean' 或
#            'VERDICT=defects_remain'，给出 --json 时同时写出同样的数字。

# State effect (EN): Read-only over the result directories; nothing under results/ is written or modified. The last
#                    line of a still-growing extract_raw_elite.jsonl may be truncated and is counted, not parsed.
# 状态影响（中）：只读结果目录；不写入、不修改 results/ 下任何文件。仍在增长的 extract_raw_elite.jsonl 的最后一行
#                可能被截断，只计数不解析。

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from windows import clock_to_iso  # noqa: E402

from hms_elite.enums import ObservationType, Relation, Reward, Sentiment  # noqa: E402
from hms_elite.ingest import extraction  # noqa: E402
from hms_elite.llms.scripted import ScriptedLLM  # noqa: E402
from hms_elite.memory.schema import ATTRIBUTED_TO_VALUES  # noqa: E402
from hms_elite.models import Anchor, Scope, SourceEvent  # noqa: E402
from hms_elite.timeutil import date_of  # noqa: E402
from hms_elite.utils.json import parse_llm_json  # noqa: E402

DEFAULT_RUNS = ("elite_full204", "elite_full204_r2", "elite_full204_r3")
EXTRACTION_LOGGER = "hms_elite.ingest.extraction"
JSON_LOGGER = "hms_elite.utils.json"
FALLBACK_DATE = "2024-01-01"
SCOPE = Scope(user_id="egolife-replay")
# Bucket keys of the rejection classes (kept short so the table columns line up).
# 拒绝类别的桶键（保持简短以便表格列对齐）。
WITNESS = "witness"
WRONG_TYPE = "wrong_type"
EMPTY_TEXT = "empty_text"
EP_CLOSURE = "ep_closure"
VOCABULARY = "vocabulary"
_ITEM_ID_RE = re.compile(r"item '[^']*'|item \"[^\"]*\"|item None|item \d+")
_VALUE_RE = re.compile(r"input_value=(?:'[^']*'|\"[^\"]*\"|[^,\]]*)")
_REPR_RE = re.compile(r"(unknown ref |unreadable \w+ |event_idx |unknown \w+ |kappa )(?:'[^']*'|\"[^\"]*\"|[^\s;]+)")


# ---------------------------------------------------------------------------
# Log capture
# 日志捕获
# ---------------------------------------------------------------------------
class _Capture(logging.Handler):
    """Collects (level, message) pairs from the extraction and json loggers; cleared per item.
    收集抽取与 json 两个 logger 的 (级别, 消息) 对；每个条目前清空。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append the formatted message.
        追加格式化后的消息。
        """
        self.records.append((record.levelname, record.getMessage()))

    def take(self) -> list[tuple[str, str]]:
        """Return and clear the buffer.
        返回并清空缓冲。
        """
        out, self.records = self.records, []
        return out


def _attach(capture: _Capture) -> None:
    """Attach the capture handler to both loggers at DEBUG (propagation off so nothing reaches stderr).
    把捕获处理器以 DEBUG 级别挂到两个 logger（关闭传播，避免输出到 stderr）。
    """
    for name in (EXTRACTION_LOGGER, JSON_LOGGER):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        lg.addHandler(capture)


def _template(message: str) -> str:
    """Strip item ids and values from a log line so equal defects share one bucket.
    去掉日志行中的条目 id 与取值，使同类缺陷落入同一个桶。
    """
    text = _ITEM_ID_RE.sub("item <id>", message)
    text = _VALUE_RE.sub("input_value=<v>", text)
    text = _REPR_RE.sub(lambda m: m.group(1) + "<v>", text)
    text = re.sub(r"valid_to \S+ precedes \S+", "valid_to <t> precedes <t>", text)
    text = re.sub(r"replaced by \S+", "replaced by <d>", text)
    text = re.sub(r"for observation '[^']*'", "for observation <id>", text)
    text = re.sub(r"\s*For further information visit \S+", "", text)
    return text.replace("\n", " | ")


def _classify(message: str) -> str:
    """Rejection class of a 'skipped' / 'dropped' warning: witness, wrong_type, empty_text, ep_closure, else vocabulary.
    'skipped' / 'dropped' 警告的拒绝类别：witness、wrong_type、empty_text、ep_closure，其余归 vocabulary。
    """
    if "invalid event_idx" in message:
        return WITNESS
    if "must be a string" in message or "must be a list" in message:
        return WRONG_TYPE
    if "String should have at least 1 character" in message or "text" in message and "min_length" in message:
        return EMPTY_TEXT
    if "closure" in message.lower() or "slots" in message.lower() and "u" in message:
        return EP_CLOSURE
    return VOCABULARY


# ---------------------------------------------------------------------------
# Witness reconstruction
# 见证重建
# ---------------------------------------------------------------------------
def _load_windows(run_dir: Path) -> dict[str, dict[str, Any]]:
    """windows.jsonl keyed by window_id.
    以 window_id 为键的 windows.jsonl。
    """
    out: dict[str, dict[str, Any]] = {}
    path = run_dir / "windows.jsonl"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[str(row.get("window_id"))] = row
    return out


def _load_batches(run_dir: Path) -> dict[str, list[str]]:
    """batches.json → {batch_id: [window_id, ...]}.
    batches.json → {batch_id: [window_id, ...]}。
    """
    path = run_dir / "batches.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r.get("batch_id")): list(r.get("window_ids") or []) for r in rows if isinstance(r, dict)}


def _slice(window_ids: list[str], split: Any) -> list[str]:
    """Apply a bisection split 'lo-hi' (r2/r3 runs) to the batch's window list.
    把二分切片 'lo-hi'（r2/r3 run）应用到批次的窗口列表。
    """
    if isinstance(split, str) and re.fullmatch(r"\d+-\d+", split):
        lo, hi = (int(x) for x in split.split("-"))
        return window_ids[lo:hi]
    return window_ids


def _instant(window: dict[str, Any] | None) -> str:
    """The synthetic UTC instant of a window (clock_to_iso of its day and start), or the fallback date's midnight.
    窗口的合成 UTC 时刻（其 day 与 start 的 clock_to_iso），否则为回退日期的午夜。
    """
    if window and window.get("day") is not None and window.get("start_cs") is not None:
        return clock_to_iso(int(window["day"]), int(window["start_cs"]))
    return f"{FALLBACK_DATE}T00:00:00.000000Z"


def _witnesses(texts: list[tuple[str, str]], n: int) -> list[tuple[SourceEvent, Anchor]]:
    """Build n (SourceEvent, Anchor) pairs: role user, visibility private, fixed scope; real window text where known,
    a placeholder beyond the real window count (so any event_idx is witnessable, as the task specifies).
    构造 n 个 (SourceEvent, Anchor)：角色 user、可见性 private、固定 scope；已知窗口用真实文本，
    超出真实窗口数的位置用占位文本（这样任意 event_idx 都可被见证，符合任务要求）。
    """
    out: list[tuple[SourceEvent, Anchor]] = []
    last_stamp = texts[-1][1] if texts else f"{FALLBACK_DATE}T00:00:00.000000Z"
    for i in range(n):
        text, stamp = texts[i] if i < len(texts) else (f"<synthetic message {i}>", last_stamp)
        ev = SourceEvent(
            event_id=f"ev-{i}", role="user", visibility="private", scope=SCOPE, timestamp=stamp,
            payload={"text": text}, raw_ref=f"replay://{i}",
        )  # fmt: skip
        anc = Anchor(
            anchor_id=f"anc-{i}", source_event_id=ev.event_id, hash=ev.payload_hash, source_ref=ev.raw_ref,
            timestamp=ev.timestamp, ingested_at=ev.timestamp, permission_scope="private", scope=SCOPE,
        )  # fmt: skip
        out.append((ev, anc))
    return out


# ---------------------------------------------------------------------------
# Coercion inference (input value vs output value)
# 规整推断（输入值 vs 输出值）
# ---------------------------------------------------------------------------
def _norm_enum(value: Any) -> str | None:
    """The bare lower-cased string of an enum-ish input (None/'' → None).
    枚举类输入的小写裸字符串（None/'' → None）。
    """
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if isinstance(v, str) and v.strip()), "")
        if not value:
            return None
    return str(value).strip().lower()


def _enum_diff(raw: Any, out: str, values: set[str], field: str, into: Counter) -> None:
    """Count '<field>_defaulted' (missing) or '<field>_folded' (present but not the member as written).
    计数 '<field>_defaulted'（缺失）或 '<field>_folded'（存在但不是照写的成员值）。
    """
    text = _norm_enum(raw)
    if text is None:
        into[f"{field}_defaulted"] += 1
    elif text not in values or text != out:
        into[f"{field}_folded"] += 1


def _prob_diff(raw: Any, out: float, field: str, into: Counter) -> None:
    """Count '<field>_defaulted' (missing) or '<field>_rescaled' (numeric but not equal to the output).
    计数 '<field>_defaulted'（缺失）或 '<field>_rescaled'（是数值但与输出不等）。
    """
    if raw is None or raw == "" or isinstance(raw, bool):
        into[f"{field}_defaulted"] += 1
        return
    try:
        number = float(raw)
    except (TypeError, ValueError):
        into[f"{field}_rescaled"] += 1
        return
    if abs(number - out) > 1e-9:
        into[f"{field}_rescaled"] += 1


def _instant_diff(raw: Any, out: str | None, field: str, into: Counter) -> None:
    """Count '<field>_dropped' (given, now None) or '<field>_normalised' (given, spelled differently after coercion).
    计数 '<field>_dropped'（给了但变为 None）或 '<field>_normalised'（给了但规整后写法不同）。
    """
    if raw is None or raw == "":
        return
    if out is None:
        into[f"{field}_dropped"] += 1
    elif str(raw) != out:
        into[f"{field}_normalised"] += 1


def _list_diff(raw: Any, out: list[str], field: str, into: Counter) -> None:
    """Count '<field>_flattened' (a dict/list/number item), '<field>_split' (one string became several) and
    '<field>_deduped' (fewer strings out than in).
    计数 '<field>_flattened'（dict/list/数字项）、'<field>_split'（一个字符串变多个）与 '<field>_deduped'（输出少于输入）。
    """
    if isinstance(raw, str):
        if len(out) > 1:
            into[f"{field}_split"] += 1
        return
    if not isinstance(raw, (list, tuple)):
        return
    if any(isinstance(v, (dict, list, tuple, int, float)) and not isinstance(v, bool) for v in raw):
        into[f"{field}_flattened"] += 1
    strings = [v for v in raw if isinstance(v, str)]
    if len(strings) == len(raw) and len(out) > len(raw):
        into[f"{field}_split"] += 1
    if len(strings) == len(raw) and len(out) < len([s for s in strings if s.strip()]):
        into[f"{field}_deduped"] += 1


def _observation_coercions(item: dict, obs: Any, n_real: int, into: Counter, values: Counter) -> None:
    """Infer every coercion that fired on one accepted observation item from input vs output; `values` tallies the
    raw attributed_to strings that were not a legal side (→ 'raw -> speaker|side-only').
    从输入与输出推断一个被接受的观察条目上触发的全部规整；`values` 统计不是合法侧别的原始 attributed_to 字符串
    （→ 'raw -> speaker|side-only'）。
    """
    _enum_diff(item.get("type"), obs.type, set(ObservationType.values()), "type", into)
    _enum_diff(item.get("relation"), obs.relation, set(Relation.values()), "relation", into)
    _enum_diff(item.get("sentiment"), obs.sentiment, set(Sentiment.values()), "sentiment", into)
    _prob_diff(item.get("confidence"), float(obs.confidence), "confidence", into)
    _instant_diff(item.get("valid_from"), obs.valid_from, "valid_from", into)
    _instant_diff(item.get("valid_to"), obs.valid_to, "valid_to", into)
    raw_date = item.get("date_observed")
    if raw_date is None or raw_date == "":
        into["date_observed_defaulted"] += 1
    elif str(raw_date) != obs.date_observed:
        into["date_observed_replaced"] += 1
    idx = item.get("event_idx")
    if idx is None:
        into["event_idx_defaulted"] += 1
    elif not isinstance(idx, int) or isinstance(idx, bool):
        into["event_idx_read_from_string_or_float"] += 1
    if isinstance(idx, int) and not isinstance(idx, bool) and idx >= n_real:
        into["event_idx_beyond_real_window_count"] += 1
    raw_att = item.get("attributed_to")
    if raw_att is None or raw_att == "":
        into["attributed_to_defaulted"] += 1
    elif not isinstance(raw_att, str):
        into["attributed_to_non_string_ignored"] += 1
    elif raw_att.strip().lower() not in ATTRIBUTED_TO_VALUES:
        into["attributed_to_side_fallback"] += 1
        if obs.speaker and obs.speaker == raw_att.strip():
            into["speaker_from_attributed_to"] += 1
            values[f"{raw_att.strip()!r} -> speaker, side={obs.attributed_to}"] += 1
        else:
            into["attributed_to_side_word_or_collective"] += 1
            values[f"{raw_att.strip()!r} -> side-only {obs.attributed_to}"] += 1
    if isinstance(item.get("speaker"), str) and item["speaker"].strip():
        into["speaker_explicit_accepted" if obs.speaker == item["speaker"].strip() else "speaker_explicit_ignored"] += 1
    mentions = extraction._str_list(item.get("entity_mentions"), what="replay", field="entity_mentions")
    if obs.speaker and obs.speaker.lower() not in {m.lower() for m in mentions}:
        into["speaker_appended_to_entity_mentions"] += 1
    if item.get("subject") in (None, ""):
        into["subject_defaulted"] += 1
    if item.get("predicate") in (None, ""):
        into["predicate_defaulted"] += 1
    _list_diff(item.get("objects"), list(obs.objects), "objects", into)
    _list_diff(item.get("topic_tags"), list(obs.topic_tags), "topic_tags", into)
    # The appended speaker is its own coercion (counted above); it must not read as an entity_mentions split.
    # 追加的说话人是独立的规整（上面已计数）；不能被误读为 entity_mentions 的切分。
    mentions_out = [m for m in obs.entity_mentions if not (obs.speaker and m == obs.speaker and m not in mentions)]
    _list_diff(item.get("entity_mentions"), mentions_out, "entity_mentions", into)
    hints = item.get("supersedes_hint_ids")
    if hints:
        into["supersedes_hint_ids_present_unmapped_in_replay"] += 1


def _ep_coercions(item: dict, ep: Any, into: Counter) -> None:
    """Infer every coercion that fired on one accepted EP item.
    推断一个被接受的 EP 条目上触发的全部规整。
    """
    _enum_diff(item.get("r"), ep.slots.r, set(Reward.values()), "ep_r", into)
    _prob_diff(item.get("confidence"), float(ep.confidence), "ep_confidence", into)
    raw_anchor = item.get("time_anchor")
    if raw_anchor is None or raw_anchor == "":
        into["ep_time_anchor_defaulted"] += 1
    elif str(raw_anchor) != ep.time_anchor:
        into["ep_time_anchor_normalised_or_event_time"] += 1
    kappa = item.get("kappa")
    if kappa is not None and not isinstance(kappa, dict):
        into["ep_kappa_non_dict"] += 1
    if isinstance(kappa, dict) and kappa.get("visibility") not in (None, "", ep.slots.kappa.visibility):
        into["ep_kappa_visibility_replaced"] += 1
    _list_diff(item.get("topic_tags"), list(ep.topic_tags), "ep_topic_tags", into)
    _list_diff(item.get("entity_mentions"), list(ep.entity_mentions), "ep_entity_mentions", into)
    if item.get("uses_ep_ids"):
        into["ep_uses_ep_ids_present_unmapped_in_replay"] += 1


# ---------------------------------------------------------------------------
# Replay of one run
# 单个 run 的回放
# ---------------------------------------------------------------------------
def _fresh_stats() -> dict[str, Any]:
    """The empty AFTER counters of one run.
    单个 run 的空 AFTER 计数器。
    """
    return {
        "rows": 0, "rows_truncated": 0, "replies_total": 0, "replies_blank": 0, "replies_unparseable": 0,
        "replies_list_as_observations": 0, "replies_non_object": 0, "finish_reasons": Counter(),
        "obs_items": 0, "obs_accepted": 0, "obs_rejected": 0, "obs_reject_class": Counter(),
        "ep_items": 0, "ep_accepted": 0, "ep_rejected": 0, "ep_reject_class": Counter(),
        "rejection_templates": Counter(), "rejection_verbatim": {}, "non_rejecting_warnings": Counter(),
        "library_debug": Counter(), "coercions": Counter(), "speakers": 0, "speaker_names": Counter(),
        "attributed_to_values": Counter(), "unparseable_by_finish": Counter(), "decisions_items": 0, "batches": set(),
    }  # fmt: skip


def _parse_reply(content: str, capture: _Capture, stats: dict[str, Any]) -> dict | None:
    """parse_llm_json(key=None) plus extraction.py's list → {'observations': [...]} rule; None when unusable.
    parse_llm_json(key=None) 加上 extraction.py 的 list → {'observations': [...]} 规则；不可用时返回 None。
    """
    if not (content or "").strip():
        stats["replies_blank"] += 1
        return None
    raw = parse_llm_json(content, key=None, task="memory_extraction")
    errors = [m for lvl, m in capture.take() if lvl == "ERROR"]
    if isinstance(raw, list) and raw:
        stats["replies_list_as_observations"] += 1
        return {"observations": raw}
    if isinstance(raw, list):
        stats["replies_unparseable" if errors else "replies_non_object"] += 1
        return None
    if not isinstance(raw, dict):
        stats["replies_non_object"] += 1
        return None
    return raw


def _record_item_logs(
    logs: list[tuple[str, str]], accepted: bool, stats: dict[str, Any], klass_counter: Counter, verbatim_cap: int
) -> None:
    """Route one item's captured log lines into rejection / non-rejecting / debug buckets.
    把一个条目捕获到的日志行分到拒绝 / 非拒绝警告 / debug 三类桶。
    """
    rejected_seen = False
    for level, message in logs:
        template = _template(message)
        if level == "DEBUG":
            stats["library_debug"][template] += 1
            continue
        is_rejection = (" skipped: " in message or "invalid event_idx" in message) and not accepted
        if is_rejection and not rejected_seen:
            rejected_seen = True
            klass = _classify(message)
            klass_counter[klass] += 1
            key = f"[{klass}] {template}"
            stats["rejection_templates"][key] += 1
            bucket = stats["rejection_verbatim"].setdefault(key, [])
            if len(bucket) < verbatim_cap:
                bucket.append(message.replace("\n", " | "))
        else:
            stats["non_rejecting_warnings"][template] += 1


def replay_run(run_dir: Path, capture: _Capture, *, verbatim_cap: int) -> dict[str, Any]:
    """Replay every reply of one run; returns the AFTER counters.
    回放一个 run 的每条回复；返回 AFTER 计数器。
    """
    stats = _fresh_stats()
    windows = _load_windows(run_dir)
    batches = _load_batches(run_dir)
    llm = ScriptedLLM({"responses": []})
    path = run_dir / "extract_raw_elite.jsonl"
    # Read once; the run may still be growing, so the final line can be a partial JSON document.
    # 只读一次；该 run 可能仍在增长，最后一行可能是不完整的 JSON 文档。
    lines = path.read_text(encoding="utf-8").split("\n")
    for line in lines:
        if not line.strip():
            continue
        stats["rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            stats["rows_truncated"] += 1
            continue
        stats["replies_total"] += 1
        stats["finish_reasons"][str(row.get("finish_reason"))] += 1
        batch_id = str(row.get("batch_id"))
        stats["batches"].add(batch_id)
        window_ids = _slice(batches.get(batch_id, []), row.get("split"))
        texts = [(str(windows.get(w, {}).get("text") or f"<window {w}>"), _instant(windows.get(w))) for w in window_ids]
        observation_date = date_of(texts[-1][1]) if texts else FALLBACK_DATE
        capture.take()
        unparseable_before = stats["replies_unparseable"]
        raw = _parse_reply(row.get("content") or "", capture, stats)
        if raw is None:
            if stats["replies_unparseable"] > unparseable_before:
                stats["unparseable_by_finish"][str(row.get("finish_reason"))] += 1
            continue
        obs_items = extraction._root_items(raw, "observations", what="memory_extraction")
        ep_items = extraction._root_items(raw, "eps", what="memory_extraction")
        decisions = raw.get("decisions")
        stats["decisions_items"] += len(decisions) if isinstance(decisions, (list, dict)) else 0
        for level, message in capture.take():
            stats["non_rejecting_warnings" if level != "DEBUG" else "library_debug"][_template(message)] += 1
        # Events long enough for any event_idx the reply cites (real windows first, placeholders after).
        # 事件列表长到足以覆盖回复引用的任意 event_idx（真实窗口在前，占位在后）。
        cited = [
            int(float(i.get("event_idx")))
            for i in obs_items + ep_items
            if isinstance(i.get("event_idx"), (int, float, str)) and not isinstance(i.get("event_idx"), bool)
            and re.fullmatch(r"\d+(\.0+)?", str(i.get("event_idx")).strip())
        ]
        n_events = max(len(texts), (max(cited) + 1) if cited else 0, 1)
        events = _witnesses(texts, n_events)
        for item in obs_items:
            stats["obs_items"] += 1
            obs = extraction._observation_from_item(
                llm, item, events=events, observation_date=observation_date, scope=SCOPE, id_map={},
                what="memory_extraction.observations",
            )  # fmt: skip
            logs = capture.take()
            if obs is None:
                stats["obs_rejected"] += 1
            else:
                stats["obs_accepted"] += 1
                _observation_coercions(item, obs, len(texts), stats["coercions"], stats["attributed_to_values"])
                if obs.speaker:
                    stats["speakers"] += 1
                    stats["speaker_names"][obs.speaker] += 1
            _record_item_logs(logs, obs is not None, stats, stats["obs_reject_class"], verbatim_cap)
        for item in ep_items:
            stats["ep_items"] += 1
            ep = extraction._ep_from_item(
                llm, item, events=events, ep_map={}, scope=SCOPE, what="memory_extraction.eps"
            )
            logs = capture.take()
            if ep is None:
                stats["ep_rejected"] += 1
            else:
                stats["ep_accepted"] += 1
                _ep_coercions(item, ep, stats["coercions"])
            _record_item_logs(logs, ep is not None, stats, stats["ep_reject_class"], verbatim_cap)
    stats["batches"] = len(stats["batches"])
    return stats


# ---------------------------------------------------------------------------
# BEFORE numbers from retain logs
# 取自 retain 日志的 BEFORE 数字
# ---------------------------------------------------------------------------
def before_from_retain(run_dir: Path) -> dict[str, Any]:
    """Count the library warnings the real run logged: items skipped by class, replies dropped as non-object lists,
    replies read as lists, other warnings; plus rows and the sum of `added`.
    统计真实 run 记录的库警告：按类别的条目跳过数、作为非对象列表丢弃的回复数、按列表读取的回复数、其它警告；
    另有行数与 `added` 之和。
    """
    out: dict[str, Any] = {
        "rows": 0, "rows_truncated": 0, "added": 0, "items_skipped": 0, "skip_class": Counter(),
        "skip_templates": Counter(), "replies_dropped_non_object": 0, "replies_list_as_observations": 0,
        "parse_errors": 0, "other_warnings": Counter(),
    }  # fmt: skip
    path = run_dir / "retain_elite.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        out["rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out["rows_truncated"] += 1
            continue
        out["added"] += int(row.get("added") or 0)
        # r2/r3 rows repeat the batch's library_warnings verbatim inside every subs[].warnings (verified: the sorted
        # lists are identical on all 146 rows), so the sub lists are read only when the top-level list is empty.
        # r2/r3 的行把批次的 library_warnings 原样重复在每个 subs[].warnings 里（已核对：146 行的排序列表完全相同），
        # 因此仅当顶层列表为空时才读取子批次列表。
        warnings = list(row.get("library_warnings") or [])
        if not warnings:
            for sub in row.get("subs") or []:
                warnings += list(sub.get("warnings") or [])
        for w in warnings:
            text = str(w)
            if " skipped: " in text or "invalid event_idx" in text:
                out["items_skipped"] += 1
                klass = _classify(text)
                out["skip_class"][klass] += 1
                out["skip_templates"][f"[{klass}] {_template(text)}"] += 1
            elif "reply is not a JSON object" in text:
                out["replies_dropped_non_object"] += 1
            elif "reply is a JSON list" in text:
                out["replies_list_as_observations"] += 1
            elif "Error parsing" in text:
                out["parse_errors"] += 1
            else:
                out["other_warnings"][_template(text)] += 1
    return out


# ---------------------------------------------------------------------------
# Reporting
# 报告
# ---------------------------------------------------------------------------
def _merge(total: dict[str, Any], part: dict[str, Any]) -> None:
    """Add one run's counters into the total (ints summed, Counters merged, verbatim lists concatenated).
    把一个 run 的计数并入合计（整数相加、Counter 合并、原文列表拼接）。
    """
    for key, value in part.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, Counter):
            total.setdefault(key, Counter()).update(value)
        elif isinstance(value, dict):
            bucket = total.setdefault(key, {})
            for k, v in value.items():
                bucket.setdefault(k, []).extend(v)


def _row(label: str, before: Any, after: Any) -> str:
    """One aligned BEFORE | AFTER table line.
    一行对齐的 BEFORE | AFTER 表格行。
    """
    return f"  {label:<46} {str(before):>12} {str(after):>12}"


def print_run(name: str, before: dict[str, Any], after: dict[str, Any], *, verbatim_cap: int) -> None:
    """Print the side-by-side table of one run (or the total).
    打印一个 run（或合计）的并排表格。
    """
    print(f"\n=== {name} ===")
    print(_row("metric", "BEFORE(retain)", "AFTER(replay)"))
    print(_row("rows (retain / extract_raw)", before["rows"], after["rows"]))
    print(_row("rows truncated (partial last line)", before["rows_truncated"], after["rows_truncated"]))
    print(_row("replies replayed", "-", after["replies_total"]))
    print(_row("  finish_reason", "-", dict(after["finish_reasons"])))
    print(_row("replies blank", "-", after["replies_blank"]))
    print(_row("replies unparseable (parse ladder -> [])", before["parse_errors"], after["replies_unparseable"]))
    print(_row("  unparseable by finish_reason", "-", dict(after["unparseable_by_finish"])))
    print(_row("replies dropped: non-object list", before["replies_dropped_non_object"], after["replies_non_object"]))
    print(_row("replies read as list->observations", before["replies_list_as_observations"], after["replies_list_as_observations"]))
    print(_row("observation items seen", "-", after["obs_items"]))
    print(_row("observation items accepted", before["added"], after["obs_accepted"]))
    print(_row("observation items rejected", before["items_skipped"], after["obs_rejected"]))
    for klass in (VOCABULARY, WITNESS, WRONG_TYPE, EMPTY_TEXT):
        print(_row(f"  rejected: {klass}", before["skip_class"].get(klass, 0), after["obs_reject_class"].get(klass, 0)))
    print(_row("ep items seen / accepted / rejected", "-", f"{after['ep_items']}/{after['ep_accepted']}/{after['ep_rejected']}"))
    for klass, n in sorted(after["ep_reject_class"].items()):
        print(_row(f"  ep rejected: {klass}", "-", n))
    print(_row("speakers recovered (Observation.speaker)", 0, after["speakers"]))
    print(_row("decision items seen (not replayed)", "-", after["decisions_items"]))
    print("  BEFORE added: sum of retain `added` = atoms stored, not items accepted (the two differ by dedup/NONE).")
    print("  BEFORE added：retain `added` 之和 = 已存原子数，不等于接受条目数（两者因去重/NONE 有差）。")
    if after["rejection_templates"]:
        print("  -- rejection reasons AFTER (bucketed by warning template) / 回放后的拒绝原因（按警告模板分桶）")
        for key, n in after["rejection_templates"].most_common():
            print(f"     {n:>6}  {key}")
            for sample in after["rejection_verbatim"].get(key, [])[:verbatim_cap]:
                print(f"             verbatim: {sample}")
    if before["skip_templates"]:
        print("  -- rejection reasons BEFORE (retain logs) / 回放前的拒绝原因（retain 日志）")
        for key, n in before["skip_templates"].most_common():
            print(f"     {n:>6}  {key}")
    if after["non_rejecting_warnings"]:
        print("  -- non-rejecting WARNING/ERROR lines AFTER / 回放后不导致拒绝的 WARNING/ERROR 行")
        for key, n in after["non_rejecting_warnings"].most_common():
            print(f"     {n:>6}  {key}")
    if before["other_warnings"]:
        print("  -- other warnings BEFORE / 回放前的其它警告")
        for key, n in before["other_warnings"].most_common():
            print(f"     {n:>6}  {key}")
    print("  -- coercions fired AFTER (input vs output) / 回放后触发的规整（输入 vs 输出）")
    for key, n in sorted(after["coercions"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"     {n:>6}  {key}")
    if after["library_debug"]:
        print("  -- library DEBUG lines AFTER / 回放后库的 DEBUG 行")
        for key, n in after["library_debug"].most_common():
            print(f"     {n:>6}  {key}")
    if after["speaker_names"]:
        print("  -- speaker names (top 12) / 说话人名（前 12）")
        print("     " + ", ".join(f"{k}={v}" for k, v in after["speaker_names"].most_common(12)))
    if after["attributed_to_values"]:
        print("  -- raw attributed_to values outside user|assistant (top 25) / user|assistant 之外的原始 attributed_to 值（前 25）")
        for key, n in after["attributed_to_values"].most_common(25):
            print(f"     {n:>6}  {key}")


def _jsonable(value: Any) -> Any:
    """Counters → dicts, sets → sorted lists, recursively.
    递归地把 Counter 转 dict、set 转排序列表。
    """
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(value)
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry: replay each run, print tables and the total, emit the verdict.
    命令行入口：回放每个 run，打印各表与合计，输出判定。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--results-root", default=str(HERE / "results" / "real"))
    parser.add_argument("--json", default=None, help="optional JSON output path / 可选的 JSON 输出路径")
    parser.add_argument("--verbatim", type=int, default=3, help="verbatim lines per bucket / 每桶原文行数")
    args = parser.parse_args(argv)
    capture = _Capture()
    _attach(capture)
    total_after: dict[str, Any] = {}
    total_before: dict[str, Any] = {}
    per_run: dict[str, dict[str, Any]] = {}
    for name in args.runs:
        run_dir = Path(args.results_root) / name
        if not (run_dir / "extract_raw_elite.jsonl").exists():
            print(f"\n=== {name} === missing extract_raw_elite.jsonl; skipped / 缺少 extract_raw_elite.jsonl；跳过")
            continue
        after = replay_run(run_dir, capture, verbatim_cap=args.verbatim)
        before = before_from_retain(run_dir)
        per_run[name] = {"before": before, "after": after}
        _merge(total_after, after)
        _merge(total_before, before)
        print_run(name, before, after, verbatim_cap=args.verbatim)
    if not per_run:
        print("no runs replayed / 没有回放任何 run")
        return 2
    for key in ("skip_class", "skip_templates", "other_warnings"):
        total_before.setdefault(key, Counter())
    for key in ("finish_reasons", "obs_reject_class", "ep_reject_class", "rejection_templates", "non_rejecting_warnings",
                "library_debug", "coercions", "speaker_names", "attributed_to_values", "unparseable_by_finish"):
        total_after.setdefault(key, Counter())
    total_after.setdefault("rejection_verbatim", {})
    print_run("TOTAL", total_before, total_after, verbatim_cap=args.verbatim)
    vocabulary_left = total_after["obs_reject_class"].get(VOCABULARY, 0) + total_after["ep_reject_class"].get(VOCABULARY, 0)
    verdict = "defects_remain" if vocabulary_left else "clean"
    print(f"\nVERDICT={verdict} vocabulary_rejections_after={vocabulary_left}")
    if args.json:
        payload = {"runs": _jsonable(per_run), "total": {"before": _jsonable(total_before), "after": _jsonable(total_after)}, "verdict": verdict}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
