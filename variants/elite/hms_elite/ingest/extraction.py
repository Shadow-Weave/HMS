"""LLM-backed extraction (observations, EPs, revision decisions, procedural summaries) — runs with NO lock held.
LLM 驱动的抽取（观察、EP、修订决策、过程摘要）——在不持有任何锁的阶段运行。
"""

# Purpose (EN): Build the §6 prompts, call the LLM task contracts (staged trio or the single memory_extraction call),
# and map synthetic refs back to real ids (witness first).
# 目的（中）：构造 §6 提示词，调用 LLM 任务契约（分段三件套或单次 memory_extraction 调用），
# 并把合成引用映射回真实 id（先有见证）。

# Input (EN): Anchored message events, neighbourhood context (existing atoms, last events), constraints, recent EPs.
# 输入（中）：已落锚的消息事件、邻域上下文（已有原子、近期事件）、约束与近期 EP。

# Output (EN): Observations, EPs, (Decision, target_id, reason) triples (or one ExtractionBundle) and procedural markdown.
# 输出（中）：观察、EP、(Decision, target_id, reason) 三元组（或一个 ExtractionBundle）以及过程 markdown。

# State effect (EN): Read-only (store reads and index search); nothing is written.
# 状态影响（中）：只读（读取 store 与索引搜索）；不写入任何内容。

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pydantic

from hms_elite.configs.prompts import (
    EP_EXTRACTION_PROMPT,
    MEMORY_EXTRACTION_COMPACT_PROMPT,
    MEMORY_EXTRACTION_LINES_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    OBSERVATION_EXTRACTION_COMPACT_PROMPT,
    OBSERVATION_EXTRACTION_PROMPT,
    REVISION_DECISION_PROMPT,
    build_ep_prompt,
    build_memory_extraction_prompt,
    build_observation_prompt,
    build_procedural_messages,
    build_revision_prompt,
)
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.enums import Decision, ObservationType, Relation, Reward, Sentiment, StrEnum, Visibility, rank
from hms_elite.exceptions import ValidationError
from hms_elite.index.base import IndexBase
from hms_elite.ingest.atoms import _MIDNIGHT
from hms_elite.ingest.derive import SIDE_SUBJECTS, expand_compact_item, expand_lines_ep
from hms_elite.ingest.lines import parse_lines
from hms_elite.llms.base import LLMBase
from hms_elite.memory.schema import ATTRIBUTED_TO_VALUES
from hms_elite.memory.storage import StateStore
from hms_elite.models import EP, Anchor, Kappa, MemoryAtom, Observation, Scope, Slots, SourceEvent
from hms_elite.timeutil import date_of, to_utc_iso
from hms_elite.utils.text import content_tokens

logger = logging.getLogger(__name__)

_REVISION_CANDIDATE_LIMIT = 5
# 'full' = the LLM emits every observation field (CONTRACT §6.5); 'compact' = the LLM emits
# COMPACT_OBSERVATION_KEYS only and ingest.derive fills the rest locally; 'lines' (CONTRACT §6.6) = the LLM answers
# with tab-separated OBS / EP / DEC text (no JSON mode) that ingest.lines parses into compact-shaped items, the rest
# derived exactly as in 'compact' — single extraction mode only; 'lines' is the IngestConfig default since 2026-09-18
# (CONTRACT §6.6, changelog row 108). Selected by IngestConfig.extraction_output; the `output` parameters below
# default to 'full' as the contract name, the Ingestor always passes the configured value.
# 'full' = LLM 输出每个观察字段（CONTRACT §6.5）；'compact' = LLM 只输出 COMPACT_OBSERVATION_KEYS，
# 其余由 ingest.derive 在本地补齐；'lines'（CONTRACT §6.6）= LLM 以 tab 分隔的 OBS / EP / DEC 文本回答（不用 JSON 模式），
# 由 ingest.lines 解析为紧凑形状的条目，其余与 'compact' 完全一样推导——仅 single 抽取模式；自 2026-09-18 起 'lines' 是
# IngestConfig 的默认值（CONTRACT §6.6，变更记录第 108 行）。由 IngestConfig.extraction_output 选择；下文各 `output` 参数以
# 契约名 'full' 为参数默认，Ingestor 总是传入配置值。
EXTRACTION_OUTPUTS = ("full", "compact", "lines")
# The outputs whose observations are derived locally from a compact-shaped item (ingest.derive).
# 其观察由紧凑形状条目在本地推导（ingest.derive）的输出契约。
_DERIVED_OUTPUTS = frozenset({"compact", "lines"})


def neighbourhood(
    store: StateStore,
    index: IndexBase,
    embedder: EmbedderBase,
    *,
    scope: Scope,
    texts: list[str],
    k: int,
    last_k: int | None = None,
) -> tuple[list[dict], list[SourceEvent]]:
    """Phase-0 context: the k nearest valid atoms (synthetic ids '0'..) and the last k message events of the scope.
    第 0 阶段上下文：最近的 k 个有效原子（合成 id '0'..）与该范围最近的 k 条消息事件。

    Input / 输入：
        store: The state store.
        store：状态存储。

        index: The vector index.
        index：向量索引。

        embedder: The embedder (query embedding, memory_action 'search').
        embedder：嵌入器（查询向量，memory_action 为 'search'）。

        scope: Scope whose live atoms are candidates (store.valid_ids).
        scope：其存活原子作为候选的范围（store.valid_ids）。

        texts: The new message lines ('role: text'); joined with newlines and embedded once.
        texts：新消息行（'role: text'）；用换行拼接并嵌入一次。

        k: Number of neighbours.
        k：邻居数量。

        last_k: Number of recent message events (defaults to k).
        last_k：近期消息事件数量（默认 k）。

    Output / 输出：
        (existing, last_events): existing = [{'id': '0'.., 'text', 'atom_id', 'memory_id', 'hash'}] in similarity order.
        (existing, last_events)：existing = 按相似度排序的 [{'id': '0'.., 'text', 'atom_id', 'memory_id', 'hash'}]。

    State effect / 状态影响：
        None (reads only).
        无（仅读取）。
    """
    existing: list[dict] = []
    candidate_ids = store.valid_ids(scope)["atoms"]
    query = "\n".join(t for t in texts if t)
    if candidate_ids and k > 0 and query.strip():
        vector = embedder.embed(query, "search")
        hits = index.search(vector, candidate_ids=candidate_ids, top_k=k)
        atoms = {a.atom_id: a for a in store.get_atoms([atom_id for atom_id, _score in hits])}
        for atom_id, _score in hits:
            atom = atoms.get(atom_id)
            if atom is None:
                continue
            existing.append(
                {
                    "id": str(len(existing)),
                    "text": atom.text,
                    "atom_id": atom.atom_id,
                    "memory_id": atom.memory_id,
                    "hash": atom.hash,
                }
            )
    last = store.last_events(scope, object_types=["message"], limit=max(0, int(last_k if last_k is not None else k)))
    return existing, last


def _map_refs(refs: Any, id_map: dict[str, str], *, what: str) -> list[str]:
    """Map synthetic refs through id_map; a value that already is a mapped real id passes through; unknown refs are
    dropped with a warning.
    通过 id_map 映射合成引用；本身已是映射目标真实 id 的值直接通过；未知引用记录警告并丢弃。
    """
    out: list[str] = []
    real_ids = set(id_map.values())
    for ref in refs or []:
        key = str(ref).strip()
        mapped = id_map.get(key)
        if mapped is None and key in real_ids:
            mapped = key
        if mapped is None:
            logger.warning("%s: unknown ref %r dropped", what, key)
            continue
        if mapped not in out:
            out.append(mapped)
    return out


def _event_index(item: dict, n: int, *, what: str) -> int | None:
    """Validate an item's event_idx against the prompt's event list.
    校验条目的 event_idx 是否落在提示词事件列表内。

    A digit string ('0') or an integral float (1.0) is read as its int (vocabulary, not meaning). A missing index is
    defaulted to 0 only when the list has exactly one event (the witness is then unambiguous). An out-of-range or
    otherwise unreadable index still skips the item: the witness anchor is the existence condition of the memory
    (WITNESS_SEMANTICS), so guessing one would attribute the statement to the wrong message.
    数字串（'0'）或整数值浮点（1.0）按其整数读取（这是词表问题，不是语义问题）。仅当列表恰有一个事件时（见证无歧义），
    缺失的索引才默认为 0。越界或无法读取的索引仍然跳过该条目：见证锚点是记忆的存在条件（WITNESS_SEMANTICS），
    猜一个会把陈述归到错误的消息上。
    """
    idx: Any = item.get("event_idx")
    if isinstance(idx, bool):
        idx = None
    elif isinstance(idx, float) and idx.is_integer():
        idx = int(idx)
    elif isinstance(idx, str) and idx.strip().isdigit():
        idx = int(idx.strip())
    if idx is None and n == 1:
        logger.debug("%s: event_idx missing; the single event is the witness", what)
        idx = 0
    if not isinstance(idx, int) or not 0 <= idx < n:
        logger.warning("%s: invalid event_idx %r dropped", what, item.get("event_idx"))
        return None
    return idx


_ITEM_TEXT_KEYS = ("name", "text", "label", "value", "id")


def _item_text(value: Any) -> list[str]:
    """Flatten one LLM list item into strings: str → [str]; dict → its name/text/label/value/id (or first string
    value); list/tuple → flattened items; number → [str]; None/bool/empty → [].
    把 LLM 列表中的一个条目展平为字符串：str → [str]；dict → 其 name/text/label/value/id（或首个字符串值）；
    list/tuple → 展平各项；数字 → [str]；None/bool/空 → []。
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        for key in _ITEM_TEXT_KEYS:
            if isinstance(value.get(key), str) and value[key].strip():
                return [value[key].strip()]
        return next(([v.strip()] for v in value.values() if isinstance(v, str) and v.strip()), [])
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _item_text(v)]
    return [str(value)]


def _str_list(value: Any, *, what: str, field: str) -> list[str]:
    """Coerce an LLM list field: None → [], a string → [string], list/tuple → flattened string items (dict items
    yield their name/text, nested lists are flattened, None/empties dropped, duplicates removed); a scalar or a dict
    where the list belongs is a malformed type and raises TypeError (the caller skips the item).
    规整 LLM 的列表字段：None → []，字符串 → [字符串]，list/tuple → 展平后的字符串项（dict 项取其 name/text，
    嵌套列表展平，丢弃 None/空项与重复项）；列表位置出现标量或 dict 属于类型畸形，抛 TypeError（调用方跳过该条目）。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for text in (t for v in value for t in _item_text(v)):
            if text not in out:
                out.append(text)
        return out
    raise TypeError(f"{what}: {field} must be a list of strings, got {type(value).__name__}")


_TAG_SPLIT_RE = re.compile(r"[,;，；]")


def _tags(value: Any, *, what: str) -> list[str]:
    """Topic tags: _str_list, then a single 'a, b; c' string is split on commas / semicolons.
    主题标签：先 _str_list，再把单个 'a, b; c' 形式的字符串按逗号 / 分号切分。
    """
    out: list[str] = []
    for tag in _str_list(value, what=what, field="topic_tags"):
        for part in _TAG_SPLIT_RE.split(tag):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def _str_field(value: Any, *, what: str, field: str, default: str = "") -> str:
    """Coerce an LLM text field: None or '' → default, a string → itself; any other type → TypeError.
    规整 LLM 的文本字段：None 或 '' → default，字符串 → 原样；其它类型抛 TypeError。
    """
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value
    raise TypeError(f"{what}: {field} must be a string, got {type(value).__name__}")


def _min_visibility(*values: str) -> str:
    """Most restrictive visibility among the values (an out-of-vocabulary value is ignored, i.e. the event's wins).
    各值中最严格的可见性（词表外的值被忽略，即以事件的可见性为准）。
    """
    best = Visibility.public.value
    for value in values:
        if str(value) in Visibility.values() and rank(str(value)) < rank(best):
            best = str(value)
    return best


_DATE_SEPARATOR_RE = re.compile(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})")


def _date(value: Any, default: str, *, what: str = "extraction") -> str:
    """Coerce an LLM date to YYYY-MM-DD ('YYYY/MM/DD' and 'YYYY.MM.DD' separators accepted), falling back to default
    with a DEBUG line so the replacement is observable.
    将 LLM 给出的日期规整为 YYYY-MM-DD（接受 'YYYY/MM/DD' 与 'YYYY.MM.DD' 分隔符），失败时回退默认值并记一条 DEBUG 日志，
    使替换可观察。
    """
    if value is None or value == "" or isinstance(value, bool):
        return default
    text = _DATE_SEPARATOR_RE.sub(lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", str(value))
    try:
        return date_of(text)
    except (ValidationError, ValueError, TypeError):
        logger.debug("%s: unreadable date_observed %r replaced by %s", what, value, default)
        return default


def _datetime(value: Any, *, what: str, field: str) -> str | None:
    """Coerce an LLM instant to the UTC ISO form: an ISO datetime or a date-only string (→ T00:00:00+00:00) is
    accepted; None/''/anything else (a bare year, free text, a number) → None with a DEBUG line. Numbers are never
    read as epoch seconds here: an LLM writing 2024 means a year, and the epoch reading stored 1970 on the atom.
    把 LLM 给出的时刻规整为 UTC ISO 形式：接受 ISO 日期时间或仅日期字符串（→ T00:00:00+00:00）；
    None/''/其它（裸年份、自由文本、数字）→ None 并记一条 DEBUG 日志。这里绝不把数字当 epoch 秒：
    LLM 写 2024 指的是年份，而 epoch 读法会把 1970 存到原子上。
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            return to_utc_iso(value)
        except (ValidationError, ValueError, TypeError):
            pass
    logger.debug("%s: unreadable %s %r dropped", what, field, value)
    return None


_CONFIDENCE_WORDS = (("high", 0.9), ("medium", 0.7), ("moderate", 0.7), ("low", 0.4))


def _prob(value: Any, *, default: float = 0.7, what: str = "extraction", field: str = "confidence") -> float:
    """Coerce an LLM confidence to [0, 1] without changing its meaning: a number in [0, 1] as is; numeric strings
    parsed; an explicit percentage ('85%') and a bare number in (10, 100] read as a percentage ('85', 85 → 0.85);
    high/medium/low → 0.9/0.7/0.4; None/''/bool → default silently; a non-numeric word → default with a DEBUG line.
    Any other number — negative, in (1, 10] (a 1–5 or 1–10 rating, or a typo like 1.2), above 100, NaN — is
    neither a probability nor a percentage: it becomes `default` with a WARNING naming the field, never a value near
    0 that the prune phase would later archive as decayed.
    把 LLM 的置信度规整到 [0, 1] 且不改变其含义：[0, 1] 内的数原样；解析数字串；显式百分比（'85%'）与 (10, 100] 内的
    裸数字按百分比读（'85'、85 → 0.85）；high/medium/low → 0.9/0.7/0.4；None/''/布尔 → 静默取 default；非数字的词 →
    default 并记一条 DEBUG。其它任何数——负数、(1, 10] 内（1–5 或 1–10 评分，或 1.2 这样的笔误）、大于 100、NaN——
    既不是概率也不是百分比：取 `default` 并记一条点名字段的 WARNING，绝不变成接近 0、随后被 prune 阶段当作衰减归档的值。
    """
    if value is None or value == "" or isinstance(value, bool):
        return default
    number: float | None = None
    percent = False
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        percent = text.endswith("%")
        try:
            number = float(text.rstrip("%").strip())
        except ValueError:
            word = next((score for word, score in _CONFIDENCE_WORDS if word in text), None)
            if word is None:
                logger.debug("%s: non-numeric %s %r; default %s", what, field, value, default)
                return default
            return word
    if number is None:
        logger.debug("%s: %s %r is not a number; default %s", what, field, value, default)
        return default
    if number == number:
        if percent or 10.0 < number <= 100.0:
            number /= 100.0
        if 0.0 <= number <= 1.0:
            return number
    logger.warning("%s: %s %r is neither a probability nor a percentage; default %s", what, field, value, default)
    return default


# Words that negate the member they precede ('not positive', 'non-habit'): folding such a value to the member it
# names would store its opposite, so it falls to the default instead.
# 否定其后成员的词（'not positive'、'non-habit'）：把这种值折叠到它点名的成员会存成相反含义，因此改取默认值。
_NEGATION_WORDS = frozenset({"not", "non", "no", "never", "isnt", "isn't", "aint", "ain't", "neither", "nor", "without"})


def _contains_words(tokens: list[str], words: list[str]) -> bool:
    """Whether `words` occur as a contiguous run of whole tokens in `tokens` ('very positive' ⊃ 'positive';
    'artifact' ⊅ 'fact', 'inhabit' ⊅ 'habit').
    `words` 是否作为整词的连续片段出现在 `tokens` 中（'very positive' ⊃ 'positive'；'artifact' ⊅ 'fact'，
    'inhabit' ⊅ 'habit'）。
    """
    n = len(words)
    return n > 0 and any(tokens[i : i + n] == words for i in range(len(tokens) - n + 1))


def _enum_value(
    value: Any, enum: type[StrEnum], *, default: str, aliases: dict[str, str], what: str, field: str
) -> str:
    """Coerce an LLM enum field to a member value: lower-cased, stripped, spaces/hyphens → '_'; exact member, alias,
    a member present as whole words in the text ('very positive'; never a substring: 'artifact' is not 'fact') or a
    prefix of one ('preference'); None/'' → default silently. A negated value ('not positive', 'non-habit',
    'not a constraint') never folds to the member it names — that would store the opposite — but to the default
    with a WARNING. Every other fold is traced with a DEBUG line (not WARNING: it is per item and would flood the
    log); an unknown value → default with a DEBUG line.
    把 LLM 的枚举字段规整为成员值：小写、去空白、空格/连字符 → '_'；依次尝试精确成员、别名、以整词出现在文本中的成员
    （'very positive'；绝不按子串：'artifact' 不是 'fact'）、成员的前缀（'preference'）；None/'' → 静默取 default。
    被否定的值（'not positive'、'non-habit'、'not a constraint'）绝不折叠到它点名的成员——那会存成相反含义——
    而是取 default 并记一条 WARNING。其它每一次折叠都记一条 DEBUG（不用 WARNING：逐条目记录会淹没日志）；
    未知值 → default 并记一条 DEBUG。
    """
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if isinstance(v, str) and v.strip()), "")
        if not value:
            return default
    text = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    values = enum.values()
    if text in values:
        return text
    if text in aliases:
        logger.debug("%s: %s %r read as %r", what, field, value, aliases[text])
        return aliases[text]
    tokens = [t for t in text.split("_") if t]
    if any(t in _NEGATION_WORDS for t in tokens):
        logger.warning("%s: negated %s %r cannot be folded to a member; %r", what, field, value, default)
        return default
    for member in sorted(values, key=len, reverse=True):
        if _contains_words(tokens, member.split("_")):
            logger.debug("%s: %s %r contains %r; folded", what, field, value, member)
            return member
    for member in values:
        if len(text) >= 3 and member.startswith(text):
            logger.debug("%s: %s %r is a prefix of %r; folded", what, field, value, member)
            return member
    logger.debug("%s: unknown %s %r coerced to %r", what, field, value, default)
    return default


_TYPE_ALIASES = {
    "preference": "preference_event", "preferences": "preference_event", "like": "preference_event",
    "likes": "preference_event", "dislike": "preference_event", "dislikes": "preference_event",
    "taste": "preference_event", "routine": "habit", "habits": "habit", "pattern": "habit", "rule": "constraint",
    "rules": "constraint", "requirement": "constraint", "restriction": "constraint", "must": "constraint",
    "must_not": "constraint", "temporal": "temporal_fact", "dated": "temporal_fact", "event": "temporal_fact",
    "schedule": "temporal_fact", "date": "temporal_fact", "plan": "fact", "goal": "fact", "intent": "fact",
    "opinion": "fact", "belief": "fact", "statement": "fact", "facts": "fact", "attribute": "fact",
}  # fmt: skip
_RELATION_ALIASES = {
    "update": "supersession", "updated": "supersession", "revision": "supersession", "revise": "supersession",
    "change": "supersession", "changed": "supersession", "replace": "supersession", "replaces": "supersession",
    "supersede": "supersession", "supersedes": "supersession", "switch": "supersession", "compare": "comparison",
    "compared": "comparison", "prefers_over": "comparison", "vs": "comparison", "versus": "comparison",
    "routine": "habit", "pattern": "habit", "rule": "constraint", "requirement": "constraint", "must": "constraint",
    "restriction": "constraint", "single": "single_fact", "fact": "single_fact", "statement": "single_fact",
    "none": "single_fact", "simple": "single_fact",
}  # fmt: skip
_SENTIMENT_ALIASES = {
    "pos": "positive", "happy": "positive", "good": "positive", "like": "positive", "likes": "positive",
    "joy": "positive", "excited": "positive", "neg": "negative", "bad": "negative", "dislike": "negative",
    "dislikes": "negative", "angry": "negative", "sad": "negative", "frustrated": "negative", "upset": "negative",
    "ambivalent": "mixed", "conflicted": "mixed", "none": "unknown", "n/a": "unknown", "na": "unknown",
    "null": "unknown", "unclear": "unknown",
}  # fmt: skip
_REWARD_ALIASES = {
    "success": "positive", "succeeded": "positive", "pass": "positive", "passed": "positive", "works": "positive",
    "worked": "positive", "ok": "positive", "good": "positive", "fixed": "positive", "win": "positive",
    "fail": "negative", "failed": "negative", "failure": "negative", "broken": "negative", "error": "negative",
    "bad": "negative", "regression": "negative", "mixed": "neutral", "partial": "neutral", "none": "unknown",
    "n/a": "unknown", "na": "unknown", "null": "unknown",
}  # fmt: skip
_DECISION_ALIASES = {
    "UPDATE": "REVISE", "UPDATED": "REVISE", "REPLACE": "REVISE", "SUPERSEDE": "REVISE", "SUPERSEDES": "REVISE",
    "MERGE": "REVISE", "REVISED": "REVISE", "REVISION": "REVISE", "DUPLICATE": "NONE", "DUP": "NONE", "SAME": "NONE",
    "CONTRADICT": "CONFLICT", "CONTRADICTS": "CONFLICT", "CONFLICTING": "CONFLICT", "CONFLICTS": "CONFLICT",
    "KEEP": "ADD", "NEW": "ADD", "INSERT": "ADD", "ADDED": "ADD", "CREATE": "ADD", "ADDITIVE": "ADD",
}  # fmt: skip
# SKIP / NOOP / IGNORE are deliberately absent: they do not say "duplicate of the target", and NONE would discard the
# new observation; they stay unknown verdicts, which store the observation as ADD with a warning.
# 刻意不收录 SKIP / NOOP / IGNORE：它们并不表示"与目标重复"，而 NONE 会丢弃新观察；它们保持为未知裁决，
# 即按 ADD 存储观察并记一条警告。
# Words an LLM writes in `attributed_to` that name a side rather than a person.
# LLM 写在 `attributed_to` 里、指代侧别而非人物的词。
_ASSISTANT_SIDE_WORDS = frozenset({"assistant", "system", "tool", "bot", "ai", "model", "agent", "llm"})
_USER_SIDE_WORDS = frozenset({"user", "human", "me", "i", "self", "the_user"})
# Collective or placeholder words that are not a speaker's name (they would otherwise become entity mentions).
# 不是说话人名字的集体词或占位词（否则会变成实体提及）。
_GENERIC_SPEAKERS = frozenset(
    {"group", "others", "other", "everyone", "all", "unknown", "none", "n/a", "na", "null", "people", "someone",
     "speaker", "speakers", "participant", "participants", "multiple", "various", "anonymous"}
)  # fmt: skip


def _attributed_to(value: Any, *, role: str) -> tuple[str, str | None]:
    """Normalise an LLM `attributed_to`: ('user'|'assistant', None) for a legal side (case-insensitive; system / tool
    / bot / ai / model count as the assistant side, human / me as the user side); a collective word (group, others)
    or a non-string is no speaker; any other non-empty string is the side derived from the message role plus that
    string as the speaker name to keep; empty → (role side, None).
    规整 LLM 的 `attributed_to`：合法侧别（不分大小写；system / tool / bot / ai / model 视为 assistant 侧，
    human / me 视为 user 侧）返回 ('user'|'assistant', None)；集体词（group、others）或非字符串不算说话人；
    其它非空字符串返回按消息角色推得的侧别，并把该字符串作为要保留的说话人名；为空 → (角色侧别, None)。
    """
    side = "assistant" if role == "assistant" else "user"
    if value is None or value == "" or not isinstance(value, str):
        return side, None
    text = value.strip()
    low = re.sub(r"\s+", "_", text.lower())
    if low in ATTRIBUTED_TO_VALUES:
        return low, None
    if low in _ASSISTANT_SIDE_WORDS:
        return "assistant", None
    if low in _USER_SIDE_WORDS or low in _GENERIC_SPEAKERS:
        return side, None
    return side, (text or None)


def _speaker(value: Any) -> str | None:
    """An explicit LLM `speaker`: a non-empty string that is neither a side word nor a collective word; else None.
    LLM 显式给出的 `speaker`：既不是侧别词也不是集体词的非空字符串；否则为 None。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    low = re.sub(r"\s+", "_", text.lower())
    if low in ATTRIBUTED_TO_VALUES or low in _ASSISTANT_SIDE_WORDS or low in _USER_SIDE_WORDS:
        return None
    if low in _GENERIC_SPEAKERS:
        return None
    return text


def _dict_items(items: Any) -> list[dict]:
    """The dict items of an LLM list reply; a non-list reply (or a missing root) is an empty list.
    LLM 列表回复中的字典条目；非列表回复（或缺失的根）视为空列表。
    """
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _root_items(raw: dict, root: str, *, what: str) -> list[dict]:
    """The dict items under one root of a composite reply. A single item dict where the list belongs is read as a
    one-item list; a dict keyed by item id becomes items carrying that id; any other non-list root logs one warning.
    复合回复某个根下的字典条目。列表位置只有一个条目 dict 时按单元素列表读取；以条目 id 为键的 dict 变成带该 id 的条目；
    其它非列表根记一条警告。
    """
    items = raw.get(root)
    if items is None or isinstance(items, list):
        return _dict_items(items)
    if isinstance(items, dict):
        if any(isinstance(v, dict) for v in items.values()) and not any(k in items for k in ("id", "text", "u")):
            logger.warning("%s: %s root is a dict keyed by id; reading its values as the items", what, root)
            return [{"id": k, **v} for k, v in items.items() if isinstance(v, dict)]
        logger.warning("%s: %s root is a single item; reading it as a one-item list", what, root)
        return [items]
    logger.warning("%s: %s root is not a list (%s); ignored", what, root, type(items).__name__)
    return []


def _observation_from_item(
    llm: LLMBase,
    item: dict,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    observation_date: str,
    scope: Scope,
    id_map: dict[str, str],
    what: str,
) -> Observation | None:
    """Coerce one observation item of an LLM reply into an Observation; None (with a warning) only when it cannot be
    witnessed (bad event_idx), has no text, or a field carries a malformed type (a dict / number where a string or
    list belongs). Vocabulary mismatches never skip: type / relation / sentiment fall back to fact / single_fact /
    unknown, confidence is rescaled and clamped, unreadable instants become None, and a named speaker is kept.
    把 LLM 回复中的一个观察条目规整为 Observation；只有无法见证（event_idx 非法）、没有文本、或字段类型畸形
    （字符串或列表位置出现 dict / 数字）时才返回 None（并记录警告）。词表不匹配绝不跳过：type / relation / sentiment
    回退为 fact / single_fact / unknown，confidence 重新缩放并夹紧，无法读取的时刻变为 None，具名说话人被保留。
    """
    idx = _event_index(item, len(events), what=what)
    if idx is None:
        return None
    ev, anc = events[idx]
    role = str(ev.role) if ev.role is not None else "user"
    # Field coercion runs inside the guard: a scalar where a list is expected, or a non-string text field, skips
    # the item with a warning instead of escaping HMS.add() as a raw TypeError or persisting a corrupt atom.
    # 字段规整放在守卫之内：列表位置出现标量、或文本字段不是字符串时，记录警告并跳过该条目，
    # 而不是以原生 TypeError 逃出 HMS.add()，也不会持久化为损坏的原子。
    try:
        # `attributed_to` is the conversation side the statement came from, never a person's name. Multi-speaker
        # transcripts make LLMs write the speaker ("Katrina") here; an explicit `speaker` key is the preferred
        # carrier. Either way the name is kept on Observation.speaker and as an entity mention, and the side falls
        # back to the witnessed message's role instead of rejecting the whole observation.
        # `attributed_to` 是陈述来自对话的哪一侧，永远不是人名。多说话人转录会让 LLM 在这里写说话人（"Katrina"）；
        # 显式的 `speaker` 键是更优的载体。两种来源的名字都保留在 Observation.speaker 与实体提及中，
        # 侧别回退为被见证消息的角色，而不是整条观察被拒绝。
        attributed_to, named = _attributed_to(item.get("attributed_to"), role=role)
        speaker = _speaker(item.get("speaker")) or named
        entity_mentions = _str_list(item.get("entity_mentions"), what=what, field="entity_mentions")
        if speaker and speaker.lower() not in {m.lower() for m in entity_mentions}:
            entity_mentions.append(speaker)
        obs_type = _enum_value(
            item.get("type"), ObservationType, default="fact", aliases=_TYPE_ALIASES, what=what, field="type"
        )
        date_observed = _date(item.get("date_observed"), observation_date, what=what)
        valid_from = _datetime(item.get("valid_from"), what=what, field="valid_from")
        valid_to = _datetime(item.get("valid_to"), what=what, field="valid_to")
        # MemoryAtom requires valid_to >= valid_from, and atom_from_observation fills a missing valid_from with the
        # atom's time anchor — date_observed midnight for a temporal_fact, the witnessing event's time otherwise;
        # an expiry earlier than that floor would fail the invariant outside the per-item guard and lose the whole
        # batch, so the contradictory expiry is dropped here, against the same floor atom_from_observation will use.
        # MemoryAtom 要求 valid_to >= valid_from，且 atom_from_observation 用原子的时间锚补缺失的 valid_from——
        # temporal_fact 用 date_observed 零点，其它用见证事件的时间；早于该下限的到期时刻会在逐条守卫之外触发
        # 该不变量并丢掉整个批次，因此矛盾的到期时刻在这里按 atom_from_observation 将使用的同一下限丢弃。
        if valid_from:
            floor = valid_from
        elif obs_type == ObservationType.temporal_fact.value:
            floor = date_observed + _MIDNIGHT
        else:
            floor = anc.timestamp
        if valid_to is not None and floor and valid_to < floor:
            logger.warning("%s: item %r valid_to %s precedes %s; valid_to dropped", what, item.get("id"), valid_to, floor)
            valid_to = None
        fields: dict[str, Any] = {
            "type": obs_type,
            "subject": _str_field(item.get("subject"), what=what, field="subject", default="user"),
            "predicate": _str_field(item.get("predicate"), what=what, field="predicate", default="states"),
            "objects": _str_list(item.get("objects"), what=what, field="objects"),
            "relation": _enum_value(
                item.get("relation"), Relation, default="single_fact", aliases=_RELATION_ALIASES, what=what,
                field="relation",
            ),
            "text": _str_field(item.get("text"), what=what, field="text").strip(),
            "date_observed": date_observed,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": _prob(item.get("confidence"), what=what, field="confidence"),
            "attributed_to": attributed_to,
            "speaker": speaker,
            "sentiment": _enum_value(
                item.get("sentiment"), Sentiment, default="unknown", aliases=_SENTIMENT_ALIASES, what=what,
                field="sentiment",
            ),
            "topic_tags": _tags(item.get("topic_tags"), what=what),
            "entity_mentions": entity_mentions,
            "supersedes_hint_ids": _map_refs(
                _str_list(item.get("supersedes_hint_ids"), what=what, field="supersedes_hint_ids"),
                id_map,
                what="supersedes_hint_ids",
            ),
            "extracted_by": llm.identity,
            "visibility": ev.visibility,
            "scope": scope,
            "anchor_ids": [anc.anchor_id],
        }  # fmt: skip
        return Observation(**fields)
    except (pydantic.ValidationError, ValueError, TypeError) as exc:
        logger.warning("%s: item %r skipped: %s", what, item.get("id"), exc)
        return None


def _check_output(output: str, *, staged: bool = False) -> str:
    """Validate an extraction_output value and return it; VALIDATION_013 for an unknown value, and for 'lines' on the staged path (the lines contract has no observation-only prompt).
    校验 extraction_output 的值并原样返回；未知值抛 VALIDATION_013，分段路径上的 'lines' 同样抛出（行式契约没有只抽观察的提示词）。
    """
    if output not in EXTRACTION_OUTPUTS:
        raise ValidationError(
            f"unknown extraction_output {output!r}",
            "VALIDATION_013",
            details={"extraction_output": output, "supported": list(EXTRACTION_OUTPUTS)},
        )
    if staged and output == "lines":
        raise ValidationError(
            "extraction_output='lines' supports extraction_mode='single' only (CONTRACT §6.6) / "
            "extraction_output='lines' 只支持 extraction_mode='single'（CONTRACT §6.6）",
            "VALIDATION_013",
            details={"extraction_output": output, "extraction_mode": "staged", "supported": ["single"]},
        )
    return output


def _entities_hint(item: dict, *, what: str) -> list[str]:
    """The compact `entities` hint through _str_list (None → [], a string → one item, dict items → their text); a
    malformed shape (a number or a dict where the list belongs) is ignored with a warning instead of skipping the
    item — the hint is optional and the derivation always has extractor(text) ∪ speaker to fall back on.
    紧凑契约的 `entities` 提示经 _str_list 规整（None → []，字符串 → 单元素，dict 项取其文本）；形状畸形（列表位置出现
    数字或 dict）时记一条警告并忽略，而不是跳过该条目——提示是可选的，推导总有 extractor(text) ∪ speaker 可回退。
    """
    try:
        return _str_list(item.get("entities"), what=what, field="entities")
    except TypeError as exc:
        logger.warning("%s: item %r entities ignored: %s", what, item.get("id"), exc)
        return []


def _compact_observation_from_item(
    llm: LLMBase,
    item: dict,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    observation_date: str,
    scope: Scope,
    id_map: dict[str, str],
    extractor: EntityExtractorBase | None,
    what: str,
    speaker_from_subject: bool = False,
    zh_rules: bool = False,
) -> Observation | None:
    """Coerce one compact-contract item: witness it first (the same event_idx rule as the full contract), expand it
    through ingest.derive into the full item shape from the witnessed message's role and date (the `entities` hint
    coerced by _entities_hint joins the derived entity_mentions), then run the exact full-contract coercion — every
    derived value passes _enum_value / _prob / _datetime / _date, so vocabulary never rejects, and `date_observed`
    falls back to the witnessed message's date rather than the batch date. With `speaker_from_subject` (the lines
    contract, §6.6, which has no speaker slot) a derived subject that is not a conversation side — the leading
    proper name of the statement — becomes the speaker when the item names none, so it reaches entity_mentions and
    actor_id exactly as a compact `speaker` does.
    规整一个紧凑契约条目：先见证（与完整契约相同的 event_idx 规则），再经 ingest.derive 依据被见证消息的角色与日期
    扩展为完整条目形状（经 _entities_hint 规整的 `entities` 提示并入推导出的 entity_mentions），最后走一模一样的完整
    契约规整——每个推导值都经过 _enum_value / _prob / _datetime / _date，因此词表绝不拒绝；`date_observed` 回退到
    被见证消息的日期而不是批次日期。带 `speaker_from_subject`（§6.6 的行式契约，没有说话人槽位）时，推导出的主语若
    不是对话侧别——即陈述开头的专有名——在条目未点名说话人时成为说话人，从而与紧凑契约的 `speaker` 一样到达
    entity_mentions 与 actor_id。`zh_rules` 透传给 ingest.derive（中文陈述规则，IngestConfig.zh_rules）。
    """
    idx = _event_index(item, len(events), what=what)
    if idx is None:
        return None
    ev, _anc = events[idx]
    role = str(ev.role) if ev.role is not None else "user"
    try:
        message_date = date_of(ev.timestamp)
    except (ValidationError, ValueError, TypeError):
        message_date = observation_date
    obs_type = _enum_value(
        item.get("type"), ObservationType, default="fact", aliases=_TYPE_ALIASES, what=what, field="type"
    )
    full = expand_compact_item(
        item, obs_type=obs_type, speaker=_speaker(item.get("speaker")), role=role, message_date=message_date,
        extractor=extractor, entities=_entities_hint(item, what=what), zh_rules=zh_rules,
    )  # fmt: skip
    full["event_idx"] = idx
    if speaker_from_subject and not full.get("speaker") and full.get("subject") not in SIDE_SUBJECTS:
        full["speaker"] = full["subject"]
    return _observation_from_item(
        llm, full, events=events, observation_date=message_date, scope=scope, id_map=id_map, what=what
    )


def _kappa_dict(value: Any, *, what: str) -> dict:
    """The κ object of an EP item: a dict as is; a list is a constraint list; a non-empty string is one constraint;
    anything else → {} with a DEBUG line.
    EP 条目的 κ 对象：dict 原样；列表视为约束列表；非空字符串视为一条约束；其它 → {} 并记一条 DEBUG 日志。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {"constraints": list(value)}
    if isinstance(value, str) and value.strip():
        return {"constraints": [value.strip()]}
    if value not in (None, ""):
        logger.debug("%s: kappa %r is not an object; ignored", what, value)
    return {}


def _ep_ref(ref: Any) -> str:
    """An EP ref as the prompt spells it: a bare index (0, '0') becomes 'e0'.
    按提示词写法的 EP 引用：裸索引（0、'0'）变为 'e0'。
    """
    text = str(ref).strip()
    return f"e{text}" if text.isdigit() else text


def _ep_from_item(
    llm: LLMBase,
    item: dict,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    ep_map: dict[str, str],
    scope: Scope,
    what: str,
) -> EP | None:
    """Coerce one EP item of an LLM reply into an EP; None (with a warning) only when it cannot be witnessed, a slot
    is not a string, or the closure (u, a, and y or a constraint) is missing. Vocabulary mismatches never skip: r
    falls back to unknown, an unreadable time_anchor to the event time, confidence is rescaled and clamped, an
    out-of-vocabulary κ.visibility to the event's visibility.
    把 LLM 回复中的一个 EP 条目规整为 EP；只有无法见证、槽位不是字符串、或闭包（u、a 及 y 或约束）缺失时才返回 None
    （并记录警告）。词表不匹配绝不跳过：r 回退为 unknown，无法读取的 time_anchor 回退为事件时间，confidence 重新缩放并夹紧，
    词表外的 κ.visibility 回退为事件的可见性。
    """
    idx = _event_index(item, len(events), what=what)
    if idx is None:
        return None
    ev, anc = events[idx]
    raw_kappa = _kappa_dict(item.get("kappa"), what=what)
    # Same coercion discipline as the observation builder: a string constraint list is one constraint, never its
    # characters, and a non-string slot skips the item (so a character list can never satisfy Slots._closure).
    # 与观察构造器相同的规整纪律：字符串形式的约束列表是一条约束而非其逐个字符，
    # 槽位不是字符串则跳过该条目（字符列表因此永远无法满足 Slots._closure）。
    try:
        kappa = Kappa(
            constraints=_str_list(raw_kappa.get("constraints"), what=what, field="kappa.constraints"),
            visibility=_min_visibility(str(raw_kappa.get("visibility") or ev.visibility), str(ev.visibility)),
            evidence=[anc.anchor_id],
        )
        slots = Slots(
            u=_str_field(item.get("u"), what=what, field="u").strip(),
            b=_str_field(item.get("b"), what=what, field="b"),
            a=_str_field(item.get("a"), what=what, field="a").strip(),
            y=_str_field(item.get("y"), what=what, field="y"),
            r=_enum_value(item.get("r"), Reward, default="unknown", aliases=_REWARD_ALIASES, what=what, field="r"),
            kappa=kappa,
        )
        return EP(
            slots=slots,
            time_anchor=_datetime(item.get("time_anchor"), what=what, field="time_anchor") or ev.timestamp,
            entity_mentions=_str_list(item.get("entity_mentions"), what=what, field="entity_mentions"),
            topic_tags=_tags(item.get("topic_tags"), what=what),
            uses_ep_ids=_map_refs(
                [_ep_ref(r) for r in _str_list(item.get("uses_ep_ids"), what=what, field="uses_ep_ids")],
                ep_map,
                what=f"{what}.uses_ep_ids",
            ),
            confidence=_prob(item.get("confidence"), what=what, field="confidence"),
            visibility=kappa.visibility,
            extracted_by=llm.identity,
            scope=scope,
            anchor_ids=[anc.anchor_id],
        )
    except (pydantic.ValidationError, ValueError, TypeError) as exc:
        logger.warning("%s: item %r skipped: %s", what, item.get("id"), exc)
        return None


def _new_messages(events: list[tuple[SourceEvent, Anchor]]) -> list[dict]:
    """The `New Messages` prompt rows of the observation-routed events.
    分流到观察的事件在提示词 `New Messages` 中的行。
    """
    return [
        {"idx": i, "role": str(ev.role), "text": str((ev.payload or {}).get("text") or ""), "timestamp": ev.timestamp}
        for i, (ev, _anc) in enumerate(events)
    ]


def _last_messages(last_events: list[SourceEvent]) -> list[dict]:
    """The `Last k Messages` prompt rows.
    提示词 `Last k Messages` 的行。
    """
    return [
        {"role": str(ev.role) if ev.role is not None else "user", "text": str((ev.payload or {}).get("text") or "")}
        for ev in last_events
    ]


def _prompt_events(events: list[tuple[SourceEvent, Anchor]]) -> list[dict]:
    """The `Events` prompt rows of the state-routed events.
    分流到 state 的事件在提示词 `Events` 中的行。
    """
    return [
        {
            "idx": i,
            "role": str(ev.role) if ev.role is not None else "user",
            "object_type": str(ev.object_type),
            "text": str((ev.payload or {}).get("text") or ""),
            "timestamp": ev.timestamp,
        }
        for i, (ev, _anc) in enumerate(events)
    ]


def _prompt_recent_eps(recent_eps: list[EP]) -> list[dict]:
    """The `Recent EPs` prompt rows (`e0..` refs).
    提示词 `Recent EPs` 的行（`e0..` 引用）。
    """
    return [
        {"id": f"e{i}", "u": e.slots.u, "a": e.slots.a, "y": e.slots.y, "r": str(e.slots.r)}
        for i, e in enumerate(recent_eps)
    ]


def extract_observations(
    llm: LLMBase,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    existing: list[dict],
    last_events: list[SourceEvent],
    observation_date: str,
    current_date: str,
    custom_instructions: str | None,
    scope: Scope,
    id_map: dict[str, str],
    output: str = "full",
    entity_extractor: EntityExtractorBase | None = None,
    zh_rules: bool = False,
) -> list[Observation]:
    """One observation_extraction call over the new messages; each item becomes an Observation witnessed by its message anchor.
    对新消息执行一次 observation_extraction 调用；每个条目成为由其消息锚点见证的 Observation。

    Input / 输入：
        llm: The LLM provider.
        llm：LLM 提供者。

        events: `[(message event, anchor)]` in order (their positions are the prompt's idx values).
        events：有序的 `[(消息事件, 锚点)]`（其位置即提示词中的 idx）。

        existing: Neighbourhood atoms `[{'id', 'text', ...}]` (only id/text reach the prompt).
        existing：邻域原子 `[{'id', 'text', ...}]`（只有 id/text 进入提示词）。

        last_events: Recent message events rendered as context lines.
        last_events：渲染为上下文行的近期消息事件。

        observation_date: date_of(last new message timestamp).
        observation_date：date_of(最后一条新消息的时间戳)。

        current_date: date_of(store.now()).
        current_date：date_of(store.now())。

        custom_instructions: Optional caller guidance.
        custom_instructions：可选的调用方指导。

        scope: Scope of the observations.
        scope：观察的范围。

        id_map: Synthetic existing id → atom_id (resolves supersedes_hint_ids).
        id_map：合成已有 id → atom_id（解析 supersedes_hint_ids）。

        output: 'full' (the observation_extraction contract) | 'compact' (observation_extraction_compact: the LLM emits COMPACT_OBSERVATION_KEYS only; the rest is derived locally, ingest.derive); 'lines' is rejected here (single mode only, §6.6).
        output：'full'（observation_extraction 契约）| 'compact'（observation_extraction_compact：LLM 只输出 COMPACT_OBSERVATION_KEYS，其余在本地推导，ingest.derive）；'lines' 在此被拒绝（仅 single 模式，§6.6）。

        entity_extractor: The configured entity extractor for the compact derivation of entity_mentions (None → rule); unused in 'full'.
        entity_extractor：紧凑模式推导 entity_mentions 用的实体抽取器（None → 规则）；'full' 下不使用。

        zh_rules: IngestConfig.zh_rules, forwarded to the compact derivation (Chinese statement rules); unused in 'full'.
        zh_rules：IngestConfig.zh_rules，透传给紧凑推导（中文陈述规则）；'full' 下不使用。

    Output / 输出：
        Validated Observations in extraction order; invalid items (bad event_idx, a non-list/non-string list field, a non-string text field, pydantic failure) are skipped with a warning.
        按抽取顺序的已校验 Observation；无效条目（event_idx 非法、列表字段既非列表也非字符串、文本字段不是字符串、pydantic 失败）记录警告并跳过。

    State effect / 状态影响：
        None; an unknown `output` raises ValidationError(VALIDATION_013).
        无；未知的 `output` 抛出 ValidationError(VALIDATION_013)。
    """
    compact = _check_output(output, staged=True) == "compact"
    user = build_observation_prompt(
        existing_memories=[{"id": str(e.get("id")), "text": str(e.get("text") or "")} for e in existing],
        new_messages=_new_messages(events),
        last_messages=_last_messages(last_events),
        observation_date=observation_date,
        current_date=current_date,
        custom_instructions=custom_instructions,
    )
    what = "observation_extraction_compact" if compact else "observation_extraction"
    system = OBSERVATION_EXTRACTION_COMPACT_PROMPT if compact else OBSERVATION_EXTRACTION_PROMPT
    items = llm.generate_json(what, system=system, user=user)
    return [
        obs
        for obs in (
            _compact_observation_from_item(
                llm, item, events=events, observation_date=observation_date, scope=scope, id_map=id_map,
                extractor=entity_extractor, what=what, zh_rules=zh_rules,
            )
            if compact
            else _observation_from_item(
                llm, item, events=events, observation_date=observation_date, scope=scope, id_map=id_map, what=what
            )
            for item in _dict_items(items)
        )
        if obs is not None
    ]  # fmt: skip


def extract_eps(
    llm: LLMBase,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    known_constraints: list[MemoryAtom],
    recent_eps: list[EP],
    observation_date: str,
    scope: Scope,
) -> list[EP]:
    """One ep_extraction call over the state-routed messages; each item becomes an EP witnessed by its message anchor.
    对分流到 state 的消息执行一次 ep_extraction 调用；每个条目成为由其消息锚点见证的 EP。

    Input / 输入：
        llm: The LLM provider.
        llm：LLM 提供者。

        events: `[(message event, anchor)]` routed to 'state'.
        events：分流到 'state' 的 `[(消息事件, 锚点)]`。

        known_constraints: Valid constraint atoms in scope (`c0..` refs).
        known_constraints：范围内有效的约束原子（`c0..` 引用）。

        recent_eps: Recent valid EPs in scope (`e0..` refs, resolved into uses_ep_ids).
        recent_eps：范围内近期有效的 EP（`e0..` 引用，解析为 uses_ep_ids）。

        observation_date: YYYY-MM-DD of the batch.
        observation_date：批次的 YYYY-MM-DD。

        scope: Scope of the EPs.
        scope：EP 的范围。

    Output / 输出：
        Validated EPs (κ.evidence = [anchor], anchor_ids = [anchor], time_anchor = item or event time); items whose list fields are neither lists nor strings, whose slots are not strings, or that fail pydantic are skipped with a warning.
        已校验的 EP（κ.evidence = [锚点]，anchor_ids = [锚点]，time_anchor 为条目或事件时间）；列表字段既非列表也非字符串、槽位不是字符串或 pydantic 失败的条目记录警告并跳过。

    State effect / 状态影响：
        None.
        无。
    """
    if not events:
        return []
    user = build_ep_prompt(
        events=_prompt_events(events),
        known_constraints=[{"id": f"c{i}", "text": c.text} for i, c in enumerate(known_constraints)],
        recent_eps=_prompt_recent_eps(recent_eps),
        observation_date=observation_date,
    )
    ep_map = {f"e{i}": e.ep_id for i, e in enumerate(recent_eps) if e.ep_id}
    items = llm.generate_json("ep_extraction", system=EP_EXTRACTION_PROMPT, user=user)
    return [
        ep
        for ep in (
            _ep_from_item(llm, item, events=events, ep_map=ep_map, scope=scope, what="ep_extraction")
            for item in _dict_items(items)
        )
        if ep is not None
    ]


def _candidate_dict(atom: MemoryAtom, ref: str) -> dict:
    """Prompt shape of a revision candidate.
    修订候选在提示词中的形状。
    """
    content = atom.content if isinstance(atom.content, dict) else {}
    return {
        "id": ref,
        "type": str(atom.obs_type) if atom.obs_type else None,
        "subject": atom.subject,
        "predicate": atom.predicate,
        "objects": list(content.get("objects") or []),
        "relation": content.get("relation") or "single_fact",
        "text": atom.text,
        "date_observed": content.get("date_observed") or (date_of(atom.time_anchor) if atom.time_anchor else None),
        "valid_from": atom.valid_from,
    }


def decide_revisions(
    store: StateStore,
    llm: LLMBase,
    *,
    new: list[tuple[Observation, MemoryAtom]],
    scope: Scope,
    current_date: str,
    revision_mode: str,
) -> list[tuple[Decision, str | None, str]]:
    """Decide ADD / REVISE / CONFLICT / NONE per new observation atom against its revision candidates (one batched LLM call).
    针对修订候选为每个新观察原子决定 ADD / REVISE / CONFLICT / NONE（一次批量 LLM 调用）。

    Input / 输入：
        store: The state store (revision_candidates).
        store：状态存储（revision_candidates）。

        llm: The LLM provider.
        llm：LLM 提供者。

        new: `[(observation, atom)]` pairs in extraction order.
        new：按抽取顺序的 `[(观察, 原子)]` 对。

        scope: Scope searched for candidates.
        scope：搜索候选的范围。

        current_date: YYYY-MM-DD of the clock.
        current_date：时钟的 YYYY-MM-DD。

        revision_mode: 'inline' (decide now) | 'deferred' (ADD; consolidate decides) | 'off' (ADD).
        revision_mode：'inline'（立即决定）| 'deferred'（ADD；由 consolidate 决定）| 'off'（ADD）。

    Output / 输出：
        One (Decision, target atom_id or None, reason) per input pair, same order.
        每个输入对一个 (Decision, 目标 atom_id 或 None, 原因)，顺序一致。

    State effect / 状态影响：
        None (reads only).
        无（仅读取）。
    """
    results: list[tuple[Decision, str | None, str] | None] = [None] * len(new)
    pending: list[tuple[int, list[MemoryAtom]]] = []
    for position, (obs, atom) in enumerate(new):
        candidates = store.revision_candidates(
            scope,
            subject=obs.subject,
            predicate=obs.predicate,
            obs_type=str(obs.type),
            topics=list(obs.topic_tags),
            objects=list(obs.objects),
        )[:_REVISION_CANDIDATE_LIMIT]
        exact = next((c for c in candidates if c.hash == atom.hash), None)
        if exact is not None:
            results[position] = (Decision.NONE, exact.atom_id, "exact duplicate")
            continue
        if not candidates:
            results[position] = (Decision.ADD, None, "no candidates")
            continue
        if revision_mode == "off":
            results[position] = (Decision.ADD, None, "revision_mode=off")
            continue
        if revision_mode == "deferred":
            results[position] = (Decision.ADD, None, "revision_mode=deferred")
            continue
        pending.append((position, candidates))
    if pending:
        new_items: list[dict] = []
        candidate_refs: dict[str, str] = {}
        candidate_dicts: list[dict] = []
        own: dict[str, set[str]] = {}
        for position, candidates in pending:
            obs = new[position][0]
            new_id = str(position)
            new_items.append(
                {
                    "id": new_id,
                    "type": str(obs.type),
                    "subject": obs.subject,
                    "predicate": obs.predicate,
                    "objects": list(obs.objects),
                    "relation": str(obs.relation),
                    "text": obs.text,
                    "date_observed": obs.date_observed,
                }
            )
            own[new_id] = set()
            for candidate in candidates:
                ref = next((r for r, a in candidate_refs.items() if a == candidate.atom_id), None)
                if ref is None:
                    ref = str(len(candidate_dicts))
                    candidate_refs[ref] = candidate.atom_id
                    candidate_dicts.append(_candidate_dict(candidate, ref))
                own[new_id].add(ref)
        user = build_revision_prompt(new_items=new_items, candidates=candidate_dicts, current_date=current_date)
        decisions = llm.generate_json("revision_decision", system=REVISION_DECISION_PROMPT, user=user)
        by_new: dict[str, dict] = {}
        for decision in decisions if isinstance(decisions, list) else []:
            if isinstance(decision, dict) and decision.get("new_id") is not None:
                by_new.setdefault(str(decision["new_id"]), decision)
        for position, _candidates in pending:
            new_id = str(position)
            decision = by_new.get(new_id)
            if decision is None:
                logger.warning("revision_decision: no decision for new id %r; ADD", new_id)
                results[position] = (Decision.ADD, None, "no decision")
                continue
            verdict = _verdict(decision.get("decision"))
            reason = str(decision.get("reason") or "")
            if verdict not in Decision.values():
                logger.warning("revision_decision: unknown decision %r; ADD", verdict)
                results[position] = (Decision.ADD, None, f"unknown decision {verdict}")
                continue
            if verdict == Decision.ADD.value:
                results[position] = (Decision.ADD, None, reason or "additive")
                continue
            target_ref = decision.get("target_id")
            target = candidate_refs.get(str(target_ref)) if target_ref is not None else None
            if target is None or str(target_ref) not in own[new_id]:
                logger.warning("revision_decision: unknown target %r for new id %r; ADD", target_ref, new_id)
                results[position] = (Decision.ADD, None, f"unknown target {target_ref!r}")
                continue
            results[position] = (Decision(verdict), target, reason)
    return [r if r is not None else (Decision.ADD, None, "additive") for r in results]


@dataclass
class ExtractionBundle:
    """What one memory_extraction call produced: observations, EPs and one revision decision per observation.
    一次 memory_extraction 调用的产出：观察、EP 以及每个观察一个修订决策。
    """

    observations: list[Observation] = field(default_factory=list)
    eps: list[EP] = field(default_factory=list)
    # Parallel to `observations`: (Decision, target atom_id or None, reason).
    # 与 `observations` 平行：(Decision, 目标 atom_id 或 None, 原因)。
    decisions: list[tuple[Decision, str | None, str]] = field(default_factory=list)


def _verdict(value: Any, *, what: str = "revision_decision") -> str:
    """The upper-cased, stripped decision verdict with synonyms folded (UPDATE / MERGE → REVISE, DUPLICATE / DUP /
    SAME → NONE, CONTRADICTS → CONFLICT, KEEP / NEW → ADD); a one-item list yields its item; None / '' → 'ADD'.
    A synonym that folds to NONE is the only fold that discards the new observation, so it is traced with a
    WARNING; SKIP / NOOP / IGNORE are not synonyms of anything and come back unchanged (→ ADD with a warning later).
    大写并去空白的裁决，同义词折叠（UPDATE / MERGE → REVISE，DUPLICATE / DUP / SAME → NONE，CONTRADICTS → CONFLICT，
    KEEP / NEW → ADD）；单元素列表取其元素；None / '' → 'ADD'。折叠到 NONE 是唯一会丢弃新观察的折叠，
    因此记一条 WARNING；SKIP / NOOP / IGNORE 不是任何裁决的同义词，原样返回（随后 → ADD 并记警告）。
    """
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if v not in (None, "")), None)
    if value is None or value == "":
        return Decision.ADD.value
    text = re.sub(r"[\s\-]+", "_", str(value).strip().upper())
    verdict = _DECISION_ALIASES.get(text, text)
    if verdict == Decision.NONE.value and text != verdict:
        logger.warning("%s: verdict %r read as NONE (duplicate of its target); the observation is not stored", what, value)
    return verdict


def _head_tokens(value: str | None) -> set[str]:
    """The content tokens of a subject / predicate value, tokenised like the text after underscores become blanks
    (`lives_in` → {live}, "Shure's GF" → {shure, gf}, `user` → {} as a stopword).
    主语 / 谓词取值的内容词元，下划线换成空白后按文本同样的方式切分（`lives_in` → {live}，"Shure's GF" →
    {shure, gf}，`user` 作为停用词 → {}）。
    """
    return set(content_tokens((value or "").replace("_", " ")))


def statement_tokens(text: str, subject: str | None = None, predicate: str | None = None) -> set[str]:
    """The stemmed content tokens of a statement minus its head (the subject tokens and the predicate tokens).
    陈述的词干化内容词元减去其头部（主语词元与谓词词元）。

    The head is what a positionally paired verdict shares — the same speaker and the same reporting verb ("Jake
    said …") — so it carries no evidence that two statements state the same fact; the remainder (the value: object,
    place, time, quantity) does. Subject and predicate tokens are tokenised and stemmed like the text (underscores
    become blanks: `lives_in` → {live}); `user` is already a `content_tokens` stopword. None / '' removes nothing.
    头部正是按位置误配的裁决所共享的部分——同一说话人、同一转述动词（"Jake said …"）——因此不构成两条陈述同一事实的
    证据；其余部分（取值：宾语、地点、时间、数量）才是。主语与谓词词元同文本一样切分并词干化（下划线换成空白：
    `lives_in` → {live}）；`user` 本就是 `content_tokens` 的停用词。None / '' 不去除任何词元。

    Input / 输入：
        text: The statement text.
        text：陈述文本。

        subject: The atom's subject, removed from the set (None / '' → nothing removed).
        subject：原子的主语，从集合中去除（None / '' → 不去除）。

        predicate: The atom's predicate, removed from the set (None / '' → nothing removed).
        predicate：原子的谓词，从集合中去除（None / '' → 不去除）。

    Output / 输出：
        The token set.
        词元集合。

    State effect / 状态影响：
        None.
        无。
    """
    return set(content_tokens(text or "")) - _head_tokens(subject) - _head_tokens(predicate)


def verdict_similarity(
    new_text: str,
    old_text: str,
    *,
    new_subject: str | None = None,
    old_subject: str | None = None,
    new_predicate: str | None = None,
    old_predicate: str | None = None,
    with_shared: bool = False,
) -> float | tuple[float, int]:
    """Containment (overlap coefficient) of two statements' token sets after their head is removed: |A ∩ B| / min(|A|, |B|) over `statement_tokens`, each side's own subject taken out and the predicate taken out only when both sides carry the same predicate (case-insensitive, stripped).
    两条陈述去掉头部后的词元集合包含度（overlap coefficient）：对 `statement_tokens` 计算 |A ∩ B| / min(|A|, |B|)，每一侧去掉自己的主语，谓词只在双方谓词相同（不分大小写、去空白）时去掉。

    0.0 when either side has no token left. Containment rather than Jaccard because a legitimate NONE pairs a
    statement with a longer restatement (Jaccard penalises the length asymmetry). Removing the head rather than
    adding the subject (the 2026-09-11 first cut): with the subject counted, a four-token statement that shared only
    its speaker and a reporting verb with its target scored exactly 0.5 ("User has dogs." / "User has cats." with
    subject `user`; "Jake said there are two pots." / "Jake said it was put in the fridge."), so 112 index-paired
    NONE verdicts survived the 0.5 bar on the recorded EgoLife compact replies (80 at exactly 0.5, 94 sharing a
    reporting verb) and the guarded replay stopped at hit@20 151/204 against the force-ADD ceiling 154; with the head
    removed those pairs score 0.0, 10 NONE survive the guard (which also asks for two shared tokens, see
    `Ingestor._guard_verdict`) and the replay reaches 154 (`benchmarks/egolife/results/replay_diag2/summary.md`,
    G1v2). A differing predicate stays in both sets: it is discriminating content ("Jake asked …" against
    "Jake said …" share less).
    任一侧不剩词元时为 0.0。选包含度而非 Jaccard：合法的 NONE 把一条陈述与更长的复述配对（Jaccard 会惩罚长度差）。
    去掉头部而非补上主语（2026-09-11 的第一版）：主语计入时，一条只与目标共享说话人和一个转述动词的四词元陈述恰好得
    0.5（主语 `user` 的 "User has dogs." / "User has cats."；"Jake said there are two pots." / "Jake said it was put
    in the fridge."），记录的 EgoLife 紧凑回复上 112 条按下标配对的 NONE 越过 0.5 门槛（80 条恰为 0.5、94 条共享
    转述动词），守卫回放停在 hit@20 151/204，而强制 ADD 的上限是 154；去掉头部后这些配对得 0.0，只剩 10 条 NONE 过
    守卫（守卫还要求两个共同词元，见 `Ingestor._guard_verdict`），回放达到 154
    （`benchmarks/egolife/results/replay_diag2/summary.md`，G1v2）。谓词不同时两侧都保留：它是有区分度的内容
    （"Jake asked …" 与 "Jake said …" 共享得更少）。

    Input / 输入：
        new_text: The new statement's text.
        new_text：新陈述的文本。

        old_text: The target atom's text.
        old_text：目标原子的文本。

        new_subject: The new atom's subject (see statement_tokens).
        new_subject：新原子的主语（见 statement_tokens）。

        old_subject: The target atom's subject.
        old_subject：目标原子的主语。

        new_predicate: The new atom's predicate (removed only when equal to old_predicate).
        new_predicate：新原子的谓词（只在与 old_predicate 相同时去掉）。

        old_predicate: The target atom's predicate.
        old_predicate：目标原子的谓词。

        with_shared: True → also return the number of shared tokens |A ∩ B| (the guard's single-token rule).
        with_shared：为 True 时同时返回共享词元数 |A ∩ B|（守卫的单词元规则）。

    Output / 输出：
        A float in [0, 1]; with `with_shared`, the pair (similarity, |A ∩ B|).
        [0, 1] 内的浮点数；`with_shared` 时为 (相似度, |A ∩ B|) 二元组。

    State effect / 状态影响：
        None.
        无。
    """
    shared_predicate = (new_predicate or "").strip().lower()
    if not shared_predicate or shared_predicate != (old_predicate or "").strip().lower():
        new_predicate = old_predicate = None
    a = statement_tokens(new_text, new_subject, new_predicate)
    b = statement_tokens(old_text, old_subject, old_predicate)
    shared = len(a & b)
    similarity = shared / min(len(a), len(b)) if a and b else 0.0
    return (similarity, shared) if with_shared else similarity


def _parse_decision(
    decision: dict | None,
    *,
    obs_key: str,
    candidate_refs: dict[str, str],
    hint_ids: list[str],
    what: str,
) -> tuple[Decision, str | None, str]:
    """Resolve one decision item against the presented candidates. An unknown verdict is stored without revision
    (ADD; HMS has no verdict that keeps the observation AND takes no action other than ADD — NONE means 'duplicate,
    do not store'). A target that is not a candidate ref but is a candidate atom_id passes through; otherwise the
    observation's own resolved supersedes_hint_ids stand in when they name exactly one candidate; else ADD.
    对照呈现过的候选解析一个决策条目。未知裁决按不修订存储（ADD；HMS 没有既保留观察又不做任何动作的其它裁决——
    NONE 表示"重复，不存储"）。目标不是候选引用但是候选 atom_id 时直接通过；否则当观察自身解析出的
    supersedes_hint_ids 恰好指向一个候选时用它替代；再否则 ADD。
    """
    if decision is None:
        logger.warning("%s: no decision for observation %r; ADD", what, obs_key)
        return Decision.ADD, None, "no decision"
    verdict = _verdict(decision.get("decision"), what=what)
    reason = str(decision.get("reason") or "")
    if verdict not in Decision.values():
        logger.warning("%s: unknown decision %r for observation %r; ADD", what, verdict, obs_key)
        return Decision.ADD, None, f"unknown decision {verdict}"
    if verdict == Decision.ADD.value:
        return Decision.ADD, None, reason or "additive"
    target_ref = decision.get("target_id")
    if isinstance(target_ref, (list, tuple)):
        target_ref = next((v for v in target_ref if v not in (None, "")), None)
    target = candidate_refs.get(str(target_ref).strip()) if target_ref is not None else None
    if target is None and target_ref is not None and str(target_ref).strip() in candidate_refs.values():
        target = str(target_ref).strip()
    if target is None:
        hinted = [atom_id for atom_id in hint_ids if atom_id in candidate_refs.values()]
        if len(hinted) == 1:
            logger.warning(
                "%s: unknown target %r for observation %r; using its supersedes hint", what, target_ref, obs_key
            )
            return Decision(verdict), hinted[0], reason or "target from supersedes_hint_ids"
        logger.warning("%s: unknown target %r for observation %r; ADD", what, target_ref, obs_key)
        return Decision.ADD, None, f"unknown target {target_ref!r}"
    return Decision(verdict), target, reason


def extract_all(
    llm: LLMBase,
    *,
    events: list[tuple[SourceEvent, Anchor]],
    ep_events: list[tuple[SourceEvent, Anchor]],
    existing: list[dict],
    candidates: list[tuple[str, MemoryAtom]],
    last_events: list[SourceEvent],
    known_constraints: list[MemoryAtom],
    recent_eps: list[EP],
    observation_date: str,
    current_date: str,
    custom_instructions: str | None,
    scope: Scope,
    id_map: dict[str, str],
    output: str = "full",
    entity_extractor: EntityExtractorBase | None = None,
    zh_rules: bool = False,
) -> ExtractionBundle:
    """One memory_extraction call: observations + EPs + revision decisions from a single LLM round trip.
    一次 memory_extraction 调用：单次 LLM 往返同时得到观察、EP 与修订决策。

    Input / 输入：
        llm: The LLM provider.
        llm：LLM 提供者。

        events: `[(message event, anchor)]` routed to 'observation' (their positions are the observation idx values).
        events：分流到 'observation' 的 `[(消息事件, 锚点)]`（其位置即观察的 idx）。

        ep_events: `[(message event, anchor)]` routed to 'state' (the EP idx values; empty → no EPs asked for).
        ep_events：分流到 'state' 的 `[(消息事件, 锚点)]`（EP 的 idx；为空 → 不要求 EP）。

        existing: Neighbourhood atoms `[{'id', 'text', ...}]` (only id/text reach the prompt).
        existing：邻域原子 `[{'id', 'text', ...}]`（只有 id/text 进入提示词）。

        candidates: `[(synthetic id, atom)]` revision candidates shown under `Candidate Existing Memories`; decision targets must be among these ids.
        candidates：呈现在 `Candidate Existing Memories` 下的修订候选 `[(合成 id, 原子)]`；决策目标必须落在这些 id 中。

        last_events: Recent message events rendered as context lines.
        last_events：渲染为上下文行的近期消息事件。

        known_constraints: Valid constraint atoms in scope (`c0..` refs).
        known_constraints：范围内有效的约束原子（`c0..` 引用）。

        recent_eps: Recent valid EPs in scope (`e0..` refs, resolved into uses_ep_ids).
        recent_eps：范围内近期有效的 EP（`e0..` 引用，解析为 uses_ep_ids）。

        observation_date: date_of(last new message timestamp).
        observation_date：date_of(最后一条新消息的时间戳)。

        current_date: date_of(store.now()).
        current_date：date_of(store.now())。

        custom_instructions: Optional caller guidance.
        custom_instructions：可选的调用方指导。

        scope: Scope of the observations and EPs.
        scope：观察与 EP 的范围。

        id_map: Synthetic existing id → atom_id (resolves supersedes_hint_ids).
        id_map：合成已有 id → atom_id（解析 supersedes_hint_ids）。

        output: 'full' (memory_extraction) | 'compact' (memory_extraction_compact: observations carry COMPACT_OBSERVATION_KEYS only and the rest is derived locally; EPs and decisions keep their shapes, and an absent or empty decisions root with no candidates shown is the contract, not a defect — silent) | 'lines' (memory_extraction_lines, §6.6: the reply is tab-separated text through LLMBase.generate_lines and ingest.lines.parse_lines, then the compact item path with the leading proper name as speaker, EPs with entity_mentions / topic_tags / b derived locally by ingest.derive.expand_lines_ep; a missing DEC line is ADD, silently, candidates shown or not).
        output：'full'（memory_extraction）| 'compact'（memory_extraction_compact：观察只带 COMPACT_OBSERVATION_KEYS，其余在本地推导；EP 与决策形状不变；未呈现候选时缺失或为空的 decisions 根是契约本身而非缺陷——不出声）| 'lines'（memory_extraction_lines，§6.6：回复是 tab 分隔文本，经 LLMBase.generate_lines 与 ingest.lines.parse_lines，再走紧凑条目路径并以开头的专有名为说话人，EP 的 entity_mentions / topic_tags / b 由 ingest.derive.expand_lines_ep 在本地推导；缺少 DEC 行即 ADD，无论是否呈现候选都不出声）。

        entity_extractor: The configured entity extractor for the compact derivation of entity_mentions (None → rule); unused in 'full'.
        entity_extractor：紧凑模式推导 entity_mentions 用的实体抽取器（None → 规则）；'full' 下不使用。

        zh_rules: IngestConfig.zh_rules, forwarded to the compact / lines derivation (ingest.derive: Chinese statement rules, sentiment / topic lists, Chinese speaker); unused in 'full'.
        zh_rules：IngestConfig.zh_rules，透传给紧凑 / 行式推导（ingest.derive：中文陈述规则、情感 / 主题词表、中文说话人）；'full' 下不使用。

    Output / 输出：
        ExtractionBundle; malformed observation / EP items are skipped with a warning exactly as in the staged functions, and a missing or malformed decisions root, an unknown verdict or a target outside `candidates` degrades that observation to ADD with a warning.
        ExtractionBundle；畸形的观察 / EP 条目与分段函数一样记录警告并跳过；decisions 根缺失或畸形、未知裁决或目标不在 `candidates` 内时，该观察降级为 ADD 并记录警告。

    State effect / 状态影响：
        None; an unknown `output` raises ValidationError(VALIDATION_013).
        无；未知的 `output` 抛出 ValidationError(VALIDATION_013)。
    """
    output = _check_output(output)
    compact, lines = output in _DERIVED_OUTPUTS, output == "lines"
    what = "memory_extraction" if output == "full" else f"memory_extraction_{output}"
    system = {
        "full": MEMORY_EXTRACTION_PROMPT,
        "compact": MEMORY_EXTRACTION_COMPACT_PROMPT,
        "lines": MEMORY_EXTRACTION_LINES_PROMPT,
    }[output]
    user = build_memory_extraction_prompt(
        existing_memories=[{"id": str(e.get("id")), "text": str(e.get("text") or "")} for e in existing],
        new_messages=_new_messages(events),
        last_messages=_last_messages(last_events),
        observation_date=observation_date,
        current_date=current_date,
        custom_instructions=custom_instructions,
        events=_prompt_events(ep_events),
        known_constraints=[{"id": f"c{i}", "text": c.text} for i, c in enumerate(known_constraints)],
        recent_eps=_prompt_recent_eps(recent_eps),
        candidates=[_candidate_dict(atom, ref) for ref, atom in candidates],
    )
    ep_map = {f"e{i}": e.ep_id for i, e in enumerate(recent_eps) if e.ep_id}
    candidate_refs = {str(ref): atom.atom_id for ref, atom in candidates}
    if lines:
        # The lines contract is plain text: no JSON mode, no parse ladder; parse_lines yields the compact roots.
        # 行式契约是纯文本：不用 JSON 模式、不走解析阶梯；parse_lines 产出紧凑形状的各根。
        raw: Any = parse_lines(llm.generate_lines(what, system=system, user=user), what=what)
    else:
        # ROOT_KEY has no entry for the composite task, so generate_json hands back the whole object (key=None).
        # ROOT_KEY 没有复合任务的条目，因此 generate_json 返回整个对象（key=None）。
        raw = llm.generate_json(what, system=system, user=user)
    if isinstance(raw, list) and raw:
        # A non-empty bare list is the observations array without its envelope (seen with deepseek-flash on long
        # batches); reading it as {"observations": [...]} keeps the batch instead of dropping every window in it.
        # An empty list is the parser's "nothing parsed" value and falls through to the warning below.
        # 非空裸列表是没有信封的 observations 数组（deepseek-flash 在长批次上出现过）；按 {"observations": [...]}
        # 读取可保住该批次，而不是丢掉其中每一个窗口。空列表是解析器的"什么都没解析出"值，落到下面的告警。
        logger.warning("%s: reply is a JSON list; reading it as the observations array", what)
        raw = {"observations": raw}
    if not isinstance(raw, dict):
        logger.warning("%s: reply is not a JSON object (%s); nothing extracted", what, type(raw).__name__)
        raw = {}
    bundle = ExtractionBundle()
    keys: list[str] = []
    for position, item in enumerate(_root_items(raw, "observations", what=what)):
        if compact:
            obs = _compact_observation_from_item(
                llm, item, events=events, observation_date=observation_date, scope=scope, id_map=id_map,
                extractor=entity_extractor, what=f"{what}.observations", speaker_from_subject=lines, zh_rules=zh_rules,
            )  # fmt: skip
        else:
            obs = _observation_from_item(
                llm, item, events=events, observation_date=observation_date, scope=scope, id_map=id_map,
                what=f"{what}.observations",
            )  # fmt: skip
        if obs is None:
            continue
        bundle.observations.append(obs)
        keys.append(str(item["id"]) if item.get("id") is not None else str(position))
    if ep_events:
        for item in _root_items(raw, "eps", what=what):
            if lines:
                # The lines wire format carries u / b / a / y / r / k only: entity_mentions, topic_tags and an empty b
                # are derived locally (ingest.derive.expand_lines_ep), exactly as the rule LLM fills its own EPs.
                # 行式线上格式只有 u / b / a / y / r / k：entity_mentions、topic_tags 与空的 b 在本地推导
                # （ingest.derive.expand_lines_ep），与规则 LLM 补齐自己的 EP 完全一样。
                item = expand_lines_ep(item, extractor=entity_extractor, zh_rules=zh_rules)
            ep = _ep_from_item(llm, item, events=ep_events, ep_map=ep_map, scope=scope, what=f"{what}.eps")
            if ep is not None:
                bundle.eps.append(ep)
    decisions = raw.get("decisions")
    if isinstance(decisions, dict):
        # A single decision object, or an object keyed by observation id, carries the verdicts all the same.
        # 单个决策对象、或以观察 id 为键的对象，同样承载着裁决。
        decisions = _root_items(raw, "decisions", what=what)
    if not isinstance(decisions, list):
        # One warning for the whole reply, not one per observation: the root itself is unusable. The compact contract
        # asks for decisions only when candidates were shown, so its absence without candidates is silent.
        # 整个回复只记一条警告而非每个观察一条：是根本身不可用。紧凑契约只在呈现了候选时才要求决策，
        # 因此没有候选时缺失该根不出声。
        if bundle.observations and not (compact and not candidate_refs):
            logger.warning("%s: decisions root missing or not a list; every observation is ADD", what)
        bundle.decisions = [(Decision.ADD, None, "no decisions") for _key in keys]
        return bundle
    if compact and not decisions and not candidate_refs:
        # `"decisions": []` is exactly what the compact prompt asks for when no candidates were shown: nothing was
        # there to judge, so every observation is ADD without a per-observation warning.
        # 未呈现候选时，`"decisions": []` 正是紧凑提示词要求的输出：本就无可判定，因此每条观察都是 ADD，
        # 不逐条记录警告。
        bundle.decisions = [(Decision.ADD, None, "no candidates") for _key in keys]
        return bundle
    if lines:
        # The lines contract writes a DEC line only for a non-ADD verdict: an observation without one is ADD by
        # definition (no warning), and the parser already keyed every line by its OBS line number.
        # 行式契约只为非 ADD 裁决写 DEC 行：没有对应行的观察按定义就是 ADD（不记警告），解析器也已按 OBS 行号为每行设键。
        by_line = {str(d.get("id")): d for d in _dict_items(decisions)}
        bundle.decisions = [
            _parse_decision(
                by_line[key], obs_key=key, candidate_refs=candidate_refs, hint_ids=list(obs.supersedes_hint_ids),
                what=f"{what}.decisions",
            )  # fmt: skip
            if key in by_line
            else (Decision.ADD, None, "no DEC line")
            for key, obs in zip(keys, bundle.observations)
        ]
        return bundle
    by_key = _decisions_by_key(_dict_items(decisions), keys, what=what)
    bundle.decisions = [
        _parse_decision(
            by_key.get(key), obs_key=key, candidate_refs=candidate_refs, hint_ids=list(obs.supersedes_hint_ids),
            what=f"{what}.decisions",
        )  # fmt: skip
        for key, obs in zip(keys, bundle.observations)
    ]
    return bundle


def _decisions_by_key(decisions: list[dict], keys: list[str], *, what: str) -> dict[str, dict]:
    """Join decision items to observation keys: exact id first; then ids equal after dropping non-digits ('obs_0' ↔
    '0'); when no id matches at all and the counts agree, positional pairing (one warning).
    把决策条目连接到观察键：先精确匹配 id；再匹配去掉非数字字符后相等的 id（'obs_0' ↔ '0'）；
    当没有任何 id 匹配且数量一致时按位置配对（一条警告）。
    """
    by_key: dict[str, dict] = {}
    for decision in decisions:
        if decision.get("id") is not None:
            by_key.setdefault(str(decision["id"]).strip(), decision)
    matched = {key: by_key[key] for key in keys if key in by_key}
    if len(matched) == len(keys):
        return matched
    by_digits: dict[str, dict] = {}
    for raw_key, decision in by_key.items():
        digits = re.sub(r"\D+", "", raw_key)
        if digits:
            by_digits.setdefault(digits, decision)
    for key in keys:
        if key not in matched and re.sub(r"\D+", "", key) in by_digits:
            matched[key] = by_digits[re.sub(r"\D+", "", key)]
    if not matched and keys and len(decisions) == len(keys):
        logger.warning("%s: no decision id matches an observation id; pairing decisions by position", what)
        return dict(zip(keys, decisions))
    return matched


def procedural_summary(llm: LLMBase, *, messages: list[dict], prompt: str | None) -> str:
    """Generate the Mem0-shaped procedural markdown of a conversation (procedural_summary task).
    生成对话的 Mem0 形状过程 markdown（procedural_summary 任务）。

    Input / 输入：
        llm: The LLM provider.
        llm：LLM 提供者。

        messages: Coerced chat messages.
        messages：规整后的聊天消息。

        prompt: Replacement system prompt (None → PROCEDURAL_SUMMARY_PROMPT).
        prompt：替代的系统提示词（None → PROCEDURAL_SUMMARY_PROMPT）。

    Output / 输出：
        Non-empty markdown text.
        非空的 markdown 文本。

    State effect / 状态影响：
        None; empty output raises ValueError('Procedural memory generation failed').
        无；空输出抛出 ValueError('Procedural memory generation failed')。
    """
    text = llm.generate_text("procedural_summary", messages=build_procedural_messages(messages, prompt))
    if not text or not text.strip():
        raise ValueError("Procedural memory generation failed")
    return text.strip()


__all__ = [
    "EXTRACTION_OUTPUTS",
    "ExtractionBundle",
    "decide_revisions",
    "extract_all",
    "extract_eps",
    "extract_observations",
    "neighbourhood",
    "procedural_summary",
]
