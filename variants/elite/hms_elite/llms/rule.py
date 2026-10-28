"""Offline rule-based LLM: recognizes the HMS_TASK marker and answers each prompt contract with regex heuristics.
离线规则 LLM：识别 HMS_TASK 标记，用正则启发式回答每个提示词契约。
"""

# Purpose (EN): Deterministic default provider; also exports the pure rule tables reused by the graph/ops planes.
# 目的（中）：确定性默认提供者；同时导出供图/运维平面复用的纯规则表。

# Input (EN): System + user prompts built by configs.prompts (§6); sections are parsed with parse_sections.
# 输入（中）：由 configs.prompts（§6）构造的系统 + 用户提示词；分节由 parse_sections 解析。

# Output (EN): JSON strings matching the §6.3 contracts (markdown for procedural_summary).
# 输出（中）：符合 §6.3 契约的 JSON 字符串（procedural_summary 为 markdown）。

# State effect (EN): None (only the in-memory `calls` log of each RuleLLM instance).
# 状态影响（中）：无（仅每个 RuleLLM 实例的内存 `calls` 日志）。

from __future__ import annotations

import json
import re
from typing import Any, Callable, ClassVar, Literal

from hms_elite.configs.prompts import COMPACT_OBSERVATION_KEYS, TYPE_LETTERS, parse_sections, task_of
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.entities.rule import RuleEntityExtractor
from hms_elite.enums import Decision, rank
from hms_elite.exceptions import LLMError, ValidationError
from hms_elite.llms.base import BaseLlmConfig, LLMBase
from hms_elite.timeutil import date_of, to_utc_iso
from hms_elite.utils.text import (
    ACTION_RE,
    CAUSE_RE,
    CLAUSE_SPLIT_RE,
    CONSTRAINT_RE,
    CONSTRAINT_ZH_RE,
    FUTURE_CUES,
    MUST_RE,
    NEGATIVE_OUTCOMES,
    OUTCOME_RE,
    POSITIVE_OUTCOMES,
    STOP_CAPS,
    SUPERSESSION_CUES_EN,
    SUPERSESSION_CUES_ZH,
    VERB_LEXICON,
    ZH_LEAD_ADVERB_ALT,
    ZH_STATEMENT_VERBS,
    clean_object,
    content_tokens,
    entity_key,
    extract_dates,
    extract_entities,
    identifier_tokens,
    is_first_person,
    ordered_union,
    rewrite_first_person,
    sentences,
    sentiment,
    tokenize,
    topic_tags,
    zh_name_stop_prefix,
)

# ---------------------------------------------------------------------------
# Rule tables shared with group G (§11.3 / §16.5)
# 与 G 组共享的规则表（§11.3 / §16.5）
# ---------------------------------------------------------------------------
PREDICATE_FAMILIES: dict[str, frozenset[str]] = {
    "PREF": frozenset({"likes", "dislikes", "prefers", "avoids"}),
    "IDENTITY": frozenset({"is_named", "is", "lives_in", "works_at", "studies_at"}),
    "CONSTRAINT": frozenset({"must_not", "must"}),
}
EXCLUSIVE: frozenset[str] = frozenset({"lives_in", "works_at", "studies_at", "is_named", "prefers"})

# ---------------------------------------------------------------------------
# §11.1 regexes (every pattern is applied with re.search unless it begins with ^)
# §11.1 正则表（除非以 ^ 开头，否则每个模式都用 re.search 应用）
# ---------------------------------------------------------------------------
PREF_RE = re.compile(
    r"\b(?P<subj>I|we)\s+(?:really\s+|absolutely\s+|still\s+|now\s+|also\s+)?(?:(?P<neg>don't|do not|no longer)\s+)?"
    r"(?P<verb>like|love|prefer|enjoy|adore|hate|dislike|can't stand|cannot stand|switched to|switch to)\s+"
    r"(?P<obj>[^,.;!?]+?)(?:\s+(?:over|rather than|instead of|more than)\s+(?P<alt>[^,.;!?]+))?\s*$",
    re.I,
)
PREF_ZH_RE = re.compile(
    r"(?P<subj>我|我们)(?:现在|已经|还是|也)?(?:很|更)?(?P<verb>喜欢|偏好|讨厌|不喜欢|改喝|改成|换成|爱喝)(?P<obj>[^，。；！？]+)"
)
IDENTITY_NAME_RE = re.compile(r"\b(?:my name is|I am called|call me)\s+(?P<obj>[A-Z][\w-]+)", re.I)
IDENTITY_IS_RE = re.compile(r"\bI(?:'m| am)\s+(?:a|an)\s+(?P<obj>[^,.;]+)", re.I)
IDENTITY_ZH_RE = re.compile(r"我叫(?P<obj>\S+)")
LIVE_WORK_RE = re.compile(r"\bI\s+(?P<verb>live|work|study)\s+(?:in|at|for)\s+(?P<obj>[^,.;]+)", re.I)
LIVE_WORK_ZH_RE = re.compile(r"我(?:住在(?P<live>[^，。；！？]+)|在(?P<work>.+?)工作|在(?P<study>.+?)上学)")
# R5 relocation twin: 'I moved/relocated to X' states a new residence, never a preference switch.
# R5 搬迁孪生：'I moved/relocated to X' 陈述新的居住地，绝不是偏好切换。
MOVED_RE = re.compile(
    r"\bI\s+(?:just\s+|recently\s+)?(?P<verb>moved|relocated)\s+(?:back\s+)?to\s+(?P<obj>[^,.;]+)", re.I
)
MOVED_ZH_RE = re.compile(r"我(?:现在|已经|刚)?(?P<verb>搬到|搬去|搬回)(?P<obj>[^，。；！？]+)")
HABIT_RE = re.compile(
    r"\b(always|usually|every (?:morning|day|week|weekend|month)|each (?:day|week|month)|twice a|tend to)\b", re.I
)
HABIT_ZH_RE = re.compile(r"每天|每周|通常|总是|习惯")
COMPARE_RE = re.compile(
    r"(?P<a>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\s+(?:vs\.?|versus|or|and)\s+(?:the\s+)?(?P<b>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"
)
COMPARE_CUE_RE = re.compile(r"compar|decid|choos|between|torn|还是|比较", re.I)
ASSISTANT_CUE_RE = re.compile(
    r"\b(I recommend|I suggest|I've (?:scheduled|booked|created)|I have (?:scheduled|booked|created)|I will|I'll"
    r"|I created|I scheduled)\b",
    re.I,
)
GREETING_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay)\b", re.I)
RAISE_RE = re.compile(r"^RAISE:\s*(?P<msg>.*)$", re.M)
SUPERSESSION_EN_RE = re.compile(SUPERSESSION_CUES_EN, re.I)
SUPERSESSION_ZH_RE = re.compile(SUPERSESSION_CUES_ZH)
FUTURE_RE = re.compile(FUTURE_CUES, re.I)
PREF_MEMORY_RE = re.compile(r"^(?:User (?:likes|prefers|dislikes)\b|用户(?:喜欢|偏好|讨厌))")
TEAM_SUBJECT_RE = re.compile(r"^(?:we|you|the team|team)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:not|never|no)\b|n't\b", re.I)
FIRST_PERSON_SPAN_RE = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|we)\b|我们|我", re.I)
PRONOUN_STRIP_RE = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|we)\b|我们|我", re.I)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
CJK_ANY_RE = re.compile(r"[一-鿿]")
_ZH_PRONOUN_PREFIXES = ("我", "你", "他", "她", "它")
# zh_rules (IngestConfig.zh_rules, CONTRACT §9.6): the predicate a Chinese statement verb maps to, matched by regex at
# the head of the remainder (never through tokens: under bigrams a one-char verb is no token), and the first-person
# subject 我 (the glossary form of "Me") → 'user'.
# zh_rules（IngestConfig.zh_rules，CONTRACT §9.6）：中文陈述动词映射到的谓词，在剩余部分开头用正则匹配（绝不经词元：
# 二元组下单字动词不是词元），以及第一人称主语 我（"Me" 的术语表形式）→ 'user'。
VERB_LEXICON_ZH: dict[str, str] = {
    "说": "says", "问": "asks", "表示": "states", "提到": "states", "告诉": "tells", "回答": "says", "建议": "suggests",
    "想": "plans", "要": "plans", "想要": "plans", "计划": "plans", "打算": "plans", "准备": "plans", "决定": "decided",
    "需要": "needs", "喜欢": "likes", "讨厌": "dislikes", "不喜欢": "dislikes", "去了": "went_to", "去": "went_to",
    "在": "is_at", "是": "is", "给": "gives", "让": "asks", "吃": "eats", "买": "buys", "做": "does", "玩": "plays",
    "看": "watches", "拿": "takes", "带": "brings", "觉得": "thinks", "负责": "owns", "加入": "joined", "和": "states",
    "跟": "states",
}  # fmt: skip
# The adverb / time-word prefix (utils.text.ZH_LEAD_ADVERB_ALT, shared with _zh_name_candidates) a verb may follow.
# 动词之前可有的副词 / 时间词前缀（utils.text.ZH_LEAD_ADVERB_ALT，与 _zh_name_candidates 共用）。
_ZH_STATEMENT_VERB_RE = re.compile(
    r"^(?:" + ZH_LEAD_ADVERB_ALT + r")*(?P<verb>"
    + "|".join(re.escape(v) for v in ZH_STATEMENT_VERBS)
    + r")(?P<obj>[^，。；！？]*)"
)
_ZH_FIRST_PERSON_RE = re.compile(r"^我(?!们)")
# Other pronoun heads of a Chinese statement: 我们 is the team (the English `we` → "the team" → project rewrite),
# the third / second persons name nobody (subject None) but are stepped over so the verb after them is read.
# 中文陈述的其它代词开头：我们 即团队（对应英文 `we` → "the team" → project 的改写），第三 / 第二人称不指名任何人
# （主语 None）但被跨过，使其后的动词能被读到。
_ZH_PRONOUN_HEAD_RE = re.compile(r"^(?P<team>我们|咱们)|^(?P<other>你们|他们|她们|它们|大家|你|他|她|它)")
# Verbs whose object is itself a clause we descend into (计划明天去公园 → 公园; 和露西亚一起去公园 → 公园).
# 宾语本身是子句、需要继续下钻的动词（计划明天去公园 → 公园；和露西亚一起去公园 → 公园）。
_ZH_LIGHT_VERBS = frozenset({"和", "跟", "给", "计划", "打算", "准备", "想", "要", "想要", "需要", "决定"})
# Coverbs (和 / 跟 / 给 + name): they carry no predicate of their own, so the predicate is that of the verb the clause
# descends to (爱丽丝和塔莎去了超市 → went_to; 给了杰克一杯咖啡 with nothing after → gives). 计划 / 打算 … keep 'plans'.
# 介动词（和 / 跟 / 给 + 人名）：自身无谓词，谓词取子句下钻到的动词（爱丽丝和塔莎去了超市 → went_to；给了杰克一杯咖啡
# 之后无动词 → gives）。计划 / 打算 … 保持 'plans'。
_ZH_COVERBS = frozenset({"和", "跟", "给"})
_ZH_ASPECT_RE = re.compile(r"^(?:了|过|着)")
_ZH_LEAD_ADVERB_RE = re.compile(r"^(?:" + ZH_LEAD_ADVERB_ALT + r")+")
_TERMINAL_PUNCT = ".!?。！？"
_NON_IMPERATIVE_STARTS = frozenset(
    {
        "i", "we", "you", "he", "she", "it", "they", "the", "a", "an", "this", "that", "these", "those", "my", "our",
        "your", "his", "her", "its", "their", "there", "here", "who", "what", "when", "where", "why", "how", "if",
        "because", "since", "after", "before", "while", "so", "and", "but", "or",
    }
)  # fmt: skip
_PRED_PHRASE = {"likes": "likes", "prefers": "prefers", "dislikes": "dislikes"}
_PRED_PHRASE_ZH = {"likes": "喜欢", "prefers": "偏好", "dislikes": "讨厌"}
_PREF_VERB_MAP = {
    "like": "likes", "love": "likes", "enjoy": "likes", "adore": "likes", "prefer": "prefers", "hate": "dislikes",
    "dislike": "dislikes", "can't stand": "dislikes", "cannot stand": "dislikes",
}  # fmt: skip
_PREF_VERB_MAP_ZH = {"喜欢": "likes", "爱喝": "likes", "偏好": "prefers", "讨厌": "dislikes", "不喜欢": "dislikes"}
# Switch verb phrases shared by R1 (preference switch) and R5 (relocation); both set relation=supersession.
# R1（偏好切换）与 R5（搬迁）共享的切换动词短语；两者都置 relation=supersession。
_SWITCH_VERBS = frozenset(
    {"switched to", "switch to", "moved to", "relocated to", "改喝", "改成", "换成", "搬到", "搬去", "搬回"}
)
_LIVE_WORK_PRED = {"live": "lives_in", "work": "works_at", "study": "studies_at"}
_OBSERVATION_KEYS = (
    "id", "event_idx", "type", "subject", "predicate", "objects", "relation", "text", "date_observed", "valid_from",
    "valid_to", "confidence", "attributed_to", "speaker", "sentiment", "topic_tags", "entity_mentions",
    "supersedes_hint_ids",
)  # fmt: skip
# The observation keys a revision judgement reads (the `New Memories` item shape of §6.2).
# 修订判定读取的观察键（§6.2 的 `New Memories` 条目形状）。
_REVISION_ITEM_KEYS = ("id", "type", "subject", "predicate", "objects", "relation", "text", "date_observed")


def _line_field(value: Any) -> str:
    """One field of a §6.6 line: the value as text with tabs and newlines flattened to spaces (None → '').
    §6.6 一行中的一个字段：值的文本形式，tab 与换行压平为空格（None → ''）。
    """
    return str(value if value is not None else "").replace("\t", " ").replace("\n", " ").strip()
PROCEDURAL_INSTRUCTION = "Create procedural memory of the above conversation."


class RuleLlmConfig(BaseLlmConfig):
    """Rule provider config: lexicon language and strictness.
    规则提供者配置：词表语言与严格模式。
    """

    language: Literal["auto", "en", "zh"] = "auto"
    strict: bool = False


# ---------------------------------------------------------------------------
# Small shared helpers
# 共享的小型辅助函数
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, collapse whitespace and strip terminal punctuation (the §11.3 rule-1 normalizer).
    小写、合并空白并去除句末标点（§11.3 规则 1 的归一化）。

    Input / 输入：
        text: Any text.
        text：任意文本。

    Output / 输出：
        Normalized string.
        归一化字符串。

    State effect / 状态影响：
        None.
        无。
    """
    out = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return out.rstrip(_TERMINAL_PUNCT).strip()


def family(predicate: str) -> str:
    """Predicate family name (PREF / IDENTITY / CONSTRAINT) or the predicate itself.
    谓词族名（PREF / IDENTITY / CONSTRAINT），否则返回谓词本身。

    Input / 输入：
        predicate: Lowercase snake predicate.
        predicate：小写蛇形谓词。

    Output / 输出：
        Family name or the predicate.
        族名或谓词本身。

    State effect / 状态影响：
        None.
        无。
    """
    pred = str(predicate or "")
    for name, members in PREDICATE_FAMILIES.items():
        if pred in members:
            return name
    return pred


def _strip_terminal(text: str) -> str:
    """Strip whitespace and terminal punctuation.
    去除首尾空白与句末标点。
    """
    return str(text or "").strip().rstrip(_TERMINAL_PUNCT).strip()


def _time_key(value: Any) -> str:
    """Comparable instant key: fixed-width UTC ISO when parseable, else the raw string ('' for None).
    可比较的时刻键：可解析时为定宽 UTC ISO，否则为原始字符串（None 为 ''）。
    """
    if value is None or value == "":
        return ""
    try:
        return to_utc_iso(value)
    except (ValidationError, ValueError, TypeError):
        return str(value)


def _date_key(value: Any) -> str:
    """Comparable date key ('' for None).
    可比较的日期键（None 为 ''）。
    """
    return "" if value is None else str(value)[:10]


def _token_set(text: Any) -> set[str]:
    """Set of stemmed content tokens.
    词干化内容词元集合。
    """
    return set(content_tokens(str(text or "")))


def _ident_forms(text: Any) -> set[str]:
    """Lower-cased identifier forms of a text (full identifier and its entity key).
    文本的小写标识符形式（完整标识符及其实体键）。
    """
    forms: set[str] = set()
    for ident in identifier_tokens(str(text or "")):
        forms.add(ident.lower())
        forms.add(entity_key(ident))
    return {f for f in forms if f}


def _ident_in(src_text: Any, dst_text: Any) -> bool:
    """True when an identifier of src_text appears in dst_text.
    src_text 的某个标识符出现在 dst_text 中时为真。
    """
    dst = str(dst_text or "").lower()
    return any(form in dst for form in _ident_forms(src_text))


def _objects_overlap(new: dict, old: dict) -> bool:
    """§11.3 rule 5: any object pair equal after normalization or with token-Jaccard >= 0.5.
    §11.3 规则 5：任一对象对归一化后相等，或词元 Jaccard >= 0.5。
    """
    for a in new.get("objects") or []:
        for b in old.get("objects") or []:
            if normalize(a) == normalize(b):
                return True
            ta, tb = _token_set(a) or set(tokenize(str(a))), _token_set(b) or set(tokenize(str(b)))
            union = ta | tb
            if union and len(ta & tb) / len(union) >= 0.5:
                return True
    return False


def _entities(text: str, extractor: EntityExtractorBase | None = None, zh_rules: bool = False) -> list:
    """The extractor's candidates (rule when None, with zh_rules forwarded) minus CJK PROPER fragments that start with a pronoun (我现 from 我现在…).
    抽取器的候选（None 时为规则，透传 zh_rules）去掉以代词开头的 CJK PROPER 片段（如 我现在… 中的 我现）。
    """
    out = []
    candidates = extractor.extract(text) if extractor is not None else extract_entities(text, zh_rules=zh_rules)
    for candidate in candidates:
        if (
            candidate.entity_type == "PROPER"
            and CJK_ANY_RE.match(candidate.text)
            and candidate.text.startswith(_ZH_PRONOUN_PREFIXES)
        ):
            continue
        out.append(candidate)
    return out


def mention_texts(text: str, extractor: EntityExtractorBase | None = None, *, zh_rules: bool = False) -> list[str]:
    """Entity mention texts of a sentence: the extractor's candidates (rule when None) minus pronoun-led CJK fragments.
    句子的实体提及文本：抽取器的候选（None 时为规则）去掉以代词开头的 CJK 片段。

    Input / 输入：
        text: A sentence or an observation text.
        text：句子或观察文本。

        extractor: Entity extractor; None means the rule extractor.
        extractor：实体抽取器；None 表示规则抽取器。

        zh_rules: Forwarded to the rule extractor when `extractor` is None (a configured extractor carries its own switch).
        zh_rules：`extractor` 为 None 时透传给规则抽取器（已配置的抽取器自带开关）。

    Output / 输出：
        Candidate texts in text order (the R1–R9 `entity_mentions` and the compact contract's derived mentions).
        按文本顺序的候选文本（R1–R9 的 `entity_mentions` 与紧凑契约推导出的提及）。

    State effect / 状态影响：
        None.
        无。
    """
    return [c.text for c in _entities(text, extractor, zh_rules)]


def predicate_of(text: str, *, zh_rules: bool = False) -> str:
    """The R9 predicate heuristic: the first VERB_LEXICON token of the text, else 'states' (zh_rules: a ZH_STATEMENT_VERBS verb at the head of the text, after optional adverbs, maps through VERB_LEXICON_ZH first).
    R9 谓词启发式：文本中第一个落在 VERB_LEXICON 的词元，否则为 'states'（zh_rules：文本开头、可跳过副词的 ZH_STATEMENT_VERBS 动词先经 VERB_LEXICON_ZH 映射）。

    Input / 输入：
        text: The text after its subject span (R9) or a whole third-person statement (compact derivation).
        text：主语片段之后的文本（R9）或整句第三人称陈述（紧凑推导）。

        zh_rules: IngestConfig.zh_rules.
        zh_rules：IngestConfig.zh_rules。

    Output / 输出：
        A lowercase snake verb.
        小写 snake 形式的动词。

    State effect / 状态影响：
        None.
        无。
    """
    if zh_rules:
        verb, _objects = _zh_clause(text.strip(), [])
        if verb is not None:
            return VERB_LEXICON_ZH.get(verb, "states")
    return next((t for t in tokenize(text) if t in VERB_LEXICON), "states")


def _zh_clause(rest: str, others: list[str]) -> tuple[str | None, list[str]]:
    """zh_rules clause reading: (predicate verb, objects). The objects are the clause after the head verb up to the
    first clause punctuation — descending through light verbs (和 / 跟 / 给 + a known name, 计划 / 打算 / 要 …) to the
    real object, cutting a 在-locative at the next verb, dropping a leading aspect particle — else the other entity
    mentions. The predicate verb is the head verb, except after a coverb (和 / 跟 / 给), where it is the verb the
    descent reached (爱丽丝和塔莎去了超市 → 去了; 给了杰克一杯咖啡 → 给). (None, others) when no verb heads the clause.
    zh_rules 的子句读取：(谓词动词, 宾语)。宾语是开头动词之后直到第一个子句标点的片段——穿过轻动词（和 / 跟 / 给 +
    已知人名、计划 / 打算 / 要 …）下钻到真正的宾语，在-处所短语在下一个动词处截断，去掉开头的体貌助词——否则取其余实体
    提及。谓词动词是开头动词，但介动词（和 / 跟 / 给）之后取下钻到达的动词（爱丽丝和塔莎去了超市 → 去了；给了杰克一杯
    咖啡 → 给）。子句不以动词开头时为 (None, others)。
    """
    mz = _ZH_STATEMENT_VERB_RE.match(rest.strip())
    if mz is None:
        return None, list(others)
    verb, tail = mz.group("verb"), mz.group("obj")
    pred_verb = verb
    for _ in range(4):
        if verb not in _ZH_LIGHT_VERBS:
            break
        # Adverbs, then the aspect particle (给了杰克…), then a known name (和塔莎…) — in that order.
        # 先副词，再体貌助词（给了杰克…），再已知人名（和塔莎…）——按此顺序。
        tail = _ZH_ASPECT_RE.sub("", _ZH_LEAD_ADVERB_RE.sub("", tail))
        for name in sorted(others, key=len, reverse=True):
            if tail.startswith(name):
                tail = tail[len(name) :]
                break
        nxt = _ZH_STATEMENT_VERB_RE.match(tail)
        if nxt is None and verb in _ZH_COVERBS:
            # No known name to strip: the verb may sit after an unlisted 2–4 char name (和塔莎去了超市 without names).
            # 没有可剥离的已知人名：动词可能在未列出的 2–4 字人名之后（无人名表时的 和塔莎去了超市）。
            nxt = next((m for i in range(2, min(5, len(tail))) if (m := _ZH_STATEMENT_VERB_RE.match(tail[i:]))), None)
        if nxt is None:
            break
        verb, tail = nxt.group("verb"), nxt.group("obj")
        if pred_verb in _ZH_COVERBS:
            pred_verb = verb
    if verb == "在":
        cut = next((i for i in range(1, len(tail)) if _ZH_STATEMENT_VERB_RE.match(tail[i:])), None)
        tail = tail[:cut] if cut else tail
    obj = clean_object(_ZH_ASPECT_RE.sub("", tail))
    return pred_verb, ([obj] if obj else list(others))


def _zh_objects(rest: str, others: list[str]) -> list[str]:
    """The objects half of _zh_clause.
    _zh_clause 的宾语部分。
    """
    return _zh_clause(rest, others)[1]


# The topic tag an observation type adds to its lexicon topics (R1 → preference, R2 / R3 → constraint).
# 观察类型附加到词表主题上的主题标签（R1 → preference，R2 / R3 → constraint）。
_TYPE_TOPIC = {"preference_event": "preference", "constraint": "constraint"}


def observation_topics(text: str, obs_type: str, *, zh_rules: bool = False) -> list[str]:
    """Topic tags of an observation: TOPIC_TABLE hits on the text plus the type's tag, at most four.
    观察的主题标签：文本命中的 TOPIC_TABLE 主题加上类型标签，最多四个。

    Input / 输入：
        text: The sentence (R1–R9) or the statement text (compact derivation).
        text：句子（R1–R9）或陈述文本（紧凑推导）。

        obs_type: The observation type ('preference_event' adds 'preference', 'constraint' adds 'constraint').
        obs_type：观察类型（'preference_event' 加 'preference'，'constraint' 加 'constraint'）。

        zh_rules: Also consult utils.text.TOPIC_TABLE_ZH.
        zh_rules：同时查 utils.text.TOPIC_TABLE_ZH。

    Output / 输出：
        At most four topic names.
        最多四个主题名。

    State effect / 状态影响：
        None.
        无。
    """
    tags = topic_tags(text, zh_rules=zh_rules)
    type_tag = _TYPE_TOPIC.get(obs_type)
    if type_tag:
        tags = ordered_union(tags, [type_tag])
    return tags[:4]


def _assistant_text(text: str) -> str:
    """Rewrite 'User …' third-person text to 'Assistant …'.
    将 'User …' 的第三人称文本改写为 'Assistant …'。
    """
    out = re.sub(r"\bUser's\b", "Assistant's", text)
    out = re.sub(r"\buser's\b", "assistant's", out)
    out = re.sub(r"\bthe user\b", "the assistant", out)
    out = re.sub(r"\bUser\b", "Assistant", out)
    return out


def _outcome_word(match: re.Match) -> str:
    """The lower-cased outcome word of an OUTCOME_RE match.
    OUTCOME_RE 匹配的小写结果词。
    """
    return (match.group(1) or match.group(0)).lower()


# ---------------------------------------------------------------------------
# Third-person statement parsing — the compact contract's subject / predicate / objects (§6.5)
# 第三人称陈述解析——紧凑契约的 subject / predicate / objects（§6.5）
# ---------------------------------------------------------------------------
# The compact contract hands the library a third-person statement ("User likes black coffee.", "Sarah joined the
# billing team in March.") and nothing else; statement_triple reads the subject, predicate and objects back out of it
# with the same cue tables R1–R9 apply to the first-person sentence (_PREF_VERB_MAP, _SWITCH_VERBS, _LIVE_WORK_PRED,
# CONSTRAINT_RE / MUST_RE, VERB_LEXICON, the entity extractor), so the two paths cannot drift apart.
# 紧凑契约交给库的只有一句第三人称陈述（"User likes black coffee."、"Sarah joined the billing team in March."）；
# statement_triple 用 R1–R9 施加于第一人称句子的同一套线索表（_PREF_VERB_MAP、_SWITCH_VERBS、_LIVE_WORK_PRED、
# CONSTRAINT_RE / MUST_RE、VERB_LEXICON、实体抽取器）从中读回主语、谓词与宾语，因此两条路径不会分叉。
_STATEMENT_SIDE_RE = re.compile(
    r"^(?:(?P<user>(?:the\s+)?user\b|用户)|(?P<assistant>(?:the\s+)?assistant\b|助手)"
    r"|(?P<project>(?:the\s+)?(?:team|project)\b|团队|项目))",
    re.I,
)
_SIDE_SUBJECT = {"user": "user", "assistant": "assistant", "project": "project"}


def _third_person(verb: str) -> str:
    """Third-person present of a one-word preference verb ('like' → 'likes'); phrases are kept as they are.
    单词偏好动词的第三人称现在时（'like' → 'likes'）；短语原样保留。
    """
    return verb if " " in verb else verb + "s"


_PREF_VERB_MAP_3P = {**_PREF_VERB_MAP, **{_third_person(v): p for v, p in _PREF_VERB_MAP.items()}}
_PREF_SWITCH_EN = sorted((v for v in _SWITCH_VERBS if v.startswith("switch")), key=len, reverse=True)
_PREF_SWITCH_ZH = sorted((v for v in _SWITCH_VERBS if v.startswith(("改", "换"))), key=len, reverse=True)
_STATEMENT_PREF_RE = re.compile(
    r"^(?:really\s+|absolutely\s+|still\s+|now\s+|also\s+)?(?:(?P<neg>doesn't|does not|don't|do not|no longer)\s+)?"
    r"(?P<verb>"
    + "|".join(re.escape(v) for v in sorted(set(_PREF_VERB_MAP_3P) | set(_PREF_SWITCH_EN), key=len, reverse=True))
    + r")\s+(?P<obj>[^,.;!?]+?)(?:\s+(?:over|rather than|instead of|more than)\s+(?P<alt>[^,.;!?]+))?\s*$",
    re.I,
)
_STATEMENT_PREF_ZH_RE = re.compile(
    r"^(?:现在|已经|还是|也)?(?:很|更)?(?P<verb>"
    + "|".join(sorted(set(_PREF_VERB_MAP_ZH) | set(_PREF_SWITCH_ZH), key=len, reverse=True))
    + r")(?P<obj>[^，。；！？]+)"
)
_STATEMENT_LIVE_WORK_RE = re.compile(
    r"^(?P<verb>lives?|lived|works?|worked|stud(?:y|ies|ied))\s+(?:in|at|for)\s+(?P<obj>[^,.;]+)", re.I
)
_STATEMENT_MOVED_RE = re.compile(
    r"^(?:just\s+|recently\s+)?(?P<verb>moved|relocated)\s+(?:back\s+)?to\s+(?P<obj>[^,.;]+)", re.I
)
_STATEMENT_LIVE_WORK_ZH_RE = re.compile(r"^(?:住在(?P<live>[^，。；！？]+)|在(?P<work>.+?)工作|在(?P<study>.+?)上学)")
_STATEMENT_MOVED_ZH_RE = re.compile(r"^(?:现在|已经|刚)?(?P<verb>搬到|搬去|搬回)(?P<obj>[^，。；！？]+)")
_STATEMENT_IDENTITY_RE = re.compile(
    r"^is\s+(?:(?:named|called)\s+(?P<name>[A-Z][\w-]+)|(?:a|an)\s+(?P<obj>[^,.;]+))", re.I
)
_STATEMENT_IDENTITY_ZH_RE = re.compile(r"^叫(?P<obj>\S+)")
_STATEMENT_COMPARED_RE = re.compile(r"^compared\s+", re.I)
# 'does not' is the third-person form of the 'do not' cue CONSTRAINT_RE knows.
# 'does not' 是 CONSTRAINT_RE 认识的 'do not' 线索的第三人称形式。
_DOES_NOT_RE = re.compile(r"\b(?:does not|doesn't)\b", re.I)
_STATEMENT_VERB_RE = re.compile(r"[a-z]+(?:s|ed)")
# Sentence-initial tokens that end in s / ed without being a verb.
# 以 s / ed 结尾却不是动词的句首词元。
_STATEMENT_NON_VERBS = frozenset(
    {
        "always", "sometimes", "perhaps", "thus", "yes", "as", "its", "his", "hers", "ours", "theirs", "us", "less",
        "plus", "across", "towards", "besides", "unless", "whereas",
    }
)  # fmt: skip


def constraint_objects(rest: str) -> list[str]:
    """Objects of a constraint: the rest minus its first token when it has >= 2 tokens (R2 / R3 and the compact derivation).
    约束的宾语：rest 有 >= 2 个词元时去掉首个词元（R2 / R3 与紧凑推导共用）。

    Input / 输入：
        rest: The text after the obligation cue.
        rest：义务线索之后的文本。

    Output / 输出：
        One cleaned object phrase.
        一个清理后的宾语短语。

    State effect / 状态影响：
        None.
        无。
    """
    tokens = rest.split()
    if len(tokens) >= 2:
        obj = clean_object(" ".join(tokens[1:]))
        if obj:
            return [obj]
    return [clean_object(rest) or rest.strip()]


def _live_work_stem(verb: str) -> str:
    """The _LIVE_WORK_PRED key of an inflected live / work / study verb.
    live / work / study 动词变形对应的 _LIVE_WORK_PRED 键。
    """
    low = verb.lower()
    return "study" if low.startswith("stud") else ("live" if low.startswith("liv") else "work")


def _statement_verb(rest: str, zh_rules: bool = False) -> str:
    """Predicate of a statement's remainder: its first token when that is an inflected verb (third-person -s, past -ed),
    else the R9 lexicon heuristic (VERB_LEXICON_ZH first under zh_rules).
    陈述剩余部分的谓词：首个词元是变形动词（第三人称 -s、过去式 -ed）时取它，否则用 R9 词表启发式（zh_rules 下先查
    VERB_LEXICON_ZH）。
    """
    tokens = rest.split()
    first = tokens[0].strip(",.;:!?").lower() if tokens else ""
    if _STATEMENT_VERB_RE.fullmatch(first) and first not in _STATEMENT_NON_VERBS and first not in _NON_IMPERATIVE_STARTS:
        return first
    return predicate_of(rest, zh_rules=zh_rules)


def statement_triple(
    text: str, obs_type: str, extractor: EntityExtractorBase | None = None, *, zh_rules: bool = False
) -> dict:
    """Subject, predicate, objects and relation cue read from a third-person statement (the compact contract's `text`).
    从第三人称陈述（紧凑契约的 `text`）读出的主语、谓词、宾语与关系线索。

    The subject is the opening conversation side (User / Assistant / the team / 用户 / 助手 → user / assistant /
    project) or a PROPER entity the statement opens with, else None (the caller falls back to the speaker / role).
    The remainder is read by type with the R1–R9 cue tables: preference_event → likes / dislikes / prefers and the
    object(s) (a switch verb or 'no longer' → supersession; an alternative → comparison); constraint → must_not with
    CONSTRAINT_RE / CONSTRAINT_ZH_RE ('does not' read as 'do not'), else must with MUST_RE, objects as in R2 / R3;
    habit → does with the remainder minus its habit cue; temporal_fact → plans / did with the entities minus the
    subject; fact → is_named / is / lives_in / works_at / studies_at ('moved to' → supersession) / compares, else the
    first inflected verb of the remainder or the R9 lexicon predicate, with the entities minus the subject.
    主语是开头的对话侧别（User / Assistant / the team / 用户 / 助手 → user / assistant / project）或陈述开头的
    PROPER 实体，否则为 None（调用方回退到说话人 / 角色）。剩余部分按类型用 R1–R9 的线索表读取：preference_event →
    likes / dislikes / prefers 与宾语（切换动词或 'no longer' → supersession；有备选项 → comparison）；constraint →
    经 CONSTRAINT_RE / CONSTRAINT_ZH_RE 为 must_not（'does not' 视为 'do not'），否则经 MUST_RE 为 must，宾语同
    R2 / R3；habit → does，宾语为去掉习惯线索的剩余部分；temporal_fact → plans / did，宾语为实体减去主语；fact →
    is_named / is / lives_in / works_at / studies_at（'moved to' → supersession）/ compares，否则取剩余部分的首个
    变形动词或 R9 词表谓词，宾语为实体减去主语。

    Input / 输入：
        text: The statement.
        text：陈述文本。

        obs_type: The coerced observation type.
        obs_type：规整后的观察类型。

        extractor: Entity extractor; None means the rule extractor.
        extractor：实体抽取器；None 表示规则抽取器。

        zh_rules: IngestConfig.zh_rules (CONTRACT §9.6): the rule extractor's Chinese proper-name rules supply the leading subject (杰克说… → 杰克), a leading 我 is the user, 我们 / 咱们 is project (the English `we` mapping), another pronoun (他 / 她 / 他们 / 大家 / 你 …) names nobody but is stepped over, the fallback predicate maps a head verb through VERB_LEXICON_ZH, and a fact / temporal_fact takes the clause after the verb as its object (_zh_objects).
        zh_rules：IngestConfig.zh_rules（CONTRACT §9.6）：规则抽取器的中文人名规则提供开头主语（杰克说… → 杰克），开头的 我 即 user，我们 / 咱们 即 project（与英文 `we` 的映射一致），其它代词（他 / 她 / 他们 / 大家 / 你 …）不指名任何人但被跨过，回退谓词把开头动词经 VERB_LEXICON_ZH 映射，fact / temporal_fact 取动词之后的子句为宾语（_zh_objects）。

    Output / 输出：
        `{"subject": str | None, "predicate": str, "objects": list[str], "relation": "supersession" | "comparison" | None}`.
        `{"subject": str | None, "predicate": str, "objects": list[str], "relation": "supersession" | "comparison" | None}`。

    State effect / 状态影响：
        None.
        无。
    """
    core = _strip_terminal(text)
    entities = _entities(core, extractor, zh_rules) if core else []
    subject: str | None = None
    span_end = 0
    side = _STATEMENT_SIDE_RE.match(core)
    if side:
        subject = _SIDE_SUBJECT[side.lastgroup or "user"]
        span_end = side.end()
    else:
        # zh_rules: a leading stop / time word (今天杰克…, 下午我…) is skipped before the subject is read, mirroring
        # the prefix strip of utils.text._zh_name_candidates.
        # zh_rules：读主语之前跳过开头的停用词 / 时间词（今天杰克…、下午我…），与 utils.text._zh_name_candidates 的前缀
        # 剥离一致。
        lead = zh_name_stop_prefix(core) if zh_rules else 0
        head = core[lead:]
        proper = next((c.text for c in entities if c.entity_type == "PROPER" and head.startswith(c.text)), None)
        if proper:
            subject, span_end = proper, lead + len(proper)
        elif zh_rules and _ZH_FIRST_PERSON_RE.match(head):
            subject, span_end = "user", lead + 1
        elif zh_rules and (pm := _ZH_PRONOUN_HEAD_RE.match(head)) is not None:
            # 今天我们去了公园 → project / went_to / [公园]; 她喜欢吃披萨 → None / likes / [吃披萨]: the verb after the
            # pronoun is read.
            # 今天我们去了公园 → project / went_to / [公园]；她喜欢吃披萨 → None / likes / [吃披萨]：读取代词之后的动词。
            subject, span_end = ("project" if pm.lastgroup == "team" else None), lead + pm.end()
        elif zh_rules and (pm := _ZH_PRONOUN_HEAD_RE.match(core)) is not None:
            # The plural pronouns are also ZH_NAME_STOP words, so they were stripped as `lead`: 我们打算晚上吃火锅 →
            # project / plans / [火锅]; 大家一起去了公园 → None / went_to / [公园].
            # 复数代词同时是 ZH_NAME_STOP 词，已作为 `lead` 被剥离：我们打算晚上吃火锅 → project / plans / [火锅]；
            # 大家一起去了公园 → None / went_to / [公园]。
            subject, span_end = ("project" if pm.lastgroup == "team" else None), pm.end()
    rest = core[span_end:].strip()
    others = [c.text for c in entities if c.text != subject]
    relation: str | None = None
    predicate: str | None = None
    objects: list[str] = []
    if obs_type == "preference_event":
        m = _STATEMENT_PREF_RE.search(rest)
        if m:
            verb = m.group("verb").lower()
            if verb in _PREF_SWITCH_EN:
                predicate, relation = "likes", "supersession"
            else:
                predicate = _PREF_VERB_MAP_3P[verb]
            neg = m.group("neg")
            if neg:
                predicate = {"likes": "dislikes", "dislikes": "likes"}.get(predicate, predicate)
                if neg.lower() == "no longer":
                    relation = "supersession"
            objects = [clean_object(m.group("obj"))]
            if m.group("alt"):
                predicate, relation = "prefers", "comparison"
                objects.append(clean_object(m.group("alt")))
        else:
            mz = _STATEMENT_PREF_ZH_RE.search(rest)
            if mz:
                verb = mz.group("verb")
                if verb in _PREF_SWITCH_ZH:
                    predicate, relation = "likes", "supersession"
                else:
                    predicate = _PREF_VERB_MAP_ZH[verb]
                objects = [clean_object(mz.group("obj"))]
            else:
                predicate = "dislikes" if sentiment(core) == "negative" else "likes"
    elif obs_type == "constraint":
        m = CONSTRAINT_RE.search(_DOES_NOT_RE.sub("do not", rest))
        mz = CONSTRAINT_ZH_RE.search(rest) if m is None else None
        if m or mz:
            predicate = "must_not"
            objects = constraint_objects((m or mz).group("rest"))
        else:
            predicate = "must"
            must = MUST_RE.search(rest)
            tail = rest[must.end() :].strip() if must else rest
            objects = constraint_objects(tail or rest) if rest else []
    elif obs_type == "habit":
        predicate = "does"
        m = HABIT_RE.search(rest) or HABIT_ZH_RE.search(rest)
        remainder = (rest[: m.start()] + " " + rest[m.end() :]) if m else rest
        obj = clean_object(remainder) or clean_object(rest)
        objects = [obj] if obj else []
    elif obs_type == "temporal_fact":
        planned = FUTURE_RE.search(core) or (zh_rules and predicate_of(rest, zh_rules=True) == "plans")
        predicate = "plans" if planned else "did"
        objects = _zh_objects(rest, others) if zh_rules else others
    else:
        m = _STATEMENT_IDENTITY_RE.search(rest)
        if m:
            predicate = "is_named" if m.group("name") else "is"
            objects = [m.group("name") or clean_object(m.group("obj"))]
        elif (mz := _STATEMENT_IDENTITY_ZH_RE.search(rest)) is not None:
            predicate, objects = "is_named", [clean_object(mz.group("obj"))]
        elif (m := _STATEMENT_LIVE_WORK_RE.search(rest)) is not None:
            predicate, objects = _LIVE_WORK_PRED[_live_work_stem(m.group("verb"))], [clean_object(m.group("obj"))]
        elif (m := _STATEMENT_MOVED_RE.search(rest)) is not None and f"{m.group('verb').lower()} to" in _SWITCH_VERBS:
            predicate, objects, relation = "lives_in", [clean_object(m.group("obj"))], "supersession"
        elif (mz := _STATEMENT_LIVE_WORK_ZH_RE.search(rest)) is not None:
            group = mz.lastgroup or "live"
            predicate, objects = _LIVE_WORK_PRED[group], [clean_object(mz.group(group))]
        elif (mz := _STATEMENT_MOVED_ZH_RE.search(rest)) is not None and mz.group("verb") in _SWITCH_VERBS:
            predicate, objects, relation = "lives_in", [clean_object(mz.group("obj"))], "supersession"
        elif _STATEMENT_COMPARED_RE.match(rest) and COMPARE_CUE_RE.search(core) and (mc := COMPARE_RE.search(rest)):
            a = " ".join(t for t in mc.group("a").split() if t not in STOP_CAPS)
            b = " ".join(t for t in mc.group("b").split() if t not in STOP_CAPS)
            if a and b:
                predicate, objects, relation = "compares", [a, b], "comparison"
        if predicate is None:
            if zh_rules and rest:
                verb, objects = _zh_clause(rest, others)
                predicate = VERB_LEXICON_ZH.get(verb, "states") if verb is not None else _statement_verb(rest, zh_rules)
            else:
                predicate = _statement_verb(rest, zh_rules) if rest else "states"
                objects = _zh_objects(rest, others) if zh_rules else others
    objects = [o for o in objects if o]
    return {"subject": subject, "predicate": predicate, "objects": objects, "relation": relation}


# ---------------------------------------------------------------------------
# §11.3 revision decision
# §11.3 修订决策
# ---------------------------------------------------------------------------
def decide_revision(new: dict, old: dict) -> tuple[Decision, str]:
    """Decide ADD / REVISE / CONFLICT / NONE for one (new, existing) memory pair (§11.3 table, first match wins).
    对一对（新, 已有）记忆决定 ADD / REVISE / CONFLICT / NONE（§11.3 表，首个命中生效）。

    Input / 输入：
        new: New memory dict with type, subject, predicate, objects, relation, text, date_observed.
        new：新记忆字典，含 type、subject、predicate、objects、relation、text、date_observed。

        old: Candidate existing memory dict with the same keys (plus valid_from).
        old：候选已有记忆字典，键相同（另有 valid_from）。

    Output / 输出：
        (Decision, reason string).
        (Decision, 原因字符串)。

    State effect / 状态影响：
        None.
        无。
    """
    if normalize(new.get("text", "")) == normalize(old.get("text", "")):
        return Decision.NONE, "identical text"
    if new.get("type") != old.get("type"):
        return Decision.ADD, "different type"
    if new.get("type") in {"temporal_fact", "habit"}:
        return Decision.ADD, "temporal facts and habits are additive"
    if new.get("subject") != old.get("subject"):
        return Decision.ADD, "different subject"
    same_predicate = new.get("predicate") == old.get("predicate")
    overlap = _objects_overlap(new, old)
    if same_predicate and overlap:
        return Decision.NONE, "duplicate information"
    new_date, old_date = _date_key(new.get("date_observed")), _date_key(old.get("date_observed"))
    if (
        new.get("relation") == "supersession"
        and family(new.get("predicate", "")) == family(old.get("predicate", ""))
        and new_date >= old_date
    ):
        return Decision.REVISE, "supersession cue"
    if same_predicate and new.get("predicate") in EXCLUSIVE and not overlap:
        if new_date >= old_date:
            return Decision.REVISE, "exclusive attribute updated"
        return Decision.CONFLICT, "older statement contradicts newer"
    polarity = {"likes", "dislikes"}
    if new.get("predicate") in polarity and old.get("predicate") in polarity and not same_predicate and overlap:
        if new.get("relation") == "supersession":
            return Decision.REVISE, "polarity flip with supersession cue"
        return Decision.CONFLICT, "polarity flip"
    if (
        new.get("type") == "constraint"
        and old.get("type") == "constraint"
        and overlap
        and {new.get("predicate"), old.get("predicate")} == {"must", "must_not"}
    ):
        return Decision.CONFLICT, "constraint polarity conflict"
    return Decision.ADD, "additive"


# ---------------------------------------------------------------------------
# §11.4 edge proposal
# §11.4 边提议
# ---------------------------------------------------------------------------
def propose_edges(nodes: list[dict], pairs: list[list[str]], witnesses: list[dict]) -> list[dict]:
    """Propose typed, witnessed edges for candidate node pairs (§11.4 rule table).
    为候选节点对提出带 witness 的类型化边（§11.4 规则表）。

    Input / 输入：
        nodes: §6.2 node dicts (ref, kind, u, b, a, y, r, subject, predicate, objects, time, entities, topics, anchor_refs).
        nodes：§6.2 节点字典（ref, kind, u, b, a, y, r, subject, predicate, objects, time, entities, topics, anchor_refs）。

        pairs: `[[ref_x, ref_y], ...]` candidate pairs; only these may receive an edge.
        pairs：候选配对 `[[ref_x, ref_y], ...]`；只有它们可能得到边。

        witnesses: `[{ref, node_ref, source_ref, time}]` (accepted for contract parity; anchor_refs are used).
        witnesses：`[{ref, node_ref, source_ref, time}]`（为契约一致而接受；实际使用 anchor_refs）。

    Output / 输出：
        The `edges` list of the edge_proposal contract (src_ref, dst_ref, edge_type, label, witness_refs, confidence).
        edge_proposal 契约的 `edges` 列表（src_ref, dst_ref, edge_type, label, witness_refs, confidence）。

    State effect / 状态影响：
        None.
        无。
    """
    by_ref = {str(n.get("ref")): n for n in nodes if isinstance(n, dict)}
    witness_nodes = {str(w.get("ref")): str(w.get("node_ref")) for w in witnesses if isinstance(w, dict)}

    def _anchor(node: dict) -> str | None:
        refs = [str(r) for r in (node.get("anchor_refs") or [])]
        refs = [r for r in refs if not witness_nodes or witness_nodes.get(r, node.get("ref")) == node.get("ref")]
        return refs[0] if refs else None

    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        x, y = by_ref.get(str(pair[0])), by_ref.get(str(pair[1]))
        if x is None or y is None or x is y:
            continue
        tx, ty = _time_key(x.get("time")), _time_key(y.get("time"))
        if tx and ty and tx > ty:
            x, y = y, x
        xa, xy, xr = str(x.get("a") or ""), str(x.get("y") or ""), x.get("r")
        yu, ya, yb, yr = str(y.get("u") or ""), str(y.get("a") or ""), str(y.get("b") or ""), y.get("r")
        evidence = 0
        if len(_token_set(xy) & _token_set(yu)) >= 2:
            evidence += 1
        if _ident_in(xa, yu):
            evidence += 1
        if xr == "negative" and set(x.get("topics") or []) & set(y.get("topics") or []):
            evidence += 1
        if evidence:
            edge_type = "causal"
        elif _ident_in(xa, ya) or _ident_in(xa, yb) or _ident_in(xy, ya) or _ident_in(xy, yb):
            edge_type, evidence = "use", 1
        elif _ident_forms(xa) & _ident_forms(ya) and xr == "negative" and yr == "positive":
            edge_type, evidence = "revision", 1
        elif (
            x.get("kind") == "observation"
            and y.get("kind") == "observation"
            and x.get("subject") is not None
            and x.get("predicate") is not None
            and (x.get("subject"), x.get("predicate")) == (y.get("subject"), y.get("predicate"))
            and not _objects_overlap(x, y)
        ):
            edge_type, evidence = "conflict", 1
        else:
            continue
        witness_refs = [_anchor(y)]
        if edge_type in {"revision", "conflict"}:
            witness_refs.append(_anchor(x))
        witness_refs = [w for w in witness_refs if w]
        if not witness_refs:
            continue
        key = (str(x.get("ref")), str(y.get("ref")), edge_type)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "src_ref": str(x.get("ref")),
                "dst_ref": str(y.get("ref")),
                "edge_type": edge_type,
                "label": f"{edge_type}: {xa[:40]} → {(yu or ya)[:40]}",
                "witness_refs": ordered_union(witness_refs),
                "confidence": round(min(0.9, 0.6 + 0.1 * evidence), 4),
            }
        )
    return edges


# ---------------------------------------------------------------------------
# §11.5 coarsen
# §11.5 粗化
# ---------------------------------------------------------------------------
def coarsen_rule(
    support_nodes: list[dict], internal_edges: list[dict], boundary_edges: list[dict], target_scale: str
) -> dict | None:
    """Build the `interface_ep` dict (slots, slot_support, invariants, residuals) from EP / InterfaceEP nodes (§11.5).
    由 EP / InterfaceEP 节点构造 `interface_ep` 字典（slots、slot_support、invariants、residuals）（§11.5）。

    Input / 输入：
        support_nodes: `[{ref, kind, u, b, a, y, r, constraints, time, anchor_refs}]` (any order; sorted by time here).
        support_nodes：`[{ref, kind, u, b, a, y, r, constraints, time, anchor_refs}]`（任意顺序；此处按时间排序）。

        internal_edges: `[{ref, src, dst, type}]` edges inside the support set (conflict → residual).
        internal_edges：支持集内部的边 `[{ref, src, dst, type}]`（conflict → 残差）。

        boundary_edges: `[{ref, inner, outer, direction, type}]` (accepted for contract parity).
        boundary_edges：`[{ref, inner, outer, direction, type}]`（为契约一致而接受）。

        target_scale: 'L2_meso' | 'L3_macro' (informational; the operator is scale-agnostic).
        target_scale：'L2_meso' | 'L3_macro'（信息性；算子与尺度无关）。

    Output / 输出：
        The interface_ep dict, or None when no ep/interface_ep node is present (the caller returns None).
        interface_ep 字典；没有 ep/interface_ep 节点时为 None（调用方返回 None）。

    State effect / 状态影响：
        None.
        无。
    """
    ordered = [n for _, n in sorted(enumerate(support_nodes), key=lambda p: (_time_key(p[1].get("time")), p[0]))]
    known_refs = {str(n.get("ref")) for n in ordered}
    members = [n for n in ordered if n.get("kind") in {"ep", "interface_ep"}]
    if not members:
        return None
    refs = [str(n.get("ref")) for n in members]
    actions = [str(n.get("a") or "") for n in members]
    a_slot = " → ".join(actions[:8]) + (" → …" if len(actions) > 8 else "")
    y_node = next((n for n in reversed(members) if str(n.get("y") or "").strip()), None)
    y_slot = str(y_node.get("y")).strip() if y_node else ""
    last_r = str(members[-1].get("r") or "unknown")
    negatives = [n for n in members if n.get("r") == "negative"]
    r_slot = "negative" if negatives and last_r != "positive" else last_r
    constraints = ordered_union(*[[str(c) for c in (n.get("constraints") or [])] for n in members])
    b_parts = [[s for s in str(n.get("b")).split(" ; ") if s] for n in members if n.get("b")]
    b_slot = " ; ".join(ordered_union(*b_parts, constraints))
    visibility = "private"
    for n in members:
        vis = str(n.get("visibility") or "private")
        try:
            if rank(vis) < rank(visibility):
                visibility = vis
        except KeyError:
            continue
    contributing = [str(n.get("ref")) for n in members if n.get("constraints") or n.get("b")] or [refs[0]]
    r_support = [refs[-1]]
    if r_slot == "negative":
        r_support = ordered_union(r_support, [str(n.get("ref")) for n in negatives])
    slot_support = {
        "u": [refs[0]],
        "b": list(contributing),
        "a": list(refs),
        "y": [str(y_node.get("ref"))] if y_node else [],
        "r": r_support,
        "kappa": list(contributing),
    }
    invariants: list[dict] = []
    for text in constraints:
        support = [str(n.get("ref")) for n in members if text in [str(c) for c in (n.get("constraints") or [])]]
        if len(support) >= 2:
            invariants.append({"text": text, "kind": "constraint", "support_refs": support})
    for n in negatives:
        invariants.append(
            {"text": f"do not: {n.get('a') or ''}", "kind": "failed_path", "support_refs": [str(n.get("ref"))]}
        )
    residuals: list[dict] = []
    for n in members:
        if n.get("r") == "unknown" and not str(n.get("y") or "").strip():
            residuals.append(
                {
                    "text": f"outcome of {n.get('ref')} unknown",
                    "kind": "missing_outcome",
                    "support_refs": [str(n.get("ref"))],
                }
            )
    for edge in internal_edges or []:
        if isinstance(edge, dict) and edge.get("type") == "conflict":
            ends = [str(edge.get("src")), str(edge.get("dst"))]
            support = [e for e in ends if e in known_refs]
            if support:
                residuals.append(
                    {
                        "text": f"unresolved conflict between {ends[0]} and {ends[1]}",
                        "kind": "unresolved_conflict",
                        "support_refs": support,
                    }
                )
    for n in members:
        for sentence in sentences(str(n.get("y") or "")):
            if sentence.endswith(("?", "？")):
                residuals.append({"text": sentence, "kind": "open_question", "support_refs": [str(n.get("ref"))]})
    return {
        "slots": {
            "u": str(members[0].get("u") or ""),
            "b": b_slot,
            "a": a_slot,
            "y": y_slot,
            "r": r_slot,
            "kappa": {"constraints": constraints, "visibility": visibility},
        },
        "slot_support": slot_support,
        "invariants": invariants,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# The provider
# 提供者实现
# ---------------------------------------------------------------------------
class RuleLLM(LLMBase):
    """Deterministic regex provider answering every HMS task contract offline.
    离线回答所有 HMS 任务契约的确定性正则提供者。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = RuleLlmConfig
    provider_name: ClassVar[str] = "rule"

    def __init__(
        self, config: RuleLlmConfig | dict | None = None, *, entity_extractor: EntityExtractorBase | None = None
    ) -> None:
        """Bind the task handlers, the entity extractor (rule when None) and start an empty `calls` log.
        绑定任务处理器、实体抽取器（None 时为规则）并初始化空的 `calls` 日志。
        """
        super().__init__(config)
        # entity_mentions of observations / EPs come from this extractor; the facade injects the configured one.
        # 观察 / EP 的 entity_mentions 由该抽取器给出；门面注入配置中选定的那一个。
        self.entity_extractor: EntityExtractorBase = entity_extractor or RuleEntityExtractor()
        self.calls: list[dict] = []
        self._handlers: dict[str, Callable[[dict], Any]] = {
            "observation_extraction": self._task_observation_extraction,
            "observation_extraction_compact": self._task_observation_extraction_compact,
            "ep_extraction": self._task_ep_extraction,
            "revision_decision": self._task_revision_decision,
            "memory_extraction": self._task_memory_extraction,
            "memory_extraction_compact": self._task_memory_extraction_compact,
            "memory_extraction_lines": self._task_memory_extraction_lines,
            "edge_proposal": self._task_edge_proposal,
            "coarsen": self._task_coarsen,
            "procedural_summary": self._task_procedural_summary,
        }

    @property
    def identity(self) -> str:
        """The constant identity 'rule@v1'.
        常量身份 'rule@v1'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The constant identity string.
            常量身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return "rule@v1"

    def describe_image(self, url_or_data_uri: str, *, detail: str | None = None) -> str | None:
        """The rule provider has no perception: every image is reported as undescribed (None).
        规则提供者没有感知能力：每张图片都报告为未描述（None）。

        Input / 输入：
            url_or_data_uri: Ignored.
            url_or_data_uri：忽略。

            detail: Ignored.
            detail：忽略。

        Output / 输出：
            None, always (the part is counted in dropped_parts).
            恒为 None（该部件计入 dropped_parts）。

        State effect / 状态影响：
            None.
            无。
        """
        return None

    @property
    def _en(self) -> bool:
        """English lexicons active.
        英文词表是否启用。
        """
        return self.config.language in {"auto", "en"}  # type: ignore[attr-defined]

    @property
    def _zh(self) -> bool:
        """Chinese lexicons active.
        中文词表是否启用。
        """
        return self.config.language in {"auto", "zh"}  # type: ignore[attr-defined]

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: System prompt with an HMS_TASK marker plus the built user prompt (§6 builders).
            messages：带 HMS_TASK 标记的系统提示词以及构造好的用户提示词（§6 构造器）。

            tools: Ignored by the rule provider.
            tools：规则提供者忽略。

            tool_choice: Ignored by the rule provider.
            tool_choice：规则提供者忽略。

            kwargs: Ignored (`response_format` is implied by the task).
            kwargs：忽略（`response_format` 由任务隐含）。

        Output / 输出：
            str or dict: JSON string per §6.3 (markdown for procedural_summary; tab-separated lines per §6.6 for memory_extraction_lines).
            文本或字典：符合 §6.3 的 JSON 字符串（procedural_summary 为 markdown；memory_extraction_lines 为 §6.6 的 tab 分隔行）。

        State effect / 状态影响：
            Appends `{'task', 'sections'}` to `calls`; unknown marker → LLMError(LLM_002); `RAISE:` line → LLMError(LLM_001).
            向 `calls` 追加 `{'task', 'sections'}`；未知标记 → LLMError(LLM_002)；`RAISE:` 行 → LLMError(LLM_001)。
        """
        system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
        task = task_of(system) if isinstance(system, str) else None
        if task not in self._handlers:
            raise LLMError(
                f"unknown task marker: {task!r}",
                "LLM_002",
                details={"task": task, "provider": self.identity},
                suggestion="build prompts with hms_elite.configs.prompts so the first system line is 'HMS_TASK: <task>'",
            )
        user_texts = [
            m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        for text in user_texts:
            hook = RAISE_RE.search(text)
            if hook:
                raise LLMError(
                    hook.group("msg").strip() or "raised by the RAISE test hook",
                    "LLM_001",
                    details={"task": task, "provider": self.identity},
                )
        if task == "procedural_summary":
            sections: dict[str, Any] = {"messages": [dict(m) for m in messages]}
        else:
            sections = parse_sections(user_texts[-1] if user_texts else "")
        self.calls.append({"task": task, "sections": sections})
        result = self._handlers[task](sections)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------ observation_extraction (§11.1)
    # ------------------------------------------------------------------ observation_extraction 观测抽取（§11.1）
    def _task_observation_extraction(self, sections: dict) -> dict:
        """Apply R1–R9 to every user/assistant sentence of `## New Messages`.
        对 `## New Messages` 的每个 user/assistant 句子应用 R1–R9。
        """
        existing = [m for m in (sections.get("Existing Memories") or []) if isinstance(m, dict)]
        new_messages = [m for m in (sections.get("New Messages") or []) if isinstance(m, dict)]
        observation_date = str(sections.get("Observation Date") or sections.get("Current Date") or "1970-01-01")[:10]
        results: list[dict] = []
        seen: set[tuple] = set()
        for position, message in enumerate(new_messages):
            role = str(message.get("role") or "user")
            if role not in {"user", "assistant"}:
                continue
            idx = message.get("idx", position)
            base_date = observation_date
            if message.get("timestamp"):
                try:
                    base_date = date_of(str(message["timestamp"]))
                except (ValidationError, ValueError, TypeError):
                    base_date = observation_date
            text = str(message.get("text") or "")
            for sentence in sentences(text):
                obs = self._observe_sentence(sentence, role=role, base_date=base_date, existing=existing)
                if obs is None:
                    continue
                key = (obs["type"], obs["predicate"], tuple(obs["objects"]), normalize(obs["text"]))
                if key in seen:
                    continue
                seen.add(key)
                obs["id"] = str(len(results))
                obs["event_idx"] = idx
                results.append({k: obs[k] for k in _OBSERVATION_KEYS})
        return {"observations": results}

    def _observe_sentence(self, sentence: str, *, role: str, base_date: str, existing: list[dict]) -> dict | None:
        """Run the sentence gates and the rule table; return one observation dict or None.
        运行句子门控与规则表；返回一个观察字典或 None。
        """
        core = _strip_terminal(sentence)
        if not core or sentence.rstrip().endswith(("?", "？")):
            return None
        if GREETING_RE.search(core):
            return None
        latin, cjk = len(LATIN_TOKEN_RE.findall(core)), bool(CJK_ANY_RE.search(core))
        # language='en' 时绝不救回纯 CJK 句子
        if latin < 3 and (not cjk or not self._zh):  # language='en' never rescues a CJK-only sentence
            return None
        # language='zh' 时只处理 CJK 句子
        if not cjk and not self._en:  # language='zh' processes CJK sentences only
            return None
        strict = bool(self.config.strict)  # type: ignore[attr-defined]
        first_person = is_first_person(core)
        if role == "assistant":
            if not (ASSISTANT_CUE_RE.search(core) or ACTION_RE.search(core)):
                return None
            if strict:
                return None
            hit = self._rule_fact(core, sentence, first_person=True)
            if hit is None:
                return None
            hit["subject"] = "assistant"
            hit["text"] = _assistant_text(hit["text"])
            rule = "R9"
        else:
            rule, hit = self._apply_rules(core, sentence, base_date=base_date, first_person=first_person, strict=strict)
            if hit is None:
                return None
            if first_person:
                hit["subject"] = "user"
        obs_text = hit["text"]
        # R1 is the only preference_event rule and R2 / R3 the only constraint rules, so the type-keyed helper adds
        # exactly the tag the rule name used to select.
        # R1 是唯一产生 preference_event 的规则，R2 / R3 是唯一产生 constraint 的规则，
        # 因此按类型取标签的辅助函数加的正是原先按规则名选出的那个标签。
        tags = observation_topics(sentence, hit["type"])
        hint_ids = self._supersedes_hints(obs_text, existing, preference=(rule == "R1"))
        return {
            "id": "0",
            "event_idx": 0,
            "type": hit["type"],
            "subject": hit.get("subject", "user"),
            "predicate": hit["predicate"],
            "objects": list(hit.get("objects") or []),
            "relation": hit.get("relation", "single_fact"),
            "text": obs_text,
            "date_observed": hit.get("date_observed") or base_date,
            "valid_from": None,
            "valid_to": None,
            "confidence": hit["confidence"],
            "attributed_to": "assistant" if role == "assistant" else "user",
            # The rule extractor reads plain user/assistant chat: there is never a named speaker to report.
            # 规则抽取器读取的是普通 user/assistant 对话：永远没有具名说话人可报告。
            "speaker": None,
            "sentiment": hit.get("sentiment", "neutral"),
            "topic_tags": tags,
            "entity_mentions": mention_texts(sentence, self.entity_extractor),
            "supersedes_hint_ids": hint_ids,
        }

    def _apply_rules(
        self, core: str, sentence: str, *, base_date: str, first_person: bool, strict: bool
    ) -> tuple[str, dict | None]:
        """R1–R9 in order; first match wins.
        依次尝试 R1–R9；首个命中生效。
        """
        for name, fn in (
            ("R1", self._rule_preference),
            ("R2", self._rule_constraint),
            ("R3", self._rule_must),
            ("R4", self._rule_identity),
            ("R5", self._rule_live_work),
            ("R6", self._rule_habit),
        ):
            hit = fn(core, sentence)
            if hit is not None:
                return name, hit
        hit = self._rule_temporal(core, sentence, base_date=base_date, first_person=first_person)
        if hit is not None:
            return "R7", hit
        hit = self._rule_comparison(core, sentence)
        if hit is not None:
            return "R8", hit
        if strict:
            return "R9", None
        return "R9", self._rule_fact(core, sentence, first_person=first_person)

    def _rule_preference(self, core: str, sentence: str) -> dict | None:
        """R1 PREFERENCE (EN + ZH).
        R1 偏好（英文 + 中文）。
        """
        if self._en:
            m = PREF_RE.search(core)
            if m:
                verb = m.group("verb").lower()
                relation = "single_fact"
                if verb in _SWITCH_VERBS:
                    predicate = "likes"
                    relation = "supersession"
                else:
                    predicate = _PREF_VERB_MAP[verb]
                neg = m.group("neg")
                if neg:
                    predicate = {"likes": "dislikes", "dislikes": "likes"}.get(predicate, predicate)
                    if neg.lower() == "no longer":
                        relation = "supersession"
                objects = [clean_object(m.group("obj"))]
                alt = m.group("alt")
                if alt:
                    relation = "comparison"
                    predicate = "prefers"
                    objects = [objects[0], clean_object(alt)]
                cue = SUPERSESSION_EN_RE.search(sentence) or (self._zh and SUPERSESSION_ZH_RE.search(sentence))
                if cue:
                    relation = "supersession"
                if alt:
                    text = f"User prefers {objects[0]} over {objects[1]}."
                else:
                    text = f"User {_PRED_PHRASE[predicate]} {objects[0]}."
                return {
                    "type": "preference_event",
                    "relation": relation,
                    "predicate": predicate,
                    "objects": objects,
                    "sentiment": "negative" if predicate == "dislikes" else "positive",
                    "confidence": 0.9 if cue else 0.8,
                    "text": text,
                    "subject": "user",
                }
        if self._zh:
            m = PREF_ZH_RE.search(core)
            if m:
                verb = m.group("verb")
                relation = "single_fact"
                if verb in _SWITCH_VERBS:
                    predicate = "likes"
                    relation = "supersession"
                else:
                    predicate = _PREF_VERB_MAP_ZH[verb]
                obj = clean_object(m.group("obj"))
                cue = SUPERSESSION_ZH_RE.search(sentence) or (self._en and SUPERSESSION_EN_RE.search(sentence))
                if cue:
                    relation = "supersession"
                return {
                    "type": "preference_event",
                    "relation": relation,
                    "predicate": predicate,
                    "objects": [obj],
                    "sentiment": "negative" if predicate == "dislikes" else "positive",
                    "confidence": 0.9 if cue else 0.8,
                    "text": f"用户{_PRED_PHRASE_ZH[predicate]}{obj}。",
                    "subject": "user",
                }
        return None

    @staticmethod
    def _constraint_objects(rest: str) -> list[str]:
        """Objects of a constraint (the module-level constraint_objects, shared with the compact derivation).
        约束的宾语（模块级 constraint_objects，与紧凑推导共用）。
        """
        return constraint_objects(rest)

    def _rule_constraint(self, core: str, sentence: str) -> dict | None:
        """R2 CONSTRAINT (must_not).
        R2 约束（must_not）。
        """
        m = CONSTRAINT_RE.search(core) if self._en else None
        subject = "project"
        rest: str | None = None
        if m:
            rest = m.group("rest")
            subj = (m.group("subj") or "").lower()
            subject = "user" if subj in {"i", "we"} else "project"
        elif self._zh:
            mz = CONSTRAINT_ZH_RE.search(core)
            if mz:
                rest = mz.group("rest")
                subject = "user" if is_first_person(core) else "project"
        if rest is None:
            return None
        return {
            "type": "constraint",
            "relation": "constraint",
            "predicate": "must_not",
            "objects": self._constraint_objects(rest),
            "sentiment": "neutral",
            "confidence": 0.9,
            "text": rewrite_first_person(sentence),
            "subject": subject,
        }

    def _is_imperative(self, core: str) -> bool:
        """Imperative heuristic: the first token is neither a pronoun/determiner nor a PROPER entity.
        祈使句启发式：首个词元既不是代词/限定词也不是 PROPER 实体。
        """
        tokens = core.split()
        if not tokens:
            return False
        first = tokens[0].strip(",.;:").lower()
        if first in _NON_IMPERATIVE_STARTS:
            return False
        proper_first = {
            c.text.split()[0].lower() for c in _entities(core, self.entity_extractor) if c.entity_type == "PROPER"
        }
        return first not in proper_first

    def _rule_must(self, core: str, sentence: str) -> dict | None:
        """R3 MUST (must) for imperatives or we/you/the team subjects without negation.
        R3 必须（must）：祈使句或 we/you/the team 主语且无否定。
        """
        if not self._en:
            return None
        m = MUST_RE.search(core)
        if not m or NEGATION_RE.search(core):
            return None
        if not (self._is_imperative(core) or TEAM_SUBJECT_RE.search(core)):
            return None
        rest = core[m.end() :].strip() or core
        return {
            "type": "constraint",
            "relation": "constraint",
            "predicate": "must",
            "objects": self._constraint_objects(rest),
            "sentiment": "neutral",
            "confidence": 0.85,
            "text": rewrite_first_person(sentence),
            "subject": "user" if is_first_person(core) else "project",
        }

    def _rule_identity(self, core: str, sentence: str) -> dict | None:
        """R4 IDENTITY (is_named / is).
        R4 身份（is_named / is）。
        """
        predicate: str | None = None
        obj = ""
        if self._en:
            m = IDENTITY_NAME_RE.search(core)
            if m:
                predicate, obj = "is_named", m.group("obj").strip()
            else:
                m = IDENTITY_IS_RE.search(core)
                if m:
                    predicate, obj = "is", clean_object(m.group("obj"))
        if predicate is None and self._zh:
            m = IDENTITY_ZH_RE.search(core)
            if m:
                predicate, obj = "is_named", clean_object(m.group("obj"))
        if predicate is None or not obj:
            return None
        return {
            "type": "fact",
            "relation": "single_fact",
            "predicate": predicate,
            "objects": [obj],
            "sentiment": sentiment(core),
            "confidence": 0.85,
            "text": rewrite_first_person(sentence),
            "subject": "user",
        }

    def _rule_live_work(self, core: str, sentence: str) -> dict | None:
        """R5 LIVE/WORK (lives_in / works_at / studies_at), including the relocation twin (supersession).
        R5 居住/工作（lives_in / works_at / studies_at），含搬迁孪生（supersession）。
        """
        predicate: str | None = None
        obj = ""
        relation = "single_fact"
        if self._en:
            m = LIVE_WORK_RE.search(core)
            if m:
                predicate, obj = _LIVE_WORK_PRED[m.group("verb").lower()], clean_object(m.group("obj"))
            else:
                # 'I moved to Berlin.' replaces the previous residence, so it carries a supersession relation
                # 'I moved to Berlin.' 取代此前的居住地，因此带 supersession 关系
                moved = MOVED_RE.search(core)
                if moved and f"{moved.group('verb').lower()} to" in _SWITCH_VERBS:
                    predicate, obj, relation = "lives_in", clean_object(moved.group("obj")), "supersession"
        if predicate is None and self._zh:
            m = LIVE_WORK_ZH_RE.search(core)
            if m:
                for group, pred in (("live", "lives_in"), ("work", "works_at"), ("study", "studies_at")):
                    if m.group(group):
                        predicate, obj = pred, clean_object(m.group(group))
                        break
            if predicate is None:
                # ZH twin of the relocation branch above ('我搬到柏林。')
                # 上面搬迁分支的中文孪生（'我搬到柏林。'）
                moved_zh = MOVED_ZH_RE.search(core)
                if moved_zh and moved_zh.group("verb") in _SWITCH_VERBS:
                    predicate, obj, relation = "lives_in", clean_object(moved_zh.group("obj")), "supersession"
        if predicate is None or not obj:
            return None
        return {
            "type": "fact",
            "relation": relation,
            "predicate": predicate,
            "objects": [obj],
            "sentiment": sentiment(core),
            "confidence": 0.85,
            "text": rewrite_first_person(sentence),
            "subject": "user",
        }

    def _rule_habit(self, core: str, sentence: str) -> dict | None:
        """R6 HABIT (does) for first-person habit cues.
        R6 习惯（does）：第一人称习惯线索。
        """
        m = HABIT_RE.search(core) if self._en else None
        if m is not None and not is_first_person(core):
            m = None
        if m is None and self._zh:
            m = HABIT_ZH_RE.search(core)
        if m is None:
            return None
        rest = core[: m.start()] + " " + core[m.end() :]
        rest = PRONOUN_STRIP_RE.sub(" ", rest, count=1)
        obj = clean_object(rest) or clean_object(core)
        return {
            "type": "habit",
            "relation": "habit",
            "predicate": "does",
            "objects": [obj],
            "sentiment": sentiment(core),
            "confidence": 0.65,
            "text": rewrite_first_person(sentence),
            "subject": "user",
        }

    def _rule_temporal(self, core: str, sentence: str, *, base_date: str, first_person: bool) -> dict | None:
        """R7 TEMPORAL (temporal_fact) when a non-`relative_this` date window is present.
        R7 时间事实（temporal_fact）：存在非 `relative_this` 的日期窗口。
        """
        try:
            windows = [w for w in extract_dates(core, base_date=base_date) if w[0] != "relative_this"]
        except (ValidationError, ValueError):
            windows = []
        if not windows:
            return None
        entities = _entities(core, self.entity_extractor)
        subject = "user"
        if not first_person:
            proper = next((c.text for c in entities if c.entity_type == "PROPER"), None)
            subject = proper or "user"
        return {
            "type": "temporal_fact",
            "relation": "single_fact",
            "predicate": "plans" if FUTURE_RE.search(core) else "did",
            "objects": [c.text for c in entities if c.text != subject],
            "sentiment": sentiment(core),
            "confidence": 0.75,
            "text": rewrite_first_person(sentence),
            "subject": subject,
            "date_observed": windows[0][1][:10],
        }

    def _rule_comparison(self, core: str, sentence: str) -> dict | None:
        """R8 COMPARISON (compares) between two PROPER names with a comparison cue.
        R8 比较（compares）：两个 PROPER 名称加比较线索。
        """
        if not COMPARE_CUE_RE.search(core):
            return None
        m = COMPARE_RE.search(core)
        if not m:
            return None
        a = " ".join(t for t in m.group("a").split() if t not in STOP_CAPS)
        b = " ".join(t for t in m.group("b").split() if t not in STOP_CAPS)
        if not a or not b:
            return None
        return {
            "type": "fact",
            "relation": "comparison",
            "predicate": "compares",
            "objects": [a, b],
            "sentiment": "neutral",
            "confidence": 0.8,
            "text": f"User compared {a} and {b}; no exclusive final choice was stated.",
            "subject": "user",
        }

    def _rule_fact(self, core: str, sentence: str, *, first_person: bool) -> dict | None:
        """R9 FACT fallback (single_fact) for sentences with an entity or a first-person subject and >= 4 tokens.
        R9 事实兜底（single_fact）：含实体或第一人称主语且 >= 4 个词元的句子。
        """
        entities = _entities(core, self.entity_extractor)
        has_entity = any(c.entity_type in {"PROPER", "FILE", "IDENTIFIER"} for c in entities)
        if not (has_entity or first_person) or len(tokenize(core)) < 4:
            return None
        subject = "user"
        span_end = 0
        if first_person:
            m = FIRST_PERSON_SPAN_RE.search(core)
            span_end = m.end() if m else 0
        else:
            proper = next((c.text for c in entities if c.entity_type == "PROPER"), None)
            if proper:
                subject = proper
                pos = core.find(proper)
                span_end = pos + len(proper) if pos >= 0 else 0
        predicate = predicate_of(core[span_end:])
        return {
            "type": "fact",
            "relation": "single_fact",
            "predicate": predicate,
            "objects": [c.text for c in entities if c.text != subject],
            "sentiment": sentiment(core),
            "confidence": 0.6,
            "text": rewrite_first_person(sentence),
            "subject": subject,
        }

    @staticmethod
    def _supersedes_hints(text: str, existing: list[dict], *, preference: bool) -> list[str]:
        """Existing-memory ids this observation may supersede (§11.1 hint rule).
        本观察可能取代的已有记忆 id（§11.1 提示规则）。
        """
        tokens = _token_set(text)
        topics = set(topic_tags(text))
        out: list[str] = []
        for memory in existing:
            mem_text = str(memory.get("text") or "")
            mem_id = memory.get("id")
            if mem_id is None:
                continue
            shared = tokens & _token_set(mem_text)
            if preference:
                if PREF_MEMORY_RE.search(mem_text) and (shared or topics & set(topic_tags(mem_text))):
                    out.append(str(mem_id))
            elif len(shared) >= 2:
                out.append(str(mem_id))
        return out

    # ------------------------------------------------------------------ ep_extraction (§11.2)
    # ------------------------------------------------------------------ ep_extraction EP 抽取（§11.2）
    def _task_ep_extraction(self, sections: dict) -> dict:
        """One EP per action clause (max 3 per message) with trigger, outcome, reward and matched constraints.
        每个动作从句一个 EP（每条消息最多 3 个），含触发、结果、回报与匹配的约束。
        """
        events = [e for e in (sections.get("Events") or []) if isinstance(e, dict)]
        known = [c for c in (sections.get("Known Constraints") or []) if isinstance(c, dict)]
        recent = [e for e in (sections.get("Recent EPs") or []) if isinstance(e, dict)]
        eps: list[dict] = []
        for position, event in enumerate(events):
            text = str(event.get("text") or "")
            if not text.strip():
                continue
            idx = event.get("idx", position)
            for ep in self._eps_from_text(text, known=known, recent=recent):
                ep["id"] = str(len(eps))
                ep["event_idx"] = idx
                ep["time_anchor"] = event.get("timestamp")
                eps.append(
                    {
                        key: ep[key]
                        for key in (
                            "id",
                            "event_idx",
                            "scale",
                            "u",
                            "b",
                            "a",
                            "y",
                            "r",
                            "kappa",
                            "time_anchor",
                            "entity_mentions",
                            "topic_tags",
                            "uses_ep_ids",
                            "confidence",
                        )  # fmt: skip
                    }
                )
        return {"eps": eps}

    def _eps_from_text(self, text: str, *, known: list[dict], recent: list[dict]) -> list[dict]:
        """Build the EP dicts of one message text.
        构造一条消息文本的 EP 字典。
        """
        sents = sentences(text)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence in sents:
            start = text.find(sentence, cursor)
            if start < 0:
                start = cursor
            spans.append((start, start + len(sentence)))
            cursor = start + len(sentence)
        message_tokens = _token_set(text)
        constraints = [str(c.get("text")) for c in known if _token_set(c.get("text")) & message_tokens]
        out: list[dict] = []
        for match in list(ACTION_RE.finditer(text))[:3]:
            a = f"{match.group('verb').lower()} {match.group('target').strip()}".strip()
            sent_idx = next((i for i, (s, e) in enumerate(spans) if s <= match.start() < e), 0)
            y, word = "", ""
            for j in range(sent_idx, len(sents)):
                core = _strip_terminal(sents[j])
                hit = OUTCOME_RE.search(core)
                if not hit:
                    continue
                word = _outcome_word(hit)
                y = self._clause_containing(core, hit.start())
                break
            if word in NEGATIVE_OUTCOMES:
                r = "negative"
            elif word in POSITIVE_OUTCOMES:
                r = "positive"
            else:
                r = "unknown" if not y else "neutral"
            cause = CAUSE_RE.search(sents[sent_idx]) if sents else None
            u = ""
            if cause:
                u = (cause.group("u") or cause.group("uzh") or "").strip()
            if not u and sent_idx > 0:
                u = _strip_terminal(sents[sent_idx - 1])
            if not u:
                u = f"task: {text[:60]}"
            a_idents = _ident_forms(a)
            uses = [str(e.get("id")) for e in recent if _ident_forms(e.get("a")) & a_idents]
            if not (u and a and (y or constraints)):
                continue
            out.append(
                {
                    "scale": "L1_micro",
                    "u": u,
                    "b": " ; ".join(constraints),
                    "a": a,
                    "y": y,
                    "r": r,
                    "kappa": {"constraints": list(constraints), "visibility": "private"},
                    "entity_mentions": [c.text for c in _entities(f"{a} {y} {u}", self.entity_extractor)],
                    "topic_tags": topic_tags(f"{a} {y} {u}"),
                    "uses_ep_ids": uses,
                    "confidence": 0.7,
                }
            )
        return out

    @staticmethod
    def _clause_containing(core: str, position: int) -> str:
        """The CLAUSE_SPLIT_RE clause of `core` that contains character `position`.
        `core` 中包含字符位置 `position` 的 CLAUSE_SPLIT_RE 从句。
        """
        cursor = 0
        best = core.strip()
        for raw in CLAUSE_SPLIT_RE.split(core):
            clause = raw.strip()
            if not clause:
                continue
            start = core.find(clause, cursor)
            if start < 0:
                continue
            end = start + len(clause)
            cursor = end
            if start <= position < end:
                return clause
        return best

    # ------------------------------------------------------------------ revision_decision (§11.3)
    # ------------------------------------------------------------------ revision_decision 修订决策（§11.3）
    def _task_revision_decision(self, sections: dict) -> dict:
        """One decision per new memory: first non-ADD candidate wins, else ADD.
        每条新记忆一个决策：首个非 ADD 的候选生效，否则 ADD。
        """
        new_items = [m for m in (sections.get("New Memories") or []) if isinstance(m, dict)]
        candidates = [m for m in (sections.get("Candidate Existing Memories") or []) if isinstance(m, dict)]
        decisions: list[dict] = []
        for new in new_items:
            decision, target, reason = Decision.ADD, None, "no candidates" if not candidates else "additive"
            for candidate in candidates:
                verdict, why = decide_revision(new, candidate)
                if verdict != Decision.ADD:
                    decision, target, reason = verdict, str(candidate.get("id")), why
                    break
                reason = why
            decisions.append(
                {"new_id": str(new.get("id")), "decision": decision.value, "target_id": target, "reason": reason}
            )
        return {"decisions": decisions}

    # ------------------------------------------------------------------ memory_extraction (composite of §11.1–§11.3)
    # ------------------------------------------------------------------ memory_extraction 复合任务（§11.1–§11.3 合一）
    def _task_memory_extraction(self, sections: dict) -> dict:
        """One reply with the three roots: observations (R1–R9), eps (action closures) and decisions (§11.3 table).
        一次回复三个根：observations（R1–R9）、eps（行动闭包）与 decisions（§11.3 表）。
        """
        observations = self._task_observation_extraction(sections)["observations"]
        # The EP rules only run when the pipeline routed at least one message to `state` (an empty Events list
        # means "no closure candidates", exactly like the staged path skipping ep_extraction altogether).
        # 只有流水线把至少一条消息分流到 `state` 时才跑 EP 规则（Events 为空列表意味着"没有闭包候选"，
        # 与分段路径完全跳过 ep_extraction 一致）。
        eps = self._task_ep_extraction(sections)["eps"] if sections.get("Events") else []
        new_items = [
            {key: obs.get(key) for key in _REVISION_ITEM_KEYS} for obs in observations if isinstance(obs, dict)
        ]
        judged = self._task_revision_decision(
            {
                "New Memories": new_items,
                "Candidate Existing Memories": sections.get("Candidate Existing Memories") or [],
                "Current Date": sections.get("Current Date"),
            }
        )["decisions"]
        decisions = [
            {
                "id": decision["new_id"],
                "decision": decision["decision"],
                "target_id": decision["target_id"],
                "reason": decision["reason"],
            }
            for decision in judged
        ]
        return {"observations": observations, "eps": eps, "decisions": decisions}

    # ------------------------------------------------------------------ compact contracts (ingest.extraction_output='compact')
    # ------------------------------------------------------------------ 紧凑契约（ingest.extraction_output='compact'）
    @staticmethod
    def _message_dates(sections: dict) -> dict[Any, str]:
        """`idx → YYYY-MM-DD` of every `## New Messages` row (its timestamp, else Observation Date).
        `## New Messages` 每一行的 `idx → YYYY-MM-DD`（其时间戳，否则 Observation Date）。
        """
        observation_date = str(sections.get("Observation Date") or sections.get("Current Date") or "1970-01-01")[:10]
        dates: dict[Any, str] = {}
        for position, message in enumerate(m for m in (sections.get("New Messages") or []) if isinstance(m, dict)):
            date = observation_date
            if message.get("timestamp"):
                try:
                    date = date_of(str(message["timestamp"]))
                except (ValidationError, ValueError, TypeError):
                    date = observation_date
            dates[message.get("idx", position)] = date
        return dates

    def _compact_observations(self, observations: list[dict], sections: dict) -> list[dict]:
        """Project full R1–R9 observations onto COMPACT_OBSERVATION_KEYS: the five required keys always, `entities`
        (the full observation's entity_mentions) and the hint list only when non-empty, `when` only when date_observed
        differs from the witnessed message's date, `valid_to` only when set — exactly what the compact prompt asks a
        model to emit.
        把完整的 R1–R9 观察投影到 COMPACT_OBSERVATION_KEYS：五个必需键总是输出，`entities`（完整观察的 entity_mentions）
        与提示列表仅在非空时输出，`when` 仅在 date_observed 与被见证消息的日期不同时输出，`valid_to` 仅在设置时输出——
        正是紧凑提示词要求模型输出的内容。
        """
        dates = self._message_dates(sections)
        compact: list[dict] = []
        for obs in observations:
            item = {key: obs.get(key) for key in COMPACT_OBSERVATION_KEYS[:5]}
            if obs.get("entity_mentions"):
                item["entities"] = list(obs["entity_mentions"])
            if obs.get("supersedes_hint_ids"):
                item["supersedes_hint_ids"] = list(obs["supersedes_hint_ids"])
            message_date = dates.get(obs.get("event_idx"))
            if obs.get("date_observed") and obs["date_observed"] != message_date:
                item["when"] = obs["date_observed"]
            if obs.get("valid_to"):
                item["valid_to"] = obs["valid_to"]
            compact.append(item)
        return compact

    def _task_observation_extraction_compact(self, sections: dict) -> dict:
        """observation_extraction (R1–R9) projected onto the compact observation keys.
        观察抽取（R1–R9）投影到紧凑观察键。
        """
        observations = self._task_observation_extraction(sections)["observations"]
        return {"observations": self._compact_observations(observations, sections)}

    def _task_memory_extraction_compact(self, sections: dict) -> dict:
        """memory_extraction with its observations projected onto the compact keys; the decisions were judged on the
        full observations (objects, predicate) before the projection, and the EP root is unchanged.
        观察投影到紧凑键的 memory_extraction；决策在投影之前基于完整观察（objects、predicate）判定，EP 根不变。
        """
        full = self._task_memory_extraction(sections)
        return {
            "observations": self._compact_observations(full["observations"], sections),
            "eps": full["eps"],
            "decisions": full["decisions"],
        }

    # ------------------------------------------------------------------ lines contract (ingest.extraction_output='lines')
    # ------------------------------------------------------------------ 行式契约（ingest.extraction_output='lines'）
    def _task_memory_extraction_lines(self, sections: dict) -> str:
        """memory_extraction rendered as the tab-separated OBS / EP / DEC text of CONTRACT §6.6.
        按 CONTRACT §6.6 渲染为 tab 分隔 OBS / EP / DEC 文本的 memory_extraction。

        Observations become `<event_idx>\\t<T>\\t<text>[\\t@date_observed]` (the letter from TYPE_LETTERS; the date only
        when it differs from the witnessed message's date, as in the compact projection); EPs become
        `<event_idx>\\tu=..\\ta=..\\ty=..\\tr=..[\\tk=c1;c2]`; only non-ADD decisions become DEC lines, naming the
        observation by its 1-based OBS line number (the R1–R9 ids are positional, so line = id + 1). Tabs inside a
        slot are replaced by spaces; the EP and DEC sections are omitted when empty; the reply ends with a newline.
        The decisions were judged on the full R1–R9 observations before the rendering, exactly like the compact twin.
        观察渲染为 `<event_idx>\\t<T>\\t<text>[\\t@date_observed]`（字母取自 TYPE_LETTERS；日期仅在与被见证消息的日期不同
        时输出，与紧凑投影相同）；EP 渲染为 `<event_idx>\\tu=..\\ta=..\\ty=..\\tr=..[\\tk=c1;c2]`；只有非 ADD 的决策成为
        DEC 行，以 1 起的 OBS 行号点名观察（R1–R9 的 id 是位置序号，故行号 = id + 1）。槽位内的 tab 替换为空格；EP 与
        DEC 分节为空时省略；回复以换行结尾。决策在渲染之前基于完整的 R1–R9 观察判定，与紧凑孪生完全一样。
        """
        full = self._task_memory_extraction(sections)
        dates = self._message_dates(sections)
        flat = _line_field
        out = ["OBS"]
        for obs in full["observations"]:
            line = f"{obs.get('event_idx')}\t{TYPE_LETTERS.get(str(obs.get('type')), 'F')}\t{flat(obs.get('text'))}"
            if obs.get("date_observed") and obs["date_observed"] != dates.get(obs.get("event_idx")):
                line += f"\t@{obs['date_observed']}"
            out.append(line)
        if full["eps"]:
            out.append("EP")
            for ep in full["eps"]:
                fields = [str(ep.get("event_idx"))] + [f"{key}={flat(ep.get(key))}" for key in ("u", "a", "y", "r")]
                constraints = [flat(c) for c in (ep.get("kappa") or {}).get("constraints") or [] if flat(c)]
                if constraints:
                    fields.append("k=" + ";".join(c.replace(";", ",") for c in constraints))
                out.append("\t".join(fields))
        dec = [
            f"{int(d['id']) + 1}\t{d['decision']}\t{flat(d.get('target_id'))}\t{flat(d.get('reason'))}"
            for d in full["decisions"]
            if d.get("decision") != Decision.ADD.value and str(d.get("id", "")).isdigit()
        ]
        if dec:
            out.append("DEC")
            out.extend(dec)
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ edge_proposal (§11.4)
    # ------------------------------------------------------------------ edge_proposal 边提议（§11.4）
    def _task_edge_proposal(self, sections: dict) -> dict:
        """Thin wrapper over propose_edges.
        propose_edges 的薄封装。
        """
        nodes = [n for n in (sections.get("Nodes") or []) if isinstance(n, dict)]
        pairs = [p for p in (sections.get("Candidate Pairs") or []) if isinstance(p, (list, tuple))]
        witnesses = [w for w in (sections.get("Witnesses") or []) if isinstance(w, dict)]
        return {"edges": propose_edges(nodes, pairs, witnesses)}

    # ------------------------------------------------------------------ coarsen (§11.5)
    # ------------------------------------------------------------------ coarsen 粗化（§11.5）
    def _task_coarsen(self, sections: dict) -> dict:
        """Thin wrapper over coarsen_rule.
        coarsen_rule 的薄封装。
        """
        support = [n for n in (sections.get("Support Nodes") or []) if isinstance(n, dict)]
        internal = [e for e in (sections.get("Internal Edges") or []) if isinstance(e, dict)]
        boundary = [e for e in (sections.get("Boundary Edges") or []) if isinstance(e, dict)]
        scale = str(sections.get("Target Scale") or "L2_meso")
        return {"interface_ep": coarsen_rule(support, internal, boundary, scale)}

    # ------------------------------------------------------------------ procedural_summary (§11.6)
    # ------------------------------------------------------------------ procedural_summary 流程摘要（§11.6）
    def _task_procedural_summary(self, sections: dict) -> str:
        """Mem0-shaped markdown summary of the conversation messages.
        Mem0 形状的对话 markdown 摘要。
        """
        messages = [m for m in (sections.get("messages") or []) if isinstance(m, dict)]
        convo = [m for m in messages if m.get("role") != "system"]
        if (
            convo
            and convo[-1].get("role") == "user"
            and str(convo[-1].get("content") or "").strip() == PROCEDURAL_INSTRUCTION
        ):
            convo = convo[:-1]
        turns = [(str(m.get("role") or "user"), _content_text(m.get("content"))) for m in convo]
        objective = next((t for role, t in turns if role == "user" and t.strip()), "")
        objective = objective.replace("\n", " ").strip()[:120] or "n/a"
        steps: list[tuple[str, str]] = []
        for i, (role, text) in enumerate(turns):
            if role != "assistant":
                continue
            action = next((s for s in sentences(text) if ACTION_RE.search(s)), None)
            if action is None:
                continue
            result = next(
                (t.replace("\n", " ").strip()[:200] for r, t in turns[i + 1 :] if r in {"tool", "user"} and t.strip()),
                "n/a",
            )
            steps.append((action, result or "n/a"))
        findings: list[str] = []
        errors: list[str] = []
        last_word = ""
        for _role, text in turns:
            for sentence in sentences(text):
                for hit in OUTCOME_RE.finditer(sentence):
                    word = _outcome_word(hit)
                    last_word = word
                    if word in POSITIVE_OUTCOMES and sentence not in findings:
                        findings.append(sentence)
                    elif word in NEGATIVE_OUTCOMES and sentence not in errors:
                        errors.append(sentence)
        if last_word in POSITIVE_OUTCOMES:
            status = "completed"
        elif last_word in NEGATIVE_OUTCOMES:
            status = "blocked"
        else:
            status = "in progress"
        current = turns[-1][1].replace("\n", " ").strip()[:200] if turns else ""
        lines = [
            "## Summary of the agent's execution history",
            "",
            f"**Task Objective**: {objective}",
            f"**Progress Status**: {status}",
            "",
        ]
        if not steps:
            steps = [("n/a", "n/a")]
        for number, (action, result) in enumerate(steps, start=1):
            lines.append(f"{number}. **Agent Action**: {action}")
            lines.append(f"   **Action Result**: {result}")
        lines.extend(
            [
                "",
                f"**Key Findings**: {'; '.join(findings) if findings else 'n/a'}",
                f"**Errors & Challenges**: {'; '.join(errors) if errors else 'n/a'}",
                f"**Current Context**: {current or 'n/a'}",
            ]
        )
        return "\n".join(lines)


def _content_text(content: Any) -> str:
    """Reduce a message content (str or parts list) to text.
    将消息内容（字符串或分片列表）归约为文本。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


__all__ = [
    "ASSISTANT_CUE_RE",
    "COMPARE_RE",
    "EXCLUSIVE",
    "PREDICATE_FAMILIES",
    "PREF_RE",
    "PREF_ZH_RE",
    "RuleLLM",
    "RuleLlmConfig",
    "coarsen_rule",
    "constraint_objects",
    "decide_revision",
    "family",
    "mention_texts",
    "normalize",
    "observation_topics",
    "predicate_of",
    "propose_edges",
    "statement_triple",
]
