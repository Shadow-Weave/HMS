"""Local derivation of the observation fields the compact extraction contract no longer asks the LLM for.
紧凑抽取契约不再要求 LLM 输出的观察字段的本地推导。
"""

# Purpose (EN): Turn a compact observation item (id, event_idx, text, type, speaker, supersedes_hint_ids, when,
# valid_to, entities) into the full item shape by deriving subject / predicate / objects / relation / date_observed /
# valid_from / confidence / attributed_to / sentiment / topic_tags from the text, the witnessed message and the
# configured entity extractor, and entity_mentions as the union of the LLM's `entities` hint, the extractor's hits in
# the text and the speaker — so the entity axis is never narrower than in full mode. Each heuristic has exactly one
# implementation: the sentiment and topic tables in utils.text (group A); the third-person statement parser
# (statement_triple, built on the R1–R9 cue tables), the topic-tag and the mention helpers in llms.rule (group B).
# This module (group C) defines no cue of its own — it only composes those functions with the witnessed message, so
# the rule LLM and the compact path cannot drift apart.
# 目的（中）：把紧凑观察条目（id、event_idx、text、type、speaker、supersedes_hint_ids、when、valid_to、entities）扩展为
# 完整条目形状：从文本、被见证消息与配置的实体抽取器推导 subject / predicate / objects / relation / date_observed /
# valid_from / confidence / attributed_to / sentiment / topic_tags，entity_mentions 则取 LLM 的 `entities` 提示、
# 抽取器在文本中的命中与说话人三者的并集——因此实体轴绝不比完整模式更窄。每个启发式只有一份实现：情感与主题表在
# utils.text（A 组）；第三人称陈述解析器（statement_triple，建立在 R1–R9 线索表之上）、主题标签与提及辅助在
# llms.rule（B 组）。本模块（C 组）不定义任何自己的线索——只把这些函数与被见证消息组合起来，
# 因此规则 LLM 与紧凑路径不会分叉。

# Input (EN): One compact item dict, its coerced `entities` hint, the witnessed message's role and date, the entity extractor.
# 输入（中）：一个紧凑条目字典、其规整后的 `entities` 提示、被见证消息的角色与日期、实体抽取器。

# Output (EN): A full-shaped item dict that ingest.extraction coerces exactly like a full-contract reply.
# 输出（中）：完整形状的条目字典，由 ingest.extraction 按完整契约回复同样的方式规整。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

import re
from typing import Any

from hms_elite.configs.prompts import COMPACT_OBSERVATION_KEYS
from hms_elite.entities.base import EntityExtractorBase
from hms_elite.llms.rule import mention_texts, observation_topics, statement_triple
from hms_elite.utils.text import sentiment, topic_tags

# The compact contract's confidence: one constant, the same default the full path uses for a missing value.
# 紧凑契约的置信度：一个常量，与完整路径对缺失值使用的默认值相同。
COMPACT_CONFIDENCE = 0.7
# The conversation sides statement_triple reads as a subject; any other subject is a named person (§6.6 speaker rule).
# statement_triple 读作主语的对话侧别；其它主语都是具名的人（§6.6 说话人规则）。
SIDE_SUBJECTS = frozenset({"user", "assistant", "project"})
# A full datetime carries a time of day after the date (ISO 'T' or a space); a bare date does not.
# 完整的日期时间在日期之后带有时刻（ISO 的 'T' 或空格）；裸日期没有。
_FULL_DATETIME_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}[T ]\d{1,2}:\d{2}")


def subject_for(named: str | None, speaker: str | None, obs_type: str, role: str) -> str:
    """Subject of a compact observation: what the statement names, else the speaker, else the side the type implies.
    紧凑观察的主语：陈述所点名者，否则为说话人，否则为类型蕴含的侧别。

    Input / 输入：
        named: The subject statement_triple read from the text (a side or a PROPER name), or None.
        named：statement_triple 从文本读出的主语（侧别或 PROPER 名字），或 None。

        speaker: The LLM-named speaker (already coerced by extraction._speaker), or None.
        speaker：LLM 点名的说话人（已经 extraction._speaker 规整），或 None。

        obs_type: The coerced observation type.
        obs_type：规整后的观察类型。

        role: The witnessed message's role.
        role：被见证消息的角色。

    Output / 输出：
        `named`, else `speaker`, else 'project' for a constraint (R2 / R3's subject for an imperative), else
        'assistant' for an assistant message, else 'user'.
        `named`；否则 `speaker`；否则约束为 'project'（R2 / R3 对祈使句的主语）；否则 assistant 消息为 'assistant'；
        否则 'user'。

    State effect / 状态影响：
        None.
        无。
    """
    if named:
        return named
    if speaker:
        return speaker
    if obs_type == "constraint":
        return "project"
    return "assistant" if role == "assistant" else "user"


def relation_of(obs_type: str, hint_ids: Any, cue: str | None = None) -> str:
    """Relation of a compact observation: supersession when a superseded memory is cited, else the statement's own cue, else what the type implies.
    紧凑观察的关系：引用了被取代的记忆时为 supersession，否则为陈述自身的线索，否则为类型蕴含的关系。

    The type implies habit → habit, constraint → constraint, everything else single_fact.
    类型蕴含 habit → habit、constraint → constraint，其余 single_fact。

    Input / 输入：
        obs_type: The coerced observation type.
        obs_type：规整后的观察类型。

        hint_ids: The raw `supersedes_hint_ids` value of the item (any non-empty list or string counts as a cue).
        hint_ids：条目原始的 `supersedes_hint_ids` 值（任何非空列表或字符串都算线索）。

        cue: The relation statement_triple read from the text ('supersession' for a switch / relocation verb, 'comparison' for an alternative), or None.
        cue：statement_triple 从文本读出的关系（切换 / 搬迁动词为 'supersession'，有备选项为 'comparison'），或 None。

    Output / 输出：
        A Relation member value.
        Relation 成员值。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(hint_ids, str):
        cited = bool(hint_ids.strip())
    elif isinstance(hint_ids, (list, tuple)):
        cited = any(h not in (None, "") for h in hint_ids)
    else:
        cited = False
    if cited:
        return "supersession"
    if cue:
        return cue
    if obs_type in ("habit", "constraint"):
        return obs_type
    return "single_fact"


def is_full_datetime(value: Any) -> bool:
    """Whether a `when` value is a datetime (date plus time of day) rather than a bare date or something else.
    `when` 值是否为日期时间（日期加时刻），而不是裸日期或其它。

    Input / 输入：
        value: The raw `when` value.
        value：原始 `when` 值。

    Output / 输出：
        True only for a string that starts with a date followed by a time.
        仅当字符串以日期开头且后接时刻时为 True。

    State effect / 状态影响：
        None.
        无。
    """
    return isinstance(value, str) and bool(_FULL_DATETIME_RE.match(value))


def union_mentions(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    """Ordered union of mention lists with case-insensitive de-duplication: the first spelling of a name is kept.
    提及列表的有序并集，不分大小写去重：保留一个名字最先出现的写法。

    Input / 输入：
        groups: Mention lists in priority order (the LLM's `entities`, then the extractor's hits, then the speaker); None and empty strings are skipped.
        groups：按优先级排列的提及列表（LLM 的 `entities`，其次抽取器的命中，最后说话人）；None 与空字符串被跳过。

    Output / 输出：
        The stripped mentions in first-seen order, one per case-folded key.
        按首次出现顺序、去掉首尾空白的提及，每个折叠大小写后的键只保留一个。

    State effect / 状态影响：
        None.
        无。
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for mention in group or ():
            text = mention.strip() if isinstance(mention, str) else ""
            key = text.lower()
            if key and key not in seen:
                seen.add(key)
                out.append(text)
    return out


def expand_compact_item(
    item: dict,
    *,
    obs_type: str,
    speaker: str | None,
    role: str,
    message_date: str,
    extractor: EntityExtractorBase | None,
    entities: list[str] | None = None,
    zh_rules: bool = False,
) -> dict:
    """Build the full-shaped item of a compact observation; the caller then coerces it exactly like a full reply.
    构造紧凑观察的完整形状条目；调用方随后按完整回复同样的方式规整它。

    Input / 输入：
        item: The compact item (COMPACT_OBSERVATION_KEYS; extra keys are ignored — the compact contract derives, never trusts, the dropped fields).
        item：紧凑条目（COMPACT_OBSERVATION_KEYS；多余的键被忽略——紧凑契约对被删字段只推导、不信任）。

        entities: The item's `entities` hint after the caller's list coercion (extraction._str_list); None or [] when the LLM gave none.
        entities：条目的 `entities` 提示，已由调用方做列表规整（extraction._str_list）；LLM 未给出时为 None 或 []。

        obs_type: The item's type after enum coercion.
        obs_type：枚举规整后的条目类型。

        speaker: The item's speaker after coercion (None in a plain chat).
        speaker：规整后的条目说话人（普通聊天为 None）。

        role: The witnessed message's role.
        role：被见证消息的角色。

        message_date: `YYYY-MM-DD` of the witnessed message (the date_observed fallback).
        message_date：被见证消息的 `YYYY-MM-DD`（date_observed 的回退值）。

        extractor: The configured entity extractor (None → rule).
        extractor：配置的实体抽取器（None → 规则）。

        zh_rules: IngestConfig.zh_rules — Chinese statement rules in statement_triple / predicate_of, the Chinese sentiment and topic lists, the Chinese proper-name rules of the rule extractor (CONTRACT §9.6); False keeps the historical outputs.
        zh_rules：IngestConfig.zh_rules——statement_triple / predicate_of 的中文陈述规则、中文情感与主题词表、规则抽取器的中文人名规则（CONTRACT §9.6）；False 保持历史输出。

    Output / 输出：
        A dict with every full-contract key: attributed_to = role side; subject / predicate / objects read from the text by llms.rule.statement_triple (subject falling back to the speaker, then the side — subject_for); relation from the hint list, the statement's cue or the type; sentiment and topic_tags derived from the text; entity_mentions = union_mentions(entities, mention_texts(text, extractor), [speaker]) — the LLM's hint first, then the extractor's hits, then the speaker, case-insensitively de-duplicated; confidence 0.7; date_observed = `when` or the message date; valid_from = `when` only when it is a full datetime; text / type / speaker / supersedes_hint_ids / valid_to / id / event_idx passed through.
        带完整契约全部键的字典：attributed_to = 角色侧别；subject / predicate / objects 由 llms.rule.statement_triple 从文本读出（主语回退到说话人、再到侧别——subject_for）；relation 来自提示列表、陈述线索或类型；sentiment 与 topic_tags 由文本推导；entity_mentions = union_mentions(entities, mention_texts(text, extractor), [speaker])——LLM 提示在前，其次抽取器命中，最后说话人，不分大小写去重；confidence 0.7；date_observed = `when` 或消息日期；valid_from 仅当 `when` 是完整日期时间时取它；text / type / speaker / supersedes_hint_ids / valid_to / id / event_idx 原样透传。

    State effect / 状态影响：
        None.
        无。
    """
    text = item.get("text")
    plain = text.strip() if isinstance(text, str) else ""
    when = item.get("when")
    hints = item.get("supersedes_hint_ids")
    triple = statement_triple(plain, obs_type, extractor, zh_rules=zh_rules)
    return {
        "id": item.get("id"),
        "event_idx": item.get("event_idx"),
        "type": obs_type,
        "subject": subject_for(triple["subject"], speaker, obs_type, role),
        "predicate": triple["predicate"],
        "objects": triple["objects"],
        "relation": relation_of(obs_type, hints, triple["relation"]),
        "text": text,
        "date_observed": when if when not in (None, "") else message_date,
        "valid_from": when if is_full_datetime(when) else None,
        "valid_to": item.get("valid_to"),
        "confidence": COMPACT_CONFIDENCE,
        "attributed_to": "assistant" if role == "assistant" else "user",
        "speaker": speaker,
        "sentiment": sentiment(plain, zh_rules=zh_rules),
        "topic_tags": observation_topics(plain, obs_type, zh_rules=zh_rules),
        "entity_mentions": union_mentions(
            entities, mention_texts(plain, extractor, zh_rules=zh_rules) if plain else [], [speaker] if speaker else []
        ),
        "supersedes_hint_ids": hints,
    }


def expand_lines_ep(item: dict, *, extractor: EntityExtractorBase | None, zh_rules: bool = False) -> dict:
    """Fill the EP fields the lines wire format (CONTRACT §6.6: u / b / a / y / r / k only) does not carry, as the rule LLM fills its own EPs, so a lines EP stores the same entity / topic / background columns as a full one.
    补齐行式线上格式（CONTRACT §6.6：只有 u / b / a / y / r / k）不携带的 EP 字段，方式与规则 LLM 给自己的 EP 补齐相同，使行式 EP 存下与完整 EP 相同的实体 / 主题 / 背景列。

    Input / 输入：
        item: A parsed EP item (ingest.lines.parse_lines: 'event_idx', 'u', 'b', 'a', 'y', 'r', 'kappa').
        item：解析出的 EP 条目（ingest.lines.parse_lines：'event_idx'、'u'、'b'、'a'、'y'、'r'、'kappa'）。

        extractor: The configured entity extractor (None → rule).
        extractor：配置的实体抽取器（None → 规则）。

        zh_rules: IngestConfig.zh_rules (Chinese proper-name rules and topic list).
        zh_rules：IngestConfig.zh_rules（中文人名规则与主题词表）。

    Output / 输出：
        A copy of the item with `entity_mentions` = mention_texts("<a> <y> <u>") and `topic_tags` = topic_tags("<a> <y> <u>")
        (both only when the item carries none), and `b` = the k= constraints joined by ' ; ' when the item's b is empty
        (the rule LLM's own background: the constraints in force). Everything else passes through unchanged.
        条目的副本：`entity_mentions` = mention_texts("<a> <y> <u>")、`topic_tags` = topic_tags("<a> <y> <u>")（均仅在条目
        没有时填入），条目的 b 为空时 `b` = k= 约束以 ' ; ' 连接（规则 LLM 自己的背景：生效中的约束）。其余原样透传。

    State effect / 状态影响：
        None.
        无。
    """
    out = dict(item)
    kappa = out.get("kappa")
    constraints = [str(c).strip() for c in (kappa.get("constraints") or []) if str(c).strip()] if isinstance(kappa, dict) else []
    if not out.get("b"):
        out["b"] = " ; ".join(constraints)
    text = " ".join(str(out.get(key) or "").strip() for key in ("a", "y", "u")).strip()
    if not out.get("entity_mentions"):
        out["entity_mentions"] = mention_texts(text, extractor, zh_rules=zh_rules) if text else []
    if not out.get("topic_tags"):
        out["topic_tags"] = topic_tags(text, zh_rules=zh_rules) if text else []
    return out


__all__ = [
    "COMPACT_CONFIDENCE",
    "COMPACT_OBSERVATION_KEYS",
    "SIDE_SUBJECTS",
    "expand_compact_item",
    "expand_lines_ep",
    "is_full_datetime",
    "relation_of",
    "subject_for",
    "union_mentions",
]
