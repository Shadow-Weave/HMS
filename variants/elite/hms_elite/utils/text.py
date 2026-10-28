"""Deterministic text utilities and EN/ZH cue lexicons shared by the rule provider, planner and scorer.
规则提供者、查询规划器与打分器共享的确定性文本工具与中英线索词表。
"""

# Purpose (EN): One home for every regex, lexicon and tokenizer so all planes agree on words, entities and dates.
# 目的（中）：集中存放全部正则、词表与分词器，使各平面对词、实体与日期的理解一致。

# Input (EN): Plain text (EN/ZH) and base dates.
# 输入（中）：纯文本（中英）与基准日期。

# Output (EN): Tokens, entity candidates, topics, sentiment, date windows, rewritten sentences.
# 输出（中）：词元、实体候选、主题、情感、日期窗口、改写后的句子。

# State effect (EN): None (pure functions). The only process state is the CJK tokenizer mode ContextVar
#                    (cjk_tokens_scope), which the facade sets per call from HMSConfig.text.cjk_tokens.
# 状态影响（中）：无（纯函数）。唯一的进程状态是 CJK 分词模式的 ContextVar（cjk_tokens_scope），
#                门面按 HMSConfig.text.cjk_tokens 在每次调用时设置。

from __future__ import annotations

import functools
import hashlib
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, timedelta
from typing import Callable, Iterable, Iterator, NamedTuple, Sequence

from hms_elite.timeutil import day_window, month_window, week_window, year_window

# ---------------------------------------------------------------------------
# Constants (CONTRACT §9.2, exact)
# 常量（CONTRACT §9.2，逐字一致）
# ---------------------------------------------------------------------------
IDENT_RE = re.compile(
    r"(?<![\w/])(?:[\w./-]*/)?[\w-]+\.(?:ts|tsx|js|jsx|py|go|rs|md|json|yaml|yml|sql|toml|css|html)\b"
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[A-Z]{2,}-\d+\b|#\d+\b"
)
FILE_EXT_RE = re.compile(r"\.(?:ts|tsx|js|jsx|py|go|rs|md|json|yaml|yml|sql|toml|css|html)$")
QUOTED_RE = re.compile(r"[\"“「『]([^\"”」』]{1,60})[\"”」』]|'([^']{2,60})'")
PROPER_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b")
CJK_RE = re.compile(r"[一-鿿]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
STOP_CAPS = {
    "I", "A", "An", "The", "This", "That", "These", "Those", "It", "He", "She", "They", "We", "You", "My", "Our",
    "Your", "His", "Her", "Its", "Their", "Me", "Us",
    "And", "But", "Or", "So", "If", "When", "While", "Because", "After", "Before", "Since", "Until", "Then", "Now",
    "Today", "Yesterday", "Tomorrow",
    "Do", "Does", "Did", "Don't", "Never", "Not", "No", "Yes", "Please", "Actually", "Also", "However", "Still", "Just",
    "Only", "Maybe",
    "What", "Why", "How", "Where", "Who", "Which", "Is", "Are", "Was", "Were", "Have", "Has", "Had", "Can", "Could",
    "Should", "Would", "Will", "Must", "May",
    "Let", "Ok", "OK", "Hi", "Hello", "Thanks", "Thank", "User", "Assistant", "Agent", "System", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
}  # fmt: skip
STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "not", "no", "yes", "about", "into", "over", "under",
    "again", "also", "just", "than", "too", "very", "can", "could", "should", "would", "will", "shall", "may", "might",
    "must", "done", "user", "s",
}  # fmt: skip
STOPWORDS_ZH = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "我们",
    "你",
    "他",
    "她",
    "它",
    "和",
    "与",
    "或",
    "吗",
    "呢",
    "吧",
    "啊",
    "这",
    "那",
    "什么",
    "怎么",
    "为什么",
}
# 按在文本中首次出现的顺序排列；每个 topic 最多命中一次
TOPIC_TABLE: list[tuple[re.Pattern, str]] = [  # first-appearance order in the text; each topic at most once
    (re.compile(r"\b(coffee|latte|lattes|espresso|cappuccino|tea|mocha)\b|咖啡|拿铁|茶", re.I), "coffee"),
    (re.compile(r"\b(guitar|fender|gibson|stratocaster|piano|drums|band|music)\b|吉他|音乐", re.I), "music"),
    (
        re.compile(
            r"\b(retry|patch|test|tests|deploy|webhook|bug|refactor|checkout|regression|backoff|idempotency"
            r"|commit|diff)\b|重试|测试|部署|回滚",
            re.I,
        ),
        "engineering",
    ),
    (re.compile(r"\b(billing|subscription|payment|invoice|checkout)\b|计费|订阅|支付", re.I), "billing"),
    (re.compile(r"\b(trip|flight|hotel|visit|visited|travel|vacation)\b|旅行|航班|酒店", re.I), "travel"),
    (re.compile(r"\b(meeting|standup|review|design|team|joined|shipped)\b|会议|团队|设计", re.I), "work"),
]
PREF_VERBS_POS = {"like", "love", "enjoy", "adore", "prefer"}
PREF_VERBS_NEG = {"hate", "dislike", "can't stand", "cannot stand"}
SUPERSESSION_CUES_EN = (
    r"\b(now|these days|nowadays|switched to|switch to|moved to|no longer|not anymore|anymore|instead|actually"
    r"|from now on)\b"
)
SUPERSESSION_CUES_ZH = r"现在|改成|换成|改喝|不再|以后|如今"
OBJ_TRAILING_CUES = {
    "now",
    "these",
    "days",
    "nowadays",
    "anymore",
    "instead",
    "currently",
    "actually",
    "though",
    "了",
    "的",
}
ARTICLES = {"a", "an", "the", "some", "my", "our"}
EVENT_VERBS_PAST = {
    "joined", "shipped", "visited", "went", "met", "moved", "started", "finished", "bought", "released",
    "launched", "fixed", "deployed", "left", "came", "attended", "flew", "arrived", "signed", "closed", "opened",
    "merged", "migrated",
}  # fmt: skip
FUTURE_CUES = r"\b(will|next|tomorrow|plan to|plans to|planning|going to)\b|明天|下周|下个月|将"
VERB_LEXICON = {
    "is", "are", "was", "were", "has", "have", "had", "does", "did", "works", "worked", "lives", "lived", "uses",
    "used",
    "joined", "joins", "met", "moved", "moves", "started", "starts", "finished", "shipped", "ships", "reviewing",
    "reviews", "reviewed", "plans", "planned", "wants", "needs", "likes", "prefers", "decided", "chose", "bought",
    "visited", "went", "goes", "came", "left", "built", "wrote", "said", "told", "asked", "runs", "ran", "fixed",
    "patched", "leads", "owns", "manages", "reports",
}  # fmt: skip
ACTION_VERBS = (
    r"patched|edited|changed|modified|ran|re-ran|reran|tried|reverted|rolled back|deployed|added|removed|fixed"
    r"|refactored|tested|applied|updated|installed|migrated|rewrote|merged"
)
# stops before a causal/coordinating conjunction, a clause separator, a sentence-final dot (dots inside identifiers
# survive) or end of text
# 在因果/并列连词、子句分隔符、句末句点（标识符内部的点保留）或文本结尾之前停止
CLAUSE_CUT = r"(?=\s+(?:because|since|after|when|so that|and|but)\b|[,;!?]|\.(?!\w)|$)"
ACTION_RE = re.compile(
    r"\b(?:I|we|agent|it|he|she|they)?\s*(?P<verb>" + ACTION_VERBS + r")\b\s*(?P<target>.+?)" + CLAUSE_CUT, re.I
)
# the trigger clause after the conjunction, cut at the next clause separator
# 连词之后的触发子句，在下一个子句分隔符处截断
CAUSE_RE = re.compile(
    r"\b(?:because|since|after|when|so that)\s+(?P<u>.+?)(?=[,;!?]|\.(?!\w)|$)|(?:因为|之后)(?P<uzh>[^，。；！？]+)",
    re.I,
)
# 用于填充 `y` 的子句切分器
CLAUSE_SPLIT_RE = re.compile(r",\s*(?:and|but|so)?\s*|;\s*|\s+(?:and|but)\s+|，|；")  # clause splitter used for `y`
OUTCOME_RE = re.compile(
    r"\b(failed|fails|failing|passing|passed|error|errors|errored|succeeded|works now|worked|broke|broken|regression"
    r"|regressed|green|red|timeout|timed out|crash|crashed)\b|失败|报错|通过|成功|崩溃",
    re.I,
)
NEGATIVE_OUTCOMES = {
    "failed", "fails", "failing", "error", "errors", "errored", "broke", "broken", "regression", "regressed", "red",
    "timeout", "timed out", "crash", "crashed", "失败", "报错", "崩溃",
}  # fmt: skip
POSITIVE_OUTCOMES = {"passed", "passing", "succeeded", "works now", "worked", "green", "通过", "成功"}
CONSTRAINT_RE = re.compile(
    r"^(?:(?P<subj>I|We|You|The team|Team)\s+)?(?:(?:must|should|can|could)\s*(?:not|n't)|mustn't|shouldn't"
    r"|cannot|can't|never|do not|don't|(?:is|are|am)\s+not\s+allowed\s+to|no changes to)\s+(?P<rest>.+)$",
    re.I,
)
CONSTRAINT_ZH_RE = re.compile(r"(禁止|不能|不要|不得|不允许|绝不)(?P<rest>.+)")
MUST_RE = re.compile(r"\b(must|should|always|has to|have to|need to|required to)\b", re.I)
WHY_RE = re.compile(r"\b(why|because|root cause|what caused|reason|explain)\b|为什么|原因|为何", re.I)
HANDOFF_RE = re.compile(
    r"\b(handoff|hand off|status|state of|where (?:are|were) we|next step|what's next|whats next|continue|resume"
    r"|current state|pick up|work state)\b|接手|进度|下一步|继续|当前状态",
    re.I,
)
TEMPORAL_QUERY_RE = re.compile(
    r"\b(when|what happened|timeline|history|during|on what date)\b|什么时候|发生了什么|时间线", re.I
)
ENTITY_QUERY_RE = re.compile(
    r"\b(everything about|all about|tell me about|who is|what do you know about|anything about)\b|关于|是谁", re.I
)
PREF_QUERY_RE = re.compile(
    r"\b(prefer|prefers|like|likes|favorite|favourite|choice|chose|compare|dislike|enjoy)\b|喜欢|偏好|选择|讨厌", re.I
)
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5, "june": 6,
    "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
# 按顺序应用，首个匹配生效（§13.2 与 §11.1 共用）
DATE_RES = [  # applied in order; first match wins (§13.2 / §11.1 share them)
    ("iso_date", r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    ("iso_month", r"\b(?P<y>\d{4})-(?P<m>\d{2})\b(?!-)"),
    (
        "month_en",
        r"\b(?:on|in|by|since|until|during|of|from)?\s*(?P<mon>January|February|March|April|May|June|July|August"
        r"|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
        r"(?:\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(?P<y>\d{4}))?\b",
    ),
    ("month_zh", r"(?:(?P<y>\d{4})年)?(?P<m>\d{1,2})月(?:(?P<d>\d{1,2})[日号])?"),
    (
        "relative",
        r"\b(?P<rel>yesterday|today|tomorrow|last week|last month|last year|next week|next month)\b"
        r"|(?P<relzh>昨天|今天|明天|上周|上个月|下周|下个月)",
    ),
    # 仅供 planner 使用；§11.1 R7 忽略它
    (
        "relative_this",
        r"\b(?P<rel>this week|this month)\b|(?P<relzh>本周|这周|这个月)",
    ),  # planner only; §11.1 R7 ignores it / 仅供规划器使用；§11.1 R7 会忽略它
    ("ago", r"\b(?P<n>\d+)\s+(?P<unit>days?|weeks?|months?|years?)\s+ago\b|(?P<nzh>\d+)(?P<unitzh>天|周|个月|年)前"),
    ("in_n", r"\bin\s+(?P<n>\d+)\s+(?P<unit>days?|weeks?|months?)\b"),
]

# ---------------------------------------------------------------------------
# Private helpers
# 私有辅助函数
# ---------------------------------------------------------------------------
_LATIN_OR_CJK_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*|[一-鿿]+")
_WS_RE = re.compile(r"\s+")
_POSITIVE_SENT_RE = re.compile(
    r"\b(?:like|love|enjoy|great|good|happy|passed|works|green|success)|喜欢|通过|成功", re.I
)
_NEGATIVE_SENT_RE = re.compile(r"\b(?:hate|dislike|bad|broke|failed|regression|error|crash)|失败|报错|讨厌", re.I)
_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|we)\b|我们|我", re.I)
_FIRST_PERSON_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bI've\b"), "User has"),
    (re.compile(r"\bI'm\b"), "User is"),
    (re.compile(r"\bI'd\b"), "User would"),
    (re.compile(r"\bI'll\b"), "User will"),
    (re.compile(r"\bI am\b"), "User is"),
    (re.compile(r"\bI have\b"), "User has"),
    (re.compile(r"\bI do not\b"), "User does not"),
    (re.compile(r"\bI don't\b"), "User does not"),
    (re.compile(r"\bI was\b"), "User was"),
    (re.compile(r"\bI\b"), "User"),
    (re.compile(r"\bmy\b"), "user's"),
    (re.compile(r"\bMy\b"), "User's"),
    (re.compile(r"\bme\b"), "the user"),
    (re.compile(r"\bwe\b"), "the team"),
    (re.compile(r"\bWe\b"), "The team"),
    (re.compile(r"\bour\b"), "the team's"),
]
_CJK_PROPER_RE = re.compile(r"([一-鿿]{2,4})(?=是|在|和|喜欢|加入|负责)")
_MAY_PRECEDING_RE = re.compile(r"\b(?:in|on|during|of|since|until)\s*$", re.I)
_TERMINAL_PUNCT = ".!?。！？"
_DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (kind, re.compile(pattern, 0 if kind in {"iso_date", "iso_month", "month_en", "month_zh"} else re.I))
    for kind, pattern in DATE_RES
]
# Candidate priorities: NAME = a configured proper name (RuleEntityExtractorConfig.names), ZH = the zh_rules name
# rules, CJK = the legacy _CJK_PROPER_RE lookahead (run only when zh_rules is False; its 2–4 char heads such as
# 要求团队 / 他的家乡 are fragments, not names); lower wins on span overlap.
# 候选优先级：NAME = 配置的专有名（RuleEntityExtractorConfig.names），ZH = zh_rules 的人名规则，CJK = 既有的
# _CJK_PROPER_RE 前瞻（只在 zh_rules 为 False 时运行；它的 2–4 字开头如 要求团队 / 他的家乡 是片段而非人名）；
# 区间重叠时数值小者胜出。
_PRIORITY = {"FILE": 0, "IDENTIFIER": 1, "QUOTED": 2, "PROPER": 3, "NAME": 4, "ZH": 5, "CJK": 6}

# ---------------------------------------------------------------------------
# CJK tokenizer mode (HMSConfig.text.cjk_tokens): 'bigram' = character bigrams (the default and the historical
# behaviour), 'none' = one token per CJK run (what an English-centric FTS tokenizer does; the EgoLife-ZH run A baseline).
# Read through a ContextVar so every tokenize / content_tokens call of one facade call sees the same mode on both the
# index and the query side without threading a parameter through ~25 call sites.
# CJK 分词模式（HMSConfig.text.cjk_tokens）：'bigram' = 字符二元组（默认，也是历史行为），'none' = 每个 CJK 连续串一个
# 词元（英文中心的 FTS 分词器的做法；EgoLife-ZH 运行 A 的基线）。通过 ContextVar 读取，使一次门面调用内的每个
# tokenize / content_tokens 在索引侧与查询侧看到同一模式，而不必把参数穿过约 25 处调用点。
# ---------------------------------------------------------------------------
CJK_TOKEN_MODES = ("bigram", "none")
_CJK_TOKENS: ContextVar[str] = ContextVar("hms_cjk_tokens", default="bigram")


def cjk_tokens_mode() -> str:
    """The CJK tokenizer mode in force on the calling thread / task ('bigram' unless a cjk_tokens_scope is active).
    调用线程 / 任务当前生效的 CJK 分词模式（没有活动的 cjk_tokens_scope 时为 'bigram'）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        'bigram' | 'none'.
        'bigram' | 'none'。

    State effect / 状态影响：
        None.
        无。
    """
    return _CJK_TOKENS.get()


@contextmanager
def cjk_tokens_scope(mode: str) -> Iterator[None]:
    """Set the CJK tokenizer mode for the block (re-entrant; restored on exit).
    为块内设置 CJK 分词模式（可重入；退出时恢复）。

    Input / 输入：
        mode: 'bigram' | 'none' (anything else → ValueError).
        mode：'bigram' | 'none'（其它值 → ValueError）。

    Output / 输出：
        A context manager yielding None.
        产出 None 的上下文管理器。

    State effect / 状态影响：
        Sets and resets the ContextVar (per thread / asyncio task).
        设置并恢复 ContextVar（按线程 / asyncio 任务）。
    """
    if mode not in CJK_TOKEN_MODES:
        raise ValueError(f"cjk_tokens must be one of {CJK_TOKEN_MODES}, got {mode!r}")
    token = _CJK_TOKENS.set(mode)
    try:
        yield
    finally:
        _CJK_TOKENS.reset(token)


def text_scoped(method: Callable) -> Callable:
    """Decorator for methods of objects carrying `self.config.text.cjk_tokens`: the body runs inside cjk_tokens_scope.
    用于带 `self.config.text.cjk_tokens` 的对象方法的装饰器：方法体在 cjk_tokens_scope 内运行。

    Input / 输入：
        method: An unbound method.
        method：未绑定的方法。

    Output / 输出：
        The wrapped method (functools.wraps: signature, name and docstring preserved).
        包装后的方法（functools.wraps：签名、名称与文档字符串保留）。

    State effect / 状态影响：
        None at decoration time.
        装饰时无。
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # noqa: ANN001 - mirrors the wrapped method / 与被包装方法同形
        with cjk_tokens_scope(self.config.text.cjk_tokens):
            return method(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# zh_rules lexicons (IngestConfig.zh_rules; off by default so English and legacy Chinese outputs are byte-identical)
# zh_rules 词表（IngestConfig.zh_rules；默认关闭，英文与既有中文输出逐字节不变）
# ---------------------------------------------------------------------------
# Verbs a Chinese third-person statement puts right after its subject; longest alternatives first.
# 中文第三人称陈述紧跟主语之后的动词；长的备选项在前。
ZH_STATEMENT_VERBS: tuple[str, ...] = (
    "不喜欢", "去了", "表示", "提到", "计划", "打算", "准备", "需要", "决定", "喜欢", "讨厌", "觉得", "告诉", "建议", "回答",
    "负责", "加入", "想要", "说", "问", "想", "要", "在", "是", "和", "跟", "给", "让", "吃", "买", "做", "玩", "看", "拿", "带", "去",
)  # fmt: skip
_ZH_VERB_ALT = "|".join(re.escape(v) for v in ZH_STATEMENT_VERBS)
# Time words that may sit between a subject and its verb (杰克晚上去了超市) or open a statement (晚上去了超市): part of
# the adverb prefix the verb regexes skip, never a name, and stripped as a prefix before the name (下午杰克 → 杰克).
# 可位于主语与动词之间（杰克晚上去了超市）或开启陈述（晚上去了超市）的时间词：动词正则跳过的副词前缀的一部分，永不成为
# 人名，并在人名之前作为前缀剥离（下午杰克 → 杰克）。
ZH_TIME_WORDS: frozenset[str] = frozenset(
    {
        "今天", "明天", "昨天", "现在", "早上", "上午", "中午", "下午", "晚上", "傍晚", "周末", "刚刚", "刚才", "之后", "然后",
        "后来", "已经",
    }
)  # fmt: skip
# The adverbs a Chinese statement may put between subject and verb (also / already / together / the time words); one
# alternation shared with llms.rule's _ZH_STATEMENT_VERB_RE and _ZH_LEAD_ADVERB_RE so the two sides agree.
# 中文陈述可置于主语与动词之间的副词（也 / 已经 / 一起 / 时间词）；与 llms.rule 的 _ZH_STATEMENT_VERB_RE、_ZH_LEAD_ADVERB_RE
# 共用同一个备选式，使两侧一致。
ZH_LEAD_ADVERBS: tuple[str, ...] = tuple(
    sorted({"也", "都", "就", "还", "又", "已经", "刚", "正在", "一起", "一块", "经常", "总是", "可能"} | ZH_TIME_WORDS, key=lambda w: (-len(w), w))
)
ZH_LEAD_ADVERB_ALT: str = "|".join(re.escape(w) for w in ZH_LEAD_ADVERBS)
# The words that end a name candidate when they open or close it (zh_name_ok): verbs and lead adverbs / time words.
# 位于人名候选开头或结尾时终止候选的词（zh_name_ok）：动词与前导副词 / 时间词。
_ZH_NAME_EDGE_WORDS: tuple[str, ...] = ZH_STATEMENT_VERBS + ZH_LEAD_ADVERBS
_ZH_VERB_HEAD_RE = re.compile(r"^(?:" + ZH_LEAD_ADVERB_ALT + r")*(?P<verb>" + _ZH_VERB_ALT + r")")
# What may follow a name after 和 / 跟 / 给 besides a verb: a quantifier phrase or a particle (了 is not one: a name is
# never followed by the aspect particle, so 塔莎打扫了… does not end a name at 了).
# 和 / 跟 / 给 之后的人名除动词外还可后接的内容：量词短语或助词（了 不算：人名之后从不直接接体貌助词，因此
# 塔莎打扫了… 不在 了 处结束人名）。
_ZH_NAME_TAIL_RE = re.compile(r"^(?:一[个些下条杯份瓶张本件块碗盘箱袋包封支把双顿口点]|两个|几个|的|们)")
_ZH_AFTER_RE = re.compile(r"[和跟给]")
# The aspect particle a coverb may carry before the name (跟了塔莎, 给了杰克): skipped before the name is read.
# 介动词后、人名之前可带的体貌助词（跟了塔莎、给了杰克）：读人名之前跳过。
_ZH_ASPECT_HEAD_RE = re.compile(r"^(?:了|过|着)")
# Heads that are not names: pronouns, time / discourse adverbs and demonstratives (also stripped as a prefix, so
# '今天杰克说…' still yields 杰克).
# 不是人名的开头：代词、时间 / 话语副词与指示词（也作为前缀剥离，使「今天杰克说…」仍给出 杰克）。
ZH_NAME_STOP: frozenset[str] = frozenset(
    {
        "我们", "你们", "他们", "她们", "它们", "大家", "这个", "那个", "这些", "那些", "可能", "所以", "但是", "因为", "如果",
        "还有", "以及", "或者", "有人", "别人", "其他", "其他人",
    }
    | ZH_TIME_WORDS
)  # fmt: skip
_ZH_NAME_STOP_BY_LEN: tuple[str, ...] = tuple(sorted(ZH_NAME_STOP, key=len, reverse=True))
_ZH_PRONOUN_HEADS = ("我", "你", "他", "她", "它")
# 爱 only in bounded forms (喜爱 / 热爱 / 爱吃 / 爱上 / 爱玩): a bare 爱 would make every statement naming 爱丽丝 positive.
# 爱 只取有边界的形式（喜爱 / 热爱 / 爱吃 / 爱上 / 爱玩）：单独的 爱 会让每条提到 爱丽丝 的陈述都成为正面。
_POSITIVE_SENT_ZH_RE = re.compile(r"喜欢|喜爱|热爱|爱吃|爱上|爱玩|好吃|好玩|开心|高兴|满意|不错|棒|成功|顺利|享受|愉快")
_NEGATIVE_SENT_ZH_RE = re.compile(r"不喜欢|讨厌|难吃|失败|坏了|生气|难过|糟糕|烦|不行|坏|不满|抱怨|失望")
# Extra topic rows for Chinese (appended after TOPIC_TABLE; a topic already in TOPIC_TABLE keeps its earliest hit).
# 中文附加主题行（追加在 TOPIC_TABLE 之后；TOPIC_TABLE 已有的主题保留其最早命中）。
TOPIC_TABLE_ZH: list[tuple[re.Pattern, str]] = [
    (re.compile(r"做饭|烹饪|做菜|吃饭|早餐|午餐|晚餐|早饭|午饭|晚饭|披萨|饺子|面条|火锅|烧烤|汉堡|蛋糕|外卖"), "food"),
    (re.compile(r"购物|买|淘宝|美团|超市|拼多多|京东|团购"), "shopping"),
    (re.compile(r"游戏|打牌|桌游|扑克|麻将|狼人杀"), "games"),
    (re.compile(r"音乐|唱歌|跳舞|吉他|钢琴|乐队"), "music"),
    (re.compile(r"旅行|钓鱼|公园|散步|旅游|出门|露营"), "travel"),
    (re.compile(r"打扫|洗碗|洗衣|清洁|收拾|扫地|拖地"), "housework"),
    (re.compile(r"计划|安排|会议|工作|项目|日程"), "work"),
]


class EntityCandidate(NamedTuple):
    """An entity mention; `key` is the identity stored in entities.normalized_text.
    实体提及；`key` 是存入 entities.normalized_text 的身份键。
    """

    entity_type: str
    text: str
    normalized: str
    key: str


def _plain_tokens(segment: str, cjk: str = "bigram") -> list[str]:
    """Tokenize a non-identifier segment: Latin words and CJK bigrams ('bigram') or whole CJK runs ('none'), lowercase.
    对非标识符片段分词：拉丁词与 CJK 二元组（'bigram'）或整个 CJK 连续串（'none'），全小写。
    """
    out: list[str] = []
    for match in _LATIN_OR_CJK_RE.finditer(segment.lower()):
        token = match.group(0)
        if CJK_RE.fullmatch(token):
            if len(token) == 1 or cjk == "none":
                out.append(token)
            else:
                out.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            out.append(token)
    return out


def zh_name_stop_prefix(text: str) -> int:
    """Length of the one ZH_NAME_STOP word (longest first) that opens `text` and leaves something after it, else 0.
    开启 `text`、其后仍有内容的那个 ZH_NAME_STOP 词（长者优先）的长度，否则为 0。

    Input / 输入：
        text: A CJK run or statement (今天杰克说… → 2; 杰克说… → 0; 今天 alone → 0).
        text：CJK 连续串或陈述（今天杰克说… → 2；杰克说… → 0；只有 今天 → 0）。

    Output / 输出：
        The prefix length in characters; one prefix only.
        前缀的字符数；只剥一个前缀。

    State effect / 状态影响：
        None.
        无。
    """
    for stop in _ZH_NAME_STOP_BY_LEN:
        if text.startswith(stop) and len(text) > len(stop):
            return len(stop)
    return 0


def zh_name_ok(name: str) -> bool:
    """True when a 2–4 char CJK head may be a name: not a stop word, not pronoun-led, and neither opening nor ending with a statement verb or a lead adverb / time word (the two positions where such a word marks a boundary).
    2–4 字的 CJK 开头可为人名：不是停用词、不以代词开头、开头与结尾都不是陈述动词或前导副词 / 时间词（只有这两个位置上这类词才构成边界）。

    Examples: 打算晚上, 杰克晚上, 在客厅 are fragments, not names. A verb character inside the head is allowed (王和平), but a
    name ending in an adverb character (李刚) is still rejected: that suffix is indistinguishable from 李 + 刚 — a documented
    limitation; the configured `names` list catches such names.
    例：打算晚上、杰克晚上、在客厅 是片段而非人名。头部中间的动词字允许（王和平），但以副词字结尾的人名（李刚）仍被拒绝：
    该后缀与 李 + 刚 无法区分——已记录的限制；配置的 `names` 表可以接住这类人名。

    Input / 输入：
        name: The candidate head (2–4 CJK characters).
        name：候选开头（2–4 个 CJK 字）。

    Output / 输出：
        bool.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    if name in ZH_NAME_STOP or name.startswith(_ZH_PRONOUN_HEADS):
        return False
    return not any(name.startswith(w) or name.endswith(w) for w in _ZH_NAME_EDGE_WORDS)


def _zh_name_candidates(text: str) -> list[tuple[int, int, str]]:
    """The zh_rules proper-name spans of a text: a 2–4 char run head (optionally after a ZH_NAME_STOP prefix) right
    before a ZH_STATEMENT_VERBS verb (lead adverbs / time words between them are skipped), and a 2–4 char run right
    after 和 / 跟 / 给 (a 了 / 过 / 着 particle skipped) that is followed by a verb, a quantifier / particle or the end
    of the run; heads that zh_name_ok rejects (pronoun-led, stop words, containing a verb or a time word) are never names.
    文本中 zh_rules 的专有名区间：紧接 ZH_STATEMENT_VERBS 动词之前的 2–4 字连续串开头（可在 ZH_NAME_STOP 前缀之后；
    二者之间的前导副词 / 时间词被跳过），以及 和 / 跟 / 给 之后（跳过 了 / 过 / 着）、后接动词、量词 / 助词或位于串尾的
    2–4 字串；zh_name_ok 拒绝的开头（代词开头、停用词、含动词或时间词）永不成为人名。
    """

    def _boundary_at(run: str, pos: int) -> bool:
        rest = run[pos:]
        return not rest or _ZH_VERB_HEAD_RE.match(rest) is not None or _ZH_NAME_TAIL_RE.match(rest) is not None

    out: list[tuple[int, int, str]] = []
    for run_match in CJK_RE.finditer(text):
        run, base = run_match.group(0), run_match.start()
        for j in range(2, len(run)):
            if not _ZH_VERB_HEAD_RE.match(run[j:]):
                continue
            head = run[:j]
            # Strip a leading stop word ('今天杰克说…' → 杰克); one prefix only.
            # 剥离开头的停用词（「今天杰克说…」→ 杰克）；只剥一个前缀。
            offset = zh_name_stop_prefix(head)
            head = head[offset:]
            if 2 <= len(head) <= 4 and zh_name_ok(head):
                out.append((base + offset, base + j, head))
        for sep in _ZH_AFTER_RE.finditer(run):
            tail_start = sep.end()
            aspect = _ZH_ASPECT_HEAD_RE.match(run[tail_start:])
            if aspect:
                tail_start += aspect.end()
            tail = run[tail_start:]
            picked = next((n for n in (2, 3, 4) if len(tail) >= n and _boundary_at(run, tail_start + n)), None)
            if picked is not None and zh_name_ok(tail[:picked]):
                out.append((base + tail_start, base + tail_start + picked, tail[:picked]))
    return out


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    """True when the span intersects any claimed span.
    当区间与任一已占用区间相交时为真。
    """
    start, end = span
    return any(start < c_end and c_start < end for c_start, c_end in claimed)


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (year, month) pair by delta months.
    将 (年, 月) 平移 delta 个月。
    """
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _safe_day_window(year: int, month: int, day: int) -> tuple[str, str] | None:
    """Day window for a calendar triple, None when the date is invalid.
    给定年月日的当日窗口；日期非法时返回 None。
    """
    try:
        return day_window(date(year, month, day).isoformat())
    except ValueError:
        return None


def _window_for(kind: str, match: re.Match, base: date, text: str) -> tuple[str, str] | None:
    """Compute the (gte, lte) window for one DATE_RES hit.
    为一次 DATE_RES 命中计算 (gte, lte) 窗口。
    """
    groups = match.groupdict()
    if kind == "iso_date":
        return _safe_day_window(int(groups["y"]), int(groups["m"]), int(groups["d"]))
    if kind == "iso_month":
        year, month = int(groups["y"]), int(groups["m"])
        return month_window(year, month) if 1 <= month <= 12 else None
    if kind in {"month_en", "month_zh"}:
        if kind == "month_en":
            mon_token = groups["mon"]
            month = MONTHS[mon_token.lower()]
            if mon_token == "May" and not groups.get("d") and not _MAY_PRECEDING_RE.search(text[: match.start("mon")]):
                return None
        else:
            month = int(groups["m"])
            if not 1 <= month <= 12:
                return None
        year = int(groups["y"]) if groups.get("y") else base.year
        if groups.get("d"):
            return _safe_day_window(year, month, int(groups["d"]))
        return month_window(year, month)
    if kind == "relative":
        rel = (groups.get("rel") or groups.get("relzh") or "").lower()
        if rel in {"yesterday", "昨天"}:
            return day_window((base - timedelta(days=1)).isoformat())
        if rel in {"today", "今天"}:
            return day_window(base.isoformat())
        if rel in {"tomorrow", "明天"}:
            return day_window((base + timedelta(days=1)).isoformat())
        if rel in {"last week", "上周"}:
            return week_window((base - timedelta(days=7)).isoformat())
        if rel in {"last month", "上个月"}:
            return month_window(*_add_months(base.year, base.month, -1))
        if rel == "last year":
            return year_window(base.year - 1)
        if rel in {"next week", "下周"}:
            return week_window((base + timedelta(days=7)).isoformat())
        if rel in {"next month", "下个月"}:
            return month_window(*_add_months(base.year, base.month, 1))
        return None
    if kind == "relative_this":
        rel = (groups.get("rel") or groups.get("relzh") or "").lower()
        if rel in {"this week", "本周", "这周"}:
            return week_window(base.isoformat())
        return month_window(base.year, base.month)
    if kind in {"ago", "in_n"}:
        n = int(groups.get("n") or groups.get("nzh") or 0)
        unit = (groups.get("unit") or groups.get("unitzh") or "").lower()
        sign = -1 if kind == "ago" else 1
        if unit.startswith("day") or unit == "天":
            return day_window((base + timedelta(days=sign * n)).isoformat())
        if unit.startswith("week") or unit == "周":
            return week_window((base + timedelta(days=sign * 7 * n)).isoformat())
        if unit.startswith("month") or unit == "个月":
            return month_window(*_add_months(base.year, base.month, sign * n))
        if unit.startswith("year") or unit == "年":
            return year_window(base.year + sign * n)
        return None
    return None


# ---------------------------------------------------------------------------
# Public functions (CONTRACT §9.2)
# 公开函数（CONTRACT §9.2）
# ---------------------------------------------------------------------------
def sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation or newlines.
    按终止标点或换行将文本分句。

    Input / 输入：
        text: Free text (EN/ZH).
        text：自由文本（中英）。

    Output / 输出：
        Stripped, non-empty sentences; a trailing sentence without punctuation is kept.
        去空白的非空句子；末尾无标点的句子也保留。

    State effect / 状态影响：
        None.
        无。
    """
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def tokenize(text: str, *, cjk: str | None = None) -> list[str]:
    """Lowercase tokens: identifiers kept whole, Latin words, CJK character bigrams (whole CJK runs under cjk='none'); order and duplicates kept.
    小写词元：标识符整体保留，拉丁词，CJK 字符二元组（cjk='none' 时为整个 CJK 连续串）；保持顺序与重复。

    Input / 输入：
        text: Free text.
        text：自由文本。

        cjk: 'bigram' | 'none' | None (None → the cjk_tokens_scope in force, default 'bigram').
        cjk：'bigram' | 'none' | None（None → 当前生效的 cjk_tokens_scope，默认 'bigram'）。

    Output / 输出：
        Token list.
        词元列表。

    State effect / 状态影响：
        None.
        无。
    """
    mode = cjk_tokens_mode() if cjk is None else cjk
    tokens: list[str] = []
    pos = 0
    for match in IDENT_RE.finditer(text):
        tokens.extend(_plain_tokens(text[pos : match.start()], mode))
        tokens.append(match.group(0).lower())
        pos = match.end()
    tokens.extend(_plain_tokens(text[pos:], mode))
    return tokens


def stem(token: str) -> str:
    """Light stemmer: identifiers unchanged; ies→y; trailing s (len>3, not ss); ing/ed when the remainder ≥ 4 chars.
    轻量词干化：标识符不变；ies→y；去尾 s（长度>3 且非 ss）；余部 ≥4 字符时去 ing/ed。

    Input / 输入：
        token: A lowercase token.
        token：小写词元。

    Output / 输出：
        The stemmed token.
        词干化后的词元。

    State effect / 状态影响：
        None.
        无。
    """
    if is_identifier(token) or CJK_RE.fullmatch(token):
        return token
    out = token
    if out.endswith("ies") and len(out) > 4:
        out = out[:-3] + "y"
    elif out.endswith("s") and not out.endswith("ss") and len(out) > 3:
        out = out[:-1]
    if out.endswith("ing") and len(out) - 3 >= 4:
        out = out[:-3]
    elif out.endswith("ed") and len(out) - 2 >= 4:
        out = out[:-2]
    return out


def content_tokens(text: str, *, cjk: str | None = None) -> list[str]:
    """Tokens minus EN/ZH stopwords, each stemmed.
    去除中英停用词并词干化的词元。

    Input / 输入：
        text: Free text.
        text：自由文本。

        cjk: CJK tokenizer mode as in tokenize (None → the scope in force).
        cjk：CJK 分词模式，同 tokenize（None → 当前生效的模式）。

    Output / 输出：
        Stemmed content tokens in order (duplicates kept).
        按顺序的词干化内容词元（保留重复）。

    State effect / 状态影响：
        None.
        无。
    """
    return [stem(tok) for tok in tokenize(text, cjk=cjk) if tok not in STOPWORDS_EN and tok not in STOPWORDS_ZH]


def is_identifier(token: str) -> bool:
    """True for IDENT_RE full matches or tokens containing one of ./_- plus a letter.
    IDENT_RE 完整匹配，或含 ./_- 之一且含字母的词元为真。

    Input / 输入：
        token: A token.
        token：词元。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    if IDENT_RE.fullmatch(token):
        return True
    return any(ch in token for ch in "./_-") and any(ch.isalpha() for ch in token)


def identifier_tokens(text: str) -> list[str]:
    """Every IDENT_RE hit, de-duplicated, order kept.
    全部 IDENT_RE 命中，去重并保持顺序。

    Input / 输入：
        text: Free text.
        text：自由文本。

    Output / 输出：
        Identifier strings as they appear in the text.
        文本中出现形式的标识符字符串。

    State effect / 状态影响：
        None.
        无。
    """
    return ordered_union(IDENT_RE.findall(text))


def normalize_entity(text: str) -> str:
    """Strip, lowercase and collapse whitespace.
    去首尾空白、转小写并合并空白。

    Input / 输入：
        text: Entity text.
        text：实体文本。

    Output / 输出：
        Normalized string.
        归一化字符串。

    State effect / 状态影响：
        None.
        无。
    """
    return _WS_RE.sub(" ", text.strip().lower())


def entity_key(text: str) -> str:
    """Identity key of an entity text: FILE identifiers use the lower-cased basename, others normalize_entity.
    实体文本的身份键：FILE 标识符取小写基名，其余用 normalize_entity。

    Input / 输入：
        text: Entity text or identifier.
        text：实体文本或标识符。

    Output / 输出：
        The key stored in entities.normalized_text.
        存入 entities.normalized_text 的键。

    State effect / 状态影响：
        None.
        无。
    """
    stripped = text.strip()
    if IDENT_RE.fullmatch(stripped) and ("/" in stripped or FILE_EXT_RE.search(stripped)):
        return stripped.rsplit("/", 1)[-1].lower()
    return normalize_entity(stripped)


def extract_entities(text: str, *, zh_rules: bool = False, names: Sequence[str] = ()) -> list[EntityCandidate]:
    """Entity candidates in priority order FILE, IDENTIFIER, QUOTED, PROPER, NAME, ZH, CJK with span overlap suppression (CJK only when zh_rules is False).
    按 FILE、IDENTIFIER、QUOTED、PROPER、NAME、ZH、CJK 优先级提取实体候选，并按区间抑制重叠（CJK 只在 zh_rules 为 False 时参与）。

    Input / 输入：
        text: Free text (a sentence or an atom text).
        text：自由文本（句子或原子文本）。

        zh_rules: Add the Chinese proper-name rules (_zh_name_candidates: 2–4 chars before a statement verb or after 和 / 跟 / 给) as PROPER candidates and skip the legacy _CJK_PROPER_RE lookahead, whose 2–4 char heads (要求团队, 他的家乡, 硬盘假装 …) are fragments, not names; False (default) keeps the historical output byte-identical.
        zh_rules：加入中文人名规则（_zh_name_candidates：陈述动词之前或 和 / 跟 / 给 之后的 2–4 字）作为 PROPER 候选，并跳过既有的 _CJK_PROPER_RE 前瞻——它的 2–4 字开头（要求团队、他的家乡、硬盘假装 …）是片段而非人名；False（默认）保持历史输出逐字节不变。

        names: Known proper names matched exactly wherever they occur (longest first), above every CJK rule.
        names：已知专有名，在任何出现位置精确匹配（长者优先），优先级高于全部 CJK 规则。

    Output / 输出：
        Candidates in text order, de-duplicated by key (first kept); month/weekday names never become entities.
        按文本顺序的候选，按 key 去重（保留首个）；月份/星期名从不成为实体。

    State effect / 状态影响：
        None.
        无。
    """
    # 元素为 (priority, start, end, candidate)
    raw: list[tuple[int, int, int, EntityCandidate]] = []  # (priority, start, end, candidate)
    for match in IDENT_RE.finditer(text):
        token = match.group(0)
        if "/" in token or FILE_EXT_RE.search(token):
            key = token.rsplit("/", 1)[-1].lower()
            raw.append(
                (_PRIORITY["FILE"], match.start(), match.end(), EntityCandidate("FILE", token, token.lower(), key))
            )
        else:
            raw.append(
                (
                    _PRIORITY["IDENTIFIER"],
                    match.start(),
                    match.end(),
                    EntityCandidate("IDENTIFIER", token, token.lower(), token.lower()),
                )
            )
    for match in QUOTED_RE.finditer(text):
        inner = match.group(1) or match.group(2) or ""
        if inner.strip():
            key = normalize_entity(inner)
            raw.append(
                (_PRIORITY["QUOTED"], match.start(), match.end(), EntityCandidate("QUOTED", inner.strip(), key, key))
            )
    run: list[re.Match] = []

    def _flush_run() -> None:
        if not run:
            return
        joined = " ".join(m.group(0) for m in run)
        key = normalize_entity(joined)
        raw.append((_PRIORITY["PROPER"], run[0].start(), run[-1].end(), EntityCandidate("PROPER", joined, key, key)))
        run.clear()

    for match in PROPER_TOKEN_RE.finditer(text):
        if match.group(0) in STOP_CAPS:
            _flush_run()
            continue
        # Consecutive capitalized tokens (separated by whitespace only) form one multi-word name.
        # 连续的首字母大写 token（仅由空白分隔）构成一个多词名称。
        if run and not text[run[-1].end() : match.start()].isspace():
            _flush_run()
        run.append(match)
    _flush_run()
    if zh_rules:
        for start, end, token in _zh_name_candidates(text):
            raw.append((_PRIORITY["ZH"], start, end, EntityCandidate("PROPER", token, token, token)))
    else:
        # The legacy lookahead is the zh_rules=False (English / baseline) behaviour only: under zh_rules the name rules
        # and the `names` glossary carry every Chinese name, and the lookahead's heads (要求团队 before 在) are fragments.
        # 既有前瞻只属于 zh_rules=False（英文 / 基线）行为：zh_rules 下人名规则与 `names` 术语表已覆盖全部中文人名，
        # 而前瞻的开头（在 之前的 要求团队）是片段。
        for match in _CJK_PROPER_RE.finditer(text):
            token = match.group(1)
            raw.append((_PRIORITY["CJK"], match.start(1), match.end(1), EntityCandidate("PROPER", token, token, token)))
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        pos = text.find(name)
        while pos >= 0:
            key = normalize_entity(name)
            raw.append((_PRIORITY["NAME"], pos, pos + len(name), EntityCandidate("PROPER", name, key, key)))
            pos = text.find(name, pos + len(name))
    claimed: list[tuple[int, int]] = []
    survivors: list[tuple[int, EntityCandidate]] = []
    for priority, start, end, candidate in sorted(raw, key=lambda item: (item[0], item[1], item[2])):
        if _overlaps((start, end), claimed):
            continue
        claimed.append((start, end))
        survivors.append((start, candidate))
    seen: set[str] = set()
    out: list[EntityCandidate] = []
    for _start, candidate in sorted(survivors, key=lambda item: item[0]):
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        out.append(candidate)
    return out


def topic_tags(text: str, *, zh_rules: bool = False) -> list[str]:
    """TOPIC_TABLE hits ordered by first-match position (ties by table order), each topic once.
    TOPIC_TABLE 命中，按首次匹配位置排序（并列按表序），每个主题一次。

    Input / 输入：
        text: Free text; underscores are treated as word boundaries so 'billing_e2e' yields 'billing'.
        text：自由文本；下划线视为词边界，故 'billing_e2e' 产生 'billing'。

        zh_rules: Also consult TOPIC_TABLE_ZH (a topic present in both tables keeps its earliest hit).
        zh_rules：同时查 TOPIC_TABLE_ZH（两表都有的主题保留最早命中）。

    Output / 输出：
        Topic names.
        主题名列表。

    State effect / 状态影响：
        None.
        无。
    """
    view = text.replace("_", " ")
    table = TOPIC_TABLE + TOPIC_TABLE_ZH if zh_rules else TOPIC_TABLE
    best: dict[str, tuple[int, int]] = {}
    for order, (pattern, topic) in enumerate(table):
        match = pattern.search(view)
        if match and (topic not in best or (match.start(), order) < best[topic]):
            best[topic] = (match.start(), order)
    return [topic for topic, _key in sorted(best.items(), key=lambda item: item[1])]


def sentiment(text: str, *, zh_rules: bool = False) -> str:
    """Lexicon sentiment: positive, negative, mixed (both) or neutral (none).
    词表情感：positive、negative、mixed（两者皆有）或 neutral（皆无）。

    Input / 输入：
        text: Free text.
        text：自由文本。

        zh_rules: Also consult the Chinese lists (_POSITIVE_SENT_ZH_RE / _NEGATIVE_SENT_ZH_RE); negated phrases such as 不喜欢 count as negative only.
        zh_rules：同时查中文词表（_POSITIVE_SENT_ZH_RE / _NEGATIVE_SENT_ZH_RE）；不喜欢 之类的否定短语只计为负面。

    Output / 输出：
        One of 'positive' | 'negative' | 'mixed' | 'neutral'.
        'positive' | 'negative' | 'mixed' | 'neutral' 之一。

    State effect / 状态影响：
        None.
        无。
    """
    if zh_rules:
        stripped = _NEGATIVE_SENT_ZH_RE.sub(" ", text)
        pos = bool(_POSITIVE_SENT_RE.search(stripped)) or bool(_POSITIVE_SENT_ZH_RE.search(stripped))
        neg = bool(_NEGATIVE_SENT_RE.search(text)) or bool(_NEGATIVE_SENT_ZH_RE.search(text))
    else:
        pos = bool(_POSITIVE_SENT_RE.search(text))
        neg = bool(_NEGATIVE_SENT_RE.search(text))
    if pos and neg:
        return "mixed"
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "neutral"


def rewrite_first_person(sentence: str) -> str:
    """Rewrite first person to third person ('I' → 'User', 'we' → 'the team'), capitalize, ensure a final period.
    将第一人称改写为第三人称（'I' → 'User'，'we' → 'the team'），首字母大写并补句号。

    Input / 输入：
        sentence: One sentence.
        sentence：一个句子。

    Output / 输出：
        The rewritten sentence.
        改写后的句子。

    State effect / 状态影响：
        None.
        无。
    """
    out = sentence.strip()
    for pattern, replacement in _FIRST_PERSON_RULES:
        out = pattern.sub(replacement, out)
    if out:
        out = out[0].upper() + out[1:]
    if out and out[-1] not in _TERMINAL_PUNCT:
        out += "."
    return out


def clean_object(phrase: str) -> str:
    """Clean an object phrase: drop trailing cues, leading articles and trailing CJK particles; collapse whitespace.
    清理宾语短语：去尾部线索词、首部冠词与尾部 了/的/。，合并空白。

    Input / 输入：
        phrase: Raw object phrase.
        phrase：原始宾语短语。

    Output / 输出：
        Cleaned phrase.
        清理后的短语。

    State effect / 状态影响：
        None.
        无。
    """
    tokens = phrase.strip().split()
    while tokens and tokens[-1].lower().strip(".,;!?") in OBJ_TRAILING_CUES:
        tokens.pop()
    while tokens and tokens[0].lower() in ARTICLES:
        tokens.pop(0)
    out = " ".join(tokens)
    out = re.sub(r"[了的。，,;]+$", "", out)
    return _WS_RE.sub(" ", out).strip()


def extract_dates(text: str, *, base_date: str) -> list[tuple[str, str, str]]:
    """Every DATE_RES hit as a (kind, gte, lte) window in text order.
    将每个 DATE_RES 命中转换为按文本顺序的 (kind, gte, lte) 窗口。

    Input / 输入：
        text: Free text.
        text：自由文本。

        base_date: 'YYYY-MM-DD' used for relative dates and months without a year.
        base_date：'YYYY-MM-DD'，用于相对日期与无年份的月份。

    Output / 输出：
        Windows; 'May' counts only when followed by a digit or preceded by in/on/during/of/since/until.
        窗口列表；'May' 仅在后接数字或前接 in/on/during/of/since/until 时视为日期。

    State effect / 状态影响：
        None.
        无。
    """
    base = date.fromisoformat(base_date[:10])
    claimed: list[tuple[int, int]] = []
    hits: list[tuple[int, int, str, str, str]] = []
    for order, (kind, pattern) in enumerate(_DATE_PATTERNS):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span[0] == span[1] or _overlaps(span, claimed):
                continue
            window = _window_for(kind, match, base, text)
            if window is None:
                continue
            claimed.append(span)
            hits.append((span[0], order, kind, window[0], window[1]))
    return [(kind, gte, lte) for _pos, _order, kind, gte, lte in sorted(hits)]


def resolve_date(text: str, *, base_date: str) -> str | None:
    """The 'YYYY-MM-DD' of the first date window's gte, or None.
    首个日期窗口 gte 的 'YYYY-MM-DD'，无则 None。

    Input / 输入：
        text: Free text.
        text：自由文本。

        base_date: 'YYYY-MM-DD' base for relative dates.
        base_date：相对日期的 'YYYY-MM-DD' 基准。

    Output / 输出：
        Date string or None.
        日期字符串或 None。

    State effect / 状态影响：
        None.
        无。
    """
    windows = extract_dates(text, base_date=base_date)
    return windows[0][1][:10] if windows else None


def has_action_closure(text: str) -> bool:
    """True when an action verb is present with an outcome or a constraint sentence.
    存在动作动词且伴随结果或约束句时为真。

    Input / 输入：
        text: Message text.
        text：消息文本。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    if not ACTION_RE.search(text):
        return False
    if OUTCOME_RE.search(text):
        return True
    return any(CONSTRAINT_RE.search(s.rstrip(_TERMINAL_PUNCT)) for s in sentences(text))


def ordered_union(*seqs: Iterable[str]) -> list[str]:
    """Concatenate sequences dropping later duplicates; never sorted (§0.1 ordered union).
    拼接序列并丢弃后出现的重复项；从不排序（§0.1 有序并集）。

    Input / 输入：
        seqs: Any number of iterables of strings.
        seqs：任意数量的字符串可迭代对象。

    Output / 输出：
        First-occurrence-ordered list.
        按首次出现顺序的列表。

    State effect / 状态影响：
        None.
        无。
    """
    seen: set[str] = set()
    out: list[str] = []
    for seq in seqs:
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def is_first_person(sentence: str) -> bool:
    """re.search for a first-person subject (I / I'm / I've / I'd / I'll / my / we, or the Chinese pronouns).
    用 re.search 检测第一人称主语（I / I'm / I've / I'd / I'll / my / we / 我 / 我们）。

    Input / 输入：
        sentence: One sentence.
        sentence：一个句子。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    return bool(_FIRST_PERSON_RE.search(sentence))


def sha256_hex(s: str) -> str:
    """SHA-256 hex digest of a UTF-8 string.
    UTF-8 字符串的 SHA-256 十六进制摘要。

    Input / 输入：
        s: Text.
        s：文本。

    Output / 输出：
        64-char hex string.
        64 字符十六进制串。

    State effect / 状态影响：
        None.
        无。
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def md5_hex(s: str) -> str:
    """MD5 hex digest of a UTF-8 string (Mem0 atom hash parity).
    UTF-8 字符串的 MD5 十六进制摘要（与 Mem0 原子哈希一致）。

    Input / 输入：
        s: Text.
        s：文本。

    Output / 输出：
        32-char hex string.
        32 字符十六进制串。

    State effect / 状态影响：
        None.
        无。
    """
    return hashlib.md5(s.encode("utf-8")).hexdigest()


__all__ = [
    "ACTION_RE",
    "ACTION_VERBS",
    "ARTICLES",
    "CAUSE_RE",
    "CJK_RE",
    "CJK_TOKEN_MODES",
    "CLAUSE_CUT",
    "CLAUSE_SPLIT_RE",
    "CONSTRAINT_RE",
    "CONSTRAINT_ZH_RE",
    "DATE_RES",
    "ENTITY_QUERY_RE",
    "EVENT_VERBS_PAST",
    "FILE_EXT_RE",
    "FUTURE_CUES",
    "HANDOFF_RE",
    "IDENT_RE",
    "MONTHS",
    "MUST_RE",
    "NEGATIVE_OUTCOMES",
    "OBJ_TRAILING_CUES",
    "OUTCOME_RE",
    "POSITIVE_OUTCOMES",
    "PREF_QUERY_RE",
    "PREF_VERBS_NEG",
    "PREF_VERBS_POS",
    "PROPER_TOKEN_RE",
    "QUOTED_RE",
    "SENTENCE_SPLIT_RE",
    "STOPWORDS_EN",
    "STOPWORDS_ZH",
    "STOP_CAPS",
    "SUPERSESSION_CUES_EN",
    "SUPERSESSION_CUES_ZH",
    "TEMPORAL_QUERY_RE",
    "TOPIC_TABLE",
    "TOPIC_TABLE_ZH",
    "VERB_LEXICON",
    "WHY_RE",
    "ZH_LEAD_ADVERBS",
    "ZH_LEAD_ADVERB_ALT",
    "ZH_NAME_STOP",
    "ZH_STATEMENT_VERBS",
    "ZH_TIME_WORDS",
    "EntityCandidate",
    "cjk_tokens_mode",
    "cjk_tokens_scope",
    "clean_object",
    "content_tokens",
    "entity_key",
    "extract_dates",
    "extract_entities",
    "has_action_closure",
    "identifier_tokens",
    "is_first_person",
    "is_identifier",
    "md5_hex",
    "normalize_entity",
    "ordered_union",
    "resolve_date",
    "rewrite_first_person",
    "sentences",
    "sentiment",
    "sha256_hex",
    "stem",
    "text_scoped",
    "tokenize",
    "topic_tags",
    "zh_name_ok",
    "zh_name_stop_prefix",
]
