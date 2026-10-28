"""Chinese statement rules (IngestConfig.zh_rules): statement_triple / predicate_of / entity / sentiment / topic behaviour with the switch on, byte-identical outputs with it off, the extractor identity, and the lines facade storing Chinese statements with a Chinese speaker.
中文陈述规则（IngestConfig.zh_rules）：开关打开时的 statement_triple / predicate_of / 实体 / 情感 / 主题行为、关闭时逐字节不变的输出、抽取器身份，以及行式门面以中文说话人存下中文陈述。
"""

# Purpose (EN): Pin CONTRACT §9.6 (b): every zh_rules branch and the untouched defaults.
# 目的（中）：钉住 CONTRACT §9.6 (b)：每个 zh_rules 分支与未触碰的默认值。

# Input (EN): Chinese third-person statements; an in-memory lines facade with a scripted LLM.
# 输入（中）：中文第三人称陈述；带脚本化 LLM 的内存行式门面。

# Output (EN): pytest assertions.
# 输出（中）：pytest 断言。

# State effect (EN): None outside in-memory stores.
# 状态影响（中）：除内存存储外无副作用。

from __future__ import annotations

from typing import Any

from hms_elite import HMS
from hms_elite.configs.base import HMSConfig, IngestConfig
from hms_elite.entities.rule import RuleEntityExtractor
from hms_elite.ingest import derive
from hms_elite.llms import rule
from hms_elite.memory.storage import StateStore
from hms_elite.utils import text as utext
from tests.conftest import FrozenClock, SequentialIds

STAMP = "2026-06-15T10:04:00Z"
ZH_MESSAGES = [
    {"role": "user", "content": "杰克: 我明天要去钓鱼。\n塔莎: 我喜欢吃披萨。"},
    {"role": "user", "content": "爱丽丝: 我们和杰克去了超市。"},
]
ZH_REPLY = "OBS\n0\tT\t杰克说他明天要去钓鱼。\n0\tP\t塔莎喜欢吃披萨。\n1\tT\t杰克和爱丽丝去了超市。\n1\tP\t我讨厌洗碗。\n"


def _facade(zh_rules: bool, responses: list[Any]) -> HMS:
    """An offline in-memory lines facade with a scripted LLM, with or without the Chinese rules.
    带脚本化 LLM 的离线内存行式门面，可开关中文规则。
    """
    store = StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds())
    config = HMSConfig(
        vector_store={"provider": "memory"},
        store={"path": ":memory:"},
        llm={"provider": "scripted", "config": {"responses": responses}},
        ingest={"extraction_output": "lines", "zh_rules": zh_rules},
    )
    return HMS(config, store=store)


def test_statement_triple_with_and_without_zh_rules():
    """With zh_rules the Chinese subject, predicate (VERB_LEXICON_ZH) and clause object are read (the leading name before a statement verb, the first-person pronoun as user, is_at / buys / plans heads, the clause after the verb as object); without it the outputs are today's (subject None except the _CJK_PROPER_RE cases, predicate states / plans / did / dislikes, objects empty), and English statements are identical in both.
    zh_rules 下读出中文主语、谓词（VERB_LEXICON_ZH）与子句宾语（杰克说… → 杰克 / plans；塔莎喜欢吃披萨 → likes / 吃披萨；杰克和爱丽丝去了超市 → 超市；我讨厌洗碗 → user / dislikes / 洗碗；卡特里娜在厨房做饭 → is_at / 厨房；塔莎打算晚上吃饺子 → plans / 饺子，时间词 晚上 与副词一样被跳过）；关闭时为今日输出（除 _CJK_PROPER_RE 情形外主语 None，谓词 states / plans / did / dislikes，宾语为空），英文陈述两者相同。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on llms.rule.statement_triple and predicate_of.
        对 llms.rule.statement_triple 与 predicate_of 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    on = {
        ("杰克说他明天要去钓鱼。", "temporal_fact"): ("杰克", "plans", ["他明天要去钓鱼"]),
        ("塔莎喜欢吃披萨。", "preference_event"): ("塔莎", "likes", ["吃披萨"]),
        ("杰克和爱丽丝去了超市。", "temporal_fact"): ("杰克", "did", ["超市"]),
        ("我讨厌洗碗。", "preference_event"): ("user", "dislikes", ["洗碗"]),
        ("卡特里娜在厨房做饭。", "fact"): ("卡特里娜", "is_at", ["厨房"]),
        ("舒尔买了一个新吉他。", "fact"): ("舒尔", "buys", ["一个新吉他"]),
        ("杰克计划明天和露西亚一起去公园。", "temporal_fact"): ("杰克", "plans", ["公园"]),
        ("塔莎打算晚上吃饺子。", "temporal_fact"): ("塔莎", "plans", ["饺子"]),
    }
    off = {
        ("杰克说他明天要去钓鱼。", "temporal_fact"): (None, "plans", []),
        ("塔莎喜欢吃披萨。", "preference_event"): ("塔莎", "likes", ["吃披萨"]),
        ("杰克和爱丽丝去了超市。", "temporal_fact"): ("杰克", "did", []),
        ("我讨厌洗碗。", "preference_event"): (None, "dislikes", []),
        ("卡特里娜在厨房做饭。", "fact"): ("卡特里娜", "states", []),
        ("舒尔买了一个新吉他。", "fact"): (None, "states", []),
        ("杰克计划明天和露西亚一起去公园。", "temporal_fact"): (None, "plans", ["计划明天"]),
        ("塔莎打算晚上吃饺子。", "temporal_fact"): (None, "did", []),
    }
    for (text, obs_type), expected in on.items():
        got = rule.statement_triple(text, obs_type, zh_rules=True)
        assert (got["subject"], got["predicate"], got["objects"]) == expected, text
        plain = rule.statement_triple(text, obs_type)
        assert (plain["subject"], plain["predicate"], plain["objects"]) == off[(text, obs_type)], text
        assert got["relation"] is None and plain["relation"] is None
    for text, obs_type in (("User likes oat-milk lattes.", "preference_event"), ("Sarah joined the billing team in March.", "fact")):
        assert rule.statement_triple(text, obs_type, zh_rules=True) == rule.statement_triple(text, obs_type)
    assert rule.predicate_of("说他明天要去", zh_rules=True) == "says" and rule.predicate_of("说他明天要去") == "states"
    assert rule.predicate_of("也喜欢黑咖啡", zh_rules=True) == "likes" and rule.predicate_of("joined the team", zh_rules=True) == "joined"
    assert derive.statement_triple is rule.statement_triple


def test_entities_sentiment_topics_and_extractor_identity():
    """extract_entities(zh_rules=True) adds 2–4 char names before a statement verb or after the with / to particles (never pronoun heads or stop words; a stop-word prefix is stripped), the configured `names` list wins over every CJK rule; the Chinese sentiment and topic lists apply only under the switch (a negated liking is negative, not mixed); the rule extractor reports rule@entities-v1 by default and rule@entities-zh-v1 when zh_rules or names are set, and HMS.__init__ forwards ingest.zh_rules to it.
    extract_entities(zh_rules=True) 加入陈述动词之前或 和 / 跟 / 给 之后的 2–4 字人名（绝不取代词开头或停用词；停用词前缀被剥离），配置的 `names` 表优先于全部 CJK 规则；中文情感与主题词表只在开关下生效（不喜欢 → negative 而非 mixed）；规则抽取器默认报告 rule@entities-v1，设置 zh_rules 或 names 时为 rule@entities-zh-v1，HMS.__init__ 把 ingest.zh_rules 转给它。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text and entities.rule.
        对 utils.text 与 entities.rule 的断言。

    State effect / 状态影响：
        None outside one in-memory store.
        除一个内存存储外无副作用。
    """
    names = lambda text, **kw: [c.text for c in utext.extract_entities(text, **kw)]  # noqa: E731
    assert names("杰克说他明天要去钓鱼。", zh_rules=True) == ["杰克"] and names("杰克说他明天要去钓鱼。") == []
    assert names("杰克和爱丽丝去了超市。", zh_rules=True) == ["杰克", "爱丽丝"] and names("杰克和爱丽丝去了超市。") == ["杰克"]
    assert names("今天杰克说要买菜", zh_rules=True) == ["杰克"] and names("他说要去", zh_rules=True) == []
    assert names("爱丽丝跟塔莎一起买菜", zh_rules=True) == ["爱丽丝", "塔莎"]
    assert names("杰克给露西亚一个苹果", zh_rules=True) == ["杰克", "露西亚"]
    assert names("卡特里娜在厨房做饭。", zh_rules=True) == ["卡特里娜"] == names("卡特里娜在厨房做饭。")
    assert names("杰克的朋友来了", names=["杰克", "露西亚"]) == ["杰克"] and names("杰克的朋友来了") == []
    # A configured name beats the legacy _CJK_PROPER_RE fragment (杰克说他 before 喜欢), which only zh_rules=False emits.
    # 配置的人名胜过既有 _CJK_PROPER_RE 片段（喜欢 之前的 杰克说他）；该片段只在 zh_rules=False 下产生。
    assert names("杰克说他喜欢吃饺子。") == ["杰克说他"] and names("杰克说他喜欢吃饺子。", names=["杰克"]) == ["杰克"]
    assert names("杰克说他喜欢吃饺子。", zh_rules=True) == ["杰克"]
    assert utext.sentiment("塔莎不喜欢洗碗", zh_rules=True) == "negative" and utext.sentiment("塔莎不喜欢洗碗") == "positive"
    assert utext.sentiment("大家玩得很开心", zh_rules=True) == "positive" and utext.sentiment("大家玩得很开心") == "neutral"
    assert utext.topic_tags("杰克买了披萨然后去钓鱼", zh_rules=True) == ["shopping", "food", "travel"]
    assert utext.topic_tags("杰克买了披萨然后去钓鱼") == []
    assert utext.topic_tags("我喜欢咖啡", zh_rules=True) == ["coffee"] == utext.topic_tags("我喜欢咖啡")
    assert utext.topic_tags("唱歌和音乐", zh_rules=True) == ["music"] and utext.topic_tags("打扫房间", zh_rules=True) == ["housework"]
    assert rule.observation_topics("塔莎喜欢吃披萨", "preference_event", zh_rules=True) == ["food", "preference"]
    assert rule.observation_topics("塔莎喜欢吃披萨", "preference_event") == ["preference"]
    plain, zh = RuleEntityExtractor(), RuleEntityExtractor({"zh_rules": True, "names": ["杰克"]})
    assert plain.identity == "rule@entities-v1" and zh.identity == "rule@entities-zh-v1"
    assert RuleEntityExtractor({"names": ["杰克"]}).identity == "rule@entities-zh-v1"
    assert plain.extract("杰克说他明天要去钓鱼。") == utext.extract_entities("杰克说他明天要去钓鱼。")
    assert [c.text for c in zh.extract("今天杰克说要买菜")] == ["杰克"]
    assert IngestConfig().zh_rules is False
    hms = HMS({"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}, "ingest": {"zh_rules": True}})
    assert hms.entity_extractor.identity == "rule@entities-zh-v1" and hms.entity_extractor.config.zh_rules is True
    hms.close()
    hms = HMS({"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}})
    assert hms.entity_extractor.identity == "rule@entities-v1"
    hms.close()


def test_lines_facade_stores_chinese_statements_with_chinese_speakers():
    """A scripted lines reply in Chinese through HMS.add with zh_rules stores the statements verbatim with the Chinese subject as speaker / actor_id, its entity key in entity_ids, VERB_LEXICON_ZH predicates and the Chinese topic; without zh_rules the same reply stores the same texts but the subject falls back to the message side where no _CJK_PROPER_RE name leads the statement.
    带 zh_rules 时，中文的脚本化行式回复经 HMS.add 原样存下陈述，中文主语成为 speaker / actor_id，其实体键进入 entity_ids，谓词来自 VERB_LEXICON_ZH，主题为中文主题；不带 zh_rules 时同一回复存下相同文本，但没有 _CJK_PROPER_RE 人名开头的陈述主语回退到消息侧别。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the add() rows and the atoms.
        对 add() 行与原子的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    hms = _facade(True, [ZH_REPLY])
    rows = hms.add(ZH_MESSAGES, user_id="alice", run_id="run-zh", timestamp=STAMP)["results"]
    assert [r["memory"] for r in rows] == ["杰克说他明天要去钓鱼。", "塔莎喜欢吃披萨。", "杰克和爱丽丝去了超市。", "我讨厌洗碗。"]
    assert [r["event"] for r in rows] == ["ADD"] * 4
    assert [r["hms"]["subject"] for r in rows] == ["杰克", "塔莎", "杰克", "user"]
    assert [r["hms"]["predicate"] for r in rows] == ["plans", "likes", "did", "dislikes"]
    assert [r.get("actor_id") for r in rows] == ["杰克", "塔莎", "杰克", None]
    assert rows[0]["hms"]["entity_ids"] == ["杰克"] and rows[2]["hms"]["entity_ids"] == ["杰克", "爱丽丝"]
    assert "food" in rows[1]["hms"]["topic_tags"] and rows[1]["hms"]["sentiment"] == "positive"
    assert rows[3]["hms"]["sentiment"] == "negative" and "housework" in rows[3]["hms"]["topic_tags"]
    atoms = hms.store.get_atoms([r["atom_id"] for r in rows])
    assert [a.content["objects"] for a in atoms] == [["他明天要去钓鱼"], ["吃披萨"], ["超市"], ["洗碗"]]
    assert all(utext.CJK_RE.search(a.text) for a in atoms)
    assert hms.last_downgraded == []
    hms.close()

    # HMS.last_downgraded carries the guard-downgraded verdicts of the latest add() (the Mem0 envelope cannot).
    # HMS.last_downgraded 带有最近一次 add() 被守卫降级的裁决（Mem0 信封装不下）。
    guarded = _facade(True, [ZH_REPLY, "OBS\n0\tF\t杰克买了一把新吉他。\nDEC\n1\tNONE\t0\tsame\n"])
    guarded.add(ZH_MESSAGES, user_id="alice", run_id="run-zh", timestamp=STAMP)
    rows = guarded.add([{"role": "user", "content": "杰克: 我买了一把新吉他。"}], user_id="alice", run_id="run-zh", timestamp="2026-06-15T11:04:00Z")["results"]
    assert set(rows[0]) >= {"memory", "event"} and rows[0]["event"] == "ADD" and "downgraded" not in guarded.add.__wrapped__.__annotations__.get("return", "")
    assert [(d["kind"], d["verdict"], d["text"]) for d in guarded.last_downgraded] == [("verdict", "NONE", "杰克买了一把新吉他。")]
    guarded.close()

    plain = _facade(False, [ZH_REPLY])
    rows = plain.add(ZH_MESSAGES, user_id="alice", run_id="run-zh", timestamp=STAMP)["results"]
    assert [r["memory"] for r in rows] == ["杰克说他明天要去钓鱼。", "塔莎喜欢吃披萨。", "杰克和爱丽丝去了超市。", "我讨厌洗碗。"]
    assert [r["hms"]["subject"] for r in rows] == ["user", "塔莎", "杰克", "user"]
    assert [r.get("actor_id") for r in rows] == [None, "塔莎", "杰克", None]
    assert rows[0]["hms"]["entity_ids"] == [] and rows[2]["hms"]["entity_ids"] == ["杰克"]
    plain.close()


def test_name_fragments_particles_time_words_and_coverbs():
    """The zh_rules name rules reject heads that contain a statement verb or a time word (verb+time, name+time and locative fragments), skip an aspect particle after a coverb, and strip time words as prefixes; statement_triple reads the subject after a leading time word, skips a time word between subject and verb, takes the descended verb's predicate after a coverb (went_to / says / is_at / gives, never 'with', which exists in no predicate vocabulary) and keeps plans for a planning verb; every English statement and every zh_rules=False output is unchanged.
    zh_rules 的人名规则拒绝含陈述动词或时间词的开头（打算晚上、杰克晚上、在客厅、塔莎打扫 都是片段），跳过 和 / 跟 / 给 之后的 了 / 过 / 着（跟了塔莎 → 塔莎），时间词作为前缀剥离（下午杰克 → 杰克）；statement_triple 在开头时间词之后读主语（今天杰克和爱丽丝一起去了公园 → 杰克 / 公园），跳过主语与动词之间的 晚上，介动词之后取下钻动词的谓词（爱丽丝和塔莎去了超市 → went_to；跟杰克说 → says；和舒尔在阳台 → is_at；给了杰克一杯咖啡 → gives / 一杯咖啡）——'with' 不在任何谓词词表里——计划 保持 plans；英文陈述与 zh_rules=False 的输出全部不变。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text.extract_entities / zh_name_ok / zh_name_stop_prefix and llms.rule.statement_triple / predicate_of / VERB_LEXICON_ZH.
        对 utils.text.extract_entities / zh_name_ok / zh_name_stop_prefix 与 llms.rule.statement_triple / predicate_of / VERB_LEXICON_ZH 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    names = lambda text, **kw: [c.text for c in utext.extract_entities(text, **kw)]  # noqa: E731
    # (a) fragments that are not names — no glossary, zh_rules alone.
    # (a) 不是人名的片段——不带术语表，只开 zh_rules。
    assert names("我们打算晚上吃火锅", zh_rules=True) == [] and names("晚上去了超市", zh_rules=True) == []
    assert names("杰克晚上去了超市", zh_rules=True) == ["杰克"] and names("下午杰克和塔莎打扫了厨房", zh_rules=True) == ["杰克"]
    assert "在客厅" not in names("大家在客厅玩游戏", zh_rules=True)
    assert names("杰克跟了塔莎去超市", zh_rules=True) == ["杰克", "塔莎"] and names("露西亚给了杰克一杯咖啡", zh_rules=True) == ["露西亚", "杰克"]
    assert names("杰克给了塔莎一本书", zh_rules=True) == ["杰克", "塔莎"]
    assert all(not utext.zh_name_ok(h) for h in ("打算晚上", "杰克晚上", "在客厅", "我们", "今天", "他说", "已经"))
    assert all(utext.zh_name_ok(h) for h in ("杰克", "塔莎", "卡特里娜", "乔伊斯特", "爱丽丝", "露西亚"))
    assert utext.zh_name_stop_prefix("今天杰克说") == 2 and utext.zh_name_stop_prefix("杰克说") == 0 and utext.zh_name_stop_prefix("今天") == 0
    assert utext.ZH_TIME_WORDS <= utext.ZH_NAME_STOP and "晚上" in utext.ZH_LEAD_ADVERBS and "一起" in utext.ZH_LEAD_ADVERBS
    assert utext.ZH_LEAD_ADVERB_ALT == "|".join(utext.ZH_LEAD_ADVERBS) and rule._ZH_LEAD_ADVERB_RE.match("晚上一起去") is not None
    # Every zh_rules=False output on the same texts is the historical one (the CJK rule alone).
    # 同样文本在 zh_rules=False 下的输出全是历史输出（只有 CJK 规则）。
    assert names("杰克晚上去了超市") == [] and names("露西亚给了杰克一杯咖啡") == [] and names("今天杰克和爱丽丝一起去了公园。") == ["今天杰克"]
    # (b) statement_triple after a time word, across coverbs, with 了 after 给.
    # (b) 开头时间词之后、穿过介动词、给 之后有 了 的 statement_triple。
    cases = {
        ("今天杰克和爱丽丝一起去了公园。", "temporal_fact"): ("杰克", "did", ["公园"]),
        ("杰克晚上去了超市。", "temporal_fact"): ("杰克", "did", ["超市"]),
        ("下午我去了超市。", "temporal_fact"): ("user", "did", ["超市"]),
        ("爱丽丝和塔莎去了超市。", "fact"): ("爱丽丝", "went_to", ["超市"]),
        ("塔莎跟杰克说她累了。", "fact"): ("塔莎", "says", ["她累"]),
        ("乔伊斯特和舒尔在阳台聊天。", "fact"): ("乔伊斯特", "is_at", ["阳台聊天"]),
        ("露西亚给了杰克一杯咖啡。", "fact"): ("露西亚", "gives", ["一杯咖啡"]),
        ("杰克给了塔莎一本书。", "fact"): ("杰克", "gives", ["一本书"]),
        ("杰克跟了塔莎去超市。", "fact"): ("杰克", "went_to", ["超市"]),
        ("杰克计划明天和露西亚一起去公园。", "fact"): ("杰克", "plans", ["公园"]),
    }
    for (text, obs_type), expected in cases.items():
        got = rule.statement_triple(text, obs_type, zh_rules=True)
        assert (got["subject"], got["predicate"], got["objects"]) == expected, text
        assert got["predicate"] != "with"
    assert "with" not in rule.VERB_LEXICON_ZH.values() and rule.VERB_LEXICON_ZH["和"] == rule.VERB_LEXICON_ZH["跟"] == "states"
    assert rule.predicate_of("和塔莎去了超市", zh_rules=True) == "went_to" and rule.predicate_of("和塔莎一起", zh_rules=True) == "states"
    assert rule.predicate_of("给了杰克一杯咖啡", zh_rules=True) == "gives"
    # The zh_rules=False outputs of these statements are the historical ones.
    # 这些陈述在 zh_rules=False 下的输出是历史输出。
    off = rule.statement_triple("爱丽丝和塔莎去了超市。", "fact")
    assert (off["subject"], off["predicate"], off["objects"]) == ("爱丽丝", "states", [])
    off = rule.statement_triple("今天杰克和爱丽丝一起去了公园。", "temporal_fact")
    assert (off["subject"], off["predicate"], off["objects"]) == ("今天杰克", "did", [])
    for text, obs_type in (("Sarah went to the market with Tom.", "fact"), ("User likes oat-milk lattes.", "preference_event")):
        assert rule.statement_triple(text, obs_type, zh_rules=True) == rule.statement_triple(text, obs_type)


def test_review_fixes_sentiment_names_pronouns_and_read_side_entity_axis():
    """Review fixes of 2026-09-22: the Chinese positive list carries no bare 'love' character, so a statement naming Alice (whose glossary form starts with it) is not positive by her name (Alice hates dishwashing → negative, Alice went to the supermarket → neutral); zh_name_ok rejects a statement verb or lead adverb only at the head's edges (a name with a verb character inside is accepted; one ending in an adverb character stays rejected — documented limitation, the `names` list catches it); a leading pronoun run other than the first person is stepped over so the verb after it is read (we plan → project / plans / hotpot; she likes → None / likes; he says → None / says); and the read side of the entity axis uses the facade's extractor (zh_rules + names): a Chinese question yields entity hints, an entity_boost > 0 and 'entity' among recalled_by, while the plain facade yields none of them.
    2026-09-22 的评审修正：中文正面词表不含裸字 爱，因此提到 爱丽丝 的陈述不因其名字而为正面（爱丽丝讨厌洗碗 → negative，爱丽丝去了超市 → neutral）；zh_name_ok 只在开头的两端拒绝陈述动词或前导副词（王和平 是人名；李刚 仍被拒绝——已记录的限制，`names` 表可接住）；开头的非 我 代词被跨过，使其后的动词被读取（我们打算晚上吃火锅 → project / plans / 火锅；她喜欢吃披萨 → None / likes / 吃披萨；他说他明天要去钓鱼 → None / says）；实体轴的读侧使用门面的抽取器（zh_rules + names）：中文提问得到实体线索、entity_boost > 0 且 recalled_by 含 'entity'，而普通门面三者皆无。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text, llms.rule, mvmi.planner and HMS.search(explain=True).
        对 utils.text、llms.rule、mvmi.planner 与 HMS.search(explain=True) 的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    from hms_elite.mvmi.planner import plan_query

    # (a) sentiment: 爱丽丝 is a name, not a liking.
    # (a) 情感：爱丽丝 是人名，不是喜爱。
    assert utext.sentiment("爱丽丝讨厌洗碗", zh_rules=True) == "negative" and utext.sentiment("爱丽丝去了超市", zh_rules=True) == "neutral"
    assert utext.sentiment("杰克讨厌洗碗", zh_rules=True) == "negative" and utext.sentiment("塔莎爱吃披萨", zh_rules=True) == "positive"
    assert utext.sentiment("爱丽丝去了超市") == "neutral" and "爱" not in utext._POSITIVE_SENT_ZH_RE.pattern.split("|")
    # (b) zh_name_ok: edge test only.
    # (b) zh_name_ok：只看两端。
    assert utext.zh_name_ok("王和平") and utext.zh_name_ok("杰克") and not utext.zh_name_ok("李刚") and not utext.zh_name_ok("在客厅") and not utext.zh_name_ok("杰克晚上")
    names = lambda text, **kw: [c.text for c in utext.extract_entities(text, **kw)]  # noqa: E731
    # 李刚 is never a ZH candidate; until `names` lists him the 4-char ZH head 李刚说他 (before 明天要) remains — a
    # _zh_name_candidates head, not a _CJK_PROPER_RE one (the lookahead is off under zh_rules and yields nothing here anyway).
    # 李刚 永不成为 ZH 候选；在 `names` 列出他之前剩下的是 4 字 ZH 开头 李刚说他（明天要 之前）——它来自
    # _zh_name_candidates 而非 _CJK_PROPER_RE（前瞻在 zh_rules 下不运行，且在此本就什么也不产出）。
    assert names("王和平说他明天要去钓鱼。", zh_rules=True) == ["王和平"] and names("李刚说他明天要去钓鱼。", zh_rules=True) == ["李刚说他"]
    assert names("李刚说他明天要去钓鱼。", zh_rules=True, names=("李刚",)) == ["李刚"]
    # (c) pronoun heads: the verb after 我们 / 他 / 她 / 他们 / 大家 is read.
    # (c) 代词开头：我们 / 他 / 她 / 他们 / 大家 之后的动词被读取。
    cases = {
        ("我们打算晚上吃火锅。", "fact"): ("project", "plans", ["火锅"]),
        ("她喜欢吃披萨。", "preference_event"): (None, "likes", ["吃披萨"]),
        ("他说他明天要去钓鱼。", "fact"): (None, "says", ["他明天要去钓鱼"]),
        ("我们和杰克去了超市。", "fact"): ("project", "went_to", ["超市"]),
        ("大家在客厅玩游戏。", "fact"): (None, "is_at", ["客厅"]),
    }
    for (text, obs_type), expected in cases.items():
        got = rule.statement_triple(text, obs_type, zh_rules=True)
        assert (got["subject"], got["predicate"], got["objects"]) == expected, text
        plain = rule.statement_triple(text, obs_type)
        assert plain["predicate"] in ("states", "plans", "likes", "did") and plain["objects"] in ([], ["吃披萨"]), text
    # (d) read side of the entity axis: plan_query with the facade's extractor, HMS.search explain.
    # (d) 实体轴读侧：带门面抽取器的 plan_query，HMS.search 的 explain。
    reply = "OBS\n0\tF\t杰克买了一把新吉他。\n0\tF\t塔莎说爱丽丝很累。\n0\tF\t露西亚在厨房做饭。\n"
    messages = [{"role": "user", "content": "杰克: 我买了一把新吉他。\n塔莎: 爱丽丝很累。\n露西亚: 我在厨房做饭。"}]
    small_k = {"k_semantic": 1, "k_keyword": 1, "k_entity": 1, "k_recency": 1, "pool_max_factor": 1, "hard_gate_pool": False}
    zh = HMS(HMSConfig(
        vector_store={"provider": "memory"}, store={"path": ":memory:"}, llm={"provider": "scripted", "config": {"responses": [reply]}},
        ingest={"extraction_output": "lines", "zh_rules": True, "entity_extractor_config": {"names": ["杰克", "爱丽丝", "塔莎"]}}, retrieval=small_k,
    ), store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()))
    rows = zh.add(messages, user_id="u", timestamp=STAMP)["results"]
    assert [r["hms"]["entity_ids"] for r in rows] == [["杰克"], ["塔莎", "爱丽丝"], ["露西亚"]]
    plan = plan_query("杰克买了什么？", as_of=STAMP, extractor=zh.entity_extractor).as_dict()
    assert [h["key"] for h in plan["entity_hints"]] == ["杰克"] and plan_query("杰克买了什么？", as_of=STAMP).as_dict()["entity_hints"] == []
    hit = zh.search("杰克买了什么？", filters={"user_id": "u"}, explain=True, threshold=0.0, top_k=1)["results"][0]
    assert hit["memory"] == "杰克买了一把新吉他。" and hit["score_details"]["entity_boost"] > 0 and "entity" in hit["score_details"]["recalled_by"]
    assert hit["score_details"]["channel_sizes"].get("entity") == 1
    zh.close()
    plain = HMS(HMSConfig(
        vector_store={"provider": "memory"}, store={"path": ":memory:"}, llm={"provider": "scripted", "config": {"responses": [reply]}},
        ingest={"extraction_output": "lines"}, retrieval=small_k,
    ), store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()))
    rows = plain.add(messages, user_id="u", timestamp=STAMP)["results"]
    assert [r["hms"]["entity_ids"] for r in rows] == [[], [], ["露西亚"]]
    hit = plain.search("杰克买了什么？", filters={"user_id": "u"}, explain=True, threshold=0.0, top_k=1)["results"][0]
    assert hit["score_details"]["entity_boost"] == 0.0 and "entity" not in hit["score_details"]["recalled_by"] and "entity" not in hit["score_details"]["channel_sizes"]
    plain.close()


def test_zh_rules_skip_the_legacy_cjk_lookahead():
    """Under zh_rules the legacy _CJK_PROPER_RE lookahead does not run, so its 2–4 char heads before the copula / locative / with / like / join / in-charge cue characters (the six EgoLife-ZH smoke fragments) are never entities at the function, the extractor or the lines-facade level (entity_ids); the entities of a zh_rules text are the `names` glossary hits and the _zh_name_candidates heads (Jake / Lucia / Shure, Lucia), while every zh_rules=False output on the same texts is pinned to the output captured before the change (the lookahead alone, or with `names`).
    zh_rules 下既有的 _CJK_PROPER_RE 前瞻不运行，因此它在 是 / 在 / 和 / 喜欢 / 加入 / 负责 之前取的 2–4 字开头（EgoLife-ZH 冒烟片段 要求团队 / 都写 / 硬盘假装 / 他的家乡 / 做的布线 / 并让大家）在函数、抽取器与行式门面（entity_ids）三层都不成为实体；zh_rules 文本的实体是 `names` 术语表命中与 _zh_name_candidates 开头（杰克 / 露西亚 / 舒尔, 露西亚），而同样文本在 zh_rules=False 下的每个输出都钉在改动前捕获的输出上（只有前瞻，或前瞻加 `names`）。

    Input / 输入：
        The three smoke sentences, five sentences carrying the other smoke fragments, an in-memory lines facade with a scripted reply.
        三条冒烟句、五条带其余冒烟片段的句子、带脚本化回复的内存行式门面。

    Output / 输出：
        Assertions on utils.text.extract_entities, entities.rule.RuleEntityExtractor and HMS.add rows.
        对 utils.text.extract_entities、entities.rule.RuleEntityExtractor 与 HMS.add 行的断言。

    State effect / 状态影响：
        None outside two in-memory stores.
        除两个内存存储外无副作用。
    """
    names = lambda text, **kw: [c.text for c in utext.extract_entities(text, **kw)]  # noqa: E731
    glossary = ["杰克", "露西亚", "舒尔", "塔莎", "爱丽丝", "卡特里娜", "乔伊斯特"]
    fragments = {"要求团队", "都写", "硬盘假装", "他的家乡", "做的布线", "并让大家"}
    # (a) The three smoke sentences: clean names under zh_rules, the pinned legacy output without it.
    # (a) 三条冒烟句：zh_rules 下为干净人名，关闭时为钉住的既有输出。
    smoke = {
        "杰克要求团队在讨论开始前标记一个时间戳。": (["杰克"], ["杰克"], ["要求团队"], ["杰克", "要求团队"]),
        "露西亚以前拆过这个硬盘，并被称为导师。": (["露西亚"], [], [], ["露西亚"]),
        "舒尔告诉露西亚如果需要充电宝可以从这里拿一个。": (["舒尔", "露西亚"], ["舒尔"], [], ["舒尔", "露西亚"]),
    }
    for text, (zh_glossary, zh_alone, legacy, legacy_glossary) in smoke.items():
        assert names(text, zh_rules=True, names=glossary) == zh_glossary, text
        assert names(text, zh_rules=True) == zh_alone, text
        assert names(text) == legacy and names(text, names=glossary) == legacy_glossary, text
    # (b) Each remaining smoke fragment: emitted by zh_rules=False alone, never under zh_rules (with or without the glossary).
    # (b) 其余每个冒烟片段：只有 zh_rules=False 产出，zh_rules 下（带或不带术语表）从不产出。
    carriers = {
        "每个人的名字，都写在纸上。": ("都写", []),
        "他把这个硬盘假装是新的。": ("硬盘假装", []),
        "杰克说他的家乡在山东。": ("他的家乡", ["杰克"]),
        "舒尔做的布线是最好的。": ("做的布线", ["舒尔"]),
        "露西亚并让大家在这里等。": ("并让大家", ["露西亚"]),
    }
    for text, (fragment, zh_glossary) in carriers.items():
        assert fragment in fragments and names(text) == [fragment], text
        assert names(text, zh_rules=True, names=glossary) == zh_glossary, text
        assert fragments.isdisjoint(names(text, zh_rules=True)) and fragments.isdisjoint(names(text, zh_rules=True, names=glossary)), text
    # 李刚说他 is a _zh_name_candidates head (before 明天要), not a lookahead fragment: zh_rules=False yields nothing here.
    # 李刚说他 是 _zh_name_candidates 的开头（明天要 之前），不是前瞻片段：zh_rules=False 在此什么也不产出。
    assert names("李刚说他明天要去钓鱼。") == [] and names("李刚说他明天要去钓鱼。", zh_rules=True) == ["李刚说他"]
    # (c) The configured extractor and the lines facade: entity_ids carry the names only; the plain facade keeps today's rows.
    # (c) 配置的抽取器与行式门面：entity_ids 只含人名；普通门面保持今日的行。
    zh_extractor = RuleEntityExtractor({"zh_rules": True, "names": glossary})
    assert [c.text for c in zh_extractor.extract("杰克要求团队在讨论开始前标记一个时间戳。")] == ["杰克"]
    assert [c.text for c in RuleEntityExtractor().extract("杰克要求团队在讨论开始前标记一个时间戳。")] == ["要求团队"]
    reply = "OBS\n0\tF\t杰克要求团队在讨论开始前标记一个时间戳。\n0\tF\t舒尔告诉露西亚如果需要充电宝可以从这里拿一个。\n0\tF\t露西亚以前拆过这个硬盘，并被称为导师。\n"
    messages = [{"role": "user", "content": "杰克: 大家在讨论开始前标记一个时间戳。\n舒尔: 露西亚，需要充电宝可以从这里拿一个。\n塔莎: 露西亚以前拆过这个硬盘。"}]
    expected = {False: [["要求团队"], [], []], True: [["杰克"], ["舒尔", "露西亚"], ["露西亚"]]}
    for zh_rules, entity_ids in expected.items():
        ingest: dict[str, Any] = {"extraction_output": "lines", "zh_rules": zh_rules}
        if zh_rules:
            ingest["entity_extractor_config"] = {"names": glossary[:4]}
        hms = HMS(HMSConfig(
            vector_store={"provider": "memory"}, store={"path": ":memory:"}, llm={"provider": "scripted", "config": {"responses": [reply]}}, ingest=ingest,
        ), store=StateStore(":memory:", now=FrozenClock().now, id_factory=SequentialIds()))
        rows = hms.add(messages, user_id="u", timestamp=STAMP)["results"]
        assert [r["hms"]["entity_ids"] for r in rows] == entity_ids, zh_rules
        hms.close()
