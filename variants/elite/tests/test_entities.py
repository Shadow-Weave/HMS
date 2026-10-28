"""Entity extractor tests: rule parity, span merge with a fake spaCy pipeline, dependency guards, facade wiring.
实体抽取器测试：规则一致性、用假 spaCy 流水线测试区间合并、依赖守卫、门面接线。
"""

# Purpose (EN): Prove the `rule` provider is byte-identical to utils.text.extract_entities, that spaCy spans merge
#               into (never replace) the rule result, and that the configured extractor reaches ingest, the rule LLM,
#               feedback corrections and package import.
# 目的（中）：证明 `rule` 提供者与 utils.text.extract_entities 逐字节一致，spaCy 区间并入（而非替换）规则结果，
#           以及配置中选定的抽取器能到达摄取、规则 LLM、反馈纠正与交接包导入。

# Input (EN): The entities package, the factories, HMSConfig, an in-memory HMS; a fake spaCy pipeline object.
# 输入（中）：entities 包、工厂、HMSConfig、内存 HMS；一个假的 spaCy 流水线对象。

# Output (EN): pytest results; the real-model test skips without spacy / en_core_web_sm.
# 输出（中）：pytest 结果；真实模型测试在缺少 spacy / en_core_web_sm 时跳过。

# State effect (EN): In-memory stores only; REGISTRY gains and loses the 'tag' provider inside one test.
# 状态影响（中）：仅内存存储；REGISTRY 在单个测试内增删 'tag' 提供者。

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, ClassVar

import pydantic
import pytest

from hms_elite.configs.base import HMSConfig, IngestConfig
from hms_elite.configs.registry import REGISTRY, supported
from hms_elite.entities import (
    DEFAULT_LABELS,
    BaseEntityExtractorConfig,
    EntityExtractorBase,
    RuleEntityExtractor,
    SpacyEntityExtractor,
    SpacyEntityExtractorConfig,
    merge_candidates,
)
from hms_elite.exceptions import ConfigError, DependencyError
from hms_elite.index.memory import MemoryIndex
from hms_elite.ingest.atoms import link_atom_axes
from hms_elite.ingest.pipeline import Ingestor
from hms_elite.llms.rule import RuleLLM
from hms_elite.memory.main import HMS
from hms_elite.models import MemoryAtom, Scope
from hms_elite.utils.factory import EntityExtractorFactory, LlmFactory, register_provider
from hms_elite.utils.text import EntityCandidate, extract_entities

CORPUS = [
    "Sarah joined the billing team in March.",
    "file_diff src/billing/retry.ts: add exponential backoff (intent: fix checkout webhook retry)",
    "'patch src/billing/retry.ts: add exponential backoff' failed",
    "Yesterday I visited the Berlin office with Sarah Connor in New York.",
    "Do not modify the subscription state machine.",
    "我现在在上海负责计费系统，张伟是我的同事。",
    "Ticket ABC-123 and #42 touch config.yaml and billing_e2e.",
    "",
    "   ",
]


@dataclass
class FakeEnt:
    """One fake spaCy entity span.
    一个假的 spaCy 实体区间。
    """

    text: str
    label_: str
    start_char: int
    end_char: int


class FakeDoc:
    """A fake spaCy Doc exposing only `.ents`.
    只暴露 `.ents` 的假 spaCy Doc。
    """

    def __init__(self, ents: list[FakeEnt]) -> None:
        """Hold the spans.
        持有区间。
        """
        self.ents = ents


class FakeNLP:
    """A fake spaCy pipeline: (surface, label) pairs located in the text by their first occurrence.
    假的 spaCy 流水线：按首次出现位置在文本中定位 (表面, 标签) 对。
    """

    def __init__(self, mentions: list[tuple[str, str]]) -> None:
        """Remember the mentions to emit.
        记住要输出的提及。
        """
        self.mentions = mentions
        self.calls = 0

    def __call__(self, text: str) -> FakeDoc:
        """Locate every mention in `text` (missing ones are skipped).
        在 `text` 中定位每个提及（找不到的跳过）。
        """
        self.calls += 1
        ents: list[FakeEnt] = []
        for surface, label in self.mentions:
            start = text.find(surface)
            if start >= 0:
                ents.append(FakeEnt(surface, label, start, start + len(surface)))
        return FakeDoc(ents)


class TagExtractorConfig(BaseEntityExtractorConfig):
    """Config of the test provider: the PROPER tag appended to every non-empty text.
    测试提供者的配置：追加到每段非空文本的 PROPER 标签。
    """

    tag: str = "Zorblax"


class TagExtractor(EntityExtractorBase):
    """Test provider: the rule result plus one extra PROPER candidate (`config.tag`) for every non-empty text.
    测试提供者：规则结果加上每段非空文本一个额外的 PROPER 候选（`config.tag`）。
    """

    config_class: ClassVar[type[BaseEntityExtractorConfig]] = TagExtractorConfig
    provider_name: ClassVar[str] = "tag"

    def extract(self, text: str) -> list[EntityCandidate]:
        """Rule candidates followed by the tag candidate (key = lower-cased tag).
        规则候选之后追加标签候选（key 为小写标签）。

        Input / 输入：
            text: Free text.
            text：自由文本。

        Output / 输出：
            extract_entities(text) + [tag] when the text is non-blank.
            文本非空时为 extract_entities(text) + [标签]。

        State effect / 状态影响：
            None.
            无。
        """
        base = extract_entities(text)
        if not text.strip():
            return base
        tag = self.config.tag
        return base + [EntityCandidate("PROPER", tag, tag.lower(), tag.lower())]


def _keys(candidates: list[EntityCandidate]) -> list[str]:
    """Keys of a candidate list.
    候选列表的 key。
    """
    return [c.key for c in candidates]


def _entity_keys_of(hms: HMS, memory_id: str) -> set[str]:
    """normalized_text of every entity linked to the head atom of a lineage.
    某谱系头原子所链接实体的 normalized_text 集合。
    """
    head = hms.store.resolve_head(memory_id)
    assert head is not None
    return {row["normalized_text"] for row in hms.store.entities_for_atom(head.atom_id)}


# ---------------------------------------------------------------------------
# 1. The rule provider is the function
# 1. 规则提供者就是那个函数
# ---------------------------------------------------------------------------
def test_rule_extractor_is_byte_identical_and_registered():
    """RuleEntityExtractor == extract_entities on the corpus; registry kind, factory, config default and CONFIG_002 paths.
    RuleEntityExtractor 在语料上与 extract_entities 完全一致；注册表种类、工厂、配置默认值与 CONFIG_002 路径。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the extractor, EntityExtractorFactory, REGISTRY, IngestConfig and link_atom_axes.
        对抽取器、EntityExtractorFactory、REGISTRY、IngestConfig 与 link_atom_axes 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    rule = RuleEntityExtractor()
    for text in CORPUS:
        assert rule.extract(text) == extract_entities(text), text
    assert rule.identity == "rule@entities-v1" and rule.provider_name == "rule"
    assert supported("entity_extractor") == ["rule", "spacy"]
    assert EntityExtractorFactory.get_supported_providers() == ["rule", "spacy"]
    assert REGISTRY["entity_extractor"]["rule"] == (
        "hms_elite.entities.rule.RuleEntityExtractor",
        "hms_elite.entities.rule.RuleEntityExtractorConfig",
    )
    built = EntityExtractorFactory.create("rule")
    assert isinstance(built, RuleEntityExtractor) and built.extract(CORPUS[3]) == extract_entities(CORPUS[3])
    with pytest.raises(ConfigError) as info:
        EntityExtractorFactory.create("regex")
    assert info.value.error_code == "CONFIG_002" and info.value.details == {
        "kind": "entity_extractor",
        "provider": "regex",
        "supported": ["rule", "spacy"],
    }
    with pytest.raises(ConfigError) as info:
        EntityExtractorFactory.create("rule", {"bogus": 1})
    assert info.value.error_code == "CONFIG_001" and "bogus" in str(info.value)
    # Config surface: default 'rule', an empty config dict, None coerced to {}, unknown names rejected on both paths.
    # 配置面：默认 'rule'、空配置字典、None 规整为 {}、两条路径都拒绝未知名称。
    defaults = HMSConfig()
    assert defaults.ingest.entity_extractor == "rule" and defaults.ingest.entity_extractor_config == {}
    assert IngestConfig(entity_extractor_config=None).entity_extractor_config == {}
    with pytest.raises(ConfigError) as info:
        HMSConfig.from_dict({"ingest": {"entity_extractor": "regex"}})
    assert info.value.error_code == "CONFIG_002" and info.value.details == {
        "kind": "entity_extractor",
        "provider": "regex",
        "supported": ["rule", "spacy"],
    }
    with pytest.raises(pydantic.ValidationError):
        IngestConfig(entity_extractor="regex")
    spacy_cfg = HMSConfig.from_dict({"ingest": {"entity_extractor": "spacy", "entity_extractor_config": {"model": "x"}}})
    assert (spacy_cfg.ingest.entity_extractor, spacy_cfg.ingest.entity_extractor_config) == ("spacy", {"model": "x"})
    # link_atom_axes: extractor=None and the rule provider agree on keys and topics.
    # link_atom_axes：extractor=None 与规则提供者在实体键与主题上一致。
    atom = MemoryAtom(
        atom_type="state",
        content={"text": CORPUS[1]},
        text=CORPUS[1],
        scale_level="state",
        source_kind="observation",
        source_id="obs_x",
        extracted_by="rule@obs-v1",
        scope=Scope(user_id="alice"),
        anchor_ids=["anc_000001"],
    )
    plain = link_atom_axes(atom, extra_mentions=["Sarah Connor", "src/billing/retry.ts"])
    assert plain == link_atom_axes(atom, extra_mentions=["Sarah Connor", "src/billing/retry.ts"], extractor=rule)
    assert [k[0] for k in plain[0]] == ["file_diff", "retry.ts", "sarah connor"]
    # The rule LLM binds a rule extractor by default; LlmFactory forwards `entity_extractor` only to constructors
    # that declare it.
    # 规则 LLM 默认绑定规则抽取器；LlmFactory 只把 `entity_extractor` 转交给声明了它的构造函数。
    assert isinstance(RuleLLM().entity_extractor, RuleEntityExtractor)
    custom = TagExtractor()
    bound = LlmFactory.create("rule", {"language": "en"}, entity_extractor=custom)
    assert isinstance(bound, RuleLLM) and bound.entity_extractor is custom and bound.config.language == "en"
    scripted = LlmFactory.create("scripted", {"responses": ["a"]}, entity_extractor=custom)
    assert not hasattr(scripted, "entity_extractor") and scripted.config.responses == ["a"]


# ---------------------------------------------------------------------------
# 2. Merge semantics with a fake pipeline (no model needed)
# 2. 用假流水线验证合并语义（不需要模型）
# ---------------------------------------------------------------------------
def test_spacy_merge_keeps_rule_candidates_and_adds_ner_spans():
    """Rule spans win overlaps, unmapped labels are dropped, keys dedup, articles/possessives are stripped, order is textual.
    规则区间在重叠时胜出，未映射标签被丢弃，key 去重，冠词/所有格被去掉，顺序按文本。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over SpacyEntityExtractor(nlp=FakeNLP) and merge_candidates.
        对 SpacyEntityExtractor(nlp=FakeNLP) 与 merge_candidates 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    text = "Sarah Connor patched src/billing/retry.ts for the billing team at Acme Corp in Berlin yesterday."
    nlp = FakeNLP(
        [
            ("Sarah Connor", "PERSON"),
            ("the billing team", "ORG"),
            ("Acme Corp", "ORG"),
            ("Berlin", "GPE"),
            ("yesterday", "DATE"),
            ("retry.ts", "PRODUCT"),
        ]
    )
    extractor = SpacyEntityExtractor({"model": "fake_md"}, nlp=nlp)
    assert extractor.identity == "spacy@fake_md" and extractor.provider_name == "spacy"
    rule = extract_entities(text)
    assert _keys(rule) == ["sarah connor", "retry.ts", "acme corp", "berlin"]
    merged = extractor.extract(text)
    assert merged == extractor.extract(text) and nlp.calls == 2
    assert _keys(merged) == ["sarah connor", "retry.ts", "billing team", "acme corp", "berlin"]
    assert [c for c in merged if c.key in set(_keys(rule))] == rule
    added = merged[2]
    assert added == EntityCandidate("PROPER", "billing team", "billing team", "billing team")
    assert merged[1].entity_type == "FILE" and merged[1].normalized == "src/billing/retry.ts"
    # Key dedup across casing: the rule already owns 'berlin'; a later lower-case GPE span repeats the key.
    # 跨大小写的 key 去重：规则已拥有 'berlin'；后面小写的 GPE 区间重复了该 key。
    twice = SpacyEntityExtractor(nlp=FakeNLP([("berlin", "GPE")]))
    assert _keys(twice.extract("Berlin is big. I like berlin.")) == ["berlin"]
    # Articles and possessives are trimmed from the NER surface; the span offsets move with the trim.
    # NER 表面去掉冠词与所有格；区间偏移随之移动。
    possessive = SpacyEntityExtractor(nlp=FakeNLP([("the acme corp's", "ORG")]))
    got = possessive.extract("we asked the acme corp's engineers")
    assert got == [EntityCandidate("PROPER", "acme corp", "acme corp", "acme corp")]
    kept = SpacyEntityExtractor({"strip_articles": False}, nlp=FakeNLP([("the acme corp", "ORG")]))
    assert _keys(kept.extract("we asked the acme corp")) == ["the acme corp"]
    # Label map override: PRODUCT → TOPIC; unknown EntityType values are rejected at config time.
    # 标签映射覆盖：PRODUCT → TOPIC；未知的 EntityType 值在配置时被拒绝。
    topical = SpacyEntityExtractor({"labels": {"PRODUCT": "TOPIC"}}, nlp=FakeNLP([("kubernetes", "PRODUCT")]))
    assert topical.extract("we deploy on kubernetes") == [
        EntityCandidate("TOPIC", "kubernetes", "kubernetes", "kubernetes")
    ]
    assert topical.extract("Sarah deploys") == extract_entities("Sarah deploys")
    with pytest.raises(pydantic.ValidationError):
        SpacyEntityExtractorConfig(labels={"PRODUCT": "THING"})
    with pytest.raises(ConfigError) as info:
        EntityExtractorFactory.create("spacy", {"labels": {"PRODUCT": "THING"}})
    assert info.value.error_code == "CONFIG_001"
    assert set(DEFAULT_LABELS.values()) == {"PROPER"} and "DATE" not in DEFAULT_LABELS
    # Blank text short-circuits before the pipeline runs.
    # 空白文本在流水线运行前直接返回。
    lazy = FakeNLP([])
    assert SpacyEntityExtractor(nlp=lazy).extract("   ") == [] and lazy.calls == 0
    # merge_candidates directly: a rule PROPER run joined across a newline is still located (regex fallback), so the
    # overlapping NER span is suppressed; a span with end <= start is ignored.
    # 直接测 merge_candidates：跨换行拼接的规则 PROPER 连串仍能定位（正则回退），因此重叠的 NER 区间被抑制；
    # end <= start 的区间被忽略。
    multiline = "Sarah\nConnor met Bob."
    base = extract_entities(multiline)
    assert [c.text for c in base] == ["Sarah Connor", "Bob"]
    spans = [
        (0, 12, EntityCandidate("PROPER", "Sarah\nConnor", "sarah\nconnor", "sarah\nconnor")),
        (17, 20, EntityCandidate("PROPER", "Bob", "bob", "bob")),
        (5, 5, EntityCandidate("PROPER", "ghost", "ghost", "ghost")),
    ]
    assert merge_candidates(multiline, base, spans) == base
    assert merge_candidates(multiline, base, []) == base
    assert merge_candidates("", [], [(0, 0, EntityCandidate("PROPER", "x", "x", "x"))]) == []


# ---------------------------------------------------------------------------
# 3. Dependency guards
# 3. 依赖守卫
# ---------------------------------------------------------------------------
def test_spacy_dependency_and_model_guards(monkeypatch):
    """Missing spacy → DependencyError(extra='spacy'); missing model → DependencyError with the download hint; `disable` forwarded.
    缺少 spacy → DependencyError(extra='spacy')；缺少模型 → 带下载提示的 DependencyError；`disable` 被转发。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over SpacyEntityExtractor, EntityExtractorFactory and HMS construction.
        对 SpacyEntityExtractor、EntityExtractorFactory 与 HMS 构造的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules['spacy'] (restored by monkeypatch).
        临时替换 sys.modules['spacy']（由 monkeypatch 恢复）。
    """
    monkeypatch.setitem(sys.modules, "spacy", None)
    for build in (
        lambda: SpacyEntityExtractor(),
        lambda: EntityExtractorFactory.create("spacy"),
        lambda: HMS({"store": {"path": ":memory:"}, "ingest": {"entity_extractor": "spacy"}}),
    ):
        with pytest.raises(DependencyError) as info:
            build()
        assert info.value.error_code == "DEPS_001" and info.value.details == {"package": "spacy", "extra": "spacy"}
        assert info.value.suggestion == "pip install 'hms-elite[spacy]'" and isinstance(
            info.value.__cause__, ImportError
        )
    # A present SDK without the model package: spaCy raises OSError [E050]; the hint names the download command.
    # SDK 存在但模型包缺失：spaCy 抛出 OSError [E050]；提示给出下载命令。
    fake = types.ModuleType("spacy")
    loads: list[tuple[str, list[str]]] = []

    def _load(name: str, disable: list[str] | None = None) -> Any:
        """Record the call; fail for the default model, succeed for 'fake_md'.
        记录调用；默认模型失败，'fake_md' 成功。
        """
        loads.append((name, list(disable or [])))
        if name != "fake_md":
            raise OSError(f"[E050] Can't find model '{name}'.")
        return FakeNLP([("sarah connor", "PERSON")])

    fake.load = _load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", fake)
    with pytest.raises(DependencyError) as info:
        EntityExtractorFactory.create("spacy")
    assert info.value.details == {"package": "spacy", "extra": "spacy", "model": "en_core_web_sm"}
    assert info.value.suggestion == "python -m spacy download en_core_web_sm" and isinstance(
        info.value.__cause__, OSError
    )
    loaded = EntityExtractorFactory.create("spacy", {"model": "fake_md", "disable": ["parser", "lemmatizer"]})
    assert loads == [("en_core_web_sm", []), ("fake_md", ["parser", "lemmatizer"])]
    assert loaded.identity == "spacy@fake_md"
    assert _keys(loaded.extract("sarah connor fixed retry.ts")) == ["sarah connor", "retry.ts"]


def test_spacy_real_model_merges_into_rule_result():
    """With spacy and en_core_web_sm installed, the merged result keeps every rule candidate and adds NER mentions.
    安装了 spacy 与 en_core_web_sm 时，合并结果保留每个规则候选并补充 NER 提及。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over a real SpacyEntityExtractor (skipped without the SDK or the model).
        对真实 SpacyEntityExtractor 的断言（缺少 SDK 或模型时跳过）。

    State effect / 状态影响：
        Loads the spaCy model from disk.
        从磁盘加载 spaCy 模型。
    """
    pytest.importorskip("spacy")
    try:
        extractor = EntityExtractorFactory.create("spacy")
    except DependencyError as exc:
        pytest.skip(f"spaCy model missing: {exc.suggestion}")
    text = "Sarah Connor patched src/billing/retry.ts for the billing team at Acme Corp in Berlin yesterday."
    rule = extract_entities(text)
    merged = extractor.extract(text)
    assert [c for c in merged if c.key in set(_keys(rule))] == rule
    assert merged[1] == rule[1] and merged[1].entity_type == "FILE"
    assert len(_keys(merged)) == len(set(_keys(merged)))
    assert all(c.entity_type in {"PROPER", "IDENTIFIER", "FILE", "QUOTED", "TOPIC"} for c in merged)
    assert "yesterday" not in _keys(merged)


# ---------------------------------------------------------------------------
# 4. Wiring: the configured extractor reaches every linking site
# 4. 接线：配置中选定的抽取器到达每个链接点
# ---------------------------------------------------------------------------
def test_configured_extractor_reaches_ingest_rule_llm_feedback_and_import(store, hash_embedder, rule_llm):
    """A registered custom provider tags every atom's entities through add(), add_events(), feedback() and import_package().
    注册的自定义提供者经 add()、add_events()、feedback() 与 import_package() 给每个原子的实体打上标签。

    Input / 输入：
        store: In-memory StateStore fixture (used by the standalone Ingestor check).
        store：内存 StateStore 夹具（供独立 Ingestor 检查使用）。

        hash_embedder: 64-dim HashEmbedder fixture.
        hash_embedder：64 维 HashEmbedder 夹具。

        rule_llm: A default RuleLLM (rule extractor bound).
        rule_llm：默认 RuleLLM（绑定规则抽取器）。

    Output / 输出：
        Assertions over entity links, observation entity_mentions and the shared extractor instance.
        对实体链接、观察 entity_mentions 与共享抽取器实例的断言。

    State effect / 状态影响：
        Registers and removes the 'tag' entity_extractor provider; in-memory HMS instances only.
        登记并移除 'tag' 实体抽取器提供者；仅内存 HMS 实例。
    """
    register_provider("entity_extractor", "tag", "tests.test_entities.TagExtractor", "tests.test_entities.TagExtractorConfig")
    try:
        assert supported("entity_extractor") == ["rule", "spacy", "tag"]
        config = HMSConfig.from_dict(
            {"store": {"path": ":memory:"}, "ingest": {"entity_extractor": "tag", "entity_extractor_config": {"tag": "Zorblax"}}}
        )
        assert config.ingest.entity_extractor == "tag"
        # A standalone Ingestor builds its extractor from the config; an injected one is kept as-is.
        # 独立构造的 Ingestor 按配置创建抽取器；注入的抽取器原样保留。
        own = Ingestor(store, MemoryIndex(dims=64), hash_embedder, rule_llm, config)
        assert isinstance(own.entity_extractor, TagExtractor) and own.entity_extractor.config.tag == "Zorblax"
        given = TagExtractor({"tag": "Quux"})
        assert Ingestor(store, MemoryIndex(dims=64), hash_embedder, rule_llm, config, given).entity_extractor is given
        assert isinstance(Ingestor(store, MemoryIndex(dims=64), hash_embedder, rule_llm, HMSConfig()).entity_extractor, RuleEntityExtractor)

        hms = HMS(config)
        try:
            ext = hms.entity_extractor
            assert isinstance(ext, TagExtractor)
            assert hms.llm.entity_extractor is ext and hms.ingestor.entity_extractor is ext
            assert hms.feedback_processor.entity_extractor is ext
            # add(): the observation's entity_mentions (rule LLM) and the atom's entity links (Ingestor) carry the tag.
            # add()：观察的 entity_mentions（规则 LLM）与原子的实体链接（Ingestor）都带上标签。
            row = hms.add("Sarah joined the billing team in March.", user_id="alice")["results"][0]
            assert row["event"] == "ADD"
            assert _entity_keys_of(hms, row["id"]) == {"sarah", "zorblax"}
            head = hms.store.resolve_head(row["id"])
            obs = hms.store.get_observation(head.source_id)
            assert obs is not None and obs.entity_mentions == ["Sarah", "Zorblax"]
            # add_events() structured path: the closure builder's EP entity_mentions use the same extractor.
            # add_events() 结构化路径：闭合构建器的 EP entity_mentions 使用同一抽取器。
            out = hms.add_events(
                [
                    {
                        "object_type": "file_diff",
                        "timestamp": "2026-06-15T09:30:00Z",
                        "payload": {"path": "src/billing/retry.ts", "patch": "+backoff", "intent": "fix retry"},
                    },
                    {
                        "object_type": "test_log",
                        "timestamp": "2026-06-15T09:40:00Z",
                        "payload": {"suite": "billing_e2e", "passed": 3, "failed": 0},
                    },
                ],
                user_id="alice",
                agent_id="coder",
                run_id="run-1",
            )
            assert len(out["eps"]) == 1
            ep = hms.store.get_ep(out["eps"][0])
            assert ep is not None and "Zorblax" in ep.entity_mentions and "src/billing/retry.ts" in ep.entity_mentions
            assert "zorblax" in {r["normalized_text"] for r in hms.store.entities_for_atom(ep.atom_id)}
            # feedback(): the correction successor is re-linked through the same extractor.
            # feedback()：纠正后继经同一抽取器重新链接。
            fixed = hms.feedback(row["id"], "state_wrong", correction="Sarah joined the payments team in March.")
            successor = next(a for a in fixed["actions"] if a["action"] == "create")
            assert {r["normalized_text"] for r in hms.store.entities_for_atom(successor["atom_id"])} == {"sarah", "zorblax"}
            # import_package(): the receiver re-links imported atoms with ITS extractor (a different tag here).
            # import_package()：接收方用自己的抽取器重新链接导入的原子（这里是不同的标签）。
            pkg = hms.handoff(filters={"user_id": "alice"}, goal="review")
            receiver = HMS(
                HMSConfig.from_dict(
                    {"store": {"path": ":memory:"}, "ingest": {"entity_extractor": "tag", "entity_extractor_config": {"tag": "Quux"}}}
                )
            )
            try:
                report = receiver.import_package(pkg)
                assert report["imported"]["atoms"] >= 1
                imported = receiver.store.get_atom(report["id_map"][successor["atom_id"]])
                assert imported is not None
                linked = {r["normalized_text"] for r in receiver.store.entities_for_atom(imported.atom_id)}
                assert "quux" in linked and "sarah" in linked
            finally:
                receiver.close()
        finally:
            hms.close()
        # The default facade never sees the tag: the same add() links only the rule entity.
        # 默认门面看不到标签：同样的 add() 只链接规则实体。
        plain = HMS({"store": {"path": ":memory:"}})
        try:
            assert isinstance(plain.entity_extractor, RuleEntityExtractor)
            row = plain.add("Sarah joined the billing team in March.", user_id="alice")["results"][0]
            assert _entity_keys_of(plain, row["id"]) == {"sarah"}
        finally:
            plain.close()
    finally:
        REGISTRY["entity_extractor"].pop("tag", None)
    assert supported("entity_extractor") == ["rule", "spacy"]
