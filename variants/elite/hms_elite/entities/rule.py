"""Rule entity extractor: the default provider that delegates byte-for-byte to utils.text.extract_entities.
规则实体抽取器：默认提供者，逐字节委托给 utils.text.extract_entities。
"""

# Purpose (EN): Wrap the regex extractor in the provider contract so ingest and the rule LLM take any extractor.
# 目的（中）：把正则抽取器包进提供者契约，使摄取与规则 LLM 可以接受任意抽取器。

# Input (EN): RuleEntityExtractorConfig (zh_rules, names; both off by default) and free text.
# 输入（中）：RuleEntityExtractorConfig（zh_rules、names；默认都关闭）与自由文本。

# Output (EN): Exactly extract_entities(text) under the defaults; extract_entities(text, zh_rules=, names=) otherwise.
# 输出（中）：默认下与 extract_entities(text) 完全一致；否则为 extract_entities(text, zh_rules=, names=)。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

from typing import ClassVar

from hms_elite.entities.base import BaseEntityExtractorConfig, EntityExtractorBase
from hms_elite.utils.text import EntityCandidate, extract_entities


class RuleEntityExtractorConfig(BaseEntityExtractorConfig):
    """Rule extractor config: the Chinese proper-name rules switch and an optional closed name list (the lexicons live in utils.text).
    规则抽取器配置：中文人名规则开关与可选的封闭人名表（词表在 utils.text 中）。
    """

    # zh_rules: add utils.text's Chinese proper-name rules (2–4 chars before a statement verb or after 和 / 跟 / 给);
    # HMS.__init__ sets it from IngestConfig.zh_rules. names: proper names matched exactly wherever they occur
    # (longest first, above every CJK rule) — the EgoLife-ZH glossary names, for instance. Both default off so the
    # identity stays 'rule@entities-v1' and extract(text) == extract_entities(text) byte for byte.
    # zh_rules：加入 utils.text 的中文人名规则（陈述动词之前或 和 / 跟 / 给 之后的 2–4 字）；HMS.__init__ 按
    # IngestConfig.zh_rules 设置。names：在任何位置精确匹配的专有名（长者优先，高于全部 CJK 规则）——例如 EgoLife-ZH
    # 的术语表人名。两者默认关闭，因此身份仍是 'rule@entities-v1'，且 extract(text) 与 extract_entities(text) 逐字节一致。
    zh_rules: bool = False
    names: list[str] = []


class RuleEntityExtractor(EntityExtractorBase):
    """Deterministic regex entity extractor (FILE, IDENTIFIER, QUOTED, PROPER, CJK) — the offline default.
    确定性正则实体抽取器（FILE、IDENTIFIER、QUOTED、PROPER、CJK）——离线默认。
    """

    config_class: ClassVar[type[BaseEntityExtractorConfig]] = RuleEntityExtractorConfig
    provider_name: ClassVar[str] = "rule"

    @property
    def identity(self) -> str:
        """'rule@entities-v1' under the defaults; 'rule@entities-zh-v1' when zh_rules or names are set.
        默认下为 'rule@entities-v1'；设置了 zh_rules 或 names 时为 'rule@entities-zh-v1'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Identity string.
            身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        cfg = self.config
        return "rule@entities-zh-v1" if (cfg.zh_rules or cfg.names) else "rule@entities-v1"  # type: ignore[attr-defined]

    def extract(self, text: str) -> list[EntityCandidate]:
        """Delegate to utils.text.extract_entities (byte-identical under the default config).
        委托给 utils.text.extract_entities（默认配置下逐字节一致）。

        Input / 输入：
            text: Free text.
            text：自由文本。

        Output / 输出：
            extract_entities(text) — with zh_rules= / names= from the config when set.
            extract_entities(text)——配置设置时带 zh_rules= / names=。

        State effect / 状态影响：
            None.
            无。
        """
        cfg = self.config
        if cfg.zh_rules or cfg.names:  # type: ignore[attr-defined]
            return extract_entities(text, zh_rules=bool(cfg.zh_rules), names=tuple(cfg.names))  # type: ignore[attr-defined]
        return extract_entities(text)


__all__ = ["RuleEntityExtractor", "RuleEntityExtractorConfig"]
