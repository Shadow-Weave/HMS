"""spaCy entity extractor (optional `spacy` extra): NER spans merged INTO the rule candidates, never replacing them.
spaCy 实体抽取器（可选 `spacy` extra）：NER 区间并入规则候选，而不是替换它们。
"""

# Purpose (EN): Add learned PERSON/ORG/GPE/... mentions to the regex FILE/IDENTIFIER/QUOTED/PROPER candidates.
# 目的（中）：在正则的 FILE/IDENTIFIER/QUOTED/PROPER 候选之上补充学习模型识别的 PERSON/ORG/GPE/… 提及。

# Input (EN): SpacyEntityExtractorConfig (model name, label → EntityType map, disabled pipes) and free text.
# 输入（中）：SpacyEntityExtractorConfig（模型名、标签 → EntityType 映射、停用的组件）与自由文本。

# Output (EN): merge_candidates(text, extract_entities(text), NER spans); DependencyError when spacy or the model is missing.
# 输出（中）：merge_candidates(text, extract_entities(text), NER 区间)；缺少 spacy 或模型时抛出 DependencyError。

# State effect (EN): None beyond loading the spaCy pipeline once per extractor instance (the import is lazy).
# 状态影响（中）：除每个抽取器实例加载一次 spaCy 流水线外无副作用（导入是惰性的）。

from __future__ import annotations

import re
from typing import Any, ClassVar, Sequence

from pydantic import Field, field_validator

from hms_elite.entities.base import BaseEntityExtractorConfig, EntityExtractorBase, Span, merge_candidates
from hms_elite.enums import EntityType
from hms_elite.exceptions import DependencyError
from hms_elite.utils.text import EntityCandidate, entity_key, extract_entities, normalize_entity

DEFAULT_MODEL = "en_core_web_sm"
# spaCy (OntoNotes) labels that name a thing; numeric / temporal labels are left to the date and keyword axes.
# spaCy（OntoNotes）中指称事物的标签；数值 / 时间类标签交给日期轴与关键词轴处理。
DEFAULT_LABELS: dict[str, str] = {
    "PERSON": EntityType.PROPER.value,
    "ORG": EntityType.PROPER.value,
    "GPE": EntityType.PROPER.value,
    "LOC": EntityType.PROPER.value,
    "FAC": EntityType.PROPER.value,
    "NORP": EntityType.PROPER.value,
    "PRODUCT": EntityType.PROPER.value,
    "EVENT": EntityType.PROPER.value,
    "WORK_OF_ART": EntityType.PROPER.value,
    "LAW": EntityType.PROPER.value,
    "LANGUAGE": EntityType.PROPER.value,
}
_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"(?:'s|’s)$")


class SpacyEntityExtractorConfig(BaseEntityExtractorConfig):
    """spaCy extractor config: model name, label map, disabled pipeline components, article stripping.
    spaCy 抽取器配置：模型名、标签映射、停用的流水线组件、是否去掉冠词。
    """

    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    labels: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_LABELS))
    disable: list[str] = Field(default_factory=list)
    strip_articles: bool = True

    @field_validator("labels")
    @classmethod
    def _known_entity_types(cls, value: dict[str, str]) -> dict[str, str]:
        """Every mapped value must be an EntityType member.
        每个映射值都必须是 EntityType 成员。
        """
        allowed = {member.value for member in EntityType}
        for label, entity_type in value.items():
            if entity_type not in allowed:
                raise ValueError(f"labels[{label!r}] = {entity_type!r} is not an EntityType ({sorted(allowed)})")
        return value


def load_pipeline(model: str, *, disable: Sequence[str] = ()) -> Any:
    """Import spacy lazily and load a pipeline; both failure modes become DependencyError(extra='spacy').
    惰性导入 spacy 并加载流水线；两种失败都转为 DependencyError(extra='spacy')。

    Input / 输入：
        model: spaCy package name (e.g. 'en_core_web_sm').
        model：spaCy 模型包名（如 'en_core_web_sm'）。

        disable: Pipeline components to disable at load time.
        disable：加载时停用的流水线组件。

    Output / 输出：
        The loaded `spacy.Language`.
        已加载的 `spacy.Language`。

    State effect / 状态影响：
        Imports spacy and reads the model package from disk.
        导入 spacy 并从磁盘读取模型包。
    """
    try:
        import spacy
    except ImportError as exc:
        raise DependencyError(
            "The 'spacy' package is required for the spacy entity extractor",
            "DEPS_001",
            details={"package": "spacy", "extra": "spacy"},
        ) from exc
    try:
        return spacy.load(model, disable=list(disable))
    except OSError as exc:
        # spaCy raises OSError [E050] when the model package is not installed.
        # 模型包未安装时 spaCy 抛出 OSError [E050]。
        raise DependencyError(
            f"spaCy model {model!r} is not installed: {exc}",
            "DEPS_001",
            details={"package": "spacy", "extra": "spacy", "model": model},
            suggestion=f"python -m spacy download {model}",
        ) from exc


class SpacyEntityExtractor(EntityExtractorBase):
    """Rule candidates plus spaCy NER spans (rule wins overlaps; DATE/CARDINAL/... labels are ignored by default).
    规则候选加 spaCy NER 区间（重叠时规则胜出；默认忽略 DATE/CARDINAL/… 标签）。
    """

    config_class: ClassVar[type[BaseEntityExtractorConfig]] = SpacyEntityExtractorConfig
    provider_name: ClassVar[str] = "spacy"

    def __init__(self, config: SpacyEntityExtractorConfig | dict | None = None, *, nlp: Any = None) -> None:
        """Load the configured model unless a ready pipeline (`nlp`, any callable returning a doc with `.ents`) is injected.
        除非注入现成流水线（`nlp`，任何返回带 `.ents` 文档的可调用对象），否则加载配置的模型。
        """
        super().__init__(config)
        self.nlp = nlp if nlp is not None else load_pipeline(self.config.model, disable=self.config.disable)

    @property
    def identity(self) -> str:
        """'spacy@<model>'.
        'spacy@<模型名>'。

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
        return f"spacy@{self.config.model}"

    def extract(self, text: str) -> list[EntityCandidate]:
        """Rule candidates merged with the mapped NER spans of `self.nlp(text)`.
        规则候选与 `self.nlp(text)` 的映射后 NER 区间合并。

        Input / 输入：
            text: Free text.
            text：自由文本。

        Output / 输出：
            merge_candidates(text, extract_entities(text), spans) — the rule result is always a subsequence.
            merge_candidates(text, extract_entities(text), spans)——规则结果总是其子序列。

        State effect / 状态影响：
            None.
            无。
        """
        base = extract_entities(text)
        if not text.strip():
            return base
        spans: list[Span] = []
        for ent in getattr(self.nlp(text), "ents", ()):
            span = self._span(text, ent)
            if span is not None:
                spans.append(span)
        return merge_candidates(text, base, spans)

    def _span(self, text: str, ent: Any) -> Span | None:
        """Map one spaCy entity to a (start, end, candidate) span; None for unmapped labels or empty surfaces.
        把一个 spaCy 实体映射为 (start, end, candidate) 区间；标签未映射或表面为空时为 None。
        """
        entity_type = self.config.labels.get(str(getattr(ent, "label_", "")))
        if entity_type is None:
            return None
        start, end = int(ent.start_char), int(ent.end_char)
        surface = text[start:end]
        if self.config.strip_articles:
            article = _ARTICLE_RE.match(surface)
            if article is not None:
                start += article.end()
                surface = surface[article.end() :]
        possessive = _POSSESSIVE_RE.search(surface)
        if possessive is not None:
            surface = surface[: possessive.start()]
        start += len(surface) - len(surface.lstrip())
        surface = surface.strip()
        if not surface:
            return None
        end = start + len(surface)
        return start, end, EntityCandidate(entity_type, surface, normalize_entity(surface), entity_key(surface))


__all__ = ["DEFAULT_LABELS", "DEFAULT_MODEL", "SpacyEntityExtractor", "SpacyEntityExtractorConfig", "load_pipeline"]
