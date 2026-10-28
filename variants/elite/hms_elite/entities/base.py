"""Entity extractor provider contract: BaseEntityExtractorConfig, EntityExtractorBase and the span-merge helper.
实体抽取提供者契约：BaseEntityExtractorConfig、EntityExtractorBase 与区间合并辅助函数。
"""

# Purpose (EN): Define the one interface every entity extractor (rule, spacy) implements over utils.text.EntityCandidate.
# 目的（中）：定义所有实体抽取器（rule、spacy）在 utils.text.EntityCandidate 之上实现的统一接口。

# Input (EN): Free text (a sentence or an atom text); for merge_candidates also rule candidates and NER spans.
# 输入（中）：自由文本（句子或原子文本）；merge_candidates 另接收规则候选与 NER 区间。

# Output (EN): EntityCandidate lists in text order, de-duplicated by key (first kept).
# 输出（中）：按文本顺序、按 key 去重（保留首个）的 EntityCandidate 列表。

# State effect (EN): None; extractors are pure functions of their input text.
# 状态影响（中）：无；抽取器是输入文本的纯函数。

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

from pydantic import model_validator

from hms_elite.models import HMSModel
from hms_elite.utils.factory import check_extra_fields, coerce_config
from hms_elite.utils.text import EntityCandidate

# A span is (start, end, candidate) in character offsets of the extracted text.
# 区间为 (start, end, candidate)，以被抽取文本的字符偏移计。
Span = tuple[int, int, EntityCandidate]


class BaseEntityExtractorConfig(HMSModel):
    """Fields shared by every entity extractor config (none today; the unknown-key guard is the shared part).
    所有实体抽取器配置共享的字段（目前没有；共享的是未知键守卫）。
    """

    @model_validator(mode="before")
    @classmethod
    def _extra_fields(cls, data: Any) -> Any:
        """Unknown keys → ConfigError(CONFIG_001) with a did-you-mean hint.
        未知键 → 带 did-you-mean 提示的 ConfigError(CONFIG_001)。
        """
        return check_extra_fields(cls, data)


class EntityExtractorBase(ABC):
    """Abstract entity extractor: text → EntityCandidate list with the utils.text ordering and dedup semantics.
    抽象实体抽取器：文本 → 遵循 utils.text 顺序与去重语义的 EntityCandidate 列表。
    """

    config_class: ClassVar[type[BaseEntityExtractorConfig]] = BaseEntityExtractorConfig
    provider_name: ClassVar[str] = "base"

    def __init__(self, config: BaseEntityExtractorConfig | dict | None = None) -> None:
        """Coerce the config into the provider's config class (dict → class, None → defaults).
        将配置规整为提供者的配置类（字典 → 类，None → 默认值）。
        """
        self.config: BaseEntityExtractorConfig = coerce_config(self.config_class, config)  # type: ignore[assignment]

    @property
    def identity(self) -> str:
        """Provider identity string (default: the provider name).
        提供者身份字符串（默认为提供者名）。

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
        return self.provider_name

    @abstractmethod
    def extract(self, text: str) -> list[EntityCandidate]:
        """Entity candidates of a text in text order, de-duplicated by key (first kept).
        文本中的实体候选，按文本顺序、按 key 去重（保留首个）。

        Input / 输入：
            text: Free text (a sentence or an atom text).
            text：自由文本（句子或原子文本）。

        Output / 输出：
            EntityCandidate list; the same input always yields the same list.
            EntityCandidate 列表；相同输入总是得到相同列表。

        State effect / 状态影响：
            None.
            无。
        """


def _locate(text: str, candidates: Sequence[EntityCandidate]) -> list[tuple[int, int] | None]:
    """Character spans of rule candidates found left to right in the text (None when the text cannot be located).
    从左到右在文本中定位规则候选的字符区间（定位不到时为 None）。
    """
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for candidate in candidates:
        needle = candidate.text
        if not needle:
            spans.append(None)
            continue
        start = text.find(needle, cursor)
        if start < 0:
            # PROPER runs are joined with single spaces; the source may have had other whitespace between tokens.
            # PROPER 连串以单个空格拼接；原文各词之间可能是其他空白。
            pattern = re.compile(r"\s+".join(re.escape(part) for part in needle.split()))
            match = pattern.search(text, cursor) or pattern.search(text)
            if match is None:
                spans.append(None)
                continue
            start, end = match.start(), match.end()
        else:
            end = start + len(needle)
        spans.append((start, end))
        cursor = end
    return spans


def _overlaps(start: int, end: int, claimed: Sequence[tuple[int, int]]) -> bool:
    """True when [start, end) intersects any claimed span.
    当 [start, end) 与任一已占用区间相交时为真。
    """
    return any(start < c_end and c_start < end for c_start, c_end in claimed)


def merge_candidates(text: str, base: Sequence[EntityCandidate], spans: Sequence[Span]) -> list[EntityCandidate]:
    """Merge NER spans into rule candidates: rule spans win overlaps, keys dedup (first kept), text order preserved.
    把 NER 区间并入规则候选：规则区间在重叠时胜出，按 key 去重（保留首个），保持文本顺序。

    Input / 输入：
        text: The extracted text (used to locate the rule candidates).
        text：被抽取的文本（用于定位规则候选）。

        base: Rule candidates (utils.text.extract_entities order and dedup already applied).
        base：规则候选（已按 utils.text.extract_entities 的顺序与去重）。

        spans: (start, end, candidate) triples from the NER model, any order.
        spans：来自 NER 模型的 (start, end, candidate) 三元组，顺序任意。

    Output / 输出：
        Candidates ordered by start offset (rule first on ties); NER spans overlapping a rule span or repeating a key are dropped.
        按起始偏移排序的候选（并列时规则在前）；与规则区间重叠或重复 key 的 NER 区间被丢弃。

    State effect / 状态影响：
        None.
        无。
    """
    located = _locate(text, base)
    claimed = [span for span in located if span is not None]
    # (start, priority, order, candidate): rule candidates keep their order; unlocated ones inherit the previous end.
    # (start, 优先级, 序号, 候选)：规则候选保持顺序；未定位的沿用前一个候选的结束位置。
    entries: list[tuple[int, int, int, EntityCandidate]] = []
    previous_end = 0
    for order, (candidate, span) in enumerate(zip(base, located)):
        start = span[0] if span is not None else previous_end
        entries.append((start, 0, order, candidate))
        if span is not None:
            previous_end = span[1]
    for order, (start, end, candidate) in enumerate(sorted(spans, key=lambda s: (s[0], s[1]))):
        if end <= start or _overlaps(start, end, claimed):
            continue
        entries.append((start, 1, order, candidate))
    out: list[EntityCandidate] = []
    seen: set[str] = set()
    for _start, _priority, _order, candidate in sorted(entries, key=lambda e: (e[0], e[1], e[2])):
        if not candidate.key or candidate.key in seen:
            continue
        seen.add(candidate.key)
        out.append(candidate)
    return out


__all__ = ["BaseEntityExtractorConfig", "EntityExtractorBase", "Span", "merge_candidates"]
