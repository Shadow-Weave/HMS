"""Reranker provider contract: BaseRerankerConfig, RerankerBase and the shared stable-ordering helpers.
重排器提供者契约：BaseRerankerConfig、RerankerBase 与共享的稳定排序辅助函数。
"""

# Purpose (EN): Define the one interface every reranker (overlap, llm) implements, shaped like Mem0's BaseReranker.
# 目的（中）：定义所有重排器（overlap、llm）实现的统一接口，形状与 Mem0 的 BaseReranker 一致。

# Input (EN): A query string and Mem0-shaped result rows (each carrying a text field, 'memory' by default).
# 输入（中）：查询字符串与 Mem0 形状的结果行（每行含文本字段，默认为 'memory'）。

# Output (EN): Copies of the rows with 'rerank_score' appended, sorted by that score (ties keep input order).
# 输出（中）：追加 'rerank_score' 的行副本，按该分数降序（并列保持输入顺序）。

# State effect (EN): None; helpers are pure and providers never mutate the caller's rows.
# 状态影响（中）：无；辅助函数为纯函数，提供者从不修改调用方的行。

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

from pydantic import ConfigDict, Field, model_validator

from hms_elite.exceptions import ValidationError
from hms_elite.models import HMSModel
from hms_elite.utils.factory import check_extra_fields, coerce_config

# Fallback keys tried after config.text_key (Mem0's LLMReranker reads memory → text → content).
# config.text_key 之后依次尝试的回退键（Mem0 的 LLMReranker 读取 memory → text → content）。
TEXT_FALLBACK_KEYS: tuple[str, ...] = ("memory", "text", "content")
# Score attached to every row that the provider could not score (it is ranked after every scored row).
# 提供者未能打分的行所附的分数（这些行排在所有已打分行之后）。
MISSING_SCORE = 0.0


class BaseRerankerConfig(HMSModel):
    """Fields shared by every reranker config (Mem0 BaseRerankerConfig parity: top_k; plus the row text key).
    所有重排器配置共享的字段（与 Mem0 BaseRerankerConfig 对齐的 top_k；外加行文本键）。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True, validate_default=True)

    top_k: int | None = Field(default=None, ge=1)
    text_key: str = Field(default="memory", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _extra_fields(cls, data: Any) -> Any:
        """Unknown keys → ConfigError(CONFIG_001) with a did-you-mean hint.
        未知键 → 带 did-you-mean 提示的 ConfigError(CONFIG_001)。
        """
        return check_extra_fields(cls, data)


def clamp_score(value: Any) -> float | None:
    """Coerce a model-provided score to a finite float clamped into [0, 1]; None when unusable.
    将模型给出的分数规整为落在 [0, 1] 内的有限浮点数；不可用时返回 None。

    Input / 输入：
        value: Any JSON scalar (number, numeric string, bool, None, ...).
        value：任意 JSON 标量（数字、数字字符串、布尔、None……）。

    Output / 输出：
        The clamped float, or None for booleans, non-numeric values, NaN and infinities.
        截断后的浮点数；布尔、非数字、NaN 与无穷大返回 None。

    State effect / 状态影响：
        None.
        无。
    """
    # bool is an int subclass; a JSON true/false is not a relevance score.
    # bool 是 int 的子类；JSON 的 true/false 不是相关性分数。
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return min(max(score, 0.0), 1.0)


def row_text(row: dict, text_key: str = "memory") -> str:
    """Read the text of a result row: `text_key` first, then memory → text → content, else ''.
    读取结果行的文本：先取 `text_key`，再依次 memory → text → content，否则为 ''。

    Input / 输入：
        row: A Mem0-shaped result row.
        row：Mem0 形状的结果行。

        text_key: The preferred key (config.text_key).
        text_key：优先读取的键（config.text_key）。

    Output / 输出：
        The first string value found (non-string values are stringified), or ''.
        找到的第一个字符串值（非字符串值转为字符串），否则为 ''。

    State effect / 状态影响：
        None.
        无。
    """
    for key in (text_key, *TEXT_FALLBACK_KEYS):
        value = row.get(key)
        if value is None or value == "":
            continue
        return value if isinstance(value, str) else str(value)
    return ""


def resolve_top_k(top_k: int | None, config_top_k: int | None) -> int | None:
    """Pick the effective cap: the call argument wins over the config value; None means no cap.
    确定生效的上限：调用参数优先于配置值；None 表示不截断。

    Input / 输入：
        top_k: The rerank() call argument.
        top_k：rerank() 的调用参数。

        config_top_k: The provider's config.top_k.
        config_top_k：提供者的 config.top_k。

    Output / 输出：
        A positive int or None; a call argument below 1 raises ValidationError(VALIDATION_009).
        正整数或 None；调用参数小于 1 时抛出 ValidationError(VALIDATION_009)。

    State effect / 状态影响：
        None.
        无。
    """
    if top_k is None:
        return config_top_k
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValidationError(
            f"top_k must be a positive integer, got {top_k!r}",
            "VALIDATION_009",
            details={"top_k": top_k},
            suggestion="pass top_k >= 1 or None",
        )
    return top_k


def rank_rows(rows: Sequence[dict], scores: Sequence[float | None], *, top_k: int | None = None) -> list[dict]:
    """Order row copies by score (descending, stable); unscored rows follow in input order; then truncate.
    按分数（降序、稳定）排列行副本；未打分的行按输入顺序排在其后；最后截断。

    Input / 输入：
        rows: The result rows (never mutated).
        rows：结果行（从不被修改）。

        scores: One float per row, or None for a row the provider could not score.
        scores：每行一个浮点分数，未能打分的行为 None。

        top_k: Optional cap applied after ordering.
        top_k：排序后应用的可选上限。

    Output / 输出：
        New dicts (`dict(row)` + 'rerank_score'); unscored rows carry MISSING_SCORE; len(scores) must equal len(rows).
        新字典（`dict(row)` + 'rerank_score'）；未打分行携带 MISSING_SCORE；len(scores) 必须等于 len(rows)。

    State effect / 状态影响：
        None.
        无。
    """
    if len(scores) != len(rows):
        raise ValueError(f"rank_rows: {len(scores)} scores for {len(rows)} rows")
    scored = [(i, float(s)) for i, s in enumerate(scores) if s is not None]
    # sorted() is stable, so equal scores keep their retrieval order.
    # sorted() 是稳定排序，因此分数相同的行保持检索顺序。
    scored.sort(key=lambda pair: -pair[1])
    ordered: list[tuple[int, float]] = scored + [(i, MISSING_SCORE) for i, s in enumerate(scores) if s is None]
    if top_k is not None:
        ordered = ordered[:top_k]
    out: list[dict] = []
    for index, score in ordered:
        row = dict(rows[index])
        row["rerank_score"] = score
        out.append(row)
    return out


class RerankerBase(ABC):
    """Abstract reranker: one `rerank(query, rows, *, top_k)` primitive with Mem0's BaseReranker shape.
    抽象重排器：一个与 Mem0 BaseReranker 同形的 `rerank(query, rows, *, top_k)` 原语。
    """

    config_class: ClassVar[type[BaseRerankerConfig]] = BaseRerankerConfig
    provider_name: ClassVar[str] = "base"

    def __init__(self, config: BaseRerankerConfig | dict | None = None) -> None:
        """Coerce the config into the provider's config class (dict → class, None → defaults).
        将配置规整为提供者的配置类（字典 → 类，None → 默认值）。
        """
        self.config: BaseRerankerConfig = coerce_config(self.config_class, config)  # type: ignore[assignment]

    @property
    def identity(self) -> str:
        """Provider identity string such as 'overlap@v1' or 'llm:rule@v1'.
        提供者身份字符串，如 'overlap@v1' 或 'llm:rule@v1'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `f"{provider_name}@v1"` by default; subclasses override.
            默认 `f"{provider_name}@v1"`；子类可覆盖。

        State effect / 状态影响：
            None.
            无。
        """
        return f"{self.provider_name}@v1"

    @abstractmethod
    def rerank(self, query: str, rows: list[dict], *, top_k: int | None = None) -> list[dict]:
        """Score rows against the query and return them re-ordered.
        按查询为各行打分并返回重新排序的行。

        Input / 输入：
            query: The search query text.
            query：搜索查询文本。

            rows: Mem0-shaped result rows in retrieval order (never mutated).
            rows：按检索顺序排列的 Mem0 形状结果行（从不被修改）。

            top_k: Cap on the returned rows; None → config.top_k; both None → all rows.
            top_k：返回行数上限；None → config.top_k；两者皆 None → 全部行。

        Output / 输出：
            Row copies with 'rerank_score' appended, sorted by it descending (ties keep retrieval order).
            追加 'rerank_score' 的行副本，按其降序排列（并列保持检索顺序）。

        State effect / 状态影响：
            None for offline providers; network for LLM-backed providers.
            离线提供者无；LLM 提供者有网络调用。
        """


__all__ = [
    "MISSING_SCORE",
    "TEXT_FALLBACK_KEYS",
    "BaseRerankerConfig",
    "RerankerBase",
    "clamp_score",
    "rank_rows",
    "resolve_top_k",
    "row_text",
]
