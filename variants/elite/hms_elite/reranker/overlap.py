"""Offline deterministic reranker: weighted content-token coverage of the query by each row's text.
离线确定性重排器：每行文本对查询内容词元的加权覆盖率。
"""

# Purpose (EN): The default reranker; zero dependencies, zero network, reproducible scores from utils.text tokens.
# 目的（中）：默认重排器；零依赖、零网络，基于 utils.text 词元的可复现分数。

# Input (EN): A query string and result rows with a text field.
# 输入（中）：查询字符串与带文本字段的结果行。

# Output (EN): Row copies ordered by 'rerank_score' in [0, 1] (identifier tokens weighted by identifier_weight).
# 输出（中）：按 [0, 1] 内的 'rerank_score' 排序的行副本（标识符词元按 identifier_weight 加权）。

# State effect (EN): None (pure).
# 状态影响（中）：无（纯函数）。

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from hms_elite.reranker.base import BaseRerankerConfig, RerankerBase, rank_rows, resolve_top_k, row_text
from hms_elite.utils.text import content_tokens, is_identifier, ordered_union


class OverlapRerankerConfig(BaseRerankerConfig):
    """Overlap reranker config: the weight of identifier tokens relative to plain words.
    覆盖率重排器配置：标识符词元相对普通词的权重。
    """

    identifier_weight: float = Field(default=2.0, ge=0.0)


def _ident_match(query_token: str, row_tokens: set[str]) -> bool:
    """True when an identifier token of the query appears in the row, also as a path suffix either way.
    查询的标识符词元出现在行中（含双向路径后缀匹配）时为真。
    """
    if query_token in row_tokens:
        return True
    # tokenize() keeps 'src/billing/retry.ts' whole while a query may say 'retry.ts' (or the reverse).
    # tokenize() 会把 'src/billing/retry.ts' 整体保留，而查询可能只写 'retry.ts'（或相反）。
    return any(
        tok.endswith("/" + query_token) or query_token.endswith("/" + tok)
        for tok in row_tokens
        if is_identifier(tok)
    )


class OverlapReranker(RerankerBase):
    """Weighted coverage reranker: share of the query's content tokens (identifiers ×weight) found in the row.
    加权覆盖率重排器：查询内容词元（标识符 ×权重）在行中出现的比例。
    """

    config_class: ClassVar[type[BaseRerankerConfig]] = OverlapRerankerConfig
    provider_name: ClassVar[str] = "overlap"

    def score(self, query: str, text: str) -> float:
        """Weighted coverage of the query's distinct content tokens by the text, rounded to 6 decimals.
        文本对查询去重内容词元的加权覆盖率，保留 6 位小数。

        Input / 输入：
            query: The search query.
            query：搜索查询。

            text: One candidate text.
            text：一条候选文本。

        Output / 输出：
            Float in [0, 1]; 0.0 when the query has no content tokens.
            [0, 1] 内的浮点数；查询无内容词元时为 0.0。

        State effect / 状态影响：
            None.
            无。
        """
        query_tokens = ordered_union(content_tokens(query))
        if not query_tokens:
            return 0.0
        row_tokens = set(content_tokens(text))
        weight = float(self.config.identifier_weight)  # type: ignore[attr-defined]
        total = 0.0
        hit = 0.0
        for tok in query_tokens:
            ident = is_identifier(tok)
            w = weight if ident else 1.0
            total += w
            if (ident and _ident_match(tok, row_tokens)) or (not ident and tok in row_tokens):
                hit += w
        return round(hit / total, 6) if total > 0 else 0.0

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
            None.
            无。
        """
        limit = resolve_top_k(top_k, self.config.top_k)
        if not rows:
            return []
        key = self.config.text_key
        scores = [self.score(query, row_text(row, key)) for row in rows]
        return rank_rows(rows, scores, top_k=limit)


__all__ = ["OverlapReranker", "OverlapRerankerConfig"]
