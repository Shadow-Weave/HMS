"""LLM-backed reranker: one `rerank` JSON task per call over synthetic candidate ids, with an offline fallback.
LLM 重排器：每次调用执行一个基于合成候选 id 的 `rerank` JSON 任务，并带离线回退。
"""

# Purpose (EN): Rerank rows with any LLMBase provider through generate_json; fall back to OverlapReranker when the
#               provider cannot serve the task (LLM_002, e.g. the rule LLM).
# 目的（中）：通过 generate_json 用任意 LLMBase 提供者重排各行；提供者无法执行该任务（LLM_002，如规则 LLM）时
#            回退到 OverlapReranker。

# Input (EN): A query, result rows, and an LLMBase (injected, built from config.llm, or the rule default).
# 输入（中）：查询、结果行，以及一个 LLMBase（注入、由 config.llm 构造，或默认规则提供者）。

# Output (EN): Row copies ordered by 'rerank_score'; unknown ids dropped, omitted rows ranked last in retrieval order.
# 输出（中）：按 'rerank_score' 排序的行副本；未知 id 丢弃，未提及的行按检索顺序排在末尾。

# State effect (EN): Records `last_route` ('llm' | 'overlap' | 'passthrough'); network only inside the LLM provider.
# 状态影响（中）：记录 `last_route`（'llm' | 'overlap' | 'passthrough'）；网络调用仅发生在 LLM 提供者内部。

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from hms_elite.exceptions import LLMError
from hms_elite.llms.base import LLMBase
from hms_elite.reranker.base import BaseRerankerConfig, RerankerBase, clamp_score, rank_rows, resolve_top_k, row_text
from hms_elite.reranker.overlap import OverlapReranker
from hms_elite.reranker.prompts import RERANK_PROMPT, RERANK_ROOT_KEY, RERANK_TASK, build_rerank_prompt
from hms_elite.utils.factory import LlmFactory

logger = logging.getLogger(__name__)


class LLMRerankerConfig(BaseRerankerConfig):
    """LLM reranker config: optional `{provider, config}` LLM envelope, input truncation and the fallback switch.
    LLM 重排器配置：可选的 `{provider, config}` LLM 信封、输入截断与回退开关。
    """

    llm: dict[str, Any] | None = None
    max_chars: int = Field(default=4000, ge=1)
    fallback: bool = True


def _ranking_items(raw: Any) -> list | None:
    """Extract the ranking list from a parsed reply; None when the reply is unusable.
    从解析后的回复中提取 ranking 列表；回复不可用时返回 None。
    """
    if isinstance(raw, dict):
        items = raw.get(RERANK_ROOT_KEY)
        return items if isinstance(items, list) else None
    # parse_llm_json returns [] on garbage; a bare non-empty list is accepted as the ranking itself.
    # parse_llm_json 遇到垃圾输出返回 []；裸的非空列表被接受为 ranking 本身。
    if isinstance(raw, list) and raw:
        return raw
    return None


class LLMReranker(RerankerBase):
    """Rerank through an LLM: candidates are shown as '0'..'n', the reply is `{"ranking": [{"id", "score"}]}`.
    通过 LLM 重排：候选以 '0'..'n' 呈现，回复为 `{"ranking": [{"id", "score"}]}`。
    """

    config_class: ClassVar[type[BaseRerankerConfig]] = LLMRerankerConfig
    provider_name: ClassVar[str] = "llm"

    def __init__(self, config: LLMRerankerConfig | dict | None = None, *, llm: LLMBase | None = None) -> None:
        """Bind the LLM (injected > config.llm envelope > rule default) and the overlap fallback.
        绑定 LLM（注入 > config.llm 信封 > 默认规则提供者）与覆盖率回退器。
        """
        super().__init__(config)
        if llm is not None:
            self.llm: LLMBase = llm
        else:
            envelope = dict(self.config.llm or {})  # type: ignore[attr-defined]
            provider = str(envelope.get("provider") or "rule")
            self.llm = LlmFactory.create(provider, envelope.get("config"))
        self._overlap = OverlapReranker({"top_k": self.config.top_k, "text_key": self.config.text_key})
        self.last_route: str = "llm"

    @property
    def identity(self) -> str:
        """`'llm:' + the bound LLM's identity`, e.g. 'llm:rule@v1'.
        `'llm:' + 所绑定 LLM 的身份`，如 'llm:rule@v1'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The identity string.
            身份字符串。

        State effect / 状态影响：
            None.
            无。
        """
        return f"llm:{self.llm.identity}"

    def rerank(self, query: str, rows: list[dict], *, top_k: int | None = None) -> list[dict]:
        """Score rows against the query and return them re-ordered.
        按查询为各行打分并返回重新排序的行。

        Input / 输入：
            query: The search query text (truncated to config.max_chars in the prompt).
            query：搜索查询文本（提示词中截断到 config.max_chars）。

            rows: Mem0-shaped result rows in retrieval order (never mutated).
            rows：按检索顺序排列的 Mem0 形状结果行（从不被修改）。

            top_k: Cap on the returned rows; None → config.top_k; both None → all rows.
            top_k：返回行数上限；None → config.top_k；两者皆 None → 全部行。

        Output / 输出：
            Row copies with 'rerank_score'; unknown ids dropped (warning), omitted rows last in retrieval order.
            带 'rerank_score' 的行副本；未知 id 丢弃（告警），未提及的行按检索顺序排在末尾。

        State effect / 状态影响：
            Sets `last_route`; LLM_002 → overlap fallback with a warning (or raised when fallback=False);
            other LLMError propagates; an unusable reply keeps the retrieval order with a warning.
            设置 `last_route`；LLM_002 → 告警并回退覆盖率（fallback=False 时抛出）；其他 LLMError 向上传播；
            不可用的回复保持检索顺序并告警。
        """
        limit = resolve_top_k(top_k, self.config.top_k)
        if not rows:
            self.last_route = "passthrough"
            return []
        max_chars = int(self.config.max_chars)  # type: ignore[attr-defined]
        key = self.config.text_key
        candidates = [{"id": str(i), "text": row_text(row, key)[:max_chars]} for i, row in enumerate(rows)]
        user = build_rerank_prompt(query=str(query)[:max_chars], candidates=candidates)
        try:
            raw = self.llm.generate_json(RERANK_TASK, system=RERANK_PROMPT, user=user)
        except LLMError as exc:
            if exc.error_code == "LLM_002" and self.config.fallback:  # type: ignore[attr-defined]
                logger.warning(
                    "rerank: provider %s does not support the rerank task; falling back to %s",
                    self.llm.identity,
                    self._overlap.identity,
                )
                self.last_route = "overlap"
                return self._overlap.rerank(query, rows, top_k=limit)
            raise
        items = _ranking_items(raw)
        if items is None:
            logger.warning("rerank: unusable ranking from %s; keeping the retrieval order", self.llm.identity)
            self.last_route = "passthrough"
            return rank_rows(rows, [None] * len(rows), top_k=limit)
        scores: list[float | None] = [None] * len(rows)
        for item in items:
            if not isinstance(item, dict):
                logger.warning("rerank: dropping non-object ranking item %r", item)
                continue
            raw_id = item.get("id")
            index = _candidate_index(raw_id, len(rows))
            if index is None:
                logger.warning("rerank: dropping unknown candidate id %r", raw_id)
                continue
            if scores[index] is not None:
                # The first mention of an id wins; later duplicates are ignored.
                # 同一 id 以首次出现为准；之后的重复项忽略。
                continue
            score = clamp_score(item.get("score"))
            if score is None:
                logger.warning("rerank: dropping candidate %r with unusable score %r", raw_id, item.get("score"))
                continue
            scores[index] = score
        self.last_route = "llm"
        return rank_rows(rows, scores, top_k=limit)


def _candidate_index(raw_id: Any, count: int) -> int | None:
    """Map a reply id ('3', 3) back to a candidate index; None for unknown or malformed ids.
    把回复中的 id（'3'、3）映射回候选下标；未知或畸形 id 返回 None。
    """
    if isinstance(raw_id, bool) or raw_id is None:
        return None
    text = str(raw_id).strip()
    if not text.isdigit():
        return None
    index = int(text)
    return index if 0 <= index < count else None


__all__ = ["LLMReranker", "LLMRerankerConfig"]
