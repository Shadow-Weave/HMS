"""API-call ledger for the real-LLM EgoLife run: one record per attempt, DeepSeek pricing by rate tier, per-arm totals.
真实 LLM EgoLife 评测的 API 调用台账：每次尝试一条记录，按档位套用 DeepSeek 定价，逐臂汇总。
"""

# Purpose (EN): Keep every DeepSeek attempt (ok, timeout, http_error, empty) with the provider-returned usage, the
#               UTC rate tier at send time and the resulting USD cost; keep embedding work as local counts (no API
#               token); aggregate per arm and stage for the token / cost / stability tables.
# 目的（中）：保留每次 DeepSeek 尝试（ok、timeout、http_error、empty）及其 API 返回的 usage、发送时刻的 UTC 档位与
#            由此算出的美元费用；嵌入工作只记本地计数（无 API token）；按臂与阶段汇总，供 token / 费用 / 稳定性表使用。

# Input (EN): ApiCall fields filled by real_arms.DeepSeekClient; embedding counts from the arm wrappers.
# 输入（中）：由 real_arms.DeepSeekClient 填好的 ApiCall 字段；臂包装层报告的嵌入计数。

# Output (EN): calls.jsonl lines, Meter.summary() dicts; cost_usd is None (reported as 未实测) when usage lacks
#              the cache split, never an estimate.
# 输出（中）：calls.jsonl 各行、Meter.summary() 字典；usage 缺少缓存拆分时 cost_usd 为 None（报告为「未实测」），绝不估算。

# State effect (EN): In-memory lists only; write_jsonl writes one file.
# 状态影响（中）：仅内存列表；write_jsonl 写一个文件。

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Official list price, USD per 1M tokens, captured 2026-09-10 from https://api-docs.deepseek.com/quick_start/pricing.
# 官方定价（USD / 1M tokens），2026-09-10 抓自 https://api-docs.deepseek.com/quick_start/pricing。
PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing (captured 2026-09-10)"
MODEL = "deepseek-flash"
PRICES_USD_PER_M: dict[str, dict[str, float]] = {
    "offpeak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
    "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
}
# Peak = weekdays (Mon..Fri) in UTC [01:00, 04:00) and [06:00, 10:00); everything else is off-peak (half price).
# 高峰 = 工作日（周一至周五）UTC [01:00, 04:00) 与 [06:00, 10:00)；其余时段为非高峰（半价）。
PEAK_WINDOWS_UTC: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))
STAGES = ("extract", "answer")
STATUSES = ("ok", "timeout", "http_error", "empty")
STAGE_EMBED = "embedding"


def rate_tier(at: datetime) -> str:
    """Classify a UTC instant as 'peak' or 'offpeak' under the DeepSeek schedule above.
    按上面的 DeepSeek 时段表把一个 UTC 时刻判为 'peak' 或 'offpeak'。

    Input / 输入：
        at: Timezone-aware datetime (converted to UTC).
        at：带时区的 datetime（转换为 UTC）。

    Output / 输出：
        'peak' or 'offpeak'.
        'peak' 或 'offpeak'。

    State effect / 状态影响：
        None.
        无。
    """
    utc = at.astimezone(timezone.utc)
    if utc.weekday() >= 5:
        return "offpeak"
    return "peak" if any(lo <= utc.hour < hi for lo, hi in PEAK_WINDOWS_UTC) else "offpeak"


def cost_usd(
    completion_tokens: int | None, cache_hit_tokens: int | None, cache_miss_tokens: int | None, tier: str
) -> float | None:
    """USD cost of one attempt: miss x miss-price + hit x hit-price + completion x output-price (per-tier prices).
    一次尝试的美元费用：未命中 x 未命中价 + 命中 x 命中价 + 补全 x 输出价（按档位取价）。

    Input / 输入：
        completion_tokens: Provider completion_tokens (None when the reply carried no usage).
        completion_tokens：API 返回的 completion_tokens（回复没有 usage 时为 None）。

        cache_hit_tokens: Provider prompt_cache_hit_tokens.
        cache_hit_tokens：API 返回的 prompt_cache_hit_tokens。

        cache_miss_tokens: Provider prompt_cache_miss_tokens.
        cache_miss_tokens：API 返回的 prompt_cache_miss_tokens。

        tier: 'peak' | 'offpeak'.
        tier：'peak' | 'offpeak'。

    Output / 输出：
        Float USD, or None when any of the three counts is missing (reported as 未实测, never estimated).
        美元浮点数；三项任一缺失时为 None（报告为「未实测」，绝不估算）。

    State effect / 状态影响：
        None.
        无。
    """
    if completion_tokens is None or cache_hit_tokens is None or cache_miss_tokens is None:
        return None
    price = PRICES_USD_PER_M[tier]
    return (
        cache_miss_tokens * price["cache_miss"] + cache_hit_tokens * price["cache_hit"] + completion_tokens * price["output"]
    ) / 1_000_000.0


@dataclass
class ApiCall:
    """One DeepSeek attempt (a retried request produces several ApiCalls with increasing `attempt`).
    一次 DeepSeek 尝试（被重试的请求产生多条 ApiCall，`attempt` 递增）。
    """

    arm: str  # engine or label: mem0_real | elite (extract) | mem0_real | elite_rows | elite_hybrid | elite_card (answer) / 引擎或标签
    stage: str  # extract | answer / 阶段
    question_id: str | None
    attempt: int  # 0 = first try, 1..2 = retries / 0 = 首次，1..2 = 重试
    started_utc: str
    latency_ms: float
    status: str  # ok | timeout | http_error | empty / 状态
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    rate_tier: str = "offpeak"
    cost_usd: float | None = None
    error: str | None = None
    model: str = MODEL
    batch_id: str | None = None
    cold_start: bool = False  # first API call of this arm / 该臂的第一次 API 调用
    http_status: int | None = None
    finish_reason: str | None = None
    usage_reconciles: bool | None = None  # hit + miss == prompt_tokens / 命中 + 未命中 == prompt_tokens
    reasoning_tokens: int | None = None  # completion_tokens_details.reasoning_tokens (hidden thinking) / 隐藏思考 token
    content_chars: int | None = None
    # Sub-add of a bisected batch ('<lo>-<hi>' window offsets inside the batch; None = the whole batch). A truncated
    # extraction (finish_reason == 'length') is never accepted: the batch is re-submitted as two halves, each a new
    # priced request with its own ApiCall.
    # 被二分批次的子 add（批次内窗口偏移 '<lo>-<hi>'；None = 整批）。被截断的抽取（finish_reason == 'length'）不被接受：
    # 该批次拆成两半重提，每半是一次新的计费请求、各有自己的 ApiCall。
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSONL.
        用于 JSONL 的普通字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            asdict(self).
            asdict(self)。

        State effect / 状态影响：
            None.
            无。
        """
        return asdict(self)


@dataclass
class EmbeddingWork:
    """One local embedding call: text count and wall time (no API token, no cost).
    一次本地嵌入调用：文本数与墙钟耗时（无 API token，无费用）。
    """

    arm: str
    stage: str  # extract | recall | other / 阶段
    texts: int
    local_ms: float
    cold_start: bool = False


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _sum_opt(values: list[int | None]) -> int | None:
    """Sum ints; None when every value is None (nothing measured).
    整数求和；全部为 None（没有任何实测值）时返回 None。
    """
    known = [v for v in values if v is not None]
    return sum(known) if known else None


@dataclass
class Meter:
    """Ledger of ApiCalls and EmbeddingWork with per-arm / per-stage aggregation.
    ApiCall 与 EmbeddingWork 的台账，按臂 / 阶段汇总。
    """

    calls: list[ApiCall] = field(default_factory=list)
    embeddings: list[EmbeddingWork] = field(default_factory=list)
    _seen_arms: set[str] = field(default_factory=set)
    _seen_embed_arms: set[str] = field(default_factory=set)

    def next_is_cold(self, arm: str) -> bool:
        """Whether the next API call of `arm` is its first (cold start); marks the arm as seen.
        `arm` 的下一次 API 调用是否为首次（冷启动）；同时把该臂标记为已见。

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

        Output / 输出：
            True exactly once per arm.
            每个臂恰好一次为 True。

        State effect / 状态影响：
            Adds arm to the seen set.
            把臂加入已见集合。
        """
        if arm in self._seen_arms:
            return False
        self._seen_arms.add(arm)
        return True

    def add(self, call: ApiCall) -> ApiCall:
        """Append one attempt.
        追加一次尝试。

        Input / 输入：
            call: The record (cost computed here when missing and the usage split is present).
            call：记录（cost 缺失且 usage 拆分齐全时在此计算）。

        Output / 输出：
            The same record.
            同一条记录。

        State effect / 状态影响：
            Appends to calls.
            追加到 calls。
        """
        if call.cost_usd is None:
            call.cost_usd = cost_usd(call.completion_tokens, call.cache_hit_tokens, call.cache_miss_tokens, call.rate_tier)
        if call.prompt_tokens is not None and call.cache_hit_tokens is not None and call.cache_miss_tokens is not None:
            call.usage_reconciles = call.cache_hit_tokens + call.cache_miss_tokens == call.prompt_tokens
        self.calls.append(call)
        return call

    def add_embedding(self, arm: str, stage: str, texts: int, local_ms: float) -> EmbeddingWork:
        """Record one local embedding call.
        记录一次本地嵌入调用。

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

            stage: extract | recall | other.
            stage：extract | recall | other。

            texts: Number of texts embedded.
            texts：嵌入的文本数。

            local_ms: Wall time.
            local_ms：墙钟耗时。

        Output / 输出：
            The record.
            该记录。

        State effect / 状态影响：
            Appends to embeddings; the first call per arm is flagged cold_start.
            追加到 embeddings；每个臂的第一次调用标记 cold_start。
        """
        cold = arm not in self._seen_embed_arms
        self._seen_embed_arms.add(arm)
        work = EmbeddingWork(arm=arm, stage=stage, texts=int(texts), local_ms=float(local_ms), cold_start=cold)
        self.embeddings.append(work)
        return work

    def calls_for(self, arm: str, stage: str | None = None, question_id: str | None = None) -> list[ApiCall]:
        """Filter the ledger.
        过滤台账。

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

            stage: Optional stage filter.
            stage：可选阶段过滤。

            question_id: Optional question filter.
            question_id：可选题目过滤。

        Output / 输出：
            Matching calls in ledger order.
            按台账顺序的匹配调用。

        State effect / 状态影响：
            None.
            无。
        """
        return [
            c
            for c in self.calls
            if c.arm == arm and (stage is None or c.stage == stage) and (question_id is None or c.question_id == question_id)
        ]

    def stage_summary(self, calls: list[ApiCall]) -> dict[str, Any]:
        """Token / cost / stability totals over a call subset (failed and retried attempts included).
        一个调用子集上的 token / 费用 / 稳定性总量（含失败与重试的尝试）。

        Input / 输入：
            calls: ApiCall subset.
            calls：ApiCall 子集。

        Output / 输出：
            Dict: attempts, requests (attempt==0), retries, by status, tokens (prompt/completion/hit/miss, None when
            未实测), cost_usd (sum over priced attempts), unpriced_attempts, tier split, latency percentiles
            (all attempts / ok attempts / ok excluding cold start).
            字典：尝试数、请求数（attempt==0）、重试数、按状态计数、token（prompt/completion/hit/miss，未实测为 None）、
            cost_usd（已计价尝试之和）、unpriced_attempts、档位拆分、延迟分位数（全部 / ok / ok 且不含冷启动）。

        State effect / 状态影响：
            None.
            无。
        """
        by_status = {s: sum(1 for c in calls if c.status == s) for s in STATUSES}
        priced = [c for c in calls if c.cost_usd is not None]
        lat_all = [c.latency_ms for c in calls]
        lat_ok = [c.latency_ms for c in calls if c.status in ("ok", "empty")]
        lat_warm = [c.latency_ms for c in calls if c.status in ("ok", "empty") and not c.cold_start]
        tiers = {t: sum(1 for c in calls if c.rate_tier == t) for t in ("peak", "offpeak")}
        requests = sum(1 for c in calls if c.attempt == 0)
        failed_attempts = by_status["timeout"] + by_status["http_error"]
        return {
            "attempts": len(calls),
            "requests": requests,
            "retries": len(calls) - requests,
            "by_status": by_status,
            "failed_attempts": failed_attempts,
            "timeout_rate": round(by_status["timeout"] / len(calls), 4) if calls else None,
            "failure_rate": round(failed_attempts / len(calls), 4) if calls else None,
            "empty_rate": round(by_status["empty"] / len(calls), 4) if calls else None,
            "tokens": {
                "prompt": _sum_opt([c.prompt_tokens for c in calls]),
                "completion": _sum_opt([c.completion_tokens for c in calls]),
                "cache_hit": _sum_opt([c.cache_hit_tokens for c in calls]),
                "cache_miss": _sum_opt([c.cache_miss_tokens for c in calls]),
                "reasoning": _sum_opt([c.reasoning_tokens for c in calls]),
                "attempts_without_usage": sum(1 for c in calls if c.prompt_tokens is None),
                "usage_reconciles_all": all(c.usage_reconciles for c in calls if c.usage_reconciles is not None),
            },
            "cost_usd": round(sum(c.cost_usd for c in priced), 6),
            "priced_attempts": len(priced),
            "unpriced_attempts": len(calls) - len(priced),
            "tier_attempts": tiers,
            "cost_usd_by_tier": {
                t: round(sum(c.cost_usd for c in priced if c.rate_tier == t), 6) for t in ("peak", "offpeak")
            },
            "latency_ms": {
                "all_attempts": {"median": _p(lat_all, 0.5), "p95": _p(lat_all, 0.95), "n": len(lat_all)},
                "ok": {"median": _p(lat_ok, 0.5), "p95": _p(lat_ok, 0.95), "n": len(lat_ok)},
                "ok_excluding_cold_start": {"median": _p(lat_warm, 0.5), "p95": _p(lat_warm, 0.95), "n": len(lat_warm)},
                "cold_start_ms": next((c.latency_ms for c in calls if c.cold_start), None),
            },
        }

    def embedding_summary(self, arm: str) -> dict[str, Any]:
        """Local embedding totals for one arm (texts, calls, wall time) split by stage.
        一个臂的本地嵌入总量（文本数、调用数、墙钟耗时），按阶段拆分。

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

        Output / 输出：
            {'api_tokens': '本地（无 API token）', stage: {calls, texts, local_ms_total, local_ms_median, cold_start_ms}}.
            {'api_tokens': '本地（无 API token）', 阶段: {调用数, 文本数, 本地总耗时, 中位数, 冷启动耗时}}。

        State effect / 状态影响：
            None.
            无。
        """
        rows = [e for e in self.embeddings if e.arm == arm]
        out: dict[str, Any] = {"api_tokens": "本地（无 API token）", "calls": len(rows), "texts": sum(e.texts for e in rows)}
        out["local_ms_total"] = round(sum(e.local_ms for e in rows), 3)
        out["cold_start_ms"] = next((round(e.local_ms, 3) for e in rows if e.cold_start), None)
        by_stage: dict[str, list[EmbeddingWork]] = defaultdict(list)
        for e in rows:
            by_stage[e.stage].append(e)
        out["by_stage"] = {
            s: {
                "calls": len(v),
                "texts": sum(e.texts for e in v),
                "local_ms_total": round(sum(e.local_ms for e in v), 3),
                "local_ms_median": round(statistics.median(e.local_ms for e in v), 3) if v else None,
            }
            for s, v in sorted(by_stage.items())
        }
        return out

    def summary(self, arm: str) -> dict[str, Any]:
        """Per-stage summaries plus the arm total for one arm label.
        一个臂标签的逐阶段汇总加臂总计。

        Input / 输入：
            arm: Arm label.
            arm：臂标签。

        Output / 输出：
            {'extract': ..., 'answer': ..., 'total': ..., 'embedding': ...}.
            {'extract': ..., 'answer': ..., 'total': ..., 'embedding': ...}。

        State effect / 状态影响：
            None.
            无。
        """
        out = {s: self.stage_summary(self.calls_for(arm, s)) for s in STAGES}
        out["total"] = self.stage_summary(self.calls_for(arm))
        out["embedding"] = self.embedding_summary(arm)
        return out

    def write_jsonl(self, path: Path) -> int:
        """Write every ApiCall as one JSON line (failed attempts included).
        把每条 ApiCall 写成一行 JSON（含失败的尝试）。

        Input / 输入：
            path: Destination.
            path：目标文件。

        Output / 输出：
            Number of lines written.
            写入的行数。

        State effect / 状态影响：
            Writes path.
            写入 path。
        """
        with path.open("w", encoding="utf-8") as fh:
            for c in self.calls:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        return len(self.calls)


def utc_now() -> datetime:
    """Timezone-aware UTC now.
    带时区的当前 UTC 时间。

    Input / 输入：
        none.
        无。

    Output / 输出：
        datetime in UTC.
        UTC 的 datetime。

    State effect / 状态影响：
        None.
        无。
    """
    return datetime.now(timezone.utc)


__all__ = [
    "MODEL",
    "PEAK_WINDOWS_UTC",
    "PRICES_USD_PER_M",
    "PRICING_SOURCE",
    "STAGES",
    "STATUSES",
    "STAGE_EMBED",
    "ApiCall",
    "EmbeddingWork",
    "Meter",
    "cost_usd",
    "rate_tier",
    "utc_now",
]
