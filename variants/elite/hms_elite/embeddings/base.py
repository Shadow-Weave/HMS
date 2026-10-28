"""Embedding provider contract: BaseEmbedderConfig and EmbedderBase (embed / embed_batch / dims / identity).
嵌入提供者契约：BaseEmbedderConfig 与 EmbedderBase（embed / embed_batch / dims / identity）。
"""

# Purpose (EN): Define the one interface every embedder (hash, openai) implements, with Mem0's method names.
# 目的（中）：定义所有嵌入器（hash、openai）实现的统一接口，沿用 Mem0 的方法名。

# Input (EN): Texts and the memory_action hint ('add' | 'search' | 'update').
# 输入（中）：文本与 memory_action 提示（'add' | 'search' | 'update'）。

# Output (EN): Float vectors; EmbeddingError(EMBED_001) on provider failure.
# 输出（中）：浮点向量；提供者失败时抛出 EmbeddingError(EMBED_001)。

# State effect (EN): Every embed / embed_batch call is metered into `self.usage` (calls, texts, wall time) and, when a
#                    telemetry trace is active, appended to it as an 'embedding.*' stage; network for real providers.
# 状态影响（中）：每次 embed / embed_batch 调用都计入 `self.usage`（调用数、文本数、墙钟耗时），存在活动追踪时
#               作为 'embedding.*' 阶段追加；真实提供者有网络调用。

from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, ClassVar, Literal

from pydantic import model_validator

from hms_elite.exceptions import EmbeddingError
from hms_elite.memory.utils import estimate_tokens
from hms_elite.models import HMSModel
from hms_elite.telemetry import (
    STAGE_EMBED,
    STAGE_EMBED_BATCH,
    MeteredProvider,
    UsageMeter,
    record_stage,
    usage_tokens,
)
from hms_elite.utils.factory import check_extra_fields, coerce_config

# Marker set on a wrapped embed / embed_batch so a subclass of a subclass never nests two meters.
# 打在已包装 embed / embed_batch 上的标记，避免子类的子类再次包装而叠加两层计量。
_INSTRUMENTED = "__hms_metered__"
# Nesting depth of metered embed calls in this context: 0 outside, 1 inside a metered call, 2 inside an item call of a
# metered embed_batch. Deeper re-entry (an override calling super().embed) passes through unmetered.
# 本上下文中计量 embed 调用的嵌套深度：外部为 0，计量调用内为 1，计量 embed_batch 的逐项调用内为 2。
# 更深的重入（覆盖方法调用 super().embed）不计量直接透传。
_DEPTH: ContextVar[int] = ContextVar("hms_elite_embed_depth", default=0)
# While an embed_batch runs this holds the per-item token counts its nested embed calls report (None when a provider
# gave no usage), so the batch stage can sum exact counts and the items are not metered a second time.
# embed_batch 运行期间这里保存其内部逐条 embed 报告的 token 数（提供者未给用量时为 None），
# 使批阶段能汇总精确计数，且逐条调用不被二次计量。
_BATCH_ITEMS: ContextVar[list[int | None] | None] = ContextVar("hms_elite_embed_batch_items", default=None)


def _meter_embed(original: Callable[..., Any], *, batch: bool) -> Callable[..., Any]:
    """Wrap embed (batch=False) or embed_batch (batch=True): count texts, time the call, feed meter + trace.
    包装 embed（batch=False）或 embed_batch（batch=True）：统计文本数、计时，喂给计量器与追踪。
    """
    if getattr(original, _INSTRUMENTED, False):
        return original
    operation = STAGE_EMBED_BATCH if batch else STAGE_EMBED

    @functools.wraps(original)
    def metered(self: EmbedderBase, payload: Any, *args: Any, **kwargs: Any) -> Any:
        depth = _DEPTH.get()
        items = _BATCH_ITEMS.get()
        if depth >= 2 or (depth == 1 and (batch or items is None)):
            return original(self, payload, *args, **kwargs)
        depth_token = _DEPTH.set(depth + 1)
        try:
            if depth == 1:
                # Item call of a metered embed_batch: run unmetered, report the exact count (or None) to the batch.
                # 计量 embed_batch 的逐项调用：不计量地运行，把精确计数（或 None）报告给批调用。
                self._last_usage = None
                result = original(self, payload, *args, **kwargs)
                exact = usage_tokens(self._last_usage)
                self._last_usage = None
                items.append(exact[0] if exact is not None else None)  # type: ignore[union-attr]
                return result
            texts = [str(t) for t in payload] if batch else [str(payload)]
            items_token = _BATCH_ITEMS.set([]) if batch else None
            self._last_usage = None
            started = time.perf_counter()
            try:
                result = original(self, payload, *args, **kwargs)
            except BaseException as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                self.usage.add_embedding(len(texts), duration_ms)
                record_stage(
                    operation,
                    duration_ms=duration_ms,
                    input_tokens=sum(estimate_tokens(t) for t in texts),
                    estimated=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                collected = _BATCH_ITEMS.get() if batch else None
                if items_token is not None:
                    _BATCH_ITEMS.reset(items_token)
            duration_ms = (time.perf_counter() - started) * 1000
            exact = usage_tokens(self._last_usage)
            self._last_usage = None
            if exact is not None:
                prompt, estimated = exact[0], False
            elif collected and all(item is not None for item in collected):
                prompt, estimated = sum(collected), False  # type: ignore[arg-type]
            else:
                prompt, estimated = sum(estimate_tokens(t) for t in texts), True
            self.usage.add_embedding(len(texts), duration_ms)
            record_stage(operation, duration_ms=duration_ms, input_tokens=prompt, estimated=estimated)
            return result
        finally:
            _DEPTH.reset(depth_token)

    setattr(metered, _INSTRUMENTED, True)
    return metered


class BaseEmbedderConfig(HMSModel):
    """Fields shared by every embedder config.
    所有嵌入器配置共享的字段。
    """

    model: str | None = None
    embedding_dims: int | None = None
    api_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _extra_fields(cls, data: Any) -> Any:
        """Unknown keys → ConfigError(CONFIG_001) with a did-you-mean hint.
        未知键 → 带 did-you-mean 提示的 ConfigError(CONFIG_001)。
        """
        return check_extra_fields(cls, data)


class EmbedderBase(MeteredProvider, ABC):
    """Abstract embedder.
    抽象嵌入器。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = BaseEmbedderConfig
    provider_name: ClassVar[str] = "base"

    def __init__(self, config: BaseEmbedderConfig | dict | None = None) -> None:
        """Coerce the config into the provider's config class; start a UsageMeter.
        将配置规整为提供者的配置类；启动一个 UsageMeter。
        """
        self.config: BaseEmbedderConfig = coerce_config(self.config_class, config)  # type: ignore[assignment]
        self.usage = UsageMeter()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Meter the subclass's own embed / embed_batch overrides (signatures and return values unchanged).
        为子类自己覆盖的 embed / embed_batch 加计量（签名与返回值不变）。
        """
        super().__init_subclass__(**kwargs)
        own_embed = cls.__dict__.get("embed")
        if callable(own_embed):
            cls.embed = _meter_embed(own_embed, batch=False)  # type: ignore[method-assign]
        own_batch = cls.__dict__.get("embed_batch")
        if callable(own_batch):
            cls.embed_batch = _meter_embed(own_batch, batch=True)  # type: ignore[method-assign]

    @abstractmethod
    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text to embed.
            text：待嵌入文本。

            memory_action: Mem0 hint ('add' | 'search' | 'update'); ignored by offline providers.
            memory_action：Mem0 提示（'add' | 'search' | 'update'）；离线提供者忽略。

        Output / 输出：
            A vector of `dims` floats.
            `dims` 个浮点数组成的向量。

        State effect / 状态影响：
            None (network for real providers).
            无（真实提供者有网络调用）。
        """

    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]:
        """Embed several texts sequentially.
        顺序嵌入多段文本。

        Input / 输入：
            texts: Texts in order.
            texts：按顺序的文本。

            memory_action: Mem0 hint forwarded to embed().
            memory_action：转发给 embed() 的 Mem0 提示。

        Output / 输出：
            One vector per text, same order.
            每段文本一个向量，顺序一致。

        State effect / 状态影响：
            None; a per-item failure raises EmbeddingError(EMBED_001, details={'index'}) from the cause.
            无；单项失败抛出 EmbeddingError(EMBED_001, details={'index'}) 并保留原因。
        """
        out: list[list[float]] = []
        for index, text in enumerate(texts):
            try:
                out.append(self.embed(text, memory_action))  # type: ignore[arg-type]
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding failed for item {index}: {exc}",
                    "EMBED_001",
                    details={"index": index, "provider": self.identity},
                    suggestion="Please check your embedding model configuration",
                ) from exc
        return out

    # The base class is not its own subclass, so its embed_batch is metered here by hand.
    # 基类不是自己的子类，因此它的 embed_batch 在此手动加计量。
    embed_batch = _meter_embed(embed_batch, batch=True)

    @property
    def dims(self) -> int:
        """Vector dimensionality.
        向量维数。

        Input / 输入：
            none.
            无。

        Output / 输出：
            config.embedding_dims (0 when unknown).
            config.embedding_dims（未知时为 0）。

        State effect / 状态影响：
            None.
            无。
        """
        return int(self.config.embedding_dims or 0)

    @property
    def identity(self) -> str:
        """Embedder identity such as 'hash@64' or 'openai:text-embedding-3-small@1536'.
        嵌入器身份，如 'hash@64' 或 'openai:text-embedding-3-small@1536'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            `f"{provider_name}:{model}@{dims}"` by default; subclasses override.
            默认 `f"{provider_name}:{model}@{dims}"`；子类可覆盖。

        State effect / 状态影响：
            None.
            无。
        """
        return f"{self.provider_name}:{self.config.model}@{self.dims}"


__all__ = ["BaseEmbedderConfig", "EmbedderBase"]
