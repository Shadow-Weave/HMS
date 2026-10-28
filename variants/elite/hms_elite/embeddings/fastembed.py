"""fastembed local ONNX embeddings provider (optional `fastembed` extra; DependencyError when the package is missing).
fastembed 本地 ONNX 嵌入提供者（可选 `fastembed` extra；缺少包时抛出 DependencyError）。
"""

# Purpose (EN): Real semantic embeddings computed locally through fastembed's TextEmbedding (no API key, no server).
# 目的（中）：通过 fastembed 的 TextEmbedding 在本地计算真实语义嵌入（无 API key，无服务端）。

# Input (EN): FastEmbedEmbedderConfig (model, embedding_dims, cache_dir, threads) and texts.
# 输入（中）：FastEmbedEmbedderConfig（模型、维数、cache_dir、threads）与文本。

# Output (EN): Float vectors from TextEmbedding.embed; library exceptions propagate (embed_batch wraps them).
# 输出（中）：TextEmbedding.embed 返回的浮点向量；库异常传播（embed_batch 负责包装）。

# State effect (EN): The first construction may download model weights into cache_dir (fastembed behaviour).
# 状态影响（中）：首次构造可能把模型权重下载到 cache_dir（fastembed 自身行为）。

from __future__ import annotations

from typing import Any, ClassVar, Literal

from hms_elite.embeddings.base import BaseEmbedderConfig, EmbedderBase
from hms_elite.exceptions import DependencyError

# Mem0-style import guard: the check runs at module import; the error is raised when a provider is constructed so
# that the module itself stays importable (test_bilingual_docstrings imports every module).
# Mem0 风格的导入守卫：检查在模块导入时执行，错误则推迟到构造 provider 时抛出，
# 从而让模块本身保持可导入（test_bilingual_docstrings 会导入每一个模块）。
_IMPORT_ERROR: ImportError | None = None
try:
    from fastembed import TextEmbedding
except ImportError as _exc:  # pragma: no cover - depends on the environment
    TextEmbedding = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedderConfig(BaseEmbedderConfig):
    """fastembed embedder config (`api_key` is accepted for parity and ignored).
    fastembed 嵌入器配置（`api_key` 为对齐而保留，实际忽略）。
    """

    model: str | None = DEFAULT_MODEL
    embedding_dims: int = 384
    cache_dir: str | None = None
    threads: int | None = None


class FastEmbedEmbedder(EmbedderBase):
    """fastembed local embeddings provider.
    fastembed 本地嵌入提供者。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = FastEmbedEmbedderConfig
    provider_name: ClassVar[str] = "fastembed"

    def __init__(self, config: FastEmbedEmbedderConfig | dict | None = None) -> None:
        """Build `TextEmbedding(model_name, cache_dir, threads)`; None-valued options are not passed.
        构造 `TextEmbedding(model_name, cache_dir, threads)`；值为 None 的选项不传。
        """
        if _IMPORT_ERROR is not None:
            raise DependencyError(
                "The 'fastembed' package is required for the fastembed embedder",
                "DEPS_001",
                details={"package": "fastembed", "extra": "fastembed"},
            ) from _IMPORT_ERROR
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        options: dict[str, Any] = {"model_name": self.config.model}
        cache_dir = getattr(self.config, "cache_dir", None)
        if cache_dir:
            options["cache_dir"] = cache_dir
        threads = getattr(self.config, "threads", None)
        if threads:
            options["threads"] = int(threads)
        self.model = TextEmbedding(**options)

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text (newlines replaced by spaces, Mem0 idiom).
            text：文本（换行替换为空格，Mem0 惯例）。

            memory_action: Ignored.
            memory_action：忽略。

        Output / 输出：
            The first vector yielded by `TextEmbedding.embed([text])`, as Python floats.
            `TextEmbedding.embed([text])` 产出的第一个向量，转为 Python 浮点数。

        State effect / 状态影响：
            None (local ONNX inference).
            无（本地 ONNX 推理）。
        """
        cleaned = str(text).replace("\n", " ")
        vector = next(iter(self.model.embed([cleaned])))
        return [float(v) for v in vector]

    @property
    def dims(self) -> int:
        """config.embedding_dims (default 384 for BAAI/bge-small-en-v1.5).
        config.embedding_dims（BAAI/bge-small-en-v1.5 默认 384）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Positive integer.
            正整数。

        State effect / 状态影响：
            None.
            无。
        """
        return int(self.config.embedding_dims or 384)

    @property
    def identity(self) -> str:
        """'fastembed:<model>@<dims>'.
        'fastembed:<模型>@<维数>'。

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
        return f"fastembed:{self.config.model}@{self.dims}"


__all__ = ["DEFAULT_MODEL", "FastEmbedEmbedder", "FastEmbedEmbedderConfig"]
