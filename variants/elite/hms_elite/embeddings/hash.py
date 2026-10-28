"""Hashed character n-gram embedder: pure Python, deterministic across processes, no model download.
哈希字符 n-gram 嵌入器：纯 Python、跨进程确定性、无需下载模型。
"""

# Purpose (EN): The offline default embedder so every test and demo runs without a network or numpy.
# 目的（中）：离线默认嵌入器，使所有测试与演示无需网络或 numpy 即可运行。

# Input (EN): Text (EN/ZH), HashEmbedderConfig (embedding_dims, ngram_min, ngram_max).
# 输入（中）：文本（中英）、HashEmbedderConfig（embedding_dims、ngram_min、ngram_max）。

# Output (EN): L2-normalized vectors (all zeros for empty text).
# 输出（中）：L2 归一化向量（空文本为全零）。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

import hashlib
import math
from typing import ClassVar, Literal

from hms_elite.embeddings.base import BaseEmbedderConfig, EmbedderBase


class HashEmbedderConfig(BaseEmbedderConfig):
    """Hash embedder config.
    哈希嵌入器配置。
    """

    embedding_dims: int = 64
    ngram_min: int = 2
    ngram_max: int = 3


class HashEmbedder(EmbedderBase):
    """SHA-256 bucketed character n-grams with a sign bit, L2-normalized.
    以 SHA-256 分桶、带符号位、L2 归一化的字符 n-gram 嵌入。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = HashEmbedderConfig
    provider_name: ClassVar[str] = "hash"

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text to embed (lower-cased, padded with one space on each side).
            text：待嵌入文本（转小写，两侧各补一个空格）。

            memory_action: Ignored.
            memory_action：忽略。

        Output / 输出：
            L2-normalized vector of `dims` floats; all zeros for blank text.
            `dims` 维 L2 归一化向量；空白文本为全零。

        State effect / 状态影响：
            None.
            无。
        """
        dims = self.dims
        vector = [0.0] * dims
        if not text or not str(text).strip():
            return vector
        padded = f" {str(text).lower()} "
        lo = max(1, int(self.config.ngram_min))  # type: ignore[attr-defined]
        hi = max(lo, int(self.config.ngram_max))  # type: ignore[attr-defined]
        for n in range(lo, hi + 1):
            for i in range(len(padded) - n + 1):
                digest = hashlib.sha256(padded[i : i + n].encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % dims
                sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
                vector[bucket] += sign
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    @property
    def dims(self) -> int:
        """Vector dimensionality from config.embedding_dims (default 64).
        由 config.embedding_dims 决定的向量维数（默认 64）。

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
        return int(self.config.embedding_dims or 64)

    @property
    def identity(self) -> str:
        """'hash@<dims>'.
        'hash@<维数>'。

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
        return f"hash@{self.dims}"


__all__ = ["HashEmbedder", "HashEmbedderConfig"]
