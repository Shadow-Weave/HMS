"""OpenAI embeddings provider (optional `openai` extra; DependencyError when the SDK is missing).
OpenAI 嵌入提供者（可选 `openai` extra；缺少 SDK 时抛出 DependencyError）。
"""

# Purpose (EN): Real semantic embeddings through the OpenAI SDK with Mem0's client conventions.
# 目的（中）：以 Mem0 的客户端约定，通过 OpenAI SDK 提供真实语义嵌入。

# Input (EN): OpenAIEmbedderConfig (model, embedding_dims, api_key, base_url) and texts.
# 输入（中）：OpenAIEmbedderConfig（模型、维数、api_key、base_url）与文本。

# Output (EN): Float vectors from embeddings.create; SDK exceptions propagate (embed_batch wraps them).
# 输出（中）：embeddings.create 返回的浮点向量；SDK 异常传播（embed_batch 负责包装）。

# State effect (EN): Network calls; each response's `usage` is handed to EmbedderBase's meter via `_last_usage`.
# 状态影响（中）：网络调用；每次响应的 `usage` 经 `_last_usage` 交给 EmbedderBase 的计量器。

from __future__ import annotations

import os
from typing import ClassVar, Literal

from hms_elite.embeddings.base import BaseEmbedderConfig, EmbedderBase
from hms_elite.exceptions import DependencyError

# Mem0-style import guard: the check runs at module import; the error is raised when a provider is constructed so
# that the module itself stays importable (test_bilingual_docstrings imports every module).
# Mem0 风格的导入守卫：检查在模块导入时执行，错误则推迟到构造 provider 时抛出，
# 从而让模块本身保持可导入（test_bilingual_docstrings 会导入每一个模块）。
_IMPORT_ERROR: ImportError | None = None
try:
    from openai import OpenAI
except ImportError as _exc:  # pragma: no cover - depends on the environment
    OpenAI = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedderConfig(BaseEmbedderConfig):
    """OpenAI embedder config.
    OpenAI 嵌入器配置。
    """

    model: str | None = DEFAULT_MODEL
    embedding_dims: int = 1536
    openai_base_url: str | None = None


class OpenAIEmbedder(EmbedderBase):
    """OpenAI embeddings provider.
    OpenAI 嵌入提供者。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = OpenAIEmbedderConfig
    provider_name: ClassVar[str] = "openai"

    def __init__(self, config: OpenAIEmbedderConfig | dict | None = None) -> None:
        """Build the client from config.api_key / OPENAI_API_KEY and config.openai_base_url / OPENAI_BASE_URL.
        由 config.api_key / OPENAI_API_KEY 与 config.openai_base_url / OPENAI_BASE_URL 构建客户端。
        """
        if _IMPORT_ERROR is not None:
            raise DependencyError(
                "The 'openai' package is required for the openai embedder",
                "DEPS_001",
                details={"package": "openai", "extra": "openai"},
            ) from _IMPORT_ERROR
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = getattr(self.config, "openai_base_url", None) or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text (newlines replaced by spaces, Mem0 idiom).
            text：文本（换行替换为空格，Mem0 惯例）。

            memory_action: Ignored.
            memory_action：忽略。

        Output / 输出：
            The embedding from `embeddings.create(model, input=text, dimensions=embedding_dims)`.
            `embeddings.create(model, input=text, dimensions=embedding_dims)` 返回的嵌入。

        State effect / 状态影响：
            None (network for real providers).
            无（真实提供者有网络调用）。
        """
        cleaned = str(text).replace("\n", " ")
        response = self.client.embeddings.create(model=self.config.model, input=cleaned, dimensions=self.dims)
        # The SDK usage (prompt_tokens) feeds EmbedderBase's meter; None keeps the estimate path.
        # SDK 用量（prompt_tokens）喂给 EmbedderBase 的计量器；为 None 时走估算路径。
        self._last_usage = getattr(response, "usage", None)
        return [float(v) for v in response.data[0].embedding]

    @property
    def dims(self) -> int:
        """config.embedding_dims (default 1536).
        config.embedding_dims（默认 1536）。

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
        return int(self.config.embedding_dims or 1536)

    @property
    def identity(self) -> str:
        """'openai:<model>@<dims>'.
        'openai:<模型>@<维数>'。

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
        return f"openai:{self.config.model}@{self.dims}"


__all__ = ["DEFAULT_MODEL", "OpenAIEmbedder", "OpenAIEmbedderConfig"]
