"""Ollama embeddings provider over the local HTTP API (`/api/embeddings`; stdlib urllib, no SDK, no API key).
Ollama 嵌入提供者，走本地 HTTP API（`/api/embeddings`；仅用标准库 urllib，无 SDK，无 API key）。
"""

# Purpose (EN): Real semantic embeddings from a local Ollama server; dims are inferred from the first reply.
# 目的（中）：由本地 Ollama 服务提供真实语义嵌入；维数从首个回复推断。

# Input (EN): OllamaEmbedderConfig (model, embedding_dims, ollama_base_url, timeout) and texts.
# 输入（中）：OllamaEmbedderConfig（模型、维数、ollama_base_url、超时）与文本。

# Output (EN): Float vectors from `/api/embeddings`; HTTP exceptions propagate (embed_batch wraps them).
# 输出（中）：`/api/embeddings` 返回的浮点向量；HTTP 异常传播（embed_batch 负责包装）。

# State effect (EN): Network calls only (loopback by default); the inferred dims are cached on the instance.
# 状态影响（中）：仅有网络调用（默认走本机回环）；推断出的维数缓存在实例上。

from __future__ import annotations

import json
import os
from typing import ClassVar, Literal
from urllib.request import Request, urlopen

from hms_elite.embeddings.base import BaseEmbedderConfig, EmbedderBase

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"
EMBEDDINGS_PATH = "/api/embeddings"
# The text embedded once to learn the vector size when embedding_dims is not configured.
# 未配置 embedding_dims 时，为得知向量大小而嵌入一次的探测文本。
DIMS_PROBE_TEXT = "dimension probe"


class OllamaEmbedderConfig(BaseEmbedderConfig):
    """Ollama embedder config (`embedding_dims` None → inferred from the first reply).
    Ollama 嵌入器配置（`embedding_dims` 为 None → 从首个回复推断）。
    """

    model: str | None = DEFAULT_MODEL
    ollama_base_url: str | None = None
    timeout: float = 120.0


def _post_json(url: str, body: dict, timeout: float) -> dict:
    """POST a JSON body with urllib and decode the JSON reply (HTTPError / URLError propagate).
    用 urllib POST 一个 JSON 体并解码 JSON 回复（HTTPError / URLError 直接传播）。
    """
    data = json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaEmbedder(EmbedderBase):
    """Ollama embeddings provider.
    Ollama 嵌入提供者。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = OllamaEmbedderConfig
    provider_name: ClassVar[str] = "ollama"

    def __init__(self, config: OllamaEmbedderConfig | dict | None = None) -> None:
        """Resolve the base url from config.ollama_base_url / OLLAMA_BASE_URL / the localhost default.
        由 config.ollama_base_url / OLLAMA_BASE_URL / 本机默认值解析基础 url。
        """
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        base_url = getattr(self.config, "ollama_base_url", None) or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/")
        self._dims: int | None = int(self.config.embedding_dims) if self.config.embedding_dims else None

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text (newlines replaced by spaces, Mem0 idiom).
            text：文本（换行替换为空格，Mem0 惯例）。

            memory_action: Ignored.
            memory_action：忽略。

        Output / 输出：
            The `embedding` list of `POST /api/embeddings {'model', 'prompt'}`; its length is cached as dims.
            `POST /api/embeddings {'model', 'prompt'}` 返回的 `embedding` 列表；其长度缓存为 dims。

        State effect / 状态影响：
            One HTTP POST; the first reply fixes `dims` when it was not configured.
            一次 HTTP POST；未配置时首个回复固定 `dims`。
        """
        cleaned = str(text).replace("\n", " ")
        body = {"model": self.config.model, "prompt": cleaned}
        response = _post_json(f"{self.base_url}{EMBEDDINGS_PATH}", body, float(self.config.timeout))
        vector = [float(v) for v in response["embedding"]]
        if self._dims is None:
            self._dims = len(vector)
        return vector

    @property
    def dims(self) -> int:
        """config.embedding_dims, else the size learned from the first reply (probing once when nothing was embedded yet).
        config.embedding_dims，否则为首个回复得到的大小（尚未嵌入过时探测一次）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Positive integer.
            正整数。

        State effect / 状态影响：
            One probe HTTP POST when dims are neither configured nor cached.
            维数既未配置也未缓存时发一次探测 HTTP POST。
        """
        if self._dims is None:
            self.embed(DIMS_PROBE_TEXT)
        return int(self._dims or 0)

    @property
    def identity(self) -> str:
        """'ollama:<model>@<dims>'.
        'ollama:<模型>@<维数>'。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Identity string.
            身份字符串。

        State effect / 状态影响：
            May trigger the dims probe (see `dims`).
            可能触发维数探测（见 `dims`）。
        """
        return f"ollama:{self.config.model}@{self.dims}"


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DIMS_PROBE_TEXT",
    "EMBEDDINGS_PATH",
    "OllamaEmbedder",
    "OllamaEmbedderConfig",
]
