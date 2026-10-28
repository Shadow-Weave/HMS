"""Volcengine Ark multimodal embeddings provider (`/embeddings/multimodal`; stdlib urllib, no SDK, API key required).
火山方舟多模态嵌入提供者（`/embeddings/multimodal`；仅用标准库 urllib，无 SDK，需要 API key）。
"""

# Purpose (EN): Real semantic embeddings from Ark's doubao-embedding-vision family. Those models are served on the
#               dedicated multimodal path and are rejected by the OpenAI-compatible `/embeddings`, so neither the
#               openai nor the ollama provider can reach them.
# 目的（中）：由方舟 doubao-embedding-vision 系列提供真实语义嵌入。该系列模型只在专用多模态路径上提供服务，
#           会被 OpenAI 兼容的 `/embeddings` 拒绝，因此 openai 与 ollama 两个提供者都够不到它们。

# Input (EN): ArkEmbedderConfig (model, embedding_dims, api_key, ark_base_url, timeout) and texts.
# 输入（中）：ArkEmbedderConfig（模型、维数、api_key、ark_base_url、超时）与文本。

# Output (EN): Float vectors from `/embeddings/multimodal`; HTTP exceptions propagate (embed_batch wraps them).
# 输出（中）：`/embeddings/multimodal` 返回的浮点向量；HTTP 异常传播（embed_batch 负责包装）。

# State effect (EN): Network calls only; each response's `usage` is handed to EmbedderBase's meter via `_last_usage`.
# 状态影响（中）：仅有网络调用；每次响应的 `usage` 经 `_last_usage` 交给 EmbedderBase 的计量器。

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Literal
from urllib.request import Request, urlopen

from hms_elite.embeddings.base import BaseEmbedderConfig, EmbedderBase

DEFAULT_MODEL = "doubao-embedding-vision-251215"  # 默认模型：方舟多模态嵌入
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 方舟 OpenAI 兼容端点前缀
EMBEDDINGS_PATH = "/embeddings/multimodal"  # 多模态嵌入路径（非 OpenAI 兼容的 /embeddings）
# The model's native vector size; Ark also accepts the `dimensions` parameter for a smaller projection.
# 模型的原始向量维数；方舟同时接受 `dimensions` 参数以投影到更小的维数。
DEFAULT_DIMS = 2048


class ArkEmbedderConfig(BaseEmbedderConfig):
    """Volcengine Ark embedder config.
    火山方舟嵌入器配置。
    """

    model: str | None = DEFAULT_MODEL
    embedding_dims: int = DEFAULT_DIMS
    ark_base_url: str | None = None
    timeout: float = 120.0


def _post_json(url: str, body: dict, api_key: str, timeout: float) -> dict:
    """POST a JSON body with urllib and decode the JSON reply (HTTPError / URLError propagate).
    用 urllib POST 一个 JSON 体并解码 JSON 回复（HTTPError / URLError 直接传播）。
    """
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = Request(url, data=data, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class ArkEmbedder(EmbedderBase):
    """Volcengine Ark multimodal embeddings provider.
    火山方舟多模态嵌入提供者。
    """

    config_class: ClassVar[type[BaseEmbedderConfig]] = ArkEmbedderConfig
    provider_name: ClassVar[str] = "ark"

    def __init__(self, config: ArkEmbedderConfig | dict | None = None) -> None:
        """Resolve the base url from config.ark_base_url / ARK_BASE_URL / the Ark default and the key from config / ARK_API_KEY.
        由 config.ark_base_url / ARK_BASE_URL / 方舟默认值解析基础 url，由 config / ARK_API_KEY 解析密钥。

        Input / 输入：
            config: Provider config dict / model, or None for the defaults.
            config：提供者配置字典 / 模型；为 None 时用默认值。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            None (no network call at construction).
            无（构造时不发起网络调用）。
        """
        super().__init__(config)
        if not self.config.model:
            self.config.model = DEFAULT_MODEL
        base_url = getattr(self.config, "ark_base_url", None) or os.getenv("ARK_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/")  # 已归一化的基础 url（无尾斜杠）
        self.api_key = self.config.api_key or os.getenv("ARK_API_KEY") or ""  # 调用时放入 Authorization 头

    def embed(self, text: str, memory_action: Literal["add", "search", "update"] | None = None) -> list[float]:
        """Embed one text.
        嵌入一段文本。

        Input / 输入：
            text: The text (newlines replaced by spaces, Mem0 idiom).
            text：文本（换行替换为空格，Mem0 惯例）。

            memory_action: Ignored.
            memory_action：忽略。

        Output / 输出：
            The `data.embedding` list of `POST /embeddings/multimodal`.
            `POST /embeddings/multimodal` 返回的 `data.embedding` 列表。

        State effect / 状态影响：
            One HTTP POST; the reply's `usage` feeds the embedder meter.
            一次 HTTP POST；回复的 `usage` 喂给嵌入器计量器。
        """
        cleaned = str(text).replace("\n", " ")
        # The multimodal endpoint takes one content-part list per call and returns a single vector, so a batch of
        # N texts costs N calls (EmbedderBase.embed_batch drives that loop).
        # 多模态端点每次调用接收一条内容部件列表并返回单个向量，故 N 段文本需 N 次调用（由 EmbedderBase.embed_batch 驱动）。
        body = {
            "model": self.config.model,
            "input": [{"type": "text", "text": cleaned}],
            "dimensions": self.dims,
        }
        response = _post_json(f"{self.base_url}{EMBEDDINGS_PATH}", body, self.api_key, float(self.config.timeout))
        # The reply's usage (prompt_tokens) feeds EmbedderBase's meter; None keeps the estimate path.
        # 回复的用量（prompt_tokens）喂给 EmbedderBase 的计量器；为 None 时走估算路径。
        self._last_usage = response.get("usage")
        data: Any = response.get("data")
        # `data` is {"embedding": [...]} for a single input; a list form is tolerated for forward compatibility.
        # 单条输入时 `data` 为 {"embedding": [...]}；列表形式一并兼容以便向前兼容。
        vector = data["embedding"] if isinstance(data, dict) else data[0]["embedding"]
        return [float(v) for v in vector]

    @property
    def dims(self) -> int:
        """config.embedding_dims (default 2048, the model's native size).
        config.embedding_dims（默认 2048，即模型的原始维数）。

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
        return int(self.config.embedding_dims or DEFAULT_DIMS)

    @property
    def identity(self) -> str:
        """'ark:<model>@<dims>'.
        'ark:<模型>@<维数>'。

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
        return f"ark:{self.config.model}@{self.dims}"


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DIMS",
    "DEFAULT_MODEL",
    "EMBEDDINGS_PATH",
    "ArkEmbedder",
    "ArkEmbedderConfig",
]
