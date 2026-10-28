"""Provider registry: names → dotted class paths; the single truth shared by envelopes and factories.
提供者注册表：名称 → 点分类路径；信封校验与工厂共享的唯一真相。
"""

# Purpose (EN): Map provider kinds/names (llm, embedder, index, reranker, entity_extractor) to implementation and
#               config classes without importing them.
# 目的（中）：在不导入的前提下，将提供者种类/名称（llm、embedder、index、reranker、entity_extractor）映射到实现类与配置类。

# Input (EN): Nothing at import time; register_provider (utils/factory) mutates REGISTRY in place.
# 输入（中）：导入时无输入；register_provider（utils/factory）原地修改 REGISTRY。

# Output (EN): REGISTRY, EXTRA_OF_PROVIDER, supported().
# 输出（中）：REGISTRY、EXTRA_OF_PROVIDER、supported()。

# State effect (EN): None at import; REGISTRY is module-level mutable state shared by design.
# 状态影响（中）：导入时无；REGISTRY 是刻意共享的模块级可变状态。

from __future__ import annotations

REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "llm": {
        "rule": ("hms_elite.llms.rule.RuleLLM", "hms_elite.llms.rule.RuleLlmConfig"),
        "scripted": ("hms_elite.llms.scripted.ScriptedLLM", "hms_elite.llms.scripted.ScriptedLlmConfig"),
        "openai": ("hms_elite.llms.openai.OpenAILLM", "hms_elite.llms.openai.OpenAILlmConfig"),
        "anthropic": ("hms_elite.llms.anthropic.AnthropicLLM", "hms_elite.llms.anthropic.AnthropicLlmConfig"),
        "deepseek": ("hms_elite.llms.deepseek.DeepSeekLLM", "hms_elite.llms.deepseek.DeepSeekLlmConfig"),
        "ollama": ("hms_elite.llms.ollama.OllamaLLM", "hms_elite.llms.ollama.OllamaLlmConfig"),
    },
    "embedder": {
        "hash": ("hms_elite.embeddings.hash.HashEmbedder", "hms_elite.embeddings.hash.HashEmbedderConfig"),
        "openai": ("hms_elite.embeddings.openai.OpenAIEmbedder", "hms_elite.embeddings.openai.OpenAIEmbedderConfig"),
        "ollama": ("hms_elite.embeddings.ollama.OllamaEmbedder", "hms_elite.embeddings.ollama.OllamaEmbedderConfig"),
        "fastembed": (
            "hms_elite.embeddings.fastembed.FastEmbedEmbedder",
            "hms_elite.embeddings.fastembed.FastEmbedEmbedderConfig",
        ),
        "ark": ("hms_elite.embeddings.ark.ArkEmbedder", "hms_elite.embeddings.ark.ArkEmbedderConfig"),
    },
    "index": {
        "sqlite": ("hms_elite.index.sqlite.SQLiteIndex", "hms_elite.index.sqlite.SQLiteIndexConfig"),
        "memory": ("hms_elite.index.memory.MemoryIndex", "hms_elite.index.memory.MemoryIndexConfig"),
        "milvus": ("hms_elite.index.milvus.MilvusIndex", "hms_elite.index.milvus.MilvusIndexConfig"),
    },
    "reranker": {
        "overlap": ("hms_elite.reranker.overlap.OverlapReranker", "hms_elite.reranker.overlap.OverlapRerankerConfig"),
        "llm": ("hms_elite.reranker.llm.LLMReranker", "hms_elite.reranker.llm.LLMRerankerConfig"),
    },
    # Entity extractors are selected by IngestConfig.entity_extractor (a plain name, not an envelope).
    # 实体抽取器由 IngestConfig.entity_extractor（纯名称，而非信封）选择。
    "entity_extractor": {
        "rule": ("hms_elite.entities.rule.RuleEntityExtractor", "hms_elite.entities.rule.RuleEntityExtractorConfig"),
        "spacy": ("hms_elite.entities.spacy.SpacyEntityExtractor", "hms_elite.entities.spacy.SpacyEntityExtractorConfig"),
    },
}
# pip extra hint per provider name (deepseek rides on the openai SDK; ollama and ark are stdlib-only, their extras
# are empty).
# 每个 provider 名称对应的 pip extra 提示（deepseek 复用 openai SDK；ollama 与 ark 仅用标准库，其 extra 为空）。
EXTRA_OF_PROVIDER = {
    "openai": "openai",
    "anthropic": "anthropic",
    "milvus": "milvus",
    "deepseek": "openai",
    "ollama": "ollama",
    "fastembed": "fastembed",
    "spacy": "spacy",
    "ark": "ark",
}


def supported(kind: str) -> list[str]:
    """Sorted provider names registered for a kind.
    某种类下已注册的提供者名（排序）。

    Input / 输入：
        kind: 'llm' | 'embedder' | 'index' | 'reranker' | 'entity_extractor'.
        kind：'llm' | 'embedder' | 'index' | 'reranker' | 'entity_extractor'。

    Output / 输出：
        Sorted list of names; unknown kind raises KeyError.
        排序后的名称列表；未知种类抛出 KeyError。

    State effect / 状态影响：
        None.
        无。
    """
    return sorted(REGISTRY[kind])


__all__ = ["EXTRA_OF_PROVIDER", "REGISTRY", "supported"]
