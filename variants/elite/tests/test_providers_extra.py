"""W6 provider tests: deepseek, ollama (llm + embedder) and fastembed against fake SDKs / fake HTTP, fully offline.
W6 提供者测试：deepseek、ollama（llm + 嵌入器）与 fastembed，对照替身 SDK / 替身 HTTP，完全离线。
"""

# Purpose (EN): Pin the request bodies, response parsing, env ladders and DependencyError hints of the W6 providers.
# 目的（中）：固定 W6 提供者的请求体、响应解析、环境变量阶梯与 DependencyError 提示。

# Input (EN): sys.modules patches (fake `openai` / `fastembed`), a fake `urlopen`, monkeypatched env vars.
# 输入（中）：sys.modules 补丁（替身 `openai` / `fastembed`）、替身 `urlopen`、打补丁的环境变量。

# Output (EN): Assertions only.
# 输出（中）：仅断言。

# State effect (EN): None outside the patches restored by monkeypatch / patch.dict.
# 状态影响（中）：除 monkeypatch / patch.dict 恢复的补丁外无副作用。

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest import mock
from urllib.error import URLError

import pytest

from hms_elite.configs import prompts
from hms_elite.configs.registry import EXTRA_OF_PROVIDER, REGISTRY, supported
from hms_elite.embeddings import fastembed as fastembed_module
from hms_elite.embeddings import ollama as ollama_embed_module
from hms_elite.exceptions import ConfigError, DependencyError, EmbeddingError, LLMError
from hms_elite.llms import deepseek as deepseek_module
from hms_elite.llms import ollama as ollama_llm_module
from hms_elite.utils.factory import EmbedderFactory, LlmFactory

GUARDED = {"deepseek": "hms_elite.llms.deepseek", "fastembed": "hms_elite.embeddings.fastembed"}


class _FakeResponse:
    """A context-managed HTTP reply exposing `read()`.
    暴露 `read()` 的上下文管理式 HTTP 回复。
    """

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        """Return the raw payload.
        返回原始负载。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The bytes given at construction.
            构造时给定的字节。

        State effect / 状态影响：
            None.
            无。
        """
        return self.payload


class _FakeUrlopen:
    """A stand-in `urlopen` recording each Request and replaying queued JSON replies (or raising queued errors).
    记录每个 Request 并按队列回放 JSON 回复（或抛出队列中的错误）的替身 `urlopen`。
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "content_type": request.get_header("Content-type"),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(json.dumps(reply).encode("utf-8"))


def _fake_openai(calls: list[dict], *, tool_calls: list | None = None) -> types.ModuleType:
    """A stand-in `openai` module recording OpenAI(...) and chat.completions.create kwargs.
    记录 OpenAI(...) 与 chat.completions.create 参数的替身 `openai` 模块。
    """
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            calls.append({"api": "chat", **kwargs})
            message = types.SimpleNamespace(content='{"observations": []}', tool_calls=tool_calls)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        def __init__(self, **kwargs):
            calls.append({"api": "client", **kwargs})
            self.chat = types.SimpleNamespace(completions=_Completions())

    module.OpenAI = OpenAI
    return module


def _fake_fastembed(calls: list[dict], vector: list[float]) -> types.ModuleType:
    """A stand-in `fastembed` module whose TextEmbedding records its kwargs and yields one fixed vector per text.
    替身 `fastembed` 模块：其 TextEmbedding 记录构造参数，并为每段文本产出一个固定向量。
    """
    module = types.ModuleType("fastembed")

    class TextEmbedding:
        def __init__(self, **kwargs):
            calls.append({"api": "init", **kwargs})

        def embed(self, documents, **kwargs):
            calls.append({"api": "embed", "documents": list(documents), **kwargs})
            for _ in documents:
                yield list(vector)

    module.TextEmbedding = TextEmbedding
    return module


def _missing_sdk(provider: str, package: str, factory: Any) -> DependencyError:
    """Create `provider` with `package` absent from sys.modules and return the DependencyError raised.
    在 sys.modules 中缺失 `package` 的情况下创建 `provider`，返回抛出的 DependencyError。
    """
    with mock.patch.dict(sys.modules, {package: None}):
        sys.modules.pop(GUARDED[provider], None)
        with pytest.raises(DependencyError) as info:
            factory.create(provider)
    return info.value


# ---------------------------------------------------------------------------
# Registry entries and extras
# 注册表条目与 extras
# ---------------------------------------------------------------------------
def test_registry_and_extras_list_the_w6_providers():
    """REGISTRY / EXTRA_OF_PROVIDER / supported() / factories all expose deepseek, ollama and fastembed.
    REGISTRY / EXTRA_OF_PROVIDER / supported() / 工厂都暴露 deepseek、ollama 与 fastembed。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the registry tables and factory listings.
        对注册表与工厂列表的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert REGISTRY["llm"]["deepseek"] == (
        "hms_elite.llms.deepseek.DeepSeekLLM",
        "hms_elite.llms.deepseek.DeepSeekLlmConfig",
    )
    assert REGISTRY["llm"]["ollama"] == ("hms_elite.llms.ollama.OllamaLLM", "hms_elite.llms.ollama.OllamaLlmConfig")
    assert REGISTRY["embedder"]["ollama"] == (
        "hms_elite.embeddings.ollama.OllamaEmbedder",
        "hms_elite.embeddings.ollama.OllamaEmbedderConfig",
    )
    assert REGISTRY["embedder"]["fastembed"] == (
        "hms_elite.embeddings.fastembed.FastEmbedEmbedder",
        "hms_elite.embeddings.fastembed.FastEmbedEmbedderConfig",
    )
    assert {"deepseek", "ollama"} <= set(supported("llm")) and {"ollama", "fastembed"} <= set(supported("embedder"))
    assert supported("llm") == sorted(supported("llm")) and supported("embedder") == sorted(supported("embedder"))
    assert EXTRA_OF_PROVIDER["deepseek"] == "openai" and EXTRA_OF_PROVIDER["fastembed"] == "fastembed"
    assert EXTRA_OF_PROVIDER["ollama"] == "ollama"
    assert {"deepseek", "ollama"} <= set(LlmFactory.get_supported_providers())
    assert {"ollama", "fastembed"} <= set(EmbedderFactory.get_supported_providers())
    # 未知提供者的 CONFIG_002 提示里也列出新条目
    with pytest.raises(ConfigError) as info:  # the CONFIG_002 hint lists the new entries too
        LlmFactory.create("bogus")
    assert info.value.error_code == "CONFIG_002" and "deepseek" in info.value.details["supported"]


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible endpoint)
# DeepSeek（OpenAI 兼容端点）
# ---------------------------------------------------------------------------
def test_deepseek_missing_sdk_hint():
    """Without `openai` the deepseek provider raises DependencyError(DEPS_001) pointing at the openai extra.
    缺少 `openai` 时 deepseek 提供者抛出指向 openai extra 的 DependencyError(DEPS_001)。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the error code, details, suggestion and cause.
        对错误码、details、suggestion 与原因的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by patch.dict).
        临时替换 sys.modules 条目（由 patch.dict 恢复）。
    """
    error = _missing_sdk("deepseek", "openai", LlmFactory)
    assert error.error_code == "DEPS_001" and error.details == {"package": "openai", "extra": "openai"}
    assert error.suggestion == "pip install 'hms-elite[openai]'" and isinstance(error.__cause__, ImportError)


def test_deepseek_requests_and_thinking_toggle(monkeypatch):
    """DeepSeek uses api.deepseek.com / DEEPSEEK_API_KEY by default and sends `thinking` only through extra_body when set.
    DeepSeek 默认使用 api.deepseek.com / DEEPSEEK_API_KEY，仅在设置时通过 extra_body 发送 `thinking`。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules and env patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules 与环境变量）。

    Output / 输出：
        Assertions over the recorded OpenAI(...) and chat.completions.create kwargs.
        对记录的 OpenAI(...) 与 chat.completions.create 参数的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries and env vars (restored by monkeypatch).
        临时替换 sys.modules 条目与环境变量（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    calls: list[dict] = []
    monkeypatch.setattr(deepseek_module, "_IMPORT_ERROR", None)
    monkeypatch.setattr(deepseek_module, "OpenAI", _fake_openai(calls).OpenAI)

    llm = LlmFactory.create("deepseek", {"temperature": 0.2, "max_tokens": 77})
    assert llm.identity == "deepseek:deepseek-chat" and llm.config.thinking is None
    assert calls[0] == {"api": "client", "api_key": "env-key", "base_url": "https://api.deepseek.com"}
    out = llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u")
    sent = calls[-1]
    assert out == [] and sent["api"] == "chat" and sent["response_format"] == {"type": "json_object"}
    assert sent["model"] == "deepseek-chat" and sent["temperature"] == 0.2 and sent["max_tokens"] == 77
    assert sent["messages"][0]["role"] == "system" and "extra_body" not in sent and "tools" not in sent
    text = llm.generate_text("procedural_summary", messages=[{"role": "user", "content": "x"}])
    assert text == '{"observations": []}' and "response_format" not in calls[-1]
    # DeepSeek 对话模型无视觉能力：沿用基类的 None
    assert llm.describe_image("data:image/png;base64,AA==") is None  # no vision: the base default stays

    # 显式的 config.api_key / deepseek_base_url 覆盖环境变量
    explicit = LlmFactory.create(  # explicit config wins over the env ladder
        "deepseek", {"api_key": "k", "deepseek_base_url": "https://gw.example/v1", "thinking": "disabled"}
    )
    assert calls[-1] == {"api": "client", "api_key": "k", "base_url": "https://gw.example/v1"}
    explicit.generate_response([{"role": "user", "content": "x"}])
    assert calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}
    # 调用方的 extra_body 与开关合并，其他键保留
    explicit.generate_response(  # the caller's extra_body is merged, other keys survive
        [{"role": "user", "content": "x"}], extra_body={"thinking": {"budget": 1}, "foo": "bar"}
    )
    assert calls[-1]["extra_body"] == {"thinking": {"budget": 1, "type": "disabled"}, "foo": "bar"}

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://ark.example/v3")
    enabled = LlmFactory.create("deepseek", {"model": "deepseek-reasoner", "thinking": "enabled"})
    assert enabled.identity == "deepseek:deepseek-reasoner" and calls[-1]["base_url"] == "https://ark.example/v3"
    with_tools = enabled.generate_response(
        [{"role": "user", "content": "x"}], tools=[{"type": "function"}], tool_choice="auto"
    )
    assert with_tools == {"content": '{"observations": []}', "tool_calls": []}
    assert calls[-1]["tool_choice"] == "auto" and calls[-1]["extra_body"] == {"thinking": {"type": "enabled"}}

    with pytest.raises(ConfigError) as info:
        LlmFactory.create("deepseek", {"thinkng": "enabled"})
    assert info.value.error_code == "CONFIG_001" and "Did you mean 'thinking'?" in info.value.message
    with pytest.raises(ConfigError) as info:
        LlmFactory.create("deepseek", {"thinking": "maybe"})
    assert info.value.error_code == "CONFIG_001"


def test_deepseek_records_exact_usage_with_cache_fields(monkeypatch):
    """A DeepSeek reply's usage (prompt / completion / cache hit / cache miss) is metered exactly and kept in `last_usage`.
    DeepSeek 回复的 usage（提示 / 补全 / 缓存命中 / 缓存未命中）被精确计量并保留在 `last_usage`；无 usage 时回落到估算。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over the UsageMeter snapshot and `last_usage`.
        对 UsageMeter 快照与 `last_usage` 的断言。

    State effect / 状态影响：
        Temporarily replaces the openai module seen by the provider (restored by monkeypatch).
        临时替换提供者看到的 openai 模块（由 monkeypatch 恢复）。
    """
    usages: list[Any] = [
        types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=13, prompt_cache_hit_tokens=4, prompt_cache_miss_tokens=6
        ),
        None,
    ]
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            message = types.SimpleNamespace(content="B", tool_calls=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usages.pop(0))

    class OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    module.OpenAI = OpenAI
    monkeypatch.setattr(deepseek_module, "_IMPORT_ERROR", None)
    monkeypatch.setattr(deepseek_module, "OpenAI", module.OpenAI)
    llm = deepseek_module.DeepSeekLLM({"api_key": "k", "model": "deepseek-flash"})
    assert llm.last_usage is None

    assert llm.generate_response([{"role": "user", "content": "q"}]) == "B"
    assert llm.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_cache_hit_tokens": 4,
        "prompt_cache_miss_tokens": 6,
    }
    snap = llm.usage.snapshot()
    assert (snap.llm_calls, snap.input_tokens, snap.output_tokens, snap.estimated) == (1, 10, 3, False)
    assert llm._last_usage is None  # the meter consumed the slot / 计量器已消费该槽位

    # A dict-shaped usage and a pydantic-like model_dump both work; missing cache fields are simply absent.
    # 字典形式与带 model_dump 的 pydantic 形式都可用；缺失的缓存字段直接不出现。
    assert deepseek_module.usage_to_dict({"prompt_tokens": "7", "completion_tokens": 1}) == {
        "prompt_tokens": 7,
        "completion_tokens": 1,
    }
    dumped = types.SimpleNamespace(model_dump=lambda: {"prompt_tokens": 2, "completion_tokens": 0, "prompt_cache_hit_tokens": 2})
    assert deepseek_module.usage_to_dict(dumped) == {"prompt_tokens": 2, "completion_tokens": 0, "prompt_cache_hit_tokens": 2}
    assert deepseek_module.usage_to_dict(None) is None and deepseek_module.usage_to_dict({}) is None

    # Second reply carries no usage: the meter estimates and flags it.
    # 第二个回复没有 usage：计量器估算并打上标记。
    llm.generate_response([{"role": "user", "content": "q"}])
    assert llm.last_usage is None and llm.usage.snapshot().estimated is True


def test_deepseek_parses_tool_call_arguments(monkeypatch):
    """Tool-call arguments are JSON-decoded, and undecodable strings are kept under 'raw'.
    工具调用参数做 JSON 解码，无法解码的字符串保留在 'raw' 下。

    Input / 输入：
        monkeypatch: pytest monkeypatch (module attribute patching).
        monkeypatch：pytest monkeypatch（打补丁模块属性）。

    Output / 输出：
        Assertions over the parsed tool_calls.
        对解析后 tool_calls 的断言。

    State effect / 状态影响：
        Temporarily replaces module attributes (restored by monkeypatch).
        临时替换模块属性（由 monkeypatch 恢复）。
    """
    tool_calls = [
        types.SimpleNamespace(function=types.SimpleNamespace(name="f", arguments='{"a": 1}')),
        types.SimpleNamespace(function=types.SimpleNamespace(name="g", arguments="not json")),
    ]
    calls: list[dict] = []
    monkeypatch.setattr(deepseek_module, "_IMPORT_ERROR", None)
    monkeypatch.setattr(deepseek_module, "OpenAI", _fake_openai(calls, tool_calls=tool_calls).OpenAI)
    llm = deepseek_module.DeepSeekLLM({"api_key": "k"})
    out = llm.generate_response([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert out["tool_calls"] == [{"name": "f", "arguments": {"a": 1}}, {"name": "g", "arguments": {"raw": "not json"}}]


# ---------------------------------------------------------------------------
# Ollama LLM (local /api/chat)
# Ollama LLM（本地 /api/chat）
# ---------------------------------------------------------------------------
def test_ollama_llm_requests(monkeypatch):
    """The ollama llm POSTs `/api/chat` with stream=False, `format='json'` for JSON tasks and Mem0-shaped options.
    ollama llm 向 `/api/chat` 发送 stream=False 的 POST，JSON 任务带 `format='json'`，options 为 Mem0 形状。

    Input / 输入：
        monkeypatch: pytest monkeypatch (urlopen and env patching).
        monkeypatch：pytest monkeypatch（打补丁 urlopen 与环境变量）。

    Output / 输出：
        Assertions over the recorded requests and the parsed replies.
        对记录的请求与解析后回复的断言。

    State effect / 状态影响：
        Temporarily replaces `hms_elite.llms.ollama.urlopen` and env vars (restored by monkeypatch).
        临时替换 `hms_elite.llms.ollama.urlopen` 与环境变量（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    http = _FakeUrlopen(
        [
            {"message": {"role": "assistant", "content": '{"observations": []}'}},
            {"message": {"role": "assistant", "content": "```\nsummary\n```"}},
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "f", "arguments": {"a": 1}}},
                        {"function": {"name": "g", "arguments": '{"b": 2}'}},
                    ],
                }
            },
            URLError("connection refused"),
        ]
    )
    monkeypatch.setattr(ollama_llm_module, "urlopen", http)

    llm = LlmFactory.create("ollama", {"temperature": 0.4, "max_tokens": 55, "timeout": 9})
    assert llm.identity == "ollama:llama3.1" and llm.base_url == "http://localhost:11434"
    out = llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u")
    sent = http.requests[-1]
    assert out == [] and sent["url"] == "http://localhost:11434/api/chat" and sent["method"] == "POST"
    assert sent["content_type"] == "application/json" and sent["timeout"] == 9.0
    assert sent["body"] == {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": prompts.OBSERVATION_EXTRACTION_PROMPT},
            {"role": "user", "content": "u"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.4, "num_predict": 55, "top_p": 0.1, "top_k": 1},
    }
    text = llm.generate_text("procedural_summary", messages=[{"role": "user", "content": "x"}])
    assert text == "summary" and "format" not in http.requests[-1]["body"]
    with_tools = llm.generate_response([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert with_tools == {
        "content": "",
        "tool_calls": [{"name": "f", "arguments": {"a": 1}}, {"name": "g", "arguments": {"b": 2}}],
    }
    assert http.requests[-1]["body"]["tools"] == [{"type": "function"}] and "tool_choice" not in http.requests[-1]["body"]
    # 传输错误由 LLMBase 包装为 LLM_001 并保留原因
    with pytest.raises(LLMError) as info:  # transport errors become LLM_001 with the cause kept
        llm.generate_json("observation_extraction", system="s", user="u")
    assert info.value.error_code == "LLM_001" and isinstance(info.value.__cause__, URLError)
    assert info.value.details == {"task": "observation_extraction", "provider": "ollama:llama3.1"}

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.local:11434/")
    assert LlmFactory.create("ollama").base_url == "http://ollama.local:11434"
    custom = LlmFactory.create("ollama", {"ollama_base_url": "http://10.0.0.2:11434", "model": "qwen3"})
    assert custom.base_url == "http://10.0.0.2:11434" and custom.identity == "ollama:qwen3"
    with pytest.raises(ConfigError) as info:
        LlmFactory.create("ollama", {"ollama_base_ur": "x"})
    assert "Did you mean 'ollama_base_url'?" in info.value.message


# ---------------------------------------------------------------------------
# Ollama embedder
# Ollama 嵌入器
# ---------------------------------------------------------------------------
def test_ollama_embedder_infers_and_caches_dims(monkeypatch):
    """The ollama embedder POSTs `/api/embeddings {'model', 'prompt'}` and learns dims from the first reply once.
    ollama 嵌入器向 `/api/embeddings {'model', 'prompt'}` 发送 POST，并仅从首个回复学习一次维数。

    Input / 输入：
        monkeypatch: pytest monkeypatch (urlopen and env patching).
        monkeypatch：pytest monkeypatch（打补丁 urlopen 与环境变量）。

    Output / 输出：
        Assertions over the recorded requests, vectors, dims caching and error wrapping.
        对记录的请求、向量、维数缓存与错误包装的断言。

    State effect / 状态影响：
        Temporarily replaces `hms_elite.embeddings.ollama.urlopen` and env vars (restored by monkeypatch).
        临时替换 `hms_elite.embeddings.ollama.urlopen` 与环境变量（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    http = _FakeUrlopen(
        [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [1, 2, 3]},
            {"embedding": [0.5, 0.5, 0.5]},
            URLError("connection refused"),
        ]
    )
    monkeypatch.setattr(ollama_embed_module, "urlopen", http)

    embedder = EmbedderFactory.create("ollama", {"timeout": 3})
    assert embedder.config.model == "nomic-embed-text" and embedder._dims is None
    assert embedder.embed("a\nb") == [0.1, 0.2, 0.3]
    assert http.requests[-1]["url"] == "http://localhost:11434/api/embeddings" and http.requests[-1]["timeout"] == 3.0
    assert http.requests[-1]["body"] == {"model": "nomic-embed-text", "prompt": "a b"}
    # 维数来自首个回复并缓存：identity / dims 不再发请求
    assert embedder.dims == 3 and embedder.identity == "ollama:nomic-embed-text@3"  # cached: no extra request
    assert len(http.requests) == 1
    assert embedder.embed_batch(["x"]) == [[1.0, 2.0, 3.0]] and embedder.dims == 3
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_batch(["x", "y"])
    assert info.value.error_code == "EMBED_001" and info.value.details["index"] == 1
    assert isinstance(info.value.__cause__, URLError)

    # 尚未嵌入时读取 dims 会探测一次
    http.replies.append({"embedding": [0.0] * 4})  # reading dims before any embed probes once
    fresh = EmbedderFactory.create("ollama", {"ollama_base_url": "http://box:11434/"})
    assert fresh.dims == 4 and http.requests[-1]["url"] == "http://box:11434/api/embeddings"
    assert http.requests[-1]["body"]["prompt"] == ollama_embed_module.DIMS_PROBE_TEXT
    assert fresh.dims == 4 and http.requests[-1]["body"]["prompt"] == ollama_embed_module.DIMS_PROBE_TEXT

    # 配置了 embedding_dims 则不探测
    configured = EmbedderFactory.create("ollama", {"embedding_dims": 768})  # configured dims skip the probe
    before = len(http.requests)
    assert configured.dims == 768 and configured.identity == "ollama:nomic-embed-text@768"
    assert len(http.requests) == before
    with pytest.raises(ConfigError):
        EmbedderFactory.create("ollama", {"api_ke": "x"})


# ---------------------------------------------------------------------------
# fastembed (local ONNX)
# fastembed（本地 ONNX）
# ---------------------------------------------------------------------------
def test_fastembed_missing_sdk_hint():
    """Without `fastembed` the embedder raises DependencyError(DEPS_001) pointing at the fastembed extra.
    缺少 `fastembed` 时嵌入器抛出指向 fastembed extra 的 DependencyError(DEPS_001)。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the error code, details, suggestion and cause.
        对错误码、details、suggestion 与原因的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by patch.dict).
        临时替换 sys.modules 条目（由 patch.dict 恢复）。
    """
    error = _missing_sdk("fastembed", "fastembed", EmbedderFactory)
    assert error.error_code == "DEPS_001" and error.details == {"package": "fastembed", "extra": "fastembed"}
    assert error.suggestion == "pip install 'hms-elite[fastembed]'" and isinstance(error.__cause__, ImportError)


def test_fastembed_embeds_locally(monkeypatch):
    """fastembed builds TextEmbedding(model_name[, cache_dir, threads]) and returns 384-dim float vectors.
    fastembed 构造 TextEmbedding(model_name[, cache_dir, threads]) 并返回 384 维浮点向量。

    Input / 输入：
        monkeypatch: pytest monkeypatch (module attribute patching).
        monkeypatch：pytest monkeypatch（打补丁模块属性）。

    Output / 输出：
        Assertions over the recorded TextEmbedding kwargs and the embedded vectors.
        对记录的 TextEmbedding 参数与嵌入向量的断言。

    State effect / 状态影响：
        Temporarily replaces module attributes (restored by monkeypatch).
        临时替换模块属性（由 monkeypatch 恢复）。
    """
    calls: list[dict] = []
    vector = [0.25] * 384
    monkeypatch.setattr(fastembed_module, "_IMPORT_ERROR", None)
    monkeypatch.setattr(fastembed_module, "TextEmbedding", _fake_fastembed(calls, vector).TextEmbedding)

    embedder = EmbedderFactory.create("fastembed")
    assert calls == [{"api": "init", "model_name": "BAAI/bge-small-en-v1.5"}]
    assert embedder.dims == 384 and embedder.identity == "fastembed:BAAI/bge-small-en-v1.5@384"
    out = embedder.embed("a\nb")
    assert out == vector and all(isinstance(v, float) for v in out)
    assert calls[-1] == {"api": "embed", "documents": ["a b"]}
    assert embedder.embed_batch(["x", "y"]) == [vector, vector]

    tuned = EmbedderFactory.create(
        "fastembed", {"model": "BAAI/bge-base-en-v1.5", "embedding_dims": 768, "cache_dir": "/models", "threads": 2}
    )
    assert calls[-1] == {"api": "init", "model_name": "BAAI/bge-base-en-v1.5", "cache_dir": "/models", "threads": 2}
    assert tuned.dims == 768 and tuned.identity == "fastembed:BAAI/bge-base-en-v1.5@768"
    with pytest.raises(ConfigError) as info:
        EmbedderFactory.create("fastembed", {"cache_di": "/models"})
    assert "Did you mean 'cache_dir'?" in info.value.message
