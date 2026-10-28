"""Tests for hms_elite.telemetry and the provider metering wrappers (usage meters, request traces, token capture).
hms_elite.telemetry 与提供者计量包装层的测试（用量计量器、请求追踪、token 捕获）。
"""

# Purpose (EN): Pin the W2 metering contract: estimated vs exact tokens, stage names, thread safety, usage_of.
# 目的（中）：钉住 W2 计量契约：估算与精确 token、阶段名、线程安全、usage_of。

# Input (EN): Rule / scripted / hash providers, fake `openai` and `anthropic` SDK modules injected into sys.modules.
# 输入（中）：规则 / 脚本化 / 哈希提供者，注入 sys.modules 的替身 `openai` 与 `anthropic` SDK 模块。

# Output (EN): Assertions over Usage snapshots and RequestTrace stages.
# 输出（中）：对 Usage 快照与 RequestTrace 阶段的断言。

# State effect (EN): None beyond temporary sys.modules patches (restored by patch.dict).
# 状态影响（中）：除临时的 sys.modules 补丁（由 patch.dict 恢复）外无。

from __future__ import annotations

import sys
import threading
import types
from typing import Any
from unittest.mock import patch

import pytest

from hms_elite.configs import prompts
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.exceptions import EmbeddingError, LLMError
from hms_elite.llms.base import LLMBase
from hms_elite.llms.rule import RuleLLM
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.telemetry import (
    STAGE_EMBED,
    STAGE_EMBED_BATCH,
    STAGE_LLM_GENERATE,
    RequestTrace,
    StageRecord,
    Usage,
    UsageMeter,
    current_trace,
    estimate_completion_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    stage,
    trace,
    usage_of,
    usage_tokens,
)

# Provider modules whose SDK import guard runs at import time; they are dropped so a fake SDK is picked up.
# 在导入时执行 SDK 导入守卫的提供者模块；先移除它们以便替身 SDK 被采用。
GUARDED = ("hms_elite.llms.openai", "hms_elite.llms.anthropic", "hms_elite.embeddings.openai")
MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi to 张三."}]


def _fake_openai(*, prompt: int, completion: int, embed_prompt: int | None) -> types.ModuleType:
    """A stand-in `openai` module whose responses carry a `usage` object.
    响应带 `usage` 对象的替身 `openai` 模块。
    """
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            message = types.SimpleNamespace(content="hello", tool_calls=None)
            usage = types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)

    class _Embeddings:
        def create(self, **kwargs):
            usage = None if embed_prompt is None else types.SimpleNamespace(prompt_tokens=embed_prompt)
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.5, 0.5, 0.5, 0.5])], usage=usage)

    class OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())
            self.embeddings = _Embeddings()

    module.OpenAI = OpenAI
    return module


def _fake_anthropic(*, input_tokens: int, output_tokens: int) -> types.ModuleType:
    """A stand-in `anthropic` module whose replies carry a `usage` object.
    回复带 `usage` 对象的替身 `anthropic` 模块。
    """
    module = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="hello")], usage=usage)

    class Anthropic:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    module.Anthropic = Anthropic
    return module


def _patched_modules(**fakes: types.ModuleType) -> dict[str, Any]:
    """sys.modules overlay: the fake SDKs plus the guarded provider modules removed so their guards re-run.
    sys.modules 覆盖层：替身 SDK，外加移除受守卫的提供者模块使其守卫重新执行。
    """
    overlay = dict(sys.modules)
    for name in GUARDED:
        overlay.pop(name, None)
    overlay.update(fakes)
    return overlay


def test_usage_meter_accumulates_and_resets():
    """add_llm / add_embedding sum into a detached snapshot; reset zeroes; merge adds two snapshots.
    add_llm / add_embedding 累加进分离快照；reset 归零；merge 相加两个快照。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over Usage fields.
        对 Usage 字段的断言。

    State effect / 状态影响：
        None.
        无。
    """
    meter = UsageMeter()
    assert meter.snapshot() == Usage()
    meter.add_llm(10, 5, estimated=False, duration_ms=1.5)
    meter.add_llm(3, 2, estimated=True, duration_ms=0.5)
    meter.add_embedding(4, 2.0)
    snap = meter.snapshot()
    assert (snap.llm_calls, snap.input_tokens, snap.output_tokens, snap.total_tokens) == (2, 13, 7, 20)
    assert snap.estimated is True and (snap.embedding_calls, snap.embedded_texts) == (1, 4)
    assert snap.duration_ms == pytest.approx(4.0)
    # The snapshot is detached from later adds; to_dict mirrors every field.
    # 快照与之后的累加分离；to_dict 映射每个字段。
    meter.add_llm(1, 1)
    assert snap.llm_calls == 2 and meter.snapshot().llm_calls == 3
    assert set(snap.to_dict()) == {
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated",
        "embedding_calls",
        "embedded_texts",
        "duration_ms",
    }
    merged = snap.merge(Usage(llm_calls=1, input_tokens=1, total_tokens=1))
    assert (merged.llm_calls, merged.input_tokens, merged.total_tokens, merged.estimated) == (3, 14, 21, True)
    meter.reset()
    assert meter.snapshot() == Usage()


def test_token_helpers_reuse_memory_utils_estimate():
    """estimate_tokens is memory.utils' formula; message / completion estimates and usage_tokens follow Lite's names.
    estimate_tokens 即 memory.utils 的公式；消息 / 补全估算与 usage_tokens 沿用 Lite 的字段名。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the helper outputs.
        对辅助函数输出的断言。

    State effect / 状态影响：
        None.
        无。
    """
    from hms_elite.memory.utils import estimate_tokens as original

    assert estimate_tokens is original and estimate_tokens("abcd") == (4 + 3) // 4 + 8
    assert estimate_messages_tokens(MESSAGES) == sum(estimate_tokens(m["content"]) for m in MESSAGES)
    parts = [{"role": "user", "content": [{"type": "text", "text": "ab"}, "cd"]}, {"role": "user", "content": None}]
    assert estimate_messages_tokens(parts) == estimate_tokens("ab\ncd") + estimate_tokens("")
    assert estimate_completion_tokens("hello") == estimate_tokens("hello")
    tool_reply = {"content": None, "tool_calls": [{"name": "f", "arguments": {"a": 1}}]}
    assert estimate_completion_tokens(tool_reply) > estimate_tokens("")
    assert usage_tokens(types.SimpleNamespace(prompt_tokens=3, completion_tokens=4)) == (3, 4)
    assert usage_tokens({"input_tokens": 5, "output_tokens": 6}) == (5, 6)
    assert usage_tokens(types.SimpleNamespace(prompt_tokens=3)) == (3, 0)
    assert usage_tokens(None) is None and usage_tokens(types.SimpleNamespace()) is None


def test_rule_llm_counts_estimated_tokens(rule_llm):
    """The rule provider has no SDK usage, so every call is metered with estimate_tokens and estimated=True.
    规则提供者没有 SDK 用量，因此每次调用都用 estimate_tokens 计量且 estimated=True。

    Input / 输入：
        rule_llm: RuleLLM fixture.
        rule_llm：RuleLLM 夹具。

    Output / 输出：
        Assertions over the provider's usage snapshot.
        对提供者用量快照的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert isinstance(rule_llm.usage, UsageMeter) and rule_llm.usage.snapshot() == Usage()
    system = prompts.OBSERVATION_EXTRACTION_PROMPT
    user = "I moved to Berlin last week and I love espresso."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = rule_llm.generate_response(messages, response_format={"type": "json_object"})
    snap = rule_llm.usage.snapshot()
    assert snap.llm_calls == 1 and snap.estimated is True
    assert snap.input_tokens == estimate_messages_tokens(messages)
    assert snap.output_tokens == estimate_completion_tokens(raw) == estimate_tokens(raw)
    assert snap.total_tokens == snap.input_tokens + snap.output_tokens and snap.duration_ms >= 0.0
    # generate_json goes through the same metered primitive; embedding counters stay untouched.
    # generate_json 走同一个被计量的原语；嵌入计数器保持不变。
    rule_llm.generate_json("observation_extraction", system=system, user=user)
    snap = rule_llm.usage.snapshot()
    assert snap.llm_calls == 2 and snap.input_tokens == 2 * estimate_messages_tokens(messages)
    assert (snap.embedding_calls, snap.embedded_texts) == (0, 0)
    # A failing call (unknown task marker) is still one attempt: prompt estimated, no completion.
    # 失败的调用（未知任务标记）仍算一次尝试：提示词估算、无补全。
    with pytest.raises(LLMError):
        rule_llm.generate_response([{"role": "user", "content": "no marker"}])
    snap = rule_llm.usage.snapshot()
    assert snap.llm_calls == 3 and snap.output_tokens == 2 * estimate_tokens(raw)


def test_scripted_llm_metering_keeps_signature_and_return_value(scripted_llm):
    """The wrapper leaves generate_response's return value and recorded calls untouched; dict replies are estimated too.
    包装层不改变 generate_response 的返回值与记录的调用；字典回复同样按估算计量。

    Input / 输入：
        scripted_llm: ScriptedLLM factory.
        scripted_llm：ScriptedLLM 工厂。

    Output / 输出：
        Assertions over the return value, `calls` and the usage snapshot.
        对返回值、`calls` 与用量快照的断言。

    State effect / 状态影响：
        None.
        无。
    """
    llm = scripted_llm(["plain text", {"facts": ["a", "b"]}])
    assert llm.generate_response(MESSAGES) == "plain text"
    second = llm.generate_response(MESSAGES, tools=[{"type": "function"}], tool_choice="auto")
    assert second == '{"facts": ["a", "b"]}'
    assert len(llm.calls) == 2 and llm.calls[1]["messages"] == MESSAGES
    snap = llm.usage.snapshot()
    assert snap.llm_calls == 2 and snap.estimated is True
    assert snap.input_tokens == 2 * estimate_messages_tokens(MESSAGES)
    assert snap.output_tokens == estimate_tokens("plain text") + estimate_tokens(second)
    # The wrapped method still reads as the subclass's own function (docstring, module, signature).
    # 被包装的方法仍呈现为子类自己的函数（文档字符串、模块、签名）。
    func = ScriptedLLM.__dict__["generate_response"]
    assert func.__module__ == "hms_elite.llms.scripted" and "Produce" in (func.__doc__ or "")
    assert getattr(func, "__wrapped__", None) is not None

    class Shouting(ScriptedLLM):
        def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
            """Upper-case the scripted reply.
            把脚本回复转成大写。
            """
            return str(super().generate_response(messages, tools, tool_choice, **kwargs)).upper()

    # An override delegating to super() is metered once, with the override's return value.
    # 委托给 super() 的覆盖方法只计量一次，且按覆盖方法的返回值计量。
    loud = Shouting({"responses": ["quiet"]})
    with trace("t") as tr:
        assert loud.generate_response(MESSAGES) == "QUIET"
    assert loud.usage.snapshot().llm_calls == 1 and len(tr.stages) == 1
    assert tr.stages[0].output_tokens == estimate_tokens("QUIET")


def test_openai_llm_records_exact_usage():
    """With a fake `openai` SDK, prompt_tokens / completion_tokens are recorded exactly and estimated=False.
    使用替身 `openai` SDK 时，prompt_tokens / completion_tokens 被精确记录且 estimated=False。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the usage snapshot and the trace stage.
        对用量快照与追踪阶段的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by patch.dict).
        临时替换 sys.modules 条目（由 patch.dict 恢复）。
    """
    fake = _fake_openai(prompt=11, completion=5, embed_prompt=4)
    with patch.dict(sys.modules, _patched_modules(openai=fake), clear=True):
        from hms_elite.embeddings.openai import OpenAIEmbedder
        from hms_elite.llms.openai import OpenAILLM

        llm = OpenAILLM({"api_key": "k"})
        with trace("add", request_id="req-1") as tr:
            assert llm.generate_response(MESSAGES, response_format={"type": "json_object"}) == "hello"
            assert llm.generate_json("observation_extraction", system="s", user="u") == []
        snap = llm.usage.snapshot()
        assert (snap.llm_calls, snap.input_tokens, snap.output_tokens, snap.total_tokens) == (2, 22, 10, 32)
        assert snap.estimated is False and llm._last_usage is None
        assert tr.request_id == "req-1" and [s.operation for s in tr.stages] == [STAGE_LLM_GENERATE] * 2
        assert (tr.stages[0].input_tokens, tr.stages[0].output_tokens, tr.stages[0].total_tokens) == (11, 5, 16)
        assert tr.stages[0].estimated is False and tr.stages[0].provider_calls == 1
        assert (tr.input_tokens, tr.output_tokens, tr.total_tokens, tr.estimated) == (22, 10, 32, False)
        # The embedder records the provider's prompt_tokens in the stage and counts texts in the meter.
        # 嵌入器把提供者的 prompt_tokens 记入阶段，并在计量器中统计文本数。
        embedder = OpenAIEmbedder({"embedding_dims": 4, "api_key": "k"})
        with trace("embed") as etr:
            embedder.embed("a\nb")
            embedder.embed_batch(["x", "y", "z"])
        esnap = embedder.usage.snapshot()
        assert (esnap.embedding_calls, esnap.embedded_texts, esnap.llm_calls) == (2, 4, 0)
        assert [s.operation for s in etr.stages] == [STAGE_EMBED, STAGE_EMBED_BATCH]
        assert etr.stages[0].input_tokens == 4 and etr.stages[0].estimated is False
        # embed_batch runs the sequential base implementation: its stage sums the per-item exact counts.
        # embed_batch 走基类的顺序实现：其阶段汇总逐项精确计数。
        assert etr.stages[1].estimated is False and etr.stages[1].input_tokens == 12
    # Without a provider usage object the batch stage falls back to the estimate.
    # 提供者没有用量对象时，批阶段回落到估算值。
    fake = _fake_openai(prompt=1, completion=1, embed_prompt=None)
    with patch.dict(sys.modules, _patched_modules(openai=fake), clear=True):
        from hms_elite.embeddings.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder({"embedding_dims": 4, "api_key": "k"})
        with trace("embed") as etr:
            embedder.embed_batch(["x", "y"])
        assert etr.stages[0].estimated is True and etr.stages[0].input_tokens == 2 * estimate_tokens("x")
        assert embedder.usage.snapshot().embedded_texts == 2


def test_anthropic_llm_records_exact_usage():
    """With a fake `anthropic` SDK, input_tokens / output_tokens are recorded exactly and estimated=False.
    使用替身 `anthropic` SDK 时，input_tokens / output_tokens 被精确记录且 estimated=False。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the usage snapshot.
        对用量快照的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by patch.dict).
        临时替换 sys.modules 条目（由 patch.dict 恢复）。
    """
    fake = _fake_anthropic(input_tokens=7, output_tokens=3)
    with patch.dict(sys.modules, _patched_modules(anthropic=fake), clear=True):
        from hms_elite.llms.anthropic import AnthropicLLM

        llm = AnthropicLLM({"api_key": "k"})
        assert llm.generate_response(MESSAGES) == "hello"
        assert llm.generate_text("procedural_summary", messages=MESSAGES) == "hello"
        snap = llm.usage.snapshot()
        assert (snap.llm_calls, snap.input_tokens, snap.output_tokens, snap.total_tokens) == (2, 14, 6, 20)
        assert snap.estimated is False


def test_trace_collects_llm_and_embedding_stages(rule_llm, hash_embedder):
    """One traced block records the llm and embedding stages in call order; outside a trace nothing is recorded.
    一个被追踪的块按调用顺序记录 llm 与 embedding 阶段；追踪之外不记录任何内容。

    Input / 输入：
        rule_llm: RuleLLM fixture.
        rule_llm：RuleLLM 夹具。

        hash_embedder: HashEmbedder fixture.
        hash_embedder：HashEmbedder 夹具。

    Output / 输出：
        Assertions over RequestTrace / StageRecord.
        对 RequestTrace / StageRecord 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert current_trace() is None
    with stage("noop") as record:
        assert record is None
    system = prompts.OBSERVATION_EXTRACTION_PROMPT
    with trace("add") as tr:
        assert current_trace() is tr and isinstance(tr, RequestTrace) and len(tr.request_id) == 32
        rule_llm.generate_json("observation_extraction", system=system, user="I like tea.")
        hash_embedder.embed("I like tea.", "add")
        hash_embedder.embed_batch(["a", "b", "c"], "add")
        with stage("custom") as record:
            assert isinstance(record, StageRecord) and record.provider_calls == 0
        with trace("inner") as inner:
            assert current_trace() is inner
            hash_embedder.embed("x")
        assert current_trace() is tr and inner.stages[0].operation == STAGE_EMBED
    assert current_trace() is None
    names = [s.operation for s in tr.stages]
    assert names == [STAGE_LLM_GENERATE, STAGE_EMBED, STAGE_EMBED_BATCH, "custom"]
    assert all(s.name == s.operation and s.duration_ms >= 0.0 for s in tr.stages)
    assert tr.stages[0].estimated is True and tr.stages[0].output_tokens > 0
    assert tr.stages[2].input_tokens == sum(estimate_tokens(t) for t in ["a", "b", "c"])
    assert tr.stages[2].total_tokens == tr.stages[2].input_tokens and tr.stages[2].output_tokens is None
    assert tr.duration_ms >= sum(s.duration_ms for s in tr.stages) * 0.5 and tr.error is None
    # embed_batch is one call covering three texts: the per-item embed calls inside it are not double counted.
    # embed_batch 是覆盖三段文本的一次调用：其内部逐条 embed 不被重复计数。
    esnap = hash_embedder.usage.snapshot()
    assert (esnap.embedding_calls, esnap.embedded_texts) == (3, 5)
    # Errors inside the block are captured on the trace and re-raised.
    # 块内错误被记录到追踪并重新抛出。
    with pytest.raises(ValueError):
        with trace("boom") as failed:
            raise ValueError("x")
    assert failed.error == "ValueError: x" and current_trace() is None


def test_trace_to_dict_splits_llm_and_embedding_tokens(rule_llm: RuleLLM, hash_embedder: HashEmbedder) -> None:
    """RequestTrace.to_dict() separates LLM input/output tokens from embedding tokens and lists every stage.
    RequestTrace.to_dict() 把 LLM 输入/输出 token 与嵌入 token 分开，并列出每个阶段。

    Input / 输入：
        rule_llm: RuleLLM fixture (offline, so its tokens are marked estimated).
        rule_llm：RuleLLM 夹具（离线，因此其 token 被标为估算值）。

        hash_embedder: HashEmbedder fixture.
        hash_embedder：HashEmbedder 夹具。

    Output / 输出：
        Assertions over the serialized report.
        对序列化报告的断言。

    State effect / 状态影响：
        None.
        无。
    """
    with trace("add") as tr:
        rule_llm.generate_json(
            "observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="I like tea."
        )
        hash_embedder.embed_batch(["a", "b"], "add")
        # A plain stage carries no tokens, so it must not disturb any of the split counters.
        # 普通 stage 不含 token，因此不得干扰任何拆分数。
        with stage("custom"):
            pass
    report = tr.to_dict()

    # The key set and its order are the response contract the service face mirrors.
    # 键集合及其顺序是服务面镜像的响应契约。
    assert list(report) == [
        "request_id",
        "llm_input_tokens",
        "llm_output_tokens",
        "embedding_tokens",
        "total_tokens",
        "estimated",
        "stages",
    ]  # fmt: skip
    assert report["request_id"] == tr.request_id
    # Each split counts only its own stage kind, so an LLM price and an embedding price can be applied apart.
    # 每个拆分数只统计自己那类阶段，因此 LLM 单价与嵌入单价可以分别套用。
    assert report["llm_input_tokens"] == tr.stages[0].input_tokens
    assert report["llm_output_tokens"] == tr.stages[0].output_tokens
    assert report["embedding_tokens"] == tr.stages[1].input_tokens
    assert report["llm_output_tokens"] > 0 and report["embedding_tokens"] > 0
    assert report["total_tokens"] == (
        report["llm_input_tokens"] + report["llm_output_tokens"] + report["embedding_tokens"]
    )
    # One estimated stage makes the whole report estimated: a cost figure built on it cannot be fully trusted.
    # 只要有一个阶段是估算值，整份报告即为估算：据此算出的成本不能完全采信。
    assert report["estimated"] is True
    assert [stage_report["operation"] for stage_report in report["stages"]] == [
        STAGE_LLM_GENERATE,
        STAGE_EMBED_BATCH,
        "custom",
    ]  # fmt: skip
    assert report["stages"][0]["provider_calls"] == 1 and report["stages"][0]["error"] is None
    # An untokened stage keeps its None fields None rather than reporting a misleading zero.
    # 无 token 的阶段其字段保持 None，而不是报告一个会误导人的 0。
    assert report["stages"][2]["input_tokens"] is None and report["stages"][2]["provider_calls"] == 0

    # A trace that spent nothing still serializes to the same shape, with every counter at zero.
    # 什么都没消耗的追踪仍序列化为同一形状，所有计数器为 0。
    with trace("noop") as empty:
        pass
    empty_report = empty.to_dict()
    assert empty_report["stages"] == [] and empty_report["estimated"] is False
    assert (empty_report["llm_input_tokens"], empty_report["llm_output_tokens"]) == (0, 0)
    assert empty_report["embedding_tokens"] == 0 and empty_report["total_tokens"] == 0


def test_embedding_failures_are_counted_and_wrapped():
    """A failing embed is one metered attempt with an error stage; embed_batch still wraps it as EMBED_001.
    失败的 embed 是一次带错误阶段的计量尝试；embed_batch 仍把它包装为 EMBED_001。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the meter and the trace.
        对计量器与追踪的断言。

    State effect / 状态影响：
        None.
        无。
    """

    class Broken(HashEmbedder):
        def embed(self, text, memory_action=None):
            """Fail on the text 'bad', otherwise hash-embed.
            文本为 'bad' 时失败，否则做哈希嵌入。
            """
            if text == "bad":
                raise RuntimeError("down")
            return super().embed(text, memory_action)

    embedder = Broken({"embedding_dims": 8})
    assert embedder.embed("ok")[0] == pytest.approx(embedder.embed("ok")[0])
    with trace("t") as tr:
        with pytest.raises(EmbeddingError) as info:
            embedder.embed_batch(["ok", "bad"])
    assert info.value.error_code == "EMBED_001" and info.value.details["index"] == 1
    snap = embedder.usage.snapshot()
    assert (snap.embedding_calls, snap.embedded_texts) == (3, 4)
    assert tr.stages[-1].operation == STAGE_EMBED_BATCH and tr.stages[-1].estimated is True
    assert tr.stages[-1].error == "EmbeddingError: embedding failed for item 1: down"
    # A subclass of a subclass is not wrapped twice: one stage per call.
    # 子类的子类不会被包装两次：每次调用一个阶段。
    assert len(tr.stages) == 1
    assert isinstance(embedder, EmbedderBase)


def test_concurrent_calls_lose_no_counts(rule_llm, hash_embedder):
    """Threads hammering one provider produce exactly N metered calls and N estimated token sums.
    多线程并发调用同一提供者，恰好得到 N 次计量调用与 N 份估算 token 之和。

    Input / 输入：
        rule_llm: RuleLLM fixture.
        rule_llm：RuleLLM 夹具。

        hash_embedder: HashEmbedder fixture.
        hash_embedder：HashEmbedder 夹具。

    Output / 输出：
        Assertions over both usage snapshots.
        对两个用量快照的断言。

    State effect / 状态影响：
        Spawns and joins worker threads.
        创建并等待工作线程。
    """
    system = prompts.OBSERVATION_EXTRACTION_PROMPT
    user = "I like tea."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    per_thread, n_threads = 40, 8
    errors: list[BaseException] = []

    def _work() -> None:
        try:
            for _ in range(per_thread):
                rule_llm.generate_response(messages, response_format={"type": "json_object"})
                hash_embedder.embed(user)
                hash_embedder.embed_batch([user, user])
        except BaseException as exc:  # pragma: no cover - surfaces in the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    total = per_thread * n_threads
    snap = rule_llm.usage.snapshot()
    assert snap.llm_calls == total and snap.input_tokens == total * estimate_messages_tokens(messages)
    assert snap.total_tokens == snap.input_tokens + snap.output_tokens and snap.estimated is True
    esnap = hash_embedder.usage.snapshot()
    assert (esnap.embedding_calls, esnap.embedded_texts) == (2 * total, 3 * total)


def test_usage_of_merges_llm_and_embedder(hms):
    """usage_of(hms) sums hms.llm.usage and hms.embedder.usage; it tolerates objects without providers.
    usage_of(hms) 把 hms.llm.usage 与 hms.embedder.usage 相加；对没有提供者的对象也能容忍。

    Input / 输入：
        hms: HMS fixture (rule LLM + hash embedder).
        hms：HMS 夹具（规则 LLM + 哈希嵌入器）。

    Output / 输出：
        Assertions over the merged Usage.
        对合并后 Usage 的断言。

    State effect / 状态影响：
        Adds one message to the fixture HMS.
        向夹具 HMS 添加一条消息。
    """
    assert usage_of(object()) == Usage()
    assert isinstance(hms.llm, LLMBase) and isinstance(hms.embedder, EmbedderBase)
    before = usage_of(hms)
    assert before == Usage()
    hms.add("I moved to Berlin last week and I love espresso.", user_id="alice")
    merged = usage_of(hms)
    llm_snap, emb_snap = hms.llm.usage.snapshot(), hms.embedder.usage.snapshot()
    assert merged.llm_calls == llm_snap.llm_calls >= 1
    assert merged.embedding_calls == emb_snap.embedding_calls >= 1 and merged.embedded_texts == emb_snap.embedded_texts
    assert merged.input_tokens == llm_snap.input_tokens and merged.estimated is True
    assert merged.duration_ms == pytest.approx(llm_snap.duration_ms + emb_snap.duration_ms)
    hms.llm.usage.reset()
    hms.embedder.usage.reset()
    assert usage_of(hms) == Usage()


def test_rule_llm_metering_is_thread_local_for_sdk_usage(rule_llm):
    """`_last_usage` is a per-thread slot: a value left by one thread never leaks into another's call.
    `_last_usage` 是线程私有槽位：一个线程留下的值绝不会泄漏进另一线程的调用。

    Input / 输入：
        rule_llm: RuleLLM fixture.
        rule_llm：RuleLLM 夹具。

    Output / 输出：
        Assertions over the slot and the resulting snapshot.
        对槽位与结果快照的断言。

    State effect / 状态影响：
        Spawns and joins one worker thread.
        创建并等待一个工作线程。
    """
    # 哨兵取一个任何真实提示词都到不了的值，避免与估算的 token 数撞车 / a sentinel no real prompt estimate can
    # reach, so it cannot collide with the estimated token count
    sentinel = 10**6
    rule_llm._last_usage = types.SimpleNamespace(prompt_tokens=sentinel, completion_tokens=sentinel)
    seen: list[Any] = []

    def _read() -> None:
        seen.append(rule_llm._last_usage)
        rule_llm.generate_response(
            [{"role": "system", "content": prompts.OBSERVATION_EXTRACTION_PROMPT}, {"role": "user", "content": "hi"}]
        )

    worker = threading.Thread(target=_read)
    worker.start()
    worker.join()
    assert seen == [None] and rule_llm.usage.snapshot().estimated is True
    assert rule_llm.usage.snapshot().input_tokens < sentinel
    assert rule_llm._last_usage is not None and isinstance(rule_llm, RuleLLM)
