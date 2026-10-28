"""Group B tests: hash embedder, index parity, the rule LLM against the §19 goldens, scripted LLM, guards, factories.
B 组测试：哈希嵌入器、索引一致性、规则 LLM 对照 §19 黄金输出、脚本化 LLM、导入守卫、工厂。
"""

# Purpose (EN): Pin the provider layer (CONTRACT §10–§11, §18.3 tests 16–23) with fully offline, deterministic tests.
# 目的（中）：以完全离线、确定性的测试固定提供者层（契约 §10–§11、§18.3 测试 16–23）。

# Input (EN): tests/fixtures/events/*.json, tests/fixtures/golden/*.json, the shared conftest fixtures.
# 输入（中）：tests/fixtures/events/*.json、tests/fixtures/golden/*.json、共享的 conftest 夹具。

# Output (EN): Assertions (and rewritten golden files under --update-golden).
# 输出（中）：断言（以及 --update-golden 下重写的黄金文件）。

# State effect (EN): None outside the in-memory store and the registry entries added and removed within one test.
# 状态影响（中）：除内存 store 与单个测试内增删的注册表条目外无副作用。

from __future__ import annotations

import base64
import inspect
import json
import math
import subprocess
import sys
import types
from typing import Any, ClassVar

import pytest

from hms_elite import models
from hms_elite.configs import prompts
from hms_elite.configs.base import HMSConfig, LlmConfig
from hms_elite.configs.registry import REGISTRY
from hms_elite.embeddings.hash import HashEmbedder
from hms_elite.enums import (
    Decision,
    EdgeType,
    InvariantKind,
    ObservationType,
    Relation,
    ResidualKind,
    Reward,
    Sentiment,
)
from hms_elite.exceptions import ConfigError, DependencyError, EmbeddingError, LLMError
from hms_elite.index import memory as memory_module
from hms_elite.index.memory import MemoryIndex
from hms_elite.index.sqlite import SQLiteIndex
from hms_elite.llms.base import DESCRIBE_IMAGE_PROMPT, BaseLlmConfig, LLMBase
from hms_elite.llms.rule import (
    EXCLUSIVE,
    PREDICATE_FAMILIES,
    RuleLLM,
    coarsen_rule,
    decide_revision,
    family,
    propose_edges,
)
from hms_elite.llms.scripted import ScriptedLLM
from hms_elite.utils.factory import EmbedderFactory, IndexFactory, LlmFactory, load_class, register_provider
from tests.conftest import GOLDEN_DIR

SCOPE = models.Scope(user_id="alice", run_id="run-test")
GUARDED_MODULES = ("hms_elite.llms.openai", "hms_elite.llms.anthropic", "hms_elite.embeddings.openai")
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_DERIVED = f"Do not re-apply '{S4_EP1_A}': billing_e2e failed: subscription_state_regression"


class FakeLlmConfig(BaseLlmConfig):
    """Config of the registered test provider.
    注册的测试提供者的配置。
    """

    greeting: str = "hi"


class FakeLLM(LLMBase):
    """A provider registered at run time through register_provider.
    通过 register_provider 在运行时注册的提供者。
    """

    config_class: ClassVar[type[BaseLlmConfig]] = FakeLlmConfig
    provider_name: ClassVar[str] = "x"

    def generate_response(
        self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs: Any
    ) -> str | dict:
        """Produce a completion for chat messages.
        生成对话补全。

        Input / 输入：
            messages: Ignored.
            messages：忽略。

            tools: Ignored.
            tools：忽略。

            tool_choice: Ignored.
            tool_choice：忽略。

            kwargs: Ignored.
            kwargs：忽略。

        Output / 输出：
            str or dict: the configured greeting as JSON.
            文本或字典：配置的问候语（JSON）。

        State effect / 状态影响：
            None.
            无。
        """
        return json.dumps({"observations": [], "greeting": self.config.greeting})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# 辅助函数
# ---------------------------------------------------------------------------
def _observe(rule_llm: RuleLLM, messages: list[dict], *, chain: bool = True, current_date: str = "2026-09-02"):
    """Run one observation_extraction call per message, feeding earlier observations back as existing memories.
    每条消息一次 observation_extraction 调用，并把先前的观察作为已有记忆回灌。
    """
    existing: list[dict] = []
    calls: list[list[dict]] = []
    for message in messages:
        timestamp = message.get("timestamp")
        user = prompts.build_observation_prompt(
            existing_memories=existing,
            new_messages=[{"idx": 0, "role": message["role"], "text": message["content"], "timestamp": timestamp}],
            last_messages=[],
            observation_date=(timestamp or current_date)[:10],
            current_date=current_date,
        )
        out = rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=user)
        calls.append(out)
        if chain:
            existing = existing + [{"id": str(len(existing) + i), "text": o["text"]} for i, o in enumerate(out)]
    return calls


def _as_observation(item: dict) -> models.Observation:
    """Validate one rule observation dict against the group-A Observation model.
    用 A 组的 Observation 模型校验一个规则观察字典。
    """
    fields = {k: v for k, v in item.items() if k not in {"id", "event_idx"}}
    return models.Observation(**fields, scope=SCOPE, anchor_ids=["anc_000001"])


def _seed_atoms(store, texts: dict[str, str]) -> None:
    """Insert one witnessed state atom per (atom_id, text) so atom_vectors rows satisfy their foreign key.
    为每个 (atom_id, text) 插入一个带见证的 state 原子，使 atom_vectors 行满足外键。
    """
    for atom_id, text in texts.items():
        event = models.SourceEvent(scope=SCOPE, role="user", payload={"text": text}, timestamp="2026-06-15T09:00:00Z")
        stored, _ = store.insert_source_event(event)
        anchor = store.get_or_create_anchor(stored)
        store.insert_atom(
            models.MemoryAtom(
                atom_id=atom_id,
                atom_type="state",
                content={"text": text},
                text=text,
                scale_level="state",
                source_kind="observation",
                source_id=f"obs_{atom_id}",
                extracted_by="rule@obs-v1",
                scope=SCOPE,
                anchor_ids=[anchor.anchor_id],
                obs_type="fact",
                subject="user",
                predicate="states",
                time_anchor="2026-06-15T09:00:00Z",
            )  # fmt: skip
        )


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors.
    两个向量的点积。
    """
    return math.fsum(x * y for x, y in zip(a, b))


def _purge_guarded_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the guarded provider modules from sys.modules so the next import re-runs their guards.
    从 sys.modules 移除受守卫的提供者模块，使下一次导入重新执行守卫。
    """
    for name in GUARDED_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _fake_openai(calls: list[dict]) -> types.ModuleType:
    """A stand-in `openai` module recording chat.completions.create / embeddings.create kwargs.
    记录 chat.completions.create / embeddings.create 参数的替身 `openai` 模块。
    """
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            calls.append({"api": "chat", **kwargs})
            message = types.SimpleNamespace(content='{"observations": []}', tool_calls=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _Embeddings:
        def create(self, **kwargs):
            calls.append({"api": "embeddings", **kwargs})
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.5, 0.5, 0.5, 0.5])])

    class OpenAI:
        def __init__(self, **kwargs):
            calls.append({"api": "client", **kwargs})
            self.chat = types.SimpleNamespace(completions=_Completions())
            self.embeddings = _Embeddings()

    module.OpenAI = OpenAI
    return module


def _fake_anthropic(calls: list[dict]) -> types.ModuleType:
    """A stand-in `anthropic` module recording messages.create kwargs.
    记录 messages.create 参数的替身 `anthropic` 模块。
    """
    module = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            calls.append({"api": "messages", **kwargs})
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text='{"observations": []}')])

    class Anthropic:
        def __init__(self, **kwargs):
            calls.append({"api": "client", **kwargs})
            self.messages = _Messages()

    module.Anthropic = Anthropic
    return module


def _fake_anthropic_signature(calls: list[dict], *, sampling: bool) -> types.ModuleType:
    """A stand-in `anthropic` whose Messages.create has an explicit signature: 0.x-shaped (sampling=True) or 1.x-shaped.
    Messages.create 带显式签名的替身 `anthropic`：0.x 形状（sampling=True）或 1.x 形状（无采样参数、无 **kwargs）。
    """
    module = types.ModuleType("anthropic")

    def _reply(**kwargs):
        calls.append({"api": "messages", **kwargs})
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text='{"observations": []}')])

    def create_0x(self, *, model, messages, max_tokens, system=None, tools=None, tool_choice=None, temperature=None):
        return _reply(model=model, messages=messages, max_tokens=max_tokens, system=system, temperature=temperature)

    def create_1x(self, *, model, messages, max_tokens, system=None, tools=None, tool_choice=None):
        return _reply(model=model, messages=messages, max_tokens=max_tokens, system=system)

    messages_cls = type("Messages", (), {"create": create_0x if sampling else create_1x})

    class Anthropic:
        def __init__(self, **kwargs):
            calls.append({"api": "client", **kwargs})
            self.messages = messages_cls()

    module.Anthropic = Anthropic
    return module


def _s4_nodes() -> list[dict]:
    """The S4 EP1 / EP2 nodes in the §6.2 edge-prompt shape.
    §6.2 边提示词形状的 S4 EP1 / EP2 节点。
    """
    return [
        {
            "ref": "n0", "kind": "ep", "u": "fix checkout webhook retry", "b": S4_CONSTRAINT, "a": S4_EP1_A,
            "y": "billing_e2e failed: subscription_state_regression", "r": "negative",
            "subject": None, "predicate": None, "objects": [], "time": "2026-06-15T10:04:00+00:00",
            "entities": ["retry.ts", "billing_e2e", "subscription_state_regression"],
            "topics": ["engineering", "billing"], "anchor_refs": ["w0", "w1"],
        },
        {
            "ref": "n1", "kind": "ep", "u": "fix checkout webhook retry", "b": f"{S4_CONSTRAINT} ; {S4_DERIVED}",
            "a": S4_EP2_A, "y": "billing_e2e passed", "r": "positive",
            "subject": None, "predicate": None, "objects": [], "time": "2026-06-15T10:14:00+00:00",
            "entities": ["idempotency.ts", "billing_e2e"], "topics": ["engineering", "billing"],
            "anchor_refs": ["w2", "w3"],
        },
    ]  # fmt: skip


def _s4_witnesses() -> list[dict]:
    """Witness refs of the S4 nodes.
    S4 节点的 witness 引用。
    """
    return [
        {"ref": "w0", "node_ref": "n0", "source_ref": "test://billing_e2e", "time": "2026-06-15T10:04:00+00:00"},
        {"ref": "w1", "node_ref": "n0", "source_ref": "file://src/billing/retry.ts", "time": "2026-06-15T10:00:00+00:00"},
        {"ref": "w2", "node_ref": "n1", "source_ref": "test://billing_e2e", "time": "2026-06-15T10:14:00+00:00"},
        {"ref": "w3", "node_ref": "n1", "source_ref": "file://src/billing/idempotency.ts", "time": "2026-06-15T10:10:00+00:00"},
    ]  # fmt: skip


def _s4_support_nodes() -> list[dict]:
    """The S4 EP nodes in the coarsen-prompt shape.
    coarsen 提示词形状的 S4 EP 节点。
    """
    return [
        {
            "ref": "n0", "kind": "ep", "u": "fix checkout webhook retry", "b": S4_CONSTRAINT, "a": S4_EP1_A,
            "y": "billing_e2e failed: subscription_state_regression", "r": "negative",
            "constraints": [S4_CONSTRAINT], "time": "2026-06-15T10:04:00+00:00", "anchor_refs": ["w0", "w1"],
        },
        {
            "ref": "n1", "kind": "ep", "u": "fix checkout webhook retry", "b": f"{S4_CONSTRAINT} ; {S4_DERIVED}",
            "a": S4_EP2_A, "y": "billing_e2e passed", "r": "positive",
            "constraints": [S4_CONSTRAINT, S4_DERIVED], "time": "2026-06-15T10:14:00+00:00", "anchor_refs": ["w2", "w3"],
        },
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# 16. HashEmbedder
# 16. HashEmbedder（哈希嵌入器）
# ---------------------------------------------------------------------------
def test_hash_embedder_deterministic_normalized(hash_embedder):
    """Same text → identical unit vectors across instances; similar texts score higher; CJK input is non-zero.
    相同文本在不同实例间得到相同单位向量；相似文本得分更高；CJK 输入非零。

    Input / 输入：
        hash_embedder: The 64-dim HashEmbedder fixture.
        hash_embedder：64 维 HashEmbedder 夹具。

    Output / 输出：
        Assertions over HashEmbedder.embed / embed_batch / dims / identity.
        对 HashEmbedder.embed / embed_batch / dims / identity 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    other = HashEmbedder({"embedding_dims": 64})
    v1 = hash_embedder.embed("oat latte", "add")
    assert v1 == other.embed("oat latte") == hash_embedder.embed("oat latte", "search")
    assert len(v1) == 64 and hash_embedder.dims == 64 and hash_embedder.identity == "hash@64"
    assert math.sqrt(_cosine(v1, v1)) == pytest.approx(1.0, abs=1e-9)
    assert _cosine(v1, hash_embedder.embed("oat-milk latte")) > _cosine(v1, hash_embedder.embed("billing test"))
    # 大小写不敏感
    assert hash_embedder.embed("Oat Latte") == v1  # case-insensitive
    cjk = hash_embedder.embed("我喜欢燕麦拿铁")
    assert any(abs(x) > 0 for x in cjk) and math.sqrt(_cosine(cjk, cjk)) == pytest.approx(1.0, abs=1e-9)
    assert hash_embedder.embed("") == [0.0] * 64 and hash_embedder.embed("   ") == [0.0] * 64
    batch = hash_embedder.embed_batch(["oat latte", "billing test"])
    assert batch[0] == v1 and len(batch) == 2
    small = HashEmbedder({"embedding_dims": 8, "ngram_min": 1, "ngram_max": 1})
    assert len(small.embed("abc")) == 8 and small.identity == "hash@8"
    with pytest.raises(ConfigError) as info:
        HashEmbedder({"embedding_dim": 32})
    assert info.value.error_code == "CONFIG_001" and "Did you mean 'embedding_dims'?" in info.value.message

    class _Broken(HashEmbedder):
        def embed(self, text, memory_action=None):
            raise RuntimeError("boom")

    with pytest.raises(EmbeddingError) as err:
        _Broken().embed_batch(["x"])
    assert err.value.error_code == "EMBED_001" and isinstance(err.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# 17. Index parity (NOT parametrized: both providers in one test)
# 17. 索引一致性（未参数化：两个提供方在同一个测试内）
# ---------------------------------------------------------------------------
def test_index_parity_sqlite_memory(store, hash_embedder):
    """SQLiteIndex and MemoryIndex return identical orderings and scores; candidate_ids and reset behave alike.
    SQLiteIndex 与 MemoryIndex 返回相同的排序与分数；candidate_ids 与 reset 行为一致。

    Input / 输入：
        store: In-memory StateStore (atom_vectors host).
        store：内存 StateStore（atom_vectors 宿主）。

        hash_embedder: Vector source.
        hash_embedder：向量来源。

    Output / 输出：
        Assertions over upsert / upsert_batch / get / delete / search / count / reset / info of both providers.
        对两种提供者 upsert / upsert_batch / get / delete / search / count / reset / info 的断言。

    State effect / 状态影响：
        Writes atom_vectors in the in-memory store.
        写入内存 store 的 atom_vectors。
    """
    sqlite_index = SQLiteIndex({"use_numpy": "auto"}, store=store, dims=64)
    memory_index = MemoryIndex(dims=64)
    plain = SQLiteIndex({"use_numpy": "never"}, store=store, dims=64)
    texts = {
        "atm_000001": "User likes oat-milk lattes in the morning.",
        "atm_000002": "User likes black coffee.",
        "atm_000003": "billing_e2e failed after the retry patch",
        "atm_000004": "Sarah joined the billing team in March.",
    }
    _seed_atoms(store, texts)
    items = [(atom_id, hash_embedder.embed(text)) for atom_id, text in texts.items()]
    sqlite_index.upsert_batch(items[:2], embedder=hash_embedder.identity)
    for atom_id, vector in items[2:]:
        sqlite_index.upsert(atom_id, vector, embedder=hash_embedder.identity)
    memory_index.upsert_batch(items, embedder=hash_embedder.identity)
    assert sqlite_index.count() == memory_index.count() == 4 == store.count_vectors()
    query = hash_embedder.embed("what coffee does the user like")
    a = sqlite_index.search(query, candidate_ids=None, top_k=10)
    b = memory_index.search(query, candidate_ids=None, top_k=10)
    c = plain.search(query, candidate_ids=None, top_k=10)
    assert [h[0] for h in a] == [h[0] for h in b] == [h[0] for h in c]
    assert [h[1] for h in a] == pytest.approx([h[1] for h in b], abs=1e-9)
    assert [h[1] for h in a] == pytest.approx([h[1] for h in c], abs=1e-9)
    assert a[0][0] in {"atm_000001", "atm_000002"} and all(0.0 <= s <= 1.0 for _, s in a)
    assert [s for _, s in a] == sorted((s for _, s in a), reverse=True)
    restricted = sqlite_index.search(query, candidate_ids=["atm_000003", "atm_000004", "atm_missing"], top_k=10)
    assert {h[0] for h in restricted} == {"atm_000003", "atm_000004"}
    assert restricted == memory_index.search(query, candidate_ids=["atm_000003", "atm_000004", "atm_missing"], top_k=10)
    assert (
        sqlite_index.search(query, candidate_ids=[], top_k=10)
        == memory_index.search(query, candidate_ids=[], top_k=10)
        == []
    )
    assert len(sqlite_index.search(query, candidate_ids=None, top_k=2)) == 2
    assert (
        sqlite_index.search(query, candidate_ids=None, top_k=0)
        == memory_index.search(query, candidate_ids=None, top_k=0)
        == []
    )
    assert sqlite_index.get("atm_000001") == pytest.approx(memory_index.get("atm_000001"), abs=1e-9)
    assert sqlite_index.get("atm_none") is None and memory_index.get("atm_none") is None
    sqlite_index.delete("atm_000004")
    memory_index.delete("atm_000004")
    sqlite_index.delete("atm_none")
    assert sqlite_index.count() == memory_index.count() == 3
    for index, provider in ((sqlite_index, "sqlite"), (memory_index, "memory")):
        info = index.info()
        assert set(info) == {"provider", "dims", "count", "numpy"} and info["provider"] == provider
        assert info["dims"] == 64 and info["count"] == 3 and isinstance(info["numpy"], bool)
    assert plain.info()["numpy"] is False
    sqlite_index.reset()
    memory_index.reset()
    assert sqlite_index.count() == memory_index.count() == 0 == store.count_vectors()
    assert sqlite_index.search(query, candidate_ids=None, top_k=5) == []
    with pytest.raises(ConfigError) as info:
        SQLiteIndex(store=None)
    assert info.value.error_code == "CONFIG_001"
    with pytest.raises(ConfigError) as info:
        SQLiteIndex({"use_nunpy": "never"}, store=store)
    assert "Did you mean 'use_numpy'?" in info.value.message


# ---------------------------------------------------------------------------
# 17b. MemoryIndex.search iterates a snapshot; hms_elite.index re-exports milvus lazily
# 17b. MemoryIndex.search 迭代的是快照；hms_elite.index 惰性再导出 milvus
# ---------------------------------------------------------------------------
def test_memory_index_search_snapshots_the_live_dict(monkeypatch):
    """A concurrent upsert() while rank_hits iterates the candidates must not raise (AsyncHMS worker threads).
    rank_hits 迭代候选时并发的 upsert() 不得抛错（AsyncHMS 工作线程）。

    Input / 输入：
        monkeypatch: Wraps hms_elite.index.memory.rank_hits with a writer that upserts mid-iteration.
        monkeypatch：把 hms_elite.index.memory.rank_hits 包成一个在迭代中途写入的版本。

    Output / 输出：
        The three pre-existing ids in the hits, no RuntimeError, and the late rows counted afterwards.
        命中里只有原有的三个 id、无 RuntimeError，且事后能数到后写入的行。

    State effect / 状态影响：
        None outside the MemoryIndex instance.
        除 MemoryIndex 实例外无副作用。
    """
    index = MemoryIndex(dims=4)
    for i in range(3):
        index.upsert(f"atm_{i:06d}", [1.0, 0.0, 0.0, 0.0], embedder="hash@4")
    real = memory_module.rank_hits

    def racing(query, vectors, *, top_k, use_numpy):
        # a writer lands while the candidates are being walked: the live dict would raise "changed size"
        # 候选被遍历时有写入者落地：若传的是实时字典会抛 "changed size"
        seen = 0
        for _atom_id in vectors:
            seen += 1
            index.upsert(f"atm_late_{seen}", [0.0, 1.0, 0.0, 0.0], embedder="hash@4")
        return real(query, vectors, top_k=top_k, use_numpy=use_numpy)

    monkeypatch.setattr(memory_module, "rank_hits", racing)
    hits = index.search([1.0, 0.0, 0.0, 0.0], candidate_ids=None, top_k=10)
    assert [h[0] for h in hits] == ["atm_000000", "atm_000001", "atm_000002"]
    assert index.count() == 6
    # the candidate_ids path already builds its own dict; it must keep doing so
    # candidate_ids 路径本来就构造自己的字典；必须保持如此
    restricted = index.search([1.0, 0.0, 0.0, 0.0], candidate_ids=["atm_000001", "atm_late_1"], top_k=10)
    assert [h[0] for h in restricted] == ["atm_000001", "atm_late_1"]


def test_index_package_reexports_milvus_without_importing_pymilvus():
    """hms_elite.index exposes MilvusIndex / MilvusIndexConfig, and importing it never loads pymilvus (§0.4).
    hms_elite.index 暴露 MilvusIndex / MilvusIndexConfig，且导入它绝不加载 pymilvus（§0.4）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Identity of the re-exported names and a fresh interpreter's sys.modules after `import hms_elite.index`.
        再导出名称的同一性，以及新解释器 `import hms_elite.index` 之后的 sys.modules。

    State effect / 状态影响：
        Spawns one child interpreter (no network, no writes).
        启动一个子解释器（无网络、无写入）。
    """
    import hms_elite.index as index_package
    from hms_elite.index.milvus import MilvusIndex, MilvusIndexConfig

    assert index_package.MilvusIndex is MilvusIndex and index_package.MilvusIndexConfig is MilvusIndexConfig
    assert {"MilvusIndex", "MilvusIndexConfig"} <= set(index_package.__all__)
    probe = (
        "import sys, hms_elite.index as i; "
        "assert i.MilvusIndex.__module__ == 'hms_elite.index.milvus'; "
        "print(sorted(m for m in sys.modules if m.startswith('pymilvus')))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# 18. Rule LLM observation goldens
# 18. Rule LLM 观察黄金用例
# ---------------------------------------------------------------------------
def test_rule_llm_observation_golden(rule_llm, events, golden):
    """RuleLLM observations for chat_preference / chat_multisession / chat_zh equal §19 and the golden files.
    RuleLLM 对 chat_preference / chat_multisession / chat_zh 的观察等于 §19 与黄金文件。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

        events: Fixture loader.
        events：夹具加载器。

        golden: Golden helper (masked JSON).
        golden：黄金比对辅助（掩码 JSON）。

    Output / 输出：
        Assertions over RuleLLM.generate_json('observation_extraction') and the three observation goldens.
        对 RuleLLM.generate_json('observation_extraction') 与三个观察黄金文件的断言。

    State effect / 状态影响：
        Rewrites the golden files under --update-golden.
        --update-golden 下重写黄金文件。
    """
    pref = _observe(rule_llm, events("chat_preference").messages)
    assert [len(call) for call in pref] == [1, 1]
    first, second = pref[0][0], pref[1][0]
    assert first == {
        "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
        "objects": ["oat-milk lattes in the morning"], "relation": "single_fact",
        "text": "User likes oat-milk lattes in the morning.", "date_observed": "2026-04-02", "valid_from": None,
        "valid_to": None, "confidence": 0.8, "attributed_to": "user", "speaker": None, "sentiment": "positive",
        "topic_tags": ["coffee", "preference"], "entity_mentions": [], "supersedes_hint_ids": [],
    }  # fmt: skip
    assert second == {
        "id": "0", "event_idx": 0, "type": "preference_event", "subject": "user", "predicate": "likes",
        "objects": ["black coffee"], "relation": "supersession", "text": "User likes black coffee.",
        "date_observed": "2026-04-20", "valid_from": None, "valid_to": None, "confidence": 0.9,
        "attributed_to": "user", "speaker": None, "sentiment": "positive", "topic_tags": ["coffee", "preference"],
        "entity_mentions": [], "supersedes_hint_ids": ["0"],
    }  # fmt: skip
    assert decide_revision(second, {**first, "id": "0"}) == (Decision.REVISE, "supersession cue")
    assert rule_llm.calls[-1]["task"] == "observation_extraction"
    assert rule_llm.calls[-1]["sections"]["Existing Memories"] == [{"id": "0", "text": first["text"]}]
    golden.masked("observations_chat_preference.json", pref)

    multi = _observe(rule_llm, events("chat_multisession").messages, chain=False)
    assert [len(call) for call in multi] == [1, 1, 1]
    s1, s2, s3 = multi[0][0], multi[1][0], multi[2][0]
    assert (s1["type"], s1["subject"], s1["predicate"], s1["objects"]) == ("temporal_fact", "Sarah", "did", [])
    assert s1["relation"] == "single_fact" and s1["date_observed"] == "2026-03-01" and s1["confidence"] == 0.75
    assert s1["text"] == "Sarah joined the billing team in March." and s1["topic_tags"] == ["work", "billing"]
    assert s1["entity_mentions"] == ["Sarah"] and s1["sentiment"] == "neutral"
    assert (s2["type"], s2["predicate"], s2["date_observed"]) == ("temporal_fact", "did", "2026-05-06")
    assert s2["text"] == "Sarah shipped the webhook retry design on May 6." and s2["topic_tags"] == [
        "work",
        "engineering",
    ]
    assert (s3["type"], s3["subject"], s3["predicate"], s3["relation"]) == ("fact", "Sarah", "is", "single_fact")
    assert s3["text"] == "Sarah is reviewing the checkout refactor this week." and s3["confidence"] == 0.6
    assert s3["topic_tags"] == ["engineering", "billing"] and s3["date_observed"] == "2026-06-11"
    assert all(o["supersedes_hint_ids"] == [] and o["attributed_to"] == "user" for call in multi for o in call)
    golden.masked("observations_chat_multisession.json", multi)

    zh = _observe(rule_llm, events("chat_zh").messages)
    assert [len(call) for call in zh] == [1, 1]
    z1, z2 = zh[0][0], zh[1][0]
    assert (z1["type"], z1["predicate"], z1["objects"], z1["relation"]) == (
        "preference_event",
        "likes",
        ["燕麦拿铁"],
        "single_fact",
    )
    assert (
        z1["text"] == "用户喜欢燕麦拿铁。" and z1["topic_tags"] == ["coffee", "preference"] and z1["confidence"] == 0.8
    )
    assert (z2["predicate"], z2["objects"], z2["relation"]) == ("likes", ["黑咖啡"], "supersession")
    assert z2["text"] == "用户喜欢黑咖啡。" and z2["confidence"] == 0.9 and z2["supersedes_hint_ids"] == ["0"]
    assert z1["entity_mentions"] == [] and z2["entity_mentions"] == [] and z2["date_observed"] == "2026-04-20"
    assert decide_revision(z2, {**z1, "id": "0"})[0] is Decision.REVISE
    golden.masked("observations_chat_zh.json", zh)

    for call in pref + multi + zh:
        for item in call:
            assert list(item) == [
                "id", "event_idx", "type", "subject", "predicate", "objects", "relation", "text", "date_observed",
                "valid_from", "valid_to", "confidence", "attributed_to", "speaker", "sentiment", "topic_tags",
                "entity_mentions", "supersedes_hint_ids",
            ]  # fmt: skip
            _as_observation(item)


def test_rule_llm_observation_rules_habit_temporal_conflict_assistant(rule_llm, events):
    """R2/R5/R6/R7 and the assistant gate on the remaining fixtures; chat_conflict decides CONFLICT.
    其余夹具上的 R2/R5/R6/R7 与 assistant 门控；chat_conflict 判定为 CONFLICT。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

        events: Fixture loader.
        events：夹具加载器。

    Output / 输出：
        Assertions over the observation rules R2, R5, R6, R7, R9 and decide_revision.
        对观察规则 R2、R5、R6、R7、R9 与 decide_revision 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    habit = _observe(rule_llm, events("chat_habit_temporal").messages)
    visited, runs = habit[0][0], habit[1][0]
    assert (visited["type"], visited["subject"], visited["predicate"], visited["objects"]) == (
        "temporal_fact",
        "user",
        "did",
        ["Berlin"],
    )
    assert visited["date_observed"] == "2026-05-09" and visited["text"] == "Yesterday User visited the Berlin office."
    assert (
        visited["topic_tags"] == ["travel"]
        and visited["entity_mentions"] == ["Berlin"]
        and visited["confidence"] == 0.75
    )
    assert (runs["type"], runs["predicate"], runs["relation"], runs["objects"]) == (
        "habit",
        "does",
        "habit",
        ["run every morning"],
    )
    assert (
        runs["confidence"] == 0.65
        and runs["date_observed"] == "2026-09-02"
        and runs["text"] == "User usually run every morning."
    )

    conflict = _observe(rule_llm, events("chat_conflict").messages)
    paris, berlin = conflict[0][0], conflict[1][0]
    assert (paris["type"], paris["predicate"], paris["objects"], paris["date_observed"]) == (
        "fact",
        "lives_in",
        ["Paris"],
        "2026-05-01",
    )
    assert (berlin["predicate"], berlin["objects"], berlin["date_observed"]) == ("lives_in", ["Berlin"], "2026-04-01")
    assert paris["confidence"] == 0.85 and paris["entity_mentions"] == ["Paris"] and paris["subject"] == "user"
    # 只共享一个内容词（'live'）
    assert berlin["supersedes_hint_ids"] == []  # only one shared content token ('live')
    assert decide_revision(berlin, {**paris, "id": "0"}) == (Decision.CONFLICT, "older statement contradicts newer")
    assert decide_revision(paris, {**berlin, "id": "0"}) == (Decision.REVISE, "exclusive attribute updated")

    closure = _observe(rule_llm, events("chat_action_closure").messages)
    assistant = closure[0][0]
    assert assistant["subject"] == "assistant" and assistant["attributed_to"] == "assistant"
    assert assistant["text"] == (
        "Assistant patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."
    )
    assert (assistant["type"], assistant["predicate"], assistant["objects"]) == (
        "fact",
        "patched",
        ["retry.ts", "billing_e2e"],
    )
    assert assistant["confidence"] == 0.6 and assistant["topic_tags"] == ["engineering", "billing"]

    constraint = _observe(rule_llm, [events("structured_billing").constraint])[0][0]
    assert constraint == {
        "id": "0", "event_idx": 0, "type": "constraint", "subject": "project", "predicate": "must_not",
        "objects": ["subscription state machine"], "relation": "constraint", "text": S4_CONSTRAINT,
        "date_observed": "2026-06-15", "valid_from": None, "valid_to": None, "confidence": 0.9,
        "attributed_to": "user", "speaker": None, "sentiment": "neutral", "topic_tags": ["billing", "constraint"],
        "entity_mentions": [], "supersedes_hint_ids": [],
    }  # fmt: skip

    mixed = [
        {"role": "user", "content": "Hi there! What do you think? My name is Alice. I'm a backend engineer."},
        {
            "role": "user",
            "content": "We must ship the checkout fix by Friday. I'm torn between the Fender and the Gibson.",
        },
        {"role": "assistant", "content": "Sure, that sounds great. I recommend the Fender Stratocaster for jazz."},
        {"role": "system", "content": "I like everything."},
        {"role": "user", "content": "I like tea over coffee. I no longer like espresso. I don't like mocha."},
        {"role": "user", "content": "I like tea over coffee."},
    ]
    user = prompts.build_observation_prompt(
        existing_memories=[
            {"id": "0", "text": "User likes black coffee."},
            {"id": "1", "text": "Sarah lives in Paris."},
        ],
        new_messages=[
            {"idx": i, "role": m["role"], "text": m["content"], "timestamp": None} for i, m in enumerate(mixed)
        ],
        last_messages=[{"role": "user", "content": "earlier"}],
        observation_date="2026-07-01",
        current_date="2026-09-02",
        custom_instructions="none",
    )
    out = rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=user)
    by_text = {o["text"]: o for o in out}
    assert [o["id"] for o in out] == [str(i) for i in range(len(out))]
    assert by_text["User's name is Alice."]["predicate"] == "is_named" and by_text["User's name is Alice."][
        "objects"
    ] == ["Alice"]
    assert by_text["User is a backend engineer."]["predicate"] == "is" and by_text["User is a backend engineer."][
        "objects"
    ] == ["backend engineer"]
    must = by_text["The team must ship the checkout fix by Friday."]
    assert (must["type"], must["predicate"], must["subject"], must["event_idx"]) == ("constraint", "must", "user", 1)
    assert must["topic_tags"] == ["engineering", "billing", "constraint"] and must["confidence"] == 0.85
    torn = by_text["User compared Fender and Gibson; no exclusive final choice was stated."]
    assert (torn["relation"], torn["predicate"], torn["objects"], torn["sentiment"]) == (
        "comparison",
        "compares",
        ["Fender", "Gibson"],
        "neutral",
    )
    rec = by_text["Assistant recommend the Fender Stratocaster for jazz."]
    assert rec["subject"] == "assistant" and rec["attributed_to"] == "assistant" and rec["event_idx"] == 2
    tea = by_text["User prefers tea over coffee."]
    assert (tea["relation"], tea["predicate"], tea["objects"], tea["supersedes_hint_ids"]) == (
        "comparison",
        "prefers",
        ["tea", "coffee"],
        ["0"],
    )
    espresso = by_text["User dislikes espresso."]
    assert (espresso["relation"], espresso["predicate"], espresso["confidence"]) == ("supersession", "dislikes", 0.9)
    mocha = by_text["User dislikes mocha."]
    assert (mocha["relation"], mocha["predicate"], mocha["sentiment"], mocha["confidence"]) == (
        "single_fact",
        "dislikes",
        "negative",
        0.8,
    )
    # 批内去重
    assert sum(1 for o in out if o["text"] == "User prefers tea over coffee.") == 1  # batch dedup
    assert all(o["event_idx"] != 3 for o in out) and all(o["date_observed"] == "2026-07-01" for o in out)
    assert "Sure, that sounds great." not in by_text and not any("Hi there" in t for t in by_text)

    en_only = RuleLLM({"language": "en"})
    assert _observe(en_only, events("chat_zh").messages) == [[], []]
    zh_only = RuleLLM({"language": "zh"})
    assert _observe(zh_only, events("chat_preference").messages) == [[], []]
    strict = RuleLLM({"strict": True})
    assert _observe(strict, events("chat_multisession").messages, chain=False)[2] == []
    assert _observe(strict, events("chat_action_closure").messages) == [[]]
    with pytest.raises(ConfigError) as info:
        RuleLLM({"languag": "en"})
    assert info.value.error_code == "CONFIG_001" and "Did you mean 'language'?" in info.value.message


# ---------------------------------------------------------------------------
# 18b. R5 relocation twin (regression: 'moved to' must not be read as an R1 preference switch)
# 18b. R5 搬迁孪生（回归：'moved to' 不得被当作 R1 的偏好切换）
# ---------------------------------------------------------------------------
def test_moved_to_yields_lives_in_supersession(rule_llm):
    """'I moved to Berlin.' is R5 lives_in / supersession, not R1 likes, and §11.3 REVISEs the prior residence.
    'I moved to Berlin.' 命中 R5 的 lives_in / supersession，而非 R1 的 likes，并按 §11.3 修订此前的居住地。

    Input / 输入：
        rule_llm: The shared RuleLLM fixture (language='auto').
        rule_llm：共享的 RuleLLM 夹具（language='auto'）。

    Output / 输出：
        Assertions over the EN and ZH relocation observations and decide_revision on the pair.
        对中英搬迁 observation 以及该配对 decide_revision 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    user = prompts.build_observation_prompt(
        existing_memories=[{"id": "0", "text": "User live in Paris."}],
        new_messages=[
            {"idx": 0, "role": "user", "text": "I live in Paris.", "timestamp": None},
            {"idx": 1, "role": "user", "text": "I moved to Berlin.", "timestamp": None},
            {"idx": 2, "role": "user", "text": "我搬到柏林。", "timestamp": None},
        ],
        last_messages=[],
        observation_date="2026-07-01",
        current_date="2026-07-01",
    )
    out = rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=user)
    by_text = {o["text"]: o for o in out}
    paris = by_text["User live in Paris."]
    assert (paris["type"], paris["predicate"], paris["relation"], paris["objects"]) == (
        "fact",
        "lives_in",
        "single_fact",
        ["Paris"],
    )
    berlin = by_text["User moved to Berlin."]
    assert (berlin["type"], berlin["predicate"], berlin["relation"], berlin["objects"]) == (
        "fact",
        "lives_in",
        "supersession",
        ["Berlin"],
    )
    assert (berlin["subject"], berlin["confidence"], berlin["sentiment"]) == ("user", 0.85, "neutral")
    # 中文孪生走同一分支
    zh = by_text["我搬到柏林。"]  # the ZH twin takes the same branch
    assert (zh["predicate"], zh["relation"], zh["objects"], zh["confidence"]) == (
        "lives_in",
        "supersession",
        ["柏林"],
        0.85,
    )
    assert all(o["predicate"] != "likes" for o in out) and all(o["type"] != "preference_event" for o in out)
    assert decide_revision(berlin, paris) == (Decision.REVISE, "supersession cue")
    relocated = _observe(rule_llm, [{"role": "user", "content": "I relocated to Berlin."}])[0][0]
    # 变体也不得落回 R1
    assert relocated["predicate"] == "lives_in"  # the variant must not fall back to R1 either


# ---------------------------------------------------------------------------
# 19. EP / revision / edge / coarsen contracts
# 19. EP / 修订 / 边 / 粗化契约
# ---------------------------------------------------------------------------
def test_rule_llm_ep_and_revision_and_edge_and_coarsen_contracts(rule_llm, events, golden):
    """EP extraction, decide_revision table, propose_edges and coarsen_rule produce §6.3-valid outputs.
    EP 抽取、decide_revision 表、propose_edges 与 coarsen_rule 产出符合 §6.3 的结果。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

        events: Fixture loader.
        events：夹具加载器。

        golden: Golden helper.
        golden：黄金比对辅助。

    Output / 输出：
        Assertions over generate_json for ep_extraction / revision_decision / edge_proposal / coarsen and the pure rules.
        对 ep_extraction / revision_decision / edge_proposal / coarsen 的 generate_json 与纯规则函数的断言。

    State effect / 状态影响：
        Rewrites eps_chat_action_closure.json under --update-golden.
        --update-golden 下重写 eps_chat_action_closure.json。
    """
    message = events("chat_action_closure").messages[0]
    user = prompts.build_ep_prompt(
        events=[{"idx": 0, "role": "assistant", "object_type": "message", "text": message["content"], "timestamp": "2026-06-15T10:04:00+00:00"}],
        known_constraints=[{"id": "c0", "text": S4_CONSTRAINT}, {"id": "c1", "text": "Never retry the webhook more than five times."}],
        recent_eps=[{"id": "e0", "u": "flaky billing", "a": "reran retry.ts tests", "y": "billing_e2e passed", "r": "positive"}, {"id": "e1", "u": "x", "a": "edited README.md", "y": "", "r": "unknown"}],
        observation_date="2026-06-15",
    )  # fmt: skip
    eps = rule_llm.generate_json("ep_extraction", system=prompts.EP_EXTRACTION_PROMPT, user=user)
    assert len(eps) == 1
    ep = eps[0]
    assert ep == {
        "id": "0", "event_idx": 0, "scale": "L1_micro", "u": "the webhook kept timing out",
        "b": "Never retry the webhook more than five times.", "a": "patched retry.ts to add backoff",
        "y": "billing_e2e failed", "r": "negative",
        "kappa": {"constraints": ["Never retry the webhook more than five times."], "visibility": "private"},
        "time_anchor": "2026-06-15T10:04:00+00:00", "entity_mentions": ["retry.ts", "billing_e2e"],
        "topic_tags": ["engineering", "billing"], "uses_ep_ids": ["e0"], "confidence": 0.7,
    }  # fmt: skip
    models.EP(
        slots=models.Slots(u=ep["u"], b=ep["b"], a=ep["a"], y=ep["y"], r=ep["r"], kappa=models.Kappa(**ep["kappa"])),
        time_anchor=ep["time_anchor"], entity_mentions=ep["entity_mentions"], topic_tags=ep["topic_tags"],
        uses_ep_ids=ep["uses_ep_ids"], confidence=ep["confidence"], anchor_ids=["anc_000001"], scope=SCOPE,
    )  # fmt: skip
    bare = prompts.build_ep_prompt(
        events=[{"idx": 0, "role": "assistant", "object_type": "message", "text": message["content"], "timestamp": "2026-06-15T10:04:00+00:00"}],
        known_constraints=[], recent_eps=[], observation_date="2026-06-15",
    )  # fmt: skip
    bare_eps = rule_llm.generate_json("ep_extraction", system=prompts.EP_EXTRACTION_PROMPT, user=bare)
    assert bare_eps[0]["b"] == "" and bare_eps[0]["kappa"]["constraints"] == [] and bare_eps[0]["uses_ep_ids"] == []
    golden.masked("eps_chat_action_closure.json", bare_eps)
    more = prompts.build_ep_prompt(
        events=[
            {"idx": 0, "role": "user", "object_type": "message", "text": "I will look into it later.", "timestamp": None},
            {"idx": 1, "role": "assistant", "object_type": "message", "text": "The deploy was flaky. I reverted the config change. Tests passed afterwards.", "timestamp": "2026-06-16T10:00:00+00:00"},
            {"idx": 2, "role": "assistant", "object_type": "message", "text": "I edited api.py again.", "timestamp": None},
            {"idx": 3, "role": "assistant", "object_type": "message", "text": "I migrated the schema to keep the state machine intact, and the run went green.", "timestamp": None},
        ],
        known_constraints=[{"id": "c0", "text": S4_CONSTRAINT}], recent_eps=[], observation_date="2026-06-16",
    )  # fmt: skip
    more_eps = rule_llm.generate_json("ep_extraction", system=prompts.EP_EXTRACTION_PROMPT, user=more)
    assert [(e["event_idx"], e["id"]) for e in more_eps] == [(1, "0"), (3, "1")]
    assert (more_eps[0]["u"], more_eps[0]["a"], more_eps[0]["y"], more_eps[0]["r"]) == (
        "The deploy was flaky",
        "reverted the config change",
        "Tests passed afterwards",
        "positive",
    )
    assert (
        more_eps[1]["u"].startswith("task: ")
        and more_eps[1]["r"] == "positive"
        and more_eps[1]["kappa"]["constraints"] == [S4_CONSTRAINT]
    )
    assert all(e["r"] in Reward.values() for e in more_eps)

    rows = json.loads((GOLDEN_DIR / "revision_cases.json").read_text(encoding="utf-8"))
    assert len(rows) >= 14 and {row["rule"] for row in rows} == set(range(1, 11))
    for row in rows:
        decision, reason = decide_revision(row["new"], row["old"])
        assert isinstance(decision, Decision) and decision.value == row["decision"], row["name"]
        assert isinstance(reason, str) and reason
    assert family("likes") == family("avoids") == "PREF" and family("lives_in") == "IDENTITY"
    assert family("must") == "CONSTRAINT" and family("did") == "did"
    assert EXCLUSIVE == {"lives_in", "works_at", "studies_at", "is_named", "prefers"}
    assert set(PREDICATE_FAMILIES) == {"PREF", "IDENTITY", "CONSTRAINT"}
    revise_user = prompts.build_revision_prompt(
        new_items=[{k: rows[6]["new"][k] for k in ("id", "type", "subject", "predicate", "objects", "relation", "text", "date_observed")}],
        candidates=[{**rows[9]["old"], "id": "0"}, {**rows[6]["old"], "id": "1"}],
        current_date="2026-09-02",
    )  # fmt: skip
    decisions = rule_llm.generate_json("revision_decision", system=prompts.REVISION_DECISION_PROMPT, user=revise_user)
    assert decisions == [{"new_id": "0", "decision": "REVISE", "target_id": "1", "reason": "supersession cue"}]
    add_user = prompts.build_revision_prompt(new_items=[rows[14]["new"]], candidates=[], current_date="2026-09-02")
    assert rule_llm.generate_json("revision_decision", system=prompts.REVISION_DECISION_PROMPT, user=add_user) == [
        {"new_id": "0", "decision": "ADD", "target_id": None, "reason": "no candidates"}
    ]

    nodes, witnesses = _s4_nodes(), _s4_witnesses()
    edges = propose_edges(nodes, [["n0", "n1"]], witnesses)
    assert edges == [
        {
            "src_ref": "n0", "dst_ref": "n1", "edge_type": "causal",
            "label": f"causal: {S4_EP1_A[:40]} → fix checkout webhook retry",
            "witness_refs": ["w2"], "confidence": 0.7,
        }
    ]  # fmt: skip
    # 时间顺序决定 src/dst
    assert propose_edges(nodes, [["n1", "n0"]], witnesses) == edges  # time order decides src/dst
    edge_user = prompts.build_edge_prompt(
        nodes=nodes, pairs=[["n0", "n1"], ["n0", "n0"], ["n0", "n9"]], witnesses=witnesses
    )
    assert rule_llm.generate_json("edge_proposal", system=prompts.EDGE_PROPOSAL_PROMPT, user=edge_user) == edges
    assert edges[0]["edge_type"] in EdgeType.values()
    obs_nodes = [
        {"ref": "o0", "kind": "observation", "a": "User live in Paris.", "subject": "user", "predicate": "lives_in", "objects": ["Paris"], "time": "2026-05-01T09:00:00+00:00", "anchor_refs": ["wa"], "r": "unknown"},
        {"ref": "o1", "kind": "observation", "a": "User live in Berlin.", "subject": "user", "predicate": "lives_in", "objects": ["Berlin"], "time": "2026-04-01T09:00:00+00:00", "anchor_refs": ["wb"], "r": "unknown"},
        {"ref": "e0", "kind": "ep", "u": "retry.ts flaked", "a": "patched retry.ts", "y": "billing_e2e failed", "r": "negative", "time": "2026-06-15T10:04:00+00:00", "anchor_refs": ["wc"], "topics": ["engineering"]},
        {"ref": "e1", "kind": "ep", "u": "retry.ts billing_e2e failed", "a": "patched retry.ts again", "y": "billing_e2e passed", "r": "positive", "time": "2026-06-15T10:14:00+00:00", "anchor_refs": ["wd"], "topics": ["engineering"]},
        {"ref": "e2", "kind": "ep", "u": "unrelated", "b": "", "a": "deployed retry.ts", "y": "green", "r": "positive", "time": "2026-06-15T11:00:00+00:00", "anchor_refs": [], "topics": []},
    ]  # fmt: skip
    proposals = propose_edges(obs_nodes, [["o0", "o1"], ["e0", "e1"], ["e1", "e2"], ["e2", "e0"]], [])
    kinds = {(e["src_ref"], e["dst_ref"]): e for e in proposals}
    assert kinds[("o1", "o0")]["edge_type"] == "conflict" and kinds[("o1", "o0")]["witness_refs"] == ["wa", "wb"]
    assert kinds[("e0", "e1")]["edge_type"] == "causal" and kinds[("e0", "e1")]["confidence"] == 0.9
    # 没有锚点引用 → 没有见证 → 不生成边
    assert ("e1", "e2") not in kinds and ("e0", "e2") not in kinds  # no anchor ref → no witness → no edge

    support = _s4_support_nodes()
    iep = coarsen_rule(list(reversed(support)), [], [], "L2_meso")
    assert iep["slots"] == {
        "u": "fix checkout webhook retry", "b": f"{S4_CONSTRAINT} ; {S4_DERIVED}", "a": f"{S4_EP1_A} → {S4_EP2_A}",
        "y": "billing_e2e passed", "r": "positive",
        "kappa": {"constraints": [S4_CONSTRAINT, S4_DERIVED], "visibility": "private"},
    }  # fmt: skip
    assert iep["slot_support"] == {
        "u": ["n0"],
        "b": ["n0", "n1"],
        "a": ["n0", "n1"],
        "y": ["n1"],
        "r": ["n1"],
        "kappa": ["n0", "n1"],
    }
    assert iep["invariants"] == [
        {"text": S4_CONSTRAINT, "kind": "constraint", "support_refs": ["n0", "n1"]},
        {"text": f"do not: {S4_EP1_A}", "kind": "failed_path", "support_refs": ["n0"]},
    ]
    assert iep["residuals"] == []
    coarsen_user = prompts.build_coarsen_prompt(
        support_nodes=support,
        internal_edges=[{"ref": "x0", "src": "n0", "dst": "n1", "type": "causal"}],
        boundary_edges=[],
        target_scale="L2_meso",
    )
    assert rule_llm.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user=coarsen_user) == iep
    models.Slots(**iep["slots"])
    l3_nodes = [
        {"ref": "n0", "kind": "interface_ep", "u": "fix checkout webhook retry", "b": S4_CONSTRAINT, "a": f"{S4_EP1_A} → {S4_EP2_A}", "y": "billing_e2e passed", "r": "positive", "constraints": [S4_CONSTRAINT], "time": "2026-06-15T10:04:00+00:00", "anchor_refs": ["w0"]},
        {"ref": "n1", "kind": "interface_ep", "u": "stabilize checkout", "b": S4_CONSTRAINT, "a": "patch src/billing/checkout.ts: add guard", "y": "", "r": "unknown", "constraints": [S4_CONSTRAINT], "time": "2026-06-16T10:04:00+00:00", "anchor_refs": ["w4"], "visibility": "team"},
        {"ref": "n2", "kind": "observation", "u": "", "b": "", "a": "noise", "y": "", "r": "unknown", "constraints": [], "time": None, "anchor_refs": ["w5"]},
    ]  # fmt: skip
    l3 = coarsen_rule(
        l3_nodes,
        [{"ref": "x0", "src": "n0", "dst": "n1", "type": "conflict"}],
        [{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type": "use"}],
        "L3_macro",
    )
    assert l3["slots"]["u"] == "fix checkout webhook retry" and l3["slots"]["r"] == "unknown"
    assert l3["slots"]["a"] == f"{S4_EP1_A} → {S4_EP2_A} → patch src/billing/checkout.ts: add guard"
    assert (
        l3["slots"]["y"] == "billing_e2e passed"
        and l3["slot_support"]["y"] == ["n0"]
        and l3["slot_support"]["a"] == ["n0", "n1"]
    )
    assert l3["slots"]["kappa"] == {"constraints": [S4_CONSTRAINT], "visibility": "private"}
    assert l3["invariants"] == [{"text": S4_CONSTRAINT, "kind": "constraint", "support_refs": ["n0", "n1"]}]
    assert l3["residuals"] == [
        {"text": "outcome of n1 unknown", "kind": "missing_outcome", "support_refs": ["n1"]},
        {"text": "unresolved conflict between n0 and n1", "kind": "unresolved_conflict", "support_refs": ["n0", "n1"]},
    ]
    assert all(i["kind"] in InvariantKind.values() for i in l3["invariants"]) and all(
        r["kind"] in ResidualKind.values() for r in l3["residuals"]
    )
    assert coarsen_rule([l3_nodes[2]], [], [], "L2_meso") is None
    empty_user = prompts.build_coarsen_prompt(
        support_nodes=[l3_nodes[2]], internal_edges=[], boundary_edges=[], target_scale="L2_meso"
    )
    assert rule_llm.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user=empty_user) == []
    negative_last = coarsen_rule([support[1], {**support[0], "time": "2026-06-15T10:24:00+00:00"}], [], [], "L2_meso")
    assert negative_last["slots"]["r"] == "negative" and negative_last["slot_support"]["r"] == ["n0"]
    assert negative_last["slots"]["y"] == "billing_e2e failed: subscription_state_regression"
    assert (
        all(o in ObservationType.values() for o in ObservationType.values())
        and Relation.supersession in Relation.values()
    )
    assert Sentiment.positive in Sentiment.values()


def test_rule_llm_procedural_summary_and_generate_text(rule_llm):
    """procedural_summary yields the Mem0 markdown skeleton through generate_text; blank replies stay blank.
    procedural_summary 经 generate_text 产出 Mem0 markdown 骨架；空回复保持为空。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

    Output / 输出：
        Assertions over generate_text('procedural_summary') and build_procedural_messages.
        对 generate_text('procedural_summary') 与 build_procedural_messages 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    conversation = [
        {"role": "user", "content": "fix checkout webhook retry"},
        {"role": "assistant", "content": "I patched retry.ts to add backoff."},
        {"role": "tool", "content": "billing_e2e failed: subscription_state_regression"},
        {"role": "assistant", "content": "I reverted the patch and added an idempotency key; billing_e2e passed."},
    ]
    markdown = rule_llm.generate_text(
        "procedural_summary", messages=prompts.build_procedural_messages(conversation, None)
    )
    for heading in prompts.PROCEDURAL_SUMMARY_HEADINGS:
        assert heading in markdown
    assert "**Task Objective**: fix checkout webhook retry" in markdown
    assert "**Progress Status**: completed" in markdown
    assert "1. **Agent Action**: I patched retry.ts to add backoff." in markdown
    assert "   **Action Result**: billing_e2e failed: subscription_state_regression" in markdown
    assert "2. **Agent Action**: I reverted the patch and added an idempotency key; billing_e2e passed." in markdown
    assert "**Errors & Challenges**: billing_e2e failed: subscription_state_regression" in markdown
    assert "**Key Findings**: I reverted the patch and added an idempotency key; billing_e2e passed." in markdown
    assert markdown.startswith("## Summary of the agent's execution history\n\n**Task Objective**")
    assert rule_llm.calls[-1]["task"] == "procedural_summary" and "messages" in rule_llm.calls[-1]["sections"]
    blocked = rule_llm.generate_text(
        "procedural_summary", messages=prompts.build_procedural_messages(conversation[:3], None)
    )
    assert "**Progress Status**: blocked" in blocked and "**Key Findings**: n/a" in blocked
    idle = rule_llm.generate_text(
        "procedural_summary", messages=prompts.build_procedural_messages([{"role": "user", "content": "hello"}], None)
    )
    assert "**Progress Status**: in progress" in idle and "1. **Agent Action**: n/a" in idle
    assert rule_llm.identity == "rule@v1" and LlmFactory.create("rule").identity == "rule@v1"


# ---------------------------------------------------------------------------
# 20. Unknown task + RAISE hook
# 20. 未知任务 + RAISE 钩子
# ---------------------------------------------------------------------------
def test_rule_llm_unknown_task_and_raise_hook(rule_llm):
    """A system prompt without a known HMS_TASK marker → LLM_002; a `RAISE:` line in the user prompt → LLM_001.
    没有已知 HMS_TASK 标记的系统提示词 → LLM_002；用户提示词中的 `RAISE:` 行 → LLM_001。

    Input / 输入：
        rule_llm: A RuleLLM.
        rule_llm：一个 RuleLLM。

    Output / 输出：
        Assertions over RuleLLM.generate_response and LLMBase.generate_json error codes.
        对 RuleLLM.generate_response 与 LLMBase.generate_json 错误码的断言。

    State effect / 状态影响：
        None.
        无。
    """
    with pytest.raises(LLMError) as info:
        rule_llm.generate_response(
            [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
        )
    assert info.value.error_code == "LLM_002" and info.value.details["provider"] == "rule@v1"
    with pytest.raises(LLMError) as info:
        rule_llm.generate_response(
            [{"role": "system", "content": "HMS_TASK: nonsense\n# ROLE"}, {"role": "user", "content": "x"}]
        )
    assert info.value.error_code == "LLM_002" and info.value.details["task"] == "nonsense"
    with pytest.raises(LLMError) as info:
        rule_llm.generate_json("observation_extraction", system="no marker here", user="x")
    assert info.value.error_code == "LLM_002" and isinstance(info.value.__cause__, LLMError)
    with pytest.raises(LLMError) as info:
        rule_llm.generate_response([{"role": "user", "content": "no system message at all"}])
    assert info.value.error_code == "LLM_002"
    user = prompts.build_observation_prompt(
        existing_memories=[], new_messages=[{"idx": 0, "role": "user", "text": "RAISE: transport down", "timestamp": None}],
        last_messages=[], observation_date="2026-09-02", current_date="2026-09-02",
    )  # fmt: skip
    hooked = "## Last k Messages\nRAISE: transport down\n" + user
    with pytest.raises(LLMError) as info:
        rule_llm.generate_response(
            [{"role": "system", "content": prompts.OBSERVATION_EXTRACTION_PROMPT}, {"role": "user", "content": hooked}]
        )
    assert info.value.error_code == "LLM_001" and info.value.message == "transport down"
    with pytest.raises(LLMError) as info:
        rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=hooked)
    assert info.value.error_code == "LLM_001" and info.value.details == {
        "task": "observation_extraction",
        "provider": "rule@v1",
    }
    # 被拒绝的提示词不会被记录
    assert rule_llm.calls == []  # nothing recorded for rejected prompts
    fine = rule_llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user=user)
    assert fine == [] and rule_llm.calls[-1]["task"] == "observation_extraction"


# ---------------------------------------------------------------------------
# 21. ScriptedLLM
# 21. ScriptedLLM（脚本化 LLM）
# ---------------------------------------------------------------------------
def test_scripted_llm_queue_and_cause(scripted_llm):
    """dict / str / Exception / callable responses replay in order; exhaustion → LLM_003; generate_json keeps the cause.
    dict / str / 异常 / 可调用响应按序回放；耗尽 → LLM_003；generate_json 保留原因。

    Input / 输入：
        scripted_llm: Factory `responses -> ScriptedLLM`.
        scripted_llm：工厂 `responses -> ScriptedLLM`。

    Output / 输出：
        Assertions over ScriptedLLM.generate_response / push / calls and LLMBase.generate_json.
        对 ScriptedLLM.generate_response / push / calls 与 LLMBase.generate_json 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    seen: list[list[dict]] = []

    def _reply(messages: list[dict]) -> dict:
        seen.append(messages)
        return {"observations": [{"id": "0"}]}

    llm = scripted_llm([{"decisions": []}, "plain text", RuntimeError("boom"), _reply, LLMError("queued", "LLM_009")])
    assert llm.identity == "scripted@v1" and isinstance(llm, LLMBase)
    system = prompts.REVISION_DECISION_PROMPT
    assert llm.generate_json("revision_decision", system=system, user="u") == []
    assert (
        llm.generate_response([{"role": "user", "content": "x"}], response_format={"type": "json_object"})
        == "plain text"
    )
    with pytest.raises(LLMError) as info:
        llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u")
    assert info.value.error_code == "LLM_001" and isinstance(info.value.__cause__, RuntimeError)
    assert info.value.details == {"task": "observation_extraction", "provider": "scripted@v1"}
    assert llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u") == [
        {"id": "0"}
    ]
    assert seen and seen[0][1] == {"role": "user", "content": "u"}
    with pytest.raises(LLMError) as info:
        llm.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user="u")
    assert info.value.error_code == "LLM_009" and isinstance(info.value.__cause__, LLMError)
    with pytest.raises(LLMError) as info:
        llm.generate_response([{"role": "system", "content": system}])
    assert info.value.error_code == "LLM_003" and info.value.suggestion == "queue more responses"
    with pytest.raises(LLMError) as info:
        llm.generate_json("revision_decision", system=system, user="u")
    assert info.value.error_code == "LLM_003" and isinstance(info.value.__cause__, LLMError)
    assert [c["task"] for c in llm.calls] == [
        "revision_decision",
        None,
        "observation_extraction",
        "observation_extraction",
        "coarsen",
        "revision_decision",
        "revision_decision",
    ]
    assert (
        llm.calls[0]["kwargs"] == {"response_format": {"type": "json_object"}}
        and llm.calls[0]["messages"][0]["role"] == "system"
    )
    llm.push("late", ValueError)
    assert llm.generate_text("procedural_summary", messages=[{"role": "user", "content": "x"}]) == "late"
    with pytest.raises(LLMError) as info:
        llm.generate_text("procedural_summary", messages=[{"role": "user", "content": "x"}])
    assert isinstance(info.value.__cause__, ValueError)
    looping = ScriptedLLM({"responses": ["a", "b"], "loop": True})
    assert [looping.generate_response([]) for _ in range(5)] == ["a", "b", "a", "b", "a"]
    quiet = ScriptedLLM({"responses": [], "raise_when_empty": False})
    assert (
        quiet.generate_response([]) == ""
        and quiet.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user="u") == []
    )
    assert quiet.generate_text("procedural_summary", messages=[]) == "" and len(quiet.calls) == 3
    fenced = ScriptedLLM({"responses": ['```json\n{"edges": [{"src_ref": "n0"}]}\n```', "```md\n# hi\n```"]})
    assert fenced.generate_json("edge_proposal", system=prompts.EDGE_PROPOSAL_PROMPT, user="u") == [{"src_ref": "n0"}]
    assert fenced.generate_text("procedural_summary", messages=[]) == "# hi"


# ---------------------------------------------------------------------------
# 22. Optional providers behind import guards
# 22. import 守卫背后的可选提供方
# ---------------------------------------------------------------------------
def test_optional_providers_guarded(monkeypatch):
    """Missing SDK → DependencyError(DEPS_001) with a pip hint; fake SDKs receive JSON response_format / system split.
    缺少 SDK → 带 pip 提示的 DependencyError(DEPS_001)；替身 SDK 收到 JSON response_format / 系统提示拆分。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over LlmFactory / EmbedderFactory.create for openai and anthropic and the provider classes.
        对 openai 与 anthropic 的 LlmFactory / EmbedderFactory.create 及提供者类的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by monkeypatch).
        临时替换 sys.modules 条目（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    try:
        for package, provider, factory in (
            ("openai", "openai", LlmFactory),
            ("anthropic", "anthropic", LlmFactory),
            ("openai", "openai", EmbedderFactory),
        ):
            _purge_guarded_modules(monkeypatch)
            monkeypatch.setitem(sys.modules, package, None)
            with pytest.raises(DependencyError) as info:
                factory.create(provider)
            assert info.value.error_code == "DEPS_001" and info.value.details == {"package": package, "extra": package}
            assert info.value.suggestion == f"pip install 'hms-elite[{package}]'" and isinstance(
                info.value.__cause__, ImportError
            )
            assert factory.kind in {"llm", "embedder"}

        calls: list[dict] = []
        _purge_guarded_modules(monkeypatch)
        monkeypatch.setitem(sys.modules, "openai", _fake_openai(calls))
        llm = LlmFactory.create("openai", {"api_key": "k", "temperature": 0.3})
        assert llm.identity == "openai:gpt-5-mini" and llm.config.api_key == "k" and calls[0]["api"] == "client"
        assert calls[0]["api_key"] == "k" and calls[0]["base_url"] is None
        out = llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u")
        assert out == [] and calls[-1]["api"] == "chat" and calls[-1]["response_format"] == {"type": "json_object"}
        assert (
            calls[-1]["model"] == "gpt-5-mini"
            and calls[-1]["temperature"] == 0.3
            and calls[-1]["messages"][0]["role"] == "system"
        )
        assert "tools" not in calls[-1]
        text = llm.generate_text("procedural_summary", messages=[{"role": "user", "content": "x"}])
        assert text == '{"observations": []}' and "response_format" not in calls[-1]
        with_tools = llm.generate_response(
            [{"role": "user", "content": "x"}], tools=[{"type": "function"}], tool_choice="auto"
        )
        assert (
            with_tools == {"content": '{"observations": []}', "tool_calls": []} and calls[-1]["tool_choice"] == "auto"
        )
        embedder = EmbedderFactory.create("openai", {"embedding_dims": 4, "api_key": "k"})
        assert embedder.identity == "openai:text-embedding-3-small@4" and embedder.dims == 4
        assert embedder.embed("a\nb") == [0.5, 0.5, 0.5, 0.5]
        assert calls[-1]["api"] == "embeddings" and calls[-1] == {
            "api": "embeddings",
            "model": "text-embedding-3-small",
            "input": "a b",
            "dimensions": 4,
        }
        assert embedder.embed_batch(["x", "y"]) == [[0.5] * 4, [0.5] * 4]
        with pytest.raises(ConfigError) as info:
            LlmFactory.create("openai", {"modle": "gpt"})
        assert "Did you mean 'model'?" in info.value.message

        acalls: list[dict] = []
        _purge_guarded_modules(monkeypatch)
        monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(acalls))
        allm = LlmFactory.create("anthropic", {"api_key": "k2", "max_tokens": 99})
        assert allm.identity == "anthropic:claude-sonnet-4-6" and acalls[0] == {
            "api": "client",
            "api_key": "k2",
            "base_url": None,
        }
        messages = [{"role": "system", "content": prompts.COARSEN_PROMPT}, {"role": "user", "content": "u"}]
        assert allm.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user="u") == []
        sent = acalls[-1]
        assert (
            sent["system"] == prompts.COARSEN_PROMPT
            and sent["max_tokens"] == 99
            and sent["model"] == "claude-sonnet-4-6"
        )
        assert sent["messages"] == [{"role": "user", "content": "u\nRespond with JSON only."}]
        # 调用方传入的列表从不被修改
        assert messages[1]["content"] == "u"  # the caller's list is never mutated
        assert allm.generate_response(messages) == '{"observations": []}'
        assert acalls[-1]["messages"] == [{"role": "user", "content": "u"}] and all(
            m["role"] != "system" for m in acalls[-1]["messages"]
        )
    finally:
        _purge_guarded_modules(monkeypatch)


# ---------------------------------------------------------------------------
# 22b. Anthropic sampling parameters follow the installed SDK signature (anthropic>=1.0 removed `temperature`)
# 22b. Anthropic 采样参数跟随已安装 SDK 的签名（anthropic>=1.0 移除了 `temperature`）
# ---------------------------------------------------------------------------
def test_anthropic_temperature_follows_sdk_signature(monkeypatch):
    """A 1.x-shaped Messages.create (no `temperature`, no **kwargs) is called without it; a 0.x-shaped one still receives config.temperature.
    1.x 形状的 Messages.create（无 `temperature`、无 **kwargs）调用时不带该参数；0.x 形状的仍收到 config.temperature。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over the kwargs each fake SDK received from generate_json.
        对每个替身 SDK 从 generate_json 收到的参数的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by monkeypatch).
        临时替换 sys.modules 条目（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    try:
        for sampling in (False, True):
            calls: list[dict] = []
            _purge_guarded_modules(monkeypatch)
            monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_signature(calls, sampling=sampling))
            llm = LlmFactory.create("anthropic", {"api_key": "k", "temperature": 0.3, "max_tokens": 77})
            # Without the signature check the 1.x shape raises TypeError before any request → LLMError(LLM_001).
            # 没有签名检查时，1.x 形状会在任何请求之前抛出 TypeError → LLMError(LLM_001)。
            assert llm.generate_json("coarsen", system=prompts.COARSEN_PROMPT, user="u") == []
            sent = calls[-1]
            assert sent["api"] == "messages" and sent["model"] == "claude-sonnet-4-6" and sent["max_tokens"] == 77
            assert sent["system"] == prompts.COARSEN_PROMPT
            assert sent["messages"] == [{"role": "user", "content": "u\nRespond with JSON only."}]
            if sampling:
                assert sent["temperature"] == 0.3
            else:
                assert "temperature" not in sent
    finally:
        _purge_guarded_modules(monkeypatch)


# ---------------------------------------------------------------------------
# 23. Factory + registry
# 23. 工厂 + 注册表
# ---------------------------------------------------------------------------
def test_factory_registry_single_source_of_truth(store):
    """register_provider extends both LlmConfig validation and LlmFactory.create; caller dicts are never mutated.
    register_provider 同时扩展 LlmConfig 校验与 LlmFactory.create；调用方字典从不被修改。

    Input / 输入：
        store: In-memory StateStore for IndexFactory.
        store：供 IndexFactory 使用的内存 StateStore。

    Output / 输出：
        Assertions over register_provider / LlmFactory / EmbedderFactory / IndexFactory / load_class.
        对 register_provider / LlmFactory / EmbedderFactory / IndexFactory / load_class 的断言。

    State effect / 状态影响：
        Adds and removes the 'x' llm provider in REGISTRY.
        在 REGISTRY 中添加并移除 'x' llm 提供者。
    """
    assert LlmFactory.get_supported_providers() == ["anthropic", "deepseek", "ollama", "openai", "rule", "scripted"]
    assert EmbedderFactory.get_supported_providers() == [
        "ark",
        "fastembed",
        "hash",
        "ollama",
        "openai",
    ] and IndexFactory.get_supported_providers() == ["memory", "milvus", "sqlite"]
    with pytest.raises(ConfigError) as info:
        LlmFactory.create("bogus")
    assert info.value.error_code == "CONFIG_002" and info.value.details == {
        "kind": "llm",
        "provider": "bogus",
        "supported": ["anthropic", "deepseek", "ollama", "openai", "rule", "scripted"],
    }
    with pytest.raises(Exception):
        LlmConfig(provider="x")
    try:
        register_provider("llm", "x", "tests.test_providers.FakeLLM", "tests.test_providers.FakeLlmConfig")
        assert LlmConfig(provider="x").provider == "x" and "x" in LlmFactory.get_supported_providers()
        config = {"greeting": "yo", "model": "m"}
        snapshot = dict(config)
        llm = LlmFactory.create("x", config, temperature=0.5)
        assert isinstance(llm, FakeLLM) and llm.config.greeting == "yo" and llm.config.temperature == 0.5
        assert config == snapshot and llm.identity == "x:m"
        assert llm.generate_json("observation_extraction", system=prompts.OBSERVATION_EXTRACTION_PROMPT, user="u") == []
        LlmFactory.register_provider("x", "tests.test_providers.FakeLLM", "tests.test_providers.FakeLlmConfig")
        assert LlmFactory.create("x", FakeLlmConfig(greeting="hey")).config.greeting == "hey"
        assert LlmFactory.create("x", None).config.greeting == "hi"
        with pytest.raises(ConfigError) as info:
            LlmFactory.create("x", {"greting": "yo"})
        assert info.value.error_code == "CONFIG_001" and "Did you mean 'greeting'?" in info.value.message
        with pytest.raises(ConfigError) as info:
            LlmFactory.create("x", 42)
        assert info.value.error_code == "CONFIG_001"
    finally:
        REGISTRY["llm"].pop("x", None)
    assert "x" not in LlmFactory.get_supported_providers()
    with pytest.raises(ConfigError) as info:
        register_provider("gizmo", "x", "a.B", "a.C")
    assert info.value.error_code == "CONFIG_002"
    assert load_class("hms_elite.llms.rule.RuleLLM") is RuleLLM
    with pytest.raises(ImportError):
        load_class("hms_elite.nonexistent.Thing")
    rule = LlmFactory.create("rule", {"language": "zh"})
    assert isinstance(rule, RuleLLM) and rule.config.language == "zh"
    scripted = LlmFactory.create("scripted", {"responses": ["a"]})
    assert isinstance(scripted, ScriptedLLM) and scripted.generate_response([]) == "a"
    hashed = EmbedderFactory.create("hash", {"embedding_dims": 16})
    assert isinstance(hashed, HashEmbedder) and hashed.dims == 16 and len(hashed.embed("x")) == 16
    sqlite_index = IndexFactory.create("sqlite", {"use_numpy": "never"}, store=store, dims=16)
    memory_index = IndexFactory.create("memory", None, store=store, dims=16)
    assert isinstance(sqlite_index, SQLiteIndex) and isinstance(memory_index, MemoryIndex)
    assert sqlite_index.dims == memory_index.dims == 16 and sqlite_index.store is store
    assert IndexFactory.create("memory", {}, dims=8).info() == {
        "provider": "memory",
        "dims": 8,
        "count": 0,
        "numpy": IndexFactory.create("memory", {}, dims=8).uses_numpy,
    }
    with pytest.raises(ConfigError) as info:
        IndexFactory.create("qdrant", {"host": "localhost"}, store=store, dims=16)
    assert info.value.error_code == "CONFIG_002" and info.value.details["supported"] == ["memory", "milvus", "sqlite"]


# ---------------------------------------------------------------------------
# Group V: the describe_image vision hook
# V 组：describe_image 视觉钩子
# ---------------------------------------------------------------------------
_V_PNG = b"\x89PNG\r\n\x1a\n" + b"vision-bytes" * 4
_V_URI = "data:image/png;base64," + base64.b64encode(_V_PNG).decode("ascii")
_V_URL = "https://img.example/photo.jpg"


def test_describe_image_default_rule_and_scripted():
    """LLMBase / rule return None (no vision); scripted returns its configured description and records the call.
    LLMBase / rule 返回 None（无视觉）；scripted 返回配置的描述并记录调用。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over describe_image results and ScriptedLLM.calls.
        对 describe_image 结果与 ScriptedLLM.calls 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert FakeLLM().describe_image(_V_URI, detail="low") is None
    assert RuleLLM().describe_image(_V_URL) is None and RuleLLM().calls == []
    plain = ScriptedLLM({"responses": []})
    assert plain.describe_image(_V_URI) is None
    assert plain.calls == [{"task": "describe_image", "messages": [], "kwargs": {"url": _V_URI, "detail": None}}]
    vision = ScriptedLLM({"responses": [], "image_description": "a cat on a sofa"})
    assert vision.describe_image(_V_URL, detail="high") == "a cat on a sofa"
    assert vision.calls[-1]["kwargs"] == {"url": _V_URL, "detail": "high"}
    # The hook is keyword-only for `detail` on every provider (the pipeline always passes it by name).
    # 每个提供者的钩子 `detail` 都是仅关键字参数（流水线总是按名传递）。
    for cls in (LLMBase, RuleLLM, ScriptedLLM):
        assert str(inspect.signature(cls.describe_image)) == "(self, url_or_data_uri: 'str', *, detail: 'str | None' = None) -> 'str | None'"


def test_openai_describe_image_sends_image_url_part(monkeypatch):
    """OpenAI describe_image posts one user message with [text prompt, image_url part]; data URIs pass through; detail only when given.
    OpenAI 的 describe_image 发送一条含 [文本提示, image_url 部件] 的 user 消息；data URI 原样透传；仅在给定时带 detail。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over the chat.completions.create kwargs the fake SDK received.
        对替身 SDK 收到的 chat.completions.create 参数的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by monkeypatch).
        临时替换 sys.modules 条目（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    try:
        calls: list[dict] = []
        _purge_guarded_modules(monkeypatch)
        monkeypatch.setitem(sys.modules, "openai", _fake_openai(calls))
        llm = LlmFactory.create("openai", {"api_key": "k", "temperature": 0.2, "max_tokens": 50})
        assert llm.describe_image(_V_URI, detail="high") == '{"observations": []}'
        sent = calls[-1]
        assert sent["api"] == "chat" and sent["model"] == "gpt-5-mini" and sent["temperature"] == 0.2
        assert sent["max_tokens"] == 50 and "response_format" not in sent and "tools" not in sent
        assert sent["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": _V_URI, "detail": "high"}},
                ],
            }
        ]
        assert llm.describe_image(_V_URL) == '{"observations": []}'
        assert calls[-1]["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": _V_URL}}
    finally:
        _purge_guarded_modules(monkeypatch)


def test_anthropic_describe_image_sends_image_block(monkeypatch):
    """Anthropic describe_image sends an `image` block (base64 source for data URIs, url source for http) then the prompt; unusable refs → None without a call.
    Anthropic 的 describe_image 发送 `image` 块（data URI → base64 来源，http → url 来源）再接提示词；不可用引用 → None 且不调用。

    Input / 输入：
        monkeypatch: pytest monkeypatch (sys.modules patching).
        monkeypatch：pytest monkeypatch（打补丁 sys.modules）。

    Output / 输出：
        Assertions over the messages.create kwargs the fake SDK received.
        对替身 SDK 收到的 messages.create 参数的断言。

    State effect / 状态影响：
        Temporarily replaces sys.modules entries (restored by monkeypatch).
        临时替换 sys.modules 条目（由 monkeypatch 恢复）。
    """
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    try:
        calls: list[dict] = []
        _purge_guarded_modules(monkeypatch)
        monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(calls))
        llm = LlmFactory.create("anthropic", {"api_key": "k", "max_tokens": 64, "temperature": 0.4})
        assert llm.describe_image(_V_URI, detail="low") == '{"observations": []}'
        sent = calls[-1]
        assert sent["api"] == "messages" and sent["model"] == "claude-sonnet-4-6" and sent["max_tokens"] == 64
        # The fake's create(**kwargs) accepts temperature, so the 0.x path forwards it; no system prompt is sent.
        # 替身的 create(**kwargs) 接受 temperature，因此 0.x 路径转发它；不发送 system 提示词。
        assert sent["temperature"] == 0.4 and "system" not in sent and "tools" not in sent
        assert sent["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(_V_PNG).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                ],
            }
        ]
        assert llm.describe_image(_V_URL) == '{"observations": []}'
        assert calls[-1]["messages"][0]["content"][0] == {"type": "image", "source": {"type": "url", "url": _V_URL}}
        before = len(calls)
        assert llm.describe_image("ftp://nope/x.png") is None and llm.describe_image("data:text/plain,hi") is None
        assert len(calls) == before
    finally:
        _purge_guarded_modules(monkeypatch)


def test_llm_config_vision_switch_and_mem0_placement():
    """LlmConfig carries enable_vision / vision_details; Mem0's placement inside llm.config is lifted to the envelope.
    LlmConfig 携带 enable_vision / vision_details；Mem0 放在 llm.config 里的写法被提升到信封。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over HMSConfig.from_dict / LlmConfig for both placements and the defaults.
        对两种写法与默认值下 HMSConfig.from_dict / LlmConfig 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    defaults = HMSConfig.from_dict({})
    assert defaults.llm.enable_vision is False and defaults.llm.vision_details is None
    lifted = HMSConfig.from_dict(
        {"llm": {"provider": "scripted", "config": {"enable_vision": True, "vision_details": "low", "loop": True}}}
    )
    assert lifted.llm.enable_vision is True and lifted.llm.vision_details == "low"
    # The provider config no longer sees the lifted keys, so `extra="forbid"` provider configs still construct.
    # 提供者配置不再看到被提升的键，因此 `extra="forbid"` 的提供者配置仍可构造。
    assert lifted.llm.config == {"loop": True} and LlmFactory.create("scripted", lifted.llm.config).config.loop is True
    explicit = LlmConfig(provider="rule", enable_vision=True, config={"enable_vision": False, "vision_details": "high"})
    assert explicit.enable_vision is True and explicit.vision_details == "high" and explicit.config == {}
    with pytest.raises(ConfigError) as info:
        HMSConfig.from_dict({"llm": {"provider": "rule", "enable_visoin": True}})
    assert info.value.error_code == "CONFIG_001" and "Did you mean 'enable_vision'?" in info.value.message


# ---------------------------------------------------------------------------
# 17c. Bounded search (matmul selection + exact rescoring) is bit-identical to rank_hits; contract extensions
# 17c. 有界搜索（矩阵乘挑选 + 精确重算）与 rank_hits 逐位相同；契约扩展
# ---------------------------------------------------------------------------
def _seeded_unit_vectors(count: int, dims: int, seed: int, *, negative_every: int = 0) -> list[list[float]]:
    """Deterministic unit vectors; every `negative_every`-th one is pushed to a negative first component.
    确定性单位向量；每第 `negative_every` 个被推到负的首分量。
    """
    import random

    rng = random.Random(seed)
    out: list[list[float]] = []
    for i in range(count):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dims)]
        if negative_every and i % negative_every == 0:
            raw[0] = -abs(raw[0]) - 2.0
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        out.append([x / norm for x in raw])
    return out


def test_bounded_search_is_bit_identical_to_rank_hits(store):
    """sqlite (numpy auto / never) and memory searches equal rank_hits over the same vectors for every top_k, bit for bit.
    sqlite（numpy auto / never）与 memory 的搜索对每个 top_k 都与同一批向量上的 rank_hits 逐位相等。

    Input / 输入：
        store: In-memory StateStore (atom_vectors host).
        store：内存 StateStore（atom_vectors 宿主）。

    Output / 输出：
        Equality assertions over 24 queries × 7 top_k values × full / subset candidate sets, including the zero-score
        tail (a third of the vectors have a negative cosine with e1) and top_k >= n; select_top_k asserted directly.
        对 24 个查询 × 7 个 top_k × 全集 / 子集候选的相等断言，含零分尾部（三分之一向量与 e1 的余弦为负）
        与 top_k >= n；并直接断言 select_top_k。

    State effect / 状态影响：
        Writes atom_vectors in the in-memory store.
        写入内存 store 的 atom_vectors。
    """
    from hms_elite.index.base import numpy_module, rank_hits, select_top_k

    ids = [f"atm_{i:06d}" for i in range(400)]
    _seed_atoms(store, {atom_id: f"vector {atom_id}" for atom_id in ids})
    items = list(zip(ids, _seeded_unit_vectors(400, 8, seed=3, negative_every=3)))
    auto = SQLiteIndex({"use_numpy": "auto"}, store=store, dims=8)
    plain = SQLiteIndex({"use_numpy": "never"}, store=store, dims=8)
    memory_index = MemoryIndex(dims=8)
    auto.upsert_batch(items, embedder="hash@8")
    memory_index.upsert_batch(items, embedder="hash@8", attrs={ids[0]: {"archived": True}})
    stored = store.load_vectors(None)
    subset = ids[100:300] + ["atm_missing", ids[150]]
    queries = _seeded_unit_vectors(24, 8, seed=11)
    queries[0] = [1.0] + [0.0] * 7  # 133 positives only: top_k=300 reaches the zero tail / 只有 133 个正分：top_k=300 进入零分尾部
    queries[1] = [-1.0] + [0.0] * 7  # 267 positives / 267 个正分
    for query in queries:
        for top_k in (1, 5, 50, 134, 300, 399, 400, 500):
            expected = rank_hits(query, stored, top_k=top_k, use_numpy=True)
            assert auto.search(query, candidate_ids=None, top_k=top_k) == expected
            assert memory_index.search(query, candidate_ids=None, top_k=top_k) == expected
            assert plain.search(query, candidate_ids=None, top_k=top_k) == rank_hits(
                query, stored, top_k=top_k, use_numpy=False
            )
            wanted = {i: stored[i] for i in dict.fromkeys(subset) if i in stored}
            expected_subset = rank_hits(query, wanted, top_k=top_k, use_numpy=True)
            assert auto.search(query, candidate_ids=subset, top_k=top_k) == expected_subset
            assert memory_index.search(query, candidate_ids=subset, top_k=top_k) == expected_subset
            assert len(expected) == min(top_k, 400) and len(expected_subset) == min(top_k, 200)
    np = numpy_module()
    if np is None:
        # The bounded numpy path is the subject here; without numpy only the exhaustive path exists.
        # 这里考察的是有界 numpy 路径；没有 numpy 时只有穷举路径。
        pytest.skip("numpy not installed")
    assert auto.uses_numpy and not plain.uses_numpy
    matrix = np.asarray([stored[i] for i in ids], dtype="<f4")
    for top_k in (0, 1, 7, 133, 134, 200, 400, 401):
        assert select_top_k(queries[0], ids, matrix, top_k=top_k) == rank_hits(queries[0], stored, top_k=top_k)
    assert select_top_k(queries[2], [], matrix[:0], top_k=5) == []
    # The matmul only selects: scores are those of `dot` (asserted through rank_hits above), and a mixed-dims store
    # falls back to the exhaustive path instead of a malformed matrix.
    # 矩阵乘只负责挑选：分数就是 `dot` 的分数（上面已经由 rank_hits 断言），维数混杂的存储回退到穷举路径而非畸形矩阵。
    _seed_atoms(store, {"atm_short": "short"})
    auto.upsert("atm_short", [1.0, 0.0], embedder="hash@2")
    mixed = store.load_vectors(None)
    assert auto.search(queries[0], candidate_ids=None, top_k=10) == rank_hits(queries[0], mixed, top_k=10)
    memory_index.upsert("atm_short", [1.0, 0.0], embedder="hash@2")
    assert memory_index.search(queries[0], candidate_ids=None, top_k=10) == rank_hits(queries[0], mixed, top_k=10)
    # The memory matrix cache follows writes: a delete after a search changes the next result.
    # 内存矩阵缓存跟随写入：搜索之后的删除会改变下一次结果。
    top = memory_index.search(queries[3], candidate_ids=None, top_k=1)[0][0]
    memory_index.delete(top)
    assert memory_index.search(queries[3], candidate_ids=None, top_k=1)[0][0] != top


def test_index_contract_extensions_and_filter_expr(store):
    """IndexFilter / attrs / update_attrs / close / supports_filter on the core providers, and the Milvus filter grammar.
    核心提供者上的 IndexFilter / attrs / update_attrs / close / supports_filter，以及 Milvus 过滤语法。

    Input / 输入：
        store: In-memory StateStore.
        store：内存 StateStore。

    Output / 输出：
        Assertions on filter refusal (VALIDATION_000), attrs acceptance, the no-op hooks, atom_index_attrs /
        normalize_attrs, and filter_expr rendering (scope keys, validity, whitelists, time bounds, escaping).
        对过滤器拒绝（VALIDATION_000）、attrs 接受、空操作钩子、atom_index_attrs / normalize_attrs 与 filter_expr
        渲染（scope 键、有效期、白名单、时间界、转义）的断言。

    State effect / 状态影响：
        Writes atom_vectors in the in-memory store.
        写入内存 store 的 atom_vectors。
    """
    from hms_elite.exceptions import ValidationError
    from hms_elite.index import INDEX_ATTR_KEYS, IndexFilter, atom_index_attrs
    from hms_elite.index.base import IndexBase, normalize_attrs
    from hms_elite.index.milvus import filter_expr

    _seed_atoms(store, {"atm_000001": "one"})
    atom = store.get_atom("atm_000001")
    attrs = atom_index_attrs(atom)
    assert tuple(attrs) == INDEX_ATTR_KEYS and attrs["user_id"] == "alice" and attrs["agent_id"] == ""
    assert attrs["archived"] is False and attrs["valid_to"] == "" and attrs["valid_from"] == atom.valid_from
    assert attrs == store.atom_attrs(["atm_000001", "atm_none"])["atom_000001".replace("atom", "atm")]
    assert normalize_attrs({"archived": 1, "valid_to": "2026-03-01", "obs_type": None, "bogus": 1}) == {
        "archived": True,
        "valid_to": "2026-03-01T00:00:00.000000+00:00",
        "obs_type": "",
    }
    assert normalize_attrs(None) == {}
    spec = IndexFilter(scope=models.Scope(user_id="alice"), as_of="2026-06-15T09:00:00Z")
    for index in (SQLiteIndex(store=store, dims=4), MemoryIndex(dims=4)):
        # The sqlite provider pushes a filter down (numpy mask over its cache); the memory provider still refuses.
        # sqlite 提供者下推过滤器（其缓存上的 numpy 掩码）；memory 提供者仍然拒绝。
        assert index.supports_filter is isinstance(index, SQLiteIndex) and index.close() is None
        index.upsert("atm_000001", [1.0, 0.0, 0.0, 0.0], embedder="hash@4", attrs=attrs)
        index.upsert_batch([("atm_000001", [0.0, 1.0, 0.0, 0.0])], embedder="hash@4", attrs={"atm_000001": attrs})
        assert index.update_attrs("atm_000001", {"archived": True}) is None
        assert index.search([0.0, 1.0, 0.0, 0.0], candidate_ids=["atm_000001"], top_k=1) == [("atm_000001", 1.0)]
        if isinstance(index, SQLiteIndex):
            # With numpy, update_attrs overwrites the cached scalars like Milvus does: the row is gated out until
            # the scalar is set back; without numpy the filter is the SQL gate itself (gate_ids), which knows nothing
            # of a direct update_attrs. Either way the filter then equals the SQL gate.
            # 有 numpy 时 update_attrs 与 Milvus 一样覆盖缓存标量：该行被门控排除，直到标量改回；无 numpy 时过滤器
            # 就是 SQL 门控本身（gate_ids），它不知道直接的 update_attrs。两种情况下过滤器随后都等于 SQL 门控。
            if index.uses_numpy:
                assert index.search([0.0, 1.0, 0.0, 0.0], filter=spec, top_k=1) == []
            index.update_attrs("atm_000001", {"archived": False})
            assert index.search([0.0, 1.0, 0.0, 0.0], filter=spec, top_k=1) == index.search(
                [0.0, 1.0, 0.0, 0.0], candidate_ids=store.axis_ids(spec.scope, as_of=spec.as_of), top_k=1
            ) == [("atm_000001", 1.0)]
            continue
        with pytest.raises(ValidationError) as info:
            index.search([0.0, 1.0, 0.0, 0.0], filter=spec, top_k=1)
        assert info.value.error_code == "VALIDATION_000" and "axis_ids" in info.value.message
    assert IndexBase.supports_filter.fget(MemoryIndex(dims=4)) is False
    with pytest.raises(AttributeError):
        spec.as_of = "x"  # type: ignore[misc]  # frozen / 冻结
    # filter_expr: the SQL gate rendered for Milvus scalars (Scope.matches: unset keys add nothing).
    # filter_expr：为 Milvus 标量渲染的 SQL 门控（Scope.matches：未设置的键不加谓词）。
    assert filter_expr(None) == "" and filter_expr(IndexFilter(include_expired=True)) == ""
    assert filter_expr(spec) == (
        'user_id == "alice" and archived == false and valid_from <= "2026-06-15T09:00:00.000000+00:00" '
        'and (valid_to == "" or valid_to > "2026-06-15T09:00:00.000000+00:00")'
    )
    full = IndexFilter(
        scope=models.Scope(user_id="a\\b", agent_id='q"uote', run_id="r"),
        as_of="2026-06-15",
        scale_in=("state", "event"),
        obs_type_in=("preference_event",),
        time_from="2026-03-01",
        time_to="2026-03-31T23:59:59Z",
    )
    assert filter_expr(full) == (
        'user_id == "a\\\\b" and agent_id == "q\\"uote" and run_id == "r" and archived == false '
        'and valid_from <= "2026-06-15T00:00:00.000000+00:00" '
        'and (valid_to == "" or valid_to > "2026-06-15T00:00:00.000000+00:00") '
        'and scale_level in ["state", "event"] and obs_type in ["preference_event"] '
        'and time_anchor >= "2026-03-01T00:00:00.000000+00:00" and time_anchor <= "2026-03-31T23:59:59.000000+00:00"'
    )
    expired = filter_expr(IndexFilter(scope=models.Scope(run_id="r"), include_expired=True, scale_in=()))
    assert expired == 'run_id == "r"'
    # as_of defaults to the library clock: the expression carries a fixed-width UTC instant either way.
    # as_of 默认库时钟：表达式无论如何都带定宽 UTC 时刻。
    assert 'valid_from <= "20' in filter_expr(IndexFilter(scope=models.Scope(user_id="alice")))
