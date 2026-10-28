"""Tests for the hms4eye compatibility layer: signature parity, engine behaviour and field mapping.
hms4eye 兼容层的测试：签名对齐、引擎行为与字段映射。
"""

# Purpose (EN): Prove HMSEngineAdapter can replace hms4eye v0.6.1's Mem0Adapter without touching that service.
# 目的（中）：证明 HMSEngineAdapter 可以在不改动该服务的前提下替换 hms4eye v0.6.1 的 Mem0Adapter。

# Input (EN): The shared `hms` / `events` fixtures plus literal mem0-shaped calls and filter trees.
# 输入（中）：共享的 `hms` / `events` 夹具，以及字面的 mem0 形状调用与过滤树。

# Output (EN): Assertions over inspect.signature strings, returned envelopes, stage names and mapped rows.
# 输出（中）：对 inspect.signature 字符串、返回信封、阶段名与映射后行的断言。

# State effect (EN): None outside the in-memory store of the `hms` fixture and extract()'s throwaway store.
# 状态影响（中）：除 `hms` 夹具的内存存储与 extract() 的一次性存储外无副作用。

from __future__ import annotations

import contextlib
import inspect
from typing import Any

from hms_elite.hms4eye import (
    HMSEngineAdapter,
    fact_type_of,
    split_reserved,
    to_memory_unit,
    to_recall_result,
)

# Verbatim str(inspect.signature(...)) of every Mem0Adapter method in the hms4eye service
# hms4eye 服务中 Mem0Adapter 每个方法的 str(inspect.signature(...)) 原文
# (src/hms_lite/mem0_adapter.py; `from __future__ import annotations` makes every annotation a string).
# （src/hms_lite/mem0_adapter.py；`from __future__ import annotations` 使每个注解都成为字符串）。
MEM0_ADAPTER_SIGNATURES = {
    "add": (
        "(self, messages: 'Any', *, user_id: 'str', metadata: 'dict[str, Any] | None' = None, "
        "infer: 'bool' = True, prompt: 'str | None' = None) -> 'dict[str, Any]'"
    ),
    "search": (
        "(self, query: 'str', *, user_id: 'str', top_k: 'int' = 20, threshold: 'float' = 0.1, "
        "metadata_filters: 'dict[str, Any] | None' = None, explain: 'bool' = False) -> 'dict[str, Any]'"
    ),
    "extract": "(self, messages: 'list[dict[str, str]]', *, prompt: 'str | None' = None) -> 'list[str]'",
    "get_all": "(self, *, user_id: 'str', top_k: 'int | None' = None) -> 'dict[str, Any]'",
    "get": "(self, memory_id: 'str') -> 'dict[str, Any] | None'",
    "history": "(self, memory_id: 'str') -> 'list[dict[str, Any]] | dict[str, Any]'",
    "delete_all": "(self, *, user_id: 'str') -> 'Any'",
}

# The scope id hms4eye derives from (tenant_id, bank_id) before every adapter call.
# hms4eye 在每次适配器调用前由 (tenant_id, bank_id) 推导出的范围 id。
SCOPE = "tenant-a:bank-1"


def _retain_metadata(**extra: Any) -> dict[str, Any]:
    """The reserved metadata block MemoryService.retain() builds for one item.
    MemoryService.retain() 为一个条目构造的保留元数据块。
    """
    metadata: dict[str, Any] = {"_hms_tenant_id": "tenant-a", "_hms_bank_id": "bank-1"}
    metadata.update(extra)
    return metadata


class _RecordingTelemetry:
    """A TelemetryRecorder stand-in that only remembers the stage names it was asked to open.
    TelemetryRecorder 的替身，只记住被要求开启的阶段名。
    """

    def __init__(self) -> None:
        """Start with an empty stage log.
        以空的阶段日志开始。
        """
        self.stages: list[str] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        """Record `name` and yield, mirroring the real recorder's context-manager shape.
        记录 `name` 并让出控制权，与真实记录器的上下文管理器形状一致。
        """
        self.stages.append(name)
        yield


def _seed(adapter: HMSEngineAdapter) -> list[dict[str, Any]]:
    """Retain the three hms4eye-shaped items used by the end-to-end tests; return the added rows.
    留存端到端测试所用的三条 hms4eye 形状条目；返回新增的行。
    """
    rows: list[dict[str, Any]] = []
    items = [
        (
            "Sarah moved the billing review to March 1.",
            _retain_metadata(
                _hms_document_id="doc-1",
                _hms_tags=["work", "billing"],
                _hms_context="quarterly planning",
                _hms_event_date="2026-03-01",
            ),
        ),
        ("I prefer dark roast coffee in the morning.", _retain_metadata(_hms_tags=["food"])),
        ("The team standup happens at 9am on Mondays.", _retain_metadata(_hms_tags=["work"])),
    ]
    for content, metadata in items:
        result = adapter.add([{"role": "user", "content": content}], user_id=SCOPE, metadata=metadata)
        rows.extend(result["results"])
    return rows


def test_hms4eye_adapter_signatures_match_the_mem0_adapter():
    """Every one of the seven seam methods has the byte-identical signature of hms4eye's Mem0Adapter.
    七个接缝方法中的每一个都与 hms4eye 的 Mem0Adapter 具有逐字节相同的签名。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over str(inspect.signature(...)) per method.
        对每个方法 str(inspect.signature(...)) 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    for name, expected in MEM0_ADAPTER_SIGNATURES.items():
        method = getattr(HMSEngineAdapter, name)
        assert str(inspect.signature(method)) == expected, name
    # hms4eye constructs its adapter positionally as `Mem0Adapter(memory, recorder)`, so only the
    # first two positional slots are load-bearing; every extra knob must stay keyword-only.
    # hms4eye 以位置参数方式构造适配器 `Mem0Adapter(memory, recorder)`，因此只有前两个位置槽位是
    # 承重的；其余每个旋钮都必须保持仅关键字。
    parameters = list(inspect.signature(HMSEngineAdapter.__init__).parameters.values())
    positional = [p for p in parameters if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert [p.name for p in positional] == ["self", "hms", "telemetry"]
    assert positional[2].default is None
    assert all(
        p.default is not inspect.Parameter.empty
        for p in parameters
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    )


def test_hms4eye_adapter_exposes_no_mem0_only_mutation_commands():
    """The adapter keeps hms4eye's guard: no public `update` and no public `delete` attribute.
    适配器遵守 hms4eye 的守卫：没有公开的 `update`，也没有公开的 `delete` 属性。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions replicating tests/mem0/test_adapter_calls.py of the hms4eye service.
        复刻 hms4eye 服务中 tests/mem0/test_adapter_calls.py 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert not hasattr(HMSEngineAdapter, "update")
    assert not hasattr(HMSEngineAdapter, "delete")


def test_hms4eye_adapter_round_trips_the_tenant_guard_metadata(hms):
    """retain / recall / list / get / history / clear all work, and `_hms_*` keys survive verbatim.
    retain / recall / list / get / history / clear 全部可用，且 `_hms_*` 键原样存活。

    Input / 输入：
        hms: The shared HMS fixture (rule LLM, hash embedder, parametrized index).
        hms：共享的 HMS 夹具（规则 LLM、哈希嵌入器、参数化索引）。

    Output / 输出：
        Assertions over the seven-method round trip hms4eye's MemoryService performs.
        对 hms4eye 的 MemoryService 所执行七方法往返的断言。

    State effect / 状态影响：
        Writes and then archives three memories in the fixture's in-memory store.
        在夹具的内存存储中写入并随后归档三条记忆。
    """
    adapter = HMSEngineAdapter(hms)
    rows = _seed(adapter)
    assert rows and all(row["event"] in {"ADD", "UPDATE", "NONE"} for row in rows)

    recalled = adapter.search("billing review", user_id=SCOPE, top_k=20, threshold=0.1)
    assert any("billing" in row["memory"].lower() for row in recalled["results"])

    listed = adapter.get_all(user_id=SCOPE, top_k=50)
    assert len(listed["results"]) == len(rows)
    assert all("score" not in row for row in listed["results"])

    # get_memory() re-checks these two keys to enforce tenant isolation: they must come back untouched.
    # get_memory() 会二次核对这两个键以实施租户隔离：它们必须原样返回。
    memory_id = rows[0]["id"]
    fetched = adapter.get(memory_id)
    assert fetched is not None
    assert fetched["metadata"]["_hms_tenant_id"] == "tenant-a"
    assert fetched["metadata"]["_hms_bank_id"] == "bank-1"
    assert fetched["metadata"]["_hms_tags"] == ["work", "billing"]
    assert adapter.get("mem_does_not_exist") is None

    history = adapter.history(memory_id)
    assert isinstance(history, list) and history
    assert history[0]["memory_id"] == memory_id and history[0]["event"] == "ADD"

    adapter.delete_all(user_id=SCOPE)
    assert adapter.get_all(user_id=SCOPE)["results"] == []


def test_hms4eye_adapter_applies_the_tag_filter_tree(hms):
    """The `{'OR': [{'_hms_tags': {'contains': tag}}]}` tree hms4eye builds filters recall correctly.
    hms4eye 构造的 `{'OR': [{'_hms_tags': {'contains': tag}}]}` 树能正确过滤召回结果。

    Input / 输入：
        hms: The shared HMS fixture.
        hms：共享的 HMS 夹具。

    Output / 输出：
        Assertions that the OR tree, the AND tree and a bare leaf each select the expected memories.
        断言 OR 树、AND 树与裸叶节点各自选出预期的记忆。

    State effect / 状态影响：
        Writes three memories in the fixture's in-memory store.
        在夹具的内存存储中写入三条记忆。
    """
    adapter = HMSEngineAdapter(hms)
    _seed(adapter)

    work = adapter.search(
        "what is happening",
        user_id=SCOPE,
        top_k=20,
        threshold=0.1,
        metadata_filters={"OR": [{"_hms_tags": {"contains": "work"}}]},
    )
    assert work["results"]
    assert all("work" in (row["metadata"] or {})["_hms_tags"] for row in work["results"])

    # `tags_match='all'` builds an AND tree; 'work' + 'billing' co-occur on exactly one item.
    # Threshold 0 isolates the filter tree: hms4eye pins threshold=0.1, which would also drop a
    # tag-matching row whose fused relevance to an unrelated query is low (0.042 here).
    # `tags_match='all'` 构造 AND 树；'work' 与 'billing' 恰好在一个条目上共现。
    # 阈值取 0 以隔离过滤树：hms4eye 固定 threshold=0.1，会连同标签命中但与查询无关的行一起丢弃（此处融合分 0.042）。
    both = adapter.search(
        "what is happening",
        user_id=SCOPE,
        top_k=20,
        threshold=0.0,
        metadata_filters={"AND": [{"_hms_tags": {"contains": "work"}}, {"_hms_tags": {"contains": "billing"}}]},
    )
    assert len(both["results"]) == 1
    assert set(both["results"][0]["metadata"]["_hms_tags"]) == {"work", "billing"}

    # The same AND tree through get_all() has no relevance stage at all, so it selects the row
    # regardless of the query — proof that the filter, not the ranker, did the selecting above.
    # 同一 AND 树经 get_all() 时完全没有相关性阶段，因此无论查询为何都会选出该行——证明上面是过滤器而非排序器在起作用。
    listed = hms.get_all(
        filters={
            "user_id": SCOPE,
            "AND": [{"_hms_tags": {"contains": "work"}}, {"_hms_tags": {"contains": "billing"}}],
        }
    )
    assert len(listed["results"]) == 1

    # A single tag is passed as a bare leaf, without any AND/OR wrapper.
    # 单个标签会作为裸叶节点传入，不带任何 AND/OR 包装。
    food = adapter.search(
        "coffee",
        user_id=SCOPE,
        top_k=20,
        threshold=0.1,
        metadata_filters={"_hms_tags": {"contains": "food"}},
    )
    assert len(food["results"]) == 1 and "coffee" in food["results"][0]["memory"].lower()


def test_hms4eye_adapter_records_the_mem0_stage_names(hms):
    """Each method opens exactly one telemetry stage, named 'mem0.<method>' unless the prefix is overridden.
    每个方法恰好开启一个遥测阶段，除非覆盖前缀，否则命名为 'mem0.<方法名>'。

    Input / 输入：
        hms: The shared HMS fixture.
        hms：共享的 HMS 夹具。

    Output / 输出：
        Assertions over the recorded stage-name sequence for both prefixes.
        对两种前缀下所记录阶段名序列的断言。

    State effect / 状态影响：
        Writes and archives one memory in the fixture's in-memory store.
        在夹具的内存存储中写入并归档一条记忆。
    """
    telemetry = _RecordingTelemetry()
    adapter = HMSEngineAdapter(hms, telemetry)
    added = adapter.add([{"role": "user", "content": "Sarah owns the billing runbook."}], user_id=SCOPE)
    memory_id = added["results"][0]["id"]
    adapter.search("billing", user_id=SCOPE)
    adapter.extract([{"role": "user", "content": "Sarah owns the billing runbook."}])
    adapter.get_all(user_id=SCOPE)
    adapter.get(memory_id)
    adapter.history(memory_id)
    adapter.delete_all(user_id=SCOPE)
    assert telemetry.stages == [
        "mem0.add",
        "mem0.search",
        "mem0.extract",
        "mem0.get_all",
        "mem0.get",
        "mem0.history",
        "mem0.delete_all",
    ]

    renamed = _RecordingTelemetry()
    HMSEngineAdapter(hms, renamed, stage_prefix="hms").get_all(user_id=SCOPE)
    assert renamed.stages == ["hms.get_all"]

    # A missing recorder must not change behaviour: the built-in no-op stage is used instead.
    # 缺少记录器不得改变行为：改用内置的空操作阶段。
    assert HMSEngineAdapter(hms).get_all(user_id=SCOPE)["results"] == []


def test_hms4eye_adapter_extract_returns_texts_without_writing_to_the_engine(hms):
    """extract() returns a list of fact texts and leaves the engine store untouched.
    extract() 返回事实文本列表，且不改动引擎存储。

    Input / 输入：
        hms: The shared HMS fixture.
        hms：共享的 HMS 夹具。

    Output / 输出：
        Assertions over the returned list type/content and over the unchanged store.
        对返回列表类型/内容以及存储未变的断言。

    State effect / 状态影响：
        None on the fixture store; extract() builds and closes its own in-memory store.
        对夹具存储无影响；extract() 自行构建并关闭内存存储。
    """
    adapter = HMSEngineAdapter(hms)
    before = hms.store.stats()
    facts = adapter.extract(
        [{"role": "user", "content": "Alice lives in Berlin. She prefers tea over coffee."}],
        prompt="facts",
    )
    assert isinstance(facts, list) and facts
    assert all(isinstance(text, str) and text for text in facts)
    assert len(set(facts)) == len(facts)
    assert hms.store.stats() == before
    assert adapter.get_all(user_id=SCOPE)["results"] == []


def test_hms4eye_mapping_projects_a_real_row_onto_a_recall_result(hms):
    """A real search row maps onto all 13 RecallResult fields, with witnesses in `source_fact_ids`.
    一条真实搜索行映射为全部 13 个 RecallResult 字段，且见证落在 `source_fact_ids` 中。

    Input / 输入：
        hms: The shared HMS fixture.
        hms：共享的 HMS 夹具。

    Output / 输出：
        Assertions over the mapped RecallResult, the memory unit and the reserved-key split.
        对映射后的 RecallResult、记忆单元与保留键拆分的断言。

    State effect / 状态影响：
        Writes three memories in the fixture's in-memory store.
        在夹具的内存存储中写入三条记忆。
    """
    adapter = HMSEngineAdapter(hms)
    _seed(adapter)
    row = next(
        item
        for item in adapter.search("billing review", user_id=SCOPE, top_k=20, threshold=0.1)["results"]
        if "billing" in item["memory"].lower()
    )

    result = to_recall_result(row)
    assert set(result) == {
        "id",
        "text",
        "type",
        "entities",
        "context",
        "occurred_start",
        "occurred_end",
        "mentioned_at",
        "document_id",
        "metadata",
        "chunk_id",
        "tags",
        "source_fact_ids",
    }
    assert result["id"] == row["id"] and result["text"] == row["memory"]
    assert result["type"] == "world"
    assert result["context"] == "quarterly planning"
    assert result["document_id"] == "doc-1"
    assert result["occurred_start"] == row["valid_from"] and result["occurred_end"] is None
    assert result["mentioned_at"] == row["created_at"]
    assert result["chunk_id"] is None
    assert {"work", "billing"} <= set(result["tags"])
    # hms4eye's mem0 path can never fill this field; the anchors of the atom do.
    # hms4eye 的 mem0 路径永远填不上这个字段；原子的锚点填上了它。
    assert result["source_fact_ids"] == row["anchor_ids"] and result["source_fact_ids"]
    # Reserved bookkeeping is stripped from the caller-facing metadata.
    # 保留簿记键已从面向调用方的元数据中剥离。
    assert result["metadata"] == {}

    reserved, user_keys = split_reserved(row["metadata"])
    assert reserved["_hms_tenant_id"] == "tenant-a" and user_keys == {}

    unit = to_memory_unit(row)
    assert set(unit) == {"id", "text", "type", "context", "date", "entities"}
    assert unit["id"] == row["id"] and unit["date"] == row["valid_from"]


def test_hms4eye_mapping_types_an_ep_row_as_experience(hms, events):
    """An EP row maps to 'experience', an interface_ep/skill row to 'observation', anything else to 'world'.
    EP 行映射为 'experience'，interface_ep/skill 行映射为 'observation'，其余映射为 'world'。

    Input / 输入：
        hms: The shared HMS fixture.
        hms：共享的 HMS 夹具。

        events: The fixture loader for tests/fixtures/events/.
        events：tests/fixtures/events/ 的夹具加载器。

    Output / 输出：
        Assertions over fact_type_of for a real EP row and over the declared-type override.
        对真实 EP 行的 fact_type_of 以及显式类型覆盖的断言。

    State effect / 状态影响：
        Ingests the structured billing events into the fixture's in-memory store.
        把结构化 billing 事件摄取进夹具的内存存储。
    """
    fixture = events("structured_billing")
    ingested = hms.add_events([*fixture.events_fail, *fixture.events_fix], user_id=SCOPE)
    ep_row = next(row for row in ingested["results"] if row["hms"]["source_kind"] == "ep")
    assert fact_type_of(ep_row) == "experience"
    assert to_recall_result(ep_row)["type"] == "experience"
    assert to_recall_result(ep_row)["source_fact_ids"] == ep_row["anchor_ids"]

    observation_row = next(row for row in ingested["results"] if row["hms"]["source_kind"] == "observation")
    assert fact_type_of(observation_row) == "world"

    # Consolidation output (interface_ep / skill) is what hms4eye calls the 'observation' fact type.
    # 凝聚产物（interface_ep / skill）正是 hms4eye 所说的 'observation' 事实类型。
    assert fact_type_of({"hms": {"source_kind": "interface_ep"}}) == "observation"
    assert fact_type_of({"hms": {"source_kind": "skill"}}) == "observation"
    # An explicit _hms_type always wins, which is how hms4eye pins a type it wrote itself.
    # 显式的 _hms_type 始终优先，这也是 hms4eye 固定自己写入类型的方式。
    assert fact_type_of({"metadata": {"_hms_type": "experience"}, "hms": {"source_kind": "observation"}}) == "experience"
    assert fact_type_of({}) == "world"
