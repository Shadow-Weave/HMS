"""Group H tests: Mem0 signature parity, envelopes and error codes, lineage updates, from_config, migrate, CLI, native flow.
H 组测试：Mem0 签名一致性、信封与错误码、谱系更新、from_config、迁移、命令行与原生流程。
"""

# Purpose (EN): Assert CONTRACT §17 (and §18.3 tests 57–62) over the HMS facade, plus one end-to-end native flow.
# 目的（中）：在 HMS 门面上断言契约 §17（以及 §18.3 测试 57–62），外加一条端到端的原生流程。

# Input (EN): tests/fixtures/events/*.json, tests/fixtures/golden/* and the conftest fixtures.
# 输入（中）：tests/fixtures/events/*.json、tests/fixtures/golden/* 与 conftest 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory stores and files under tmp_path (HMS_DIR).
# 状态影响（中）：除内存存储与 tmp_path（HMS_DIR）下的文件外无副作用。

from __future__ import annotations

import inspect
import json
import logging
import pathlib
import re
import shlex
import warnings
from copy import deepcopy
from typing import Any

import pytest

from hms_elite import HMS, HMSConfig, __version__
from hms_elite.cli import main as cli_main
from hms_elite.compiler import StateCompiler
from hms_elite.embeddings.base import EmbedderBase
from hms_elite.exceptions import ConfigError, ValidationError
from hms_elite.graph import SupportGraph
from hms_elite.index.base import IndexBase
from hms_elite.ingest.pipeline import Ingestor
from hms_elite.llms.base import LLMBase
from hms_elite.memory.base import MemoryBase
from hms_elite.memory.main import _OSSProject
from hms_elite.memory.storage import StateStore
from hms_elite.migrate import from_mem0
from hms_elite.models import StatePackage
from hms_elite.ops import FeedbackProcessor, StateScheduler
from hms_elite.reranker import LLMReranker, OverlapReranker, RerankerBase, rank_rows
from hms_elite.utils.factory import RerankerFactory
from tests.conftest import SequentialIds

U = "alice"
S1 = {"user_id": U, "run_id": "run-coffee"}
S4_QUERY = "fix checkout webhook retry"
S4_AS_OF = "2026-06-15T12:00:00Z"
S4_CONSTRAINT = "Do not modify the subscription state machine."
S4_EP1_A = "patch src/billing/retry.ts: add exponential backoff"
S4_EP1_Y = "billing_e2e failed: subscription_state_regression"
S4_DC = f"Do not re-apply '{S4_EP1_A}': {S4_EP1_Y}"
S6_CORRECTION = "retry.ts may change once billing_e2e is isolated."
LATTE = "I like oat-milk lattes in the morning."
LATTE_FACT = "User likes oat-milk lattes in the morning."
MEM0_METHODS = ("add", "get", "get_all", "search", "update", "delete", "delete_all", "history", "reset", "close")
PUBLIC_ATTRS = {
    "config": HMSConfig,
    "store": StateStore,
    "index": IndexBase,
    "embedder": EmbedderBase,
    "llm": LLMBase,
    "graph": SupportGraph,
    "ingestor": Ingestor,
    "compiler": StateCompiler,
    "feedback_processor": FeedbackProcessor,
    "scheduler": StateScheduler,
}
EXPECTED_SIGNATURES = {
    "__init__": "(self, config: HMSConfig | dict | None = None, *, store: StateStore | None = None) -> None",
    "add": (
        "(self, messages, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, "
        "metadata: dict | None = None, timestamp: Any = None, expiration_date: Any = None, infer: bool = True, "
        "memory_type: str | None = None, prompt: str | None = None) -> dict"
    ),
    "get": "(self, memory_id: str) -> dict | None",
    "get_all": "(self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False, **kwargs) -> dict",
    "search": (
        "(self, query: str, *, top_k: int = 20, filters: dict | None = None, threshold: float = 0.1, "
        "rerank: bool = False, explain: bool = False, reference_date: Any = None, show_expired: bool = False, "
        "**kwargs) -> dict"
    ),
    "update": (
        "(self, memory_id: str, text: str | None = None, metadata: dict | None = None, "
        "expiration_date: Any = _UNSET, data: str | None = None) -> dict"
    ),
    "delete": "(self, memory_id: str) -> dict",
    "delete_all": "(self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> dict",
    "history": "(self, memory_id: str) -> list[dict]",
    "reset": "(self) -> None",
    "close": "(self) -> None",
    "chat": "(self, query: str) -> None",
    "add_events": (
        "(self, events: list[SourceEvent | dict], *, user_id: str | None = None, agent_id: str | None = None, "
        "run_id: str | None = None, infer: bool = True) -> dict"
    ),
    "compile": (
        "(self, query: str, *, filters: dict, budget_tokens: int | None = None, mode: str | None = None, "
        "as_of: Any = None, require_witness: bool | None = None) -> ContextIR"
    ),
    "axis_query": "(self, aq: AxisQuery | dict) -> AxisResult",
    "why": (
        "(self, target_id: str, *, hops: int = 2, filters: dict | None = None, edge_types: list[str] | None = None, "
        "include_invalid: bool = False) -> dict"
    ),
    "feedback": (
        "(self, target_id: str | None, feedback: FeedbackType | str, *, reason: str | None = None, "
        "correction: str | dict | None = None, actor_id: str | None = None) -> dict"
    ),
    "revoke": "(self, anchor_id: str, *, reason: str | None = None, actor_id: str | None = None) -> dict",
    "purge": (
        "(self, *, filters: dict, older_than_days: int, dry_run: bool = True, keep_witnessed: bool = True) -> dict"
    ),
    "vacuum": "(self) -> dict",
    "handoff": (
        "(self, *, filters: dict, goal: str | None = None, budget_tokens: int | None = None, "
        "checkpoint: bool = True, include_edges: bool = True) -> StatePackage"
    ),
    "import_package": (
        "(self, pkg: StatePackage | str | dict, *, user_id: str | None = None, agent_id: str | None = None, "
        "run_id: str | None = None) -> dict"
    ),
    "checkpoint": "(self, *, filters: dict, label: str | None = None) -> str",
    "rollback": "(self, checkpoint_id: str, *, reason: str | None = None) -> dict",
    "transitions": (
        "(self, *, filters: dict, since: Any = None, scale: str | None = None, fmt: str = tuples, "
        "kinds: list[str] | None = None) -> list | str"
    ),
    "consolidate": "(self, *, filters: dict | None = None, phases: list[str] | None = None) -> dict",
    "graph_stats": "(self, *, filters: dict | None = None) -> dict",
    "context": (
        "(self, query: str, *, filters: dict, top_k: int | None = None, as_of: Any = None, "
        "budget_tokens: int | None = None) -> dict"
    ),
}


def _sig(func: Any) -> str:
    """Signature text with the string-annotation quotes removed (from __future__ import annotations).
    去掉字符串注解引号后的签名文本（from __future__ import annotations）。
    """
    return str(inspect.signature(func)).replace("'", "").replace('"', "")


def _code(exc_info: Any) -> str:
    """error_code of a raised HMSError.
    抛出的 HMSError 的 error_code。
    """
    return exc_info.value.error_code


def _s4(hms: HMS, clock: Any, events: Any, scope4: dict) -> dict[str, Any]:
    """Run §19.3 S4 through the facade (constraint + events_fail + card1 + events_fix + card2) and return the ids.
    通过门面运行 §19.3 的 S4（约束 + events_fail + card1 + events_fix + card2）并返回 id。
    """
    fixture = events("structured_billing")
    clock.tick(seconds=60)
    r0 = hms.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    clock.tick(seconds=60)
    ra = hms.add_events(fixture.events_fail, **scope4)
    clock.tick(seconds=60)
    card1 = hms.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF).to_text()
    clock.tick(seconds=60)
    rb = hms.add_events(fixture.events_fix, **scope4)
    clock.tick(seconds=60)
    card2 = hms.compile(S4_QUERY, filters=scope4, budget_tokens=800, as_of=S4_AS_OF).to_text()
    return {
        "mem_uc": r0["results"][0]["id"],
        "uc": r0["results"][0]["atom_id"],
        "ra": ra,
        "rb": rb,
        "card1": card1,
        "card2": card2,
        "anc_e2": [e["anchor_id"] for e in ra["events"] if e["object_type"] == "test_log"][0],
        "e2": [row["atom_id"] for row in ra["results"] if row["atom_type"] == "event"][1],
    }


class _FakeMem0:
    """A Mem0 stand-in whose get_all() records its calls and hides expired rows unless show_expired.
    Mem0 替身：其 get_all() 记录调用，并在未设置 show_expired 时隐藏过期行。
    """

    def __init__(self, rows: list[dict]) -> None:
        """Keep the rows and start an empty call log.
        保存行并初始化空的调用日志。
        """
        self.rows = rows
        self.calls: list[dict] = []

    def get_all(self, *, filters: dict | None = None, top_k: int = 20, show_expired: bool = False) -> dict:
        """Mem0-shaped get_all().
        Mem0 形状的 get_all()。
        """
        self.calls.append({"filters": filters, "top_k": top_k, "show_expired": show_expired})
        rows = [dict(r) for r in self.rows if show_expired or not r.get("expired")]
        for row in rows:
            row.pop("expired", None)
        return {"results": rows}


# ---------------------------------------------------------------------------
# 57
# ---------------------------------------------------------------------------
def test_mem0_signatures_verbatim(hms):
    """inspect.signature of the Mem0 and HMS-native methods equals §17.2/§17.3; MemoryBase abstract set; public attrs.
    Mem0 与 HMS 原生方法的 inspect.signature 等于 §17.2/§17.3；MemoryBase 抽象集合；公开属性。

    Input / 输入：
        hms: The HMS fixture (rule LLM, hash embedder, parametrized index).
        hms：HMS 夹具（规则 LLM、哈希嵌入器、参数化索引）。

    Output / 输出：
        Assertions on signatures, abstract methods, public attributes, chat() and project.update().
        对签名、抽象方法、公开属性、chat() 与 project.update() 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    for name, expected in EXPECTED_SIGNATURES.items():
        assert _sig(getattr(HMS, name)) == expected, name
    assert _sig(HMS.from_config) == "(config_dict: dict[str, Any]) -> HMS"
    assert isinstance(HMS.project, property) and isinstance(hms.project, _OSSProject)
    assert _sig(hms.project.update) == (
        "(custom_instructions: Any = None, custom_categories: Any = None, multilingual: Any = None, "
        "decay: Any = None) -> None"
    )
    assert MemoryBase.__abstractmethods__ == frozenset({"get", "get_all", "update", "delete", "history"})
    assert issubclass(HMS, MemoryBase) and all(callable(getattr(hms, m)) for m in MEM0_METHODS)
    for attr, kind in PUBLIC_ATTRS.items():
        assert isinstance(getattr(hms, attr), kind), attr
    assert hms.reranker is None and hms.scheduler.compiler is hms.compiler and hms.ingestor.store is hms.store
    with pytest.raises(NotImplementedError):
        hms.chat("hello")
    with pytest.raises(ValueError):
        hms.project.update(custom_instructions="x")
    assert isinstance(__version__, str) and __version__
    with HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"})) as scoped:
        assert isinstance(scoped, HMS)
    # 幂等
    scoped.close()  # idempotent


# ---------------------------------------------------------------------------
# 58
# ---------------------------------------------------------------------------
def test_mem0_envelopes_and_validation_codes(hms, clock):
    """add/update/delete/delete_all envelopes verbatim; get missing → None; history bare list; the VALIDATION codes.
    add/update/delete/delete_all 信封逐字一致；get 缺失 → None；history 裸列表；各 VALIDATION 错误码。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions on envelopes and error codes.
        对信封与错误码的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    clock.tick(seconds=60)
    r1 = hms.add(LATTE, timestamp="2026-04-02T08:00:00Z", **S1)
    assert set(r1) == {"results"} and len(r1["results"]) == 1
    row = r1["results"][0]
    assert list(row)[:4] == ["id", "memory", "hash", "metadata"] and row["event"] == "ADD"
    assert row["memory"] == LATTE_FACT and row["version"] == 1 and row["anchor_ids"]
    mem = row["id"]
    assert hms.get(mem)["memory"] == LATTE_FACT and hms.get("mem_missing") is None
    history = hms.history(mem)
    assert isinstance(history, list) and [h["event"] for h in history] == ["ADD"]
    assert set(history[0]) == {
        "id", "memory_id", "old_memory", "new_memory", "event", "created_at", "updated_at", "is_deleted",
        "actor_id", "role", "anchor_id",
    }  # fmt: skip
    assert history[0]["is_deleted"] is False and hms.history("mem_missing") == []
    assert hms.get_all(filters=S1)["results"][0]["id"] == mem and "score" not in hms.get_all(filters=S1)["results"][0]
    assert hms.search("oat-milk lattes", filters={"user_id": U}, threshold=0.0)["results"][0]["id"] == mem
    clock.tick(seconds=60)
    assert hms.update(mem, text="User likes flat whites.") == {"message": "Memory updated successfully!"}
    clock.tick(seconds=60)
    assert hms.delete(mem) == {"message": "Memory deleted successfully!"}
    assert hms.get(mem) is None
    with pytest.raises(ValueError) as missing:
        hms.delete(mem)
    assert _code(missing) == "NOT_FOUND_001" and f"Memory with id {mem} not found" in str(missing.value)
    with pytest.raises(ValueError) as unknown:
        hms.update("mem_missing", text="x")
    assert _code(unknown) == "NOT_FOUND_001"
    clock.tick(seconds=60)
    hms.add("Sarah joined the billing team in March.", timestamp="2026-03-02T09:00:00Z", user_id=U, run_id="run-a")
    assert hms.delete_all(user_id=U, run_id="run-a") == {"message": "Memories deleted successfully!"}
    assert hms.get_all(filters={"user_id": U, "run_id": "run-a"})["results"] == []
    with pytest.raises(ValidationError) as no_filter:
        hms.delete_all()
    assert _code(no_filter) == "VALIDATION_001" and str(no_filter.value) == (
        "At least one filter is required to delete all memories. "
        "If you want to delete all memories, use the `reset()` method."
    )
    with pytest.raises(ValidationError) as e001:
        hms.add("no scope")
    assert _code(e001) == "VALIDATION_001"
    with pytest.raises(ValueError) as whitespace:
        hms.add("bad id", user_id="   ")
    assert _code(whitespace) == "VALIDATION_001"
    with pytest.raises(ValidationError) as e002:
        hms.add("x", user_id=U, memory_type="semantic_memory")
    assert _code(e002) == "VALIDATION_002"
    with pytest.raises(ValidationError) as e003:
        hms.add(123, user_id=U)
    assert _code(e003) == "VALIDATION_003"
    with pytest.raises(ValidationError) as e004:
        hms.add("x", user_id=U, timestamp="not-a-date")
    assert _code(e004) == "VALIDATION_004"
    with pytest.raises(ValueError) as e008:
        hms.get_all(user_id=U)
    assert _code(e008) == "VALIDATION_008" and isinstance(e008.value, ValidationError)
    with pytest.raises(ValueError) as e008b:
        hms.search("coffee", user_id=U)
    assert _code(e008b) == "VALIDATION_008"
    with pytest.raises(ValidationError) as e001b:
        hms.get_all(filters={"topic": "coffee"})
    assert _code(e001b) == "VALIDATION_001"
    with pytest.raises(ValueError) as e009:
        hms.search("coffee", filters={"user_id": U}, threshold=2.0)
    assert _code(e009) == "VALIDATION_009"
    with pytest.raises(ValueError) as e009b:
        hms.get_all(filters={"user_id": U}, top_k=-1)
    assert _code(e009b) == "VALIDATION_009"
    with pytest.raises(ValidationError) as e010:
        hms.search("   ", filters={"user_id": U})
    assert _code(e010) == "VALIDATION_010"
    with pytest.raises(ValueError) as e011:
        hms.update(mem)
    assert _code(e011) == "VALIDATION_011"
    with pytest.raises(ValidationError) as e013:
        hms.compile("coffee", filters=S1, mode="bogus")
    assert _code(e013) == "VALIDATION_013"
    with pytest.raises(ValidationError) as e013b:
        hms.transitions(filters=S1, fmt="csv")
    assert _code(e013b) == "VALIDATION_013"
    with pytest.raises(ValidationError) as e013c:
        hms.why(mem, hops=7)
    assert _code(e013c) == "VALIDATION_013"
    with pytest.raises(ValidationError) as e006:
        hms.feedback(mem, "not_a_type")
    assert _code(e006) == "VALIDATION_006"
    with pytest.raises(ValidationError) as e007:
        hms.add_events([{"object_type": "feedback", "payload": {"target_id": "x", "feedback": "y"}}], user_id=U)
    assert _code(e007) == "VALIDATION_007"
    with pytest.raises(ValueError) as e404:
        hms.why("nonsense-id")
    assert _code(e404) == "NOT_FOUND_001"


# ---------------------------------------------------------------------------
# 59
# ---------------------------------------------------------------------------
def test_update_delete_expiration_lineage(hms, clock, store):
    """update(text) → version 2 + revision edge + UPDATE row; metadata-only → no version; expiration → valid_to; delete.
    update(text) → 版本 2 + revision 边 + UPDATE 行；仅元数据 → 不新增版本；过期日期 → valid_to；delete。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        store: The shared StateStore fixture.
        store：共享的 StateStore 夹具。

    Output / 输出：
        Assertions on lineage, edges, history rows and validity windows.
        对谱系、边、历史行与有效期窗口的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    clock.tick(seconds=60)
    row = hms.add(LATTE, timestamp="2026-04-02T08:00:00Z", metadata={"source": "chat"}, **S1)["results"][0]
    mem, v1 = row["id"], row["atom_id"]
    assert row["metadata"] == {"source": "chat"}
    clock.tick(seconds=60)
    hms.update(mem, text="User likes flat whites.")
    head = hms.get(mem)
    assert head["id"] == mem and head["version"] == 2 and head["memory"] == "User likes flat whites."
    assert head["metadata"] == {"source": "chat"} and head["anchor_ids"][1:] == row["anchor_ids"]
    old = store.get_atom(v1)
    assert old.archived_reason == "revised" and old.valid_to == clock.now()
    edge = store.find_edge(v1, head["atom_id"], "revision")
    assert edge is not None and edge.created_by == "ops" and edge.witness_anchor_ids[0] == head["anchor_ids"][0]
    assert store.get_anchor(head["anchor_ids"][0]).source_ref.startswith("hms://")
    history = hms.history(mem)
    assert [h["event"] for h in history] == ["ADD", "UPDATE"]
    assert (history[1]["old_memory"], history[1]["new_memory"]) == (LATTE_FACT, "User likes flat whites.")
    assert history[1]["anchor_id"] == head["anchor_ids"][0] and history[1]["updated_at"] == clock.now()
    assert len(hms.get_all(filters=S1)["results"]) == 1 and len(hms.get_all(filters=S1, show_expired=True)["results"]) == 2  # fmt: skip
    clock.tick(seconds=60)
    hms.update(mem, metadata={"k": "v", "user_id": "ignored"})
    head = hms.get(mem)
    assert head["version"] == 2 and head["metadata"] == {"source": "chat", "k": "v"}
    assert [h["event"] for h in hms.history(mem)] == ["ADD", "UPDATE", "UPDATE"]
    clock.tick(seconds=60)
    # 已废弃的别名：记录一条警告，行为与 text= 相同
    hms.update(mem, data="User likes cortados.")  # deprecated alias: logs a warning, behaves like text=
    assert hms.get(mem)["version"] == 3 and hms.get(mem)["memory"] == "User likes cortados."
    clock.tick(seconds=60)
    hms.update(mem, expiration_date="2026-12-31")
    head = hms.get(mem)
    assert head["version"] == 3 and head["valid_to"] == "2027-01-01T00:00:00.000000+00:00"
    assert head["expiration_date"] == "2026-12-31"
    clock.tick(seconds=60)
    hms.update(mem, expiration_date=None)
    head = hms.get(mem)
    assert head["valid_to"] is None and "expiration_date" not in head
    with pytest.raises(ValidationError) as bad_date:
        hms.update(mem, expiration_date="31/12/2026")
    assert _code(bad_date) == "VALIDATION_004"
    clock.tick(seconds=60)
    hms.update(mem, text="User likes espresso.", expiration_date="2027-06-30")
    head = hms.get(mem)
    assert head["version"] == 4 and head["valid_to"] == "2027-07-01T00:00:00.000000+00:00"
    clock.tick(seconds=60)
    hms.delete(mem)
    assert hms.get(mem) is None
    latest = store.latest_version(mem)
    assert latest.archived_reason == "deleted" and latest.text == "User likes espresso."
    expired = hms.get_all(filters=S1, show_expired=True)["results"]
    assert [r["hms"]["archived_reason"] for r in expired].count("deleted") == 1
    last = hms.history(mem)[-1]
    assert last["event"] == "DELETE" and last["is_deleted"] is True and last["new_memory"] is None
    assert store.stats()["memory_atoms"] == 4 and store.find_edge(v1, head["atom_id"], "revision") is None


# ---------------------------------------------------------------------------
# 59b: retention escape hatch through the facade (§16.6)
# 59b：经门面的保留策略逃生舱（§16.6）
# ---------------------------------------------------------------------------
def test_purge_facade_is_explicit_default_off_and_audited(hms, clock, store, index_provider):
    """purge() is a dry run by default, refuses to materialize while retention is disabled, and leaves PURGE history + a working why().
    purge() 默认试运行，保留策略关闭时拒绝真正执行，执行后留下 PURGE 历史且 why() 仍可用。

    Input / 输入：
        hms: The HMS fixture (retention disabled by default).
        hms：HMS 夹具（保留策略默认关闭）。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        store: The shared StateStore fixture.
        store：共享的 StateStore 夹具。

        index_provider: 'sqlite' | 'memory'.
        index_provider：'sqlite' | 'memory'。

    Output / 输出：
        Assertions on the dry-run report, the CONFIG_001 refusal, the materialized purge, history(), get_all() and why().
        对试运行报告、CONFIG_001 拒绝、真正执行的清理、history()、get_all() 与 why() 的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    gone = hms.add("I like oat-milk lattes in the morning.", user_id=U, run_id="run-coffee")["results"][0]["id"]
    clock.tick(seconds=60)
    kept = hms.add("I moved to Berlin last spring.", user_id=U, run_id="run-coffee")["results"][0]["id"]
    clock.tick(seconds=60)
    hms.delete(gone)
    clock.tick(days=100)
    plan = hms.purge(filters=S1, older_than_days=90)
    assert plan["dry_run"] is True and plan["purged"] == 0 and [p["memory_id"] for p in plan["purgeable"]] == [gone]
    assert hms.get_all(filters=S1, show_expired=True)["results"][-1]["hms"]["archived_reason"] == "deleted"
    with pytest.raises(ConfigError) as refused:
        hms.purge(filters=S1, older_than_days=90, dry_run=False)
    assert _code(refused) == "CONFIG_001" and refused.value.details == {"retention.enabled": False, "dry_run": False}
    assert store.latest_version(gone) is not None
    with pytest.raises(ValidationError) as no_scope:
        hms.purge(filters={}, older_than_days=90)
    assert _code(no_scope) == "VALIDATION_001"
    armed = HMS(
        HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}, retention={"enabled": True}),
        store=store,
    )
    armed.index = hms.index
    rep = armed.purge(filters=S1, older_than_days=90, dry_run=False)
    assert rep["purged"] == 1 and rep["anchor_id"].startswith("anc_") and rep["keep_witnessed"] is True
    assert armed.get(gone) is None and store.latest_version(gone) is None
    assert [r["id"] for r in armed.get_all(filters=S1, show_expired=True)["results"]] == [kept]
    history = armed.history(gone)
    assert [h["event"] for h in history] == ["ADD", "DELETE", "PURGE"]
    assert history[-1]["is_deleted"] is True and history[-1]["old_memory"] is None
    assert history[-1]["anchor_id"] == rep["anchor_id"] and sorted(history[-1]) == sorted(history[0])
    assert store.get_anchor(history[0]["anchor_id"]) is not None
    why = armed.why(kept)
    assert why["target"]["id"] == store.resolve_head(kept).atom_id and why["anchors"]
    with pytest.raises(ValidationError) as missing:
        armed.why(gone)
    assert _code(missing) == "NOT_FOUND_001"
    size = armed.vacuum()
    assert set(size) == {"path", "before_bytes", "after_bytes", "freed_bytes"} and size["path"] == ":memory:"
    assert size["after_bytes"] <= size["before_bytes"] and size["freed_bytes"] == size["before_bytes"] - size["after_bytes"]  # fmt: skip
    assert armed.purge(filters=S1, older_than_days=90, dry_run=False)["purged"] == 0


# ---------------------------------------------------------------------------
# 60
# ---------------------------------------------------------------------------
def test_from_config_mem0_dict_and_rerank_noop(events, tmp_path, caplog):
    """A Mem0 dict with offline providers constructs (store.path == history_db_path); the reranker envelope builds a real OverlapReranker (no warning), an unknown reranker provider is CONFIG_002; RENAMED/REMOVED.
    带离线提供者的 Mem0 字典可构造（store.path == history_db_path）；重排器信封构造真实的 OverlapReranker（无警告），未知重排器提供者为 CONFIG_002；RENAMED/REMOVED 处理。

    Input / 输入：
        events: Fixture loader (mem0_config.json).
        events：夹具加载器（mem0_config.json）。

        tmp_path: pytest temporary directory for the history database.
        tmp_path：pytest 临时目录，用于历史数据库文件。

        caplog: pytest log capture (from_config logs the full error).
        caplog：pytest 日志捕获（from_config 记录完整错误）。

    Output / 输出：
        Assertions on construction, warnings, search(rerank=True) and the config error codes.
        对构造、警告、search(rerank=True) 与配置错误码的断言。

    State effect / 状态影响：
        Creates a SQLite file under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    d = deepcopy(dict(events("mem0_config")))
    d["llm"] = {"provider": "rule"}
    d["embedder"] = {"provider": "hash"}
    d["vector_store"] = {"provider": "sqlite", "config": {}}
    d["reranker"] = {"provider": "overlap", "config": {}}
    d["history_db_path"] = str(tmp_path / "history.db")
    assert set(d) >= {"history_db_path", "version", "custom_instructions"}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        hms = HMS.from_config(d)
    # the envelope is a real provider now: nothing warns about it
    # 该信封现在是真实的提供者：不再有关于它的警告
    assert sum("reranker" in str(w.message) for w in record) == 0
    try:
        assert hms.store.path == d["history_db_path"] and hms.config.db_path() == d["history_db_path"]
        assert hms.config.version == "v1.1" and hms.config.custom_instructions is None
        assert isinstance(hms.reranker, OverlapReranker) and hms.reranker.identity == "overlap@v1"
        assert hms.config.llm.provider == "rule" and hms.embedder.dims == 64
        hms.add("I like tea.", user_id=U)
        hits = hms.search("tea", filters={"user_id": U}, rerank=True)["results"]
        assert hits and "tea" in hits[0]["memory"].lower() and all("rerank_score" in r for r in hits)
        again = hms.search("tea", filters={"user_id": U}, rerank=True)["results"]
        # 真实时钟：分数会漂移，顺序不会
        assert [r["id"] for r in again] == [r["id"] for r in hits]  # real clock: scores drift, order does not
    finally:
        hms.close()
    with pytest.raises(ConfigError) as cohere:
        HMS.from_config({**d, "reranker": {"provider": "cohere", "config": {}}})
    assert _code(cohere) == "CONFIG_002" and cohere.value.details == {
        "kind": "reranker",
        "provider": "cohere",
        "supported": ["llm", "overlap"],
    }
    with pytest.warns(DeprecationWarning):
        renamed = HMS.from_config(
            {
                "llm": {"provider": "rule"},
                "embedder": {"provider": "hash"},
                "index": {"provider": "memory"},
                "store": {"path": ":memory:"},
            }
        )
    assert renamed.config.vector_store.provider == "memory" and renamed.index.provider_name == "memory"
    renamed.close()
    with pytest.raises(ConfigError) as removed:
        HMS.from_config({"graph_store": {"provider": "neo4j"}, "store": {"path": ":memory:"}})
    assert _code(removed) == "CONFIG_003"
    with pytest.raises(ConfigError) as unknown:
        HMS.from_config({"vectr_store": {"provider": "sqlite"}, "store": {"path": ":memory:"}})
    assert _code(unknown) == "CONFIG_001" and "vectr_store" in str(unknown.value)
    with caplog.at_level(logging.ERROR, logger="hms_elite.memory.main"):
        with pytest.raises(ConfigError) as provider:
            HMS.from_config({"vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}})
    assert _code(provider) == "CONFIG_002" and any("from_config" in r.getMessage() for r in caplog.records)
    with pytest.raises(ConfigError):
        HMS(config="not-a-config")  # type: ignore[arg-type]
    plain = HMS({"store": {"path": ":memory:"}, "vector_store": {"provider": "memory"}})
    assert plain.config.vector_store.provider == "memory"
    plain.close()


# ---------------------------------------------------------------------------
# 60b. Group Q1: infer=False atoms are reachable through the scale-gate fallback ladder
# 60b. Q1 组：infer=False 的原子可经尺度门控回退阶梯找到
# ---------------------------------------------------------------------------
RAW_TEA = "Alice prefers jasmine tea and avoids coffee after 3 PM."
RAW_QUERY = "What drink does Alice prefer?"
# Deterministic scores (frozen clock, sequential ids, hash embedder); identical for the sqlite and memory indexes.
# 确定性分数（冻结时钟、顺序 id、哈希嵌入器）；sqlite 与 memory 索引下相同。
RAW_TEA_SCORE = 0.5667041237517819
LATTE_PREF_SCORE = 0.5935392293333908


def test_search_infer_false_atoms_hit_via_fallback_ladder(hms):
    """The exact defect case: a raw add(infer=False) message is rank 1 for a preference query; infer=True scores unchanged.
    精确的缺陷用例：add(infer=False) 的原始消息在偏好查询下排第一；infer=True 的分数不变。

    Input / 输入：
        hms: The HMS fixture (rule LLM, hash embedder, parametrized index, frozen clock).
        hms：HMS 夹具（规则 LLM、哈希嵌入器、参数化索引、冻结时钟）。

    Output / 输出：
        Assertions on search (default and zero threshold), explain provenance, compile candidates and the
        infer=True preference path.
        对 search（默认与零阈值）、explain 溯源、compile 候选以及 infer=True 偏好路径的断言。

    State effect / 状态影响：
        Writes to the in-memory store.
        写入内存存储。
    """
    hms.add([{"role": "user", "content": RAW_TEA}], user_id="u", infer=False)
    hits = hms.search(RAW_QUERY, filters={"user_id": "u"})["results"]
    assert len(hits) == 1 and hits[0]["memory"] == RAW_TEA and hits[0]["scale_level"] == "L1_micro"
    assert hits[0]["atom_type"] == "message" and hits[0]["hms"]["obs_type"] is None
    assert hits[0]["score"] == pytest.approx(RAW_TEA_SCORE, abs=1e-12)
    assert hms.search(RAW_QUERY, filters={"user_id": "u"}, threshold=0.0)["results"] == hits
    explained = hms.search(RAW_QUERY, filters={"user_id": "u"}, explain=True)["results"]
    details = explained[0]["score_details"]
    # the plan still carries the preference gates; primary_axis says they were relaxed
    # 规划仍带偏好门控；primary_axis 说明它们已被放宽
    assert details["route"] == "preference" and details["primary_axis"] == "fallback"
    assert details["axis_gates"]["obs_type"] == ["preference_event", "habit"]
    # compile() runs the same candidate stage: the raw atom is a candidate and the level is recorded
    # compile() 走同一候选阶段：原始原子成为候选，且级别被记录
    ir = hms.compile(RAW_QUERY, filters={"user_id": "u"})
    assert ir.retrieval["candidates"] == 1 and ir.retrieval["primary_axis"] == "fallback"
    assert any(item.text == RAW_TEA for item in ir.items())
    # the infer=True path stays on ladder level 1 with its exact score (the golden scenario S1 path)
    # infer=True 路径仍停留在阶梯第一级并保持精确分数（黄金场景 S1 的路径）
    hms.add(LATTE, user_id=U)
    pref = hms.search("what coffee does the user like", filters={"user_id": U}, explain=True)["results"]
    assert [r["memory"] for r in pref] == [LATTE_FACT]
    assert pref[0]["score_details"]["primary_axis"] == "type" and pref[0]["score_details"]["route"] == "preference"
    assert pref[0]["score"] == pytest.approx(LATTE_PREF_SCORE, abs=1e-12)
    assert list(pref[0]["score_details"]["axis_gates"]) == ["time", "entity", "obs_type", "scope", "validity"]


# ---------------------------------------------------------------------------
# 60c. Group Q2: reranker wiring
# 60c. Q2 组：重排器接线
# ---------------------------------------------------------------------------
class _TailFirstReranker(RerankerBase):
    """Test reranker: the last retrieved row scores highest (a guaranteed order change).
    测试重排器：最后检索到的行分数最高（保证顺序改变）。
    """

    provider_name = "tailfirst"

    def rerank(self, query: str, rows: list[dict], *, top_k: int | None = None) -> list[dict]:
        """Score row i as i / (n - 1).
        把第 i 行打分为 i / (n - 1)。
        """
        n = max(len(rows) - 1, 1)
        return rank_rows(rows, [i / n for i in range(len(rows))], top_k=top_k)


class _BoomReranker(RerankerBase):
    """Test reranker that always raises (the facade must keep the retrieval order).
    总是抛错的测试重排器（门面必须保持检索顺序）。
    """

    provider_name = "boom"

    def rerank(self, query: str, rows: list[dict], *, top_k: int | None = None) -> list[dict]:
        """Raise RuntimeError.
        抛出 RuntimeError。
        """
        raise RuntimeError("boom")


TEA_ROWS = ["Alice prefers jasmine tea.", "Bob drinks black coffee at noon.", "Carol takes green tea with honey."]


def _seed_raw(hms: HMS, user_id: str) -> list[dict]:
    """Add TEA_ROWS as raw messages (infer=False: zero LLM calls) and return the un-reranked search rows.
    以原始消息添加 TEA_ROWS（infer=False：零 LLM 调用），返回未重排的搜索行。
    """
    for text in TEA_ROWS:
        hms.add([{"role": "user", "content": text}], user_id=user_id, infer=False)
    return hms.search("tea", filters={"user_id": user_id}, threshold=0.0)["results"]


def test_search_rerank_wiring_and_fallback_discipline(hms, caplog):
    """rerank=True reorders through self.reranker (default overlap), appends rerank_score, survives reranker errors.
    rerank=True 经 self.reranker（默认 overlap）重排、追加 rerank_score，并在重排器出错时保持原序。

    Input / 输入：
        hms: The HMS fixture (no reranker configured).
        hms：HMS 夹具（未配置重排器）。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the default overlap path, a constructive order change, top_k, explain keys and the
        exception fallback.
        对默认 overlap 路径、构造性的顺序改变、top_k、explain 键与异常回退的断言。

    State effect / 状态影响：
        Writes to the in-memory store.
        写入内存存储。
    """
    assert hms.reranker is None
    plain = _seed_raw(hms, "tea-user")
    assert len(plain) == 3 and all("rerank_score" not in r for r in plain)
    # default path: an OverlapReranker is built lazily, announced once at info level, and its order is used
    # 默认路径：惰性构造 OverlapReranker，以 info 级别宣告一次，并采用其顺序
    with caplog.at_level(logging.INFO, logger="hms_elite.memory.main"):
        ranked = hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0, rerank=True, explain=True)["results"]
        hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0, rerank=True)
    assert sum("default offline overlap reranker" in r.getMessage() for r in caplog.records) == 1
    assert hms.reranker is None
    expected = OverlapReranker().rerank("tea", plain, top_k=20)
    assert [r["id"] for r in ranked] == [r["id"] for r in expected]
    assert [r["rerank_score"] for r in ranked] == [1.0, 1.0, 0.0]
    assert all("tea" in r["memory"] for r in ranked[:2]) and "coffee" in ranked[2]["memory"]
    for row in ranked:
        # §4.1 keys keep their order; rerank_score is the last row key; explain gets two appended keys
        # §4.1 键序不变；rerank_score 是行末键；explain 追加两个键
        assert list(row)[-1] == "rerank_score"
        assert list(row["score_details"])[-2:] == ["rerank_score", "reranker"]
        assert row["score_details"]["reranker"] == "overlap@v1"
        assert row["score_details"]["rerank_score"] == row["rerank_score"]
        assert row["score_details"]["final_score"] == row["score"]
    # rerank=False is untouched
    # rerank=False 不受影响
    assert hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0)["results"] == plain
    # constructive order change through the public attribute, with top_k applied after reranking
    # 通过公开属性构造性地改变顺序，top_k 在重排之后生效
    hms.reranker = _TailFirstReranker()
    flipped = hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0, rerank=True)["results"]
    assert [r["id"] for r in flipped] == [r["id"] for r in reversed(plain)]
    assert [r["rerank_score"] for r in flipped] == [1.0, 0.5, 0.0]
    top1 = hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0, rerank=True, top_k=1)["results"]
    assert len(top1) == 1 and top1[0]["id"] == plain[0]["id"]  # retrieval top_k=1 leaves one row to rerank
    # reranker failure: warning, retrieval order kept, no rerank_score
    # 重排器失败：告警，保持检索顺序，无 rerank_score
    hms.reranker = _BoomReranker()
    with caplog.at_level(logging.WARNING, logger="hms_elite.memory.main"):
        kept = hms.search("tea", filters={"user_id": "tea-user"}, threshold=0.0, rerank=True, explain=True)["results"]
    assert any("rerank failed (RuntimeError: boom)" in r.getMessage() for r in caplog.records)
    assert [r["id"] for r in kept] == [r["id"] for r in plain] and all("rerank_score" not in r for r in kept)
    assert all("reranker" not in r["score_details"] for r in kept)
    # empty result sets never call the reranker
    # 空结果集从不调用重排器
    assert hms.search("tea", filters={"user_id": "nobody"}, rerank=True)["results"] == []


def test_llm_reranker_from_config_uses_scripted_llm(store, caplog):
    """reranker={'provider': 'llm'} binds the facade LLM (scripted → the script's ranking) or its own envelope.
    reranker={'provider': 'llm'} 绑定门面 LLM（scripted → 按脚本排序）或其自带的信封。

    Input / 输入：
        store: The in-memory StateStore with the frozen clock.
        store：带冻结时钟的内存 StateStore。

        caplog: pytest log capture.
        caplog：pytest 日志捕获。

    Output / 输出：
        Assertions on the LLM route, the exhausted-queue fallback, the config-bound LLM, the rule fallback and the
        factory.
        对 LLM 路由、队列耗尽回退、配置绑定的 LLM、规则回退与工厂的断言。

    State effect / 状态影响：
        Writes to the in-memory store.
        写入内存存储。
    """
    script = {"ranking": [{"id": "2", "score": 0.9}, {"id": "0", "score": 0.4}]}
    hms = HMS(
        HMSConfig(
            llm={"provider": "scripted", "config": {"responses": [script]}},
            reranker={"provider": "llm"},
            vector_store={"provider": "memory"},
            store={"path": ":memory:"},
        ),
        store=store,
    )
    assert isinstance(hms.reranker, LLMReranker) and hms.reranker.llm is hms.llm
    assert hms.reranker.identity == "llm:scripted@v1"
    plain = _seed_raw(hms, "llm-user")
    assert hms.llm.calls == []
    ranked = hms.search("tea", filters={"user_id": "llm-user"}, threshold=0.0, rerank=True, explain=True)["results"]
    assert hms.reranker.last_route == "llm" and hms.llm.calls[-1]["task"] == "rerank"
    assert [r["id"] for r in ranked] == [plain[2]["id"], plain[0]["id"], plain[1]["id"]]
    assert [r["rerank_score"] for r in ranked] == [0.9, 0.4, 0.0]
    assert ranked[0]["score_details"]["reranker"] == "llm:scripted@v1"
    # the queue is exhausted (LLM_003): the facade keeps the retrieval order instead of raising
    # 队列耗尽（LLM_003）：门面保持检索顺序而不是抛出
    with caplog.at_level(logging.WARNING, logger="hms_elite.memory.main"):
        kept = hms.search("tea", filters={"user_id": "llm-user"}, threshold=0.0, rerank=True)["results"]
    assert [r["id"] for r in kept] == [r["id"] for r in plain] and "rerank_score" not in kept[0]
    assert any("rerank failed (LLMError" in r.getMessage() for r in caplog.records)
    hms.close()
    # a reranker with its own llm envelope does not share the facade LLM
    # 自带 llm 信封的重排器不共享门面 LLM
    own = HMS(
        HMSConfig(
            reranker={"provider": "llm", "config": {"llm": {"provider": "scripted", "config": {"responses": [script]}}}},
            vector_store={"provider": "memory"},
            store={"path": ":memory:"},
        )
    )
    assert own.llm.provider_name == "rule" and own.reranker.llm is not own.llm
    assert own.reranker.identity == "llm:scripted@v1"
    own.close()
    # the default rule LLM cannot serve the task (LLM_002): LLMReranker falls back to overlap and says so
    # 默认规则 LLM 无法执行该任务（LLM_002）：LLMReranker 回退到 overlap 并如实记录
    rule = HMS(HMSConfig(reranker={"provider": "llm"}, vector_store={"provider": "memory"}, store={"path": ":memory:"}))
    assert rule.reranker.identity == "llm:rule@v1"
    plain = _seed_raw(rule, "rule-user")
    with caplog.at_level(logging.WARNING, logger="hms_elite.reranker.llm"):
        ranked = rule.search("tea", filters={"user_id": "rule-user"}, threshold=0.0, rerank=True)["results"]
    assert rule.reranker.last_route == "overlap"
    assert [r["id"] for r in ranked] == [r["id"] for r in OverlapReranker().rerank("tea", plain)]
    rule.close()
    # the factory is the single source of truth for the 'reranker' kind
    # 工厂是 'reranker' 种类的唯一真相
    assert RerankerFactory.get_supported_providers() == ["llm", "overlap"]
    assert isinstance(RerankerFactory.create("overlap", {"top_k": 2}), OverlapReranker)
    assert isinstance(RerankerFactory.create("overlap", None, llm=object()), OverlapReranker)
    with pytest.raises(ConfigError) as bogus:
        RerankerFactory.create("bogus")
    assert _code(bogus) == "CONFIG_002" and bogus.value.details["kind"] == "reranker"
    with pytest.raises(ConfigError) as unknown_field:
        RerankerFactory.create("llm", {"max_char": 3})
    assert _code(unknown_field) == "CONFIG_001" and "max_chars" in str(unknown_field.value)


# ---------------------------------------------------------------------------
# 61
# ---------------------------------------------------------------------------
def test_migrate_from_mem0_walks_get_all(hms, clock):
    """from_mem0 walks get_all (plus show_expired=True) and re-adds rows with infer=False and timestamp=created_at.
    from_mem0 遍历 get_all（外加 show_expired=True），并以 infer=False、timestamp=created_at 重新添加各行。

    Input / 输入：
        hms: The target HMS fixture.
        hms：目标 HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions on the report, the recorded get_all calls and the re-added rows.
        对报告、记录的 get_all 调用与重新添加行的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存存储外无副作用。
    """
    fake = _FakeMem0(
        [
            {
                "id": "m1",
                "memory": LATTE,
                "created_at": "2026-04-02T08:00:00Z",
                "metadata": {"source": "mem0"},
                "user_id": U,
            },
            {
                "id": "m2",
                "memory": "Sarah joined the billing team.",
                "created_at": "2026-03-02T09:00:00Z",
                "metadata": None,
                "user_id": U,
                "agent_id": "coder",
            },
            {"id": "m3", "memory": "", "created_at": "2026-03-03T09:00:00Z", "user_id": U},
            {
                "id": "m4",
                "memory": "Old fact.",
                "created_at": "2026-01-01T00:00:00Z",
                "expiration_date": "2026-02-01",
                "user_id": U,
                "expired": True,
            },
        ]
    )
    clock.tick(seconds=60)
    report = from_mem0(fake, hms, filters={"user_id": U}, page_size=50)
    assert report["migrated"] == 3 and report["skipped"] == 1 and set(report["ids"]) == {"m1", "m2", "m4"}
    assert fake.calls == [
        {"filters": {"user_id": U}, "top_k": 50, "show_expired": False},
        {"filters": {"user_id": U}, "top_k": 50, "show_expired": True},
    ]
    row = hms.get(report["ids"]["m1"])
    assert row["memory"] == LATTE and row["metadata"] == {"source": "mem0"} and row["role"] == "user"
    assert row["hms"]["source_kind"] == "message" and row["hms"]["time_anchor"] == "2026-04-02T08:00:00.000000+00:00"
    assert row["user_id"] == U and "agent_id" not in row
    assert [h["event"] for h in hms.history(report["ids"]["m1"])] == ["ADD"]
    row2 = hms.get(report["ids"]["m2"])
    assert row2["agent_id"] == "coder" and row2["hms"]["entity_ids"] == ["sarah"]
    row4 = hms.get(report["ids"]["m4"])
    assert row4["valid_to"] == "2026-02-02T00:00:00.000000+00:00" and row4["expiration_date"] == "2026-02-01"
    live = {r["id"] for r in hms.get_all(filters={"user_id": U})["results"]}
    assert live == {report["ids"]["m1"], report["ids"]["m2"]}
    assert len(hms.get_all(filters={"user_id": U}, show_expired=True)["results"]) == 3


# ---------------------------------------------------------------------------
# 62
# ---------------------------------------------------------------------------
def test_cli_smoke(tmp_path, capsys):
    """hms add/search/compile/why/transitions/consolidate/stats via main([...]) on HMS_DIR=tmp_path.
    在 HMS_DIR=tmp_path 上通过 main([...]) 运行 hms add/search/compile/why/transitions/consolidate/stats。

    Input / 输入：
        tmp_path: pytest temporary directory (HMS_DIR is redirected there by the autouse fixture).
        tmp_path：pytest 临时目录（autouse 夹具已把 HMS_DIR 重定向到这里）。

        capsys: pytest stdout capture.
        capsys：pytest 标准输出捕获。

    Output / 输出：
        Assertions on the printed JSON / card text and the exit codes.
        对打印的 JSON / 卡片文本与退出码的断言。

    State effect / 状态影响：
        Creates tmp_path/hms.db.
        创建 tmp_path/hms.db。
    """
    assert cli_main(["add", LATTE, "--user", U, "--run", "run-coffee"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["results"][0]["event"] == "ADD" and added["results"][0]["memory"] == LATTE_FACT
    assert (tmp_path / "hms.db").exists()
    assert cli_main(["search", "latte", "--user", U, "--top-k", "3"]) == 0
    found = json.loads(capsys.readouterr().out)
    assert found["results"][0]["id"] == added["results"][0]["id"] and "score" in found["results"][0]
    assert cli_main(["compile", "coffee", "--user", U, "--run", "run-coffee", "--mode", "fast"]) == 0
    card = capsys.readouterr().out
    assert card.startswith("# Work State Card\n") and "Mode: fast" in card and LATTE_FACT in card
    sarah = "Sarah joined the billing team in March."
    assert cli_main(["add", sarah, "--user", U, "--run", "run-coffee"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["memory"] == sarah
    assert cli_main(["why", "Sarah", "--user", U, "--run", "run-coffee"]) == 0
    why = json.loads(capsys.readouterr().out)
    assert why["target"]["resolved_from"] == "query" and why["target"]["text"] == sarah
    assert cli_main(["transitions", "--user", U, "--fmt", "jsonl"]) == 0
    assert capsys.readouterr().out.strip() == ""
    assert cli_main(["consolidate", "--user", U]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["phases"]) == {"orient", "collect", "consolidate", "prune"}
    assert report["phases"]["consolidate"]["entity_atoms"] == 1
    assert cli_main(["stats", "--user", U]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["nodes"]["observation"] == 2 and stats["nodes"]["entity"] == 1 and stats["transitions"] == 0
    with pytest.raises(SystemExit):
        # --user 是必填参数
        cli_main(["search", "latte"])  # --user is required


def _readme_cli_commands(path: pathlib.Path) -> list[list[str]]:
    """argv of every `hms ...` line in a README's §12 ```bash walkthrough (comments and the `export` line dropped).
    README §12 ```bash 演示中每条 `hms ...` 命令的 argv（去掉注释与 `export` 行）。
    """
    blocks = re.findall(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    # §1 also has a ```bash block (pip install); the walkthrough is the one that runs `hms why`.
    # §1 也有一个 ```bash 块（pip install）；演示块是含 `hms why` 的那一个。
    block = next((b for b in blocks if "\nhms why " in b), None)
    assert block is not None, f"{path.name}: no ```bash walkthrough block"
    words = [shlex.split(line, comments=True) for line in block.splitlines()]
    return [argv[1:] for argv in words if argv and argv[0] == "hms"]


# ---------------------------------------------------------------------------
# 62b. README §12 runs verbatim (both languages), including `why ... --run run-billing`
# 62b. README §12 可逐字运行（两种语言），包括 `why ... --run run-billing`
# ---------------------------------------------------------------------------
def test_cli_readme_walkthrough_runs_verbatim(tmp_path, capsys):
    """Every `hms ...` line of README §12 (identical in README.zh.md) exits 0 in order on a fresh HMS_DIR and `why` resolves its question.
    README §12 的每条 `hms ...` 命令（README.zh.md 与之一致）在全新 HMS_DIR 上按序退出码 0，且 `why` 能解析出它的问题。

    Input / 输入：
        tmp_path: pytest temporary directory (HMS_DIR is redirected there by the autouse fixture).
        tmp_path：pytest 临时目录（autouse 夹具已把 HMS_DIR 重定向到这里）。

        capsys: pytest stdout capture.
        capsys：pytest 标准输出捕获。

    Output / 输出：
        Assertions on every exit code and on the why() JSON.
        对每个退出码与 why() JSON 的断言。

    State effect / 状态影响：
        Creates tmp_path/hms.db.
        创建 tmp_path/hms.db。
    """
    readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    commands = _readme_cli_commands(readme)
    assert commands == _readme_cli_commands(readme.with_name("README.zh.md")) and len(commands) >= 7
    outputs: dict[str, str] = {}
    for argv in commands:
        assert cli_main(argv) == 0, argv
        outputs[argv[0]] = capsys.readouterr().out
    why = json.loads(outputs["why"])
    assert why["target"]["resolved_from"] == "query" and why["target"]["kind"] == "atom"
    assert (tmp_path / "hms.db").exists()


# ---------------------------------------------------------------------------
# 62c. An HMSError inside a command is structured JSON on stderr and exit code 1, not a traceback
# 62c. 命令内的 HMSError 以结构化 JSON 输出到标准错误并返回退出码 1，而不是回溯
# ---------------------------------------------------------------------------
def test_cli_reports_hms_errors_as_json(tmp_path, capsys):
    """`hms why <question>` on an empty run exits 1 with the NOT_FOUND_001 error dict on stderr and nothing on stdout.
    在空 run 上执行 `hms why <问题>` 以退出码 1 结束，标准错误是 NOT_FOUND_001 错误字典，标准输出为空。

    Input / 输入：
        tmp_path: pytest temporary directory (HMS_DIR is redirected there by the autouse fixture).
        tmp_path：pytest 临时目录（autouse 夹具已把 HMS_DIR 重定向到这里）。

        capsys: pytest stdout/stderr capture.
        capsys：pytest 标准输出/标准错误捕获。

    Output / 输出：
        Assertions on the exit code and the stderr JSON.
        对退出码与标准错误 JSON 的断言。

    State effect / 状态影响：
        Creates tmp_path/hms.db.
        创建 tmp_path/hms.db。
    """
    question = "why must retry.ts not be modified"
    assert cli_main(["why", question, "--user", U, "--run", "run-empty"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error_type"] == "ValidationError" and error["error_code"] == "NOT_FOUND_001"
    assert error["details"] == {"target_id": question} and question in error["message"]


# ---------------------------------------------------------------------------
# extra: the HMS-native flow end to end through the facade (S4 → S5 → S6 → epilogue → handoff/import)
# 补充：通过门面端到端跑通 HMS 原生流程（S4 → S5 → S6 → 收尾 → 交接/导入）
# ---------------------------------------------------------------------------
def test_native_flow_through_facade(hms, clock, store, events, scope4, golden, index_provider):
    """add_events → compile().to_text() → why → feedback → checkpoint/rollback → transitions → consolidate → graph_stats → handoff/import.
    原生流程：add_events → compile().to_text() → why → 反馈 → 检查点/回滚 → 转移 → 巩固 → 图统计 → 交接/导入。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        store: The shared StateStore fixture.
        store：共享的 StateStore 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

        golden: Golden helper.
        golden：黄金比对辅助。

        index_provider: 'sqlite' | 'memory' (the receiver uses the same provider).
        index_provider：'sqlite' | 'memory'（接收方使用同一提供者）。

    Output / 输出：
        Assertions on the §19.3–§19.7 expectations reached through the facade only.
        仅通过门面达成的 §19.3–§19.7 期望的断言。

    State effect / 状态影响：
        None outside the in-memory stores.
        除内存存储外无副作用。
    """
    ids = _s4(hms, clock, events, scope4)
    golden.text("work_state_card_failed.txt", ids["card1"])
    golden.text("work_state_card_fixed.txt", ids["card2"])
    ra, rb = ids["ra"], ids["rb"]
    assert len(ra["eps"]) == 1 and len(ra["edges"]) == 2 and len(ra["constraints"]) == 1 and len(rb["edges"]) == 4
    assert [e["dedup"] for e in ra["events"]] == [False, False] and rb["constraints"] == []
    hits = hms.search("retry.ts", filters=scope4, top_k=3, explain=True)["results"]
    assert len(hits) == 3 and all("retry.ts" in h["hms"]["entity_ids"] for h in hits)
    assert all(h["score_details"]["entity_boost"] == 1.0 for h in hits)
    aq = hms.axis_query({"filters": {**scope4, "entity": "retry.ts"}, "aggregation": "count"})
    assert aq.count == 3 and aq.hits == []
    clock.tick(seconds=60)
    trace = hms.why("why must retry.ts not be modified", filters=scope4)
    assert trace["target"]["text"] == S4_DC and trace["target"]["resolved_from"] == "query"
    assert trace["explanation"] == "\n".join(trace["chain"]) and trace["unresolved_conflicts"] == []
    golden.text("why_chain.txt", "\n".join(trace["chain"]))
    assert hms.why(ra["eps"][0])["target"]["resolved_from"] == "id"
    assert hms.why(ids["uc"], hops=0)["nodes"][0]["node_id"] == ids["uc"]
    clock.tick(seconds=60)
    cp = hms.checkpoint(filters=scope4, label="before-feedback")
    clock.tick(seconds=60)
    fb = hms.feedback(ids["mem_uc"], "state_outdated", correction=S6_CORRECTION)
    assert [a["action"] for a in fb["actions"]] == ["archive", "create", "revision_edge"]
    assert hms.get(ids["mem_uc"])["memory"] == S6_CORRECTION and hms.get(ids["mem_uc"])["version"] == 2
    clock.tick(seconds=60)
    mid = hms.compile(S4_QUERY, filters=scope4)
    assert [c.text for c in mid.constraints] == [S4_DC, S6_CORRECTION] and mid.as_of == clock.now()
    clock.tick(seconds=60)
    rolled = hms.rollback(cp)
    assert rolled["message"] == "Rolled back to before-feedback" and rolled["restored"] == [ids["uc"]]
    assert hms.get(ids["mem_uc"])["memory"] == S4_CONSTRAINT and hms.get(ids["mem_uc"])["version"] == 1
    assert [h["event"] for h in hms.history(ids["mem_uc"])] == ["ADD", "FEEDBACK", "UPDATE", "ROLLBACK", "RESTORE"]
    clock.tick(seconds=60)
    tuples = hms.transitions(filters=scope4)
    assert [t[1] for t in tuples] == [S4_EP1_A, "patch src/billing/idempotency.ts: add idempotency key", "patch src/billing/idempotency.ts: add idempotency key", "revise", "rollback"]  # fmt: skip
    records = hms.transitions(filters=scope4, fmt="records")
    assert [r["source_kind"] for r in records] == ["ep", "ep", "causal", "revision", "rollback"]
    golden.masked("transitions_s6.jsonl", hms.transitions(filters=scope4, fmt="jsonl"))
    assert store.stats()["transitions"] == 5 and hms.transitions(filters=scope4, kinds=["ep"], fmt="records")[0]["scale"] == "L1_micro"  # fmt: skip
    clock.tick(seconds=60)
    report = hms.consolidate(filters=scope4)
    assert report["phases"]["consolidate"] == {
        "entity_atoms": 6, "temporal_edges": 0, "causal_edges": 0, "conflicts": 0, "interface_eps": 1,
        "projected_edges": 3, "interface_eps_l3": 0, "skills": 1, "transitions": 1, "compiled_states": 1,
    }  # fmt: skip
    stats = hms.graph_stats(filters=scope4)
    assert stats["nodes"] == {"message": 0, "observation": 3, "event": 4, "entity": 0, "ep": 2, "interface_ep": 1, "skill": 1}  # fmt: skip
    assert stats["edges"] == {"temporal": 1, "causal": 1, "use": 5, "distill": 3, "revision": 2, "conflict": 0}
    assert (stats["valid_edges"], stats["invalid_edges"], stats["boundary_edges"], stats["transitions"]) == (
        11,
        1,
        3,
        6,
    )
    assert hms.graph_stats(filters={"user_id": U, "agent_id": "coder"})["nodes"]["entity"] == 6
    assert hms.graph_stats()["nodes"]["entity"] == 6
    deep = hms.compile(S4_QUERY, filters=scope4, mode="deep")
    assert [s.text for s in deep.skills] == [f"Skill: {S4_CONSTRAINT}"] and deep.retrieval["ablation"] == "axis_graph_support"  # fmt: skip
    assert store.last_compiled_state(store.get_atom(ids["uc"]).scope)["card"].startswith("# Work State Card")
    clock.tick(seconds=60)
    pkg = hms.handoff(filters=scope4, goal=S4_QUERY, budget_tokens=800)
    assert isinstance(pkg, StatePackage) and pkg.checkpoint_id and pkg.goal == S4_QUERY
    assert StatePackage.from_json(pkg.to_json()) == pkg and store.get_checkpoint(pkg.checkpoint_id).label == f"handoff:{S4_QUERY}"  # fmt: skip
    receiver = HMS(
        HMSConfig(vector_store={"provider": index_provider}, store={"path": ":memory:"}),
        store=StateStore(":memory:", now=clock.now, id_factory=SequentialIds()),
    )
    imported = receiver.import_package(pkg.to_json())
    assert imported["imported"]["atoms"] == len(pkg.atoms) and imported["imported"]["anchors"] == len(pkg.anchors)
    assert receiver.get(ids["mem_uc"])["memory"] == S4_CONSTRAINT
    again = receiver.import_package(pkg.model_dump(mode="json"))
    assert all(v == 0 for v in again["imported"].values()) and again["skipped"]["atoms"] == len(pkg.atoms)
    rescoped = receiver.import_package(pkg, run_id="run-import")
    assert rescoped["imported"]["atoms"] == len(pkg.atoms)
    assert receiver.get_all(filters={"user_id": U, "run_id": "run-import"})["results"]
    revoked = hms.revoke(ids["anc_e2"])
    assert revoked["archived"] == [ids["e2"]] and store.get_anchor(ids["anc_e2"]).revoked_at is not None
    assert hms.revoke(ids["anc_e2"])["noop"] is True
    receiver.close()


# ---------------------------------------------------------------------------
# 60d. close() reaches the index provider's close hook
# 60d. close() 会调到索引提供者的 close 钩子
# ---------------------------------------------------------------------------
def test_close_reaches_the_index_close_hook():
    """HMS.close() calls the index's close() when the provider has one, then closes the store; idempotent either way.
    索引提供者带 close() 时 HMS.close() 会调用它，再关闭存储；两种情况下都幂等。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the hook call count, on providers without a hook and on the store being closed even when the
        hook raises.
        对钩子调用次数、无钩子提供者、以及钩子抛错时存储仍被关闭的断言。

    State effect / 状态影响：
        Opens and closes in-memory stores only.
        只打开并关闭内存存储。
    """
    hooked = HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}))
    calls: list[str] = []
    # MemoryIndex has no close hook; this instance gets one so the facade path is observable offline (milvus is the
    # shipped provider with a real hook: tests/test_index_milvus.py covers it end to end)
    # MemoryIndex 没有 close 钩子；给这个实例装一个，使门面路径可离线观察（milvus 是自带真实钩子的提供者：
    # tests/test_index_milvus.py 端到端覆盖）
    hooked.index.close = lambda: calls.append("index")  # type: ignore[attr-defined]
    hooked.close()
    assert calls == ["index"] and hooked.store._conn is None
    hooked.close()
    assert calls == ["index", "index"]
    plain = HMS(HMSConfig(vector_store={"provider": "sqlite"}, store={"path": ":memory:"}))
    # sqlite inherits the no-op IndexBase.close() (it shares the store's connection): nothing to release.
    # sqlite 继承空操作的 IndexBase.close()（它共用 store 的连接）：没有什么可释放。
    assert plain.index.close() is None
    plain.close()
    assert plain.store._conn is None
    broken = HMS(HMSConfig(vector_store={"provider": "memory"}, store={"path": ":memory:"}))

    def boom() -> None:
        raise RuntimeError("boom")

    broken.index.close = boom  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="boom"):
        broken.close()
    assert broken.store._conn is None
